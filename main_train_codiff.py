import os
import os.path
import argparse
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader
import transformers
from accelerate import Accelerator
from torchvision import transforms
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
from collections import OrderedDict
import cv2
import pyiqa
import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from diffusion.codiff import CODiff_train
from diffusion.models.discriminator import Discriminator

from pathlib import Path
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs

from data.dataset_jpeg import DatasetJPEG
from utils import utils_image as util
from utils import utils_option as option

from diffusion.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

import warnings

warnings.filterwarnings("ignore")

import wandb
from datetime import datetime


tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])

def parse_float_list(arg):
    try:
        return [float(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List elements should be floats")


def parse_int_list(arg):
    try:
        return [int(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List elements should be integers")


def parse_str_list(arg):
    return arg.split(',')


def parse_args(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()

    # training details
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512, )
    parser.add_argument("--max_train_steps", type=int, default=100000, ) # 100000
    parser.add_argument("--checkpointing_steps", type=int, default=5000, )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--gradient_checkpointing", action="store_true", )
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help=(
                            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                            ' "constant", "constant_with_warmup"]'
                        ),
                        )
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
                        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
                        )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0, )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
                        help=(
                            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
                            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
                        ),
                        )
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help=(
                            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
                        ),
                        )
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], )
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true",
                        help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true", )
    parser.add_argument("--logging_dir", type=str, default="logs")

    parser.add_argument("--tracker_project_name", type=str, default="train_codiff",
                        help="The name of the wandb project to log to.")
    parser.add_argument("--pretrained_model", default=None, type=str)
    parser.add_argument('--gan_dis_weight', default=1e-2, type=float)
    parser.add_argument('--gan_gen_weight', default=5e-3, type=float)

    # lora setting
    parser.add_argument("--lora_rank", default=4, type=int)

    # dataset setting
    parser.add_argument("--datasets", default='options/codiff.json')

    # other setting
    parser.add_argument('--cave_path', type=str, required=True)
    parser.add_argument('--val_path', required=True)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument('--debug', action='store_true')


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    args.tracker_project_name = os.path.join("training_results", args.tracker_project_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    logging_dir = Path(args.tracker_project_name, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.tracker_project_name, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.tracker_project_name, "checkpoints"), exist_ok=True)
        if not args.debug:
            wandb.init(project="diff-car", name=args.tracker_project_name)

    model_gen = CODiff_train(args)
    model_gen.set_train()
    model_reg = Discriminator(args=args, accelerator=accelerator)
    model_reg.set_train()

    loss_fn = pyiqa.create_metric('dists', device=accelerator.device, as_loss=True)

    # set vae adapter
    model_gen.vae.set_adapter(['default_encoder'])
    # set gen adapter
    model_gen.unet.set_adapter(['default_encoder', 'default_decoder', 'default_others'])

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model_gen.unet.enable_xformers_memory_efficient_attention()
            model_reg.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        model_gen.unet.enable_gradient_checkpointing()
        model_reg.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # make the optimizer
    layers_to_opt = []
    for n, _p in model_gen.unet.named_parameters():
        if "lora" in n:
            layers_to_opt.append(_p)
    layers_to_opt += list(model_gen.unet.conv_in.parameters())
    if hasattr(model_gen, 'proj'):
        layers_to_opt += list(model_gen.proj.parameters())
    for n, _p in model_gen.vae.named_parameters():
        if "lora" in n:
            layers_to_opt.append(_p)

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                  eps=args.adam_epsilon, )
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                 num_training_steps=args.max_train_steps,
                                 num_cycles=args.lr_num_cycles, power=args.lr_power, )

    layers_to_opt_reg = []
    for n, _p in model_reg.unet.named_parameters():
        if "lora" in n:
            layers_to_opt_reg.append(_p)
    for _p in model_reg.cls_pred_branch.parameters():
        layers_to_opt_reg.append(_p)
    layers_to_opt_reg.append(model_reg.embeddings)

    optimizer_reg = torch.optim.AdamW(layers_to_opt_reg, lr=args.learning_rate,
                                      betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                      eps=args.adam_epsilon, )
    lr_scheduler_reg = get_scheduler(args.lr_scheduler, optimizer=optimizer_reg,
                                     num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                     num_training_steps=args.max_train_steps,
                                     num_cycles=args.lr_num_cycles, power=args.lr_power)

    args.datasets = option.parse_dataset(args.datasets)['datasets']
    for phase, dataset_opt in args.datasets.items():
        if phase == 'train':
            train_set = DatasetJPEG(dataset_opt)
            train_set.normalize = True
            dl_train = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      shuffle=dataset_opt['dataloader_shuffle'],
                                      num_workers=dataset_opt['dataloader_num_workers'],
                                      drop_last=True,
                                      pin_memory=True)

    from cave.cave import CaVE
    cave = CaVE(in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='BR')
    cave.load_state_dict(torch.load(args.cave_path), strict=True)
    cave.eval()
    for k, v in cave.named_parameters():
        v.requires_grad = False
    cave = cave.to("cuda")

    # Prepare everything with our `accelerator`.
    model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg = accelerator.prepare(
        model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg
    )

    if accelerator.is_main_process:
        del args.datasets
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
                        disable=not accelerator.is_local_main_process, total=args.max_train_steps)

    # start the training loop
    global_step = 0
    while True:
        for step, batch in enumerate(dl_train):
            global_step += 1
            if global_step > args.max_train_steps:
                exit()
            m_acc = [model_gen, model_reg]
            with accelerator.accumulate(*m_acc):
                x_src = batch["L"].to("cuda")
                x_tgt = batch["H"].to("cuda")

                x_src_cave = batch["img_L_array"].permute(0, 3, 1, 2).float().div(255.)
                with torch.no_grad():
                    visual_embedding = cave.get_visual_embedding(x_src_cave)
                # forward pass
                x_pred, latents_pred = model_gen(x_src, visual_embedding)

                # Reconstruction loss
                x_pred = x_pred * 0.5 + 0.5
                x_tgt = x_tgt * 0.5 + 0.5

                l2_loss = F.mse_loss(x_pred.float(), x_tgt.float(), reduction="mean")
                dists_loss = loss_fn(x_pred.float(), x_tgt.float())
                if torch.cuda.device_count() > 1:
                    generator_loss = model_reg.module.compute_generator_loss(latents_pred)
                else:
                    generator_loss = model_reg.compute_generator_loss(latents_pred)

                loss = l2_loss + dists_loss + generator_loss * args.gan_gen_weight

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                # discriminator loss
                x_tgt = 2 * x_tgt - 1
                if torch.cuda.device_count() > 1:
                    gt_latents = model_reg.module.compute_gt_latents(x_tgt)
                    loss_d = model_reg.module.compute_discriminator_loss(gt_latents, latents_pred) * args.gan_dis_weight
                else:
                    gt_latents = model_reg.compute_gt_latents(x_tgt)
                    loss_d = model_reg.compute_discriminator_loss(gt_latents, latents_pred,) * args.gan_dis_weight
                accelerator.backward(loss_d)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model_reg.parameters(), args.max_grad_norm)
                optimizer_reg.step()
                lr_scheduler_reg.step()
                optimizer_reg.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)

                if accelerator.is_main_process:

                    logs = {}
                    # log all the losses
                    logs["loss_d"] = loss_d.detach().item()
                    logs["loss_g"] = generator_loss.detach().item()
                    logs["loss_l2"] = l2_loss.detach().item()
                    logs["loss_dists"] = dists_loss.detach().item()
                    progress_bar.set_postfix(**logs)
                    if not args.debug:
                        wandb.log({'loss_l2': logs["loss_l2"], 'loss_dists': logs['loss_dists'], 'loss_g': logs['loss_g'],
                                   'loss_d': logs['loss_d']})

                    accelerator.log(logs, step=global_step)

            if global_step % args.checkpointing_steps == 0:
                # checkpoint the model
                outf = os.path.join(args.tracker_project_name, "checkpoints", f"model_{global_step}.pkl")
                accelerator.unwrap_model(model_gen).save_model(outf)

                lpips_metric = pyiqa.create_metric('lpips', device="cuda")
                dists_metric = pyiqa.create_metric('dists', device="cuda")
                musiq_metric = pyiqa.create_metric('musiq', device="cuda")
                maniqa_metric = pyiqa.create_metric('maniqa', device="cuda")
                clipiqa_metric = pyiqa.create_metric('clipiqa', device="cuda")

                if torch.cuda.device_count() > 1:
                    model_gen.module.set_eval()
                else:
                    model_gen.set_eval()

                H_paths = util.get_image_paths(args.val_path)

                quality_factor = 10
                test_results = OrderedDict()
                test_results['lpips'] = []
                test_results['dists'] = []
                test_results['musiq'] = []
                test_results['maniqa'] = []
                test_results['clipiqa'] = []

                for idx, img in tqdm(enumerate(H_paths)):
                    img_name, ext = os.path.splitext(os.path.basename(img))

                    img_H = Image.open(img).convert('RGB')

                    # vae can only process images with height and width multiples of 8
                    new_width = img_H.width - img_H.width % 8
                    new_height = img_H.height - img_H.height % 8
                    img_H = img_H.resize((new_width, new_height), Image.LANCZOS)

                    img_L = img_H.copy()

                    img_L = np.array(img_L)
                    n_channels = img_L.shape[-1]
                    if n_channels == 3:
                        img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR)
                    _, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
                    img_L = cv2.imdecode(encimg, 0) if n_channels == 1 else cv2.imdecode(encimg, 3)
                    if n_channels == 3:
                        img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)

                    img_L_cave = util.uint2tensor4(img_L).to("cuda")
                    with torch.no_grad():
                        visual_embedding = cave.get_visual_embedding(img_L_cave)
                    img_L = Image.fromarray(img_L)
                    lq = tensor_transforms(img_L).unsqueeze(0).to("cuda")
                    # translate the image
                    with torch.no_grad():
                        lq = lq * 2 - 1
                        if torch.cuda.device_count() > 1:
                            img_E = model_gen.module.eval(lq, visual_embedding)
                        else:
                            img_E = model_gen.eval(lq, visual_embedding)
                        img_E = transforms.ToPILImage()(img_E[0].cpu() * 0.5 + 0.5)
                        if args.align_method == 'adain':
                            img_E = adain_color_fix(target=img_E, source=img_L)
                        elif args.align_method == 'wavelet':
                            img_E = wavelet_color_fix(target=img_E, source=img_L)
                        else:
                            pass
                    img_H = np.array(img_H)
                    img_E = np.array(img_E)

                    util.imsave(img_E, os.path.join(args.tracker_project_name, img_name + '.png'))

                    img_E, img_H = img_E / 255., img_H / 255.
                    img_E, img_H = torch.tensor(img_E, device="cuda").permute(2, 0, 1).unsqueeze(0), torch.tensor(img_H,
                                                                                                                  device="cuda").permute(
                        2, 0, 1).unsqueeze(0)
                    img_E, img_H = img_E.type(torch.float32), img_H.type(torch.float32)

                    lpips_score = lpips_metric(img_E, img_H)
                    dists = dists_metric(img_E, img_H)
                    musiq = musiq_metric(img_E, img_H)
                    maniqa = maniqa_metric(img_E, img_H)
                    clipiqa = clipiqa_metric(img_E, img_H)

                    test_results['lpips'].append(lpips_score.item())
                    test_results['dists'].append(dists.item())
                    test_results['musiq'].append(musiq.item())
                    test_results['maniqa'].append(maniqa.item())
                    test_results['clipiqa'].append(clipiqa.item())

                avg_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
                avg_dists = sum(test_results['dists']) / len(test_results['dists'])
                avg_musiq = sum(test_results['musiq']) / len(test_results['musiq'])
                avg_maniqa = sum(test_results['maniqa']) / len(test_results['maniqa'])
                avg_clipiqa = sum(test_results['clipiqa']) / len(test_results['clipiqa'])
                if accelerator.is_main_process and not args.debug:
                    wandb.log(
                        {'LPIPS': avg_lpips, 'DISTS': avg_dists, 'MANIQA': avg_maniqa,
                         'MUSIQ': avg_musiq, 'CLIPIQA': avg_clipiqa})

                if torch.cuda.device_count() > 1:
                    model_gen.module.set_train()
                else:
                    model_gen.set_train()

if __name__ == "__main__":
    args = parse_args()
    main(args)
