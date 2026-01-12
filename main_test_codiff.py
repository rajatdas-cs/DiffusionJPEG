import os
import sys
sys.path.append(os.getcwd())

import argparse
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
from collections import OrderedDict
import pyiqa
from tqdm import tqdm

import utils.utils_image as utils
from diffusion.codiff import CODiff_test
from diffusion.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    parser.add_argument('--pretrained_model', type=str, required=True, help='sd model path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument("--process_size", type=int, default=128)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--codiff_path", type=str, required=True)
    parser.add_argument('--cave_path', type=str, required=True)
    # precision setting
    parser.add_argument("--mixed_precision", type=str, choices=['fp16', 'fp32'], default="fp16")
    # merge lora
    parser.add_argument("--merge_and_unload_lora", default=False)  # merge lora weights before inference
    # tile setting
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)

    args = parser.parse_args()

    # initialize the model
    model = CODiff_test(args)

    from cave.cave import CaVE

    cave = CaVE(in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='BR')
    cave.load_state_dict(torch.load(args.cave_path), strict=True)
    cave.eval()
    for k, v in cave.named_parameters():
        v.requires_grad = False
    cave = cave.to("cuda")

    # weight type
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    # make the output dir
    os.makedirs(args.output_dir, exist_ok=True)

    H_paths = utils.get_image_paths(args.input_image)
    print(f'There are {len(H_paths)} images.')

    device = 'cuda'
    lpips_metric = pyiqa.create_metric('lpips', device=device)
    dists_metric = pyiqa.create_metric('dists', device=device)
    niqe_metric = pyiqa.create_metric('niqe', device=device)
    musiq_metric = pyiqa.create_metric('musiq', device=device)
    maniqa_metric = pyiqa.create_metric('maniqa', device=device)
    clipiqa_metric = pyiqa.create_metric('clipiqa', device=device)

    f = open(os.path.join(args.output_dir, 'results.csv'), 'a')
    for quality_factor in [5, 10, 20]:
        os.makedirs(os.path.join(args.output_dir, str(quality_factor)), exist_ok=True)
        test_results = OrderedDict()
        test_results['lpips'] = []
        test_results['dists'] = []
        test_results['musiq'] = []
        test_results['maniqa'] = []
        test_results['clipiqa'] = []

        cnt = 0
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
            lq_raw = torch.from_numpy(img_L).permute(2, 0, 1).float().div(255.).unsqueeze(0).to("cuda")     # 0, 1

            lq = lq_raw * 2 - 1     # -1, 1

            with torch.no_grad():
                visual_embedding = cave.get_visual_embedding(lq_raw)
                img_E = model(lq, visual_embedding)

            img_H = np.array(img_H)

            img_E = transforms.ToPILImage()(img_E[0].cpu() * 0.5 + 0.5)
            if args.align_method == 'adain':
                img_E = adain_color_fix(target=img_E, source=img_L)
            elif args.align_method == 'wavelet':
                img_E = wavelet_color_fix(target=img_E, source=img_L)
            else:
                pass

            img_E = np.array(img_E)
            utils.imsave(img_E, os.path.join(args.output_dir, str(quality_factor), img_name + '.png'))
            img_E, img_H_tensor = torch.tensor(img_E, device=device).permute(2, 0, 1).unsqueeze(0), torch.tensor(img_H,
                                                                                                          device=device).permute(
                2, 0, 1).unsqueeze(0)
            img_E, img_H_tensor = img_E / 255., img_H_tensor / 255.
            lpips = lpips_metric(img_E, img_H_tensor)
            dists = dists_metric(img_E, img_H_tensor)
            musiq = musiq_metric(img_E, img_H_tensor)
            maniqa = maniqa_metric(img_E, img_H_tensor)
            clipiqa = clipiqa_metric(img_E, img_H_tensor)

            test_results['lpips'].append(lpips.item())
            test_results['dists'].append(dists.item())
            test_results['musiq'].append(musiq.item())
            test_results['maniqa'].append(maniqa.item())
            test_results['clipiqa'].append(clipiqa.item())

        avg_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
        avg_dists = sum(test_results['dists']) / len(test_results['dists'])
        avg_musiq = sum(test_results['musiq']) / len(test_results['musiq'])
        avg_maniqa = sum(test_results['maniqa']) / len(test_results['maniqa'])
        avg_clipiqa = sum(test_results['clipiqa']) / len(test_results['clipiqa'])

        print(
            'Average LPIPIS,DISTS,MUSIQ,MANIQA,CLIPIQA - {} -: {:.4f},{:.4f},{:.4f},{:.4f},{:.4f}'.format(
                str(quality_factor), avg_lpips, avg_dists, avg_musiq, avg_maniqa, avg_clipiqa))

        print(quality_factor, avg_lpips, avg_dists, avg_musiq, avg_maniqa,
              avg_clipiqa, sep=',', end='\n', file=f)

    f.close()
