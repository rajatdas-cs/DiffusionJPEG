import os
import math
import argparse
import random
import numpy as np
import logging
import torch
from torch.utils.data import DataLoader
from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from data.dataset_jpeg import DatasetJPEG
from cave.trainer import CaVETrainer
from tqdm import tqdm
import wandb
wandb.login()


def main(json_path='options/cave.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)

    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    init_iter, init_path_G = 0, None
    opt['path']['pretrained_netG'] = init_path_G
    current_step = init_iter
    border = 0

    option.save(opt)
    opt = option.dict_to_nonedict(opt)

    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    wandb.init(project="diff-car", name=f'cave_small_dataset')

    dataset_type = opt['datasets']['train']['dataset_type']

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = DatasetJPEG(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            train_loader = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      shuffle=dataset_opt['dataloader_shuffle'],
                                      num_workers=dataset_opt['dataloader_num_workers'],
                                      drop_last=True,
                                      pin_memory=True)
        elif phase == 'test':
            test_set = DatasetJPEG(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    trainer = CaVETrainer(opt)

    if opt['merge_bn'] and current_step > opt['merge_bn_startpoint']:
        trainer.merge_bnorm_test()

    trainer.init_train()

    max_train_steps = 200000
    progress_bar = tqdm(range(0, max_train_steps), initial=0, desc="Steps")

    for epoch in range(1000000):  # keep running
        for i, train_data in enumerate(train_loader):
            current_step += 1
            if dataset_type == 'dnpatch' and current_step % 20000 == 0:  # for 'train400'
                train_loader.dataset.update_data()

            trainer.update_learning_rate(current_step)

            trainer.feed_data(train_data)

            G_loss, QF_loss = trainer.optimize_parameters(current_step)
            wandb.log({'epoch': epoch, 'G_loss': G_loss, 'QF_loss': QF_loss})

            progress_bar.update(1)
            info = {"G_loss": G_loss.item(), "QF_loss": QF_loss.item()}
            progress_bar.set_postfix(**info)

            if opt['merge_bn'] and opt['merge_bn_startpoint'] == current_step:
                trainer.merge_bnorm_train()
                trainer.print_network()

            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = trainer.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, trainer.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            if current_step % opt['train']['checkpoint_save'] == 0:
                logger.info('Saving the trainer.')
                trainer.save(current_step)

            if current_step >= max_train_steps:
                break

            if current_step >= max_train_steps:
                break

            if current_step % opt['train']['checkpoint_test'] == 0:

                avg_psnr = 0.0
                avg_ssim = 0.0
                avg_psnrb = 0.0
                idx = 0

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['H_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    trainer.feed_data(test_data)
                    trainer.test()

                    visuals = trainer.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])
                    QF = 1 - visuals['QF']

                    save_img_path = os.path.join(img_dir, '{:s}.png'.format(img_name))
                    util.imsave(E_img, save_img_path)

                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)

                    avg_psnr += current_psnr

                    current_ssim = util.calculate_ssim(E_img, H_img, border=border)

                    avg_ssim += current_ssim

                    current_psnrb = util.calculate_psnrb(H_img, E_img, border=border)
                    avg_psnrb += current_psnrb

                    logger.info(
                        '{:->4d}--> {:>10s} | PSNR : {:<4.2f}dB | SSIM : {:<4.3f}dB | PSNRB : {:<4.2f}dB'.format(
                            idx, image_name_ext, current_psnr, current_ssim, current_psnrb))
                    logger.info('predicted quality factor: {:<4.2f}'.format(float(QF)))

                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx
                avg_psnrb = avg_psnrb / idx

                # testing log
                logger.info(
                    '<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB, Average SSIM : {:<.3f}dB, Average PSNRB : {:<.2f}dB\n'.format(
                        epoch, current_step, avg_psnr, avg_ssim, avg_psnrb))
                wandb.log({'epoch': epoch, 'PSNR': avg_psnr, 'SSIM': avg_ssim, 'PSNRB': avg_psnrb})
    logger.info('Saving the final trainer.')
    trainer.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()
