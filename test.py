"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import glob
import shutil
import torch
from models.structured_trans_model import StructuredTransModel
import monai
import numpy as np
from util.util import tensor2im 
from PIL import Image
import re

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    opt.half_data = False

    # Create dataset and model
    dataset = create_dataset(opt)  # create a dataset
    model = create_model(opt)      # create a model

    model.setup(opt)
    model.eval()# regular setup: load and print networks


    # Create a directory for results
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    print(web_dir)

    # New folder to store all images
    low_high_dir = os.path.join(web_dir, 'low_high')
    os.makedirs(low_high_dir, exist_ok=True)

    if isinstance(model, StructuredTransModel):
        model.gt_shape_assist = False
        print(model)

    # Inspect dataset structure
    print("Inspecting dataset structure...")
    for i, data in enumerate(dataset):
        print(f"Dataset item {i}: {data}")
        if i >= 5:  # Print the first 5 items and stop
            break

    # Ensure numerical order of dataset file names
    dataset_files = list(dataset)  # Assuming `dataset` is iterable
    try:
        dataset_files.sort(key=lambda x: int(re.search(r'\d+', str(x.get("A_paths", ""))).group()))
    except Exception as e:
        print(f"Error during sorting: {e}")
        print("Please inspect the dataset structure and ensure 'A_paths' is a string containing the filename.")
        exit(1)

    # Loop through dataset in numerical order and save images
    for i, data in enumerate(dataset_files):
        if opt.num_test > 0 and i >= opt.num_test:  # only process `opt.num_test` images if specified
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # Print progress every 5 images
            print('processing (%04d)-th image... %s' % (i, img_path))
        if not opt.no_save:
            save_images(webpage, visuals, img_path, width=opt.display_winsize)

    webpage.save()  # save the HTML

    if not opt.no_save:
        # Move saved images to the new 'low_high' directory
        for filepath in glob.glob(f'{web_dir}/images/*real_A*'):
            filename = os.path.basename(filepath)
            shutil.move(filepath, f'{low_high_dir}/{filename}')
        
        for filepath in glob.glob(f'{web_dir}/images/*real_B*'):
            filename = os.path.basename(filepath)
            shutil.move(filepath, f'{low_high_dir}/{filename}')
        
        for filepath in glob.glob(f'{web_dir}/images/*fake_B*'):
            filename = os.path.basename(filepath)
            shutil.move(filepath, f'{low_high_dir}/{filename}')
        
        for filepath in glob.glob(f'{web_dir}/images/*fake_A*'):
            filename = os.path.basename(filepath)
            shutil.move(filepath, f'{low_high_dir}/{filename}')

    print("Images have been saved successfully.")
# if __name__ == '__main__':
#     opt = TestOptions().parse()  # get test options
#     opt.num_threads = 0   # test code only supports num_threads = 1
#     opt.batch_size = 1    # test code only supports batch_size = 1
#     opt.serial_batches = True  # disable data shuffling
#     opt.no_flip = True    # no flip
#     opt.display_id = -1   # no visdom display
#     opt.half_data = False

#     # Create dataset and model
#     dataset = create_dataset(opt)  # create a dataset
#     model = create_model(opt)      # create a model
#     model.setup(opt)               # regular setup: load and print networks

#     # Create a directory for results
#     web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
#     webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
#     print(web_dir)

#     out_ct = f'{web_dir}/real_B'
#     out_mri = f'{web_dir}/real_A'
#     out_fake_ct = f'{web_dir}/fake_B'
#     out_fake_mri = f'{web_dir}/fake_A'

#     os.makedirs(out_ct, exist_ok=True)
#     os.makedirs(out_mri, exist_ok=True)
#     os.makedirs(out_fake_ct, exist_ok=True)
#     os.makedirs(out_fake_mri, exist_ok=True)

#     if isinstance(model, StructuredTransModel):
#         model.gt_shape_assist = False
#         print(model)

#     # Loop through dataset and save images
#     for i, data in enumerate(dataset):
#         if opt.num_test > 0 and i >= opt.num_test:  # only process `opt.num_test` images if specified
#             break
#         model.set_input(data)  # unpack data from data loader
#         model.test()           # run inference
#         visuals = model.get_current_visuals()  # get image results
#         img_path = model.get_image_paths()     # get image paths
#         if i % 5 == 0:  # Print progress every 5 images
#             print('processing (%04d)-th image... %s' % (i, img_path))
#         if not opt.no_save:
#             save_images(webpage, visuals, img_path, width=opt.display_winsize)

#     webpage.save()  # save the HTML

#     if not opt.no_save:
#         # Move saved images to organized directories
#         for filepath in glob.glob(f'{web_dir}/images/*real_A*'):
#             filename = os.path.basename(filepath)
#             shutil.move(filepath, f'{out_mri}/{filename}')
        
#         for filepath in glob.glob(f'{web_dir}/images/*real_B*'):
#             filename = os.path.basename(filepath)
#             shutil.move(filepath, f'{out_ct}/{filename}')
        
#         for filepath in glob.glob(f'{web_dir}/images/*fake_B*'):
#             filename = os.path.basename(filepath)
#             shutil.move(filepath, f'{out_fake_ct}/{filename}')
        
#         for filepath in glob.glob(f'{web_dir}/images/*fake_A*'):
#             filename = os.path.basename(filepath)
#             shutil.move(filepath, f'{out_fake_mri}/{filename}')

    print("Images have been saved successfully.")
# def normalize_scan(img):
#     B, C, D, H, W = img.size()
#     img_min, _ = img.view(B, C, D, H*W).min(dim=-1)
#     img_max, _ = img.view(B, C, D, H*W).max(dim=-1)
#     normalized = (img - img_min[..., None, None]) / (img_max[..., None, None] - img_min[..., None, None])
#     normalized = torch.nan_to_num(normalized, nan=0.0)
#     normalized = normalized *255
#     return normalized.int()

# def eval(pred, true):
#     # pred = normalize_scan(pred)
#     # true =  normalize_scan(true)
#     pred = pred * 127.5 + 127.5
#     true = true * 127 + 127.5
#     ssim = monai.metrics.SSIMMetric(spatial_dims=2, data_range=255)
#     mae = monai.metrics.MAEMetric()
#     psnr = monai.metrics.PSNRMetric(255)
#     return mae(pred, true).item(), ssim(pred, true).item(), psnr(pred, true).item()

# if __name__ == '__main__':
#     opt = TestOptions().parse()  # get test options
#     # hard-code some parameters for test
#     opt.num_threads = 0   # test code only supports num_threads = 1
#     opt.batch_size = 1    # test code only supports batch_size = 1
#     opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
#     opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
#     opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
#     opt.half_data = False
#     dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
#     model = create_model(opt)      # create a model given opt.model and other options
#     model.setup(opt)               # regular setup: load and print networks; create schedulers
#     # create a website
#     web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
#     webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
#     print(web_dir)
#     # out_ct = f'{opt.results_dir}/{opt.name}/real_B'
#     # out_mri = f'{opt.results_dir}/{opt.name}/real_A'
#     # out_fake_ct = f'{opt.results_dir}/{opt.name}/fake_B'
#     # out_fake_mri = f'{opt.results_dir}/{opt.name}/fake_A'
#     # out_mae = f'{opt.results_dir}/{opt.name}/MAE'
#     out_ct = f'{web_dir}/real_B'
#     out_mri = f'{web_dir}/real_A'
#     out_fake_ct = f'{web_dir}/fake_B'
#     out_fake_mri = f'{web_dir}/fake_A'
#     out_mae = f'{web_dir}/MAE'
#     os.makedirs(out_ct, exist_ok=True)
#     os.makedirs(out_mri, exist_ok=True)
#     os.makedirs(out_fake_ct, exist_ok=True)
#     os.makedirs(out_fake_mri, exist_ok=True)
#     os.makedirs(out_mae, exist_ok=True)
#     ssims_b = []
#     maes_b = []
#     psnrs_b = []

#     ssims_a = []
#     maes_a = []
#     psnrs_a = []
#     # test with eval mode. This only affects layers like batchnorm and dropout.
#     # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
#     # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
#     #if opt.eval:
#     # model.eval()
#     if isinstance(model, StructuredTransModel):
#         model.gt_shape_assist = False
#         print(model)
#     for i, data in enumerate(dataset):
#         # break
#         if opt.num_test > 0 and i >= opt.num_test:  # only apply our model to opt.num_test images.
#             break
#         model.set_input(data)  # unpack data from data loader
#         model.test()           # run inference
#         visuals = model.get_current_visuals()  # get image results
#         mae_b, ssim_b, psnr_b = eval(visuals['fake_B'], visuals['real_B'])
#         maes_b.append(mae_b)
#         ssims_b.append(ssim_b)
#         psnrs_b.append(psnr_b)
#         # visuals['MAE_B'] = torch.abs(visuals['real_B'] - visuals['fake_B'])
        
#         if 'fake_A' in visuals:
#             mae_a, ssim_a, psnr_a = eval(visuals['fake_A'], visuals['real_A'])
#             maes_a.append(mae_a)
#             ssims_a.append(ssim_a)
#             psnrs_a.append(psnr_a)
#         ## Fix background CT
        
#             # visuals['MAE_A'] = torch.abs(visuals['real_A'] - visuals['fake_A'])
        
#         img_path = model.get_image_paths()     # get image paths
#         if i % 5 == 0:  # save images to an HTML file
#             print('processing (%04d)-th image... %s' % (i, img_path))
#         if not opt.no_save:
#             save_images(webpage, visuals, img_path, width=opt.display_winsize)
#     webpage.save()  # save the HTML
#     print("MAE_B ", np.mean(maes_b))
#     print("SSIM_B ", np.mean(ssims_b))
#     print("PSNR_B ", np.mean(psnrs_b))

#     # print("MAE_A ", np.mean(maes_a))
#     # print("SSIM_A ", np.mean(ssims_a))
#     # print("PSNR_A ", np.mean(psnrs_a))
#     if not opt.no_save:
#         ### Save results
#         for filepath in glob.glob(f'{web_dir}/images/*real_A*'):
#             filename = os.path.basename(filepath)
#             shutil.move(filepath, f'{out_mri}/{filename}')
        
#         for filepath in glob.glob(f'{web_dir}/images/*real_B*'):
#             filename = os.path.basename(filepath)
#             shutil.move(filepath, f'{out_ct}/{filename}')
#         for filepath in glob.glob(f'{web_dir}/images/*fake_B*'):
#             filename = os.path.basename(filepath)
#             shutil.move(filepath, f'{out_fake_ct}/{filename}')

#         for filepath in glob.glob(f'{web_dir}/images/*fake_A*'):
#             filename = os.path.basename(filepath)
#             shutil.move(filepath, f'{out_fake_mri}/{filename}')
        
#         for filepath in glob.glob(f'{web_dir}/images/*MAE*'):
#             filename = os.path.basename(filepath)
#             shutil.move(filepath, f'{out_mae}/{filename}')


