#!/usr/bin/env python
# coding: utf-8

import os
import glob
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
from model import CSRNet
import argparse
import torch
from skimage.measure import compare_psnr,compare_ssim, compare_mse
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from swd import swd
from torch.autograd import Variable
import torch
from torchvision import datasets, transforms



def run_test(model_path,test):
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
    ])
    root = os.path.dirname(os.path.abspath(__file__))


    # now generate the ShanghaiA's ground truth
    part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
    part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
    part_B_train = os.path.join(root, 'part_B_final/train_data', 'images')
    part_B_test = os.path.join(root, 'part_B_final/test_data', 'images')
    path_sets = [part_B_test] if test =='B' else [part_A_test]

    model_name = model_path
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

    model = CSRNet()
    model = model.cuda()
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['state_dict'])
    from matplotlib import cm

    mae = 0
    mse = 0
    ssim = 0
    psnr = 0
    swd_result = 0

    for i in range(len(img_paths)):
        plane_image = Image.open(img_paths[i]).convert('RGB')
        img = transform(plane_image).cuda()
        gt_file = h5py.File(img_paths[i].replace('.jpg', '.h5').replace('images', 'ground'), 'r')
        groundtruth = np.asarray(gt_file['density'])
        sum_convovled_kernel = np.ones((8, 8))
        target = sg.convolve2d(groundtruth, sum_convovled_kernel[::-1, ::-1], mode='valid')[::8, ::8]
        output_turch = model(img.unsqueeze(0))
        output = np.array(output_turch.data.cpu()[0, 0, :, :])
        cur_mae = abs(output.sum() - np.sum(groundtruth))
        cur_mse = np.square(output.sum() - np.sum(groundtruth))
        mae += cur_mae
        mse += cur_mse
        cur_psnr = compare_psnr(target, output, data_range=1.0)
        psnr += cur_psnr
        cur_ssim = compare_ssim(target, output)
        ssim += cur_ssim
        target_turch = torch.from_numpy(target)
        target_turch  = target_turch.type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).cuda()
        target_turch = Variable(target_turch)
        cur_swd = swd.swd(target_turch,output_turch).detach().float().unsqueeze(0).data[0].float().item()
        swd_result += cur_swd
    print('---------total results --------')

    print('the model: ',model_path)
    print('\nthe mae')
    print(mae / len(img_paths))
    print('the psnr')
    print(psnr / len(img_paths))
    print('the swd result')
    print(swd_result/len(img_paths))
    final_result= {}
    final_result['mae'] = mae / len(img_paths)
    final_result['mse'] = mse / len(img_paths)
    final_result['psnr'] = psnr / len(img_paths)
    final_result['ssim'] = ssim / len(img_paths)
    final_result['swd'] = swd_result / len(img_paths)
    final_result['model_name']= model_path.split('/')[-1].split('.')[0]
    return final_result

def choose_best(dir_path,part):
    fns = [dir_path] if os.path.isfile(dir_path) and dir_path.split('.')[-1] == 'tar' else sorted(
        [os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.split('.')[-1] == 'tar'])
    best_model_dict = {'mea_best':(np.inf,'no_model'),'swd_best':(np.inf,'no_model')}
    for model_path in fns:
        results = run_test(model_path, part)
        if results['mae'] < best_model_dict['mea_best'][0]:
            best_model_dict['mea_best'] = (results['mae'],results['model_name'])
        if results['swd'] < best_model_dict['swd_best'][0]:
            best_model_dict['swd_best'] = (results['swd'], results['model_name'])
        print('for now the mae winner ')
        print(best_model_dict['mea_best'][1])
        print('for now the swd winner ')
        print(best_model_dict['swd_best'][1])
    print('------------------------the winner is -----------------------')
    print('final winner mae winner ')
    print(best_model_dict['mea_best'][1])
    print('final winner swd winner ')
    print(best_model_dict['swd_best'][1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_path', default='/home/eldan/Python_Projects/CSRNet-pytorch-master/training_models', type=str)
    parser.add_argument('--part', default="A", type=str)
    args, _ = parser.parse_known_args()
    choose_best(args.models_path, args.part)