#!/usr/bin/env python
# coding: utf-8


import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
from skimage.measure import compare_psnr,compare_ssim, compare_mse
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from itertools import product
from matplotlib.figure import figaspect
from copy import deepcopy
from swd import swd
# In[10]:
from torch.autograd import Variable
import torch
from torchvision import datasets, transforms
import argparse


def create_heat_map_compare(image_list,gt_list,predict_list,swd_list,mae_list,psnr_list,gt_number_list,predict_number_list):

    h_max = np.max([im.shape[0] for im in gt_list])
    total_h = [h_max] * 3
    total_w = [cur_image.shape[1] * (h_max / cur_image.shape[0]) for cur_image in gt_list]

    row = 3
    col = len(image_list)
    fig, axs = plt.subplots(row, col,
                            gridspec_kw={'wspace': 0, 'hspace': 0, 'width_ratios': total_w, 'height_ratios': total_h},
                            constrained_layout=True)
    for i, ax in enumerate(fig.axes):
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    for i in range(len(gt_list)):
        ratio_h = (h_max / image_list[i].shape[0])
        label = 'GT: ' + str(gt_number_list[i]) + '\n' + 'PREDICTED: ' + str(
            predict_number_list[i]) + '\n' + 'MEA: ' + str(mae_list[i]) + '\n' + 'PSNR: ' + str(psnr_list[i])+'\nSWD: '+str(swd_list[i])
        if len(gt_list) == 1:
            axs[0].imshow(cv2.resize(image_list[i], (int(image_list[i].shape[1] * ratio_h), int(h_max)),
                                        interpolation=cv2.INTER_CUBIC))
            axs[1].imshow(cv2.resize(gt_list[i], (int(gt_list[i].shape[1] * ratio_h), int(h_max)),
                                        interpolation=cv2.INTER_CUBIC) / (ratio_h * ratio_h), cmap=CM.jet)
            axs[2].imshow(cv2.resize(predict_list[i], (int(predict_list[i].shape[1] * ratio_h), int(h_max)),
                                        interpolation=cv2.INTER_CUBIC) / (ratio_h * ratio_h), cmap=CM.jet)
            axs[2].annotate(label, (0.0, 0.0), xycoords='axes fraction', color='white')
        else:
            axs[0, i].imshow(cv2.resize(image_list[i], (int(image_list[i].shape[1] * ratio_h), int(h_max)),
                                        interpolation=cv2.INTER_CUBIC))
            axs[1, i].imshow(cv2.resize(gt_list[i], (int(gt_list[i].shape[1] * ratio_h), int(h_max)),
                                        interpolation=cv2.INTER_CUBIC) / (ratio_h * ratio_h), cmap=CM.jet)
            axs[2, i].imshow(cv2.resize(predict_list[i], (int(predict_list[i].shape[1] * ratio_h), int(h_max)),
                                        interpolation=cv2.INTER_CUBIC) / (ratio_h * ratio_h), cmap=CM.jet)
            axs[2, i].annotate(label, (0.0, 0.0), xycoords='axes fraction', color='white')
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.set_size_inches(16, 12)
    plt.tight_layout()
    return fig



def create_heat_map_compare_resize(original_gt_list, reg_resize_list, correct_resize_gt_list, mae_list_between_non_correct_resize,mae_list_between_correct_resize):

    h_max = np.max([im.shape[0] for im in original_gt_list])

    row = 3
    col = len(original_gt_list)
    fig, axs = plt.subplots(row, col,
                            gridspec_kw={'wspace': 0, 'hspace': 0},
                            constrained_layout=True)
    for i, ax in enumerate(fig.axes):
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    for i in range(len(original_gt_list)):
        ratio_h = (h_max / original_gt_list[i].shape[0])
        label = 'gt: ' + str(round(original_gt_list[i].sum(),2)) + '\n' + 'resize gt: ' + str(
            round(reg_resize_list[i].sum(),2)) + '\ncorrect resize gt:  '+str(round(correct_resize_gt_list[i].sum(),2))+'\n mea resize: '\
                + str(mae_list_between_non_correct_resize[i]) + '\n mea convolve resize:' + str(mae_list_between_correct_resize[i])
        axs[0, i].imshow(original_gt_list[i], cmap=CM.jet)
        axs[1, i].imshow(correct_resize_gt_list[i], cmap=CM.jet)
        axs[2, i].imshow(reg_resize_list[i], cmap=CM.jet)
        axs[2, i].annotate(label, (0.0, 0.0), xycoords='axes fraction', color='white')
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.set_size_inches(16, 12)
    plt.tight_layout()
    return fig

def create_graph_per_number_of_gt(result_dict, criterion):
    label = '% of mistake Vs number of people' if criterion =='MAE' else 'score Vs number of people'
    for k,v in result_dict.items():
        if criterion !='MSE' or criterion !='MAE':
            result_dict[k] = np.mean(v)
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.bar(range(len(result_dict)), list(result_dict.values()), align='center', width=0.4)
    plt.title(criterion)
    plt.ylabel(label)

    plt.xticks(range(len(result_dict)), list(list(result_dict.keys())))
    return fig



def run_test(model_path,test):
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
    ])
    test = test
    very_small = '<10' if test == 'B' else '<200'
    small = '10-50' if test == 'B' else '200-500'
    medium = '50-100' if test == 'B' else '500-1000'
    large = '100-200' if test == 'B' else '1000-1500'
    very_large = '200<' if test == 'B' else '1000<'
    very_small_criterion = 10 if test == 'B' else 200
    small_criterion = 50 if test == 'B' else 500
    medium_criterion = 100 if test == 'B' else 1000
    large_criterion = 200 if test == 'B' else 1500


    result_dict_mea = {very_small: [], small: [], medium: [], large: [], very_large: []}
    result_dict_msa = {very_small: [], small: [], medium: [], large: [], very_large: []}
    result_dict_ssim = {very_small: [], small: [], medium: [], large: [], very_large: []}
    result_dict_psnr = {very_small: [], small: [], medium: [], large: [], very_large: []}
    result_dict_gt = {very_small: [], small: [], medium: [], large: [], very_large: []}
    result_dict_swd = {very_small: [], small: [], medium: [], large: [], very_large: []}


    root = os.path.dirname(os.path.abspath(__file__))


    # now generate the ShanghaiA's ground truth
    part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
    part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
    part_B_train = os.path.join(root, 'part_B_final/train_data', 'images')
    part_B_test = os.path.join(root, 'part_B_final/test_data', 'images')
    path_sets = [part_B_test] if  test == 'B' else [part_A_test]

    model_name = model_path
    print(model_name)
    print('test set: ',path_sets)
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

    model = CSRNet()
    model = model.cuda()
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['state_dict'])
    print(checkpoint['epoch'])

    mae = 0
    mse = 0
    ssim = 0
    psnr = 0
    psnr_resize = 0
    ssim_resize = 0
    swd_result = 0
    swd_list = []
    image_list = []
    gt_list = []
    predict_list = []
    ssim_list = []
    mae_list = []
    psnr_list = []
    gt_number_list = []
    predict_number_list = []

    original_gt_list = []
    correct_resize_gt_list = []
    reg_resize_list = []
    mae_list_between_correct_resize = []
    mae_list_between_non_correct_resize = []
    original_image_list = []
    path = os.path.join(root,'results_evel')
    os.makedirs(path, exist_ok=True)
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
        cur_mse = np.square(output.sum() - np.sum(groundtruth)).sum()
        mae += cur_mae
        mse += cur_mse
        cur_psnr = compare_psnr(target, output, data_range=1.0)
        psnr += cur_psnr
        cur_ssim = compare_ssim(target, output)
        ssim += cur_ssim
        gt_sum = np.sum(groundtruth)
        target_turch = torch.from_numpy(target)
        target_turch  = target_turch.type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).cuda()
        target_turch = Variable(target_turch)
        cur_swd = swd.swd(target_turch,output_turch).detach().float().unsqueeze(0).data[0].float().item()
        swd_result += cur_swd
        if gt_sum <= very_small_criterion:
            criterion = very_small
        elif gt_sum > very_small_criterion and gt_sum <= small_criterion:
            criterion = small
        elif gt_sum > small_criterion and gt_sum <= medium_criterion:
            criterion = medium
        elif gt_sum > medium_criterion and gt_sum <= large_criterion:
            criterion = large
        else:
            criterion = very_large

        result_dict_swd[criterion].append(cur_swd)
        result_dict_gt[criterion].append(gt_sum)
        result_dict_mea[criterion].append(cur_mae)
        result_dict_msa[criterion].append(cur_mse)
        result_dict_psnr[criterion].append(cur_psnr)
        result_dict_ssim[criterion].append(cur_ssim)


        resize_output = cv2.resize(output, (int(groundtruth.shape[1]), int(groundtruth.shape[0])),
                                   interpolation=cv2.INTER_CUBIC) / 64
        resize_target = cv2.resize(target, (int(groundtruth.shape[1]), int(groundtruth.shape[0])),
                                   interpolation=cv2.INTER_CUBIC) / 64
        original_image_list.append(np.array(plane_image).astype(np.uint8))
        original_gt_list.append(groundtruth)
        correct_resize_gt_list.append(target)
        reg_resize_list.append(resize_target)

        cur_mae_correct = abs(target.sum() - np.sum(groundtruth))
        cur_mae_non_correct = abs(resize_target.sum() - np.sum(groundtruth))
        mae_list_between_correct_resize.append(round(cur_mae_correct,2))
        mae_list_between_non_correct_resize.append(round(cur_mae_non_correct,2))

        run_it = False
        if run_it:
            image_list.append(np.array(plane_image).astype(np.uint8))
            gt_list.append(resize_target)
            predict_list.append(resize_output)
            ssim_list.append(round(cur_ssim, 2))
            mae_list.append(round(cur_mae, 2))
            swd_list.append(round(cur_swd, 2))
            psnr_list.append(round(cur_psnr, 2))
            gt_number_list.append(round(np.sum(groundtruth), 2))
            predict_number_list.append(round(output.sum(), 2))
        if run_it and len(image_list) > 4:
            fig = create_heat_map_compare(image_list,gt_list,predict_list,swd_list,mae_list,psnr_list,gt_number_list,predict_number_list)
            name = model_name.split('/')[-1].split('.')[0]
            cur_path = os.path.join(path,name)
            os.makedirs(cur_path,exist_ok=True)
            fig.savefig(os.path.join(cur_path,name+'_compare_images_to_gt_'+str(i)+'.png'))
            image_list = []
            gt_list = []
            predict_list = []
            swd_list=[]
            mae_list=[]
            psnr_list=[]
            gt_number_list=[]
            predict_number_list=[]

        print('---------')
        print('gt: ', np.sum(groundtruth))
        print('pred: ', output.sum())
        print('the mae')
        print(i, cur_mae)
        print('cur avg mae')
        print(i, mae/(i+1))
        print('the mse')
        print(i, cur_mse)
        print('the ssim')
        print(i, cur_ssim)
        print('the psnr')
        print(i, cur_psnr)
        print('swd')
        print(cur_swd)
        print('---------')
    name = model_name.split('/')[-1].split('.')[0]
    cur_path = os.path.join(path, name)
    cur_result_dict_mea = {}
    cur_result_dict_msa = {}
    for key in result_dict_mea.keys():
        cur_result_dict_mea[key] = np.sum(result_dict_mea[key]) / np.sum(result_dict_gt[key]) if np.sum(
            result_dict_gt[key]) > 0 else 0
        cur_result_dict_msa[key] = np.sqrt(np.sum(result_dict_msa[key]) / len(result_dict_gt[key])) if np.sum(
            result_dict_gt[key]) > 0 else 0
    fig_mea = create_graph_per_number_of_gt(deepcopy(cur_result_dict_mea), 'MAE')
    fig_msa = create_graph_per_number_of_gt(deepcopy(cur_result_dict_msa), 'MSE')
    fig_ssim = create_graph_per_number_of_gt(deepcopy(result_dict_ssim), 'SSIM')
    fig_psnr = create_graph_per_number_of_gt(deepcopy(result_dict_psnr), 'PSNR')
    fig_swd = create_graph_per_number_of_gt(deepcopy(result_dict_swd), 'SWD')
    os.makedirs(cur_path, exist_ok=True)
    fig_mea.savefig(os.path.join(cur_path, name + '_mea_bar_' + str(i) + '.png'))
    os.makedirs(cur_path, exist_ok=True)
    fig_msa.savefig(os.path.join(cur_path, name + '_MSA_bar_' + str(i) + '.png'))
    os.makedirs(cur_path, exist_ok=True)
    fig_psnr.savefig(os.path.join(cur_path, name + '_PSNR_bar_' + str(i) + '.png'))
    os.makedirs(cur_path, exist_ok=True)
    fig_swd.savefig(os.path.join(cur_path, name + '_swd_bar_' + str(i) + '.png'))
    os.makedirs(cur_path, exist_ok=True)
    fig_ssim.savefig(os.path.join(cur_path, name + '_ssim_bar_' + str(i) + '.png'))

    print('---------total results --------')
    print('the mae')
    print(mae / len(img_paths))
    print('the mse')
    print(np.sqrt(mse / len(img_paths)))
    print('the ssim')
    print(ssim / len(img_paths))
    print('the ssim_resize')
    print(ssim_resize / len(img_paths))
    print('the psnr')
    print(psnr / len(img_paths))
    print('the psnr resize')
    print(psnr_resize / len(img_paths))
    print('the swd result')
    print(swd_result/len(img_paths))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/home/eldan/Python_Projects/CSRNet-pytorch-master/training_models/partBmodel_best.pth.tar', type=str)
    parser.add_argument('--part', default="A", type=str)
    args, _ = parser.parse_known_args()
    run_test(args.model_path, args.part)