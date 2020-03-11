#!/usr/bin/env python
# coding: utf-8

# In[42]:


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


def create_graph_per_number_of_gt_vs_two_models(result_dict_1,result_dict_2, criterion):
    label = '% of mistake Vs number of people' if criterion =='MAE' else 'score Vs number of people'
    for k,v in result_dict_1.items():
        if criterion !='MAE':
            result_dict_1[k] = np.mean(v)
            result_dict_2[k] = np.mean(result_dict_2[k])
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.title(criterion)
    plt.ylabel(label)
    # plt.xticks(range(len(result_dict)), x_bar_name)


    ax = plt.subplot(111)
    ax.bar(np.array(range(len(result_dict_1))) - 0.2, list(result_dict_1.values()), width=0.2, color='b', align='center')
    ax.bar(np.array(range(len(result_dict_1))), list(result_dict_2.values()), width=0.2, color='g', align='center')
    plt.xticks(np.array(range(len(result_dict_1))), list(list(result_dict_1.keys())))
    ax.xaxis_date()


    return fig

def create_heat_map_compare_two_models(image_list, gt_list, predict_list_1, predict_list_2, mae_list,mse_list, psnr_list, gt_number_list, predict_number_list,swd_list):

    h_max = np.max([im.shape[0] for im in gt_list])
    w_max = np.max([im.shape[1] for im in gt_list])
    total_h = [h_max] * 4
    total_w = [cur_image.shape[1] * (h_max / cur_image.shape[0]) for cur_image in gt_list]

    row = 4
    col = len(image_list)
    fig, axs = plt.subplots(row, col,
                            gridspec_kw={'wspace': 0, 'hspace': 0, 'width_ratios': total_w, 'height_ratios': total_h},
                            constrained_layout=True)
    # w, h = figaspect(row * w_max / (col * h_max))
    # fig.set_size_inches(w, h)
    for i, ax in enumerate(fig.axes):
        # ax.grid('on',linestyle = '--')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.set_aspect('auto')
    for i in range(len(gt_list)):
        ratio_h = (h_max / image_list[i].shape[0])
        label_0 = 'gt: ' + str(gt_number_list[i])
        label_1 =  'predicted: ' + str( predict_number_list[i][0]) + '\n' + 'MAE: ' + str(mae_list[i][0]) + '\n' \
                +'MSE: ' + str(mse_list[i][0]) + '\n'+ 'PSNR: ' + str(psnr_list[i][0])+'\nSWD: '+str(swd_list[i][0])
        label_2 = 'predicted: ' + str(
            predict_number_list[i][1]) + '\n' + 'MAE: ' + str(mae_list[i][1]) + '\n' \
                  +'MSE: ' + str(mse_list[i][1]) + '\n' + 'PSNR: ' + str(psnr_list[i][1])+'\nSWD: '+str(swd_list[i][1])
        # resize_target = cv2.resize(target, (int(h_max), int(target.shape[1]*ratio_h)),
        #                            interpolation=cv2.INTER_CUBIC) / (ratio_h*ratio_h)
        # axs[0, i].set_size_inches(int(image_list[i].shape[1]*ratio_h),int(h_max))

        axs[0, i].imshow(cv2.resize(image_list[i], (int(image_list[i].shape[1] * ratio_h), int(h_max)),
                                    interpolation=cv2.INTER_CUBIC))
        axs[1, i].imshow(cv2.resize(gt_list[i], (int(image_list[i].shape[1] * ratio_h), int(h_max)),
                                    interpolation=cv2.INTER_CUBIC), cmap=CM.jet)
        axs[1, i].annotate(label_0, (0.0, 0.0), xycoords='axes fraction', color='white')
        axs[2, i].imshow(cv2.resize(predict_list_1[i], (int(predict_list_1[i].shape[1] * ratio_h), int(h_max)),
                                    interpolation=cv2.INTER_CUBIC) / (ratio_h * ratio_h), cmap=CM.jet)
        axs[2, i].annotate(label_1, (0.0, 0.0), xycoords='axes fraction', color='white')
        axs[3, i].imshow(cv2.resize(predict_list_2[i], (int(predict_list_2[i].shape[1] * ratio_h), int(h_max)),
                                    interpolation=cv2.INTER_CUBIC) / (ratio_h * ratio_h), cmap=CM.jet)
        axs[3, i].annotate(label_2, (0.0, 0.0), xycoords='axes fraction', color='white')
        
    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.axes('off')
    fig.set_size_inches(16, 12)
    plt.tight_layout()
    return fig

def run_test(model_path_1,model_path_2,test):
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
    ])
    root = os.path.dirname(os.path.abspath(__file__))

    # In[4]:
    very_small = '<10' if test == 'B' else '<200'
    small = '10-50' if test == 'B' else '200-500'
    medium = '50-100' if test == 'B' else '500-1000'
    large = '100-200' if test == 'B' else '1000-1500'
    very_large = '200<' if test == 'B' else '1500<'
    very_small_criterion = 10 if test == 'B' else 200
    small_criterion = 50 if test == 'B' else 500
    medium_criterion = 100 if test == 'B' else 1000
    large_criterion = 200 if test == 'B' else 1500

    result_dict_mea = {very_small: [], small: [], medium: [], large: [], very_large: []}
    result_dict_msa = {very_small: [], small: [], medium: [], large: [], very_large: []}
    result_dict_msa_pixel = {very_small: [], small: [], medium: [], large: [], very_large: []}
    result_dict_psnr = {very_small: [], small: [], medium: [], large: [], very_large: []}
    result_dict_gt = {very_small: [], small: [], medium: [], large: [], very_large: []}
    result_dict_swd = {very_small: [], small: [], medium: [], large: [], very_large: []}
    result_dict_mea_1 = {very_small: [], small: [], medium: [], large: [], very_large: []}
    result_dict_msa_1 = {very_small: [], small: [], medium: [], large: [], very_large: []}
    result_dict_msa_pixel_1 = {very_small: [], small: [], medium: [], large: [], very_large: []}
    result_dict_psnr_1 = {very_small: [], small: [], medium: [], large: [], very_large: []}
    result_dict_gt_1 = {very_small: [], small: [], medium: [], large: [], very_large: []}
    result_dict_swd_1 = {very_small: [], small: [], medium: [], large: [], very_large: []}
    # now generate the ShanghaiA's ground truth
    part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
    part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
    part_B_train = os.path.join(root, 'part_B_final/train_data', 'images')
    part_B_test = os.path.join(root, 'part_B_final/test_data', 'images')
    path_sets = [part_B_test] if test == 'B' else [part_A_test]

    model_name_1 = model_path_1
    model_name_2 = model_path_2

    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

    model_1 = CSRNet()
    model_1 = model_1.cuda()
    checkpoint = torch.load(model_name_1)
    model_1.load_state_dict(checkpoint['state_dict'])

    model_2 = CSRNet()
    model_2 = model_2.cuda()
    checkpoint = torch.load(model_name_2)
    model_2.load_state_dict(checkpoint['state_dict'])
    from matplotlib import cm
    image_list = []
    gt_list = []
    predict_list_1 = []
    predict_list_2=[]
    mse_list = []
    mae_list = []
    psnr_list = []
    gt_number_list = []
    predict_number_list = []
    swd_list = []
    original_gt_list = []
    correct_resize_gt_list = []
    reg_resize_list = []
    original_image_list = []
    for i in range(len(img_paths)):
        plane_image = Image.open(img_paths[i]).convert('RGB')
        img = transform(plane_image).cuda()
        gt_file = h5py.File(img_paths[i].replace('.jpg', '.h5').replace('images', 'ground'), 'r')
        groundtruth = np.asarray(gt_file['density'])
        sum_convovled_kernel = np.ones((8, 8))
        target = sg.convolve2d(groundtruth, sum_convovled_kernel[::-1, ::-1], mode='valid')[::8, ::8]
        output_turch_1 = model_1(img.unsqueeze(0))
        output_1 = np.array(output_turch_1.data.cpu()[0, 0, :, :])
        cur_mae_1 = abs(output_1.sum() - np.sum(groundtruth))
        cur_mse_1 = np.square(output_1.sum() - target.sum())
        cur_image_mse_1 = np.square(output_1 - target).sum()/output_1.size
        gt_sum = np.sum(groundtruth)
        output_turch_2 = model_2(img.unsqueeze(0))
        output_2 = np.array(output_turch_2.data.cpu()[0, 0, :, :])
        cur_mae_2 = abs(output_2.sum() - np.sum(groundtruth))
        cur_mse_2 = np.square(output_2.sum() - target.sum())
        cur_image_mse_2 = np.square(output_2 -target).sum()/(output_2.shape[0]*output_2.shape[1])
        cur_psnr_1 = compare_psnr(target, output_1, data_range=1.0)
        cur_ssim_1 = compare_ssim(target, output_1)
        target_turch = torch.from_numpy(target)
        target_turch  = target_turch.type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).cuda()
        target_turch = Variable(target_turch)
        cur_swd_1 = swd.swd(target_turch,output_turch_1).detach().float().unsqueeze(0).data[0].float().item()
        resize_output_1 = cv2.resize(output_1, (int(groundtruth.shape[1]), int(groundtruth.shape[0])),
                                   interpolation=cv2.INTER_CUBIC) / 64
        resize_output_2 = cv2.resize(output_2, (int(groundtruth.shape[1]), int(groundtruth.shape[0])),
                                     interpolation=cv2.INTER_CUBIC) / 64
        resize_target = cv2.resize(target, (int(groundtruth.shape[1]), int(groundtruth.shape[0])),
                                   interpolation=cv2.INTER_CUBIC) / 64

        cur_psnr_2 = compare_psnr(target, output_2, data_range=1.0)
        cur_swd_2 = swd.swd(target_turch, output_turch_2).detach().float().unsqueeze(0).data[0].float().item()

        original_image_list.append(np.array(plane_image).astype(np.uint8))
        original_gt_list.append(groundtruth)
        correct_resize_gt_list.append(target)
        reg_resize_list.append(resize_target)

        image_list.append(np.array(plane_image).astype(np.uint8))
        gt_list.append(resize_target)
        predict_list_1.append(resize_output_1)
        predict_list_2.append(resize_output_2)
        mae_list.append((round(cur_mae_1, 2),round(cur_mae_2, 2)))
        mse_list.append((round(cur_mse_1, 2),round(cur_mse_2, 2)))
        psnr_list.append((round(cur_psnr_1, 2),round(cur_psnr_2, 2)))
        swd_list.append((round(cur_swd_1, 2),round(cur_swd_2, 2)))
        gt_number_list.append(round(np.sum(groundtruth), 2))
        predict_number_list.append((round(output_1.sum(), 2),round(output_2.sum(), 2)))
        
        run_it = True
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

        result_dict_swd[criterion].append(cur_swd_1)
        result_dict_gt[criterion].append(gt_sum)
        result_dict_mea[criterion].append(cur_mae_1)
        result_dict_msa[criterion].append(cur_mse_1)
        result_dict_msa_pixel[criterion].append(cur_image_mse_1)
        result_dict_psnr[criterion].append(cur_psnr_1)

        result_dict_swd_1[criterion].append(cur_swd_2)
        result_dict_gt_1[criterion].append(gt_sum)
        result_dict_mea_1[criterion].append(cur_mae_2)
        result_dict_msa_1[criterion].append(cur_mse_2)
        result_dict_msa_pixel_1[criterion].append(cur_image_mse_2)
        result_dict_psnr_1[criterion].append(cur_psnr_2)

        if run_it and len(image_list) > 4:
            #image_list, gt_list, predict_list_1, predict_list_2, mae_list, psnr_list, gt_number_list, predict_number_list,swd_list

            fig = create_heat_map_compare_two_models(image_list, gt_list, predict_list_1, predict_list_2, mae_list,mse_list, psnr_list, gt_number_list,predict_number_list, swd_list)
            name_1 = model_name_1.split('/')[-1].split('.')[0]
            name_2 = model_name_2.split('/')[-1].split('.')[0]
            path = os.path.join(root,'compare_models',name_1+'_'+name_2)
            os.makedirs(path,exist_ok=True)
            fig.savefig(os.path.join(path,name_1+'_'+name_2+'_compare_models_'+str(i)+'.png'))
            # fig_mea = crate_graph_per_number_of_gt(deepcopy(result_dict_mea),'MEA')
            # fig_msa = crate_graph_per_number_of_gt(deepcopy(result_dict_msa), 'MSA')
            # fig_ssim = crate_graph_per_number_of_gt(deepcopy(result_dict_ssim), 'SSIM')
            # fig_psnr = crate_graph_per_number_of_gt(deepcopy(result_dict_psnr), 'PSNR')
            # fig_swd = crate_graph_per_number_of_gt(deepcopy(result_dict_swd), 'SWD')
            image_list = []
            gt_list  = []
            predict_list_1   = []
            predict_list_2 = []
            ssim_list = []
            mae_list = []
            psnr_list = []
            gt_number_list = []
            predict_number_list = []



        print('the psnr_1')
        print(i, cur_psnr_1)
        print('the psnr_2')
        print(i, cur_psnr_2)
        print('the mae_1')
        print(i, cur_mae_1)
        print('the mae_2')
        print(i, cur_mae_2)
        print('the mse_1')
        print(i, cur_mse_1)
        print('the mse_2')
        print(i, cur_mse_2)
        print('the mse_pixel_1')
        print(i, cur_image_mse_1)
        print('the mse_pixel_1')
        print(i, cur_image_mse_2)
    name_1 = model_name_1.split('/')[-1].split('.')[0]
    name_2 = model_name_2.split('/')[-1].split('.')[0]
    cur_path = os.path.join(root,'compare_models', name_1 + '_' + name_2)
    os.makedirs(cur_path, exist_ok=True)
    cur_result_dict_mea_1 = {}
    cur_result_dict_msa_1 = {}
    cur_result_dict_mea_2 = {}
    cur_result_dict_msa_2 = {}
    for key in result_dict_mea.keys():
        cur_result_dict_mea_1[key] = np.sum(result_dict_mea[key]) / np.sum(result_dict_gt[key]) if np.sum(result_dict_gt[key])  > 0 else 0
        cur_result_dict_msa_1[key] = result_dict_msa[key]
        cur_result_dict_mea_2[key] = np.sum(result_dict_mea_1[key]) / np.sum(result_dict_gt[key])if np.sum(result_dict_gt[key])  > 0 else 0
        cur_result_dict_msa_2[key] = result_dict_msa_1[key]
    fig_mea = create_graph_per_number_of_gt_vs_two_models(deepcopy(cur_result_dict_mea_1),deepcopy(cur_result_dict_mea_2), 'MAE')
    fig_msa = create_graph_per_number_of_gt_vs_two_models(deepcopy(cur_result_dict_msa_1),deepcopy(cur_result_dict_msa_2), 'MSE')
    # fig_ssim = create_graph_per_number_of_gt_vs_two_models(deepcopy(result_dict_ssim),deepcopy(cur_result_dict_mea), 'SSIM')
    fig_psnr = create_graph_per_number_of_gt_vs_two_models(deepcopy(result_dict_psnr),deepcopy(result_dict_psnr_1), 'PSNR')
    fig_swd = create_graph_per_number_of_gt_vs_two_models(deepcopy(result_dict_swd),deepcopy(result_dict_swd_1), 'SWD')
    fig_mse_pixel = create_graph_per_number_of_gt_vs_two_models(deepcopy(result_dict_msa_pixel), deepcopy(result_dict_msa_pixel_1), 'Pixel MSE')
    os.makedirs(cur_path, exist_ok=True)
    fig_mea.savefig(os.path.join(cur_path, 'MEA_bar_' + str(i) + '.png'))
    os.makedirs(cur_path, exist_ok=True)
    fig_msa.savefig(os.path.join(cur_path,'MSA_bar_' + str(i) + '.png'))
    os.makedirs(cur_path, exist_ok=True)
    fig_psnr.savefig(os.path.join(cur_path,'PSNR_bar_' + str(i) + '.png'))
    os.makedirs(cur_path, exist_ok=True)
    fig_swd.savefig(os.path.join(cur_path,'swd_bar_' + str(i) + '.png'))
    os.makedirs(cur_path, exist_ok=True)
    fig_mse_pixel.savefig(os.path.join(cur_path, 'per_pixel_mse_bar_' + str(i) + '.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_path_1', default='/home/eldan/Python_Projects/CSRNet-pytorch-master/training_models/partBmodel_best.pth.tar',
                        type=str)
    parser.add_argument('--models_path_2', default='/media/eldan/Storage/csr_net_models/part_b/new_fixed_swd_lr_1e-9_part_b_model_best.pth.tar',
                        type=str)
    parser.add_argument('--part', default="A", type=str)
    args, _ = parser.parse_known_args()
    run_test(args.models_path_1,args.models_path_2, args.part)