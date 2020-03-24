import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2
import scipy.signal as sg

def load_data(args):
    img_path, train = args
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    if False:
        crop_size = (int(img.size[0]/2),int(img.size[1]/2))
        if random.randint(0,9)<= 2:

            crop_size = (int(img.size[0]),int(img.size[1]))
            dx = 0
            dy = 0
        else:
            dx = int(random.random()*img.size[0]*1./2)
            dy = int(random.random()*img.size[1]*1./2)
        
        
        
        img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        target = target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]
        
        
        
        
        if random.random()>0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    sum_convovled_kernel = np.ones((8, 8))
    # target = cv2.resize(target,(int(target.shape[1]/8),int(target.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64
    target = sg.convolve2d(target, sum_convovled_kernel[::-1, ::-1], mode='valid')[::8, ::8]
    
    return img, target
