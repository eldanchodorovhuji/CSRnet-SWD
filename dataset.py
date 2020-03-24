import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import *
import torchvision.transforms.functional as F
from multiprocessing import dummy
import itertools

class listDataset(Dataset):
    def __init__(self, root_images, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=12):
        # if train:
        #     root = root_images *4
        random.shuffle(root_images)
        self.nSamples = len(root_images)
        self.pool = dummy.Pool(processes=num_workers)

        self.loaded_images_and_masks = self.pool.map(load_data, zip(root_images, [train]*self.nSamples))
        print('finished loading')
        self.cycle = itertools.cycle(self.loaded_images_and_masks)
        self.pool.close()
        self.lines = root_images
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def shuffle(self):
        random.shuffle(self.loaded_images_and_masks)
        self.cycle = itertools.cycle(self.loaded_images_and_masks)
        
    def __len__(self):
        return self.nSamples


    def __getitem__(self, index):

        assert index <= len(self), 'index range error'
        img, target = next(self.cycle)
        if False:
            crop_size = (int(img.size[0] / 2), int(img.size[1] / 2))
            if random.randint(0, 9) <= 2:

                crop_size = (int(img.size[0]), int(img.size[1]))
                dx = 0
                dy = 0
            else:
                dx = int(random.random() * img.size[0] * 1. / 2)
                dy = int(random.random() * img.size[1] * 1. / 2)

            img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
            target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

        if random.random() > 0.8:
            target = np.fliplr(target).copy()
            img = img.transpose(Image.FLIP_LEFT_RIGHT)


        
        
        if self.transform is not None:
            img = self.transform(img)
        return img, target
