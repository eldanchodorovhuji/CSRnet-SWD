import sys
import os

import warnings

from model import CSRNet

from utils import save_checkpoint
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
# from swd.swd import swd
import numpy as np
import argparse
import json
import cv2
import dataset
import time
from swd.swd import swd

parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('--train_json', metavar='TRAIN',
                    help='path to train json',default='/home/eldan/Python_Projects/CSRNet-pytorch-master/part_A_train.json')
parser.add_argument('--test_json', metavar='TEST',
                    help='path to test json',default='/home/eldan/Python_Projects/CSRNet-pytorch-master/part_A_val.json')

parser.add_argument('--pre', '-p', metavar='PRETRAINED',type=str,
                    help='path to the pretrained model',default = '')
parser.add_argument('--gpu',metavar='GPU', type=str,
                    help='GPU id to use.',default='0')

parser.add_argument('--task',metavar='TASK', type=str,
                    help='task id to use.',default='swd_lr_1e-8_part_a_')

def main():
    global args, best_prec1

    best_prec1 = 1e9

    args = parser.parse_args()
    args.original_lr = 1e-8
    args.lr = 1e-8
    args.batch_size = 1
    args.momentum = 0.95
    args.decay = 5 * 1e-4
    args.start_epoch = 0
    args.epochs = 400*4
    args.steps = [-1, 30, 100, 150]
    args.scales = [1, 0.1, 0.1, 0.1]
    args.workers = 8
    args.seed = time.time()
    args.print_freq = 30
    print(args.task)
    print(args.lr)
    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:
        val_list = json.load(outfile)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)

    model = CSRNet()

    model = model.cuda()

    # criterion = nn.MSELoss(size_average=False).cuda()
    criterion = swd
    # optimizer = torch.optim.Adam(model.parameters(), args.lr)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))
    data_loader = dataset.listDataset(train_list,
                                     shuffle=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                     std=[0.229, 0.224, 0.225]),
                                     ]),
                                     train=True,
                                     seen=model.seen,
                                     batch_size=args.batch_size,
                                     num_workers=args.workers)
    data_loader_val = dataset.listDataset(val_list,
                                         shuffle=False,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                         std=[0.229, 0.224, 0.225]),
                                         ]), train=False)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        train(model, criterion, optimizer, epoch, data_loader)
        prec1 = validate(model, args.task, data_loader_val)
        data_loader.shuffle()
        data_loader_val.shuffle()
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.task)


def train(model, criterion, optimizer, epoch,data_loader):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))

    model.train()
    end = time.time()

    for i, (img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img = img.cuda()
        img = Variable(img)
        output = model(img)

        target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        target = Variable(target)

        loss = criterion(output, target)
        # loss.requires_grad = True
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))


def validate(model,model_name,data_loader_val):
    print('begin test')
    test_loader = torch.utils.data.DataLoader(
        data_loader_val,
        batch_size=args.batch_size)

    model.eval()

    mae = 0

    for i, (img, target) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)
        output = model(img)
        # output = cv2.resize(np.array(output.data.cpu()[0, 0, :, :]),
        #                     (int(groundtruth.shape[1]), int(groundtruth.shape[0])), interpolation=cv2.INTER_CUBIC) / 64
        if not i:
            base_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'CSRNet-pytorch-master','training_models')
            os.makedirs(base_name+model_name,exist_ok=True)
            plt.imsave(base_name+model_name+'/output',np.array(output.data.cpu())[0,0,:,:])
            plt.imsave(base_name+model_name + '/gt', np.array(target.data.cpu())[0, :, :])
        mae += abs(output.data.sum() - target.sum().type(torch.FloatTensor).cuda())

    mae = mae / len(test_loader)
    print(' * MAE {mae:.3f} '
          .format(mae=mae))

    return mae


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    args.lr = args.original_lr

    for i in range(len(args.steps)):

        scale = args.scales[i] if i < len(args.scales) else 1

        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
    print('reduced lr to:',args.lr)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
