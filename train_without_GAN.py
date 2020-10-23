import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch

from GDmodels import GeneratorUNet, weights_init_normal
from datasets import DRIVEDataset
from test import testt_accuracy

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=2001, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="DRIVE", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=5, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=50, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=512, help='size of image height')
parser.add_argument('--img_width', type=int, default=512, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=1000, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=50, help='interval between model checkpoints')
parser.add_argument('--path', type=str, default='./', help='path to code and data')

opt = parser.parse_args()
print(opt)

os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_model/%s' % opt.dataset_name, exist_ok=True)

print(torch.cuda.is_available())

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion = nn.BCELoss()

criterion_pixelwise = torch.nn.L1Loss()

# 初始化 Unet
Unet = GeneratorUNet()


if cuda:
    Unet = torch.nn.DataParallel(Unet).cuda()
    criterion.cuda()
    criterion_pixelwise.cuda()

if opt.epoch != 0:
    # Load pretrained models
    # Unet.load_state_dict(torch.load(os.path.join(opt.path, '/saved_model/%s/generator_%d.pth' % (opt.dataset_name, opt.epoch))))
    Unet.load_state_dict(torch.load("./saved_model/DRIVE/generator_1350.pth"))
    accs = list(np.load(os.path.join("D:/SegmentationProject2/saved_model/DRIVE/accs.npy")))
    losses = list(np.load(os.path.join("D:/SegmentationProject2/saved_model/DRIVE/glosses.npy")))
else:
    # Initialize weights
    Unet.apply(weights_init_normal)
    losses = []
    accs = []

# Optimizers
optimizer = torch.optim.Adam(Unet.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders
transforms_ = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
               transforms.RandomResizedCrop(512),
               transforms.RandomHorizontalFlip(),
               transforms.RandomVerticalFlip(),
               # transforms.RandomRotation(),
               transforms.ToTensor()]

transforms_A = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5),(0.5))]
transforms_B = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                transforms.ToTensor(),]

dataloader = DataLoader(DRIVEDataset(os.path.join(opt.path, "dataset/%s" % opt.dataset_name), transformsA=transforms_A, transformsB=transforms_B),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

print('len of train batch is: ', len(dataloader))
val_dataloader = DataLoader(DRIVEDataset(os.path.join(opt.path, "dataset/%s" % opt.dataset_name), transformsA=transforms_A, transformsB=transforms_B, mode='val'),
                            batch_size=1, shuffle=True, num_workers=0)
print('len of val batch is: ', len(val_dataloader))



# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_images(batches_done, path):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs['A'].type(Tensor))
    real_B = Variable(imgs['B'].type(Tensor))
    pred_B = Unet(real_A)
    os.makedirs(os.path.join(path, 'images/%s/img/' % opt.dataset_name), exist_ok=True)
    os.makedirs(os.path.join(path, 'images/%s/gt/' % opt.dataset_name), exist_ok=True)
    os.makedirs(os.path.join(path, 'images/%s/pred/' % opt.dataset_name), exist_ok=True)
    save_image(real_A, 'images/%s/img/%s_image.png' % (opt.dataset_name, batches_done), nrow=3, normalize=True, scale_each=True)
    save_image(real_B, 'images/%s/gt/%s_image.png' % (opt.dataset_name, batches_done), nrow=3, normalize=True, scale_each=True)
    save_image(pred_B, 'images/%s/pred/%s_image.png' % (opt.dataset_name, batches_done), nrow=3, normalize=True, scale_each=True)

def dice_coeff(seg, gt):
    smooth = .0001
    return (np.sum(seg[gt == 1])*2.0 + smooth) / (np.sum(seg) + np.sum(gt) + smooth)

def evaluationMetric(image, thresh, path):
    # print(image)
    path = os.path.join(path, 'test_results/pred_map/')
    map_GT = Image.open(image)
    map_pred = Image.open(os.path.join(path, image.split('\\')[-1]))
    w, h = map_GT.size

    # Binarize the prediction and groundtruth
    true_labels = np.reshape(np.asarray(map_GT)[:,:,1], (w*h,1))
    true_labels = np.where(true_labels > thresh, 1, 0)
    pred_labels = np.reshape(np.asarray(map_pred)[:,:,1], (w*h,1))
    pred_labels = np.where(pred_labels > thresh, 1, 0)

    ##Dice Coeff
    dice = dice_coeff(pred_labels,true_labels)

    return dice

def sample_images2(imgs, model, path, idx=None,save=False):
    """Saves a generated sample from the test set"""
    true_map = Variable(imgs['B'].type(Tensor))
    save_image(true_map, os.path.join(path, 'test_results/gt', str(idx)) +'.jpg', normalize=True)
    fake_B = model(Variable(imgs['A'].type(Tensor)))
    if save:
        save_image(fake_B, os.path.join(path, 'test_results/pred_map', str(idx)) +'.jpg', normalize=True)
    return fake_B


prev_time = time.time()

start_time = time.process_time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # transforms.Compose([transforms.Normalize((0.5),(0.5))])(batch['A'])
        # Model inputs
        # real_A = Variable(transforms.Compose(transforms_A)(batch['A']).type(Tensor))
        # real_B = Variable(transforms.Compose([transforms.Normalize((0.5),(0.5))])(batch['B']).type(Tensor))
        real_A = Variable(batch['A'].type(Tensor))
        real_B = Variable(batch['B'].type(Tensor))

        # ------------------
        #  训练网络
        # ------------------

        optimizer.zero_grad()

        # loss
        predicted = Unet(real_A)
        predicted = F.sigmoid(predicted)
        loss = criterion(predicted, real_B)

        loss.backward()



        optimizer.step()

        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()


        sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %s" %
                         (epoch, opt.n_epochs,
                          i, len(dataloader),
                          loss.item(),
                          time_left))
        print(" loss len: " + str(len(losses)) + " acc len: " + str(len(accs)))
        # 输出一次验证集结果
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done, opt.path)

    losses.append(loss.item())
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(Unet.state_dict(), os.path.join(opt.path, 'saved_model/%s/generator_%d.pth' % (opt.dataset_name, epoch)))
        np.save(os.path.join(opt.path, 'saved_model/%s/glosses.npy'% (opt.dataset_name)), losses)

        # calculate accuracy and append
        Unet.eval()
        acc = testt_accuracy(if_save=False)
        print("current acc:" + str(acc))
        accs.append(acc)
        np.save(os.path.join(opt.path, 'saved_model/%s/accs'% (opt.dataset_name)), accs)
        Unet.train()



print("use time:" + str(time.process_time() - start_time))