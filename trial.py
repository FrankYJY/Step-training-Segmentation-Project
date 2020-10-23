# -------------------------------
#
# really trials
# 试验，无作用
#
# -------------------------------

import random
import cv2
from torchvision import transforms
import torchvision.transforms.functional as ttf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import os
import numpy as np
import glob
import time
import datetime
import sys
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch.nn as nn
import cv2 as cv




root = "dataset/FIRE/"




# def makecon(name,num):
#     manual_img_path = root + "m" + "_" + str(num) + ".jpg"
#     manual_img = PIL.Image.open(manual_img_path)
#     gen_img_path = root + name + "_" + str(num) + ".jpg"
#     gen_img = PIL.Image.open(gen_img_path)
#     manual_img = manual_img.convert("1")
#     result = np.array(gen_img)
#     gen_img = gen_img.convert("1")
#     # manual_img.show()
#     # gen_img.show()
#     manual = np.array(manual_img)
#     gen = np.array(gen_img)
#     # print(result)
#
#     # print(manual.shape)
#     # print(gen.shape)
#
#     for i in range(512):
#         for j in range(512):
#             if(manual[i][j] > gen[i][j]):
#                 result[i][j][0] = 255
#                 result[i][j][1] = 0
#                 result[i][j][2] = 0
#             if(manual[i][j] < gen[i][j]):
#                 result[i][j][0] = 0
#                 result[i][j][1] = 255
#                 result[i][j][2] = 0
#     res = Image.fromarray(result)
#     res.save(root + name + "_" + str(num) + "contrast.jpg")

# for i in range(6):
#     a_path = root + str(i) + ".jpg"
#     a =PIL.Image.open(a_path)
#     a = a.resize((512,512),Image.ANTIALIAS)
#     a.save(root + "0"+str(i+1)+"_test.tif")

# os.makedirs('./aaa/', exist_ok=True)

# a = [1,16,16]
# np.ones(3,a)

# print(np.ones((1*2*3),(4*5*6)).size)


# y1 = np.load("E:/SegmentationProject2/storage/GAN lightness img-lightness rotate crop flip seg-rotate crop flip/Glosses_epoch_average.npy")
# y2 = np.load("E:/SegmentationProject2/storage/GAN lightness img-lightness rotate crop flip seg-rotate crop flip/accuracies.npy")
# print(y1.size)
# x = np.arange(1, y1.size + 1, 1)
# plt.figure()
# plt.xlim((0, 80))
# plt.ylim((0, 1))
#
# l1, = plt.plot(x, y1, label='generator losses')
# l2, = plt.plot(x, y2, label='accuracy', color='red', linewidth=1.0, linestyle='--')
# plt.legend(loc='upper right')
# plt.show()

# Glosses = []
# a = 0.23423423
# for epoch in range(0, 10000):
#     Glosses.append(a)
# np.save("./trial.npy", Glosses)

# class net(nn.Module):
#     def __init__(self):
#         super(net,self).__init__()
#         self.fc = nn.Linear(1,10)
#     def forward(self,x):
#         return self.fc(x)
#
# import numpy as np
# lr_list = []
# model = net()
# LR = 0.01
# T_max = 10
# optimizer = torch.optim.Adam(model.parameters(),lr = LR)
# lambda1 = lambda epoch:np.sin(epoch) / epoch
# # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = lambda1)
# # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
#
#
#
# for epoch in range(1,100):
#     scheduler.step(metrics=1)
#     lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
# plt.plot(range(1,100),lr_list,color = 'r')
# plt.show()

# np.set_printoptions(threshold=np.inf)
# img_A = Image.open("./mid_images/GAN_Unet -losstrial dropout0.2 img-lightness rotate crop flip seg-rotate crop flip/syn_seg/45_image.png")
# print(np.asarray(img_A))



# a=Image.open('./dataset/DRIVE/training/images/37_training.tif')
# a = np.array(a)
# print(a.shape)
# b = np.zeros((584,565,3))
# b[:,:,0] = a[:,:,0]
# b[:,:,1] = a[:,:,0]
# b[:,:,2] = a[:,:,0]
#
# print(b.shape)
# cv.imshow(",,",b)
# cv.waitKey(0)
# cv.destroyAllWindows()

a=np.array([[0,1,1],[1,0,1],[0,0,1]])
b=np.array([[0,1,1],[1,1,1],[1,1,1]])
print(a)
print(b)
print(a[b==0])
print(np.sum(a[b==0]))