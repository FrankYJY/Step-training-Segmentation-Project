import random
import cv2
from torchvision import transforms
import torchvision.transforms.functional as ttf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL

path_training = "dataset/DRIVE/training/images/"
path_manual = "dataset/DRIVE/training/1st_manual/"
outpath_training = "dataset/augmented/training/"
outpath_manual = "dataset/augmented/manual/"


def rotate_pair(image, shadow):
    # 拿到角度的随机数。angle是一个-180到180之间的一个数
    angle = transforms.RandomRotation.get_params([-180, 180])
    # 对image和mask做相同的旋转操作，保证他们都旋转angle角度
    image = image.rotate(angle)
    shadow = shadow.rotate(angle)

    # image = ttf.to_tensor(image)
    # shadow = ttf.to_tensor(shadow)
    return image, shadow


def flip_pair(image, shadow):
    # 50%的概率应用垂直，水平翻转。
    if random.random() > 0.5:
        image = ttf.hflip(image)
        shadow = ttf.hflip(shadow)
    if random.random() > 0.5:
        image = ttf.vflip(image)
        shadow = ttf.vflip(shadow)
    # image = ttf.to_tensor(image)
    # shadow = ttf.to_tensor(shadow)
    return image, shadow

def blockedgauss(mu,sigma,upper,bottom):
    while True:
        numb = random.gauss(mu,sigma)
        if (numb > bottom and numb < upper):
            break
    return numb

def jitter_pair(image, shadow):
    image = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0)(image)
    return image, shadow

def zoom_pair(image, shadow):
    coefficient1 = random.random()
    coefficient2 = random.random()
    img_w = image.width
    img_h = image.height
    sha_w = shadow.width
    sha_h = shadow.height
    image = transforms.Resize((100, 200))(image)
    shadow = transforms.Resize((100, 200))(shadow)

    transforms.RandomResizedCrop()


#
# # crop function
# def rand_crop(image, shadow):
#     width1 = random.randint(0, image.size[0] - img_w)
#     height1 = random.randint(0, image.size[1] - img_h)
#     width2 = width1 + img_w
#     height2 = height1 + img_h
#
#     image = image.crop((width1, height1, width2, height2))
#     shadow = shadow.crop((width1, height1, width2, height2))
#
#     return image, shadow


def pair_deal(path1, path2):
    count = 1
    for i in range(21, 41):
        img_path1 = path1 + str(i) + "_training.tif"  # 拼出图片路径和文件名
        img1 = PIL.Image.open(img_path1)  # 读入图片
        img_path2 = path2 + str(i) + "_manual1.gif"  # 拼出图片路径和文件名
        img2 = PIL.Image.open(img_path2)  # 读入图片
        img1.save(outpath_training + str(count) + '_training.tif')  #store raw image
        img2.save(outpath_manual + str(count) + '_manual1.gif')
        count = count + 1
        for j in range(1, 30):
            img3, img4 = rotate_pair(img1, img2)
            img3, img4 = flip_pair(img3, img4)
            img3, img4 = jitter_pair(img3, img4)
            img3.save(outpath_training+ str(count)+ '_training.tif')
            img4.save(outpath_manual+ str(count)+ '_manual1.gif')
            count = count + 1
        # 生成图片先不归一化
        print('done:' + str(i))  # 打印状态提示



pair_deal(path_training, path_manual)
# new_im.save(os.path.join(outfile, '1.jpg'))
