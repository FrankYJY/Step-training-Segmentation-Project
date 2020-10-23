import random
import cv2
from torchvision import transforms
import torchvision.transforms.functional as ttf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL








def makecon():
    path1 = "./dataset/DRIVE/test/1st_manual/"
    path2 = "./dataset/DRIVE/test/1st_manual_tv_resized/"
    save_path = "./dataset/DRIVE/test/contrast/"
    for i in range(1, 21):
        print("doing "+str(i))

        if i<10:
            prefix = "0"+str(i)
        else:
            prefix = str(i)
        img1_path = path1 + prefix + "_manual1.gif"  # 拼出图片路径和文件名
        img2_path = path2 + prefix + "_manual1.gif"  # 拼出图片路径和文件名
        img1 = PIL.Image.open(img1_path)  # 读入图片
        img2 = PIL.Image.open(img2_path)  # 读入图片



        # manual_img_path = root + "m" + "_" + str(num) + ".jpg"
        # manual_img = PIL.Image.open(manual_img_path)
        # gen_img_path = root + name + "_" + str(num) + ".jpg"
        # gen_img = PIL.Image.open(gen_img_path)
        # manual_img = manual_img.convert("1")
        # result = np.array(gen_img)
        # gen_img = gen_img.convert("1")
        # # manual_img.show()
        # # gen_img.show()
        # manual = np.array(manual_img)
        # gen = np.array(gen_img)
        result = np.array(img1)

        img1 = np.array(img1)
        img2 = np.array(img2)
        # print(result)

        # print(manual.shape)
        # print(gen.shape)
        count=0
        for i in range(512):
            for j in range(512):
                if(img1[i][j] > img2[i][j]):
                    result[i][j][0] = 255
                    result[i][j][1] = 0
                    result[i][j][2] = 0
                    count=count+1
                if(img1[i][j] < img2[i][j]):
                    result[i][j][0] = 0
                    result[i][j][1] = 255
                    result[i][j][2] = 0
                    count = count + 1
        print(str(count))
        result = Image.fromarray(result)

        result.save(save_path + prefix + "_"+"contrast.jpg")

makecon()