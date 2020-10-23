import torch
import argparse
import os
import numpy as np
import time
import glob
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from data import ImageFolder


from datasets import DRIVEDataset

'''import models'''
from GDmodels import GeneratorUNet, GeneratorUnetW
from officialUnet import UNet
'''----------------configuration---------------------'''


parser = argparse.ArgumentParser()
parser.add_argument('--normed_dir',  type=str, default='./test_results/syn_seg/', help='save generated segmented images under directory')
parser.add_argument('--threshed_dir',  type=str, default='./test_results/syn_threshed_seg/', help='save generated segmented images under directory')

def make_threshed(norm_img_path, thresh, threshed_img_dir):
    norm_img = Image.open(norm_img_path)
    w, h = norm_img.size
    # Binarize the prediction and groundtruth
    labels = np.asarray(norm_img)
    # labels = np.reshape(np.asarray(norm_img)[:,:,1], (w*h,1))
    labels = np.where(labels > thresh, 255, 0)
    # plt.imshow(labels)
    # plt.show()
    # print(labels.shape)
    im = Image.fromarray(np.uint8(labels))
    # im.show()
    im.save(os.path.join(threshed_img_dir, norm_img_path.split('\\')[-1]))


opt = parser.parse_args()


def gen():
    # elapse = []#count time
    os.makedirs(opt.threshed_dir, exist_ok=True)
    for seg_path in glob.glob(os.path.join(opt.normed_dir, '*.jpg'), recursive=True):
        print(seg_path)
        thresh =40   # grayscale intensity
        make_threshed(seg_path,thresh,opt.threshed_dir)



if __name__ == "__main__":
    gen()

