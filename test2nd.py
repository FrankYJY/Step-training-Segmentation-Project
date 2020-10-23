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
from PIL import Image
from data import ImageFolder


from datasets import DRIVEDataset

'''import models'''
from GDmodels import GeneratorUNet, GeneratorUnetW
from officialUnet import  UNet

'''----------------configuration---------------------'''

#rarely changed
parser = argparse.ArgumentParser()
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')

#data settings
parser.add_argument('--dataset_name', type=str, default="DRIVE", help='name of the dataset')
parser.add_argument('--img_height', type=int, default=512, help='size of image height')
parser.add_argument('--img_width', type=int, default=512, help='size of image width')

#pathes
#path is for specified file path， readable
#dir is the directory, folder
parser.add_argument('--root', type=str, default='./', help='root path')
parser.add_argument('--test_root', type=str, default='./test_results/', help='containing folders of different training names')
parser.add_argument('--real_img_dir', type=str, default='./test_results/real_img/', help='save raw images under directory')
parser.add_argument('--real_seg_dir', type=str, default='./test_results/real_seg/', help='save manual segmented images under directory')
parser.add_argument('--syn_seg_dir',  type=str, default='./test_results/syn_seg/', help='save generated segmented images under directory')
parser.add_argument('--model_dir', type=str, default='./test_using_models/generator_100_official_2ndGAN_rightseq.pth', help='trained model')

generator = UNet(in_channels=1)

'''--------------------------------------------------'''

def dice_coeff(seg, real_seg):


    smooth = .0001
    TP =np.sum(seg[real_seg == 1])
    FP =np.sum(seg[real_seg == 0])
    FN =np.sum(1-seg[real_seg == 1])
    TN =np.sum(1-seg[real_seg == 0])
    P =TP + FN
    N =FP + TN
    precision = TP/(TP+FP)
    sensitivity = recall = TP/(TP+FN)
    specificity = TN/(TN+FP)
    SP = TN/(TN+FP)
    acc = (TP + TN) / float(TP + TN + FP + FN)
    SN = TP / (TP + FN)
    f1 =(2*precision*recall)/(precision+recall)
    IOU = TP / float(FP + TP + FN)
    dice = (np.sum(seg[real_seg == 1])*2.0 + smooth) / (np.sum(seg) + np.sum(real_seg) + smooth) #origin
    #f1 = dice
    return dice


def generate_fake(imgs, model, root, idx=None, save=True):
    """Saves a generated sample from the test set"""
    true_map = Variable(imgs['B'].type(Tensor))
    fake_B = model(Variable(imgs['A'].type(Tensor)))
    if save:
        save_image(true_map, os.path.join(opt.real_seg_dir, str(idx)) + '.jpg', normalize=True)
        save_image(fake_B, os.path.join(opt.syn_seg_dir, str(idx)) +'.jpg', normalize=True)
    return fake_B

def evaluationMetric(real_seg_path, thresh, syn_seg_dir):
    real_img = Image.open(real_seg_path)
    map_pred = Image.open(os.path.join(syn_seg_dir, real_seg_path.split('\\')[-1]))#:file with same name
    w, h = real_img.size
    # Binarize the prediction and groundtruth
    true_labels = np.reshape(np.asarray(real_img)[:,:,1], (w*h,1))
    true_labels = np.where(true_labels > thresh, 1, 0)
    pred_labels = np.reshape(np.asarray(map_pred)[:,:,1], (w*h,1))
    pred_labels = np.where(pred_labels > thresh, 1, 0)

    ##Dice Coeff
    dice = dice_coeff(pred_labels,true_labels)

    return dice


# unet  0.7593516820601052
# gan   0.7723407320835018

# unet1 0.4247292582876866

opt = parser.parse_args()
print("parser parameters:"+str(opt))

cuda = True if torch.cuda.is_available() else False

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

if cuda:
    generator = torch.nn.DataParallel(generator).cuda()

use_GAN = False

# 测试加入GAN机制的分割
if use_GAN == True:
    test_dataset = ImageFolder(mode="test")
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=0)

# 测试单纯使用UNET分割
else:
    transforms_as =[transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                        # transforms.Grayscale(num_output_channels=1),
                        # transforms.RandomRotation(180),
                        # transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.05),
                        #
                        # transforms.RandomResizedCrop(512),
                        # transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(0.5, 0.5)
                 ]
    transforms_bs = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                     # transforms.RandomRotation(180),
                     # transforms.RandomResizedCrop(512),
                     # transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                     transforms.Normalize((0.5),(0.5))
                     ]
    test_dataloader = DataLoader(DRIVEDataset(os.path.join(opt.root, "dataset/%s" % opt.dataset_name), transformsA=transforms_as, transformsB=transforms_bs, mode='val',
                                              is2ndtrain=True),
                                 batch_size=1, shuffle=False, num_workers=0)

generator.load_state_dict(torch.load(opt.model_dir))
generator.eval()


#mathod name can not start with test_, or install pytest module
#if need to save, set all args
def testt_accuracy():
    # elapse = []#count time
    os.makedirs(opt.real_seg_dir, exist_ok=True)
    os.makedirs(opt.real_img_dir, exist_ok=True)
    os.makedirs(opt.syn_seg_dir, exist_ok=True)
    #os.path.join(opt.root, 'test_results/pred_map/')
    for i, image in enumerate(test_dataloader):
        print('Testing image: ', i)
        # start = time.process_time()
        generate_fake(image, generator, opt.root, idx=i)
        # elapsed = (time.process_time() - start)
        # elapse.append(elapsed)
        save_image(image['A'], os.path.join(opt.real_img_dir, str(i) +'.jpg'), normalize=True)
    # print('Average testing time is ', np.mean(elapse))
    BER_avg   = []
    dice_avg = []
    print('dicing all...')
    for real_seg_path in glob.glob(os.path.join(opt.real_seg_dir, '*.jpg'), recursive=True):
        thresh = 40  # grayscale intensity
        dice = evaluationMetric(real_seg_path, thresh, opt.syn_seg_dir)
        dice_avg.append(dice)

    accuracy = sum(dice_avg) / float(len(dice_avg))
    print("accuracy:" + str(accuracy))
    return accuracy

if __name__ == "__main__":
    testt_accuracy()

