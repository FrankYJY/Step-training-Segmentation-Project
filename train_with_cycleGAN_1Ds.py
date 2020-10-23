import itertools

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
import matplotlib.pyplot as plt

from datasets import DRIVEDataset

'''import models'''
from officialUnet import UNet,UNetSoft
from GDmodels import GeneratorUNet,GeneratorUNetIN1,GeneratorUNetWide,GeneratorUNetDeep,GeneratorUnetW, PixelDiscriminator,PixelDiscriminatorIN1, weights_init_normal
from AllResNetModels import resnet34, resnet50
# from officialUnet_actF_trial import UNet_LRelu

'''----------------configuration---------------------'''

# ----------------rarely changed-----------------
parser = argparse.ArgumentParser()
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')

# Adam optimizer settings
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--lrdecay', type=float, default=25, help='adam: decay of first order momentum of gradient')



# ------------------------------------------------

# loop settings
parser.add_argument('--start_epoch', type=int, default=0, help='==0: initialize, !=0: load parameters in epoch x and train')
parser.add_argument('--end_epoch', type=int, default=75, help='epoch to end, inclusive')
parser.add_argument('--mid_sample_interval', type=int, default=5,
                    help='interval for making middle results, for check in running process; checkpoint_interval % mid_sample_interval == 0, for reading file and generate accuracy')
parser.add_argument('--checkpoint_interval', type=int, default=25, help='interval of checkpoints, calculate accuracy and save all')

# data settings
parser.add_argument('--dataset_name', type=str, default="DRIVE", help='name of the dataset')
parser.add_argument('--img_height', type=int, default=512, help='size of image height')
parser.add_argument('--img_width', type=int, default=512, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--patch_numbers_along_orientations'
                    '', type=int, default=2 ** 5, help='patch numbers along orientations: row, column')

# pathes
# path is for specified file path， readable
# dir is the directory, folder
parser.add_argument('--root', type=str, default='./', help='root path')
parser.add_argument('--save_root', type=str, default='./storage/', help='save D, G, D loss, G loss , accuracy')
parser.add_argument('--mid_root', type=str, default='./mid_images/', help='save intermediate result')
#path that save temp result when test accuracy with test dataset, nothing to do with training
parser.add_argument('--temp_test_root', type=str, default='./test_results/tempcycle/', help='store temp test results')

# name of training
parser.add_argument('--net_name', type=str, default='offcyc norm trial',
                    help='the name of training this time, part of the subdirectory of save and image')

# choose generator and discriminator
generator_forward = UNet(drop=0.3, in_channels=1)
generator_backward = UNet(drop=0.3, in_channels=1)
discriminator_forward = PixelDiscriminator(input_nc=2)
discriminator_backward = PixelDiscriminator(input_nc=2)

# should be same if not transfer lightness, contrast, .etc
# if gray, use IN1 generator and discriminator, and ban convert"L" in dataset-DRIVEDataset()
img_dealing_method = "right lightness rotate crop flip"
seg_dealing_method = "rotate crop flip"


'''--------------------------------------------------'''

opt = parser.parse_args()
print(opt)

if opt.checkpoint_interval % opt.mid_sample_interval != 0:
    raise Exception("sample interval is inside checkpoint interval, CI%SI must be 0")
# create folders

store_name = opt.net_name + " img-"+ img_dealing_method +" seg-" +seg_dealing_method

os.makedirs(opt.mid_root, exist_ok=True)#./mid_images/
mid_dir = opt.mid_root + store_name + "/"#./mid_images/NAME/
os.makedirs(mid_dir, exist_ok=True)  # train name folder
mid_real_seg_dir = mid_dir + 'real_seg/'   # gt  #./mid_images/NAME/real_seg/
mid_real_img_dir = mid_dir + 'real_img/'   # img
mid_syn_seg_dir  = mid_dir + 'syn_seg/'     # pred
mid_syn_img_dir  = mid_dir + 'syn_img/'     # pred
os.makedirs(mid_real_seg_dir, exist_ok=True)
os.makedirs(mid_real_img_dir, exist_ok=True)
os.makedirs(mid_syn_seg_dir, exist_ok=True)
os.makedirs(mid_syn_img_dir, exist_ok=True)

os.makedirs(opt.save_root, exist_ok=True)
save_dir = opt.save_root + store_name + "/"#./storage/NAME/
os.makedirs(save_dir, exist_ok=True)

os.makedirs(opt.temp_test_root, exist_ok=True)#./test_results/temp/
temp_real_seg_dir = opt.temp_test_root + 'real_img/'#./test_results/temp/real_img/
temp_real_img_dir = opt.temp_test_root + 'real_seg/'
temp_syn_seg_dir = opt.temp_test_root + 'syn_seg/'
temp_syn_img_dir = opt.temp_test_root + 'syn_img/'
os.makedirs(temp_real_seg_dir, exist_ok=True)
os.makedirs(temp_real_img_dir, exist_ok=True)
os.makedirs(temp_syn_img_dir, exist_ok=True)
os.makedirs(temp_syn_seg_dir, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_GAN = torch.nn.MSELoss()  # 均方根误差损失函数
# criterion_GAN = torch.nn.BCELoss()  # 均方根误差损失函数
criterion_pixelwise = torch.nn.L1Loss()  # 每一个像素的平均绝对误差损失函数

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // opt.patch_numbers_along_orientations, opt.img_width // opt.patch_numbers_along_orientations)  # （1，16，16）
print("patch:" + str(patch))


acc = []

if cuda:
    generator_forward = torch.nn.DataParallel(generator_forward).cuda()
    discriminator_forward = torch.nn.DataParallel(discriminator_forward).cuda()
    generator_backward = torch.nn.DataParallel(generator_backward).cuda()
    discriminator_backward = torch.nn.DataParallel(discriminator_backward).cuda()

    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

if opt.start_epoch != 0:
    # Load pretrained models
    generator_forward.load_state_dict(
        torch.load(os.path.join(save_dir + 'generator_forward_%d.pth' % opt.start_epoch)))
    discriminator_forward.load_state_dict(
        torch.load(os.path.join(save_dir + 'discriminator_forward_%d.pth' % opt.start_epoch)))
    generator_forward.load_state_dict(
        torch.load(os.path.join(save_dir + 'generator_backward_%d.pth' % opt.start_epoch)))
    discriminator_forward.load_state_dict(
        torch.load(os.path.join(save_dir + 'discriminator_backward_%d.pth' % opt.start_epoch)))
    Glosses = list(np.load(os.path.join(save_dir + 'Glosses_all.npy')))
    Dlosses_forward = list(np.load(os.path.join(save_dir + 'Dlosses_forward_all.npy')))
    Dlosses_backward = list(np.load(os.path.join(save_dir + 'Dlosses_backward_all.npy')))
    Glosses_epoch_average = list(np.load(os.path.join(save_dir + 'Glosses_epoch_average.npy')))
    Dlosses_forward_epoch_average = list(np.load(os.path.join(save_dir + 'Dlosses_forward_epoch_average.npy')))
    Dlosses_backward_epoch_average = list(np.load(os.path.join(save_dir + 'Dlosses_backward_epoch_average.npy')))
    accuracies = list(np.load(os.path.join(save_dir + 'accuracies.npy')))
    elapse = list(np.load(os.path.join(save_dir + 'training_time_by_epoch.npy'),allow_pickle=True))
# reference:
# torch.save(generator_forward.state_dict(), os.path.join(save_dir + 'generator_forward_%d.pth' % epoch))
# torch.save(discriminator_forward.state_dict(), os.path.join(save_dir + 'discriminator_forward_%d.pth' % epoch))
# torch.save(generator_forward.state_dict(), os.path.join(save_dir + 'generator_backward_%d.pth' % epoch))
# torch.save(discriminator_forward.state_dict(), os.path.join(save_dir + 'discriminator_backward_%d.pth' % epoch))
# np.save(os.path.join(save_dir + 'Glosses_all.npy'), Glosses)
# np.save(os.path.join(save_dir + 'Dlosses_forward_all.npy'), Dlosses_forward)
# np.save(os.path.join(save_dir + 'Dlosses_backward_all.npy'), Dlosses_backward)
# np.save(os.path.join(save_dir + 'Glosses_epoch_average.npy'), Glosses_epoch_average)
# np.save(os.path.join(save_dir + 'Dlosses_forward_epoch_average.npy'), Dlosses_forward_epoch_average)
# np.save(os.path.join(save_dir + 'Dlosses_backward_epoch_average.npy'), Dlosses_backward_epoch_average)
# np.save(os.path.join(save_dir + 'accuracies.npy'), accuracies)
# np.save(os.path.join(save_dir + 'training_time_by_epoch.npy'), elapse)

else:
    # ==0:initialize parameters
    generator_forward.apply(weights_init_normal)
    generator_backward.apply(weights_init_normal)
    discriminator_forward.apply(weights_init_normal)
    discriminator_backward.apply(weights_init_normal)
    Glosses = []#by every training
    Dlosses_forward = []
    Dlosses_backward = []
    Glosses_epoch_average = []#average by every epoch
    Dlosses_forward_epoch_average = []
    Dlosses_backward_epoch_average = []
    accuracies = []# every epoch, test with test dataset
    elapse = []#already used time, record by epoch

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(generator_forward.parameters(), generator_backward.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=40, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=10, eta_min=0, last_epoch=-1)

optimizer_D_forward = torch.optim.Adam(discriminator_forward.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_backward = torch.optim.Adam(discriminator_backward.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

def choose_transforms_method(mode = "default"):
    default = False
    if mode == "validate" or "train" :
        trans = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                          transforms.ToTensor(),
                          transforms.Normalize(0.5, 0.5)]
    elif mode == "no norm":
        trans = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                          transforms.ToTensor(),
                          transforms.Normalize(0.5, 0.5)]
    elif mode == "crop flip":
        trans = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                       transforms.RandomResizedCrop(512),
                       transforms.RandomVerticalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize(0.5, 0.5)]
    elif mode == "rotate flip":
        trans = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                        transforms.RandomRotation(180),
                        transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(0.5, 0.5)]
    elif mode == "rotate crop flip":
        trans = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                        transforms.RandomRotation(180),
                        transforms.RandomResizedCrop(512),
                        transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(0.5, 0.5)]
    elif mode == "rotate crop flip no norm":
        trans = [
            transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                        transforms.RandomRotation(180),
                        transforms.RandomResizedCrop(512),
                        transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),
                        ]
    # elif mode == "gray rotate crop flip":
    #     trans = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    #                     # transforms.Grayscale(num_output_channels=1),
    #                     transforms.RandomRotation(180),
    #                     transforms.RandomResizedCrop(512),
    #                     transforms.RandomVerticalFlip(),
    #                     transforms.ToTensor(),
    #                     transforms.Normalize(0.5, 0.5)
    #              ]
    # elif mode == "lightness rotate crop flip":
    #     #wrong
    #     trans = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    #                     # transforms.Grayscale(num_output_channels=1),
    #                     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
    #                     transforms.RandomRotation(180),
    #                     transforms.RandomResizedCrop(512),
    #                     transforms.RandomVerticalFlip(),
    #                     transforms.ToTensor(),
    #                     transforms.Normalize(0.5, 0.5)
    #              ]0
    elif mode == "right lightness rotate crop flip":
        trans = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                        # transforms.Grayscale(num_output_channels=1),
                        transforms.RandomRotation(180),
                        transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.2),
                 # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
                        transforms.RandomResizedCrop(512),
                        transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(0.5, 0.5)
                 ]
    else:
        trans = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                          transforms.ToTensor(),
                          transforms.Normalize(0.5, 0.5)]
    print("you are using transforms: " + mode)

    return trans

# # Configure dataloaders and Data Augmentation
# transforms_ = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
#                transforms.RandomResizedCrop(512),
#                transforms.RandomHorizontalFlip(),
#                transforms.RandomVerticalFlip(),
#                transforms.ToTensor(),
#                transforms.Normalize(0.5, 0.5)]
#
# transforms_val = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
#                   transforms.ToTensor(),
#                   transforms.Normalize(0.5, 0.5)]

transforms_methodA = choose_transforms_method(img_dealing_method)#real image
transforms_methodB = choose_transforms_method(seg_dealing_method)#real segmentation
transforms_valA = choose_transforms_method()# validate and test, no need changing
# transforms_valB = choose_transforms_method("no norm")# validate and test, no need changing


dataloader = DataLoader(DRIVEDataset(os.path.join(opt.root, "dataset/%s" % opt.dataset_name), mode = "train", transformsA=transforms_methodA, transformsB=transforms_methodB),
                        batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

# print('len of train batch is: ', len(dataloader))
#validation used for test cnd calculate loss
val_dataloader = DataLoader(
    DRIVEDataset(os.path.join(opt.root, "dataset/%s" % opt.dataset_name), mode='val', transformsA=transforms_valA, transformsB=transforms_valA),
    batch_size=1, shuffle=False, num_workers=opt.n_cpu)

val_dataloader_for_mid = DataLoader(DRIVEDataset(os.path.join(opt.root, "dataset/%s" % opt.dataset_name), mode='val',transformsA=transforms_methodA, transformsB=transforms_methodB),
    batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

# print('len of val batch is: ', len(val_dataloader))
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


unloader = transforms.ToPILImage()
def tensorshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(10)  # pause a bit so that plots are updated

# 用于从验证集中读取图像并且保存到相应文件夹中
def make_mid_results(epoch):
#Saves a generated sample from the validation set
    imgs = next(iter(dataloader))
    real_A = Variable(imgs['A'].type(Tensor))
    real_B = Variable(imgs['B'].type(Tensor))
    fake_B = generator_forward(real_A)
    fake_A = generator_backward(fake_B)

    # tensorshow(real_A)

    save_image(real_A, mid_real_img_dir + '%s_image.png' % epoch, nrow=3,
               normalize=False, scale_each=True)
    save_image(real_B, mid_real_seg_dir + '%s_image.png' % epoch, nrow=3,
               normalize=False, scale_each=True)
    save_image(fake_B, mid_syn_seg_dir + '%s_image.png' % epoch, nrow=3,
               normalize=False, scale_each=True)
    save_image(fake_A, mid_syn_img_dir + '%s_image.png' % epoch, nrow=3,
               normalize=False, scale_each=True)

# def generate_fake(imgs, model, idx=None):
#     """Saves a generated sample from the test set"""
#     true_map = Variable(imgs['B'].type(Tensor))
#     fake_B = model(Variable(imgs['A'].type(Tensor)))
#     save_image(true_map, os.path.join(opt.real_seg_dir, str(idx)) + '.jpg', normalize=True)
#     save_image(fake_B, os.path.join(opt.syn_seg_dir, str(idx)) +'.jpg', normalize=True)
#     return fake_B

def dice_coeff(seg, real_seg):
    smooth = .0001
    return (np.sum(seg[real_seg == 1])*2.0 + smooth) / (np.sum(seg) + np.sum(real_seg) + smooth)

def evaluationMetric(real_path, thresh, syn_dir):#real_path include the name of img, generate same-name file in syn directory, adverse real and seg can also work
    real_img = Image.open(real_path)
    map_pred = Image.open(os.path.join(syn_dir, real_path.split('\\')[-1]))#:get index of real seg, find syn seg with same name
    w, h = real_img.size
    # Binarize the prediction and groundtruth
    true_labels = np.reshape(np.asarray(real_img)[:,:,1], (w*h,1))
    true_labels = np.where(true_labels > thresh, 1, 0)
    pred_labels = np.reshape(np.asarray(map_pred)[:,:,1], (w*h,1))
    pred_labels = np.where(pred_labels > thresh, 1, 0)

    ##Dice Coeff
    dice = dice_coeff(pred_labels,true_labels)

    return dice

def testt_accuracy(generator_to_test):
    for i, image in enumerate(val_dataloader):
        # generate_fake(image, generator_to_test, idx=i) # for reference
        true_map = Variable(image['B'].type(Tensor))
        fake_B = generator_to_test(Variable(image['A'].type(Tensor)))

        save_image(image['A'], os.path.join(temp_real_img_dir, str(i) +'.jpg'), normalize=False)
        save_image(true_map, os.path.join(temp_real_seg_dir, str(i)) + '.jpg', normalize=False)
        save_image(fake_B, os.path.join(temp_syn_seg_dir, str(i)) + '.jpg', normalize=False)

    # print('Average testing time is ', np.mean(elapse))
    dice_avg = []
    for real_seg_path in glob.glob(os.path.join(temp_real_seg_dir, '*.jpg'), recursive=True):
        thresh = 40  # grayscale intensity
        dice = evaluationMetric(real_seg_path, thresh, temp_syn_seg_dir)
        dice_avg.append(dice)
    accuracy = sum(dice_avg) / float(len(dice_avg))
    return accuracy



start_time = time.process_time()
prev_time = time.time()
lrdecay_times =1

for epoch in range(opt.start_epoch, opt.end_epoch+1):
    Glosses_epoch_temp =[]
    Dlosses_forward_epoch_temp =[]
    Dlosses_backward_epoch_temp =[]
    for i, pair in enumerate(dataloader):

        # Model inputs
        real_A = Variable(pair['A'].type(Tensor))
        real_B = Variable(pair['B'].type(Tensor))


        # print((real_A.size(0)))
        # print(*patch)
        # Adversarial ground truths （B,1,16,16）
        # patch2 = [1, 16, 16]
        #is [1,1,16,16]
        #for discriminator use
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)
        # ------------------
        #  训练生成器模型
        # ------------------


        optimizer_G.zero_grad()
        # print("realA shape:", real_A.shape)
        # make fake
        fake_B = generator_forward(real_A)
        fake_A = generator_backward(fake_B)


        pred_fake_forward = discriminator_forward(real_A, fake_B)
        pred_fake_backward = discriminator_backward(real_B, fake_A)
        loss_GAN_forward = criterion_GAN(pred_fake_forward, valid)  # 均方根误差损失函数
        loss_GAN_backward = criterion_GAN(pred_fake_backward, valid)  # 均方根误差损失函数


        # Pixel-wise loss
        loss_pixel_forward = criterion_pixelwise(fake_B, real_B)  # 平均绝对误差损失函数
        loss_pixel_backward = criterion_pixelwise(fake_A, real_A)  # 平均绝对误差损失函数

        # 损失权重  D投票的均方差太大，不能直接加   pixel / D_result
        lambda_pixel = 0.99

        # importance: forward/backward
        lambda_f_b = 0.8
        # print(str(loss_GAN.item())+" "+str(loss_pixel.item()))
        # Total loss
        loss_G = lambda_f_b * ((1 - lambda_pixel) * loss_GAN_forward + lambda_pixel * loss_pixel_forward) + (1-lambda_f_b) * ((1 - lambda_pixel) * loss_GAN_backward + lambda_pixel * loss_pixel_backward)

        loss_G.backward()

        optimizer_G.step()


        # ---------------------
        #  训练判别器1
        # ---------------------

        optimizer_D_forward.zero_grad()

        # Real loss
        pred_real_forward = discriminator_forward(real_A, real_B)  # 真实的
        loss_real_forward = criterion_GAN(pred_real_forward, valid)
        # print(pred_real.shape)
        # print(valid.shape)

        # Fake loss
        pred_fake_forward = discriminator_forward(real_A, fake_B.detach())  # 虚假的
        loss_fake_forward = criterion_GAN(pred_fake_forward, fake)

        # Total loss
        loss_D_forward = 0.5 * (loss_real_forward + loss_fake_forward)  # 判别器的损失函数 让这个函数越小越好

        loss_D_forward.backward()
        optimizer_D_forward.step()


        # ---------------------
        #  训练判别器2
        # ---------------------
        optimizer_D_backward.zero_grad()

        # Real loss
        pred_real_backward = discriminator_backward(real_A, real_B)  # 真实的
        loss_real_backward = criterion_GAN(pred_real_backward, valid)
        # print(pred_real.shape)
        # print(valid.shape)



        # Fake loss
        pred_fake_backward = discriminator_backward(real_B, fake_A.detach())  # 虚假的
        loss_fake_backward = criterion_GAN(pred_fake_backward, fake)

        # Total loss
        loss_D_backward = 0.5 * (loss_real_backward + loss_fake_backward)  # 判别器的损失函数 让这个函数越小越好

        loss_D_backward.backward()
        optimizer_D_backward.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        # batches_done = epoch * len(dataloader) + i
        # batches_left = opt.end_epoch * len(dataloader) - batches_done
        # time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        # prev_time = time.time()

        # Print log
        sys.stdout.write("\r[Epoch %d/%d Batch %d/%d] [inGloss: D1: %f, P1: %f, D2: %f, P2: %f] [D1 loss: %f, D2 loss: %f, G loss: %f]" %  #, pixel: %f, adv: %f
                         (epoch, opt.end_epoch,
                          i, len(dataloader),
                        loss_GAN_forward ,loss_pixel_forward,loss_GAN_backward , loss_pixel_backward,
                          loss_D_forward.item(),loss_D_backward.item(), loss_G.item(),
                          # loss_pixel_forward.item(), loss_GAN_forward.item(),
                          ))
        # record every train (20 record for 20 images in a epoch)
        Glosses.append(loss_G.item())
        Dlosses_forward.append(loss_D_forward.item())
        Dlosses_backward.append(loss_D_backward.item())
        Glosses_epoch_temp.append(loss_G.item())
        Dlosses_forward_epoch_temp.append(loss_D_forward.item())
        Dlosses_backward_epoch_temp.append(loss_D_backward.item())

    # scheduler.step(metrics=loss_G)

    #record every epoch
    Glosses_current_epoch_average = sum(Glosses_epoch_temp) / float(len(Glosses_epoch_temp))
    Dlosses_forward_current_epoch_average = sum(Dlosses_forward_epoch_temp) / float(len(Dlosses_forward_epoch_temp))
    Dlosses_backward_current_epoch_average = sum(Dlosses_backward_epoch_temp) / float(len(Dlosses_backward_epoch_temp))

    Glosses_epoch_average.append(Glosses_current_epoch_average)
    Dlosses_forward_epoch_average.append(Dlosses_forward_current_epoch_average)
    Dlosses_backward_epoch_average.append(Dlosses_backward_current_epoch_average)

    generator_forward.eval()
    # acc = testt_accuracy(generator_to_test=generator_forward)
    acc =0.5#skip val(no effect), can save a lot of time
    accuracies.append(acc)
    generator_forward.train()

    time_used = datetime.timedelta(seconds=(time.process_time() - start_time))
    elapse.append(time_used)
    time_left = datetime.timedelta(seconds=(time.process_time() - start_time)/(epoch+1-opt.start_epoch)*(opt.end_epoch-epoch))


    #sometimes accuracy can be 0, just restart
    print(" [current epoch average: D1 loss: %f, D2 loss: %f, G loss: %f] [Accuracy: %f] [already use time: %s time left: %s] [lr %f]" %
          (Dlosses_forward_current_epoch_average, Dlosses_backward_current_epoch_average, Glosses_current_epoch_average, acc, time_used, time_left, optimizer_G.state_dict()['param_groups'][0]['lr']))

    if epoch != 0 and epoch % opt.lrdecay == 0:
        optimizer_G = torch.optim.Adam(generator_forward.parameters(), lr=opt.lr / (10 ** lrdecay_times), betas=(opt.b1, opt.b2))
        lrdecay_times = lrdecay_times+1

    # If at sample interval save image
    if opt.mid_sample_interval !=-1 and epoch % opt.mid_sample_interval == 0:
        make_mid_results(epoch)

    if opt.checkpoint_interval != -1 and epoch != 0 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator_forward.state_dict(), os.path.join(save_dir + 'generator_forward_%d.pth' % epoch))
        torch.save(discriminator_forward.state_dict(), os.path.join(save_dir + 'discriminator_forward_%d.pth' % epoch))
        torch.save(generator_backward.state_dict(), os.path.join(save_dir + 'generator_backward_%d.pth' % epoch))
        torch.save(discriminator_backward.state_dict(), os.path.join(save_dir + 'discriminator_backward_%d.pth' % epoch))
        np.save(os.path.join(save_dir + 'Glosses_all.npy'), Glosses)
        np.save(os.path.join(save_dir + 'Dlosses_forward_all.npy'), Dlosses_forward)
        np.save(os.path.join(save_dir + 'Dlosses_backward_all.npy'), Dlosses_backward)
        np.save(os.path.join(save_dir + 'Glosses_epoch_average.npy'), Glosses_epoch_average)
        np.save(os.path.join(save_dir + 'Dlosses_forward_epoch_average.npy'), Dlosses_forward_epoch_average)
        np.save(os.path.join(save_dir + 'Dlosses_backward_epoch_average.npy'), Dlosses_backward_epoch_average)
        np.save(os.path.join(save_dir + 'accuracies.npy'), accuracies)
        np.save(os.path.join(save_dir + 'training_time_by_epoch.npy' ), elapse)

