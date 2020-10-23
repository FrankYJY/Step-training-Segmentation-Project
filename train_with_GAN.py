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
from officialUnet import UNet
from GDmodels import GeneratorUNet,GeneratorUNetIN1,GeneratorUNetWide,GeneratorUNetDeep,GeneratorUnetW,GeneratorUNetHardTanh_Double,GeneratorUnetW,GeneratorUNetDeep, PixelDiscriminator,PixelDiscriminatorIN1, weights_init_normal
from AllResNetModels import resnet34, resnet50
from model3 import Resnet34_Unet,Resnet34_Unet2

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
parser.add_argument('--end_epoch', type=int, default=100, help='epoch to end, inclusive')
parser.add_argument('--mid_sample_interval', type=int, default=5,
                    help='interval for making middle results, for check in running process; checkpoint_interval % mid_sample_interval == 0, for reading file and generate accuracy')
parser.add_argument('--checkpoint_interval', type=int, default=50, help='interval of checkpoints, calculate accuracy and save all')

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
parser.add_argument('--temp_test_root', type=str, default='./test_results/temp/', help='store temp test results')

# name of training
parser.add_argument('--net_name', type=str, default='resnet34 unet2 trial',#GANdeep hardtanh lightness trial   official Unet in GAN drop0.3
                    help='the name of training this time, part of the subdirectory of save and image')

# choose generator and discriminator
# generator = UNet(in_channels=1,drop=0.3)
generator = Resnet34_Unet2()
discriminator = PixelDiscriminator()

# should be same if not transfer lightness, contrast, .etc
# if gray, use IN1 generator and discriminator, and ban convert"L" in dataset-DRIVEDataset()
img_dealing_method = "right lightness rotate crop flip"
seg_dealing_method = "rotate crop flip"


'''-----------------------------------------------------------------------------------------------------------------'''

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
os.makedirs(mid_real_seg_dir, exist_ok=True)
os.makedirs(mid_real_img_dir, exist_ok=True)
os.makedirs(mid_syn_seg_dir, exist_ok=True)

os.makedirs(opt.save_root, exist_ok=True)
save_dir = opt.save_root + store_name + "/"#./storage/NAME/
os.makedirs(save_dir, exist_ok=True)

os.makedirs(opt.temp_test_root, exist_ok=True)#./test_results/temp/
temp_real_seg_dir = opt.temp_test_root + 'real_img/'#./test_results/temp/real_img/
temp_real_img_dir = opt.temp_test_root + 'real_seg/'
temp_syn_seg_dir = opt.temp_test_root + 'syn_seg/'
os.makedirs(temp_real_seg_dir, exist_ok=True)
os.makedirs(temp_real_img_dir, exist_ok=True)
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
    generator = torch.nn.DataParallel(generator).cuda()
    discriminator = torch.nn.DataParallel(discriminator).cuda()

    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

if opt.start_epoch != 0:
    # Load pretrained models
    generator.load_state_dict(
        torch.load(os.path.join(save_dir + 'generator_%d.pth' % opt.start_epoch)))
    discriminator.load_state_dict(
        torch.load(os.path.join(save_dir + 'discriminator_%d.pth' % opt.start_epoch)))
    Glosses = list(np.load(os.path.join(save_dir + 'Glosses_all.npy')))
    Dlosses = list(np.load(os.path.join(save_dir + 'Dlosses_all.npy')))
    Glosses_epoch_average = list(np.load(os.path.join(save_dir + 'Glosses_epoch_average.npy')))
    Dlosses_epoch_average = list(np.load(os.path.join(save_dir + 'Dlosses_epoch_average.npy')))
    accuracies = list(np.load(os.path.join(save_dir + 'accuracies.npy')))
    elapse = list(np.load(os.path.join(save_dir + 'training_time_by_epoch.npy'),allow_pickle=True))
# reference:
# torch.save(generator.state_dict(), os.path.join(save_dir + 'generator_%d.pth' % epoch))
# torch.save(discriminator.state_dict(), os.path.join(save_dir + 'discriminator_%d.pth' % epoch))
# np.save(os.path.join(save_dir + 'Glosses_all.npy'), Glosses)
# np.save(os.path.join(save_dir + 'Dlosses_all.npy'), Dlosses)
# np.save(os.path.join(save_dir + 'Glosses_epoch_average.npy'), Glosses_epoch_average)
# np.save(os.path.join(save_dir + 'Dlosses_epoch_average.npy'), Dlosses_epoch_average)
# np.save(os.path.join(save_dir + 'accuracies.npy'), accuracies)
# np.save(os.path.join(save_dir + 'training_time_%s-%s_by_epoch.npy' % (opt.start_epoch, opt.end_epoch)), elapse)
else:
    # ==0:initialize parameters
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    Glosses = []#by every training
    Dlosses = []
    Glosses_epoch_average = []#average by every epoch
    Dlosses_epoch_average = []
    accuracies = []# every epoch, test with test dataset
    elapse = []#already used time, record by epoch

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=40, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=10, eta_min=0, last_epoch=-1)

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

def choose_transforms_method(mode = "default"):
    default = False
    if mode == "validate" or "train" :
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
    elif mode == "gray rotate crop flip":
        trans = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                        # transforms.Grayscale(num_output_channels=1),
                        transforms.RandomRotation(180),
                        transforms.RandomResizedCrop(512),
                        transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(0.5, 0.5)
                 ]
    elif mode == "lightness rotate crop flip":
        #wrong
        trans = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                        # transforms.Grayscale(num_output_channels=1),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
                        transforms.RandomRotation(180),
                        transforms.RandomResizedCrop(512),
                        transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(0.5, 0.5)
                 ]
    elif mode == "right lightness rotate crop flip":
        trans = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                        # transforms.Grayscale(num_output_channels=1),
                        transforms.RandomRotation(180),
                        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),
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
transforms_val = choose_transforms_method()# validate and test, no need changing


dataloader = DataLoader(DRIVEDataset(os.path.join(opt.root, "dataset/%s" % opt.dataset_name), mode = "train", transformsA=transforms_methodA, transformsB=transforms_methodB),
                        batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

# print('len of train batch is: ', len(dataloader))
#validation used for test cnd calculate loss
val_dataloader = DataLoader(
    DRIVEDataset(os.path.join(opt.root, "dataset/%s" % opt.dataset_name), mode='val',transformsA=transforms_val, transformsB=transforms_val),
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
    fake_B = generator(real_A)

    # tensorshow(real_A)

    save_image(real_A, mid_real_img_dir + '%s_image.png' % epoch, nrow=3,
               normalize=False, scale_each=True)
    save_image(real_B, mid_real_seg_dir + '%s_image.png' % epoch, nrow=3,
               normalize=False, scale_each=True)
    save_image(fake_B, mid_syn_seg_dir + '%s_image.png' % epoch, nrow=3,
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

def evaluationMetric(real_seg_path, thresh, syn_seg_dir):
    real_img = Image.open(real_seg_path)
    map_pred = Image.open(os.path.join(syn_seg_dir, real_seg_path.split('\\')[-1]))#:get index of real seg, find syn seg with same name
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

        save_image(image['A'], os.path.join(temp_real_img_dir, str(i) +'.jpg'), normalize=True)
        save_image(true_map, os.path.join(temp_real_seg_dir, str(i)) + '.jpg', normalize=True)
        save_image(fake_B, os.path.join(temp_syn_seg_dir, str(i)) + '.jpg', normalize=True)

    # print('Average testing time is ', np.mean(elapse))
    dice_avg = []
    for real_seg_path in glob.glob(os.path.join(temp_real_seg_dir, '*.jpg'), recursive=True):
        thresh = 200  # grayscale intensity
        dice = evaluationMetric(real_seg_path, thresh, temp_syn_seg_dir)
        dice_avg.append(dice)
    accuracy = sum(dice_avg) / float(len(dice_avg))
    return accuracy



start_time = time.process_time()
prev_time = time.time()
lrdecay_times =1

for epoch in range(opt.start_epoch, opt.end_epoch+1):
    Glosses_epoch_temp =[]
    Dlosses_epoch_temp =[]
    for i, pair in enumerate(dataloader):

        # Model inputs
        real_A = Variable(pair['A'].type(Tensor))
        real_B = Variable(pair['B'].type(Tensor))


        # print((real_A.size(0)))
        # print(*patch)
        # Adversarial ground truths （B,1,16,16）
        # patch2 = [1, 16, 16]
        #is [1,1,16,16]
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)
        # ------------------
        #  训练生成器模型
        # ------------------


        optimizer_G.zero_grad()
        # print("realA shape:", real_A.shape)
        # GAN loss
        fake_B = generator(real_A)
        # print("fake_B shape:", fake_B.shape)
        #判别器有真A，未知B，判断B是真分割还是假分割
        pred_fake = discriminator(real_A, fake_B)
        loss_GAN = criterion_GAN(pred_fake, valid)  # 均方根误差损失函数 , 越小判别器判断为真的几率增大
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B)  # 平均绝对误差损失函数

        # 损失权重  多生成:与正确的差别
        lambda_pixel = 0.99

        # print(str(loss_GAN.item())+" "+str(loss_pixel.item()))
        # Total loss
        loss_G = (1 - lambda_pixel) * loss_GAN + lambda_pixel * loss_pixel

        loss_G.backward()

        optimizer_G.step()


        # ---------------------
        #  训练判别器
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_A, real_B)  # 真实的
        loss_real = criterion_GAN(pred_real, valid)
        # print(pred_real.shape)
        # print(valid.shape)

        # Fake loss
        pred_fake = discriminator(real_A, fake_B.detach())  # 虚假的
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)  # 判别器的损失函数 让这个函数越小越好

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        # batches_done = epoch * len(dataloader) + i
        # batches_left = opt.end_epoch * len(dataloader) - batches_done
        # time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        # prev_time = time.time()

        # Print log
        sys.stdout.write("\r[Epoch %d/%d Batch %d/%d] [D loss: %f, G loss: %f, pixel: %f, adv: %f]" %
                         (epoch, opt.end_epoch,
                          i, len(dataloader),
                          loss_D.item(), loss_G.item(),
                          loss_pixel.item(), loss_GAN.item(),
                          ))
        # record every train (20 record for 20 images in a epoch)
        # Glosses.append(loss_G.item())
        # Dlosses.append(loss_D.item())
        Glosses_epoch_temp.append(loss_G.item())
        Dlosses_epoch_temp.append(loss_D.item())

    # scheduler.step(metrics=loss_G)

    #record every epoch
    Glosses_current_epoch_average = sum(Glosses_epoch_temp) / float(len(Glosses_epoch_temp))
    Dlosses_current_epoch_average = sum(Dlosses_epoch_temp) / float(len(Dlosses_epoch_temp))
    Glosses_epoch_average.append(Glosses_current_epoch_average)
    Dlosses_epoch_average.append(Dlosses_current_epoch_average)

    generator.eval()
    acc = testt_accuracy(generator_to_test=generator)
    # acc = 0.5
    accuracies.append(acc)
    generator.train()

    time_used = datetime.timedelta(seconds=(time.process_time() - start_time))
    elapse.append(time_used)
    time_left = datetime.timedelta(seconds=(time.process_time() - start_time)/(epoch+1-opt.start_epoch)*(opt.end_epoch-epoch))


    #sometimes accuracy can be 0, just restart
    print(" [current epoch average: D loss: %f, G loss: %f] [Accuracy: %f] [already use time: %s time left: %s] [lr %f]" % (Dlosses_current_epoch_average, Glosses_current_epoch_average, acc, time_used,time_left,optimizer_G.state_dict()['param_groups'][0]['lr']))

    if epoch != 0 and epoch % opt.lrdecay == 0:
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr/(10 ** lrdecay_times), betas=(opt.b1, opt.b2))
        lrdecay_times = lrdecay_times+1

    # If at sample interval save image
    if opt.mid_sample_interval !=-1 and epoch % opt.mid_sample_interval == 0:
        make_mid_results(epoch)

    if opt.checkpoint_interval != -1 and epoch != 0 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), os.path.join(save_dir + 'generator_%d.pth' % epoch))
        torch.save(discriminator.state_dict(), os.path.join(save_dir + 'discriminator_%d.pth' % epoch))
        np.save(os.path.join(save_dir + 'Glosses_all.npy'), Glosses)
        np.save(os.path.join(save_dir + 'Dlosses_all.npy'), Dlosses)
        np.save(os.path.join(save_dir + 'Glosses_epoch_average.npy'), Glosses_epoch_average)
        np.save(os.path.join(save_dir + 'Dlosses_epoch_average.npy'), Dlosses_epoch_average)
        np.save(os.path.join(save_dir + 'accuracies.npy'), accuracies)
        np.save(os.path.join(save_dir + 'training_time_by_epoch.npy' ), elapse)

    torch.cuda.empty_cache()