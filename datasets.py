import os
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt

unloader = transforms.ToPILImage()
def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(10)  # pause a bit so that plots are updated

def read_DRIVE_datasets(root_path, mode='train'):
    images = []
    segmentations = []
    if mode == 'train':
        image_root = os.path.join(root_path, 'training/images/')
        gt_root = os.path.join(root_path, 'training/1st_manual/')
        for image_name in os.listdir(image_root):
            image_path = os.path.join(image_root, image_name.split('.')[0] + '.tif')
            label_path = os.path.join(gt_root, image_name.split('_')[0] + '_manual1.gif')
            # image_path = os.path.join(image_root, image_name.split('.')[0] + '.jpg')
            # label_path = os.path.join(gt_root, image_name.split('.')[0] + '.jpg')

            images.append(image_path)
            segmentations.append(label_path)
    else:
        image_root = os.path.join(root_path, 'test/images')
        gt_root = os.path.join(root_path, 'test/1st_manual')
        for image_name in os.listdir(image_root):
            image_path = os.path.join(image_root, image_name.split('.')[0] + '.tif')
            label_path = os.path.join(gt_root, image_name.split('_')[0] + '_manual1.gif')
            # print(str(image_path))
            # print(str(label_path))
            # print("jajajaj")

            # image_path = os.path.join(image_root, image_name.split('.')[0] + '.jpg')
            # label_path = os.path.join(gt_root, image_name.split('.')[0] + '.jpg')

            images.append(image_path)
            segmentations.append(label_path)



    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(images)
    random.seed(randnum)
    random.shuffle(segmentations)

    return images, segmentations



def read_special_datasets_for_2nd(root_path, mode='train'):
    images = []
    segmentations = []
    if mode=='train':
        image_root = "./test_results/storage of 1st step/syn_seg"
        gt_root = "./test_results/storage of 1st step/real_seg"
    if mode=='test':
        image_root = "./test_results/storage of 1st step test/syn_seg"
        gt_root = "./test_results/storage of 1st step test/real_seg"
    print("is 2nd")
    for image_name in os.listdir(image_root):
        # print('reading'+image_name)
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.jpg')
        label_path = os.path.join(gt_root, image_name.split('.')[0] + '.jpg')
        images.append(image_path)
        segmentations.append(label_path)

    randnum = random.randint(0, 100)
    # random.seed(randnum)
    # random.shuffle(images)
    # random.seed(randnum)
    # random.shuffle(segmentations)

    return images, segmentations
            
class DRIVEDataset(Dataset):
    def __init__(self, root, mode, transformsA, transformsB, is2ndtest = False, is2ndtrain = False ):
        # print(transforms_)
        # print(transforms_2)
        self.transform = transforms.Compose(transformsA)
        self.transform_2 = transforms.Compose(transformsB)

        if is2ndtest:
            self.images , self.masks = read_special_datasets_for_2nd(root, mode='test')
        elif is2ndtrain:
            self.images, self.masks = read_special_datasets_for_2nd(root, mode='train')
        else:
            self.images , self.masks = read_DRIVE_datasets(root,mode=mode)



    def __getitem__(self, index):

        img_A = Image.open(self.images[index % len(self.images)])
        img_A = img_A.convert('L')
        img_B = Image.open(self.masks[index % len(self.images)])
        img_B = img_B.convert('L')
        # print(img_B.size)
        # np.set_printoptions(threshold=np.inf)
        # print(np.asarray(img_B))
        # img_B = img_B.convert('1')
        # print(img_A)
        # print(img_B.size)

        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)  # apply this seed to img tranfsorms
        img_A = self.transform(img_A)

        # print(img_A.shape)
        # print(img_A)
        # imshow(img_A)
        # img_B.show()

        random.seed(seed)  # apply this seed to target tranfsorms
        img_B = self.transform_2(img_B)

        # imshow(img_B)
        # img_B = torch.clamp(img_B, 0, 1)
        # imshow(img_B)

        return {'A': img_A, 'B': img_B}

    def __len__(self):
        return len(self.images)