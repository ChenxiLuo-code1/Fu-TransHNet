import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import cv2


class SkinDataset(data.Dataset):
    """
    dataloader for skin lesion segmentation tasks
    """
    def __init__(self, image_root, gt_root):
        self.images = np.load(image_root)     #np.load()从.npy扩展名的磁盘文件返回输入数组,读取.npy文件
        self.gts = np.load(gt_root)
        self.size = len(self.images)          #获得数据集的大小

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),                                 #由HWC转换为CHW格式，然后变成float格式，最后每个像素除以255
            transforms.Normalize([0.485, 0.456, 0.406],            #归一化的mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]
                                 [0.229, 0.224, 0.225])            #x=(x-mean)/std
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])
        
        self.transform = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),
                A.ColorJitter(),
                A.HorizontalFlip(),
                A.VerticalFlip()
            ]
        )

    def __getitem__(self, index):      #给你一个索引的时候返回一个图像的tensor和target的tensor
        
        image = self.images[index]
        gt = self.gts[index]
        gt = gt/255.0

        transformed = self.transform(image=image, mask=gt)
        image = self.img_transform(transformed['image'])
        gt = self.gt_transform(transformed['mask'])
        return image, gt

    def __len__(self):      #返回数据的个数   self.size=1450

        return self.size


def get_loader(image_root, gt_root, batchsize, shuffle=True, num_workers=4, pin_memory=True):

    dataset = SkinDataset(image_root, gt_root)

    data_loader = data.DataLoader(dataset=dataset,   # 从数据库中每次抽出batch_size个样本
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root):

        self.images = np.load(image_root)
        self.gts = np.load(gt_root)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.images[self.index]
        image = self.transform(image).unsqueeze(0)


        gt = self.gts[self.index]

        gt = gt/255.0

        self.index += 1

        return image, gt



if __name__ == '__main__':
    path = 'D:/Wyy_FuTransHNet/TrainData/'
    tt = SkinDataset(path+'data_val.npy', path+'mask_val.npy')

    for i in range(50):
        img, gt = tt.__getitem__(i)

        img = torch.transpose(img, 0, 1)
        img = torch.transpose(img, 1, 2)
        img = img.numpy()
        gt = gt.numpy()

        plt.imshow(img)
        #plt.savefig('vis/'+str(i)+".jpg")
 
        plt.imshow(gt[0])
        #plt.savefig('vis/'+str(i)+'_gt.jpg')
