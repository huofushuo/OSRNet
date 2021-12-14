#coding=utf-8

import os
import cv2
import random
import numpy as np
try:
    from . import transform
except:
    import transform
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
#rgbt
mean_rgb = np.array([[[0.551, 0.619, 0.532]]])*255
mean_t =np.array([[[0.341,  0.360, 0.753]]])*255
std_rgb = np.array([[[0.241, 0.236, 0.244]]])*255
std_t = np.array([[[0.208, 0.269, 0.241]]])*255

#DUTD
# mean_rgb = np.array([[[ 0.42454678, 0.3971446,0.44439128]]])*255
# std_rgb = np.array([[[ 0.25133714, 0.25833002,0.25390804]]])*255
#
# mean_t = np.array([[[0.5156196, 0.5156196, 0.5156196]]])*255
# std_t = np.array([[[0.27192974, 0.27192974, 0.27192974]]])*255

# [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
# mean_rgb = np.array([[[0.551, 0.619, 0.532]]])*255
# mean_t =np.array([[[0.341,  0.360, 0.753]]])*255
# std_rgb = np.array([[[0.241, 0.236, 0.244]]])*255
# std_t = np.array([[[0.208, 0.269, 0.241]]])*255

# mean_rgb = np.array([[[0.485*255, 0.456*255, 0.406*255]]])
# mean_t = np.array([[[0.485*255, 0.456*255, 0.406*255]]])
# std_rgb = np.array([[[0.229 * 255, 0.224 * 255, 0.225 * 255]]])
# std_t = np.array([[[0.229 * 255, 0.224 * 255, 0.225 * 255]]])

class Data(Dataset):
    def __init__(self, root,mode='train', dataset=''):
        self.samples = []
        self.mode = mode
        self.dataset = dataset
        lines = os.listdir(os.path.join(root, 'GT'))
        for line in lines:
            rgbpath = os.path.join(root, 'RGB', line[:-4]+'.jpg')
            tpath = os.path.join(root, 'T', line[:-4]+'.jpg')
            maskpath = os.path.join(root, 'GT', line)
            self.samples.append([rgbpath,tpath,maskpath])

        if mode == 'train':

            self.transform = transform.Compose( transform.Normalize(mean1=mean_rgb,mean2=mean_t,std1=std_rgb,std2=std_t),


                                                # transform.Resize(500, 500),
                                                # transform.RandomCrop(300,300),
                                                transform.Resize(320,320),

                                                # transform.RandomCrop(288, 288),
                                                transform.RandomHorizontalFlip(),
                                                # transform.randomRotation(),

                                                transform.ToTensor(),

                                               )

        elif mode == 'test':


            self.transform = transform.Compose(
                transform.Normalize(mean1=mean_rgb, mean2=mean_t, std1=std_rgb, std2=std_t),
                transform.Resize(320, 320),
                transform.ToTensor(),
                )

    def __getitem__(self, idx):

        rgbpath,tpath,maskpath = self.samples[idx]
        rgb = cv2.imread(rgbpath)
        t = cv2.imread(tpath)
        mask = cv2.imread(maskpath)

        H, W, C = mask.shape
        rgb = rgb.astype('float32')
        t = t.astype('float32')
        mask = mask.astype('float32')
        rgb, t, mask = self.transform(rgb, t, mask)
        maskpath_short = maskpath.split('\\')[-1]

        return rgb, t, mask, (H, W), maskpath_short


    def __len__(self):
        return len(self.samples)