import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import random


'''
TODO: 
    [] decide how to implement lazy loading
        > three kind of loading: 
    [] change in_memory -> lazy_load

TODO: in train.py
    [] saving parameter file without param folder pre-made
    [] break loop when lr is below some value
    [] log time spent on loading all images when in_memory
    [] change lr on plateau scale...
'''


class SRDataset(Dataset):
    def __init__(self, LR_path, GT_path, in_memory=False, transform=None, dataset_size=-1):
        self.LR_path = LR_path
        self.GT_path = GT_path
        self.in_memory = in_memory
        self.transform = transform
        self.dataset_size = dataset_size
        
        self.LR_img = sorted(os.listdir(LR_path))
        self.GT_img = sorted(os.listdir(GT_path))
        self.real_size = len(self.LR_img)
        
        if in_memory:
            self.LR_img = [np.array(Image.open(os.path.join(self.LR_path, lr)).convert('RGB')).astype(np.uint8) for lr in self.LR_img]
            self.GT_img = [np.array(Image.open(os.path.join(self.GT_path, gt)).convert('RGB')).astype(np.uint8) for gt in self.GT_img]
        

    def __len__(self):
        if(self.dataset_size == -1):
            return len(self.LR_img)
        else:
            return self.dataset_size
        

    def __getitem__(self, i):
        img_item = {}
        
        if self.in_memory:
            GT = self.GT_img[i % self.real_size].astype(np.float32)
            LR = self.LR_img[i % self.real_size].astype(np.float32)
        else:
            GT = np.array(Image.open(os.path.join(self.GT_path, self.GT_img[i % self.real_size])).convert('RGB'))
            LR = np.array(Image.open(os.path.join(self.LR_path, self.LR_img[i % self.real_size])).convert('RGB'))

        img_item['GT'] = GT
        img_item['LR'] = LR

        if self.transform is not None:
            img_item = self.transform(img_item)

        img_item['GT'] = img_item['GT'].transpose(2, 0, 1).astype(np.float32) / 255.
        img_item['LR'] = img_item['LR'].transpose(2, 0, 1).astype(np.float32) / 255.

        return img_item
    



class crop(object):
    def __init__(self, scale, patch_size):
        self.scale = scale
        self.patch_size = patch_size
        

    def __call__(self, sample):
        LR_img, GT_img = sample['LR'], sample['GT']
        ih, iw = LR_img.shape[:2]
        
        ix = random.randrange(0, iw - self.patch_size +1)
        iy = random.randrange(0, ih - self.patch_size +1)
        
        LR_patch = LR_img[iy : iy + self.patch_size, ix : ix + self.patch_size]
        GT_patch = GT_img[iy : iy + self.patch_size, ix : ix + self.patch_size]
        
        return {'LR' : LR_patch, 'GT' : GT_patch}



class augmentation(object):
    def __call__(self, sample):
        LR_img, GT_img = sample['LR'], sample['GT']
        
        hor_flip = random.randrange(0,2)
        ver_flip = random.randrange(0,2)
        rot = random.randrange(0,2)
    
        if hor_flip:
            temp_LR = np.fliplr(LR_img)
            LR_img = temp_LR.copy()
            temp_GT = np.fliplr(GT_img)
            GT_img = temp_GT.copy()
            
            del temp_LR, temp_GT
        
        if ver_flip:
            temp_LR = np.flipud(LR_img)
            LR_img = temp_LR.copy()
            temp_GT = np.flipud(GT_img)
            GT_img = temp_GT.copy()
            
            del temp_LR, temp_GT
            
        if rot:
            LR_img = LR_img.transpose(1, 0, 2)
            GT_img = GT_img.transpose(1, 0, 2)
        
        
        return {'LR' : LR_img, 'GT' : GT_img}
        

