import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from imresize import ImresizeWrapper
import numpy as np
import random
from time import time




class SRDataset(Dataset):
    def __init__(self, LR_path, GT_path, lazy_load=True, transform=None, dataset_size=-1):
        self.LR_path = LR_path
        self.GT_path = GT_path
        self.lazy_load = lazy_load
        self.transform = transform
        self.dataset_size = dataset_size
        
        self.LR_img = sorted(os.listdir(LR_path))
        self.GT_img = sorted(os.listdir(GT_path))
        self.real_size = len(self.LR_img)
        
        if(not lazy_load):
            start_time = time()
            self.LR_img = [np.array(Image.open(os.path.join(self.LR_path, lr)).convert('RGB')).astype(np.uint8) for lr in self.LR_img]
            self.GT_img = [np.array(Image.open(os.path.join(self.GT_path, gt)).convert('RGB')).astype(np.uint8) for gt in self.GT_img]
            print('[*] Time spent on non-lazy loading: {:.1f}s'.format(time() - start_time))
        

    def __len__(self):
        if(self.dataset_size == -1):
            return len(self.LR_img)
        else:
            return self.dataset_size
        

    def __getitem__(self, i):
        if(self.lazy_load):
            GT = np.array(Image.open(os.path.join(self.GT_path, self.GT_img[i % self.real_size])).convert('RGB'))
            LR = np.array(Image.open(os.path.join(self.LR_path, self.LR_img[i % self.real_size])).convert('RGB'))
        else:
            GT = self.GT_img[i % self.real_size].astype(np.float32)
            LR = self.LR_img[i % self.real_size].astype(np.float32)

        if self.transform is not None:
            for tr in self.transform:
                GT, LR = tr(GT, LR)

        img_item = {}
        img_item['GT'] = GT
        img_item['LR'] = LR
        img_item['GT'] = img_item['GT'].transpose(2, 0, 1).astype(np.float32) / 255.
        img_item['LR'] = img_item['LR'].transpose(2, 0, 1).astype(np.float32) / 255.

        return img_item
    


class SRDatasetOnlyGT(Dataset):
    def __init__(self, GT_path, LR_transform=0.5, lazy_load=True, transform=None, dataset_size=-1):
        self.GT_path = GT_path
        self.lazy_load = lazy_load
        self.transform = transform
        self.LR_transform = ImresizeWrapper(scale_factor=LR_transform, vdsr_like=True) if isinstance(LR_transform, float) else LR_transform
        self.dataset_size = dataset_size
        
        self.GT_img = sorted(os.listdir(GT_path))
        self.real_size = len(self.GT_img)
        
        if(not lazy_load):
            start_time = time()
            self.GT_img = [np.array(Image.open(os.path.join(self.GT_path, gt)).convert('RGB')).astype(np.uint8) for gt in self.GT_img]
            print('[*] Time spent on non-lazy loading: {:.1f}s'.format(time() - start_time))
        

    def __len__(self):
        if(self.dataset_size == -1):
            return len(self.GT_img)
        else:
            return self.dataset_size
        

    def __getitem__(self, i):
        if(self.lazy_load):
            GT = np.array(Image.open(os.path.join(self.GT_path, self.GT_img[i % self.real_size])).convert('RGB'))
        else:
            GT = self.GT_img[i % self.real_size].astype(np.float32)

        if self.transform is not None:
            for tr in self.transform:
                GT = tr(GT)

        LR = self.LR_transform(GT)

        img_item = {}
        img_item['GT'] = GT
        img_item['LR'] = LR
        img_item['GT'] = img_item['GT'].transpose(2, 0, 1).astype(np.float32) / 255.
        img_item['LR'] = img_item['LR'].transpose(2, 0, 1).astype(np.float32) / 255.

        return img_item
    



class crop(object):
    def __init__(self, scale, patch_size):
        self.scale = scale
        self.patch_size = patch_size
        

    def __call__(self, *inputs):
        ih, iw = inputs[0].shape[:2]
        ix = random.randrange(0, iw - self.patch_size +1)
        iy = random.randrange(0, ih - self.patch_size +1)

        output_list = [] 
        for inp in inputs:
            output_list.append(inp[iy : iy + self.patch_size, ix : ix + self.patch_size])
        
        if(len(output_list) > 1):
            return output_list
        else:
            return output_list[0]



class augmentation(object):
    def __call__(self, *inputs):

        hor_flip = random.randrange(0,2)
        ver_flip = random.randrange(0,2)
        rot = random.randrange(0,2)

        output_list = []
        for inp in inputs:
            if hor_flip:
                tmp_inp = np.fliplr(inp)
                inp = tmp_inp.copy()
                del tmp_inp
            if ver_flip:
                tmp_inp = np.flipud(inp)
                inp = tmp_inp.copy()
                del tmp_inp
            if rot:
                inp = inp.transpose(1, 0, 2)
            output_list.append(inp)

        if(len(output_list) > 1):
            return output_list
        else:
            return output_list[0]

