from .utils_io import get_rand_aug, augmentation, randPos
from utils import get_all_img_paths, np_crop, resize_img

import numpy as np

from PIL import Image



class IO_SR():
	def __init__(self, img_root):
		self.img_paths = get_all_img_paths(img_root)
		
		self.gt = []
		for img_path in self.img_paths:
			img = Image.open(img_path)
			img = np.asarray(img)
			self.gt.append(img)
		
		self.ds_size = len(self.gt)

	
	def _next(self, lr_size, hr_size, img_idx=None):
		if img_idx == None:
			img_idx = np.random.randint(self.ds_size)

		gt = self.gt[img_idx]
		h, w, _ = gt.shape
		
		y, x = randPos(h, w, hr_size)
		hr = np_crop(gt, y, x, hr_size)
		hr_flip, hr_rot = get_rand_aug()
		hr = augmentation(hr, hr_flip, hr_rot)

		lr = resize_img(hr, lr_size, lr_size, Image.BICUBIC)
		upcubic = resize_img(lr, hr_size, hr_size, Image.BICUBIC)
		
		return lr, upcubic, hr


	def random_crop(self, lr_size, hr_size, img_idx=None):
		if img_idx == None:
			img_idx = np.random.randint(self.ds_size)

		gt = self.gt[img_idx]
		h, w, _ = gt.shape
		
		y, x = randPos(h, w, hr_size)
		hr = np_crop(gt, y, x, hr_size)
		hr_flip, hr_rot = get_rand_aug()
		hr = augmentation(hr, hr_flip, hr_rot)

		# lr = resize_img(hr, lr_size, lr_size, Image.BICUBIC)
		# upcubic = resize_img(lr, hr_size, hr_size, Image.BICUBIC)
		
		return hr