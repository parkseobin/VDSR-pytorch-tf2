import numpy as np

def get_rand_aug():
	flip_idx = np.random.randint(2)
	rot_idx = np.random.randint(4)
	return flip_idx, rot_idx
	
def randInt(x, size):
	return np.random.randint(x - size)
	
def randPos(h, w, size):
	y = randInt(h, size)
	x = randInt(w, size)
	return y, x
	
def augmentation(img_np, flip_idx, rot_idx):
	if(flip_idx):
		img_np = np.fliplr(img_np)
	img_np = np.rot90(img_np, rot_idx)
	return img_np

def gaussian_noise_np(img, std):
	noise = np.random.normal(scale=std, size=np.shape(img))
	return np.clip(img + noise, 0, 255)