import tensorflow as tf
import os
import numpy as np
from PIL import Image
import skimage.color as sc

def make_dir(dir_path):
	if not(os.path.isdir(dir_path)):
		os.makedirs(dir_path)
	
def get_all_img_paths_gen(path_root, min_h=0, min_w=0):
	for (dirpath, dirnames, filenames) in os.walk(path_root):
		filenames = [f for f in filenames if not f[0] == '.']
		dirnames[:] = [d for d in dirnames if not d[0] == '.']

		for file in filenames:
			if (file.endswith(tuple(['.bmp', '.jpg', '.png']))):
				path = os.path.join(dirpath, file)
				if min_h == 0 and min_w == 0:
					yield path
				else:
					img = Image.open(path)
					if(img.mode != 'RGB'):
						continue
					img_size = img.size
					h = img_size[1]
					w = img_size[0]
					if(min_h <= h and min_w <= w):
						yield path
					
def get_all_img_paths(path_root, min_h=0, min_w=0):
	paths = []
	for path in get_all_img_paths_gen(path_root, min_h, min_w):
		paths.append(path)
	return sorted(paths)

def get_all_imgs(path_root):
	paths = get_all_img_paths(path_root)
	imgs = []
	for path in paths:
		img = Image.open(path)
		imgs.append(np.array(img))
	return imgs

def get_frame_paths(paths):
	ret = []
	tmp_s = set()

	for p in paths:
		frame_dir_path = os.path.dirname(p)
		file_name = os.path.basename(p)

		if not frame_dir_path in tmp_s:
			tmp_s.add(frame_dir_path)
			ret.append([frame_dir_path, []])

		ret[-1][1].append(file_name)
	return ret
					
def psnr(img1, img2, ycbcr=False, shave=0):
	if ycbcr:
		a = np.float32(img1)
		b = np.float32(img2)
		a = sc.rgb2ycbcr(a / 255)[:, :, 0]
		b = sc.rgb2ycbcr(b / 255)[:, :, 0]
	else:
		a = np.array(img1).astype(np.float32)
		b = np.array(img2).astype(np.float32)
		
	if shave:
		a = a[shave:-shave, shave:-shave]
		b = b[shave:-shave, shave:-shave]
	
	mse = np.mean((a - b) ** 2 )
	if mse == 0:
		return 100
	PIXEL_MAX = 255.0
	return np.minimum(100.0, 20 * np.math.log10(PIXEL_MAX) - 10 * np.math.log10(mse))

def get_folder_img_name(img_path):
	img_path = os.path.abspath(img_path)
	img_path_split = img_path.split('/')
	folder_name = img_path_split[-2]
	img_name = img_path_split[-1]
	img_name = os.path.splitext(img_name)[0]
	return folder_name, img_name

def save_png(img_np, path):
	path = os.path.abspath(path)
	dir_path = os.path.dirname(path)
	make_dir(dir_path)
	
	if (img_np.dtype != np.uint8):
		img_np = quantize_np(img_np)
	img = Image.fromarray(img_np)
	img.save(path + '.png')

def normalize(img):
	return ((img/255)-0.5)*2
	
def unnormalize(img):
	return (img+1)*0.5*255

def quantize_np(img_np):
	return np.uint8(np.add(img_np, 0.5).clip(0, 255))

def downscale_args(scale, *args):
	return np.array(args) // scale

def upscale_args(scale, *args):
	return np.array(args) * scale

def resize_img(img_np, height, width, resample=Image.BICUBIC):
	img = Image.fromarray(img_np)
	resized = img.resize((width, height), resample=resample)
	return np.array(resized)

def downscale_img(img_np, scale, resample=Image.BICUBIC):
	h, w, c = img_np.shape
	h, w = downscale_args(scale, h, w)
	return resize_img(img_np, h, w, resample)
	
def upscale_img(img_np, scale, resample=Image.BICUBIC):
	h, w, c = img_np.shape
	h, w = upscale_args(scale, h, w)
	return resize_img(img_np, h, w, resample)

def np_crop(nparr, y, x, h, w=None):
	if w == None:
		w = h
	return nparr[y:y+h, x:x+w]

def modular_crop(img_np, scale):
	h, w, c = img_np.shape
	h, w = downscale_args(scale, h, w)
	h, w = upscale_args(scale, h, w)
	return np_crop(img_np, 0, 0, h, w)
	
def resize_set(img_np, scale, resample=Image.BICUBIC):
	h, w, c = img_np.shape
	nw = w / scale
	nh = h / scale
	assert(nw == int(nw) and nh == int(nh))
	lr = downscale_img(img_np, scale, resample)
	bicubic = upscale_img(lr, scale, resample)
	return lr, bicubic

def file_dir_path(file):
	file_path = os.path.relpath(file)
	dir_path = os.path.dirname(file_path)
	return dir_path




#======================tensorflow=============================

def getVariable(shape, name):
	initializer = tf.initializers.GlorotUniform()
	return tf.Variable(initializer(shape=shape), name=name)

def conv2d(x, w, stride=1, padding='SAME', dilation=1):
	return tf.nn.conv2d(x, w, strides=[1,stride,stride,1], padding=padding, dilations=[1, dilation, dilation, 1])

def clip255(x):
	return tf.clip_by_value(x, 0, 255)

def mae_img(pred, label):
	mae = tf.math.abs(pred - label)
	mae = tf.math.reduce_mean(mae, [1,2,3])
	return mae
		
def mse_img(pred, label):
	mse = tf.math.squared_difference(pred, label)
	mse = tf.math.reduce_mean(mse, [1,2,3])
	return mse

def lrelu(x):
	return tf.nn.leaky_relu(x, alpha=0.01)

def tconv2d(value, filter, output_shape, stride, padding='SAME'):
	return tf.nn.conv2d_transpose(value, filter, output_shape, strides=[1, stride, stride, 1], padding=padding)