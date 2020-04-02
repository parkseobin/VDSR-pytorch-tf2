from config import *
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

from io_dataset.io_sr import IO_SR
from utils import get_all_imgs, psnr, make_dir, normalize, unnormalize, clip255, save_png, modular_crop, resize_img

import numpy as np
import tensorflow as tf
import datetime
import threading, time


MODEL = MODEL_DICT[args.model]['model']
BATCH_SIZE = MODEL_DICT[args.model]['batch_size']
PATCH_SIZE = MODEL_DICT[args.model]['patch_size']
EPOCH = MODEL_DICT[args.model]['epoch']
ITER_PER_EPOCH =  1000
SCALES = [2, 3, 4]

TRAIN_GT_ROOT = DS_DICT[args.train]['train']
TEST_GT_ROOT = DS_DICT[args.test]['test']

TOTAL_ITER = EPOCH * ITER_PER_EPOCH
LEARNING_RATE = 1e-4
LAMBDA = 1e-3

def cosine_decay(global_step):
	cosine_decay = 0.5 * (1 + np.cos(np.pi * global_step / TOTAL_ITER))
	decayed = (1 - LAMBDA) * cosine_decay + LAMBDA
	decayed_learning_rate = LEARNING_RATE * decayed
	return decayed_learning_rate


# model
net = MODEL()

# optimizer
opt = tf.optimizers.Adam(LEARNING_RATE)




# train dataset
train_ds = IO_SR(TRAIN_GT_ROOT)


def gen(lr_size, hr_size):
	while True:
		hr = train_ds.random_crop(lr_size, hr_size)
		yield hr

def cubic(hr, lr_size, hr_size):
	hr.set_shape([None, hr_size, hr_size, 3])
	tlr = tf.image.resize(hr, [lr_size, lr_size], tf.image.ResizeMethod.BICUBIC, antialias=True)
	tup = tf.image.resize(tlr, [hr_size, hr_size], tf.image.ResizeMethod.BICUBIC)
	return tlr, tup, hr


def get_iter(lr_size, hr_size):
	dataset = tf.data.Dataset.from_generator(gen, (tf.float32), args=(lr_size, hr_size))
	dataset = dataset.batch(BATCH_SIZE)
	dataset = dataset.map(lambda x: cubic(x, lr_size, hr_size))
	dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
	return iter(dataset)


ds2_it = get_iter(PATCH_SIZE//2, PATCH_SIZE)
ds3_it = get_iter(PATCH_SIZE//3, PATCH_SIZE)
ds4_it = get_iter(PATCH_SIZE//4, PATCH_SIZE)

# test set
eval_gt_imgs = get_all_imgs(TEST_GT_ROOT)


CODE_NAME = 'run_sr'
DIR_NAME = net.name
DIR_PATH = os.path.join(CODE_NAME, DIR_NAME)
LOG_PATH = os.path.join('logs', '{}'.format(DIR_PATH))



# train step
@tf.function
def _train_step(next_lr, input2, input3, input4, hr2, hr3, hr4):
	# print ('_train_step')
	opt.learning_rate.assign(next_lr)

	with tf.GradientTape() as tape:
		output2 = net(input2)
		output3 = net(input3)
		output4 = net(input4)

		loss2 = tf.reduce_mean(tf.abs(output2 - normalize(hr2)))
		loss3 = tf.reduce_mean(tf.abs(output3 - normalize(hr3)))
		loss4 = tf.reduce_mean(tf.abs(output4 - normalize(hr4)))
		cost = loss2 + loss3 + loss4
		# tf.print(loss2)

	grads = tape.gradient(cost, net.trainable_variables)
	opt.apply_gradients(zip(grads, net.trainable_variables))

	y_pred2 = clip255(unnormalize(output2))
	y_pred3 = clip255(unnormalize(output3))
	y_pred4 = clip255(unnormalize(output4))

	
	db2 = tf.reduce_mean(tf.math.minimum(tf.image.psnr(y_pred2, hr2, 255), 100))
	db3 = tf.reduce_mean(tf.math.minimum(tf.image.psnr(y_pred3, hr3, 255), 100))
	db4 = tf.reduce_mean(tf.math.minimum(tf.image.psnr(y_pred4, hr4, 255), 100))
	return cost, db2, db3, db4




def test():
	net.load_model()
	for scale in SCALES:
		print ('scale', scale, validate(scale))
		
def validate(scale):
	dbs = []

	for i in range(len(eval_gt_imgs)):
		hr = eval_gt_imgs[i]
		hr = modular_crop(hr, scale)
		h, w, _ = hr.shape

		lr = resize_img(hr, h//scale, w//scale)
		upcubic = resize_img(lr, h, w)

		inputs = np.asarray([lr], np.float32), np.asarray([upcubic], np.float32), scale
		output = net(inputs)
		y_pred = clip255(unnormalize(output))
		dbs.append(psnr(y_pred, hr))

	return np.mean(dbs)


def train():
	make_dir(LOG_PATH)
	log_file = open(os.path.join(LOG_PATH, '{}_{}.txt'.format(args.train, args.test)), mode='w')

	for epoch in range(1, EPOCH + 1):
		iter_str = '{:03}'.format(epoch)
		print ('\nloop : ', iter_str, file=log_file, flush=True)
		print (datetime.datetime.now(), file=log_file, flush=True)

		log_lists = np.zeros(4)

		for i in range(ITER_PER_EPOCH):
			print (i, '\r', end='')

			lr2, upcubic2, hr2 = next(ds2_it)
			lr3, upcubic3, hr3 = next(ds3_it)
			lr4, upcubic4, hr4 = next(ds4_it)

			input2 = lr2, upcubic2, 2
			input3 = lr3, upcubic3, 3
			input4 = lr4, upcubic4, 4

			next_lr = tf.constant(cosine_decay(ITER_PER_EPOCH * (epoch - 1) + i), dtype=tf.float32)
			cost, db2, db3, db4 = _train_step(next_lr, input2, input3, input4, hr2, hr3, hr4)
			log_lists += np.asarray([cost, db2, db3, db4])



		print (list(log_lists / ITER_PER_EPOCH), file=log_file, flush=True)
		
		if 1:#epoch % 10 == 0:
			net.save_model()
			for scale in SCALES:
				print ('scale', scale, validate(scale), file=log_file, flush=True)



train()
# test()