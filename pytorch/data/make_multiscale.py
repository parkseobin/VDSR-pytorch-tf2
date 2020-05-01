import os
import sys
sys.path.append('..')
from PIL import Image
from imresize import ImresizeWrapper, imresize_pil



target_path = '/home/dataset/DIV2K/DIV2K_train_HR'
save_path = 'multiscale_DIV2K'
target_scales = [0.75, 0.5]
img_name_list = sorted(os.listdir(target_path))
img_pil_list = [Image.open(os.path.join(target_path, gt)) for gt in img_name_list]


if(not os.path.exists(save_path)): os.makedirs(save_path)

for img, name in zip(img_pil_list, img_name_list):
    img.save(os.path.join(save_path, name))

for scale in target_scales:
    for img, name in zip(img_pil_list, img_name_list):
        resized_img = imresize_pil(img, scale_factor=scale)
        img_name = '{}_{}'.format(scale, name)
        resized_img.save(os.path.join(save_path, img_name))
        print('>> saved {}'.format(img_name), flush=True)

print('\n[*] END')
