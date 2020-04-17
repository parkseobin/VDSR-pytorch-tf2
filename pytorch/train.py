import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import *
from models import VDSR
import numpy as np
from PIL import Image
from skimage.color import rgb2ycbcr
from skimage.measure import compare_psnr
from datetime import datetime


'''
TODO:   
        - Learning rate decay (https://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate)
            > ReduceLROnPlateau reduces lr too early?? => learning rate step on epoch end, not gradient end
            > Seems not bad..

        - Solved rgb2ycbcr code mystery
            > input is 0~1 but output is 0~255
            > reason: https://github.com/scikit-image/scikit-image/issues/2573

        - LSGAN: http://openaccess.thecvf.com/content_ICCV_2017/papers/Mao_Least_Squares_Generative_ICCV_2017_paper.pdf

TODO: in train.py
    [] saving parameter file without param folder pre-made
    [] break loop when lr is below some value
    [] change lr on plateau scale...

    [] ref: https://github.com/yulunzhang/RCAN
'''


def train(args):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    transform = [crop(args.scale, args.patch_size), augmentation()]
    '''
    dataset = SRDataset(GT_path=args.GT_path, LR_path=args.LR_path, lazy_load=args.lazy_load, 
                    transform=transform, dataset_size=args.train_iteration*args.batch_size)
    '''
    dataset = SRDatasetOnlyGT(GT_path=args.GT_path, lazy_load=args.lazy_load, LR_transform=1./args.scale,
                    transform=transform, dataset_size=args.train_iteration*args.batch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    sr_network = VDSR()
    if not args.parameter_restore_path is None:
        sr_network.load_state_dict(torch.load(args.parameter_restore_path))
        print('[*] pre-trained model is loaded from {}'.format(args.parameter_restore_path))
    if not args.parameter_save_path is None:
        if(not os.path.exists(args.parameter_save_path)):
            os.makedirs(args.parameter_save_path)
    sr_network = sr_network.to(device)
    sr_network.train()

    l2_loss = nn.MSELoss()
    optimizer = optim.Adam(sr_network.parameters(), lr = args.learning_rate)
    learning_rate_scheduler = ReduceLROnPlateau(optimizer, 'min')

    best_psnr = 0
    loss_list = []
    print('[*] Start training\n', flush=True)
    for i, train_data in enumerate(loader):
        gt = train_data['GT'].to(device)
        low_res = train_data['LR'].to(device)

        output = sr_network(low_res)
        loss = l2_loss(gt, output)
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if((i+1) % args.log_step == 0):
            loss_mean = np.mean(loss_list)
            learning_rate_scheduler.step(loss_mean)
            loss_list = []
            now = datetime.now()
            print("[{}]".format(now.strftime('%Y-%m-%d %H:%M:%S')), flush=True)
            print('>> [{}/{}] \t loss: {:.6f} (lr: {:.2e})\n'.format(
                i+1, args.train_iteration, loss_mean, optimizer.param_groups[0]['lr']), flush=True)

        if((i+1) % args.validation_step == 0):
            val_psnr = validation(args, device, sr_network)
            print('>> avg psnr : {:.4f}dB'.format(val_psnr), flush=True)
            if(val_psnr > best_psnr):
                best_psnr = val_psnr
                save_path = os.path.join(args.parameter_save_path, args.parameter_name)
                torch.save(sr_network.state_dict(), save_path)
                print('>> parameter saved in {}'.format(save_path))
            print() 
        


def validation(args, device, sr_network):
    dataset = SRDataset(GT_path=args.test_GT_path, LR_path=args.test_LR_path)
    loader = DataLoader(dataset, num_workers=args.num_workers)
    sr_network.eval()
    
    psnr_list = []
    with torch.no_grad():
        for i, te_data in enumerate(loader):
            gt = te_data['GT'].to(device)
            low_res = te_data['LR'].to(device)

            output = sr_network(low_res)
            output = output[0].cpu().numpy()
            output = np.clip(output, 0., 1.0)
            gt = gt[0].cpu().numpy()
            output = output.transpose(1,2,0)
            gt = gt.transpose(1,2,0)

            y_output = rgb2ycbcr(output)[args.scale:-args.scale, args.scale:-args.scale, :1]
            y_gt = rgb2ycbcr(gt)[args.scale:-args.scale, args.scale:-args.scale, :1]

            psnr = compare_psnr(y_output, y_gt, data_range=255)
            psnr_list.append(psnr)

    sr_network.train()
    return np.mean(psnr_list)



