import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import *
from models import VDSR
import numpy as np
from PIL import Image
from skimage.color import rgb2ycbcr
from skimage.measure import compare_psnr
from datetime import datetime





def train(args):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    # Set dataset
    transform = [crop(args.scale, args.patch_size), augmentation()]
    dataset = SRDatasetOnlyGT(GT_path=args.GT_path, lazy_load=args.lazy_load, LR_transform=1./args.scale,
                    transform=transform, dataset_size=args.train_iteration*args.batch_size, rgb=args.rgb)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # Set network
    sr_network = VDSR(rgb=args.rgb)
    if(not args.parameter_restore_path is None):
        sr_network.load_state_dict(torch.load(args.parameter_restore_path))
        print('[*] pre-trained model is loaded from {}'.format(args.parameter_restore_path))
    if(not args.parameter_save_path is None):
        if(not os.path.exists(args.parameter_save_path)):
            os.makedirs(args.parameter_save_path)
    sr_network = sr_network.to(device)
    sr_network.train()

    # Set optimizer
    l2_loss = nn.MSELoss(size_average=False)
    optimizer = optim.SGD(sr_network.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    learning_rate_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5)

    # Log hyperparameters
    print('[*] \tlearning rate: {:.2e}\n\tbatch size: {}\n\tpatch size: {}\n\ttrain set: {}'.format(
        args.learning_rate, args.batch_size, 
        args.patch_size, args.GT_path
    ), flush=True)

    best_psnr = 0
    loss_list = []
    print('\n[*] Start training\n', flush=True)
    for i, train_data in enumerate(loader):
        gt = train_data['GT'].to(device)
        low_res = train_data['LR'].to(device)

        output = sr_network(low_res)
        loss = l2_loss(gt, output)
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(sr_network.parameters(), 0.1)
        optimizer.step()

        # Log step
        if((i+1) % args.log_step == 0):
            loss_mean = np.mean(loss_list)
            loss_list = []
            current_learning_rate = optimizer.param_groups[0]['lr']

            now = datetime.now()
            print('[{}]'.format(now.strftime('%Y-%m-%d %H:%M:%S')), flush=True)
            print('>> [{}/{}] \t loss: {:.6f} (lr: {:.2e})\n'.format(
                i+1, args.train_iteration, loss_mean, current_learning_rate), flush=True)
            if(current_learning_rate < 1e-6):
                print('[*] Train end due to small learning rate')
                break

        # Validation step
        if((i+1) % args.validation_step == 0):
            val_psnr = validation(args, device, sr_network)
            print('>> avg psnr : {:.4f}dB'.format(val_psnr), flush=True)
            if(val_psnr > best_psnr):
                best_psnr = val_psnr
                save_path = os.path.join(args.parameter_save_path, args.parameter_name)
                torch.save(sr_network.state_dict(), save_path)
                print('>> parameter saved in {}'.format(save_path), flush=True)
            save_path = os.path.join(args.parameter_save_path, 'last_{}'.format(args.parameter_name))
            torch.save(sr_network.state_dict(), save_path)
            learning_rate_scheduler.step(val_psnr)
            print() 
        


def validation(args, device, sr_network):
    dataset = SRDatasetOnlyGT(GT_path=args.test_GT_path, LR_transform=1./args.scale, lazy_load=True, rgb=args.rgb)
    loader = DataLoader(dataset, num_workers=args.num_workers)
    sr_network.eval()
    
    psnr_list = []
    with torch.no_grad():
        for i, test_data in enumerate(loader):
            gt = test_data['GT'].to(device)
            low_res = test_data['LR'].to(device)

            output = sr_network(low_res)
            output = output[0].cpu().numpy()
            output = np.clip(output, 0., 1.0)
            gt = gt[0].cpu().numpy()
            output = output.transpose(1,2,0)
            gt = gt.transpose(1,2,0)
            if(args.rgb):
                y_output = rgb2ycbcr(output)[args.scale:-args.scale, args.scale:-args.scale, :1]
                y_gt = rgb2ycbcr(gt)[args.scale:-args.scale, args.scale:-args.scale, :1]
            else:
                y_output = output[args.scale:-args.scale, args.scale:-args.scale, :] * 255
                y_gt = gt[args.scale:-args.scale, args.scale:-args.scale, :] * 255

            psnr = compare_psnr(y_output, y_gt, data_range=255)
            psnr_list.append(psnr)

    sr_network.train()
    return np.mean(psnr_list)



def test(args):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    # Set network
    sr_network = VDSR(args.rgb)
    if not args.parameter_restore_path is None:
        sr_network.load_state_dict(torch.load(args.parameter_restore_path))
        print('[*] pre-trained model is loaded from {}'.format(args.parameter_restore_path))
    else:
        print('[*] Need to set restore parameter path!')
        exit()
    sr_network = sr_network.to(device)

    psnr_out = validation(args, device, sr_network)
    print('\n[*] PSNR of dataset in {}: {:.3f}dB'.format(args.test_GT_path, psnr_out))


