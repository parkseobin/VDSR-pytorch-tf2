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

        - Make validation efficient
'''


def train(args):
    
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([crop(args.scale, args.patch_size), augmentation()])
    dataset = mydata(GT_path=args.GT_path, LR_path=args.LR_path, in_memory=args.in_memory, 
                    transform=transform, dataset_size=args.train_iteration*args.batch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    sr_network = VDSR()
    if not args.parameter_restore_path is None:
        sr_network.load_state_dict(torch.load(args.parameter_restore_path))
        print('pre-trained model is loaded from {}'.format(args.parameter_restore_path))
    sr_network = sr_network.to(device)
    sr_network.train()

    l2_loss = nn.MSELoss()
    optimizer = optim.Adam(sr_network.parameters(), lr = args.learning_rate)
    learning_rate_scheduler = ReduceLROnPlateau(optimizer, 'min')

    print('[*] Start training\n', flush=True)
    
    #### Train using L2_loss
    best_psnr = 0
    loss_list = []
    for i, train_data in enumerate(loader):
        '''
        [*] Iterates (data length) / (batch size) times
        '''
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
            print('>> iteration {} \t loss: {:.6f} (lr: {:.2e})\n'.format(
                i+1, loss_mean, optimizer.param_groups[0]['lr']), flush=True)

        if((i+1) % args.validation_step == 0):
            val_psnr = validation(args, device, sr_network)
            print('>> avg psnr : {:.4f}dB'.format(val_psnr), flush=True)
            if(val_psnr > best_psnr):
                best_psnr = val_psnr
                torch.save(sr_network.state_dict(), args.parameter_save_path)
                print('>> parameter saved in {}'.format(args.parameter_save_path))
            print() 
        


def validation(args, device, sr_network):
    dataset = mydata(GT_path=args.test_GT_path, LR_path=args.test_LR_path, in_memory=False, transform=None)
    loader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = args.num_workers)
    sr_network.eval()
    
    psnr_list = []
    with torch.no_grad():
        for i, te_data in enumerate(loader):
            gt = te_data['GT'].to(device)
            low_res = te_data['LR'].to(device)

            #bs, c, h, w = lr.size()
            #gt = gt[:, :, : h * args.scale, : w *args.scale]

            output = sr_network(low_res)

            output = output[0].cpu().numpy()
            output = np.clip(output, 0., 1.0)
            gt = gt[0].cpu().numpy()

            output = output.transpose(1,2,0)
            gt = gt.transpose(1,2,0)

            #output = 255*output
            #gt = 255*gt
            #! >>> rgb2ycbcr 0~1 => 0~255 ????
            y_output = rgb2ycbcr(output)[args.scale:-args.scale, args.scale:-args.scale, :1]
            y_gt = rgb2ycbcr(gt)[args.scale:-args.scale, args.scale:-args.scale, :1]

            psnr = compare_psnr(y_output / 255.0, y_gt / 255.0, data_range = 1.0)
            psnr_list.append(psnr)
            #f.write('psnr : %04f \n' % psnr)

            #result = Image.fromarray((output * 255.0).astype(np.uint8))
            #result.save('./result/res_%04d.png'%i)
    
    sr_network.train()

    return np.mean(psnr_list)



