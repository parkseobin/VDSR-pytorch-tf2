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
from time import time


'''
TODO:   
        - Learning rate decay (https://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate)
            > ReduceLROnPlateau reduces lr too early?? => learning rate step on epoch end, not gradient end
            > Seems not bad..

        - Solve slow dataset problem
        - Make validation efficient
'''







def train(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform  = transforms.Compose([crop(args.scale, args.patch_size), augmentation()])
    dataset = mydata(GT_path=args.GT_path, LR_path=args.LR_path, in_memory=args.in_memory, 
                    transform=transform, scale_dataset=args.scale_dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    sr_network = VDSR()
    if not args.parameter_restore_path is None:
        sr_network.load_state_dict(torch.load(args.parameter_restore_path))
        print('pre-trained model is loaded from {}'.format(args.parameter_restore_path))
    sr_network = sr_network.to(device)
    sr_network.train()

    l2_loss = nn.MSELoss()
    optimizer = optim.Adam(sr_network.parameters(), lr = args.learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min')

    print('[*] Start training\n', flush=True)
    
    #### Train using L2_loss
    best_psnr = 0
    train_epoch = 0
    last_tick = time()
    while train_epoch < args.train_epoch:
        epoch_losses = []
        for i, tr_data in enumerate(loader):
            '''
            [*] Iterates (data length) / (batch size) times
            '''
            if(i == 0):
                print('***> dataloader delay: {:.4f}'.format(time()-last_tick))

            _last_tick = time()
            print('>> start of iteration {} (time interval: {:.4f})'.format(i, _last_tick-last_tick), '\r', end='')
            last_tick = _last_tick

            gt = tr_data['GT'].to(device)
            lr = tr_data['LR'].to(device)

            output = sr_network(lr)
            loss = l2_loss(gt, output)
            epoch_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _last_tick = time()
            print('>> end of iteration {} (time interval: {:.4f})'.format(i, _last_tick-last_tick), '\r', end='')
            last_tick = _last_tick

        loss_mean = np.mean(epoch_losses)
        lr_scheduler.step(loss_mean)
        train_epoch += 1
        now = datetime.now()
        print('\n\n')
        print("[{}]".format(now.strftime('%Y-%m-%d %H:%M:%S')), flush=True)
        print('>> epoch {} \t loss: {:.6f} (lr: {:.2e})\n'.format(train_epoch, loss_mean, optimizer.param_groups[0]['lr']), flush=True)
        last_tick = time()

        if(train_epoch % 20 == 0):
            torch.save(sr_network.state_dict(), args.parameter_save_path)
            validation(args)
        


def validation(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = mydata(GT_path=args.test_GT_path, LR_path=args.test_LR_path, in_memory=False, transform=None)
    loader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = args.num_workers)
    
    sr_network = VDSR()
    sr_network.load_state_dict(torch.load(args.parameter_save_path))
    sr_network = sr_network.to(device)
    sr_network.eval()
    
    psnr_list = []
    with torch.no_grad():
        for i, te_data in enumerate(loader):
            gt = te_data['GT'].to(device)
            lr = te_data['LR'].to(device)

            bs, c, h, w = lr.size()
            gt = gt[:, :, : h * args.scale, : w *args.scale]

            output = sr_network(lr)

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

        print('>> avg psnr : {:.4f}dB\n'.format(np.mean(psnr_list)), flush=True)



