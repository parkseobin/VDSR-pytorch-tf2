import torch
import torch.nn as nn
from ops import *
from math import sqrt



class Generator(nn.Module):
    def __init__(self, img_feat = 3, n_feats = 64, kernel_size = 3, num_block = 16, act = nn.PReLU(), scale=4):
        super(Generator, self).__init__()
        self.conv01 = conv(in_channel = img_feat, out_channel = n_feats, kernel_size = 9, BN = False, act = act)
        resblocks = [ResBlock(channels = n_feats, kernel_size = 3, act = act) for _ in range(num_block)]
        self.body = nn.Sequential(*resblocks)
        self.conv02 = conv(in_channel = n_feats, out_channel = n_feats, kernel_size = 3, BN = True, act = None)
        if(scale == 4):
            upsample_blocks = [Upsampler(channel = n_feats, kernel_size = 3, scale = 2, act = act) for _ in range(2)]
        else:
            upsample_blocks = [Upsampler(channel = n_feats, kernel_size = 3, scale = scale, act = act)]
        self.tail = nn.Sequential(*upsample_blocks)
        self.last_conv = conv(in_channel = n_feats, out_channel = img_feat, kernel_size = 3, BN = False, act = nn.Tanh())
        

    def forward(self, x):
        
        x = self.conv01(x)
        _skip_connection = x
        
        x = self.body(x)
        x = self.conv02(x)
        feat = x + _skip_connection
        
        x = self.tail(feat)
        x = self.last_conv(x)
        
        return x, feat



class Discriminator(nn.Module):
    def __init__(self, img_feat = 3, n_feats = 64, kernel_size = 3, act = nn.LeakyReLU(inplace = True), num_of_block = 3, patch_size = 96):
        super(Discriminator, self).__init__()
        self.act = act
        
        self.conv01 = conv(in_channel = img_feat, out_channel = n_feats, kernel_size = 3, BN = False, act = self.act)
        self.conv02 = conv(in_channel = n_feats, out_channel = n_feats, kernel_size = 3, BN = False, act = self.act, stride = 2)
        
        body = [discrim_block(in_feats = n_feats * (2 ** i), out_feats = n_feats * (2 ** (i + 1)), kernel_size = 3, act = self.act) for i in range(num_of_block)]    
        self.body = nn.Sequential(*body)
        
        self.linear_size = ((patch_size // (2 ** (num_of_block + 1))) ** 2) * (n_feats * (2 ** num_of_block))
        
        tail = []
        
        tail.append(nn.Linear(self.linear_size, 1024))
        tail.append(self.act)
        tail.append(nn.Linear(1024, 1))
        tail.append(nn.Sigmoid())
        
        self.tail = nn.Sequential(*tail)
        
        
    def forward(self, x):
        
        x = self.conv01(x)
        x = self.conv02(x)
        x = self.body(x)        
        x = x.view(-1, self.linear_size)
        x = self.tail(x)
        
        return x



class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        return self.relu(self.conv(x))



class VDSR(nn.Module):
    '''
    [*] VDSR from https://github.com/twtygqyy/pytorch-vdsr/blob/master/vdsr.py
    '''
    def __init__(self):
        super(VDSR, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)


    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        return out


