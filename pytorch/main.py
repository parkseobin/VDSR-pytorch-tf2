from mode import *
import argparse
import os



parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true')

parser.add_argument("--LR_path", type=str, default='/home/dataset/DIV2K/DIV2K_train_BICUBIC')
#parser.add_argument("--LR_path", type=str, default='/home/dataset/DIV2K/DIV2K_train_BICUBICx4')
parser.add_argument("--GT_path", type=str, default='/home/dataset/DIV2K/DIV2K_train_HR')
parser.add_argument("--test-LR_path", type=str, default='/home/dataset/DIV2K/DIV2K_valid_BICUBIC')
#parser.add_argument("--test-LR_path", type=str, default='/home/dataset/DIV2K/DIV2K_valid_BICUBICx4')
parser.add_argument("--test-GT_path", type=str, default='/home/dataset/DIV2K/DIV2K_valid_HR')
parser.add_argument('--parameter-save-path', type=str, default='parameters/x2/vdsr.pt')
#parser.add_argument('--parameter-save-path', type=str, default='parameters/from_x4_to_x2/vdsr.pt')
#parser.add_argument('--parameter-save-path', type=str, default='parameters/from_x2_to_x4/vdsr.pt')
parser.add_argument('--parameter-restore-path', type=str, default=None)
#parser.add_argument('--parameter-restore-path', type=str, default='parameters/x4/vdsr.pt')
#parser.add_argument('--parameter-restore-path', type=str, default='parameters/x2/vdsr.pt')
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument('--learning-rate', type=float, default=1e-4)
parser.add_argument("--train_epoch", type=int, default=800)
#parser.add_argument("--scale", type=int, default=2)
parser.add_argument("--scale", type=int, default=2)
parser.add_argument("--patch_size", type=int, default=192)
parser.add_argument("--in_memory", type=str2bool, default=True)
parser.add_argument('--gpu', type=str, default='3')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
train(args)
    

