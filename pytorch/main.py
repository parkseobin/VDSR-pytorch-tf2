from train import train
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--LR-path', type=str, default='/home/dataset/DIV2K/DIV2K_train_BICUBIC')
parser.add_argument('--GT-path', type=str, default='/home/dataset/DIV2K/DIV2K_train_HR')
parser.add_argument('--test-LR-path', type=str, default='/home/dataset/DIV2K/DIV2K_valid_BICUBIC')
parser.add_argument('--test-GT-path', type=str, default='/home/dataset/DIV2K/DIV2K_valid_HR')
parser.add_argument('--parameter-save-path', type=str, default='parameters/x2/vdsr.pt')
parser.add_argument('--parameter-restore-path', type=str, default=None)
parser.add_argument('--num-workers', type=int, default=16)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--learning-rate', type=float, default=1e-3)
parser.add_argument('--train-iteration', type=int, default=20000)
parser.add_argument('--log-step', type=int, default=50)
parser.add_argument('--validation-step', type=int, default=200)
parser.add_argument('--scale', type=int, default=2)
parser.add_argument('--patch-size', type=int, default=48)
parser.add_argument('--in-memory', action='store_true', default=False)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

train(args)
    

