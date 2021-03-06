from train import train, test
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--GT-path', type=str, default='tmp/multiscale_DIV2K')
parser.add_argument('--test-GT-path', type=str, default='/home/dataset/DIV2K/DIV2K_valid_HR')
parser.add_argument('--parameter-save-path', type=str, default='parameters/x4')
parser.add_argument('--parameter-restore-path', type=str, default=None)
parser.add_argument('--parameter-name', type=str, default='vdsr.pt')
parser.add_argument('--num-workers', type=int, default=16)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--learning-rate', type=float, default=1e-1)
parser.add_argument('--train-iteration', type=int, default=50000)
parser.add_argument('--log-step', type=int, default=50)
parser.add_argument('--validation-step', type=int, default=200)
parser.add_argument('--scale', type=int, default=4)
parser.add_argument('--patch-size', type=int, default=128)
parser.add_argument('--lazy-load', action='store_true', default=False)
parser.add_argument('--rgb', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
print('[*] Using GPU: {}'.format(args.gpu), flush=True)


test(args) if(args.test) else train(args)

