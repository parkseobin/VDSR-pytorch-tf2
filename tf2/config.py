import argparse, os
from networks.vdsr import VDSR

# args parser
parser = argparse.ArgumentParser(description='SR args')
parser.add_argument('--gpu', type=str, default='3')
parser.add_argument('--model', type=str, choices=['MSSRNet', 'ESPCN', 'VDSR'], default='VDSR')
parser.add_argument('--train', type=str, choices=['DIV2K', 'BSD432'], default='DIV2K')
parser.add_argument('--test', type=str, choices=['Urban100', 'DIV2K', 'BSD68'], default='Urban100')
args = parser.parse_args()


# model dict
MODEL_DICT = {
	'VDSR' : {
		'model' : VDSR,
		'batch_size' : 32,
		'patch_size' : 48,
		'epoch' : 100
	}
}


# dataset path root
DATA_HOME = '/home/dataset/'

# _291 = os.path.join(DATA_HOME, '291')
BSD432 = os.path.join(DATA_HOME, 'CBSD432')
#DIV2K_TRAIN = os.path.join(DATA_HOME, 'DIV2K_train_HR')
DIV2K_TRAIN = '/home/dataset/Urban100'

SET5 = os.path.join(DATA_HOME, 'Set5')
SET14 = os.path.join(DATA_HOME, 'Set14')


#URBAN100 = '/home/shlee/works/dataset/Urban100'
URBAN100 = '/home/dataset/Urban100'

#DIV2K = '/home/shlee/works/dataset/DIV2KValidx2'
DIV2K = '/home/dataset/Urban100'

#BSD68 = '/home/shlee/works/dataset/BSD68'
BSD68 = '/home/dataset/Urban100'




# dataset dict
DS_DICT = {
	'Set5' : {
		'test' : SET5
	},
	'Set14' : {
		'test' : SET14
	},
	'Urban100' : {
		'test' : URBAN100
	},
	'DIV2K' : {
		'train' : DIV2K_TRAIN,
		'test' : DIV2K
	},
	'BSD68' : {
		'test' : BSD68
	},
	'BSD432' : {
		'train' : BSD432
	}
}