import argparse, os
from networks.vdsr import VDSR

# args parser
parser = argparse.ArgumentParser(description='SR args')
parser.add_argument('--gpu', type=str, default='-1')
parser.add_argument('--model', type=str, choices=['MSSRNet', 'ESPCN', 'VDSR'], required=True)
parser.add_argument('--train', type=str, choices=['DIV2K', 'BSD432'], required=True)
parser.add_argument('--test', type=str, choices=['Urban100', 'DIV2K', 'BSD68'], required=True)
args = parser.parse_args()


# model dict
MODEL_DICT = {
	'MSSRNet' : {
		'model' : MSSRNet,
		'batch_size' : 16,
		'patch_size' : 12*10,
		'epoch' : 1000
	},
	'ESPCN' : {
		'model' : ESPCN,
		'batch_size' : 32,
		'patch_size' : 48,
		'epoch' : 100
	},
	'VDSR' : {
		'model' : VDSR,
		'batch_size' : 32,
		'patch_size' : 48,
		'epoch' : 100
	}
}


# dataset path root
DATA_HOME = '/home/shlee/works/dataset/'

# _291 = os.path.join(DATA_HOME, '291')
BSD432 = os.path.join(DATA_HOME, 'CBSD432')
DIV2K_TRAIN = os.path.join(DATA_HOME, 'DIV2K_train_HR')

SET5 = os.path.join(DATA_HOME, 'Set5')
SET14 = os.path.join(DATA_HOME, 'Set14')


URBAN100 = '/home/shlee/works/dataset/Urban100'

DIV2K = '/home/shlee/works/dataset/DIV2KValidx2'

BSD68 = '/home/shlee/works/dataset/BSD68'




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