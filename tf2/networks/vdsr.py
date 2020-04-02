from .baseNet import BaseNet
from utils import getVariable, conv2d, normalize, unnormalize, clip255
import tensorflow as tf



class VDSR(BaseNet):
	def __init__(self, name=None):
		
		super(VDSR, self).__init__(name=name)

	# @tf.Module.with_name_scope
	def __call__(self, inputs, thetas=None):
		v_idx = [-1]

		if thetas == None:
			thetas = self.weights

		def next_var():
			v_idx[0] += 1
			assert(v_idx[0] < len(thetas))
			return thetas[v_idx[0]]

		lr, upcubic, scale = inputs

		act_fn = tf.nn.relu
		layer = norm_cuibc = normalize(upcubic)

		layer = conv2d(layer, next_var())
		layer = act_fn(layer)
		for _ in range(18):
			layer = conv2d(layer, next_var())
			layer = act_fn(layer)

		layer = conv2d(layer, next_var())
		self.output = layer + norm_cuibc
		return self.output


	def _build(self):
		K = 64
		thetas = []

		thetas.append(getVariable([3, 3, 3, K], 'g0_first'))
		for i in range(18):
			thetas.append(getVariable([3, 3, K, K], 'go_{:02}'.format(i)))
		thetas.append(getVariable([3, 3, K, 3], 'g_last'))
		return thetas