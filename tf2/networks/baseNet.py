from utils import make_dir, file_dir_path

import os
import abc
import tensorflow as tf


class BaseNet(tf.Module, metaclass=abc.ABCMeta):
	dir_path = file_dir_path(__file__)

	def __init__(self, name=None):
		super(BaseNet, self).__init__(name=name)

		with self.name_scope:
			self.weights = self._build()

		self.checkpoint = tf.train.Checkpoint(model=self)


	def save_model(self, dirname=None, filename='ckpt'):
		if dirname == None:
			dirname = self.name

		checkpoint_prefix = os.path.join(self.dir_path, 'checkpoints', dirname, filename)
		full_save_path = self.checkpoint.save(file_prefix=checkpoint_prefix)
		print ('save...', full_save_path)

		
	def load_model(self, dirname=None, filename=None):
		if dirname == None:
			dirname = self.name

		if filename == None:
			checkpoint_path = tf.train.latest_checkpoint(os.path.join(self.dir_path, 'checkpoints', dirname))
		else:
			checkpoint_path = os.path.join(self.dir_path, 'checkpoints', dirname, filename)

		status = self.checkpoint.restore(checkpoint_path)
		status.assert_consumed()
		# status.assert_existing_objects_matched()
		print ('load...', checkpoint_path)
		

	@abc.abstractmethod
	def _build(self):
		pass