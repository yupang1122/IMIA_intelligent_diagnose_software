import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf 
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow.python.util.deprecation as deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

class restored_model(object):

	def __init__(self, model_name, model_folder):
		self.graph=tf.Graph()
		self.sess=tf.compat.v1.Session(graph=self.graph)
		print('Read model: ', model_name)

		with self.graph.as_default():
			self.model_saver=tf.train.import_meta_graph(model_name)
			self.model_saver.restore(self.sess, tf.train.latest_checkpoint(model_folder+'/.'))
			self.graph=self.graph
			self.sample_in=self.graph.get_tensor_by_name('sample:0')
			self.c_mask_out=self.graph.get_tensor_by_name('c_mask:0')

	def run_sess(self, patches):
		feed_dict={self.sample_in: patches}
		generated_mask=self.sess.run([self.c_mask_out], feed_dict)
		return generated_mask

	def close_sess(self):
		self.sess.close()