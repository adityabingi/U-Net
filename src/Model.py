from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

_BATCH_NORM_DECAY = 0.99
_BATCH_NORM_EPSILON = 0.001
_SUMMARIES_DIR = '../summary/'


def batch_norm(inputs, training, data_format):

	"""Performs a batch normalization using a standard set of parameters."""
	# We set fused=True for a significant performance boost. See
	# https://www.tensorflow.org/performance/performance_guide#common_fused_ops


	return tf.layers.batch_normalization(
	  inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
	  momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
	  scale=True, training=training, fused=True)


def conv_layer(inputs, filters, kernel_size, strides, data_format, activation=None):

	"""convolutional layer"""

	return tf.layers.conv2d(
	  inputs  = inputs, filters=filters, kernel_size=kernel_size, strides=strides,
	  padding ='SAME', use_bias=True,
	  kernel_initializer = tf.variance_scaling_initializer(),
	  activation= activation,
	  data_format = data_format)



def conv_block(inputs, filters, training, data_format):

	"""conv_block for both encoder and decoder"""

	inputs = conv_layer(inputs,filters=filters, kernel_size = 3, strides=(1,1), data_format=data_format)

	inputs = batch_norm(inputs, training, data_format)

	inputs = tf.nn.relu(inputs)

	inputs = conv_layer(inputs, filters=filters, kernel_size=3,  strides=(1,1), data_format=data_format)

	inputs = batch_norm(inputs, training, data_format)

	inputs = tf.nn.relu(inputs)

	return inputs

def downsample(inputs, data_format):

	""" downsampling (maxpooling) layer for encoder"""
	return tf.layers.max_pooling2d(inputs, pool_size = 2, strides = (2,2), padding='valid',
    data_format=data_format)

def upsample(inputs, filters, data_format):

	""" upsampling (convolution transpose) layer for decoder"""

	return tf.layers.conv2d_transpose(inputs, filters, 2,
    strides = (2,2),
    padding = 'same',
    data_format = data_format,
    use_bias = True)


def encoder_block(inputs, filters, training, data_format):

	""" single encoder block"""

	encoder = conv_block(inputs, filters, training, data_format)

	encoder_pool = downsample(encoder, data_format)

	return encoder_pool, encoder


def decoder_block(inputs, concat_tensor, filters, training, data_format):

	""" single decoder block"""

	decoder = upsample(inputs, filters, data_format)

	decoder = tf.concat([concat_tensor, decoder], axis = -1)

	decoder = batch_norm(decoder, training, data_format)

	decoder = tf.nn.relu(decoder)

	decoder = conv_block(decoder, filters, training, data_format)

	return decoder


# class for U_net model
class Model:

	"minimalistic model for U-Net"

	def __init__(self, val_dataset, train_dataset=None, mustRestore = False, prefetch_batch_buffer=1, data_format = 'channels_last'):
		self.mustRestore = mustRestore
		self.data_format = data_format

		if(train_dataset):
			self.train_dataset = train_dataset.prefetch(prefetch_batch_buffer)

		self.val_dataset = val_dataset.prefetch(prefetch_batch_buffer)
		
		self.training  = tf.placeholder(tf.bool, shape=[])
		

		self.snapID = 0 
		self.no_of_filters = [32, 64, 128, 256, 512, 1024]


		# A reinitializable iterator is defined by its structure. We could use the
		# `output_types` and `output_shapes` properties of either `training_dataset`
		# or `validation_dataset` here, because they are compatible.

		self.handle = tf.placeholder(tf.string, shape=[])
		iterator = tf.data.Iterator.from_string_handle(self.handle, self.val_dataset.output_types,\
	 				self.val_dataset.output_shapes)

		next_element = iterator.get_next()

		self.imgBatch, self.labels = next_element


		# You can use feedable iterators with a variety of different kinds of iterator
		# (such as one-shot and initializable iterators).


		if(train_dataset):
			self.training_iterator = self.train_dataset.make_one_shot_iterator()

		self.validation_iterator = self.val_dataset.make_initializable_iterator()


		self.preds = self.u_net_graph(self.imgBatch)

		self.loss = self.loss_function()

		tf.summary.scalar('loss', self.loss)


		#self.correct_pred = tf.equal(tf.argmax(self.preds, -1), tf.argmax(self.labels, -1))
		#self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

		#tf.summary.scalar('accuarcy', self.accuracy)

		self.learningRate = tf.placeholder(tf.float32, shape=[])

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.optimizer = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss)

		self.merged = tf.summary.merge_all()

		(self.sess, self.saver) = self.setupTF()

		self.train_writer = tf.summary.FileWriter(_SUMMARIES_DIR + '/train',
                                      self.sess.graph)
		self.test_writer = tf.summary.FileWriter(_SUMMARIES_DIR + '/test')



	def u_net_graph(self, inputs):

		""" tf graph for U_Net"""


		with tf.variable_scope('U_net_model'):


			print("Tensor into U_Net")
			print(inputs.shape)

			encoder1_pool, encoder1 = encoder_block(inputs, self.no_of_filters[0], self.training, self.data_format)

			print("After applying 1st encoder and downsampling")
			print(encoder1_pool.shape)

			encoder2_pool, encoder2 = encoder_block(encoder1_pool,self.no_of_filters[1], self.training, self.data_format)

			print("After applying 2nd encoder and downsampling")
			print(encoder2_pool.shape)

			encoder3_pool, encoder3 = encoder_block(encoder2_pool,self.no_of_filters[2], self.training, self.data_format)

			print("After applying 3rd encoder and downsampling")
			print(encoder3_pool.shape)

			encoder4_pool, encoder4 = encoder_block(encoder3_pool,self.no_of_filters[3], self.training, self.data_format)

			print("After applying 4th encoder and downsampling")
			print(encoder4_pool.shape)


			encoder5_pool, encoder5 = encoder_block(encoder4_pool,self.no_of_filters[4], self.training, self.data_format)

			print("After applying 5th encoder and downsampling")
			print(encoder5_pool.shape)


			center = conv_block(encoder5_pool, self.no_of_filters[5], self.training, self.data_format)

			print("After center convolutional block")
			print(center.shape)


			decoder5 = decoder_block(center, encoder5, self.no_of_filters[4], self.training, self.data_format )

			print("After applying 1st decoder and upsampling")
			print(decoder5.shape)

			decoder4 = decoder_block(decoder5, encoder4, self.no_of_filters[3], self.training, self.data_format)

			print("After applying 2nd decoder and upsampling")
			print(decoder4.shape)

			decoder3 = decoder_block(decoder4, encoder3, self.no_of_filters[2], self.training, self.data_format)

			print("After applying 3rd decoder and upsampling")
			print(decoder3.shape)

			decoder2 = decoder_block(decoder3, encoder2, self.no_of_filters[1], self.training, self.data_format)

			print("After applying 4th decoder and upsampling")
			print(decoder2.shape)

			decoder1 = decoder_block(decoder2, encoder1, self.no_of_filters[0], self.training, self.data_format)

			print("After applying 5th decoder and upsampling")
			print(decoder1.shape)

			outputs = conv_layer(decoder1, 1, 1, (1,1), self.data_format, activation='sigmoid')

			print(outputs.shape)

			return outputs


	def loss_function(self):

		""" dice loss function and combined dice & binary cross entropy loss function;
		    any one of them can be used"""


		def dice_coeff(y_true, y_pred):
			smooth = 1.
			# Flatten
			y_true_f = tf.reshape(y_true, [-1])
			y_pred_f = tf.reshape(y_pred, [-1])
			intersection = tf.reduce_sum(y_true_f * y_pred_f)
			score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
			return score

		def dice_loss(y_true, y_pred):
			loss = 1 - dice_coeff(y_true, y_pred)
			return loss

		def bce_dice_loss(y_true, y_pred):

			loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
			return loss

		return dice_loss(self.labels, self.preds)


	def setupTF(self):

		"""initialize TF"""
		
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		sess=tf.Session(config = config) # TF session

		saver = tf.train.Saver(max_to_keep=3) # saver saves model to file
		modelDir = '../model/'
		latestSnapshot = tf.train.latest_checkpoint(modelDir) # is there a saved model?

		# if model must be restored (for inference), there must be a snapshot
		if self.mustRestore and not latestSnapshot:
			raise Exception('No saved model found in: ' + modelDir)

		# load saved model if available
		if self.mustRestore and latestSnapshot:
			print('Init with stored values from ' + latestSnapshot)
			saver.restore(sess, latestSnapshot)
		else:
			print('Init with new values')
			sess.run(tf.global_variables_initializer())

		return (sess,saver)


	def train(self, train_batches_per_epoch, no_of_val_batches, FilePaths):

		""" training loop for model"""

		epoch = 0
		best_error_rate = float('inf')
		batchesTrained = 0
		noImprovementSince = 0
		earlyStopping = 4


		training_handle = self.sess.run(self.training_iterator.string_handle())
		          
		validation_handle = self.sess.run(self.validation_iterator.string_handle())

		
		while True:

			epoch+=1

			for i in range(train_batches_per_epoch):

				print("Epoch :", epoch)
				print("Training")
				print("_"*20)

				rate = 0.001 if epoch < 8  else 0.0001

				_, summary,lossval = self.sess.run([self.optimizer, self.merged, self.loss], \
					feed_dict={self.learningRate : rate, 
					self.training : True, self.handle: training_handle})

				self.train_writer.add_summary(summary, (i + (train_batches_per_epoch *(epoch-1))))

				print('Batch:', i,'/', train_batches_per_epoch, 'Loss:', lossval)

				batchesTrained +=1


			mean_dice_loss = 0
			self.sess.run(self.validation_iterator.initializer)

			for i in range(no_of_val_batches):


				print("Validating")
				print("_"*20)

				summary, lossval = self.sess.run([self.merged, self.loss],\
				 feed_dict={ self.training : False, 
					self.handle: validation_handle})

				self.test_writer.add_summary(summary, train_batches_per_epoch*epoch )

				print('Batch:', i,'/', no_of_val_batches, 'Loss:', lossval)

				mean_dice_loss+= lossval

			mean_dice_loss = mean_dice_loss/no_of_val_batches

			print('Validation error rate after %d epochs: %f%%' % (epoch, mean_dice_loss*100.0))

			if(mean_dice_loss<best_error_rate):

				print("Error rate improved")


				best_error_rate = mean_dice_loss

				self.save()

				with open(FilePaths.fnAccuracy, 'w') as file:
					file.write('Validation error rate of saved model: %f%%' % (best_error_rate*100.0))

			else:

				print('Error rate not improved')
				noImprovementSince += 1

				# stop training if no more improvement in the last x epochs
			if noImprovementSince >= earlyStopping:
				print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
				break



	"""def evaluate(self):


		validation_handle = sess.run(validation_iterator.string_handle())

		mean_dice_loss = 0

		while True:

			try:
				loss = self.sess.run([self.loss],feed_dict={self.training : False,\
				 self.handle: validation_handle})

				mean_dice_loss+=loss

			except tf.errors.OutOfRangeError:
				break

		return 1-mean_dice_loss"""

	def infer(self):

		""" inference function for single image"""

		validation_handle = self.sess.run(self.validation_iterator.string_handle())
		self.sess.run(self.validation_iterator.initializer)
		while True:

			try:
				preds = self.sess.run([self.preds],feed_dict={ self.training : False, \
					self.handle: validation_handle})
				print("predicion of image successful")
				print(len(preds), len(preds[0]), len(preds[0][0]), len(preds[0][0][0]))

				break

			except tf.errors.OutOfRangeError:
				break
		#preds = preds.eval()
		return np.array(preds[0])

	def save(self):
		"""save model to file"""
		self.snapID += 1
		self.saver.save(self.sess, '../model/snapshot', global_step=self.snapID)

