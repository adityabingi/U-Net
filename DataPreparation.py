import os
import functools

from sklearn.model_selection import train_test_split
import pandas as pd

import tensorflow as tf
import tensorflow.contrib as tfcontrib



def split_data(filepaths):

	img_dir = filepaths.fnTrain
	label_dir = filepaths.fnLabels

	df_train = pd.read_csv(filepaths.fnLabelsCsv)
	ids_train = df_train['img'].map(lambda s: s.split('.')[0])

	x_train_filenames = []
	y_train_filenames = []
	for img_id in ids_train:
		x_train_filenames.append(os.path.join(img_dir, "{}.jpg".format(img_id)))
		y_train_filenames.append(os.path.join(label_dir, "{}_mask.gif".format(img_id)))


	x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = \
					train_test_split(x_train_filenames, y_train_filenames, test_size=0.2, random_state=42)

	return x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames


def _process_pathnames(fname, label_path):

	# We map this function onto each pathname pair  
	img_str = tf.read_file(fname)
	img = tf.image.decode_jpeg(img_str, channels=3)

	label_img_str = tf.read_file(label_path)
	# These are gif images so they return as (num_frames, h, w, c)
	try:
		label_img = tf.image.decode_gif(label_img_str)[0]
	except:
		label_img = tf.image.decode_jpeg(label_img_str)
	# The label image should only have values of 1 or 0, indicating pixel wise
	# object (car) or not (background). We take the first channel only. 
	label_img = label_img[:, :, 0]
	label_img = tf.expand_dims(label_img, axis=-1)
	return img, label_img


def shift_img(output_img, label_img, width_shift_range, height_shift_range):
	"""This fn will perform the horizontal or vertical shift"""
	img_shape = output_img.get_shape().as_list()
	if width_shift_range or height_shift_range:
		if width_shift_range:
			width_shift_range = tf.random_uniform([],
									-width_shift_range * img_shape[1],
									width_shift_range * img_shape[1])
		if height_shift_range:
			height_shift_range = tf.random_uniform([],
									-height_shift_range * img_shape[0],
									height_shift_range * img_shape[0])
	# Translate both 
	output_img = tfcontrib.image.translate(output_img,
	                         [width_shift_range, height_shift_range])
	label_img = tfcontrib.image.translate(label_img,
	                         [width_shift_range, height_shift_range])
	return output_img, label_img


def flip_img(horizontal_flip, tr_img, label_img):
	if horizontal_flip:
		flip_prob = tf.random_uniform([], 0.0, 1.0)
		tr_img, label_img = tf.cond(tf.less(flip_prob, 0.5),
		                            lambda: (tf.image.flip_left_right(tr_img), tf.image.flip_left_right(label_img)),
		                            lambda: (tr_img, label_img))
	return tr_img, label_img


def augment(img,
         label_img,
         resize=None,  # Resize the image to some size e.g. [256, 256]
         scale=1,  # Scale image e.g. 1 / 255.
         hue_delta=0,  # Adjust the hue of an RGB image by random factor
         horizontal_flip=False,  # Random left right flip,
         width_shift_range=0,  # Randomly translate the image horizontally
         height_shift_range=0):  # Randomly translate the image vertically 
	if resize is not None:
	# Resize both images
		label_img = tf.image.resize_images(label_img, resize)
		img = tf.image.resize_images(img, resize)

	if hue_delta:
		img = tf.image.random_hue(img, hue_delta)

	img, label_img = flip_img(horizontal_flip, img, label_img)
	img, label_img = shift_img(img, label_img, width_shift_range, height_shift_range)
	label_img = tf.cast(label_img, tf.float32) * scale
	img = tf.cast(img, tf.float32) * scale 
	return img, label_img


def get_baseline_dataset(filenames, 
                         labels,
                         batch_size,
                         preproc_fn=functools.partial(augment),
                         threads=4, 
                         shuffle=True):           
	num_x = len(filenames)
	# Create a dataset from the filenames and labels
	dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
	# Map our preprocessing function to every element in our dataset, taking
	# advantage of multithreading
	dataset = dataset.map(_process_pathnames, num_parallel_calls=threads)
	if preproc_fn.keywords is not None and 'resize' not in preproc_fn.keywords:
		assert batch_size == 1, "Batching images must be of the same size"

	dataset = dataset.map(preproc_fn, num_parallel_calls=threads)

	if shuffle:
		dataset = dataset.shuffle(num_x)


	# It's necessary to repeat our data for all epochs 
	dataset = dataset.repeat().batch(batch_size)
	return dataset
