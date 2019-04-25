from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import functools
import argparse
import os
import cv2

from DataPreparation import get_baseline_dataset, split_data, augment
from Model import Model


_IMG_SHAPE = (512, 512, 3)
_BATCH_SIZE = 1


class FilePaths:
	fnAccuracy = '../model/accuracy.txt'
	fnTrain = '../data/train/'
	fnLabels = '../data/train_masks/'
	fnLabelsCsv ='../data/train_masks.csv'
	fnInfer = '../data/test/'
	fnResults ='../results/'


def preprocess_function(train):

	if(train):
		cfg = {
		'resize': [_IMG_SHAPE[0], _IMG_SHAPE[1]],
		'scale': 1 / 255.,
		'hue_delta': 0.1,
		'horizontal_flip': True,
		'width_shift_range': 0.1,
		'height_shift_range': 0.1
		}
	else:
		cfg = {
		'resize': [_IMG_SHAPE[0], _IMG_SHAPE[1]],
		'scale': 1 / 255.
		}

	preprocessing_fn = functools.partial(augment, **cfg)

	return preprocessing_fn

# Helper function to write u_net prediction to an image

def preds_to_img(pred, actual_img, fname):

	scale = 255.
	pred = np.reshape(pred,(_IMG_SHAPE[0], _IMG_SHAPE[1]))
	pred = pred[:,:]*scale
	#pred = pred.astype(int)
	pred = np.reshape(pred,(_IMG_SHAPE[0],_IMG_SHAPE[1],1))

	cv2.imwrite(os.path.join(FilePaths.fnResults, "{}.jpg".format(fname)), actual_img)


	cv2.imwrite(os.path.join(FilePaths.fnResults, "{}_result.jpg".format(fname)), pred)


def main():
	print("Inside main")
	
	# optional command line args
	parser = argparse.ArgumentParser()
	parser.add_argument("--train", help="train the NN", action="store_true")

	#parser.add_argument("--validate", help="validate the NN", action="store_true")
	
	parser.add_argument("--predict",nargs=1)
	args = parser.parse_args()


	if args.train:
	# load training data, create TF model

		x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = split_data(FilePaths)

		train_batches_per_epoch = int(len(x_train_filenames)/_BATCH_SIZE) + 1

		no_of_val_batches = int(len(x_val_filenames)/_BATCH_SIZE) + 1

		train_ds = get_baseline_dataset(x_train_filenames,
			y_train_filenames,
			batch_size=_BATCH_SIZE,
			preproc_fn=preprocess_function(train=True),
			)

		val_ds = get_baseline_dataset(x_val_filenames,
			y_val_filenames,
			batch_size=_BATCH_SIZE,
			preproc_fn= preprocess_function(train=False),
			)


		model = Model(val_dataset =val_ds, train_dataset=train_ds, mustRestore = False)

		model.train(train_batches_per_epoch, no_of_val_batches, FilePaths)

	#elif args.validate:

		#model = Model(val_dataset =val_ds, mustRestore = False)
		#model.validate()

	# infer on test image
	elif args.predict:

	# We pass test_img as dummy label to maintain dataset structure

		x_val_filenames, y_val_filenames = [args.predict[0]]*32, [args.predict[0]]*32

		val_ds = get_baseline_dataset(x_val_filenames,
			y_val_filenames, 
			batch_size=_BATCH_SIZE,
			preproc_fn= preprocess_function(train=False),
			threads=1)

		print(open(FilePaths.fnAccuracy).read())

		model = Model(val_dataset =val_ds, mustRestore = True)
		prediction = model.infer()

		fname = args.predict[0].split('/')[-1].split('.')[0]

		test_img = cv2.imread(args.predict[0])

		test_img = cv2.resize(test_img, (_IMG_SHAPE[0], _IMG_SHAPE[1]))

		preds_to_img(prediction, test_img, fname)


if __name__ == '__main__':
	main()
