import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import rankdata
import time
import warnings

def make_1d2d(arr):
	assert arr.ndim == 1
	return arr.reshape(arr.shape[0], 1)

def onehot_encode_labels(y):
	"""
	One-hot encode integer labels y. The number of classes is assumed to be
	the largest value in y

	Args:
		y: array with shape (n_examples,)

	Returns:
		array with shape (n_examples, n_classes)
	"""
	return OneHotEncoder(categories="auto", sparse=False).fit_transform(y.reshape(y.shape[0],1))

def load_mnist(onehot_encode=True, flatten_x=False, crop_x=0, classes=None):
	"""
	Load the MNIST dataset

	Args:
		onehot_encode: Boolean indicating whether to one-hot encode training
						and test labels (default True)
		flatten_x: Boolean indicating whether to flatten the training and 
					test inputs to 2D arrays with shape (n_examples, image_size**2).
					If False, returned inputs have shape (n_examples, image_size, image_size
					(default False)
		crop_x: Integer controlling the size of the border to be removed from the input 
				images (default 0, meaning no cropping).
		classes: None to include all classes (default). Otherwise include a list of two 
				 integers that will be encoded as 0, 1 in the order they appear.

	Returns:
		x_train, y_train, x_test, y_test: train and test inputs and labels.
											First dimension is always the number of examples

	"""
	(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
	x_train, x_test = x_train / 255.0, x_test / 255.0

	if classes is None and not onehot_encode:
		onehot_encode = True
		warnings.warn("Multi-class classification requires one-hot encoded labels")
	elif classes is not None and onehot_encode:
		onehot_encode = False
		warnings.warn("Binary classification doesn't use one-hot encoded labels")

	def crop(X, crop_size):
		assert crop_x < X.shape[1]/2
		assert crop_x < X.shape[2]/2
		return X[:,crop_size:-crop_size,crop_size:-crop_size]

	if crop_x > 0:
		x_train = crop(x_train, crop_x)
		x_test = crop(x_test, crop_x)

	# Flatten to 2d arrays (each example 1d)
	def flatten_image(X):
	    return X.reshape(X.shape[0], X.shape[1]*X.shape[1])
	if flatten_x:
		x_train = flatten_image(x_train)
		x_test = flatten_image(x_test)

	if onehot_encode:
		y_train = onehot_encode_labels(y_train)
		y_test = onehot_encode_labels(y_test)

	if classes is not None:
		assert len(classes) == 2
		c0, c1 = classes
		train_idxs_to_keep = np.logical_or(y_train==c0, y_train==c1)
		x_train, y_train = x_train[train_idxs_to_keep,:], y_train[train_idxs_to_keep]
		test_idxs_to_keep = np.logical_or(y_test==c0, y_test==c1)
		x_test, y_test = x_test[test_idxs_to_keep,:], y_test[test_idxs_to_keep]

		y_train = (y_train==c1).astype(int)[:,np.newaxis]
		y_test = (y_test==c1).astype(int)[:,np.newaxis]

	print("x_train has shape {}".format(x_train.shape))
	print("y_train has shape {}".format(y_train.shape))
	print("x_test has shape {}".format(x_test.shape))
	print("y_test has shape {}".format(y_test.shape))

	return x_train, y_train, x_test, y_test

def make_square(arr):
	"""
	Reshape a 1D array (or 2D array with .shape[2]==1) into a square 2D array
	"""
	assert arr.ndim==1 or arr.ndim==2, "array must be 1 or 2-D"
	if arr.ndim==2:
		assert arr.shape[1]==1, "If array is 2d then second dimension must be 1"
		arr = arr.reshape(arr.shape[0])
	assert arr.shape[0]**0.5 == int(arr.shape[0]**0.5), "array shape must be square (it is {})".format(arr.shape[0])
	return arr.reshape(int(arr.shape[0]**0.5), int(arr.shape[0]**0.5))

def accuracy_onehot(labels, preds):
	"""
	Compute the accuracy of predictions using one-hot encoded labels

	Args:
		labels: array of labels with shape (n_examples, n_classes). Must be one-hot encoded
				or result may be nonsense (this is not checked)
		preds: array of predictions with shape (n_examples, n_classes)

	Returns:
		Accuracy as float. Result is in [0,1]
	"""
	assert labels.shape[0]==preds.shape[0]
	return np.sum(np.argmax(preds, axis=1) == np.argmax(labels, axis=1))/float(labels.shape[0])

def accuracy(labels, preds):
	"""
	Compute the accuracy of predictions using integer labels

	Args:
		labels: array of labels with shape (n_examples,)
		preds: array of predictions with shape (n_examples, n_classes)

	Returns:
		Accuracy as float. Result is in [0,1]
	"""
	assert labels.shape[0]==preds.shape[0]
	return np.sum(preds==labels)/float(labels.shape[0])

def get_nullify_idxs(original_size, border_size):
	"""
	Get the indices of a flattened image that lie within border_size of the 
	edge of an image (use to pass to nullify argument in RATE function)

	Args:
		original size: Integer giving the size of the image
		border_size: Integer giving the size of the border to be removed.

	Returns:
		Array of (integer) indices that lie in the border.
	"""
	assert border_size < original_size/2, "Border too large to be removed from image of this size"
	tmp = np.zeros((original_size, original_size), dtype=int)
	tmp[:border_size,:] = 1
	tmp[-border_size:,:] = 1
	tmp[:,-border_size:] = 1
	tmp[:,:border_size] = 1
	tmp = tmp.reshape(tmp.shape[0]*tmp.shape[1])
	return np.where(tmp==1)[0]

def idx2pixel(idx, image_size):
	"""
	Get the 2D pixel location corresponding to the index of its flattened array

	Args:
		idx: integer index to be converted to pixel location
		image_size: integer giving size of the image

	Returns:
		i, j: the location of the pixel corresponding to idx
	"""
	assert idx < image_size**2, "index {} too large for image size {}".format(idx, image_size)
	tmp = np.zeros(image_size**2)
	tmp[idx] = 1
	tmp = tmp.reshape(image_size, image_size)
	i, j = np.where(tmp==1)
	return i[0], j[0]

def sampled_accuracies(pred_proba_samples, labels):
	"""
	Get the sampled accuracies over the entire test set from logit samples. 

	Args:
		pred_proba_samples: array of predicted probability samples with shape
							(n_mc_samples, n_examples, n_classes)/(n_mc_samples, n_examples)
					   		for multiclass/binary classification. (This is the shape returned by BNN_Classifier.predict).
		labels: array of one-hot encoded labels with shape (n_examples, n_classes) for non-binary clasification
				or (n_examples,1) for binary classification.

	Returns:
		Array of test accuracies for each round of MC samples with shape (n_mc_samples,)
	"""
	binary_labels = labels.shape[1]==1
    
	assert pred_proba_samples.shape[1]==labels.shape[0], "Different number of examples in logit samples and labels"

	if not binary_labels:
		assert pred_proba_samples.shape[2]==labels.shape[1], "Different number of classes in logit samples and labels"
		sampled_test_accuracies = np.sum(
			np.argmax(pred_proba_samples, axis=2) == np.argmax(labels, axis=1)[:,np.newaxis], axis=1)/float(labels.shape[0])
		
	else:
		sampled_test_accuracies = np.sum((pred_proba_samples[:,:]>0.5) == labels[:,0], axis=1)/float(labels.shape[0])

	return sampled_test_accuracies

def accuracy_hist(pred_proba_samples, labels):
	"""
	Plot a histogram showing test accuracies.
	Just calls sampled_accuracies then plots the result.
	"""
	sampled_acc = sampled_accuracies(pred_proba_samples, labels)
	avg_accuracy = round(np.mean(sampled_acc) * 100, 3)
	print("average accuracy across " + str(pred_proba_samples.shape[0]) + " samples: " + str(avg_accuracy) + "%\n")
	fig, ax = plt.subplots(figsize=(10,5))
	sns.distplot(100*sampled_acc, ax=ax, rug=True, kde=False)
	ax.set_xlabel("Test set accuracy (%)", fontsize=30)
	ax.set_ylabel("Frequency density", fontsize=30);
	ax.tick_params("both", labelsize=15)
	return sampled_acc

def rank_array(arr):
	assert arr.ndim==1
	return (arr.shape[0] - rankdata(arr)).astype(int)

def reverse_ranks(rankarr):
	return rankarr.shape[0] - rankarr - 1

def compute_power(pvals, SNPs):
	"""
	Compute the power for identifying causal predictors.
	Args:
		Ps: list of causal predictors
	Output: matrix with dimension (num. predictors, 2), where columns are FPR, TPR
	"""
	nsnps = len(pvals)
	all_snps = np.arange(0, nsnps)
	pos = SNPs
	negs = list(set(all_snps) - set(SNPs))

	pvals_rank = rank_array(pvals)

	rocr = np.zeros((nsnps, 2))
	for i in all_snps:
		v = pvals_rank[0:i]  # test positives
		z = list(set(all_snps) - set(v))  # test negatives

		TP = len(set(v) & set(pos))
		FP = len(set(v) & set(negs))
		TN = len(set(z) & set(negs))
		FN = len(set(z) & set(pos))

		TPR = 1.0*TP/(TP+FN); FPR = 1.0*FP/(FP+TN); #FDR = 1.0*FP/(FP+TP)

		rocr[i, :] = [FPR, TPR]

	return rocr