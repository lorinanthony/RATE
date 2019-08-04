import pytest

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

from rate import BnnBinaryClassifier, BnnScalarRegressor

def toy_binary_classification_data(n, p):
	X, y = make_classification(
		n_samples=n, n_features=p, n_informative=p, n_redundant=0, n_repeated=0,
		n_classes=2, n_clusters_per_class=3, flip_y=0.05, shuffle=False,
		random_state=123)
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.3, random_state=123)
	return (X_train, y_train), (X_test, y_test)

def network_layers(p, C):
	layers = []
	layers.append(tf.keras.layers.Dense(128, activation='relu', input_shape=(p,)))
	layers.append(tf.keras.layers.BatchNormalization())
	layers.append(tfp.layers.DenseLocalReparameterization(C))
	return layers

def test_unfitted_models():
	"""Calling predict before fit should raise exception
	"""
	bnn = BnnBinaryClassifier()
	with pytest.raises(NotFittedError):
		bnn.predict(np.random.randn(10,3))
		bnn.predict_proba(np.random.randn(10,3))
		bnn.predict_samples(np.random.randn(10,3))
		bnn.predict_proba_samples(np.random.randn(10,3))
	bnn = BnnScalarRegressor()
	with pytest.raises(NotFittedError):
		bnn.predict(np.random.randn(10,3))
		bnn.predict_samples(np.random.randn(10,3))

# Test using predict on unfitted model
def test_input_shapes_classification():
	"""Input dimensionality should match model's expectation
	(either from layers passed in constructor or first fit call)
	"""
	p = 100
	# First check default layer behaviour (p not set until fit)
	bnn = BnnBinaryClassifier()
	bnn.fit(np.random.randn(100, p), np.random.randint(2, size=100))
	with pytest.raises(ValueError):
		bnn.fit(np.random.randn(100, p+1), np.random.randint(2, size=100))

	# Now check if layers are provided
	bnn = BnnBinaryClassifier(layers=network_layers(p, 1))
	with pytest.raises(ValueError):
		bnn.fit(np.random.randn(p+1, 102), np.random.randint(2, size=100))

def test_input_shapes_regression():
	"""Input dimensionality should match model's expectation
	(either from layers passed in constructor or first fit call)
	"""
	p = 100
	# First check default layer behaviour (p not set until fit)
	bnn = BnnScalarRegressor()
	bnn.fit(np.random.randn(100, p), np.random.rand(100))
	with pytest.raises(ValueError):
		bnn.fit(np.random.randn(100, p+1), np.random.rand(100))

	# Now check if layers are provided
	bnn = BnnBinaryClassifier(layers=network_layers(p, 1))
	with pytest.raises(ValueError):
		bnn.fit(np.random.randn(p+1, 102), np.random.rand(100))

def test_label_types_classification():
	"""Binary classifier only accepts bianry labels
	"""
	p = 100
	# First check default layer behaviour (p not set until fit)
	bnn = BnnBinaryClassifier()
	with pytest.raises(ValueError):
		X_train = np.random.randn(100, p)
		bnn.fit(X_train, np.random.rand(100)) # Float labels
		bnn.fit(X_train, np.random.randint(3, size=100)) # Multiclass classification labels

def test_label_types_regression():
	"""Scalar regressino only accepts floats
	"""
	p = 100
	# First check default layer behaviour (p not set until fit)
	bnn = BnnScalarRegressor()
	with pytest.raises(ValueError):
		X_train = np.random.randn(100, p)
		bnn.fit(X_train, np.random.randint(2, size=100)) # Binary labels
		bnn.fit(X_train, np.random.randint(3, size=100)) # Multiclass classification labels
		# RTODO multiclass continuous labels

def test_predictions_classification():
	"""Binary classifier only predicts 0s and 1s for labels and values in
	[0,1] for prbabilities. Also checks shapes
	"""
	n, p = 100, 10
	n_test = 50
	n_mc_samples = 15
	#(X_train, y_train), (X_test, y_test) = toy_binary_classification_data(n, p)
	bnn = BnnBinaryClassifier(verbose=0)
	bnn.fit(np.random.randn(n, p), np.random.randint(2, size=n))
	yhat_labels = bnn.predict(np.random.randn(n_test, p))
	assert yhat_labels.shape[0] == n_test
	assert np.sum(yhat_labels==0) + np.sum(yhat_labels==1) == n_test
	yhat_proba = bnn.predict_proba(np.random.randn(n_test, p))
	assert np.all([prob >= 0.0 or prob <= 1.0 for prob in yhat_proba])
	yhat_labels_samples = bnn.predict_samples(np.random.randn(n_test, p), n_mc_samples)
	assert yhat_labels_samples.shape == (n_mc_samples, n_test)
	assert np.all([val in [0,1] for val in yhat_labels_samples.flat])
	yhat_proba_samples = bnn.predict_proba_samples(np.random.randn(n_test, p), n_mc_samples)
	assert yhat_proba_samples.shape == (n_mc_samples, n_test)
	assert np.all([prob >= 0.0 or prob <= 1.0 for prob in yhat_proba_samples.flat])

def test_H_classification():
	"""The shape of H should match the network architecture
	"""
	n, p = 100, 20
	bnn = BnnBinaryClassifier(network_layers(p, 1))
	with pytest.raises(ValueError):
		H_arr = bnn.H(np.random.randn(n, p))
	bnn.fit(np.random.randn(n, p), np.random.randint(2, size=n))
	H_arr = bnn.H(np.random.randn(n, p))
	assert H_arr.shape == (n, 128)

def test_H_regression():
	"""The shape of H should match the network architecture
	"""
	n, p = 100, 20
	bnn = BnnScalarRegressor(network_layers(p, 1))
	with pytest.raises(ValueError):
		H_arr = bnn.H(np.random.randn(n, p))
	bnn.fit(np.random.randn(n, p), np.random.rand(n))
	H_arr = bnn.H(np.random.randn(n, p))
	assert H_arr.shape == (n, 128)

def test_score_classification():
	"""The result of score should be correct for the bianry classifier (returns accuracy)
	"""
	n, p = 100, 10
	(X_train, y_train), (X_test, y_test) = toy_binary_classification_data(n, p)
	bnn = BnnBinaryClassifier(verbose=0)
	bnn.fit(X_train, y_train)
	yhat_labels = bnn.predict(X_test)
	assert bnn.score(X_test, y_test) == np.sum(y_test==yhat_labels)/float(y_test.shape[0])

def test_var_params_classification():
	n, p = 100, 20
	bnn = BnnBinaryClassifier(network_layers(p, 1))
	bnn.fit(np.random.randn(n, p), np.random.randint(2, size=n))
	W_mean, W_var, b = bnn.var_params()
	assert W_mean.shape[0] == 128
	assert W_var.shape[0] == 128
	assert b.shape[0] == 1

# Same for r2 in regression

# Chech output shapes

# Test input shapes error handling

# test default constructors

# Check label exception raising

# Check accuracy metricc