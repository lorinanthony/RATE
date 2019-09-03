import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
import warnings

from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from tensorflow.losses import mean_squared_error
from tensorflow.nn import sigmoid_cross_entropy_with_logits, softmax_cross_entropy_with_logits_v2

from tensorflow.keras.optimizers import Adam

from .utils import make_1d2d

#
# TODO: return samples from predict methods - done but not debugged
# TODO: debug the predict functions properly - do this in notebook
# TODO: replace fit checks with a decorator
#
def default_layers(p, C):
	"""The default model layers - single 128-unit dense layer, batch normalization then a 
	Bayesian DenseLocalReparameterization final layer.
	"""
	layers = []
	layers.append(tf.keras.layers.Dense(128, activation='relu', input_shape=(p,)))
	layers.append(tf.keras.layers.BatchNormalization())
	layers.append(tfp.layers.DenseLocalReparameterization(C))
	return layers

class BnnBase(BaseEstimator, metaclass=ABCMeta):
	"""Bayesian neural network base class.
	"""

	@abstractmethod
	def __init__(self, layers, optimiser_fn, n_mc_samples, 
				 verbose, target_type):
		self.layers = layers
		self.optimiser_fn = optimiser_fn
		self.n_mc_samples = n_mc_samples
		self.verbose = verbose
		self.target_type = target_type

	def _build_model(self, X, y):
		"""Compile the keras model for the logits
		"""
		if len(self.layers) == 0:
			self.layers = self._get_layers(X, y)
		self._logit_model = tf.keras.Sequential()
		for l in self.layers:
			self._logit_model.add(l)
		self.p = self._logit_model.layers[0].input_shape[1]
		self.C = self._logit_model.layers[-1].units
		self._logit_model.compile(loss=self._elbo(), optimizer=self.optimiser_fn())
		self._hmodel = tf.keras.Model(self._logit_model.input , self._logit_model.layers[-2].output)

	def fit(self, X, y, **kwargs):
		"""Train the model on examples X and labels y
		"""
		if X.ndim == 1:
			X = make_1d2d(X)
		try:
			getattr(self, "_logit_model")
		except AttributeError:
			# Construct model if running for the first time - maybe move this to _fit
			self._build_model(X, y)
		if X.shape[1] != self.p:
			raise ValueError("Model expects {} input dimensions, not {}".format(self.p, X.shape[1]))
		self._check_labels(y)
		# TODO: check optimizer here
		self.fit_history = self._logit_model.fit(X, y, verbose=self.verbose, **kwargs)
		return self

	@abstractmethod
	def _get_layers(self, X, y):
		pass

	def loss(self, X, y, n_mc_samples, mean_only=True, **kwargs):
		"""Return the loss evaluated on examples X and labels y

		# TODO: separate loss and loss_samples methods?
		"""
		if n_mc_samples is None: n_mc_samples = self.n_mc_samples
		loss_samples = np.array(
			[self._logit_model.evaluate(X, y, verbose=self.verbose, **kwargs) for _ in range(n_mc_samples)])
		if mean_only:
			return loss_samples.mean(axis=0)
		else:
			return loss_samples

	# Could make this protected
	def H(self, X, **kwargs):
		"""The (deterministic) activation of the penultimate layer for a set of examples X.

		Args:
			X: input with shape (n_examples, n_input_dimensions)

		Returns:
			H: activations of the penultimate network layer, an array with shape (n_examples, penultimate_layer_size)
		"""
		check_is_fitted(self, "_logit_model")
		return self._hmodel.predict(X, verbose=self.verbose, **kwargs)

	def var_params(self):
		"""The variational parameters of the final layer weights (the bias is deterministic but is
		is included with the weight posteriors as they are needed together to calculate RATE).

		TODO: what happens to these shapes when variational posterior is not fully factorised?

		Returns:
			M_W: the means of the variational posterior, shape (penultimate layer size, final layer size)
			V_W: the variances of the variational posterior, shape (penultimate layer size, final layer size)
			b: deterministic bias of the final layer, shape (final layer size,)
		"""
		check_is_fitted(self, "_logit_model")
		W1_loc, W1_scale, b = [K.eval(self._logit_model.layers[-1].kernel_posterior.distribution.loc),
							   K.eval(self._logit_model.layers[-1].kernel_posterior.distribution.scale),
							   K.eval(self._logit_model.layers[-1].bias_posterior.distribution.loc)]
		return W1_loc, np.square(W1_scale), b

	# Should have the class index as the first index, so that we can use numpy broadcasting for the batch multiplication
	def logit_posterior(self, X):
		"""The means and covariance of the posterior over the logits. Calculated using the variational
		posterior over the final layer weights.

		TODO: shapes may break for C > 1

		Returns:
			logit_posterior, a 2-tuple containing:
				1. an array of logit means with shape (n_classes, n_examples)
				2. an array of logit covariances with shape (n_classes, n_examples, n_examples)
		"""
		H_X = self.H(X)
		M_W, V_W, b = self.var_params()
		M_F = np.matmul(H_X, M_W) + b[np.newaxis,:]
		V_F = np.array([np.matmul(H_X*V_W[:,c], H_X.T) for c in range(self.C)])
		return M_F.T, V_F

	@abstractmethod
	def _nll(self, labels, logits):
		"""Negative log likelihood - the reconstruction term in the ELBO
		"""
		pass

	def _elbo(self):
		return lambda y_true, logits: tf.reduce_sum(self._nll(y_true, logits)) + sum(self._logit_model.losses)/K.cast(K.shape(y_true)[0], "float32")

	@abstractmethod
	def predict_samples(self, X, n_mc_samples=None, **kwargs):
		pass

	def _check_labels(self, y):
		"""Check labels match model
		"""
		if type_of_target(y) != self.target_type:
			raise ValueError("Label type is {} but model expects {}".format(type_of_target(y), self.target_type))

class BnnBinaryClassifier(BnnBase, ClassifierMixin):
	"""Bayesian neural network for binary classification
	"""

	def __init__(self, layers=[], optimiser_fn=Adam, n_mc_samples=100, verbose=1):
		super().__init__(
			layers=layers,
			optimiser_fn=optimiser_fn,
			n_mc_samples=n_mc_samples,
			verbose=verbose,
			target_type="binary")

	def _nll(self, labels, logits):
		"""Negative log likelihood - the reconstruction term in the ELBO
		"""
		return sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

	def predict(self, X, n_mc_samples=None, **kwargs):
		"""Predicted class labels over Monte Carlo samples. The mean prediction probability over the samples is
		thresholded
		"""
		check_is_fitted(self, "_logit_model")
		if X.ndim == 1:
			X = make_1d2d(X)
		if n_mc_samples is None: n_mc_samples = self.n_mc_samples
		logit_preds = np.squeeze(np.array([self._logit_model.predict(X, verbose=self.verbose, **kwargs) for _ in range(n_mc_samples)]))
		proba_preds = 1.0/(1.0+np.exp(-logit_preds))
		return (proba_preds.mean(axis=0) >= 0.5).astype(int)

	def predict_proba(self, X, n_mc_samples=None, **kwargs):
		"""Returns mean of predicted class probabilities over Monte Carlo samples
		"""
		check_is_fitted(self, "_logit_model")
		if n_mc_samples is None: n_mc_samples = self.n_mc_samples
		logit_preds = np.squeeze(np.array([self._logit_model.predict(X, verbose=self.verbose, **kwargs) for _ in range(n_mc_samples)]))
		proba_preds = 1.0/(1.0+np.exp(-logit_preds))
		return proba_preds.mean(axis=0)

	def _get_layers(self, X, y):
		return default_layers(X.shape[1], 1)

	def predict_proba_samples(self, X, n_mc_samples=None, **kwargs):
		check_is_fitted(self, "_logit_model")
		if n_mc_samples is None: n_mc_samples = self.n_mc_samples
		logit_preds = np.squeeze(np.array([self._logit_model.predict(X, verbose=self.verbose, **kwargs) for _ in range(n_mc_samples)]))
		return 1.0/(1.0+np.exp(-logit_preds))

	def predict_samples(self, X, n_mc_samples=None, **kwargs):
		check_is_fitted(self, "_logit_model")
		if n_mc_samples is None: n_mc_samples = self.n_mc_samples
		logit_preds = np.squeeze(np.array([self._logit_model.predict(X, verbose=self.verbose, **kwargs) for _ in range(n_mc_samples)]))
		proba_preds = 1.0/(1.0+np.exp(-logit_preds))
		return (proba_preds > 0.5).astype(int)

class BnnScalarRegressor(BnnBase, RegressorMixin):
	"""Bayesian neural network for scalar regression
	"""

	def __init__(self, layers=[], optimiser_fn=Adam, n_mc_samples=100, verbose=1):
		super().__init__(
			layers=layers,
			optimiser_fn=optimiser_fn,
			n_mc_samples=n_mc_samples,
			verbose=verbose,
			target_type="continuous")

	def _nll(self, labels, logits):
		"""Negative log likelihood - the reconstruction term in the ELBO
		"""
		return mean_squared_error(labels=labels, predictions=logits)

	def predict(self, X, n_mc_samples=None, **kwargs):
		"""Returns mean prediction over Monte Carlo samples
		"""
		check_is_fitted(self, "_logit_model")
		if n_mc_samples is None: n_mc_samples = self.n_mc_samples
		return np.mean([self._logit_model.predict(X, verbose=self.verbose, **kwargs) for _ in range(n_mc_samples)], axis=0)

	def predict_samples(self, X, n_mc_samples=None, **kwargs):
		check_is_fitted(self, "_logit_model")
		if n_mc_samples is None: n_mc_samples = self.n_mc_samples
		return np.array([self._logit_model.predict(X, verbose=self.verbose, **kwargs) for _ in range(n_mc_samples)])

	def _get_layers(self, X, y):
		return default_layers(X.shape[1], 1)