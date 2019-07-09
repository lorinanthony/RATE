import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
import warnings

class BNN_Classifier:
	"""
	Class implementing a Bayesian neural network for classification. This uses just the Keras API and can be 
	wrapped as an sklearn model (sklearn wrapping is not finished but enough is done to work with rfpimp package)

	Trained by maximising a variational lower bound with a sigmoid or softmax
	log-likelihood, depending on the number of classes (binary classification -> sigmoid,
	multi-class -> softmax).
	"""

	def __init__(self, layers, p, C):
		"""
		Args:

		layers: a list of layers that make up the network. These will be added to 
				a Keras.Sequential object in the order they appear in the list
		p: integer giving the input dimensionality
		C: integer giving the output dimensionality
		"""

		self.n_layers = len(layers)
		self.p = p
		self.C = C
		self.model = self.build_model(layers, True)
		self.hmodel = tf.keras.Model(self.model.input , self.model.layers[-2].output)

		if self.C == 2:
			warnings.warn("Binary classification uses C=1")
			self.C = 1
        
		# Entropy/log likelihood term for binary/non-binary case
		if self.C == 1:
			cross_entopy = lambda y_true, logits : tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=logits)
			y_activation = tf.keras.layers.Activation("sigmoid")
		elif self.C > 2:
			cross_entopy = lambda y_true, logits : tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=logits) # change to tf.nn.sparse_softmax_cross_entropy_with_logits?
			y_activation = tf.keras.layers.Activation("softmax")
		else:
			raise ValueError("C={} is not a valid number of classes".format(self.C))
            
		self.ymodel = self.build_model(self.model.layers+[y_activation], False) # The same model but returning predicted class probabilities

		# Objective function
		def elbo(y_true, logits):
			return tf.reduce_sum(cross_entopy(y_true, logits)) + sum(self.model.losses)/K.cast(K.shape(y_true)[0], "float32")

		# Variational parameters
		self.W1_loc = self.model.layers[-1].kernel_posterior.distribution.loc
		self.W1_scale = self.model.layers[-1].kernel_posterior.distribution.scale
		self.b1_loc = self.model.layers[-1].bias_posterior.distribution.loc

		self.model.compile(loss=elbo, optimizer=tf.keras.optimizers.Adam(1e-3), metrics=["accuracy"])
		self.ymodel.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(1e-3), metrics=["accuracy"]) # Compile this model so we can use .evaluate, but it is not actually trained

	def build_model(self, layers, check_shapes=True):
		model = tf.keras.Sequential()
		for l in layers:
			model.add(l)
		if check_shapes:
			assert model.layers[0].input_shape[1]==self.p, "p ({}) does not match the input dimension of the first layer ({})".format(self.p, model.layers[0].input_shape[1])
			assert model.layers[-1].units==self.C, "C ({}) does not match the output dimension of the final layer ({})".format(self.C, model.layers[-1].units)
		return model
    
	def fit(self, X, y, lr=None, callbacks=[], **kwargs):
		"""
		Train the model on examples X and labels y.

		lr is the learning rate for the Adam optimizer. If None (default) then the value is unchanged (it starts at 1e-3).

		callbacks are used to monitor training (see https://keras.io/callbacks/). Currently used to stop if ELBO objective
		reutrns an NaN value but add more by passing a list of callbacks.

		**kwargs will be passed to keras.model.fit (e.g. for batch size, number of epochs...see https://keras.io/models/model/#fit)
		One useful option is automatic cross-validation (argument is validation_split)

		So to train for 10 epochs with batch size 64 use
		bnn.fit(X, y, epochs=10, batch_size=10)
		"""
		if lr is not None:
			K.set_value(self.model.optimizer.lr, lr)
		return self.model.fit(X, y, callbacks=callbacks+[tf.keras.callbacks.TerminateOnNaN()], **kwargs)
        
	def predict(self, X, n_mc_samples, return_logits=False, **kwargs):
		"""
		Predict labels of X using MC samples. Can return logits by setting return_logits to True.
		Output shape is (n_mc_samples, n_examples, n_classes) for multi-class classification and
		(n_mc_samples, n_examples) for binary classification

		**kwargs are passed to keras.model.predict - see https://keras.io/models/model/#predict
		"""
		if return_logits:
			out = np.array([self.model.predict(X, **kwargs) for i in range(n_mc_samples)])
		else:
			out = np.array([self.ymodel.predict(X, **kwargs) for i in range(n_mc_samples)])

		if self.C == 1:
			return out[:,:,0]
		else:
			return out
        
	def sample_logits(self, X, n_mc_samples, **kwargs):
		"""
		For backward compatibility with the previous BNN class. A wrapper for predict().
		"""
		return self.predict(X, n_mc_samples, True, **kwargs)

	def train(self, X, y, n_epochs, batch_size, learning_rate=0.000):
		"""
		For backward compatibility with the previous BNN class. A wrapper for fit().
		"""
		return self.fit(X, y, epochs=n_epochs, batch_size=batch_size)
        
	def __call__(self):
		return self.model
    
	def var_params(self):
		"""
		Returns the variational parameters of the final layer (plus the bias term)

		Returns:
			The means and variances of the elements of the final layer weight matrix
			plus the final layer bias term (which is deterministic)
		"""
		W1_loc, W1_scale, b = [K.eval(self.W1_loc), K.eval(self.W1_scale), K.eval(self.b1_loc)]
		return W1_loc, np.square(W1_scale), b
    
	def compute_H(self, X, **kwargs):
		"""
		Compute the (deterministic) output of the penultimate layer for some input array X.
		This is what is passed to the Bayesian final layer.

		**kwargs passed to predict - see https://keras.io/models/model/#predict 

		Args:
			X: input array with shape (n_examples, input_dimensionality)

		Returns:
			Array of hidden unit outputs with shape (n_examples, number of units in final hidden layer)
		"""
		return self.hmodel.predict(X, **kwargs)

	def score(self, X, y, n_mc_samples, std=False, **kwargs):
		"""
		Mean score over MC samples. Score returns the metrics defined in __init__ (currently just accuracy).
        std=True will also return the std of the accuracy over the MC samples.

		Computes the mean accuracy over the examples (X) predicting the labels (y) for n_mc_samples MC 
		sampled networks. The returned value is the mean accuracy over the MC samples.
		"""
		out = [self.ymodel.evaluate(X, y, verbose=0, **kwargs)[1] for i in range(n_mc_samples)]
		if not np.isfinite(out).all():
			warnings.warn("There were {} non-finite results when evaluating the scores. These were removed.".format(np.isnan(out).sum()))
		if std:
			return np.nanmean(out), np.nanstd(out)
		else:
			return np.nanmean(out)

class BNN_Regressor:
	"""
	Class implementing a Bayesian neural network for scalar regression. This uses just the Keras API and can be 
	wrapped as an sklearn model (sklearn wrapping is not finished but enough is done to work with rfpimp package).

	Trained by maximising a variational lower bound with a sigmoid or softmax
	log-likelihood, depending on the number of classes (binary classification -> sigmoid,
	multi-class -> softmax).
	"""

	def __init__(self, layers, p):
		"""
		Args:

		layers: a list of layers that make up the network. These will be added to 
				a Keras.Sequential object in the order they appear in the list
		p: integer giving the input dimensionality
		C: integer giving the output dimensionality
		"""

		self.n_layers = len(layers)
		self.p = p
		self.C = 1
		self.model = self.build_model(layers, True)
		self.hmodel = tf.keras.Model(self.model.input , self.model.layers[-2].output)
		
		# Objective function
		def elbo(y_true, y_pred):
			return tf.losses.mean_squared_error(y_true, y_pred) + sum(self.model.losses)/K.cast(K.shape(y_true)[0], "float32")

		# Variational parameters
		self.W1_loc = self.model.layers[-1].kernel_posterior.distribution.loc
		self.W1_scale = self.model.layers[-1].kernel_posterior.distribution.scale
		self.b1_loc = self.model.layers[-1].bias_posterior.distribution.loc

		self.model.compile(loss=elbo, optimizer=tf.keras.optimizers.Adam(1e-3), metrics=["mse"])

	def build_model(self, layers, check_shapes=True):
		model = tf.keras.Sequential()
		for l in layers:
			model.add(l)
		if check_shapes:
			assert model.layers[0].input_shape[1]==self.p, "p ({}) does not match the input dimension of the first layer ({})".format(self.p, model.layers[0].input_shape[1])
			assert model.layers[-1].units==1, "Output dimension of the final layer is not 1 (it is {})".format(model.layers[-1].units)
		return model
    
	def fit(self, X, y, callbacks=[], **kwargs):
		"""
		Train the model on examples X and labels y.

		callbacks are used to monitor training (see https://keras.io/callbacks/). Currently used to stop if ELBO objective
		reutrns an NaN value but add more by passing a list of callbacks.

		**kwargs will be passed to keras.model.fit (e.g. for batch size, number of epochs...see https://keras.io/models/model/#fit)
		One useful option is automatic cross-validation (argument is validation_split)

		So to train for 10 epochs with batch size 64 use
		bnn.fit(X, y, epochs=10, batch_size=10)
		"""
		return self.model.fit(X, y, callbacks=callbacks+[tf.keras.callbacks.TerminateOnNaN()], **kwargs)
        
	def predict(self, X, n_mc_samples, return_logits=False, **kwargs):
		"""
		Predict real-valued output using n_mc_samples MC samples.

		The return_logits argument is to match the equivalent BNN_Classifier method - it doesn't do anything in regression
		(the final layer activation is the identity function).

		**kwargs are passed to keras.model.predict - see https://keras.io/models/model/#predict
		"""
		return np.array([self.model.predict(X, **kwargs) for i in range(n_mc_samples)])
        
	def sample_logits(self, X, n_mc_samples, **kwargs):
		"""
		For backward compatibility with the previous BNN class. A wrapper for predict().
		"""
		return self.predict(X, n_mc_samples, True, **kwargs)

	def train(self, X, y, n_epochs, batch_size, learning_rate=0.000):
		"""
		For backward compatibility with the previous BNN class. A wrapper for fit().
        Learning_rate is ignored.
		"""
		return self.fit(X, y, epochs=n_epochs, batch_size=batch_size)
        
	def __call__(self):
		return self.model
    
	def var_params(self):
		"""
		Returns the variational parameters of the final layer (plus the bias term)

		Returns:
			The means and variances of the elements of the final layer weight matrix
			plus the final layer bias term (which is deterministic)
		"""
		W1_loc, W1_scale, b = [K.eval(self.W1_loc), K.eval(self.W1_scale), K.eval(self.b1_loc)]
		return W1_loc, np.square(W1_scale), b
    
	def compute_H(self, X, **kwargs):
		"""
		Compute the (deterministic) output of the penultimate layer for some input array X.
		This is what is passed to the Bayesian final layer.

		**kwargs passed to predict - see https://keras.io/models/model/#predict 

		Args:
			X: input array with shape (n_examples, input_dimensionality)

		Returns:
			Array of hidden unit outputs with shape (n_examples, number of units in final hidden layer)
		"""
		return self.hmodel.predict(X, **kwargs)

	def score(self, X, y, n_mc_samples, std=False, **kwargs):
		"""
		Mean score over MC samples. Score returns the metrics defined in __init__ (currently just meas-squared error).
        If std=True this will also return the standard deviation of the MSE across the MC samples

		Computes the mean MSE over the examples (X) predicting the output (y) for n_mc_samples MC 
		sampled networks. The returned value is the mean of the MSE over the MC samples.
		"""
		out = [self.model.evaluate(X, y, verbose=0, **kwargs)[1] for i in range(n_mc_samples)]
		if not np.isfinite(out).all():
			warnings.warn("There were {} non-finite results when evaluating the scores. These were removed.".format(np.isnan(out).sum()))
		if std:
			return np.nanmean(out), np.nanstd(out)
		else:
			return np.nanmean(out)
