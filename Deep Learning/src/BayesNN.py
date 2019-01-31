import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from utils import load_mnist, softmax, accuracy, accuracy_onehot

class BNN:
	"""
	Class implementing a Bayesian neural network.

	Trained by maximising a variational lower bound with a sigmoid or softmax
	log-likelihood, depending on the number of classes (binary classification -> sigmoid,
	multi-class -> softmax).
	"""


	def __init__(self, layers, p, C, verbose=True):
		"""
		Args:

		layers: a list of layers that make up the network. These will be added to 
				a Keras Sequential object in the order they appear in the list
		p: integer giving the input dimensionality
		C: integer giving the output dimensionality
		verbose: Bool controlling whether or not to print info
		"""

		self.n_layers = len(layers)
		self.p = p
		self.C = C
		self.model = self.build_model(layers)
		self.verbose = verbose

		# Tensorflow placeholders and ops
		self.X_ph = tf.placeholder(tf.float32, [None, self.p])
		self.y_ph = tf.placeholder(tf.float32, [None, self.C])
		self.batch_size_ph = tf.placeholder(tf.float32, [])
		self.lr = tf.placeholder(tf.float32, [])
		self.logits = self.model(self.X_ph)
		self.H = tf.keras.Model(self.model.input , self.model.layers[-2].output)(self.X_ph)

		# Objective function + optimisation
		neg_log_likelihood = None
		if self.C == 1:
			neg_log_likelihood = tf.reduce_sum(
				tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_ph, logits=self.logits))
		elif self.C > 1:
			neg_log_likelihood = tf.reduce_sum(
				tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_ph, logits=self.logits))
		kl = sum(self.model.losses)
		self.loss = neg_log_likelihood + kl/self.batch_size_ph
		self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

		# Variational parameters
		self.W1_loc = self.model.layers[-1].kernel_posterior.distribution.loc
		self.W1_scale = self.model.layers[-1].kernel_posterior.distribution.scale
		self.b1_loc = self.model.layers[-1].bias_posterior.distribution.loc

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())


	def build_model(self, layers):
		model = tf.keras.Sequential()
		for l in layers:
			model.add(l)

		assert model.layers[0].input_shape[1]==self.p, "p ({}) does not match the input dimension of the first layer ({})".format(self.p, model.layers[0].input_shape[1])
		assert model.layers[-1].units==self.C, "C ({}) does not match the output dimension of the final layer ({})".format(self.C, model.layers[-1].units)

		return model

	def summary(self):
		"""
		Print the Keras summary of the model
		"""
		self.model.summary()

	def close_session(self):
		"""
		Close the Tensorflow session
		"""
		self.sess.close()

	def train(self, X_train, y_train,
		n_epochs, batch_size, learning_rate=0.001):

		"""
		Train the network by minimising the Evidence Lower Bound (ELBO)

		Args:
			X_train: array of training examples with shape (n_examples, input_dimensionality)
			y_train: array of training labels with shape (n_examples, output_dimensionality)
			n_epochs: number of training epochs
			batch_size: minibatch size

		Returns:
			The training loss at each epoch (mean over the minibatches).
		"""

		assert X_train.shape[0]==y_train.shape[0]
		assert y_train.shape[1]==self.C

		if self.C==1:
			assert np.sum(np.logical_or(y_train[:,0]==0, y_train[:,0]==1)) == y_train.shape[0]
		n_train = X_train.shape[0]

		train_losses = [] # Mean of training losses at each epoch

		if self.verbose:
			print "Training...\nEpoch ({} in total):".format(n_epochs)

		# Training loop (minibatches)
		for i in range(n_epochs):
			if self.verbose:
				print i,

			minibatch_idxs = np.random.choice(n_train, size=n_train, replace=False)
			minibatch_idxs = np.array_split(minibatch_idxs, n_train/batch_size)
	        
			this_epoch_train_losses = np.zeros(len(minibatch_idxs))

			for idx, mb in enumerate(minibatch_idxs):

				_, _loss = self.sess.run([self.train_op, self.loss],
									feed_dict={self.X_ph : X_train[mb,:], self.y_ph : y_train[mb,:],
												self.batch_size_ph : batch_size, self.lr : learning_rate})
				this_epoch_train_losses[idx] = (_loss)

			train_losses.append(np.mean(this_epoch_train_losses))

		if self.verbose:
			print ""
		return train_losses

	def sample_logits(self, X, n_mc_samples):
		"""
		Sample logits from the network for inputs X. This function can run out of memory
		quite quickly if X is reasonably large (E.g. 1,000 MC samples for 20,000 predictions)

		Args:
			X: input array with shape (n_examples, input_dimensionality)
			n_mc_samples: number of Monte Carlo samples

		Returns:
			Array of MC samples from the logit posterior with shape
			 (n_mc_samples, n_examples, output_dimensionality)
		"""
		sampled_logits = np.zeros((n_mc_samples, X.shape[0], self.C))
		if self.verbose:
			print "Sampling ({} total)...".format(n_mc_samples)
		for i in range(n_mc_samples):
			if self.verbose:
				if (i+1)%100==0:
					print i+1,
			sampled_logits[i,:,:] = self.sess.run(self.logits, feed_dict={self.X_ph : X})
		if self.verbose:
			print " done"
		return sampled_logits

	def var_params(self):
		"""
		Returns the variational parameters of the final layer (plus the bias term)

		Returns:
			The means and variances of the elements of the final layer weight matrix
			plus the final layer bias term (which is deterministic)
		"""
		W1_loc, W1_scale, b = self.sess.run([self.W1_loc, self.W1_scale, self.b1_loc])
		return W1_loc, np.square(W1_scale), b

	def compute_H(self, X):
		"""
		Compute the (deterministic) output of the penultimate layer for some input array X.
		This is what is passed to the Bayesian final layer.

		Args:
			X: input array with shape (n_examples, input_dimensionality)

		Returns:
			Array of hidden unit outputs with shape (n_examples, number of units in final hidden layer)
		"""
		return self.sess.run(self.H, feed_dict={self.X_ph : X})