import numpy as np

import warnings

from abc import ABCMeta, abstractmethod

class ProjectionBase(metaclass=ABCMeta):
	"""Base class for projections used to calculate effect size analogues
	"""
	@abstractmethod
	def __init__(self):
		self._X = None # Cached data matrix

	@abstractmethod
	def posterior(self, X, M_F, V_F):
		pass

class CovarianceProjection(ProjectionBase):
	"""The covariance projection
	"""
	def __init__(self):
		super().__init()
		self._X_c = None # Cached centred data matrix

	def posterior(self, X, M_F, V_F):
		"""Calculate the mean and covariance of the effect size analogue posterior.

		Args:
			X: array of inputs with shape (n_examples, n_variables)
			M_F: array of logit posterior means with shape (n_examples, n_classes)
			V_F: array of logit posterior covariances with shape (n_classes, n_examples, n_examples)
		
		Returns:
			effect_size_analogue_posterior: a 2-tuple containing:
				1. array of posterior means with shape (n_variables, n_classes)
				2. array of posterior covariances with shape (n_classes, n_variables, n_variables)
		"""

		# Check if X has been cached
		if self._X is None:
			self._X = X
			self._X_c = X - X.mean(axis=0)[np.newaxis,:]
		elif not np.array_equal(self._X, X):
			self._X = X
			self._X_c = X - X.mean(axis=0)[np.newaxis,:]

		# Calculate effect size analogue posterior mean and variance
		C = V_F.shape[2]

		M_F_c = M_F - M_F.mean(axis=1)
		M_B = 1.0/(n-1.0) * np.matmul(self._X_c.T, M_F_c[:,:,np.newaxis])[:,:,0]
		# M_B = 1.0/(n-1.0) * np.einsum('ij,kj -> ik', M_F_c) check if these two lines are equivalent
		V_B = 1.0/(n-1.0)**2.0 * np.matmul(np.matmul(self._X_c.T, V_F), self._X_c)

		return M_B, V_B

class PseudoinverseProjection(ProjectionBase):
	"""The pseudoinverse projection
	"""
	def __init__(self):
		super().__init__()
		self._X_dagger = None

	def posterior(self, X, M_F, V_F):
		"""Calculate the mean and covariance of the effect size analogue posterior.

		Args:
			X: array of inputs with shape (n_examples, n_variables)
			M_F: array of logit posterior means with shape (n_examples, n_classes)
			V_F: array of logit posterior covariances with shape (n_classes, n_examples, n_examples)
		
		Returns:
			effect_size_analogue_posterior: a 2-tuple containing:
				1. array of posterior means with shape (n_variables, n_classes)
				2. array of posterior covariances with shape (n_classes, n_variables, n_variables)
		"""

		# Check if X has been cached 
		if self._X is None:
			self._X = X
			self._X_dagger = np.linalg.pinv(self._X)
		elif not np.array_equal(self._X, X):
			self._X = X
			self._X_dagger = np.linalg.pinv(self._X)

		# Calculate effect size analogue posterior mean and variance
		C = V_F.shape[2]
		M_B = np.matmul(self._X_dagger, M_F)
		V_B = np.matmul(np.matmul(self._X_dagger, V_F), self._X_dagger.T)

		return M_B, V_B


