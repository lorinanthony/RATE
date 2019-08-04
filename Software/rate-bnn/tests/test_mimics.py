import pytest

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from rate.rate_base import RATE, groupRATE

def test_rate_results():
	p = 20
	mu = np.random.randn(p)
	Sigma = np.random.randn(p, p)
	Sigma = np.dot(Sigma, Sigma.transpose())

	rate_res = RATE(mu, Sigma)
	grouprate_res = groupRATE(mu, Sigma, groups=[[j] for j in range(p)])
	assert ((rate_res-grouprate_res)==0).all()
