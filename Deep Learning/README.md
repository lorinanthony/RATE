# Variable Importance for Bayesian Neural Networks

Here, we demonstrate how to implement RATE with the Bayesian neural network architectures as described in [Ish-Horowicz et al. (2019)](https://arxiv.org/abs/1901.09839). The `Notebooks` directory contains notebooks used to generate each of the plots in the paper. These are meant to serve as examples of how to build and train Bayesian neural networks and compute corresponding RATE values for its input variables.

The source code in `src` organized as follows:
* `rate_bnn.py` contains the code for computing RATE values for a Bayesian neural network.
* `BayesNN.py` contains a class implementing the Bayesian neural network.
* `utils.py` contains functions for loading data and computing accuracies across Monte Carlo sampled predicitons.
