import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression

import time
import rfpimp as rfp

def perm_importances(model, X, y, features=None, n_examples=None, n_mc_samples=100):
	"""
	Calculate permutation importances for a BNN or its mimic. Also returns the time taken
    so result is a 2-tuple (array of importance values, time)

	Args:
		model: a BNN_Classifier, RandomForestClassifier or GradientBoostingClassifier
		X, y: examples and labels. The permutation importances are computed by shuffling columns
			  of X and seeing how the prediction accuracy for y is affected
		features: How many features to compute importances for. Default (None) is to compute
				  for every feature. Otherwise use a list of integers
		n_examples: How many examples to use in the computation. Default (None) uses all the
					features. Otherwise choose a positive integer that is less than 
					the number of rows of X/y.
		n_mc_samples: number of MC samples (BNN only)

	Returns a 1D array of permutation importance values in the same order as the columns of X
	"""
	X_df, y_df = pd.DataFrame(X), pd.DataFrame(y)
	X_df.columns = X_df.columns.map(str) # rfpimp doesn't like integer column names

	if n_examples is None:
		n_examples = -1
	start_time = time.time()
	if isinstance(model, BNN_Classifier):
		imp_vals = np.squeeze(rfp.importances(model, X_df, y_df,
								metric=lambda model, X, y, sw: model.score(X, y, n_mc_samples, sample_weight=sw), n_samples=n_examples, sort=False).values)
	elif isinstance(model, RandomForestClassifier) or isinstance(model, GradientBoostingClassifier):
		imp_vals = np.squeeze(rfp.importances(model, X_df, y_df, n_samples=n_examples, sort=False).values)
	time_taken = time.time() - start_time
	return imp_vals, time_taken

def mean_bnn_prediction(bnn, X, n_mc_samples):
	"""
	Mean prediction by a BNN over MC samples
	"""
	return bnn.predict(X, n_mc_samples, return_logits=False).mean(axis=0)

def get_lm_mimic_coefficients(bnn_object, x_train, x_test=None, n_mc_samples=100):
	"""
	Return logistic regression slopes from individual models on each feature in x_train
	predicting or mimicking the mode of the predictions from the BNN.
		
	Args:
		bnn_object: an instance of the BNN class
		x_train: array of training examples with shape (n_examples, n_features).
					The random forest will be trained on these examples and their
					BNN predictions
		n_mc_samples: the number of MC samples used when making BNN predictions.
						Their mean is used as the labels for the random forest.
	"""

	# Fit linear model on mode BNN prediction
	bnn_prediction = most_common_bnn_prediction(bnn_object, x_train, n_mc_samples)

	start_time = time.time()
	lm = LogisticRegression(C=1e20) # No regularisation penalty
	lm.fit(x_train, bnn_prediction)
	fit_time = time.time() - start_time

	# Check RF predictions against BNN predictions on held-out data, if provided
	if x_test is not None:
		print("Linear model has MIMIC accuracy of {:.5f}".format(lm.score(
					x_test, most_common_bnn_prediction(bnn_object, x_test, n_mc_samples))))

	return lm.coef_.reshape(x_train.shape[1]), fit_time

def get_rf_mimic(bnn_object, x_train, x_test=None, n_mc_samples=100, cv_grid=None,
	n_jobs=1, best_model_only=True, **kwargs):
	"""
	Get the random forest trained to mimic the mean
	predictions of a Bayesian neural network. The mimic model is a regression model trained
	trained on the soft predictions (the logits) of the BNN.

	Model selection is performed using random search cross-validation with 10 iterations and 5 folds - this can be quite
	slow but shouldn't take more than 10 minutes when parallelised over all available
	cores. Default behaviour is to use one core.
	
	Args:
		bnn_object: an instance of the BNN class
		x_train: array of training examples with shape (n_examples, n_features).
					The random forest will be trained on these examples and their
					BNN predictions
		x_test: array of test examples with shape (n_examples, n_features).
					If provided (default is None) then the random forest will be
					evaluated by comparing its predictions
					to those of the BNN.
		n_mc_samples: the number of MC samples used when making BNN predictions.
						Their mean is used as the labels for the random forest.
		cv_grid: dictionary of {hyperparameter : values} over which the RF will be
				 cross-validated. Default value is None, which uses the grid defined
				 in the source code.
		n_jobs: the number of threads to use when running the GBM
				cross validation (default is 1). An integer for the number of threads or 
				-1 for all available threads.
		best_model_only: should the best model be returned or should the CV result
						 be returned. The CV result is the dict described at
						 https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
		**kwargs: passed to RandomizedSearchCV (controls e.g. number of iterations, CV folds etc)
	"""

	# Grid of random forest hyperparameter choices. (Arbitrary) default
	if cv_grid is None:
		cv_grid = {'n_estimators': [int(x) for x in np.linspace(100, 2000, 10)],
				   'max_features': ['auto', 'sqrt'],
				   'max_depth': [int(x) for x in np.linspace(10, 110, 11)] + [None],
				   'min_samples_split': [2, 5, 10],
				   'min_samples_leaf': [1, 2, 4],
				   'bootstrap': [True, False]}

	# Perform model selection using cross-validation
	random_search = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=cv_grid,
										n_jobs=n_jobs, **kwargs)

	bnn_prediction = mean_bnn_prediction(bnn_object, x_train, n_mc_samples)
	start_time = time.time()
	random_search.fit(x_train, bnn_prediction)
	cv_time = time.time() - start_time

	print("Cross-validation took {:.2f} seconds".format(cv_time))

	# Check RF predictions against BNN predictions on held-out data, if provided
	if x_test is not None:
		print("Best RF model has MIMIC R^2 of {:.5f}".format(random_search.best_estimator_.score(
					x_test, mean_bnn_prediction(bnn_object, x_test, n_mc_samples))))
	if best_model_only:
		return random_search.best_estimator_, cv_time
	else:
		return random_search, cv_time

def get_gbm_mimic(bnn_object, x_train, x_test=None, n_mc_samples=100,
	cv_grid=None, n_jobs=1, best_model_only=True, **kwargs):
	"""
	Get a gradient boosting machine (GBM) trained to mimic the mean
	predictions of a Bayesian neural network. The mimic is a regression model
	trained on the soft predictions (the prediction probabilities) of the BNN.

	The GBM is chosen using random search cross-validation with 10 iterations and 5 folds -
	this can be quite slow but shouldn't take more than 10 minutes when parallelised over
	all available cores. Default behaviour is to use one core.

 	Args:
		bnn_object: an instance of the BNN class
		x_train: array of training examples with shape (n_examples, n_features).
					The GBM will be trained on these examples and their
					BNN predictions
		x_test: array of test examples with shape (n_examples, n_features).
					If provided (default is None) then the GBM will be
					evaluated by comparing its predictions
					to those of the BNN.
		n_mc_samples: the number of MC samples used when making BNN predictions.
						Their mean is used as the labels for the random forest.
		cv_grid: dictionary of {hyperparameter : values} over which the RF will be
				 cross-validated. Default value is None, which uses the grid defined
				 in the source code.
		n_jobs: the number of threads to use when running the GBM
				cross validation (default is 1). An integer for the number of threads or 
				-1 for all available threads.
		best_model_only: should the best model be returned or should the CV result
						 be returned. The CV result is the dict described at
						 https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
		**kwargs: passed to RandomizedSearchCV (controls e.g. number of iterations (n_iter), CV folds (cv)...)

	"""
	# Grid of GBM hyperparameter choices
	if cv_grid is None:
		cv_grid = {'n_estimators': [int(x) for x in np.linspace(100, 2000, 10)], 
					'learning_rate': [0.1, 0.05, 0.02],
					'max_depth': [int(x) for x in np.linspace(10, 110, 11)] + [None], 
					'min_samples_leaf': [1, 2, 4], 
					'max_features': ['auto', 'sqrt'] }

	random_search = RandomizedSearchCV(estimator=GradientBoostingRegressor(), param_distributions=cv_grid,
		n_jobs=n_jobs, **kwargs)

	# Compute most common BNN prediction over MC samples
	bnn_prediction = mean_bnn_prediction(bnn_object, x_train, n_mc_samples)
	start_time = time.time()
	random_search.fit(x_train, bnn_prediction)
	cv_time = time.time() - start_time

	print("Cross-validation took {:.2f} seconds".format(cv_time))

	# Check GBM predictions against BNN predictions on held-out data, if provided
	if x_test is not None:
		print("Best GBM model has MIMIC R^2 of {:.5f}".format(random_search.best_estimator_.score(
					x_test, mean_bnn_prediction(bnn_object, x_test, n_mc_samples))))
	if best_model_only:
		return random_search.best_estimator_, cv_time
	else:
		return random_search, cv_time