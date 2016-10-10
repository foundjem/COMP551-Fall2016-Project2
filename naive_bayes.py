import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

__all__ = ['multinomial_nb','bernoulli_nb']

class multinomial_nb(BaseEstimator, ClassifierMixin):
	def __init__(self, alpha=1.0):
		self.alpha = alpha
		
	def fit(self, X, y):
		n,m = X.shape
		uniques, counts = np.unique(y, return_counts=True)
		
		# Store label names for later access
		self.classes_ = uniques
		
		# Separate samples by class
		subs = [X[rows,:] for rows in y == uniques[:,None]]

		# Individual word counts by class
		counts_by_class = np.vstack([s.sum(axis=0) for s in subs])
		
		# Add smoothing term
		counts_by_class = counts_by_class + self.alpha
		
		# Total counts by class
		totals = counts_by_class.sum(axis=1).astype(float)
		
		# Feature log probabilities
		self.f_log_probs_ = np.log(counts_by_class / totals).T
		
		# Class log priors
		self.log_priors_ = np.log(counts/float(n))
	
		return self
	
	def predict_log_proba(self, X):
		check_is_fitted(self, "classes_")
		probs = (X.dot(self.f_log_probs_) + self.log_priors_).astype(float)
		log_prob_x = np.ma.log(np.sum(np.exp(probs),axis=1)).filled(0)
		return (probs - log_prob_x).A
	
	def predict_proba(self, X):
		return np.exp(self.predict_log_proba(X))

	def predict(self, X, return_probs=False):
		# Report class with highest probability
		log_probs = self.predict_log_proba(X)
		labels = self.classes_[np.argmax(log_probs, axis=1)]
		if return_probs:
			probs = np.exp(np.max(log_probs, axis=1))
			return labels, probs
		return labels

	def score(self, X, y):
		predicted = self.predict(X)
		accuracy = np.sum(y == predicted) / float(y.size)
		return accuracy

	def get_params(self, deep=True):
		return {'alpha':self.alpha}

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self


class bernoulli_nb(object):
	def __init__(self, alpha=1.0):
		self.alpha = alpha
		
	def fit(self, X, y):
		X = X.astype(bool)
		
		n,m = X.shape
		uniques, counts = np.unique(y, return_counts=True)
		
		# Store label names for later access
		self.classes_ = uniques

		# Separate samples by class
		subs = [X[rows,:] for rows in y == uniques[:,None]]

		# Individual word counts by class
		counts_by_class = np.vstack([s.sum(axis=0) for s in subs])

		# Add smoothing term
		counts_by_class = counts_by_class + self.alpha

		# Total counts by class
		totals = counts_by_class.sum(axis=1).astype(float)
	
		# Feature probabilities
		f_probs = (counts_by_class / (counts+self.alpha*2.0)[:,None]).T
		self.f_log_probs_ = np.log(np.vstack((f_probs, 1.0 - f_probs)))
		
		# Append the class log priors
		self.log_priors_ = np.log(counts/float(n))
	
		return self
	
	def predict_log_proba(self, X):
		check_is_fitted(self, "classes_")
		X = np.hstack((X.astype(bool), np.logical_not(X)))
		probs = (X.dot(self.f_log_probs_) + self.log_priors_).astype(float)
		log_prob_x = np.ma.log(np.sum(np.exp(probs),axis=1)).filled(0)
		return probs - log_prob_x
	
	def predict_proba(self, X):
		return np.exp(self.predict_log_proba(X))

	def predict(self, X, return_probs=False):
		# Report class with highest probability
		log_probs = self.predict_log_proba(X)
		labels = self.classes_[np.argmax(log_probs, axis=1)]
		if return_probs:
			probs = np.exp(np.max(log_probs, axis=1))
			return labels, probs
		return labels

	def score(self, X, y):
		predicted = self.predict(X)
		accuracy = np.sum(y == predicted) / float(y.size)
		return accuracy

	def get_params(self, deep=True):
		return {'alpha':self.alpha}

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self