import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from utils import oversample

__all__ = ['multinomial_nb','bernoulli_nb']

class multinomial_nb(BaseEstimator, ClassifierMixin):
	def __init__(self, alpha=1.0, oversample_data=False):
		self.alpha = alpha
		self.oversample_data = oversample_data
		
	def fit(self, X, y, sample_weight=None):
		if self.oversample_data:
			X,y = oversample(X,y)
		
		n,m = X.shape
		
		# Convert to one-hot vector (counting becomes matrix multiplication)
		self.lb_ = LabelBinarizer()
		Y = self.lb_.fit_transform(y).astype(float)
		
		# If we have a sample weight, weight the examples
		if sample_weight is not None:
			Y = sample_weight[:,None] * Y
		
		# Store label names for later access
		self.classes_ = self.lb_.classes_
		
		# Number of examples belonging to each class
		counts = Y.sum(axis=0)
		
		# Sum of each feature by class (smoothed)
		counts_by_class = Y.T * X + self.alpha
		
		# Sum of all features by class
		totals = counts_by_class.sum(axis=1)
		
		# Feature log probabilities
		self.f_log_probs_ = np.log(counts_by_class.T / totals)
		
		# Class log priors
		self.log_priors_ = np.log(counts / float(n))
	
		return self
	
	def predict_log_proba(self, X):
		check_is_fitted(self, "classes_")
		probs = X.dot(self.f_log_probs_) + self.log_priors_
		log_prob_x = np.ma.log(np.exp(probs).sum(axis=1)).filled(0)
		return probs - log_prob_x[:,None]
	
	def predict_proba(self, X):
		return np.exp(self.predict_log_proba(X))

	def predict(self, X, return_probs=False):
		# Report class with highest probability
		log_probs = self.predict_log_proba(X)
		labels = self.lb_.inverse_transform(log_probs)
		if return_probs:
			probs = np.exp(np.max(log_probs, axis=1))
			return labels, probs
		return labels

	def score(self, X, y):
		predicted = self.predict(X)
		accuracy = np.sum(y == predicted) / float(y.size)
		return accuracy

	def get_params(self, deep=True):
		return {'alpha':self.alpha,'oversample_data':self.oversample_data}

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self


class bernoulli_nb(object):
	def __init__(self, alpha=1.0):
		self.alpha = alpha
		
	def fit(self, X, y, sample_weight=None):
		X = X.astype(bool)
		n,m = X.shape

		# Convert to one-hot vector (counting becomes matrix multiplication)
		self.lb_ = LabelBinarizer()
		Y = self.lb_.fit_transform(y).astype(float)
		
		# If we have a sample weight, weight the examples
		if sample_weight is not None:
			Y = sample_weight[:,None] * Y
		
		# Store label names for later access
		self.classes_ = self.lb_.classes_
		
		# Number of examples belonging to each class
		counts = Y.sum(axis=0)
		
		# Sum of each feature by class (smoothed)
		counts_by_class = Y.T * X + self.alpha

		# Total counts by class
		totals = counts_by_class.sum(axis=1)
	
		# Feature probabilities
		f_probs = (counts_by_class / (counts+self.alpha*2.0)[:,None]).T
		self.f_log_probs_0 = np.log(1.0 - f_probs)
		self.f_log_probs_1 = np.log(f_probs)
		
		# Append the class log priors
		self.log_priors_ = np.log(counts/float(n))
	
		return self
	
	def predict_log_proba(self, X):
		check_is_fitted(self, "classes_")
		X1 = X.astype(bool)
		X0 = np.logical_not(X1.A)
		probs = X1.dot(self.f_log_probs_1)
		probs += X0.dot(self.f_log_probs_0)
		probs += self.log_priors_
		log_prob_x = np.ma.log(np.exp(probs).sum(axis=1)).filled(0)
		return probs - log_prob_x[:,None]
	
	def predict_proba(self, X):
		return np.exp(self.predict_log_proba(X))

	def predict(self, X, return_probs=False):
		# Report class with highest probability
		log_probs = self.predict_log_proba(X)
		labels = self.lb_.inverse_transform(log_probs)
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