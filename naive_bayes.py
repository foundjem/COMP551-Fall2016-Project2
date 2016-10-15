import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from utils import oversample

__all__ = ['multinomial_nb','bernoulli_nb']

class multinomial_nb(BaseEstimator, ClassifierMixin):
	def __init__(self, alpha=1.0):
		self.alpha = alpha
	
	def fit(self, X, y, sample_weight=None):
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

	def predict(self, X):
		log_probs = self.predict_log_proba(X)
		return self.lb_.inverse_transform(log_probs)

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

	def predict(self, X):
		log_probs = self.predict_log_proba(X)
		return self.lb_.inverse_transform(log_probs)

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


if __name__ == '__main__':
	import sys
	import pandas as pd
	from scipy import sparse
	from sklearn.grid_search import GridSearchCV
	from sklearn.ensemble import BaggingClassifier
	from sklearn.cross_validation import cross_val_score, ShuffleSplit
	from sklearn.learning_curve import validation_curve, learning_curve
	from sklearn.metrics import confusion_matrix, classification_report
	from utils import load_sparse_csr, semi_supervised, plot_learning_curve
	
	rng = np.random.RandomState(42)

	############################################
	################# Get data #################
	############################################
	print "Reading files...........................",
	sys.stdout.flush()
	X_trn = load_sparse_csr('X_trn_tfidf.npz')
	X_val = load_sparse_csr('X_val_tfidf.npz')
	X_all = load_sparse_csr('X_all_tfidf.npz')
	X_tst = load_sparse_csr('X_tst_tfidf.npz')
	
	ids_trn, X_trn = X_trn[:,0].toarray().astype(int), X_trn[:,1:]
	ids_val, X_val = X_val[:,0].toarray().astype(int), X_val[:,1:]
	ids_all, X_all = X_all[:,0].toarray().astype(int), X_all[:,1:]
	ids_tst, X_tst = X_tst[:,0].toarray().astype(int), X_tst[:,1:]

	Y_trn = pd.read_csv('Y_trn.csv', usecols=[1]).values.flatten()
	Y_val = pd.read_csv('Y_val.csv', usecols=[1]).values.flatten()
	Y_all = pd.read_csv('Y_all.csv', usecols=[1]).values.flatten()
	print "Done."
	
	############################################
	################# Vanilla ##################
	############################################
	print "Vanilla Multinomial Naive Bayes:"
	print "   Grid search for best parameters......",
	sys.stdout.flush()
	model = GridSearchCV(multinomial_nb(),
						 {'alpha':np.logspace(-20,0,11)},
						  cv=10, n_jobs=-1)
	model.fit(X_trn, Y_trn)
	cv_results = model.grid_scores_
	model = model.best_estimator_
	print "Done."
	print "   Best parameter set:"
	for k in model.get_params().keys():
		print "      %s: %s" % (k,model.get_params()[k])

	print "   Testing on validation data...........",
	sys.stdout.flush()
	accuracy = model.fit(X_trn,Y_trn).score(X_val, Y_val)
	predicted = model.predict(X_val)
	print "Done."
	print "   Classification report:\n   ",
	print classification_report(Y_val, predicted, target_names=model.classes_)
#	title = "Learning Curves (Naive Bayes)"
#	cv = ShuffleSplit(X_all.shape[0], n_iter=100,
#						test_size=0.05, random_state=rng)
#	plot_learning_curve(model, title, X_all, Y_all, ylim=(0.7, 1.01),
#						cv=cv, n_jobs=4, train_sizes=np.linspace(.01, 1.0, 20)).show()

	print "   Training on all data.................",
	sys.stdout.flush()
	scores = cross_val_score(model, X_all, Y_all, cv=10, n_jobs=-1)
	model.fit(X_all, Y_all)
	print "Done."
	print "   Validation set accuracy:  %f" % accuracy
	print "   Cross-validated accuracy: %f" % np.mean(scores)
	
	print "   Predicting labels for unseen data....",
	sys.stdout.flush()
	Y_tst = model.predict(X_tst)
	print "Done."
	
	print "   Writing to file......................",
	sys.stdout.flush()
	Y_tst = np.hstack((ids_tst, Y_tst[:,None]))
	Y_tst = np.vstack((['id','category'], Y_tst))
	np.savetxt('Y_tst_NB_VANILLA.csv', Y_tst, delimiter=',', fmt='%s')
	print "Done."
	
	############################################
	############# Semi-supervised ##############
	############################################
	print "Semi-supervised Multinomial Naive Bayes:"
	print "   Using vanilla parameters"

	print "   Testing on validation data...........",
	sys.stdout.flush()
	accuracy = semi_supervised(model, X_trn, Y_trn, X_val, 0.9).score(X_val, Y_val)
	print "Done."
	print "   Validation set accuracy:  %f" % accuracy

	print "   Training on all data.................",
	sys.stdout.flush()
	model = semi_supervised(model, X_all, Y_all, X_tst, 0.9)
	print "Done."
	
	print "   Predicting labels for unseen data....",
	sys.stdout.flush()
	Y_tst = model.predict(X_tst)
	print "Done."
	
	print "   Writing to file......................",
	sys.stdout.flush()
	Y_tst = np.hstack((ids_tst, Y_tst[:,None]))
	Y_tst = np.vstack((['id','category'], Y_tst))
	np.savetxt('Y_tst_NB_SEMI_SUPERVISED.csv', Y_tst, delimiter=',', fmt='%s')
	print "Done."

	############################################
	################# Ensemble #################
	############################################
	print "Ensemble of Multinomial Naive Bayes:"
	print "   Using vanilla parameters"

	print "   Building ensemble....................",
	sys.stdout.flush()
	n_estimators = 100
	model = BaggingClassifier(model, n_estimators=n_estimators,
							  bootstrap = True, n_jobs=-1, random_state=rng)
	print "Done."

	print "   Testing on validation data...........",
	sys.stdout.flush()
	accuracy = model.fit(X_trn,Y_trn).score(X_val, Y_val)
	print "Done."
	title = "Learning Curves (Naive Bayes)"
	cv = ShuffleSplit(X_all.shape[0], n_iter=100,
						test_size=0.05, random_state=rng)
	plot_learning_curve(model, title, X_all, Y_all, ylim=(0.7, 1.01),
						cv=cv, n_jobs=1, train_sizes=np.linspace(.01, 1.0, 10)).show()

	print "   Training on all data.................",
	sys.stdout.flush()
	scores = cross_val_score(model, X_all, Y_all, cv=10)
	model.fit(X_all, Y_all)
	print "Done."
	print "   Validation set accuracy:  %f" % accuracy
	print "   Cross-validated accuracy: %f" % np.mean(scores)
	
	print "   Predicting labels for unseen data....",
	sys.stdout.flush()
	Y_tst = model.predict(X_tst)
	print "Done."
	
	print "   Writing to file......................",
	sys.stdout.flush()
	Y_tst = np.hstack((ids_tst, Y_tst[:,None]))
	Y_tst = np.vstack((['id','category'], Y_tst))
	np.savetxt('Y_tst_NB_ENSEMBLE.csv', Y_tst, delimiter=',', fmt='%s')
	print "Done."
	
	############################################
	######### Semi-supervised Ensemble #########
	############################################
	print "Semi-supervised Ensemble of Multinomial Naive Bayes:"
	print "   Using vanilla parameters"

	print "   Semi-supervised learning.............",
	sys.stdout.flush()
	model = semi_supervised(model, X_all, Y_all, X_tst, 0.9)
	print "Done."
	
	print "   Predicting labels for unseen data....",
	sys.stdout.flush()
	Y_tst = model.predict(X_tst)
	print "Done."
	
	print "   Writing to file......................",
	sys.stdout.flush()
	Y_tst = np.hstack((ids_tst, Y_tst[:,None]))
	Y_tst = np.vstack((['id','category'], Y_tst))
	np.savetxt('Y_tst_NB_SS_ENSEMBLE.csv', Y_tst, delimiter=',', fmt='%s')
	print "Done.\a"