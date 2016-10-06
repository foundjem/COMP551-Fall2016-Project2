import sys, csv
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score

class multinomial_nb(object):
	def __init__(self, alpha=1.0):
		self.alpha = alpha
		
	def fit(self, X, y):
		n,m = X.shape
		uniques, counts = np.unique(y, return_counts=True)
	
		# Store label names for later access
		self.labels = dict(enumerate(uniques))
		
		# Separate samples by class
		subs = [X[rows,:] for rows in y == uniques[:,None]]

		# Individual word counts by class
		counts_by_class = np.vstack([s.sum(axis=0) for s in subs])
		
		# Add smoothing term
		counts_by_class = counts_by_class + self.alpha
		
		# Total counts by class
		totals = counts_by_class.sum(axis=1).astype(float)
		
		# Feature log probabilities
		self.w_ = np.log(counts_by_class / totals[:,None]).T
		
		# Append the class log priors
		self.w_ = np.vstack((self.w_, np.log(counts/float(n))[None]))
	
		return self
	
	def log_probabilities(self, X):
		# Add column of ones for the bias term
		X = np.append(X,np.ones(X.shape[0])[:,None],axis=1)
		return X.dot(self.w_)

	def predict(self, X):
		# Report class with highest probability
		codes = np.argmax(self.log_probabilities(X), axis=1)
		return np.array([self.labels[c] for c in codes])

	def score(self, X, y, return_prediction=False):
		predicted = self.predict(X)
		accuracy = np.sum(y == predicted) / float(y.size)
		if return_prediction: return (accuracy, predicted)
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
		self.labels = dict(enumerate(uniques))

		# Separate samples by class
		subs = [X[rows,:] for rows in y == uniques[:,None]]

		# Individual word counts by class
		counts_by_class = np.vstack([s.sum(axis=0) for s in subs])

		# Add smoothing term
		counts_by_class = counts_by_class + self.alpha

		# Total counts by class
		totals = counts_by_class.sum(axis=1).astype(float)
	
		# Feature probabilities
		self.w_ = (counts_by_class / (counts+self.alpha*2.0)[:,None]).T
		self.w_ = np.log(np.vstack((self.w_, 1.0 - self.w_)))
		
		# Append the class log priors
		self.w_ = np.vstack((self.w_, np.log(counts/float(n))[None]))
	
		return self
	
	def log_probabilities(self, X):
		X0 = np.logical_not(X)
		X1 = np.logical_not(X0)
		X = np.hstack((X1,X0,np.ones((X.shape[0],1))))
		# Add column of ones for the bias term
		return X.dot(self.w_)

	def predict(self, X):
		# Report class with highest probability
		codes = np.argmax(self.log_probabilities(X), axis=1)
		return np.array([self.labels[c] for c in codes])

	def score(self, X, y, return_prediction=False):
		predicted = self.predict(X)
		accuracy = np.sum(y == predicted) / float(y.size)
		if return_prediction: return (accuracy, predicted)
		return accuracy

	def get_params(self, deep=True):
		return {'alpha':self.alpha}

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self


if __name__ == '__main__':

	############################################
	################# Get data #################
	############################################
	print "Reading files........................",
	sys.stdout.flush()

	X_trn = np.genfromtxt('X_trn.csv', delimiter=',', dtype=int)
	X_val = np.genfromtxt('X_val.csv', delimiter=',', dtype=int)
	X_all = np.genfromtxt('X_all.csv', delimiter=',', dtype=int)
	X_tst = np.genfromtxt('X_tst.csv', delimiter=',', dtype=int)

	ids_trn, X_trn = X_trn[:,0][:,None], X_trn[:,1:]
	ids_val, X_val = X_val[:,0][:,None], X_val[:,1:]
	ids_all, X_all = X_all[:,0][:,None], X_all[:,1:]
	ids_tst, X_tst = X_tst[:,0][:,None], X_tst[:,1:]

	Y_trn = np.genfromtxt('Y_trn.csv', delimiter=',', dtype=str, usecols=[1])
	Y_val = np.genfromtxt('Y_val.csv', delimiter=',', dtype=str, usecols=[1])
	Y_all = np.genfromtxt('Y_all.csv', delimiter=',', dtype=str, usecols=[1])
	print "Done."

	print "Training classifier..................",
	sys.stdout.flush()
	m_nb = GridSearchCV(multinomial_nb(),{'alpha':[0.0001,0.001,0.01,0.1,1,10]},cv=10)
	b_nb = GridSearchCV(bernoulli_nb(),{'alpha':[0.0001,0.001,0.01,0.1,1,10]},cv=10)
						
	m_nb.fit(X_trn, Y_trn)
	b_nb.fit(X_trn, Y_trn)
	
	m_nb = m_nb.best_estimator_
	b_nb = b_nb.best_estimator_
	print "Done."

	print "Testing on validation data...........",
	sys.stdout.flush()
	m_accuracy, m_fitted = m_nb.score(X_val, Y_val, return_prediction=True)
	b_accuracy, b_fitted = b_nb.score(X_val, Y_val, return_prediction=True)
	print "Done."

	print "Training on all data.................",
	sys.stdout.flush()
	m_scores = cross_val_score(m_nb, X_all, Y_all, cv=10)
	b_scores = cross_val_score(b_nb, X_all, Y_all, cv=10)
	m_nb.fit(X_all, Y_all)
	b_nb.fit(X_all, Y_all)
	print "Done."
	print "Multinomial:"
	print "   Validation set accuracy:  %f" % m_accuracy
	print "   Cross-validated accuracy: %f" % np.mean(m_scores)
	print "Bernoulli:"
	print "   Validation set accuracy:  %f" % b_accuracy
	print "   Cross-validated accuracy: %f" % np.mean(b_scores)
	
	print "Predicting labels for unseen data....",
	sys.stdout.flush()
	m_Y_tst = m_nb.predict(X_tst)
	b_Y_tst = b_nb.predict(X_tst)
	print "Done."
	
	print "Writing to file......................",
	sys.stdout.flush()
	m_Y_tst = np.hstack((ids_tst, m_Y_tst[:,None]))
	b_Y_tst = np.hstack((ids_tst, b_Y_tst[:,None]))
	
	m_Y_tst = np.vstack((['id','category'], m_Y_tst))
	b_Y_tst = np.vstack((['id','category'], b_Y_tst))
	
	np.savetxt('Y_tst_MULTINOMIAL.csv', m_Y_tst, delimiter=',', fmt='%s')
	np.savetxt('Y_tst_BERNOULLI.csv', b_Y_tst, delimiter=',', fmt='%s')
	print "Done."





