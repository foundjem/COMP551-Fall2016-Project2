import sys, csv
import numpy as np
from sklearn.grid_search import GridSearchCV

class multinomial_nb(object):
	def __init__(self, alpha=1.0):
		self.alpha = alpha
		
	def fit(self, X, y):
		n,m = X.shape
		uniques, counts = np.unique(y, return_counts=True)
	
		# Store label names for later access
		self.labels = dict(enumerate(uniques))
		
		# Separate samples by class
		subs = [[x for x, c1 in zip(X, y) if c1 == c0] for c0 in uniques]
		
		# Individual word counts by class
		counts_by_class = np.array([np.sum(np.array(s),axis=0) for s in subs])
		
		# Add smoothing term
		counts_by_class = counts_by_class + self.alpha
		
		# Total counts by class
		totals = np.sum(counts_by_class, axis=1).astype(float)
		
		# Feature log probabilities
		self.w_ = np.log(counts_by_class / totals[:,None]).T
		
		# Append the class log priors
		self.w_ = np.append(self.w_, np.log(counts/float(n))[None], axis=0)
	
		return self
	
	def log_probabilities(self, X):
		# Add column of ones for the bias term
		X = np.append(X,np.ones(X.shape[0])[:,None],axis=1)
		return X.dot(self.w_)

	def predict(self, X):
		# Report class with highest probability
		codes = np.argmax(self.log_probabilities(X), axis=1)
		return np.array([self.labels[c] for c in codes])

	def score(self, X, y):
		predicted = self.predict(X)
		return np.sum(y == predicted) / float(y.size)

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
	clf = GridSearchCV(multinomial_nb(), {'alpha':[0.0001,0.001,0.01,0.1,1,10]}, cv=10)
	clf.fit(X_trn, Y_trn)
	nb = clf.best_estimator_
	print "Done."

	print "Testing on validation data...........",
	sys.stdout.flush()
	fitted = nb.predict(X_val)
	print "Done."
	print "Accuracy:  %f" % (np.sum(Y_val == fitted) / float(fitted.size))

	print "Training on all data.................",
	sys.stdout.flush()
	clf = GridSearchCV(multinomial_nb(), {'alpha':[0.0001,0.001,0.01,0.1,1,10]}, cv=10)
	clf.fit(X_all, Y_all)
	nb = clf.best_estimator_
	print "Done."
	
	print "Predicting labels for unseen data....",
	sys.stdout.flush()
	Y_tst = nb.predict(X_tst)
	print "Done."
	
	print "Writing to file......................",
	sys.stdout.flush()
	Y_tst = np.hstack((ids_tst, Y_tst[:,None]))
	Y_tst = np.vstack((['id','category'],Y_tst))
	np.savetxt('Y_tst.csv', Y_tst, delimiter=',', fmt='%s')
	print "Done."





