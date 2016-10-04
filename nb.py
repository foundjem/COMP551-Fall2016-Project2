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
	
	with open('train_in.csv', 'rt') as f:
		reader = csv.reader(f)
		header = next(reader)
		train_data = [[int(x) for x in l[1:]] for l in list(reader)]
	
	with open('all_train_in.csv', 'rt') as f:
		reader = csv.reader(f)
		header = next(reader)
		train_all_data = [[int(x) for x in l[1:]] for l in list(reader)]
	
	with open('validation_in.csv', 'rt') as f:
		reader = csv.reader(f)
		header = next(reader)
		val_data = [[int(x) for x in l[1:]] for l in list(reader)]
	
	with open('test_in.csv', 'rt') as f:
		reader = csv.reader(f)
		header = next(reader)
		test_data = [[int(x) for x in l[1:]] for l in list(reader)]
	
	with open('train_out.csv', 'rt') as f:
		reader = csv.reader(f)
		header = next(reader)
		train_labels = [l[1] for l in list(reader)]
	
	with open('all_train_out.csv', 'rt') as f:
		reader = csv.reader(f)
		header = next(reader)
		train_all_labels = [l[1] for l in list(reader)]
	
	with open('validation_out.csv', 'rt') as f:
		reader = csv.reader(f)
		header = next(reader)
		val_labels = [l[1] for l in list(reader)]


	X_train = np.array(train_data)
	X_val = np.array(val_data)
	X_train2 = np.array(train_all_data)
	X_test = np.array(test_data)
	
	Y_train = np.array(train_labels)
	Y_val = np.array(val_labels)
	Y_train2 = np.array(train_all_labels)
	print "Done."

	print "Training classifier..................",
	sys.stdout.flush()
	clf = GridSearchCV(multinomial_nb(), {'alpha':[0.0001,0.001,0.01,0.1,1,10,100]}, cv=5)
	clf.fit(X_train, Y_train)
	nb = clf.best_estimator_
	print "Done."

	print "Testing on validation data...........",
	sys.stdout.flush()
	fitted = nb.predict(X_val)
	print "Done."
	print "Accuracy:  %f" % (np.sum(Y_val == fitted) / float(fitted.size))


	print "Training on all data.................",
	sys.stdout.flush()
	clf = GridSearchCV(multinomial_nb(), {'alpha':[0.0001,0.001,0.01,0.1,1,10,100]}, cv=5)
	clf.fit(X_train2, Y_train2)
	nb = clf.best_estimator_
	fitted = nb.predict(X_test)


	with open('test_out.csv', 'wb') as f:
		writer = csv.writer(f, delimiter=',')
		writer.writerow(['id','category'])
		for id,pred in enumerate(fitted):
			writer.writerow([id,pred])






