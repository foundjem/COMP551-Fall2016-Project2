import sys, csv
import numpy as np
import pandas as pd


def entropy(y):
	""" H(y) = -sum_v[ P(y=v) log(P(y=v)) ] """
	P = np.unique(y, return_counts=True)[1] / float(y.size)
	return -np.sum(P[P.nonzero()] * np.log2(P[P.nonzero()]))

def cond_entropy(x,y):
	""" H(y|x) = sum_v[ P(x=v) H(y|x=v) ] """
	uniques, counts = np.unique(x,return_counts=True)
	priors = counts / float(x.size)
	ents = np.array([entropy(y[x==v]) for v in uniques])
	return np.sum(priors * ents)

def info_gain(X,y):
	""" IG(x) = H(D) - H(D|x) """
	h = entropy(y)
	return np.array([h - cond_entropy(x,y) for x in X.T])

def majority_class(y):
	uniques, indices = np.unique(y, return_inverse=True)
	return uniques[np.argmax(np.bincount(indices))]


class decision_node(object):
	def __init__(self, X, y, cols=None):
		if cols is None: cols = np.arange(X.shape[1])
		if np.all(y == y[0]):
			self.leaf = True
			self.maj = y[0]
		elif cols.size == 0:
			self.leaf = True
			self.maj = majority_class(y)
		else:
			self.leaf = False
			self.maj = majority_class(y)
		
			col = np.argmax(info_gain(X[:,cols], y))
			rest = np.delete(cols,col)
			self.col = cols[col]
			
			self.branches = np.unique(X[:,col]).tolist()
			splits = [X[:,col] == b for b in self.branches]
			sub_Xs = [X[s,:] for s in splits]
			sub_ys = [y[s] for s in splits]
			
			self.children = [decision_node(Xs,ys,rest) for Xs,ys in zip(sub_Xs,sub_ys)]

	def classify(self, item):
		if self.leaf: return self.maj
		if item[self.col] not in self.branches: return self.maj
		branch = self.branches.index(item[self.col])
		return self.children[branch].classify(item)

	def __str__(self):
		return self.tostring()

	def tostring(self, ind=0):
		if self.leaf: return '   '*ind + str(self.maj)
		head = '   '*ind + '{'+str(self.col)+'}' + ': ' + str(self.branches)
		body = '\n'.join(c.tostring(ind+1) for b,c in zip(self.branches,self.children))
		return head + '\n' + body


class decision_tree(object):
	def __init__(self):
		pass

	def fit(self, X, y):
		self.head = decision_node(X,y)
		return self

	def predict(self, X):
		return np.array([self.head.classify(x) for x in X])

	def score(self, X, y):
		predicted = self.predict(X)
		return np.sum(y == predicted) / float(y.size)

	def get_params(self, deep=True):
		return {}

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self


if __name__ == '__main__':

	X = np.array([[0,0,0], #0
				  [0,0,1], #0
				  [0,1,0], #0
				  [0,1,1], #1
				  [1,0,0], #0
				  [1,0,1], #0
				  [1,1,0], #1
				  [1,1,1]])#1
	y = np.array([0,0,0,1,0,0,1,1])
	
	dt = decision_tree()
	dt.fit(X,y)
	print dt.head
	print dt.predict(X)


	############################################
	################# Get data #################
	############################################
	print "Reading files........................",
	sys.stdout.flush()

	X_trn = pd.read_csv('X_trn.csv', dtype=int, header=None).values
	X_val = pd.read_csv('X_val.csv', dtype=int, header=None).values
	X_all = pd.read_csv('X_all.csv', dtype=int, header=None).values
	X_tst = pd.read_csv('X_tst.csv', dtype=int, header=None).values

	ids_trn, X_trn = X_trn[:,0][:,None], X_trn[:,1:]
	ids_val, X_val = X_val[:,0][:,None], X_val[:,1:]
	ids_all, X_all = X_all[:,0][:,None], X_all[:,1:]
	ids_tst, X_tst = X_tst[:,0][:,None], X_tst[:,1:]

	Y_trn = pd.read_csv('Y_trn.csv', usecols=[1], header=None).values.flatten()
	Y_val = pd.read_csv('Y_val.csv', usecols=[1], header=None).values.flatten()
	Y_all = pd.read_csv('Y_all.csv', usecols=[1], header=None).values.flatten()
	print "Done."

	print "Training classifier..................",
	sys.stdout.flush()
	dt = decision_tree()
	dt.fit(X_trn, Y_trn)
	print "Done."

	print "Fitting validation data..............",
	sys.stdout.flush()
	fitted = dt.predict(X_val)
	print "Done."
	print np.sum(Y_val == fitted) / float(fitted.size)




