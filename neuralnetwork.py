import sys, csv
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelBinarizer
from utils import load_sparse_csr, oversample, one_hot_encode, one_hot_decode
import pandas as pd


def sigmoid(x, d=False):
	if d: return x * (1.0 - x)
	else: return 1.0 / (1.0 + np.exp(-x))

def quad_cost(y0, y1, d=False):
	r = y1 - y0
	if d: return r
	else: return 0.5 * r.dot(r)

class neural_network(object):


	def __init__(self, d_in, d_out, layers,
				 activation_func=sigmoid,
				 cost_function=quad_cost,
				 epochs=100, eta=0.001):
		nn_in = [d_in] + layers
		nn_out = layers = [d_out]
		self.w = [np.random.randn(i,o) for i,o in zip(nn_in,nn_out)]
		self.b = [np.zeros(o) for o in nn_out]
		self.a_func = activation_func
		self.c_func = cost_function
		self.epochs = epochs
		self.eta = eta
		pass

	def fit(self, X, y):
		n,m = X.shape
		uniques, counts = np.unique(y, return_counts=True)
		
		# Store label names for later access
		self.classes_ = uniques
		
		self.lb = LabelBinarizer()
		y = self.lb.fit_transform(y)
		
		for i in xrange(self.epochs):
			grad_w, grad_b = self.backprop(X,y)
			self.w = self.w - (self.eta / n) * grad_w
			self.b = self.b - (self.eta / n) * grad_b

	def predict(self, X):
		return self.lb.inverse_transform(self.predict(X), self.classes_)
	
	def forward(self, a):
		for w,b in zip(self.w, self.b):
			a = self.a_func(a.dot(w) + b)
		return a
	
	def score(self, X, y):
		predicted = self.predict(X)
		accuracy = np.sum(y == predicted) / float(y.size)
		return accuracy
		
	def backprop(self, X, y):
		Z, A = [], [X]
		for w,b in zip(self.w, self.b):
			Z.append(A[-1].dot(w) + b)
			A.append(self.a_func(Z[-1]))
		
		print ' '.join(str(type(a)) for a in A)
		print ' '.join(str(a.shape) for a in A)
		print ' '.join(str(type(z)) for z in Z)
		print ' '.join(str(z.shape) for z in Z)
		
		# Backward pass
		delta = self.c_func(A[-1], y, d=True) * self.a_func(Z[-1], d=True)
		grad_b = [delta]
		grad_w = [delta.dot(A[-2].T)]
		
		for z,w,a in zip(Z[:-2], self.w[:-1], A)[::-1]:
			print ""
			print delta_b[-1].shape
			print delta_b[-1].shape
			print z.shape
			print w.shape
			print a.shape
			sp = self.a_func(z, d=True)
			delta = w.T.dot(delta) * sp
			grad_b = [delta] + grad_b
			grad_w = [delta.dot(a.T)] + grad_w

		return np.array(grad_w), np.array(grad_b)

if __name__ == '__main__':

	############################################
	################# Get data #################
	############################################
	print "Reading files........................",
	sys.stdout.flush()
	X_trn = load_sparse_csr('X_trn_tfidf.npz')
	X_val = load_sparse_csr('X_val_tfidf.npz')
#	X_all = load_sparse_csr('X_all_tfidf.npz')
#	X_tst = load_sparse_csr('X_tst_tfidf.npz')
	X_all = load_sparse_csr('X_all_cheat_tfidf.npz')
	X_tst = load_sparse_csr('X_tst_cheat_tfidf.npz')

	ids_trn, X_trn = X_trn[:,0].toarray().astype(int), X_trn[:,1:]
	ids_val, X_val = X_val[:,0].toarray().astype(int), X_val[:,1:]
	ids_all, X_all = X_all[:,0].toarray().astype(int), X_all[:,1:]
	ids_tst, X_tst = X_tst[:,0].toarray().astype(int), X_tst[:,1:]

	Y_trn = pd.read_csv('Y_trn.csv', usecols=[1]).values.flatten()
	Y_val = pd.read_csv('Y_val.csv', usecols=[1]).values.flatten()
	Y_all = pd.read_csv('Y_all.csv', usecols=[1]).values.flatten()
	print "Done."
	
	n,m = X_trn.shape
	ann = neural_network(m,4,[10])
	ann.fit(X_trn, Y_trn)


