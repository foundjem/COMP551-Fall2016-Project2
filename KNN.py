from scipy.spatial.distance import seuclidean #imports abridged
import scipy 
import sys, csv
import operator
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from scipy import sparse
from scipy import stats


def read_data(file):
	with open(file, 'rt') as f:
		reader = csv.reader(f)
		header = next(reader)
		data = list(reader)
	header = {h:i for i,h in enumerate(header)}
	return header, data

def dist_func(X, sample):
	""" Hamming distance """
	print(X.shape, sample.shape)
	total = np.sum(X != sample, axis=1) / float(sample.size)
	return total

class KNNClassifier(object):
	def __init__(self, k):
		self.k = k

	def fit(self, X, y):
		self.X = X
		self.y = y
		return self

	def predict(self,X):
		distances = np.array([dist_func(self.X, x.A).flatten() for x in X])
		k_nearest = self.y[np.argsort(distances, axis=1)[:,:self.k]]
		# uniques, counts = np.unique(k_nearest, return_counts=True)
		stats.mode(k_nearest,axis=1)[0]
		return stats.mode(k_nearest,axis=1)[0]


def load_sparse_csr(file):
    loader = np.load(file)
    return sparse.csr_matrix((loader['data'],
							  loader['indices'],
							  loader['indptr']),
							  shape = loader['shape'])

if __name__ == '__main__':
	# print("Reading files")
	# h, trn_x = read_data("X_trn.csv")
	# h, trn_y = read_data("Y_trn.csv")
	# print("Done")



	print("Reading files........................")
	X_trn = load_sparse_csr('X_trn_counts.npz')
	X_val = load_sparse_csr('X_val_counts.npz')
	X_all = load_sparse_csr('X_all_counts.npz')
	X_tst = load_sparse_csr('X_tst_counts.npz')
	# X_all = load_sparse_csr('X_all_cheat_tfidf.npz')
	# X_tst = load_sparse_csr('X_tst_cheat_tfidf.npz')

	ids_trn, X_trn = X_trn[:,0].toarray().astype(int), X_trn[:,1:]
	ids_val, X_val = X_val[:,0].toarray().astype(int), X_val[:,1:]
	ids_all, X_all = X_all[:,0].toarray().astype(int), X_all[:,1:]
	ids_tst, X_tst = X_tst[:,0].toarray().astype(int), X_tst[:,1:]

	Y_trn = np.genfromtxt('Y_trn.csv',delimiter=',', skip_header=1, usecols=[1], dtype=str).flatten()
	Y_val = np.genfromtxt('Y_val.csv',delimiter=',', skip_header=1, usecols=[1], dtype=str).flatten()
	Y_all = np.genfromtxt('Y_all.csv',delimiter=',', skip_header=1, usecols=[1], dtype=str).flatten()
	print("Done.")

	clf = KNNClassifier(3).fit(X_trn, Y_trn)
	# fitted = np.array([clf.predict(x.A) for x in X_val])
	fitted = clf.predict(X_val[:100,:])
	print(np.sum(fitted == Y_val[:100]) / float(Y_val[:100].size))


