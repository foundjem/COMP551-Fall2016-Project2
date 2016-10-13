import numpy as np
from scipy import sparse

def save_sparse_csr(file, mat):
    np.savez(file, data=mat.data,
				   indices=mat.indices,
				   indptr=mat.indptr,
				   shape=mat.shape)

def load_sparse_csr(file):
    loader = np.load(file)
    return sparse.csr_matrix((loader['data'],
							  loader['indices'],
							  loader['indptr']),
							  shape = loader['shape'])

def oversample(X,y):
	uniques, counts = np.unique(y,return_counts=True)
	
	# Separate samples by class
	subs = [X[rows,:] for rows in y == uniques[:,None]]
	
	# Random sample function
	nb_samples = np.max(counts)
	sample = lambda n: np.random.randint(n, size=nb_samples)
	
	# Oversample X and y
	X = sparse.vstack([s[sample(s.shape[0]),:] for s in subs])
	y = uniques.repeat(nb_samples)
	
	# Shuffle
	return shuffle_together(X,y)

def shuffle_together(X,y):
	p = np.random.permutation(y.shape[0])
	return X[p,:], y[p]