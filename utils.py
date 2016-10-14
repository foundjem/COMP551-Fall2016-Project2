import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve

__all__ = ['save_sparse_csr','load_sparse_csr','oversample','shuffle_together','semi_supervised']

def save_sparse_csr(file, mat):
	""" Save the given SciPy sparse CSR matrix as a .npz file 
	
	Parameters
	----------
	file : str
		File to save the matrix to. Will automatically append .npz
		extension.
		
	mat : SciPy sparse CSR matrix
		Matrix to be saved
	"""
	np.savez(file, data=mat.data,
				   indices=mat.indices,
				   indptr=mat.indptr,
				   shape=mat.shape)

def load_sparse_csr(file):
	""" Load a SciPy sparse CSR matrix from a .npz file
	
	Parameters
	----------
	file : str
		File to load matrix from. Must be a file saved by 
		save_sparse_csr function
		
	Returns
	-------
	M : SciPy sparse CSR matrix
		The matrix saved in the given file
	"""
	loader = np.load(file)
	return sparse.csr_matrix((loader['data'],
							  loader['indices'],
							  loader['indptr']),
							  shape = loader['shape'])

def oversample(X, y, random_state=np.random):
	""" Oversample the minority classes of the given dataset
	
    Parameters
    ----------
	X : Sparse matrix, shape = (# samples, # features)
		Sample features
		
	y : Numpy array, shape = (# samples,)
		Sample classes
		
	random_state : Numpy RandomState object
		Random number generator (for deterministic results)
		
	Returns
	-------
	X : Sparse matrix, shape = (# expanded samples, # features)
		Sample features with balanced classes, shuffled
	
	y : Numpy array, shape = (# expanded samples,)
		Sample classes, balanced, shuffled
	"""
	uniques, counts = np.unique(y,return_counts=True)
	
	# Separate samples by class
	subs = [X[rows,:] for rows in y == uniques[:,None]]
	
	# Random sample function
	nb_samples = np.max(counts)
	sample = lambda n: random_state.randint(n, size=nb_samples)
	
	# Oversample X and y
	X = sparse.vstack([s[sample(s.shape[0]),:] for s in subs])
	y = uniques.repeat(nb_samples)
	
	# Shuffle
	return shuffle_together(X,y,random_state)

def shuffle_together(X,y,random_state=np.random):
	""" Shuffle a sparse feature matrix and an array of classes in parallel
	
	Parameters
	----------
	X : Sparse matrix, shape = (# samples, # features)
		Sample features
		
	y : Numpy array, shape = (# samples,)
		Sample classes
		
	random_state : Numpy RandomState object
		Random number generator (for deterministic results)
		
	Returns
	-------
	X : Sparse matrix, shape = (# samples, # features)
		Shuffled sample reatures
	
	y : Numpy array, shape = (# samples,)
		Shuffled sample classes
	"""
	p = random_state.permutation(y.shape[0])
	return X[p,:], y[p]

def semi_supervised(model, X, y, Xt, thresh=0.9, predict_proba=None):
	""" Perform semi-supervised learning with the given model and data.
	The training will take place over several iterations. At each step, the
	model will be trained on the training set, and predict labels on the
	test set. Testing examples that the model is sure of (probability above the 
	given threshold) will be removed from the test set and added to the
	training set.
	
	Parameters
	----------
	model : SKLearn-like classifier
		The model to perform semi-supervised training on. Must have
		fit and predict functions, consistent with SKLearn estimators
		
	X : Sparse matrix, shape = (# training samples, # features)
		Training feature matrix
		
	y : Numpy array, shape = (# training samples,)
		Training labels
		
	Xt : Sparse matrix, shape = (# testing samples, # features)
		Testing feature matrix to be used for semi-supervised learning
		
	thresh : scalar
		The threshold at which the model is considered to be "sure" of its
		prediction. Should be between 0 and 1
		
	predict_proba : function
		The probability function, representing how "sure" a model is
		of its predictions. If none is provided, the given model's
		predict_proba function will be used. This parameter is 
		provided for cases like SKLearn's LinearSVC, which has no predict_proba
		function, but does have a decision function which more or less
		encodes the model's "surety" of predictions.
		
	Returns
	-------
	model : SKLearn-like classifier
		The model after being trained
	"""
	if predict_proba is None: predict_proba = model.predict_proba
	while True:
		# Train on expanded training set
		model.fit(X, y)
		
		# Test on unsure test set
		yt = model.predict(Xt)
		yp = np.max(predict_proba(Xt), axis=1)
		
		# Get testing examples the model is sure of
		good_idx = np.where(yp >= thresh)[0]
		bad_idx = np.where(yp < thresh)[0]
		
		# Break if there aren't any good samples
		if not good_idx.size: break
		
		# Add confident samples to training set
		X = sparse.vstack((X, Xt[good_idx,:]))
		y = np.hstack((y, yt[good_idx]))
		
		# Remove confident samples from test set
		Xt = Xt[bad_idx,:]

	return model



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
	TAKEN FROM SKLEARN 0.17 DOCUMENTATION
		http://scikit-learn.org/0.17/auto_examples/model_selection/plot_learning_curve.html#example-model-selection-plot-learning-curve-py
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt