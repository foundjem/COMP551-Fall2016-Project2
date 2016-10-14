import sys
import numpy as np
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from naive_bayes import multinomial_nb
from utils import load_sparse_csr, oversample


if __name__ == '__main__':

	############################################
	################# Get data #################
	############################################
	print "Reading files........................",
	sys.stdout.flush()
	X_trn = load_sparse_csr('X_trn_counts.npz')
	X_val = load_sparse_csr('X_val_counts.npz')
#	X_all = load_sparse_csr('X_all_tfidf.npz')
#	X_tst = load_sparse_csr('X_tst_tfidf.npz')
	X_all = load_sparse_csr('X_all_cheat_counts.npz')
	X_tst = load_sparse_csr('X_tst_cheat_counts.npz')

	ids_trn, X_trn = X_trn[:,0].toarray().astype(int), X_trn[:,1:]
	ids_val, X_val = X_val[:,0].toarray().astype(int), X_val[:,1:]
	ids_all, X_all = X_all[:,0].toarray().astype(int), X_all[:,1:]
	ids_tst, X_tst = X_tst[:,0].toarray().astype(int), X_tst[:,1:]

	Y_trn = pd.read_csv('Y_trn.csv', usecols=[1]).values.flatten()
	Y_val = pd.read_csv('Y_val.csv', usecols=[1]).values.flatten()
	Y_all = pd.read_csv('Y_all.csv', usecols=[1]).values.flatten()
	print "Done."

	############################################
	################# Training #################
	############################################
	print "Training classifier..................",
	sys.stdout.flush()
	nb = GridSearchCV(multinomial_nb(),{'alpha':10.0**np.arange(-20,-1),
										'oversample_data':[True,False]},cv=10)
	nb.fit(X_all, Y_all)
	nb = nb.best_estimator_
	print "Done."
	print "Best parameter set:"
	for k in nb.get_params().keys():
		print "   %s: %s" % (k,nb.get_params()[k])

	print "Testing on validation data...........",
	sys.stdout.flush()
	accuracy = nb.fit(X_trn,Y_trn).score(X_val, Y_val)
	print "Done."

	print "Training on all data.................",
	sys.stdout.flush()
	scores = cross_val_score(nb, X_all, Y_all, cv=10)
	nb.fit(X_all, Y_all)
	print "Done."
	print "Multinomial Naive Bayes:"
	print "   Validation set accuracy:  %f" % accuracy
	print "   Cross-validated accuracy: %f" % np.mean(scores)
	
	############################################
	################ Prediction ################
	############################################
	print "Predicting labels for unseen data....",
	sys.stdout.flush()
	Y_tst = nb.predict(X_tst)
	print "Done."
	
	print "Writing to file......................",
	sys.stdout.flush()
	Y_tst = np.hstack((ids_tst, Y_tst[:,None]))
	Y_tst = np.vstack((['id','category'], Y_tst))
	np.savetxt('Y_tst_MULTINOMIAL.csv', Y_tst, delimiter=',', fmt='%s')
	print "Done.\a"





