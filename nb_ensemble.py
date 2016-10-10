import numpy as np
import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import BaggingClassifier
from naive_bayes import multinomial_nb
from utils import load_sparse_csr


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
	
	############################################
	################# Training #################
	############################################
	print "Building ensemble....................",
	sys.stdout.flush()
	n_estimators = 100
	ensemble = BaggingClassifier(multinomial_nb(alpha=0.0001),
								 n_estimators=n_estimators,
								 bootstrap = True,
								 n_jobs=4)
	print "Done."
								 
	print "Training on all data.................",
	sys.stdout.flush()
	scores = cross_val_score(ensemble, X_all, Y_all, cv=10)
	ensemble.fit(X_all, Y_all)
	print "Done."
	print "Ensemble of %d bagged classifiers:" % n_estimators
	print "   Cross-validated accuracy: %f" % np.mean(scores)
	
	
	############################################
	################ Prediction ################
	############################################
	print "Predicting labels for unseen data....",
	sys.stdout.flush()
	Y_tst = ensemble.predict(X_tst)
	print "Done."
	
	print "Writing to file......................",
	sys.stdout.flush()
	Y_tst = np.hstack((ids_tst, Y_tst[:,None]))
	Y_tst = np.vstack((['id','category'], Y_tst))
	np.savetxt('Y_tst_ENSEMBLE.csv', Y_tst, delimiter=',', fmt='%s')
	print "Done.\a"





