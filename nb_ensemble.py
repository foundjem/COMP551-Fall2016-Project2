import sys
import numpy as np
import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import BaggingClassifier
from naive_bayes import multinomial_nb
from utils import load_sparse_csr
from scipy import sparse


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
	bagged = BaggingClassifier(multinomial_nb(alpha=0.0001, oversample_data=False),
							   n_estimators=n_estimators,
							   bootstrap = True,
							   n_jobs=-1)
	print "Done."
	
	print "Training classifier..................",
	sys.stdout.flush()
	bagged.fit(X_trn, Y_trn)
	print "Done."

	print "Testing on validation data...........",
	sys.stdout.flush()
	bagged_accuracy = bagged.score(X_val, Y_val)
	print "Done."
								 
	print "Training on all data.................",
	sys.stdout.flush()
	bagged_scores = cross_val_score(bagged, X_all, Y_all, cv=10)
	bagged.fit(X_all, Y_all)
	print "Done."
	print "Ensemble of %d bagged classifiers:" % n_estimators
	print "   Validation set accuracy:  %f" % bagged_accuracy
	print "   Cross-validated accuracy: %f" % np.mean(bagged_scores)

	############################################
	######### Semi-supervised learning #########
	############################################
	print "Semi-supervised learning.............",
	sys.stdout.flush()
	X_working = X_tst
	thresh = 0.9
	top_n = 20
	iters, nb_added = 0, 0
	while True:
		bagged.fit(X_all, Y_all)
		Y_probs = np.max(bagged.predict_proba(X_working), axis=1)
		Y_working = bagged.predict(X_working)
#		if Y_probs.size <= top_n or np.max(Y_probs) < thresh: break
#		good_idx = np.argsort(Y_probs)[-top_n:]
#		bad_idx = np.argsort(Y_probs)[:top_n]
		good_idx = np.where(Y_probs >= thresh)[0]
		bad_idx = np.where(Y_probs < thresh)[0]
		if not good_idx.size: break
		iters = iters + 1
		nb_added = nb_added + good_idx.size
		X_all = sparse.vstack((X_all, X_working[good_idx,:]))
		Y_all = np.hstack((Y_all, Y_working[good_idx]))
		X_working = X_working[bad_idx,:]
	print "Done."
	print "Semi-supervised Learning:"
	print "   %d iterations" % iters
	print "   %d examples added to training corpus" % nb_added

	
	############################################
	################ Prediction ################
	############################################
	print "Predicting labels for unseen data....",
	sys.stdout.flush()
	bagged_Y_tst = bagged.predict(X_tst)
	print "Done."
	
	print "Writing to file......................",
	sys.stdout.flush()
	bagged_Y_tst = np.hstack((ids_tst, bagged_Y_tst[:,None]))
	bagged_Y_tst = np.vstack((['id','category'], bagged_Y_tst))
	np.savetxt('Y_tst_BAGGED.csv', bagged_Y_tst, delimiter=',', fmt='%s')
	print "Done.\a"





