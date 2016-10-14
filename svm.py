import numpy as np
from sklearn import svm

if __name__ == '__main__':
	import sys
	import pandas as pd
	from scipy import sparse
	from sklearn.grid_search import GridSearchCV
	from sklearn.ensemble import BaggingClassifier
	from sklearn.cross_validation import cross_val_score
	from utils import load_sparse_csr, oversample, semi_supervised
	import warnings
	warnings.filterwarnings("ignore")
	
	rng = np.random.RandomState(42)

	############################################
	################# Get data #################
	############################################
	print "Reading files...........................",
	sys.stdout.flush()
	X_trn = load_sparse_csr('X_trn_tfidf.npz')
	X_val = load_sparse_csr('X_val_tfidf.npz')
	X_all = load_sparse_csr('X_all_tfidf.npz')
	X_tst = load_sparse_csr('X_tst_tfidf.npz')

	ids_trn, X_trn = X_trn[:,0].toarray().astype(int), X_trn[:,1:]
	ids_val, X_val = X_val[:,0].toarray().astype(int), X_val[:,1:]
	ids_all, X_all = X_all[:,0].toarray().astype(int), X_all[:,1:]
	ids_tst, X_tst = X_tst[:,0].toarray().astype(int), X_tst[:,1:]

	Y_trn = pd.read_csv('Y_trn.csv', usecols=[1]).values.flatten()
	Y_val = pd.read_csv('Y_val.csv', usecols=[1]).values.flatten()
	Y_all = pd.read_csv('Y_all.csv', usecols=[1]).values.flatten()
	print "Done."
	
	print "Oversampling data.......................",
	sys.stdout.flush()
	X_all, Y_all = oversample(X_all, Y_all, random_state=rng)
	print "Done."
	
	############################################
	################# Vanilla ##################
	############################################
	print "Vanilla SVM:"
	print "   Grid search for best parameters......",
	sys.stdout.flush()
	model = GridSearchCV(svm.LinearSVC(random_state=rng),
						 {'C':[0.001,0.01,0.1,1,10]}, n_jobs=-1,cv=10)
	model.fit(X_val, Y_val)
	model = model.best_estimator_
	print "Done."
	print "   Best parameter set:"
	for k in model.get_params().keys():
		print "      %s: %s" % (k,model.get_params()[k])

	print "   Testing on validation data...........",
	sys.stdout.flush()
	accuracy = model.fit(X_trn,Y_trn).score(X_val, Y_val)
	print "Done."

	print "   Training on all data.................",
	sys.stdout.flush()
	scores = cross_val_score(model, X_all, Y_all, cv=10, n_jobs=-1)
	model.fit(X_all, Y_all)
	print "Done."
	print "   Validation set accuracy:  %f" % accuracy
	print "   Cross-validated accuracy: %f" % np.mean(scores)
	
	print "   Predicting labels for unseen data....",
	sys.stdout.flush()
	Y_tst = model.predict(X_tst)
	print "Done."
	
	print "   Writing to file......................",
	sys.stdout.flush()
	Y_tst = np.hstack((ids_tst, Y_tst[:,None]))
	Y_tst = np.vstack((['id','category'], Y_tst))
	np.savetxt('Y_tst_SVM_VANILLA.csv', Y_tst, delimiter=',', fmt='%s')
	print "Done."

	############################################
	############# Semi-supervised ##############
	############################################
	print "Semi-supervised SVM:"
	print "   Using vanilla parameters"

	print "   Semi-supervised learning.............",
	sys.stdout.flush()
	model = semi_supervised(model, X_all, Y_all, X_tst, 0.9, model.decision_function)
	print "Done."
	
	print "   Predicting labels for unseen data....",
	sys.stdout.flush()
	Y_tst = model.predict(X_tst)
	print "Done."
	
	print "   Writing to file......................",
	sys.stdout.flush()
	Y_tst = np.hstack((ids_tst, Y_tst[:,None]))
	Y_tst = np.vstack((['id','category'], Y_tst))
	np.savetxt('Y_tst_SVM_SEMI_SUPERVISED.csv', Y_tst, delimiter=',', fmt='%s')
	print "Done."

	############################################
	################# Ensemble #################
	############################################
	print "Ensemble of SVMs:"
	print "   Using vanilla parameters"

	print "   Building ensemble....................",
	sys.stdout.flush()
	n_estimators = 100
	model = BaggingClassifier(model, n_estimators=n_estimators,
							  bootstrap = True, n_jobs=-1, random_state=rng)
	print "Done."

	print "   Testing on validation data...........",
	sys.stdout.flush()
	accuracy = model.fit(X_trn,Y_trn).score(X_val, Y_val)
	print "Done."

	print "   Training on all data.................",
	sys.stdout.flush()
	scores = cross_val_score(model, X_all, Y_all, cv=10)
	model.fit(X_all, Y_all)
	print "Done."
	print "   Validation set accuracy:  %f" % accuracy
	print "   Cross-validated accuracy: %f" % np.mean(scores)
	
	print "   Predicting labels for unseen data....",
	sys.stdout.flush()
	Y_tst = model.predict(X_tst)
	print "Done."
	
	print "   Writing to file......................",
	sys.stdout.flush()
	Y_tst = np.hstack((ids_tst, Y_tst[:,None]))
	Y_tst = np.vstack((['id','category'], Y_tst))
	np.savetxt('Y_tst_SVM_ENSEMBLE.csv', Y_tst, delimiter=',', fmt='%s')
	print "Done."
	
	############################################
	######### Semi-supervised Ensemble #########
	############################################
	print "Semi-supervised Ensemble of SVMs:"
	print "   Using vanilla parameters"

	print "   Semi-supervised learning.............",
	sys.stdout.flush()
	model = semi_supervised(model, X_all, Y_all, X_tst, 0.9, model.decision_function)
	print "Done."
	
	print "   Predicting labels for unseen data....",
	sys.stdout.flush()
	Y_tst = model.predict(X_tst)
	print "Done."
	
	print "   Writing to file......................",
	sys.stdout.flush()
	Y_tst = np.hstack((ids_tst, Y_tst[:,None]))
	Y_tst = np.vstack((['id','category'], Y_tst))
	np.savetxt('Y_tst_SVM_SS_ENSEMBLE.csv', Y_tst, delimiter=',', fmt='%s')
	print "Done.\a"