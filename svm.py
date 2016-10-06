import sys, csv
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn import svm

if __name__ == '__main__':

	############################################
	################# Get data #################
	############################################
	print "Reading files........................",
	sys.stdout.flush()

	X_trn = np.genfromtxt('X_trn.csv', delimiter=',', dtype=int)
	X_val = np.genfromtxt('X_val.csv', delimiter=',', dtype=int)
	X_all = np.genfromtxt('X_all.csv', delimiter=',', dtype=int)
	X_tst = np.genfromtxt('X_tst.csv', delimiter=',', dtype=int)

	ids_trn, X_trn = X_trn[:,0][:,None], X_trn[:,1:]
	ids_val, X_val = X_val[:,0][:,None], X_val[:,1:]
	ids_all, X_all = X_all[:,0][:,None], X_all[:,1:]
	ids_tst, X_tst = X_tst[:,0][:,None], X_tst[:,1:]

	Y_trn = np.genfromtxt('Y_trn.csv', delimiter=',', dtype=str, usecols=[1])
	Y_val = np.genfromtxt('Y_val.csv', delimiter=',', dtype=str, usecols=[1])
	Y_all = np.genfromtxt('Y_all.csv', delimiter=',', dtype=str, usecols=[1])
	print "Done."

	print "Training classifier..................",
	sys.stdout.flush()
	model = svm.SVC(kernel='linear')
#	model = GridSearchCV(svm.SVC(),
#						{'degree':[2,3,4],
#						 'kernel':['linear','poly'],
#						 'C':[0.001,0.01,0.1,1,10]},cv=10)
	model.fit(X_trn, Y_trn)
#	model = model.best_estimator_
	print "Done."

	print "Testing on validation data...........",
	sys.stdout.flush()
	accuracy = model.score(X_val, Y_val)
	print "Done."

	print "Training on all data.................",
	sys.stdout.flush()
	scores = cross_val_score(model, X_all, Y_all, cv=10)
	model.fit(X_all, Y_all)
	print "Done."
	print "Validation set accuracy:  %f" % accuracy
	print "Cross-validated accuracy: %f" % np.mean(scores)
	
	print "Predicting labels for unseen data....",
	sys.stdout.flush()
	Y_tst = model.predict(X_tst)
	print "Done."
	
	print "Writing to file......................",
	sys.stdout.flush()
	Y_tst = np.hstack((ids_tst, Y_tst[:,None]))
	Y_tst = np.vstack((['id','category'], Y_tst))
	
	np.savetxt('Y_tst_SVM.csv', Y_tst, delimiter=',', fmt='%s')
	print "Done."