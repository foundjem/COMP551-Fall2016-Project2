import sys, csv
import numpy as np
import pandas as pd

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier

from utils import load_sparse_csr
import uuid

if __name__ == '__main__':

	############################################
	################# Get data #################
	############################################
	print "Reading files........................"

	DATA_DIR = '100000'
	ROW = None 

	X_trn = load_sparse_csr('%s/X_trn_tfidf.npz' % DATA_DIR)
	X_val = load_sparse_csr('%s/X_val_tfidf.npz' % DATA_DIR)
#	X_all = load_sparse_csr('%s/X_all_tfidf.npz' % DATA_DIR)
#	X_tst = load_sparse_csr('%s/X_tst_tfidf.npz' % DATA_DIR)
	X_all = load_sparse_csr('%s/X_all_cheat_tfidf.npz' % DATA_DIR)
	X_tst = load_sparse_csr('%s/X_tst_cheat_tfidf.npz' % DATA_DIR)

	ids_trn, X_trn = X_trn[:,0].toarray().astype(int), X_trn[:,1:]
	ids_val, X_val = X_val[:,0].toarray().astype(int), X_val[:,1:]
	ids_all, X_all = X_all[:,0].toarray().astype(int), X_all[:,1:]
	ids_tst, X_tst = X_tst[:,0].toarray().astype(int), X_tst[:,1:]

	if ROW is not None:
		ids_trn, X_trn = ids_trn[:ROW,], X_trn[:ROW, ]
		ids_val, X_val = ids_val[:ROW,], X_val[:ROW, ]
		ids_all, X_all = ids_all[:ROW,], X_all[:ROW, ]
		ids_tst, X_tst = ids_tst[:ROW,], X_tst[:ROW, ]

	Y_trn = pd.read_csv('Y_trn.csv', usecols=[1], nrows = ROW).values.flatten()
	Y_val = pd.read_csv('Y_val.csv', usecols=[1], nrows = ROW).values.flatten()
	Y_all = pd.read_csv('Y_all.csv', usecols=[1], nrows = ROW).values.flatten()
	print "Done."

# 	print "Training classifier..................",
# 	sys.stdout.flush()
	model = RandomForestClassifier(n_estimators = 1000, max_features = 'sqrt', n_jobs = 6, criterion = 'entropy', warm_start = False)
# #	model = GridSearchCV(model,
# #						{'degree':[2,3,4],
# #						 'kernel':['linear','poly'],
# #						 'C':[0.001,0.01,0.1,1,10]},cv=10)
# 	model.fit(X_trn, Y_trn)
# #	model = model.best_estimator_
# 	print "Done."

	# print "Testing on validation data...........",
	# sys.stdout.flush()
	# accuracy = model.score(X_val, Y_val)
	# print "Done."

	print "Training on all data................."
	scores = cross_val_score(model, X_all, Y_all, cv=10)
	model.fit(X_all, Y_all)
	print "Done."
	print "Cross-validated accuracy: %f" % np.mean(scores)

	print "Predicting labels for unseen data...."
	Y_tst = model.predict(X_tst)
	print "Done."

	print "Writing to file......................"
	Y_tst = np.hstack((ids_tst, Y_tst[:,None]))
	Y_tst = np.vstack((['id','category'], Y_tst))

	new_id = uuid.uuid4().hex
	fname = 'Y_tst_lib_random_forest_%s_%s_acc_%s.csv' % (DATA_DIR, new_id, np.mean(scores))
	print "Saving to %s" % fname
	np.savetxt(fname, Y_tst, delimiter=',', fmt='%s')
	print "Done."
