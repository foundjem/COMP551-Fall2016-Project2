import sys, csv
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

tf_idf = False

def read_data(file):
	with open(file, 'rt') as f:
		reader = csv.reader(f)
		header = next(reader)
		data = list(reader)
	header = {h:i for i,h in enumerate(header)}
	return header, data

# Main driver code
if __name__ == '__main__':

	############################################
	############# Token extraction #############
	############################################
	print "Reading files........................",
	sys.stdout.flush()
	dat_all = pd.read_csv('datasets/train_in.csv').values
	dat_tst = pd.read_csv('datasets/test_in.csv').values
	lab_all = pd.read_csv('datasets/train_out.csv').values
	
	# Get rid of invalid rows
	keepers = lab_all[:,1] != 'category'
	dat_all = dat_all[keepers,:]
	lab_all = lab_all[keepers,:]
	
	# Split into training and validation sets
	v_split = int(0.8 * dat_all.shape[0])
	ids_trn = dat_all[:v_split,0].astype(int)
	ids_val = dat_all[v_split:,0].astype(int)
	ids_all = dat_all[:,0].astype(int)
	ids_tst = dat_tst[:,0].astype(int)
	
	doc_trn = dat_all[:v_split,1]
	doc_val = dat_all[v_split:,1]
	doc_all = dat_all[:,1]
	doc_tst = dat_tst[:,1]
	
	lab_trn = lab_all[:v_split,1]
	lab_val = lab_all[v_split:,1]
	lab_all = lab_all[:,1]
	print "Done."

	print "Building vectorizer..................",
	sys.stdout.flush()
	vectorizer = CountVectorizer(ngram_range=(1,2),
								 stop_words='english',
								 lowercase=True,
								 max_features=1000,
								 min_df=1)
	ng_vectorizer = CountVectorizer(ngram_range=(2,3),
									stop_words='english',
									lowercase=True,
									max_features=400,
									min_df=1)
	print "Done."

	print "Extracting training features.........",
	sys.stdout.flush()
	X_trn = vectorizer.fit_transform(doc_trn)
	X_trn = sparse.hstack((X_trn, ng_vectorizer.fit_transform(doc_trn)))
	X_val = vectorizer.transform(doc_val)
	X_val = sparse.hstack((X_val, ng_vectorizer.transform(doc_val)))
	print "Done."
	
	print "Extracting testing features..........",
	sys.stdout.flush()
	X_all = vectorizer.fit_transform(doc_all)
	X_all = sparse.hstack((X_all, ng_vectorizer.fit_transform(doc_all)))
	X_tst = vectorizer.transform(doc_tst)
	X_tst = sparse.hstack((X_tst, ng_vectorizer.transform(doc_tst)))
	print "Done."
	
	if tf_idf:
		print "Building TF-IDF transformer..........",
		sys.stdout.flush()
		transformer = TfidfTransformer()
		print "Done."
		
		print "Applying TF-IDF transform............",
		sys.stdout.flush()
		X_trn = transformer.fit_transform(X_trn)
		X_val = transformer.transform(X_val)
		X_all = transformer.fit_transform(X_all)
		X_tst = transformer.transform(X_tst)
		print "Done."
	
#	print "Features:"
#	for f in vectorizer.get_feature_names(): print "   "+f
#	for f in ng_vectorizer.get_feature_names(): print "   "+f

	# Preppend IDs
	ids_trn = np.array(ids_trn).astype(int)[:,None]
	ids_val = np.array(ids_val).astype(int)[:,None]
	ids_all = np.array(ids_all).astype(int)[:,None]
	ids_tst = np.array(ids_tst).astype(int)[:,None]

	X_trn =	sparse.hstack((ids_trn, X_trn))
	X_val =	sparse.hstack((ids_val, X_val))
	X_all =	sparse.hstack((ids_all, X_all))
	X_tst =	sparse.hstack((ids_tst, X_tst))
	
	Y_trn = np.array(lab_trn)[:,None]
	Y_val = np.array(lab_val)[:,None]
	Y_all = np.array(lab_all)[:,None]

	Y_trn =	np.hstack((ids_trn, Y_trn))
	Y_val =	np.hstack((ids_val, Y_val))
	Y_all =	np.hstack((ids_all, Y_all))

	print "Writing to file......................",
	sys.stdout.flush()
	fmt = '%f' if tf_idf else '%d'
	np.savetxt('X_trn.csv', X_trn.toarray(), delimiter=',', fmt=fmt)
	np.savetxt('X_val.csv', X_val.toarray(), delimiter=',', fmt=fmt)
	np.savetxt('X_all.csv', X_all.toarray(), delimiter=',', fmt=fmt)
	np.savetxt('X_tst.csv', X_tst.toarray(), delimiter=',', fmt=fmt)
	
	np.savetxt('Y_trn.csv', Y_trn, delimiter=',', fmt='%s')
	np.savetxt('Y_val.csv', Y_val, delimiter=',', fmt='%s')
	np.savetxt('Y_all.csv', Y_all, delimiter=',', fmt='%s')
	print "Done."