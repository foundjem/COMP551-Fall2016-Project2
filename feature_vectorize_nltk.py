import sys, csv
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer



def read_data(file):
	with open(file, 'rt') as f:
		reader = csv.reader(f)
		header = next(reader)
		data = list(reader)
	header = {h:i for i,h in enumerate(header)}
	return header, data

def write_data(file, header, data):
	with open(file,'wb') as f:
		writer = csv.writer(f, delimiter=',')
		writer.writerow(header)
		for row in data:
			writer.writerow(row)





# Main driver code
if __name__ == '__main__':

	############################################
	############# Token extraction #############
	############################################
	print "Reading files........................",
	sys.stdout.flush()
	h1, l1 = read_data('datasets/train_in.csv')
	h2, l2 = read_data('datasets/train_out.csv')
	h3, l3 = read_data('datasets/test_in.csv')
	ids_all, doc_all = zip(*l1)
	ids_all, lab_all = zip(*l2)
	ids_tst, doc_tst = zip(*l3)

	# Split into training and validation sets
	v_split = int(0.8 * len(ids_all))
	ids_trn, ids_val = ids_all[:v_split], ids_all[v_split:]
	doc_trn, doc_val = doc_all[:v_split], doc_all[v_split:]
	lab_trn, lab_val = lab_all[:v_split], lab_all[v_split:]
	
	print "Done."

	print "Building vectorizer..................",
	sys.stdout.flush()
	vectorizer = CountVectorizer(ngram_range=(1,2),
								 stop_words='english',
								 lowercase=True,
								 max_features=1000,
								 min_df=1)
	print "Done."

	print "Extracting training features.........",
	sys.stdout.flush()
	X_trn = vectorizer.fit_transform(doc_trn)
	X_val = vectorizer.transform(doc_val)
	print "Done."
	
	print "Extracting testing features..........",
	sys.stdout.flush()
	X_all = vectorizer.fit_transform(doc_all)
	X_tst = vectorizer.transform(doc_tst)
	print "Done."
	
	ids_trn = np.array(ids_trn).astype(int)[:,None]
	ids_val = np.array(ids_val).astype(int)[:,None]
	ids_all = np.array(ids_all).astype(int)[:,None]
	ids_tst = np.array(ids_tst).astype(int)[:,None]
	
	Y_trn = np.array(lab_trn)[:,None]
	Y_val = np.array(lab_val)[:,None]
	Y_all = np.array(lab_all)[:,None]

	X_trn =	sparse.hstack((ids_trn, X_trn))
	X_val =	sparse.hstack((ids_val, X_val))
	X_all =	sparse.hstack((ids_all, X_all))
	X_tst =	sparse.hstack((ids_tst, X_tst))

	Y_trn =	np.hstack((ids_trn, Y_trn))
	Y_val =	np.hstack((ids_val, Y_val))
	Y_all =	np.hstack((ids_all, Y_all))

	print "Writing to file......................",
	sys.stdout.flush()
	np.savetxt('X_trn.csv', X_trn.toarray(), delimiter=',', fmt='%d')
	np.savetxt('X_val.csv', X_val.toarray(), delimiter=',', fmt='%d')
	np.savetxt('X_all.csv', X_all.toarray(), delimiter=',', fmt='%d')
	np.savetxt('X_tst.csv', X_tst.toarray(), delimiter=',', fmt='%d')
	
	np.savetxt('Y_trn.csv', Y_trn, delimiter=',', fmt='%s')
	np.savetxt('Y_val.csv', Y_val, delimiter=',', fmt='%s')
	np.savetxt('Y_all.csv', Y_all, delimiter=',', fmt='%s')
	print "Done."