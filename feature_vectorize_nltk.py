import sys, csv, re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk

tf_idf = True


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


class stem_tokenizer(object):
	def __init__(self, stemmer=nltk.stem.SnowballStemmer('english').stem):
		self.tokenizer = re.compile(r"(?u)\b[a-zA-Z]{2,}\b").findall
		self.stemmer = stemmer
	def __call__(self, doc):
		return [self.stemmer(t) for t in self.tokenizer(doc)]

class lemma_tokenizer(object):
	def __init__(self):
		self.tokenizer = re.compile(r"(?u)\b[a-zA-Z]{2,}\b").findall
		self.td ={'NN':'n','JJ':'a','VB':'v','RB':'r'}
		self.lmtzr = nltk.stem.WordNetLemmatizer().lemmatize
	def __call__(self, doc):
		tokens = self.tokenizer(doc)
		tags = [(w,t[:2]) for w,t in nltk.pos_tag(tokens)]
		tags = [(w,self.td[t] if t in self.td else 'n') for w,t in tags]
		return [self.lmtzr(*pair) for pair in tags]

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
	doc_cht = np.hstack((doc_all, doc_tst))
	
	lab_trn = lab_all[:v_split,1]
	lab_val = lab_all[v_split:,1]
	lab_all = lab_all[:,1]
	print "Done."

	print "Building vectorizer..................",
	sys.stdout.flush()
	vectorizer = CountVectorizer(tokenizer=lemma_tokenizer(),
								 stop_words='english',
								 lowercase=True,
								 max_features=2000,
								 min_df=1)
	ng_vectorizer = CountVectorizer(tokenizer=lemma_tokenizer(),
									ngram_range=(2,3),
									stop_words='english',
									lowercase=True,
									max_features=200,
									min_df=1)
	print "Done."
	
	print "Building TF-IDF transformer..........",
	sys.stdout.flush()
	transformer = TfidfTransformer()
	print "Done."

	print "Extracting training features.........",
	sys.stdout.flush()
	X_trn_counts = vectorizer.fit_transform(doc_trn)
	X_trn_counts = sparse.hstack((X_trn_counts,
								  ng_vectorizer.fit_transform(doc_trn)))
								  
	X_val_counts = vectorizer.transform(doc_val)
	X_val_counts = sparse.hstack((X_val_counts,
								  ng_vectorizer.transform(doc_val)))
								  
	header_trn = ['id']
	header_trn.extend(vectorizer.get_feature_names())
	header_trn.extend(ng_vectorizer.get_feature_names())
	header_trn = np.array(header_trn)
	print "Done."
	
	print "Extracting testing features..........",
	sys.stdout.flush()
	X_all_counts = vectorizer.fit_transform(doc_all)
	X_all_counts = sparse.hstack((X_all_counts,
								  ng_vectorizer.fit_transform(doc_all)))
								  
	X_tst_counts = vectorizer.transform(doc_tst)
	X_tst_counts = sparse.hstack((X_tst_counts,
								  ng_vectorizer.transform(doc_tst)))
								  
	header_all = ['id']
	header_all.extend(vectorizer.get_feature_names())
	header_all.extend(ng_vectorizer.get_feature_names())
	header_all = np.array(header_all)
	print "Done."
	
	print "Extracting cheat features............",
	sys.stdout.flush()
	vectorizer.fit(doc_cht)
	ng_vectorizer.fit(doc_cht)
	X_all_cheat_counts = vectorizer.transform(doc_all)
	X_all_cheat_counts = sparse.hstack((X_all_cheat_counts,
										ng_vectorizer.transform(doc_all)))
										
	X_tst_cheat_counts = vectorizer.transform(doc_tst)
	X_tst_cheat_counts = sparse.hstack((X_tst_cheat_counts,
										ng_vectorizer.transform(doc_tst)))
	header_cheat = ['id']
	header_cheat.extend(vectorizer.get_feature_names())
	header_cheat.extend(ng_vectorizer.get_feature_names())
	header_cheat = np.array(header_cheat)
	print "Done."
	
	print "Applying TF-IDF transform............",
	sys.stdout.flush()
	X_trn_tfidf = transformer.fit_transform(X_trn_counts)
	X_val_tfidf = transformer.transform(X_val_counts)
	
	X_all_tfidf = transformer.fit_transform(X_all_counts)
	X_tst_tfidf = transformer.transform(X_tst_counts)
	
	X_all_cheat_tfidf = transformer.fit_transform(X_all_cheat_counts)
	X_tst_cheat_tfidf = transformer.transform(X_tst_cheat_counts)
	print "Done."

	# Preppend IDs
	ids_trn = np.array(ids_trn).astype(int)[:,None]
	ids_val = np.array(ids_val).astype(int)[:,None]
	ids_all = np.array(ids_all).astype(int)[:,None]
	ids_tst = np.array(ids_tst).astype(int)[:,None]

	X_trn_counts = sparse.hstack((ids_trn, X_trn_counts))
	X_val_counts = sparse.hstack((ids_val, X_val_counts))
	X_all_counts = sparse.hstack((ids_all, X_all_counts))
	X_tst_counts = sparse.hstack((ids_tst, X_tst_counts))
	X_all_cheat_counts = sparse.hstack((ids_all, X_all_cheat_counts))
	X_tst_cheat_counts = sparse.hstack((ids_tst, X_tst_cheat_counts))

	X_trn_tfidf = sparse.hstack((ids_trn, X_trn_tfidf))
	X_val_tfidf = sparse.hstack((ids_val, X_val_tfidf))
	X_all_tfidf = sparse.hstack((ids_all, X_all_tfidf))
	X_tst_tfidf = sparse.hstack((ids_tst, X_tst_tfidf))
	X_all_cheat_tfidf = sparse.hstack((ids_all, X_all_cheat_tfidf))
	X_tst_cheat_tfidf = sparse.hstack((ids_tst, X_tst_cheat_tfidf))
	
	Y_trn = np.array(lab_trn)[:,None]
	Y_val = np.array(lab_val)[:,None]
	Y_all = np.array(lab_all)[:,None]

	Y_trn =	np.hstack((ids_trn, Y_trn))
	Y_val =	np.hstack((ids_val, Y_val))
	Y_all =	np.hstack((ids_all, Y_all))
	
	header_mat = np.vstack((header_trn, header_all, header_cheat)).T

	print "Writing to file......................",
	sys.stdout.flush()
#	fmt = '%f' if tf_idf else '%d'
#	ext = '_tfidf' if tf_idf else '_counts'
	save_sparse_csr('X_trn_counts', sparse.csr_matrix(X_trn_counts))
	save_sparse_csr('X_val_counts', sparse.csr_matrix(X_val_counts))
	save_sparse_csr('X_all_counts', sparse.csr_matrix(X_all_counts))
	save_sparse_csr('X_tst_counts', sparse.csr_matrix(X_tst_counts))
	save_sparse_csr('X_all_cheat_counts', sparse.csr_matrix(X_all_cheat_counts))
	save_sparse_csr('X_tst_cheat_counts', sparse.csr_matrix(X_tst_cheat_counts))
	save_sparse_csr('X_trn_tfidf', sparse.csr_matrix(X_trn_tfidf))
	save_sparse_csr('X_val_tfidf', sparse.csr_matrix(X_val_tfidf))
	save_sparse_csr('X_all_tfidf', sparse.csr_matrix(X_all_tfidf))
	save_sparse_csr('X_tst_tfidf', sparse.csr_matrix(X_tst_tfidf))
	save_sparse_csr('X_all_cheat_tfidf', sparse.csr_matrix(X_all_cheat_tfidf))
	save_sparse_csr('X_tst_cheat_tfidf', sparse.csr_matrix(X_tst_cheat_tfidf))
#	np.savetxt('X_trn'+ext, X_trn.toarray(), delimiter=',', fmt=fmt, header=header_trn)
#	np.savetxt('X_val'+ext, X_val.toarray(), delimiter=',', fmt=fmt, header=header_trn)
#	np.savetxt('X_all'+ext, X_all.toarray(), delimiter=',', fmt=fmt, header=header_all)
#	np.savetxt('X_tst'+ext, X_tst.toarray(), delimiter=',', fmt=fmt, header=header_all)
	np.savetxt('features.csv', header_mat, delimiter=',', fmt='%s',
			   header='training,testing,cheat')

	np.savetxt('Y_trn.csv', Y_trn, delimiter=',', fmt='%s', header='id,category')
	np.savetxt('Y_val.csv', Y_val, delimiter=',', fmt='%s', header='id,category')
	np.savetxt('Y_all.csv', Y_all, delimiter=',', fmt='%s', header='id,category')
	print "Done.\a" # Beep to let you know it's done
