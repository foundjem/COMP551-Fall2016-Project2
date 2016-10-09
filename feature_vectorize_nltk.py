import sys, csv, re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk

tf_idf = True

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

	print "Extracting training features.........",
	sys.stdout.flush()
	X_trn = vectorizer.fit_transform(doc_trn)
	X_trn = sparse.hstack((X_trn, ng_vectorizer.fit_transform(doc_trn)))
	X_val = vectorizer.transform(doc_val)
	X_val = sparse.hstack((X_val, ng_vectorizer.transform(doc_val)))
	header_trn = ['id'] + vectorizer.get_feature_names() + ng_vectorizer.get_feature_names()
	header_trn = ','.join(header_trn)
	print "Done."
	
	print "Extracting testing features..........",
	sys.stdout.flush()
	X_all = vectorizer.fit_transform(doc_all)
	X_all = sparse.hstack((X_all, ng_vectorizer.fit_transform(doc_all)))
	X_tst = vectorizer.transform(doc_tst)
	X_tst = sparse.hstack((X_tst, ng_vectorizer.transform(doc_tst)))
	header_all = ['id'] + vectorizer.get_feature_names() + ng_vectorizer.get_feature_names()
	header_all = ','.join(header_all)
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
	ext = '_tfidf.csv' if tf_idf else '.csv'
	np.savetxt('X_trn'+ext, X_trn.toarray(), delimiter=',', fmt=fmt, header=header_trn)
	np.savetxt('X_val'+ext, X_val.toarray(), delimiter=',', fmt=fmt, header=header_trn)
	np.savetxt('X_all'+ext, X_all.toarray(), delimiter=',', fmt=fmt, header=header_all)
	np.savetxt('X_tst'+ext, X_tst.toarray(), delimiter=',', fmt=fmt, header=header_all)
	
	np.savetxt('Y_trn.csv', Y_trn, delimiter=',', fmt='%s', header='id,category')
	np.savetxt('Y_val.csv', Y_val, delimiter=',', fmt='%s', header='id,category')
	np.savetxt('Y_all.csv', Y_all, delimiter=',', fmt='%s', header='id,category')
	print "Done.\a" # Beep to let you know it's done
