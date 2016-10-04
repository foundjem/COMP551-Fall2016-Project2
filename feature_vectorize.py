'''
Created on Jul 14, 2015

@author: jcheung
'''

import sys, csv
import operator
import collections
import sklearn
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn import svm, linear_model, naive_bayes
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix

stoplist = set(stopwords.words('english'))
lmtzr = WordNetLemmatizer()

# model parameters
n = 2
limit_features = True
nb_features_per_ngram = [2000,200,40,0,0]
lemma = True
lower = True
stop = True


def read_data(file):
	with open(file, 'rt') as f:
		reader = csv.reader(f)
		header = next(reader)
		data = list(reader)
	header = {h:i for i,h in enumerate(header)}
	return header, data

def write_data(file, data):
	with open(file,'wb') as f:
		writer = csv.writer(f, delimiter=',')
		for row in data:
			writer.writerow(row)

def get_tokens(lines):
	""" Tokenize file into words. """
	paras = [[s for s in sent_tokenize(l)] for l in lines]
	# Remove punctuation
	not_punct = lambda c: c.isalnum() or c == ' ' or c == '-'
	paras = [[''.join([c for c in s if not_punct(c)]) for s in l] for l in paras]
	return [[t for s in sents for t in word_tokenize(s)] for sents in paras]

def treat_tokens(tokens, lemma, lower, stop):
	""" Treat the tokens according to the provided rules
		lemma: (boolean) whether or not to lemmatize
		lower: (boolean) whether or not to lowercase everything
		stop:  (boolean) whether or not to filter out stop words
		"""
	if lower:
		tokens = [t.lower() for t in tokens]
	if stop:
		tokens = [t for t in tokens if t not in stoplist]
	if lemma:
		tokens = lemmatize(tokens)
	return tokens

def lemmatize(tokens):
	""" Lemmatize the given set of tokens """
	tag_dict ={'NN':'n','JJ':'a','VB':'v','RB':'r'}
	tags = zip(*nltk.pos_tag(tokens))[1]
	
	# Convert tags to 'n','a','v','r'
	tags = [tag_dict[p[:2]] if p[:2] in tag_dict else 'n' for p in tags]
	return [lmtzr.lemmatize(*pair) for pair in zip(tokens,tags)]

def get_ngrams(tokens, N):
	""" Get all ngrams for n <= N 
		Returns list of lists. Kth list contains all K-grams.
		"""
	lists = [tokens[n:] for n in range(N)]
	return [zip(*lists[:n+1]) for n in range(N)]

def feature_vector(feature_tokens, tokens):
	""" Extracts features from tokens into a feature vector """
	return [tokens.count(t) for t in feature_tokens]

def sort_by_count(tokens):
	""" Gives unique tokens sorted descending by number of occurrences """
	counts = collections.Counter(tokens)
	return sorted(counts.keys(), key=counts.get, reverse=True)


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
	id1, train_lines = zip(*l1)
	id2, train_labels = zip(*l2)
	id3, test_lines = zip(*l3)
	print "Done."
	
	print "Extracting tokens....................",
	sys.stdout.flush()
	train_tokens = get_tokens(train_lines)
	test_tokens = get_tokens(test_lines)
	print "Done."
	
	print "Treating tokens......................",
	sys.stdout.flush()
	train_tokens = [treat_tokens(doc, lemma, lower, stop) for doc in train_tokens]
	test_tokens = [treat_tokens(doc, lemma, lower, stop) for doc in test_tokens]
	print "Done."
	
	
	print "Treating labels......................",
	sys.stdout.flush()
	label_codes = {l:i for i,l in enumerate(list(set(train_labels)))}
#	train_labels = [label_codes[l] for l in train_labels]
	print "Done."
	
	

	############################################
	############ N-gram calculation ############
	############################################
	print "Constructing N-grams.................",
	sys.stdout.flush()
	train_ngrams = [get_ngrams(tokens,n) for tokens in train_tokens]
	test_ngrams = [get_ngrams(tokens,n) for tokens in test_tokens]
	print "Done"
	
	v_split = int(0.8 * len(train_labels))
	t_ids, v_ids = id1[:v_split], id1[v_split:]
	t_labels, v_labels = train_labels[:v_split], train_labels[v_split:]
	t_ngrams, v_ngrams = train_ngrams[:v_split], train_ngrams[v_split:]
	
	print "Merging interdocument N-grams........",
	sys.stdout.flush()
	all_ngrams = [[g for doc in kgrams for g in doc] for kgrams in zip(*train_ngrams)]
	all_t_ngrams = [[g for doc in kgrams for g in doc] for kgrams in zip(*t_ngrams)]
	print "Done"
	
	print "Merging intradocument N-grams........",
	sys.stdout.flush()
	t_ngrams = [[g for ngrams in doc for g in ngrams] for doc in t_ngrams]
	v_ngrams = [[g for ngrams in doc for g in ngrams] for doc in v_ngrams]
	train_ngrams = [[g for ngrams in doc for g in ngrams] for doc in train_ngrams]
	test_ngrams = [[g for ngrams in doc for g in ngrams] for doc in test_ngrams]
	print "Done"
	

	############################################
	############ Feature selection #############
	############################################
	print "Selecting feature N-grams............",
	sys.stdout.flush()
	
	feature_t_ngrams = []
	feature_ngrams = []
	if limit_features:
		for k,kgrams in enumerate(all_t_ngrams):
			common = sort_by_count(kgrams)[:nb_features_per_ngram[k]]
			feature_t_ngrams.extend(common)
		
		for k,kgrams in enumerate(all_ngrams):
			common = sort_by_count(kgrams)[:nb_features_per_ngram[k]]
			feature_ngrams.extend(common)
	else:
		feature_t_ngrams = [k for kgrams in all_t_ngrams for k in kgrams]
		feature_ngrams = [k for kgrams in all_ngrams for k in kgrams]
	print "Done"
	


	############################################
	######## Feature vector construction #######
	############################################
	print "Constructing feature vectors.........",
	sys.stdout.flush()
	Xt = [feature_vector(feature_t_ngrams, ngrams) for ngrams in t_ngrams]
	Yt = t_labels
	
	Xv = [feature_vector(feature_t_ngrams, ngrams) for ngrams in v_ngrams]
	Yv = v_labels
	
	X = [feature_vector(feature_ngrams, ngrams) for ngrams in train_ngrams]
	Y = train_labels
	
	XT = [feature_vector(feature_ngrams, ngrams) for ngrams in test_ngrams]
	print "Done."


	print "Writing to file......................",
	sys.stdout.flush()
	write_data('train_in.csv', [['ID'] + feature_t_ngrams] + zip(*[t_ids]+zip(*Xt)))
	write_data('train_out.csv', [['ID','label']] + zip(t_ids, Yt))
	
	write_data('validation_in.csv', [['ID'] + feature_t_ngrams] + zip(*[v_ids]+zip(*Xv)))
	write_data('validation_out.csv', [['ID','label']] + zip(v_ids, Yv))
	
	write_data('all_train_in.csv', [['ID'] + feature_ngrams] + zip(*[id1]+zip(*X)))
	write_data('all_train_out.csv', [['ID','label']] + zip(id1, Y))
	
	write_data('test_in.csv', [['ID'] + feature_ngrams] + zip(*[id3]+zip(*XT)))
	print "Done."

