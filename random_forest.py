import sys
import math
import random
import collections
from multiprocessing import Process, Queue
import time
import uuid
import argparse

import numpy as np
from sklearn.grid_search import GridSearchCV
from utils import load_sparse_csr
import pandas as pd
from scipy import sparse

def execute_in_parallel(lambda_list, args, timeout_seconds = None, max_worker = 8):
	"""
		Execute a list of functions in parallel.
	"""
	all_processes = []
	for i, l in enumerate(lambda_list):
		p = Process(target=l, args = (args[i], ))
		all_processes.append(p)
		p.start()

	for p in all_processes:
		p.join()

MID_VALUE = 0.5

total_time = 0

log2 = lambda x : math.log(x, 2)

def entropy(counts, total):
	if len(counts) == 0 or total == 0:
		return 0

	assert sum(counts) == total

	output = 0
	total = float(total)
	for count in counts:
		p = count / total
		output += - p * log2(p)

	return output

class Decision(object):
	"""A Decision"""
	def __init__(self, feature_index, threshold):
		super(Decision, self).__init__()
		self.feature_index = feature_index
		self.threshold = threshold # Value greater than or equals to this threshold is considered true and false otherwise

	def test(self, x):
		return x[self.feature_index] >= self.threshold

	def split(self, data):
		# true = np.where(data[:,self.feature_index].A >= self.threshold)[0]
		# false = np.where(data[:,self.feature_index].A < self.threshold)[0]

		bool_vec = data[:,self.feature_index].A >= self.threshold
		true = np.where(bool_vec)[0]
		false = np.where(np.logical_not(bool_vec))[0]

		return true, false

	def __str__(self):
		return "Decision[index={0}, threshold={1}".format(self.feature_index, self.threshold)

class TreeNode(object):
	"""docstring for TreeNode"""
	def __init__(self, X, y):
		super(TreeNode, self).__init__()
		self.X = X
		self.y = y
		self._size = len(self.y)

		self.decision = None
		self.true = None
		self.false = None

	def size(self):
		return self._size

	def is_leaf(self):
		return self.decision is None and self.true is None and self.false is None

	def predict(self, x):
		assert self.is_leaf()
		if len(self.y) > 0:
			return collections.Counter(self.y).most_common()[0][0]
		else: # We have no data in this node, so the best we can do is random.
			return random.choice(['math', 'cs', 'stat', 'physics'])

	def split(self, decision):
		s = time.time()
		true, false = decision.split(self.X)
		global total_time
		total_time += time.time() - s

		s = time.time()
		true_X = self.X[true]
		total_time += time.time() - s
		true_y = self.y[true.flatten()]
#        true_X = np.array(map(lambda index : self.X[index], true))
#        true_y = list(map(lambda index : self.y[index], true))

		s = time.time()
		false_X = self.X[false]
		false_y = self.y[false.flatten()]
		total_time += time.time() - s

#        false_X = np.array(map(lambda index : self.X[index], false))
#        false_y = list(map(lambda index : self.y[index], false))

		self.decision = decision
		self.true = TreeNode(true_X, true_y)
		self.false = TreeNode(false_X, false_y)


	def entropy_root(self):
		"""
			Calculating entropy of this node
		"""
		if self.size() == 0:
			return 0

		counter = collections.Counter(self.y)
		return entropy([x[1] for x in counter.items()], self.size())

	def entropy_children(self):
		"""
			Calculating weighted entropy of the children nodes.
		"""
		assert self.true
		assert self.false

		my_size = float(self.size())
		return (self.true.size() / my_size) * self.true.entropy_root() + (self.false.size() / my_size) * self.false.entropy_root()


def min_sparse(X):
	# return np.min(X)

	if len(X.data) == 0:
		return 0
	m = X.data.min()
	return m if X.getnnz() == X.size else min(m, 0)

def max_sparse(X):
	# return np.max(X)

	if len(X.data) == 0:
		return 0
	m = X.data.max()
	return m if X.getnnz() == X.size else max(m, 0)

class RandomForest(object):
	"""Implementation of random forest model"""
	def __init__(self, metadata = None, k = 5, m = 10, min_node_size = 20):
		super(RandomForest, self).__init__()
		self.k = k
		self.m = m

		# Minimum size of a node. Stop adding decision if the node size is below this size
		self.min_node_size = min_node_size

		# Metadata contains the information about each input feature: either 'b' for boolean or 'c' for continuous
		self.metadata = metadata

	def get_params(self, deep=True):
		return { 'k' : self.k, 'm' : self.m }

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self, parameter, value)

		assert self.m <= len(self.metadata)
		return self

	def _build_decision(self, feature_index, data):
		assert feature_index < len(self.metadata)
		feature_type = self.metadata[feature_index]
		if feature_type == 'b':
			return Decision(feature_index, MID_VALUE)
		else:
			min_value = min_sparse(data[:,feature_index])
			max_value = max_sparse(data[:,feature_index])

			return Decision(feature_index, random.uniform(min_value, max_value))

	def fit(self, X, y):
		assert len(self.metadata) == X.shape[1]
		print "Random forest object with k = {0}, m = {1} and min_node_size = {2}".format(self.k, self.m , self.min_node_size)

		self.root_nodes = []
		# non_zero_leaf_nodes = [[node] for node in self.root_nodes] # List of stacks of nodes to consider
		self.tree_count = 0

		def job(root_node):
			def inner():
				non_zero_leaf_nodes = [root_node]

				while len(non_zero_leaf_nodes) > 0:
					# Pick m features randomly
					selected_features = random.sample(xrange(X.shape[1]), self.m)

					# Pick the leaf node from the stack
					leaf_node = non_zero_leaf_nodes.pop()
					if leaf_node.size() < self.min_node_size:
						continue

					current_entropy = leaf_node.entropy_root()


					min_entropy = 9999999
					min_decision = None

					# From m features construct m tests and select the best one based on entropy
					for feature_index in selected_features:
						decision = self._build_decision(feature_index, leaf_node.X)
						leaf_node.split(decision)

						new_entropy = leaf_node.entropy_children()

						if new_entropy < min_entropy:
							min_entropy = new_entropy
							min_decision = decision

						if new_entropy == 0: # This has to be the smallest
							break

					assert min_decision is not None
					if min_entropy >= current_entropy:
						# print "Warning: Entropy not decreasing"
						continue
					leaf_node.split(min_decision) # Assign the chosen decision to this node

					# Then delete data from the node to save memory (since data is already saved in the node's children)
					leaf_node.X = None
					leaf_node.y = None

					if leaf_node.true.entropy_root() > 0:
						non_zero_leaf_nodes.append(leaf_node.true)

					if leaf_node.false.entropy_root() > 0:
						non_zero_leaf_nodes.append(leaf_node.false)

				self.tree_count += 1
				print "\r==> Done %s" % self.tree_count
				sys.stdout.flush()
			return inner

		jobs = []
		for tree_index in xrange(self.k):
			new_root = TreeNode(X, y)
			job(new_root)()
			self.root_nodes.append(new_root)

		print "Done training"

	def predict_single(self, x):
		results = collections.Counter()
		for root in self.root_nodes:
			node = root
			while True: # Go down the tree
				if node.is_leaf():
					results[node.predict(x)] += 1
					break
				elif node.decision.test(x):
					node = node.true
				else:
					node = node.false

		return results.most_common()[0][0]

	def predict(self, X):
		return [self.predict_single(x.A.flatten()) for x in X]

	def score(self, X, y):
		prediction = self.predict(X)
		accurate_count = np.sum(prediction == y)
	   # accurate_count = sum(1 for index, value in enumerate(prediction) if value == y[index])

		accuracy = float(accurate_count) / len(y)
		print "Accuracy: {0}".format(accuracy)
		return accuracy


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = 'Random forest')

	parser.add_argument('-f', '--feature-directory', dest = 'feature_directory', default = '55000', help = 'Select custom feature directory. Default to 55000.')
	parser.add_argument('-k', dest = 'k', default = 10, help = 'Value of k. Default to 10.', type = int)
	parser.add_argument('-l', '--limit', dest = 'limit', default = False, action='store_true', help = 'Limiting the number of samples to train and validate on. Default to False (no limit).')
	parser.add_argument('-m', dest = 'm', default = 10, help = 'Value of m. Default to 10.', type = int)
	parser.add_argument('-s', '--min-node-size', dest = 'min_node_size', default = 20, help = 'Value of minimum size of a node in the trained tree. Default to 20.', type = int)
	parser.add_argument('-t', '--test', dest = 'test', default = False, action='store_true', help = 'Perform prediction on test set and save result.')
	parser.add_argument('-v', '--no-validation', dest = 'validation', default = True, action='store_false', help = 'Skip training on train set and validation.')
	args = parser.parse_args()


	start_time = time.time()
	############################################
	################# Get data #################
	############################################
	print "Reading files........................"

#    X_trn = np.genfromtxt('extracted_features/X_trn.csv', delimiter=',', dtype=int)
#    X_val = np.genfromtxt('extracted_features/X_val.csv', delimiter=',', dtype=int)
#    # X_all = np.genfromtxt('extracted_features/X_all.csv', delimiter=',', dtype=int)
#    X_tst = np.genfromtxt('extracted_features/X_tst.csv', delimiter=',', dtype=int)

	ROW_COUNT = 2000
	X_trn = load_sparse_csr('{0}/X_trn_tfidf.npz'.format(args.feature_directory))
	X_val = load_sparse_csr('{0}/X_val_tfidf.npz'.format(args.feature_directory))
	X_all = load_sparse_csr('{0}/X_all_tfidf.npz'.format(args.feature_directory))
	X_tst = load_sparse_csr('{0}/X_tst_tfidf.npz'.format(args.feature_directory))
#	X_all = load_sparse_csr('X_all_cheat_tfidf.npz')
#	X_tst = load_sparse_csr('X_tst_cheat_tfidf.npz')
	if args.limit:
		X_trn = X_trn[ : ROW_COUNT, ]
		X_val = X_val[ : ROW_COUNT, ]
		X_all = X_all[ : ROW_COUNT, ]
		X_tst = X_tst[ : ROW_COUNT, ]

	ids_trn, X_trn = X_trn[:,0].toarray().astype(int), X_trn[:,1:]
	ids_val, X_val = X_val[:,0].toarray().astype(int), X_val[:,1:]
	ids_all, X_all = X_all[:,0].toarray().astype(int), X_all[:,1:]
	ids_tst, X_tst = X_tst[:,0].toarray().astype(int), X_tst[:,1:]

#    ids_trn, X_trn = X_trn[:,0][:,None], X_trn[:,1:]
#    ids_val, X_val = X_val[:,0][:,None], X_val[:,1:]
#    # ids_all, X_all = X_all[:,0][:,None], X_all[:,1:]
#    ids_tst, X_tst = X_tst[:,0][:,None], X_tst[:,1:]

	features_count = X_trn.shape[1]
#
#    Y_trn = np.genfromtxt('extracted_features/Y_trn.csv', delimiter=',', dtype=str, usecols=[1])
#    Y_val = np.genfromtxt('extracted_features/Y_val.csv', delimiter=',', dtype=str, usecols=[1])
#    # Y_all = np.genfromtxt('extracted_features/Y_all.csv', delimiter=',', dtype=str, usecols=[1])

	Y_trn = pd.read_csv('Y_trn.csv', usecols=[1], nrows = ROW_COUNT if args.limit else None).values.flatten()
	Y_val = pd.read_csv('Y_val.csv', usecols=[1], nrows = ROW_COUNT if args.limit else None).values.flatten()
	Y_all = pd.read_csv('Y_all.csv', usecols=[1], nrows = ROW_COUNT if args.limit else None).values.flatten()

	print "Done."

	model = RandomForest(['c' for i in xrange(features_count)], k = args.k, m = args.m, min_node_size = args.min_node_size)
	# model = GridSearchCV(model, {'k':[10,20,30,50,100], 'm': [10, 20, 30, 40], 'min_node_size' : [20, 50], 'metadata' : [['c' for i in xrange(features_count)]]}, cv = 10)

	if args.validation:
		print "Training classifier.................."
		model.fit(X_trn, Y_trn)
		print "Done."

		print "Testing on validation data...........",
		sys.stdout.flush()

		accuracy = model.score(X_val, Y_val)
		print "Done."
		print "Profiled total time %s" % total_time # Run time for a particular block that we wish to profile
		print "Total run time %s" % (time.time() - start_time) # Total run time of the whole process

	if args.test:
		training_start = time.time()
		print "Training on X_all......................."
		model.fit(X_all, Y_all)
		print "Done training on X_all. This training took {0}s".format(time.time() - training_start)

		print "Predicting on test data.................",
		sys.stdout.flush()
		test_prediction = model.predict(X_tst)
		new_id = uuid.uuid4().hex
		with open('out_random_forest_%s.csv' % new_id, 'w') as f:
			np.savetxt(f, np.c_[ids_tst.flatten(), test_prediction], delimiter=',', fmt='%s')
		print "Saved test prediction to out_random_forest_%s.csv" % new_id
