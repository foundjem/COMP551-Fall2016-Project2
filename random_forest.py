import math
import random

import numpy as np

POSITIVE_VALUE = 1
NEGATIVE_VALUE = 0
MID_VALUE = 0.5

log2 = lambda x : math.log(x, 2)

def entropy(p, n):
    if p == 0 or n == 0:
        return 0

    p_true = float(p) / (p + n)
    p_false = 1 - p_true
    return - p_true * log2(p_true) - p_false * log2(p_false)

class Decision(object):
    """A Decision"""
    def __init__(self, feature_index, threshold):
        super(Decision, self).__init__()
        self.feature_index = feature_index
        self.threshold = threshold # Value greater than or equals to this threshold is considered true and false otherwise

    def test(self, x):
        return x[self.feature_index] >= self.threshold

    def split(self, data):
        true = np.array([index for index, line in enumerate(data) if line[self.feature_index] >= self.threshold])
        false = np.array([index for index, line in enumerate(data) if line[self.feature_index] < self.threshold])

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
            return self.y[0]
        else: # We have no data in this node, so the best we can do is random.
            return random.choice([NEGATIVE_VALUE,POSITIVE_VALUE])

    def split(self, decision):
        true, false = decision.split(self.X)

        true_X = np.array(map(lambda index : self.X[index], true))
        true_y = list(map(lambda index : self.y[index], true))

        false_X = np.array(map(lambda index : self.X[index], false))
        false_y = list(map(lambda index : self.y[index], false))

        self.decision = decision
        self.true = TreeNode(true_X, true_y)
        self.false = TreeNode(false_X, false_y)

    def entropy_root(self):
        """
            Calculating entropy of this node
        """
        if self.size() == 0:
            return 0

        positive = sum(1 for value in self.y if value == POSITIVE_VALUE)
        negative = self.size() - positive

        return entropy(positive, negative)

    def entropy_children(self):
        """
            Calculating weighted entropy of the children nodes.
        """
        assert self.true
        assert self.false

        my_size = float(self.size())
        return (self.true.size() / my_size) * self.true.entropy_root() + (self.false.size() / my_size) * self.false.entropy_root()


class RandomForest(object):
    """Implementation of random forest model"""
    def __init__(self, k, m, metadata):
        super(RandomForest, self).__init__()
        self.k = k
        self.m = m

        # Metadata contains the information about each input feature: either 'boolean' or 'continuous'
        assert m <= len(metadata)
        self.metadata = metadata

    def _build_decision(self, feature_index, data):
        assert feature_index < len(self.metadata)
        feature_type = self.metadata[feature_index]
        if feature_type == 'boolean':
            return Decision(feature_index, MID_VALUE)
        else:
            min_value = min(data[:,feature_index])
            max_value = max(data[:,feature_index])

            return Decision(feature_index, random.uniform(min_value, max_value))

    def fit(self, X, y):
        assert len(self.metadata) == X.shape[1]

        self.root_nodes = [TreeNode(X, y) for _ in xrange(self.k)]
        non_zero_leaf_nodes = [[node] for node in self.root_nodes] # List of stacks of nodes to consider

        while len([1 for tree in non_zero_leaf_nodes for leaf in tree]) > 0:
            # Pick m features randomly
            selected_features = random.sample(xrange(X.shape[1]), self.m)


            for tree_index in xrange(self.k):
                if len(non_zero_leaf_nodes[tree_index]) == 0:
                    continue

                # Pick the leaf node from the stack
                leaf_node = non_zero_leaf_nodes[tree_index].pop()

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
                leaf_node.split(min_decision) # Assign the chosen decision to this node

                # Then delete data from the node to save memory (since data is already saved in the node's children)
                leaf_node.X = None
                leaf_node.y = None

                if leaf_node.true.entropy_root() > 0:
                    non_zero_leaf_nodes[tree_index].append(leaf_node.true)

                if leaf_node.false.entropy_root() > 0:
                    non_zero_leaf_nodes[tree_index].append(leaf_node.false)

    def predict_single(self, x):
        results = []
        for root in self.root_nodes:
            node = root
            while True: # Go down the tree
                if node.is_leaf():
                    results.append(node.predict(x))
                    break
                elif node.decision.test(x):
                    node = node.true
                else:
                    node = node.false

        positive_count = sum(1 for _ in results)
        return POSITIVE_VALUE if positive_count >= len(results) else NEGATIVE_VALUE # Major vote

    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def evaluate(self, X, y):
        prediction = self.predict(X)
        error_count = sum(1 for index, value in enumerate(prediction) if value == y[index])

        error = float(error_count) / len(y)
        print "Error rate: {0}".format(error)
        return error


if __name__ == "__main__":
    rf = RandomForest(3, 2, ['boolean', 'continuous', 'boolean'])

    X = np.array([
                    [NEGATIVE_VALUE, 7, NEGATIVE_VALUE],
                    [POSITIVE_VALUE, 6, POSITIVE_VALUE],
                    [NEGATIVE_VALUE, 4, NEGATIVE_VALUE],
                    [POSITIVE_VALUE, 5, POSITIVE_VALUE],
                    [POSITIVE_VALUE, 3, POSITIVE_VALUE],
                    [NEGATIVE_VALUE, 9, NEGATIVE_VALUE],
                    [NEGATIVE_VALUE, 12, NEGATIVE_VALUE],
                    [POSITIVE_VALUE, 14, POSITIVE_VALUE]
                    ])
    y = [POSITIVE_VALUE, NEGATIVE_VALUE, POSITIVE_VALUE, POSITIVE_VALUE, POSITIVE_VALUE, NEGATIVE_VALUE, NEGATIVE_VALUE, POSITIVE_VALUE]
    rf.fit(X, y)
    rf.evaluate(X, y)