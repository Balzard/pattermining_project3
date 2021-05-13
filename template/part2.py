"""The main program that runs gSpan. Two examples are provided"""
# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import commonprefix
import sys
import numpy
from sklearn import naive_bayes
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

from gspan_mining import gSpan
from gspan_mining import GraphDatabase
from bisect import insort
import os


class PatternGraphs:
    """
    This template class is used to define a task for the gSpan implementation.
    You should not modify this class but extend it to define new tasks
    """

    def __init__(self, database):
        # A list of subsets of graph identifiers.
        # Is used to specify different groups of graphs (classes and training/test sets).
        # The gid-subsets parameter in the pruning and store function will contain for each subset, all the occurrences
        # in which the examined pattern is present.
        self.gid_subsets = []

        self.database = database  # A graphdatabase instance: contains the data for the problem.

    def store(self, dfs_code, gid_subsets):
        """
        Code to be executed to store the pattern, if desired.
        The function will only be called for patterns that have not been pruned.
        In correlated pattern mining, we may prune based on confidence, but then check further conditions before storing.
        :param dfs_code: the dfs code of the pattern (as a string).
        :param gid_subsets: the cover (set of graph ids in which the pattern is present) for each subset in self.gid_subsets
        """
        print("Please implement the store function in a subclass for a specific mining task!")

    def prune(self, gid_subsets):
        """
        prune function: used by the gSpan algorithm to know if a pattern (and its children in the search tree)
        should be pruned.
        :param gid_subsets: A list of the cover of the pattern for each subset.
        :return: true if the pattern should be pruned, false otherwise.
        """
        print("Please implement the prune function in a subclass for a specific mining task!")

class FrequentGraphs(PatternGraphs):
    """
    Finds the frequent (support >= minsup) subgraphs among the positive graphs.
    This class provides a method to build a feature matrix for each subset.
    """

    def __init__(self, minsup, database, subsets, k):
        """
        Initialize the task.
        :param minsup: the minimum positive support
        :param database: the graph database
        :param subsets: the subsets (train and/or test sets for positive and negative class) of graph ids.
        """
        super().__init__(database)
        self.patterns = []  # The patterns found in the end (as dfs codes represented by strings) with their cover (as a list of graph ids).
        self.minsup = minsup
        self.gid_subsets = subsets
        self.k = k
        self.nb_top = 0 # nb top k items

    def compare_pattern_with_bests(self, confidence, support, method):
        if method == "same_freq":
            for i in self.patterns:
                if i[0] == confidence and i[1] == support:
                    return True
            return False

        if method == "low_freq":
            for i in self.patterns:
                if i[0] == confidence and i[1] < support:
                    return True
            return False

    def store(self, dfs_code, gid_subsets):
        p = len(gid_subsets[0])
        n = len(gid_subsets[2])
        support = p+n
        confidence = p/support

        # if same conf and freq than a top k item then add
        if self.compare_pattern_with_bests(confidence, support, "same_freq"):
            insort(self.patterns, [confidence, support, dfs_code, gid_subsets])

        # already k top items
        elif self.nb_top == self.k:
            # confidence of new item is higher than the lowest confidence in top k items
            if self.patterns[0][0] < confidence:
                self.patterns = [i for i in self.patterns if i[0] != self.patterns[0][0] or i[1] != self.patterns[0][1]]
                insort(self.patterns, [confidence, support, dfs_code, gid_subsets])

            # if same conf but higher freq than a top k item then delete top item and add new
            elif self.patterns[0][0] == confidence and self.compare_pattern_with_bests(confidence, support, "low_freq"):
                self.patterns = [i for i in self.patterns if i[0] != self.patterns[0][0] or i[1] != self.patterns[0][1]]
                insort(self.patterns, [confidence, support, dfs_code, gid_subsets])

        # new confidence and not yet k best items
        elif self.nb_top < self.k:
            insort(self.patterns, [confidence, support, dfs_code, gid_subsets]) # sort on confidence, then support
            self.nb_top += 1

    # Prunes any pattern that is not frequent in the both classes
    def prune(self, gid_subsets):
        p = len(gid_subsets[0])
        n = len(gid_subsets[2])
        support = p + n
        return support < self.minsup

    """ creates a column for a feature matrix. """
    def create_fm_col(self, all_gids, subset_gids):
        subset_gids = set(subset_gids)
        bools = []
        for i, val in enumerate(all_gids):
            if val in subset_gids:
                bools.append(1)
            else:
                bools.append(0)
        return bools

    """ return a feature matrix for each subset of examples, in which the columns correspond to patterns
         and the rows to examples in the subset. """
    def get_feature_matrices(self):
        matrices = [[] for _ in self.gid_subsets]
        for pattern in self.patterns:
            for i, gid_subset in enumerate(pattern[3]):
                matrices[i].append(self.create_fm_col(self.gid_subsets[i], gid_subset))
        return [numpy.array(matrix).transpose() for matrix in matrices]

def example2():
    """
    Runs gSpan with the specified positive and negative graphs; finds all frequent subgraphs in the training subset of
    the positive class with a minimum support of minsup.
    Uses the patterns found to train a naive bayesian classifier using Scikit-learn and evaluates its performances on
    the test set.
    Performs a k-fold cross-validation.
    """

    args = sys.argv
    database_file_name_pos = args[1]  # First parameter: path to positive class file
    database_file_name_neg = args[2]  # Second parameter: path to negative class file
    k = int(args[3])  # Third parameter: k
    minsup = int(args[4])  # Fourth parameter: minimum support
    nfolds = int(args[5])  # Fifth parameter: number of folds to use in the k-fold cross-validation.

    if not os.path.exists(database_file_name_pos):
        print('{} does not exist.'.format(database_file_name_pos))
        sys.exit()
    if not os.path.exists(database_file_name_neg):
        print('{} does not exist.'.format(database_file_name_neg))
        sys.exit()

    graph_database = GraphDatabase()  # Graph database object
    pos_ids = graph_database.read_graphs(database_file_name_pos)  # Reading positive graphs, adding them to database and getting ids
    neg_ids = graph_database.read_graphs(database_file_name_neg)  # Reading negative graphs, adding them to database and getting ids

    # If less than two folds: using the same set as training and test set (note this is not an accurate way to evaluate the performances!)
    if nfolds < 2:
        subsets = [
            pos_ids,  # Positive training set
            pos_ids,  # Positive test set
            neg_ids,  # Negative training set
            neg_ids  # Negative test set
        ]
        # Printing fold number:
        print('fold {}'.format(1))
        train_and_evaluate(minsup, graph_database, subsets, k)

    # Otherwise: performs k-fold cross-validation:
    else:
        pos_fold_size = len(pos_ids) // nfolds
        neg_fold_size = len(neg_ids) // nfolds
        for i in range(nfolds):
            # Use fold as test set, the others as training set for each class;
            # identify all the subsets to be maintained by the graph mining algorithm.
            subsets = [
                numpy.concatenate((pos_ids[:i * pos_fold_size], pos_ids[(i + 1) * pos_fold_size:])),  # Positive training set
                pos_ids[i * pos_fold_size:(i + 1) * pos_fold_size],  # Positive test set
                numpy.concatenate((neg_ids[:i * neg_fold_size], neg_ids[(i + 1) * neg_fold_size:])),  # Negative training set
                neg_ids[i * neg_fold_size:(i + 1) * neg_fold_size],  # Negative test set
            ]
            # Printing fold number:
            print('fold {}'.format(i+1))
            train_and_evaluate(minsup, graph_database, subsets, k)


def train_and_evaluate(minsup, database, subsets, k):
	task = FrequentGraphs(minsup, database, subsets, k)  # Creating task

	gSpan(task).run()  # Running gSpan

	# Creating feature matrices for training and testing:
	features = task.get_feature_matrices()
	train_fm = numpy.concatenate((features[0], features[2]))  # Training feature matrix
	train_labels = numpy.concatenate((numpy.full(len(features[0]), 1, dtype=int), numpy.full(len(features[2]), -1, dtype=int)))  # Training labels
	test_fm = numpy.concatenate((features[1], features[3]))  # Testing feature matrix
	test_labels = numpy.concatenate((numpy.full(len(features[1]), 1, dtype=int), numpy.full(len(features[3]), -1, dtype=int)))  # Testing labels

	classifier = DecisionTreeClassifier(random_state=1)  # Creating model object
	classifier.fit(train_fm, train_labels)  # Training model

	predicted = classifier.predict(test_fm)  # Using model to predict labels of testing data

	accuracy = metrics.accuracy_score(test_labels, predicted)  # Computing accuracy:

	# Printing frequent patterns along with their positive support:
	for conf, support, code, gid_subsets in task.patterns:
		print('{} {} {}'.format(code, conf, support))
	# printing classification results:
	print(predicted.tolist())
	print('accuracy: {}'.format(accuracy))
	print()  # Blank line to indicate end of fold



if __name__ == '__main__':
	example2()
	