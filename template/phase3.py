"""The main program that runs gSpan. Two examples are provided"""
# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy
import copy
from sklearn import naive_bayes
from sklearn import tree
from sklearn import metrics

from gspan_mining import gSpan
from gspan_mining import GraphDatabase
from bisect import insort, bisect_left


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
        print(
            "Please implement the store function in a subclass for a specific mining task!"
        )

    def prune(self, gid_subsets):
        """
        prune function: used by the gSpan algorithm to know if a pattern (and its children in the search tree)
        should be pruned.
        :param gid_subsets: A list of the cover of the pattern for each subset.
        :return: true if the pattern should be pruned, false otherwise.
        """
        print(
            "Please implement the prune function in a subclass for a specific mining task!"
        )


class FrequentPositiveGraphs(PatternGraphs):
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
        self.nb_top = 0  # nb top k items

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
        
        pos_c = p / support
        neg_c = n / support
        if pos_c >= neg_c:
            confidence = pos_c
            is_pos = True
        else:
            confidence = neg_c
            is_pos = False


        # if same conf and freq than a top k item then add
        if self.compare_pattern_with_bests(confidence, support, "same_freq"):
            insort(self.patterns, [confidence, support, dfs_code, gid_subsets, is_pos])

        # already k top items
        elif self.nb_top == self.k:
            # confidence of new item is higher than the lowest confidence in top k items
            if self.patterns[0][0] < confidence:
                self.patterns = [i for i in self.patterns if i[0] != self.patterns[0][0] or i[1] != self.patterns[0][1]]
                insort(self.patterns, [confidence, support, dfs_code, gid_subsets, is_pos])

            # if same conf but higher freq than a top k item then delete top item and add new
            elif self.patterns[0][0] == confidence and self.compare_pattern_with_bests(confidence, support, "low_freq"):
                self.patterns = [i for i in self.patterns if i[0] != self.patterns[0][0] or i[1] != self.patterns[0][1]]
                insort(self.patterns, [confidence, support, dfs_code, gid_subsets, is_pos])

        # new confidence and not yet k best items
        elif self.nb_top < self.k:
            insort(self.patterns, [confidence, support, dfs_code, gid_subsets, is_pos]) # sort on confidence, then support
            self.nb_top += 1

     # Prunes any pattern that is not frequent in the both classes
    def prune(self, gid_subsets):
        p = len(gid_subsets[0])
        n = len(gid_subsets[2])
        support = p + n
        return support < self.minsup

def example3():
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
    k = int(args[3])  # Third parameter: top K
    minsup = int(args[4])  # Forth parameter: minimum support (note: this parameter will be k in case of top-k mining)
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
            print('fold {}'.format(i + 1))
            train_and_evaluate(minsup, graph_database, subsets, k)


"""remove list1's elements from list 2"""


def remove(list1, list2):
    for i in range(len(list1)):
        list2[i] = [x for x in list2[i] if x not in list1[i]]
    return list2


def train_and_evaluate(minsup, database, subsets, k):

    pos_ids = copy.deepcopy(subsets[1])
    neg_ids = copy.deepcopy(subsets[3])
    new_subsets = []
    for subset in subsets:
        if type(subset) != type([]):
            new_subset = subset.tolist()
            new_subsets.append(new_subset)
        else:
            new_subsets.append(subset)

    result = []
    test_is_pos = []
    for _ in range(k):
        task = FrequentPositiveGraphs(minsup, database, new_subsets, 1)
        gSpan(task).run()
        sort_list = []
        for pattern in task.patterns:
            sort_list.append([pattern[2], pattern[0], pattern[1], pattern[3]])
        sort_list.sort()
        if len(sort_list) > 0:
            result.append(sort_list[0])
            subsets_list = sort_list[0][3]
            test_list = subsets_list[1] + subsets_list[3]

            for item in test_list:
                insort(test_is_pos, [item, pattern[4]])

            new_subsets = remove(subsets_list, new_subsets)

    test_list = new_subsets[1] + new_subsets[3]
    length_pos = len(new_subsets[0])
    length_neg = len(new_subsets[2])

    if length_pos >= length_neg:
        is_pos = True
    else:
        is_pos = False

    for item in test_list:
        insort(test_is_pos, [item, is_pos])

    for pattern in result:
        print('{} {} {}'.format(pattern[0], pattern[1], pattern[2]))

    pred_result = []
    for pred in test_is_pos:
        if pred[1]:
            pred_result.append(1)
        else:
            pred_result.append(-1)

    print(pred_result)

    counter = 0

    for is_pos in test_is_pos:
        if is_pos[0] in pos_ids:
            if is_pos[1]:
                counter += 1
        if is_pos[0] in neg_ids:
            if not is_pos[1]:
                counter += 1
    accuracy = counter / len(test_is_pos)
    print('accuracy: {}'.format(accuracy))
    print()


if __name__ == '__main__':
    example3()