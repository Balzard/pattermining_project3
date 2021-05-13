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
		n = len(gid_subsets[1])
		support = p+n
		confidence = p/support

		# if same conf and freq than a top k item then add
		if self.compare_pattern_with_bests(confidence, support, "same_freq"):
			insort(self.patterns, [confidence, support, dfs_code])

		# if same conf but higher freq than top k item then delete top item and add new
		elif self.compare_pattern_with_bests(confidence, support, "low_freq"):
			self.patterns = [i for i in self.patterns if i[0] != confidence]
			insort(self.patterns, [confidence, support, dfs_code])

		# new confidence and not yet k best items
		elif self.nb_top < self.k:
			insort(self.patterns, [confidence, support, dfs_code]) # sort on confidence, then support
			self.nb_top += 1

		# already k top items
		elif self.nb_top == self.k:
			# confidence of new item is higher than the lowest confidence in top k items
			if self.patterns[0][0] < confidence:
				self.patterns = [i for i in self.patterns if i[0] > self.patterns[0][0]]
				insort(self.patterns, [confidence, support, dfs_code])

	# Prunes any pattern that is not frequent in the both classes
	def prune(self, gid_subsets):
		return len(gid_subsets[0]) + len(gid_subsets[1]) < self.minsup


if __name__ == '__main__':
	# "./data/molecules-small.pos"
	args = sys.argv
	database_file_name_pos = args[1]  # First parameter: path to positive class file
	database_file_name_neg = args[2]  # Second parameter: path to negative class file
	k = int(args[3])  # Third parameter: k
	minsup = int(args[4])  # Fourth parameter: minimum support

	if not os.path.exists(database_file_name_pos):
		print('{} does not exist.'.format(database_file_name_pos))
		sys.exit()
	if not os.path.exists(database_file_name_neg):
		print('{} does not exist.'.format(database_file_name_neg))
		sys.exit()

	graph_database = GraphDatabase()  # Graph database object
	pos_ids = graph_database.read_graphs(database_file_name_pos)  # Reading positive graphs, adding them to database and getting ids
	neg_ids = graph_database.read_graphs(database_file_name_neg)  # Reading negative graphs, adding them to database and getting ids

	subsets = [pos_ids, neg_ids]  # The ids for the positive and negative labelled graphs in the database
	task = FrequentGraphs(minsup, graph_database, subsets,k)  # Creating task

	gSpan(task).run()  # Running gSpan

	for pattern in task.patterns:
		support = pattern[1]
		confidence = pattern[0]
		print('{} {} {}'.format(pattern[2], confidence, support))
	