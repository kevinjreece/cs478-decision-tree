import numpy as np
import math

class DecisionTree:
	def __init__(self):
		self.root_node = self.Node()

	def learn(self, all_data, all_labels):
		self.root_node.learn(all_data, all_labels)

	def predict(self, data):
		return root_node.predict(data)

	def __str__(self):
		node_queue = [("", self.root_node)]
		level_size = 0
		count = 1
		result = ""

		while(len(node_queue) > 0):
			count -= 1
			k, n = node_queue[0]
			del node_queue[0]
			result += '(' + k + ', ' + n.split_attribute_name + ', ' + n.label + ')'

			for k, v in n.children.iteritems():
				node_queue.append((k, v))
			if count == 0:
				count = len(node_queue)
				result += '\n\n'

		return result

	class Node:
		def __init__(self):
			self.label = ""
			self.split_attribute_index = -1
			self.split_attribute_name = ""
			self.children = {}

		def learn(self, all_data, column_names):
			if len(all_data) == 0: # there is no more data to train on
				# return 'unknown' and have the parent node use its label
				self.label = 'unknown'
				# print "no more data to train on:", self.label
				# raw_input()
				return

			all_labels = list(all_data[:, -1])
			set_of_labels = set(all_labels)

			# handle base cases
			if all_data.shape[1] == 1: # no more attributes
				# set label of node equal to most common label of data
				self.label = max(set_of_labels, key=all_labels.count)
				# print "no more attributes:", self.label
				# raw_input()
				return
			elif len(set_of_labels) == 1: # node is pure
				# set label of node equal to only remaining label
				self.label = list(set_of_labels)[0]
				# print "node is pure:", self.label
				# raw_input()
				return

			self.label = max(set_of_labels, key=all_labels.count)
			
			# find attribute to split on
			self.split_attribute_index = self.selectBestSplitAttribute(all_data)
			self.split_attribute_name = column_names[self.split_attribute_index]

			# split data and create children
			attribute_choices = set(all_data[:, self.split_attribute_index])
			for attr in attribute_choices:
				subset_of_data = np.array(filter(lambda x: x[self.split_attribute_index] == attr, all_data))
				subset_of_columns = column_names[:]

				subset_of_data = np.delete(subset_of_data, self.split_attribute_index, 1)
				del subset_of_columns[self.split_attribute_index]

				child = DecisionTree.Node()				
				child.learn(subset_of_data, subset_of_columns)
				self.children[attr] = child

		def predict(self, data):
			if self.split_attribute_index == -1:
				return self.label

			attr = data[split_attribute_index]
			subset_of_data = np.delete(data, split_attribute_index, 1)
			prediction = self.children[data[split_attribute_index]].predict(data)
			return prediction if prediction != 'unknown' else self.label

		def selectBestSplitAttribute(self, all_data):
			return np.argmin([calculateAttributeInformation(all_data, col) for col in range(all_data.shape[1] - 1)])

def calculateAttributeInformation(all_data, column):
	attribute_choices = list(set(all_data[:, column]))

	choice_sum = 0
	for choice in attribute_choices:
		choice_data = np.array(filter(lambda x: x[column] == choice, all_data))
		choice_count = len(choice_data)
		choice_ratio = choice_count / float(len(all_data))
		choice_sum += choice_ratio * calculateAttributeChoiceInformation(choice_data, column, choice)

	return choice_sum

def calculateAttributeChoiceInformation(choice_data, column, choice):
	labels = list(set(choice_data[:, -1]))
	
	label_sum = 0
	for label in labels:
		label_count = len(np.array(filter(lambda x: x[-1] == label, choice_data)))
		label_ratio = label_count / float(len(choice_data))
		label_partial_sum = label_ratio * math.log(label_ratio, 2)
		label_sum -= label_partial_sum

	return label_sum






