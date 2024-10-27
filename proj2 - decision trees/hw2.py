import math
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5: 0.45,
                 0.25: 1.32,
                 0.1: 2.71,
                 0.05: 3.84,
                 0.0001: 100000},
             2: {0.5: 1.39,
                 0.25: 2.77,
                 0.1: 4.60,
                 0.05: 5.99,
                 0.0001: 100000},
             3: {0.5: 2.37,
                 0.25: 4.11,
                 0.1: 6.25,
                 0.05: 7.82,
                 0.0001: 100000},
             4: {0.5: 3.36,
                 0.25: 5.38,
                 0.1: 7.78,
                 0.05: 9.49,
                 0.0001: 100000},
             5: {0.5: 4.35,
                 0.25: 6.63,
                 0.1: 9.24,
                 0.05: 11.07,
                 0.0001: 100000},
             6: {0.5: 5.35,
                 0.25: 7.84,
                 0.1: 10.64,
                 0.05: 12.59,
                 0.0001: 100000},
             7: {0.5: 6.35,
                 0.25: 9.04,
                 0.1: 12.01,
                 0.05: 14.07,
                 0.0001: 100000},
             8: {0.5: 7.34,
                 0.25: 10.22,
                 0.1: 13.36,
                 0.05: 15.51,
                 0.0001: 100000},
             9: {0.5: 8.34,
                 0.25: 11.39,
                 0.1: 14.68,
                 0.05: 16.92,
                 0.0001: 100000},
             10: {0.5: 9.34,
                  0.25: 12.55,
                  0.1: 15.99,
                  0.05: 18.31,
                  0.0001: 100000},
             11: {0.5: 10.34,
                  0.25: 13.7,
                  0.1: 17.27,
                  0.05: 19.68,
                  0.0001: 100000}}


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    class_count = data[:, -1]
    label_count = np.unique(class_count, return_counts=True)[1]
    p_i = label_count / len(class_count)
    gini = 1 - np.sum(p_i ** 2)
    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    class_count = data[:, -1]
    label_count = np.unique(class_count, return_counts=True)[1]
    p_i = label_count / len(class_count)
    entropy = -1 * (np.sum(p_i * np.log2(p_i)))
    return entropy


class DecisionNode:

    def __init__(self, data, impurity_func, feature=-1, depth=0, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data  # the relevant data for the node
        self.feature = feature  # column index of criteria being tested
        self.pred = self.calc_node_pred()  # the prediction of the node
        self.depth = depth  # the current depth of the node
        self.children = []  # array that holds this nodes children
        self.children_values = []
        self.terminal = False  # determines if the node is a leaf
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0

    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        lables = self.data[:, -1]
        labels_names, lables_count = np.unique(lables, return_counts=True)
        max_index = np.argmax(lables_count)  # finds the class index with majority
        pred = labels_names[max_index]
        return pred

    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)

    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        goodness_grade, _ = self.goodness_of_split(self.feature)
        self.feature_importance = (self.data.shape[0] / n_total_sample) * goodness_grade  # formula

    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        goodness = 0
        groups = {}  # groups[feature_value] = data_subset
        total_impurity = self.impurity_func(self.data)
        value_options = np.unique(self.data[:, feature])
        for value in value_options:  # create the groups according to the given feature
            groups[value] = self.data[self.data[:, feature] == value]
        if self.gain_ratio:  # using entropy when calculating ratio
            func = calc_entropy
        else:
            func = self.impurity_func
        for group in groups.values():
            goodness += func(group) * (group.shape[0] / self.data.shape[
                0])  # going over the groups and calculating the goodness according to formula
        goodness = total_impurity - goodness
        data_length = self.data.shape[0]
        information_sum = 0
        if self.gain_ratio:  # calculating information sum and gain_ratio if needed
            for group in groups.values():
                information_sum += (group.shape[0] / data_length) * np.log(group.shape[0] / data_length)
            split_information = -1 * information_sum
            goodness = goodness / split_information if split_information > 0 else 0

        return goodness, groups

    def chi_test(self, groups):
        if self.chi == 1:  # not using chi based pruning
            return True
        value_options = np.unique(self.data[:, -1], return_counts=True)[1]
        p_0 = value_options[0] / sum(value_options)
        p_1 = value_options[1] / sum(value_options)
        curr_chi = 0.0
        for group in groups.values():
            d_f = len(group)
            uniq, uniq_count = np.unique(group[:, -1], return_counts=True)
            if uniq[0] == 'p':
                p_f = 0
                n_f = uniq_count[0]
            else:  # uniq[0] = 'e'
                p_f = uniq_count[0]
                n_f = uniq_count[1] if (len(uniq) > 1) else 0
            e0 = d_f * p_0
            e1 = d_f * p_1
            curr_chi += ((p_f - e0) ** 2) / e0 + ((n_f - e1) ** 2) / e1  # sum (sigma) according to the formula
        degree = len(groups) - 1  # degree is number of attributes -1
        return curr_chi > chi_table[degree][self.chi]  # equivalent cell in chi table

    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        if self.depth == self.max_depth:
            self.terminal = True
            return

        best_feature = -1
        max_goodness = 0  # if the goodness of a split will be 0 it will not grow this split
        max_groups = {}
        for i in range(self.data.shape[1] - 1):
            goodness, groups = self.goodness_of_split(i)
            if goodness > max_goodness and self.chi_test(groups):  # finding the best feature for split
                max_goodness = goodness
                max_groups = groups
                best_feature = i
        self.feature = best_feature
        for key, group_data in max_groups.items():
            self.add_child(DecisionNode(group_data, self.impurity_func, depth=self.depth + 1, chi=self.chi,
                                        max_depth=self.max_depth, gain_ratio=self.gain_ratio), key)


class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data  # the relevant data for the tree
        self.impurity_func = impurity_func  # the impurity function to be used in the tree
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio  #
        self.root = None  # the root node of the tree

    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = DecisionNode(data=self.data, impurity_func=self.impurity_func, chi=self.chi,
                                 max_depth=self.max_depth, gain_ratio=self.gain_ratio)
        nodes_queue = [self.root]
        while nodes_queue:  # there is still nodes in the queue that we need to split
            node = nodes_queue.pop(0)
            if len(np.unique(node.data[:, -1])) == 1:  # there is only one class on the data so its a leaf, goodness
                # of split will be 0
                node.terminal = True
                continue
            if node.depth >= self.max_depth:
                node.terminal = True
                continue
            node.split()
            node.calc_feature_importance(self.data.shape[0])
            nodes_queue.extend(node.children)  # append the children to the queue, so they will be split

    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        pred = None
        current_node = self.root
        while not current_node.terminal:  # the current node is not a leaf
            current_feature = current_node.feature
            instance_val = instance[current_feature]
            if instance_val not in current_node.children_values:
                break
            next_index = current_node.children_values.index(
                instance_val)  # find the index of the next node according to the instance data of the current node feature
            current_node = current_node.children[next_index]
        return current_node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        for instance in dataset:
            prediction = self.predict(instance)
            if prediction == instance[-1]:
                accuracy += 1
        return accuracy / len(dataset) * 100

    def depth(self):
        return self.root.depth()


# recursively calcs the depth
def calc_tree_depth(node):
    curr_max = node.depth
    if node.terminal:
        return node.depth
    for child in node.children:
        curr_max = max(curr_max, calc_tree_depth(child))
    return curr_max


def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation = []
    root = None

    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        current_tree = DecisionTree(data=X_train, impurity_func=calc_entropy, max_depth=max_depth, gain_ratio=True)
        current_tree.build_tree()
        training.append(current_tree.calc_accuracy(X_train))
        validation.append(current_tree.calc_accuracy(X_validation))
    return training, validation


def chi_pruning(X_train, X_test):
    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc = []
    depth = []

    for chi_value in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        tree = DecisionTree(data=X_train, impurity_func=calc_entropy, chi=chi_value, gain_ratio=True)
        tree.build_tree()
        chi_training_acc.append(tree.calc_accuracy(X_train))
        chi_validation_acc.append(tree.calc_accuracy(X_test))
        depth.append(calc_tree_depth(tree.root))

    return chi_training_acc, chi_validation_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    if node.terminal: return 1  # leaf
    count = 1
    for child in node.children:
        count += count_nodes(child)
    return count
