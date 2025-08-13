"""
Vectorized version of the decision tree module for improved performance
"""

from scipy.stats import entropy
import numpy as np
from utils import *

class Tree():
    def __init__(self, conjunctions, splitting_values, max_depth):
        self.conjunctions = conjunctions
        self.splitting_values = splitting_values
        self.max_depth = max_depth

    def split(self):
        if len(self.conjunctions) == 1 or self.max_depth == 0:
            self.selected_feature = None
            self.left = None
            self.right = None
            return

        # Vectorized check for unique labels
        labels = np.array([np.argmax(conj.label_probas) for conj in self.conjunctions])
        if len(np.unique(labels)) > 1:
            self.selected_feature, self.selected_value, self.entropy, \
            l_conjunctions, r_conjunctions = select_splitting_feature_by_entropy(self.conjunctions, self.splitting_values)
        else:
            self.selected_feature, self.selected_value, self.entropy, \
            l_conjunctions, r_conjunctions = select_splitting_feature_by_max_splitting(self.conjunctions, self.splitting_values)

        if self.selected_feature is None:
            return

        descending_splitting_values = {k: ([i for i in v if i != self.selected_value] if k == self.selected_feature else v)
                                     for k, v in self.splitting_values.items()}
        self.left = Tree(l_conjunctions, descending_splitting_values, max_depth=self.max_depth-1)
        self.right = Tree(r_conjunctions, descending_splitting_values, max_depth=self.max_depth-1)
        self.left.split()
        self.right.split()

    def predict_instance_proba(self, inst):
        if self.selected_feature is None:
            # Vectorized computation
            label_probas = np.array([c.label_probas for c in self.conjunctions])
            softmax_probas = np.array([softmax(proba) for proba in label_probas])
            return softmax_probas.mean(axis=0)[0] if softmax_probas.ndim > 1 else softmax_probas[0]

        if inst[self.selected_feature] >= self.selected_value:
            return self.left.predict_instance_proba(inst)
        else:
            return self.right.predict_instance_proba(inst)

    def get_instance_decision_path(self, inst, result=[]):
        result = list(result)
        if self.selected_feature is None:
            # Vectorized computation for leaf
            label_probas = np.array([c.label_probas for c in self.conjunctions])
            softmax_probas = np.array([softmax(proba) for proba in label_probas])
            mean_proba = softmax_probas.mean(axis=0)[0] if softmax_probas.ndim > 1 else softmax_probas[0]
            result.append('labels: ' + str(mean_proba))
            return result
        else:
            if inst[self.selected_feature] >= self.selected_value:
                result.append(str(self.selected_feature) + '>=' + str(self.selected_value))
                return self.left.get_instance_decision_path(inst, result)
            else:
                result.append(str(self.selected_feature) + '<' + str(self.selected_value))
                return self.right.get_instance_decision_path(inst, result)

    def predict_proba(self, data):
        # Vectorized batch prediction
        data_array = data.values
        probas = np.array([self.predict_instance_proba(inst) for inst in data_array])
        return probas

    def get_decision_paths(self, data):
        # Vectorized batch path computation
        return [self.get_instance_decision_path(inst) for inst in data.values]

    def predict_proba_and_depth(self, data):
        # Vectorized batch prediction with depth
        results = [self.predict_instance_proba_and_depth(inst) for inst in data.values]
        probas, depths = zip(*results)
        return np.array(probas), list(depths)

    def predict_instance_proba_and_depth(self, inst):
        if self.selected_feature is None:
            label_probas = np.array([c.label_probas for c in self.conjunctions])
            softmax_probas = np.array([softmax(proba) for proba in label_probas])
            result = softmax_probas.mean(axis=0)[0] if softmax_probas.ndim > 1 else softmax_probas[0]
            return result, 0

        if inst[self.selected_feature] >= self.selected_value:
            probas, depth = self.left.predict_instance_proba_and_depth(inst)
            return probas, depth + 1
        else:
            probas, depth = self.right.predict_instance_proba_and_depth(inst)
            return probas, depth + 1


def select_splitting_feature_by_entropy(conjunctions, splitting_values):
    conjunctions_len = len(conjunctions)
    # Vectorized entropy calculation
    label_probas = np.array([c.label_probas for c in conjunctions])
    best_entropy = get_entropy(label_probas)

    selected_feature, selected_value, l_conjunctions, r_conjunctions = None, None, None, None

    for feature, values in splitting_values.items():
        if len(values) == 0:
            continue

        # Vectorized feature bounds extraction
        upper_bounds = np.array([conj.features_upper[feature] for conj in conjunctions])
        lower_bounds = np.array([conj.features_lower[feature] for conj in conjunctions])

        for value in values:
            temp_l_conjunctions, temp_r_conjunctions, temp_entropy = calculate_entropy_for_split_vectorized(
                conjunctions, upper_bounds, lower_bounds, value, label_probas)

            if (temp_entropy < best_entropy and
                len(temp_l_conjunctions) < conjunctions_len and
                len(temp_r_conjunctions) < conjunctions_len):
                best_entropy = temp_entropy
                selected_feature = feature
                selected_value = value
                l_conjunctions = temp_l_conjunctions
                r_conjunctions = temp_r_conjunctions

    return selected_feature, selected_value, best_entropy, l_conjunctions, r_conjunctions


def select_splitting_feature_by_max_splitting(conjunctions, splitting_values):
    best_value = len(conjunctions)
    selected_feature, selected_value, l_conjunctions, r_conjunctions = None, None, None, None

    for feature, values in splitting_values.items():
        if len(values) == 0:
            continue

        # Vectorized feature bounds extraction
        upper_bounds = np.array([conj.features_upper[feature] for conj in conjunctions])
        lower_bounds = np.array([conj.features_lower[feature] for conj in conjunctions])

        for value in values:
            temp_l_conjunctions, temp_r_conjunctions, temp_value = calculate_max_for_split_vectorized(
                conjunctions, upper_bounds, lower_bounds, value)

            if temp_value < best_value:
                best_value = temp_value
                selected_feature = feature
                selected_value = value
                l_conjunctions = temp_l_conjunctions
                r_conjunctions = temp_r_conjunctions

    return selected_feature, selected_value, 0, l_conjunctions, r_conjunctions


def calculate_entropy_for_split_vectorized(conjunctions, upper_bounds, lower_bounds, value, label_probas):
    # Vectorized splitting logic
    left_mask = lower_bounds >= value
    right_mask = upper_bounds <= value
    both_mask = ~(left_mask | right_mask)

    l_conjunctions = []
    r_conjunctions = []
    l_probas = []
    r_probas = []

    # Process each conjunction based on masks
    for i, conj in enumerate(conjunctions):
        if left_mask[i]:
            l_conjunctions.append(conj)
            l_probas.append(label_probas[i])
        elif right_mask[i]:
            r_conjunctions.append(conj)
            r_probas.append(label_probas[i])
        else:  # both_mask[i]
            l_conjunctions.append(conj)
            r_conjunctions.append(conj)
            l_probas.append(label_probas[i])
            r_probas.append(label_probas[i])

    return l_conjunctions, r_conjunctions, calculate_weighted_entropy_vectorized(l_probas, r_probas)


def calculate_max_for_split_vectorized(conjunctions, upper_bounds, lower_bounds, value):
    # Same splitting logic as entropy version
    left_mask = lower_bounds >= value
    right_mask = upper_bounds <= value

    l_conjunctions = []
    r_conjunctions = []

    for i, conj in enumerate(conjunctions):
        if left_mask[i]:
            l_conjunctions.append(conj)
        elif right_mask[i]:
            r_conjunctions.append(conj)
        else:
            l_conjunctions.append(conj)
            r_conjunctions.append(conj)

    return l_conjunctions, r_conjunctions, max(len(l_conjunctions), len(r_conjunctions))


def calculate_weighted_entropy_vectorized(l_probas, r_probas):
    if not l_probas or not r_probas:
        return float('inf')

    l_entropy = get_entropy_vectorized(np.array(l_probas))
    r_entropy = get_entropy_vectorized(np.array(r_probas))
    l_size, r_size = len(l_probas), len(r_probas)
    overall_size = l_size + r_size

    return (l_size * l_entropy + r_size * r_entropy) / overall_size


def get_entropy_vectorized(probas_array):
    # Vectorized entropy calculation
    if len(probas_array) == 0:
        return 0

    values = np.argmax(probas_array, axis=1)
    unique_values, counts = np.unique(values, return_counts=True)
    probabilities = counts / np.sum(counts)

    return entropy(probabilities)


def get_entropy(probas):
    # Keep original function for backward compatibility
    if isinstance(probas, np.ndarray) and probas.ndim > 1:
        return get_entropy_vectorized(probas)

    values = np.array([np.argmax(x) for x in probas])
    values, counts = np.unique(values, return_counts=True)
    probabilities = counts / np.sum(counts)
    return entropy(probabilities)


# Keep original functions for backward compatibility
def calculate_entropy_for_split(conjunctions, feature, value):
    l_conjunctions = []
    r_conjunctions = []
    l_probas = []
    r_probas = []
    for conj in conjunctions:
        if conj.features_upper[feature] <= value:
            r_conjunctions.append(conj)
            r_probas.append(conj.label_probas)
        elif conj.features_lower[feature] >= value:
            l_conjunctions.append(conj)
            l_probas.append(conj.label_probas)
        else:
            r_conjunctions.append(conj)
            r_probas.append(conj.label_probas)
            l_conjunctions.append(conj)
            l_probas.append(conj.label_probas)
    return l_conjunctions, r_conjunctions, calculate_weighted_entropy(l_probas, r_probas)


def calculate_weighted_entropy(l_probas, r_probas):
    l_entropy, r_entropy = get_entropy(l_probas), get_entropy(r_probas)
    l_size, r_size = len(l_probas), len(r_probas)
    overall_size = l_size + r_size
    return (l_size * l_entropy + r_size * r_entropy) / overall_size


def calculate_max_for_split(conjunctions, feature, value):
    l_conjunctions = []
    r_conjunctions = []
    l_probas = []
    r_probas = []
    for conj in conjunctions:
        if conj.features_upper[feature] <= value:
            r_conjunctions.append(conj)
            r_probas.append(conj.label_probas)
        elif conj.features_lower[feature] >= value:
            l_conjunctions.append(conj)
            l_probas.append(conj.label_probas)
        else:
            r_conjunctions.append(conj)
            r_probas.append(conj.label_probas)
            l_conjunctions.append(conj)
            l_probas.append(conj.label_probas)
    return l_conjunctions, r_conjunctions, max(len(l_conjunctions), len(r_conjunctions))