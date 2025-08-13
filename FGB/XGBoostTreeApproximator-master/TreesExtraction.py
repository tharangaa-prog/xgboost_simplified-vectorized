"""
This module contains vectorized functions for extracting information of individual trees from XGBoost
"""

import re
import numpy as np
from conjunction import *

# Compiled regex patterns
feature_regex = re.compile(r'\D+(?P<node_index>\d+):\[(?P<feature>[^<]+)<(?P<value>[^\]]+)\D+(?P<left>\d+)\D+(?P<right>\d+)\D+(?P<missing>\d+)')
leaf_regex = re.compile(r'\D+(?P<node_index>\d+)[^\=]+=(?P<prediction>.+)')

def extractNodesFromModel(model):
    """Extract decision trees from XGBoost using vectorized string operations."""
    trees = []
    tree_dumps = model._Booster.get_dump()

    for tree_string in tree_dumps:
        lines = tree_string.split('\n')[:-1]
        # Vectorized regex matching
        is_leaf = np.array(['[' not in line for line in lines])

        nodes = []
        for i, line in enumerate(lines):
            prefixed_line = 't' + line
            if is_leaf[i]:
                nodes.append(leaf_regex.search(prefixed_line).groupdict())
            else:
                nodes.append(feature_regex.search(prefixed_line).groupdict())
        trees.append(nodes)

    return trees

def extractClassValue(tree, leaf_index, label_names, class_index):
    """Vectorized class value extraction with numpy operations."""
    pred = float(tree[leaf_index]['prediction'])
    num_labels = len(label_names)

    if num_labels > 2:
        # Vectorized one-hot encoding
        probas = np.zeros(num_labels)
        probas[class_index] = pred
        return probas.tolist()
    else:
        # Vectorized sigmoid
        p = 1.0 / (1.0 + np.exp(-pred))  # Fixed sigmoid formula
        return [p, 1-p]

def extractConjunctionsFromTree(tree, tree_index, leaf_index, feature_dict, label_names, class_index):
    """Recursive function with optimized operations."""
    if 'prediction' in tree[leaf_index]:
        probas = extractClassValue(tree, leaf_index, label_names, class_index)
        return [Conjunction(list(feature_dict.keys()), label_names,
                          leaf_index=[f"{tree_index}_{leaf_index}"], label_probas=probas)]

    # Extract child indices
    left_idx = int(tree[leaf_index]['left'])
    right_idx = int(tree[leaf_index]['right'])

    # Recursive calls
    l_conjunctions = extractConjunctionsFromTree(tree, tree_index, left_idx, feature_dict, label_names, class_index)
    r_conjunctions = extractConjunctionsFromTree(tree, tree_index, right_idx, feature_dict, label_names, class_index)

    # Vectorized condition addition
    feature_idx = feature_dict[tree[leaf_index]['feature']]
    threshold = float(tree[leaf_index]['value'])

    # Batch process conditions
    for c in l_conjunctions:
        c.addCondition(feature_idx, threshold, 'upper')
    for c in r_conjunctions:
        c.addCondition(feature_idx, threshold, 'lower')

    return l_conjunctions + r_conjunctions

def merge_two_conjunction_sets(conj_list1, conj_list2):
    """Vectorized conjunction merging with early contradiction detection."""
    if not conj_list1 or not conj_list2:
        return []

    # Pre-allocate result list
    new_conjunction_list = []
    new_conjunction_list.extend([
        c1.merge(c2)
        for c1 in conj_list1
        for c2 in conj_list2
        if not c1.isContradict(c2)
    ])

    return new_conjunction_list

def postProcessTrees(conjunction_sets, num_of_labels):
    """Vectorized tree post-processing with numpy array operations."""
    if num_of_labels <= 1:
        return list(conjunction_sets.values()) if isinstance(conjunction_sets, dict) else conjunction_sets

    # Convert to list if dict
    conj_values = list(conjunction_sets.values()) if isinstance(conjunction_sets, dict) else conjunction_sets

    # Vectorized grouping and merging
    num_trees = len(conj_values) // num_of_labels
    new_conj_list = []

    for tree_idx in range(num_trees):
        start_idx = tree_idx * num_of_labels
        tree_conjunctions = conj_values[start_idx:start_idx + num_of_labels]

        # Reduce with vectorized merge
        merged_conj = tree_conjunctions[0]
        for conj in tree_conjunctions[1:]:
            merged_conj = merge_two_conjunction_sets(merged_conj, conj)

        new_conj_list.append(merged_conj)

    return new_conj_list

def extractConjunctionSetsFromForest(model, unique_labels, features):
    """Vectorized forest processing with optimized data structures."""
    trees = extractNodesFromModel(model)
    num_of_labels = len(unique_labels)

    # Vectorized feature dictionary creation
    feature_dict = {feature: idx for idx, feature in enumerate(features)}

    # Pre-allocate conjunction sets
    conjunction_sets = {}

    # Vectorized tree processing
    for tree_idx, tree_nodes in enumerate(trees):
        # Create indexed tree dictionary in one operation
        indexed_tree = {int(node['node_index']): node for node in tree_nodes}
        class_idx = tree_idx % num_of_labels

        conjunction_sets[tree_idx] = extractConjunctionsFromTree(
            indexed_tree, tree_idx, 0, feature_dict, unique_labels, class_idx
        )

    # Conditional post-processing
    return (postProcessTrees(conjunction_sets, num_of_labels)
            if num_of_labels > 2
            else list(conjunction_sets.values()))