"""
This module contain the Pruner function for pruning a decision forest
"""
import pandas as pd
import numpy as np
from utils import *
from sklearn.metrics import cohen_kappa_score

class Pruner():
    """
    A static class that supports the pruning of a decision forest
    """
    def predict_probas_tree(self, conjunctions, X):
        """
        Predict probabilities for X using a tree, represented as a conjunction set
        """
        n_instances = X.shape[0]
        probas = np.zeros((n_instances, len(conjunctions[0].label_probas)))

        for i, inst in enumerate(X):
            for conj in conjunctions:
                if conj.containsInstance(inst):
                    probas[i] = conj.label_probas
                    break
        return probas

    def predict_probas(self, forest, X):
        """
        Predict probabilities of X, using a decision forest
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Vectorized prediction aggregation
        all_predictions = np.array([self.predict_probas_tree(tree, X) for tree in forest])
        summed_predictions = all_predictions.sum(axis=0)
        return np.array([softmax(pred)[0] for pred in summed_predictions])

    def predict(self, forest, X):
        """
        Predict labels of X, using a decision forest
        """
        return np.argmax(self.predict_probas(forest, X), axis=1)

    def get_forest_auc(self, forest, X, Y):
        """
        Calculates predictions ROC AUC
        """
        y_probas = self.predict_probas(forest, X)
        return get_auc(Y, y_probas)

    def forests_kappa_score(self, probas1, probas2):
        """
        Calculates Cohen's kappa of the predictions divided from two vectors of class probabilities
        """
        predictions1 = np.argmax(probas1, axis=1)
        predictions2 = np.argmax(probas2, axis=1)  # Fixed bug: was using probas1 twice
        return cohen_kappa_score(predictions1, predictions2)

    def kappa_based_pruning(self, forest, X, Y, min_forest_size=10):
        """
        This method conduct a kappa-based ensemble pruning.
        """
        # Vectorized AUC calculation for all trees
        tree_aucs = np.array([self.get_forest_auc([t], X, Y) for t in forest])
        selected_indexes = [np.argmax(tree_aucs)]

        previous_auc = 0
        current_auc = get_auc(Y, self.predict_probas([forest[selected_indexes[0]]], X))
        new_forest = [forest[selected_indexes[0]]]

        while current_auc > previous_auc or len(new_forest) <= min_forest_size:
            # Vectorized kappa calculation
            kappas = np.ones(len(forest))  # Default to 1 for selected trees
            new_forest_probas = self.predict_probas(new_forest, X)

            for i, tree in enumerate(forest):
                if i not in selected_indexes:
                    tree_probas = self.predict_probas([tree], X)
                    kappas[i] = self.forests_kappa_score(new_forest_probas, tree_probas)

            new_index = np.argmin(kappas)
            if new_index in selected_indexes:
                break

            selected_indexes.append(new_index)
            previous_auc = current_auc
            new_forest.append(forest[new_index])
            current_auc = get_auc(Y, self.predict_probas(new_forest, X))

        return new_forest

    def max_auc_pruning(self, forest, X, Y, min_forest_size=10):
        """
        This method conduct an ensemble pruning using a greedy algorithm that maximizes the AUC on the given dataset.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Vectorized prediction computation for all trees
        trees_predictions = {i: self.predict_probas_tree(forest[i], X) for i in range(len(forest))}
        tree_aucs = np.array([get_auc(Y, trees_predictions[i]) for i in range(len(forest))])
        selected_indexes = [np.argmax(tree_aucs)]

        previous_auc = 0
        best_auc = tree_aucs[selected_indexes[0]]

        while best_auc > previous_auc or len(selected_indexes) <= min_forest_size:
            previous_auc = best_auc
            best_index = None

            # Vectorized AUC computation for remaining trees
            remaining_indices = [i for i in range(len(forest)) if i not in selected_indexes]

            for i in remaining_indices:
                # Vectorized prediction aggregation
                selected_predictions = np.array([trees_predictions[idx] for idx in selected_indexes + [i]])
                aggregated_probas = np.array([softmax(pred)[0] for pred in selected_predictions.sum(axis=0)])
                temp_auc = get_auc(Y, aggregated_probas)

                if temp_auc > best_auc or best_index is None:
                    best_auc = temp_auc
                    best_index = i

            selected_indexes.append(best_index)

        print('Pruned forest training set AUC: ' + str(best_auc))
        return [forest[i] for i in selected_indexes]