"""
This module contains the conjunction class

"""
import numpy as np
from utils import *

class Conjunction():
    """
    A conjunction is a combination of feature bounds mapped into a class probability vector
    """
    def __init__(self,feature_names,label_names,leaf_index=None,label_probas=None):
        """
        :param feature_names: list of strings. Also determine the dimensionality
        :param label_names: list of labels. Determines the number of labels too
        :param leaf_index: This feature is optional. Can be relevant if we'd like to document the leaves that were used from the input forest
        :param label_probas: also optional. Relevant if we'd like to determine the class probabilities within the constructor
        """
        self.feature_names = feature_names
        self.number_of_features = len(feature_names)
        self.label_names = label_names

        # upper and lower bounds of the feature for each rule
        self.features_upper = [np.inf] * len(feature_names)
        self.features_lower = [-np.inf] * len(feature_names)

        self.label_probas = np.array(label_probas)
        self.leaf_index = leaf_index

        #The following dict is used for excluding irrelevant merges of different dummy variables that come from the same categorical feature
        self.categorical_features_dict={}

    def addCondition(self, feature, threshold, bound):
        """
        This method adds a condition to the conjunction if relevant (rule isn't already contained in the conjunction)

        :param feature: relevant feature
        :param threshold: upper\lower bound
        :param bound: bound direction

        """
        #Check if the rule isn't already contained in the conjunction
        if bound == 'lower':
            if self.features_lower[feature] < threshold:
                self.features_lower[feature] = threshold
        else:
            if self.features_upper[feature] > threshold:
                self.features_upper[feature] = threshold

        #Address categorical features:
        if '=' in self.feature_names[feature] and threshold >= 1 and bound == 'lower':
            splitted = self.feature_names[feature].split('=')
            self.categorical_features_dict[splitted[0]] = splitted[1]

    import numpy as np

    def isContradict(self, other_conjunction):
        """
        :param other_conjunction: conjunction object
        :return: True if other and self have at least one contradiction, otherwise False
        """

        # Vectorized bounds contradiction check
        upper_bounds = np.array(self.features_upper)
        lower_bounds = np.array(self.features_lower)
        other_upper = np.array(other_conjunction.features_upper)
        other_lower = np.array(other_conjunction.features_lower)

        if np.any((upper_bounds <= other_lower) | (lower_bounds >= other_upper)):
            return True

        # Vectorized categorical features contradiction
        common_features = set(self.categorical_features_dict.keys()) & set(
            other_conjunction.categorical_features_dict.keys())
        if common_features:
            self_values = np.array([self.categorical_features_dict[f] for f in common_features])
            other_values = np.array([other_conjunction.categorical_features_dict[f] for f in common_features])
            if np.any(self_values != other_values):
                return True

        return False

    def merge(self, other):
        """
        :param other: conjunction
        :return: new_conjunction - a merge of the self conjunction with other
        """
        new_conjunction = Conjunction(self.feature_names, self.label_names,
                                      self.leaf_index + other.leaf_index, self.label_probas + other.label_probas)

        # Vectorized min/max operations
        new_conjunction.features_upper = np.minimum(self.features_upper, other.features_upper).tolist()
        new_conjunction.features_lower = np.maximum(self.features_lower, other.features_lower).tolist()

        new_conjunction.categorical_features_dict = self.categorical_features_dict.copy()
        new_conjunction.categorical_features_dict.update(other.categorical_features_dict)
        return new_conjunction

    def containsInstance(self, inst):
        """
        Checks whether the input instance falls under the conjunction
        :param inst:
        :return: True if instance is contained
        """
        inst_array = np.array(inst)
        lower_bounds = np.array(self.features_lower)
        upper_bounds = np.array(self.features_upper)

        return np.all((inst_array >= lower_bounds) & (inst_array < upper_bounds))

    def has_low_interval(self, lowest_intervals):
        """Check if any interval is below threshold"""
        intervals = np.array(self.features_upper) - np.array(self.features_lower)
        return np.any(intervals < np.array(lowest_intervals))

    def predict_probas(self):
        """
        :return: softmax of the result vector
        """
        return softmax(self.label_probas)

    def toString(self):
        """
        This function creates a string representation of the conjunction (only for demonstration purposes)
        """
        s = ""

        # Vectorized lower bounds processing
        lower_bounds = np.array(self.features_lower)
        valid_lower = lower_bounds != -np.inf
        if np.any(valid_lower):
            lower_indices = np.where(valid_lower)[0]
            for i in lower_indices:
                s += self.feature_names[i] + ' >= ' + str(np.round(lower_bounds[i], 3)) + ", "

        # Vectorized upper bounds processing
        upper_bounds = np.array(self.features_upper)
        valid_upper = upper_bounds != np.inf
        if np.any(valid_upper):
            upper_indices = np.where(valid_upper)[0]
            for i in upper_indices:
                s += self.feature_names[i] + ' < ' + str(np.round(upper_bounds[i], 3)) + ", "

        s += 'labels: [' + str(self.label_probas) + ']'
        return s

    def get_data_point(self, min_values, max_values, mean_values):
        """Get representative data point for conjunction"""
        lower_bounds = np.array(self.features_lower)
        upper_bounds = np.array(self.features_upper)

        # Vectorized condition check
        use_mean = (lower_bounds == -np.inf) & (upper_bounds == np.inf)

        X = np.zeros(len(self.feature_names))

        # Use mean values where bounds are infinite
        mean_mask = use_mean
        X[mean_mask] = [mean_values[self.feature_names[i]] for i in np.where(mean_mask)[0]]

        # Calculate midpoint for bounded features
        bounded_mask = ~use_mean
        if np.any(bounded_mask):
            bounded_indices = np.where(bounded_mask)[0]
            for i in bounded_indices:
                feature = self.feature_names[i]
                lower = max(min_values[feature], lower_bounds[i])
                upper = min(max_values[feature], upper_bounds[i])
                X[i] = (lower + upper) / 2

        return X