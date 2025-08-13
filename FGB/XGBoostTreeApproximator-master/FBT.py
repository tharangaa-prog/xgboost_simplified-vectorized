"""
This module contains a forest based tree class (FBT).

The class takes an XGBoost as an input and generates a decision aims at preserving the predictive performance of
the XGboost model
"""
from conjunctionset import *
from tree import *
from pruning import *

class FBT():
    """
    This class creates a decision tree from an XGboost
    """
    def __init__(self,max_depth,min_forest_size,max_number_of_conjunctions,pruning_method=None):
        """

        :param max_depth: Maximum allowed depths of the generated tree
        :param min_forest_size: Minimum size of the pruned forest (relevant for the pruning stage)
        :param max_number_of_conjunctions:
        :param pruning_method: Pruning method. If None then there's no pruning. 'auc' is for greedy auc-bsed pruning
        :param xgb_model: Trained XGboost model
        """
        self.min_forest_size = min_forest_size
        self.max_number_of_conjunctions = max_number_of_conjunctions
        self.pruning_method = pruning_method
        self.max_depth = max_depth

    def fit(self,train,feature_cols,label_col, xgb_model, pruned_forest=None, trees_conjunctions_total=None):
        """
        Generates the decision tree by applying the following stages:
        1. Generating a conjunction set that represents each tree of the decision forest
        2. Prune the decision forest according to the given pruning approach
        3. Generate the conjunction set (stage 1 in the algorithm presented)
        4. Create a decision tree out of the generated conjunction set

        :param train: pandas dataframe that was used for training the XGBoost
        :param feature_cols: feature column names
        :param label_col: label column name
        :param xgb_model: XGBoost
        :param pruned_forest: A list of trees, represnt a post-pruning forest. Relevant mostly for the experiment presented in the paper
        :param tree_conjunctions: This para
        """
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.int_cols = [k for k,v in train[feature_cols].dtypes.items() if 'int' in str(v)]
        self.xgb_model = xgb_model
        if pruned_forest is None or trees_conjunctions_total is None:
            self.trees_conjunctions_total = extractConjunctionSetsFromForest(self.xgb_model,train[self.label_col].unique(),self.feature_cols)
            print('Start pruning')
            self.prune(train)
        else:
            self.pruner = Pruner()
            self.trees_conjunctions_total = trees_conjunctions_total
            self.trees_conjunctions = pruned_forest
        self.cs = ConjunctionSet(max_number_of_conjunctions=self.max_number_of_conjunctions)
        self.cs.fit(self.trees_conjunctions,train, feature_cols,label_col,int_features=self.int_cols)
        print('Start ordering conjunction set in a tree structure')
        self.tree = Tree(self.cs.conjunctions, self.cs.splitting_points,self.max_depth)
        self.tree.split()
        print('Construction of tree has been completed')

    def prune(self,train):
        """

        :param train: pandas dataframe used as a pruning dataset
        :return: creates a pruned decision forest (include only the relevant trees)
        """
        if self.pruning_method == None:
            self.trees_conjunctions = self.trees_conjunctions_total
        self.pruner = Pruner()
        if self.pruning_method == 'auc':
            self.trees_conjunctions = self.pruner.max_auc_pruning(self.trees_conjunctions_total, train[self.feature_cols],
                                                                      train[self.label_col], min_forest_size=self.min_forest_size)

    def predict_proba(self,X):
        """
        Returns class probabilities

        :param X: Pandas dataframe or a numpy matrix
        :return: class probabilities for the corresponding data
        """
        return self.tree.predict_proba(X)

    def predict(self, X):
        """
        Get predictions vector

        :param X: Pandas dataframe or a numpy matrix
        :return: Predicted classes
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def get_decision_paths(self, X):
        """
        :param X: Pandas data frame of [number_of_instances, number_of_features] dimension
        :return: A list of decision paths where each decision path represented as a string of nodes
        """
        paths = self.tree.get_decision_paths(X)

        processed_paths = []
        for path in paths:
            # Convert to pandas Series for vectorized operations
            nodes = pd.Series(path)

            # Handle label nodes (no change needed)
            label_mask = nodes.str.startswith('label')

            # Handle decision nodes with '<'
            lt_mask = nodes.str.contains('<', na=False) & ~label_mask
            # Handle decision nodes with '>='
            gte_mask = nodes.str.contains('>=', na=False) & ~label_mask

            # Process nodes vectorized
            processed_nodes = nodes.copy()

            if lt_mask.any():
                split_data = nodes[lt_mask].str.split('<', expand=True)
                processed_nodes[lt_mask] = (split_data[0].astype(int).map(lambda x: self.feature_cols[x]) +
                                            ' < ' + split_data[1])

            if gte_mask.any():
                split_data = nodes[gte_mask].str.split('>=', expand=True)
                processed_nodes[gte_mask] = (split_data[0].astype(int).map(lambda x: self.feature_cols[x]) +
                                             ' >= ' + split_data[1])

            processed_paths.append(processed_nodes.tolist())

        return processed_paths
