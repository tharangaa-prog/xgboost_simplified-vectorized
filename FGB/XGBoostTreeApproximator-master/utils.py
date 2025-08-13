"""
This module contains several functions that are used in various stages of the process
"""
import numpy as np
from sklearn.metrics import roc_curve, auc
import xgboost as xg
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import random

RANDOM_SEED = 1

def softmax(x):
    """
    This function is useful for converting the aggregated results come from the different trees into class probabilities
    :param x: Numpy k-dimensional array
    :return: Softmax of X
    """
    exp_x = np.exp(x - np.max(x))  # Numerical stability
    return exp_x / np.sum(exp_x, keepdims=True)

def get_auc(test_y, y_score):
    """
    :param test_y: Labels
    :param y_score: probabilities of labels
    :return: ROC AUC score
    """
    np.random.seed(RANDOM_SEED)
    n_classes = y_score.shape[1]
    y_test_binarize = np.eye(n_classes)[test_y]  # Vectorized one-hot encoding
    fpr, tpr, _ = roc_curve(y_test_binarize.ravel(), y_score.ravel())
    return auc(fpr, tpr)

def train_decision_tree(train, feature_cols, label_col):
    """
    This function gets a dataframe as an input and optimizes a decision tree to the data
    """
    np.random.seed(RANDOM_SEED)
    parameters = {'criterion': ['entropy', 'gini'],
                  'max_depth': [3, 5, 10, 20, 50],
                  'min_samples_leaf': [1, 2, 5, 10]}
    model = DecisionTreeClassifier()
    clfGS = GridSearchCV(model, parameters, cv=3)
    clfGS.fit(train[feature_cols].values, train[label_col])
    return clfGS.best_estimator_

def train_rf_model(train, feature_cols, label_col):
    """
    This function gets a dataframe as an input and optimizes a random forest classifier to the data
    """
    np.random.seed(RANDOM_SEED)
    parameters = {'n_estimators': [50, 100],
                  'criterion': ['entropy'],
                  'min_samples_leaf': [1, 10, 100],
                  'max_features': ['auto', 'log2']}
    model = RandomForestClassifier()
    clfGS = GridSearchCV(model, parameters, cv=3)
    clfGS.fit(train[feature_cols].values, train[label_col])
    return clfGS.best_estimator_

def train_xgb_classifier(train, feature_cols, label_col, xgb_params):
    """
    Train an XGBoost to the input dataframe
    """
    np.random.seed(RANDOM_SEED)
    tuning_params = {'colsample_bytree': [0.3, 0.5, 0.9],
                     'learning_rate': [0.01, 0.1],
                     'max_depth': [2, 5, 10],
                     'alpha': [1, 10],
                     'n_estimators': [50, 100]}
    if train[label_col].nunique() > 2:
        xgb_params['objective'] = "multi:softprob"
    else:
        xgb_params['objective'] = "binary:logitraw"
    model = xg.XGBClassifier(xgb_params)
    clfGS = GridSearchCV(model, tuning_params, cv=3)
    clfGS.fit(train[feature_cols], train[label_col])
    return clfGS.best_estimator_

def decision_tree_instance_depth(inst, dt):
    """
    :param inst: Instance to be inferenced - numpy vector
    :param dt: sklearn decision tree
    :return: The depth of the leaf that corresponds the instance
    """
    indx = 0
    depth = 0
    epsilon = 0.0000001
    t = dt.tree_
    while t.feature[indx] >= 0:
        if inst[t.feature[indx]] <= t.threshold[indx] + epsilon:
            indx = t.children_left[indx]
        else:
            indx = t.children_right[indx]
        depth += 1
    return depth

def decision_tree_depths(test, feature_cols, dt):
    """
    Vectorized calculation of prediction depths for all instances
    """
    X = test[feature_cols].values
    n_samples = X.shape[0]
    depths = np.zeros(n_samples, dtype=int)

    # Get tree structure once
    t = dt.tree_
    epsilon = 0.0000001

    # Process all samples simultaneously
    indices = np.zeros(n_samples, dtype=int)  # Current node for each sample
    active = np.ones(n_samples, dtype=bool)   # Which samples are still active

    while np.any(active):
        # Get current features and thresholds for active samples
        curr_indices = indices[active]
        curr_features = t.feature[curr_indices]

        # Find samples that haven't reached leaves
        non_leaf_mask = curr_features >= 0
        if not np.any(non_leaf_mask):
            break

        # Get active samples that are at non-leaf nodes
        active_non_leaf = np.where(active)[0][non_leaf_mask]

        if len(active_non_leaf) == 0:
            break

        # Vectorized comparison for branching
        curr_thresholds = t.threshold[indices[active_non_leaf]]
        feature_vals = X[active_non_leaf, curr_features[non_leaf_mask]]

        # Determine which way to branch
        go_left = feature_vals <= curr_thresholds + epsilon

        # Update indices
        left_children = t.children_left[indices[active_non_leaf]]
        right_children = t.children_right[indices[active_non_leaf]]

        indices[active_non_leaf] = np.where(go_left, left_children, right_children)
        depths[active_non_leaf] += 1

        # Deactivate samples that reached leaves
        new_features = t.feature[indices[active_non_leaf]]
        reached_leaf = new_features < 0
        if np.any(reached_leaf):
            leaf_samples = active_non_leaf[reached_leaf]
            active[leaf_samples] = False

    return depths.tolist()

# Unused functions (kept for compatibility)
def train_xgb_classifier2(train, feature_cols, label_col, xgb_params):
    if train[label_col].nunique() > 2:
        obj = "multi:softprob"
    else:
        obj = "binary:logitraw"
    xgb_model = xg.XGBClassifier(**xgb_params)
    xgb_model.fit(train[feature_cols], train[label_col])
    return xgb_model

def ensemble_prediction_depth(X, rf):
    """Vectorized ensemble depth calculation"""
    n_samples = X.shape[0]
    total_depths = np.zeros(n_samples)

    for estimator in rf.estimators_:
        depths = np.array([tree_prediction_depth(inst, estimator.tree_) for inst in X])
        total_depths += depths

    return total_depths.tolist()

def tree_prediction_depth(inst, t):
    indx = 0
    depth = 0
    epsilon = 0.0000001
    while t.feature[indx] >= 0:
        if inst[t.feature[indx]] <= t.threshold[indx] + epsilon:
            indx = t.children_left[indx]
        else:
            indx = t.children_right[indx]
        depth += 1
    return depth

def get_features_statistics(data):
    """Vectorized feature statistics calculation"""
    return (data.min().to_dict(),
            data.max().to_dict(),
            data.mean().to_dict())