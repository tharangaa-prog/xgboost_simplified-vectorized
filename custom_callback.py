from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def rfe_with_callback(estimator, X_train, y_train, X_test=None, y_test=None,
                      n_features_to_select=1, callback=None):
    n_features = X_train.shape[1]
    features = np.arange(n_features)
    
    while len(features) > n_features_to_select:
        # Fit on current features
        estimator.fit(X_train[:, features], y_train)
        
        # Get feature importances
        if hasattr(estimator, "coef_"):
            importances = np.abs(estimator.coef_).sum(axis=0)
        elif hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        else:
            raise ValueError("Estimator has no coef_ or feature_importances_")
        
        # Remove the least important feature
        remove_idx = np.argmin(importances)
        removed_feature = features[remove_idx]
        features = np.delete(features, remove_idx)
        
        # Callback - pass the fitted estimator that matches the remaining features
        if callback:
            # Refit the model on the remaining features for accurate evaluation
            estimator.fit(X_train[:, features], y_train)
            callback(step_features=features, removed_feature=removed_feature,
                     estimator=estimator, X_test=X_test, y_test=y_test)
    
    return features

# Callback function
def my_callback(step_features, removed_feature, estimator, X_test=None, y_test=None):
    print(f"Removed feature: {removed_feature}")
    print(f"Remaining features: {step_features}")
    if X_test is not None and y_test is not None:
        y_pred = estimator.predict(X_test[:, step_features])
        acc = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy after this step: {acc:.4f}\n")

# Run RFE with callback
model = LogisticRegression(max_iter=500)
selected_features = rfe_with_callback(model, X_train, y_train,
                                      X_test=X_test, y_test=y_test,
                                      n_features_to_select=2, callback=my_callback)

print(f"Final selected features: {selected_features}")

# Final model performance
model.fit(X_train[:, selected_features], y_train)
final_pred = model.predict(X_test[:, selected_features])
final_acc = accuracy_score(y_test, final_pred)
print(f"Final model accuracy with {len(selected_features)} features: {final_acc:.4f}")