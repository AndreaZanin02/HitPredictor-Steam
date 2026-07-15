import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Let's assume these are your custom imports
# from data_preprocessing import SteamFeatureExtractor, FeatureNameSanitizer, PLSDAWrapper

# 1. Define your unified Pipeline
# The pipeline acts as a single cohesive estimator where data flows sequentially
pipeline = Pipeline([
    ('extractor', SteamFeatureExtractor()),      # Raw data -> Feature Matrix
    ('sanitizer', FeatureNameSanitizer()),       # Clean up column names (brackets, etc.)
    ('pls_da', PLSDAWrapper()),                  # Supervised dimensionality reduction
    ('model', XGBClassifier(tree_method='hist', device='cuda')) # Final classifier
])

# 2. Construct the Joint Parameter Grid
# Use the double-underscore prefix ('stepname__parametername') to target specific steps
param_grid = {
    # Preprocessing / Feature Space parameters
    'pls_da__n_components': [5, 10, 20],
    'extractor__tfidf_max_features': [5, 10, 15],
    
    # Estimator / Tree structure parameters
    'model__max_depth': [3, 5, 7],
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.01, 0.1]
}

# 3. Initialize the joint grid search
# This evaluates all cross-combinations within a clean cross-validation loop
unified_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,                    # Nested inside your outer CV loop
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
)

# 4. Run the optimization
# When you call fit, the feature extractor, PLS-DA, and XGBoost are fit jointly!
# unified_search.fit(X_train, y_train)

# print(f"Best Unified Score: {unified_search.best_score_}")
# print(f"Best Parameter Matrix: {unified_search.best_params_}")

