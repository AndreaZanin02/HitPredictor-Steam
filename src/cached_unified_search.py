import numpy as np
import pandas as pd
from tempfile import mkdtemp
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Assuming these match your project's local module imports
# from data_preprocessing import SteamFeatureExtractor, FeatureNameSanitizer, PLSDAWrapper

# 1. Create a persistent temporary directory for pipeline step caching.
# This prevents redundant re-runs of TF-IDF and PLS-DA when testing different model parameters.
cachedir = mkdtemp()
print(f"Pipeline caching directory established at: {cachedir}")

# 2. Define the Unified Pipeline with the caching directory
pipeline = Pipeline([
    ('extractor', SteamFeatureExtractor()),      # Raw data -> Feature Matrix
    ('sanitizer', FeatureNameSanitizer()),       # Sanitizes column names
    ('pls_da', PLSDAWrapper()),                  # Supervised dimensionality reduction
    ('model', XGBClassifier(tree_method='hist', device='cuda')) # GPU-accelerated XGBoost
], memory=cachedir)                              # <--- Active disk cache

# 3. Construct the Parameter Grid
param_grid = {
    # Preprocessing / Feature Space parameters
    'pls_da__n_components': [5, 10, 20],
    'extractor__tfidf_max_features': [5, 10, 15],
    
    # Estimator / Tree structure parameters
    'model__max_depth': [3, 5, 7],
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.01, 0.1]
}

# 4. Initialize the joint grid search
# Setting n_jobs=-1 forces scikit-learn to utilize ALL available CPU cores.
unified_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,                    
    scoring='f1_macro',
    n_jobs=-1,               # <--- Parallelizes training across all physical CPU cores
    verbose=2
)

# 5. Run the optimization
# unified_search.fit(X_train, y_train)

# print(f"Best Unified Score: {unified_search.best_score_}")
# print(f"Best Parameter Matrix: {unified_search.best_params_}")

# Clean up the cache directory afterward if necessary
# import shutil
# shutil.rmtree(cachedir)

