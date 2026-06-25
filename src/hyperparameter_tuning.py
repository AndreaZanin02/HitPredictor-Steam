import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from tqdm.auto import tqdm
import contextlib

from data_preprocessing import (
    fit_categorical_features, 
    transform_categorical_features,
    fit_text_features, 
    transform_text_features
)
from sklearn.base import BaseEstimator, TransformerMixin

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to force joblib to communicate with the tqdm bar
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

sklearn.set_config(transform_output="pandas")

# ------------------- Pre-processing classes ----------------------
class SteamFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Encapsulates custom preprocessing to prevent data leakage
    during cross-validation
    """
    def fit(self, X, y=None):
        # The fit occurs only on the training portion of the current fold
        self.cat_artifacts_ = fit_categorical_features(X, top_n=50)
        self.text_artifacts_ = fit_text_features(X)
        return self
        
    def transform(self, X):
        X_enc = transform_categorical_features(X, self.cat_artifacts_)
        X_enc = transform_text_features(X_enc, self.text_artifacts_)
        # Removing textual residues
        return X_enc.select_dtypes(include=['number', 'bool'])

class CorrelationRemover(BaseEstimator, TransformerMixin):
    """
    Dynamically removes highly correlated features based on the training fold distribution
    """
    def __init__(self, threshold=0.97):
        self.threshold = threshold
        self.to_drop_ = []

    def fit(self, X, y=None):
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_ = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self

    def transform(self, X):
        return X.drop(columns=self.to_drop_, errors='ignore')

# ----------------------- Loading and splitting the dataset -----------------------
DATASET_PATH = "../dataset/clean_data/clean_dataset.csv"
df = pd.read_csv(DATASET_PATH)

X = df.drop(columns=['target_owners', 'name'])
y = df['target_owners'].astype(int)

# Tuning prototype mode
PROTOTYPE_FRAC = 0.15
print(f"\nATTENTION: Prototype mode is active. Using {PROTOTYPE_FRAC*100}% of the dataset")

# Extracting the subset while keeping the stratification intact
X_proto, _, y_proto, _ = train_test_split(
    X, y, 
    train_size=PROTOTYPE_FRAC, 
    stratify=y, 
    random_state=42
)

print(f"Reduced dataset size: {len(X)} elements")
print(f"Class distribution maintained:\n{y.value_counts(normalize=True)}")

# ---------------------- Pipeline Setup ------------------------
# Scaler defined only for continuous variables
# The rest of the variables will pass through unchanged
numeric_cols = ['price', 'days_since_release', 'min_ram_gb', 
                'required_age', 'num_dlc', 'num_achievements', 
                'num_languages_supported', 'metacritic_score', 'review_ratio',
                'discount', 'average_forever', 'average_2weeks', 'median_forever',
                'median_2weeks']

scaler_step = ColumnTransformer(
    transformers=[('num_scaler', RobustScaler(), numeric_cols)],
    remainder='passthrough',
    verbose_feature_names_out=False
)

feature_selector = SelectFromModel(
    RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
)

pipe = Pipeline([
    ('steam_extractor', SteamFeatureExtractor()),
    ('scaler', scaler_step),
    ('corr_remover', CorrelationRemover(threshold=0.95)), 
    ('feature_selection', feature_selector),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# ----------------- Hyperparameter Tuning ---------------------
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 20],
    'classifier__min_samples_split': [2, 5],

    # The feature_selection__threshold looks for the best threashold to use for selecting feature
    # 'median': Keeps features whose importance is higher than the global median. It discards exactly 50% of the features
    # '0.75*median': Is more lenient. It lets more features through, useful if the final model needs to capture finer nuances
    # '1.25*median': Is more stringent. It lets fewer features through, keeping only the truly strong ones. It reduces the risk of overfitting
    # 'mean': Uses the mathematical average of the feature importance
    # It is usually very stringent because a few very important features dramatically raise the average
    'feature_selection__threshold': ['median', '0.75*median', 'mean']
}

# Inner and outer loops
cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipe, 
    param_grid=param_grid, 
    cv=cv_inner, 
    scoring='f1_macro', 
    n_jobs=-1,
    verbose=0 
)

# Nested CV for stability evaluation
print("\nStarting Nested Cross-Validation on the prototype...")
NUM_OUTER_FITS = len(cv_outer.split(X_proto, y_proto)) * len(cv_inner.split(X_proto, y_proto)) * 24

scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

with tqdm_joblib(tqdm(desc="Nested CV Progress", total=NUM_OUTER_FITS)):
    cv_results = cross_validate(
        grid_search, X_proto, y_proto, 
        cv=cv_outer, 
        scoring=scoring_metrics, 
        n_jobs=-1,
        return_train_score=False
    )

print("\nRESULTS NESTED CV")
print(f"Accuracy (Mean ± Std): {np.mean(cv_results['test_accuracy']):.3f} ± {np.std(cv_results['test_accuracy']):.3f}")
print(f"F1 Macro (Mean ± Std): {np.mean(cv_results['test_f1_macro']):.3f} ± {np.std(cv_results['test_f1_macro']):.3f}")

# Best params research
print("\nFinal GridSearch training on the prototype to extract parameters...")
NUM_INNER_FITS = len(cv_inner.split(X_proto, y_proto)) * 24

with tqdm_joblib(tqdm(desc="Parameter Extraction", total=NUM_INNER_FITS)):
    grid_search.fit(X_proto, y_proto)

print("\nBest parameters:")
print("Use this parameters in the training pipeline:")
for param, value in grid_search.best_params_.items():
    print(f" - {param}: {value}")