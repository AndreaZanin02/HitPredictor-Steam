import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin

# Import custom pre-processing
from data_preprocessing import (
    fit_categorical_features, 
    transform_categorical_features,
    fit_text_features, 
    transform_text_features
)

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

# ----------------------- Loading and splitting dataset -----------------------
DATASET_PATH = "../dataset/clean_data/clean_dataset.csv"
df = pd.read_csv(DATASET_PATH)

# Removing target column (y) and the name of the games from features (X)
X = df.drop(columns=['target_owners', 'name'])
y = df['target_owners'].astype(int)

# Splitting the dataset
# test_size indicates the dimension of test set (ex: 0.3 --> 30% of dataset)
# stratify=y ensures the same class ratio between test and training sets
# random_state=fixed_number ensure deterministic results
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------- Training pipeline ------------------------
# Scaler defined only for continuous variables
# The rest of the variables will pass through unchanged
numeric_cols = ['price', 'days_since_release', 'min_ram_gb', 
                'required_age', 'num_dlc', 'num_achievements', 
                'num_languages_supported', 'metacritic_score', 'review_ratio',
                'discount', 'average_forever', 'average_2weeks', 'median_forever',
                'median_2weeks'
                ]
scaler_step = ColumnTransformer(
    transformers=[
        ('num_scaler', RobustScaler(), numeric_cols)
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
)

feature_selector = SelectFromModel(
    RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
    threshold='median'
)

pipe = Pipeline([
    ('steam_extractor', SteamFeatureExtractor()),
    ('scaler', scaler_step),
    ('corr_remover', CorrelationRemover(threshold=0.95)), 
    ('feature_selection', feature_selector),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# ----------------- Nested Cross_validation ---------------------
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 20],
    'classifier__min_samples_split': [2, 5]


    # The feature_selection__threshold looks for the best threashold to use for selecting feature
    # 'median': Keeps features whose importance is higher than the global median. It discards exactly 50% of the features
    # '0.75*median': Is more lenient. It lets more features through, useful if the final model needs to capture finer nuances
    # '1.25*median': Is more stringent. It lets fewer features through, keeping only the truly strong ones. It reduces the risk of overfitting
    # 'mean': Uses the mathematical average of the feature importance
    # It is usually very stringent because a few very important features dramatically raise the average
    # IMPORTANT: this research is very very low
    'feature_selection__threshold': ['median', '0.75*median', '1.25*median', 'mean']
}

cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipe, 
    param_grid=param_grid, 
    cv=cv_inner, 
    scoring='f1_macro', 
    n_jobs=-1,
    verbose=2
)

print("\nRunning Nested Cross-Validation (it will take a lot of time)...")
scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

cv_results = cross_validate(
    grid_search, X_train, y_train, 
    cv=cv_outer, 
    scoring=scoring_metrics, 
    n_jobs=-1,
    return_train_score=False
)

print("\nRESULTS OF OUTER LOOP")
print(f"Accuracy (Mean ± Std): {np.mean(cv_results['test_accuracy']):.3f} ± {np.std(cv_results['test_accuracy']):.3f}")
print(f"F1 Macro (Mean ± Std): {np.mean(cv_results['test_f1_macro']):.3f} ± {np.std(cv_results['test_f1_macro']):.3f}")

# ----------------- Final training ----------------
print("\nTraining final model...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("\nBest hyperparameter found:")
for param, value in grid_search.best_params_.items():
    print(f" - {param}: {value}")

print("\nEvaluating on the test set...")
y_pred = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))