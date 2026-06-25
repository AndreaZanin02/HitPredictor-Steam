import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

from data_preprocessing import (
    fit_categorical_features, 
    transform_categorical_features,
    fit_text_features, 
    transform_text_features
)

sklearn.set_config(transform_output="pandas")

class SteamFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Encapsulates custom preprocessing to prevent data leakage
    during cross-validation
    """
    def fit(self, X, y=None):
        self.cat_artifacts_ = fit_categorical_features(X, top_n=50)
        self.text_artifacts_ = fit_text_features(X)
        return self
        
    def transform(self, X):
        X_enc = transform_categorical_features(X, self.cat_artifacts_)
        X_enc = transform_text_features(X_enc, self.text_artifacts_)
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

# ----------------------- Loading Full Dataset -----------------------
DATASET_PATH = "../dataset/clean_data/clean_dataset.csv"
df = pd.read_csv(DATASET_PATH)

X = df.drop(columns=['target_owners', 'name'])
y = df['target_owners'].astype(int)

# Split 80/20 of the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------- Pipeline Setup  ------------------------
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

# ------------- UPDATE WITH THE HYPERPARAMETER TUNING RESULTS --------------
BEST_THRESHOLD = 'median' 
BEST_N_ESTIMATORS = 200
BEST_MAX_DEPTH = 20
BEST_MIN_SAMPLES_SPLIT = 5
# --------------------------------------------------------------------------

feature_selector = SelectFromModel(
    RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
    threshold=BEST_THRESHOLD
)

pipe = Pipeline([
    ('steam_extractor', SteamFeatureExtractor()),
    ('scaler', scaler_step),
    ('corr_remover', CorrelationRemover(threshold=0.95)), 
    ('feature_selection', feature_selector),
    ('classifier', RandomForestClassifier(
        n_estimators=BEST_N_ESTIMATORS,
        max_depth=BEST_MAX_DEPTH,
        min_samples_split=BEST_MIN_SAMPLES_SPLIT,
        class_weight='balanced', 
        random_state=42,
        n_jobs=-1,
        verbose=2
    ))
])

# ----------------- Final Training and Evaluation ---------------------
print("Starting the training...")
pipe.fit(X_train, y_train)

print("\nTraining test set. Evaluation on test set...")
y_pred = pipe.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model
MODEL_PATH = "model.pkl"
joblib.dump(pipe, MODEL_PATH)
print(f"\nModel saved in: {MODEL_PATH}")