import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from tqdm.auto import tqdm
import ast
import contextlib
from data_preprocessing import precompute_detailed_embeddings, SteamFeatureExtractor, CorrelationRemover

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

# Loading clean dataset
DATASET_PATH = "../dataset/clean_data/clean_dataset.csv"
df = pd.read_csv(DATASET_PATH)

X = df.drop(columns=['target_owners', 'name'])
y = df['target_owners'].astype(int)

# Reducing the dataset in order to speed up the tuning 
PROTOTYPE_FRAC = 0.20
X_proto, _, y_proto, _ = train_test_split(X, y, train_size=PROTOTYPE_FRAC, stratify=y, random_state=42)

# From string to python lists
list_columns = ['categories', 'genres', 'tags', 'publishers', 'developers', 'languages']
for col in list_columns:
    if col in df.columns:
        X_proto[col] = X_proto[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Copy of the prototype dataset in order to analyze the embeddings
df_proto = X_proto.copy()
df_proto['target_owners'] = y_proto

# Using the function of data_preprocessing for the embeddings calculation on the prototype dataset
df_proto_processed, emb_cols = precompute_detailed_embeddings(df_proto, text_col='detailed_description')

# Saving final x_proto and y_proto
X_proto = df_proto_processed.drop(columns=['target_owners'])
y_proto = df_proto_processed['target_owners']

print(f"Reduced dataset size: {len(X_proto)} elements")
print(f"Class distribution maintained:\n{y_proto.value_counts(normalize=True)}")

# Definition of the numeric columns
numeric_cols = ['price', 'days_since_release', 'min_ram_gb', 'rec_ram_gb', 'required_age', 'num_dlc', 
                'num_achievements', 'num_languages_supported', 'metacritic_score', 
                'review_ratio', 'discount', 'average_forever', 'average_2weeks', 
                'median_forever', 'median_2weeks']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), 
    ('scaler', RobustScaler())
])

# The ColumnTransformer handles PCA in isolation on the pre-calculated columns.
# fit_transform will occur dynamically within each individual fold of the CV
scaler_and_pca = ColumnTransformer(
    transformers=[
        ('num_pipeline', numeric_transformer, numeric_cols),
        ('pca_pipeline', PCA(n_components=50), emb_cols)
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
)

# Final pipeline
pipe = Pipeline([
    ('steam_extractor', SteamFeatureExtractor()),
    ('scaler_and_pca', scaler_and_pca),
    ('corr_remover', CorrelationRemover(threshold=0.95)), 
    ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=8))),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=8))
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
    n_jobs=1,
    verbose=3,
    error_score='raise'
)

# Nested CV for stability evaluation
print("\nStarting Nested Cross-Validation on the prototype...")
NUM_OUTER_FITS = (cv_inner.n_splits * 24 + 1) * cv_outer.n_splits

scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

with tqdm_joblib(tqdm(desc="Nested CV Progress", total=NUM_OUTER_FITS)):
    cv_results = cross_validate(
        grid_search, X_proto, y_proto, 
        cv=cv_outer, 
        scoring=scoring_metrics, 
        n_jobs=1,
        return_train_score=False
    )

print("\nRESULTS NESTED CV")
print(f"Accuracy (Mean ± Std): {np.mean(cv_results['test_accuracy']):.3f} ± {np.std(cv_results['test_accuracy']):.3f}")
print(f"F1 Macro (Mean ± Std): {np.mean(cv_results['test_f1_macro']):.3f} ± {np.std(cv_results['test_f1_macro']):.3f}")

# Best params research
print("\nFinal GridSearch training on the prototype to extract parameters...")
NUM_INNER_FITS = cv_inner.n_splits * 24

with tqdm_joblib(tqdm(desc="Parameter Extraction", total=NUM_INNER_FITS)):
    grid_search.fit(X_proto, y_proto)

print("\nBest parameters:")
print("Use this parameters in the training pipeline:")
for param, value in grid_search.best_params_.items():
    print(f" - {param}: {value}")
