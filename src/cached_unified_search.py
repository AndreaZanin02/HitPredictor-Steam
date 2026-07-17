import argparse
import ast
import os
import shutil
import warnings
from tempfile import mkdtemp

import numpy as np
import pandas as pd
import sklearn
import torch
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import cohen_kappa_score, make_scorer

# Importing custom modules from your project
from data_preprocessing import (
    precompute_detailed_embeddings, SteamFeatureExtractor, CorrelationRemover,
    dynamic_undersample, FeatureNameSanitizer, DynamicSMOTENC, dynamic_oversample,
    SupervisedPLSDATransformer, get_base_preprocessor
)
from hyperparameter_tuning import WeightedXGBClassifier

# Global configuration
sklearn.set_config(transform_output="pandas")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
warnings.filterwarnings('ignore', category=UserWarning)

# Custom scorer: Quadratic Weighted Kappa or Macro F1 (matching your goals)
qwk_scorer = make_scorer(cohen_kappa_score, weights='quadratic')

def main():
    parser = argparse.ArgumentParser(description="Cached Unified Pipeline Optimization Sweep")
    parser.add_argument('-f', '--full', action='store_true', help="Use the entire dataset (else uses a 20% prototype)")
    parser.add_argument('-s', '--seed', type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument('-p', '--pre_release', action='store_true', help="Filter out post-release columns")
    args = parser.parse_args()

    SEED = args.seed
    DATASET_PATH = "../dataset/clean_data/clean_dataset.csv"

    print(f"Loading dataset from: {DATASET_PATH}")
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: File not found in {DATASET_PATH}")
        return

    # 1. Pre-release signal handling
    if args.pre_release:
        post_release_features = [
            'num_achievements', 'metacritic_score', 'review_ratio', 'num_dlc',
            'discount', 'average_forever', 'average_2weeks', 'median_forever', 'median_2weeks'
        ]
        cols_to_drop = [col for col in post_release_features if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        print(f"Pre-release mode: Discarded post-launch data columns.")

    X = df.drop(columns=['target_owners', 'name'])
    y = df['target_owners'].astype(int)

    # Use 20% prototype unless --full is flagged
    if not args.full:
        from sklearn.model_selection import train_test_split
        X, _, y, _ = train_test_split(X, y, train_size=0.20, stratify=y, random_state=SEED)
        print("Running optimization on 20% prototype split.")

    # Parse string lists back to Python lists for categorical extractor compatibility
    list_columns = ['categories', 'genres', 'tags', 'publishers', 'developers', 'languages']
    for col in list_columns:
        if col in X.columns:
            X[col] = X[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # 2. Extract and precompute description text embeddings (completely leak-safe)
    print("Precalculating semantic description embeddings...")
    df_temp = X.copy()
    df_temp['target_owners'] = y
    df_processed, emb_cols = precompute_detailed_embeddings(df_temp, text_col='detailed_description')
    
    X_processed = df_processed.drop(columns=['target_owners'])
    y_processed = df_processed['target_owners']

    # Identify valid numeric columns
    all_numeric_cols = [
        'price', 'release_year', 'min_ram_gb', 'rec_ram_gb', 'required_age', 
        'num_dlc', 'num_achievements', 'num_languages_supported', 'metacritic_score', 
        'review_ratio', 'discount', 'average_forever', 'average_2weeks', 
        'median_forever', 'median_2weeks'
    ]
    numeric_cols = [col for col in all_numeric_cols if col in X_processed.columns]
    categorical_cols = ['release_month']
    base_preprocessor = get_base_preprocessor(numeric_cols, categorical_cols)

    # 3. Setting up persistent cache directory
    cache_dir = os.path.join(".", "ml_cache", "pipeline_cache_unified")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Caching intermediate pipeline layers to disk path: {cache_dir}")

    # 4. Constructing the Complete ImbPipeline
    # Gridsearch sets the order of the searched hyperparameters based on alphabetical order.
    # Since tfidf and plsda change the number of columns in the dataset, invalidating the cache each time they change,
    # the numbers inserted in the name ensure that they are the first elements in the list,
    # i.e., those that change least frequently with each grid search iteration,
    # maximizing the cache's lifespan and its benefits
    pipeline = ImbPipeline([
        ('0_steam_extractor', SteamFeatureExtractor()), 
        ('base_preprocessor', base_preprocessor),
        ('rus', RandomUnderSampler(sampling_strategy=dynamic_undersample, random_state=SEED)),
        ('1_pls_da', SupervisedPLSDATransformer(emb_cols=emb_cols)), 
        ('corr_remover', CorrelationRemover(threshold=0.95)),
        ('smote_nc', DynamicSMOTENC(sampling_strategy=dynamic_oversample, k_neighbors=3, random_state=SEED)),
        ('tomek', TomekLinks()),
        ('sanitizer', FeatureNameSanitizer()),
        ('2_classifier', WeightedXGBClassifier(random_state=SEED, eval_metric='mlogloss', tree_method='hist', device=DEVICE))
    ], memory=cache_dir)

    # 5. Combined Parameter Search Space
    # Using specific class prefix tags aligned directly with the pipeline definition steps,
    # so as to change the hyperparameters that invalidate the cache as little as possible
    param_grid = {
        # Feature Extraction Search Space
        '0_steam_extractor__max_tfidf_features': [15, 30, 50],
        '1_pls_da__n_components': [20, 50, 80],
        
        # Classifier Hyperparameter Search Space
        '2_classifier__max_depth': [3, 6, 9],
        '2_classifier__n_estimators': [100, 200, 400],
        '2_classifier__learning_rate': [0.05, 0.1, 0.19]
    }

    print("Initializing Multi-Core Parallelized Unified Grid Search...")
    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_inner,
        scoring=qwk_scorer,
        n_jobs=-1,
        verbose=3,
        error_score='raise'
    )

    # 6. Fit and execute the joint optimization
    try:
        grid_search.fit(X_processed, y_processed)
        print("\n" + "="*50)
        print("🏆 OPTIMAL COMBINATION FOUND")
        print("="*50)
        print(f"Best score (QWK): {grid_search.best_score_:.4f}")
        print(f"Best Parameters:\n{grid_search.best_params_}")
    finally:
        # Clean up temporary caches to free workspace memory
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print("Completed workflow. Disk cache space freed.")

if __name__ == "__main__":
    main()