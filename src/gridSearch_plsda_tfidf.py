"""
Fast grid search to reduce the enormous computation time of hyperparameter_tuning and estimate
the best number of textual features (max_tfidf_features and pls_da__n_components), before using
them to tune the hyperparameters.
This is repeated for each of the three "lightweight" pre-tuning models to determine whether the optimal value is shared
or classifier-dependent, and then implement it in the tuning pipeline.
"""
import argparse
import ast
import warnings
from data_preprocessing import (
    precompute_detailed_embeddings, SteamFeatureExtractor, CorrelationRemover, 
    dynamic_undersample, FeatureNameSanitizer, DynamicSMOTENC, dynamic_oversample,
    SupervisedPLSDATransformer, get_base_preprocessor
)
from hyperparameter_tuning import WeightedXGBClassifier
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler, TomekLinks


# Global sklearn config: forces every transformer in the pipeline to output
# pandas DataFrames instead of raw numpy arrays, so column names survive
# each pipeline step
sklearn.set_config(transform_output="pandas")

# Use GPU for XGBoost if available, otherwise fall back to CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Silence noisy sklearn/imblearn UserWarnings that would otherwise clutter
# the grid-search logs
warnings.filterwarnings('ignore', category=UserWarning)

# Custom scorer: Quadratic Weighted Kappa, because the target is ordinal
# (ownership tiers 0-4) and QWK penalizes predictions that are "far" from 
# the true tier more than close misses
qwk_scorer = make_scorer(cohen_kappa_score, weights='quadratic')


def main():
    """
    Run a lightweight grid search over two preprocessing hyperparameters:
    -   max_tfidf_features (SteamFeatureExtractor): size of the TF-IDF vocabulary
        built from 'short_description'
    -   pls_da__n_components (SupervisedPLSDATransformer): number of PLS-DA
        components used to compress the precomputed text embeddings

    The search is performed with three "probe" models (DecisionTree,
    RandomForest, XGBoost) instead of the full hyperparameter_tuning models,
    to keep runtime manageable. Results are exported to a CSV so the best
    combination can be selected before running the full, much more expensive
    Optuna-based tuning in hyperparameter_tuning.py
    """

    # ----------------------- CLI arguments -------------------------
    parser = argparse.ArgumentParser(description="Tfidf/pls_da grid search")
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-p', '--pre_release', action='store_true')
    parser.add_argument('--sample_frac', type=float, default=1.0,
                         help="Fraction of the dataset to use (default 100%%)")
    parser.add_argument('--cv_folds', type=int, default=3)
    args = parser.parse_args()

    SEED = args.seed

    # ------------------------- Dataset loading --------------------------
    DATASET_PATH = "../dataset/clean_data/clean_dataset.csv"
    df = pd.read_csv(DATASET_PATH)

    # Optionally drop features that would only be known after a game's
    # release (achievements, reviews, discounts, playtime, ...), to
    # simulate the "pre-release" prediction scenario
    if args.pre_release:
        post_release_feature = [
            'num_achievements', 'metacritic_score', 'review_ratio', 'num_dlc',
            'discount', 'average_forever', 'average_2weeks', 'median_forever', 'median_2weeks'
        ]
        df = df.drop(columns=[c for c in post_release_feature if c in df.columns])

    # Target: 5-tier ordinal ownership class. 'name' is dropped as it carries
    # no predictive signal (pure identifier)
    X = df.drop(columns=['target_owners', 'name'])
    y = df['target_owners'].astype(int)

    # Optional subsampling for faster iteration during development.
    # Stratified so that the class proportions (heavily imbalanced towards
    # tier 0) are preserved in the reduced sample
    if args.sample_frac < 1.0:
        X, _, y, _ = train_test_split(X, y, train_size=args.sample_frac, stratify=y, random_state=SEED)

    # Columns that were exported as stringified Python lists by the cleaning
    # pipeline (CSV round-trip) need to be parsed back into real lists
    # before SteamFeatureExtractor can use MultiLabelBinarizer on them
    list_columns = ['categories', 'genres', 'tags', 'publishers', 'developers', 'languages']
    for col in list_columns:
        if col in X.columns:
            X[col] = X[col].apply(lambda v: ast.literal_eval(v) if isinstance(v, str) else v)

    # ------------------- Text embeddings -------------------------
    # Embeddings come from a frozen sentence-transformer model, so computing
    # them once here (instead of inside every CV fold/pipeline run) is safe
    # from a data leakage standpoint and saves a large amount of compute
    df_temp = X.copy()
    df_temp['target_owners'] = y
    df_processed, emb_cols = precompute_detailed_embeddings(df_temp, text_col='detailed_description')

    X_proc = df_processed.drop(columns=['target_owners'])
    y_proc = df_processed['target_owners']

    # -------------------- Preprocessing setup ------------------------
    all_numeric_cols = [
        'price', 'release_year', 'min_ram_gb', 'rec_ram_gb', 'required_age',
        'num_dlc', 'num_achievements', 'num_languages_supported', 'metacritic_score',
        'review_ratio', 'discount', 'average_forever', 'average_2weeks',
        'median_forever', 'median_2weeks'
    ]
    # Only keep numeric columns that actually survived pre_release filtering.
    numeric_cols = [c for c in all_numeric_cols if c in X_proc.columns]
    categorical_cols = ['release_month']
    base_preprocessor = get_base_preprocessor(numeric_cols, categorical_cols)

    # Use "Lightweight" probe models. They are cheap to train, which
    # makes them suitable for a brute-force grid search over many combinations
    probe_models = {
        'DecisionTree': DecisionTreeClassifier(class_weight='balanced', random_state=SEED),
        'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=SEED, n_jobs=-1),
        'XGBoost': WeightedXGBClassifier(n_estimators=100, random_state=SEED, n_jobs=-1,
                                        eval_metric='mlogloss', tree_method='hist', device=DEVICE)
    }

    # Hyperparameter grids under evaluation.
    tfidf_grid = [15, 30, 50]
    pls_grid = [20, 50, 80]

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=SEED)

    results = []
    total = len(probe_models) * len(tfidf_grid) * len(pls_grid)
    done = 0

    # ------------------ Grid search --------------------
    for model_name, clf in probe_models.items():
        for n_tfidf in tfidf_grid:
            for n_pls in pls_grid:
                done += 1
                print(f"[{done}/{total}] {model_name} | tfidf={n_tfidf} | pls={n_pls}")

                # Full pipeline: feature extraction -> scaling/encoding ->
                # undersampling of the majority class -> supervised PLS-DA
                # compression of the text embeddings -> correlation pruning ->
                # SMOTENC oversampling of minority classes -> Tomek links
                # cleanup -> feature-name sanitization -> classifier.
                pipe = ImbPipeline([
                    ('steam_extractor', SteamFeatureExtractor(max_tfidf_features=n_tfidf)),
                    ('base_preprocessor', base_preprocessor),
                    ('rus', RandomUnderSampler(sampling_strategy=dynamic_undersample, random_state=SEED)),
                    ('pls_da', SupervisedPLSDATransformer(emb_cols=emb_cols, n_components=n_pls)),
                    ('corr_remover', CorrelationRemover(threshold=0.95)),
                    ('smote_nc', DynamicSMOTENC(sampling_strategy=dynamic_oversample, k_neighbors=3, random_state=SEED)),
                    ('tomek', TomekLinks()),
                    ('sanitizer', FeatureNameSanitizer()),
                    ('classifier', clf)
                ])

                try:
                    # Evaluate the pipeline with QWK across cv folds
                    scores = cross_val_score(pipe, X_proc, y_proc, cv=cv, scoring=qwk_scorer,
                                              n_jobs=1, error_score='raise')
                    mean_score, std_score = scores.mean(), scores.std()
                except Exception as e:
                    # A failed combination (e.g. incompatible n_components vs
                    # available samples/features) should not stop the whole
                    # grid search: log it and record NaN scores instead
                    print(f"   -> FAILED: {e}")
                    mean_score, std_score = np.nan, np.nan

                results.append({
                    'model': model_name,
                    'max_tfidf_features': n_tfidf,
                    'pls_da_n_components': n_pls,
                    'qwk_mean': mean_score,
                    'qwk_std': std_score
                })

    # ------------------- Results export and summary ----------------------
    results_df = pd.DataFrame(results)
    results_df.to_csv("tfidf_pls_grid_results.csv", index=False)

    # Best (tfidf, pls) combination per individual model
    print("\n" + "="*60)
    print(" BEST COMBINATIONS")
    print("="*60)
    best_per_model = results_df.loc[results_df.groupby('model')['qwk_mean'].idxmax()]
    print(best_per_model.to_string(index=False))

    # Average QWK per (tfidf, pls) combination across all probe models,
    # useful to check whether the optimal setting is model-agnostic
    # before using it into the full tuning pipeline
    print("\n" + "="*60)
    print(" MEAN VALUE BETWEEN THE MODELS")
    print("="*60)
    avg_across_models = results_df.groupby(['max_tfidf_features', 'pls_da_n_components'])['qwk_mean'].mean()
    avg_across_models = avg_across_models.reset_index().sort_values('qwk_mean', ascending=False)
    print(avg_across_models.to_string(index=False))


if __name__ == "__main__":
    main()