"""
Machine learning pipeline for game ownership prediction.
Implements a nested cross-validation strategy with hyperparameter tuning via Optuna.
Supports dynamic over/undersampling, feature selection, and pre-release feature filtering
"""
from xgboost import XGBClassifier
import argparse
import pandas as pd
import numpy as np
import sklearn
import torch
import json
from optuna.integration import OptunaSearchCV
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_selection import SelectFromModel
from joblib import Memory
import shutil
import os
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, cohen_kappa_score, make_scorer
)
from tqdm.auto import tqdm
import ast
import warnings
from data_preprocessing import (
    precompute_detailed_embeddings, SteamFeatureExtractor, CorrelationRemover,
    dynamic_undersample, FeatureNameSanitizer, DynamicSMOTENC, dynamic_oversample,
    SupervisedPLSDATransformer, get_base_preprocessor
)

# Output configuration and global variables
sklearn.set_config(transform_output="pandas")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
warnings.filterwarnings('ignore', category=UserWarning)

# Custom scorer for Optuna optimization
qwk_scorer = make_scorer(cohen_kappa_score, weights='quadratic')

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to seamlessly convert NumPy data types 
    into native Python types for JSON serialization
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


class WeightedXGBClassifier(XGBClassifier):
    """
    Custom XGBoost wrapper that dynamically computes and applies 
    balanced sample weights during the fitting phase
    """
    def fit(self, X, y, **kwargs):
        # Weights calculation
        weights = compute_sample_weight(class_weight='balanced', y=y)
        # Uses our weights in the fit() method of xgboost
        return super().fit(X, y, sample_weight=weights, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Nested CV Hyperparameter Tuning")
    parser.add_argument('-f', '--full', action='store_true', 
                        help="Use the entire dataset. If omitted, use a 20% prototype.")
    parser.add_argument('-s', '--seed', type=int, default=1, 
                        help="Seed for reproducibility (default: 1)")
    parser.add_argument('-p', '--pre_release', action='store_true', 
                        help="Discard the list of post-release features before tuning.")
    args = parser.parse_args()

    SEED = args.seed
    
    print(f"Tuning setup")
    print(f"Mode: {'FULL DATASET' if args.full else 'PROTOTYPE (20%)'}")
    print(f"Random Seed: {SEED}")

    # Loading the dataset
    DATASET_PATH = "../dataset/clean_data/clean_dataset.csv"
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: File not found in {DATASET_PATH}")
        return

    if args.pre_release:
        # Features collected after the release of a game or related to the future support
        post_release_feature = [
            'num_achievements', 'metacritic_score', 'review_ratio', 'num_dlc',
            'discount', 'average_forever', 'average_2weeks', 'median_forever', 'median_2weeks'
        ] 
        
        # If pre-release is active, they are dropped
        cols_to_drop = [col for col in post_release_feature if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        print(f"Pre-release mode active. Columns discarded: {cols_to_drop}")

    X = df.drop(columns=['target_owners', 'name'])
    y = df['target_owners'].astype(int)

    if not args.full:
        X_data, _, y_data, _ = train_test_split(X, y, train_size=0.20, stratify=y, random_state=SEED)
    else:
        X_data, y_data = X, y

    # From string to Python list
    list_columns = ['categories', 'genres', 'tags', 'publishers', 'developers', 'languages']
    for col in list_columns:
        if col in X_data:
            X_data[col] = X_data[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Precomputation embeddings
    df_temp = X_data.copy()
    df_temp['target_owners'] = y_data
    df_processed, emb_cols = precompute_detailed_embeddings(df_temp, text_col='detailed_description')
    
    X_processed = df_processed.drop(columns=['target_owners'])
    y_processed = df_processed['target_owners']

    print(f"Dataset size: {len(X_processed)} elements")

    # Complete list of numerical columns
    all_numeric_cols = [
        'price', 'release_year', 'min_ram_gb', 'rec_ram_gb', 'required_age', 
        'num_dlc', 'num_achievements', 'num_languages_supported', 'metacritic_score', 
        'review_ratio', 'discount', 'average_forever', 'average_2weeks', 
        'median_forever', 'median_2weeks'
    ]

    # Complete list of categorical columns
    categorical_cols = ['release_month']

    # Checking the numeric columns actually present
    # (if the pre-release flag is active, some numeric columns have been dropped)
    numeric_cols = [col for col in all_numeric_cols if col in X_processed.columns]

    # Processing numeric features
    base_preprocessor = get_base_preprocessor(numeric_cols, categorical_cols)

    # Dictionary defining model architectures, internal feature selectors, and search spaces.
    # SelectFromModel uses lightweight versions of the estimators to optimize runtime
    models_config = {
        'DecisionTree': {
            'selector_estimator': DecisionTreeClassifier(class_weight='balanced', random_state=SEED),
            'estimator': DecisionTreeClassifier(class_weight='balanced', random_state=SEED),
            'n_iter': 5,
            'tfidf_features': 50,
            'pls_components': 20,
            'param_distributions': {
                'classifier__max_depth': IntDistribution(5, 30),
                'classifier__min_samples_split': IntDistribution(2, 20),
                'classifier__min_samples_leaf': IntDistribution(1, 10),
                'feature_selection__threshold': CategoricalDistribution(['median', 'mean'])
            }
        },
        'RandomForest': {
            'selector_estimator': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=SEED, n_jobs=-1),
            'estimator': RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=-1),
            'n_iter': 40,
            'tfidf_features': 30,
            'pls_components': 20,
            'param_distributions': {
                'classifier__n_estimators': IntDistribution(100, 400),
                'classifier__max_depth': IntDistribution(5, 30),
                'classifier__min_samples_split': IntDistribution(2, 15),
                'classifier__min_samples_leaf': IntDistribution(1, 8),
                'classifier__max_features': FloatDistribution(0.1, 0.6),
                'feature_selection__threshold': CategoricalDistribution(['median', 'mean'])
            }
        },
        'XGBoost': {
            'selector_estimator': WeightedXGBClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, eval_metric='mlogloss', tree_method='hist', device=DEVICE),
            'estimator': WeightedXGBClassifier(random_state=SEED, n_jobs=-1, eval_metric='mlogloss', tree_method='hist', device=DEVICE),
            'n_iter': 50,
            'tfidf_features': 15,
            'pls_components': 20,
            'param_distributions': {
                'classifier__n_estimators': IntDistribution(100, 400),
                'classifier__max_depth': IntDistribution(3, 9),
                'classifier__learning_rate': FloatDistribution(0.05, 0.19),
                'classifier__min_child_weight': IntDistribution(1, 8),
                'classifier__subsample': FloatDistribution(0.6, 1.0),
                'classifier__reg_lambda': FloatDistribution(0.5, 3.0),
                'feature_selection__threshold': CategoricalDistribution(['median', 'mean'])
            }
        }
    }

    # Override dimensionality reduction parameters (using the optimal values found by gridsearch)
    # if pre-release mode is active. All the model shares max_tfidf=15 and pls_components=20 as optimal values
    # in pre-release mode
    if args.pre_release:
        print("The best hyperparameters for pre_release are: max_tfidf=15 and pls_components=20 for every model")
        for model in models_config:
            models_config[model]['tfidf_features'] = 15
            models_config[model]['pls_components'] = 20

    # Nested Cross-Validation (Outer Loop)
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    all_results = {}

    for model_name, config in models_config.items():
        print(f"\n{'='*40}")
        print(f"Inizio valutazione: {model_name}")
        print(f"{'='*40}")

        cache_dir = os.path.join(".", "ml_cache", f"pipeline_cache_{model_name}")
        memory = Memory(location=cache_dir, verbose=0)
        
        # Build the full imblearn pipeline (the same of training_pipeline.py)
        base_pipe = ImbPipeline([
            ('steam_extractor', SteamFeatureExtractor(max_tfidf_features=config['tfidf_features'])),
            ('base_preprocessor', base_preprocessor),
            ('rus', RandomUnderSampler(sampling_strategy=dynamic_undersample, random_state=SEED)),
            ('pls_da', SupervisedPLSDATransformer(emb_cols=emb_cols, n_components=config['pls_components'])),
            ('corr_remover', CorrelationRemover(threshold=0.95)),
            ('smote_nc', DynamicSMOTENC(sampling_strategy=dynamic_oversample, k_neighbors=3, random_state=SEED)),
            ('tomek', TomekLinks()),
            ('sanitizer', FeatureNameSanitizer()),
            ('feature_selection', SelectFromModel(config['selector_estimator'])),
            ('classifier', config['estimator'])
        ], memory=memory)

        model_results = {
            'folds_data': [],
            'best_params_final_fit': None,
            'final_selected_features': None
        }

        # Nested CV Outer Loop
        outer_fold_idx = 1
        for train_ix, test_ix in tqdm(cv_outer.split(X_processed, y_processed), total=cv_outer.n_splits, desc=f"Outer CV ({model_name})"):
            X_train, X_test = X_processed.iloc[train_ix], X_processed.iloc[test_ix]
            y_train, y_test = y_processed.iloc[train_ix], y_processed.iloc[test_ix]

            # Nested CV Inner Loop handled by OptunaSearchCV
            search = OptunaSearchCV(
                estimator=base_pipe,
                param_distributions=config['param_distributions'],
                n_trials=config['n_iter'],
                cv=cv_inner,
                scoring=qwk_scorer,
                random_state=SEED,
                n_jobs=1, 
                verbose=3,
                error_score='raise'
            )

            search.fit(X_train, y_train)
            
            # Extracting the winning pipeline for this fold
            best_pipe_fold = search.best_estimator_
            
            # Extracting the names of the selected features
            # get_feature_names_out() will return a string array with the names of the columns that survived the selector
            selected_features = list(best_pipe_fold.named_steps['feature_selection'].get_feature_names_out())

            y_pred = best_pipe_fold.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro')
            qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic')
            cm = confusion_matrix(y_test, y_pred).tolist()

            model_results['folds_data'].append({
                'fold': outer_fold_idx,
                'best_params_fold': search.best_params_,
                'selected_features_count': len(selected_features),
                'selected_features': selected_features,
                'metrics': {
                    'accuracy': acc,
                    'precision_macro': prec,
                    'recall_macro': rec,
                    'f1_macro': f1,
                    'quadratic_weighted_kappa': qwk
                },
                'confusion_matrix': cm
            })
            outer_fold_idx += 1
            memory.clear(warn=False)
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)

        # Final fit
        print(f"Extracting optimal parameters and final features for {model_name} over the entire set...")
        final_search = OptunaSearchCV(
            estimator=base_pipe, 
            param_distributions=config['param_distributions'], 
            n_trials=config['n_iter'],
            cv=cv_inner, 
            scoring=qwk_scorer, 
            random_state=SEED,
            n_jobs=1, 
            error_score='raise'
        )
        final_search.fit(X_processed, y_processed)
        
        # Saving parameters
        model_results['best_params_final_fit'] = final_search.best_params_
        
        # Saving selected features
        final_selected = list(final_search.best_estimator_.named_steps['feature_selection'].get_feature_names_out())
        model_results['final_selected_features_count'] = len(final_selected)
        model_results['final_selected_features'] = final_selected
        
        all_results[model_name] = model_results

        memory.clear(warn=False)
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

    # Saving JSON
    output_file = f"tuning_results_{'pre_release' if args.pre_release else 'post_release'}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, cls=NumpyEncoder)
    
    print(f"\nTuning complete. Data saved in '{output_file}'")

if __name__ == "__main__":
    main()