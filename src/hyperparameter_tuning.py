import argparse
import pandas as pd
import numpy as np
import sklearn
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from tqdm.auto import tqdm
import ast
from data_preprocessing import precompute_detailed_embeddings, SteamFeatureExtractor, CorrelationRemover, dynamic_undersample

sklearn.set_config(transform_output="pandas")

class WeightedXGBClassifier(XGBClassifier):
    """
    A wrapper that automatically calculates sample_weights
    during fit() based on the incoming data at that time
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
        # Features collected after the release of a game
        post_release_feature = [
            'days_since_release', 'num_achievements', 'metacritic_score', 'review_ratio',
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
        if col in df.columns:
            X_data[col] = X_data[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Precomputation Embeddings
    df_temp = X_data.copy()
    df_temp['target_owners'] = y_data
    df_processed, emb_cols = precompute_detailed_embeddings(df_temp, text_col='detailed_description')
    
    X_processed = df_processed.drop(columns=['target_owners'])
    y_processed = df_processed['target_owners']

    print(f"Dataset size: {len(X_processed)} elements")

    # Complete list of numerical columns
    all_numeric_cols = [
        'price', 'days_since_release', 'min_ram_gb', 'rec_ram_gb', 'required_age', 
        'num_dlc', 'num_achievements', 'num_languages_supported', 'metacritic_score', 
        'review_ratio', 'discount', 'average_forever', 'average_2weeks', 
        'median_forever', 'median_2weeks'
    ]

    # Checking the numeric columns actually present
    # (if the pre-release flag is active, some numeric columns have been dropped)
    numeric_cols = [col for col in all_numeric_cols if col in X_processed.columns]

    # Processing numeric features
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', RobustScaler())
    ])
    scaler_and_pca = ColumnTransformer(
        transformers=[
            ('num_pipeline', numeric_transformer, numeric_cols),
            ('pca_pipeline', PCA(n_components=50, random_state=SEED), emb_cols)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    # Model Configuration: each model defines their own estimator and feature selection
    # For feature selectors, lightweight versions of the same models are used to save time
    models_config = {
        'DecisionTree': {
            'selector_estimator': DecisionTreeClassifier(class_weight='balanced', random_state=SEED),
            'estimator': DecisionTreeClassifier(class_weight='balanced', random_state=SEED),
            'param_grid': {
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 10],
                'feature_selection__threshold': ['median', 'mean']
            }
        },
        'RandomForest': {
            'selector_estimator': RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=SEED, n_jobs=4),
            'estimator': RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=4),
            'param_grid': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 20],
                'classifier__min_samples_split': [2, 5],
                'feature_selection__threshold': ['median', '0.75*median', 'mean']
            }
        },
        'XGBoost': {
            'selector_estimator': WeightedXGBClassifier(n_estimators=50, random_state=SEED, n_jobs=4, eval_metric='logloss', tree_method='hist'),
            'estimator': WeightedXGBClassifier(random_state=SEED, n_jobs=4, eval_metric='logloss', tree_method='hist'),
            'param_grid': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 6],
                'classifier__learning_rate': [0.01, 0.1],
                'feature_selection__threshold': ['median', 'mean']
            }
        }
    }

    # Nested Cross-Validation (Outer Loop)
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    all_results = {}

    for model_name, config in models_config.items():
        print(f"\n{'='*40}")
        print(f"Inizio valutazione: {model_name}")
        print(f"{'='*40}")
        
        base_pipe = ImbPipeline([
            ('steam_extractor', SteamFeatureExtractor()),
            ('scaler_and_pca', scaler_and_pca),
            ('corr_remover', CorrelationRemover(threshold=0.95)),

            # Dynamic undersampling of class 0
            ('rus', RandomUnderSampler(sampling_strategy=dynamic_undersample, random_state=SEED)),
            # Cleaning bounds between classes
            ('tomek', TomekLinks()), 

            ('feature_selection', SelectFromModel(config['selector_estimator'])),
            ('classifier', config['estimator'])
        ])

        model_results = {
            'folds_data': [],
            'best_params_final_fit': None,
            'final_selected_features': None
        }

        outer_fold_idx = 1
        for train_ix, test_ix in tqdm(cv_outer.split(X_processed, y_processed), total=cv_outer.n_splits, desc=f"Outer CV ({model_name})"):
            X_train, X_test = X_processed.iloc[train_ix], X_processed.iloc[test_ix]
            y_train, y_test = y_processed.iloc[train_ix], y_processed.iloc[test_ix]

            grid_search = GridSearchCV(
                estimator=base_pipe, 
                param_grid=config['param_grid'], 
                cv=cv_inner, 
                scoring='f1_macro', 
                n_jobs=1,
                verbose=3,
                error_score='raise'
            )
            
            grid_search.fit(X_train, y_train)
            
            # Extracting the winning pipeline for this fold
            best_pipe_fold = grid_search.best_estimator_
            
            # Extracting the names of the selected features
            # get_feature_names_out() will return a string array with the names of the columns that survived the selector
            selected_features = list(best_pipe_fold.named_steps['feature_selection'].get_feature_names_out())

            y_pred = best_pipe_fold.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro')
            cm = confusion_matrix(y_test, y_pred).tolist()

            model_results['folds_data'].append({
                'fold': outer_fold_idx,
                'best_params_fold': grid_search.best_params_,
                'selected_features_count': len(selected_features),
                'selected_features': selected_features,
                'metrics': {
                    'accuracy': acc,
                    'precision_macro': prec,
                    'recall_macro': rec,
                    'f1_macro': f1
                },
                'confusion_matrix': cm
            })
            outer_fold_idx += 1

        # Final fit
        print(f"Extracting optimal parameters and final features for {model_name} over the entire set...")
        final_search = GridSearchCV(base_pipe, config['param_grid'], cv=cv_inner, scoring='f1_macro', n_jobs=1)
        final_search.fit(X_processed, y_processed)
        
        # Saving parameters
        model_results['best_params_final_fit'] = final_search.best_params_
        
        # Saving selected features
        final_selected = list(final_search.best_estimator_.named_steps['feature_selection'].get_feature_names_out())
        model_results['final_selected_features_count'] = len(final_selected)
        model_results['final_selected_features'] = final_selected
        
        all_results[model_name] = model_results

    # Saving JSON
    output_file = "tuning_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nTuning complete. Data saved in '{output_file}'")

if __name__ == "__main__":
    main()