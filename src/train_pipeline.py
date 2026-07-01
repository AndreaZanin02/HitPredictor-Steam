import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import os
import ast
import warnings
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.feature_selection import SelectFromModel
from data_preprocessing import (precompute_detailed_embeddings, SteamFeatureExtractor, 
                                CorrelationRemover, dynamic_undersample, FeatureNameSanitizer)
from hyperparameter_tuning import scaler_and_pca, WeightedXGBClassifier

sklearn.set_config(transform_output="pandas")
warnings.filterwarnings('ignore')

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Final training and XAI")
    parser.add_argument('-p', '--pre_release', action='store_true', 
                        help="Discard post-release features before training.")
    parser.add_argument('-s', '--seed', type=int, default=1, 
                        help="Seed for reproducibility (default: 1)")
    args = parser.parse_args()
    
    # Dynamic seed assignment
    SEED = args.seed
    print(f"Random Seed set to: {SEED}")

    # Dynamic path
    if args.pre_release:
        base_dir = "../results/pre_release_model"
        print("Mode: pre_release active. Post-release features will be ignored")
    else:
        base_dir = "../results/post_release_model"
        print("Mode: post-release active. All features will be used")
        
    xai_dir = os.path.join(base_dir, "xai_plots")
    os.makedirs(xai_dir, exist_ok=True)
    print(f"XAI plots will be saved in: {xai_dir}")

    # Loading and pre-processing of the dataset
    DATASET_PATH = "../dataset/clean_data/clean_dataset.csv"
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: File not found in {DATASET_PATH}")
        return

    # If --pre_release is active, post-release features are dropped
    if args.pre_release:
        post_release_feature = [
            'days_since_release', 'num_achievements', 'metacritic_score', 'review_ratio',
            'discount', 'average_forever', 'average_2weeks', 'median_forever', 'median_2weeks'
        ] 
        cols_to_drop = [col for col in post_release_feature if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        print(f"Post-release columns dropped: {len(cols_to_drop)} feature ignored")

    X = df.drop(columns=['target_owners', 'name'])
    y = df['target_owners'].astype(int)

    # Cast to list
    list_columns = ['categories', 'genres', 'tags', 'publishers', 'developers', 'languages']
    for col in list_columns:
        if col in X.columns:
            X[col] = X[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Split: A separate test set is needed to compute SHAP and LIME on unseen data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)

    # Embeddings 
    df_temp_train = X_train.copy()
    df_temp_train['target_owners'] = y_train
    df_train_proc, emb_cols = precompute_detailed_embeddings(df_temp_train, text_col='detailed_description')
    
    df_temp_test = X_test.copy()
    df_temp_test['target_owners'] = y_test
    df_test_proc, _ = precompute_detailed_embeddings(df_temp_test, text_col='detailed_description')

    X_train_proc = df_train_proc.drop(columns=['target_owners'])
    y_train_proc = df_train_proc['target_owners']
    X_test_proc = df_test_proc.drop(columns=['target_owners'])
    y_test_proc = df_test_proc['target_owners']

    # Best model configuration
    xgb_estimator = WeightedXGBClassifier(
        learning_rate=0.1,
        max_depth=6,
        n_estimators=200,
        random_state=SEED,
        n_jobs=-1,
        eval_metric='mlogloss',
        tree_method='hist',
        device='cuda'
    )

    xgb_selector = WeightedXGBClassifier(
        n_estimators=50, random_state=SEED, n_jobs=-1, eval_metric='mlogloss', tree_method='hist', device='cuda'
    )

    # Training pipeline
    final_pipe = ImbPipeline([
        ('steam_extractor', SteamFeatureExtractor()),
        ('scaler_and_pca', scaler_and_pca),
        ('corr_remover', CorrelationRemover(threshold=0.95)),
        ('rus', RandomUnderSampler(sampling_strategy=dynamic_undersample, random_state=SEED)),
        ('tomek', TomekLinks()),
        ('sanitizer', FeatureNameSanitizer()),
        ('feature_selection', SelectFromModel(xgb_selector, threshold='mean')),
        ('classifier', xgb_estimator)
    ])

    # Training
    print("\nFinal model training in progress...")
    final_pipe.fit(X_train_proc, y_train_proc)
    
    # Evaluation
    y_pred = final_pipe.predict(X_test_proc)
    print("\nClassification report on test set:")
    print(classification_report(y_test_proc, y_pred))

    # ------------------------ EXPLAINABLE AI SECTION ----------------------------------
    print("\nData preparation for eXplainable AI (XAI)...")
    
    preprocessing_pipe = final_pipe[:-1]
    X_test_transformed = preprocessing_pipe.transform(X_test_proc)
    
    model = final_pipe.named_steps['classifier']
    feature_names = final_pipe.named_steps['feature_selection'].get_feature_names_out()
    
    X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names)

    # XGBOOST NATIVE FEATURE IMPORTANCE
    print("Generating XGBoost Native Importance...")
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model, max_num_features=20, importance_type='gain', title='XGBoost Feature Importance (Gain)')
    plt.tight_layout()
    plt.savefig(os.path.join(xai_dir, 'xgb_native_importance.png'))
    plt.close()

    # SHAP
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)

    # Due to the unbalanced dataset, train_test_split is used to do stratified sampling
    max_samples = min(2000, len(X_test_transformed_df))
    if max_samples < len(X_test_transformed_df):
        _, X_test_sample, _, _ = train_test_split(
            X_test_transformed_df, 
            y_test_proc, 
            test_size=max_samples, 
            stratify=y_test_proc, 
            random_state=SEED
        )
    else:
        X_test_sample = X_test_transformed_df
    
    shap_values = explainer(X_test_sample)

    print("Generating SHAP summary plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(xai_dir, 'shap_summary_plot.png'))
    plt.close()

    print("Generating SHAP waterfall plot of a istance...")
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(xai_dir, 'shap_waterfall.png'))
    plt.close()

    # LIME
    print("Executing LIME...")
    
    # Reconstruction of the undersampled dataset seen by the model
    # Initial trasformations
    X_tr_step = final_pipe.named_steps['steam_extractor'].transform(X_train_proc)
    X_tr_step = final_pipe.named_steps['scaler_and_pca'].transform(X_tr_step)
    X_tr_step = final_pipe.named_steps['corr_remover'].transform(X_tr_step)
    
    # Resampling using the same seed
    X_resampled, y_resampled = final_pipe.named_steps['rus'].fit_resample(X_tr_step, y_train_proc)
    X_resampled, y_resampled = final_pipe.named_steps['tomek'].fit_resample(X_resampled, y_resampled)
    
    # Trasformations post undersampling
    X_resampled = final_pipe.named_steps['sanitizer'].transform(X_resampled)
    X_train_transformed = final_pipe.named_steps['feature_selection'].transform(X_resampled)
    
    # Extraction of the final numerical matrix
    lime_bg_data = X_train_transformed.values if isinstance(X_train_transformed, pd.DataFrame) else X_train_transformed

    # Explainer initialization with the right dataset (the same seen by the model)
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=lime_bg_data,
        feature_names=feature_names,
        class_names=['Not Owner', 'Owner'],
        mode='classification',
        random_state=SEED
    )

    instance_idx = 0
    instance_to_explain = X_test_transformed_df.iloc[instance_idx].values

    predict_fn = lambda x: model.predict_proba(x)

    exp = lime_explainer.explain_instance(
        data_row=instance_to_explain, 
        predict_fn=predict_fn,
        num_features=10 
    )
    
    exp.save_to_file(os.path.join(xai_dir, 'lime_explanation.html'))
    
    print(f"\nXAI completed. Check the results in '{xai_dir}'")

    # --------------------- FINAL TRAINING USING 100% OF THE DATASET -------------------------
    print("\nPreparation of the complete dataset (100%) for the production model...")
    
    # Recalculation the embeddings on the entire dataset to avoid losing 20% ​​of the textual information
    df_full = X.copy()
    df_full['target_owners'] = y
    df_full_proc, _ = precompute_detailed_embeddings(df_full, text_col='detailed_description')
    
    X_full_proc = df_full_proc.drop(columns=['target_owners'])
    y_full_proc = df_full_proc['target_owners']

    print("Training the final model on the entire dataset...")
    final_pipe.fit(X_full_proc, y_full_proc)

    # Saving the model
    import joblib
    model_filename = 'pre_release_model.pkl' if args.pre_release else 'post_release_model.pkl'
    model_path = os.path.join(base_dir, model_filename)
    
    joblib.dump(final_pipe, model_path)
    print(f"Final model trained and saved successfully in: {model_path}")


if __name__ == "__main__":
    main()