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
import joblib
from scipy.optimize import minimize
import torch
import sklearn
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.feature_selection import SelectFromModel
from data_preprocessing import (
    precompute_detailed_embeddings, SteamFeatureExtractor, CorrelationRemover, 
    dynamic_undersample, FeatureNameSanitizer, DynamicSMOTENC, dynamic_oversample
)
from hyperparameter_tuning import get_scaler_and_pca, WeightedXGBClassifier

sklearn.set_config(transform_output="pandas")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
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
            'num_achievements', 'metacritic_score', 'review_ratio', 'num_dlc',
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

    # Numerical columns
    all_numeric_cols = [
        'price', 'release_year', 'min_ram_gb', 'rec_ram_gb', 'required_age', 
        'num_dlc', 'num_achievements', 'num_languages_supported', 'metacritic_score', 
        'review_ratio', 'discount', 'average_forever', 'average_2weeks', 
        'median_forever', 'median_2weeks'
    ]
    numeric_cols = [col for col in all_numeric_cols if col in X_train_proc.columns]

    # Complete list of categorical columns
    categorical_cols = ['release_month']

    # PCA calculation
    scaler_and_pca = get_scaler_and_pca(numeric_cols, categorical_cols, emb_cols, SEED)


    # Best model configuration
    xgb_estimator = WeightedXGBClassifier(
        learning_rate=0.1,
        max_depth=6,
        n_estimators=200,
        random_state=SEED,
        n_jobs=-1,
        eval_metric='mlogloss',
        tree_method='hist',
        device=DEVICE
    )

    xgb_selector = WeightedXGBClassifier(
        n_estimators=50, random_state=SEED, n_jobs=-1, eval_metric='mlogloss', tree_method='hist', device=DEVICE
    )

    # Training pipeline
    final_pipe = ImbPipeline([
        ('steam_extractor', SteamFeatureExtractor(max_tfidf_features=30)),
        ('scaler_and_pca', scaler_and_pca),
        ('corr_remover', CorrelationRemover(threshold=0.95)),
        ('rus', RandomUnderSampler(sampling_strategy=dynamic_undersample, random_state=SEED)),
        ('smote_nc', DynamicSMOTENC(sampling_strategy=dynamic_oversample, k_neighbors=3, random_state=SEED)),
        ('tomek', TomekLinks()),
        ('sanitizer', FeatureNameSanitizer()),
        ('feature_selection', SelectFromModel(xgb_selector, threshold='mean')),
        ('classifier', xgb_estimator)
    ])

    # Training
    print("\nFinal model training in progress...")
    final_pipe.fit(X_train_proc, y_train_proc)
    
    # Weights calibration
    print("\nWeights calibration (out-of-fold on train)...")

    cv_calibration = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Each row of X_train_proc receives a predict_proba from a model
    # that, for that row, was in the validation fold (it didn't see it in fit --> no leakage)
    y_proba_oof = cross_val_predict(
        final_pipe, X_train_proc, y_train_proc,
        cv=cv_calibration, method='predict_proba', n_jobs=1
    )

    def loss_function(weights):
        weighted_proba = y_proba_oof * weights
        pred = np.argmax(weighted_proba, axis=1)
        return -f1_score(y_train_proc, pred, average='macro')

    initial_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    bounds = [(0.1, 2.0)] * 5

    result = minimize(loss_function, initial_weights, bounds=bounds,
                    method='Powell', options={'xtol': 1e-3, 'ftol': 1e-3})
    best_weights = result.x
    print(f"-> Best weights: {np.round(best_weights, 3)}")

    # Final evaluation using the best weights
    y_proba_test = final_pipe.predict_proba(X_test_proc)
    final_proba = y_proba_test * best_weights
    y_pred_adjusted = np.argmax(final_proba, axis=1)

    print("\nClassification report:")
    print(classification_report(y_test_proc, y_pred_adjusted))

    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(y_test_proc, y_pred_adjusted)
    print(cm)

    # Plotting confusion matrix
    tier_labels = ["Tier 0", "Tier 1", "Tier 2", "Tier 3", "Tier 4"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=tier_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax)

    plt.title("Confusion Matrix (calibrated thresholds)", fontsize=14, pad=15)
    plt.tight_layout()

    # Saving the plot
    plt.savefig(os.path.join(base_dir, 'calibrated_confusion_matrix.png'), dpi=150)
    print("-> Confusion Matrix saved as 'calibrated_confusion_matrix.png'") 
    plt.close()

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

    # Real names of the tiers
    tier_labels = [
        "Tier 0 (<20k)", 
        "Tier 1 (20k-100k)", 
        "Tier 2 (100k-500k)", 
        "Tier 3 (500k-2M)", 
        "Tier 4 (>2M)"
    ]

    print("\nGenerating SHAP plots for each tier...")
    for class_idx in range(5):
        print(f"--> Elaborating {tier_labels[class_idx]}...")
        
        # Beeswarm plot of each class
        plt.figure(figsize=(12, 8))
        # Slicing: selecting games of the current class
        shap.plots.beeswarm(shap_values[:, :, class_idx], show=False)
        plt.title(f"Importance of the Feature for - {tier_labels[class_idx]}", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(xai_dir, f'shap_beeswarm_class_{class_idx}.png'), dpi=150)
        plt.close()

        # Waterfall plot for the first instance of each class
        plt.figure(figsize=(12, 6))
        # Slicing: istance 0, of the current tier
        shap.plots.waterfall(shap_values[0, :, class_idx], show=False)
        plt.title(f"Explaining istance (Idx 0) - {tier_labels[class_idx]}", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(xai_dir, f'shap_waterfall_class_{class_idx}.png'), dpi=150)
        plt.close()
        
    print(f"All the SHAP plots are saved in: {xai_dir}")


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
        class_names=['Tier 0 (<20k)', 'Tier 1 (20-100k)', 'Tier 2 (100-500k)', 'Tier 3 (500k-2M)', 'Tier 4 (>2M)'],
        mode='classification',
        random_state=SEED
    )

    instance_idx = 0
    instance_to_explain = X_test_transformed_df.iloc[instance_idx].values

    predict_fn = lambda x: model.predict_proba(pd.DataFrame(x, columns=feature_names))
    exp = lime_explainer.explain_instance(
        data_row=instance_to_explain, 
        predict_fn=predict_fn,
        num_features=10 
    )
    
    # Saving LIME plot
    fig = exp.as_pyplot_figure()
    fig.tight_layout() 
    fig.savefig(os.path.join(xai_dir, 'lime_explanation.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    
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

    # Dictionary with models and weights
    production_artifact = {
    'pipeline': final_pipe,
    'best_weights': best_weights
    }

    model_filename = 'pre_release_model.pkl' if args.pre_release else 'post_release_model.pkl'
    model_path = os.path.join(base_dir, model_filename)

    # Saving model and weights
    joblib.dump(production_artifact, model_path)


if __name__ == "__main__":
    main()