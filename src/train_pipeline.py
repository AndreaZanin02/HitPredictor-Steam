"""
This script performs the end-to-end final training of the Steam ownership-tier
classifier, starting from the cleaned dataset produced by data_cleaning.py.

Because the target ('target_owners') is heavily imbalanced (Tier 0 dominates
with ~3769 samples in the test set vs only ~56 for Tier 4), the pipeline does
NOT rely on a single fit/predict cycle.

Instead it:
1. Trains the full imbalance-aware pipeline (undersampling + SMOTENC + Tomek links)
   on a held-out 80% train split.
2. Calibrates per-class probability weights via Differential Evolution on
   out-of-fold predictions, to correct the residual bias toward majority tiers
   that survives even after resampling.
3. Evaluates the calibrated model on the untouched 20% test split.
4. Produces SHAP and LIME explanations to make the model's tier decisions
   interpretable for human review.
5. Refits the final production pipeline on 100% of the data for deployment.

Running this script twice with --pre_release toggled trains two separate
production artifacts (pre-release vs post-release feature sets)
"""
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
import torch
import sklearn
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.feature_selection import SelectFromModel
from data_preprocessing import (
    precompute_detailed_embeddings, SteamFeatureExtractor, CorrelationRemover, 
    dynamic_undersample, FeatureNameSanitizer, DynamicSMOTENC, dynamic_oversample,
    SupervisedPLSDATransformer, get_base_preprocessor
)
from hyperparameter_tuning import WeightedXGBClassifier
from scipy.optimize import differential_evolution

sklearn.set_config(transform_output="pandas")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
warnings.filterwarnings('ignore')


def loss_function(weights, y_proba_oof, y_train_proc):
    """
    Objective function minimized by Differential Evolution during weight calibration
    """
    weighted_proba = y_proba_oof * weights
    pred = np.argmax(weighted_proba, axis=1)
    return -cohen_kappa_score(y_train_proc, pred, weights='quadratic')


def main():
    # Argument parsing
    # --pre_release lets us train two distinct models: one restricted to
    # features known before a game launches (for pre-launch forecasting),
    # and one using the full feature set including post-launch signals
    # (achievements unlocked, review ratio, discount history, playtime...)
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

    # Post-release features are dropped in --pre_release mode because they are
    # only observable after a game has already been on the market for a
    # while (achievements unlocked, accumulated reviews, discounts applied,
    # playtime stats). Keeping them for a "pre-release" model would leak
    # future information that is unavailable at prediction time in a real
    # pre-launch forecasting scenario
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

    # Split: a held-out test set is required so that SHAP/LIME
    # explanations are computed on data the model never saw during fit
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)

    # Sentence embeddings for 'detailed_description' are computed here, once,
    # for train and test SEPARATELY (rather than inside the sklearn pipeline)
    # purely for performance: the sentence-transformer model is expensive to
    # run and is frozen (not fine-tuned on our data), so calling it once per
    # split instead of once per CV fold saves a huge amount of redundant GPU/CPU
    # work without introducing any data leakage.
    # The embedding model never learns anything from our target labels
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
    
    # get_base_preprocessor builds a ColumnTransformer that imputes+scales numeric_cols
    # (RobustScaler is used instead of StandardScaler specifically because financial/temporal features like
    # 'price' and playtime have heavy outliers, which RobustScaler is
    # resistant to) and imputes+one-hot-encodes categorical_cols.
    # All other columns pass through untouched
    base_preprocessor = get_base_preprocessor(numeric_cols, categorical_cols)

    # Supervised PLS-DA compresses the 768-dimensional sentence embeddings
    # down to 20 components that are maximally discriminative with respect to
    # the target tiers, which both reduces dimensionality (helping the
    # downstream SMOTENC/tree-based steps) and injects target-aware signal
    # into an otherwise unsupervised text representation
    plsda_step = SupervisedPLSDATransformer(emb_cols=emb_cols, n_components=20)

    # Best model configuration
    if args.pre_release:
        print("Using pre-release hyperparameter configuration")
        xgb_estimator = WeightedXGBClassifier(
            n_estimators=397,
            max_depth=5,
            learning_rate=0.1074709860640873,
            min_child_weight=5,
            subsample=0.9652105402477317,
            reg_lambda=0.8250421259522612,
            random_state=SEED,
            n_jobs=-1,
            eval_metric='mlogloss',
            tree_method='hist',
            device=DEVICE
        )
    else:
        print("Using post-release hyperparameter configuration")
        xgb_estimator = WeightedXGBClassifier(
            n_estimators=236,
            max_depth=7,
            learning_rate=0.14685290784404992,
            min_child_weight=2,
            subsample=0.9675284682571256,
            reg_lambda=0.7907934154092928,
            random_state=SEED,
            n_jobs=-1,
            eval_metric='mlogloss',
            tree_method='hist',
            device=DEVICE
        )

    # selector_estimator
    xgb_selector = WeightedXGBClassifier(
        n_estimators=100, random_state=SEED, n_jobs=-1, eval_metric='mlogloss', tree_method='hist', device=DEVICE
    )

    FEATURE_SELECTION_THRESHOLD = 'median'

    # Training pipeline
    # Using imblearn's Pipeline (not sklearn's) is very important here because it
    # supports resampling steps (rus, smote_nc, tomek) that change the number
    # of rows; something a plain sklearn Pipeline cannot do

    #  1. steam_extractor   : turns categories/genres/tags/languages/publishers/
    #                         developers/short_description into numeric
    #                         multi-hot + TF-IDF columns
    #  2. base_preprocessor : imputes + scales numeric columns, one-hot encodes
    #                         'release_month'; embeddings pass through
    #  3. rus (RandomUnderSampler): shrinks the dominant Tier 0 class first
    #                         (via dynamic_undersample, to 25% of its current
    #                         fold size) so the following SMOTE-based
    #                         oversampling doesn't have to synthesize an
    #                         unreasonable number of minority samples just to
    #                         approach balance, and so overall pipeline
    #                         runtime stays manageable
    #  4. pls_da            : compresses text embeddings into 20 components
    #                         that are supervised by the (already-undersampled)
    #                         training labels
    #  5. corr_remover      : drops near-duplicate/redundant numeric features
    #                         (|corr| > 0.95) that add noise to the resamplers
    #                         and to the classifier without new information
    #  6. smote_nc (DynamicSMOTENC): synthetically oversamples Tier 4 (and
    #                         Tier 3 when relevant) via dynamic_oversample,
    #                         using k_neighbors=3 because of how few minority
    #                         samples remain in a single CV fold. Mixed
    #                         categorical/continuous columns are auto-detected
    #                         at fit-time (nunique() <= 2 => treated as
    #                         categorical), since the exact column layout
    #                         changes every fold
    #  7. tomek (TomekLinks): removes borderline/overlapping majority-class
    #                         points sitting right next to minority points
    #                         created by SMOTE, sharpening the decision
    #                         boundary between adjacent tiers
    #  8. sanitizer         : strips characters ('[', ']', '<') that XGBoost/
    #                         LightGBM reject from column names, which can
    #                         appear after one-hot encoding or multi-label
    #                         binarization
    #  9. feature_selection : keeps only the features whose importance (per
    #                         xgb_selector) is above the chosen threshold,
    #                         reducing dimensionality/noise before the final
    #                         classifier
    # 10. classifier        : the tuned WeightedXGBClassifier, which additionally
    #                         applies balanced sample_weight at fit() time as a
    #                         second, complementary layer of imbalance
    #                         correction on top of the resampling above
    final_pipe = ImbPipeline([
        ('steam_extractor', SteamFeatureExtractor(max_tfidf_features=15)),
        ('base_preprocessor', base_preprocessor),
        ('rus', RandomUnderSampler(sampling_strategy=dynamic_undersample, random_state=SEED)),
        ('pls_da', plsda_step),
        ('corr_remover', CorrelationRemover(threshold=0.95)),
        ('smote_nc', DynamicSMOTENC(sampling_strategy=dynamic_oversample, k_neighbors=3, random_state=SEED)),
        ('tomek', TomekLinks()),
        ('sanitizer', FeatureNameSanitizer()),
        ('feature_selection', SelectFromModel(xgb_selector, threshold=FEATURE_SELECTION_THRESHOLD)),
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
        cv=cv_calibration, method='predict_proba', n_jobs=1, verbose=3
    )

    bounds = [(0.1, 2.0)] * 5

    print("\nStarting Differential Evolution to maximize Quadratic Weighted Kappa...")
    # Differential Evolution is used instead of a gradient-based optimizer
    # because loss_function is non-differentiable (it involves an argmax and
    # a discrete metric, cohen_kappa_score). DE is a population-based,
    # gradient-free global optimizer well suited to this kind of small
    # (5-dimensional), noisy, non-convex objective
    result = differential_evolution(
        loss_function, 
        bounds, 
        args=(y_proba_oof, y_train_proc),
        strategy='best1bin',  # standard, robust mutation/crossover strategy
        maxiter=100,          # maximum number of generations
        popsize=15,           # population size (15 * 5 variables = 75 candidate vectors evaluated per generation)
        tol=1e-3,             # relative tolerance for early convergence stop
        seed=SEED,            # ensures reproducibility of the search
        workers=-1            # parallelizes the fitness evaluations across all CPU cores
    )

    best_weights = result.x

    # Normalizing so the weights sum to 1 doesn't change the resulting argmax
    # (it's a uniform rescaling across all classes), but keeps the values
    # interpretable as relative "importance" multipliers rather than an
    # arbitrary unbounded scale
    best_weights /= np.sum(best_weights) 
    print(f"-> Best weights found: {np.round(best_weights, 3)}")

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

    def _unscale_numeric_features(df_scaled, pipeline, numeric_cols):
        """
        Convert selected numeric features back to their original, human-readable
        scale by applying the inverse_transform of the RobustScaler fitted inside
        the pipeline's base_preprocessor step
        """
        df_unscaled = df_scaled.copy()
        base_prep = pipeline.named_steps['base_preprocessor']
        scaler = base_prep.named_transformers_['num_pipeline'].named_steps['scaler']
        
        num_buffer = pd.DataFrame(0.0, index=df_scaled.index, columns=numeric_cols)
        for col in numeric_cols:
            if col in df_scaled.columns:
                num_buffer[col] = df_scaled[col]

        unscaled_array = scaler.inverse_transform(num_buffer)
        df_inv = pd.DataFrame(unscaled_array, columns=numeric_cols, index=df_scaled.index)
        

        for col in numeric_cols:
            if col in df_scaled.columns:
                df_unscaled[col] = df_inv[col]
        
        return df_unscaled

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

    # Unscale of the numerical columns before generate the plots
    X_test_sample_unscaled = _unscale_numeric_features(X_test_sample, final_pipe, numeric_cols)
    shap_values.data = X_test_sample_unscaled.values

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
        
        # Beeswarm plot
        plt.figure(figsize=(12, 8))
        shap.plots.beeswarm(shap_values[:, :, class_idx], show=False)
        plt.title(f"Importance of the Feature for - {tier_labels[class_idx]}", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(xai_dir, f'shap_beeswarm_class_{class_idx}.png'), dpi=150)
        plt.close()

        # Waterfall plot
        plt.figure(figsize=(12, 6))
        shap.plots.waterfall(shap_values[0, :, class_idx], show=False)
        plt.title(f"Explaining instance (Idx 0) - {tier_labels[class_idx]}", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(xai_dir, f'shap_waterfall_class_{class_idx}.png'), dpi=150)
        plt.close()
        
    print(f"All the SHAP plots are saved in: {xai_dir}")


    # LIME
    print("Executing LIME...")
    
    # LIME needs a "background" dataset representing the distribution the
    # model was actually trained on, to generate its local perturbations
    # around each explained instance.
    #To get the true post-resampling training distribution, we
    # manually replay each pipeline step in order, calling fit_resample() on
    # the resamplers exactly as final_pipe.fit() would have done internally
    X_tr_step = final_pipe.named_steps['steam_extractor'].transform(X_train_proc)
    X_tr_step = final_pipe.named_steps['base_preprocessor'].transform(X_tr_step)
    X_resampled, y_resampled = final_pipe.named_steps['rus'].fit_resample(X_tr_step, y_train_proc)
    X_resampled = final_pipe.named_steps['pls_da'].transform(X_resampled)
    X_resampled = final_pipe.named_steps['corr_remover'].transform(X_resampled)
    X_resampled, y_resampled = final_pipe.named_steps['smote_nc'].fit_resample(X_resampled, y_resampled)
    X_resampled, y_resampled = final_pipe.named_steps['tomek'].fit_resample(X_resampled, y_resampled)
    X_resampled = final_pipe.named_steps['sanitizer'].transform(X_resampled)
    X_train_transformed = final_pipe.named_steps['feature_selection'].transform(X_resampled)
    
    X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=feature_names)

    # LIME's perturbation/statistics are far more meaningful to a human reader in real-world units
    X_train_unscaled = _unscale_numeric_features(X_train_transformed_df, final_pipe, numeric_cols)
    lime_bg_data = X_train_unscaled.values
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=lime_bg_data,
        feature_names=feature_names,
        class_names=['Tier 0 (<20k)', 'Tier 1 (20-100k)', 'Tier 2 (100-500k)', 'Tier 3 (500k-2M)', 'Tier 4 (>2M)'],
        mode='classification',
        random_state=SEED
    )

    # Custom prediction function required because LIME generates its
    # perturbed neighborhood samples in the same (unscaled) space as
    # lime_bg_data, but the actual classifier expects scaled numeric inputs
    def predict_fn_lime(x):
        df_perturbed = pd.DataFrame(x, columns=feature_names)
        base_prep = final_pipe.named_steps['base_preprocessor']
        scaler = base_prep.named_transformers_['num_pipeline'].named_steps['scaler']
        
        num_buffer = pd.DataFrame(0.0, index=df_perturbed.index, columns=numeric_cols)
        for col in numeric_cols:
            if col in df_perturbed.columns:
                num_buffer[col] = df_perturbed[col]
                
        # Re-scale the perturbed samples generated by LIME using the scaler
        # fitted during the original pipeline training
        scaled_array = scaler.transform(num_buffer)
        df_scaled_inv = pd.DataFrame(scaled_array, columns=numeric_cols, index=df_perturbed.index)
        
        df_perturbed_scaled = df_perturbed.copy()
        for col in numeric_cols:
            if col in df_perturbed.columns:
                df_perturbed_scaled[col] = df_scaled_inv[col]
                
        return model.predict_proba(df_perturbed_scaled)

    print("\nGenerating LIME plots for each tier...")
    for class_idx in range(5):
        candidate_positions = np.where(y_pred_adjusted == class_idx)[0]

        if len(candidate_positions) == 0:
            print(f"--> WARNING: the model never predicts {tier_labels[class_idx]} correctly")
            continue

        instance_pos = candidate_positions[0]
        
        # Convert the single instance to explain back into real-world units
        instance_scaled_df = X_test_transformed_df.iloc[[instance_pos]]
        instance_unscaled_df = _unscale_numeric_features(instance_scaled_df, final_pipe, numeric_cols)
        instance_to_explain = instance_unscaled_df.iloc[0].values

        exp = lime_explainer.explain_instance(
            data_row=instance_to_explain,
            predict_fn=predict_fn_lime,
            num_features=10,
            labels=[class_idx]
        )

        fig = exp.as_pyplot_figure(label=class_idx)
        fig.tight_layout()
        fig.savefig(os.path.join(xai_dir, f'lime_explanation_class_{class_idx}.png'), bbox_inches='tight', dpi=300)
        plt.close(fig)

    print(f"All the LIME plots are saved in: {xai_dir}")

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