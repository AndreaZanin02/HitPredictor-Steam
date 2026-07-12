import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon
import warnings

# Silences seaborn/matplotlib UserWarnings that
# would otherwise clutter the console output during plotting
warnings.filterwarnings('ignore', category=UserWarning)

def load_data(json_path):
    """
    Load the nested-CV tuning results produced by hyperparameter_tuning.py
    (e.g. tuning_results_post_release.json / tuning_results_pre_release.json)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_metrics_to_dataframe(results):
    """
    Flatten the metrics stored in the JSON results into a long-format DataFrame
    (one row per model-fold combination), ready for plotting and statistical testing
    """
    records = []
    for model_name, model_data in results.items():
        for fold_data in model_data.get('folds_data', []):
            metrics = fold_data.get('metrics', {})
            records.append({
                'Model': model_name,
                'Fold': fold_data.get('fold'),
                'Accuracy': metrics.get('accuracy'),
                'Precision (Macro)': metrics.get('precision_macro'),
                'Recall (Macro)': metrics.get('recall_macro'),
                'F1 (Macro)': metrics.get('f1_macro'),
                # QWK: it's an important metric, it penalizes predictions that
                # land far from the true tier more than adjacent-tier errors
                'QWK': metrics.get('quadratic_weighted_kappa')
            })
    return pd.DataFrame(records)

def plot_metrics(df):
    """
    Draw one boxplot per metric, comparing all models side by side
    """
    metrics = ['QWK', 'Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)']

    # 2x3 grid: 5 metrics used, the 6th (unused) axis is hidden below.
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Distribution of Metrics (Nested CV outer folds)', fontsize=16)

    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        sns.boxplot(data=df, x='Model', y=metric, ax=axes[i], hue='Model', palette='Set2', legend=False)
        sns.stripplot(data=df, x='Model', y=metric, color='black', alpha=0.6, ax=axes[i], jitter=False)
        axes[i].set_title(metric)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Score')

    # Hide the empty 6th subplot (5 metrics in a 2x3 grid)
    axes[-1].axis('off')

    plt.tight_layout()
    plt.show()

def perform_statistical_analysis(df, results, metric='QWK'):
    """
    Rank models by mean score on `metric` across outer folds, test whether
    the differences are statistically significant (Friedman test across all
    models, then pairwise Wilcoxon tests of the best model against each of
    the others), and print the winning model's final hyperparameters.

    Metric defaults to QWK rather than F1-Macro because QWK is the scoring
    function actually used during hyperparameter tuning (qwk_scorer in
    hyperparameter_tuning.py) and is the metric that best reflects the goal
    of a diagonal confusion matrix on this ordinal, imbalanced target
    """
    print("\n" + "="*50)
    print(f" STATISTICAL ANALYSIS ({metric})")
    print("="*50)
    
    # Grouping to average metrics
    mean_scores = df.groupby('Model')[metric].mean().sort_values(ascending=False)
    print(f"\nAverage {metric} per model:")
    print(mean_scores.to_string())
    
    models = mean_scores.index.tolist()
    
    # Extract arrays for statistical testing (folds are paired for the same seed)
    model_arrays = {model: df[df['Model'] == model][metric].values for model in models}
    
    # Friedman test (suitable for repeated/paired measures across multiple algorithms)
    stat, p_value = friedmanchisquare(*[model_arrays[m] for m in models])
    print(f"\nGlobal Friedman test - p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("There is a statistically significant difference between the models")
        print("Running a Wilcoxon test between the first and the others...")
        
        # If a model wins in all folds against the other models, the mathematically
        # possible minimum p-value of a two-tailed Wilcoxon test is 0.0625 due to the sample size (N=5).
        # The Wilcoxon test is underpowered and cannot exceed the threshold of 0.05 because
        # the mathematical formula for the minimum achievable p-value is, in fact,
        # p = 2 x (1/2^N)
        best_model = models[0]
        for other_model in models[1:]:
            stat_w, p_w = wilcoxon(model_arrays[best_model], model_arrays[other_model])
            print(f" - {best_model} vs {other_model}: p-value = {p_w:.4f}")
    else:
        print("There is no statistically significant difference between the models (small sample, n=5)")
    
    # Best model selection
    best_model_name = models[0]
    print("\n" + "="*50)
    print(f" BEST MODEL: {best_model_name}")
    print("="*50)
    
    best_params = results[best_model_name].get('best_params_final_fit', {})
    final_features_count = results[best_model_name].get('final_selected_features_count', 'N/A')
    
    print(f"\nBest Hyperparameters:")
    print(json.dumps(best_params, indent=4))
    print(f"\nNumber of final features selected: {final_features_count}")

def main():
    parser = argparse.ArgumentParser(description="Results analysis Nested CV")
    parser.add_argument('--path_json', required=True, type=str, help="Path of the JSON file to analyze")
    parser.add_argument('--metric', default='QWK', type=str,
                         choices=['QWK', 'Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)'],
                         help="Metric used for model ranking and statistical testing (default: QWK)")
    args = parser.parse_args()
    
    # Load the nested-CV results produced by hyperparameter_tuning.py
    try:
        results = load_data(args.path_json)
    except FileNotFoundError:
        print(f"Error: Cannot find file '{args.path_json}'")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{args.path_json}' is not a valid JSON")
        return
        
    # Flatten JSON results into a long-format DataFrame (one row per model/fold)
    df = extract_metrics_to_dataframe(results)
    
    # Plotting boxplots for all metrics (QWK, Accuracy, Precision, Recall, F1)
    plot_metrics(df)
    
    # Statistical test + best-model report, ranked on args.metric (QWK by default)
    perform_statistical_analysis(df, results, metric=args.metric)

if __name__ == "__main__":
    main()