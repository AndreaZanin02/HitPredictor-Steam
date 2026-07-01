import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_metrics_to_dataframe(results):
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
                'F1 (Macro)': metrics.get('f1_macro')
            })
    return pd.DataFrame(records)

def plot_metrics(df):
    metrics = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Distribution of Metrics (Nested CV outer folds)', fontsize=16)
    
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        sns.boxplot(data=df, x='Model', y=metric, ax=axes[i], hue='Model', palette='Set2', legend=False)
        sns.stripplot(data=df, x='Model', y=metric, color='black', alpha=0.6, ax=axes[i], jitter=False)
        axes[i].set_title(metric)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Score')
        
    plt.tight_layout()
    plt.show()

def perform_statistical_analysis(df, results):
    print("\n" + "="*50)
    print(" STATISTICAL ANALYSIS (F1-Macro)")
    print("="*50)
    
    # Grouping to average metrics
    mean_scores = df.groupby('Model')['F1 (Macro)'].mean().sort_values(ascending=False)
    print("\nAverage F1-Macro per model:")
    print(mean_scores.to_string())
    
    models = mean_scores.index.tolist()
    
    # Extract arrays for statistical testing (folds are paired for the same seed)
    model_arrays = {model: df[df['Model'] == model]['F1 (Macro)'].values for model in models}
    
    # Friedman test (suitable for repeated/paired measures across multiple algorithms)
    stat, p_value = friedmanchisquare(*[model_arrays[m] for m in models])
    print(f"\nGlobal Friedman test - p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("There is a statistically significant difference between the models.")
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
    args = parser.parse_args()
    
    try:
        results = load_data(args.path_json)
    except FileNotFoundError:
        print(f"Error: Cannot find file '{args.path_json}'")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{args.path_json}' is not a valid JSON")
        return
        
    df = extract_metrics_to_dataframe(results)
    
    # Plotting boxplots
    plot_metrics(df)
    
    # Statistical test
    perform_statistical_analysis(df, results)

if __name__ == "__main__":
    main()