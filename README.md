# Hit Predictor-Steam 🎮📊

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

**A machine learning pipeline for game success prediction on Steam**

## 📖 Project Overview
The **Hit Predictor-Steam** project delivers an end-to-end machine learning pipeline designed to model, evaluate, and predict the commercial adoption trajectories (ownership tiers) of PC games on Steam. The primary objective is to provide independent developers and publishers with a data-driven decision-making tool prior to capital allocation.

To prevent **Data Leakage**, the pipeline strictly operates in two distinct modes:
*   **Pre-Release Mode (Day-Zero):** Uses only intrinsic game metadata available *before* launch (e.g., pricing, hardware constraints, language localization, semantic descriptions).
*   **Post-Release Mode:** Incorporates community engagement endpoints (reviews, playtime, concurrent users).

## Machine Learning Architecture

### 1. Advanced Feature Engineering & NLP
*   **Semantic Embeddings:** High-dimensional dense textual embeddings of game descriptions using `SentenceTransformer` (`all-mpnet-base-v2`).
*   **Supervised Dimensionality Reduction (PLS-DA):** Projects raw sentence embeddings into discriminative latent components correlated with ownership tiers.
*   **Dynamic Parsing:** Automated extraction of hardware requirements (RAM in GB, CPU/GPU tiers) and flattening of nested JSON storefront metadata.

### 2. Class Imbalance Resolution (The Power-Law Problem)
The PC gaming market follows a steep power-law distribution. To counter the massive "Indie Long-Tail", the pipeline uses:
*   **Dynamic Majority Undersampling:** Caps the dominant class volume safely.
*   **Targeted SMOTENC:** Synthesizes minority instances (e.g., AAA Hits) directly in a mixed continuous/categorical space.
*   **Tomek Links:** Prunes spatial decision boundaries to reduce adjacent-class ambiguity.
*   **Differential Evolution Calibration:** Post-hoc probability calibration to maximize the Quadratic Weighted Kappa (QWK) penalty function.

### 3. Hyperparameter Tuning & Validation
*   **Validation Strategy:** $5\times3$ Nested Cross-Validation to guarantee unbiased generalization metrics.
*   **Optimization Engines:** Bayesian Optimization via **Optuna** and a custom **Cached Unified Grid Search** (`joblib` cache) to manage high computational overhead.
*   **Statistical Testing:** Non-parametric hypothesis testing (Friedman & Wilcoxon Signed-Rank) to validate model superiority.

### 4. Explainable AI (XAI) Suite
*   **Inverse Scaling Mechanism:** Unscales numeric features back to real-world units (e.g., True Price, GB of RAM) before interpretation.
*   **Global & Cohort (SHAP):** Beeswarm and Waterfall plots to define market-wide drivers for Hits vs. Flops.
*   **Local Explanations (LIME):** Targeted "what-if" analyses for individual game prototypes.

## Repository Structure

```text
├── data_cleaning.py              # Main execution script for raw data processing
├── data_cleaning_utils.py        # Utility functions for JSON parsing, HW extraction, text cleaning
├── data_preprocessing.py         # Custom sklearn transformers (SteamFeatureExtractor, PLSDA, SMOTENC)
├── gridSearch_plsda_tfidf.py     # Lightweight grid search for structural text parameters
├── hyperparameter_tuning.py      # Optuna Bayesian optimization with Nested CV
├── cached_unified_search.py      # Brute-force joint optimization with joblib disk caching
├── analyze_results.py            # Automated statistical significance testing (Friedman/Wilcoxon)
└── train_pipeline.py             # Final production training, DE calibration, and XAI plot generation
```

## Execution & Usage

### 1. Data Cleaning
Transforms raw `steam_app_data.csv` and `steamspy_data.csv` into an ML-ready state:
```bash
python data_cleaning.py
```

### 2. Hyperparameter Tuning (Nested CV)
Execute the Bayesian search. Use the `--pre_release` flag to prevent data leakage for day-zero predictions.
```bash
python hyperparameter_tuning.py --pre_release --seed 1
```

### 3. Statistical Analysis
Analyze the out-of-fold metrics across models:
```bash
python analyze_results.py --path_json tuning_results_pre_release.json --metric QWK
```

### 4. Final Training & XAI Generation
Train the final champion model (XGBoost), calibrate probability weights via Differential Evolution, and generate local/global interpretability plots.
```bash
python train_pipeline.py --pre_release
```

## Academic Context
This project was developed as part of the Master's Degree in Artificial Intelligence and Data Engineering at the University of Pisa.
### Students:
- Andrea Zanin
- Pedro Carneiro