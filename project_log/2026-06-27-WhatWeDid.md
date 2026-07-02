# What We Did So Far

This log tracks our progressive engineering steps, architectural milestones, and technical choices during the development of the **HitPredictor-Steam** project. It serves as our clear, simple-language record of how we got here and why we made specific design choices along the way.

---

## 🚀 Milestone 1: Environment & Project Infrastructure

* **What we did:** We set up our working environment inside a **WSL (Windows Subsystem for Linux)** environment and fully containerized our application stack using **Docker Compose**. We initialized our Git repository with rigorous `.gitignore` parameters to protect against tracking heavy data assets or caching pools.
* **Why we chose this:** We chose a containerized Docker setup to ensure that both of us share a completely identical, isolated Python runtime. This eliminates any "it works on my machine" library variance or operating system conflicts, providing total reproducibility from day one.

## 🧼 Milestone 2: Baseline Data Hygiene & Cleaning (Updated 27/06/2026)

* **What we did:** We implemented a major optimization and decoupling of `data_cleaning.py`. 
    1. We extracted all localized helper functions, column-parsing engines, and text format cleaners into an independent file module named `data_cleaning_utils.py`.
    2. We refactored `data_cleaning.py` to stream its execution pipeline directly out of standalone `.zip` archives (`steam_app_data.zip` and `steamspy_data.zip`) using Pandas' native zipped stream-reading backend. 
    3. We permanently purged the uncompressed CSV data files and the heavy global `raw_dataset.zip` container archive from our local infrastructure space.
* **Why we chose this:** Decoupling logic isolates stateful calculations from macro orchestration loops, making the high-level codebase clean and highly human-readable for the entire team. Ingesting directly from tight, localized compressed archives optimizes shared container disk storage and eliminates the manual, fragile host decompression steps from our pipeline setup.

## ⚖️ Milestone 3: Resolving the Target Class Imbalance

* **What we did:** We updated our data cleaning pipeline to directly resolve an extreme class imbalance present in the original SteamSpy ownership categories. Because Steam market adoption profiles adhere to a strict power-law distribution, a minor group of blockbuster games represent extreme outliers while thousands of entries sit in the long-tail (with some native classes holding only a single game). We map and group these original 13 string-based categories into **5 macro-balanced, ordinal tiers** (Class 0: *The Indie Long-Tail* up to Class 4: *Mega-Hit/Blockbuster AAA*). Additionally, we removed the redundant `is_free` indicator since its variance is natively captured by the continuous `price` feature.
* **Why we chose this:** Standard machine learning classifiers completely ignore or collapse on minority classes when distributions are heavily imbalanced. Compressing the targets into broader semantic ranges guarantees our algorithms receive a healthy distribution of training examples per target tier, allowing the models to isolate real predictive signatures.

## 🏗️ Milestone 4: Leakage-Free Hyperparameter Tuning & Pipelines

* **What we did:** We engineered a highly professional machine learning scaffolding by dividing our code into specialized production scripts: `data_preprocessing.py`, `hyperparameter_tuning.py`, and `train_pipeline.py`. 
    1. We built a custom scikit-learn estimator called `SteamFeatureExtractor` to encapsulate our multi-label binarization (MLB), text TF-IDF vectorization, and Sentence Transformer embedding/PCA workflows.
    2. We introduced a `CorrelationRemover` to dynamically drop highly linear features ($r > 0.95$) based purely on the training folds.
    3. We implemented **Nested Cross-Validation** (5-fold outer, 3-fold inner) coupled with a `SelectFromModel` feature selector to isolate the best structural parameters for a `RandomForestClassifier` using macro-F1 scoring optimization.
* **Why we chose this:** Standard categorical and text vectorization routines (like TF-IDF or PCA) cause silent **Data Leakage** if they are applied globally before splitting data. Wrapping our custom feature extraction routines directly inside a scikit-learn `Pipeline` guarantees that vectorization, scaling, feature selection, and model tuning are fit *exclusively* on training splits and then purely applied during validation. This ensures our evaluation metrics are perfectly stable and free of target leakage.

---

## 🧪 Project Decisions and Experimentations Log

### 1. Fast Model Testing Strategy via Stratified Prototyping
* **The Problem:** Running a dense grid search over multi-thousand feature arrays inside a nested cross-validation loop is computationally prohibitive on local development machines, creating immense iteration friction when choosing between model families (e.g., Random Forest, Decision Trees, XGBoost).
* **The Choice:** We chose **Stratified Train-Test Split Prototyping** to draw a balanced 15% subset of the full dataset for rapid testing phase rounds.
* **Why we chose this option:** A naive random sample risks completely dropping or heavily underrepresenting our rare high-end blockbuster instances (Class 4). Stratifying ensures that our prototype subset precisely maintains the core characteristics, class properties, and target percentages of the parent dataset. This safe mathematical trade-off drops training execution speeds by 80–90%, unlocking rapid architecture diagnostics while maintaining an unbiased experimental lane.

### 2. Validation of Dataset Shape and Target Distributions
* **The Problem:** Verifying the data source dimensions and consistency across the initial raw files (`steamspy_data.csv`, `app_list.csv`, and `steam_app_data.csv`) before pipeline orchestration.
* **The Experiment:** We evaluated the raw shapes and heads directly inside an isolated kernel session. The check showed exactly **29,235 rows** spanning across all three target assets (20 features in SteamSpy, 2 features in the basic index, and 39 rich descriptive columns in Steam App Data).
* **Why we chose this option:** This profiling check confirms a clean 1-to-1 primary key relationship via `appid` across our raw structural endpoints. It establishes the baseline target distribution bounds and anchors our operational cleaning parameters, confirming that our macro-binning dictionaries map perfectly to the 29,235 records without unhandled loss.

### 3. Parallelization Pitfalls and Process-Limit Resolution (27/06/2026)
* **The Problem:** The nested evaluation loop completely deadlocked local machine runtimes overnight without making step-wise progress due to process over-subscription.
* **The Experiment:** Diagnostic terminal analysis confirmed that hard-coding `n_jobs=-1` at multiple nesting depths caused an explosive fork-bomb behavior. Upon restricting execution parameters, we initiated a controlled verification execution utilizing a 15% stratified prototype sample size ($28,871$ instances) to check class percentage distributions (ranging from Class 0 at $69.0\%$ down to Class 4 at $0.9\%$).
* **Why we chose this option:** We limited the parallel orchestration tree to exactly **3 active concurrent processes running sequential batch validation blocks**. Visual tracking via localized progress bars confirmed steady iteration velocities ($\approx 15.0\text{s}$ per iteration) without deadlocking the Linux hardware environment, establishing a stable baseline to capture hyperparameter tuning metrics.

### 4. Resolving Nested Cross-Validation Macro-F1 `NaN` Metrics (27/06/2026)
* **The Problem:** The cross-validation tracking logs threw an unhandled `F1 score = NaN` condition across every iteration trial step, threatening runtime validation stability.
* **The Diagnostic Analysis:** Identified two cascading pipeline bugs:
    1. **Data Dilution Erasure:** Shrinking the evaluation slice down inside nested splits causes rare high-tier instances (Class 4/blockbusters) to completely disappear from minor validation folds, forcing metric calculation divisions by zero.
    2. **Over-Aggressive Filtering:** Hyperparameter variance checks within `SelectFromModel` were choosing threshold parameters (mean/median thresholds) that wiped out essential features entirely before hitting the estimator layer.
* **Why we chose this option (The Debugging Framework):** To resolve this safely without long execution loops, we established a strict three-tier environment progression protocol:
    * **Phase 1: Tiny Prototype Split:** Inject a synthetic, ultra-small "micro-dataset" specifically for local print-debugging to eliminate loop latencies.
    * **Phase 2: Inner-Loop Isolation:** Extricate the inner cross-validation loop entirely from the complex nested tree framework, debugging its hyperparameter bounds and feature selection cutoff margins as a standalone script.
    * **Phase 3: Fallback Parameters:** Explicitly inject precision handling parameters into the scoring layers (such as `zero_division=0` in scikit-learn's metric hooks) to protect against runtime zero-count crashes on sparse classes, before slowly scaling execution back up to the master production loop.
