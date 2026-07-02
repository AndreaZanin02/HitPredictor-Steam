# What We Did So Far

This log tracks our progressive engineering steps, architectural milestones, and technical choices during the development of the **HitPredictor-Steam** project. It serves as our clear, simple-language record of how we got here and why we made specific design choices along the way.

---

## 🚀 Milestone 1: Environment & Project Infrastructure

* **What we did:** We set up our working environment inside a **WSL (Windows Subsystem for Linux)** environment and fully containerized our application stack using **Docker Compose**. We initialized our Git repository with rigorous `.gitignore` parameters to protect against tracking heavy data assets or caching pools.
* **Why we chose this:** We chose a containerized Docker setup to ensure that both of us share a completely identical, isolated Python runtime. This eliminates any "it works on my machine" library variance or operating system conflicts, providing total reproducibility from day one.

## 🧼 Milestone 2: Baseline Data Hygiene & Cleaning (Updated 28/06/2026)

* **What we did:** We implemented a major optimization and decoupling of `data_cleaning.py`. 
    1. We extracted all localized helper functions, column-parsing engines, and text format cleaners into an independent file module named `data_cleaning_utils.py`.
    2. We refactored `data_cleaning.py` to stream its execution pipeline directly out of standalone `.zip` archives (`steam_app_data.zip` and `steamspy_data.zip`) using Pandas' native zipped stream-reading backend. 
    3. We permanently purged the uncompressed CSV data files and the heavy global `raw_dataset.zip` container archive from our local infrastructure space.
    4. We expanded the pipeline to explicitly filter out target characters outside western text encoding fields and handle missing numeric attributes. Specifically, we modified the script to drop games containing native Japanese text strings and implemented full imputation across remaining feature arrays.
* **Why we chose this:** Decoupling logic isolates stateful calculations from macro orchestration loops, making the high-level codebase clean and highly human-readable for the entire team. Ingesting directly from tight, localized compressed archives optimizes shared container disk storage and eliminates the manual, fragile host decompression steps from our pipeline setup. Furthermore, native Japanese characters caused unhandled formatting failures inside our text preprocessing and embedding vectorization matrices. Filtering out these records removed roughly 700 rows, adjusting our master working dataset size down to approximately 28,100 records. Because the overwhelming majority of these dropped titles belonged to the heavy Class 0 category, this pruning step safely reduced our largest target group (adjusting Class 0 representation down to 64%) without losing highly predictive high-end blockbuster instances.

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
* **Why we chose this option:** We limited the parallel orchestration tree to exactly **3 active concurrent processes running sequential batch validation blocks**. Visual tracking via localized progress bars confirmed steady iteration velocities ($ pprox 15.0	ext{s}$ per iteration) without deadlocking the Linux hardware environment, establishing a stable baseline to capture hyperparameter tuning metrics.

### 4. Resolving Nested Cross-Validation Macro-F1 `NaN` Metrics (Updated 28/06/2026)
* **The Problem:** The nested cross-validation grid search threw persistent `score=nan` validation flags across every single hyperparameter combination trial step during pipeline iteration.
* **The Diagnostic Analysis:** In-depth debugging of the runtime inputs revealed a fatal data pipeline mismatch. The data cleaning engine was passing unhandled `NaN` elements hidden inside the Recommended Hardware RAM configuration columns straight into the estimator. Random Forest classifiers inherently cannot process missing values natively, causing them to fail silently during validation runs and nullify the scoring outputs across all evaluation thresholds (`mean`, `median`, `0.75*median`).
* **Why we chose this option:** We updated the upstream `data_cleaning.py` pipeline to enforce full `NaN` numerical imputation across all missing values before feeding inputs to the model. Resolving this data hygiene bottleneck restored pipeline stability, allowing the nested cross-validation loop to run end-to-end and successfully yield initial validation metrics.

### 5. Text Vectorizer Computational Redundancy & Training Loop Latency (28/06/2026)
* **The Problem:** Training execution times spiked significantly during nested loop evaluations, taking nearly 4 minutes for a single execution fit even when restricted to a downscaled 15–20% dataset split (~4,300 to 5,500 games).
* **The Experiment:** Profiling the internal execution tree revealed that the `transform_text_feature` logic inside `data_preprocessing.py` was instantiating and fitting a completely fresh transformer instance from scratch inside every single inner cross-validation split, leading to hundreds of redundant initializations that wasted heavy memory and CPU cycles.
* **Why we chose this option:** We localized this vectorization bottleneck as a priority for structural script refactoring. Overhauling this architecture will prevent text-transformer reconstruction loops, keeping the data runtime automated and efficient.

### 6. Baseline Validation Benchmarks & Model Selection Architecture (Updated 28/06/2026)
* **The Problem:** Establishing an empirical baseline on a stratified prototype split (~5,500 games) to verify that the integrated data engineering structures can capture predictive signatures.
* **The Choice:** We successfully executed the first complete baseline run using a Random Forest model on a university GPU workstation. The cross-validation loop yielded a **Mean Accuracy of $0.769 \pm 0.004$** and a **Mean Macro-F1 score of $0.405 \pm 0.017$**. Given that the target space is distributed across 5 distinct classes (where a naive random guess represents a 20% baseline), this initial result confirms that the pipeline is capturing distinct mathematical patterns.
* **Why we chose this option:** This validation baseline proves our core preprocessing architecture works safely. The pipeline successfully isolated the optimal parameter matrix for subsequent production runs:
    * `classifier__max_depth`: 20
    * `classifier__min_samples_split`: 5
    * `classifier__n_estimators`: 100
    * `feature_selection__threshold`: `mean`
    Moving forward, we can expand our script to incorporate automated multi-model selection (comparing Simple Decision Trees, Random Forests, and advanced gradient boosters), execute a final 15-hour full-scale dataset run overnight, and integrate Explainable AI (XAI) frameworks to map our predictive signatures.

### 7. Core Compute Infrastructure & Resource Provisioning (28/06/2026)
* **The Problem:** Searing summer temperatures risk causing thermal throttling on local development machines during heavy, multi-hour full-dataset execution runs. Additionally, academic constraints prevent access to university datacenter hardware for this project module.
* **The Choice:** We initiated an infrastructure expansion strategy. We are leveraging internal corporate workspace availability to request a dedicated high-performance Virtual Machine (VM) from a commercial datacenter environment, using the local workstation's hardware profile as a reference parameters anchor.
* **Why we chose this option:** Offloading 15-hour execution blocks to robust, temperature-controlled server environments protects local developer machines from thermal wear, safeguards execution stability, and guarantees rapid training iteration times for our full-scale model comparison.

### 8. Strict Algorithmic Reproducibility and Seed Harmonization (29/06/2026)
*   **The Problem:** Evolving complex stochastic estimators (such as Random Forest and XGBoost) across varying execution environments creates a high risk of numerical variance. If pseudo-random initializations diverge between developer machines or the high-performance GPU workstation, the process breaks the gold standard of scientific reproducibility.
*   **The Choice:** We established a strict protocol making all pipeline entry points completely deterministic. While the inner pipeline architecture defaults to an explicit constant random state seed of `42`, we overhauled the framework to expose the seed directly as a script parameter. 
*   **Why we chose this option:** Enforcing a central, exposed seed ensures that any experiment can be accurately repeated on any deployment architecture. When an execution configuration is re-run, it will yield identical cross-validation splits, feature selection matrices, and model weight parameters. This approach guarantees verifiable transparency for final grading and assessment.

### 9. Version Control Protocol for Model Comparisons vs. Source Code Evolution (29/06/2026)
*   **The Problem:** As we test multiple machine learning algorithms, modifying a single file risks losing the exact configurations and history of previous attempts, while archiving minor bug fixes across multiple file clones creates redundant script bloat in the main repository path.
*   **The Choice:** We decoupled our version management into two clear strategies:
    1.  **Source/Utility Code:** Incremental data engine fixes (such as adding malformed line filtering routines in `data_cleaning.py` and `data_cleaning_utils.py`) are handled exclusively through systematic **GitHub commits**. This tracks the lines changed without littering the workspace with duplicate scripts.
    2.  **Model Execution & Pipelines:** Distinct model pipeline training sweeps are explicitly split out (e.g., `train_ppln_1.py`, `train_ppln_2.py`) or structured inside versioned folders.
*   **Why we chose this option:** This strategy preserves full process transparency and reproducibility for the evaluating professors. It keeps the core `src/` directory production-ready while archiving our historical exploratory trajectory, making it easy to prove exactly how we reached our final architectural model selection.

### 10. Multi-Model Evaluation and Dynamic Command-Line Orchestration (29/06/2026)
*   **The Problem:** Scaling beyond a single classifier required manually changing scripts to swap models, modify feature categories, or adjust dataset splits, which added friction to the development loop.
*   **The Choice:** We completed a massive evolution of the nested cross-validation pipeline by integrating `argparse` to create a highly flexible, parameter-driven CLI framework. The updated `hyperparameter_tuning.py` automatically evaluates an expanded algorithm pool including **Decision Trees, Random Forests, and XGBoost**, governed by key functional switches:
    *   `--full`: Toggles execution between the 20% stratified prototype split and the complete 100% dataset matrix.
    *   `--seed`: Configures the base synchronization random state seed (defaulting to 1).
    *   `--pre_release`: Automatically drops post-launch metrics (e.g., reviews, Metacritic counts, player sentiment scores) to train a specialized predictive model focused purely on pre-release indicators.
*   **Why we chose this option:** This design decouples our core algorithmic execution from hardcoded configurations. The script automatically serializes a comprehensive metrics `json` artifact containing cross-validation scores, the confusion matrix for each algorithm, optimal hyperparameter pairs, and the selected features. It also exports automated **boxplots** of cross-validation results to streamline model interpretation.

### 11. Boundary-Cleaning Undersampling for Extreme Long-Tail Class Imbalance (29/06/2026)
*   **The Problem:** Class 0 (The Indie Long-Tail) represents a crushing 69% majority of the dataset. Leaving this distribution untouched forces estimators to optimize globally for Class 0 accuracy while ignoring rarer classes, while standard random undersampling risks discarding valuable underlying data patterns.
*   **The Choice:** We implemented a targeted, non-random spatial undersampling routine specifically on Class 0 data points within the tuning pipeline. This operation scales the massive Class 0 cohort down by a factor of 4 (reducing it from 18,971 instances down to approximately 5,000 instances). Instead of choosing records completely at random, the algorithm uses a boundary-cleaning function to clarify feature spaces.
*   **Why we chose this option:** Randomly pruning elements can strip away vital edge-case data characteristics. This non-random function cleans up overlapping instances near class boundaries, creating distinct spatial divisions between target groups in the feature space. This reduction significantly accelerates cross-validation training speeds, allowing the pipeline to iterate smoothly across multi-model grids without deadlocking our computing resources.

### 12. Strategic Model Forking: Pre-Release vs. Post-Release Architectures (30/06/2026)
*   **The Problem:** Predictive features have fundamentally different lifecycles. A developer needs to predict market success *before* publishing a game, meaning they cannot use post-launch metrics like critic scores or user reviews. Conversely, a post-launch analysis requires all available data to isolate long-term player retention factors.
*   **The Choice:** We split our machine learning goals into two distinct functional pipelines governed by the custom `--pre_release` CLI switch:
    *   **Pre-Release Pipeline:** Restricted exclusively to upstream features available prior to launch (e.g., system requirements, hardware RAM minimums, genre configurations, text descriptions). It serves as a decision-support tool for game developers.
    *   **Post-Release Pipeline:** Utilizes the comprehensive dataset including downstream interaction features (e.g., critic scores, user review counts, average playtime metrics) to map out long-tail retention signatures.
*   **Why we chose this option:** Training a single model on all features introduces target leakage for pre-launch use cases. Separating the models ensures our tools are practically useful for developers during the early production stages while still allowing for deep, post-launch market analysis.

### 13. Feature Constraints and Class Confusion Analysis (30/06/2026)
*   **The Problem:** Initial baseline runs on the 20% stratified prototype split revealed high predictive accuracy for lower tiers, but noticeable confusion and misclassifications within the highest success brackets (Classes 3 and 4), as documented in the `tuning_results.json` log in `image_7bc3e1.png`.
*   **The Diagnostic Analysis:** A review of the feature matrix showed that technical specifications (like required RAM gigabytes) and text metadata are highly effective for rough, baseline classification but lack the granularity to distinguish high-end blockbusters from moderate hits. Real-world market success is driven by external variables outside the scope of raw software files, such as marketing budgets, advertising campaigns, and streaming/influencer traction.
*   **Why we chose this option:** Rather than artificially forcing the models to find signal in noisy technical features, we logged this structural limitation. This behavior supports our transition to XGBoost—which handles these non-linear feature interactions better than simpler models—and justifies the introduction of post-launch metrics to clear up confusion in the upper tiers.

### 14. Empirical Model Sweep and Transition to Black-Box XAI (30/06/2026)
*   **The Problem:** Selecting a final model family requires objective, data-driven proof across all cross-validation folds, especially when moving from highly interpretable white-box models (Decision Trees) to complex black-box models (XGBoost).
*   **The Experiment:** We initiated an overnight multi-model nested cross-validation loop on our compute infrastructure. The tuning process automatically outputs a detailed master JSON file tracking evaluation metrics, selected features, and historical confusion matrices across all 5 outer splits. Preliminary logs indicate that XGBoost achieves a dominant Macro-F1 score of approximately **$0.60$** on the 5-class distribution, outperforming simpler models by a margin of $+10\%$.
*   **Why we chose this option:** While white-box Decision Trees offer built-in interpretability, the significant performance boost of XGBoost makes it our clear choice for deployment. Because XGBoost operates as a complex black-box model, this decision directly sets up our next step: integrating Post-Hoc **Explainable AI (XAI)** frameworks (such as SHAP or LIME) to extract feature importance and keep the model transparent.

### 15. Macro-Class Compaction Proposal for Pre-Launch Predictors (30/06/2026)
*   **The Problem:** Because pre-release models must drop highly predictive post-launch interaction data (like user reviews), forcing them to classify across 5 highly unbalanced groups accelerates performance degradation.
*   **The Choice:** We proposed an experimental path to group our 5 ordinal target tiers into **3 macro-balanced, semantic categories**: `flop`, `middle`, and `success`.
*   **Why we chose this option:** Compacting the targets reduces class sparsity and stabilizes the model's decision boundaries. This structural adjustment helps compensate for the loss of post-launch features, preserving stable Macro-F1 performance and providing developers with clear, reliable target groups.

### 16. Statistical Model Selection and Boxplot Validation (01/07/2026)
*   **The Problem:** Selecting the final model architecture requires rigorous validation across all outer folds to ensure that a single model's outperformance is statistically stable and not an artifact of favorable data splits.
*   **The Experiment:** We plotted the distribution of performance metrics across the nested cross-validation outer folds using `image_7bd644.png`. The visualization definitively establishes **XGBoost** as our top-performing model family across all four audited dimensions:
    *   **Accuracy:** XGBoost achieves a stable baseline tier between $0.75$ and $0.77$, outperforming Random Forest ($\approx 0.743 - 0.755$) and Decision Trees ($\approx 0.69 - 0.71$).
    *   **Precision (Macro):** XGBoost leads within a tight interval of $0.575 - 0.61$.
    *   **Recall (Macro):** XGBoost clusters tightly around $0.615 - 0.625$.
    *   **F1 (Macro):** XGBoost dominates with a median score of approximately **$0.60$**, outperforming Random Forest (median $\approx 0.56$) and leaving baseline Decision Trees behind (median $\approx 0.48$).
*   **Why we chose this option:** The distribution boxes confirm that XGBoost delivers superior macro-F1 values alongside minimal variance across folds. This statistical proof justifies using this black-box model as our primary architecture, while confirming our earlier observation that technical data limits performance to the $0.60 - 0.70$ macro-F1 range due to unmodeled external market forces like marketing budgets and viral influencer traction.

### 17. Implementation of Twin Post-Hoc Explainable AI (XAI) Frameworks (01/07/2026)
*   **The Problem:** Because XGBoost operates as a highly complex non-linear ensemble, it acts as a black box. To make our predictions trustworthy for developers and stakeholders, we must extract human-readable explanations of why the model flags specific games as hits or flops.
*   **The Choice:** We integrated an automated explainability suite into `train_pipeline.py`. Once the pipeline fits the optimal XGBoost model parameters, it automatically computes and serializes both global and local post-hoc explanations:
    *   **SHAP (SHapley Additive exPlanations):** Computes game-theoretic attribution values to map global feature importance across the entire dataset.
    *   **LIME (Local Interpretable Model-agnostic Explanations):** Generates local perturbations around individual sample points to explain specific predictions on a case-by-case basis.
*   **Why we chose this option:** Combining SHAP and LIME gives us a comprehensive explanation framework. SHAP provides a mathematically sound view of overall feature importance across the whole pipeline, while LIME allows developers to inspect individual games and see exactly how minor tweaks to pre-release properties might alter market success predictions.

### 18. Execution and Scalability of the Pre-Release Pipeline Sweep (01/07/2026)
*   **The Problem:** Validating the developer-focused pre-release model requires running our full multi-model nested cross-validation loop on a limited feature set, which demands a long, uninterrupted compute window.
*   **The Choice:** We set up the execution parameters to run our complete training pipeline on the high-performance workstation using the simple CLI command flag `-p` or `--pre_release`. This automatically drops post-launch variables and routes the remaining data through the 14-hour optimization loop.
*   **Why we chose this option:** Our modular CLI design allows us to run the massive pre-release experiment without making manual code modifications. The script automatically handles the feature constraints and exports a standalone results JSON, keeping our experiment reproducible while we evaluate whether compacting the targets into 3 macro-classes balances out the loss of post-launch features.
