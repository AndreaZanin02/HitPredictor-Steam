# MASTER ARCHITECTURE AND LOG DOCUMENT: HitPredictor-Steam

This document consolidates the history of decisions, experimentations, and the definitive architectural blueprint of the **HitPredictor-Steam** project, focused on predicting the commercial trajectory and market adoption status ("hit status") of video games using historical data from the Steam platform and SteamSpy metrics.

---

## SECTION 1: Development History & Decision Logs (Project Evolution)

This section organizes the chronological logs of project progress, detailing the scientific motivations and engineering choices made by the team throughout iterations.

### 1.1. Initial Progress Log (2026-06-21)

*   **Project Environment & Infrastructure:**
    *   **What was done:** Configuration of the working environment inside the **WSL (Windows Subsystem for Linux)** subsystem and full containerization using **Docker Compose**. Initialization of the Git repository with structured parameters in `.gitignore` to protect against tracking heavy data assets.
    *   **Why we chose this:** The containerized setup via Docker ensures that the entire team develops on a completely identical and isolated Python runtime. This eliminates "it works on my machine" library version conflicts and guarantees code reproducibility from day one.
*   **Base Data Hygiene & Parsing:**
    *   **What was done:** Creation of the initial data cleaning script (`data_cleaning.py`) to process raw sources (*Steam Store Metadata* and *SteamSpy Stats*). The script filters entries to keep only real games, discards unnecessary metadata (support links, EULA descriptions), removes HTML tags from textual descriptions, and executes regular expressions to extract structured numbers from system requirements (such as Minimum and Recommended RAM in GB).
    *   **Why we chose this:** Raw dumps from digital storefronts contain intense structural noise. By removing records that are not games (like DLCs or software) and eliminating presentation pollution early on, we protect downstream machine learning models from wasting memory and processing time on non-informative features.
*   **Confronting the Target Paradox:**
    *   **What was done:** Evaluation of the intended target variable — SteamSpy ownership fields — identifying that it was split into 13 high-variance intervals. Since Steam data follows a strict power-law curve, a tiny number of games are massive blockbusters, while thousands of long-tail games have low adoption. One of the classes contained only a single game. It was decided to compress these 13 intervals into **5 macro-balanced ordinal tiers** (Class 0 to Class 4).
    *   **Why we chose this:** Machine learning classifiers break down or entirely ignore rare classes if the target distribution is severely imbalanced. By re-binning targets into macro-significant cohorts (from *Indie Long-Tail* to *Mega-Hit Blockbuster*), we guarantee sufficient training instances per class for algorithms to extract real patterns.
*   **Design for Feature Leakage Mitigation:**
    *   **What was done:** Addressing a critical domain scope question: is the model a tool for developers planning an unreleased game, or a post-launch analysis tool? Including features like user review scores or total achievements to predict success *before* release creates a fatal data leak (target peeking). It was decided to separate the preprocessing approach into two distinct experimental tracks: **Pre-Launch Mode** and **Post-Launch Mode**.
    *   **Why we chose this:** A game cannot accumulate user reviews or Metacritic scores before it is released. Separating feature matrices into a pre-launch configuration (using base properties like price, genre, publisher history, and localized languages) ensures the simulation remains completely realistic for "day-zero" investment predictions.

---

### 1.2. Consolidated Decision and Experimentation Logs (Iterations from 26/06 to 02/07)

The following logs detail architectural choices made as the pipeline advanced from local prototypes to high-performance models. The original sequential numbers were extended with letters (`a`, `b`, `c`, etc.) to preserve history without generating numbering overlap.

*   **1. Rapid Model Testing Strategy via Stratified Prototyping:**
    *   **The Problem:** Running a dense grid search over matrices of thousands of features within a nested cross-validation loop is computationally prohibitive on local development machines, generating high iteration friction when choosing between model families (e.g., Random Forest, Decision Trees, XGBoost).
    *   **The Choice:** Selection of **Prototyping with Stratified Train-Test Split** to extract a balanced 15% subset of the full dataset for rapid testing rounds.
    *   **Why we chose this option:** Naive random sampling risks completely discarding or severely under-representing rare top-tier blockbusters (Class 4). Stratification ensures that the prototype subset precisely maintains the key characteristics, class properties, and target percentages of the parent dataset. This mathematical trade-off reduces training execution time by 80–90%, unlocking rapid architectural diagnostics while maintaining an unbiased experimental track.
*   **2. Dataset Format Validation and Target Distributions:**
    *   **The Problem:** Verifying data source dimensions and consistency across the initial raw files (`steamspy_data.csv`, `app_list.csv`, and `steam_app_data.csv`) before pipeline orchestration.
    *   **The Experiment:** Evaluation of raw formats directly within an isolated kernel session. The check displayed exactly **29,235 rows** spanning all three assets (20 features in SteamSpy, 2 features in the basic index, and 39 rich descriptive columns in Steam App Data).
    *   **Why we chose this option:** This profiling check confirms a clean 1-to-1 primary key relationship via `appid` across our structural raw endpoints. It establishes the operational boundaries of the target distribution and anchors our cleaning parameters, confirming that the macro-binning dictionaries map all 29,235 records perfectly without unhandled loss.
*   **3. Parallelization Pitfalls and Process Limit Resolution:**
    *   **The Problem:** The nested evaluation loop completely locked up team local machine runtimes overnight due to process over-subscription.
    *   **The Experiment:** Terminal diagnostic analysis confirmed that setting `n_jobs=-1` at multiple nesting levels caused an explosive "fork-bomb" behavior. By restricting execution parameters, a controlled verification run was initiated using the 15% stratified prototype sample ($28,871$ instances before strict cleaning) to check class percentage distributions (ranging from Class 0 at $69.0\%$ down to Class 4 at $0,9\%$).
    *   **Why we chose this option:** Limiting the parallel orchestration tree to exactly **3 active concurrent processes running sequential batch validation folds**. Visual tracking through local progress bars confirmed stable iteration speeds ($\approx 15.0\text{s}$ per iteration) without locking up the Linux hardware environment, establishing a stable foundation for capturing hyperparameter tuning metrics.
*   **4. Resolution of Macro-F1 `NaN` Metrics in Nested Cross-Validation:**
    *   **The Problem:** Cross-validation tracking logs threw a permanent `score=nan` / `F1 score = NaN` condition at each hyperparameter iteration step, threatening pipeline stability.
    *   **The Diagnostic Analysis:** Deep input debugging revealed two cascading bugs:
        1.  *Data Dilution:* Shrinking the evaluation slice within nested splits caused rare high-tier instances (Class 4/blockbusters) to completely disappear from smaller validation folds, forcing division-by-zero errors in metric calculations.
        2.  *Over-Hygienic Filtering:* The data cleaning engine was passing unhandled hardware recommended RAM `NaN` values hidden within configuration columns directly to the estimator. Native Random Forest classifiers do not process missing values, failing silently and nullifying outputs across all feature selection cutoff boundaries (`mean`, `median`, `0.75*median`).
    *   **Why we chose this option (The Debugging Framework):** Upgrading the upstream pipeline in `data_cleaning.py` to enforce full numerical imputation of `NaN` across all missing entries before feeding the model. To resolve this safely without long loops, a three-tier environment progression protocol was established:
        *   *Phase 1 (Micro-Dataset):* Injection of an ultra-reduced synthetic set specifically for local print-debugging, eliminating latencies.
        *   *Phase 2 (Inner Loop Isolation):* Extraction of the inner cross-validation loop to debug its hyperparameter boundaries and feature selection cutoff margins in isolation.
        *   *Phase 3 (Contingency Parameters):* Explicit injection of tolerance parameters in scoring layers (such as `zero_division=0` in scikit-learn metric hooks) to protect against runtime crashes on sparse classes, gradually scaling execution back to the master production loop. Resolving this bottleneck restored pipeline stability, allowing the loop to yield end-to-end validation metrics.
*   **5. Text Vectorizer Computational Redundancy & Training Loop Latency:**
    *   **The Problem:** Training execution times surged significantly during nested loop evaluations, taking nearly 4 minutes for a single fit even when restricted to a reduced 15–20% dataset split (~4,300 to 5,500 games).
    *   **The Experiment:** Profiling the internal execution tree revealed that the `transform_text_feature` logic within `data_preprocessing.py` was instantiating and fitting a brand new transformer from scratch inside each individual inner cross-validation split, driving hundreds of redundant initializations that drained memory and CPU cycles.
    *   **Why we chose this option (Resolution of Circular Dependency):** Identifying this vectorization bottleneck as a priority for script structural refactoring. The redesign unified text extraction engineering (TF-IDF, PLS-DA components) and classifier matrices into a simultaneous, cohesive search space, eliminating inefficient re-initializations and slashing pipeline stage processing times.
*   **6. Baseline Validation Benchmarks & Model Selection Architecture:**
    *   **The Problem:** Establishing an empirical baseline on a stratified prototype split (~5,500 games) to verify if integrated data engineering structures capture predictive signatures.
    *   **The Choice:** Successful execution of the first full baseline run using a Random Forest model on an enterprise/academic GPU workstation. The cross-validation loop generated a **Mean Accuracy of $0.769 \pm 0.004$** and a **Mean Macro-F1 score of $0.405 \pm 0.017$**. Given that the target space is distributed across 5 distinct classes (where an ingenious random guess represents a 20% base), this initial result confirmed that the pipeline captures distinct mathematical patterns.
    *   **Why we chose this option:** This validation baseline proved that our core preprocessing architecture operates safely. The pipeline successfully isolated the ideal parameter matrix for subsequent production runs: `classifier__max_depth: 20`, `classifier__min_samples_split: 5`, `classifier__n_estimators: 100`, and `feature_selection__threshold: 'mean'`. This paved the way for expanding scripts into automated multi-model selection (Decision Trees, Random Forests, and XGBoost), full-scale overnight executions, and Explainable AI (XAI) integration.
*   **7. Core Processing Infrastructure & Resource Provisioning:**
    *   **The Problem:** High summer temperatures risk causing thermal throttling on local developer setups during heavy, full-dataset execution runs lasting over 14 hours. Additionally, academic constraints prevented continuous access to university datacenter hardware for this specific project module.
    *   **The Choice:** Initiation of an infrastructure expansion strategy. Leveraging internal corporate workspace availability to request a dedicated high-performance Virtual Machine (VM) in a commercial datacenter environment, using the local workstation hardware profile as an anchor for benchmark metrics.
    *   **Why we chose this option:** Offloading 15-hour execution blocks to robust, temperature-controlled servers protects developers' local setups from thermal wear, safeguards execution stability, and ensures fast turnaround times for wide-scale model comparison.
*   **8. Strict Algorithmic Reproducibility and Seed Harmonization:**
    *   **The Problem:** Evolving complex stochastic estimators (like Random Forest and XGBoost) across varying execution environments creates a high risk of numerical variance. If pseudo-random initializations diverge between developer machines or the high-performance GPU workstation, the process breaks the gold standard of scientific reproducibility.
    *   **The Choice:** Establishment of a strict protocol making all pipeline entry points completely deterministic. While internal pipeline architecture defaults to a constant random state seed of `42`, the framework was refactored to expose the seed directly as a script parameter.
    *   **Why we chose this option:** Enforcing a central, exposed seed guarantees that any experiment can be repeated precisely across any deployment architecture. When an execution configuration is rerun, it will generate identical cross-validation splits, feature selection matrices, and model weight parameters. This ensures verifiable transparency for the final academic evaluation.
*   **9. Version Control Protocol: Model Comparisons vs. Source-Code Evolution:**
    *   **The Problem:** As multiple machine learning algorithms are tested, modifying a single file risks losing the exact settings and history of prior attempts, while archiving minor bug fixes in cloned duplicate files creates redundant script bloat in the main repository path.
    *   **The Choice:** Decoupling version management into two clear strategies:
        1.  *Source/Utility Code:* Incremental fixes to the data engine (such as adding malformed row filtering routines into `data_cleaning.py` and `data_cleaning_utils.py`) are managed exclusively through **systematic GitHub commits**. This tracks changed lines without polluting the workspace with duplicate scripts.
        2.  *Model Execution & Pipelines:* Model training sweeps for distinct pipelines are explicitly split out (e.g., `train_ppln_1.py`, `train_ppln_2.py`) or structured within versioned folders.
    *   **Why we chose this option:** This strategy preserves full process transparency and reproducibility for grading professors. It keeps the core `src/` directory production-ready while archiving our historical exploratory trajectory, making it simple to prove exactly how we reached our final architectural model selection.
*   **10. Multi-Model Evaluation and Dynamic Command-Line Orchestration (CLI):**
    *   **The Problem:** Scaling beyond a single classifier required manually altering scripts to toggle models, modify feature categories, or adjust dataset splits, adding friction to the development cycle.
    *   **The Choice:** Completion of a massive evolution in the nested cross-validation pipeline by integrating the `argparse` module to build a highly flexible, parameter-driven CLI framework. The upgraded script `hyperparameter_tuning.py` automatically evaluates an expanded pool of algorithms including **Decision Trees, Random Forests, and XGBoost**, governed by key functional switches:
        *   `--full`: Toggles execution between the 20% stratified prototype split and the full 100% dataset matrix.
        *   `--seed`: Sets the base random state synchronization seed (defaults to 1).
        *   `--pre_release`: Automatically strips out post-launch metrics (reviews, Metacritic counts, sentiment scores) to train a specialized predictive model focused purely on pre-launch indicators.
    *   **Why we chose this option:** This design decouples our core algorithmic execution from hardcoded configurations. The script automatically serializes a comprehensive `json` artifact containing cross-validation scores, the confusion matrix for each algorithm, ideal hyperparameter pairs, and selected features. It also exports automated **boxplots** of cross-validation results to streamline model interpretation.
*   **11. Boundary-Cleaning Undersampling for Extreme Long-Tail Imbalance:**
    *   **The Problem:** Class 0 (The Indie Long-Tail) represents an overwhelming 69% majority of the initial dataset. Leaving this distribution untouched forces estimators to optimize globally for Class 0 accuracy while ignoring rarer classes, while standard random undersampling risks discarding valuable underlying data patterns.
    *   **The Choice:** Implementation of a targeted, non-random spatial undersampling routine specifically on Class 0 data points within the tuning pipeline. This operation shrinks the massive Class 0 cohort by a factor of 4 (reducing it from 18,971 instances down to roughly 5,000 instances). Instead of picking records entirely at random, the algorithm utilizes a boundary-cleaning function to clarify feature spaces.
    *   **Why we chose this option:** Pruning elements randomly can strip away vital data characteristics from borderline cases. This non-random function cleans overlapping instances near cross-class boundaries, establishing clear spatial splits between target groups in the feature space. This reduction significantly accelerated cross-validation training speeds, allowing the pipeline to iterate smoothly through multi-model grids without stalling computational resources.
*   **12. Strategic Model Bifurcation: Pre-Launch vs. Post-Launch Architectures:**
    *   **The Problem:** Predictive features possess fundamentally different lifecycles. A developer needs to forecast market success *before* publishing a game, meaning they cannot use post-launch metrics like critic scores or user reviews. Conversely, a post-launch analysis requires all available data to isolate long-term player retention factors.
    *   **The Choice:** Splitting machine learning objectives into two separate functional pipelines governed by the custom `--pre_release` CLI switch:
        *   *Pre-Launch Pipeline:* Restricted exclusively to upstream features available before launch (system specs, hardware RAM minimums, genre configurations, text descriptions). Operates as a decision-support tool for game developers.
        *   *Post-Launch Pipeline:* Utilizes the comprehensive dataset, including downstream interaction features (Metacritic scores, user review counts, average playtime metrics) to map long-tail retention signatures.
    *   **Why we chose this option:** Training a single model with all features introduces data leakage for pre-launch use cases. Splitting the models ensures our tools are practically useful for developers during early production stages while still enabling deep post-launch market analysis.
*   **13. Feature Constraints and Class Confusion Analysis:**
    *   **The Problem:** Initial baseline runs on the 20% stratified prototype split revealed high predictive accuracy for lowest tiers, but notable confusion and misclassification within the highest success bands (Classes 3 and 4), as documented in the `tuning_results.json` log.
    *   **The Diagnostic Analysis:** A review of the feature matrix showed that technical specifications (like required gigabytes of RAM) and text metadata are highly effective for rough baseline classification but lack the granularity needed to distinguish top-tier blockbusters from moderate successes. Real-world market success is driven by external variables outside the scope of raw software files, such as marketing budgets, advertising campaigns, and organic streaming/influencer traction.
    *   **Why we chose this option:** Rather than artificially forcing models to find signal in noisy technical features, we logged this structural limitation. This behavior supported our transition to XGBoost — which handles these non-linear feature interactions better than simple models — and justified the future introduction of post-launch metrics to clear up top-tier confusion.
*   **14. Empirical Model Sweeps and Transition to Black-Box XAI:**
    *   **The Problem:** Selecting the final model family requires objective, data-driven proof across all cross-validation folds, especially when moving from highly interpretable white-box models (Decision Trees) to complex black-box ones (XGBoost).
    *   **The Experiment:** Execution of an overnight multi-model nested cross-validation loop on the dedicated processing infrastructure. The tuning process automatically outputs a detailed master JSON file tracking evaluation metrics, selected features, and historical confusion matrices across all 5 outer splits. Logs indicated that XGBoost achieves a dominant Macro-F1 score of approximately **$0.60$** across the 5-class distribution, outperforming simpler models by a $+10\%$ margin.
    *   **Why we chose this option:** While white-box Decision Trees offer native interpretability, XGBoost’s massive performance boost made it our clear choice for deployment. Because XGBoost operates as a complex black-box ensemble, this decision directly set up the next step: integrating **Post-Hoc Explainable AI (XAI)** frameworks (like SHAP or LIME) to extract feature importance and keep the model transparent.
*   **15. Target Macro-Class Compression Proposal for Pre-Release Predictors:**
    *   **The Problem:** Because pre-release models must discard highly predictive post-launch interaction data (like user reviews), forcing them to classify across 5 highly imbalanced groups accelerates performance degradation.
    *   **The Choice:** Proposal of an experimental path to collapse our 5 ordinal target tiers into **3 macro-balanced semantic categories**: `flop`, `middle`, and `success`.
    *   **Why we chose this option:** Compressing targets reduces class sparsity and stabilizes model decision boundaries. This structural adjustment helps compensate for the loss of post-launch features, preserving stable Macro-F1 performance and providing developers with clear, reliable target groups.
*   **16. Statistical Model Selection and Boxplot Validation:**
    *   **The Problem:** Selecting the final model architecture demands rigorous validation across all outer folds to ensure a model's superior performance is statistically stable and not an artifact of favorable data splits.
    *   **The Experiment:** Plotting the distribution of performance metrics across the outer folds of the nested cross-validation. The visualization definitively established **XGBoost** as our top-performing model family across all four audited dimensions:
        *   *Accuracy:* XGBoost achieves a stable baseline between $0.75$ and $0.77$, outperforming Random Forest ($\approx 0.743 - 0.755$) and base Decision Trees ($\approx 0.69 - 0,71$).
        *   *Precision (Macro):* XGBoost leads within a tight $0.575 - 0.61$ range.
        *   *Recall (Macro):* XGBoost tightly clusters around $0.615 - 0.625$.
        *   *F1 (Macro):* XGBoost dominates with a median score of roughly **$0.60$**, outperforming Random Forest (median $\approx 0.56$) and leaving base Decision Trees far behind (median $\approx 0.48$).
    *   **Why we chose this option:** Boxplot distributions confirm that XGBoost delivers superior macro-F1 values combined with minimal variance across folds. This statistical proof justifies using this black-box model as our primary architecture, confirming our earlier observation that technical data limits performance to the $0.60 - 0.70$ macro-F1 bracket due to unmodeled external market forces (like marketing spend and viral traction).
*   **17. Deployment of Dual Post-Hoc Explainable AI (XAI) Frameworks:**
    *   **The Problem:** Because XGBoost operates as a highly complex non-linear ensemble, it acts as a black box. To make our predictions trustworthy for developers and stakeholders, we need to extract human-readable explanations of why the model flags specific games as hits or flops.
    *   **The Choice:** Integration of an automated explainability suite into `train_pipeline.py`. As soon as the pipeline fits optimal XGBoost parameters, it automatically computes and serializes global and local post-hoc explanations:
        *   *SHAP (SHapley Additive exPlanations):* Computes game-theory attribution values to map global feature importance across the entire dataset.
        *   *LIME (Local Interpretable Model-agnostic Explanations):* Generates local perturbations around individual sample points to explain specific predictions on a case-by-case basis.
    *   **Why we chose this option:** Combining SHAP and LIME provides a comprehensive explanation framework. SHAP offers a mathematically sound view of overall feature importance across the whole pipeline, while LIME allows developers to inspect individual games and see exactly how minor tweaks to pre-launch properties might alter market success forecasts.
*   **18. Pre-Launch Pipeline Sweep Execution and Scalability:**
    *   **The Problem:** Validating the developer-focused pre-launch model requires running our full multi-model nested cross-validation loop on a restricted feature set, which demands a long, uninterrupted processing window.
    *   **The Choice:** Configuring execution parameters to run our full training pipeline on the high-performance workstation using the `-p` or `--pre_release` CLI flag. This automatically strips post-launch variables and routes remaining data through the 14-hour optimization loop.
    *   **Why we chose this option:** Our modular CLI design allows us to run the massive pre-launch experiment without making manual code modifications. The script automatically handles feature restrictions and exports an independent results JSON, keeping the experiment reproducible as we evaluate whether compressing targets into 3 macro-classes compensates for the loss of post-launch features.

---

## SECTION 2: System Architecture & Production Roadmap

This section details the final operational blueprint of the repository ecosystem, consolidating file layouts, implemented algorithmic strategies, and task checklists extracted from reference discussions.

### 2.1. Pipeline Layout and Repository Structure

The repository is systematically organized to ensure complete isolation of data splits and information-leak-free pipelines:

```
HitPredictor-Steam/
├── dataset/

│   ├── raw_data/          # Compressed raw zip assets (steam_app_data.zip, steamspy_data.zip)
│   └── clean_data/        # Deterministic pipeline outputs (clean_dataset.csv, clean_dataset.zip)
├── src/                   # Core Python production modules and utilities
│   ├── andrea_stuff/      # Custom experimental sandbox (tracked under .gitignore)
│   ├── pedro_stuff/       # Custom experimental sandbox (tracked under .gitignore)
│   ├── data_cleaning.py   # Master ingestion pipeline executing core filters
│   ├── data_cleaning_utils.py # Modular heuristics, text scrubbers, and hardware extractors
│   ├── data_preprocessing.py # Scikit-learn compliant custom estimators and transformers
│   ├── unified_search.py  # Optimized bus coupling feature extraction and estimators
│   ├── hyperparameter_tuning.py # Nested Cross-Validation Grid Sweeper (5-Outer, 3-Inner)
│   ├── train_pipeline.py  # Production pipeline executing final training and XAI logging
│   └── analyze_results.py # Non-parametric statistical validation tests (Friedman/Wilcoxon)
├── .gitattributes
├── .gitignore             # Configured to exclude heavy binaries, pickled models, and caches
├── docker-compose.yml     # WSL/Docker environment encapsulation for unified execution
├── LICENSE
├── README.md
├── Reference.md           # Master architectural project operational guide
└── requirements.txt       # Frozen and pinned production dependency versions
```

---

### 2.2. Algorithmic Paradigms and Implemented Core Strategies

#### A. Imbalance Handling via Spatial Boundary Pruning (Tomek Links)
The dataset suffers from a hyper-skewed power-law distribution. The production pipeline handles this dynamically within each cross-validation fold via a staged approach:
1.  **Ordinal Re-Binning:** Consolidation of the original 13 SteamSpy ownership classes into 5 macroscopically viable ordinal tiers:
    *   *Class 0:* Indie Long-Tail (0 .. 20,000 copies)
    *   *Class 1:* Healthy Niche (20.000 .. 100.000 copies)
    *   *Class 2:* Mid-Tier Success (100.000 .. 500.000 copies)
    *   *Class 3:* Major Success / Viral AA (500.000 .. 2.000.000 copies)
    *   *Class 4:* Mega-Hit / Blockbuster AAA (2.000.000 .. 200M+ copies)
2.  **Dynamic Majority Undersampling:** Class 0 is automatically capped to 25% of its original fold volume using an adaptive ratio calculator.
3.  **Tomek Links Cleaning:** Plugged in immediately post-undersampling to discard ambiguous, overlapping instances sitting right on the decision boundaries of ordinal classes.

#### B. Isolated Feature Engineering and Secure Ingestion
*   **Custom Scikit-Learn Transformers:** Development of isolated estimators (`SteamFeatureExtractor`, `CorrelationRemover`) to safely manage categorical `MultiLabelBinarizer` and strip out highly collinear metrics ($r > 0.95$) strictly inside active cross-validation splits.
*   **Hybrid Text Vectorization:** Short descriptions are processed via traditional lexical frameworks (`TF-IDF`). Dense, detailed descriptions are converted into deep learning embeddings using the `SentenceTransformer('all-mpnet-base-v2')` architecture, controlled via Principal Component Analysis (`PCA`) components to guarantee uniform feature spacing.
*   **Feature Name Sanitization:** Integration of a dedicated regex routine (`FeatureNameSanitizer`) to scan and strip out bracket notations and symbols (`[`, `]`, `<`) from feature column headers, preventing internal breaks across gradient boosting frameworks.
*   **Strict Global Hygiene:** Cleaning routines discard non-Western script records across core text dimensions via strict Unicode block mapping to eliminate embedding vocabulary noise.

#### C. Validation Framework and XAI
The pipeline rejects simplistic accuracy-driven validation in favor of statistically structured precision-recall metrics and Wilcoxon/Friedman post-hoc tests for mathematical model superiority verification. Interpretability is guaranteed by the automatic export of global **SHAP** value summary plots and local HTML perturbation summaries via **LIME**.

---

### 2.3. Future Action Checklist (Production Roadmap)

#### Completed Milestones (Validated via Evolution History)
*   [x] **CLI-Driven Model Bifurcation (Phase 2):** Full implementation of the `--pre_release` flag to purge downstream indicators (like reviews and Metacritic scores), isolating the developer-centric pipeline.
*   [x] **Split-First Design (Phase 2):** Structural fix against computational redundancies and data leakage, ensuring transformers operate strictly within inner cross-validation folds.
*   [x] **Grid Search Circular Dependency Resolution (Phase 3):** Completion of the `unified_search.py` module, simultaneously optimizing and evaluating feature extraction parameters and core classifier hyperparameters.
*   [x] **Infrastructure & Resource Provisioning (Phase 3):** Successful migration to a dedicated high-performance VM in a commercial datacenter, mitigating local thermal throttling issues.
*   [x] **Dynamic Multi-Model CLI Orchestration (Phase 3):** Integration of `argparse` managing automated runs (`--full`, `--seed`, `--pre_release`) for Decision Trees, Random Forest, and XGBoost.
*   [x] **Statistical Mapping & Production XAI Suite (Phase 3):** Consolidation of outer fold performance boxplots and coupling post-hoc **SHAP** and **LIME** interpreters to the champion XGBoost model.

#### Pending Tasks

##### Phase 1: Code Quality, Styling & Artifact Serialization
*   [ ] **Audit and Lean Down Codebase:** Scan `data_preprocessing.py`, `hyperparameter_tuning.py`, `train_pipeline.py`, and `analyze_results.py` modules to strip out unused library imports, dead dependencies, and obsolete diagnostic tracking prints.
*   [ ] **Standardize Style & Commenting:** Refactor the repository for strict PEP 8 alignment, ensure clear and consistent variable naming, and enrich function docstrings for academic evaluation.
*   [ ] **Automate Component Pickling:** Update the `train_pipeline.py` script to pickle its entire preprocessing pipeline — including fitted custom transformers (`SteamFeatureExtractor`, `FeatureNameSanitizer`, `CorrelationRemover`) and the `scaler_and_pca` step — along with the trained XGBoost model as a cohesive production artifact.
*   [ ] **Establish Artifact Versioning:** Add an automated timestamp or git-hash naming convention to output `.pkl` or `.joblib` files to prevent subsequent training runs from silently overwriting ideal models.
*   [ ] **Refactor ZIP Compression for Dataset Storage:** Modify `data_cleaning.py` to write out `clean_dataset.csv` directly as a compressed `.zip` archive to optimize container disk space and I/O. Adjust downstream reading scripts to ensure Pandas processes the compressed file transparently.

##### Phase 2: Target Synchronization and Final Mapping
*   [ ] **Synchronize Target Compression in Production:** Update the `data_cleaning_utils.py` file with the final dictionary to materialize the proposed 3 ordinal macro-class mapping (`flop`, `middle`, `success`), validating whether clustering reduces error rates observed in old brackets 2 and 3 of the tuning log.

##### Phase 3: Large-Scale Execution
*   [ ] **Run Full-Scale Sweeps (100% Dataset):** Fire up the final production sweep using the production flags over the full historical game base on the dedicated VM, harvesting final production macro-F1 coefficients.

##### Phase 4: Inference Layer & Robustness (predict.py)
*   [ ] **Build Single-Instance Inference Module:** Create a clean, standalone execution script (`predict.py`) that loads serialized pipeline artifacts and accepts a single-game query in JSON or CSV format.
*   [ ] **Port Pre-Inference Guards:** Embed the string parsing routines, missing feature imputation handlers, and non-Western Unicode character filters directly into the inference script to alert users gracefully instead of throwing runtime crashes.
*   [ ] **Integrate Pre/Post-Launch Inference Logic:** Ensure `predict.py` mirrors CLI flag behavior to dynamically drop post-launch features and restructure input shapes depending on the loaded model.

##### Phase 5: Academic Deliverables & Submission Space
*   [ ] **Draft Technical Paper Report (.pdf):** Compile the final scientific research paper covering the data hygiene pipeline, nested validation strategy, statistical results, and interpretability findings, exporting to PDF according to the UniPi curriculum.
*   [ ] **Design Presentation Slides (.pptx):** Build a structured slide deck providing a polished technical walkthrough of the pipeline architecture, data leak controls, and engineering milestones.
*   [ ] **Record Collaborative Video Discussion:** Schedule the team video session to record the joint project presentation and code walkthrough as per university guidelines.