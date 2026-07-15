# ARCHITECTURE & ROADMAP: HitPredictor-Steam

This file serves as our master operational architectural blueprint. This project focuses on predicting a game's commercial and adoption trajectory ("hit status") using historical Steam store data, player trends, user reviews, hardware constraints, and pricing configurations.

---

## 🛠️ Project Architecture & Pipeline Layout

To satisfy the strict technical and organizational standards of the **UniPi Data Mining (12 CFU)** curriculum, our repository is systematically organized. Based on our final production workspace, the current structural tree layout is as follows:

```
HitPredictor-Steam/
├── dataset/

│   ├── raw_data/          # Input compressed zip archives (steam_app_data.zip, steamspy_data.zip).
│   └── clean_data/        # Deterministic pipeline outputs (clean_dataset.csv).
├── src/                   # Core Python application modules and production utilities.
│   ├── andrea_stuff/      # Custom experimental sandbox (in .gitignore) for testing.
│   ├── pedro_stuff/       # Custom experimental sandbox (in .gitignore) for testing.
│   ├── data_cleaning.py   # Baseline execution pipeline orchestrating core ingestion filters.
│   ├── data_cleaning_utils.py # Modular parsing heuristics, text sanitizers, and hardware extractors.
│   ├── data_preprocessing.py # Custom scikit-learn compatible fit/transform estimators and sanitizers.
│   ├── hyperparameter_tuning.py # Multi-model Nested Cross-Validation (5-Outer, 3-Inner) grid sweeper.
│   ├── train_pipeline.py  # Production training pipeline executing global/local XAI logging.
│   └── analyze_results.py # Non-parametric statistical validation testing (Friedman/Wilcoxon).
├── .gitattributes
├── .gitignore             # Formatted to exclude heavy raw assets, pickled models, and cache pools.
├── docker-compose.yml     # WSL/Docker environment encapsulation for unified execution.
├── LICENSE
├── README.md
├── Reference.md           # This very file. Project Northstar.
└── requirements.txt       # Frozen pinned production library dependency versions.
```

---

## ⚖️ Core Algorithmic Strategies & Paradigms

### 1. Power-Law Imbalance Handling & Boundary Resampling
Steam ownership data follows a hyper-skewed power-law distribution. To prevent model collapse or extreme majority-class bias, the original 13 string-based ranges from SteamSpy are consolidated into 5 macro-balanced, semantically meaningful ordinal tiers:

| Original Class | SteamSpy Ownership Range | New Aggregated Tier | Target Meaning |
| :--- | :--- | :--- | :--- |
| **0** | 0 .. 20,000 | **0** | **The Indie Long-Tail** (Low adoption) |
| **1 - 2** | 20,000 .. 100,000 | **1** | **Healthy Niche** (Sustainable indie) |
| **3 - 4** | 100,000 .. 500,000 | **2** | **Mid-Market Success** (Breakout hits) |
| **5 - 6** | 500,000 .. 2,000,000 | **3** | **Major Success** (AA level / Viral hits) |
| **7 - 12** | 2,000,000 .. 200M+ | **4** | **Mega-Hit / Blockbuster** (AAA status) |

*Advanced Resampling Suite:* Within each training fold of our cross-validation framework, class balancing is dynamically achieved via a multi-stage pipeline:
1. **Dynamic Majority Undersampling:** Class 0 is automatically clamped to 25% of its original split volume using an adaptive ratio calculator.
2. **Tomek Links Cleaning:** Chained immediately post-sampling to drop ambiguous, overlapping instances directly on class boundaries.

### 2. Dual Domain Scoping (Developer vs. Market Analyst)
To avoid architectural data leakage (target peeking) and demonstrate advanced domain awareness, our execution loops are bifurcated into independent evaluation frameworks via terminal flags (`-p` / `--pre_release`):
* **Pre-Launch Mode (The Developer's Decision Matrix):** Simulates an unreleased game. Models are trained *exclusively* on foundational features available prior to launch (`price`, `genres`, `categories`, `tags`, `languages`, `top_publishers/developers`, and engineered hardware continuous metrics `min_ram_gb`/`rec_ram_gb`, alongside text embeddings from `short_description`). Post-launch dynamics (`review_ratio`, `metacritic_score`, `num_achievements`, playtime) are strictly omitted.
* **Post-Launch Mode (The Market Analyst's Tool):** Utilizes the complete consolidated feature set to predict long-term ownership growth and isolate driving player-retention metrics.

### 3. Preprocessing, Feature Ingestion & Code Safety
* **Custom Scikit-Learn Transformers:** Built isolated custom estimators (`SteamFeatureExtractor`, `CorrelationRemover`) to safely manage categorical MultiLabelBinarization and remove highly collinear metrics ($r > 0.95$) strictly within active cross-validation splits.
* **Hybrid Text Layout:** Short text descriptions are vectorized via traditional lexical `TF-IDF` features. Dense detailed descriptions are embedded using deep learning architectures (`SentenceTransformer('all-mpnet-base-v2')`) compressed through PCA components to guarantee uniform spacing.
* **Feature Name Sanitation:** Integrated a dedicated regex sanitizer (`FeatureNameSanitizer`) to sweep and remove bracket notations (`[`, `]`, `<`) from feature columns, preventing crashes within gradient boosting frameworks.
* **Advanced Global Hygiene:** Data cleaning routines actively drop non-Western string records across major text dimensions via strict Unicode block mapping to eliminate embedding vocabulary noise.

### 4. Explainable AI (XAI) & Non-Parametric Validation
To align with EU Responsible AI frameworks emphasized by UniPi, model validation bypasses simple accuracy metrics in favor of stratified precision-recall analysis. Post-modeling features are evaluated globally and locally using **SHAP** and **LIME** frameworks to isolate exactly which combination of features drives a game over a target milestone.

---

## 📈 Project Status & Checkpoint

* [x] **Data Ingestion & Cleaning Pipeline:** Complete. Code modularized into `data_cleaning.py` and `data_cleaning_utils.py`, reading zipped binary data streams with no local uncompressed footprints.
* [x] **Advanced Preprocessing Isolation:** Complete. Built `data_preprocessing.py` containing custom Transformers to prevent multi-label or embedding feature target leaks across folds.
* [x] **Multi-Model Tuning Scaffolding:** Complete. Engineered `hyperparameter_tuning.py` to run nested cross-validation across Decision Trees, Random Forests, and Weighted XGBoost engines.
* [x] **Statistical Post-Hoc Verification:** Complete. Implemented `analyze_results.py` executing Friedman global checks and Wilcoxon signed-rank tests to mathematically verify the winning model family.
* [x] **Twin Explainable AI Frameworks:** Complete. Production script `train_pipeline.py` automatically exports global SHAP value mappings and local HTML-based LIME perturbation summaries.

### To Do:
1. Code Quality, Style & Artifact Serialization
[ ] Audit and Prune Codebase: Scan data_preprocessing.py, hyperparameter_tuning.py, train_pipeline.py, and analyze_results.py to remove unused library imports, dead dependencies, and obsolete diagnostic tracking prints or comment blocks.

[ ] Standardize Style & Comments: Refactor the repository to strict PEP 8 alignment, ensure clear and consistent variable naming, and enrich function docstrings and inline descriptions for academic evaluation.

[ ] Automate Component Pickling: Update train_pipeline.py to pickle your entire preprocessing pipeline—including custom fitted transformers (SteamFeatureExtractor, FeatureNameSanitizer, CorrelationRemover) and your scaler_and_pca step—alongside the trained XGBoost model as a cohesive production artifact.

[ ] Establish Version Control for Artifacts: Add an automated timestamping or git-hash naming convention to the output .pkl or .joblib files to prevent subsequent training runs from silently overwriting your optimal models.

2. Architecture Audit & Pre-Release Target Compaction
[ ] Analyze Tuning Confusion Matrices: Open the experimental nested cross-validation JSON outputs in the result/ folder and evaluate the exact misclassification rates and overlap thresholds between Class 2 and Class 3.

[ ] Determine Pre-Release Class Sizing: Based on the matrix audit, decide whether to keep the pre-release mode at 5 classes or compress it into 3 macro-classes (flop, middle, success) to account for the missing post-launch feature capacity.

[ ] Synchronize Data Cleaning Target Binning: If you collapse the classes to 3 for the pre-release run, ensure that your target mapping dictionary in data_cleaning_utils.py updates cleanly to handle the new boundaries without breaking the workflow.

3. Infrastructure Lock-In & Server-Side Execution
[ ] Freeze Dependency Environments: Verify that requirements.txt freezes strict, explicit versions of key libraries (e.g., xgboost==x.y.z, shap==x.y.z) to ensure your custom estimators and GPU CUDA operations don't break with future updates.

[ ] Validate the Docker Build Loop: Run a clean docker compose down --volumes followed by a fresh up-build to ensure the containerized stack compiles flawlessly from a zero-cache state on your WSL setup.

[ ] Execute 14-Hour Multi-Model Training Loop: Pull the optimized scripts into the high-performance workstation and kick off the complete nested cross-validation hyperparameter sweep with full configuration flags (-f / --full and -p / --pre_release).

[ ] Run Statistical & XAI Pipelines: Run analyze_results.py on the resulting JSON to compute Friedman and Wilcoxon signed-rank tests, and execute the final XAI routines to generate your SHAP summary charts, local HTML LIME explanations, and cross-validation boxplots.

4. Inference Layer & Production Robustness (predict.py)
[ ] Build Single-Instance Inference Module: Create a standalone, clean predict.py execution script that loads your serialized pipeline artifacts and accepts a single raw game's JSON or CSV query.

[ ] Port Ingestion & Pre-Inference Guards: Integrate your string-parsing logic, missing feature imputation handlers, and non-Western Unicode character filters directly into the inference script to gracefully warn users or handle anomalies instead of throwing runtime crashes.

[ ] Integrate Pre/Post-Release Logic into Inference: Ensure predict.py mirrors the -p / --pre_release CLI flag structure so it dynamically drops post-launch features and restructures input shapes depending on the chosen model mode.

5. Academic Deliverables & Submission Space
[ ] Draft the Technical Project Report (.pdf): Compile your final comprehensive research paper covering your data hygiene pipeline, nested validation strategy, statistical results, and interpretability findings, exporting it as a PDF document matching the professor's strict syllabus requirements.

[ ] Design the Presentation Deck (.pptx): Construct a highly structured PowerPoint presentation providing a polished technical walkthrough of your pipeline architecture, data leakage controls, and engineering milestones.

[ ] Record Collaborative Video Discussion: Schedule your team video session, utilizing screen sharing to record your joint project presentation and code walkthrough according to University guidelines.

---

# Added on 2026-07-15:

## 🚀 Progressive Phase Checklist

### ⏹️ Phase 1: Code Quality, Style & Serialization
* [ ] **Audit and Prune Codebase:** Scan application modules to remove unused library imports, dead dependencies, and obsolete tracking prints.
* [ ] **Standardize Style & Comments:** Bring core scripts into full PEP 8 alignment and enrich function docstrings for academic review.
* [ ] **Automate Component Pickling:** Update the pipeline to serialize custom fitted transformers (`SteamFeatureExtractor`, `FeatureNameSanitizer`, `CorrelationRemover`) alongside the scalar step into a single production `.pkl` or `.joblib` artifact.
* [ ] **Establish Version Control for Artifacts:** Integrate automated timestamping/git-hash naming conventions to avoid overwriting optimal weights.
* [ ] **Analyse ZIP Compression for Dataset Storage:** Refactor `data_cleaning.py` to write `clean_dataset.csv` directly as a `.zip` file to save disk space and optimize I/O. Test downstream read scripts to verify that Pandas transparently parses the compressed archive with zero syntax changes beyond the file extension update.

### ⏹️ Phase 2: Dual-Mode Validation & Pipeline Hardening
* [ ] **Enforce Split-First Paradigm:** Ensure `data_preprocessing.py` strictly isolates training splits before executing TF-IDF vectorization, supervised dimensionality reduction (PLS-DA), scaling, or resampling to eliminate risk of data leakage.
* [ ] **Harmonize Pre-Launch Model Bounds:** Verify that the pre-launch pipeline is fully configured to purge all post-release signals (`ccu`, `metacritic_score`, `review_ratio`, etc.) from the training and test matrices dynamically.
* [ ] **Integrate Regular Expression Name Sanitization:** Build structural string sanitization routines to automatically clean special characters and brackets (such as `[`, `]`, `<`) from feature header names to secure model pipeline training.

### 💻 Phase 3: Infrastructure Lock-In & Production Computation
* [ ] **Analyze Grid Search & Hyperparameter Tuning Circular Dependency:** Investigate refactoring the optimization scripts to evaluate the feature extraction grid parameters (`PLS-DA` components, `TF-IDF` max features) and model hyperparameters (`max_depth`, `n_estimators`) **simultaneously** inside a single, unified search space. This eliminates the circular dependency where feature-space selection is evaluated on un-optimized basic models, ensuring maximum scientific rigor.
* [ ] **Implement Unified Pipeline Code Alternative:** Test feasibility of executing `unified_search.py` on our active development environments to compare computational runtime against current sequential tuning results.
* [ ] **Execute Long-Running 14-Hour Optimization Sweeps:** Launch the complete nested cross-validation runs on the server utilizing PLS-DA to finalize our optimal XGBoost weights.