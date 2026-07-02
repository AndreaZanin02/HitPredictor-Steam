# ARCHITECTURE & ROADMAP: HitPredictor-Steam

This file serves as our operational architectural blueprint. This project focuses on predicting a game's commercial and adoption trajectory ("hit status") using historical Steam store data, player trends, user reviews, and pricing strategies.

---

## 🛠️ Project Architecture & Pipeline Layout

To satisfy the strict technical and organizational standards of the **UniPi Data Mining (12 CFU)** curriculum, our repository is systematically organized. Based on our workspace profile, the current structural tree layout is as follows:

```
HitPredictor-Steam/
├── dataset/                  
│   ├── raw_data/          # Initial raw CSV dumps (steam_app_data.csv, steamspy_data.csv).
│   └── clean_data/        # Deterministic pipeline outputs (clean_dataset.csv).
├── src/                   # Core Python application modules and workspace tools.
│   ├── andrea_stuff/      # Custom experimental sandbox (in .gitignore) for testing and validation.
│   ├── pedro_stuff/       # Custom experimental sandbox (in .gitignore) for testing and validation.
│   ├── data_cleaning.py   # Baseline feature extraction, filtering, and data hygiene.
│   └── data_preprocessing.py # Fit/Transform pipelines for multi-label text and categories.
├── notebooks/             # [Planned Direction] Dedicated directory for sequential .ipynb workflows
├── .gitattributes
├── .gitignore             # Formatted to exclude heavy raw JSON/CSV assets and cache pools.
├── docker-compose.yml     # WSL/Docker environment encapsulation for unified execution.
├── LICENSE
├── README.md
├── Reference.md           # This very file. It is our project northstar. We'll keep updating.
└── requirements.txt
```

---

## ⚖️ Core Algorithmic Strategies & Paradigms

### 1. Power-Law Imbalance Handling (Target Re-Binning)
Steam ownership data follows a hyper-skewed power-law distribution, leaving some extreme high-end SteamSpy ranges with single-digit counts. To prevent model collapse or extreme majority-class bias, the original 13 string-based ranges from SteamSpy are consolidated into 5 macro-balanced, semantically meaningful ordinal tiers:

| Original Class | SteamSpy Ownership Range | New Aggregated Tier | Target Meaning |
| :--- | :--- | :--- | :--- |
| **0** | 0 .. 20,000 | **0** | **The Indie Long-Tail** (Low adoption) |
| **1 - 2** | 20,000 .. 100,000 | **1** | **Healthy Niche** (Sustainable indie) |
| **3 - 4** | 100,000 .. 500,000 | **2** | **Mid-Market Success** (Breakout hits) |
| **5 - 6** | 500,000 .. 2,000,000 | **3** | **Major Success** (AA level / Viral hits) |
| **7 - 12** | 2,000,000 .. 200M+ | **4** | **Mega-Hit / Blockbuster** (AAA status) |

### 2. Dual Domain Scoping (Developer vs. Market Analyst)
To avoid architectural data leakage (target peeking) and demonstrate advanced domain awareness, our models are bifurcated into two independent evaluation frameworks:
* **Pre-Launch Mode (The Developer's Decision Matrix):** Simulates an unreleased game. Models are trained *exclusively* on foundational features available prior to launch (`price`, `is_free`, `genres`, `categories`, `tags`, `languages`, `top_publishers/developers`, and text embeddings from `short_description`). Post-launch dynamics (`review_ratio`, `metacritic_score`, `num_achievements`) are strictly omitted.
* **Post-Launch Mode (The Market Analyst's Tool):** Utilizes the complete consolidated feature set to predict long-term ownership growth based on early-reception performance signals.

### 3. Dimensionality & Text Vectorization
* **Categorical Constraints:** High-cardinality multi-label features (such as user-submitted `tags`) are frequency-capped during the `fit` phase to prevent massive feature matrix sparsity.
* **Hybrid Text Layout:** Short text descriptions are vectorized using traditional lexical feature structures (`TF-IDF`). Dense, detailed descriptions are processed via semantic deep-learning architectures (`Sentence Transformers`), controlled through Principle Component Analysis (`PCA`) components to maintain optimal feature spacing.
* **Topological Projections:** Non-linear dimensionality reduction (**UMAP** or **$t$-SNE**) will be used within the exploratory notebook to verify if the engineered tiers occupy distinct topological spaces.

### 4. Explainable AI (XAI) & Responsible Mining
To align with EU Responsible AI frameworks emphasized by UniPi, model validation bypasses simple accuracy metrics in favor of stratified precision-recall analysis. Post-modeling features are evaluated globally and locally using **SHAP** and **LIME** frameworks to isolate exactly which combination of features drives a game over a target milestone.

---

## 📈 Project Status & Checkpoint

* [x] **Data Strategy & Cleaning Pipeline:** Custom cleaning scripts are established (`data_cleaning.py`), converting stringified list elements, standardizing text fields, filtering out non-game records, and vectorizing raw system requirement texts into deterministic values.
* [x] **Workspace Infrastructure Environment:** Verified functional setup running WSL and Docker Compose. Git repositories initialized with custom `.gitignore` controls.
* [ ] **Target Re-Binning Integration:** Transitioning the baseline dictionary mapping inside `data_cleaning.py` to compile the 5 macro-balanced cohorts.
* [ ] **Exploratory Data Analysis Log:** Populating the `eda_log/` to systematically capture data variance and initial target distribution behaviors.
