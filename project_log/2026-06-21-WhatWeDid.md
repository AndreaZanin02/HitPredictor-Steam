# What We Did So Far

This log tracks our steps, milestones, and technical choices during the development of the **HitPredictor-Steam** project. It serves as our clear, simple-language record of how we got here and why we made specific engineering decisions.

---

## Environment & Project Infrastructure
* **What we did:** We set up our working environment inside a **WSL (Windows Subsystem for Linux)** subsystem and fully containerized it using **Docker Compose**. We also initialized our Git repository with structured `.gitignore` parameters to protect against tracking heavy data assets.
* **Why we chose this:** We chose a containerized Docker setup to ensure that both of us are developing on completely identical, isolated Python runtimes. This eliminates any "it works on my machine" library version conflicts and ensures reproducible code execution from day one.

## Baseline Data Hygiene & Parsing
* **What we did:** We created our initial data cleaning script (`data_cleaning.py`) to handle the raw data sources (*Steam Store Metadata* and *SteamSpy Stats*). The script filters our data to keep only actual games, discards unnecessary metadata (like support links and EULA descriptions), cleans HTML tags out of text descriptions, and runs regular expressions to pull structured numbers from system requirements (like Minimum and Recommended RAM in GB).
* **Why we chose this:** Raw store dumps contain intense structural noise. By removing entries that are not games (like DLCs or software) and stripping presentation clutter early, we protect our downstream machine learning models from wasting memory and processing time on uninformative features.

## Tackling the Target Paradox
* **What we did:** We evaluated our intended target variable—the SteamSpy ownership fields—and realized it was split into 13 high-variance ranges. Because Steam data follows a strict power-law curve, a tiny number of games are massive blockbusters while thousands of long-tail games have low adoption. One class even had only a single game in it. We decided to compress these 13 ranges down into **5 macro-balanced ordinal tiers** (Class 0 to Class 4).
* **Why we chose this:** Machine learning classifiers break or entirely ignore rare classes if the target distribution is heavily unbalanced. By re-binning the targets into meaningful, statistically viable macro-cohorts (from *Indie Long-Tail* to *Mega-Hit Blockbuster*), we ensure our algorithms have enough training instances per class to extract meaningful patterns.

## Designing to Mitigate Feature Leakage
* **What we did:** We addressed a critical domain scoping question: Is our model a tool for developers planning an unreleased game, or a post-launch analytics tool? We realized that including features like user review ratios or achievement totals to predict success *before* a game comes out creates a major data leakage (target peeking) flaw. We decided to split our preprocessing approach into two separate experimental lanes: **Pre-Launch Mode** and **Post-Launch Mode**.
* **Why we chose this:** A game cannot accumulate user reviews or metacritic scores before it launches. Separating our feature matrices into a Pre-Launch configuration (using only core properties like price, genre, publisher track record, and localized languages) ensures our simulation stays completely realistic for zero-day investment forecasting.
