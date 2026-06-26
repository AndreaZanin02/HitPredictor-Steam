"""
Steam Dataset Cleaning and Feature Engineering Pipeline
=======================================================

Purpose
-------
This script is a pipeline to transform the raw datasets to a clean dataset useful for our model.

It transforms two raw Steam datasets:

1. steam_app_data.csv   (Steam Store metadata)
2. steamspy_data.csv    (SteamSpy popularity statistics)

into a single clean dataset suitable for machine learning.

The primary prediction target is:

    target_owners

which represents the SteamSpy ownership tier for each game.

Pipeline Overview
-----------------
1. Load and merge Steam Store and SteamSpy datasets.
2. Keep only rows representing actual games.
3. Create the target variable from SteamSpy ownership ranges.
4. Extract useful features from JSON-like columns.
5. Engineer hardware requirement features (RAM, CPU, GPU).
6. Engineer financial and temporal features.
7. Clean text fields.
8. Remove redundant, noisy, and target-leaking columns.
9. Export the final dataset.

Design Principles
-----------------
- Preserve information useful for popularity prediction.
- Remove features that directly leak the target.
- Convert nested JSON/string structures into ML-friendly features.
- Keep transformations deterministic and reproducible.
- Avoid manual intervention during preprocessing.

Expected Output
---------------
A CSV dataset where:
- Numeric features are ready for modeling.
- Categorical list features are normalized.
- Text fields are cleaned.
- Target variable is encoded as an integer class.
"""

# Imports
import pandas as pd
import ast
import re
import numpy as np
from datetime import datetime

# Import clean isolated helper utilities
from data_cleaning_utils import (
    load_and_merge,
    process_game_features,
    process_target_owners,
    extract_json_features,
    extract_language_features,
    extract_ram_features,
    extract_gpu_cpu_features,
    extract_financial_and_temporal,
    extract_restrictions_features,
    clean_text_descriptions,
    reorder_and_rename_columns,
    clean_and_export
)


# =============================================================================================
# PIPELINE EXECUTION ORDER
# =============================================================================================
#  
# (Raw Steam Store Data + Raw SteamSpy Data) -> Raw StreamSpy Data -> Merge Datasets -+
#                                                                                     |
#                                                                                     v
# Extract RAM Features <- Extract Language Features <- Extract JSON Features <- Keep Only Games
#          |
#          v
# Extract CPU/GPU Features -> Extract Financial Features -> Extract Restrictions -----+
#                                                                                     |
#                                                                                     v
#                     (clean_dataset.csv) <- Final Cleanup <- Reorder Columns <- Clean Text 
# ============================================================================================
if __name__ == "__main__":

    FILE_STEAM_STORE = "../dataset/raw_data/steam_app_data.csv"
    FILE_STEAM_SPY = "../dataset/raw_data/steamspy_data.csv"
    
    df_merged = load_and_merge(FILE_STEAM_STORE, FILE_STEAM_SPY)
    df_filtered = process_game_features(df_merged)
    df_target = process_target_owners(df_filtered)
    df_json = extract_json_features(df_target)
    df_lang = extract_language_features(df_json)
    df_ram = extract_ram_features(df_lang)
    df_hw = extract_gpu_cpu_features(df_ram)
    df_fin_temp = extract_financial_and_temporal(df_hw)
    df_res = extract_restrictions_features(df_fin_temp)
    df_clean_text = clean_text_descriptions(df_res)
    df_reordered = reorder_and_rename_columns(df_clean_text)
    df_final = clean_and_export(df_reordered, output_filename="../dataset/clean_data/clean_dataset.csv")
