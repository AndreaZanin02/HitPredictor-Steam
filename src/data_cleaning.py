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
1. Load and merge Steam Store and SteamSpy datasets
2. Keep only rows representing actual games
3. Create the target variable from SteamSpy ownership ranges
4. Extract useful features from JSON-like columns
5. Engineer hardware requirement features (RAM, CPU, GPU)
6. Engineer financial and temporal features
7. Clean text fields
8. Remove redundant, noisy, and target-leaking columns
9. Export the final dataset

Design Principles
-----------------
- Preserve information useful for popularity prediction
- Remove features that directly leak the target
- Convert nested JSON/string structures into ML-friendly features
- Keep transformations deterministic and reproducible
- Avoid manual intervention during preprocessing

Expected Output
---------------
A CSV dataset where:
- Numeric features are ready for modeling
- Categorical list features are normalized
- Text fields are cleaned
- Target variable is encoded as an integer class
"""

# Import helper utilities
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
    advanced_quality_filtering,
    reorder_and_rename_columns,
    clean_and_export
)


# ================================================================================================================
# PIPELINE EXECUTION ORDER
# ================================================================================================================
#
# Raw Steam Store Data ---+
#                         |--> Merge Datasets --> Keep Only Games --> Create Target (target_owners) ---+
# Raw SteamSpy Data ------+                                                                            |
#                                                                                                      v
#      Extract CPU/GPU Features <- Extract RAM Features <- Extract Language Features <- Extract JSON Features
#                |
#                v
#      Extract Financial/Temporal Features --> Extract Restrictions Features --> Clean Text Descriptions ----+
#                                                                                                            |
#                                                                                                            v
#           (clean_dataset.csv) <- Final Cleanup (clean_and_export) <- Reorder/Rename <- Advanced Quality Filter
# ================================================================================================================


if __name__ == "__main__":

    # Raw input paths (zipped CSVs, read directly by pandas via load_and_merge)
    FILE_STEAM_STORE = "../dataset/raw_data/steam_app_data.zip"
    FILE_STEAM_SPY = "../dataset/raw_data/steamspy_data.zip"
    
    # Merge Steam Store metadata with SteamSpy stats on steam_appid/appid
    # Rows that got shifted/misaligned by malformed CSV lines are dropped here
    df_merged = load_and_merge(FILE_STEAM_STORE, FILE_STEAM_SPY)

    # Keep only rows where type == 'game' (drop DLC/software/videos/etc.)
    # and engineer basic flags (is_controller_supported, num_dlc)
    df_filtered = process_game_features(df_merged)

    # Build the prediction target by mapping SteamSpy 'owners' ranges
    # into 5 ordinal ownership tiers (0 = long tail, 4 = hit/AAA)
    df_target = process_target_owners(df_filtered)

    # Flatten JSON-like columns (platforms, metacritic, achievements)
    # and normalize categories/genres/tags/publishers/developers into token lists
    df_json = extract_json_features(df_target)

    # Parse supported languages into a token list and derive
    # num_languages_supported
    df_lang = extract_language_features(df_json)

    # Parse minimum/recommended system requirements text to extract
    # min_ram_gb and rec_ram_gb
    df_ram = extract_ram_features(df_lang)

    # Derive binary GPU/CPU requirement flags via keyword matching
    # on the requirements text
    df_hw = extract_gpu_cpu_features(df_ram)

    # Compute price from initialprice, extract release_year/month,
    # drop unreleased/future-dated titles, and compute review_ratio
    df_fin_temp = extract_financial_and_temporal(df_hw)

    # Convert DRM notice and external account requirement
    # into binary flags (has_third_party_drm, requires_ext_account)
    df_res = extract_restrictions_features(df_fin_temp)

    # Strip HTML tags and normalize whitespace in name/description fields
    df_clean_text = clean_text_descriptions(df_res)

    # Drop rows with missing name/description, remove entries containing
    # CJK/Cyrillic text, and clean/impute remaining numeric quality columns
    df_advanced = advanced_quality_filtering(df_clean_text)

    # Rename name_store -> name and reorder columns into logical
    # groups, keeping target_owners as the last column to avoid leakage
    df_reordered = reorder_and_rename_columns(df_advanced)

    # Drop IDs, redundant/raw columns and features that leak the
    # target (ccu, positive, negative), then export the ML-ready CSV
    df_final = clean_and_export(df_reordered, output_filename="../dataset/clean_data/clean_dataset.csv")