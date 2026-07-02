#!/usr/bin/env python3
"""
HitPredictor-Steam: Production Inference Interface
==================================================
Author: Team HitPredictor
Date: 02/07/2026

This script serves as an isolated, end-to-end inference execution engine that 
ingests raw game parameters via a JSON query string, loads serialized preprocessing
pipelines and model artifacts to map features without data leakage, and predicts a
single game's commercial success macro-trajectory while respecting execution constraints.

1. HOW TO RUN (EXAMPLES):
   
   # Post-Launch Evaluation (Default Mode):
   python3 predict.py --query '{"name": "Cyberpunk 2077", "price": 59.99, "short_description": "An open-world action-adventure...", "genres": ["Action", "RPG"], "categories": ["Single-player"], "tags": ["Sci-fi", "Open World"], "ccu": 45000, "positive": 120000, "negative": 30000, "metacritic_score": 86, "review_ratio": 0.80}'

   # Pre-Launch Evaluation (The Developer's Decision Matrix):
   python3 predict.py --pre_release --query '{"name": "Indie Quest", "price": 14.99, "short_description": "A retro pixel-art dungeon crawler...", "genres": ["Indie", "Adventure"], "categories": ["Single-player"], "tags": ["Pixel Art", "Difficult"], "pc_requirements": "Minimum: 8 GB RAM, GTX 1060"}'

2. EXPECTED INPUT SPECIFICATION:
   The script expects a single-line, escape-quoted raw JSON string passed to the `-q`/`--query` parameter.
   
   Core Schema Fields:
   - name (str): The official game title.
   - price (float): Listed retail price in USD.
   - short_description (str): Lexical store summary snippet (processed via TF-IDF).
   - detailed_description (str, optional): Long-form text content (processed via Sentence Transformers).
   - genres (list of str): Primary genre tags (e.g., ["Action", "Indie"]).
   - categories (list of str): Technical feature brackets (e.g., ["Single-player", "Co-op"]).
   - tags (list of str): High-cardinality player tags.
   - pc_requirements (str, optional): Raw text specs used for RAM/GPU continuous proxy parsing.
   
   *Post-Launch Mode Only* Fields (Omitted automatically if `--pre_release` flag is active):
   - metacritic_score (int), review_ratio (float), positive (int), negative (int), ccu (int).

3. EXPECTED OUTPUT SPECIFICATION:
   All structural diagnostic workflows are piped to `sys.stderr` to keep `sys.stdout` clean 
   for pipeline orchestration parsing. 
   
   The primary standard output returns a clean text-based terminal card structured as follows:
   - Predicted Success Category : Semantic description of the model class mapping.
   - Model Classification Index : The discrete target ordinal tier integer [0 to 4].
   - Confidence Distribution    : An array of soft-max probability values tracking confidence across classes.

"""

import argparse
import sys
import os
import json
import re
import joblib
import pandas as pd
import numpy as np
import warnings

# Suppress underlying library warnings for clean stdout delivery
warnings.filterwarnings('ignore')

# Attempt to load custom pipeline modules from the current directory
try:
    from data_preprocessing import SteamFeatureExtractor, CorrelationRemover, FeatureNameSanitizer
    from data_cleaning_utils import parse_dict
except ImportError:
    print("[!] Execution Warning: Custom transformer files must be present in the python path.", 
          file=sys.stderr)


def apply_pre_inference_guards(raw_data: dict) -> dict:
    """
    Enforces deterministic validation rules on incoming JSON payloads
    to ensure character integrity and stability before feature vector matrix transformation.
    """
    # 1. Unicode Block Checking for Non-Western Alphabet Noise (CJK & Cyrillic script)
    text_check_pool = [
        str(raw_data.get('name', '')),
        str(raw_data.get('short_description', '')),
        str(raw_data.get('detailed_description', ''))
    ]
    combined_text = " ".join(text_check_pool)
    
    cjk_cyrillic_pattern = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff]')
    if cjk_cyrillic_pattern.search(combined_text):
        print("[!] Input Reject: Query contains non-Western character strings (CJK/Cyrillic). "
              "This exceeds the downstream vocabulary alignment rules.", file=sys.stderr)
        sys.exit(1)
        
    # 2. Heuristic Hardware Extraction from raw requirements text fields if numeric fields are absent
    if 'min_ram_gb' not in raw_data or pd.isna(raw_data['min_ram_gb']):
        pc_reqs = str(raw_data.get('pc_requirements', ''))
        ram_match = re.search(r'(\d+)\s*(?:GB|gb)\s*(?:RAM|ram)', pc_reqs)
        raw_data['min_ram_gb'] = float(ram_match.group(1)) if ram_match else np.nan

    if 'rec_ram_gb' not in raw_data or pd.isna(raw_data['rec_ram_gb']):
        raw_data['rec_ram_gb'] = raw_data.get('min_ram_gb', np.nan)
        
    return raw_data


def main():
    parser = argparse.ArgumentParser(description="Inference interface for HitPredictor-Steam production models.")
    parser.add_argument('-q', '--query', type=str, required=True,
                        help="Raw single-instance JSON format data string matching the ingestion feature schema.")
    parser.add_argument('-p', '--pre_release', action='store_true',
                        help="Route the query matrix exclusively through the Pre-Launch Model architecture.")
    parser.add_argument('-m', '--model_path', type=str, default=None,
                        help="Provide custom path to model .pkl bundle file.")
    args = parser.parse_args()

    # 1. Dynamic path assignment for model artifact files
    if args.model_path:
        artifact_path = args.model_path
    else:
        model_dir = "pre_release_model" if args.pre_release else "post_release_model"
        artifact_path = f"../results/{model_dir}/production_pipeline_bundle.pkl"

    if not os.path.exists(artifact_path):
        print(f"[-] Critical Error: Serialized artifact bundle not found at location: {artifact_path}", 
              file=sys.stderr)
        sys.exit(1)

    # 2. Parse and validate the single input query
    try:
        raw_payload = json.loads(args.query)
    except json.JSONDecodeError as e:
        print(f"[-] Critical Error: Malformed JSON query string submitted. Trace: {e}", file=sys.stderr)
        sys.exit(1)

    # Apply character encoding validations and initial transformations
    sanitized_payload = apply_pre_inference_guards(raw_payload)
    
    # 3. Restructure query payload dictionary to full pandas structure match
    input_df = pd.DataFrame([sanitized_payload])

    # 4. Enforce strict Dual-Mode scoping schema separation
    if args.pre_release:
        post_launch_features = [
            'metacritic_score', 'review_ratio', 'num_achievements', 
            'ccu', 'positive', 'negative', 'average_playtime_forever'
        ]
        input_df = input_df.drop(columns=post_launch_features, errors='ignore')
        print("[*] Mode Selected: Pre-Launch evaluation activated (post-launch vectors purged).")
    else:
        print("[*] Mode Selected: Post-Launch analysis activated (complete feature scope utilized).")

    # 5. Deserialize the scikit-learn compatible full training artifact pipeline step
    try:
        print(f"[*] Ingesting serialized pipeline components from: {artifact_path}")
        pipeline_bundle = joblib.load(artifact_path)
        
        # Verify package structural dictionary strategy
        if isinstance(pipeline_bundle, dict):
            model = pipeline_bundle['model']
            feature_extractor = pipeline_bundle['extractor']
            sanitizer = pipeline_bundle['sanitizer']
            scaler_pca = pipeline_bundle['scaler_pca']
        else:
            # Fallback wrapper if saved as a pure combined pipeline object
            print("[-] Ingestion Exception: Loaded file does not map to standard pipeline bundle layout.", 
                  file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"[-] Critical Error unpacking serialized pipeline artifacts: {e}", file=sys.stderr)
        sys.exit(1)

    # 6. Stepwise transformation execution to prevent data leakage and index drift
    try:
        # Run custom scikit-learn transformers
        X_extracted = feature_extractor.transform(input_df)
        X_sanitized = sanitizer.transform(X_extracted)
        X_scaled_pca = scaler_pca.transform(X_sanitized)
        
        # 7. Execute probabilistic inference step
        predicted_class = model.predict(X_scaled_pca)[0]
        prediction_probabilities = model.predict_proba(X_scaled_pca)[0]
        
        # Define semantic target tiers (adapts dynamically to 3 or 5 class setup)
        class_mappings = {
            0: "Class 0: The Indie Long-Tail (Low Adoption)",
            1: "Class 1: Healthy Niche (Sustainable Indie)",
            2: "Class 2: Mid-Market Success (Breakout Hits)",
            3: "Class 3: Major Success (AA Level / Viral Hits)",
            4: "Class 4: Mega-Hit / Blockbuster (AAA Status)"
        }
        
        # Output clean response to stdout
        print("\n" + "="*60)
        print("                 INFERENCE PREDICTION RESULTS               ")
        print("="*60)
        print(f"Predicted Success Category : {class_mappings.get(predicted_class, f'Tier {predicted_class}')}")
        print(f"Model Classification Index : Tier {predicted_class}")
        print(f"Confidence Distribution    : {np.round(prediction_probabilities, 4)}")
        print("="*60 + "\n")

    except Exception as e:
        print(f"[-] Runtime Error during feature transformation pipeline: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

