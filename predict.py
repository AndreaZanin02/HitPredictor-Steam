#!/usr/bin/env python3
"""
HitPredictor-Steam: Production Inference Interface
==================================================

This script serves as an isolated, end-to-end inference execution engine.
It applies pre-inference guards, precomputes text embeddings, executes the
unified prediction pipeline, and calibrates the raw probabilities using 
the optimized Differential Evolution weights
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

# -------------------------------------------------------------------------
# PATH INJECTION & NAMESPACE RESOLUTION
# -------------------------------------------------------------------------
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from data_preprocessing import precompute_detailed_embeddings
except ImportError as e:
    print(f"Execution Warning: Custom transformer files must be present in the python path. Details: {e}", 
          file=sys.stderr)
    sys.exit(1)


def apply_pre_inference_guards(raw_data: dict) -> dict:
    """
    Enforces deterministic validation rules on incoming JSON payloads.
    """
    text_check_pool = [
        str(raw_data.get('name', '')),
        str(raw_data.get('short_description', '')),
        str(raw_data.get('detailed_description', ''))
    ]
    combined_text = " ".join(text_check_pool)
    
    cjk_cyrillic_pattern = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff]')
    if cjk_cyrillic_pattern.search(combined_text):
        print("Input Reject: Query contains non-Western character strings (CJK/Cyrillic).", 
              file=sys.stderr)
        sys.exit(1)
        
    return raw_data


def pad_missing_features(raw_data: dict) -> dict:
    """
    Validates the input payload against the strict training schema.
    Injects default neutral values for missing features and raises a verbose
    stderr warning to track incomplete data submissions
    """
    expected_schema = {
        'required_age': 0, 'developers': [], 'publishers': [], 
        'release_year': 2026, 'release_month': 1, 'categories': [], 
        'genres': [], 'tags': [], 'detailed_description': "", 
        'short_description': "", 'num_dlc': 0, 'num_achievements': 0, 
        'languages': [], 'num_languages_supported': 0, 'metacritic_score': 0, 
        'review_ratio': 0.0, 'price': 0.0, 'discount': 0.0, 
        'average_forever': 0, 'average_2weeks': 0, 'median_forever': 0, 
        'median_2weeks': 0, 'is_controller_supported': 0, 'platform_windows': 1, 
        'platform_mac': 0, 'platform_linux': 0, 'min_ram_gb': 0.0, 
        'rec_ram_gb': 0.0, 'req_high_end_gpu': 0, 'req_dedicated_gpu': 0, 
        'req_high_cpu': 0, 'req_mid_cpu': 0, 'has_third_party_drm': 0, 
        'requires_ext_account': 0
    }
    
    missing_keys = []
    
    for key, default_val in expected_schema.items():
        if key not in raw_data:
            raw_data[key] = default_val
            missing_keys.append(key)
        elif pd.isna(raw_data.get(key)):
            raw_data[key] = default_val
            missing_keys.append(f"{key} (NaN replaced)")

    if missing_keys:
        print(f"INFERENCE WARNING: JSON query is missing {len(missing_keys)} structural features.", file=sys.stderr)
        print(f"    The following features were auto-filled with neutered default values:", file=sys.stderr)
        for i in range(0, len(missing_keys), 5):
            print(f"    - {', '.join(missing_keys[i:i+5])}", file=sys.stderr)
        print("\n    Note: Prediction accuracy may be severely degraded due to sparse data matrix.\n", file=sys.stderr)
        
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

    if args.model_path:
        artifact_path = args.model_path
    else:
        model_dir = "pre_release_model" if args.pre_release else "post_release_model"
        model_filename = "pre_release_model.pkl" if args.pre_release else "post_release_model.pkl"
        artifact_path = f"../results/{model_dir}/{model_filename}"

    if not os.path.exists(artifact_path):
        print(f"[-] Critical Error: Serialized artifact bundle not found at location: {artifact_path}", 
              file=sys.stderr)
        sys.exit(1)

    try:
        raw_payload = json.loads(args.query)
    except json.JSONDecodeError as e:
        print(f"[-] Critical Error: Malformed JSON query string submitted. Trace: {e}", file=sys.stderr)
        sys.exit(1)

    # Payload validation
    sanitized_payload = apply_pre_inference_guards(raw_payload)
    padded_payload = pad_missing_features(sanitized_payload)
    
    input_df = pd.DataFrame([padded_payload])

    if args.pre_release:
        post_launch_features = [
            'num_achievements', 'metacritic_score', 'review_ratio', 'num_dlc',
            'discount', 'average_forever', 'average_2weeks', 'median_forever', 'median_2weeks',
            'ccu', 'positive', 'negative', 'average_playtime_forever'
        ]
        input_df = input_df.drop(columns=post_launch_features, errors='ignore')
        print("Mode Selected: Pre-Launch evaluation activated (post-launch vectors purged).")
    else:
        print("Mode Selected: Post-Launch analysis activated (complete feature scope utilized).")

    try:
        production_artifact = joblib.load(artifact_path)
        
        if isinstance(production_artifact, dict) and 'pipeline' in production_artifact and 'best_weights' in production_artifact:
            pipeline = production_artifact['pipeline']
            best_weights = production_artifact['best_weights']
        else:
            print("Ingestion Exception: Loaded file does not map to the expected {'pipeline', 'best_weights'} layout.", 
                  file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"Critical Error unpacking serialized artifacts: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        # Embeddings
        input_df['target_owners'] = 0 
        input_df, _ = precompute_detailed_embeddings(input_df, text_col='detailed_description')
        
        columns_to_drop = ['target_owners']
        if 'name' in input_df.columns:
            columns_to_drop.append('name')
        input_df = input_df.drop(columns=columns_to_drop)
        
        # Inference
        raw_probabilities = pipeline.predict_proba(input_df)[0]
        
        # Calibration
        adjusted_probabilities = raw_probabilities * best_weights
        predicted_class = np.argmax(adjusted_probabilities)
        normalized_confidence = adjusted_probabilities / np.sum(adjusted_probabilities)
        
        class_mappings = {
            0: "Class 0: The Indie Long-Tail (Low Adoption)",
            1: "Class 1: Healthy Niche (Sustainable Indie)",
            2: "Class 2: Mid-Market Success (Breakout Hits)",
            3: "Class 3: Major Success (AA Level / Viral Hits)",
            4: "Class 4: Mega-Hit / Blockbuster (AAA Status)"
        }
        
        print("\n" + "="*60)
        print("                 INFERENCE PREDICTION RESULTS               ")
        print("="*60)
        print(f"Predicted Success Category : {class_mappings.get(predicted_class, f'Tier {predicted_class}')}")
        print(f"Model Classification Index : Tier {predicted_class}")
        print(f"Raw Probabilities          : {np.round(raw_probabilities, 4)}")
        print(f"Applied Weights Setup      : {np.round(best_weights, 4)}")
        print(f"Adjusted Confidence (Norm) : {np.round(normalized_confidence, 4)}")
        print("="*60 + "\n")

    except Exception as e:
        print(f"Runtime Error during feature transformation and inference pipeline: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()