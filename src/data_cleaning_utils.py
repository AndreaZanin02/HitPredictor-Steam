import pandas as pd
import ast
import re
import numpy as np
import csv
from datetime import datetime
from pathlib import Path


"""
Utility Functions
├─ parse_dict
├─ extract_list_from_dicts
├─ extract_keys_from_dict
└─ parse_simple_list
"""
def parse_dict(val):
    """
    Safely convert a stringified dictionary into a Python dict.
    Handles NaN, None, real dicts, or malformed strings.
    Returns parsed dict or {} on failure.
    Uses ast.literal_eval (safe alternative to eval).
    Designed to be defensive: any parsing error is absorbed
    to prevent pipeline failures in downstream steps.
    """
    if pd.isna(val): return {}
    try: return ast.literal_eval(val)
    except: return {}

def extract_list_from_dicts(val):
    """
    Convert Steam-style list-of-dicts into ML-ready token list.
    Extracts 'description' from each dict, normalizes text by replacing
    spaces with underscores, and ignores invalid entries.
    Accepts stringified lists, real lists, or NaN values.
    Returns cleaned list or [] if input is invalid/unparseable.
    Fail-safe: any parsing error is absorbed to keep pipeline stable.
    """
  
    if pd.isna(val): return []
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return [str(item.get('description', '')).replace(' ', '_') for item in parsed if isinstance(item, dict)]
        return []
    except: return []

def extract_keys_from_dict(val, top_n=5):
    """
    Extract top-N highest scoring keys from a dict-like structure.
    Used mainly for SteamSpy tags where values represent relevance scores.
    Parses input safely via parse_dict() and returns empty list on failure.
    Output is sorted by value (descending) and normalized with underscores.
    Returns up to top_n keys as ML-ready tokens.
    Fail-safe: malformed or missing data never breaks the pipeline.
    """

    parsed = parse_dict(val)
    if not parsed: return []
    # Top n tags
    sorted_tags = sorted(parsed.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return [str(tag[0]).replace(' ', '_') for tag in sorted_tags]

# Parsing string lists
def parse_simple_list(val):
    """
    Parse list-like values into a normalized list of strings.
    Supports stringified Python lists, comma-separated strings,
    and missing values.
    Normalizes items by trimming whitespace and replacing spaces
    with underscores.
    Falls back to comma splitting if literal_eval() fails.
    Returns cleaned list or [] for empty, invalid, or malformed input.
    """

    if pd.isna(val) or val == '[]': 
        return []
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return [str(item).strip().replace(' ', '_') for item in parsed]
        return []
    except (ValueError, SyntaxError):
        if isinstance(val, str):
            return [item.strip().replace(' ', '_') for item in val.split(',') if item.strip()]
        return []

"""
Dataset Loading
└─ load_and_merge
"""
def load_and_merge(steam_app_path, steam_spy_path):
    """
    Load Steam Store and SteamSpy datasets and merge them into a
    single DataFrame.
    Joins both sources using the common Steam application identifier
    (steam_appid in Store, appid in SteamSpy).
    Applies '_store' and '_spy' suffixes to overlapping column names.
    Returns a unified dataset combining metadata, player statistics,
    tags, reviews, and ownership information for ML processing.
    """
    print("Loading and merging original datasets...")

    # Skipping bad lines
    try:
        df_store = pd.read_csv(steam_app_path, on_bad_lines='skip', engine='python')
        df_spy = pd.read_csv(steam_spy_path, on_bad_lines='skip', engine='python')
    except TypeError:
        df_store = pd.read_csv(steam_app_path, error_bad_lines=False, engine='python')
        df_spy = pd.read_csv(steam_spy_path, error_bad_lines=False, engine='python')

    # Merging
    df = pd.merge(df_store, df_spy, left_on='steam_appid', right_on='appid', suffixes=('_store', '_spy'))

    # Check on the schema
    # If a malformed line had exactly the same number of commas, but the data 
    # still shifted (e.g., the description text ended up in the price column), 
    # the ID column (which must be purely numeric) will contain text fragments.
    initial_len = len(df)
    
    # We force the ID to be numeric. Anything that's shifted text will become NaN
    df['steam_appid'] = pd.to_numeric(df['steam_appid'], errors='coerce')
    
    # Dropping shifted lines
    df = df.dropna(subset=['steam_appid']).copy()
    
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"   -> Dropped {dropped} malformed/shifted rows post-merge.")

    # Cast back the ID into int
    df['steam_appid'] = df['steam_appid'].astype(int)
    return df

"""
Target Engineering
└─ process_target_owners
"""
def process_target_owners(df):
    """
    Convert SteamSpy ownership ranges into 5 ordinal target tiers.
    Mapping logic derived from project architecture:
    Tier 0: 0 .. 20k           (The Indie Long-Tail - Low adoption)
    Tier 1: 20k .. 100k        (Healthy Niche - Sustainable indie)
    Tier 2: 100k .. 500k       (Mid-Market Success - Breakout hits)
    Tier 3: 500k .. 2M         (Major Success - AA level)
    Tier 4: 2M+                (Hit / AAA status)
    """
    print("Processing the target: 'owners' (Re-Binning in 5 Tiers)...")
    
    steamspy_tiers = {
        # Tier 0
        '0 .. 20,000': 0, 
        
        # Tier 1
        '20,000 .. 50,000': 1, 
        '50,000 .. 100,000': 1,
        
        # Tier 2
        '100,000 .. 200,000': 2, 
        '200,000 .. 500,000': 2, 
        
        # Tier 3
        '500,000 .. 1,000,000': 3, 
        '1,000,000 .. 2,000,000': 3,
        
        # Tier 4
        '2,000,000 .. 5,000,000': 4, 
        '5,000,000 .. 10,000,000': 4,
        '10,000,000 .. 20,000,000': 4, 
        '20,000,000 .. 50,000,000': 4, 
        '50,000,000 .. 100,000,000': 4, 
        '100,000,000 .. 200,000,000': 4
    }
    
    # Mappatura dei valori
    df['target_owners'] = df['owners'].map(steamspy_tiers)
    
    # Controllo per eventuali valori anomali
    unmapped = df['target_owners'].isna().sum()
    if unmapped > 0:
        print(f"   -> WARNING: {unmapped} lines dropped (range 'owners' non valido o mancante)")
    
    # Rimozione righe con target nullo e cast esplicito a intero
    df = df.dropna(subset=['target_owners']).copy()
    df['target_owners'] = df['target_owners'].astype(int)
    
    return df

"""
Basic Game Feature Engineering
└─ process_game_features
"""
# Filtering games and cleaning game features
def process_game_features(df):
    """
    Filter non-game entries and engineer basic gameplay features.
    Keeps only rows where type == 'game', excluding DLCs, software,
    videos, and other store items.
    Creates:
    - is_controller_supported (binary flag)
    - num_dlc (DLC count)
    Converts raw metadata into simple ML-friendly features.
    """
    print("Cleaning type, controller support, and dlc features...")
    
    # Filtering on game
    if 'type' in df.columns:
        initial_len = len(df)
        df = df[df['type'] == 'game'].copy()
        print(f"   -> Dropped {initial_len - len(df)} non-game rows.")
        
    # Binarizing controller support
    if 'controller_support' in df.columns:
        df['is_controller_supported'] = df['controller_support'].notna().astype(int)
        df = df.drop(columns=['controller_support'])
        
    # Counting DLCs
    if 'dlc' in df.columns:
        def count_dlcs(val):
            if pd.isna(val) or val == '[]': 
                return 0
            if isinstance(val, str):
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list): 
                        return len(parsed)
                except (ValueError, SyntaxError): 
                    return 0
            elif isinstance(val, list):
                return len(val)
            return 0
            
        df['num_dlc'] = df['dlc'].apply(count_dlcs)
        df = df.drop(columns=['dlc'])
        
    return df

"""
Restrictions Features
└─ extract_restrictions_features
"""
# Extract DRM and external account requirements
def extract_restrictions_features(df):
    """
    Convert DRM and external account requirements into binary features.
    Creates:
    - has_third_party_drm
    - requires_ext_account
    Uses the presence of restriction notices rather than their text,
    reducing feature complexity while preserving relevant signals.
    Original text columns are removed after feature extraction.
    """

    print("Extracting DRM and external account features...")
    
    if 'drm_notice' in df.columns:
        df['has_third_party_drm'] = df['drm_notice'].notna().astype(int)
        df = df.drop(columns=['drm_notice'])
        
    if 'ext_user_account_notice' in df.columns:
        df['requires_ext_account'] = df['ext_user_account_notice'].notna().astype(int)
        df = df.drop(columns=['ext_user_account_notice'])
        
    return df

"""
JSON Feature Extraction
├─ extract_achievements_total
└─ extract_json_features
"""
# Extract the value of the key 'total' from achievements dict
def extract_achievements_total(val):
    """
    Extract total number of achievements from Steam metadata.
    Steam stores achievements as a stringified dictionary containing
    a 'total' field.
    Returns 0 when:
    - value is NaN or missing
    - parsing fails
    - structure is not a dict
    - 'total' field is absent
    Ensures safe numeric output for ML pipelines.
    """
    if pd.isna(val): 
        return 0
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, dict):
            return int(parsed.get('total', 0))
        return 0
    except (ValueError, SyntaxError, TypeError):
        return 0

# Extract feature from JSON structures
def extract_json_features(df):
    """
    Flatten JSON-like Steam metadata into ML-ready features.
    Extracts:
    - Platform support (Windows/Mac/Linux as binary flags)
    - metacritic_score
    - num_achievements
    - categories, genres, tags (normalized token lists)
    Converts nested structures into tabular features suitable for ML.
    Ensures compatibility with standard machine learning pipelines.
    """

    print("Extraction features from JSON structures...")
    
    if 'platforms' in df.columns:
        platforms_parsed = df['platforms'].apply(parse_dict)
        df['platform_windows'] = platforms_parsed.apply(lambda x: x.get('windows', False)).astype(int)
        df['platform_mac'] = platforms_parsed.apply(lambda x: x.get('mac', False)).astype(int)
        df['platform_linux'] = platforms_parsed.apply(lambda x: x.get('linux', False)).astype(int)

    if 'metacritic' in df.columns:
        df['metacritic_score'] = df['metacritic'].apply(lambda x: parse_dict(x).get('score', 0))

    # Extracting categorical features into lists
    if 'categories' in df.columns: df['categories'] = df['categories'].apply(extract_list_from_dicts)
    if 'genres' in df.columns: df['genres'] = df['genres'].apply(extract_list_from_dicts)
    if 'publishers' in df.columns: df['publishers'] = df['publishers'].apply(parse_simple_list)
    if 'developers' in df.columns: df['developers'] = df['developers'].apply(parse_simple_list)
    if 'tags' in df.columns: df['tags'] = df['tags'].apply(extract_keys_from_dict)
    
    if 'achievements' in df.columns:
        df['num_achievements'] = df['achievements'].apply(extract_achievements_total)
    return df


"""
Text Processing
└─ clean_text_descriptions
"""
# Cleaning HTML tags from descriptions
def clean_text_descriptions(df):
    """
    Remove HTML tags and normalize whitespace in Steam descriptions.
    Cleans:
    - detailed_description
    - short_description
    Steps:
    - Strip HTML markup
    - Normalize repeated whitespace
    - Trim leading/trailing spaces
    Improves text quality for NLP tasks like TF-IDF, embeddings,
    and classification by removing formatting noise.
    """

    print("Cleaning HTML tags and malformed characters from text descriptions...")
    
    for col in ['name_store', 'name', 'detailed_description', 'short_description']:
        if col in df.columns:
            # Cast to string
            df[col] = df[col].astype(str)
            # Removing HTML tags
            df[col] = df[col].str.replace(r'<[^>]+>', ' ', regex=True)
            # Removing newline, carriage return and tabs
            df[col] = df[col].str.replace(r'[\n\r\t]+', ' ', regex=True)
            # Removing multiple spaces
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True).str.strip()
            
            # Marking as NaN the nan string
            df.loc[df[col] == 'nan', col] = np.nan
            
    return df

"""
Hardware Requirement Features
├─ extract_ram_features
└─ extract_gpu_cpu_features
"""
def extract_ram_features(df):
    """
    Extract and normalize RAM requirements from system specs.
    Creates:
    - min_ram_gb
    - rec_ram_gb
    Parses semi-structured requirement text and converts values
    into standardized gigabytes.
    Handles formats like MB/GB and HTML-laced strings.
    Provides a proxy for game technical complexity.
    """
    print("Extracting min and recommended ram features...")

    def get_req_string(val, req_type):
        if pd.isna(val) or val == '[]': 
            return ""
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, dict):
                return parsed.get(req_type, "")
        except: 
            pass
        return ""

    def parse_ram_to_gb(text):
        if not text or pd.isna(text): 
            return np.nan
            
        text = str(text).lower()
        # Preentive cleaning
        text = re.sub(r'(video\s*memory|vram|graphics\s*memory|storage|hard\s*drive|space)', '', text)
        
        ram_pattern = re.compile(r'(?:memory[:\s-]*(\d+[\.,]?\d*)\s*(mb|gb|kb))|(?:(\d+[\.,]?\d*)\s*(mb|gb|kb)\s*ram)')
        matches = ram_pattern.findall(text)
        
        if not matches: 
            return np.nan
            
        match = matches[0]
        value_str = match[0] if match[0] else match[2]
        unit = match[1] if match[1] else match[3]
        
        try:
            value = float(value_str.replace(',', '.'))
            if unit == 'kb': 
                return value / (1024.0 * 1024.0)
            elif unit == 'mb': 
                return value / 1024.0
            else: 
                return value
        except: 
            return np.nan

    # Processing minimum and recommended
    for req_type in ['minimum', 'recommended']:
        req_strings = (
            df.get('pc_requirements', pd.Series([""]*len(df))).apply(lambda x: get_req_string(x, req_type)) + " " +
            df.get('mac_requirements', pd.Series([""]*len(df))).apply(lambda x: get_req_string(x, req_type)) + " " +
            df.get('linux_requirements', pd.Series([""]*len(df))).apply(lambda x: get_req_string(x, req_type))
        )
        
        # HTML cleaning
        req_strings = req_strings.str.replace(r'<[^>]+>', ' ', regex=True)
        
        # Creating columns
        col_name = 'min_ram_gb' if req_type == 'minimum' else 'rec_ram_gb'
        df[col_name] = req_strings.apply(parse_ram_to_gb)

    df['rec_ram_gb'] = df['rec_ram_gb'].fillna(df['min_ram_gb'])
    return df

def extract_gpu_cpu_features(df):
    """
    Extract hardware requirement signals from system specs text.
    Creates binary indicators for:
    - req_high_end_gpu
    - req_dedicated_gpu
    - req_high_cpu
    - req_mid_cpu
    Uses heuristic pattern matching to estimate hardware demands.
    Not a precise benchmark, but a proxy for technical complexity.
    """
    print("Extracting CPU/GPU features (vectorized)...")
    
    reqs = (
        df.get('pc_requirements', pd.Series(['']*len(df))).fillna('') + ' ' +
        df.get('mac_requirements', pd.Series(['']*len(df))).fillna('') + ' ' +
        df.get('linux_requirements', pd.Series(['']*len(df))).fillna('')
    ).str.lower()
    
    # Booleani 0/1 to show if a specific CPU/GPU is required
    df['req_high_end_gpu'] = reqs.str.contains(r'rtx\s*\d{4}|rx\s*\d{4}|radeon\s*vii', regex=True).astype(int)
    df['req_dedicated_gpu'] = reqs.str.contains(r'gtx|geforce|radeon\s*r|nvidia|amd', regex=True).astype(int)
    df['req_high_cpu'] = reqs.str.contains(r'i7|i9|ryzen\s*5|ryzen\s*7', regex=True).astype(int)
    df['req_mid_cpu'] = reqs.str.contains(r'i5|ryzen\s*3|fx-', regex=True).astype(int)
    
    return df

"""
Financial and Temporal Features
└─ extract_financial_and_temporal
"""
def extract_financial_and_temporal(df):
    """
    Extract financial, temporal, and engagement features.
    Creates:
    - price
    - release_year
    - release_month
    - review_ratio
    Removes invalid or future-dated releases.
    Review ratio measures user sentiment from positive/negative reviews.
    Captures pricing, age, and popularity signals for modeling.
    """

    print("Extracting financial and time features...")

    if 'initialprice' in df.columns:
        df['price'] = df['initialprice'] / 100.0  
        
        # Check using is_free
        if 'is_free' in df.columns:
            # If is_free == True the price and discount are 0.0
            df.loc[df['is_free'] == True, 'price'] = 0.0
            df.loc[df['is_free'] == True, 'discount'] = 0.0
        else:
            df['is_free'] = (df['price'] == 0.0)

    def parse_date(date_str):
        if not date_str: return pd.NaT
        return pd.to_datetime(date_str, errors='coerce')

    if 'release_date' in df.columns:
        parsed_dates = df['release_date'].apply(lambda x: parse_dict(x).get('date', '') if isinstance(x, str) and '{' in x else str(x))
        dates = parsed_dates.apply(parse_date)
        
        # Temp column for using filters
        df['temp_date'] = dates
        initial_len = len(df)
        
        # Dropping games with not already published
        reference_date = pd.to_datetime(datetime.now().date())
        df = df[df['temp_date'].notna() & (df['temp_date'] <= reference_date)].copy()
        print(f"   -> Dropped {initial_len - len(df)} rows with missing or future release dates.")

        # Extracting month and year
        df['release_year'] = df['temp_date'].dt.year.astype(int)
        df['release_month'] = df['temp_date'].dt.month.astype(int)
        
        # Dropping temp column
        df = df.drop(columns=['temp_date'])

    if 'positive' in df.columns and 'negative' in df.columns:
        total_reviews = df['positive'] + df['negative']
        df['review_ratio'] = np.where(total_reviews > 0, df['positive'] / total_reviews, 0)
    return df

"""
Language Features
└─ extract_language_features
"""
# Extracting language features
def extract_language_features(df):
    """
    Normalize Steam language data into ML features.
    Creates:
    - languages (cleaned token list)
    - num_languages_supported
    Converts comma-separated strings into structured features.
    Missing values become empty lists.
    Serves as a proxy for localization effort and market reach.
    """

    print("Extracting language features...")
    if 'languages' in df.columns:
        
        df['languages'] = df['languages'].fillna('').apply(
            lambda x: [lang.strip().replace(' ', '_') for lang in str(x).split(',') if lang.strip()]
        )

        df['num_languages_supported'] = df['languages'].apply(len)
    return df

"""
Advanced quality filter
├─ drops games without name or descriptions
├─ drops games with descriptions or names that uses oriental alphabets
└─ checks for NaN values
"""
def advanced_quality_filtering(df):
    print("Applying advanced quality filters (NaN dropping, CJK/Cyrillic removal, imputations)...")
    
    initial_len = len(df)
    
    # Drop games without name or descriptions
    df = df.dropna(subset=['name_store', 'short_description', 'detailed_description']).copy()
    
    # Drops games with Chinese Japanese Korean or Cirillic alphabets
    # Range Unicode:
    # \u4e00-\u9fff : CJK Unified Ideographs
    # \u3040-\u30ff : Hiragana & Katakana
    # \uac00-\ud7af : Hangul
    # \u0400-\u04ff : Cirillic
    cjk_cyrillic_pattern = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff]')
    
    def has_foreign_chars(val):
        # If it's a list, it trasforms it to a string
        if isinstance(val, (list, np.ndarray)):
            val = " ".join(str(x) for x in val)
    
        if pd.isna(val): 
            return False
        
        return bool(cjk_cyrillic_pattern.search(str(val)))

    # Dropping the lines
    for col in ['name_store', 'short_description', 'detailed_description', 'tags', 'genres', 'publishers', 'developers']:
        if col in df.columns:
            mask = df[col].apply(has_foreign_chars)
            df = df[~mask]
    
    # Cleaning required_age (NaN -> 0)
    if 'required_age' in df.columns:
        df['required_age'] = pd.to_numeric(df['required_age'], errors='coerce').fillna(0).astype(int)
        
    # Cleaning NaN playtime
    playtime_cols = ['average_2weeks', 'median_2weeks', 'median_forever']
    for col in playtime_cols:
        if col in df.columns:
            # NaN -> 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Check for binary columns errors
    binary_cols = ['is_controller_supported', 'has_third_party_drm', 'requires_ext_account', 
                   'platform_windows', 'platform_mac', 'platform_linux', 
                   'req_high_end_gpu', 'req_dedicated_gpu', 'req_high_cpu', 'req_mid_cpu']
    for col in binary_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            df[col] = df[col].clip(0, 1)

    print(f"   -> Dropped {initial_len - len(df)} rows due to quality filtering (NaNs or Foreign Chars).")
    return df

"""
Dataset Organization
├─ reorder_and_rename_columns
└─ clean_and_export
"""
def reorder_and_rename_columns(df):
    """
    Standardize column names and enforce ML-ready column ordering.
    Groups features into logical categories (game metadata, content,
    language, engagement, pricing, and engineered features).
    Renames:
    - name_store -> name
    Ensures target_owners is placed last to avoid leakage.
    Improves readability, debugging, and reproducibility of the dataset.
    """

    print("Renaming and reordering columns...")
    
    df = df.rename(columns={'name_store': 'name'})
    
    group_name = ['name', 'required_age', 'developers', 'publishers', 'release_year', 'release_month']
    group_topic = ['categories', 'genres', 'tags', 'detailed_description', 'short_description', 'num_dlc', 'num_achievements']
    group_lan = ['languages', 'num_languages_supported']
    group_scores = ['metacritic_score', 'review_ratio']
    group_price = ['is_free', 'price', 'discount']
    
    new_cols = ['name'] if 'name' in df.columns else []
    
    for col in group_name + group_topic + group_lan + group_scores + group_price:
        if col in df.columns and col not in new_cols:
            new_cols.append(col)
            
    for col in df.columns:
        if col not in new_cols and col != 'target_owners':
            new_cols.append(col)
            
    if 'target_owners' in df.columns:
        new_cols.append('target_owners')
        
    return df[new_cols]

def clean_and_export(df, output_filename='steam_dataset_ready.csv'):
    """
    Final dataset cleanup before export.
    Removes:
    - Identifiers (no predictive value)
    - Raw processed columns (already engineered)
    - Duplicate merge artifacts
    - Target leakage features (e.g., ccu, positive, negative)
    - Irrelevant metadata
    Exports final ML-ready dataset to CSV and prints shape.
    Ensures clean separation between features and target signal.
    """
    print("Final cleaning...")

    cols_to_drop = [
        # Unique IDs of the games
        'steam_appid', 'appid', 'name_spy',
        
        # Features with zero variance
        'type', 'fullgame', 'score_rank', 'userscore',

        # Lists of links
        'movies', 'screenshots', 'header_image', 'background', 'website',

        # ALready processed data
        'pc_requirements', 'mac_requirements', 'linux_requirements', 'reviews', 'platforms',
        'owners', 'owners_lower_bound', 'release_date', 'initialprice', 'achievements',

        # Redundant features (due to datasets merge)
        'developer', 'publisher', 'supported_languages', 'metacritic',
        'price_overview', 'about_the_game', 'packages', 'package_groups', 'demos',
        'content_descriptors', 'recommendations', 'genre', 'is_free',

        # Features too correlated to our target (number of users) 
        'ccu', 'positive', 'negative',

        # Useless feature for our domain (EULA standard, support mails, ...)
        'legal_notice', 'support_info'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # Saving CSV
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_filename, index=False, quoting=csv.QUOTE_MINIMAL, escapechar='\\')
    print(f"Dataset saved into '{output_filename}' with shape: {df.shape}")
    return df
