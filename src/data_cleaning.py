import pandas as pd
import ast
import re
import numpy as np
from datetime import datetime

# Pipeline to transform the raw datasets to a clean dataset useful for our model

def parse_dict(val):
    if pd.isna(val): return {}
    try: return ast.literal_eval(val)
    except: return {}

def extract_list_from_dicts(val):
    if pd.isna(val): return []
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return [str(item.get('description', '')).replace(' ', '_') for item in parsed if isinstance(item, dict)]
        return []
    except: return []

def extract_keys_from_dict(val, top_n=5):
    parsed = parse_dict(val)
    if not parsed: return []
    # Top n tags
    sorted_tags = sorted(parsed.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return [str(tag[0]).replace(' ', '_') for tag in sorted_tags]

# Parsing string lists
def parse_simple_list(val):
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

def load_and_merge(steam_app_path, steam_spy_path):
    print("Loading and merging original datasets...")
    df_store = pd.read_csv(steam_app_path)
    df_spy = pd.read_csv(steam_spy_path)
    df = pd.merge(df_store, df_spy, left_on='steam_appid', right_on='appid', suffixes=('_store', '_spy'))
    return df

def process_target_owners(df):
    print("Processing the target: 'owners'...")
    steamspy_tiers = {
        '0 .. 20,000': 0, '20,000 .. 50,000': 1, '50,000 .. 100,000': 2,
        '100,000 .. 200,000': 3, '200,000 .. 500,000': 4, '500,000 .. 1,000,000': 5,
        '1,000,000 .. 2,000,000': 6, '2,000,000 .. 5,000,000': 7,
        '5,000,000 .. 10,000,000': 8, '10,000,000 .. 20,000,000': 9,
        '20,000,000 .. 50,000,000': 10, '50,000,000 .. 100,000,000': 11,
        '100,000,000 .. 200,000,000': 12
    }
    
    df['target_owners'] = df['owners'].map(steamspy_tiers)
    unmapped = df['target_owners'].isna().sum()
    if unmapped > 0:
        print(f"   -> WARNING: {unmapped} lines dropped (range 'owners' non valido)")
    
    df = df.dropna(subset=['target_owners']).copy()
    return df

# Filtering games and cleaning game features
def process_game_features(df):
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

# Extract DRM and external account requirements
def extract_restrictions_features(df):
    print("Extracting DRM and external account features...")
    
    if 'drm_notice' in df.columns:
        df['has_third_party_drm'] = df['drm_notice'].notna().astype(int)
        df = df.drop(columns=['drm_notice'])
        
    if 'ext_user_account_notice' in df.columns:
        df['requires_ext_account'] = df['ext_user_account_notice'].notna().astype(int)
        df = df.drop(columns=['ext_user_account_notice'])
        
    return df

# Extract the value of the key 'total' from achievements dict
def extract_achievements_total(val):
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

# Cleaning HTML tags from descriptions
def clean_text_descriptions(df):
    print("Cleaning HTML tags from text descriptions...")
    
    # Pulisce sia la descrizione dettagliata che quella breve
    for col in ['detailed_description', 'short_description']:
        if col in df.columns:
            # 1. Rimuove i tag HTML sostituendoli con uno spazio
            df[col] = df[col].str.replace(r'<[^>]+>', ' ', regex=True)
            
            # 2. Rimuove eventuali spazi multipli generati dalla sostituzione
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            
            # 3. Rimuove gli spazi iniziali e finali
            df[col] = df[col].str.strip()
            
    return df

def extract_ram_features(df):
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

    return df

def extract_gpu_cpu_features(df):
    print("Extracting CPU/GPU features (vectorized)...")
    
    reqs = (df.get('pc_requirements', '') + df.get('mac_requirements', '') + df.get('linux_requirements', '')).fillna('').str.lower()
    
    # Booleani 0/1 to show if a specific CPU/GPU is required
    df['req_high_end_gpu'] = reqs.str.contains(r'rtx\s*\d{4}|rx\s*\d{4}|radeon\s*vii', regex=True).astype(int)
    df['req_dedicated_gpu'] = reqs.str.contains(r'gtx|geforce|radeon\s*r|nvidia|amd', regex=True).astype(int)
    df['req_high_cpu'] = reqs.str.contains(r'i7|i9|ryzen\s*5|ryzen\s*7', regex=True).astype(int)
    df['req_mid_cpu'] = reqs.str.contains(r'i5|ryzen\s*3|fx-', regex=True).astype(int)
    
    return df

def extract_financial_and_temporal(df):
    print("Extracting financial and time features...")

    if 'initialprice' in df.columns:
        df['price'] = df['initialprice'] / 100.0  
        
        # Check using is_free
        if 'is_free' in df.columns:
            # If is_free == True the price and discount are 0.0
            df.loc[df['is_free'] == True, 'price'] = 0.0
            df.loc[df['is_free'] == True, 'discount'] = 0.0
            # If the price is 0, the flag is_free is True
            df.loc[df['price'] == 0.0, 'is_free'] = True
        else:
            df['is_free'] = (df['price'] == 0.0)

    def parse_date(date_str):
        if not date_str: return pd.NaT
        return pd.to_datetime(date_str, errors='coerce')

    if 'release_date' in df.columns:
        parsed_dates = df['release_date'].apply(lambda x: parse_dict(x).get('date', '') if isinstance(x, str) and '{' in x else str(x))
        dates = parsed_dates.apply(parse_date)
        
        reference_date = pd.to_datetime(datetime.now().date()) 
        df['days_since_release'] = (reference_date - dates).dt.days

        initial_len = len(df)
        df = df[df['days_since_release'] > 0].copy()
        print(f"   -> Dropped {initial_len - len(df)} rows with missing or future/zero release dates.")

        df['days_since_release'] = df['days_since_release'].astype(int)

    if 'positive' in df.columns and 'negative' in df.columns:
        total_reviews = df['positive'] + df['negative']
        df['review_ratio'] = np.where(total_reviews > 0, df['positive'] / total_reviews, 0)
    return df

# Extracting language features
def extract_language_features(df):
    print("Extracting language features...")
    if 'languages' in df.columns:
        
        df['languages'] = df['languages'].fillna('').apply(
            lambda x: [lang.strip().replace(' ', '_') for lang in str(x).split(',') if lang.strip()]
        )

        df['num_languages_supported'] = df['languages'].apply(len)
    return df

def reorder_and_rename_columns(df):
    print("Renaming and reordering columns...")
    
    df = df.rename(columns={'name_store': 'name'})
    
    group_name = ['name', 'required_age', 'developers', 'publishers', 'days_since_release']
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
        'content_descriptors', 'recommendations', 'genre',

        # Features too correlated to our target (number of users) 
        'ccu', 'positive', 'negative',

        # Useless feature for our domain (EULA standard, support mails, ...)
        'legal_notice', 'support_info'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    df.to_csv(output_filename, index=False)
    print(f"Dataset saved into '{output_filename}' with shape: {df.shape}")
    return df



# Cleaning pipeline
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