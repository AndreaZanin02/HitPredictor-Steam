import pandas as pd
import ast
import re
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import torch


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

def extract_json_features(df):
    print("Estrazione feature dalle strutture JSON...")
    
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
    if 'publishers' in df.columns: df['publishers'] = df['publishers'].apply(extract_list_from_dicts)
    if 'developers' in df.columns: df['developers'] = df['developers'].apply(extract_list_from_dicts)
    if 'tags' in df.columns: df['tags'] = df['tags'].apply(extract_keys_from_dict)
    
    return df

def extract_ram_features(df):
    print("Extracting ram features...")
    def extract_text_from_req(val):
        if pd.isna(val) or val == '[]': return ""
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, dict):
                return f"{parsed.get('minimum', '')} {parsed.get('recommended', '')}".lower()
        except: pass
        return ""

    df['all_requirements'] = (
        df.get('pc_requirements', pd.Series([""]*len(df))).apply(extract_text_from_req) + " " +
        df.get('mac_requirements', pd.Series([""]*len(df))).apply(extract_text_from_req) + " " +
        df.get('linux_requirements', pd.Series([""]*len(df))).apply(extract_text_from_req)
    )
    df['all_requirements'] = df['all_requirements'].str.replace(r'<[^>]+>', ' ', regex=True)

    def parse_ram_to_gb(text):
        if not text: return np.nan
        text = re.sub(r'(video\s*memory|vram|graphics\s*memory|storage|hard\s*drive|space)', '', text)
        ram_pattern = re.compile(r'(?:memory[:\s-]*(\d+[\.,]?\d*)\s*(mb|gb))|(?:(\d+[\.,]?\d*)\s*(mb|gb)\s*ram)')
        matches = ram_pattern.findall(text)
        if not matches: return np.nan
        match = matches[0]
        value_str = match[0] if match[0] else match[2]
        unit = match[1] if match[1] else match[3]
        try:
            value = float(value_str.replace(',', '.'))
            return value / 1024.0 if unit == 'mb' else value 
        except: return np.nan

    df['min_ram_gb'] = df['all_requirements'].apply(parse_ram_to_gb)
    return df

def extract_gpu_cpu_features(df):
    print("Extracting CPU/GPU features (vectorized)...")
    
    reqs = df['all_requirements'].fillna('').str.lower()
    
    # Booleani 0/1 to show if a specific CPU/GPU is required
    df['req_high_end_gpu'] = reqs.str.contains(r'(rtx\s*\d{4}|rx\s*\d{4}|radeon\s*vii)', regex=True).astype(int)
    df['req_dedicated_gpu'] = reqs.str.contains(r'(gtx|geforce|radeon\s*r|nvidia|amd)', regex=True).astype(int)
    df['req_high_cpu'] = reqs.str.contains(r'(i7|i9|ryzen\s*5|ryzen\s*7)', regex=True).astype(int)
    df['req_mid_cpu'] = reqs.str.contains(r'(i5|ryzen\s*3|fx-)', regex=True).astype(int)
    
    df = df.drop(columns=['all_requirements'], errors='ignore')
    return df

def extract_financial_and_temporal(df):
    print("Extracting financial and time features...")
    df['is_free'] = df['is_free'].fillna(False).astype(int)
    if 'initialprice' in df.columns:
        df['price'] = df['initialprice'].fillna(0) / 100.0  
    
    def parse_date(date_str):
        if not date_str: return pd.NaT
        return pd.to_datetime(date_str, errors='coerce')

    if 'release_date' in df.columns:
        parsed_dates = df['release_date'].apply(lambda x: parse_dict(x).get('date', '') if isinstance(x, str) and '{' in x else str(x))
        dates = parsed_dates.apply(parse_date)
        reference_date = pd.to_datetime('2026-06-06') 
        df['days_since_release'] = (reference_date - dates).dt.days.fillna(0).astype(int)

    if 'positive' in df.columns and 'negative' in df.columns:
        total_reviews = df['positive'] + df['negative']
        df['review_ratio'] = np.where(total_reviews > 0, df['positive'] / total_reviews, 0)

    return df

def clean_and_export(df, output_filename='steam_dataset_ready.csv'):
    print("Final cleaning...")
    cols_to_drop = [
        'steam_appid', 'appid', 'name_store', 'name_spy', 'name',
        'about_the_game', 'reviews', 'platforms', 'owners',
        'owners_lower_bound', 'metacritic', 'release_date', 'initialprice',
        'price_overview', 'movies', 'screenshots', 'header_image', 'background', 
        'website', 'drm_notice', 'ext_user_account_notice', 'support_info', 
        'legal_notice', 'pc_requirements', 'mac_requirements', 'linux_requirements', 
        'demos', 'packages', 'package_groups', 'content_descriptors', 'supported_languages'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    df.to_csv(output_filename, index=False)
    print(f"Dataset saved into '{output_filename}' with shape: {df.shape}")
    return df


# ------------------------------------ DATA LEAKAGE - USE ONLY AFTER TRAIN/TEST SPLIT ------------------------------------------


# ----- Use on Train Set -----

# Training MLB for categories, genres, tags and extract top developers/publishers
def fit_categorical_features(df_train, top_n=50):

    mlb_cat = MultiLabelBinarizer()
    mlb_cat.fit(df_train['categories'].tolist() if 'categories' in df_train.columns else [])
    
    mlb_genres = MultiLabelBinarizer()
    mlb_genres.fit(df_train['genres'].tolist() if 'genres' in df_train.columns else [])

    mlb_tags = MultiLabelBinarizer()
    mlb_tags.fit(df_train['tags'].tolist() if 'tags' in df_train.columns else [])
    
    top_publishers = df_train['publishers'].explode().dropna().value_counts().head(top_n).index.tolist() if 'publishers' in df_train.columns else []
    top_developers = df_train['developers'].explode().dropna().value_counts().head(top_n).index.tolist() if 'developers' in df_train.columns else []
    
    return {
        'mlb_cat': mlb_cat, 'mlb_genres': mlb_genres, 'mlb_tags': mlb_tags,
        'top_publishers': top_publishers, 'top_developers': top_developers
    }

#  Training TF-IDF and PCA on descriptions
def fit_text_features(df_train):

    df_train['short_description'] = df_train['short_description'].fillna("")
    df_train['detailed_description'] = df_train['detailed_description'].fillna("")
    
    tfidf = TfidfVectorizer(max_features=30, stop_words='english')
    tfidf.fit(df_train['short_description'])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-mpnet-base-v2', device=device)
    embeddings = model.encode(df_train['detailed_description'].tolist(), show_progress_bar=True)
    
    pca = PCA(n_components=50)
    pca.fit(embeddings)
    
    return {'tfidf': tfidf, 'pca': pca, 'st_model': model}


# ----- Use on Train and Test/Inference sets -----

# Apply the categorical transformer
def transform_categorical_features(df, fitted_objects):
    df_out = df.copy()
    
    if 'categories' in df_out.columns:
        encoded = pd.DataFrame(fitted_objects['mlb_cat'].transform(df_out['categories']), columns=[f"cat_{c}" for c in fitted_objects['mlb_cat'].classes_], index=df_out.index)
        df_out = pd.concat([df_out, encoded], axis=1)
        
    if 'genres' in df_out.columns:
        encoded = pd.DataFrame(fitted_objects['mlb_genres'].transform(df_out['genres']), columns=[f"genre_{c}" for c in fitted_objects['mlb_genres'].classes_], index=df_out.index)
        df_out = pd.concat([df_out, encoded], axis=1)

    if 'tags' in df_out.columns:
        encoded = pd.DataFrame(fitted_objects['mlb_tags'].transform(df_out['tags']), columns=[f"tag_{c}" for c in fitted_objects['mlb_tags'].classes_], index=df_out.index)
        df_out = pd.concat([df_out, encoded], axis=1)

    if 'publishers' in df_out.columns:
        for pub in fitted_objects['top_publishers']:
            df_out[f'pub_{str(pub).replace(" ", "_")}'] = df_out['publishers'].apply(lambda x: 1 if pub in x else 0)
            
    if 'developers' in df_out.columns:
        for dev in fitted_objects['top_developers']:
            df_out[f'dev_{str(dev).replace(" ", "_")}'] = df_out['developers'].apply(lambda x: 1 if dev in x else 0)

    return df_out.drop(columns=['categories', 'genres', 'tags', 'publishers', 'developers'], errors='ignore')

# Apply TF-IDF e PCA on the text
def transform_text_features(df, fitted_objects):

    df_out = df.copy()
    df_out['short_description'] = df_out['short_description'].fillna("")
    df_out['detailed_description'] = df_out['detailed_description'].fillna("")
    
    # Transform TF-IDF
    tfidf_matrix = fitted_objects['tfidf'].transform(df_out['short_description'])
    tfidf_cols = [f"tfidf_{w}" for w in fitted_objects['tfidf'].get_feature_names_out()]
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols, index=df_out.index)
    
    # Transform Embeddings + PCA
    embeddings = fitted_objects['st_model'].encode(df_out['detailed_description'].tolist(), show_progress_bar=False)
    pca_matrix = fitted_objects['pca'].transform(embeddings)
    pca_cols = [f"pca_{i}" for i in range(pca_matrix.shape[1])]
    df_pca = pd.DataFrame(pca_matrix, columns=pca_cols, index=df_out.index)
    
    df_out = pd.concat([df_out, df_tfidf, df_pca], axis=1)
    return df_out.drop(columns=['short_description', 'detailed_description'], errors='ignore')

# -------------------------------------------------------------------------------------------------------------------------------------------------


# Cleaning pipeline
if __name__ == "__main__":

    FILE_STEAM_STORE = "/dataset/raw_data/steam_app_data.csv"
    FILE_STEAM_SPY = "/dataset/raw_data/steamspy_data.csv"
    
    df_merged = load_and_merge(FILE_STEAM_STORE, FILE_STEAM_SPY)
    df_target = process_target_owners(df_merged)
    df_json = extract_json_features(df_target)
    df_ram = extract_ram_features(df_json)
    df_hw = extract_gpu_cpu_features(df_ram)
    df_fin_temp = extract_financial_and_temporal(df_hw)
    df_final = clean_and_export(df_fin_temp, output_filename="/dataset/cleaned_data/clean_dataset.csv")