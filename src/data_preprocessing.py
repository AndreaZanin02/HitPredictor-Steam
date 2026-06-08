import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

""" Pipeline to preprocess the clean dataset """


# ----- Use on Train Set -----

# Training MLB for categories, genres, tags, languages and extract top developers/publishers
def fit_categorical_features(df_train, top_n=50):

    mlb_cat = MultiLabelBinarizer()
    mlb_cat.fit(df_train['categories'].tolist() if 'categories' in df_train.columns else [])
    
    mlb_genres = MultiLabelBinarizer()
    mlb_genres.fit(df_train['genres'].tolist() if 'genres' in df_train.columns else [])

    mlb_tags = MultiLabelBinarizer()
    mlb_tags.fit(df_train['tags'].tolist() if 'tags' in df_train.columns else [])

    mlb_langs = MultiLabelBinarizer()
    mlb_langs.fit(df_train['languages'].tolist() if 'languages' in df_train.columns else [])
    
    top_publishers = df_train['publishers'].explode().dropna().value_counts().head(top_n).index.tolist() if 'publishers' in df_train.columns else []
    top_developers = df_train['developers'].explode().dropna().value_counts().head(top_n).index.tolist() if 'developers' in df_train.columns else []
    
    return {
        'mlb_cat': mlb_cat, 'mlb_genres': mlb_genres, 'mlb_tags': mlb_tags, 'mlb_langs': mlb_langs,
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

# ------------------------------------------------


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

    if 'languages' in df_out.columns:
        encoded = pd.DataFrame(fitted_objects['mlb_langs'].transform(df_out['languages']), columns=[f"lang_{c}" for c in fitted_objects['mlb_langs'].classes_], index=df_out.index)
        df_out = pd.concat([df_out, encoded], axis=1)

    if 'publishers' in df_out.columns:
        for pub in fitted_objects['top_publishers']:
            df_out[f'pub_{str(pub).replace(" ", "_")}'] = df_out['publishers'].apply(lambda x: 1 if pub in x else 0)
            
    if 'developers' in df_out.columns:
        for dev in fitted_objects['top_developers']:
            df_out[f'dev_{str(dev).replace(" ", "_")}'] = df_out['developers'].apply(lambda x: 1 if dev in x else 0)

    return df_out.drop(columns=['categories', 'genres', 'tags', 'publishers', 'developers', 'languages'], errors='ignore')

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

# ------------------------------------------------