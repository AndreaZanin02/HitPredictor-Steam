"""
Centralized module for preprocessing and feature encoding
Contains classes and functions that can be reused in tuning and training

This module transforms a cleaned dataset into ML-ready numerical features
using:
- MultiLabelBinarizer (categorical multi-label features)
- TF-IDF (short text descriptions)
"""

import pandas as pd
import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.impute import SimpleImputer

class CorrelationRemover(BaseEstimator, TransformerMixin):
    """
    Dynamically removes highly correlated features based on the training fold distribution
    """
    def __init__(self, threshold=0.97):
        self.threshold = threshold
        self.to_drop_ = []

    def fit(self, X, y=None):
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_ = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self

    def transform(self, X):
        return X.drop(columns=self.to_drop_, errors='ignore')


def dynamic_undersample(y):
    """
    Dynamically calculates how many samples to keep for Class 0
    based on the current fold size
    """
    import pandas as pd
    
    # Counting occurences
    counts = pd.Series(y).value_counts().to_dict()
    
    # Reducing class 0 to 25% of its current volume
    target_class_0 = int(counts[0] * 0.25)
    
    # Ensuring that Class 0 never becomes smaller than Class 1
    safe_target = max(target_class_0, counts.get(1, 0))
    
    return {0: safe_target}


class FeatureNameSanitizer(BaseEstimator, TransformerMixin):
    """
    Dynamically renames DataFrame columns by removing the prohibited characters
    ([, ], and <) from XGBoost/LightGBM.
    This should be inserted into the pipeline before selectors or classifiers
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_clean = X.copy()
            # Replacing problematic characthers with a _
            X_clean.columns = X_clean.columns.str.replace(r'[\[\]<]', '_', regex=True)
            return X_clean
        return X


# Embedding pre-computation function
def precompute_detailed_embeddings(df, text_col='detailed_description'):
    """
    It performs heavy embedding extraction only once to avoid OOMs in VRAM
    Being a frozen model, it does not generate data leakage if executed upstream
    """
    print("Pre-compuation of textual embeddings...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-mpnet-base-v2', device=device)
    
    texts = df[text_col].fillna("").tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    
    emb_cols = [f"raw_emb_{i}" for i in range(embeddings.shape[1])]
    df_emb = pd.DataFrame(embeddings, columns=emb_cols, index=df.index)
    
    # Cleaning VRAM after the computation
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
        
    return pd.concat([df.drop(columns=[text_col], errors='ignore'), df_emb], axis=1), emb_cols


# Categorical encoder and TF-IDF
class SteamFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    It encapsulates the fitting of MultiLabelBinarizer and TF-IDF.
    When inserted into the Pipeline, it guarantees the fit only on the current training fold
    """
    def __init__(self, top_n_creators=50, max_tfidf_features=30):
        self.top_n_creators = top_n_creators
        self.max_tfidf_features = max_tfidf_features
        
    def fit(self, X, y=None):
        # Setup and fit binarizers on the train fold data
        self.mlb_cat_ = MultiLabelBinarizer().fit(X['categories'].tolist() if 'categories' in X.columns else [])
        self.mlb_genres_ = MultiLabelBinarizer().fit(X['genres'].tolist() if 'genres' in X.columns else [])
        self.mlb_tags_ = MultiLabelBinarizer().fit(X['tags'].tolist() if 'tags' in X.columns else [])
        self.mlb_langs_ = MultiLabelBinarizer().fit(X['languages'].tolist() if 'languages' in X.columns else [])
        
        self.top_publishers_ = X['publishers'].explode().dropna().value_counts().head(self.top_n_creators).index.tolist() if 'publishers' in X.columns else []
        self.top_developers_ = X['developers'].explode().dropna().value_counts().head(self.top_n_creators).index.tolist() if 'developers' in X.columns else []
        
        # TF-IDF fit only on the short_description of this specific fold
        short_desc = X['short_description'].fillna("") if 'short_description' in X.columns else pd.Series([""]*len(X))
        self.tfidf_ = TfidfVectorizer(max_features=self.max_tfidf_features, stop_words='english')
        self.tfidf_.fit(short_desc)

        # Removing NaN from the RAM using the median of the training set
        if 'min_ram_gb' in X.columns:
            self.ram_imputer_ = SimpleImputer(strategy='median')
            self.ram_imputer_.fit(X[['min_ram_gb']])
        
        return self
        
    def transform(self, X):
        df_out = X.copy()
        
        # List for the new features
        new_features = []
        
        # Applying MultiLabelBinarizer
        for col, mlb, prefix in [('categories', self.mlb_cat_, 'cat'), ('genres', self.mlb_genres_, 'genre'), 
                                 ('tags', self.mlb_tags_, 'tag'), ('languages', self.mlb_langs_, 'lang')]:
            if col in df_out.columns:
                encoded = pd.DataFrame(mlb.transform(df_out[col]), columns=[f"{prefix}_{c}" for c in mlb.classes_], index=df_out.index)
                new_features.append(encoded)
                
        # Analysis of creators
        if 'publishers' in df_out.columns:
            pub_cols = {f'pub_{str(p).replace(" ", "_")}': df_out['publishers'].apply(lambda x: 1 if p in x else 0) for p in self.top_publishers_}
            new_features.append(pd.DataFrame(pub_cols, index=df_out.index))
            
        if 'developers' in df_out.columns:
            dev_cols = {f'dev_{str(d).replace(" ", "_")}': df_out['developers'].apply(lambda x: 1 if d in x else 0) for d in self.top_developers_}
            new_features.append(pd.DataFrame(dev_cols, index=df_out.index))
            
        # Applying TF-IDF
        if 'short_description' in df_out.columns:
            tfidf_matrix = self.tfidf_.transform(df_out['short_description'].fillna(""))
            tfidf_cols = [f"tfidf_{w}" for w in self.tfidf_.get_feature_names_out()]
            df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols, index=df_out.index)
            new_features.append(df_tfidf)

        # Concat of the new features
        if new_features:
            df_out = pd.concat([df_out] + new_features, axis=1)

        # Handling NaN values
        if hasattr(self, 'ram_imputer_') and 'min_ram_gb' in df_out.columns:
            # NaN -> median calculated during the fit
            df_out['min_ram_gb'] = self.ram_imputer_.transform(df_out[['min_ram_gb']])
            
        if 'rec_ram_gb' in df_out.columns:
            df_out['rec_ram_gb'] = df_out['rec_ram_gb'].fillna(df_out['min_ram_gb'])
            
        # Dropping original textual columns
        cols_to_drop = ['categories', 'genres', 'tags', 'publishers', 'developers', 'languages', 'short_description']
        df_out = df_out.drop(columns=[c for c in cols_to_drop if c in df_out.columns], errors='ignore')
        
        return df_out.select_dtypes(include=['number', 'bool'])