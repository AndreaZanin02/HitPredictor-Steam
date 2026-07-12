"""
Centralized module for preprocessing and feature encoding
Contains classes and functions that can be reused in tuning and training

This module transforms a cleaned dataset into ML-ready numerical features
using:
- MultiLabelBinarizer (categorical multi-label features)
- TF-IDF (short text descriptions)

Offers dynamic undersampling tools for reducing the size of class 0 and
SMOTENC to oversample the minority classes 3 and 4
"""

import pandas as pd
import numpy as np
import torch
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTENC
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

class CorrelationRemover(BaseEstimator, TransformerMixin):
    """
    Custom transformer that drops highly correlated numeric features
    Only one column of each highly correlated pair is removed
    """
    def __init__(self, threshold=0.97):
        self.threshold = threshold

    def fit(self, X, y=None):
        """
        Compute the absolute correlation matrix on the training fold
        and mark for removal any column that has a correlation above
        threshold with at least one other column appearing before it
        """
        corr_matrix = X.corr().abs()
        # Keep only the upper triangle (excluding the diagonal) so each
        # correlated pair is evaluated once, not twice
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_ = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self

    def transform(self, X):
        """
        Drop the columns identified during fit(). Uses errors='ignore'
        so it doesn't break if a column is already missing (e.g. removed
        by an earlier pipeline step)
        """
        return X.drop(columns=self.to_drop_, errors='ignore')


def dynamic_undersample(y):
    """
    Sampling-strategy callback for RandomUnderSampler.
    Computes, for the current fold/split, how many samples Class 0
    (the dominant "long-tail" class) should be reduced to.
    Class 0 is capped at 25% of its original volume, but never pushed
    below the size of Class 1, so it stays the majority class without
    becoming distorted relative to the other tiers.
    Returns a dict compatible with imblearn's sampling_strategy callable
    interface: {class_label: target_count}
    """
    import pandas as pd

    # Count how many samples belong to each class in this fold
    counts = pd.Series(y).value_counts().to_dict()

    # Reduce Class 0 to 25% of its current volume
    target_class_0 = int(counts[0] * 0.25)

    # Safety floor: never undersample Class 0 below Class 1's size
    safe_target = max(target_class_0, counts.get(1, 0))

    return {0: safe_target}


class DynamicSMOTENC(BaseEstimator):
    """
    Wrapper around imblearn's SMOTENC that auto-detects which columns are
    categorical/binary at resample time, instead of requiring a fixed list
    of column indices.
    This is necessary because SteamFeatureExtractor produces a different
    number/order of one-hot, tag, and embedding-derived columns on every
    fold, so a static categorical_features list would silently break.
    """

    def __init__(self, sampling_strategy='auto', k_neighbors=5, random_state=None):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Required by the imblearn Pipeline interface. Samplers don't learn
        parameters during fit(); resampling only happens in fit_resample(),
        so this just returns self as a no-op
        """
        return self

    def fit_resample(self, X, y):
        """
        Detect categorical/binary columns on the fly and run SMOTENC.
        A column is treated as categorical if it has 2 or fewer unique
        values (covers one-hot/binary flags produced upstream); everything
        else (numeric/continuous, including PLS-DA and embedding columns)
        is treated as continuous.
        Returns the resampled (X, y), preserving the DataFrame type/columns
        if the input was a DataFrame
        """
        is_df = isinstance(X, pd.DataFrame)
        columns = X.columns if is_df else None

        cat_mask = [X[col].nunique() <= 2 for col in X.columns] if is_df else None

        smote = SMOTENC(
            categorical_features=cat_mask,
            sampling_strategy=self.sampling_strategy,
            k_neighbors=self.k_neighbors,
            random_state=self.random_state
        )

        # Temporarily suspend the global rule transform_output="pandas"
        # to allow SMOTENC to handle sparse matrices internally
        with sklearn.config_context(transform_output="default"):
            X_res, y_res = smote.fit_resample(X, y)

        if is_df and not isinstance(X_res, pd.DataFrame):
            X_res = pd.DataFrame(X_res, columns=columns)

        return X_res, y_res


def dynamic_oversample(y):
    """
    Sampling strategy callback for SMOTENC oversampling.
    Boosts Class 4 (the rarest tier) up to 50% of Class 3's volume in the
    current fold, using Class 3 as an anchor since it's the closest more
    populated neighbor tier. If Class 3 is missing from the fold, Class 4's
    own count is used as the reference (no-op in that case)
    """
    counts = pd.Series(y).value_counts().to_dict()
    strategy = {}

    if 4 in counts:
        # Use Class 3's volume as the anchor, or Class 4's own volume
        # if Class 3 doesn't exist in this fold
        base_reference = counts.get(3, counts[4])

        # Boost Class 4 up to 50% of Class 3's volume, never below its
        # current count
        target_4 = max(counts[4], int(base_reference * 0.5))

        # Only add to the strategy if there's an actual increase
        if target_4 > counts[4]:
            strategy[4] = target_4

    return strategy


class FeatureNameSanitizer(BaseEstimator, TransformerMixin):
    """
    Renames DataFrame columns to remove characters ([, ], <) that XGBoost
    reject in feature names.
    Column names can contain these characters after one-hot encoding or
    multi-label binarization (e.g. tag/category values with brackets),
    so this step must run after those transformers
    """
    def fit(self, X, y=None):
        """
        No-op fit: sanitizing column names doesn't require any fitted
        state, it only needs to be applied at transform time
        """
        return self

    def transform(self, X):
        """
        Replace any occurrence of [, ], or < in column names with '_'
        """
        if isinstance(X, pd.DataFrame):
            X_clean = X.copy()
            # Replace problematic characters with an underscore
            X_clean.columns = X_clean.columns.str.replace(r'[\[\]<]', '_', regex=True)
            return X_clean
        return X


def precompute_detailed_embeddings(df, text_col='detailed_description'):
    """
    Encode a text column into dense sentence embeddings using a frozen
    pretrained SentenceTransformer (all-mpnet-base-v2), and append them
    as new numeric columns.
    Run once upstream of the CV/pipeline (not inside a fold-fitted step)
    because the encoder is frozen (no training happens on this data), so
    precomputing it does not leak information between folds, it only
    avoids paying the embedding cost repeatedly for every fold/trial
    """
    print("Pre-compuation of textual embeddings...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-mpnet-base-v2', device=device)

    texts = df[text_col].fillna("").tolist()
    embeddings = model.encode(texts, show_progress_bar=True)

    emb_cols = [f"raw_emb_{i}" for i in range(embeddings.shape[1])]
    df_emb = pd.DataFrame(embeddings, columns=emb_cols, index=df.index)

    # Free VRAM once encoding is done, the model isn't needed anymore
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    return pd.concat([df.drop(columns=[text_col], errors='ignore'), df_emb], axis=1), emb_cols


def get_base_preprocessor(numeric_cols, categorical_cols):
    """
    Constructs the baseline preprocessing pipeline for numeric and categorical features
    """
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', RobustScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
    ])
    
    base_preprocessor = ColumnTransformer(
        transformers=[
            ('num_pipeline', numeric_transformer, numeric_cols),
            ('cat_pipeline', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    return base_preprocessor


class SteamFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer that turns the raw categorical/text Steam columns
    into numeric ML features:
    -   categories / genres / tags / languages -> multi-hot encoding
        (MultiLabelBinarizer)
    -   publishers / developers -> binary flags for the top-N most
        frequent creators seen in the training fold
    -    short_description -> TF-IDF vector (1-2 grams)
    All binarizers/vectorizers/imputers are fit only on the data passed
    to fit(), so this must be placed inside the CV pipeline (not applied
    upfront) to avoid leaking fold-specific vocabulary/statistics.
    transform() returns only numeric/boolean columns: any remaining
    non-numeric column (e.g. free text that wasn't vectorized) is
    silently dropped by the final select_dtypes call
    """
    def __init__(self, top_n_creators=50, max_tfidf_features=30):
        self.top_n_creators = top_n_creators
        self.max_tfidf_features = max_tfidf_features

    def fit(self, X, y=None):
        """
        Fit one MultiLabelBinarizer per multi-label column, determine the
        top-N most frequent publishers/developers, fit the TF-IDF
        vectorizer on short_description, and fit a median imputer on
        min_ram_gb — all restricted to the current training fold X
        """
        self.mlb_cat_ = MultiLabelBinarizer().fit(X['categories'].tolist() if 'categories' in X.columns else [])
        self.mlb_genres_ = MultiLabelBinarizer().fit(X['genres'].tolist() if 'genres' in X.columns else [])
        self.mlb_tags_ = MultiLabelBinarizer().fit(X['tags'].tolist() if 'tags' in X.columns else [])
        self.mlb_langs_ = MultiLabelBinarizer().fit(X['languages'].tolist() if 'languages' in X.columns else [])

        self.top_publishers_ = X['publishers'].explode().dropna().value_counts().head(self.top_n_creators).index.tolist() if 'publishers' in X.columns else []
        self.top_developers_ = X['developers'].explode().dropna().value_counts().head(self.top_n_creators).index.tolist() if 'developers' in X.columns else []

        # TF-IDF fit only on the short_description of this specific fold
        short_desc = X['short_description'].fillna("") if 'short_description' in X.columns else pd.Series([""]*len(X))
        self.tfidf_ = TfidfVectorizer(max_features=self.max_tfidf_features, stop_words='english', ngram_range=(1, 2))
        self.tfidf_.fit(short_desc)

        # Impute RAM NaNs using the median of the training set only (no data leakage)
        if 'min_ram_gb' in X.columns:
            self.ram_imputer_ = SimpleImputer(strategy='median')
            self.ram_imputer_.fit(X[['min_ram_gb']])

        return self

    def transform(self, X):
        """
        Apply the fitted encoders/vectorizer/imputer to X and append
        the resulting numeric columns, then drop the original
        categorical/text columns
        """
        df_out = X.copy()

        # Collect new feature blocks here before concatenating once
        new_features = []

        # Multi-hot encode categories/genres/tags/languages
        for col, mlb, prefix in [('categories', self.mlb_cat_, 'cat'), ('genres', self.mlb_genres_, 'genre'),
                                 ('tags', self.mlb_tags_, 'tag'), ('languages', self.mlb_langs_, 'lang')]:
            if col in df_out.columns:
                encoded = pd.DataFrame(mlb.transform(df_out[col]), columns=[f"{prefix}_{c}" for c in mlb.classes_], index=df_out.index)
                new_features.append(encoded)

        # Binary flag per top-N publisher/developer (1 if present in the game's list)
        if 'publishers' in df_out.columns:
            pub_cols = {f'pub_{str(p).replace(" ", "_")}': df_out['publishers'].apply(lambda x: 1 if p in x else 0) for p in self.top_publishers_}
            new_features.append(pd.DataFrame(pub_cols, index=df_out.index))

        if 'developers' in df_out.columns:
            dev_cols = {f'dev_{str(d).replace(" ", "_")}': df_out['developers'].apply(lambda x: 1 if d in x else 0) for d in self.top_developers_}
            new_features.append(pd.DataFrame(dev_cols, index=df_out.index))

        # TF-IDF on short_description
        if 'short_description' in df_out.columns:
            tfidf_matrix = self.tfidf_.transform(df_out['short_description'].fillna(""))
            tfidf_cols = [f"tfidf_{w.replace(' ', '_')}" for w in self.tfidf_.get_feature_names_out()]
            df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols, index=df_out.index)
            new_features.append(df_tfidf)

        # Concatenate all new feature blocks at once
        if new_features:
            df_out = pd.concat([df_out] + new_features, axis=1)

        # Impute min_ram_gb with the fold's training median, then backfill
        # rec_ram_gb from min_ram_gb where still missing
        if hasattr(self, 'ram_imputer_') and 'min_ram_gb' in df_out.columns:
            df_out['min_ram_gb'] = self.ram_imputer_.transform(df_out[['min_ram_gb']])

        if 'rec_ram_gb' in df_out.columns:
            df_out['rec_ram_gb'] = df_out['rec_ram_gb'].fillna(df_out['min_ram_gb'])

        # Drop the original categorical/text columns, now redundant
        cols_to_drop = ['categories', 'genres', 'tags', 'publishers', 'developers', 'languages', 'short_description']
        df_out = df_out.drop(columns=[c for c in cols_to_drop if c in df_out.columns], errors='ignore')

        return df_out.select_dtypes(include=['number', 'bool'])
    
class SupervisedPLSDATransformer(BaseEstimator, TransformerMixin):
    """
    Supervised dimensionality reduction (PLS-DA) applied only to the raw
    sentence-embedding columns (emb_cols), while leaving every other
    already-processed feature (numeric, one-hot, multi-hot, TF-IDF, ...)
    untouched and passed through unchanged.
    The target y is one-hot encoded and used as the response matrix for
    PLSRegression, so the embedding space is projected onto components
    that are discriminative for the target classes (unlike unsupervised
    PCA). Must be fit only on the training fold to avoid leakage
    """
    def __init__(self, emb_cols, n_components=50):
        self.emb_cols = emb_cols
        self.n_components = n_components

    def fit(self, X, y=None):
        """
        Fit the PLS regression on the embedding columns against the
        one-hot encoded target. Raises if y is missing (PLS-DA is a
        supervised method) or if none of `emb_cols` are found in X
        """
        if y is None:
            raise ValueError("PLS-DA needs target y for training")

        self.pls = PLSRegression(n_components=self.n_components)
        self.ohe = OneHotEncoder(sparse_output=False)

        # Restrict to the embedding columns actually present in this fold
        self._transformed_cols = [col for col in self.emb_cols if col in X.columns]

        if not self._transformed_cols:
            raise ValueError("No embeddings column found for PLS-DA.")

        X_target = X[self._transformed_cols]
        y_encoded = self.ohe.fit_transform(np.array(y).reshape(-1, 1))

        self.pls.fit(X_target, y_encoded)
        return self

    def transform(self, X):
        """
        Project the embedding columns into n_components PLS-DA
        components and concatenate them back with all the other
        (non-embedding) columns, which are passed through unchanged
        """
        X_target = X[self._transformed_cols]
        X_remainder = X.drop(columns=self._transformed_cols)

        X_trans = self.pls.transform(X_target)

        if isinstance(X_trans, pd.DataFrame):
            X_trans = X_trans.to_numpy()

        pls_col_names = [f"pls_da_{i}" for i in range(self.n_components)]
        df_pls = pd.DataFrame(X_trans, columns=pls_col_names, index=X.index)

        return pd.concat([X_remainder, df_pls], axis=1)