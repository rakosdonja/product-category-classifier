# feature_utils.py
import pandas as pd
import re

def compute_title_features(X):
    """
    Accepts a DataFrame with a 'product_title' column and returns a DataFrame with:
    - product_title
    - title_length
    - title_word_count
    - title_digit_count
    - title_special_char_count
    """
    if isinstance(X, pd.Series):
        X = X.to_frame()

    if "product_title" not in X.columns:
        raise ValueError("Missing required column: 'product_title'")

    titles = X["product_title"].fillna("").astype(str)

    out = pd.DataFrame({
        "product_title": titles,
        "title_length": titles.str.len(),
        "title_word_count": titles.str.split().str.len(),
        "title_digit_count": titles.str.count(r"\d"),
        "title_special_char_count": titles.apply(lambda s: len(re.findall(r"[^a-zA-Z0-9\s]", s))),
    })

    return out
