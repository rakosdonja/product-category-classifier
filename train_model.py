"""
train_model.py

Train a product category classifier based on product titles (+ simple engineered features)
and save the trained pipeline to: model/product_category_model.pkl
"""

import os
import sys
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# ----------------------------
# Helper functions
# ----------------------------
def compute_title_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered numeric features from product_title.
    Returns a DataFrame with:
      - product_title (text)
      - title_length
      - title_word_count
      - title_digit_count
      - title_special_char_count
    """
    out = df.copy()

    # Ensure string
    title = out["product_title"].astype(str).fillna("")

    out["title_length"] = title.str.len()
    out["title_word_count"] = title.str.split().apply(len)
    out["title_digit_count"] = title.apply(lambda x: sum(ch.isdigit() for ch in x))

    # Count special characters (anything that is not alphanumeric or whitespace)
    out["title_special_char_count"] = title.apply(
        lambda x: len(re.findall(r"[^A-Za-z0-9\s]", x))
    )

    return out[
        [
            "product_title",
            "title_length",
            "title_word_count",
            "title_digit_count",
            "title_special_char_count",
        ]
    ]


def standardize_category_labels(series: pd.Series) -> pd.Series:
    """
    Standardize category labels to reduce duplicates like:
      'CPU' -> 'cpus'
      'cpu' -> 'cpus'
      'Mobile Phone' -> 'mobile phones'
      'fridge' -> 'fridges'
    Everything lowercased + stripped.
    """
    s = series.astype(str).str.lower().str.strip()

    # Map known singular/variants -> canonical labels
    canonical_map = {
        "cpu": "cpus",
        "cpus": "cpus",
        "mobile phone": "mobile phones",
        "mobile phones": "mobile phones",
        "fridge": "fridges",
        "fridges": "fridges",
        "tv": "tvs",
        "tvs": "tvs",
        "dishwasher": "dishwashers",
        "dishwashers": "dishwashers",
        "freezer": "freezers",
        "freezers": "freezers",
        "microwave": "microwaves",
        "microwaves": "microwaves",
        "digital camera": "digital cameras",
        "digital cameras": "digital cameras",
        "washing machine": "washing machines",
        "washing machines": "washing machines",
        "fridge freezer": "fridge freezers",
        "fridge freezers": "fridge freezers",
    }

    s = s.replace(canonical_map)

    return s


def main():
    try:
        # ----------------------------
        # Paths
        # ----------------------------
        data_path = os.path.join("data", "products.csv")
        model_dir = "model"
        model_path = os.path.join(model_dir, "product_category_model.pkl")

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"CSV not found at '{data_path}'. Make sure data/products.csv exists."
            )

        # ----------------------------
        # Load data
        # ----------------------------
        df = pd.read_csv(data_path)

        # Standardize raw column names (strip spaces)
        df.columns = [c.strip() for c in df.columns]

        # Rename columns from CSV to our internal standard
        rename_map = {
            "Product Title": "product_title",
            "Category Label": "category_label",
        }
        df = df.rename(columns=rename_map)

        # Validate required columns
        required_cols = ["product_title", "category_label"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")

        # ----------------------------
        # Basic cleaning
        # ----------------------------
        df = df.dropna(subset=["product_title", "category_label"]).copy()

        # Standardize target labels
        df["category_label"] = standardize_category_labels(df["category_label"])

        # Drop any remaining invalid labels (e.g. "nan" strings after astype(str))
        df = df[df["category_label"].notna()]
        df = df[df["category_label"] != "nan"]

        # ----------------------------
        # Define X and y
        # ----------------------------
        X = df[["product_title"]].copy()  # we will generate extra features inside pipeline
        y = df["category_label"].copy()

        # ----------------------------
        # Preprocessing + Model
        # ----------------------------
        feature_builder = FunctionTransformer(compute_title_features, validate=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ("title_tfidf", TfidfVectorizer(), "product_title"),
                ("length", "passthrough", ["title_length"]),
                ("word_count", "passthrough", ["title_word_count"]),
                ("digit_count", "passthrough", ["title_digit_count"]),
                ("special_char_count", "passthrough", ["title_special_char_count"]),
            ],
            remainder="drop",
        )

        # Best model from your notebook summary
        clf = LinearSVC()

        pipeline = Pipeline(
            steps=[
                ("feature_engineering", feature_builder),
                ("preprocessing", preprocessor),
                ("classifier", clf),
            ]
        )

        # ----------------------------
        # Optional: quick evaluation split (sanity check)
        # (You can remove this block if you want pure full-data training only.)
        # ----------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        acc = accuracy_score(y_test, preds)
        print(f"\n✅ Sanity-check accuracy (80/20 split): {acc:.4f}\n")
        print("Classification report:")
        print(classification_report(y_test, preds, digits=4))

        # ----------------------------
        # Train final model on ALL data
        # ----------------------------
        pipeline.fit(X, y)

        # ----------------------------
        # Save model
        # ----------------------------
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(pipeline, model_path)

        print(f"\n✅ Model trained on full dataset and saved to: {model_path}\n")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

