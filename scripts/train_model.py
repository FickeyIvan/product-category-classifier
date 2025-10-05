# scripts/train_model.py

import pandas as pd
import pickle
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
from scripts.text_cleaner import TextCleaner


DATA_PATH = os.path.join(BASE_DIR, "data", "products.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

df = df.dropna(subset=["product_title", "category_label"])
df = df.drop_duplicates(subset=["product_title", "category_label"])

df["title_length"] = df["product_title"].apply(lambda x: len(str(x).split()))
df["digit_count"] = df["product_title"].apply(lambda x: sum(c.isdigit() for c in str(x)))

X = df[["product_title", "title_length", "digit_count"]]
y = df["category_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("cleaner", TextCleaner()),
    ("tfidf", TfidfVectorizer(max_features=1000)),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline