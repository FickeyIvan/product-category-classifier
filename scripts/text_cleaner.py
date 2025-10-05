# scripts/text_cleaner.py

import re
from sklearn.base import BaseEstimator, TransformerMixin

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(lambda x: re.sub(r"[^a-zA-Z0-9 ]", "", str(x).lower()))
