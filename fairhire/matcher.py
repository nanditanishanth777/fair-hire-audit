"""The resume matcher under audit: TF-IDF + logistic regression.

Any object with ``score(list_of_texts) -> array_of_floats`` (higher = better
match) can be audited; this class is the reference implementation, trained on
the biased historical decisions from ``data.historical_labels``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class ResumeMatcher:
    def __init__(self, C: float = 2.0, seed: int = 0):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        self.clf = LogisticRegression(C=C, max_iter=1000, random_state=seed)

    def fit(self, texts, labels) -> "ResumeMatcher":
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        return self

    def score(self, texts) -> np.ndarray:
        return self.clf.predict_proba(self.vectorizer.transform(texts))[:, 1]

    def token_weights(self, tokens) -> pd.DataFrame:
        """Learned coefficient for each unigram in ``tokens`` that is in the vocabulary."""
        vocab = self.vectorizer.vocabulary_
        coef = self.clf.coef_[0]
        rows = [(t, float(coef[vocab[t]])) for t in tokens if t in vocab]
        return pd.DataFrame(rows, columns=["token", "weight"]).sort_values("weight", ascending=False)
