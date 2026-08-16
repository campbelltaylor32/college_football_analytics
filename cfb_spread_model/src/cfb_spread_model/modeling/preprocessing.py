"""Shared preprocessing pipeline for the candidate classifiers. Tree models ignore scaling, but
sharing one ColumnTransformer across all models is simpler than branching per model family.
SimpleImputer is a defensive final layer -- the upstream R pipeline's na.omit() and this
project's cleaning.py should already guarantee no NA reaches here, this guards against a future
feature addition that skips both."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessing_pipeline(feature_cols: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    return ColumnTransformer([("num", numeric_pipeline, feature_cols)])
