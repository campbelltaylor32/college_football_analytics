"""Shared preprocessing pipeline for the candidate regressors. Tree models ignore scaling,
but sharing one ColumnTransformer across all models is simpler than branching per model
family. SimpleImputer is a defensive final layer -- cleaning.py already imputes at the
dataset level, this guards against a future feature addition that skips it. Ported verbatim
from the sibling cfb_win_total_model project."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessing_pipeline(feature_cols: list[str], categorical_cols: list[str] | None = None) -> ColumnTransformer:
    categorical_cols = categorical_cols or []
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    numeric_pipeline = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    transformers = [("num", numeric_pipeline, numeric_cols)]
    if categorical_cols:
        categorical_pipeline = Pipeline(
            [("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]
        )
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers)
