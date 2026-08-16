"""Stage 5 (alternative reducer, compared head-to-head with stages 3+4, not chained after
them): collapse each semantically-related stat family (e.g. every Offense_* EPA/success-rate
column) into a handful of principal components, fit on a single fold's training rows only.
"""
from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def fit_pca_reducer(
    X_train: pd.DataFrame,
    family_prefixes: list[str],
    variance_retained: float = 0.90,
    random_state: int = 42,
) -> dict:
    matched_cols = [c for c in X_train.columns if any(fp in c for fp in family_prefixes)]
    other_cols = [c for c in X_train.columns if c not in matched_cols]

    if len(matched_cols) < 2:
        return {
            "scaler": None,
            "pca": None,
            "matched_cols": matched_cols,
            "other_cols": X_train.columns.tolist(),
            "n_components": 0,
        }

    scaler = StandardScaler().fit(X_train[matched_cols])
    X_scaled = scaler.transform(X_train[matched_cols])
    pca = PCA(n_components=variance_retained, svd_solver="full", random_state=random_state)
    pca.fit(X_scaled)

    return {
        "scaler": scaler,
        "pca": pca,
        "matched_cols": matched_cols,
        "other_cols": other_cols,
        "n_components": pca.n_components_,
    }


def apply_pca_reducer(reducer: dict, X: pd.DataFrame) -> pd.DataFrame:
    other = X[reducer["other_cols"]].reset_index(drop=True)
    if reducer["pca"] is None:
        return other

    X_scaled = reducer["scaler"].transform(X[reducer["matched_cols"]])
    components = reducer["pca"].transform(X_scaled)
    comp_cols = [f"pca_family_{i}" for i in range(components.shape[1])]
    comp_df = pd.DataFrame(components, columns=comp_cols, index=X.index).reset_index(drop=True)
    return pd.concat([other, comp_df], axis=1)
