"""Stage 3: fold-local correlation-cluster pruning.

Fit on a single walk-forward fold's training rows only. Hierarchically clusters features
by |Spearman correlation| and keeps one representative per cluster - the member most
correlated with the training-fold target, not just the first alphabetically or the
highest-variance one, so the kept feature is at least plausibly informative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def prune_correlated_features(
    X_train: pd.DataFrame, y_train: pd.Series, correlation_threshold: float = 0.90
) -> tuple[list[str], dict]:
    features = list(X_train.columns)
    if len(features) <= 1:
        return features, {"n_clusters": len(features), "clusters": []}

    corr = X_train.corr(method="spearman").abs().fillna(0.0)
    dist = 1.0 - corr.to_numpy()
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)
    dist = (dist + dist.T) / 2.0  # enforce exact symmetry against float round-off
    condensed = squareform(dist, checks=False)

    Z = linkage(condensed, method="average")
    cluster_ids = fcluster(Z, t=1.0 - correlation_threshold, criterion="distance")

    y = y_train.to_numpy(dtype=float)
    target_corr = {}
    for col in features:
        x = X_train[col].to_numpy(dtype=float)
        target_corr[col] = abs(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else 0.0

    clusters: dict[int, list[str]] = {}
    for feat, cid in zip(features, cluster_ids):
        clusters.setdefault(int(cid), []).append(feat)

    kept, report_rows = [], []
    for cid, feats in clusters.items():
        rep = max(feats, key=lambda f: target_corr.get(f, 0.0))
        kept.append(rep)
        report_rows.append(
            {"cluster_id": cid, "size": len(feats), "representative": rep, "members": feats}
        )

    return sorted(kept), {"n_clusters": len(clusters), "clusters": report_rows}
