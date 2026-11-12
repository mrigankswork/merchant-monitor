"""Hidden-cluster detection: Gaussian mixture over payer-level features, per merchant."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from engine import config as C
from engine.features import bet_shape_score, shape_inputs


def _fit(X: np.ndarray) -> np.ndarray:
    best, best_bic = None, np.inf
    for k in C.GMM_K:
        if k > 1 and len(X) < 10 * k:
            break
        gm = GaussianMixture(n_components=k, covariance_type="full", reg_covar=1e-3,
                             random_state=C.SEED, n_init=1).fit(X)
        bic = gm.bic(X)
        if bic < best_bic:
            best, best_bic = gm, bic
    return best.predict(X)


def _merge_small(X: np.ndarray, labels: np.ndarray, protect) -> np.ndarray:
    """Merge clusters under 5% of payers into their nearest neighbour, unless `protect(mask)` says the group
    carries outsized money (a small group moving a lot of volume is exactly the risk)."""
    labels = labels.copy()
    kept = set()
    while True:
        ids, counts = np.unique(labels, return_counts=True)
        small = [k for k, c in zip(ids, counts) if c < C.MIN_CLUSTER_SHARE * len(labels) and k not in kept]
        if not small or len(ids) == 1:
            break
        s = min(small, key=lambda k: (labels == k).sum())
        if protect(labels == s):
            kept.add(s)
            continue
        keep = ids[ids != s]
        cents = np.array([X[labels == k].mean(axis=0) for k in keep])
        idx = np.where(labels == s)[0]
        dist = ((X[idx, None, :] - cents[None, :, :]) ** 2).sum(-1)
        labels[idx] = keep[dist.argmin(1)]
    _, labels = np.unique(labels, return_inverse=True)
    return labels


def _share_since(amount: float, first_day: int, daily: pd.Series | None) -> float:
    if daily is None:
        return 0.0
    return float(amount / max(daily.loc[first_day:].sum(), 1.0))


def cluster_merchant(pf_m: pd.DataFrame, daily: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (cluster summary, per-payer frame with cluster + PCA coordinates)."""
    pf_m = pf_m.reset_index(drop=True)
    X = pf_m[C.CLUSTER_FEATURES].to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)
    amt, first = pf_m["total_amount"].to_numpy(), pf_m["first_day"].to_numpy()

    def protect(mask):
        return mask.sum() >= 5 and _share_since(amt[mask].sum(), int(first[mask].min()), daily) >= C.HIDDEN_VOL_SHARE

    if len(pf_m) >= C.MIN_PAYERS:
        labels = _merge_small(Xs, _fit(Xs), protect)
    else:
        labels = np.zeros(len(pf_m), dtype=int)
    if len(pf_m) >= 3:
        xy = PCA(n_components=2, random_state=C.SEED).fit_transform(Xs)
    else:
        xy = np.zeros((len(pf_m), 2))
    payers = pf_m.assign(cluster=labels, pc1=xy[:, 0], pc2=xy[:, 1])

    total_amt = payers["total_amount"].sum()
    rows = []
    for k, g in payers.groupby("cluster"):
        inputs = shape_inputs(g["txns_per_active_day"].mean(), g["round_share"].mean(),
                              g["escalation_share"].mean(), g["prematch_share"].mean(),
                              g["night_share"].mean(), g["remote_flag"].mean())
        rows.append(dict(cluster=int(k), first_day=int(g["first_day"].min()),
                         payer_count=len(g), payer_share=len(g) / len(payers),
                         volume_share=g["total_amount"].sum() / max(total_amt, 1), volume_inr=g["total_amount"].sum(),
                         volume_share_since=_share_since(g["total_amount"].sum(), int(g["first_day"].min()), daily),
                         score=bet_shape_score(inputs),
                         **{f"mean_{c}": g[c].mean() for c in C.CLUSTER_FEATURES},
                         mean_amount=g["mean_amount"].mean()))
    summary = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    # big enough to matter: 15% of volume (over the window, or since the group first appeared) or 10% of payers
    summary["flag_size"] = ((summary["volume_share"] >= C.HIDDEN_VOL_SHARE)
                            | (summary["volume_share_since"] >= C.HIDDEN_VOL_SHARE)
                            | (summary["payer_share"] >= C.HIDDEN_PAYER_SHARE))
    summary["flagged"] = (summary["score"] >= C.HIDDEN_SCORE) & summary["flag_size"]
    return summary, payers


def cluster_all(pf: pd.DataFrame, daily: pd.DataFrame | None = None) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    return {mid: cluster_merchant(g, None if daily is None else daily.loc[mid])
            for mid, g in pf.groupby("merchant_id", sort=True)}
