"""Bet-shape scores, declaration gap, independent signals and tiers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine import config as C
from engine.network import neighbours


def cluster_scores(merchants: pd.DataFrame, clusters: dict, f: pd.DataFrame) -> pd.DataFrame:
    """Per-merchant cluster summary. A hidden betting cluster needs enough history to be trusted."""
    daily = f.attrs["daily"]
    rows = []
    for mid in merchants["merchant_id"]:
        if mid not in clusters:
            rows.append(dict(merchant_id=mid, max_cluster_score=0.0, n_clusters=0, hidden_cluster=False,
                             bet_cluster=None, bet_cluster_payer_share=0.0, bet_cluster_volume_share=0.0,
                             bet_first_day=None, bet_volume_share_since=0.0))
            continue
        cs = clusters[mid][0]
        top = cs.iloc[0]
        flagged = cs[cs["flagged"]]
        bet = cs[cs["score"] >= C.HIDDEN_SCORE]
        first = int(bet["first_day"].min()) if len(bet) else None
        since = (float(bet["volume_inr"].sum() / max(daily.loc[mid, first:].sum(), 1)) if first else 0.0)
        rows.append(dict(merchant_id=mid, max_cluster_score=float(top["score"]), n_clusters=len(cs),
                         hidden_cluster=len(flagged) > 0 and len(cs) > 1 and bool(f.at[mid, "sufficient"]),
                         bet_cluster=list(bet["cluster"]) if len(bet) else None,
                         bet_cluster_payer_share=float(bet["payer_share"].sum()),
                         bet_cluster_volume_share=float(bet["volume_share"].sum()),
                         bet_first_day=first, bet_volume_share_since=since))
    return pd.DataFrame(rows).set_index("merchant_id")


def declaration_gap(merchants: pd.DataFrame, f: pd.DataFrame, cs: pd.DataFrame) -> pd.DataFrame:
    m = merchants.set_index("merchant_id")
    out = pd.DataFrame(index=m.index)
    out["observed"] = np.maximum(cs["max_cluster_score"], f["merchant_score"])
    cohort_mask = ~m["persona_key"].isin(C.FRONT_PERSONAS)
    out["gap_z"] = np.nan
    out["cohort_median"] = np.nan
    for cat in C.CATEGORIES:
        in_cat = m["declared_category"] == cat
        base = out.loc[in_cat & cohort_mask, "observed"]
        med = base.median()
        mad = (base - med).abs().median()
        out.loc[in_cat, "cohort_median"] = med
        out.loc[in_cat, "gap_z"] = (out.loc[in_cat, "observed"] - med) / (mad * 1.4826 + C.GAP_EPS)
    out["contradicts"] = (out["gap_z"] > C.GAP_Z) & (out["observed"] >= C.GAP_MIN_OBSERVED) & f["sufficient"]

    # recategorisation suggestion
    sugg = []
    for mid in m.index:
        declared = m.at[mid, "declared_category"]
        dd = f.at[mid, f"catdist::{declared}"]
        betting = out.at[mid, "observed"] >= C.GAP_MIN_OBSERVED or cs.at[mid, "hidden_cluster"]
        s = None
        if not betting and pd.notna(dd):
            others = {c: f.at[mid, f"catdist::{c}"] for c in C.CATEGORIES if c != declared}
            best = min(others, key=others.get)
            if others[best] < C.RECAT_RATIO * dd:
                s = best
        sugg.append(s)
    out["recat_suggestion"] = sugg
    return out


def national_hits(merchants: pd.DataFrame, f: pd.DataFrame, lists: pd.DataFrame) -> pd.DataFrame:
    m = merchants.set_index("merchant_id")
    fri = set(lists.loc[lists["list"] == "fri_high_risk_phones", "value"])
    i4c = set(lists.loc[lists["list"] == "i4c_mule_accounts", "value"])
    out = pd.DataFrame(index=m.index)
    out["fri_phone"] = m["onboarding_phone_hash"].isin(fri)
    out["i4c_account"] = m["settlement_account_hash"].isin(i4c)
    out["blocked_domain"] = f["blocked_referrer"]
    out["any"] = out.any(axis=1)
    return out


def base_signals(f: pd.DataFrame, nat: pd.DataFrame) -> pd.DataFrame:
    s = pd.DataFrame(index=f.index)
    s["channel"] = ((f["s2s_undeclared_share"] > C.S2S_SHARE) | (f["link_spike"] > C.LINK_SPIKE)
                    | f["blocked_referrer"] | f["offshore_api_nondeclared"])
    s["matchday"] = f["matchday_z"].fillna(0) > C.MATCHDAY_Z
    s["national"] = nat["any"]
    s["network"] = False
    return s


def assign_tiers(f, cs, gap, sig, edges, confirmed: set, successors: pd.DataFrame, cleared: set = frozenset()):
    """Evaluate tiers in order 4 > 3 > 2 > 1, iterating the network-link signal to a fixed point.
    Merchants an analyst has cleared stay at Tier 1 and do not count as flagged neighbours."""
    ids = f.index
    succ = set(successors.loc[successors["flagged"], "merchant_id"]) if len(successors) else set()
    linked = edges[edges["linked"]]
    adj = {mid: set() for mid in ids}
    for a, b in zip(linked["a"], linked["b"]):
        adj[a].add(b)
        adj[b].add(a)
    suff = f["sufficient"]
    betting = (cs["hidden_cluster"] | (f["merchant_score"] >= C.MERCHANT_SCORE_FLAG)) & suff
    unusual = ((f["amount_z"].fillna(0) > C.TIER2_Z) | (f["hour_z"].fillna(0) > C.TIER2_Z)
               | (f["surge_rel"].fillna(0) > C.SURGE_RATIO) | gap["recat_suggestion"].notna()) & suff
    nat_insufficient = sig["national"] & ~suff & (f["active_days"] > 0)  # only once the merchant is trading
    sig = sig.copy()
    tiers = pd.Series(1, index=ids)
    reasons3 = pd.Series([[] for _ in ids], index=ids)
    for _ in range(6):
        prev = tiers.copy()
        new = pd.Series(1, index=ids)
        why = {}
        for mid in ids:
            if mid in confirmed:
                new[mid] = 4
                continue
            n_sig = int(sig.loc[mid, ["channel", "matchday", "network", "national"]].sum())
            linked_t4 = any(n in confirmed for n in adj[mid])
            r = []
            if betting[mid] and gap.at[mid, "contradicts"] and n_sig >= 1:
                r.append("betting_shape")
            if linked_t4:
                r.append("linked_confirmed")
            if mid in succ:
                r.append("successor")
            if nat_insufficient[mid]:
                r.append("national_new")
            if mid in cleared:
                new[mid] = 1
            elif r:
                new[mid] = 3
                why[mid] = r
            elif unusual[mid]:
                new[mid] = 2
        tiers = new
        reasons3 = pd.Series([why.get(mid, []) for mid in ids], index=ids)
        hot = set(tiers[tiers >= 3].index)
        sig["network"] = [bool(adj[mid] & hot) for mid in ids]
        if (tiers == prev).all() and _ > 0:
            break
    return tiers, sig, reasons3, unusual


def linked_to(edges: pd.DataFrame, mid: str, targets: set) -> list[str]:
    return [o for o in neighbours(edges, mid)["other"] if o in targets]
