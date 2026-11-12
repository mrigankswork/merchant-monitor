"""Runs the full engine for a given simulated day and demo state."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from engine import config as C
from engine.clustering import cluster_all
from engine.features import enrich, merchant_features, payer_features
from engine.generate import load_or_generate
from engine.network import build_edges, detect_successors
from engine.scoring import (assign_tiers, base_signals, cluster_scores, declaration_gap, national_hits)


@dataclass
class Result:
    current_day: int
    confirmed: set
    suspended: dict
    data: dict
    tx: pd.DataFrame
    pf: pd.DataFrame
    f: pd.DataFrame
    clusters: dict
    cs: pd.DataFrame
    gap: pd.DataFrame
    nat: pd.DataFrame
    sig: pd.DataFrame
    edges: pd.DataFrame
    successors: pd.DataFrame
    tiers: pd.Series
    tier3_why: pd.Series
    unusual: pd.Series
    extra: dict = field(default_factory=dict)

    @property
    def merchants(self) -> pd.DataFrame:
        return self.data["merchants"]

    def pid(self, key: str) -> str:
        return f"M-{key}"


def run(current_day: int = C.DEFAULT_DAY, confirmed=(), suspended=None, data=None, cleared=()) -> Result:
    """confirmed: persona keys or merchant ids analyst-confirmed as fronts.
    suspended: {merchant_id: suspension day}. cleared: merchants an analyst marked as false positives."""
    data = data or load_or_generate()
    confirmed = {c if c.startswith("M-") else f"M-{c}" for c in confirmed}
    suspended = {(k if k.startswith("M-") else f"M-{k}"): v for k, v in (suspended or {}).items()}
    cleared = {c if c.startswith("M-") else f"M-{c}" for c in cleared}
    merchants = data["merchants"]
    tx = data["transactions"]
    tx = tx[tx["day"] <= current_day]
    t = enrich(tx, merchants, data["payers"])
    api = data["api_logs"]
    api = api[api["day"] <= current_day]
    logins = data["dashboard_logins"]
    logins = logins[logins["ts"] < pd.Timestamp(C.DAY1) + pd.Timedelta(days=current_day)]
    payouts = data["payouts"]
    payouts = payouts[payouts["ts"] < pd.Timestamp(C.DAY1) + pd.Timedelta(days=current_day)]

    pf = payer_features(t)
    f = merchant_features(t, pf, merchants, api, payouts, current_day)
    clusters = cluster_all(pf, f.attrs["daily"])
    cs = cluster_scores(merchants, clusters, f)
    gap = declaration_gap(merchants, f, cs)
    nat = national_hits(merchants, f, data["national_lists"])
    sig = base_signals(f, nat)
    edges = build_edges(pf, merchants, logins, api)

    bet_payers = {}  # X's flagged cluster (formal successor rule)
    for x in suspended:
        summ, payers = clusters.get(x, (None, None))
        if summ is not None:
            flagged = summ[summ["flagged"]]
            bet_payers[x] = set(payers.loc[payers["cluster"].isin(flagged["cluster"]), "payer_id"])
    successors = detect_successors(t, suspended, bet_payers)
    model_tiers = assign_tiers(f, cs, gap, sig, edges, confirmed | set(suspended), successors)[0]
    tiers, sig, why, unusual = assign_tiers(f, cs, gap, sig, edges, confirmed | set(suspended), successors, cleared)
    r = Result(current_day, confirmed, suspended, data, t, pf, f, clusters, cs, gap, nat, sig, edges,
               successors, tiers, why, unusual)
    r.extra.update(model_tiers=model_tiers, cleared=cleared)
    return r
