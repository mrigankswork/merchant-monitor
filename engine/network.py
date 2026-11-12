"""Merchant graph: edge weighting, propagation and successor detection."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from engine import config as C

EDGE_TYPES = ["payers", "device", "settlement", "director", "ip_fixed", "ip_cloud", "ip_carrier", "server_ip"]


def _pairs_from_groups(groups: pd.Series, idx: dict) -> set[tuple[int, int]]:
    """groups: Series key -> list of merchant ids sharing that key."""
    out = set()
    for mids in groups:
        ms = sorted({idx[m] for m in mids})
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                out.add((ms[i], ms[j]))
    return out


def build_edges(pf: pd.DataFrame, merchants: pd.DataFrame, logins: pd.DataFrame, api: pd.DataFrame) -> pd.DataFrame:
    ids = list(merchants["merchant_id"])
    idx = {m: i for i, m in enumerate(ids)}
    n = len(ids)
    W = {t: np.zeros((n, n)) for t in EDGE_TYPES}
    shared_cnt = np.zeros((n, n))

    # shared payers, idf-weighted
    payer_codes, payer_index = pd.factorize(pf["payer_id"])
    rows = pf["merchant_id"].map(idx).to_numpy()
    B = sparse.csr_matrix((np.ones(len(pf)), (rows, payer_codes)), shape=(n, len(payer_index)))
    degree = np.asarray(B.sum(axis=0)).ravel()
    idf = np.log(n / degree)
    S = (B @ sparse.diags(idf) @ B.T).toarray()
    shared_cnt = (B @ B.T).toarray()
    np.fill_diagonal(S, 0)
    np.fill_diagonal(shared_cnt, 0)
    W["payers"] = S / S.max() if S.max() > 0 else S

    def add(kind, pairs, w):
        for i, j in pairs:
            W[kind][i, j] = W[kind][j, i] = w

    if len(logins):
        add("device", _pairs_from_groups(logins.groupby("device_id")["merchant_id"].unique(), idx), C.EDGE_W["device"])
        for t, key in [("FIXED", "ip_fixed"), ("CLOUD", "ip_cloud"), ("CARRIER", "ip_carrier")]:
            sub = logins[logins["ip_type"] == t]
            add(key, _pairs_from_groups(sub.groupby("ip")["merchant_id"].unique(), idx), C.EDGE_W[key])
    m = merchants
    add("settlement", _pairs_from_groups(m.groupby("settlement_account_hash")["merchant_id"].unique(), idx),
        C.EDGE_W["settlement"])
    add("director", _pairs_from_groups(m.groupby("director_id_hash")["merchant_id"].unique(), idx), C.EDGE_W["director"])
    if len(api):
        srv = api.groupby(["server_ip", "server_asn_label"])["merchant_id"].unique().reset_index()
        for r in srv.itertuples(index=False):
            w = C.EDGE_W["server_common"] if r.server_asn_label in C.COMMON_PLATFORM_ASNS else C.EDGE_W["server_ip"]
            pairs = _pairs_from_groups(pd.Series([r.merchant_id]), idx)
            for i, j in pairs:
                W["server_ip"][i, j] = W["server_ip"][j, i] = max(W["server_ip"][i, j], w)

    total = sum(W.values())
    iu, ju = np.triu_indices(n, 1)
    keep = total[iu, ju] > 0
    iu, ju = iu[keep], ju[keep]
    edges = pd.DataFrame({"a": [ids[i] for i in iu], "b": [ids[j] for j in ju],
                          "shared_payers": shared_cnt[iu, ju].astype(int)})
    for t in EDGE_TYPES:
        edges[t] = W[t][iu, ju]
    edges["weight"] = total[iu, ju]
    edges["linked"] = edges["weight"] >= C.LINK_THRESHOLD
    edges["main_type"] = edges[EDGE_TYPES].idxmax(axis=1)
    return edges.sort_values("weight", ascending=False).reset_index(drop=True)


def neighbours(edges: pd.DataFrame, mid: str, linked_only=True) -> pd.DataFrame:
    e = edges[(edges["a"] == mid) | (edges["b"] == mid)]
    if linked_only:
        e = e[e["linked"]]
    e = e.assign(other=np.where(e["a"] == mid, e["b"], e["a"]))
    return e


def detect_successors(t: pd.DataFrame, suspended: dict[str, int], bet_payers: dict[str, set]) -> pd.DataFrame:
    """For each suspended merchant X at day s, share of X's betting-cluster payers that
    newly appear at another merchant during days s+1..s+window."""
    rows = []
    for x, s in suspended.items():
        payers = bet_payers.get(x, set())
        if not payers:
            continue
        win = t[(t["day"] > s) & (t["day"] <= s + C.SUCCESSOR_WINDOW) & (t["merchant_id"] != x)]
        before = t[(t["day"] <= s) & t["payer_id"].isin(payers)]
        seen_before = set(zip(before["merchant_id"], before["payer_id"]))
        win = win[win["payer_id"].isin(payers)]
        pairs = win[["merchant_id", "payer_id"]].drop_duplicates()
        pairs = pairs[[(a, b) not in seen_before for a, b in zip(pairs["merchant_id"], pairs["payer_id"])]]
        share = pairs.groupby("merchant_id").size() / len(payers)
        for mid, sh in share.items():
            rows.append(dict(merchant_id=mid, predecessor=x, share=float(sh),
                             flagged=sh >= C.SUCCESSOR_SHARE))
    return pd.DataFrame(rows, columns=["merchant_id", "predecessor", "share", "flagged"])


def layout(merchants: pd.DataFrame, edges: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Deterministic two-zone layout. Left: components that contain a real link (>= 0.8), spread by a
    seeded spring layout. Right: every other merchant in a grid, grouped by declared category."""
    strong = edges[edges["weight"] >= C.WEAK_EDGE_HIDE]
    G = nx.Graph()
    for r in strong.itertuples(index=False):
        G.add_edge(r.a, r.b, linked=r.linked)
    core = set()
    for comp in nx.connected_components(G):
        if any(d["linked"] for _, _, d in G.subgraph(comp).edges(data=True)):
            core |= comp
    pos = {}
    if core:
        sp = nx.spring_layout(G.subgraph(sorted(core)), seed=C.SEED, k=2.0, iterations=300, weight=None)
        # even spacing on a circle, keeping the spring layout's angular order (labels never collide)
        order = sorted(sp, key=lambda k: np.arctan2(sp[k][1], sp[k][0]))
        for j, k in enumerate(order):
            a = 2 * np.pi * j / len(order) + np.pi / 2
            pos[k] = (float(0.75 * np.cos(a)), float(0.82 * np.sin(a)))
    rest = merchants[~merchants["merchant_id"].isin(core)]
    cols, dx, dy = 9, 0.21, 0.17
    y = 1.2
    for cat in C.CATEGORIES:
        sub = rest[rest["declared_category"] == cat].sort_values(["is_persona", "merchant_id"],
                                                                  ascending=[False, True])
        for k, mid in enumerate(sub["merchant_id"]):
            pos[mid] = (2.4 + (k % cols) * dx, y - (k // cols) * dy)
        y -= (len(sub) - 1) // cols * dy + dy + 0.16
    return pos
