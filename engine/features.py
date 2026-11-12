"""Payer-level and merchant-level features."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from engine import config as C


def enrich(tx: pd.DataFrame, merchants: pd.DataFrame, payers: pd.DataFrame) -> pd.DataFrame:
    """Add hour, window flags, round flag and remote flag to transactions."""
    t = tx.copy()
    t["hour"] = t["ts"].dt.hour + t["ts"].dt.minute / 60.0
    t["hour_bucket"] = t["ts"].dt.hour
    t["match_day"] = (t["day"] % C.MATCH_EVERY) == 0
    t["prematch"] = t["match_day"] & (t["hour"] >= C.PREMATCH[0]) & (t["hour"] < C.PREMATCH[1])
    t["night"] = (t["hour"] >= C.NIGHT[0]) & (t["hour"] < C.NIGHT[1])
    t["is_round"] = (t["amount_inr"] % 100) == 0
    t["festival"] = t["day"].between(*C.FESTIVAL_DAYS)
    region = merchants.set_index("merchant_id")["region"]
    home = payers.set_index("payer_id")["home_region"]
    t["remote"] = t["payer_id"].map(home).values != t["merchant_id"].map(region).values
    # same-day consecutive escalation, per merchant-payer
    t = t.sort_values(["merchant_id", "payer_id", "ts"], kind="mergesort")
    prev_amt = t.groupby(["merchant_id", "payer_id", "day"], sort=False)["amount_inr"].shift(1)
    t["pair"] = prev_amt.notna()
    t["escalate"] = t["pair"] & (t["amount_inr"] >= 1.5 * prev_amt)
    return t.sort_values("ts", kind="mergesort")


def payer_features(t: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby(["merchant_id", "payer_id"], sort=False)
    per_day = t.groupby(["merchant_id", "payer_id", "day"], sort=False).size()
    out = pd.DataFrame({
        "n_txns": g.size(),
        "active_days": g["day"].nunique(),
        "first_day": g["day"].min(),
        "max_txns_in_day": per_day.groupby(level=[0, 1]).max(),
        "round_share": g["is_round"].mean(),
        "pairs": g["pair"].sum(),
        "esc": g["escalate"].sum(),
        "prematch_share": g["prematch"].mean(),
        "night_share": g["night"].mean(),
        "remote_flag": g["remote"].first().astype(float),
        "total_amount": g["amount_inr"].sum(),
        "mean_amount": g["amount_inr"].mean(),
    })
    out["txns_per_active_day"] = out["n_txns"] / out["active_days"]
    out["escalation_share"] = np.where(out["pairs"] > 0, out["esc"] / out["pairs"].clip(lower=1), 0.0)
    out["log_mean_amount"] = np.log(out["mean_amount"])
    return out.reset_index()


def shape_inputs(rhythm_tpd, round_share, esc_share, prematch_share, night_share, remote_share):
    """Normalise raw shape measures into the six [0,1] score inputs."""
    return dict(
        rhythm=float(np.clip((rhythm_tpd - 1) / 4, 0, 1)),
        round=float(np.clip(round_share, 0, 1)),
        escalation=float(np.clip(esc_share / 0.4, 0, 1)),
        prematch=float(np.clip(prematch_share / 0.5, 0, 1)),
        night=float(np.clip(night_share / 0.3, 0, 1)),
        remote=float(np.clip(remote_share, 0, 1)),
    )


def bet_shape_score(inputs: dict) -> float:
    w = C.SCORE_W
    z = w["bias"] + sum(w[k] * inputs[k] for k in ("rhythm", "round", "escalation", "prematch", "night", "remote"))
    return float(1 / (1 + np.exp(-z)))


def _robust_z(values: pd.Series, cohort: pd.Series) -> pd.Series:
    cohort = cohort.dropna()
    med = cohort.median()
    mad = (cohort - med).abs().median() * 1.4826
    return (values - med) / (mad + 1e-9)


def merchant_features(t: pd.DataFrame, pf: pd.DataFrame, merchants: pd.DataFrame, api: pd.DataFrame,
                      payouts: pd.DataFrame, current_day: int, rng_seed: int = C.SEED) -> pd.DataFrame:
    m = merchants.set_index("merchant_id")
    ids = m.index
    g = t.groupby("merchant_id")
    f = pd.DataFrame(index=ids)
    f["n_txns"] = g.size()
    f["volume_inr"] = g["amount_inr"].sum()
    f["n_payers"] = pf.groupby("merchant_id").size()
    f["active_days"] = g["day"].nunique()
    f[["n_txns", "volume_inr", "n_payers", "active_days"]] = f[["n_txns", "volume_inr", "n_payers",
                                                                 "active_days"]].fillna(0)
    f["sufficient"] = ((f["active_days"] >= C.MIN_ACTIVE_DAYS) & (f["n_txns"] >= C.MIN_TXNS)
                       & (f["n_payers"] >= C.MIN_PAYERS))
    # payer-weighted: every payer counts once, so many genuine payers outweigh a small betting group
    pg = pf.groupby("merchant_id")
    f["m_rhythm_tpd"] = pg["txns_per_active_day"].mean()
    f["m_round_share"] = pg["round_share"].mean()
    f["m_esc_share"] = pg["escalation_share"].mean()
    f["m_prematch_share"] = pg["prematch_share"].mean()
    f["m_night_share"] = pg["night_share"].mean()
    f["m_remote_share"] = pg["remote_flag"].mean()
    f["merchant_score"] = [
        bet_shape_score(shape_inputs(r.m_rhythm_tpd, r.m_round_share, r.m_esc_share, r.m_prematch_share,
                                     r.m_night_share, r.m_remote_share)) if r.n_txns > 0 else 0.0
        for r in f.fillna(0).itertuples()]
    cohort_mask = ~m["persona_key"].isin(C.FRONT_PERSONAS)

    # ---------- daily volume series and surge relative to the cohort's own surge
    daily = t.pivot_table(index="merchant_id", columns="day", values="amount_inr", aggfunc="sum", fill_value=0.0)
    daily = daily.reindex(index=ids, columns=range(1, current_day + 1), fill_value=0.0)
    onboard = m["onboard_day"]
    last7 = daily.loc[:, current_day - 6: current_day].mean(axis=1)
    prev_med = daily.loc[:, max(1, current_day - 34): current_day - 7].median(axis=1)
    has_hist = (current_day - 7 - onboard) >= C.MIN_HISTORY_DAYS
    f["surge_ratio"] = np.where(has_hist, last7 / prev_med.clip(lower=1), np.nan)
    f["cohort_surge"] = np.nan
    for cat in C.CATEGORIES:
        in_cat = m["declared_category"] == cat
        f.loc[in_cat, "cohort_surge"] = f.loc[in_cat & cohort_mask & f["sufficient"], "surge_ratio"].median()
    f["surge_rel"] = f["surge_ratio"] / f["cohort_surge"]
    f["today_volume"] = daily[current_day]

    # ---------- event-aware peer shape distances: normal days vs the cohort's normal days,
    #            festival window vs the cohort's festival window; the larger robust z is used
    t = t.assign(logamt=np.log(t["amount_inr"].values))
    rng = np.random.default_rng(rng_seed)
    periods = {"normal": t[~t["festival"]], "festival": t[t["festival"]]}
    min_n = {"normal": C.MIN_TXNS, "festival": 50}
    d = pd.DataFrame(index=ids)
    pooled_hour = {}
    for per, pt in periods.items():
        by_m = {k: v for k, v in pt.groupby("merchant_id")}
        for cat in C.CATEGORIES:
            cm = ids[(m["declared_category"] == cat) & cohort_mask & f["sufficient"]]
            s = pt[pt["merchant_id"].isin(cm)]
            if s.empty:
                continue
            a = s["logamt"].values
            pa = rng.choice(a, size=min(len(a), 30_000), replace=False)
            h = np.bincount(s["hour_bucket"].values, minlength=24).astype(float)
            ph = h / h.sum()
            if per == "normal":
                pooled_hour[cat] = ph
            for mid in ids:
                x = by_m.get(mid)
                if x is None or len(x) < min_n[per] or not f.at[mid, "sufficient"]:
                    continue
                hx = np.bincount(x["hour_bucket"].values, minlength=24).astype(float)
                d.loc[mid, f"amt::{per}::{cat}"] = wasserstein_distance(x["logamt"].values, pa)
                d.loc[mid, f"hour::{per}::{cat}"] = jensenshannon(hx / hx.sum(), ph, base=2) ** 2
    for per in periods:
        for kind in ("amt", "hour"):
            f[f"{kind}_dist_{per}"] = [d.at[mid, f"{kind}::{per}::{m.at[mid, 'declared_category']}"]
                                       if f"{kind}::{per}::{m.at[mid, 'declared_category']}" in d else np.nan
                                       for mid in ids]
    for cat in C.CATEGORIES:  # recategorisation distance on normal days
        f[f"catdist::{cat}"] = d.get(f"amt::normal::{cat}", np.nan) + d.get(f"hour::normal::{cat}", np.nan)

    # ---------- channel features
    s2s = t[t["channel"] == "S2S_API"].groupby("merchant_id")["amount_inr"].sum()
    f["s2s_share"] = (s2s.reindex(ids).fillna(0) / f["volume_inr"].clip(lower=1))
    f["s2s_undeclared_share"] = np.where(m["integration_type"] == "API", 0.0, f["s2s_share"])
    hosted = t[t["method"] == "HOSTED_CHECKOUT"]
    ref_mismatch = (hosted["referrer_domain"] != hosted["merchant_id"].map(m["registered_domain"]))
    f["referrer_mismatch_share"] = ref_mismatch.groupby(hosted["merchant_id"]).mean().reindex(ids).fillna(0)
    blocked = hosted[hosted["referrer_domain"].isin(C.BLOCKED_DOMAINS)]
    f["blocked_referrer"] = ids.isin(blocked["merchant_id"].unique())
    f["blocked_referrers"] = [sorted(blocked.loc[blocked["merchant_id"] == mid, "referrer_domain"].unique().tolist())
                              for mid in ids]
    links = api.pivot_table(index="merchant_id", columns="day", values="payment_links_created",
                            aggfunc="sum", fill_value=0).reindex(index=ids, columns=range(1, current_day + 1),
                                                                 fill_value=0)
    l7 = links.loc[:, current_day - 6: current_day].mean(axis=1)
    lprev = links.loc[:, max(1, current_day - 34): current_day - 7].median(axis=1)
    f["link_spike"] = (l7 + 1) / (lprev + 1)
    f["links_last7"] = l7
    offshore = api[api["server_country"] != "IN"]
    f["offshore_api"] = ids.isin(offshore["merchant_id"].unique())
    f["offshore_api_nondeclared"] = f["offshore_api"] & (m["integration_type"] != "API")

    # ---------- match-day uplift (pre-match window, match days vs non-match days)
    win = t[(t["hour"] >= C.PREMATCH[0]) & (t["hour"] < C.PREMATCH[1])]
    wd = win.groupby(["merchant_id", "match_day"])["amount_inr"].sum().unstack(fill_value=0)
    wd = wd.reindex(index=ids, fill_value=0)
    days_active = [np.arange(max(1, o), current_day + 1) for o in onboard]
    n_match = pd.Series([int((da % C.MATCH_EVERY == 0).sum()) for da in days_active], index=ids)
    n_non = pd.Series([len(da) for da in days_active], index=ids) - n_match
    mean_match = wd.get(True, 0) / n_match.clip(lower=1)
    mean_non = wd.get(False, 0) / n_non.clip(lower=1)
    f["matchday_uplift"] = (mean_match + 100) / (mean_non + 100)
    f.loc[n_match < 5, "matchday_uplift"] = np.nan

    # ---------- card mix
    card = t[t["method"] == "CARD"]
    cg = card.groupby("merchant_id")
    f["card_txns"] = cg.size().reindex(ids).fillna(0)
    f["card_tokens_per_100"] = (cg["card_token_hash"].nunique() / cg.size() * 100).reindex(ids)
    f["prepaid_share"] = cg["card_type"].apply(lambda s: (s == "PREPAID").mean()).reindex(ids)
    f["foreign_share"] = cg["card_issuer_country"].apply(lambda s: (s != "IN").mean()).reindex(ids)

    # ---------- payouts
    if len(payouts):
        po = payouts.assign(week=(payouts["ts"] - pd.Timestamp(C.DAY1)).dt.days // 7)
        wk = po.groupby(["merchant_id", "week"])["beneficiary_count"].sum().groupby(level=0).max()
        f["max_weekly_beneficiaries"] = wk.reindex(ids).fillna(0)
    else:
        f["max_weekly_beneficiaries"] = 0
    f["payout_fanout"] = f["max_weekly_beneficiaries"] >= 20

    # ---------- robust z against declared-category cohort (fronts excluded)
    zcols = [("amt_dist_normal", "amount_z_normal"), ("amt_dist_festival", "amount_z_festival"),
             ("hour_dist_normal", "hour_z_normal"), ("hour_dist_festival", "hour_z_festival"),
             ("matchday_uplift", "matchday_z")]
    for col, zcol in zcols:
        f[zcol] = np.nan
        # distances are positive and right-skewed, so they are compared on a log scale
        x = np.log(f[col].clip(lower=1e-9)) if col != "matchday_uplift" else f[col]
        for cat in C.CATEGORIES:
            in_cat = m["declared_category"] == cat
            f.loc[in_cat, zcol] = _robust_z(x[in_cat], x[in_cat & cohort_mask])
    f["amount_z"] = f[["amount_z_normal", "amount_z_festival"]].max(axis=1)
    f["hour_z"] = f[["hour_z_normal", "hour_z_festival"]].max(axis=1)
    f.attrs["pooled_hour"] = pooled_hour
    f.attrs["daily"] = daily
    f.attrs["daily_n"] = t.pivot_table(index="merchant_id", columns="day", values="amount_inr", aggfunc="size",
                                       fill_value=0).reindex(index=ids, columns=range(1, current_day + 1), fill_value=0)
    return f
