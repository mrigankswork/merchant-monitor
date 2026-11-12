"""Proportionate response: reserve, clean days, earned access, national checks, tier actions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine import config as C

RESERVE_NOTICE = ("Volume consistent with your usual business settles normally. Volume outside your usual "
                  "pattern is held for up to 48 hours. Upload invoices or delivery records to release it sooner.")
NUDGE_NOTICE = "Your recent activity differs from similar businesses. Please confirm or update your business details."
TIER4_ACTIONS = ["Acceptance suspended", "Pending settlement held under contract",
                 "Suspicious transaction report drafted", "Referred to I4C (simulated)", "Network sweep run"]


def flagged_payers(r, mid: str) -> set:
    """Payers in clusters that meet the flagged-cluster rule (score >= 0.7 and enough volume or payers)."""
    summ, payers = r.clusters.get(mid, (None, None))
    if summ is None:
        return set()
    fl = summ[summ["flagged"]]
    return set(payers.loc[payers["cluster"].isin(fl["cluster"]), "payer_id"])


def _tx(r, mid):
    return r.tx[r.tx["merchant_id"] == mid]


def anomaly_days(r, mid: str) -> set:
    """Days with flagged-cluster activity or server-API traffic from an offshore / undeclared source."""
    t = _tx(r, mid)
    fp = flagged_payers(r, mid)
    days = set(t.loc[t["payer_id"].isin(fp), "day"])
    api = r.data["api_logs"]
    api = api[(api["merchant_id"] == mid) & (api["day"] <= r.current_day)]
    days |= set(api.loc[api["server_country"] != "IN", "day"])
    integ = r.merchants.set_index("merchant_id").at[mid, "integration_type"]
    if integ != "API":
        days |= set(t.loc[t["channel"] == "S2S_API", "day"])
    return days


def clean_days(r, mid: str) -> dict:
    """A clean day had >= 1 transaction, no flagged-cluster or offshore/undeclared API activity, and the
    merchant has no national-list hit. History before day 1 counts as clean for long-standing merchants."""
    onboard = int(r.merchants.set_index("merchant_id").at[mid, "onboard_day"])
    if bool(r.nat.at[mid, "any"]):
        return dict(observed=0, prior=0, total=0, national_hit=True)
    active = set(_tx(r, mid)["day"])
    observed = len(active - anomaly_days(r, mid))
    prior = max(0, -onboard)
    first_bad = min(anomaly_days(r, mid), default=None)
    before_first = len({d for d in active if first_bad is None or d < first_bad}) + prior
    return dict(observed=observed, prior=prior, total=observed + prior, national_hit=False,
                before_first_anomaly=before_first, first_anomaly=first_bad)


def _cohort_baseline(r, cat: str) -> float:
    m = r.merchants
    ids = m.loc[(m["declared_category"] == cat) & ~m["persona_key"].isin(C.FRONT_PERSONAS), "merchant_id"]
    ids = [i for i in ids if bool(r.f.at[i, "sufficient"])]
    daily = r.f.attrs["daily"].loc[ids, max(1, r.current_day - 28): r.current_day - 1]
    return float(daily.median(axis=1).median())


def reserve(r, mid: str) -> dict:
    """Tier 3 only.
    Flagged payer cluster: the genuine clusters settle normally and today's flagged-cluster volume is held.
    No flagged cluster: baseline = median daily volume over the 28 days before the first anomaly day (cohort
    median when there is less history); reserve = max(0, today - 1.5 x baseline)."""
    m = r.merchants.set_index("merchant_id").loc[mid]
    tier = int(r.tiers[mid])
    t = _tx(r, mid)
    today_t = t[t["day"] == r.current_day]
    today = float(today_t["amount_inr"].sum())
    fp = flagged_payers(r, mid) if bool(r.cs.at[mid, "hidden_cluster"]) else set()
    if fp:
        held = float(today_t.loc[today_t["payer_id"].isin(fp), "amount_inr"].sum())
        out = dict(mode="cluster", baseline=None, flagged_today=held, genuine_today=today - held,
                   basis="today's volume from the flagged payer cluster is held; all other payers settle normally")
    else:
        first = min(anomaly_days(r, mid), default=r.current_day)
        daily = r.f.attrs["daily"].loc[mid]
        hist = daily.loc[max(1, first - 28): first - 1]
        hist = hist[hist.index >= max(1, int(m["onboard_day"]))]
        if len(hist) >= C.MIN_ACTIVE_DAYS:
            baseline, basis = float(hist.median()), f"median daily volume over the 28 days before day {first}"
        else:
            baseline = _cohort_baseline(r, m["declared_category"])
            basis = "declared-category cohort median (too little history before the first anomaly)"
        held = max(0.0, today - C.RESERVE_MULT * baseline)
        out = dict(mode="baseline", baseline=baseline, flagged_today=None, genuine_today=None, basis=basis)
    applies = tier == 3
    held = held if applies else 0.0
    return dict(out, applies=applies, today=today, reserve_held=held, settles=today - held,
                notice=RESERVE_NOTICE if applies else None)


def national_checks(r, mid: str) -> list[dict]:
    n = r.nat.loc[mid]
    return [
        dict(check="Onboarding phone vs DoT FRI high-risk numbers", short="phone on DoT FRI list",
             source="simulated", hit=bool(n["fri_phone"])),
        dict(check="Settlement account vs I4C Suspect Registry", short="settlement account on I4C list",
             source="simulated", hit=bool(n["i4c_account"])),
        dict(check="Referrer domains vs MeitY blocked list", short="referrer on MeitY blocked list",
             source="simulated", hit=bool(n["blocked_domain"]), detail=", ".join(r.f.at[mid, "blocked_referrers"]) or None),
    ]


def earned_access(r, mid: str) -> dict:
    m = r.merchants.set_index("merchant_id").loc[mid]
    onboard = int(m["onboard_day"])
    cd = clean_days(r, mid)
    nat_hit = cd["national_hit"]
    hits = [c["short"] for c in national_checks(r, mid) if c["hit"]]
    fast = bool(m.get("fast_track_video_kyc", False))
    sufficient = bool(r.f.at[mid, "sufficient"])
    unlocked = (not nat_hit) and (fast or (cd["total"] >= C.EARNED_ACCESS_DAYS and int(r.tiers[mid]) < 3))
    route = ("fast-track video KYC" if fast and unlocked else
             f"{C.EARNED_ACCESS_DAYS} clean days" if unlocked else None)
    counterfactual = None
    tier = int(r.tiers[mid])
    if tier >= 3:
        if nat_hit or onboard > 0:
            n = 0 if nat_hit else cd["before_first_anomaly"]
            counterfactual = (f"Under Earned Access, server API and bulk links would have stayed locked. Onboarded "
                              f"day {onboard}, {n} clean days of the {C.EARNED_ACCESS_DAYS} required; national hit: "
                              f"{', '.join(hits) if hits else 'none'}.")
            if nat_hit:
                counterfactual += f" Fast track would be refused: {hits[0]} (simulated)."
        else:
            counterfactual = (f"Earned Access would not have stopped this account: {cd['before_first_anomaly']} clean "
                              f"days before the first anomaly (day {cd['first_anomaly']}) and no national hit. "
                              f"The hidden-cluster layer is what catches it.")
    elif fast:
        counterfactual = (f"Onboarded day {onboard} with {cd['total']} clean days, but verified through fast-track "
                          f"video KYC, so server API is open from day one. Earned Access does not penalise "
                          f"genuine new businesses.")
    elif not sufficient and onboard > 0:
        counterfactual = (f"Insufficient history: monitoring. {cd['total']} clean days of the "
                          f"{C.EARNED_ACCESS_DAYS} required, so earned channels stay locked.")
    return dict(always_open=["Counter QR", "Website checkout"], earned=["Server API", "Bulk payment links", "Payouts"],
                unlocked=unlocked, route=route, clean=cd, onboard_day=onboard, national_hit=nat_hit,
                fast_track=fast, counterfactual=counterfactual)


def total_reserve(r, released=()) -> float:
    return float(sum(reserve(r, mid)["reserve_held"] for mid in r.tiers[r.tiers == 3].index if mid not in released))
