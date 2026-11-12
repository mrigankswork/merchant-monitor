"""Plain-language alert reasons, generated from computed numbers only."""
from __future__ import annotations

import numpy as np

from engine import config as C
from engine.network import neighbours

EDGE_NAMES = {"payers": "shared payers", "device": "same dashboard device", "settlement": "same settlement account",
              "director": "same director", "ip_fixed": "same fixed login IP", "server_ip": "same API server"}
CAT_PLURAL = {"Bakery": "bakeries", "Wallet/recharge top-up": "top-up merchants",
              "General merchandise": "general stores", "IT services": "IT services firms"}
NOUN = {"Bakery": "bakery", "Wallet/recharge top-up": "wallet top-up merchant",
        "General merchandise": "general store", "IT services": "IT services firm"}
INTEGRATION_NAME = {"QR": "counter QR", "website": "website checkout", "API": "server API"}


def pct(x) -> str:
    return f"{100 * x:.0f}%"


def peer_stats(r, cat: str) -> dict:
    m = r.merchants
    ids = m.loc[(m["declared_category"] == cat) & ~m["persona_key"].isin(C.FRONT_PERSONAS), "merchant_id"]
    f = r.f.loc[ids]
    return dict(round=f["m_round_share"].median(), rhythm=f["m_rhythm_tpd"].median(),
                prematch=f["m_prematch_share"].median(), night=f["m_night_share"].median(),
                uplift=f["matchday_uplift"].median(), score=r.gap.loc[ids, "observed"].median(),
                remote=f["m_remote_share"].median())


def name(r, mid) -> str:
    return r.merchants.set_index("merchant_id").at[mid, "name"]


def festival_uplift(r, mid) -> float | None:
    """Transaction-count uplift in the festival window vs the 28 days before it."""
    d = r.f.attrs["daily_n"].loc[mid]
    lo, hi = C.FESTIVAL_DAYS
    if r.current_day < hi:
        return None
    pre = d.loc[lo - 28: lo - 1]
    if pre.median() <= 0:
        return None
    return float(d.loc[lo:hi].mean() / pre.median())


def peer_festival_uplift(r, cat) -> float:
    m = r.merchants
    ids = m.loc[(m["declared_category"] == cat) & ~m["is_persona"], "merchant_id"]
    vals = [festival_uplift(r, i) for i in ids]
    return float(np.nanmedian([v for v in vals if v is not None]))


def bet_cluster_stats(r, mid) -> dict:
    summ, payers = r.clusters[mid]
    bet = summ[summ["score"] >= C.HIDDEN_SCORE]
    bp = payers[payers["cluster"].isin(bet["cluster"])]
    tsub = r.tx[(r.tx["merchant_id"] == mid) & r.tx["payer_id"].isin(set(bp["payer_id"]))]
    return dict(round=tsub["is_round"].mean(), prematch=tsub["prematch"].mean(),
                rhythm=len(tsub) / max(1, tsub.groupby(["payer_id", "day"]).ngroups),
                score=float(bet["score"].max()), payer_share=float(bet["payer_share"].sum()),
                volume_share=float(bet["volume_share"].sum()))


def reasons_tagged(r, mid: str, tier: int | None = None) -> list[tuple[str, str]]:
    """(kind, text) pairs for Tier 2+ merchants, most important first, at most five.
    `tier` overrides the displayed tier (used to keep the model's findings for an analyst-cleared case)."""
    tier = int(r.tiers[mid]) if tier is None else tier
    if tier < 2:
        return []
    m = r.merchants.set_index("merchant_id").loc[mid]
    f, cs, gap = r.f.loc[mid], r.cs.loc[mid], r.gap.loc[mid]
    cat = m["declared_category"]
    peers = peer_stats(r, cat)
    plural = CAT_PLURAL[cat]
    out = []
    why3 = r.tier3_why[mid]

    if "successor" in why3:
        s = r.successors[(r.successors["merchant_id"] == mid) & r.successors["flagged"]].iloc[0]
        out.append(("successor", f"Likely successor of {name(r, s['predecessor'])}: {pct(s['share'])} of its bettors "
                                 f"started paying here within {C.SUCCESSOR_WINDOW} days of its suspension."))

    if not f["sufficient"]:
        out.append(("clear", f"Insufficient history for shape tiers ({int(f['active_days'])} active days, "
                             f"{int(f['n_txns'])} transactions, {int(f['n_payers'])} payers; needs {C.MIN_ACTIVE_DAYS}, "
                             f"{C.MIN_TXNS} and {C.MIN_PAYERS}). Flagged through links, not money shape."))
    if f["sufficient"] and (bool(cs["hidden_cluster"]) or f["merchant_score"] >= C.MERCHANT_SCORE_FLAG):
        b = bet_cluster_stats(r, mid)
        if cs["hidden_cluster"] and cs["bet_cluster_volume_share"] < 0.8:
            since = (f" since it appeared on day {int(cs['bet_first_day'])} ({pct(b['volume_share'])} over the full window)"
                     if cs["bet_first_day"] and cs["bet_first_day"] > 1 else "")
            out.append(("cluster", f"Betting-shaped payer cluster hidden inside normal sales: {pct(b['payer_share'])} "
                                   f"of payers, {pct(cs['bet_volume_share_since'])} of volume{since}. Cluster score "
                                   f"{b['score']:.2f}; whole-merchant score only {f['merchant_score']:.2f}, because "
                                   f"each payer counts once."))
        else:
            out.append(("cluster", f"Betting-shaped money: {pct(b['volume_share'])} of volume and {pct(b['payer_share'])} "
                                   f"of payers in a cluster scoring {b['score']:.2f} (peer {plural}: "
                                   f"{peers['score']:.2f}; declaration gap z = {gap['gap_z']:.0f})."))
        out.append(("shape", f"In that cluster: round amounts {pct(b['round'])} vs {pct(peers['round'])} for peer "
                             f"{plural}; payers averaging {b['rhythm']:.1f} payments/day vs {peers['rhythm']:.1f}; "
                             f"{pct(b['prematch'])} of payments in pre-match windows vs {pct(peers['prematch'])}."))

    if f["s2s_undeclared_share"] > C.S2S_SHARE or f["offshore_api_nondeclared"]:
        out.append(("channel", f"{pct(f['s2s_undeclared_share'])} of volume created via server API from an offshore "
                               f"server (SG); merchant declared {INTEGRATION_NAME[m['integration_type']]}."))
    if f["blocked_referrer"]:
        out.append(("domain", f"Checkout traffic referred from {', '.join(f['blocked_referrers'])} (on blocked list, "
                              f"simulated)."))
    if r.sig.at[mid, "matchday"] and len(out) < 5:
        out.append(("matchday", f"Pre-match window volume on T20 match days is {f['matchday_uplift']:.1f}× non-match "
                                f"days (peer {plural}: {peers['uplift']:.1f}×)."))
    nat = r.nat.loc[mid]
    if nat["fri_phone"]:
        out.append(("national", "Onboarding phone number is on the DoT FRI high-risk list (simulated)."))
    if nat["i4c_account"]:
        out.append(("national", "Settlement account is on the I4C Suspect Registry (simulated)."))
    if r.sig.at[mid, "network"] or "linked_confirmed" in why3:
        hot = set(r.tiers[r.tiers >= 3].index)
        nb = neighbours(r.edges, mid)
        nb = nb[nb["other"].isin(hot)]
        if len(nb):
            e = nb.iloc[0]
            kinds = [EDGE_NAMES[k] for k in EDGE_NAMES if e[k] >= 0.1]
            lead = "Linked to confirmed front" if e["other"] in r.confirmed or e["other"] in r.suspended else "Linked to"
            more = f" and {len(nb) - 1} other flagged merchant(s)." if len(nb) > 1 else "."
            out.append(("network", f"{lead} {name(r, e['other'])} (link weight {e['weight']:.1f}: "
                                   f"{', '.join(kinds)}){more}"))
    if f["payout_fanout"] and tier >= 3:
        out.append(("payout", f"Payouts fan out to up to {int(f['max_weekly_beneficiaries'])} beneficiaries a week."))

    if tier == 2:
        if f["surge_rel"] > C.SURGE_RATIO:
            out.append(("surge", f"Volume over the last 7 days is {f['surge_ratio']:.1f}× its prior 28-day median, "
                                 f"against {f['cohort_surge']:.1f}× for peer {plural} over the same days."))
        if gap["recat_suggestion"]:
            out.append(("recat", f"Possible miscategorisation: looks like {gap['recat_suggestion']}."))
        if f["amount_z"] > C.TIER2_Z or f["hour_z"] > C.TIER2_Z:
            t = r.tx[r.tx["merchant_id"] == mid]
            out.append(("shape", f"Ticket sizes and timing differ from peer {plural} (median ticket "
                                 f"₹{t['amount_inr'].median():,.0f}; amount-shape z = {f['amount_z']:.1f}, "
                                 f"hour-shape z = {f['hour_z']:.1f})."))
        regions = r.tx.loc[r.tx["merchant_id"] == mid, "payer_region"].nunique()
        out.append(("clear", f"Not betting-shaped: highest payer-cluster score {cs['max_cluster_score']:.2f} (betting "
                             f"clusters score above 0.8); payers from {regions} regions, about "
                             f"{f['m_rhythm_tpd']:.1f} payment per day, {pct(f['m_round_share'])} round amounts."))
    return out[:5]


def reasons(r, mid: str) -> list[str]:
    return [t for _, t in reasons_tagged(r, mid)]


def green_notes(r, mid: str) -> list[tuple[str, str]]:
    """Why a merchant is *not* escalated (Tier 1), from computed numbers."""
    m = r.merchants.set_index("merchant_id").loc[mid]
    f, cs, gap = r.f.loc[mid], r.cs.loc[mid], r.gap.loc[mid]
    cat = m["declared_category"]
    plural = CAT_PLURAL[cat]
    peers = peer_stats(r, cat)
    out = []
    if not f["sufficient"]:
        out.append(("clear", f"Insufficient history: monitoring. {int(f['active_days'])} active days, "
                             f"{int(f['n_txns'])} transactions and {int(f['n_payers'])} payers so far (needs "
                             f"{C.MIN_ACTIVE_DAYS}, {C.MIN_TXNS} and {C.MIN_PAYERS}). Shape-based tiers wait; links, "
                             f"successor checks and national lists still apply."))
        if m.get("fast_track_video_kyc", False):
            out.append(("clear", "Verified through fast-track video KYC, so its server API is open despite few clean "
                                 "days."))
    else:
        out.append(("clear", f"No betting-shaped payer group (highest cluster score {cs['max_cluster_score']:.2f}; "
                             f"betting clusters score above 0.8)."))
    fu = festival_uplift(r, mid)
    if fu is not None and fu > 1.5 and f["sufficient"]:
        pu = peer_festival_uplift(r, cat)
        z = f["amount_z_festival"]
        out.append(("festival", f"Festival spike {fu:.1f}× vs peer median {pu:.1f}× during the same window, within the "
                                f"normal range; its festival ticket mix is in line with peer {plural}' festival days "
                                f"(z = {z:.1f})."))
    if f["m_round_share"] > 0.5 and f["sufficient"]:
        out.append(("clear", f"Round amounts {pct(f['m_round_share'])}, which is normal for the declared "
                             f"{cat.lower()} category (peer median {pct(peers['round'])})."))
    if f["sufficient"]:
        out.append(("clear", f"Money shape matches the declared business: declaration gap z = "
                             f"{round(gap['gap_z'], 1) + 0:.1f}; ticket-size z = {f['amount_z']:.1f}, hour-of-day "
                             f"z = {f['hour_z']:.1f} (event-aware, festival days compared with peers' festival days)."))
    if r.sig.at[mid, "network"]:
        out.append(("network", "Has a link to a flagged merchant, but a link alone never escalates a merchant whose "
                               "own money is not betting-shaped."))
    return out[:4]


def headline(r, mid: str) -> str:
    """One short line for cards and tables."""
    tier = int(r.tiers[mid])
    f, cs = r.f.loc[mid], r.cs.loc[mid]
    cat = r.merchants.set_index("merchant_id").at[mid, "declared_category"]
    if tier == 4:
        return "Confirmed front · acceptance suspended"
    if "national_new" in r.tier3_why[mid] and "successor" not in r.tier3_why[mid]:
        return "New merchant with a national-list hit"
    if "successor" in r.tier3_why[mid]:
        s = r.successors[(r.successors["merchant_id"] == mid) & r.successors["flagged"]].iloc[0]
        return f"Successor of {name(r, s['predecessor'])} · {pct(s['share'])} of its bettors moved here"
    if tier == 3 and cs["hidden_cluster"] and cs["bet_cluster_volume_share"] < 0.8:
        return f"Hidden betting cluster · {pct(cs['bet_volume_share_since'])} of volume since day {int(cs['bet_first_day'])}"
    if tier == 3:
        return f"Betting-shaped money · {pct(cs['bet_cluster_volume_share'])} of volume"
    if tier == 2:
        if f["surge_rel"] > C.SURGE_RATIO:
            return f"Surge {f['surge_ratio']:.1f}× vs peers {f['cohort_surge']:.1f}× · not betting-shaped"
        return "Unusual for its category · not betting-shaped"
    if not f["sufficient"]:
        return "Insufficient history · monitoring" + (" · fast-track KYC" if r.merchants.set_index(
            "merchant_id").at[mid, "fast_track_video_kyc"] else "")
    fu = festival_uplift(r, mid)
    if fu is not None and fu > 2 * peer_festival_uplift(r, cat):
        return f"Festival surge {fu:.1f}× · in line with the season"
    if f["m_round_share"] > 0.5:
        return "Round amounts · consistent with declared top-up"
    return "Normal pattern for its category"


def _tenure(m, current_day) -> str:
    onboard = int(m["onboard_day"])
    days = current_day - onboard
    if days >= 365:
        return f"for {days / 365:.1f} years"
    return f"for {days} days" if onboard <= 0 else f"since day {onboard}"


def narrative(r, mid: str) -> str:
    """A short case summary in plain English, built only from computed numbers."""
    m = r.merchants.set_index("merchant_id").loc[mid]
    f, cs = r.f.loc[mid], r.cs.loc[mid]
    tier = int(r.tiers[mid])
    t = r.tx[r.tx["merchant_id"] == mid]
    noun = NOUN[m["declared_category"]]
    a = "an" if noun[0] in "aeiouAEIOU" else "a"
    fp, betlike = set(), set()
    summ, payers = r.clusters.get(mid, (None, None))
    if summ is not None:
        fp = set(payers.loc[payers["cluster"].isin(summ.loc[summ["flagged"], "cluster"]), "payer_id"])
        betlike = set(payers.loc[payers["cluster"].isin(summ.loc[summ["score"] >= 0.5, "cluster"]), "payer_id"])
    genuine = t[~t["payer_id"].isin(fp | betlike)]
    front = cs["bet_cluster_volume_share"] >= 0.8
    per_day = len(genuine) / max(1, genuine["day"].nunique())
    trade = "Its ordinary-looking cover trade is" if front else "Its regular trade is"
    base = (f"{m['name']} is {a} {noun} in {m['city']}, on the platform {_tenure(m, r.current_day)}. "
            f"{trade} about {per_day:.0f} payment{'' if round(per_day) == 1 else 's'} a day from {genuine['payer_id'].nunique():,} payers, "
            f"with a typical ticket of ₹{genuine['amount_inr'].median():,.0f}.")
    if not f["sufficient"] and tier < 3:
        return base + " There is not yet enough history to judge the shape of its money, so it is being monitored."
    if "successor" in r.tier3_why[mid]:
        s = r.successors[(r.successors["merchant_id"] == mid) & r.successors["flagged"]].iloc[0]
        return (base + f" Within {C.SUCCESSOR_WINDOW} days of {name(r, s['predecessor'])} being suspended, "
                f"{pct(s['share'])} of that front's bettors started paying here, through the same offshore server.")
    if fp and (cs["hidden_cluster"] or f["merchant_score"] >= C.MERCHANT_SCORE_FLAG):
        bt = t[t["payer_id"].isin(fp)]
        regions = bt["payer_region"].nunique()
        hidden = cs["bet_cluster_volume_share"] < 0.8
        lead = (f" Since day {int(bt['day'].min())}, a separate group of {len(fp)} payers from {regions} states has "
                f"been paying it" if hidden else f" Almost all of its money comes from {len(fp)} payers in "
                                                 f"{regions} states paying")
        return (base + lead + f" in round, escalating amounts ({pct(bt['is_round'].mean())} round, about "
                f"{len(bt) / max(1, bt.groupby(['payer_id', 'day']).ngroups):.1f} payments per active day), "
                f"concentrated before T20 matches. They are {pct(cs['bet_cluster_payer_share'])} of its payers but "
                f"{pct(cs['bet_volume_share_since'] if hidden else cs['bet_cluster_volume_share'])} of its volume"
                f"{' since they appeared' if hidden else ''}. That is not how {a} {noun} gets paid.")
    if tier == 2:
        return (base + f" Over the last week its volume rose {f['surge_ratio']:.1f}× while peers moved "
                f"{f['cohort_surge']:.1f}×, but the new money looks like ordinary customers: one payment each, normal "
                f"ticket sizes, normal hours. Unusual, not betting-shaped.")
    return base + " Its payers, ticket sizes and timing look like its declared category. Nothing suggests betting."


def payer_trail(r, mid: str) -> tuple[str, int, "pd.DataFrame"] | None:
    """One representative flagged payer's busiest match day: the clearest picture of 'betting-shaped'."""
    summ, payers = r.clusters.get(mid, (None, None))
    if summ is None or not summ["flagged"].any():
        return None
    fp = set(payers.loc[payers["cluster"].isin(summ.loc[summ["flagged"], "cluster"]), "payer_id"])
    t = r.tx[(r.tx["merchant_id"] == mid) & r.tx["payer_id"].isin(fp) & r.tx["match_day"]]
    if t.empty:
        return None
    per = t.groupby(["payer_id", "day"]).agg(n=("txn_id", "size"), esc=("escalate", "sum"))
    pid, day = per.sort_values(["esc", "n"], ascending=False).index[0]
    trail = t[(t["payer_id"] == pid) & (t["day"] == day)].sort_values("ts")
    return pid, int(day), trail
