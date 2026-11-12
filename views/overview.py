"""Portfolio page: KPIs, tier distribution, demo personas and the full merchant table."""
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from engine import config as C
from engine import explain, response
from views.common import (INK, MUTED, detection, flash, goto, html, inr, page_header, section, tier_badge, tier_text,
                          timeline_progress)

SIGNALS = [("channel", "CH"), ("matchday", "MD"), ("network", "NW"), ("national", "NL")]
CAT_SHORT = {"Wallet/recharge top-up": "Wallet / top-up", "General merchandise": "General merch."}


def table(r) -> pd.DataFrame:
    m = r.merchants.set_index("merchant_id")
    persona_rank = m["persona_key"].map(lambda k: int(k[1:]) if isinstance(k, str) else 99)
    codes = [" ".join(c for key, c in SIGNALS if r.sig.at[mid, key]) or "—" for mid in m.index]
    df = pd.DataFrame({
        "merchant_id": m.index,
        "Merchant": [m.at[i, "name"] + (f"  ·  {m.at[i, 'persona_key']}" if m.at[i, "is_persona"] else "")
                     for i in m.index],
        "Tier": [tier_text(int(t)) for t in r.tiers[m.index]],
        "Declared": m["declared_category"].replace(CAT_SHORT),
        "City": m["city"],
        "Merchant score": r.f["merchant_score"].values,
        "Max cluster": r.cs["max_cluster_score"].values,
        "Bet cluster · payers": 100 * r.cs["bet_cluster_payer_share"].values,
        "Bet cluster · volume": 100 * r.cs["bet_cluster_volume_share"].values,
        "Gap z": np.round(r.gap["gap_z"].values, 1) + 0.0,
        "Signals": codes,
        "History": np.where(r.f["sufficient"].values, "", "Insufficient"),
        "_tier": r.tiers[m.index].values,
        "_rank": persona_rank.values,
        "_cat": m["declared_category"].values,
    })
    return df.sort_values(["_rank", "_tier", "Merchant"], ascending=[True, False, True]).reset_index(drop=True)


def kpis(r):
    ss = st.session_state
    tiers = r.tiers.value_counts()
    held = response.total_reserve(r, ss.released)
    clusters = int((r.cs["hidden_cluster"] & (r.tiers >= 3)).sum())
    cards = [("Monitored", f"{len(r.tiers)}", "merchants scored today", "#8C8A81")]
    subs = {1: "settle normally", 2: "polite nudge", 3: "flagged volume held", 4: "suspended"}
    for t in (1, 2, 3, 4):
        cards.append((C.TIER_SHORT[t], f"{int(tiers.get(t, 0))}", f"T{t} · {subs[t]}", C.TIER_COLOR[t]))
    cards += [("Reserve held", inr(held), "today, up to 48 h", C.INK),
              ("Clusters", f"{clusters}", "hidden betting groups", C.TIER_COLOR[3])]
    html('<div class="kpis">' + "".join(
        f'<div class="kpi" style="--c:{c}"><div class="l"><i></i>{escape(l)}</div><div class="v">{v}</div>'
        f'<div class="s">{s}</div></div>' for l, v, s, c in cards) + "</div>")
    n = len(r.tiers)
    seg = "".join(f'<div style="width:{100 * tiers.get(t, 0) / n:.2f}%;background:{C.TIER_COLOR[t]}"></div>'
                  for t in (1, 2, 3, 4) if tiers.get(t, 0))
    leg = "".join(f'<span><i style="background:{C.TIER_COLOR[t]}"></i>{C.TIER_LABEL[t]} '
                  f'{100 * tiers.get(t, 0) / n:.0f}%</span>' for t in (1, 2, 3, 4))
    html(f'<div class="dist">{seg}</div><div class="dist-legend">{leg}</div>')


ACTION = {2: "Nudge sent · no hold", 3: "Flagged volume held · review", 4: "Suspended"}


def attention_queue(r):
    """Today's work queue: everything at Tier 2 or above, worst first."""
    ss = st.session_state
    m = r.merchants.set_index("merchant_id")
    q = r.tiers[r.tiers >= 2].sort_values(ascending=False)
    q = q.loc[sorted(q.index, key=lambda i: (-int(q[i]), m.at[i, "persona_key"] if isinstance(m.at[i, "persona_key"], str) else "Z", i))]
    section("Needs attention", f"{len(q)} merchants · worst first")
    if not len(q):
        html('<div class="card">Nothing needs attention today.</div>')
        return
    with st.container(border=True):
        hdr = st.columns([2.1, 1.15, 2.8, 1.9, 0.95], vertical_alignment="center")
        for c, h in zip(hdr, ["Merchant", "Tier", "Why", "Detection", ""]):
            c.markdown(f'<div class="micro">{h}</div>', unsafe_allow_html=True)
        for mid in q.index:
            t = int(q[mid])
            det = detection(r, mid)
            if det is None:
                done, total = timeline_progress()
                det_txt = f"replaying… {done}/{total}"
            elif t == 2:
                det_txt = "Unusual, not betting-shaped"
            elif det["first_flag"] is None:
                det_txt = "Flagged today"
            elif det["onset"] is None or det["lag"] is None:
                det_txt = f"Flagged day {det['first_flag']}"
            else:
                lag = det["lag"]
                det_txt = (f"Flagged day {det['first_flag']} · "
                           + ("same day as onset" if lag <= 0 else f"{lag} day{'s' if lag != 1 else ''} after onset"))
            cols = st.columns([2.1, 1.15, 2.8, 1.9, 0.95], vertical_alignment="center")
            cols[0].markdown(f'<div class="serif" style="font-size:1.08rem;line-height:1.2">{escape(m.at[mid, "name"])}'
                             f'</div><div class="micro">{escape(m.at[mid, "declared_category"])} · '
                             f'{escape(m.at[mid, "city"])}</div>', unsafe_allow_html=True)
            cols[1].markdown(tier_badge(t, short=True), unsafe_allow_html=True)
            cols[2].markdown(f'<div style="font-size:.9rem;color:{INK}">{escape(explain.headline(r, mid))}</div>'
                             f'<div style="font-size:.78rem;color:{MUTED}">{ACTION[t]}</div>', unsafe_allow_html=True)
            cols[3].markdown(f'<div style="font-size:.84rem;color:{MUTED}">{det_txt}</div>', unsafe_allow_html=True)
            if cols[4].button("Open", key=f"q_{mid}", width="stretch"):
                goto("Merchant", mid)
    if ss.cleared:
        names = ", ".join(m.at[c, "name"] for c in ss.cleared)
        st.caption(f"Cleared by analyst and back at Tier 1: {names}.")


def persona_cards(r):
    m = r.merchants.set_index("merchant_id")
    ids = [f"M-P{i}" for i in range(1, 10)]
    for row in range(3):
        cols = st.columns(3, gap="medium")
        for col, mid in zip(cols, ids[row * 3:(row + 1) * 3]):
            t = int(r.tiers[mid])
            with col.container(border=True):
                html(f'<div style="display:flex;justify-content:space-between;align-items:center">'
                     f'<span class="pk">{m.at[mid, "persona_key"]} · {escape(m.at[mid, "city"]).upper()}</span>'
                     f'{tier_badge(t, short=True)}</div>'
                     f'<div class="pc-name">{escape(m.at[mid, "name"])}</div>'
                     f'<div class="pc-meta">{escape(m.at[mid, "declared_category"])}</div>'
                     f'<div class="pc-head" style="--c:{C.TIER_COLOR[t]}">{escape(explain.headline(r, mid))}</div>'
                     f'<div class="pc-stats"><div>Merchant<b>{r.f.at[mid, "merchant_score"]:.2f}</b></div>'
                     f'<div>Max cluster<b>{r.cs.at[mid, "max_cluster_score"]:.2f}</b></div>'
                     f'<div>Gap z<b>{round(r.gap.at[mid, "gap_z"], 1) + 0:.1f}</b></div></div>')
                if st.button("Open", key=f"open_{mid}", icon=":material/arrow_outward:", width="stretch"):
                    goto("Merchant", mid)


def render(r):
    ss = st.session_state
    page_header("Portfolio", "Portfolio risk overview",
                "Every merchant is scored for betting-shaped money and checked against the business it declared. "
                "Only a contradiction backed by an independent signal leads to a hold.")
    names = r.merchants.set_index("merchant_id")["name"]
    if ss.suspended:
        for row in r.successors[r.successors["flagged"]].itertuples(index=False):
            flash(4, "Successor found", f"{names[row.predecessor]} was suspended on day {C.DEFAULT_DAY}. "
                                        f"<b>{names[row.merchant_id]}</b> is its likely successor: {row.share:.0%} of "
                                        f"its bettors moved there within {C.SUCCESSOR_WINDOW} days.")
    elif ss.confirmed:
        hot = [names[m] for m in r.tiers.index if r.sig.at[m, "network"] and r.tiers[m] == 3]
        flash(3, "Network sweep", f"Royal Cakes & Bakes confirmed as a front. Under review: {', '.join(hot)}.")
    kpis(r)
    attention_queue(r)

    section("Demo cohort", "nine scripted merchants · computed live")
    persona_cards(r)

    section("All merchants", "personas first · select a row to open it")
    df = table(r)
    f1, f2, f3 = st.columns([2.4, 1.3, 1.3])
    tiers_sel = f1.pills("Tier", [1, 2, 3, 4], selection_mode="multi", default=[1, 2, 3, 4],
                         format_func=tier_text, label_visibility="collapsed")
    cats = f2.selectbox("Category", ["All categories"] + list(C.CATEGORIES), label_visibility="collapsed")
    q = f3.text_input("Search", placeholder="Search merchant", label_visibility="collapsed")
    if tiers_sel:
        df = df[df["_tier"].isin(tiers_sel)]
    if cats != "All categories":
        df = df[df["_cat"] == cats]
    if q:
        df = df[df["Merchant"].str.contains(q, case=False, regex=False)]
    view = df.drop(columns=["merchant_id", "_tier", "_rank", "_cat"])
    pctc = dict(format="%.0f%%", width="small")
    sel = st.dataframe(
        view, hide_index=True, width="stretch", height=min(520, 38 + 35 * max(1, len(view))),
        on_select="rerun", selection_mode="single-row", key=f"tbl{ss.table_ver}",
        column_config={
            "Merchant": st.column_config.TextColumn(width="medium"),
            "Tier": st.column_config.TextColumn(width="small"),
            "Declared": st.column_config.TextColumn(width="small"),
            "City": st.column_config.TextColumn(width="small"),
            "Merchant score": st.column_config.ProgressColumn(
                format="%.2f", min_value=0, max_value=1, width="small",
                help="Payer-weighted bet-shape score of the whole merchant (each payer counts once)"),
            "Max cluster": st.column_config.ProgressColumn(
                format="%.2f", min_value=0, max_value=1, width="small",
                help="Highest bet-shape score among the merchant's payer clusters"),
            "Bet cluster · payers": st.column_config.NumberColumn(help="Payer share of clusters scoring ≥ 0.7", **pctc),
            "Bet cluster · volume": st.column_config.NumberColumn(help="Volume share of clusters scoring ≥ 0.7", **pctc),
            "Gap z": st.column_config.NumberColumn(format="%.1f", width="small",
                                                   help="Declaration gap: robust z vs the declared category"),
            "Signals": st.column_config.TextColumn(width="small",
                                                   help="CH channel · MD match-day · NW network · NL national list"),
            "History": st.column_config.TextColumn(width="small"),
        })
    st.caption("Signals: CH channel anomaly · MD match-day uplift · NW link to a flagged merchant · NL national list "
               "hit (simulated). Insufficient = fewer than 14 active days, 200 transactions or 30 payers.")
    rows = sel.selection.rows if sel else []
    if rows:
        goto("Merchant", df.iloc[rows[0]]["merchant_id"])
