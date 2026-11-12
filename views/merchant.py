"""Merchant deep-dive page."""
from html import escape

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from engine import config as C
from engine import explain, response
from views.common import (BORDER, FAINT, INK, MUTED, SERIES, chart, clear_merchant, confirm_front, detection, html,
                          inr, mask_payer, merchant_options, meter, ms, page_header, reason_rows, reopen_merchant,
                          request_docs, style_fig, tier_badge, timeline_progress)

TABS = ["Payer clusters", "Money shape", "Timeline", "Channel and origin", "Settlement and reserve",
        "Earned Access and national checks"]
CLUSTER_GREYS = ["#8C8A81", "#B5B2A8", "#5F5E57", "#D0CDC3"]
BET_COLOR = SERIES["bet"]


def _bet_clusters(r, mid):
    summ, _ = r.clusters[mid]
    return set(summ.loc[summ["score"] >= C.HIDDEN_SCORE, "cluster"]) if len(summ) > 1 or \
        r.f.at[mid, "merchant_score"] >= C.MERCHANT_SCORE_FLAG else set()


def _peer_ids(r, cat):
    m = r.merchants
    return set(m.loc[(m["declared_category"] == cat) & ~m["persona_key"].isin(C.FRONT_PERSONAS), "merchant_id"])


def _title(text, sub=""):
    html(f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin:6px 0 10px">'
         f'<div class="serif" style="font-size:1.25rem">{text}</div><div class="micro">{sub}</div></div>')


# ------------------------------------------------------------ top section
def header(r, ids, names):
    ss = st.session_state
    m = r.merchants.set_index("merchant_id").loc[ss.selected]
    page_header("Merchant deep-dive", m["name"])
    left, right = st.columns([3.3, 1], vertical_alignment="center")
    with right:
        if ss.get("msel") != ss.selected:
            ss.msel = ss.selected
        mid = st.selectbox("Switch merchant", ids, key="msel", format_func=lambda i: names[i],
                           on_change=lambda: ss.update(selected=ss.msel), label_visibility="collapsed")
    m = r.merchants.set_index("merchant_id").loc[mid]
    onboard = int(m["onboard_day"])
    onboard_txt = f"Onboarded day {onboard}" if onboard > 0 else f"Merchant for {-onboard + r.current_day} days"
    chips = [("category", f"{m['declared_category']} · MCC {m['mcc']}"), ("location_on", m["city"]),
             ("event", onboard_txt), ("integration_instructions", f"Integration: {m['integration_type']}")]
    if m["registered_domain"]:
        chips.append(("language", m["registered_domain"]))
    if m.get("fast_track_video_kyc", False):
        chips.append(("verified_user", "Fast-track video KYC"))
    with left:
        html("".join(f'<span class="chip">{ms(i)}{escape(str(t))}</span>' for i, t in chips))
    return mid


def assessment(r, mid):
    tier = int(r.tiers[mid])
    color = C.TIER_COLOR[tier]
    title = {1: "Why it stays green", 2: "Why it is nudged", 3: "Why it is held for review",
             4: "Why it is a confirmed front"}[tier]
    items = explain.reasons_tagged(r, mid) if tier >= 2 else explain.green_notes(r, mid)
    head = explain.headline(r, mid)
    extra = ""
    model_tier = int(r.extra.get("model_tiers", r.tiers)[mid])
    if mid in st.session_state.cleared:
        title = "Cleared by analyst"
        head = f"Alert cleared as a false positive · model findings kept on file (model: T{model_tier})"
        items = explain.reasons_tagged(r, mid, tier=model_tier) or items
    if tier == 2:
        sugg = r.gap.at[mid, "recat_suggestion"]
        text = f"Possible miscategorisation: looks like {sugg}." if sugg else response.NUDGE_NOTICE
        extra = f'<div class="notice" style="margin-top:14px"><b>Nudge sent to merchant</b>{escape(text)}</div>'
    if tier == 4:
        extra = ('<div class="notice" style="margin-top:14px"><b>Actions taken</b>'
                 + " · ".join(response.TIER4_ACTIONS) + "</div>")
    html(f'<div class="card"><div style="display:flex;justify-content:space-between;align-items:center;gap:12px">'
         f'<div class="micro">Assessment</div>{tier_badge(tier, large=True)}</div>'
         f'<div class="serif" style="font-size:1.75rem;line-height:1.2;margin:10px 0 2px">{title}</div>'
         f'<div style="color:{color};font-size:.95rem;margin-bottom:14px">{escape(head)}</div>'
         f'{reason_rows(items, color)}{extra}</div>')


def side_panel(r, mid):
    tier = int(r.tiers[mid])
    color = C.TIER_COLOR[tier]
    cat = r.merchants.set_index("merchant_id").at[mid, "declared_category"]
    peers = explain.peer_stats(r, cat)
    ms_score, cl_score = float(r.f.at[mid, "merchant_score"]), float(r.cs.at[mid, "max_cluster_score"])
    meters = (meter("Whole-merchant bet-shape", ms_score, INK, C.MERCHANT_SCORE_FLAG, peers["score"],
                    "Payer-weighted · line = 0.7 flag · box = peer median")
              + meter("Highest payer-cluster score", cl_score, BET_COLOR if cl_score >= 0.7 else INK,
                      C.HIDDEN_SCORE, None, f"Betting clusters score above 0.8 · gap z = "
                                            f"{round(r.gap.at[mid, 'gap_z'], 1) + 0:.1f}"))
    res = response.reserve(r, mid)
    held = 0 if mid in st.session_state.released else res["reserve_held"]
    if mid in st.session_state.cleared:
        act = ("Settle normally · cleared", "An analyst cleared this alert. Settlement runs normally; the model keeps "
                                             "scoring daily and the case can be reopened.")
    elif tier == 1:
        act = ("Settle normally", "No action. The merchant is re-scored every day.")
    elif tier == 2:
        act = ("Send a nudge", "Ask the merchant to confirm or update its business details. Settlement is unaffected.")
    elif tier == 3:
        act = ("Hold flagged volume · review",
               f"Hold {inr(held)} of today's {inr(res['today'])}; {inr(res['today'] - held)} settles normally. "
               f"Released within {C.RESERVE_HOURS} h, or earlier on proof of fulfilment.")
    else:
        act = ("Front confirmed", "Acceptance suspended and settlement held under contract.")
    sig = r.sig.loc[mid]
    suff = bool(r.f.at[mid, "sufficient"])
    items = [("Betting-shaped money", bool(r.cs.at[mid, "hidden_cluster"]) or (suff and ms_score >= 0.7)),
             ("Contradicts declared business", bool(r.gap.at[mid, "contradicts"])),
             ("Channel anomaly", bool(sig["channel"])), ("Match-day uplift", bool(sig["matchday"])),
             ("Network link to flagged merchant", bool(sig["network"])), ("National list hit", bool(sig["national"]))]
    sig_html = "".join(f'<div class="sig {"on" if on else "off"}"><span>{label}</span>'
                       f'<span class="v">{"Present" if on else "—"}</span></div>' for label, on in items)
    hist = "" if suff else ('<div class="notice" style="margin-top:12px"><b>Insufficient history</b>Monitoring. '
                            'Shape-based tiers need 14 active days, 200 transactions and 30 payers.</div>')
    html(f'<div class="card"><div class="micro" style="margin-bottom:10px">Scores</div>{meters}'
         f'<div class="act" style="--c:{color}"><div class="micro" style="margin-bottom:4px">Recommended action</div>'
         f'<div class="h"><i></i>{act[0]}</div><div class="b">{escape(act[1])}</div></div>'
         f'<div class="micro" style="margin:14px 0 4px">Evidence</div>{sig_html}'
         f'<div style="color:{MUTED};font-size:.76rem;margin-top:8px">Tier 3 needs the first two plus at least one '
         f'of the four independent signals.</div>{hist}</div>')


def case_summary(r, mid):
    tier = int(r.tiers[mid])
    html(f'<div class="card" style="height:100%;border-left:3px solid {C.TIER_COLOR[tier]}">'
         f'<div class="micro">Case summary</div><div class="serif" style="font-size:1.28rem;line-height:1.5;'
         f'margin-top:8px;color:{INK}">{escape(explain.narrative(r, mid))}</div></div>')


def detection_card(r, mid):
    det = detection(r, mid)
    if det is None:
        done, total = timeline_progress()
        html(f'<div class="card" style="height:100%"><div class="micro">Time to detection</div>'
             f'<div style="color:{MUTED};font-size:.9rem;margin-top:10px">Replaying every day to measure when this '
             f'merchant was first flagged… {done}/{total} days. This runs once and is cached.</div></div>')
        return
    first, onset, lag = det["first_flag"], det["onset"], det["lag"]
    if first is None:
        big, sub = "Never flagged", "Tier 3 was not reached on any day so far"
    elif onset is None:
        big, sub = f"Day {first}", "first flagged"
    else:
        big = "Same day" if lag <= 0 else f"{lag} day{'s' if lag != 1 else ''}"
        sub = f"from first suspicious activity (day {onset}) to first flag (day {first})"
    html(f'<div class="card" style="padding-bottom:6px"><div class="micro">Time to detection</div>'
         f'<div class="serif" style="font-size:2.1rem;line-height:1.1;margin-top:8px">{big}</div>'
         f'<div style="color:{MUTED};font-size:.84rem;margin-top:4px">{sub}</div></div>')
    hist = det["history"]
    m = r.merchants.set_index("merchant_id").loc[mid]
    days = list(range(1, r.current_day + 1))
    tier_by_day = dict(zip(hist["day"], hist["tier"]))
    onboard = int(m["onboard_day"])
    col, txt = [], []
    for d in days:
        tv = tier_by_day.get(d)
        if d < max(1, onboard):
            col.append("#ECEAE3"); txt.append(f"Day {d} · not yet onboarded")
        elif tv is None:
            col.append("#E2E0D8"); txt.append(f"Day {d} · before replay window")
        else:
            col.append(C.TIER_COLOR[tv]); txt.append(f"Day {d} · T{tv} {C.TIER_SHORT[tv]}")
    fig = go.Figure(go.Bar(x=days, y=[1] * len(days), marker_color=col, hovertext=txt, hoverinfo="text",
                           marker_line_width=0))
    if onset:
        fig.add_vline(x=onset - 0.5, line=dict(color=INK, width=1, dash="dot"))
        fig.add_annotation(x=onset - 0.5, y=1.25, text=f"onset d{onset}", showarrow=False, xanchor="left",
                           font=dict(size=10, family="Geist Mono, monospace", color=MUTED))
    style_fig(fig, "", 92, legend=False)
    fig.update_layout(margin=dict(l=4, r=4, t=14, b=4), bargap=0.12)
    fig.update_xaxes(showgrid=False, tickfont=dict(size=9), dtick=10, range=[0.5, r.current_day + 0.5])
    fig.update_yaxes(visible=False, range=[0, 1.45])
    chart(fig)
    st.caption("Tier on each day, replayed day by day.")


def case_actions(r, mid):
    ss = st.session_state
    tier = int(r.tiers[mid])
    model_tier = int(r.extra.get("model_tiers", r.tiers)[mid])
    cleared = mid in ss.cleared
    if cleared:
        html(f'<div class="notice" style="margin-bottom:10px"><b>Cleared by analyst</b>The model still scores this '
             f'merchant T{model_tier} · {C.TIER_LABEL[model_tier]}. It stays at T1 unless reopened.</div>')
    html('<div class="micro" style="margin:2px 0 8px">Analyst actions</div>')
    c = st.columns(3)
    if c[0].button("Confirm front", key=f"cf_{mid}", disabled=tier != 3 and model_tier != 3 or tier == 4,
                   width="stretch", type="primary" if tier == 3 else "secondary"):
        confirm_front(mid)
        st.rerun()
    if cleared:
        if c[1].button("Reopen case", key=f"ro_{mid}", width="stretch"):
            reopen_merchant(mid)
            st.rerun()
    elif c[1].button("False positive", key=f"cl_{mid}", disabled=tier not in (2, 3), width="stretch",
                     help="Clear the alert: the merchant returns to Tier 1 and stays there unless reopened"):
        clear_merchant(mid)
        st.rerun()
    if c[2].button("Request docs", key=f"rd_{mid}", disabled=tier < 2 and not cleared, width="stretch",
                   help="Ask for invoices, delivery records and a photo of the business premises"):
        request_docs(mid)
        st.rerun()
    log = ss.cases.get(mid, [])
    if log:
        rows = "".join(f'<tr><td class="mono" style="white-space:nowrap">Day {e["day"]}</td><td>{escape(e["action"])}</td>'
                       f'<td style="color:{MUTED}">{escape(e["note"])}</td></tr>' for e in reversed(log))
        html(f'<div class="card" style="padding:4px 14px;margin-top:6px"><table class="tbl"><tr><th>When</th>'
             f'<th>Action</th><th>Detail</th></tr>{rows}</table></div>')


def payer_trail(r, mid):
    res = explain.payer_trail(r, mid)
    if res is None:
        return
    pid, day, tr = res
    secs = (tr["ts"] - tr["ts"].dt.normalize()).dt.total_seconds() / 3600
    secs = np.where(tr["day"].values > day, secs + 24, secs)  # after-midnight deposits continue the same session
    fig = go.Figure()
    fig.add_vrect(x0=C.PREMATCH[0], x1=C.PREMATCH[1], fillcolor=BET_COLOR, opacity=0.07, line_width=0,
                  annotation_text="pre-match", annotation_position="top left",
                  annotation_font=dict(size=10, color=BET_COLOR))
    fig.add_vrect(x0=C.INPLAY[0], x1=C.INPLAY[1], fillcolor=INK, opacity=0.04, line_width=0,
                  annotation_text="in play", annotation_position="top left", annotation_font=dict(size=10, color=MUTED))
    fig.add_vline(x=C.MATCH_START_H, line=dict(color=INK, width=1, dash="dot"))
    fig.add_trace(go.Scatter(x=secs, y=tr["amount_inr"], mode="lines+markers+text", line=dict(color=BET_COLOR, width=1.5,
                             shape="hv"), marker=dict(size=9, color=BET_COLOR),
                             text=[f"₹{a:,.0f}" for a in tr["amount_inr"]], textposition="top center",
                             textfont=dict(family="Geist Mono, monospace", size=10, color=INK), hoverinfo="skip"))
    region = tr["payer_region"].dropna()
    who = f"{mask_payer(pid)} · paying from {region.iloc[0] if len(region) else 'unknown region'}"
    style_fig(fig, "One payer's match day", 300, legend=False,
              sub=f"{who} · day {day} (T20 match at 19:30) · {len(tr)} deposits, ₹{tr['amount_inr'].sum():,.0f} in total")
    fig.update_xaxes(title="time of day (IST)", range=[12, 26.5], tickvals=list(range(12, 27, 2)),
                     ticktext=[f"{h % 24:02d}:00" for h in range(12, 27, 2)])
    fig.update_yaxes(title="₹ per deposit", rangemode="tozero")
    chart(fig)
    st.caption("A bakery customer pays once, a few hundred rupees, in the morning or evening. This payer deposits "
               "again and again in round amounts, doubling the stake, from before the match until late night.")


# ------------------------------------------------------------ tabs
def tab_clusters(r, mid):
    summ, payers = r.clusters[mid]
    bet = _bet_clusters(r, mid)
    fig = go.Figure()
    order = sorted(payers["cluster"].unique(), key=lambda k: (k in bet, k))
    grey_i = 0
    for k in order:
        p = payers[payers["cluster"] == k]
        row = summ[summ["cluster"] == k].iloc[0]
        is_bet = k in bet
        color = BET_COLOR if is_bet else CLUSTER_GREYS[grey_i % len(CLUSTER_GREYS)]
        grey_i += 0 if is_bet else 1
        label = (f"{'Betting group' if is_bet else 'Group'} {k} · {row['payer_share']:.0%} of payers · "
                 f"{row['volume_share']:.0%} of volume · score {row['score']:.2f}")
        fig.add_trace(go.Scattergl(
            x=p["pc1"], y=p["pc2"], mode="markers", name=label,
            marker=dict(size=9 if is_bet else 5, color=color, opacity=0.95 if is_bet else 0.5,
                        line=dict(width=0.6 if is_bet else 0, color="white")),
            customdata=np.stack([p["txns_per_active_day"], p["round_share"], p["prematch_share"],
                                 p["mean_amount"]], axis=-1),
            hovertemplate="payments/day %{customdata[0]:.1f}<br>round %{customdata[1]:.0%}<br>"
                          "pre-match %{customdata[2]:.0%}<br>mean ₹%{customdata[3]:,.0f}<extra></extra>"))
    sub = ("Normal overall, betting inside: one small group of payers behaves completely differently"
           if r.cs.at[mid, "hidden_cluster"] and r.f.at[mid, "merchant_score"] < 0.6
           else "Each dot is a payer, grouped by payment behaviour (Gaussian mixture, k chosen by BIC)")
    style_fig(fig, "Payers by behaviour", 460, sub=sub)
    fig.update_layout(legend=dict(orientation="v", y=1, yanchor="top", x=1.0, xanchor="right",
                                  bgcolor="rgba(255,255,255,.92)", bordercolor=BORDER, borderwidth=1))
    fig.update_xaxes(title="principal component 1", showticklabels=False)
    fig.update_yaxes(title="principal component 2", showticklabels=False)
    chart(fig)
    show = summ.assign(**{
        "Group": summ["cluster"].map(lambda k: f"{'Betting · ' if k in bet else ''}{k}"),
        "Payers": summ["payer_count"], "Payer share": 100 * summ["payer_share"],
        "Volume share": 100 * summ["volume_share"], "Bet-shape score": summ["score"],
        "Payments / active day": summ["mean_txns_per_active_day"], "Round share": 100 * summ["mean_round_share"],
        "Pre-match share": 100 * summ["mean_prematch_share"], "Remote payers": 100 * summ["mean_remote_flag"],
        "Mean ticket": summ["mean_amount"], "First seen": summ["first_day"].map(lambda d: f"day {d}"),
    })[["Group", "Payers", "Payer share", "Volume share", "Bet-shape score", "Payments / active day", "Round share",
        "Pre-match share", "Remote payers", "Mean ticket", "First seen"]]
    pctc = st.column_config.NumberColumn(format="%.0f%%")
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "Payer share": pctc, "Volume share": pctc, "Round share": pctc, "Pre-match share": pctc,
        "Remote payers": pctc,
        "Bet-shape score": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
        "Payments / active day": st.column_config.NumberColumn(format="%.1f"),
        "Mean ticket": st.column_config.NumberColumn(format="₹%.0f")})
    cs = r.cs.loc[mid]
    if not (cs["hidden_cluster"] and cs["bet_first_day"] and cs["bet_first_day"] > 1):
        payer_trail(r, mid)
    if cs["hidden_cluster"] and cs["bet_first_day"] and cs["bet_first_day"] > 1:
        st.caption(f"Volume share is over the full window. Since the betting group first appeared on day "
                   f"{int(cs['bet_first_day'])}, it carries {cs['bet_volume_share_since']:.0%} of this merchant's "
                   f"volume.")
    payer_trail(r, mid)


def tab_shape(r, mid):
    m = r.merchants.set_index("merchant_id").loc[mid]
    cat = m["declared_category"]
    t = r.tx[r.tx["merchant_id"] == mid]
    peers = r.tx[r.tx["merchant_id"].isin(_peer_ids(r, cat) - {mid})]
    plural = explain.CAT_PLURAL[cat]
    c1, c2 = st.columns(2, gap="medium")
    bins = np.linspace(1.0, 5.4, 56)
    hm, _ = np.histogram(np.log10(t["amount_inr"]), bins=bins)
    hp, _ = np.histogram(np.log10(peers["amount_inr"]), bins=bins)
    x = 10 ** ((bins[:-1] + bins[1:]) / 2)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=hp / max(hp.sum(), 1), name=f"Peer {plural}", marker_color=SERIES["peer"],
                         width=np.diff(10 ** bins) * 0.92))
    fig.add_trace(go.Bar(x=x, y=hm / max(hm.sum(), 1), name="This merchant", marker_color=INK,
                         opacity=0.85, width=np.diff(10 ** bins) * 0.92))
    style_fig(fig, "Ticket sizes", 380, sub=f"{t['is_round'].mean():.0%} exact multiples of ₹100 · peers "
                                            f"{peers['is_round'].mean():.0%}")
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(type="log", title="amount (₹, log scale)", tickvals=[10, 100, 1000, 10000, 100000],
                     ticktext=["10", "100", "1k", "10k", "1L"])
    fig.update_yaxes(title="share of payments", tickformat=".0%")
    with c1:
        chart(fig)
    hm = np.bincount(t["hour_bucket"], minlength=24) / max(len(t), 1)
    hp = np.bincount(peers["hour_bucket"], minlength=24) / max(len(peers), 1)
    fig = go.Figure()
    fig.add_vrect(x0=15, x1=19, fillcolor=BET_COLOR, opacity=0.06, line_width=0,
                  annotation_text="pre-match", annotation_position="top left",
                  annotation_font=dict(color=BET_COLOR, size=10))
    fig.add_trace(go.Bar(x=list(range(24)), y=hp, name=f"Peer {plural}", marker_color=SERIES["peer"]))
    fig.add_trace(go.Bar(x=list(range(24)), y=hm, name="This merchant", marker_color=INK))
    style_fig(fig, "Hour of day", 380, sub=f"{t['night'].mean():.0%} between midnight and 5am · peers "
                                           f"{peers['night'].mean():.0%}")
    fig.update_layout(barmode="group", bargap=0.2)
    fig.update_xaxes(title="hour (IST)", dtick=3)
    fig.update_yaxes(title="share of payments", tickformat=".0%")
    with c2:
        chart(fig)
    f = r.f.loc[mid]
    if f["sufficient"]:
        st.caption(f"Event-aware peer comparison: normal days vs peers' normal days (ticket z "
                   f"{f['amount_z_normal']:.1f}, hour z {f['hour_z_normal']:.1f}); festival window vs peers' festival "
                   f"window (ticket z {f['amount_z_festival']:.1f}, hour z {f['hour_z_festival']:.1f}). The larger "
                   f"value counts.")


def tab_timeline(r, mid):
    m = r.merchants.set_index("merchant_id").loc[mid]
    measure = st.segmented_control("Measure", ["₹ volume", "Transactions"], key="measure",
                                   label_visibility="collapsed") or "₹ volume"
    count = measure == "Transactions"
    days = list(range(1, r.current_day + 1))
    flagged = response.flagged_payers(r, mid)
    t = r.tx[r.tx["merchant_id"] == mid]
    agg = "size" if count else "sum"
    daily = t.groupby("day")["amount_inr"].agg(agg).reindex(days, fill_value=0)
    bet = t[t["payer_id"].isin(flagged)].groupby("day")["amount_inr"].agg(agg).reindex(days, fill_value=0)
    other = daily - bet
    src = r.f.attrs["daily_n" if count else "daily"]
    peer_med = src.loc[list(_peer_ids(r, m["declared_category"]) - {mid})].median(axis=0)
    lo = C.FESTIVAL_DAYS[0]
    pre = other.loc[max(1, lo - 28): lo - 1]
    scale = pre[pre > 0].median() / max(peer_med.loc[max(1, lo - 28): lo - 1].median(), 1e-9) \
        if (pre > 0).any() else 1
    fig = go.Figure()
    fig.add_vrect(x0=C.FESTIVAL_DAYS[0] - 0.5, x1=C.FESTIVAL_DAYS[1] + 0.5, fillcolor=SERIES["festival"], opacity=0.7,
                  line_width=0, layer="below", annotation_text="festival (synthetic)", annotation_position="top left",
                  annotation_font=dict(color=MUTED, size=10))
    for d in days:
        if C.is_match_day(d):
            fig.add_vline(x=d, line=dict(color="#E2E0D8", width=1, dash="dot"))
    fig.add_trace(go.Scatter(x=days, y=other, name="Other payers" if flagged else "This merchant", stackgroup="v",
                             line=dict(color=INK, width=1.8), fillcolor="rgba(26,26,24,0.06)"))
    if flagged:
        fig.add_trace(go.Scatter(x=days, y=bet, name="Flagged betting group", stackgroup="v",
                                 line=dict(color=BET_COLOR, width=1.8), fillcolor="rgba(180,83,42,0.28)"))
    fig.add_trace(go.Scatter(x=days, y=peer_med * scale, name="Peer median, scaled",
                             line=dict(color=SERIES["peer_line"], width=1.5, dash="dash")))
    for s_mid, s_day in r.suspended.items():
        if s_mid == mid:
            fig.add_vline(x=s_day + 0.5, line=dict(color=C.TIER_COLOR[4], width=1.5),
                          annotation_text="suspended", annotation_position="top right")
    style_fig(fig, f"Daily {'transactions' if count else 'volume'}", 440,
              sub="dotted lines mark T20 match days · shaded band is the festival window")
    fig.update_xaxes(title="day", range=[0.5, r.current_day + 0.5])
    fig.update_yaxes(title="transactions per day" if count else "₹ per day")
    chart(fig)


def tab_channel(r, mid):
    m = r.merchants.set_index("merchant_id").loc[mid]
    t = r.tx[r.tx["merchant_id"] == mid]
    mix = t.groupby("channel")["amount_inr"].sum().sort_values()
    c1, c2 = st.columns([1, 1.15], gap="medium")
    colors = [BET_COLOR if (ch == "S2S_API" and m["integration_type"] != "API") else INK for ch in mix.index]
    fig = go.Figure(go.Bar(x=mix.values / mix.sum(), y=mix.index, orientation="h", marker_color=colors,
                           text=[f"{v / mix.sum():.0%}" for v in mix.values], textposition="outside",
                           textfont=dict(family="Geist Mono, monospace", size=11)))
    style_fig(fig, "Channel mix by volume", 300, legend=False,
              sub=f"declared integration: {m['integration_type']} · rust = undeclared server API")
    fig.update_xaxes(tickformat=".0%", range=[0, 1.18])
    with c1:
        chart(fig)
    api = r.data["api_logs"]
    api = api[(api["merchant_id"] == mid) & (api["day"] <= r.current_day)]
    with c2:
        _title("API servers", "daily aggregates")
        if len(api):
            last7 = api[api["day"] > r.current_day - 7]
            g = api.groupby(["server_ip", "server_asn_label", "server_country"]).agg(
                days=("day", "nunique"), links_created=("payment_links_created", "sum")).reset_index()
            g["links_last_7d"] = g["server_ip"].map(
                last7.groupby("server_ip")["payment_links_created"].sum()).fillna(0)
            g["offshore"] = np.where(g["server_country"] != "IN", "yes", "")
            st.dataframe(g, hide_index=True, width="stretch")
        else:
            st.caption("No API activity.")
        note = ("API use is normal for this declared category." if m["integration_type"] == "API" else
                f"Server-API share from a non-API merchant: {r.f.at[mid, 's2s_undeclared_share']:.0%}.")
        st.caption(f"{note} Payment-link spike (7d vs prior): {r.f.at[mid, 'link_spike']:.1f}×")
    _title("Hosted-checkout referrer domains")
    hosted = t[t["method"] == "HOSTED_CHECKOUT"]
    if len(hosted):
        ref = hosted.groupby("referrer_domain").agg(payments=("txn_id", "size"), volume=("amount_inr", "sum"))
        ref = ref.reset_index().sort_values("payments", ascending=False)
        ref["status"] = np.where(ref["referrer_domain"].isin(C.BLOCKED_DOMAINS), "Blocked list (simulated)",
                                 np.where(ref["referrer_domain"] == m["registered_domain"], "Registered domain",
                                          "Other"))
        st.dataframe(ref, hide_index=True, width="stretch",
                     column_config={"volume": st.column_config.NumberColumn(format="₹%.0f")})
    else:
        st.caption("No hosted-checkout payments.")
    if len(t[t["method"] == "CARD"]) >= 20:
        st.caption(f"Cards: {r.f.at[mid, 'card_tokens_per_100']:.0f} distinct tokens per 100 card payments · "
                   f"prepaid {r.f.at[mid, 'prepaid_share']:.0%} · foreign-issued {r.f.at[mid, 'foreign_share']:.0%}")
    st.caption("Data availability: device ID and referrer exist only for hosted checkout (and some card payments). "
               "UPI QR and UPI intent carry neither.")


def tab_reserve(r, mid):
    res = response.reserve(r, mid)
    ss = st.session_state
    released = mid in ss.released
    held = 0 if released else res["reserve_held"]
    if res["mode"] == "cluster":
        stats = [("Today's volume", res["today"]), ("Genuine payers · settles", res["genuine_today"]),
                 ("Flagged group · today", res["flagged_today"]), ("Reserve held", held)]
    else:
        above = max(0.0, res["today"] - C.RESERVE_MULT * res["baseline"])
        stats = [("Baseline per day", res["baseline"]), (f"Today's volume · day {r.current_day}", res["today"]),
                 ("Above 1.5× baseline", above), ("Reserve held", held)]
    c = st.columns(4)
    for col, (label, val) in zip(c, stats):
        col.markdown(f'<div class="stat"><div class="v">{inr(val)}</div><div class="l">{label}</div></div>',
                     unsafe_allow_html=True)
    st.caption(f"Rule: {res['basis']}.")
    if not res["applies"]:
        st.info("No reserve. Reserves apply only to Tier 3 (high suspicion); settlement runs normally.",
                icon=":material/check_circle:")
        return
    settle = res["today"] - held
    fig = go.Figure()
    fig.add_trace(go.Bar(y=[""], x=[settle], orientation="h", name="Settles normally", marker_color=C.TIER_COLOR[1],
                         text=[inr(settle)], textposition="inside", insidetextfont=dict(color="white")))
    if held > 0:
        fig.add_trace(go.Bar(y=[""], x=[held], orientation="h", name=f"Held up to {C.RESERVE_HOURS} h",
                             marker_color=BET_COLOR, text=[inr(held)], textposition="inside",
                             insidetextfont=dict(color="white")))
    style_fig(fig, "Today's settlement", 160)
    fig.update_layout(barmode="stack", margin=dict(t=50, b=6))
    fig.update_xaxes(showticklabels=False, showgrid=False)
    chart(fig)
    m = r.merchants.set_index("merchant_id").loc[mid]
    html(f'<div class="card" style="max-width:620px;padding:0;overflow:hidden"><div style="padding:10px 16px;'
         f'border-bottom:1px solid {BORDER};display:flex;justify-content:space-between"><span class="micro">Message to '
         f'merchant · app + SMS</span><span class="micro">Day {r.current_day} · 23:59 IST</span></div>'
         f'<div style="padding:14px 16px;font-size:.92rem;line-height:1.55;color:{INK}">Hello {escape(m["name"])}, '
         f'{escape(res["notice"][0].lower() + res["notice"][1:])}<div style="margin-top:10px;color:{MUTED};font-size:.82rem">'
         f'Held today: {inr(held)} · Settling today: {inr(res["today"] - held)}</div></div></div>')
    st.write("")
    if released:
        st.success("Proof of fulfilment uploaded. Reserve released to the merchant.", icon=":material/task_alt:")
    elif st.button("Simulate proof uploaded", type="primary", icon=":material/upload_file:"):
        ss.released = ss.released + [mid]
        from views.common import log_case
        log_case(mid, "Reserve released", "Merchant uploaded invoices and delivery records")
        st.rerun()


def tab_access(r, mid):
    ea = response.earned_access(r, mid)
    cd = ea["clean"]
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        state = f"Unlocked via {ea['route']}" if ea["unlocked"] else "Locked"
        cf = f'<div class="notice" style="margin-top:14px"><b>Counterfactual</b>{escape(ea["counterfactual"])}</div>' \
            if ea["counterfactual"] else ""
        clean_txt = ("0 (national-list hit)" if cd["national_hit"] else
                     f"{cd['total']}" + (f" ({cd['prior']} before day 1 + {cd['observed']} observed)" if cd["prior"]
                                         else ""))
        html(f'<div class="card"><div class="micro">Earned Access</div>'
             f'<div class="serif" style="font-size:1.6rem;margin:6px 0 12px">{state}</div>'
             f'<div class="sig"><span>Always open</span><span>{" · ".join(ea["always_open"])}</span></div>'
             f'<div class="sig"><span>Earned</span><span>{" · ".join(ea["earned"])}</span></div>'
             f'<div class="sig"><span>Clean days</span><span>{clean_txt} of {C.EARNED_ACCESS_DAYS} required</span></div>'
             f'<div class="sig"><span>Fast-track video KYC</span><span>{"Yes" if ea["fast_track"] else "No"}</span></div>'
             f'<div style="color:{MUTED};font-size:.78rem;margin-top:10px">A clean day had at least one transaction, '
             f'no flagged-group or offshore API activity, and the merchant has no national-list hit.</div>{cf}</div>')
    with c2:
        rows = ""
        for chk in response.national_checks(r, mid):
            col = C.TIER_COLOR[4] if chk["hit"] else FAINT
            extra = f" · {escape(chk['detail'])}" if chk.get("detail") else ""
            rows += (f'<div class="sig" style="align-items:flex-start"><span>{escape(chk["check"])}{extra}'
                     f'<br><span class="micro">{chk["source"]}</span></span>'
                     f'<span class="mono" style="font-size:.68rem;letter-spacing:.08em;color:{col}">'
                     f'{"MATCH" if chk["hit"] else "NO MATCH"}</span></div>')
        html(f'<div class="card"><div class="micro">National checks</div>'
             f'<div class="serif" style="font-size:1.6rem;margin:6px 0 12px">'
             f'{"Hit" if ea["national_hit"] else "No hits"}</div>{rows}</div>')


def render(r):
    ss = st.session_state
    ids, names = merchant_options(r)
    if ss.selected not in ids:
        ss.selected = ids[0]
    mid = header(r, ids, names)
    st.write("")
    a, b = st.columns([1.75, 1], gap="medium")
    with a:
        case_summary(r, mid)
    with b:
        detection_card(r, mid)
    st.write("")
    a, b = st.columns([1.75, 1], gap="medium")
    with a:
        assessment(r, mid)
        st.write("")
        case_actions(r, mid)
    with b:
        side_panel(r, mid)
    st.write("")
    default = ss.mtab if ss.mtab in TABS else None
    tabs = st.tabs(TABS, default=default, key=f"mtab{ss.mtab_ver}")
    for tab, fn in zip(tabs, [tab_clusters, tab_shape, tab_timeline, tab_channel, tab_reserve, tab_access]):
        with tab:
            fn(r, mid)
