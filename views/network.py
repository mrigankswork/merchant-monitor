"""Network page: merchant graph, link explanations and the confirm → suspend → successor scenario."""
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import config as C
from engine.network import neighbours
from views.common import (BORDER, BORDER2, INK, MUTED, MONO, SERIF, chart, confirm_p6, flash, html, merchant_options,
                          page_header, reset_demo, stage, style_fig, suspend_p6, tier_badge, tier_text)

EDGE_STYLE = {
    "payers": ("Shared payers (IDF-weighted)", "#A3A198"),
    "device": ("Same dashboard device", "#7E2621"),
    "settlement": ("Same settlement account", "#5B4A7A"),
    "director": ("Same director", "#A47A22"),
    "ip_fixed": ("Same fixed login IP", "#8C5A6B"),
    "server_ip": ("Same API server", "#3D6A7A"),
    "ip_cloud": ("Shared cloud IP (noise)", "#E2E0D8"),
    "ip_carrier": ("Shared carrier IP (noise)", "#ECEAE3"),
}
CAT_LABEL = {"Bakery": "Bakeries", "Wallet/recharge top-up": "Top-up", "General merchandise": "General",
             "IT services": "IT services"}
SHORT = {"payers": "payers", "device": "device", "settlement": "account", "director": "director",
         "ip_fixed": "fixed IP", "server_ip": "server", "ip_cloud": "cloud IP", "ip_carrier": "carrier IP"}
WHY = {
    "device": "One physical device running several merchant dashboards: a strong operator signal (+1.0).",
    "settlement": "Money lands in the same bank account (+1.0).",
    "director": "Same director on both onboarding files (+0.8).",
    "ip_fixed": "Same fixed-line login IP (+0.6).",
    "server_ip": "Same API server (+0.6). A common checkout plugin counts only +0.02.",
    "payers": "Each shared payer adds log(N / payer degree), so payers who pay everyone count for almost nothing.",
    "ip_cloud": "Popular cloud hosting that thousands share (+0.05).",
    "ip_carrier": "Mobile carrier NAT, meaningless (+0.0).",
}


def node_notes(r) -> dict:
    """Short on-graph annotations for the scenario."""
    notes = {}
    hot_conf = set(r.confirmed) | set(r.suspended)
    for mid in r.tiers.index:
        if "successor" in r.tier3_why[mid]:
            s = r.successors[(r.successors["merchant_id"] == mid) & r.successors["flagged"]].iloc[0]
            notes[mid] = f"likely successor · {s['share']:.0%} of bettors"
        elif mid in hot_conf:
            notes[mid] = "suspended" if mid in r.suspended else "confirmed front"
        elif hot_conf and r.tiers[mid] == 3 and any(o in hot_conf for o in neighbours(r.edges, mid)["other"]):
            notes[mid] = "linked to confirmed front"
    return notes


def _label_pos(r, ids, pos):
    """Cluster nodes label away from the cluster centre; grid nodes label above."""
    core = [i for i in ids if pos[i][0] < 1.9]
    out = []
    for i in ids:
        x, y = pos[i]
        if i not in core:
            out.append("top center")
        elif abs(x) < 0.2:
            out.append("top center" if y > 0 else "bottom center")
        else:
            out.append("middle left" if x < 0 else "middle right")
    return out


def figure(r, show_weak: bool, focus: str):
    m = r.merchants.set_index("merchant_id")
    pos = r.extra["pos"]
    e = r.edges[r.edges["weight"] >= (0.02 if show_weak else C.WEAK_EDGE_HIDE)]
    fig = go.Figure()
    shown = set()
    for etype, (label, color) in EDGE_STYLE.items():
        sub = e[e["main_type"] == etype]
        for w_lo, w_hi, width, op in [(0, 0.3, 0.6, 0.35), (0.3, C.LINK_THRESHOLD, 1.6, 0.5),
                                      (C.LINK_THRESHOLD, 99, 4.5, 0.9)]:
            s = sub[(sub["weight"] >= w_lo) & (sub["weight"] < w_hi)]
            if s.empty:
                continue
            xs, ys = [], []
            for a, b in zip(s["a"], s["b"]):
                xs += [pos[a][0], pos[b][0], None]
                ys += [pos[a][1], pos[b][1], None]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width), opacity=op,
                                     hoverinfo="skip", name=label, legendgroup=etype, showlegend=etype not in shown))
            shown.add(etype)
    vol = r.f["volume_inr"].reindex(m.index).fillna(0)
    size = np.where([pos[i][0] >= 1.9 for i in m.index], 10, 14 + 34 * np.sqrt(vol / vol.max()))
    tiers = r.tiers.reindex(m.index)
    notes = node_notes(r)
    labels = []
    for i in m.index:
        if not m.at[i, "is_persona"]:
            labels.append("")
            continue
        if pos[i][0] >= 1.9:  # grid zone: short key keeps the grid readable
            labels.append(f"<b>{m.at[i, 'persona_key']}</b>")
            continue
        nm = m.at[i, "name"].replace(" (rented)", "").replace(" (viral)", "")
        labels.append(f"<span style='font-family:Newsreader, Georgia, serif;font-size:15px'>{nm}</span>"
                      + (f"<br><span style='color:#B4532A;font-family:Geist Mono, monospace;font-size:10px'>"
                         f"{notes[i].upper()}</span>" if i in notes else ""))
    fig.add_trace(go.Scatter(
        x=[pos[i][0] for i in m.index], y=[pos[i][1] for i in m.index], mode="markers+text",
        marker=dict(size=size, color=[C.TIER_COLOR[int(t)] for t in tiers], opacity=0.95,
                    line=dict(width=[4 if i == focus else 1.2 for i in m.index],
                              color=[INK if i == focus else "white" for i in m.index])),
        text=labels, textposition=_label_pos(r, m.index, pos),
        textfont=dict(size=12, color=INK),
        customdata=np.stack([m.index, m["name"], m["declared_category"], [tier_text(int(t)) for t in tiers]], axis=-1),
        hovertemplate="<b>%{customdata[1]}</b><br>%{customdata[2]}<br>%{customdata[3]}<br>"
                      "<i>click to inspect</i><extra></extra>",
        showlegend=False))
    style_fig(fig, "Merchant network", 620,
              sub="node colour = tier · size = volume · thick edge = linked (weight ≥ 0.8) · click a node to inspect")
    fig.update_layout(legend=dict(orientation="h", y=-0.02, yanchor="top", x=0, xanchor="left"),
                      clickmode="event+select", dragmode=False)
    grid = [i for i in m.index if pos[i][0] >= 1.9]
    for cat in C.CATEGORIES:
        ys = [pos[i][1] for i in grid if m.at[i, "declared_category"] == cat]
        if ys:
            fig.add_annotation(x=4.2, y=max(ys), text=f"{CAT_LABEL[cat].upper()} · {len(ys)}", showarrow=False,
                               xanchor="left", font=dict(size=10, family=MONO, color=MUTED))
    fig.add_annotation(x=0, y=1.42, text="LINKED CLUSTERS · WEIGHT ≥ 0.8", showarrow=False,
                       font=dict(size=10, family=MONO, color=MUTED))
    fig.add_annotation(x=3.6, y=1.42, text="EVERYONE ELSE · NO LINK ≥ 0.8", showarrow=False,
                       font=dict(size=10, family=MONO, color=MUTED))
    fig.update_xaxes(visible=False, range=[-2.35, 4.95])
    fig.update_yaxes(visible=False, range=[-1.25, 1.55])
    return fig, len(fig.data) - 1


def link_table(r, mid):
    nb = neighbours(r.edges, mid, linked_only=False)
    nb = nb[nb["weight"] >= 0.1].sort_values("weight", ascending=False).head(12)
    names = r.merchants.set_index("merchant_id")["name"]
    rows = []
    for e in nb.itertuples(index=False):
        parts = [f"{SHORT[k]} {getattr(e, k):.2f}" for k in EDGE_STYLE if getattr(e, k) > 0]
        rows.append({"Linked merchant": f"{names[e.other]}  ·  T{int(r.tiers[e.other])}",
                     "Weight": e.weight, "Link types": " · ".join(parts)})
    return pd.DataFrame(rows)


def stepper(r):
    s = stage()
    ss = st.session_state
    steps = [("radar", "Detect", "Fronts flagged from their own money and signals"),
             ("gavel", "Confirm", "Analyst confirms Royal Cakes; linked merchants go to review"),
             ("block", "Suspend and sweep", "Suspend it, advance to day 60, find the successor")]
    cols = st.columns([1, 1, 1, 0.55], gap="small", vertical_alignment="top")
    for i, (icon, title, sub) in enumerate(steps):
        done, now = i < s, i == s
        border = INK if now else BORDER2 if done else BORDER
        badge = "done" if done else "current" if now else f"step {i + 1}"
        with cols[i]:
            html(f'<div class="card" style="padding:14px 16px;border-color:{border};'
                 f'{"border-top:2px solid " + INK + ";" if now else ""}">'
                 f'<div style="display:flex;justify-content:space-between;align-items:baseline">'
                 f'<span class="micro">{i + 1:02d}</span><span class="micro" style="color:{INK if now else MUTED}">'
                 f'{badge}</span></div><div class="serif" style="font-size:1.25rem;margin:4px 0 2px">{title}</div>'
                 f'<div style="color:{MUTED};font-size:.82rem;min-height:2.5em">{sub}</div></div>')
            if i == 0:
                st.button("Runs daily, automatically", disabled=True, width="stretch",
                          key="net_auto")
            elif i == 1:
                st.button("Confirm Royal Cakes", on_click=confirm_p6, disabled="M-P6" in ss.confirmed,
                          type="primary" if now else "secondary", width="stretch",
                          key="net_confirm")
            elif i == 2:
                st.button("Suspend and advance", on_click=suspend_p6, disabled=bool(ss.suspended),
                          type="primary" if now else "secondary", width="stretch",
                          key="net_suspend")
    with cols[3]:
        st.button("Reset", on_click=reset_demo, key="reset_net", width="stretch")


def stats(r):
    e = r.edges
    linked = e[e["linked"]]
    cluster = set(linked["a"]) | set(linked["b"])
    noise_only = e[(e["weight"] < C.WEAK_EDGE_HIDE)]
    noise_types = e[(e["ip_cloud"] > 0) | (e["ip_carrier"] > 0) | ((e["server_ip"] > 0) & (e["server_ip"] < 0.1))]
    items = [(f"{len(cluster)}", "merchants in linked clusters"), (f"{len(linked)}", "links at or above 0.8"),
             (f"{len(noise_only):,}", "weak links ignored (< 0.3)"),
             (f"{len(noise_types):,}", "shared cloud / carrier IP or common plugin pairs, none escalated")]
    cols = st.columns(4, gap="small")
    for col, (v, l) in zip(cols, items):
        col.markdown(f'<div class="stat"><div class="v">{v}</div><div class="l">{l}</div></div>',
                     unsafe_allow_html=True)


def render(r):
    ss = st.session_state
    page_header("Network tracer", "Front-merchant network",
                "Merchants linked by shared bettors, devices, accounts, directors and servers. Common noise links "
                "(cloud and carrier IPs, popular checkout plugins) are drawn faintly and never add up to a link.")
    names = r.merchants.set_index("merchant_id")["name"]
    if ss.suspended:
        for row in r.successors[r.successors["flagged"]].itertuples(index=False):
            flash(4, "Successor found", f"<b>{names[row.merchant_id]}</b> is the likely successor of "
                                    f"{names[row.predecessor]}: {row.share:.0%} of its bettors moved there within "
                                    f"{C.SUCCESSOR_WINDOW} days of the suspension.")
    elif ss.confirmed:
        hot = [names[m] for m in r.tiers.index if r.sig.at[m, "network"] and r.tiers[m] == 3]
        flash(3, "Network sweep", f"<b>Royal Cakes & Bakes confirmed as a front.</b> The network sweep put "
                               f"{', '.join(hot)} under review. A link alone never confirms a front.")
    stepper(r)
    st.write("")
    stats(r)
    st.write("")

    ids, opts = merchant_options(r)
    if ss.net_focus not in ids:
        ss.net_focus = "M-P6"
    show_weak = st.toggle("Show weak and noise links", value=False,
                          help="Cloud IPs, carrier IPs and the common checkout plugin are shared by many genuine "
                               "merchants. They are drawn, but never add up to a link.")
    fig, node_trace = figure(r, show_weak, ss.net_focus)
    ev = chart(fig, on_select="rerun", selection_mode="points", key=f"netfig{ss.table_ver}")
    pts = [p for p in (ev.selection.points if ev else []) if p.get("curve_number") == node_trace]
    if pts and pts[0].get("customdata"):
        picked = pts[0]["customdata"][0]
        if picked != ss.net_focus:
            ss.net_focus = picked
            ss.table_ver += 1
            st.rerun()

    left, right = st.columns([1.35, 1], gap="medium")
    with left:
        if ss.get("nsel") != ss.net_focus:
            ss.nsel = ss.net_focus
        focus = st.selectbox("Inspect merchant", ids, key="nsel", format_func=lambda i: opts[i],
                             on_change=lambda: ss.update(net_focus=ss.nsel))
        ss.net_focus = focus
        lt = link_table(r, focus)
        html(f'<div class="card" style="padding:14px 16px;margin-bottom:10px"><div style="display:flex;'
             f'justify-content:space-between;align-items:center;gap:8px"><span class="serif" style="font-size:1.3rem">'
             f'{escape(opts[focus])}</span>{tier_badge(int(r.tiers[focus]))}</div><div style="color:{MUTED};'
             f'font-size:.84rem;margin-top:4px">{len(lt)} link(s) above 0.1 · '
             f'{int((lt["Weight"] >= C.LINK_THRESHOLD).sum()) if len(lt) else 0} at or above the 0.8 threshold'
             f'</div></div>')
        if not lt.empty:
            st.dataframe(lt, hide_index=True, width="stretch", height=min(320, 38 + 35 * len(lt)),
                         column_config={"Weight": st.column_config.ProgressColumn(
                             format="%.2f", min_value=0, max_value=3.5, width="small"),
                             "Link types": st.column_config.TextColumn(width="large")})
    with right:
        rows = "".join(f"<div style='font-size:.86rem;margin:7px 0;line-height:1.45'><span style='display:inline-block;"
                       f"width:14px;height:2px;background:{EDGE_STYLE[k][1]};vertical-align:middle;margin-right:8px'>"
                       f"</span><b style='font-weight:500'>{EDGE_STYLE[k][0]}</b> · {WHY[k]}</div>"
                       for k in ["device", "settlement", "director", "ip_fixed", "server_ip", "payers", "ip_cloud",
                                 "ip_carrier"])
        html(f'<div class="card" style="margin-top:28px"><div class="micro" style="margin-bottom:8px">'
             f'Why each link type counts</div>{rows}</div>')
