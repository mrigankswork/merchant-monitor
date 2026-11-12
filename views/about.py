"""How it works: method, tiers, data notes, assumptions and the demo script."""
from html import escape

import streamlit as st

from engine import config as C
from views.common import MUTED, STEPS, apply_step, html, page_header, section, tier_badge

PARTS = [
    ("scatter_plot", "Bet-Shape Score with hidden clusters",
     "Scores how betting-like money is from six shape measures: repeat payments per day, round amounts, stake "
     "escalation, pre-match timing, late-night timing and remote payers. A Gaussian mixture splits each merchant's "
     "payers into groups, so a betting cluster hiding inside genuine sales still stands out."),
    ("rule", "Declaration Gap",
     "Betting-like money only matters when it contradicts the declared business. Each merchant is compared with "
     "others in its declared category (robust z). Round amounts are normal for a top-up merchant and abnormal for a "
     "bakery."),
    ("hub", "Network Tracer",
     "Links merchants through IDF-weighted shared payers, dashboard devices, settlement accounts, directors, fixed "
     "IPs and API servers. Cloud and carrier IPs and common plugins count for almost nothing. After a suspension it "
     "finds the successor merchant the bettors move to."),
    ("lock_clock", "Earned Access",
     f"Counter QR and website checkout are open from day one. Server API, bulk payment links and payouts unlock "
     f"after {C.EARNED_ACCESS_DAYS} clean days with no national hits, or through fast-track video KYC."),
    ("account_balance", "National Intelligence Layer",
     "Checks onboarding phones, settlement accounts and referrer domains against national lists: DoT FRI, the I4C "
     "Suspect Registry and the MeitY blocked list. All three are simulated here."),
    ("balance", "Proportionate Response",
     f"Tier 1 is left alone and Tier 2 gets a polite nudge. In Tier 3 the merchant's genuine customers settle as "
     f"normal and only the flagged payer group's volume is held, for up to {C.RESERVE_HOURS} hours (without a flagged "
     f"group, the surge above {C.RESERVE_MULT}× the pre-anomaly baseline). Tier 4 only follows analyst confirmation."),
]

PIPELINE = [("receipt_long", "Transactions"), ("tune", "Payer features"), ("scatter_plot", "Hidden clusters"),
            ("speed", "Bet-shape score"), ("rule", "Declaration gap"), ("verified", "Independent signals"),
            ("stacked_bar_chart", "Tier"), ("balance", "Response")]

TIERS = [
    (1, "Nothing unusual, or unusual in a way its declared category explains", "Settles normally; re-scored daily"),
    (2, "Shape distance or surge above 3 (robust z / ratio), or a better-fitting category; not betting-shaped",
     "Polite nudge to confirm or update business details"),
    (3, "Betting-shaped money (whole merchant or hidden cluster) and a contradicted declaration and at least one "
        "independent signal; or linked to a confirmed front; or a likely successor",
     f"Genuine payers settle; flagged-group volume (or surge above {C.RESERVE_MULT}× baseline) held up to "
     f"{C.RESERVE_HOURS} h; analyst review"),
    (4, "Analyst confirmation only", "Acceptance suspended, settlement held, STR drafted, I4C referral, network sweep"),
]

ASSUMPTIONS = [
    "Event-aware peer comparison: festival days are compared with peers' festival days, normal days with peers' "
    "normal days, and the larger robust z counts. Volume surge is divided by the peers' surge over the same days.",
    "Clustering and peer-distance tiers need 14 active days, 200 transactions and 30 payers; below that the merchant "
    "is monitored and can reach Tier 3 only through links, successor detection or national-list hits.",
    "Reserve: with a flagged payer cluster, only today's volume from that cluster is held; otherwise the surge above "
    "1.5× the pre-anomaly baseline is held.",
    "Successor detection counts only bettors who are new to a merchant after the suspension.",
    "The network-link signal depends on other merchants' tiers, so it is iterated until tiers stop changing.",
    "Card mix and payout fan-out are shown as evidence but do not change tiers.",
    "Shape distances are positive and right-skewed, so robust z-scores are taken on the log of each distance.",
    "The merchant-level score averages payer features with each payer counted once (payer-weighted).",
]


def render(r):
    page_header("Method", "How it works",
                "No single payment looks wrong when a bakery takes bets. The fraud shows only in the "
                "<b>shape of the money</b> over time and across networks of front merchants.")
    flow = "".join(f'<div style="padding:14px 12px;border-right:1px solid #E2E0D8">'
                   f'<div class="micro">{i + 1:02d}</div>'
                   f'<div class="serif" style="font-size:1.05rem;margin-top:6px;line-height:1.2">{t}</div></div>'
                   for i, (ic, t) in enumerate(PIPELINE))
    html(f'<div class="card" style="padding:0;display:grid;grid-template-columns:repeat({len(PIPELINE)},minmax(0,1fr))">'
         f'{flow}</div>')

    section("Six parts", "each one visible in the prototype")
    for row in range(2):
        cols = st.columns(3, gap="medium")
        for col, (ic, title, body) in zip(cols, PARTS[row * 3:(row + 1) * 3]):
            with col:
                html(f'<div class="card" style="height:100%"><div class="micro">Part {PARTS.index((ic, title, body)) + 1:02d}</div>'
                     f'<div class="serif" style="font-size:1.4rem;margin:8px 0 8px;line-height:1.2">{title}</div>'
                     f'<div style="color:#45443F;font-size:.9rem;line-height:1.55">{escape(body)}</div></div>')
        st.write("")

    section("Tiers", "evaluated in order 4 → 3 → 2 → 1")
    rows = "".join(f'<tr><td style="white-space:nowrap">{tier_badge(t)}</td><td>{escape(w)}</td><td>{escape(a)}</td></tr>'
                   for t, w, a in TIERS)
    html(f'<div class="card" style="padding:6px 14px"><table class="tbl"><tr><th>Tier</th><th>When</th>'
         f'<th>Response</th></tr>{rows}</table></div>')

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        section("Data availability")
        html('<div class="card" style="font-size:.9rem;line-height:1.6">Device ID and referrer domain exist only '
             'for hosted checkout and some card payments. UPI QR and UPI intent carry neither. For UPI, payer region '
             'comes from the bank code; for hosted checkout, from the IP address. The engine never relies on fields '
             f'the real world would not provide.<div style="color:{MUTED};margin-top:10px">{C.N_DAYS} simulated days '
             f'from {C.DAY1:%d %b %Y} (IST) · T20 match every {C.MATCH_EVERY}rd day at 19:30 · festival days '
             f'{C.FESTIVAL_DAYS[0]}–{C.FESTIVAL_DAYS[1]} · seed {C.SEED}. Every name, IP (RFC 5737) and domain '
             f'(.example) is fictional.</div></div>')
    with c2:
        section("Modelling assumptions")
        html('<div class="card" style="font-size:.88rem;line-height:1.5"><ul style="margin:0;padding-left:18px">'
             + "".join(f"<li style='margin:4px 0'>{escape(a)}</li>" for a in ASSUMPTIONS) + "</ul></div>")

    section("Demo script", "the guided demo walks through these steps; click one to jump there")
    for i, s in enumerate(STEPS):
        c1, c2 = st.columns([0.14, 1], vertical_alignment="center")
        if c1.button(f"{i + 1}", key=f"jump{i}", width="stretch"):
            apply_step(i)
        c2.markdown(f"**{s['title']}** · <span style='color:{MUTED}'>{escape(s['note'].replace('*', ''))}</span>",
                    unsafe_allow_html=True)
