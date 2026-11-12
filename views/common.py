"""Shared UI: design system, cached engine runs, navigation, demo state and reusable components.

Visual language: editorial and restrained. Warm paper, near-black ink, serif headings and figures,
hairline borders, mono micro-labels, and muted tier colours used sparingly.
"""
from __future__ import annotations

from datetime import timedelta
from html import escape

import plotly.graph_objects as go
import streamlit as st

from engine import config as C
from engine.generate import load_or_generate
from engine.network import layout
from engine import timeline as TL
from engine.pipeline import run

PAGES = ["Portfolio", "Merchant", "Network", "How it works"]
PAGE_ICONS = {"Portfolio": "space_dashboard", "Merchant": "storefront", "Network": "hub", "How it works": "article"}

# design tokens
INK, INK2, MUTED, FAINT = "#1A1A18", "#45443F", "#75736B", "#A3A198"
PAPER, SURFACE, SUBTLE = "#F4F3EE", "#FFFFFF", "#FAF9F6"
BORDER, BORDER2 = "#E2E0D8", "#CFCCC2"
RUST = C.TIER_COLOR[3]
SANS = "Geist, ui-sans-serif, -apple-system, Segoe UI, sans-serif"
SERIF = "Newsreader, Georgia, 'Times New Roman', serif"
MONO = "'Geist Mono', ui-monospace, SFMono-Regular, Menlo, monospace"
ACCENT = INK

REASON_LABEL = {"cluster": "Payer cluster", "shape": "Money shape", "channel": "Channel", "domain": "Referrer",
                "matchday": "Match day", "national": "National list", "network": "Network", "payout": "Payouts",
                "successor": "Successor", "surge": "Volume", "recat": "Category", "clear": "Assessment",
                "festival": "Festival"}

CSS = f"""
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@300..700&family=Geist+Mono:wght@400..600&family=Newsreader:opsz,wght@6..72,400..700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,300,0,0" rel="stylesheet">
<style>
  .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}
  [data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stAppDeployButton"],
  [data-testid="stDecoration"], footer {{ display: none !important; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  .ms {{ font-family: 'Material Symbols Outlined'; font-weight: 300; font-style: normal; line-height: 1;
         display: inline-block; vertical-align: -3px; font-size: 18px; -webkit-font-smoothing: antialiased; }}
  .mono {{ font-family: {MONO}; }}
  .serif {{ font-family: {SERIF}; }}
  .micro {{ font-family: {MONO}; font-size: 0.68rem; letter-spacing: .08em; text-transform: uppercase; color: {MUTED}; }}
  /* sidebar */
  [data-testid="stSidebar"] {{ border-right: 1px solid #2C2C29; }}
  [data-testid="stSidebar"] [role="radiogroup"] {{ gap: 0; width: 100%; }}
  [data-testid="stSidebar"] [data-testid="stRadioOption"] {{ padding: 8px 12px; border-radius: 4px; width: 100%;
      margin: 0; border-left: 2px solid transparent; }}
  [data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {{ background: #1C1C1A; }}
  [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {{ background: #1E1E1C;
      border-left-color: #E8E6DF; }}
  [data-testid="stSidebar"] [data-testid="stRadioOption"] p {{ color: #9B998F; font-size: .92rem; }}
  [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p {{ color: #F4F3EE; }}
  [data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child {{ display: none; }}
  [data-testid="stSidebar"] button[kind="primary"] {{ background: #E8E6DF !important; border-color: #E8E6DF !important; }}
  [data-testid="stSidebar"] button[kind="primary"] * {{ color: #121211 !important; }}
  [data-testid="stSidebar"] button[kind="primary"]:disabled {{ background: #1E1E1C !important; border-color: #2C2C29 !important; }}
  [data-testid="stSidebar"] button[kind="primary"]:disabled * {{ color: #55534D !important; }}
  .brand {{ margin: 2px 0 22px; }}
  .brand .t1 {{ font-family: {SERIF}; font-size: 1.45rem; color: #F4F3EE; letter-spacing: -0.01em; line-height: 1.1; }}
  .brand .t2 {{ font-family: {MONO}; font-size: .64rem; letter-spacing: .1em; text-transform: uppercase;
      color: #8A887F; margin-top: 6px; }}
  .side-card {{ border-top: 1px solid #2C2C29; padding: 14px 2px 6px; margin: 10px 0 4px; }}
  .side-card .k {{ font-family: {MONO}; color: #8A887F; font-size: .64rem; text-transform: uppercase;
      letter-spacing: .1em; }}
  .side-card .v {{ font-family: {SERIF}; color: #F4F3EE; font-size: 1.6rem; line-height: 1.2; }}
  .side-card .s {{ color: #8A887F; font-size: .8rem; }}
  .stage {{ display: flex; align-items: center; gap: 9px; font-size: .82rem; margin: 6px 0; color: #8A887F; }}
  .stage .dot {{ width: 7px; height: 7px; border-radius: 50%; border: 1px solid #55534D; flex: none; }}
  .stage.done .dot {{ background: #8A887F; border-color: #8A887F; }}
  .stage.now .dot {{ background: #F4F3EE; border-color: #F4F3EE; }}
  .stage.now span {{ color: #F4F3EE; }}
  /* page header */
  .ph {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 24px; padding-bottom: 18px;
      margin-bottom: 22px; border-bottom: 1px solid {BORDER}; }}
  .ph .eyebrow {{ font-family: {MONO}; color: {MUTED}; font-size: .7rem; letter-spacing: .12em; text-transform: uppercase; }}
  .ph h1 {{ font-family: {SERIF}; font-size: 2.5rem; font-weight: 400; letter-spacing: -0.02em; margin: 6px 0 6px;
      padding: 0; color: {INK}; line-height: 1.08; }}
  .ph .sub {{ color: {INK2}; font-size: 1rem; max-width: 760px; line-height: 1.5; }}
  .pills {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
  .pill {{ font-family: {MONO}; font-size: .7rem; letter-spacing: .05em; text-transform: uppercase; color: {INK2};
      border: 1px solid {BORDER2}; padding: 5px 10px; border-radius: 3px; white-space: nowrap; background: {SURFACE}; }}
  .pill.proto {{ border-color: {INK}; color: {INK}; }}
  /* cards and sections */
  .card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px; padding: 20px 22px; }}
  [data-testid="stVerticalBlockBorderWrapper"] {{ background: {SURFACE}; border-color: {BORDER} !important;
      border-radius: 6px !important; }}
  .sec {{ display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin: 34px 0 12px;
      padding-bottom: 8px; border-bottom: 1px solid {BORDER}; }}
  .sec .h {{ font-family: {SERIF}; font-size: 1.45rem; color: {INK}; letter-spacing: -0.01em; }}
  .sec span {{ font-family: {MONO}; font-size: .68rem; letter-spacing: .08em; text-transform: uppercase; color: {MUTED}; }}
  .kpis {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)) minmax(0, 1.45fr) minmax(0, 1fr); border: 1px solid {BORDER};
      border-radius: 6px; background: {SURFACE}; }}
  .kpi {{ padding: 16px 18px; border-right: 1px solid {BORDER}; }}
  .kpi:last-child {{ border-right: none; }}
  .kpi .l {{ font-family: {MONO}; color: {MUTED}; font-size: .66rem; letter-spacing: .09em; text-transform: uppercase;
      display: flex; align-items: center; gap: 7px; }}
  .kpi .l i {{ width: 7px; height: 7px; display: inline-block; background: var(--c); border-radius: 1px; }}
  .kpi .v {{ font-family: {SERIF}; font-size: 2.1rem; color: {INK}; margin-top: 6px; line-height: 1.1; white-space: nowrap; }}
  .kpi .s {{ color: {MUTED}; font-size: .78rem; margin-top: 2px; }}
  .dist {{ display: flex; height: 6px; overflow: hidden; margin: 14px 0 8px; background: {BORDER}; gap: 2px; }}
  .dist-legend {{ display: flex; gap: 22px; font-family: {MONO}; font-size: .68rem; letter-spacing: .05em;
      text-transform: uppercase; color: {MUTED}; }}
  .dist-legend i {{ display: inline-block; width: 7px; height: 7px; margin-right: 6px; }}
  .tb {{ display: inline-flex; align-items: center; gap: 6px; font-family: {MONO}; font-size: .68rem;
      letter-spacing: .06em; text-transform: uppercase; padding: 3px 8px; border-radius: 3px; white-space: nowrap;
      color: var(--c); background: var(--t); border: 1px solid color-mix(in srgb, var(--c) 30%, transparent); }}
  .tb i {{ width: 6px; height: 6px; background: var(--c); display: inline-block; border-radius: 1px; }}
  .tb.lg {{ font-size: .74rem; padding: 5px 10px; }}
  .pk {{ font-family: {MONO}; font-size: .68rem; color: {MUTED}; letter-spacing: .08em; }}
  .pc-name {{ font-family: {SERIF}; font-size: 1.3rem; color: {INK}; margin: 12px 0 2px; line-height: 1.2; }}
  .pc-meta {{ color: {MUTED}; font-size: .82rem; }}
  .pc-head {{ font-size: .9rem; color: {INK2}; margin: 12px 0 14px; min-height: 2.7em; line-height: 1.45;
      border-left: 2px solid var(--c); padding-left: 10px; }}
  .pc-stats {{ display: flex; gap: 22px; border-top: 1px solid {BORDER}; padding-top: 12px; }}
  .pc-stats div {{ font-family: {MONO}; font-size: .62rem; color: {MUTED}; text-transform: uppercase; letter-spacing: .08em; }}
  .pc-stats b {{ display: block; font-family: {SERIF}; font-weight: 400; font-size: 1.25rem; color: {INK};
      letter-spacing: 0; text-transform: none; margin-top: 2px; }}
  .chip {{ display: inline-flex; align-items: center; gap: 6px; color: {INK2}; font-size: .84rem; padding: 0 16px 0 0;
      margin: 0 16px 6px 0; border-right: 1px solid {BORDER}; }}
  .chip:last-child {{ border-right: none; }}
  .chip .ms {{ color: {FAINT}; }}
  /* reasons, meters, actions */
  .rs {{ display: grid; grid-template-columns: 34px 1fr; gap: 6px 10px; padding: 13px 0; border-top: 1px solid {BORDER}; }}
  .rs .n {{ font-family: {MONO}; font-size: .72rem; color: {FAINT}; padding-top: 3px; }}
  .rs .lab {{ font-family: {MONO}; font-size: .64rem; letter-spacing: .09em; text-transform: uppercase; color: var(--c); }}
  .rs .tx {{ font-size: .95rem; color: {INK}; line-height: 1.5; margin-top: 3px; }}
  .meter {{ margin: 4px 0 18px; }}
  .meter .top {{ display: flex; justify-content: space-between; align-items: baseline; }}
  .meter .top span {{ font-family: {MONO}; font-size: .66rem; letter-spacing: .08em; text-transform: uppercase; color: {MUTED}; }}
  .meter .top b {{ font-family: {SERIF}; font-weight: 400; color: {INK}; font-size: 1.5rem; }}
  .meter .bar {{ position: relative; height: 4px; background: {BORDER}; margin-top: 6px; }}
  .meter .fill {{ position: absolute; left: 0; top: 0; bottom: 0; }}
  .meter .tick {{ position: absolute; top: -5px; bottom: -5px; width: 1px; background: {INK}; }}
  .meter .peer {{ position: absolute; top: -4px; width: 8px; height: 12px; margin-left: -4px; background: {SURFACE};
      border: 1px solid {MUTED}; }}
  .meter .cap {{ font-size: .74rem; color: {MUTED}; margin-top: 8px; }}
  .act {{ border-top: 1px solid {BORDER}; border-bottom: 1px solid {BORDER}; padding: 14px 0; margin: 4px 0 6px; }}
  .act .h {{ font-family: {SERIF}; font-size: 1.15rem; color: {INK}; display: flex; gap: 8px; align-items: center; }}
  .act .h i {{ width: 8px; height: 8px; background: var(--c); display: inline-block; }}
  .act .b {{ font-size: .88rem; color: {INK2}; margin-top: 6px; line-height: 1.5; }}
  .sig {{ display: flex; align-items: center; justify-content: space-between; padding: 7px 0; font-size: .88rem;
      color: {INK}; border-bottom: 1px dotted {BORDER2}; }}
  .sig .v {{ font-family: {MONO}; font-size: .66rem; letter-spacing: .08em; text-transform: uppercase; }}
  .sig.on .v {{ color: {RUST}; }} .sig.off {{ color: {MUTED}; }} .sig.off .v {{ color: {FAINT}; }}
  .notice {{ border-left: 2px solid {INK}; padding: 10px 14px; background: {SUBTLE}; color: {INK2}; font-size: .9rem;
      line-height: 1.5; }}
  .notice b {{ font-family: {MONO}; font-size: .64rem; letter-spacing: .1em; text-transform: uppercase; color: {MUTED};
      display: block; margin-bottom: 4px; font-weight: 500; }}
  .caption-bar {{ display: grid; grid-template-columns: 70px 1fr; gap: 16px; align-items: start; background: {INK};
      color: {PAPER}; border-radius: 6px; padding: 18px 22px; margin: 0 0 22px; }}
  .caption-bar .n {{ font-family: {MONO}; font-size: .74rem; color: #A3A198; padding-top: 6px; letter-spacing: .06em; }}
  .caption-bar .t {{ font-family: {SERIF}; font-size: 1.4rem; color: {PAPER}; line-height: 1.2; }}
  .caption-bar .d {{ color: #CFCCC2; font-size: .95rem; margin-top: 6px; line-height: 1.5; }}
  .flash {{ display: grid; grid-template-columns: 110px 1fr; gap: 14px; align-items: baseline; padding: 14px 18px;
      margin: 0 0 18px; border: 1px solid {BORDER}; border-left: 3px solid var(--c); background: {SURFACE};
      font-size: .95rem; color: {INK}; border-radius: 0 6px 6px 0; }}
  .flash .lab {{ font-family: {MONO}; font-size: .66rem; letter-spacing: .1em; text-transform: uppercase; color: var(--c); }}
  .stat {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px; padding: 14px 18px; height: 100%; }}
  .stat .v {{ font-family: {SERIF}; font-size: 1.9rem; color: {INK}; line-height: 1.1; }}
  .stat .l {{ font-family: {MONO}; font-size: .64rem; letter-spacing: .08em; text-transform: uppercase; color: {MUTED};
      margin-top: 6px; }}
  .tbl {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
  .tbl th {{ font-family: {MONO}; font-size: .64rem; letter-spacing: .09em; text-transform: uppercase; color: {MUTED};
      text-align: left; font-weight: 500; padding: 10px 12px; border-bottom: 1px solid {BORDER2}; }}
  .tbl td {{ padding: 12px; border-bottom: 1px solid {BORDER}; vertical-align: top; color: {INK}; }}
  /* streamlit widgets */
  .stTabs [data-baseweb="tab-list"] {{ gap: 18px; border-bottom: 1px solid {BORDER}; }}
  .stTabs [data-baseweb="tab"] {{ padding: 10px 0; }}
  .stTabs [data-baseweb="tab"] p {{ font-size: .9rem; }}
  .stTabs [data-baseweb="tab-highlight"] {{ background-color: {INK}; height: 2px; }}
  [data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 6px; overflow: hidden; background: {SURFACE}; }}
</style>
"""
CSS = "\n".join(line for line in CSS.splitlines() if line.strip())  # blank lines would end the HTML block


# ============================================================ state
def init_state():
    ss = st.session_state
    defaults = dict(current_day=C.DEFAULT_DAY, confirmed=[], suspended={}, released=[], cleared=[], cases={},
                    selected="M-P1",
                    nav=PAGES[0], table_ver=0, mtab=None, mtab_ver=0, net_focus="M-P6", measure="₹ volume",
                    demo_step=None, captions=True)
    for k, v in defaults.items():
        ss.setdefault(k, v)
    if "_qp_done" not in ss:  # deep links for recording, e.g. ?page=Merchant&m=P5&state=suspend
        ss._qp_done = True
        qp = st.query_params
        if qp.get("page") in PAGES:
            ss["nav"] = qp["page"]
        if qp.get("m"):
            ss.selected = ss.net_focus = qp["m"] if qp["m"].startswith("M-") else f"M-{qp['m']}"
        if qp.get("tab"):
            ss.mtab, ss.mtab_ver = qp["tab"], ss.mtab_ver + 1
        if qp.get("state") == "confirm":
            confirm_p6()
        elif qp.get("state") == "suspend":
            suspend_p6()
        if qp.get("step", "").isdigit():
            apply_step(int(qp["step"]) - 1, rerun=False)
    if "_goto" in ss:
        ss["nav"] = ss.pop("_goto")


@st.cache_resource(show_spinner=False)
def _data():
    return load_or_generate(C.SEED)


@st.cache_resource(show_spinner="Scoring merchants…", max_entries=12)
def _run(day: int, confirmed: tuple, suspended: tuple, cleared: tuple = ()):
    r = run(day, confirmed=set(confirmed), suspended=dict(suspended), data=_data(), cleared=set(cleared))
    r.extra["pos"] = layout(r.merchants, r.edges)
    return r


def result():
    ss = st.session_state
    return _run(ss.current_day, tuple(sorted(ss.confirmed)), tuple(sorted(ss.suspended.items())),
                tuple(sorted(ss.cleared)))


# ------------------------------------------------------------ day-by-day replay (time to detection)
@st.cache_resource(show_spinner=False)
def _timeline_job():
    progress = {"done": 0, "total": TL.LAST_DAY - TL.FIRST_DAY + 1}
    _data()  # make sure the data exists before the thread reads it
    thread = TL.start_background(progress)
    return {"progress": progress, "thread": thread}


def timeline():
    """Replay frame, or None while it is still being computed (progress in timeline_progress())."""
    tl = TL.load()
    if tl is None:
        _timeline_job()
    return tl


def timeline_progress() -> tuple[int, int]:
    p = _timeline_job()["progress"]
    return p["done"], p["total"]


def detection(r, mid):
    tl = timeline()
    return None if tl is None else TL.detection(tl, r.data, mid, r.current_day)


# ------------------------------------------------------------ analyst case actions
def log_case(mid: str, action: str, note: str = ""):
    ss = st.session_state
    cases = dict(ss.cases)
    cases[mid] = cases.get(mid, []) + [dict(day=ss.current_day, action=action, note=note)]
    ss.cases = cases


def confirm_front(mid: str):
    ss = st.session_state
    if mid not in ss.confirmed:
        ss.confirmed = ss.confirmed + [mid]
    ss.cleared = [c for c in ss.cleared if c != mid]
    log_case(mid, "Confirmed as front", "Acceptance suspended; STR drafted; I4C referral (simulated)")


def clear_merchant(mid: str):
    ss = st.session_state
    if mid not in ss.cleared:
        ss.cleared = ss.cleared + [mid]
    log_case(mid, "Cleared as false positive", "Merchant returned to Tier 1; model keeps scoring daily")


def reopen_merchant(mid: str):
    ss = st.session_state
    ss.cleared = [c for c in ss.cleared if c != mid]
    log_case(mid, "Case reopened")


def request_docs(mid: str):
    log_case(mid, "Documents requested", "Invoices, delivery records and business-premises photo")


def goto(page: str, merchant: str | None = None, tab: str | None = None):
    ss = st.session_state
    if merchant:
        ss.selected = merchant
    if tab:
        ss.mtab, ss.mtab_ver = tab, ss.mtab_ver + 1
    ss["_goto"] = page
    ss.table_ver += 1
    st.rerun()


# ============================================================ demo actions
def confirm_p6():
    ss = st.session_state
    if "M-P6" not in ss.confirmed:
        ss.confirmed = ss.confirmed + ["M-P6"]
        log_case("M-P6", "Confirmed as front", "Network sweep run")


def suspend_p6():
    ss = st.session_state
    confirm_p6()
    if not ss.suspended:
        log_case("M-P6", "Suspended", "Acceptance stopped on day 55; simulation advanced to day 60")
    ss.suspended = {"M-P6": C.DEFAULT_DAY}
    ss.current_day = C.SUSPEND_DAY


def reset_demo():
    ss = st.session_state
    ss.current_day, ss.confirmed, ss.suspended, ss.released = C.DEFAULT_DAY, [], {}, []
    ss.cleared, ss.cases = [], {}
    ss.table_ver += 1


def stage() -> int:
    ss = st.session_state
    return 2 if ss.suspended else 1 if ss.confirmed else 0


# ============================================================ guided demo
STEPS = [
    dict(title="Every merchant, scored daily", page="Portfolio", state=0,
         note="Ninety merchants and 55 days of synthetic payments. Six need attention today, worst first, each with "
              "the day it was flagged. Genuine merchants stay green."),
    dict(title="A festival spike that stays green", page="Merchant", m="M-P2", tab="Timeline", state=0,
         measure="Transactions",
         note="Diwali Delights quadruples its orders in the festival week. Peer bakeries rise on the same days, its "
              "ticket mix matches theirs, and nothing about the money is betting-shaped."),
    dict(title="Viral is unusual, not betting", page="Merchant", m="M-P3", tab="Payer clusters", state=0,
         note="Crumb & Co goes viral across India. Volume jumps about 10× while peers stay flat, yet no payer group "
              "looks like betting, so it only gets a polite nudge."),
    dict(title="Round amounts that match the declaration", page="Merchant", m="M-P4", tab="Money shape", state=0,
         note="QuickTopUp's tickets are all round, like betting stakes, but it is declared as a top-up merchant, so "
              "there is no declaration gap and it stays green."),
    dict(title="Betting hidden inside a real bakery", page="Merchant", m="M-P5", tab="Payer clusters", state=0,
         note="Gupta Sweets looks normal overall. Splitting its payers exposes a small group whose stakes double "
              "through each match evening. It was flagged five days after that group first appeared."),
    dict(title="A proportionate response", page="Merchant", m="M-P5", tab="Settlement and reserve", state=0,
         note="The bakery's own customers settle as normal. Only today's volume from the flagged payer group is held, "
              "for up to 48 hours, and released early on proof of fulfilment."),
    dict(title="The operator behind the fronts", page="Network", m="M-P6", state=0,
         note="Royal Cakes, Shree Ganesh and TechNova share one dashboard device, a settlement account and hundreds "
              "of bettors. Gupta Sweets connects through the same offshore API server."),
    dict(title="Confirm one front, sweep the network", page="Network", m="M-P6", state=1,
         note="An analyst confirms Royal Cakes. Its linked merchants go to review. A link alone never confirms a "
              "front; it only triggers review."),
    dict(title="Suspend it and catch the successor", page="Network", m="M-P9", state=2,
         note="Royal Cakes is suspended on day 55. Within five days most of its bettors appear at Urban Crumbs, a "
              "quiet new bakery that is now flagged as the likely successor."),
]


def apply_step(i: int, rerun: bool = True):
    ss = st.session_state
    i = max(0, min(len(STEPS) - 1, i))
    s = STEPS[i]
    reset_demo()
    if s["state"] >= 1:
        confirm_p6()
    if s["state"] >= 2:
        suspend_p6()
    if s.get("m"):
        ss.selected = ss.net_focus = s["m"]
    if s.get("tab"):
        ss.mtab, ss.mtab_ver = s["tab"], ss.mtab_ver + 1
    ss.measure = s.get("measure", "₹ volume")
    ss.demo_step = i
    ss["_goto"] = s["page"]
    if rerun:
        st.rerun()


# ============================================================ formatting
def day_date(day: int) -> str:
    return (C.DAY1 + timedelta(days=day - 1)).strftime("%a %d %b %Y")


def inr(x: float) -> str:
    """Indian rupee formatting with lakh/crore shorthand."""
    if x >= 1e7:
        return f"₹{x / 1e7:.2f} Cr"
    if x >= 1e5:
        return f"₹{x / 1e5:.2f} L"
    return f"₹{x:,.0f}"


def ms(name: str, style: str = "") -> str:
    return f'<span class="ms" style="{style}">{name}</span>'


def tier_badge(tier: int, large=False, short=False) -> str:
    label = C.TIER_SHORT[tier] if short else C.TIER_LABEL[tier]
    return (f'<span class="tb{" lg" if large else ""}" style="--c:{C.TIER_COLOR[tier]};--t:{C.TIER_TINT[tier]}">'
            f'<i></i>T{tier} · {label}</span>')


def tier_text(tier: int) -> str:
    return f"T{tier} · {C.TIER_SHORT[tier]}"


def mask_payer(pid: str, method: str = "UPI") -> str:
    """Display-only masking, the way an operations console shows payment identifiers."""
    tail = pid[-4:]
    return f"Card ••••{tail}" if method == "CARD" else f"UPI ••••{tail}"


def merchant_options(r):
    m = r.merchants.copy()
    m["order"] = m["persona_key"].fillna("Z") + m["merchant_id"]
    m = m.sort_values("order")
    return list(m["merchant_id"]), dict(zip(m["merchant_id"], m["name"]))


# ============================================================ components
def html(s: str):
    st.markdown(s, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, sub: str = ""):
    ss = st.session_state
    match = " · T20 match day" if C.is_match_day(ss.current_day) else ""
    pills = (f'<span class="pill">Day {ss.current_day} · {day_date(ss.current_day)}{match}</span>'
             f'<span class="pill proto">Prototype · synthetic data</span>')
    html(f'<div class="ph"><div><div class="eyebrow">{escape(eyebrow)}</div><h1>{escape(title)}</h1>'
         f'<div class="sub">{sub}</div></div><div class="pills">{pills}</div></div>')
    scene_caption()


def scene_caption():
    ss = st.session_state
    if ss.demo_step is None or not ss.captions:
        return
    s = STEPS[ss.demo_step]
    if s["page"] != ss.nav:
        return
    html(f'<div class="caption-bar"><div class="n">{ss.demo_step + 1:02d} / {len(STEPS):02d}</div><div>'
         f'<div class="t">{escape(s["title"])}</div><div class="d">{escape(s["note"])}</div></div></div>')


def section(title: str, sub: str = ""):
    html(f'<div class="sec"><div class="h">{escape(title)}</div><span>{sub}</span></div>')


def flash(tier: int, label: str, text: str):
    html(f'<div class="flash" style="--c:{C.TIER_COLOR[tier]}"><div class="lab">{escape(label)}</div><div>{text}</div></div>')


def reason_rows(items: list[tuple[str, str]], color: str) -> str:
    return "".join(f'<div class="rs" style="--c:{color}"><div class="n">{i + 1:02d}</div><div>'
                   f'<div class="lab">{REASON_LABEL.get(k, k)}</div><div class="tx">{escape(t)}</div></div></div>'
                   for i, (k, t) in enumerate(items))


def meter(label: str, value: float, color: str, threshold: float | None = None, peer: float | None = None,
          caption: str = "") -> str:
    v = max(0.0, min(1.0, value))
    tick = f'<div class="tick" style="left:{threshold * 100:.1f}%"></div>' if threshold is not None else ""
    pm = f'<div class="peer" style="left:{peer * 100:.1f}%" title="peer median"></div>' if peer is not None else ""
    return (f'<div class="meter"><div class="top"><span>{label}</span><b>{value:.2f}</b></div>'
            f'<div class="bar"><div class="fill" style="width:{v * 100:.1f}%;background:{color}"></div>{tick}{pm}'
            f'</div><div class="cap">{caption}</div></div>')


# ============================================================ charts
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
SERIES = {"accent": INK, "peer": "#D6D3C9", "peer_line": "#8C8A81", "bet": RUST, "festival": "#EDE5CF"}


def style_fig(fig: go.Figure, title: str = "", height: int = 380, legend: bool = True, sub: str = "") -> go.Figure:
    t = (f"<span style='font-family:Newsreader, Georgia, serif;font-size:19px'>{title}</span>"
         + (f"<br><span style='font-size:12px;color:{MUTED}'>{sub}</span>" if sub else ""))
    fig.update_layout(
        template="plotly_white", height=height, font=dict(family=SANS, size=12, color=INK2),
        title=dict(text=t, x=0, xanchor="left", y=0.97, yanchor="top", font=dict(color=INK)),
        margin=dict(l=6, r=6, t=74 if sub else 54, b=6), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(size=11, color=MUTED),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font=dict(family=SANS), bgcolor=SURFACE, bordercolor=BORDER2),
    )
    fig.update_xaxes(gridcolor="#EFEDE6", zerolinecolor=BORDER, linecolor=BORDER2, tickfont=dict(family=MONO, size=10))
    fig.update_yaxes(gridcolor="#EFEDE6", zerolinecolor=BORDER, linecolor=BORDER2, tickfont=dict(family=MONO, size=10))
    return fig


def chart(fig, **kw):
    return st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, theme=None, **kw)
