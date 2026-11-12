"""Betting-Shaped Money: merchant monitor prototype. Run: streamlit run app.py"""
import streamlit as st

from engine import config as C
from views import about, merchant, network, overview
from views.common import (CSS, PAGE_ICONS, PAGES, STEPS, apply_step, confirm_p6, day_date, html, init_state,
                          reset_demo, result, stage, suspend_p6)

st.set_page_config(page_title="Merchant Monitor", page_icon="◼", layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)
init_state()
ss = st.session_state


def stop_guide():
    ss.demo_step = None


with st.sidebar:
    html('<div class="brand"><div class="t1">Merchant Monitor</div>'
         '<div class="t2">Betting-shaped money · prototype</div></div>')
    st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed",
             format_func=lambda p: f":material/{PAGE_ICONS[p]}: {p}")

    day = ss.current_day
    nxt = "T20 match today" if C.is_match_day(day) else f"Next T20 match day {day + (-day % 3)}"
    stages = ["Monitoring", "Royal Cakes confirmed", "Suspended · network swept"]
    st_html = "".join(f'<div class="stage {"done" if i < stage() else "now" if i == stage() else ""}">'
                      f'<div class="dot"></div><span>{s}</span></div>' for i, s in enumerate(stages))
    html(f'<div class="side-card"><div class="k">Simulated day</div><div class="v">Day {day}</div>'
         f'<div class="s">{day_date(day)} · {nxt}</div><div style="height:12px"></div>{st_html}</div>')

    html('<div class="side-card"><div class="k">Guided demo</div></div>')
    if ss.demo_step is None:
        if st.button("Start guided demo", icon=":material/play_arrow:", type="primary", width="stretch"):
            apply_step(0)
    else:
        s = STEPS[ss.demo_step]
        html(f'<div style="font-family:Newsreader,Georgia,serif;font-size:1.15rem;color:#F4F3EE;line-height:1.25;'
             f'margin:-2px 0 10px"><span style="font-family:Geist Mono,monospace;font-size:.7rem;color:#8A887F">'
             f'{ss.demo_step + 1:02d} / {len(STEPS):02d}</span><br>{s["title"]}</div>')
        c1, c2 = st.columns(2)
        if c1.button("Back", icon=":material/arrow_back:", disabled=ss.demo_step == 0, width="stretch"):
            apply_step(ss.demo_step - 1)
        if c2.button("Next", icon=":material/arrow_forward:", type="primary",
                     disabled=ss.demo_step == len(STEPS) - 1, width="stretch"):
            apply_step(ss.demo_step + 1)
        st.toggle("Scene captions", key="captions")
        st.button("Exit guided demo", on_click=stop_guide, width="stretch")

    with st.expander("Manual controls", expanded=False):
        st.button("Confirm Royal Cakes as front", on_click=confirm_p6, disabled="M-P6" in ss.confirmed,
                  width="stretch")
        st.button("Suspend it, advance to day 60", on_click=suspend_p6, disabled=bool(ss.suspended), width="stretch")
        st.button("Reset demo", on_click=reset_demo, width="stretch")

r = result()
{"Portfolio": overview.render, "Merchant": merchant.render, "Network": network.render,
 "How it works": about.render}[ss.nav](r)
