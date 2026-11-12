"""Day-by-day replay: the tier every merchant would have had on each day, to measure time to detection.

Each replay day is a full engine run, so the result is computed once and cached to data/."""
from __future__ import annotations

import threading

import pandas as pd

from engine import config as C
from engine.generate import DATA_DIR, load_or_generate
from engine.pipeline import run

FIRST_DAY, LAST_DAY = 12, C.N_DAYS
PATH = DATA_DIR / f"timeline_seed{C.SEED}.parquet"


def compute(progress: dict | None = None) -> pd.DataFrame:
    """Base scenario (no analyst actions) replayed day by day. Day 56-60 rows assume the day-55 suspension,
    matching the demo's storyline."""
    data = load_or_generate(C.SEED)
    rows = []
    days = list(range(FIRST_DAY, LAST_DAY + 1))
    for i, d in enumerate(days):
        susp = {"P6": C.DEFAULT_DAY} if d > C.DEFAULT_DAY else {}
        conf = {"P6"} if d > C.DEFAULT_DAY else set()
        r = run(d, confirmed=conf, suspended=susp, data=data)
        rows += [dict(merchant_id=m, day=d, tier=int(t)) for m, t in r.tiers.items()]
        if progress is not None:
            progress["done"], progress["total"] = i + 1, len(days)
    df = pd.DataFrame(rows)
    df.to_parquet(PATH, index=False)
    return df


def load() -> pd.DataFrame | None:
    return pd.read_parquet(PATH) if PATH.exists() else None


def start_background(progress: dict) -> threading.Thread | None:
    """Compute the timeline in a daemon thread if it is not cached yet."""
    if PATH.exists():
        return None
    th = threading.Thread(target=compute, args=(progress,), daemon=True)
    th.start()
    return th


def onset_day(data, mid: str) -> int | None:
    """First day of suspicious activity: offshore / undeclared server-API traffic or a blocked referrer."""
    tx = data["transactions"]
    m = data["merchants"].set_index("merchant_id").loc[mid]
    t = tx[tx["merchant_id"] == mid]
    api = data["api_logs"]
    days = set(api.loc[(api["merchant_id"] == mid) & (api["server_country"] != "IN"), "day"])
    days |= set(t.loc[t["referrer_domain"].isin(C.BLOCKED_DOMAINS), "day"])
    if m["integration_type"] != "API":
        days |= set(t.loc[t["channel"] == "S2S_API", "day"])
    return int(min(days)) if days else None


def detection(tl: pd.DataFrame, data, mid: str, upto: int) -> dict:
    """First day the merchant reached Tier 3+ (up to `upto`), and lag from the onset of suspicious activity."""
    rows = tl[(tl["merchant_id"] == mid) & (tl["day"] <= upto)]
    hit = rows[rows["tier"] >= 3]
    first = int(hit["day"].min()) if len(hit) else None
    onset = onset_day(data, mid)
    lag = (first - onset) if (first is not None and onset is not None) else None
    return dict(first_flag=first, onset=onset, lag=lag, history=rows.sort_values("day")[["day", "tier"]])
