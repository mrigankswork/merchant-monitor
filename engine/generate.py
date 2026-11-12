"""Synthetic world generator. Deterministic for a given seed.

Produces: merchants, payers, transactions, api_logs, dashboard_logins, payouts, national_lists.
All names, IPs (RFC 5737) and domains (.example) are fictional.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from engine import config as C

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GIFT_BOXES = [1149, 1249, 1349, 1499, 1649, 1799, 1999, 2249, 2499]
NAME_STEMS = ["Saffron", "Banyan", "Peacock", "Lotus", "Monsoon", "Cardamom", "Neem", "Tulsi", "Marigold",
              "Jasmine", "Mango", "Coconut", "Kesar", "Amla", "Jamun", "Chikoo", "Sitaphal", "Gulmohar",
              "Ashoka", "Kadamb", "Champa", "Rangoli", "Diya", "Kolam", "Mehendi", "Chandan", "Pista",
              "Elaichi", "Sandal", "Moti", "Sona", "Chanda", "Tara", "Neelam", "Pukhraj", "Heera", "Panna",
              "Moonga", "Lajwanti", "Parijat", "Bakul", "Shefali", "Malati", "Kanak"]
NAME_SUFFIX = {"Bakery": ["Bakehouse", "Oven", "Bakers", "Patisserie"],
               "Wallet/recharge top-up": ["Recharge Point", "TopUp Kiosk", "Pay Point"],
               "General merchandise": ["Stores", "Mart", "General Store", "Traders"],
               "IT services": ["Infotech", "Software", "Systems", "Digital"]}


def _h(prefix: str, text: str) -> str:
    return f"{prefix}-{hashlib.sha1(text.encode()).hexdigest()[:10]}"


class _World:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.payers: list[tuple] = []  # (pid, home_region, bank_code, has_upi, has_card)
        self.chunks: list[dict] = []
        self.merchants: list[dict] = []

    def new_payers(self, n: int, regions) -> np.ndarray:
        rng = self.rng
        start = len(self.payers)
        regs = [regions] * n if isinstance(regions, str) else list(rng.choice(regions, size=n))
        for i, r in enumerate(regs):
            bank = f"BK{C.REGIONS.index(r):02d}{int(rng.integers(1, 4))}"
            self.payers.append((start + i, r, bank, rng.random() < 0.95, rng.random() < 0.35))
        return np.arange(start, start + n)

    def mixed_payers(self, n: int, home: str, local_share: float) -> np.ndarray:
        n_local = int(round(n * local_share))
        others = [r for r in C.REGIONS if r != home]
        return np.concatenate([self.new_payers(n_local, home), self.new_payers(n - n_local, others)])

    def add(self, mid, payer, day, sec, amount, method, channel, referrer=None,
            card_type=None, card_country=None, device=None, src="gen"):
        n = len(payer)
        if n == 0:
            return

        def arr(x):
            if x is None or np.isscalar(x):
                return np.full(n, x, dtype=object)
            return np.asarray(x, dtype=object)
        self.chunks.append(dict(
            merchant_id=np.full(n, mid, dtype=object), payer=np.asarray(payer, dtype=np.int64),
            day=np.asarray(day, dtype=np.int64), sec=np.asarray(sec, dtype=np.float64),
            amount_inr=np.asarray(amount, dtype=np.float64), method=arr(method), channel=arr(channel),
            referrer_domain=arr(referrer), card_type=arr(card_type), card_issuer_country=arr(card_country),
            device_id=arr(device), src=np.full(n, src, dtype=object)))


def _dev(payers):
    return np.array([f"DV{p}" for p in payers], dtype=object)


# ------------------------------------------------------------------ samplers
def _bakery_hours(rng, n, shift=0.0, w_morning=0.42):
    comp = rng.random(n)
    h = np.where(comp < w_morning, rng.normal(8.4 + shift, 0.85, n),
                 np.where(comp < 0.94, rng.normal(19.0 + shift, 1.15, n), rng.uniform(10.5, 16.5, n)))
    return np.clip(h, 6.2, 22.9) * 3600


def _bakery_amounts(rng, n, median=220.0, sigma=0.5):
    a = np.exp(rng.normal(np.log(median), sigma, n))
    return np.maximum(20, np.round(a / 5) * 5)


def _nonround(a):
    a = np.round(a)
    return np.where(a % 100 == 0, a + 9, a)


def _sample(rng, pool, weights, n):
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    return rng.choice(pool, size=n, replace=n > len(pool), p=weights)


def _counter_or_web(w: _World, mid, payers, day, sec, amt, domain, web_share):
    """Counter QR merchants with an optional hosted website checkout."""
    rng = w.rng
    n = len(payers)
    web = (rng.random(n) < web_share) & (domain is not None)
    card = web & (rng.random(n) < 0.2)
    hosted = web & ~card
    qr = ~web
    w.add(mid, payers[qr], day[qr], sec[qr], amt[qr], "UPI_QR", "COUNTER_QR")
    w.add(mid, payers[hosted], day[hosted], sec[hosted], amt[hosted], "HOSTED_CHECKOUT", "WEBSITE",
          referrer=domain, device=_dev(payers[hosted]))
    w.add(mid, payers[card], day[card], sec[card], amt[card], "CARD", "WEBSITE", card_type="DEBIT",
          card_country="IN", device=_dev(payers[card]))


# ------------------------------------------------------------------ genuine generators
def gen_bakery(w: _World, m: dict, city_pool, days, festival_mult=1.0, gift_share=None, params=None,
               pool_size=None):
    """Genuine bakery. In the festival window volume rises and gift boxes appear (15-30% of txns)."""
    rng = w.rng
    p = params or {}
    mean = p.get("mean", rng.uniform(40, 80))
    median = p.get("median", rng.uniform(185, 260))
    shift = p.get("shift", rng.normal(0, 0.35))
    w_morn = p.get("w_morning", rng.uniform(0.34, 0.5))
    web_share = p.get("web_share", rng.uniform(0.06, 0.14))
    gift_share = rng.uniform(0.15, 0.30) if gift_share is None else gift_share
    size = pool_size or int(rng.integers(600, 1001))
    n_shared = int(size * 0.08)
    pool = np.concatenate([w.new_payers(size - n_shared, m["region"]),
                           rng.choice(city_pool, size=n_shared, replace=False)])
    wts = rng.uniform(0.10, 0.25, len(pool))
    wts /= wts.sum()
    for d in days:
        fest = C.is_festival(d)
        n = rng.poisson(mean * (festival_mult if fest else 1.0))
        payers = _sample(rng, pool, wts, n)
        amt = _bakery_amounts(rng, n, median)
        if fest:
            gift = rng.random(n) < gift_share
            amt = np.where(gift, rng.choice(GIFT_BOXES, size=n), amt)
        _counter_or_web(w, m["merchant_id"], payers, np.full(n, d), _bakery_hours(rng, n, shift, w_morn), amt,
                        m["registered_domain"], web_share)
    return mean


def gen_recharge(w: _World, m: dict, days, params=None):
    rng = w.rng
    p = params or {}
    mean = p.get("mean", rng.uniform(150, 300))
    probs = np.array([0.25, 0.25, 0.3, 0.15, 0.05])
    if params is None:
        probs = probs * rng.uniform(0.85, 1.15, 5)
    probs /= probs.sum()
    rates = rng.uniform(2, 6, int(mean * 30 / 4))
    pool = w.mixed_payers(len(rates), m["region"], 0.6)
    wts = rates / rates.sum()
    mid = m["merchant_id"]
    for d in days:
        n = rng.poisson(mean)
        payers = _sample(rng, pool, wts, n)
        sec = rng.uniform(8, 23, n) * 3600
        amt = rng.choice([100, 200, 500, 1000, 2000], size=n, p=probs).astype(float)
        r = rng.random(n)
        link, hosted, card = r < 0.5, (r >= 0.5) & (r < 0.9), r >= 0.9
        dd = np.full(n, d)
        w.add(mid, payers[link], dd[link], sec[link], amt[link], "UPI_INTENT", "PAYMENT_LINK")
        w.add(mid, payers[hosted], dd[hosted], sec[hosted], amt[hosted], "HOSTED_CHECKOUT", "WEBSITE",
              referrer=m["registered_domain"], device=_dev(payers[hosted]))
        w.add(mid, payers[card], dd[card], sec[card], amt[card], "CARD", "WEBSITE",
              card_type=rng.choice(["DEBIT", "CREDIT"], size=card.sum(), p=[0.7, 0.3]).astype(object),
              card_country="IN", device=_dev(payers[card]))


def gen_general(w: _World, m: dict, days, festival_mult=1.0, n_mean=None):
    """30-70 txns/day; lognormal median Rs 650, sigma 0.7, rounded to Rs 1; 10:00-21:00;
    70% local / 30% regional payers, each active on 2-8% of days; 70% counter QR, 30% website."""
    rng = w.rng
    mean = n_mean or rng.uniform(30, 70)
    rates = rng.uniform(0.02, 0.08, 4000)
    size = int(np.clip(mean / rates.mean(), 300, 4000))
    rates = rates[:size]
    pool = w.mixed_payers(size, m["region"], 0.7)
    wts = rates / rates.sum()
    for d in days:
        n = rng.poisson(mean * (festival_mult if C.is_festival(d) else 1.0))
        payers = _sample(rng, pool, wts, n)
        h = rng.uniform(10, 21, n) * 3600
        amt = np.maximum(20, np.round(np.exp(rng.normal(np.log(650), 0.7, n))))
        _counter_or_web(w, m["merchant_id"], payers, np.full(n, d), h, amt, m["registered_domain"] or
                        (f"{m['merchant_id'].lower()}.example"), 0.30)


def gen_it(w: _World, m: dict, days, n_mean=None, per_week=None):
    """Weekdays 10:00-19:00; invoices lognormal median Rs 25,000 (5k-2L), often round;
    repeat business clients paying 1-3 times a month; payment links and server API from own Indian server."""
    rng = w.rng
    mean = n_mean or rng.uniform(2, 10)
    size = max(12, int(mean * 21 / 2))
    pool = w.mixed_payers(size, m["region"], 0.5)
    rates = rng.uniform(1, 3, size)
    wts = rates / rates.sum()
    mid = m["merchant_id"]
    for d in days:
        if not C.is_weekday(d):
            continue
        n = rng.poisson(per_week / 5 if per_week else mean)
        payers = _sample(rng, pool, wts, n)
        sec = rng.uniform(10, 19, n) * 3600
        amt = np.clip(np.exp(rng.normal(np.log(25_000), 0.8, n)), 5_000, 200_000)
        amt = np.where(rng.random(n) < 0.6, np.round(amt / 1000) * 1000, np.round(amt))
        link = rng.random(n) < 0.5
        dd = np.full(n, d)
        w.add(mid, payers[link], dd[link], sec[link], amt[link], "UPI_INTENT", "PAYMENT_LINK")
        w.add(mid, payers[~link], dd[~link], sec[~link], amt[~link], "UPI_INTENT", "S2S_API")


# ------------------------------------------------------------------ betting
def gen_betting(w: _World, mid: str, bettors: np.ndarray, appetite: np.ndarray, days, routing: float,
                hosted_blocked_share=0.0):
    """Each assigned bettor is active on ~60% of match days and ~20% of non-match days (scaled by
    the operator's routing weight to this front), with 3-8 escalating deposits per active day."""
    rng = w.rng
    stakes, stake_p = C.BET_STAKES
    for d in days:
        match = C.is_match_day(d)
        p = (C.BET_P_MATCH if match else C.BET_P_NONMATCH) * routing * appetite
        active = bettors[rng.random(len(bettors)) < np.clip(p, 0, 0.95)]
        P, D, S, A = [], [], [], []
        for b in active:
            k = int(rng.integers(3, 9))
            if match:
                comp = rng.random(k)
                t = np.where(comp < 0.5, rng.uniform(*C.PREMATCH, k),
                             np.where(comp < 0.75, rng.uniform(*C.INPLAY, k), rng.uniform(22, 26, k)))
            else:
                t = np.where(rng.random(k) < 0.8, rng.uniform(20, 26, k), rng.uniform(12, 20, k))
            t = np.sort(t)
            amt = np.empty(k)
            amt[0] = rng.choice(stakes, p=stake_p)
            for i in range(1, k):
                amt[i] = min(amt[i - 1] * 2, C.BET_CAP) if rng.random() < C.BET_ESCALATE_P else amt[i - 1]
            amt[rng.random(k) < 0.04] = 5000
            P.append(np.full(k, b)); D.append(np.full(k, d)); S.append(t * 3600); A.append(amt)
        if not P:
            continue
        P, D, S, A = map(np.concatenate, (P, D, S, A))
        n = len(P)
        r = rng.random(n)
        hosted = r < hosted_blocked_share
        card = (~hosted) & (r < hosted_blocked_share + 0.15)
        upi = ~hosted & ~card
        w.add(mid, P[upi], D[upi], S[upi], A[upi], "UPI_INTENT", "S2S_API", src="bet")
        nc = int(card.sum())
        ctype = rng.choice(["PREPAID", "DEBIT", "CREDIT"], size=nc, p=[0.3, 0.5, 0.2]).astype(object)
        ccountry = np.where(rng.random(nc) < 0.1, rng.choice(["AE", "SG", "GB"], size=nc), "IN").astype(object)
        cdev = np.where(rng.random(nc) < 0.3, _dev(P[card]), None)
        w.add(mid, P[card], D[card], S[card], A[card], "CARD", "S2S_API", card_type=ctype,
              card_country=ccountry, device=cdev, src="bet")
        w.add(mid, P[hosted], D[hosted], S[hosted], A[hosted], "HOSTED_CHECKOUT", "WEBSITE",
              referrer=C.BLOCKED_BETTING_DOMAIN, device=_dev(P[hosted]), src="bet")


# ------------------------------------------------------------------ main
def generate(seed: int = C.SEED) -> dict[str, pd.DataFrame]:
    w = _World(seed)
    rng = w.rng
    all_days = np.arange(1, C.N_DAYS + 1)
    city_pools = {city: w.new_payers(300, reg) for city, reg in C.CITIES.items()}

    def mk(mid, name, cat, city, onboard, integration, domain, persona=None, fast_track=False):
        m = dict(merchant_id=mid, name=name, declared_category=cat, mcc=C.CATEGORIES[cat], city=city,
                 region=C.CITIES[city], onboard_day=onboard, integration_type=integration,
                 registered_domain=domain, onboarding_phone_hash=_h("PH", mid + "phone"),
                 director_id_hash=_h("DIR", mid + "dir"), settlement_account_hash=_h("ACC", mid + "acc"),
                 is_persona=persona is not None, persona_key=persona, fast_track_video_kyc=fast_track)
        w.merchants.append(m)
        return m

    M = {}
    for key, p in C.PERSONAS.items():
        M[key] = mk(f"M-{key}", p["name"], p["cat"], p["city"], p["onboard"], p["integration"], p["domain"], key)
    M["P8"]["settlement_account_hash"] = M["P6"]["settlement_account_hash"]
    M["P9"]["director_id_hash"] = M["P7"]["director_id_hash"]

    # ---------------- background: generated fictional names, 8 cities, long clean histories
    stems = list(rng.permutation(NAME_STEMS))
    used = set()
    bg = []
    code = {"Bakery": "B", "Wallet/recharge top-up": "R", "General merchandise": "G", "IT services": "T"}
    for cat, n in C.BACKGROUND.items():
        for i in range(1, n + 1):
            while True:
                nm = f"{stems[len(used) % len(stems)]} {NAME_SUFFIX[cat][(len(used) // len(stems) + i) % len(NAME_SUFFIX[cat])]}"
                if nm not in used:
                    break
                used.add(nm + "#")
            used.add(nm)
            city = str(rng.choice(C.BACKGROUND_CITIES))
            slug = nm.lower().replace(" ", "")
            if cat == "Bakery":
                integ, dom = ("website" if rng.random() < 0.6 else "QR"), f"{slug}.example"
            elif cat == "Wallet/recharge top-up":
                integ, dom = "website", f"{slug}.example"
            elif cat == "General merchandise":
                integ, dom = ("website" if rng.random() < 0.5 else "QR"), f"{slug}.example"
            else:
                integ, dom = "API", f"{slug}.example"
            bg.append(mk(f"M-{code[cat]}{i:02d}", nm, cat, city, -int(rng.integers(30, 701)), integ, dom))
    ft = C.FAST_TRACK
    kiran = mk("M-K01", ft["name"], ft["cat"], ft["city"], ft["onboard"], "API", "kiransoftwarelabs.example",
               fast_track=True)

    for m in bg:
        cat = m["declared_category"]
        if cat == "Bakery":
            gen_bakery(w, m, city_pools[m["city"]], all_days, festival_mult=rng.uniform(1.3, 2.0))
        elif cat == "Wallet/recharge top-up":
            gen_recharge(w, m, all_days)
        elif cat == "General merchandise":
            gen_general(w, m, all_days, festival_mult=rng.uniform(1.2, 1.5))
        else:
            gen_it(w, m, all_days)
    gen_it(w, kiran, np.arange(C.FAST_TRACK["onboard"], C.N_DAYS + 1), n_mean=6)

    # ---------------- genuine personas
    centre = dict(mean=60, median=220, shift=0.0, w_morning=0.42, web_share=0.10)
    gen_bakery(w, M["P1"], city_pools["Mumbai"], all_days, festival_mult=1.6, gift_share=0.22, params=centre)
    # P2: festival surge x4 with many new local payers; ticket mix like peers (gift boxes at the top of their range)
    p2_mean = gen_bakery(w, M["P2"], city_pools["Pune"], all_days, festival_mult=1.0, gift_share=0.28,
                         params=dict(centre, mean=58, median=225))
    fest_pool = w.new_payers(1600, M["P2"]["region"])
    fest_w = np.ones(len(fest_pool)) / len(fest_pool)
    for d in range(C.FESTIVAL_DAYS[0], C.FESTIVAL_DAYS[1] + 1):
        n = rng.poisson(p2_mean * 3)
        payers = _sample(rng, fest_pool, fest_w, n)
        amt = np.where(rng.random(n) < 0.28, rng.choice(GIFT_BOXES, size=n), _bakery_amounts(rng, n, 235))
        _counter_or_web(w, M["P2"]["merchant_id"], payers, np.full(n, d), _bakery_hours(rng, n, 0.0, 0.38), amt,
                        M["P2"]["registered_domain"], 0.10)
    p3_mean = gen_bakery(w, M["P3"], city_pools["Bengaluru"], all_days, festival_mult=1.5, gift_share=0.2,
                         params=dict(centre, mean=55, median=230, web_share=0.12))
    viral_pool = w.new_payers(6000, [r for r in C.REGIONS if r != M["P3"]["region"]])
    for d in range(46, C.N_DAYS + 1):
        n = rng.poisson(p3_mean * 5)
        payers = rng.choice(viral_pool, size=n, replace=False)
        sec = np.clip(rng.normal(15.5, 3.2, n), 8.5, 22.5) * 3600
        amt = _nonround(np.exp(rng.normal(np.log(480), 0.45, n)))
        w.add(M["P3"]["merchant_id"], payers, np.full(n, d), sec, amt, "HOSTED_CHECKOUT", "WEBSITE",
              referrer=M["P3"]["registered_domain"], device=_dev(payers))
    gen_recharge(w, M["P4"], all_days, params=dict(mean=220))
    gen_bakery(w, M["P5"], city_pools["Jaipur"], all_days, festival_mult=1.5, gift_share=0.2,
               params=dict(centre, mean=75, median=240, web_share=0.0), pool_size=600)

    # ---------------- thin genuine-looking cover for the fronts
    gen_bakery(w, M["P6"], city_pools["Lucknow"], np.arange(20, 56), festival_mult=1.3,
               params=dict(mean=20, median=240, web_share=0.0), pool_size=180)
    gen_general(w, M["P7"], np.arange(15, C.N_DAYS + 1), n_mean=15)
    gen_it(w, M["P8"], np.arange(18, C.N_DAYS + 1), per_week=2)
    gen_bakery(w, M["P9"], city_pools["Kanpur"], np.arange(50, C.N_DAYS + 1),
               params=dict(mean=20, median=210, web_share=0.0), pool_size=150)

    # ---------------- betting operator
    pool = w.new_payers(C.BETTOR_POOL, C.REGIONS)
    appetite = dict(zip(pool, rng.uniform(0.5, 1.5, len(pool))))
    p6 = rng.choice(pool, size=C.BETTORS["P6"], replace=False)
    not_p6 = np.setdiff1d(pool, p6)

    def sibling(key):
        n = C.BETTORS[key]
        k = int(np.ceil(n * C.BETTOR_OVERLAP_WITH_P6[key]))
        return np.concatenate([rng.choice(p6, size=k, replace=False), rng.choice(not_p6, size=n - k, replace=False)])

    sets = {"P6": p6, "P7": sibling("P7"), "P8": sibling("P8"), "P5": sibling("P5")}
    sets["P9"] = np.array(sorted(p6, key=lambda b: -appetite[b])[: int(np.ceil(len(p6) * C.P9_INHERIT_SHARE))])
    for key in ["P6", "P7", "P8", "P5", "P9"]:
        b = sets[key]
        gen_betting(w, M[key]["merchant_id"], b, np.array([appetite[x] for x in b]),
                    np.arange(C.BET_START[key], C.BET_END[key] + 1), C.BET_ROUTING[key],
                    hosted_blocked_share=0.05 if key == "P6" else 0.0)

    # ---------------- assemble transactions
    cols = {k: np.concatenate([c[k] for c in w.chunks]) for k in w.chunks[0]}
    tx = pd.DataFrame(cols)
    overflow = tx["sec"] >= 86400
    tx.loc[overflow, "day"] += 1
    tx.loc[overflow, "sec"] -= 86400
    tx = tx[tx["day"] <= C.N_DAYS].copy()
    tx["sec"] = np.floor(tx["sec"]).astype(np.int64)
    tx["ts"] = pd.Timestamp(C.DAY1) + pd.to_timedelta(tx["day"] - 1, unit="D") + pd.to_timedelta(tx["sec"], unit="s")
    payers = pd.DataFrame(w.payers, columns=["pid", "home_region", "bank_code", "has_upi", "has_card"])
    tx = tx.merge(payers[["pid", "home_region"]], left_on="payer", right_on="pid", how="left")
    upi = tx["method"].isin(["UPI_INTENT", "UPI_QR"])
    tx.loc[upi, ["device_id", "referrer_domain"]] = None  # data availability mirrors reality
    tx["payer_region"] = tx["home_region"]  # UPI: bank-code mapping; hosted: IP geolocation
    tx.loc[(tx["method"] == "CARD") & tx["device_id"].isna(), "payer_region"] = None
    card = tx["method"] == "CARD"
    tx["card_token_hash"] = None
    tx.loc[card, "card_token_hash"] = [f"CT-{x:08x}" for x in rng.integers(0, 2**31, int(card.sum()))]
    tx["settled"] = True
    tx = tx.sort_values(["ts", "merchant_id"], kind="mergesort").reset_index(drop=True)
    tx["txn_id"] = [f"T{i:07d}" for i in range(len(tx))]
    tx["payer_id"] = tx["payer"].map(lambda x: f"PY{x:06d}")
    merchants = pd.DataFrame(w.merchants)
    api_logs = _api_logs(tx, merchants, rng)
    tx = tx[["txn_id", "merchant_id", "payer_id", "ts", "day", "amount_inr", "method", "channel",
             "payer_region", "device_id", "referrer_domain", "card_type", "card_issuer_country",
             "card_token_hash", "settled"]]
    payers_df = pd.DataFrame({
        "payer_id": payers["pid"].map(lambda x: f"PY{x:06d}"),
        "upi_id_hash": [(_h("UPI", str(p)) if u else None) for p, u in zip(payers["pid"], payers["has_upi"])],
        "card_token_hash": [(_h("CARD", str(p)) if c else None) for p, c in zip(payers["pid"], payers["has_card"])],
        "bank_code": payers["bank_code"], "home_region": payers["home_region"],
    })
    return dict(merchants=merchants, payers=payers_df, transactions=tx, api_logs=api_logs,
                dashboard_logins=_logins(merchants, rng), payouts=_payouts(merchants, rng),
                national_lists=_national_lists(merchants))


def _api_logs(tx, merchants, rng):
    """Daily API aggregates per server. Betting traffic is created from the operator's offshore server;
    API merchants' own traffic comes from their own Indian server; website checkouts use a common plugin."""
    m = merchants.set_index("merchant_id")
    persona = m["persona_key"].to_dict()
    own = {mid: f"192.0.2.{150 + i}" for i, mid in enumerate(m.index[m["integration_type"] == "API"])}
    t = tx[tx["channel"].isin(["S2S_API", "PAYMENT_LINK", "WEBSITE"])].copy()
    t["kind"] = np.where(t["src"] == "bet", "bet", t["channel"])
    g = t.groupby(["merchant_id", "day", "kind"]).size().unstack(fill_value=0).reset_index()
    for c in ["bet", "S2S_API", "PAYMENT_LINK", "WEBSITE"]:
        if c not in g:
            g[c] = 0
    rows = []
    for r in g.itertuples(index=False):
        mid, day = r.merchant_id, r.day
        if r.bet > 0:
            rows.append(dict(merchant_id=mid, day=day, payment_links_created=int(r.bet * rng.uniform(1.3, 1.6)),
                             s2s_api_calls=int(r.bet * 3), server_ip=C.OFFSHORE_IPS[persona[mid]],
                             server_asn_label=C.OFFSHORE_ASN, server_country="SG"))
        if r.S2S_API > 0 or r.PAYMENT_LINK > 0:
            ip, asn = (own[mid], C.OWN_SERVER_ASN) if mid in own else (C.PLUGIN_IP, C.PLUGIN_ASN)
            rows.append(dict(merchant_id=mid, day=day, payment_links_created=int(r.PAYMENT_LINK * 1.1),
                             s2s_api_calls=int(r.S2S_API * 2 + r.PAYMENT_LINK), server_ip=ip,
                             server_asn_label=asn, server_country="IN"))
        if r.WEBSITE > 0:
            rows.append(dict(merchant_id=mid, day=day, payment_links_created=0, s2s_api_calls=int(r.WEBSITE),
                             server_ip=C.PLUGIN_IP, server_asn_label=C.PLUGIN_ASN, server_country="IN"))
    return pd.DataFrame(rows)


def _logins(merchants, rng):
    rows = []
    carrier = [f"198.51.100.{200 + i}" for i in range(10)]
    cloud = [f"192.0.2.{10 + i}" for i in range(5)]
    base = pd.Timestamp(C.DAY1)
    for m in merchants.itertuples(index=False):
        pk = m.persona_key
        if pk in ("P6", "P7", "P8", "P9"):
            start = 56 if pk == "P9" else m.onboard_day  # operator moves onto Urban Crumbs after the suspension
            end = 55 if pk == "P6" else C.N_DAYS
            for d in range(start, end + 1):
                for _ in range(int(rng.integers(1, 3))):
                    ts = base + pd.Timedelta(days=d - 1, seconds=int(rng.integers(10 * 3600, 23 * 3600)))
                    rows.append(dict(merchant_id=m.merchant_id, ts=ts, device_id=C.OPERATOR_DEVICE,
                                     ip=C.OPERATOR_IP, ip_type="FIXED"))
            continue
        for d in range(max(1, m.onboard_day), C.N_DAYS + 1):
            if rng.random() > 0.5:
                continue
            if rng.random() < 0.6:
                ip, t = str(rng.choice(carrier)), "CARRIER"
            else:
                ip, t = str(rng.choice(cloud)), "CLOUD"
            ts = base + pd.Timedelta(days=d - 1, seconds=int(rng.integers(8 * 3600, 22 * 3600)))
            rows.append(dict(merchant_id=m.merchant_id, ts=ts, device_id=f"DEV-{m.merchant_id}", ip=ip, ip_type=t))
    return pd.DataFrame(rows)


def _payouts(merchants, rng):
    rows = []
    base = pd.Timestamp(C.DAY1)
    for m in merchants.itertuples(index=False):
        if m.persona_key in ("P6", "P7"):
            end = 55 if m.persona_key == "P6" else C.N_DAYS
            for d in range(m.onboard_day + 2, end + 1, 2):
                rows.append(dict(merchant_id=m.merchant_id,
                                 ts=base + pd.Timedelta(days=d - 1, hours=int(rng.integers(1, 5))),
                                 amount_inr=float(rng.integers(150_000, 600_000)),
                                 beneficiary_count=int(rng.integers(25, 70))))
        elif not m.is_persona and m.declared_category == "General merchandise":
            for d in range(int(rng.integers(1, 8)), C.N_DAYS + 1, 7):  # weekly suppliers and salaries
                rows.append(dict(merchant_id=m.merchant_id, ts=base + pd.Timedelta(days=d - 1, hours=11),
                                 amount_inr=float(rng.integers(20_000, 120_000)),
                                 beneficiary_count=int(rng.integers(2, 10))))
    return pd.DataFrame(rows)


def _national_lists(merchants):
    m = merchants.set_index("persona_key")
    phones = [m.loc["P6", "onboarding_phone_hash"]] + [_h("PH", f"fri{i}") for i in range(30)]
    accts = [m.loc["P8", "settlement_account_hash"]] + [_h("ACC", f"i4c{i}") for i in range(30)]
    rows = ([dict(list="fri_high_risk_phones", value=v, source="DoT FRI, simulated") for v in phones]
            + [dict(list="i4c_mule_accounts", value=v, source="I4C Suspect Registry, simulated") for v in accts]
            + [dict(list="blocked_domains", value=v, source="MeitY blocked list, simulated") for v in C.BLOCKED_DOMAINS])
    return pd.DataFrame(rows)


def load_or_generate(seed: int = C.SEED) -> dict[str, pd.DataFrame]:
    """Load cached parquet files from data/, generating them on first run."""
    names = ["merchants", "payers", "transactions", "api_logs", "dashboard_logins", "payouts", "national_lists"]
    paths = {n: DATA_DIR / f"{n}_seed{seed}.parquet" for n in names}
    if all(p.exists() for p in paths.values()):
        return {n: pd.read_parquet(p) for n, p in paths.items()}
    data = generate(seed)
    DATA_DIR.mkdir(exist_ok=True)
    for stale in DATA_DIR.glob(f"timeline_seed{seed}.parquet"):  # replay depends on the data
        stale.unlink()
    for n, df in data.items():
        df.to_parquet(paths[n], index=False)
    return data


if __name__ == "__main__":
    import time
    t = time.time()
    d = generate()
    tx = d["transactions"]
    print(f"generated in {time.time() - t:.1f}s; {len(tx):,} txns, {len(d['merchants'])} merchants")
    s = tx.merge(d["merchants"][["merchant_id", "persona_key"]], on="merchant_id")
    print(s[s["persona_key"].notna()].groupby("persona_key").agg(n=("txn_id", "size"), inr=("amount_inr", "sum")))
