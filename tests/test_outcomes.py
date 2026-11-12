"""Acceptance tests (design doc §10 plus the clarifications). Seed 42 throughout."""
import pandas as pd
import pytest

from engine import config as C
from engine import response
from engine.generate import generate, load_or_generate
from engine.pipeline import run


def M(key):
    return f"M-{key}"


@pytest.fixture(scope="module")
def data():
    return load_or_generate(C.SEED)


@pytest.fixture(scope="module")
def r55(data):
    return run(C.DEFAULT_DAY, data=data)


@pytest.fixture(scope="module")
def r_confirm(data):
    return run(C.DEFAULT_DAY, confirmed={"P6"}, data=data)


@pytest.fixture(scope="module")
def r60(data):
    return run(C.SUSPEND_DAY, confirmed={"P6"}, suspended={"P6": C.DEFAULT_DAY}, data=data)


# ------------------------------------------------------------ day 55
@pytest.mark.parametrize("key", ["P1", "P2", "P4"])
def test_genuine_personas_green(r55, key):
    assert r55.tiers[M(key)] == 1


def test_viral_bakery_nudge_only(r55):
    assert r55.tiers[M("P3")] == 2
    assert not r55.cs.at[M("P3"), "hidden_cluster"]
    assert r55.cs.at[M("P3"), "max_cluster_score"] < 0.5


def test_rented_bakery_hidden_cluster(r55):
    mid = M("P5")
    assert r55.tiers[mid] == 3
    assert r55.f.at[mid, "merchant_score"] < 0.6
    summ = r55.clusters[mid][0]
    bet = summ[summ["score"] >= 0.7]
    assert len(bet) >= 1
    assert 0.25 <= bet["volume_share"].sum() <= 0.55


def test_rented_bakery_share_from_day_30(r55):
    assert 0.35 <= r55.cs.at[M("P5"), "bet_volume_share_since"] <= 0.45


@pytest.mark.parametrize("key", ["P6", "P7", "P8"])
def test_fronts_flagged(r55, key):
    assert r55.tiers[M(key)] == 3


def test_fronts_mostly_betting(r55):
    for key in ["P6", "P7", "P8"]:
        assert r55.cs.at[M(key), "bet_cluster_volume_share"] >= 0.80


def test_api_category_not_channel_anomaly(r55):
    assert not r55.sig.at[M("P8"), "channel"]


def test_successor_quiet_before_suspension(r55):
    assert r55.tiers[M("P9")] == 1
    assert not r55.f.at[M("P9"), "sufficient"]


def test_background_merchants(r55):
    bg = r55.merchants.loc[~r55.merchants["is_persona"], "merchant_id"]
    tiers = r55.tiers[bg]
    assert (tiers <= 2).mean() >= 0.95
    assert (tiers >= 3).sum() == 0


def test_no_genuine_linked_to_front(r55):
    genuine = set(r55.merchants.loc[~r55.merchants["persona_key"].isin(C.FRONT_PERSONAS), "merchant_id"])
    fronts = {M(k) for k in C.FRONT_PERSONAS}
    e = r55.edges[r55.edges["linked"]]
    bad = e[(e["a"].isin(genuine) & e["b"].isin(fronts)) | (e["b"].isin(genuine) & e["a"].isin(fronts))]
    assert bad.empty, bad


# ------------------------------------------------------------ after "Confirm P6"
def test_confirm_propagates(r_confirm):
    assert r_confirm.tiers[M("P6")] == 4
    for key in ["P7", "P8"]:
        assert r_confirm.tiers[M(key)] == 3
        assert r_confirm.sig.at[M(key), "network"]


# ------------------------------------------------------------ after "Suspend P6 and advance"
def test_successor_detected(r60):
    assert r60.tiers[M("P9")] == 3
    s = r60.successors
    row = s[(s["merchant_id"] == M("P9")) & (s["predecessor"] == M("P6"))]
    assert len(row) == 1 and bool(row["flagged"].iloc[0])
    assert row["share"].iloc[0] >= 0.30
    # existing siblings are not successors
    assert not s.loc[s["merchant_id"] == M("P7"), "flagged"].any()


# ------------------------------------------------------------ data and response
def test_data_availability(data):
    tx = data["transactions"]
    upi = tx[tx["method"].isin(["UPI_INTENT", "UPI_QR"])]
    assert upi["device_id"].isna().all()
    assert upi["referrer_domain"].isna().all()


def test_determinism():
    a, b = generate(C.SEED)["transactions"], generate(C.SEED)["transactions"]
    assert len(a) == len(b)
    assert a["amount_inr"].sum() == b["amount_inr"].sum()
    pd.testing.assert_frame_equal(a, b)


def test_reserve_for_rented_bakery(r55):
    res = response.reserve(r55, M("P5"))
    assert res["applies"] and res["mode"] == "cluster"
    assert res["reserve_held"] > 0
    assert res["genuine_today"] > 0 and res["settles"] == pytest.approx(res["genuine_today"])


def test_clean_days(r55):
    for key in ["P6", "P7", "P8"]:
        assert response.clean_days(r55, M(key))["total"] == 0
    assert response.clean_days(r55, M("P9"))["total"] == 6


def test_fast_track_unlocks_genuine_new_merchant(r55):
    ea = response.earned_access(r55, "M-K01")
    assert ea["fast_track"] and ea["unlocked"]
    assert r55.tiers["M-K01"] == 1
    p8 = response.earned_access(r55, M("P8"))
    assert "Fast track would be refused" in p8["counterfactual"]


def test_merchant_score_is_payer_weighted(r55):
    pf = r55.pf[r55.pf["merchant_id"] == M("P5")]
    assert r55.f.at[M("P5"), "m_rhythm_tpd"] == pytest.approx(pf["txns_per_active_day"].mean())


def test_analyst_clear_overrides_but_model_keeps_score(data):
    r = run(C.DEFAULT_DAY, data=data, cleared={"P5"})
    assert r.tiers[M("P5")] == 1
    assert r.extra["model_tiers"][M("P5")] == 3
    assert r.tiers[M("P6")] == 3  # clearing one merchant does not hide the rest of the network
