# Betting-Shaped Money: Merchant Monitor (prototype)

A working prototype that spots payment-aggregator merchants whose money is *betting-shaped* (for example, "a bakery that takes bets"), even when betting is hidden inside genuine sales. It links front merchants into networks and responds proportionately so genuine merchants are not harmed. Working prototype link is present here: https://merchant-monitor-75zpzoeul7zm3tnzxyajvg.streamlit.app/ 

**Prototype · synthetic data.** Every merchant, person, IP (RFC 5737) and domain (`.example`) is fictional. There are no real names and no external calls, and no figure claims real-world accuracy.

## Setup and run

Requires Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first launch generates the synthetic world (about 490k transactions across 90 merchants, seed 42) into `data/` and caches it. Later launches load it from disk. Scoring takes about 4 seconds per demo state and is cached for the session.

Run the acceptance tests:

```bash
pytest -q
```

## What the analyst sees

- **Needs attention** (Portfolio): every Tier 2+ merchant, worst first, with the reason, the response and the day it was first flagged.
- **Case summary**: a plain-English paragraph per merchant, generated from the computed numbers.
- **Time to detection**: the engine is replayed for every day (days 12–60) to find when each merchant was first flagged, relative to its first suspicious activity. Gupta Sweets: 5 days. Royal Cakes and TechNova: same day, through national lists. Shree Ganesh: 13 days, held back by the 14-active-day minimum (Earned Access would have kept its server API locked meanwhile). Urban Crumbs: same day as the successor move. No genuine merchant reaches Tier 3 on any day.
- **One payer's match day**: a real flagged payer's deposits on one match evening (₹500 → ₹10,000), the clearest picture of "betting-shaped".
- **Analyst actions**: confirm a front, clear a false positive (the model's findings stay on file and the case can be reopened), request documents, release a reserve. Every action is written to the case log.
- **Message to merchant**: a preview of the notice the merchant receives when volume is held.

The day-by-day replay runs once in the background on first launch (about 2 minutes) and is cached in `data/`. Open the app a couple of minutes before recording, or run `python -c "from engine import timeline; timeline.compute()"` beforehand.

## Presenting

- **Guided demo** (sidebar → *Start guided demo*): nine steps, each of which jumps to the right page, merchant, tab and scenario state and shows a scene caption. **Back** and **Next** move between steps; captions can be switched off for a clean recording.
- **Manual controls** (sidebar expander or the Network page stepper): Confirm Royal Cakes → Suspend and advance to day 60 → Reset.
- **Deep links**: `?step=5` opens guided step 5 directly. `?page=Merchant&m=P5&tab=Timeline`, `?page=Network&state=confirm` and `?page=Network&state=suspend&m=P9` also work.
- Every number on screen is computed from the synthetic data; nothing is hard-coded.

### Demo script (2 minutes)

1. **Portfolio**: Sharma Bakery and Diwali Delights are green, Crumb & Co is amber, QuickTopUp is green.
2. **Diwali Delights → Timeline (Transactions)**: 4× festival spike while peer bakeries rise 1.6× on the same days, with a ticket mix in line with theirs, so it stays green.
3. **Crumb & Co → Payer clusters**: viral, pan-India web orders. Volume is 10.2× while peers are flat, but no betting cluster, so it gets a nudge only.
4. **QuickTopUp → Money shape**: round amounts, declared as top-up, so it stays green.
5. **Gupta Sweets → Payer clusters**: merchant-level score 0.04, but a hidden cluster (6% of payers, 43% of volume since day 30) scores 0.89.
6. **Gupta Sweets → Settlement and reserve**: its own customers settle (₹17,720); today's ₹15,000 from the betting group is held.
7. **Network**: Royal Cakes, Shree Ganesh and TechNova linked by device, account and bettors; Gupta Sweets via the API server.
8. **Confirm**: linked merchants go to review.
9. **Suspend and advance to day 60**: Urban Crumbs is flagged as the successor (74% of bettors).

## Project layout

```
app.py                 Streamlit entry: sidebar, routing, demo controls
views/                 overview.py · merchant.py · network.py · about.py · common.py
engine/
  config.py            every weight, threshold, date and the seed
  generate.py          synthetic world (merchants, payers, transactions, API logs, logins, payouts, lists)
  features.py          payer- and merchant-level features, peer distances, robust z
  clustering.py        GMM hidden-cluster detection (k by BIC, small clusters merged), PCA coords
  scoring.py           bet-shape score, declaration gap, independent signals, tiers
  network.py           edge weighting (IDF payers, device, account, director, IPs, servers), successors, layout
  response.py          reserve, earned access, national checks
  explain.py           plain-language reasons, generated from computed numbers
  pipeline.py          runs everything for a given day and demo state
tests/test_outcomes.py acceptance tests (design doc §10)
```

## Spec decisions

### Still open (text missing from the original design doc)

1. **§5.4 independent signals** is cut off after "National hit: FRI phon…". Implemented: FRI phone, I4C account or blocked referrer domain. Card mix and payout fan-out appear as evidence only.
2. **§6 Network** is missing its heading and opening. Implemented: nodes = merchants, undirected edges, all observed days, payer component normalised by the largest pair.
3. **Tier 2 rule** reads "any …obust z > 3". Implemented: event-aware ticket-size or hour-of-day robust z > 3, or relative surge > 3, or a recategorisation suggestion.

### Clarified and implemented

- **Genuine general merchandise and IT services** generators use the parameters given (IT on weekdays only, from its own Indian server "OwnServer-IN"; API use is not a channel anomaly for API-declared merchants).
- **Bettors** are active on ~60% of match days and ~20% of non-match days, with 3–8 deposits per active day. Fronts: P7 ≈ 75% and P8 ≈ 65% of P6; P9 ≈ 75–80% of P6's prior daily volume. P5's betting group is 43% of its volume from day 30 (30% over days 1–55).
- **Thin cover** for the fronts (betting is ≥ 98% of each front's volume). Onboarding: P6 day 20, P7 day 15, P8 day 18.
- **Background**: generated fictional names, eight cities, onboarding between day −700 and −30, CARRIER or CLOUD logins, weekly supplier payouts for general merchandise only.
- **Reserve**: with a flagged payer cluster, only today's flagged-cluster volume is held and genuine payers settle; otherwise the surge above 1.5× the pre-anomaly baseline (cohort median when history is short).
- **Clean days** are computed: a day with ≥ 1 transaction, no flagged-cluster or offshore/undeclared API activity, and no national-list hit. History before day 1 counts as clean for long-standing merchants. P6–P8 = 0, P9 = 6.
- **Successor rule** as formalised (new payers only; ≥ 30% of X's flagged cluster).
- **Event-aware comparison**: festival days against peers' festival days, normal days against normal days; the larger robust z counts. Surge is divided by the peer median surge. Background bakeries sell gift boxes on 15–30% of festival-day transactions.
- **Minimum data**: 14 active days, 200 transactions, 30 payers; below that "Insufficient history: monitoring".
- **Kiran Software Labs** (IT, day 40, fast-track video KYC) shows Earned Access unlocking for a genuine new business.
- **Explanation panel for every tier**; the merchant score is payer-weighted; payer share and volume share are shown together.

### Interpretation calls worth confirming

- Parts of the clarification text were garbled. Read as: "round invoices are not betting-shaped"; "no payouts for bakery and recharge merchants"; "no national-list hit".
- **Robust z on log distances.** Distances are positive and right-skewed, so a plain robust z put about 10 of 81 genuine merchants above 3 by chance. Taking the log first drops that to 1 of 81.
- **P2's festival ticket mix** matches peers (gift boxes on 28% of festival-day transactions) rather than "median ₹650". This keeps it green under the event-aware rule you asked for.
- **New merchants with a national-list hit** reach Tier 3 even with insufficient history (as clarification 11 allows).

### Realism fixes found by the replay

- Payer groups under 5% of payers are no longer merged away when they carry 15%+ of volume since they appeared: a small group moving a lot of money is the risk. This cut Gupta Sweets' detection lag from 20 days to 5.
- The volume-share test for a flagged cluster uses either the full window or the period since the group appeared.
- National-list hits only escalate once a merchant is trading (not before onboarding).

### Known demo sensitivities

- The Gupta Sweets ↔ Royal Cakes link is 0.82 at day 55 (threshold 0.8). At day 60 it falls to 0.76 as payer weights renormalise. Gupta Sweets stays Tier 3 on its own evidence.
- The recategorisation suggestion is implemented but never fires in this dataset.

## Determinism

Seed 42 is used throughout: the data generator, GMM `random_state`, PCA and the network layout. `tests/test_outcomes.py::test_determinism` regenerates the world twice and asserts identical frames.
