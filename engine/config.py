"""All constants, weights, thresholds and dates for the prototype.

Everything here is synthetic. Tune weights only here.
"""
from datetime import datetime, timedelta

SEED = 42

# ---------------------------------------------------------------- time
N_DAYS = 60
DAY1 = datetime(2026, 9, 1)  # Day 1, IST (timestamps are naive IST); a Tuesday
DEFAULT_DAY = 55
SUSPEND_DAY = 60
FESTIVAL_DAYS = (40, 46)  # inclusive, "Diwali (synthetic)"

# ---------------------------------------------------------------- fixtures
MATCH_EVERY = 3  # T20 match (synthetic) on days 3, 6, 9, ... 60
MATCH_START_H = 19.5
PREMATCH = (15.5, 19.5)  # 15:30-19:30
INPLAY = (19.5, 23.0)  # 19:30-23:00
NIGHT = (0.0, 5.0)  # 00:00-05:00


def is_match_day(day: int) -> bool:
    return day % MATCH_EVERY == 0


def is_weekday(day: int) -> bool:
    return (DAY1 + timedelta(days=int(day) - 1)).weekday() < 5


def is_festival(day: int) -> bool:
    return FESTIVAL_DAYS[0] <= day <= FESTIVAL_DAYS[1]


# ---------------------------------------------------------------- categories
CATEGORIES = {
    "Bakery": 5462,
    "Wallet/recharge top-up": 4814,
    "General merchandise": 5399,
    "IT services": 7372,
}

CITIES = {
    "Mumbai": "MH", "Pune": "MH", "Bengaluru": "KA", "Delhi": "DL",
    "Jaipur": "RJ", "Chennai": "TN", "Kolkata": "WB", "Hyderabad": "TS",
    "Lucknow": "UP", "Indore": "MP", "Kanpur": "UP",
}
BACKGROUND_CITIES = ["Mumbai", "Pune", "Bengaluru", "Delhi", "Jaipur", "Chennai", "Kolkata", "Hyderabad"]
REGIONS = ["MH", "KA", "DL", "TN", "WB", "UP", "GJ", "RJ", "KL", "TS", "AP", "PB",
           "HR", "MP", "BR", "OR", "AS", "JH", "CG", "UK", "HP", "GA", "JK", "TR", "ML"]

# ---------------------------------------------------------------- personas
PERSONAS = {
    "P1": dict(name="Sharma Bakery", cat="Bakery", city="Mumbai", onboard=-420,
               integration="QR", domain="sharmabakery.example", story="Genuine neighbourhood bakery"),
    "P2": dict(name="Diwali Delights Bakery", cat="Bakery", city="Pune", onboard=-610,
               integration="QR", domain="diwalidelights.example",
               story="Genuine; big festival surge on days 40-46 (peers surge too)"),
    "P3": dict(name="Crumb & Co (viral)", cat="Bakery", city="Bengaluru", onboard=-380,
               integration="website", domain="crumbandco.example",
               story="Genuine; goes viral days 46-55, pan-India website orders"),
    "P4": dict(name="QuickTopUp", cat="Wallet/recharge top-up", city="Delhi", onboard=-700,
               integration="website", domain="quicktopup.example",
               story="Genuine; round amounts and repeat payers, declared as top-up"),
    "P5": dict(name="Gupta Sweets & Bakery (rented)", cat="Bakery", city="Jaipur", onboard=-900,
               integration="QR", domain="guptasweets.example",
               story="Genuine sales; from day 30 a betting cluster rides inside the account"),
    "P6": dict(name="Royal Cakes & Bakes", cat="Bakery", city="Lucknow", onboard=20,
               integration="QR", domain="royalcakes.example", story="Betting front from onboarding"),
    "P7": dict(name="Shree Ganesh Traders", cat="General merchandise", city="Indore", onboard=15,
               integration="QR", domain=None, story="Sibling front, same operator"),
    "P8": dict(name="TechNova IT Solutions", cat="IT services", city="Hyderabad", onboard=18,
               integration="API", domain="technova-it.example",
               story="Sibling front, declared API integration"),
    "P9": dict(name="Urban Crumbs", cat="Bakery", city="Kanpur", onboard=50,
               integration="QR", domain=None,
               story="Successor front; near-idle until Royal Cakes & Bakes is suspended"),
}
GENUINE_PERSONAS = ["P1", "P2", "P3", "P4"]
FRONT_PERSONAS = ["P5", "P6", "P7", "P8", "P9"]  # excluded from cohort baselines

BACKGROUND = {
    "Bakery": 40,
    "Wallet/recharge top-up": 15,
    "General merchandise": 15,
    "IT services": 10,
}
# a genuine new IT firm onboarded through fast-track video KYC
FAST_TRACK = dict(name="Kiran Software Labs", cat="IT services", city="Bengaluru", onboard=40)

# ---------------------------------------------------------------- betting operator
BETTOR_POOL = 400
BETTORS = {"P6": 250, "P7": 180, "P8": 150, "P5": 120}
BETTOR_OVERLAP_WITH_P6 = {"P7": 0.62, "P8": 0.52, "P5": 0.70}
P9_INHERIT_SHARE = 0.80  # share of P6's top bettors moved to P9 after suspension
BET_P_MATCH = 0.60  # chance an assigned bettor is active on a match day
BET_P_NONMATCH = 0.20  # ... and on a non-match day
BET_STAKES = ([500, 1000, 2000], [0.5, 0.35, 0.15])
BET_ESCALATE_P = 0.4
BET_CAP = 10_000
# routing weight: share of an assigned bettor's activity the operator sends to each front
BET_ROUTING = {"P6": 1.0, "P7": 1.0, "P8": 1.0, "P9": 0.95, "P5": 0.040}
BET_START = {"P6": 20, "P7": 15, "P8": 18, "P5": 30, "P9": 56}
BET_END = {"P6": 55, "P7": 60, "P8": 60, "P5": 60, "P9": 60}

OPERATOR_DEVICE = "DEV-OPR-01"
OPERATOR_IP = "198.51.100.23"
OFFSHORE_ASN = "OffshoreHost-SG"
OFFSHORE_IPS = {"P6": "203.0.113.10", "P5": "203.0.113.10", "P7": "203.0.113.11",
                "P8": "203.0.113.12", "P9": "203.0.113.10"}
PLUGIN_IP = "192.0.2.250"
PLUGIN_ASN = "ShopPlugin-IN"
OWN_SERVER_ASN = "OwnServer-IN"
COMMON_PLATFORM_ASNS = {PLUGIN_ASN}
BLOCKED_BETTING_DOMAIN = "luckystrike-exch.example"
BLOCKED_DOMAINS = [BLOCKED_BETTING_DOMAIN, "cricwin-365.example", "fastpaisa-bet.example"]

# ---------------------------------------------------------------- bet-shape score
SCORE_W = dict(bias=-4.0, rhythm=2.0, round=1.2, escalation=1.5, prematch=1.5, night=0.8, remote=0.8)
GMM_K = (1, 2, 3, 4)
MIN_CLUSTER_SHARE = 0.05
CLUSTER_FEATURES = ["txns_per_active_day", "max_txns_in_day", "round_share", "escalation_share",
                    "prematch_share", "night_share", "remote_flag", "log_mean_amount"]

HIDDEN_SCORE = 0.7
HIDDEN_VOL_SHARE = 0.15
HIDDEN_PAYER_SHARE = 0.10
MERCHANT_SCORE_FLAG = 0.7

# minimum data for clustering and peer-distance tiers
MIN_ACTIVE_DAYS = 14
MIN_TXNS = 200
MIN_PAYERS = 30

# declaration gap
GAP_Z = 3.0
GAP_MIN_OBSERVED = 0.6
GAP_EPS = 0.02
RECAT_RATIO = 0.5

# independent signals and tier 2
S2S_SHARE = 0.20
LINK_SPIKE = 5.0
MATCHDAY_Z = 3.0
TIER2_Z = 3.0
SURGE_RATIO = 3.0  # merchant surge ratio / cohort median surge ratio
MIN_HISTORY_DAYS = 28

# ---------------------------------------------------------------- network
EDGE_W = dict(device=1.0, settlement=1.0, director=0.8, ip_fixed=0.6, ip_cloud=0.05,
              ip_carrier=0.0, server_ip=0.6, server_common=0.02)
LINK_THRESHOLD = 0.8
WEAK_EDGE_HIDE = 0.3
SUCCESSOR_WINDOW = 5
SUCCESSOR_SHARE = 0.30

# ---------------------------------------------------------------- response
RESERVE_MULT = 1.5
RESERVE_HOURS = 48
EARNED_ACCESS_DAYS = 45

TIER_LABEL = {1: "Normal", 2: "Nudge", 3: "Reserve + review", 4: "Confirmed front"}
TIER_SHORT = {1: "Normal", 2: "Nudge", 3: "Review", 4: "Confirmed"}
# muted, editorial palette: sage, ochre, rust, oxblood
TIER_COLOR = {1: "#4A6B53", 2: "#A47A22", 3: "#B4532A", 4: "#7E2621"}
TIER_TINT = {1: "#EAF0EA", 2: "#F6EEDC", 3: "#F7E6DC", 4: "#F2DEDC"}
INK = "#1A1A18"
ACCENT = INK
