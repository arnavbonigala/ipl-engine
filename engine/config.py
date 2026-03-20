"""Strategy constants and API endpoints."""

KELLY_FRACTION = 0.5
MAX_BET_FRACTION = 0.25
MAX_ENTRY_PRICE = 0.55
USE_STOP_LOSS = False
STOP_LOSS_CENTS = 0.15
MIN_CONFIDENCE = 0.60
MIN_EDGE = 0.03
POLL_INTERVAL_SECS = 30
SCRAPE_POLL_SECS = 60

POLYMARKET_HOST = "https://clob.polymarket.com"
GAMMA_HOST = "https://gamma-api.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet

STATE_FILE = "engine_state.json"
DASHBOARD_PORT = 8050

TEAM_ABBREVS = {
    "Mumbai Indians": "mi",
    "Chennai Super Kings": "csk",
    "Royal Challengers Bengaluru": "rcb",
    "Kolkata Knight Riders": "kkr",
    "Rajasthan Royals": "rr",
    "Sunrisers Hyderabad": "srh",
    "Delhi Capitals": "dc",
    "Punjab Kings": "pbks",
    "Lucknow Super Giants": "lsg",
    "Gujarat Titans": "gt",
}

TEAM_SLUG_NAMES = {
    "Mumbai Indians": "mumbai-indians",
    "Chennai Super Kings": "chennai-super-kings",
    "Royal Challengers Bengaluru": "royal-challengers-bangalore",
    "Kolkata Knight Riders": "kolkata-knight-riders",
    "Rajasthan Royals": "rajasthan-royals",
    "Sunrisers Hyderabad": "sunrisers-hyderabad",
    "Delhi Capitals": "delhi-capitals",
    "Punjab Kings": "punjab-kings",
    "Lucknow Super Giants": "lucknow-super-giants",
    "Gujarat Titans": "gujarat-titans",
}

TEAM_KEYWORDS = {
    "Mumbai Indians": ["mumbai"],
    "Chennai Super Kings": ["chennai"],
    "Royal Challengers Bengaluru": ["bangalore", "bengaluru", "challengers"],
    "Kolkata Knight Riders": ["kolkata"],
    "Rajasthan Royals": ["rajasthan"],
    "Sunrisers Hyderabad": ["hyderabad", "sunrisers"],
    "Delhi Capitals": ["delhi"],
    "Punjab Kings": ["punjab"],
    "Lucknow Super Giants": ["lucknow"],
    "Gujarat Titans": ["gujarat"],
}
