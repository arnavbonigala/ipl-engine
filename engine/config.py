"""Strategy constants and API endpoints."""

import os

KELLY_FRACTION = 0.5
MAX_BET_FRACTION = 0.25
MAX_ENTRY_PRICE = 0.75
USE_STOP_LOSS = False
STOP_LOSS_CENTS = 0.15
MIN_CONFIDENCE = 0.60
MIN_EDGE = 0.03
POLL_INTERVAL_SECS = 30
SCRAPE_POLL_SECS = 60

KALSHI_API_BASE_PROD = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_API_BASE_DEMO = "https://demo-api.kalshi.co/trade-api/v2"
KALSHI_API_BASE = KALSHI_API_BASE_DEMO if os.environ.get("KALSHI_DEMO") == "1" else KALSHI_API_BASE_PROD

KALSHI_IPL_SERIES = "KXIPLGAME"

STATE_FILE = "engine_state.json"
DASHBOARD_PORT = 8050

TEAM_KALSHI_ABBREVS = {
    "Mumbai Indians": "MI",
    "Chennai Super Kings": "CSK",
    "Royal Challengers Bengaluru": "RCB",
    "Kolkata Knight Riders": "KKR",
    "Rajasthan Royals": "RR",
    "Sunrisers Hyderabad": "SRH",
    "Delhi Capitals": "DC",
    "Punjab Kings": "PBKS",
    "Lucknow Super Giants": "LSG",
    "Gujarat Titans": "GT",
}

KALSHI_ABBREV_TO_TEAM = {v: k for k, v in TEAM_KALSHI_ABBREVS.items()}

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
