"""Team name normalization, result parsing, and match filtering."""

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

TEAM_CANONICAL = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
}

TEAM_HOME_CITIES = {
    "Chennai Super Kings": {"Chennai"},
    "Mumbai Indians": {"Mumbai"},
    "Royal Challengers Bengaluru": {"Bengaluru", "Bangalore"},
    "Kolkata Knight Riders": {"Kolkata"},
    "Delhi Capitals": {"Delhi", "New Delhi"},
    "Punjab Kings": {"Chandigarh", "Mohali", "Mullanpur"},
    "Rajasthan Royals": {"Jaipur"},
    "Sunrisers Hyderabad": {"Hyderabad"},
    "Gujarat Titans": {"Ahmedabad"},
    "Lucknow Super Giants": {"Lucknow"},
    "Rising Pune Supergiants": {"Pune"},
    "Gujarat Lions": {"Rajkot", "Ahmedabad"},
}

# Opposition abbreviation → canonical team name (for player_innings mapping)
OPP_ABBREV = {
    "v CSK": "Chennai Super Kings",
    "v MI": "Mumbai Indians",
    "v RCB": "Royal Challengers Bengaluru",
    "v KKR": "Kolkata Knight Riders",
    "v DC": "Delhi Capitals",
    "v Daredevils": "Delhi Capitals",
    "v PBKS": "Punjab Kings",
    "v Kings XI": "Punjab Kings",
    "v RR": "Rajasthan Royals",
    "v SRH": "Sunrisers Hyderabad",
    "v GT": "Gujarat Titans",
    "v LSG": "Lucknow Super Giants",
    "v Supergiant": "Rising Pune Supergiants",
    "v Supergiants": "Rising Pune Supergiants",
    "v Guj Lions": "Gujarat Lions",
}


def normalize_team(name: str) -> str:
    return TEAM_CANONICAL.get(name, name)


def is_home(team: str, city: str) -> bool:
    cities = TEAM_HOME_CITIES.get(team, set())
    return city in cities


_MARGIN_RE = re.compile(
    r"won by (\d+) (run|wicket)s?\b"
)
_TIED_RE = re.compile(
    r"Match tied \((\w[\w\s]*?) won"
)


def parse_result(result_text: str) -> dict:
    """Parse the result text into structured fields.

    Returns dict with keys:
        valid: bool  (False for no-result / abandoned)
        win_by_runs: int or None
        win_by_wickets: int or None
        is_super_over: bool
        is_dls: bool
    """
    if not result_text or "no result" in result_text.lower() or "abandoned" in result_text.lower():
        return {"valid": False, "win_by_runs": None, "win_by_wickets": None,
                "is_super_over": False, "is_dls": False}

    is_dls = bool(re.search(r"D/?L|DLS", result_text))
    is_super = "tied" in result_text.lower() or "super over" in result_text.lower()

    m = _MARGIN_RE.search(result_text)
    runs = wkts = None
    if m:
        val = int(m.group(1))
        if m.group(2) == "run":
            runs = val
        else:
            wkts = val
    elif is_super:
        runs = 0
        wkts = 0

    return {"valid": True, "win_by_runs": runs, "win_by_wickets": wkts,
            "is_super_over": is_super, "is_dls": is_dls}


def load_matches() -> list[dict]:
    """Load master_matches.csv, normalize teams, parse results, filter invalid."""
    rows = []
    with open(DATA_DIR / "master_matches.csv") as f:
        for row in csv.DictReader(f):
            row["team1"] = normalize_team(row["team1"])
            row["team2"] = normalize_team(row["team2"])
            row["toss_winner"] = normalize_team(row["toss_winner"])
            row["winner"] = normalize_team(row["winner"])

            parsed = parse_result(row["result"])
            if not parsed["valid"]:
                continue

            row["win_by_runs"] = parsed["win_by_runs"]
            row["win_by_wickets"] = parsed["win_by_wickets"]
            row["is_super_over"] = parsed["is_super_over"]
            row["is_dls"] = parsed["is_dls"]

            row["label"] = 1 if row["winner"] == row["team1"] else 0
            row["season"] = int(row["season"])
            rows.append(row)
    return rows


if __name__ == "__main__":
    matches = load_matches()
    print(f"Valid matches: {len(matches)}")
    seasons = {}
    for m in matches:
        seasons[m["season"]] = seasons.get(m["season"], 0) + 1
    for s in sorted(seasons):
        print(f"  {s}: {seasons[s]}")
