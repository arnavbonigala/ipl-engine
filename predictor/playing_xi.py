"""Extract playing XIs per match from ball-by-ball data."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATCHES_DIR = ROOT / "data" / "matches"


def extract_xi(match_id: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return (team_batting_first_players, team_batting_second_players).

    Each is {player_id: player_name}.
    Team A = batsmen in innings 1 + bowlers in innings 2.
    Team B = batsmen in innings 2 + bowlers in innings 1.
    """
    path = MATCHES_DIR / f"{match_id}.csv"
    team_a, team_b = {}, {}

    with open(path) as f:
        for row in csv.DictReader(f):
            inn = row["innings"]
            bid, bname = row["batsman_id"], row["batsman"]
            wid, wname = row["bowler_id"], row["bowler"]

            if inn == "1":
                if bid:
                    team_a[bid] = bname
                if wid:
                    team_b[wid] = wname
            elif inn == "2":
                if bid:
                    team_b[bid] = bname
                if wid:
                    team_a[wid] = wname

    return team_a, team_b


def extract_all_xis(match_rows: list[dict]) -> dict[str, dict]:
    """For each match, extract XIs and map to team1/team2.

    Returns {match_id: {"team1_xi": {pid: name}, "team2_xi": {pid: name}}}.

    The BBB innings 1 batting team might be team1 or team2 depending on
    toss. We determine this by checking which team batted first from the
    result text and toss info.
    """
    result = {}
    for m in match_rows:
        mid = m["match_id"]
        path = MATCHES_DIR / f"{mid}.csv"
        if not path.exists():
            continue

        team_bat_first, team_bat_second = extract_xi(mid)

        # Determine which team (team1 or team2) batted first.
        # If toss winner chose to bat → toss_winner batted first.
        # If toss winner chose to field → the other team batted first.
        toss_winner = m["toss_winner"]
        toss_decision = m["toss_decision"]

        if toss_decision == "bat":
            bat_first_team = toss_winner
        else:
            bat_first_team = m["team1"] if toss_winner == m["team2"] else m["team2"]

        if bat_first_team == m["team1"]:
            t1_xi, t2_xi = team_bat_first, team_bat_second
        else:
            t1_xi, t2_xi = team_bat_second, team_bat_first

        result[mid] = {"team1_xi": t1_xi, "team2_xi": t2_xi}

    return result


if __name__ == "__main__":
    from predictor.normalize import load_matches

    matches = load_matches()
    xis = extract_all_xis(matches)
    sizes = [len(v["team1_xi"]) + len(v["team2_xi"]) for v in xis.values()]
    print(f"Extracted XIs for {len(xis)} matches")
    print(f"Players per match (avg): {sum(sizes) / len(sizes):.1f}")

    sample_mid = matches[0]["match_id"]
    if sample_mid in xis:
        xi = xis[sample_mid]
        print(f"\nSample match {sample_mid} ({matches[0]['team1']} vs {matches[0]['team2']}):")
        print(f"  Team1 XI ({len(xi['team1_xi'])}): {', '.join(xi['team1_xi'].values())}")
        print(f"  Team2 XI ({len(xi['team2_xi'])}): {', '.join(xi['team2_xi'].values())}")
