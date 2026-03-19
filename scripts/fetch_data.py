#!/usr/bin/env python3
"""Fetch IPL 2015-2025 match data and player T20 career innings from ESPNCricinfo."""

import asyncio
import csv
import re
import sys
import time
from pathlib import Path

from cricdata import AsyncCricinfoClient

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MATCHES_DIR = DATA_DIR / "matches"
PLAYER_DIR = DATA_DIR / "player_innings"
MASTER_CSV = DATA_DIR / "master_matches.csv"

IPL_SERIES = {
    2015: "pepsi-indian-premier-league-2015-791129",
    2016: "indian-premier-league-2016-968923",
    2017: "indian-premier-league-2017-1078425",
    2018: "indian-premier-league-2018-1131611",
    2019: "indian-premier-league-2019-1165643",
    2020: "indian-premier-league-2020-21-1210595",
    2021: "indian-premier-league-2021-1249214",
    2022: "indian-premier-league-2022-1298423",
    2023: "indian-premier-league-2023-1345038",
    2024: "indian-premier-league-2024-1410320",
    2025: "ipl-2025-1449924",
}

CONCURRENCY = 3
MAX_RETRIES = 4


async def retry(coro_fn, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            return await coro_fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt + 1
            print(f"    Retry {attempt+1}/{retries} after {wait}s: {e}", file=sys.stderr)
            await asyncio.sleep(wait)


def team_name_by_id(teams, team_id):
    for t in teams:
        if t["team"]["id"] == team_id:
            return t["team"]["longName"]
    return ""


def build_master_csv(all_matches):
    rows = []
    for year, series_slug, m in all_matches:
        teams = m.get("teams", [])
        team1 = teams[0]["team"] if teams else {}
        team2 = teams[1]["team"] if len(teams) > 1 else {}
        ground = m.get("ground", {})

        toss_choice = m.get("tossWinnerChoice")
        toss_decision = {1: "bat", 2: "field"}.get(toss_choice, "")

        match_slug = f"{m['slug']}-{m['objectId']}"
        rows.append({
            "match_id": m["objectId"],
            "season": year,
            "date": (m.get("startDate") or "")[:10],
            "match_number": m.get("title", ""),
            "venue": ground.get("name", ""),
            "city": ground.get("town", {}).get("name", ""),
            "team1": team1.get("longName", ""),
            "team1_id": team1.get("id", ""),
            "team2": team2.get("longName", ""),
            "team2_id": team2.get("id", ""),
            "toss_winner": team_name_by_id(teams, m.get("tossWinnerTeamId")),
            "toss_decision": toss_decision,
            "winner": team_name_by_id(teams, m.get("winnerTeamId")),
            "result": m.get("statusText", ""),
            "series_slug": series_slug,
            "match_slug": match_slug,
        })

    rows.sort(key=lambda r: r["date"])
    with open(MASTER_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


BBB_FIELDS = [
    "match_id", "innings", "over", "ball", "batsman_id", "batsman",
    "bowler_id", "bowler", "runs", "play_type", "is_wicket",
    "wicket_type", "total_runs", "total_wickets", "short_text",
]


async def fetch_match_bbb(ci, sem, series_slug, match_slug, match_id):
    outpath = MATCHES_DIR / f"{match_id}.csv"
    if outpath.exists():
        return True

    async with sem:
        await asyncio.sleep(0.5)
        try:
            innings_list = await retry(
                lambda: ci.match_ball_by_ball(series_slug, match_slug)
            )
        except Exception as e:
            print(f"  WARN bbb {match_id}: {e}", file=sys.stderr)
            return False

    rows = []
    for innings_data in innings_list:
        for ball in innings_data:
            over = ball.get("over", {})
            inn = ball.get("innings", {})
            dis = ball.get("dismissal", {})
            batsman = dis.get("batsman", {}).get("athlete", {})
            bowler = dis.get("bowler", {}).get("athlete", {})

            rows.append({
                "match_id": match_id,
                "innings": inn.get("number", ""),
                "over": over.get("overs", ""),
                "ball": over.get("ball", ""),
                "batsman_id": batsman.get("id", ""),
                "batsman": batsman.get("displayName", ""),
                "bowler_id": bowler.get("id", ""),
                "bowler": bowler.get("displayName", ""),
                "runs": ball.get("scoreValue", 0),
                "play_type": ball.get("playType", {}).get("description", ""),
                "is_wicket": dis.get("dismissal", False),
                "wicket_type": dis.get("type", ""),
                "total_runs": inn.get("runs", ""),
                "total_wickets": inn.get("wickets", ""),
                "short_text": ball.get("shortText", ""),
            })

    if rows:
        with open(outpath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=BBB_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    return True


def collect_player_ids():
    players = {}
    for f in sorted(MATCHES_DIR.glob("*.csv")):
        with open(f) as fh:
            for row in csv.DictReader(fh):
                bid, bname = row.get("batsman_id", ""), row.get("batsman", "")
                if bid and bname:
                    players[bid] = bname
                wid, wname = row.get("bowler_id", ""), row.get("bowler", "")
                if wid and wname:
                    players[wid] = wname
    return players


PLAYER_FIELDS = [
    "player_id", "player_name", "start_date", "opposition", "ground",
    "bat_innings", "bat_position", "bat_runs", "bat_mins", "bat_bf",
    "bat_4s", "bat_6s", "bat_sr", "bat_dismissal",
    "bowl_innings", "bowl_overs", "bowl_maidens", "bowl_runs",
    "bowl_wickets", "bowl_economy",
]


def safe_filename(name, pid):
    return re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_") + f"_{pid}"


async def fetch_player_data(ci, sem, player_id, player_name):
    fname = safe_filename(player_name, player_id)
    outpath = PLAYER_DIR / f"{fname}.csv"
    if outpath.exists():
        return True

    async with sem:
        await asyncio.sleep(0.5)
        try:
            bat = await retry(
                lambda: ci.player_innings(player_id, fmt="t20", stat_type="batting")
            )
            await asyncio.sleep(0.3)
            bowl = await retry(
                lambda: ci.player_innings(player_id, fmt="t20", stat_type="bowling")
            )
        except Exception as e:
            print(f"  WARN player {player_name} ({player_id}): {e}", file=sys.stderr)
            return False

    bat_inn = bat.get("innings", [])
    bowl_inn = bowl.get("innings", [])

    rows = []
    for i, bi in enumerate(bat_inn):
        bw = bowl_inn[i] if i < len(bowl_inn) else {}
        rows.append({
            "player_id": player_id,
            "player_name": player_name,
            "start_date": bi.get("Start Date", ""),
            "opposition": bi.get("Opposition", ""),
            "ground": bi.get("Ground", ""),
            "bat_innings": bi.get("Inns", ""),
            "bat_position": bi.get("Pos", ""),
            "bat_runs": bi.get("Runs", ""),
            "bat_mins": bi.get("Mins", ""),
            "bat_bf": bi.get("BF", ""),
            "bat_4s": bi.get("4s", ""),
            "bat_6s": bi.get("6s", ""),
            "bat_sr": bi.get("SR", ""),
            "bat_dismissal": bi.get("Dismissal", ""),
            "bowl_innings": bw.get("Inns", ""),
            "bowl_overs": bw.get("Overs", ""),
            "bowl_maidens": bw.get("Mdns", ""),
            "bowl_runs": bw.get("Runs", ""),
            "bowl_wickets": bw.get("Wkts", ""),
            "bowl_economy": bw.get("Econ", ""),
        })

    if rows:
        with open(outpath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PLAYER_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    return True


async def run_batched(coro_fn, items, label, batch_size=10):
    total = len(items)
    done, failed = 0, 0
    for start in range(0, total, batch_size):
        batch = items[start : start + batch_size]
        results = await asyncio.gather(*[coro_fn(item) for item in batch])
        done += len(batch)
        failed += results.count(False)
        print(f"  {label}: {done}/{total} (errors: {failed})", flush=True)


async def main():
    MATCHES_DIR.mkdir(parents=True, exist_ok=True)
    PLAYER_DIR.mkdir(parents=True, exist_ok=True)

    async with AsyncCricinfoClient() as ci:
        sem = asyncio.Semaphore(CONCURRENCY)

        # --- Step 1: Fetch match lists ---
        print("Step 1/5: Fetching match lists for IPL 2015-2025...", flush=True)
        all_matches = []
        for year, slug in IPL_SERIES.items():
            result = await ci.series_matches(slug)
            matches = result["content"]["matches"]
            for m in matches:
                all_matches.append((year, slug, m))
            print(f"  {year}: {len(matches)} matches", flush=True)
        print(f"  Total: {len(all_matches)} matches", flush=True)

        # --- Step 2: Build master CSV ---
        print("Step 2/5: Building master_matches.csv...", flush=True)
        match_rows = build_master_csv(all_matches)
        print(f"  Wrote {len(match_rows)} rows to master_matches.csv", flush=True)

        # --- Step 3: Fetch ball-by-ball ---
        print("Step 3/5: Fetching ball-by-ball data for each match...", flush=True)
        await run_batched(
            lambda row: fetch_match_bbb(
                ci, sem, row["series_slug"], row["match_slug"], row["match_id"]
            ),
            match_rows,
            "Ball-by-ball",
        )

        # --- Step 4: Collect player IDs ---
        print("Step 4/5: Scanning matches for unique players...", flush=True)
        players = collect_player_ids()
        print(f"  Found {len(players)} unique players", flush=True)

        # --- Step 5: Fetch player innings ---
        print("Step 5/5: Fetching T20 career innings for each player...", flush=True)
        player_items = list(players.items())
        await run_batched(
            lambda item: fetch_player_data(ci, sem, item[0], item[1]),
            player_items,
            "Players",
        )

    print("\nDone!")
    print(f"  master_matches.csv: {MASTER_CSV}")
    print(f"  matches/: {sum(1 for _ in MATCHES_DIR.glob('*.csv'))} files")
    print(f"  player_innings/: {sum(1 for _ in PLAYER_DIR.glob('*.csv'))} files")


if __name__ == "__main__":
    t0 = time.time()
    asyncio.run(main())
    print(f"Elapsed: {time.time() - t0:.0f}s")
