"""Simulate live-inference accuracy on 2026 using announced XIs.

For every 2026 match we:
  1. Ensure announced XI is cached (re-scrape missing entries).
  2. Rebuild features using the announced XI instead of the BBB-derived XI
     (mirrors what `predictor/predict.py` would do pre-match).
  3. Walk-forward: at match i, train the stacked LR on dataset.csv rows
     with season < 2026 plus the 2026 matches already played (BBB).
  4. Predict the i-th 2026 match twice — once with the BBB feature vector
     from dataset.csv (our 71.4% backtest), once with the announced-XI
     feature vector — and record both correctness bits.

Reports per-source accuracy on the subset of 2026 matches where announced
XI data is available.
"""

import asyncio
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from cricdata import AsyncCricinfoClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.config import TEAM_KEYWORDS
from predictor.features import build_match_features, load_bios
from predictor.normalize import load_matches, normalize_team
from predictor.playing_xi import extract_all_xis
from predictor.train import (
    _all_feature_sets,
    COVID_SEASONS,
    HALF_LIFE,
    IMPACT_ERA_BOOST,
    IMPACT_PLAYER_START,
    LR_C,
    META_C,
    MIN_TRAIN_SEASON,
)

DATASET = ROOT / "data" / "dataset.csv"
MASTER = ROOT / "data" / "master_matches.csv"
ANNOUNCED = ROOT / "data" / "announced_xis.csv"
IMPACT_CACHE = ROOT / "scripts" / "impact_subs_2026.json"
OPTIONS_CACHE = ROOT / "scripts" / "impact_sub_options_2026.json"
SEASON = 2026
CONCURRENCY = 20
META_COLS = {
    "match_id", "date", "season", "team1", "team2", "label",
    "winner", "venue", "city", "toss_winner", "toss_decision",
}


# ---------- announced-XI backfill ----------

async def _fetch_one(ci, sem, mid, series_slug, match_slug):
    async with sem:
        try:
            sc = await ci.match_scorecard(series_slug, match_slug)
        except Exception as e:
            return mid, None, None, None, str(e)
    if not sc:
        return mid, None, None, None, "empty"
    content = sc.get("content", {})
    players = []
    for tp in content.get("matchPlayers", {}).get("teamPlayers", []):
        if tp.get("type") != "PLAYING":
            continue
        tid = tp.get("team", {}).get("id")
        for p in tp.get("players", []):
            player = p.get("player", {})
            pid = str(player.get("id", ""))
            name = player.get("longName") or player.get("name", "")
            if pid and name:
                players.append((mid, tid, pid, name))
    impact_used = _parse_impact_subs_used(content)
    impact_opts = _parse_impact_sub_options(content)
    if len(players) < 11:
        return mid, None, impact_used, impact_opts, f"only {len(players)} players"
    return mid, players, impact_used, impact_opts, None


def _parse_impact_subs_used(content):
    """Return list of names who came ON as impact sub in this match.

    Parses notes like:
      "<Team> Impact Player Sub: V Sooryavanshi for Sandeep Sharma ..."
      "<Team> Impact Player Sub: Devdutt Padikkal in for Kartik Sharma ..."
    Strips trailing " in" / " out" markers Cricinfo sometimes appends.
    """
    names = []
    notes = content.get("notes", {})
    if not isinstance(notes, dict):
        return names
    for g in notes.get("groups", []):
        for note in g.get("notes", []):
            if not isinstance(note, str):
                continue
            low = note.lower()
            if "impact player sub:" not in low or "sub options" in low:
                continue
            tail = note.split(":", 1)[-1].strip()
            if " for " not in tail.lower():
                continue
            on_name = tail.split(" for ", 1)[0].strip()
            for suffix in (" in", " out"):
                if on_name.lower().endswith(suffix):
                    on_name = on_name[:-len(suffix)].strip()
            if on_name:
                names.append(on_name)
    return names


def _parse_impact_sub_options(content):
    """Return {canonical_team: [name1,...,name5]} of pre-match bench options.

    Parses notes like:
      "<Team> Impact Player Subs: A, B, C, D and E"
    (note plural 'Subs' – singular 'Sub:' announces the used one).
    """
    out = {}
    notes = content.get("notes", {})
    if not isinstance(notes, dict):
        return out
    for g in notes.get("groups", []):
        for note in g.get("notes", []):
            if not isinstance(note, str):
                continue
            low = note.lower()
            if "impact player subs:" not in low:
                continue
            prefix, tail = note.split(":", 1)
            prefix_low = prefix.lower()
            team = None
            for canonical, kws in TEAM_KEYWORDS.items():
                if any(kw in prefix_low for kw in kws):
                    team = canonical
                    break
            if not team:
                continue
            raw = tail.replace(" and ", ",").split(",")
            names = [n.strip() for n in raw if n.strip()]
            if names:
                out[team] = names
    return out


async def backfill_all():
    """Ensure announced_xis.csv + impact-sub caches are populated for 2026.

    Caches:
      impact_subs_2026.json         - {mid: [names who came on]}
      impact_sub_options_2026.json  - {mid: {team: [5 bench option names]}}
    """
    master = pd.read_csv(MASTER)
    master["season"] = master["season"].astype(int)
    season_matches = master[master["season"] == SEASON]
    have = set()
    if ANNOUNCED.exists():
        with open(ANNOUNCED) as f:
            rd = csv.DictReader(f)
            for row in rd:
                have.add(str(row["match_id"]))
    impact_cache = {}
    if IMPACT_CACHE.exists():
        impact_cache = json.loads(IMPACT_CACHE.read_text())
    options_cache = {}
    if OPTIONS_CACHE.exists():
        options_cache = json.loads(OPTIONS_CACHE.read_text())

    to_fetch = [r for _, r in season_matches.iterrows()
                if str(r["match_id"]) not in have
                or str(r["match_id"]) not in impact_cache
                or str(r["match_id"]) not in options_cache]
    print(f"2026 matches: {len(season_matches)}, "
          f"need to fetch (missing xi/impact/options): {len(to_fetch)}",
          flush=True)
    if not to_fetch:
        return impact_cache, options_cache

    sem = asyncio.Semaphore(CONCURRENCY)
    async with AsyncCricinfoClient() as ci:
        tasks = [
            _fetch_one(ci, sem, str(r["match_id"]),
                        r["series_slug"], r["match_slug"])
            for r in to_fetch
        ]
        results = await asyncio.gather(*tasks)
    new_rows = []
    fail = []
    for mid, players, impact_used, impact_opts, err in results:
        if impact_used is not None:
            impact_cache[mid] = impact_used
        if impact_opts is not None:
            options_cache[mid] = impact_opts
        if players:
            new_rows.extend(players)
        else:
            fail.append((mid, err))
    if new_rows:
        with open(ANNOUNCED, "a", newline="") as f:
            w = csv.writer(f)
            for row in new_rows:
                w.writerow(row)
        print(f"  appended {len(new_rows)} rows from "
              f"{len({r[0] for r in new_rows})} matches", flush=True)
    if fail:
        print(f"  failed: {len(fail)} matches (likely future/unplayed)",
              flush=True)
    IMPACT_CACHE.write_text(json.dumps(impact_cache, indent=2))
    OPTIONS_CACHE.write_text(json.dumps(options_cache, indent=2))
    print(f"  impact-sub-used cache: {len(impact_cache)} matches", flush=True)
    print(f"  impact-sub-options cache: {len(options_cache)} matches",
          flush=True)
    return impact_cache, options_cache


# ---------- ID mapping & XI rebuild ----------

def build_name_to_bbb_id(all_xis):
    m = {}
    for xi_data in all_xis.values():
        for xi in (xi_data["team1_xi"], xi_data["team2_xi"]):
            for pid, pname in xi.items():
                m[pname.lower().strip()] = pid
    return m


def load_announced_xis_by_match():
    by_match = {}
    with open(ANNOUNCED) as f:
        for row in csv.DictReader(f):
            by_match.setdefault(row["match_id"], []).append(
                (row["team_id"], row["player_id"], row["player_name"]))
    return by_match


def _canonical_team_name(team_longname):
    if not team_longname:
        return None
    return normalize_team(team_longname)


def _norm_name(s):
    return " ".join(s.lower().strip().replace(".", "").split())


def _role_rank(role: str, prefer_bowler: bool) -> int:
    """Lower == better match for the desired impact-sub archetype."""
    r = (role or "").lower()
    is_bowler = "bowl" in r and "round" not in r
    is_bat_ar = ("bat" in r) and ("round" in r)
    is_bowl_ar = ("bowl" in r) and ("round" in r)
    is_generic_ar = "allround" in r and not (is_bat_ar or is_bowl_ar)
    is_batter = (
        not is_bowler and not is_generic_ar and not is_bat_ar and not is_bowl_ar
        and any(k in r for k in ("bat", "keeper", "opening", "middle", "top-order"))
    )
    if prefer_bowler:
        if is_bowler: return 0
        if is_bowl_ar: return 1
        if is_generic_ar: return 2
        if is_bat_ar: return 3
        if is_batter: return 4
    else:
        if is_batter: return 0
        if is_bat_ar: return 1
        if is_generic_ar: return 2
        if is_bowl_ar: return 3
        if is_bowler: return 4
    return 5


def pick_predicted_sub(candidate_names, bios, name_to_bbb_id,
                        prefer_bowler, exclude_ids=None):
    """Choose the most likely impact-sub from a pre-match 5-option bench.

    Returns (bbb_id, name) or (None, None).
    """
    exclude = exclude_ids or set()
    scored = []
    for nm in candidate_names or []:
        pid = name_to_bbb_id.get(_norm_name(nm))
        if not pid or pid in exclude:
            continue
        role = bios.get(pid, {}).get("role", "")
        scored.append((_role_rank(role, prefer_bowler), nm, pid))
    if not scored:
        return None, None
    scored.sort(key=lambda x: (x[0], x[1]))
    _, nm, pid = scored[0]
    return pid, nm


def _drop_impact_sub(players, impact_names):
    """Remove announced players whose name matches an impact-sub-used name.

    Cricinfo notes sometimes shorten names (e.g. "V Sooryavanshi" vs
    "Vaibhav Sooryavanshi"). Use initial+surname matching as fallback.
    """
    if not impact_names:
        return players
    drop_keys = set()
    for nm in impact_names:
        tok = _norm_name(nm).split()
        if not tok:
            continue
        surname = tok[-1]
        initial = tok[0][0] if tok[0] else ""
        drop_keys.add((initial, surname))
    kept = []
    for tid, pid, name in players:
        ntok = _norm_name(name).split()
        if not ntok:
            kept.append((tid, pid, name))
            continue
        k = (ntok[0][0] if ntok[0] else "", ntok[-1])
        if k in drop_keys:
            continue
        kept.append((tid, pid, name))
    return kept


def build_announced_xi_for_match(match_row, announced_players,
                                   name_to_bbb_id, bbb_xi,
                                   impact_names_used=None):
    """Return {"team1_xi": {bbb_id: name}, "team2_xi": {bbb_id: name}}.

    announced_players: list of (team_id, scorecard_player_id, name) rows.
    name_to_bbb_id: lowercased name -> Statsguru id from BBB data.
    bbb_xi: {"team1_xi": {pid: name}, "team2_xi": {pid: name}} ground truth
        from BBB data — used to disambiguate which Cricinfo team_id maps to
        which canonical team by player-overlap voting.
    impact_names_used: if given, names of players to strip from the XI (the
        impact sub who ultimately came on) so the XI reflects what a
        pre-match live scrape would see.
    """
    if impact_names_used:
        announced_players = _drop_impact_sub(
            list(announced_players), impact_names_used)

    teams_seen = []
    for tid, _, _ in announced_players:
        if tid not in teams_seen:
            teams_seen.append(tid)
    if len(teams_seen) != 2:
        return None

    bbb_t1_ids = set(bbb_xi["team1_xi"].keys())
    bbb_t2_ids = set(bbb_xi["team2_xi"].keys())

    tid_scores = {}
    for tid in teams_seen:
        ann_ids = set()
        for t, _, name in announced_players:
            if t != tid:
                continue
            pid = name_to_bbb_id.get(name.lower().strip())
            if pid:
                ann_ids.add(pid)
        tid_scores[tid] = (
            len(ann_ids & bbb_t1_ids),
            len(ann_ids & bbb_t2_ids),
        )

    tid_a, tid_b = teams_seen
    a_t1_vote = tid_scores[tid_a][0] + tid_scores[tid_b][1]
    a_t2_vote = tid_scores[tid_a][1] + tid_scores[tid_b][0]
    if a_t1_vote >= a_t2_vote:
        tid_to_team = {tid_a: "team1", tid_b: "team2"}
    else:
        tid_to_team = {tid_a: "team2", tid_b: "team1"}

    t1_xi, t2_xi = {}, {}
    unresolved = []
    for tid, sc_pid, name in announced_players:
        pid = name_to_bbb_id.get(name.lower().strip())
        if pid is None:
            unresolved.append(name)
            continue
        if tid_to_team[tid] == "team1":
            t1_xi[pid] = name
        else:
            t2_xi[pid] = name
    return {"team1_xi": t1_xi, "team2_xi": t2_xi, "unresolved": unresolved}


# ---------- training / prediction utilities ----------

def _weights(seasons, target_year, half_life=None):
    decay = np.log(2) / (half_life or HALF_LIFE)
    w = np.exp(-decay * (target_year - 1 - seasons))
    w[seasons >= IMPACT_PLAYER_START] *= IMPACT_ERA_BOOST
    return w


def _train_and_predict_one(train_df, feat_vec, feat_sets, half_life=None):
    """Train stacked LR on train_df, predict P(t1 wins) for a single feature dict."""
    y = train_df["label"].values
    max_yr = int(train_df["season"].max())
    w = _weights(train_df["season"].values, max_yr + 1, half_life)

    seasons = sorted(train_df["season"].unique())
    oof = np.zeros((len(train_df), len(feat_sets)))
    for mi, (_, feats) in enumerate(feat_sets):
        for vs in seasons:
            tr = train_df["season"] != vs
            va = train_df["season"] == vs
            if tr.sum() == 0 or va.sum() == 0:
                continue
            w_tr = _weights(
                train_df.loc[train_df.index[tr], "season"].values, vs, half_life)
            pipe = Pipeline([
                ("s", StandardScaler()),
                ("lr", LogisticRegression(C=LR_C, max_iter=5000)),
            ])
            pipe.fit(
                train_df.loc[train_df.index[tr], feats].fillna(0).values,
                y[tr], lr__sample_weight=w_tr)
            oof[va, mi] = pipe.predict_proba(
                train_df.loc[train_df.index[va], feats].fillna(0).values
            )[:, 1]

    preds = []
    for mi, (_, feats) in enumerate(feat_sets):
        pipe = Pipeline([
            ("s", StandardScaler()),
            ("lr", LogisticRegression(C=LR_C, max_iter=5000)),
        ])
        pipe.fit(train_df[feats].fillna(0).values, y, lr__sample_weight=w)
        X = np.array([[feat_vec.get(c, 0.0) for c in feats]])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        preds.append(float(pipe.predict_proba(X)[0, 1]))

    meta = Pipeline([
        ("s", StandardScaler()),
        ("lr", LogisticRegression(C=META_C, max_iter=5000)),
    ])
    meta.fit(oof, y, lr__sample_weight=w)
    return float(meta.predict_proba(np.array([preds]))[0, 1])


# ---------- main simulation ----------

def main():
    impact_cache, options_cache = asyncio.run(backfill_all())

    print("\nLoading data…", flush=True)
    df = pd.read_csv(DATASET)
    df["season"] = df["season"].astype(int)
    for c in df.columns:
        if c in META_COLS:
            continue
        df[c] = df[c].fillna(0).replace([np.inf, -np.inf], 0)

    matches = load_matches()
    all_xis = extract_all_xis(matches)
    name_to_bbb = build_name_to_bbb_id(all_xis)
    bios = load_bios()
    print(f"  name_to_bbb map: {len(name_to_bbb)} entries", flush=True)
    print(f"  bios: {len(bios)} players", flush=True)

    announced = load_announced_xis_by_match()

    m26 = (df[df["season"] == SEASON]
           .sort_values("date")
           .reset_index(drop=True))
    match_meta = {m["match_id"]: m for m in matches}
    print(f"  2026 dataset rows: {len(m26)}", flush=True)

    feat_sets = _all_feature_sets()
    pre = df[(df["season"] >= MIN_TRAIN_SEASON)
             & (df["season"] < SEASON)
             & ~df["season"].isin(COVID_SEASONS)]

    rows = []
    for i, row in m26.iterrows():
        mid = str(row["match_id"])
        ann = announced.get(mid)
        if not ann:
            rows.append({"match_id": mid, "reason": "no_announced_xi"})
            continue
        meta = match_meta.get(mid)
        if not meta:
            rows.append({"match_id": mid, "reason": "no_match_meta"})
            continue
        bbb_xi = all_xis.get(mid)
        if not bbb_xi:
            rows.append({"match_id": mid, "reason": "no_bbb_xi"})
            continue

        xi_ann12 = build_announced_xi_for_match(
            row, ann, name_to_bbb, bbb_xi, impact_names_used=None)
        impact_used = impact_cache.get(mid, [])
        xi_ann11 = build_announced_xi_for_match(
            row, ann, name_to_bbb, bbb_xi, impact_names_used=impact_used)

        if xi_ann12 is None or not xi_ann12["team1_xi"] or not xi_ann12["team2_xi"]:
            rows.append({"match_id": mid, "reason": "xi_rebuild_failed"})
            continue

        # --- 11 + predicted / ensembled sub (pre-match live inference) ---
        tw = normalize_team(str(meta.get("toss_winner", "")))
        tdec = str(meta.get("toss_decision", "")).lower()
        t1n, t2n = meta["team1"], meta["team2"]
        if tdec.startswith("bat"):
            bat_first = tw
        else:
            bat_first = t2n if tw == t1n else t1n
        t1_bat_first = (bat_first == t1n)

        opts = options_cache.get(mid, {}) or {}
        t1_opts = opts.get(t1n, [])
        t2_opts = opts.get(t2n, [])

        # Best-guess pick (role-matched).
        t1_sub_pid, t1_sub_nm = pick_predicted_sub(
            t1_opts, bios, name_to_bbb,
            prefer_bowler=t1_bat_first,
            exclude_ids=set(xi_ann11["team1_xi"].keys()),
        )
        t2_sub_pid, t2_sub_nm = pick_predicted_sub(
            t2_opts, bios, name_to_bbb,
            prefer_bowler=not t1_bat_first,
            exclude_ids=set(xi_ann11["team2_xi"].keys()),
        )

        # Build train frame including earlier 2026 matches (with BBB features)
        earlier_26 = df[(df["season"] == SEASON) & (df["date"] < row["date"])]
        train_df = pd.concat([pre, earlier_26], ignore_index=True)

        # Prior matches for feature builders (only matches BEFORE this one)
        prior_m = [m for m in matches if m.get("date", "") < row["date"]
                   and m.get("season", 0) >= 2008]

        all_xis_12 = dict(all_xis); all_xis_12[mid] = xi_ann12
        all_xis_11 = dict(all_xis); all_xis_11[mid] = xi_ann11

        ann12_feats = build_match_features(meta, prior_m, all_xis_12)
        ann11_feats = build_match_features(meta, prior_m, all_xis_11)
        bbb_feat_vec = {c: row[c] for c in df.columns if c not in META_COLS}

        # Single-pick prediction.
        xi_ann11_pred = {
            "team1_xi": dict(xi_ann11["team1_xi"]),
            "team2_xi": dict(xi_ann11["team2_xi"]),
        }
        if t1_sub_pid:
            xi_ann11_pred["team1_xi"][t1_sub_pid] = t1_sub_nm
        if t2_sub_pid:
            xi_ann11_pred["team2_xi"][t2_sub_pid] = t2_sub_nm
        all_xis_pred = dict(all_xis); all_xis_pred[mid] = xi_ann11_pred
        pred_feats = build_match_features(meta, prior_m, all_xis_pred)

        # Role-averaged ensemble: for each team, build one XI per bench
        # candidate whose role matches the archetype and average P across them.
        # Falls back to the 11-starter XI when no candidates resolve.
        def _role_matched_pids(opts_names, prefer_bowler, exclude_ids):
            picks = []
            for nm in opts_names or []:
                pid = name_to_bbb.get(_norm_name(nm))
                if not pid or pid in exclude_ids:
                    continue
                rank = _role_rank(bios.get(pid, {}).get("role", ""),
                                   prefer_bowler)
                if rank <= 2:  # bowler/bowling-ar/allrounder OR batter/batting-ar/allrounder
                    picks.append((pid, nm))
            return picks

        t1_candidates = _role_matched_pids(
            t1_opts, t1_bat_first,
            set(xi_ann11["team1_xi"].keys())) or [(None, None)]
        t2_candidates = _role_matched_pids(
            t2_opts, not t1_bat_first,
            set(xi_ann11["team2_xi"].keys())) or [(None, None)]

        ens_probs = []
        for t1p, t1nm in t1_candidates:
            for t2p, t2nm in t2_candidates:
                xi_var = {
                    "team1_xi": dict(xi_ann11["team1_xi"]),
                    "team2_xi": dict(xi_ann11["team2_xi"]),
                }
                if t1p:
                    xi_var["team1_xi"][t1p] = t1nm
                if t2p:
                    xi_var["team2_xi"][t2p] = t2nm
                all_xis_var = dict(all_xis); all_xis_var[mid] = xi_var
                var_feats = build_match_features(meta, prior_m, all_xis_var)
                ens_probs.append(
                    _train_and_predict_one(train_df, var_feats, feat_sets))

        p_ens = float(np.mean(ens_probs)) if ens_probs else 0.5

        p_bbb = _train_and_predict_one(train_df, bbb_feat_vec, feat_sets)
        p_ann12 = _train_and_predict_one(train_df, ann12_feats, feat_sets)
        p_ann11 = _train_and_predict_one(train_df, ann11_feats, feat_sets)
        p_pred = _train_and_predict_one(train_df, pred_feats, feat_sets)
        label = int(row["label"])
        r = {
            "match_id": mid,
            "date": row["date"],
            "team1": row["team1"],
            "team2": row["team2"],
            "label": label,
            "bat_first": bat_first,
            "p_bbb": p_bbb,
            "p_ann12": p_ann12,
            "p_ann11": p_ann11,
            "p_pred": p_pred,
            "p_ens": p_ens,
            "bbb_correct": int((p_bbb > 0.5) == bool(label)),
            "ann12_correct": int((p_ann12 > 0.5) == bool(label)),
            "ann11_correct": int((p_ann11 > 0.5) == bool(label)),
            "pred_correct": int((p_pred > 0.5) == bool(label)),
            "ens_correct": int((p_ens > 0.5) == bool(label)),
            "n_ens": len(ens_probs),
            "impact_used": impact_used,
            "picked_subs": {
                t1n: t1_sub_nm,
                t2n: t2_sub_nm,
            },
        }
        rows.append(r)
        pick_str = (f"{(t1_sub_nm or '-')[:14]}|"
                    f"{(t2_sub_nm or '-')[:14]}")
        print(f"  [{i+1}/{len(m26)}] {row['date']} "
              f"{row['team1'][:18]:>18} v {row['team2'][:18]:<18}  "
              f"BBB={p_bbb:.3f}{'Y' if r['bbb_correct'] else 'N'}  "
              f"A11={p_ann11:.3f}{'Y' if r['ann11_correct'] else 'N'}  "
              f"A11+pred={p_pred:.3f}{'Y' if r['pred_correct'] else 'N'}  "
              f"ens{len(ens_probs)}={p_ens:.3f}{'Y' if r['ens_correct'] else 'N'}  "
              f"picks={pick_str}",
              flush=True)

    graded = [r for r in rows if "bbb_correct" in r]
    if graded:
        n = len(graded)
        bbb = sum(r["bbb_correct"] for r in graded)
        a12 = sum(r["ann12_correct"] for r in graded)
        a11 = sum(r["ann11_correct"] for r in graded)
        pred = sum(r["pred_correct"] for r in graded)
        ens = sum(r["ens_correct"] for r in graded)
        print(f"\n{'='*70}")
        print(f"2026 matches graded: {n}")
        print(f"  BBB post-hoc (12 actuals, dataset.csv):        "
              f"{bbb/n:.1%} ({bbb}/{n})")
        print(f"  Announced XI, 12 players (post-scrape):        "
              f"{a12/n:.1%} ({a12}/{n})")
        print(f"  Announced XI, 11 starters (baseline live):     "
              f"{a11/n:.1%} ({a11}/{n})")
        print(f"  Announced 11 + single role-matched sub:        "
              f"{pred/n:.1%} ({pred}/{n})")
        print(f"  Announced 11 + role-averaged bench ensemble:   "
              f"{ens/n:.1%} ({ens}/{n})")
        print(f"  Delta ens vs A11: {(ens-a11)/n:+.1%}    "
              f"Delta ens vs BBB: {(ens-bbb)/n:+.1%}")
        print(f"{'='*70}")
    skipped = [r for r in rows if "bbb_correct" not in r]
    if skipped:
        print(f"Skipped: {len(skipped)} (reasons: "
              f"{ {r['reason'] for r in skipped} })")

    out = ROOT / "scripts" / "live_inference_results.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
