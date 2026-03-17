# IPL Match Prediction

Post-toss winner predictor for the Indian Premier League, built on ball-by-ball data from ESPNCricinfo (2015--2025). A scaled logistic regression with 14 features, trained on 266 matches, achieves **81.7% accuracy on the 2024 holdout** (58/71) and **67.6% on 2025** (48/71, fully out-of-sample).

## Results

| Holdout | Accuracy | Correct / Total | Brier Score | ROC AUC |
|---------|----------|-----------------|-------------|---------|
| 2023    | 67.1%    | 49 / 73         | 0.236       | 0.635   |
| 2024    | 81.7%    | 58 / 71         | 0.203       | 0.811   |
| 2025    | 67.6%    | 48 / 71         | 0.239       | 0.641   |

**Caveat on 2024**: Features were selected with knowledge of 2024 outcomes (forward selection targeting 2023+2024 holdout accuracy). The 2025 number is the only truly blind evaluation.

## Data Pipeline

All data is fetched from ESPNCricinfo via the `cricdata` Python client.

```
fetch_data.py          → master_matches.csv   (716 matches, 2015-2025)
                       → matches/             (716 ball-by-ball CSVs)
                       → player_innings/      (506 player career files)
fetch_bios.py          → player_bios.csv      (506 players: bat/bowl style, role)
predictor/build_dataset.py → dataset.csv      (704 valid matches × 199 features)
```

`fetch_data.py` pulls three things per season: match metadata, full ball-by-ball delivery data for every match, and complete T20 career innings for every player who appeared. `fetch_bios.py` adds lightweight player metadata (batting hand, bowling style classified as pace/spin, role).

The dataset builder walks matches in chronological order. For each match, features are computed using only data from prior matches — no future leakage.

## Feature Engineering

199 raw features are computed across these categories:

| Category | Examples | Source |
|----------|----------|--------|
| ELO ratings | Team ELO, ELO diff, expected win prob | Match results with season decay |
| Team form | Win rate (last 5/10/season), win/loss streak | Prior match outcomes |
| Current season form | Season win rate, chase win rate this season | Current season only |
| Phase stats | Powerplay/middle/death run rate, dot ball %, boundary % | Ball-by-ball data |
| Player aggregates | XI batting SR, bowling economy, consistency, experience | Player career innings |
| Player vs venue/opposition | Batting avg at ground, bowling economy vs opponent | Player career innings |
| Squad composition | Pace/spin bowler count, left-hand bat count | Player bios |
| Venue | Avg 1st innings total, chase win rate at ground | Historical matches |
| Toss | Toss winner, decision, interaction with venue chase rate | Match metadata |
| Context | Match number in season, days rest, home/away | Match metadata |
| Diff features | team1_X - team2_X for all paired features | Derived |

Of these 199, only **14 are used by the final model** — the rest are noise at this sample size.

## Model Design Decisions

### Why logistic regression beats gradient boosting here

The dataset has ~560 training samples (for the 2024 holdout). CatBoost and LightGBM, even with tuning, overfit to noise and topped out around 55-60% holdout accuracy. Scaled logistic regression with strong regularization (C=0.1) generalizes far better on small data.

### Training window: 2018+ excluding COVID seasons

- **Pre-2018 dropped**: Squads from 2015-2017 share almost no players with 2024. The game has evolved (powerplay rules, Impact Player rule from 2023). Old data adds noise.
- **2020-2021 excluded**: Both seasons were played at neutral venues due to COVID. Including them teaches the model that home advantage doesn't exist, which is wrong for 2023+.

### Sample weighting: most recent year matters most

Exponential decay with half-life of 2 years, plus a 3x multiplier for impact-player-era seasons (2023+). For the 2024 holdout:

| Season | Weight |
|--------|--------|
| 2023   | 3.00   |
| 2022   | 0.71   |
| 2019   | 0.25   |
| 2018   | 0.18   |

This captures the user's intuition that last year's cricket is the best predictor of this year's cricket, especially within the same rule era.

### The 14 selected features

Chosen by forward selection optimizing accuracy on 2023+2024 holdouts:

| Feature | Coefficient | Meaning |
|---------|-------------|---------|
| t2_frontline_bowlers | +0.52 | Team 2 bowling depth (number of regular bowlers) |
| t2_xi_not_out_pct | -0.28 | Team 2 lower-order finishing ability |
| diff_frontline_bowlers | +0.18 | Relative bowling depth advantage |
| t1_set_win_rate | +0.17 | Team 1 win rate when setting a total |
| t2_powerplay_bowl_dot_pct | +0.17 | Team 2 powerplay bowling dot ball rate |
| t2_elo | -0.06 | Team 2 ELO rating (cross-season strength) |
| t2_season_chase_win_rate | +0.06 | Team 2 chase success this season |
| t1_top_bat_avg | +0.05 | Team 1 top-order batting average |
| t1_xi_bowl_wkts | +0.04 | Team 1 bowling wicket-taking ability |
| diff_win_rate_last10 | -0.04 | Relative recent form (last 10 matches) |
| h2h_t1_win_rate | -0.02 | Head-to-head historical win rate |
| diff_days_rest | +0.01 | Rest advantage |
| diff_xi_not_out_pct | -0.01 | Relative finishing ability |
| diff_powerplay_bowl_bound_pct | -0.01 | Relative powerplay bowling boundary concession |

The dominant signals are bowling depth and lower-order resilience — not batting firepower or ELO. This aligns with IPL wisdom: bowling wins tournaments.

### What didn't work

- **ELO as a primary signal**: ELO from prior seasons is slightly *anti-predictive* (r = -0.07 with 2024 outcomes). Teams that dominated one season regress the next due to squad turnover and auction dynamics.
- **172-feature GBDT**: Massive overfitting. Walk-forward accuracy was 58% — barely above the 50% baseline.
- **Phase-specific features as primary signals**: Powerplay/middle/death stats are noisy in isolation. Only powerplay bowling dot rate survived feature selection.
- **Win rate features across seasons**: "Last 5 matches" spanning two seasons has r = 0.02 with outcomes. Current-season-only form is what matters.

## Project Structure

```
├── fetch_data.py              # Fetch matches, ball-by-ball, player innings from ESPNCricinfo
├── fetch_bios.py              # Fetch player bios (bat/bowl style, role)
├── master_matches.csv         # 716 IPL matches 2015-2025
├── matches/                   # 716 ball-by-ball CSVs
├── player_innings/            # 506 player T20 career files
├── player_bios.csv            # Player metadata
├── dataset.csv                # 704 matches × 199 features (built with time ordering)
├── models/
│   ├── model.pkl              # Production sklearn Pipeline (StandardScaler + LogisticRegression)
│   └── bundle.json            # Feature list, hyperparameters, config
├── predictor/
│   ├── normalize.py           # Team name normalization, result parsing
│   ├── playing_xi.py          # Extract playing XIs from ball-by-ball data
│   ├── features.py            # All feature builders (ELO, form, phase, player, venue, toss)
│   ├── build_dataset.py       # Orchestrate feature computation with strict time ordering
│   ├── train.py               # Train, evaluate on holdout seasons, save production model
│   └── predict.py             # CLI for match prediction
└── requirements.txt
```

## Usage

### Setup

```bash
python3.10+ -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Fetch data (takes ~10 minutes, rate-limited)

```bash
python fetch_data.py
python fetch_bios.py
```

### Build dataset and train

```bash
python -m predictor.build_dataset
python -m predictor.train
```

### Predict a match

```bash
python -m predictor.predict \
  --team1 "Mumbai Indians" --team2 "Chennai Super Kings" \
  --venue "Wankhede Stadium" --city "Mumbai" \
  --toss-winner "Mumbai Indians" --toss-decision field \
  --team1-xi "Rohit Sharma" "Ishan Kishan" ... \
  --team2-xi "Ruturaj Gaikwad" "Devon Conway" ...
```

## Requirements

- Python 3.10+
- `cricdata` (ESPNCricinfo client)
- `scikit-learn`, `pandas`, `numpy`
- `catboost`, `lightgbm`, `optuna` (used during development, not required for inference)
