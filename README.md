# IPL Match Prediction

Post-toss winner predictor for the Indian Premier League, built on ball-by-ball data from ESPNCricinfo (2015--2025). A scaled logistic regression with 9 features achieves **76.3% average holdout accuracy** across the 2023, 2024, and 2025 seasons.

## Results

| Holdout | Accuracy | Correct / Total | Brier Score | ROC AUC |
|---------|----------|-----------------|-------------|---------|
| 2023    | 78.1%    | 57 / 73         | 0.211       | 0.765   |
| 2024    | 76.1%    | 54 / 71         | 0.210       | 0.738   |
| 2025    | 74.6%    | 53 / 71         | 0.200       | 0.756   |

Features were selected via multi-start forward selection optimizing average accuracy across all three holdout years.

## Data Pipeline

All data is fetched from ESPNCricinfo via the `cricdata` Python client.

```
fetch_data.py          → master_matches.csv   (716 matches, 2015-2025)
                       → matches/             (716 ball-by-ball CSVs)
                       → player_innings/      (506 player career files)
fetch_bios.py          → player_bios.csv      (506 players: bat/bowl style, role)
fetch_weather.py       → match_weather.csv    (weather + day/night for all matches)
predictor/build_dataset.py → dataset.csv      (704 valid matches × 254 features)
```

`fetch_data.py` pulls three things per season: match metadata, full ball-by-ball delivery data for every match, and complete T20 career innings for every player who appeared. `fetch_bios.py` adds lightweight player metadata (batting hand, bowling style classified as pace/spin, role).

The dataset builder walks matches in chronological order. For each match, features are computed using only data from prior matches — no future leakage.

## Feature Engineering

254 raw features are computed across these categories:

| Category | Examples | Source |
|----------|----------|--------|
| ELO ratings | Team ELO, ELO diff, expected win prob | Match results with season decay |
| Team form | Win rate (last 5/10/season), win/loss streak | Prior match outcomes |
| Current season form | Season win rate, chase win rate this season | Current season only |
| Phase stats | Powerplay/middle/death run rate, dot ball %, boundary % | Ball-by-ball data |
| Player aggregates | XI batting SR, bowling economy, consistency, experience | Player career innings |
| Player vs venue/opposition | Batting avg at ground, bowling economy vs opponent | Player career innings |
| Squad composition | Pace/spin bowler count, left-hand bat count, player role balance | Player bios |
| Venue | Avg 1st innings total, chase win rate at ground, boundary size | Historical matches |
| Toss | Toss winner, decision, interaction with venue chase rate | Match metadata |
| Context | Match number in season, days rest, home/away, playoff flag | Match metadata |
| Points table | Pre-match points, position, NRR | Reconstructed from BBB data |
| Win margin | Average margin of victory (runs/wickets+balls) in current season | Match results |
| Squad churn | Players changed from previous match XI | Playing XI tracking |
| Weather / time | Temperature, humidity, dew point, wind, cloud cover, day/night | Open-Meteo via cricdata |
| Diff features | team1_X - team2_X for all paired features | Derived |

Of these 254, only **9 are used by the final model** — the rest are noise at this sample size.

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

### The 9 selected features

Chosen by multi-start forward selection optimizing average accuracy across 2023+2024+2025 holdouts:

| Feature | Coefficient | Meaning |
|---------|-------------|---------|
| t2_spin_count | +0.72 | Team 2 number of spin bowlers |
| t2_pace_count | +0.69 | Team 2 number of pace bowlers |
| diff_season_matches | -0.38 | Difference in matches played this season |
| diff_win_streak | -0.30 | Difference in current win/loss streak |
| diff_middle_bowl_extras_per_match | -0.15 | Relative middle-overs bowling extras |
| diff_left_bowl_count | -0.12 | Relative left-arm bowler count |
| diff_table_nrr | -0.07 | Difference in pre-match net run rate |
| t2_low_bat_sr | -0.06 | Team 2 low-strike-rate batsmen count |
| h2h_t1_win_rate | -0.06 | Head-to-head historical win rate |

The dominant signals are bowling composition and current-season momentum — teams with more bowling options and better recent form win more.

### What didn't work

- **ELO as a primary signal**: ELO from prior seasons is slightly *anti-predictive* (r = -0.07 with 2024 outcomes). Teams that dominated one season regress the next due to squad turnover and auction dynamics.
- **254-feature GBDT**: Massive overfitting. Walk-forward accuracy was ~58% — barely above the 50% baseline.
- **Weather / time features**: Temperature, humidity, dew point, wind, and cloud cover — none improved accuracy. The toss decision and venue features already capture this signal implicitly.
- **Points table position / points**: Collinear with existing win streak and season matches features. Only NRR (which encodes *dominance* of wins) added signal.
- **Squad churn / win margins / phase matchups / playoff flag / boundary size / player role balance**: All tested individually and in combination; none improved the 9-feature baseline at this sample size.
- **Win rate features across seasons**: "Last 5 matches" spanning two seasons has r = 0.02 with outcomes. Current-season-only form is what matters.

## Project Structure

```
├── fetch_data.py              # Fetch matches, ball-by-ball, player innings from ESPNCricinfo
├── fetch_bios.py              # Fetch player bios (bat/bowl style, role)
├── fetch_weather.py           # Fetch weather + day/night data for all matches
├── master_matches.csv         # 716 IPL matches 2015-2025
├── matches/                   # 716 ball-by-ball CSVs
├── player_innings/            # 506 player T20 career files
├── player_bios.csv            # Player metadata
├── match_weather.csv          # Weather and time data per match
├── dataset.csv                # 704 matches × 254 features (built with time ordering)
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
