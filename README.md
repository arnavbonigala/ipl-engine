# IPL Match Prediction

A pre-match winner prediction model for the Indian Premier League, trained on ball-by-ball data from ESPNCricinfo (2015--2025). The final model is a stacked ensemble of logistic regressions that averages **78.6% holdout accuracy** across the 2023, 2024, and 2025 seasons.

## Results

| Holdout | Accuracy | Correct / Total | Brier Score | ROC AUC |
|---------|----------|-----------------|-------------|---------|
| 2023    | 76.7%    | 56 / 73         | 0.229       | 0.767   |
| 2024    | 77.5%    | 55 / 71         | 0.173       | 0.819   |
| 2025    | 81.7%    | 58 / 71         | 0.152       | 0.859   |

Each holdout year is predicted using a model trained exclusively on prior seasons. No future data leaks into training at any point.

## Approach

### Data collection

All data comes from ESPNCricinfo via the [`cricdata`](https://pypi.org/project/cricdata/) Python client. The pipeline fetches match metadata, full ball-by-ball deliveries, T20 career innings for every player who appeared, player bios, match weather, captaincy records, and innings-level fielding stats.

```
scripts/fetch_data.py              → data/master_matches.csv, data/matches/, data/player_innings/
scripts/fetch_bios.py              → data/player_bios.csv
scripts/fetch_weather.py           → data/match_weather.csv
scripts/fetch_captains.py          → data/match_captains.csv
scripts/fetch_fielding.py          → data/player_fielding.csv
scripts/fetch_fielding_innings.py  → data/player_fielding_innings/
predictor/build_dataset.py         → data/dataset.csv (704 matches × 284 features)
```

### Feature engineering

I computed 284 features across 15 categories. The dataset builder processes matches in strict chronological order — every feature for a given match is derived only from data available before that match.

| Category | Examples | Source |
|----------|----------|--------|
| ELO ratings | Team ELO, ELO diff, expected win prob | Match results with season decay |
| Team form | Win rate (last 5/10/season), win/loss streak | Prior match outcomes |
| Current season form | Season win rate, chase win rate this season | Current season only |
| Phase stats | Powerplay/middle/death run rate, dot %, boundary % | Ball-by-ball data |
| Player aggregates | XI batting SR, bowling economy, consistency, experience | Player career innings |
| Player vs venue/opposition | Batting avg at ground, bowling economy vs opponent | Player career innings |
| Squad composition | Pace/spin bowler count, left-hand bat count, role balance | Player bios |
| Venue | Avg 1st innings total, chase win rate, boundary size | Historical matches |
| Toss | Toss winner, decision, interaction with venue chase rate | Match metadata |
| Context | Match number in season, days rest, home/away, playoff flag | Match metadata |
| Points table | Pre-match points, position, NRR | Reconstructed from BBB |
| Win margin | Average margin of victory in current season | Match results |
| Squad churn | Players changed from previous match XI | Playing XI tracking |
| Weather / time | Temperature, humidity, dew point, wind, cloud cover, day/night | Open-Meteo via cricdata |
| Diff features | team1_X - team2_X for all paired features | Derived |

Of these 284, only 23 are used by the final model. The rest are noise at this sample size.

### Model architecture

The core challenge is sample size: the IPL has ~70 matches per season, so even with data back to 2018 there are only ~200 usable training samples for any given holdout year. Gradient boosting methods (CatBoost, LightGBM, XGBoost) all overfit badly here, topping out around 55--60% holdout accuracy. Logistic regression with strong L2 regularization (C=0.1) generalizes much better.

A single LR model with 9 forward-selected features reached 76.3% average accuracy. To squeeze more signal out without overfitting, I split the feature space into thematic clusters and trained a separate LR on each. A meta-learner then combines their out-of-fold predictions.

| Base Model | Features |
|------------|----------|
| Baseline (9) | diff_win_streak, t2_spin_count, t2_pace_count, diff_season_matches, h2h_t1_win_rate, diff_left_bowl_count, t2_low_bat_sr, diff_middle_bowl_extras_per_match, diff_table_nrr |
| Composition (5) | t2_specialist_bowler, t2_bowling_ar, t2_specialist_bat, t2_allrounder, t2_wk_bat |
| Form (5) | t2_win_streak, diff_season_matches, diff_matches_played, t1_win_rate_last10, t2_loss_streak |
| Phase (4) | t1_middle_bat_dot_pct, t2_powerplay_bowl_rr, diff_death_bowl_bound_pct, diff_death_bowl_extras_per_match |

The meta-learner is itself an LR (C=0.1) trained on out-of-fold probabilities generated via leave-one-season-out CV within the training data.

### Training configuration

- **Training window**: 2018 onward, excluding 2020 and 2021 (COVID neutral-venue seasons that distort home advantage signals).
- **Sample weighting**: Exponential decay with a 2-year half-life, plus a 3x multiplier for impact-player-era matches (2023+). For predicting 2024, a 2023 match gets weight 3.00, 2022 gets 0.71, 2019 gets 0.25, and 2018 gets 0.18.

### Error analysis

Looking at the misclassified matches across all three holdout years:

- **2023**: All 17 errors are coin-flips with less than 10% confidence. The model sees these as genuine 50/50 matchups, and there's nothing in the features to separate them.
- **2024**: 6 confident errors, mostly driven by RCB's late-season comeback (3 wins the model predicted against) and Gujarat Titans' unexpected collapse. Both are mid-season regime changes that backward-looking features can't anticipate.
- **2025**: 6 confident errors dominated by two patterns. CSK went 4-10 but pulled off 3 upsets against teams the model strongly favored — a bad team occasionally winning is indistinguishable from noise. Mumbai Indians started poorly then went on a 7-match winning streak; rolling features lagged behind the turnaround.

The common thread is that the model can't detect momentum shifts that happen over 2--3 matches within a season. Features like win streak and form update gradually, so a team that suddenly clicks (or falls apart) gets mispredicted until the stats catch up.

### What didn't work

I tested a large number of approaches that failed to improve accuracy:

- **ELO as a primary signal**: Anti-predictive across seasons (r = -0.07 with 2024 outcomes). IPL teams regress hard year-to-year due to auction turnover.
- **284-feature GBDT**: Walk-forward accuracy around 58%, barely above coin-flip.
- **Weather and time features**: Temperature, humidity, dew point, wind, cloud cover — none added signal beyond what toss and venue already capture.
- **Points table position / points**: Collinear with win streak and season match count. Only NRR contributed independently.
- **Squad churn, win margins, phase matchups, playoff flag, boundary size, player role balance**: Tested individually and in combination. None improved the baseline.
- **Cross-season win rate features**: "Last 5 matches" spanning two seasons has r = 0.02 with outcomes. Only current-season form matters.
- **Cross-league T20 data (BBL, PSL, CPL, SA20)**: Different rules, conditions, and player pools. Signals don't transfer.
- **Historical betting odds**: No freely available dataset. Would require scraping OddsPortal with browser automation, and wouldn't cover training years anyway.
- **Time-ordered fielding quality**: Career fielding stats initially looked strong (80.5% solo accuracy) but were leaking future data. Once computed with strict time ordering using innings-level data, the signal disappeared.

### Where the ceiling is

With publicly available pre-match data and ~200 training samples, 78--79% appears to be close to the ceiling. The remaining errors split into coin-flips that no feature can resolve and mid-season regime changes that backward-looking statistics react to too slowly. The only realistic paths beyond this would be integrating pre-match betting odds (market consensus) or day-of information like pitch conditions and last-minute squad changes.

## Project Structure

```
├── data/
│   ├── master_matches.csv         # 716 IPL matches 2015-2025
│   ├── player_bios.csv            # Player metadata
│   ├── match_weather.csv          # Weather and time data
│   ├── match_captains.csv         # Captain data
│   ├── player_fielding.csv        # Career fielding stats
│   ├── dataset.csv                # 704 matches × 284 features
│   ├── matches/                   # 716 ball-by-ball CSVs
│   ├── player_innings/            # 506 player T20 career files
│   └── player_fielding_innings/   # 506 innings-level fielding CSVs
├── scripts/
│   ├── fetch_data.py              # Fetch matches, ball-by-ball, player innings
│   ├── fetch_bios.py              # Fetch player bios
│   ├── fetch_weather.py           # Fetch weather and day/night data
│   ├── fetch_captains.py          # Fetch captain information
│   ├── fetch_fielding.py          # Fetch career fielding stats
│   └── fetch_fielding_innings.py  # Fetch innings-level fielding data
├── predictor/
│   ├── normalize.py               # Team name normalization, result parsing
│   ├── playing_xi.py              # Extract playing XIs from ball-by-ball data
│   ├── features.py                # Feature builders
│   ├── build_dataset.py           # Dataset construction with strict time ordering
│   ├── train.py                   # Training, holdout evaluation, model export
│   └── predict.py                 # Match prediction CLI
├── models/
│   ├── model.pkl                  # Stacked ensemble (4 base LRs + meta-learner)
│   └── bundle.json                # Feature sets, hyperparameters, config
└── requirements.txt
```

## Usage

### Setup

```bash
python3.10+ -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Fetch data

```bash
python scripts/fetch_data.py
python scripts/fetch_bios.py
python scripts/fetch_weather.py
python scripts/fetch_captains.py
python scripts/fetch_fielding.py
python scripts/fetch_fielding_innings.py
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
