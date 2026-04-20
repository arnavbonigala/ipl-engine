# IPL Engine

A live IPL forecasting and market execution engine that combines leakage-safe match prediction, real-time lineup/toss ingestion, automated decision logic, and cloud-deployed monitoring/retraining.

IPL Engine ingests historical and live match data, waits for toss and confirmed lineup information, generates pre-match win probabilities using a stacked ensemble trained on ball-by-ball data, compares them against market prices, applies Kelly criterion risk sizing, executes decisions automatically, tracks outcomes and P&L, and retrains the model after each match day.

## Key Capabilities

- **Forecasting**: Stacked ensemble of logistic regressions averaging **74.3% walk-forward accuracy** across 2023–2025 seasons, trained on 284 engineered features with strict chronological ordering
- **Live data ingestion**: Polls ESPNCricinfo for toss results and confirmed playing XIs, with on-the-fly player data fetching for previously unseen players
- **Market comparison**: Discovers relevant prediction markets, extracts live prices, computes model edge over market-implied probabilities
- **Risk sizing**: Half-Kelly criterion with configurable fraction caps, minimum confidence (60%) and minimum edge (3%) filters — coin-flip predictions are automatically skipped
- **Automated execution**: Places and tracks positions, prevents duplicate bets across restarts, reconciles local state against exchange positions on startup
- **Retraining**: Rebuilds the dataset and retrains the model after each match day, incorporating new results while excluding unplayed fixtures
- **Monitoring**: Real-time dashboard with live match tracking, upcoming schedule, match history, and P&L statistics

## Modeling Approach

### Data

All data comes from ESPNCricinfo via the [`cricdata`](https://pypi.org/project/cricdata/) Python client: match metadata, full ball-by-ball deliveries, T20 career innings for every player, player bios, weather, captaincy records, and innings-level fielding stats.

### Features

284 features across 15 categories, computed in strict chronological order — every feature for a given match uses only data available before that match.

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
| Points table | Pre-match points, position, NRR | Reconstructed from ball-by-ball |
| Win margin | Average margin of victory in current season | Match results |
| Squad churn | Players changed from previous match XI | Playing XI tracking |
| Weather / time | Temperature, humidity, dew point, wind, cloud cover, day/night | Open-Meteo via cricdata |
| Diff features | team1\_X - team2\_X for all paired features | Derived |

Of these 284, only 23 are used by the final model. The rest are noise at this sample size.

### Architecture

The IPL has ~70 matches per season, yielding only ~200–400 usable training samples. Gradient boosting methods (CatBoost, LightGBM, XGBoost) overfit badly, topping out around 55–60%. Logistic regression with L2 regularization (C=0.1) generalizes much better.

The feature space is split into thematic clusters, each trained as a separate base model. A meta-learner combines their out-of-fold predictions:

| Base Model | Features |
|------------|----------|
| Baseline (9) | diff\_win\_streak, t2\_spin\_count, t2\_pace\_count, diff\_season\_matches, h2h\_t1\_win\_rate, diff\_left\_bowl\_count, t2\_low\_bat\_sr, diff\_middle\_bowl\_extras\_per\_match, diff\_table\_nrr |
| Composition (5) | t2\_specialist\_bowler, t2\_bowling\_ar, t2\_specialist\_bat, t2\_allrounder, t2\_wk\_bat |
| Form (5) | t2\_win\_streak, diff\_season\_matches, diff\_matches\_played, t1\_win\_rate\_last10, t2\_loss\_streak |
| Phase (4) | t1\_middle\_bat\_dot\_pct, t2\_powerplay\_bowl\_rr, diff\_death\_bowl\_bound\_pct, diff\_death\_bowl\_extras\_per\_match |

The meta-learner is itself an LR (C=0.1) trained on out-of-fold probabilities generated via leave-one-season-out CV within the training data.

**Training configuration**: 2018 onward, excluding 2020–2021 (COVID neutral-venue seasons). Sample weighting uses exponential decay with a 2.5-year half-life, plus 2x multiplier for impact-player-era matches (2023+).

### Results

Walk-forward holdout evaluation — each season is predicted using a model trained exclusively on prior seasons:

| Holdout | Accuracy | Correct / Total | Brier Score | ROC AUC |
|---------|----------|-----------------|-------------|---------|
| 2023    | 74.0%    | 54 / 73         | 0.233       | 0.717   |
| 2024    | 73.2%    | 52 / 71         | 0.191       | 0.791   |
| 2025    | 75.7%    | 53 / 70         | 0.180       | 0.812   |

Three-season average: **74.3%** accuracy. Early 2026 live-inference (announced-XI, walk-forward retrain before each match) currently runs at **67.9%** (19 / 28) — rising as the season accumulates more training data.

## Live Execution Engine

The engine runs as a continuous loop deployed to a cloud server:

1. **Market discovery**: On each match day, discovers available prediction markets and extracts team mappings, tickers, and expiration dates
2. **Time-aware polling**: Sleeps until 2 hours before match start, then polls ESPNCricinfo every 60 seconds for toss result and confirmed playing XIs — fresh client instances on each poll to avoid stale cached data
3. **Prediction**: Once toss and XIs are confirmed, builds features from all historical data and runs inference through the stacked ensemble
4. **Signal generation**: Compares model probability against market ask price. Filters: minimum 60% model confidence, minimum 3% edge over market. Matches below threshold are logged and skipped
5. **Position sizing**: Half-Kelly criterion capped at 25% of bankroll per position
6. **Execution**: Places the order, records position with entry price and contract count
7. **Settlement**: Monitors positions for settlement, computes realized P&L, removes settled matches from the active list
8. **Retraining**: After each match day, fetches updated results, rebuilds `dataset.csv` (filtering out unplayed fixtures), and retrains the production model
9. **State reconciliation**: On every startup, syncs local position state against the exchange to prevent duplicates and clear stale entries

### Duplicate Prevention

Multiple safeguards prevent the same match from being bet twice across restarts or redeployments:

- `_already_acted` check against local history and open positions before processing any match
- Pre-execution query to the exchange API to confirm no existing position on the specific ticker
- Startup sync that reconciles local state with live exchange positions

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      IPL Engine                             │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │  Market   │  │ Cricinfo │  │  Model   │  │  Executor  │  │
│  │ Discovery │→ │ Scraper  │→ │ Predict  │→ │  (Kalshi)  │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────────┘  │
│       ↑                                          │          │
│       │         ┌──────────┐  ┌──────────┐       │          │
│       └─────────│  State   │← │ Settler  │←──────┘          │
│                 └──────────┘  └──────────┘                  │
│                      ↓                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ Retrain  │  │Dashboard │  │  Logs    │                  │
│  │ Pipeline │  │ (FastAPI)│  │          │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

| Component | Role |
|-----------|------|
| `engine/run.py` | Main loop: daily scheduling, match processing, toss polling |
| `engine/scraper.py` | ESPNCricinfo integration: fixtures, toss detection, playing XIs |
| `engine/market.py` | Market discovery: finds relevant tickers and extracts prices |
| `engine/signal.py` | Signal generation: model inference, edge calculation, Kelly sizing |
| `engine/executor.py` | Exchange API: authentication, order placement, balance/position queries |
| `engine/state.py` | JSON persistence: positions, history, upcoming matches, event log |
| `engine/server.py` | FastAPI dashboard: live matches, upcoming, history, stats APIs |
| `predictor/` | Offline pipeline: data fetching, feature engineering, training, inference |

## Deployment

The system runs on a cloud droplet via Docker Compose:

- **App container**: Python process running the engine loop and FastAPI dashboard
- **Reverse proxy**: Caddy with automatic TLS and HTTP basic authentication
- **Persistent volume**: Match data, player data, and engine state survive container rebuilds

The dashboard is access-restricted rather than public. It runs alongside credentialed exchange services, so access is kept behind authentication until deployment is separated more cleanly.

## Project Structure

```
├── engine/
│   ├── run.py                 # Main engine loop and match processing
│   ├── scraper.py             # ESPNCricinfo live data ingestion
│   ├── market.py              # Market discovery and price fetching
│   ├── signal.py              # Prediction → signal → sizing pipeline
│   ├── executor.py            # Exchange API integration
│   ├── state.py               # State persistence and position management
│   ├── server.py              # FastAPI dashboard backend
│   ├── config.py              # Strategy constants and API configuration
│   └── static/                # Dashboard frontend (HTML/CSS/JS)
├── predictor/
│   ├── normalize.py           # Team name normalization, result parsing
│   ├── playing_xi.py          # Extract playing XIs from ball-by-ball data
│   ├── features.py            # 284-feature builder with strict time ordering
│   ├── build_dataset.py       # Dataset construction pipeline
│   ├── train.py               # Training, holdout evaluation, model export
│   └── predict.py             # Match prediction inference
├── scripts/                   # Data fetching scripts (ESPNCricinfo)
├── models/
│   ├── model.pkl              # Stacked ensemble (4 base LRs + meta-learner)
│   └── bundle.json            # Feature sets and hyperparameters
├── data/
│   ├── master_matches.csv     # IPL matches 2015–2026
│   ├── dataset.csv            # Engineered features (706 matches × 284 columns)
│   ├── matches/               # Ball-by-ball CSVs
│   ├── player_innings/        # T20 career data per player
│   └── ...                    # Bios, weather, captains, fielding
├── Dockerfile
├── docker-compose.yml
├── Caddyfile
└── requirements.txt
```

## Setup

```bash
python3.10+ -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Future Work

- Cleaner service separation (engine process vs dashboard vs retraining as independent services)
- Public-safe dashboard deployment decoupled from exchange credentials
- Improved risk controls: per-day loss limits, drawdown circuit breakers
- Broader market and sport support
- Alerting and monitoring (Slack/Discord notifications on bets, errors, model drift)
- In-play probability updates using live ball-by-ball data
