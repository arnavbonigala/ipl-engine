"""Model prediction + edge/Kelly computation."""

from engine.config import (
    KELLY_FRACTION, MAX_BET_FRACTION, MAX_ENTRY_PRICE, MIN_CONFIDENCE, MIN_EDGE,
)
from engine.market import get_market_price
from predictor.predict import predict


def generate_signal(match_details: dict, market_info: dict, bankroll: float) -> dict | None:
    """Run model, compare to market, compute Kelly sizing.

    Args:
        match_details: dict with team1, team2, venue, city,
                       toss_winner, toss_decision, team1_xi, team2_xi
        market_info: dict with t1_ticker, t2_ticker
        bankroll: current bankroll in dollars

    Returns:
        Signal dict or None if no edge / below confidence.
    """
    impact = match_details.get("impact_subs", {})
    result = predict(
        team1=match_details["team1"],
        team2=match_details["team2"],
        venue=match_details["venue"],
        city=match_details["city"],
        toss_winner=match_details["toss_winner"],
        toss_decision=match_details["toss_decision"],
        team1_xi=match_details["team1_xi"],
        team2_xi=match_details["team2_xi"],
        team1_impact=impact.get(match_details["team1"]),
        team2_impact=impact.get(match_details["team2"]),
        squad_ids=match_details.get("squad_ids"),
    )

    model_t1_prob = result["team1_win_prob"]
    model_confidence = max(model_t1_prob, 1 - model_t1_prob)

    if model_confidence < MIN_CONFIDENCE:
        return None

    model_picks_t1 = model_t1_prob > 0.5
    chosen_team = match_details["team1"] if model_picks_t1 else match_details["team2"]
    model_our_prob = model_t1_prob if model_picks_t1 else 1 - model_t1_prob

    # Map by team name — cricdata and Kalshi may order teams differently
    team_to_ticker = {
        market_info["team1"]: market_info["t1_ticker"],
        market_info["team2"]: market_info["t2_ticker"],
    }
    ticker = team_to_ticker.get(chosen_team)
    if not ticker:
        return None

    entry_price = get_market_price(ticker)
    if entry_price is None:
        return None

    if entry_price <= 0.01 or entry_price >= 0.99:
        return None

    if entry_price > MAX_ENTRY_PRICE:
        return None

    edge = model_our_prob - entry_price
    if edge < MIN_EDGE:
        return None

    kelly_full = edge / (1 - entry_price)
    kelly = min(kelly_full * KELLY_FRACTION, MAX_BET_FRACTION)
    kelly = max(kelly, 0)

    bet_amount = round(bankroll * kelly, 2)
    contracts = round(bet_amount / entry_price, 2)

    side = "T1" if model_picks_t1 else "T2"

    return {
        "side": side,
        "team": chosen_team,
        "ticker": ticker,
        "model_prob": round(model_our_prob, 4),
        "model_t1_prob": round(model_t1_prob, 4),
        "market_price": round(entry_price, 4),
        "edge": round(edge, 4),
        "kelly_fraction": round(kelly, 4),
        "bet_amount": bet_amount,
        "contracts": contracts,
        "team1": match_details["team1"],
        "team2": match_details["team2"],
    }
