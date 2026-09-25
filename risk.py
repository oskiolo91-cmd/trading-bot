"""Position sizing."""

from __future__ import annotations

import math


def calc_position_size(
    portfolio_value: float,
    entry_price: float,
    stop_loss_price: float,
    risk_pct: float = 0.01,
    max_cap_pct: float = 0.10,
) -> int:
    """Return the number of shares risking ``risk_pct`` of the portfolio between entry and stop.

    The size is additionally capped so the position notional never exceeds ``max_cap_pct``
    of the portfolio (protects against tiny ATR -> huge size).

    Args:
        portfolio_value: Current portfolio equity.
        entry_price: Planned entry price.
        stop_loss_price: Protective stop price (must be below entry for a long).
        risk_pct: Fraction of portfolio to risk on the trade.
        max_cap_pct: Maximum fraction of portfolio allocated to the position notional.

    Returns:
        Integer share count (floored), 0 if inputs are invalid.
    """
    if portfolio_value <= 0 or entry_price <= 0:
        return 0
    risk_per_share = entry_price - stop_loss_price
    if not math.isfinite(risk_per_share) or risk_per_share <= 0:
        return 0
    by_risk = portfolio_value * risk_pct / risk_per_share
    by_cap = portfolio_value * max_cap_pct / entry_price
    return max(0, math.floor(min(by_risk, by_cap)))
