"""Per-symbol FIFO position and realized P&L accounting."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from threading import RLock
from zoneinfo import ZoneInfo


CENT = Decimal("0.01")
EASTERN = ZoneInfo("America/New_York")


def _decimal(value: Decimal | float | int | str) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


@dataclass
class _Lot:
    quantity: Decimal
    price: Decimal
    remaining_entry_fee: Decimal


class SymbolState:
    """Track long fills, fractional position size, and net realized P&L for one symbol."""

    def __init__(
        self,
        symbol: str,
        commission_pct: Decimal | float = 0.0,
        timezone_: ZoneInfo = EASTERN,
    ) -> None:
        if not symbol:
            raise ValueError("symbol must not be empty")
        self.symbol = symbol
        self.commission_pct = _decimal(commission_pct)
        if self.commission_pct < 0:
            raise ValueError("commission_pct must not be negative")
        self.timezone = timezone_
        self._lots: deque[_Lot] = deque()
        self._daily_realized: dict[datetime.date, Decimal] = {}
        self._execution_ids: set[str] = set()
        self._lock = RLock()

    @property
    def position_qty(self) -> Decimal:
        with self._lock:
            return sum((lot.quantity for lot in self._lots), Decimal("0"))

    @property
    def average_entry_price(self) -> Decimal | None:
        with self._lock:
            quantity = self.position_qty
            if quantity == 0:
                return None
            cost = sum((lot.quantity * lot.price for lot in self._lots), Decimal("0"))
            return cost / quantity

    def daily_realized_pnl(self, now: datetime | None = None) -> Decimal:
        current = now or datetime.now(self.timezone)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        day = current.astimezone(self.timezone).date()
        with self._lock:
            return _money(self._daily_realized.get(day, Decimal("0.00")))

    def record_fill(
        self,
        side: str,
        qty: Decimal | float | int | str,
        price: Decimal | float | int | str,
        timestamp: datetime,
        execution_id: str | None = None,
        commission: Decimal | float | int | str | None = None,
    ) -> Decimal:
        """Apply one incremental fill and return its realized net P&L contribution."""
        normalized_side = str(getattr(side, "value", side)).lower()
        quantity = _decimal(qty)
        fill_price = _decimal(price)
        if normalized_side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        if quantity <= 0 or fill_price <= 0:
            raise ValueError("fill quantity and price must be positive")
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        fill_day = timestamp.astimezone(self.timezone).date()
        fee = _money(_decimal(commission)) if commission is not None else _money(
            quantity * fill_price * self.commission_pct
        )
        execution_key = str(execution_id) if execution_id is not None else None

        with self._lock:
            if execution_key and execution_key in self._execution_ids:
                return Decimal("0.00")
            if normalized_side == "buy":
                self._lots.append(_Lot(quantity, fill_price, fee))
                realized = Decimal("0.00")
            else:
                if quantity > self.position_qty:
                    raise ValueError(f"sell fill exceeds the tracked {self.symbol} position")
                realized = self._apply_sell(quantity, fill_price, fee)
                self._daily_realized[fill_day] = self._daily_realized.get(
                    fill_day, Decimal("0.00")
                ) + realized
            if execution_key:
                self._execution_ids.add(execution_key)
            return _money(realized)

    def _apply_sell(self, quantity: Decimal, price: Decimal, sell_fee: Decimal) -> Decimal:
        original_quantity = quantity
        remaining = quantity
        fee_remaining = sell_fee
        realized = Decimal("0.00")
        while remaining > 0:
            lot = self._lots[0]
            matched = min(remaining, lot.quantity)
            if matched == lot.quantity:
                entry_fee = lot.remaining_entry_fee
            else:
                entry_fee = _money(lot.remaining_entry_fee * matched / lot.quantity)
            if matched == remaining:
                matched_sell_fee = fee_remaining
            else:
                matched_sell_fee = _money(sell_fee * matched / original_quantity)
                matched_sell_fee = min(matched_sell_fee, fee_remaining)

            realized += matched * (price - lot.price) - entry_fee - matched_sell_fee
            lot.quantity -= matched
            lot.remaining_entry_fee -= entry_fee
            remaining -= matched
            fee_remaining -= matched_sell_fee
            if lot.quantity == 0:
                self._lots.popleft()
        return realized