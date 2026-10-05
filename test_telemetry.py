from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import telemetry
import live_trader
from alpaca.trading.enums import OrderSide, OrderType


def test_send_telegram_message_posts_to_configured_chat(monkeypatch):
    captured = {}
    monkeypatch.setattr(telemetry, "_telegram_credentials", lambda: ("test-token", "chat-42"))
    monkeypatch.setattr(
        telemetry.requests,
        "post",
        lambda url, **kwargs: captured.update(url=url, **kwargs) or SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"ok": True},
        ),
    )

    assert telemetry.send_telegram_message("🟢 fill")
    assert captured["url"] == "https://api.telegram.org/bottest-token/sendMessage"
    assert captured["json"] == {"chat_id": "chat-42", "text": "🟢 fill"}
    assert captured["timeout"] == (3, 10)


def test_send_telegram_message_is_best_effort_when_request_fails(monkeypatch):
    monkeypatch.setattr(telemetry, "_telegram_credentials", lambda: ("test-token", "chat-42"))

    def fail_request(*args, **kwargs):
        raise telemetry.requests.ConnectionError("offline")

    monkeypatch.setattr(telemetry.requests, "post", fail_request)

    assert not telemetry.send_telegram_message("⚠️ test")


def test_critical_alert_uses_warning_message(monkeypatch):
    messages = []
    monkeypatch.setattr(telemetry, "send_telegram_message", messages.append)

    telemetry.send_critical_alert("stream Alpaca", "connection lost")

    assert messages == ["⚠️ Rilevata anomalia (stream Alpaca): connection lost"]


def test_live_fill_telemetry_formats_buy_and_trailing_stop(monkeypatch):
    messages = []
    monkeypatch.setattr(live_trader, "send_telegram_message", messages.append)
    buy = SimpleNamespace(
        order=SimpleNamespace(symbol="SPY", side=OrderSide.BUY, type=None),
        qty="2",
        price="101.25",
    )
    stop = SimpleNamespace(
        order=SimpleNamespace(symbol="SPY", side=OrderSide.SELL, type=OrderType.TRAILING_STOP),
        qty="2",
        price="105.00",
    )

    live_trader._send_fill_telemetry(buy, 0.0)
    live_trader._send_fill_telemetry(stop, 7.35)

    assert messages[0] == "🟢 Eseguito BUY su SPY - Qtà: 2, Prezzo: $101.25, Controvalore: $202.50"
    assert messages[1] == "🔴 Scattato Trailing Stop su SPY - Qtà: 2, Prezzo: $105.00, P&L operazione: $+7.35"


def test_live_trade_update_returns_net_realized_pnl_and_claims_execution_once():
    symbol = f"TEL{uuid4().hex[:8].upper()}"

    class EmptyActivitiesClient:
        def get(self, path, query):
            return []

    client = EmptyActivitiesClient()

    def update(side, price, execution_id):
        return SimpleNamespace(
            event="fill",
            order=SimpleNamespace(
                id=execution_id,
                symbol=symbol,
                side=side,
                filled_qty="1",
                type=OrderType.TRAILING_STOP if side == OrderSide.SELL else None,
            ),
            qty="1",
            price=str(price),
            timestamp=datetime(2026, 10, 2, 14, tzinfo=timezone.utc),
            execution_id=execution_id,
        )

    buy = update(OrderSide.BUY, 100, f"buy-{symbol}")
    sell = update(OrderSide.SELL, 110, f"sell-{symbol}")
    state, buy_pnl = live_trader._apply_trade_update_with_pnl(client, buy)
    _, sell_pnl = live_trader._apply_trade_update_with_pnl(client, sell)

    assert state is not None
    assert buy_pnl == 0
    assert sell_pnl == 9.79
    assert live_trader._claim_telemetry_execution(sell)
    assert not live_trader._claim_telemetry_execution(sell)
