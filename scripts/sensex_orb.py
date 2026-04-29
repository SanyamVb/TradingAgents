#!/usr/bin/env python3
"""
Sensex ORB (Opening Range Breakout) Options Trading Script
============================================================

Strategy
--------
1. Wait until 09:15 IST and capture SENSEX as the open price.
2. Wait until 09:30 IST (15-minute observation window).
3. Poll SENSEX every 30s until 10:00 IST. On a +/-50 point move from the
   open price, buy the ATM weekly option (CE on +50, PE on -50) at
   (option_LTP - 12) limit. Each direction triggers at most once per session.
4. After fill, place an OCO Smart Order to bracket the position with a
   target of avg_buy + 45 and a stop-loss of avg_buy - 40.
5. At 11:30 IST, hard-close any still-open position (cancel OCO + market sell).

Deployment (Groww Playground)
-----------------------------
- Use the TOTP-token auth flow, NOT "API key & Secret":
    1. Open https://groww.in/trade-api/api-keys
    2. Click the dropdown next to "Generate API Key" and choose
       "Generate TOTP token".
    3. Copy the TOTP token AND the base32 Secret it shows.
       (The "API key & Secret" flow uses non-base32 secrets and
       requires daily manual approval - it will NOT work here.)
- Configure these environment variables in the Playground:
    GROWW_API_KEY     - the TOTP token from step 2
    GROWW_API_SECRET  - the base32 Secret from step 2 (only A-Z and 2-7,
                        optional '=' padding)
    DRY_RUN           - "true" (default) or "false" to enable live trading
    SMOKE_TEST        - "true" forces an end-to-end DRY_RUN exercise NOW
                        (skips time gates, force-fires both CE and PE legs
                        at the live SENSEX price, then runs the hard-exit
                        sweep). Requires DRY_RUN=true. Use to verify the
                        script works without waiting until 09:15.
    TRIGGER_POINTS    - override the 50-point breakout threshold (e.g. set
                        to "1" to force the trade path on any tick during
                        a real DRY_RUN session). Default 50.
    LOTS              - override the number of lots (default 2; set to 1
                        for a minimum-exposure live test).
- Schedule the script to launch around 09:13 IST on trading days.
  The script self-terminates a few minutes after 11:30 IST.

Verification ladder (recommended order before going live)
---------------------------------------------------------
1. SMOKE_TEST=true DRY_RUN=true  (any time, ~30s)
   - Verifies auth, instrument lookup, expiry resolution, ATM strike
     selection from the option chain, premium filter, OCO price math,
     and hard-exit sweep. No market hours required.
2. DRY_RUN=true TRIGGER_POINTS=1  (during 09:15-10:00 IST)
   - Runs the full timeline-driven flow with a tiny threshold so the
     trade path almost always fires on a real tick. Confirms the time
     gates and live LTP polling work end-to-end.
3. DRY_RUN=false LOTS=1  (during a quiet trading session)
   - Smallest possible live trade. Real money but capped at 1 lot.
4. DRY_RUN=false LOTS=2  (production)

Local testing
-------------
- A local .env file with the same vars works (python-dotenv is optional).
- Run with DRY_RUN=true (default) to log all intended orders without
  hitting the order book.

Requirements
------------
    pip install growwapi pyotp
    pip install python-dotenv   # optional, for local .env loading
"""

import binascii
import os
import re
import sys
import logging
import time as time_module
from datetime import datetime, time
from typing import Any, Dict, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import pyotp
from growwapi import GrowwAPI


# =============================================================================
# Configuration
# =============================================================================

IST = ZoneInfo("Asia/Kolkata")

DRY_RUN    = os.getenv("DRY_RUN", "true").strip().lower() != "false"
SMOKE_TEST = os.getenv("SMOKE_TEST", "false").strip().lower() == "true"

# Strategy parameters (TRIGGER_POINTS, LOTS overridable for testing)
TRIGGER_POINTS         = int(os.getenv("TRIGGER_POINTS", "50"))
ENTRY_DISCOUNT         = 12.0    # rupees below option LTP for limit buy
TARGET_PROFIT          = 45.0    # rupees above avg buy
STOP_LOSS              = 40.0    # rupees below avg buy
LOTS                   = int(os.getenv("LOTS", "2"))
MIN_PREMIUM            = 30.0
MAX_PREMIUM            = 300.0
TICK_SIZE              = 0.05    # SENSEX option tick size

# Loop timings (seconds)
POLL_INTERVAL_SEC      = 30
ORDER_FILL_TIMEOUT_SEC = 60
ORDER_FILL_POLL_SEC    = 5
LTP_RETRY_SEC          = 5
LTP_RETRY_ATTEMPTS     = 6

# Timeline (IST)
OPEN_TIME           = time(9, 15)
OBSERVATION_END     = time(9, 30)
TRIGGER_WINDOW_END  = time(10, 0)
HARD_EXIT_TIME      = time(11, 30)

# Underlying instrument (SENSEX is a BSE index)
UNDERLYING            = "SENSEX"
UNDERLYING_GROWW_KEY  = "BSE-SENSEX"
UNDERLYING_LTP_KEY    = "BSE_SENSEX"


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("sensex_orb")


# =============================================================================
# Auth
# =============================================================================

_BASE32_RE = re.compile(r"^[A-Z2-7]+=*$")


def _clean_secret(raw: str) -> str:
    """Strip whitespace, surrounding quotes, and uppercase a TOTP seed.

    Common copy-paste artefacts (newlines, surrounding quotes, lowercase
    letters) all decode to the same base32 value once normalised.
    """
    cleaned = raw.strip().strip("\"'").strip()
    cleaned = "".join(cleaned.split())
    return cleaned.upper()


def init_groww_client() -> GrowwAPI:
    """Authenticate via TOTP and return a GrowwAPI client."""
    api_key_raw = os.getenv("GROWW_API_KEY") or os.getenv("GROW_API_KEY")
    api_secret_raw = os.getenv("GROWW_API_SECRET") or os.getenv("GROW_API_SECRET")
    if not api_key_raw or not api_secret_raw:
        raise RuntimeError(
            "Missing GROWW_API_KEY / GROWW_API_SECRET environment variables"
        )

    api_key = api_key_raw.strip().strip("\"'").strip()
    api_secret = _clean_secret(api_secret_raw)

    if not _BASE32_RE.match(api_secret):
        masked = (
            f"len={len(api_secret)} "
            f"prefix={api_secret[:2]!r} "
            f"suffix={api_secret[-2:]!r}"
        )
        raise RuntimeError(
            "GROWW_API_SECRET is not a valid base32 TOTP seed "
            f"({masked}). The secret must contain only A-Z and 2-7 (with "
            "optional '=' padding).\n"
            "If your secret has special characters like '!', '#', '@' or "
            "lowercase letters mixed with digits 0/1/8/9, you generated the "
            "WRONG kind of credential. Open "
            "https://groww.in/trade-api/api-keys , click the dropdown next "
            "to 'Generate API Key', and choose 'Generate TOTP token' "
            "instead. Use the TOTP token as GROWW_API_KEY and the base32 "
            "Secret it gives you as GROWW_API_SECRET. The plain 'API key & "
            "Secret' flow requires daily approval and cannot be used here."
        )

    try:
        totp_gen = pyotp.TOTP(api_secret)
        # Trigger one .now() to surface any decode error before handing the
        # generator to the SDK.
        totp_gen.now()
    except binascii.Error as exc:
        raise RuntimeError(
            "Failed to derive TOTP from GROWW_API_SECRET. The value passes "
            "the base32 character check but base64.b32decode still rejected "
            f"it: {exc}. Re-copy the base32 Secret from Groww's 'Generate "
            "TOTP token' flow."
        ) from exc

    # The SDK accepts either the generator object or the current code; pass
    # the generator so the SDK can re-derive it if there's an internal retry.
    access_token = GrowwAPI.get_access_token(api_key=api_key, totp=totp_gen)
    return GrowwAPI(access_token)


# =============================================================================
# Bootstrap: expiry, lot size, ATM helpers
# =============================================================================

def get_current_week_expiry(groww: GrowwAPI) -> str:
    """Resolve the nearest future SENSEX option expiry from the instruments CSV.

    Returns expiry as YYYY-MM-DD. Robust to expiry-day changes (Tue/Thu/Fri)
    because we sort all future expiries and pick the earliest.
    """
    instruments = groww.get_all_instruments()
    sensex_options = instruments[
        (instruments["underlying_symbol"] == UNDERLYING)
        & (instruments["instrument_type"].isin(["CE", "PE"]))
        & (instruments["exchange"] == "BSE")
    ]
    if sensex_options.empty:
        raise RuntimeError("No SENSEX options found in instruments CSV")

    today = datetime.now(IST).date()
    future_expiries = []
    for raw in sensex_options["expiry_date"].dropna().unique():
        try:
            exp_d = datetime.strptime(str(raw), "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        if exp_d >= today:
            future_expiries.append(exp_d)

    if not future_expiries:
        raise RuntimeError("No future SENSEX expiries available")
    nearest = min(future_expiries)
    return nearest.strftime("%Y-%m-%d")


def get_sensex_lot_size(groww: GrowwAPI, expiry: str) -> int:
    """Look up SENSEX option lot size for the given expiry."""
    instruments = groww.get_all_instruments()
    matches = instruments[
        (instruments["underlying_symbol"] == UNDERLYING)
        & (instruments["expiry_date"] == expiry)
        & (instruments["instrument_type"].isin(["CE", "PE"]))
    ]
    if matches.empty:
        raise RuntimeError(f"No SENSEX options found for expiry {expiry}")
    lot_size = int(matches.iloc[0]["lot_size"])
    if lot_size <= 0:
        raise RuntimeError(f"Invalid lot size {lot_size} for SENSEX expiry {expiry}")
    return lot_size


def get_atm_strike(price: float, step: int = 100) -> int:
    """Round a price to the nearest 100 for ATM strike selection."""
    return int(round(price / step) * step)


def round_to_tick(price: float, tick: float = TICK_SIZE) -> float:
    """Round a price to the nearest tick."""
    return round(round(price / tick) * tick, 2)


# =============================================================================
# Market data
# =============================================================================

def get_sensex_ltp(groww: GrowwAPI) -> float:
    """Fetch the latest SENSEX index LTP."""
    resp = groww.get_ltp(
        segment=groww.SEGMENT_CASH,
        exchange_trading_symbols=(UNDERLYING_LTP_KEY,),
    )
    if UNDERLYING_LTP_KEY not in resp:
        raise RuntimeError(f"LTP response missing {UNDERLYING_LTP_KEY}: {resp}")
    return float(resp[UNDERLYING_LTP_KEY])


def get_sensex_ltp_with_retry(groww: GrowwAPI) -> float:
    """LTP fetch with bounded retries (handles brief outages at market open)."""
    last_err: Optional[Exception] = None
    for attempt in range(1, LTP_RETRY_ATTEMPTS + 1):
        try:
            return get_sensex_ltp(groww)
        except Exception as exc:
            last_err = exc
            log.warning("LTP fetch attempt %d/%d failed: %s",
                        attempt, LTP_RETRY_ATTEMPTS, exc)
            time_module.sleep(LTP_RETRY_SEC)
    raise RuntimeError(f"LTP fetch failed after {LTP_RETRY_ATTEMPTS} attempts") from last_err


def get_atm_option(
    groww: GrowwAPI, expiry: str, opt_type: str, sensex_price: float
) -> Tuple[str, int, float]:
    """Resolve ATM option from the option chain.

    Returns (trading_symbol, strike, ltp). opt_type must be 'CE' or 'PE'.
    """
    chain = groww.get_option_chain(
        exchange=groww.EXCHANGE_BSE,
        underlying=UNDERLYING,
        expiry_date=expiry,
    )
    strikes_dict = chain.get("strikes") or {}
    if not strikes_dict:
        raise RuntimeError(f"Empty option chain for {UNDERLYING} {expiry}")

    target_strike = get_atm_strike(sensex_price)
    available = sorted(int(k) for k in strikes_dict.keys())
    closest = min(available, key=lambda s: abs(s - target_strike))

    leg = strikes_dict[str(closest)].get(opt_type)
    if not leg:
        raise RuntimeError(f"No {opt_type} leg at strike {closest} in chain")

    trading_symbol = leg.get("trading_symbol")
    ltp = leg.get("ltp")
    if not trading_symbol or ltp is None:
        raise RuntimeError(f"Incomplete option leg: {leg}")
    return trading_symbol, closest, float(ltp)


# =============================================================================
# Order helpers
# =============================================================================

def make_ref_id(prefix: str) -> str:
    """8 to 20 char alphanumeric reference ID (with at most two hyphens)."""
    stamp = datetime.now(IST).strftime("%H%M%S")
    return f"{prefix}-{stamp}"[:20]


def place_buy_limit(
    groww: GrowwAPI, trading_symbol: str, qty: int, price: float, ref_id: str
) -> Dict[str, Any]:
    """Place a BUY LIMIT order on the option (FNO + MIS)."""
    if DRY_RUN:
        log.info("[DRY_RUN] BUY LIMIT %s qty=%d price=%.2f ref=%s",
                 trading_symbol, qty, price, ref_id)
        return {"groww_order_id": f"DRY-{ref_id}", "order_status": "DRY_RUN"}

    return groww.place_order(
        trading_symbol=trading_symbol,
        quantity=qty,
        validity=groww.VALIDITY_DAY,
        exchange=groww.EXCHANGE_BSE,
        segment=groww.SEGMENT_FNO,
        product=groww.PRODUCT_MIS,
        order_type=groww.ORDER_TYPE_LIMIT,
        transaction_type=groww.TRANSACTION_TYPE_BUY,
        price=price,
        order_reference_id=ref_id,
    )


def wait_for_fill(
    groww: GrowwAPI, order_id: str, timeout_sec: int
) -> Optional[Dict[str, Any]]:
    """Poll order status until EXECUTED, terminal-failed, or timeout."""
    if DRY_RUN:
        log.info("[DRY_RUN] Skipping fill wait for %s; assuming EXECUTED", order_id)
        return {"order_status": "EXECUTED", "filled_quantity": -1}

    deadline = time_module.time() + timeout_sec
    terminal_failed = {"CANCELLED", "REJECTED", "FAILED"}
    while time_module.time() < deadline:
        try:
            status = groww.get_order_status(
                groww_order_id=order_id,
                segment=groww.SEGMENT_FNO,
            )
        except Exception as exc:
            log.warning("get_order_status failed for %s: %s", order_id, exc)
            time_module.sleep(ORDER_FILL_POLL_SEC)
            continue

        order_status = (status.get("order_status") or "").upper()
        if order_status in ("EXECUTED", "COMPLETED"):
            return status
        if order_status in terminal_failed:
            log.warning("Order %s ended in terminal status %s", order_id, order_status)
            return None
        time_module.sleep(ORDER_FILL_POLL_SEC)
    return None


def get_avg_fill_price(groww: GrowwAPI, order_id: str) -> Optional[float]:
    """Return the average fill price for a completed order, or None."""
    if DRY_RUN:
        return None
    try:
        detail = groww.get_order_detail(
            groww_order_id=order_id,
            segment=groww.SEGMENT_FNO,
        )
        avg = detail.get("average_fill_price")
        return float(avg) if avg else None
    except Exception as exc:
        log.warning("get_order_detail failed for %s: %s", order_id, exc)
        return None


def cancel_order(groww: GrowwAPI, order_id: str) -> None:
    if DRY_RUN:
        log.info("[DRY_RUN] Cancel order %s", order_id)
        return
    try:
        groww.cancel_order(
            segment=groww.SEGMENT_FNO,
            groww_order_id=order_id,
        )
        log.info("Cancelled order %s", order_id)
    except Exception as exc:
        log.warning("Cancel failed for %s: %s", order_id, exc)


def place_oco_bracket(
    groww: GrowwAPI,
    trading_symbol: str,
    qty: int,
    avg_buy: float,
    ref_id: str,
) -> Optional[Dict[str, Any]]:
    """Place an OCO Smart Order: target LIMIT SELL + SL-M SELL on the long position."""
    target_price = round_to_tick(avg_buy + TARGET_PROFIT)
    sl_trigger   = round_to_tick(avg_buy - STOP_LOSS)

    if DRY_RUN:
        log.info(
            "[DRY_RUN] OCO %s qty=%d entry=%.2f target=%.2f sl=%.2f ref=%s",
            trading_symbol, qty, avg_buy, target_price, sl_trigger, ref_id,
        )
        return {"smart_order_id": f"DRY-OCO-{ref_id}", "status": "DRY_RUN"}

    return groww.create_smart_order(
        smart_order_type=groww.SMART_ORDER_TYPE_OCO,
        reference_id=ref_id,
        segment=groww.SEGMENT_FNO,
        trading_symbol=trading_symbol,
        quantity=qty,
        product_type=groww.PRODUCT_MIS,
        exchange=groww.EXCHANGE_BSE,
        duration=groww.VALIDITY_DAY,
        net_position_quantity=qty,
        transaction_type=groww.TRANSACTION_TYPE_SELL,
        target={
            "trigger_price": f"{target_price:.2f}",
            "order_type": groww.ORDER_TYPE_LIMIT,
            "price": f"{target_price:.2f}",
        },
        stop_loss={
            "trigger_price": f"{sl_trigger:.2f}",
            "order_type": groww.ORDER_TYPE_STOP_LOSS_MARKET,
            "price": None,
        },
    )


def cancel_oco(groww: GrowwAPI, smart_order_id: str) -> None:
    if DRY_RUN:
        log.info("[DRY_RUN] Cancel OCO %s", smart_order_id)
        return
    try:
        groww.cancel_smart_order(
            segment=groww.SEGMENT_FNO,
            smart_order_type=groww.SMART_ORDER_TYPE_OCO,
            smart_order_id=smart_order_id,
        )
        log.info("Cancelled OCO %s", smart_order_id)
    except Exception as exc:
        log.warning("OCO cancel failed for %s: %s", smart_order_id, exc)


def market_sell(
    groww: GrowwAPI, trading_symbol: str, qty: int, ref_id: str
) -> Optional[Dict[str, Any]]:
    """Submit a MARKET SELL on the long option position."""
    if DRY_RUN:
        log.info("[DRY_RUN] MARKET SELL %s qty=%d ref=%s", trading_symbol, qty, ref_id)
        return {"groww_order_id": f"DRY-EXIT-{ref_id}", "order_status": "DRY_RUN"}

    return groww.place_order(
        trading_symbol=trading_symbol,
        quantity=qty,
        validity=groww.VALIDITY_DAY,
        exchange=groww.EXCHANGE_BSE,
        segment=groww.SEGMENT_FNO,
        product=groww.PRODUCT_MIS,
        order_type=groww.ORDER_TYPE_MARKET,
        transaction_type=groww.TRANSACTION_TYPE_SELL,
        order_reference_id=ref_id,
    )


# =============================================================================
# Trade execution
# =============================================================================

def execute_trade(
    groww: GrowwAPI, ctx: Dict[str, Any], opt_type: str, sensex_price: float
) -> None:
    """Resolve ATM option, place buy, await fill, attach OCO bracket."""
    side_label = "CALL" if opt_type == "CE" else "PUT"
    log.info(
        "=== %s trigger fired | sensex=%.2f open=%.2f diff=%+.2f ===",
        side_label, sensex_price, ctx["open_price"], sensex_price - ctx["open_price"],
    )

    trading_symbol, strike, ltp = get_atm_option(
        groww, ctx["expiry"], opt_type, sensex_price
    )
    log.info("ATM %s -> symbol=%s strike=%d ltp=%.2f",
             side_label, trading_symbol, strike, ltp)

    if ltp < MIN_PREMIUM or ltp > MAX_PREMIUM:
        log.warning("Skipping %s: premium %.2f outside [%.0f, %.0f]",
                    side_label, ltp, MIN_PREMIUM, MAX_PREMIUM)
        return

    buy_price = round_to_tick(ltp - ENTRY_DISCOUNT)
    qty = ctx["quantity"]
    buy_ref = make_ref_id(f"orb{opt_type}")

    log.info("Placing BUY LIMIT %s qty=%d at %.2f (ltp %.2f - %.2f)",
             trading_symbol, qty, buy_price, ltp, ENTRY_DISCOUNT)
    order_resp = place_buy_limit(groww, trading_symbol, qty, buy_price, buy_ref)
    order_id = order_resp.get("groww_order_id")
    log.info("Buy order placed id=%s status=%s",
             order_id, order_resp.get("order_status"))

    fill = wait_for_fill(groww, order_id, ORDER_FILL_TIMEOUT_SEC)
    if not fill:
        log.warning("BUY %s did not fill in %ds; cancelling",
                    side_label, ORDER_FILL_TIMEOUT_SEC)
        cancel_order(groww, order_id)
        return

    avg_buy = get_avg_fill_price(groww, order_id) or buy_price
    log.info("BUY %s filled avg=%.2f", side_label, avg_buy)

    oco_ref = make_ref_id(f"oco{opt_type}")
    oco_resp = place_oco_bracket(groww, trading_symbol, qty, avg_buy, oco_ref)
    smart_id = oco_resp.get("smart_order_id") if oco_resp else None
    log.info("OCO placed smart_order_id=%s", smart_id)

    ctx["open_positions"][trading_symbol] = {
        "side": side_label,
        "qty": qty,
        "avg_buy": avg_buy,
        "entry_order_id": order_id,
        "oco_id": smart_id,
    }


# =============================================================================
# Hard exit
# =============================================================================

def flat_close_all(groww: GrowwAPI, ctx: Dict[str, Any]) -> None:
    if not ctx["open_positions"]:
        log.info("Hard exit: no open positions")
        return

    log.info("Hard exit: %d open position(s) at %s",
             len(ctx["open_positions"]), datetime.now(IST).strftime("%H:%M:%S"))
    for symbol, pos in list(ctx["open_positions"].items()):
        if pos.get("oco_id"):
            cancel_oco(groww, pos["oco_id"])
        exit_ref = make_ref_id("exit")
        try:
            market_sell(groww, symbol, pos["qty"], exit_ref)
            log.info("Market exit submitted %s qty=%d ref=%s",
                     symbol, pos["qty"], exit_ref)
        except Exception as exc:
            log.exception("Market exit failed for %s: %s", symbol, exc)


# =============================================================================
# Time helpers
# =============================================================================

def now_ist() -> datetime:
    return datetime.now(IST)


def wait_until(target: time, label: str = "") -> None:
    """Sleep (in chunks of <=60s) until current IST time reaches target."""
    while True:
        cur = now_ist()
        target_dt = datetime.combine(cur.date(), target, tzinfo=IST)
        delta = (target_dt - cur).total_seconds()
        if delta <= 0:
            return
        suffix = f" ({label})" if label else ""
        log.info("Sleeping %.0fs until %s%s",
                 delta, target.strftime("%H:%M"), suffix)
        time_module.sleep(min(delta, 60))


# =============================================================================
# Smoke test
# =============================================================================

def run_smoke_test(groww: GrowwAPI, ctx: Dict[str, Any]) -> None:
    """Exercise every code path without waiting on the clock or a real trigger.

    Use SMOKE_TEST=true (with DRY_RUN=true) to run this. Useful any time of
    day - it doesn't depend on market hours, only on the SDK responding.
    """
    log.info("=== SMOKE TEST: skipping time gates and force-firing both legs ===")

    open_price = get_sensex_ltp_with_retry(groww)
    ctx["open_price"] = open_price
    log.info("Live SENSEX = %.2f (used as simulated open price)", open_price)

    for opt_type in ("CE", "PE"):
        try:
            execute_trade(groww, ctx, opt_type, open_price)
        except Exception as exc:
            log.exception("Smoke test %s leg failed: %s", opt_type, exc)
        ctx["call_traded" if opt_type == "CE" else "put_traded"] = True

    log.info("--- Simulating hard-exit sweep ---")
    flat_close_all(groww, ctx)

    log.info("=== SMOKE TEST complete ===")


def log_summary(ctx: Dict[str, Any]) -> None:
    """Print a final human-readable summary of what the run did."""
    log.info("================ RUN SUMMARY ================")
    log.info("DRY_RUN=%s SMOKE_TEST=%s", DRY_RUN, SMOKE_TEST)
    log.info("Expiry=%s lot_size=%d lots=%d quantity=%d trigger=%d",
             ctx.get("expiry"), ctx.get("lot_size"), LOTS,
             ctx.get("quantity"), TRIGGER_POINTS)
    log.info("Open price=%s call_traded=%s put_traded=%s",
             ctx.get("open_price"),
             ctx.get("call_traded"), ctx.get("put_traded"))
    positions = ctx.get("open_positions") or {}
    if not positions:
        log.info("Positions opened: 0")
    else:
        log.info("Positions opened: %d", len(positions))
        for symbol, pos in positions.items():
            log.info(
                "  %s side=%s qty=%d avg_buy=%.2f target=%.2f sl=%.2f oco=%s",
                symbol, pos["side"], pos["qty"], pos["avg_buy"],
                pos["avg_buy"] + TARGET_PROFIT,
                pos["avg_buy"] - STOP_LOSS,
                pos.get("oco_id"),
            )
    log.info("=============================================")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    log.info("=== Sensex ORB starting | DRY_RUN=%s SMOKE_TEST=%s TRIGGER=%d LOTS=%d ===",
             DRY_RUN, SMOKE_TEST, TRIGGER_POINTS, LOTS)
    log.info("Now (IST): %s", now_ist().isoformat(timespec="seconds"))

    if SMOKE_TEST and not DRY_RUN:
        raise RuntimeError(
            "SMOKE_TEST=true requires DRY_RUN=true. Refusing to force-fire "
            "trades against real money."
        )

    groww = init_groww_client()
    log.info("Groww client initialised via TOTP")

    expiry = get_current_week_expiry(groww)
    lot_size = get_sensex_lot_size(groww, expiry)
    quantity = lot_size * LOTS
    log.info("Resolved expiry=%s lot_size=%d lots=%d quantity=%d",
             expiry, lot_size, LOTS, quantity)

    ctx: Dict[str, Any] = {
        "expiry": expiry,
        "lot_size": lot_size,
        "quantity": quantity,
        "open_price": None,
        "call_traded": False,
        "put_traded": False,
        "open_positions": {},
    }

    if SMOKE_TEST:
        run_smoke_test(groww, ctx)
        log_summary(ctx)
        return

    # 1. Wait for market open and capture open price
    wait_until(OPEN_TIME, "market open")
    open_price = get_sensex_ltp_with_retry(groww)
    ctx["open_price"] = open_price
    log.info(">>> Captured SENSEX open @ 09:15 = %.2f", open_price)

    # 2. Wait through the 15-minute observation window
    wait_until(OBSERVATION_END, "observation end")
    log.info(">>> Observation window ended; starting trigger watch")

    # 3. Trigger watch loop (until 10:00 IST or both sides traded)
    while True:
        cur = now_ist()
        if cur.time() >= TRIGGER_WINDOW_END:
            log.info("Trigger window closed (10:00 IST reached)")
            break
        if ctx["call_traded"] and ctx["put_traded"]:
            log.info("Both directions already traded; exiting trigger loop")
            break

        try:
            current = get_sensex_ltp(groww)
        except Exception as exc:
            log.warning("LTP fetch failed: %s; retrying in %ds",
                        exc, POLL_INTERVAL_SEC)
            time_module.sleep(POLL_INTERVAL_SEC)
            continue

        diff = current - open_price
        log.info("Tick sensex=%.2f open=%.2f diff=%+.2f call=%s put=%s",
                 current, open_price, diff,
                 ctx["call_traded"], ctx["put_traded"])

        if diff >= TRIGGER_POINTS and not ctx["call_traded"]:
            try:
                execute_trade(groww, ctx, "CE", current)
            except Exception as exc:
                log.exception("CE trade failed: %s", exc)
            ctx["call_traded"] = True

        if diff <= -TRIGGER_POINTS and not ctx["put_traded"]:
            try:
                execute_trade(groww, ctx, "PE", current)
            except Exception as exc:
                log.exception("PE trade failed: %s", exc)
            ctx["put_traded"] = True

        time_module.sleep(POLL_INTERVAL_SEC)

    # 4. Hard exit at 11:30 IST
    if ctx["open_positions"]:
        wait_until(HARD_EXIT_TIME, "hard exit")
        flat_close_all(groww, ctx)
    else:
        log.info("No positions opened; skipping hard-exit wait")

    log_summary(ctx)
    log.info("=== Sensex ORB run complete ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("Fatal error in main()")
        sys.exit(1)
