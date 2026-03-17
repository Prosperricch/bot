"""
USD/JPY Telegram Signal Bot
Connects to live price via yfinance (no MT5 needed for hosting)
Detects S&R zones, scores confidence, calculates SL/TP/risk
Sends signals to Telegram only when confidence >= threshold
"""

import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# ============================================================
#  YOUR SETTINGS — fill these in
# ============================================================
TELEGRAM_TOKEN   = "8336577956:AAHtWI2tsC1o0Ghyd0KdgUyJzp66dNF_tpo"       # from BotFather
TELEGRAM_CHAT_ID = "7537945961"         # from @userinfobot
SYMBOL           = "USDJPY=X"                  # yfinance symbol
ACCOUNT_BALANCE  = 10000                       # your account size in USD
RISK_PERCENT     = 1.0                         # % of account to risk per trade
MIN_CONFIDENCE   = 70                          # only send if score >= this
CHECK_INTERVAL   = 60                          # seconds between checks
MIN_RR_RATIO     = 1.5                         # minimum reward:risk ratio
# ============================================================

last_signal_price = None
last_signal_time  = None
COOLDOWN_MINUTES  = 60  # don't re-alert same zone within this time


def get_price_data(symbol, period="5d", interval="1h"):
    """Fetch OHLCV data from Yahoo Finance"""
    try:
        df = yf.download(symbol, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty:
            return None
        df.columns = [c.lower() for c in df.columns]
        return df.reset_index()
    except Exception as e:
        print(f"[ERROR] Price fetch failed: {e}")
        return None


def get_atr(df, period=14):
    """Average True Range"""
    high  = df['high'].values
    low   = df['low'].values
    close = df['close'].values
    tr    = np.maximum(high[1:] - low[1:],
            np.maximum(abs(high[1:] - close[:-1]),
                       abs(low[1:]  - close[:-1])))
    atr = np.convolve(tr, np.ones(period)/period, mode='valid')
    return float(atr[-1]) if len(atr) > 0 else 0.5


def find_swing_highs_lows(df, lookback=5):
    """
    swing high: high[i] > high[i-1] AND high[i] > high[i+1]
    swing low:  low[i]  < low[i-1]  AND low[i]  < low[i+1]
    Extended to check 'lookback' bars each side for stronger confirmation
    """
    highs_list = []
    lows_list  = []
    highs = df['high'].values
    lows  = df['low'].values

    for i in range(lookback, len(df) - lookback):
        is_swing_high = all(highs[i] > highs[i-j] and
                            highs[i] > highs[i+j] for j in range(1, lookback+1))
        is_swing_low  = all(lows[i]  < lows[i-j]  and
                            lows[i]  < lows[i+j]  for j in range(1, lookback+1))
        if is_swing_high:
            highs_list.append(highs[i])
        if is_swing_low:
            lows_list.append(lows[i])

    return highs_list, lows_list


def cluster_zones(levels, atr, zone_atr_mult=0.8):
    """Merge levels within ATR * mult into a single zone"""
    if not levels:
        return []
    threshold = atr * zone_atr_mult
    sorted_levels = sorted(levels)
    zones = []
    cluster = [sorted_levels[0]]

    for level in sorted_levels[1:]:
        if level - max(cluster) < threshold:
            cluster.append(level)
        else:
            zones.append({
                'price':    sum(cluster) / len(cluster),
                'high':     max(cluster) + atr * 0.3,
                'low':      min(cluster) - atr * 0.3,
                'strength': len(cluster)
            })
            cluster = [level]

    zones.append({
        'price':    sum(cluster) / len(cluster),
        'high':     max(cluster) + atr * 0.3,
        'low':      min(cluster) - atr * 0.3,
        'strength': len(cluster)
    })
    return zones


def detect_candlestick_pattern(df):
    """Returns pattern name and bullish/bearish flag"""
    if len(df) < 3:
        return None, None

    o  = float(df['open'].iloc[-2])
    h  = float(df['high'].iloc[-2])
    l  = float(df['low'].iloc[-2])
    c  = float(df['close'].iloc[-2])
    o1 = float(df['open'].iloc[-3])
    h1 = float(df['high'].iloc[-3])
    l1 = float(df['low'].iloc[-3])
    c1 = float(df['close'].iloc[-3])

    body      = abs(c - o)
    rng       = h - l
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    if rng == 0:
        return None, None

    # Doji
    if body < rng * 0.1:
        return "Doji", None
    # Hammer
    if lower_wick > body * 2 and upper_wick < body * 0.5:
        return "Hammer", True
    # Shooting Star
    if upper_wick > body * 2 and lower_wick < body * 0.5:
        return "Shooting Star", False
    # Bullish Engulfing
    if c > o and c1 < o1 and o < c1 and c > o1:
        return "Bullish Engulfing", True
    # Bearish Engulfing
    if c < o and c1 > o1 and o > c1 and c < o1:
        return "Bearish Engulfing", False
    # Piercing Line
    if c1 < o1 and c > o and c > (o1 + c1) / 2 and c < o1:
        return "Piercing Line", True
    # Dark Cloud Cover
    if c1 > o1 and c < o and c < (o1 + c1) / 2 and c > c1:
        return "Dark Cloud Cover", False

    return None, None


def get_mtf_bias(symbol):
    """
    Check 1H, 4H, 1D timeframes for EMA9 vs EMA21 alignment
    Returns score: +1 per bullish TF, -1 per bearish TF
    """
    intervals = {'1h': '5d', '4h': '60d', '1d': '60d'}
    score = 0
    details = {}

    for interval, period in intervals.items():
        try:
            df = yf.download(symbol, period=period, interval=interval,
                             progress=False, auto_adjust=True)
            if len(df) < 22:
                continue
            close = df['Close'].values
            ema9  = pd.Series(close).ewm(span=9,  adjust=False).mean().values
            ema21 = pd.Series(close).ewm(span=21, adjust=False).mean().values
            if ema9[-1] > ema21[-1] and close[-1] > ema9[-1]:
                score += 1
                details[interval] = "BUY"
            elif ema9[-1] < ema21[-1] and close[-1] < ema9[-1]:
                score -= 1
                details[interval] = "SELL"
            else:
                details[interval] = "NEUT"
        except:
            details[interval] = "N/A"

    return score, details


def calculate_confidence(zone, current_price, atr, pattern_name,
                          pattern_bull, mtf_score, is_support):
    """
    Score 0-100 based on:
    - Zone strength (how many swings clustered)   → up to 30 pts
    - MTF alignment                               → up to 30 pts
    - Candlestick pattern confirmation            → up to 25 pts
    - How clean the price is to the zone centre   → up to 15 pts
    """
    score = 0
    reasons = []

    # 1. Zone strength (max 30)
    strength_pts = min(zone['strength'] * 10, 30)
    score += strength_pts
    reasons.append(f"Zone strength {zone['strength']}* (+{strength_pts})")

    # 2. MTF alignment (max 30)
    direction = is_support  # True = looking for bounce up
    if direction and mtf_score > 0:
        mtf_pts = min(mtf_score * 10, 30)
        score += mtf_pts
        reasons.append(f"MTF aligned bullish (+{mtf_pts})")
    elif not direction and mtf_score < 0:
        mtf_pts = min(abs(mtf_score) * 10, 30)
        score += mtf_pts
        reasons.append(f"MTF aligned bearish (+{mtf_pts})")
    else:
        reasons.append("MTF mixed (+0)")

    # 3. Candlestick confirmation (max 25)
    if pattern_name:
        if pattern_bull is None:
            score += 10
            reasons.append(f"Pattern: {pattern_name} (+10)")
        elif pattern_bull == is_support:
            score += 25
            reasons.append(f"Pattern confirms: {pattern_name} (+25)")
        else:
            score -= 10
            reasons.append(f"Pattern contradicts: {pattern_name} (-10)")

    # 4. Price proximity to zone centre (max 15)
    dist = abs(current_price - zone['price'])
    proximity = max(0, 15 - int((dist / atr) * 15))
    score += proximity
    reasons.append(f"Proximity to zone (+{proximity})")

    return max(0, min(100, score)), reasons


def calculate_trade_params(current_price, atr, is_support, account_balance, risk_pct):
    """Calculate SL, TP, position size, risk amount, potential profit"""
    sl_buffer = atr * 1.5

    if is_support:
        direction  = "BUY"
        stop_loss  = round(current_price - sl_buffer, 3)
        take_profit_1 = round(current_price + atr * 2.0, 3)
        take_profit_2 = round(current_price + atr * 3.5, 3)
    else:
        direction  = "SELL"
        stop_loss  = round(current_price + sl_buffer, 3)
        take_profit_1 = round(current_price - atr * 2.0, 3)
        take_profit_2 = round(current_price - atr * 3.5, 3)

    sl_pips       = abs(current_price - stop_loss)
    tp1_pips      = abs(current_price - take_profit_1)
    tp2_pips      = abs(current_price - take_profit_2)
    rr_ratio_1    = round(tp1_pips / sl_pips, 2) if sl_pips > 0 else 0
    rr_ratio_2    = round(tp2_pips / sl_pips, 2) if sl_pips > 0 else 0

    risk_amount   = round(account_balance * (risk_pct / 100), 2)
    potential_1   = round(risk_amount * rr_ratio_1, 2)
    potential_2   = round(risk_amount * rr_ratio_2, 2)

    # Standard lot size calc (approx for JPY pairs, 1 pip ≈ $9 per 0.01 lot)
    pip_value     = 0.09  # per 0.01 lot for USDJPY approx
    lot_size      = round(risk_amount / (sl_pips * 10000 * pip_value), 2)
    lot_size      = max(0.01, min(lot_size, 10.0))

    return {
        'direction':     direction,
        'entry':         round(current_price, 3),
        'stop_loss':     stop_loss,
        'take_profit_1': take_profit_1,
        'take_profit_2': take_profit_2,
        'rr_1':          rr_ratio_1,
        'rr_2':          rr_ratio_2,
        'sl_pips':       round(sl_pips * 100, 1),
        'tp1_pips':      round(tp1_pips * 100, 1),
        'risk_amount':   risk_amount,
        'potential_1':   potential_1,
        'potential_2':   potential_2,
        'lot_size':      lot_size,
    }


def format_telegram_message(zone, trade, confidence, reasons,
                             pattern_name, mtf_details, is_support):
    """Format the Telegram alert message"""
    emoji_dir  = "LONG" if is_support else "SHORT"
    zone_type  = "SUPPORT" if is_support else "RESISTANCE"
    stars      = "*" * min(zone['strength'], 5)
    conf_bar   = "".join(["█" if i < confidence//10 else "░" for i in range(10)])
    time_str   = datetime.utcnow().strftime("%H:%M UTC")

    mtf_lines = ""
    for tf, sig in mtf_details.items():
        icon = "▲" if sig == "BUY" else "▼" if sig == "SELL" else "—"
        mtf_lines += f"  {tf.upper()}: {icon} {sig}\n"

    msg = f"""
📊 *USD/JPY SIGNAL — {emoji_dir}*
🕐 {time_str}

*Zone:* {zone_type} @ {zone['price']:.3f} {stars}
*Entry:* {trade['entry']:.3f}
*Direction:* {trade['direction']}

🎯 *Targets*
  TP1: {trade['take_profit_1']:.3f}  ({trade['tp1_pips']} pips | R:R {trade['rr_1']}x)
  TP2: {trade['take_profit_2']:.3f}  (R:R {trade['rr_2']}x)
  SL:  {trade['stop_loss']:.3f}  ({trade['sl_pips']} pips)

💰 *Money*
  Risk:     ${trade['risk_amount']} ({RISK_PERCENT}% of account)
  Profit T1: +${trade['potential_1']}
  Profit T2: +${trade['potential_2']}
  Lot size: {trade['lot_size']}

📈 *Confidence: {confidence}%*
{conf_bar}
{chr(10).join(f"  • {r}" for r in reasons)}

🕯 *Pattern:* {pattern_name if pattern_name else 'None detected'}

🔀 *MTF Alignment*
{mtf_lines}
⚠️ _Not financial advice. Always use your own judgement._
""".strip()

    return msg


def send_telegram(token, chat_id, message):
    """Send message to Telegram"""
    url  = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {
        "chat_id":    chat_id,
        "text":       message,
        "parse_mode": "Markdown"
    }
    try:
        r = requests.post(url, data=data, timeout=10)
        if r.status_code == 200:
            print(f"[OK] Telegram message sent")
            return True
        else:
            print(f"[ERROR] Telegram: {r.status_code} {r.text}")
            return False
    except Exception as e:
        print(f"[ERROR] Telegram send failed: {e}")
        return False


def is_in_cooldown(price, zone_price):
    """Avoid spamming same zone repeatedly"""
    global last_signal_price, last_signal_time
    if last_signal_time is None or last_signal_price is None:
        return False
    mins_elapsed = (time.time() - last_signal_time) / 60
    if mins_elapsed < COOLDOWN_MINUTES and abs(price - last_signal_price) < 0.3:
        return True
    return False


def run_bot():
    global last_signal_price, last_signal_time

    print(f"[START] USD/JPY Signal Bot running. Min confidence: {MIN_CONFIDENCE}%")
    send_telegram(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
                  "USD/JPY Signal Bot started. Monitoring key levels...")

    while True:
        try:
            print(f"\n[{datetime.utcnow().strftime('%H:%M:%S')}] Checking market...")

            # 1. Get price data
            df = get_price_data(SYMBOL, period="10d", interval="1h")
            if df is None or len(df) < 30:
                print("[WARN] Not enough data, skipping.")
                time.sleep(CHECK_INTERVAL)
                continue

            current_price = float(df['close'].iloc[-1])
            atr           = get_atr(df)
            print(f"  Price: {current_price:.3f}  ATR: {atr:.3f}")

            # 2. Find swings and build zones
            swing_highs, swing_lows = find_swing_highs_lows(df, lookback=5)
            resist_zones = cluster_zones(swing_highs, atr)
            support_zones = cluster_zones(swing_lows,  atr)
            print(f"  Zones — Support: {len(support_zones)}, Resistance: {len(resist_zones)}")

            # 3. Check if price is near any zone
            alert_dist = atr * 1.5
            zones_to_check = [(z, True) for z in support_zones] + \
                             [(z, False) for z in resist_zones]

            for zone, is_support in zones_to_check:
                dist = abs(current_price - zone['price'])
                if dist > alert_dist:
                    continue

                if is_in_cooldown(current_price, zone['price']):
                    print(f"  [COOLDOWN] Zone {zone['price']:.3f} skipped")
                    continue

                print(f"  [HIT] {'Support' if is_support else 'Resistance'} @ {zone['price']:.3f}")

                # 4. Get MTF bias
                mtf_score, mtf_details = get_mtf_bias(SYMBOL)

                # 5. Detect pattern
                pattern_name, pattern_bull = detect_candlestick_pattern(df)

                # 6. Calculate confidence
                confidence, reasons = calculate_confidence(
                    zone, current_price, atr,
                    pattern_name, pattern_bull,
                    mtf_score, is_support
                )
                print(f"  Confidence: {confidence}%")

                # 7. Check minimum R:R
                trade = calculate_trade_params(
                    current_price, atr, is_support,
                    ACCOUNT_BALANCE, RISK_PERCENT
                )

                if trade['rr_1'] < MIN_RR_RATIO:
                    print(f"  [SKIP] R:R too low ({trade['rr_1']}x < {MIN_RR_RATIO}x)")
                    continue

                # 8. Send only if confidence meets threshold
                if confidence >= MIN_CONFIDENCE:
                    msg = format_telegram_message(
                        zone, trade, confidence, reasons,
                        pattern_name, mtf_details, is_support
                    )
                    sent = send_telegram(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, msg)
                    if sent:
                        last_signal_price = current_price
                        last_signal_time  = time.time()
                else:
                    print(f"  [SKIP] Confidence {confidence}% below threshold {MIN_CONFIDENCE}%")

        except Exception as e:
            print(f"[ERROR] Main loop: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    run_bot()
