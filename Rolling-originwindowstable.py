#!/usr/bin/env python3
"""
Robust rolling-origin window selection for Bitcoin.
Uses 30-day rolling metrics and 200-day MA to define three distinct market regimes.
Output: LaTeX table with training and forward-test periods.
"""
import pandas as pd
import numpy as np
from datetime import timedelta

CSV_FILE = "btcusd_2014_2026.csv"
OUTPUT_FILE = "rolling_windows_table.txt"
TRAINING_DAYS = 500
ROLLING_WINDOW = 30          # 30 days to capture sustained phases
MIN_TEST_DAYS = 60
MAX_TEST_DAYS = 200

# Regime definitions (simpler, more robust)
VOL_THRESH = 0.04            # 4% daily volatility for "high-vol"
RETURN_BOUND = 0.002         # |return| < 0.2% per day for high-vol correction

# Load data
df = pd.read_csv(CSV_FILE, parse_dates=['Date'], index_col='Date', dayfirst=True)
df = df.sort_index()
prices = df['Close'].copy()

# Compute rolling metrics
returns = prices.pct_change().dropna()
rolling_ret = returns.rolling(ROLLING_WINDOW).mean()
rolling_vol = returns.rolling(ROLLING_WINDOW).std()
long_ma = prices.rolling(200, min_periods=50).mean()

# Align valid data (remove NaNs from the first 200 days)
valid_idx = rolling_ret.dropna().index.intersection(rolling_vol.dropna().index).intersection(long_ma.dropna().index)
rolling_ret = rolling_ret[valid_idx]
rolling_vol = rolling_vol[valid_idx]
long_ma = long_ma[valid_idx]
prices_aligned = prices[valid_idx]

print(f"Data range: {prices_aligned.index[0].date()} to {prices_aligned.index[-1].date()}, {len(prices_aligned)} days")

# Classify regimes
regime = pd.Series(0, index=prices_aligned.index)
# Expansion: price above 200-day MA AND positive 30-day return
exp_cond = (prices_aligned > long_ma) & (rolling_ret > 0)
# Downward: price below 200-day MA AND negative 30-day return
down_cond = (prices_aligned < long_ma) & (rolling_ret < 0)
# High-volatility correction: high vol and low magnitude return
highvol_cond = (rolling_vol > VOL_THRESH) & (np.abs(rolling_ret) < RETURN_BOUND)

regime[down_cond] = 2
regime[highvol_cond & (regime == 0)] = 3
regime[exp_cond & (regime == 0)] = 1

# Count days per regime
for lbl, name in [(1, 'Expansion'), (2, 'Downward'), (3, 'High-vol')]:
    print(f"{name}: {sum(regime==lbl)} days")

# Find contiguous segments of minimum length MIN_TEST_DAYS
def get_contiguous_segments(regime_series, label, min_len, max_len):
    mask = (regime_series == label)
    start_mask = mask & ~mask.shift(1).fillna(False)
    end_mask = mask & ~mask.shift(-1).fillna(False)
    start_dates = mask.index[start_mask]
    end_dates = mask.index[end_mask]
    segments = []
    for s, e in zip(start_dates, end_dates):
        length = (e - s).days
        if min_len <= length <= max_len:
            segments.append((s, e, length))
    # Sort by length descending
    segments.sort(key=lambda x: x[2], reverse=True)
    return segments

# Find candidates for each regime
segments_by_label = {}
for lbl in [1,2,3]:
    segs = get_contiguous_segments(regime, lbl, MIN_TEST_DAYS, MAX_TEST_DAYS)
    segments_by_label[lbl] = segs
    print(f"Regime {lbl}: {len(segs)} candidate segments (longest = {segs[0][2] if segs else 0} days)")

# Select non-overlapping windows (longest for each regime)
selected = {}
candidates = []
for lbl, segs in segments_by_label.items():
    if segs:
        candidates.append((segs[0][0], segs[0][1], lbl, segs[0][2]))
candidates.sort(key=lambda x: x[0])   # sort by start date

used_end = None
for s, e, lbl, length in candidates:
    if used_end is not None and s <= used_end:
        continue
    selected[lbl] = (s, e)
    used_end = e
    if len(selected) == 3:
        break

# If still missing regimes, try next longest for missing ones
if len(selected) < 3:
    print("\n⚠️ Not all regimes found non-overlapping. Trying next longest candidates...")
    for lbl in [1,2,3]:
        if lbl in selected:
            continue
        for s, e, length in segments_by_label[lbl]:
            overlap = False
            for s2, e2 in selected.values():
                if not (e < s2 or s > e2):
                    overlap = True
                    break
            if not overlap:
                selected[lbl] = (s, e)
                break

if len(selected) != 3:
    print("\n❌ Could not automatically find three non-overlapping windows.")
    print("Please use the manual script provided earlier (with your own date ranges).")
    exit(1)

# Order chronologically: R1 earliest, R2 middle, R3 latest
ordered = []
for lbl, (s, e) in selected.items():
    ordered.append((s, e, lbl))
ordered.sort(key=lambda x: x[0])

result = {}
for idx, (s, e, lbl) in enumerate(ordered, 1):
    name = f"R{idx}"
    desc = {1: "Expansion / recovery", 2: "Sustained downward phase", 3: "High-volatility correction / mixed regime"}[lbl]
    result[name] = {'test_start': s, 'test_end': e, 'character': desc}

# Add training periods
total_start = prices_aligned.index.min()
for name, info in result.items():
    test_start = info['test_start']
    train_end = test_start - timedelta(days=1)
    train_start = train_end - timedelta(days=TRAINING_DAYS)
    if train_start < total_start:
        train_start = total_start
    info['train_start'] = train_start
    info['train_end'] = train_end
    print(f"\n{name}: {info['character']}")
    print(f"  Training : {train_start.strftime('%Y-%m-%d')} – {train_end.strftime('%Y-%m-%d')} ({(train_end - train_start).days} days)")
    print(f"  Forward-test: {test_start.strftime('%Y-%m-%d')} – {e.strftime('%Y-%m-%d')} ({(e - test_start).days} days)")

# Write LaTeX table
with open(OUTPUT_FILE, 'w') as f:
    f.write("\\begin{tabular}{l l l l}\n\\toprule\n")
    f.write("\\textbf{Window} & \\textbf{Training period} & \\textbf{Forward-test period} & \\textbf{Dominant market character} \\\\\n\\midrule\n")
    for name in ['R1', 'R2', 'R3']:
        info = result[name]
        train_str = f"{info['train_start'].strftime('%Y-%m-%d')} – {info['train_end'].strftime('%Y-%m-%d')}"
        test_str = f"{info['test_start'].strftime('%Y-%m-%d')} – {info['test_end'].strftime('%Y-%m-%d')}"
        f.write(f"{name} & {train_str} & {test_str} & {info['character']} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

print(f"\n✅ LaTeX table written to {OUTPUT_FILE}")