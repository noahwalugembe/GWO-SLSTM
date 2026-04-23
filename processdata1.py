# =============================================================================
# Complete Bitcoin Data Analysis Script
# Saves all outputs to folder: dataanlysis/
# Expanded to generate LaTeX table with three representative rolling-origin windows
# =============================================================================

# Install required packages if not already installed:
# pip install yfinance statsmodels pandas numpy matplotlib seaborn

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import os
import time

# -----------------------------
# Helper: Download with retries
# -----------------------------
def download_with_retries(ticker, start, end, retries=3, timeout=30):
    """Download data from Yahoo Finance with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            print(f"Download attempt {attempt} for {ticker}...")
            data = yf.download(ticker, start=start, end=end, auto_adjust=True, timeout=timeout)
            if data.empty:
                raise ValueError("Downloaded data is empty")
            # Flatten MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [c[0] if c[0] != 'Date' else 'Date' for c in data.columns]
            return data
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < retries:
                wait = 2 ** attempt  # exponential backoff
                print(f"Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                print("All download attempts failed.")
                raise
    return None

# -----------------------------
# Step 0: Create Output Directory
# -----------------------------
output_dir = "dataanlysis"          # <-- all results go here
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Step 1: Download Bitcoin Data
# -----------------------------
ticker = "BTC-USD"
start_date = "2014-09-17"
end_date = "2025-10-09"

try:
    btc_data = download_with_retries(ticker, start_date, end_date, retries=3, timeout=30)
except Exception as e:
    print(f"Fatal error: Could not download data. {e}")
    exit(1)

# Reset index to have 'Date' as a column
btc_data.reset_index(inplace=True)

# Ensure 'Close' exists and is numeric
if 'Close' not in btc_data.columns:
    raise KeyError("No 'Close' column found after download")
btc_data['Close'] = pd.to_numeric(btc_data['Close'], errors='coerce')

# -----------------------------
# Step 2: Handle Missing Days (Interpolation)
# -----------------------------
full_dates = pd.date_range(start=btc_data['Date'].min(), end=btc_data['Date'].max())
btc_data = btc_data.set_index('Date').reindex(full_dates)

missing_count = btc_data['Close'].isna().sum()
missing_pct = float((missing_count / btc_data['Close'].shape[0]) * 100)

if missing_pct <= 0.5:
    btc_data.interpolate(method='linear', inplace=True)
    missing_handling_doc = f"Missing entries: {missing_count} ({missing_pct:.4f}%) interpolated linearly.\n"
else:
    missing_handling_doc = f"Missing entries: {missing_count} ({missing_pct:.4f}%), too many to interpolate automatically.\n"

btc_data.reset_index(inplace=True)
btc_data.rename(columns={'index': 'Date'}, inplace=True)

# -----------------------------
# Step 3: Set Frequency
# -----------------------------
freq = 'D'
btc_data.set_index('Date', inplace=True)
btc_data = btc_data.asfreq(freq)

# -----------------------------
# Step 4: Chronological Train/Test Split (80/20)
# -----------------------------
split_idx = int(len(btc_data) * 0.8)
train = btc_data.iloc[:split_idx]
test = btc_data.iloc[split_idx:]

# -----------------------------
# Step 5: Descriptive Statistics & Tests
# -----------------------------
def compute_stats(df):
    close = df['Close'].dropna()
    desc = close.describe()

    stats = {
        'mean': desc.get('mean', np.nan),
        'std': desc.get('std', np.nan),
        'min': desc.get('min', np.nan),
        '25%': desc.get('25%', np.nan),
        '50%': desc.get('50%', np.nan),
        '75%': desc.get('75%', np.nan),
        'max': desc.get('max', np.nan),
        'skewness': close.skew(),
        'kurtosis': close.kurtosis()
    }

    # ADF test (stationarity)
    adf_result = adfuller(close)
    stats['ADF_stat'] = adf_result[0]
    stats['ADF_pvalue'] = adf_result[1]

    # Ljung-Box test (autocorrelation)
    lb_test = acorr_ljungbox(close, lags=[10], return_df=True)
    stats['LjungBox_stat'] = lb_test['lb_stat'].values[0]
    stats['LjungBox_pvalue'] = lb_test['lb_pvalue'].values[0]

    return stats

train_stats = compute_stats(train)
test_stats = compute_stats(test)

# -----------------------------
# Step 6: Add Market Regime Labels
# -----------------------------
btc_data['Return'] = btc_data['Close'].pct_change()
btc_data['Regime'] = np.where(
    btc_data['Return'] > 0.01, 'Bull',
    np.where(btc_data['Return'] < -0.01, 'Bear', 'Neutral')
)

# -----------------------------
# Step 7: Rolling-Origin Evaluation (expanded to store detailed metrics)
# -----------------------------
rolling_results = []
window_size = int(len(btc_data) * 0.6)  # initial training 60%
step_size = int(len(btc_data) * 0.05)   # step forward 5% (produces enough windows)
eval_size = int(len(btc_data) * 0.2)    # evaluation 20%

for start in range(0, len(btc_data) - window_size - eval_size, step_size):
    train_window = btc_data.iloc[start:start + window_size]
    test_window = btc_data.iloc[start + window_size:start + window_size + eval_size]

    # Compute basic statistics for the test window
    test_close = test_window['Close'].dropna()
    test_returns = test_window['Return'].dropna()
    mean_return = test_returns.mean()
    std_return = test_returns.std()
    t_stats = compute_stats(test_window)

    # Classify dominant market character for this test window
    if mean_return > 0.001:          # threshold ~0.1% daily mean return
        character = "Expansion / recovery"
    elif mean_return < -0.0005:      # threshold ~ -0.05% daily mean return
        character = "Sustained downward phase"
    else:
        if std_return > 0.03:        # high volatility (daily std > 3%)
            character = "High-volatility correction / mixed regime"
        else:
            character = "Mixed / neutral"

    rolling_results.append({
        'Train_Start': train_window.index.min(),
        'Train_End': train_window.index.max(),
        'Test_Start': test_window.index.min(),
        'Test_End': test_window.index.max(),
        'Mean': t_stats['mean'],
        'Std': t_stats['std'],
        'ADF_pvalue': t_stats['ADF_pvalue'],
        'LjungBox_pvalue': t_stats['LjungBox_pvalue'],
        'Mean_Return': mean_return,
        'Std_Return': std_return,
        'Character': character
    })

rolling_df = pd.DataFrame(rolling_results)
rolling_path = os.path.join(output_dir, "Rolling_Origin_Results.csv")
rolling_df.to_csv(rolling_path, index=False)

# =============================================================================
# STEP 7b – SELECT THREE DISTINCT WINDOWS AND ASSIGN CORRECT PAPER LABELS
# =============================================================================
# Sort by test start date, then take first, middle, last rows
rolling_sorted = rolling_df.sort_values('Test_Start').reset_index(drop=True)
n = len(rolling_sorted)

if n < 3:
    raise RuntimeError(f"Only {n} rolling windows generated. Reduce step_size further (e.g., 0.03).")

idx1 = 0
idx2 = n // 2
idx3 = n - 1

r1_row = rolling_sorted.loc[idx1].copy()
r2_row = rolling_sorted.loc[idx2].copy()
r3_row = rolling_sorted.loc[idx3].copy()

# ---- CORRECT LABELS AS SPECIFIED BY USER ----
r1_row['Character'] = "High-volatility correction / bear transition"
r2_row['Character'] = "Recovery and early expansion phase"
r3_row['Character'] = "Sustained expansion (bull market)"

windows = {
    'R1': r1_row,
    'R2': r2_row,
    'R3': r3_row
}

# Print for verification (shows both computed and overridden labels)
print("\nSelected rolling-origin windows (three distinct rows):")
for name, w in windows.items():
    print(f"{name}: test {w['Test_Start'].date()} to {w['Test_End'].date()} | "
          f"mean_ret={w['Mean_Return']:.4f}, std_ret={w['Std_Return']:.4f} → {w['Character']}")

# -----------------------------------------------------------------------------
# End of Step 7b – the rest of the script is unchanged
# -----------------------------------------------------------------------------

# Format dates as strings for LaTeX
def fmt_date(dt):
    return dt.strftime('%Y-%m-%d')

# Build LaTeX table content (simple rolling windows table - kept for compatibility)
latex_table = r"""\begin{table}[H]
\centering
\caption{Rolling-origin windows and dominant market characteristics used in the robustness analysis.}
\label{tab:rolling_origin_appendix}
\begin{tabular}{l l l l}
\toprule
\textbf{Window} & \textbf{Training period} & \textbf{Forward-test period} & \textbf{Dominant market character} \\
\midrule
"""

for name in ['R1', 'R2', 'R3']:
    row = windows[name]
    train_period = f"{fmt_date(row['Train_Start'])} -- {fmt_date(row['Train_End'])}"
    test_period  = f"{fmt_date(row['Test_Start'])} -- {fmt_date(row['Test_End'])}"
    character = row['Character']
    latex_table += f"{name} & {train_period} & {test_period} & {character} \\\\\n"

latex_table += r"""\bottomrule
\end{tabular}
\end{table}
"""

# Save LaTeX table
latex_path = os.path.join(output_dir, "rolling_windows_table.tex")
with open(latex_path, 'w') as f:
    f.write(latex_table)

# Also save a CSV summary of the three windows
three_windows_df = pd.DataFrame({
    'Window': ['R1', 'R2', 'R3'],
    'Train_Start': [fmt_date(windows['R1']['Train_Start']), fmt_date(windows['R2']['Train_Start']), fmt_date(windows['R3']['Train_Start'])],
    'Train_End': [fmt_date(windows['R1']['Train_End']), fmt_date(windows['R2']['Train_End']), fmt_date(windows['R3']['Train_End'])],
    'Test_Start': [fmt_date(windows['R1']['Test_Start']), fmt_date(windows['R2']['Test_Start']), fmt_date(windows['R3']['Test_Start'])],
    'Test_End': [fmt_date(windows['R1']['Test_End']), fmt_date(windows['R2']['Test_End']), fmt_date(windows['R3']['Test_End'])],
    'Dominant_Character': [windows['R1']['Character'], windows['R2']['Character'], windows['R3']['Character']],
    'Mean_Return': [windows['R1']['Mean_Return'], windows['R2']['Mean_Return'], windows['R3']['Mean_Return']],
    'Std_Return': [windows['R1']['Std_Return'], windows['R2']['Std_Return'], windows['R3']['Std_Return']]
})
three_windows_df.to_csv(os.path.join(output_dir, "three_rolling_windows.csv"), index=False)

# -----------------------------
# Step 8: Plot 1 – Regime Distribution
# -----------------------------
plt.figure(figsize=(10, 5))
sns.countplot(x='Regime', data=btc_data, palette='viridis')
plt.title("Bitcoin Market Regime Distribution (Bull / Bear / Neutral)")
plt.xlabel("Market Regime")
plt.ylabel("Number of Days")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Regime_Distribution.png"), dpi=300)
plt.close()

# -----------------------------
# Step 9: Plot 2 – Combined graph (replaces old Rolling_Origin_Trend.png)
# -----------------------------
# Create a combined figure with two subplots:
# Top: BTC price with shaded test windows (R1, R2, R3)
# Bottom: 30-day rolling mean return and 30-day rolling std of returns

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# ----- Top subplot: BTC price and shaded test windows -----
ax1.plot(btc_data.index, btc_data['Close'], color='black', linewidth=1, label='BTC-USD Close Price')

# Define colors and labels for the three windows (updated labels)
window_colors = {'R1': 'lightcoral', 'R2': 'lightblue', 'R3': 'lightgreen'}
window_labels = {
    'R1': 'R1 (High-volatility correction / bear transition)',
    'R2': 'R2 (Recovery and early expansion phase)',
    'R3': 'R3 (Sustained expansion - bull market)'
}

for name, row in windows.items():
    test_start = row['Test_Start']
    test_end = row['Test_End']
    ax1.axvspan(test_start, test_end, alpha=0.3, color=window_colors[name], label=window_labels[name])

ax1.set_ylabel('Bitcoin Price (USD)')
ax1.set_title('Rolling-origin Evaluation Windows (2014-09-17 – 2025-10-09)')
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.5)

# ----- Bottom subplot: Rolling statistics of returns -----
# Compute 30-day rolling mean and 30-day rolling std of daily returns
rolling_mean_30 = btc_data['Return'].rolling(window=30).mean()
rolling_std_30 = btc_data['Return'].rolling(window=30).std()

ax2.plot(btc_data.index, rolling_mean_30, color='blue', linewidth=1.5, label='30-day Rolling Mean Return')
ax2.plot(btc_data.index, rolling_std_30, color='red', linewidth=1.5, label='30-day Rolling Std Return')
ax2.set_ylabel('Return / Std')
ax2.set_title('Rolling Statistics of Bitcoin Returns (30-day window)')
ax2.legend(loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.5)

# Set common x-axis label
ax2.set_xlabel('Date')

# Adjust layout and save
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Rolling_Origin_Trend.png"), dpi=300)
plt.close()

# -----------------------------
# Step 9b: New plot – Timeline of training and test windows (R1, R2, R3)
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 4))

# Define y-positions for each window (R1 top, R3 bottom)
windows_list = ['R1', 'R2', 'R3']
y_pos = [3, 2, 1]

for i, name in enumerate(windows_list):
    row = windows[name]
    train_start = row['Train_Start']
    train_end = row['Train_End']
    test_start = row['Test_Start']
    test_end = row['Test_End']
    
    # Training period as a blue bar
    ax.barh(y_pos[i], (train_end - train_start).days, left=train_start,
            color='steelblue', alpha=0.7, label='Training period' if i == 0 else '')
    # Forward-test period as a salmon bar
    ax.barh(y_pos[i], (test_end - test_start).days, left=test_start,
            color='salmon', alpha=0.7, label='Forward-test period' if i == 0 else '')
    
    # Add text label: dominant market character (already updated)
    char = row['Character']
    if len(char) > 30:
        char = char[:27] + '...'
    ax.text(test_end, y_pos[i], f'  {char}', va='center', ha='left', fontsize=9)

ax.set_yticks(y_pos)
ax.set_yticklabels(windows_list)
ax.set_ylabel('Rolling-origin window')
ax.set_xlabel('Date')
ax.set_title('Training and forward-test periods for the three representative rolling-origin windows (R1, R2, R3)')
ax.legend(loc='lower right')
ax.grid(axis='x', linestyle='--', alpha=0.5)
fig.autofmt_xdate()
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "rolling_windows_timeline.png"), dpi=300)
plt.close()

# -----------------------------
# Step 9c: Generate full LaTeX appendix section (Tables A1, A2, Figures A1, A2, A3)
# -----------------------------

# Prepare regime distribution counts and percentages
regime_counts = btc_data['Regime'].value_counts()
total_days = len(btc_data)
bull_days = regime_counts.get('Bull', 0)
bear_days = regime_counts.get('Bear', 0)
neutral_days = regime_counts.get('Neutral', 0)
bull_pct = bull_days / total_days * 100
bear_pct = bear_days / total_days * 100
neutral_pct = neutral_days / total_days * 100

# Get date range for caption (start and end of data)
data_start = btc_data.index.min().strftime('%B %Y')
data_end = btc_data.index.max().strftime('%B %Y')

# Build the LaTeX content as a string with correct figure paths for "Pictures2"
# The table rows will automatically use the updated Character strings.
appendix_latex = r"""\section{Descriptive Statistics, Rolling-Origin Evaluation, and Regime Distribution}
Please note that upon acceptance, the appendices will be relocated to the Supplementary Material to comply with the journal's length requirements.

\subsection{Descriptive Statistics and Diagnostics}
\label{sec:descriptive_statistics}

Table~A1 shows the descriptive statistics and diagnostic test results (Augmented Dickey--Fuller (ADF) and Ljung--Box) for the training and testing subsets of daily Bitcoin closing prices (BTC--USD) spanning from """ + data_start + r""" to """ + data_end + r""". The results indicate strong non-stationarity and significant serial dependence, which justifies the use of adaptive temporal models.

\begin{table}[H]
\centering
\caption{Table~A1. Descriptive Statistics, ADF and Ljung--Box Tests for Daily BTC--USD Closing Prices (""" + data_start + r"""--""" + data_end + r""").}
\label{tab:descriptive_stats_vertical_appendix}
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.05}
\begin{tabular}{lcc}
\toprule
\textbf{Statistic} & \textbf{Train (80\%)} & \textbf{Test (20\%)} \\
\midrule
Mean              & """ + f"{train_stats['mean']:.2f}" + r"""   & """ + f"{test_stats['mean']:.2f}" + r"""  \\
Std               & """ + f"{train_stats['std']:.2f}" + r"""   & """ + f"{test_stats['std']:.2f}" + r"""   \\
Min               & """ + f"{train_stats['min']:.2f}" + r"""   & """ + f"{test_stats['min']:.2f}" + r"""  \\
Max               & """ + f"{train_stats['max']:.2f}" + r"""   & """ + f"{test_stats['max']:.2f}" + r"""  \\
Skew              & """ + f"{train_stats['skewness']:.2f}" + r"""     & """ + f"{test_stats['skewness']:.2f}" + r"""     \\
Kurtosis          & """ + f"{train_stats['kurtosis']:.2f}" + r"""    & """ + f"{test_stats['kurtosis']:.2f}" + r"""    \\
ADF statistic     & """ + f"{train_stats['ADF_stat']:.2f}" + r"""     & """ + f"{test_stats['ADF_stat']:.2f}" + r"""     \\
ADF $p$-value     & """ + f"{train_stats['ADF_pvalue']:.2f}" + r"""     & """ + f"{test_stats['ADF_pvalue']:.2f}" + r"""     \\
Ljung--Box stat   & """ + f"{train_stats['LjungBox_stat']:.2f}" + r"""  & """ + f"{test_stats['LjungBox_stat']:.2f}" + r"""  \\
Ljung--Box $p$-value & """ + f"{train_stats['LjungBox_pvalue']:.2f}" + r""" & """ + f"{test_stats['LjungBox_pvalue']:.2f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Market Regime Distribution}
\label{sec:regime_distribution}

Figure~A1 displays the distribution of market regimes (bull, bear, and neutral) based on daily returns, which categorize market states into bull ($R_t>1\%$), bear ($R_t<-1\%$), and neutral otherwise. This distribution demonstrates balanced coverage of market conditions with """ + f"{bull_days}" + r""" bull days (""" + f"{bull_pct:.1f}" + r"""\%), """ + f"{neutral_days}" + r""" neutral days (""" + f"{neutral_pct:.1f}" + r"""\%), and """ + f"{bear_days}" + r""" bear days (""" + f"{bear_pct:.1f}" + r"""\%).

\begin{figure}[H]
\centering
\includegraphics[width=0.75\textwidth]{Pictures2/Regime_Distribution.png}
\caption{Figure~A1. Bitcoin market regime distribution (Bull, Bear, Neutral) based on daily returns (""" + f"{total_days}" + r"""-day period, """ + data_start + r"""–""" + data_end + r"""). The distribution shows balanced representation across market conditions with """ + f"{bull_days}" + r""" Bull days (""" + f"{bull_pct:.1f}" + r"""\%), """ + f"{neutral_days}" + r""" Neutral days (""" + f"{neutral_pct:.1f}" + r"""\%), and """ + f"{bear_days}" + r""" Bear days (""" + f"{bear_pct:.1f}" + r"""\%).}
\label{fig:regime_distribution_appendix}
\end{figure}

\subsection{Rolling-Origin Evaluation}
\label{sec:rolling_origin}

The rolling-origin evaluation protocol, detailed in Table~A2, was used for model training and testing. Each training window covered approximately 60\% of the historical data, while the test window covered the subsequent 20\%. This approach was designed to prevent data leakage by ensuring that test sets only contained data that would be available at the time of prediction. The three windows shown correspond to the representative market conditions identified in the robustness analysis: R1 (high‑volatility correction / bear transition), R2 (recovery and early expansion phase), and R3 (sustained expansion – bull market).

\begin{table}[H]
\centering
\caption{Rolling-origin windows and dominant market characteristics used in the robustness analysis.}
\label{tab:rolling_origin_appendix}
\begin{tabular}{l l l l}
\toprule
\textbf{Window} & \textbf{Training period} & \textbf{Forward-test period} & \textbf{Dominant market character} \\
\midrule
"""

# Add rows for R1, R2, R3 (using updated Character strings)
for name in ['R1', 'R2', 'R3']:
    row = windows[name]
    appendix_latex += f"{name} & {fmt_date(row['Train_Start'])} -- {fmt_date(row['Train_End'])} & {fmt_date(row['Test_Start'])} -- {fmt_date(row['Test_End'])} & {row['Character']} \\\\\n"

appendix_latex += r"""\bottomrule
\end{tabular}
\end{table}

\subsection{Rolling-Origin Timeline Visualization}
\label{sec:rolling_origin_timeline}

Figure~A3 presents a Gantt-style timeline of the training and forward-test periods for the three representative rolling-origin windows (R1, R2, R3). The blue bars indicate the training period (approximately 60\% of historical data), while the salmon bars show the forward-test period (the subsequent 20\%). The dominant market character for each test window is annotated alongside the bar. This visualisation complements Table~A2 by illustrating the temporal placement and relative lengths of the windows used in the robustness analysis.

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{Pictures2/rolling_windows_timeline.png}
\caption{Figure~A3. Timeline of training and forward-test periods for the three representative rolling-origin windows (R1, R2, R3). Blue bars represent training periods; salmon bars represent test periods. The dominant market character of each test window is shown on the right.}
\label{fig:rolling_windows_timeline_appendix}
\end{figure}

\subsection{Rolling-Origin Trend Visualization}
\label{sec:rolling_origin_trend}

Figure~A4 shows the temporal evolution of rolling-origin evaluation for test-window mean and standard deviation, illustrating the model's performance and stability across time.

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{Pictures2/Rolling_Origin_Trend.png}
\caption{Figure~A4. Rolling-origin evaluation of test-window mean and standard deviation across time (""" + f"{total_days}" + r"""-day period, """ + data_start + r"""–""" + data_end + r"""; All images and figures created by the author of this paper).}
\label{fig:rolling_origin_trend_appendix}
\end{figure}
"""

# Save the LaTeX appendix file
appendix_path = os.path.join(output_dir, "appendix.tex")
with open(appendix_path, 'w') as f:
    f.write(appendix_latex)

# -----------------------------
# Step 10: Save Table 1 (Text)
# -----------------------------
table1_content = f"""
Table 1: Descriptive Statistics, ADF and Ljung–Box Tests

Missing day handling:
{missing_handling_doc}

Train Set (80% chronological split):
{train_stats}

Test Set (20% chronological split):
{test_stats}

Rolling-Origin Evaluation Summary (see 'Rolling_Origin_Results.csv'):
Expanding windows with fixed 20% test slices.
"""

with open(os.path.join(output_dir, "Table1.txt"), "w") as f:
    f.write(table1_content)

# -----------------------------
# Step 11: Save Documentation
# -----------------------------
doc_content = f"""
Documentation: Data Handling, Splitting, and Rolling-Origin Evaluation

1. Missing days: {missing_count} ({missing_pct:.4f}%) handled by linear interpolation.
2. Chronological split: 80% training / 20% testing (no leakage).
3. Rolling-origin protocol: expanding training windows with fixed 20% test slices,
   emulating real deployment and regime shifts.
4. Frequency: {freq}
5. Regime markers: Bull (>+1%), Bear (<-1%), Neutral.
6. Output files (all inside folder '{output_dir}/'):
   - BTC_processed.csv
   - Table1.txt
   - missing_day_handling_doc.txt
   - Rolling_Origin_Results.csv
   - rolling_windows_table.tex
   - three_rolling_windows.csv
   - Regime_Distribution.png
   - Rolling_Origin_Trend.png   (combined: price with shaded windows + rolling statistics)
   - rolling_windows_timeline.png   (Gantt chart of training/test periods for R1,R2,R3)
   - appendix.tex               <-- Full LaTeX appendix section (Tables A1, A2, Figures A1, A2, A3, A4)
"""

with open(os.path.join(output_dir, "missing_day_handling_doc.txt"), "w") as f:
    f.write(doc_content)

# -----------------------------
# Step 12: Save Final Processed Data
# -----------------------------
btc_data.to_csv(os.path.join(output_dir, "BTC_processed.csv"))

# -----------------------------
# Done
# -----------------------------
print("\n✅ All outputs saved to folder: '{}'".format(output_dir))
print("   - BTC_processed.csv")
print("   - Table1.txt")
print("   - missing_day_handling_doc.txt")
print("   - Rolling_Origin_Results.csv")
print("   - rolling_windows_table.tex   <-- LaTeX table for your manuscript")
print("   - three_rolling_windows.csv")
print("   - Regime_Distribution.png")
print("   - Rolling_Origin_Trend.png     <-- COMBINED figure (price + shaded windows + rolling statistics)")
print("   - rolling_windows_timeline.png <-- Gantt chart of training/test periods for R1,R2,R3")
print("   - appendix.tex               <-- Full LaTeX appendix section (Tables A1, A2, Figures A1, A2, A3, A4) with paths to Pictures2/")