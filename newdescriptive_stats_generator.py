import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.preprocessing import MinMaxScaler
import warnings
import os

warnings.filterwarnings('ignore')

# Create results directory
os.makedirs('descriptive_stats_results', exist_ok=True)

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def generate_descriptive_statistics():
    """
    Generate correct descriptive statistics for the 1,000-point subset (2014-2017)
    """
    print("Loading and processing Bitcoin data for 2014-2017 subset...")
    
    # Load your actual BTC-USD data
    df = pd.read_csv('TSF-BTC-LSTM-RNN-PSO-GWO-main/BTC-USD.csv')
    
    # Convert Date column to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    # Filter for exact 1000 points from 2014-09-17 to 2017-06-12
    start_date = '2014-09-17'
    end_date = '2017-06-12'
    df_subset = df[start_date:end_date]
    
    # Ensure exactly 1000 points
    if len(df_subset) > 1000:
        df_subset = df_subset.iloc[:1000]
    elif len(df_subset) < 1000:
        print(f"Warning: Only {len(df_subset)} points found in the specified date range")
    
    print(f"Dataset size: {len(df_subset)}")
    print(f"Date range: {df_subset.index[0].strftime('%Y-%m-%d')} to {df_subset.index[-1].strftime('%Y-%m-%d')}")
    
    # Calculate train-test split (80%-20%)
    train_size = int(len(df_subset) * 0.8)
    train_data = df_subset.iloc[:train_size]['Close']
    test_data = df_subset.iloc[train_size:]['Close']
    
    def calculate_stats(data, name):
        """Calculate comprehensive statistics for a dataset"""
        stats_dict = {}
        
        # Basic statistics
        stats_dict['Mean'] = np.mean(data)
        stats_dict['Std'] = np.std(data)
        stats_dict['Min'] = np.min(data)
        stats_dict['Max'] = np.max(data)
        stats_dict['Skew'] = stats.skew(data)
        stats_dict['Kurtosis'] = stats.kurtosis(data)
        
        # ADF test for stationarity
        adf_result = adfuller(data)
        stats_dict['ADF'] = adf_result[0]
        stats_dict['ADF-p'] = adf_result[1]
        
        # Ljung-Box test for autocorrelation
        lb_result = acorr_ljungbox(data, lags=10, return_df=True)
        stats_dict['Ljung-Box Stat'] = lb_result['lb_stat'].iloc[-1]
        stats_dict['Ljung-Box p'] = lb_result['lb_pvalue'].iloc[-1]
        
        return stats_dict
    
    # Calculate statistics for train and test sets
    train_stats = calculate_stats(train_data, "Train")
    test_stats = calculate_stats(test_data, "Test")
    
    return train_stats, test_stats, train_data, test_data, df_subset

def create_descriptive_table(train_stats, test_stats):
    """
    Create LaTeX code for descriptive statistics table
    """
    latex_table = """
\\begin{{table}}[H]
\\centering
\\caption{{Descriptive Statistics, ADF and Ljung--Box Tests for Daily BTC-USD Closing Prices (1,000-Point Subset, September 2014–June 2017)}}
\\label{{tab:descriptive_stats_vertical}}
\\begin{{tabular}}{{lcc}}
\\hline
\\textbf{{Subset}} & \\textbf{{Statistic}} & \\textbf{{Value}} \\\\
\\hline
\\multirow{{10}}{{*}}{{Train (80\\%)}} 
 & Mean & {train_mean:.2f} \\\\
 & Std & {train_std:.2f} \\\\
 & Min & {train_min:.2f} \\\\
 & Max & {train_max:.2f} \\\\
 & Skew & {train_skew:.2f} \\\\
 & Kurtosis & {train_kurtosis:.2f} \\\\
 & ADF & {train_adf:.2f} \\\\
 & ADF-$p$ & {train_adf_p:.2f} \\\\
 & Ljung--Box Stat & {train_lb:.2f} \\\\
 & Ljung--Box $p$ & {train_lb_p:.2f} \\\\
\\hline
\\multirow{{10}}{{*}}{{Test (20\\%)}} 
 & Mean & {test_mean:.2f} \\\\
 & Std & {test_std:.2f} \\\\
 & Min & {test_min:.2f} \\\\
 & Max & {test_max:.2f} \\\\
 & Skew & {test_skew:.2f} \\\\
 & Kurtosis & {test_kurtosis:.2f} \\\\
 & ADF & {test_adf:.2f} \\\\
 & ADF-$p$ & {test_adf_p:.2f} \\\\
 & Ljung--Box Stat & {test_lb:.2f} \\\\
 & Ljung--Box $p$ & {test_lb_p:.2f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
""".format(
        train_mean=train_stats['Mean'],
        train_std=train_stats['Std'],
        train_min=train_stats['Min'],
        train_max=train_stats['Max'],
        train_skew=train_stats['Skew'],
        train_kurtosis=train_stats['Kurtosis'],
        train_adf=train_stats['ADF'],
        train_adf_p=train_stats['ADF-p'],
        train_lb=train_stats['Ljung-Box Stat'],
        train_lb_p=train_stats['Ljung-Box p'],
        test_mean=test_stats['Mean'],
        test_std=test_stats['Std'],
        test_min=test_stats['Min'],
        test_max=test_stats['Max'],
        test_skew=test_stats['Skew'],
        test_kurtosis=test_stats['Kurtosis'],
        test_adf=test_stats['ADF'],
        test_adf_p=test_stats['ADF-p'],
        test_lb=test_stats['Ljung-Box Stat'],
        test_lb_p=test_stats['Ljung-Box p']
    )
    
    return latex_table

def create_rolling_origin_table():
    """
    Create LaTeX code for rolling origin evaluation table
    """
    latex_table = """
\\begin{{table}}[H]
\\centering
\\caption{{Rolling-Origin Evaluation Windows (1,000-Point Subset, September 2014–June 2017)}}
\\label{{tab:rolling_origin}}
\\begin{{tabular}}{{cccc}}
\\hline
\\textbf{{Train Start}} & \\textbf{{Train End}} & \\textbf{{Test Start}} & \\textbf{{Test End}} \\\\
\\hline
2014-09-17 & 2016-05-15 & 2016-05-16 & 2016-12-31 \\\\
2015-03-17 & 2016-11-15 & 2016-11-16 & 2017-04-30 \\\\
2015-09-17 & 2017-01-09 & 2017-01-10 & 2017-06-12 \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    return latex_table

def plot_regime_distribution(df_subset, save_path='descriptive_stats_results/Regime_Distribution.png'):
    """
    Plot market regime distribution based on daily returns
    """
    # Calculate daily returns
    returns = df_subset['Close'].pct_change() * 100
    
    # Define regimes
    bull_mask = returns > 1
    bear_mask = returns < -1
    neutral_mask = ~(bull_mask | bear_mask)
    
    # Count regimes
    regime_counts = {
        'Bull (Return > +1%)': bull_mask.sum(),
        'Neutral (-1% ≤ Return ≤ +1%)': neutral_mask.sum(),
        'Bear (Return < -1%)': bear_mask.sum()
    }
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
    bars = ax.bar(regime_counts.keys(), regime_counts.values(), color=colors, alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}', ha='center', va='bottom', fontsize=12)
    
    ax.set_ylabel('Number of Days', fontsize=12)
    ax.set_title('Bitcoin Market Regime Distribution (2014-2017)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add summary statistics - FIXED: Use total dataset length instead of returns length
    total_days = len(df_subset)  # This should be 1000
    bull_percent = (bull_mask.sum() / total_days) * 100
    neutral_percent = (neutral_mask.sum() / total_days) * 100
    bear_percent = (bear_mask.sum() / total_days) * 100
    
    ax.text(0.02, 0.98, f'Total Trading Days: {total_days}\n'
                        f'Bull Days: {bull_percent:.1f}%\n'
                        f'Neutral Days: {neutral_percent:.1f}%\n'
                        f'Bear Days: {bear_percent:.1f}%',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return regime_counts

def plot_rolling_origin_trend(df_subset, save_path='descriptive_stats_results/Rolling_Origin_Trend.png'):
    """
    Plot rolling origin evaluation trend
    """
    # Define rolling windows for 2014-2017 period
    windows = [
        {'train_start': '2014-09-17', 'train_end': '2016-05-15', 'test_start': '2016-05-16', 'test_end': '2016-12-31'},
        {'train_start': '2015-03-17', 'train_end': '2016-11-15', 'test_start': '2016-11-16', 'test_end': '2017-04-30'},
        {'train_start': '2015-09-17', 'train_end': '2017-01-09', 'test_start': '2017-01-10', 'test_end': '2017-06-12'}
    ]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot price series with windows
    ax1.plot(df_subset.index, df_subset['Close'], linewidth=1, color='black', alpha=0.7, label='BTC Price')
    
    colors = ['red', 'blue', 'green']
    for i, window in enumerate(windows):
        # Highlight test periods
        test_period = df_subset[window['test_start']:window['test_end']]
        ax1.fill_between(test_period.index, test_period['Close'], 
                        alpha=0.3, color=colors[i], label=f'Test Window {i+1}')
    
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.set_title('Rolling-Origin Evaluation Windows (2014-2017)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot rolling statistics
    returns = df_subset['Close'].pct_change()
    rolling_mean = returns.rolling(window=30).mean() * 100  # 30-day rolling mean return
    rolling_std = returns.rolling(window=30).std() * 100   # 30-day rolling std
    
    ax2.plot(df_subset.index, rolling_mean, label='30-day Rolling Mean Return (%)', color='blue')
    ax2.plot(df_subset.index, rolling_std, label='30-day Rolling Std Dev (%)', color='red')
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_title('Rolling Statistics of Bitcoin Returns', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to generate all descriptive statistics and figures
    """
    print("Generating Descriptive Statistics and Figures for Paper...")
    print("=" * 60)
    
    # Generate descriptive statistics
    train_stats, test_stats, train_data, test_data, df_subset = generate_descriptive_statistics()
    
    # Print statistics summary
    print("\n" + "="*60)
    print("DESCRIPTIVE STATISTICS SUMMARY")
    print("="*60)
    print(f"Training Set ({len(train_data)} points):")
    print(f"  Mean: ${train_stats['Mean']:.2f}, Std: ${train_stats['Std']:.2f}")
    print(f"  Range: ${train_stats['Min']:.2f} - ${train_stats['Max']:.2f}")
    print(f"  ADF p-value: {train_stats['ADF-p']:.4f} ({'Non-stationary' if train_stats['ADF-p'] > 0.05 else 'Stationary'})")
    
    print(f"\nTest Set ({len(test_data)} points):")
    print(f"  Mean: ${test_stats['Mean']:.2f}, Std: ${test_stats['Std']:.2f}")
    print(f"  Range: ${test_stats['Min']:.2f} - ${test_stats['Max']:.2f}")
    print(f"  ADF p-value: {test_stats['ADF-p']:.4f} ({'Non-stationary' if test_stats['ADF-p'] > 0.05 else 'Stationary'})")
    
    # Generate LaTeX tables
    descriptive_table = create_descriptive_table(train_stats, test_stats)
    rolling_origin_table = create_rolling_origin_table()
    
    # Save tables to files
    with open('descriptive_stats_results/descriptive_stats_table.tex', 'w') as f:
        f.write(descriptive_table)
    
    with open('descriptive_stats_results/rolling_origin_table.tex', 'w') as f:
        f.write(rolling_origin_table)
    
    print("\n" + "="*60)
    print("LATEX TABLES GENERATED")
    print("="*60)
    print("✓ descriptive_stats_table.tex")
    print("✓ rolling_origin_table.tex")
    
    # Generate plots
    print("\n" + "="*60)
    print("GENERATING FIGURES")
    print("="*60)
    
    regime_counts = plot_regime_distribution(df_subset)
    print("✓ Regime Distribution Plot")
    
    plot_rolling_origin_trend(df_subset)
    print("✓ Rolling Origin Trend Plot")
    
    # Print final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Dataset: {len(df_subset)} Bitcoin daily prices (2014-2017)")
    print(f"Training set: {len(train_data)} points (80%)")
    print(f"Test set: {len(test_data)} points (20%)")
    print(f"Market regimes: {regime_counts}")
    print("\nAll files generated successfully!")
    print("Use the .tex files to update your paper tables.")
    print("Use the .png files for your paper figures.")

if __name__ == "__main__":
    main()