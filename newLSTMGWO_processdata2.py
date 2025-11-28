import os
import random
import numpy as np
import torch
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import math
import time  # Added for time complexity analysis

# -----------------------------
# Reproducibility setup
# -----------------------------
SEED = 25
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Function to save figure data
def save_figure_data(figure_name, description, data_dict, results_dir):
    """
    Save values and descriptions for each figure
    """
    data_file = os.path.join(results_dir, f"{figure_name}_data.txt")
    with open(data_file, "w") as f:
        f.write(f"Figure: {figure_name}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Description: {description}\n")
        f.write("=" * 50 + "\n\n")
        f.write("Values used in the figure:\n")
        f.write("-" * 30 + "\n")
        
        for key, value in data_dict.items():
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value}\n")
            elif isinstance(value, (list, np.ndarray)):
                f.write(f"{key}: {list(value[:10])} ... (showing first 10 of {len(value)} values)\n")
            else:
                f.write(f"{key}: {value}\n")

# Load the dataset
dataset = pd.read_csv('TSF-BTC-LSTM-RNN-PSO-GWO-main/BTC-USD.csv')
dataset = dataset[:1000]  # Reduce dataset to 1000 values

# Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset['Close'].values.reshape(-1, 1))

# Define time steps FIRST
time_steps = 30  # MOVED THIS LINE UP

# Split data into train and test sets - FIXED with time_steps defined
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size-time_steps:]  # NOW time_steps is defined

# Function to create input features and target variable
def create_dataset(data, time_steps=1):
    X, Y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        Y.append(data[i + time_steps, 0])
    return np.array(X), np.array(Y)

# Create train/test datasets
X_train, Y_train = create_dataset(train_data, time_steps)
X_test, Y_test = create_dataset(test_data, time_steps)

# Reshape input features for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Define the LSTM model
def build_lstm():
    model = Sequential()
    model.add(LSTM(units=32, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    return model

model = build_lstm()

# Define the objective function for GWO
def objective_function(params):
    lr, decay = params
    model = build_lstm()
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, Y_train, epochs=5, batch_size=16, verbose=0)
    mse = history.history['loss'][-1]
    return mse

# Grey Wolf Optimizer - UPDATED to match SLSTM parameters
class GreyWolfOptimizer:
    def __init__(self, objective_function, lb, ub, dim, population_size, iterations):
        self.objective_function = objective_function
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.population_size = population_size
        self.iterations = iterations
        self.rng = np.random.default_rng(SEED)  # Add reproducibility

    def initialize_population(self):
        # Use reproducible random initialization
        return self.rng.uniform(low=self.lb, high=self.ub, size=(self.population_size, self.dim))

    def search(self):
        alpha_pos = np.zeros(self.dim)
        alpha_score = float("inf")
        beta_pos = np.zeros(self.dim)
        beta_score = float("inf")
        delta_pos = np.zeros(self.dim)
        delta_score = float("inf")
        positions = self.initialize_population()
        convergence_curve = np.zeros(self.iterations)

        for iteration in range(self.iterations):
            for i in range(self.population_size):
                # Update alpha, beta, delta
                score = self.objective_function(positions[i])
                if score < alpha_score:
                    delta_score, delta_pos = beta_score, beta_pos.copy()
                    beta_score, beta_pos = alpha_score, alpha_pos.copy()
                    alpha_score, alpha_pos = score, positions[i].copy()
                elif score < beta_score:
                    delta_score, delta_pos = beta_score, beta_pos.copy()
                    beta_score, beta_pos = score, positions[i].copy()
                elif score < delta_score:
                    delta_score, delta_pos = score, positions[i].copy()

                # Update positions
                a = 2 - (iteration * (2 / self.iterations))
                for j, leader in enumerate([alpha_pos, beta_pos, delta_pos]):
                    r1, r2 = self.rng.random(self.dim), self.rng.random(self.dim)  # Use reproducible RNG
                    A, C = 2 * a * r1 - a, 2 * r2
                    D = np.abs(C * leader - positions[i])
                    X = leader - A * D
                    if j == 0: X1 = X
                    elif j == 1: X2 = X
                    else: X3 = X
                positions[i] = (X1 + X2 + X3) / 3
                # Ensure positions stay within bounds
                positions[i] = np.clip(positions[i], self.lb, self.ub)
                
            convergence_curve[iteration] = alpha_score
        return alpha_pos, alpha_score, convergence_curve

# -----------------------------
# TIME COMPLEXITY ANALYSIS - ADDED SECTION
# -----------------------------
print("Starting time complexity analysis...")

# Start timing for training phase
training_start_time = time.time()

# GWO search - UPDATED to match SLSTM parameters (5 wolves, 5 iterations)
lb = [0.0001, 0.0001]
ub = [0.1, 0.9]
dim = 2
population_size = 5  # Changed from 10 to 5
iterations = 5       # Changed from 10 to 5
gwo = GreyWolfOptimizer(objective_function, lb, ub, dim, population_size, iterations)
alpha_pos, alpha_score, convergence_curve = gwo.search()

print(f"Best parameters found: LR={alpha_pos[0]:.4f}, Decay={alpha_pos[1]:.4f}")
print(f"Best score: {alpha_score:.4f}")

# Create results directory
results_dir = "BTC-USD_results"
os.makedirs(results_dir, exist_ok=True)

# -----------------------------
# CORRECTED PBIAS calculation function (same as SLSTM)
# -----------------------------
def calculate_pbias(actual, predicted):
    """Calculate Percent Bias (PBIAS) with proper error handling"""
    # Ensure arrays are flattened and have same shape
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()
    
    if len(actual) != len(predicted):
        raise ValueError(f"Arrays must have same length: actual={len(actual)}, predicted={len(predicted)}")
    
    # Check for zero sum in actual values
    sum_actual = np.sum(actual)
    if abs(sum_actual) < 1e-10:  # Avoid division by zero
        return float('inf') if np.sum(actual - predicted) > 0 else float('-inf')
    
    # Calculate PBIAS using standard formula: [Σ(actual - predicted) / Σ(actual)] * 100
    pbias = (np.sum(actual - predicted) / sum_actual) * 100
    return pbias

# -----------------------------
# IMPROVED MAPE calculation function (same as SLSTM)
# -----------------------------
def calculate_mape(actual, predicted):
    """Calculate Mean Absolute Percentage Error with robust error handling"""
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()
    
    # Avoid division by zero and handle very small values
    mask = np.abs(actual) > 1e-10  # Filter out near-zero values
    if np.sum(mask) == 0:
        return float('inf')  # All values are near zero
    
    actual_filtered = actual[mask]
    predicted_filtered = predicted[mask]
    
    # Calculate MAPE
    mape = np.mean(np.abs((actual_filtered - predicted_filtered) / actual_filtered)) * 100
    return mape

# -----------------------------
# COMPREHENSIVE TRAINING WITH METRIC TRACKING (like SLSTM)
# -----------------------------

# Initialize metric tracking - ADD PBIAS (same as SLSTM)
train_mse_history, train_rmse_history, train_mape_history, train_pbias_history = [], [], [], []
val_mse_history, val_rmse_history, val_mape_history, val_pbias_history = [], [], [], []

# Compile model with optimal parameters
model.compile(optimizer='adam', loss='mean_squared_error')

print("Starting comprehensive training with metric tracking...")
for epoch in range(50):
    # Train for one epoch
    history = model.fit(X_train, Y_train, epochs=1, batch_size=16, verbose=0, 
                       validation_data=(X_test, Y_test))
    
    # Get predictions for both train and test sets
    train_pred = model.predict(X_train, verbose=0)
    test_pred = model.predict(X_test, verbose=0)
    
    # Rescale predictions
    train_pred_rescaled = scaler.inverse_transform(train_pred)
    test_pred_rescaled = scaler.inverse_transform(test_pred)
    Y_train_rescaled_current = scaler.inverse_transform(Y_train.reshape(-1, 1))
    Y_test_rescaled_current = scaler.inverse_transform(Y_test.reshape(-1, 1))
    
    # Calculate metrics for training set (using same functions as SLSTM)
    train_mse_history.append(mean_squared_error(Y_train_rescaled_current, train_pred_rescaled))
    train_rmse_history.append(math.sqrt(train_mse_history[-1]))
    train_mape_history.append(calculate_mape(Y_train_rescaled_current, train_pred_rescaled))
    train_pbias_history.append(calculate_pbias(Y_train_rescaled_current.flatten(), train_pred_rescaled.flatten()))
    
    # Calculate metrics for validation set
    val_mse_history.append(mean_squared_error(Y_test_rescaled_current, test_pred_rescaled))
    val_rmse_history.append(math.sqrt(val_mse_history[-1]))
    val_mape_history.append(calculate_mape(Y_test_rescaled_current, test_pred_rescaled))
    val_pbias_history.append(calculate_pbias(Y_test_rescaled_current.flatten(), test_pred_rescaled.flatten()))
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train MSE: {train_mse_history[-1]:.4f}, Val MSE: {val_mse_history[-1]:.4f}")

print("Training completed!")

# End timing for training phase
training_end_time = time.time()
training_time = training_end_time - training_start_time

# Start timing for testing phase
testing_start_time = time.time()

# Final predictions for other plots
train_predictions = model.predict(X_train, verbose=0)
test_predictions = model.predict(X_test, verbose=0)

# Rescale for final metrics
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)
Y_train_rescaled = scaler.inverse_transform(Y_train.reshape(-1, 1))
Y_test_rescaled = scaler.inverse_transform(Y_test.reshape(-1, 1))

# Calculate final metrics (using different variable names)
final_train_rmse = math.sqrt(mean_squared_error(Y_train_rescaled, train_predictions))
final_test_rmse = math.sqrt(mean_squared_error(Y_test_rescaled, test_predictions))
final_train_mse = mean_squared_error(Y_train_rescaled, train_predictions)
final_test_mse = mean_squared_error(Y_test_rescaled, test_predictions)
final_train_mape = calculate_mape(Y_train_rescaled, train_predictions)
final_test_mape = calculate_mape(Y_test_rescaled, test_predictions)
final_train_pbias = calculate_pbias(Y_train_rescaled.flatten(), train_predictions.flatten())
final_test_pbias = calculate_pbias(Y_test_rescaled.flatten(), test_predictions.flatten())

# End timing for testing phase
testing_end_time = time.time()
testing_time = testing_end_time - testing_start_time

# -----------------------------
# TIME COMPLEXITY ANALYSIS RESULTS - ADDED
# -----------------------------
print("\n" + "="*60)
print("COMPUTATION TIME COMPLEXITY ANALYSIS")
print("="*60)
print(f"Training Phase Time: {training_time:.4f} seconds")
print(f"Testing Phase Time: {testing_time:.4f} seconds")
print(f"Total Execution Time: {training_time + testing_time:.4f} seconds")
print(f"Training/Testing Ratio: {training_time/testing_time:.4f}")

# Calculate time complexity metrics
n_train_samples = X_train.shape[0]
n_test_samples = X_test.shape[0]
n_features = X_train.shape[1]
n_epochs = 50

print(f"\nDataset Characteristics:")
print(f"Training samples: {n_train_samples}")
print(f"Testing samples: {n_test_samples}")
print(f"Features per sample: {n_features}")
print(f"Training epochs: {n_epochs}")

# Time per sample metrics
train_time_per_sample = training_time / (n_train_samples * n_epochs)
test_time_per_sample = testing_time / n_test_samples

print(f"\nTime Complexity Metrics:")
print(f"Training time per sample per epoch: {train_time_per_sample:.6f} seconds")
print(f"Testing time per sample: {test_time_per_sample:.6f} seconds")

# Save time complexity results
time_complexity_data = {
    "training_time_seconds": training_time,
    "testing_time_seconds": testing_time,
    "total_time_seconds": training_time + testing_time,
    "training_testing_ratio": training_time / testing_time,
    "n_train_samples": n_train_samples,
    "n_test_samples": n_test_samples,
    "n_features": n_features,
    "n_epochs": n_epochs,
    "train_time_per_sample_per_epoch": train_time_per_sample,
    "test_time_per_sample": test_time_per_sample
}

# Enhanced plotting with PBIAS - COMPREHENSIVE like SLSTM
plt.figure(figsize=(15, 12))

plt.subplot(4, 1, 1)
plt.plot(train_mse_history, label='Train MSE')
plt.plot(val_mse_history, label='Validation MSE')
plt.title('MSE Evolution')
plt.ylabel('MSE')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(train_rmse_history, label='Train RMSE')
plt.plot(val_rmse_history, label='Validation RMSE')
plt.title('RMSE Evolution')
plt.ylabel('RMSE')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(train_mape_history, label='Train MAPE')
plt.plot(val_mape_history, label='Validation MAPE')
plt.title('MAPE Evolution')
plt.ylabel('MAPE (%)')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(train_pbias_history, label='Train PBIAS')
plt.plot(val_pbias_history, label='Validation PBIAS')
plt.title('PBIAS Evolution')
plt.ylabel('PBIAS (%)')
plt.xlabel('Epochs')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "training_history_comprehensive.png"), dpi=600)
plt.show()

# Print metrics (same format as SLSTM)
print("\nTraining MSE:", [round(x, 6) for x in train_mse_history])
print("Validation MSE:", [round(x, 6) for x in val_mse_history])
print("\nTraining RMSE:", [round(x, 6) for x in train_rmse_history])
print("Validation RMSE:", [round(x, 6) for x in val_rmse_history])
print("\nTraining MAPE (%):", [round(x, 4) for x in train_mape_history])
print("Validation MAPE (%):", [round(x, 4) for x in val_mape_history])
print("\nTraining PBIAS:", [round(x, 4) for x in train_pbias_history])
print("Validation PBIAS:", [round(x, 4) for x in val_pbias_history])

# -----------------------------
# Additional metrics and functions
# -----------------------------
def rolling_metrics_series(y_true, y_pred, window=7):
    """
    Compute rolling RMSE and rolling MAPE aligned with y_true / y_pred arrays.
    Returns pandas Series indexed same as input (NaN for first window-1 entries).
    """
    n = len(y_true)
    rmse_roll = [np.nan]*n
    mape_roll = [np.nan]*n
    for i in range(window-1, n):
        y_slice = y_true[i-window+1:i+1].flatten()
        p_slice = y_pred[i-window+1:i+1].flatten()
        rmse_roll[i] = math.sqrt(mean_squared_error(y_slice, p_slice))
        # guard against zeros for MAPE
        with np.errstate(divide='ignore', invalid='ignore'):
            mape_roll[i] = calculate_mape(y_slice, p_slice)
    return pd.Series(rmse_roll, index=range(n)), pd.Series(mape_roll, index=range(n))

# -----------------------------
# Plots: Radar, Box, Taylor, Rolling (same as before)
# -----------------------------
def plot_radar(metrics_dict, save_path):
    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(metrics_dict.keys())
    values = np.array(list(metrics_dict.values()), dtype=float)

    # Normalize automatically (higher = better). Lower-is-better inverted.
    norm_values = []
    max_val = np.max(values) if np.max(values) > 0 else 1.0
    for key, val in metrics_dict.items():
        if key.upper() in ["RMSE", "MSE", "MAPE", "PBIAS", "MAE"]:
            norm_val = 1 - (val / (max_val + 1e-8))
        else:
            norm_val = val / (max_val + 1e-8)
        norm_values.append(float(norm_val))
    norm_values = np.array(norm_values)

    # Close the radar shape
    norm_values = np.concatenate((norm_values, [norm_values[0]]))
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, norm_values, 'o-', linewidth=2, label="Model Performance")
    ax.fill(angles, norm_values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    
    # Increase the radial limit to create more space for labels
    ax.set_rorigin(-0.1)  # Move origin inward
    ax.set_ylim(0, 1.2)   # Extend radial limit

    # Annotate each metric with raw value - adjusted positions
    for i, (angle, value, label, raw) in enumerate(zip(angles[:-1], norm_values[:-1], labels, values)):
        # Position text further out to avoid overlap with outer circle
        text_radius = 1.15
        
        # Adjust horizontal alignment based on angle for better positioning
        if angle < np.pi/2 or angle > 3*np.pi/2:
            ha = 'left'
        else:
            ha = 'right'
            
        # Adjust vertical alignment
        if angle < np.pi:
            va = 'bottom'
        else:
            va = 'top'
            
        ax.text(angle, text_radius, f"{raw:.3f}", 
                ha=ha, va=va, fontsize=10, color='black', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.title("Radar Chart of Normalized Metrics", size=12, pad=20)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

def plot_box(y_true, y_pred, save_path):
    ape = np.abs((y_true - y_pred)/y_true)*100
    # remove inf/NaN points that might occur if y_true==0
    ape = ape[~np.isnan(ape) & ~np.isinf(ape)]
    fig, ax = plt.subplots(figsize=(6,6))
    ax.boxplot(ape)
    ax.set_ylabel("APE (%)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()

def plot_taylor(y_true, y_pred, save_path, metrics_txt):
    class TaylorDiagram:
        def __init__(self, ref_std, fig=None, rect=111, label="Reference"):
            self.ref_std = ref_std
            self.fig = fig or plt.figure(figsize=(6,6))
            self.ax = self.fig.add_subplot(rect, polar=True)
            self.sample_points = []
            self.ax.set_theta_zero_location('N')
            self.ax.set_theta_direction(-1)
            self.ax.set_rmax(ref_std*1.5 if ref_std>0 else 1.0)
            self.ax.plot(0, ref_std, 'bo', label=label)
        def add_sample(self, stddev, corrcoef, label, **kwargs):
            theta = np.arccos(np.clip(corrcoef, -1, 1))
            r = stddev
            point, = self.ax.plot(theta, r, 'o', label=label, **kwargs)
            self.sample_points.append(point)
            return point
        def add_contours(self, levels=5, **kwargs):
            rs, thetas = np.meshgrid(np.linspace(0, self.ax.get_rmax(), 200),
                                     np.linspace(0, np.pi/2, 200))
            rms = np.sqrt(self.ref_std**2 + rs**2 - 2*self.ref_std*rs*np.cos(thetas))
            cs = self.ax.contour(thetas, rs, rms, levels=levels, **kwargs)
            return cs

    std_ref = float(np.std(y_true)) if np.std(y_true) > 0 else 1.0
    std_pred = float(np.std(y_pred))
    corr = np.corrcoef(y_true.flatten(), y_pred.flatten())[0,1]
    fig = plt.figure(figsize=(6,6))
    dia = TaylorDiagram(std_ref, fig=fig)
    dia.add_sample(std_pred, corr, label="LSTM Prediction", color='r')
    dia.add_contours(levels=5, colors='0.5', linestyles='--')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()
    # Save metrics
    with open(metrics_txt, "a") as f:
        f.write("\nFig. 5: Taylor diagram showing std & correlation comparison.\n")
        f.write(f"Std prediction: {std_pred:.2f}, Std reference: {std_ref:.2f}, Correlation: {corr:.2f}\n")

def plot_rolling_metrics(dates, rmse_series, mape_series, save_path):
    plt.figure(figsize=(12, 5))
    plt.plot(dates, rmse_series, label='Rolling RMSE (7-day)')
    plt.plot(dates, mape_series, label='Rolling MAPE (7-day)')
    plt.title('Rolling 7-day Metrics')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()

def plot_error_vs_volatility(dates, ape_series, vol_series, save_path):
    plt.figure(figsize=(8,6))
    plt.scatter(vol_series, ape_series, s=8, alpha=0.6)
    plt.xlabel("Realized Volatility (7-day rolling std of returns)")
    plt.ylabel("APE (%)")
    plt.title("Supplemental: Error (APE) vs Realized Volatility")
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()

# -----------------------------
# Calculate additional metrics and create plots
# -----------------------------

# Save metrics (including time complexity)
metrics_file = os.path.join(results_dir, "LSTMBTC-USD_results.txt")
with open(metrics_file, "w") as f:
    f.write(f"GWO Parameters: 5 wolves, 5 iterations\n")
    f.write(f"Best parameters: LR={alpha_pos[0]:.4f}, Decay={alpha_pos[1]:.4f}\n")
    f.write(f"Best score: {alpha_score:.4f}\n")
    f.write(f"Train RMSE: {final_train_rmse}\n")
    f.write(f"Test RMSE: {final_test_rmse}\n")
    f.write(f"Train MSE: {final_train_mse}\n")
    f.write(f"Test MSE: {final_test_mse}\n")
    f.write(f"Train MAPE: {final_train_mape}\n")
    f.write(f"Test MAPE: {final_test_mape}\n")
    f.write(f"Train PBIAS: {final_train_pbias}\n")
    f.write(f"Test PBIAS: {final_test_pbias}\n")
    f.write("\n" + "="*50 + "\n")
    f.write("COMPUTATION TIME COMPLEXITY ANALYSIS\n")
    f.write("="*50 + "\n")
    f.write(f"Training Phase Time: {training_time:.4f} seconds\n")
    f.write(f"Testing Phase Time: {testing_time:.4f} seconds\n")
    f.write(f"Total Execution Time: {training_time + testing_time:.4f} seconds\n")
    f.write(f"Training/Testing Ratio: {training_time/testing_time:.4f}\n")
    f.write(f"Training samples: {n_train_samples}\n")
    f.write(f"Testing samples: {n_test_samples}\n")
    f.write(f"Features per sample: {n_features}\n")
    f.write(f"Training epochs: {n_epochs}\n")
    f.write(f"Training time per sample per epoch: {train_time_per_sample:.6f} seconds\n")
    f.write(f"Testing time per sample: {test_time_per_sample:.6f} seconds\n")

# Create radar chart
metrics_dict = {
    "RMSE": final_test_rmse,
    "MSE": final_test_mse,
    "MAPE": final_test_mape,
    "PBIAS": abs(final_test_pbias)
}
plot_radar(metrics_dict, os.path.join(results_dir, "LSTMBTC-USD_radar.png"))

# Create box plot
plot_box(Y_test_rescaled.flatten(), test_predictions.flatten(), 
         os.path.join(results_dir, "LSTMBTC-USD_boxplot.png"))

# Create Taylor diagram
plot_taylor(Y_test_rescaled, test_predictions, 
            os.path.join(results_dir, "LSTMBTC-USD_taylor.png"), metrics_file)

# Create rolling metrics (using test set)
rmse_roll, mape_roll = rolling_metrics_series(Y_test_rescaled, test_predictions, window=7)
# For dates, we'll use indices since we don't have actual dates in the test set
dates_test = range(len(Y_test_rescaled))
plot_rolling_metrics(dates_test, rmse_roll, mape_roll, 
                    os.path.join(results_dir, "LSTMBTC-USD_rolling.png"))

# Create error vs volatility plot
# Calculate returns and volatility
returns = np.diff(Y_test_rescaled.flatten()) / Y_test_rescaled.flatten()[:-1] * 100
# Add NaN at beginning to align with Y_test_rescaled
returns = np.insert(returns, 0, np.nan)
# Calculate rolling volatility (skip first value which is NaN)
volatility = pd.Series(returns).rolling(window=7).std().values
# Calculate APE
ape = np.abs((Y_test_rescaled.flatten() - test_predictions.flatten()) / Y_test_rescaled.flatten()) * 100
# Remove inf/NaN
valid_idx = ~(np.isnan(ape) | np.isinf(ape) | np.isnan(volatility))
ape_valid = ape[valid_idx]
vol_valid = volatility[valid_idx]
dates_valid = np.array(dates_test)[valid_idx]

plot_error_vs_volatility(dates_valid, ape_valid, vol_valid, 
                        os.path.join(results_dir, "LSTMBTC-USD_error_volatility.png"))

# Original prediction plot
plt.figure(figsize=(12, 6))
plt.plot(Y_test_rescaled, label='Actual Bitcoin Price')
plt.plot(test_predictions, label='Predicted Bitcoin Price (LSTM)')
plt.xlabel('Time Steps')
plt.ylabel('Bitcoin Price (USD)')
plt.title('Bitcoin Price Prediction Using LSTM with GWO Optimization')
plt.legend()
plt.grid(True)

# Add descriptive caption for paper
caption = (
    "Figure X: Predicted vs actual Bitcoin prices using the LSTM model. "
    "The red dashed line represents the model's predictions, while the blue line "
    "shows the actual observed prices. The plot demonstrates that the LSTM model "
    "captures the overall trends and fluctuations in the Bitcoin market, though "
    "some deviations occur during periods of high volatility."
)

# Save the figure
plot_file = os.path.join(results_dir, "LSTMBTC-USD_predictions.png")
plt.savefig(plot_file, dpi=600, bbox_inches='tight')
plt.show()

# -----------------------------
# Save data for each figure with descriptions
# -----------------------------
print("Saving figure data and descriptions...")

# Radar chart data
radar_data = {
    "RMSE": final_test_rmse,
    "MSE": final_test_mse,
    "MAPE": final_test_mape,
    "PBIAS": abs(final_test_pbias),
    "Normalized_RMSE": 1 - (final_test_rmse / max(final_test_rmse, final_test_mse, final_test_mape, abs(final_test_pbias))),
    "Normalized_MSE": 1 - (final_test_mse / max(final_test_rmse, final_test_mse, final_test_mape, abs(final_test_pbias))),
    "Normalized_MAPE": 1 - (final_test_mape / max(final_test_rmse, final_test_mse, final_test_mape, abs(final_test_pbias))),
    "Normalized_PBIAS": 1 - (abs(final_test_pbias) / max(final_test_rmse, final_test_mse, final_test_mape, abs(final_test_pbias)))
}
save_figure_data("LSTMBTC-USD_radar", 
                "Radar chart showing normalized performance metrics (RMSE, MSE, MAPE, PBIAS) for LSTM model",
                radar_data, results_dir)

# Box plot data
ape_for_box = np.abs((Y_test_rescaled.flatten() - test_predictions.flatten()) / Y_test_rescaled.flatten()) * 100
ape_for_box = ape_for_box[~np.isnan(ape_for_box) & ~np.isinf(ape_for_box)]
box_data = {
    "APE_values_sample": ape_for_box[:20].tolist(),
    "Min_APE": np.min(ape_for_box),
    "Max_APE": np.max(ape_for_box),
    "Median_APE": np.median(ape_for_box),
    "Mean_APE": np.mean(ape_for_box),
    "Q1_APE": np.percentile(ape_for_box, 25),
    "Q3_APE": np.percentile(ape_for_box, 75),
    "Number_of_points": len(ape_for_box)
}
save_figure_data("LSTMBTC-USD_boxplot",
                "Box plot of Absolute Percentage Errors (APE) distribution for LSTM predictions",
                box_data, results_dir)

# Taylor diagram data
std_ref_taylor = float(np.std(Y_test_rescaled)) if np.std(Y_test_rescaled) > 0 else 1.0
std_pred_taylor = float(np.std(test_predictions))
corr_taylor = np.corrcoef(Y_test_rescaled.flatten(), test_predictions.flatten())[0,1]
taylor_data = {
    "Reference_std": std_ref_taylor,
    "Prediction_std": std_pred_taylor,
    "Correlation": corr_taylor,
    "RMS_difference": math.sqrt(std_ref_taylor**2 + std_pred_taylor**2 - 2*std_ref_taylor*std_pred_taylor*corr_taylor)
}
save_figure_data("LSTMBTC-USD_taylor",
                "Taylor diagram comparing standard deviation and correlation between actual and predicted prices",
                taylor_data, results_dir)

# Rolling metrics data
rolling_data = {
    "Rolling_RMSE_series_sample": rmse_roll.dropna().head(20).tolist(),
    "Rolling_MAPE_series_sample": mape_roll.dropna().head(20).tolist(),
    "Average_Rolling_RMSE": rmse_roll.mean(),
    "Average_Rolling_MAPE": mape_roll.mean(),
    "Max_Rolling_RMSE": rmse_roll.max(),
    "Max_Rolling_MAPE": mape_roll.max(),
    "Window_size": 7
}
save_figure_data("LSTMBTC-USD_rolling",
                "7-day rolling RMSE and MAPE metrics showing temporal performance consistency",
                rolling_data, results_dir)

# Error vs volatility data
error_vol_data = {
    "APE_values_sample": ape_valid[:20].tolist(),
    "Volatility_values_sample": vol_valid[:20].tolist(),
    "Correlation_APE_Volatility": np.corrcoef(ape_valid, vol_valid)[0,1],
    "Number_of_points": len(ape_valid),
    "Min_volatility": np.min(vol_valid),
    "Max_volatility": np.max(vol_valid),
    "Mean_volatility": np.mean(vol_valid)
}
save_figure_data("LSTMBTC-USD_error_volatility",
                "Scatter plot showing relationship between prediction errors (APE) and market volatility",
                error_vol_data, results_dir)

# Prediction plot data
prediction_data = {
    "Actual_prices_sample": Y_test_rescaled.flatten()[:20].tolist(),
    "Predicted_prices_sample": test_predictions.flatten()[:20].tolist(),
    "Time_steps_sample": list(range(20)),
    "Correlation_actual_predicted": np.corrcoef(Y_test_rescaled.flatten(), test_predictions.flatten())[0,1],
    "Mean_actual_price": np.mean(Y_test_rescaled),
    "Mean_predicted_price": np.mean(test_predictions),
    "Total_points": len(Y_test_rescaled)
}
save_figure_data("LSTMBTC-USD_predictions",
                "Time series comparison of actual vs predicted Bitcoin prices using LSTM model",
                prediction_data, results_dir)

# Training history data
training_history_data = {
    "Training_MSE_values": train_mse_history,
    "Validation_MSE_values": val_mse_history,
    "Training_RMSE_values": train_rmse_history,
    "Validation_RMSE_values": val_rmse_history,
    "Training_MAPE_values": train_mape_history,
    "Validation_MAPE_values": val_mape_history,
    "Training_PBIAS_values": train_pbias_history,
    "Validation_PBIAS_values": val_pbias_history,
    "Final_training_MSE": train_mse_history[-1] if train_mse_history else 0,
    "Final_validation_MSE": val_mse_history[-1] if val_mse_history else 0,
    "Number_of_epochs": len(train_mse_history)
}
save_figure_data("training_history_comprehensive",
                "Comprehensive training history showing evolution of MSE, RMSE, MAPE, and PBIAS across epochs",
                training_history_data, results_dir)

# Save time complexity data separately
save_figure_data("time_complexity_analysis",
                "Computation time complexity analysis for training and testing phases",
                time_complexity_data, results_dir)

print("All figure data saved successfully!")

print("Metrics and all plots saved successfully.")
print("\nSuggested figure caption for paper:\n")
print(caption)