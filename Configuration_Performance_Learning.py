import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

# Configuration
systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
datasets_folder = 'C:\\Users\\chiam\\OneDrive\\Desktop\\ISE\\lab2\\datasets'  
num_repeats = 5  
train_frac = 0.7  
random_seed = 42  

# Define models
baseline_model = LinearRegression()
chosen_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=random_seed)

# Function to preprocess dataset
def preprocess_data(data):
    """Encodes categorical variables & normalizes numerical features."""
    data = data.dropna()  # Remove missing values

    # Encode categorical variables
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = LabelEncoder().fit_transform(data[col])

    # Normalize numerical columns
    scaler = StandardScaler()
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    return data

# Function to evaluate model performance
def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Trains the model, makes predictions, and calculates performance metrics."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Compute error metrics
    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test + 1e-10, predictions)  # Avoid division by zero
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return mae, mape, rmse

# Store final results
all_results = []

# Run experiments on each system
for system in systems:
    system_path = os.path.join(datasets_folder, system)

    if not os.path.exists(system_path):
        print(f"Warning: Folder {system_path} does not exist. Skipping...")
        continue

    csv_files = [f for f in os.listdir(system_path) if f.endswith('.csv')]

    for csv_file in csv_files:
        print(f"\n Analyzing {system}/{csv_file}")

        try:
            data = pd.read_csv(os.path.join(system_path, csv_file))
            data = preprocess_data(data)  # Apply preprocessing
        except Exception as e:
            print(f" Error reading {csv_file}: {e}")
            continue

        # Split features (X) and target (y)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Repeat experiment multiple times
        for i in range(num_repeats):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_frac, random_state=i)

            # Evaluate baseline (Linear Regression)
            mae_baseline, mape_baseline, rmse_baseline = evaluate_model(baseline_model, X_train, X_test, y_train, y_test)
            all_results.append([system, csv_file, "Linear Regression", mae_baseline, mape_baseline, rmse_baseline])

            # Evaluate improved model (XGBoost)
            mae_xgb, mape_xgb, rmse_xgb = evaluate_model(chosen_model, X_train, X_test, y_train, y_test)
            all_results.append([system, csv_file, "XGBoost", mae_xgb, mape_xgb, rmse_xgb])

# Save results to CSV
results_df = pd.DataFrame(all_results, columns=["System", "Dataset", "Model", "MAE", "MAPE", "RMSE"])
results_df.to_csv("model_performance_results1.csv", index=False)

print("\n Experiment completed! Results saved to 'model_performance_results1.csv'.")
