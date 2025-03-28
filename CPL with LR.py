import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# List of software systems to evaluate
systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
datasets_folder = f'C:\\Users\\chiam\\OneDrive\\Desktop\\ISE\\lab2\\datasets'
num_repeats = 5  # Repeat experiments for better accuracy
train_frac = 0.7  # 70% training, 30% testing
random_seed = 42  # Ensures reproducibility

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# Function to preprocess the dataset (handles categorical variables, normalization)
def preprocess_data(data):
    """Encodes categorical variables & normalizes numerical features."""
    data = data.dropna()  # Remove missing values

    # Encode categorical columns
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = LabelEncoder().fit_transform(data[col])

    # Normalize numeric columns
    scaler = StandardScaler()
    data[data.columns] = scaler.fit_transform(data)

    return data

# Function to evaluate a model and return performance metrics
def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Trains model, makes predictions, and calculates errors."""
    model.fit(X_train, y_train)  # Train model
    predictions = model.predict(X_test)  # Predict

    # Compute error metrics
    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test + 1e-10, predictions)  # Avoid div by zero
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return mae, mape, rmse

# Prepare a list to store all the results
all_results = []

# Run the experiment on multiple datasets
for system in systems:
    system_path = os.path.join(datasets_folder, system)

    if not os.path.exists(system_path):
        print(f"Warning: Folder {system_path} does not exist. Skipping.")
        continue

    csv_files = [f for f in os.listdir(system_path) if f.endswith('.csv')]

    for csv_file in csv_files:
        print(f"\n Analyzing {system}/{csv_file}")

        data = pd.read_csv(os.path.join(system_path, csv_file))
        data = preprocess_data(data)  # Apply preprocessing

        # Split features (X) and target (y)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Store results
        metrics_results = {name: {"MAPE": [], "MAE": [], "RMSE": []} for name in models}

        # Repeat multiple times for better accuracy
        for i in range(num_repeats):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_frac, random_state=i)

            for name, model in models.items():
                mae, mape, rmse = evaluate_model(model, X_train, X_test, y_train, y_test)

                # Store results
                metrics_results[name]["MAPE"].append(mape)
                metrics_results[name]["MAE"].append(mae)
                metrics_results[name]["RMSE"].append(rmse)

        # Print final results and save to list
        print(f"{system}/{csv_file} - Average Model Performance:")
        for name, metrics in metrics_results.items():
            avg_mape = np.mean(metrics['MAPE'])
            avg_mae = np.mean(metrics['MAE'])
            avg_rmse = np.mean(metrics['RMSE'])
            print(f"   {name}: MAPE={avg_mape:.4%}, MAE={avg_mae:.4f}, RMSE={avg_rmse:.4f}")

            # Append results to the list
            all_results.append({
                'System': system,
                'CSV File': csv_file,
                'Model': name,
                'Avg MAPE': avg_mape,
                'Avg MAE': avg_mae,
                'Avg RMSE': avg_rmse
            })

        
# Convert all results to a DataFrame
results_df = pd.DataFrame(all_results)

# Save results to a CSV file
results_df.to_csv('model_performance_results1.csv', index=False)

print("\nResults saved to 'model_performance_results1.csv'.")
