import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

def generate_aggregate_targets(cpu_path):
    """
    Converts VM-level readings into fleet-level aggregate targets.
    """
    print("Generating aggregate targets...")
    df = pd.read_csv(cpu_path, usecols=['timestamp', 'vm_id', 'avg_cpu'])

    targets = df.groupby('timestamp').agg(
        active_vm_count=('vm_id', 'nunique'),
        aggregate_cpu_utilization=('avg_cpu', 'mean')
    ).reset_index()

    return targets.sort_values('timestamp')

def create_training_set(targets, window_size=3):
    """
    Transforms the target series into a feature matrix using lagged values.
    """
    print("Creating feature matrix with lags...")
    df = targets.copy()

    for col in ['active_vm_count', 'aggregate_cpu_utilization']:
        for lag in range(1, window_size + 1):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    df['cpu_rolling_mean'] = df['aggregate_cpu_utilization'].shift(1).rolling(window=window_size).mean()

    return df.dropna().reset_index(drop=True)

def evaluate_persistence_baseline(df):
    """
    Implements the Persistence Baseline: Predict t+1 = t.
    """
    print("\nEvaluating Persistence Baseline...")
    test = df.iloc[-24:]

    results = {}
    for col in ['active_vm_count', 'aggregate_cpu_utilization']:
        y_true = test[col]
        y_pred = test[f'{col}_lag_1']

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        results[col] = {'MAE': mae, 'RMSE': rmse}
        print(f"Baseline for {col}: MAE={mae:.4f}, RMSE={rmse:.4f}")

    return results

def train_and_evaluate_ml(df):
    """
    Trains ML model on the full training window (first 110) and tests on the last 24.
    """
    print("\nTraining ML Model (RandomForest)...")

    train = df.iloc[:-24]
    test = df.iloc[-24:]

    features = [c for c in df.columns if 'lag' in c or 'rolling' in c]

    results = {}
    models = {}
    for target in ['active_vm_count', 'aggregate_cpu_utilization']:
        X_train = train[features]
        y_train = train[target]
        X_test = test[features]
        y_test = test[target]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results[target] = {'MAE': mae, 'RMSE': rmse}
        models[target] = model
        print(f"ML Model for {target}: MAE={mae:.4f}, RMSE={rmse:.4f}")

    return results, models, test, features

def generate_predictions(models, test_df, features):
    """
    Generates the final predictions.csv for submission.
    """
    print("\nGenerating predictions.csv...")
    preds = pd.DataFrame({'timestamp': test_df['timestamp']})

    for target, model in models.items():
        X_test = test_df[features]
        preds[target] = model.predict(X_test)

    preds.to_csv('predictions.csv', index=False)
    print("predictions.csv saved successfully.")

def main():
    cpu_readings_path = 'D:/INTERN_CHALLENGE/data-v1/vm_cpu_readings.csv/vm_cpu_readings.csv'

    targets_df = generate_aggregate_targets(cpu_readings_path)
    targets_df.to_csv('targets.csv', index=False)

    training_set = create_training_set(targets_df)
    training_set.to_csv('training_set.csv', index=False)

    baseline_metrics = evaluate_persistence_baseline(training_set)
    ml_metrics, models, test_df, features = train_and_evaluate_ml(training_set)

    generate_predictions(models, test_df, features)

    print("\n--- Final Comparison ---")
    for target in ['active_vm_count', 'aggregate_cpu_utilization']:
        b_mae = baseline_metrics[target]['MAE']
        m_mae = ml_metrics[target]['MAE']
        improvement = ((b_mae - m_mae) / b_mae * 100) if b_mae != 0 else 0
        print(f"{target}: Baseline MAE={b_mae:.4f} vs ML MAE={m_mae:.4f} ({improvement:.2f}% improvement)")

if __name__ == "__main__":
    main()
