import pytest
import pandas as pd
import numpy as np
from src.main import generate_aggregate_targets, create_training_set

def test_generate_aggregate_targets_correctness():
    """
    Verify that the aggregation logic correctly calculates
    the count and the mean of CPU utilization.
    """
    data = {
        'timestamp': [0, 0, 1],
        'vm_id': ['vm1', 'vm2', 'vm1'],
        'avg_cpu': [10.0, 20.0, 30.0]
    }
    df = pd.DataFrame(data)
    df.to_csv('test_dummy_cpu.csv', index=False)

    targets = generate_aggregate_targets('test_dummy_cpu.csv')

    t0 = targets[targets['timestamp'] == 0]
    assert t0['active_vm_count'].iloc[0] == 2
    assert t0['aggregate_cpu_utilization'].iloc[0] == 15.0

    t1 = targets[targets['timestamp'] == 1]
    assert t1['active_vm_count'].iloc[0] == 1
    assert t1['aggregate_cpu_utilization'].iloc[0] == 30.0

def test_create_training_set_lag_integrity():
    """
    Ensure that lag features are shifted correctly and no future data is leaked.
    """
    data = {
        'timestamp': [0, 1, 2, 3, 4],
        'active_vm_count': [100] * 5,
        'aggregate_cpu_utilization': [10.0, 20.0, 30.0, 40.0, 50.0]
    }
    df = pd.DataFrame(data)

    training_set = create_training_set(df, window_size=3)

    first_row = training_set.iloc[0]
    assert first_row['aggregate_cpu_utilization_lag_1'] == 30.0
    assert first_row['aggregate_cpu_utilization_lag_2'] == 20.0
    assert first_row['aggregate_cpu_utilization_lag_3'] == 10.0

def test_model_invariants():
    """
    Test the model's behavioral invariants: range, consistency, and basic trend logic.
    """
    # Setup a minimal model for testing
    data = {
        'timestamp': range(20),
        'active_vm_count': [100] * 20,
        'aggregate_cpu_utilization': np.linspace(10, 20, 20)
    }
    df = pd.DataFrame(data)
    training_set = create_training_set(df, window_size=3)

    features = [c for c in training_set.columns if 'lag' in c or 'rolling' in c]
    X = training_set[features]
    y = training_set['aggregate_cpu_utilization']

    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    # Range Test (CPU must be between 0 and 100)
    test_input = np.array([[100, 100, 100, 50.0, 50.0, 50.0, 50.0]])
    prediction = model.predict(test_input)[0]
    assert 0 <= prediction <= 100, f"Model predicted CPU {prediction}, which is out of bounds [0, 100]"

    # Consistency Test (Determinism)
    pred1 = model.predict(test_input)[0]
    pred2 = model.predict(test_input)[0]
    assert pred1 == pred2, "Model is non-deterministic"

    # Trend Logic (Climbing vs Falling)
    climb_input = np.array([[100, 100, 100, 20.0, 15.0, 10.0, 15.0]])
    fall_input = np.array([[100, 100, 100, 10.0, 15.0, 20.0, 15.0]])

    pred_climb = model.predict(climb_input)[0]
    pred_fall = model.predict(fall_input)[0]

    assert pred_climb >= pred_fall, "Model failed basic trend logic: climbing trend predicted lower than falling trend"
