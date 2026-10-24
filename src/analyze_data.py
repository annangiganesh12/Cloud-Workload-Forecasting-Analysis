import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_anomalies(cpu_path, meta_path):
    print("Loading data for deep-dive analysis...")
    df_cpu = pd.read_csv(cpu_path)
    df_meta = pd.read_csv(meta_path)

    # 1. Timestamp Cadence & Missingness
    print("\n--- Checking Timestamp Integrity ---")
    # Get unique timestamps and sort them
    timestamps = sorted(df_cpu['timestamp'].unique())
    ts_series = pd.Series(timestamps)
    diffs = ts_series.diff().dropna()

    expected_diff = 300 # 5 minutes
    irregularities = diffs[diffs != expected_diff]

    print(f"Total unique timestamps: {len(timestamps)}")
    print(f"Expected cadence: {expected_diff}s")
    if len(irregularities) == 0:
        print("✅ Timestamp cadence is perfectly consistent.")
    else:
        print(f"❌ Found {len(irregularities)} irregularities in cadence.")
        print(irregularities)

    # 2. Missingness per VM
    print("\n--- Checking Data Completeness ---")
    expected_count = len(timestamps)
    actual_counts = df_cpu.groupby('vm_id').size()
    missing_counts = expected_count - actual_counts

    print(f"Expected readings per VM: {expected_count}")
    print(f"Average readings per VM: {actual_counts.mean():.2f}")
    print(f"VMs with missing data: {(missing_counts > 0).sum()} / {len(df_meta)}")
    print(f"Max missing readings in a single VM: {missing_counts.max()}")

    # 3. CPU Utilization Distribution & Outliers
    print("\n--- Analyzing CPU Utilization ---")
    cpu_stats = df_cpu['avg_cpu'].describe()
    print(cpu_stats)

    # Define outliers as values beyond 3 sigma or logical bounds [0, 100]
    out_of_bounds = df_cpu[(df_cpu['avg_cpu'] < 0) | (df_cpu['avg_cpu'] > 100)]
    print(f"Readings outside [0, 100]: {len(out_of_bounds)}")

    # 4. Workload Segmentation Analysis
    print("\n--- Segment-level Analysis ---")
    # Merge metadata to analyze by VM attributes
    df_merged = df_cpu.merge(df_meta, on='vm_id')

    # Let's look at 'virtual_core_bucket' or similar if it exists
    # Checking available meta columns
    meta_cols = df_meta.columns.tolist()
    print(f"Metadata columns available: {meta_cols}")

    # If core bucket exists, analyze CPU mean per bucket
    core_col = next((c for c in meta_cols if 'core' in c.lower()), None)
    if core_col:
        print(f"Analyzing by {core_col}...")
        # The error was due to duplicate column names after merge (avg_cpu in both)
        # We use the one from df_cpu (the time-series reading) for the aggregation
        # pandas merge creates avg_cpu_x and avg_cpu_y if they overlap
        target_col = 'avg_cpu_x' if 'avg_cpu_x' in df_merged.columns else 'avg_cpu'
        segment_stats = df_merged.groupby(core_col)[target_col].agg(['mean', 'std', 'max', 'count'])
        print(segment_stats)
    else:
        print("No core/memory bucket column found for segmentation.")

    # 5. Temporal Patterns (Peak/Low)
    print("\n--- Temporal Demand Patterns ---")
    fleet_cpu = df_cpu.groupby('timestamp')['avg_cpu'].mean()
    print(f"Peak aggregate CPU: {fleet_cpu.max():.4f}% at timestamp {fleet_cpu.idxmax()}")
    print(f"Low aggregate CPU: {fleet_cpu.min():.4f}% at timestamp {fleet_cpu.idxmin()}")

if __name__ == "__main__":
    cpu_path = 'D:/INTERN_CHALLENGE/data-v1/vm_cpu_readings.csv/vm_cpu_readings.csv'
    meta_path = 'D:/INTERN_CHALLENGE/data-v1/vm_metadata.csv/vm_metadata.csv'
    analyze_anomalies(cpu_path, meta_path)
