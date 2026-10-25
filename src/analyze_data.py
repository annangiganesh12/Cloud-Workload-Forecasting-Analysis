import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_anomalies(cpu_path, meta_path):
    print("Loading data for deep-dive analysis...")
    df_cpu = pd.read_csv(cpu_path)
    df_meta = pd.read_csv(meta_path)

    # Setup plotting
    fig = plt.figure(figsize=(15, 20))
    plt.subplots_adjust(hspace=0.4)

    # 1. Timestamp Cadence & Missingness
    print("\n--- Checking Timestamp Integrity ---")
    timestamps = sorted(df_cpu['timestamp'].unique())
    ts_series = pd.Series(timestamps)
    diffs = ts_series.diff().dropna()

    expected_diff = 300 # 5 minutes
    irregularities = diffs[diffs != expected_diff]

    print(f"Total unique timestamps: {len(timestamps)}")
    print(f"Expected cadence: {expected_diff}s")
    if len(irregularities) == 0:
        print("Timestamp cadence is perfectly consistent.")
    else:
        print(f"Found {len(irregularities)} irregularities in cadence.")
        print(irregularities)

    # 2. Missingness per VM
    print("\n--- Checking Data Completeness ---")
    expected_count = len(timestamps)
    actual_counts = df_cpu.groupby('vm_id').size()
    missing_counts = expected_count - actual_counts

    print(f"Expected readings per VM: {expected_count}")
    print(f"Average readings per VM: {actual_counts.mean():.2f}")
    print(f"VMs with missing data: {(missing_counts > 0).sum()} / {len(df_meta)}")

    # Plot 1: VM Reading Counts
    ax1 = plt.subplot(4, 1, 1)
    ax1.hist(actual_counts, bins=20, color='skyblue', edgecolor='black')
    ax1.set_title("Distribution of Readings per VM")
    ax1.set_xlabel("Number of Readings")
    ax1.set_ylabel("Number of VMs")

    # 3. CPU Utilization Distribution & Outliers
    print("\n--- Analyzing CPU Utilization ---")
    cpu_stats = df_cpu['avg_cpu'].describe()
    print(cpu_stats)

    out_of_bounds = df_cpu[(df_cpu['avg_cpu'] < 0) | (df_cpu['avg_cpu'] > 100)]
    print(f"Readings outside [0, 100]: {len(out_of_bounds)}")

    # Plot 2: CPU Distribution
    ax2 = plt.subplot(4, 1, 2)
    ax2.hist(df_cpu['avg_cpu'], bins=50, color='salmon', edgecolor='black')
    ax2.set_title("Overall CPU Utilization Distribution")
    ax2.set_xlabel("CPU %")
    ax2.set_ylabel("Frequency")

    # 4. Workload Segmentation Analysis
    print("\n--- Segment-level Analysis ---")
    df_merged = df_cpu.merge(df_meta, on='vm_id')

    meta_cols = df_meta.columns.tolist()
    core_col = next((c for c in meta_cols if 'core' in c.lower()), None)

    if core_col:
        print(f"Analyzing by {core_col}...")
        target_col = 'avg_cpu_x' if 'avg_cpu_x' in df_merged.columns else 'avg_cpu'
        segment_stats = df_merged.groupby(core_col)[target_col].agg(['mean', 'std', 'count'])
        print(segment_stats)

        # Plot 3: Mean CPU per Core Bucket
        ax3 = plt.subplot(4, 1, 3)
        segment_stats['mean'].plot(kind='bar', ax=ax3, color='lightgreen', edgecolor='black')
        ax3.set_title(f"Mean CPU Utilization by {core_col}")
        ax3.set_ylabel("Mean CPU %")
        ax3.set_xlabel("Core Bucket")
        plt.xticks(rotation=45)
    else:
        print("No core/memory bucket column found for segmentation.")

    # 5. Temporal Patterns (Peak/Low)
    print("\n--- Temporal Demand Patterns ---")
    fleet_cpu = df_cpu.groupby('timestamp')['avg_cpu'].mean()
    print(f"Peak aggregate CPU: {fleet_cpu.max():.4f}% at timestamp {fleet_cpu.idxmax()}")
    print(f"Low aggregate CPU: {fleet_cpu.min():.4f}% at timestamp {fleet_cpu.idxmin()}")

    # Plot 4: Aggregate CPU over Time
    ax4 = plt.subplot(4, 1, 4)
    ax4.plot(fleet_cpu.index, fleet_cpu.values, color='blue', linewidth=2)
    ax4.set_title("Aggregate Fleet CPU Utilization Over Time")
    ax4.set_xlabel("Timestamp")
    ax4.set_ylabel("Mean CPU %")
    ax4.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('eda_deep_dive_plots.png')
    print("\nPlots saved to 'eda_deep_dive_plots.png'")

if __name__ == "__main__":
    cpu_path = 'D:/INTERN_CHALLENGE/data-v1/vm_cpu_readings.csv/vm_cpu_readings.csv'
    meta_path = 'D:/INTERN_CHALLENGE/data-v1/vm_metadata.csv/vm_metadata.csv'
    analyze_anomalies(cpu_path, meta_path)
