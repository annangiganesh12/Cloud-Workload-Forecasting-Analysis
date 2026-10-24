import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
try:
    df = pd.read_csv('training_set.csv')
except FileNotFoundError:
    print("Error: training_set.csv not found in the current directory.")
    exit(1)

# Set basic style
plt.style.use('ggplot')
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Exploratory Data Analysis: VM CPU Utilization', fontsize=20)

# 1. Time Series Plot
ax1 = plt.subplot(2, 2, 1)
ax1.plot(df['timestamp'], df['aggregate_cpu_utilization'], color='#1f77b4', linewidth=1)
ax1.set_title('Aggregate CPU Utilization Over Time')
ax1.set_xlabel('Timestamp')
ax1.set_ylabel('CPU Utilization (%)')

# 2. Correlation Heatmap (Manual Matplotlib implementation)
ax2 = plt.subplot(2, 2, 2)
corr_cols = ['aggregate_cpu_utilization', 'active_vm_count',
             'active_vm_count_lag_1', 'active_vm_count_lag_2', 'active_vm_count_lag_3',
             'aggregate_cpu_utilization_lag_1', 'aggregate_cpu_utilization_lag_2',
             'aggregate_cpu_utilization_lag_3', 'cpu_rolling_mean']

existing_cols = [c for c in corr_cols if c in df.columns]
corr = df[existing_cols].corr().values
im = ax2.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
fig.colorbar(im, ax=ax2)

# Add labels to the heatmap
ax2.set_xticks(np.arange(len(existing_cols)))
ax2.set_yticks(np.arange(len(existing_cols)))
ax2.set_xticklabels(existing_cols, rotation=45, ha='right')
ax2.set_yticklabels(existing_cols)
ax2.set_title('Feature Correlation Matrix')

# Add text annotations to the heatmap
for i in range(len(existing_cols)):
    for j in range(len(existing_cols)):
        text = ax2.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        color="white" if abs(corr[i, j]) > 0.5 else "black")

# 3. Distribution Plot (Histogram)
ax3 = plt.subplot(2, 2, 3)
ax3.hist(df['aggregate_cpu_utilization'], bins=30, color='#2ca02c', edgecolor='white', alpha=0.7)
ax3.set_title('Distribution of CPU Utilization')
ax3.set_xlabel('CPU Utilization (%)')
ax3.set_ylabel('Frequency')

# 4. Lag-1 Correlation Scatter
ax4 = plt.subplot(2, 2, 4)
if 'aggregate_cpu_utilization_lag_1' in df.columns:
    ax4.scatter(df['aggregate_cpu_utilization_lag_1'],
                df['aggregate_cpu_utilization'],
                alpha=0.5, color='#d62728', s=10)
    ax4.set_title('CPU Utilization vs Lag-1')
    ax4.set_xlabel('Lag 1 (%)')
    ax4.set_ylabel('Current (%)')
else:
    ax4.text(0.5, 0.5, 'Lag-1 data not available', ha='center', va='center')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('eda_plots.png', dpi=300)
print("EDA plots have been successfully saved to 'eda_plots.png'")
