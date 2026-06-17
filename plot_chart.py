import matplotlib.pyplot as plt
import numpy as np
import os

# Font setting for Chinese characters if possible, or fallback to english if not
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans', 'Arial']

# Data
categories = ['Preprocessing', 'Inference', 'Post-processing']
pc_cpu = [38.90, 279.18, 1.61]
rpi_4 = [157.18, 566.27, 5.48]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, pc_cpu, width, label='PC CPU (Dev Env)')
rects2 = ax.bar(x + width/2, rpi_4, width, label='Raspberry Pi 4 (Edge)')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Time (ms)')
ax.set_title('Pipeline Execution Time Comparison')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

# Save the plot
output_dir = 'assets/images'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'pipeline_time_comparison.jpg'), dpi=300)
print("Plot saved successfully.")
