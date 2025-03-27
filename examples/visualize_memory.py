# examples/visualize_memory.py
import matplotlib.pyplot as plt
import numpy as np

# Sample data from your results
seq_lengths = [3584, 4096, 4608]
components = ["after_embeddings", "after_layer_0", "after_layer_1", "after_layer_2", 
             "after_layer_3", "after_layer_4", "after_layer_5", "after_final_ln", "after_lm_head"]

# Memory values for each sequence length
memory_3584 = [14.82, 923.61, 1217.60, 1217.60, 927.17, 1217.60, 1217.60, 1217.60, 487.27]
memory_4096 = [-432.65, 279.95, 211.96, 219.96, 219.96, 219.96, 219.96, 219.96, 819.95]
memory_4608 = [-1247.55, 197.31, 208.81, 208.81, 208.81, 208.81, 208.82, 208.82, -31.44]

# Plot the data
fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(components))
width = 0.25

ax.bar(x - width, memory_3584, width, label='3584 tokens')
ax.bar(x, memory_4096, width, label='4096 tokens')
ax.bar(x + width, memory_4608, width, label='4608 tokens')

ax.set_ylabel('Memory Usage (MB)')
ax.set_title('Memory Usage Across Model Components by Sequence Length')
ax.set_xticks(x)
ax.set_xticklabels(components, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig('memory_anomaly_visualization.png')
plt.show()