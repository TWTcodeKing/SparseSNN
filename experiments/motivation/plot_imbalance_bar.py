"""Plot SM/DRAM utilization imbalance bar charts.

Figure layout: 1 row, 2 columns
  (a) SM Throughput grouped bar: Conv/GEMM vs LIF across B=1..8
  (b) DRAM Throughput grouped bar: Conv/GEMM vs LIF across B=1..8

Usage:
    python experiments/motivation/plot_imbalance_bar.py
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.patches import Patch

matplotlib.rcParams.update({
    'font.size': 14, 'font.family': 'serif',
    'axes.labelsize': 15, 'axes.titlesize': 16,
    'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'legend.fontsize': 11, 'figure.dpi': 150,
})

# ── Data from ncu: Spikformer-1-512, ImageNet 224x224 ──
batches = [1, 2, 4, 8]

compute_sm   = [52.7, 56.7, 59.1, 60.8]
neuron_sm    = [13.5, 14.5,  9.3, 12.7]

compute_dram = [48.7, 52.6, 54.9, 55.3]
neuron_dram  = [71.9, 78.9, 81.8, 82.7]

c_compute = '#F1D77E'
c_neuron  = '#E7DAD2'

# ── Figure ──
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8),
                         gridspec_kw={'wspace': 0.30})

x = np.arange(len(batches)) * 0.5
w = 0.15

# ── (a) SM Throughput ──
ax = axes[0]
ax.bar(x - w/2, compute_sm, w, color=c_compute, edgecolor='black', linewidth=0.5)
ax.bar(x + w/2, neuron_sm, w, color=c_neuron, edgecolor='black', linewidth=0.5)
for i in range(len(batches)):
    ax.text(x[i] - w/2, compute_sm[i] + 1.5, f'{compute_sm[i]:.0f}',
            ha='center', va='bottom', fontsize=25, fontweight='bold', color='#374151')
    ax.text(x[i] + w/2, neuron_sm[i] + 1.5, f'{neuron_sm[i]:.0f}',
            ha='center', va='bottom', fontsize=25, fontweight='bold', color='#374151')
ax.set_ylabel('SM Throughput (%)', fontsize=30)
ax.set_xlabel('Batch Size', fontsize=30)
ax.set_xticks(x)
ax.set_xticklabels([f'B={b}' for b in batches], fontsize=30)
ax.set_ylim(0, 80)
ax.set_yticks([0, 20, 40, 60, 80])
ax.set_yticklabels([0, 20, 40, 60, 80], fontsize=30)
ax.set_title('(a) Compute Utilization', pad=12, fontsize=30)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── (b) DRAM Throughput ──
ax = axes[1]
ax.bar(x - w/2, compute_dram, w, color=c_compute, edgecolor='black', linewidth=0.5)
ax.bar(x + w/2, neuron_dram, w, color=c_neuron, edgecolor='black', linewidth=0.5)
for i in range(len(batches)):
    ax.text(x[i] - w/2, compute_dram[i] + 1.5, f'{compute_dram[i]:.0f}',
            ha='center', va='bottom', fontsize=25, fontweight='bold', color='#374151')
    ax.text(x[i] + w/2, neuron_dram[i] + 1.5, f'{neuron_dram[i]:.0f}',
            ha='center', va='bottom', fontsize=25, fontweight='bold', color='#374151')
ax.set_ylabel('DRAM Throughput (%)', fontsize=30)
ax.set_xlabel('Batch Size', fontsize=30)
ax.set_xticks(x)
ax.set_xticklabels([f'B={b}' for b in batches], fontsize=30)
ax.set_ylim(0, 100)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_yticklabels([0, 20, 40, 60, 80, 100], fontsize=30)
ax.set_title('(b) Memory Bandwidth', pad=12, fontsize=30)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Top legend ──
fig.legend(handles=[Patch(fc=c_compute, ec='black', lw=0.5, label='Conv2d/MatMul'),
                    Patch(fc=c_neuron, ec='black', lw=0.5, label='LIF Neuron')],
           loc='upper center', ncol=2, fontsize=30, framealpha=0.9,
           bbox_to_anchor=(0.5, 1.2))
fig.subplots_adjust(left=0.02, right=0.98, top=0.85, wspace=0.30)
plt.savefig('output/imbalance_bar.pdf', bbox_inches='tight', dpi=300)
plt.savefig('output/imbalance_bar.png', bbox_inches='tight', dpi=300)
print("Saved: output/imbalance_bar.pdf and .png")
