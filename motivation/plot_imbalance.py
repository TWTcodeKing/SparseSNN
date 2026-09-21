"""Plot latency breakdown pie chart for Spikformer-1-512 on ImageNet.

Single pie chart: Conv/GEMM vs LIF Neuron vs Reshape at B=8.

Usage:
    python motivation/plot_imbalance.py
"""

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch

matplotlib.rcParams.update({
    'font.size': 14, 'font.family': 'serif',
    'figure.dpi': 150,
})

# ── Data from ncu: Spikformer-1-512, ImageNet 224x224, B=8 ──
compute_pct = 45.9
neuron_pct  = 48.4
other_pct   = 5.7

c_compute = '#F1D77E'
c_neuron  = '#E7DAD2'
c_other   = '#9CA3AF'

# ── Figure ──
fig, ax = plt.subplots(1, 1, figsize=(5, 4.8))

sizes = [compute_pct, neuron_pct, other_pct]
colors = [c_compute, c_neuron, c_other]
explode = (0.03, 0.03, 0.03)
wedges, texts, autotexts = ax.pie(
    sizes, colors=colors, explode=explode,
    autopct='%1.1f%%', startangle=90, pctdistance=0.6,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
    radius=1.3,
)
for t in autotexts:
    t.set_fontweight('bold')
    t.set_fontsize(25)
ax.set_title('Latency Breakdown (B=8)', pad=12, fontsize=22)

fig.legend(handles=[Patch(fc=c_compute, ec='black', lw=0.5, label='Conv/MatMul'),
                    Patch(fc=c_neuron, ec='black', lw=0.5, label='LIF Neuron'),
                    Patch(fc=c_other, ec='black', lw=0.5, label='Reshape')],
           loc='upper center', ncol=3, fontsize=14, framealpha=0.9,
           bbox_to_anchor=(0.5, 1.05))
fig.subplots_adjust(top=0.85)
plt.savefig('output/imbalance.pdf', bbox_inches='tight', dpi=300)
plt.savefig('output/imbalance.png', bbox_inches='tight', dpi=300)
print("Saved: output/imbalance.pdf and .png")
