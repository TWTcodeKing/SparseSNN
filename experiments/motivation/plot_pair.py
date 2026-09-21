"""Fine-grained pair profiling: Conv+LIF and Linear+LIF aggregate comparison.

Figure layout: 2x2
  (a) Pie: Conv+LIF latency breakdown
  (b) Bar: Conv+LIF SM vs DRAM
  (c) Pie: Linear+LIF latency breakdown
  (d) Bar: Linear+LIF SM vs DRAM

Usage:
    python experiments/motivation/plot_pair.py
"""

import csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import defaultdict
from matplotlib.patches import Patch

matplotlib.rcParams.update({
    'font.size': 14, 'font.family': 'serif',
    'axes.labelsize': 15, 'axes.titlesize': 16,
    'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'legend.fontsize': 11, 'figure.dpi': 150,
})

# ── Colors (same as plot_imbalance) ──
c_compute = '#F1D77E'
c_neuron  = '#E7DAD2'

# ── Parse ncu ──

def parse_ncu(path):
    with open(path) as f:
        lines = [l for l in f if not l.startswith('==') and not l.startswith('Done:')
                 and not l.startswith('Conv') and not l.startswith('Linear')]
    reader = csv.DictReader(lines)
    kernels = defaultdict(dict)
    for row in reader:
        kid, kname = row.get('ID',''), row.get('Kernel Name','')
        metric, val_s = row.get('Metric Name',''), row.get('Metric Value','0')
        if not kid or not kname: continue
        kernels[kid]['name'] = kname
        try: val = float(val_s.replace(',',''))
        except: val = 0.0
        if metric == 'Compute (SM) Throughput': kernels[kid]['sm'] = val
        elif metric == 'DRAM Throughput': kernels[kid]['dram'] = val
        elif metric == 'Duration': kernels[kid]['dur_us'] = val / 1000.0
    return dict(kernels)


def is_compute(name):
    n = name.lower()
    return ('fprop_implicit_gemm' in n or 'xmma_gemm' in n or
            'h16816gemm' in n or 's1688gemm' in n or 'cublas' in n)


def aggregate(path):
    kernels = parse_ncu(path)
    c = {'n': 0, 'us': 0, 'sm': 0, 'dram': 0}
    l = {'n': 0, 'us': 0, 'sm': 0, 'dram': 0}
    for k in kernels.values():
        d = c if is_compute(k['name']) else l
        d['n'] += 1; d['us'] += k.get('dur_us', 0)
        d['sm'] += k.get('sm', 0); d['dram'] += k.get('dram', 0)
    for d in [c, l]:
        n = max(d['n'], 1)
        d['avg_sm'] = d['sm'] / n; d['avg_dram'] = d['dram'] / n
    return c, l


# ── Load data ──
conv_c, conv_l = aggregate('output/ncu_pair_conv.csv')
lin_c, lin_l = aggregate('output/ncu_pair_linear.csv')

# ── Figure: 2x2 ──
fig, axes = plt.subplots(2, 2, figsize=(7, 4.8),
                         gridspec_kw={'hspace': 0.55, 'wspace': 0.30,
                                      'width_ratios': [0.85, 1]})

# ════════════════════════════════════════
# Row 0: Conv+LIF
# ════════════════════════════════════════

# (a) Pie
ax = axes[0, 0]
conv_total = conv_c['us'] + conv_l['us']
sizes = [conv_c['us'] / conv_total * 100, conv_l['us'] / conv_total * 100]
wedges, texts, autotexts = ax.pie(
    sizes, colors=[c_compute, c_neuron], explode=(0.03, 0.03),
    autopct='%1.1f%%', startangle=90, pctdistance=0.6,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
    radius=1.3,
)
for t in autotexts:
    t.set_fontweight('bold')
    t.set_fontsize(14)
ax.set_title('(a) Conv2d+LIF Latency', pad=12, fontsize=14)

# (b) Bar
ax = axes[0, 1]
cats = ['Conv2d', 'LIF']
sm_vals = [conv_c['avg_sm'], conv_l['avg_sm']]
dram_vals = [conv_c['avg_dram'], conv_l['avg_dram']]
xb = np.array([0, 0.2])
w = 0.04
ax.bar(xb - w/2, sm_vals, w, color=[c_compute, c_neuron], edgecolor='black', linewidth=0.5)
ax.bar(xb + w/2, dram_vals, w, color=[c_compute, c_neuron], edgecolor='black', linewidth=0.5,
       hatch='///')
for i, (s, d) in enumerate(zip(sm_vals, dram_vals)):
    ax.text(xb[i] - w/2, s + 1.5, f'{s:.0f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#374151')
    ax.text(xb[i] + w/2, d + 1.5, f'{d:.0f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#374151')
ax.set_xticks(xb)
ax.set_xticklabels(cats, fontsize=14)
ax.set_xlim(-0.08, 0.28)
ax.set_ylabel('Throughput (%)', fontsize=14)
ax.set_ylim(0, 110)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_yticklabels([0, 20, 40, 60, 80, 100], fontsize=12)
ax.set_title('(b) Conv2d+LIF SM vs DRAM', pad=12, fontsize=14)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ════════════════════════════════════════
# Row 1: Linear+LIF
# ════════════════════════════════════════

# (c) Pie
ax = axes[1, 0]
lin_total = lin_c['us'] + lin_l['us']
sizes = [lin_c['us'] / lin_total * 100, lin_l['us'] / lin_total * 100]
wedges, texts, autotexts = ax.pie(
    sizes, colors=[c_compute, c_neuron], explode=(0.03, 0.03),
    autopct='%1.1f%%', startangle=90, pctdistance=0.6,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
    radius=1.3,
)
for t in autotexts:
    t.set_fontweight('bold')
    t.set_fontsize(14)
ax.set_title('(c) MatMul+LIF Latency', pad=12, fontsize=14)

# (d) Bar
ax = axes[1, 1]
cats = ['MatMul', 'LIF']
sm_vals = [lin_c['avg_sm'], lin_l['avg_sm']]
dram_vals = [lin_c['avg_dram'], lin_l['avg_dram']]
ax.bar(xb - w/2, sm_vals, w, color=[c_compute, c_neuron], edgecolor='black', linewidth=0.5)
ax.bar(xb + w/2, dram_vals, w, color=[c_compute, c_neuron], edgecolor='black', linewidth=0.5,
       hatch='///')
for i, (s, d) in enumerate(zip(sm_vals, dram_vals)):
    ax.text(xb[i] - w/2, s + 1.5, f'{s:.0f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#374151')
    ax.text(xb[i] + w/2, d + 1.5, f'{d:.0f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#374151')
ax.set_xticks(xb)
ax.set_xticklabels(cats, fontsize=14)
ax.set_xlim(-0.08, 0.28)
ax.set_ylabel('Throughput (%)', fontsize=14)
ax.set_ylim(0, 110)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_yticklabels([0, 20, 40, 60, 80, 100], fontsize=12)
ax.set_title('(d) MatMul+LIF SM vs DRAM', pad=12, fontsize=14)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Top legend ──
fig.legend(handles=[Patch(fc=c_compute, ec='black', lw=0.5, label='Conv2d/MatMul'),
                    Patch(fc=c_neuron, ec='black', lw=0.5, label='LIF Neuron'),
                    Patch(fc='#AAAAAA', ec='black', lw=0.5, label='SM'),
                    Patch(fc='#AAAAAA', ec='black', lw=0.5, hatch='///', label='DRAM')],
           loc='upper center', ncol=4, fontsize=13, framealpha=0.9,
           bbox_to_anchor=(0.5, 1.1))
fig.subplots_adjust(left=0.02, right=0.98, top=0.90, wspace=0.30)
plt.savefig('output/pair_profile.pdf', bbox_inches='tight', dpi=300)
plt.savefig('output/pair_profile.png', bbox_inches='tight', dpi=300)
print("Saved: output/pair_profile.pdf and .png")
