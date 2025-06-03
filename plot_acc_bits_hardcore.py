import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator

# --- 1. Coleta e processamento dos dados ---
manual_fps = {
    't1w2': {500:542, 5000:6504, 50000:16261},
    't2w2': {500:678, 5000:5422, 50000:10841},
    't1w4': {500:542, 5000:6504, 50000:16261},
    't2w4': {500:678, 5000:5422, 50000:10841},
    't1w8': {500:542, 5000:6504, 50000:16261},
    't2w8': {500:678, 5000:5422, 50000:10841},
}

# Read essential bits + accuracy CSV
csv_df = pd.read_csv('./hardcore/sat6_ls_z045_zynq_stats.csv')
csv_df = csv_df.rename(columns={'Topology':'Config','Speed':'Original_FPS','essential_bits':'Essential_Bits'})  
csv_df['FPS'] = csv_df.apply(lambda r: manual_fps.get(r['Config'],{}).get(r['Original_FPS']), axis=1)
csv_df = csv_df.dropna(subset=['FPS']).astype({'FPS':'int'})
csv_df = csv_df[['Config','FPS','Essential_Bits','total_bits','total_images']]
csv_df = csv_df.rename(columns={'total_bits':'Total_Bits'})
csv_df = csv_df.sort_values(['Config','FPS']).reset_index(drop=True)

# Global styles
label_fs, legend_fs, tick_fs = 24, 22, 18
group_fs = 20
colors = {'bits':'#2ca02c'}

def format_fps(n): 
    return f"{n} FPS" if n<1000 else f"{n/1000:.1f}k FPS"

def format_yaxis_k(val, pos=None):
    if val >= 1e6:
        return f"{int(val // 1_000_000)}M"
    elif val >= 1e3:
        return f"{int(val // 1_000)}k"
    else:
        return str(int(val))

# Setup secondary axes
def apply_secondary_axes(ax, data_df):
    x = np.arange(len(data_df))
    ax.set_xticks(x)
    ax.set_xticklabels([format_fps(v) for v in data_df['FPS']], fontsize=tick_fs,
                       rotation=75, ha='center', va='top')
    configs = sorted(data_df['Config'].unique(), key=lambda s:(int(s[1]),int(s[3])))
    centers, edges = [], [-0.5]
    for cfg in configs:
        idx = data_df[data_df['Config']==cfg].index
        if len(idx):
            centers.append(idx.to_numpy().mean())
            edges.append(idx[-1]+0.5)
    sec = ax.secondary_xaxis('bottom')
    sec.set_xticks(centers)
    sec.set_xticklabels(configs, fontsize=group_fs, rotation=0, va='center')
    sec.xaxis.set_tick_params(pad=110)
    sec.tick_params(length=0)
    sec2 = ax.secondary_xaxis('bottom')
    sec2.set_xticks(edges, labels=[])
    sec2.tick_params(length=40, width=1.5, color='black')
    ax.set_xlim(-0.6, len(x)-0.4)

# Filtra apenas para total_images == 1
csv_df_filtered = csv_df.sort_values(by=['Config', 'FPS']).reset_index(drop=True)

if not csv_df_filtered.empty:
    fig, ax = plt.subplots(figsize=(12,6))
    x = np.arange(len(csv_df_filtered))

    ax.bar(x, csv_df_filtered['Essential_Bits'], color=colors['bits'], alpha=0.8, width=0.8, label='Essential Bits')
    ax.set_ylim(0, 16000000)
    ax.set_ylabel('Essential Bits', fontsize=label_fs)
    ax.tick_params(axis='y', labelsize=label_fs)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_yaxis_k))

    apply_secondary_axes(ax, csv_df_filtered)

    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('hardcore_essential_bits.pdf')
    plt.close(fig)
    print("Saved hardcore_essential_bits.pdf")
else:
    print("No data with total_images == 1 to plot.")

print("Essential Bits plot generated.")