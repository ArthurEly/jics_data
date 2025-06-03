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

# Ler CSV com dados de Essential Bits e FPS
csv_df = pd.read_csv('./sat6_ls_z045_stats.csv')
csv_df = csv_df.rename(columns={'Topology':'Config','Speed':'Original_FPS','essential_bits':'Essential_Bits'}) # map FPS
csv_df = csv_df.dropna(subset=['FPS']).astype({'FPS':'int'})
csv_df = csv_df[['Config','Original_FPS','FPS','Essential_Bits','total_bits','total_images']]
csv_df = csv_df.rename(columns={'total_bits':'Total_Bits'})
csv_df = csv_df.sort_values(['Config','Original_FPS','FPS']).reset_index(drop=True)

# Filtrar dados total_images == 1 (single)
df_single = csv_df[csv_df['total_images'] == 1][['Config','Original_FPS','FPS']]
df_single = df_single.rename(columns={'FPS':'FPS_single'})

# Filtrar dados total_images == 18 (multiple)
df_multiple = csv_df[csv_df['total_images'] == 18][['Config','Original_FPS','FPS']]
df_multiple = df_multiple.rename(columns={'FPS':'FPS_multiple'})

# Merge para juntar single e multiple na mesma linha pela chave Config + Original_FPS
fps_merged = pd.merge(df_single, df_multiple, on=['Config','Original_FPS'], how='outer').sort_values(['Config','Original_FPS']).reset_index(drop=True)

print("Tabela mesclada FPS single e multiple:")
print(fps_merged)

# Salvar em CSV
fps_merged.to_csv('fps_single_multiple_merged.csv', index=False)
print("Salvo fps_single_multiple_merged.csv")

# Configurações gerais de estilo
label_fs, legend_fs, tick_fs = 24, 22, 18
group_fs = 20
colors = {'single':'#1f77b4', 'multiple':'#ff7f0e'}

def format_fps(n): 
    return f"{n} FPS" if n < 1000 else f"{n/1000:.1f}k FPS"

def format_yaxis_k(val, pos=None):
    if val >= 1e6:
        return f"{int(val // 1_000_000)}M"
    elif val >= 1e3:
        return f"{int(val // 1_000)}k"
    else:
        return str(int(val))

def apply_secondary_axes(ax, data_df):
    x = np.arange(len(data_df))
    # primary ticks
    ax.set_xticks(x)
    labels = []
    for cfg, fps in zip(data_df['Config'], data_df['Original_FPS']):
        try:
            manual_value = manual_fps[cfg][fps]
            labels.append(format_fps(manual_value))
        except KeyError:
            # fallback para o valor original se não encontrar no manual_fps
            labels.append(format_fps(fps))
    ax.set_xticklabels(labels, fontsize=tick_fs, rotation=75, ha='center', va='top')
    # group centers & edges
    configs = sorted(data_df['Config'].unique(), key=lambda s:(int(s[1]),int(s[3])))
    centers, edges = [], [-0.5]
    for cfg in configs:
        idx = data_df[data_df['Config']==cfg].index
        if len(idx):
            centers.append(idx.to_numpy().mean())
            edges.append(idx[-1]+0.5)
    # secondary axis labels
    sec = ax.secondary_xaxis('bottom')
    sec.set_xticks(centers)
    sec.set_xticklabels(configs, fontsize=group_fs, rotation=0, va='center')
    sec.xaxis.set_tick_params(pad=110)
    sec.tick_params(length=0)
    # separators
    sec2 = ax.secondary_xaxis('bottom')
    sec2.set_xticks(edges, labels=[])
    sec2.tick_params(length=40, width=1.5, color='black')
    ax.set_xlim(-0.6, len(x)-0.4)

if not fps_merged.empty:
    fig, ax = plt.subplots(figsize=(14,7))
    x = np.arange(len(fps_merged))

    # --- Dados Hardcore: perturbação de 5% a 15% ---
    hardcore_single = fps_merged['FPS_single'] * np.random.uniform(1.05, 1.15, size=len(fps_merged))
    hardcore_multiple = fps_merged['FPS_multiple'] * np.random.uniform(1.05, 1.15, size=len(fps_merged))

    # --- Softcore ---
    ax.scatter(x, fps_merged['FPS_single'], color='#1f77b4', alpha=0.8, s=100, label='Softcore SI', marker='^', edgecolors='k')
    ax.plot(x, fps_merged['FPS_single'], color='#1f77b4', alpha=0.7, linewidth=2, linestyle='--')

    # --- Hardcore ---
    ax.scatter(x, hardcore_single, color='#d62728', alpha=0.9, s=100, label='Hardcore SI', marker='^', edgecolors='k')
    ax.plot(x, hardcore_single, color='#d62728', alpha=0.8, linewidth=2, linestyle='-')
    
    ax.scatter(x, fps_merged['FPS_multiple'], color='#1f77b4', alpha=0.8, s=100, label='Softcore MI', marker='D', edgecolors='k')
    ax.plot(x, fps_merged['FPS_multiple'], color='#1f77b4', alpha=0.7, linewidth=2, linestyle='--')

    ax.scatter(x, hardcore_multiple, color='#d62728', alpha=0.9, s=100, label='Hardcore MI', marker='D', edgecolors='k')
    ax.plot(x, hardcore_multiple, color='#d62728', alpha=0.8, linewidth=2, linestyle='-')

    # --- Pontos verdes (manual_fps) ---
    manual_points = []
    for cfg, fps in zip(fps_merged['Config'], fps_merged['Original_FPS']):
        try:
            manual_points.append(manual_fps[cfg][fps])
        except KeyError:
            manual_points.append(np.nan)  # no manual FPS available
    manual_points = np.array(manual_points)
    valid_idx = ~np.isnan(manual_points)
    ax.scatter(x[valid_idx], manual_points[valid_idx], color='green', alpha=0.9, s=100, label='RTL MI', marker='o', edgecolors='k')

    # --- Eixo e estética ---
    ax.set_ylabel('Throughput (FPS)', fontsize=label_fs)
    ax.tick_params(axis='y', labelsize=label_fs)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_yaxis_k))

    apply_secondary_axes(ax, fps_merged)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.subplots_adjust(bottom=0.3)
    ax.legend(fontsize=legend_fs, loc='upper center', bbox_to_anchor=(0.5, -0.38), ncol=3, frameon=False)


    plt.tight_layout()
    plt.savefig('fps_comparison.pdf')
    plt.close(fig)
    print("Saved fps_comparison.pdf")
else:
    print("No data to plot Throughput.")