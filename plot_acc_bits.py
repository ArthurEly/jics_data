import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator

# --- 1. Coleta e processamento dos dados ---
root_dir = './util_sheets'
axi_keywords = [
    'stream_to_memap_0 (bd_stream_to_memap_0_0)',
    'smartconnect_1 (bd_smartconnect_1_0)',
    'smartconnect_0 (bd_smartconnect_0_0)',
    'memap_to_stream_0 (bd_memap_to_stream_0_0)'
]
finn_keyword = 'finn_design_0 (bd_finn_design_0_0)'
manual_fps = {
    't1w2': {500:542, 5000:6504, 50000:16261},
    't2w2': {500:678, 5000:5422, 50000:10841},
    't1w4': {500:542, 5000:6504, 50000:16261},
    't2w4': {500:678, 5000:5422, 50000:10841},
    't1w8': {500:542, 5000:6504, 50000:16261},
    't2w8': {500:678, 5000:5422, 50000:10841},
}

records = []
for folder in sorted(os.listdir(root_dir)):
    path = os.path.join(root_dir, folder)
    if not os.path.isdir(path): continue
    m = re.search(r'(t\d+w\d+)_(\d+)fps', folder)
    if not m: continue
    config, orig_fps = m.group(1), int(m.group(2))
    fps_val = manual_fps.get(config, {}).get(orig_fps)
    if fps_val is None:
        print(f"Skipped: no manual FPS for {config}@{orig_fps}")
        continue
    xlsx = os.path.join(path, 'util.xlsx')
    if not os.path.isfile(xlsx): continue
    df_sheet = pd.read_excel(xlsx)
    sub_axi = df_sheet[df_sheet['Name'].isin(axi_keywords)]
    axi_sum = sub_axi[['Slice LUTs','Slice Registers','Block RAM Tile','DSPs']].sum()
    sub_finn = df_sheet[df_sheet['Name']==finn_keyword]
    finn_vals = {'Slice LUTs':np.nan,'Slice Registers':np.nan,'Block RAM Tile':np.nan,'DSPs':np.nan}
    if not sub_finn.empty:
        finn_vals.update(sub_finn.iloc[0][['Slice LUTs','Slice Registers','Block RAM Tile','DSPs']].to_dict())
    records.append({
        'Config':config, 'Original_FPS':orig_fps, 'FPS':fps_val,
        'AXI_LUT':axi_sum['Slice LUTs'], 'FINN_LUT':finn_vals['Slice LUTs'],
        'AXI_DFF':axi_sum['Slice Registers'], 'FINN_DFF':finn_vals['Slice Registers'],
        'AXI_BRAM':axi_sum['Block RAM Tile'], 'FINN_BRAM':finn_vals['Block RAM Tile'],
        'AXI_DSP':axi_sum['DSPs'], 'FINN_DSP':finn_vals['DSPs'],
    })

# Assemble DataFrame
df = pd.DataFrame(records)
df = df.sort_values(['Config','FPS']).reset_index(drop=True)
# Save summary
df.to_excel('summary_util_resources.xlsx', index=False)

# Read essential bits + accuracy CSV
csv_df = pd.read_csv('./sat6_ls_z045_stats.csv')
csv_df = csv_df.rename(columns={'Topology':'Config','Speed':'Original_FPS','essential_bits':'Essential_Bits'})# map FPS
print(csv_df)
csv_df['FPS'] = csv_df.apply(lambda r: manual_fps.get(r['Config'],{}).get(r['Original_FPS']), axis=1)
csv_df = csv_df.dropna(subset=['FPS']).astype({'FPS':'int'})
csv_df = csv_df[['Config','FPS','Essential_Bits','total_bits','total_images']]
csv_df = csv_df.rename(columns={'total_bits':'Total_Bits'})
csv_df = csv_df.sort_values(['Config','FPS']).reset_index(drop=True)

# Global styles
label_fs, legend_fs, tick_fs = 24, 22, 18
group_fs = 20
colors = {'axi':'#d62728','finn':'#1f77b4','bits':'#2ca02c','acc':'#ff7f0e'}

def format_fps(n): return f"{n} FPS" if n<1000 else f"{n/1000:.1f}k FPS"

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
    # primary ticks
    ax.set_xticks(x)
    ax.set_xticklabels([format_fps(v) for v in data_df['FPS']], fontsize=tick_fs,
                       rotation=75, ha='center', va='top')
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
    sec.xaxis.set_tick_params(pad=110)  # ou maior, tipo 20, 30
    sec.tick_params(length=0)
    # separators
    sec2 = ax.secondary_xaxis('bottom')
    sec2.set_xticks(edges, labels=[])  # no labels
    sec2.tick_params(length=40, width=1.5, color='black')
    ax.set_xlim(-0.6, len(x)-0.4)

# Plot functions
def plot_stacked(df_plot, col1, col2, ylabel, filename, y_step):
    fig, ax = plt.subplots(figsize=(12,6))
    x = np.arange(len(df_plot))
    ax.bar(x, df_plot[col1], color=colors['axi'], width=0.8, label=f'AMBA AXI Components')
    ax.bar(x, df_plot[col2], bottom=df_plot[col1], color=colors['finn'], width=0.8, label=f'FINN Accelerator')
    ax.set_ylabel(ylabel, fontsize=label_fs)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_yaxis_k))
    ax.yaxis.set_major_locator(MultipleLocator(y_step))  # ajuste aqui!
    ax.tick_params(axis='y', labelsize=label_fs)
    apply_secondary_axes(ax, df_plot)
    ax.legend(fontsize=legend_fs)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved {filename}")

# Agora, chama a função com os steps adequados:
plot_stacked(df, 'AXI_LUT', 'FINN_LUT', 'Total Lookup Tables (LUTs)', 'luts.pdf', y_step=10000)
plot_stacked(df, 'AXI_DFF', 'FINN_DFF', 'Flip-Flops (DFF)', 'ffs.pdf', y_step=10000)
plot_stacked(df, 'AXI_BRAM', 'FINN_BRAM', 'Block RAM (BRAM)', 'bram.pdf', y_step=5)
plot_stacked(df, 'AXI_DSP', 'FINN_DSP', 'DSPs', 'dsps.pdf', y_step=1)  # caso queira também para DSP

# Filtra apenas para total_images == 1
csv_df_filtered = csv_df[csv_df['total_images'] == 1].copy()
csv_df_filtered = csv_df_filtered.sort_values(by=['Config', 'FPS']).reset_index(drop=True)

print(csv_df_filtered)
if not csv_df_filtered.empty:
    # Plot Essential Bits apenas para total_images == 1
    fig, ax = plt.subplots(figsize=(12,6))
    x = np.arange(len(csv_df_filtered))

    # Plot barra de Essential Bits
    ax.bar(x, csv_df_filtered['Essential_Bits'], color=colors['bits'], alpha=0.8, width=0.8, label='Essential Bits')
    ax.set_ylim(0, 16000000)
    # Eixo Y
    ax.set_ylabel('Essential Bits', fontsize=label_fs)
    ax.tick_params(axis='y', labelsize=label_fs)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_yaxis_k))

    # Aplicar eixos secundários para Config e separadores
    apply_secondary_axes(ax, csv_df_filtered)

    # Grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Ajustar layout e salvar
    plt.tight_layout()
    plt.savefig('softcore_essential_bits.pdf')
    plt.close(fig)
    print("Saved softcore_essential_bits.pdf")
else:
    print("No data with total_images == 1 to plot.")


print("All plots generated.")