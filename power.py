import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- 1. Coleta e processamento dos dados ---
root_dir = './powers'

manual_fps = {
    't1w2': {500:542, 5000:6504, 50000:16261},
    't2w2': {500:678, 5000:5422, 50000:10841},
    't1w4': {500:542, 5000:6504, 50000:16261},
    't2w4': {500:678, 5000:5422, 50000:10841},
    't1w8': {500:542, 5000:6504, 50000:16261},
    't2w8': {500:678, 5000:5422, 50000:10841},
}

power_components = {
    'amba_axi': [
        'axi_bram_ctrl_0',
        'memap_to_stream_0',
        'smartconnect_0',
        'smartconnect_1',
        'stream_to_memap_0'
    ],
    'memory': 'blk_mem_gen_0',
    'finn': 'finn_design_0',
    'cpu': 'noelv_mc32_0'
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
    rpt = os.path.join(path, 'power.rpt')
    if not os.path.isfile(rpt): 
        print(f"Skipped: no power.rpt in {folder}")
        continue
    
    with open(rpt, 'r') as f:
        lines = f.readlines()
    
    data = {'Static Power': np.nan, 'AMBA AXI Components': 0.0, 'Memory (BRAM)': np.nan, 'FINN Accelerator': np.nan, 'CPU (NOEL-V)': np.nan}
    
    for line in lines:
        line_strip = line.strip()
        line_nospace = re.sub(r'\s+', '', line_strip)
    
        if line_nospace.startswith('|DeviceStatic'):
            match = re.search(r'\|\s*Device\s+Static.*\|\s*([\d\.]+)\s*\|', line_strip)
            if match:
                data['Static Power'] = float(match.group(1))
        else:
            for comp in power_components['amba_axi']:
                comp_nospace = re.sub(r'\s+', '', comp)
                if f'|{comp_nospace}|' in line_nospace:
                    match = re.search(r'\|\s*'+re.escape(comp)+r'\s*\|\s*([\d\.]+)\s*\|', line_strip)
                    if match:
                        data['AMBA AXI Components'] += float(match.group(1))

            for label, comp in {'Memory (BRAM)': 'blk_mem_gen_0', 'FINN Accelerator': 'finn_design_0', 'CPU (NOEL-V)': 'noelv_mc32_0'}.items():
                comp_nospace = re.sub(r'\s+', '', comp)
                if f'|{comp_nospace}|' in line_nospace:
                    match = re.search(r'\|\s*'+re.escape(comp)+r'\s*\|\s*([\d\.]+)\s*\|', line_strip)
                    if match:
                        data[label] = float(match.group(1))
    
    records.append({
        'Config': config,
        'Original_FPS': orig_fps,
        'FPS': fps_val,
        **data
    })

# DataFrame
df = pd.DataFrame(records)
df = df.sort_values(['Config', 'FPS']).reset_index(drop=True)
df.to_excel('summary_power.xlsx', index=False)
print(df)


def format_fps(n): return f"{n} FPS" if n<1000 else f"{n/1000:.1f}k FPS"

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

# --- 2. Plot ---
label_fs, legend_fs, tick_fs = 24, 18, 18
group_fs = 20
colors = {
    'Static Power': '#7f7f7f',
    'AMBA AXI Components': '#d62728',
    'Memory (BRAM)': '#2ca02c',
    'FINN Accelerator': '#1f77b4',
    'CPU (NOEL-V)': '#9467bd'
}

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(df))
bottom = np.zeros(len(df))

for col in ['Static Power', 'Memory (BRAM)', 'CPU (NOEL-V)', 'AMBA AXI Components', 'FINN Accelerator']:
    ax.bar(x, df[col], bottom=bottom, color=colors[col], width=0.8, label=col)
    bottom += df[col].fillna(0)

ax.set_ylabel('Estimated Power (W)', fontsize=label_fs)
ax.tick_params(axis='y', labelsize=label_fs)
ax.set_ylim(0, 2.25)
apply_secondary_axes(ax, df)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))

# Ajuste para deixar espaço extra abaixo
plt.subplots_adjust(bottom=0.38)

# Legenda fora do gráfico, abaixo e centralizada
ax.legend(
    fontsize=legend_fs,
    loc='upper center',
    bbox_to_anchor=(0.45, -0.55),
    ncol=3,  # pode ajustar o número de colunas dependendo do espaço
    frameon=False
)

plt.tight_layout()
plt.savefig('softcore_power.pdf')
plt.close(fig)

print("Saved softcore_power.pdf")