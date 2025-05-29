import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- 1. Coleta de dados a partir dos arquivos util.xlsx ---
root_dir = '/home/artti/Desktop/bolsa/JICS 2025/util_sheets'

# Palavras-chave para filtragem
axi_keywords = [
    'stream_to_memap_0 (bd_stream_to_memap_0_0)',
    'smartconnect_1 (bd_smartconnect_1_0)',
    'smartconnect_0 (bd_smartconnect_0_0)',
    'memap_to_stream_0 (bd_memap_to_stream_0_0)'
]
finn_keyword = 'finn_design_0 (bd_finn_design_0_0)'

# Dicionário com os FPS que você quer para cada configuração
manual_fps = {
    't1w2':  {500:  542, 5000: 6504, 50000: 16261},
    't2w2':  {500:  678, 5000: 5422, 50000: 10841},
    't1w4':  {500:  542, 5000: 6504, 50000: 16261},
    't2w4':  {500:  678, 5000: 5422, 50000: 10841},
    't1w8':  {500:  542, 5000: 6504, 50000: 16261},
    't2w8':  {500:  678, 5000: 5422, 50000: 10841},
}

records = []
for folder in sorted(os.listdir(root_dir)):
    path = os.path.join(root_dir, folder)
    if not os.path.isdir(path):
        continue

    # extrai tXwY e o fps “original” da pasta
    m = re.search(r'(t\d+w\d+)_(\d+)fps', folder)
    if not m:
        continue
    config   = m.group(1)           # ex: 't1w2'
    orig_fps = int(m.group(2))      # ex: 500

    # pega o fps “manual” correspondente
    fps_val = manual_fps.get(config, {}).get(orig_fps, None)
    if fps_val is None:
        print(f"> Atenção: não defini fps manual para {config} @ {orig_fps}fps")
        continue

    xlsx = os.path.join(path, 'util.xlsx')
    if not os.path.isfile(xlsx):
        print(f"> Atenção: util.xlsx não encontrado em {path}")
        continue

    df_sheet = pd.read_excel(xlsx)
    # soma AXI
    sub_axi = df_sheet[df_sheet['Name'].isin(axi_keywords)]
    axi_sum = sub_axi[['Slice LUTs','Slice Registers','Block RAM Tile','DSPs']].sum()
    # extrai FINN
    sub_finn = df_sheet[df_sheet['Name']==finn_keyword]
    finn_vals = {'Slice LUTs': np.nan, 'Slice Registers': np.nan,
                 'Block RAM Tile': np.nan, 'DSPs': np.nan}
    if not sub_finn.empty:
        finn_vals.update(
            sub_finn.iloc[0][['Slice LUTs','Slice Registers','Block RAM Tile','DSPs']].to_dict()
        )

    # **apenas UM registro** por pasta
    records.append({
        'Config':        config,
        'FPS':           fps_val,
        'AXI_LUT':       axi_sum['Slice LUTs'],
        'FINN_LUT':      finn_vals['Slice LUTs'],
        'AXI_DFF':       axi_sum['Slice Registers'],
        'FINN_DFF':      finn_vals['Slice Registers'],
        'AXI_BRAM':      axi_sum['Block RAM Tile'],
        'FINN_BRAM':     finn_vals['Block RAM Tile'],
        'AXI_DSP':       axi_sum['DSPs'],
        'FINN_DSP':      finn_vals['DSPs'],
        'Essential_Bits': np.nan,
        'Accuracy':       np.nan
    })

# monta DataFrame final
df = pd.DataFrame(records)
df = df.sort_values(['Config','FPS']).reset_index(drop=True)

# --- 2. Salvar XLSX gerado ---
output_xlsx = 'summary_util_resources.xlsx'
df.to_excel(output_xlsx, index=False)
print(f"Resumo salvo em {output_xlsx}")

# --- 2. Leitura do CSV de Essential Bits ---
# Substitua pelo caminho do seu CSV gerado
csv_path = './sat6_ls_z045_stats.csv'
csv_df = pd.read_csv(csv_path, sep=',')
# Ajusta nomes de colunas
csv_df = csv_df.rename(columns={
    'Topology': 'Config',
    'Speed': 'FPS',
    'essential_bits': 'Essential_Bits'
})
print(csv_df.columns)
# Mantém apenas colunas necessárias
csv_df = csv_df[['Config', 'FPS', 'Essential_Bits']]

# --- 4. Labels, formatação e estilos ---
def format_fps(n):
    if n < 1000:
        return f"{int(n)} FPS"
    else:
        return f"{n/1000:.1f}k FPS"

fps_labels = [format_fps(v) for v in df['FPS']]

def y_axis_k_formatter(x, pos):
    if x == 0: return '0'
    if abs(x) >= 1000: return f'{int(x/1000)}k'
    return f'{int(x)}'

configs = sorted(df['Config'].unique(),
                 key=lambda s: (int(s[1]), int(s[3])))
color_finn      = '#1f77b4'
color_axi       = '#d62728'
color_bits_new  = color_finn
color_acc       = '#ff7f0e'

label_fontsize  = 20
legend_fontsize = 18
tick_fontsize   = 16
group_label_fontsize = 18

x_positions = np.arange(len(df))
group_size   = len(df) // len(configs)
middle_index = group_size // 2

def setup_two_level_axis(ax):
    ax.set_xticks(x_positions)
    ax.set_xticklabels(fps_labels,
                       fontsize=tick_fontsize,
                       rotation=90, ha='center', va='center')
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.tick_params(axis='x', which='major', length=5, pad=40)
    ax.set_xlabel(None)

    ymin, ymax = ax.get_ylim()
    y_group_label_pos = ymin - (ymax - ymin) * 0.30

    for i, cfg in enumerate(configs):
        start  = i * group_size
        center = start + middle_index
        ax.text(center, y_group_label_pos, cfg,
                ha='center', va='top',
                fontsize=group_label_fontsize,
                clip_on=False)
        if i > 0:
            ax.axvline(x=start - 0.5,
                       color='grey', linestyle='--', linewidth=1,
                       ymin=0.02, ymax=0.98)

# --- 5. Plot 1: LUT & Dist. RAM ---
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(x_positions, df['AXI_LUT'],  label='AMBA AXI & AXI-S DMA',
        color=color_axi,  width=0.8)
ax1.bar(x_positions, df['FINN_LUT'], bottom=df['AXI_LUT'],
        label='FINN', color=color_finn, width=0.8)
ax1.set_ylabel('LUT & Distr. RAM', fontsize=label_fontsize)
ax1.legend(fontsize=legend_fontsize, frameon=True, edgecolor='black')
ax1.set_ylim(0, df['AXI_LUT'].add(df['FINN_LUT'].fillna(0)).max() * 1.1)
setup_two_level_axis(ax1)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(y_axis_k_formatter))
ax1.grid(axis='y', linestyle='--', alpha=0.7)
plt.subplots_adjust(left=0.08, right=0.96, top=0.96, bottom=0.26)
plt.savefig('grafico_lut_2level_v5.png')
plt.close(fig1)

# --- 6. Plot 2: DFF ---
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.bar(x_positions, df['AXI_DFF'],  label='AMBA AXI & AXI-S DMA',
        color=color_axi, width=0.8)
ax2.bar(x_positions, df['FINN_DFF'], bottom=df['AXI_DFF'],
        label='FINN', color=color_finn, width=0.8)
ax2.set_ylabel('Flip-flops (DFF)', fontsize=label_fontsize)
ax2.legend(fontsize=legend_fontsize, frameon=True, edgecolor='black')
ax2.set_ylim(0, df['AXI_DFF'].add(df['FINN_DFF'].fillna(0)).max() * 1.1)
setup_two_level_axis(ax2)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(y_axis_k_formatter))
ax2.grid(axis='y', linestyle='--', alpha=0.7)
plt.subplots_adjust(left=0.08, right=0.96, top=0.96, bottom=0.26)
plt.savefig('grafico_dff_2level_v5.png')
plt.close(fig2)

# --- 7. Plot 3: BRAM ---
fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.bar(x_positions, df['AXI_BRAM'],  label='AMBA AXI & AXI-S DMA',
        color=color_axi, width=0.8)
ax3.bar(x_positions, df['FINN_BRAM'], bottom=df['AXI_BRAM'],
        label='FINN', color=color_finn, width=0.8)
ax3.set_ylabel('Block RAM (BRAM)', fontsize=label_fontsize)
ax3.legend(fontsize=legend_fontsize, frameon=True, edgecolor='black')
ax3.set_ylim(0, df['AXI_BRAM'].add(df['FINN_BRAM'].fillna(0)).max() * 1.1)
setup_two_level_axis(ax3)
ax3.grid(axis='y', linestyle='--', alpha=0.7)
plt.subplots_adjust(left=0.08, right=0.96, top=0.96, bottom=0.26)
plt.savefig('grafico_bram_2level_v5.png')
plt.close(fig3)

def y_axis_m_formatter(x, pos):
    if x == 0:
        return '0'
    if abs(x) >= 1e6:
        return f'{x/1e6:.1f}M'
    elif abs(x) >= 1e3:
        return f'{int(x/1e3)}k'
    else:
        return f'{int(x)}'

csv_path = './sat6_ls_z045_stats.csv'

# Tente sem separador explícito primeiro (detecção automática)
csv_df = pd.read_csv(csv_path)
print("Colunas detectadas:", csv_df.columns)

# Ajusta nomes de colunas
csv_df = csv_df.rename(columns={
    'Topology': 'Config',
    'essential_bits': 'Essential_Bits'
})

# Mantém apenas colunas necessárias
csv_df = csv_df[['Config', 'FPS', 'Essential_Bits', 'Accuracy']]

# Ordena primeiro por Config (para agrupar os 't1w2') e depois por FPS crescente
csv_df = csv_df.sort_values(by=['Config', 'FPS'], ascending=[True, True])

# Garante index limpo
csv_df = csv_df.reset_index(drop=True)

x_positions = range(len(csv_df))
fig4, ax4 = plt.subplots(figsize=(14, 6))

ax4.plot(x_positions, csv_df['Accuracy'], label='Top-1 Accuracy',
         color=color_acc, marker='o', linewidth=2)
ax4.set_ylabel('Accuracy (%)', fontsize=label_fontsize)
ax4.set_ylim(0, 100)
ax4.yaxis.set_major_formatter(mticker.PercentFormatter())
ax4.tick_params(axis='y', labelsize=tick_fontsize)

ax4_twin = ax4.twinx()
ax4_twin.bar(x_positions, csv_df['Essential_Bits'], label='Essential bits',
             color=color_bits_new, alpha=0.8, width=0.8)
max_bits = csv_df['Essential_Bits'].max() if not csv_df['Essential_Bits'].isna().all() else 1
ax4_twin.set_ylabel('Essential bits', fontsize=label_fontsize)
ax4_twin.set_ylim(0, max_bits * 1.1)
ax4_twin.yaxis.set_major_formatter(mticker.FormatStrFormatter('%0.0E'))
ax4_twin.tick_params(axis='y', labelsize=tick_fontsize)
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(y_axis_m_formatter))
setup_two_level_axis(ax4)

lines, labs = ax4.get_legend_handles_labels()
bars, labs2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines + bars, labs + labs2,
           loc='upper left', fontsize=legend_fontsize,
           frameon=True, edgecolor='black')
ax4.grid(axis='y', linestyle='--', alpha=0.7)
ax4_twin.grid(False)

plt.subplots_adjust(left=0.08, right=0.91, top=0.96, bottom=0.26)
plt.savefig('grafico_acc_bits_2level_v5.png')
plt.close(fig4)

# --- 9. Plot DSPs ---
fig5, ax5 = plt.subplots(figsize=(12, 6))
ax5.bar(x_positions, df['AXI_DSP'], label='AMBA AXI & AXI-S DMA', color=color_axi, width=0.8)
ax5.bar(x_positions, df['FINN_DSP'], bottom=df['AXI_DSP'], label='FINN', color=color_finn, width=0.8)
ax5.set_ylabel('DSPs', fontsize=label_fontsize)
ax5.legend(fontsize=legend_fontsize, frameon=True, edgecolor='black')
ax5.set_ylim(0, df['AXI_DSP'].add(df['FINN_DSP'].fillna(0)).max() * 1.1)
setup_two_level_axis(ax5)
ax5.yaxis.set_major_formatter(mticker.FuncFormatter(y_axis_k_formatter))
ax5.grid(axis='y', linestyle='--', alpha=0.7)
plt.subplots_adjust(left=0.08, right=0.96, top=0.96, bottom=0.26)
plt.savefig('grafico_dsp_2level_v5.png')
plt.close(fig5)

print("Gráficos gerados e salvos usando seus FPS manuais. XLSX salvo como 'summary_util_resources.xlsx'.")
