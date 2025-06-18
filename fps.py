import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator

# --- 0. Dicionário de FPS manual (RTL) ---
manual_fps_rtl = {
    't1w2': {500:542, 5000:6504, 50000:16261},
    't2w2': {500:678, 5000:5422, 50000:10841},
    't1w4': {500:542, 5000:6504, 50000:16261},
    't2w4': {500:678, 5000:5422, 50000:10841},
    't1w8': {500:542, 5000:6504, 50000:16261},
    't2w8': {500:678, 5000:5422, 50000:10841},
}

# --- 1. Função para Coleta e Processamento dos Dados de um CSV ---
def process_csv_data(filepath, suffix=''):
    """
    Lê um arquivo CSV, processa os dados de FPS e retorna um DataFrame mesclado
    com FPS para single-image (SI) e multiple-image (MI).
    """
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            print(f"Aviso: Arquivo '{filepath}' está vazio.")
            return pd.DataFrame()
    except FileNotFoundError:
        print(f"Erro: Arquivo '{filepath}' não encontrado.")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print(f"Erro: Arquivo '{filepath}' está vazio ou mal formatado (EmptyDataError).")
        return pd.DataFrame()
    except Exception as e:
        print(f"Erro inesperado ao ler '{filepath}': {e}")
        return pd.DataFrame()


    df = df.rename(columns={'Topology':'Config','Speed':'Original_FPS','essential_bits':'Essential_Bits'})
    
    # Verificar se a coluna 'FPS' existe antes de tentar usá-la
    if 'FPS' not in df.columns:
        print(f"Erro: Coluna 'FPS' não encontrada no arquivo '{filepath}'.")
        return pd.DataFrame()
        
    df = df.dropna(subset=['FPS']).astype({'FPS':'int'})
    cols_to_keep = ['Config','Original_FPS','FPS','total_images']
    
    # Verificar se todas as colunas necessárias existem
    for col in cols_to_keep:
        if col not in df.columns:
            print(f"Erro: Coluna '{col}' não encontrada no arquivo '{filepath}' após renomeação.")
            # Retornar um DataFrame vazio ou apenas com as colunas que existem
            # pode ser uma opção, mas para este fluxo, é melhor falhar cedo.
            return pd.DataFrame()
            
    df = df[cols_to_keep]
    df = df.sort_values(['Config','Original_FPS','FPS']).reset_index(drop=True)

    # Filtrar dados total_images == 1 (single)
    df_single = df[df['total_images'] == 1][['Config','Original_FPS','FPS']]
    df_single = df_single.rename(columns={'FPS':f'FPS_single{suffix}'})

    # Filtrar dados total_images == 18 (multiple)
    df_multiple = df[df['total_images'] == 18][['Config','Original_FPS','FPS']]
    df_multiple = df_multiple.rename(columns={'FPS':f'FPS_multiple{suffix}'})

    # Merge para juntar single e multiple na mesma linha pela chave Config + Original_FPS
    # Garantir que as colunas de Config e Original_FPS existam antes do merge
    if df_single.empty and df_multiple.empty:
        # Retorna um dataframe vazio com as colunas esperadas para não quebrar o merge externo
        return pd.DataFrame(columns=['Config', 'Original_FPS', f'FPS_single{suffix}', f'FPS_multiple{suffix}'])
    elif df_single.empty:
        # Adiciona colunas 'FPS_single' vazias se não houver dados single
        df_multiple[f'FPS_single{suffix}'] = np.nan
        return df_multiple[['Config', 'Original_FPS', f'FPS_single{suffix}', f'FPS_multiple{suffix}']].sort_values(['Config','Original_FPS']).reset_index(drop=True)
    elif df_multiple.empty:
        # Adiciona colunas 'FPS_multiple' vazias se não houver dados multiple
        df_single[f'FPS_multiple{suffix}'] = np.nan
        return df_single[['Config', 'Original_FPS', f'FPS_single{suffix}', f'FPS_multiple{suffix}']].sort_values(['Config','Original_FPS']).reset_index(drop=True)
    else:
        fps_merged = pd.merge(df_single, df_multiple, on=['Config','Original_FPS'], how='outer')
        return fps_merged.sort_values(['Config','Original_FPS']).reset_index(drop=True)

# --- 2. Carregar e Processar Dados Softcore ---
softcore_filepath = './sat6_ls_z045_stats.csv'
fps_softcore_merged = process_csv_data(softcore_filepath, suffix='_sc')

if not fps_softcore_merged.empty:
    print("Tabela mesclada FPS Softcore (SI e MI):")
    print(fps_softcore_merged.head())
else:
    print(f"Não foi possível carregar ou processar dados de forma significativa de {softcore_filepath}")

# --- 3. Carregar e Processar Dados Hardcore ---
hardcore_filepath = './hardcore/sat6_ls_z045_zynq_stats.csv'
fps_hardcore_merged = process_csv_data(hardcore_filepath, suffix='_hc')

if not fps_hardcore_merged.empty:
    print("\nTabela mesclada FPS Hardcore (SI e MI):")
    print(fps_hardcore_merged.head())
else:
    print(f"Não foi possível carregar ou processar dados de forma significativa de {hardcore_filepath}")

# --- 4. Merge final dos dados Softcore e Hardcore (LÓGICA REVISADA) ---
all_fps_merged = pd.DataFrame()
data_loaded_successfully = False

# Definir as colunas esperadas para garantir a consistência
expected_sc_cols = ['Config', 'Original_FPS', 'FPS_single_sc', 'FPS_multiple_sc']
expected_hc_cols = ['Config', 'Original_FPS', 'FPS_single_hc', 'FPS_multiple_hc']

# Garantir que os DataFrames tenham as colunas esperadas, preenchendo com NaN se necessário
for col in expected_sc_cols:
    if col not in fps_softcore_merged.columns and ('Config' in col or 'Original_FPS' in col):
        fps_softcore_merged[col] = np.nan # Para Config e Original_FPS, embora devam vir do process_csv_data
    elif col not in fps_softcore_merged.columns:
         fps_softcore_merged[col] = np.nan
for col in expected_hc_cols:
    if col not in fps_hardcore_merged.columns and ('Config' in col or 'Original_FPS' in col):
        fps_hardcore_merged[col] = np.nan
    elif col not in fps_hardcore_merged.columns:
        fps_hardcore_merged[col] = np.nan


if not fps_softcore_merged.drop(columns=['Config', 'Original_FPS'], errors='ignore').isnull().all().all() and \
   not fps_hardcore_merged.drop(columns=['Config', 'Original_FPS'], errors='ignore').isnull().all().all():
    # Ambos têm dados válidos (além de Config/Original_FPS)
    all_fps_merged = pd.merge(fps_softcore_merged, fps_hardcore_merged, on=['Config','Original_FPS'], how='outer')
    print("\nTabela final mesclada (Softcore e Hardcore):")
    data_loaded_successfully = True
elif not fps_softcore_merged.drop(columns=['Config', 'Original_FPS'], errors='ignore').isnull().all().all(): # Apenas Softcore tem dados
    print("Dados Hardcore não puderam ser carregados ou estavam vazios/inválidos. Usando apenas dados Softcore.")
    all_fps_merged = fps_softcore_merged.copy()
    all_fps_merged['FPS_single_hc'] = np.nan
    all_fps_merged['FPS_multiple_hc'] = np.nan
    print("\nTabela de fallback (apenas Softcore):")
    data_loaded_successfully = True
elif not fps_hardcore_merged.drop(columns=['Config', 'Original_FPS'], errors='ignore').isnull().all().all(): # Apenas Hardcore tem dados
    print("Dados Softcore não puderam ser carregados ou estavam vazios/inválidos. Usando apenas dados Hardcore.")
    all_fps_merged = fps_hardcore_merged.copy()
    all_fps_merged['FPS_single_sc'] = np.nan
    all_fps_merged['FPS_multiple_sc'] = np.nan
    print("\nTabela de fallback (apenas Hardcore):")
    data_loaded_successfully = True
else:
    print("Nenhum dado de FPS (Softcore ou Hardcore) válido foi carregado. Não será possível gerar o gráfico.")
    all_fps_merged = pd.DataFrame(columns=expected_sc_cols + [c for c in expected_hc_cols if c not in expected_sc_cols]) # Cria DF vazio com todas as colunas esperadas


if data_loaded_successfully and not all_fps_merged.empty:
    # Verificar se 'Config' e 'Original_FPS' vieram corretamente ou preencher se tudo for NaN
    if 'Config' not in all_fps_merged.columns or all_fps_merged['Config'].isnull().all():
        if not fps_softcore_merged.empty and 'Config' in fps_softcore_merged.columns:
            all_fps_merged['Config'] = fps_softcore_merged['Config']
            all_fps_merged['Original_FPS'] = fps_softcore_merged['Original_FPS']
        elif not fps_hardcore_merged.empty and 'Config' in fps_hardcore_merged.columns:
            all_fps_merged['Config'] = fps_hardcore_merged['Config']
            all_fps_merged['Original_FPS'] = fps_hardcore_merged['Original_FPS']
    
    # Remover linhas onde 'Config' ou 'Original_FPS' são NaN, pois são chave
    all_fps_merged.dropna(subset=['Config', 'Original_FPS'], inplace=True)
    
    if not all_fps_merged.empty:
        print(all_fps_merged.head())
        all_fps_merged.to_csv('fps_all_merged.csv', index=False)
        print("Salvo fps_all_merged.csv")
    else:
        print("Após limpeza, 'all_fps_merged' está vazio. Verifique os dados de 'Config' e 'Original_FPS' nos arquivos CSV.")
        data_loaded_successfully = False # Atualiza status
else:
     if not data_loaded_successfully:
        print("Reiterando: Nenhum dado válido para merge.")


# --- Configurações e Funções de Plotagem ---
label_fs, legend_fs, tick_fs = 24, 22, 18
group_fs = 20
colors = {'softcore_si':'#1f77b4', 'hardcore_si':'#d62728',
          'softcore_mi':'#1f77b4', 'hardcore_mi':'#d62728',
          'rtl_mi': 'green'}

def format_fps(n):
    if pd.isna(n): return "N/A"
    return f"{n} FPS" if n < 1000 else f"{n/1000:.1f}k FPS"

def format_yaxis_k(val, pos=None):
    if val >= 1e6:
        return f"{int(val // 1_000_000)}M"
    elif val >= 1e3:
        return f"{int(val // 1_000)}k"
    else:
        return str(int(val))

def apply_secondary_axes(ax, data_df):
    if data_df.empty or not all(col in data_df.columns for col in ['Config', 'Original_FPS']):
        print("DataFrame para apply_secondary_axes vazio ou faltando colunas 'Config'/'Original_FPS'.")
        return

    x = np.arange(len(data_df))
    ax.set_xticks(x)
    labels = []
    # Usar .get(cfg, {}).get(fps, fps) para fallback mais seguro
    for cfg, fps_val in zip(data_df['Config'], data_df['Original_FPS']):
        manual_value = manual_fps_rtl.get(str(cfg), {}).get(int(fps_val) if pd.notna(fps_val) else 0 , fps_val)
        labels.append(format_fps(manual_value))
    ax.set_xticklabels(labels, fontsize=tick_fs, rotation=75, ha='center', va='top')
    
    # Garantir que Config é string para a ordenação e unique
    data_df['Config'] = data_df['Config'].astype(str)
    unique_configs = sorted(data_df['Config'].unique(), key=lambda s:(int(re.search(r't(\d+)w(\d+)', s).group(1)), int(re.search(r't(\d+)w(\d+)', s).group(2))) if re.search(r't(\d+)w(\d+)', s) else (0,0) )

    centers, edges = [], [-0.5]
    for cfg_val in unique_configs:
        idx = data_df[data_df['Config']==cfg_val].index
        if len(idx):
            centers.append(idx.to_numpy().mean())
            edges.append(idx[-1]+0.5)
        elif centers:
             edges.append(edges[-1])

    if not centers:
        ax.set_xlim(-0.6, len(x)-0.4 if len(x) > 0 else 0.4)
        return

    sec = ax.secondary_xaxis('bottom')
    sec.set_xticks(centers)
    sec.set_xticklabels(unique_configs, fontsize=group_fs, rotation=0, va='center')
    sec.xaxis.set_tick_params(pad=110)
    sec.tick_params(length=0)

    sec2 = ax.secondary_xaxis('bottom')
    sec2.set_xticks(edges, labels=[])
    sec2.tick_params(length=40, width=1.5, color='black')
    ax.set_xlim(-0.6, len(x)-0.4)

# --- 5. Plotagem (CONDIÇÃO DE ENTRADA AJUSTADA) ---
if data_loaded_successfully and not all_fps_merged.empty: # Usar a flag e verificar se não está vazio
    fig, ax = plt.subplots(figsize=(14,7))
    x = np.arange(len(all_fps_merged))

    # --- Softcore SI ---
    # Plotar apenas se a coluna existir e tiver algum dado não-NaN
    if 'FPS_single_sc' in all_fps_merged.columns and all_fps_merged['FPS_single_sc'].notna().any():
        ax.scatter(x, all_fps_merged['FPS_single_sc'], color=colors['softcore_si'], alpha=0.8, s=100, label='Softcore SI', marker='^', edgecolors='k')
        ax.plot(x, all_fps_merged['FPS_single_sc'], color=colors['softcore_si'], alpha=0.7, linewidth=2, linestyle='--')

    # --- Hardcore SI ---
    if 'FPS_single_hc' in all_fps_merged.columns and all_fps_merged['FPS_single_hc'].notna().any():
        ax.scatter(x, all_fps_merged['FPS_single_hc'], color=colors['hardcore_si'], alpha=0.9, s=100, label='Hardcore SI', marker='^', edgecolors='k')
        ax.plot(x, all_fps_merged['FPS_single_hc'], color=colors['hardcore_si'], alpha=0.8, linewidth=2, linestyle='-')
    
    # --- Softcore MI ---
    if 'FPS_multiple_sc' in all_fps_merged.columns and all_fps_merged['FPS_multiple_sc'].notna().any():
        ax.scatter(x, all_fps_merged['FPS_multiple_sc'], color=colors['softcore_mi'], alpha=0.8, s=100, label='Softcore MI', marker='D', edgecolors='k')
        ax.plot(x, all_fps_merged['FPS_multiple_sc'], color=colors['softcore_mi'], alpha=0.7, linewidth=2, linestyle='--')

    # --- Hardcore MI ---
    if 'FPS_multiple_hc' in all_fps_merged.columns and all_fps_merged['FPS_multiple_hc'].notna().any():
        ax.scatter(x, all_fps_merged['FPS_multiple_hc'], color=colors['hardcore_mi'], alpha=0.9, s=100, label='Hardcore MI', marker='D', edgecolors='k')
        ax.plot(x, all_fps_merged['FPS_multiple_hc'], color=colors['hardcore_mi'], alpha=0.8, linewidth=2, linestyle='-')

    # --- Pontos verdes (manual_fps_rtl / RTL MI) ---
    manual_points_values = []
    if 'Config' in all_fps_merged.columns and 'Original_FPS' in all_fps_merged.columns:
        for cfg, fps_orig in zip(all_fps_merged['Config'], all_fps_merged['Original_FPS']):
            try: # Garantir que fps_orig seja um número para a busca no dicionário
                fps_orig_int = int(fps_orig) if pd.notna(fps_orig) else np.nan
                if pd.notna(fps_orig_int):
                     manual_points_values.append(manual_fps_rtl.get(str(cfg), {}).get(fps_orig_int, np.nan))
                else:
                    manual_points_values.append(np.nan)
            except (KeyError, ValueError):
                manual_points_values.append(np.nan)
    
    if manual_points_values:
        manual_points_values = np.array(manual_points_values)
        valid_idx = ~np.isnan(manual_points_values)
        if np.any(valid_idx):
             ax.scatter(x[valid_idx], manual_points_values[valid_idx], color=colors['rtl_mi'], alpha=0.9, s=120, label='RTL MI', marker='o', edgecolors='k', zorder=5)

    # --- Eixo e estética ---
    ax.set_ylabel('Throughput (FPS)', fontsize=label_fs)
    ax.tick_params(axis='y', labelsize=label_fs)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_yaxis_k))

    apply_secondary_axes(ax, all_fps_merged)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Coletar handles e labels explicitamente para depuração e controle
    handles, labels = ax.get_legend_handles_labels()
    if handles: # Somente criar legenda se houver algo para legendar
        ax.legend(handles, labels, fontsize=legend_fs, loc='upper center', bbox_to_anchor=(0.5, -0.42), ncol=3, frameon=False)
    else:
        print("Nenhum item para adicionar à legenda.")

    plt.subplots_adjust(bottom=0.1, top=0.95)
    fig.subplots_adjust(bottom=0.4, top=0.99, left=0.09, right=0.999)
    # plt.tight_layout(rect=[0, 0.1, 1, 1]) # tight_layout pode conflitar com subplots_adjust

    plt.savefig('fps_comparison_updated.pdf')
    plt.close(fig)
    print("\nSalvo fps_comparison_updated.pdf")
else:
    print("Não há dados válidos em 'all_fps_merged' ou o carregamento falhou. Gráfico de Throughput não gerado.")