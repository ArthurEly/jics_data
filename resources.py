import os
import re
import pandas as pd

# Caminho raiz onde estão todas as pastas
root_dir = '/home/artti/Desktop/bolsa/JICS 2025/util_sheets'

# Palavras-chave para filtrar os dados
axi_keywords = [
    'stream_to_memap_0 (bd_stream_to_memap_0_0)',
    'smartconnect_1 (bd_smartconnect_1_0)',
    'smartconnect_0 (bd_smartconnect_0_0)',
    'memap_to_stream_0 (bd_memap_to_stream_0_0)'
]
finn_keyword = 'finn_design_0 (bd_finn_design_0_0)'

# Para guardar os resultados de todas as pastas
results = []

for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    if os.path.isdir(folder_path):
        # Extrair t'x'w'y e zzzfps do nome da pasta usando regex
        # Exemplo: sat6_ls_z045_rad_t1w2_500fps_2023.2_v0.2
        match = re.search(r't(\dw\d)_([0-9]+)fps', folder_name)
        if match:
            tw = match.group(1)   # ex: '1w2'
            fps = match.group(2)  # ex: '500'
        else:
            tw = 'unknown'
            fps = 'unknown'
        
        file_path = os.path.join(folder_path, 'util.xlsx')
        if os.path.isfile(file_path):
            # Ler a planilha
            df = pd.read_excel(file_path)
            
            # Filtrar dados AXI e somar
            axi_df = df[df['Name'].isin(axi_keywords)][['Slice LUTs', 'Slice Registers', 'Block RAM Tile', 'DSPs']]
            axi_sum = axi_df.sum()
            
            # Dados FINN
            finn_df = df[df['Name'] == finn_keyword][['Slice LUTs', 'Slice Registers', 'Block RAM Tile', 'DSPs']]
            
            # Guardar resultado numa lista
            results.append({
                'Folder': folder_name,
                't_w': tw,
                'fps': fps,
                'AXI_Slice_LUTs': axi_sum['Slice LUTs'],
                'AXI_Slice_Regs': axi_sum['Slice Registers'],
                'AXI_BRAM': axi_sum['Block RAM Tile'],
                'AXI_DSPs': axi_sum['DSPs'],
                'FINN_Slice_LUTs': finn_df['Slice LUTs'].values[0] if not finn_df.empty else None,
                'FINN_Slice_Regs': finn_df['Slice Registers'].values[0] if not finn_df.empty else None,
                'FINN_BRAM': finn_df['Block RAM Tile'].values[0] if not finn_df.empty else None,
                'FINN_DSPs': finn_df['DSPs'].values[0] if not finn_df.empty else None
            })
        else:
            print(f"Arquivo util.xlsx não encontrado em {folder_path}")

# Criar DataFrame com tudo
result_df = pd.DataFrame(results)

# Exibir resultado
print(result_df)
