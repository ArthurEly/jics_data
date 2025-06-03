import pandas as pd

# Ler a planilha Excel
arquivo = 'summary_power_hardcore.xlsx'
df = pd.read_excel(arquivo)

# Calcular o total de energia
energy_columns = ['Static Power', 'AMBA AXI Components', 'Memory (DDR)', 'FINN Accelerator', 'CPU (ARM)', 'PLL']
df['Total Energy'] = df[energy_columns].sum(axis=1)

# Calcular o percentual de consumo do FINN Accelerator
df['FINN %'] = (df['FINN Accelerator'] / df['Total Energy']) * 100

# Função auxiliar para calcular média e desvio padrão
def calc_stats(mask):
    mean = df.loc[mask, 'FINN %'].mean()
    std = df.loc[mask, 'FINN %'].std()
    return mean, std

# Máscaras para as quantizações
mask_2bit = df['Config'].str.contains('w2')
mask_4bit = df['Config'].str.contains('w4')
mask_8bit = df['Config'].str.contains('w8')

# Cálculos incluindo todos os pontos
mean_2, std_2 = calc_stats(mask_2bit)
mean_4, std_4 = calc_stats(mask_4bit)
mean_8, std_8 = calc_stats(mask_8bit)

# t1w8 na maior throughput
max_throughput_t1w8 = df.loc[(df['Config'] == 't1w8') & (df['Original_FPS'] == 50000), 'FINN %'].values[0]

# Consumo médio CPU
mean_cpu = df['CPU (ARM)'].mean()

# Mostrar resultados
print(f"2-bit: média = {mean_2:.2f}%, desvio padrão = {std_2:.2f}%")
print(f"4-bit: média = {mean_4:.2f}%, desvio padrão = {std_4:.2f}%")
print(f"8-bit (com t1w8@max): média = {mean_8:.2f}%, desvio padrão = {std_8:.2f}%")
print(f"Máximo FINN t1w8 @max throughput: {max_throughput_t1w8:.2f}%")
print(f"Média consumo CPU (NOEL-V): {mean_cpu:.3f} Watts")
