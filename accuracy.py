import matplotlib.pyplot as plt
import numpy as np

label_fs, legend_fs, tick_fs = 24, 18, 18
group_fs = 20

# Dados
configs = ['t1w2', 't2w2', 't1w4', 't2w4', 't1w8', 't2w8']
pytorch_acc = [4.59, 89.19, 86.68, 96.7, 98.58, 98.8]
pynq_acc = [3.8, 89.3, 86.7, 96.4, 99.3, 99.2]

x = np.arange(len(configs))

# Estilo
plt.figure(figsize=(10, 6))

# Cores
color_pytorch = '#ff7f0e'  # Laranja
color_hardware = '#1f77b4' # Azul (igual FINN)

# Plot Pytorch
plt.plot(x, pytorch_acc, marker='o', label='Pytorch Accuracy (81k test images)',
         linewidth=2, markersize=8, color=color_pytorch)

# Plot PYNQ / FINN Accelerator
plt.plot(x, pynq_acc, marker='s', label='FINN Accelerator Accuracy (1k test images)',
         linewidth=2, markersize=8, color=color_hardware)

# Eixos
plt.xticks(x, configs, fontsize=tick_fs)
plt.yticks(np.arange(0, 101, 10), fontsize=tick_fs)
plt.ylabel('Top-1 Accuracy (%)', fontsize=label_fs)  # ✅ Aqui ajustado
plt.xlabel('Configuration', fontsize=label_fs)
plt.ylim(0, 100)

# Grade e legenda
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=legend_fs)

# Layout
plt.tight_layout()

# Salvar figura
plt.savefig('accuracy_comparison.pdf')
plt.show()