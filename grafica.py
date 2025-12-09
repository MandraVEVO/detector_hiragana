import matplotlib.pyplot as plt
import numpy as np

# Datos
clases = ["a", "i", "u", "e", "o"]
precision = [1.00, 1.00, 1.00, 1.00, 1.00]
recall    = [0.50, 1.00, 0.50, 0.50, 1.00]
f1        = [0.67, 1.00, 0.67, 0.67, 1.00]

x = np.arange(len(clases))
width = 0.25

plt.figure(figsize=(10,6))
plt.bar(x - width, precision, width, label='Precisión')
plt.bar(x, recall, width, label='Recall')
plt.bar(x + width, f1, width, label='F1-score')

plt.xticks(x, clases, fontsize=12)
plt.xlabel("Clase", fontsize=14)
plt.ylabel("Valor", fontsize=14)
plt.title("Métricas de Desempeño por Clase", fontsize=16)
plt.ylim(0, 1.1)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("metricas_grafica.png", dpi=300)
plt.show()
