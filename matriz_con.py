import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# Leer archivo con resultados
df = pd.read_csv("resultados.csv")

y_true = df["Real"]
y_pred = df["Predicho"]

# ================
# 1. Lista fija de etiquetas japonesas (orden correto a-i-u-e-o)
# ================
hiragana = ["あ", "い", "う", "え", "お"]

# Agregar clase especial
labels = hiragana + ["No detection"]

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred, labels=labels)
cm = np.array(cm)

plt.figure(figsize=(9, 7))
sns.set(font_scale=1.4)

plt.title("Matriz de Confusión con Ceros Resaltados", fontsize=18)

# ===== HEATMAP 1: valores > 0 =====
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    cbar=True,
    mask=(cm == 0),
    linewidths=1,
    linecolor="black",
    vmin=0, vmax=cm.max()
)

# ===== HEATMAP 2: ceros =====
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Reds",
    xticklabels=labels,
    yticklabels=labels,
    cbar=False,
    mask=(cm != 0),
    linewidths=1,
    linecolor="black",
    alpha=0.45
)

plt.xlabel("Predicción", fontsize=16)
plt.ylabel("Real", fontsize=16)

plt.tight_layout()
plt.savefig("matriz_confusion_final.png", dpi=300)
plt.show()
