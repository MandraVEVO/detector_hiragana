import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

# ================================
# 1. Cargar resultados.csv
# ================================
df = pd.read_csv("resultados.csv")

# ================================
# 2. Filtrar "No detection"
# ================================
df_filtrado = df[df["Predicho"] != "No detection"]

if df_filtrado.empty:
    print("❌ No hay detecciones válidas (Predicho != 'No detection').")
    exit()

y_true = df_filtrado["Real"]
y_pred = df_filtrado["Predicho"]

# Obtener clases ordenadas
labels = sorted(y_true.unique())

# ================================
# 3. Calcular métricas por clase
# ================================
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=labels, zero_division=0
)

# Para graficar
metricas_df = pd.DataFrame({
    "Clase": labels,
    "Precisión": precision,
    "Recall": recall,
    "F1-score": f1
})

print("\n=== MÉTRICAS ===")
print(metricas_df)

# ================================
# 4. Función para graficar
# ================================
def graficar_metricas(df, columna, titulo, filename):
    plt.figure(figsize=(7, 5))
    plt.bar(df["Clase"], df[columna])
    plt.ylim(0, 1.05)
    plt.title(titulo, fontsize=15)
    plt.xlabel("Clases (Hiragana)", fontsize=12)
    plt.ylabel(columna, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    # Guardar imagen
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"✔ Gráfica guardada como: {filename}")


# ================================
# 5. Generar gráficas
# ================================
graficar_metricas(metricas_df, "Precisión", "Precisión por Clase", "precision_por_clase.png")
graficar_metricas(metricas_df, "Recall", "Recall por Clase", "recall_por_clase.png")
graficar_metricas(metricas_df, "F1-score", "F1-score por Clase", "f1score_por_clase.png")
