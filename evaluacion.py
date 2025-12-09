from ultralytics import YOLO
import os
import pandas as pd
model = YOLO("/home/mandra/.pyenv/runs/detect/hiragana_detector7/weights/best.pt")

test_folder = "/home/mandra/Documentos/Python/mineria_hiragana/test_images"

# Mapea los nombres reales
expected_label = {
    "a": "あ",
    "i": "い",
    "u": "う",
    "e": "え",
    "o": "お"
}

results_summary = []

# --- RECORRER IMÁGENES DE PRUEBA ---
for fname in sorted(os.listdir(test_folder)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    # Obtener la clase real desde el nombre del archivo
    real_class_key = fname.split("_")[0]  # ej. "a" de "a_1.jpeg"
    real_class = expected_label.get(real_class_key, "???")

    img_path = os.path.join(test_folder, fname)

    # Predicción
    pred = model.predict(img_path, conf=0.5, verbose=False)[0]

    # Extraer predicción
    if len(pred.boxes) == 0:
        predicted = "No detection"
    else:
        class_id = int(pred.boxes[0].cls[0])
        predicted = pred.names[class_id]

    # Guardar en lista
    results_summary.append([fname, real_class, predicted])
    print(f"{fname}: Real={real_class}, Predicho={predicted}")

# --- CREAR DATAFRAME ---
df = pd.DataFrame(results_summary, columns=["Imagen", "Real", "Predicho"])

# --- GUARDAR CSV ---
output_csv = "resultados.csv"
df.to_csv(output_csv, index=False)

print("\n✓ Archivo generado con éxito:", output_csv)
print(df)