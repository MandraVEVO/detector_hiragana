from ultralytics import YOLO
import yaml

def train_model():
    """Entrena el modelo YOLO con el dataset de hiragana/katakana"""
    
    # Cargar modelo pre-entrenado (recomendado para transfer learning)
    model = YOLO('yolov8n.pt')  # nano model, puedes usar 's', 'm', 'l', 'x'
    
    # Entrenar
    results = model.train(
        data='/home/mandra/Documentos/Python/mineria_hiragana/dataset/dataset.yaml',
        epochs=200,
        imgsz=640,
        batch=16,
        name='hiragana_detector',
        patience=20,
        save=True,
        plots=True
    )
    
    print("\n✓ Entrenamiento completado")
    print(f"Modelo guardado en: runs/detect/hiragana_katakana_detector/")
    
    return results

if __name__ == "__main__":
    train_model()
