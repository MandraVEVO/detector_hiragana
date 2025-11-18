from ultralytics import YOLO
import cv2
from pathlib import Path
import argparse

# Diccionario para convertir caracteres a rōmaji
CHAR_TO_ROMAJI = {
    'あ': 'a', 'い': 'i', 'う': 'u', 'え': 'e', 'お': 'o',
    'か': 'ka', 'き': 'ki', 'く': 'ku', 'け': 'ke', 'こ': 'ko',
    'さ': 'sa', 'し': 'shi', 'す': 'su', 'せ': 'se', 'そ': 'so',
    'た': 'ta', 'ち': 'chi', 'つ': 'tsu', 'て': 'te', 'と': 'to',
    'な': 'na', 'に': 'ni', 'ぬ': 'nu', 'ね': 'ne', 'の': 'no',
    'は': 'ha', 'ひ': 'hi', 'ふ': 'fu', 'へ': 'he', 'ほ': 'ho',
    'ま': 'ma', 'み': 'mi', 'む': 'mu', 'め': 'me', 'も': 'mo',
    'や': 'ya', 'ゆ': 'yu', 'よ': 'yo',
    'ら': 'ra', 'り': 'ri', 'る': 'ru', 'れ': 're', 'ろ': 'ro',
    'わ': 'wa', 'を': 'wo', 'ん': 'n',
    'ア': 'a', 'イ': 'i', 'ウ': 'u', 'エ': 'e', 'オ': 'o',
    'カ': 'ka', 'キ': 'ki', 'ク': 'ku', 'ケ': 'ke', 'コ': 'ko',
    'サ': 'sa', 'シ': 'shi', 'ス': 'su', 'セ': 'se', 'ソ': 'so',
    'タ': 'ta', 'チ': 'chi', 'ツ': 'tsu', 'テ': 'te', 'ト': 'to',
    'ナ': 'na', 'ニ': 'ni', 'ヌ': 'nu', 'ネ': 'ne', 'ノ': 'no',
    'ハ': 'ha', 'ヒ': 'hi', 'フ': 'fu', 'ヘ': 'he', 'ホ': 'ho',
    'マ': 'ma', 'ミ': 'mi', 'ム': 'mu', 'メ': 'me', 'モ': 'mo',
    'ヤ': 'ya', 'ユ': 'yu', 'ヨ': 'yo',
    'ラ': 'ra', 'リ': 'ri', 'ル': 'ru', 'レ': 're', 'ロ': 'ro',
    'ワ': 'wa', 'ヲ': 'wo', 'ン': 'n',
}

def detect_characters(model_path, image_path, confidence=0.5):
    """
    Detecta caracteres hiragana/katakana en una imagen y los convierte a rōmaji.
    
    Args:
        model_path: Ruta al modelo entrenado (.pt)
        image_path: Ruta a la imagen a analizar
        confidence: Umbral de confianza (0-1)
    """
    # Cargar modelo
    model = YOLO(model_path)
    
    # Realizar predicción
    results = model.predict(
        source=image_path,
        conf=confidence,
        save=True,
        show_labels=True,
        show_conf=True
    )
    
    # Convertir detecciones a rōmaji
    romaji_result = []
    for result in results:
        boxes = result.boxes
        print(f"\n✓ Detectados {len(boxes)} caracteres:")
        
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names[class_id]
            romaji = CHAR_TO_ROMAJI.get(class_name, class_name)  # Convertir a rōmaji
            
            romaji_result.append(romaji)
            print(f"  - {class_name} ({romaji}): {confidence:.2%}")
    
    print(f"\n✓ Resultado en rōmaji: {' '.join(romaji_result)}")
    print(f"\n✓ Resultados guardados en: runs/detect/predict/")
    
    return romaji_result

def detect_from_webcam(model_path, confidence=0.5):
    """Detecta caracteres en tiempo real desde la webcam"""
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(0)
    
    print("Presiona 'q' para salir")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.predict(frame, conf=confidence)
        annotated_frame = results[0].plot()
        
        cv2.imshow('Detección Hiragana/Katakana', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Ruta al modelo .pt')
    parser.add_argument('--image', type=str, help='Ruta a la imagen')
    parser.add_argument('--webcam', action='store_true', help='Usar webcam')
    parser.add_argument('--conf', type=float, default=0.5, help='Confianza mínima')
    
    args = parser.parse_args()
    
    if args.webcam:
        detect_from_webcam(args.model, args.conf)
    elif args.image:
        detect_characters(args.model, args.image, args.conf)
    else:
        print("Especifica --image o --webcam")
