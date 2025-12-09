import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import pandas as pd  # Para timestamp en guardado

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
    'が': 'ga', 'ぎ': 'gi', 'ぐ': 'gu', 'げ': 'ge', 'ご': 'go',
    'ざ': 'za', 'じ': 'ji', 'ず': 'zu', 'ぜ': 'ze', 'ぞ': 'zo',
    'だ': 'da', 'ぢ': 'ji', 'づ': 'zu', 'で': 'de', 'ど': 'do',
    'ば': 'ba', 'び': 'bi', 'ぶ': 'bu', 'べ': 'be', 'ぼ': 'bo',
    'ぱ': 'pa', 'ぴ': 'pi', 'ぷ': 'pu', 'ぺ': 'pe', 'ぽ': 'po',
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
    'ガ': 'ga', 'ギ': 'gi', 'グ': 'gu', 'ゲ': 'ge', 'ゴ': 'go',
    'ザ': 'za', 'ジ': 'ji', 'ズ': 'zu', 'ゼ': 'ze', 'ゾ': 'zo',
    'ダ': 'da', 'ヂ': 'ji', 'ヅ': 'zu', 'デ': 'de', 'ド': 'do',
    'バ': 'ba', 'ビ': 'bi', 'ブ': 'bu', 'ベ': 'be', 'ボ': 'bo',
    'パ': 'pa', 'ピ': 'pi', 'プ': 'pu', 'ペ': 'pe', 'ポ': 'po',
}

class CharacterDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Hiragana/Katakana")
        self.root.geometry("1000x700")
        
        # Cargar modelo automáticamente
        self.default_model_path = "/home/mandra/.pyenv/runs/detect/hiragana_detector7/weights/best.pt"
        self.model = None
        self.current_image = None
        self.original_image = None
        self.cap = None
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        
        self.create_widgets()
        self.load_default_model()
    
    def create_widgets(self):
        # Frame superior: Cargar modelo
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        ttk.Label(top_frame, text="Modelo:").grid(row=0, column=0, padx=5)
        self.model_path = tk.StringVar(value=self.default_model_path)
        ttk.Entry(top_frame, textvariable=self.model_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(top_frame, text="Cambiar Modelo", command=self.load_model).grid(row=0, column=2, padx=5)
        
        # Frame izquierdo
        left_frame = ttk.Frame(self.root, padding="10")
        left_frame.grid(row=1, column=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        
        ttk.Button(left_frame, text="Abrir Imagen", command=self.load_image, width=20).grid(row=0, column=0, pady=5)
        ttk.Button(left_frame, text="Tomar Foto (Webcam)", command=self.open_webcam, width=20).grid(row=1, column=0, pady=5)
        ttk.Button(left_frame, text="Detectar Caracteres", command=self.detect_characters, width=20).grid(row=2, column=0, pady=5)

        # ▶ NUEVO BOTÓN PARA GUARDAR IMAGEN
        ttk.Button(left_frame, text="Guardar Imagen", command=self.save_image, width=20).grid(row=3, column=0, pady=5)
        
        ttk.Label(left_frame, text="Confianza:").grid(row=4, column=0, pady=5)
        self.confidence = tk.DoubleVar(value=0.5)
        ttk.Scale(left_frame, from_=0.1, to=1.0, variable=self.confidence, orient=tk.HORIZONTAL).grid(row=5, column=0, pady=5)
        self.conf_label = ttk.Label(left_frame, text="0.5")
        self.conf_label.grid(row=6, column=0)
        self.confidence.trace('w', lambda *args: self.conf_label.config(text=f"{self.confidence.get():.2f}"))
        
        ttk.Label(left_frame, text="Resultados:").grid(row=7, column=0, pady=(20, 5))
        self.results_text = tk.Text(left_frame, width=30, height=15)
        self.results_text.grid(row=8, column=0, pady=5)
        
        # Frame derecho
        right_frame = ttk.Frame(self.root, padding="10")
        right_frame.grid(row=1, column=1, sticky=(tk.N, tk.W, tk.E, tk.S))
        
        self.canvas = tk.Canvas(right_frame, width=640, height=480, bg='gray')
        self.canvas.pack()
        
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        
        ttk.Label(right_frame, text="Haz clic y arrastra para seleccionar la región del carácter").pack(pady=5)
    
    def load_default_model(self):
        if Path(self.default_model_path).exists():
            try:
                self.model = YOLO(self.default_model_path)
                messagebox.showinfo("Modelo Cargado", f"Modelo cargado desde:\n{self.default_model_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar modelo: {str(e)}")
        else:
            messagebox.showwarning("Modelo no encontrado",
                f"No se encontró el modelo en:\n{self.default_model_path}\n\nPor favor selecciona un modelo manualmente.")
    
    def load_model(self):
        model_file = filedialog.askopenfilename(
            title="Seleccionar modelo YOLO",
            filetypes=[("Modelo YOLO", "*.pt")],
            initialdir="/home/mandra/.pyenv/runs/detect"
        )
        if model_file:
            try:
                self.model = YOLO(model_file)
                self.model_path.set(model_file)
                messagebox.showinfo("Éxito", "Modelo cargado correctamente")
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar modelo: {str(e)}")
    
    def load_image(self):
        image_file = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png")]
        )
        if image_file:
            self.original_image = cv2.imread(image_file)
            self.current_image = self.original_image.copy()
            self.display_image(self.current_image)
    
    def open_webcam(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        
        webcam_window = tk.Toplevel(self.root)
        webcam_window.title("Capturar Foto")
        
        webcam_canvas = tk.Canvas(webcam_window, width=640, height=480)
        webcam_canvas.pack()
        
        def update_frame():
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(img)
                webcam_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                webcam_canvas.img_tk = img_tk
                webcam_canvas.after(10, update_frame)
        
        def capture():
            ret, frame = self.cap.read()
            if ret:
                self.original_image = frame.copy()
                self.current_image = frame.copy()
                self.display_image(self.current_image)
                webcam_window.destroy()
        
        ttk.Button(webcam_window, text="Capturar", command=capture).pack(pady=10)
        update_frame()
    
    def display_image(self, image):
        height, width = image.shape[:2]
        max_width, max_height = 640, 480
        
        scale = min(max_width/width, max_height/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized = cv2.resize(image, (new_width, new_height))
        image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        self.photo = ImageTk.PhotoImage(img_pil)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.config(width=new_width, height=new_height)
    
    def on_mouse_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
    
    def on_mouse_drag(self, event):
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline='red', width=2
        )
    
    def on_mouse_release(self, event):
        if self.start_x and self.start_y:
            x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
            x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)
            
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            orig_height, orig_width = self.original_image.shape[:2]
            
            scale_x = orig_width / canvas_width
            scale_y = orig_height / canvas_height
            
            x1_orig = int(x1 * scale_x)
            y1_orig = int(y1 * scale_y)
            x2_orig = int(x2 * scale_x)
            y2_orig = int(y2 * scale_y)
            
            self.current_image = self.original_image[y1_orig:y2_orig, x1_orig:x2_orig]
    
    def detect_characters(self):
        if self.model is None:
            messagebox.showwarning("Advertencia", "Por favor carga un modelo primero")
            return
        
        if self.current_image is None:
            messagebox.showwarning("Advertencia", "Por favor carga una imagen o toma una foto")
            return
        
        try:
            results = self.model.predict(
                source=self.current_image,
                conf=self.confidence.get(),
                verbose=False
            )
            
            self.results_text.delete(1.0, tk.END)
            romaji_result = []
            
            for result in results:
                boxes = result.boxes
                self.results_text.insert(tk.END, f"Detectados {len(boxes)} caracteres:\n\n")
                
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = result.names[class_id]
                    romaji = CHAR_TO_ROMAJI.get(class_name, class_name)
                    
                    romaji_result.append(romaji)
                    self.results_text.insert(tk.END, f"• {class_name} ({romaji})\n")
                    self.results_text.insert(tk.END, f"  Confianza: {confidence:.2%}\n\n")
                
                annotated = result.plot()
                self.display_image(annotated)
            
            if romaji_result:
                self.results_text.insert(tk.END, f"\nRōmaji: {' '.join(romaji_result)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en la detección: {str(e)}")

    # ------------------------------------------
    # 🚀 NUEVA FUNCIÓN PARA GUARDAR LA IMAGEN
    # ------------------------------------------
    def save_image(self):
        save_dir = Path("/home/mandra/Documentos/Python/mineria_hiragana/test_images")
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.current_image is None:
            messagebox.showwarning("Advertencia", "No hay imagen para guardar.")
            return

        filename = save_dir / f"test_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"

        cv2.imwrite(str(filename), self.current_image)

        messagebox.showinfo("Imagen Guardada", f"Guardada en:\n{filename}")
    
    def __del__(self):
        if self.cap:
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = CharacterDetectorGUI(root)
    root.mainloop()
