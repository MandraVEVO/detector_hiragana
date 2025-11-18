import os
from pathlib import Path

def rename_images(directory, prefix="a"):
    """
    Renombra las imágenes en una carpeta con un formato específico.
    
    Args:
        directory (str): Ruta a la carpeta con las imágenes.
        prefix (str): Prefijo para los nombres de archivo (por defecto "a").
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"⚠️ La carpeta {directory} no existe.")
        return

    # Filtrar solo archivos de imagen (.jpg, .png)
    image_files = sorted(dir_path.glob("*.jpg")) + sorted(dir_path.glob("*.png"))
    if not image_files:
        print(f"⚠️ No se encontraron imágenes en {directory}.")
        return

    print(f"Renombrando imágenes en {directory} con el prefijo '{prefix}'...")

    for i, image_file in enumerate(image_files, start=1):
        # Crear el nuevo nombre con formato a_001.jpg
        new_name = f"{prefix}_{i:03d}{image_file.suffix}"
        new_path = dir_path / new_name

        # Renombrar el archivo
        image_file.rename(new_path)
        print(f"✓ {image_file.name} → {new_name}")

    print("✓ Renombrado completado.")

if __name__ == "__main__":
    # Configurar la carpeta y el prefijo
    images_dir = "dataset/raw_images/hiragana_お"  # Cambiar según la carpeta deseada
    prefix = "o"  # Cambiar el prefijo según sea necesario

    rename_images(images_dir, prefix)
