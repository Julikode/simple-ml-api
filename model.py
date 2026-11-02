import tensorflow as tf
from pathlib import Path
import gdown

def load_trained_model():
    # Шлях до папки та файлу моделі
    model_dir = Path("resnet50")
    model_path = model_dir / "resnet50_model.keras"
    
    # Створюємо папку, якщо її немає
    model_dir.mkdir(exist_ok=True)
    
    # Якщо файлу моделі немає локально, завантажуємо його з Google Диска
    if not model_path.exists():
        print("Модель не знайдено локально. Завантажуємо з Google Диска...")
        file_id = "1oVbpRwJKRFlzjZDWqbNCRwtZH3fTLmTG"  # твій ID без зайвих знаків
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(model_path), quiet=False)
        
    # Завантажуємо та повертаємо модель
    if model_path.exists():
        return tf.keras.models.load_model(model_path)
        
    return None
