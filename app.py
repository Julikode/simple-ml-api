import os
from pathlib import Path
import tensorflow as tf
import gdown

MODEL_DIR = Path("resnet50")
MODEL_PATH = MODEL_DIR / "resnet50_model.keras"

MODEL_DIR.mkdir(exist_ok=True)

# Якщо файлу моделі немає, скачуємо його з Google Drive автоматично
if not MODEL_PATH.exists():
    print("Модель не знайдено локально. Завантажуємо з Google Drive...")
    # Заміни < ТВОЇЙ_ID_ФАЙЛУ > на реальний ID з посилання Google Drive
    file_id = "1oVbpRwJKRFlzjZDWqbNCRwtZH3fTLmTG"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(MODEL_PATH), quiet=False)

# Завантажуємо модель у пам'ять
if MODEL_PATH.exists():
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = None
