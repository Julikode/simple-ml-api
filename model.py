import tensorflow as tf
from pathlib import Path

def load_trained_model():
    # Шлях до твоєї завантаженої моделі метеликів
    model_path = Path("models/resnet50_model.keras")
    if model_path.exists():
        return tf.keras.models.load_model(model_path)
    return None
    
