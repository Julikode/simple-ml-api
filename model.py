"""
Train and save a simple model.
Run this once: python model.py
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier


def train_and_save():
    """Train a dummy model and save it."""
    # Generate fake data
    np.random.seed(42)
    n = 1000
    
    X = pd.DataFrame({
        'age': np.random.randint(18, 70, n),
        'income': np.random.randint(20000, 150000, n),
        'credit_score': np.random.randint(300, 850, n),
    })
    
    # Fake target: approve if credit_score > 600 and income > 40000
    y = ((X['credit_score'] > 600) & (X['income'] > 40000)).astype(int)
    
    # Train
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Save
    Path('models').mkdir(exist_ok=True)
    joblib.dump(model, 'models/model.joblib')
    print("Model saved to models/model.joblib")


if __name__ == "__main__":
    train_and_save()
