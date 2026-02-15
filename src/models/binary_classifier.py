"""
Binary Classifier - AIMAI Project
Author: Mirnes
Date: 2026-02-15
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=16):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, X):
        """Predikcija klase (0 ili 1)"""
        with torch.no_grad():
            probs = self.model(torch.tensor(X, dtype=torch.float32))
            return (probs > 0.5).numpy().flatten().astype(int)
    
    def predict_proba(self, X):
        """Predikcija verovatnoće za klasu 1"""
        with torch.no_grad():
            return self.model(torch.tensor(X, dtype=torch.float32)).numpy().flatten()

def train_classifier(X_train, y_train, epochs=100, lr=0.01):
    """Trening binarnog klasifikatora"""
    model = BinaryClassifier(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Konvertuj u torch tenzore
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    print("🚀 Trening binarnog klasifikatora...")
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        
        # Backward pass i optimizacija
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    return model

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 AIMAI - Binary Classifier Demo")
    print("=" * 60)
    
    # Kreiraj sintetičke podatke za trening
    print("\n📊 Generisanje sintetičkih podataka...")
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    
    # Podeli na trening/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Treniraj model
    model = train_classifier(X_train, y_train, epochs=100)
    
    # Evaluacija
    print("\n✅ Evaluacija na test skupu:")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"]))
    
    # Primer predikcije za novi uzorak
    print("\n🔍 Primer predikcije za novi uzorak:")
    sample = X_test[0:1]
    prob = model.predict_proba(sample)[0]
    pred_class = model.predict(sample)[0]
    print(f"Verovatnoća klase 1: {prob:.4f}")
    print(f"Predikcija: Klasa {pred_class}")
    
    print("\n🎉 Trening završen uspešno!")
