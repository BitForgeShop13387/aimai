"""
AI Model - AIMAI Project
Author: Mirnes
Date: 2026-02-15
"""

import torch
import numpy as np

class AIModel:
    def __init__(self):
        self.model = self._build_model()
        print("🤖 AI Model inicijalizovan!")

    def _build_model(self):
        """Primer jednostavnog neural networka"""
        return torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )

    def predict(self, input_data):
        """Predikcija na osnovu ulaznih podataka"""
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        return self.model(input_tensor).detach().numpy()

if __name__ == "__main__":
    # Testiraj model
    model = AIModel()
    test_data = np.random.randn(10)
    print(f"Predikcija: {model.predict(test_data)}")