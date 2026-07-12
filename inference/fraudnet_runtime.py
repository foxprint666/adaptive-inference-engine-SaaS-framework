"""
inference/fraudnet_runtime.py

Concrete implementation of ModelRuntime for the FraudNet PyTorch model.
Handles PyTorch-specific loading, device placement, and inference.
"""

import os
import logging
from typing import Dict, Any
import torch
import torch.nn as nn
from inference.model_runtime import ModelRuntime

logger = logging.getLogger(__name__)


class FraudNet(nn.Module):
    """Simple 5-input PyTorch neural network for fraud detection."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)


class FraudNetRuntime(ModelRuntime):
    """
    PyTorch FraudNet model runtime.
    
    Inherits from ModelRuntime and implements:
    - load() — loads PyTorch .pt files
    - predict() — runs inference with schema validation
    """

    def __init__(self, config_path: str, device: str = "cpu"):
        """
        Initialize FraudNetRuntime.
        
        Args:
            config_path: Path to config.json or config.yaml
            device: "cpu" or "cuda" (default: "cpu")
        """
        self.device = torch.device(device)
        self.model_instance = None
        super().__init__(config_path)

    def load(self, model_path: str) -> None:
        """
        Load PyTorch model weights from disk.
        
        Args:
            model_path: Path to .pt file
        """
        if not os.path.exists(model_path):
            logger.warning(f"Model path {model_path} does not exist. Initializing random weights.")
            self.model_instance = FraudNet().to(self.device)
            self.model_instance.eval()
            return

        try:
            self.model_instance = FraudNet().to(self.device)
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model_instance.load_state_dict(state_dict)
            self.model_instance.eval()
            logger.info(f"Loaded FraudNet model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference on input features.
        
        Args:
            features: Dictionary with keys: amount, distance, velocity, age, risk_score
            
        Returns:
            {"prediction": 0 or 1, "probability": float}
        """
        if self.model_instance is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Validate features against schema
        self.validate_features(features)

        # Extract in order matching FraudNet input shape (5,)
        feature_values = [
            features["amount"],
            features["distance"],
            features["velocity"],
            features["age"],
            features["risk_score"],
        ]

        try:
            x = torch.tensor([feature_values], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                prob = self.model_instance(x).item()
            
            pred = 1 if prob >= 0.5 else 0
            
            return {
                "prediction": pred,
                "probability": prob,
            }
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
