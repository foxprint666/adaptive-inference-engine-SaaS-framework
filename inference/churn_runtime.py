"""
inference/churn_runtime.py

Concrete implementation of ModelRuntime for Scikit-Learn Random Forest churn predictor.
Demonstrates that the ModelRuntime abstraction works with completely different models.
"""

import os
import logging
import pickle
from typing import Dict, Any
import numpy as np
from inference.model_runtime import ModelRuntime

try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    raise ImportError("Scikit-Learn is required. Install with: pip install scikit-learn")

logger = logging.getLogger(__name__)


class ChurnRuntime(ModelRuntime):
    """
    Scikit-Learn Random Forest churn prediction model runtime.
    
    Different from FraudNet:
    - Framework: Scikit-Learn (not PyTorch)
    - Input features: customer_age, tenure_months, monthly_spend, support_tickets, contract_type
    - Output: binary churn prediction (0 or 1)
    """

    def __init__(self, config_path: str):
        """
        Initialize ChurnRuntime.
        
        Args:
            config_path: Path to churn model config.json
        """
        self.model_instance = None
        super().__init__(config_path)

    def load(self, model_path: str) -> None:
        """
        Load Scikit-Learn model from pickle.
        
        Args:
            model_path: Path to .pkl or .joblib file
        """
        if not os.path.exists(model_path):
            logger.warning(f"Model path {model_path} does not exist. Using deterministic demo model.")
            self.model_instance = RandomForestClassifier(
                n_estimators=10,
                max_depth=5,
                random_state=42
            )
            demo_x = np.array([
                [22, 2, 25.0, 4, 0],
                [58, 72, 120.0, 0, 2],
                [35, 8, 45.0, 3, 0],
                [46, 36, 85.0, 1, 1],
            ])
            demo_y = np.array([1, 0, 1, 0])
            self.model_instance.fit(demo_x, demo_y)
            return

        try:
            with open(model_path, "rb") as f:
                self.model_instance = pickle.load(f)
            logger.info(f"Loaded Scikit-Learn model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference on input features.
        
        Args:
            features: Dictionary with keys:
              - customer_age
              - tenure_months
              - monthly_spend
              - support_tickets
              - contract_type (0=month-to-month, 1=one-year, 2=two-year)
            
        Returns:
            {"prediction": 0 or 1, "probability": float}
        """
        if self.model_instance is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Validate features against schema
        self.validate_features(features)

        # Extract in order
        feature_values = [
            features["customer_age"],
            features["tenure_months"],
            features["monthly_spend"],
            features["support_tickets"],
            features["contract_type"],
        ]

        try:
            # Predict probability for class 1 (churned)
            proba = self.model_instance.predict_proba([[*feature_values]])[0]
            pred = self.model_instance.predict([[*feature_values]])[0]
            prob = float(proba[1])  # Probability of class 1 (churn)
            
            return {
                "prediction": int(pred),
                "probability": prob,
            }
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
