"""
inference/model_runtime.py

Abstract base class defining the interface for all model runtimes.
Decouples the deep learning framework from HTTP serving and ingestion.

All model implementations must inherit from ModelRuntime and implement:
  - load(model_path: str) -> None
  - predict(features: Dict[str, Any]) -> Dict[str, Any]
  - get_feature_schema() -> BaseModel
"""

import abc
import json
import logging
from typing import Dict, Any
from pydantic import BaseModel, create_model, ValidationError

logger = logging.getLogger(__name__)


class ModelRuntime(abc.ABC):
    """
    Abstract base class enforcing a standard interface for model serving.
    
    Decouples framework-specific loading and execution from the HTTP gateway.
    All downstream models must inherit and implement the abstract methods.
    """

    def __init__(self, config_path: str):
        """
        Initialize the model runtime with a configuration file.
        
        Args:
            config_path: Path to config file (JSON or YAML)
        """
        self.config = self._load_config(config_path)
        self.schema = self.get_feature_schema()
        self.model = None
        logger.info(f"Initialized {self.__class__.__name__} with config from {config_path}")

    def _load_config(self, path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON or YAML file.
        
        Args:
            path: File path (must end in .json or .yaml/.yml)
            
        Returns:
            Dictionary containing configuration
        """
        try:
            with open(path, "r") as f:
                if path.endswith(".json"):
                    return json.load(f)
                else:
                    try:
                        import yaml
                        return yaml.safe_load(f)
                    except ImportError:
                        logger.error("YAML support requires 'pyyaml' package")
                        raise
        except Exception as e:
            logger.error(f"Error loading config from {path}: {e}")
            raise

    @abc.abstractmethod
    def load(self, model_path: str) -> None:
        """
        Loads model weights or serialized binaries into memory.
        
        Implementation must handle:
        - Model initialization
        - Weight loading from disk
        - Device placement (CPU/GPU)
        - Error handling for corrupted/missing files
        
        Args:
            model_path: Path to model binary or weights file
        """
        pass

    @abc.abstractmethod
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a single inference pass after schema verification.
        
        Implementation must:
        - Validate input features against the schema
        - Perform inference
        - Return structured prediction output
        
        Args:
            features: Dictionary of feature name -> value
            
        Returns:
            Dictionary containing prediction results
            Must include at minimum: {"prediction": <output>}
        """
        pass

    def get_feature_schema(self) -> BaseModel:
        """
        Dynamically constructs a Pydantic model representing the expected input features.
        
        Reads from config["features"], which should have structure:
        {
            "feature_name": {
                "type": "float" | "int" | "str" | "bool",
                "description": "..."
            },
            ...
        }
        
        Returns:
            Pydantic BaseModel class for dynamic schema validation
        """
        features_config = self.config.get("features", {})
        fields = {}

        type_mapping = {
            "float": float,
            "int": int,
            "str": str,
            "bool": bool,
        }

        for feature_name, details in features_config.items():
            feature_type = details.get("type", "float")
            if feature_type not in type_mapping:
                logger.error(f"Unknown type '{feature_type}' for feature '{feature_name}'")
                raise ValueError(f"Unsupported type: {feature_type}")
            
            # (type, ...) means required field
            fields[feature_name] = (type_mapping[feature_type], ...)

        try:
            schema = create_model("DynamicSchema", **fields)
            logger.info(f"Created dynamic schema with {len(fields)} features")
            return schema
        except Exception as e:
            logger.error(f"Error creating dynamic schema: {e}")
            raise

    def validate_features(self, features: Dict[str, Any]) -> bool:
        """
        Validate that input features match the expected schema.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            True if valid, raises ValidationError if invalid
        """
        try:
            self.schema(**features)
            return True
        except ValidationError as e:
            logger.error(f"Feature validation failed: {e}")
            raise
