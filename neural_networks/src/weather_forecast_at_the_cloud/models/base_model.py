from typing import Protocol
import numpy as np
from tensorflow.keras.models import Model as KerasModel

class BaseModel(Protocol):
    """
    A protocol defining the standard interface for all weather forecasting models.
    """
    model: KerasModel
    name: str

    def train(
        self,
        window_generator,
        epochs: int = 20,
        patience: int = 2,
        verbose: int = 1
    ) -> KerasModel:
        ...

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        ...

    def save(self, path: str) -> None:
        ...

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        ...
