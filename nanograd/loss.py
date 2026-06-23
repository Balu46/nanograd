import numpy as np
import math
import random
from .tensor import Tensor

class MSE:
    """
    Mean Squared Error (MSE) loss function.
    Calculates the mean squared difference between predictions and targets.
    """
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        n = y_pred.data.size
        loss = ((y_pred - y_true) ** 2).sum() / n
        return loss