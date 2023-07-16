# Ultralytics YOLO 🚀, AGPL-3.0 license

from .predict import DecloudPredictor, predict
from .train import DecloudTrainer, train
from .val import DecloudValidator, val

__all__ = 'DecloudPredictor', 'predict', 'DecloudTrainer', 'train', 'DecloudValidator', 'val'
