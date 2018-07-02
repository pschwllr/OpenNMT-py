""" Main entry point of the ONMT library """
from .trainer import Trainer
import sys
from .utils import optimizers
utils.optimizers.Optim = optimizers.Optimizer
sys.modules["onmt.Optim"] = utils.optimizers

__all__ = ["Trainer"]
