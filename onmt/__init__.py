""" Main entry point of the ONMT library """
from __future__ import division, print_function

from . import inputters
from . import encoders
from . import decoders
from . import models
from . import utils
from . import modules
from .trainer import Trainer

import sys
from .utils import optimizers
utils.optimizers.Optim = optimizers.Optimizer
sys.modules["onmt.Optim"] = utils.optimizers

# For Flake
__all__ = [inputters, encoders, decoders, models,
           utils, modules, "Trainer"]

__version__ = "0.2.0"
