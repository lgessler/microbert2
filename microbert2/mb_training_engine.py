# Like
import os
import tempfile
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
import torch.nn as nn

from tango.common import Lazy, Registrable, Tqdm
from tango.integrations.torch import TrainingEngine
from tango.integrations.torch.model import Model
from tango.integrations.torch.optim import LRScheduler, Optimizer
from tango.integrations.torch.train_config import TrainConfig
from tango.integrations.torch.util import move_to_device

from logging import getLogger

from tango.common import Lazy
from tango.integrations.torch import Optimizer, TorchTrainingEngine, TrainingEngine

logger = getLogger(__name__)


@TrainingEngine.register("mb2")
class MB2Engine(TorchTrainingEngine):
    def __init__(self, suffixes: Optional[tuple] = ("bias", "LayerNorm.weight"), *args, **kwargs):
        self.suffixes = suffixes
        super().__init__(*args, **kwargs)

    def _construct_optimizer(self, optimizer: Lazy[Optimizer]) -> Optimizer:
        weight_decay = optimizer._params.get("weight_decay", None)
        print(optimizer._params.get("weight_decay", None))
        if weight_decay is None:
            return optimizer.construct(params=self.model.parameters())
        else:
            param_groups = [[], []]
            for n, p in self.model.named_parameters():
                if not any(n.endswith(nd) for nd in self.suffixes):
                    param_groups[0].append(p)
                    logger.info(f"Assigned {n} to receive decay")
                else:
                    param_groups[1].append(p)
                    logger.info(f"Assigned {n} to NOT receive decay")

            params = [
                {
                    "params": param_groups[0],
                    "weight_decay": weight_decay,
                },
                {
                    "params": param_groups[1],
                    "weight_decay": 0.0,
                },
            ]
            return optimizer.construct(params=params)
