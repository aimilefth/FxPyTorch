# FxPyTorch/transparent/trans_tanh.py
import torch
from torch import nn
from typing import Optional
from .activation_logger import (
    ActivationLogger,
    ActivationLoggingScope,
)

class TanhTransparent(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input: torch.Tensor,
        logger: Optional[ActivationLogger] = None,
        **kwargs,
    ) -> torch.Tensor:
        with ActivationLoggingScope(logger, self):
            output = torch.tanh(input)
            if logger:
                logger.log("input", input, self)
                logger.log("output", output, self)
        return output
