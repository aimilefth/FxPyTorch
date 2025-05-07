# FxPyTorch/transparent/trans_softmax.py
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from .activation_logger import (
    ActivationLogger,
    ActivationLoggingScope,
)


class SoftmaxTransparent(nn.Module):
    def __init__(self, dim: Optional[int] = None):
        super(SoftmaxTransparent, self).__init__()
        self.dim = dim

    def forward(
        self,
        input: torch.Tensor,
        logger: Optional[ActivationLogger] = None,
        **kwargs,  # <- accept and ignore anything extra
    ) -> torch.Tensor:
        with ActivationLoggingScope(logger, self):
            output = F.softmax(input, dim=self.dim)
            if logger:
                logger.log("input", input, self)
                logger.log("output", output, self)
        return output
