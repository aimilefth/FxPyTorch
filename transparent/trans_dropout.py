# transparent/trans_dropout.py
import torch
from torch import nn
from typing import Optional
from .activation_logger import (
    ActivationLogger,
    ActivationLoggingScope,
)


class DropoutTransparent(nn.Dropout):
    def __init__(
        self,
        p: float = 0.5,
        inplace: bool = False,
    ) -> None:
        super(DropoutTransparent, self).__init__(p, inplace)

    def forward(
        self,
        input: torch.Tensor,
        logger: Optional[ActivationLogger] = None,
        **kwargs,  # <- accept and ignore anything extra
    ) -> torch.Tensor:
        with ActivationLoggingScope(logger, self):
            output = super(DropoutTransparent, self).forward(input)
            if logger:
                logger.log("input", input, self)
                logger.log("output", output, self)

        return output
