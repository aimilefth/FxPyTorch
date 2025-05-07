# transparent/trans_linear.py
import torch
from torch import nn
from typing import Optional
from .activation_logger import (
    ActivationLogger,
    ActivationLoggingScope,
)


class LinearTransparent(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
    ) -> None:
        super(LinearTransparent, self).__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )

    def forward(
        self,
        input: torch.Tensor,
        logger: Optional[ActivationLogger] = None,
        **kwargs,  # <- accept and ignore anything extra
    ) -> torch.Tensor:
        with ActivationLoggingScope(logger, self):
            output = super(LinearTransparent, self).forward(input)
            if logger:
                logger.log("input", input, self)
                logger.log("output", output, self)

        return output
