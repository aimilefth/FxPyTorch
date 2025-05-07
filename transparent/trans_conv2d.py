# my_work/trans_conv2D.py
import torch
from torch import nn
from typing import Optional, Union
from ..transparent.activation_logger import (
    ActivationLogger,
    ActivationLoggingScope,
)
from torch.nn.common_types import _size_2_t


class Conv2DTransparent(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
    ) -> None:
        super(Conv2DTransparent, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        input: torch.Tensor,
        logger: Optional[ActivationLogger] = None,
        **kwargs,  # <- accept and ignore anything extra
    ) -> torch.Tensor:
        with ActivationLoggingScope(logger, self):
            output = super(Conv2DTransparent, self).forward(input)
            if logger:
                logger.log("input", input, self)
                logger.log("output", output, self)

        return output
