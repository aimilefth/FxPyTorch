# FxPyTorch/fxp/fxp_dropout.py

import torch
from .symmetric_quant import (
    QType,
    apply_quantize,
)
from typing import Optional, Literal
from ..transparent.activation_logger import (
    ActivationLogger,
    ActivationLoggingScope,
)
from pydantic import Field
from .symmetric_quant import QConfig
from ..transparent.trans_dropout import DropoutTransparent


class DropoutQConfig(QConfig):
    """Quantization configuration specific to FxPDropout layers."""

    # **Crucial**: Define the discriminator value using Literal
    layer_type: Literal["dropout"] = "dropout"

    """Pydantic model for quantization configuration of a FxPDropout layer."""
    input: QType = Field(default_factory=QType)
    activation: QType = Field(default_factory=QType)


class FxPDropout(DropoutTransparent):
    def __init__(
        self,
        p: float = 0.5,
        inplace: bool = False,
        q_config: DropoutQConfig = None,
    ) -> None:
        super(FxPDropout, self).__init__(p, inplace)
        self._q_config = q_config

    @property
    def q_config(self) -> DropoutQConfig:
        return self._q_config

    @q_config.setter
    def q_config(self, new_q_config: DropoutQConfig):
        """
        Setter for this layer’s quantization config.

        Logic:
          1. If there is no existing config (first assignment), we simply
             bind the provided object.  That happens during construction.
          2. If a config already exists, we update its fields _in place_
             rather than rebind.  This preserves any shared references
             (e.g., logging utilities or benchmark code that captured
             the old object).
        """
        if self._q_config is None:
            self._q_config = new_q_config
        else:
            # "Deep copy" of the configs, required for upstream usage
            self._q_config.input = new_q_config.input
            self._q_config.activation = new_q_config.activation

    def forward(
        self,
        input: torch.Tensor,
        logger: Optional[ActivationLogger] = None,
        apply_ste: bool = True,
    ) -> torch.Tensor:
        if self._q_config is None:
            # Floating point, call DropoutTransparent
            return super(FxPDropout, self).forward(input, logger)
        with ActivationLoggingScope(logger, self):
            # STE
            input_quant = apply_quantize(input, self._q_config.input, apply_ste)
            output_pre_quant = super(FxPDropout, self).forward(input_quant, logger=None)
            output = apply_quantize(
                output_pre_quant, self._q_config.activation, apply_ste
            )
            if logger:
                logger.log("input", input, self)
                logger.log("input_quant", input_quant, self)
                logger.log("output_pre_quant", output_pre_quant, self)
                logger.log("output", output, self)

        return output
