# FxPyTorch/fxp/fxp_tanh.py
import torch
from typing import Optional, Literal, Union
from pydantic import Field
from ..transparent.activation_logger import (
    ActivationLogger,
    ActivationLoggingScope,
)
from .symmetric_quant import QType, QConfig, apply_quantize
from ..transparent.trans_tanh import TanhTransparent
from .calibration import set_calibrated_activation_quant, CalibrationType


class TanhQConfig(QConfig):
    layer_type: Literal["tanh"] = "tanh"
    input: QType = Field(default_factory=QType)
    activation: QType = Field(default_factory=QType)


class FxPTanh(TanhTransparent):
    def __init__(self, q_config: Optional[TanhQConfig] = None) -> None:
        super().__init__()
        self._q_config = q_config

    @property
    def q_config(self) -> Optional[TanhQConfig]:
        return self._q_config

    @q_config.setter
    def q_config(self, new_q_config: TanhQConfig):
        if self._q_config is None:
            self._q_config = new_q_config
        else:
            self._q_config.input = new_q_config.input
            self._q_config.activation = new_q_config.activation

    def forward(
        self,
        input: torch.Tensor,
        logger: Optional[ActivationLogger] = None,
        apply_ste: bool = True,
        calibrate: bool = False,
        calibration_type: Union[str, CalibrationType] = CalibrationType.NO_OVERFLOW,
        **kwargs,
    ) -> torch.Tensor:
        # Float path identical to transparent
        if self._q_config is None:
            return super().forward(input, logger=logger)

        with ActivationLoggingScope(logger, self):
            if calibrate:
                set_calibrated_activation_quant(
                    input, self._q_config.input, calibration_type
                )
            input_quant = apply_quantize(input, self._q_config.input, apply_ste)

            output_pre_quant = super().forward(input_quant, logger=None)

            if calibrate:
                set_calibrated_activation_quant(
                    output_pre_quant, self._q_config.activation, calibration_type
                )
            output = apply_quantize(
                output_pre_quant, self._q_config.activation, apply_ste
            )

            if logger:
                logger.log("input", input, self)
                logger.log("input_quant", input_quant, self)
                logger.log("output_pre_quant", output_pre_quant, self)
                logger.log("output", output, self)
        return output
