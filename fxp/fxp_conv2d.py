# FxPyTorch/fxp/fxp_conv2d.py
import torch
import torch.nn.functional as F
from .symmetric_quant import (
    QType,
    quantize,
    apply_quantize,
    get_high_precision_tensor_quant,
    get_no_overflow_tensor_quant,
)
from typing import Optional, Literal, Union
from ..transparent.activation_logger import (
    ActivationLogger,
    ActivationLoggingScope,
)
from pydantic import Field
from .symmetric_quant import QConfig
from .utils import ValueRange, tensor_to_value_range
from ..transparent.trans_conv2d import Conv2DTransparent
from torch.nn.common_types import _size_2_t
from .calibration import set_calibrated_activation_quant, CalibrationType


class Conv2DQConfig(QConfig):
    """Quantization configuration specific to FxPConv2D layers."""

    # **Crucial**: Define the discriminator value using Literal
    layer_type: Literal["conv2d"] = "conv2d"

    input: QType = Field(default_factory=QType)
    weight: QType = Field(default_factory=QType)
    bias: QType = Field(default_factory=QType)
    activation: QType = Field(default_factory=QType)


class FxPConv2D(Conv2DTransparent):
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
        device: str = None,
        dtype: torch.dtype = None,
        q_config: Conv2DQConfig = None,
    ):
        super(FxPConv2D, self).__init__(
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
        self._q_config = q_config

        #CHANGE: calibrate and calibration_type variables
        self.calibrate: bool = False
        self.calibration_type: Union[str, CalibrationType] = CalibrationType.NO_OVERFLOW

    """
    The @property decorator is used to define "getter" methods for class attributes, but it 
    allows you to access them like regular attributes (without calling them with parentheses ()).
    """

    @property
    def q_config(self) -> Conv2DQConfig:
        return self._q_config

    # If we need to set the q_config, the setter will be executed
    @q_config.setter
    def q_config(self, new_q_config: Conv2DQConfig):
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
            self._q_config.input = new_q_config.input
            self._q_config.weight = new_q_config.weight
            self._q_config.bias = new_q_config.bias
            self._q_config.activation = new_q_config.activation

    def forward(
        self,
        input: torch.Tensor,
        logger: Optional[ActivationLogger] = None,
        apply_ste: bool = True,
    ) -> torch.Tensor:
        if self._q_config is None:
            # Floating point
            return super(FxPConv2D, self).forward(input, logger)
        with ActivationLoggingScope(logger, self):
            w = self.weight
            b = self.bias
            # STE using detach
            if self.calibrate:
                set_calibrated_activation_quant(
                    input, self._q_config.input, self.calibration_type
                )
            input_quant = apply_quantize(input, self._q_config.input, apply_ste)
            w_quant = apply_quantize(w, self._q_config.weight, apply_ste)
            b_quant = None
            if self.bias is not None:
                b_quant = apply_quantize(b, self._q_config.bias, apply_ste)
            activation = F.conv2d(
                input_quant,
                w_quant,
                b_quant,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            if self.calibrate:
                set_calibrated_activation_quant(
                    activation, self._q_config.activation, self.calibration_type
                )
            activation_quant = apply_quantize(
                activation, self._q_config.activation, apply_ste
            )
            output = activation_quant
            if logger:
                logger.log("input", input, self)
                logger.log("input_quant", input_quant, self)
                logger.log("weight", w, self)
                logger.log("bias", b, self)
                logger.log("weight_quant", w_quant, self)
                logger.log("bias_quant", b_quant, self)
                logger.log("activation", activation, self)
                logger.log("output", output, self)
        return output
    #CHANGE: set function for calibration mode
    def turn_on_calibration(self, calibration_type):
        self.calibrate = True
        self.calibration_type = calibration_type

    def turn_off_calibration(self):
        self.calibrate = False
        self.calibration_type = None

    def quantize_weights_bias(self) -> None:
        self.weight.data.copy_(quantize(self.weight, self._q_config.weight))
        if self.bias is not None:
            self.bias.data.copy_(quantize(self.bias, self._q_config.bias))

    def _get_weight_range(self) -> ValueRange:
        return tensor_to_value_range(self.weight.data)

    def _get_bias_range(self) -> ValueRange:
        return tensor_to_value_range(self.bias.data)

    def set_high_precision_w_quant(self) -> None:
        "Consider that 24 fractional bits give approximate precision to FP32, find integer bits needed for dynamic range"
        q_type = get_high_precision_tensor_quant(self.weight.data)
        self._q_config.weight.total_bits = q_type.total_bits
        self._q_config.weight.fractional_bits = q_type.fractional_bits

    def set_high_precision_b_quant(self) -> None:
        "Consider that 24 fractional bits give approximate precision to FP32, find integer bits needed for dynamic range"
        if self.bias is not None:
            q_type = get_high_precision_tensor_quant(self.bias.data)
            self._q_config.bias.total_bits = q_type.total_bits
            self._q_config.bias.fractional_bits = q_type.fractional_bits

    def set_high_precision_quant(self, same_wb: bool = False) -> None:
        "Consider that 24 fractional bits give approximate precision to FP32, find integer bits needed for dynamic range"
        self.set_high_precision_w_quant()
        self.set_high_precision_b_quant()
        if same_wb and self.bias is not None:
            # To get the same quantization in both weight and bias, choose the max total_bits required for both (24 fractional standard)
            max_total_bits = max(
                self._q_config.weight.total_bits, self._q_config.bias.total_bits
            )
            self._q_config.weight.total_bits = max_total_bits
            self._q_config.bias.total_bits = max_total_bits

    def set_no_overflow_w_quant(self) -> None:
        q_type = get_no_overflow_tensor_quant(self.weight.data, self._q_config.weight)
        self.q_config.weight.total_bits = q_type.total_bits
        self.q_config.weight.fractional_bits = q_type.fractional_bits

    def set_no_overflow_b_quant(self) -> None:
        if self.bias is not None:
            q_type = get_no_overflow_tensor_quant(self.bias.data, self._q_config.bias)
            self.q_config.bias.total_bits = q_type.total_bits
            self.q_config.bias.fractional_bits = q_type.fractional_bits

    def set_no_overflow_quant(self, same_wb: bool = False) -> None:
        self.set_no_overflow_w_quant()
        self.set_no_overflow_b_quant()
        if same_wb and self.bias is not None:
            # To avoid overflow,
            # 1) check the max total_integer_bits
            max_integer_bits = max(
                self._q_config.weight.integer_bits, self._q_config.bias.integer_bits
            )
            # 2) Make total_bits = max_total_bits
            max_total_bits = max(
                self._q_config.weight.total_bits, self._q_config.bias.total_bits
            )
            self._q_config.weight.total_bits = self._q_config.bias.total_bits = (
                max_total_bits
            )
            # 3) Change the integer_bits to max_integer_bit to avoid overflow (-> minimize fractional)
            self._q_config.weight.fractional_bits = (
                self._q_config.bias.fractional_bits
            ) = max_total_bits - max_integer_bits
