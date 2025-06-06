# FxPyTorch/fxp/fxp_layernorm.py
import torch
from .symmetric_quant import (
    QType,
    quantize,
    apply_quantize,
    get_high_precision_tensor_quant,
    get_no_overflow_tensor_quant,
)
from typing import Optional, Literal
from ..transparent.activation_logger import (
    ActivationLogger,
    ActivationLoggingScope,
)
from pydantic import Field
from .symmetric_quant import QConfig
from .utils import ValueRange, tensor_to_value_range
from ..transparent.trans_layernorm import LayerNormTransparent

from typing import Union, List
from .calibration import set_calibrated_activation_quant, CalibrationType


class LayerNormQConfig(QConfig):
    """Quantization configuration specific to FxPLayerNorm layers."""

    # **Crucial**: Define the discriminator value using Literal
    layer_type: Literal["layer_norm"] = "layer_norm"

    """Pydantic model for quantization configuration of a FxPLayerNorm layer."""
    input: QType = Field(default_factory=QType)
    weight: QType = Field(default_factory=QType)
    bias: QType = Field(default_factory=QType)
    mean_tensor: QType = Field(default_factory=QType)
    var_tensor: QType = Field(default_factory=QType)
    input_normalized: QType = Field(default_factory=QType)
    activation: QType = Field(default_factory=QType)


class FxPLayerNorm(LayerNormTransparent):
    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
        q_config: LayerNormQConfig = None,
    ) -> None:
        super(FxPLayerNorm, self).__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self._q_config = q_config

    @property
    def q_config(self) -> LayerNormQConfig:
        return self._q_config

    @q_config.setter
    def q_config(self, new_q_config: LayerNormQConfig):
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
            self._q_config.weight = new_q_config.weight
            self._q_config.bias = new_q_config.bias
            self._q_config.mean_tensor = new_q_config.mean_tensor
            self._q_config.var_tensor = new_q_config.var_tensor
            self._q_config.input_normalized = new_q_config.input_normalized
            self._q_config.activation = new_q_config.activation

    def forward(
        self,
        input: torch.Tensor,
        logger: Optional[ActivationLogger] = None,
        apply_ste: bool = True,
        calibrate: bool = False,
        calibration_type: Union[str, CalibrationType] = CalibrationType.NO_OVERFLOW,
    ) -> torch.Tensor:
        if self._q_config is None:
            # Floating point, call LayerNormTransparent
            return super(FxPLayerNorm, self).forward(input, logger)
        with ActivationLoggingScope(logger, self):
            # Copy-paste code from LayerNormTransparent
            dims_to_normalize = tuple(
                range(input.ndim - len(self.normalized_shape), input.ndim)
            )
            # STE
            input_quant = apply_quantize(input, self._q_config.input, apply_ste)
            mean = torch.mean(input_quant, dim=dims_to_normalize, keepdim=True)
            if calibrate:
                set_calibrated_activation_quant(
                    mean, self._q_config.mean_tensor, calibration_type
                )
            mean_quant = apply_quantize(mean, self._q_config.mean_tensor, apply_ste)
            var = torch.var(
                input_quant, dim=dims_to_normalize, unbiased=False, keepdim=True
            )
            if calibrate:
                set_calibrated_activation_quant(
                    var, self._q_config.var_tensor, calibration_type
                )
            var_quant = apply_quantize(var, self._q_config.var_tensor, apply_ste)
            input_normalized = (input_quant - mean_quant) / torch.sqrt(
                var_quant + self.eps
            )
            if calibrate:
                set_calibrated_activation_quant(
                    input_normalized, self._q_config.input_normalized, calibration_type
                )
            input_normalized_quant = apply_quantize(
                input_normalized, self._q_config.input_normalized, apply_ste
            )
            # Apply elementwise affine transformation if enabled
            if self.elementwise_affine:
                # Reshape weight and bias for broadcasting if necessary
                # (although typically their shape matches the normalized dimensions already)
                w = self.weight
                w_quant = apply_quantize(w, self._q_config.weight, apply_ste)
                output_pre_quant = input_normalized_quant * w_quant
                if logger:
                    logger.log("weight", w, self)
                    logger.log("weight_quant", w_quant, self)
                if self.bias is not None:
                    b = self.bias
                    b_quant = apply_quantize(b, self._q_config.bias, apply_ste)
                    output_pre_quant = output_pre_quant + b_quant
                    if logger:
                        logger.log("bias", b, self)
                        logger.log("bias_quant", b_quant, self)
            else:
                output_pre_quant = input_normalized_quant

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
                logger.log("mean", mean, self)
                logger.log("mean_quant", mean_quant, self)
                logger.log("variance", var, self)
                logger.log("variance_quant", var_quant, self)
                logger.log("input_normalized", input_normalized, self)
                logger.log("input_normalized_quant", input_normalized_quant, self)
                logger.log("output_pre_quant", output_pre_quant, self)
                logger.log("output", output, self)

        return output

    def quantize_weights_bias(self) -> None:
        self.weight.data.copy_(quantize(self.weight, self._q_config.weight))
        if self.bias is not None:
            self.bias.data.copy_(quantize(self.bias, self._q_config.bias))

    def get_q_config(self) -> LayerNormQConfig:
        return self._q_config

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
