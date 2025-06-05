# FxPyTorch/fxp/symmetric_quant.py
import torch
import math
from enum import Enum, auto
from typing import Dict, Any, Optional

# Pydantic imports
from pydantic import BaseModel, Field, model_validator, ConfigDict
from .utils import (
    ValueRange,
    tensor_to_value_range,
    get_tensor_mse,
    VALID_CALIBRATION_TYPES,
)
# Symmetrics Linear Quantization with Scaling Factor being power of 2 (float to fixed conversion)

# 64 BITS MAX TOTAL_BITS


class QMethod(Enum):
    """Enumeration for supported quantization methods/algorithms."""

    TRUNC_SATURATE = auto()
    ROUND_SATURATE = auto()

    @classmethod
    def from_string(cls, name: str) -> "QMethod":
        try:
            return cls[name.upper()]
        except KeyError:
            valid_types = [member.name for member in cls]
            raise ValueError(
                f"Invalid QMethod string '{name}'. Must be one of: {valid_types}"
            )

    def __str__(self):
        return self.name


class QType(BaseModel):
    """
    Symmetrics Linear Quantization with Scaling Factor being power of 2 (float to fixed conversion)
    Using Pydantic for validation
    """

    # Use Field for descriptions and validation constraints
    # Allow None for bit fields
    total_bits: Optional[int] = Field(
        None,
        ge=2,
        description="Total bits for quantized value. None for floating-point.",
    )
    fractional_bits: Optional[int] = Field(
        None, description="Number of fractional bits. None for floating-point."
    )
    # Allow None for method when bits are None
    q_method: Optional[QMethod] = Field(
        QMethod.ROUND_SATURATE,
        description="Quantization method (e.g., TRUNC_SATURATE). None for floating-point.",
    )

    # Pydantic V2 configuration (optional but good practice)
    model_config = ConfigDict(
        validate_assignment=True,  # Re-validate when fields are changed after creation
        extra="forbid",  # Disallow extra fields not defined in the model
        frozen=False,  # Allow modification after creation
    )

    # --- Computed Field for Integer Bits ---
    @property
    def integer_bits(self) -> Optional[int]:
        """
        Calculated number of integer bits (including the sign bit).
        """
        if self.total_bits is not None and self.fractional_bits is not None:
            # Ensure the value is valid before returning
            return self.total_bits - self.fractional_bits
        else:
            return None  # Return None if calculation is not possible

    # Use model_validator for cross-field checks and specific field validation
    @model_validator(mode="after")
    def check_config_consistency(self) -> "QType":
        """Validate the overall consistency of the quantization configuration."""
        tb = self.total_bits
        fb = self.fractional_bits
        qm = self.q_method

        # Case 1: Fully defined quantization
        if tb is not None and fb is not None:
            # Validate cross-field consistency
            if not isinstance(tb, int) or tb <= 1:
                raise ValueError(
                    f"If total_bits is set, it must be an integer > 1, got {tb}"
                )
            # If bits are set, a method must also be set
            if qm is None:
                raise ValueError(
                    "q_method must be set when total_bits and fractional_bits are defined."
                )
            if not isinstance(qm, QMethod):
                # This might be caught by Pydantic's initial type check, but good defence
                raise ValueError(
                    f"q_method must be a QMethod enum member, got {type(qm)}"
                )

        # Case 2: Allow partial definition (one bitwidth None) - used by helper functions
        else:
            # Validate individual fields if they are set
            if tb is not None:
                if not isinstance(tb, int) or tb <= 1:
                    raise ValueError(
                        f"If total_bits is set, it must be an integer > 1, got {tb}"
                    )
            # If bits are set, a method must also be set
            if qm is None:
                raise ValueError(
                    "q_method must be set when total_bits or fractional_bits are defined."
                )
            if not isinstance(qm, QMethod):
                # This might be caught by Pydantic's initial type check, but good defence
                raise ValueError(
                    f"q_method must be a QMethod enum member, got {type(qm)}"
                )
        return self  # Must return the validated model instance


class QConfig(BaseModel):
    """
    Base Class for layer-specific quantization configurations.
    Subclasses should define 'layer_type' with a Literal value for discrimination.
    """

    # The discriminator field - type is str here, Literal is in subclasses
    layer_type: str = Field(
        ..., description="Discriminator field identifying the layer type."
    )
    name: Optional[str] = Field(
        None, description="Optional descriptive name for this config."
    )

    model_config = ConfigDict(
        extra="forbid",  # Disallow fields not defined in the model
        validate_assignment=True,  # Re-validate fields if they are assigned a new value after creation
    )

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation with JSON-compatible types."""
        return self.model_dump(mode="json")

    def save_json(self, file_path: str):
        """Saves the configuration to a JSON file."""
        with open(file_path, "w") as f:
            f.write(self.model_dump_json(indent=4))


def quantize(x: torch.Tensor, q_type: QType = None):
    if q_type is None or (q_type.total_bits is None and q_type.fractional_bits is None):
        # Not configured for quantization, return input as is (float)
        return x
    elif (
        (q_type.total_bits is None and q_type.fractional_bits is not None)
        or (q_type.total_bits is not None and q_type.fractional_bits is None)
        or (q_type.q_method is None)
    ):
        raise ValueError(f"Incorrect q_type configuration for quantize, got {q_type}")
    # Step 1: Calculate the scaling factor (s = 2^(-f)), which shifts the decimal point
    # More fractional bits means a smaller scaling factor (finer resolution).
    scaling_factor = pow(2, -q_type.fractional_bits)

    # Step 2: Calculate the inverse of the scaling factor (s^-1 = 2^f).
    # This scales the values up to integers by multiplying by this factor.
    scaling_factor_div = pow(2, q_type.fractional_bits)

    # Step 3: Calculate the quantization range.
    # q_min is the smallest value that can be represented with the given number of bits.
    # q_max is the largest value that can be represented.
    q_min = -pow(2, q_type.total_bits - 1)
    q_max = pow(2, q_type.total_bits - 1) - 1
    if q_type.q_method == QMethod.TRUNC_SATURATE:
        # Step 4: Scale the input tensor by the inverse of the scaling factor to convert it into integers.
        # Using truncation to remove the fractional part and handle quantization.
        x_integer = torch.trunc(x * scaling_factor_div)

        # Step 5: Saturate the values to the range [q_min, q_max].
        # This ensures that the values do not exceed the representable range.
        x_integer_saturate = torch.clamp(x_integer, min=q_min, max=q_max)
    elif q_type.q_method == QMethod.ROUND_SATURATE:
        # Step 4: Scale the input tensor to integer space using rounding instead of truncation.
        # Rounding tends to give better accuracy compared to truncation.
        x_integer = torch.round(x * scaling_factor_div)

        # Step 5: Saturate the rounded values to the range [q_min, q_max].
        x_integer_saturate = torch.clamp(x_integer, min=q_min, max=q_max)
    else:
        raise ValueError(
            f"q_type.q_method must be either QMethod.TRUNC_SATURATE or QMethod.ROUND_SATURATE, got {q_type.q_method}"
        )
    # Step 6: Convert the saturated integers back to the original scale (floating-point).
    # Multiply by the scaling factor to reverse the scaling applied in Step 4.
    x_quant = x_integer_saturate * scaling_factor
    return x_quant


def apply_quantize(
    x: torch.Tensor, q_type: QType, apply_ste: bool = True
) -> torch.Tensor:
    if apply_ste:
        return x + (quantize(x, q_type) - x).detach()
    else:
        return quantize(x, q_type)


def get_integer_bits_for_value_range(value_range: ValueRange, q_type: QType) -> int:
    # Unpack the minimum and maximum values from the value range
    min_val, max_val = value_range.min_val, value_range.max_val
    assert q_type.total_bits is not None or q_type.fractional_bits is not None, (
        "Need at least one of total_bits or fractional_bits, got both None"
    )

    # Calculate the number of integer bits required for the minimum value
    if min_val >= 0:
        # If the minimum value is non-negative (all weights are positive),
        # set integer_bits_for_min to negative infinity as it's not needed
        integer_bits_for_min = float("-inf")
    else:
        # Calculate the integer bits needed to represent the negative minimum value
        # math.log2(-min_val) computes the log base 2 of the absolute minimum value
        # Adding 1 accounts for the sign bit
        integer_bits_for_min = math.ceil(math.log2(-min_val) + 1)

    # Calculate the number of integer bits required for the maximum value
    if max_val <= 0:
        # If the maximum value is negative (all weights are negative),
        # set integer_bits_for_max to negative infinity as it's not needed
        integer_bits_for_max = float("-inf")
    else:
        if q_type.fractional_bits is None:
            # If fractional_bits is not specified, calculate integer bits with total_bits
            # Adding 1 accounts for the sign bit
            assert q_type.total_bits >= 2, (
                f" total_bits should be at least 2, got {q_type.total_bits}"
            )
            integer_bits_for_max = math.ceil(
                math.log2(max_val / (1 - pow(2, 1 - q_type.total_bits))) + 1
            )
        else:
            # If fractional_bits is specified, adjust the maximum value by subtracting 2^(-fractional_bits)
            # This accounts for the smallest fractional increment
            if max_val < pow(2, -q_type.fractional_bits):
                # If max_val is smaller than the smallest fractional increment, then we have unique case
                # integer_bits would be 1 - fractional, then total_bits = 1, which is error in our case (issue with upstream QType values)
                # This could come up in all zeros case. Give 1 more integer bit in this case
                integer_bits_for_max = integer_bits_for_max = math.ceil(
                    math.log2(2 * pow(2, -q_type.fractional_bits)) + 1
                )
            else:
                integer_bits_for_max = math.ceil(
                    math.log2(max_val + pow(2, -q_type.fractional_bits)) + 1
                )
    if min_val == 0.0 and max_val == 0.0:
        # Special case where all values are 0, return high precision values:
        if q_type.fractional_bits is None:
            # Make all total_bits be used as fractional == 24 (high-precision)
            return q_type.total_bits - 24
        else:
            # Make integer, so all total_bits = 2 (t >=2 -> i = -f + 2)
            return 2 - q_type.fractional_bits
    # The required number of integer bits is the maximum of the bits needed for min and max values
    return max(integer_bits_for_min, integer_bits_for_max)


def get_high_precision_tensor_quant(x: torch.Tensor) -> QType:
    # Returns a QType with the quant for this tensor
    FP32_PRECISION_FRACTIONAL_BITS = 24
    q_type = QType()
    "Consider that 24 fractional bits give approximate precision to FP32, find integer bits needed for dynamic range"
    q_type.fractional_bits = FP32_PRECISION_FRACTIONAL_BITS
    integer_bits = get_integer_bits_for_value_range(tensor_to_value_range(x), q_type)
    q_type.total_bits = integer_bits + q_type.fractional_bits
    return q_type


def get_no_overflow_tensor_quant(x: torch.Tensor, q_type: QType) -> QType:
    return_q_type = QType()
    if q_type.fractional_bits is None and q_type.total_bits is None:
        raise ValueError(
            "Need either total_bits or fractional_bits to be specified, got both None"
        )
    elif q_type.total_bits is not None:
        # Have set total_bits, calculate needed integer bits for dynamic range, and give the rest to the functional bits
        return_q_type.total_bits = q_type.total_bits
        integer_bits = get_integer_bits_for_value_range(
            tensor_to_value_range(x), q_type
        )
        return_q_type.fractional_bits = return_q_type.total_bits - integer_bits
    else:
        # q_type.fractional_bits is not None
        # Have set fractional_bits (precision), calculate needed integer bits for dynamic range, and get total bits
        return_q_type.fractional_bits = q_type.fractional_bits
        integer_bits = get_integer_bits_for_value_range(
            tensor_to_value_range(x), q_type
        )
        return_q_type.total_bits = integer_bits + return_q_type.fractional_bits
    return return_q_type


def get_min_mse_tensor_quant(
    x: torch.Tensor, q_type: QType, depth: int = 10, verbose: bool = False
) -> QType:
    if q_type.total_bits is None:
        raise ValueError("Need either total_bits to be specified, got None")
    # Get the 'no-overflow' quantization as a starting point for fractional bits.
    # This assumes self._q_config.weight.total_bits and q_method are already set.
    no_overflow_q_type = get_no_overflow_tensor_quant(x, q_type)
    mse_list = []
    qtypes_list = []
    # Iterate 'depth' times, increasing fractional bits from the no-overflow starting point.
    for i in range(depth + 1):  # i=0 is no-overflow, i=1..depth are increments
        current_f_bits = no_overflow_q_type.fractional_bits + i
        # Create a candidate QType with the current fractional bits.
        candidate_q_type = QType(
            total_bits=no_overflow_q_type.total_bits,
            fractional_bits=current_f_bits,
            q_method=no_overflow_q_type.q_method,
        )
        # Quantize the original weights using the candidate QType.
        quantized_x = quantize(x, candidate_q_type)
        # Calculate MSE between original and quantized weights.
        current_mse = get_tensor_mse(x, quantized_x)
        qtypes_list.append(candidate_q_type)
        mse_list.append(current_mse)

    # Find the index of the QType that resulted in the minimum MSE.
    index_of_min = mse_list.index(min(mse_list))
    if verbose:
        print(mse_list)
    return qtypes_list[index_of_min]


def set_calibrated_activation_quant(
    activation: torch.Tensor,
    q_type: QType,
    calibration_type: str = "no_overflow",
) -> QType:
    if calibration_type not in VALID_CALIBRATION_TYPES:
        raise ValueError(f"calibration type is invalid, got: {calibration_type}")
    if calibration_type == "no_overflow":
        returned_q_type = get_no_overflow_tensor_quant(activation, q_type)
    elif calibration_type == "min_mse":
        returned_q_type = get_min_mse_tensor_quant(activation, q_type)
    else:
        raise ValueError(
            f"calibration type, got: {calibration_type}, passed from VALID_CALIBRATION_TYPES, shouldn't be here"
        )
    q_type.total_bits = returned_q_type.total_bits
    q_type.fractional_bits = returned_q_type.fractional_bits
