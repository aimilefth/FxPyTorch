# FxPyTorch/fxp/calibration.py
from enum import Enum
import torch
from .symmetric_quant import (
    get_no_overflow_tensor_quant,
    get_min_mse_tensor_quant,
    QType,
)
from typing import Union


class CalibrationType(str, Enum):
    """
    Allowed calibration strategies.

    Inherit from ``str`` so that:
      • Pydantic accepts raw strings and converts them automatically.
      • The value is JSON-serialisable out of the box.
    """

    NO_OVERFLOW = "no_overflow"
    MIN_MSE = "min_mse"

    def __str__(self) -> str:  # Optional, nice to have
        return self.value


def set_calibrated_activation_quant(
    activation: torch.Tensor,
    q_type: QType,
    calibration_type: Union[str, CalibrationType] = CalibrationType.NO_OVERFLOW,
) -> QType:
    # Convert occasional string to CalibrationType
    if isinstance(calibration_type, str):
        try:
            calibration_type = CalibrationType(calibration_type)
        except ValueError as e:
            raise ValueError(f"Invalid calibration_type: {calibration_type}") from e

    if calibration_type is CalibrationType.NO_OVERFLOW:
        returned_q_type = get_no_overflow_tensor_quant(activation, q_type)
    elif calibration_type is CalibrationType.MIN_MSE:
        returned_q_type = get_min_mse_tensor_quant(activation, q_type)
    else:
        raise ValueError(
            f"calibration type, got: {calibration_type}, passed from VALID_CALIBRATION_TYPES, shouldn't be here"
        )
    q_type.total_bits = returned_q_type.total_bits
    q_type.fractional_bits = returned_q_type.fractional_bits
