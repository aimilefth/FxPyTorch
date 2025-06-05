# FxPyTorch/fxp/utils.py
import torch

# Pydantic imports
from pydantic import BaseModel, model_validator, ConfigDict


class ValueRange(BaseModel):
    # Correctly define fields as floats. They are required, so no default needed.
    min_val: float
    max_val: float

    # --- Pydantic Model Configuration ---
    model_config = ConfigDict(
        extra="forbid",  # Disallow fields not defined in the model
        validate_assignment=True,  # Re-validate fields if they are assigned (though frozen makes this less relevant after creation)
        frozen=True,  # Make instances immutable after creation
    )

    # Use model_validator for cross-field checks
    @model_validator(mode="after")
    def check_min_max_consistency(self) -> "ValueRange":  # Use Self or "ValueRange"
        """Validates that min_val is less than or equal to max_val."""
        # Validate cross-field consistency
        if self.min_val > self.max_val:
            raise ValueError(
                f"min_val ({self.min_val}) must be less than or equal to max_val ({self.max_val})"
            )
        # Must return self for validators
        return self


def tensor_to_value_range(tensor: torch.Tensor) -> ValueRange:
    """
    Calculates the minimum and maximum values in a tensor and returns them
    as an immutable ValueRange object.

    Args:
        tensor: The input torch.Tensor.

    Returns:
        A ValueRange instance containing the min and max values.

    Raises:
        RuntimeError: If the tensor is empty.
    """
    if tensor.numel() == 0:
        raise RuntimeError("Cannot calculate value range for an empty tensor.")
        # Alternatively, decide on a default behavior for empty tensors,
        # e.g., return ValueRange(min_val=0.0, max_val=0.0) or raise specific error.

    # Get min/max values as Python floats
    min_val = torch.min(tensor).item()
    max_val = torch.max(tensor).item()

    # Create and return the ValueRange instance
    # Pydantic will automatically run the validator upon creation
    return ValueRange(min_val=min_val, max_val=max_val)


def get_tensor_mse(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """
    Calculates the Mean Squared Error (MSE) between two tensors.

    The tensors must have the same shape.
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError(
            f"Input tensors must have the same shape. "
            f"Got {tensor1.shape} and {tensor2.shape}."
        )
    return torch.mean((tensor1 - tensor2) ** 2).item()