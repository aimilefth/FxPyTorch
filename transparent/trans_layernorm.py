# FxPyTorch/transparent/trans_layernorm.py
import torch
from torch import nn
from typing import Optional, Union, List
import numbers
from torch.nn.parameter import Parameter
from .activation_logger import (
    ActivationLogger,
    ActivationLoggingScope,
)


class LayerNormTransparent(nn.Module):
    r"""Applies Layer Normalization over a mini-batch of inputs
    without using F.layer_norm.

    Calculates the mean and variance over the last D dimensions, where D
    is the dimension of :attr:`normalized_shape`.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size [* x normalized_shape[0] x ... x normalized_shape[-1]]
            If a single integer, it is treated as a singleton list, normalizing over the last dimension.
        eps (float): a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine (bool): If True, this module has learnable per-element
            affine parameters initialized to ones (for weights) and zeros (for biases). Default: True.
        bias (bool): If set to ``False``, the layer will not learn an additive bias (only relevant if
            :attr:`elementwise_affine` is ``True``). Default: ``True``.
        device: The desired device of the parameters. Default: None (uses default device)
        dtype: The desired floating point type of the parameters. Default: None (uses default dtype)

    Attributes:
        weight: the learnable weights (gamma) of shape :attr:`normalized_shape`.
        bias:   the learnable bias (beta) of shape :attr:`normalized_shape`.
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # Handle single integer input for normalized_shape
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                # Important: Use register_parameter to indicate bias is intentionally None
                self.register_parameter("bias", None)
        else:
            # Important: Use register_parameter for consistency when affine is false
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize or reset the parameters."""
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight)
            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)

    def forward(
        self,
        input: torch.Tensor,
        logger: Optional[ActivationLogger] = None,
        **kwargs,  # <- accept and ignore anything extra
    ) -> torch.Tensor:
        """Apply the layer normalization."""
        # input shape: (N, *, normalized_shape)
        # normalized_shape: (D1, D2, ..., Dk)
        # We need to compute mean and variance over the last k dimensions
        with ActivationLoggingScope(logger, self):
            if logger:
                logger.log("input", input, self)
            # Determine the dimensions to normalize over
            # These are the last len(self.normalized_shape) dimensions
            dims_to_normalize = tuple(
                range(input.ndim - len(self.normalized_shape), input.ndim)
            )
            # Calculate mean and variance over the specified dimensions
            # Keep dimensions for broadcasting
            mean = torch.mean(input, dim=dims_to_normalize, keepdim=True)
            # Use biased variance like torch.nn.LayerNorm
            var = torch.var(input, dim=dims_to_normalize, unbiased=False, keepdim=True)
            # Normalize the input
            input_normalized = (input - mean) / torch.sqrt(var + self.eps)
            # Apply elementwise affine transformation if enabled
            if self.elementwise_affine:
                # Reshape weight and bias for broadcasting if necessary
                # (although typically their shape matches the normalized dimensions already)
                output = input_normalized * self.weight
                if self.bias is not None:
                    output = output + self.bias
            else:
                output = input_normalized
            if logger:
                logger.log("mean", mean, self)
                logger.log("variance", var, self)
                logger.log("input_normalized", input_normalized, self)
                logger.log("output", output, self)

        return output

    def extra_repr(self) -> str:
        """Set the extra representation of the module."""
        return (
            f"{self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )
