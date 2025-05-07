# FxPyTorch: Fixed-Point Quantization for PyTorch Layers

**FxPyTorch** is a Python library that extends PyTorch's `nn.Module` system to support **symmetric, linear fixed-point quantization**. It provides tools for simulating fixed-point arithmetic where the **scaling factor is constrained to be a power of 2**, which is often efficient for hardware implementations. The library helps analyze quantization effects and prepare models for deployment on hardware with fixed-point capabilities.

## Features

*   **Fixed-Point Layer Implementations:**
    *   `FxPLinear`: Fixed-point Linear layer.
    *   `FxPLayerNorm`: Fixed-point Layer Normalization.
    *   `FxPMultiheadAttention`: Fixed-point Multi-Head Attention.
    *   `FxPTransformerEncoderLayer`: Fixed-point Transformer Encoder Layer.
    *   `FxPSoftmax`: Fixed-point Softmax.
    *   `FxPDropout`: Fixed-point Dropout (quantizes input/output, dropout itself is standard).
*   **Flexible Quantization Configuration (Symmetric, Power-of-2 Scaling):**
    *   Implements symmetric linear quantization around zero.
    *   Uses power-of-2 scaling factors (determined by `fractional_bits`) for efficient hardware mapping (e.g., bit shifts instead of multiplications).
    *   Define `total_bits` and `fractional_bits` for weights, biases, and activations.
    *   Choose rounding methods (e.g., `ROUND_SATURATE`, `TRUNC_SATURATE`).
    *   Pydantic-based configuration models (`QType`, `LinearQConfig`, etc.) for validation and clarity.
*   **Helper Utilities:**
    *   `set_high_precision_quant()`: Configure layers for maximum precision (e.g., 24 fractional bits) given their dynamic range, adhering to the symmetric, power-of-2 scheme.
    *   `set_no_overflow_quant()`: Configure layers to use a specified total number of bits for parameters, automatically calculating fractional bits to prevent overflow based on weight/bias dynamic range, adhering to the symmetric, power-of-2 scheme.
*   **Transparent Base Layers:**
    *   Includes "transparent" versions of standard PyTorch layers (`LinearTransparent`, `LayerNormTransparent`, etc.) that act as drop-in replacements for `nn.Module` equivalents but include hooks for activation logging. These serve as the base for the `FxP` layers.
*   **Activation Logging:**
    *   `ActivationLogger` utility to inspect intermediate tensor values and their quantized counterparts throughout the model.


## Installation

### Prerequisites

*   Python (>=3.8 recommended)
*   PyTorch (>=2.2.0 recommended, see `pyproject.toml` for specific version)
*   Pydantic (>=2.0, see `pyproject.toml`)

### From Git (Recommended for development or as a submodule)

You can include FxPyTorch in your project as a Git submodule:
```bash
git submodule add https://github.com/yourusername/FxPyTorch.git
```

## Quick Start

```python
import torch
from FxPyTorch.fxp.fxp_linear import FxPLinear, LinearQConfig
from FxPyTorch.fxp.symmetrics_quant import QType, QMethod


# Define a quantization configuration for a linear layer
# Example: 8-bit weights, 8-bit bias, 16-bit input/activation with 8 fractional bits
linear_q_config = LinearQConfig(
    input=QType(total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE),
    weight=QType(total_bits=8, q_method=QMethod.ROUND_SATURATE), # Fractional bits determined by set_no_overflow_quant
    bias=QType(total_bits=8, q_method=QMethod.ROUND_SATURATE),   # Fractional bits determined by set_no_overflow_quant
    activation=QType(total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE)
)

# Create a fixed-point linear layer
fxp_linear_layer = FxPLinear(in_features=10, out_features=5, bias=True, q_config=linear_q_config)

# Initialize weights (e.g., load from a pre-trained floating-point model)
# fxp_linear_layer.load_state_dict(...)

# If total_bits for weights/bias are set but fractional_bits are not,
# you can automatically determine fractional_bits to avoid overflow:
fxp_linear_layer.set_no_overflow_quant()

print("Quantization Config after set_no_overflow_quant:")
print(fxp_linear_layer.q_config.model_dump_json(indent=2))

# Create dummy input
dummy_input = torch.randn(1, 10)

# Forward pass (simulates fixed-point arithmetic)
# apply_ste=True uses Straight-Through Estimator for gradients during training
output = fxp_linear_layer(dummy_input, apply_ste=True)
print("\nOutput:", output)

# To get truly quantized weights (e.g., for export):
fxp_linear_layer.quantize_weights_bias()
print("\nQuantized Weight:", fxp_linear_layer.weight.data)
```

See the `tests/` directory for more detailed usage examples of different layers and quantization scenarios.

## Core Concepts

*   **`QType`**: Defines the bit-width (`total_bits`, `fractional_bits`) and `QMethod` for a specific tensor (input, weight, bias, activation).
*   **`*QConfig` (e.g., `LinearQConfig`)**: A Pydantic model that groups `QType` configurations for all relevant tensors within a specific layer type.
*   **`FxP*` layers**: PyTorch modules that implement fixed-point behavior. They typically inherit from a corresponding `*Transparent` layer.
    *   If `q_config` is `None`, they behave like standard floating-point layers.
    *   If `q_config` is provided, they simulate quantization during the forward pass.
*   **`set_no_overflow_quant()`**: A method on `FxP*` layers. If `total_bits` is specified in the `QType` for weights/biases, this method calculates the optimal `fractional_bits` to maximize precision while ensuring the current weight/bias values do not overflow.
*   **`set_high_precision_quant()`**: A method that configures weights/biases to use a high number of fractional bits (e.g., 24) and calculates the `total_bits` needed to represent their current dynamic range.
*   **`quantize_weights_bias()`**: A method to permanently alter the layer's weight and bias tensors to their quantized values. Useful before exporting weights.
*   **`ActivationLogger`**: A utility to log intermediate tensor values during the forward pass for debugging and analysis.

## Modules

*   **`fxp/`**: Contains the fixed-point layer implementations and core quantization logic.
    *   `symmetrics_quant.py`: Core symmetric quantization functions and `QType`/`QConfig` base.
    *   `utils.py`: Helper utilities like `ValueRange`.
    *   `fxp_*.py`: Specific fixed-point layer implementations.
*   **`transparent/`**: Contains "transparent" base layers that mirror standard PyTorch layers but include hooks for activation logging.
    *   `activation_logger.py`: The `ActivationLogger` class.
    *   `trans_*.py`: Specific transparent layer implementations.
*   **`tests/`**: Unit tests and usage examples.

## TODO / Future Work

*   [ ] Explore quantization schemes with non-power-of-2 scaling factors
*   [ ] Add support for asymmetric quantization.
*   [ ] More comprehensive testing scenarios.
*   [ ] Detailed documentation for each module and function.
*   [ ] Performance benchmarking.
*   [ ] Examples of exporting quantized weights for specific hardware targets.

## License

This project is licensed under the [MIT License](LICENSE). <!-- Make sure to add a LICENSE file -->
```
