# FxPyTorch/tests/fxp_tests.py

import torch
import os
import copy
from torch.nn.common_types import _size_2_t
from typing import Union

# Layer and Config Imports
from ..fxp.fxp_linear import FxPLinear, LinearQConfig
from ..fxp.fxp_dropout import FxPDropout, DropoutQConfig
from ..fxp.fxp_softmax import FxPSoftmax, SoftmaxQConfig
from ..fxp.fxp_layernorm import FxPLayerNorm, LayerNormQConfig
from ..fxp.fxp_multiheadattention import (
    FxPMultiheadAttention,
    MultiheadAttentionQConfig,
)
from ..fxp.fxp_transformer_encoder import (
    FxPTransformerEncoderLayer,
    TransformerEncoderLayerQConfig,
)
from ..fxp.fxp_conv2d import FxPConv2D, Conv2DQConfig
from ..fxp.fxp_sigmoid import FxPSigmoid, SigmoidQConfig
from ..fxp.fxp_tanh import FxPTanh, TanhQConfig
from ..fxp.fxp_lstm import FxPLSTM, LSTMQConfig

# Utility Imports
from ..transparent.activation_logger import ActivationLogger
from ..fxp.symmetric_quant import QType, QMethod

# ------------------------------------------------------------------------------
# Add a fixed random seed and generator for reproducibility
SEED = 42
generator = torch.Generator().manual_seed(SEED)
# ------------------------------------------------------------------------------

# --- FxPLinear Test (Existing) ---


def test_fxp_linear(
    in_features: int = 10,
    out_features: int = 5,
    batch_size: int = 1,
    calibration_batch_size: int = 128,
    bias: bool = True,
    output_path_tests: str = "outputs/fxp_linear_tests",
) -> None:
    # ------------------------------------------------------------------------------
    # Configuration & Initialization
    # ------------------------------------------------------------------------------
    IN_FEATURES = in_features
    OUT_FEATURES = out_features
    BATCH_SIZE = batch_size
    BIAS = bias
    OUTPUTS_PATH_TESTS = output_path_tests  # Folder for test outputs
    CALIBRATION_BATCH_SIZE = calibration_batch_size

    # Ensure the output directory exists
    os.makedirs(OUTPUTS_PATH_TESTS, exist_ok=True)
    print(f"FxPLinear test outputs will be saved in: {OUTPUTS_PATH_TESTS}")

    # Create a dummy input tensor (for a linear layer, shape: [Batch, Features])
    dummy_input = torch.ones((BATCH_SIZE, IN_FEATURES))
    calibration_dummy_input = torch.randn(
        (CALIBRATION_BATCH_SIZE, IN_FEATURES), generator=generator
    )
    print(f"\nDummy Input Shape: {dummy_input.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 1: Default Float (No explicit QConfig)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 20 + " FxPLinear Scenario 1: Default Float " + "=" * 20)
    # Create a layer with no explicit QConfig; it uses the default float behavior.
    layer1 = FxPLinear(IN_FEATURES, OUT_FEATURES, bias=BIAS)
    layer1.eval()  # Set to evaluation mode
    base_state_dict = layer1.state_dict()  # Store state for later loading
    print("Layer1 initialized (default float)")
    # Initialize the activation logger (store full tensors for detailed inspection)
    logger = ActivationLogger(enabled=True, store_full_tensors=True, model=layer1)

    # Run inference
    output1 = layer1(dummy_input, logger=logger)

    # Save activation logs
    log_path1 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_linear_float.json")
    logger.save_to_json(log_path1)

    print(f"Output1 shape: {output1.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 2: Explicit Float QConfig
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 20 + " FxPLinear Scenario 2: Explicit Float QConfig " + "=" * 20)
    logger.clear()

    # Create an explicit QConfig (defaults are float)
    float_qconfig = LinearQConfig()
    print(f"Explicit Float QConfig:\n{float_qconfig.model_dump_json(indent=2)}")

    # Instantiate a layer using the explicit float QConfig and load the baseline state_dict
    layer2 = FxPLinear(IN_FEATURES, OUT_FEATURES, bias=BIAS, q_config=float_qconfig)
    layer2.load_state_dict(base_state_dict)
    layer2.eval()
    print("Layer2 initialized (explicit float config)")

    # Run inference
    output2 = layer2(dummy_input, logger=logger)

    # Save activation logs
    log_path2 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_linear_float2.json")
    logger.save_to_json(log_path2)

    print(f"Output2 shape: {output2.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 3: High Precision Configuration ("High Precision" Params)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 20 + " FxPLinear Scenario 3: High Precision Config " + "=" * 20)
    logger.clear()

    # Create a new layer with a fresh QConfig for high precision
    layer3_qconfig = LinearQConfig()  # Start with default
    layer3 = FxPLinear(IN_FEATURES, OUT_FEATURES, bias=BIAS, q_config=layer3_qconfig)
    layer3.load_state_dict(base_state_dict)
    layer3.eval()
    print("Layer3 initialized")

    # Configure layer3 for high precision quantization
    layer3.set_high_precision_quant()
    print(
        f"\nLayer3 QConfig after set_high_precision_quant:\n{layer3.q_config.model_dump_json(indent=2)}"
    )

    # Run inference
    output3 = layer3(dummy_input, logger=logger)

    # Save activation logs
    log_path3 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_linear_high_precision.json")
    logger.save_to_json(log_path3)

    print(f"\nOutput3 shape: {output3.shape}")
    print(f"Output3:\n{output3}")

    # ------------------------------------------------------------------------------
    # Scenario 4: No Overflow (Fixed Total Bits = 16 for Params)
    # ------------------------------------------------------------------------------
    print(
        "\n" + "=" * 20 + " FxPLinear Scenario 4: No Overflow (T=16 Params) " + "=" * 20
    )
    logger.clear()

    # Create a QConfig specifying only total_bits for weight and bias
    qconfig4 = LinearQConfig(
        weight=QType(total_bits=16, q_method=QMethod.ROUND_SATURATE),
        bias=QType(total_bits=16, q_method=QMethod.ROUND_SATURATE) if BIAS else QType(),
        # Input/Activation remain float unless specified
    )
    print(f"Initial QConfig4 (partial):\n{qconfig4.model_dump_json(indent=2)}")

    # Instantiate and initialize the layer with qconfig4
    layer4 = FxPLinear(IN_FEATURES, OUT_FEATURES, bias=BIAS, q_config=qconfig4)
    layer4.load_state_dict(base_state_dict)
    layer4.eval()
    print("Layer4 initialized")

    # Calculate and set the missing fractional bits to avoid overflow
    layer4.set_no_overflow_quant()
    print(
        f"\nLayer4 QConfig after set_no_overflow_quant:\n{layer4.q_config.model_dump_json(indent=2)}"
    )

    # Run inference
    output4 = layer4(dummy_input, logger=logger)

    # Save activation logs
    log_path4 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_linear_t16_no_overflow.json")
    logger.save_to_json(log_path4)

    print(f"\nOutput4 shape: {output4.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 5: Mixed No Overflow (T=8 Params, T=16/F=8 Activation)
    # ------------------------------------------------------------------------------
    print(
        "\n"
        + "=" * 20
        + " FxPLinear Scenario 5: Mixed No Overflow (T=8 Params, 16/8 Act) "
        + "=" * 20
    )
    logger.clear()

    # Create a QConfig with explicit settings for input, weight, bias, and activation
    qconfig5 = LinearQConfig(
        input=QType(total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE),
        weight=QType(total_bits=8, q_method=QMethod.ROUND_SATURATE),
        bias=QType(total_bits=8, q_method=QMethod.ROUND_SATURATE) if BIAS else QType(),
        activation=QType(
            total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE
        ),
    )
    layer5 = FxPLinear(IN_FEATURES, OUT_FEATURES, bias=BIAS, q_config=qconfig5)
    layer5.load_state_dict(base_state_dict)
    layer5.eval()
    print("Layer5 initialized")
    print(
        f"Initial QConfig5 (partial weights/bias, fixed activation):\n{qconfig5.model_dump_json(indent=2)}"
    )

    # Set no overflow parameters for weights and bias (activation remains unchanged)
    layer5.set_no_overflow_quant()
    print(
        f"\nLayer5 QConfig after set_no_overflow_quant:\n{layer5.q_config.model_dump_json(indent=2)}"
    )

    # Run inference
    output5 = layer5(dummy_input, logger=logger)

    # Save activation logs
    log_path5 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_linear_mixed_no_overflow.json")
    logger.save_to_json(log_path5)

    print(f"\nOutput5 shape: {output5.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 6: Mixed No Overflow Calibrated (T=8 Params, T=16 Activation)
    # ------------------------------------------------------------------------------
    print(
        "\n"
        + "=" * 20
        + " FxPLinear Scenario 6: Mixed No Overflow Calibrated (T=8 Params, T=16 Act) "
        + "=" * 20
    )
    logger.clear()

    # Create a QConfig with explicit settings for input, weight, bias, and activation
    qconfig6 = LinearQConfig(
        input=QType(total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE),
        weight=QType(total_bits=8, q_method=QMethod.ROUND_SATURATE),
        bias=QType(total_bits=8, q_method=QMethod.ROUND_SATURATE) if BIAS else QType(),
        activation=QType(total_bits=16, q_method=QMethod.ROUND_SATURATE),
    )
    layer6 = FxPLinear(IN_FEATURES, OUT_FEATURES, bias=BIAS, q_config=qconfig6)
    layer6.load_state_dict(base_state_dict)
    layer6.eval()
    print("Layer6 initialized")
    print(
        f"Initial QConfig6 (partial weights/bias, fixed activation):\n{qconfig6.model_dump_json(indent=2)}"
    )

    # Set no overflow parameters for weights and bias (activation remains unchanged)
    layer6.set_no_overflow_quant()
    # Calibrate activations
    output6 = layer6(
        calibration_dummy_input,
        logger=logger,
        calibrate=True,
        calibration_type="no_overflow",
    )

    print(
        f"\nLayer6 QConfig after set_no_overflow_quant and calibration:\n{layer6.q_config.model_dump_json(indent=2)}"
    )

    # Save activation logs
    log_path6 = os.path.join(
        OUTPUTS_PATH_TESTS, "fxp_linear_mixed_no_overflow_calibration.json"
    )
    logger.save_to_json(log_path6)

    print(f"\nOutput6 shape: {output6.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 7: Mixed MinMSE Calibrated (T=8 Params, T=16 Activation)
    # ------------------------------------------------------------------------------
    print(
        "\n"
        + "=" * 20
        + " FxPLinear Scenario 7: Mixed MinMSE Calibrated (T=8 Params, T=16 Act) "
        + "=" * 20
    )
    logger.clear()

    # Create a QConfig with explicit settings for input, weight, bias, and activation
    qconfig7 = LinearQConfig(
        input=QType(total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE),
        weight=QType(total_bits=8, q_method=QMethod.ROUND_SATURATE),
        bias=QType(total_bits=8, q_method=QMethod.ROUND_SATURATE) if BIAS else QType(),
        activation=QType(total_bits=16, q_method=QMethod.ROUND_SATURATE),
    )
    layer7 = FxPLinear(IN_FEATURES, OUT_FEATURES, bias=BIAS, q_config=qconfig7)
    layer7.load_state_dict(base_state_dict)
    layer7.eval()
    print("Layer7 initialized")
    print(
        f"Initial QConfig7 (partial weights/bias, fixed activation):\n{qconfig7.model_dump_json(indent=2)}"
    )

    # Set min_mse parameters for weights and bias (activation remains unchanged)
    layer7.set_min_mse_quant()
    # Calibrate activations
    output7 = layer7(
        calibration_dummy_input,
        logger=logger,
        calibrate=True,
        calibration_type="min_mse",
    )

    print(
        f"\nLayer7 QConfig after set_min_mse_quant and min_mse calibration:\n{layer7.q_config.model_dump_json(indent=2)}"
    )

    # Save activation logs
    log_path7 = os.path.join(
        OUTPUTS_PATH_TESTS, "fxp_linear_mixed_min_mse_calibration.json"
    )
    logger.save_to_json(log_path7)

    print(f"\nOutput7 shape: {output7.shape}")

    # ------------------------------------------------------------------------------
    # Completion Message
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 20 + " FxPLinear Testing Complete " + "=" * 20)


# --- FxPDropout Test ---


def test_fxp_dropout(
    features: int = 10,  # Input feature size
    batch_size: int = 4,
    calibration_batch_size: int = 128,
    dropout_p: float = 0.5,  # Dropout probability
    output_path_tests: str = "outputs/fxp_dropout_tests",
) -> None:
    # ------------------------------------------------------------------------------
    # Configuration & Initialization
    # ------------------------------------------------------------------------------
    FEATURES = features
    BATCH_SIZE = batch_size
    DROPOUT_P = dropout_p
    OUTPUTS_PATH_TESTS = output_path_tests
    CALIBRATION_BATCH_SIZE = calibration_batch_size

    os.makedirs(OUTPUTS_PATH_TESTS, exist_ok=True)
    print(f"\nFxPDropout test outputs will be saved in: {OUTPUTS_PATH_TESTS}")

    # Create dummy input tensor
    dummy_input = torch.randn((BATCH_SIZE, FEATURES))  # Use randn for more variety
    calibration_dummy_input = torch.randn(
        (CALIBRATION_BATCH_SIZE, FEATURES), generator=generator
    )
    print(f"\nDummy Input Shape: {dummy_input.shape}")

    # --- IMPORTANT NOTE on Dropout ---
    # Standard nn.Dropout and likely FxPDropout behave differently in train() vs eval() mode.
    # In eval() mode (default for testing), dropout is typically *disabled* (identity function).
    # The quantization STE might still apply, but the dropout mask won't.
    # We keep .eval() for consistency with test_fxp_linear, but be aware of this.
    # If you need to test the dropout masking effect, you'd need layer.train().

    # ------------------------------------------------------------------------------
    # Scenario 1: Default Float (No explicit QConfig)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 20 + " FxPDropout Scenario 1: Default Float " + "=" * 20)

    layer1 = FxPDropout(p=DROPOUT_P)
    layer1.eval()  # Explicitly set to eval mode
    # Dropout has no state_dict to save/load
    print(f"Layer1 initialized (default float, p={DROPOUT_P})")
    # Initialize the activation logger (store full tensors for detailed inspection)
    logger = ActivationLogger(enabled=True, store_full_tensors=True, model=layer1)
    output1 = layer1(dummy_input, logger=logger)
    log_path1 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_dropout_float.json")
    logger.save_to_json(log_path1)

    print(f"Output1 shape: {output1.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 2: Explicit Float QConfig
    # ------------------------------------------------------------------------------
    print(
        "\n" + "=" * 20 + " FxPDropout Scenario 2: Explicit Float QConfig " + "=" * 20
    )
    logger.clear()

    float_qconfig = DropoutQConfig()
    print(f"Explicit Float QConfig:\n{float_qconfig.model_dump_json(indent=2)}")

    layer2 = FxPDropout(p=DROPOUT_P, q_config=float_qconfig)
    layer2.eval()
    print("Layer2 initialized (explicit float config)")

    output2 = layer2(dummy_input, logger=logger)
    log_path2 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_dropout_float2.json")
    logger.save_to_json(log_path2)
    print(f"Output2 shape: {output2.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 3: High Precision Configuration (Activations)
    # ------------------------------------------------------------------------------
    print(
        "\n"
        + "=" * 20
        + " FxPDropout Scenario 3: High Precision Config (Acts) "
        + "=" * 20
    )
    logger.clear()

    # FxPDropout has no parameters, so 'high precision' applies to input/activation
    # We manually set high precision QTypes for input/activation
    layer3_qconfig = DropoutQConfig(
        input=QType(total_bits=32, fractional_bits=24, q_method=QMethod.ROUND_SATURATE),
        activation=QType(
            total_bits=32, fractional_bits=24, q_method=QMethod.ROUND_SATURATE
        ),
    )
    layer3 = FxPDropout(p=DROPOUT_P, q_config=layer3_qconfig)
    layer3.eval()
    print("Layer3 initialized (High Precision Acts)")
    print(f"\nLayer3 QConfig:\n{layer3.q_config.model_dump_json(indent=2)}")

    # No set_high_precision_quant() method for Dropout as it has no params

    output3 = layer3(dummy_input, logger=logger)
    log_path3 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_dropout_high_precision.json")
    logger.save_to_json(log_path3)
    print(f"\nOutput3 shape: {output3.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 4: No Overflow (Fixed Total Bits = 16 for Activations)
    # ------------------------------------------------------------------------------
    print(
        "\n" + "=" * 20 + " FxPDropout Scenario 4: No Overflow (T=16 Acts) " + "=" * 20
    )
    logger.clear()

    # FxPDropout has no parameters, so 'no overflow' applies to input/activation
    # Manually define T=16 and choose reasonable fractional bits (e.g., F=8)
    # There's no automatic calculation based on parameter range.
    qconfig4 = DropoutQConfig(
        input=QType(total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE),
        activation=QType(
            total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE
        ),
    )
    print(f"Initial QConfig4:\n{qconfig4.model_dump_json(indent=2)}")

    layer4 = FxPDropout(p=DROPOUT_P, q_config=qconfig4)
    layer4.eval()
    print("Layer4 initialized (T=16 Acts)")

    # No set_no_overflow_quant() method for Dropout

    output4 = layer4(dummy_input, logger=logger)
    log_path4 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_dropout_t16_no_overflow.json")
    logger.save_to_json(log_path4)
    print(f"\nOutput4 shape: {output4.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 5: Mixed No Overflow (T=16/F=8 Input, T=8/F=4 Activation)
    # ------------------------------------------------------------------------------
    print(
        "\n" + "=" * 20 + " FxPDropout Scenario 5: Mixed No Overflow (Acts) " + "=" * 20
    )
    logger.clear()

    qconfig5 = DropoutQConfig(
        input=QType(total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE),
        activation=QType(
            total_bits=8, fractional_bits=4, q_method=QMethod.ROUND_SATURATE
        ),
    )
    layer5 = FxPDropout(p=DROPOUT_P, q_config=qconfig5)
    layer5.eval()
    print("Layer5 initialized (Mixed Precision Acts)")
    print(f"QConfig5:\n{qconfig5.model_dump_json(indent=2)}")

    # No set_no_overflow_quant() method for Dropout

    output5 = layer5(dummy_input, logger=logger)
    log_path5 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_dropout_mixed_no_overflow.json")
    logger.save_to_json(log_path5)
    print(f"\nOutput5 shape: {output5.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 6: Mixed No Overflow Calibrated (T=16/F=8 Input, T=8 Activation)
    # ------------------------------------------------------------------------------
    print(
        "\n" + "=" * 20 + " FxPDropout Scenario 6: Mixed No Overflow (Acts) " + "=" * 20
    )
    logger.clear()

    qconfig6 = DropoutQConfig(
        input=QType(total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE),
        activation=QType(total_bits=8, q_method=QMethod.ROUND_SATURATE),
    )
    layer6 = FxPDropout(p=DROPOUT_P, q_config=qconfig6)
    layer6.eval()
    print("Layer6 initialized (Mixed Precision Acts)")
    print(f"QConfig6:\n{qconfig6.model_dump_json(indent=2)}")

    # No set_no_overflow_quant() method for Dropout

    output6 = layer6(
        calibration_dummy_input,
        logger=logger,
        calibrate=True,
        calibration_type="no_overflow",
    )
    print(
        f"\nLayer6 QConfig after calibration:\n{layer6.q_config.model_dump_json(indent=2)}"
    )
    log_path6 = os.path.join(
        OUTPUTS_PATH_TESTS, "fxp_dropout_mixed_no_overflow_calibration.json"
    )
    logger.save_to_json(log_path6)
    print(f"\nOutput6 shape: {output6.shape}")
    # ------------------------------------------------------------------------------
    # Scenario 7: Mixed MinMSE Calibrated (T=16/F=8 Input, T=8 Activation)
    # ------------------------------------------------------------------------------
    print(
        "\n"
        + "=" * 20
        + " FxPDropout Scenario 7: Mixed MinMSE Calibrated (Acts) "
        + "=" * 20
    )
    logger.clear()

    qconfig7 = DropoutQConfig(
        input=QType(total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE),
        activation=QType(total_bits=8, q_method=QMethod.ROUND_SATURATE),
    )
    layer7 = FxPDropout(p=DROPOUT_P, q_config=qconfig7)
    layer7.eval()
    print("Layer7 initialized (Mixed MinMSE Calibrated Acts)")
    print(f"QConfig7:\n{qconfig7.model_dump_json(indent=2)}")

    # No set_no_overflow_quant() method for Dropout

    output7 = layer7(
        calibration_dummy_input,
        logger=logger,
        calibrate=True,
        calibration_type="min_mse",
    )
    print(
        f"\nLayer7 QConfig after calibration:\n{layer7.q_config.model_dump_json(indent=2)}"
    )
    log_path7 = os.path.join(
        OUTPUTS_PATH_TESTS, "fxp_dropout_mixed_min_mse_calibration.json"
    )
    logger.save_to_json(log_path7)
    print(f"\nOutput7 shape: {output7.shape}")

    print("\n" + "=" * 20 + " FxPDropout Testing Complete " + "=" * 20)


# --- FxPSoftmax Test ---


def test_fxp_softmax(
    features: int = 10,  # Input feature size
    batch_size: int = 4,
    calibration_batch_size: int = 128,
    softmax_dim: int = -1,  # Dimension along which softmax is computed
    output_path_tests: str = "outputs/fxp_softmax_tests",
) -> None:
    # ------------------------------------------------------------------------------
    # Configuration & Initialization
    # ------------------------------------------------------------------------------
    FEATURES = features
    BATCH_SIZE = batch_size
    SOFTMAX_DIM = softmax_dim
    OUTPUTS_PATH_TESTS = output_path_tests
    CALIBRATION_BATCH_SIZE = calibration_batch_size

    os.makedirs(OUTPUTS_PATH_TESTS, exist_ok=True)
    print(f"\nFxPSoftmax test outputs will be saved in: {OUTPUTS_PATH_TESTS}")

    # Create dummy input tensor (pre-softmax logits)
    dummy_input = torch.randn((BATCH_SIZE, FEATURES))
    calibration_dummy_input = torch.randn(
        (CALIBRATION_BATCH_SIZE, FEATURES), generator=generator
    )
    print(f"\nDummy Input Shape: {dummy_input.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 1: Default Float (No explicit QConfig)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 20 + " FxPSoftmax Scenario 1: Default Float " + "=" * 20)

    layer1 = FxPSoftmax(dim=SOFTMAX_DIM)
    layer1.eval()
    # Softmax has no state_dict
    print(f"Layer1 initialized (default float, dim={SOFTMAX_DIM})")
    # Initialize the activation logger (store full tensors for detailed inspection)
    logger = ActivationLogger(enabled=True, store_full_tensors=True, model=layer1)
    output1 = layer1(dummy_input, logger=logger)
    log_path1 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_softmax_float.json")
    logger.save_to_json(log_path1)
    print(f"Output1 shape: {output1.shape}")
    # Optional: Check if output sums to 1 along the dimension
    # print(f"Output1 sums (dim={SOFTMAX_DIM}): {output1.sum(dim=SOFTMAX_DIM)}")

    # ------------------------------------------------------------------------------
    # Scenario 2: Explicit Float QConfig
    # ------------------------------------------------------------------------------
    print(
        "\n" + "=" * 20 + " FxPSoftmax Scenario 2: Explicit Float QConfig " + "=" * 20
    )
    logger.clear()

    float_qconfig = SoftmaxQConfig()
    print(f"Explicit Float QConfig:\n{float_qconfig.model_dump_json(indent=2)}")

    layer2 = FxPSoftmax(dim=SOFTMAX_DIM, q_config=float_qconfig)
    layer2.eval()
    print("Layer2 initialized (explicit float config)")

    output2 = layer2(dummy_input, logger=logger)
    log_path2 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_softmax_float2.json")
    logger.save_to_json(log_path2)
    print(f"Output2 shape: {output2.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 3: High Precision Configuration (Activations)
    # ------------------------------------------------------------------------------
    print(
        "\n"
        + "=" * 20
        + " FxPSoftmax Scenario 3: High Precision Config (Acts) "
        + "=" * 20
    )
    logger.clear()

    # FxPSoftmax has no parameters. Manually set high precision for input/activation.
    layer3_qconfig = SoftmaxQConfig(
        input=QType(
            total_bits=32, fractional_bits=24, q_method=QMethod.ROUND_SATURATE
        ),  # Input (logits)
        activation=QType(
            total_bits=32, fractional_bits=30, q_method=QMethod.ROUND_SATURATE
        ),  # Output (probs, [0,1]) - need many frac bits
    )
    layer3 = FxPSoftmax(dim=SOFTMAX_DIM, q_config=layer3_qconfig)
    layer3.eval()
    print("Layer3 initialized (High Precision Acts)")
    print(f"\nLayer3 QConfig:\n{layer3.q_config.model_dump_json(indent=2)}")

    # No set_high_precision_quant() method

    output3 = layer3(dummy_input, logger=logger)
    log_path3 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_softmax_high_precision.json")
    logger.save_to_json(log_path3)
    print(f"\nOutput3 shape: {output3.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 4: No Overflow (Fixed Total Bits = 16 for Activations)
    # ------------------------------------------------------------------------------
    print(
        "\n" + "=" * 20 + " FxPSoftmax Scenario 4: No Overflow (T=16 Acts) " + "=" * 20
    )
    logger.clear()

    # FxPSoftmax has no parameters. Manually define T=16.
    # Input (logits) might need some integer bits. Output (probs) needs mostly fractional bits.
    qconfig4 = SoftmaxQConfig(
        input=QType(
            total_bits=16, fractional_bits=10, q_method=QMethod.ROUND_SATURATE
        ),  # T=16, F=10 for logits
        activation=QType(
            total_bits=16, fractional_bits=15, q_method=QMethod.ROUND_SATURATE
        ),  # T=16, F=15 for probs [0,1]
    )
    print(f"Initial QConfig4:\n{qconfig4.model_dump_json(indent=2)}")

    layer4 = FxPSoftmax(dim=SOFTMAX_DIM, q_config=qconfig4)
    layer4.eval()
    print("Layer4 initialized (T=16 Acts)")

    # No set_no_overflow_quant() method

    output4 = layer4(dummy_input, logger=logger)
    log_path4 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_softmax_t16_no_overflow.json")
    logger.save_to_json(log_path4)
    print(f"\nOutput4 shape: {output4.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 5: Mixed No Overflow (T=16/F=10 Input, T=8/F=7 Activation)
    # ------------------------------------------------------------------------------
    print(
        "\n" + "=" * 20 + " FxPSoftmax Scenario 5: Mixed No Overflow (Acts) " + "=" * 20
    )
    logger.clear()

    qconfig5 = SoftmaxQConfig(
        input=QType(
            total_bits=16, fractional_bits=10, q_method=QMethod.ROUND_SATURATE
        ),  # Input logits
        activation=QType(
            total_bits=8, fractional_bits=7, q_method=QMethod.ROUND_SATURATE
        ),  # Output probs [0,1] need high frac proportion
    )
    layer5 = FxPSoftmax(dim=SOFTMAX_DIM, q_config=qconfig5)
    layer5.eval()
    print("Layer5 initialized (Mixed Precision Acts)")
    print(f"QConfig5:\n{qconfig5.model_dump_json(indent=2)}")

    # No set_no_overflow_quant() method

    output5 = layer5(dummy_input, logger=logger)
    log_path5 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_softmax_mixed_no_overflow.json")
    logger.save_to_json(log_path5)
    print(f"\nOutput5 shape: {output5.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 6: Mixed No Overflow Calibrated (T=16/F=10 Input, T=8 Activation)
    # ------------------------------------------------------------------------------
    print(
        "\n"
        + "=" * 20
        + " FxPSoftmax Scenario 6: Mixed No Overflow Calibrated (Acts) "
        + "=" * 20
    )
    logger.clear()

    qconfig6 = SoftmaxQConfig(
        input=QType(
            total_bits=16, fractional_bits=10, q_method=QMethod.ROUND_SATURATE
        ),  # Input logits
        activation=QType(
            total_bits=8, q_method=QMethod.ROUND_SATURATE
        ),  # Output probs [0,1] need high frac proportion
    )
    layer6 = FxPSoftmax(dim=SOFTMAX_DIM, q_config=qconfig6)
    layer6.eval()
    print("Layer6 initialized (Mixed Precision Acts)")
    print(f"QConfig6:\n{qconfig6.model_dump_json(indent=2)}")

    # No set_no_overflow_quant() method

    output6 = layer6(
        calibration_dummy_input,
        logger=logger,
        calibrate=True,
        calibration_type="no_overflow",
    )
    print(
        f"\nLayer6 QConfig after calibration:\n{layer6.q_config.model_dump_json(indent=2)}"
    )
    log_path6 = os.path.join(
        OUTPUTS_PATH_TESTS, "fxp_softmax_mixed_no_overflow_calibration.json"
    )
    logger.save_to_json(log_path6)
    print(f"\nOutput6 shape: {output6.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 7:  Mixed MinMSE Calibrated (T=16/F=10 Input, T=8 Activation)
    # ------------------------------------------------------------------------------
    print(
        "\n"
        + "=" * 20
        + " FxPSoftmax Scenario 7: Mixed MinMSE Calibrated (Acts) "
        + "=" * 20
    )
    logger.clear()

    qconfig7 = SoftmaxQConfig(
        input=QType(
            total_bits=16, fractional_bits=10, q_method=QMethod.ROUND_SATURATE
        ),  # Input logits
        activation=QType(
            total_bits=8, q_method=QMethod.ROUND_SATURATE
        ),  # Output probs [0,1] need high frac proportion
    )
    layer7 = FxPSoftmax(dim=SOFTMAX_DIM, q_config=qconfig7)
    layer7.eval()
    print("Layer7 initialized (Mixed MinMSE Calibrated Acts)")
    print(f"QConfig7:\n{qconfig7.model_dump_json(indent=2)}")

    # No set_no_overflow_quant() method

    output7 = layer7(
        calibration_dummy_input,
        logger=logger,
        calibrate=True,
        calibration_type="min_mse",
    )
    print(
        f"\nLayer7 QConfig after calibration:\n{layer7.q_config.model_dump_json(indent=2)}"
    )
    log_path7 = os.path.join(
        OUTPUTS_PATH_TESTS, "fxp_softmax_mixed_min_mse_calibration.json"
    )
    logger.save_to_json(log_path7)
    print(f"\nOutput7 shape: {output7.shape}")

    print("\n" + "=" * 20 + " FxPSoftmax Testing Complete " + "=" * 20)


# --- FxPLayerNorm Test ---


def test_fxp_layernorm(
    features: int = 10,  # Feature size to normalize
    batch_size: int = 4,
    calibration_batch_size: int = 128,
    eps: float = 1e-5,
    elementwise_affine: bool = True,  # Test with learnable params
    bias: bool = True,
    output_path_tests: str = "outputs/fxp_layernorm_tests",
) -> None:
    # ------------------------------------------------------------------------------
    # Configuration & Initialization
    # ------------------------------------------------------------------------------
    FEATURES = features
    NORMALIZED_SHAPE = features  # Normalize the last dimension
    BATCH_SIZE = batch_size
    EPS = eps
    AFFINE = elementwise_affine
    BIAS = bias and AFFINE  # Bias only relevant if affine is True
    OUTPUTS_PATH_TESTS = output_path_tests
    CALIBRATION_BATCH_SIZE = calibration_batch_size

    os.makedirs(OUTPUTS_PATH_TESTS, exist_ok=True)
    print(f"\nFxPLayerNorm test outputs will be saved in: {OUTPUTS_PATH_TESTS}")

    # Create dummy input tensor
    dummy_input = torch.randn((BATCH_SIZE, FEATURES)) * 5 + 2  # Add offset and scale
    calibration_dummy_input = (
        torch.randn((CALIBRATION_BATCH_SIZE, FEATURES), generator=generator) * 5 + 2
    )  # Add offset and scale
    print(f"\nDummy Input Shape: {dummy_input.shape}")
    print(f"Input Range: [{dummy_input.min():.2f}, {dummy_input.max():.2f}]")

    # ------------------------------------------------------------------------------
    # Scenario 1: Default Float (No explicit QConfig)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 20 + " FxPLayerNorm Scenario 1: Default Float " + "=" * 20)

    layer1 = FxPLayerNorm(
        NORMALIZED_SHAPE, eps=EPS, elementwise_affine=AFFINE, bias=BIAS
    )
    layer1.eval()
    base_state_dict = (
        layer1.state_dict() if AFFINE else None
    )  # Save state if params exist
    print("Layer1 initialized (default float)")
    # Initialize the activation logger (store full tensors for detailed inspection)
    logger = ActivationLogger(enabled=True, store_full_tensors=True, model=layer1)

    output1 = layer1(dummy_input, logger=logger)
    log_path1 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_layernorm_float.json")
    logger.save_to_json(log_path1)
    print(f"Output1 shape: {output1.shape}")
    # Optional: Check mean/std of output (should be approx 0/1 if affine=False or not quantized)
    # print(f"Output1 Mean: {output1.mean():.4f}, Std: {output1.std():.4f}")

    # ------------------------------------------------------------------------------
    # Scenario 2: Explicit Float QConfig
    # ------------------------------------------------------------------------------
    print(
        "\n" + "=" * 20 + " FxPLayerNorm Scenario 2: Explicit Float QConfig " + "=" * 20
    )
    logger.clear()

    float_qconfig = LayerNormQConfig()
    print(f"Explicit Float QConfig:\n{float_qconfig.model_dump_json(indent=2)}")

    layer2 = FxPLayerNorm(
        NORMALIZED_SHAPE,
        eps=EPS,
        elementwise_affine=AFFINE,
        bias=BIAS,
        q_config=float_qconfig,
    )
    if AFFINE:
        layer2.load_state_dict(base_state_dict)
    layer2.eval()
    print("Layer2 initialized (explicit float config)")

    output2 = layer2(dummy_input, logger=logger)
    log_path2 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_layernorm_float2.json")
    logger.save_to_json(log_path2)
    print(f"Output2 shape: {output2.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 3: High Precision Configuration (Params & Acts)
    # ------------------------------------------------------------------------------
    print(
        "\n" + "=" * 20 + " FxPLayerNorm Scenario 3: High Precision Config " + "=" * 20
    )
    logger.clear()

    layer3_qconfig = LayerNormQConfig()  # Start with default float
    # Manually set high precision for internal calculations too if desired

    layer3 = FxPLayerNorm(
        NORMALIZED_SHAPE,
        eps=EPS,
        elementwise_affine=AFFINE,
        bias=BIAS,
        q_config=layer3_qconfig,
    )
    if AFFINE:
        layer3.load_state_dict(base_state_dict)
    layer3.eval()
    print("Layer3 initialized")

    # Configure learnable parameters (if they exist) for high precision
    if AFFINE:
        layer3.set_high_precision_quant()
    print(
        f"\nLayer3 QConfig after set_high_precision_quant (if affine):\n{layer3.q_config.model_dump_json(indent=2)}"
    )

    output3 = layer3(dummy_input, logger=logger)
    log_path3 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_layernorm_high_precision.json")
    logger.save_to_json(log_path3)
    print(f"\nOutput3 shape: {output3.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 4: No Overflow (Fixed Total Bits = 16 for Params)
    # ------------------------------------------------------------------------------
    print(
        "\n"
        + "=" * 20
        + " FxPLayerNorm Scenario 4: No Overflow (T=16 Params) "
        + "=" * 20
    )
    logger.clear()

    # Define QConfig with T=16 for weight/bias only (if affine)
    # Set other parts to float or a default reasonable quantization (e.g., T=16/F=8)
    qconfig4 = LayerNormQConfig(
        weight=QType(total_bits=16, q_method=QMethod.ROUND_SATURATE)
        if AFFINE
        else QType(),
        bias=QType(total_bits=16, q_method=QMethod.ROUND_SATURATE) if BIAS else QType(),
        # Let's quantize internal steps too for a more complete test
        input=QType(total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE),
        mean_tensor=QType(
            total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE
        ),
        var_tensor=QType(
            total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE
        ),
        input_normalized=QType(
            total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE
        ),
        activation=QType(
            total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE
        ),
    )
    print(f"Initial QConfig4 (partial params):\n{qconfig4.model_dump_json(indent=2)}")

    layer4 = FxPLayerNorm(
        NORMALIZED_SHAPE,
        eps=EPS,
        elementwise_affine=AFFINE,
        bias=BIAS,
        q_config=qconfig4,
    )
    if AFFINE:
        layer4.load_state_dict(base_state_dict)
    layer4.eval()
    print("Layer4 initialized")

    # Calculate fractional bits for weight/bias (if affine)
    if AFFINE:
        layer4.set_no_overflow_quant()
    print(
        f"\nLayer4 QConfig after set_no_overflow_quant (if affine):\n{layer4.q_config.model_dump_json(indent=2)}"
    )

    output4 = layer4(dummy_input, logger=logger)
    log_path4 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_layernorm_t16_no_overflow.json")
    logger.save_to_json(log_path4)
    print(f"\nOutput4 shape: {output4.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 5: Mixed No Overflow (T=8 Params, T=16/F=8 Others)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 20 + " FxPLayerNorm Scenario 5: Mixed No Overflow " + "=" * 20)
    logger.clear()

    # Define T=8 for weight/bias, T=16/F=8 for others
    qconfig5 = LayerNormQConfig(
        input=QType(total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE),
        weight=QType(total_bits=8, q_method=QMethod.ROUND_SATURATE)
        if AFFINE
        else QType(),
        bias=QType(total_bits=8, q_method=QMethod.ROUND_SATURATE) if BIAS else QType(),
        mean_tensor=QType(
            total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE
        ),
        var_tensor=QType(
            total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE
        ),
        input_normalized=QType(
            total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE
        ),
        activation=QType(
            total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE
        ),
    )
    layer5 = FxPLayerNorm(
        NORMALIZED_SHAPE,
        eps=EPS,
        elementwise_affine=AFFINE,
        bias=BIAS,
        q_config=qconfig5,
    )
    if AFFINE:
        layer5.load_state_dict(base_state_dict)
    layer5.eval()
    print("Layer5 initialized")
    print(f"Initial QConfig5:\n{qconfig5.model_dump_json(indent=2)}")

    # Set no overflow frac bits for T=8 weight/bias (if affine)
    if AFFINE:
        layer5.set_no_overflow_quant()
    print(
        f"\nLayer5 QConfig after set_no_overflow_quant (if affine):\n{layer5.q_config.model_dump_json(indent=2)}"
    )

    output5 = layer5(dummy_input, logger=logger)
    log_path5 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_layernorm_mixed_no_overflow.json")
    logger.save_to_json(log_path5)
    print(f"\nOutput5 shape: {output5.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 6: Mixed No Overflow Calibrated (T=8 Params, T=16 Others)
    # ------------------------------------------------------------------------------
    print(
        "\n"
        + "=" * 20
        + " FxPLayerNorm Scenario 6: Mixed No Overflow Calibrated"
        + "=" * 20
    )
    logger.clear()

    # Define T=8 for weight/bias, T=16for others
    qconfig6 = LayerNormQConfig(
        input=QType(total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE),
        weight=QType(total_bits=8, q_method=QMethod.ROUND_SATURATE)
        if AFFINE
        else QType(),
        bias=QType(total_bits=8, q_method=QMethod.ROUND_SATURATE) if BIAS else QType(),
        mean_tensor=QType(total_bits=16, q_method=QMethod.ROUND_SATURATE),
        var_tensor=QType(total_bits=16, q_method=QMethod.ROUND_SATURATE),
        input_normalized=QType(total_bits=16, q_method=QMethod.ROUND_SATURATE),
        activation=QType(total_bits=16, q_method=QMethod.ROUND_SATURATE),
    )
    layer6 = FxPLayerNorm(
        NORMALIZED_SHAPE,
        eps=EPS,
        elementwise_affine=AFFINE,
        bias=BIAS,
        q_config=qconfig6,
    )
    if AFFINE:
        layer6.load_state_dict(base_state_dict)
    layer6.eval()
    print("Layer6 initialized")
    print(f"Initial QConfig6:\n{qconfig6.model_dump_json(indent=2)}")

    # Set no overflow frac bits for T=8 weight/bias (if affine)
    if AFFINE:
        layer6.set_no_overflow_quant()
    output6 = layer6(
        calibration_dummy_input,
        logger=logger,
        calibrate=True,
        calibration_type="no_overflow",
    )
    print(
        f"\nLayer6 QConfig after set_no_overflow_quant and calibration (if affine):\n{layer6.q_config.model_dump_json(indent=2)}"
    )

    log_path6 = os.path.join(
        OUTPUTS_PATH_TESTS, "fxp_layernorm_mixed_no_overflow_calibration.json"
    )
    logger.save_to_json(log_path6)
    print(f"\nOutput6 shape: {output6.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 7: Mixed MinMSE Calibrated (T=8 Params, T=16 Others)
    # ------------------------------------------------------------------------------
    print(
        "\n" + "=" * 20 + " FxPLayerNorm Scenario 7: Mixed MinMSE Calibrated" + "=" * 20
    )
    logger.clear()

    # Define T=8 for weight/bias, T=16for others
    qconfig7 = LayerNormQConfig(
        input=QType(total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE),
        weight=QType(total_bits=8, q_method=QMethod.ROUND_SATURATE)
        if AFFINE
        else QType(),
        bias=QType(total_bits=8, q_method=QMethod.ROUND_SATURATE) if BIAS else QType(),
        mean_tensor=QType(total_bits=16, q_method=QMethod.ROUND_SATURATE),
        var_tensor=QType(total_bits=16, q_method=QMethod.ROUND_SATURATE),
        input_normalized=QType(total_bits=16, q_method=QMethod.ROUND_SATURATE),
        activation=QType(total_bits=16, q_method=QMethod.ROUND_SATURATE),
    )
    layer7 = FxPLayerNorm(
        NORMALIZED_SHAPE,
        eps=EPS,
        elementwise_affine=AFFINE,
        bias=BIAS,
        q_config=qconfig7,
    )
    if AFFINE:
        layer7.load_state_dict(base_state_dict)
    layer7.eval()
    print("Layer7 initialized")
    print(f"Initial QConfig7:\n{qconfig7.model_dump_json(indent=2)}")

    # Set no overflow frac bits for T=8 weight/bias (if affine)
    if AFFINE:
        layer7.set_min_mse_quant()
    output7 = layer7(
        calibration_dummy_input,
        logger=logger,
        calibrate=True,
        calibration_type="min_mse",
    )
    print(
        f"\nLayer7 QConfig after set_min_mse_quant and min_mse calibration (if affine):\n{layer7.q_config.model_dump_json(indent=2)}"
    )

    log_path7 = os.path.join(
        OUTPUTS_PATH_TESTS, "fxp_layernorm_mixed_min_mse_calibration.json"
    )
    logger.save_to_json(log_path7)
    print(f"\nOutput7 shape: {output7.shape}")

    print("\n" + "=" * 20 + " FxPLayerNorm Testing Complete " + "=" * 20)


def test_fxp_multiheadattention(
    embed_dim: int = 16,  # Keep dimensions small for testing
    num_heads: int = 2,  # Must divide embed_dim
    seq_len: int = 8,
    batch_size: int = 4,
    calibration_batch_size: int = 128,
    dropout: float = 0.1,  # Corresponds to dropout AFTER softmax * V
    bias: bool = True,  # Bias for out_proj
    add_bias_kv: bool = False,  # Bias for K and V projections
    add_bias_q: bool = False,  # Bias for Q projection
    output_path_tests: str = "outputs/fxp_mha_tests",
) -> None:
    # ------------------------------------------------------------------------------
    # Configuration & Initialization
    # ------------------------------------------------------------------------------
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
    EMBED_DIM = embed_dim
    NUM_HEADS = num_heads
    SEQ_LEN = seq_len
    BATCH_SIZE = batch_size
    DROPOUT = dropout
    BIAS = bias
    ADD_BIAS_KV = add_bias_kv
    ADD_BIAS_Q = add_bias_q
    OUTPUTS_PATH_TESTS = output_path_tests
    CALIBRATION_BATCH_SIZE = calibration_batch_size

    os.makedirs(OUTPUTS_PATH_TESTS, exist_ok=True)
    print(
        f"\nFxPMultiheadAttention test outputs will be saved in: {OUTPUTS_PATH_TESTS}"
    )

    # Create dummy input tensor (common for query, key, value in self-attention)
    dummy_input = torch.ones((BATCH_SIZE, SEQ_LEN, EMBED_DIM))
    calibration_dummy_input = torch.randn(
        (CALIBRATION_BATCH_SIZE, SEQ_LEN, EMBED_DIM), generator=generator
    )
    print(f"\nDummy Input Shape (Query/Key/Value): {dummy_input.shape}")
    print(f"Input Range: [{dummy_input.min():.2f}, {dummy_input.max():.2f}]")

    # ------------------------------------------------------------------------------
    # Scenario 1: Default Float (No explicit QConfig)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 20 + " FxPMHA Scenario 1: Default Float " + "=" * 20)

    layer1 = FxPMultiheadAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        bias=BIAS,
        add_bias_kv=ADD_BIAS_KV,
        add_bias_q=ADD_BIAS_Q,
        batch_first=True,  # Important, FxP version assumes this
        # Using default internal layers (FxPLinear, FxPSoftmax, FxPDropout)
    )
    layer1.eval()
    base_state_dict = layer1.state_dict()  # Save state for later loading
    print("Layer1 initialized (default float)")
    # Initialize the activation logger (store full tensors for detailed inspection)
    logger = ActivationLogger(enabled=True, store_full_tensors=True, model=layer1)
    # MHA returns output and optionally attention weights
    output1, _ = layer1(
        dummy_input, dummy_input, dummy_input, logger=logger, need_weights=False
    )
    log_path1 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_mha_float.json")
    logger.save_to_json(log_path1)
    print(f"Output1 shape: {output1.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 2: Explicit Float QConfig
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 20 + " FxPMHA Scenario 2: Explicit Float QConfig " + "=" * 20)
    logger.clear()

    float_qconfig = MultiheadAttentionQConfig()  # Creates default (float) config
    print(f"Explicit Float QConfig:\n{float_qconfig.model_dump_json(indent=2)}")

    layer2 = FxPMultiheadAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        bias=BIAS,
        add_bias_kv=ADD_BIAS_KV,
        add_bias_q=ADD_BIAS_Q,
        batch_first=True,
        q_config=float_qconfig,  # Pass the float config
    )
    layer2.load_state_dict(base_state_dict)
    layer2.eval()
    print("Layer2 initialized (explicit float config)")

    output2, _ = layer2(
        dummy_input, dummy_input, dummy_input, logger=logger, need_weights=False
    )
    log_path2 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_mha_float2.json")
    logger.save_to_json(log_path2)
    print(f"Output2 shape: {output2.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 3: High Precision Configuration (Params & Acts)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 20 + " FxPMHA Scenario 3: High Precision Config " + "=" * 20)
    logger.clear()

    # Start with a default config, then modify
    layer3_qconfig = MultiheadAttentionQConfig()

    layer3 = FxPMultiheadAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        bias=BIAS,
        add_bias_kv=ADD_BIAS_KV,
        add_bias_q=ADD_BIAS_Q,
        batch_first=True,
        q_config=layer3_qconfig,  # Pass the modified high-precision config
    )
    layer3.load_state_dict(base_state_dict)
    layer3.eval()
    print("Layer3 initialized (High Precision)")

    # Configure learnable parameters for high precision using the layer's method
    layer3.set_high_precision_quant()
    print(
        f"\nLayer3 QConfig after set_high_precision_quant:\n{layer3.q_config.model_dump_json(indent=2)}"
    )

    output3, _ = layer3(
        dummy_input, dummy_input, dummy_input, logger=logger, need_weights=False
    )
    log_path3 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_mha_high_precision.json")
    logger.save_to_json(log_path3)
    print(f"\nOutput3 shape: {output3.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 4: No Overflow (Fixed Total Bits = 16 for Params)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 20 + " FxPMHA Scenario 4: No Overflow (T=16 Params) " + "=" * 20)
    logger.clear()

    # Define QConfig: T=16 for weights/biases in linear layers.
    t16_param_qtype = QType(
        total_bits=16, q_method=QMethod.ROUND_SATURATE
    )  # Only total_bits specified for params

    t16_linear_qconfig = LinearQConfig(
        weight=copy.deepcopy(t16_param_qtype),
        bias=copy.deepcopy(t16_param_qtype)
        if BIAS or ADD_BIAS_KV or ADD_BIAS_Q
        else QType(),  # Apply if any bias exists
    )

    qconfig4 = MultiheadAttentionQConfig(
        qlinear=copy.deepcopy(
            t16_linear_qconfig
        ),  # Use deepcopy to avoid aliasing issues if modifying later
        klinear=copy.deepcopy(t16_linear_qconfig),
        vlinear=copy.deepcopy(t16_linear_qconfig),
        out_proj=copy.deepcopy(t16_linear_qconfig),
    )
    # Need to handle biases specifically based on flags
    if not ADD_BIAS_Q:
        qconfig4.qlinear.bias = QType()
    if not ADD_BIAS_KV:
        qconfig4.klinear.bias = QType()
        qconfig4.vlinear.bias = QType()
    if not BIAS:
        qconfig4.out_proj.bias = QType()

    print(f"Initial QConfig4 (partial params):\n{qconfig4.model_dump_json(indent=2)}")

    layer4 = FxPMultiheadAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        bias=BIAS,
        add_bias_kv=ADD_BIAS_KV,
        add_bias_q=ADD_BIAS_Q,
        batch_first=True,
        q_config=qconfig4,
    )
    layer4.load_state_dict(base_state_dict)
    layer4.eval()
    print("Layer4 initialized")

    # Calculate fractional bits for T=16 parameters
    layer4.set_no_overflow_quant()
    print(
        f"\nLayer4 QConfig after set_no_overflow_quant:\n{layer4.q_config.model_dump_json(indent=2)}"
    )

    output4, _ = layer4(
        dummy_input, dummy_input, dummy_input, logger=logger, need_weights=False
    )
    log_path4 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_mha_t16_no_overflow.json")
    logger.save_to_json(log_path4)
    print(f"\nOutput4 shape: {output4.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 5: Mixed No Overflow (T=8 Params, T=16/F=8 Others)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 20 + " FxPMHA Scenario 5: Mixed No Overflow " + "=" * 20)
    logger.clear()

    # Define QConfig: T=8 for params, T=16/F=8 for others
    t16f8_qtype = QType(
        total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE
    )
    t8_param_qtype = QType(
        total_bits=8, q_method=QMethod.ROUND_SATURATE
    )  # Only T=8 for params

    t8_linear_qconfig = LinearQConfig(
        input=copy.deepcopy(t16f8_qtype),
        weight=copy.deepcopy(t8_param_qtype),
        bias=copy.deepcopy(t8_param_qtype)
        if BIAS or ADD_BIAS_KV or ADD_BIAS_Q
        else QType(),
        activation=copy.deepcopy(t16f8_qtype),
    )
    # Reuse T=16/F=8 configs for softmax/dropout from Scenario 4
    t16_softmax_qconfig = SoftmaxQConfig(
        input=copy.deepcopy(t16f8_qtype),
        activation=QType(
            total_bits=16, fractional_bits=15, q_method=QMethod.ROUND_SATURATE
        ),
    )  # Activation doesnt need more than 1 integer
    t16_dropout_qconfig = DropoutQConfig(
        input=copy.deepcopy(t16f8_qtype), activation=copy.deepcopy(t16f8_qtype)
    )

    qconfig5 = MultiheadAttentionQConfig(
        input_query=copy.deepcopy(t16f8_qtype),
        input_key=copy.deepcopy(t16f8_qtype),
        input_value=copy.deepcopy(t16f8_qtype),
        qlinear=copy.deepcopy(t8_linear_qconfig),
        klinear=copy.deepcopy(t8_linear_qconfig),
        vlinear=copy.deepcopy(t8_linear_qconfig),
        q_scaled=copy.deepcopy(t16f8_qtype),
        attn_scores_raw=copy.deepcopy(
            t16f8_qtype
        ),  # Raw scores might need more range? TBD
        softmax=t16_softmax_qconfig,
        dropout=t16_dropout_qconfig,
        attn_output=copy.deepcopy(t16f8_qtype),
        out_proj=copy.deepcopy(t8_linear_qconfig),
    )
    # Handle biases specifically
    if not ADD_BIAS_Q:
        qconfig5.qlinear.bias = QType()
    if not ADD_BIAS_KV:
        qconfig5.klinear.bias = QType()
        qconfig5.vlinear.bias = QType()
    if not BIAS:
        qconfig5.out_proj.bias = QType()

    print(
        f"Initial QConfig5 (partial T=8 params):\n{qconfig5.model_dump_json(indent=2)}"
    )

    layer5 = FxPMultiheadAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        bias=BIAS,
        add_bias_kv=ADD_BIAS_KV,
        add_bias_q=ADD_BIAS_Q,
        batch_first=True,
        q_config=qconfig5,
    )
    layer5.load_state_dict(base_state_dict)
    layer5.eval()
    print("Layer5 initialized")

    # Calculate fractional bits for T=8 parameters
    layer5.set_no_overflow_quant()
    print(
        f"\nLayer5 QConfig after set_no_overflow_quant:\n{layer5.q_config.model_dump_json(indent=2)}"
    )

    output5, _ = layer5(
        dummy_input, dummy_input, dummy_input, logger=logger, need_weights=False
    )
    log_path5 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_mha_mixed_no_overflow.json")
    logger.save_to_json(log_path5)
    print(f"\nOutput5 shape: {output5.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 6: Mixed No Overflow Calibrated (T=8 Params, T=16 Others)
    # ------------------------------------------------------------------------------
    print(
        "\n" + "=" * 20 + " FxPMHA Scenario 6: Mixed No Overflow Calibrated" + "=" * 20
    )
    logger.clear()

    # Define QConfig: T=8 for params, T=16 for others
    t16_qtype = QType(total_bits=16, q_method=QMethod.ROUND_SATURATE)
    t8_param_qtype = QType(
        total_bits=8, q_method=QMethod.ROUND_SATURATE
    )  # Only T=8 for params

    t8_linear_qconfig = LinearQConfig(
        input=QType(),
        weight=copy.deepcopy(t8_param_qtype),
        bias=copy.deepcopy(t8_param_qtype)
        if BIAS or ADD_BIAS_KV or ADD_BIAS_Q
        else QType(),
        activation=copy.deepcopy(t16_qtype),
    )
    # Reuse T=16 configs for softmax/dropout from Scenario 4
    t16_softmax_qconfig = SoftmaxQConfig(
        input=QType(),
        activation=QType(total_bits=16, q_method=QMethod.ROUND_SATURATE),
    )  # Activation doesnt need more than 1 integer
    t16_dropout_qconfig = DropoutQConfig(
        input=QType(), activation=copy.deepcopy(t16_qtype)
    )

    qconfig6 = MultiheadAttentionQConfig(
        input_query=QType(),
        input_key=QType(),
        input_value=QType(),
        qlinear=copy.deepcopy(t8_linear_qconfig),
        klinear=copy.deepcopy(t8_linear_qconfig),
        vlinear=copy.deepcopy(t8_linear_qconfig),
        q_scaled=copy.deepcopy(t16_qtype),
        attn_scores_raw=copy.deepcopy(
            t16_qtype
        ),  # Raw scores might need more range? TBD
        softmax=t16_softmax_qconfig,
        dropout=t16_dropout_qconfig,
        attn_output=copy.deepcopy(t16_qtype),
        out_proj=copy.deepcopy(t8_linear_qconfig),
    )
    # Handle biases specifically
    if not ADD_BIAS_Q:
        qconfig6.qlinear.bias = QType()
    if not ADD_BIAS_KV:
        qconfig6.klinear.bias = QType()
        qconfig6.vlinear.bias = QType()
    if not BIAS:
        qconfig6.out_proj.bias = QType()

    print(
        f"Initial QConfig6 (partial T=8 params):\n{qconfig6.model_dump_json(indent=2)}"
    )

    layer6 = FxPMultiheadAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        bias=BIAS,
        add_bias_kv=ADD_BIAS_KV,
        add_bias_q=ADD_BIAS_Q,
        batch_first=True,
        q_config=qconfig6,
    )
    layer6.load_state_dict(base_state_dict)
    layer6.eval()
    print("Layer6 initialized")

    # Calculate fractional bits for T=8 parameters
    layer6.set_no_overflow_quant()
    output6, _ = layer6(
        calibration_dummy_input,
        calibration_dummy_input,
        calibration_dummy_input,
        logger=logger,
        need_weights=False,
        calibrate=True,
        calibration_type="no_overflow",
    )
    print(
        f"\nLayer6 QConfig after set_no_overflow_quant and calibration:\n{layer6.q_config.model_dump_json(indent=2)}"
    )

    log_path6 = os.path.join(
        OUTPUTS_PATH_TESTS, "fxp_mha_mixed_no_overflow_calibration.json"
    )
    logger.save_to_json(log_path6)
    print(f"\nOutput6 shape: {output6.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 7: Mixed MinMSE Calibrated (T=8 Params, T=16 Others)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 20 + " FxPMHA Scenario 7: Mixed MinMSE Calibrated" + "=" * 20)
    logger.clear()

    # Define QConfig: T=8 for params, T=16 for others
    t16_qtype = QType(total_bits=16, q_method=QMethod.ROUND_SATURATE)
    t8_param_qtype = QType(
        total_bits=8, q_method=QMethod.ROUND_SATURATE
    )  # Only T=8 for params

    t8_linear_qconfig = LinearQConfig(
        input=QType(),
        weight=copy.deepcopy(t8_param_qtype),
        bias=copy.deepcopy(t8_param_qtype)
        if BIAS or ADD_BIAS_KV or ADD_BIAS_Q
        else QType(),
        activation=copy.deepcopy(t16_qtype),
    )
    # Reuse T=16 configs for softmax/dropout from Scenario 4
    t16_softmax_qconfig = SoftmaxQConfig(
        input=QType(),
        activation=QType(total_bits=16, q_method=QMethod.ROUND_SATURATE),
    )  # Activation doesnt need more than 1 integer
    t16_dropout_qconfig = DropoutQConfig(
        input=QType(), activation=copy.deepcopy(t16_qtype)
    )

    qconfig7 = MultiheadAttentionQConfig(
        input_query=QType(),
        input_key=QType(),
        input_value=QType(),
        qlinear=copy.deepcopy(t8_linear_qconfig),
        klinear=copy.deepcopy(t8_linear_qconfig),
        vlinear=copy.deepcopy(t8_linear_qconfig),
        q_scaled=copy.deepcopy(t16_qtype),
        attn_scores_raw=copy.deepcopy(
            t16_qtype
        ),  # Raw scores might need more range? TBD
        softmax=t16_softmax_qconfig,
        dropout=t16_dropout_qconfig,
        attn_output=copy.deepcopy(t16_qtype),
        out_proj=copy.deepcopy(t8_linear_qconfig),
    )
    # Handle biases specifically
    if not ADD_BIAS_Q:
        qconfig7.qlinear.bias = QType()
    if not ADD_BIAS_KV:
        qconfig7.klinear.bias = QType()
        qconfig7.vlinear.bias = QType()
    if not BIAS:
        qconfig7.out_proj.bias = QType()

    print(
        f"Initial QConfig7 (partial T=8 params):\n{qconfig7.model_dump_json(indent=2)}"
    )

    layer7 = FxPMultiheadAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        bias=BIAS,
        add_bias_kv=ADD_BIAS_KV,
        add_bias_q=ADD_BIAS_Q,
        batch_first=True,
        q_config=qconfig7,
    )
    layer7.load_state_dict(base_state_dict)
    layer7.eval()
    print("Layer7 initialized")

    # Calculate fractional bits for T=8 parameters
    layer7.set_min_mse_quant()
    output7, _ = layer7(
        calibration_dummy_input,
        calibration_dummy_input,
        calibration_dummy_input,
        logger=logger,
        need_weights=False,
        calibrate=True,
        calibration_type="min_mse",
    )
    print(
        f"\nLayer7 QConfig after set_min_mse_quant and min_mse calibration:\n{layer7.q_config.model_dump_json(indent=2)}"
    )

    log_path7 = os.path.join(
        OUTPUTS_PATH_TESTS, "fxp_mha_mixed_min_mse_calibration.json"
    )
    logger.save_to_json(log_path7)
    print(f"\nOutput7 shape: {output7.shape}")

    print("\n" + "=" * 20 + " FxPMultiheadAttention Testing Complete " + "=" * 20)


def test_fxp_transformer_encoder(
    d_model: int = 16,
    num_heads: int = 2,
    seq_len: int = 8,
    batch_size: int = 4,
    calibration_batch_size: int = 128,
    dim_feedforward: int = 8,
    dropout: float = 0.1,
    att_dropout: float = 0.0,
    output_path_tests: str = "outputs/fxp_transformer_encoder_tests",
) -> None:
    os.makedirs(output_path_tests, exist_ok=True)
    print(
        f"\nFxPTransformerEncoderLayer test outputs will be saved in: {output_path_tests}"
    )

    # Dummy input tensor: [B, T, E]
    dummy_input = torch.ones((batch_size, seq_len, d_model))
    calibration_dummy_input = torch.randn(
        (calibration_batch_size, seq_len, d_model), generator=generator
    )
    print(f"Dummy Input Shape: {dummy_input.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 1: Default float (transparent)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 20 + " Scenario 1: Default Float " + "=" * 20)

    layer1 = FxPTransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        att_dropout=att_dropout,
        batch_first=True,
        q_config=None,  # falls back to transparent
    )
    layer1.eval()
    base_state_dict = layer1.state_dict()
    # Initialize the activation logger (store full tensors for detailed inspection)
    logger = ActivationLogger(enabled=True, store_full_tensors=True, model=layer1)
    output1 = layer1(dummy_input, logger=logger)
    log_path1 = os.path.join(output_path_tests, "fxp_transformer_encoder_float.json")
    logger.save_to_json(log_path1)
    print(f"Output1 shape: {output1.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 2: Explicit float QConfig
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 20 + " Scenario 2: Explicit Float QConfig " + "=" * 20)
    logger.clear()
    float_qconfig = TransformerEncoderLayerQConfig()
    print(f"Explicit Float QConfig:\n{float_qconfig.model_dump_json(indent=2)}")
    layer2 = FxPTransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        att_dropout=att_dropout,
        batch_first=True,
        q_config=float_qconfig,
    )
    layer2.load_state_dict(base_state_dict)
    layer2.eval()
    output2 = layer2(dummy_input, logger=logger)
    log_path2 = os.path.join(output_path_tests, "fxp_transformer_encoder_float2.json")
    logger.save_to_json(log_path2)
    print(f"Output2 shape: {output2.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 3: High‐precision quantization (all submodules)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 20 + " Scenario 3: High Precision Config " + "=" * 20)
    logger.clear()
    hp_qconfig = TransformerEncoderLayerQConfig()
    layer3 = FxPTransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        att_dropout=att_dropout,
        batch_first=True,
        q_config=hp_qconfig,
    )
    layer3.load_state_dict(base_state_dict)
    layer3.eval()
    layer3.set_high_precision_quant()
    print(
        f"Layer3 QConfig after set_high_precision_quant:\n"
        f"{layer3.q_config.model_dump_json(indent=2)}"
    )
    output3 = layer3(dummy_input, logger=logger)
    log_path3 = os.path.join(
        output_path_tests, "fxp_transformer_encoder_high_precision.json"
    )
    logger.save_to_json(log_path3)
    print(f"Output3 shape: {output3.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 4: No‐overflow quantization (T=16 bits for weights)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 20 + " Scenario 4: No Overflow (T=16 Params) " + "=" * 20)
    logger.clear()

    # 16‐bit saturating for all linear weights/biases (self‐attn & feedforward)
    t16 = QType(total_bits=16, q_method=QMethod.ROUND_SATURATE)
    t16_linear_qconfig = LinearQConfig(
        weight=copy.deepcopy(t16), bias=copy.deepcopy(t16)
    )
    t16_mha_qconfig = MultiheadAttentionQConfig(
        qlinear=copy.deepcopy(
            t16_linear_qconfig
        ),  # Use deepcopy to avoid aliasing issues if modifying later
        klinear=copy.deepcopy(t16_linear_qconfig),
        vlinear=copy.deepcopy(t16_linear_qconfig),
        out_proj=copy.deepcopy(t16_linear_qconfig),
    )
    t16_layernorm_qconfig = LayerNormQConfig(
        weight=copy.deepcopy(t16), bias=copy.deepcopy(t16)
    )
    noov_qconfig = TransformerEncoderLayerQConfig(
        norm1=copy.deepcopy(t16_layernorm_qconfig),
        self_attn=copy.deepcopy(t16_mha_qconfig),
        norm2=copy.deepcopy(t16_layernorm_qconfig),
        linear1=copy.deepcopy(t16_linear_qconfig),
        linear2=copy.deepcopy(t16_linear_qconfig),
    )
    print(f"Initial QConfig4 (partial):\n{noov_qconfig.model_dump_json(indent=2)}")
    layer4 = FxPTransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        att_dropout=att_dropout,
        batch_first=True,
        q_config=noov_qconfig,
    )
    layer4.load_state_dict(base_state_dict)
    layer4.eval()
    layer4.set_no_overflow_quant()
    print(
        f"Layer4 QConfig after set_no_overflow_quant:\n"
        f"{layer4.q_config.model_dump_json(indent=2)}"
    )
    output4 = layer4(dummy_input, logger=logger)
    log_path4 = os.path.join(
        output_path_tests, "fxp_transformer_encoder_t16_no_overflow.json"
    )
    logger.save_to_json(log_path4)
    print(f"Output4 shape: {output4.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 5: Mixed no‐overflow (T=8 bits weights, T=16/F=8 elsewhere)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 20 + " Scenario 5: Mixed No Overflow " + "=" * 20)
    logger.clear()
    t8 = QType(total_bits=8, q_method=QMethod.ROUND_SATURATE)
    t16f8 = QType(total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE)
    # FF linears params & activation
    mixed_linear_qconfig = LinearQConfig(
        input=copy.deepcopy(t16f8),
        weight=copy.deepcopy(t8),
        bias=copy.deepcopy(t8),
        activation=copy.deepcopy(t16f8),
    )
    t16_softmax_qconfig = SoftmaxQConfig(
        input=copy.deepcopy(t16f8),
        activation=QType(
            total_bits=16, fractional_bits=15, q_method=QMethod.ROUND_SATURATE
        ),
    )  # Activation doesnt need more than 1 integer
    t16_dropout_qconfig = DropoutQConfig(
        input=copy.deepcopy(t16f8), activation=copy.deepcopy(t16f8)
    )
    mixed_mha_qconfig = MultiheadAttentionQConfig(
        input_query=copy.deepcopy(t16f8),
        input_key=copy.deepcopy(t16f8),
        input_value=copy.deepcopy(t16f8),
        qlinear=copy.deepcopy(
            mixed_linear_qconfig
        ),  # Use deepcopy to avoid aliasing issues if modifying later
        klinear=copy.deepcopy(mixed_linear_qconfig),
        vlinear=copy.deepcopy(mixed_linear_qconfig),
        q_scaled=copy.deepcopy(t16f8),
        attn_scores_raw=copy.deepcopy(t16f8),  # Raw scores might need more range? TBD
        softmax=t16_softmax_qconfig,
        dropout=t16_dropout_qconfig,
        attn_output=copy.deepcopy(t16f8),
        out_proj=copy.deepcopy(mixed_linear_qconfig),
    )
    mixed_layernorm_qconfig = LayerNormQConfig(
        input=copy.deepcopy(t16f8),
        weight=copy.deepcopy(t8),
        bias=copy.deepcopy(t8),
        mean_tensor=copy.deepcopy(t16f8),
        var_tensor=copy.deepcopy(t16f8),
        input_normalized=copy.deepcopy(t16f8),
        activation=copy.deepcopy(t16f8),
    )
    mix_qconfig = TransformerEncoderLayerQConfig(
        input=copy.deepcopy(t16f8),
        norm1=copy.deepcopy(mixed_layernorm_qconfig),
        self_attn=copy.deepcopy(mixed_mha_qconfig),
        self_attn_dropout=copy.deepcopy(t16_dropout_qconfig),
        residual_1=copy.deepcopy(t16f8),
        norm2=copy.deepcopy(mixed_layernorm_qconfig),
        linear1=copy.deepcopy(mixed_linear_qconfig),
        ff_activation=copy.deepcopy(t16f8),
        dropout1=copy.deepcopy(t16_dropout_qconfig),
        linear2=copy.deepcopy(mixed_linear_qconfig),
        dropout2=copy.deepcopy(t16_dropout_qconfig),
        residual_2=copy.deepcopy(t16f8),
    )
    print(f"Initial QConfig5 (partial):\n{mix_qconfig.model_dump_json(indent=2)}")
    layer5 = FxPTransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        att_dropout=att_dropout,
        batch_first=True,
        q_config=mix_qconfig,
    )
    layer5.load_state_dict(base_state_dict)
    layer5.eval()
    layer5.set_no_overflow_quant()
    print(
        f"Layer5 QConfig after set_no_overflow_quant:\n"
        f"{layer5.q_config.model_dump_json(indent=2)}"
    )
    output5 = layer5(dummy_input, logger=logger)
    log_path5 = os.path.join(
        output_path_tests, "fxp_transformer_encoder_mixed_no_overflow.json"
    )
    logger.save_to_json(log_path5)
    print(f"Output5 shape: {output5.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 6: Mixed no‐overflow Calibrated (T=8 bits weights, T=16 elsewhere)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 20 + " Scenario 6: Mixed No Overflow Calibrated " + "=" * 20)
    logger.clear()
    t8 = QType(total_bits=8, q_method=QMethod.ROUND_SATURATE)
    t16 = QType(total_bits=16, q_method=QMethod.ROUND_SATURATE)
    # FF linears params & activation
    linear_qconfig6 = LinearQConfig(
        input=QType(),
        weight=copy.deepcopy(t8),
        bias=copy.deepcopy(t8),
        activation=copy.deepcopy(t16),
    )
    t16_softmax_qconfig = SoftmaxQConfig(
        input=QType(),
        activation=copy.deepcopy(t16),
    )  # Activation doesnt need more than 1 integer
    t16_dropout_qconfig = DropoutQConfig(input=QType(), activation=copy.deepcopy(t16))
    mha_qconfig6 = MultiheadAttentionQConfig(
        input_query=QType(),
        input_key=QType(),
        input_value=QType(),
        qlinear=copy.deepcopy(
            linear_qconfig6
        ),  # Use deepcopy to avoid aliasing issues if modifying later
        klinear=copy.deepcopy(linear_qconfig6),
        vlinear=copy.deepcopy(linear_qconfig6),
        q_scaled=copy.deepcopy(t16),
        attn_scores_raw=copy.deepcopy(t16),  # Raw scores might need more range? TBD
        softmax=t16_softmax_qconfig,
        dropout=t16_dropout_qconfig,
        attn_output=copy.deepcopy(t16),
        out_proj=copy.deepcopy(linear_qconfig6),
    )
    layernorm_qconfig6 = LayerNormQConfig(
        input=QType(),
        weight=copy.deepcopy(t8),
        bias=copy.deepcopy(t8),
        mean_tensor=copy.deepcopy(t16),
        var_tensor=copy.deepcopy(t16),
        input_normalized=copy.deepcopy(t16),
        activation=copy.deepcopy(t16),
    )
    qconfig6 = TransformerEncoderLayerQConfig(
        input=QType(),
        norm1=copy.deepcopy(layernorm_qconfig6),
        self_attn=copy.deepcopy(mha_qconfig6),
        self_attn_dropout=copy.deepcopy(t16_dropout_qconfig),
        residual_1=copy.deepcopy(t16),
        norm2=copy.deepcopy(layernorm_qconfig6),
        linear1=copy.deepcopy(linear_qconfig6),
        ff_activation=copy.deepcopy(t16),
        dropout1=copy.deepcopy(t16_dropout_qconfig),
        linear2=copy.deepcopy(linear_qconfig6),
        dropout2=copy.deepcopy(t16_dropout_qconfig),
        residual_2=copy.deepcopy(t16),
    )
    print(f"Initial QConfig6 (partial):\n{qconfig6.model_dump_json(indent=2)}")
    layer6 = FxPTransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        att_dropout=att_dropout,
        batch_first=True,
        q_config=qconfig6,
    )
    layer6.load_state_dict(base_state_dict)
    layer6.eval()
    layer6.set_no_overflow_quant()
    output6 = layer6(
        calibration_dummy_input,
        logger=logger,
        calibrate=True,
        calibration_type="no_overflow",
    )
    print(
        f"Layer6 QConfig after set_no_overflow_quant and calibration:\n"
        f"{layer6.q_config.model_dump_json(indent=2)}"
    )
    log_path6 = os.path.join(
        output_path_tests, "fxp_transformer_encoder_mixed_no_overflow_calibration.json"
    )
    logger.save_to_json(log_path6)
    print(f"Output6 shape: {output6.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 7: Mixed MinMSE Calibrated (T=8 bits weights, T=16 elsewhere)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 20 + " Scenario 7: Mixed MinMSE Calibrated " + "=" * 20)
    logger.clear()
    t8 = QType(total_bits=8, q_method=QMethod.ROUND_SATURATE)
    t16 = QType(total_bits=16, q_method=QMethod.ROUND_SATURATE)
    # FF linears params & activation
    linear_qconfig7 = LinearQConfig(
        input=QType(),
        weight=copy.deepcopy(t8),
        bias=copy.deepcopy(t8),
        activation=copy.deepcopy(t16),
    )
    t16_softmax_qconfig = SoftmaxQConfig(
        input=QType(),
        activation=copy.deepcopy(t16),
    )  # Activation doesnt need more than 1 integer
    t16_dropout_qconfig = DropoutQConfig(input=QType(), activation=copy.deepcopy(t16))
    mha_qconfig7 = MultiheadAttentionQConfig(
        input_query=QType(),
        input_key=QType(),
        input_value=QType(),
        qlinear=copy.deepcopy(
            linear_qconfig7
        ),  # Use deepcopy to avoid aliasing issues if modifying later
        klinear=copy.deepcopy(linear_qconfig7),
        vlinear=copy.deepcopy(linear_qconfig7),
        q_scaled=copy.deepcopy(t16),
        attn_scores_raw=copy.deepcopy(t16),  # Raw scores might need more range? TBD
        softmax=t16_softmax_qconfig,
        dropout=t16_dropout_qconfig,
        attn_output=copy.deepcopy(t16),
        out_proj=copy.deepcopy(linear_qconfig7),
    )
    layernorm_qconfig7 = LayerNormQConfig(
        input=QType(),
        weight=copy.deepcopy(t8),
        bias=copy.deepcopy(t8),
        mean_tensor=copy.deepcopy(t16),
        var_tensor=copy.deepcopy(t16),
        input_normalized=copy.deepcopy(t16),
        activation=copy.deepcopy(t16),
    )
    qconfig7 = TransformerEncoderLayerQConfig(
        input=QType(),
        norm1=copy.deepcopy(layernorm_qconfig7),
        self_attn=copy.deepcopy(mha_qconfig7),
        self_attn_dropout=copy.deepcopy(t16_dropout_qconfig),
        residual_1=copy.deepcopy(t16),
        norm2=copy.deepcopy(layernorm_qconfig7),
        linear1=copy.deepcopy(linear_qconfig7),
        ff_activation=copy.deepcopy(t16),
        dropout1=copy.deepcopy(t16_dropout_qconfig),
        linear2=copy.deepcopy(linear_qconfig7),
        dropout2=copy.deepcopy(t16_dropout_qconfig),
        residual_2=copy.deepcopy(t16),
    )
    print(f"Initial QConfig7 (partial):\n{qconfig7.model_dump_json(indent=2)}")
    layer7 = FxPTransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        att_dropout=att_dropout,
        batch_first=True,
        q_config=qconfig7,
    )
    layer7.load_state_dict(base_state_dict)
    layer7.eval()
    layer7.set_min_mse_quant()
    output7 = layer7(
        calibration_dummy_input,
        logger=logger,
        calibrate=True,
        calibration_type="min_mse",
    )
    print(
        f"Layer7 QConfig after set_min_mse_quant and min_mse calibration:\n"
        f"{layer7.q_config.model_dump_json(indent=2)}"
    )
    log_path7 = os.path.join(
        output_path_tests,
        "fxp_transformer_encoder_mixed_min_mse_calibration.json",
    )
    logger.save_to_json(log_path7)
    print(f"Output7 shape: {output7.shape}")

    print("\n" + "=" * 20 + " FxPTransformerEncoderLayer Testing Complete " + "=" * 20)


def test_fxp_conv2d(
    in_channels: int = 10,
    out_channels: int = 5,
    kernel_size: _size_2_t = 3,
    stride: _size_2_t = 1,
    padding: Union[str, _size_2_t] = 0,
    dilation: _size_2_t = 1,
    groups: int = 1,
    bias: bool = True,
    padding_mode: str = "zeros",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    batch_size: int = 1,
    calibration_batch_size: int = 128,
    output_path_tests: str = "outputs/fxp_conv2d_tests",
) -> None:
    # ------------------------------------------------------------------------------
    # Configuration & Initialization
    # ------------------------------------------------------------------------------
    IN_CHANNELS = in_channels
    OUT_CHANNELS = out_channels
    KERNEL_SIZE = kernel_size
    STRIDE = stride
    PADDING = padding
    DILATION = dilation
    GROUPS = groups
    BIAS = bias
    PADDING_MODE = padding_mode
    DEVICE = device
    DTYPE = dtype
    BATCH_SIZE = batch_size
    OUTPUTS_PATH_TESTS = output_path_tests  # Folder for test outputs
    CALIBRATION_BATCH_SIZE = calibration_batch_size

    # Ensure the output directory exists
    os.makedirs(OUTPUTS_PATH_TESTS, exist_ok=True)
    print(f"FxPConv2d test outputs will be saved in: {OUTPUTS_PATH_TESTS}")

    # Create a dummy input tensor (for a conv2d layer)
    data_height = 5
    data_width = 5
    dummy_input = torch.ones((BATCH_SIZE, IN_CHANNELS, data_height, data_width))
    calibration_dummy_input = torch.randn(
        (CALIBRATION_BATCH_SIZE, IN_CHANNELS, data_height, data_width),
        generator=generator,
    )
    print(f"\nDummy Input Shape: {dummy_input.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 1: Default Float (No explicit QConfig)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 20 + " FxPConv2d Scenario 1: Default Float " + "=" * 20)
    # Create a layer with no explicit QConfig; it uses the default float behavior.
    layer1 = FxPConv2D(
        IN_CHANNELS,
        OUT_CHANNELS,
        KERNEL_SIZE,
        STRIDE,
        PADDING,
        DILATION,
        GROUPS,
        BIAS,
        PADDING_MODE,
        DEVICE,
        DTYPE,
    )
    layer1.eval()  # Set to evaluation mode
    base_state_dict = layer1.state_dict()  # Store state for later loading
    print("Layer1 initialized (default float)")
    # Initialize the activation logger (store full tensors for detailed inspection)
    logger = ActivationLogger(enabled=True, store_full_tensors=True, model=layer1)

    # Run inference
    output1 = layer1.forward(dummy_input, logger)

    # Save activation logs
    log_path1 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_conv2d_float.json")
    logger.save_to_json(log_path1)

    print(f"Output1 shape: {output1.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 2: Explicit Float QConfig
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 20 + " FxPConv2d Scenario 2: Explicit Float QConfig " + "=" * 20)
    logger.clear()

    # Create an explicit QConfig (defaults are float)
    float_qconfig = Conv2DQConfig()
    print(f"Explicit Float QConfig:\n{float_qconfig.model_dump_json(indent=2)}")

    # Instantiate a layer using the explicit float QConfig and load the baseline state_dict
    layer2 = FxPConv2D(
        IN_CHANNELS,
        OUT_CHANNELS,
        KERNEL_SIZE,
        STRIDE,
        PADDING,
        DILATION,
        GROUPS,
        BIAS,
        PADDING_MODE,
        DEVICE,
        DTYPE,
        float_qconfig,
    )
    layer2.load_state_dict(base_state_dict)
    layer2.eval()
    print("Layer2 initialized (explicit float config)")

    # Run inference
    output2 = layer2(dummy_input, logger=logger)

    # Save activation logs
    log_path2 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_conv2d_float2.json")
    logger.save_to_json(log_path2)

    print(f"Output2 shape: {output2.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 3: High Precision Configuration ("High Precision" Params)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 20 + " FxPConv2d Scenario 3: High Precision Config " + "=" * 20)
    logger.clear()

    # Create a new layer with a fresh QConfig for high precision
    layer3_qconfig = Conv2DQConfig()  # Start with default
    layer3 = FxPConv2D(
        IN_CHANNELS,
        OUT_CHANNELS,
        KERNEL_SIZE,
        STRIDE,
        PADDING,
        DILATION,
        GROUPS,
        BIAS,
        PADDING_MODE,
        DEVICE,
        DTYPE,
        layer3_qconfig,
    )
    layer3.load_state_dict(base_state_dict)
    layer3.eval()
    print("Layer3 initialized")

    # Configure layer3 for high precision quantization
    layer3.set_high_precision_quant()
    print(
        f"\nLayer3 QConfig after set_high_precision_quant:\n{layer3.q_config.model_dump_json(indent=2)}"
    )

    # Run inference
    output3 = layer3(dummy_input, logger=logger)

    # Save activation logs
    log_path3 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_conv2d_high_precision.json")
    logger.save_to_json(log_path3)

    print(f"\nOutput3 shape: {output3.shape}")
    print(f"Output3:\n{output3}")

    # ------------------------------------------------------------------------------
    # Scenario 4: No Overflow (Fixed Total Bits = 16 for Params)
    # ------------------------------------------------------------------------------
    print(
        "\n" + "=" * 20 + " FxPConv2d Scenario 4: No Overflow (T=16 Params) " + "=" * 20
    )
    logger.clear()

    # Create a QConfig specifying only total_bits for weight and bias
    qconfig4 = Conv2DQConfig(
        weight=QType(total_bits=16, q_method=QMethod.ROUND_SATURATE),
        bias=QType(total_bits=16, q_method=QMethod.ROUND_SATURATE) if BIAS else QType(),
        # Input/Activation remain float unless specified
    )
    print(f"Initial QConfig4 (partial):\n{qconfig4.model_dump_json(indent=2)}")

    # Instantiate and initialize the layer with qconfig4
    layer4 = FxPConv2D(
        IN_CHANNELS,
        OUT_CHANNELS,
        KERNEL_SIZE,
        STRIDE,
        PADDING,
        DILATION,
        GROUPS,
        BIAS,
        PADDING_MODE,
        DEVICE,
        DTYPE,
        qconfig4,
    )
    layer4.load_state_dict(base_state_dict)
    layer4.eval()
    print("Layer4 initialized")

    # Calculate and set the missing fractional bits to avoid overflow
    layer4.set_no_overflow_quant()
    print(
        f"\nLayer4 QConfig after set_no_overflow_quant:\n{layer4.q_config.model_dump_json(indent=2)}"
    )

    # Run inference
    output4 = layer4(dummy_input, logger=logger)

    # Save activation logs
    log_path4 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_conv2d_t16_no_overflow.json")
    logger.save_to_json(log_path4)

    print(f"\nOutput4 shape: {output4.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 5: Mixed No Overflow (T=8 Params, T=16/F=8 Activation)
    # ------------------------------------------------------------------------------
    print(
        "\n"
        + "=" * 20
        + " FxPConv2d Scenario 5: Mixed No Overflow (T=8 Params, 16/8 Act) "
        + "=" * 20
    )
    logger.clear()

    # Create a QConfig with explicit settings for input, weight, bias, and activation
    qconfig5 = Conv2DQConfig(
        input=QType(total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE),
        weight=QType(total_bits=8, q_method=QMethod.ROUND_SATURATE),
        bias=QType(total_bits=8, q_method=QMethod.ROUND_SATURATE) if BIAS else QType(),
        activation=QType(
            total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE
        ),
    )
    layer5 = FxPConv2D(
        IN_CHANNELS,
        OUT_CHANNELS,
        KERNEL_SIZE,
        STRIDE,
        PADDING,
        DILATION,
        GROUPS,
        BIAS,
        PADDING_MODE,
        DEVICE,
        DTYPE,
        qconfig5,
    )
    layer5.load_state_dict(base_state_dict)
    layer5.eval()
    print("Layer5 initialized")
    print(
        f"Initial QConfig5 (partial weights/bias, fixed activation):\n{qconfig5.model_dump_json(indent=2)}"
    )

    # Set no overflow parameters for weights and bias (activation remains unchanged)
    layer5.set_no_overflow_quant()
    print(
        f"\nLayer5 QConfig after set_no_overflow_quant:\n{layer5.q_config.model_dump_json(indent=2)}"
    )

    # Run inference
    output5 = layer5(dummy_input, logger=logger)

    # Save activation logs
    log_path5 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_conv2d_mixed_no_overflow.json")
    logger.save_to_json(log_path5)

    print(f"\nOutput5 shape: {output5.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 6: Mixed No Overflow Calibrated (T=8 Params, T=16 Activation)
    # ------------------------------------------------------------------------------
    print(
        "\n"
        + "=" * 20
        + " FxPConv2d Scenario 6: Mixed No Overflow Calibrated (T=8 Params, 16 Act) "
        + "=" * 20
    )
    logger.clear()

    # Create a QConfig with explicit settings for input, weight, bias, and activation
    qconfig6 = Conv2DQConfig(
        input=QType(total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE),
        weight=QType(total_bits=8, q_method=QMethod.ROUND_SATURATE),
        bias=QType(total_bits=8, q_method=QMethod.ROUND_SATURATE) if BIAS else QType(),
        activation=QType(total_bits=16, q_method=QMethod.ROUND_SATURATE),
    )
    layer6 = FxPConv2D(
        IN_CHANNELS,
        OUT_CHANNELS,
        KERNEL_SIZE,
        STRIDE,
        PADDING,
        DILATION,
        GROUPS,
        BIAS,
        PADDING_MODE,
        DEVICE,
        DTYPE,
        qconfig6,
    )
    layer6.load_state_dict(base_state_dict)
    layer6.eval()
    print("Layer6 initialized")
    print(
        f"Initial QConfig6 (partial weights/bias, fixed activation):\n{qconfig6.model_dump_json(indent=2)}"
    )

    # Set no overflow parameters for weights and bias (activation remains unchanged)
    layer6.set_no_overflow_quant()
    # Run inference
    output6 = layer6(
        calibration_dummy_input,
        logger=logger,
        calibrate=True,
        calibration_type="no_overflow",
    )
    print(
        f"\nLayer6 QConfig after set_no_overflow_quant and calibration:\n{layer6.q_config.model_dump_json(indent=2)}"
    )

    # Save activation logs
    log_path6 = os.path.join(
        OUTPUTS_PATH_TESTS, "fxp_conv2d_mixed_no_overflow_calibration.json"
    )
    logger.save_to_json(log_path6)

    print(f"\nOutput6 shape: {output6.shape}")

    # ------------------------------------------------------------------------------
    # Scenario 7: Mixed Min MSE Calibrated (T=8 Params, T=16 Activation)
    # ------------------------------------------------------------------------------
    print(
        "\n"
        + "=" * 20
        + " FxPConv2d Scenario 7: Mixed Min MSE Calibrated (T=8 Params, 16 Act) "
        + "=" * 20
    )
    logger.clear()

    # Create a QConfig with explicit settings for input, weight, bias, and activation
    qconfig7 = Conv2DQConfig(
        input=QType(total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE),
        weight=QType(total_bits=8, q_method=QMethod.ROUND_SATURATE),
        bias=QType(total_bits=8, q_method=QMethod.ROUND_SATURATE) if BIAS else QType(),
        activation=QType(total_bits=16, q_method=QMethod.ROUND_SATURATE),
    )
    layer7 = FxPConv2D(
        IN_CHANNELS,
        OUT_CHANNELS,
        KERNEL_SIZE,
        STRIDE,
        PADDING,
        DILATION,
        GROUPS,
        BIAS,
        PADDING_MODE,
        DEVICE,
        DTYPE,
        qconfig7,
    )
    layer7.load_state_dict(base_state_dict)
    layer7.eval()
    print("Layer7 initialized")
    print(
        f"Initial QConfig7 (partial weights/bias, fixed activation):\n{qconfig7.model_dump_json(indent=2)}"
    )

    # Set no overflow parameters for weights and bias (activation remains unchanged)
    layer7.set_min_mse_quant()
    # Run inference
    output7 = layer7(
        calibration_dummy_input,
        logger=logger,
        calibrate=True,
        calibration_type="min_mse",
    )
    print(
        f"\nLayer7 QConfig after set_min_mse_quant and min_mse calibration:\n{layer7.q_config.model_dump_json(indent=2)}"
    )

    # Save activation logs
    log_path7 = os.path.join(
        OUTPUTS_PATH_TESTS, "fxp_conv2d_mixed_min_mse_calibration.json"
    )
    logger.save_to_json(log_path7)

    print(f"\nOutput7 shape: {output7.shape}")

    # ------------------------------------------------------------------------------
    # Completion Message
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 20 + " FxPConv2d Testing Complete " + "=" * 20)


def test_fxp_sigmoid(
    features: int = 10,
    batch_size: int = 4,
    calibration_batch_size: int = 128,
    output_path_tests: str = "outputs/fxp_sigmoid_tests",
) -> None:
    """
    Mirrors test_fxp_softmax structure, adapted for Sigmoid.
    """
    FEATURES = features
    BATCH_SIZE = batch_size
    CALIBRATION_BATCH_SIZE = calibration_batch_size
    OUTPUTS_PATH_TESTS = output_path_tests

    os.makedirs(OUTPUTS_PATH_TESTS, exist_ok=True)
    print(f"\nFxPSigmoid test outputs will be saved in: {OUTPUTS_PATH_TESTS}")

    dummy_input = torch.randn((BATCH_SIZE, FEATURES), generator=generator)
    calibration_dummy_input = torch.randn(
        (CALIBRATION_BATCH_SIZE, FEATURES), generator=generator
    )
    print(f"\nDummy Input Shape: {dummy_input.shape}")

    # 1) Default Float
    print("\n" + "=" * 20 + " FxPSigmoid Scenario 1: Default Float " + "=" * 20)
    layer1 = FxPSigmoid()
    layer1.eval()
    logger = ActivationLogger(enabled=True, store_full_tensors=True, model=layer1)
    output1 = layer1(dummy_input, logger=logger)
    log_path1 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_sigmoid_float.json")
    logger.save_to_json(log_path1)
    print(f"Output1 shape: {output1.shape}")

    # 2) Explicit Float QConfig
    print(
        "\n" + "=" * 20 + " FxPSigmoid Scenario 2: Explicit Float QConfig " + "=" * 20
    )
    logger.clear()
    float_qconfig = SigmoidQConfig()
    print(f"Explicit Float QConfig:\n{float_qconfig.model_dump_json(indent=2)}")
    layer2 = FxPSigmoid(q_config=float_qconfig)
    layer2.eval()
    output2 = layer2(dummy_input, logger=logger)
    log_path2 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_sigmoid_float2.json")
    logger.save_to_json(log_path2)
    print(f"Output2 shape: {output2.shape}")

    # 3) High Precision (Acts)
    print(
        "\n"
        + "=" * 20
        + " FxPSigmoid Scenario 3: High Precision Config (Acts) "
        + "=" * 20
    )
    logger.clear()
    layer3_qconfig = SigmoidQConfig(
        input=QType(total_bits=32, fractional_bits=24, q_method=QMethod.ROUND_SATURATE),
        activation=QType(
            total_bits=32, fractional_bits=30, q_method=QMethod.ROUND_SATURATE
        ),  # [0,1]
    )
    layer3 = FxPSigmoid(q_config=layer3_qconfig)
    layer3.eval()
    print(f"\nLayer3 QConfig:\n{layer3.q_config.model_dump_json(indent=2)}")
    output3 = layer3(dummy_input, logger=logger)
    log_path3 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_sigmoid_high_precision.json")
    logger.save_to_json(log_path3)
    print(f"Output3 shape: {output3.shape}")

    # 4) No Overflow (T=16 Acts)
    print(
        "\n" + "=" * 20 + " FxPSigmoid Scenario 4: No Overflow (T=16 Acts) " + "=" * 20
    )
    logger.clear()
    qconfig4 = SigmoidQConfig(
        input=QType(total_bits=16, fractional_bits=10, q_method=QMethod.ROUND_SATURATE),
        activation=QType(
            total_bits=16, fractional_bits=15, q_method=QMethod.ROUND_SATURATE
        ),
    )
    print(f"Initial QConfig4:\n{qconfig4.model_dump_json(indent=2)}")
    layer4 = FxPSigmoid(q_config=qconfig4)
    layer4.eval()
    output4 = layer4(dummy_input, logger=logger)
    log_path4 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_sigmoid_t16_no_overflow.json")
    logger.save_to_json(log_path4)
    print(f"Output4 shape: {output4.shape}")

    # 5) Mixed No Overflow (T=16/F=10 Input, T=8/F=7 Activation)
    print(
        "\n" + "=" * 20 + " FxPSigmoid Scenario 5: Mixed No Overflow (Acts) " + "=" * 20
    )
    logger.clear()
    qconfig5 = SigmoidQConfig(
        input=QType(total_bits=16, fractional_bits=10, q_method=QMethod.ROUND_SATURATE),
        activation=QType(
            total_bits=8, fractional_bits=7, q_method=QMethod.ROUND_SATURATE
        ),
    )
    print(f"QConfig5:\n{qconfig5.model_dump_json(indent=2)}")
    layer5 = FxPSigmoid(q_config=qconfig5)
    layer5.eval()
    output5 = layer5(dummy_input, logger=logger)
    log_path5 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_sigmoid_mixed_no_overflow.json")
    logger.save_to_json(log_path5)
    print(f"Output5 shape: {output5.shape}")

    # 6) Mixed No Overflow Calibrated
    print(
        "\n"
        + "=" * 20
        + " FxPSigmoid Scenario 6: Mixed No Overflow Calibrated (Acts) "
        + "=" * 20
    )
    logger.clear()
    qconfig6 = SigmoidQConfig(
        input=QType(total_bits=16, fractional_bits=10, q_method=QMethod.ROUND_SATURATE),
        activation=QType(total_bits=8, q_method=QMethod.ROUND_SATURATE),
    )
    layer6 = FxPSigmoid(q_config=qconfig6)
    layer6.eval()
    print(f"QConfig6:\n{qconfig6.model_dump_json(indent=2)}")
    output6 = layer6(
        calibration_dummy_input,
        logger=logger,
        calibrate=True,
        calibration_type="no_overflow",
    )
    print(
        f"\nLayer6 QConfig after calibration:\n{layer6.q_config.model_dump_json(indent=2)}"
    )
    log_path6 = os.path.join(
        OUTPUTS_PATH_TESTS, "fxp_sigmoid_mixed_no_overflow_calibration.json"
    )
    logger.save_to_json(log_path6)
    print(f"Output6 shape: {output6.shape}")

    # 7) Mixed MinMSE Calibrated
    print(
        "\n"
        + "=" * 20
        + " FxPSigmoid Scenario 7: Mixed MinMSE Calibrated (Acts) "
        + "=" * 20
    )
    logger.clear()
    qconfig7 = SigmoidQConfig(
        input=QType(total_bits=16, fractional_bits=10, q_method=QMethod.ROUND_SATURATE),
        activation=QType(total_bits=8, q_method=QMethod.ROUND_SATURATE),
    )
    layer7 = FxPSigmoid(q_config=qconfig7)
    layer7.eval()
    print(f"QConfig7:\n{qconfig7.model_dump_json(indent=2)}")
    output7 = layer7(
        calibration_dummy_input,
        logger=logger,
        calibrate=True,
        calibration_type="min_mse",
    )
    print(
        f"\nLayer7 QConfig after calibration:\n{layer7.q_config.model_dump_json(indent=2)}"
    )
    log_path7 = os.path.join(
        OUTPUTS_PATH_TESTS, "fxp_sigmoid_mixed_min_mse_calibration.json"
    )
    logger.save_to_json(log_path7)
    print(f"Output7 shape: {output7.shape}")

    print("\n" + "=" * 20 + " FxPSigmoid Testing Complete " + "=" * 20)


def test_fxp_tanh(
    features: int = 10,
    batch_size: int = 4,
    calibration_batch_size: int = 128,
    output_path_tests: str = "outputs/fxp_tanh_tests",
) -> None:
    """
    Mirrors test_fxp_softmax structure, adapted for Tanh (range [-1, 1]).
    """
    FEATURES = features
    BATCH_SIZE = batch_size
    CALIBRATION_BATCH_SIZE = calibration_batch_size
    OUTPUTS_PATH_TESTS = output_path_tests

    os.makedirs(OUTPUTS_PATH_TESTS, exist_ok=True)
    print(f"\nFxPTanh test outputs will be saved in: {OUTPUTS_PATH_TESTS}")

    dummy_input = torch.randn((BATCH_SIZE, FEATURES), generator=generator)
    calibration_dummy_input = torch.randn(
        (CALIBRATION_BATCH_SIZE, FEATURES), generator=generator
    )
    print(f"\nDummy Input Shape: {dummy_input.shape}")

    # 1) Default Float
    print("\n" + "=" * 20 + " FxPTanh Scenario 1: Default Float " + "=" * 20)
    layer1 = FxPTanh()
    layer1.eval()
    logger = ActivationLogger(enabled=True, store_full_tensors=True, model=layer1)
    output1 = layer1(dummy_input, logger=logger)
    log_path1 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_tanh_float.json")
    logger.save_to_json(log_path1)
    print(f"Output1 shape: {output1.shape}")

    # 2) Explicit Float QConfig
    print("\n" + "=" * 20 + " FxPTanh Scenario 2: Explicit Float QConfig " + "=" * 20)
    logger.clear()
    float_qconfig = TanhQConfig()
    print(f"Explicit Float QConfig:\n{float_qconfig.model_dump_json(indent=2)}")
    layer2 = FxPTanh(q_config=float_qconfig)
    layer2.eval()
    output2 = layer2(dummy_input, logger=logger)
    log_path2 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_tanh_float2.json")
    logger.save_to_json(log_path2)
    print(f"Output2 shape: {output2.shape}")

    # 3) High Precision (Acts)
    print(
        "\n"
        + "=" * 20
        + " FxPTanh Scenario 3: High Precision Config (Acts) "
        + "=" * 20
    )
    logger.clear()
    layer3_qconfig = TanhQConfig(
        input=QType(total_bits=32, fractional_bits=24, q_method=QMethod.ROUND_SATURATE),
        activation=QType(
            total_bits=32, fractional_bits=30, q_method=QMethod.ROUND_SATURATE
        ),  # [-1,1]
    )
    layer3 = FxPTanh(q_config=layer3_qconfig)
    layer3.eval()
    print(f"\nLayer3 QConfig:\n{layer3.q_config.model_dump_json(indent=2)}")
    output3 = layer3(dummy_input, logger=logger)
    log_path3 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_tanh_high_precision.json")
    logger.save_to_json(log_path3)
    print(f"Output3 shape: {output3.shape}")

    # 4) No Overflow (T=16 Acts)
    print("\n" + "=" * 20 + " FxPTanh Scenario 4: No Overflow (T=16 Acts) " + "=" * 20)
    logger.clear()
    qconfig4 = TanhQConfig(
        input=QType(total_bits=16, fractional_bits=10, q_method=QMethod.ROUND_SATURATE),
        activation=QType(
            total_bits=16, fractional_bits=14, q_method=QMethod.ROUND_SATURATE
        ),  # allow 2 ints
    )
    print(f"Initial QConfig4:\n{qconfig4.model_dump_json(indent=2)}")
    layer4 = FxPTanh(q_config=qconfig4)
    layer4.eval()
    output4 = layer4(dummy_input, logger=logger)
    log_path4 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_tanh_t16_no_overflow.json")
    logger.save_to_json(log_path4)
    print(f"Output4 shape: {output4.shape}")

    # 5) Mixed No Overflow
    print("\n" + "=" * 20 + " FxPTanh Scenario 5: Mixed No Overflow (Acts) " + "=" * 20)
    logger.clear()
    qconfig5 = TanhQConfig(
        input=QType(total_bits=16, fractional_bits=10, q_method=QMethod.ROUND_SATURATE),
        activation=QType(
            total_bits=8, fractional_bits=6, q_method=QMethod.ROUND_SATURATE
        ),  # [-1,1]
    )
    layer5 = FxPTanh(q_config=qconfig5)
    layer5.eval()
    print(f"QConfig5:\n{qconfig5.model_dump_json(indent=2)}")
    output5 = layer5(dummy_input, logger=logger)
    log_path5 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_tanh_mixed_no_overflow.json")
    logger.save_to_json(log_path5)
    print(f"Output5 shape: {output5.shape}")

    # 6) Mixed No Overflow Calibrated
    print(
        "\n"
        + "=" * 20
        + " FxPTanh Scenario 6: Mixed No Overflow Calibrated (Acts) "
        + "=" * 20
    )
    logger.clear()
    qconfig6 = TanhQConfig(
        input=QType(total_bits=16, fractional_bits=10, q_method=QMethod.ROUND_SATURATE),
        activation=QType(total_bits=8, q_method=QMethod.ROUND_SATURATE),
    )
    layer6 = FxPTanh(q_config=qconfig6)
    layer6.eval()
    print(f"QConfig6:\n{qconfig6.model_dump_json(indent=2)}")
    output6 = layer6(
        calibration_dummy_input,
        logger=logger,
        calibrate=True,
        calibration_type="no_overflow",
    )
    print(
        f"\nLayer6 QConfig after calibration:\n{layer6.q_config.model_dump_json(indent=2)}"
    )
    log_path6 = os.path.join(
        OUTPUTS_PATH_TESTS, "fxp_tanh_mixed_no_overflow_calibration.json"
    )
    logger.save_to_json(log_path6)
    print(f"Output6 shape: {output6.shape}")

    # 7) Mixed MinMSE Calibrated
    print(
        "\n"
        + "=" * 20
        + " FxPTanh Scenario 7: Mixed MinMSE Calibrated (Acts) "
        + "=" * 20
    )
    logger.clear()
    qconfig7 = TanhQConfig(
        input=QType(total_bits=16, fractional_bits=10, q_method=QMethod.ROUND_SATURATE),
        activation=QType(total_bits=8, q_method=QMethod.ROUND_SATURATE),
    )
    layer7 = FxPTanh(q_config=qconfig7)
    layer7.eval()
    print(f"QConfig7:\n{qconfig7.model_dump_json(indent=2)}")
    output7 = layer7(
        calibration_dummy_input,
        logger=logger,
        calibrate=True,
        calibration_type="min_mse",
    )
    print(
        f"\nLayer7 QConfig after calibration:\n{layer7.q_config.model_dump_json(indent=2)}"
    )
    log_path7 = os.path.join(
        OUTPUTS_PATH_TESTS, "fxp_tanh_mixed_min_mse_calibration.json"
    )
    logger.save_to_json(log_path7)
    print(f"Output7 shape: {output7.shape}")

    print("\n" + "=" * 20 + " FxPTanh Testing Complete " + "=" * 20)


def test_fxp_lstm(
    input_size: int = 8,
    hidden_size: int = 12,
    seq_len: int = 6,
    batch_size: int = 4,
    calibration_batch_size: int = 64,
    dropout: float = 0.0,
    bias: bool = True,
    output_path_tests: str = "outputs/fxp_lstm_tests",
) -> None:
    """
    Mirrors test_fxp_multiheadattention structure, adapted for LSTM.
    Single-layer, unidirectional, batch_first=True (as in the module).
    """
    BATCH_SIZE = batch_size
    SEQ_LEN = seq_len
    INPUT_SIZE = input_size
    HIDDEN_SIZE = hidden_size
    CALIBRATION_BATCH_SIZE = calibration_batch_size
    OUTPUTS_PATH_TESTS = output_path_tests
    BIAS = bias

    os.makedirs(OUTPUTS_PATH_TESTS, exist_ok=True)
    print(f"\nFxPLSTM test outputs will be saved in: {OUTPUTS_PATH_TESTS}")

    dummy_input = torch.randn((BATCH_SIZE, SEQ_LEN, INPUT_SIZE), generator=generator)
    calibration_dummy_input = torch.randn(
        (CALIBRATION_BATCH_SIZE, SEQ_LEN, INPUT_SIZE), generator=generator
    )
    print(f"\nDummy Input Shape: {dummy_input.shape}")

    # 1) Default Float (no explicit qconfig)
    print("\n" + "=" * 20 + " FxPLSTM Scenario 1: Default Float " + "=" * 20)
    layer1 = FxPLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        batch_first=True,
        bias=BIAS,
        dropout=dropout,
        q_config=None,
    )
    layer1.eval()
    base_state_dict = layer1.state_dict()
    logger = ActivationLogger(enabled=True, store_full_tensors=True, model=layer1)
    out1, (hn1, cn1) = layer1(dummy_input, logger=logger)
    log_path1 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_lstm_float.json")
    logger.save_to_json(log_path1)
    print(f"Output1 shape: {out1.shape}, hn: {hn1.shape}, cn: {cn1.shape}")

    # 2) Explicit Float QConfig
    print("\n" + "=" * 20 + " FxPLSTM Scenario 2: Explicit Float QConfig " + "=" * 20)
    logger.clear()
    float_qconfig = LSTMQConfig()
    print(f"Explicit Float QConfig:\n{float_qconfig.model_dump_json(indent=2)}")
    layer2 = FxPLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        batch_first=True,
        bias=BIAS,
        dropout=dropout,
        q_config=float_qconfig,
    )
    layer2.load_state_dict(base_state_dict)
    layer2.eval()
    out2, (hn2, cn2) = layer2(dummy_input, logger=logger)
    log_path2 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_lstm_float2.json")
    logger.save_to_json(log_path2)
    print(f"Output2 shape: {out2.shape}, hn: {hn2.shape}, cn: {cn2.shape}")

    # 3) High Precision (Params)
    print("\n" + "=" * 20 + " FxPLSTM Scenario 3: High Precision Config " + "=" * 20)
    logger.clear()
    layer3_qconfig = LSTMQConfig()  # start default, then call helper
    layer3 = FxPLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        batch_first=True,
        bias=BIAS,
        dropout=dropout,
        q_config=layer3_qconfig,
    )
    layer3.load_state_dict(base_state_dict)
    layer3.eval()
    layer3.set_high_precision_quant(same_wb=True, same_fc=True)
    print(
        f"\nLayer3 QConfig after set_high_precision_quant:\n{layer3.q_config.model_dump_json(indent=2)}"
    )
    out3, (hn3, cn3) = layer3(dummy_input, logger=logger)
    log_path3 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_lstm_high_precision.json")
    logger.save_to_json(log_path3)
    print(f"Output3 shape: {out3.shape}, hn: {hn3.shape}, cn: {cn3.shape}")

    # 4) No Overflow (T=16 Params)
    print(
        "\n" + "=" * 20 + " FxPLSTM Scenario 4: No Overflow (T=16 Params) " + "=" * 20
    )
    logger.clear()
    t16_param = QType(total_bits=16, q_method=QMethod.ROUND_SATURATE)
    t16_linear_cfg = LinearQConfig(
        weight=t16_param, bias=t16_param if BIAS else QType()
    )
    qconfig4 = LSTMQConfig(
        w_ih=copy.deepcopy(t16_linear_cfg),
        w_hh=copy.deepcopy(t16_linear_cfg),
    )
    print(f"Initial QConfig4 (partial params):\n{qconfig4.model_dump_json(indent=2)}")
    layer4 = FxPLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        batch_first=True,
        bias=BIAS,
        dropout=dropout,
        q_config=qconfig4,
    )
    layer4.load_state_dict(base_state_dict)
    layer4.eval()
    layer4.set_no_overflow_quant(same_wb=True, same_fc=True)
    print(
        f"\nLayer4 QConfig after set_no_overflow_quant:\n{layer4.q_config.model_dump_json(indent=2)}"
    )
    out4, (hn4, cn4) = layer4(dummy_input, logger=logger)
    log_path4 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_lstm_t16_no_overflow.json")
    logger.save_to_json(log_path4)
    print(f"Output4 shape: {out4.shape}, hn: {hn4.shape}, cn: {cn4.shape}")

    # 5) Mixed No Overflow (T=8 params, T=16/F=8 others)
    print("\n" + "=" * 20 + " FxPLSTM Scenario 5: Mixed No Overflow " + "=" * 20)
    logger.clear()
    t16f8 = QType(total_bits=16, fractional_bits=8, q_method=QMethod.ROUND_SATURATE)
    t8_param = QType(total_bits=8, q_method=QMethod.ROUND_SATURATE)

    t8_linear_cfg = LinearQConfig(
        input=copy.deepcopy(t16f8),
        weight=copy.deepcopy(t8_param),
        bias=copy.deepcopy(t8_param) if BIAS else QType(),
        activation=copy.deepcopy(t16f8),
    )
    t16_sigmoid_cfg = SigmoidQConfig(
        input=copy.deepcopy(t16f8),
        activation=copy.deepcopy(t16f8),
    )

    t16_tanh_cfg = TanhQConfig(
        input=copy.deepcopy(t16f8),
        activation=copy.deepcopy(t16f8),
    )

    qconfig5 = LSTMQConfig(
        input=copy.deepcopy(t16f8),
        h0=copy.deepcopy(t16f8),
        c0=copy.deepcopy(t16f8),
        w_ih=copy.deepcopy(t8_linear_cfg),
        w_hh=copy.deepcopy(t8_linear_cfg),
        i_act=copy.deepcopy(t16_sigmoid_cfg),
        f_act=copy.deepcopy(t16_sigmoid_cfg),
        g_act=copy.deepcopy(t16_tanh_cfg),
        o_act=copy.deepcopy(t16_sigmoid_cfg),
        ct_tanh=copy.deepcopy(t16_tanh_cfg),
        i_dot_g=copy.deepcopy(t16f8),
        f_dot_c_t=copy.deepcopy(t16f8),
        c_t=copy.deepcopy(t16f8),
        h_t=copy.deepcopy(t16f8),
        output=copy.deepcopy(t16f8),
    )
    print(f"Initial QConfig5:\n{qconfig5.model_dump_json(indent=2)}")
    layer5 = FxPLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        batch_first=True,
        bias=BIAS,
        dropout=dropout,
        q_config=qconfig5,
    )
    layer5.load_state_dict(base_state_dict)
    layer5.eval()
    layer5.set_no_overflow_quant(same_wb=True, same_fc=True)
    print(
        f"\nLayer5 QConfig after set_no_overflow_quant:\n{layer5.q_config.model_dump_json(indent=2)}"
    )
    out5, (hn5, cn5) = layer5(dummy_input, logger=logger)
    log_path5 = os.path.join(OUTPUTS_PATH_TESTS, "fxp_lstm_mixed_no_overflow.json")
    logger.save_to_json(log_path5)
    print(f"Output5 shape: {out5.shape}, hn: {hn5.shape}, cn: {cn5.shape}")

    # 6) Mixed No Overflow Calibrated (params 8-bit, others 16-bit)
    print(
        "\n"
        + "=" * 20
        + " FxPLSTM Scenario 6: Mixed No Overflow Calibrated "
        + "=" * 20
    )
    logger.clear()
    t16_any = QType(total_bits=16, q_method=QMethod.ROUND_SATURATE)
    t8_param2 = QType(total_bits=8, q_method=QMethod.ROUND_SATURATE)
    t8_lin_cfg2 = LinearQConfig(
        input=QType(),  # let calibration set precise bits if used inside FxPLinear
        weight=copy.deepcopy(t8_param2),
        bias=copy.deepcopy(t8_param2) if BIAS else QType(),
        activation=copy.deepcopy(t16_any),
    )
    t16_sigmoid_cfg2 = SigmoidQConfig(
        input=QType(),
        activation=copy.deepcopy(t16_any),
    )
    t16_tanh_cfg2 = TanhQConfig(
        input=QType(),
        activation=copy.deepcopy(t16_any),
    )
    qconfig6 = LSTMQConfig(
        input=QType(),
        h0=QType(),
        c0=QType(),
        w_ih=copy.deepcopy(t8_lin_cfg2),
        w_hh=copy.deepcopy(t8_lin_cfg2),
        i_act=copy.deepcopy(t16_sigmoid_cfg2),
        f_act=copy.deepcopy(t16_sigmoid_cfg2),
        g_act=copy.deepcopy(t16_tanh_cfg2),
        o_act=copy.deepcopy(t16_sigmoid_cfg2),
        ct_tanh=copy.deepcopy(t16_tanh_cfg2),
        i_dot_g=copy.deepcopy(t16_any),
        f_dot_c_t=copy.deepcopy(t16_any),
        c_t=copy.deepcopy(t16_any),
        h_t=copy.deepcopy(t16_any),
        output=copy.deepcopy(t16_any),
    )
    print(f"Initial QConfig6:\n{qconfig6.model_dump_json(indent=2)}")
    layer6 = FxPLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        batch_first=True,
        bias=BIAS,
        dropout=dropout,
        q_config=qconfig6,
    )
    layer6.load_state_dict(base_state_dict)
    layer6.eval()
    layer6.set_no_overflow_quant(same_wb=True, same_fc=True)
    out6, (hn6, cn6) = layer6(
        calibration_dummy_input,
        logger=logger,
        calibrate=True,
        calibration_type="no_overflow",
    )
    print(
        f"\nLayer6 QConfig after set_no_overflow_quant and calibration:\n{layer6.q_config.model_dump_json(indent=2)}"
    )
    log_path6 = os.path.join(
        OUTPUTS_PATH_TESTS, "fxp_lstm_mixed_no_overflow_calibration.json"
    )
    logger.save_to_json(log_path6)
    print(f"Output6 shape: {out6.shape}, hn: {hn6.shape}, cn: {cn6.shape}")

    # 7) Mixed MinMSE Calibrated
    print("\n" + "=" * 20 + " FxPLSTM Scenario 7: Mixed MinMSE Calibrated " + "=" * 20)
    logger.clear()
    qconfig7 = LSTMQConfig(
        input=QType(),
        h0=QType(),
        c0=QType(),
        w_ih=copy.deepcopy(t8_lin_cfg2),
        w_hh=copy.deepcopy(t8_lin_cfg2),
        i_act=copy.deepcopy(t16_sigmoid_cfg2),
        f_act=copy.deepcopy(t16_sigmoid_cfg2),
        g_act=copy.deepcopy(t16_tanh_cfg2),
        o_act=copy.deepcopy(t16_sigmoid_cfg2),
        ct_tanh=copy.deepcopy(t16_tanh_cfg2),
        i_dot_g=copy.deepcopy(t16_any),
        f_dot_c_t=copy.deepcopy(t16_any),
        c_t=copy.deepcopy(t16_any),
        h_t=copy.deepcopy(t16_any),
        output=copy.deepcopy(t16_any),
    )
    print(f"Initial QConfig7:\n{qconfig7.model_dump_json(indent=2)}")
    layer7 = FxPLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        batch_first=True,
        bias=BIAS,
        dropout=dropout,
        q_config=qconfig7,
    )
    layer7.load_state_dict(base_state_dict)
    layer7.eval()
    layer7.set_min_mse_quant()
    out7, (hn7, cn7) = layer7(
        calibration_dummy_input,
        logger=logger,
        calibrate=True,
        calibration_type="min_mse",
    )
    print(
        f"\nLayer7 QConfig after set_min_mse_quant and min_mse calibration:\n{layer7.q_config.model_dump_json(indent=2)}"
    )
    log_path7 = os.path.join(
        OUTPUTS_PATH_TESTS, "fxp_lstm_mixed_min_mse_calibration.json"
    )
    logger.save_to_json(log_path7)
    print(f"Output7 shape: {out7.shape}, hn: {hn7.shape}, cn: {cn7.shape}")

    print("\n" + "=" * 20 + " FxPLSTM Testing Complete " + "=" * 20)


# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting FxP Layer Tests...")
    test_fxp_linear()
    test_fxp_dropout()
    test_fxp_softmax()
    test_fxp_layernorm()
    test_fxp_multiheadattention()
    test_fxp_transformer_encoder()
    test_fxp_conv2d()
    test_fxp_sigmoid()
    test_fxp_tanh()
    test_fxp_lstm()
    print("\nAll FxP Layer Tests Finished.")
    print("Check the 'outputs/' subdirectories for JSON logs.")
