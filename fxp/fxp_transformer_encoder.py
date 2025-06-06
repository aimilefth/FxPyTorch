# FxPyTorch/fxp/fxp_transformer_encoder.py
import torch
from torch import nn
import torch.nn.functional as F
from .symmetric_quant import (
    QType,
    apply_quantize,
    QConfig,
)
from typing import Optional, Literal, Callable, Union
from ..transparent.activation_logger import (
    ActivationLogger,
    ActivationLoggingScope,
)
from pydantic import Field
from ..transparent.trans_transformer_encoder import (
    TransformerEncoderLayerTransparent,
)
from .fxp_dropout import FxPDropout, DropoutQConfig
from .fxp_softmax import FxPSoftmax
from .fxp_linear import FxPLinear, LinearQConfig
from .fxp_multiheadattention import (
    FxPMultiheadAttention,
    MultiheadAttentionQConfig,
)
from .fxp_layernorm import FxPLayerNorm, LayerNormQConfig
from .calibration import set_calibrated_activation_quant, CalibrationType


class TransformerEncoderLayerQConfig(QConfig):
    """Quantization configuration specific to FxPTransformerEncoderLayer layers."""

    # **Crucial**: Define the discriminator value using Literal
    layer_type: Literal["transformer_encoder_layer"] = "transformer_encoder_layer"
    """Pydantic model for quantization configuration of a FxPTransformerEncoderLayer layer."""
    input: QType = Field(default_factory=QType)
    norm1: LayerNormQConfig = Field(default_factory=LayerNormQConfig)
    self_attn: MultiheadAttentionQConfig = Field(
        default_factory=MultiheadAttentionQConfig
    )
    self_attn_dropout: DropoutQConfig = Field(default_factory=DropoutQConfig)
    residual_1: QType = Field(default_factory=QType)
    norm2: LayerNormQConfig = Field(default_factory=LayerNormQConfig)
    linear1: LinearQConfig = Field(default_factory=LinearQConfig)
    ff_activation: QType = Field(default_factory=QType)
    dropout1: DropoutQConfig = Field(default_factory=DropoutQConfig)
    linear2: LinearQConfig = Field(default_factory=LinearQConfig)
    dropout2: DropoutQConfig = Field(default_factory=DropoutQConfig)
    residual_2: QType = Field(default_factory=QType)


class FxPTransformerEncoderLayer(TransformerEncoderLayerTransparent):
    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        att_dropout: float = 0.0,
        activation: Union[
            str, Callable[[torch.Tensor], torch.Tensor]
        ] = nn.functional.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_bias_q: bool = False,
        dropout_fun: nn.Module = FxPDropout,
        softmax_fun: nn.Module = FxPSoftmax,
        layernorm_fun: nn.Module = FxPLayerNorm,
        device=None,
        dtype=None,
        q_config: MultiheadAttentionQConfig = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(FxPTransformerEncoderLayer, self).__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            att_dropout=att_dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_bias_q=add_bias_q,
            dropout_fun=dropout_fun,
            softmax_fun=softmax_fun,
            layernorm_fun=layernorm_fun,
            device=device,
            dtype=dtype,
        )
        self._q_config = q_config
        if self._q_config is not None:
            self.self_attn = FxPMultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=att_dropout,
                bias=bias,
                add_bias_kv=add_bias_kv,
                add_bias_q=add_bias_q,
                dropout_fun=dropout_fun,
                softmax_fun=softmax_fun,
                q_config=self._q_config.self_attn,
                **factory_kwargs,
            )
            # Implementation of Feedforward model
            self.linear1 = FxPLinear(
                d_model,
                dim_feedforward,
                bias=bias,
                q_config=self._q_config.linear1,
                **factory_kwargs,
            )
            self.dropout = dropout_fun(
                dropout, q_config=self._q_config.self_attn_dropout
            )
            self.linear2 = FxPLinear(
                dim_feedforward,
                d_model,
                bias=bias,
                q_config=self._q_config.linear2,
                **factory_kwargs,
            )

            self.norm_first = norm_first
            self.norm1 = layernorm_fun(
                d_model,
                eps=layer_norm_eps,
                bias=bias,
                q_config=self._q_config.norm1,
                **factory_kwargs,
            )
            self.norm2 = layernorm_fun(
                d_model,
                eps=layer_norm_eps,
                bias=bias,
                q_config=self._q_config.norm2,
                **factory_kwargs,
            )
            self.dropout1 = dropout_fun(dropout, q_config=self._q_config.dropout1)
            self.dropout2 = dropout_fun(dropout, q_config=self._q_config.dropout2)

    @property
    def q_config(self) -> TransformerEncoderLayerQConfig:
        return self._q_config

    @q_config.setter
    def q_config(self, new_q_config: TransformerEncoderLayerQConfig):
        """
        Merge a brand‑new attention QConfig into the existing one, in place, so that:

          • All four top‑level QType fields (input, residual_1, ff_activation, residual_2) are updated,
            preserving the exact same objects (no rebinding).

          • Each nested sub‑config (attention, linear1 etc.)
            is merged into its existing instance via that submodule’s own setter.

        After this, all submodules still point at those same shared instances,
        but with new field values.
        """
        if self._q_config is None:
            self._q_config = new_q_config
        else:
            # merge top‑level fields in place
            for name in (
                "input",
                "residual_1",
                "ff_activation",
                "residual_2",
            ):
                # Glorified set._q_config.input_query = new_q_config.input_query
                setattr(self._q_config, name, getattr(new_q_config, name))
        # Re‑wire each submodule so it holds the same nested config objects
        self.norm1.q_config = new_q_config.norm1
        self.self_attn.q_config = new_q_config.self_attn
        self.dropout.q_config = new_q_config.self_attn_dropout
        self.norm2.q_config = new_q_config.norm2
        self.linear1.q_config = new_q_config.linear1
        self.dropout1.q_config = new_q_config.dropout1
        self.linear2.q_config = new_q_config.linear2
        self.dropout2.q_config = new_q_config.dropout2

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        # ----- New arguments for verbose logging -----
        logger: Optional[ActivationLogger] = None,
        apply_ste: bool = True,
        calibrate: bool = False,
        calibration_type: Union[str, CalibrationType] = CalibrationType.NO_OVERFLOW,
    ) -> torch.Tensor:
        if self._q_config is None:
            # Floating point, call TransformerEncoderLayerTransparent
            return super(FxPTransformerEncoderLayer, self).forward(
                src=src,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
                logger=logger,
            )
        assert src_key_padding_mask is None, (
            "TransformerEncoderLayerTransparent does not support src_key_padding_mask"
        )
        assert is_causal is False, (
            "TransformerEncoderLayerTransparent does not support is_causal"
        )
        with ActivationLoggingScope(logger, self):
            src_mask = nn.functional._canonical_mask(
                mask=src_mask,
                mask_name="src_mask",
                other_type=None,
                other_name="",
                target_type=src.dtype,
                check_other=False,
            )

            # --- Main Layer Logic ---
            if calibrate:
                set_calibrated_activation_quant(
                    src, self._q_config.input, calibration_type
                )
            x = apply_quantize(src, self._q_config.input, apply_ste)
            # --- Log Initial Inputs ---
            if logger:
                logger.log("input", src, self)
                logger.log("input_quant", x, self)
                if src_mask is not None:
                    logger.log("input_src_mask", src_mask, self)

            if self.norm_first:
                # 1. LayerNorm
                norm1_out = self.norm1(
                    x,
                    logger=logger,
                    apply_ste=apply_ste,
                    calibrate=calibrate,
                    calibration_type=calibration_type,
                )
                # 2. Self-Attention (+ dropout) + Residual
                # Pass verbose flags down. return_log_dict is passed as verbose to signal MHA
                # to *collect* logs if verbose is True, regardless of whether *this* layer returns them.
                attention_out = self.self_attn(
                    query=norm1_out,
                    key=norm1_out,
                    value=norm1_out,
                    attn_mask=src_mask,
                    need_weights=False,
                    key_padding_mask=None,
                    is_causal=False,
                    logger=logger,
                    apply_ste=apply_ste,
                    calibrate=calibrate,
                    calibration_type=calibration_type,
                )[0]

                sa_dropout_out = self.dropout(
                    attention_out,
                    logger=logger,
                    apply_ste=apply_ste,
                    calibrate=calibrate,
                    calibration_type=calibration_type,
                )
                residual_1 = x + sa_dropout_out  # First residual connection
                if calibrate:
                    set_calibrated_activation_quant(
                        residual_1, self._q_config.residual_1, calibration_type
                    )
                x = apply_quantize(residual_1, self._q_config.residual_1, apply_ste)
                if logger:
                    logger.log("residual1_after_attn", residual_1, self)
                    logger.log("residual1_after_attn_quant", x, self)

                # 3. LayerNorm
                norm2_out = self.norm2(
                    x,
                    logger=logger,
                    apply_ste=apply_ste,
                    calibrate=calibrate,
                    calibration_type=calibration_type,
                )

                # 4. Feed Forward (+ dropout) + Residual
                # Pass the log target to the helper block
                ff_out = self._ff_block(
                    norm2_out,
                    logger=logger,
                    apply_ste=apply_ste,
                    calibrate=calibrate,
                    calibration_type=calibration_type,
                )

                residual_2 = x + ff_out  # Second residual connection
                if calibrate:
                    set_calibrated_activation_quant(
                        residual_2, self._q_config.residual_2, calibration_type
                    )
                x = apply_quantize(residual_2, self._q_config.residual_2, apply_ste)
                if logger:
                    logger.log("residual_2_pre_quant", residual_2, self)
                    logger.log("output", x, self)

            else:  # Post-LN
                # 1. Self-Attention (+ dropout) + Residual + LayerNorm
                attention_out = self.self_attn(
                    query=x,
                    key=x,
                    value=x,
                    attn_mask=src_mask,
                    need_weights=False,
                    key_padding_mask=None,
                    is_causal=False,
                    logger=logger,
                    apply_ste=apply_ste,
                    calibrate=calibrate,
                    calibration_type=calibration_type,
                )[0]

                sa_dropout_out = self.dropout(
                    attention_out,
                    logger=logger,
                    apply_ste=apply_ste,
                    calibrate=calibrate,
                    calibration_type=calibration_type,
                )

                add1_out = x + sa_dropout_out  # Add before norm
                if calibrate:
                    set_calibrated_activation_quant(
                        add1_out, self._q_config.residual_1, calibration_type
                    )
                add1_out_quant = apply_quantize(
                    add1_out, self._q_config.residual_1, apply_ste
                )
                if logger:
                    logger.log("residual1_after_attn", add1_out, self)
                    logger.log("residual1_after_attn_quant", add1_out_quant, self)

                x = self.norm1(
                    add1_out_quant,
                    logger=logger,
                    apply_ste=apply_ste,
                    calibrate=calibrate,
                    calibration_type=calibration_type,
                )  # First LayerNorm (post-add)

                # 2. Feed Forward (+ dropout) + Residual + LayerNorm
                # Pass the log target to the helper block
                ff_out = self._ff_block(
                    x,
                    logger=logger,
                    apply_ste=apply_ste,
                    calibrate=calibrate,
                    calibration_type=calibration_type,
                )

                add2_out = x + ff_out  # Add before norm
                if calibrate:
                    set_calibrated_activation_quant(
                        add2_out, self._q_config.residual_2, calibration_type
                    )
                add2_out_quant = apply_quantize(
                    add2_out, self._q_config.residual_2, apply_ste
                )
                if logger:
                    logger.log("residual2_after_ffn_output", add2_out, self)
                    logger.log("residual2_after_ffn_output_quant", add2_out_quant, self)

                x = self.norm2(
                    add2_out_quant,
                    logger=logger,
                    apply_ste=apply_ste,
                    calibrate=calibrate,
                    calibration_type=calibration_type,
                )  # Second LayerNorm (post-add)
                if logger:
                    logger.log("output", x, self)

        # --- Prepare Return Value ---
        output = x
        return output

    # Feed forward block helper modified to accept log target
    def _ff_block(
        self,
        x: torch.Tensor,
        logger: Optional[ActivationLogger],
        apply_ste: bool = True,
        calibrate: bool = False,
        calibration_type: Union[str, CalibrationType] = CalibrationType.NO_OVERFLOW,
    ) -> torch.Tensor:
        # ---------- FLOATING‑POINT PATH ----------
        if self._q_config is None:
            # Delegate straight to the transparent implementation
            return TransformerEncoderLayerTransparent._ff_block(self, x, logger)
        # ---------- QUANTISED PATH (current code) ----------
        with ActivationLoggingScope(logger, "ffn"):
            linear1_out = self.linear1(
                x,
                logger=logger,
                apply_ste=apply_ste,
                calibrate=calibrate,
                calibration_type=calibration_type,
            )
            act_out = self.activation(linear1_out)
            if calibrate:
                set_calibrated_activation_quant(
                    act_out, self._q_config.ff_activation, calibration_type
                )
            act_out_quant = apply_quantize(
                act_out, self._q_config.ff_activation, apply_ste
            )
            if logger:
                logger.log("ff_activation", act_out, self)
                logger.log("ff_activation_quant", act_out_quant, self)

            dropout_act_out = self.dropout1(
                act_out_quant,
                logger=logger,
                apply_ste=apply_ste,
                calibrate=calibrate,
                calibration_type=calibration_type,
            )
            linear2_out = self.linear2(
                dropout_act_out,
                logger=logger,
                apply_ste=apply_ste,
                calibrate=calibrate,
                calibration_type=calibration_type,
            )
            ff_dropout_out = self.dropout2(
                linear2_out,
                logger=logger,
                apply_ste=apply_ste,
                calibrate=calibrate,
                calibration_type=calibration_type,
            )
            if logger:
                logger.log("output", ff_dropout_out, self)  # This is the block output
            # The final output of this block is logged by the caller as "ff_block_out"
        return ff_dropout_out

    def quantize_weights_bias(self) -> None:
        self.norm1.quantize_weights_bias()
        self.self_attn.quantize_weights_bias()
        self.norm2.quantize_weights_bias()
        self.linear1.quantize_weights_bias()
        self.linear2.quantize_weights_bias()

    def set_high_precision_quant(
        self,
        same_ff: bool = False,
        same_all_attn: bool = False,
        same_qkv: bool = False,
        same_wb: bool = False,
    ) -> None:
        # Dont bother with same_all, too much
        self.norm1.set_high_precision_quant(same_wb)
        self.self_attn.set_high_precision_quant(
            same_all=same_all_attn, same_qkv=same_qkv, same_wb=same_wb
        )
        self.norm2.set_high_precision_quant(same_wb)
        self.linear1.set_high_precision_quant(same_wb)
        self.linear2.set_high_precision_quant(same_wb)
        if same_ff:
            # To get the same quantization, choose the max total_bits required (24 fractional standard)
            max_total_bits_w = max(
                self.linear1.q_config.weight.total_bits,
                self.linear2.q_config.weight.total_bits,
            )
            self.linear1.q_config.weight.total_bits = (
                self.linear2.q_config.weight.total_bits
            ) = max_total_bits_w
            max_total_bits_b = max(
                self.linear1.q_config.bias.total_bits,
                self.linear2.q_config.bias.total_bits,
            )
            self.linear1.q_config.bias.total_bits = (
                self.linear2.q_config.bias.total_bits
            ) = max_total_bits_b

    def set_no_overflow_quant(
        self,
        same_ff: bool = False,
        same_all_attn: bool = False,
        same_qkv: bool = False,
        same_wb: bool = False,
    ) -> None:
        # Dont bother with same_all, too much
        self.norm1.set_no_overflow_quant(same_wb)
        self.self_attn.set_no_overflow_quant(
            same_all=same_all_attn, same_qkv=same_qkv, same_wb=same_wb
        )
        self.norm2.set_no_overflow_quant(same_wb)
        self.linear1.set_no_overflow_quant(same_wb)
        self.linear2.set_no_overflow_quant(same_wb)
        if same_ff:
            # To avoid overflow,
            # 1) check the max total_integer_bits
            max_integer_bits_w = max(
                self.linear1.q_config.weight.integer_bits,
                self.linear2.q_config.weight.integer_bits,
            )
            # 2) Make total_bits = max_total_bits
            max_total_bits_w = max(
                self.linear1.q_config.weight.total_bits,
                self.linear2.q_config.weight.total_bits,
            )
            self.linear1.q_config.weight.total_bits = (
                self.linear2.q_config.weight.total_bits
            ) = max_total_bits_w
            # 3) Change the integer_bits to max_integer_bit to avoid overflow (-> minimize fractional)
            self.linear1.q_config.weight.fractional_bits = (
                self.linear2.q_config.weight.fractional_bits
            ) = max_total_bits_w - max_integer_bits_w
            # To avoid overflow,
            # 1) check the max total_integer_bits
            max_integer_bits_b = max(
                self.linear1.q_config.bias.integer_bits,
                self.linear2.q_config.bias.integer_bits,
            )
            # 2) Make total_bits = max_total_bits
            max_total_bits_b = max(
                self.linear1.q_config.bias.total_bits,
                self.linear2.q_config.bias.total_bits,
            )
            self.linear1.q_config.bias.total_bits = (
                self.linear2.q_config.bias.total_bits
            ) = max_total_bits_b
            # 3) Change the integer_bits to max_integer_bit to avoid overflow (-> minimize fractional)
            self.linear1.q_config.bias.fractional_bits = (
                self.linear2.q_config.bias.fractional_bits
            ) = max_total_bits_b - max_integer_bits_b


def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")
