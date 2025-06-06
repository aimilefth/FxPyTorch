# FxPyTorch/fxp/fxp_multiheadattention.py
import torch
from torch import nn
from .symmetric_quant import (
    QType,
    apply_quantize,
    QConfig,
)
from typing import Optional, Literal, Union
from ..transparent.activation_logger import (
    ActivationLogger,
    ActivationLoggingScope,
)
from pydantic import Field
from typing import Tuple
from ..transparent.trans_multiheadattention import (
    MultiheadAttentionTransparent,
)
from .fxp_dropout import FxPDropout, DropoutQConfig
from .fxp_softmax import FxPSoftmax, SoftmaxQConfig
from .fxp_linear import FxPLinear, LinearQConfig
import warnings
from .calibration import set_calibrated_activation_quant, CalibrationType


class MultiheadAttentionQConfig(QConfig):
    """Quantization configuration specific to FxPMultiheadAttention layers."""

    # **Crucial**: Define the discriminator value using Literal
    layer_type: Literal["multihead_attention"] = "multihead_attention"
    """Pydantic model for quantization configuration of a FxPMultiheadAttention layer."""
    input_query: QType = Field(default_factory=QType)
    input_key: QType = Field(default_factory=QType)
    input_value: QType = Field(default_factory=QType)
    qlinear: LinearQConfig = Field(default_factory=LinearQConfig)
    klinear: LinearQConfig = Field(default_factory=LinearQConfig)
    vlinear: LinearQConfig = Field(default_factory=LinearQConfig)
    q_scaled: QType = Field(default_factory=QType)
    attn_scores_raw: QType = Field(default_factory=QType)
    softmax: SoftmaxQConfig = Field(default_factory=SoftmaxQConfig)
    dropout: DropoutQConfig = Field(default_factory=DropoutQConfig)
    attn_output: QType = Field(default_factory=QType)
    out_proj: LinearQConfig = Field(default_factory=LinearQConfig)


class FxPMultiheadAttention(MultiheadAttentionTransparent):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_bias_q: bool = False,
        add_zero_att: bool = False,
        kdim: int = None,
        vdim: int = None,
        batch_first: bool = True,
        dropout_fun: nn.Module = FxPDropout,
        softmax_fun: nn.Module = FxPSoftmax,
        device=None,
        dtype=None,
        q_config: MultiheadAttentionQConfig = None,
    ):
        super(FxPMultiheadAttention, self).__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_bias_q=add_bias_q,
            add_zero_att=add_zero_att,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            dropout_fun=dropout_fun,
            softmax_fun=softmax_fun,
            device=device,
            dtype=dtype,
        )
        self._q_config = q_config
        if self._q_config is not None:
            factory_kwargs = {"device": device, "dtype": dtype}
            self.qlinear = FxPLinear(
                embed_dim,
                embed_dim,
                bias=add_bias_q,
                q_config=self._q_config.qlinear,
                **factory_kwargs,
            )
            self.klinear = FxPLinear(
                embed_dim,
                embed_dim,
                bias=add_bias_kv,
                q_config=self._q_config.klinear,
                **factory_kwargs,
            )
            self.vlinear = FxPLinear(
                embed_dim,
                embed_dim,
                bias=add_bias_kv,
                q_config=self._q_config.vlinear,
                **factory_kwargs,
            )
            self.out_proj = FxPLinear(
                embed_dim,
                embed_dim,
                bias=bias,
                q_config=self._q_config.out_proj,
                **factory_kwargs,
            )
            self.dropout = dropout_fun(dropout, q_config=self._q_config.dropout)
            self.softmax = softmax_fun(dim=-1, q_config=self._q_config.softmax)

    @property
    def q_config(self) -> MultiheadAttentionQConfig:
        return self._q_config

    @q_config.setter
    def q_config(self, new_q_config: MultiheadAttentionQConfig):
        """
        Merge a brand‑new attention QConfig into the existing one, in place, so that:

          • All six top‑level QType fields (input_query, input_key, …) are updated,
            preserving the exact same objects (no rebinding).

          • Each nested sub‑config (qlinear, klinear, vlinear, out_proj, dropout, softmax)
            is merged into its existing instance via that submodule’s own setter.

        After this, all submodules still point at those same shared instances,
        but with new field values.
        """
        if self._q_config is None:
            self._q_config = new_q_config
        else:
            # merge top‑level fields in place
            for name in (
                "input_query",
                "input_key",
                "input_value",
                "q_scaled",
                "attn_scores_raw",
                "attn_output",
            ):
                # Glorified set._q_config.input_query = new_q_config.input_query
                setattr(self._q_config, name, getattr(new_q_config, name))
        # Re‑wire each submodule so it holds the same nested config objects
        self.qlinear.q_config = new_q_config.qlinear
        self.klinear.q_config = new_q_config.klinear
        self.vlinear.q_config = new_q_config.vlinear
        self.dropout.q_config = new_q_config.dropout
        self.softmax.q_config = new_q_config.softmax
        self.out_proj.q_config = new_q_config.out_proj

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        logger: Optional[ActivationLogger] = None,
        apply_ste: bool = True,
        calibrate: bool = False,
        calibration_type: Union[str, CalibrationType] = CalibrationType.NO_OVERFLOW,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._q_config is None:
            # Floating point, call DropoutTransparent
            return super(FxPMultiheadAttention, self).forward(
                query=query,
                key=key,
                value=value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                is_causal=is_causal,
                logger=logger,
            )
        # --- Assertions ---
        assert key_padding_mask is None, (
            "MultiheadAttentionTransparent does not work with key_padding_mask"
        )
        is_batched = query.dim() == 3
        assert is_batched, "The query must have a dimension of 3."

        r"""
        As per https://github.com/pytorch/opacus/issues/596, we have to include ``is_causal`` as a dummy parameter of the function,
        since it is used in the ``forward`` function of parent class ``nn.TransformerEncoderLayer``.
        """
        assert not is_causal, (
            "We currently do not support causal mask. Will fix it in the future."
        )
        with ActivationLoggingScope(logger, self):
            r"""
            Using the same logic with ``nn.MultiheadAttention`` (https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).
            """
            bsz, tgt_len, embed_dim = query.size()
            if embed_dim != self.embed_dim:
                raise ValueError(
                    f"query has as size of {embed_dim} while the embedding"
                    " size is {self.embed_dim}"
                )
            # STE
            if calibrate:
                set_calibrated_activation_quant(
                    query, self._q_config.input_query, calibration_type
                )
                set_calibrated_activation_quant(
                    key, self._q_config.input_key, calibration_type
                )
                set_calibrated_activation_quant(
                    value, self._q_config.input_value, calibration_type
                )
            query_quant = apply_quantize(query, self._q_config.input_query, apply_ste)
            key_quant = apply_quantize(key, self._q_config.input_key, apply_ste)
            value_quant = apply_quantize(value, self._q_config.input_value, apply_ste)

            # Log inputs if verbose
            if logger:
                logger.log("input_query", query, self)
                logger.log("input_key", key, self)
                logger.log("input_value", value, self)
                logger.log("input_query_quant", query_quant, self)
                logger.log("input_key_quant", key_quant, self)
                logger.log("input_value_quant", value_quant, self)
                if attn_mask is not None:
                    logger.log("input_attn_mask", attn_mask, self)

            head_dim = (
                embed_dim // self.num_heads
            )  # Already checked divisibility in init
            scaling = float(head_dim) ** -0.5

            q_quant = self.qlinear(
                query_quant,
                logger=logger,
                apply_ste=apply_ste,
                calibrate=calibrate,
                calibration_type=calibration_type,
            )
            k_quant = self.klinear(
                key_quant,
                logger=logger,
                apply_ste=apply_ste,
                calibrate=calibrate,
                calibration_type=calibration_type,
            )
            v_quant = self.vlinear(
                value_quant,
                logger=logger,
                apply_ste=apply_ste,
                calibrate=calibrate,
                calibration_type=calibration_type,
            )

            q_scaled = q_quant * scaling
            if calibrate:
                set_calibrated_activation_quant(
                    q_scaled, self._q_config.q_scaled, calibration_type
                )
            q_scaled_quant = apply_quantize(
                q_scaled, self._q_config.q_scaled, apply_ste
            )
            if logger:
                logger.log("q_scaled", q_scaled, self)
                logger.log("q_scaled_quant", q_scaled_quant, self)
            # Do this transpose for head calculation (make them (tgt_len, bsz, -1))
            q, k, v = [x.transpose(0, 1) for x in (q_scaled_quant, k_quant, v_quant)]

            # Note: This mask processing happens *before* reshaping QKV for bmm in the original code.
            if attn_mask is not None:
                if attn_mask.dtype not in (
                    torch.float32,
                    torch.float64,
                    torch.uint8,
                    torch.bool,
                ):
                    raise ValueError(
                        f"Only float, byte, and bool types are supported for attn_mask, "
                        f"not {attn_mask.dtype}."
                    )

                if attn_mask.dtype == torch.uint8:
                    warnings.warn(
                        "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated."
                        "Use bool tensor instead."
                    )
                    attn_mask = attn_mask.to(torch.bool)

                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0)
                    if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                        raise ValueError("The size of the 2D attn_mask is not correct.")
                elif attn_mask.dim() == 3:
                    if list(attn_mask.size()) != [
                        bsz * self.num_heads,
                        query.size(0),
                        key.size(0),
                    ]:
                        raise ValueError("The size of the 3D attn_mask is not correct.")
                else:
                    raise ValueError(
                        "attn_mask's dimension {} is not supported".format(
                            attn_mask.dim()
                        )
                    )
                # attn_mask's dim is 3 now.

            # Reshape for BMM (original logic)
            # Return to batch_first-like format but withT heads merged into batch dim
            # q: (T, B, E) -> (T, B*H, D) -> (B*H, T, D)
            q = (
                q.contiguous()
                .view(tgt_len, bsz * self.num_heads, head_dim)
                .transpose(0, 1)
            )
            if k is not None:
                # k: (S, B, E) -> (S, B*H, D) -> (B*H, S, D)
                k_src_len = k.size(0)  # Save src len after transpose
                k = (
                    k.contiguous()
                    .view(k_src_len, bsz * self.num_heads, head_dim)
                    .transpose(0, 1)
                )
            if v is not None:
                # v: (S, B, E) -> (S, B*H, D) -> (B*H, S, D)
                v_src_len = v.size(0)  # Save src len after transpose (should match k)
                v = (
                    v.contiguous()
                    .view(v_src_len, bsz * self.num_heads, head_dim)
                    .transpose(0, 1)
                )

            # Use the source length from the reshaped key tensor
            src_len = k.size(1)  # src_len is now dim 1

            # Calculate Attention Scores (original logic)
            attn_scores_raw = torch.bmm(q, k.transpose(1, 2))
            if calibrate:
                set_calibrated_activation_quant(
                    attn_scores_raw, self._q_config.attn_scores_raw, calibration_type
                )
            attn_scores_raw_quant = apply_quantize(
                attn_scores_raw, self._q_config.attn_scores_raw, apply_ste
            )
            # Assert shape (original logic)
            assert list(attn_scores_raw_quant.size()) == [
                bsz * self.num_heads,
                tgt_len,
                src_len,
            ]
            # Log attn_output_weights before mask
            if logger:
                logger.log(
                    "attn_scores_raw_BxHxTxS",
                    attn_scores_raw.view(bsz, self.num_heads, tgt_len, src_len),
                    self,
                )
                logger.log(
                    "attn_scores_raw_quant_BxHxTxS",
                    attn_scores_raw_quant.view(bsz, self.num_heads, tgt_len, src_len),
                    self,
                )
                # Apply Attention Mask (original logic)
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_scores_raw_quant = attn_scores_raw_quant.masked_fill_(
                        attn_mask, float("-inf")
                    )
                else:
                    # Ensure dtype matches for addition
                    attn_scores_raw_quant = attn_scores_raw_quant + attn_mask.to(
                        attn_scores_raw_quant.dtype
                    )

                # Log attn_output_weights after mask
                if logger:
                    logger.log(
                        "attn_scores_quant_masked_B,HxTxS",
                        attn_scores_raw_quant.view(
                            bsz, self.num_heads, tgt_len, src_len
                        ),
                        self,
                    )

            attn_weights_softmax_quant = self.softmax(
                attn_scores_raw_quant,
                logger=logger,
                apply_ste=apply_ste,
                calibrate=calibrate,
                calibration_type=calibration_type,
            )
            if logger:
                logger.log(
                    "attn_weights_softmax_quant_BxHxTxS",
                    attn_weights_softmax_quant.view(
                        bsz, self.num_heads, tgt_len, src_len
                    ),
                    self,
                )

            # Dropout (original logic - using the module)
            attn_weights_dropout_quant = self.dropout(
                attn_weights_softmax_quant,
                logger=logger,
                apply_ste=apply_ste,
                calibrate=calibrate,
                calibration_type=calibration_type,
            )
            # Log attn_output_weights after dropout
            if logger:
                logger.log(
                    "attn_weights_dropout_quant_BxHxTxS",
                    attn_weights_dropout_quant.view(
                        bsz, self.num_heads, tgt_len, src_len
                    ),
                    self,
                )
            attn_output = torch.bmm(attn_weights_dropout_quant, v)
            if calibrate:
                set_calibrated_activation_quant(
                    attn_output, self._q_config.attn_output, calibration_type
                )
            attn_output_quant = apply_quantize(
                attn_output, self._q_config.attn_output, apply_ste
            )
            assert list(attn_output_quant.size()) == [
                bsz * self.num_heads,
                tgt_len,
                head_dim,
            ]
            # Log attn_output
            if logger:
                logger.log(
                    "attn_output_weighted_sum_quant_BxHxTxD",
                    attn_output_quant.view(bsz, self.num_heads, tgt_len, head_dim),
                    self,
                )

            # Reshape Output (original logic) - Creates 'concat_output' implicitly
            # (B*H, T, D) -> (B, T, E)
            # This tensor was called 'concat_output' in the request.
            concat_output = attn_output_quant.contiguous().view(bsz, tgt_len, embed_dim)
            # Log concat_output
            if logger:
                logger.log("attn_output_quant_reshaped_BxTxE", concat_output, self)

            total_output_quant = self.out_proj(
                concat_output,
                logger=logger,
                apply_ste=apply_ste,
                calibrate=calibrate,
                calibration_type=calibration_type,
            )
            # Log total_output
            if logger:
                logger.log("output", total_output_quant, self)
        # --- End of original calculation logic ---

        if need_weights:
            # average attention weights over heads
            attn_weights_softmax_quant = attn_weights_softmax_quant.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            return (
                total_output_quant,
                attn_weights_softmax_quant.sum(dim=1) / self.num_heads,
            )
        else:
            return (
                total_output_quant,
                None,
            )

    def quantize_weights_bias(self) -> None:
        self.qlinear.quantize_weights_bias()
        self.klinear.quantize_weights_bias()
        self.vlinear.quantize_weights_bias()
        self.out_proj.quantize_weights_bias()

    def set_high_precision_quant(
        self, same_all: bool = False, same_qkv: bool = False, same_wb: bool = False
    ) -> None:
        # Important. Change the q_configs INSIDE the sub-layers
        self.qlinear.set_high_precision_quant(same_wb)
        self.klinear.set_high_precision_quant(same_wb)
        self.vlinear.set_high_precision_quant(same_wb)
        self.out_proj.set_high_precision_quant(same_wb)
        if same_all:
            # To get the same quantization, choose the max total_bits required (24 fractional standard)
            max_total_bits_w = max(
                self.qlinear.q_config.weight.total_bits,
                self.klinear.q_config.weight.total_bits,
                self.vlinear.q_config.weight.total_bits,
                self.out_proj.q_config.weight.total_bits,
            )
            self.qlinear.q_config.weight.total_bits = (
                self.klinear.q_config.weight.total_bits
            ) = self.vlinear.q_config.weight.total_bits = (
                self.out_proj.q_config.weight.total_bits
            ) = max_total_bits_w
            max_total_bits_b = max(
                self.qlinear.q_config.bias.total_bits,
                self.klinear.q_config.bias.total_bits,
                self.vlinear.q_config.bias.total_bits,
                self.out_proj.q_config.bias.total_bits,
            )
            self.qlinear.q_config.bias.total_bits = (
                self.klinear.q_config.bias.total_bits
            ) = self.vlinear.q_config.bias.total_bits = (
                self.out_proj.q_config.bias.total_bits
            ) = max_total_bits_b
        elif same_qkv:
            max_total_bits_w = max(
                self.qlinear.q_config.weight.total_bits,
                self.klinear.q_config.weight.total_bits,
                self.vlinear.q_config.weight.total_bits,
            )
            self.qlinear.q_config.weight.total_bits = (
                self.klinear.q_config.weight.total_bits
            ) = self.vlinear.q_config.weight.total_bits = max_total_bits_w
            max_total_bits_b = max(
                self.qlinear.q_config.bias.total_bits,
                self.klinear.q_config.bias.total_bits,
                self.vlinear.q_config.bias.total_bits,
            )
            self.qlinear.q_config.bias.total_bits = (
                self.klinear.q_config.bias.total_bits
            ) = self.vlinear.q_config.bias.total_bits = max_total_bits_b

    def set_no_overflow_quant(
        self, same_all: bool = False, same_qkv: bool = False, same_wb: bool = False
    ) -> None:
        self.qlinear.set_no_overflow_quant(same_wb)
        self.klinear.set_no_overflow_quant(same_wb)
        self.vlinear.set_no_overflow_quant(same_wb)
        self.out_proj.set_no_overflow_quant(same_wb)
        if same_all:
            # To avoid overflow,
            # 1) check the max total_integer_bits
            max_integer_bits_w = max(
                self.qlinear.q_config.weight.integer_bits,
                self.klinear.q_config.weight.integer_bits,
                self.vlinear.q_config.weight.integer_bits,
                self.out_proj.q_config.weight.integer_bits,
            )
            # 2) Make total_bits = max_total_bits
            max_total_bits_w = max(
                self.qlinear.q_config.weight.total_bits,
                self.klinear.q_config.weight.total_bits,
                self.vlinear.q_config.weight.total_bits,
                self.out_proj.q_config.weight.total_bits,
            )
            self.qlinear.q_config.weight.total_bits = (
                self.klinear.q_config.weight.total_bits
            ) = self.vlinear.q_config.weight.total_bits = (
                self.out_proj.q_config.weight.total_bits
            ) = max_total_bits_w
            # 3) Change the integer_bits to max_integer_bit to avoid overflow (-> minimize fractional)
            self.qlinear.q_config.weight.fractional_bits = (
                self.klinear.q_config.weight.fractional_bits
            ) = self.vlinear.q_config.weight.fractional_bits = (
                self.out_proj.q_config.weight.fractional_bits
            ) = max_total_bits_w - max_integer_bits_w
            # To avoid overflow,
            # 1) check the max total_integer_bits
            max_integer_bits_b = max(
                self.qlinear.q_config.bias.integer_bits,
                self.klinear.q_config.bias.integer_bits,
                self.vlinear.q_config.bias.integer_bits,
                self.out_proj.q_config.bias.integer_bits,
            )
            # 2) Make total_bits = max_total_bits
            max_total_bits_b = max(
                self.qlinear.q_config.bias.total_bits,
                self.klinear.q_config.bias.total_bits,
                self.vlinear.q_config.bias.total_bits,
                self.out_proj.q_config.bias.total_bits,
            )
            self.qlinear.q_config.bias.total_bits = (
                self.klinear.q_config.bias.total_bits
            ) = self.vlinear.q_config.bias.total_bits = (
                self.out_proj.q_config.bias.total_bits
            ) = max_total_bits_b
            # 3) Change the integer_bits to max_integer_bit to avoid overflow (-> minimize fractional)
            self.qlinear.q_config.bias.fractional_bits = (
                self.klinear.q_config.bias.fractional_bits
            ) = self.vlinear.q_config.bias.fractional_bits = (
                self.out_proj.q_config.bias.fractional_bits
            ) = max_total_bits_b - max_integer_bits_b
        elif same_qkv:
            # To avoid overflow,
            # 1) check the max total_integer_bits
            max_integer_bits_w = max(
                self.qlinear.q_config.weight.integer_bits,
                self.klinear.q_config.weight.integer_bits,
                self.vlinear.q_config.weight.integer_bits,
            )
            # 2) Make total_bits = max_total_bits
            max_total_bits_w = max(
                self.qlinear.q_config.weight.total_bits,
                self.klinear.q_config.weight.total_bits,
                self.vlinear.q_config.weight.total_bits,
            )
            self.qlinear.q_config.weight.total_bits = (
                self.klinear.q_config.weight.total_bits
            ) = self.vlinear.q_config.weight.total_bits = max_total_bits_w
            # 3) Change the integer_bits to max_integer_bit to avoid overflow (-> minimize fractional)
            self.qlinear.q_config.weight.fractional_bits = (
                self.klinear.q_config.weight.fractional_bits
            ) = self.vlinear.q_config.weight.fractional_bits = (
                max_total_bits_w - max_integer_bits_w
            )
            # To avoid overflow,
            # 1) check the max total_integer_bits
            max_integer_bits_b = max(
                self.qlinear.q_config.bias.integer_bits,
                self.klinear.q_config.bias.integer_bits,
                self.vlinear.q_config.bias.integer_bits,
            )
            # 2) Make total_bits = max_total_bits
            max_total_bits_b = max(
                self.qlinear.q_config.bias.total_bits,
                self.klinear.q_config.bias.total_bits,
                self.vlinear.q_config.bias.total_bits,
            )
            self.qlinear.q_config.bias.total_bits = (
                self.klinear.q_config.bias.total_bits
            ) = self.vlinear.q_config.bias.total_bits = max_total_bits_b
            # 3) Change the integer_bits to max_integer_bit to avoid overflow (-> minimize fractional)
            self.qlinear.q_config.bias.fractional_bits = (
                self.klinear.q_config.bias.fractional_bits
            ) = self.vlinear.q_config.bias.fractional_bits = (
                max_total_bits_b - max_integer_bits_b
            )
