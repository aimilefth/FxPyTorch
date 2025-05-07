# FxPyTorch/transparent/trans_transformer_encoder.py
import torch
from torch import nn
from typing import Callable, Optional, Union
import torch.nn.functional as F
from .activation_logger import (
    ActivationLogger,
    ActivationLoggingScope,
)
from .trans_linear import LinearTransparent
from .trans_dropout import DropoutTransparent
from .trans_softmax import SoftmaxTransparent
from .trans_layernorm import LayerNormTransparent
from .trans_multiheadattention import (
    MultiheadAttentionTransparent,
)


class TransformerEncoderLayerTransparent(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.

    .. note::
        See `this tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>`_
        for an in depth discussion of the performant building blocks PyTorch offers for building your own
        transformer layers.

    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    TransformerEncoderLayer can handle either traditional torch.tensor inputs,
    or Nested Tensor inputs.  Derived classes are expected to similarly accept
    both input formats.  (Not all combinations of inputs are currently
    supported by TransformerEncoderLayer while Nested Tensor is in prototype
    state.)

    If you are implementing a custom layer, you may derive it either from
    the Module or TransformerEncoderLayer class.  If your custom layer
    supports both torch.Tensors and Nested Tensors inputs, make its
    implementation a derived class of TransformerEncoderLayer. If your custom
    Layer supports only torch.Tensor inputs, derive its implementation from
    Module.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Fast path:
        forward() will use a special optimized implementation described in
        `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`_ if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mTransoferEncoderLayer_transparentask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.

        .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
         https://arxiv.org/abs/2205.14135

    """

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
        dropout_fun: nn.Module = DropoutTransparent,
        softmax_fun: nn.Module = SoftmaxTransparent,
        layernorm_fun: nn.Module = LayerNormTransparent,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert batch_first, (
            "TransformerEncoderLayerTransparent works only for batch_first"
        )
        self.self_attn = MultiheadAttentionTransparent(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=att_dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_bias_q=add_bias_q,
            dropout_fun=dropout_fun,
            softmax_fun=softmax_fun,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = LinearTransparent(
            d_model, dim_feedforward, bias=bias, **factory_kwargs
        )
        self.dropout = dropout_fun(dropout)
        self.linear2 = LinearTransparent(
            dim_feedforward, d_model, bias=bias, **factory_kwargs
        )

        self.norm_first = norm_first
        self.norm1 = layernorm_fun(
            d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
        )
        self.norm2 = layernorm_fun(
            d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
        )
        self.dropout1 = dropout_fun(dropout)
        self.dropout2 = dropout_fun(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is nn.functional.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is nn.functional.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = nn.functional.relu

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        # ----- New arguments for verbose logging -----
        logger: Optional[ActivationLogger] = None,
        **kwargs,  # <- accept and ignore anything extra
    ) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
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
            # --- Log Initial Inputs ---
            if logger:
                logger.log("input_src", src, self)
                if src_mask is not None:
                    logger.log("input_src_mask", src_mask, self)

            # --- Main Layer Logic ---
            x = src

            if self.norm_first:
                # 1. LayerNorm
                norm1_out = self.norm1(x, logger=logger)
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
                )[0]

                sa_dropout_out = self.dropout(attention_out, logger=logger)

                x = x + sa_dropout_out  # First residual connection
                if logger:
                    logger.log("residual1_after_attn", x, self)

                # 3. LayerNorm
                norm2_out = self.norm2(x, logger=logger)

                # 4. Feed Forward (+ dropout) + Residual
                # Pass the log target to the helper block
                ff_out = self._ff_block(norm2_out, logger=logger)

                x = x + ff_out  # Second residual connection
                if logger:
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
                )[0]

                sa_dropout_out = self.dropout(attention_out, logger=logger)

                add1_out = x + sa_dropout_out  # Add before norm
                if logger:
                    logger.log("residual1_after_attn", add1_out, self)

                x = self.norm1(add1_out, logger=logger)  # First LayerNorm (post-add)

                # 2. Feed Forward (+ dropout) + Residual + LayerNorm
                # Pass the log target to the helper block
                ff_out = self._ff_block(x, logger=logger)

                add2_out = x + ff_out  # Add before norm
                if logger:
                    logger.log("residual2_after_ffn_output", add2_out, self)

                x = self.norm2(add2_out, logger=logger)  # Second LayerNorm (post-add)
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
        **kwargs,  # <- accept and ignore anything extra
    ) -> torch.Tensor:
        with ActivationLoggingScope(logger, "ffn"):
            linear1_out = self.linear1(x, logger=logger)
            act_out = self.activation(linear1_out)
            if logger:
                logger.log("activation_output", act_out, self)
            dropout_act_out = self.dropout1(act_out, logger=logger)
            linear2_out = self.linear2(dropout_act_out, logger=logger)
            ff_dropout_out = self.dropout2(linear2_out, logger=logger)
            if logger:
                logger.log("output", ff_dropout_out, self)  # This is the block output
            # The final output of this block is logged by the caller as "ff_block_out"
        return ff_dropout_out


def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")
