# transparent/trans_multiheadattention.py
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Callable, Optional, Union, Dict, List, Tuple
from collections import OrderedDict
import numbers
from torch.nn.parameter import Parameter
from .activation_logger import (
    ActivationLogger,
    ActivationLoggingScope,
)
import warnings
from .trans_linear import LinearTransparent
from .trans_dropout import DropoutTransparent
from .trans_softmax import SoftmaxTransparent


class MultiheadAttentionTransparent(nn.Module):
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
        dropout_fun: nn.Module = DropoutTransparent,
        softmax_fun: nn.Module = SoftmaxTransparent,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MultiheadAttentionTransparent, self).__init__()
        assert batch_first, f"{self.__class__.__name__} works only on batch_first"
        assert not add_zero_att, (
            f"{self.__class__.__name__} does not work with add_zero_att"
        )
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        # when self._qkv_same_embed_dim = True, "in_proj_weight" rather than "q,k,v_weight" and fast path calculation will be used in "nn.transformer", which should be avoided. This is why we force self._qkv_same_embed_dim = False.
        self._qkv_same_embed_dim = False

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        self.qlinear = LinearTransparent(
            embed_dim, embed_dim, bias=add_bias_q, **factory_kwargs
        )
        self.klinear = LinearTransparent(
            self.kdim, embed_dim, bias=add_bias_kv, **factory_kwargs
        )
        self.vlinear = LinearTransparent(
            self.vdim, embed_dim, bias=add_bias_kv, **factory_kwargs
        )

        self.out_proj = LinearTransparent(
            embed_dim, embed_dim, bias=bias, **factory_kwargs
        )
        self.dropout = dropout_fun(dropout)
        self.softmax = softmax_fun(dim=-1)
        # to avoid null pointers in Transformer.forward
        self.in_proj_weight = None
        self.in_proj_bias = None

    def load_state_dict(self, state_dict):
        r"""
        Loads module from previously saved state.

        Supports loading from both :class:`torch.nn.MultiheadAttention` and
        :class:`opacus.layers.dp_multihead_attention.DPMultiheadAttention`.

        Args:
            state_dict: Please refer to
                https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html.
        """
        if "in_proj_weight" in state_dict:
            qweight, kweight, vweight = state_dict["in_proj_weight"].chunk(3, dim=0)

            state_dict["qlinear.weight"] = qweight
            state_dict["klinear.weight"] = kweight
            state_dict["vlinear.weight"] = vweight
            del state_dict["in_proj_weight"]

        if "in_proj_bias" in state_dict:
            qbias, kbias, vbias = state_dict["in_proj_bias"].chunk(3, dim=0)

            state_dict["qlinear.bias"] = qbias
            state_dict["klinear.bias"] = kbias
            state_dict["vlinear.bias"] = vbias
            del state_dict["in_proj_bias"]

        if "bias_k" in state_dict:
            state_dict["seq_bias_k.bias"] = state_dict["bias_k"].squeeze()
            del state_dict["bias_k"]

        if "bias_v" in state_dict:
            state_dict["seq_bias_v.bias"] = state_dict["bias_v"].squeeze()
            del state_dict["bias_v"]

        if "q_proj_weight" in state_dict:
            state_dict["qlinear.weight"] = state_dict["q_proj_weight"]
            del state_dict["q_proj_weight"]

        if "k_proj_weight" in state_dict:
            state_dict["klinear.weight"] = state_dict["k_proj_weight"]
            del state_dict["k_proj_weight"]

        if "v_proj_weight" in state_dict:
            state_dict["vlinear.weight"] = state_dict["v_proj_weight"]
            del state_dict["v_proj_weight"]

        super(MultiheadAttentionTransparent, self).load_state_dict(state_dict)

    # flake8: noqa C901
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
        **kwargs,  # <- accept and ignore anything extra
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        # ----- Verbose Logging Setup -----
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
            # Log inputs if verbose
            if logger:
                logger.log("input_query", query, self)
                logger.log("input_key", key, self)
                logger.log("input_value", value, self)
                if attn_mask is not None:
                    logger.log("input_attn_mask", attn_mask, self)

            head_dim = (
                embed_dim // self.num_heads
            )  # Already checked divisibility in init
            scaling = float(head_dim) ** -0.5

            q = self.qlinear(query, logger=logger)
            k = self.klinear(key, logger=logger)
            v = self.vlinear(value, logger=logger)

            q = q * scaling
            # Log q after scaling
            if logger:
                logger.log("q_scaled", q, self)

            # Do this transpose for head calculation (make them (tgt_len, bsz, -1))
            q, k, v = [x.transpose(0, 1) for x in (q, k, v)]

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
            # Return to batch_first-like format but with heads merged into batch dim
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
            # Assert shape (original logic)
            assert list(attn_scores_raw.size()) == [
                bsz * self.num_heads,
                tgt_len,
                src_len,
            ]
            # Log attn_scores_raw before mask
            if logger:
                logger.log(
                    "attn_scores_raw_BxHxTxS",
                    attn_scores_raw.view(bsz, self.num_heads, tgt_len, src_len),
                    self,
                )

            # Apply Attention Mask (original logic)
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_scores_raw = attn_scores_raw.masked_fill_(
                        attn_mask, float("-inf")
                    )
                else:
                    # Ensure dtype matches for addition
                    attn_scores_raw = attn_scores_raw + attn_mask.to(
                        attn_scores_raw.dtype
                    )

                # Log attn_output_weights after mask
                if logger:
                    logger.log(
                        "attn_scores_masked_B,HxTxS",
                        attn_scores_raw.view(bsz, self.num_heads, tgt_len, src_len),
                        self,
                    )

            attn_weights_softmax = self.softmax(attn_scores_raw, logger=logger)
            # Log attn_output_weights after softmax
            if logger:
                logger.log(
                    "attn_weights_softmax_BxHxTxS",
                    attn_weights_softmax.view(bsz, self.num_heads, tgt_len, src_len),
                    self,
                )

            # Dropout (original logic - using the module)
            attn_weights_dropout = self.dropout(attn_weights_softmax, logger=logger)
            # Log attn_output_weights after dropout
            if logger:
                logger.log(
                    "attn_weights_dropout_BxHxTxS",
                    attn_weights_dropout.view(bsz, self.num_heads, tgt_len, src_len),
                    self,
                )

            attn_output = torch.bmm(attn_weights_dropout, v)
            assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, head_dim]
            # Log attn_output
            if logger:
                logger.log(
                    "attn_output_weighted_sum_BxHxTxD",
                    attn_output.view(bsz, self.num_heads, tgt_len, head_dim),
                    self,
                )

            # Reshape Output (original logic) - Creates 'concat_output' implicitly
            # (B*H, T, D) -> (B, T, E)
            # This tensor was called 'concat_output' in the request.
            concat_output = attn_output.contiguous().view(bsz, tgt_len, embed_dim)
            # Log concat_output
            if logger:
                logger.log("attn_output_reshaped_BxTxE", concat_output, self)
            # Final Projection (original logic)
            total_output = self.out_proj(concat_output, logger=logger)
            # Log total_output
            if logger:
                logger.log("output", total_output, self)

        # --- End of original calculation logic ---

        if need_weights:
            # average attention weights over heads
            attn_weights_softmax = attn_weights_softmax.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            return (
                total_output,
                attn_weights_softmax.sum(dim=1) / self.num_heads,
            )
        else:
            return (
                total_output,
                None,
            )
