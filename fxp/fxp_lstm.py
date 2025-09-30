# FxPyTorch/fxp/fxp_lstm.py
import torch
from torch import nn
from typing import Optional, Tuple, Literal, Type, Union
from pydantic import Field

from ..transparent.activation_logger import (
    ActivationLogger,
    ActivationLoggingScope,
)
from ..transparent.trans_lstm import LSTMTransparent
from .symmetric_quant import QType, QConfig, apply_quantize
from .fxp_linear import FxPLinear, LinearQConfig
from .fxp_sigmoid import FxPSigmoid, SigmoidQConfig
from .fxp_tanh import FxPTanh, TanhQConfig
from .calibration import set_calibrated_activation_quant, CalibrationType


class LSTMQConfig(QConfig):
    layer_type: Literal["lstm"] = "lstm"

    # Inputs / initial states
    input: QType = Field(default_factory=QType)
    h0: QType = Field(default_factory=QType)
    c0: QType = Field(default_factory=QType)

    # Linear projections (nested configs)
    w_ih: LinearQConfig = Field(default_factory=LinearQConfig)  # x_t -> 4H
    w_hh: LinearQConfig = Field(default_factory=LinearQConfig)  # h_t -> 4H

    # Gate activations (modules get these q_configs)
    i_act: SigmoidQConfig = Field(default_factory=SigmoidQConfig)
    f_act: SigmoidQConfig = Field(default_factory=SigmoidQConfig)
    g_act: TanhQConfig = Field(default_factory=TanhQConfig)
    o_act: SigmoidQConfig = Field(default_factory=SigmoidQConfig)
    ct_tanh: TanhQConfig = Field(default_factory=TanhQConfig)  # tanh(c_t)

    # Explicit intermediate quant points
    i_dot_g: QType = Field(default_factory=QType)
    f_dot_c_t: QType = Field(default_factory=QType)

    # States / outputs
    c_t: QType = Field(default_factory=QType)
    h_t: QType = Field(default_factory=QType)
    output: QType = Field(default_factory=QType)


class FxPLSTM(LSTMTransparent):
    """
    Quantised LSTM (single layer, batch_first) with explicit quant points and calibration,
    mirroring the design used in FxPMultiheadAttention.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device=None,
        dtype=None,
        q_config: Optional[LSTMQConfig] = None,
        sigmoid_fun: Type[nn.Module] = FxPSigmoid,
        tanh_fun: Type[nn.Module] = FxPTanh,
    ) -> None:
        # Build transparent baseline (names of submodules are preserved)
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            device=device,
            dtype=dtype,
            sigmoid_fun=sigmoid_fun,
            tanh_fun=tanh_fun,
        )

        self._q_config = q_config
        if self._q_config is not None:
            # Swap projections for quantised ones
            self.w_ih = FxPLinear(
                input_size,
                4 * hidden_size,
                bias=bias,
                q_config=self._q_config.w_ih,
                device=device,
                dtype=dtype,
            )
            self.w_hh = FxPLinear(
                hidden_size,
                4 * hidden_size,
                bias=bias,
                q_config=self._q_config.w_hh,
                device=device,
                dtype=dtype,
            )
            # Per-gate activation modules with their q_configs
            self.sigmoid_i = sigmoid_fun(q_config=self._q_config.i_act)
            self.sigmoid_f = sigmoid_fun(q_config=self._q_config.f_act)
            self.sigmoid_o = sigmoid_fun(q_config=self._q_config.o_act)
            self.tanh_g = tanh_fun(q_config=self._q_config.g_act)
            self.tanh_ct = tanh_fun(q_config=self._q_config.ct_tanh)

    @property
    def q_config(self) -> Optional[LSTMQConfig]:
        return self._q_config

    @q_config.setter
    def q_config(self, new_q_config: LSTMQConfig):
        if self._q_config is None:
            self._q_config = new_q_config
        else:
            # copy top-level QTypes
            for name in (
                "input",
                "h0",
                "c0",
                "i_dot_g",
                "f_dot_c_t",
                "c_t",
                "h_t",
                "output",
            ):
                setattr(self._q_config, name, getattr(new_q_config, name))
        # re-wire nested configs via submodules’ setters
        self.w_ih.q_config = new_q_config.w_ih
        self.w_hh.q_config = new_q_config.w_hh
        self.sigmoid_i.q_config = new_q_config.i_act
        self.sigmoid_f.q_config = new_q_config.f_act
        self.sigmoid_o.q_config = new_q_config.o_act
        self.tanh_g.q_config = new_q_config.g_act
        self.tanh_ct.q_config = new_q_config.ct_tanh

    def forward(
        self,
        input: torch.Tensor,  # (B, T, input_size)
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        logger: Optional[ActivationLogger] = None,
        apply_ste: bool = True,
        calibrate: bool = False,
        calibration_type: Union[str, CalibrationType] = CalibrationType.NO_OVERFLOW,
        **kwargs,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Float path
        if self._q_config is None:
            return super().forward(input=input, hx=hx, logger=logger, **kwargs)

        assert input.dim() == 3, "input must be (B, T, input_size)"
        B, T, _ = input.shape

        with ActivationLoggingScope(logger, self):
            # Initial states
            if hx is None:
                h_t = input.new_zeros(B, self.hidden_size)
                c_t = input.new_zeros(B, self.hidden_size)
            else:
                h_t, c_t = hx

            # Calibrate + quantise initial states / input stream
            if calibrate:
                set_calibrated_activation_quant(
                    h_t, self._q_config.h0, calibration_type
                )
                set_calibrated_activation_quant(
                    c_t, self._q_config.c0, calibration_type
                )
                set_calibrated_activation_quant(
                    input, self._q_config.input, calibration_type
                )
            h_t = apply_quantize(h_t, self._q_config.h0, apply_ste)
            c_t = apply_quantize(c_t, self._q_config.c0, apply_ste)

            outputs = []
            if logger:
                logger.log("input", input, self)
                logger.log("h0_quant", h_t, self)
                logger.log("c0_quant", c_t, self)

            for t in range(T):
                x_t = input[:, t, :]
                x_q = apply_quantize(x_t, self._q_config.input, apply_ste)

                # Projections (FxPLinear handles its own inner quant + optional calibration)
                gates = self.w_ih(
                    x_q,
                    logger=logger,
                    apply_ste=apply_ste,
                    calibrate=calibrate,
                    calibration_type=calibration_type,
                ) + self.w_hh(
                    h_t,
                    logger=logger,
                    apply_ste=apply_ste,
                    calibrate=calibrate,
                    calibration_type=calibration_type,
                )
                i_lin, f_lin, g_lin, o_lin = gates.chunk(4, dim=-1)

                # Gate activations (FxP modules handle their own quant + optional calibration)
                i = self.sigmoid_i(
                    i_lin,
                    logger=logger,
                    apply_ste=apply_ste,
                    calibrate=calibrate,
                    calibration_type=calibration_type,
                )
                f = self.sigmoid_f(
                    f_lin,
                    logger=logger,
                    apply_ste=apply_ste,
                    calibrate=calibrate,
                    calibration_type=calibration_type,
                )
                g = self.tanh_g(
                    g_lin,
                    logger=logger,
                    apply_ste=apply_ste,
                    calibrate=calibrate,
                    calibration_type=calibration_type,
                )
                o = self.sigmoid_o(
                    o_lin,
                    logger=logger,
                    apply_ste=apply_ste,
                    calibrate=calibrate,
                    calibration_type=calibration_type,
                )

                # Explicit intermediates with their own QTypes
                f_dot_c_t = f * c_t
                if calibrate:
                    set_calibrated_activation_quant(
                        f_dot_c_t, self._q_config.f_dot_c_t, calibration_type
                    )
                f_dot_c_t = apply_quantize(
                    f_dot_c_t, self._q_config.f_dot_c_t, apply_ste
                )

                i_dot_g = i * g
                if calibrate:
                    set_calibrated_activation_quant(
                        i_dot_g, self._q_config.i_dot_g, calibration_type
                    )
                i_dot_g = apply_quantize(i_dot_g, self._q_config.i_dot_g, apply_ste)

                # Cell update + quant
                c_t = f_dot_c_t + i_dot_g
                if calibrate:
                    set_calibrated_activation_quant(
                        c_t, self._q_config.c_t, calibration_type
                    )
                c_t = apply_quantize(c_t, self._q_config.c_t, apply_ste)

                # Hidden update: h = o * tanh(c)
                ct_act = self.tanh_ct(
                    c_t,
                    logger=logger,
                    apply_ste=apply_ste,
                    calibrate=calibrate,
                    calibration_type=calibration_type,
                )
                h_t = o * ct_act
                if calibrate:
                    set_calibrated_activation_quant(
                        h_t, self._q_config.h_t, calibration_type
                    )
                h_t = apply_quantize(h_t, self._q_config.h_t, apply_ste)

                if logger:
                    with ActivationLoggingScope(logger, f"t{t}"):
                        logger.log("x_t", x_t, self)
                        logger.log("x_t_quant", x_q, self)
                        logger.log("i_lin", i_lin, self)
                        logger.log("f_lin", f_lin, self)
                        logger.log("g_lin", g_lin, self)
                        logger.log("o_lin", o_lin, self)
                        logger.log("i", i, self)
                        logger.log("f", f, self)
                        logger.log("g", g, self)
                        logger.log("o", o, self)
                        logger.log("f_dot_c_t", f_dot_c_t, self)
                        logger.log("i_dot_g", i_dot_g, self)
                        logger.log("c_t", c_t, self)
                        logger.log("h_t", h_t, self)

                outputs.append(h_t.unsqueeze(1))

            output = torch.cat(outputs, dim=1)
            if calibrate:
                set_calibrated_activation_quant(
                    output, self._q_config.output, calibration_type
                )
            output = apply_quantize(output, self._q_config.output, apply_ste)
            hn, cn = h_t.unsqueeze(0), c_t.unsqueeze(0)

            if logger:
                logger.log("output", output, self)
                logger.log("hn", hn, self)
                logger.log("cn", cn, self)

        return output, (hn, cn)

    # ---- utility methods (like FxPMultiheadAttention) ----
    def quantize_weights_bias(self) -> None:
        self.w_ih.quantize_weights_bias()
        self.w_hh.quantize_weights_bias()

    def set_high_precision_quant(
        self, same_wb: bool = False, same_fc: bool = False
    ) -> None:
        self.w_ih.set_high_precision_quant(same_wb)
        self.w_hh.set_high_precision_quant(same_wb)
        if same_fc:
            # unify total_bits for linears (weights & bias) like same_qkv
            max_tb_w = max(
                self.w_ih.q_config.weight.total_bits,
                self.w_hh.q_config.weight.total_bits,
            )
            self.w_ih.q_config.weight.total_bits = (
                self.w_hh.q_config.weight.total_bits
            ) = max_tb_w
            max_tb_b = max(
                self.w_ih.q_config.bias.total_bits, self.w_hh.q_config.bias.total_bits
            )
            self.w_ih.q_config.bias.total_bits = self.w_hh.q_config.bias.total_bits = (
                max_tb_b
            )

    def set_no_overflow_quant(
        self, same_wb: bool = False, same_fc: bool = False
    ) -> None:
        self.w_ih.set_no_overflow_quant(same_wb)
        self.w_hh.set_no_overflow_quant(same_wb)
        if same_fc:
            max_ib_w = max(
                self.w_ih.q_config.weight.integer_bits,
                self.w_hh.q_config.weight.integer_bits,
            )
            max_tb_w = max(
                self.w_ih.q_config.weight.total_bits,
                self.w_hh.q_config.weight.total_bits,
            )
            self.w_ih.q_config.weight.total_bits = (
                self.w_hh.q_config.weight.total_bits
            ) = max_tb_w
            self.w_ih.q_config.weight.fractional_bits = (
                self.w_hh.q_config.weight.fractional_bits
            ) = max_tb_w - max_ib_w
            max_ib_b = max(
                self.w_ih.q_config.bias.integer_bits,
                self.w_hh.q_config.bias.integer_bits,
            )
            max_tb_b = max(
                self.w_ih.q_config.bias.total_bits, self.w_hh.q_config.bias.total_bits
            )
            self.w_ih.q_config.bias.total_bits = self.w_hh.q_config.bias.total_bits = (
                max_tb_b
            )
            self.w_ih.q_config.bias.fractional_bits = (
                self.w_hh.q_config.bias.fractional_bits
            ) = max_tb_b - max_ib_b

    def set_min_mse_quant(self, depth: int = 10) -> None:
        self.w_ih.set_min_mse_quant(depth)
        self.w_hh.set_min_mse_quant(depth)
