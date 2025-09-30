# FxPyTorch/transparent/trans_lstm.py
import torch
from torch import nn
from typing import Optional, Tuple
from .activation_logger import (
    ActivationLogger,
    ActivationLoggingScope,
)
from .trans_linear import LinearTransparent
from .trans_sigmoid import SigmoidTransparent
from .trans_tanh import TanhTransparent


class LSTMTransparent(nn.Module):
    """
    A single-layer, unidirectional LSTM (batch_first only),
    implemented transparently with LinearTransparent + activations so we can log gates.
    Matches nn.LSTM(batch_first=True, num_layers=1, bidirectional=False) I/O shapes.

    Forward signature mirrors torch.nn.LSTM: (input, hx=None) -> (output, (hn, cn))
    but also accepts logger= and **kwargs (ignored).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,  # kept for API parity (no effect for single layer)
        bidirectional: bool = False,
        device=None,
        dtype=None,
        sigmoid_fun: nn.Module = SigmoidTransparent,
        tanh_fun: nn.Module = TanhTransparent,
    ) -> None:
        super().__init__()
        assert batch_first, (
            f"{self.__class__.__name__} works only with batch_first=True"
        )
        assert not bidirectional, (
            f"{self.__class__.__name__} does not support bidirectional=True"
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first

        factory = {"device": device, "dtype": dtype}
        # input→4H and hidden→4H linear projections (separate biases like nn.LSTM)
        self.w_ih = LinearTransparent(
            input_size, 4 * hidden_size, bias=bias, device=device, dtype=dtype
        )
        self.w_hh = LinearTransparent(
            hidden_size, 4 * hidden_size, bias=bias, device=device, dtype=dtype
        )

        # gate activations
        self.sigmoid_i = sigmoid_fun()
        self.sigmoid_f = sigmoid_fun()
        self.sigmoid_o = sigmoid_fun()
        self.tanh_g = tanh_fun()
        self.tanh_ct = tanh_fun()

    def forward(
        self,
        input: torch.Tensor,  # (B, T, input_size)
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        logger: Optional[ActivationLogger] = None,
        **kwargs,  # <- accept and ignore anything extra
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert input.dim() == 3, "input must be (B, T, input_size)"
        B, T, _ = input.shape

        with ActivationLoggingScope(logger, self):
            if hx is None:
                h_t = input.new_zeros(B, self.hidden_size)
                c_t = input.new_zeros(B, self.hidden_size)
            else:
                h_t, c_t = hx
                assert h_t.shape == (B, self.hidden_size)
                assert c_t.shape == (B, self.hidden_size)

            outputs = []
            if logger:
                logger.log("input", input, self)
                logger.log("h0", h_t, self)
                logger.log("c0", c_t, self)

            for t in range(T):
                x_t = input[:, t, :]  # (B, input_size)
                # pre-activations
                gates = self.w_ih(x_t, logger=logger) + self.w_hh(h_t, logger=logger)
                i_lin, f_lin, g_lin, o_lin = gates.chunk(4, dim=-1)

                # activations
                i = self.sigmoid_i(i_lin, logger=logger)
                f = self.sigmoid_f(f_lin, logger=logger)
                g = self.tanh_g(g_lin, logger=logger)
                o = self.sigmoid_o(o_lin, logger=logger)

                # cell / hidden updates
                f_dot_c_t = f * c_t
                i_dot_g = i * g
                c_t = f_dot_c_t + i_dot_g
                h_t = o * self.tanh_ct(c_t, logger=logger)

                if logger:
                    with ActivationLoggingScope(logger, f"t{t}"):
                        logger.log("x_t", x_t, self)
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

                outputs.append(h_t.unsqueeze(1))  # (B,1,H)

            output = torch.cat(outputs, dim=1)  # (B, T, H)
            hn, cn = h_t.unsqueeze(0), c_t.unsqueeze(0)  # (1,B,H)

            if logger:
                logger.log("output", output, self)
                logger.log("hn", hn, self)
                logger.log("cn", cn, self)

        return output, (hn, cn)
