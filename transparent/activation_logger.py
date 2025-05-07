# FxPyTorch/transparent/activation_logger.py

import torch
from torch import nn
from collections import OrderedDict
from typing import Dict, Any, Optional, List, Union
import os
import json


class ActivationLogger:
    """
    Handles the logging of intermediate activations within nn.Module forward passes.
    """

    def __init__(
        self,
        enabled: bool = False,
        store_full_tensors: bool = True,
        model: Optional[nn.Module] = None,
    ):
        """
        Initializes the logger.

        Args:
            enabled (bool): If True, logging is active.
            store_full_tensors (bool): If True, store the full tensor data (as list).
                                      If False, only store shape and dtype (memory efficient).
            model (nn.Module): if provided, used to map module instances → their attribute names
        """
        self.enabled = enabled
        self.store_full_tensors = store_full_tensors
        self.log_data: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._scope_stack: List[str] = []
        # build map from module instance → its “path” name
        self._module_names: Dict[nn.Module, str] = {}
        if model is not None:
            for name, module in model.named_modules():
                # Note: named_modules() yields "" → root module, we skip that
                if name:
                    self._module_names[module] = name

    def get_module_name(self, module: nn.Module) -> str:
        """
        Return the attribute‐path name for `module` if we saw it in named_modules(),
        otherwise fall back to its class name.
        """
        full = self._module_names.get(module)
        if full:
            return full.split(".")[-1]  # just the last segment
        return type(module).__name__

    def is_enabled(self) -> bool:
        """Checks if logging is currently enabled."""
        return self.enabled

    def _get_current_scope_prefix(self) -> str:
        """Gets the prefix based on the current scope stack."""
        return ".".join(self._scope_stack) + "." if self._scope_stack else ""

    def enter_scope(self, scope_name: str):
        """Enters a new logging scope (e.g., for a specific layer)."""
        if self.enabled:
            self._scope_stack.append(scope_name)

    def exit_scope(self):
        """Exits the current logging scope."""
        if self.enabled and self._scope_stack:
            self._scope_stack.pop()

    def log(
        self,
        name: str,
        tensor: Optional[torch.Tensor],
        layer_instance: Optional[nn.Module] = None,
    ):
        """
        Logs information about a tensor if logging is enabled.

        Args:
            name (str): A descriptive name for the tensor within the current scope
                        (e.g., 'input', 'output', 'attention_weights').
            tensor (Optional[torch.Tensor]): The tensor to log. If None, logs a placeholder.
            layer_instance (Optional[nn.Module]): The layer instance this log relates to (optional, for context).
        """
        if not self.enabled:
            return

        full_name = self._get_current_scope_prefix() + name

        if tensor is None:
            entry = {
                "value": None,
                "shape": "N/A",
                "dtype": "N/A",
                "device": "N/A",
                "layer_type": type(layer_instance).__name__
                if layer_instance
                else "N/A",
            }
        else:
            entry = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "layer_type": type(layer_instance).__name__
                if layer_instance
                else "N/A",
            }
            if self.store_full_tensors:
                try:
                    entry["value"] = tensor.detach().cpu().tolist()
                except Exception as e:
                    entry["value"] = f"Error serializing tensor: {e}"
            else:
                entry["value"] = "[Tensor data not stored]"  # Placeholder

        self.log_data[full_name] = entry

    def get_log_data(self) -> OrderedDict[str, Dict[str, Any]]:
        """Returns the collected log data."""
        return self.log_data

    def clear(self):
        """Clears the collected log data and resets the scope."""
        self.log_data = OrderedDict()
        self._scope_stack = []

    def save_to_json(self, path: str, clear: bool = True):
        """Saves the collected log data to a JSON file."""
        if not self.log_data:
            print("Warning: Log dictionary is empty. Skipping save.")
            return
        if not self.enabled:
            print(
                "Warning: Logger was disabled. Log data might be empty. Skipping save."
            )
            return

        try:
            json_dir = os.path.dirname(path)
            if json_dir and not os.path.exists(json_dir):
                os.makedirs(json_dir, exist_ok=True)
                print(f"Created directory: {json_dir}")

            with open(path, "w") as f:
                # Use a standard JSONEncoder, potential issues handled during logging
                json.dump(self.log_data, f, indent=4)
            print(f"Log data saved successfully to: {path}")

        except TypeError as e:
            print(
                f"Error saving log data to {path}: TypeError - JSON serialization failed. Details: {e}"
            )
        except Exception as e:
            print(f"Error saving log data to {path}: {e}")

        if clear:
            self.clear()


class ActivationLoggingScope:
    def __init__(self, logger, scope_key: Union[str, nn.Module]):
        self.logger = logger
        # if they passed an actual module, look up its attribute‐path name
        if self.logger is not None:
            if isinstance(scope_key, nn.Module):
                self.scope_name = logger.get_module_name(scope_key)
            else:
                self.scope_name = scope_key

    def __enter__(self):
        if self.logger:
            self.logger.enter_scope(self.scope_name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger:
            self.logger.exit_scope()
