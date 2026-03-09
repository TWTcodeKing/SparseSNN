"""Hook management for sparse acceleration backends.

Provides utilities to register/remove forward hooks on Conv2d, Linear,
and attention modules without modifying model source files.
"""

import torch
import torch.nn as nn
from typing import Callable, Optional, Type


class SparseHookManager:
    """Manages forward hooks for sparse interception across model layers.

    Supports:
    - Forward pre-hooks (intercept inputs before layer computation)
    - Forward hooks (intercept inputs + outputs after computation)
    - Module-type-based registration (e.g., all Conv2d, all Linear)
    - Named module registration (e.g., specific attention modules)
    """

    def __init__(self):
        self._hooks = []

    def register_pre_hook(self, module: nn.Module, hook_fn: Callable,
                          name: str = '') -> None:
        """Register a forward pre-hook on a specific module."""
        handle = module.register_forward_pre_hook(hook_fn)
        self._hooks.append((handle, name))

    def register_hook(self, module: nn.Module, hook_fn: Callable,
                      name: str = '') -> None:
        """Register a forward hook on a specific module."""
        handle = module.register_forward_hook(hook_fn)
        self._hooks.append((handle, name))

    def register_on_type(self, model: nn.Module, module_type: Type[nn.Module],
                         hook_fn: Callable, pre: bool = False,
                         exclude_names: Optional[list] = None) -> int:
        """Register hooks on all modules of a given type.

        Args:
            model: Root model to search.
            module_type: e.g., nn.Conv2d, nn.Linear.
            hook_fn: Hook function. Receives (module, input) for pre-hooks
                     or (module, input, output) for forward hooks.
            pre: If True, register as pre-hook.
            exclude_names: List of module name patterns to skip.

        Returns:
            Number of hooks registered.
        """
        count = 0
        exclude = exclude_names or []
        for name, module in model.named_modules():
            if isinstance(module, module_type):
                if any(ex in name for ex in exclude):
                    continue
                if pre:
                    self.register_pre_hook(module, hook_fn, name=name)
                else:
                    self.register_hook(module, hook_fn, name=name)
                count += 1
        return count

    def register_on_class(self, model: nn.Module, class_name: str,
                          hook_fn: Callable, pre: bool = False) -> int:
        """Register hooks on modules matching a class name string.

        Useful for matching custom classes like 'SSA' without importing them.
        """
        count = 0
        for name, module in model.named_modules():
            if module.__class__.__name__ == class_name:
                if pre:
                    self.register_pre_hook(module, hook_fn, name=name)
                else:
                    self.register_hook(module, hook_fn, name=name)
                count += 1
        return count

    def remove_all(self) -> int:
        """Remove all registered hooks. Returns count removed."""
        count = len(self._hooks)
        for handle, _ in self._hooks:
            handle.remove()
        self._hooks.clear()
        return count

    @property
    def num_hooks(self) -> int:
        return len(self._hooks)

    def __repr__(self) -> str:
        return f"SparseHookManager({self.num_hooks} hooks)"


def monkey_patch_forward(module: nn.Module, new_forward: Callable,
                         ) -> Callable:
    """Replace a module's forward method, returning the original.

    Usage:
        original = monkey_patch_forward(ssa_module, sparse_ssa_forward)
        # ... later restore:
        ssa_module.forward = original
    """
    original = module.forward
    module.forward = new_forward
    return original
