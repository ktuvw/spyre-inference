# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Spyre OOT replacement for RMSNorm.

The upstream ``forward_native`` (used via ``forward_oot``) calls
``ir.ops.rms_norm``, which casts fp32-normalised ``x`` back to
``weight.dtype`` before the per-channel multiply::

    x = x.to(weight.dtype) * weight   # vllm/ir/ops/layernorm.py:20

When ``weight`` is fp16 and ``x`` was produced by an fp32 reduction, the
cast creates a mixed-EA (element-addressing) layout that Spyre's
``propagate_layouts`` pass cannot reconcile:

    Unsupported: Multi-arg pointwise with mixed EA: STANDARD input arg1_1
    must broadcast (device stick dimension size 1) to be compatible with a
    staggered EA …

Fix: keep both operands in fp32 through the multiply and cast only the
final result to the original dtype — matching the pattern established in
``SpyreLayerNorm`` (``custom_ops/layer_norm.py``) and ``SpyreGemmaRMSNorm``
(``custom_ops/gemma_rms_norm.py``).
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.transformers.fusers.rms_norm import TPAwareRMSNorm

from .lazy_compile import CompileOutermost, compile_when_outermost

logger = init_logger(__name__)


def _rms_norm_spyre(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    variance_epsilon: float,
    variance_size_override: int | None,
) -> torch.Tensor:
    """RMSNorm kernel that avoids the mixed-EA cast on Spyre.

    Identical to ``ir.ops.rms_norm`` except the per-channel multiply is
    performed in fp32 (weight promoted, not x demoted) so both operands
    share the same EA layout after the fp32 reduction.
    """
    orig_dtype = x.dtype
    x = x.float()
    x_var = x if variance_size_override is None else x[..., :variance_size_override]
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + variance_epsilon)
    if weight is not None:
        x = x * weight.float()
    return x.to(orig_dtype)


def _fused_add_rms_norm_spyre(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None,
    variance_epsilon: float,
    variance_size_override: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused add + RMSNorm kernel that avoids the mixed-EA cast on Spyre."""
    orig_dtype = x.dtype
    x = x.float() + residual.float()
    residual = x.to(orig_dtype)
    x_var = x if variance_size_override is None else x[..., :variance_size_override]
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + variance_epsilon)
    if weight is not None:
        x = x * weight.float()
    return x.to(orig_dtype), residual


@RMSNorm.register_oot(name="RMSNorm")
class SpyreRMSNorm(CompileOutermost, RMSNorm):
    """Out-of-tree RMSNorm for Spyre: avoids mixed-EA pointwise layout error."""

    @compile_when_outermost
    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        weight = self.weight.data if self.pass_weight else None
        if residual is None:
            return _rms_norm_spyre(x, weight, self.variance_epsilon, self.variance_size_override)
        weight_add = self.weight.data if self.pass_weight_add else None
        return _fused_add_rms_norm_spyre(
            x, residual, weight_add, self.variance_epsilon, self.variance_size_override
        )


# The norm fuser instantiates TPAwareRMSNorm; OOT dispatch keys on the
# concrete class name, so it needs its own entry.
@RMSNorm.register_oot(name="TPAwareRMSNorm")
class SpyreTPAwareRMSNorm(TPAwareRMSNorm, SpyreRMSNorm):
    """Spyre RMSNorm that reconstructs a TP-sharded input before normalising."""
