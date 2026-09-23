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

Fix: keep the activation in fp32 through the per-channel multiply (promoting
the weight rather than demoting ``x``) and cast only the final result back to
the original dtype.

Two rules about dtype casts on this path
---------------------------------------
**1. Cast activations with ``t.to(torch.float32)``, never ``t.float()``.**
torch-spyre monkey-patches ``Tensor.to`` (``torch_spyre/_monkey_patch.py``) so
a same-device dtype change becomes ``spyre::to_dtype_d2d``: it converts
on-device, re-injects a sliced input's ``storage_offset`` in-graph, and gives
its output the *staggered* element arrangement. ``Tensor.float()`` is a
different method — it goes straight to ``aten::_to_copy``, skips the patch, and
breaks two things independently:

* Strided inputs die at lowering. Qwen3-style qk-norm hands RMSNorm a view
  *into* the fused qkv tensor (``k_by_head``, offset 256, stride
  ``[768, 128, 1]``). Unpatched, the cast falls through to
  ``spyre::copy_from_d2d``, whose ``_reoffset`` cannot take a non-contiguous
  graph input::

      LoweringException: NotImplementedError:
        target: spyre.copy_from_d2d.default
        args[0]: InputBuffer(..., torch.float16, size=[16, 2, 128],
                             stride=[768, 128, 1], offset=256)
        args[1]: InputBuffer(..., torch.float32, size=[16, 2, 128])

* It reintroduces the very mixed-EA rejection above. A ``.float()`` result is
  STANDARD-arranged, so the multiply pairs a STANDARD operand with the
  staggered activation — the same ``Unsupported``, operands swapped.

``SpyreLayerNorm`` still spells these ``.float()``; it survives only because
CLIP hands it contiguous full-width hidden states. Do not copy that spelling.

**2. Never call ``.to()`` on a weight — let aten promote it.** The patched
``Tensor.to`` is a Python function, so Dynamo traces into it and can only name
the tensor it sees through the bound method, producing a guard source like
``...['weight'].data.to.__self__``. Two norms in one block then collide::

    AssertionError: Guard failed on the same frame it was created.
    Guard fail reason: 1/0: Duplicate tensors found:
      ["self._modules['input_layernorm']._parameters['weight'].data.to.__self__",
       "self._modules['post_attention_layernorm']._parameters['weight'].data.to.__self__"]

Writing ``x * weight`` instead leaves the fp16->fp32 promotion to aten, which
converts inside the graph with the right arrangement and needs no guard on the
parameter. This is what ``SpyreGemmaRMSNorm`` already does.
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
    x = x.to(torch.float32)
    x_var = x if variance_size_override is None else x[..., :variance_size_override]
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + variance_epsilon)
    if weight is not None:
        x = x * weight
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
    x = x.to(torch.float32) + residual.to(torch.float32)
    residual = x.to(orig_dtype)
    x_var = x if variance_size_override is None else x[..., :variance_size_override]
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + variance_epsilon)
    if weight is not None:
        x = x * weight
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
