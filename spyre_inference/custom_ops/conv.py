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

"""Spyre-specific Conv2d implementation (Vision patch embed).

vLLM lowers a patch conv (kernel == stride, no padding) to an im2col + GEMM in
`Conv2dLayer._forward_mulmat`: the on-device `permute(0,2,3,1,4,5).reshape(...)`
becomes a `copy_from_d2d` whose source-stick expression, for patch grids whose
spatial size is coprime with the 64-wide stick, is a sub-stick `Mod(k*var, 32)`
that torch-spyre's restickify pass cannot lay out.

When running on-card with explicit tiled layouts (e.g. Pixtral/Ministral and
Granite-Vision 4.1 / SigLIP), we place the weight and input into tiled
SpyreTensorLayouts and compile F.conv2d.
For shapes not matching the tiled layout assumptions, we run the patch convolution
via F.conv2d on CPU and return the result on the target device, bypassing inductor
layout/unfold errors.
"""

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.conv import Conv2dLayer

from .utils import convert
from .lazy_compile import CompileOutermost, compile_when_outermost

logger = init_logger(__name__)


def _layouts_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """The layout tuples assume in-channels inside one 64-wide stick,
    and out-channels tiling into whole sticks. That holds for vision patch
    embeddings (Bx3xHxW, out_channels a multiple of 64) for both Pixtral and
    Granite-Vision / SigLIP, but not for convs in general.
    """
    if x.dim() != 4 or weight.dim() != 4:
        return False
    c = x.shape[1]
    return c <= 64 and weight.shape[0] % 64 == 0


def _weight_layout(weight: torch.Tensor):
    """SpyreTensorLayout for a conv weight (O, C, K1, K2), sticked on out-channels.
    device_size = [K2, K1, O//64, C, 64]; the 64-wide stick walks the out-channel
    dim (host stride C*K1*K2), so out-channels tile into O//64 sticks.
    """
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype

    o, c, k1, k2 = weight.shape
    assert o % 64 == 0, f"conv out_channels {o} must be a multiple of the 64-wide stick"
    return SpyreTensorLayout(
        [k2, k1, o // 64, c, 64],
        [1, k2, c * k1 * k2 * 64, k1 * k2, c * k1 * k2],
        get_device_dtype(weight.dtype),
    )


def _input_layout(x: torch.Tensor):
    """SpyreTensorLayout for a conv input (B, C, H, W), sticked on in-channels.
    device_size = [W, H, B, 1, 64]; the 64-wide stick walks the channel dim (host
    stride H*W), padding C up to a full stick. Spatial W/H and batch B are outer loops.
    """
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype

    b, c, h, w = x.shape
    assert c <= 64, f"conv in_channels {c} must fit in one 64-wide stick"
    return SpyreTensorLayout(
        [w, h, b, 1, 64],
        [1, w, c * h * w, c * h * w, h * w],
        get_device_dtype(x.dtype),
    )


@Conv2dLayer.register_oot(name="Conv2dLayer")
class SpyreConv2d(CompileOutermost, Conv2dLayer):
    """Out-of-tree (OOT) Conv2d for IBM's Spyre device.

    Runs `F.conv2d` on-card with explicit tiled layouts when supported (B >= 1,
    in_channels <= 64, out_channels % 64 == 0), or runs F.conv2d on CPU and transfers
    the output to the target device.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._w_dev: torch.Tensor | None = None

    @compile_when_outermost
    def _conv_native(self, x: torch.Tensor, w: torch.Tensor, bias) -> torch.Tensor:
        return F.conv2d(
            x,
            w,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def _weight_on_device(self) -> torch.Tensor:
        """Place the conv weight into its tiled layout once, then cache."""
        if self._w_dev is None:
            w_cpu = self.weight.detach().to("cpu")
            self._w_dev = w_cpu.to("spyre", device_layout=_weight_layout(w_cpu))
        return self._w_dev

    def _cpu_conv(self, x: torch.Tensor) -> torch.Tensor:
        """F.conv2d on CPU tensors — never touches the Spyre dispatch path."""
        x_cpu = convert(x, device="cpu")
        w_cpu = convert(self.weight, device="cpu")
        bias_cpu = convert(self.bias, device="cpu") if self.bias is not None else None
        out_cpu = F.conv2d(
            x_cpu,
            w_cpu,
            bias_cpu,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return convert(out_cpu, device=x.device)

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4
        if x.device.type != "spyre":
            # Not on-card: use _cpu_conv to avoid mixing x (CPU) with self.weight
            # (Spyre) in _forward_conv, which would route F.conv2d through
            # torch-spyre's aten.convolution eager dispatch → RecursionError.
            return self._cpu_conv(x)
        if not _layouts_supported(x, self.weight):
            # Shape is outside the tiled-layout assumptions; move to CPU to avoid
            # calling F.conv2d on a Spyre tensor with no registered eager kernel.
            logger.warning_once(
                "Spyre conv2d: shape %s (weight %s) outside the tiled-layout "
                "assumptions (in_channels <= 64, out_channels %% 64 == 0); "
                "falling back to CPU F.conv2d.",
                tuple(x.shape),
                tuple(self.weight.shape),
            )
            return self._cpu_conv(x)
        logger.info_once("Spyre conv2d: on-card F.conv2d with tiled layouts")
        x_cpu = x.to("cpu")
        x_dev = x_cpu.to(  # ty: ignore[no-matching-overload]
            "spyre", device_layout=_input_layout(x_cpu)
        )
        return self._conv_native(x_dev, self._weight_on_device(), self.bias)