"""Connect mlx-kquant projection batching to the measured Qwen dense MLPs."""

import logging
import os

import mlx.core as mx
import mlx.nn as nn
from mlx_kquant.codec_geometry import bytes_per_row
from mlx_kquant.projection_pair import KQuantProjectionPair
from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP, swiglu

logger = logging.getLogger(__name__)


class KQuantPrefillMLP(Qwen3_5MLP):
    def __init__(self, original):
        nn.Module.__init__(self)
        # Preserve parameter names and direct gate/up/down access for MTP.
        dict.update(self, original)
        self._pair = KQuantProjectionPair(self.gate_proj, self.up_proj)
        self.eval()

    def __call__(self, x):
        if (
            self.training
            or x.shape != (1, 2048, 5120)
            or os.environ.get("OMLX_KQUANT_PREFILL_FUSION", "1") == "0"
        ):
            return super().__call__(x)
        gate, up = self._pair(x, self.gate_proj, self.up_proj)
        return self.down_proj(swiglu(gate, up))


def install(model):
    """Install only the hardware, codecs and geometry measured locally.

    Set OMLX_KQUANT_PREFILL_FUSION=0 before loading to avoid packed copies.
    The copies are excluded from checkpoint parameters, but are included in
    MLX's active GPU allocation accounting.
    """
    if os.environ.get("OMLX_KQUANT_PREFILL_FUSION", "1") == "0":
        return []
    if not mx.metal.is_available() or "M5 Ultra" not in mx.device_info()["device_name"]:
        return []
    wrappers = []
    for layer in model.language_model.model.layers:
        original = layer.mlp
        if type(original) is not Qwen3_5MLP:
            continue
        gate, up = original.gate_proj, original.up_proj
        if not KQuantProjectionPair.compatible(gate, up):
            continue
        if (
            gate.kquant_type not in {"iq3_xxs", "iq3_s", "iq4_xs"}
            or gate.weight.shape != up.weight.shape
            or gate.weight.shape[0] != 17408
            or gate.weight.shape[1] != bytes_per_row(gate.kquant_type, 5120)
            or original.down_proj.weight.shape[0] != 5120
        ):
            continue
        layer.mlp = KQuantPrefillMLP(original)
        wrappers.append(layer.mlp)
    logger.info(
        "KQuant prefill fusion: %d MLPs, %.3f GiB runtime copies",
        len(wrappers),
        sum(w._pair.nbytes for w in wrappers) / 2**30,
    )
    return wrappers
