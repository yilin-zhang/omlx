# SPDX-License-Identifier: Apache-2.0
"""Load prepared Qwen3.5-family VLM checkpoints with GGUF IQ/K wire bytes.

The converter restores grouped GDN rows without decoding quantized weights.
Only GDN output projections retain tiled input columns; permute activations at
that boundary. No changes to attention caches or speculative decoding are needed.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import mlx.core as mx

# This module is imported only after the custom-quantization dispatcher selects
# kquant, so the extension remains optional for other models.
try:
    from mlx_kquant.nn import KQuantLinear, install_kquant_modules
except ImportError as exc:
    raise ImportError(
        "Loading an IQ/K checkpoint requires mlx-kquant built against the "
        "installed MLX version. Install a compatible mlx-kquant build first."
    ) from exc


class GroupedInputKQuantLinear(KQuantLinear):
    """Adapt grouped GDN activations to packed GGUF output-projection columns."""

    def __init__(
        self,
        linear: KQuantLinear,
        *,
        key_heads: int,
        value_heads: int,
        head_dim: int,
    ) -> None:
        if key_heads <= 0 or value_heads % key_heads or head_dim <= 0:
            raise ValueError("Invalid GDN head geometry")
        super().__init__(
            value_heads * head_dim,
            linear.weight.shape[0],
            "bias" in linear,
            linear.kquant_type,
        )
        self.update(linear.parameters())
        self._key_heads = key_heads
        self._ratio = value_heads // key_heads
        self._head_dim = head_dim

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        x = x.reshape(*shape[:-1], self._key_heads, self._ratio, self._head_dim)
        x = mx.swapaxes(x, -3, -2).reshape(shape)
        return super().__call__(x)


def load(model_name: str | Path) -> tuple[Any, Any]:
    """Load a prepared Qwen VLM; reject layouts requiring another conversion."""
    from mlx_vlm import utils

    path = Path(model_name)
    raw = json.loads((path / "config.json").read_text())
    quant = raw["quantization_config"]
    if raw.get("model_type") != "qwen3_5" or quant.get("layout") != "qwen35_grouped_v1":
        raise ValueError("Unsupported kquant architecture or prepared tensor layout")
    per_tensor = quant["per_tensor"]
    if not per_tensor or any(k.startswith("vision_tower.") for k in per_tensor):
        raise ValueError(
            "Expected language IQ/K weights and an unquantized vision tower"
        )
    config = copy.deepcopy(raw)
    config.pop("quantization", None)
    config.pop("quantization_config", None)
    cls, _ = utils.get_model_and_args(config=config, model_path=path)
    args = cls.ModelConfig.from_dict(config)
    args = utils.update_module_configs(args, cls, config, ["text", "vision"])
    args = utils.apply_generation_config_defaults(args, config)
    model = cls.Model(args)
    replaced = install_kquant_modules(
        model, {k + ".weight": v for k, v in per_tensor.items()}
    )
    if replaced != len(per_tensor):
        raise ValueError(f"Installed {replaced}/{len(per_tensor)} IQ/K modules")

    text = config["text_config"]
    for layer in model.language_model.model.layers:
        if hasattr(layer, "linear_attn"):
            projection = layer.linear_attn.out_proj
            if not isinstance(projection, KQuantLinear):
                raise ValueError("Expected packed GDN output projection")
            layer.linear_attn.out_proj = GroupedInputKQuantLinear(
                projection,
                key_heads=text["linear_num_key_heads"],
                value_heads=text["linear_num_value_heads"],
                head_dim=text["linear_value_head_dim"],
            )

    index = json.loads((path / "model.safetensors.index.json").read_text())
    weights = {}
    for shard in sorted(set(index["weight_map"].values())):
        weights.update(mx.load(str(path / shard)))
    if set(weights) != set(index["weight_map"]):
        raise ValueError("Checkpoint index does not match stored tensors")
    # The converter emits runtime names/layout, including already-offset norms.
    model.load_weights(list(weights.items()), strict=True)
    # GGUF stores small norms/conv coefficients as F32; keep the activation
    # stream in BF16 like the existing oQ8e model. Integer wire bytes are intact.
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    model.eval()
    from .kquant_mlp import install as install_prefill_fusion

    install_prefill_fusion(model)
    processor = utils.load_processor(
        path,
        True,
        eos_token_ids=getattr(args, "eos_token_id", None),
        trust_remote_code=False,
    )
    image_processor = utils.load_image_processor(path, trust_remote_code=False)
    if image_processor is not None:
        processor.image_processor = image_processor
    return model, processor
