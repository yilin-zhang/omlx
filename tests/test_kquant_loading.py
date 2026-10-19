# SPDX-License-Identifier: Apache-2.0
"""Regression tests for prepared IQ/K VLM loading."""

import json

import mlx.core as mx
import numpy as np
import pytest

pytest.importorskip("mlx_kquant")

from mlx_kquant.nn import KQuantLinear

from omlx.patches import kquant
from omlx.utils.model_loading import maybe_load_custom_quantization


def test_dispatch_requires_vlm(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "kquant"}})
    )
    sentinel = (object(), object())
    monkeypatch.setattr(kquant, "load", lambda path: sentinel)
    assert maybe_load_custom_quantization(str(tmp_path), is_vlm=True) is sentinel
    with pytest.raises(ValueError, match="VLM loader"):
        maybe_load_custom_quantization(str(tmp_path), is_vlm=False)


@pytest.mark.parametrize("shape", [(512,), (3, 512), (2, 3, 512)])
def test_grouped_projection_uses_packed_column_order(shape, monkeypatch):
    # Exercise the production adapter against dense, independently permuted
    # columns. Capture its input at the kernel boundary, without a GPU kernel.
    rng = np.random.default_rng(42)
    x = rng.normal(size=shape).astype(np.float32)
    weight = rng.normal(size=(4, 512)).astype(np.float32)
    tiled_weight = weight.reshape(4, 2, 2, 128).swapaxes(1, 2).reshape(4, 512)
    monkeypatch.setattr(
        KQuantLinear, "__call__", lambda self, inputs: inputs @ mx.array(tiled_weight).T
    )
    linear = KQuantLinear(512, 4, False, "q4_k")
    adapter = kquant.GroupedInputKQuantLinear(
        linear, key_heads=2, value_heads=4, head_dim=128
    )
    assert adapter.weight is linear.weight
    actual = np.array(adapter(mx.array(x)))
    np.testing.assert_allclose(actual, x @ weight.T, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("heads", [0, 3])
def test_projection_rejects_invalid_head_geometry(heads):
    linear = KQuantLinear(512, 4, False, "q4_k")
    with pytest.raises(ValueError, match="head geometry"):
        kquant.GroupedInputKQuantLinear(
            linear, key_heads=heads, value_heads=4, head_dim=128
        )


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "llama", "quantization_config": {"layout": "qwen35_grouped_v1"}},
        {"model_type": "qwen3_5", "quantization_config": {"layout": "unknown"}},
    ],
)
def test_loader_rejects_unsupported_layout(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Unsupported kquant"):
        kquant.load(tmp_path)
