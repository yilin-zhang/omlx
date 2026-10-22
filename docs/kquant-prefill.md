# Prepared IQ/K Qwen prefill batching

The prepared IQ/K loader installs `KQuantPrefillMLP` after loading and evaluating
weights. It delegates gate/up batching to mlx-kquant's `KQuantProjectionPair`;
activation and down projection remain unchanged. No attention splitting is
installed by this integration.

The default is enabled only on M5 Ultra, for the measured dense Qwen geometry
(5120 input, 17408 intermediate), matching IQ3_XXS/IQ3_S/IQ4_XS pairs. Calls must
have shape `(1, 2048, 5120)` and BF16 dtype. Other shapes, training mode, mixed
codecs, and replaced projections use original operations. Small decode/MTP
calls do not batch gate/up.

Set `OMLX_KQUANT_PREFILL_FUSION=0` before starting the server to disable it and
avoid the additional weight cache. The prepared 27B checkpoint installs 24 pairs
and consumes 1.686 GiB extra GPU memory. MLX allocation accounting includes this;
the original model parameter names and checkpoint data remain unchanged. These
are immutable inference caches: recreate them after in-place weight mutation.
A different/older mlx-kquant installation lacking `projection_pair` must be
updated alongside this integration.

Validation on 2026-10-19, after rebasing onto oMLX upstream `9f2029ae`:

- Real model loader installed 24 pairs; parameter names unchanged.
- All 72 full-MLP comparisons (24 layers x 1/65/2048 rows) matched exactly.
- 8192-token core prefill ABBA: baseline 5.680/5.678 s, fused 4.942/4.939 s.
  Last full-block hidden states and final-position logits matched exactly,
  including four cached decode steps and a 65-token verify-shaped call.
- Isolated HTTP server: 9024-token prompt, repeat prefix-cache hit (9017 tokens),
  continuation, and 128-token generation with 40 MTP cycles succeeded.

Core timings exclude HTTP, loading and tokenization. This validation is not a
broad model quality evaluation or a fresh 128K performance measurement. The
existing production server was not restarted.
