# candle-flashfftconv

Rust/[candle](https://github.com/huggingface/candle) port of HazyResearch's
[FlashFFTConv](https://github.com/HazyResearch/flash-fft-conv) kernels — the long-conv
operators ML stacks normally gate behind custom CUDA, reimplemented as candle
`CustomOp`s that run on **CPU** (a faithful scalar reference) and **Apple Metal**
(real fused kernels) from one call site. No CUDA, no torch, no Python.

Every kernel is the third link in a verified chain — nothing ported by resemblance:
**CUDA (truth) → MLX oracle (machine-precision-gated prototypes) → this crate**,
with a `metal == cpu` differential test on each op. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the full reference chain and design notes.

## Production kernel

`depthwise_conv1d` / `depthwise_conv1d_stream` — causal depthwise conv1d with
carried streaming state (`K−1` prior samples). This is the
[LFM2](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) backbone's short-conv
path in [candle-audio-rs](https://github.com/RustiestPorts/candle-audio-rs):
one kernel serves full prefill (zero left boundary), single-step decode, and
multi-token continuation chunks against a persistent cross-turn cache — chunked
forward is numerically equal to one full-sequence forward, which is what makes
suffix-only prefill possible. bf16 on Metal (f32-accumulate / bf16-store, the
deployed regime) and bf16/f32 on CPU.

```rust
use candle_flashfftconv::depthwise_conv1d_stream;

// x: (B, H, T) input, w: (H, K) depthwise taps, cache: Option<(B, H, K-1)>
let (y, new_cache) = depthwise_conv1d_stream(&x, &w, cache.as_ref())?;
```

## Experimental surface

The FlashFFTConv research zoo, ported and differentially tested but not used by a
production model path yet:

- `fused_fft_conv` / `fused_monarch` / `butterfly_*` — the Monarch-decomposed FFT
  long-conv family.
- `irfft`, `complex_mul_dd`, `FFTConvDd` — FFT plumbing including a
  **double-double** (~quad-precision from paired f32) toolkit for accumulating
  long FFTs on Metal, where f64 doesn't exist.
- `depthwise3_causal` — fused 3-tap depthwise variant.

APIs here may change or be trimmed; the production kernel above is stable.

## Features

- default: CPU reference implementations only.
- `metal`: compiles and dispatches the embedded MSL shaders
  (`src/metal/*.metal`) through candle's Metal device.

## Tests

```bash
cargo test                    # CPU references + naive-oracle differentials
cargo test --features metal   # + metal == cpu differential per op (needs a Metal device)
```

## Provenance & license

Apache-2.0 (see [LICENSE](LICENSE)). Ported from HazyResearch's FlashFFTConv
(Apache-2.0); Metal kernels adapted from the author's MLX `mx.fast.metal_kernel`
ports. Extracted from the EmberHarmony desktop voice stack, where the production
kernel runs every token of a shipping LFM2.5-Audio app.
