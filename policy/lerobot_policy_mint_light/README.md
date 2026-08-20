# LeRobot MINT-Light policy

This package registers the production MINT-Light 1.0 policy as `--policy.type=mint_light`.

```bash
python -m pip install -r requirements.txt
python -m pip install -e ./policy/lerobot_policy_mint_light
```

Training from scratch requires `--policy.vqvae_name_or_path` pointing to a compatible
MINT-30M/SDAT tokenizer checkpoint. Evaluation uses the standard `lerobot-eval` command.

By default, each tokenizer chunk contains 16 future actions
(`--policy.action_history_steps=0`). The policy consumes a single observation and does not
require an observation or action-history buffer.

Evaluation uses intent-weighted action chunk ensembling by default. It can be disabled with
`--policy.intent_ensemble=false`; `--policy.intent_ensemble_temperature` controls the softmax
temperature and defaults to `0.5`.

Autoregressive inference samples each VQ token from the top three codebook logits by default.
Set `--policy.sample_top_k=1` for greedy decoding or override it with another positive value up
to the configured codebook size.

MINT-Light 1.0 uses a frozen DINOv3 ViT-L/16 visual tower and a frozen SigLIP2
text tower that supplies both pooled conditioning and token-level cross-attention. Camera
identity and factorized 2D position embeddings preserve the two-view layout. The action trunk
is a 384-wide, ten-layer AdaLN-Zero Transformer with six 64-wide SDPA heads and a 1024-wide
SwiGLU FFN. Together with trainable modality projections and the prediction head, it has about
34M trainable parameters; frozen encoders, tokenizer, and EMA copies are excluded. The
`[1, 2, 4]` next-scale VQ tokenizer is unchanged; inference uses configurable top-k sampling.

Version 1.0 fixes the encoder stack and removes the experimental tiny and multi-tower visual
paths. Single-DINOv3 checkpoints produced by the 0.3 implementation remain compatible. Older
DINO+SigLIP visual checkpoints are intentionally unsupported and fail during configuration
validation. SDAT/VQ-VAE tokenizer configurations remain supported.

The first production run downloads the frozen DINOv3 vision and SigLIP2 text encoders. If the
environment uses a SOCKS proxy, install proxy support with `pip install "httpx[socks]"`.
