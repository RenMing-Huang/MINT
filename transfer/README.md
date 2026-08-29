# One-Shot Transfer

Pass a local path or Hugging Face model ID for both the policy and VQ-VAE:

```bash
CHECKPOINT=huangrm/MINT-light-zero
VQVAE=huangrm/MINT-tokenizer-libero-130
```

Run one of the three transfer tests:

```bash
bash transfer/scripts/transfer_new_task.sh "$CHECKPOINT" "$VQVAE"
bash transfer/scripts/transfer_new_layout.sh "$CHECKPOINT" "$VQVAE"
bash transfer/scripts/transfer_extend_horizon.sh "$CHECKPOINT" "$VQVAE"
```

Use environment variables to override runtime options:

```bash
SEED=42 NUM_EPISODES=5 SAVE_VIDEO=1 \
bash transfer/scripts/transfer_new_task.sh "$CHECKPOINT" "$VQVAE"
```

Videos and evaluation artifacts are written under `outputs/transfer_*`.
