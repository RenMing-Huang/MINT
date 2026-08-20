# MINT policies for LeRobot

The directory contains two independent policy plugins for LeRobot 0.5.1:

- `mint`: the full MINT-4B vision-language-action policy.
- `mint_light`: the compact next-scale autoregressive policy derived from MINT-30M.

Install either or both plugins:

```bash
pip install -e ./policy/lerobot_policy_mint
pip install -e ./policy/lerobot_policy_mint_light
```

See the repository README for tokenizer setup and complete training and evaluation commands.
Both projects use the same `src/<package>/` layout and can be packaged independently.
