#!/usr/bin/env python
"""Run the one-shot MINT-Light intent-token transfer experiment on LIBERO."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
from collections import deque
import json
from pathlib import Path
import random

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from .demo import action_chunk, load_demonstration, resolve_demo_path
from .policy import MINTLightTransferPolicy
from .visualization import VideoWriter, render_trajectory_frame


def parse_levels(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    try:
        levels = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("force levels must be comma-separated integers") from exc
    if len(set(levels)) != len(levels):
        raise argparse.ArgumentTypeError("force levels must not contain duplicates")
    return levels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, help="Local MINT-Light model path or Hugging Face repo ID"
    )
    parser.add_argument(
        "--vqvae",
        default=None,
        help="Paired VQ-VAE .pth file or Hugging Face repo ID",
    )
    parser.add_argument("--suite", default="libero_90")
    parser.add_argument("--task-id", type=int, required=True, help="Target LIBERO task")
    parser.add_argument("--demo-suite", default=None)
    parser.add_argument("--demo-task-id", type=int, default=None)
    parser.add_argument("--demo-path", required=True, help="Extracted demonstration NPZ file")
    parser.add_argument("--mode", choices=("transfer", "policy", "replay"), default="transfer")
    parser.add_argument("--force-levels", type=parse_levels, default=(0,))
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--n-action-steps", type=int, default=4)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1223)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/transfer")
    parser.add_argument("--save-video", action="store_true")
    return parser


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_suite(name: str):
    from libero.libero import benchmark

    suites = benchmark.get_benchmark_dict()
    normalized = name.lower()
    if normalized not in suites:
        raise ValueError(f"Unknown LIBERO suite {name!r}; available: {sorted(suites)}")
    return suites[normalized]()


def make_episode_env(
    suite,
    suite_name: str,
    task_id: int,
    episode: int,
    max_steps: int | None,
    image_resolution: tuple[int, int],
):
    from lerobot.envs.libero import LiberoEnv

    image_height, image_width = image_resolution

    def factory():
        return LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            episode_length=max_steps,
            observation_height=image_height,
            observation_width=image_width,
            obs_type="pixels_agent_pos",
            episode_index=episode,
            init_states=True,
        )

    return gym.vector.SyncVectorEnv([factory])


def preprocess_observation(observation, instruction: str, env_preprocessor, runtime):
    from lerobot.envs.utils import preprocess_observation as to_lerobot_observation

    batch = to_lerobot_observation(observation)
    if runtime.config.use_language:
        batch["task"] = [instruction]
    batch = env_preprocessor(batch)
    return runtime.preprocessor(batch)


def current_eef_position(observation) -> np.ndarray:
    return np.asarray(observation["robot_state"]["eef"]["pos"][0]).copy()


def rollout_episode(
    args,
    episode: int,
    suite,
    task_instruction: str,
    demonstration,
    runtime: MINTLightTransferPolicy,
    env_preprocessor,
    output_dir: Path,
) -> None:
    env = make_episode_env(
        suite,
        args.suite.lower(),
        args.task_id,
        episode,
        args.max_steps,
        tuple(runtime.config.image_resolution),
    )
    observation, _ = env.reset(seed=args.seed + episode)
    runtime.reset()
    queued_actions: deque[torch.Tensor] = deque()
    executed_positions: list[np.ndarray] = []
    env_max_steps = int(env.call("_max_episode_steps")[0])
    requested_max_steps = args.max_steps or env_max_steps
    max_steps = min(requested_max_steps, env_max_steps, len(demonstration.actions))
    video = None
    stem = f"{args.suite}_task{args.task_id}_{demonstration.name}_ep{episode}_{args.mode}"
    if args.save_video:
        video = VideoWriter(output_dir / f"{stem}.mp4")

    try:
        for step in tqdm(range(max_steps), desc=f"episode {episode}", leave=False):
            executed_positions.append(current_eef_position(observation))
            if video is not None:
                video.append(
                    render_trajectory_frame(
                        env.envs[0],
                        demonstration.eef_positions,
                        np.asarray(executed_positions),
                    )
                )

            if not queued_actions:
                demo_chunk = action_chunk(
                    demonstration.actions, step, runtime.config.chunk_size
                )
                if args.mode == "replay" and demo_chunk is not None:
                    predicted = torch.from_numpy(demo_chunk).unsqueeze(0)
                    predicted = predicted[:, : runtime.config.n_action_steps]
                else:
                    batch = preprocess_observation(
                        observation, task_instruction, env_preprocessor, runtime
                    )
                    transfer_actions = None
                    if args.mode == "transfer" and demo_chunk is not None:
                        transfer_actions = torch.from_numpy(demo_chunk).unsqueeze(0)
                    output = runtime.predict(
                        batch,
                        transfer_actions,
                        force_levels=args.force_levels,
                        num_samples=args.num_samples,
                    )
                    predicted = runtime.ensemble(output)
                queued_actions.extend(predicted.transpose(0, 1))

            action = runtime.postprocessor(queued_actions.popleft())
            action_numpy = action.detach().cpu().numpy()
            observation, _, terminated, truncated, _ = env.step(action_numpy)
            if bool(terminated[0] or truncated[0]):
                break
    finally:
        if video is not None:
            video.close()
        env.close()


def main() -> None:
    args = build_parser().parse_args()
    if args.num_samples < 1 or args.num_episodes < 1 or args.n_action_steps < 1:
        raise ValueError("num-samples, num-episodes, and n-action-steps must be positive")
    seed_all(args.seed)

    target_suite = get_suite(args.suite)
    if args.task_id < 0 or args.task_id >= len(target_suite.tasks):
        raise IndexError(f"task-id {args.task_id} is outside target suite")
    demo_suite_name = args.demo_suite or args.suite
    demo_task_id = args.demo_task_id if args.demo_task_id is not None else args.task_id
    demo_suite = get_suite(demo_suite_name)
    if demo_task_id < 0 or demo_task_id >= len(demo_suite.tasks):
        raise IndexError(f"demo-task-id {demo_task_id} is outside demonstration suite")
    demo_path = resolve_demo_path(args.demo_path)
    demonstration = load_demonstration(demo_path)

    runtime = MINTLightTransferPolicy.from_pretrained(
        args.checkpoint,
        device=args.device,
        vqvae_name_or_path=args.vqvae,
        n_action_steps=args.n_action_steps,
    )
    from lerobot.processor.env_processor import LiberoProcessorStep
    from lerobot.processor.pipeline import PolicyProcessorPipeline

    env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    instruction = target_suite.get_task(args.task_id).language
    for episode in range(args.num_episodes):
        rollout_episode(
            args,
            episode,
            target_suite,
            instruction,
            demonstration,
            runtime,
            env_preprocessor,
            output_dir,
        )

    result = {
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "vqvae": runtime.config.vqvae_name_or_path,
        "target": {"suite": args.suite, "task_id": args.task_id},
        "demonstration": {
            "suite": demo_suite_name,
            "task_id": demo_task_id,
            "path": str(demo_path),
            "name": demonstration.name,
        },
        "mode": args.mode,
        "force_levels": list(args.force_levels),
        "demo_length": int(len(demonstration.actions)),
        "max_steps": args.max_steps,
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
