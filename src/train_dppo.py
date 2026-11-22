#!/usr/bin/env python
"""
CLI Entrypoint for Recovery-Aware DPPO Fine-Tuning.

Usage:
    python -m src.train_dppo --config config/train_dppo.yaml
    python -m src.train_dppo --iterations 50 --episodes-per-iter 20
    python -m src.train_dppo --eval-only --checkpoint results/dppo_checkpoints/dppo_best.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
import yaml
from pathlib import Path

from src.dppo.trainer import DPPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recovery-Aware DPPO Fine-Tuning on Robomimic Lift")
    parser.add_argument("--config", type=str, default="config/train_dppo.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Pretrained or evaluation checkpoint path")
    parser.add_argument("--iterations", type=int, default=None, help="Override total training iterations")
    parser.add_argument("--episodes-per-iter", type=int, default=None, help="Override rollouts per iteration")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Number of episodes for evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        # Fallback relative to project root
        config_path = Path(__file__).resolve().parent.parent / args.config

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Checkpoint path
    ckpt_path = args.checkpoint or cfg.get("weights", {}).get("pretrained_checkpoint")
    if not ckpt_path:
        raise ValueError("No checkpoint path provided via --checkpoint or config file.")

    device = args.device or cfg.get("experiment", {}).get("device", "cuda")

    trainer = DPPOTrainer(
        checkpoint_path=ckpt_path,
        config=cfg,
        device=device,
    )

    if args.eval_only:
        logger.info("Executing evaluation benchmark...")
        eval_results = trainer.evaluate(n_episodes=args.eval_episodes)
        print("\n" + "=" * 60)
        print("               DPPO EVALUATION BENCHMARK")
        print("=" * 60)
        for k, v in eval_results.items():
            print(f"  {k:32s}: {v:6.2f}%")
        print("=" * 60 + "\n")
        return

    total_iters = args.iterations or cfg.get("rl", {}).get("total_iterations", 100)
    episodes_per_iter = args.episodes_per_iter or cfg.get("rollout", {}).get("n_episodes", 20)

    results = trainer.train(
        total_iterations=total_iters,
        episodes_per_iter=episodes_per_iter,
    )

    logger.info("DPPO Training Complete!")
    logger.info(f"Best Checkpoint: {results['best_checkpoint']} (Score: {results['best_score']:.2f})")


if __name__ == "__main__":
    main()
