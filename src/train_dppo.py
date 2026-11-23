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

import torch

from src.dppo.trainer import DPPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(p: str) -> Path:
    """Absolute paths are used as-is; relative paths are tried from CWD, then the project root."""
    path = Path(p).expanduser()
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recovery-Aware DPPO Fine-Tuning on Robomimic Lift")
    parser.add_argument("--config", type=str, default="config/train_dppo.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Pretrained robomimic checkpoint, or a DPPO .pt to evaluate")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from a dppo_last.pt (restores optimizers, RNG, best-so-far)")
    parser.add_argument("--iterations", type=int, default=None, help="Override total training iterations")
    parser.add_argument("--episodes-per-iter", type=int, default=None, help="Override rollouts per iteration")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'cuda:1' or 'cpu')")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    parser.add_argument(
        "--eval-episodes", type=int, default=None,
        help="Episodes PER REGIME for evaluation (default: rl.eval_episodes_per_regime). "
             "Uses rl.eval_seed, so the pretrained and fine-tuned checkpoints see identical scenarios.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = resolve_path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Make save_dir independent of the launch directory and mirror logs to a file
    weights_cfg = cfg.setdefault("weights", {})
    save_dir = resolve_path(weights_cfg.get("save_dir", "results/dppo_checkpoints"))
    if not save_dir.is_absolute():
        save_dir = PROJECT_ROOT / save_dir
    weights_cfg["save_dir"] = str(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_name = "eval.log" if args.eval_only else "train.log"
    fh = logging.FileHandler(save_dir / log_name)
    fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(fh)
    logger.info(f"Logging to {save_dir / log_name}")

    # Checkpoint path. --checkpoint may be either a robomimic checkpoint (used as
    # the pretrained θ_0) or a DPPO checkpoint written by DPPOTrainer.save_checkpoint,
    # which only holds state dicts and must be loaded on top of its base checkpoint.
    config_ckpt = weights_cfg.get("pretrained_checkpoint")
    config_ckpt = str(resolve_path(config_ckpt)) if config_ckpt else None

    dppo_weights_path = None
    ckpt_path = config_ckpt
    dppo_path = args.resume or args.checkpoint
    if dppo_path is not None:
        dppo_path = str(resolve_path(dppo_path))
        loaded = torch.load(dppo_path, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict) and "policy_state_dict" in loaded:
            dppo_weights_path = dppo_path
            stored_base = loaded.get("base_checkpoint")
            # The stored path is from the machine that trained it; fall back to the
            # config when the checkpoint was copied to another machine.
            if stored_base and Path(stored_base).exists():
                ckpt_path = stored_base
            if not ckpt_path or not Path(ckpt_path).exists():
                raise FileNotFoundError(
                    f"Base robomimic checkpoint not found (stored: {stored_base}, config: {config_ckpt})"
                )
            logger.info(f"Detected DPPO checkpoint; using base robomimic checkpoint {ckpt_path}")
        elif args.resume:
            raise ValueError(f"--resume expects a DPPO checkpoint (dppo_last.pt), got {dppo_path}")
        else:
            ckpt_path = dppo_path
        del loaded

    if not ckpt_path or not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

    device = args.device or cfg.get("experiment", {}).get("device", "cuda")

    trainer = DPPOTrainer(
        checkpoint_path=ckpt_path,
        config=cfg,
        device=device,
    )

    if args.eval_only:
        if dppo_weights_path is not None:
            trainer.load_finetuned_weights(dppo_weights_path)
        logger.info("Executing evaluation benchmark...")
        eval_results = trainer.evaluate(n_per_regime=args.eval_episodes)
        print("\n" + "=" * 60)
        print("               DPPO EVALUATION BENCHMARK")
        print("=" * 60)
        for k, v in eval_results.items():
            unit = "" if k == "eval_episodes_per_regime" else "%"
            print(f"  {k:32s}: {v:6.2f}{unit}")
        print("=" * 60 + "\n")
        return

    if dppo_weights_path is not None and not args.resume:
        raise ValueError("To continue training from a DPPO checkpoint use --resume dppo_last.pt")

    total_iters = args.iterations or cfg.get("rl", {}).get("total_iterations", 100)
    episodes_per_iter = args.episodes_per_iter or cfg.get("rollout", {}).get("n_episodes", 20)

    results = trainer.train(
        total_iterations=total_iters,
        episodes_per_iter=episodes_per_iter,
        resume_from=args.resume,
    )

    logger.info("DPPO Training Complete!")
    logger.info(f"Best Checkpoint: {results['best_checkpoint']} (Score: {results['best_score']:.2f})")


if __name__ == "__main__":
    main()
