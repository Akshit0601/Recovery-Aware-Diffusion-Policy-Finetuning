"""
Checkpoint loading shared by the evaluation scripts.

Two checkpoint formats exist in this project:

* robomimic ``.pth``: the full diffusion policy (config, normalization stats, nets, EMA nets).
* DPPO ``.pt`` (written by ``src.dppo.trainer.DPPOTrainer.save_checkpoint``): only the
  fine-tuned noise-prediction UNet (``policy_state_dict``) plus the critic. It must be
  loaded on top of the robomimic checkpoint it was fine-tuned from (``base_checkpoint``).

``load_policy_from_checkpoint`` accepts either and always returns a ready robomimic
``RolloutPolicy``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

import robomimic.utils.file_utils as FileUtils

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Pretrained robomimic checkpoint the DPPO runs fine-tune (repo-relative).
DEFAULT_BASE_CHECKPOINT = str(
    PROJECT_ROOT
    / "robomimic/diffusion_policy_trained_models/test/20261024164824/models"
    / "model_epoch_150_low_dim_v15_success_1.0.pth"
)


def resolve_path(path: str) -> str:
    """Absolute paths are kept; relative paths are tried from CWD, then from the repo root."""
    p = Path(os.path.expanduser(str(path)))
    if p.is_absolute() or p.exists():
        return str(p)
    return str(PROJECT_ROOT / p)


def _torch_load(path: str) -> Any:
    # robomimic pickles numpy objects into its checkpoints; PyTorch >= 2.6 defaults to
    # weights_only=True and rejects them. These are our own checkpoints.
    return torch.load(path, map_location="cpu", weights_only=False)


def is_dppo_checkpoint(ckpt: Any) -> bool:
    return isinstance(ckpt, dict) and "policy_state_dict" in ckpt


def describe_checkpoint(path: str, base_checkpoint: Optional[str] = None) -> Dict[str, Any]:
    """Return {'path', 'kind', 'base_checkpoint', 'iteration'} without building the policy."""
    path = resolve_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = _torch_load(path)
    if not is_dppo_checkpoint(ckpt):
        return {"path": path, "kind": "robomimic", "base_checkpoint": None, "iteration": None}
    return {
        "path": path,
        "kind": "dppo",
        "base_checkpoint": _resolve_base(ckpt, base_checkpoint),
        "iteration": ckpt.get("iteration"),
    }


def _resolve_base(dppo_ckpt: Dict[str, Any], base_checkpoint: Optional[str]) -> str:
    """
    Pick the robomimic checkpoint whose UNet the DPPO weights override:
      1. base_checkpoint, if given (both eval scripts default it to
         DEFAULT_BASE_CHECKPOINT, the epoch-150 model that is also the baseline),
      2. the path stored in the DPPO checkpoint, if it exists on this machine,
      3. DEFAULT_BASE_CHECKPOINT.
    A warning is printed when the stored path names a different file than the one used.
    """
    stored = dppo_ckpt.get("base_checkpoint")
    candidates = [base_checkpoint, stored, DEFAULT_BASE_CHECKPOINT]
    tried = []
    chosen = None
    for c in candidates:
        if not c:
            continue
        c = resolve_path(c)
        tried.append(c)
        if os.path.exists(c):
            chosen = c
            break
    if chosen is None:
        raise FileNotFoundError(
            "Base robomimic checkpoint for DPPO weights not found. Tried: "
            + ", ".join(tried)
            + ". Pass --base-checkpoint."
        )
    if stored and Path(stored).name != Path(chosen).name:
        print(
            f"WARNING: DPPO checkpoint was fine-tuned from '{Path(stored).name}' but its weights are "
            f"being loaded onto '{chosen}'. Pass --base-checkpoint if that is not intended."
        )
    return chosen


def load_policy_from_checkpoint(
    path: str,
    device: torch.device,
    base_checkpoint: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    Load a robomimic RolloutPolicy from a robomimic .pth or a DPPO .pt checkpoint.

    Returns:
        policy: robomimic RolloutPolicy (fine-tuned UNet applied for DPPO checkpoints)
        ckpt_dict: the robomimic checkpoint dict (needed to build the env)
        info: {'path', 'kind', 'base_checkpoint', 'iteration'}
    """
    path = resolve_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = _torch_load(path)

    if not is_dppo_checkpoint(ckpt):
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt, device=device, verbose=verbose)
        return policy, ckpt_dict, {"path": path, "kind": "robomimic", "base_checkpoint": None, "iteration": None}

    base = _resolve_base(ckpt, base_checkpoint)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_dict=_torch_load(base), device=device, verbose=verbose
    )
    algo = policy.policy
    state = ckpt["policy_state_dict"]
    # robomimic runs inference with the EMA nets when present. DPPO fine-tunes (and
    # by default starts from) that EMA UNet, so the fine-tuned weights replace it; the
    # raw nets are overwritten too so either code path uses the fine-tuned policy.
    if getattr(algo, "ema", None) is not None:
        algo.ema.averaged_model["policy"]["noise_pred_net"].load_state_dict(state)
    algo.nets["policy"]["noise_pred_net"].load_state_dict(state)
    algo.set_eval()

    info = {"path": path, "kind": "dppo", "base_checkpoint": base, "iteration": ckpt.get("iteration")}
    return policy, ckpt_dict, info


class CheckpointLoaderMixin:
    """
    Mixin for DiffusionPolicyRunner subclasses: makes ``load_policy`` accept DPPO .pt
    checkpoints as well as robomimic .pth checkpoints.

    Set ``self.base_checkpoint`` (optional) before setup() to override the base used
    for DPPO checkpoints whose stored base path does not exist on this machine.
    """

    base_checkpoint: Optional[str] = None
    checkpoint_info: Optional[Dict[str, Any]] = None

    def load_policy(self, force: bool = False):  # type: ignore[override]
        if self.policy is not None and not force:
            return self.policy, self.ckpt_dict or {}

        ckpt_path = self.weights_cfg.get("checkpoint_path")
        if not ckpt_path:
            raise ValueError("Missing 'checkpoint_path' under 'weights' in configuration.")

        policy, ckpt_dict, info = load_policy_from_checkpoint(
            ckpt_path,
            device=self.device,
            base_checkpoint=self.base_checkpoint,
            verbose=self.system_cfg.get("verbose", False),
        )
        print(f"\n{'=' * 70}\nLoaded {info['kind']} checkpoint: {info['path']}")
        if info["kind"] == "dppo":
            print(f"  DPPO iteration {info['iteration']} on base: {info['base_checkpoint']}")
        print("=" * 70)

        self.policy = policy
        self.ckpt_dict = ckpt_dict
        self.checkpoint_info = info
        return policy, ckpt_dict
