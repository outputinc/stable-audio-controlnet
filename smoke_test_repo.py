#!/usr/bin/env python3
"""
Local smoke test for this repo.

Two modes:
1) load-only checkpoint test
2) short inference generation test

Examples:
  python smoke_test_repo.py --load-only --ckpt checkpoints/model.ckpt
  python smoke_test_repo.py --ckpt checkpoints/model.ckpt --steps 8
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Optional


def _auto_find_checkpoint() -> Optional[Path]:
    candidates = []
    for base in (Path("checkpoints"), Path("ckpts")):
        if not base.exists():
            continue
        candidates.extend(base.rglob("*.ckpt"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _resolve_checkpoint(ckpt_arg: Optional[str]) -> Path:
    if ckpt_arg:
        ckpt_path = Path(ckpt_arg)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    auto = _auto_find_checkpoint()
    if auto is None:
        raise FileNotFoundError(
            "No checkpoint found in checkpoints/ or ckpts/. "
            "Pass --ckpt <path-to-ckpt>."
        )
    return auto


def _choose_device(device_arg: str, torch_module) -> str:
    if device_arg != "auto":
        return device_arg
    if torch_module.cuda.is_available():
        return "cuda"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repo smoke test")
    parser.add_argument(
        "--exp-cfg",
        type=str,
        default="train_musdb_controlnet_audio_large",
        help="exp config name from exp/*.yaml",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="checkpoint path (.ckpt). If omitted, auto-finds latest in checkpoints/ or ckpts/",
    )
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="only test model+checkpoint loading, no generation",
    )
    parser.add_argument(
        "--input-wav",
        type=str,
        default="res/track_musdb/input.wav",
        help="conditioning wav for generation test",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="in: bass, drums; out: guitar",
        help='conditioning prompt, e.g. "in: bass, drums; out: guitar"',
    )
    parser.add_argument(
        "--output-wav",
        type=str,
        default="out/smoke_output.wav",
        help="output wav path",
    )
    parser.add_argument("--steps", type=int, default=8, help="diffusion steps for smoke test")
    parser.add_argument("--seed", type=int, default=42, help="generation seed")
    parser.add_argument("--cfg-scale", type=float, default=7.0, help="CFG scale")
    parser.add_argument("--sigma-min", type=float, default=0.3, help="sampler sigma_min")
    parser.add_argument("--sigma-max", type=float, default=500.0, help="sampler sigma_max")
    parser.add_argument(
        "--sampler-type",
        type=str,
        default="dpmpp-3m-sde",
        help="sampler type used by stable_audio_tools",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="inference device",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="strict checkpoint loading (default is strict=False)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        ckpt_path = _resolve_checkpoint(args.ckpt)
    except Exception as exc:
        print(f"[error] {exc}")
        return 1

    # MPS fallback avoids unsupported-op hard failures on Apple Silicon.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    try:
        import hydra  # type: ignore[import-not-found]
        import torch
    except Exception as exc:
        print(f"[error] missing runtime dependency: {exc}")
        return 1

    device = _choose_device(args.device, torch)
    print(f"[info] using device: {device}")
    print(f"[info] using checkpoint: {ckpt_path}")
    print(f"[info] using exp config: {args.exp_cfg}")

    with hydra.initialize(config_path=".", version_base=None):
        cfg = hydra.compose(config_name="config", overrides=[f"exp={args.exp_cfg}"])

    # Preflight the target module import so failures are explicit.
    model_cfg = cfg["model"]
    target = model_cfg.get("_target_") if hasattr(model_cfg, "get") else None
    if isinstance(target, str) and "." in target:
        target_module = target.rsplit(".", 1)[0]
        try:
            importlib.import_module(target_module)
        except Exception as exc:
            err = str(exc)
            print(f"[error] Failed importing '{target_module}': {exc}")
            if "sndfile library not found" in err or "libsndfile.dylib" in err:
                print("[hint] Fix audio backend: brew install libsndfile && python -m pip install --upgrade soundfile")
            if "No module named 'soxr'" in err:
                print("[hint] Install missing dependency: python -m pip install soxr")
            if "T5EncoderModel" in err:
                print("[hint] Use compatible HF stack: python -m pip install 'transformers<5' soxr")
            return 1

    try:
        model = hydra.utils.instantiate(cfg["model"])
    except Exception as exc:
        chain = []
        cursor = exc
        seen = set()
        while cursor is not None and id(cursor) not in seen:
            seen.add(id(cursor))
            chain.append(str(cursor))
            cursor = cursor.__cause__ if cursor.__cause__ is not None else cursor.__context__
        err = "\n".join(chain)
        if "pkg_resources" in err:
            print("[error] Missing pkg_resources in current environment.")
            print("[hint] Install legacy setuptools provider: python -m pip install 'setuptools<81'")
            return 1
        print(f"[error] Failed to instantiate model: {exc}")
        return 1

    ckpt_obj = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt_obj["state_dict"] if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj else ckpt_obj
    missing, unexpected = model.load_state_dict(state_dict, strict=args.strict)

    print("[ok] checkpoint loaded")
    print(f"[info] missing keys: {len(missing)}")
    print(f"[info] unexpected keys: {len(unexpected)}")

    if args.load_only:
        print("[ok] load-only test passed")
        return 0

    try:
        import torchaudio
        from stable_audio_tools.inference.generation import generate_diffusion_cond  # type: ignore[import-not-found]
    except Exception as exc:
        print(f"[error] missing generation dependency: {exc}")
        return 1

    input_wav = Path(args.input_wav)
    if not input_wav.exists():
        print(f"[error] input wav not found: {input_wav}")
        return 1

    y, sr = torchaudio.load(str(input_wav))
    if y.ndim != 2:
        print(f"[error] unexpected wav tensor shape: {tuple(y.shape)}")
        return 1

    if y.shape[0] == 1:
        y = y.repeat(2, 1)
    elif y.shape[0] > 2:
        y = y[:2]
    y = y.unsqueeze(0)
    y = torch.clamp(y, -1.0, 1.0)

    model = model.to(device).eval()
    conditioning = [{
        "audio": y.to(device),
        "prompt": args.prompt,
        "seconds_start": 0.0,
        "seconds_total": y.shape[-1] / float(sr),
    }]

    with torch.no_grad():
        output = generate_diffusion_cond(
            model.model,
            seed=args.seed,
            batch_size=1,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            conditioning=conditioning,
            sample_size=y.shape[-1],
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            sampler_type=args.sampler_type,
            device=device,
        )

    output_wav = Path(args.output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    out = torch.clamp(output[0].detach().cpu(), -1.0, 1.0)
    torchaudio.save(str(output_wav), out, sample_rate=sr)
    print(f"[ok] wrote: {output_wav}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
