from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch

import comfy.model_management
import folder_paths

from .vendor.omnivoice.models import OmniVoice

MODEL_FOLDER_NAME = "omnivoice_tts"
MODEL_DIR = Path(folder_paths.models_dir) / "tts" / "omnivoice"
SNAPSHOT_ROOT = Path(folder_paths.get_temp_directory()) / "omnivoice_comfy"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
MODEL_ASSET_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)
AUDIO_TOKENIZER_ASSET_FILES = (
    "config.json",
    "preprocessor_config.json",
)
MISSING_MODEL_OPTION = "<no .safetensors found>"


@dataclass
class OmniVoiceModelHandle:
    model: OmniVoice
    snapshot_dir: str
    model_weights: str
    audio_tokenizer_weights: str
    device: str
    dtype: str
    keep_in_vram: bool

    @property
    def sample_rate(self) -> int:
        return int(self.model.sampling_rate)


def register_model_path() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    folder_paths.folder_names_and_paths[MODEL_FOLDER_NAME] = (
        [str(MODEL_DIR)],
        folder_paths.supported_pt_extensions,
    )
    folder_paths.filename_list_cache.pop(MODEL_FOLDER_NAME, None)


def list_model_files() -> list[str]:
    register_model_path()
    files = folder_paths.get_filename_list(MODEL_FOLDER_NAME)
    return files if files else [MISSING_MODEL_OPTION]


def load_model_handle(
    omnivoice_model_name: str,
    audio_tokenizer_model_name: str,
    weight_dtype: str = "default",
    keep_in_vram: bool = True,
) -> OmniVoiceModelHandle:
    model_path = _resolve_model_path(omnivoice_model_name)
    audio_tokenizer_path = _resolve_model_path(audio_tokenizer_model_name)
    snapshot_dir = build_runtime_snapshot(model_path, audio_tokenizer_path)

    device = comfy.model_management.text_encoder_device()
    dtype = _resolve_dtype(weight_dtype, device)

    model = OmniVoice.from_pretrained(
        str(snapshot_dir),
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    model = model.to(device=device, dtype=dtype)
    audio_tokenizer_device = torch.device("cpu") if str(device).startswith("mps") else device
    # Keep the Higgs audio tokenizer in fp32; half precision decode was
    # producing non-finite audio on this environment.
    model.audio_tokenizer = model.audio_tokenizer.to(
        device=audio_tokenizer_device,
        dtype=torch.float32,
    )
    model.eval()

    return OmniVoiceModelHandle(
        model=model,
        snapshot_dir=str(snapshot_dir),
        model_weights=model_path.name,
        audio_tokenizer_weights=audio_tokenizer_path.name,
        device=str(device),
        dtype=str(dtype).replace("torch.", ""),
        keep_in_vram=keep_in_vram,
    )


def prepare_model_for_inference(handle: OmniVoiceModelHandle) -> None:
    target_device = torch.device(handle.device)
    target_dtype = getattr(torch, handle.dtype)

    handle.model = handle.model.to(device=target_device, dtype=target_dtype)

    audio_tokenizer_device = torch.device("cpu") if str(target_device).startswith("mps") else target_device
    handle.model.audio_tokenizer = handle.model.audio_tokenizer.to(
        device=audio_tokenizer_device,
        dtype=torch.float32,
    )
    handle.model.eval()


def offload_model_to_cpu(handle: OmniVoiceModelHandle) -> None:
    handle.model = handle.model.to(device=torch.device("cpu"), dtype=torch.float32)
    handle.model.audio_tokenizer = handle.model.audio_tokenizer.to(device=torch.device("cpu"), dtype=torch.float32)
    comfy.model_management.soft_empty_cache()


def build_runtime_snapshot(model_path: Path, audio_tokenizer_path: Path) -> Path:
    SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    snapshot_dir = SNAPSHOT_ROOT / _snapshot_key(model_path, audio_tokenizer_path)
    ready_file = snapshot_dir / ".ready"

    if ready_file.exists():
        return snapshot_dir

    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)

    (snapshot_dir / "audio_tokenizer").mkdir(parents=True, exist_ok=True)

    for file_name in MODEL_ASSET_FILES:
        shutil.copy2(ASSETS_DIR / "model" / file_name, snapshot_dir / file_name)

    for file_name in AUDIO_TOKENIZER_ASSET_FILES:
        shutil.copy2(
            ASSETS_DIR / "audio_tokenizer" / file_name,
            snapshot_dir / "audio_tokenizer" / file_name,
        )

    _symlink(snapshot_dir / "model.safetensors", model_path)
    _symlink(
        snapshot_dir / "audio_tokenizer" / "model.safetensors",
        audio_tokenizer_path,
    )
    ready_file.write_text("ok\n", encoding="ascii")
    return snapshot_dir


def _resolve_model_path(model_name: str) -> Path:
    if model_name == MISSING_MODEL_OPTION:
        raise FileNotFoundError(f"No OmniVoice weights were found in {MODEL_DIR}. Add .safetensors files to that folder first.")
    return Path(folder_paths.get_full_path_or_raise(MODEL_FOLDER_NAME, model_name))


def _resolve_dtype(weight_dtype: str, device: torch.device) -> torch.dtype:
    if weight_dtype == "fp32":
        return torch.float32
    if weight_dtype == "fp16":
        return torch.float16
    if weight_dtype == "bf16":
        return torch.bfloat16

    # OmniVoice inference is numerically unstable on this setup when the main
    # model follows Comfy's usual CUDA text-encoder default (fp16). Use fp32 as
    # the custom node's default for reliable speech generation.
    return torch.float32


def _snapshot_key(model_path: Path, audio_tokenizer_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (model_path, audio_tokenizer_path):
        resolved = path.resolve()
        stat = resolved.stat()
        digest.update(str(resolved).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()[:16]


def _symlink(link_path: Path, target_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    try:
        os.symlink(target_path, link_path)
    except OSError as exc:
        raise RuntimeError(
            "Failed to create the OmniVoice runtime symlink. "
            "Copy fallback is disabled; please place a full HuggingFace-style "
            "OmniVoice folder instead."
        ) from exc
