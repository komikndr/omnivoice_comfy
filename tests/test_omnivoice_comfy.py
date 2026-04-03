from pathlib import Path

from src.omnivoice_comfy import runtime
from src.omnivoice_comfy.nodes import OmniVoiceLoader, OmniVoiceTTS, _optional_text


def test_loader_metadata():
    assert OmniVoiceLoader.RETURN_TYPES == ("OMNIVOICE_MODEL",)
    assert OmniVoiceLoader.FUNCTION == "load_model"

    input_types = OmniVoiceLoader.INPUT_TYPES()
    assert input_types["required"]["Keep model in VRAM"][0] == "BOOLEAN"


def test_tts_metadata():
    assert OmniVoiceTTS.RETURN_TYPES == ("AUDIO",)
    assert OmniVoiceTTS.FUNCTION == "generate_audio"

    input_types = OmniVoiceTTS.INPUT_TYPES()
    assert input_types["required"]["denoise"][0] == "BOOLEAN"
    assert input_types["required"]["preprocess_prompt"][0] == "BOOLEAN"
    assert input_types["required"]["postprocess_output"][0] == "BOOLEAN"
    assert input_types["required"]["seed"][0] == "INT"


def test_optional_text():
    assert _optional_text("") is None
    assert _optional_text("  ") is None
    assert _optional_text("English") == "English"


def test_build_runtime_snapshot_symlinks(tmp_path, monkeypatch):
    assets_dir = tmp_path / "assets"
    (assets_dir / "model").mkdir(parents=True)
    (assets_dir / "audio_tokenizer").mkdir(parents=True)

    for file_name in runtime.MODEL_ASSET_FILES:
        (assets_dir / "model" / file_name).write_text(file_name, encoding="ascii")
    for file_name in runtime.AUDIO_TOKENIZER_ASSET_FILES:
        (assets_dir / "audio_tokenizer" / file_name).write_text(file_name, encoding="ascii")

    model_weights = tmp_path / "omnivoice.safetensors"
    audio_weights = tmp_path / "audio_tokenizer.safetensors"
    model_weights.write_bytes(b"model")
    audio_weights.write_bytes(b"audio")

    monkeypatch.setattr(runtime, "ASSETS_DIR", assets_dir)
    monkeypatch.setattr(runtime, "SNAPSHOT_ROOT", tmp_path / "snapshots")

    snapshot_dir = runtime.build_runtime_snapshot(model_weights, audio_weights)

    assert (snapshot_dir / "model.safetensors").is_symlink()
    assert (snapshot_dir / "audio_tokenizer" / "model.safetensors").is_symlink()
    assert Path(snapshot_dir / "model.safetensors").resolve() == model_weights.resolve()
