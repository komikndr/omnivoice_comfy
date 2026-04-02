from __future__ import annotations

import torch

from . import runtime


runtime.register_model_path()


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _audio_tuple(audio: dict | None) -> tuple[torch.Tensor, int] | None:
    if audio is None:
        return None

    waveform = audio["waveform"]
    if waveform.dim() == 3:
        waveform = waveform[0]
    elif waveform.dim() != 2:
        raise ValueError("Expected Comfy AUDIO waveform to have 2 or 3 dimensions.")

    return waveform, int(audio["sample_rate"])


class OmniVoiceLoader:
    @classmethod
    def INPUT_TYPES(cls):
        choices = runtime.list_model_files()
        return {
            "required": {
                "OmniVoice Model": (choices,),
                "Audio Tokenizer Model": (choices,),
            }
        }

    RETURN_TYPES = ("OMNIVOICE_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "audio/tts/OmniVoice"

    def load_model(self, **kwargs):
        model = runtime.load_model_handle(
            omnivoice_model_name=kwargs["OmniVoice Model"],
            audio_tokenizer_model_name=kwargs["Audio Tokenizer Model"],
            weight_dtype="default",
        )
        return (model,)


class OmniVoiceTTS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OMNIVOICE_MODEL",),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Hello from OmniVoice.",
                        "placeholder": "Text the model should speak.\n\nUse this for the final spoken content.\nFor voice cloning, this is the NEW sentence to generate, not the transcript of the reference clip.",
                        "tooltip": "The actual words OmniVoice should generate as speech. For voice cloning, put the target sentence here and put the transcript of the reference clip into ref_text.",
                    },
                ),
                "language": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "English",
                        "tooltip": "Language name for the generated speech, for example English or Chinese. Leave at English unless you are intentionally switching languages.",
                    },
                ),
                "instruct": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Comma separated: male, australian accent, low pitch",
                        "tooltip": "Optional speaking style or character direction. This affects delivery and tone, not the literal transcript.",
                    },
                ),
                "speed": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.25,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": "Playback/speaking speed target. 1.0 is normal, lower is slower, higher is faster.",
                    },
                ),
                "duration": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 300.0,
                        "step": 0.1,
                        "tooltip": "Optional target duration in seconds. Set to 0 to let OmniVoice estimate duration automatically from the text.",
                    },
                ),
                "num_step": (
                    "INT",
                    {
                        "default": 28,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                        "tooltip": "Iterative generation steps. The node default is 28 as the current quality baseline; higher values are slower and may or may not help further depending on the prompt.",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "Classifier-Free Guidance scale. The node default is 3.0 for the current tuned baseline; this pushes the result to follow the conditioning more strongly and can change timbre and articulation noticeably.",
                    },
                ),
                "t_shift": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Mask-schedule shift used by OmniVoice's iterative decoding. The node default is 1.0 as the current tuned baseline for this custom node.",
                    },
                ),
                "denoise": (
                    ("false", "true"),
                    {
                        "tooltip": "Adds OmniVoice's denoise control token during generation. The node default is disabled because that is currently giving better audio on this setup.",
                    },
                ),
                "preprocess_prompt": (
                    ("true", "false"),
                    {
                        "tooltip": "Preprocess the reference prompt before tokenization. This can trim long reference audio, remove silences, and normalize the prompt path for voice cloning.",
                    },
                ),
                "postprocess_output": (
                    ("true", "false"),
                    {
                        "tooltip": "Postprocess the generated audio after decoding. This can remove long silences, normalize level behavior, and add fade/padding to avoid abrupt starts or ends.",
                    },
                ),
            },
            "optional": {
                "ref_audio": ("AUDIO",),
                "ref_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Transcript of the reference audio used for voice cloning.\n\nThis must match what the reference clip actually says.",
                        "tooltip": "Required only for voice cloning. Enter the transcript of ref_audio, not the target text to generate.",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_audio"
    CATEGORY = "audio/tts/OmniVoice"

    def generate_audio(
        self,
        model,
        text,
        language,
        instruct,
        speed,
        duration,
        num_step,
        cfg,
        t_shift,
        denoise,
        preprocess_prompt,
        postprocess_output,
        ref_audio=None,
        ref_text="",
        unique_id=None,
    ):
        ref_audio_tuple = _audio_tuple(ref_audio)
        ref_text_value = _optional_text(ref_text)
        if ref_audio_tuple is not None and ref_text_value is None:
            raise ValueError("Voice cloning requires ref_text in omnivoice_comfy. Whisper auto-transcription is disabled.")

        model.model._comfy_progress_node_id = unique_id

        generated = (
            model.model.generate(
                text=text,
                language=_optional_text(language),
                ref_audio=ref_audio_tuple,
                ref_text=ref_text_value,
                instruct=_optional_text(instruct),
                speed=speed,
                duration=duration if duration > 0 else None,
                num_step=num_step,
                guidance_scale=cfg,
                t_shift=t_shift,
                denoise=denoise == "true",
                preprocess_prompt=preprocess_prompt == "true",
                postprocess_output=postprocess_output == "true",
            )[0]
            .detach()
            .cpu()
        )

        finite = torch.isfinite(generated)
        if generated.numel() > 0 and not bool(finite.all().item()):
            raise ValueError(
                "OmniVoice generated non-finite audio values. Try fp32, disable postprocess_output, or use a cleaner reference clip."
            )

        if generated.numel() == 0 or generated.shape[-1] == 0:
            raise ValueError(
                "OmniVoice returned empty audio. Try disabling postprocess_output, "
                "setting an explicit duration, or using a longer/cleaner reference clip."
            )

        return ({"waveform": generated.unsqueeze(0), "sample_rate": model.sample_rate},)


NODE_CLASS_MAPPINGS = {
    "OmniVoiceLoader": OmniVoiceLoader,
    "OmniVoiceTTS": OmniVoiceTTS,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniVoiceLoader": "OmniVoice Loader",
    "OmniVoiceTTS": "OmniVoice TTS",
}
