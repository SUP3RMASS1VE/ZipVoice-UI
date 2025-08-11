#!/usr/bin/env python3
import json
import logging
import os
import sys
import tempfile
import time
import gc
from pathlib import Path
from typing import Dict, Optional, Tuple

import gradio as gr
import torch

# Ensure project root is on sys.path when running this file directly
_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from zipvoice.bin.infer_zipvoice import (
    HUGGINGFACE_REPO,
    MODEL_DIR,
    generate_sentence,
    get_vocoder,
)
from huggingface_hub import hf_hub_download
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import (
    EmiliaTokenizer,
    EspeakTokenizer,
    LibriTTSTokenizer,
    SimpleTokenizer,
)
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from zipvoice.models.zipvoice_dialog import ZipVoiceDialog, ZipVoiceDialogStereo
from zipvoice.tokenizer.tokenizer import DialogTokenizer
from zipvoice.bin.infer_zipvoice_dialog import (
    generate_sentence as dialog_generate_sentence,
    generate_sentence_stereo as dialog_generate_sentence_stereo,
)
from faster_whisper import WhisperModel


class ZipVoiceService:
    def __init__(
        self,
        model_name: str = "zipvoice",
        model_dir: Optional[str] = None,
        tokenizer_name: str = "emilia",
        lang: str = "en-us",
        vocoder_path: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.model_dir = Path(model_dir) if model_dir else None
        self.tokenizer_name = tokenizer_name
        self.lang = lang
        self.vocoder_path = vocoder_path

        if torch.cuda.is_available():
            self.device = torch.device("cuda", 0)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = None
        self.vocoder = None
        self.tokenizer = None
        self.feature_extractor = None
        self.sampling_rate = 24000

        self._load_all()

    def _resolve_model_files(self) -> Tuple[Path, Path, Path]:
        if self.model_dir is not None:
            if not self.model_dir.is_dir():
                raise FileNotFoundError(f"{self.model_dir} does not exist")
            checkpoint = self.model_dir / "model.pt"
            model_json = self.model_dir / "model.json"
            tokens_txt = self.model_dir / "tokens.txt"
            for fp in [checkpoint, model_json, tokens_txt]:
                if not fp.is_file():
                    raise FileNotFoundError(f"{fp} does not exist")
            return checkpoint, model_json, tokens_txt

        # Download from HuggingFace
        checkpoint = Path(
            hf_hub_download(HUGGINGFACE_REPO, filename=f"{MODEL_DIR[self.model_name]}/model.pt")
        )
        model_json = Path(
            hf_hub_download(HUGGINGFACE_REPO, filename=f"{MODEL_DIR[self.model_name]}/model.json")
        )
        tokens_txt = Path(
            hf_hub_download(HUGGINGFACE_REPO, filename=f"{MODEL_DIR[self.model_name]}/tokens.txt")
        )
        return checkpoint, model_json, tokens_txt

    def _create_tokenizer(self, token_file: Path):
        if self.tokenizer_name == "emilia":
            return EmiliaTokenizer(token_file=str(token_file))
        if self.tokenizer_name == "libritts":
            return LibriTTSTokenizer(token_file=str(token_file))
        if self.tokenizer_name == "espeak":
            return EspeakTokenizer(token_file=str(token_file), lang=self.lang)
        assert self.tokenizer_name == "simple"
        return SimpleTokenizer(token_file=str(token_file))

    def _load_all(self) -> None:
        checkpoint, model_json, tokens_txt = self._resolve_model_files()

        with open(model_json, "r", encoding="utf-8") as f:
            config = json.load(f)

        tokenizer = self._create_tokenizer(tokens_txt)
        tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

        if self.model_name == "zipvoice":
            model = ZipVoice(**config["model"], **tokenizer_config)
        else:
            assert self.model_name == "zipvoice_distill"
            model = ZipVoiceDistill(**config["model"], **tokenizer_config)

        if str(checkpoint).endswith(".safetensors"):
            import safetensors.torch

            safetensors.torch.load_model(model, str(checkpoint))
        elif str(checkpoint).endswith(".pt"):
            load_checkpoint(filename=str(checkpoint), model=model, strict=True)
        else:
            raise NotImplementedError(f"Unsupported model checkpoint format: {checkpoint}")

        model = model.to(self.device).eval()
        vocoder = get_vocoder(self.vocoder_path).to(self.device).eval()

        if config["feature"]["type"] != "vocos":
            raise NotImplementedError(f"Unsupported feature type: {config['feature']['type']}")
        feature_extractor = VocosFbank()
        self.sampling_rate = int(config["feature"]["sampling_rate"])  # e.g., 24000

        self.model = model
        self.vocoder = vocoder
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    @torch.inference_mode()
    def synthesize(
        self,
        prompt_wav_path: str,
        prompt_text: str,
        text: str,
        num_step: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        speed: float = 1.0,
        t_shift: float = 0.5,
        target_rms: float = 0.1,
        feat_scale: float = 0.1,
    ) -> Tuple[str, Dict[str, float]]:
        # Model defaults
        if num_step is None or guidance_scale is None:
            if self.model_name == "zipvoice_distill":
                num_step = 8 if num_step is None else num_step
                guidance_scale = 3.0 if guidance_scale is None else guidance_scale
            else:
                num_step = 16 if num_step is None else num_step
                guidance_scale = 1.0 if guidance_scale is None else guidance_scale

        out_dir = Path(tempfile.mkdtemp(prefix="zipvoice_ui_"))
        out_path = out_dir / f"out_{int(time.time()*1000)}.wav"

        metrics = generate_sentence(
            save_path=str(out_path),
            prompt_text=prompt_text,
            prompt_wav=str(prompt_wav_path),
            text=text,
            model=self.model,
            vocoder=self.vocoder,
            tokenizer=self.tokenizer,  # type: ignore[arg-type]
            feature_extractor=self.feature_extractor,
            device=self.device,
            num_step=int(num_step),
            guidance_scale=float(guidance_scale),
            speed=float(speed),
            t_shift=float(t_shift),
            target_rms=float(target_rms),
            feat_scale=float(feat_scale),
            sampling_rate=self.sampling_rate,
        )
        return str(out_path), metrics


_service_cache: Dict[Tuple[str, Optional[str], str, Optional[str]], ZipVoiceService] = {}


def get_service(
    model_name: str,
    model_dir: Optional[str],
    tokenizer: str,
    lang: str,
    vocoder_path: Optional[str],
) -> ZipVoiceService:
    key = (model_name, model_dir, tokenizer, vocoder_path)
    if key not in _service_cache:
        logging.info(f"Initializing ZipVoiceService for {key} on device...")
        _service_cache[key] = ZipVoiceService(
            model_name=model_name,
            model_dir=model_dir,
            tokenizer_name=tokenizer,
            lang=lang,
            vocoder_path=vocoder_path,
        )
    return _service_cache[key]


class ZipVoiceDialogService:
    def __init__(
        self,
        model_name: str = "zipvoice_dialog",
        model_dir: Optional[str] = None,
        vocoder_path: Optional[str] = None,
    ) -> None:
        assert model_name in ("zipvoice_dialog", "zipvoice_dialog_stereo")
        self.model_name = model_name
        self.model_dir = Path(model_dir) if model_dir else None
        self.vocoder_path = vocoder_path

        if torch.cuda.is_available():
            self.device = torch.device("cuda", 0)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = None
        self.vocoder = None
        self.tokenizer = None
        self.feature_extractor = None
        self.sampling_rate = 24000

        self._load_all()

    def _resolve_model_files(self) -> Tuple[Path, Path, Path]:
        dialog_model_dir = {
            "zipvoice_dialog": "zipvoice_dialog",
            "zipvoice_dialog_stereo": "zipvoice_dialog_stereo",
        }
        if self.model_dir is not None:
            if not self.model_dir.is_dir():
                raise FileNotFoundError(f"{self.model_dir} does not exist")
            checkpoint = self.model_dir / "model.pt"
            model_json = self.model_dir / "model.json"
            tokens_txt = self.model_dir / "tokens.txt"
            for fp in [checkpoint, model_json, tokens_txt]:
                if not fp.is_file():
                    raise FileNotFoundError(f"{fp} does not exist")
            return checkpoint, model_json, tokens_txt

        checkpoint = Path(
            hf_hub_download(HUGGINGFACE_REPO, filename=f"{dialog_model_dir[self.model_name]}/model.pt")
        )
        model_json = Path(
            hf_hub_download(HUGGINGFACE_REPO, filename=f"{dialog_model_dir[self.model_name]}/model.json")
        )
        tokens_txt = Path(
            hf_hub_download(HUGGINGFACE_REPO, filename=f"{dialog_model_dir[self.model_name]}/tokens.txt")
        )
        return checkpoint, model_json, tokens_txt

    def _load_all(self) -> None:
        checkpoint, model_json, tokens_txt = self._resolve_model_files()

        with open(model_json, "r", encoding="utf-8") as f:
            config = json.load(f)

        tokenizer = DialogTokenizer(token_file=str(tokens_txt))
        tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

        if self.model_name == "zipvoice_dialog":
            model = ZipVoiceDialog(**config["model"], **tokenizer_config)
        else:
            model = ZipVoiceDialogStereo(**config["model"], **tokenizer_config)

        if str(checkpoint).endswith(".safetensors"):
            import safetensors.torch

            safetensors.torch.load_model(model, str(checkpoint))
        elif str(checkpoint).endswith(".pt"):
            load_checkpoint(filename=str(checkpoint), model=model, strict=True)
        else:
            raise NotImplementedError(f"Unsupported model checkpoint format: {checkpoint}")

        model = model.to(self.device).eval()
        vocoder = get_vocoder(self.vocoder_path).to(self.device).eval()

        if config["feature"]["type"] != "vocos":
            raise NotImplementedError(f"Unsupported feature type: {config['feature']['type']}")
        feature_extractor = VocosFbank()
        self.sampling_rate = int(config["feature"]["sampling_rate"])  # e.g., 24000

        self.model = model
        self.vocoder = vocoder
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    @torch.inference_mode()
    def synthesize(
        self,
        prompt_mode: str,  # "merged" | "split"
        prompt_wav_merged: Optional[str],
        prompt_text_merged: Optional[str],
        prompt_wav_s1: Optional[str],
        prompt_text_s1: Optional[str],
        prompt_wav_s2: Optional[str],
        prompt_text_s2: Optional[str],
        conversation_text: str,
        num_step: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        speed: float = 1.0,
        t_shift: float = 0.5,
        target_rms: float = 0.1,
        feat_scale: float = 0.1,
        silence_wav: Optional[str] = None,
    ) -> Tuple[str, Dict[str, float]]:
        if num_step is None:
            num_step = 16
        if guidance_scale is None:
            guidance_scale = 1.5

        if prompt_mode == "merged":
            assert prompt_wav_merged and prompt_text_merged
            prompt_wav = prompt_wav_merged
            prompt_text = prompt_text_merged
        else:
            assert prompt_wav_s1 and prompt_text_s1 and prompt_wav_s2 and prompt_text_s2
            prompt_wav = [prompt_wav_s1, prompt_wav_s2]
            prompt_text = f"[S1]{prompt_text_s1}[S2]{prompt_text_s2}"

        out_dir = Path(tempfile.mkdtemp(prefix="zipvoice_dialog_ui_"))
        out_path = out_dir / f"out_{int(time.time()*1000)}.wav"

        if self.model_name == "zipvoice_dialog_stereo":
            metrics = dialog_generate_sentence_stereo(
                save_path=str(out_path),
                prompt_text=prompt_text,
                prompt_wav=prompt_wav,
                text=conversation_text,
                model=self.model,  # type: ignore[arg-type]
                vocoder=self.vocoder,
                tokenizer=self.tokenizer,  # type: ignore[arg-type]
                feature_extractor=self.feature_extractor,
                device=self.device,
                num_step=int(num_step),
                guidance_scale=float(guidance_scale),
                speed=float(speed),
                t_shift=float(t_shift),
                target_rms=float(target_rms),
                feat_scale=float(feat_scale),
                sampling_rate=self.sampling_rate,
                silence_wav=silence_wav or str(Path("assets") / "silence.wav"),
            )
        else:
            metrics = dialog_generate_sentence(
                save_path=str(out_path),
                prompt_text=prompt_text,
                prompt_wav=prompt_wav,
                text=conversation_text,
                model=self.model,  # type: ignore[arg-type]
                vocoder=self.vocoder,
                tokenizer=self.tokenizer,  # type: ignore[arg-type]
                feature_extractor=self.feature_extractor,
                device=self.device,
                num_step=int(num_step),
                guidance_scale=float(guidance_scale),
                speed=float(speed),
                t_shift=float(t_shift),
                target_rms=float(target_rms),
                feat_scale=float(feat_scale),
                sampling_rate=self.sampling_rate,
            )

        return str(out_path), metrics


_dialog_service_cache: Dict[Tuple[str, Optional[str], Optional[str]], ZipVoiceDialogService] = {}


def get_dialog_service(
    model_name: str,
    model_dir: Optional[str],
    vocoder_path: Optional[str],
) -> ZipVoiceDialogService:
    key = (model_name, model_dir, vocoder_path)
    if key not in _dialog_service_cache:
        logging.info(f"Initializing ZipVoiceDialogService for {key} on device...")
        _dialog_service_cache[key] = ZipVoiceDialogService(
            model_name=model_name,
            model_dir=model_dir,
            vocoder_path=vocoder_path,
        )
    return _dialog_service_cache[key]


_whisper_cache: Dict[Tuple[str, str], WhisperModel] = {}


def get_whisper(model_size: str = "small", device: Optional[str] = None) -> WhisperModel:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (model_size, device)
    model = _whisper_cache.get(key)
    if model is None:
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _whisper_cache[key] = model
    return model


def transcribe_file(
    audio_path: str,
    model_size: str,
    lang: Optional[str],
    unload: bool = False,
) -> str:
    if not audio_path:
        return ""
    # Determine device the same way as get_whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_whisper(model_size, device=device)
    segments, _ = model.transcribe(audio_path, language=lang if lang else None)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    if unload:
        try:
            _whisper_cache.pop((model_size, device), None)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        finally:
            gc.collect()
    return text

def ui_infer_dialog(
    model_name: str,
    model_dir: Optional[str],
    vocoder_path: Optional[str],
    prompt_mode: str,
    prompt_wav_merged: Optional[str],
    prompt_text_merged: Optional[str],
    prompt_wav_s1: Optional[str],
    prompt_text_s1: Optional[str],
    prompt_wav_s2: Optional[str],
    prompt_text_s2: Optional[str],
    conversation_text: str,
    num_step: int,
    guidance_scale: float,
    speed: float,
    t_shift: float,
    target_rms: float,
    feat_scale: float,
    silence_wav: Optional[str],
):
    if not conversation_text or not conversation_text.startswith("[S1]"):
        raise gr.Error("Conversation text must start with [S1] and include speaker tags.")

    if prompt_mode == "merged":
        if not prompt_wav_merged or not prompt_text_merged:
            raise gr.Error("Merged mode requires a prompt wav and its transcription.")
    else:
        if not (prompt_wav_s1 and prompt_text_s1 and prompt_wav_s2 and prompt_text_s2):
            raise gr.Error("Split mode requires both speakers' wavs and transcriptions.")

    service = get_dialog_service(
        model_name=model_name,
        model_dir=model_dir if model_dir else None,
        vocoder_path=vocoder_path if vocoder_path else None,
    )

    out_path, metrics = service.synthesize(
        prompt_mode=prompt_mode,
        prompt_wav_merged=prompt_wav_merged,
        prompt_text_merged=prompt_text_merged,
        prompt_wav_s1=prompt_wav_s1,
        prompt_text_s1=prompt_text_s1,
        prompt_wav_s2=prompt_wav_s2,
        prompt_text_s2=prompt_text_s2,
        conversation_text=conversation_text,
        num_step=num_step,
        guidance_scale=guidance_scale,
        speed=speed,
        t_shift=t_shift,
        target_rms=target_rms,
        feat_scale=feat_scale,
        silence_wav=silence_wav,
    )

    pretty_metrics = (
        f"RTF: {metrics['rtf']:.4f} (no-vocoder: {metrics['rtf_no_vocoder']:.4f}, "
        f"vocoder: {metrics['rtf_vocoder']:.4f})\n"
        f"Time: {metrics['t']:.2f}s (no-vocoder: {metrics['t_no_vocoder']:.2f}s, "
        f"vocoder: {metrics['t_vocoder']:.2f}s)\n"
        f"Audio length: {metrics['wav_seconds']:.2f}s"
    )
    return out_path, pretty_metrics

def ui_infer(
    model_name: str,
    model_dir: Optional[str],
    tokenizer: str,
    lang: str,
    vocoder_path: Optional[str],
    prompt_wav_path: str,
    prompt_text: str,
    text: str,
    num_step: int,
    guidance_scale: float,
    speed: float,
    t_shift: float,
    target_rms: float,
    feat_scale: float,
):
    if not prompt_wav_path:
        raise gr.Error("Please upload a prompt wav file.")
    if not prompt_text:
        raise gr.Error("Please provide the transcription of the prompt wav.")
    if not text:
        raise gr.Error("Please enter the text to synthesize.")

    service = get_service(
        model_name=model_name,
        model_dir=model_dir if model_dir else None,
        tokenizer=tokenizer,
        lang=lang,
        vocoder_path=vocoder_path if vocoder_path else None,
    )

    out_path, metrics = service.synthesize(
        prompt_wav_path=prompt_wav_path,
        prompt_text=prompt_text,
        text=text,
        num_step=num_step,
        guidance_scale=guidance_scale,
        speed=speed,
        t_shift=t_shift,
        target_rms=target_rms,
        feat_scale=feat_scale,
    )

    pretty_metrics = (
        f"RTF: {metrics['rtf']:.4f} (no-vocoder: {metrics['rtf_no_vocoder']:.4f}, "
        f"vocoder: {metrics['rtf_vocoder']:.4f})\n"
        f"Time: {metrics['t']:.2f}s (no-vocoder: {metrics['t_no_vocoder']:.2f}s, "
        f"vocoder: {metrics['t_vocoder']:.2f}s)\n"
        f"Audio length: {metrics['wav_seconds']:.2f}s"
    )
    return out_path, pretty_metrics


def _make_theme():
    try:
        theme = gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="sky",
            neutral_hue="slate",
        ).set(
            body_background_fill="#0b1020",
            body_text_color="#e5e7eb",
            block_background_fill="#0f1629",
            block_border_width="1px",
            block_border_color="#232b46",
            block_shadow="0 8px 24px 0 rgba(124,58,237,.18)",
            button_primary_background_fill="#7c3aed",
            button_primary_background_fill_hover="#8b5cf6",
            button_primary_text_color="#ffffff",
            input_background_fill="#0f1629",
            input_border_color="#27304d",
            radius_md="14px",
            radius_lg="16px",
            font=["Inter", "ui-sans-serif", "system-ui"],
            font_mono=["JetBrains Mono", "ui-monospace", "SFMono-Regular"],
        )
    except Exception:
        theme = None
    return theme


_CSS = """
/* Background + overall look */
.gradio-container{background:linear-gradient(180deg,#0a0f1f 0%,#0a0e1a 40%,#0b1020 100%) fixed}
.gradio-container .markdown h3{color:#e5e7eb;font-weight:800;letter-spacing:.2px}
.gradio-container .tabs{background:rgba(255,255,255,0.02);border:1px solid #232b46;border-radius:14px;padding:6px}
.gradio-container .tabitem{padding-top:8px}

/* Cards / panels */
.gradio-container .gr-panel, .gradio-container .gr-group, .gradio-container .gr-accordion, .gradio-container .block{
  border-radius:16px!important;
  background:#0f1629!important;
  border:1px solid #232b46!important;
  box-shadow:0 8px 26px rgba(124,58,237,.15)!important;
}
.gradio-container .gr-accordion .label-wrap{background:#0f1629}

/* Inputs */
.gradio-container input, .gradio-container textarea, .gradio-container select{
  background:#0f172a!important; border:1px solid #27304d!important; color:#e5e7eb!important;
}
.gradio-container .wrap.svelte-drumf7:focus-within, .gradio-container .wrap:focus-within{
  box-shadow:0 0 0 2px #7c3aed33, 0 0 0 1px #7c3aed inset!important; border-radius:12px
}

/* Labels */
label, .gradio-container .label{text-transform:uppercase;letter-spacing:.03em;font-weight:700;color:#9aa7c7}

/* Buttons */
.gradio-container .gr-button{border:none;border-radius:12px}
.gradio-container .gr-button-primary{background:linear-gradient(90deg,#7c3aed 0%,#8b5cf6 100%); color:#fff}
.gradio-container .gr-button-primary:hover{filter:brightness(1.06); box-shadow:0 8px 20px rgba(139,92,246,.35)}
.gradio-container #cta-generate, .gradio-container #d-cta-generate{
  width:100%; height:44px; font-weight:800; letter-spacing:.02em; text-transform:uppercase
}

/* Sliders */
.gradio-container input[type=range]::-webkit-slider-thumb{background:#8b5cf6;border:2px solid #c4b5fd}
.gradio-container input[type=range]{accent-color:#8b5cf6}

/* Tabs active state */
.gradio-container .tab-nav button.svelte-1ipelgc[aria-selected="true"], .gradio-container .tabitem .tabitem-buttons .selected{
  background:#1a2140!important; color:#e8e8ff!important; box-shadow:0 10px 24px rgba(124,58,237,.18)
}

/* Audio blocks */
.gradio-container .audio{background:#0c1428;border:1px dashed #2a355a}
"""


def build_app() -> gr.Blocks:
    theme = _make_theme()
    with gr.Blocks(title="ZipVoice Web UI", theme=theme, css=_CSS) as demo:
        gr.Markdown(
            """
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
              <div style="width:10px;height:10px;border-radius:50%;background:#22d3ee;box-shadow:0 0 18px #22d3ee"></div>
              <h3 style="margin:0">ZipVoice</h3>
            </div>
            """
        )

        with gr.Tabs():
            with gr.Tab("Single-speaker"):
                with gr.Row():
                    model_name = gr.Dropdown(
                        label="Model",
                        choices=["zipvoice", "zipvoice_distill"],
                        value="zipvoice",
                    )
                    tokenizer = gr.Dropdown(
                        label="Tokenizer",
                        choices=["emilia", "libritts", "espeak", "simple"],
                        value="emilia",
                    )
                    lang = gr.Textbox(label="Language (for espeak)", value="en-us")

                with gr.Row():
                    model_dir = gr.Textbox(
                        label="Local model directory (optional)",
                        placeholder="Path with model.pt, model.json, tokens.txt",
                    )
                    vocoder_path = gr.Textbox(
                        label="Local vocoder directory (optional)",
                        placeholder="Path with config.yaml and pytorch_model.bin",
                    )

                prompt_wav = gr.Audio(
                    label="Prompt WAV (upload)",
                    type="filepath",
                    sources=["upload"],
                )
                with gr.Row():
                    prompt_text = gr.Textbox(label="Prompt transcription", lines=2)
                    auto_tx = gr.Checkbox(label="Auto-transcribe with Whisper", value=True)
                with gr.Row():
                    whisper_size = gr.Dropdown(
                        label="Whisper model",
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        value="small",
                    )
                    whisper_lang = gr.Textbox(label="Whisper language (optional)", value="")
                text = gr.Textbox(label="Text to synthesize", lines=3)

                with gr.Accordion("Advanced", open=False):
                    with gr.Row():
                        num_step = gr.Slider(1, 64, value=16, step=1, label="num_step")
                        guidance_scale = gr.Slider(0.0, 10.0, value=1.0, step=0.1, label="guidance_scale")
                    with gr.Row():
                        speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="speed")
                        t_shift = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="t_shift")
                    with gr.Row():
                        target_rms = gr.Slider(0.0, 0.5, value=0.1, step=0.01, label="target_rms")
                        feat_scale = gr.Slider(0.01, 1.0, value=0.1, step=0.01, label="feat_scale")

                btn = gr.Button("Generate", elem_id="cta-generate")

                out_audio = gr.Audio(label="Output", type="filepath")
                out_metrics = gr.Textbox(label="Metrics", lines=3)

                # Auto-transcribe on upload
                def _maybe_tx(path, enabled, size, lang):
                    if enabled and path:
                        try:
                            return transcribe_file(path, size, lang, unload=True)
                        except Exception as e:
                            return f"[ASR error] {e}"
                    return gr.update()

                prompt_wav.change(
                    fn=_maybe_tx,
                    inputs=[prompt_wav, auto_tx, whisper_size, whisper_lang],
                    outputs=[prompt_text],
                )

                btn.click(
                    fn=ui_infer,
                    inputs=[
                        model_name,
                        model_dir,
                        tokenizer,
                        lang,
                        vocoder_path,
                        prompt_wav,
                        prompt_text,
                        text,
                        num_step,
                        guidance_scale,
                        speed,
                        t_shift,
                        target_rms,
                        feat_scale,
                    ],
                    outputs=[out_audio, out_metrics],
                    api_name="/infer",
                )

            with gr.Tab("Dialogue"):
                with gr.Row():
                    d_model_name = gr.Dropdown(
                        label="Dialog Model",
                        choices=["zipvoice_dialog", "zipvoice_dialog_stereo"],
                        value="zipvoice_dialog",
                    )
                    d_model_dir = gr.Textbox(
                        label="Local dialog model dir (optional)",
                        placeholder="Path with model.pt, model.json, tokens.txt",
                    )
                    d_vocoder_path = gr.Textbox(
                        label="Local vocoder dir (optional)",
                        placeholder="Path with config.yaml and pytorch_model.bin",
                    )

                prompt_mode = gr.Radio(["merged", "split"], label="Prompt mode", value="merged")

                # Merged
                d_prompt_wav_merged = gr.Audio(
                    label="Merged prompt WAV (upload)",
                    type="filepath",
                    sources=["upload"],
                    visible=True,
                )
                with gr.Row(visible=True) as merged_row_text:
                    d_prompt_text_merged = gr.Textbox(
                        label="Merged prompt transcription (use [S1]/[S2])",
                        lines=2,
                    )
                    d_auto_tx_merged = gr.Checkbox(label="Auto-transcribe", value=True)

                # Split
                with gr.Row(visible=False) as split_row_wavs:
                    d_prompt_wav_s1 = gr.Audio(
                        label="Speaker 1 prompt WAV",
                        type="filepath",
                        sources=["upload"],
                    )
                    d_prompt_wav_s2 = gr.Audio(
                        label="Speaker 2 prompt WAV",
                        type="filepath",
                        sources=["upload"],
                    )
                with gr.Row(visible=False) as split_row_texts:
                    d_prompt_text_s1 = gr.Textbox(label="Speaker 1 transcription", lines=2)
                    d_prompt_text_s2 = gr.Textbox(label="Speaker 2 transcription", lines=2)
                    d_auto_tx_split = gr.Checkbox(label="Auto-transcribe both", value=True)
                with gr.Row():
                    d_whisper_size = gr.Dropdown(
                        label="Whisper model",
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        value="small",
                    )
                    d_whisper_lang = gr.Textbox(label="Whisper language (optional)", value="")

                d_text = gr.Textbox(
                    label="Conversation text (must start with [S1])",
                    lines=4,
                )

                with gr.Accordion("Advanced", open=False):
                    with gr.Row():
                        d_num_step = gr.Slider(1, 64, value=16, step=1, label="num_step")
                        d_guidance_scale = gr.Slider(0.0, 10.0, value=1.5, step=0.1, label="guidance_scale")
                    with gr.Row():
                        d_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="speed")
                        d_t_shift = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="t_shift")
                    with gr.Row():
                        d_target_rms = gr.Slider(0.0, 0.5, value=0.1, step=0.01, label="target_rms")
                        d_feat_scale = gr.Slider(0.01, 1.0, value=0.1, step=0.01, label="feat_scale")
                    d_silence_wav = gr.Textbox(
                        label="Silence wav for stereo split (optional)",
                        value=str(Path("assets") / "silence.wav"),
                    )

                def _toggle_prompt_mode(mode: str):
                    merged_visible = mode == "merged"
                    split_visible = not merged_visible
                    return (
                        gr.update(visible=merged_visible),
                        gr.update(visible=merged_visible),
                        gr.update(visible=split_visible),
                        gr.update(visible=split_visible),
                    )

                prompt_mode.change(
                    fn=_toggle_prompt_mode,
                    inputs=[prompt_mode],
                    outputs=[
                        d_prompt_wav_merged,
                        d_prompt_text_merged,
                        split_row_wavs,
                        split_row_texts,
                    ],
                )

                # Auto-transcribe wiring for dialogue
                def _maybe_tx_dialog(path, enabled, size, lang):
                    if enabled and path:
                        try:
                            return transcribe_file(path, size, lang, unload=True)
                        except Exception as e:
                            return f"[ASR error] {e}"
                    return gr.update()

                d_prompt_wav_merged.change(
                    fn=_maybe_tx_dialog,
                    inputs=[d_prompt_wav_merged, d_auto_tx_merged, d_whisper_size, d_whisper_lang],
                    outputs=[d_prompt_text_merged],
                )

                def _maybe_tx_pair(p1, p2, enabled, size, lang):
                    out1 = _maybe_tx_dialog(p1, enabled, size, lang)
                    out2 = _maybe_tx_dialog(p2, enabled, size, lang)
                    return out1, out2

                d_prompt_wav_s1.change(
                    fn=_maybe_tx_dialog,
                    inputs=[d_prompt_wav_s1, d_auto_tx_split, d_whisper_size, d_whisper_lang],
                    outputs=[d_prompt_text_s1],
                )
                d_prompt_wav_s2.change(
                    fn=_maybe_tx_dialog,
                    inputs=[d_prompt_wav_s2, d_auto_tx_split, d_whisper_size, d_whisper_lang],
                    outputs=[d_prompt_text_s2],
                )

                d_btn = gr.Button("Generate Dialogue", elem_id="d-cta-generate")
                d_out_audio = gr.Audio(label="Output", type="filepath")
                d_out_metrics = gr.Textbox(label="Metrics", lines=3)

                d_btn.click(
                    fn=ui_infer_dialog,
                    inputs=[
                        d_model_name,
                        d_model_dir,
                        d_vocoder_path,
                        prompt_mode,
                        d_prompt_wav_merged,
                        d_prompt_text_merged,
                        d_prompt_wav_s1,
                        d_prompt_text_s1,
                        d_prompt_wav_s2,
                        d_prompt_text_s2,
                        d_text,
                        d_num_step,
                        d_guidance_scale,
                        d_speed,
                        d_t_shift,
                        d_target_rms,
                        d_feat_scale,
                        d_silence_wav,
                    ],
                    outputs=[d_out_audio, d_out_metrics],
                    api_name="/infer_dialog",
                )

    return demo


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    logging.basicConfig(level=logging.INFO)
    app = build_app()
    # Bind explicitly to localhost
    app.launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", 7860)))


if __name__ == "__main__":
    main()


