
# ZipVoice-UI

A **Gradio-based web interface** for the [ZipVoice](https://github.com/k2-fsa/ZipVoice) speech synthesis system.
This UI makes it easy to:

* Clone a voice from a short audio prompt.
* Generate **single-speaker** or **multi-speaker dialogues**.
* Optionally auto-transcribe prompts using [faster-whisper](https://github.com/guillaumekln/faster-whisper).
* Fine-tune synthesis parameters for speed, style, and quality.

---

## 🙏 Credits

A huge thanks to the developers of the original **[ZipVoice](https://github.com/k2-fsa/ZipVoice)** project — without their work, this app would not be possible.
This UI simply wraps their powerful TTS models in an accessible web interface.

---

## 🔹 Features

### 🎤 Single-Speaker Mode

* Upload a short WAV file and transcription (or auto-transcribe with Whisper).
* Type any text to synthesize in the cloned voice.
* Adjustable synthesis parameters:

  * `num_step`
  * `guidance_scale`
  * `speed`
  * `t_shift`
  * `target_rms`
  * `feat_scale`

### 💬 Dialogue Mode

* Two prompt modes:

  * **Merged** – one WAV containing both speakers.
  * **Split** – separate WAVs + transcriptions for each speaker.
* Dialogue text must start with `[S1]` and alternate between `[S1]` / `[S2]`.
* Stereo or mono dialogue generation.
* Optional silence WAV for stereo split.

### 🔊 Whisper Auto-Transcription

* Multiple Whisper model sizes: `tiny`, `base`, `small`, `medium`, `large-v3`.
* Optional language override for transcription.

---

## 📦 Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/SUP3RMASS1VE/ZipVoice-UI.git
   cd ZipVoice-UI

   python -m venv venv
   venv\Scripts\activate

   pip install uv
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   uv pip install -r requirements.txt
   ```


---

## 🚀 Running the App

```bash
python zipvoice/bin/gradio_app.py
```

By default, the UI will be served at:

```
http://127.0.0.1:7860
```

---

## 🛠 Usage

### Single-Speaker

1. Select **Model** (`zipvoice` or `zipvoice_distill`) and **Tokenizer**.
2. Upload **Prompt WAV** + transcription (or auto-transcribe with Whisper).
3. Type your synthesis text.
4. Adjust advanced settings if desired.
5. Click **Generate**.

### Dialogue

1. Select **Dialog Model** (`zipvoice_dialog` or `zipvoice_dialog_stereo`).
2. Choose prompt mode (merged or split) and provide audio + text prompts.
3. Enter conversation text, starting with `[S1]`.
4. Adjust advanced settings if desired.
5. Click **Generate Dialogue**.

---

## 📂 Model Files

* **Automatic download** from HuggingFace is supported.
* You can also provide **local model directories** containing:

  * `model.pt` or `.safetensors`
  * `model.json`
  * `tokens.txt`
* Optional **vocoder directory**:

  * `config.yaml`
  * `pytorch_model.bin`

---

## ⚖ License

MIT License (or as defined by the repository owner).

Special thanks again to the [ZipVoice](https://github.com/k2-fsa/ZipVoice) team for their incredible work on the original model.

---


