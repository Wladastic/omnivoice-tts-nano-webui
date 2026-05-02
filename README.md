# OmniVoice Low-VRAM WebUI API

Docker Compose setup for running an OmniVoice backend with a Gradio WebUI.
The main goal of this project is to make OmniVoice practical on machines with
limited VRAM by keeping model loading, quantization, chunking, and cleanup
configurable.

It supports:

- voice cloning from short reference audio
- voice design with OmniVoice speaker tags
- saved voice profiles
- optional STT-assisted reference transcription and trimming
- low-VRAM GPU settings such as `LM_QUANT`, `MAX_VRAM_GB`, chunking, and TTL
- CPU and NVIDIA GPU compose profiles

## Requirements

- Docker and Docker Compose
- For GPU mode: NVIDIA Container Toolkit and a CUDA-capable GPU

## Quick Start

GPU:

```bash
cp .env-example .env
./start-gpu.sh -d
```

CPU:

```bash
cp .env-example .env
./start-cpu.sh -d
```

Open the WebUI at:

```text
http://localhost:7863
```

The backend API listens on:

```text
http://localhost:8883
```

## Configuration

Use `.env-example` as the public template:

```bash
cp .env-example .env
```

`.env` is ignored by git and is meant for local/private settings.

Common variables:

- `FRONTEND_PORT`: host port for the Gradio UI
- `UI_LANG`: UI language, currently `en` or `de`
- `STT_URL`: optional OpenAI-compatible transcription service URL
- `OMNIVOICE_MODEL`: Hugging Face model id or local model path
- `DEVICE`: `cpu` or `cuda`
- `DTYPE`: `float16` or `bfloat16`
- `LM_QUANT`: `none`, `nf4`, or `int8`
- `MAX_VRAM_GB`: GPU memory guard, `0` disables the limit
- `CHUNK_CHARS`: split long text into smaller generation chunks
- `MODEL_TTL_SECONDS`: unload idle backend model state after this many seconds

Compose uses the internal backend URL between containers. `API_URL` in `.env-example`
is mainly useful when running the frontend directly on the host.

## Low-VRAM Notes

This repository is tuned around keeping VRAM usage as low and predictable as
possible:

- `LM_QUANT=nf4` is the default GPU compose setting for the language-model part.
- `MAX_VRAM_GB` can stop generation when measured peak VRAM grows beyond your
  chosen budget.
- `MODEL_TTL_SECONDS` allows the backend to unload idle model state.
- `CHUNK_CHARS` avoids pushing long prompts through generation as one large item.
- The backend reports peak VRAM and total generation time in response headers.

For the lowest memory footprint, start with the GPU defaults in `.env-example`
and reduce `CHUNK_CHARS` before increasing model precision or step count.

## Optional STT

STT is optional and intentionally lives in a separate project:
[stt-nano-webui](https://github.com/Wladastic/stt-nano-webui).
This OmniVoice repo only needs an OpenAI-compatible transcription endpoint at:

```text
POST /v1/audio/transcriptions
```

If `STT_URL` is empty, transcription is disabled and auto-trim falls back to
silence detection.

For a local STT service on the host:

```env
STT_URL=http://localhost:8882
```

For a service on another machine, set that URL only in your local `.env`.

The companion `stt-nano-webui` project provides:

- WebUI: `http://localhost:7861`
- backend API: `http://localhost:8882`
- OpenAI-style transcription: `POST /v1/audio/transcriptions`
- lightweight default model: `parakeet-onnx-int8`
- optional `whisper-1` alias for OpenAI-compatible clients

Together, `omnivoice-tts-nano-webui` and `stt-nano-webui` can be used as local
speech services for tools such as OpenWebUI: this repo covers TTS/voice cloning,
while `stt-nano-webui` covers speech-to-text.

## Voice Design Tags

OmniVoice voice design does not accept free-form prompts. Use comma-separated
speaker tags, for example:

```text
female, young adult, low pitch, british accent
```

Supported English tags include:

```text
american accent, australian accent, british accent, canadian accent, child,
chinese accent, elderly, female, high pitch, indian accent, japanese accent,
korean accent, low pitch, male, middle-aged, moderate pitch, portuguese accent,
russian accent, teenager, very high pitch, very low pitch, whisper, young adult
```

## Local Data

These directories/files are intentionally ignored:

- `.env`
- `logs/`
- `models/`
- `voices/`
- `__pycache__/`

This keeps generated audio, model cache, logs, and private configuration out of git.
