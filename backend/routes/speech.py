import io
import logging
import os
import tempfile
from typing import Iterator, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from omnivoice import OmniVoiceGenerationConfig
from pydub import AudioSegment

from model_manager import get_model
from routes.tts import (
    _audio_to_wav_bytes,
    _build_kwargs,
    _chunk_text,
    _generate_chunked,
    _free_cuda_cache,
    CHUNK_CHARS,
    CHUNK_GAP_MS,
)
from voice_store import list_voices as list_saved_voices, load_voice

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["openai-compat"])

# tts-1 → fast, tts-1-hd → quality
MODEL_STEPS = {
    "tts-1": 16,
    "tts-1-hd": 32,
    "gpt-4o-mini-tts": 32,
    "gpt-4o-tts": 32,
}
DEFAULT_STEPS = 16
SUPPORTED_RESPONSE_FORMATS = {"wav", "pcm", "mp3"}


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _audio_to_pcm_bytes(audio: np.ndarray) -> bytes:
    waveform = np.clip(audio, -1.0, 1.0)
    return (waveform * 32767).astype(np.int16).tobytes()


def _wav_stream_header(sampling_rate: int) -> bytes:
    # Chunked HTTP has no fixed length up front, so use the largest valid RIFF sizes.
    # Most clients accept this for live-ish WAV streams.
    byte_rate = sampling_rate * 2
    block_align = 2
    return (
        b"RIFF"
        + (0xFFFFFFFF).to_bytes(4, "little")
        + b"WAVEfmt "
        + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + (1).to_bytes(2, "little")
        + sampling_rate.to_bytes(4, "little")
        + byte_rate.to_bytes(4, "little")
        + block_align.to_bytes(2, "little")
        + (16).to_bytes(2, "little")
        + b"data"
        + (0xFFFFFFFF).to_bytes(4, "little")
    )


def _format_audio_response(wav_bytes: bytes, response_format: str) -> tuple[bytes, str]:
    if response_format == "wav":
        return wav_bytes, "audio/wav"
    if response_format == "pcm":
        audio = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
        return audio.raw_data, "audio/pcm"
    if response_format == "mp3":
        audio = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
        buf = io.BytesIO()
        audio.export(buf, format="mp3")
        buf.seek(0)
        return buf.read(), "audio/mpeg"
    raise HTTPException(400, f"Unsupported response_format={response_format!r}")


def _stream_media_type(response_format: str) -> str:
    if response_format == "mp3":
        return "audio/mpeg"
    if response_format == "pcm":
        return "audio/pcm"
    return "audio/wav"


def _encode_stream_chunk(audio: np.ndarray, sampling_rate: int, response_format: str) -> bytes:
    if response_format == "pcm":
        return _audio_to_pcm_bytes(audio)
    if response_format == "mp3":
        wav_bytes = _audio_to_wav_bytes(audio, sampling_rate)
        segment = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
        buf = io.BytesIO()
        segment.export(buf, format="mp3")
        buf.seek(0)
        return buf.read()
    return _audio_to_pcm_bytes(audio)


def _generate_audio_stream(
    model,
    kw_base: dict,
    text: str,
    response_format: str,
    cleanup_path: Optional[str] = None,
) -> Iterator[bytes]:
    chunks = _chunk_text(text, CHUNK_CHARS)
    gap = np.zeros(int(model.sampling_rate * CHUNK_GAP_MS / 1000), dtype=np.float32)
    logger.info("[openai] streaming %d chunk%s", len(chunks), "" if len(chunks) == 1 else "s")

    try:
        if response_format == "wav":
            yield _wav_stream_header(model.sampling_rate)

        for i, chunk in enumerate(chunks):
            logger.debug("[openai] stream chunk %d/%d: %r", i + 1, len(chunks), chunk[:60])
            if i > 0 and CHUNK_GAP_MS > 0:
                yield _encode_stream_chunk(gap, model.sampling_rate, response_format)

            out = model.generate(**dict(kw_base, text=chunk))
            yield _encode_stream_chunk(out[0], model.sampling_rate, response_format)
    except Exception:
        logger.exception("OpenAI speech stream generation failed")
        raise
    finally:
        if cleanup_path:
            try:
                os.unlink(cleanup_path)
            except FileNotFoundError:
                pass
        _free_cuda_cache()


@router.post("/audio/speech")
async def openai_speech(request: Request):
    content_type = request.headers.get("content-type", "")
    ref_audio: Optional[UploadFile] = None

    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        data = dict(form)
        uploaded = data.get("ref_audio")
        if isinstance(uploaded, UploadFile) or hasattr(uploaded, "read"):
            ref_audio = uploaded
    else:
        try:
            data = await request.json()
        except Exception:
            raise HTTPException(400, "Expected JSON or form data body")

    model = str(data.get("model") or "tts-1-hd")
    input = str(data.get("input") or "").strip()
    if not input:
        raise HTTPException(422, "Field 'input' is required")
    response_format = str(data.get("response_format") or "wav").lower()
    if response_format not in SUPPORTED_RESPONSE_FORMATS:
        raise HTTPException(
            400,
            "Supported response_format values are: "
            + ", ".join(sorted(SUPPORTED_RESPONSE_FORMATS)),
        )
    speed = float(data.get("speed") or 1.0)
    voice = data.get("voice")
    language = data.get("language")
    instruct = data.get("instruct")
    guidance_scale = float(data.get("guidance_scale") or 2.0)
    ref_text = data.get("ref_text")
    stream = _truthy(data.get("stream"))

    if model.startswith("voice:"):
        voice = model.split(":", 1)[1]
        model = "tts-1-hd"

    num_step = MODEL_STEPS.get(model, DEFAULT_STEPS)
    logger.info(
        "[openai] request model=%r voice=%r response_format=%r stream=%s text_len=%d",
        model,
        voice,
        response_format,
        stream,
        len(input),
    )
    omni = get_model()

    class _Req:
        pass
    req = _Req()
    req.text = input
    req.language = language
    req.speed = speed
    req.duration = 0.0
    req.num_step = num_step
    req.guidance_scale = guidance_scale
    req.denoise = True
    req.preprocess_prompt = True
    req.postprocess_output = True

    gen_config = OmniVoiceGenerationConfig(
        num_step=num_step,
        guidance_scale=guidance_scale,
        denoise=True,
        preprocess_prompt=True,
        postprocess_output=True,
    )
    kw = _build_kwargs(req, gen_config)

    tmp_path = None
    try:
        if ref_audio is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(ref_audio.file.read())
                tmp_path = tmp.name
            try:
                kw["voice_clone_prompt"] = omni.create_voice_clone_prompt(
                    ref_audio=tmp_path,
                    ref_text=ref_text or None,
                )
            except Exception as e:
                raise HTTPException(400, f"Failed to create voice clone prompt: {e}")
        elif voice and str(voice).strip() and str(voice).strip() not in ("alloy", "echo", "fable", "onyx", "nova", "shimmer"):
            voice_id = str(voice).strip()
            saved_voice = load_voice(voice_id)
            if saved_voice is None:
                raise HTTPException(404, f"Voice '{voice_id}' not found")
            ref_audio_path = saved_voice.get("ref_audio_path")
            if not ref_audio_path:
                raise HTTPException(400, f"Voice '{voice_id}' has no reference audio")
            try:
                kw["voice_clone_prompt"] = omni.create_voice_clone_prompt(
                    ref_audio=ref_audio_path,
                    ref_text=ref_text or saved_voice.get("ref_text"),
                )
                logger.info("[openai] using saved voice %s", voice_id)
            except Exception as e:
                raise HTTPException(400, f"Failed to load voice '{voice_id}': {e}")

        if instruct and str(instruct).strip():
            kw["instruct"] = str(instruct).strip()

        if stream:
            cleanup_path = tmp_path
            tmp_path = None
            return StreamingResponse(
                _generate_audio_stream(omni, kw, input.strip(), response_format, cleanup_path),
                media_type=_stream_media_type(response_format),
            )

        try:
            audio, n_chunks = _generate_chunked(omni, kw, input.strip(), CHUNK_CHARS)
        except Exception as e:
            logger.exception("OpenAI speech generation failed")
            raise HTTPException(500, f"Generation failed: {e}")

    finally:
        if tmp_path:
            os.unlink(tmp_path)
        _free_cuda_cache()

    logger.info(f"[openai] {model} done — {n_chunks} chunk{'s' if n_chunks != 1 else ''}")
    wav_bytes = _audio_to_wav_bytes(audio, omni.sampling_rate)
    response_bytes, media_type = _format_audio_response(wav_bytes, response_format)
    return StreamingResponse(io.BytesIO(response_bytes), media_type=media_type)


@router.get("/models")
def list_models():
    voice_models = [
        {
            "id": f"voice:{voice['id']}",
            "object": "model",
            "description": f"OmniVoice saved voice: {voice.get('name', voice['id'])}",
        }
        for voice in list_saved_voices()
        if voice.get("has_audio")
    ]
    return {
        "object": "list",
        "data": [
            {"id": "tts-1", "object": "model", "description": "OmniVoice fast (16 steps)"},
            {"id": "tts-1-hd", "object": "model", "description": "OmniVoice quality (32 steps)"},
            {"id": "gpt-4o-mini-tts", "object": "model", "description": "OmniVoice quality (32 steps)"},
            {"id": "gpt-4o-tts", "object": "model", "description": "OmniVoice quality (32 steps)"},
        ] + voice_models,
    }
