import io
import logging
import os
import re
import tempfile
import time
from typing import Any, Dict, Optional

import wave

import numpy as np
import torch
from fastapi import APIRouter, Form, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from omnivoice import OmniVoiceGenerationConfig

from model_manager import get_model, _vram_info, reset_peak_vram, peak_vram_gb
from schemas import TTSDesignRequest, TTSVoiceRequest
from voice_store import load_voice

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tts", tags=["tts"])


def _build_gen_config(req) -> OmniVoiceGenerationConfig:
    return OmniVoiceGenerationConfig(
        num_step=req.num_step,
        guidance_scale=req.guidance_scale,
        denoise=req.denoise,
        preprocess_prompt=req.preprocess_prompt,
        postprocess_output=req.postprocess_output,
    )


def _build_kwargs(req, gen_config: OmniVoiceGenerationConfig) -> Dict[str, Any]:
    kw: Dict[str, Any] = dict(
        generation_config=gen_config,
    )
    lang = req.language if (req.language and req.language.lower() != "auto") else None
    if lang:
        kw["language"] = lang
    if req.speed != 1.0:
        kw["speed"] = req.speed
    if req.duration and req.duration > 0:
        kw["duration"] = req.duration
    return kw


CHUNK_CHARS = int(os.environ.get("CHUNK_CHARS", "200"))

ENGLISH_INSTRUCT_ITEMS = {
    "american accent", "australian accent", "british accent", "canadian accent",
    "child", "chinese accent", "elderly", "female", "high pitch",
    "indian accent", "japanese accent", "korean accent", "low pitch", "male",
    "middle-aged", "moderate pitch", "portuguese accent", "russian accent",
    "teenager", "very high pitch", "very low pitch", "whisper", "young adult",
}

CHINESE_INSTRUCT_ITEMS = {
    "东北话", "中年", "中音调", "云南话", "低音调", "儿童", "四川话", "女",
    "宁夏话", "少年", "极低音调", "极高音调", "桂林话", "河南话", "济南话",
    "甘肃话", "男", "石家庄话", "老年", "耳语", "贵州话", "陕西话", "青岛话",
    "青年", "高音调",
}

ENGLISH_INSTRUCT_HELP = ", ".join(sorted(ENGLISH_INSTRUCT_ITEMS))
CHINESE_INSTRUCT_HELP = "，".join(sorted(CHINESE_INSTRUCT_ITEMS))


def _normalize_instruct(instruct: str) -> str:
    value = instruct.strip()
    if not value:
        raise HTTPException(400, "Speaker instruction is required.")

    has_chinese = any("\u4e00" <= c <= "\u9fff" for c in value)
    if has_chinese:
        items = [item.strip() for item in value.split("，") if item.strip()]
        invalid = [item for item in items if item not in CHINESE_INSTRUCT_ITEMS]
        if invalid:
            raise HTTPException(
                400,
                "Unsupported Chinese speaker tag(s): "
                + ", ".join(invalid)
                + f". Use only these tags separated by full-width commas: {CHINESE_INSTRUCT_HELP}",
            )
        return "，".join(items)

    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [item for item in items if item not in ENGLISH_INSTRUCT_ITEMS]
    if invalid:
        raise HTTPException(
            400,
            "Unsupported speaker tag(s): "
            + ", ".join(invalid)
            + f". Use only these tags separated by comma + space: {ENGLISH_INSTRUCT_HELP}",
        )
    return ", ".join(items)


_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+|(?<=[.!?。！？])$")


def _chunk_text(text: str, max_chars: int = 200) -> list[str]:
    """Split text into sentence-bounded chunks under max_chars."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    if not sentences:
        return [text]

    chunks: list[str] = []
    current = ""
    for s in sentences:
        if not current:
            current = s
        elif len(current) + 1 + len(s) <= max_chars:
            current = current + " " + s
        else:
            chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks


CHUNK_GAP_MS = int(os.environ.get("CHUNK_GAP_MS", "150"))


def _generate_chunked(model, kw_base: Dict[str, Any], text: str, chunk_chars: int):
    chunks = _chunk_text(text, chunk_chars)
    if len(chunks) == 1:
        kw = dict(kw_base, text=chunks[0])
        out = model.generate(**kw)
        return out[0], len(chunks)

    sr = model.sampling_rate
    gap = np.zeros(int(sr * CHUNK_GAP_MS / 1000), dtype=np.float32)

    parts = []
    for i, chunk in enumerate(chunks):
        kw = dict(kw_base, text=chunk)
        logger.debug(f"  chunk {i+1}/{len(chunks)}: {chunk[:60]!r}")
        out = model.generate(**kw)
        if i > 0:
            parts.append(gap)
        parts.append(out[0])
    return np.concatenate(parts), len(chunks)


def _free_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _audio_to_wav_bytes(audio: np.ndarray, sampling_rate: int) -> bytes:
    waveform = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sampling_rate)
        wf.writeframes(waveform.tobytes())
    buf.seek(0)
    return buf.read()


@router.post("/clone")
def tts_clone(
    ref_audio: UploadFile = File(...),
    text: str = Form(...),
    language: Optional[str] = Form(None),
    ref_text: Optional[str] = Form(None),
    instruct: Optional[str] = Form(None),
    num_step: int = Form(32),
    guidance_scale: float = Form(2.0),
    denoise: bool = Form(True),
    speed: float = Form(1.0),
    duration: float = Form(0.0),
    preprocess_prompt: bool = Form(True),
    postprocess_output: bool = Form(True),
):
    """Voice cloning: synthesize text conditioned on uploaded reference audio."""
    t0 = time.time()
    logger.info(f"[clone] request — text={text!r} lang={language} steps={num_step} guidance={guidance_scale}")

    reset_peak_vram()
    model = get_model()
    logger.debug(f"[clone] model ready in {time.time()-t0:.2f}s")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(ref_audio.file.read())
        tmp_path = tmp.name

    try:
        class _Req:
            pass
        req = _Req()
        req.text = text
        req.language = language
        req.speed = speed
        req.duration = duration
        req.num_step = num_step
        req.guidance_scale = guidance_scale
        req.denoise = denoise
        req.preprocess_prompt = preprocess_prompt
        req.postprocess_output = postprocess_output

        gen_config = _build_gen_config(req)
        kw = _build_kwargs(req, gen_config)

        t1 = time.time()
        logger.debug(f"[clone] VRAM before clone prompt: {_vram_info()}")
        try:
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                ref_audio=tmp_path,
                ref_text=ref_text or None,
            )
        except Exception as e:
            raise HTTPException(400, f"Failed to create voice clone prompt: {e}")
        logger.debug(f"[clone] voice clone prompt ready in {time.time()-t1:.2f}s — VRAM: {_vram_info()}")

        if instruct and instruct.strip():
            kw["instruct"] = _normalize_instruct(instruct)

        t2 = time.time()
        logger.info(f"[clone] starting generation ({num_step} steps) — VRAM: {_vram_info()}")
        try:
            audio, n_chunks = _generate_chunked(model, kw, text.strip(), CHUNK_CHARS)
        except Exception as e:
            logger.exception("TTS clone generation failed")
            raise HTTPException(500, f"Generation failed: {e}")
        logger.info(f"[clone] generation done in {time.time()-t2:.2f}s ({n_chunks} chunk{'s' if n_chunks != 1 else ''}, total {time.time()-t0:.2f}s) — VRAM: {_vram_info()}")

    finally:
        os.unlink(tmp_path)
        _free_cuda_cache()

    t3 = time.time()
    wav_bytes = _audio_to_wav_bytes(audio, model.sampling_rate)
    peak = peak_vram_gb()
    total = time.time() - t0
    logger.info(f"[clone] peak VRAM: {peak:.2f}GB, total time: {total:.2f}s")
    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={"X-Peak-VRAM-GB": f"{peak:.2f}", "X-Total-Time-S": f"{total:.2f}"},
    )


@router.post("/design")
def tts_design(req: TTSDesignRequest):
    """Voice design: synthesize text with speaker attributes described in instruct."""
    t0 = time.time()
    logger.info(f"[design] request — text={req.text!r} steps={req.num_step}")
    reset_peak_vram()
    model = get_model()
    gen_config = _build_gen_config(req)
    kw = _build_kwargs(req, gen_config)
    kw["instruct"] = _normalize_instruct(req.instruct)

    logger.info(f"[design] starting generation ({req.num_step} steps)...")
    try:
        audio, n_chunks = _generate_chunked(model, kw, req.text.strip(), CHUNK_CHARS)
    except Exception as e:
        logger.exception("TTS design generation failed")
        raise HTTPException(500, f"Generation failed: {e}")
    finally:
        _free_cuda_cache()
    peak = peak_vram_gb()
    total = time.time() - t0
    logger.info(f"[design] done in {total:.2f}s ({n_chunks} chunks), peak VRAM: {peak:.2f}GB")

    wav_bytes = _audio_to_wav_bytes(audio, model.sampling_rate)
    return StreamingResponse(
        io.BytesIO(wav_bytes), media_type="audio/wav",
        headers={"X-Peak-VRAM-GB": f"{peak:.2f}", "X-Total-Time-S": f"{total:.2f}"},
    )


@router.post("/voice/{voice_id}")
def tts_voice(voice_id: str, req: TTSVoiceRequest):
    """Generate using a saved voice profile from voices/."""
    t0 = time.time()
    logger.info(f"[voice] request — voice_id={voice_id!r} text={req.text!r} steps={req.num_step}")
    reset_peak_vram()
    voice = load_voice(voice_id)
    if voice is None:
        raise HTTPException(404, f"Voice '{voice_id}' not found")

    model = get_model()
    gen_config = _build_gen_config(req)
    kw = _build_kwargs(req, gen_config)

    ref_audio = voice.get("ref_audio_path")
    ref_text = voice.get("ref_text")

    if ref_audio:
        t1 = time.time()
        try:
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
        except Exception as e:
            raise HTTPException(400, f"Failed to load voice: {e}")
        logger.debug(f"[voice] clone prompt in {time.time()-t1:.2f}s")
    elif voice.get("instruct"):
        kw["instruct"] = _normalize_instruct(voice["instruct"])

    if req.instruct and req.instruct.strip():
        kw["instruct"] = _normalize_instruct(req.instruct)

    logger.info(f"[voice] starting generation ({req.num_step} steps)...")
    try:
        audio, n_chunks = _generate_chunked(model, kw, req.text.strip(), CHUNK_CHARS)
    except Exception as e:
        logger.exception("TTS voice generation failed")
        raise HTTPException(500, f"Generation failed: {e}")
    finally:
        _free_cuda_cache()
    peak = peak_vram_gb()
    total = time.time() - t0
    logger.info(f"[voice] done in {total:.2f}s ({n_chunks} chunks), peak VRAM: {peak:.2f}GB")

    wav_bytes = _audio_to_wav_bytes(audio, model.sampling_rate)
    return StreamingResponse(
        io.BytesIO(wav_bytes), media_type="audio/wav",
        headers={"X-Peak-VRAM-GB": f"{peak:.2f}", "X-Total-Time-S": f"{total:.2f}"},
    )
