import io
import logging
import os
import tempfile
from typing import Optional

from fastapi import APIRouter, Form, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from omnivoice import OmniVoiceGenerationConfig

from model_manager import get_model
from routes.tts import _audio_to_wav_bytes, _build_kwargs, _generate_chunked, _free_cuda_cache, CHUNK_CHARS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["openai-compat"])

# tts-1 → fast, tts-1-hd → quality
MODEL_STEPS = {
    "tts-1": 16,
    "tts-1-hd": 32,
}
DEFAULT_STEPS = 16


@router.post("/audio/speech")
def openai_speech(
    model: str = Form("tts-1-hd"),
    input: str = Form(...),
    voice: Optional[str] = Form(None),
    response_format: str = Form("wav"),
    speed: float = Form(1.0),
    # extra params
    language: Optional[str] = Form(None),
    instruct: Optional[str] = Form(None),
    guidance_scale: float = Form(2.0),
    ref_audio: Optional[UploadFile] = File(None),
    ref_text: Optional[str] = Form(None),
):
    num_step = MODEL_STEPS.get(model, DEFAULT_STEPS)
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

        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

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
    return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav")


@router.get("/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "tts-1", "object": "model", "description": "OmniVoice fast (16 steps)"},
            {"id": "tts-1-hd", "object": "model", "description": "OmniVoice quality (32 steps)"},
        ],
    }
