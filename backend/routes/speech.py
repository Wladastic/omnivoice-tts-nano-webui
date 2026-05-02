import io
import logging
import os
import tempfile
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from omnivoice import OmniVoiceGenerationConfig

from model_manager import get_model
from routes.tts import _audio_to_wav_bytes, _build_kwargs, _generate_chunked, _free_cuda_cache, CHUNK_CHARS
from voice_store import list_voices as list_saved_voices, load_voice

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["openai-compat"])

# tts-1 → fast, tts-1-hd → quality
MODEL_STEPS = {
    "tts-1": 16,
    "tts-1-hd": 32,
}
DEFAULT_STEPS = 16


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
    response_format = str(data.get("response_format") or "wav")
    if response_format not in ("wav", "pcm"):
        raise HTTPException(400, "Only response_format='wav' is currently supported")
    speed = float(data.get("speed") or 1.0)
    voice = data.get("voice")
    language = data.get("language")
    instruct = data.get("instruct")
    guidance_scale = float(data.get("guidance_scale") or 2.0)
    ref_text = data.get("ref_text")

    if model.startswith("voice:"):
        voice = model.split(":", 1)[1]
        model = "tts-1-hd"

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
        ] + voice_models,
    }
