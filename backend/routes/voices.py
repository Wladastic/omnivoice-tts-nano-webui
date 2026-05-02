import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

from schemas import VoiceCreate, VoiceInfo
from voice_store import list_voices, load_voice, create_voice, save_voice_audio, delete_voice

router = APIRouter(prefix="/voices", tags=["voices"])


@router.get("", response_model=list[VoiceInfo])
def get_voices():
    return list_voices()


@router.get("/{voice_id}", response_model=VoiceInfo)
def get_voice(voice_id: str):
    v = load_voice(voice_id)
    if v is None:
        raise HTTPException(404, f"Voice '{voice_id}' not found")
    return v


@router.post("/{voice_id}", response_model=VoiceInfo)
def post_voice(voice_id: str, body: VoiceCreate):
    if load_voice(voice_id) is not None:
        raise HTTPException(409, f"Voice '{voice_id}' already exists")
    return create_voice(voice_id, body.name, body.ref_text, body.description or "")


@router.put("/{voice_id}/audio")
def upload_voice_audio(voice_id: str, file: UploadFile = File(...)):
    if load_voice(voice_id) is None:
        raise HTTPException(404, f"Voice '{voice_id}' not found")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    ok = save_voice_audio(voice_id, tmp_path)
    Path(tmp_path).unlink(missing_ok=True)
    if not ok:
        raise HTTPException(500, "Failed to save audio")
    return {"status": "ok"}


@router.get("/{voice_id}/audio")
def get_voice_audio(voice_id: str):
    from config import VOICES_DIR
    wav = VOICES_DIR / voice_id / "ref.wav"
    if not wav.exists():
        raise HTTPException(404, "No reference audio for this voice")
    return FileResponse(str(wav), media_type="audio/wav")


@router.delete("/{voice_id}")
def remove_voice(voice_id: str):
    if not delete_voice(voice_id):
        raise HTTPException(404, f"Voice '{voice_id}' not found")
    return {"status": "deleted"}
