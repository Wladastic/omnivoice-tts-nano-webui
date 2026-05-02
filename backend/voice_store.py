import json
import shutil
from pathlib import Path
from typing import Optional

from config import VOICES_DIR


def _voice_dir(voice_id: str) -> Path:
    return VOICES_DIR / voice_id


def list_voices() -> list[dict]:
    voices = []
    if not VOICES_DIR.exists():
        return voices
    for p in sorted(VOICES_DIR.iterdir()):
        if not p.is_dir():
            continue
        cfg = p / "voice.json"
        if not cfg.exists():
            continue
        try:
            data = json.loads(cfg.read_text())
            data["id"] = p.name
            data["has_audio"] = (p / "ref.wav").exists()
            voices.append(data)
        except Exception:
            pass
    return voices


def load_voice(voice_id: str) -> Optional[dict]:
    d = _voice_dir(voice_id)
    cfg = d / "voice.json"
    if not cfg.exists():
        return None
    data = json.loads(cfg.read_text())
    data["id"] = voice_id
    ref_wav = d / "ref.wav"
    if ref_wav.exists():
        data["ref_audio_path"] = str(ref_wav)
    data["has_audio"] = ref_wav.exists()
    return data


def create_voice(voice_id: str, name: str, ref_text: str, description: str = "") -> dict:
    d = _voice_dir(voice_id)
    d.mkdir(parents=True, exist_ok=True)
    data = {"name": name, "ref_text": ref_text, "description": description}
    (d / "voice.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
    data["id"] = voice_id
    data["has_audio"] = False
    return data


def save_voice_audio(voice_id: str, src_path: str) -> bool:
    d = _voice_dir(voice_id)
    if not d.exists():
        return False
    shutil.copy(src_path, d / "ref.wav")
    return True


def delete_voice(voice_id: str) -> bool:
    d = _voice_dir(voice_id)
    if not d.exists():
        return False
    shutil.rmtree(d)
    return True
