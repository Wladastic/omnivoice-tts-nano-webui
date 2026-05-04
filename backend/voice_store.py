import json
import os
import shutil
from pathlib import Path
from typing import Optional

import librosa
import soundfile as sf

from config import SAMPLING_RATE
from config import VOICES_DIR

REF_TRAILING_SILENCE_SECONDS = float(os.environ.get("REF_TRAILING_SILENCE_SECONDS", "0.15"))


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
    os.chmod(d, 0o755)
    data = {"name": name, "ref_text": ref_text, "description": description}
    cfg = d / "voice.json"
    cfg.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.chmod(cfg, 0o644)
    data["id"] = voice_id
    data["has_audio"] = False
    return data


def update_voice(voice_id: str, name: str, ref_text: str, description: str = "") -> Optional[dict]:
    d = _voice_dir(voice_id)
    if not d.exists():
        return None
    os.chmod(d, 0o755)
    data = {"name": name, "ref_text": ref_text, "description": description}
    cfg = d / "voice.json"
    cfg.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.chmod(cfg, 0o644)
    data["id"] = voice_id
    data["has_audio"] = (d / "ref.wav").exists()
    return data


def save_voice_audio(voice_id: str, src_path: str) -> bool:
    d = _voice_dir(voice_id)
    if not d.exists():
        return False
    dst = d / "ref.wav"
    try:
        audio, _ = librosa.load(src_path, sr=SAMPLING_RATE, mono=True)
        trailing = int(SAMPLING_RATE * REF_TRAILING_SILENCE_SECONDS)
        if trailing > 0:
            audio = librosa.util.fix_length(audio, size=len(audio) + trailing)
        sf.write(dst, audio, SAMPLING_RATE, subtype="PCM_24", format="WAV")
    except Exception:
        shutil.copyfile(src_path, dst)
    os.chmod(dst, 0o644)
    return True


def delete_voice(voice_id: str) -> bool:
    d = _voice_dir(voice_id)
    if not d.exists():
        return False
    shutil.rmtree(d)
    return True
