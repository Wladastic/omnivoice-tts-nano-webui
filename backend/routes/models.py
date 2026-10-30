from fastapi import APIRouter
from model_manager import is_loaded, get_model, peak_vram_gb, unload, vram_mb
from config import CHECKPOINT, DEVICE, DTYPE, SAMPLING_RATE

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
def list_models():
    return [
        {
            "id": CHECKPOINT,
            "loaded": is_loaded(),
            "device": DEVICE,
            "dtype": DTYPE,
            "sampling_rate": SAMPLING_RATE,
        }
    ]


@router.post("/load")
def load_model():
    get_model()
    return {"status": "loaded", "checkpoint": CHECKPOINT}


@router.post("/unload")
def unload_model():
    """Free the GPU now, e.g. before ComfyUI runs. The next TTS request loads the model again."""
    status = "unloaded" if unload() else "already_unloaded"
    return {"status": status, "checkpoint": CHECKPOINT, "vram": vram_mb()}


@router.get("/status")
def model_status():
    return {
        "loaded": is_loaded(),
        "checkpoint": CHECKPOINT,
        "peak_vram_gb": peak_vram_gb(),
        "vram": vram_mb(),
    }
