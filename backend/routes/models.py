from fastapi import APIRouter
from model_manager import is_loaded, get_model
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


@router.get("/status")
def model_status():
    return {"loaded": is_loaded(), "checkpoint": CHECKPOINT}
