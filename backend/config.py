import os
from pathlib import Path

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/app/models"))
VOICES_DIR = Path(os.environ.get("VOICES_DIR", "/app/voices"))
CHECKPOINT = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
DEVICE = os.environ.get("DEVICE", "cuda")
DTYPE = os.environ.get("DTYPE", "float16")  # float16 or bfloat16
LM_QUANT = os.environ.get("LM_QUANT", "none").lower()  # none, nf4, int8
LOAD_ASR = os.environ.get("LOAD_ASR", "false").lower() == "true"
MODEL_TTL_SECONDS = int(os.environ.get("MODEL_TTL_SECONDS", "3600"))
MAX_VRAM_GB = float(os.environ.get("MAX_VRAM_GB", "0"))  # 0 = no limit
SAMPLING_RATE = 24000
