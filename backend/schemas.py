from typing import Optional
from pydantic import BaseModel, Field


class TTSCloneRequest(BaseModel):
    text: str
    language: Optional[str] = None
    ref_audio_path: str
    ref_text: Optional[str] = None
    instruct: Optional[str] = None
    num_step: int = Field(32, ge=1, le=200)
    guidance_scale: float = Field(2.0, ge=0.1, le=10.0)
    denoise: bool = True
    speed: float = Field(1.0, ge=0.1, le=3.0)
    duration: float = Field(0.0, ge=0.0)  # 0 = auto
    preprocess_prompt: bool = True
    postprocess_output: bool = True


class TTSDesignRequest(BaseModel):
    text: str
    language: Optional[str] = None
    instruct: str  # required for voice design
    num_step: int = Field(32, ge=1, le=200)
    guidance_scale: float = Field(2.0, ge=0.1, le=10.0)
    denoise: bool = True
    speed: float = Field(1.0, ge=0.1, le=3.0)
    duration: float = Field(0.0, ge=0.0)
    preprocess_prompt: bool = True
    postprocess_output: bool = True


class TTSVoiceRequest(BaseModel):
    """Generate using a saved voice profile."""
    text: str
    voice_id: str
    language: Optional[str] = None
    instruct: Optional[str] = None
    num_step: int = Field(32, ge=1, le=200)
    guidance_scale: float = Field(2.0, ge=0.1, le=10.0)
    denoise: bool = True
    speed: float = Field(1.0, ge=0.1, le=3.0)
    duration: float = Field(0.0, ge=0.0)
    preprocess_prompt: bool = True
    postprocess_output: bool = True


class VoiceCreate(BaseModel):
    name: str
    ref_text: str
    description: Optional[str] = None


class VoiceInfo(BaseModel):
    id: str
    name: str
    ref_text: str
    description: Optional[str] = None
    has_audio: bool
