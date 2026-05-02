import os
import tempfile

import gradio as gr
import httpx
import librosa
import numpy as np
import soundfile as sf

from i18n import t

API_URL = os.environ.get("API_URL", "http://localhost:8883")
STT_URL = os.environ.get("STT_URL", "").strip().rstrip("/")
FRONTEND_PORT = int(os.environ.get("FRONTEND_PORT", "7861"))

LANGUAGES = [
    "Auto", "English", "German", "French", "Spanish", "Italian", "Portuguese",
    "Dutch", "Polish", "Russian", "Chinese", "Japanese", "Korean", "Arabic",
    "Hindi", "Turkish", "Swedish", "Norwegian", "Danish", "Finnish",
]

MAX_REF_SECONDS = 3.5


def _save_response_wav(response_bytes: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(fd, "wb") as f:
        f.write(response_bytes)
    return path


def _stt_url() -> str | None:
    return STT_URL or None


def _status_from_response(r) -> str:
    peak = r.headers.get("X-Peak-VRAM-GB")
    secs = r.headers.get("X-Total-Time-S")
    extras = []
    if secs:
        extras.append(f"{secs}s")
    if peak:
        extras.append(f"peak {peak}GB VRAM")
    return f"{t('done')} ({', '.join(extras)})" if extras else t("done")


def _post_wav(endpoint: str, payload: dict) -> tuple:
    try:
        r = httpx.post(f"{API_URL}{endpoint}", json=payload, timeout=120)
        r.raise_for_status()
        return _save_response_wav(r.content), _status_from_response(r)
    except httpx.HTTPStatusError as e:
        return None, f"{t('error_prefix')} {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return None, f"{t('error_prefix')}: {e}"


def _stt_word_segments(audio_path: str) -> list[dict]:
    stt_url = _stt_url()
    if not stt_url:
        print("[trim] STT_URL is not set; using silence fallback")
        return []
    try:
        with open(audio_path, "rb") as f:
            r = httpx.post(
                f"{stt_url}/v1/audio/transcriptions",
                files={"file": ("ref.wav", f, "audio/wav")},
                data={"model": "whisper-turbo", "response_format": "verbose_json"},
                timeout=60,
            )
        r.raise_for_status()
        data = r.json()
        print(f"[trim] STT response: text={data.get('text', '')!r} duration={data.get('duration')}")
        segments = data.get("segments", []) or []
        for i, s in enumerate(segments):
            print(f"[trim]   seg {i}: start={s.get('start')} end={s.get('end')} text={s.get('text', '')!r}")
        valid = [s for s in segments if s.get("end", 0) > 0]
        if not valid:
            print("[trim] STT returned segments without usable timestamps")
        return valid
    except Exception as e:
        print(f"[trim] STT segments failed: {e}")
        return []


def trim_audio(audio_path):
    if not audio_path:
        return None, t("err_no_audio")
    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)

    audio_len_s = len(data) / sr
    if audio_len_s <= MAX_REF_SECONDS:
        print(f"[trim] audio is {audio_len_s:.2f}s, no trim needed")
        return audio_path, t("trim_not_needed", seconds=f"{audio_len_s:.2f}")

    soft_limit = MAX_REF_SECONDS  # 3.5s ideal
    hard_limit = 4.0  # absolute cap so we always stay under 4s

    cut_s = None
    status = None
    segments = _stt_word_segments(audio_path)
    if segments:
        in_window = [s["end"] for s in segments
                     if soft_limit <= s.get("end", 0) <= hard_limit]
        under = [s["end"] for s in segments if s.get("end", 0) <= soft_limit]
        if in_window:
            cut_s = in_window[-1]
            status = t("trim_stt_cut", seconds=f"{cut_s:.2f}")
            print(f"[trim] STT cut at {cut_s:.2f}s (within {soft_limit}-{hard_limit}s window)")
        elif under:
            cut_s = under[-1]
            status = t("trim_stt_cut", seconds=f"{cut_s:.2f}")
            print(f"[trim] STT cut at {cut_s:.2f}s (latest before soft limit)")

    if cut_s is None:
        search_end = min(len(data), int(hard_limit * sr))
        for top_db, min_gap_s in ((35, 0.25), (30, 0.20), (25, 0.15), (20, 0.10)):
            intervals = librosa.effects.split(data[:search_end], top_db=top_db,
                                               frame_length=2048, hop_length=512)
            if len(intervals) == 0:
                continue
            candidates = []
            for i, (_, end) in enumerate(intervals):
                if end / sr > hard_limit:
                    break
                next_start = intervals[i + 1][0] if i + 1 < len(intervals) else search_end
                gap = (next_start - end) / sr
                if gap >= min_gap_s:
                    candidates.append(end / sr)
            if candidates:
                cut_s = candidates[-1]
                status = t("trim_silence_cut", seconds=f"{cut_s:.2f}")
                print(f"[trim] silence cut at {cut_s:.2f}s (top_db={top_db}, gap≥{min_gap_s*1000:.0f}ms)")
                break

    if cut_s is None:
        cut_s = soft_limit
        status = t("trim_hard_cut", seconds=f"{soft_limit:.2f}")
        print(f"[trim] no boundary found — hard cut at {soft_limit}s")

    cut_s = max(1.0, min(cut_s, audio_len_s))
    cut = int(cut_s * sr)
    cut = max(sr, min(cut, len(data)))
    print(f"[trim] final cut: {cut} samples ({cut/sr:.2f}s) of {len(data)} ({audio_len_s:.2f}s)")

    fd, new_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(new_path, data[:cut], sr, subtype="PCM_16")
    return new_path, status or t("trim_done", seconds=f"{cut_s:.2f}")


def transcribe_audio(audio_path, fallback_path=None):
    audio_path = audio_path or fallback_path
    if not audio_path:
        return t("err_no_audio")
    stt_url = _stt_url()
    if not stt_url:
        return t("stt_unconfigured")
    try:
        with open(audio_path, "rb") as f:
            r = httpx.post(
                f"{stt_url}/v1/audio/transcriptions",
                files={"file": ("ref.wav", f, "audio/wav")},
                data={"model": "parakeet-onnx-int8", "response_format": "json"},
                timeout=60,
            )
        r.raise_for_status()
        data = r.json()
        text = data.get("text", "") if isinstance(data, dict) else str(data)
        return text or ""
    except Exception as e:
        return f"{t('stt_error')}: {e}"


def generate_clone(
    text, language, ref_audio, ref_trimmed, ref_text, instruct,
    num_step, guidance_scale, denoise, speed, duration,
    preprocess_prompt, postprocess_output,
):
    if not text or not text.strip():
        return None, t("err_text_empty")
    ref_audio = ref_trimmed or ref_audio
    if not ref_audio:
        return None, t("err_no_ref_audio")

    try:
        fields = {
            "text": text,
            "num_step": str(int(num_step)),
            "guidance_scale": str(float(guidance_scale)),
            "denoise": str(denoise).lower(),
            "speed": str(float(speed)),
            "duration": str(float(duration)),
            "preprocess_prompt": str(preprocess_prompt).lower(),
            "postprocess_output": str(postprocess_output).lower(),
        }
        if language and language != "Auto":
            fields["language"] = language
        if ref_text:
            fields["ref_text"] = ref_text
        if instruct:
            fields["instruct"] = instruct

        with open(ref_audio, "rb") as f:
            r = httpx.post(
                f"{API_URL}/tts/clone",
                data=fields,
                files={"ref_audio": ("ref.wav", f, "audio/wav")},
                timeout=120,
            )
        r.raise_for_status()
        return _save_response_wav(r.content), _status_from_response(r)
    except httpx.HTTPStatusError as e:
        return None, f"{t('error_prefix')} {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return None, f"{t('error_prefix')}: {e}"


def generate_design(
    text, language, instruct,
    num_step, guidance_scale, denoise, speed, duration,
    preprocess_prompt, postprocess_output,
):
    if not text or not text.strip():
        return None, t("err_text_empty")
    if not instruct or not instruct.strip():
        return None, t("err_no_instruct")

    payload = dict(
        text=text,
        language=language if language != "Auto" else None,
        instruct=instruct,
        num_step=int(num_step),
        guidance_scale=float(guidance_scale),
        denoise=bool(denoise),
        speed=float(speed),
        duration=float(duration),
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
    )
    return _post_wav("/tts/design", payload)


def generate_voice(
    voice_id, text, language, instruct,
    num_step, guidance_scale, denoise, speed, duration,
    preprocess_prompt, postprocess_output,
):
    if not voice_id:
        return None, t("err_no_voice_selected")
    if not text or not text.strip():
        return None, t("err_text_empty")

    payload = dict(
        text=text,
        voice_id=voice_id,
        language=language if language != "Auto" else None,
        instruct=instruct or None,
        num_step=int(num_step),
        guidance_scale=float(guidance_scale),
        denoise=bool(denoise),
        speed=float(speed),
        duration=float(duration),
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
    )
    return _post_wav(f"/tts/voice/{voice_id}", payload)


def list_voices():
    try:
        r = httpx.get(f"{API_URL}/voices", timeout=10)
        r.raise_for_status()
        voices = r.json()
        choices = [
            (f"{v['name']} ({t('audio_present') if v['has_audio'] else t('audio_missing')})", v["id"])
            for v in voices
        ]
        return gr.update(choices=choices, value=choices[0][1] if choices else None)
    except Exception:
        return gr.update(choices=[], value=None)


def create_voice(voice_id, name, ref_text, description, ref_audio, ref_trimmed=None):
    ref_audio = ref_trimmed or ref_audio
    if not voice_id or not name or not ref_text:
        return t("err_id_name_text")
    try:
        r = httpx.post(
            f"{API_URL}/voices/{voice_id}",
            json={"name": name, "ref_text": ref_text, "description": description or ""},
            timeout=10,
        )
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        return f"{t('error_prefix')}: {e.response.text}"

    if ref_audio:
        with open(ref_audio, "rb") as f:
            r2 = httpx.put(
                f"{API_URL}/voices/{voice_id}/audio",
                files={"file": ("ref.wav", f, "audio/wav")},
                timeout=30,
            )
        if not r2.is_success:
            return t("voice_audio_upload_failed", detail=r2.text)

    return t("voice_saved", name=name, id=voice_id)


def delete_voice(voice_id):
    if not voice_id:
        return t("err_no_voice_to_delete")
    try:
        r = httpx.delete(f"{API_URL}/voices/{voice_id}", timeout=10)
        r.raise_for_status()
        return t("voice_deleted", id=voice_id)
    except httpx.HTTPStatusError as e:
        return f"{t('error_prefix')}: {e.response.text}"


def model_status():
    try:
        r = httpx.get(f"{API_URL}/models/status", timeout=5)
        d = r.json()
        return t("model_loaded") if d.get("loaded") else t("model_not_loaded")
    except Exception:
        return t("backend_unreachable")


def load_model():
    try:
        r = httpx.post(f"{API_URL}/models/load", timeout=300)
        r.raise_for_status()
        return t("model_loaded_done")
    except Exception as e:
        return f"{t('error_prefix')}: {e}"


with gr.Blocks(title=t("title"), theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {t('title')}\n{t('subtitle')}")

    with gr.Row():
        status_box = gr.Textbox(label=t("model_status"), interactive=False, scale=3)
        refresh_btn = gr.Button(t("refresh_status"), scale=1)
        load_btn = gr.Button(t("load_model"), scale=1)

    refresh_btn.click(model_status, outputs=status_box)
    load_btn.click(load_model, outputs=status_box)
    demo.load(model_status, outputs=status_box)

    with gr.Tabs():

        # --- Voice Clone tab ---
        with gr.Tab(t("tab_clone"), id="clone", render_children=True):
            with gr.Row():
                with gr.Column():
                    clone_text = gr.Textbox(label=t("text"), lines=4, placeholder=t("text_placeholder"))
                    clone_lang = gr.Dropdown(LANGUAGES, value="Auto", label=t("language"))
                    clone_ref_audio = gr.Audio(
                        label=t("ref_audio"),
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    clone_trim_btn = gr.Button(t("trim_clone"), size="sm")
                    clone_ref_trimmed = gr.Audio(label=t("trimmed"), type="filepath")
                    clone_transcribe_btn = gr.Button(t("transcribe"), size="sm")
                    clone_ref_text = gr.Textbox(label=t("ref_text_optional"), lines=2)
                    clone_instruct = gr.Textbox(label=t("instruct_clone"), lines=1)
                    gr.Markdown(t("advanced_options"))
                    clone_num_step = gr.Slider(1, 200, value=32, step=1, label=t("diffusion_steps"))
                    clone_guidance = gr.Slider(0.1, 10.0, value=2.0, step=0.1, label=t("guidance_scale"))
                    with gr.Row():
                        clone_denoise = gr.Checkbox(value=True, label=t("denoise"))
                        clone_preprocess = gr.Checkbox(value=True, label=t("preprocess_prompt"))
                        clone_postprocess = gr.Checkbox(value=True, label=t("postprocess_output"))
                    clone_speed = gr.Slider(0.1, 3.0, value=1.0, step=0.05, label=t("speed"))
                    clone_duration = gr.Number(value=0.0, label=t("duration"))
                    clone_btn = gr.Button(t("synthesize"), variant="primary")
                with gr.Column():
                    clone_audio_out = gr.Audio(label=t("output"), type="filepath")
                    clone_status = gr.Textbox(label=t("status"), interactive=False)

            clone_trim_btn.click(trim_audio, inputs=clone_ref_audio, outputs=[clone_ref_trimmed, clone_status])
            clone_transcribe_btn.click(
                transcribe_audio,
                inputs=[clone_ref_trimmed, clone_ref_audio],
                outputs=clone_ref_text,
            )

            clone_btn.click(
                generate_clone,
                inputs=[
                    clone_text, clone_lang, clone_ref_audio, clone_ref_trimmed, clone_ref_text, clone_instruct,
                    clone_num_step, clone_guidance, clone_denoise, clone_speed, clone_duration,
                    clone_preprocess, clone_postprocess,
                ],
                outputs=[clone_audio_out, clone_status],
            )

        # --- Voice Design tab ---
        with gr.Tab(t("tab_design"), id="design", render_children=True):
            with gr.Row():
                with gr.Column():
                    design_text = gr.Textbox(label=t("text"), lines=4, placeholder=t("text_placeholder"))
                    design_lang = gr.Dropdown(LANGUAGES, value="Auto", label=t("language"))
                    design_instruct = gr.Textbox(
                        label=t("speaker_instruct"),
                        lines=3,
                        placeholder=t("speaker_instruct_placeholder"),
                    )
                    gr.Markdown(t("advanced_options"))
                    design_num_step = gr.Slider(1, 200, value=32, step=1, label=t("diffusion_steps"))
                    design_guidance = gr.Slider(0.1, 10.0, value=2.0, step=0.1, label=t("guidance_scale"))
                    with gr.Row():
                        design_denoise = gr.Checkbox(value=True, label=t("denoise"))
                        design_preprocess = gr.Checkbox(value=True, label=t("preprocess_prompt"))
                        design_postprocess = gr.Checkbox(value=True, label=t("postprocess_output"))
                    design_speed = gr.Slider(0.1, 3.0, value=1.0, step=0.05, label=t("speed"))
                    design_duration = gr.Number(value=0.0, label=t("duration"))
                    design_btn = gr.Button(t("synthesize"), variant="primary")
                with gr.Column():
                    design_audio_out = gr.Audio(label=t("output"), type="filepath")
                    design_status = gr.Textbox(label=t("status"), interactive=False)

            design_btn.click(
                generate_design,
                inputs=[
                    design_text, design_lang, design_instruct,
                    design_num_step, design_guidance, design_denoise, design_speed, design_duration,
                    design_preprocess, design_postprocess,
                ],
                outputs=[design_audio_out, design_status],
            )

        # --- Saved Voices tab ---
        with gr.Tab(t("tab_voices"), id="voices", render_children=True):
            with gr.Row():
                with gr.Column():
                    voice_dropdown = gr.Dropdown(
                        choices=[("(click refresh)", "none")],
                        value="none",
                        label=t("voice"),
                        interactive=True,
                    )
                    voices_refresh_btn = gr.Button(t("refresh_voices"))
                    voice_text = gr.Textbox(label=t("text"), lines=4)
                    voice_lang = gr.Dropdown(LANGUAGES, value="Auto", label=t("language"))
                    voice_instruct = gr.Textbox(label=t("instruct_optional"), lines=1)
                    gr.Markdown(t("advanced_options"))
                    voice_num_step = gr.Slider(1, 200, value=32, step=1, label=t("diffusion_steps"))
                    voice_guidance = gr.Slider(0.1, 10.0, value=2.0, step=0.1, label=t("guidance_scale"))
                    with gr.Row():
                        voice_denoise = gr.Checkbox(value=True, label=t("denoise"))
                        voice_preprocess = gr.Checkbox(value=True, label=t("preprocess_prompt"))
                        voice_postprocess = gr.Checkbox(value=True, label=t("postprocess_output"))
                    voice_speed = gr.Slider(0.1, 3.0, value=1.0, step=0.05, label=t("speed"))
                    voice_duration = gr.Number(value=0.0, label=t("duration"))
                    voice_btn = gr.Button(t("synthesize"), variant="primary")
                with gr.Column():
                    voice_audio_out = gr.Audio(label=t("output"), type="filepath")
                    voice_status = gr.Textbox(label=t("status"), interactive=False)

            voices_refresh_btn.click(list_voices, outputs=voice_dropdown)
            voice_btn.click(
                generate_voice,
                inputs=[
                    voice_dropdown, voice_text, voice_lang, voice_instruct,
                    voice_num_step, voice_guidance, voice_denoise, voice_speed, voice_duration,
                    voice_preprocess, voice_postprocess,
                ],
                outputs=[voice_audio_out, voice_status],
            )

        # --- Manage Voices tab ---
        with gr.Tab(t("tab_manage"), id="manage", render_children=True):
            gr.Markdown(t("create_voice_header"))
            with gr.Row():
                with gr.Column():
                    new_voice_id = gr.Textbox(label=t("voice_id"), placeholder=t("voice_id_placeholder"))
                    new_voice_name = gr.Textbox(label=t("display_name"), placeholder=t("display_name_placeholder"))
                    new_voice_ref_text = gr.Textbox(label=t("ref_text"), lines=2)
                    new_voice_desc = gr.Textbox(label=t("description_optional"), lines=1)
                    new_voice_audio = gr.Audio(
                        label=t("ref_audio"),
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    new_voice_trim_btn = gr.Button(t("trim_clone"), size="sm")
                    new_voice_trimmed = gr.Audio(label=t("trimmed"), type="filepath")
                    new_voice_transcribe_btn = gr.Button(t("transcribe"), size="sm")
                    create_btn = gr.Button(t("create_voice_btn"), variant="primary")
                with gr.Column():
                    manage_status = gr.Textbox(label=t("status"), interactive=False)
                    gr.Markdown(t("delete_voice_header"))
                    del_voice_id = gr.Textbox(label=t("delete_voice_id"))
                    del_btn = gr.Button(t("delete_voice_btn"), variant="stop")

            new_voice_trim_btn.click(trim_audio, inputs=new_voice_audio, outputs=[new_voice_trimmed, manage_status])
            new_voice_transcribe_btn.click(
                transcribe_audio,
                inputs=[new_voice_trimmed, new_voice_audio],
                outputs=new_voice_ref_text,
            )

            create_btn.click(
                create_voice,
                inputs=[new_voice_id, new_voice_name, new_voice_ref_text, new_voice_desc, new_voice_audio, new_voice_trimmed],
                outputs=manage_status,
            )
            del_btn.click(delete_voice, inputs=del_voice_id, outputs=manage_status)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=FRONTEND_PORT)
