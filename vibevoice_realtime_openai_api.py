#!/usr/bin/env python3
"""
VibeVoice OpenAI-Compatible TTS Server

A FastAPI server that wraps VibeVoice-Realtime-0.5B with an OpenAI-compatible API,
enabling integration with Open WebUI and other OpenAI TTS-compatible applications.

Usage:
    python vibevoice_realtime_openai_api.py --port 8880

Streaming:
    True incremental streaming is supported via VibeVoice's AsyncAudioStreamer.
    The model emits PCM audio chunks as they are generated (not after full completion).

    Raw audio stream (PCM only):
        curl -N -X POST http://localhost:8880/v1/audio/speech \\
          -H "Content-Type: application/json" \\
          -d '{"model":"tts-1","voice":"Emma","input":"Hello world","response_format":"pcm","stream":true}'

    SSE stream (any format, base64-encoded chunks):
        curl -N -X POST http://localhost:8880/v1/audio/speech \\
          -H "Content-Type: application/json" \\
          -d '{"model":"tts-1","voice":"Emma","input":"Hello world","response_format":"pcm","stream":true,"stream_format":"sse"}'

    PCM playback via ffplay:
        curl -sN -X POST http://localhost:8880/v1/audio/speech \\
          -H "Content-Type: application/json" \\
          -d '{"model":"tts-1","voice":"Emma","input":"Hello world","response_format":"pcm","stream":true}' \\
          | ffplay -f s16le -ar 24000 -ac 1 -
"""

import argparse
import asyncio
import base64
import copy
import io
import json
import os
import subprocess
import threading
import time
import traceback
import urllib.request
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Any
from contextlib import asynccontextmanager

# Set HuggingFace cache BEFORE importing any HF libraries
MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).parent / "models"))
os.environ["HF_HOME"] = str(MODELS_DIR / "huggingface")

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import scipy.io.wavfile as wavfile

# VibeVoice imports (after setting HF_HOME)
from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.modular.streamer import AsyncAudioStreamer
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

SAMPLE_RATE = 24000
DEFAULT_MODEL_PATH = "microsoft/VibeVoice-Realtime-0.5B"

CFG_SCALE = float(os.environ.get("CFG_SCALE", "1.25"))

VOICES_DIR = MODELS_DIR / "voices"

VOICE_PRESETS = {
    "Carter": "en-Carter_man.pt",
    "Davis": "en-Davis_man.pt",
    "Emma": "en-Emma_woman.pt",
    "Frank": "en-Frank_man.pt",
    "Grace": "en-Grace_woman.pt",
    "Mike": "en-Mike_man.pt",
    "Samuel": "in-Samuel_man.pt",
}

VOICE_BASE_URL = "https://github.com/microsoft/VibeVoice/raw/main/demo/voices/streaming_model"

OPENAI_TO_VIBEVOICE_MAP = {
    "alloy": "Carter",
    "echo": "Davis",
    "fable": "Emma",
    "onyx": "Frank",
    "nova": "Grace",
    "shimmer": "Mike",
}

SUPPORTED_FORMATS = ["mp3", "wav", "opus", "flac", "aac", "pcm"]

# Only PCM can be streamed as raw audio — compressed formats need the full file
# to write headers / encode properly. SSE streaming works for all formats because
# we base64-encode raw PCM chunks regardless of the requested output format.
STREAMING_AUDIO_FORMATS = ["pcm"]

# ------------------------------------------------------------------------------
# Model Download Utilities
# ------------------------------------------------------------------------------

def ensure_voices_downloaded() -> None:
    """Download voice presets if not present"""
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    for voice_name, filename in VOICE_PRESETS.items():
        voice_path = VOICES_DIR / filename
        if not voice_path.exists():
            url = f"{VOICE_BASE_URL}/{filename}"
            print(f"[download] Downloading voice preset: {voice_name}...")
            try:
                urllib.request.urlretrieve(url, voice_path)
                print(f"[download] Downloaded {filename}")
            except Exception as e:
                print(f"[error] Failed to download {filename}: {e}")


def get_model_cache_dir() -> str:
    model_cache = MODELS_DIR / "huggingface"
    model_cache.mkdir(parents=True, exist_ok=True)
    return str(model_cache)


# ------------------------------------------------------------------------------
# Pydantic Models
# ------------------------------------------------------------------------------

class TTSRequest(BaseModel):
    """OpenAI-compatible TTS request"""
    input: str = Field(..., description="Text to synthesize", max_length=4096)
    voice: str = Field(default="Carter", description="Voice ID")
    model: str = Field(default="tts-1", description="Model ID (ignored, for compatibility)")
    response_format: str = Field(default="mp3", description="Audio format")
    speed: float = Field(default=1.0, description="Speed (not yet supported)")
    stream: bool = Field(default=False, description="Enable streaming response")
    stream_format: str = Field(
        default="audio",
        description=(
            "Streaming format when stream=true. "
            "'audio' = raw audio bytes (PCM only); "
            "'sse' = Server-Sent Events with base64-encoded PCM chunks"
        ),
    )


class VoiceInfo(BaseModel):
    voice_id: str
    name: str
    type: str
    gender: Optional[str] = None


class VoicesResponse(BaseModel):
    voices: List[VoiceInfo]


class StreamingCapabilities(BaseModel):
    audio_chunked: bool
    sse: bool
    true_incremental_generation: bool
    streaming_audio_formats: List[str]
    note: str


class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    device: str
    features: Dict[str, Any]


# ------------------------------------------------------------------------------
# TTS Service
# ------------------------------------------------------------------------------

class VibeVoiceTTSService:
    """Service for managing VibeVoice model and generating speech"""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.processor: Optional[VibeVoiceStreamingProcessor] = None
        self.model: Optional[VibeVoiceStreamingForConditionalGenerationInference] = None
        self.voice_presets: Dict[str, Path] = {}
        self._voice_cache: Dict[str, Any] = {}
        self._torch_device = torch.device(device)
        # Serialize model.generate() calls — the model is not thread-safe
        self._generate_lock = asyncio.Lock()

    def load(self) -> None:
        """Load model and voice presets"""
        os.environ["HF_HOME"] = get_model_cache_dir()
        ensure_voices_downloaded()

        print(f"[startup] Loading processor from {self.model_path}")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

        if self.device == "cuda":
            load_dtype = torch.bfloat16
            device_map = "cuda"
            attn_impl = "flash_attention_2"
        elif self.device == "mps":
            load_dtype = torch.float32
            device_map = None
            attn_impl = "sdpa"
        else:
            load_dtype = torch.float32
            device_map = "cpu"
            attn_impl = "sdpa"

        print(f"[startup] Loading model with dtype={load_dtype}, attn={attn_impl}")

        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=load_dtype,
                device_map=device_map,
                attn_implementation=attn_impl,
            )
            if self.device == "mps":
                self.model.to("mps")
        except Exception as e:
            if attn_impl == "flash_attention_2":
                print(f"[startup] Flash Attention failed, falling back to SDPA: {e}")
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=device_map,
                    attn_implementation="sdpa",
                )
            else:
                raise

        self.model.eval()
        self.model.set_ddpm_inference_steps(num_steps=5)
        self._load_voice_presets()
        print(f"[startup] Model ready on {self.device}")

    def _load_voice_presets(self) -> None:
        if not VOICES_DIR.exists():
            print(f"[warning] Voices directory not found: {VOICES_DIR}")
            return

        for pt_file in VOICES_DIR.glob("*.pt"):
            full_name = pt_file.stem
            self.voice_presets[full_name] = pt_file
            short_name = full_name
            if "_" in short_name:
                short_name = short_name.split("_")[0]
            if "-" in short_name:
                short_name = short_name.split("-")[-1]
            self.voice_presets[short_name] = pt_file

        print(f"[startup] Found {len(self.voice_presets)} voice presets")

    def get_available_voices(self) -> List[VoiceInfo]:
        voices = []
        seen: set = set()

        for openai_name in OPENAI_TO_VIBEVOICE_MAP:
            voices.append(VoiceInfo(voice_id=openai_name, name=openai_name, type="openai-compatible"))

        for name, path in self.voice_presets.items():
            if name not in seen and "-" not in name:
                path_stem = path.stem
                gender = "female" if "_woman" in path_stem else "male" if "_man" in path_stem else None
                voices.append(VoiceInfo(voice_id=name, name=name, type="vibevoice-native", gender=gender))
                seen.add(name)

        return voices

    def _resolve_voice(self, voice: str) -> str:
        if voice.lower() in OPENAI_TO_VIBEVOICE_MAP:
            voice = OPENAI_TO_VIBEVOICE_MAP[voice.lower()]
        if voice not in self.voice_presets:
            available = [v for v in self.voice_presets.keys() if "-" not in v]
            print(f"[warning] Voice '{voice}' not found, using 'Carter'. Available: {available}")
            voice = "Carter"
        return voice

    def _get_voice_prompt(self, voice: str) -> Any:
        if voice not in self._voice_cache:
            voice_path = self.voice_presets[voice]
            print(f"[tts] Loading voice prompt from {voice_path}")
            self._voice_cache[voice] = torch.load(
                voice_path, map_location=self._torch_device, weights_only=False
            )
        return self._voice_cache[voice]

    def _build_inputs(self, text: str, voice: str) -> tuple:
        """Prepare model inputs. Returns (inputs_dict, prefilled_outputs)."""
        voice = self._resolve_voice(voice)
        prefilled_outputs = self._get_voice_prompt(voice)
        text = text.strip().replace("\u2019", "'")

        inputs = self.processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=prefilled_outputs,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self._torch_device)

        return inputs, prefilled_outputs

    # ------------------------------------------------------------------
    # Non-streaming (original behaviour)
    # ------------------------------------------------------------------

    def generate_speech(self, text: str, voice: str, cfg_scale: float = 1.5) -> np.ndarray:
        """Generate speech and return the complete audio array."""
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded")

        inputs, prefilled_outputs = self._build_inputs(text, voice)
        print(f"[tts] Generating speech for {len(text)} chars with voice '{voice}'")
        start_time = time.time()

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            tokenizer=self.processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
            all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
        )

        elapsed = time.time() - start_time

        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            audio = outputs.speech_outputs[0]
            if torch.is_tensor(audio):
                audio = audio.detach().cpu().to(torch.float32).numpy()
            else:
                audio = np.asarray(audio, dtype=np.float32)

            if audio.ndim > 1:
                audio = audio.reshape(-1)

            peak = np.max(np.abs(audio))
            if peak > 1.0:
                audio = audio / peak

            duration = len(audio) / SAMPLE_RATE
            rtf = elapsed / duration if duration > 0 else float("inf")
            print(f"[tts] Generated {duration:.2f}s audio in {elapsed:.2f}s (RTF: {rtf:.2f}x)")
            return audio
        else:
            raise RuntimeError("No audio output generated")

    # ------------------------------------------------------------------
    # True incremental streaming via AsyncAudioStreamer
    # ------------------------------------------------------------------

    async def generate_speech_streaming(
        self, text: str, voice: str, cfg_scale: float = 1.5
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Yield PCM float32 audio chunks as they are produced by the model.

        The model's generate() is run in a background thread so it doesn't
        block the asyncio event loop. AsyncAudioStreamer bridges the thread
        boundary via asyncio.Queue / call_soon_threadsafe.
        """
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded")

        inputs, prefilled_outputs = self._build_inputs(text, voice)
        print(f"[tts] Streaming speech for {len(text)} chars with voice '{voice}'")

        # AsyncAudioStreamer requires a running event loop at construction time
        streamer = AsyncAudioStreamer(batch_size=1, stop_signal=None, timeout=60.0)

        # Run model.generate() in a thread pool so the event loop stays free
        loop = asyncio.get_event_loop()
        generate_exception: list = []  # capture thread exceptions

        def _run_generate():
            try:
                self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={"do_sample": False},
                    verbose=False,
                    all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
                    audio_streamer=streamer,
                )
            except Exception as exc:
                generate_exception.append(exc)
                # Make sure the streamer is ended so the consumer unblocks
                try:
                    loop.call_soon_threadsafe(streamer.audio_queues[0].put_nowait, None)
                except Exception:
                    pass

        thread = threading.Thread(target=_run_generate, daemon=True)
        thread.start()

        # Consume chunks from the async queue as they arrive
        async for chunk in streamer.get_stream(0):
            if chunk is None:
                break
            if torch.is_tensor(chunk):
                chunk = chunk.to(torch.float32).numpy()
            else:
                chunk = np.asarray(chunk, dtype=np.float32)
            if chunk.ndim > 1:
                chunk = chunk.reshape(-1)
            yield chunk

        thread.join(timeout=120)

        if generate_exception:
            raise generate_exception[0]


# ------------------------------------------------------------------------------
# Audio Format Conversion
# ------------------------------------------------------------------------------

def audio_float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float32 audio to raw PCM16LE bytes."""
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def convert_audio(audio: np.ndarray, format: str, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert complete audio array to the requested format."""
    format = format.lower()

    if format == "pcm":
        return audio_float32_to_pcm16(audio)

    if format == "wav":
        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16))
        return buffer.getvalue()

    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, sample_rate, (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16))
    wav_data = wav_buffer.getvalue()

    format_args = {
        "mp3": ["-f", "mp3", "-codec:a", "libmp3lame", "-q:a", "2"],
        "opus": ["-f", "opus", "-codec:a", "libopus"],
        "flac": ["-f", "flac", "-codec:a", "flac"],
        "aac": ["-f", "adts", "-codec:a", "aac"],
    }

    if format not in format_args:
        raise ValueError(f"Unsupported format: {format}")

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "wav", "-i", "pipe:0",
        *format_args[format],
        "pipe:1",
    ]

    try:
        result = subprocess.run(cmd, input=wav_data, capture_output=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"[error] ffmpeg failed: {e.stderr.decode()}")
        raise RuntimeError(f"Audio conversion failed: {e}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg.")


def get_content_type(format: str) -> str:
    types = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "opus": "audio/opus",
        "flac": "audio/flac",
        "aac": "audio/aac",
        "pcm": "audio/pcm",
    }
    return types.get(format.lower(), "application/octet-stream")


# ------------------------------------------------------------------------------
# Streaming generators
# ------------------------------------------------------------------------------

async def _stream_raw_audio(
    service: "VibeVoiceTTSService", text: str, voice: str, cfg_scale: float
) -> AsyncGenerator[bytes, None]:
    """
    Yield raw PCM16LE bytes as the model produces them.
    Suitable for response_format='pcm' with stream=true, stream_format='audio'.
    """
    async with service._generate_lock:
        async for chunk in service.generate_speech_streaming(text, voice, cfg_scale):
            yield audio_float32_to_pcm16(chunk)


async def _stream_sse(
    service: "VibeVoiceTTSService", text: str, voice: str, cfg_scale: float
) -> AsyncGenerator[bytes, None]:
    """
    Yield Server-Sent Events with base64-encoded PCM16LE audio chunks.

    Event format (OpenAI-style):
        event: audio.delta
        data: {"delta": "<base64-encoded PCM16LE bytes>"}

    Final event:
        event: audio.done
        data: {}
    """
    async with service._generate_lock:
        async for chunk in service.generate_speech_streaming(text, voice, cfg_scale):
            pcm_bytes = audio_float32_to_pcm16(chunk)
            b64 = base64.b64encode(pcm_bytes).decode("ascii")
            payload = json.dumps({"delta": b64})
            yield f"event: audio.delta\ndata: {payload}\n\n".encode()

    yield b"event: audio.done\ndata: {}\n\n"


# ------------------------------------------------------------------------------
# FastAPI Application
# ------------------------------------------------------------------------------

tts_service: Optional[VibeVoiceTTSService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_service

    model_path = os.environ.get("VIBEVOICE_MODEL_PATH", DEFAULT_MODEL_PATH)
    device = os.environ.get("VIBEVOICE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    tts_service = VibeVoiceTTSService(model_path=model_path, device=device)
    try:
        tts_service.load()
    except Exception as e:
        print(f"[FATAL] Model loading failed: {e}")
        traceback.print_exc()

    yield

    if tts_service and tts_service.model:
        del tts_service.model
        torch.cuda.empty_cache()


app = FastAPI(
    title="VibeVoice TTS Server",
    description="OpenAI-compatible TTS API powered by VibeVoice-Realtime-0.5B",
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        service="vibevoice-realtime-openai-api",
        model_loaded=tts_service is not None and tts_service.model is not None,
        device=tts_service.device if tts_service else "unknown",
        features={
            "streaming": {
                "audio_chunked": True,
                "sse": True,
                "true_incremental_generation": True,
                "streaming_audio_formats": STREAMING_AUDIO_FORMATS,
                "note": (
                    "True incremental streaming via VibeVoice AsyncAudioStreamer. "
                    "stream_format='audio' requires response_format='pcm'. "
                    "stream_format='sse' works with any response_format (chunks are raw PCM16LE base64)."
                ),
            },
            "formats": SUPPORTED_FORMATS,
            "sample_rate": SAMPLE_RATE,
        },
    )


@app.get("/v1/audio/voices", response_model=VoicesResponse)
async def list_voices():
    if not tts_service:
        raise HTTPException(status_code=503, detail="Service not ready")
    return VoicesResponse(voices=tts_service.get_available_voices())


@app.get("/v1/audio/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "tts-1",
                "object": "model",
                "created": 1699000000,
                "owned_by": "vibevoice",
                "name": "VibeVoice-Realtime-0.5B",
            },
            {
                "id": "tts-1-hd",
                "object": "model",
                "created": 1699000000,
                "owned_by": "vibevoice",
                "name": "VibeVoice-Realtime-0.5B",
            },
        ],
    }


@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    """
    Generate speech from text (OpenAI-compatible).

    When stream=false (default): returns the complete audio file.
    When stream=true:
      - stream_format='audio' (default): streams raw PCM16LE bytes.
        response_format must be 'pcm'; returns HTTP 400 for other formats.
      - stream_format='sse': streams Server-Sent Events with base64-encoded
        PCM16LE chunks. Works regardless of response_format.
    """
    if not tts_service:
        raise HTTPException(status_code=503, detail="Service not ready")

    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required")

    if len(request.input) > 4096:
        raise HTTPException(status_code=400, detail="Input text exceeds 4096 characters")

    fmt = request.response_format.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{fmt}'. Supported: {SUPPORTED_FORMATS}",
        )

    stream_fmt = request.stream_format.lower()
    if stream_fmt not in ("audio", "sse"):
        raise HTTPException(
            status_code=400,
            detail="stream_format must be 'audio' or 'sse'",
        )

    # ------------------------------------------------------------------
    # Streaming path
    # ------------------------------------------------------------------
    if request.stream:
        if stream_fmt == "audio":
            # Raw audio streaming only makes sense for PCM — compressed formats
            # need complete data to write headers / encode frames.
            if fmt not in STREAMING_AUDIO_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"stream_format='audio' only supports response_format values: "
                        f"{STREAMING_AUDIO_FORMATS}. "
                        f"Use stream_format='sse' to stream other formats as base64 SSE events, "
                        f"or set response_format='pcm'."
                    ),
                )
            return StreamingResponse(
                _stream_raw_audio(tts_service, request.input, request.voice, CFG_SCALE),
                media_type="audio/pcm",
                headers={"X-Accel-Buffering": "no"},
            )

        else:  # sse
            return StreamingResponse(
                _stream_sse(tts_service, request.input, request.voice, CFG_SCALE),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

    # ------------------------------------------------------------------
    # Non-streaming path (original behaviour)
    # ------------------------------------------------------------------
    try:
        audio = tts_service.generate_speech(
            text=request.input,
            voice=request.voice,
            cfg_scale=CFG_SCALE,
        )
        audio_bytes = convert_audio(audio, fmt)
        content_type = get_content_type(fmt)

        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename=speech.{fmt}"},
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VibeVoice OpenAI-Compatible TTS Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8880)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    os.environ["VIBEVOICE_MODEL_PATH"] = args.model_path
    os.environ["VIBEVOICE_DEVICE"] = args.device

    print(f"Starting VibeVoice TTS Server on http://{args.host}:{args.port}")
    print(f"OpenAI TTS endpoint: http://{args.host}:{args.port}/v1/audio/speech")
    print()
    print("Streaming examples:")
    print(f"  Raw PCM:  curl -sN -X POST http://{args.host}:{args.port}/v1/audio/speech \\")
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"model":"tts-1","voice":"Emma","input":"Hello","response_format":"pcm","stream":true}}\'')
    print()
    print(f"  SSE:      curl -sN -X POST http://{args.host}:{args.port}/v1/audio/speech \\")
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"model":"tts-1","voice":"Emma","input":"Hello","stream":true,"stream_format":"sse"}}\'')
    print()
    print(f"  ffplay:   curl -sN ... | ffplay -f s16le -ar 24000 -ac 1 -")

    uvicorn.run(
        "vibevoice_realtime_openai_api:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
