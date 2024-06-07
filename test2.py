"""This module implements Whisper transcription with a locally-downloaded model."""

import asyncio
import base64
import json
import time
from enum import Enum
from typing import AsyncGenerator
import numpy as np
import websockets
from loguru import logger
from pydub import AudioSegment
from io import BytesIO

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.ai_services import STTService

try:
    from faster_whisper import WhisperModel
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Whisper, you need to `pip install pipecat-ai[whisper]`.")
    raise Exception(f"Missing module: {e}")


audio_buffer = bytearray()

class Model(Enum):
    """Class of basic Whisper model selection options"""
    TINY = "tiny"
    BASE = "base"
    MEDIUM = "medium"
    LARGE = "large-v3"
    DISTIL_LARGE_V2 = "Systran/faster-distil-whisper-large-v2"
    DISTIL_MEDIUM_EN = "Systran/faster-distil-whisper-medium.en"


class WhisperSTTService(STTService):
    """Class to transcribe audio with a locally-downloaded Whisper model"""

    def __init__(self,
                 model: Model = Model.DISTIL_MEDIUM_EN,
                 device: str = "auto",
                 compute_type: str = "default",
                 no_speech_prob: float = 0.1,
                 **kwargs):

        super().__init__(**kwargs)
        self._device: str = device
        self._compute_type = compute_type
        self._model_name: Model = model
        self._no_speech_prob = no_speech_prob
        self._model: WhisperModel | None = None
        self._load()

    def _load(self):
        """Loads the Whisper model. Note that if this is the first time
        this model is being run, it will take time to download."""
        logger.debug("Loading Whisper model...")
        self._model = WhisperModel(
            self._model_name.value,
            device=self._device,
            compute_type=self._compute_type)
        logger.debug("Loaded Whisper model")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribes given audio using Whisper"""
        if not self._model:
            yield ErrorFrame("Whisper model not available")
            logger.error("Whisper model not available")
            return

        # Divide by 32768 because we have signed 16-bit data.
        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        segments, _ = await asyncio.to_thread(self._model.transcribe, audio_float)
        text: str = ""
        for segment in segments:
            if segment.no_speech_prob < self._no_speech_prob:
                text += f"{segment.text} "

        if text:
            yield TranscriptionFrame(text, "", int(time.time_ns() / 1000000))


def pcmu_to_pcm(data: bytes) -> bytes:
    """
    Convert PCMU data to PCM and resample to 16000 Hz.
    """
    audio = AudioSegment.from_raw(BytesIO(data), sample_width=1, frame_rate=8000, channels=1, codec="ulaw")
    audio = audio.set_frame_rate(16000)

    # Ensure the audio is exported as PCM with signed 16-bit samples
    pcm_io = BytesIO()
    audio.export(pcm_io, format="wav", codec="pcm_s16le")
    return pcm_io.getvalue()


async def consumer_handler(websocket):
    global audio_buffer
    try:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            if 'media' in data and 'payload' in data['media']:
                payload_base64 = data['media']['payload']
                # Decode the base64 string to binary data
                audio_data = base64.b64decode(payload_base64)
                # Append this data to the buffer
                audio_buffer.extend(audio_data)
                # Check if the buffer has 5 seconds of audio
                while len(audio_buffer) >= 40000:
                    # Convert buffer to PCM bytes
                    pcm_bytes = pcmu_to_pcm(audio_buffer[:40000])
                    # Remove the WAV header before passing to run_stt
                    pcm_bytes = pcm_bytes[44:]
                    # Run STT
                    whisper_service = WhisperSTTService()
                    async for transcription in whisper_service.run_stt(pcm_bytes):
                        print(transcription)
                    # Remove the processed audio from the buffer
                    audio_buffer = audio_buffer[40000:]
    except websockets.exceptions.ConnectionClosed:
        print("Connection closed")


async def server():
    async with websockets.serve(consumer_handler, "localhost", 8765):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(server())
