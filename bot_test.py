#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import asyncio
import base64
import io
import json
import queue
import wave
import websockets

from typing import Awaitable, Callable
from pydantic.main import BaseModel

from pipecat.frames.frames import AudioRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from loguru import logger


class WebsocketServerParams(TransportParams):
    add_wav_header: bool = False
    audio_frame_size: int = 6400  # 200ms


class WebsocketServerCallbacks(BaseModel):
    on_client_connected: Callable[[websockets.WebSocketServerProtocol], Awaitable[None]]
    on_client_disconnected: Callable[[websockets.WebSocketServerProtocol], Awaitable[None]]

import numpy as np
import base64
import asyncio
from scipy.signal import resample
from faster_whisper import WhisperModel

# μ-law to linear conversion table
mu_law_table = np.array([
    -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
    -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
    -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
    -11900, -11388, -10876, -10364, -9852, -9340, -8828, -8316,
    -7932, -7676, -7420, -7164, -6908, -6652, -6396, -6140,
    -5884, -5628, -5372, -5116, -4860, -4604, -4348, -4092,
    -3900, -3772, -3644, -3516, -3388, -3260, -3132, -3004,
    -2876, -2748, -2620, -2492, -2364, -2236, -2108, -1980,
    -1884, -1820, -1756, -1692, -1628, -1564, -1500, -1436,
    -1372, -1308, -1244, -1180, -1116, -1052, -988, -924,
    -876, -844, -812, -780, -748, -716, -684, -652,
    -620, -588, -556, -524, -492, -460, -428, -396,
    -372, -356, -340, -324, -308, -292, -276, -260,
    -244, -228, -212, -196, -180, -164, -148, -132,
    -120, -112, -104, -96, -88, -80, -72, -64,
    -56, -48, -40, -32, -24, -16, -8, 0,
    32124, 31100, 30076, 29052, 28028, 27004, 25980, 24956,
    23932, 22908, 21884, 20860, 19836, 18812, 17788, 16764,
    15996, 15484, 14972, 14460, 13948, 13436, 12924, 12412,
    11900, 11388, 10876, 10364, 9852, 9340, 8828, 8316,
    7932, 7676, 7420, 7164, 6908, 6652, 6396, 6140,
    5884, 5628, 5372, 5116, 4860, 4604, 4348, 4092,
    3900, 3772, 3644, 3516, 3388, 3260, 3132, 3004,
    2876, 2748, 2620, 2492, 2364, 2236, 2108, 1980,
    1884, 1820, 1756, 1692, 1628, 1564, 1500, 1436,
    1372, 1308, 1244, 1180, 1116, 1052, 988, 924,
    876, 844, 812, 780, 748, 716, 684, 652,
    620, 588, 556, 524, 492, 460, 428, 396,
    372, 356, 340, 324, 308, 292, 276, 260,
    244, 228, 212, 196, 180, 164, 148, 132,
    120, 112, 104, 96, 88, 80, 72, 64,
    56, 48, 40, 32, 24, 16, 8, 0
], dtype=np.int16)


def pcmu_to_pcm(pc_str):
    pc_bytes = base64.b64decode(pc_str)
    pcm_samples = np.empty(len(pc_bytes), dtype=np.int16)
    for i, byte in enumerate(pc_bytes):
        pcm_samples[i] = mu_law_table[byte]
    return pcm_samples.tobytes()


def resample_audio(audio_data, orig_sr, target_sr):
    duration = len(audio_data) / orig_sr
    target_length = int(duration * target_sr)
    return resample(audio_data, target_length).astype(np.int16)


def process_audio(pcm_data):
    sample_rate = 8000
    target_sample_rate = 16000

    # Convert PCM data to numpy array
    pcm_samples = np.frombuffer(pcm_data, dtype=np.int16)

    # Resample audio to 16000 Hz
    resampled_pcm = resample_audio(pcm_samples, sample_rate, target_sample_rate)

    # Convert to float32
    audio_float = resampled_pcm.astype(np.float32) / 32768.0

    return audio_float.tobytes()


class WebsocketServerInputTransport(BaseInputTransport):

    def __init__(
            self,
            host: str,
            port: int,
            params: WebsocketServerParams,
            callbacks: WebsocketServerCallbacks):
        super().__init__(params)

        self._host = host
        self._port = port
        self._params = params
        self._callbacks = callbacks

        self._websocket: websockets.WebSocketServerProtocol | None = None

        self._client_audio_queue = queue.Queue()
        self._stop_server_event = asyncio.Event()

    async def start(self, frame: StartFrame):
        self._server_task = self.get_event_loop().create_task(self._server_task_handler())
        await super().start(frame)

    async def stop(self):
        self._stop_server_event.set()
        await self._server_task
        await super().stop()

    def read_next_audio_frame(self) -> AudioRawFrame | None:
        try:
            return self._client_audio_queue.get(timeout=1)
        except queue.Empty:
            return None

    async def _server_task_handler(self):
        logger.info(f"Starting websocket server on {self._host}:{self._port}")
        async with websockets.serve(self._client_handler, self._host, self._port) as server:
            await self._stop_server_event.wait()

    async def _client_handler(self, websocket: websockets.WebSocketServerProtocol, path):
        logger.info(f"New client connection from {websocket.remote_address}")
        if self._websocket:
            await self._websocket.close()
            logger.warning("Only one client connected, using new connection")

        self._websocket = websocket

        # Notify
        await self._callbacks.on_client_connected(websocket)

        # Handle incoming messages
        async for message in websocket:
            data = json.loads(message)
            # print(data, flush=True)
            global sid
            sid = data['streamSid'] if data.get("streamSid") else None
            if data['event'] == 'media':

                payload = data['media']['payload']
                pcm_data = pcmu_to_pcm(payload)
                audio_bytes = process_audio(pcm_data)
                frame = AudioRawFrame(audio=audio_bytes, num_channels=1, sample_rate=16000)
                if isinstance(frame, AudioRawFrame) and self._params.audio_in_enabled:
                    self._client_audio_queue.put_nowait(frame)
                else:
                    await self._internal_push_frame(frame)

        # Notify disconnection
        await self._callbacks.on_client_disconnected(websocket)

        await self._websocket.close()
        self._websocket = None

        logger.info(f"Client {websocket.remote_address} disconnected")


class WebsocketServerOutputTransport(BaseOutputTransport):

    def __init__(self, params: WebsocketServerParams):
        super().__init__(params)

        self._params = params

        self._websocket: websockets.WebSocketServerProtocol | None = None

        self._audio_buffer = bytes()

    async def set_client_connection(self, websocket: websockets.WebSocketServerProtocol | None):
        if self._websocket:
            await self._websocket.close()
            logger.warning("Only one client allowed, using new connection")
        self._websocket = websocket

    def write_raw_audio_frames(self, frames: bytes):
        self._audio_buffer += frames
        global sid
        while len(self._audio_buffer) >= self._params.audio_frame_size:
            frame = AudioRawFrame(
                audio=self._audio_buffer[:self._params.audio_frame_size],
                sample_rate=self._params.audio_out_sample_rate,
                num_channels=self._params.audio_out_channels
            )

            if self._params.add_wav_header:
                content = io.BytesIO()
                ww = wave.open(content, "wb")
                ww.setsampwidth(2)
                ww.setnchannels(frame.num_channels)
                ww.setframerate(frame.sample_rate)
                ww.writeframes(frame.audio)
                ww.close()
                content.seek(0)
                wav_frame = AudioRawFrame(
                    content.read(),
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels)
                frame = wav_frame

            # proto = self._params.serializer.serialize(frame)
            # proto = base64.b64decode(frame.audio)

            payload = base64.b64encode(frame.audio).decode('utf-8')
            answer_dict = {"event": "media",
                           "streamSid": sid,
                           "media": {"payload": payload}}

            future = asyncio.run_coroutine_threadsafe(
                self._websocket.send(json.dumps(answer_dict)), self.get_event_loop())
            future.result()

            self._audio_buffer = self._audio_buffer[self._params.audio_frame_size:]


class WebsocketServerTransport(BaseTransport):

    def __init__(
            self,
            host: str = "localhost",
            port: int = 8765,
            params: WebsocketServerParams = WebsocketServerParams(),
            loop: asyncio.AbstractEventLoop | None = None):
        super().__init__(loop)
        self._host = host
        self._port = port
        self._params = params

        self._callbacks = WebsocketServerCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected
        )
        self._input: WebsocketServerInputTransport | None = None
        self._output: WebsocketServerOutputTransport | None = None
        self._websocket: websockets.WebSocketServerProtocol | None = None

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")

    def input(self) -> FrameProcessor:
        if not self._input:
            self._input = WebsocketServerInputTransport(
                self._host, self._port, self._params, self._callbacks)
        return self._input

    def output(self) -> FrameProcessor:
        if not self._output:
            self._output = WebsocketServerOutputTransport(self._params)
        return self._output

    async def _on_client_connected(self, websocket):
        if self._output:
            await self._output.set_client_connection(websocket)
            await self._call_event_handler("on_client_connected", websocket)
        else:
            logger.error("A WebsocketServerTransport output is missing in the pipeline")

    async def _on_client_disconnected(self, websocket):
        if self._output:
            await self._output.set_client_connection(None)
            await self._call_event_handler("on_client_disconnected", websocket)
        else:
            logger.error("A WebsocketServerTransport output is missing in the pipeline")

