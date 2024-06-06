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


import base64
import numpy as np
from pydub import AudioSegment
from io import BytesIO


def pcmu_to_pcm(pcmu_data):
    # G.711 u-law to linear PCM conversion
    pcm = np.zeros(len(pcmu_data), dtype=np.int16)
    for i, byte in enumerate(pcmu_data):
        pcm[i] = ulaw2linear(byte)
    return pcm.tobytes()


def ulaw2linear(ulawbyte):
    """Convert a u-law byte to a linear 16-bit PCM value."""
    BIAS = 0x84
    CLIP = 32635
    exp_lut = [
        0, 132, 396, 924, 1980, 4092, 8316, 16764
    ]
    ulawbyte = ~ulawbyte
    sign = (ulawbyte & 0x80)
    exponent = (ulawbyte >> 4) & 0x07
    mantissa = ulawbyte & 0x0F
    sample = exp_lut[exponent] + (mantissa << (exponent + 3))
    if sign != 0:
        sample = -sample
    return sample


def transform_audio(payload: str) -> bytes:
    # Step 1: Decode base64 payload
    pcmu_data = base64.b64decode(payload)

    # Step 2: Convert PCMU to PCM
    pcm_data = pcmu_to_pcm(pcmu_data)

    # Step 3: Create an AudioSegment from PCM data with a sample rate of 8000 Hz
    audio_segment = AudioSegment(
        data=pcm_data,
        sample_width=2,  # 16-bit audio
        frame_rate=8000,
        channels=1
    )

    # Step 4: Resample audio to 16000 Hz
    resampled_audio = audio_segment.set_frame_rate(16000)

    # Step 5: Export audio to bytes (16-bit PCM)
    pcm_bytes_io = BytesIO()
    resampled_audio.export(pcm_bytes_io, format="raw")
    pcm_bytes = pcm_bytes_io.getvalue()

    return pcm_bytes


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

                payload = transform_audio(data['media']['payload'])
                frame = AudioRawFrame(audio=payload, num_channels=1, sample_rate=16000)
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

