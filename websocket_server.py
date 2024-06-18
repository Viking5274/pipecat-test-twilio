import asyncio
import audioop
import base64
import io
import json
import wave
import websockets

from typing import Awaitable, Callable
from pydantic.main import BaseModel

from pipecat.frames.frames import AudioRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from loguru import logger


class WebsocketServerParams(TransportParams):
    add_wav_header: bool = False
    audio_frame_size: int = 6400  # 200ms
    serializer: FrameSerializer = ProtobufFrameSerializer()


class WebsocketServerCallbacks(BaseModel):
    on_client_connected: Callable[[websockets.WebSocketServerProtocol], Awaitable[None]]
    on_client_disconnected: Callable[[websockets.WebSocketServerProtocol], Awaitable[None]]


def mulaw_to_pcm16(data):
    # Convert mu-law to PCM
    pcm_data = audioop.ulaw2lin(data, 2)
    return pcm_data


def resample(pcm_data, src_rate, dst_rate):
    # Resample PCM data from src_rate to dst_rate
    resampled_data = audioop.ratecv(pcm_data, 2, 1, src_rate, dst_rate, None)[0]
    return resampled_data


def process_audio_chunk(data_chunk):
    # Decode base64 if necessary
    # import base64
    # data_chunk = base64.b64decode(data_chunk)

    # Convert mu-law to PCM
    pcm_data = mulaw_to_pcm16(data_chunk)

    # Resample PCM data from 8000 Hz to 16000 Hz
    resampled_pcm_data = resample(pcm_data, 8000, 16000)

    return resampled_pcm_data


def pcm_16000_to_ulaw_8000(pcm_16000_bytes):
    # Resample from 16000 Hz to 8000 Hz
    pcm_8000_bytes = audioop.ratecv(pcm_16000_bytes, 2, 1, 16000, 8000, None)[0]

    # Convert PCM to μ-law
    ulaw_8000_bytes = audioop.lin2ulaw(pcm_8000_bytes, 2)

    return ulaw_8000_bytes


class WebsocketServerInputTransport(BaseInputTransport):

    def __init__(
            self,
            host: str,
            port: int,
            params: WebsocketServerParams,
            callbacks: WebsocketServerCallbacks,
            **kwargs):
        super().__init__(params, **kwargs)

        self._host = host
        self._port = port
        self._params = params
        self._callbacks = callbacks

        self._websocket: websockets.WebSocketServerProtocol | None = None

        self._stop_server_event = asyncio.Event()

    async def start(self, frame: StartFrame):
        self._server_task = self.get_event_loop().create_task(self._server_task_handler())
        await super().start(frame)

    async def stop(self):
        self._stop_server_event.set()
        await self._server_task
        await super().stop()

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
            global sid
            sid = data['streamSid'] if data.get("streamSid") else None

            if data['event'] == 'media':
                payload_base64 = data['media']['payload']
                audio_data = base64.b64decode(payload_base64)

                pcm_bytes = process_audio_chunk(audio_data)
                frame = AudioRawFrame(audio=pcm_bytes, num_channels=1, sample_rate=16000)

                if not frame:
                    continue

                if isinstance(frame, AudioRawFrame):
                    self.push_audio_frame(frame)
                else:
                    await self._internal_push_frame(frame)

        # Notify disconnection
        await self._callbacks.on_client_disconnected(websocket)

        await self._websocket.close()
        self._websocket = None

        logger.info(f"Client {websocket.remote_address} disconnected")


class WebsocketServerOutputTransport(BaseOutputTransport):

    def __init__(self, params: WebsocketServerParams, **kwargs):
        super().__init__(params, **kwargs)

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

            payload = pcm_16000_to_ulaw_8000(frame.audio)
            payload = base64.b64encode(payload).decode('utf-8')
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
            input_name: str | None = None,
            output_name: str | None = None,
            loop: asyncio.AbstractEventLoop | None = None):
        super().__init__(input_name=input_name, output_name=output_name, loop=loop)
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
                self._host, self._port, self._params, self._callbacks, name=self._input_name)
        return self._input

    def output(self) -> FrameProcessor:
        if not self._output:
            self._output = WebsocketServerOutputTransport(self._params, name=self._output_name)
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
