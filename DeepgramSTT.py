import asyncio
import time

from loguru import logger
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    SystemFrame,
    TranscriptionFrame)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import AIService

from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
)


class DeepgramSTTService(AIService):
    def __init__(self,
                 api_key: str,
                 live_options: LiveOptions = LiveOptions(
                     encoding="linear16",
                     language="en-US",
                     model="nova-2-conversationalai",
                     sample_rate=16000,
                     channels=1,
                     interim_results=True,
                     smart_format=True,
                 ),
                 **kwargs):
        super().__init__(**kwargs)

        self._live_options = live_options

        self._client = DeepgramClient(api_key)
        self._connection = self._client.listen.asynclive.v("1")
        self._connection.on(LiveTranscriptionEvents.Transcript, self._on_message)

        self._create_push_task()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        elif isinstance(frame, AudioRawFrame):
            await self._connection.send(frame.audio)
        else:
            await self._push_queue.put((frame, direction))

    async def start(self, frame: StartFrame):
        if await self._connection.start(self._live_options):
            logger.debug(f"{self}: Connected to Deepgram")
        else:
            logger.error(f"{self}: Unable to connect to Deepgram")

    async def stop(self, frame: EndFrame):
        await self._connection.finish()
        await self._push_queue.put((frame, FrameDirection.DOWNSTREAM))
        await self._push_frame_task

    async def cancel(self, frame: CancelFrame):
        await self._connection.finish()
        self._push_frame_task.cancel()

    def _create_push_task(self):
        self._push_frame_task = self.get_event_loop().create_task(self._push_frame_task_handler())
        self._push_queue = asyncio.Queue()

    async def _push_frame_task_handler(self):
        running = True
        while running:
            try:
                (frame, direction) = await self._push_queue.get()
                await self.push_frame(frame, direction)
                running = not isinstance(frame, EndFrame)
            except asyncio.CancelledError:
                break

    async def _on_message(self, *args, **kwargs):
        result = kwargs["result"]
        is_final = result.is_final
        transcript = result.channel.alternatives[0].transcript
        if len(transcript) > 0:
            if is_final:
                print('===============================')
                print(transcript)
                await self._push_queue.put((TranscriptionFrame(transcript, "", int(time.time_ns() / 1000000)), FrameDirection.DOWNSTREAM))
            else:
                await self._push_queue.put((InterimTranscriptionFrame(transcript, "", int(time.time_ns() / 1000000)), FrameDirection.DOWNSTREAM))