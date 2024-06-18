
import aiohttp
import asyncio
import os
import sys

from pipecat.frames.frames import LLMMessagesFrame, Frame, AudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from pipecat.services.openai import OpenAILLMService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from websocket_server import WebsocketServerParams, WebsocketServerTransport
from pipecat.vad.silero import SileroVADAnalyzer

from loguru import logger

from dotenv import load_dotenv
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class MessagesProcessor(FrameProcessor):

    def __init__(self):
        super().__init__()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, LLMMessagesFrame):
            print(frame.messages, flush=True)

        if isinstance(frame, AudioRawFrame):
            return

        await self.push_frame(frame, direction)


async def main():
    async with aiohttp.ClientSession() as session:
        transport = WebsocketServerTransport(
            params=WebsocketServerParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                transcription_enabled=False,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True
            )
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o")

        stt = DeepgramSTTService(api_key=os.getenv('DEEPGRAM_API_KEY'))

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
            },
        ]

        tma_in = LLMUserResponseAggregator(messages)
        tma_out = LLMAssistantResponseAggregator(messages)
        mp = MessagesProcessor()

        pipeline = Pipeline([
            transport.input(),   # Websocket input from client
            stt,                 # Speech-To-Text
            tma_in,              # User responses
            mp,
            llm,                 # LLM
            tts,                 # Text-To-Speech
            transport.output(),  # Websocket output to client
            tma_out              # LLM responses
        ])

        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            # Kick off the conversation.
            messages.append(
                {"role": "system", "content": "Please introduce yourself to the user."})
            await task.queue_frames([LLMMessagesFrame(messages)])

        runner = PipelineRunner()

        await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main())