import asyncio
import json
import websockets
import base64
from datetime import datetime
import pywav
from pydub import AudioSegment
from io import BytesIO

# Buffer to store audio data
audio_buffer = bytearray()

def pcmu_to_pcm(data):
    """
    Convert PCMU data to PCM.
    """
    audio = AudioSegment.from_raw(BytesIO(data), sample_width=1, frame_rate=8000, channels=1, codec="ulaw")
    audio.set_frame_rate(16000)
    return audio.raw_data

async def save_buffer_to_file(buffer, filename):
    # Convert PCMU to PCM before saving
    pcm_data = pcmu_to_pcm(buffer)
    # Save buffer as WAV file
    wave_write = pywav.WavWrite(filename, 1, 8000, 8, 7)
    wave_write.write(pcm_data)
    wave_write.close()

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
                    # Save the current buffer to a file
                    filename = f'audio_{datetime.now().strftime("%Y%m%d%H%M%S")}.wav'
                    await save_buffer_to_file(audio_buffer[:40000], filename)
                    # Remove the saved audio from the buffer
                    audio_buffer = audio_buffer[40000:]
    except websockets.exceptions.ConnectionClosed:
        print("Connection closed")

async def server():
    async with websockets.serve(consumer_handler, "localhost", 8765):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(server())
