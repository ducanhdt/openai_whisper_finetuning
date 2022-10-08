import pyaudio
import wave

import sys

try:
    # Your code here

    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2
    fs = 16000  # Record at 44100 samples per second
    filename = "audio.wav"

    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    s = input(("Enter to start recording"))
    print("Recording Ctrl+C to stop and save audio file")

    stream = p.open(
        format=sample_format,
        channels=channels,
        rate=fs,
        frames_per_buffer=chunk,
        input=True,
    )

    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    while 1:
        data = stream.read(chunk)
        frames.append(data)

except KeyboardInterrupt:
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print("Finished recording")

    # Save the recorded data as a WAV file
    wf = wave.open(filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b"".join(frames))
    wf.close()
    sys.exit()
