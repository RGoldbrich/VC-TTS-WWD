# standard lib
import wave

# third party
import pyaudio
import sounddevice

# parameters
CHUNK = 1600
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000


def record_wave(duration_ms: int) -> tuple[bytes, int]:
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []

    for i in range(0, int(RATE / CHUNK * (duration_ms / 1000))):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    frames = b''.join(frames)

    sample_size = p.get_sample_size(FORMAT)
    p.terminate()

    return frames, sample_size


def batch_record(save_folder: str, number_samples: int, duration_ms: int) -> None:
    for n in range(number_samples):
        print("recording started {}/{}".format(n + 1, number_samples))

        frames, sample_size = record_wave(duration_ms)

        wf = wave.open(f"{save_folder}/{n:04d}.wav", 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(sample_size)
        wf.setframerate(RATE)
        wf.writeframes(frames)
        wf.close()
