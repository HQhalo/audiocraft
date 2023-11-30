from audiocraft.solvers import CompressionSolver
import torchaudio
from audiocraft.data.audio import audio_write
import julius

model = CompressionSolver.model_from_checkpoint('//sig/c9ca89f6')

melody, sr = torchaudio.load('/home/ubuntu/data/train/audio/1699168593.4240422.mp3')
melody = julius.resample_frac(melody, int(sr), int(32000))
codes, scale = model.encode(melody[None].expand(1, -1, -1))
wavs= model.decode(codes, scale)

audio_write(f"/home/ubuntu/code/audiocraft/test_sig.mp3", wavs[0].cpu(), 32000, strategy="peak", loudness_compressor=True, format="mp3", mp3_rate=64, add_suffix=False)
