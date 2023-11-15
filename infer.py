
import json
import os
import numpy as np
from tqdm import tqdm
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from text_normalizer import WhiteSpaceTokenizer

f = open("/home/ubuntu/code/audiocraft/test/public.json")
test = json.load(f)

musicgen = MusicGen.get_pretrained('facebook/musicgen-medium')
musicgen.set_generation_params(duration=10.21)

nor = WhiteSpaceTokenizer()
datasetTest = []
for filename in test:
    des = test[filename]
    datasetTest += [[filename, des]]

chunks = np.array_split(datasetTest, len(datasetTest)/20)

for chunk in tqdm(chunks):
    des = [i[1] for i in chunk]
    wavs = musicgen.generate(des)
    # save and display generated audio
    for idx, one_wav in enumerate(wavs):
        audio_write(f"/home/ubuntu/code/audiocraft/submission/{chunk[idx][0]}", one_wav.cpu(), musicgen.sample_rate, strategy="loudness", loudness_compressor=True, format="mp3", mp3_rate=64, add_suffix=False)