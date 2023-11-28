
import json
import os
import numpy as np
from tqdm import tqdm
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from text_nor import text_nor
import torch
import random

def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_all_seeds(42)

f = open("/home/ubuntu/data/test/public.json")
test = json.load(f)

musicgen = MusicGen.get_pretrained('/home/ubuntu/checkpoints/finetune_3')
musicgen.set_generation_params(duration=10.2, extend_stride= 5)

datasetTest = []
for filename in test:
    des = text_nor(test[filename])
    datasetTest += [[filename, des]]

chunks = np.array_split(datasetTest, len(datasetTest)/20)

for chunk in tqdm(chunks):
    des = [i[1] for i in chunk]
    wavs = musicgen.generate(des)
    for idx, one_wav in enumerate(wavs):
        audio_write(f"/home/ubuntu/code/audiocraft/submission/{chunk[idx][0]}", one_wav.cpu(), musicgen.sample_rate, strategy="peak", loudness_compressor=True, format="mp3", mp3_rate=64, add_suffix=False)
