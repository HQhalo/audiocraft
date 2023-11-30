import os
import json
import random
import librosa
from tqdm import tqdm

import json
import os
from sklearn.model_selection import train_test_split
import numpy as np
from text_nor import text_nor
import torchaudio

# python -m audiocraft.data.audio_dataset /home/ubuntu/data/train/audio /home/ubuntu/data/train.jsonl

f = open("/home/ubuntu/data/train.jsonl")
f1 = open("/home/ubuntu/data/zing_cap.jsonl")
f2= open("/home/ubuntu/data/music_caps.jsonl")


lines = f.readlines()
metaData = []
for line in lines:
    metaData += [json.loads(line)]

lines = f1.readlines()
for line in lines:
    metaData += [json.loads(line)]
lines = f2.readlines()
for line in lines:
    metaData += [json.loads(line)]

train, val = train_test_split(metaData, test_size=0.05, random_state=42)

os.makedirs("/home/ubuntu/code/audiocraft/egs/compress/train", exist_ok=True)
os.makedirs("/home/ubuntu/code/audiocraft/egs/compress/val", exist_ok=True)
with open("/home/ubuntu/code/audiocraft/egs/compress/train/data.jsonl", "w") as outfile:
   for l in train:
    outfile.write(json.dumps(l))
    outfile.write('\n')
with open("/home/ubuntu/code/audiocraft/egs/compress/val/data.jsonl", "w") as outfile:
   for l in val:
    outfile.write(json.dumps(l))
    outfile.write('\n')

