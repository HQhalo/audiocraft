import os
import json
import random
import librosa
from text_normalizer import WhiteSpaceTokenizer
from tqdm import tqdm

import json
import os
from sklearn.model_selection import train_test_split
import numpy as np
# python -m audiocraft.data.audio_dataset /home/ubuntu/data/zalotrain/audio /home/ubuntu/data/zalotrain/data.jsonl
# dora run solver=musicgen/musicgen_base_32khz model/lm/model_scale=small continue_from=//pretrained/facebook/musicgen-small conditioner=text2music

f = open("/home/ubuntu/data/train/train.json")
trainMeta = json.load(f)

dataset_path = "/home/ubuntu/data/zalotrain/audio/"
nor = WhiteSpaceTokenizer()

for audio in tqdm(trainMeta):
    fileName = dataset_path +audio.replace("mp3", "json")
    audiofile =  audio.replace("mp3", "wav")
    # y, sr = librosa.load(os.path.join(dataset_path,audiofile))

    # length = librosa.get_duration(y=y, sr=sr)

    entry = {
            "key": "",
            "artist": "",
            "sample_rate": 32000,
            "file_extension": "wav",
            "description": nor(trainMeta[audio]),
            "keywords": "",
            "duration": 29.449,
            "bpm": "",
            "genre": "",
            "title": "",
            "name": "",
            "instrument": "",
            "moods": "",
            "path": os.path.join(dataset_path, audiofile),
        }

    with open(fileName, "w") as outfile:
        outfile.write(json.dumps(entry))

f = open("/home/ubuntu/data/zalotrain/data.jsonl")

lines = f.readlines()
metaData = []
for line in lines:
    metaData += [json.loads(line)]


train, val = train_test_split(metaData, test_size=0.1, random_state=42)

os.makedirs("/home/ubuntu/code/audiocraft/egs/zalo/train", exist_ok=True)
os.makedirs("/home/ubuntu/code/audiocraft/egs/zalo/val", exist_ok=True)
with open("/home/ubuntu/code/audiocraft/egs/zalo/train/data.jsonl", "w") as outfile:
   for l in train:
    outfile.write(json.dumps(l))
    outfile.write('\n')
with open("/home/ubuntu/code/audiocraft/egs/zalo/val/data.jsonl", "w") as outfile:
   for l in val:
    outfile.write(json.dumps(l))
    outfile.write('\n')


# import os
# from pydub import AudioSegment
# from tqdm import tqdm
# dataset_path = "/musicvolume/data/train/audio"
# new_path = "/musicvolume/data/audio32k"
# for filename in tqdm(os.listdir(dataset_path)):
#     if filename.endswith(('.mp3')):
#         new_filename = filename.replace('mp3', 'wav')
#         audio = AudioSegment.from_file(f"{dataset_path}/{filename}")

#         # resample
#         audio = audio.set_frame_rate(32000)
#         audio.export(f"{new_path}/{new_filename}", format="wav")