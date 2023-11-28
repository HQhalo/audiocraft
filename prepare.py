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

# python -m audiocraft.data.audio_dataset /home/ubuntu/data/train/audio /home/ubuntu/data/train/data.jsonl
# python -m audiocraft.data.audio_dataset /home/ubuntu/data/music_caps /home/ubuntu/data/music_caps.jsonl
# python -m audiocraft.data.audio_dataset /home/ubuntu/data/chunk_zing_cap/audio /home/ubuntu/data/chunk_zing_cap/data.jsonl

# chroma2music
#  text2music
# dora run solver=musicgen/musicgen_base_32khz model/lm/model_scale=small conditioner=text2music continue_from=//pretrained/facebook/musicgen-small
#continue_from=//pretrained/facebook/musicgen-small
# dora run solver=compression/encodec_musicgen_32khz continue_from=//pretrained/facebook/encodec_32khz


#export AUDIOCRAFT_TEAM=default
#export AUDIOCRAFT_CLUSTER=default


# rm -rf /home/ubuntu/checkpoints/audiocraft_ubuntu/xps/c9ca89f6/
# rm -rf /home/ubuntu/checkpoints/audiocraft_ubuntu/xps/33fd1f20


def zalo_cap():
    f = open("/home/ubuntu/data/train/train.json")
    trainMeta = json.load(f)

    dataset_path = "/home/ubuntu/data/train/audio/"
    for audio in tqdm(trainMeta):
        fileName = dataset_path +audio.replace("mp3", "json")
        audiofile =  audio
        metadata = torchaudio.info(os.path.join(dataset_path, audiofile))

        entry = {
                "key": "",
                "artist": audiofile.replace(".mp3", ""),
                "sample_rate": metadata.sample_rate,
                "file_extension": "mp3",
                "description": text_nor(trainMeta[audio]),
                "keywords": "",
                "duration": metadata.num_frames/metadata.sample_rate,
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

def music_cap():
    f = open("/home/ubuntu/data/music_com.json")
    trainMeta = json.load(f)

    dataset_path = "/home/ubuntu/data/music_caps/"
    for audio in tqdm(trainMeta):
        fileName = dataset_path +audio.replace("mp3", "json")
        audiofile =  audio
        metadata = torchaudio.info(os.path.join(dataset_path, audiofile))

        entry = {
                "key": "",
                "artist": audiofile.replace(".mp3", ""),
                "sample_rate": metadata.sample_rate,
                "file_extension": "mp3",
                "description": text_nor(trainMeta[audio]),
                "keywords": "",
                "duration": metadata.num_frames/metadata.sample_rate,
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

def zing_cap():
    f = open("/home/ubuntu/data/chunk_zing_cap/train.json")
    trainMeta = json.load(f)

    dataset_path = "/home/ubuntu/data/chunk_zing_cap/audio/"
    audiofiles = set(os.listdir(dataset_path))
    for audio in tqdm(trainMeta):
        if audio not in audiofiles:
            continue
        fileName = dataset_path +audio.replace("mp3", "json")
        audiofile =  audio
        metadata = torchaudio.info(os.path.join(dataset_path, audiofile))

        entry = {
                "key": "",
                "artist": audiofile.split("_")[0],
                "sample_rate": metadata.sample_rate,
                "file_extension": "mp3",
                "description": text_nor(trainMeta[audio]),
                "keywords": "",
                "duration": metadata.num_frames/metadata.sample_rate,
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


def create_dataset():
    fzalo = open("/home/ubuntu/data/train/data.jsonl")
    lines = fzalo.readlines()
    metaData = []
    for line in lines:
        metaData += [json.loads(line)]


   
    # fmusic = open("/home/ubuntu/data/music_caps.jsonl")
    # lines = fmusic.readlines()
    # for line in lines:
    #     metaData += [json.loads(line)]

    # fzing= open("/home/ubuntu/data/chunk_zing_cap/data.jsonl")
    # lines = fzing.readlines()
    # for line in lines:
    #     metaData += [json.loads(line)]

    train, val = train_test_split(metaData, test_size=0.05, random_state=42)
    os.makedirs("/home/ubuntu/code/audiocraft/egs/onl/val", exist_ok=True)
    os.makedirs("/home/ubuntu/code/audiocraft/egs/onl/train", exist_ok=True)

    with open("/home/ubuntu/code/audiocraft/egs/onl/train/data.jsonl", "w") as outfile:
        for l in train:
            outfile.write(json.dumps(l))
            outfile.write('\n')


    with open("/home/ubuntu/code/audiocraft/egs/onl/val/data.jsonl", "w") as outfile:
        for l in val:
            outfile.write(json.dumps(l))
            outfile.write('\n')
            
# import glob 
# import random
# a = {}
# for name in glob.glob('/home/ubuntu/data/chunk_zing_cap/audio/*.mp3'): 
#     id = name.split("/")[-1].split("_")[0]
#     if id in a:
#         a[id] += [name]
#     else:
#         a[id] = [name]

# for id in a:
#     os.system(f"rm {random.choice(a[id])}")

# zalo_cap()
# music_cap()
# zing_cap()

create_dataset()