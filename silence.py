from pydub import AudioSegment, silence
import json
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import os
# f = open("/home/ubuntu/data/train/train.json")
# trainMeta = json.load(f)
# dataset_path = "/home/ubuntu/data/train/audio/"

# audio_dbfs = {}
# audiokeys = [i for i in trainMeta.keys()]

# def sil(audio):
#     myaudio = AudioSegment.from_mp3(dataset_path + audio)
#     audio_dbfs[audio] = myaudio.dBFS
# pool = ThreadPool(4)
# pool.map(
#     sil,
#     audiokeys,
# )

# pool.close()
# pool.join()

# with open("/home/ubuntu/code/audiocraft/dbfs.json", "w") as outfile:
#     outfile.write(json.dumps(audio_dbfs))

# /home/ubuntu/data/train/audio/1699168565.2699792.mp3

f = open("/home/ubuntu/code/audiocraft/dbfs.json")
audio_dbfs = json.load(f)

for k in audio_dbfs:
    if audio_dbfs[k] < -35:
        os.system(f"rm /home/ubuntu/data/train/audio/{k}")