import os
from pydub import AudioSegment
from tqdm import tqdm
import threading

dataset_path = "/home/ubuntu/data/train/audio"
filenames = os.listdir(dataset_path)
pivot = len(filenames) // 2
def func1():
  for filename in tqdm(filenames[:pivot]):
      if filename.endswith(('.mp3')):
          
          audio = AudioSegment.from_file(f"{dataset_path}/{filename}")

          # resample
          audio = audio.set_frame_rate(32000)
          combined = audio.append(audio, crossfade=350)
          combined = combined.append(audio, crossfade=450)
          output_file = filename.replace('mp3', 'wav')
          combined.export(f"/home/ubuntu/data/zalotrain/audio/{output_file}", format="wav")

def func2():
  for filename in tqdm(filenames[pivot:]):
      if filename.endswith(('.mp3')):
          
          audio = AudioSegment.from_file(f"{dataset_path}/{filename}")

          # resample
          audio = audio.set_frame_rate(32000)
          combined = audio.append(audio, crossfade=350)
          combined = combined.append(audio, crossfade=450)
          output_file = filename.replace('mp3', 'wav')
          combined.export(f"/home/ubuntu/data/zalotrain/audio/{output_file}", format="wav")



t1 = threading.Thread(target=func1)
t2 = threading.Thread(target=func2)

t1.start()
t2.start()

t1.join()
