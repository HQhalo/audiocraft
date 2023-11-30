
import json
import os
import numpy as np
from tqdm import tqdm
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from text_nor import text_nor
import torchaudio
import random
import torch
from frechet_audio_distance import FrechetAudioDistance
import laion_clap

def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

valfile = open("/home/ubuntu/code/audiocraft/egs/onl/val/data.jsonl")
lines = valfile.readlines()
valData = []
for line in lines:
    valData += [json.loads(line)]

f = open("/home/ubuntu/data/train/train.json")
trainMeta = json.load(f)

test = {}
for m in valData:
    filename = m['path'].split("/")[-1]
    test[filename] = trainMeta[filename]

def infer():
    musicgen = MusicGen.get_pretrained('/home/ubuntu/checkpoints/finetune_30_17_23')
    musicgen.set_generation_params(duration=10.2, extend_stride= 5, temperature=1, top_k=250)

    set_all_seeds(42)

    datasetTest = []
    for filename in test:
        des = text_nor(test[filename])
        datasetTest += [[filename, des]]

    chunks = np.array_split(datasetTest, len(datasetTest)/20)

    for chunk in tqdm(chunks):
        des = [i[1] for i in chunk]
        
        wavs = musicgen.generate(des)
        for idx, one_wav in enumerate(wavs):
            audio_write(f"/tmp/test/{chunk[idx][0]}", one_wav.cpu(), musicgen.sample_rate, strategy="peak", format="mp3", mp3_rate=64, add_suffix=False)

infer()

vggish = FrechetAudioDistance(
    model_name="vggish",
    sample_rate=16000,
    use_pca=False, 
    use_activation=False,
    verbose=False
)
background_embds_path = "/tmp/background_embeddings.npy"


fad_score = vggish.score("/tmp/background", "/tmp/test", background_embds_path=background_embds_path)
fas_score =1 / (1 + fad_score)


model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt('/home/ubuntu/checkpoints/clap/630k-audioset-best.pt')
# model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
# model.load_ckpt('/home/ubuntu/checkpoints/clap/music_audioset_epoch_15_esc_90.14.pt')



audio_file = [ [f"/tmp/test/{k}", test[k]] for k in test]

chunks = np.array_split(audio_file, len(audio_file)/20)
sims = None
for chunk in chunks:
    audio_embed = model.get_audio_embedding_from_filelist(x = [i[0]for i in chunk])
    text_embed = model.get_text_embedding([i[1]for i in chunk])
    s = torch.nn.functional.cosine_similarity(torch.tensor(audio_embed), torch.tensor(text_embed))
    if sims is None:
        sims = s 
    else:
        sims = torch.cat((sims, s), 0)

clap_score = torch.mean(sims).item()
avg_score = (fas_score + clap_score)/2
print("fas_score {:.5f} clap_score {:.5f}  avg_score {:.5f}".format(fas_score, clap_score, avg_score))