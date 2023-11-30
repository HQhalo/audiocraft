import subprocess
import os
from pathlib import Path

from datasets import load_dataset, Audio


def download_clip(
    video_identifier,
    output_filename,
    start_time,
    end_time,
    tmp_dir='/tmp/musiccaps',
    num_attempts=5,
    url_base='https://www.youtube.com/watch?v='
):
    status = False

    command = f"""
        yt-dlp --quiet --no-warnings -x --audio-format mp3 -f bestaudio -o "{output_filename}" --download-sections "*{start_time}-{end_time}" {url_base}{video_identifier}
    """.strip()

    attempts = 0
    while True:
        try:
            output = subprocess.check_output(command, shell=True,
                                                stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            attempts += 1
            if attempts == num_attempts:
                return status, err.output
        else:
            break

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    return status, 'Downloaded'


def main(
    data_dir: str,
    sampling_rate: int = 32000,
    limit: int = None,
    num_proc: int = 1,
    writer_batch_size: int = 1000,
):
    """
    Download the clips within the MusicCaps dataset from YouTube.
    Args:
        data_dir: Directory to save the clips to.
        sampling_rate: Sampling rate of the audio clips.
        limit: Limit the number of examples to download.
        num_proc: Number of processes to use for downloading.
        writer_batch_size: Batch size for writing the dataset. This is per process.
    """

    ds = load_dataset('google/MusicCaps', split='train')
    if limit is not None:
        print(f"Limiting to {limit} examples")
        ds = ds.select(range(limit))

    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)

    def process(example):
        outfile_path = str(data_dir / f"{example['ytid']}.mp3")
        status = True
        if not os.path.exists(outfile_path):
            # print(outfile_path)
            status = False
            status, log = download_clip(
                example['ytid'],
                outfile_path,
                example['start_s'],
                example['end_s'],
            )

        example['audio'] = outfile_path
        example['download_status'] = status
        return example

    return ds.map(
        process,
        num_proc=num_proc,
        writer_batch_size=writer_batch_size,
        keep_in_memory=False
    )
    

    

# main('/home/ubuntu/data/music_caps', num_proc=16, limit=None)

import json

files = os.listdir("/home/ubuntu/data/music_caps")

# ds = load_dataset('google/MusicCaps', split='train')
# files = set(files)
# i = 0

# metadata = {}
# for example in ds:
#     filename = f"{example['ytid']}.mp3"
#     if filename in files:
#         metadata[filename] = example['caption']
# with open("/home/ubuntu/data/music_com.json", "w") as outfile:
#     json.dump(metadata, outfile)

from tqdm import tqdm
for filename in tqdm(files):
    os.system(f"ffmpeg -i /home/ubuntu/data/music_caps/{filename} -ac 1 -ar 32000 /home/ubuntu/data/music_com/{filename} > /dev/null 2>&1")