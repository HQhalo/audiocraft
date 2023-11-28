FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04
WORKDIR /code
COPY ./ ./
RUN sudo apt-get update && sudo apt-get -y install ffmpeg
RUN python -m pip install -e .

RUN chmod +x /code/predict.sh