M pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

RUN apt-get -y update
RUN apt-get install -y git wget
RUN apt-get -y install zsh
RUN chsh -s /usr/bin/zsh
RUN sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
RUN apt-get -y install fonts-powerline
RUN pip install torch sentencepiece transformers pytorch-lightning
WORKDIR /home/dummy


