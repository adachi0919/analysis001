FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
  sudo \
  wget \
  vim \
  mecab \
  libmecab-dev \
  mecab-ipadic-utf8 \
  git \
  make \
  curl \
  xz-utils \
  file \
  g++ \
  unzip

WORKDIR /opt

RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh && \
  sh Anaconda3-2020.07-Linux-x86_64.sh -b -p /opt/anaconda3 && \
  rm -f Anaconda3-2020.07-Linux-x86_64.sh
ENV PATH /opt/anaconda3/bin:$PATH
RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git ; exit 0
RUN cd mecab-ipadic-neologd && \
  ./bin/install-mecab-ipadic-neologd -n -y && \
  echo "dicdir=/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd">/etc/mecabrc

RUN wget https://github.com/facebookresearch/fastText/archive/v0.9.2.zip && \
  unzip v0.9.2.zip && \
  cd fastText-0.9.2 && \
  make && \
  pip install .


RUN conda update -n base -c defaults conda

RUN pip install --upgrade pip && \
  pip install mecab-python3 \
  Janome \
  jaconv \
  tinysegmenter==0.3 \
  gensim \
  unidic-lite \
  japanize-matplotlib \
  fasttext

RUN conda install -c conda-forge \
  newspaper3k && \
  conda install beautifulsoup4 \
  lxml \
  html5lib \
  requests

WORKDIR /work

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
