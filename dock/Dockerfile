FROM ubuntu:latest
RUN apt-get update && apt-get install -y \
  sudo \
  wget \
  vim
WORKDIR /opt
RUN wget https://repo.continuum.io/archive/Anaconda3-2020.07-Linux-x86_64.sh && \
  sh Anaconda3-2020.07-Linux-x86_64.sh -b -p /opt/anaconda3 && \
  rm -f Anaconda3-2020.07-Linux-x86_64.sh
ENV PATH /opt/anaconda3/bin:$PATH
RUN conda update -n base -c defaults conda
RUN pip install --upgrade pip
RUN pip install \
  Janome \
  jaconv \
  tinysegmenter==0.3
RUN conda install -c conda-forge \
  newspaper3k
RUN conda install \
  beautifulsoup4 \
  lxml \
  html5lib \
  requests
WORKDIR /work
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
