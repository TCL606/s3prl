FROM aegis1/cuda10.2-cudnn7-devel-ubuntu16.04
USER root
RUN cp /etc/apt/sources.list.bak  /etc/apt/sources.list
RUN apt update
RUN apt-get -y install wget curl man git less openssl libssl-dev unzip
RUN apt install -y openssh-server

# install miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh &&\
# RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh &&\
	bash Miniconda3-latest-Linux-x86_64.sh -b &&\
        rm -rf Miniconda3-latest-Linux-x86_64.sh
        # conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free &&\
        # conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main &&\
        # conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge &&\
        # conda config --set show_channel_urls yes

ENV CONDA_EXE="/root/miniconda3/bin/conda"
ENV CONDA_TOOLS_DIR="/root/miniconda3"
RUN conda init && conda create -n superb -y python=3.8
SHELL ["conda", "run", "-n", "/bin/bash", "-c"]
RUN conda activate superb

RUN pip install sox

# install tmux
RUN apt -y install tmux

RUN pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN cd ~ && git clone https://github.com/huggingface/huggingface_hub.git && git clone https://github.com/TCL606/s3prl.git && git clone https://github.com/pytorch/fairseq.git
RUN cd huggingface_hub && pip install -e ./ && cd ..
RUN cd s3prl && git checkout TCL && pip install -e ./ && cd ..
RUN cd fairseq && git reset --hard f2146bdc7abf293186de9449bfa2272775e39e1d && pip install -e ./ && cd ..

# COPY s3prl/ ./s3prl
# COPY fairseq/ ./fairseq/
# COPY huggingface_hub/ ./huggingface_hub/
# RUN cd huggingface_hub && pip install -e ./ && cd ..
# RUN cd s3prl && pip install -e ./ && cd ..
# RUN cd fairseq && pip install -e ./

ENV SHELL=/bin/bash
CMD ["/bin/bash"]