# https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt
FROM nvcr.io/nvidia/tensorrt:21.11-py3
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update

# get precompiled pytorch
#RUN pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 torchtext -f https://download.pytorch.org/whl/cu113/torch_stable.html

# compile pytorch yourself (for cuda 11.5), needs time to compile
# https://github.com/PINTO0309/PyTorch-build/blob/main/Dockerfile
RUN apt-get update && apt-get install -y \
        automake autoconf libpng-dev nano wget npm \
        curl zip unzip libtool swig zlib1g-dev pkg-config \
        git wget xz-utils python3-mock libpython3-dev \
        libpython3-all-dev python3-pip g++ gcc make \
        pciutils cpio gosu git liblapack-dev liblapacke-dev
RUN pip3 install --upgrade pip \
    && pip3 install --upgrade onnx \
    && pip3 install --upgrade onnxruntime \
    && pip3 install --upgrade gdown \
    && pip3 install cmake==3.18.4 \
    && pip3 install --upgrade pyyaml \
    && pip3 install --upgrade ninja \
    && pip3 install --upgrade yapf \
    && pip3 install --upgrade six \
    && pip3 install --upgrade wheel \
    && pip3 install --upgrade moc \
    && ldconfig
# compiling pytorch
RUN git clone --recursive https://github.com/pytorch/pytorch && cd pytorch \
    && sed -i -e "/^#ifndef THRUST_IGNORE_CUB_VERSION_CHECK$/i #define THRUST_IGNORE_CUB_VERSION_CHECK" \
                 /usr/local/cuda/targets/x86_64-linux/include/thrust/system/cuda/config.h \
    && cat /usr/local/cuda/targets/x86_64-linux/include/thrust/system/cuda/config.h \
    && sed -i -e "/^if(DEFINED GLIBCXX_USE_CXX11_ABI)/i set(GLIBCXX_USE_CXX11_ABI 1)" \
                 CMakeLists.txt \
    && pip3 install -r requirements.txt \
    && USE_NCCL=OFF python3 setup.py build \
    && python3 setup.py bdist_wheel

RUN git clone https://github.com/pytorch/vision.git && cd vision \
    && git submodule update --init --recursive \
    && pip3 install /workspace/pytorch/dist/*.whl \
    && python3 setup.py build \
    && python3 setup.py bdist_wheel
RUN git clone https://github.com/pytorch/audio.git && cd audio \
    && git submodule update --init --recursive \
    && apt-get install -y sox libsox-dev \
    && python3 setup.py build \
    && python3 setup.py bdist_wheel
RUN pip install /workspace/vision/dist/*.whl
RUN pip install /workspace/audio/dist/*.whl

# other dependencies
RUN pip install git+https://github.com/styler00dollar/pytorch-lightning.git@fc86f4ca817d5ba1702a210a898ac2729c870112
RUN pip install git+https://github.com/vballoli/nfnets-pytorch
RUN pip install opencv-python pillow piq wget tfrecord x-transformers adamp efficientnet_pytorch tensorboardX vit-pytorch swin-transformer-pytorch madgrad timm pillow-avif-plugin kornia omegaconf

# optional stuff, skip these for faster install if you dont need them
RUN cd / && git clone https://github.com/facebookresearch/bitsandbytes && cd bitsandbytes && CUDA_VERSION=115 python setup.py install
RUN pip install mmcv-full ninja cupy
RUN apt-get install -y libturbojpeg && pip install PyTurboJPEG
RUN cd / && git clone https://github.com/JunHeum/ABME && cd ABME/correlation_package && python setup.py build install
