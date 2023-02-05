# https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt
FROM nvcr.io/nvidia/tensorrt:22.12-py3

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && apt-get install ffmpeg libsm6 libxext6 -y && apt-get autoclean -y && apt-get autoremove -y && apt-get clean -y

# get precompiled pytorch
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118 && \
# installing precompiled pytorch tensorrt
    pip install https://github.com/pytorch/TensorRT/releases/download/v1.3.0/torch_tensorrt-1.3.0-cp38-cp38-linux_x86_64.whl

# other dependencies
RUN pip install git+https://github.com/PyTorchLightning/pytorch-lightning.git \
    git+https://github.com/vballoli/nfnets-pytorch \
    basicsr albumentations IPython scipy pandas opencv-python \
    pillow wget tfrecord x-transformers adamp efficientnet_pytorch \
    tensorboardX vit-pytorch swin-transformer-pytorch madgrad timm pillow-avif-plugin kornia omegaconf torchmetrics gdown \
    https://github.com/styler00dollar/bagua/releases/download/0.1.dev765-cp38/bagua-0.1.dev765-cp38-cp38-linux_x86_64.whl \
    git+https://github.com/styler00dollar/piq.git && pip3 cache purge

# optional stuff, skip these for faster install if you dont need them
RUN git clone https://github.com/open-mmlab/mmcv.git && cd mmcv && MMCV_WITH_OPS=1 pip install -e . && \
    pip install ninja cupy bitsandbytes && \
    apt-get install -y libturbojpeg && apt-get autoclean -y && apt-get autoremove -y && apt-get clean -y && pip3 cache purge

# correlation package (needs cuda 11.8 because pytorch is 11.8)
RUN git clone https://github.com/JunHeum/ABME && cd ABME/correlation_package && python setup.py build install && \
    cd /workspace && rm -rf ABME

RUN pip install https://github.com/styler00dollar/opencv-python/releases/download/4.7.0.79e1384/opencv_contrib_python-4.7.0.79e1384-cp38-cp38-linux_x86_64.whl

# download models
RUN wget https://download.pytorch.org/models/vgg16-397923af.pth -P /root/.cache/torch/hub/checkpoints/ && \
    wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_b0-c7cc451f.pth -P /root/.cache/torch/hub/checkpoints/ && \
    wget https://github.com/photosynthesis-team/piq/releases/download/v0.5.4/PieAPPv0.1.pth -P /root/.cache/torch/hub/checkpoints/ && \
    wget https://github.com/photosynthesis-team/piq/releases/download/v0.4.1/dists_weights.pt -P /root/.cache/torch/hub/checkpoints/ && \
    wget https://github.com/styler00dollar/Colab-traiNNer/releases/download/models/encoder_epoch_20.pth -P /workspace/tensorrt/Colab-traiNNer/code/
