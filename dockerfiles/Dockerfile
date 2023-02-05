# https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt
FROM nvcr.io/nvidia/tensorrt:22.12-py3

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && apt-get install ffmpeg libsm6 libxext6 libturbojpeg libjpeg-dev libpng-dev -y && apt-get autoclean -y && apt-get autoremove -y && apt-get clean -y

# get precompiled pytorch
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
RUN pip install https://github.com/pytorch/TensorRT/releases/download/v1.3.0/torch_tensorrt-1.3.0-cp38-cp38-linux_x86_64.whl \
    pytorch_lightning basicsr git+https://github.com/vballoli/nfnets-pytorch \
    albumentations IPython scipy pandas opencv-python pillow wget tfrecord x-transformers adamp efficientnet_pytorch tensorboardX \
    vit-pytorch swin-transformer-pytorch madgrad timm pillow-avif-plugin kornia omegaconf torchmetrics gdown ninja cupy bitsandbytes \
    https://github.com/styler00dollar/bagua/releases/download/0.1.dev765-cp38/bagua-0.1.dev765-cp38-cp38-linux_x86_64.whl \
    git+https://github.com/styler00dollar/piq.git && \
    apt-get autoclean -y && apt-get autoremove -y && apt-get clean -y && pip3 cache purge

# mmcv
RUN git clone https://github.com/open-mmlab/mmcv.git && cd mmcv && MMCV_WITH_OPS=1 pip install -e . && cd .. && rm -rf mmcv

# correlation package (would need cuda 11.7 because pytorch cuda, and nightly 11.8 is broken, base image cuda is 11.8, not compiling)
#RUN git clone https://github.com/JunHeum/ABME && cd ABME/correlation_package && python setup.py build install && \
#    cd /workspace && rm -rf ABME

# download models
RUN wget https://download.pytorch.org/models/vgg16-397923af.pth -P /root/.cache/torch/hub/checkpoints/ && \
    wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_b0-c7cc451f.pth -P /root/.cache/torch/hub/checkpoints/ && \
    wget https://github.com/photosynthesis-team/piq/releases/download/v0.5.4/PieAPPv0.1.pth -P /root/.cache/torch/hub/checkpoints/ && \
    wget https://github.com/photosynthesis-team/piq/releases/download/v0.4.1/dists_weights.pt -P /root/.cache/torch/hub/checkpoints/ && \
    wget https://github.com/styler00dollar/Colab-traiNNer/releases/download/models/encoder_epoch_20.pth -P /workspace/tensorrt/Colab-traiNNer/code/
