# https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt
FROM nvcr.io/nvidia/tensorrt:21.12-py3
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && apt-get install ffmpeg libsm6 libxext6 -y && apt-get autoclean -y && apt-get autoremove -y && apt-get clean -y

# get precompiled pytorch
RUN pip install https://download.pytorch.org/whl/cu115/torch-1.11.0%2Bcu115-cp38-cp38-linux_x86_64.whl \
    https://download.pytorch.org/whl/cu115/torchvision-0.12.0%2Bcu115-cp38-cp38-linux_x86_64.whl \
    https://download.pytorch.org/whl/cu115/torchaudio-0.11.0%2Bcu115-cp38-cp38-linux_x86_64.whl && \
# installing precompiled pytorch tensorrt
    pip install https://github.com/pytorch/TensorRT/releases/download/v1.1.0/torch_tensorrt-1.1.0-cp38-cp38-linux_x86_64.whl && \

# other dependencies
    pip install git+https://github.com/styler00dollar/pytorch-lightning.git@fc86f4ca817d5ba1702a210a898ac2729c870112 \
    git+https://github.com/vballoli/nfnets-pytorch \
    basicsr albumentations IPython scipy pandas opencv-python pillow wget tfrecord x-transformers adamp efficientnet_pytorch \
        tensorboardX vit-pytorch swin-transformer-pytorch madgrad timm pillow-avif-plugin kornia omegaconf \
    git+https://github.com/styler00dollar/piq.git && pip3 cache purge

# optional stuff, skip these for faster install if you dont need them
RUN pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11.0/index.html && \
    pip install ninja cupy-cuda115 bitsandbytes-cuda111 && \
    apt-get install -y libturbojpeg && apt-get autoclean -y && apt-get autoremove -y && apt-get clean -y && pip3 cache purge

# correlation package (needs cuda 11.5 because pytorch is 11.5)
RUN git clone https://github.com/JunHeum/ABME && cd ABME/correlation_package && python setup.py build install && \
    cd /workspace && rm -rf ABME

# download models
RUN wget https://download.pytorch.org/models/vgg16-397923af.pth -P /root/.cache/torch/hub/checkpoints/ && \
    wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_b0-c7cc451f.pth -P /root/.cache/torch/hub/checkpoints/ && \
    wget https://github.com/photosynthesis-team/piq/releases/download/v0.5.4/PieAPPv0.1.pth -P /root/.cache/torch/hub/checkpoints/ && \
    wget https://github.com/photosynthesis-team/piq/releases/download/v0.4.1/dists_weights.pt -P /root/.cache/torch/hub/checkpoints/ && \
    wget https://github.com/styler00dollar/Colab-traiNNer/releases/download/models/encoder_epoch_20.pth -P /workspace/tensorrt/Colab-traiNNer/code/

# hotfix
RUN pip install torchmetrics==0.7.3 gdown && pip3 cache purge