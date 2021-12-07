# Colab-traiNNer

This repo uses a lot of code from [victorca25/traiNNer](https://github.com/victorca25/traiNNer), but this code is fundamentally different and is build with PyTorch Lightning. There are lots of differences in regards to architectures, optimizers, loss functions, augmentations, etc. It is worth looking into both. My main goal in this repo is to add as many features as possible. Be aware that the code is experimental.

Simply download the `.ipynb` and open it inside your Google Drive or click [here](https://colab.research.google.com/github/styler00dollar/Colab-traiNNer/blob/master/Colab-traiNNer.ipynb) and copy the file with `"File > Save a copy to Drive..."` into your Google Drive.

You can also use the code locally. Install commands for local usage:
```
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 torchtext -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install git+https://github.com/styler00dollar/pytorch-lightning.git@fc86f4ca817d5ba1702a210a898ac2729c870112
pip install git+https://github.com/vballoli/nfnets-pytorch
pip install opencv-python pillow piq wget tfrecord x-transformers adamp efficientnet_pytorch tensorboardX vit-pytorch swin-transformer-pytorch madgrad timm pillow-avif-plugin kornia omegaconf
```
(You need this specific `pytorch-lightning` version, or it won't work.)

Lots of stuff (`mmcv`, `ninja`, `correlation-package`, `cupy`, `Adam8Bit`) is optional and the requirements will depend on what you train how. Look into the Colab file for more details. For basic usage, the above commands should be sufficient.

## Brief guide:
Configure paths in `config.yaml`. `/content/drive/MyDrive/` is the path to your personal Google Drive folder. Be aware that everything inside `/content/` and stuff that is not in Google Drive will be deleted once Colab closes. Be also aware, that indent (the amount of spaces) is very important in the config file. Don't try to change that, or it will result in errors. Example config:
```yaml
path:
    pretrain_model_G: "/content/drive/MyDrive/model.pth"
    pretrain_model_D: 
    checkpoint_path:
    checkpoint_save_path: '/content/drive/MyDrive/my_model/'
    validation_output_path: '/content/drive/MyDrive/my_model/val'
    log_path: '/content/drive/MyDrive/my_model/'
```
Choose the correct dataloader. They all have some specific purpose.
```yaml
# training super resolution
mode: DS_lrhr
# training a video model
mode: DS_video
# there are some more, look into config.yaml
```
Uncomment the generator you want to use. Comments are done with `#`. Don't forget to comment other generators. The first line is usually just a comment, don't uncomment that. Only one generator should be uncommented at a time when you start training. If you want to use ESRGAN, then you would have this in your config file:
```yaml
    # ESRGAN:
    netG: RRDB_net
    norm_type: null
    mode: CNA
    nf: 64
    nb: 23
    in_nc: 3 # of input image channels: 3 for RGB and 1 for grayscale
    out_nc: 3 # of output image channels: 3 for RGB and 1 for grayscale
    gc: 32
    group: 1
    convtype: Conv2D # Conv2D | PartialConv2D
    net_act: leakyrelu # swish | leakyrelu
    gaussian: false # true | false
    plus: false # true | false
    finalact: None #tanh # Test
    upsample_mode: 'upconv'
    nr: 3
```
The same applies to the discriminator, but the discriminator is optional. Uncomment one by removing `#`. Example:
```yaml
    # resnet
    netD: resnet
    resnet_arch: resnet50 # resnet50, resnet101, resnet152
    num_classes: 1
    pretrain: True
```
Losses do have the extension `_weight`. If that is set to a value that is bigger than zero, then that loss will be active. Example:
```yaml
    L1Loss_weight: 1
```
If you do not want to use specific losses, just leave the number 0. Finally, adjust `lr`, `batch_size`, `HR_size`, `n_workers`, `gpus`, `scale` and `use_amp` to desired values.

If you want to start training locally, just do `python train.py`. If you want to visualize the training loss with graphs, go into the log folder and use `tensorboard --logdir .`. It will create an URL that you can open in your browser. (Only works locally, download logs to do this if you use Colab.)
