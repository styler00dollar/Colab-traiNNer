def CreateDiscriminator(cfg):
    if cfg["network_D"]["netD"] == "context_encoder":
        from arch.discriminators import context_encoder

        netD = context_encoder()

    # VGG
    elif cfg["network_D"]["netD"] == "VGG":
        from arch.discriminators import Discriminator_VGG

        netD = Discriminator_VGG(
            size=cfg["network_D"]["size"],
            in_nc=cfg["network_D"]["in_nc"],
            base_nf=cfg["network_D"]["base_nf"],
            norm_type=cfg["network_D"]["norm_type"],
            act_type=cfg["network_D"]["act_type"],
            mode=cfg["network_D"]["mode"],
            convtype=cfg["network_D"]["convtype"],
            arch=cfg["network_D"]["arch"],
        )

    elif cfg["network_D"]["netD"] == "VGG_fea":
        from arch.discriminators import Discriminator_VGG_fea

        netD = Discriminator_VGG_fea(
            size=cfg["network_D"]["size"],
            in_nc=cfg["network_D"]["in_nc"],
            base_nf=cfg["network_D"]["base_nf"],
            norm_type=cfg["network_D"]["norm_type"],
            act_type=cfg["network_D"]["act_type"],
            mode=cfg["network_D"]["mode"],
            convtype=cfg["network_D"]["convtype"],
            arch=cfg["network_D"]["arch"],
            spectral_norm=cfg["network_D"]["spectral_norm"],
            self_attention=cfg["network_D"]["self_attention"],
            max_pool=cfg["network_D"]["max_pool"],
            poolsize=cfg["network_D"]["poolsize"],
        )

    elif cfg["network_D"]["netD"] == "Discriminator_VGG_128_SN":
        from arch.discriminators import Discriminator_VGG_128_SN

        netD = Discriminator_VGG_128_SN()

    elif cfg["network_D"]["netD"] == "VGGFeatureExtractor":
        from arch.discriminators import VGGFeatureExtractor

        netD = VGGFeatureExtractor(
            feature_layer=cfg["feature_layer"]["feature_layer"],
            use_bn=cfg["network_D"]["use_bn"],
            use_input_norm=cfg["network_D"]["use_input_norm"],
            device=torch.device(cfg["network_D"]["device"]),
            z_norm=cfg["network_D"]["z_norm"],
        )

    # PatchGAN
    elif cfg["network_D"]["netD"] == "NLayerDiscriminator":
        from arch.discriminators import NLayerDiscriminator

        netD = NLayerDiscriminator(
            input_nc=cfg["network_D"]["input_nc"],
            ndf=cfg["network_D"]["ndf"],
            n_layers=cfg["network_D"]["n_layers"],
            norm_layer=cfg["network_D"]["norm_layer"],
            use_sigmoid=cfg["network_D"]["use_sigmoid"],
            get_feats=cfg["network_D"]["getIntermFeat"],
            patch=cfg["network_D"]["patch"],
            use_spectral_norm=cfg["network_D"]["use_spectral_norm"],
        )

    # Multiscale
    elif cfg["network_D"]["netD"] == "MultiscaleDiscriminator":
        from arch.discriminators import MultiscaleDiscriminator

        netD = MultiscaleDiscriminator(
            input_nc=cfg["network_D"]["input_nc"],
            ndf=cfg["network_D"]["ndf"],
            n_layers=cfg["network_D"]["n_layers"],
            use_sigmoid=cfg["network_D"]["use_sigmoid"],
            num_D=cfg["network_D"]["num_D"],
            get_feats=cfg["network_D"]["get_feats"],
        )

    elif cfg["network_D"]["netD"] == "ResNet101FeatureExtractor":
        from arch.discriminators import ResNet101FeatureExtractor

        netD = ResNet101FeatureExtractor(
            use_input_norm=cfg["network_D"]["use_input_norm"],
            device=torch.device(cfg["network_D"]["device"]),
            z_norm=cfg["network_D"]["z_norm"],
        )

    # MINC
    elif cfg["network_D"]["netD"] == "MINCNet":
        from arch.discriminators import MINCNet

        netD = MINCNet()

    # Pixel
    elif cfg["network_D"]["netD"] == "PixelDiscriminator":
        from arch.discriminators import PixelDiscriminator

        netD = PixelDiscriminator(
            input_nc=cfg["network_D"]["input_nc"],
            ndf=cfg["network_D"]["ndf"],
            norm_layer=cfg["network_D"]["norm_layer"],
        )

    # EfficientNet
    elif cfg["network_D"]["netD"] == "EfficientNet":
        from efficientnet_pytorch import EfficientNet

        netD = EfficientNet.from_pretrained(
            cfg["network_D"]["EfficientNet_pretrain"],
            num_classes=cfg["network_D"]["num_classes"],
        )

    # mobilenetV3
    elif cfg["network_D"]["netD"] == "mobilenetV3":
        from arch.mobilenetv3_arch import MobileNetV3

        netD = MobileNetV3(
            n_class=cfg["network_D"]["n_class"],
            mode=cfg["network_D"]["mode"],
            input_size=cfg["network_D"]["input_size"],
        )

    # resnet
    elif cfg["network_D"]["netD"] == "resnet":
        if cfg["network_D"]["pretrain"] is False:
            if cfg["network_D"]["resnet_arch"] == "resnet50":
                from arch.resnet_arch import resnet50

                netD = resnet50(
                    num_classes=cfg["network_D"]["num_classes"],
                    pretrain=cfg["network_D"]["pretrain"],
                )
            elif cfg["network_D"]["resnet_arch"] == "resnet101":
                from arch.resnet_arch import resnet101

                netD = resnet101(
                    num_classes=cfg["network_D"]["num_classes"],
                    pretrain=cfg["network_D"]["pretrain"],
                )
            elif cfg["network_D"]["resnet_arch"] == "resnet152":
                from arch.resnet_arch import resnet152

                netD = resnet152(
                    num_classes=cfg["network_D"]["num_classes"],
                    pretrain=cfg["network_D"]["pretrain"],
                )
            from init import weights_init

            weights_init(netD, "kaiming")
            print("Discriminator weight init complete.")

        if cfg["network_D"]["pretrain"] is True:
            # loading a pretrained network does not work by default, the amount of classes
            # needs to be adjusted in the final layer
            import torchvision.models as models

            if cfg["network_D"]["resnet_arch"] == "resnet50":
                pretrained_model = models.resnet50(pretrained=True)
            elif cfg["network_D"]["resnet_arch"] == "resnet101":
                pretrained_model = models.resnet101(pretrained=True)
            elif cfg["network_D"]["resnet_arch"] == "resnet152":
                pretrained_model = models.resnet152(pretrained=True)

            IN_FEATURES = pretrained_model.fc.in_features
            OUTPUT_DIM = cfg["network_D"]["num_classes"]

            fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
            pretrained_model.fc = fc

            from arch.resnet_arch import ResNet, Bottleneck
            from collections import namedtuple

            ResNetConfig = namedtuple("ResNetConfig", ["block", "n_blocks", "channels"])

            if cfg["network_D"]["resnet_arch"] == "resnet50":
                from arch.resnet_arch import resnet50

                resnet50_config = ResNetConfig(
                    block=Bottleneck,
                    n_blocks=[3, 4, 6, 3],
                    channels=[64, 128, 256, 512],
                )
                netD = ResNet(resnet50_config, OUTPUT_DIM)
            elif cfg["network_D"]["resnet_arch"] == "resnet101":
                from arch.resnet_arch import resnet101

                resnet101_config = ResNetConfig(
                    block=Bottleneck,
                    n_blocks=[3, 4, 23, 3],
                    channels=[64, 128, 256, 512],
                )
                netD = ResNet(resnet101_config, OUTPUT_DIM)
            elif cfg["network_D"]["resnet_arch"] == "resnet152":
                from arch.resnet_arch import resnet152

                resnet152_config = ResNetConfig(
                    block=Bottleneck,
                    n_blocks=[3, 8, 36, 3],
                    channels=[64, 128, 256, 512],
                )
                netD = ResNet(resnet152_config, OUTPUT_DIM)

            netD.load_state_dict(pretrained_model.state_dict())
            print("Resnet pretrain loaded.")

    # ResNeSt
    # ["resnest50", "resnest101", "resnest200", "resnest269"]
    elif cfg["network_D"]["netD"] == "ResNeSt":
        if cfg["network_D"]["ResNeSt_pretrain"] == "resnest50":
            from arch.discriminators import resnest50

            netD = resnest50(
                pretrained=cfg["network_D"]["pretrained"],
                num_classes=cfg["network_D"]["num_classes"],
            )
        elif cfg["network_D"]["ResNeSt_pretrain"] == "resnest101":
            from arch.discriminators import resnest101

            netD = resnest101(
                pretrained=cfg["network_D"]["pretrained"],
                num_classes=cfg["network_D"]["num_classes"],
            )
        elif cfg["network_D"]["ResNeSt_pretrain"] == "resnest200":
            from arch.discriminators import resnest200

            netD = resnest200(
                pretrained=cfg["network_D"]["pretrained"],
                num_classes=cfg["network_D"]["num_classes"],
            )
        elif cfg["network_D"]["ResNeSt_pretrain"] == "resnest269":
            from arch.discriminators import resnest269

            netD = resnest269(
                pretrained=cfg["network_D"]["pretrained"],
                num_classes=cfg["network_D"]["num_classes"],
            )

    # TODO: need fixing
    # FileNotFoundError: [Errno 2] No such file or directory:
    #   '../experiments/pretrained_models/VGG16minc_53.pth'
    # netD = MINCFeatureExtractor(
    #   feature_layer=34, use_bn=False, use_input_norm=True, device=torch.device('cpu'))

    # Transformer (Warning: uses own init!)
    elif cfg["network_D"]["netD"] == "TranformerDiscriminator":
        from arch.discriminators import TranformerDiscriminator

        netD = TranformerDiscriminator(
            img_size=cfg["network_D"]["img_size"],
            patch_size=cfg["network_D"]["patch_size"],
            in_chans=cfg["network_D"]["in_chans"],
            num_classes=cfg["network_D"]["num_classes"],
            embed_dim=cfg["network_D"]["embed_dim"],
            depth=cfg["network_D"]["depth"],
            num_heads=cfg["network_D"]["num_heads"],
            mlp_ratio=cfg["network_D"]["mlp_ratio"],
            qkv_bias=cfg["network_D"]["qkv_bias"],
            qk_scale=cfg["network_D"]["qk_scale"],
            drop_rate=cfg["network_D"]["drop_rate"],
            attn_drop_rate=cfg["network_D"]["attn_drop_rate"],
            drop_path_rate=cfg["network_D"]["drop_path_rate"],
            hybrid_backbone=cfg["network_D"]["hybrid_backbone"],
            norm_layer=cfg["network_D"]["norm_layer"],
        )

    #############################################

    elif cfg["network_D"]["netD"] == "ViT":
        from vit_pytorch import ViT

        netD = ViT(
            image_size=cfg["network_D"]["image_size"],
            patch_size=cfg["network_D"]["patch_size"],
            num_classes=cfg["network_D"]["num_classes"],
            dim=cfg["network_D"]["dim"],
            depth=cfg["network_D"]["depth"],
            heads=cfg["network_D"]["heads"],
            mlp_dim=cfg["network_D"]["mlp_dim"],
            dropout=cfg["network_D"]["dropout"],
            emb_dropout=cfg["network_D"]["emb_dropout"],
        )

    elif cfg["network_D"]["netD"] == "DeepViT":
        from vit_pytorch.deepvit import DeepViT

        netD = DeepViT(
            image_size=cfg["network_D"]["image_size"],
            patch_size=cfg["network_D"]["patch_size"],
            num_classes=cfg["network_D"]["num_classes"],
            dim=cfg["network_D"]["dim"],
            depth=cfg["network_D"]["depth"],
            heads=cfg["network_D"]["heads"],
            mlp_dim=cfg["network_D"]["mlp_dim"],
            dropout=cfg["network_D"]["dropout"],
            emb_dropout=cfg["network_D"]["emb_dropout"],
        )

    #############################################
    # RepVGG-A0, RepVGG-A1, RepVGG-A2, RepVGG-B0, RepVGG-B1, RepVGG-B1g2, RepVGG-B1g4,
    # RepVGG-B2, RepVGG-B2g2, RepVGG-B2g4, RepVGG-B3, RepVGG-B3g2, RepVGG-B3g4
    elif cfg["network_D"]["netD"] == "RepVGG":
        if cfg["network_D"]["RepVGG_arch"] == "RepVGG-A0":
            from arch.RepVGG_arch import create_RepVGG_A0

            netD = create_RepVGG_A0(
                deploy=False, num_classes=cfg["network_D"]["num_classes"]
            )
        elif cfg["network_D"]["RepVGG_arch"] == "RepVGG-A1":
            from arch.RepVGG_arch import create_RepVGG_A1

            netD = create_RepVGG_A1(
                deploy=False, num_classes=cfg["network_D"]["num_classes"]
            )
        elif cfg["network_D"]["RepVGG_arch"] == "RepVGG-A2":
            from arch.RepVGG_arch import create_RepVGG_A2

            netD = create_RepVGG_A2(
                deploy=False, num_classes=cfg["network_D"]["num_classes"]
            )
        elif cfg["network_D"]["RepVGG_arch"] == "RepVGG-B0":
            from arch.RepVGG_arch import create_RepVGG_B0

            netD = create_RepVGG_B0(
                deploy=False, num_classes=cfg["network_D"]["num_classes"]
            )
        elif cfg["network_D"]["RepVGG_arch"] == "RepVGG-B1":
            from arch.RepVGG_arch import create_RepVGG_B1

            netD = create_RepVGG_B1(
                deploy=False, num_classes=cfg["network_D"]["num_classes"]
            )
        elif cfg["network_D"]["RepVGG_arch"] == "RepVGG-B1g2":
            from arch.RepVGG_arch import create_RepVGG_B1g2

            netD = create_RepVGG_B1g2(
                deploy=False, num_classes=cfg["network_D"]["num_classes"]
            )
        elif cfg["network_D"]["RepVGG_arch"] == "RepVGG-B1g4":
            from arch.RepVGG_arch import create_RepVGG_B1g4

            netD = create_RepVGG_B1g4(
                deploy=False, num_classes=cfg["network_D"]["num_classes"]
            )
        elif cfg["network_D"]["RepVGG_arch"] == "RepVGG-B2":
            from arch.RepVGG_arch import create_RepVGG_B2

            netD = create_RepVGG_B2(
                deploy=False, num_classes=cfg["network_D"]["num_classes"]
            )
        elif cfg["network_D"]["RepVGG_arch"] == "RepVGG-B2g2":
            from arch.RepVGG_arch import create_RepVGG_B2g2

            netD = create_RepVGG_B2g2(
                deploy=False, num_classes=cfg["network_D"]["num_classes"]
            )
        elif cfg["network_D"]["RepVGG_arch"] == "RepVGG-B2g4":
            from arch.RepVGG_arch import create_RepVGG_B2g4

            netD = create_RepVGG_B2g4(
                deploy=False, num_classes=cfg["network_D"]["num_classes"]
            )
        elif cfg["network_D"]["RepVGG_arch"] == "RepVGG-B3":
            from arch.RepVGG_arch import create_RepVGG_B3

            netD = create_RepVGG_B3(
                deploy=False, num_classes=cfg["network_D"]["num_classes"]
            )
        elif cfg["network_D"]["RepVGG_arch"] == "RepVGG-B3g2":
            from arch.RepVGG_arch import create_RepVGG_B3g2

            netD = create_RepVGG_B3g2(
                deploy=False, num_classes=cfg["network_D"]["num_classes"]
            )
        elif cfg["network_D"]["RepVGG_arch"] == "RepVGG-B3g4":
            from arch.RepVGG_arch import create_RepVGG_B3g4

            netD = create_RepVGG_B3g4(
                deploy=False, num_classes=cfg["network_D"]["num_classes"]
            )

    #############################################

    elif cfg["network_D"]["netD"] == "squeezenet":
        from arch.squeezenet_arch import SqueezeNet

        netD = SqueezeNet(
            num_classes=cfg["network_D"]["num_classes"],
            version=cfg["network_D"]["version"],
        )

    #############################################

    elif cfg["network_D"]["netD"] == "SwinTransformer":
        from swin_transformer_pytorch import SwinTransformer

        netD = SwinTransformer(
            hidden_dim=cfg["network_D"]["hidden_dim"],
            layers=cfg["network_D"]["layers"],
            heads=cfg["network_D"]["heads"],
            channels=cfg["network_D"]["channels"],
            num_classes=cfg["network_D"]["num_classes"],
            head_dim=cfg["network_D"]["head_dim"],
            window_size=cfg["network_D"]["window_size"],
            downscaling_factors=cfg["network_D"]["downscaling_factors"],
            relative_pos_embedding=cfg["network_D"]["relative_pos_embedding"],
        )

    # NFNet
    elif cfg["network_D"]["netD"] == "NFNet":
        from arch.NFNet_arch import NFNet

        netD = NFNet(
            num_classes=cfg["network_D"]["num_classes"],
            variant=cfg["network_D"]["variant"],
            stochdepth_rate=cfg["network_D"]["stochdepth_rate"],
            alpha=cfg["network_D"]["alpha"],
            se_ratio=cfg["network_D"]["se_ratio"],
            activation=cfg["network_D"]["activation"],
        )
    elif cfg["network_D"]["netD"] == "lvvit":
        from arch.lvvit_arch import LV_ViT

        netD = LV_ViT(
            img_size=cfg["network_D"]["img_size"],
            patch_size=cfg["network_D"]["patch_size"],
            in_chans=cfg["network_D"]["in_chans"],
            num_classes=cfg["network_D"]["num_classes"],
            embed_dim=cfg["network_D"]["embed_dim"],
            depth=cfg["network_D"]["depth"],
            num_heads=cfg["network_D"]["num_heads"],
            mlp_ratio=cfg["network_D"]["mlp_ratio"],
            qkv_bias=cfg["network_D"]["qkv_bias"],
            qk_scale=cfg["network_D"]["qk_scale"],
            drop_rate=cfg["network_D"]["drop_rate"],
            attn_drop_rate=cfg["network_D"]["attn_drop_rate"],
            drop_path_rate=cfg["network_D"]["drop_path_rate"],
            drop_path_decay=cfg["network_D"]["drop_path_decay"],
            hybrid_backbone=cfg["network_D"]["hybrid_backbone"],
            norm_layer=nn.LayerNorm,
            p_emb=cfg["network_D"]["p_emb"],
            head_dim=cfg["network_D"]["head_dim"],
            skip_lam=cfg["network_D"]["skip_lam"],
            order=cfg["network_D"]["order"],
            mix_token=cfg["network_D"]["mix_token"],
            return_dense=cfg["network_D"]["return_dense"],
        )
    elif cfg["network_D"]["netD"] == "timm":
        import timm

        netD = timm.create_model(
            cfg["network_D"]["timm_model"], num_classes=1, pretrained=True
        )
    elif cfg["network_D"]["netD"] == "resnet3d":
        from arch.resnet3d_arch import generate_model

        netD = generate_model(cfg["network_D"]["model_depth"])
    elif cfg["network_D"]["netD"] == "FFCNLayerDiscriminator":
        from arch.lama_arch import FFCNLayerDiscriminator

        netD = FFCNLayerDiscriminator(3)
    elif cfg["network_D"]["netD"] == "effV2":
        if cfg["network_D"]["size"] == "s":
            from arch.efficientnetV2_arch import effnetv2_s

            netD = effnetv2_s()
        elif cfg["network_D"]["size"] == "m":
            from arch.efficientnetV2_arch import effnetv2_m

            netD = effnetv2_m()
        elif cfg["network_D"]["size"] == "l":
            from arch.efficientnetV2_arch import effnetv2_l

            netD = effnetv2_l()
        elif cfg["network_D"]["size"] == "xl":
            from arch.efficientnetV2_arch import effnetv2_xl

            netD = effnetv2_xl()
    elif cfg["network_D"]["netD"] == "x_transformers":
        from x_transformers import ViTransformerWrapper, Encoder

        netD = ViTransformerWrapper(
            image_size=cfg["network_D"]["image_size"],
            patch_size=cfg["network_D"]["patch_size"],
            num_classes=1,
            attn_layers=Encoder(
                dim=cfg["network_D"]["dim"],
                depth=cfg["network_D"]["depth"],
                heads=cfg["network_D"]["heads"],
            ),
        )
    elif cfg["network_D"]["netD"] == "mobilevit":
        if cfg["network_D"]["size"] == "xxs":
            from arch.mobilevit_arch import mobilevit_xxs

            netD = mobilevit_xxs()
        elif cfg["network_D"]["size"] == "xs":
            from arch.mobilevit_arch import mobilevit_xs

            netD = mobilevit_xs()
        elif cfg["network_D"]["size"] == "x":
            from arch.mobilevit_arch import mobilevit_s

            netD = mobilevit_s()
    elif cfg["network_D"]["netD"] == "hrt":
        from arch.hrt_arch import HighResolutionTransformer

        netD = HighResolutionTransformer()

    elif cfg["network_D"]["netD"] == "attention_unet":
        from arch.attention_unet_arch import UNetDiscriminator

        netD = UNetDiscriminator(
            num_in_ch=cfg["network_D"]["num_in_ch"],
            num_feat=cfg["network_D"]["num_feat"],
            skip_connection=cfg["network_D"]["skip_connection"],
        )

    elif cfg["network_D"]["netD"] == "multiscale_attention_unet":
        from arch.multiscale_attention_unet_arch import MultiscaleAttentionUnet

        netD = MultiscaleAttentionUnet(
            num_in_ch=cfg["network_D"]["num_in_ch"],
            num_feat=cfg["network_D"]["num_feat"],
            num_D=cfg["network_D"]["num_D"],
        )

    elif cfg["network_D"]["netD"] == "unet":
        from arch.unet_arch import UNetDiscriminatorSN

        netD = UNetDiscriminatorSN(
            num_in_ch=cfg["network_D"]["num_in_ch"],
            num_feat=cfg["network_D"]["num_feat"],
            skip_connection=cfg["network_D"]["skip_connection"],
        )

    if cfg["network_D"]["WSConv_replace"] == "True":
        from nfnets import replace_conv, WSConv2d, ScaledStdConv2d

        replace_conv(netD, ScaledStdConv2d)

    return netD
