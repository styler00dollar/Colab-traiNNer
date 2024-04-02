def CreateGenerator(cfg, scale):
    # generators with one output, no AMP means nan loss during training
    if cfg["netG"] == "RRDB_net":
        from arch.rrdb_arch import RRDBNet

        netG = RRDBNet(
            in_nc=cfg["in_nc"],
            out_nc=cfg["out_nc"],
            nf=cfg["nf"],
            nb=cfg["nb"],
            gc=cfg["gc"],
            upscale=scale,
            norm_type=cfg["norm_type"],
            act_type=cfg["net_act"],
            mode=cfg["mode"],
            upsample_mode=cfg["upsample_mode"],
            convtype=cfg["convtype"],
            finalact=cfg["finalact"],
            gaussian_noise=cfg["gaussian"],
            plus=cfg["plus"],
            nr=cfg["nr"],
            strided_conv=cfg["strided_conv"],
        )

    # DFNet
    elif cfg["netG"] == "DFNet":
        from arch.DFNet_arch import DFNet

        netG = DFNet(
            c_img=cfg["c_img"],
            c_mask=cfg["c_mask"],
            c_alpha=cfg["c_alpha"],
            mode=cfg["mode"],
            norm=cfg["norm"],
            act_en=cfg["act_en"],
            act_de=cfg["act_de"],
            en_ksize=cfg["en_ksize"],
            de_ksize=cfg["de_ksize"],
            blend_layers=cfg["blend_layers"],
            conv_type=cfg["conv_type"],
        )

    # AdaFill
    elif cfg["netG"] == "AdaFill":
        from arch.AdaFill_arch import InpaintNet

        netG = InpaintNet()

    # MEDFE (batch_size: 1, no AMP)
    elif cfg["netG"] == "MEDFE":
        from arch.MEDFE_arch import MEDFEGenerator

        netG = MEDFEGenerator()

    # RFR
    # conv_type = partial or deform
    # Warning: One testrun with deform resulted in Nan errors after ~60k iterations. It is also very slow.
    # 'partial' is recommended, since this is what the official implementation does use.
    elif cfg["netG"] == "RFR":
        from arch.RFR_arch import RFRNet

        netG = RFRNet(conv_type=cfg["conv_type"])

    # LBAM
    elif cfg["netG"] == "LBAM":
        from arch.LBAM_arch import LBAMModel

        netG = LBAMModel(
            inputChannels=cfg["inputChannels"], outputChannels=cfg["outputChannels"]
        )

    # DMFN
    elif cfg["netG"] == "DMFN":
        from arch.DMFN_arch import InpaintingGenerator

        netG = InpaintingGenerator(
            in_nc=4, out_nc=3, nf=64, n_res=8, norm="in", activation="relu"
        )

    # partial
    elif cfg["netG"] == "Partial":
        from arch.partial_arch import Model

        netG = Model()

    # RN
    elif cfg["netG"] == "RN":
        from arch.RN_arch import G_Net, rn_initialize_weights

        netG = G_Net(
            input_channels=cfg["input_channels"],
            residual_blocks=cfg["residual_blocks"],
            threshold=cfg["threshold"],
        )
        # using rn init to avoid errors
        if self.global_step == 0:
            rn_initialize_weights(netG, scale=0.1)

    # DSNet
    elif cfg["netG"] == "DSNet":
        from arch.DSNet_arch import DSNet

        netG = DSNet(
            layer_size=cfg["layer_sizenr"],
            input_channels=cfg["input_channels"],
            upsampling_mode=cfg["upsampling_mode"],
        )

    # context_encoder
    elif cfg["netG"] == "context_encoder":
        from arch.context_encoder_arch import Net_G

        netG = Net_G()

    # MANet
    elif cfg["netG"] == "MANet":
        from arch.MANet_arch import PConvUNet

        netG = PConvUNet()

    # GPEN
    elif cfg["netG"] == "GPEN":
        from arch.GPEN_arch import FullGenerator

        netG = FullGenerator(
            input_channels=cfg["input_channels"],
            style_dim=cfg["style_dim"],
            n_mlp=cfg["n_mlp"],
            channel_multiplier=cfg["channel_multiplier"],
            blur_kernel=cfg["blur_kernel"],
            lr_mlp=cfg["lr_mlp"],
        )

    # comodgan
    elif cfg["netG"] == "comodgan":
        from arch.comodgan_arch import Generator

        netG = Generator(
            dlatent_size=cfg["dlatent_size"],
            num_channels=cfg["num_channels"],
            resolution=cfg["resolution"],
            fmap_base=cfg["fmap_base"],
            fmap_decay=cfg["fmap_decay"],
            fmap_min=cfg["fmap_min"],
            fmap_max=cfg["fmap_max"],
            randomize_noise=cfg["randomize_noise"],
            architecture=cfg["architecture"],
            nonlinearity=cfg["nonlinearity"],
            resample_kernel=cfg["resample_kernel"],
            fused_modconv=cfg["fused_modconv"],
            pix2pix=cfg["pix2pix"],
            dropout_rate=cfg["dropout_rate"],
            cond_mod=cfg["cond_mod"],
            style_mod=cfg["style_mod"],
            noise_injection=cfg["noise_injection"],
        )

    elif cfg["netG"] == "swinir":
        from arch.swinir_arch import SwinIR

        netG = SwinIR(
            upscale=cfg["upscale"],
            in_chans=cfg["in_chans"],
            img_size=cfg["img_size"],
            window_size=cfg["window_size"],
            img_range=cfg["img_range"],
            depths=cfg["depths"],
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            mlp_ratio=cfg["mlp_ratio"],
            upsampler=cfg["upsampler"],
            resi_connection=cfg["resi_connection"],
        )

    # Experimental

    # DSNetRRDB
    elif cfg["netG"] == "DSNetRRDB":
        from arch.experimental.DSNetRRDB_arch import DSNetRRDB

        netG = DSNetRRDB(
            layer_size=8,
            input_channels=3,
            upsampling_mode="nearest",
            in_nc=4,
            out_nc=3,
            nf=128,
            nb=8,
            gc=32,
            upscale=1,
            norm_type=None,
            act_type="leakyrelu",
            mode="CNA",
            upsample_mode="upconv",
            convtype="Conv2D",
            finalact=None,
            gaussian_noise=True,
            plus=False,
            nr=3,
        )

    # DSNetDeoldify
    elif cfg["netG"] == "DSNetDeoldify":
        from arch.experimental.DSNetDeoldify_arch import DSNetDeoldify

        netG = DSNetDeoldify()

    elif cfg["netG"] == "lightweight_gan":
        from arch.experimental.lightweight_gan_arch import Generator

        netG = Generator(
            image_size=cfg["image_size"],
            latent_dim=cfg["latent_dim"],
            fmap_max=cfg["fmap_max"],
            fmap_inverse_coef=cfg["fmap_inverse_coef"],
            transparent=cfg["transparent"],
            greyscale=cfg["greyscale"],
            freq_chan_attn=cfg["freq_chan_attn"],
        )

    elif cfg["netG"] == "SimpleFontGenerator512":
        from arch.experimental.lightweight_gan_arch import SimpleFontGenerator512

        netG = SimpleFontGenerator512(
            image_size=cfg["image_size"],
            latent_dim=cfg["latent_dim"],
            fmap_max=cfg["fmap_max"],
            fmap_inverse_coef=cfg["fmap_inverse_coef"],
            transparent=cfg["transparent"],
            greyscale=cfg["greyscale"],
            freq_chan_attn=cfg["freq_chan_attn"],
        )

    elif cfg["netG"] == "SimpleFontGenerator256":
        from arch.experimental.lightweight_gan_arch import SimpleFontGenerator256

        netG = SimpleFontGenerator256(
            image_size=cfg["image_size"],
            latent_dim=cfg["latent_dim"],
            fmap_max=cfg["fmap_max"],
            fmap_inverse_coef=cfg["fmap_inverse_coef"],
            transparent=cfg["transparent"],
            greyscale=cfg["greyscale"],
            freq_chan_attn=cfg["freq_chan_attn"],
        )

    ############################

    # generators with two outputs

    # deepfillv1
    elif cfg["netG"] == "deepfillv1":
        from arch.deepfillv1_arch import InpaintSANet

        netG = InpaintSANet()

    # deepfillv2
    # conv_type = partial or deform
    elif cfg["netG"] == "deepfillv2":
        from arch.deepfillv2_arch import GatedGenerator

        netG = GatedGenerator(
            in_channels=cfg["in_channels"],
            out_channels=cfg["out_channels"],
            latent_channels=cfg["latent_channels"],
            pad_type=cfg["pad_type"],
            activation=cfg["activation"],
            norm=cfg["norm"],
            conv_type=cfg["conv_type"],
        )

    # Adaptive
    # [Warning] Adaptive does not like PatchGAN, Multiscale and ResNet.
    elif cfg["netG"] == "Adaptive":
        from arch.Adaptive_arch import PyramidNet

        netG = PyramidNet(
            in_channels=cfg["in_channels"],
            residual_blocks=cfg["residual_blocks"],
            init_weights=cfg["init_weights"],
        )

    ############################
    # exotic generators

    # Pluralistic
    elif cfg["netG"] == "Pluralistic":
        from arch.Pluralistic_arch import PluralisticGenerator

        netG = PluralisticGenerator(
            ngf_E=cfg["ngf_E"],
            z_nc_E=cfg["z_nc_E"],
            img_f_E=cfg["img_f_E"],
            layers_E=cfg["layers_E"],
            norm_E=cfg["norm_E"],
            activation_E=cfg["activation_E"],
            ngf_G=cfg["ngf_G"],
            z_nc_G=cfg["z_nc_G"],
            img_f_G=cfg["img_f_G"],
            L_G=cfg["L_G"],
            output_scale_G=cfg["output_scale_G"],
            norm_G=cfg["norm_G"],
            activation_G=cfg["activation_G"],
        )

    # EdgeConnect
    elif cfg["netG"] == "EdgeConnect":
        from arch.EdgeConnect_arch import EdgeConnectModel

        # conv_type_edge: 'normal' # normal | partial | deform (has no spectral_norm)
        netG = EdgeConnectModel(
            residual_blocks_edge=cfg["residual_blocks_edge"],
            residual_blocks_inpaint=cfg["residual_blocks_inpaint"],
            use_spectral_norm=cfg["use_spectral_norm"],
            conv_type_edge=cfg["conv_type_edge"],
            conv_type_inpaint=cfg["conv_type_inpaint"],
        )

    # FRRN
    elif cfg["netG"] == "FRRN":
        from arch.FRRN_arch import FRRNet

        netG = FRRNet()

    # PRVS
    elif cfg["netG"] == "PRVS":
        from arch.PRVS_arch import PRVSNet

        netG = PRVSNet()

    # CSA
    elif cfg["netG"] == "CSA":
        from arch.CSA_arch import InpaintNet

        netG = InpaintNet(
            c_img=cfg["c_img"],
            norm=cfg["norm"],
            act_en=cfg["act_en"],
            act_de=cfg["network_G"],
        )

    # deoldify
    elif cfg["netG"] == "deoldify":
        from arch.Deoldify_arch import Unet34

        netG = Unet34()

    # GLEAN (does init itself)
    elif cfg["netG"] == "GLEAN":
        from arch.GLEAN_arch import GLEANStyleGANv2

        if cfg["pretrained"] is False:
            netG = GLEANStyleGANv2(
                in_size=cfg["in_size"],
                out_size=cfg["out_size"],
                img_channels=cfg["img_channels"],
                img_channels_out=cfg["img_channels_out"],
                rrdb_channels=cfg["rrdb_channels"],
                num_rrdbs=cfg["num_rrdbs"],
                style_channels=cfg["style_channels"],
                num_mlps=cfg["num_mlps"],
                channel_multiplier=cfg["channel_multiplier"],
                blur_kernel=cfg["blur_kernel"],
                lr_mlp=cfg["lr_mlp"],
                default_style_mode=cfg["default_style_mode"],
                eval_style_mode=cfg["eval_style_mode"],
                mix_prob=cfg["mix_prob"],
                pretrained=None,
                bgr2rgb=cfg["bgr2rgb"],
            )
        else:
            # using stylegan pretrain
            netG = GLEANStyleGANv2(
                in_size=cfg["in_size"],
                out_size=cfg["out_size"],
                img_channels=cfg["img_channels"],
                img_channels_out=cfg["img_channels_out"],
                rrdb_channels=cfg["rrdb_channels"],
                num_rrdbs=cfg["num_rrdbs"],
                style_channels=cfg["style_channels"],
                num_mlps=cfg["num_mlps"],
                channel_multiplier=cfg["channel_multiplier"],
                blur_kernel=cfg["blur_kernel"],
                lr_mlp=cfg["lr_mlp"],
                default_style_mode=cfg["default_style_mode"],
                eval_style_mode=cfg["eval_style_mode"],
                mix_prob=cfg["mix_prob"],
                pretrained=dict(
                    ckpt_path="http://download.openmmlab.com/mmgen/stylegan2/"
                    "official_weights/stylegan2-ffhq-config-f-official_"
                    "20210327_171224-bce9310c.pth",
                    prefix="generator_ema",
                ),
                bgr2rgb=cfg["bgr2rgb"],
            )

    # srflow (weight init?)
    elif cfg["netG"] == "srflow":
        from arch.SRFlowNet_arch import SRFlowNet

        netG = SRFlowNet(
            in_nc=cfg["in_nc"],
            out_nc=cfg["out_nc"],
            nf=cfg["nf"],
            nb=cfg["nb"],
            scale=scale,
            K=cfg["flow"]["K"],
            step=None,
        )

    # DFDNet
    elif cfg["netG"] == "DFDNet":
        from arch.DFDNet_arch import UNetDictFace

        netG = UNetDictFace(64)

    # GFPGAN (error with init?)
    elif cfg["netG"] == "GFPGAN":
        from arch.GFPGAN_arch import GFPGANv1

        netG = GFPGANv1(
            input_channels=cfg["input_channels"],
            output_channels=cfg["output_channels"],
            out_size=cfg["out_size"],
            num_style_feat=cfg["num_style_feat"],
            channel_multiplier=cfg["channel_multiplier"],
            resample_kernel=cfg["resample_kernel"],
            decoder_load_path=cfg["decoder_load_path"],
            fix_decoder=cfg["fix_decoder"],
            num_mlp=cfg["num_mlp"],
            lr_mlp=cfg["lr_mlp"],
            input_is_latent=cfg["input_is_latent"],
            different_w=cfg["different_w"],
            narrow=cfg["narrow"],
            sft_half=cfg["sft_half"],
        )

    elif cfg["netG"] == "CAIN":
        from arch.CAIN_arch import CAIN

        netG = CAIN(cfg["depth"])

    elif cfg["netG"] == "rife":
        from arch.rife_arch import IFNet

        netG = IFNet(
            arch_ver=cfg["arch_ver"],
            fastmode=cfg["fastmode"],
            ensemble=cfg["ensemble"],
        )

    elif cfg["netG"] == "RRIN":
        from arch.RRIN_arch import Net

        netG = Net()

    elif cfg["netG"] == "ABME":
        from arch.ABME_arch import ABME

        netG = ABME()

    elif cfg["netG"] == "EDSC":
        from arch.EDSC_arch import Network

        netG = Network()

    elif cfg["netG"] == "CTSDG":
        from arch.CTSDG_arch import Generator

        netG = Generator()

    elif cfg["netG"] == "MST":
        from arch.MST_arch import InpaintGateGenerator

        netG = InpaintGateGenerator()

    elif cfg["netG"] == "lama":
        from arch.lama_arch import FFCResNetGenerator

        netG = FFCResNetGenerator(4, 3)

    elif cfg["netG"] == "ESRT":
        from arch.ESRT_arch import ESRT

        netG = ESRT(hiddenDim=cfg["hiddenDim"], mlpDim=cfg["mlpDim"], scaleFactor=scale)

    elif cfg["netG"] == "sepconv_enhanced":
        from arch.sepconv_enhanced_arch import Network

        netG = Network()

    elif cfg["netG"] == "sepconv_rt":
        from arch.sepconv_realtime_arch import InterpolationNet

        netG = InterpolationNet(
            real_time=cfg["real_time"],
            device=cfg["device"],
            in_channels=cfg["in_channels"],
            out_channels=cfg["out_channels"],
        )

    elif cfg["netG"] == "CDFI":
        from arch.CDFI_arch import AdaCoFNet

        netG = AdaCoFNet()

    elif cfg["netG"] == "SRVGGNetCompact":
        from arch.SRVGGNetCompact_arch import SRVGGNetCompact

        netG = SRVGGNetCompact(
            num_in_ch=cfg["num_in_ch"],
            num_out_ch=cfg["num_out_ch"],
            num_feat=cfg["num_feat"],
            num_conv=cfg["num_conv"],
            upscale=scale,
            act_type=cfg["act_type"],
            conv_mode=cfg["conv_mode"],
            rrdb=cfg["rrdb"],
            rrdb_blocks=cfg["rrdb_blocks"],
            convtype=cfg["convtype"],
        )

    elif cfg["netG"] == "restormer":
        from arch.restormer_arch import Restormer

        netG = Restormer(
            inp_channels=cfg["inp_channels"],
            out_channels=cfg["out_channels"],
            dim=cfg["dim"],
            num_blocks=cfg["num_blocks"],
            num_refinement_blocks=cfg["num_refinement_blocks"],
            heads=cfg["heads"],
            ffn_expansion_factor=cfg["ffn_expansion_factor"],
            bias=cfg["bias"],
            LayerNorm_type=cfg["LayerNorm_type"],
        )

    elif cfg["netG"] == "swinir2":
        from arch.swinir2_arch import SwinIR

        netG = SwinIR(
            upscale=scale,
            img_size=(cfg["img_size"], cfg["img_size"]),
            window_size=cfg["window_size"],
            img_range=cfg["img_range"],
            depths=cfg["depths"],
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            mlp_ratio=cfg["mlp_ratio"],
            upsampler=cfg["upsampler"],
            use_deformable_block=cfg["use_deformable_block"],
            first_conv=cfg["first_conv"],
        )

    elif cfg["netG"] == "misf":
        from arch.misf_arch import MISF

        netG = MISF(
            residual_blocks=cfg["residual_blocks"],
            use_spectral_norm=cfg["use_spectral_norm"],
        )

    elif cfg["netG"] == "mat":
        from arch.mat_arch import Generator

        netG = Generator(
            z_dim=cfg["z_dim"],
            c_dim=cfg["c_dim"],
            w_dim=cfg["w_dim"],
            img_resolution=cfg["img_resolution"],
            img_channels=cfg["img_channels"],
            noise_mode=cfg["noise_mode"],
        )

    elif cfg["netG"] == "elan":
        from arch.elan_arch import ELAN

        netG = ELAN(
            scale=scale,
            colors=cfg["colors"],
            window_sizes=cfg["window_sizes"],
            m_elan=cfg["m_elan"],
            c_elan=cfg["c_elan"],
            n_share=cfg["n_share"],
            r_expand=cfg["r_expand"],
            rgb_range=cfg["rgb_range"],
            conv=cfg["conv"],
        )

    elif cfg["netG"] == "lft":
        from arch.lft_arch import LFT

        netG = LFT(
            channels=cfg["channels"],
            angRes=cfg["angRes"],
            scale_factor=scale,
            layer_num=cfg["layer_num"],
            temperature=cfg["temperature"],
            num_heads=cfg["num_heads"],
            dropout=cfg["dropout"],
        )

    elif cfg["netG"] == "swift":
        from arch.swift_arch import Swift

        netG = Swift(
            in_channels=cfg["in_channels"],
            num_channels=cfg["num_channels"],
            num_blocks=cfg["num_blocks"],
            upscale_factor=scale,
        )

    elif cfg["netG"] == "hat":
        from arch.hat_arch import HAT

        netG = HAT(
            img_size=cfg["img_size"],
            patch_size=cfg["patch_size"],
            in_chans=cfg["in_chans"],
            embed_dim=cfg["embed_dim"],
            depths=cfg["depths"],
            num_heads=cfg["num_heads"],
            window_size=cfg["window_size"],
            compress_ratio=cfg["compress_ratio"],
            squeeze_factor=cfg["squeeze_factor"],
            conv_scale=cfg["conv_scale"],
            overlap_ratio=cfg["overlap_ratio"],
            mlp_ratio=cfg["mlp_ratio"],
            qkv_bias=cfg["qkv_bias"],
            qk_scale=cfg["qk_scale"],
            drop_rate=cfg["drop_rate"],
            attn_drop_rate=cfg["attn_drop_rate"],
            drop_path_rate=cfg["drop_path_rate"],
            ape=cfg["ape"],
            patch_norm=cfg["patch_norm"],
            use_checkpoint=cfg["use_checkpoint"],
            upscale=scale,
            img_range=cfg["img_range"],
            upsampler=cfg["upsampler"],
            resi_connection=cfg["resi_connection"],
            conv=cfg["conv"],
        )

    elif cfg["netG"] == "RLFN":
        from arch.RLFN_arch import RLFN

        netG = RLFN(
            in_nc=cfg["in_nc"],
            out_nc=cfg["out_nc"],
            nf=cfg["nf"],
            mf=cfg["mf"],
            upscale=scale,
        )

    elif cfg["netG"] == "SCET":
        from arch.scet_arch import SCET

        netG = SCET(
            hiddenDim=cfg["hiddenDim"],
            mlpDim=cfg["mlpDim"],
            scaleFactor=scale,
            conv=cfg["conv"],
        )

    elif cfg["netG"] == "GMFSS_union":
        from arch.GMFSS_union_arch import GMFSS_union

        netG = GMFSS_union()

    elif cfg["netG"] == "cugan":
        from arch.cugan_arch import cugan

        netG = cugan(
            in_channels=cfg["in_channels"],
            out_channels=cfg["out_channels"],
            scale=scale,
            pro_mode=cfg["pro_mode"],
        )

    elif cfg["netG"] == "SAFMN":
        from arch.safmn_arch import SAFMN

        netG = SAFMN(
            dim=cfg["dim"],
            n_blocks=cfg["n_blocks"],
            ffn_scale=cfg["ffn_scale"],
            upscaling_factor=scale,
        )

    elif cfg["netG"] == "MFRAN":
        from arch.mfran_arch import MFRAN

        netG = MFRAN(
            n_feats=cfg["n_feats"],
            n_blocks=cfg["n_blocks"],
            kernel_size=cfg["kernel_size_MFRAN"],
            scale=scale,
            div=cfg["div"],
            rgb_range=cfg["rgb_range"],
            n_colors=cfg["n_colors"],
            path=cfg["path"],
        )

    elif cfg["netG"] == "OmniSR":
        from arch.omnisr_arch import OmniSR

        netG = OmniSR(
            num_in_ch=cfg["num_in_ch"],
            num_out_ch=cfg["num_out_ch"],
            num_feat=cfg["num_feat"],
            window_size=cfg["window_size"],
            res_num=cfg["res_num"],
            upsampling=scale,
            bias=cfg["bias"],
            block_num=cfg["block_num"],
            pe=cfg["pe"],
            ffn_bias=cfg["ffn_bias"],
        )

    elif cfg["netG"] == "EMT":
        from arch.EMT_arch import EMT

        netG = EMT(
            upscale=scale,
            dim=cfg["dim"],
            n_blocks=cfg["n_blocks"],
            n_layers=cfg["n_layers"],
            num_heads=cfg["num_heads"],
            mlp_ratio=cfg["mlp_ratio"],
            n_GTLs=cfg["n_GTLs"],
            window_list=cfg["window_list"],
            shift_list=cfg["shift_list"],
            task=cfg["task"],
        )

    elif cfg["netG"] == "lkdn":
        from arch.lkdn_arch import LKDN

        netG = LKDN(
            num_in_ch=cfg["num_in_ch"],
            num_out_ch=cfg["num_out_ch"],
            num_feat=cfg["num_feat"],
            num_atten=cfg["num_atten"],
            num_block=cfg["num_block"],
            upscale=scale,
            num_in=cfg["num_in"],
            conv=cfg["conv"],
            upsampler=cfg["upsampler"],
        )

    elif cfg["netG"] == "DITN":
        from arch.DITN_arch import DITN

        netG = DITN(
            inp_channels=cfg["inp_channels"],
            dim=cfg["dim"],
            ITL_blocks=cfg["ITL_blocks"],
            SAL_blocks=cfg["SAL_blocks"],
            UFONE_blocks=cfg["UFONE_blocks"],
            ffn_expansion_factor=cfg["ffn_expansion_factor"],
            bias=cfg["bias"],
            LayerNorm_type=cfg["LayerNorm_type"],
            patch_size=cfg["patch_size"],
            upscale=scale,
        )

    elif cfg["netG"] == "dat":
        from arch.dat_arch import DAT

        netG = DAT(
            img_size=cfg["img_size"],
            in_chans=cfg["in_chans"],
            embed_dim=cfg["embed_dim"],
            split_size=cfg["split_size"],
            depth=cfg["depth"],
            num_heads=cfg["num_heads"],
            expansion_factor=cfg["expansion_factor"],
            qkv_bias=cfg["qkv_bias"],
            qk_scale=cfg["qk_scale"],
            drop_rate=cfg["drop_rate"],
            attn_drop_rate=cfg["attn_drop_rate"],
            drop_path_rate=cfg["drop_path_rate"],
            use_chk=cfg["use_chk"],
            upscale=scale,
            img_range=cfg["img_range"],
            resi_connection=cfg["resi_connection"],
            upsampler=cfg["upsampler"],
        )

    elif cfg["netG"] == "DCTLSA":
        from arch.dctlsa_arch import DCTLSA

        netG = DCTLSA(
            in_nc=cfg["in_nc"],
            nf=cfg["nf"],
            num_modules=cfg["num_modules"],
            out_nc=cfg["out_nc"],
            upscale=scale,
            num_head=cfg["num_head"],
        )

    elif cfg["netG"] == "grl":
        from arch.grl_arch import GRL

        netG = GRL(
            img_size=cfg["img_size"],
            window_size=cfg["window_size"],
            depths=cfg["depths"],
            embed_dim=cfg["embed_dim"],
            num_heads_window=cfg["num_heads_window"],
            num_heads_stripe=cfg["num_heads_stripe"],
            mlp_ratio=cfg["mlp_ratio"],
            qkv_proj_type=cfg["qkv_proj_type"],
            anchor_proj_type=cfg["anchor_proj_type"],
            anchor_window_down_factor=cfg["anchor_window_down_factor"],
            out_proj_type=cfg["out_proj_type"],
            conv_type=cfg["conv_type"],
            upsampler=cfg["upsampler"],
            local_connection=cfg["local_connection"],
            upscale=scale,
        )

    elif cfg["netG"] == "craft":
        from arch.craft_arch import CRAFT

        netG = CRAFT(
            in_chans=cfg["in_chans"],
            embed_dim=cfg["embed_dim"],
            depths=cfg["depths"],
            num_heads=cfg["num_heads"],
            split_size_0=cfg["split_size_0"],
            split_size_1=cfg["split_size_1"],
            mlp_ratio=cfg["mlp_ratio"],
            qkv_bias=cfg["qkv_bias"],
            qk_scale=cfg["qk_scale"],
            img_range=cfg["img_range"],
            upsampler=cfg["upsampler"],
            resi_connection=cfg["resi_connection"],
            upscale=scale,
        )

    elif cfg["netG"] == "srformer":
        from arch.srformer_arch import SRFormer

        netG = SRFormer(
            img_size=cfg["img_size"],
            patch_size=cfg["patch_size"],
            in_chans=cfg["in_chans"],
            embed_dim=cfg["embed_dim"],
            depths=cfg["depths"],
            num_heads=cfg["num_heads"],
            window_size=cfg["window_size"],
            mlp_ratio=cfg["mlp_ratio"],
            qkv_bias=cfg["qkv_bias"],
            qk_scale=cfg["qk_scale"],
            drop_rate=cfg["drop_rate"],
            attn_drop_rate=cfg["attn_drop_rate"],
            drop_path_rate=cfg["drop_path_rate"],
            ape=cfg["ape"],
            patch_norm=cfg["patch_norm"],
            use_checkpoint=cfg["use_checkpoint"],
            upscale=scale,
            img_range=cfg["img_range"],
            upsampler=cfg["upsampler"],
            resi_connection=cfg["resi_connection"],
        )

    elif cfg["netG"] == "span":
        from arch.span_arch import span

        netG = span(
            num_in_ch=cfg["num_in_ch"],
            num_out_ch=cfg["num_out_ch"],
            feature_channels=cfg["feature_channels"],
            upscale=scale,
            bias=cfg["bias"],
            img_range=cfg["img_range"],
            rgb_mean=cfg["rgb_mean"],
        )

    elif cfg["netG"] == "rgt":
        from arch.rgt_arch import RGT

        netG = RGT(
            img_size=cfg["img_size"],
            in_chans=cfg["in_chans"],
            embed_dim=cfg["embed_dim"],
            depth=cfg["depth"],
            num_heads=cfg["num_heads"],
            mlp_ratio=cfg["mlp_ratio"],
            qkv_bias=cfg["qkv_bias"],
            qk_scale=cfg["qk_scale"],
            drop_rate=cfg["drop_rate"],
            attn_drop_rate=cfg["attn_drop_rate"],
            drop_path_rate=cfg["drop_path_rate"],
            use_chk=cfg["use_chk"],
            upscale=scale,
            img_range=cfg["img_range"],
            resi_connection=cfg["resi_connection"],
            split_size=cfg["split_size"],
            c_ratio=cfg["c_ratio"],
        )

    # DFNet
    elif cfg["netG"] == "SwinFIR":
        from arch.swinfir_arch import SwinFIR

        netG = SwinFIR(
            img_size=cfg["img_size"],
            patch_size=cfg["patch_size"],
            in_chans=cfg["in_chans"],
            embed_dim=cfg["embed_dim"],
            depths=cfg["depths"],
            num_heads=cfg["num_heads"],
            window_size=cfg["window_size"],
            mlp_ratio=cfg["mlp_ratio"],
            qkv_bias=cfg["qkv_bias"],
            qk_scale=cfg["qk_scale"],
            drop_rate=cfg["drop_rate"],
            attn_drop_rate=cfg["attn_drop_rate"],
            drop_path_rate=cfg["drop_path_rate"],
            ape=cfg["ape"],
            patch_norm=cfg["patch_norm"],
            use_checkpoint=cfg["use_checkpoint"],
            upscale=scale,
            img_range=cfg["img_range"],
            upsampler=cfg["upsampler"],
            resi_connection=cfg["resi_connection"],
        )

    ############################

    if cfg["CEM"] is True:
        from arch.CEM import CEMnet

        CEM_conf = CEMnet.Get_CEM_Conf(scale)
        CEM_conf.sigmoid_range_limit = cfg["sigmoid_range_limit"]
        if CEM_conf.sigmoid_range_limit:
            CEM_conf.input_range = [-1, 1] if z_norm else [0, 1]
        kernel = None  # note: could pass a kernel here, but None will use default cubic kernel
        CEM_net = CEMnet.CEMnet(CEM_conf, upscale_kernel=kernel)
        CEM_net.WrapArchitecture(only_padders=True)
        netG = CEM_net.WrapArchitecture(
            netG, training_patch_size=cfg["datasets"]["train"]["HR_size"]
        )

    return netG
