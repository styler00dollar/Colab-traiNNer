def generate(
    cfg,
    lr_image=None,
    hr_image=None,
    netG=None,
    other=None,
    global_step=None,
    arch=None,
    arch_name=None,
):
    # train generator
    ############################
    # inpainting
    if arch == "inpainting" and cfg["network_G"]["netG"] in (
        "deepfillv1",
        "deepfillv2",
        "Adaptive",
    ):
        out, other["other_img"] = netG(lr_image, other["mask"])
    elif arch == "inpainting" and cfg["network_G"]["netG"] == "CSA":
        other["coarse_result"], out, other["csa"], other["csa_d"] = netG(
            lr_image, other["mask"]
        )
    elif arch == "inpainting" and cfg["network_G"]["netG"] in ("EdgeConnect", "misf"):
        out, other["other_img"] = netG(
            lr_image, other["edge"], other["grayscale"], other["mask"]
        )
    elif arch == "inpainting" and cfg["network_G"]["netG"] == "PVRS":
        out, _, other["edge_small"], other["edge_big"] = netG(
            lr_image, other["mask"], other["edge"]
        )
    elif arch == "inpainting" and cfg["network_G"]["netG"] == "FRRN":
        out, other["mid_x"], other["mid_mask"] = netG(lr_image, other["mask"])
    elif cfg["network_G"]["netG"] == "CTSDG":
        out, other["projected_image"], other["projected_edge"] = netG(
            lr_image, other["edge"], other["mask"]
        )
    elif arch == "inpainting":
        out = netG(lr_image, other["mask"])

    # if inpaint, masking, taking original content from HR
    if arch == "inpainting":
        out = lr_image * other["mask"] + out * (1 - other["mask"])

    #########################

    # frame interpolation
    if arch == "interpolation" and arch != "rife":
        out = netG(other["hr_image1"], other["hr_image3"])

    elif arch == "interpolation" and arch == "rife":
        out, other["flow"] = netG(other["hr_image1"], other["hr_image3"], training=True)

    ############################
    # sr networks (sr / inpainting)
    if arch_name in (
        "restormer",
        "ESRT",
        "swinir",
        "lightweight_gan",
        "RRDB_net",
        "GLEAN",
        "GPEN",
        "comodgan",
        "GFPGAN",
        "swinir2",
        "elan",
        "lft",
        "swift",
        "hat",
        "RLFN",
        "SCET",
        "UpCunet2x_fast",
    ):
        if cfg["datasets"]["train"]["mode"] in ("DS_inpaint", "DS_inpaint_TF"):
            # masked test with inpaint dataloader
            out = netG(torch.cat([lr_image, other["mask"]], 1))
            out = lr_image * other["mask"] + out * (1 - other["mask"])
        else:
            # normal dataloader
            out = netG(lr_image)

    ############################

    # esrgan with feature maps
    if arch_name in ("MRRDBNet_FM", "SRVGGNetCompact"):
        out, other["feature_maps"] = netG(lr_image)

    # deoldify
    if arch_name == "deoldify":
        out = netG(lr_image)

    # GFPGAN
    if arch_name == "GFPGAN":
        if cfg["datasets"]["train"]["mode"] in ("DS_inpaint", "DS_inpaint_TF"):
            # masked test with inpaint dataloader
            out, _ = netG(torch.cat([lr_image, other["mask"]], 1))
            out = lr_image * other["mask"] + out * (1 - other["mask"])
        else:
            out, _ = netG(lr_image)

    if arch_name == "srflow":
        from arch.SRFlowNet_arch import get_z

        # freeze rrdb in the beginning
        if global_step < cfg["network_G"]["freeze_iter"]:
            netG.set_rrdb_training(False)
        else:
            netG.set_rrdb_training(True)

        z = get_z(
            heat=0,
            seed=None,
            batch_size=lr_image.shape[0],
            lr_shape=lr_image.shape,
        )
        out, logdet = netG(
            lr=lr_image, z=z, eps_std=0, reverse=True, reverse_with_grad=True
        )

    # DFDNet
    if arch_name == "DFDNet":
        out = netG(lr_image, part_locations=other["landmarks"])
        # range [-1, 1] to [0, 1]
        out = out + 1
        out = out - out.min()
        out = out / (out.max() - out.min())

    return out, other
