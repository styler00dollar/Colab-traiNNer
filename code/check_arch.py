def check_arch(cfg):
    # if inpainting
    if cfg["network_G"]["netG"] in (  # real inpainting generators
        "lama",
        "MST",
        "MANet",
        "context_encoder",
        "DFNet",
        "AdaFill",
        "MEDFE",
        "RFR",
        "LBAM",
        "DMFN",
        "Partial",
        "RN",
        "DSNet",
        "DSNetRRDB",
        "DSNetDeoldify",
        "EdgeConnect",
        "CSA",
        "deepfillv1",
        "deepfillv2",
        "Adaptive",
        "Global",
        "Pluralistic",
        "crfill",
        "DeepDFNet",
        "pennet",
        "FRRN",
        "PRVS",
        "CRA",
        "atrous",
        "lightweight_gan",
        "CTSDG",
        "misf",
        "mat",
        # sr genrators
        "restormer",
        "SRVGGNetCompact",
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
    ) and cfg["datasets"]["train"]["mode"] in ("DS_inpaint", "DS_inpaint_TF"):
        if cfg["network_G"]["netG"] in ("PRVS", "CTSDG"):
            # arch, edge, grayscale, landmarks
            return "inpainting", True, False, False
        elif cfg["network_G"]["netG"] in ("EdgeConnect", "misf"):
            # arch, edge, grayscale, landmarks
            return "inpainting", True, True, False
        else:
            return "inpainting", False, False, False

    # if super resolution
    elif cfg["network_G"]["netG"] in (
        "restormer",
        "SRVGGNetCompact",
        "ESRT",
        "swinir",
        "lightweight_gan",
        "RRDB_net",
        "GLEAN",
        "GPEN",
        "comodgan",
        "ASRGAN",
        "PPON",
        "sr_resnet",
        "PAN",
        "sisr",
        "USRNet",
        "srflow",
        "DFDNet",
        "GFPGAN",
        "GPEN",
        "comodgan",
        "ESRT",
        "SRVGGNetCompact",
        "swinir2",
        "MRRDBNet_FM",
        "elan",
        "lft",
    ) and cfg["datasets"]["train"]["mode"] in ("DS_lrhr", "DS_realesrgan"):
        if cfg["network_G"]["netG"] == "DFDNet":
            # arch, edge, grayscale, landmarks
            "sr", False, False, True
        else:
            return "sr", False, False, False

    # if interpolation
    elif cfg["network_G"]["netG"] in (
        "CDFI",
        "sepconv_enhanced",
        "CAIN",
        "rife",
        "RRIN",
        "ABME",
        "EDSC",
        "sepconv_rt",
    ):
        # arch, edge, grayscale, landmarks
        return "interpolation", False, False, False
