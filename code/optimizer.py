def CreateOptimizer(cfg, input_G, input_D=None):
    opt_d = None  # if there is no optimizer, return nothing
    if cfg["network_G"]["finetune"] is None or cfg["network_G"]["finetune"] is False:
        if cfg["train"]["scheduler"] == "Adam":
            import torch

            opt_g = torch.optim.Adam(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = torch.optim.Adam(input_D, lr=cfg["train"]["lr_d"])
        if cfg["train"]["scheduler"] == "AdamP":
            from adamp import AdamP

            opt_g = AdamP(
                input_G,
                lr=cfg["train"]["lr_g"],
                betas=(float(cfg["train"]["betas0"]), float(cfg["train"]["betas1"])),
                weight_decay=float(cfg["train"]["weight_decay"]),
            )
            if cfg["network_D"]["netD"] is not None:
                opt_d = AdamP(
                    input_D,
                    lr=cfg["train"]["lr_d"],
                    betas=(
                        float(cfg["train"]["betas0"]),
                        float(cfg["train"]["betas1"]),
                    ),
                    weight_decay=float(cfg["train"]["weight_decay"]),
                )
        if cfg["train"]["scheduler"] == "SGDP":
            from adamp import SGDP

            opt_g = SGDP(
                input_G,
                lr=cfg["train"]["lr_g"],
                weight_decay=cfg["train"]["weight_decay"],
                momentum=cfg["train"]["momentum"],
                nesterov=cfg["train"]["nesterov"],
            )
            if cfg["network_D"]["netD"] is not None:
                opt_d = SGDP(
                    input_D,
                    lr=cfg["train"]["lr_d"],
                    weight_decay=cfg["train"]["weight_decay"],
                    momentum=cfg["train"]["momentum"],
                    nesterov=cfg["train"]["nesterov"],
                )
        if cfg["train"]["scheduler"] == "MADGRAD":
            from madgrad import MADGRAD

            opt_g = MADGRAD(
                input_G,
                lr=cfg["train"]["lr_g"],
                momentum=cfg["train"]["momentum"],
                weight_decay=cfg["train"]["weight_decay"],
                eps=cfg["train"]["eps"],
            )
            if cfg["network_D"]["netD"] is not None:
                opt_d = MADGRAD(
                    input_D,
                    lr=cfg["train"]["lr_d"],
                    momentum=cfg["train"]["momentum"],
                    weight_decay=cfg["train"]["weight_decay"],
                    eps=cfg["train"]["eps"],
                )
        if cfg["train"]["scheduler"] == "cosangulargrad":
            from arch.optimizer.cosangulargrad import cosangulargrad

            opt_g = cosangulargrad(
                input_G,
                lr=cfg["train"]["lr_g"],
                betas=(float(cfg["train"]["betas0"]), float(cfg["train"]["betas1"])),
                eps=cfg["train"]["eps"],
                weight_decay=cfg["train"]["weight_decay"],
            )
            if cfg["network_D"]["netD"] is not None:
                opt_d = cosangulargrad(
                    input_D,
                    lr=cfg["train"]["lr_d"],
                    betas=(
                        float(cfg["train"]["betas0"]),
                        float(cfg["train"]["betas1"]),
                    ),
                    eps=cfg["train"]["eps"],
                    weight_decay=cfg["train"]["weight_decay"],
                )
        if cfg["train"]["scheduler"] == "tanangulargrad":
            from arch.optimizer.tanangulargrad import tanangulargrad

            opt_g = tanangulargrad(
                input_G,
                lr=cfg["train"]["lr_g"],
                betas=(float(cfg["train"]["betas0"]), float(cfg["train"]["betas1"])),
                eps=cfg["train"]["eps"],
                weight_decay=cfg["train"]["weight_decay"],
            )
            if cfg["network_D"]["netD"] is not None:
                opt_d = tanangulargrad(
                    input_D,
                    lr=cfg["train"]["lr_d"],
                    betas=(
                        float(cfg["train"]["betas0"]),
                        float(cfg["train"]["betas1"]),
                    ),
                    eps=cfg["train"]["eps"],
                    weight_decay=cfg["train"]["weight_decay"],
                )
        if cfg["train"]["scheduler"] == "Adam8bit":
            import bitsandbytes as bnb

            opt_g = bnb.optim.Adam8bit(
                input_G,
                lr=cfg["train"]["lr_g"],
                betas=(float(cfg["train"]["betas0"]), float(cfg["train"]["betas1"])),
            )
            if cfg["network_D"]["netD"] is not None:
                opt_d = bnb.optim.Adam8bit(
                    input_D,
                    lr=cfg["train"]["lr_d"],
                    betas=(
                        float(cfg["train"]["betas0"]),
                        float(cfg["train"]["betas1"]),
                    ),
                )
        if cfg["train"]["scheduler"] == "SGD_AGC":
            from nfnets import SGD_AGC

            opt_g = SGD_AGC(
                input_G,
                lr=cfg["train"]["lr_g"],
                weight_decay=cfg["train"]["weight_decay"],
                eps=cfg["train"]["eps"],
            )
            if cfg["network_D"]["netD"] is not None:
                opt_d = SGD_AGC(
                    input_D,
                    lr=cfg["train"]["lr_d"],
                    weight_decay=cfg["train"]["weight_decay"],
                    eps=cfg["train"]["eps"],
                )

        if cfg["train"]["scheduler"] == "Adan":
            from arch.optimizer.adan import Adan

            opt_g = Adan(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = Adan(input_D, lr=cfg["train"]["lr_d"])

        if cfg["train"]["scheduler"] == "Lamb":
            from arch.optimizer.lamb import Lamb

            opt_g = Lamb(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = Lamb(input_D, lr=cfg["train"]["lr_d"])

    if cfg["train"]["AGC"] is True:
        from nfnets.agc import AGC

        opt_g = AGC(input_G, opt_g)
    return opt_g, opt_d
