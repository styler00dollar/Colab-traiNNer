import pytorch_optimizer


def CreateOptimizer(cfg, input_G, input_D=None):
    opt_d = None  # if there is no optimizer, return nothing
    if cfg["network_G"]["finetune"] is None or cfg["network_G"]["finetune"] is False:
        if cfg["train"]["scheduler"] == "AdaBelief":
            opt_g = pytorch_optimizer.AdaBelief(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.AdaBelief(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "AdaBound":
            opt_g = pytorch_optimizer.AdaBound(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.AdaBound(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "Adai":
            opt_g = pytorch_optimizer.Adai(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.Adai(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "AdamP":
            opt_g = pytorch_optimizer.AdamP(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.AdamP(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "Adan":
            opt_g = pytorch_optimizer.Adan(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.Adan(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "AdaPNM":
            opt_g = pytorch_optimizer.AdaPNM(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.AdaPNM(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "DiffGrad":
            opt_g = pytorch_optimizer.DiffGrad(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.DiffGrad(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "Lamb":
            opt_g = pytorch_optimizer.Lamb(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.Lamb(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "LARS":
            opt_g = pytorch_optimizer.LARS(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.LARS(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "MADGRAD":
            opt_g = pytorch_optimizer.MADGRAD(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.MADGRAD(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "Nero":
            opt_g = pytorch_optimizer.Nero(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.Nero(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "PNM":
            opt_g = pytorch_optimizer.PNM(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.PNM(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "RAdam":
            opt_g = pytorch_optimizer.RAdam(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.RAdam(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "Ranger":
            opt_g = pytorch_optimizer.Ranger(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.Ranger(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "Ranger21":
            opt_g = pytorch_optimizer.Ranger21(
                input_G, lr=cfg["train"]["lr_g"], num_iterations=1000
            )
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.Ranger21(
                    input_D, lr=cfg["train"]["lr_g"], num_iterations=1000
                )
        if cfg["train"]["scheduler"] == "SGDP":
            opt_g = pytorch_optimizer.SGDP(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.SGDP(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "Shampoo":
            opt_g = pytorch_optimizer.Shampoo(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.Shampoo(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "ScalableShampoo":
            opt_g = pytorch_optimizer.ScalableShampoo(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.ScalableShampoo(
                    input_D, lr=cfg["train"]["lr_g"]
                )
        if cfg["train"]["scheduler"] == "DAdaptAdaGrad":
            opt_g = pytorch_optimizer.DAdaptAdaGrad(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.DAdaptAdaGrad(
                    input_D, lr=cfg["train"]["lr_g"]
                )
        if cfg["train"]["scheduler"] == "DAdaptAdam":
            opt_g = pytorch_optimizer.DAdaptAdam(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.DAdaptAdam(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "DAdaptSGD":
            opt_g = pytorch_optimizer.DAdaptSGD(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.DAdaptSGD(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "DAdaptAdan":
            opt_g = pytorch_optimizer.DAdaptAdan(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.DAdaptAdan(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "AdamS":
            opt_g = pytorch_optimizer.AdamS(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.AdamS(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "AdaFactor":
            opt_g = pytorch_optimizer.AdaFactor(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.AdaFactor(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "Apollo":
            opt_g = pytorch_optimizer.Apollo(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.Apollo(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "NovoGrad":
            opt_g = pytorch_optimizer.NovoGrad(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.NovoGrad(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "Lion":
            opt_g = pytorch_optimizer.Lion(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.Lion(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "AliG":
            opt_g = pytorch_optimizer.AliG(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.AliG(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "AdaNorm":
            opt_g = pytorch_optimizer.AdaNorm(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.AdaNorm(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "A2Grad":
            opt_g = pytorch_optimizer.A2Grad(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.A2Grad(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "AccSGD":
            opt_g = pytorch_optimizer.AccSGD(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.AccSGD(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "SGDW":
            opt_g = pytorch_optimizer.SGDW(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.SGDW(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "ASGD":
            opt_g = pytorch_optimizer.ASGD(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.ASGD(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "Yogi":
            opt_g = pytorch_optimizer.Yogi(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.Yogi(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "SWATS":
            opt_g = pytorch_optimizer.SWATS(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.SWATS(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "Fromage":
            opt_g = pytorch_optimizer.Fromage(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.Fromage(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "MSVAG":
            opt_g = pytorch_optimizer.MSVAG(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.MSVAG(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "AdaMod":
            opt_g = pytorch_optimizer.AdaMod(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.AdaMod(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "AggMo":
            opt_g = pytorch_optimizer.AggMo(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.AggMo(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "QHAdam":
            opt_g = pytorch_optimizer.QHAdam(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.QHAdam(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "QHM":
            opt_g = pytorch_optimizer.QHM(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.QHM(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "PID":
            opt_g = pytorch_optimizer.PID(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.PID(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "AdaMax":
            opt_g = pytorch_optimizer.AdaMax(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.AdaMax(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "Gravity":
            opt_g = pytorch_optimizer.Gravity(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.Gravity(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "AdaSmooth":
            opt_g = pytorch_optimizer.AdaSmooth(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.AdaSmooth(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "SRMM":
            opt_g = pytorch_optimizer.SRMM(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.SRMM(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "AvaGrad":
            opt_g = pytorch_optimizer.AvaGrad(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.AvaGrad(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "AdaShift":
            opt_g = pytorch_optimizer.AdaShift(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.AdaShift(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "AdaDelta":
            opt_g = pytorch_optimizer.AdaDelta(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.AdaDelta(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "Amos":
            opt_g = pytorch_optimizer.Amos(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.Amos(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "SignSGD":
            opt_g = pytorch_optimizer.SignSGD(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.SignSGD(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "SophiaH":
            opt_g = pytorch_optimizer.SophiaH(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.SophiaH(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "Prodigy":
            opt_g = pytorch_optimizer.Prodigy(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.Prodigy(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "PAdam":
            opt_g = pytorch_optimizer.PAdam(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.PAdam(input_D, lr=cfg["train"]["lr_g"])
        if cfg["train"]["scheduler"] == "Tiger":
            opt_g = pytorch_optimizer.Tiger(input_G, lr=cfg["train"]["lr_g"])
            if cfg["network_D"]["netD"] is not None:
                opt_d = pytorch_optimizer.Tiger(input_D, lr=cfg["train"]["lr_g"])
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

    if cfg["train"]["AGC"] is True:
        from nfnets.agc import AGC

        opt_g = AGC(input_G, opt_g)

    return opt_g, opt_d
