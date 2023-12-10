import torch.nn.init as init


def weights_init(net, init_type="kaiming", init_gain=0.02):
    # Initialize network weights.
    # Parameters:
    #    net (network)       -- network to be initialized
    #    init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
    #    init_var (float)    -- scaling factor for normal, xavier and orthogonal.

    def init_func(m):
        classname = m.__class__.__name__

        if classname == "AnchorLinear" or classname == "AnchorConv2d":
            return

        if hasattr(m, "weight") and classname.find("Conv") != -1:
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )

            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("Linear") != -1:
            init.normal_(m.weight, 0, 0.01)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias, 0.0)

    # Apply the initialization function <init_func>
    print("Initialization method [{:s}]".format(init_type))
    net.apply(init_func)
