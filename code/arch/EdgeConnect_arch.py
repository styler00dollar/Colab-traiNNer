"""
edge_connect.py (12-12-20)
https://github.com/knazeri/edge-connect/blob/master/src/edge_connect.py
"""

import torch
import torch.nn as nn

# from models.modules.architectures.convolutions.partialconv2d import PartialConv2d
# from models.modules.architectures.convolutions.deformconv2d import DeformConv2d
import pytorch_lightning as pl


class InpaintGenerator(pl.LightningModule):
    def __init__(self, residual_blocks=8, init_weights=True, conv_type="deform"):
        super(InpaintGenerator, self).__init__()

        if conv_type == "normal":
            self.encoder = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
                nn.InstanceNorm2d(64, track_running_stats=False),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1
                ),
                nn.InstanceNorm2d(128, track_running_stats=False),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.InstanceNorm2d(256, track_running_stats=False),
                nn.ReLU(True),
            )
        elif conv_type == "partial":
            self.encoder = nn.Sequential(
                nn.ReflectionPad2d(3),
                PartialConv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
                nn.InstanceNorm2d(64, track_running_stats=False),
                nn.ReLU(True),
                PartialConv2d(
                    in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1
                ),
                nn.InstanceNorm2d(128, track_running_stats=False),
                nn.ReLU(True),
                PartialConv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.InstanceNorm2d(256, track_running_stats=False),
                nn.ReLU(True),
            )
        elif conv_type == "deform":
            self.encoder = nn.Sequential(
                nn.ReflectionPad2d(3),
                DeformConv2d(in_nc=4, out_nc=64, kernel_size=7, padding=0),
                nn.InstanceNorm2d(64, track_running_stats=False),
                nn.ReLU(True),
                DeformConv2d(in_nc=64, out_nc=128, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(128, track_running_stats=False),
                nn.ReLU(True),
                DeformConv2d(in_nc=128, out_nc=256, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(256, track_running_stats=False),
                nn.ReLU(True),
            )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1
            ),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1
            ),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = (torch.tanh(x) + 1) / 2

        return x


class EdgeGenerator(pl.LightningModule):
    def __init__(
        self,
        residual_blocks=8,
        use_spectral_norm=True,
        init_weights=True,
        conv_type="normal",
    ):
        super(EdgeGenerator, self).__init__()

        if conv_type == "normal":
            self.encoder = nn.Sequential(
                nn.ReflectionPad2d(3),
                spectral_norm(
                    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0),
                    use_spectral_norm,
                ),
                nn.InstanceNorm2d(64, track_running_stats=False),
                nn.ReLU(True),
                spectral_norm(
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    use_spectral_norm,
                ),
                nn.InstanceNorm2d(128, track_running_stats=False),
                nn.ReLU(True),
                spectral_norm(
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    use_spectral_norm,
                ),
                nn.InstanceNorm2d(256, track_running_stats=False),
                nn.ReLU(True),
            )
        elif conv_type == "partial":
            self.encoder = nn.Sequential(
                nn.ReflectionPad2d(3),
                spectral_norm(
                    PartialConv2d(
                        in_channels=3, out_channels=64, kernel_size=7, padding=0
                    ),
                    use_spectral_norm,
                ),
                nn.InstanceNorm2d(64, track_running_stats=False),
                nn.ReLU(True),
                spectral_norm(
                    PartialConv2d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    use_spectral_norm,
                ),
                nn.InstanceNorm2d(128, track_running_stats=False),
                nn.ReLU(True),
                spectral_norm(
                    PartialConv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    use_spectral_norm,
                ),
                nn.InstanceNorm2d(256, track_running_stats=False),
                nn.ReLU(True),
            )
        elif conv_type == "deform":
            # without spectral_norm
            self.encoder = nn.Sequential(
                nn.ReflectionPad2d(3),
                DeformConv2d(in_nc=3, out_nc=64, kernel_size=7, padding=0),
                nn.InstanceNorm2d(64, track_running_stats=False),
                nn.ReLU(True),
                DeformConv2d(in_nc=64, out_nc=128, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(128, track_running_stats=False),
                nn.ReLU(True),
                DeformConv2d(in_nc=128, out_nc=256, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(256, track_running_stats=False),
                nn.ReLU(True),
            )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            spectral_norm(
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=128,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                use_spectral_norm,
            ),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
            spectral_norm(
                nn.ConvTranspose2d(
                    in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1
                ),
                use_spectral_norm,
            ),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x


class ResnetBlock(pl.LightningModule):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    padding=0,
                    dilation=dilation,
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            spectral_norm(
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    padding=0,
                    dilation=1,
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class EdgeConnectModel(pl.LightningModule):
    def __init__(
        self,
        residual_blocks_edge=8,
        residual_blocks_inpaint=8,
        use_spectral_norm=True,
        conv_type_edge="normal",
        conv_type_inpaint="normal",
    ):
        super().__init__()
        self.EdgeGenerator = EdgeGenerator(
            residual_blocks=residual_blocks_edge,
            use_spectral_norm=use_spectral_norm,
            conv_type=conv_type_edge,
        )
        self.InpaintGenerator = InpaintGenerator(
            residual_blocks=residual_blocks_inpaint, conv_type=conv_type_inpaint
        )

    def forward(self, images, edges, grayscale, masks):
        images = images.type(torch.cuda.FloatTensor)
        edges = edges.type(torch.cuda.FloatTensor)
        grayscale = grayscale.type(torch.cuda.FloatTensor)
        masks = masks.type(torch.cuda.FloatTensor)

        # edge
        edges_masked = edges * masks
        grayscale_masked = grayscale * masks

        inputs = torch.cat((grayscale_masked, edges_masked, masks), dim=1)
        outputs_edge = self.EdgeGenerator(
            inputs
        )  # in: [grayscale(1) + edge(1) + mask(1)]

        # inpaint
        images_masked = (images * masks).float() + (1 - masks)
        inputs = torch.cat((images_masked, outputs_edge), dim=1)
        outputs = self.InpaintGenerator(inputs)  # in: [rgb(3) + edge(1)]
        return outputs, outputs_edge
