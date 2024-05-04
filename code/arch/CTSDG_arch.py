"""
11-Okt-2021
https://github.com/Xiefan-Guo/CTSDG/blob/main/models/generator/generator.py
https://github.com/Xiefan-Guo/CTSDG/blob/main/models/generator/projection.py
https://github.com/Xiefan-Guo/CTSDG/blob/main/models/generator/pconv.py
https://github.com/Xiefan-Guo/CTSDG/blob/main/models/generator/bigff.py
https://github.com/Xiefan-Guo/CTSDG/blob/main/models/generator/cfa.py
https://github.com/Xiefan-Guo/CTSDG/blob/main/utils/misc.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# from utils.misc import weights_init
# from models.generator.cfa import CFA
# from models.generator.bigff import BiGFF
# from models.generator.pconv import PConvBNActiv
# from models.generator.projection import Feature2Structure, Feature2Texture


def extract_patches(x, kernel_size=3, stride=1):
    if kernel_size != 1:
        x = nn.ZeroPad2d(1)(x)
    x = x.permute(0, 2, 3, 1)
    x = x.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
    return x.contiguous()


class RAL(nn.Module):
    """Region affinity learning."""

    def __init__(self, kernel_size=3, stride=1, rate=2, softmax_scale=10.0):
        super(RAL, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.rate = rate
        self.softmax_scale = softmax_scale

    def forward(self, background, foreground):
        # accelerated calculation
        foreground = F.interpolate(
            foreground,
            scale_factor=1.0 / self.rate,
            mode="bilinear",
            align_corners=True,
        )

        foreground_size, background_size = list(foreground.size()), list(
            background.size()
        )

        background_kernel_size = 2 * self.rate
        background_patches = extract_patches(
            background,
            kernel_size=background_kernel_size,
            stride=self.stride * self.rate,
        )
        background_patches = background_patches.view(
            background_size[0],
            -1,
            background_size[1],
            background_kernel_size,
            background_kernel_size,
        )
        background_patches_list = torch.split(background_patches, 1, dim=0)

        foreground_list = torch.split(foreground, 1, dim=0)
        foreground_patches = extract_patches(
            foreground, kernel_size=self.kernel_size, stride=self.stride
        )
        foreground_patches = foreground_patches.view(
            foreground_size[0],
            -1,
            foreground_size[1],
            self.kernel_size,
            self.kernel_size,
        )
        foreground_patches_list = torch.split(foreground_patches, 1, dim=0)

        output_list = []
        padding = 0 if self.kernel_size == 1 else 1
        escape_NaN = torch.FloatTensor([1e-4])
        if torch.cuda.is_available():
            escape_NaN = escape_NaN.cuda()

        for foreground_item, foreground_patches_item, background_patches_item in zip(
            foreground_list, foreground_patches_list, background_patches_list
        ):
            foreground_patches_item = foreground_patches_item[0]
            foreground_patches_item_normed = foreground_patches_item / torch.max(
                torch.sqrt(
                    (foreground_patches_item * foreground_patches_item).sum(
                        [1, 2, 3], keepdim=True
                    )
                ),
                escape_NaN,
            )

            score_map = F.conv2d(
                foreground_item,
                foreground_patches_item_normed,
                stride=1,
                padding=padding,
            )
            score_map = score_map.view(
                1,
                foreground_size[2] // self.stride * foreground_size[3] // self.stride,
                foreground_size[2],
                foreground_size[3],
            )
            attention_map = F.softmax(score_map * self.softmax_scale, dim=1)
            attention_map = attention_map.clamp(min=1e-8)

            background_patches_item = background_patches_item[0]
            output_item = (
                F.conv_transpose2d(
                    attention_map, background_patches_item, stride=self.rate, padding=1
                )
                / 4.0
            )
            output_list.append(output_item)

        output = torch.cat(output_list, dim=0)
        output = output.view(background_size)
        return output


class MSFA(nn.Module):
    """Multi-scale feature aggregation."""

    def __init__(
        self, in_channels=64, out_channels=64, dilation_rate_list=[1, 2, 4, 8]
    ):
        super(MSFA, self).__init__()

        self.dilation_rate_list = dilation_rate_list

        for _, dilation_rate in enumerate(dilation_rate_list):
            self.__setattr__(
                "dilated_conv_{:d}".format(_),
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        dilation=dilation_rate,
                        padding=dilation_rate,
                    ),
                    nn.ReLU(inplace=True),
                ),
            )

        self.weight_calc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, len(dilation_rate_list), 1),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        weight_map = self.weight_calc(x)

        x_feature_list = []
        for _, dilation_rate in enumerate(self.dilation_rate_list):
            x_feature_list.append(self.__getattr__("dilated_conv_{:d}".format(_))(x))

        output = (
            weight_map[:, 0:1, :, :] * x_feature_list[0]
            + weight_map[:, 1:2, :, :] * x_feature_list[1]
            + weight_map[:, 2:3, :, :] * x_feature_list[2]
            + weight_map[:, 3:4, :, :] * x_feature_list[3]
        )

        return output


class CFA(nn.Module):
    """Contextual Feature Aggregation."""

    def __init__(
        self,
        kernel_size=3,
        stride=1,
        rate=2,
        softmax_scale=10.0,
        in_channels=64,
        out_channels=64,
        dilation_rate_list=[1, 2, 4, 8],
    ):
        super(CFA, self).__init__()

        self.ral = RAL(
            kernel_size=kernel_size,
            stride=stride,
            rate=rate,
            softmax_scale=softmax_scale,
        )
        self.msfa = MSFA(
            in_channels=in_channels,
            out_channels=out_channels,
            dilation_rate_list=dilation_rate_list,
        )

    def forward(self, background, foreground):
        output = self.ral(background, foreground)
        output = self.msfa(output)

        return output


class BiGFF(nn.Module):
    """Bi-directional Gated Feature Fusion."""

    def __init__(self, in_channels, out_channels):
        super(BiGFF, self).__init__()

        self.structure_gate = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels + in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid(),
        )
        self.texture_gate = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels + in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid(),
        )
        self.structure_gamma = nn.Parameter(torch.zeros(1))
        self.texture_gamma = nn.Parameter(torch.zeros(1))

    def forward(self, texture_feature, structure_feature):
        energy = torch.cat((texture_feature, structure_feature), dim=1)

        gate_structure_to_texture = self.structure_gate(energy)
        gate_texture_to_structure = self.texture_gate(energy)

        texture_feature = texture_feature + self.texture_gamma * (
            gate_structure_to_texture * structure_feature
        )
        structure_feature = structure_feature + self.structure_gamma * (
            gate_texture_to_structure * texture_feature
        )

        return torch.cat((texture_feature, structure_feature), dim=1)


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        # whether the mask is multi-channel or not
        if "multi_channel" in kwargs:
            self.multi_channel = kwargs["multi_channel"]
            kwargs.pop("multi_channel")
        else:
            self.multi_channel = False

        self.return_mask = True

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
            )
        else:
            self.weight_maskUpdater = torch.ones(
                1, 1, self.kernel_size[0], self.kernel_size[1]
            )

        self.slide_winsize = (
            self.weight_maskUpdater.shape[1]
            * self.weight_maskUpdater.shape[2]
            * self.weight_maskUpdater.shape[3]
        )

        self.last_size = (None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask=None):
        if mask is not None or self.last_size != (
            input.data.shape[2],
            input.data.shape[3],
        ):
            self.last_size = (input.data.shape[2], input.data.shape[3])

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(
                            input.data.shape[0],
                            input.data.shape[1],
                            input.data.shape[2],
                            input.data.shape[3],
                        ).to(input)
                    else:
                        mask = torch.ones(
                            1, 1, input.data.shape[2], input.data.shape[3]
                        ).to(input)

                self.update_mask = F.conv2d(
                    mask,
                    self.weight_maskUpdater,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=1,
                )

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        if (
            self.update_mask.type() != input.type()
            or self.mask_ratio.type() != input.type()
        ):
            self.update_mask.to(input)
            self.mask_ratio.to(input)

        raw_out = super(PartialConv2d, self).forward(
            torch.mul(input, mask) if mask is not None else input
        )

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


# --------------------------
# PConv-BatchNorm-Activation
# --------------------------
class PConvBNActiv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bn=True,
        sample="none-3",
        activ="relu",
        bias=False,
    ):
        super(PConvBNActiv, self).__init__()

        if sample == "down-7":
            self.conv = PartialConv2d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=bias,
                multi_channel=True,
            )
        elif sample == "down-5":
            self.conv = PartialConv2d(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=bias,
                multi_channel=True,
            )
        elif sample == "down-3":
            self.conv = PartialConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=bias,
                multi_channel=True,
            )
        else:
            self.conv = PartialConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
                multi_channel=True,
            )

        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

        if activ == "relu":
            self.activation = nn.ReLU()
        elif activ == "leaky":
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, images, masks):
        images, masks = self.conv(images, masks)
        if hasattr(self, "bn"):
            images = self.bn(images)
        if hasattr(self, "activation"):
            images = self.activation(images)

        return images, masks


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, inplanes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += residual
        out = self.relu(out)

        return out


class Feature2Structure(nn.Module):
    def __init__(self, inplanes=64, planes=16):
        super(Feature2Structure, self).__init__()

        self.structure_resolver = Bottleneck(inplanes, planes)
        self.out_layer = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())

    def forward(self, structure_feature):
        x = self.structure_resolver(structure_feature)
        structure = self.out_layer(x)
        return structure


class Feature2Texture(nn.Module):
    def __init__(self, inplanes=64, planes=16):
        super(Feature2Texture, self).__init__()

        self.texture_resolver = Bottleneck(inplanes, planes)
        self.out_layer = nn.Sequential(nn.Conv2d(64, 3, 1), nn.Tanh())

    def forward(self, texture_feature):
        x = self.texture_resolver(texture_feature)
        texture = self.out_layer(x)
        return texture


class Generator(nn.Module):
    def __init__(self, image_in_channels=3, edge_in_channels=2, out_channels=3):
        super(Generator, self).__init__()

        self.freeze_ec_bn = False

        # -----------------------
        # texture encoder-decoder
        # -----------------------
        self.ec_texture_1 = PConvBNActiv(
            image_in_channels, 64, bn=False, sample="down-7"
        )
        self.ec_texture_2 = PConvBNActiv(64, 128, sample="down-5")
        self.ec_texture_3 = PConvBNActiv(128, 256, sample="down-5")
        self.ec_texture_4 = PConvBNActiv(256, 512, sample="down-3")
        self.ec_texture_5 = PConvBNActiv(512, 512, sample="down-3")
        self.ec_texture_6 = PConvBNActiv(512, 512, sample="down-3")
        self.ec_texture_7 = PConvBNActiv(512, 512, sample="down-3")

        self.dc_texture_7 = PConvBNActiv(512 + 512, 512, activ="leaky")
        self.dc_texture_6 = PConvBNActiv(512 + 512, 512, activ="leaky")
        self.dc_texture_5 = PConvBNActiv(512 + 512, 512, activ="leaky")
        self.dc_texture_4 = PConvBNActiv(512 + 256, 256, activ="leaky")
        self.dc_texture_3 = PConvBNActiv(256 + 128, 128, activ="leaky")
        self.dc_texture_2 = PConvBNActiv(128 + 64, 64, activ="leaky")
        self.dc_texture_1 = PConvBNActiv(64 + out_channels, 64, activ="leaky")

        # -------------------------
        # structure encoder-decoder
        # -------------------------
        self.ec_structure_1 = PConvBNActiv(
            edge_in_channels, 64, bn=False, sample="down-7"
        )
        self.ec_structure_2 = PConvBNActiv(64, 128, sample="down-5")
        self.ec_structure_3 = PConvBNActiv(128, 256, sample="down-5")
        self.ec_structure_4 = PConvBNActiv(256, 512, sample="down-3")
        self.ec_structure_5 = PConvBNActiv(512, 512, sample="down-3")
        self.ec_structure_6 = PConvBNActiv(512, 512, sample="down-3")
        self.ec_structure_7 = PConvBNActiv(512, 512, sample="down-3")

        self.dc_structure_7 = PConvBNActiv(512 + 512, 512, activ="leaky")
        self.dc_structure_6 = PConvBNActiv(512 + 512, 512, activ="leaky")
        self.dc_structure_5 = PConvBNActiv(512 + 512, 512, activ="leaky")
        self.dc_structure_4 = PConvBNActiv(512 + 256, 256, activ="leaky")
        self.dc_structure_3 = PConvBNActiv(256 + 128, 128, activ="leaky")
        self.dc_structure_2 = PConvBNActiv(128 + 64, 64, activ="leaky")
        self.dc_structure_1 = PConvBNActiv(64 + 2, 64, activ="leaky")
        # self.dc_structure_1 = PConvBNActiv(65, 64, activ='leaky')

        # -------------------
        # Projection Function
        # -------------------
        self.structure_feature_projection = Feature2Structure()
        self.texture_feature_projection = Feature2Texture()

        # -----------------------------------
        # Bi-directional Gated Feature Fusion
        # -----------------------------------
        self.bigff = BiGFF(in_channels=64, out_channels=64)

        # ------------------------------
        # Contextual Feature Aggregation
        # ------------------------------
        self.fusion_layer1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.cfa = CFA(in_channels=64, out_channels=64)
        self.fusion_layer2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.out_layer = nn.Sequential(
            nn.Conv2d(64 + 64 + 64, 3, kernel_size=1), nn.Tanh()
        )

        # if init_weights:
        #   self.apply(weights_init())

    def forward(self, input_image, input_edge, mask):
        input_edge = torch.cat((input_edge, input_edge), dim=1)

        ec_textures = {}
        ec_structures = {}

        input_texture_mask = torch.cat((mask, mask, mask), dim=1)
        ec_textures["ec_t_0"], ec_textures["ec_t_masks_0"] = (
            input_image,
            input_texture_mask,
        )
        ec_textures["ec_t_1"], ec_textures["ec_t_masks_1"] = self.ec_texture_1(
            ec_textures["ec_t_0"], ec_textures["ec_t_masks_0"]
        )
        ec_textures["ec_t_2"], ec_textures["ec_t_masks_2"] = self.ec_texture_2(
            ec_textures["ec_t_1"], ec_textures["ec_t_masks_1"]
        )
        ec_textures["ec_t_3"], ec_textures["ec_t_masks_3"] = self.ec_texture_3(
            ec_textures["ec_t_2"], ec_textures["ec_t_masks_2"]
        )
        ec_textures["ec_t_4"], ec_textures["ec_t_masks_4"] = self.ec_texture_4(
            ec_textures["ec_t_3"], ec_textures["ec_t_masks_3"]
        )
        ec_textures["ec_t_5"], ec_textures["ec_t_masks_5"] = self.ec_texture_5(
            ec_textures["ec_t_4"], ec_textures["ec_t_masks_4"]
        )
        ec_textures["ec_t_6"], ec_textures["ec_t_masks_6"] = self.ec_texture_6(
            ec_textures["ec_t_5"], ec_textures["ec_t_masks_5"]
        )
        ec_textures["ec_t_7"], ec_textures["ec_t_masks_7"] = self.ec_texture_7(
            ec_textures["ec_t_6"], ec_textures["ec_t_masks_6"]
        )

        input_structure_mask = torch.cat((mask, mask), dim=1)
        ec_structures["ec_s_0"], ec_structures["ec_s_masks_0"] = (
            input_edge,
            input_structure_mask,
        )
        ec_structures["ec_s_1"], ec_structures["ec_s_masks_1"] = self.ec_structure_1(
            ec_structures["ec_s_0"], ec_structures["ec_s_masks_0"]
        )
        ec_structures["ec_s_2"], ec_structures["ec_s_masks_2"] = self.ec_structure_2(
            ec_structures["ec_s_1"], ec_structures["ec_s_masks_1"]
        )
        ec_structures["ec_s_3"], ec_structures["ec_s_masks_3"] = self.ec_structure_3(
            ec_structures["ec_s_2"], ec_structures["ec_s_masks_2"]
        )
        ec_structures["ec_s_4"], ec_structures["ec_s_masks_4"] = self.ec_structure_4(
            ec_structures["ec_s_3"], ec_structures["ec_s_masks_3"]
        )
        ec_structures["ec_s_5"], ec_structures["ec_s_masks_5"] = self.ec_structure_5(
            ec_structures["ec_s_4"], ec_structures["ec_s_masks_4"]
        )
        ec_structures["ec_s_6"], ec_structures["ec_s_masks_6"] = self.ec_structure_6(
            ec_structures["ec_s_5"], ec_structures["ec_s_masks_5"]
        )
        ec_structures["ec_s_7"], ec_structures["ec_s_masks_7"] = self.ec_structure_7(
            ec_structures["ec_s_6"], ec_structures["ec_s_masks_6"]
        )

        dc_texture, dc_tecture_mask = (
            ec_structures["ec_s_7"],
            ec_structures["ec_s_masks_7"],
        )
        for _ in range(7, 0, -1):
            ec_texture_skip = "ec_t_{:d}".format(_ - 1)
            ec_texture_masks_skip = "ec_t_masks_{:d}".format(_ - 1)
            dc_conv = "dc_texture_{:d}".format(_)

            dc_texture = F.interpolate(dc_texture, scale_factor=2, mode="bilinear")
            dc_tecture_mask = F.interpolate(
                dc_tecture_mask, scale_factor=2, mode="nearest"
            )

            dc_texture = torch.cat((dc_texture, ec_textures[ec_texture_skip]), dim=1)
            dc_tecture_mask = torch.cat(
                (dc_tecture_mask, ec_textures[ec_texture_masks_skip]), dim=1
            )

            dc_texture, dc_tecture_mask = getattr(self, dc_conv)(
                dc_texture, dc_tecture_mask
            )

        dc_structure, dc_structure_masks = (
            ec_textures["ec_t_7"],
            ec_textures["ec_t_masks_7"],
        )
        for _ in range(7, 0, -1):
            ec_structure_skip = "ec_s_{:d}".format(_ - 1)
            ec_structure_masks_skip = "ec_s_masks_{:d}".format(_ - 1)
            dc_conv = "dc_structure_{:d}".format(_)

            dc_structure = F.interpolate(dc_structure, scale_factor=2, mode="bilinear")
            dc_structure_masks = F.interpolate(
                dc_structure_masks, scale_factor=2, mode="nearest"
            )
            dc_structure = torch.cat(
                (dc_structure, ec_structures[ec_structure_skip]), dim=1
            )

            dc_structure_masks = torch.cat(
                (dc_structure_masks, ec_structures[ec_structure_masks_skip]), dim=1
            )

            dc_structure, dc_structure_masks = getattr(self, dc_conv)(
                dc_structure, dc_structure_masks
            )

        # -------------------
        # Projection Function
        # -------------------
        projected_image = self.texture_feature_projection(dc_texture)
        projected_edge = self.structure_feature_projection(dc_structure)

        output_bigff = self.bigff(dc_texture, dc_structure)

        output = self.fusion_layer1(output_bigff)
        output_atten = self.cfa(output, output)
        output = self.fusion_layer2(torch.cat((output, output_atten), dim=1))
        output = F.interpolate(output, scale_factor=2, mode="bilinear")
        output = self.out_layer(torch.cat((output, output_bigff), dim=1))

        return output, projected_image, projected_edge
