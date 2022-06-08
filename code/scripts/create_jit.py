from arch.hat_arch import HAT
import torch

model = HAT(
    upscale=4,
    img_size=64,
    patch_size=1,
    in_chans=3,
    embed_dim=180,
    depths=[6, 6, 6, 6],
    num_heads=[6, 6, 6, 6],
    window_size=16,
    compress_ratio=3,
    squeeze_factor=30,
    conv_scale=0.01,
    overlap_ratio=0.5,
    mlp_ratio=2.0,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.1,
    ape=False,
    patch_norm=True,
    use_checkpoint=False,
    img_range=1.0,
    upsampler="pixelshuffle",
    resi_connection="1conv",
    conv="fft",
)
model.load_state_dict(torch.load("G.pth", map_location="cpu"))
model = model.eval()

input = torch.rand(1, 3, 64, 64)
compiled_model = torch.jit.trace(model, input)

torch.jit.save(compiled_model, "compiled_model.pt")
