import torch

path = "70000.ckpt"

checkpoint = torch.load(path)

new_model_G = {}
new_model_D = {}

for i, j in checkpoint["state_dict"].items():
    print(i)

    if "netG." in i:
        key = i.replace("netG.", "")
        new_model_G[key] = j

    if "netD." in i:
        key = i.replace("netD.", "")
        new_model_D[key] = j

torch.save(new_model_G, "G.pth")
torch.save(new_model_D, "D.pth")
