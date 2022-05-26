import torch

# from this model
model1 = torch.load("team04_rlfn.pth")
# into this model
model2 = torch.load("Checkpoint_0_0_G.pth")

for k in model1.keys():
    try:
        if 'upsampler.0' in k:
            continue

        print(f"setting {k}")
        model2[k] = model1[k]

    except Exception as e:
        print(e)
        pass

torch.save(model2, "fixed.pth")
print("done")