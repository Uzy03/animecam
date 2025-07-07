import torch, pathlib

PATH = pathlib.Path("weights/hayao.pth")   # ← 試したいファイル

ckpt = torch.load(PATH, map_location="cpu")

# チェックポイント形式かどうか分岐
if "model_state_dict" in ckpt:
    sd = ckpt["model_state_dict"]
else:
    sd = ckpt

print("1st key in state-dict:", next(iter(sd)))
