import streamlit as st, cv2, torch, numpy as np
import sys
sys.path.append("/home/vscode/animegan2")   # INSTALL_DIR と同じに

from model import Generator                 # ← AnimeGAN2 repo 内のファイル名
from pathlib import Path

A2_DIR = (Path(__file__).resolve().parent.parent / "extern" / "AnimeGANv2").resolve()

DEVICE = "cuda" if torch.cuda.is_available() else (
         "mps"  if torch.backends.mps.is_available() else "cpu")

STYLE_PATHS = {
    "Paprika":        A2_DIR / "weights" / "paprika.pt",
    "FacePaint-v1":   A2_DIR / "weights" / "face_paint_512_v1.pt",
    "FacePaint-v2":   A2_DIR / "weights" / "face_paint_512_v2.pt",
    "Celeba-Distill": A2_DIR / "weights" / "celeba_distill.pt",
}

@st.cache_resource
def load_generator(style):
    g = Generator().to(DEVICE)
    g.load_state_dict(torch.load(STYLE_PATHS[style], map_location=DEVICE))
    g.eval()
    return g

def to_anime(img_bgr, g):
    """OpenCV(BGR) → Anime → BGR"""
    # ① BGR→RGB ＋ contiguous にする
    rgb = img_bgr[..., ::-1].copy()            # または np.ascontiguousarray(img_bgr[..., ::-1])

    # ② 0-1 Tensor へ
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0

    # ③ 推論
    with torch.no_grad():
        out = g(t)[0].clamp_(0, 1)             # (3, H, W)

    # ④ RGB→BGR に戻して uint8
    out_bgr = (out.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")[..., ::-1]
    return out_bgr

st.title("AnimeCam 🎨")
style = st.sidebar.selectbox("Style", list(STYLE_PATHS.keys()))

frame = st.camera_input("Take a snapshot")
if frame:
    img = cv2.imdecode(np.frombuffer(frame.getvalue(), np.uint8), 1)
    st.image(to_anime(img, load_generator(style)), caption=style, channels="BGR")