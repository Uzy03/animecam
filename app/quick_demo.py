import streamlit as st, cv2, torch, numpy as np
# --- app/quick_demo.py の先頭付近を修正 -------------------------------
import sys, pathlib

A2_DIR = (pathlib.Path(__file__).parent / ".." / "extern" / "AnimeGANv2").resolve()

# 先頭に挿入すると必ずここが import 対象になる
if str(A2_DIR) not in sys.path:
    sys.path.insert(0, str(A2_DIR))

from model import Generator                # ← extern/AnimeGANv2/model.py が入る
# --------------------------------------------------------------------

import inspect
print("★ Generator:", inspect.getfile(Generator))

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

    ckpt = torch.load(
    STYLE_PATHS[style],
    map_location=DEVICE,
    weights_only=False   # ← これを追加
)

    # チェックポイント or 純 state_dict どちらでも動くようにしておく
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    g.load_state_dict(state_dict, strict=True)   # strict=False なら余分キーは無視
    g.eval()
    return g
#　変更日7/4
"""
def load_generator(style):
    g = Generator().to(DEVICE)
    g.load_state_dict(torch.load(STYLE_PATHS[style], map_location=DEVICE))
    g.eval()
    return g
"""

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