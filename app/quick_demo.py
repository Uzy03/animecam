import streamlit as st, cv2, torch, numpy as np
import sys
sys.path.append("/home/vscode/animegan2")   # INSTALL_DIR と同じに

from model import Generator                 # ← AnimeGAN2 repo 内のファイル名
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else (
         "mps"  if torch.backends.mps.is_available() else "cpu")

STYLE_PATHS = {
    "Hayao":   "weights/hayao.pth",
    "Shinkai": "weights/shinkai.pth",
    "Paprika": "weights/paprika.pth"
}

@st.cache_resource
def load_generator(style):
    g = Generator().to(DEVICE)
    g.load_state_dict(torch.load(STYLE_PATHS[style], map_location=DEVICE))
    g.eval()
    return g

def to_anime(bgr, gen):
    img = cv2.resize(bgr, (256, 256))
    t   = torch.from_numpy(img[..., ::-1]).permute(2, 0, 1).unsqueeze(0).float() / 255
    with torch.no_grad():
        out = gen(t.to(DEVICE))[0].permute(1, 2, 0).cpu().numpy() * 255
    return out[..., ::-1].astype("uint8")

st.title("AnimeCam 🎨")
style = st.sidebar.selectbox("Style", list(STYLE_PATHS.keys()))

frame = st.camera_input("Take a snapshot")
if frame:
    img = cv2.imdecode(np.frombuffer(frame.getvalue(), np.uint8), 1)
    st.image(to_anime(img, load_generator(style)), caption=style, channels="BGR")
