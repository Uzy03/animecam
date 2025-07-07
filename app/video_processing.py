#!/usr/bin/env python3
"""Streamlit web‑app: real‑time AnimeGAN‑v2 camera filter

Run inside the repo root:
    streamlit run app/video_processing.py  # opens http://localhost:8501

The app uses **streamlit‑webrtc** to grab webcam frames in the browser, sends
them to the server, runs AnimeGAN‑v2, and streams the stylised video back.
"""
import sys
import time
from pathlib import Path
from typing import Union

import av                       # type: ignore  # provided by streamlit-webrtc
import cv2                      # OpenCV <=4.xx compiled with FFMPEG
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
import torch

# ─────────────────────────────── paths & model ────────────────────────────────
A2_DIR = (Path(__file__).resolve().parent.parent / "extern" / "AnimeGANv2").resolve()
sys.path.insert(0, str(A2_DIR))
from model import Generator  # noqa: E402 after sys.path insert

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STYLE_PATHS = {
    "Paprika":        A2_DIR / "weights" / "paprika.pt",
    "FacePaint-v1":   A2_DIR / "weights" / "face_paint_512_v1.pt",
    "FacePaint-v2":   A2_DIR / "weights" / "face_paint_512_v2.pt",
    "Celeba-Distill": A2_DIR / "weights" / "celeba_distill.pt",
}


@st.cache_resource(show_spinner="Loading generator …")
def load_generator(style: str) -> Generator:
    g = Generator().to(DEVICE)
    sd = torch.load(STYLE_PATHS[style], map_location=DEVICE, weights_only=False)
    g.load_state_dict(sd.get("model_state_dict", sd), strict=True)
    g.eval()
    return g


# ───────────────────────────── frame processor ────────────────────────────────
class AnimeGANProcessor(VideoProcessorBase):
    """streamlit‑webrtc callback that stylises each incoming frame."""

    def __init__(self, style: str, scale: float = 1.0):
        self.style = style
        self.scale = scale
        self.g = load_generator(style)

    def _stylise(self, bgr: np.ndarray) -> np.ndarray:
        if self.scale != 1.0:
            bgr = cv2.resize(bgr, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

        rgb = np.ascontiguousarray(bgr[..., ::-1])  # BGR→RGB + positive stride
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
        with torch.no_grad():
            out = self.g(t)[0].clamp_(0, 1)
        out_bgr = (out.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")[..., ::-1]
        if self.scale != 1.0:
            out_bgr = cv2.resize(out_bgr, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
        return out_bgr

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        stylised = self._stylise(img)
        return av.VideoFrame.from_ndarray(stylised, format="bgr24")


# ──────────────────────────────── Streamlit UI ────────────────────────────────
st.set_page_config(page_title="AnimeGAN‑v2 Webcam", page_icon="🎥", layout="centered")
st.title("📸 AnimeGAN‑v2 Webcam Filter")

with st.sidebar:
    style_choice = st.selectbox("Style", list(STYLE_PATHS.keys()), index=0)
    scale = st.slider("Downscale before inference (speed)", 0.3, 1.0, 0.75, 0.05)

st.write("Allow webcam access in the dialog that appears 👆.")

webrtc_streamer(
    key="anime-gan",
    video_processor_factory=lambda: AnimeGANProcessor(style_choice, scale),
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)
