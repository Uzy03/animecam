#!/usr/bin/env bash
set -euo pipefail

# ──────────────── パス設定 ────────────────
MAIN_DIR="extern/AnimeGANv2"          # リポジトリを置く場所
WEIGHTS_DIR="${MAIN_DIR}/weights"
# ──────────────────────────────────────────

# 1) git-lfs が無ければ導入
if ! command -v git-lfs &>/dev/null; then
  echo "[INFO] installing git-lfs ..."
  sudo apt-get update -qq && sudo apt-get install -y git-lfs
  git lfs install --skip-repo
fi

# 2) 最新 AnimeGANv2 を shallow-clone（約 36 MB）
echo "[INFO] cloning AnimeGANv2 (latest) ..."
rm -rf "${MAIN_DIR}"
git clone --depth=1 https://github.com/bryandlee/animegan2-pytorch.git "${MAIN_DIR}"

# 3) 必要な 4 スタイルだけ LFS pull
echo "[INFO] pulling weights: paprika / face_paint_v1 / v2 / celeba_distill ..."
git -C "${MAIN_DIR}" lfs pull --include="weights/{paprika.pt,face_paint_512_v*.pt,celeba_distill.pt}"

# 4) サマリー表示
echo "[INFO] final file list:"
ls -lh "${WEIGHTS_DIR}"/*.pt
echo "✅  Done. Hayao / Shinkai は取得していません。"
