#!/usr/bin/env bash
set -e

INSTALL_DIR=/home/vscode/animegan2

git clone --depth 1 https://github.com/bryandlee/animegan2-pytorch.git "$INSTALL_DIR"
[[ -f "${INSTALL_DIR}/requirements.txt" ]] && pip install -r "${INSTALL_DIR}/requirements.txt"

# ここだけ書き換え
echo "export PYTHONPATH=\$PYTHONPATH:${INSTALL_DIR}" >> /home/vscode/.bashrc
