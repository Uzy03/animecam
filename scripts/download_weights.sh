#!/usr/bin/env bash
set -e
mkdir -p weights && cd weights

# 好きなスタイルの重みを落とす
curl -L -o hayao.pth   https://huggingface.co/arpan/v2a/resolve/main/generator_hayao.pth
curl -L -o shinkai.pth https://huggingface.co/akhaliq/AnimeGANv2/resolve/main/generator_shinkai.pth
curl -L -o paprika.pth https://huggingface.co/akhaliq/AnimeGANv2/resolve/main/generator_paprika.pth
