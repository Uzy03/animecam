# AnimeCam 🎨 — Real-time AnimeGAN2 Webcam Filter

<img src="https://i.imgur.com/j8fcyb1.gif" width="640" alt="demo gif"/>

リアルタイムで Web カメラ映像をアニメ風に変換するデモです。  
熊本大学 情報融合学館 オープンキャンパス展示用。

---

## ✨ Features

* **AnimeGAN2 (PyTorch)** を利用した 3 種スタイル変換（Hayao / Shinkai / Paprika）
* **Streamlit** + **streamlit-webrtc** でブラウザ上に即時プレビュー
* **Dev Container 完備** — VS Code で “Reopen in Container” を押すだけで環境再現
* Apple Silicon (M1/M2) の **MPS**、あるいは CUDA GPU に自動対応
* 重み `.pth` は初回ビルド時に自動ダウンロード

---

## ⚡ Quick Start （推奨：VS Code Dev Container）

0. **前提ツール**  
   * Docker Desktop ≥ **24**  
   * VS Code + Extension **Dev Containers (ms-vscode-remote.remote-containers)**

1. リポジトリを clone  

   ```bash
   git clone https://github.com/Uzy03/animecam.git
   ```
   
2. Dev Container を起動（macOS : ⌘ ⇧ P  /  Windows/Linux : Ctrl + Shift + Pを押して）、Command Palette を開き、Dev Containers: Open Folder in Container… でanimecamを選択して実行（初回のみ 5〜10 分程度でビルドが走ります）

3. コンテナ内ターミナルでデモを実行
   ```bash
   streamlit run app/quick_demo.py --server.port 8501
   ```

5. ブラウザ http://localhost:8501 を開き、カメラ使用を許可 →　サイドバーからスタイルを選んで撮影！

6. 同じように，リアルタイム動画版を実行
   ```bash
   streamlit run app/video_processing.py --server.port 8501
   ```


## 🐍 Local Python 環境で動かす場合
   M1/M2 など GPU が無い環境では 256×256 / 5 FPS 程度です。
   仮想環境（venv or conda）を推奨。

   ```bash
   # Python 3.10 以上を想定
   python -m venv .venv
   source .venv/bin/activate        # Windows は .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt

   # 初回のみ重みを取得
   bash scripts/download_weights.sh

   streamlit run app/quick_demo.py
   ```

## 💡 How It Works
```text
┌────────┐   webcam   ┌─────────────┐   Torch   ┌─────────┐
│ browser│ ─────────▶ │ streamlit-  │ ────────▶ │AnimeGAN2│
│  (JS)  │   WebRTC   │   webrtc    │  tensor   │Generator│
└────────┘            └─────────────┘           └─────────┘
```


  ・streamlit-webrtc がブラウザと WebRTC で双方向通信

  ・受け取った BGR フレームを AnimeGAN2 Generator へ

  ・256×256 変換結果をブラウザへ返し <video> に描画

## 📂 Repository Structure
```text
.
├── app/                # Streamlit アプリ本体
├── scripts/            # 環境セットアップ & 重みDL
├── weights/            # *.pth (git ignore 済)
├── .devcontainer/      # Dockerfile & 設定
└── requirements.txt
```

## 📜 License
   MIT © 2025 Your Name
   ```yaml
   
---

### README に載せる情報の考え方

| セクション | 目的 |
|------------|------|
| **Quick Start** | “10 秒で動く” 手順を最上位に置き、読む人のハードルを下げる |
| **Local Install** | Dev Container が使えない読者向けの代替手順 |
| **Repository Structure / How It Works** | ソースを追う人が迷わないよう道標を置く |
| **License / Contributing** | 大学プロジェクトでも OSS として体裁を整えておく |

これで共同開発者がクローンしても迷わず立ち上げられます。  
適宜プロジェクト名や著者を変えてご利用ください！
