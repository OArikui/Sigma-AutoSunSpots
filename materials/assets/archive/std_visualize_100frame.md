# std_visualize_100frame

2026.07.31

src:[Commit 1db0d32:"fix: 出力先ディレクトリのパスを修正し、標準偏差画像の保存処理を更新"](https://github.com/fuji-astronomy-club-coders/sich_sdt_score_visualize/commit/1db0d32d8465e4c1da5410afce346aab5e78472f)

```python
import os
import cv2
import sys
import json
import numpy as np
from lib.lib_frommain import (
    normalize_image,
    extract_sun_mini,
    create_colormap,
    statistics_image,
)
from lib.ffmpeg_video import compress_analysis_frames

# パラメータ宣言
DEBUG = True  # True: デバッグ情報を表示
INPUT_DIR = ""  # 処理対象の画像フォルダ
CROP_H = 800  # 抽出する画像サイズ(縦幅)
CROP_W = 800  # 抽出する画像サイズ(横幅)
OUTPUT_DIR = ""  # 出力動画の保存先フォルダ
OUTPUT_NAME = ""  # 出力動画のファイル名(拡張子なし)
OUTPUT_EXT = ""  # 出力動画の拡張子
FPS = 1  # 出力動画のフレームレート

STD_100f_OUTPUT_DIR = ""

if os.environ.get("RUN_BY_SUBPROCESS") == "true":
    print("このスクリプトは subprocess から実行されています。")
    # 標準入力から流れてきた文字列を一括で読み込む
    input_data = sys.stdin.read()

    # JSON文字列をPythonの辞書オブジェクトに復元
    locals().update(json.loads(input_data))

frames, _ = extract_sun_mini(INPUT_DIR, h_size=CROP_H, w_size=CROP_W)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# フレーム数・画像サイズの取得
n_frames, height, width = frames.shape

# 100フレームごとの標準偏差画像を作成
std_images = []

st_ed=[]
for start in range(0, n_frames, 100):

    # 終了位置
    end = min(start + 100, n_frames)

    # 最後は100枚未満でもそのまま処理
    section = frames[start:end]

    st_ed.append((start,end))
    if DEBUG:
        print(f"{start}～{end-1} frame", "shape:", section.shape)

    # 100フレーム分の各ピクセルの標準偏差を計算
    std_image = np.std(section, axis=0)

    std_images.append(std_image)

# numpy配列化
std_images = np.array(std_images)
std_frames, height, width = std_images.shape

# 標準偏差画像の確認(デバッグ表示)
if DEBUG:
    print("===入力フレーム情報===")
    print("データサイズ:", frames.shape)
    print("最小値:", frames.min())
    print("最大値:", frames.max())
    print("=====================")

    print("\n=====標準偏差画像=====")
    print("画像枚数:", std_frames)
    print("画像サイズ:", (height, width))
    print("=====================")

# カラーマップ作成
colormap_lut = create_colormap()

# =====================動画作成用====================
# 1フレームずつ取り出し、LUTを適用して保存用配列へ追加
for i,frame in enumerate(std_images):
    start,end=st_ed[i]
    statistics_image(frame, os.path.join(OUTPUT_DIR, OUTPUT_NAME+f"f{start}-f{end}" + OUTPUT_EXT))
print("画像の保存が完了しました")
```
