#ダミーデータだよ。
import numpy as np
import matplotlib.pyplot as plt

from std_score_visualize import extract_sun_mini
from std_score_visualize import calculate_hensachi

# ファイルの上のほうに追加！
INPUT_DIR = "./samples"  # 入力フォルダのパス
CROP_H = 800             # 切り取る高さ（数値）
CROP_W = 800             # 切り取る幅（数値）

# ダミーの偏差値画像（10フレーム、5×5ピクセル）
#hensachi = np.random.normal(
#    loc=50,      # 平均50
#    scale=10,    # 標準偏差10
#   size=(10, 5, 5)
#)

#y = 2
#x = 3

#pixel_values = hensachi[:, y, x]

#print(pixel_values)

#pixel_values = hensachi[:, y, x]

# フレーム番号
#frames = np.arange(len(pixel_values))

# グラフ作成
#plt.figure(figsize=(10, 4))
#plt.plot(frames, pixel_values, marker="o")
#plt.xlabel("Frame")
#plt.ylabel("Hensachi")
#plt.title(f"Hensachi of Pixel ({x}, {y})")
#plt.grid(True)
#plt.savefig("pixel_graph.png", dpi=300)
#plt.show()


#本データだよ。
#NumPyとグラフを描くためのMatplotlibを読み込む
import numpy as np
import matplotlib.pyplot as plt

from std_score_visualize import extract_sun_mini
from std_score_visualize import calculate_hensachi

frames, centers = extract_sun_mini(
    INPUT_DIR,
    h_size=CROP_H,
    w_size=CROP_W
)

mean, std, hensachi = calculate_hensachi(frames)

# 見たいピクセル座標
y = 350
x = 420

# 全フレームのそのピクセルの偏差値を取り出す
pixel_values = hensachi[:, y, x]

# フレーム番号
frames = np.arange(len(pixel_values))

# グラフ作成
plt.figure(figsize=(10, 4))
plt.plot(frames, pixel_values, marker="o")
plt.xlabel("Frame")
plt.ylabel("Hensachi")
plt.title(f"Hensachi of Pixel ({x}, {y})")
plt.grid(True)
plt.savefig("pixel_graph.png", dpi=300)
plt.show()