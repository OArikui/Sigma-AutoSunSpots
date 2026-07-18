import os
import cv2
import tqdm
import sys
import numpy as np
import glob
from MIN2ver2 import MIN2_ignore_sunspots

from samples.zip_operator import get_image_names_from_zip, load_image_from_zip_cv2
zip_path = "samples/2025-07-20-PL1.zip"
image_names = get_image_names_from_zip(zip_path)
frames = []
for name in image_names:
    img = load_image_from_zip_cv2(zip_path, name)
    frames.append(img)
frames = np.array(frames)


#パラメータ std score
INPUT_DIR = "./sun_images"   #処理対象の画像フォルダ
CROP_H = 800           #　抽出する画像サイズ(縦幅)
CROP_W = 800           #　抽出する画像サイズ(横幅)
OUT_DIR = "./output_pixels"  #  CSV保存先フォルダ

#パラメータ colormap
USE_CSV = True      # True: CSVを読み込む / False: 偏差値算出チームの変数を使用
DEBUG = True        # True: デバッグ情報を表示
OUTPUT_DIR = (
    r"C:\Users\2025005585\Desktop\python"
)                                     # 出力動画の保存先フォルダ
OUTPUT_NAME = "output_test_sample1"   # 出力動画のファイル名（拡張子なし）
OUTPUT_EXT = ".mp4"                   # 出力動画の拡張子
VIDEO_CODEC = "mp4v"                # 動画コーデック
FPS = 60.0                            # 出力動画のフレームレート
# カラーマップ作成用
def create_colormap():
    # 256要素を持つLUTを作成(偏差値0~100に対応する色を設定)
    lut = np.zeros((256,1,3),dtype=np.uint8)
    # 偏差値50未満は濃い青から白へ段階的に変化
    lut[0:5] = [100,0,0]
    lut[5:10] = [115,25,25]
    lut[10:15] = [130,50,50]
    lut[15:20] = [145,75,75]
    lut[20:25] = [160,100,100]
    lut[25:30] = [175,125,125]
    lut[30:35] = [190,150,150]
    lut[35:40] = [205,175,175]
    lut[40:45] = [220,200,200]
    lut[45:50] = [235,225,225]
    # 偏差値50以上は白から濃い赤へ段階的に変化
    lut[50:55] = [255,255,255]
    lut[55:60] = [225,225,240]
    lut[60:65] = [195,195,225]
    lut[65:70] = [165,165,210]
    lut[70:75] = [135,135,195]
    lut[75:80] = [105,105,180]
    lut[80:85] = [75,75,165]
    lut[85:90] = [45,45,150]
    lut[90:95] = [15,15,135]
    lut[95:256] = [0,0,120]
    return lut

def extract_sun_mini(zip_path:str, h_size:int,w_size:int) -> np.ndarray:
    """フォルダ内の太陽画像から太陽重心を算出し、指定サイズで切りぬいた画像配列を返します。
    画面端にかかる場合は、足りない部分を黒く塗りつぶします。

    Args:
        folder(str):対象の画像が保存されているフォルダのパス
        h_size(int):切りぬく長方形の縦幅
        w_size(int):切りぬく長方形の横幅

    Returns:
        np.ndarray:切りぬかれた画像の3次元配列（N,h_size,w_size)
    """
    print(f"---画像の読み込みと切り抜き処理を開始:{zip_path}---")
    # 画像ファイルのみ1000枚取得
    image_names = get_image_names_from_zip(zip_path)
    frames = []
    half_h = h_size//2
    half_w = w_size//2
    #tqdmによる進捗表示
    for name in tqdm.tqdm(image_names, desc="Processing images"):
        #16bit(下位12bit)画像を輝度値(1ch)のまま正しく読み込む
        img = load_image_from_zip_cv2(zip_path, name)
        if img is None: 
            continue
        try:
            cx, cy, r = MIN2_ignore_sunspots(
                img,
                show=False,
                debug=False
            )
        except Exception:
            continue

        cx = int(cx)
        cy = int(cy)
        
        #切り抜きたい理想の範囲（画面外にはみ出す可能性あり）
        h,w = img.shape
        y1,y2 =  cy - half_h, cy + half_h
        x1,x2 = cx - half_w,cx + half_w

        #画面外にはみ出している量（余白の計算）
        top =max(0,-y1)
        bottom =max(0,y2 - h)
        left = max(0,-x1)
        right =max(0,x2 - w)

        #画面内に収まる安全な範囲だけでまずは切りぬく
        crop_y1,crop_y2 = max(0,y1),min(h,y2)
        crop_x1,crop_x2 = max(0,x1),min(w,x2)
        cropped = img[crop_y1:crop_y2,crop_x1:crop_x2]

        #はみ出していた部分を黒色（0）で埋めて、常にsize x size にする 
        padded = cv2.copyMakeBorder(cropped,top,bottom,left,right,cv2.BORDER_CONSTANT,value = 0)

        frames.append(padded)
        
    return np.array(frames)

def calculate_hensachi(frames: np.ndarray):
    """平均画像・標準偏差画像・偏差値画像を計算する。"""

    # 平均画像
    mean = np.mean(frames, axis=0)

    # 標準偏差画像
    std = np.std(frames, axis=0)

    # 偏差値画像
    hensachi = np.where(
        std == 0,
        50,
        50 + 10 * (frames - mean) / std
    )

    return mean, std, hensachi

"""
#偏差値画像を1枚ずつ表示する。
for i in range(len(hensachi)):
    print(f"{i+1}枚目の偏差値画像")
    print(hensachi[i])
"""

# --- 実行とCSV保存（1フレームずつピクセル保存） ---
if __name__ == "__main__":

    # コマンドライン引数からZIPのパスを取得（未指定ならデフォルト）
    if len(sys.argv) > 1:
        target_zip = sys.argv[1]
    else:
        target_zip = "samples/2025-07-20-PL1.zip"

    # 保存先フォルダの作成（タイポ os.makediirs を修正）
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # ZIPを展開せずに一時フォルダを使って安全に読み込む
    print(f"\n--- ZIPファイルの読み込み開始: {target_zip} ---")

    frames = extract_sun_mini(
        target_zip,
        h_size=CROP_H,
        w_size=CROP_W
    )
    mean, std, hensachi = calculate_hensachi(frames)
    
    # 1フレームごと、全ピクセルをCSVに保存
    if len(frames) > 0:
        print(f"¥n--- CSV保存処理を開始:{OUT_DIR}---")
        for i, frame in enumerate(tqdm.tqdm(frames,desc="Saving CSVs")):
                np.savetxt(f"{OUT_DIR}/frame_{i+1:03d}.csv", frame, delimiter=",", fmt="%d")

                # 偏差値画像をCSVとして保存
        print("\n--- 偏差値CSV保存処理を開始 ---")

        HENSACHI_DIR = "./output_hensachi"
        os.makedirs(HENSACHI_DIR, exist_ok=True)

        for i, frame in enumerate(tqdm.tqdm(hensachi, desc="Saving Hensachi CSVs")):
            np.savetxt(
                f"{HENSACHI_DIR}/hensachi_{i+1:03d}.csv",
                frame,
                delimiter=",",
                fmt="%.2f"
            )

            print("偏差値CSVの保存が完了しました。")
            
            # 1ピクセルずつの個別アクセス（例：1枚目の座標x=10, y=20の明るさ）
            print("\n--- サンプルピクセルの確認 ---")
            print(f"個別ピクセル明るさ: {frames[0, 20, 10]}")
            
            
            
            
            
            
            
            
            
            
            """
            以下colormap
            """
        if USE_CSV:
            # CSVが保存されているフォルダ
            csv_folder = r"C:\Users\2025005585\Documents\tenmon\sich_sdt_score_visualize#sich_sdt_score_visualize\output_hensachi"
            
            # すべてのCSVファイルを取得
            csv_files = sorted(glob.glob(csv_folder + r"\hensachi_*.csv"))

            # 全CSVファイルを読み込み、各フレームをリストに格納
            frames = []

            for file in csv_files:
                frame = np.loadtxt(file, delimiter=",")
                frames.append(frame)

            # フレームのリストを3次元NumPy配列 (フレーム数 × 高さ × 幅) に変換
            data = np.array(frames)

        else:
            # 偏差値算出チームの変数をデータとして使う
            data = hensachi

        # データ形状からフレーム数・画像サイズの取得
        n_frames, height, width = data.shape

        # 入力データの確認(デバッグ表示)
        if DEBUG:
            print("====入力データ情報====")
            print("データサイズ:",data.shape)
            print("最小値:", data.min())
            print("最大値:", data.max())

            if USE_CSV:
                print("CSV枚数:", len(csv_files))
            
            print("=====================")
        colormap_lut = create_colormap()    
        
        # LUTの色を確認するためのカラーバー作成用(必要に応じて使用)
        #line = np.linspace(0, 100, width, dtype=np.uint8)
        # 高さ50ピクセルに設定
        #colorbar = np.tile(line, (50, 1))
        # グレースケール→BGR
        #colorbar_bgr = cv2.cvtColor(colorbar, cv2.COLOR_GRAY2BGR)
        # LUT適用
        #colorbar_result = cv2.LUT(colorbar_bgr, colormap_lut)
        # 保存
        #cv2.imwrite(r"C:\Users\2025005585\Desktop\python\colorbar.png", colorbar_result)


        # 動画作成用
        # 出力動画の設定
        video_writer = cv2.VideoWriter(
            OUTPUT_DIR + "\\" + OUTPUT_NAME + OUTPUT_EXT,
            cv2.VideoWriter_fourcc(*VIDEO_CODEC),
            FPS,
            (width, height)
        )

        # 1フレームずつ取り出し、LUTを適用して動画に書き込む
        for i in range(n_frames):
            frame = data[i]
            # 偏差値を0〜100に収めてuint8 型に変換
            clipped_frame = np.clip(frame, 0, 100).astype(np.uint8)
            # LUTを適用するため、グレースケール画像を3チャンネル(BGR)画像へ変換
            three_channel_frame = cv2.cvtColor(
                clipped_frame,
                cv2.COLOR_GRAY2BGR
            )
            # LUTを適用し、偏差値を対応する色へ変換
            color_mapped_frame = cv2.LUT(
                three_channel_frame,
                colormap_lut
            )
            # 動画ファイルに1フレーム書き込む
            video_writer.write(color_mapped_frame)

        video_writer.release()
        print("動画の作成が完了しました")
    else:
            print("有効なフレームが抽出されなかったため、保存処理をスキップしました。")
