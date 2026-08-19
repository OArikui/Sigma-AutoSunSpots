# 旧std_score_visualize
import os

import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from libs.MIN2ver2 import MIN2_ignore_sunspots
from samples.zip_operator import (
    get_image_names_from_dir,
    load_image_from_path_cv2,
    get_image_names_from_zip,
    load_image_from_zip_cv2,
)

INPUT_DIR = "./sun_images"  # 処理対象の画像フォルダ
CROP_H = 800  # 抽出する画像サイズ(縦幅)
CROP_W = 800  # 抽出する画像サイズ(横幅)

DEBUG = True  # True: デバッグ情報を表示
MEAN_STD_OUTPUT_DIR = Path(
    r".\save\mean_std"
)  # 平均値と標準偏差の出力画像の保存先フォルダ
IMAGE_EXT = ".png"  # 画像の拡張子


def check_exist_mkdir(path: Path) -> None:
    if not path.exists():
        path.resolve().mkdir(parents=True, exist_ok=True)
        print(f"makedir {path.resolve()}")


def get_matplotlib_lut(cmap_name="viridis") -> np.ndarray:
    # Matplotlibのカラーマップを取得 (0~1の値)
    cmap = plt.get_cmap(cmap_name)

    indices = np.linspace(0, 1, 256)  # 0~255のインデックス
    colors = cmap(indices)[:, :3] * 255  # RGBを取得し、0~255の範囲に変換
    lut = (
        colors[:, ::-1].astype(np.uint8).reshape((256, 1, 3))
    )  # BGRに変換し、(256, 1, 3)の形状に整形
    return lut


def save_statistics_image(image, filename) -> None:
    """
    画像をheatmapとして保存
    """

    three_channel_image = cv2.cvtColor(
        image, cv2.COLOR_GRAY2BGR
    )  # LUT適用のため3チャンネル化

    color_mapped_image = cv2.LUT(three_channel_image, get_matplotlib_lut())
    cv2.imwrite(filename, color_mapped_image)


def crop_and_pad(
    img: np.ndarray, cx: int, cy: int, crop_h: int, crop_w: int
) -> np.ndarray:
    # 切り抜きたい理想の範囲（画面外にはみ出す可能性あり）
    h, w = img.shape
    crop_h = int(crop_h / 2)
    crop_w = int(crop_w / 2)
    y1, y2 = cy - crop_h, cy + crop_h
    x1, x2 = cx - crop_w, cx + crop_w

    # 画面外にはみ出している量（余白の計算）
    top = max(0, -y1)
    bottom = max(0, y2 - h)
    left = max(0, -x1)
    right = max(0, x2 - w)

    # 画面内に収まる安全な範囲だけでまずは切りぬく
    crop_y1, crop_y2 = max(0, y1), min(h, y2)
    crop_x1, crop_x2 = max(0, x1), min(w, x2)
    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]

    # はみ出していた部分を黒色（0）で埋めて、常にsize x size にする
    padded = cv2.copyMakeBorder(
        cropped, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )

    return padded


def extract_sun_min2(
    dir_path: str, h_size: int, w_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """フォルダ内の太陽画像から太陽中心を算出し、指定サイズで切りぬいた画像配列を返します。
    画面端にかかる場合は、足りない部分を黒く塗りつぶします。

    Args:
        folder(str):対象の画像が保存されているフォルダのパス
        h_size(int):切りぬく長方形の縦幅
        w_size(int):切りぬく長方形の横幅

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - 切りぬかれた画像の3次元配列（N,h_size,w_size)
            - 各画像の中心座標配列（N,2）
    """
    if dir_path.endswith(".zip"):
        zip_operate: bool = True
    else:
        zip_operate: bool = False

    print(f"---画像の読み込みと切り抜き処理を開始:{dir_path}---")
    # 画像ファイルのみ1000枚取得
    image_names = (
        get_image_names_from_zip(dir_path)
        if zip_operate
        else get_image_names_from_dir(dir_path)
    )
    frames = []
    min2_centers = []
    # tqdmによる進捗表示
    for name in tqdm.tqdm(image_names, desc="Processing images"):
        # 16bit(下位12bit)画像を輝度値(1ch)のまま正しく読み込む
        img = (
            load_image_from_zip_cv2
            if zip_operate
            else load_image_from_path_cv2(dir_path, name)
        )
        if img is None:
            continue
        try:
            cx, cy, r = MIN2_ignore_sunspots(img, show=False, debug=False)
        except Exception:  # noqa: S112,BLE001
            continue
        r = int(r)
        cx = int(cx)
        cy = int(cy)

        padded = crop_and_pad(img, cx, cy, h_size, w_size)

        frames.append(padded)
        min2_centers.append([cx, cy])

    return np.array(frames), np.array(min2_centers)


def calculate_hensachi(frames: np.ndarray):
    """平均画像・標準偏差画像・偏差値画像を計算する。"""

    # 平均画像
    mean = np.mean(frames, axis=0)

    # 標準偏差画像
    std = np.std(frames, axis=0)

    # 偏差値画像
    deviation = np.where(std == 0, 50, 50 + 10 * (frames - mean) / std)

    return mean, std, deviation


def scale_to_uint8(img: np.ndarray, min_max_normalization: bool = False) -> np.ndarray:
    """画像のdtypeに応じて0-255のuint8型にスケーリングする関数"""
    if img.dtype == np.uint8:
        return img

    if min_max_normalization:
        img_min = img.min()
        img_max = img.max()
        if img_max == img_min:
            return np.zeros_like(img, dtype=np.uint8)

        # 0.0 ~ 1.0 に正規化してから 255 倍
        normalized = (img - img_min) / (img_max - img_min)
        return (normalized * 255).astype(np.uint8)

    # float型（一般的に 0.0 ~ 1.0）
    elif np.issubdtype(img.dtype, np.floating):
        img_clipped = np.clip(img, 0.0, 1.0)
        return (img_clipped * 255).astype(np.uint8)

        # uint16型
    elif img.dtype == np.uint16:
        return (img / 256).astype(np.uint8)

    else:
        raise ValueError("unknown unit")


if __name__ == "__main__":
    print(f"\n--- 画像ファイルの読み込み開始: {INPUT_DIR} ---")

    frames, centers = extract_sun_min2(INPUT_DIR, h_size=CROP_H, w_size=CROP_W)
    mean, std, hensachi = calculate_hensachi(frames)

    if DEBUG:
        print("====平均値・標準偏差データ情報====")
        print("framesサイズ:", frames.shape)
        print("meanサイズ:", mean.shape)
        print("stdサイズ:", std.shape)
        print("================================")

    check_exist_mkdir(MEAN_STD_OUTPUT_DIR)
    data = hensachi
    n_frames, height, width, _ = data.shape
    save_statistics_image(
        mean,
        os.path.join(MEAN_STD_OUTPUT_DIR, "mean" + IMAGE_EXT),
    )  # 平均値画像と標準偏差画像の出力が完了しました

    save_statistics_image(
        std,
        os.path.join(MEAN_STD_OUTPUT_DIR, "std" + IMAGE_EXT),
    )  # 標準偏差画像

    print("平均値画像と標準偏差画像の出力が完了しました")
