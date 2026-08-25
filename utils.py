# 旧std_score_visualize
import logging
from pathlib import Path
from typing import Annotated

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from numpy.typing import NDArray

from libs.MIN2ver2 import MIN2_ignore_sunspots, version
from samples.zip_operator import (
    get_image_names_from_dir,
    get_image_names_from_zip,
    load_image_from_path_cv2,
    load_image_from_zip_cv2,
)

logger = logging.getLogger(__name__)

INPUT_DIR = "./sun_images"  # 処理対象の画像フォルダ
CROP_H = 800  # 抽出する画像サイズ(縦幅)
CROP_W = 800  # 抽出する画像サイズ(横幅)

DEBUG = True  # True: デバッグ情報を表示
MEAN_STD_OUTPUT_DIR = Path(r".\save\mean_std")  # 平均値と標準偏差の出力画像の保存先フォルダ
IMAGE_EXT = ".png"  # 画像の拡張子


def check_exist_mkdir(path: Path) -> None:
    if "." in str(path):
        path = path.parent
    if not path.exists():
        path.resolve().mkdir(parents=True, exist_ok=True)
        print(f"makedir {path.resolve()}")


def get_matplotlib_lut(cmap_name: str = "viridis") -> NDArray[np.uint8]:
    # Matplotlibのカラーマップを取得 (0~1の値)
    cmap = plt.get_cmap(cmap_name)

    indices = np.linspace(0, 1, 256)  # 0~255のインデックス
    colors = cmap(indices)[:, :3] * 255  # RGBを取得し、0~255の範囲に変換
    lut = colors[:, ::-1].astype(np.uint8).reshape((256, 1, 3))  # BGRに変換し、(256, 1, 3)の形状に整形
    return lut


def save_statistics_image(image: NDArray[np.uint8], filename: str) -> None:
    """
    画像をheatmapとして保存
    """

    three_channel_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # LUT適用のため3チャンネル化

    color_mapped_image = cv2.LUT(three_channel_image, get_matplotlib_lut())
    cv2.imwrite(filename, color_mapped_image)


def crop_and_pad(
    img: NDArray[np.uint16 | np.uint8], cx: int, cy: int, crop_h: int, crop_w: int
) -> NDArray[np.uint16 | np.uint8]:
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
    padded = cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0).astype(
        img.dtype
    )

    return padded


def extract_sun_min2(dir_path: str, h_size: int, w_size: int) -> tuple[NDArray[np.uint16], np.ndarray]:
    """フォルダ内の太陽画像から太陽中心を算出し、指定サイズで切りぬいた画像配列を返します。
    画面端にかかる場合は、足りない部分を黒く塗りつぶします。

    Args:
        folder(str):対象の画像が保存されているフォルダのパス
        h_size(int):切りぬく長方形の縦幅
        w_size(int):切りぬく長方形の横幅

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - 切りぬかれた画像の3次元配列（N,h_size,w_size)
            - 各画像の太陽近似円stat（(x,y),r）
    """
    getted_param = {"h_size": h_size, "w_size": w_size, "dirpath": dir_path}
    logger.debug(f"extract_sun_min2 param {getted_param}")

    if dir_path.endswith(".zip"):
        zip_operate: bool = True
        logger.info("zip_operate mode")
    else:
        zip_operate = False

    min2_params = {"n": 10, "light_threshold": 50, "limb_wigth": 24}
    logger.debug(f"min2_version:{version}")
    logger.debug(f"min2_params:{min2_params}")

    print(f"---画像の読み込みと切り抜き処理を開始:{dir_path}---")
    # 画像ファイルのみ1000枚取得
    image_names = get_image_names_from_zip(dir_path) if zip_operate else get_image_names_from_dir(dir_path)
    logger.debug(f"image_names: \n{image_names}")
    frames: list = []
    min2_stats = []
    # tqdmによる進捗表示
    for name in tqdm.tqdm(image_names, desc="Processing images"):
        # 16bit(下位12bit)画像を輝度値(1ch)のまま正しく読み込む
        img: NDArray[np.uint16] = (
            load_image_from_zip_cv2(dir_path, name) if zip_operate else load_image_from_path_cv2(dir_path, name)
        )
        if img is None:
            logger.Warning(f"No img: {name}")
            continue
        try:
            cx, cy, r = MIN2_ignore_sunspots(img, show=False, debug=False, **min2_params)
        except Exception:  # noqa: S112,BLE001
            continue

        padded: NDArray[np.uint16] = crop_and_pad(img, int(cx), int(cy), h_size, w_size)

        frames.append(padded)
        min2_stats.append([cx, cy, r])

    min2_stats = np.array(min2_stats)
    cxes = min2_stats[:, 0]
    cyes = min2_stats[:, 1]
    rs = min2_stats[:, 2]

    logger.debug(f"cir_stat cxes (mean:{np.mean(cxes)},std:{np.std(cxes)})")
    logger.debug(f"cir_stat cyes (mean:{np.mean(cyes)},std:{np.std(cyes)})")
    logger.debug(f"cir_stat rs   (mean:{np.mean(rs)},std:{np.std(rs)})")

    return np.array(frames), min2_stats


def calculate_hensachi(
    frames: NDArray[NDArray[np.uint16]],
) -> Annotated[
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[NDArray[np.float64]],
    ],
    "RANGE = 16bit, 16bit, 0-100",
]:
    """平均画像・標準偏差画像・偏差値画像を計算する。"""

    # 平均画像
    mean: NDArray[np.float64] = np.mean(frames, axis=0)

    # 標準偏差画像
    std: NDArray[np.float64] = np.std(frames, axis=0)

    # 偏差値画像
    deviation: NDArray[np.float64] = np.where(std == 0, 50, 50 + 10 * (frames - mean) / std)

    return mean, std, deviation


def scale_to_uint8(
    img: np.ndarray,
    min_max_normalization: bool = False,
    float_range: tuple[float, float] = (0.0, 1.0),
    color_channel: bool = False,
) -> NDArray[np.uint8]:
    """画像のdtypeに応じて0-255のuint8型にスケーリングする関数"""
    if img.dtype == np.uint8:
        norm_img = img

    if min_max_normalization:
        img_min = img.min()
        img_max = img.max()
        if img_max == img_min:
            norm_img = np.zeros_like(img, dtype=np.uint8)

        # 0.0 ~ 1.0 に正規化してから 255 倍
        normalized = (img - img_min) / (img_max - img_min)
        norm_img = (normalized * 255).astype(np.uint8)

        # float
    elif np.issubdtype(img.dtype, np.floating):
        floor, loof = float_range
        img_max = np.max(img)
        img_min = np.min(img)
        if floor > img_min or loof < img_max:
            raise ValueError(
                f"__bad range for the img \nimage_range:{img_min, img_max}\nscale_range{floor, loof}"
            )
        loofed_img = img * (255 / loof)
        cliped_img = loofed_img - floor
        norm_img = cliped_img.astype(np.uint8)

        # uint16型
    elif img.dtype == np.uint16:
        norm_img = (img / 256).astype(np.uint8)

    else:
        raise ValueError("unknown unit")

    if color_channel:
        logger.info("expand the channel to three,by option 'color_channel'")
        if norm_img.ndim == 2 or norm_img.shape[2] == 1:
            expanded = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
            logger.debug(" the image is not color,Expand the channel to three.")
        else:
            expanded = norm_img
            logger.debug(" the image has already expanded the channel to three")

        return expanded
    else:
        return norm_img


def drawContours_alpha(
    image: np.ndarray, contours: np.ndarray, lineC_BGRA: tuple[int, int, int, float], line_thickness: int = 1
) -> NDArray[np.uint8]:
    getted_param = {"line_color_BGR":lineC_BGR,"line_thickness":line_thickness}
    logger.debug(f"utils.drawContours_alpha param : {getted_param}")

    image_max = np.max(image)
    logger.info(f"image_rangeを判定 (image_max = {image_max})")
    if image_max > 4096:
        float_range = (0.0, 2.0**16)
        logger.debug("the image is int16bit range")
    elif image_max > 256:
        float_range = (0.0, 2.0**12)
        logger.debug("the image is int12bit range")
    else:
        float_range = (0.0, 2.0**8)
        logger.debug("the image is int8bit range")

    norm_BGR = scale_to_uint8(image, float_range)

    overlay = norm_BGR.copy()

    lineC_BGR = lineC_BGRA[:3]
    alpha = lineC_BGRA[-1]
    if alpha > 1.0 or alpha < 0.0:
        logger.waring("reset alpha to 0.5, alpha should fall between 0 and 1.")
    cv2.drawContours(overlay, contours, -1, lineC_BGR, thickness = line_thickness)

    output = cv2.addWeighted(overlay, alpha, norm_BGR, 1 - alpha, 0)

    return output


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
        str(Path(MEAN_STD_OUTPUT_DIR) / f"mean{IMAGE_EXT}"),
    )  # 平均値画像と標準偏差画像の出力が完了しました

    save_statistics_image(
        std,
        str(Path(MEAN_STD_OUTPUT_DIR) / f"mean{IMAGE_EXT}"),
    )  # 標準偏差画像

    print("平均値画像と標準偏差画像の出力が完了しました")
