import json
import cv2

import numpy as np
from numpy.typing import NDArray
from logging import getLogger

import tifffile as tiff

logger = getLogger(__name__)


def save_scaled_std_tiff(filename: str, std_array: np.ndarray, scale_factor: float = 10.0) -> bool:
    """
    標準偏差配列(float)をスケール変換して uint16 TIFF として保存する関数
    """
    # 1. 10倍して四捨五入＆範囲制限 (0〜65535)
    scaled_data = np.round(std_array * scale_factor)
    clipped_data = np.clip(scaled_data, 0, 65535).astype(np.uint16)

    # 2. メタデータ作成
    metadata = {
        "madeby": "Sigma_AutoSunSpotsV0 https://github.com/OArikui/Sigma-AutoSunSpots",
        "data_type": "standard_deviation",
        "scale_factor": scale_factor,
        "unit_per_lsb": 1.0 / scale_factor,
        "original_dtype": str(std_array.dtype),
    }

    try:
        # 3. TIFFタグ (ImageDescription) に埋め込んで保存
        tiff.imwrite(filename, clipped_data, description=json.dumps(metadata, indent=2))
        logger.debug(f"[保存完了] {filename} (shape: {clipped_data.shape}, dtype: {clipped_data.dtype})")
        return True
    except:
        logger.exception("保存失敗")
        return False


def load_scaled_std_tiff(filename: str, uint8: bool) -> tuple[np.ndarray, float]:
    """
    スケールされた uint16 TIFF を読み込み、元の float 配列に戻す関数
    """
    with tiff.TiffFile(filename) as tif:
        # 画像読み込み
        img_uint16 = tif.asarray()

        # メタデータ読み込み
        scale_factor = 10.0  # デフォルト値
        page = tif.pages[0]
        if "ImageDescription" in page.tags:
            try:
                meta = json.loads(page.tags["ImageDescription"].value)
                scale_factor = float(meta.get("scale_factor", 10.0))
            except (json.JSONDecodeError, TypeError):
                pass

        # 元の小数値(float32/float64)に復元
        restored_std = img_uint16.astype(np.float32) / scale_factor
        if uint8:
            restored_std /= 16
            restored_std.astype(np.uint8)
        return restored_std, scale_factor


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


def scale_to_uint8(
    img: np.ndarray,
    min_max_normalization: bool = False,
    float_range: tuple[float, float] | None = None,
    color_channel: bool = False,
) -> NDArray[np.uint8]:
    """画像のdtypeに応じて0-255のuint8型にスケーリングする関数"""
    getted_param = {
        "min_max_normalization": min_max_normalization,
        "float_range": float_range,
        "color_channel": color_channel,
    }
    logger.debug(f"getted_param : {getted_param}")
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
        if float_range is None:
            image_max = np.max(img)
            logger.info(f"float_range 未設定のため、 image_range を判定します (image_max = {image_max})")
            if image_max > 4096:
                float_range = (0.0, 2.0**16)
                logger.debug("the image is int16bit range")
            elif image_max > 256:
                float_range = (0.0, 2.0**12)
                logger.debug("the image is int12bit range")
            else:
                float_range = (0.0, 2.0**8)
                logger.debug("the image is int8bit range")
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
        logger.error(f"unknown dtype: {img.dtype},Scaling is not performed.")

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

