from pathlib import Path

import cv2
import numpy as np
import utils

# isoline map <=> 等値線図


def isoline(
    sigma_img: np.ndarray,
    raw_image: np.ndarray,
    levels: list[float],
    line_color: tuple[int] = (0, 255, 0),
    line_thickness: float = 1,
) -> np.ndarray:
    """
    Description:
        sigma_img上の特定の値の等値線をraw_imageに描画します
    Arg:
        sigma_img (np.ndarray):1チャンネル。raw_imageと同じ画像sizeで。
        raw_img (np.ndarray):sigma_imgと同じ画像sizeで。
        levels (list[float]):描画する等値線の値。複数可
        line_color (tuple[int]):等値線の色 RGB
        line_thickness (float):等値線の太さ
    Return:
        result_img (np,ndarray):raw_image上にsigma_imgにおけるlevelsの等値線を描画したもの
    Raise:
        ValueError:sigma_img.shape[:2] != raw_image.shape[:2]
        ValueError:sigma_img.ndim != 2
    """

    if sigma_img.shape[:2] != raw_image.shape[:2]:
        raise ValueError(
            "sigma_img と raw_image の画像サイズが一致しません\nsigma:{sigma_img.shape}\nraw  :{raw_image.shape}"
        )

    if sigma_img.ndim != 2:
        raise ValueError("sigma_img ndim not good ,is it color?")

    result_img = raw_image.copy()

    for level in levels:
        # 閾値処理で二値化
        _, thresh = cv2.threshold(sigma_img, level, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_img, contours, -1, line_color, line_thickness)

    return result_img


if __name__ == "__main__":
    CROP_H = 600
    CROP_W = 600

    INPUT_DIR = ""
    ISOLINE_DIR = ".\\save\\isoline_highlight"
    ISOLINE_FILE_NAME = "isoline"

    image_ext = ".png"

    frames, centers = utils.extract_sun_min2(INPUT_DIR, h_size=CROP_H, w_size=CROP_W)
    mean, std, _ = utils.calculate_hensachi(frames)

    isoline_highlighted = isoline(std, mean, levels=[600])

    filename = Path(ISOLINE_DIR) / f"ISOLINE_FILE_NAME{image_ext}"
    cv2.imwrite(filename.resolve(), isoline_highlighted)
