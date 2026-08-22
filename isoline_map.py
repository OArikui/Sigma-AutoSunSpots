from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

import utils

# isoline map <=> 等値線図


def isoline(
    sigma_img: NDArray[np.float64],
    raw_image: NDArray[np.uint16 | np.uint8],
    levels: list[float],
    line_color: tuple[int, int, int] | None = None,
    line_thickness: int = 1,
    debug_show: bool = False,
) -> NDArray[np.uint8]:
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
        result_img (np,ndarray,RGB):raw_image上にsigma_imgにおけるlevelsの等値線を描画したもの
    Raise:
        ValueError:sigma_img.shape[:2] != raw_image.shape[:2]
        ValueError:sigma_img.ndim != 2
    """
    if line_color is None:
        line_color = (0, 255, 0)

    if sigma_img.shape[:2] != raw_image.shape[:2]:
        raise ValueError(
            "sigma_img と raw_image の画像サイズが一致しません\nsigma:{sigma_img.shape}\nraw  :{raw_image.shape}"
        )

    if sigma_img.ndim != 2:
        raise ValueError(f"sigma_img ndim not good:{sigma_img.ndim} ,is it color?")

    result_img = cv2.normalize(raw_image, None, 0, 255, cv2.NORM_MINMAX)
    result_img = result_img.astype(np.uint8)

    print("result_img dtype:", result_img.dtype)

    for level in levels:
        # 閾値処理で二値化
        _, thresh = cv2.threshold(sigma_img, level, 255, cv2.THRESH_BINARY)
        thresh_uint8 = thresh.astype(np.uint8)
        if debug_show:
            cv2.imshow("isoline-threshold", thresh_uint8)
            cv2.waitKey(0)  # キー入力待ち
            cv2.destroyAllWindows()

        contours, _ = cv2.findContours(
            thresh_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        result_img = cv2.drawContours(
            result_img, contours, -1, line_color, line_thickness
        )
        if debug_show:
            cv2.imshow("isoline-isoline map", result_img)
            cv2.waitKey(0)  # キー入力待ち
            cv2.destroyAllWindows()
    return result_img


if __name__ == "__main__":
    CROP_H = 600
    CROP_W = 600

    # 画像は16bitで読み込みます
    INPUT_DIR = r"E:/projects/Sigma_AutoSunSpots/samples/2025-07-20-PL1.zip"
    ISOLINE_DIR = ".\\save\\isoline_highlight"
    ISOLINE_FILE_NAME = "isoline"

    image_ext = ".png"

    DEBUGMODE = True

    sample_win_resolve = Path(INPUT_DIR).resolve()
    sample_name = (
        str(sample_win_resolve.parent.name) + "-" + str(sample_win_resolve.name)
    )
    frames, centers = utils.extract_sun_min2(INPUT_DIR, h_size=CROP_H, w_size=CROP_W)
    # frames: NDArray[NDArray[np.uint16]]
    # centers: NDArray[list[int]]

    print("frames:", frames.size) if DEBUGMODE else None
    mean, std, _ = utils.calculate_hensachi(frames)
    # mean: NDArray[np.float64] Range = int16
    # std: NDArray[np.float64] Range =  int16 (理論maxは (2^16)/2くらい)

    isoline_highlighted = isoline(std, mean, levels=[600], debug_show=DEBUGMODE)
    # isoline_highlighted: NDArray[uint8] channel=3
    filename = Path(ISOLINE_DIR) / f"ISOLINE_FILE_NAME__{sample_name}{image_ext}"
    utils.check_exist_mkdir(filename)

    if DEBUGMODE:
        print("filename", filename.resolve())
        print("img type:", isoline_highlighted.dtype)
        print(
            "max:", np.max(isoline_highlighted), "  min:", np.min(isoline_highlighted)
        )

    if cv2.imwrite(filename.resolve(), isoline_highlighted):
        print("[INFO]: save image sucessful")
    else:
        print("[WARNING]:imwrite failed")
