from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

import utils

# isoline map <=> 等値線図
import logging
from datetime import datetime

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# ログ設定（INFO以上のログを app.log ファイルに記録）


# ロガーの作成（__name__ を指定することで実行中のモジュール名がログに入る）
import logging
from datetime import datetime

# タイムスタンプの作成（例）
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. ロガーの作成
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # ロガー全体で受け入れる最小ログレベル

# 2. フォーマットの定義
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# 3. ファイル出力用ハンドラの設定
file_handler = logging.FileHandler(f"save/logs/{ts}.log", mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# 4. コンソール出力用ハンドラの設定
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# 5. ハンドラをロガーに追加
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 6. 設定完了後にログを出力する
logger.info("処理を開始しました")
logger.info("ロガー経由のメッセージです")


def get_contour_bboxes(
    contours: list[NDArray[np.int32]], min_area: int = 0, debug: bool = False
) -> list[tuple[int, int, int, int]]:
    """cv2.findContoursの戻り値countoursからbboxesのリストを返す関数

    Parameters
    ----------
    contours: list[NDArray[np.int32]]
        cv2.findContours の戻り値. contours[i].shape == (N, 1, 2)
    min_area : int, optional
        指定した面積（ピクセル数）未満のノイズを除外する閾値, by default 0
    debug : bool
        検出したbboxesのdebug情報を標準出力する
    Returns
    -------
    bboxes : [tuple[int, int, int, int]]
        検出された bound box のリスト [(x, y, w, h), ...] x,y は bound の左上(原点に一番近い頂点)の座標

        Notes
        -------
        bbox:(x,y,w,h)
        Origin on the image
        (0, 0)────────────────────────────┐
        │                                 │
        │     (x, y) ─────────── w ──────┐│
        │       │                        ││
        │       │[ connected component ] ││
        │       h                        ││
        │       │                        ││
        │       └─────────────────────── ○│
        │                          (x+w, y+h)
        └─────────────────────────────────┘
    """

    bboxes = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if debug:
            logger.debug(f"計算された Bounding ({i}) : bbox = {x, y, w, h}")
        if area >= min_area:
            bboxes.append((x, y, w, h))
        else:
            if debug:
                logger.debug(
                    f" Bounding ({i}) は {area} < min_area({min_area}) により無視されます。"
                )

    return bboxes


def isoline(
    sigma_img: NDArray[np.float64],
    raw_image: NDArray[np.uint16 | np.uint8],
    levels: list[float],
    highlight_bound: bool = False,
    line_color: tuple[int, int, int] | None = None,  # RGB,default=(0,255,0)
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
    if debug_show:
        logger.debug(" starting isoline_map.isoline as DEBUGMODE")

    if line_color is None:
        line_color = (0, 255, 0)
        if debug_show:
            logger.debug(
                f"line_color is not instructed. Useing deafaul = {line_color}(RGB)"
            )

    R, G, B = line_color
    line_color_BGR = B, G, R

    if sigma_img.shape[:2] != raw_image.shape[:2]:
        raise ValueError(
            f"sigma_img と raw_image の画像サイズが一致しません\n"
            f"sigma:{sigma_img.shape}\nraw  :{raw_image.shape}"
        )

    if sigma_img.ndim != 2:
        raise ValueError(f"sigma_img ndim not good:{sigma_img.ndim} ,is it color?")

    # 1. 0-255の8bitに正規化
    norm_img = cv2.normalize(raw_image, None, 0, 255, cv2.NORM_MINMAX)
    norm_img = norm_img.astype(np.uint8)

    # 2. グレースケール(1ch)の場合は3ch(BGR)に変換してカラー描画を可能にする
    if norm_img.ndim == 2 or norm_img.shape[2] == 1:
        result_img = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
        if debug_show:
            logger.debug(" raw_img is not color,Expand the channel to three.")
    else:
        result_img = norm_img.copy()

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

        if highlight_bound:
            bboxes = get_contour_bboxes(contours, debug=debug_show)
            for x, y, w, h in bboxes:
                if line_thickness == -1:  # 塗りつぶし矩形
                    cv2.rectangle(
                        result_img,
                        (x, y),
                        (x + w, y + h),
                        line_color_BGR,
                        thickness=cv2.FILLED,
                    )
                else:
                    cv2.rectangle(
                        result_img,
                        (x, y),
                        (x + w, y + h),
                        line_color_BGR,
                        thickness=line_thickness,
                        lineType=cv2.LINE_AA,
                    )
        else:
            result_img = cv2.drawContours(
                result_img, contours, -1, line_color_BGR, line_thickness
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
    INPUT_PARENT_DIR = Path(r"E:/projects/Sigma_AutoSunSpots/samples/")
    INPUT_DIR_NAMES = [
        "2025-07-20-PL1.zip",
        "2025-08-30-LT1.zip",
        "2026-01-12-LT1.zip",
        "2026-01-17-PL1.zip",
    ]
    ISOLINE_DIR = ".\\save\\isoline_highlight"
    ISOLINE_FILE_NAME = "isoline"

    image_ext = ".png"

    BOUND = True
    DEBUGMODE = True

    for i, INPUT_DIR_NAME in enumerate(INPUT_DIR_NAMES):
        logger.info(
            f"=== SunSpots highlight process ({i + 1} / {len(INPUT_DIR_NAMES)}) ==="
        )
        logger.info(f" current = {INPUT_DIR_NAME}")

        INPUT_DIR = str(INPUT_PARENT_DIR / INPUT_DIR_NAME)
        sample_win_resolve = Path(INPUT_DIR).resolve()
        sample_name = (
            str(sample_win_resolve.parent.name) + "-" + str(sample_win_resolve.name)
        )
        frames, centers = utils.extract_sun_min2(
            INPUT_DIR, h_size=CROP_H, w_size=CROP_W
        )
        # frames: NDArray[NDArray[np.uint16]]
        # centers: NDArray[list[int]]

        logger.debug(f"frames:{frames.size}") if DEBUGMODE else None
        mean, std, _ = utils.calculate_hensachi(frames)
        # mean: NDArray[np.float64] Range = int16
        # std: NDArray[np.float64] Range =  int16 (理論maxは (2^16)/2くらい)

        isoline_highlighted = isoline(
            std, mean, levels=[600], debug_show=DEBUGMODE, highlight_bound=BOUND
        )
        # isoline_highlighted: NDArray[uint8] channel=3
        filename = Path(ISOLINE_DIR) / f"ISOLINE_FILE_NAME__{sample_name}{image_ext}"
        utils.check_exist_mkdir(filename)

        if DEBUGMODE:
            logger.debug(f"figure save path{filename.resolve()}")
            logger.debug(f"figure image type:{isoline_highlighted.dtype}")
            logger.debug(f"  max:{np.max(isoline_highlighted)}")
            logger.debug(f"  min:{np.min(isoline_highlighted)}")

        if cv2.imwrite(filename.resolve(), isoline_highlighted):
            logger.info("save image sucessful")
        else:
            logger.warning("imwrite failed")
