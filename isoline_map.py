# isoline map <=> 等値線図
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

import utils


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
                logger.debug(f" Bounding ({i}) は {area} < min_area({min_area}) により無視されます。")

    return bboxes


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
    ISOLINE_FILE_NAME = "bounded"

    SIGMA_DIR = ".\\save\\Sigma_img"
    SIGMA_THRESH_FILE_NAME = "thresh_uint8"

    image_ext = ".png"

    sigma_threshold = 600  # int16bit 想定
    min_area = 0

    line_color_BGR = (0, 255, 0)
    line_thickness = 1

    BOUND = True
    DEBUGMODE = True

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(f"save/logs/{ts}.log", mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("処理を開始しました")
    logger.info("ロガー経由のメッセージです")

    for i, INPUT_DIR_NAME in enumerate(INPUT_DIR_NAMES):
        logger.info(f"=== SunSpots highlight process ({i + 1} / {len(INPUT_DIR_NAMES)}) ===")
        logger.info(f" current = {INPUT_DIR_NAME}")

        INPUT_DIR = str(INPUT_PARENT_DIR / INPUT_DIR_NAME)

        #  sample_name を定義
        sample_win_resolve = Path(INPUT_DIR).resolve()
        sample_name = str(sample_win_resolve.parent.name) + "-" + str(sample_win_resolve.name)

        #  highlighted と thresh_img の保存パスを設定
        highlight_path = Path(ISOLINE_DIR) / f"{ISOLINE_FILE_NAME}__{sample_name}{image_ext}"
        thresh_path = Path(ISOLINE_DIR) / f"{SIGMA_THRESH_FILE_NAME}__{sample_name}{image_ext}"
        utils.check_exist_mkdir(highlight_path)
        utils.check_exist_mkdir(thresh_path)

        #  INPUT_DIR 内の 全画像 と その座標リストを取得
        frames, centers = utils.extract_sun_min2(
            INPUT_DIR, h_size=CROP_H, w_size=CROP_W
        )  # frames: NDArray[NDArray[np.uint16]]

        if DEBUGMODE:
            logger.debug(f"frames size:{frames.size}")

        #  平均値, 標準偏差 を取得 (偏差値は棄てる)
        mean, std, _ = utils.calculate_hensachi(frames)  # mean,std共にnp.float64だが、範囲はint16bit

        # thresh_uint8 を作製
        _, thresh = cv2.threshold(std, sigma_threshold, 255, cv2.THRESH_BINARY)
        thresh_uint8 = thresh.astype(np.uint8)

        # contours を取得
        contours, _ = cv2.findContours(thresh_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # bound boxes を取得
        bboxes = get_contour_bboxes(contours)

        if BOUND:
            highlights_Fcnt = []
            for x, y, w, h in bboxes:
                pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
                pts = pts.reshape((-1, 1, 2))
                highlights_Fcnt.append(pts)
        else:
            highlights_Fcnt = contours

        norm_img = utils.scale_to_uint8(mean, float_range=(0.0, 2.0**16))

        if norm_img.ndim == 2 or norm_img.shape[2] == 1:
            background_img = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
            if DEBUGMODE:
                logger.debug(" background_img is not color,Expand the channel to three.")
        else:
            background_img = norm_img

        highlighted = cv2.drawContours(background_img, highlights_Fcnt, -1, line_color_BGR, line_thickness)

        if cv2.imwrite(highlight_path.resolve(), highlighted):
            logger.info("save highlighted image sucessful")
        else:
            logger.warning("highlighted image imwrite failed")

        if cv2.imwrite(thresh_path.resolve(), thresh_uint8):
            logger.info("save thresh_uint8 image sucessful")
        else:
            logger.warning("thresh_uint8 imwrite failed")
