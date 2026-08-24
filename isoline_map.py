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

    sigma_threshold = 600  # int16bit 想定

    min_area = 0.0  # 指定した面積（ピクセル数）未満のノイズを除外する閾値 (0.0~)
    diame_ratio = 0.5  # 縁と判断される bbox の 長辺 の 太陽直径 に対する 最小 の 割合 (0.0~1.0)

    line_color_BGR = (0, 255, 0)
    line_thickness = 1  # 画像は16bitで読み込みます

    INPUT_PARENT_DIR = Path(r"E:/projects/Sigma_AutoSunSpots/samples/")
    INPUT_DIR_NAMES = [
        "2025-07-20-PL1.zip",
        "2025-08-30-LT1.zip",
        "2026-01-12-LT1.zip",
        "2026-01-17-PL1.zip",
    ]
    SAVE_DIR = f".\\save\\isoline_highlight\\P{sigma_threshold}_{min_area}"

    image_ext = ".png"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s [%(module)s.%(funcName)s:%(lineno)d]   [%(levelname)s] %(message)s"
    )
    log_path = Path(SAVE_DIR) / "{ts}.log"

    file_handler = logging.FileHandler(f"{log_path.resolve()}", mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("処理を開始しました")
    logger.info("ロガー経由のメッセージです")

    logger.debug(f"sigma_threshold:{sigma_threshold}")
    logger.debug(f"min_area:{min_area}")
    logger.debug(f"diame_ratio:{diame_ratio}")
    # param_valid
    if min_area < 0:
        raise ValueError("param.min_area should be 0 or greater.")
    if diame_ratio < 0 or diame_ratio > 1.0:
        raise ValueError("param.diameter ratio should fall between 0 and 1. ")

    for i, INPUT_DIR_NAME in enumerate(INPUT_DIR_NAMES):
        logger.info(f"=== SunSpots highlight process ({i + 1} / {len(INPUT_DIR_NAMES)}) ===")
        logger.info(f" current = {INPUT_DIR_NAME}")

        logger.info(f"--- setting path ---")
        logger.info(f"set sample path")
        INPUT_DIR = str(INPUT_PARENT_DIR / INPUT_DIR_NAME)
        logger.debug(f"INPUT_DIR: '{INPUT_DIR}'")

        #  sample_name を定義
        sample_win_resolve = Path(INPUT_DIR).resolve()
        sample_name = str(sample_win_resolve.parent.name) + "-" + str(sample_win_resolve.name)
        logger.debug(f"sample_name: '{sample_name}'")

        logger.info("set save path")
        isoline_path = Path(SAVE_DIR) / f"isoline__{sample_name}{image_ext}"
        utils.check_exist_mkdir(isoline_path)
        logger.debug(f"isoline_path: '{isoline_path}'")

        bounded_path = Path(SAVE_DIR) / f"bounded__{sample_name}{image_ext}"
        utils.check_exist_mkdir(bounded_path)
        logger.debug(f"bounded_path: '{bounded_path}'")

        thresh_path = Path(SAVE_DIR) / f"thresh_uint8__{sample_name}{image_ext}"
        utils.check_exist_mkdir(thresh_path)
        logger.debug(f"thresh_path: '{thresh_path}'")

        logger.info("--- calculate sigma image ---")
        logger.info("loading sample images and compute circle stats")
        #  INPUT_DIR 内の 全画像 と その座標リストを取得
        frames, stats = utils.extract_sun_min2(INPUT_DIR, h_size=CROP_H, w_size=CROP_W)
        mean_r = stats[:, 2].mean()

        logger.info("compute mean,std")
        #  平均値, 標準偏差 を取得 (偏差値は棄てる)
        mean, std, _ = utils.calculate_hensachi(frames)  # mean,std共にnp.float64だが、範囲はint16bit

        logger.debug(f"sigma image size: {std.size}")

        logger.info("--- highlight based on sigma image ---")
        logger.info("calculate contours")
        # thresh_uint8 を作製
        _, thresh = cv2.threshold(std, sigma_threshold, 255, cv2.THRESH_BINARY)
        thresh_uint8 = thresh.astype(np.uint8)

        # contours を取得
        contours, _ = cv2.findContours(thresh_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        logger.debug(f"num of contours  : {len(contours)}")

        logger.info(f"bounding and inspecting contours (bboxstat = (x,y,w,h,area))")

        un_compatible_Fcnt = []
        bound_Fcnt = []
        for i, cnt in enumerate(contours):
            compatible = True

            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            logger.debug(f"検出された bbox ({str(i).zfill(len(str(len(contours))))}) :{x, y, w, h, area} ")

            pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))

            if area < min_area:
                compatible = False
                logger.debug(f" ノイズとして除外しました {area} < {min_area} (min_area)")
            elif np.max([w, h]) >= mean_r * diame_ratio * 2:
                compatible = False
                logger.debug(f" 縁として除外しました {np.max([w, h])} (長辺) > {mean_r * diame_ratio * 2}")

            if compatible:
                bound_Fcnt.append(pts)
            else:
                un_compatible_Fcnt.append(pts)

        logger.debug(f"Number of defective items : {len(un_compatible_Fcnt)} / {len(contours)}")

        logger.info("--- draw highlights on the image. ---")
        logger.info("normalize image")

        norm_img = utils.scale_to_uint8(mean, float_range=(0.0, 2.0**16))

        if norm_img.ndim == 2 or norm_img.shape[2] == 1:
            background_img = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
            logger.debug(" background_img is not color,Expand the channel to three.")
        else:
            background_img = norm_img

        bounded_img = cv2.drawContours(background_img, bound_Fcnt, -1, line_color_BGR, line_thickness)

        isoline_img = cv2.drawContours(background_img, contours, -1, line_color_BGR, line_thickness)

        if cv2.imwrite(bounded_path.resolve(), bounded_img):
            logger.info("save bounded_img image sucessful")
        else:
            logger.warning("bounded_img image imwrite failed")

        if cv2.imwrite(isoline_path.resolve(), isoline_img):
            logger.info("save isoline_img image sucessful")
        else:
            logger.warning("isoline_img image imwrite failed")

        if cv2.imwrite(thresh_path.resolve(), thresh_uint8):
            logger.info("save thresh_uint8 image sucessful")
        else:
            logger.warning("thresh_uint8 imwrite failed")
