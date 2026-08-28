# isoline map <=> 等値線図
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.pyplot import Button, Slider

import utils

__version__ = "Sigma-AutoSunsSpots_Tmethod_V1"


def histogram(image: np.ndarray) -> np.ndarray:
    dt = image.dtype.itemsize * 8
    print("dt:", dt)
    print("img's Max value:", np.max(image))
    if np.max(image) > 2**12:
        q = 1
    else:
        q = 16

    y = []
    for i in range(2**dt):
        if i % q == 0:
            y.append(0)

    for i in image:
        for p in i:
            y[p] += 1

    return np.array(y)


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


def drawContours_alpha(
    image: np.ndarray, contours: np.ndarray, lineC_BGRA: tuple[int, int, int, float], line_thickness: int = 1
) -> NDArray[np.uint8]:
    getted_param = {"line_color_BGR": lineC_BGRA, "line_thickness": line_thickness}
    logger.debug(f"getted_param : {getted_param}")

    norm_BGR = utils.scale_to_uint8(image, float_range=None)

    overlay = norm_BGR.copy()

    lineC_BGR = lineC_BGRA[:3]
    alpha = lineC_BGRA[-1]
    if alpha > 1.0 or alpha < 0.0:
        logger.waring("reset alpha to 0.5, alpha should fall between 0 and 1.")
    cv2.drawContours(overlay, contours, -1, lineC_BGR, thickness=line_thickness)

    output = cv2.addWeighted(overlay, alpha, norm_BGR, 1 - alpha, 0)

    return output


if __name__ == "__main__":
    CROP_H = 600
    CROP_W = 600

    sigma_threshold = 600  # int16bit 想定

    min_area = 0.0  # 指定した面積（ピクセル数）未満のノイズを除外する閾値 (0.0~)
    diame_ratio = 0.5  # 縁と判断される bbox の 長辺 の 太陽直径 に対する 最小 の 割合 (0.0~1.0)

    lineC_BGR = (0, 255, 0)
    un_compat_lineC_BGRA = (10, 10, 255, 0.3)
    line_thickness = 1  # 画像は16bitで読み込みます

    hist_show = False

    INPUT_PARENT_DIR = Path(r"E:/projects/AutoSunsSpots/Sigma_AutoSunSpots/samples")
    INPUT_DIR_NAMES = [
        "2025-07-20-PL1.zip",
        "2025-08-30-LT1.zip",
        "2026-01-12-LT1.zip",
        "2026-01-17-PL1.zip",
    ]
    SAVE_DIR = Path(__file__).resolve().parent / ".\\save\\isoline_highlight"

    image_ext = ".png"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s [%(module)s.%(funcName)s:%(lineno)d]   [%(levelname)s] %(message)s"
    )
    log_path = SAVE_DIR.parent / "logs" / f"{ts}.log"
    utils.check_exist_mkdir(log_path)

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
        logger.info(
            f"====================== SunSpots highlight process ({i + 1} / {len(INPUT_DIR_NAMES)}) ========================="
        )
        logger.info(f" current = {INPUT_DIR_NAME}")

        logger.info("--- setting path ---")
        logger.info("set sample path")
        INPUT_DIR = str(INPUT_PARENT_DIR / INPUT_DIR_NAME)
        logger.debug(f"INPUT_DIR: '{INPUT_DIR}'")

        #  sample_name を定義
        sample_win_resolve = Path(INPUT_DIR).resolve()
        sample_name = str(sample_win_resolve.parent.name) + "-" + str(sample_win_resolve.name)
        logger.debug(f"sample_name: '{sample_name}'")

        logger.info("setting save path ...")

        img_suffix = f"__{sample_name}{image_ext}"
        isoline_path = SAVE_DIR / f"isoline{img_suffix}"
        utils.check_exist_mkdir(isoline_path)
        logger.debug(f"isoline_path: '{isoline_path}'")

        bounded_path = SAVE_DIR / f"bounded{img_suffix}"
        utils.check_exist_mkdir(bounded_path)
        logger.debug(f"bounded_path: '{bounded_path}'")

        with_disb_path = SAVE_DIR / f"with_disb{img_suffix}"
        utils.check_exist_mkdir(with_disb_path)
        logger.debug(f"with_disb_path: '{with_disb_path}'")

        sigma_path = SAVE_DIR / f"sigma_x10val{img_suffix}"
        utils.check_exist_mkdir(sigma_path)
        logger.debug(f"sigma_path: '{sigma_path}'")

        thresh_path = SAVE_DIR / f"thresh_uint8{img_suffix}"
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
        logger.debug(f"std  dtype: {std.dtype},range({np.min(std)},{np.max(std)})")

        logger.debug(f"sigma image size: {std.size}")

        logger.info("--- highlight based on sigma image ---")
        logger.info("calculate contours")

        sigma_image = std.copy()
        std_int = std.astype(np.uint16)
        std_hist = histogram(std_int)

        if hist_show:
            fig, ax = plt.subplots()

            fig.tight_layout(rect=[0, 0.18, 1, 1])

            x = list(range(len(std_hist)))
            init_bin = 50

            def redraw(binn):
                current_scale = ax.get_yscale()
                ax.cla()  # 前の描画をクリア
                ax.hist(
                    x,
                    bins=int(binn),
                    weights=std_hist,
                    color="purple",
                    alpha=0.2,
                    label=f"Histogram (bins={int(binn)})",
                )
                ax.plot(x, std_hist, alpha=0.4, color="blue", label="histogram")
                ax.set_yscale(current_scale)  # 対数/線形スケールを維持
                ax.legend(loc="upper right")

            # 初回の描画
            redraw(init_bin)

            ax_slider = plt.axes([0.15, 0.08, 0.55, 0.04])
            slider_bin = Slider(
                ax=ax_slider,
                label="Bins ",
                valmin=1,
                valmax=len(std_hist),  # 最大値はデータ長に合わせて調整可能
                valinit=init_bin,
                valstep=1,
                valfmt="%d",
            )

            ax_button = plt.axes([0.78, 0.07, 0.17, 0.06])
            btn_log = Button(ax_button, "Toggle Log")

            # イベントハンドラ
            def update_bin(val):
                redraw(val)
                fig.canvas.draw_idle()

            def toggle_log(event):
                if ax.get_yscale() == "log":
                    ax.set_yscale("linear")
                else:
                    ax.set_yscale("log", base=10)
                fig.canvas.draw_idle()

            slider_bin.on_changed(update_bin)
            btn_log.on_clicked(toggle_log)

            plt.show()

        # thresh_uint8 を作製
        _, thresh = cv2.threshold(std, sigma_threshold, 255, cv2.THRESH_BINARY)
        thresh_uint8 = thresh.astype(np.uint8)

        # contours を取得
        contours, _ = cv2.findContours(thresh_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        logger.debug(f"num of contours  : {len(contours)}")

        logger.info("bounding and inspecting contours (bboxstat = (x,y,w,h,area))")

        un_compat_Fcnt = []
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
                un_compat_Fcnt.append(pts)

        logger.debug(f"Number of defective items : {len(un_compat_Fcnt)} / {len(contours)}")

        logger.info("--- draw highlights on the image. ---")
        logger.info("normalize image")

        background_img = utils.scale_to_uint8(mean, float_range=(0.0, 2.0**16), color_channel=True)

        bounded_img = cv2.drawContours(background_img.copy(), bound_Fcnt, -1, lineC_BGR, line_thickness)

        withDisb_img = drawContours_alpha(bounded_img, un_compat_Fcnt, un_compat_lineC_BGRA, line_thickness)

        isoline_img = cv2.drawContours(background_img, contours, -1, lineC_BGR, line_thickness)

        if cv2.imwrite(bounded_path.resolve(), bounded_img):
            logger.info("save bounded_img image sucessful")
        else:
            logger.warning("bounded_img image imwrite failed")

        if cv2.imwrite(with_disb_path.resolve(), withDisb_img):
            logger.info("save with_disb image sucessful")
        else:
            logger.warning("with_disb image imwrite failed")

        if cv2.imwrite(isoline_path.resolve(), isoline_img):
            logger.info("save isoline_img image sucessful")
        else:
            logger.warning("isoline_img image imwrite failed")

        if utils.save_scaled_std_tiff(sigma_path.resolve(), sigma_image, scale_factor=10):
            logger.info("save sigma_img image sucessful")
        else:
            logger.warning("sigma_img image imwrite failed")

        if cv2.imwrite(thresh_path.resolve(), thresh_uint8):
            logger.info("save thresh_uint8 image sucessful")
        else:
            logger.warning("thresh_uint8 imwrite failed")
