import cv2
import numpy as np
import tqdm
from numpy.typing import NDArray

from core.optImg import crop_and_pad
from core.mods.zip_operator import get_image_names_from_dir,get_image_names_from_zip,load_image_from_path_cv2,load_image_from_zip_cv2
from core.mods.MIN2ver2 import MIN2_ignore_sunspots,version

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
    logger.debug(f"getted param : {getted_param}")

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
) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[NDArray[np.float64]]

]:
    """平均画像・標準偏差画像・偏差値画像を計算する。"""

    # 平均画像
    mean: NDArray[np.float64] = np.mean(frames, axis=0)

    # 標準偏差画像
    std: NDArray[np.float64] = np.std(frames, axis=0)

    # 偏差値画像
    deviation: NDArray[np.float64] = np.where(std == 0, 50, 50 + 10 * (frames - mean) / std)

    return mean, std, deviation

