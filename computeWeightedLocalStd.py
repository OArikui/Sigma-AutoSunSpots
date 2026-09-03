import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_weighted_local_std(
    image: np.ndarray,
    r: int,
    weight_model: str = "gaussian",
    use_circle: bool = True,
    sigma: float | None = None,
    debug_show: bool = False,
) -> np.ndarray:
    """
    2Dグレースケール画像の各ピクセルに対し、選択した形状と距離減衰モデルで
    重み付けした局所標準偏差を計算します。

    Args:
        image (np.ndarray): 入力グレースケール画像 (2次元配列)
        r (int): 考慮する最大半径 (ピクセル単位, カーネルサイズは 2r+1 x 2r+1)
        weight_model (str): 減衰モデル ('gaussian', 'inverse', 'linear')
        use_circle (bool): Trueで円形近傍、Falseで正方形近傍
        sigma (float, optional): 'gaussian'時の標準偏差。デフォルトは r/2
        debug_show (bool, optional): Trueのとき、計算結果をMatplotlibでブロック表示します。

    Returns:
        np.ndarray: 各要素の重み付き局所標準偏差 (float32)
    """
    if image.ndim != 2:
        raise ValueError(
            "入力画像は単一チャンネルのグレースケール画像である必要があります。"
        )

    valid_models = ["gaussian", "inverse", "linear"]
    if weight_model not in valid_models:
        raise ValueError(
            f"weight_model は {valid_models} のいずれかを指定してください。"
        )

    img_f = image.astype(np.float32)

    # 距離グリッドの生成
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    dist = np.sqrt(x**2 + y**2)

    # 減衰モデルに応じた重みカーネルの計算
    if weight_model == "gaussian":
        if sigma is None:
            sigma = r / 2.0 if r > 0 else 1.0
        kernel = np.exp(-(dist**2) / (2.0 * sigma**2))

    elif weight_model == "inverse":
        kernel = 1.0 / (1.0 + dist)

    elif weight_model == "linear":
        # 正方形モードの場合、角の最大距離(r * √2)まで減衰スケールを拡張
        max_dist = r if use_circle else r * np.sqrt(2)
        if max_dist == 0:
            kernel = np.ones_like(dist, dtype=float)
        else:
            kernel = np.maximum(0.0, 1.0 - (dist / max_dist))

    # 円形モードの場合、半径rの外側を完全にカットオフ
    if use_circle:
        kernel[dist > r] = 0.0

    # カーネルの正規化 (和を1.0にする)
    k_sum = kernel.sum()
    if k_sum > 0:
        kernel /= k_sum

    kernel_f32 = kernel.astype(np.float32)

    # OpenCVによる高速な畳み込み計算
    mean_w = cv2.filter2D(img_f, -1, kernel_f32, borderType=cv2.BORDER_REFLECT)
    mean_sq_w = cv2.filter2D(img_f**2, -1, kernel_f32, borderType=cv2.BORDER_REFLECT)

    # 重み付き分散と標準偏差の算出 V = E[X^2] - (E[X])^2
    variance_w = mean_sq_w - (mean_w**2)

    # 浮動小数点誤差の補正
    variance_w = np.maximum(variance_w, 0.0)
    std_dev_w = np.sqrt(variance_w)

    # デバッグ表示機能
    if debug_show:
        plt.figure(figsize=(8, 6))
        plt.title(f"Local Standard Deviation (r={r}, model={weight_model})")
        # 輝度のばらつき度合いが見やすいようにカラーマップ(viridis等)を適用
        plt.imshow(std_dev_w, cmap="viridis")
        plt.colorbar(label="Standard Deviation")
        plt.show(block=True)

    return std_dev_w
