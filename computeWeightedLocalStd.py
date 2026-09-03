import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def compute_weighted_local_std(
    image: np.ndarray,
    r: int,
    weight_model: str = "gaussian",
    use_circle: bool = True,
    sigma: float | None = None,
    debug_show: bool = False,
    debug_pos: tuple[int, int] | None = None,
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
        debug_pos (tuple[int, int], optional): デバッグ表示する対象のピクセル座標 (X, Y)。
                                               省略時は画像の中心を使用します。

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
    h, w = image.shape

    # 距離グリッドの生成
    y_grid, x_grid = np.ogrid[-r : r + 1, -r : r + 1]
    dist = np.sqrt(x_grid**2 + y_grid**2)

    # 減衰モデルに応じた重みカーネルの計算
    if weight_model == "gaussian":
        if sigma is None:
            sigma = r / 2.0 if r > 0 else 1.0
        kernel = np.exp(-(dist**2) / (2.0 * sigma**2))

    elif weight_model == "inverse":
        kernel = 1.0 / (1.0 + dist)

    elif weight_model == "linear":
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

    # OpenCVによる高速な畳み込み計算 (BORDER_REFLECTで境界の折り返し処理)
    mean_w = cv2.filter2D(img_f, -1, kernel_f32, borderType=cv2.BORDER_REFLECT)
    mean_sq_w = cv2.filter2D(img_f**2, -1, kernel_f32, borderType=cv2.BORDER_REFLECT)

    # 重み付き分散と標準偏差の算出
    variance_w = mean_sq_w - (mean_w**2)
    variance_w = np.maximum(variance_w, 0.0)
    std_dev_w = np.sqrt(variance_w)

    # デバッグ・サンプリングの視覚化処理
    if debug_show:
        # debug_posが指定されていない場合は画像中心をターゲットにする
        if debug_pos is None:
            cx, cy = w // 2, h // 2
        else:
            cx, cy = debug_pos
            # 座標が画像範囲内に収まるようクリップ
            cx = int(np.clip(cx, 0, w - 1))
            cy = int(np.clip(cy, 0, h - 1))

        # filter2Dと同じ BORDER_REFLECT の条件で、該当座標の周辺ピクセルを抽出
        img_padded = cv2.copyMakeBorder(img_f, r, r, r, r, cv2.BORDER_REFLECT)
        patch = img_padded[cy : cy + 2 * r + 1, cx : cx + 2 * r + 1]

        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"Local Standard Deviation Sampling Analysis", fontsize=14)

        # 1. 太陽像全体とターゲット位置
        axs[0].imshow(image, cmap="gray", origin="upper")
        axs[0].plot(cx, cy, "r+", markersize=12, markeredgewidth=2)
        # 参照している範囲を示す赤い枠線を描画
        rect = patches.Rectangle((cx - r, cy - r), 2 * r, 2 * r, 
                                 linewidth=1.5, edgecolor="r", facecolor="none")
        axs[0].add_patch(rect)
        axs[0].set_title(f"Position on Sun Image\n(X={cx}, Y={cy})")

        # 2. 抽出された局所ピクセル (Target Pixels)
        im1 = axs[1].imshow(patch, cmap="gray", vmin=0, vmax=255)
        axs[1].set_title(f"Local Pixels Matrix\n(Size: {2*r+1}x{2*r+1})")
        fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        # 3. 適用される重み (Weight Kernel)
        # weightの強弱がわかりやすいようにmagmaカラーマップを使用
        im2 = axs[2].imshow(kernel_f32, cmap="magma")
        axs[2].set_title(f"Weight Values\n(Model: {weight_model})")
        fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

        # 4. 全体の移動標準偏差の算出結果マップ
        im3 = axs[3].imshow(std_dev_w, cmap="viridis")
        axs[3].plot(cx, cy, "r+", markersize=12, markeredgewidth=2)
        
        # ターゲット位置の実際の標準偏差数値をタイトルに付与
        target_std = std_dev_w[cy, cx]
        axs[3].set_title(f"Result StdDev Map\nStdDev at pos: {target_std:.2f}")
        fig.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show(block=True)

    return std_dev_w
