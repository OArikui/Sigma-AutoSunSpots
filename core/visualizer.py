import json
import cv2
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from matplotlib.backends.backend_agg import FigureCanvasAgg


def heatmap_customL(
    data: np.ndarray,
    val_range: tuple[float, float],
    cmap_name: str,
    overd_cmap_name: str,
    underd_cmap_name: str = "Blues_r",
    colorbar_thick: int = 12,
    inner_ticks: bool = True,
) -> np.ndarray:
    """正方形のヒートマップの左縁と下縁に沿って、右下->左下->左上の順で

    値が高くなるL字型の特殊なカラーバーを配置した画像を生成します。
    data.dtype が整数型の場合は目盛りテキストを整数表記にします。
    """
    # 0. 入力データの次元調整・dtype判定
    # 画像のdtypeが整数型(int8, uint8, int32, uint16等)かどうかの判定
    is_integer = np.issubdtype(data.dtype, np.integer)

    if data.ndim == 3:
        if data.shape[2] in (3, 4):
            data = np.mean(data[:, :, :3], axis=2)
        else:
            data = data[:, :, 0]

    vmin, vmax = val_range
    data_min = float(np.nanmin(data))
    data_max = float(np.nanmax(data))

    full_min = min(vmin, data_min)
    full_max = max(vmax, data_max)

    has_underflow = full_min < vmin
    has_overflow = full_max > vmax

    # 1. カラーマップの作成 (アンダー/オーバーフロー結合)

    if total_range > 0 and (has_underflow or has_overflow):
        colors_list = []

        if has_underflow:
            ratio_under = (vmin - full_min) / total_range
            n_under = max(2, int(256 * ratio_under))
            cmap_underd = plt.get_cmap(underd_cmap_name)
            colors_list.append(cmap_underd(np.linspace(0, 1, n_under)))

        v_norm_min = max(vmin, full_min)
        v_norm_max = min(vmax, full_max)
        ratio_normal = (v_norm_max - v_norm_min) / total_range
        n_normal = max(2, int(256 * ratio_normal))
        cmap_base = plt.get_cmap(cmap_name)
        colors_list.append(cmap_base(np.linspace(0, 1, n_normal)))

        if has_overflow:
            ratio_over = (full_max - vmax) / total_range
            n_over = max(2, int(256 * ratio_over))
            cmap_overd = plt.get_cmap(overd_cmap_name)
            colors_list.append(cmap_overd(np.linspace(0, 1, n_over)))

        combined_colors = np.vstack(colors_list)
        cmap = mcolors.ListedColormap(combined_colors)
    else:
        cmap = plt.get_cmap(cmap_name).copy()

    cmap.set_bad(color="white", alpha=0)

    # 描画用のフィギュアを作成
    fig = plt.figure(figsize=(8, 8), dpi=100)
    canvas = FigureCanvasAgg(fig)

    R = max(0.01, min(colorbar_thick / 100.0, 0.4))
    ax_cbar = fig.add_axes([0.0, 0.0, 1.0, 1.0])

    # 2. L字型カラーバーのグラデーションを作成
    N = 1000
    T = int(N * R)
    X, Y = np.meshgrid(np.arange(N), np.arange(N))

    S = 2 * N - 2 - X - Y
    cbar_gradient = full_min + (S / (2 * N - 2)) * total_range

    mask = (X >= T) & (Y < N - T)
    cbar_gradient = np.where(mask, np.nan, cbar_gradient)

    ax_cbar.imshow(
        cbar_gradient,
        cmap=cmap,
        origin="upper",
        extent=[0, 1, 0, 1],
        vmin=full_min,
        vmax=full_max,
    )
    ax_cbar.axis("off")

    # 3. 重なり防止機能付き目盛り管理・間引き処理
    candidate_ticks = []

    # 3-1. 標準の等間隔目盛り
    default_s_list = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    for s in default_s_list:
        val = full_min + s * total_range
        candidate_ticks.append({"s": s, "val": val, "priority": 1, "tag": "default"})

    # 3-2. 特殊境界・実測値目盛り
    if total_range > 0:
        if has_underflow:
            s_vmin = (vmin - full_min) / total_range
            candidate_ticks.append({"s": s_vmin, "val": vmin, "priority": 2, "tag": "vmin"})

        if has_overflow:
            s_vmax = (vmax - full_min) / total_range
            candidate_ticks.append({"s": s_vmax, "val": vmax, "priority": 2, "tag": "vmax"})

        if data_min > vmin:
            s_dmin = (data_min - full_min) / total_range
            candidate_ticks.append({"s": s_dmin, "val": data_min, "priority": 2, "tag": "min"})

        if data_max < vmax:
            s_dmax = (data_max - full_min) / total_range
            candidate_ticks.append({"s": s_dmax, "val": data_max, "priority": 2, "tag": "max"})

    candidate_ticks.sort(key=lambda item: item["s"])

    # 3-3. 近接目盛りの自動間引き
    min_s_dist = 0.07
    filtered_ticks = []

    for tick in candidate_ticks:
        if not filtered_ticks:
            filtered_ticks.append(tick)
            continue

        prev = filtered_ticks[-1]
        dist = tick["s"] - prev["s"]

        if dist < min_s_dist:
            if tick["priority"] > prev["priority"]:
                filtered_ticks[-1] = tick
            elif (
                tick["priority"] == prev["priority"]
                and tick["tag"] in ("min", "max", "vmin", "vmax")
                and prev["tag"] not in ("min", "max", "vmin", "vmax")
            ):
                filtered_ticks[-1] = tick
        else:
            filtered_ticks.append(tick)

    # 4. カラーバー内数値・目盛り線の描画
    def add_embedded_text(x, y, text_str, is_boundary=False):
        text_color = "#FFD700" if is_boundary else "white"
        font_size = 10 if is_boundary else 8
        stroke_width = 3.0 if is_boundary else 2.5

        txt = ax_cbar.text(
            x,
            y,
            text_str,
            color=text_color,
            fontsize=font_size,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=10,
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=stroke_width, foreground="black")])

    # 目盛り文字列のフォーマット関数（int判断）
    def get_label_text(tick_item):
        val = tick_item["val"]
        tag = tick_item["tag"]

        # dtype が整数型の場合は小数点以下を削除（四捨五入して整数化）
        val_str = f"{int(round(val))}" if is_integer else f"{val:.1f}"

        if tag == "min":
            return f"min:{val_str}"
        if tag == "max":
            return f"max:{val_str}"
        return val_str

    pad_margin = 0.05

    for tick in filtered_ticks:
        s = tick["s"]
        tag = tick["tag"]
        label_str = get_label_text(tick)
        is_boundary = tag in ("vmin", "vmax")

        line_color = "#FFD700" if is_boundary else "white"
        line_width = 2.0 if is_boundary else 1.0

        if s <= 0.5:
            # --- 下辺カラーバー内への描画 ---
            xt = 1.0 - 2.0 * s
            x_end = min(1.0, xt + R)
            y_end = x_end - xt

            if inner_ticks:
                ax_cbar.plot(
                    [xt, x_end],
                    [0, y_end],
                    color=line_color,
                    linewidth=line_width,
                    zorder=6 if is_boundary else 5,
                )

            tx = (xt + x_end) / 2.0
            ty = y_end / 2.0

            tx = np.clip(tx, pad_margin, 1.0 - pad_margin)
            ty = max(pad_margin, ty)

            add_embedded_text(tx, ty, label_str, is_boundary=is_boundary)
        else:
            # --- 左辺カラーバー内への描画 ---
            yt = 2.0 * (s - 0.5)
            y_end = min(1.0, yt + R)
            x_end = y_end - yt

            if inner_ticks:
                ax_cbar.plot(
                    [0, x_end],
                    [yt, y_end],
                    color=line_color,
                    linewidth=line_width,
                    zorder=6 if is_boundary else 5,
                )

            tx = x_end / 2.0
            ty = (yt + y_end) / 2.0

            tx = max(pad_margin, tx)
            ty = np.clip(ty, pad_margin, 1.0 - pad_margin)

            add_embedded_text(tx, ty, label_str, is_boundary=is_boundary)

    # L字型カラーバー外枠線
    x_outline = [0.0, 1.0, 1.0, R, R, 0.0, 0.0]
    y_outline = [0.0, 0.0, R, R, 1.0, 1.0, 0.0]
    ax_cbar.plot(x_outline, y_outline, color="white", linewidth=2.0, zorder=20)

    # 5. メインのヒートマップ画像の設定
    ax_main = fig.add_axes([R, R, 1.0 - R, 1.0 - R])
    ax_main.imshow(
        data,
        cmap=cmap,
        vmin=full_min,
        vmax=full_max,
        origin="lower",
        aspect="auto",
    )
    ax_main.axis("off")

    # 6. 画像配列(np.ndarray)として取得して返す
    canvas.draw()
    img_arr = np.asarray(canvas.buffer_rgba())
    plt.close(fig)

    return img_arr


if __name__ == "__main__":
    """np.random.seed(42)
    dummy_data = 20.0 + np.random.rand(50, 50) * 60.0 + 100
"""
    dummy_data, _ = utils.load_scaled_std_tiff(
        r"E:\projects\AutoSunsSpots\Sigma_AutoSunSpots\save\isoline_highlight\sigma_x10val__samples-2025-07-20-PL1.zip.png",
        False,
    )
    val_range = (0, 3690)
    cmap_name = "hot"
    overd_cmap_name = "cool"
    colorbar_thick = 10

    result_image = heatmap_customL(
        data=dummy_data,
        val_range=val_range,
        cmap_name=cmap_name,
        overd_cmap_name=overd_cmap_name,
        underd_cmap_name=overd_cmap_name,
        colorbar_thick=colorbar_thick,
        inner_ticks=True,
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(result_image)
    plt.axis("off")
    plt.title("Heatmap with Full Colorbar Outline")
    plt.show()
