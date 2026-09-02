import numpy as np


def correct_limb_darkening(image_16bit, center_x, center_y, radius, u=0.6):
    """
    16bitグレースケール画像の太陽の周縁減光を補正する関数

    Parameters:
    -----------
    image_16bit : numpy.ndarray
        入力画像（16bitグレースケール, dtype=np.uint16）
    center_x : float or int
        太陽の中心のX座標（ピクセル）
    center_y : float or int
        太陽の中心のY座標（ピクセル）
    radius : float or int
        太陽の半径（ピクセル）
    u : float
        線形周縁減光係数（可視光の太陽の場合、およそ0.5〜0.6程度）

    Returns:
    --------
    numpy.ndarray
        補正後の16bitグレースケール画像
    """
    # 計算時のオーバーフローを防ぐため、float32に変換
    img_float = image_16bit.astype(np.float32)
    h, w = img_float.shape

    # 画像全体のX座標、Y座標のグリッドを生成
    Y, X = np.indices((h, w))

    # 中心からの距離 r を計算
    r = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

    # 太陽の円盤内（r <= radius）を示すマスク
    disk_mask = r <= radius

    # 視線角度のコサイン（μ = cosθ）を計算
    # 縁（r=radius）でμ=0、中心（r=0）でμ=1となる
    r_normalized = np.clip(r / radius, 0, 1.0)
    mu = np.sqrt(1.0 - r_normalized**2)

    # 線形周縁減光のプロファイルを計算: I(θ)/I(0) = 1 - u + u * μ
    profile = 1.0 - u + u * mu

    # ゼロ割りや極端な強調を防ぐためのクリッピング（uが1に近い場合のフェイルセーフ）
    profile = np.clip(profile, 1e-5, 1.0)

    # 補正画像の初期化（元画像をコピー）
    corrected_img = np.copy(img_float)

    # 太陽の円盤内のみ、プロファイルで割ることで減光を補正（縁を明るく持ち上げる）
    corrected_img[disk_mask] = img_float[disk_mask] / profile[disk_mask]

    # 16bitの最大値(65535)を超過したピクセルをクリッピングし、元の型に戻す
    corrected_img = np.clip(corrected_img, 0, 65535).astype(np.uint16)

    return corrected_img
