import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDarray

"""
ZIPファイル操作用のユーティリティ関数
"""
__version__ = 2.1  # 型ヒントを追加


def load_image_from_path_cv2(dirpath: str, image_name: str) -> NDarray[np.uint16]:
    """ファイルシステムから直接画像を読み込んでOpenCV（NumPy配列）形式で返す。

    Args:
        image_path (str): 対象とする画像ファイルのパス。

    Returns:
        np.ndarray: デコードされた画像データ（BGR形式のNumPy配列）。
    """
    image_path: str = os.path.join(dirpath, image_name)
    # 16bit画像を正しく読み込むためにIMREAD_UNCHANGEDを使用
    return cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)


def get_image_names_from_dir(
    directory_path: str, extensions: list[str] | None = None
) -> list[str]:
    """指定したディレクトリ（およびサブディレクトリ）から指定された拡張子に一致する画像ファイル名の一覧を取得する。

    Args:
        directory_path (str): 検索対象とするディレクトリのパス。
        extensions (list, optional): 抽出対象とする画像拡張子のリスト。デフォルトは ['.tiff']。

    Returns:
        list[str]: 条件に一致した画像ファイルのフルパスリスト。
    """
    if extensions is None:
        extensions = [".tiff"]
    # 検索対象の拡張子を小文字のタプルに変換
    image_extensions: tuple = tuple(ext.lower() for ext in extensions)

    image_paths: list = []

    # pathlibを使用してディレクトリ内を再帰的に検索[cite: 1]
    for path in Path(directory_path).rglob("*"):
        # ファイルかつ拡張子が一致するもののみを抽出
        if path.is_file() and path.suffix.lower() in image_extensions:
            image_paths.append(str(path))

    return image_paths


def load_image_from_zip_cv2(
    zip_path: str, image_name: str
) -> NDarray[NDarray[np.uint16]]:
    """ZIPファイルから直接画像を読み込んでOpenCV（NumPy配列）形式で返す。

    Args:
        zip_path (str): 対象とするZIPファイルのパス。
        image_name (str): ZIP内にある画像ファイルのパス（名前）。

    Returns:
        np.ndarray: デコードされた画像データ（BGR形式のNumPy配列）。
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        # バイナリデータを取得
        img_bytes: bytes = zf.read(image_name)

        # バイナリデータを一時的に1次元のNumPy配列（バイトデータ配列）に変換
        nparr: bytes = np.frombuffer(img_bytes, np.uint8)

        # 画像をデコードする
        img: NDarray[np.uint16] = cv2.imdecode(
            nparr, cv2.IMREAD_UNCHANGED
        )  # 16bit画像を正しく読み込むためにIMREAD_UNCHANGEDを使用

    return img


def get_image_names_from_zip(
    zip_path: str, extensions: list[str] | None = None
) -> list[str]:
    """ZIPファイル内から指定された拡張子に一致する画像ファイル名の一覧を取得する。

    Args:
        zip_path (str): 対象とするZIPファイルのパス。
        extensions (list, optional): 抽出対象とする画像拡張子のリスト。default = ['.tiff']。

    Returns:
        list[str]: 条件に一致した画像ファイル名（パス）のリスト。
    """
    if extensions is None:
        extensions = [".tiff"]
    # 対象にしたい画像の拡張子を指定（小文字で判定するため、すべて小文字で定義）
    image_extensions: tuple = tuple(ext.lower() for ext in extensions)

    image_names: list = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        # ZIP内のすべてのファイルパスを取得
        for file_name in zf.namelist():
            # フォルダのエントリ（末尾が / ）は除外
            if file_name.endswith("/"):
                continue

            # 拡張子が一致するものを抽出
            if file_name.lower().endswith(image_extensions):
                image_names.append(file_name)

    return image_names
