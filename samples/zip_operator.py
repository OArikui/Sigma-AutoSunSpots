import zipfile
import cv2
import numpy as np
"""
ZIPファイル操作用のユーティリティ関数
version: 1.1 - 16bit TIFF画像の読み込みに対応
"""

def load_image_from_zip_cv2(zip_path, image_name):
    """ZIPファイルから直接画像を読み込んでOpenCV（NumPy配列）形式で返す。

    Args:
        zip_path (str): 対象とするZIPファイルのパス。
        image_name (str): ZIP内にある画像ファイルのパス（名前）。

    Returns:
        np.ndarray: デコードされた画像データ（BGR形式のNumPy配列）。
    """
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # バイナリデータを取得
        img_bytes = zf.read(image_name)
        
        # バイナリデータを一時的に1次元のNumPy配列（バイトデータ配列）に変換
        nparr = np.frombuffer(img_bytes, np.uint8)
        
        # 画像をデコードする
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)  # 16bit画像を正しく読み込むためにIMREAD_UNCHANGEDを使用
        
    return img

def get_image_names_from_zip(zip_path, extensions:list=['.tiff']):
    """ZIPファイル内から指定された拡張子に一致する画像ファイル名の一覧を取得する。

    Args:
        zip_path (str): 対象とするZIPファイルのパス。
        extensions (list, optional): 抽出対象とする画像拡張子のリスト。デフォルトは ['.tiff']。

    Returns:
        list[str]: 条件に一致した画像ファイル名（パス）のリスト。
    """
    # 対象にしたい画像の拡張子を指定（小文字で判定するため、すべて小文字で定義）
    image_extensions = tuple(ext.lower() for ext in extensions)
    
    image_names = []
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # ZIP内のすべてのファイルパスを取得
        for file_name in zf.namelist():
            # フォルダのエントリ（末尾が / ）は除外
            if file_name.endswith('/'):
                continue
            
            # 拡張子が一致するものを抽出
            if file_name.lower().endswith(image_extensions):
                image_names.append(file_name)
                
    return image_names