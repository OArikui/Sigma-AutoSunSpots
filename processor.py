import json
import os
import subprocess
import sys
import datetime
from pathlib import Path

version=1.0

PARENT_OUTDIR = r"O:\std_score_visualize"

# 1. ログ保存先のディレクトリパス（processing_log）を定義
log_dir = os.path.join(PARENT_OUTDIR, "processing_log")

# 2. processing_log フォルダを作成
Path(log_dir).mkdir(parents=True, exist_ok=True)

# 3. 日時 YYYY-MM-DD_hh-mm-ss
now_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_name = f"console_log_{now_str}.txt"

# 4. 画面とファイルの両方に出力するためのクラスを定義
class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout  # 元のコンソール出力を保持
        self.log = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)  # 画面に出力
        self.log.write(message)       # ファイルに出力
        self.log.flush()              # リアルタイムでファイルに書き込み反映させる

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# sys.stdout を DualLogger に置き換える
log_path = os.path.join(log_dir, log_name)
sys.stdout = DualLogger(log_path)
# 現在の環境変数をコピーし、subprocess用の目印を追加
my_env = os.environ.copy()
my_env["RUN_BY_SUBPROCESS"] = "true"

input_dirs = [
    
    r"J:\2025-07-20\2025-07-20pic-LT",
    r"J:\2025-07-20\2025-07-20pic-PL",
    r"J:\2025-08-30\2025-08-30pic-LT",
    r"J:\2025-08-30\2025-08-30pic-PL",
    r"J:\2026-01-17Z\2026-1-17pic-LT",
    r"J:\2026-01-17Z\2026-1-17pic-PL",
    r"J:\2025-12-26Z\2025-12-26pic-LT",
    r"J:\2026-01-12Z\2026-01-12pic-LT",
    r"J:\2026-02-01Z\2026-02-01LT1",
    r"J:\2026-07-10\2026-07-10pic\2026-07-10pic_chosen"
    
    
]

params = {
    "OUTPUT_MODE":"VIDEO",
    # パラメータ std score
    "INPUT_DIR": "./sun_images",  # 処理対象の画像フォルダ
    "CROP_H": 800,  # 抽出する画像サイズ(縦幅)
    "CROP_W": 800,  # 抽出する画像サイズ(横幅)
    "OUT_DIR_CSV": "./output_pixels",  # CSV保存先フォルダ
    # パラメータ colormap
    "DEBUG": True,  # True: デバッグ情報を表示
    "OUTPUT_DIR": r"",  # 出力動画の保存先フォルダ
    "OUTPUT_NAME": "output_test_sample",  # 出力動画のファイル名（拡張子なし）
    "OUTPUT_EXT": ".mp4",  # 出力動画の拡張子
    "VIDEO_CODEC": "mp4v",  # 動画コーデック
    "FPS": 60.0,  # 出力動画のフレームレート
    "MEAN_STD_OUTPUT_DIR": r"",  # 平均と標準偏差の出力画像の保存先フォルダ
    "MEAN_IMAGE_NAME": "mean_image",  # 平均値の出力画像のファイル名
    "STD_IMAGE_NAME": "std_image",  # 標準偏差の出力画像のファイル名
    "IMAGE_EXT": ".png",  # 平均と標準偏差の出力画像の拡張子
}  # 環境変数を指定して子スクリプトを実行

EroNum = 0
SuccessNum = 0
print("[INFO] from processor:start processing")
for i, base_path in enumerate(input_dirs):
    print(
        f"[INFO] from processor: start parent dir ({i + 1}/{len(input_dirs)})'{base_path}'"
    )
    basename = Path(base_path).name.replace("pic", "")
    
    # 修正1: Path() で囲んでから iterdir() を呼ぶ
    sub_folders = [p for p in Path(base_path).iterdir() if p.is_dir()]
    
    # 修正2: enumerate を使ってインデックスを取得する
    for ii, dirpath in enumerate(sub_folders):
        print(f"[INFO] from processor:({ii + 1}/{len(sub_folders)})'{dirpath}'")
        dirname = dirpath.name # Pathオブジェクトなので .name が使える

        dir_params = params.copy()
        # 修正3: Pathオブジェクトを文字列にキャストしてからJSONの辞書に格納
        dir_params["INPUT_DIR"] = str(dirpath)
        dir_params["OUTPUT_NAME"] = dirname
        dir_params["MEAN_IMAGE_NAME"] = dirname + "_MEAN"
        dir_params["STD_IMAGE_NAME"] = dirname + "_STD"

        OUTPUT_DIRS = os.path.join(PARENT_OUTDIR,"output", basename)
        dir_params["OUTPUT_DIR"] = OUTPUT_DIRS+"_video"
        dir_params["OUT_DIR"] = OUTPUT_DIRS+"_csv"
        dir_params["MEAN_STD_OUTPUT_DIR"] = OUTPUT_DIRS
        Path(OUTPUT_DIRS).mkdir(parents=True, exist_ok=True)

        json_payload = json.dumps(dir_params)
        try:
            subprocess.run(
                ["python", "std_score_visualize.py"], input=json_payload, text=True, env=my_env,check=True
            )
            SuccessNum+=1
        except Exception as e:
            print(f"[ERROR] from processor:{e}")
            EroNum += 1
print(
    f"[INFO] from processor: finish all processing (successful={SuccessNum}/{EroNum+SuccessNum})"
)
