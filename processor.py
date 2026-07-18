import json
import os
import subprocess
import sys
import datetime
from pathlib import Path

version=1.0

PARENT_OUTDIR=r"O:\std_score_visualize"
Path(PARENT_OUTDIR).mkdir(parents=True, exist_ok=True)

# 日時 YYYY-MM-DD_hh-mm-ss
now_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%m-%S')

log_name = f"console_log_{now_str}.txt"

original_stdout = sys.stdout
sys.stdout = open(os.path.join(PARENT_OUTDIR,"processing_log",log_name), 'w', encoding='utf-8')

# 現在の環境変数をコピーし、subprocess用の目印を追加
my_env = os.environ.copy()
my_env["RUN_BY_SUBPROCESS"] = "true"

input_dirs = [
    
    r""
]

params = {
    # パラメータ std score
    "INPUT_DIR": "./sun_images",  # 処理対象の画像フォルダ
    "CROP_H": 800,  # 抽出する画像サイズ(縦幅)
    "CROP_W": 800,  # 抽出する画像サイズ(横幅)
    "OUT_DIR": "./output_pixels",  # CSV保存先フォルダ
    # パラメータ colormap
    "USE_CSV": False,  # True: CSVを読み込む / False: 偏差値算出チームの変数を使用
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
    sub_folders = [p for p in base_path.iterdir() if p.is_dir()]
    for ii, dirpath in sub_folders:
        print(f"[INFO] from processor:({i + 1}/{len(sub_folders)})'{dirpath}'")
        dirname = Path(dirpath).name

        dir_params = params.copy()
        dir_params["INPUT_DIR"] = dirpath
        dir_params["OUTPUT_NAME"] = dirname
        dir_params["MEAN_IMAGE_NAME"] = dirname + "_MEAN"
        dir_params["STD_IMAGE_NAME"] = dirname + "_STD"

        OUTPUT_DIRS = os.path.join(PARENT_OUTDIR,"output", basename)
        dir_params["OUTPUT_DIR"] = OUTPUT_DIRS
        dir_params["MEAN_STD_OUTPUT_DIR"] = OUTPUT_DIRS
        Path(OUTPUT_DIRS).mkdir(parents=True, exist_ok=True)

        json_payload = json.dumps(dir_params)
        try:
            subprocess.run(
                ["python", "child.py"], input=json_payload, text=True, env=my_env
            )
        except Exception as e:
            print(f"[ERROR] from processor:{e}")
            EroNum += 1
print(
    f"[INFO] from processor: finish all processing (successful={len(input_dirs) - EroNum}/{len(input_dirs)})"
)
