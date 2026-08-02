# processor.py リファレンス (v1.0)

2026.07.31

> 本書は [`processor.py`](assets\archive\processor.md) (version = 1.0) を用いた太陽画像バッチ処理の実行手順と、`std_score_visualize.py` などのサブプロセスへパラメータを渡す仕組みを解説するリファレンスです。用語の補足を本文中に挿入しています。

- 対象バージョン: **processor.py v1.0**
- 並走サンプル(参考・実装例であり API ではない)
  - [`std_score_visualize.py`](assets\archive\std_score_visualize.md)
  - [`std_visualize_100frame.py`](assets\archive\std_visualize_100frame.md)
  - [`processor.py`](assets\archive\processor.md) 自体が起動する子プロセスの例

---

## 目次

1. [概要](#概要)
2. [前提環境](#前提環境)
3. [実行方法](#実行方法)
4. [パラメータ一覧](#パラメータ一覧)
5. [ログ出力の仕組み](#ログ出力の仕組み)
6. [サブプロセスへのパラメータ受け渡し機序](#サブプロセスへのパラメータ受け渡し機序)
7. [実装例(添付スクリプト)](#実装例添付スクリプト)
8. [注意点・よくあるエラー](#注意点よくあるエラー)
9. [用語集](#用語集)

---

## 概要

`processor.py` は、太陽観測画像が保存された複数の親ディレクトリを順番に処理するための **ドライバ(親スクリプト)** です。

処理の大まかな流れは以下のとおりです。

1. ログ保存用ディレクトリ `.\std_score_visualize\processing_log` を準備する
2. 設定済みの入力ディレクトリ群 (`input_dirs`) を 1 つずつ走査する
3. 各親ディレクトリ直下のサブフォルダを順に開き、それぞれに対して子プロセス `std_score_visualize.py` を起動する
4. 子プロセスには JSON 形式のパラメータを **標準入力(stdin)** 経由で渡し、終了コードが 0 になることを確認する
5. すべての処理が終わった段階で、成功数 / 失敗数を集計してログに出力する

> [!NOTE]
> 
> - 「ドライバ」とは「他のプログラムを起動して管理する司令塔役のスクリプト」のことです。
> - 「サブプロセス」とは、親の Python から `subprocess.run(...)` で起動する別の Python プロセスのことです。
> - 「標準入力(stdin)」とは、キーボード入力の代わりに文字列を渡すことができるパイプのようなものです。

---

## 前提環境

### 必要なランタイム

| 項目            | 内容                                                                                          |
| ------------- | ------------------------------------------------------------------------------------------- |
| Python インタプリタ | `python` コマンドで起動できるバージョン (PATH が通っていること)                                                    |
| OS            | `O:\` および `J:\` ドライブがマウントされた Windows (astro-pc)                                             |
| 依存ライブラリ       | 子プロセス側スクリプトの依存 (`cv2`, `numpy`, `tqdm`, `MIN2ver2`, `CSV_frames`, `samples.zip_operator` 等) |
| 親子配置          | `processor.py` と `std_score_visualize.py`など は **同じ作業ディレクトリ** に置く                            |

> [!WARN]
> 
> `subprocess.run(["python", "std_score_visualize.py"], ...)` はファイル名のみを指定するため、`cwd`(現在の作業ディレクトリ)に両ファイルが存在しないと起動できません。

### 動作確認済みの想定構成

```text
.\std_score_visualize\                  # PARENT_OUTDIR (出力ルート)
└── processing_log\
    └── console_log_YYYY-MM-DD_HH-MM-SS.txt

J:\Observed_result\YYYY-MM-DD\YYYY-MM-DDpic-(LT or PL)\          # 入力ルート (input_dirs の中身)
├── sub_folder_01\
├── sub_folder_02\
└── ...
```

---

## 実行方法

最もシンプルな起動手順:

```bash
python processor.py
```

> [!NOTE] 
> ターミナル(コマンドプロンプト / PowerShell)を開き、`processor.py` を保存したフォルダへ `cd` で移動してから上記を実行します。
> 
> ```text
> cd C:\path\to\where\processor.py\is
> python processor.py
> ```

実行中は `console_log_<日時>.txt` に **画面と同じ内容** が逐次書き込まれます。エラー発生時もログファイルから原因を追跡できます。

---

## パラメータ一覧

`processor.py` 自身が定義する **設定可能な定数およびデータ構造** をまとめます。記載なき値はすべて「未定義」(=`processor.py` 内で明示されていない)です。

### 1. グローバル定数

| 名前              | 型       | 既定値 (v1.0)                 | 必須  | 説明                              |
| --------------- | ------- | -------------------------- | --- | ------------------------------- |
| `version`       | `float` | `1.0`                      | ◯   | スクリプトのバージョン番号。実装上の挙動には影響しない識別子。 |
| `PARENT_OUTDIR` | `str`   | `r".\std_score_visualize"` | ◯   | すべての出力(ログ・動画・画像)が作られる親ディレクトリ。   |

### 2. 入力ディレクトリ一覧: `input_dirs`

| 要素        | 内容                                                            |
| --------- | ------------------------------------------------------------- |
| 型         | `list[str]`                                                   |
| 個数 (v1.0) | 10 個                                                          |
| 用途        | 各エントリは「観測日 + 観測タイプ(LT/PL)」を表す親フォルダパス                          |
| 走査方法      | `Path(base_path).iterdir()` でサブフォルダを列挙し、**ディレクトリのみ** を処理対象とする |

```python
input_dirs = [
    r"J:\2025-07-20\2025-07-20pic-LT",
    r"J:\2025-07-20\2025-07-20pic-PL",
    # ...(全 10 件)
]
```

> [!NOTE]
> `r"..."` は **生の文字列リテラル** を意味します。Windows の `\` をエスケープせずそのままパスとして扱える書き方です。

### 3. サブプロセスへ渡す既定パラメータ: `params`

サブプロセスへ JSON として送る既定値の表です。**ループ内で 4 つのキーが上書き** されます。

| キー                    | 型       | 既定値 (v1.0)             | 上書き      | 内容                                             |
| --------------------- | ------- | ---------------------- | -------- | ---------------------------------------------- |
| `INPUT_DIR`           | `str`   | `"./sun_images"`       | ◯ (ループ内) | 処理対象の画像フォルダ。ループで対象サブフォルダの絶対パスに差し替えられる。         |
| `CROP_H`              | `int`   | `800`                  | ×        | 抽出する画像サイズ(縦幅, px)。                             |
| `CROP_W`              | `int`   | `800`                  | ×        | 抽出する画像サイズ(横幅, px)。                             |
| `OUT_DIR`             | `str`   | `"./output_pixels"`    | ×        | CSV 保存先フォルダ。                                   |
| `USE_CSV`             | `bool`  | `False`                | ×        | CSV を読むか (`True`) / 偏差値算出チームの変数を使うか (`False`)。 |
| `DEBUG`               | `bool`  | `True`                 | ×        | デバッグ情報の表示フラグ。                                  |
| `OUTPUT_DIR`          | `str`   | `r""` (空文字)            | ◯ (ループ内) | 出力動画/画像の保存先フォルダ。                               |
| `OUTPUT_NAME`         | `str`   | `"output_test_sample"` | ◯ (ループ内) | 出力動画のベース名(拡張子なし)。                              |
| `OUTPUT_EXT`          | `str`   | `".mp4"`               | ×        | 出力動画の拡張子。                                      |
| `VIDEO_CODEC`         | `str`   | `"mp4v"`               | ×        | OpenCV `VideoWriter_fourcc` に渡す 4 文字コード。       |
| `FPS`                 | `float` | `60.0`                 | ×        | 動画のフレームレート。                                    |
| `MEAN_STD_OUTPUT_DIR` | `str`   | `r""` (空文字)            | ◯ (ループ内) | 平均画像・標準偏差画像の保存先。                               |
| `MEAN_IMAGE_NAME`     | `str`   | `"mean_image"`         | ◯ (ループ内) | 平均画像のファイル名(拡張子なし)。                             |
| `STD_IMAGE_NAME`      | `str`   | `"std_image"`          | ◯ (ループ内) | 標準偏差画像のファイル名(拡張子なし)。                           |
| `IMAGE_EXT`           | `str`   | `".png"`               | ×        | 平均 / 標準偏差画像の拡張子。                               |

### 4. 出力パスの決定ルール

ループ内で以下の通り組み立てられます。

| 変数                                  | 組み立て式 (v1.0)                                      | 例                                             |
| ----------------------------------- | ------------------------------------------------- | --------------------------------------------- |
| `basename`                          | `Path(base_path).name.replace("pic", "")`         | `2025-07-20pic-LT` → `2025-07-20-LT`          |
| `dirname`                           | `dirpath.name` (直下サブフォルダ名)                        | `0001` など                                     |
| `OUTPUT_DIRS`                       | `os.path.join(PARENT_OUTDIR, "output", basename)` | `O:\std_score_visualize\output\2025-07-20-LT` |
| `dir_params["INPUT_DIR"]`           | `str(dirpath)`                                    | `J:\2025-07-20\2025-07-20pic-LT\0001`         |
| `dir_params["OUTPUT_NAME"]`         | `dirname`                                         | `0001`                                        |
| `dir_params["MEAN_IMAGE_NAME"]`     | `dirname + "_MEAN"`                               | `0001_MEAN`                                   |
| `dir_params["STD_IMAGE_NAME"]`      | `dirname + "_STD"`                                | `0001_STD`                                    |
| `dir_params["OUTPUT_DIR"]`          | `OUTPUT_DIRS`                                     | 上記の出力フォルダ                                     |
| `dir_params["MEAN_STD_OUTPUT_DIR"]` | `OUTPUT_DIRS`                                     | 同上                                            |

> [!NOTE]
> 
>  `os.path.join(a, b, c)` は OS ごとに適切な区切り文字(`\` か `/`)を挟んでパスを連結する関数です。Windows では `\` が使われます。

### 5. 集計用カウンタ

| 名前           | 初期値 | 役割                                   |
| ------------ | --- | ------------------------------------ |
| `EroNum`     | `0` | 失敗件数 (`subprocess` 呼び出しで例外が起きた時に +1) |
| `SuccessNum` | `0` | 成功件数 (`subprocess` が正常終了した時に +1)     |

---

## ログ出力の仕組み

`processor.py` は **標準出力 (`sys.stdout`) を独自クラス `DualLogger` に差し替える** ことで、`print(...)` メッセージを **画面とファイルの両方に同時に** 出力します。

```python
sys.stdout = DualLogger(log_path)
```

### `DualLogger` の挙動(コードからの抜粋)

| メソッド                 | 処理内容                                                                    |
| -------------------- | ----------------------------------------------------------------------- |
| `__init__(filepath)` | 元の `sys.stdout` を `self.terminal` に保持し、ログファイル(`filepath`)を `utf-8` で開く。 |
| `write(message)`     | 同じ文字列を (1) 元のコンソール (2) ログファイル の両方に書き、毎回 `flush()` する。                   |
| `flush()`            | コンソールとログファイル両方のバッファをフラッシュする。                                            |

### ログファイル名のパターン

```text
O:\std_score_visualize\processing_log\console_log_YYYY-MM-DD_HH-MM-SS.txt
```

例: `console_log_2026-07-31_14-23-01.txt`

> [!NOTE]
> 
>  `flush()` とはバッファに溜まった内容を **強制的にディスクや画面へ書き出す** ことです。これがないと急に電源が切れたときなどにログが消失します。

---

## サブプロセスへのパラメータ受け渡し機序

ここが本書の最重要ポイントです。`processor.py` は子プロセスと **次の 2 経路** で情報をやり取りします。

| 経路                                | 役割                          |
| --------------------------------- | --------------------------- |
| **環境変数 `RUN_BY_SUBPROCESS=true`** | 子プロセス側が「親から呼ばれた」と判定するためのフラグ |
| **標準入力 (`stdin`) への JSON 文字列**    | パラメータ辞書そのもの                 |

### 1. 起動 API の仕様

```python
subprocess.run(
    ["python", "std_score_visualize.py"],
    input=json_payload,
    text=True,
    env=my_env,
    check=True,
)
```

| 引数           | 設定値 (v1.0)                             | 効果                                                                              |
| ------------ | -------------------------------------- | ------------------------------------------------------------------------------- |
| 第1引数         | `["python", "std_score_visualize.py"]` | 起動コマンド。リストで渡すと `shell=False` 安全モード。                                             |
| `input`      | `json_payload` (JSON 文字列)              | 子プロセスの `stdin` に流す文字列。                                                          |
| `text=True`  | bool                                   | `input` を **テキストモード** で扱う(`bytes` ではなく `str`)。                                  |
| `env`        | `my_env`                               | 親の環境変数を引き継ぎつつ `RUN_BY_SUBPROCESS=true` を追加した辞書。                                 |
| `check=True` | bool                                   | 子プロセスの終了コードが 0 以外なら `CalledProcessError` を送出し、try/except により `EroNum += 1` される。 |

`shell=False` (リスト指定) であるため、`PATH` 上の `python` ではなく **明示的に `python`** を解決して起動します。

> [!IMPORTANT]
> 
> `cwd`, `timeout`, `stdout`, `stderr` は指定されていないため、Python の既定挙動に従います。`cwd` は `processor.py` を起動したディレクトリ、`stdout/stderr` は **親ターミナルへそのまま流れる** ので、結果が画面と `DualLogger` の両方に現れます。

### 2. 子プロセス側での受け取り(参考)

子プロセス側は次のような定型句で受け取ります(`std_score_visualize.py` および `std_visualize_100frame.py` の冒頭にあった受け取り方です)。

```python
if os.environ.get("RUN_BY_SUBPROCESS") == "true":
    print("このスクリプトは subprocess から実行されています。")
    input_data = sys.stdin.read()
    locals().update(json.loads(input_data))
```

- `RUN_BY_SUBPROCESS` を確認することで「親から呼ばれたか・単体で実行されたか」を区別
- `sys.stdin.read()` で **全行を一括で読み込み**
- `json.loads(...)` で文字列を **Python 辞書** に復元
- `locals().update(...)` でモジュールローカル変数(`INPUT_DIR`, `OUTPUT_NAME` など)を **そのまま書き換え**

> [!NOTE]
> 
> - `locals()` はそのファイル内で有効な変数の一覧を返す関数です。これに `update(...)` すると、辞書の中身を **個別変数として直接代入したのと同じ** 効果が得られます。手作業で `INPUT_DIR = payload["INPUT_DIR"]` を 16 個並べる必要がないわけです。
> - `subprocess.run(...check=True)` の戻り値は `CompletedProcess` で、`returncode` は 0 が正常終了、それ以外は失敗扱いです。

### 3. データフロー全体図

```text
[processor.py]
  ├── input_dirs を反復
  ├── params を deepcopy ではなく dict.copy() で複製
  ├── 4 つのキーを実行対象サブフォルダに応じて上書き
  ├── json.dumps() で JSON 文字列化  ─────────────►  標準入力  ─►  [子プロセス]
  │                                                                       │
  │                                                                       ├── ENV: RUN_BY_SUBPROCESS=true
  │                                                                       ├── stdin: JSON 文字列
  │                                                                       ├── stdout/stderr ──► 親ターミナル(=DualLogger)
  │                                                                       └── 終了コード 0 → 成功 / 非 0 → catch して EroNum+1
  └── 最後に SuccessNum / (EroNum+SuccessNum) を print で総括
```

### 4. `params.copy()` の意味

```python
dir_params = params.copy()
```

- `params` 自体は改変しない(次回ループでも既定値として再利用できる)
- **シャローコピー** であるため、トップレベルキーの差し替えには十分(値は str / int / bool などのイミュータブル)
- ネストされた dict を扱う v1.0 では実質的に deep copy と同等の安全性

---

## 実装例(添付スクリプト)

参考までに、添付された 2 種類のサブプロセス雛形がそれぞれ **どんなパターンで stdin を読むか** を抜粋します。`processor.py` 自体の API サーフェスではない点に注意してください。

### 例 A: `std_score_visualize.py` の冒頭

```python
# パラメータ section (トップレベル)
INPUT_DIR = "./sun_images"
CROP_H = 800
CROP_W = 800
OUTPUT_DIR = r"C:\Users\2025005585\Desktop\python"
OUTPUT_NAME = "output_test_sample"
# ...(詳細は省略)

if os.environ.get("RUN_BY_SUBPROCESS") == "true":
    print("このスクリプトは subprocess から実行されています。")
    input_data = sys.stdin.read()
    locals().update(json.loads(input_data))
```

→ 親 (`processor.py`) から渡される JSON のキーがそのまま同名変数として注入されます。

### 例 B: `std_visualize_100frame.py` の冒頭

```python
# パラメータ宣言
DEBUG = True
INPUT_DIR = ""
CROP_H = 800
CROP_W = 800
OUTPUT_DIR = ""
OUTPUT_NAME = ""
OUTPUT_EXT = ""
FPS = 1

if os.environ.get("RUN_BY_SUBPROCESS") == "true":
    print("このスクリプトは subprocess から実行されています。")
    input_data = sys.stdin.read()
    locals().update(json.loads(input_data))
```

→ こちらは動画ではなく 100 フレームごとの標準偏差 **静止画** を生成するパターンで、同じく `RUN_BY_SUBPROCESS` で実装を分岐しています。

> [!NOTE]
> どちらの子プロセスも、**サブプロセス経由でない場合**(人が手動で `python std_score_visualize.py` を実行した場合)は上部の既定値がそのまま使われます。サブプロセス経由なら上書きされる、という二段構えです。

---

## 注意点・よくあるエラー

### 1. パスが存在しない

- 症状: `FileNotFoundError: [WinError 2] 指定されたファイルが見つかりません。`
- 対処: `input_dirs` の 10 個のパスと、各親直下のサブフォルダ名が実在するか確認してください。

### 2. `PARENT_OUTDIR` のドライブがマウントされていない

- 症状: `O:\std_score_visualize\processing_log` 作成時に `FileNotFoundError`
- 対処: `PARENT_OUTDIR` を変更するか、ドライブをマウントしてください。環境ごとに合うよう **v1.0 ではコード冒頭の定数を直接編集** する運用です。

### 3. `cwd` が想定と違う

- 症状: 子プロセス `python std_score_visualize.py` が見つからず `FileNotFoundError`
- 対処: `processor.py` と `std_score_visualize.py` を **同じフォルダに置き、そのフォルダで** `python processor.py` を起動してください。

### 4. `json.dumps` できない値が混入

- 症状: `TypeError: Object of type X is not JSON serializable`
- 対処: `dir_params` の値は `processor.py` 側で原則 `str / int / float / bool` のみが入っています。`Path` オブジェクトを入れたまま `json.dumps` すると失敗するため、**代入時に必ず `str(...)` で文字列化** してください(本コードでも `dir_params["INPUT_DIR"] = str(dirpath)` と明示的にキャストしている)。

### 5. 終了コードが 0 以外

- 症状: `subprocess.CalledProcessError` が except に入り `EroNum += 1`
- 対処: ログファイル (`console_log_*.txt`) を確認すると、子プロセスが出力した例外トレースバックが残っています。

### 6. サブプロセス側のキー不足

- 症状: 子プロセスで `KeyError: 'X'` または `NameError: name 'X' is not defined`
- 対処: 親が渡した JSON のキー名と、子プロセス側の変数名が **完全一致** しているか確認してください。

### 7. 文字化け

- 症状: ログファイル内の日本語が `??` などになる
- 対処: `DualLogger` は `encoding='utf-8'` で開いています。ファイルを **UTF-8 対応エディタ** で開いてください。

### 8. タイムアウト未設定

- 注意: `subprocess.run(...)` に `timeout` 引数は **v1.0 では設定されていません**。ハングしても手動で Ctrl-C する必要があります。「未定義」として認識してください。

> [!NOTE]
> 
> 「Ctrl-C してもいい?」について: ターミナルで `Ctrl + C` を押すと Python プロセスに割り込みが送られ、通常は安全に終了します。ただ、サブプロセスが残る場合があるため、`taskkill /F /IM python.exe` などで後始末が必要になることがあります(Windows 限定)。

---

## 用語集

| 用語                                   | 意味                                                                         |
| ------------------------------------ | -------------------------------------------------------------------------- |
| ドライバ                                 | 他のプログラムを起動して管理する親スクリプト。                                                    |
| サブプロセス                               | 親から `subprocess` で起動される別のプロセス。`os.environ` や `stdin/stdout` を介して通信することがある。 |
| 環境変数                                 | `os.environ` で参照できる OS 単位の変数辞書。`PATH` や `PYTHONPATH` など。                   |
| 標準入力 (stdin)                         | プロセスがキーボード(または他のプロセス)から文字列を受け取るパイプ。                                        |
| 標準出力 (stdout)                        | プロセスが `print` した内容を画面(または他のプロセス)に送るパイプ。                                    |
| JSON                                 | 文字列と数の組をやり取りする汎用データ形式。`json.dumps` / `json.loads` で変換する。                   |
| 終了コード                                | プロセスが最後に返す整数値。0 が「正常終了」、それ以外は「異常終了」を示す慣習。                                  |
| `subprocess.run`                     | Python から別プロセスを起動し、完了まで待つ標準 API。                                           |
| `check=True`                         | 終了コードが 0 でないとき例外を投げる指定。                                                    |
| `cwd` (current working directory)    | 「現在の作業ディレクトリ」。相対パスが解決される起点。                                                |
| Path オブジェクト                          | `pathlib.Path` で表されるパス。`name` / `iterdir()` など便利な属性・メソッドが豊富。               |
| `mkdir(parents=True, exist_ok=True)` | 中間フォルダを含めて作成。既存でもエラーにならない指定。                                               |
| ロウ文字列リテラル (`r"..."`)                 | `\` をエスケープしない Windows パス向けの記法。                                             |
| シャローコピー (`.copy()`)                  | 辞書の最上位だけを複製する操作。ネストした dict までは複製しない。                                       |
| `flush()`                            | ファイルバッファを強制的に書き出す操作。                                                       |

---

## 付録: 1 サイクル分の疑似コード

`processor.py` が 1 つのサブフォルダに対して行う処理の、骨格だけを抜粋したものです。

```python
for base_path in input_dirs:
    basename = Path(base_path).name.replace("pic", "")
    sub_folders = [p for p in Path(base_path).iterdir() if p.is_dir()]

    for dirpath in sub_folders:
        dir_params = params.copy()                          # 既定値をコピー
        dir_params["INPUT_DIR"] = str(dirpath)              # サブフォルダを imput に
        dir_params["OUTPUT_NAME"] = dirpath.name            # 出力名も合わせる
        dir_params["MEAN_IMAGE_NAME"] = dirpath.name + "_MEAN"
        dir_params["STD_IMAGE_NAME"]  = dirpath.name + "_STD"

        OUTPUT_DIRS = os.path.join(PARENT_OUTDIR, "output", basename)
        dir_params["OUTPUT_DIR"] = OUTPUT_DIRS
        dir_params["MEAN_STD_OUTPUT_DIR"] = OUTPUT_DIRS
        Path(OUTPUT_DIRS).mkdir(parents=True, exist_ok=True)

        json_payload = json.dumps(dir_params)
        try:
            subprocess.run(
                ["python", "std_score_visualize.py"],
                input=json_payload, text=True, env=my_env, check=True,
            )
            SuccessNum += 1
        except Exception as e:
            print(f"[ERROR] from processor:{e}")
            EroNum += 1
```

---

## 改訂履歴

| 版    | 備考                                        |
| ---- | ----------------------------------------- |
| v1.0 | 初版。本リファレンスは `version = 1.0` のコードに準拠しています。 |
