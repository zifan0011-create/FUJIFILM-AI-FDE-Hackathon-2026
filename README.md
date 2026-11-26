# 富士フイルム AI FDEハッカソン 2026 — Pコン跡の画像分類

**画像特徴量設計・機械学習・再現性検証から現場への応用構想まで**

本プロジェクトでは、コンクリート壁面画像の P コン跡を識別するため、465次元の画像特徴量と LightGBM を組み合わせ、元写真単位の交差検証、難負例への重み付け、データ拡張、アンサンブル、写真グループごとの重み補正を検討しました。

## プロジェクトのポイント

- 正例が少ない課題に対し、F1 と誤検出・見逃しの両方を重視。
- 元写真単位で学習・検証を分離し、撮影条件の違いによる汎化の課題を検討。
- 同一の特徴量を使い、異なる改善仮説を比較。5方式の再現コードを公開。
- 依存バージョン、分割、閾値、CSV の SHA-256 を固定し、提出結果の同一性を検証。

## ドキュメント

| 資料 | 内容 |
| --- | --- |
| [技術設計・検証レポート](docs/TECHNICAL_REPORT.md) | 特徴量の設計、処理構成、学習方法、誤分類分析 |
| [実験比較とモデル選択](docs/MODEL_COMPARISON.md) | 同点の改善方式、比較結果、モデル選択の考え方 |
| [再現マニフェスト](docs/REPRODUCIBILITY.md) | 出力名、行数、陽性数、SHA-256、確認範囲 |
| [現場導入・事業価値の検討](docs/DEPLOYMENT_CONCEPT.md) | 大学施設管理への応用、クラウド構成、障害対応、実証計画 |

上記4資料は2026年11月26日の提出後に追加した補足説明です。競技時点のコードと README は [元の提出コミット 05a44d1](https://github.com/zifan0011-create/FUJIFILM-AI-FDE-Hackathon-2026/tree/05a44d1ed10a6edf14a9efe1e3978df9f7ad4549) で確認できます。モデルの学習・推論コードや固定設定は変更していません。

---

# 再現手順

## 課題概要

256 × 256 ピクセルのコンクリート壁面画像から、Pコン跡の有無を二値分類します。C05M2-J、C05M2-Final、C05M2-F、C04、C05M2-E5 の提出用 CSV を再現できます。学習には公式の学習用画像とラベルを使います。

## 動作環境

Windows x64（OS ビルド 26200）、CPython 3.13.14（64 bit）、AMD Ryzen 7 8745HS で確認しています。学習と推論は CPU の 6 スレッドで実行します。実際に使用した依存ライブラリのバージョンは `requirements.txt` に固定しています。

## ディレクトリ構成とデータの配置方法

展開したソースと同じ階層に、公式配布の CSV とテキストを置きます。画像の ZIP を展開し、公式ファイルに記載された `W/CW/...` および `W/UW/...` の相対パスで参照できるように配置します。

```text
.
├── c04.py
├── c05m2.py
├── features.py
├── requirements.txt
├── train_label.csv
├── test.txt
├── sample_submission.csv
└── DATA_Maguire_20180517_ALL/
    └── SDNET2018/
        └── W/
            ├── CW/
            └── UW/
```

## ライブラリのインストール方法

ソースを展開したディレクトリで、PowerShell から実行します。

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 特徴量抽出

画像を双線形補間で 128 × 128 に縮小し、次のブロックを記載順に連結して 465 次元の `float32` 特徴量を生成します。

| ブロック | 次元数 |
| --- | ---: |
| 符号付き DoG のコントラスト、連結成分の形状、テクスチャ | 100（48 + 30 + 22） |
| 局所正規化 DoG のリング応答、連結成分、局所の色とテクスチャ | 200（88 + 64 + 48） |
| 多尺度 Hessian の候補と放射方向の勾配、画像端の帯状領域、明暗統計 | 165（108 + 54 + 3） |

## モデル学習

ファイル名からタイル分割前の写真 ID を取得します。公式の学習行順を保ち、`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` で分割します。生成した分割の SHA-256 を C04 の記録と照合します。

各 fold の学習部分で LightGBM を一度学習します。学習部分の負例を陽性スコア順に並べ、上位 `max(1, 負例数 // 50)` 件に重み 3.0 を設定します。残りのサンプルの重みは 1.0 です。同じ学習部分を使って重み付きで再学習します。

両方の学習で `n_estimators=240`、`learning_rate=0.035`、`num_leaves=15`、`min_child_samples=25`、`colsample_bytree=0.85`、`scale_pos_weight=1`、`reg_lambda=2`、`random_state=42`、`n_jobs=6`、`verbosity=-1` を使用します。その他のパラメータは LightGBM 4.7.0 の既定値を使用します。

## 推論・CSV生成

5 モデルの陽性確率を等重みで平均します。平均が C04 の固定閾値 0.2042770082 以上の場合に 1 を出力します。計算には元の精度 `0.20427700824353937` を使用します。出力は `test.txt` と同じ順序のヘッダーなし CSV です。

## 実行コマンド

```powershell
.\.venv\Scripts\python.exe c04.py extract
.\.venv\Scripts\python.exe c04.py train
.\.venv\Scripts\python.exe c04.py infer
.\.venv\Scripts\python.exe c04.py verify
```

`extract` は特徴量を `work/` に保存し、`train` は同じディレクトリに 5 本のモデルを保存します。`infer` はソースと同じ階層に `submission_c04.csv` を生成します。

## 再現性の確認方法

`verify` は行順、二値ラベル、陽性数、CSV 全体の SHA-256 を照合します。元の提出 CSV は 8,059 行、陽性 168 件、157,744 バイトで、SHA-256 は次の値です。

```text
7ea929eb932ab7466ae4b868b48dd2b352147f51d40406d38f6af7c44b8e14c8
```

## C05M2-J・C05M2-Final・C05M2-F・C05M2-E5 の再現

各方式は同じ 465 次元の特徴量と固定の写真単位 5 分割を使用します。J、Final、F の LightGBM は C04 と同じ設定で、学習と推論のスレッド数は 1 です。J と Final は各 fold の学習部分で Hard Negative の重みを決定します。F の提出用モデルは全学習データで同じ手順を実行し、写真とクラスごとの補正を適用します。

| 方式 | 学習・融合 | 固定閾値 | 陽性数 |
| --- | --- | ---: | ---: |
| C05M2-J | 5 fold を等重み。元画像 0.50、測光変換 0.25、回転または反転 0.25。各視点に元画像の Hard Negative 重みを掛ける | `0.15873688188386165` | 180 |
| C05M2-Final | 各 fold で Hard Negative 学習モデル 0.80、通常学習モデル 0.20。5 fold を等重み | `0.16461504711798983` | 176 |
| C05M2-F | 全学習データで再学習した単一モデル。各重みを同一写真・同一クラスの件数の 0.25 乗で割り、クラスごとに補正前の重み総和へ正規化 | `0.09902811548177957` | 180 |
| C05M2-E5 | C04 の Hard Negative 5 モデルを等重み | `0.1937088038619091` | 170 |

J の拡張は公式の学習画像 10,079 枚を対象とします。公式の学習ファイルの 0 始まりの行番号が偶数の場合の測光変換は gamma 1.06・gain 0.96・offset −2、奇数の場合は gamma 0.94・gain 1.04・offset +2 です。同じ行番号を 3 で割った余りが 0、1、2 の場合に、それぞれ 90 度回転、左右反転、上下反転を使用します。拡張画像は学習に使用し、推論では元画像の特徴量を使用します。

### 実行コマンド

最初に共通の特徴量を抽出します。以下の J、Final、F の手順は、それぞれ独立して実行できます。

```powershell
.\.venv\Scripts\python.exe c04.py extract

# C05M2-J
.\.venv\Scripts\python.exe c05m2.py j augment
.\.venv\Scripts\python.exe c05m2.py j train
.\.venv\Scripts\python.exe c05m2.py j infer
.\.venv\Scripts\python.exe c05m2.py j verify

# C05M2-Final
.\.venv\Scripts\python.exe c05m2.py final train
.\.venv\Scripts\python.exe c05m2.py final infer
.\.venv\Scripts\python.exe c05m2.py final verify

# C05M2-F
.\.venv\Scripts\python.exe c05m2.py f train
.\.venv\Scripts\python.exe c05m2.py f infer
.\.venv\Scripts\python.exe c05m2.py f verify

# C05M2-E5
.\.venv\Scripts\python.exe c04.py train
.\.venv\Scripts\python.exe c05m2.py e5 infer
.\.venv\Scripts\python.exe c05m2.py e5 verify
```

J、Final、F、E5 の出力はそれぞれ `submission_c05m2_c05j.csv`、`submission_c05m2_c05final.csv`、`submission_c05m2_c05f.csv`、`submission_c05m2_e5.csv` です。`verify` で公式の行順、8,059 行、陽性数、および各出力の SHA-256 を照合します。
