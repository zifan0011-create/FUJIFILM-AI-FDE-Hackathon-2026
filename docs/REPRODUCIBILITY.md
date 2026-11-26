# 再現マニフェスト

> 2026年11月26日・提出後の補足資料。提出時の検証コードに記録された固定値を一覧にしたものです。再現実験の新規実施を意味しません。

## 提出時点のソース

- 固定コミット: [05a44d1ed10a6edf14a9efe1e3978df9f7ad4549](https://github.com/zifan0011-create/FUJIFILM-AI-FDE-Hackathon-2026/tree/05a44d1ed10a6edf14a9efe1e3978df9f7ad4549)
- Python 3.13.14、Windows x64、AMD Ryzen 7 8745HS、CPU
- 依存関係: [requirements.txt](../requirements.txt)
- 元写真単位の固定5分割、乱数シード42
- 特徴量: 465次元 float32
- 公式学習行: 10,079、評価行: 8,059

## 出力ファイル

| 方式 | ファイル名 | 陽性数 | SHA-256 |
| --- | --- | ---: | --- |
| C04 | submission_c04.csv | 168 | `7ea929eb932ab7466ae4b868b48dd2b352147f51d40406d38f6af7c44b8e14c8` |
| C05M2-J | submission_c05m2_c05j.csv | 180 | `81d1dc42364593a4c0db9260b050544c2341d01b8e7882b91fec8d85676d089e` |
| C05M2-Final | submission_c05m2_c05final.csv | 176 | `2afb51af8fc5571bcab3ebc389291a16aa3d4ab73c4edd79c365cf8152d3a454` |
| C05M2-F | submission_c05m2_c05f.csv | 180 | `18e0fa2cdc45b511b7a36361c933ee67e8a8bbe4e9bcdcc8833bab3b727d7388` |
| C05M2-E5 | submission_c05m2_e5.csv | 170 | `90b864474fabb9a0c8ae1412ee1279d234668ca1be766d336e9fb500a69cde56` |

## 検証方法

固定の学習分割マニフェストの SHA-256 は `e730b29021490128f28bc753fa92eb4343606c89f94873a94dd5442629e6efb5` です。各方式の verify コマンドでは公式の画像行順、8,059行、二値ラベル、陽性数と CSV 全体の SHA-256 を確認します。

C05M2-Final の再現手順:

```powershell
.\.venv\Scripts\python.exe c04.py extract
.\.venv\Scripts\python.exe c05m2.py final train
.\.venv\Scripts\python.exe c05m2.py final infer
.\.venv\Scripts\python.exe c05m2.py final verify
Get-FileHash .\submission_c05m2_c05final.csv -Algorithm SHA256
```

環境構築とデータ配置は [README](../README.md) を参照してください。

## 確認済み範囲と限界

参加時のローカル検証記録によれば、C04 は元画像からの一連の処理で出力を確認し、改善方式は既存の特徴量キャッシュからの再学習・推論による出力照合を実施しています。今回の追記では公開コードの期待値を照合したのみであり、元画像を使った新たな実行はしていません。

評価用画像・公式ラベル・予測CSVの実体は公開リポジトリに追加していません。再現には正規に入手した公式データが必要です。公開スコアの確定、選択したCSVの公式評価への反映は主催者の確認に従います。
