# 3業務テーマの小規模実験

0.4.0a5の既存select/reportを呼ぶ、今回限りの数値列による分析アダプタです。汎用的なカテゴリ/欠損処理や新しいモデルを本体へ追加していません。結果を見て特徴量・seed・設定を選び直していません。

- `protocol.json`: モデル結果を見る前に保存した条件・列・仮説・参考ルール。外部への事前登録を意味しません。
- `run.py`: 加工・固定抽出・既存5モデルの学習側選定・選定モデルのtest・参考ルール・図付きExcel。
- `verify.py`: 保存ハッシュ、分割、scaler、選定、指標、全予測、図、原本からの再構築と再購入の期間/顧客対応を確認。assertを使うのでpython -Oは禁止。
- `make_figure.py`: 学習側の記述統計から記事用PNGを生成。描画入力とPNGのハッシュも保存。

## 公開ミラーから原本を取得

URLをコミットに固定しています。取得物はrun.py内のSHA256・Git blob・件数・列数と照合します。UCI原配布先の直接取得はこの環境では403となったためミラーを使用しました。原配布ファイルとのバイト一致は確認していません。データの元出典・利用条件はdocs/BUSINESS_QUICK_RESULTS.mdを参照してください。以下のcurlを実行できるネットワークが必要です。

```sh
mkdir -p /tmp/diy-business-raw
curl -fL --max-time 60 -o /tmp/diy-business-raw/hotel_bookings.csv 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/d75aaa0d31596ad6487ae0db20138067d301e031/data/2020/2020-02-11/hotels.csv'
curl -fL --max-time 60 -o /tmp/diy-business-raw/coupon.csv 'https://raw.githubusercontent.com/artessari/MLProject_In-VehicleCouponRecommendation/c1e183fc5b2de34b5cd75759eb322aaab246bbc2/in-vehicle-coupon-recommendation.csv'
curl -fL --max-time 60 -o /tmp/diy-business-raw/Online_Retail.xlsx 'https://raw.githubusercontent.com/SarahMestiri/online-retail-case/ba0c6609ad283ce1aa08e627dcd10654358a9af5/Online%20Retail.xlsx'
```

実行はリポジトリルートから、新しい出力先を指定します。既存の保存結果を上書きしません。TensorFlowは不要です。依存導入はREADMEのコア環境手順に従ってください。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python experiments/business_quick_20261108/run.py --raw-dir /tmp/diy-business-raw --output runs/business-quick-new
.venv/bin/python experiments/business_quick_20261108/verify.py runs/business-quick-new --raw-dir /tmp/diy-business-raw
```

学習済みの保存結果だけを確認する場合はraw-dirを省略できます。その場合、原本からの再構築は検証しません。

```sh
.venv/bin/python experiments/business_quick_20261108/verify.py validation/business-quick-20261108
```

記事用の図を再生成する場合は、必要な日本語フォントを用意してください。画像の上書きを避けるため例ではruns配下に出します。描画入力とmanifestはこの実験フォルダへ更新されるので、元の記録を保ちたい場合は別checkoutで実行してください。

```sh
.venv/bin/python experiments/business_quick_20261108/make_figure.py validation/business-quick-20261108 --output runs/business-associations.png
```

完全一致レコードの除去や、数値列・二値化の限定には情報損失と対象集団の変更があります。今回の同一母集団内の分類結果を、別施設・別回答者・将来期間・施策効果へ一般化しません。
