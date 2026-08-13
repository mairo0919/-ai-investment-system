# AI Investment Engine

日本株向け **AI投資エンジン** です。

このシステムは「株価の上げ下げを当てる予測AI」ではありません。  
市場全体から **期待値の高い銘柄をスコアリング・順位付け** し、売買判断は Rule Engine が行う設計です。

現在はデータ取得〜特徴量〜ベースライン学習（Phase 1〜3C）に加え、  
**研究用 Global100 Universe・Walk-Forward・Cross-sectional Ranking 検証** まで実装済みです。

当面の市場データは **yfinance に統一**（J-Quants 等は将来差し替え用の設計のみ）。  
売買シグナルやバックテストはまだ実装しません。

- 基本思想: [docs/product-philosophy.md](docs/product-philosophy.md)
- 世界株データ: [docs/data-global.md](docs/data-global.md)

## セットアップ方法

### 前提条件

- Python 3.14+
- 仮想環境の利用を推奨

### インストール

```bash
cd ai-investment-system
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e .
```

開発用ツール（pytest / ruff）も入れる場合:

```bash
pip install -e ".[dev]"
```

### 環境変数

```bash
cp .env.example .env
```

必要に応じて `.env` を編集します。

| 変数 | 説明 | デフォルト |
|---|---|---|
| `DATA_PROVIDER` | データ取得元 | `yfinance` |
| `TICKERS` | カンマ区切りティッカー | `7203.T,5333.T` |
| `LOOKBACK_YEARS` | 取得年数 | `10` |
| `INTERVAL` | 足種 | `1d` |
| `RAW_DATA_DIR` | 生データ CSV 保存先 | `data/raw` |
| `PROCESSED_DATA_DIR` | 特徴量 CSV 保存先 | `data/processed` |
| `MODELS_DIR` | 学習済みモデル保存先 | `models` |
| `REPORTS_DIR` | 評価レポート保存先 | `reports` |
| `TRAIN_RATIO` / `VALID_RATIO` / `TEST_RATIO` | 時系列分割比率（合計1.0） | `0.70` / `0.15` / `0.15` |
| `LOG_DIR` | ログ出力先 | `logs` |
| `LOG_LEVEL` | ログレベル | `INFO` |

## 実行方法

### データ取得（Phase 1 / Global yfinance）

```bash
# 設定ティッカー（デフォルト）
fetch-market-data

# 日/米/欧ミックスのサンプル Universe（キャッシュ利用、差分更新）
fetch-market-data --universe config/universe.sample.json

# 研究用 Global100（約100銘柄）
fetch-market-data --universe config/universe.global100.json

# キャッシュを無視して全再取得
fetch-market-data --universe config/universe.global100.json --force-refresh

# またはモジュール実行
python -m src.data.service
```

成功すると `data/raw/` に CSV が生成/更新されます（2回目以降は差分取得）。

取得項目: `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`

ログは `logs/app.log` と標準出力に出力されます。

### 特徴量生成（Phase 2）

Phase 1 で取得した raw CSV を読み込み、特徴量付き CSV を `data/processed/` へ保存します。

```bash
# 推奨
generate-features

# またはモジュール実行
python -m src.features.pipeline
```

出力例:

- `data/processed/7203_T_features.csv`
- `data/processed/5333_T_features.csv`

#### Phase 2 の方針

- **raw データは変更しない**（読み取り専用。出力は必ず `data/processed/`）
- **Look-ahead bias を防ぐ**（その日の終値までに取得可能な情報のみ使用。`shift(-1)` や未来価格は使わない）
- **目的変数（target）は作らない**（特徴量計算とラベル生成を分離。ラベルは Phase 3 以降）
- ウォームアップ不足行の NaN は `fillna(0)` せず、計算後に除外する

#### 生成される主な特徴量

| カテゴリ | 列 |
|---|---|
| リターン | `return_1d`, `return_5d`, `return_20d`, `log_return_1d` |
| 移動平均 | `sma_5`, `sma_20`, `sma_60`, `ema_12`, `ema_26`, `close_sma_20_ratio`, `close_sma_60_ratio` |
| オシレータ | `rsi_14`, `macd`, `macd_signal`, `macd_hist` |
| ボリンジャー | `bb_middle`, `bb_upper`, `bb_lower`, `bb_width` |
| 変動性 | `atr_14`, `volatility_20` |
| 出来高 | `volume_change_1d`, `volume_sma_20`, `volume_ratio_20` |
| 日中 | `high_low_range`, `open_close_return` |

詳細な計算仕様は [docs/features.md](docs/features.md) を参照してください。

### モデル学習（Phase 3A / 3B）

Phase 2 の特徴量から「翌営業日に Adj Close が上昇する確率」を学習します。

```bash
# LogisticRegression + LightGBM を同一分割で学習・比較
train-model

# またはモジュール実行
python -m src.ml.trainer
```

成果物:

- `models/logistic_regression.joblib` / `.metadata.json`
- `models/lightgbm.joblib` / `.metadata.json`
- `reports/logistic_regression_metrics.json`
- `reports/lightgbm_metrics.json`
- `reports/model_comparison.json`

#### Phase 3 の方針

- **target**: `target_up = 1` if `AdjClose[t+1] > AdjClose[t]` else `0`（`shift(-1)` は target 生成のみ）
- **時系列分割**: Date 基準で Train 70% / Validation 15% / Test 15%。shuffle 禁止
- **3A モデル**: `StandardScaler + LogisticRegression`（Scaler は Train のみ fit）
- **3B モデル**: `LightGBM`（Scaler なし、Validation AUC で early stopping。Test は不使用）
- **Baseline**: 多数クラス予測、および `return_1d > 0` の継続予測と比較
- **評価**: Accuracy / Precision / Recall / F1 / ROC-AUC / Confusion Matrix（全体 + 銘柄別）
- **Look-ahead bias 防止**: 未来特徴量禁止、Test を学習・early stopping・チューニングに不使用

詳細は [docs/ml-training.md](docs/ml-training.md) を参照してください。

> macOS で LightGBM を使う場合、OpenMP（`libomp`）が必要なことがあります。  
> `brew install libomp` 後に `pip install -e .` を実行してください。

### 実験（Phase 3C）

複数 Target（1d / 5d / 10d / 5d threshold）× Feature Set（株のみ / 株+市場）× モデル（LR / LightGBM）を Validation 優先で比較します。

```bash
# 市場指数（日経225）取得
fetch-market-indices

# 実験実行（結果は reports/experiments/ に追記保存）
run-experiments
```

方針:

- Target / Feature Set の選択は **Validation** で行う（Test は最終確認のみ）
- 5d/10d Target は **purged split** で境界リークを防止
- Probability decile / 上位確率群の将来リターン関係を分析（売買バックテストではない）
- TOPIX 現金指数が yfinance で取得できない場合は代替せず、日経のみで Feature Set B を構築

### Walk-Forward + Cross-sectional Ranking（研究用）

研究用固定 Universe（`config/universe.global100.json`）で Expanding Window Walk-Forward と日次 Cross-sectional Ranking を評価します。売買・バックテストは含みません。

```bash
# Global100 を取得（初回）→ Walk-Forward 実行
fetch-market-data --universe config/universe.global100.json
run-walk-forward --universe config/universe.global100.json

# メタデータ補完（任意・遅い）
run-walk-forward --universe config/universe.global100.json --enrich-metadata
```

成果物: `reports/walk_forward/`（overview + Target×Model 別 JSON）

### Ranking Edge 安定性診断（Phase 3E）

既存 Walk-Forward Validation を再スコアし、Country内 Ranking・score分布・レジーム・寄与分解を診断します（売買なし）。

```bash
run-stability-analysis --universe config/universe.global100.json
```

成果物: `reports/stability/`（country_ranking / score_distribution / regime_analysis / group_contribution / bucket_analysis / one_day_ic_diagnostics）

### Learning-to-Rank（Phase 3F）

Country内（Date × Country）を ranking group とし、`LGBMRanker`（lambdarank）で順位学習します。主評価は Global raw ではなく Country内 NDCG / IC / Top excess です。

```bash
run-learning-to-rank --universe config/universe.global100.json
```

成果物: `reports/ranking/`  
比較: Common Ranker / Country-specific Ranker / momentum baseline（5d・20d）

### 特徴量拡張（Phase 3G）

Cross-sectional / Sector-relative / Macro・FX / Breadth・Dispersion を追加し、Feature Set A/B/C を Country内 LTR で比較します。

```bash
run-feature-expansion --universe config/universe.global100.json
```

成果物: `reports/feature_expansion/`  
※ yfinance の現時点 fundamentals（PER等）は point-in-time 不可のため特徴量禁止

### Label / Score Resolution（Phase 3H）

5段階 relevance による情報圧縮と score tie を診断し、decile / 0–99 label と明示的 `label_gain` を比較します。

```bash
run-label-resolution --universe config/universe.global100.json
```

成果物: `reports/label_resolution/`  
Label D（0..N-1 絶対順位）は group size 差のため未実装（理由をレポートに記載）

方針:

- Expanding window、最低 5 fold、Target horizon に応じた purge
- Target: 主に `target_up_5d` / `target_up_10d`（1d は LR のみ Reference）
- Model: LogisticRegression / LightGBM のみ
- 評価: Top5/10/20%・Excess Return・IC・Country / Sector 別
- 履歴 5 年未満の銘柄は全 fold から除外（部分参加しない）
- Global プールは通貨混在（為替換算なし）。解釈は Country 内 Ranking を優先
- Survivorship bias のある研究用固定 Universe（Known Limitation）

### テスト

```bash
pytest
```

## フォルダ構成

```text
ai-investment-system/
├── src/
│   ├── config/                 # 設定（環境変数読込）
│   ├── core/                   # 共通例外など
│   ├── data/
│   │   ├── providers/
│   │   │   ├── base.py         # Provider 抽象クラス
│   │   │   └── yfinance_provider.py
│   │   └── service.py          # 取得〜CSV保存のオーケストレーション
│   ├── features/
│   │   ├── indicators.py       # 特徴量の純粋関数
│   │   └── pipeline.py         # 読込・検証・生成・保存
│   ├── ml/
│   │   ├── dataset.py          # target生成・データセット
│   │   ├── split.py            # 時系列分割
│   │   ├── baseline.py         # 比較用Baseline
│   │   ├── evaluator.py        # 評価指標
│   │   ├── model_registry.py   # 共通予測・評価ヘルパー
│   │   ├── lightgbm_model.py   # LightGBM固有処理
│   │   └── trainer.py          # 学習・比較・保存
│   └── utils/                  # logging など
├── tests/
├── data/
│   ├── raw/                    # 取得した生データ（CSV, Git管理外）
│   └── processed/              # 特徴量データ（CSV, Git管理外）
├── models/                     # 学習済みモデル（Git管理外）
├── reports/                    # 評価レポート（Git管理外）
├── logs/
├── config/
│   ├── universe.sample.json      # 日/米/欧パイロットUniverse
│   └── universe.global100.json   # 研究用固定 Global100
├── docs/
│   ├── product-philosophy.md   # AI投資エンジンの基本思想
│   ├── data-global.md          # yfinance世界株データ方針
│   ├── architecture.md
│   ├── features.md
│   └── ml-training.md
├── pyproject.toml
├── .env.example
└── README.md
```

### Provider の切り替え方針

`src/data/service.py` の `create_provider()` に実装クラスを登録します。公式データ Provider は必要時のみ。

## ロードマップ

### 実装済み（基盤）

| Phase | 内容 | 状態 |
|---|---|---|
| 1 | データ取得 | 実装済み |
| 2 | 特徴量生成 | 実装済み |
| 3 | ベースライン学習 / 実験基盤（3A/3B/3C） | 実装済み |

### 今後の順序（Global AI投資エンジン）

| Step | 内容 | 状態 |
|---|---|---|
| 1 | yfinance 世界株データ基盤（キャッシュ・メタデータ・Universe） | 実装済み |
| 2 | Universe 拡充 + Walk-Forward + Cross-sectional Ranking | 実装済み |
| 3 | 期待値分析 | 次候補（安定 edge 確認後） |
| 4 | バックテスト / Rule Engine | 未着手 |
| 5 | リスク管理 | 未着手 |
| 6 | フォワードテスト | 未着手 |
| 7 | 証券会社API / 必要時のみ公式データProvider | 未着手 |

評価の主軸は Accuracy ではなく、**Top10%/20% Average Return・Excess Return・IC・Rank Correlation** です。

## 技術スタック

- Python 3.14
- pandas / numpy / yfinance / scikit-learn / joblib / lightgbm
- 将来追加予定: J-Quants, OpenAI API, Backtesting.py

## ライセンス

MIT
