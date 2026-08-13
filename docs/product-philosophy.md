# Product Philosophy — Global AI Investment Engine

## 何ではないか

- 株価の上げ下げを当てる予測AIではない
- AIが単独で「買う/売る」を決めるシステムではない
- 日本株専用システムではない

## 何か

**世界株対応の AI投資エンジン**

市場全体から期待値が高い銘柄をスコアリングし、順位付けする。

```text
データ取得 (当面: yfinance)
  → 正規化 / キャッシュ
  → 特徴量生成
  → 市場データ統合
  → AI分析
  → スコアリング
  → ランキング評価
  → リスク評価
  → Rule Engine（売買）
  → バックテスト
  → フォワードテスト
  → 証券会社API
```

## データ取得方針（当面）

| 項目 | 方針 |
|---|---|
| Provider | **yfinance に統一** |
| 対象市場 | 日本 / 米国 / 欧州 / yfinanceで取れるその他 |
| J-Quants等 | 将来差し替え可能な設計のみ維持。**今は実装しない** |
| Universe | 最初は 50〜100 銘柄。全銘柄は取らない |
| キャッシュ | ローカル保存、差分更新、force refresh |

Provider 抽象（`BaseDataProvider`）を維持し、将来:

- `JQuantsProvider`
- `USOfficialProvider`
- `GlobalMarketProvider`

を追加できる。呼び出し側は yfinance 固有APIに直接依存しない。

## 銘柄メタデータ

可能な範囲で管理（取得不可は null 許容）:

`Symbol`, `Exchange`, `Country`, `Currency`, `Sector`, `Industry`, `MarketCap`, `Timezone`

## ランキング評価（売買ではない）

将来的に評価可能にする軸:

- Global Ranking
- Country Ranking
- Market Ranking
- Sector Ranking

いま実装するのは **評価指標基盤** であり、売買ランキングや発注ロジックではない。

## AIの出力

| 項目 | 例 |
|---|---|
| Symbol | 7203.T |
| Expected Score | 92.3 |
| Confidence | 84% |
| Expected Return | +3.2% |
| Risk | Low |
| Market Strength | Strong |

## 売買判断

売買は **Rule Engine** が行う。AI単独で「買う」とは判断しない。

## 評価思想

「当たったか」より「上位に置いた銘柄ほど良かったか」。

- Top10% / Top20% Average Return
- Excess Return
- Information Coefficient
- Rank Correlation

## 現時点でやらないこと

- J-Quants / 有料マーケットデータ
- 証券会社API
- 売買シグナル
- バックテスト
- 実売買

## 開発順序（当面）

1. yfinance 世界株データ基盤
2. ランキング評価基盤
3. Universe 拡充（日/米/欧）
4. Walk-Forward + Cross-sectional Ranking
5. 期待値分析
6. バックテスト以降
