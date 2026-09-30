---
title: "ViT と画像パッチ埋め込み / Vision Transformer and Patch Embedding(実装・実験編 3/4)"
---

この記事は後編(実装・実験編 3/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/019_vision_transformer-practice-2)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/019_vision_transformer-practice-4)。

### 6.1 実験宣言セル: 共通の設定・検証すること・判定基準・前提条件

**この節の内容は本番実行の前に確定させ、結果を見た後に変更しない。**

#### 共通の設定

- **標準の ViT**: $D = 192$、$L = 6$、$h = 3$、$d_{\mathrm{ff}} = 768$、$P = 4$($N = 64$)、位置埋め込みは学習可能な 1 次元、読み出しは
  [CLS] トークン、dropout なし、正規化前置、GELU。分類ヘッドは零で初期化する。
- **CNN**: ResNet-56(He et al. [7] の 4.2 節の CIFAR-10 用の構成、ショートカットは方式 A。5.4 節でスクラッチ実装)。
- **全条件で共通**: バッチサイズ 128、学習ステップ数 $T$(**6.4 節で実行計画として自動選択される値。候補は $\{3968, 1984, 992\}$**。
  選ばれた $T$ は、較正を含む全条件・全シードで同一)、
  AdamW(重み減衰 0.05、行列形の重みのみ)、最初の 10% で線形 warmup の後 cosine で最大値の 0.01 倍まで減衰、gradient clipping の
  閾値 1.0、データ拡張はパディング 4 の random crop と左右反転、CUDA では FP16 の autocast と動的損失スケーリング。
- **評価**: 学習の最終ステップの重みで、**テスト集合 10,000 枚全件の top-1 正解率** $a$ を判定に使う。テストの交差エントロピーは
  診断量とする。テスト集合は較正に使わない。
- **シード**: シード $s$ は初期化とデータの順序・データ拡張を決める(5.5 節)。同じ $s$ の条件どうしは、同じ部分集合を使う限り
  同じ順序で同じ拡張をした画像を見る(対応のある比較)。シードは $0, 1, \dots$ を使い、数は削る段階で決まる(段階 0 ですべて 5)。
- **標準条件 S の共有**: S($P = 4$・学習可能な 1 次元・全データ)は、実験 A の条件 1・実験 B の $P = 4$・実験 C の ViT の $f = 1$ として
  共有する。S のシード数は、S を使う実験のシード数の最大値とする(実験 B は削らないので常に 5)。

#### 学習率の較正(本番の冒頭、6.5 節)

- ViT(標準の構成)と CNN のそれぞれについて、本番と同じ $T$・全データで、**較正専用のシード**(シード番号 90。実験のシードと共有しない)の
  1 シードを使い、公比 2 の格子 $\{5 \times 10^{-4}, 10^{-3}, 2 \times 10^{-3}\}$ の各学習率で学習し、**検証集合の正解率** が最大の学習率を選ぶ
  (同点なら小さい方)。
- **拡張の規則**: 最良の学習率が格子の端に来た場合は、その方向に公比 2 で 1 点だけ格子を拡張して学習し、改めて最良を選ぶ(1 回のみ)。
- 較正のシードを実験のシードと分けるのは、選ばれた学習率の学習を実験のシード 0 と同じ乱数で行うと、検証集合で選ばれた学習
  (勝者)がそのまま標準条件のシード 0 になり、その値だけが選択によって楽観的に偏るためである。
- ViT の全条件(実験 A・B・C)は、標準の構成で選んだ **同じ学習率** を使う。条件ごとに最適な学習率が異なりうること(位置埋め込みの
  方式・パッチサイズ・データ量によって最適値が動くこと)は、固定した条件による **交絡** である。CNN の全データ量も同様に、全データで
  選んだ学習率を使う。

#### 共通の前提条件

- **P0(較正)**: ViT と CNN の較正で選んだ学習率が、格子の **内点** であること(拡張後も最良が端なら不成立)。ViT の P0 は実験 A・B・C、
  CNN の P0 は実験 C の前提条件である。
- **P1(学習の成立)**: 各実験の全条件で、シード平均のテスト正解率が $0.25$ 以上であること(一様な推測の 0.1 の 2.5 倍)。

前提条件が 1 つでも成立しない実験は、判定関数の結果によらず **前提不成立** と記録する(判定不能とは区別する)。

#### 判定の形

各実験の対比量 $\Delta$ とその標準偏差 $\sigma$ に対し、$\Delta > 2\sigma$ なら **支持**、$\Delta < -2\sigma$ なら **反証**、それ以外は
**判定不能** とする。$\sigma$ はシード間のばらつきから求める(テスト集合は全条件で共通なので、テスト事例の抽出によるばらつきは
条件間の比較では共通に効き、対比量の標準偏差には含めない)。

#### 実行計画(学習ステップ数 $T$ と削る段階の組)と、その自動選択

**削る段階**:

| 段階 | 内容 | 実験 A のシード数 | 実験 B の水準・シード数 | 実験 C のシード数 |
|---|---|---|---|---|
| 0 | 全実験 5 シード | 5 | $P \in \{8, 4, 2\}$、5 | 5 |
| 1 | 実験 B の $P = 2$ を除く | 5 | $P \in \{8, 4\}$、5 | 5 |
| 2 | 段階 1 に加え、実験 C のシード数を 3 にする | 5 | $P \in \{8, 4\}$、5 | 3 |
| 3 | 段階 2 に加え、実験 A のシード数を 3 にする | 3 | $P \in \{8, 4\}$、5 | 3 |

**学習ステップ数の候補**: $T \in \{3968, 1984, 992\}$(公比 2)。全データ・$f = 1/4$・$f = 1/16$ でのエポック数(バッチ 128)は次のとおり。

| $T$ | $f = 1$(45,000 枚) | $f = 1/4$(11,250 枚) | $f = 1/16$(2,810 枚) |
|---|---|---|---|
| 3968 | 11.3 | 45.1 | 180.7 |
| 1984 | 5.6 | 22.6 | 90.4 |
| 992 | 2.8 | 11.3 | 45.2 |

**実行計画**: $T$ の候補と段階 0〜3 を組み合わせた 12 通りに、優先順位の高い順に番号を付ける。優先順位は **$T$ の大きい順を第 1 キー、
段階の番号の小さい順を第 2 キー** とする。

| 計画 | $T$ | 段階 |
|---|---|---|
| 0〜3 | 3968 | 0〜3 |
| 4〜7 | 1984 | 0〜3 |
| 8〜11 | 992 | 0〜3 |

- **選択の規則**: 学習を始める前に(6.4 節、較正の前)、T4 上のスケーリングの計測(6.3 節)から、各計画の残りの実行時間(較正と
  本番の学習・評価)を見積もる。予算は「120 分 − データの準備・確認・計測にすでに使った実測時間(ノートブックの開始からの経過時間)」
  とし、予算内に収まる **番号の最も小さい計画** を選ぶ。計画 11 でも超える場合は、学習の前に例外で停止する。**選択は見積もりのみに
  基づき、どの実験の結果も参照しない。** 標準条件 S は実験 B が削られないため常に 5 シードを学習する(段階 3 でも、S は実験 A・C の
  最小シード数以上を確保している)。
- **優先順位の理由**: $T$ は、実験 C が主張の対象(十分に学習した状態での、データ量による汎化の差)を測れているかどうかを決める。
  学習が足りない状態では、データ量の違いが正解率に現れる前に学習を打ち切ることになり(たとえば $T = 992$ では全データを 2.8 回しか
  見ない)、「帰納バイアスとデータ量」とは別の量(学習の速さ)を測ることにつながる。一方、シード数と $P = 2$ の水準は、判定の
  **検出力** を決めるだけである。検出力の低下は判定不能を増やすが、誤った量を測ることにはならない。そのため $T$ を優先し、同じ $T$
  の中では段階の小さい(シード数・水準の多い)計画を優先する。
- **段階の順序の理由**(同じ $T$ の中): $P = 2$ は 1 回の学習が標準の約 4〜5 倍の計算量(3.2.2 節)で、単独の条件として最も重い。次に、学習の数が
  最も多い実験 C(ViT 2 水準 + CNN 3 水準)のシード数を削る。実験 A は対応のある差を対比量とし、シード間のばらつきが条件間で
  打ち消し合うので、シード数の減少の影響が相対的に小さいと考え、最後に削る。

#### 実験 A: 位置埋め込みの寄与

**検証すること**: 位置埋め込みなしの ViT は、学習可能な 1 次元の位置埋め込みの ViT より、テスト正解率が低い(3.4 節)。

**条件**(標準の ViT、全データ、$T$ ステップ、同じ学習率):

| 条件 | 位置埋め込み |
|---|---|
| 0 | なし(出力はパッチの並べ替えに対して不変、3.4.2 節) |
| 1 | 学習可能な 1 次元(= 標準条件 S) |
| 2 | 2 次元正弦波(Beyer et al. [3]、[CLS] には零ベクトル) |

**対比量**: シード $s$ ごとの対応付きの差 $d_s = a_{1,s} - a_{0,s}$($a_{c,s}$ は条件 $c$・シード $s$ のテスト正解率)の平均
$\Delta_A = \bar{d} = \frac{1}{n_A} \sum_s d_s$($n_A$ は実験 A のシード数)。

**標準偏差の導出**: 同じシードの 2 条件は初期化の乱数の種・データの順序・データ拡張を共有するので、差をシードごとにとって
シード間の共通の変動を打ち消す。$d_s$ は互いに独立なので、$\sigma_A = \mathrm{sd}(d_s) / \sqrt{n_A}$($\mathrm{sd}$ は不偏標本標準偏差)。

**判定**: $\Delta_A > 2\sigma_A$ なら支持、$\Delta_A < -2\sigma_A$ なら反証、それ以外は判定不能。

**前提条件**: P0(ViT)、P1(条件 0・1・2)。

**作用点の記述**: 位置埋め込みが直接作用するのは入力の系列 $z_0$ であり、モデルがパッチの **配置** を区別できるかどうかである。
それを直接測る量は、パッチを並べ替えた画像に対する出力の変化である。テスト正解率は、そこから「配置の情報を学習で使う」
「使うことで分類が良くなる」の 2 段を経た下流の量だが、検証したい主張(位置埋め込みが正解率に寄与する)そのものなので対比量に
選んだ。直接の作用点に近い量として、**パッチを固定の置換で並べ替えたテスト画像での正解率** とその低下幅(元のテスト正解率との差)を
各条件で診断量として併記する(条件 0 は並べ替え不変なので低下幅は 0)。

**判定を置かないもの**: 条件 2 は判定の対象外とし、テスト正解率と条件 1 との差を診断量として併記する。学習された位置埋め込みの
余弦類似度に 2 次元の構造が現れるか(原論文の図 7 中央に相当)、層ごとの平均の注意距離(原論文の図 7 右に相当)は、判定なしの
観察とする(6.8 節、シード 0)。

**アサーション**(実験ではなく不変条件): 条件 0 のモデルで、パッチの並べ替えに対して出力が不変であること(FP32、5.4 節)。

#### 実験 B: パッチサイズと系列長

**検証すること**: パッチサイズ $P$ を小さくする(系列長 $N$ を長くする)ほど、テスト正解率が上がる。

**条件**: $P \in \{8, 4, 2\}$、すなわち $N \in \{16, 64, 256\}$(公比 4 の等比の水準)。位置埋め込みは学習可能な 1 次元、全データ、
$T$ ステップ、同じ学習率。$P = 4$ は標準条件 S。段階 1 以降は $P = 2$ を除き、$N \in \{16, 64\}$ とする。

**対比量**: シード $s$ ごとに、テスト正解率 $a_{P,s}$ を $\log_2 N$ に最小二乗で回帰した傾き $\beta_s$ を求め、その平均
$\bar{\beta} = \frac{1}{n_B} \sum_s \beta_s$ を対比量とする($n_B$ は実験 B のシード数)。段階 1 以降は残る 2 水準で同じ定義を使う
(2 点の最小二乗の傾きは差を $\log_2$ の間隔 2 で割ったもの)。等間隔の 3 点($\log_2 N = 4, 6, 8$)の最小二乗の傾きは
$(a_{N=256} - a_{N=16}) / 4$ に等しく、中央の水準は傾きに寄与しない(中央の水準は曲線の形の診断に使う)。

**標準偏差の導出**: $\beta_s$ はシードごとに対応のある 3 条件(同じ初期化の種・データの順序・データ拡張)から作る線形結合で、
シード間で独立なので、$\sigma_B = \mathrm{sd}(\beta_s) / \sqrt{n_B}$。

**判定**: $\bar{\beta} > 2\sigma_B$ なら支持、$\bar{\beta} < -2\sigma_B$ なら反証、それ以外は判定不能。

**前提条件**: P0(ViT)、P1(使った全水準)。

**アサーション**: 条件間で非埋め込みパラメータ数が完全に一致すること(5.4 節、6.11 節)。パッチ埋め込み($P^2 C D + D$)と
位置埋め込み($(N+1)D$)のパラメータ数は $P$ によって変わる。

**作用点の記述**: $P$ が直接作用するのは系列長 $N$ と、1 トークンが表す画素の数 $P^2 C$ である。テスト正解率はその下流の量で、
しかも **$T$ を揃えても 1 ステップの計算量が $N$ とともに増える**(3.2.2 節の $F_{\mathrm{layer}}$、$P = 2$ は $P = 4$ の約 4.5 倍)ので、
「系列が長いこと」と「計算量が多いこと」の効果を分離できない(交絡)。診断量として、計算量の閉形式(1 枚の順伝播の演算数)と、
実行したデバイスでの 1 ステップあたりの時間の実測(6.3 節)を併記する(判定なし)。

#### 実験 C: 帰納バイアスとデータ量

**検証すること**: データ量の対数 $\log_2 \tilde{f}$ に対するテスト正解率の傾きが、ViT の方が CNN より大きい(3.5 節、データを減らしたとき
の正解率の落ち方が ViT の方が大きい)。CIFAR 規模では ViT が CNN を追い越さないことが予想されるので、**絶対値の逆転は判定しない**。

**条件**: ViT(S と同じ構成・同じ学習率)と CNN(ResNet-56、CNN の較正で選んだ学習率)のそれぞれについて、データ量
$f \in \{1/16, 1/4, 1\}$(入れ子の部分集合、5.3 節)で学習する。**学習ステップ数は選ばれた $T$ に固定する**(データ量が少ないほど
同じデータを多く繰り返す。エポック数は上の候補ごとの表のとおり)。ViT の $f = 1$ は S。

**対比量**: モデル $m \in \{\mathrm{ViT}, \mathrm{CNN}\}$・シード $s$ ごとに、テスト正解率を $\log_2 \tilde{f}$($\tilde{f}$ は実際の枚数の比、
5.3 節)に最小二乗で回帰した傾き $\gamma_{m,s}$ を求め、

$$
\Delta_C = \bar{\gamma}_{\mathrm{ViT}} - \bar{\gamma}_{\mathrm{CNN}}
$$

とする($\bar{\gamma}_m$ はシード平均)。

**標準偏差の導出**: ViT と CNN は異なるモデルで、同じシード番号でも初期化の乱数の使われ方が違うので、対応のない 2 群の差として
扱う。$s_m$ を $\gamma_{m,s}$ の不偏標本標準偏差、$n_C$ を実験 C のシード数として、

$$
\sigma_C = \sqrt{s_{\mathrm{ViT}}^2 / n_C + s_{\mathrm{CNN}}^2 / n_C}
$$

(2 つの独立な平均の差の分散は、各平均の分散の和)。

**判定**: $\Delta_C > 2\sigma_C$ なら支持、$\Delta_C < -2\sigma_C$ なら反証、それ以外は判定不能。

**前提条件**: P0(ViT・CNN の両方)、P1(ViT・CNN の各 3 水準)。

**交絡と注意**:

- **計算量とパラメータ数**: 1 枚あたりの計算量は、ResNet-56 が約 251 MFLOP、ViT($P = 4$)が約 366 MFLOP で、同じ桁に揃う(比は
  約 0.69。5.4 節で閉形式の値と比を印字する)。パラメータ数は ViT(約 2.69M)の方が ResNet-56(約 0.85M)より約 3 倍多い。
  「帰納バイアス」と「モデルの大きさ」の効果は分離できない。
- **CNN を ResNet-56 にした理由**: 旧版は torchvision の ResNet-18 を CIFAR 向けに変えたもの(約 11.2M パラメータ、1 枚あたり
  約 1111 MFLOP)を使っていた。1 枚あたりの計算量が ViT の約 3 倍あり、「帰納バイアス」と「計算量」の交絡が大きかったため、
  計算量が ViT と同じ桁で、原論文の CIFAR-10 の構成として定着した ResNet-56 に変えた。この変更は、どちらのモデルの学習結果も
  見る前に、閉形式の計算量とパラメータ数のみから決めた(旧版で実行したのはスモークテストのみである)。
- **天井効果**: CNN の正解率が全データで高い(1 に近い)と、正解率の上限のために $f$ を増やしたときの伸びが圧縮され、CNN の傾きが
  小さく出る(ViT の傾きが相対的に大きく見える)。その対処として、正解率のロジット $\mathrm{logit}(a) = \log(a / (1 - a))$ を
  $\log_2 \tilde{f}$ に回帰した傾きで同じ対比量と標準偏差を計算し、**診断量として** 併記する(判定には使わない)。
- 学習率は全データで選んだ値を全データ量に使うので、データ量ごとの最適な学習率の違いは交絡として残る。

**作用点の記述**: データ量が直接作用するのは、学習に使う異なる事例の数であり、それが最初に現れるのは **汎化の差**(学習データでの
正解率とテスト正解率の差)である。テスト正解率の傾きは、その下流で、かつ主張そのものを表す量なので対比量に選んだ。直接の
作用点に近い量として、全データ量の条件に共通に含まれる $S_{1/16}$ での正解率(データ拡張なし)と、それとテスト正解率との差を
診断量として併記する。

### 6.2 学習ステップ数 $T$ の候補と優先順位の決め方

**現在の決め方**: $T$ は固定せず、候補 $\{3968, 1984, 992\}$ と削る段階を組み合わせた実行計画から、本番の冒頭の T4 の見積もりのみで
自動選択する(6.1 節・6.4 節)。

- **最大の候補 3968**: 全データで約 11.3 エポックにあたる。CIFAR-10 で ViT を学習する標準的な設定(数十〜数百エポック)には届かないが、
  1 セッションの予算の中で全データを 10 回以上見られる大きさとして選んだ。
- **公比 2**: 学習量を下げるときの刻みを等比にする(学習量の効果は対数の尺度で現れると考えられるため)。最小の候補 992 は、旧版で
  固定していた値である。
- **候補が 3 つである理由**: 候補を増やすほど、スケーリングの計測は最大の候補を基準に行う必要がある一方で、選ばれる $T$ の精度は
  それほど上がらない。公比 2 の 3 点で、1 セッションの予算に対して 4 倍の幅を持たせた。

**旧版の経緯**(本番実行の前の改訂):

- 旧版は $T = 992$ に固定していた。T4 は本番の冒頭まで使えないので、ローカル(Apple M4 の MPS)で 1 ステップの時間を実測し、
  「T4 は MPS の 3 倍速い」と仮定して、段階 0 の見積もりが約 100 分になる $T$ を逆算した。仮定の根拠は、018 の同じ種類の計測
  (スクラッチ実装の Transformer の学習、FP32)で T4 が MPS の約 2.5 倍速かったことに、FP16 の autocast の分を上乗せしたもの
  だった。
- この決め方をやめた理由は 2 つある。(1)仮定の根拠が **別の種類の負荷**(018 の小さなモデルの LoRA 学習)の計測であり、本トピックの
  ViT・CNN の T4 での速さを代表しない。仮定が外れると、$T$ を不必要に小さくするか、段階を大きく削ることになる。(2)Colab での本番の
  実行を 1 回に保つには、T4 での計測を本番の冒頭で行い、そこで $T$ も選ぶ必要がある($T$ を決めるための事前の実行をしないため)。
- **この変更は結果に依存しない。** 旧版で実行したのはスモークテスト(ローカル、$T = 64$)のみであり、その判定の数値は変更の判断に
  使っていない。変更の根拠は、見積もりの仮定の妥当性と実行の手順だけである。判定基準(対比量・標準偏差の導出・閾値・期待する差の
  方向)と前提条件 P0・P1 は変えていない。

### 6.3 スケーリングの計測と外挿(1 セッションの見積もり)

本番でステップ数がスモークテストの何倍にもなる重い処理(4 種類のモデルの学習と評価)について、3 点の規模で実行時間を実測し、
$\log t = \log a + b \log n$ をあてはめてべき指数 $b$ を推定し、本番の規模へ外挿する。本番では、この計測を **本番の実行の冒頭に T4 上で**
行い、その値のみから削る段階を選ぶ(6.4 節)。外挿値と、最大の計測点の実測値を比例で伸ばした値の大きいほうを見積もりとする
(固定費があると $b < 1$ となり、外挿値が過小になりうるため)。

| 処理 | 計測する規模 | 外挿先(本番) | 本番での回数 |
|---|---|---|---|
| ViT $P = 8, 4, 2$ と CNN の学習(モデルの構築を含む) | ステップ数 $T_{\max}/32, T_{\max}/16, T_{\max}/8$($T_{\max} = 3968$ で 124・248・496) | 各候補の $T$(3968・1984・992) | 各計画の学習の数(6.4 節) + 較正 |
| ViT $P = 8, 4, 2$ と CNN の評価(FP32、データ拡張なし) | 画像の枚数 1,250・2,500・5,000 | 50,000 枚から 1 枚あたりに換算 | 学習ごとに約 3.3 万枚(実験 A の鍵は約 4.3 万枚) |

- あてはめたべき乗則から、各候補の $T$ への外挿値を出す。外挿値と比例の値の大きい方を使う規則は候補ごとに適用する。
- 計測の前に、各モデルで $T_{\max}/32$ ステップの準備運転を 1 回行う(cuDNN のアルゴリズム選択・MPS のカーネルの準備の時間を計測から
  除く)。計測専用のシード(シード番号 91)を使い、学習率は $10^{-3}$ とする(時間は学習率によらない)。
- 学習ごとの評価の枚数: 途中の検証 3 回 × 5,000 + 最後の検証 5,000 + テスト 10,000 + 共通部分 $S_{1/16}$ 2,810 = 32,810 枚。実験 A の
  鍵では並べ替えたテスト画像 10,000 枚を加える。較正の学習は検証のみの 20,000 枚。
- データの準備(ダウンロードを含む)・5.4 節の確認・この計測自体は実測値を使う。
- **スモークテストの出力の計画の見積もりは、ローカル(MPS)の値であり、T4 での本番の目安にならない。** 計画の選択には本番の冒頭の
  T4 での計測が使われる。


```python
def fit_and_extrapolate(label: str, sizes, times, targets) -> dict:
    # 外挿値と、最大の計測点の実測値を比例で伸ばした値の大きい方を、外挿先ごとに返す
    fit = fit_power_law_exponent(sizes, times)
    detail = ", ".join(f"n={n:,}: {t:.2f}s" for n, t in zip(sizes, times, strict=True))
    result, parts = {}, []
    for target in targets:
        extrapolated = fit.coefficient * target**fit.exponent
        proportional = times[-1] * target / sizes[-1]
        result[target] = max(extrapolated, proportional)
        parts.append(f"n={target:,.0f}: 外挿 {extrapolated:.1f}s・比例 {proportional:.1f}s")
    print(f"[{label}] {detail} -> b={fit.exponent:.3f}(標準誤差 {fit.exponent_stderr:.3f}), R^2={fit.r_squared:.4f}; " + "、".join(parts))
    return result


_t0_scaling = time.time()
TIMING_KEYS = {
    "vit_p8": key_vit(8, seed=TIMING_SEED_INDEX),
    "vit_p4": key_vit(4, seed=TIMING_SEED_INDEX),
    "vit_p2": key_vit(2, seed=TIMING_SEED_INDEX),
    "cnn": key_cnn(1.0, TIMING_SEED_INDEX),
}
EVAL_TIMING_SIZES = (1250, 2500, 5000)
ESTIMATE_TRAIN: dict[str, dict] = {}  # 本番の T の候補ごとの学習 1 回(秒)
ESTIMATE_EVAL_PER_IMAGE: dict[str, float] = {}  # 評価の 1 枚あたり(秒)
STEP_SECONDS: dict[str, float] = {}  # 最大の計測点での 1 ステップあたりの時間(実験 B の診断量)
for _kind, _key in TIMING_KEYS.items():
    train(build_model_for(_key), _key, 1e-3, SCALING_STEP_COUNTS[0])  # 準備運転(計測しない)
    _times = [
        timed_call(lambda n=n: train(build_model_for(_key), _key, 1e-3, n)) for n in SCALING_STEP_COUNTS
    ]
    ESTIMATE_TRAIN[_kind] = fit_and_extrapolate(
        f"{_kind} の学習(ステップ数)", SCALING_STEP_COUNTS, _times, PROD_T_CANDIDATES
    )
    STEP_SECONDS[_kind] = _times[-1] / SCALING_STEP_COUNTS[-1]
    _model = build_model_for(_key)
    _arch = _key[0]
    evaluate(_model, TRAIN_IMAGES[:500], TRAIN_LABELS[:500], _arch)  # 準備運転
    _eval_times = [
        timed_call(lambda n=n: evaluate(_model, TRAIN_IMAGES[:n], TRAIN_LABELS[:n], _arch))
        for n in EVAL_TIMING_SIZES
    ]
    ESTIMATE_EVAL_PER_IMAGE[_kind] = (
        fit_and_extrapolate(f"{_kind} の評価(画像の枚数)", EVAL_TIMING_SIZES, _eval_times, (50_000,))[50_000] / 50_000
    )
    del _model
    empty_device_cache()
SCALING_SECONDS = time.time() - _t0_scaling
print(f"スケーリングの計測自体: {SCALING_SECONDS:.1f}s")
print(
    f"1 ステップあたりの時間({device}、最大の計測点): "
    + "、".join(f"{k} {v * 1000:.1f} ms" for k, v in STEP_SECONDS.items())
)
```

    [vit_p8 の学習(ステップ数)] n=124: 4.44s, n=248: 9.57s, n=496: 18.17s -> b=1.017(標準誤差 0.053), R^2=0.9973; n=3,968: 外挿 153.7s・比例 145.4s、n=1,984: 外挿 76.0s・比例 72.7s、n=992: 外挿 37.5s・比例 36.3s
    [vit_p8 の評価(画像の枚数)] n=1,250: 0.08s, n=2,500: 0.14s, n=5,000: 0.29s -> b=0.906(標準誤差 0.084), R^2=0.9915; n=50,000: 外挿 2.3s・比例 2.9s
    [vit_p4 の学習(ステップ数)] n=124: 7.52s, n=248: 15.10s, n=496: 30.10s -> b=1.000(標準誤差 0.003), R^2=1.0000; n=3,968: 外挿 241.0s・比例 240.8s、n=1,984: 外挿 120.5s・比例 120.4s、n=992: 外挿 60.3s・比例 60.2s
    [vit_p4 の評価(画像の枚数)] n=1,250: 0.30s, n=2,500: 0.56s, n=5,000: 1.13s -> b=0.960(標準誤差 0.030), R^2=0.9990; n=50,000: 外挿 10.1s・比例 11.3s
    [vit_p2 の学習(ステップ数)] n=124: 30.17s, n=248: 61.17s, n=496: 125.47s -> b=1.028(標準誤差 0.005), R^2=1.0000; n=3,968: 外挿 1062.1s・比例 1003.8s、n=1,984: 外挿 520.8s・比例 501.9s、n=992: 外挿 255.4s・比例 250.9s
    [vit_p2 の評価(画像の枚数)] n=1,250: 1.65s, n=2,500: 3.25s, n=5,000: 6.52s -> b=0.992(標準誤差 0.007), R^2=1.0000; n=50,000: 外挿 63.8s・比例 65.2s
    [cnn の学習(ステップ数)] n=124: 5.69s, n=248: 11.89s, n=496: 23.58s -> b=1.025(標準誤差 0.022), R^2=0.9996; n=3,968: 外挿 200.5s・比例 188.6s、n=1,984: 外挿 98.5s・比例 94.3s、n=992: 外挿 48.4s・比例 47.2s
    [cnn の評価(画像の枚数)] n=1,250: 0.32s, n=2,500: 0.45s, n=5,000: 0.90s -> b=0.748(標準誤差 0.153), R^2=0.9597; n=50,000: 外挿 4.7s・比例 9.0s
    スケーリングの計測自体: 411.4s
    1 ステップあたりの時間(cuda、最大の計測点): vit_p8 36.6 ms、vit_p4 60.7 ms、vit_p2 253.0 ms、cnn 47.5 ms


### 6.4 実行計画の選択($T$ と削る段階)

6.3 節の外挿値から、12 通りの実行計画(6.1 節)ごとに残りの実行時間(較正と本番の学習・評価)を見積もり、予算
「120 分 − ノートブックの開始からの経過時間」に収まる番号の最も小さい計画を選ぶ。**選択は見積もりのみに基づき、どの実験の結果も
参照しない。** この時点では、較正も本番の学習も行っていない。較正は、格子の拡張が起きる場合(各モデル 4 回の学習)を、選ばれた
$T$ で行う前提で見積もりに含める(安全側)。


```python
VALIDATION_SIZE = len(VALIDATION_INDICES)
EVAL_IMAGES_MAIN = (len(INTERMEDIATE_EVAL_FRACTIONS) + 1) * VALIDATION_SIZE + len(TEST_LABELS) + len(COMMON_SUBSET_LABELS)
EVAL_IMAGES_A_EXTRA = len(TEST_LABELS) + ATTENTION_DISTANCE_IMAGES  # 並べ替えたテスト画像(と注意距離、安全側)
EVAL_IMAGES_CALIBRATION = (len(INTERMEDIATE_EVAL_FRACTIONS) + 1) * VALIDATION_SIZE
CALIBRATION_KEYS = {"vit": key_vit(seed=CALIBRATION_SEED_INDEX), "cnn": key_cnn(1.0, CALIBRATION_SEED_INDEX)}


def kind_of(key: tuple) -> str:
    return "cnn" if key[0] == "cnn" else f"vit_p{key[1]}"


def run_estimate(key: tuple, num_steps: int, purpose: str = "main") -> float:
    kind = kind_of(key)
    if purpose == "calibration":
        images = EVAL_IMAGES_CALIBRATION
    else:
        images = EVAL_IMAGES_MAIN + (EVAL_IMAGES_A_EXTRA if is_a_key(key) else 0)
    return ESTIMATE_TRAIN[kind][num_steps] + images * ESTIMATE_EVAL_PER_IMAGE[kind]


def plan_runs(stage: dict) -> list[tuple]:
    # 段階が決める学習の鍵の一覧(重複を除き、実験 A -> B -> C の順)
    keys: list[tuple] = []
    for s in range(stage["SEEDS_A"]):
        keys += [key_vit(STANDARD_PATCH_SIZE, pe, 1.0, s) for pe in POSITION_EMBEDDINGS]
    for s in range(stage["SEEDS_B"]):
        keys += [key_vit(p, "learned", 1.0, s) for p in stage["B_PATCH_SIZES"]]
    for s in range(stage["SEEDS_C"]):
        keys += [key_vit(STANDARD_PATCH_SIZE, "learned", f, s) for f in DATA_FRACTIONS]
        keys += [key_cnn(f, s) for f in DATA_FRACTIONS]
    return list(dict.fromkeys(keys))


def estimate_plan(plan: dict) -> dict:
    # 本番の値の計画(T は本番の候補)の残りの実行時間の内訳
    t = plan["T"]
    runs = plan_runs(STAGES["prod"][plan["stage"]])
    parts = {
        "較正(ViT・CNN 各 4 学習、拡張を含む安全側)": sum(
            (len(LR_GRID[arch]) + 1) * run_estimate(key, t, "calibration") for arch, key in CALIBRATION_KEYS.items()
        )
    }
    for key in runs:
        name = f"本番: {kind_of(key)}"
        parts[name] = parts.get(name, 0.0) + run_estimate(key, t)
    return parts


class PlanBudgetExceededError(RuntimeError):
    pass


def select_plan(totals: dict[int, float], budget_seconds: float) -> int:
    # 予算内に収まる番号の最も小さい計画
    for plan in sorted(totals):
        if totals[plan] <= budget_seconds:
            return plan
    raise PlanBudgetExceededError(
        f"計画 {max(totals)} でも見積もり {totals[max(totals)] / 60:.1f} 分が予算 {budget_seconds / 60:.1f} 分を超える"
    )


ELAPSED_BEFORE_SELECTION = time.time() - NOTEBOOK_START_TIME
REMAINING_BUDGET_SECONDS = SESSION_BUDGET_SECONDS - ELAPSED_BEFORE_SELECTION
PLAN_ESTIMATES = {p["plan"]: estimate_plan(p) for p in PLANS["prod"]}
PLAN_TOTALS = {k: sum(v.values()) for k, v in PLAN_ESTIMATES.items()}
print(
    f"すでに使った時間(ノートブックの開始から、データの準備・確認・計測を含む): {ELAPSED_BEFORE_SELECTION / 60:.1f} 分"
    f"(データの準備 {DATA_SECONDS:.0f}s、確認 {CHECK_SECONDS:.0f}s、スケーリングの計測 {SCALING_SECONDS:.0f}s)"
)
print(
    f"残りの予算: {SESSION_BUDGET_SECONDS / 60:.0f} 分 - {ELAPSED_BEFORE_SELECTION / 60:.1f} 分 = {REMAINING_BUDGET_SECONDS / 60:.1f} 分"
)
print("1 回の学習と評価の見積もり(秒、本番の T の候補ごと):")
for _kind, _key in TIMING_KEYS.items():
    print(f"  {_kind}: " + "、".join(f"T = {t}: {run_estimate(_key, t):.1f}s" for t in PROD_T_CANDIDATES))
print(f"\n--- 実行計画ごとの残りの実行時間の見積もり({device} 基準、残りの予算 {REMAINING_BUDGET_SECONDS / 60:.1f} 分)---")
print("計画 | T | 段階 | 学習の数(本番 + 較正) | 見積もり(分) | 残りの予算に対する比")
for _p in PLANS["prod"]:
    _total = PLAN_TOTALS[_p["plan"]]
    print(
        f"{_p['plan']:4d} | {_p['T']:4d} | {_p['stage']} | {len(plan_runs(STAGES['prod'][_p['stage']]))} + 8 | "
        f"{_total / 60:8.1f} | {_total / max(REMAINING_BUDGET_SECONDS, 1e-9):.1%}"
    )
for _t in PROD_T_CANDIDATES:  # 同じ T では段階が上がると見積もりが減る
    _same = [PLAN_TOTALS[p["plan"]] for p in PLANS["prod"] if p["T"] == _t]
    assert all(a >= b for a, b in zip(_same, _same[1:], strict=False)), _t
print("--- 優先順位の最も高い計画(計画 0)の内訳 ---")
for _k, _v in PLAN_ESTIMATES[0].items():
    print(f"  {_k}: {_v:,.1f}s")

try:
    SELECTED_PLAN = select_plan(PLAN_TOTALS, REMAINING_BUDGET_SECONDS)
    PLAN_SELECTION_MESSAGE = (
        f"残りの予算 {REMAINING_BUDGET_SECONDS / 60:.1f} 分に収まる番号の最も小さい計画として、計画 {SELECTED_PLAN} を選んだ"
        f"(見積もり {PLAN_TOTALS[SELECTED_PLAN] / 60:.1f} 分、{device} 基準)"
    )
except PlanBudgetExceededError as _error:
    if not SMOKE_TEST:
        print(f"\n警告: {_error}。本番の学習を始める前に停止する。")
        raise
    SELECTED_PLAN = max(PLAN_TOTALS)
    PLAN_SELECTION_MESSAGE = (
        f"{_error}(本番なら学習の前に停止する)。スモークテストのため停止せず、計画 {SELECTED_PLAN} で動作確認を続ける"
    )
if FORCED_PLAN is not None:  # テスト専用の上書き(5.2 節。スモークテストでのみ有効)
    assert SMOKE_TEST
    print("\n" + "!" * 100)
    print(
        f"!!! テスト専用の上書き: AI_THEORIES_FORCE_PLAN={FORCED_PLAN} により、見積もりによる選択(計画 {SELECTED_PLAN})の"
        f"代わりに計画 {FORCED_PLAN} を使う(スモークテストのみ。この出力は通常の実行の記録ではない)"
    )
    print("!" * 100)
    PLAN_SELECTION_MESSAGE = f"テスト専用の上書きで計画 {FORCED_PLAN} を強制(見積もりによる選択は計画 {SELECTED_PLAN})"
    SELECTED_PLAN = FORCED_PLAN

if SMOKE_TEST:  # 予算を人為的に変えて、選択の規則の分岐を確かめる(予算の定数は変えない)
    for _budget in sorted(set(PLAN_TOTALS.values())):
        _expected = min(k for k, v in PLAN_TOTALS.items() if v <= _budget)
        assert select_plan(PLAN_TOTALS, _budget) == _expected
    try:
        select_plan(PLAN_TOTALS, 0.5 * min(PLAN_TOTALS.values()))
        raise AssertionError("どの計画でも超える予算で停止しなかった")
    except PlanBudgetExceededError:
        pass
    print(
        "選択の規則の確認(人為的な予算、確認のみ): 各計画の見積もりに等しい予算で「番号の最も小さい収まる計画」が選ばれる、"
        f"どの計画でも超える予算で停止する: OK。予算の定数は {SESSION_BUDGET_SECONDS / 60:.0f} 分のまま"
    )
    assert SESSION_BUDGET_SECONDS == 120 * 60

# --- 選ばれた計画の値(以降のすべてのセルがこれを使う) ---
SELECTED = PLANS[CURRENT_LEVEL_NAME][SELECTED_PLAN]
TRAIN_STEPS = SELECTED["T"]  # T
SELECTED_STAGE = SELECTED["stage"]
INTERMEDIATE_EVAL_STEPS = intermediate_eval_steps_for(TRAIN_STEPS)
STAGE = STAGES[CURRENT_LEVEL_NAME][SELECTED_STAGE]
SEEDS_A = tuple(range(STAGE["SEEDS_A"]))
SEEDS_B = tuple(range(STAGE["SEEDS_B"]))
SEEDS_C = tuple(range(STAGE["SEEDS_C"]))
B_PATCH_SIZES = STAGE["B_PATCH_SIZES"]
PLAN = plan_runs(STAGE)
SEEDS_S = tuple(sorted({k[4] for k in PLAN if k == key_vit(seed=k[4])}))
print(f"\n計画の選択: {PLAN_SELECTION_MESSAGE}")
print(
    f"選ばれた計画 {SELECTED_PLAN}(水準 {CURRENT_LEVEL_NAME!r}): T = {TRAIN_STEPS}、段階 {SELECTED_STAGE}、"
    f"warmup {warmup_steps_for(TRAIN_STEPS)} ステップ、途中の評価 {INTERMEDIATE_EVAL_STEPS}"
)
print(
    f"このノートブックで使う値: 実験 A のシード {SEEDS_A}、実験 B のシード {SEEDS_B}・P {B_PATCH_SIZES}、"
    f"実験 C のシード {SEEDS_C}、標準条件 S のシード {SEEDS_S}、本番の学習 {len(PLAN)} 回"
)
if device.type != "cuda":
    print("注意: CUDA 以外での見積もりであり、T4 での時間とは異なる")
```

    すでに使った時間(ノートブックの開始から、データの準備・確認・計測を含む): 21.2 分(データの準備 847s、確認 1s、スケーリングの計測 411s)
    残りの予算: 120 分 - 21.2 分 = 98.8 分
    1 回の学習と評価の見積もり(秒、本番の T の候補ごと):
      vit_p8: T = 3968: 155.6s、T = 1984: 77.9s、T = 992: 39.5s
      vit_p4: T = 3968: 250.9s、T = 1984: 130.4s、T = 992: 70.1s
      vit_p2: T = 3968: 1104.9s、T = 1984: 563.6s、T = 992: 298.2s
      cnn: T = 3968: 206.5s、T = 1984: 104.4s、T = 992: 54.3s
    
    --- 実行計画ごとの残りの実行時間の見積もり(cuda 基準、残りの予算 98.8 分)---
    計画 | T | 段階 | 学習の数(本番 + 較正) | 見積もり(分) | 残りの予算に対する比
       0 | 3968 | 0 | 50 + 8 |    290.8 | 294.4%
       1 | 3968 | 1 | 45 + 8 |    198.7 | 201.2%
       2 | 3968 | 2 | 35 + 8 |    161.5 | 163.5%
       3 | 3968 | 3 | 31 + 8 |    144.8 | 146.6%
       4 | 1984 | 0 | 50 + 8 |    148.6 | 150.5%
       5 | 1984 | 1 | 45 + 8 |    101.7 | 102.9%
       6 | 1984 | 2 | 35 + 8 |     82.7 | 83.7%
       7 | 1984 | 3 | 31 + 8 |     74.0 | 74.9%
       8 |  992 | 0 | 50 + 8 |     78.3 | 79.3%
       9 |  992 | 1 | 45 + 8 |     53.5 | 54.1%
      10 |  992 | 2 | 35 + 8 |     43.5 | 44.1%
      11 |  992 | 3 | 31 + 8 |     38.8 | 39.3%
    --- 優先順位の最も高い計画(計画 0)の内訳 ---
      較正(ViT・CNN 各 4 学習、拡張を含む安全側): 1,798.8s
      本番: vit_p4: 6,247.8s
      本番: vit_p8: 778.1s
      本番: vit_p2: 5,524.6s
      本番: cnn: 3,096.9s
    
    計画の選択: 残りの予算 98.8 分に収まる番号の最も小さい計画として、計画 6 を選んだ(見積もり 82.7 分、cuda 基準)
    選ばれた計画 6(水準 'prod'): T = 1984、段階 2、warmup 198 ステップ、途中の評価 (496, 992, 1488)
    このノートブックで使う値: 実験 A のシード (0, 1, 2, 3, 4)、実験 B のシード (0, 1, 2, 3, 4)・P (8, 4)、実験 C のシード (0, 1, 2)、標準条件 S のシード (0, 1, 2, 3, 4)、本番の学習 35 回


### 6.5 学習率の較正(P0)

6.1 節の規則で、ViT(標準の構成)と CNN の学習率を別々に選ぶ。各格子点で、6.4 節で選ばれた $T$・全データ・較正専用のシードで学習し、
**検証集合の正解率のみ** を見る(テスト集合は評価しない)。


```python
_t0_calibration = time.time()


def calibrate(arch: str) -> dict:
    key = key_vit(seed=CALIBRATION_SEED_INDEX) if arch == "vit" else key_cnn(1.0, CALIBRATION_SEED_INDEX)
    records: dict[float, dict] = {}

    def best() -> float:  # 検証集合の正解率が最大(同点なら小さい学習率)
        return max(sorted(records), key=lambda lr: (records[lr]["val"]["accuracy"], -lr))

    for lr in LR_GRID[arch]:
        records[lr] = run(key, lr, "calibration")
        print(f"  {_tag}較正 {describe(records[lr])}")
    extended = None
    if best() == min(records):
        extended = min(records) / LR_GRID_RATIO
    elif best() == max(records):
        extended = max(records) * LR_GRID_RATIO
    if extended is not None:  # 端に来たら、その方向に 1 点だけ拡張する(1 回のみ)
        records[extended] = run(key, extended, "calibration")
        print(f"  {_tag}較正(拡張) {describe(records[extended])}")
    chosen = best()
    return {
        "records": records,
        "chosen": chosen,
        "extended": extended,
        "interior": min(records) < chosen < max(records),
    }


CALIBRATION = {}
for _arch in ("vit", "cnn"):
    print(f"--- {_arch} の較正(格子 {LR_GRID[_arch]}) ---")
    CALIBRATION[_arch] = calibrate(_arch)
LEARNING_RATE = {arch: c["chosen"] for arch, c in CALIBRATION.items()}
precondition_status["P0(ViT)"] = CALIBRATION["vit"]["interior"]
precondition_status["P0(CNN)"] = CALIBRATION["cnn"]["interior"]
CALIBRATION_SECONDS = time.time() - _t0_calibration
for _arch, _c in CALIBRATION.items():
    _row = "、".join(f"{lr:g}: {r['val']['accuracy']:.4f}" for lr, r in sorted(_c["records"].items()))
    print(
        f"{_tag}{_arch}: 検証集合の正解率 {{{_row}}} -> 選んだ学習率 {_c['chosen']:g}"
        f"(拡張 {'なし' if _c['extended'] is None else format(_c['extended'], 'g')}、内点 {_c['interior']})"
    )
print(
    f"{_tag}前提条件 P0(ViT) = {precondition_status['P0(ViT)']}、P0(CNN) = {precondition_status['P0(CNN)']}、"
    f"較正の実行時間 {CALIBRATION_SECONDS / 60:.1f} 分"
)
```

    --- vit の較正(格子 (0.0005, 0.001, 0.002)) ---
      較正 ViT P=4 learned f=1.0000 s=90 lr=0.0005: val 0.5884、最終の訓練損失 1.445、飛ばしたステップ 0、132s
      較正 ViT P=4 learned f=1.0000 s=90 lr=0.001: val 0.5870、最終の訓練損失 1.359、飛ばしたステップ 0、131s
      較正 ViT P=4 learned f=1.0000 s=90 lr=0.002: val 0.4434、最終の訓練損失 1.709、飛ばしたステップ 0、132s
      較正(拡張) ViT P=4 learned f=1.0000 s=90 lr=0.00025: val 0.5616、最終の訓練損失 1.474、飛ばしたステップ 0、132s
    --- cnn の較正(格子 (0.0005, 0.001, 0.002)) ---
      較正 CNN f=1.0000 s=90 lr=0.0005: val 0.7314、最終の訓練損失 0.920、飛ばしたステップ 1、99s
      較正 CNN f=1.0000 s=90 lr=0.001: val 0.7926、最終の訓練損失 0.666、飛ばしたステップ 1、103s
      較正 CNN f=1.0000 s=90 lr=0.002: val 0.8274、最終の訓練損失 0.603、飛ばしたステップ 1、100s
      較正(拡張) CNN f=1.0000 s=90 lr=0.004: val 0.8384、最終の訓練損失 0.476、飛ばしたステップ 1、100s
    vit: 検証集合の正解率 {0.00025: 0.5616、0.0005: 0.5884、0.001: 0.5870、0.002: 0.4434} -> 選んだ学習率 0.0005(拡張 0.00025、内点 True)
    cnn: 検証集合の正解率 {0.0005: 0.7314、0.001: 0.7926、0.002: 0.8274、0.004: 0.8384} -> 選んだ学習率 0.004(拡張 0.004、内点 False)
    前提条件 P0(ViT) = True、P0(CNN) = False、較正の実行時間 15.5 分


### 6.6 本番の学習と評価(実験 A・B・C)

6.4 節で選んだ段階の学習(`PLAN`)をすべて行い、辞書`RECORDS`に鍵ごとに記録する。同じ鍵の学習は 1 回だけ行う(標準条件 S を
実験 A・B・C で共有する)。学習したモデルは評価の直後に破棄する。


```python
_t0_training = time.time()
RECORDS: dict[tuple, dict] = {}
for _i, _key in enumerate(PLAN):
    RECORDS[_key] = run(_key, LEARNING_RATE[_key[0]], "main")
    print(f"{_tag}[{_i + 1}/{len(PLAN)}] {describe(RECORDS[_key])}")
TRAINING_SECONDS = time.time() - _t0_training
print(f"本番の学習と評価: {len(RECORDS)} 学習、{TRAINING_SECONDS / 60:.1f} 分")


def test_accuracy(key: tuple) -> float:
    return RECORDS[key]["test"]["accuracy"]
```

    [1/35] ViT P=4 none f=1.0000 s=0 lr=0.0005: val 0.5460、test 0.5410(CE 1.289)、最終の訓練損失 1.253、飛ばしたステップ 0、138s
    [2/35] ViT P=4 learned f=1.0000 s=0 lr=0.0005: val 0.5836、test 0.5832(CE 1.151)、最終の訓練損失 1.142、飛ばしたステップ 0、139s
    [3/35] ViT P=4 sinusoidal_2d f=1.0000 s=0 lr=0.0005: val 0.6308、test 0.6225(CE 1.062)、最終の訓練損失 1.066、飛ばしたステップ 0、139s
    [4/35] ViT P=4 none f=1.0000 s=1 lr=0.0005: val 0.5720、test 0.5496(CE 1.248)、最終の訓練損失 1.187、飛ばしたステップ 0、138s
    [5/35] ViT P=4 learned f=1.0000 s=1 lr=0.0005: val 0.6022、test 0.5889(CE 1.153)、最終の訓練損失 1.064、飛ばしたステップ 0、138s
    [6/35] ViT P=4 sinusoidal_2d f=1.0000 s=1 lr=0.0005: val 0.6262、test 0.6149(CE 1.059)、最終の訓練損失 1.019、飛ばしたステップ 0、139s
    [7/35] ViT P=4 none f=1.0000 s=2 lr=0.0005: val 0.5526、test 0.5420(CE 1.283)、最終の訓練損失 1.149、飛ばしたステップ 0、139s
    [8/35] ViT P=4 learned f=1.0000 s=2 lr=0.0005: val 0.5804、test 0.5693(CE 1.198)、最終の訓練損失 1.033、飛ばしたステップ 0、139s
    [9/35] ViT P=4 sinusoidal_2d f=1.0000 s=2 lr=0.0005: val 0.6266、test 0.6175(CE 1.048)、最終の訓練損失 0.935、飛ばしたステップ 0、139s
    [10/35] ViT P=4 none f=1.0000 s=3 lr=0.0005: val 0.5464、test 0.5440(CE 1.273)、最終の訓練損失 1.251、飛ばしたステップ 0、138s
    [11/35] ViT P=4 learned f=1.0000 s=3 lr=0.0005: val 0.6024、test 0.5935(CE 1.145)、最終の訓練損失 1.189、飛ばしたステップ 0、139s
    [12/35] ViT P=4 sinusoidal_2d f=1.0000 s=3 lr=0.0005: val 0.6268、test 0.6215(CE 1.065)、最終の訓練損失 0.965、飛ばしたステップ 0、138s
    [13/35] ViT P=4 none f=1.0000 s=4 lr=0.0005: val 0.5558、test 0.5461(CE 1.264)、最終の訓練損失 1.312、飛ばしたステップ 0、138s
    [14/35] ViT P=4 learned f=1.0000 s=4 lr=0.0005: val 0.5998、test 0.5841(CE 1.157)、最終の訓練損失 1.206、飛ばしたステップ 0、138s
    [15/35] ViT P=4 sinusoidal_2d f=1.0000 s=4 lr=0.0005: val 0.6400、test 0.6220(CE 1.042)、最終の訓練損失 1.118、飛ばしたステップ 0、138s
    [16/35] ViT P=8 learned f=1.0000 s=0 lr=0.0005: val 0.4868、test 0.4816(CE 1.430)、最終の訓練損失 1.379、飛ばしたステップ 0、78s
    [17/35] ViT P=8 learned f=1.0000 s=1 lr=0.0005: val 0.4890、test 0.4784(CE 1.426)、最終の訓練損失 1.335、飛ばしたステップ 0、77s
    [18/35] ViT P=8 learned f=1.0000 s=2 lr=0.0005: val 0.4876、test 0.4819(CE 1.428)、最終の訓練損失 1.319、飛ばしたステップ 0、78s
    [19/35] ViT P=8 learned f=1.0000 s=3 lr=0.0005: val 0.4732、test 0.4712(CE 1.456)、最終の訓練損失 1.344、飛ばしたステップ 0、78s
    [20/35] ViT P=8 learned f=1.0000 s=4 lr=0.0005: val 0.4890、test 0.4903(CE 1.417)、最終の訓練損失 1.394、飛ばしたステップ 0、77s
    [21/35] ViT P=4 learned f=0.0625 s=0 lr=0.0005: val 0.4904、test 0.4771(CE 1.959)、最終の訓練損失 0.284、飛ばしたステップ 0、136s
    [22/35] ViT P=4 learned f=0.2500 s=0 lr=0.0005: val 0.5790、test 0.5735(CE 1.201)、最終の訓練損失 1.058、飛ばしたステップ 0、135s
    [23/35] CNN f=0.0625 s=0 lr=0.004: val 0.7082、test 0.6998(CE 1.735)、最終の訓練損失 0.002、飛ばしたステップ 1、101s
    [24/35] CNN f=0.2500 s=0 lr=0.004: val 0.8184、test 0.8104(CE 0.626)、最終の訓練損失 0.307、飛ばしたステップ 1、101s
    [25/35] CNN f=1.0000 s=0 lr=0.004: val 0.8368、test 0.8253(CE 0.516)、最終の訓練損失 0.572、飛ばしたステップ 1、101s
    [26/35] ViT P=4 learned f=0.0625 s=1 lr=0.0005: val 0.4848、test 0.4744(CE 1.949)、最終の訓練損失 0.264、飛ばしたステップ 0、136s
    [27/35] ViT P=4 learned f=0.2500 s=1 lr=0.0005: val 0.5858、test 0.5717(CE 1.206)、最終の訓練損失 1.133、飛ばしたステップ 0、136s
    [28/35] CNN f=0.0625 s=1 lr=0.004: val 0.7108、test 0.7041(CE 1.777)、最終の訓練損失 0.005、飛ばしたステップ 1、102s
    [29/35] CNN f=0.2500 s=1 lr=0.004: val 0.8080、test 0.8035(CE 0.639)、最終の訓練損失 0.343、飛ばしたステップ 1、101s
    [30/35] CNN f=1.0000 s=1 lr=0.004: val 0.8386、test 0.8213(CE 0.518)、最終の訓練損失 0.528、飛ばしたステップ 1、101s
    [31/35] ViT P=4 learned f=0.0625 s=2 lr=0.0005: val 0.4800、test 0.4767(CE 1.912)、最終の訓練損失 0.322、飛ばしたステップ 0、135s
    [32/35] ViT P=4 learned f=0.2500 s=2 lr=0.0005: val 0.5848、test 0.5757(CE 1.196)、最終の訓練損失 1.125、飛ばしたステップ 0、135s
    [33/35] CNN f=0.0625 s=2 lr=0.004: val 0.6994、test 0.6932(CE 1.782)、最終の訓練損失 0.004、飛ばしたステップ 1、101s
    [34/35] CNN f=0.2500 s=2 lr=0.004: val 0.8142、test 0.8048(CE 0.641)、最終の訓練損失 0.389、飛ばしたステップ 1、101s
    [35/35] CNN f=1.0000 s=2 lr=0.004: val 0.8372、test 0.8287(CE 0.503)、最終の訓練損失 0.474、飛ばしたステップ 2、102s
    本番の学習と評価: 35 学習、69.8 分


### 6.7 実験 A: 位置埋め込みの寄与


```python
A_ACCURACY = np.array(
    [[test_accuracy(key_vit(STANDARD_PATCH_SIZE, pe, 1.0, s)) for s in SEEDS_A] for pe in POSITION_EMBEDDINGS]
)  # (3 条件, n_A)
_d = A_ACCURACY[1] - A_ACCURACY[0]
CONTRAST_A = {"value": float(_d.mean()), "sigma": float(_d.std(ddof=1) / math.sqrt(len(SEEDS_A)))}
verdict_A = judge(CONTRAST_A["value"], CONTRAST_A["sigma"])
precondition_status["P1(A)"] = bool((A_ACCURACY.mean(axis=1) >= ACCURACY_MIN).all())

print(f"{_tag}テスト正解率(行: 条件 0 なし・1 学習可能な 1 次元・2 の 2 次元正弦波、列: シード {SEEDS_A})")
for _c, _pe in enumerate(POSITION_EMBEDDINGS):
    _shuffled = np.array([RECORDS[key_vit(STANDARD_PATCH_SIZE, _pe, 1.0, s)]["shuffled_test"]["accuracy"] for s in SEEDS_A])
    _ce = np.array([RECORDS[key_vit(STANDARD_PATCH_SIZE, _pe, 1.0, s)]["test"]["cross_entropy"] for s in SEEDS_A])
    print(
        f"{_tag}  条件 {_c}({_pe}): {np.round(A_ACCURACY[_c], 4).tolist()} 平均 {A_ACCURACY[_c].mean():.4f}、"
        f"テストの交差エントロピー平均 {_ce.mean():.4f}、並べ替えたテスト画像の正解率 平均 {_shuffled.mean():.4f}"
        f"(低下幅 {(A_ACCURACY[_c] - _shuffled).mean():+.4f})"
    )
print(f"{_tag}d_s = a_1 - a_0: {np.round(_d, 4).tolist()}")
print(
    f"{_tag}対比量 Delta_A = {CONTRAST_A['value']:+.4f}、sigma_A = sd(d_s)/sqrt({len(SEEDS_A)}) = {CONTRAST_A['sigma']:.4f}、"
    f"閾値 2 sigma_A = {2 * CONTRAST_A['sigma']:.4f} -> 判定関数の結果: {verdict_A}"
)
print(
    f"{_tag}診断量(判定なし): 条件 2 - 条件 1 = {(A_ACCURACY[2] - A_ACCURACY[1]).mean():+.4f}"
    f"(シードごと {np.round(A_ACCURACY[2] - A_ACCURACY[1], 4).tolist()})"
)
print(
    f"{_tag}前提条件: P0(ViT) = {precondition_status['P0(ViT)']}、P1(A)(シード平均 >= {ACCURACY_MIN}) = "
    f"{precondition_status['P1(A)']}"
)

_fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
for _s_i, _s in enumerate(SEEDS_A):
    _axes[0].plot(range(3), A_ACCURACY[:, _s_i], marker="o", color="gray", alpha=0.6)
_axes[0].plot(range(3), A_ACCURACY.mean(axis=1), marker="s", color="black", lw=2, label="seed mean")
_axes[0].set_xticks(range(3), [f"{i}: {pe}" for i, pe in enumerate(POSITION_EMBEDDINGS)])
_axes[0].set_ylabel("test accuracy")
_axes[0].set_title(f"{_plot_tag}Experiment A: position embedding (lines = seeds)")
_axes[0].legend()
_curve_steps = list(INTERMEDIATE_EVAL_STEPS) + [TRAIN_STEPS]
for _c, _pe in enumerate(POSITION_EMBEDDINGS):
    _curves = np.array(
        [
            [RECORDS[key_vit(STANDARD_PATCH_SIZE, _pe, 1.0, s)]["history"]["evaluations"][t]["val_accuracy"] for t in INTERMEDIATE_EVAL_STEPS]
            + [RECORDS[key_vit(STANDARD_PATCH_SIZE, _pe, 1.0, s)]["val"]["accuracy"]]
            for s in SEEDS_A
        ]
    )
    _axes[1].plot(_curve_steps, _curves.mean(axis=0), marker="o", label=f"{_c}: {_pe}")
_axes[1].set_xlabel("step")
_axes[1].set_ylabel("validation accuracy (seed mean)")
_axes[1].set_title("Validation accuracy during training")
_axes[1].legend()
plt.tight_layout()
plt.show()
```

    テスト正解率(行: 条件 0 なし・1 学習可能な 1 次元・2 の 2 次元正弦波、列: シード (0, 1, 2, 3, 4))
      条件 0(none): [0.541, 0.5496, 0.542, 0.544, 0.5461] 平均 0.5445、テストの交差エントロピー平均 1.2716、並べ替えたテスト画像の正解率 平均 0.5445(低下幅 +0.0000)
      条件 1(learned): [0.5832, 0.5889, 0.5693, 0.5935, 0.5841] 平均 0.5838、テストの交差エントロピー平均 1.1607、並べ替えたテスト画像の正解率 平均 0.5089(低下幅 +0.0749)
      条件 2(sinusoidal_2d): [0.6225, 0.6149, 0.6175, 0.6215, 0.622] 平均 0.6197、テストの交差エントロピー平均 1.0551、並べ替えたテスト画像の正解率 平均 0.4002(低下幅 +0.2195)
    d_s = a_1 - a_0: [0.0422, 0.0393, 0.0273, 0.0495, 0.038]
    対比量 Delta_A = +0.0393、sigma_A = sd(d_s)/sqrt(5) = 0.0036、閾値 2 sigma_A = 0.0072 -> 判定関数の結果: 支持
    診断量(判定なし): 条件 2 - 条件 1 = +0.0359(シードごと [0.0393, 0.026, 0.0482, 0.028, 0.0379])
    前提条件: P0(ViT) = True、P1(A)(シード平均 >= 0.25) = True



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/019_vision_transformer/output_33_1.png)
    


### 6.8 観察: 位置埋め込みの類似度と注意距離(判定基準を設けない)

**判定基準を設けない観察である。** 実験 A のシード 0 のモデルについて、次を描く。

- 学習可能な 1 次元の位置埋め込み(条件 1)と 2 次元正弦波の位置埋め込み(条件 2、学習しない)の、パッチの位置どうしの余弦類似度。
  各小図は格子の 1 つの位置について、その位置の埋め込みと全位置の埋め込みの余弦類似度を $8 \times 8$ の格子に並べたもの
  (原論文の図 7 中央と同じ描き方)。学習可能な埋め込みに、同じ行・同じ列で類似度が高い 2 次元の構造が現れるかを見る。
- 層ごと・ヘッドごとの平均の注意距離(画素、検証集合の 1,000 枚)。原論文の図 7 右と同じ量で、下位層に局所的な(距離の小さい)
  ヘッドが現れるか、位置埋め込みの方式で違うかを見る。位置埋め込みなし(条件 0)では、パッチの配置を区別できないので、
  注意の重みは距離と無関係になるはずであり、平均の注意距離は全パッチの組の平均距離に近くなると予想される。


```python
_grid = IMAGE_SIZE // STANDARD_PATCH_SIZE
for _pe in ("learned", "sinusoidal_2d"):
    _sim = RECORDS[key_vit(STANDARD_PATCH_SIZE, _pe, 1.0, 0)]["position_similarity"]
    _fig, _axes = plt.subplots(_grid, _grid, figsize=(7, 7))
    for _r in range(_grid):
        for _c in range(_grid):
            _axes[_r, _c].imshow(_sim[_r * _grid + _c].reshape(_grid, _grid), vmin=-1, vmax=1, cmap="viridis")
            _axes[_r, _c].set_xticks([])
            _axes[_r, _c].set_yticks([])
    _fig.suptitle(f"{_plot_tag}position embedding cosine similarity ({_pe}, seed 0)")
    plt.tight_layout()
    plt.show()

_rows, _cols = np.divmod(np.arange(_grid * _grid), _grid)
_all_pairs = STANDARD_PATCH_SIZE * np.sqrt((_rows[:, None] - _rows[None, :]) ** 2 + (_cols[:, None] - _cols[None, :]) ** 2)
_fig, _ax = plt.subplots(figsize=(7, 4))
for _c, (_pe, _marker) in enumerate(zip(POSITION_EMBEDDINGS, ("x", "o", "^"), strict=True)):
    _dist = RECORDS[key_vit(STANDARD_PATCH_SIZE, _pe, 1.0, 0)]["attention_distance"]  # (L, h)
    for _layer in range(NUM_LAYERS):
        _ax.scatter(
            [_layer + 1 + (_c - 1) * 0.15] * NUM_HEADS, _dist[_layer], marker=_marker, color=f"C{_c}",
            label=f"{_c}: {_pe}" if _layer == 0 else None,
        )
    print(f"{_tag}平均の注意距離(画素、行: 層、列: ヘッド)条件 {_c}({_pe}): {np.round(_dist.astype(np.float64), 2).tolist()}")
_ax.axhline(_all_pairs.mean(), color="gray", ls="--", label="mean distance over all patch pairs")
_ax.set_xlabel("layer")
_ax.set_ylabel("mean attention distance (pixels)")
_ax.set_title(f"{_plot_tag}mean attention distance (seed 0, {ATTENTION_DISTANCE_IMAGES} validation images)")
_ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
print(f"全パッチの組の平均距離: {_all_pairs.mean():.2f} 画素")
```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/019_vision_transformer/output_35_0.png)
    



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/019_vision_transformer/output_35_1.png)
    


    平均の注意距離(画素、行: 層、列: ヘッド)条件 0(none): [[15.08, 15.1, 14.57], [15.75, 16.03, 16.34], [15.94, 16.02, 16.13], [17.26, 15.89, 16.21], [16.5, 16.2, 16.02], [16.19, 16.65, 16.13]]
    平均の注意距離(画素、行: 層、列: ヘッド)条件 1(learned): [[14.41, 14.39, 16.21], [15.23, 15.89, 15.77], [16.28, 16.65, 16.67], [15.96, 15.86, 16.24], [16.56, 16.04, 16.47], [16.46, 16.45, 16.52]]
    平均の注意距離(画素、行: 層、列: ヘッド)条件 2(sinusoidal_2d): [[14.54, 12.83, 14.98], [13.48, 14.33, 13.8], [15.62, 17.71, 15.12], [17.35, 16.55, 17.41], [16.86, 16.64, 16.87], [16.32, 16.64, 16.11]]



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/019_vision_transformer/output_35_3.png)
    


    全パッチの組の平均距離: 16.55 画素


### 6.9 実験 B: パッチサイズと系列長


```python
B_LOG2_N = np.array([math.log2((IMAGE_SIZE // p) ** 2) for p in B_PATCH_SIZES])
B_ACCURACY = np.array([[test_accuracy(key_vit(p, "learned", 1.0, s)) for p in B_PATCH_SIZES] for s in SEEDS_B])  # (n_B, 水準)
B_SLOPES = ols_slope(B_LOG2_N, B_ACCURACY)
CONTRAST_B = {"value": float(B_SLOPES.mean()), "sigma": float(B_SLOPES.std(ddof=1) / math.sqrt(len(SEEDS_B)))}
verdict_B = judge(CONTRAST_B["value"], CONTRAST_B["sigma"])
precondition_status["P1(B)"] = bool((B_ACCURACY.mean(axis=0) >= ACCURACY_MIN).all())

print(f"{_tag}水準 P = {B_PATCH_SIZES}(log2 N = {B_LOG2_N.tolist()})")
for _j, _p in enumerate(B_PATCH_SIZES):
    _ce = np.mean([RECORDS[key_vit(_p, "learned", 1.0, s)]["test"]["cross_entropy"] for s in SEEDS_B])
    print(
        f"{_tag}  P = {_p}(N = {(IMAGE_SIZE // _p) ** 2}): テスト正解率 {np.round(B_ACCURACY[:, _j], 4).tolist()} "
        f"平均 {B_ACCURACY[:, _j].mean():.4f}、交差エントロピー平均 {_ce:.4f}"
    )
print(f"{_tag}シードごとの傾き beta_s: {np.round(B_SLOPES, 5).tolist()}")
print(
    f"{_tag}対比量 beta_bar = {CONTRAST_B['value']:+.5f}(正解率 / log2 N)、sigma_B = sd(beta_s)/sqrt({len(SEEDS_B)}) = "
    f"{CONTRAST_B['sigma']:.5f}、閾値 {2 * CONTRAST_B['sigma']:.5f} -> 判定関数の結果: {verdict_B}"
)
print(f"{_tag}前提条件: P0(ViT) = {precondition_status['P0(ViT)']}、P1(B) = {precondition_status['P1(B)']}")
print("診断量(判定なし): 計算量と 1 ステップの時間")
for _p in PATCH_SIZES:
    _used = "" if _p in B_PATCH_SIZES else "(この段階では学習していない)"
    print(
        f"  P = {_p}: 順伝播 {VIT_FORWARD_FLOPS[_p] / 1e6:,.1f} MFLOP / 枚(P = 4 の {VIT_FORWARD_FLOPS[_p] / VIT_FORWARD_FLOPS[4]:.2f} 倍)、"
        f"{device} での 1 ステップ {STEP_SECONDS[f'vit_p{_p}'] * 1000:.1f} ms"
        f"(P = 4 の {STEP_SECONDS[f'vit_p{_p}'] / STEP_SECONDS['vit_p4']:.2f} 倍){_used}"
    )

_fig, _ax = plt.subplots(figsize=(6, 4))
for _s_i in range(len(SEEDS_B)):
    _ax.plot(B_LOG2_N, B_ACCURACY[_s_i], marker="o", color="gray", alpha=0.6)
_ax.plot(B_LOG2_N, B_ACCURACY.mean(axis=0), marker="s", color="black", lw=2, label="seed mean")
_ax.set_xticks(B_LOG2_N, [f"N={int(2**x)}\n(P={p})" for x, p in zip(B_LOG2_N, B_PATCH_SIZES, strict=True)])
_ax.set_xlabel("log2 N")
_ax.set_ylabel("test accuracy")
_ax.set_title(f"{_plot_tag}Experiment B: patch size (lines = seeds)")
_ax.legend()
plt.tight_layout()
plt.show()
```

    水準 P = (8, 4)(log2 N = [4.0, 6.0])
      P = 8(N = 16): テスト正解率 [0.4816, 0.4784, 0.4819, 0.4712, 0.4903] 平均 0.4807、交差エントロピー平均 1.4314
      P = 4(N = 64): テスト正解率 [0.5832, 0.5889, 0.5693, 0.5935, 0.5841] 平均 0.5838、交差エントロピー平均 1.1607
    シードごとの傾き beta_s: [0.0508, 0.05525, 0.0437, 0.06115, 0.0469]
    対比量 beta_bar = +0.05156(正解率 / log2 N)、sigma_B = sd(beta_s)/sqrt(5) = 0.00308、閾値 0.00616 -> 判定関数の結果: 支持
    前提条件: P0(ViT) = True、P1(B) = True
    診断量(判定なし): 計算量と 1 ステップの時間
      P = 8: 順伝播 92.8 MFLOP / 枚(P = 4 の 0.25 倍)、cuda での 1 ステップ 36.6 ms(P = 4 の 0.60 倍)
      P = 4: 順伝播 365.7 MFLOP / 枚(P = 4 の 1.00 倍)、cuda での 1 ステップ 60.7 ms(P = 4 の 1.00 倍)
      P = 2: 順伝播 1,669.8 MFLOP / 枚(P = 4 の 4.57 倍)、cuda での 1 ステップ 253.0 ms(P = 4 の 4.17 倍)(この段階では学習していない)



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/019_vision_transformer/output_37_1.png)
    




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/05_vision_language/019_vision_transformer.ipynb
