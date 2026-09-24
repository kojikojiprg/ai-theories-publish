---
title: "Flash Attention(実装・実験編 2/4)"
---

この記事は後編(実装・実験編 2/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/014_flash_attention-practice-1)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/014_flash_attention-practice-3)。

### 6.1 実験宣言セル: 共通の設定・検証すること・判定基準・前提条件

**この節の内容は本番実行の前に確定させ、結果を見た後に変更しない。**

#### 共通の設定

- 入力は乱数で作る(5.3 節)。Query・Key の要素の標準偏差を $\sigma$ とし、「通常」は $\sigma = 1$(スコアの標準偏差 1)、
  「大」は $\sigma = 6$(スコアの標準偏差 36)とする。「大」では 1 行のスコアの最大値が FP32 の指数関数の
  オーバーフローの境界(約 88.7)を超える行が現れ、最大値を引かない素朴な softmax は非有限値を出す(実験 A の診断量で観察する)。
- $d = 64$。バッチ・ヘッド数・ブロックの行数は実験ごとに 5.2 節の値に固定し、水準によらず共通とする。
- 系列長の水準は公比 2 の等比数列とする(5.2 節)。メモリ量(実験 C・D)と速度比(実験 E)は $\log N$ に対する傾きを
  見るので、等比の刻みで各水準が回帰に等しく寄与するようにする。
- 判定は **支持 / 反証 / 判定不能** の 3 値とする。前提条件が 1 つでも成立しなかった実験は、判定関数の結果に関わらず
  **前提不成立** とし、判定不能とは区別して報告する。
- **標本の独立性**: 実験 A・B の各標本は、系列長・因果マスクの有無・スケール・シードの組ごとに異なる乱数のシード
  (`sample_seed()`)で入力を作る。同じ入力を複数の条件で共有しないので、標本は互いに独立である。
- **実験 C・D の計測は決定的である** (同じ入力・同じコードでは毎回同じバイト数になる)。したがって、べき指数の標準誤差は
  測定のばらつきではなく、べき乗則からのずれ(両対数での曲がり)を表す。

#### 実験 A: 順伝播の数値的な一致

- **検証すること**: タイリングと online softmax による順伝播`flash_attention_forward()`の出力の、FP64 の参照値に対する誤差が、
  標準の Attention(001 の`scaled_dot_product_attention()`)の FP32 の実装の誤差と同程度である(Flash Attention は近似ではなく、
  和を取る順序と正規化の時点を変えただけの厳密な計算であるため)。
- **条件**: 系列長 $N \in \{80, 160, 320, 640, 1280\}$(本番。$B_r = B_c = 64$ で割り切れない 80・160 と、割り切れる 320 以上を
  含む)× 因果マスクの有無 × スケール(通常・大)× シード $R = 5$ 個。標本数 $n_A = 5 \times 2 \times 2 \times 5 = 100$。
  入力はすべて FP32。
- **誤差**: 相対誤差 $\mathrm{err} = \lVert O - O_{\mathrm{ref}} \rVert_\infty / \lVert O_{\mathrm{ref}} \rVert_\infty$。
  $\lVert \cdot \rVert_\infty$ は要素の絶対値の最大値、$O_{\mathrm{ref}}$ は同じ FP32 の入力を FP64 に変換して 001 の関数で
  計算した参照値である。タイリングの誤差を $\mathrm{err}_{\mathrm{tiled}}$、標準の実装の誤差を $\mathrm{err}_{\mathrm{standard}}$ とする。
- **対比量**: 標本 $s$ ごとの $r_s = \log(\mathrm{err}_{\mathrm{tiled},s} / \mathrm{err}_{\mathrm{standard},s})$ の平均
  $\bar{r} = \frac{1}{n_A} \sum_s r_s$ と、その標準誤差 $\mathrm{SE} = \hat{s}_r / \sqrt{n_A}$($\hat{s}_r$ は $r_s$ の標本標準偏差、
  不偏分散による)。標本は独立なので、平均の標準誤差はこの式で与えられる。
- **判定基準**:
  - 支持: $\bar{r} + 2\,\mathrm{SE} < \log 2$(タイリングの誤差が、標準の実装の誤差の 2 倍以上には悪くない)
  - 反証: $\bar{r} - 2\,\mathrm{SE} > \log 2$
  - 判定不能: 上記以外
  - タイリングの誤差が非有限値(NaN・無限大)になった標本が 1 つでもあれば、その標本の $r_s = +\infty$ とみなし、反証とする。
- **閾値 $\log 2$ の意図**: 厳密な計算どうしの誤差は、どちらも FP32 の丸め誤差($2^{-24} \approx 6 \times 10^{-8}$ の定数倍)の
  大きさであり、和を取る順序や指数関数を呼ぶ回数の違いによって定数倍の範囲で変わりうる。2 倍はその程度の違いを許す幅である。
  一方、近似(項の打ち切りなど)や再スケールの誤りがあれば、誤差は丸め誤差より何桁も大きくなるので、$\log 2$ を大きく超える。
  この基準は「タイリングの方が精度が良い」ことは主張しない(片側の基準である)。
- **前提条件 P-A**: 全標本で $\mathrm{err}_{\mathrm{standard}}$ が有限かつ正であること(対比量の分母が定義されること。
  参照側の実装のみに関する条件で、検証する仮説とは独立である)。
- **対比量と直接の作用点**: タイリングと online softmax が直接作用するのは出力 $O$ の計算の手順であり、$O$ の誤差はその直接の
  帰結である。対比量は直接の作用点にあたる。
- **診断量**: FP16 の入力での同じ量($\bar{r}$ を FP16 について計算したもの。参照値は FP16 に丸めた入力の FP64 での計算)。
  **タイリングの実装は統計量と出力の累積を FP32 で行い、標準の実装(001)は FP16 のまま計算する。** 両者は累積の精度が
  異なるため、この診断量の差をタイリングそのものの効果とは解釈しない。
  最大値を引かない素朴な softmax(FP32)で非有限値を含む行の割合(スケールごと)。因果マスクで飛ばしたブロックの割合(系列長ごと)。
  条件(因果マスクの有無 × スケール)ごとの $\bar{r}$。

#### 実験 B: 再計算による逆伝播の数値的な一致

- **検証すること**: 再計算による逆伝播(`FlashAttentionFunction`)の勾配 $dQ, dK, dV$ の誤差が、標準の Attention を autograd で
  微分した FP32 の勾配の誤差と同程度である。
- **条件**: 実験 A と同じ(系列長・因果マスク・スケール・シードの数、標本数 $n_B = 100$)。出力の勾配 $dO$ は標準正規分布の乱数。
  参照値は、FP32 の入力を FP64 に変換して 001 の関数を autograd で微分した勾配。
- **対比量**: 3 つの勾配 $g \in \{dQ, dK, dV\}$ それぞれの相対誤差の比の対数
  $r_{s,g} = \log(\mathrm{err}_{\mathrm{tiled},s,g} / \mathrm{err}_{\mathrm{standard},s,g})$ のうち最大のものを、標本ごとの値
  $r_s = \max_g r_{s,g}$ とする。その平均と標準誤差は実験 A と同じ式。
- **判定基準・閾値の意図**: 実験 A と同じ(支持: $\bar{r} + 2\,\mathrm{SE} < \log 2$、反証: $\bar{r} - 2\,\mathrm{SE} > \log 2$、
  それ以外は判定不能。非有限値の扱いも同じ)。
- **前提条件 P-B**: 全標本・全勾配で $\mathrm{err}_{\mathrm{standard}}$ が有限かつ正であること。
- **対比量と直接の作用点**: 再計算が直接作用するのは逆伝播で使う $P_{ij}$ の求め方であり、勾配はその直接の帰結である。
  3 つの勾配の最大値は極値統計(3 つの中の最大)なので、同じ現象を平均で捉える診断量として、勾配ごとの $\bar{r}_g$ を併記する。
- `torch.autograd.gradcheck`(FP64・小さな形状)による確認は判定ではなく、5.5 節のアサーションで行った。

#### 実験 C: 順伝播のピークメモリの系列長に対する次数

- **検証すること**: 順伝播(`torch.no_grad()`)のピークメモリの増分の系列長 $N$ に対するべき指数が、標準の Attention では約 2、
  タイリングでは約 1 以下である。
- **ピークメモリの増分**: `torch.cuda.max_memory_allocated()`から、呼び出しの直前の`torch.cuda.memory_allocated()`
  (入力 $Q, K, V$ を含む)と、出力 $O$ のバイト数を差し引いたもの。ここで「出力」は Attention の出力 $O$ のみを指す。
  標準の実装が返す $P$ と、タイリングが返す $L$ は、中間の量として増分に含める。
- **水準**: $N \in \{1024, 2048, 4096, 8192, 16384\}$(本番)、バッチ 1・ヘッド 4・$d = 64$・FP32、$B_r = B_c = 128$(全水準で固定)。
  最大の系列長は、5.2 節の閉形式の見積もり(標準の実装で 8.0 GiB。$N = 32768$ では 32 GiB)から、T4 の約 15 GiB の 60% 以下に
  収まる最大の水準として決めた。
- **理論上の見込み**: 標準の実装の増分は $S$ と $P$ の $2 \cdot BHN^2 \cdot 4$ バイトが支配的なので、べき指数は 2 に近い。
  タイリングの増分は $L$ の $BHN \cdot 4$ バイト(系列長に比例)と、ブロックの一時的なテンソル($B_r B_c$ と $B_r d$ に比例し、
  系列長によらない)の和なので、べき指数は 0 と 1 の間になる(系列長によらない項が大きいほど 0 に近い)。
  したがって差は 1 以上になると見込む。判定は差に対して行う。
- **対比量**: 各実装について $\log(\text{increment}) = \log a + b \log N$ を最小二乗であてはめたべき指数 $b$ の差
  $\Delta b = b_{\mathrm{standard}} - b_{\mathrm{tiled}}$ と、その標準誤差
  $\mathrm{SE}_{\Delta b} = \sqrt{\mathrm{SE}_{\mathrm{standard}}^2 + \mathrm{SE}_{\mathrm{tiled}}^2}$。$\mathrm{SE}_{\mathrm{standard}}$・
  $\mathrm{SE}_{\mathrm{tiled}}$ は各回帰の傾きの標準誤差(`fit_power_law_exponent()`、残差の分散と $\log N$ の偏差平方和から
  求める)で、2 つの回帰は別々の計測から作られるので、差の分散は和になる(誤差伝播)。
- **判定基準**:
  - 支持: $\Delta b - 2\,\mathrm{SE}_{\Delta b} > 0.5$
  - 反証: $\Delta b + 2\,\mathrm{SE}_{\Delta b} < 0.5$
  - 判定不能: 上記以外
  - 閾値 0.5 は、帰無仮説(タイリングがメモリの次数を変えない、差 0)と理論値(差 1)の中間として置いた値である。
- **前提条件**:
  - P-C1: デバイスが CUDA であること。ローカルのスモークテスト(CPU)では、`LiveTensorBytesTracker`による生存中のテンソルの
    バイト数で経路の動作を確認するのみで、判定しない。
  - P-C2: 全水準で両方の実装が OOM(メモリ不足)にならずに完走すること。
- **対比量と直接の作用点**: タイリングが直接作用するのは、$N \times N$ の中間行列を実体化するかどうかである。ピークメモリは
  その直接の帰結であり、対比量は作用点に近い。
- **診断量**: 閉形式の見積もりと実測の比。`LiveTensorBytesTracker`による生存中のテンソルのバイト数のピーク(CUDA のアロケータの
  値との比較)。タイリングの HBM の読み書き量の理論値(ブロックの行数の関数)と標準の実装の読み書き量の比。

#### 実験 D: 逆伝播のために保存する活性化の系列長に対する次数

- **検証すること**: 逆伝播のために保存するテンソルのバイト数の系列長に対するべき指数が、標準の Attention(autograd)では約 2、
  `FlashAttentionFunction`では約 1 である。
- **計測**: `torch.autograd.graph.saved_tensors_hooks`で保存されるテンソルを記録し、記憶領域ごとに 1 回だけバイト数を数える
  (5.6 節)。**デバイスに依存しない計測なので、ローカルのスモークテストでも本番と同じ計測になる。**
- **水準**: $N \in \{512, 1024, 2048, 4096, 8192\}$(本番)、バッチ 1・ヘッド 4・$d = 64$・FP32、$B_r = B_c = 128$、因果マスクなし。
- **理論上の見込み**: 標準の実装は $Q, K, V$ と $P$ を保存するので $4 BH(3Nd + N^2)$ バイト、`FlashAttentionFunction`は
  $Q, K, V, O, L$ を保存するので $4 BH(4Nd + N)$ バイトである。後者は $N$ に厳密に比例する(べき指数 1)。前者は $3Nd$ の項のため、
  この範囲の $N$ ではべき指数が 2 をやや下回る。
- **この実験の位置づけ**: 保存するバイト数の計測は決定的であり、実測は閉形式と一致すると見込まれる。本番の水準では、
  閉形式から $\Delta b \approx 0.9$ が得られる(6.6 節で閉形式から計算して印字する)。したがってこの実験は、
  実装が閉形式どおりのテンソルのみを保存していることの確認にあたる。
- **対比量・判定基準**: 実験 C と同じ形(保存するバイト数のべき指数の差 $\Delta b = b_{\mathrm{standard}} - b_{\mathrm{flash}}$ と、
  誤差伝播による標準誤差。支持: $\Delta b - 2\,\mathrm{SE}_{\Delta b} > 0.5$、反証: $\Delta b + 2\,\mathrm{SE}_{\Delta b} < 0.5$、
  それ以外は判定不能)。
- **前提条件**: 設けない(決定的でデバイスに依存しない計測であり、成立を確かめるべき状態がない)。
- **対比量と直接の作用点**: 再計算が直接作用するのは、逆伝播のために $P$ を保存するかどうかであり、保存するバイト数はその
  直接の量である。
- **診断量**: 閉形式の保存バイト数と実測の比。逆伝播の時間の比(再計算による演算量の増加と Python のループのオーバーヘッドを
  含む。CUDA でのみ計測する)。

#### 実験 E: 読み書き量の削減による速度の系列長依存性(SDPA のバックエンドの比較)

- **検証すること**: SDPA の memory-efficient バックエンドの math バックエンドに対する速度比
  $t_{\mathrm{math}} / t_{\mathrm{mem\ efficient}}$ が、系列長とともに大きくなる。
- **注意**: 本ノートブックのスクラッチ実装は Python のループで書いており、ブロックの計算ごとに複数のカーネルを呼び出すので、
  速度は Flash Attention の本来の性能を反映しない。**速度に関する主張は、この実験の参照実装(PyTorch の融合カーネル)の比較でのみ
  検証する。** スクラッチ実装の時間は診断量として添えるにとどめる。
- **計測**: FP16 の入力、順伝播のみ(`torch.no_grad()`)、バッチ 1・ヘッド 8・$d = 64$、因果マスクなし。
  $N \in \{256, 512, 1024, 2048, 4096, 8192\}$(本番)。各水準でウォームアップ 3 回の後、math と memory-efficient を
  1 回ずつ続けて実行する反復を 20 回行い、それぞれ`torch.cuda.synchronize()`で挟んで時間を測る。直前のカーネルの影響が
  一方の条件にだけ系統的に乗らないよう、**反復ごとに実行の順序を入れ替える** (偶数回目は math が先、奇数回目は
  memory-efficient が先)。ウォームアップの順序は問わない。
- **対比量(改訂後)**: 水準 $N$ の反復 $k$ ごとの比 $\rho_{N,k} = t_{\mathrm{math},N,k} / t_{\mathrm{mem\ efficient},N,k}$ から、
  水準ごとの代表値 $m_N = \mathrm{median}_k \log \rho_{N,k}$ を求め、$m_N = c + \beta \log N$ を 6 水準(本番)で最小二乗で
  あてはめた傾き $\beta$ と、その標準誤差
  $\mathrm{SE} = \sqrt{\hat{\sigma}^2 / \sum_N (\log N - \overline{\log N})^2}$ を対比量とする。$\hat{\sigma}^2$ は残差平方和を
  自由度(水準数 $- 2$、本番では 4)で割った残差の分散、$\overline{\log N}$ は水準の $\log N$ の平均である。反復(本番 20 回)は、
  水準ごとの代表値 $m_N$ の精度を上げるためのものと位置づける。同じ反復の 2 つの時間は続けて測るので、比はその間の GPU の
  状態の変動の一部を打ち消す。
- **判定基準**:
  - 支持: $\beta - 2\,\mathrm{SE} > 0$
  - 反証: $\beta + 2\,\mathrm{SE} < 0$
  - 判定不能: 上記以外
- **判定基準の改訂の記録(スモークテストの後・本番実行の前に改訂した)**:
  - **旧基準**: 全水準・全反復の $6 \times 20 = 120$ 点の $\log \rho_{N,k}$ をまとめて $\log \rho_{N,k} = c + \beta \log N$ で
    最小二乗であてはめ、傾き $\beta$ とその標準誤差(残差の分散(自由度 118)と $\log N$ の偏差平方和から求める)で判定する。
    判定基準の形(支持: $\beta - 2\,\mathrm{SE} > 0$、反証: $\beta + 2\,\mathrm{SE} < 0$、それ以外は判定不能)は新基準と同じ。
  - **新基準**: 上の「対比量(改訂後)」のとおり、水準ごとの $\log \rho_{N,k}$ の中央値 $m_N$ の 6 点の回帰の傾きと、自由度 4 の
    残差の分散から求めた標準誤差で判定する。判定基準の形と閾値 0 は変えない。
  - **改訂の理由**: 同じ水準の反復は、同じ入力・同じカーネルの選択・ほぼ同じ GPU の状態で測った値であり、回帰のモデル
    ($\log N$ に対して線形)に対して互いに独立ではない。速度比の $\log N$ に対する関係が直線からずれる場合(短い系列長で
    カーネルの起動のオーバーヘッドが支配的になる、長い系列長で頭打ちになるなど)、そのずれは水準単位の誤差であるが、
    120 点の回帰では反復の数だけ標準誤差が過小になる(擬似反復、pseudoreplication)。独立な単位は水準であるため、水準ごとの
    代表値で回帰する。**この改訂は標本の独立性という一般論に基づくもので、スモークテストの結果の方向には依存しない**
    (スモークテストは CPU で実行され、前提条件 P-E1 が不成立であり、判定に使っていない)。
  - **スモークテストの旧基準による結果(記録)**: コミット`8df5b5e`のセル出力で、$\beta = -0.0643$、$\mathrm{SE} = 0.0338$、
    判定関数の結果は判定不能、最終判定は前提不成立(ローカルの CPU、memory-efficient の代わりに CPU の flash バックエンドを
    使った経路の確認、系列長 128・256・512、反復 5 回)。
  - 旧基準による $\beta$ と SE は、本番でも診断量として計算・印字する(判定には使わない)。
- **前提条件**:
  - P-E1: デバイスが CUDA であること(ローカルのスモークテストでは、CPU で使える flash バックエンドを memory-efficient の代わりに
    使って経路の動作を確認するのみで、判定しない)。
  - P-E2: 全水準で、両方のバックエンドが実際に選ばれて実行されたこと(profiler で記録した演算の名前が、math では
    `aten::_scaled_dot_product_attention_math`、memory-efficient では`aten::_scaled_dot_product_efficient_attention`の
    1 つだけであること。別のバックエンドへのフォールバックがないこと)。
- **対比量と直接の作用点**: 読み書き量の削減が直接作用するのは HBM の読み書き量だが、T4 ではこれを torch から直接は計測できない。
  時間はその下流の量であり、カーネルの起動のオーバーヘッドや演算器の使用率など、読み書き量以外の要因も含む。系列長が短い水準では
  両方のバックエンドとも起動のオーバーヘッドが支配的になりうる。
- **診断量**: 旧基準(120 点の回帰)による $\beta$ と SE。水準ごとの時間の中央値と比の中央値。スクラッチ実装の時間(本来の性能を反映しないことを添えて)。各バックエンドの
  ピークメモリの増分(CUDA でのみ計測)。

### 6.2 スケーリングの計測と外挿

本番のデータ量・回数がスモークテストと異なる重い処理について、3 点の系列長で実行時間を実測し、
$\log t = \log a + b \log n$ をあてはめてべき指数 $b$ を推定し、本番の各水準へ外挿する。外挿値には本番での実行回数を乗じる。

| 処理 | 計測する系列長 | 本番の実行回数 |
|---|---|---|
| 実験 A の 1 標本(FP32・FP16・素朴な softmax、因果マスクなし) | 本番の上位 3 水準(320・640・1280) | 各水準で 因果マスク 2 × スケール 2 × シード 5 |
| 実験 B の 1 標本(FP64 の参照・標準・再計算の逆伝播) | 同上 | 同上 |
| 実験 C の 1 水準(両方の実装) | スモークテストの水準(256・512・1024) | 各水準 1 回 |
| 実験 D の 1 水準(両方の実装) | スモークテストの水準(512・1024・2048) | 各水準 1 回 |
| 実験 E の 1 水準(本番の反復回数) | スモークテストの水準(128・256・512) | 各水準 1 回 |

実験 A・B は因果マスクなし(計算するブロックが最も多い)で計測し、見積もりの上限とする。**CUDA 以外で計測した場合、外挿値は
その環境の値であり、T4 での時間とは異なる。** スクラッチ実装は 1 ブロックごとに複数のカーネルを起動するので、T4 では
カーネルの起動のオーバーヘッドが支配的になりうる。実験 D の逆伝播の時間(診断量)は CUDA でのみ測るので、CUDA 以外の見積もりには
含まれない。


```python
def fit_and_extrapolate(label: str, sizes, times, targets) -> dict[int, float]:
    fit = fit_power_law_exponent(sizes, times)
    detail = ", ".join(f"n={n}: {t:.3f}s" for n, t in zip(sizes, times, strict=True))
    extrapolated = {n: fit.coefficient * n**fit.exponent for n in targets}
    print(
        f"[{label}] {detail} -> b={fit.exponent:.3f}(標準誤差 {fit.exponent_stderr:.3f}), "
        f"R^2={fit.r_squared:.4f}, 外挿: "
        + ", ".join(f"n={n}: {t:.2f}s" for n, t in extrapolated.items())
    )
    return extrapolated


_t0_scaling = time.time()
_prod = LEVELS["prod"]
_repeats_ab = len(CAUSAL_AB) * len(SIGMAS_AB) * _prod["NUM_SEEDS_AB"]
_sizes_ab = list(_prod["SEQUENCE_LENGTHS_AB"][-3:])
run_sample_a(_sizes_ab[0], False, 1.0, 0)  # 初回呼び出しのオーバーヘッドを除く
_ext_a = fit_and_extrapolate(
    "実験 A の 1 標本",
    _sizes_ab,
    [timed_call(lambda n=n: run_sample_a(n, False, 1.0, 0)) for n in _sizes_ab],
    _prod["SEQUENCE_LENGTHS_AB"],
)
_ext_b = fit_and_extrapolate(
    "実験 B の 1 標本",
    _sizes_ab,
    [timed_call(lambda n=n: run_sample_b(n, False, 1.0, 0)) for n in _sizes_ab],
    _prod["SEQUENCE_LENGTHS_AB"],
)
_sizes_c = list(LEVELS["smoke"]["SEQUENCE_LENGTHS_C"])
_ext_c = fit_and_extrapolate(
    "実験 C の 1 水準",
    _sizes_c,
    [
        timed_call(lambda n=n: [measure_forward_memory(i, n) for i in ("standard", "tiled")])
        for n in _sizes_c
    ],
    _prod["SEQUENCE_LENGTHS_C"],
)
_sizes_d = list(LEVELS["smoke"]["SEQUENCE_LENGTHS_D"])
_ext_d = fit_and_extrapolate(
    "実験 D の 1 水準",
    _sizes_d,
    [
        timed_call(lambda n=n: [measure_saved_tensor_bytes(i, n) for i in ("standard", "tiled")])
        for n in _sizes_d
    ],
    _prod["SEQUENCE_LENGTHS_D"],
)
_sizes_e = list(LEVELS["smoke"]["SEQUENCE_LENGTHS_E"])
_ext_e = fit_and_extrapolate(
    f"実験 E の 1 水準(反復 {_prod['REPETITIONS_E']} 回)",
    _sizes_e,
    [timed_call(lambda n=n: run_level_e(n, _prod["REPETITIONS_E"])) for n in _sizes_e],
    _prod["SEQUENCE_LENGTHS_E"],
)
_t_scaling = time.time() - _t0_scaling

ESTIMATE_SECONDS = {
    f"実験 A(各水準 {_repeats_ab} 標本)": sum(_ext_a.values()) * _repeats_ab,
    f"実験 B(各水準 {_repeats_ab} 標本)": sum(_ext_b.values()) * _repeats_ab,
    "実験 C(5 水準 x 1 回)": sum(_ext_c.values()),
    "実験 D(5 水準 x 1 回)": sum(_ext_d.values()),
    "実験 E(6 水準 x 1 回)": sum(_ext_e.values()),
    "スケーリング計測自体(このセルの実測)": _t_scaling,
}
_total_estimate = sum(ESTIMATE_SECONDS.values())
print(f"\n--- 本番実行の見積もり(実行回数を乗じた値、{device} 基準)---")
for _k, _v in ESTIMATE_SECONDS.items():
    print(f"  {_k}: {_v:,.1f}s")
print(
    f"  合計: {_total_estimate:,.1f}s = {_total_estimate / 60:.1f} 分"
    f"(予算 {SESSION_BUDGET_SECONDS / 60:.0f} 分の {_total_estimate / SESSION_BUDGET_SECONDS:.1%})"
)
if _total_estimate > SESSION_BUDGET_SECONDS:
    print("警告: 見積もりがセッションの予算を超える")
if device.type != "cuda":
    print(
        "注意: CUDA 以外での見積もりであり、T4 での時間とは異なる(実験 D の逆伝播の時間の計測を含まない)"
    )
```

    [実験 A の 1 標本] n=320: 0.103s, n=640: 0.468s, n=1280: 1.396s -> b=1.879(標準誤差 0.174), R^2=0.9915, 外挿: n=80: 0.01s, n=160: 0.03s, n=320: 0.11s, n=640: 0.41s, n=1280: 1.50s
    [実験 B の 1 標本] n=320: 0.089s, n=640: 0.356s, n=1280: 1.195s -> b=1.877(標準誤差 0.076), R^2=0.9984, 外挿: n=80: 0.01s, n=160: 0.02s, n=320: 0.09s, n=640: 0.34s, n=1280: 1.23s
    [実験 C の 1 水準] n=256: 0.061s, n=512: 0.102s, n=1024: 0.355s -> b=1.277(標準誤差 0.305), R^2=0.9461, 外挿: n=1024: 0.31s, n=2048: 0.76s, n=4096: 1.85s, n=8192: 4.47s, n=16384: 10.84s
    [実験 D の 1 水準] n=512: 0.158s, n=1024: 0.512s, n=2048: 1.460s -> b=1.604(標準誤差 0.054), R^2=0.9989, 外挿: n=512: 0.16s, n=1024: 0.49s, n=2048: 1.49s, n=4096: 4.53s, n=8192: 13.79s
    [実験 E の 1 水準(反復 20 回)] n=128: 0.085s, n=256: 0.094s, n=512: 0.150s -> b=0.406(標準誤差 0.148), R^2=0.8825, 外挿: n=256: 0.11s, n=512: 0.14s, n=1024: 0.19s, n=2048: 0.25s, n=4096: 0.33s, n=8192: 0.43s
    
    --- 本番実行の見積もり(実行回数を乗じた値、cuda 基準)---
      実験 A(各水準 20 標本): 41.0s
      実験 B(各水準 20 標本): 33.8s
      実験 C(5 水準 x 1 回): 18.2s
      実験 D(5 水準 x 1 回): 20.5s
      実験 E(6 水準 x 1 回): 1.4s
      スケーリング計測自体(このセルの実測): 6.8s
      合計: 121.8s = 2.0 分(予算 120 分の 1.7%)


### 6.3 実験 A: 順伝播の数値的な一致


```python
def sample_seed(
    experiment_offset: int, n_index: int, causal: bool, sigma_name: str, replicate: int
) -> int:
    # 系列長・因果マスク・スケール・シードの組ごとに異なるシード(条件間で入力を共有しない)
    condition = (n_index * len(CAUSAL_AB) + CAUSAL_AB.index(causal)) * len(SIGMAS_AB) + list(
        SIGMAS_AB
    ).index(sigma_name)
    return experiment_offset + 1000 * condition + replicate


def log_ratios(err_tiled: np.ndarray, err_standard: np.ndarray) -> np.ndarray:
    # タイリングの誤差が非有限値の標本は +inf とする(6.1 節)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.log(err_tiled / err_standard)
    return np.where(np.isfinite(err_tiled), values, np.inf)


def mean_and_standard_error(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(len(values)))


def ratio_verdict(values: np.ndarray) -> tuple[str, float, float]:
    if not np.all(np.isfinite(values)):
        return "反証", float("inf"), float("nan")
    mean, se = mean_and_standard_error(values)
    if mean + 2 * se < LOG2:
        return "支持", mean, se
    if mean - 2 * se > LOG2:
        return "反証", mean, se
    return "判定不能", mean, se


def verdict_label(computed: str, preconditions: list[str]) -> str:
    return computed if all(precondition_status.get(p) for p in preconditions) else "前提不成立"


records_a = []
_t0 = time.time()
for _i_n, _n in enumerate(SEQUENCE_LENGTHS_AB):
    for _causal in CAUSAL_AB:
        for _sigma_name, _sigma in SIGMAS_AB.items():
            for _r in SEEDS_AB:
                _seed = sample_seed(0, _i_n, _causal, _sigma_name, _r)
                _rec = run_sample_a(_n, _causal, _sigma, _seed)
                _rec.update(n=_n, causal=_causal, sigma=_sigma_name, seed=_seed)
                records_a.append(_rec)
print(f"実験 A: {len(records_a)} 標本、{time.time() - _t0:.1f}s")
assert len({r["seed"] for r in records_a}) == len(records_a), "条件間でシードが重複している"

_err_t_a = np.array([r["err_tiled"] for r in records_a])
_err_s_a = np.array([r["err_standard"] for r in records_a])
precondition_status["P-A"] = bool(np.all(np.isfinite(_err_s_a)) and np.all(_err_s_a > 0))
R_A = log_ratios(_err_t_a, _err_s_a)
verdict_A, RBAR_A, SE_A = ratio_verdict(R_A)
print(
    f"前提条件 P-A(標準の実装の誤差が全標本で有限かつ正): {'成立' if precondition_status['P-A'] else '不成立'}"
)
print(
    f"{_smoke_tag}r_bar = {RBAR_A:+.4f}, SE = {SE_A:.4f}, "
    f"[r_bar - 2SE, r_bar + 2SE] = [{RBAR_A - 2 * SE_A:+.4f}, {RBAR_A + 2 * SE_A:+.4f}], log 2 = {LOG2:.4f}"
    f" -> 判定関数の結果: {verdict_A}"
)

# --- 診断量 ---
print("\n条件ごとの r_bar(FP32)と FP16 の入力での r_bar、誤差の中央値:")
print(
    f"{'causal':>6} | {'sigma':>6} | {'r_bar FP32':>10} | {'r_bar FP16':>10} | {'err_tiled 中央値':>16} | {'err_standard 中央値':>18}"
)
for _causal in CAUSAL_AB:
    for _sigma_name in SIGMAS_AB:
        _sel = [r for r in records_a if r["causal"] == _causal and r["sigma"] == _sigma_name]
        _r32 = log_ratios(
            np.array([r["err_tiled"] for r in _sel]), np.array([r["err_standard"] for r in _sel])
        )
        _r16 = log_ratios(
            np.array([r["err_tiled_fp16"] for r in _sel]),
            np.array([r["err_standard_fp16"] for r in _sel]),
        )
        print(
            f"{str(_causal):>6} | {_sigma_name:>6} | {_r32.mean():>+10.4f} | {_r16.mean():>+10.4f} | "
            f"{np.median([r['err_tiled'] for r in _sel]):>16.3e} | {np.median([r['err_standard'] for r in _sel]):>18.3e}"
        )
_r16_all = log_ratios(
    np.array([r["err_tiled_fp16"] for r in records_a]),
    np.array([r["err_standard_fp16"] for r in records_a]),
)
print(
    f"FP16 の入力での r_bar(全標本) = {_r16_all.mean():+.4f}, SE = {_r16_all.std(ddof=1) / math.sqrt(len(_r16_all)):.4f}"
)
for _sigma_name in SIGMAS_AB:
    _frac = [r["naive_nonfinite_row_fraction"] for r in records_a if r["sigma"] == _sigma_name]
    print(
        f"最大値を引かない素朴な softmax(FP32)で非有限値を含む行の割合(sigma={_sigma_name}): "
        f"平均 {np.mean(_frac):.4f}、最大 {np.max(_frac):.4f}"
    )
for _n in SEQUENCE_LENGTHS_AB:
    _p, _tot = count_block_pairs(_n, _n, *BLOCK_AB, True)
    print(
        f"因果マスクで飛ばしたブロックの割合(N={_n}, B_r=B_c={BLOCK_AB[0]}): {1 - _p / _tot:.3f}({_p}/{_tot} を計算)"
    )

fig, ax = plt.subplots(figsize=(7, 4))
for _causal in CAUSAL_AB:
    for _sigma_name in SIGMAS_AB:
        _sel = [
            (r["n"], math.log(r["err_tiled"] / r["err_standard"]))
            for r in records_a
            if r["causal"] == _causal and r["sigma"] == _sigma_name
        ]
        ax.scatter(
            [x for x, _ in _sel],
            [y for _, y in _sel],
            s=14,
            label=f"causal={_causal}, sigma={_sigma_name}",
        )
ax.axhline(LOG2, color="gray", linestyle="--", label="log 2")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xscale("log", base=2)
ax.set_xlabel("sequence length N")
ax.set_ylabel("log(err_tiled / err_standard)")
ax.set_title(f"{_plot_tag}Experiment A (FP32)")
ax.legend(fontsize=7)
plt.show()
```

    実験 A: 100 標本、10.8s
    前提条件 P-A(標準の実装の誤差が全標本で有限かつ正): 成立
    r_bar = -0.0788, SE = 0.0194, [r_bar - 2SE, r_bar + 2SE] = [-0.1175, -0.0400], log 2 = 0.6931 -> 判定関数の結果: 支持
    
    条件ごとの r_bar(FP32)と FP16 の入力での r_bar、誤差の中央値:
    causal |  sigma | r_bar FP32 | r_bar FP16 |    err_tiled 中央値 |   err_standard 中央値
     False | normal |    -0.2790 |    -1.2651 |        6.497e-07 |          8.431e-07
     False |  large |    +0.0003 |    -3.8490 |        8.408e-06 |          8.382e-06
      True | normal |    -0.0005 |    -0.5426 |        2.220e-07 |          2.636e-07
      True |  large |    -0.0359 |    -3.7153 |        8.057e-06 |          8.071e-06
    FP16 の入力での r_bar(全標本) = -2.3430, SE = 0.1516
    最大値を引かない素朴な softmax(FP32)で非有限値を含む行の割合(sigma=normal): 平均 0.0000、最大 0.0000
    最大値を引かない素朴な softmax(FP32)で非有限値を含む行の割合(sigma=large): 平均 0.6540、最大 0.9879
    因果マスクで飛ばしたブロックの割合(N=80, B_r=B_c=64): 0.250(3/4 を計算)
    因果マスクで飛ばしたブロックの割合(N=160, B_r=B_c=64): 0.333(6/9 を計算)
    因果マスクで飛ばしたブロックの割合(N=320, B_r=B_c=64): 0.400(15/25 を計算)
    因果マスクで飛ばしたブロックの割合(N=640, B_r=B_c=64): 0.450(55/100 を計算)
    因果マスクで飛ばしたブロックの割合(N=1280, B_r=B_c=64): 0.475(210/400 を計算)



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/014_flash_attention/output_30_1.png)
    


### 6.4 実験 B: 再計算による逆伝播の数値的な一致


```python
records_b = []
_t0 = time.time()
for _i_n, _n in enumerate(SEQUENCE_LENGTHS_AB):
    for _causal in CAUSAL_AB:
        for _sigma_name, _sigma in SIGMAS_AB.items():
            for _r in SEEDS_AB:
                _seed = sample_seed(1_000_000, _i_n, _causal, _sigma_name, _r)
                _rec = run_sample_b(_n, _causal, _sigma, _seed)
                _rec.update(n=_n, causal=_causal, sigma=_sigma_name, seed=_seed)
                records_b.append(_rec)
print(f"実験 B: {len(records_b)} 標本、{time.time() - _t0:.1f}s")
assert len({r["seed"] for r in records_b}) == len(records_b)

_err_t_b = np.array([r["err_tiled"] for r in records_b])  # (標本, 3)
_err_s_b = np.array([r["err_standard"] for r in records_b])
precondition_status["P-B"] = bool(np.all(np.isfinite(_err_s_b)) and np.all(_err_s_b > 0))
_r_bg = log_ratios(_err_t_b, _err_s_b)  # (標本, 3)
R_B = _r_bg.max(axis=1)
verdict_B, RBAR_B, SE_B = ratio_verdict(R_B)
print(
    f"前提条件 P-B(標準の実装の誤差が全標本・全勾配で有限かつ正): {'成立' if precondition_status['P-B'] else '不成立'}"
)
print(
    f"{_smoke_tag}r_bar = {RBAR_B:+.4f}, SE = {SE_B:.4f}, "
    f"[r_bar - 2SE, r_bar + 2SE] = [{RBAR_B - 2 * SE_B:+.4f}, {RBAR_B + 2 * SE_B:+.4f}], log 2 = {LOG2:.4f}"
    f" -> 判定関数の結果: {verdict_B}"
)

# --- 診断量: 勾配ごとの r_bar(極値統計の代わりに平均で捉える)、条件ごとの r_bar ---
for _g, _name in enumerate(GRADIENT_NAMES):
    _m, _se = mean_and_standard_error(_r_bg[:, _g])
    print(
        f"  {_name}: r_bar = {_m:+.4f}(SE {_se:.4f})、err_tiled 中央値 {np.median(_err_t_b[:, _g]):.3e}、"
        f"err_standard 中央値 {np.median(_err_s_b[:, _g]):.3e}"
    )
for _causal in CAUSAL_AB:
    for _sigma_name in SIGMAS_AB:
        _idx = [
            i
            for i, r in enumerate(records_b)
            if r["causal"] == _causal and r["sigma"] == _sigma_name
        ]
        print(
            f"  causal={_causal}, sigma={_sigma_name}: r_bar(3 勾配の最大) = {R_B[_idx].mean():+.4f}"
        )
```

    実験 B: 100 標本、9.6s
    前提条件 P-B(標準の実装の誤差が全標本・全勾配で有限かつ正): 成立
    r_bar = +0.1633, SE = 0.0287, [r_bar - 2SE, r_bar + 2SE] = [+0.1059, +0.2208], log 2 = 0.6931 -> 判定関数の結果: 支持
      dQ: r_bar = +0.0084(SE 0.0360)、err_tiled 中央値 1.860e-06、err_standard 中央値 2.366e-06
      dK: r_bar = -0.2003(SE 0.0440)、err_tiled 中央値 2.389e-06、err_standard 中央値 3.218e-06
      dV: r_bar = -0.1867(SE 0.0535)、err_tiled 中央値 1.150e-06、err_standard 中央値 1.631e-06
      causal=False, sigma=normal: r_bar(3 勾配の最大) = -0.0050
      causal=False, sigma=large: r_bar(3 勾配の最大) = +0.1553
      causal=True, sigma=normal: r_bar(3 勾配の最大) = +0.2781
      causal=True, sigma=large: r_bar(3 勾配の最大) = +0.2250


### 6.5 実験 C: 順伝播のピークメモリの系列長に対する次数


```python
def fit_exponent_difference(ns, standard_values, other_values):
    fit_s = fit_power_law_exponent(ns, standard_values)
    fit_o = fit_power_law_exponent(ns, other_values)
    diff = fit_s.exponent - fit_o.exponent
    se = math.sqrt(fit_s.exponent_stderr**2 + fit_o.exponent_stderr**2)
    return fit_s, fit_o, diff, se


def exponent_verdict(diff: float, se: float) -> str:
    if diff - 2 * se > EXPONENT_DIFF_THRESHOLD:
        return "支持"
    if diff + 2 * se < EXPONENT_DIFF_THRESHOLD:
        return "反証"
    return "判定不能"


MEMORY_SOURCE_C = "cuda_allocator" if device.type == "cuda" else "tracker"
records_c = {"standard": [], "tiled": []}
_t0 = time.time()
for _n in SEQUENCE_LENGTHS_C:
    for _impl in records_c:
        records_c[_impl].append(measure_forward_memory(_impl, _n))
print(f"実験 C: {time.time() - _t0:.1f}s、ピークメモリの出典: {MEMORY_SOURCE_C}")
# 条件間で入力の形状・dtype が揃っていること
for _i, _n in enumerate(SEQUENCE_LENGTHS_C):
    assert (
        records_c["standard"][_i]["shape"]
        == records_c["tiled"][_i]["shape"]
        == [BATCH_C, HEADS_C, _n, HEAD_DIM]
    )
    assert records_c["standard"][_i]["dtype"] == records_c["tiled"][_i]["dtype"] == "torch.float32"

precondition_status["P-C1"] = device.type == "cuda"
precondition_status["P-C2"] = all(r["completed"] for rs in records_c.values() for r in rs)
INCREMENT_C = {
    impl: [peak_increment_bytes(r, MEMORY_SOURCE_C) for r in rs] for impl, rs in records_c.items()
}
print(
    f"前提条件 P-C1(CUDA): {'成立' if precondition_status['P-C1'] else '不成立'}、"
    f"P-C2(全水準で完走): {'成立' if precondition_status['P-C2'] else '不成立'}"
)
print(
    f"{'N':>6} | {'標準の増分':>14} | {'閉形式 2BHN^2*4':>15} | {'比':>6} | {'タイリングの増分':>16} | {'閉形式(L + ブロック)':>20}"
)
for _i, _n in enumerate(SEQUENCE_LENGTHS_C):
    _s, _t = INCREMENT_C["standard"][_i], INCREMENT_C["tiled"][_i]
    _est_s = estimate_standard_peak_bytes_c(_n)
    _est_t = BATCH_C * HEADS_C * (_n * 4 + 3 * BLOCK_C[0] * (BLOCK_C[1] + HEAD_DIM) * 4)
    _s_text = "-" if _s is None else f"{_s:,}"
    _t_text = "-" if _t is None else f"{_t:,}"
    _ratio_text = "-" if _s is None else f"{_s / _est_s:.3f}"
    print(
        f"{_n:>6} | {_s_text:>14} | {_est_s:>15,} | {_ratio_text:>6} | {_t_text:>16} | {_est_t:>20,}"
    )

FIT_C = None
if all(v is not None and v > 0 for vs in INCREMENT_C.values() for v in vs):
    _fs, _ft, DIFF_C, SE_DIFF_C = fit_exponent_difference(
        list(SEQUENCE_LENGTHS_C), INCREMENT_C["standard"], INCREMENT_C["tiled"]
    )
    FIT_C = (_fs, _ft)
    verdict_C = exponent_verdict(DIFF_C, SE_DIFF_C)
    print(
        f"{_smoke_tag}b_standard = {_fs.exponent:.4f}(SE {_fs.exponent_stderr:.4f}), "
        f"b_tiled = {_ft.exponent:.4f}(SE {_ft.exponent_stderr:.4f}), "
        f"diff = {DIFF_C:.4f}, SE_diff = {SE_DIFF_C:.4f}, "
        f"[diff - 2SE, diff + 2SE] = [{DIFF_C - 2 * SE_DIFF_C:.4f}, {DIFF_C + 2 * SE_DIFF_C:.4f}] "
        f"-> 判定関数の結果: {verdict_C}"
    )
else:
    DIFF_C = SE_DIFF_C = float("nan")
    verdict_C = "判定不能(完走しなかった水準があり、あてはめを行わない)"
    print(verdict_C)

# --- 診断量: tracker と CUDA のアロケータの比較(CUDA のみ) ---
if device.type == "cuda":
    for _impl, _rs in records_c.items():
        print(
            f"  {_impl}: tracker の増分 / アロケータの増分 = "
            + ", ".join(
                f"{peak_increment_bytes(r, 'tracker') / peak_increment_bytes(r, 'cuda_allocator'):.3f}"
                if r["completed"]
                else "-"
                for r in _rs
            )
        )
        print(
            f"  {_impl}: 呼び出し直前の確保量 - 入力のバイト数 = "
            + ", ".join(
                f"{r['cuda_baseline_bytes'] - r['input_bytes']:,}" if r["completed"] else "-"
                for r in _rs
            )
        )


# --- 診断量: HBM の読み書き量の理論値(要素数、バッチ x ヘッド分) ---
def tiled_io_elements(n: int, block_q: int, block_k: int, causal: bool = False) -> int:
    # Q を 1 回読み、O と L を 1 回書き、計算するブロックの組ごとに K_j と V_j を読む
    pairs = 0
    for i0 in range(0, n, block_q):
        i1 = min(i0 + block_q, n)
        for j0 in range(0, min(i1, n) if causal else n, block_k):
            pairs += min(j0 + block_k, n) - j0
    return BATCH_C * HEADS_C * (n * HEAD_DIM + n * HEAD_DIM + n + 2 * HEAD_DIM * pairs)


def standard_io_elements(n: int) -> int:
    return BATCH_C * HEADS_C * (4 * n * HEAD_DIM + 4 * n * n)


print("\nHBM の読み書き量の理論値(要素数)、B_r = B_c = 128:")
for _n in SEQUENCE_LENGTHS_C:
    _io_t, _io_s = tiled_io_elements(_n, *BLOCK_C), standard_io_elements(_n)
    print(f"  N={_n:>6}: 標準 {_io_s:,}、タイリング {_io_t:,}(標準に対する比 {_io_t / _io_s:.3f})")
_n_max = max(SEQUENCE_LENGTHS_C)
print(
    f"ブロックの行数の関数(N={_n_max}、B_r = B_c): "
    + ", ".join(
        f"{b}: {tiled_io_elements(_n_max, b, b) / standard_io_elements(_n_max):.3f}"
        for b in (16, 32, 64, 128, 256)
    )
)

fig, ax = plt.subplots(figsize=(6, 4))
for _impl in INCREMENT_C:
    _pts = [(n, v) for n, v in zip(SEQUENCE_LENGTHS_C, INCREMENT_C[_impl], strict=False) if v]
    ax.loglog([p[0] for p in _pts], [p[1] for p in _pts], "o-", base=2, label=_impl)
ax.set_xlabel("sequence length N")
ax.set_ylabel("peak memory increment (bytes)")
ax.set_title(f"{_plot_tag}Experiment C ({MEMORY_SOURCE_C})")
ax.legend()
plt.show()
```

    実験 C: 31.4s、ピークメモリの出典: cuda_allocator
    前提条件 P-C1(CUDA): 成立、P-C2(全水準で完走): 成立
         N |          標準の増分 |    閉形式 2BHN^2*4 |      比 |         タイリングの増分 |        閉形式(L + ブロック)
      1024 |     33,554,432 |      33,554,432 |  1.000 |        1,204,224 |            1,196,032
      2048 |    134,217,728 |     134,217,728 |  1.000 |        1,220,608 |            1,212,416
      4096 |    536,870,912 |     536,870,912 |  1.000 |        1,253,376 |            1,245,184
      8192 |  2,147,483,648 |   2,147,483,648 |  1.000 |        1,318,912 |            1,310,720
     16384 |  8,589,934,592 |   8,589,934,592 |  1.000 |        1,449,984 |            1,441,792
    b_standard = 2.0000(SE 0.0000), b_tiled = 0.0648(SE 0.0134), diff = 1.9352, SE_diff = 0.0134, [diff - 2SE, diff + 2SE] = [1.9084, 1.9621] -> 判定関数の結果: 支持
      standard: tracker の増分 / アロケータの増分 = 1.000, 1.000, 1.000, 1.000, 1.000
      standard: 呼び出し直前の確保量 - 入力のバイト数 = 129,864,704, 129,864,704, 129,864,704, 129,864,704, 129,864,704
      tiled: tracker の増分 / アロケータの増分 = 1.000, 1.000, 1.000, 1.000, 1.000
      tiled: 呼び出し直前の確保量 - 入力のバイト数 = 129,864,704, 129,864,704, 129,864,704, 129,864,704, 129,864,704
    
    HBM の読み書き量の理論値(要素数)、B_r = B_c = 128:
      N=  1024: 標準 17,825,792、タイリング 4,722,688(標準に対する比 0.265)
      N=  2048: 標準 69,206,016、タイリング 17,833,984(標準に対する比 0.258)
      N=  4096: 標準 272,629,760、タイリング 69,222,400(標準に対する比 0.254)
      N=  8192: 標準 1,082,130,432、タイリング 272,662,528(標準に対する比 0.252)
      N= 16384: 標準 4,311,744,512、タイリング 1,082,195,968(標準に対する比 0.251)
    ブロックの行数の関数(N=16384、B_r = B_c): 16: 1.994, 32: 0.998, 64: 0.500, 128: 0.251, 256: 0.126



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/014_flash_attention/output_34_1.png)
    


### 6.6 実験 D: 逆伝播のために保存する活性化の系列長に対する次数


```python
records_d = {"standard": [], "flash": []}
_t0 = time.time()
for _n in SEQUENCE_LENGTHS_D:
    records_d["standard"].append(measure_saved_tensor_bytes("standard", _n))
    records_d["flash"].append(measure_saved_tensor_bytes("tiled", _n))
print(f"実験 D: {time.time() - _t0:.1f}s")
for _i, _n in enumerate(SEQUENCE_LENGTHS_D):
    assert (
        records_d["standard"][_i]["shape"]
        == records_d["flash"][_i]["shape"]
        == [BATCH_D, HEADS_D, _n, HEAD_DIM]
    )
    assert records_d["flash"][_i]["num_saved_storages"] == 5, (
        "FlashAttentionFunction が Q, K, V, O, L 以外を保存した"
    )

SAVED_D = {impl: [r["saved_bytes"] for r in rs] for impl, rs in records_d.items()}
_fs, _ff, DIFF_D, SE_DIFF_D = fit_exponent_difference(
    list(SEQUENCE_LENGTHS_D), SAVED_D["standard"], SAVED_D["flash"]
)
verdict_D = exponent_verdict(DIFF_D, SE_DIFF_D)
print(
    f"{_smoke_tag}b_standard = {_fs.exponent:.4f}(SE {_fs.exponent_stderr:.4f}), "
    f"b_flash = {_ff.exponent:.4f}(SE {_ff.exponent_stderr:.4f}), diff = {DIFF_D:.4f}, SE_diff = {SE_DIFF_D:.4f}, "
    f"[diff - 2SE, diff + 2SE] = [{DIFF_D - 2 * SE_DIFF_D:.4f}, {DIFF_D + 2 * SE_DIFF_D:.4f}] "
    f"-> 判定関数の結果: {verdict_D}"
)

# --- 診断量: 閉形式との比較、逆伝播の時間の比(CUDA のみ) ---
_bh = BATCH_D * HEADS_D
print(
    f"{'N':>6} | {'標準(実測)':>14} | {'4BH(3Nd+N^2)':>14} | {'Flash(実測)':>12} | {'4BH(4Nd+N)':>12} | 保存した記憶領域の数(標準 / Flash)"
)
for _i, _n in enumerate(SEQUENCE_LENGTHS_D):
    print(
        f"{_n:>6} | {SAVED_D['standard'][_i]:>14,} | {4 * _bh * (3 * _n * HEAD_DIM + _n * _n):>14,} | "
        f"{SAVED_D['flash'][_i]:>12,} | {4 * _bh * (4 * _n * HEAD_DIM + _n):>12,} | "
        f"{records_d['standard'][_i]['num_saved_storages']} / {records_d['flash'][_i]['num_saved_storages']}"
    )
_closed = fit_power_law_exponent(
    list(SEQUENCE_LENGTHS_D), [4 * _bh * (3 * n * HEAD_DIM + n * n) for n in SEQUENCE_LENGTHS_D]
)
print(f"閉形式 4BH(3Nd+N^2) を同じ水準であてはめたべき指数: {_closed.exponent:.4f}")


def closed_form_delta_b_d(ns) -> float:
    # 閉形式の保存バイト数から求めた Δb(標準 4BH(3Nd+N^2)、Flash 4BH(4Nd+N))
    return (
        fit_power_law_exponent(
            list(ns), [4 * _bh * (3 * n * HEAD_DIM + n * n) for n in ns]
        ).exponent
        - fit_power_law_exponent(list(ns), [4 * _bh * (4 * n * HEAD_DIM + n) for n in ns]).exponent
    )


print(
    f"閉形式から見込まれる Δb: 現在の水準 {SEQUENCE_LENGTHS_D} で {closed_form_delta_b_d(SEQUENCE_LENGTHS_D):.4f}、"
    f"本番の水準 {LEVELS['prod']['SEQUENCE_LENGTHS_D']} で "
    f"{closed_form_delta_b_d(LEVELS['prod']['SEQUENCE_LENGTHS_D']):.4f}"
)
if device.type == "cuda":
    print(
        "逆伝播の時間の比(FlashAttentionFunction / 標準、Python のループのオーバーヘッドを含む): "
        + ", ".join(
            f"N={n}: {f['backward_seconds_median'] / s['backward_seconds_median']:.2f}"
            for n, s, f in zip(SEQUENCE_LENGTHS_D, records_d["standard"], records_d["flash"], strict=False)
        )
    )

fig, ax = plt.subplots(figsize=(6, 4))
for _impl, _vals in SAVED_D.items():
    ax.loglog(SEQUENCE_LENGTHS_D, _vals, "o-", base=2, label=_impl)
ax.set_xlabel("sequence length N")
ax.set_ylabel("bytes saved for backward")
ax.set_title(f"{_plot_tag}Experiment D")
ax.legend()
plt.show()
```

    実験 D: 12.7s
    b_standard = 1.8966(SE 0.0205), b_flash = 1.0000(SE 0.0000), diff = 0.8966, SE_diff = 0.0205, [diff - 2SE, diff + 2SE] = [0.8557, 0.9376] -> 判定関数の結果: 支持
         N |         標準(実測) |   4BH(3Nd+N^2) |    Flash(実測) |   4BH(4Nd+N) | 保存した記憶領域の数(標準 / Flash)
       512 |      5,767,168 |      5,767,168 |    2,105,344 |    2,105,344 | 4 / 5
      1024 |     19,922,944 |     19,922,944 |    4,210,688 |    4,210,688 | 4 / 5
      2048 |     73,400,320 |     73,400,320 |    8,421,376 |    8,421,376 | 4 / 5
      4096 |    281,018,368 |    281,018,368 |   16,842,752 |   16,842,752 | 4 / 5
      8192 |  1,098,907,648 |  1,098,907,648 |   33,685,504 |   33,685,504 | 4 / 5
    閉形式 4BH(3Nd+N^2) を同じ水準であてはめたべき指数: 1.8966
    閉形式から見込まれる Δb: 現在の水準 (512, 1024, 2048, 4096, 8192) で 0.8966、本番の水準 (512, 1024, 2048, 4096, 8192) で 0.8966
    逆伝播の時間の比(FlashAttentionFunction / 標準、Python のループのオーバーヘッドを含む): N=512: 11.58, N=1024: 15.90, N=2048: 22.98, N=4096: 22.55, N=8192: 26.07



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/014_flash_attention/output_36_1.png)
    


### 6.7 実験 E: 読み書き量の削減による速度の系列長依存性(SDPA のバックエンドの比較)


```python
records_e = []
_t0 = time.time()
for _n in SEQUENCE_LENGTHS_E:
    records_e.append(run_level_e(_n, REPETITIONS_E))
print(
    f"実験 E: {time.time() - _t0:.1f}s、バックエンド: "
    + ", ".join(f"{k}={v.name}" for k, v in SDPA_BACKENDS_E.items())
    + f"、dtype: {DTYPE_E}"
)
for _i, _n in enumerate(SEQUENCE_LENGTHS_E):  # 条件間で入力の形状・dtype が揃っていること
    assert records_e[_i]["shape"] == [BATCH_E, HEADS_E, _n, HEAD_DIM] and records_e[_i][
        "dtype"
    ] == str(DTYPE_E)
    assert all(len(records_e[_i]["times"][name]) == REPETITIONS_E for name in SDPA_BACKENDS_E)

precondition_status["P-E1"] = device.type == "cuda"
precondition_status["P-E2"] = all(
    r["ops"][name] == [EXPECTED_SDPA_OPS[name]] for r in records_e for name in SDPA_BACKENDS_E
)
print(
    f"前提条件 P-E1(CUDA): {'成立' if precondition_status['P-E1'] else '不成立'}、"
    f"P-E2(指定したバックエンドで実行): {'成立' if precondition_status['P-E2'] else '不成立'}"
)
for _n, _r in zip(SEQUENCE_LENGTHS_E, records_e, strict=False):
    print(f"  N={_n}: 実行された演算 {_r['ops']}")


def fit_line(x, y) -> tuple[float, float]:
    # y = c + beta x の最小二乗。傾き beta と、その標準誤差 sqrt(sigma^2 / sum (x - mean x)^2)
    # (sigma^2 は残差平方和 / (点の数 - 2))を返す
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    assert len(x) >= 3, "標準誤差には 3 点以上が必要"
    xc = x - x.mean()
    beta = float((xc * (y - y.mean())).sum() / (xc**2).sum())
    residuals = y - (y.mean() + beta * xc)
    sigma2 = float((residuals**2).sum()) / (len(x) - 2)
    return beta, math.sqrt(sigma2 / float((xc**2).sum()))


def level_median_log_ratios(records) -> list[float]:
    # 水準ごとの代表値 m_N = median_k log(t_math / t_fused)
    return [
        float(np.median(np.log(np.array(r["times"]["math"]) / np.array(r["times"]["fused"]))))
        for r in records
    ]


def slope_of_level_medians(ns, records) -> tuple[float, float]:
    # 新基準: m_N = c + beta log N を水準数の点であてはめる(自由度 = 水準数 - 2)
    return fit_line(np.log(ns), level_median_log_ratios(records))


def slope_of_all_repetitions(ns, records) -> tuple[float, float]:
    # 旧基準(診断量): 全水準・全反復の点をまとめて log rho = c + beta log N をあてはめる
    xs, ys = [], []
    for n, r in zip(ns, records, strict=False):
        for t_math, t_fused in zip(r["times"]["math"], r["times"]["fused"], strict=False):
            xs.append(math.log(n))
            ys.append(math.log(t_math / t_fused))
    return fit_line(xs, ys)


def slope_verdict(beta: float, se: float) -> str:
    if beta - 2 * se > 0:
        return "支持"
    if beta + 2 * se < 0:
        return "反証"
    return "判定不能"


BETA_E, SE_BETA_E = slope_of_level_medians(SEQUENCE_LENGTHS_E, records_e)
verdict_E = slope_verdict(BETA_E, SE_BETA_E)
print(
    f"{_smoke_tag}新基準(水準ごとの中央値 m_N の {len(SEQUENCE_LENGTHS_E)} 点の回帰、自由度 {len(SEQUENCE_LENGTHS_E) - 2}): "
    f"beta = {BETA_E:+.4f}, SE = {SE_BETA_E:.4f}, "
    f"[beta - 2SE, beta + 2SE] = [{BETA_E - 2 * SE_BETA_E:+.4f}, {BETA_E + 2 * SE_BETA_E:+.4f}] -> 判定関数の結果: {verdict_E}"
)
print(
    "  m_N = "
    + ", ".join(
        f"N={n}: {m:+.4f}" for n, m in zip(SEQUENCE_LENGTHS_E, level_median_log_ratios(records_e), strict=False)
    )
)
BETA_E_OLD, SE_BETA_E_OLD = slope_of_all_repetitions(SEQUENCE_LENGTHS_E, records_e)
_n_points_old = sum(len(r["times"]["math"]) for r in records_e)
print(
    f"診断量(旧基準、判定には使わない、全 {_n_points_old} 点の回帰、自由度 {_n_points_old - 2}): "
    f"beta = {BETA_E_OLD:+.4f}, SE = {SE_BETA_E_OLD:.4f} -> 旧基準の判定関数の結果(参考): "
    f"{slope_verdict(BETA_E_OLD, SE_BETA_E_OLD)}"
)
# fit_line が fit_power_law_exponent(両対数の最小二乗、010)と同じ傾き・標準誤差を返すことの確認
_check = fit_power_law_exponent(
    list(SEQUENCE_LENGTHS_E), [math.exp(m) for m in level_median_log_ratios(records_e)]
)
assert math.isclose(_check.exponent, BETA_E, rel_tol=1e-9, abs_tol=1e-12)
assert math.isclose(_check.exponent_stderr, SE_BETA_E, rel_tol=1e-9, abs_tol=1e-12)

print(
    f"\n{'N':>6} | {'math 中央値 (ms)':>16} | {'fused 中央値 (ms)':>17} | {'比の中央値':>10} | {'スクラッチ (ms)':>14}"
)
for _n, _r in zip(SEQUENCE_LENGTHS_E, records_e, strict=False):
    _ratios = np.array(_r["times"]["math"]) / np.array(_r["times"]["fused"])
    print(
        f"{_n:>6} | {np.median(_r['times']['math']) * 1e3:>16.3f} | {np.median(_r['times']['fused']) * 1e3:>17.3f} | "
        f"{np.median(_ratios):>10.3f} | {_r['scratch_seconds'] * 1e3:>14.1f}"
    )
    if "cuda_peak_increment_bytes" in _r:
        print(
            "         ピークメモリの増分: "
            + ", ".join(f"{k} {v:,}" for k, v in _r["cuda_peak_increment_bytes"].items())
        )
print(
    "(スクラッチ実装の時間は Python のループによるもので、Flash Attention の本来の性能を反映しない)"
)

fig, ax = plt.subplots(figsize=(6, 4))
for _n, _r in zip(SEQUENCE_LENGTHS_E, records_e, strict=False):
    _ratios = np.array(_r["times"]["math"]) / np.array(_r["times"]["fused"])
    ax.scatter([_n] * len(_ratios), _ratios, s=8, color="tab:blue", alpha=0.5)
ax.plot(
    SEQUENCE_LENGTHS_E,
    [np.median(np.array(r["times"]["math"]) / np.array(r["times"]["fused"])) for r in records_e],
    "o-",
    color="tab:red",
    label="median",
)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("sequence length N")
ax.set_ylabel("t_math / t_fused")
ax.set_title(f"{_plot_tag}Experiment E ({SDPA_BACKENDS_E['fused'].name})")
ax.legend()
plt.show()
```

    実験 E: 7.5s、バックエンド: math=MATH, fused=EFFICIENT_ATTENTION、dtype: torch.float16
    前提条件 P-E1(CUDA): 成立、P-E2(指定したバックエンドで実行): 成立
      N=256: 実行された演算 {'math': ['aten::_scaled_dot_product_attention_math'], 'fused': ['aten::_scaled_dot_product_efficient_attention']}
      N=512: 実行された演算 {'math': ['aten::_scaled_dot_product_attention_math'], 'fused': ['aten::_scaled_dot_product_efficient_attention']}
      N=1024: 実行された演算 {'math': ['aten::_scaled_dot_product_attention_math'], 'fused': ['aten::_scaled_dot_product_efficient_attention']}
      N=2048: 実行された演算 {'math': ['aten::_scaled_dot_product_attention_math'], 'fused': ['aten::_scaled_dot_product_efficient_attention']}
      N=4096: 実行された演算 {'math': ['aten::_scaled_dot_product_attention_math'], 'fused': ['aten::_scaled_dot_product_efficient_attention']}
      N=8192: 実行された演算 {'math': ['aten::_scaled_dot_product_attention_math'], 'fused': ['aten::_scaled_dot_product_efficient_attention']}
    新基準(水準ごとの中央値 m_N の 6 点の回帰、自由度 4): beta = +0.2016, SE = 0.0391, [beta - 2SE, beta + 2SE] = [+0.1234, +0.2799] -> 判定関数の結果: 支持
      m_N = N=256: +1.2921, N=512: +1.5302, N=1024: +1.4574, N=2048: +1.7295, N=4096: +2.0123, N=8192: +1.9267
    診断量(旧基準、判定には使わない、全 120 点の回帰、自由度 118): beta = +0.2000, SE = 0.0092 -> 旧基準の判定関数の結果(参考): 支持
    
         N |    math 中央値 (ms) |    fused 中央値 (ms) |      比の中央値 |     スクラッチ (ms)
       256 |            0.404 |             0.115 |      3.640 |            2.3
             ピークメモリの増分: math 6,556,160, fused 0
       512 |            0.873 |             0.190 |      4.619 |            7.7
             ピークメモリの増分: math 22,548,992, fused 0
      1024 |            2.354 |             0.547 |      4.295 |           26.9
             ピークメモリの増分: math 82,846,208, fused 0
      2048 |            8.013 |             1.423 |      5.638 |          106.7
             ピークメモリの増分: math 316,686,848, fused 0
      4096 |           26.829 |             3.563 |      7.481 |          406.8
             ピークメモリの増分: math 1,237,352,960, fused 0
      8192 |          114.219 |            16.631 |      6.867 |         1599.5
             ピークメモリの増分: math 4,890,624,512, fused 0
    (スクラッチ実装の時間は Python のループによるもので、Flash Attention の本来の性能を反映しない)



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/014_flash_attention/output_38_1.png)
    




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/03_efficient_training/014_flash_attention.ipynb
