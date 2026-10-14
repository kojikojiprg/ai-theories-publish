---
title: "Inverse Cloze Task と passage 単位の 2 段階検索 / Inverse Cloze Task and Two-Stage Retrieval at Passage Level(実装・実験編 5/8)"
---

この記事は後編(実装・実験編 5/8)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/026_inverse_cloze_task_and_two_stage_retrieval-practice-4)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/026_inverse_cloze_task_and_two_stage_retrieval-practice-6)。

### 6.1 実験宣言セル: 共通の設定・検証すること・判定基準・前提条件

**この節の内容は本番実行の前に確定させ、結果を見た後に変更しない。**

#### 共通の設定

- **モデル**: 008 と同じ小型 GPT(4 層、$d_{\mathrm{model}} = 256$、ヘッド数 8、RoPE、RMSNorm、SwiGLU の順伝播ネットワーク(中間次元 683)、正規化前置、埋め込みと出力層の重み共有、系列長 256、語彙サイズ 8192)。
  第 1 段(R1・R2)は encoder として使い、query と passage で重みを共有する。最終正規化層の後の隠れ状態の平均プール(右側のパディングを除く)を L2 正規化したものを埋め込みとする(射影層なし、次元 $d = 256$)。**全パラメータを更新する。**
- **データ**: 5.3 節のとおり。コーパスの先頭の 356 記事を除く 9,470 記事を、記事を単位に学習用(9,070 記事)・検証用(100 記事)・評価用(300 記事)に分ける([024](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/024_text_embedding_and_retriever-theory) と同じ分割)。
  **学習用・検証用・評価用のすべてが、008 の事前学習とトークナイザの学習の範囲(参照コーパス)の外にある**(5.3 節のセルで、文字位置の範囲のアサーションにより確かめる)。
- **第 1 段の学習**: 損失は in-batch negatives の InfoNCE(query から passage への方向、温度 $\tau = 0.05$ で固定)と Matryoshka Representation Learning($\mathcal{M} = \{16, 32, 64, 128, 256\}$、重み $c_m = 1 / \lvert \mathcal{M} \rvert$)で、
  024 の条件 C5 と同じ損失の形。AdamW(重み減衰 0.1、行列形の重みにのみ掛ける)、バッチ 64 組(**32 記事 × 各記事 2 組**。同じ記事の組を意図的に同じバッチに入れる。損失での扱いだけを定義に合わせる: R1 は同じ記事の組の passage を負例から除き(1 行あたりの負例 62 個)、R2 は負例として数える(63 個))、最初の 10% のステップで線形 warmup の後 cosine で最大値の 0.01 倍まで減衰、gradient clipping の閾値 1.0。
  CUDA では FP16 の autocast と動的損失スケーリング。埋め込みの正規化・類似度・損失・評価は FP32。**学習ステップ数は $T_1 = 1024$**(全条件・全段階で共通。024 の C5 と同じ)。**query は、どちらの条件も「窓から抜いた 1 文」で、文を文脈に残す確率 $r = 0.1$ も共通。** 正例は、R2 が query を抜いた窓の文脈、R1 が同じ記事のもう一方の窓の文脈(3.2 節)。
- **cross-encoder**: [025](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/025_ann_search_and_reranking-theory) と同じ構成(008 の重みから始め、列`[query] [区切り] [passage] [終端]`を因果マスクのもとで符号化して、終端の位置の出力に線形のヘッドを付ける。全パラメータを更新する)。
  損失は 1 個の正例と 7 個の負例の 8 個の候補の上の softmax 交差エントロピー(in-batch negatives は使わない)。1 ステップ 8 個の query。学習率の形と gradient clipping は第 1 段と同じ。
  **学習ステップ数 $T_2$** は、候補 $(2048, 4096, 8192)$ から 6.3 節のパイロットで自動で選ぶ。
- **シード**: シード $s$ が動かすのは、記事の並び・窓・抜く文・文を文脈に残す組(第 1 段は $26100 + s$。R1 と R2 で同一)と、cross-encoder の組($26500 + s$)・ヘッドの初期化($26200 + s$)である。R1・R2 の初期値はシードによらず 008 の重みで同一。
  同じシードの R1 と R2 は同じ記事・窓・抜く文を見る(6.12 節で確かめる)。cross-encoder のシード $s$ は、同じシード $s$ の R2 の第 1 段の上位 $K$ 件で学習し、評価する。
- **評価**: Recall@10(3.5 節)。学習の最終ステップの重みで、FP32、全件の内積で順位をつける。$R_s(c)$ を条件 $c$・シード $s$ の Recall@10 とする。
  **実験 A・B は同じ索引・同じ query** で、関連のラベルだけを変える。索引はコーパス全体(すべての passage を、抜く文を 1 つ固定して除いた形にしたもの。対象の passage 約 9.4 万個、動作確認では約 5 千個)で、
  query は評価用の記事の passage から取り除いた文(約 3,200 個。記事単位のラベルの正解が 1 つ以上ある query に限る)。**実験 A** は passage 単位のラベル(正解は query の passage だけ)、
  **実験 B** は記事単位のラベル(正解は同じ記事の他の passage。query の passage 自身は候補から除く。Recall@10 は上位 10 件に正解が 1 つ以上ある query の割合。024 と同じ定義)。
  診断量として、024 と同じ形式の記事単位の評価(024 の評価用の索引と query、3,244 個。固定長)を、R1・R2・024 の Hub のモデルで測る(判定には使わない)。
- **008 の重みは 1 つのチェックポイントである。** 実験 A・B の結論は、その 1 つのチェックポイント(`kojikojiprg/ai-theories-small-gpt-en`の`main`)から始めた場合に限る。

#### 条件の対応表

| 記号 | 正例の定義 | 学習の組 | 並べ替え | 使う実験 |
|---|---|---|---|---|
| R1 | 記事単位 | query = 窓から抜いた文。正例 = 同じ記事のもう一方の窓の、文を除いた形。同じ記事の組を負例から除く | なし | 実験 A の基準、実験 B の条件 1 |
| R2 | passage 単位 | Inverse Cloze Task: query = 窓から抜いた文。正例 = その窓の文脈(10% は文を残す)。同じ記事の組を負例として数える | なし | 実験 A の条件 2、実験 B の条件 2、実験 C の基準 |
| R3 | passage 単位 | R2 の上位 50 件を cross-encoder で並べ替える | cross-encoder | 実験 C の条件 |

数式の添字の R1・R2 はこの表の記号である。

#### 揃えた量と揃えなかった量

| 量 | 扱い |
|---|---|
| 起点(008 の`main`)・モデルの構成・総パラメータ数・$T_1$・バッチサイズ・温度・損失の形・重み減衰・学習率スケジュールの形・gradient clipping の閾値 | **揃える**(R1 と R2 で同じ) |
| 記事の並び・窓・抜く文・文を残す組・バッチ(32 記事 × 2 組。同じシードの R1 と R2) | **揃える**(同じバッチの query と passage のトークンは R1 と R2 で同じ) |
| 入力の形式(query = 抜いた文、passage = 文を除いた形)・評価の索引・query・評価の手順 | **揃える**(R1 と R2 は同じ索引・同じ query で評価し、関連のラベルだけを変える) |
| 学習率 | **揃えない**(条件ごとに較正する。下の「学習率の較正」) |
| 正例の対応と、同じ記事の組の扱い(正例 + in-batch の負例) | **揃えない**(関連性の定義を、正例と in-batch の負例の両方に一貫して当てはめる。これが実験 A・B の介入) |
| 1 行あたりの負例の数 | **揃えない**(R1 は 62 個、R2 は 63 個。同じ記事の組を除くことの結果で、介入に含まれる) |
| cross-encoder の追加の学習量 | **実験 C では揃えない**(実用の問い。実験 C の「解釈の注意」) |

#### 学習量についての注意

1. **エポック数**: 組を作れる学習用の記事は 8,639 本(141,126,568 トークン)で、1 ステップは 64 個の窓($64 \times 128 = 8{,}192$ トークン)を見る(query は窓の一部)。$T_1 = 1024$ は **約 0.059 エポック分** で、学習用の記事のトークンの大半は一度も見ない
   (5.3 節の出力)。窓は記事のランダムな位置から引くので、同じ passage を繰り返し見ることは稀である。
2. **学習が飽和していない状態での比較になりうる**: 024 の C5 は $T_1 = 1024$ でも検証用の Recall@10 が学習の最後まで伸びていた。実験 A・B の結果は、この学習量での比較である。
3. **cross-encoder の学習量**: $T_2 = 2048, 4096, 8192$ は、8 query/ステップで見る query が 16,384・32,768・65,536 個で、学習に使える query(正例が上位 50 件に入る query)の数に対するエポック数は 5.3 節・6.3 節の出力に印字する。
   学習量を下げる段階(「実行計画」)が選ばれうるのは第 1 段のシード数・cross-encoder のシード数・学習率の較正の方式などで、$T_1$ と $T_2$ は段階では下げない。
   ただし、$T_2$ の候補のうち小さいものが選ばれる(パイロットで早く立ち上がる)場合は、学習が飽和していない状態での比較になる。

#### 学習率の較正(本番の冒頭、6.5 節)

- **格子**: 第 1 段は R1・R2 とも、中心 $4.8 \times 10^{-4}$(024 の C5 の較正で選ばれた値)の $\{1/4, 1, 4\}$ 倍(公比 4 の 3 点: $1.2 \times 10^{-4}$、$4.8 \times 10^{-4}$、$1.92 \times 10^{-3}$)。
  cross-encoder は、中心 $2.4 \times 10^{-4}$(025 の学習のパイロットで最後の区間の訓練損失が最も低かった学習率)の $\{1/4, 1, 4\}$ 倍($6 \times 10^{-5}$、$2.4 \times 10^{-4}$、$9.6 \times 10^{-4}$)。
  公比を 4 にした理由は、指標が学習率に対して緩やかにしか変わらず、公比 2 の格子では隣の点との差が評価の変動より小さくなって、最良が観測のノイズで端になりやすいためである(024 と同じ。観測の結果の向きに依らない、格子の刻みについての一般論による選択)。
- **較正のスケール**: 較正は **本番と同じスケール**(第 1 段は $T_1 = 1024$、cross-encoder は選ばれた $T_2$)で、**較正専用のシード(番号 90。実験のシードと共有しない)** の 1 シードで各格子点を学習して行う。評価用の集合は較正に使わない。
- **指標**: 第 1 段は、検証用の記事の passage だけを索引にした **同じ索引・同じ query** での、passage 単位のラベルと記事単位のラベルの Recall@10 の **平均** $M$。R1 と R2 の両方を同じ指標で較正する(どちらのラベルにも偏らない指標。一方だけに自分のラベルの指標を使うと、実験 A・B の対比量が、自分のラベルで較正した側に有利に偏るため)。
  cross-encoder は、検証用の query の並べ替えの Recall@10(R2 のシード 90 の上位 50 件を並べ替える)。
- **最良と次点の差が評価の変動より小さい場合の選び方(本番の前に宣言)**: 指標が最大の点から、**指標の標準偏差(記事を単位とするブートストラップ)の 1 倍以内** の候補(学習の成立の前提条件 P1(a) を満たす点だけ)のうち、
  **学習の成立の余裕が最大の点** を選ぶ。余裕は、第 1 段では「1 − 最後の区間の訓練損失 ÷ 同じ組での学習前のモデルの損失」、cross-encoder では「最後の区間の損失の差の平均(学習後 - 学習前)を標準誤差で割った値の符号を反転したもの」(どちらも大きいほど学習前より損失が下がっている)。
  同点なら小さい学習率。P1(a) を満たす点が 1 つもないときは、余裕が最大の点を選び、P1 の不成立として進める(その条件が入る実験は「前提不成立」になる)。
- **拡張の規則**: 選ばれた点が格子の端に来た場合は、その方向に公比 4 で 1 点だけ格子を拡張して学習し、改めて同じ規則で選ぶ(1 回のみ)。
- **P0(較正)**: その実験の対比量に入る条件のうち、自分自身を較正した条件で、選んだ学習率が格子の **内点** であること(拡張後も端なら不成立)。規則で決めた条件(下の段階 4・5)は対象外。
- **規則で学習率を決める段階(削る段階 4・5)**: 段階 4 以降は、R1 の較正をやめ、**024 の C5(同じ構成・同じ $T_1$・同じ起点。組の作り方は違う)の較正で選ばれた $4.8 \times 10^{-4}$** を使う。
  段階 5 以降は、cross-encoder の較正をやめ、中心の $2.4 \times 10^{-4}$ を使う。規則で決めた条件は P0 の対象外になる。**最適からずれたときの対比量の偏りの向き**:
  - R1(段階 4 以降): $4.8 \times 10^{-4}$ は 024 の組の作り方で較正した値で、本トピックの R1 の組の作り方(query が抜いた文、同じ記事の組を負例から除く)での最適とは限らない。R1 の学習率が最適からずれていると、R1 の Recall@10 が下がる。
    **実験 A の対比量は正の向き(仮説に有利)、実験 B の対比量は負の向き(仮説に不利)に偏る**。したがって実験 A の **支持** と実験 B の **反証・判定不能** は、偏りの向きを考慮して読む(A の支持は偏りで過大になりうる。B の支持は偏りがあっても成り立つ結果として読む)。
  - cross-encoder(段階 5 以降): 学習率が最適からずれていると、並べ替えの Recall@10 が下がり、**実験 C の対比量は負の向き(仮説に不利)に偏る**。支持は偏りがあっても成り立つ結果として読み、判定不能は隠れた正の効果を排除しない。反証は、並べ替えが劣ることと学習率の不足を区別できない。
  段階 4・5 を使わないとき(段階 0〜3)は、この制限はない。
- **較正の段階で、学習の成立の余裕が全格子点で小さい条件を検出した場合**(cross-encoder で、すべての格子点で P1(a) が不成立): その条件が入る実験(実験 C)は、P1 の不成立として「前提不成立」で進める。本番は 1 回で完結させるので、較正の出力を見てから設計を変えない。

#### 共通の前提条件

前提条件は、検証したい仮説(対比量の向き)とは独立な量で定義する。成立しなかった実験は、支持・反証・判定不能のいずれとも判定せず「前提不成立」として記録する。

- **P0(較正)**: 上の「学習率の較正」のとおり。対象は、実験 A: R1・R2、実験 B: R1・R2、実験 C: R2・cross-encoder(規則で決めた条件は対象外)。
- **P1(学習の成立)**:
  - **第 1 段(実験 A・B)**: (a)対象の条件の全シードで、訓練損失がすべてのステップで有限であり、**最後の 5% のステップの訓練損失が、同じステップの組での学習前のモデルの損失の 0.9 倍以下**。
    (b)対象の条件の全シードで、**自分の定義のラベルの検証用の Recall@10(R1: 記事単位のラベル、R2: passage 単位のラベル。検証用の記事の passage だけの索引・同じ query)が、学習前の 008(同じ構成)の同じ集合での値より 0.04 以上高い**。
  - **cross-encoder(実験 C)**: (a)全シードで、訓練損失が有限で、最後の 5% のステップの訓練損失が、同じ組での学習前のモデルの損失より **標準誤差の 2 倍以上低く**、かつ一様な予測の損失 $\ln(1 + n) = \ln 8$ 未満。
    (b)検証用の query の並べ替えの Recall@10 の、学習前の cross-encoder からの増分のシード平均が、**判定と同じ式の標準偏差 $\sigma_g$ の 2 倍以上**。
  すべて、**更新なしの状態では成立しない量** で定義する: (a)の比は更新がなければ厳密に 1(差は 0)、(b)の増分は更新がなければ 0 である(学習率 0 の学習で不成立になることは 5.4 節の単体テストで確かめる)。
- **P2(改善の余地・天井と床)**: 実験 A・B は、基準となる条件の Recall@10 が **床 $0.02$ 以上、天井 $0.95$ 以下**(実験 A: $R(\mathrm{R1})$ の passage 単位のラベル、実験 B: $R(\mathrm{R1})$ の記事単位のラベル)。差ではなく基準となる条件の値で定義する。
  床の 0.02 は、passage 単位の評価でランダムな順位の期待値(約 0.0001)の約 200 倍で、第 1 段が検索として働いていることの最低限の確認である。実験 C は、$R(\mathrm{R2})$ が coverage より 0.05 以上低いこと。
- **P3(実験 C の第 1 段の候補)**: 第 1 段(R2)の上位 $K = 50$ 件に正例が入る割合(coverage)が **0.35 以上**、かつ coverage − $R(\mathrm{R2})$(並べ替えの余地)が **0.10 以上**(全シードの平均)。

**閾値の根拠**(観測の結果の向きに依らない一般論による):

- **P1(a)の 0.9**: 024 の C5($T_1 = 1024$、中心の 1/4・1・4 倍の学習率)の最後の区間の訓練損失 ÷ 学習前のモデルの損失は 0.538〜0.613 で、学習が進んでいない場合は 1(更新がない場合は厳密に 1)。0.9 は、その間にあり、
  024 の C2 の $T = 256$ の 5 シードの標準偏差 0.011(ばらつきが最大の見込みの値)に対して、正常な最大の 0.613 から約 26 標準偏差、学習が進んでいない場合の 1 から約 9 標準偏差離れている。**測っていない範囲**: R1・R2 の比(組の作り方が 024 の C5 と違う)と、R1・R2 の格子の外側(拡張した点)での比。比が 0.9 と 1 の間に来る場合は、学習が弱いと判定され、前提不成立になる。R1 は同じ記事の組を負例から除くので、比の分母(学習前のモデルの損失)も違う。
- **P1(b)の 0.04**: 024 の C2 の検証用の Recall@10 の学習前からの増分(+0.1373〜+0.1599、$T = 1024$)と、増分が 0 の学習が進んでいない場合の間に置いた値(024 の閾値と同じ)。**測っていない範囲**: R1・R2 の、自分のラベルでの増分(組の作り方と評価のラベルが 024 と違う)。
- **cross-encoder の P1(a)の標準誤差 2 倍**: 025 と同じ(最後の区間の差の平均の標準誤差)。更新がなければ差は 0 で、(a)は成り立たない。
- **P2 の床 0.02・天井 0.95、P3 の 0.35・0.10**: 025 の第 1 段(記事単位の定義の 024 のモデル、コーパス全体 94,486 passage)で、上位 50 件の coverage が 0.4094、並べ替えなしの Recall@10 が 0.2199 だった値を、P3 の下限(0.35)と余地(0.10)の目安にした
  (passage 単位の R2 の値は事前に分からない。測っていない範囲)。

#### 前提条件の閾値の下限と、想定される増分(評価の変動の項から)

cross-encoder の P1(b)は「増分のシード平均 $\ge 2\sigma_g$」の形で、$\sigma_g = \sqrt{\mathrm{Var}_s(g_s) / n + \sigma_{\mathrm{boot}}^2}$ のブートストラップの項 $\sigma_{\mathrm{boot}}$ は、シード数 $n$ を増やしても小さくならない。
閾値は常に $2\sigma_{\mathrm{boot}}$ 以上である。評価の変動の項の見積もりは、025 の同じ大きさの検証用の query(877 個、100 記事)での値を使う: 学習した dual encoder の増分(5 シード平均)で 0.0070、ランダムな負例で学習した cross-encoder で 0.0137、パイロット(1 シード)で 0.0176。
本トピックの検証用の query は passage 単位で約 1,000 個(100 記事)で、同程度の大きさなので、$\sigma_{\mathrm{boot}} \approx 0.007$〜$0.018$、閾値の下限は **約 0.014〜0.035** である(学習は不要な計算)。

**想定される増分との比較**: 025 で、学習した dual encoder の増分は +0.018(閾値 0.014 に対して成立)、ランダムな負例で学習した cross-encoder の増分は +0.0176 で、ブートストラップの項 0.0137 の 2 倍(0.0274)に届かず不成立だった。この下限は、増分が小さい状態を検出するために置いている。
並べ替えの余地は、coverage − 学習前の cross-encoder の Recall@10 で、025 では 0.47 − 0.18 = 約 0.29 だった。passage 単位の R2 の coverage は事前に分からないが、cross-encoder が学習の余地の 12% 以上
(0.035 ÷ 0.29)を実現すれば、閾値の上端でも成立する。学習が成立すれば超える見込みで、025 のように学習が立ち上がらなければ超えない。**パイロットの規則(6.3 節)は、この増分が閾値を超えるかを、本番の $T_2$ の選択に使う。**

#### 判定の形と標準偏差の単位

すべての実験で、対比量 $\Delta$ は **同じシードどうしで対応をとった差 $d_s$ のシード平均** である。標準偏差は

$$
\sigma = \sqrt{\frac{\mathrm{Var}_s(d_s)}{n} + \sigma_{\mathrm{boot}}^2}
$$

とする($n$ はシード数、$\mathrm{Var}_s$ はシード間の不偏分散)。**導出**: $d_s = \delta + a_s + e$ と書く。$\delta$ は真の効果、$a_s$ は学習の乱数(記事の並びと組、cross-encoder ではヘッドの初期化)による変動、$e$ は評価用の query が有限であることによる変動である。
評価用の query は全シードで共通なので、$e$ はシードによらず同じ値をとり、シード間の分散には現れない。したがってシード平均の分散は $\mathrm{Var}(a_s)/n + \mathrm{Var}(e)$ で、第 1 項をシード間の分散から、第 2 項を評価用の query の再標本化から見積もる。
$\sigma_{\mathrm{boot}}$ は、**記事を単位とする対応付きのクラスタブートストラップ**(015・024 と同じ方式。同じ記事から作った query どうしは独立でないので、記事を復元抽出する。比べる全条件・全シードに同じ再標本を使う)で、
対比量(シード平均)を反復ごとに作り直したときの標準偏差である。Recall@10 は query ごとの 0 か 1 なので、各反復の値は選ばれた記事の query の hit の割合である。

判定は 3 値で行う(閾値は $2\sigma$): **支持** は $\Delta > 2\sigma$、**反証** は $\Delta < -2\sigma$(期待と逆の向きの差が閾値を超えて現れた)、**判定不能** は $\lvert \Delta \rvert \le 2\sigma$(ノイズ床の 2 倍以内)。

#### 実行計画(削る段階)と、その自動選択

本番は Colab の 1 セッション(T4 で 120 分)で完結させる。本番の冒頭で測った時間(6.2 節)と、cross-encoder のパイロット(6.3 節。その時間も予算に含める)の後の経過時間から、次の 8 通りの計画の残りの実行時間を見積もり(6.4 節)、
**予算に収まる計画のうち番号の最も小さいもの** を選ぶ。選択は見積もりと経過時間のみに基づき、どの実験の結果も参照しない。最も下位の計画でも収まらなければ、本番の学習の前に停止する。

**見積もりの安全係数**: 学習(第 1 段・cross-encoder)の時間の見積もりには、本番の冒頭で測った 1 ステップの時間に **1.1 倍の安全係数** を掛ける(パイロットの学習は、実際に使った時間を経過時間に入れるので、見積もりに入れない)。
理由: 025 の再実行に向けたパイロットで、T4 の実際の 1 ステップ(80〜81 ms)が本番の冒頭の計測値(75 ms)より約 7% 長かったため、計測値の外れを吸収する。学習以外の時間(評価・採掘・HNSW の構築・固定の余裕)には掛けない。

**削る段階**(番号が大きいほど多く削る。各段階は前の段階に 1 つだけ変更を加える。シード数は、削る前 $(5)$、中 $(3)$、低 $(2)$ の 3 水準):

| 段階 | 変更 | 観察 D の索引の件数 | cross-encoder のシード数 | 第 1 段のシード数 | R1 の較正 | cross-encoder の較正 |
|---|---|---|---|---|---|---|
| 0 | (削らない) | 16384 | 5 | 5 | する | する |
| 1 | 観察 D の索引の件数を 1 段下げる | 8192 | 5 | 5 | する | する |
| 2 | cross-encoder のシード数を中にする | 8192 | 3 | 5 | する | する |
| 3 | 第 1 段のシード数を中にする | 8192 | 3 | 3 | する | する |
| 4 | R1 の較正をやめ、024 の値にする | 8192 | 3 | 3 | しない | する |
| 5 | cross-encoder の較正をやめ、中心の値にする | 8192 | 3 | 3 | しない | しない |
| 6 | cross-encoder のシード数を低にする | 8192 | 2 | 3 | しない | しない |
| 7 | 第 1 段のシード数を低にする | 8192 | 2 | 2 | しない | しない |

**この順にした理由**:

1. **観察 D を最初に削る**: 判定を設けない観察で、削っても判定に影響しない。
2. **cross-encoder のシード数を次に削る**(段階 2): cross-encoder の 1 シードあたりの学習が最も長い($T_2$ ステップ)ので、削ったときの時間の節約が大きい。第 1 段のシード数以下に保つ。
3. **第 1 段のシード数を削る**(段階 3): 実験 A・B・C のシード数が減るが、対比量の標準偏差は記事を単位とするブートストラップの項が支配的な見込み(024 の実験 A・B・D で、ブートストラップの項がシード間の項の約 1.7〜2.3 倍)で、5 から 3 に減らしたときの閾値の増え方が小さい。
4. **学習率の較正をやめる**(段階 4・5): 較正は学習を条件ごとに最大 4 回行うので、時間の節約が大きいが、規則で決めた学習率が最適からずれたときの対比量の偏りの向きが条件によって逆になる(上の「学習率の較正」)。**較正の範囲を削る段階** として、(1)規則で決めた条件は P0 の対象外になること、(2)偏りの向きを上に記した。
   支持を誤らせうる偏り(実験 A の R1)を持つ段階 4 を、仮説に不利な向きの偏り(実験 C の cross-encoder)を持つ段階 5 より先にした。
5. **シード数を最後に「低」にする**(段階 6・7): シード数 2 は、シード間の分散を推定できる最小の数である。

**$T_1$ と $T_2$ は段階では下げない**: $T_1 = 1024$ は 024 の C5 の再現(R1)の前提で、下げると R1 の再現の確認が意味を持たなくなる。$T_2$ は、パイロットが決める学習の成立の量であり、予算のために下げると、前提条件が成立しない状態を作る。

**パイロットの分岐の規則(本番の前に宣言。6.3 節)**: 学習を伴うパイロットは、本番と同じセッションの冒頭で、単一の条件の量だけを測って行う。

1. **第 1 段のパイロット**: R2 を、較正専用のシード 90・$T_1 = 1024$・中心の学習率 $4.8 \times 10^{-4}$ で学習し、第 1 段の上位 $K = 50$ 件を求める(学習用・検証用・評価用の query)。この学習は、R2 の較正の格子の中心の点として **使い回す**(同じ設定であることをアサーションで確かめる)。
2. **cross-encoder のパイロット**: シード 90 の上位 50 件で、cross-encoder を、中心の学習率 $2.4 \times 10^{-4}$・較正専用のシード 90 で、$T_2 = 2048, 4096, 8192$ の順に学習する。各 $T_2$ の後に、検証用の query で次の 3 つを調べ、**3 つがすべて成立した最初の $T_2$ で止める**。
   - (i)学習の成立 P1(a)(最後の区間の訓練損失が、同じ組での学習前のモデルの損失より標準誤差の 2 倍以上低く、かつ $\ln 8$ 未満)。
   - (ii)検証用の query の並べ替えの Recall@10 の、学習前の cross-encoder からの増分が、記事を単位とするブートストラップの標準偏差(増分の再標本の標準偏差)の 2 倍以上(P1(b) の閾値の式の、シード間の項がない 1 シード版)。
   - (iii)検証用の query の並べ替えの Recall@10 が、**R2 だけ(並べ替えなし)の値を上回る**(点推定)。
3. **本番の $T_2$** は、(i)〜(iii)が成立した最小の候補。3 つの候補のどれでも成立しなければ、**最大の候補(8192)を使い**、実験 C は事前の規則のとおり進める(前提条件の成否は本番の結果で決まる。パイロットが失敗しても実験 C を中止しない: パイロットは 1 シード・1 つの学習率の量で、中止すると実験 C の情報が得られないため)。
4. パイロットの時間は予算に含め、パイロットの後の経過時間で、削る段階を選び直す。パイロットで学習した cross-encoder の最後の 1 回は、同じ設定(シード 90、選ばれた $T_2$、中心の学習率)の較正の格子の中心の点として使い回す。

**パイロットの分岐に下流の指標((iii)の Recall@10)を含める理由と、その影響**: 025 では、訓練損失が下がっても、並べ替えの Recall@10 は何もしない場合を上回らなかった(損失が下がることと、下流の指標が基準を上回ることは別)。
訓練損失だけで $T_2$ を決めると、損失が下がっても役に立たないモデルに予算を使う。(iii)を含めると、選ばれた $T_2$ は「検証用の query で、1 シードの cross-encoder が R2 だけを上回った」最小の学習量になる。
影響: $T_2$ の選択に、検証用の query(判定には使わない集合)の下流の指標を使うので、本番の cross-encoder が、実験 C に有利な向きに $T_2$ を選ばれている(選択の偏り)。偏りは $T_2$ の 3 通りの選択に限られ、評価用の query と別のシード・別の集合での判定なので、
実験 C の支持は「このセッションのパイロットで R2 を上回る見込みのある学習量で、評価用の query でも上回ったか」として読む。(i)〜(iii)がどれも成立しなかった場合は、選択の偏りがなく、前提不成立または反証・判定不能の見込みが高い。

#### 実験 A: passage 単位のラベルの評価での、関連性の定義の効果

**検証すること**: 同じ索引・同じ query の passage 単位のラベルの評価(正解は query の passage だけ)で、passage 単位の定義(Inverse Cloze Task)で学習した検索器(R2)の Recall@10 が、記事単位の定義で学習した検索器(R1)より高いか。

**条件**: R1 と R2。どちらも 008 の重みから始め、平均プール・因果マスク・Matryoshka Representation Learning の損失・同じ記事・同じ窓・同じ抜く文・同じ入力の形式・同じ $T_1$ で、
**違いは関連性の定義だけ** である(正例の対応と、in-batch の同じ記事の組の扱い。3.2・3.3 節)。

**対比量**:

$$
\Delta_A = \frac{1}{n} \sum_{s} \left( R_s(\mathrm{R2}) - R_s(\mathrm{R1}) \right)
$$

(正を期待。$n$ は実験 A のシード数、$R_s(c)$ はシード $s$ の条件 $c$ の passage 単位のラベルでの Recall@10。)標準偏差 $\sigma_A$ は「判定の形」のとおり、$d_s = R_s(\mathrm{R2}) - R_s(\mathrm{R1})$ のシード間の分散と、記事を単位とするクラスタブートストラップの分散を合成する。

**判定基準**: $\Delta_A > 2\sigma_A$ なら支持、$\Delta_A < -2\sigma_A$ なら反証、それ以外は判定不能。

**介入の直接の作用点と、対比量までの距離**: 介入(関連性の定義)が直接作用するのは、**学習の信号の中身** である。正例の対応と、in-batch の負例のうち同じ記事の組をどう扱うかが、その結果として「同じ記事の中の passage どうしを区別できるか」に現れる。
Recall@10 は、その埋め込みの内積による順位づけと上位 10 件での二値化という 2 段階だけ下流の量で、作用点に近い。作用点に近い診断量として、(1)上位 10 件のうち、**query の正例と同じ記事の passage の割合**、
(2)1 位が **同じ記事の別の passage** である query の割合(話題は合っているのに passage を取り違えた割合)、(3)閾値を使わない順位の量として平均逆順位(Mean Reciprocal Rank)を併記する(判定には使わない)。

**対比量が閾値を超えうる状態と、仮説との整合**: $\Delta_A > 2\sigma_A$ となりうる状態は次のとおり。(i)R2 が同じ記事の passage を区別する表現を学んだ(仮説どおり)。(ii)R1 の学習が進んでいない(P1 で検出して前提不成立にする)。
(iii)R1 の学習率が最適からずれている(段階 4 以降。上の「学習率の較正」のとおり、仮説に有利な向きの偏り)。(iv)R1 の値が上限に近く差が出ない(P2 で検出する。これは支持ではなく判定不能・前提不成立の側)。
入力の形式・索引・query は両条件で同じなので、「各条件が学習で見た形式の評価で勝つ」経路はない。ただし、評価の passage 単位のラベルは、R2 の学習の目標(正例が 1 つの passage)と一致し、R1 の目標(同じ記事のどの passage でも正解)とは一致しない。
この一致は **関連性の定義そのものの帰結** であり、介入の効果に含まれる(実験 A は、各定義で学習した検索器が、その定義のラベルで評価されたときに優れるかを見る)。

**解釈の注意(R2 が同じ記事の困難な負例を見ること)**: 本トピックは、in-batch negatives に同じ記事の組を意図的に入れ、R2 はそれを負例として数える。R2 が同じ記事の passage を区別する学習の信号を得ることは、
**実験 A の対比量を正の向きに大きくしうる**。これは「関連性の定義を、正例と in-batch の負例の両方に一貫して当てはめる」という介入の一部で、設計どおりの効果である。
一方、R1 は同じ記事の組を負例から除くので、負例の数が 1 行あたり 62 個と R2(63 個)より 1 つ少ない。この差は小さく、介入に含まれる(6.6 節で数を印字する)。

**前提条件**: P0(R1・R2)、P1(R1・R2)、P2($R(\mathrm{R1})$ の passage 単位のラベルが $[0.02, 0.95]$)。

**診断量(判定なし)**: 上の(1)〜(3)、R1・R2 の最後の区間の訓練損失、学習率とその出どころ、clipping の発動、1 行あたりの負例の数、切り詰めた query の割合、
024 と同じ形式の記事単位の評価(R1・R2・024 の Hub のモデル。下の「診断量: 024 と同じ形式の評価と R1 の再現」)。

#### 実験 B: 記事単位のラベルの評価での、関連性の定義の効果

**検証すること**: 実験 A と同じ索引・同じ query の記事単位のラベルの評価(正解は同じ記事の他の passage。query の passage 自身は候補から除く)で、R1 の Recall@10 が R2 より高いか。

**条件**: 実験 A と同じ R1 と R2(同じ学習。再学習しない)。評価の関連のラベルだけが違う。

**対比量**:

$$
\Delta_B = \frac{1}{n} \sum_{s} \left( R_s(\mathrm{R1}) - R_s(\mathrm{R2}) \right)
$$

(正を期待。$R_s(c)$ は記事単位のラベルでの Recall@10。上位 10 件に正解が 1 つ以上ある query の割合。)標準偏差 $\sigma_B$ は実験 A と同じ構成で、ブートストラップのクラスタも実験 A と同じ(同じ query)。

**判定基準**: $\Delta_B > 2\sigma_B$ なら支持、$\Delta_B < -2\sigma_B$ なら反証、それ以外は判定不能。

**介入の直接の作用点と、対比量までの距離**: 実験 A と同じ。作用点に近い診断量として、記事単位のラベルの平均逆順位と正規化割引累積利得(上位 10 件)を併記する。

**対比量が閾値を超えうる状態と、仮説との整合**: $\Delta_B > 2\sigma_B$ となりうる状態は、(i)記事単位の定義の学習が、記事単位のラベルの評価で優れる(仮説どおり)、(ii)R2 の学習が進んでいない(P1 で検出)、
(iii)R1 の値が上限に近い(P2 で検出)。入力の形式は両条件で同じで、形式の不一致による不利は、どちらの条件にもない。

**規則で決めた学習率による偏り**: 実験 A の「規則で学習率を決める段階」のとおり、段階 4 以降は R1 の学習率が最適からずれうる。実験 B では、R1 の値が下がり、対比量は **負の向き(仮説に不利)** に偏る。

**前提条件**: P0(R1・R2)、P1(R1・R2)、P2($R(\mathrm{R1})$ の記事単位のラベルが $[0.02, 0.95]$)。

**実験 A と B の判定の組み合わせの読み方**(事前の宣言。部分ごとの判定は別々に記録し、組み合わせはここで宣言した読み方だけに使う。どちらかが前提不成立のときは、組み合わせの読み方を適用しない)。
入力の形式・索引・query が同じなので、両方が支持のとき、それは入力の形式の一致の効果ではなく、関連性の定義(正例と in-batch の負例への一貫した当てはめ)の効果と読む:

| A \ B | 支持(R1 > R2) | 判定不能 | 反証(R2 > R1) |
|---|---|---|---|
| 支持(R2 > R1 で passage 単位のラベル) | **交差**: 各定義の検索器が、その定義のラベルの評価で優れる。定義が学ぶものを決める | R2 の利点は passage 単位のラベルで示されたが、R1 の記事単位のラベルでの優位は示されない。交差とは言えない | R2 が両方のラベルで優れる。定義の交差ではなく、passage 単位の定義の全面的な優位 |
| 判定不能 | R1 の利点は記事単位のラベルで示されたが、R2 の passage 単位のラベルでの優位は示されない。交差とは言えない | どちらのラベルでも差はノイズ床以内。この学習量・この設定で、定義の違いは検出されない | R2 は記事単位のラベルでも優れ、passage 単位のラベルでは差がない |
| 反証(R1 > R2) | R1 が両方のラベルで優れる。定義の交差ではなく、記事単位の定義の全面的な優位 | R1 は passage 単位のラベルで優れ、記事単位のラベルでは差がない | **逆の交差**: 各定義の検索器が、その定義と反対のラベルの評価で優れる |

(段階 4 以降に R1 の学習率を規則で決めた場合は、実験 A の「R1 の学習率が最適からずれて R2 が有利に見える」偏りと、実験 B の「R1 が不利に見える」偏りを考慮して読む。)

#### 実験 C: 2 段階の検索の効果(アプリの構成の判断)

**検証すること**: passage 単位の評価で、R2 の上位 $K = 50$ 件を cross-encoder で並べ替えた Recall@10 が、R2 だけ(並べ替えなし)より高いか。

**条件**: R2(並べ替えなし)と、R2 + cross-encoder(同じシードの R2 の上位 50 件を、同じシードの cross-encoder で並べ替える)。cross-encoder は、上の学習の組(第 1 段 R2 の上位 50 件から正例・負例を選ぶ)で学習する。

**対比量**:

$$
\Delta_C = \frac{1}{n} \sum_{s} \left( R_s(\mathrm{R3}) - R_s(\mathrm{R2}) \right)
$$

(正を期待。$n$ は実験 C のシード数、$R_s(\mathrm{R3})$ は R2 の上位 $K$ 件を cross-encoder で並べ替えた構成 R3 の Recall@10。)標準偏差 $\sigma_C$ は実験 A と同じ構成で、評価に使う query は 6.1 節の「cross-encoder の評価に使う query」(本番は評価用のすべて)。

**判定基準**: $\Delta_C > 2\sigma_C$ なら支持、$\Delta_C < -2\sigma_C$ なら反証、それ以外は判定不能。

**解釈の注意(学習量との交絡。実用の問い)**: 「R2 + cross-encoder」は、R2 に加えて cross-encoder の追加の学習($T_2$ ステップ、1 シードあたり 8 query × 8 候補 × $T_2$ 組の順伝播)を使う。
**同じ学習量での優劣を比べる実験ではない**。問いは「アプリで並べ替えを足す価値があるか」という実用の問いで、費用(順伝播の回数 $K$、観察 D)と合わせて読む。
並べ替えで到達できる Recall@10 の上限は coverage(上位 50 件に正例が入る割合)で、$\Delta_C$ の最大は coverage − $R(\mathrm{R2})$ である。

**介入の直接の作用点と、対比量までの距離**: 介入(並べ替え)が直接作用するのは、**上位 50 件の候補の順序** である。Recall@10 は、その順序を上位 10 件で二値化した量で、作用点に近い。
診断量として、並べ替え後の平均逆順位と正規化割引累積利得(上位 10 件)、候補 50 件のスコアの標準偏差の中央値(出力が定数に崩れていないかの検出)を併記する(判定には使わない)。

**対比量が閾値を超えうる状態と、仮説との整合**: $\Delta_C > 2\sigma_C$ となりうるのは、(i)cross-encoder が上位 50 件の中で正例を上位に並べ替える(仮説どおり)、(ii)R2 の学習が崩れている(P1 で検出)、
(iii)パイロットで選ばれた $T_2$ が実験 C に有利な向きの選択である(上の「パイロットの分岐」の偏り)。(iv)R2 だけの値が coverage に近く差が出ない(P2・P3 で検出。支持ではなく判定不能・前提不成立の側)。

**前提条件**: P0(R2・cross-encoder)、P1(R2 は実験 A・B の P1、cross-encoder は P1)、P2($R(\mathrm{R2}) \le$ coverage − 0.05)、P3(coverage ≥ 0.35 かつ coverage − $R(\mathrm{R2})$ ≥ 0.10)。

**規則で決めた学習率による偏り**: 段階 5 以降の cross-encoder の学習率は中心の値で、最適からずれると $R(\mathrm{R3})$ が下がり、$\Delta_C$ は負の向き(仮説に不利)に偏る(上の「学習率の較正」)。

**診断量(判定なし)**: 並べ替え後の平均逆順位・正規化割引累積利得、coverage、cross-encoder の最後の区間の訓練損失、スコアの標準偏差の中央値、学習に使える query の数と割合、
学習用の query・検証用の query・評価用の query での coverage(第 1 段 R2 が学習用の passage を見ていることによる、学習時と推論時の分布の違いの検出)。

#### 観察 D・E(判定基準を設けない)

- **観察 D(2 段階の検索の費用と指標)**: R2 のシード 0 の埋め込みの部分集合(件数 $N_{\mathrm{obs}}$)で、第 1 段を全探索と HNSW(025 の実装。$M = 5$、$\mathrm{efConstruction} = 100$。探索の幅は、全探索の上位 100 件との一致率が 0.9 に初めて届いた値)にし、
  cross-encoder(シード 0)で上位 $K = 10, 20, 50, 100$ 件を並べ替える。1 query あたりの第 1 段の距離計算の回数、第 2 段の順伝播の回数 $K$、並べ替えなしと並べ替えた後の Recall@10 と coverage を示す。
  関連性は部分集合の中の passage 単位(正解は query の passage だけ)で、部分集合の query の数は 6.1 節の水準で決まる(本番 600)。この表の Recall@10 の絶対値は、実験 A・C(コーパス全体)と比べない。
- **観察 E(検索結果の例)**: 同じ評価用の query に対する R1 と R2(シード 0)の上位の結果を示す。選び方は、評価用の query の並び順で、R1 と R2 の一方だけが正例を上位 10 件に入れた query の先頭から、
  R2 だけが入れたものを 2 件、R1 だけが入れたものを 1 件(結果を見て選ばない決定的な規則。存在しなければ、ある分だけ)。

#### 診断量: 024 と同じ形式の評価と R1 の再現(判定なし)

024 と同じ形式の記事単位の評価(024 の評価用の索引と query、固定長の passage と query)で、R1・R2・024 の Hub のモデル(C5 のシード 0)の Recall@10 を測る。
本トピックの R1 は、024 の C5 と入力の形式(query は抜いた文、passage は文を除いた形)・正例の作り方・バッチ(同じ記事の組を含む)が違うので、**ビット単位の一致も、024 の値との一致も期待しない**。
R1 のシード 0 の値と 024 の Hub の重みの値の差を、024 の C5 の記事単位の Recall@10 のシード間の標準偏差の目安(0.002〜0.009)と並べて印字する。判定には使わない。

#### 検出力の事前確認

本番の前に、基準の条件の値のシード間のばらつきをパイロットで測ることは、このセッションの時間の都合でできない(別の測定を設けない)。代わりに、同じ構成の 024 の本番(T4)の値から見積もる。

- 024 の C2・C5(記事単位の評価、3,244 query、5 シード): $R$ のシード間の標準偏差は 0.0023(C2)。実験 A の対比量の標準偏差は 0.0075(シード間の項 0.0035、ブートストラップの項 0.0066)、実験 B は 0.0049(0.0025、0.0042)、実験 D は 0.0074(0.0029、0.0068)で、
  ブートストラップの項(評価の query が有限であることによる変動)が支配的だった。閾値($2\sigma$)は 0.010〜0.018 だった。
- 本トピックの対比量の標準偏差の見込みは、同程度の $\sigma \approx 0.005$〜$0.010$(閾値 $2\sigma$ は **約 0.01〜0.02**)で、passage 単位のラベルは正例が 1 つの passage に限られる分、query ごとの hit の分散は大きくなりうる
  (評価の変動の下限は 024 の見積もり、上限は 2 条件の評価の変動が独立で打ち消しがまったく起きない場合の $\sqrt{2}$ 倍で、閾値は最大で約 0.03)。
- **想定される効果の大きさに比べた判断**: 想定される効果の大きさは測っていない(対比量の向きの情報を含むため)。検出できる最小の効果量(約 0.01〜0.03)が、Recall@10 の余地(基準の条件の値から天井まで)に比べて小さい見込みなので、**設計は見直さず本番に進む**。
  第 1 段のシード数を中(3)や低(2)に削る段階では、シード間の項が増えるが、支配的なブートストラップの項は変わらないので、閾値はほとんど増えない見込みである。

#### 仮説と判定の一覧

| 実験 | 仮説(対比量の向き) | 対比量 | 基準の条件 | 判定を除く診断量 |
|---|---|---|---|---|
| A | $\Delta_A > 0$ | $R(\mathrm{R2}) - R(\mathrm{R1})$(passage 単位のラベル) | R1 | 同じ記事の passage の割合、取り違えの割合、平均逆順位 |
| B | $\Delta_B > 0$ | $R(\mathrm{R1}) - R(\mathrm{R2})$(記事単位のラベル) | R2 | 平均逆順位、正規化割引累積利得 |
| C | $\Delta_C > 0$ | $R(\mathrm{R3}) - R(\mathrm{R2})$(passage 単位の評価) | R2 | 並べ替え後の平均逆順位、coverage、スコアの標準偏差 |

#### Hugging Face Hub への公開の条件(判定ではない。6.14 節)

公開の条件は、判定基準とは別に、アップロードするモデルが「使わない場合より悪くならない」ことの確認である。読み込み直した重みを評価して確かめ、満たさない場合はアップロードせず、その旨と値を印字する。

- **R2 のシード 0**(`kojikojiprg/ai-theories-text-embedding-en`の`inverse-cloze-task`ブランチ): 読み込み直した重みの passage 単位の評価での Recall@10 が、R1 のシード 0 の passage 単位の評価での Recall@10 を上回ること。
- **cross-encoder のシード 0**(`kojikojiprg/ai-theories-cross-encoder-en`): 読み込み直した重みで、シード 0 の R2 の上位 50 件を並べ替えた Recall@10 が、並べ替えなしの Recall@10 を上回ること(評価用の query)。

#### 設計の改訂の記録(本番の前)

次の 4 つの改訂を、本番の実行の前に入れた。いずれも、観測された結果の向きに依らない一般論(比較の妥当性、学習の信号の一貫性、入力の切り詰め、見積もりの余裕)による。
動作確認(縮小スケールの実行)の数値は、判定基準の根拠に使っていない。実験 A・B・C の対比量の定義、閾値の導出式(標準偏差の 2 倍)、期待する向きは変えていない。変えたのは、条件の作り方と評価のラベルの定義である。

| | 旧設計 | 新設計 | 理由 |
|---|---|---|---|
| 1. 入力の形式 | R1 は固定長の窓を query・正例にし、R2 は抜いた文と 1 文を除いた文脈を見る。実験 A は R2 の形式、実験 B は 024 の形式(固定長)で評価した | R1・R2 とも、query は窓から抜いた 1 文(文を残す確率 $r = 0.1$ も共通)。正例は、R2 が query を抜いた窓の文脈、R1 が同じ記事の別の窓(同じバッチのもう一方の組)の、文を除いた形。評価は、実験 A・B とも同じ索引(すべての passage を 1 文を除いた形にしたもの)・同じ query で、関連のラベルだけを変える(A: 正解は query の passage だけ、B: 正解は同じ記事の他の passage)。024 と同じ形式の記事単位の評価は診断量にする | 旧設計では、各モデルが学習で見た形式の評価で勝つだけで、A・B の両方が支持になりうる。関連性の定義の効果と、入力の形式の一致の効果を分けられない |
| 2. in-batch negatives | R2 も 1 つのバッチに同じ記事の組を入れない。R1 と R2 の負例はどちらも別の記事の passage だけ | 1 バッチ 64 組を 32 記事 × 2 組にし、同じ記事の組を意図的に同じバッチに入れる。R1 は同じ記事の組の passage を負例から除き(負例 62 個)、R2 は負例として数える(63 個)。「関連性の定義を、正例と in-batch の負例の両方に一貫して当てはめる」ことが、実験 A・B の介入 | 旧設計では、passage 単位の定義で同じ記事の他の passage が困難な負例になる(3.3 節)という本筋が、学習に現れない |
| 3. 抜く文の選び方 | 条件を満たす文(長さ 4〜96 トークン)から一様に選び、32 トークンを超える文は query の先頭で切る | query の長さの上限(32 トークン)以内に収まる文があればその中から選び、収まる文がない窓だけ長い文から選ぶ。学習の組と評価の索引の両方に適用する | 旧設計では、query の約 3 割が切り詰められ、文の後半の情報が query から失われる。切り詰めは学習と評価で同じ規則で行われる |
| 4. 見積もりの安全係数 | 学習の時間の見積もりは、本番の冒頭で測った 1 ステップの時間をそのまま使う | 学習(第 1 段・cross-encoder)の時間の見積もりに 1.1 倍の安全係数を掛ける | 025 の再実行に向けたパイロットで、T4 の実際の 1 ステップ(80〜81 ms)が本番の冒頭の計測値(75 ms)より約 7% 長かった。計測値の外れによる予算の超過を避ける |

**改訂に伴う変更**: 学習率の較正の指標は、新しい設計の同じ索引・同じ query の 2 つのラベルの Recall@10 の平均にした(改訂 1)。R1 は 024 の C5 と同じ入力の形式・バッチではなくなったので、R1 の再現の確認(024 の Hub のモデルとの比較)は、
024 と同じ形式の記事単位の評価(診断量)で行い、ビット単位の一致を期待しない。実験 B の評価の集合は、024 の評価用の索引(固定長)から、実験 A と同じ索引・同じ query に変わった。

#### 停止した実行の記録(学習の組の生成の不具合)

最初の本番の実行(Colab T4、コミット `90b3dde`、2026-10-14T02:52:33 UTC)は、6.5 節の本番の学習の最初(R1・シード 0 の学習の組の生成)で、次のエラーにより停止した。**学習と評価の判定は計算していない。**

```
RuntimeError: 記事 8646 から、重ならない対象の窓を 2 個取れない
(src/data/inverse_cloze.py の sample_cloze_pair_batches)
```

- **止まった箇所**: 6.5 節のあとの本番の学習で、シード 0 の学習の組(1024 ステップ × 32 記事)を作る関数。6.2 節(計測)・6.3 節(パイロット)・6.4 節(計画の選択。計画 0)・6.5 節の較正までは完了していた。
- **原因**: 学習の組は、1 つの記事から重ならない対象の窓を 2 個引く。記事を選ぶ重みは「トークン数が窓 2 個ぶん(256)以上」だけで決めており、**対象の窓が実際に重ならずに 2 個取れるか** は確かめていなかった。学習用の記事の中の添字 8646 の記事(277 トークン)は、対象の窓(文が 2 つ以上あり、抜ける長さの文を持つ窓)の開始位置が 0〜26 の範囲にしかなく、窓の長さ 128 以上離れた 2 個の窓が存在しなかった。引き直しの回数の上限による見つけ損ないではなく、組み合わせが存在しない記事だった。重みが正の 8,338 記事のうち、このような記事はこの 1 件だけである(本番と同じトークン列で、全記事を厳密に調べた)。この記事が 1024 ステップのどこかで選ばれると、必ず停止する。スモークテストは学習用の記事を 80 件に絞り、ステップ数も小さいので、この記事を引かなかった。
- **修正の内容**(学習の組の生成に限る):
  - `pairable_articles()` を追加した。対象の窓の開始位置の最小値と最大値を記事の両端から探し、「最大値 − 最小値 ≥ 窓の長さ」を組が取れる必要十分条件として、記事ごとに厳密に判定する。取れない記事は、抽出の重みを 0 にする(5.3 節で、除いた記事の数を印字する)。
  - 対象の窓がまれな記事で、記事のすべての窓から組を選ぶ処理を、引き直しをやめて、重ならない相手を持つ窓から一様に選ぶ方式に改めた。組が取れない記事を選んでしまった場合は、原因を示して停止する。
  - 記事のすべての窓の走査の結果を、呼び出しをまたいで使い回せるようにした(出力は変わらず、時間だけが変わる)。
  - 本番の冒頭で計画を選んだ直後(6.4.1 節)に、較正のシードと本番のシードの学習の組(1024 ステップ)の生成を学習の前に済ませて確かめ、その時間を見積もりに含める。
  - `REQUIRED_ANCESTOR_COMMIT` に `90b3dde` を指定し、実行するコミットがその子孫(自身を除く)であることを確かめる。
- **停止の前に完了していた処理と、その主な値**(Colab T4、停止した実行の出力による。検証用の記事・1 シードの値で、判定ではない):
  - 6.2 節(計測): 1 ステップの時間は R1 76.8 ms、R2 76.6 ms、cross-encoder 76.8 ms。cross-encoder の推論は 1 組あたり 0.329 ms。データの準備 380.3 秒、スケーリングの計測 170.2 秒。学習前の 008 の Recall@10 は、検証用が passage 単位 0.3494・記事単位 0.5266、評価用が passage 単位 0.0762・記事単位 0.1054。
  - 6.3 節(パイロット): 第 1 段(R2、シード 90、$T_1 = 1024$、学習率 $4.8 \times 10^{-4}$)は、最後の区間の訓練損失 1.923(同じ組での学習前のモデルの損失 3.497)、検証用の Recall@10 は passage 単位 0.5246・記事単位 0.7028。R2 の上位 50 件に正例が入る割合は、学習用 0.2328・検証用 0.2392・評価用 0.2290。R2 だけの Recall@10 は、検証用 0.1299・評価用 0.1246。学習に使える query は 20,898 個(学習用の 23.3%)。cross-encoder($T_2 = 2048$、学習率 $2.4 \times 10^{-4}$、シード 90)は、最後の区間の訓練損失 0.4035(学習前 2.0982)、検証用の並べ替えの Recall@10 0.2343(学習前 0.0463、増分 +0.1880、標準偏差 0.0162)。3 つの条件がすべて成立し、$T_2 = 2048$ が選ばれた。パイロット全体は 5.5 分。
  - 6.4 節(計画の選択): 経過時間 15.8 分(パイロットを含む)で、計画 0(削らない)が選ばれた。見積もりは残り 74.4 分、終了の見込み 90.2 分。
  - 6.5 節(較正): 較正の指標は、passage 単位と記事単位の検証用の Recall@10 の平均。R2 は、学習率 $1.2 \times 10^{-4}$・$4.8 \times 10^{-4}$・$1.92 \times 10^{-3}$ で 0.5556・0.6137・0.5576(passage 単位 0.4557・0.5246・0.4636、記事単位 0.6555・0.7028・0.6516)となり、$4.8 \times 10^{-4}$(内点)を選んだ。R1 は、同じ学習率で 0.5344・0.5719・0.4562(passage 単位 0.4262・0.4587・0.3258、記事単位 0.6427・0.685・0.5866)となり、$4.8 \times 10^{-4}$(内点)を選んだ。cross-encoder($T_2 = 2048$)は、学習率 $6 \times 10^{-5}$・$2.4 \times 10^{-4}$・$9.6 \times 10^{-4}$・$3.84 \times 10^{-3}$(上端を拡張)で、検証用の並べ替えの Recall@10 が 0.2224・0.2343・0.2382・0.2224 となり、$9.6 \times 10^{-4}$(内点)を選んだ。較正は 15.7 分。止まったのは、その直後の本番の学習の最初(R1・シード 0 の学習の組の生成)。
- **条件間の大小が読み取れる値**: 較正とパイロットの出力に、次の 2 点が表示されていた。どちらも検証用の記事・1 シードの値で、判定ではない。判定は計算していない。
  - 較正では、passage 単位・記事単位のどちらの検証用の Recall@10 でも、選ばれた学習率の R2 が R1 を上回っていた(passage 単位 0.5246 と 0.4587、記事単位 0.7028 と 0.685)。記事単位は、実験 B の仮説と逆の向きである。
  - パイロットと較正では、cross-encoder の並べ替えの Recall@10 が、R2 だけの値(検証用 0.1299)を上回っていた(0.2343 など)。
- **これらの値の扱い**: これらの値を、修正や設計の変更に使っていない。判定基準・前提条件・水準・実行計画は変えていない。停止した実行の出力そのものは、この再実行の出力で上書きされる。再実行では学習の組の抽出が変わる(同じシードでの記事の集合の一致は 2〜4%)ので、再実行の値は、較正・パイロットを含め、停止した実行の値と一致しない。
- **6.2 節の計測**: 学習の組の生成時間は(停止した実行の Colab で)64・128・256 ステップで 0.062・0.116・1.23 秒、べき指数 2.155 だった。ステップ数に比例させた見積もりを使っているため、原因を調べた。時間は引き直しの繰り返しではなく、記事のすべての窓から選ぶ処理(長い記事の全窓の走査)の回数に左右されていた(同じ関数の引き直しの回数は、組あたり平均 1.2 回で、ステップ数に比例する)。走査の結果を使い回し、最初の呼び出しを別に測ることで、比例の見積もりと実測を揃えた(6.2 節)。
- **再実行で変えていないもの**: 実験 A・B・C の判定基準(対比量・閾値の導出式・期待する向き)、前提条件、水準、実行計画の規則と段階、学習率の格子、シード数は変えていない。変えたのは学習の組の生成だけである。この修正で、記事を選ぶ重みの正規化がわずかに変わるため、同じシードでも選ばれる記事の並びは修正前と異なる(修正前の実行は、判定に使える結果を出していない)。

### 6.2 スケーリングの計測と外挿(1 セッションの見積もり)

本番でデータ量や回数が大きくなる処理を、実行するデバイスの上で計測する。

- **学習**(1 ステップの処理が一定の反復): 第 1 段(R1・R2)と cross-encoder のそれぞれで、ウォームアップ 8 ステップの後、8・16・32 ステップの時間を測る。本番と同じ関数・同じ精度・同じバッチで測る。
  見積もりは **定常状態の 1 ステップの時間(32 ステップの計測から)をステップ数に比例させる**。べき指数 $b$ も推定し、べき乗則の外挿値と比例の値の差を印字する。
- **評価**: 検証用の評価(較正の 1 回ごと)、本番の評価(評価用の集合を、passage 単位のコーパス全体の索引と記事単位の索引で。R2 は第 1 段の上位 $K$ 件の採掘を含む)の時間を測る。評価用のコーパス全体の符号化は、最大の計測点が本番の量そのものである(外挿は不要)。
  あわせて、学習前の 008(R1・R2 と同じ構成)の検証用・評価用の Recall@10 を測る(P1(b) の基準と、学習前後の診断量。学習は行わない)。
- **cross-encoder の推論**: 候補の組の順伝播 1 組あたりの時間を測る(検証用・評価用の query の並べ替えの見積もりに使う)。
- **学習前のモデルの損失の測定**(P1(a) の分母): 32 ステップ分の時間を測り、学習 1 回ごとに最後の区間のステップ数だけ加える。
- **学習の組の生成**(第 1 段の R1・R2 の組、cross-encoder の組): ステップ数 64・128・256 の 3 点で計測し、べき指数を推定する。見積もりは 1 ステップあたりの時間を比例させる。
- **HNSW**(観察 D): 索引の件数 3 点($1024, 2048, 4096$)で構築の時間を測り、べき乗則で観察 D の件数(本番 16384・8192)まで外挿する。
- **参考**: バッチサイズを半分にしたときの第 1 段の 1 ステップの時間を測り、時間がバッチサイズに比例するか(固定費が支配的でないか)を印字する。
- **データの準備**: 5.3 節で、段階ごとの時間とピークのメモリを印字した。本番の冒頭で 1 回だけ行い(以降は全条件・全シードで共有する)、6.4 節の経過時間に含まれる。
- **判定・ブートストラップ・図・アップロードの準備**: 規模が小さく、合計しても数十秒〜数分である。固定の余裕(300 秒)に含める。


```python
_t0_scaling = time.time()
SCALING_STEP_COUNTS = (8, 16, 32)  # 水準によらず同じ
SCALING_WARMUP_STEPS = 8
SCHEDULE_STEP_COUNTS = (64, 128, 256)


def measure_first_stage(condition: str, step_counts, warmup_steps: int, batch_size: int = BATCH_SIZE) -> list[float]:
    # 本番と同じ学習の関数・精度で、定常状態の時間を測る(評価は行わない)
    total = max(warmup_steps + sum(step_counts), max(step_counts), warmup_steps)
    model = build_embedding_model(TIMING_SEED_INDEX).to(device)
    arrays = pair_arrays(condition, TIMING_SEED_INDEX, total)
    articles = pair_batches(TIMING_SEED_INDEX, total).articles

    def make(n: int) -> VariableBatches:
        # 同じ記事の 2 組はバッチの中で隣り合うので、先頭の batch_size 組を取れば組の対応は保たれる
        return VariableBatches(*(a[:n, :batch_size] for a in arrays), device, article_ids=articles[:n, :batch_size], mask_same_article=condition == "R1")

    def run(batches: VariableBatches) -> None:
        train_text_embedding_from_batches(
            model, batches, peak_learning_rate=FIRST_STAGE_LEARNING_RATE_CENTER[condition], warmup_steps=1,
            min_learning_rate=FIRST_STAGE_LEARNING_RATE_CENTER[condition] * MIN_LEARNING_RATE_RATIO, weight_decay=WEIGHT_DECAY,
            gradient_clip_threshold=GRADIENT_CLIP_THRESHOLD, temperature=TEMPERATURE, matryoshka_dimensions=MATRYOSHKA_DIMENSIONS,
            use_fp16_autocast=USE_FP16_AUTOCAST, init_loss_scale=INIT_LOSS_SCALE, loss_scale_growth_interval=LOSS_SCALE_GROWTH_INTERVAL,
        )

    warm = make(warmup_steps)
    timed_call(lambda: run(warm))
    times = []
    for n in step_counts:
        batches = make(n)
        times.append(timed_call(lambda b=batches: run(b)))
    del model
    empty_device_cache()
    return times


# 学習(R1・R2): 1 ステップの時間
STEP_SECONDS, SCALING_ROWS = {}, {}
for _condition in FIRST_STAGE_CONDITIONS:
    _times = measure_first_stage(_condition, SCALING_STEP_COUNTS, SCALING_WARMUP_STEPS)
    STEP_SECONDS[_condition] = _times[-1] / SCALING_STEP_COUNTS[-1]
    _fit = fit_power_law_exponent(SCALING_STEP_COUNTS, _times)
    _power = _times[-1] * (FIRST_STAGE_STEPS / SCALING_STEP_COUNTS[-1]) ** _fit.exponent
    SCALING_ROWS[_condition] = {"exponent": _fit.exponent, "power_minus_proportional": _power - STEP_SECONDS[_condition] * FIRST_STAGE_STEPS}
    print(
        f"{_condition}: {dict(zip(SCALING_STEP_COUNTS, rounded(_times, 2), strict=True))} 秒、1 ステップ {STEP_SECONDS[_condition] * 1000:.1f} ms、べき指数 b = {_fit.exponent:.3f}、"
        f"T_1 = {FIRST_STAGE_STEPS} でのべき乗則の外挿 - 比例の値 = {SCALING_ROWS[_condition]['power_minus_proportional']:+.1f} 秒"
    )
_half = measure_first_stage("R1", (SCALING_STEP_COUNTS[1],), SCALING_WARMUP_STEPS, batch_size=BATCH_SIZE // 2)[0] / SCALING_STEP_COUNTS[1]
print(
    f"参考 R1: バッチ {BATCH_SIZE // 2} の 1 ステップ {_half * 1000:.1f} ms、バッチ {BATCH_SIZE} は {STEP_SECONDS['R1'] * 1000:.1f} ms"
    f"(比 {STEP_SECONDS['R1'] / _half:.2f}。2 に近ければ時間はバッチサイズに比例し、1 に近ければ固定費が支配的)"
)

# 学習の組の生成(シードごとに 1 回。R1 と R2 で共通)
# 最初の呼び出しは、まれな記事(対象の窓が少ない記事)の全窓の走査を含む。走査の結果は使い回すので、初回を別に測り、その後の 3 点は走査を含まない生成の時間を測る
PAIR_BATCHES.clear()
ELIGIBLE_CACHE.clear()
_start = time.time()
_warmup_exhaustive = pair_batches(TIMING_SEED_INDEX, SCHEDULE_STEP_COUNTS[-1]).num_exhaustive
FIRST_CALL_SECONDS = time.time() - _start
_times, _exhaustive_counts = [], []
for _steps in SCHEDULE_STEP_COUNTS:
    PAIR_BATCHES.clear()
    _start = time.time()
    _exhaustive_counts.append(pair_batches(TIMING_SEED_INDEX, _steps).num_exhaustive)
    _times.append(time.time() - _start)
_fit = fit_power_law_exponent(SCHEDULE_STEP_COUNTS, _times)
BATCH_SECONDS_PER_STEP = {c: _times[-1] / SCHEDULE_STEP_COUNTS[-1] for c in FIRST_STAGE_CONDITIONS}
FIRST_SCAN_SECONDS = max(FIRST_CALL_SECONDS - _times[-1], 0.0)  # 最初の走査にかかる時間(シードをまたいで一部は使い回される)
print(f"学習の組の生成: 最初の呼び出し {FIRST_CALL_SECONDS:.3f} 秒(記事のすべての窓から選んだ組 {_warmup_exhaustive} 個の走査を含む。走査分 {FIRST_SCAN_SECONDS:.3f} 秒)。走査の結果を使い回した後の生成(記事のすべての窓から選んだ組の数 {_exhaustive_counts}): {dict(zip(SCHEDULE_STEP_COUNTS, rounded(_times, 3), strict=True))} 秒、べき指数 b = {_fit.exponent:.3f}、1 ステップあたり {BATCH_SECONDS_PER_STEP['R1'] * 1000:.2f} ms(見積もりは比例)")
PAIR_BATCHES.clear()

# 学習前のモデルの損失の測定
_probe = build_embedding_model(TIMING_SEED_INDEX).to(device)
_timing_batches = first_stage_batches("R1", TIMING_SEED_INDEX, 32)
compute_pair_losses_from_batches(_probe, _timing_batches, range(32), TEMPERATURE, MATRYOSHKA_DIMENSIONS, USE_FP16_AUTOCAST)  # ウォームアップ
INITIAL_LOSS_SECONDS_PER_STEP = timed_call(lambda: compute_pair_losses_from_batches(_probe, _timing_batches, range(32), TEMPERATURE, MATRYOSHKA_DIMENSIONS, USE_FP16_AUTOCAST)) / 32
print(f"学習前のモデルの損失の測定: 1 ステップあたり {INITIAL_LOSS_SECONDS_PER_STEP * 1000:.1f} ms(学習 1 回につき最後の区間のステップ数だけ。T_1 = {FIRST_STAGE_STEPS} で {INITIAL_LOSS_SECONDS_PER_STEP * final_loss_window(FIRST_STAGE_STEPS):.1f} 秒)")
del _timing_batches

# 評価の時間と、学習前の 008 の値(R1・R2 と同じ構成、学習しない)
evaluate_calibration(_probe)  # ウォームアップ
CALIBRATION_EVALUATION_SECONDS = timed_call(lambda: evaluate_calibration(_probe))
UNTRAINED_VALIDATION_FIRST_STAGE = evaluate_calibration(_probe)
_start = time.time()
UNTRAINED_MAIN = evaluate_main(_probe, keep_embeddings=True)
sync_device()
MAIN_EVALUATION_SECONDS = time.time() - _start
_untrained_contexts = UNTRAINED_MAIN.pop("contexts")
UNTRAINED_MAIN.pop("queries")
MINING_SECONDS = timed_call(lambda: mine_candidates(_probe, _untrained_contexts))
print(
    f"評価の時間: 検証用 {CALIBRATION_EVALUATION_SECONDS:.2f} 秒、本番の評価(passage 単位のコーパス全体 + 記事単位)の合計 {MAIN_EVALUATION_SECONDS:.2f} 秒、R2 の第 1 段の上位 {RERANK_CANDIDATES} 件の採掘 {MINING_SECONDS:.2f} 秒"
    f"(学習用 {len(TRAIN_ENTRIES):,}・検証用 {len(VALIDATION_ENTRIES):,}・評価用 {len(EVALUATION_ENTRIES):,} query)"
)
print(
    f"学習前の 008(平均プール・因果マスク)の Recall@10: 検証用 passage 単位 {UNTRAINED_VALIDATION_FIRST_STAGE['passage_hits'].mean():.4f}・記事単位 {UNTRAINED_VALIDATION_FIRST_STAGE['article_hits'].mean():.4f}"
    f"(P1(b) の基準)、評価用 passage 単位 {UNTRAINED_MAIN['passage_hits'].mean():.4f}・記事単位 {UNTRAINED_MAIN['article_hits'].mean():.4f}。"
    f"ランダムな順位の期待値 passage 単位 {RANDOM_RECALL_PASSAGE:.6f}"
)
# コーパス全体の符号化の時間の外挿(索引の件数に比例するかを、1/4・1/2・全部で確かめる)
_sizes = [NUM_INDEX // 4, NUM_INDEX // 2, NUM_INDEX]
_encode_times = [timed_call(lambda n=n: embed_contexts(_probe, np.arange(n))) for n in _sizes]
_encode_fit = fit_power_law_exponent(_sizes, _encode_times)
print(f"索引の符号化: {dict(zip(_sizes, rounded(_encode_times, 3), strict=True))} 秒(べき指数 {_encode_fit.exponent:.3f}。最大の計測点が本番の量そのもの)")
del _untrained_contexts

# cross-encoder: 1 ステップの時間、推論の 1 組あたりの時間、学習の組の生成
_rng = np.random.default_rng(0)
_fake_mined = np.concatenate([TRAIN_ENTRIES[:, None], _rng.integers(0, NUM_INDEX, size=(len(TRAIN_ENTRIES), RERANK_CANDIDATES - 1))], axis=1)
_total_cross_encoder = SCALING_WARMUP_STEPS + sum(SCALING_STEP_COUNTS)
_cross_encoder_schedule_times = []
for _steps in SCHEDULE_STEP_COUNTS:
    _start = time.time()
    _cross_encoder_schedule = sample_cloze_reranking_schedule(TRAIN_ENTRIES, _fake_mined, _steps, BATCH_QUERIES, NUM_NEGATIVES, seed=1)
    _cross_encoder_schedule_times.append(time.time() - _start)
_fit = fit_power_law_exponent(SCHEDULE_STEP_COUNTS, _cross_encoder_schedule_times)
CROSS_ENCODER_SCHEDULE_SECONDS_PER_STEP = _cross_encoder_schedule_times[-1] / SCHEDULE_STEP_COUNTS[-1]
print(f"cross-encoder の学習の組の生成: {dict(zip(SCHEDULE_STEP_COUNTS, rounded(_cross_encoder_schedule_times, 3), strict=True))} 秒、べき指数 b = {_fit.exponent:.3f}、1 ステップあたり {CROSS_ENCODER_SCHEDULE_SECONDS_PER_STEP * 1000:.2f} ms(見積もりは比例)")
_cross_encoder_full = sample_cloze_reranking_schedule(TRAIN_ENTRIES, _fake_mined, _total_cross_encoder, BATCH_QUERIES, NUM_NEGATIVES, seed=2)


def measure_cross_encoder(step_counts, warmup_steps: int) -> list[float]:
    model = build_cross_encoder(TIMING_SEED_INDEX).to(device)

    def run(n: int) -> None:
        from src.training.reranking_variable import ClozeRerankingSchedule

        part = ClozeRerankingSchedule(_cross_encoder_full.query_ids[:n], _cross_encoder_full.candidate_ids[:n], _cross_encoder_full.num_usable)
        train_reranker_variable(
            model, CROSS_ENCODER_TABLES, part, peak_learning_rate=CROSS_ENCODER_LEARNING_RATE_CENTER, warmup_steps=1, min_learning_rate=CROSS_ENCODER_LEARNING_RATE_CENTER * MIN_LEARNING_RATE_RATIO,
            weight_decay=WEIGHT_DECAY, gradient_clip_threshold=GRADIENT_CLIP_THRESHOLD, use_fp16_autocast=USE_FP16_AUTOCAST,
            init_loss_scale=INIT_LOSS_SCALE, loss_scale_growth_interval=LOSS_SCALE_GROWTH_INTERVAL,
        )

    timed_call(lambda: run(warmup_steps))
    times = [timed_call(lambda n=n: run(n)) for n in step_counts]
    del model
    empty_device_cache()
    return times


_times = measure_cross_encoder(SCALING_STEP_COUNTS, SCALING_WARMUP_STEPS)
CROSS_ENCODER_STEP_SECONDS = _times[-1] / SCALING_STEP_COUNTS[-1]
_fit = fit_power_law_exponent(SCALING_STEP_COUNTS, _times)
print(
    f"cross-encoder の学習: {dict(zip(SCALING_STEP_COUNTS, rounded(_times, 2), strict=True))} 秒、1 ステップ {CROSS_ENCODER_STEP_SECONDS * 1000:.1f} ms、べき指数 b = {_fit.exponent:.3f}、"
    f"T_2 = {max(CROSS_ENCODER_STEP_CANDIDATES)} でのべき乗則の外挿 - 比例の値 = {_times[-1] * (max(CROSS_ENCODER_STEP_CANDIDATES) / SCALING_STEP_COUNTS[-1]) ** _fit.exponent - CROSS_ENCODER_STEP_SECONDS * max(CROSS_ENCODER_STEP_CANDIDATES):+.1f} 秒"
)
_half_cross_encoder = measure_cross_encoder((SCALING_STEP_COUNTS[1],), SCALING_WARMUP_STEPS)[0] / SCALING_STEP_COUNTS[1]
print(f"参考 cross-encoder: 全ステップ {CROSS_ENCODER_STEP_SECONDS * 1000:.1f} ms(参考: 同じ設定で 16 ステップの平均 {_half_cross_encoder * 1000:.1f} ms)")
_probe_cross_encoder = build_cross_encoder(TIMING_SEED_INDEX).to(device)
_pair_queries = RERANK_EVALUATION_ENTRIES[:40]
_pair_candidates = np.tile(np.arange(RERANK_CANDIDATES), (len(_pair_queries), 1)) + 0 * _pair_queries[:, None]
_pair_candidates = _fake_mined[np.searchsorted(TRAIN_ENTRIES, TRAIN_ENTRIES[:40])]  # 実際の候補の長さ分布に近い組(学習用の passage)
evaluate_rerank(_probe_cross_encoder, TRAIN_ENTRIES[:4], _pair_candidates[:4])  # ウォームアップ
CROSS_ENCODER_PAIR_SECONDS = timed_call(lambda: evaluate_rerank(_probe_cross_encoder, TRAIN_ENTRIES[:40], _pair_candidates)) / (40 * RERANK_CANDIDATES)
print(f"cross-encoder の推論: 1 組あたり {CROSS_ENCODER_PAIR_SECONDS * 1000:.3f} ms(候補 {RERANK_CANDIDATES} 件 x query 40 個から)")
del _probe_cross_encoder, _probe, _cross_encoder_full, _fake_mined
empty_device_cache()

# HNSW(観察 D): 索引の件数 3 点で構築の時間を測り、べき乗則で観察 D の件数まで外挿する
_hnsw_vectors = UNTRAINED_EMBEDDING_SAMPLE if "UNTRAINED_EMBEDDING_SAMPLE" in globals() else None
_probe2 = build_embedding_model(TIMING_SEED_INDEX).to(device)
_hnsw_pool = embed_contexts(_probe2, np.arange(min(NUM_INDEX, max(LEVEL_SETTINGS["HNSW_TIMING_SIZES"])))).cpu().numpy()
del _probe2
empty_device_cache()
_hnsw_times, _hnsw_search_times = [], []
for _n in LEVEL_SETTINGS["HNSW_TIMING_SIZES"]:
    _start = time.time()
    _index = HierarchicalNavigableSmallWorld(_hnsw_pool[:_n], HNSW_MAX_CONNECTIONS, HNSW_CONSTRUCTION_WIDTH, seed=OBSERVATION_SEED)
    _index.add(_n)
    _hnsw_times.append(time.time() - _start)
    _start = time.time()
    _index.search_batch(_hnsw_pool[:20], 11, OBSERVATION_SEARCH_WIDTHS[-1])
    _hnsw_search_times.append((time.time() - _start) / 20)
_hnsw_fit = fit_power_law_exponent(LEVEL_SETTINGS["HNSW_TIMING_SIZES"], _hnsw_times)
HNSW_SEARCH_SECONDS_PER_QUERY = _hnsw_search_times[-1]
print(
    f"HNSW の構築: {dict(zip(LEVEL_SETTINGS['HNSW_TIMING_SIZES'], rounded(_hnsw_times, 2), strict=True))} 秒、べき指数 b = {_hnsw_fit.exponent:.3f}、"
    f"観察 D の件数への外挿: " + "、".join(f"N = {n}: {_hnsw_times[-1] * (n / LEVEL_SETTINGS['HNSW_TIMING_SIZES'][-1]) ** _hnsw_fit.exponent:.0f} 秒" for n in LEVEL_SETTINGS["OBSERVATION_N_CANDIDATES"])
    + f"。探索 {HNSW_SEARCH_SECONDS_PER_QUERY * 1000:.1f} ms / query(探索の幅 {OBSERVATION_SEARCH_WIDTHS[-1]})"
)
del _hnsw_pool, _index


def hnsw_build_seconds(n: int) -> float:
    return _hnsw_times[-1] * (n / LEVEL_SETTINGS["HNSW_TIMING_SIZES"][-1]) ** _hnsw_fit.exponent


print("データの準備(5.3 節、1 回だけ): " + "、".join(f"{k} {v:.1f} 秒" for k, v in DATA_SECONDS_BY_STAGE.items()) + f"。合計 {DATA_SECONDS:.1f} 秒(6.4 節の経過時間に含まれる)")
SCALING_SECONDS = time.time() - _t0_scaling
print(f"スケーリングの計測 {SCALING_SECONDS:.1f} 秒")
```

    R1: {8: 0.61, 16: 1.21, 32: 2.43} 秒、1 ステップ 76.0 ms、べき指数 b = 1.001、T_1 = 1024 でのべき乗則の外挿 - 比例の値 = +0.2 秒
    R2: {8: 0.61, 16: 1.21, 32: 2.41} 秒、1 ステップ 75.5 ms、べき指数 b = 0.994、T_1 = 1024 でのべき乗則の外挿 - 比例の値 = -1.6 秒
    参考 R1: バッチ 32 の 1 ステップ 49.7 ms、バッチ 64 は 76.0 ms(比 1.53。2 に近ければ時間はバッチサイズに比例し、1 に近ければ固定費が支配的)
    学習の組の生成: 最初の呼び出し 1.389 秒(記事のすべての窓から選んだ組 9 個の走査を含む。走査分 1.186 秒)。走査の結果を使い回した後の生成(記事のすべての窓から選んだ組の数 [2, 4, 9]): {64: 0.058, 128: 0.115, 256: 0.203} 秒、べき指数 b = 0.907、1 ステップあたり 0.79 ms(見積もりは比例)
    学習前のモデルの損失の測定: 1 ステップあたり 25.7 ms(学習 1 回につき最後の区間のステップ数だけ。T_1 = 1024 で 1.3 秒)
    評価の時間: 検証用 0.50 秒、本番の評価(passage 単位のコーパス全体 + 記事単位)の合計 38.66 秒、R2 の第 1 段の上位 50 件の採掘 7.84 秒(学習用 89,752・検証用 1,016・評価用 3,227 query)
    学習前の 008(平均プール・因果マスク)の Recall@10: 検証用 passage 単位 0.3494・記事単位 0.5266(P1(b) の基準)、評価用 passage 単位 0.0762・記事単位 0.1054。ランダムな順位の期待値 passage 単位 0.000106
    索引の符号化: {23498: 9.339, 46997: 18.674, 93995: 37.635} 秒(べき指数 1.005。最大の計測点が本番の量そのもの)
    cross-encoder の学習の組の生成: {64: 0.015, 128: 0.023, 256: 0.04} 秒、べき指数 b = 0.711、1 ステップあたり 0.16 ms(見積もりは比例)
    cross-encoder の学習: {8: 0.59, 16: 1.17, 32: 2.36} 秒、1 ステップ 73.6 ms、べき指数 b = 1.005、T_2 = 8192 でのべき乗則の外挿 - 比例の値 = +16.9 秒
    参考 cross-encoder: 全ステップ 73.6 ms(参考: 同じ設定で 16 ステップの平均 73.6 ms)
    cross-encoder の推論: 1 組あたり 0.311 ms(候補 50 件 x query 40 個から)
    HNSW の構築: {1024: 2.68, 2048: 5.9, 4096: 12.35} 秒、べき指数 b = 1.101、観察 D の件数への外挿: N = 16384: 57 秒、N = 8192: 26 秒。探索 5.0 ms / query(探索の幅 254)
    データの準備(5.3 節、1 回だけ): トークナイザの取得 2.1 秒、参照コーパスの取得 3.1 秒、コーパスのダウンロード 7.7 秒、コーパスの読み込み 8.2 秒、記事の境界の検証 0.0 秒、記事の符号化と検証 332.0 秒、分割 0.1 秒、コーパスと索引の構成 4.9 秒、組が取れる記事の判定 19.2 秒、学習用の記事の連結 19.2 秒。合計 379.1 秒(6.4 節の経過時間に含まれる)
    スケーリングの計測 160.5 秒




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/026_inverse_cloze_task_and_two_stage_retrieval.ipynb
