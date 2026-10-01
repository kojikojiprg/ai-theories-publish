---
title: "CLIP と対照学習 / CLIP and Contrastive Learning(実装・実験編 3/5)"
---

この記事は後編(実装・実験編 3/5)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/020_clip_contrastive_learning-practice-2)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/020_clip_contrastive_learning-practice-4)。

### 6.2 学習で見る事例数 $E$ の候補・学習率の格子・前提条件の閾値の決め方(パイロット)

**$E$ の候補**: $E$ は固定せず、候補 $\{2^{19}, 2^{18}\}$(公比 2)と削る段階を組み合わせた実行計画から、本番の冒頭の T4 の見積もりのみで
自動選択する(6.1 節・6.4 節)。最大の候補 $2^{19}$ は学習用のシーンの約 11 エポック、各キャプションを約 363 回見る量である。

**パイロット**(第 1 段階、ローカルの Apple M4 の MPS、FP32、本トピックの`src/`のコードそのもの)。**判定に使う量(未見の組み合わせの
評価集合の指標)は計算していない。** 見たのは、前提条件 P1 の量(既知の組み合わせの検証集合の検索の正解率)と訓練損失のみである。
**損失の間の比較になる softmax 以外の条件は、P1 の閾値 0.1 を上回ったかどうかの真偽のみを印字し、数値を見ていない。** パイロットのシーンは
本番と同じ規則・同じ枚数(学習用は各キャプション 32 枚、検証は各 2 枚)で、別のシードで描いた。

| 損失 | $N$ | $E$ | 学習率 | 既知の組み合わせの検証集合の検索の正解率 |
|---|---|---|---|---|
| softmax | 256 | $2^{17}$ | $2 \times 10^{-3}$ | 0.406 |
| softmax | 256 | $2^{18}$ | $10^{-3}$ | 0.702 |
| softmax | 256 | $2^{18}$ | $2 \times 10^{-3}$ | 0.768 |
| softmax | 256 | $2^{18}$ | $4 \times 10^{-3}$ | 0.597 |
| softmax | 256 | $2^{19}$ | $10^{-3}$ | 0.917 |
| softmax | 16 | $2^{17}$ | $5 \times 10^{-4}$ | 0.241 |
| softmax | 16 | $2^{18}$ | $1.25 \times 10^{-4}$ | 0.326 |
| softmax | 16 | $2^{18}$ | $2.5 \times 10^{-4}$ | 0.334 |
| softmax | 16 | $2^{18}$ | $5 \times 10^{-4}$ | 0.296 |
| softmax | 16 | $2^{18}$ | $10^{-3}$ | 0.116 |
| sigmoid | 256 | $2^{17}$ | $2 \times 10^{-3}$ | 閾値 0.1 以上: 成立 |
| sigmoid | 16 | $2^{17}$ | $5 \times 10^{-4}$ | 閾値 0.1 以上: **不成立** |
| sigmoid | 16 | $2^{18}$ | $2.5 \times 10^{-4}$ | 閾値 0.1 以上: 成立 |
| NegCLIP | 256 | $2^{17}$ | $2 \times 10^{-3}$ | 閾値 0.1 以上: 成立 |

**本番前の改訂 1: 学習率の格子**

- 当初は、格子の中心を $N = 256$ で $2 \times 10^{-3}$ とし、他の $N$ へ平方根の規則 $\sqrt{N/256}$ でずらす設計だった($N = 16$ の格子は
  $\{2.5 \times 10^{-4}, 5 \times 10^{-4}, 10^{-3}\}$)。softmax・$N = 16$・$E = 2^{18}$ のパイロットで、最良がこの格子の下端 $2.5 \times 10^{-4}$ に
  来た。このまま本番で較正すると拡張の 1 回に頼ることになり、019 の実験 C と同じ前提不成立の危険が大きい。
- softmax の最良値の比($N = 256$ の $2 \times 10^{-3}$ と $N = 16$ の $2.5 \times 10^{-4}$ で 8 倍 $= 16^{3/4}$)から、中心を
  $2 \times 10^{-3} (N/256)^{3/4}$ に改めた($N = 16$ の格子は $\{1.25 \times 10^{-4}, 2.5 \times 10^{-4}, 5 \times 10^{-4}\}$)。改めた格子では、
  softmax の最良が $N = 256$($10^{-3}$・$2 \times 10^{-3}$・$4 \times 10^{-3}$ で 0.702・0.768・0.597)・$N = 16$($1.25$・$2.5$・$5 \times 10^{-4}$ で
  0.326・0.334・0.296、ただし差は小さい)のどちらでも内点にある。
- **この改訂は判定基準(対比量・閾値の導出・期待する差の方向)を変えず、前提条件 P0 の定義も変えない。** 根拠は softmax のみの、判定に
  使わない量であり、sigmoid の最良の学習率がどこにあるかは見ていない。sigmoid の格子が softmax と同じ規則で適切かどうかは確かめておらず、
  P0 が不成立になる危険は残る。

**本番前の改訂 2: $E$ の候補から $2^{17}$ を除いた**

- 当初は候補を $\{2^{19}, 2^{18}, 2^{17}\}$(12 通りの実行計画)とし、下限 $2^{17}$ で全条件が P1 を満たす見込みをパイロットで確かめる予定だった。
  $E = 2^{17}$ で、softmax($N = 16$ で 0.24、$N = 256$ で 0.41)・sigmoid の $N = 256$・NegCLIP は閾値を上回ったが、**sigmoid の $N = 16$ は下回った。**
  $E = 2^{18}$ では sigmoid の $N = 16$ も閾値を上回った(改めた格子の中心 $2.5 \times 10^{-4}$)。そこで、全条件で前提条件 P1 を満たす見込みの
  ある $E = 2^{18}$ を下限とし、$2^{17}$ を候補から除いた(8 通りの実行計画)。
- 改訂の根拠は「ある条件で学習が進まない(前提条件の量が閾値に届かない)学習量を候補から除く」ことであり、全条件の学習量を一律に
  増やす方向の変更である。判定基準・P1 の閾値(0.1、「チャンス水準 1/1444 より十分高い」値として先に決めたもの)は変えていない。
- **この真偽の確認は、実験 A の対比量についての部分的な情報を含む。** $N = 16$・$E = 2^{17}$ の既知の組み合わせで、softmax が 0.24 だった
  設定の近く(学習率は同じ $5 \times 10^{-4}$)で、sigmoid が 0.1 に届かなかった。これは、少なくともこの設定では sigmoid が softmax より低い
  ことを示唆する(判定に使う $M$ ではなく、学習率も較正していない)。6.1 節の実験 A の宣言(対比量・標準偏差の導出・閾値・期待する差の
  方向)は、この結果を見る前に書いたものから変えていない。
- $E = 2^{18}$ でも、$N = 256$ の softmax の正解率(0.77)は $E = 2^{19}$(0.92)より低く、学習はまだ飽和していない。$E = 2^{18}$ が選ばれた場合、
  その点を結果・考察で明記する。

**P1 の閾値 0.1**: 「チャンス水準(1/1444)より十分高く、学習が進んでいない条件を除ける」値として 0.1 とした。パイロットは、候補の下限の
$E$ でこの閾値を満たせるかを確かめるために使った(閾値を満たすように $E$ の候補の下限を選んだのであり、閾値は動かしていない)。

**パイロットと本番の違い**: 上の表の sigmoid のパイロットでは、sigmoid の損失にも倍率 $e^{t'}$ の 100 以下への切り詰めを掛けていた
(本番では掛けない、6.1 節)。真偽のみを確かめたこれらの結果は、この点で本番の設定と異なる。下の改訂 3 のパイロットは切り詰めなしで
行った。パイロットはすべてシード 0・1 シードである。

#### 第 1 段階のレビューによる改訂(本番実行の前)

以下の改訂はすべて本番実行の前に行った。**判定基準(対比量の定義・閾値の導出式・期待する差の方向)と前提条件の定義は変えていない。**
改訂の根拠は、交絡の分離・評価の設計の整合性という一般論であり、観測結果の方向には依存しない。改訂の前に実行したのはスモークテスト
(縮小した規模、前提条件がすべて不成立で判定に意味がない)とパイロット(判定に使う量を計算していない)のみである。

**改訂 3: sigmoid の学習率の格子の中心**

- 旧定義: sigmoid の格子も softmax と同じ中心 $2 \times 10^{-3} (N/256)^{3/4}$(改訂 1)。sigmoid について格子の中心を確かめていなかった。
- パイロット: sigmoid・$N = 16$ と sigmoid・$N = 256$ を、$E = 2^{18}$・本番と同じ設定(倍率の切り詰めなし)・旧定義の格子の 3 点で学習した。
  **印字したのは、最良の格子点の位置と P1 の閾値を上回ったかの真偽のみであり、正解率の数値は見ていない。**

  | 損失 | $N$ | 旧定義の格子 | 最良の格子点の位置 | 最良の点で P1 の閾値 0.1 以上 |
  |---|---|---|---|---|
  | sigmoid | 16 | $\{1.25, 2.5, 5\} \times 10^{-4}$ | 下端 | 成立 |
  | sigmoid | 256 | $\{1, 2, 4\} \times 10^{-3}$ | 下端 | 成立 |

- 事前に決めた規則: 「最良が下端なら中心を公比 2 だけ下げ、上端なら上げる。内点なら変えない」。位置の情報のみを使う。
- 新定義: sigmoid の $N = 256$ の中心を $10^{-3}$ とし、格子を $\{5 \times 10^{-4}, 10^{-3}, 2 \times 10^{-3}\}$($N = 16$ は
  $\{6.25 \times 10^{-5}, 1.25 \times 10^{-4}, 2.5 \times 10^{-4}\}$)にした。旧定義の最良の点が新しい格子の中心になる。softmax の格子は変えない。
- 理由: 本番の較正で最良が格子の端に来ると、拡張の 1 回に頼ることになり、前提条件 P0 が不成立になる危険が大きい(019 の実験 C)。
- 以前の(切り詰めありの)sigmoid のパイロットとの関係: 以前のパイロットの学習率($N = 256$ で $2 \times 10^{-3}$、$N = 16$ で $5 \times 10^{-4}$ と
  $2.5 \times 10^{-4}$)は旧定義の格子の中心か上端で、今回のパイロットで最良ではなかった側にある。以前のパイロットは P1 の見込みの確認
  のみに使い、格子の決定には使っていない。切り詰めの有無が今回の位置の結果に影響したかどうかは確かめていない。
- 注意: 格子の中心が損失によって違うのは、損失ごとに最適な学習率が異なる可能性を反映したものであり、実験 A の比較は条件ごとの較正で
  選んだ学習率どうしで行う(6.1 節)ことに変わりはない。

**改訂 4: NegCLIP の困難な負例から、除外した組を含むキャプションを除いた**

- 旧定義: 各正例の色の入れ替え・順序の入れ替えを、すべて困難な負例として分母に加えていた。
- 新定義: 除外した (色, 形) の組を含むキャプションは困難な負例に使わない(テキスト encoder に入れず、分母にも加えない)。
- 理由: 学習用のキャプションの色の入れ替えのうち 700 個(約 48%、5.3 節の出力)が除外した組を含む。旧定義では NegCLIP のテキスト
  encoder が学習中に除外した組を見ることになり、「学習に一度も出てこない組み合わせ」で評価するという前提が崩れていた。全学習について、
  テキスト encoder に入れた系列が除外した組を含まないことを記録から確かめる(6.11 節)。

**改訂 5: 実験 B のランダムな負例を、困難な負例と「既知 / 未見」の状態で対応させた**

- 旧定義: ランダムな負例 2 個を、どちらも正例以外の未見の組み合わせのキャプションから一様に選ぶ。
- 新定義: 1 個目は順序の入れ替えに対応させて未見の組み合わせから、2 個目は色の入れ替えに対応させてその色の入れ替えと同じ集合
  (学習用 / 未見の組み合わせ)から、正例とその困難な負例を除いて選ぶ(6.1 節の共通の設定)。試行の数(6,368)とシード(`20004`)は変えない。
- 理由: 未見の組み合わせのキャプションの色の入れ替えは約 88% が学習用のキャプションであり、旧定義では「学習で見たキャプションに
  引き寄せられる」傾向が bag-of-words 化と区別できない(交絡。6.1 節の実験 B の「解釈の注意」)。旧定義の $\mathrm{acc}_{\mathrm{rand}}$ と、
  既知性の偏りの大きさは診断量として残す。実験 C の診断量の $\mathrm{acc}_{\mathrm{rand}}$ も新定義で計算する。

**改訂 6: 記号の衝突の解消**

- 旧定義: 共通の埋め込みの次元と学習で見る事例数に、同じ記号 $E$ を使っていた。各 encoder の出力次元を $D_I$・$D_T$ としていた。
- 新定義: CLIP の原論文の図 3 の擬似コードに従い、共通の埋め込みの次元を $d_e$、各 encoder の出力次元を $d_i$・$d_t$ とした。学習で見る
  事例数は $E$ のままとする。

**改訂 7(記述の訂正)**: 3.7 節で、色の入れ替えが偶然同じバッチに入る確率を順序の入れ替えと同じとしていたのを訂正した。色の入れ替えの
キャプションが学習用として存在するのは正例の約 52% だけなので、確率は $(N - 1)/1443$ にその割合を掛けた平均になる(5.3 節の出力)。

#### 停止した本番の実行(コミット`81adad2`)

**学習と評価は一度も行われておらず、結果の情報は得ていない。** 本番の実行(Google Colab T4、コミット`81adad2`、実行日時
2026-09-30T08:55 UTC)は、6.4 節で見積もりが予算を超えたため、規則どおり学習の前に停止した。

- **T4 の 1 ステップの時間**(6.3 節の出力、当時は FP16 の autocast と動的損失スケーリングのみ):

  | 損失 | $N = 16$ | $N = 64$ | $N = 256$ |
  |---|---|---|---|
  | softmax | 45.4 ms | 47.2 ms | 66.6 ms |
  | sigmoid | 45.4 ms | 45.5 ms | 68.8 ms |
  | NegCLIP | | | 74.6 ms |

- **見積もり**: 1 回の学習と評価($E = 2^{18}$)は softmax・$N = 16$ で 848.7 秒、sigmoid・$N = 16$ で 746.9 秒、$N = 256$ で各約 73〜79 秒。
  較正の方式`"all"`で計画 4 が 329.4 分、計画 7 が 206.9 分で、予算 111.4 分(データの準備・確認・計測に 8.6 分)を超えた。参考として、
  `"representative"`なら計画 6 が 105.8 分だった。
- **停止の原因**: 1 ステップの時間が $N$ にほとんど依存しないので、計算ではなく **1 ステップの固定費**(演算の起動、ホストとデバイスの同期、
  Python・CPU 側の処理)が支配的だった。第 1 段階のローカル(Apple M4 の MPS)では $N = 16$ が約 20 ms / ステップで、T4 のほうが 2 倍以上
  遅い。第 1 段階の完了報告の T4 の見込みには、019 の「T4 は MPS より速い」という速度比を使ったが、その比は計算量の効く処理で測った
  ものであり、本トピックには当てはまらなかった(別のトピックの速度比の流用)。
- **固定費の内訳**(第 1 段階のコードの 1 ステップ、ローカルの profiler による。MPS のディスパッチの水準で数えた、カーネルを起動する
  演算の目安): 順伝播・逆伝播(損失・マージンを含む)が約 1,260、勾配のノルムと clipping が約 230(うち`_foreach_*`の呼び出し 2)、
  AdamW が`_foreach_*`の呼び出し 21(CUDA では融合したカーネルなので数十の起動、MPS では 1 テンソルずつに分解されて約 2,030)。
  ホストとデバイスの同期は、FP16 の動的損失スケーリングの非有限値の検査(毎ステップ 1 回)。そのほか毎ステップ、CPU でのバッチの
  切り出しと、ホストからデバイスへの添字の転送(NegCLIP では可変長の困難な負例の構築と 3 回の転送)があった。
- **対応**(判定基準・水準・前提条件は変えていない):
  1. 学習全体のバッチの添字とテキスト encoder に入れるキャプションの番号を学習の前に作ってデバイスに置き、各ステップでは切り出す
     だけにした(毎ステップの CPU の処理と転送をなくした)。テキスト encoder に入れたキャプションの集合などの記録も、事前に作った添字
     から作る。
  2. NegCLIP の困難な負例を固定長($2N$)にした。除外した組を含む負例は、その行の正例のキャプションで埋めてマスクで分母から除く
     (除外した組を含むキャプションをテキスト encoder に入れない不変条件は維持する。5.4 節・6.11 節)。
  3. 1 ステップの実行の方式を、FP16(動的損失スケーリングのため毎ステップ同期する)・FP32(同期なし)・FP32 と CUDA graph(順伝播・
     逆伝播・clipping を記録して再生し、約 1,260 の演算の起動を 1 回の再生にまとめる)から、T4 での時間の計測のみで自動で選ぶように
     した(5.1 節・6.3 節・6.4 節)。CUDA graph の経路は、T4 の上で eager の経路との一致を確かめてから候補に残す(6.3 節)。
  4. 較正の方式`"representative"`の計画を下位の計画 8〜11 として加えた(6.1 節)。
- **等価性**: 1・2 の前後の学習の一致を、コミット`81adad2`との比較で確かめる(5.4 節)。softmax・sigmoid は bit 単位で一致し、NegCLIP は
  固定長化による丸めの差のみである。
- **`torch.compile`は採らなかった**: 記録・コンパイルの時間(Colab の CPU で形ごとに数分かかりうる)、T4(compute capability 7.5)での
  コード生成の対応の不確かさ、融合による丸めの変化があり、ローカルで T4 の挙動を確かめられないため。CUDA graph は、同じカーネルを
  再生するだけで丸めが変わらず、失敗や不一致を T4 の上で検出して eager の経路に戻れるので採った。

#### 本番で選ばれた $E = 2^{18}$ とパイロットの関係(本番実行の後の追記)

- 本番では $E = 2^{18}$(計画 11)が選ばれた。上の表のパイロットのうち、この $E$ で行ったものとの関係は次のとおり。いずれもパイロットは
  別の環境(MPS)・別の描画のシード・1 シードである。
- softmax・$N = 256$・学習率 $2 \times 10^{-3}$: パイロットの既知の組み合わせの検証集合の検索の正解率 0.768 に対し、本番の較正(シード番号 90)は
  0.8352、本番の 3 シードは 0.8272・0.7604・0.7791 で、同じ程度の範囲にあった。
- softmax・$N = 16$・学習率 $2.5 \times 10^{-4}$: パイロット 0.334 に対し、本番の 3 シードは 0.3625・0.3521・0.3314 だった。
- sigmoid: 改訂 3 のパイロットでは $N = 256$ の最良が旧定義の格子の下端($10^{-3}$)だったが、本番の較正(新しい格子)では $2 \times 10^{-3}$ が
  最良だった($10^{-3}$ は 0.4997、$2 \times 10^{-3}$ は 0.5443)。パイロットと本番で最良の位置が一致しなかった。本番では sigmoid・$N = 16$ の
  学習率 0.00025 が $N = 256$ の値から規則で決まり、改訂 3 のパイロットで $N = 16$ の最良だった旧定義の格子の下端 $1.25 \times 10^{-4}$ の 2 倍に
  あたる(7.2 節・7.7 節)。
- $E = 2^{18}$ では学習が飽和していない(パイロットの softmax・$N = 256$ は $E = 2^{19}$ で 0.917)。その影響は 7.7 節に記す。

### 6.3 スケーリングの計測と外挿(1 セッションの見積もり)

本番で規模がスモークテストの何倍にもなる重い処理(描画・学習・評価)について、3 点の規模で実行時間を実測し、
$\log t = \log a + b \log n$ をあてはめてべき指数 $b$ を推定し、本番の規模へ外挿する。本番では、この計測を **本番の実行の冒頭に T4 上で**
行い、その値のみから実行の方式と実行計画を選ぶ(6.4 節)。外挿値と、最大の計測点の実測値を比例で伸ばした値の大きいほうを見積もりとする
(固定費があると $b < 1$ となり、外挿値が過小になりうるため)。

| 処理 | 計測する規模 | 外挿先(本番) | 本番での回数 |
|---|---|---|---|
| 描画(numpy) | シーンの数 4,096・8,192・16,384 | 52,280 枚(学習用・検証・評価の合計) | 1 回(5.3 節で実行済み。実測値を使う) |
| 学習(実行の方式の候補ごとに、損失 × $N$ の 7 種類: softmax・sigmoid × $N \in \{16, 64, 256\}$ と NegCLIP・$N = 256$。モデルの構築を含む) | ステップ数: eager は 32・64・128、CUDA graph は 128・256・512 | 各候補の $E$ の $T_N = E/N$ | 各計画の学習の数(6.4 節) + 較正 |
| 評価(画像の埋め込み・テキストの埋め込み、FP32) | 画像 800・1,600・3,200 枚、キャプション 560・1,120・2,240 個 | 1 枚・1 個あたりに換算 | 学習ごとに画像 20,808 枚・キャプション 13,440 個(較正は 11,552 枚・8,960 個) |

- **CUDA graph の等価性の確認**(CUDA のときのみ、計測の前): softmax・sigmoid の $N = 16$ と NegCLIP の $N = 256$ について、同じ初期値・同じ
  バッチの列で`fp32_eager`と`fp32_graph`を 6 ステップずつ学習し、重みの最大の相対差と損失の差を印字する。CUDA graph は同じカーネルを
  再生するだけなので一致するはずである。記録や再生で例外が起きた場合、または重みの最大の相対差が $10^{-5}$ を超えた場合は、`fp32_graph`を
  候補から外す(結果ではなく実装の正しさに基づく判断)。
- 計測の前に、各条件で最小の計測点のステップ数の準備運転を 1 回行う(カーネルの準備の時間を計測から除く)。計測専用のシード(シード番号 91)を
  使い、学習率は $10^{-3}$ とする(時間は学習率によらない)。CUDA graph の方式の計測時間には、記録(準備運転 3 回と記録)の時間も含まれる。
- softmax と sigmoid は 1 ステップの計算がほぼ同じだが、別々に計測する。NegCLIP はテキスト encoder に $3N$ 個の系列(正例 $N$ 個と、固定長の
  困難な負例 $2N$ 個。除外した組を含む負例は正例のキャプションで埋めてマスクで除く)を通すので別に計測する。計測は本番と同じ学習関数で
  行うので、1 ステップの時間はこの固定長化を反映している。
- **スモークテストの出力の見積もりは、ローカル(MPS)の値であり、T4 での本番の目安にならない。** 選択には本番の冒頭の T4 での計測が使われる。


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
# (0) CUDA graph の等価性の確認(CUDA のみ)
GRAPH_CHECK: dict[str, object] = {}
if "fp32_graph" in EXECUTION_MODE_CANDIDATES:
    try:
        for _kind in (("softmax", 16), ("sigmoid", 16), ("negclip", STANDARD_BATCH_SIZE)):
            _key = key_of(*_kind, TIMING_SEED_INDEX)
            _results = {}
            for _mode in ("fp32_eager", "fp32_graph"):
                _model = build_model_for(_key)
                _h = train_run(_model, _key, 1e-3, 6, execution_mode=_mode)
                _results[_mode] = ({k: v.detach().clone() for k, v in _model.state_dict().items()}, _h["loss"])
            _ref, _new = _results["fp32_eager"], _results["fp32_graph"]
            _rel = max(float((_ref[0][k] - _new[0][k]).abs().max() / _ref[0][k].abs().max().clamp_min(1e-12)) for k in _ref[0])
            _loss_diff = max(abs(a - b) for a, b in zip(_ref[1], _new[1], strict=True))
            GRAPH_CHECK[f"{_kind[0]} N={_kind[1]}"] = (_rel, _loss_diff)
            print(f"CUDA graph の等価性({_kind[0]} N={_kind[1]}、6 ステップ): 重みの最大の相対差 {_rel:.2e}、損失の最大の差 {_loss_diff:.2e}")
        if max(v[0] for v in GRAPH_CHECK.values()) > 1e-5:
            raise AssertionError("fp32_graph と fp32_eager の重みの相対差が 1e-5 を超えた")
        print("CUDA graph の等価性の確認: OK(fp32_graph を候補に残す)")
    except Exception as _error:  # noqa: BLE001  # 記録・再生の失敗も含めて、候補から外して続ける
        EXECUTION_MODE_CANDIDATES.remove("fp32_graph")
        GRAPH_CHECK["error"] = f"{type(_error).__name__}: {_error}"
        print(f"CUDA graph の等価性の確認に失敗したため、fp32_graph を候補から外す: {GRAPH_CHECK['error'][:300]}")
    empty_device_cache()
else:
    print(f"CUDA ではない({device})ため、fp32_graph は候補にない。候補 {EXECUTION_MODE_CANDIDATES}")

# (1) 描画
RENDER_TIMING_SIZES = (4096, 8192, 16384)
TOTAL_SCENES = sum(len(ids) for ids, _ in SCENE_SPECS.values())
_render_ids = np.resize(TRAIN_CAPTION_IDS, RENDER_TIMING_SIZES[-1])
render_scenes(UNIVERSE, _render_ids[:1024], 12345)  # 準備運転
_render_times = [timed_call(lambda n=n: render_scenes(UNIVERSE, _render_ids[:n], 12345)) for n in RENDER_TIMING_SIZES]
ESTIMATE_RENDER = fit_and_extrapolate("描画(シーンの数)", RENDER_TIMING_SIZES, _render_times, (TOTAL_SCENES,))[TOTAL_SCENES]

# (2) 学習(実行の方式の候補ごと)
TRAINING_KINDS = [(loss, n) for loss in A_LOSSES for n in BATCH_SIZES] + [("negclip", STANDARD_BATCH_SIZE)]
ESTIMATE_TRAIN: dict[str, dict] = {}  # 方式 -> (損失, N) -> ステップ数 -> 学習 1 回の秒数
STEP_SECONDS: dict[str, dict] = {}  # 方式 -> (損失, N) -> 最大の計測点での 1 ステップあたりの時間
for _mode in EXECUTION_MODE_CANDIDATES:
    _counts = SCALING_STEP_COUNTS["graph" if EXECUTION_MODES[_mode][1] else "eager"]
    ESTIMATE_TRAIN[_mode], STEP_SECONDS[_mode] = {}, {}
    for _kind in TRAINING_KINDS:
        _key = key_of(*_kind, TIMING_SEED_INDEX)
        train_run(build_model_for(_key), _key, 1e-3, _counts[0], execution_mode=_mode)  # 準備運転(計測しない)
        _times = [
            timed_call(lambda n=n, k=_key, m=_mode: train_run(build_model_for(k), k, 1e-3, n, execution_mode=m)) for n in _counts
        ]
        _targets = sorted({steps_for(e, _kind[1]) for level in EXAMPLE_CANDIDATES for e in EXAMPLE_CANDIDATES[level]})
        ESTIMATE_TRAIN[_mode][_kind] = fit_and_extrapolate(f"{_mode} {_kind[0]} N={_kind[1]} の学習(ステップ数)", _counts, _times, _targets)
        STEP_SECONDS[_mode][_kind] = _times[-1] / _counts[-1]
        empty_device_cache()

# (3) 評価
EVAL_IMAGE_TIMING_SIZES = (800, 1600, 3200)
EVAL_TEXT_TIMING_SIZES = (560, 1120, 2240)
_model = build_model_for(key_of("softmax", STANDARD_BATCH_SIZE, TIMING_SEED_INDEX))
encode_images(_model, UNSEEN_IMAGES[:400])  # 準備運転
_image_times = [timed_call(lambda n=n, m=_model: encode_images(m, UNSEEN_IMAGES[:n])) for n in EVAL_IMAGE_TIMING_SIZES]
_text_times = [timed_call(lambda n=n, m=_model: encode_texts(m, TOKENS[:n])) for n in EVAL_TEXT_TIMING_SIZES]
EVAL_SECONDS_PER_IMAGE = fit_and_extrapolate("評価: 画像の埋め込み(枚数)", EVAL_IMAGE_TIMING_SIZES, _image_times, (20_000,))[20_000] / 20_000
EVAL_SECONDS_PER_TEXT = fit_and_extrapolate("評価: テキストの埋め込み(個数)", EVAL_TEXT_TIMING_SIZES, _text_times, (20_000,))[20_000] / 20_000
del _model
empty_device_cache()

# 学習 1 回あたりの評価の量(5.5 節の run の手順から数える)
_seen, _unseen, _texts = len(VALIDATION_IMAGES), len(UNSEEN_IMAGES), NUM_CAPTIONS
_intermediate = len(INTERMEDIATE_EVAL_FRACTIONS)
EVAL_COUNTS = {
    # 途中の評価 3 回 + 最後の既知の検証 + 最後の評価(未見・既知の画像とテキスト)+ 学習前の modality gap
    "main": ((_intermediate + 1) * _seen + _unseen + _seen + _unseen, (_intermediate + 1) * _texts + _texts + _texts),
    "calibration": ((_intermediate + 1) * _seen, (_intermediate + 1) * _texts),
}
SCALING_SECONDS = time.time() - _t0_scaling
print(f"スケーリングの計測自体: {SCALING_SECONDS:.1f}s。描画の見積もり(全 {TOTAL_SCENES:,} 枚): {ESTIMATE_RENDER:.1f}s")
print(f"学習 1 回あたりの評価の量(画像, キャプション): {EVAL_COUNTS}")
for _mode in EXECUTION_MODE_CANDIDATES:
    print(
        f"1 ステップあたりの時間({device}、{_mode}、最大の計測点): "
        + "、".join(f"{k[0]} N={k[1]} {v * 1000:.1f} ms" for k, v in STEP_SECONDS[_mode].items())
    )
```

    CUDA graph の等価性(softmax N=16、6 ステップ): 重みの最大の相対差 1.55e-05、損失の最大の差 0.00e+00
    CUDA graph の等価性(sigmoid N=16、6 ステップ): 重みの最大の相対差 2.71e-06、損失の最大の差 2.38e-07
    CUDA graph の等価性(negclip N=256、6 ステップ): 重みの最大の相対差 7.33e-06、損失の最大の差 4.77e-07
    CUDA graph の等価性の確認に失敗したため、fp32_graph を候補から外す: AssertionError: fp32_graph と fp32_eager の重みの相対差が 1e-5 を超えた
    [描画(シーンの数)] n=4,096: 1.10s, n=8,192: 1.94s, n=16,384: 3.23s -> b=0.777(標準誤差 0.023), R^2=0.9991; n=52,280: 外挿 8.0s・比例 10.3s
    [fp16_eager softmax N=16 の学習(ステップ数)] n=32: 1.50s, n=64: 2.94s, n=128: 8.14s -> b=1.218(標準誤差 0.145), R^2=0.9860; n=256: 外挿 17.9s・比例 16.3s、n=512: 外挿 41.6s・比例 32.5s、n=16,384: 外挿 2831.9s・比例 1041.4s、n=32,768: 外挿 6588.4s・比例 2082.8s
    [fp16_eager softmax N=64 の学習(ステップ数)] n=32: 1.64s, n=64: 3.70s, n=128: 6.06s -> b=0.943(標準誤差 0.133), R^2=0.9806; n=64: 外挿 3.3s・比例 3.0s、n=128: 外挿 6.4s・比例 6.1s、n=4,096: 外挿 168.0s・比例 194.0s、n=8,192: 外挿 323.1s・比例 388.0s
    [fp16_eager softmax N=256 の学習(ステップ数)] n=32: 2.16s, n=64: 5.91s, n=128: 8.61s -> b=0.996(標準誤差 0.262), R^2=0.9353; n=16: 外挿 1.2s・比例 1.1s、n=32: 外挿 2.4s・比例 2.2s、n=1,024: 外挿 75.9s・比例 68.9s、n=2,048: 外挿 151.5s・比例 137.8s
    [fp16_eager sigmoid N=16 の学習(ステップ数)] n=32: 1.53s, n=64: 3.19s, n=128: 6.18s -> b=1.008(標準誤差 0.030), R^2=0.9991; n=256: 外挿 12.6s・比例 12.4s、n=512: 外挿 25.3s・比例 24.7s、n=16,384: 外挿 834.4s・比例 791.5s、n=32,768: 外挿 1678.6s・比例 1583.0s
    [fp16_eager sigmoid N=64 の学習(ステップ数)] n=32: 1.52s, n=64: 2.91s, n=128: 6.27s -> b=1.022(標準誤差 0.049), R^2=0.9977; n=64: 外挿 3.0s・比例 3.1s、n=128: 外挿 6.1s・比例 6.3s、n=4,096: 外挿 212.5s・比例 200.6s、n=8,192: 外挿 431.6s・比例 401.2s
    [fp16_eager sigmoid N=256 の学習(ステップ数)] n=32: 2.15s, n=64: 4.30s, n=128: 8.92s -> b=1.025(標準誤差 0.017), R^2=0.9997; n=16: 外挿 1.1s・比例 1.1s、n=32: 外挿 2.1s・比例 2.2s、n=1,024: 外挿 74.7s・比例 71.4s、n=2,048: 外挿 152.1s・比例 142.8s
    [fp16_eager negclip N=256 の学習(ステップ数)] n=32: 2.44s, n=64: 5.06s, n=128: 9.72s -> b=0.998(標準誤差 0.033), R^2=0.9989; n=16: 外挿 1.2s・比例 1.2s、n=32: 外挿 2.5s・比例 2.4s、n=1,024: 外挿 78.5s・比例 77.8s、n=2,048: 外挿 156.9s・比例 155.6s
    [fp32_eager softmax N=16 の学習(ステップ数)] n=32: 1.42s, n=64: 2.51s, n=128: 4.89s -> b=0.894(標準誤差 0.040), R^2=0.9980; n=256: 外挿 8.9s・比例 9.8s、n=512: 外挿 16.6s・比例 19.6s、n=16,384: 外挿 368.6s・比例 626.5s、n=32,768: 外挿 685.0s・比例 1252.9s
    [fp32_eager softmax N=64 の学習(ステップ数)] n=32: 1.61s, n=64: 3.08s, n=128: 4.92s -> b=0.807(標準誤差 0.074), R^2=0.9917; n=64: 外挿 2.9s・比例 2.5s、n=128: 外挿 5.1s・比例 4.9s、n=4,096: 外挿 83.2s・比例 157.6s、n=8,192: 外挿 145.5s・比例 315.2s
    [fp32_eager softmax N=256 の学習(ステップ数)] n=32: 3.02s, n=64: 6.03s, n=128: 11.73s -> b=0.979(標準誤差 0.011), R^2=0.9999; n=16: 外挿 1.5s・比例 1.5s、n=32: 外挿 3.0s・比例 2.9s、n=1,024: 外挿 90.2s・比例 93.8s、n=2,048: 外挿 177.8s・比例 187.7s
    [fp32_eager sigmoid N=16 の学習(ステップ数)] n=32: 1.28s, n=64: 2.72s, n=128: 5.73s -> b=1.082(標準誤差 0.004), R^2=1.0000; n=256: 外挿 12.2s・比例 11.5s、n=512: 外挿 25.7s・比例 22.9s、n=16,384: 外挿 1092.4s・比例 733.7s、n=32,768: 外挿 2312.1s・比例 1467.3s
    [fp32_eager sigmoid N=64 の学習(ステップ数)] n=32: 1.23s, n=64: 2.49s, n=128: 5.56s -> b=1.089(標準誤差 0.042), R^2=0.9985; n=64: 外挿 2.6s・比例 2.8s、n=128: 外挿 5.5s・比例 5.6s、n=4,096: 外挿 238.7s・比例 178.0s、n=8,192: 外挿 507.9s・比例 356.0s
    [fp32_eager sigmoid N=256 の学習(ステップ数)] n=32: 2.88s, n=64: 5.77s, n=128: 11.62s -> b=1.006(標準誤差 0.002), R^2=1.0000; n=16: 外挿 1.4s・比例 1.5s、n=32: 外挿 2.9s・比例 2.9s、n=1,024: 外挿 94.0s・比例 93.0s、n=2,048: 外挿 188.7s・比例 186.0s
    [fp32_eager negclip N=256 の学習(ステップ数)] n=32: 3.68s, n=64: 7.32s, n=128: 14.55s -> b=0.992(標準誤差 0.001), R^2=1.0000; n=16: 外挿 1.8s・比例 1.8s、n=32: 外挿 3.7s・比例 3.6s、n=1,024: 外挿 114.6s・比例 116.4s、n=2,048: 外挿 227.9s・比例 232.7s
    [評価: 画像の埋め込み(枚数)] n=800: 0.08s, n=1,600: 0.16s, n=3,200: 0.32s -> b=1.008(標準誤差 0.029), R^2=0.9992; n=20,000: 外挿 2.0s・比例 2.0s
    [評価: テキストの埋め込み(個数)] n=560: 0.01s, n=1,120: 0.02s, n=2,240: 0.03s -> b=0.711(標準誤差 0.073), R^2=0.9895; n=20,000: 外挿 0.1s・比例 0.3s
    スケーリングの計測自体: 239.0s。描画の見積もり(全 52,280 枚): 10.3s
    学習 1 回あたりの評価の量(画像, キャプション): {'main': (20808, 13440), 'calibration': (11552, 8960)}
    1 ステップあたりの時間(cuda、fp16_eager、最大の計測点): softmax N=16 63.6 ms、softmax N=64 47.4 ms、softmax N=256 67.3 ms、sigmoid N=16 48.3 ms、sigmoid N=64 49.0 ms、sigmoid N=256 69.7 ms、negclip N=256 76.0 ms
    1 ステップあたりの時間(cuda、fp32_eager、最大の計測点): softmax N=16 38.2 ms、softmax N=64 38.5 ms、softmax N=256 91.6 ms、sigmoid N=16 44.8 ms、sigmoid N=64 43.5 ms、sigmoid N=256 90.8 ms、negclip N=256 113.6 ms


### 6.4 実行の方式と実行計画の選択

6.3 節の外挿値から、実行の方式の候補ごとに、12 通りの実行計画(6.1 節)の残りの実行時間(較正と本番の学習・評価)を見積もる。

1. **実行の方式**: 最も優先順位の高い計画 0 の見積もりが最も短い方式を選ぶ(全計画に同じ方式を使う。1 ステップの時間の比は計画によらず
   ほぼ同じなので、計画 0 の見積もりで比べる)。
2. **実行計画**: 選んだ方式の見積もりで、予算「120 分 − ノートブックの開始からの経過時間」に収まる番号の最も小さい計画を選ぶ。
   計画 0〜7(較正の方式`"all"`)で収まる計画があれば、常にそれが選ばれる。
3. 学習率の較正の方式`CALIBRATION_MODE`は、選ばれた計画から決まる。

**選択は時間の見積もりのみに基づき、どの実験の結果も参照しない。** この時点では、較正も本番の学習も行っていない。較正は、格子の拡張が
起きる場合(各条件 4 回の学習)を、選ばれた $E$ で行う前提で見積もりに含める(安全側)。


```python
def run_estimate(mode: str, kind: tuple, examples: int, purpose: str) -> float:
    images, texts = EVAL_COUNTS[purpose]
    return ESTIMATE_TRAIN[mode][kind][steps_for(examples, kind[1])] + images * EVAL_SECONDS_PER_IMAGE + texts * EVAL_SECONDS_PER_TEXT


def plan_runs(stage: dict) -> list[tuple]:
    # 段階が決める本番の学習の鍵(重複を除き、実験 A -> 実験 B・C の順)
    keys: list[tuple] = []
    for s in range(stage["SEEDS_A"]):
        keys += [key_of(loss, n, s) for n in stage["A_BATCH_SIZES"] for loss in A_LOSSES]
    for s in range(stage["SEEDS_BC"]):
        keys += [key_of("softmax", STANDARD_BATCH_SIZE, s), key_of("negclip", STANDARD_BATCH_SIZE, s)]
    return list(dict.fromkeys(keys))


def calibration_conditions(stage: dict, calibration_mode: str) -> list[tuple]:
    batch_sizes = stage["A_BATCH_SIZES"] if calibration_mode == "all" else (STANDARD_BATCH_SIZE,)
    return [(loss, n) for n in batch_sizes for loss in A_LOSSES]


def estimate_plan(plan: dict, mode: str) -> dict:
    # 本番の値の計画(E は本番の候補)の残りの実行時間の内訳
    e = plan["E"]
    stage = STAGES["prod"][plan["stage"]]
    parts = {"較正(各条件 4 学習、拡張を含む安全側)": sum(
        (len(LR_GRID_MULTIPLIERS) + 1) * run_estimate(mode, c, e, "calibration")
        for c in calibration_conditions(stage, plan["calibration_mode"])
    )}
    for key in plan_runs(stage):
        name = f"本番: {key[0]} N={key[1]}"
        parts[name] = parts.get(name, 0.0) + run_estimate(mode, (key[0], key[1]), e, "main")
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
PLAN_ESTIMATES = {mode: {p["plan"]: estimate_plan(p, mode) for p in PLANS["prod"]} for mode in EXECUTION_MODE_CANDIDATES}
PLAN_TOTALS_BY_MODE = {mode: {k: sum(v.values()) for k, v in est.items()} for mode, est in PLAN_ESTIMATES.items()}
# (1) 実行の方式: 計画 0 の見積もりが最も短い方式(同じ値なら候補の順で先のもの)
EXECUTION_MODE = min(EXECUTION_MODE_CANDIDATES, key=lambda m: (PLAN_TOTALS_BY_MODE[m][0], EXECUTION_MODE_CANDIDATES.index(m)))
PLAN_TOTALS = PLAN_TOTALS_BY_MODE[EXECUTION_MODE]
print(
    f"すでに使った時間(ノートブックの開始から、データの準備・確認・計測を含む): {ELAPSED_BEFORE_SELECTION / 60:.1f} 分"
    f"(データの準備 {DATA_SECONDS:.0f}s、確認 {CHECK_SECONDS:.0f}s、スケーリングの計測 {SCALING_SECONDS:.0f}s)"
)
print(f"残りの予算: {SESSION_BUDGET_SECONDS / 60:.0f} 分 - {ELAPSED_BEFORE_SELECTION / 60:.1f} 分 = {REMAINING_BUDGET_SECONDS / 60:.1f} 分")
print("1 回の学習と評価の見積もり(秒、本番の E の候補ごと):")
for _mode in EXECUTION_MODE_CANDIDATES:
    for _kind in TRAINING_KINDS:
        print(
            f"  {_mode} {_kind[0]} N={_kind[1]}: "
            + "、".join(f"E = 2^{int(math.log2(e))}: {run_estimate(_mode, _kind, e, 'main'):.1f}s" for e in PROD_EXAMPLE_CANDIDATES)
        )
print(f"\n--- 実行計画ごとの残りの実行時間の見積もり(分、{device} 基準、残りの予算 {REMAINING_BUDGET_SECONDS / 60:.1f} 分)---")
print("計画 | E | 段階 | 較正の方式 | 本番の学習の数 | 較正の条件数 | " + " | ".join(EXECUTION_MODE_CANDIDATES))
for _p in PLANS["prod"]:
    _stage = STAGES["prod"][_p["stage"]]
    print(
        f"{_p['plan']:4d} | 2^{int(math.log2(_p['E']))} | {_p['stage']} | {_p['calibration_mode']:14s} | {len(plan_runs(_stage)):3d} | "
        f"{len(calibration_conditions(_stage, _p['calibration_mode']))} | "
        + " | ".join(f"{PLAN_TOTALS_BY_MODE[m][_p['plan']] / 60:8.1f}" for m in EXECUTION_MODE_CANDIDATES)
    )
for _mode, _totals in PLAN_TOTALS_BY_MODE.items():  # 同じ E・同じ較正の方式では、段階が上がると見積もりが減る
    for _group in ({(p["E"], p["calibration_mode"]) for p in PLANS["prod"]}):
        _same = [_totals[p["plan"]] for p in PLANS["prod"] if (p["E"], p["calibration_mode"]) == _group]
        assert all(a >= b for a, b in zip(_same, _same[1:], strict=False)), (_mode, _group)
print(
    f"実行の方式: 計画 0 の見積もり {'、'.join(f'{m}: {PLAN_TOTALS_BY_MODE[m][0] / 60:.1f} 分' for m in EXECUTION_MODE_CANDIDATES)} -> "
    f"{EXECUTION_MODE!r} を選んだ(時間の見積もりのみによる)"
)
print(f"--- 優先順位の最も高い計画(計画 0)の内訳({EXECUTION_MODE}) ---")
for _k, _v in PLAN_ESTIMATES[EXECUTION_MODE][0].items():
    print(f"  {_k}: {_v:,.1f}s")

try:
    SELECTED_PLAN = select_plan(PLAN_TOTALS, REMAINING_BUDGET_SECONDS)
    PLAN_SELECTION_MESSAGE = (
        f"残りの予算 {REMAINING_BUDGET_SECONDS / 60:.1f} 分に収まる番号の最も小さい計画として、計画 {SELECTED_PLAN} を選んだ"
        f"(見積もり {PLAN_TOTALS[SELECTED_PLAN] / 60:.1f} 分、{device} 基準、実行の方式 {EXECUTION_MODE!r})"
    )
except PlanBudgetExceededError as _error:
    if not SMOKE_TEST:
        print(f"\n警告: {_error}。本番の学習を始める前に停止する。")
        raise
    SELECTED_PLAN = max(PLAN_TOTALS)
    PLAN_SELECTION_MESSAGE = f"{_error}(本番なら学習の前に停止する)。スモークテストのため停止せず、計画 {SELECTED_PLAN} で動作確認を続ける"
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
    # 計画 0〜7 のどれかが収まる予算では、計画 8〜11 は選ばれない
    _budget = max(PLAN_TOTALS[p] for p in range(8))
    assert select_plan(PLAN_TOTALS, _budget) < 8
    print(
        "選択の規則の確認(人為的な予算、確認のみ): 各計画の見積もりに等しい予算で「番号の最も小さい収まる計画」が選ばれる、"
        f"計画 0〜7 が収まる予算では計画 8〜11 が選ばれない、どの計画でも超える予算で停止する: OK。予算の定数は {SESSION_BUDGET_SECONDS / 60:.0f} 分のまま"
    )
    assert SESSION_BUDGET_SECONDS == 120 * 60

# --- 選ばれた計画の値(以降のすべてのセルがこれを使う) ---
SELECTED = PLANS[CURRENT_LEVEL_NAME][SELECTED_PLAN]
EXAMPLES = SELECTED["E"]  # E
SELECTED_STAGE = SELECTED["stage"]
CALIBRATION_MODE = SELECTED["calibration_mode"]
STAGE = STAGES[CURRENT_LEVEL_NAME][SELECTED_STAGE]
SEEDS_A = tuple(range(STAGE["SEEDS_A"]))
SEEDS_BC = tuple(range(STAGE["SEEDS_BC"]))
A_BATCH_SIZES = STAGE["A_BATCH_SIZES"]
PLAN = plan_runs(STAGE)
CALIBRATION_CONDITIONS = calibration_conditions(STAGE, CALIBRATION_MODE)
UPLOAD_KEY = key_of("negclip", STANDARD_BATCH_SIZE, UPLOAD_KEY_SEED)
assert UPLOAD_KEY in PLAN
print(f"\n計画の選択: {PLAN_SELECTION_MESSAGE}")
print(
    f"選ばれた計画 {SELECTED_PLAN}(水準 {CURRENT_LEVEL_NAME!r}): E = {EXAMPLES:,}、段階 {SELECTED_STAGE}、"
    f"較正の方式 CALIBRATION_MODE = {CALIBRATION_MODE!r}、実行の方式 EXECUTION_MODE = {EXECUTION_MODE!r}、"
    f"ステップ数 " + "、".join(f"T_{n} = {steps_for(EXAMPLES, n):,}" for n in BATCH_SIZES)
)
print(
    f"このノートブックで使う値: 実験 A のシード {SEEDS_A}・N {A_BATCH_SIZES}、実験 B・C のシード {SEEDS_BC}、"
    f"本番の学習 {len(PLAN)} 回、較正する条件 {CALIBRATION_CONDITIONS}"
)
if device.type != "cuda":
    print("注意: CUDA 以外での見積もりであり、T4 での時間とは異なる")
```

    すでに使った時間(ノートブックの開始から、データの準備・確認・計測を含む): 5.2 分(データの準備 35s、確認 21s、スケーリングの計測 239s)
    残りの予算: 120 分 - 5.2 分 = 114.8 分
    1 回の学習と評価の見積もり(秒、本番の E の候補ごと):
      fp16_eager softmax N=16: E = 2^19: 6590.7s、E = 2^18: 2834.2s
      fp16_eager softmax N=64: E = 2^19: 390.3s、E = 2^18: 196.3s
      fp16_eager softmax N=256: E = 2^19: 153.8s、E = 2^18: 78.2s
      fp16_eager sigmoid N=16: E = 2^19: 1680.9s、E = 2^18: 836.7s
      fp16_eager sigmoid N=64: E = 2^19: 433.9s、E = 2^18: 214.8s
      fp16_eager sigmoid N=256: E = 2^19: 154.4s、E = 2^18: 77.0s
      fp16_eager negclip N=256: E = 2^19: 159.2s、E = 2^18: 80.8s
      fp32_eager softmax N=16: E = 2^19: 1255.2s、E = 2^18: 628.8s
      fp32_eager softmax N=64: E = 2^19: 317.5s、E = 2^18: 159.9s
      fp32_eager softmax N=256: E = 2^19: 190.0s、E = 2^18: 96.1s
      fp32_eager sigmoid N=16: E = 2^19: 2314.4s、E = 2^18: 1094.7s
      fp32_eager sigmoid N=64: E = 2^19: 510.2s、E = 2^18: 241.0s
      fp32_eager sigmoid N=256: E = 2^19: 191.0s、E = 2^18: 96.3s
      fp32_eager negclip N=256: E = 2^19: 235.0s、E = 2^18: 118.7s
    
    --- 実行計画ごとの残りの実行時間の見積もり(分、cuda 基準、残りの予算 114.8 分)---
    計画 | E | 段階 | 較正の方式 | 本番の学習の数 | 較正の条件数 | fp16_eager | fp32_eager
       0 | 2^19 | 0 | all            |  35 | 6 |   1423.5 |    735.9
       1 | 2^19 | 1 | all            |  25 | 4 |   1300.0 |    611.9
       2 | 2^19 | 2 | all            |  19 | 4 |   1019.1 |    486.5
       3 | 2^19 | 3 | all            |  15 | 4 |   1008.7 |    472.4
       4 | 2^18 | 0 | all            |  35 | 6 |    641.9 |    357.0
       5 | 2^18 | 1 | all            |  25 | 4 |    580.4 |    297.0
       6 | 2^18 | 2 | all            |  19 | 4 |    455.5 |    236.3
       7 | 2^18 | 3 | all            |  15 | 4 |    450.2 |    229.2
       8 | 2^18 | 0 | representative |  35 | 2 |    370.1 |    215.6
       9 | 2^18 | 1 | representative |  25 | 2 |    335.8 |    182.2
      10 | 2^18 | 2 | representative |  19 | 2 |    210.9 |    121.6
      11 | 2^18 | 3 | representative |  15 | 2 |    205.6 |    114.4
    実行の方式: 計画 0 の見積もり fp16_eager: 1423.5 分、fp32_eager: 735.9 分 -> 'fp32_eager' を選んだ(時間の見積もりのみによる)
    --- 優先順位の最も高い計画(計画 0)の内訳(fp32_eager) ---
      較正(各条件 4 学習、拡張を含む安全側): 19,089.0s
      本番: softmax N=16: 6,276.1s
      本番: sigmoid N=16: 11,571.9s
      本番: softmax N=64: 1,587.3s
      本番: sigmoid N=64: 2,551.2s
      本番: softmax N=256: 949.8s
      本番: sigmoid N=256: 954.8s
      本番: negclip N=256: 1,175.2s
    
    計画の選択: 残りの予算 114.8 分に収まる番号の最も小さい計画として、計画 11 を選んだ(見積もり 114.4 分、cuda 基準、実行の方式 'fp32_eager')
    選ばれた計画 11(水準 'prod'): E = 262,144、段階 3、較正の方式 CALIBRATION_MODE = 'representative'、実行の方式 EXECUTION_MODE = 'fp32_eager'、ステップ数 T_16 = 16,384、T_64 = 4,096、T_256 = 1,024
    このノートブックで使う値: 実験 A のシード (0, 1, 2)・N (16, 256)、実験 B・C のシード (0, 1, 2)、本番の学習 15 回、較正する条件 [('softmax', 256), ('sigmoid', 256)]


### 6.5 学習率の較正(P0)

6.1 節の規則で、較正する条件ごとに学習率を選ぶ。各格子点で、6.4 節で選ばれた $E$・全学習データ・較正専用のシードで学習し、
**既知の組み合わせの検証集合の検索の正解率のみ** を見る(未見の組み合わせの評価集合は評価しない)。


```python
_t0_calibration = time.time()


def calibrate(condition: tuple) -> dict:
    loss_type, batch_size = condition
    key = key_of(loss_type, batch_size, CALIBRATION_SEED_INDEX)
    records: dict[float, dict] = {}

    def best() -> float:  # 既知の組み合わせの検証集合の検索の正解率が最大(同点なら小さい学習率)
        return max(sorted(records), key=lambda lr: (records[lr]["seen"]["accuracy"], -lr))

    for lr in lr_grid_for(loss_type, batch_size):
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
    return {"records": records, "chosen": chosen, "extended": extended, "interior": min(records) < chosen < max(records)}


CALIBRATION: dict[tuple, dict] = {}
for _condition in CALIBRATION_CONDITIONS:
    print(f"--- {_condition[0]} N={_condition[1]} の較正(格子 {tuple(float(f'{x:.3g}') for x in lr_grid_for(*_condition))}) ---")
    CALIBRATION[_condition] = calibrate(_condition)
    precondition_status[f"P0({_condition[0]},{_condition[1]})"] = CALIBRATION[_condition]["interior"]
CALIBRATION_SECONDS = time.time() - _t0_calibration
for _condition, _c in CALIBRATION.items():
    _row = "、".join(f"{lr:.3g}: {r['seen']['accuracy']:.4f}" for lr, r in sorted(_c["records"].items()))
    print(
        f"{_tag}{_condition[0]} N={_condition[1]}: 既知の組み合わせの検証集合の正解率 {{{_row}}} -> 選んだ学習率 {_c['chosen']:.3g}"
        f"(拡張 {'なし' if _c['extended'] is None else format(_c['extended'], '.3g')}、内点 {_c['interior']})"
    )
_used_lrs = {k: learning_rate_for(k) for k in PLAN}
print(
    f"{_tag}本番で使う学習率(較正の方式 {CALIBRATION_MODE!r}、NegCLIP は S の値): "
    + "、".join(f"{k[0]} N={k[1]}: {v:.3g}" for k, v in sorted({(k[0], k[1]): v for k, v in _used_lrs.items()}.items()))
)
print(
    f"{_tag}前提条件 P0: " + "、".join(f"{k} = {v}" for k, v in precondition_status.items() if k.startswith("P0"))
    + f"、較正の実行時間 {CALIBRATION_SECONDS / 60:.1f} 分"
)
```

    --- softmax N=256 の較正(格子 (0.001, 0.002, 0.004)) ---
      較正 softmax N=256 s=90 lr=0.001: 既知の検証 0.7538、最終の訓練損失 0.143、exp(t) 24.4、飛ばしたステップ 0、93s
      較正 softmax N=256 s=90 lr=0.002: 既知の検証 0.8352、最終の訓練損失 0.078、exp(t) 32.5、飛ばしたステップ 0、94s
      較正 softmax N=256 s=90 lr=0.004: 既知の検証 0.6021、最終の訓練損失 0.224、exp(t) 39.5、飛ばしたステップ 0、94s
    --- sigmoid N=256 の較正(格子 (0.0005, 0.001, 0.002)) ---
      較正 sigmoid N=256 s=90 lr=0.0005: 既知の検証 0.3840、最終の訓練損失 1.388、exp(t) 12.2、飛ばしたステップ 0、93s
      較正 sigmoid N=256 s=90 lr=0.001: 既知の検証 0.4997、最終の訓練損失 1.132、exp(t) 14.1、飛ばしたステップ 0、94s
      較正 sigmoid N=256 s=90 lr=0.002: 既知の検証 0.5443、最終の訓練損失 1.005、exp(t) 16.7、飛ばしたステップ 0、94s
      較正(拡張) sigmoid N=256 s=90 lr=0.004: 既知の検証 0.3760、最終の訓練損失 1.336、exp(t) 16.4、飛ばしたステップ 0、94s
    softmax N=256: 既知の組み合わせの検証集合の正解率 {0.001: 0.7538、0.002: 0.8352、0.004: 0.6021} -> 選んだ学習率 0.002(拡張 なし、内点 True)
    sigmoid N=256: 既知の組み合わせの検証集合の正解率 {0.0005: 0.3840、0.001: 0.4997、0.002: 0.5443、0.004: 0.3760} -> 選んだ学習率 0.002(拡張 0.004、内点 True)
    本番で使う学習率(較正の方式 'representative'、NegCLIP は S の値): negclip N=256: 0.002、sigmoid N=16: 0.00025、sigmoid N=256: 0.002、softmax N=16: 0.00025、softmax N=256: 0.002
    前提条件 P0: P0(softmax,256) = True、P0(sigmoid,256) = True、較正の実行時間 10.9 分


### 6.6 本番の学習と評価(実験 A・B・C)

6.4 節で選んだ段階の学習(`PLAN`)をすべて行い、辞書`RECORDS`に鍵ごとに記録する。同じ鍵の学習は 1 回だけ行う(標準条件 S を
実験 A・B・C で共有する)。学習したモデルは評価の直後に破棄する(アップロードの対象の NegCLIP・シード 0 のみ、重みを CPU に残す)。


```python
_t0_training = time.time()
RECORDS: dict[tuple, dict] = {}
for _i, _key in enumerate(PLAN):
    RECORDS[_key] = run(_key, learning_rate_for(_key), "main", keep_state=(_key == UPLOAD_KEY))
    print(f"{_tag}[{_i + 1}/{len(PLAN)}] {describe(RECORDS[_key])}")
TRAINING_SECONDS = time.time() - _t0_training
print(f"本番の学習と評価: {len(RECORDS)} 学習、{TRAINING_SECONDS / 60:.1f} 分")


def final_of(key: tuple) -> dict:
    return RECORDS[key]["final"]


def zero_shot_accuracy(key: tuple) -> float:
    return final_of(key)["unseen_retrieval"]["accuracy"]


def seen_accuracy_mean(loss_type: str, batch_size: int, seeds) -> float:
    return float(np.mean([RECORDS[key_of(loss_type, batch_size, s)]["seen"]["accuracy"] for s in seeds]))
```

    [1/15] softmax N=16 s=0 lr=0.00025: 既知の検証 0.3625、M 0.4256、2 択 swap 0.9998・rand 0.9978、最終の訓練損失 0.017、exp(t) 32.1、飛ばしたステップ 0、648s
    [2/15] sigmoid N=16 s=0 lr=0.00025: 既知の検証 0.2909、M 0.3486、2 択 swap 0.9992・rand 0.9962、最終の訓練損失 0.244、exp(t) 13.8、飛ばしたステップ 0、653s
    [3/15] softmax N=256 s=0 lr=0.002: 既知の検証 0.8272、M 0.7563、2 択 swap 0.9991・rand 0.9989、最終の訓練損失 0.059、exp(t) 33.6、飛ばしたステップ 0、95s
    [4/15] sigmoid N=256 s=0 lr=0.002: 既知の検証 0.5031、M 0.5273、2 択 swap 0.9989・rand 0.9986、最終の訓練損失 1.038、exp(t) 15.6、飛ばしたステップ 0、95s
    [5/15] softmax N=16 s=1 lr=0.00025: 既知の検証 0.3521、M 0.4256、2 択 swap 0.9992・rand 0.9984、最終の訓練損失 0.034、exp(t) 32.0、飛ばしたステップ 0、675s
    [6/15] sigmoid N=16 s=1 lr=0.00025: 既知の検証 0.2348、M 0.3379、2 択 swap 0.9995・rand 0.9978、最終の訓練損失 0.355、exp(t) 13.5、飛ばしたステップ 0、669s
    [7/15] softmax N=256 s=1 lr=0.002: 既知の検証 0.7604、M 0.7054、2 択 swap 0.9983・rand 0.9986、最終の訓練損失 0.118、exp(t) 33.5、飛ばしたステップ 0、95s
    [8/15] sigmoid N=256 s=1 lr=0.002: 既知の検証 0.5471、M 0.5377、2 択 swap 0.9981・rand 0.9983、最終の訓練損失 1.026、exp(t) 16.2、飛ばしたステップ 0、95s
    [9/15] softmax N=16 s=2 lr=0.00025: 既知の検証 0.3314、M 0.4124、2 択 swap 0.9997・rand 0.9981、最終の訓練損失 0.081、exp(t) 33.1、飛ばしたステップ 0、653s
    [10/15] sigmoid N=16 s=2 lr=0.00025: 既知の検証 0.2084、M 0.3043、2 択 swap 1.0000・rand 0.9973、最終の訓練損失 0.572、exp(t) 13.4、飛ばしたステップ 0、654s
    [11/15] softmax N=256 s=2 lr=0.002: 既知の検証 0.7791、M 0.7365、2 択 swap 0.9989・rand 0.9997、最終の訓練損失 0.093、exp(t) 33.6、飛ばしたステップ 0、95s
    [12/15] sigmoid N=256 s=2 lr=0.002: 既知の検証 0.5925、M 0.5788、2 択 swap 0.9980・rand 0.9986、最終の訓練損失 0.936、exp(t) 15.9、飛ばしたステップ 0、95s
    [13/15] negclip N=256 s=0 lr=0.002: 既知の検証 0.8691、M 0.7805、2 択 swap 0.9987・rand 0.9992、最終の訓練損失 0.081、exp(t) 34.7、飛ばしたステップ 0、118s
    [14/15] negclip N=256 s=1 lr=0.002: 既知の検証 0.8064、M 0.7249、2 択 swap 0.9994・rand 0.9995、最終の訓練損失 0.145、exp(t) 34.4、飛ばしたステップ 0、119s
    [15/15] negclip N=256 s=2 lr=0.002: 既知の検証 0.8560、M 0.7814、2 択 swap 0.9991・rand 0.9992、最終の訓練損失 0.094、exp(t) 35.5、飛ばしたステップ 0、119s
    本番の学習と評価: 15 学習、81.3 分




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/05_vision_language/020_clip_contrastive_learning.ipynb
