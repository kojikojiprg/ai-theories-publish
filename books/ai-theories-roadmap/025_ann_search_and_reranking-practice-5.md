---
title: "ANN 検索とリランキング / ANN Search and Reranking(実装・実験編 5/8)"
---

この記事は後編(実装・実験編 5/8)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/025_ann_search_and_reranking-practice-4)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/025_ann_search_and_reranking-practice-6)。

### 6.2 パイロットの記録(本番実行の前)

本番の前に、ローカル(Apple Silicon の CPU。索引の計算は CPU で行う)で、近似最近傍探索の設定(HNSW の $M$ と格子、転置ファイルの格子、Product Quantization の符号長)と、基準の条件の値のシード間のばらつきを測った。
パイロットのスクリプトは使い捨てのもので、リポジトリには置いていない。この節の数値は、そのスクリプトの出力を転記したものである。**時間はローカルのもので、T4 の見込みではない**(T4 の時間は 6.3 節で本番の冒頭に測る)。
並べ替えモデル(cross-encoder・dual encoder)の学習は、D1・D2・E2 を中心の学習率・$T = 1024$ の 1 シードで **1 回ずつ**(計 3 回)測っただけである(6.2.6 節)。**判定基準・前提条件の変更の記録は 6.2.7 節にある。**

#### 6.2.1 測った量と、測らなかった量

**測った量**(いずれも、対比量の向きの情報を含まない量。ただし下の「設計の過程で見てしまった量」を除く):

1. **HNSW の格子**: 探索の幅 $ef = 11$(格子の最小点)での recall を、$M$ と $N$ の組ごとに(6.2.2 節)。選んだ $M = 5$ で、全水準($N = 1024, \dots, 65536$)の格子が目標の recall を挟むこと。
2. **基準の条件の値のシード間のばらつき**: HNSW・転置ファイルの $\log c$(目標の recall での費用の対数)のシード間の標準偏差と、単一の索引のブートストラップの標準偏差($N = 8192$、5 シード)。**平均は印字せず、標準偏差だけを記録した。**
   転置ファイルの探索するリストの数 $p = 1$ での recall の範囲(格子が目標を挟むことの確認)。
3. **Product Quantization の符号長**: $m = 16, 32, 64$ の、対称距離計算の recall と量子化の平均二乗誤差($N = 32768$、1 シード)。選んだ $m = 32$ の対称距離計算の recall のシード間の標準偏差と、単一の条件のブートストラップの標準偏差
   ($N = 16384, 32768, 65536$ のすべての $N_{\max}$ の候補、5 シード)。
4. **第 1 段の性質**: 評価用・検証用の記事の全 query での、上位 $K$ 件に正例が入る割合(coverage)と、並べ替えなしの Recall@10(6.2.5 節)。
5. **並べ替えモデルの学習**(D1・D2・E2 を 1 回ずつ。中心の学習率・$T = 1024$・シード番号 92): 訓練損失の推移、最後の区間の訓練損失と同じ組での学習前のモデルの損失の差(標準誤差つき)、clipping の発動率、
   較正の query での Recall@10 と平均逆順位(学習前のモデルの値と学習後の値、記事を単位とするブートストラップの標準偏差)(6.2.6 節)。**条件どうしの差は計算も印字もしていない。**

**測らなかった量**: 並べ替えモデルの学習の、中心以外の学習率・$T = 1024$ 以外での値、基準の条件のシード間のばらつき(測ったのは 1 シード)、評価用の記事の query での値、Product Quantization の非対称距離計算の recall、
転置ファイルの目標の recall での費用、実験 A・B・C・D・E の対比量。

**設計の過程で見てしまった量**: HNSW の $M$ を決める過程で、HNSW の recall と **費用**(1 query あたりの距離計算の回数)の曲線を、$N = 1024$ から 65536 までの複数の $N$ と複数の $M$ で見た。目標の recall での費用が $N$ とともに緩やかに増える
(約 64 倍の $N$ に対して数倍)ことを、本番の前に知った。これは実験 A の対比量の符号と大きさの目安にあたる。判定基準(対比量の定義・閾値の導出・期待する向き)は、それを見て変えていない。$M$ は、格子が目標の recall を挟むという一点だけで選んだ(6.2.2 節)。

**表から条件間の大小が読み取れること**: 6.2.6 節の表には、条件ごと(D1・D2・E2)の値が並んでおり(2 回のパイロットとも)、1 シード・中心の学習率での条件間の大小が、本番の前に読み取れる状態になっている。
条件どうしの差は計算も印字していないが、並んだ値から読み取れる。判定基準(対比量の定義・閾値の導出式・期待する差の向き)は、それを見て変えていない。2 回目の修正(6.2.7 節)は、条件間の差の向きに依らない理由によるものである。

#### 6.2.2 HNSW の $M$ と格子

$M = 16$(広く使われる値)では、小さい $N$ で、探索の幅の格子の最小点(下限の $ef = k + 1 = 11$)でも recall が目標の 0.9 を超え、**目標の recall を挟む格子が作れない**(格子の最小点で既に超えると、費用の補間が外挿になる)。
$M$ を小さくすると、探索の幅が小さいときの recall が下がる(原論文も、小さい $M$ は低い recall に、大きい $M$ は高い recall に向くとする)。探索の幅 $ef = 11$ での recall(全探索の上位 10 件との一致率、query 400 個、1 シード、$efConstruction = 100$)は次のとおり。

| $M$ | $N = 1024$ | 2048 | 4096 | 8192 | 16384 | 32768 | 65536 |
|---|---|---|---|---|---|---|---|
| 4 | - | 0.783 | - | 0.683 | - | 0.564 | - |
| 5 | 0.875 | 0.836 | 0.790 | - | 0.686 | - | 0.609 |
| 6 | 0.901 | 0.876 | 0.840 | 0.791 | 0.745 | 0.713 | 0.667 |
| 8 | - | 0.912 | - | 0.845 | - | 0.787 | - |
| 12 | - | 0.953 | - | 0.902 | - | 0.858 | - |
| 16(ef = 16) | 0.990 | 0.984 | 0.974 | 0.966 | 0.946 | - | - |

($-$ は測っていない。$M = 16$ の行は、探索の幅が 16 の値で(query 500 個)、$ef = 11$ ではさらに低いが測っていない。)すべての水準で $ef = 11$ の recall が 0.9 を下回るのは $M \le 5$ である($M = 6$ は $N = 1024$ で 0.901)。
$M = 4$ は原論文が示す妥当な範囲(5〜48)の外なので、**範囲の下限の $M = 5$ を選んだ**。選んだ $M = 5$ で、本番と同じ query・格子(1 シード、$N = 1024, \dots, 65536$)の掃引が目標の recall を挟むことを確かめた。

| $N$ | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 | 65536 |
|---|---|---|---|---|---|---|---|
| 格子の最小点($ef = 11$)の recall | 0.868 | 0.829 | 0.801 | 0.756 | 0.706 | 0.672 | 0.620 |
| 掃引の終点の $ef$(recall が 0.95 以上になった最小の点) | 28 | 35 | 44 | 55 | 88 | 111 | 140 |

全水準で、格子の最小点の recall は目標(0.9)未満、終点の recall は 0.95 以上で、費用を補間できる。最小点と目標の差が最も小さい $N = 1024$ で 0.032 である(この水準が選ばれるのは、実行計画で $N_{\max} = 16384$ になる場合)。
**選んだ $M = 5$ は、広く使われる値(16 など)より小さく、原論文が高い recall に向くとする大きい $M$ より HNSW に不利な側にある**(実験 B の「解釈の注意」)。

#### 6.2.3 Product Quantization の符号長

$m = 16, 32, 64$($k^* = 256$、$N = 32768$、1 シード)の、対称距離計算の recall と、量子化の平均二乗誤差は次のとおり。

| $m$ | 符号(バイト) | 圧縮率 | 対称距離計算の recall | 量子化の平均二乗誤差 $D$ |
|---|---|---|---|---|
| 16 | 16 | 64 倍 | 0.2509 | 0.1478 |
| 32 | 32 | 32 倍 | 0.4234 | 0.0997 |
| 64 | 64 | 16 倍 | 0.6824 | 0.0364 |

どの $m$ でも床にも天井にも届いていない。**$m = 32$ を選んだ**(圧縮率 32 倍で、対称距離計算の recall が中程度。非対称距離計算に改善の余地を残す)。選んだ $m = 32$ の対称距離計算の recall のシード間の変動は 6.2.4 節。

#### 6.2.4 基準の条件のばらつきと検出力の見込み(実験 A・B・C)

**HNSW・転置ファイル**($N = 8192$、5 シード(シード番号 100〜104。実験のシード $0, 1, \dots$ と共有しない)、query 890 個、HNSW は $M = 5$・$efConstruction = 100$、転置ファイルはリスト数 90 = $\mathrm{round}(\sqrt{8192})$):

| 量 | HNSW | 転置ファイル(リスト数 90) |
|---|---|---|
| $\log c$ のシード間の標準偏差 | 0.0149 | 0.0045 |
| $\log c$ の、単一の索引のブートストラップ(記事を単位、2,000 回)の標準偏差(5 索引の平均) | 0.0341 | 0.0328 |

転置ファイルの探索するリストの数 $p = 1$ での recall の範囲は 0.532〜0.556(目標 0.9 未満で、格子が目標を挟む)。

**Product Quantization の対称距離計算**($m = 32$、$k^* = 256$、5 シード、query 890 個):

| $N$ | 16384 | 32768 | 65536 |
|---|---|---|---|
| recall の範囲(5 シード) | 0.466〜0.478 | 0.420〜0.431 | 0.376〜0.392 |
| シード間の標準偏差 | 0.0045 | 0.0038 | 0.0057 |
| 単一の条件のブートストラップの標準偏差(5 シード平均) | 0.0054 | 0.0055 | 0.0060 |

**検出力の見込み**(各実験の宣言の「検出力の事前確認」に導出):

| 実験 | 見込みの $\sigma$(上限) | 閾値の見込み($2\sigma$) | 備考 |
|---|---|---|---|
| A | 約 0.016(下限は約 0.003) | 約 0.032(下限は約 0.006) | 傾きの標準偏差。水準間の誤差が独立なら上限 |
| B | 約 0.048 | 約 0.095 | $\log$ の費用の差。2 つの索引の評価の誤差が独立なら上限 |
| C | 約 0.0092 | 約 0.018 | 2 条件の評価の変動が独立なら上限 |

いずれも、**効果がない場合の見積もり** ではなく、2 つの量の誤差が独立という **上限(悲観的な見積もり)** である。同じ query・同じ正解・同じ部分集合で評価するので、実際はこれより小さい見込みで、
下限は 2 つの量の評価の変動が完全に相関する場合にあたる(A・B ではシード間の項だけ)。

#### 6.2.5 第 1 段の性質(024 のモデルの全探索)

評価用の記事の全 query(3,244 個)と検証用の記事の全 query(1,019 個)で、コーパス全体(94,486 passage)の全探索の上位 $K$ 件に、同じ記事の passage(query の元の passage を除く)が 1 つ以上入る割合(coverage):

| $K$ | 10 | 20 | 30 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|
| 評価用の記事 | 0.2312 | 0.3024 | 0.3502 | 0.4180 | 0.5040 | 0.5968 |
| 検証用の記事 | 0.2738 | 0.3543 | 0.4014 | 0.4612 | 0.5604 | 0.6536 |

$K = 10$ の値が、並べ替えなしの Recall@10 である。評価用の記事で、$K = 50$ の coverage は 0.4180、並べ替えなしの Recall@10 は 0.2312 で、並べ替えで到達できる余地は 0.187(前提条件 P3 の閾値は coverage 0.35・余地 0.10)。
正例が上位 $K$ 件に入る数の平均は、$K = 50$ で 1.18 個である。024 の評価用の索引(3,244 passage)での Recall@10(0.6477)に比べて、コーパス(94,486 passage)が約 29 倍大きいので、Recall@10 は大きく下がる。
本番の query(評価用の記事から等間隔に選んだ約 1,500 個)での値は、5.3 節の出力にある。

#### 6.2.6 並べ替えモデルの学習の最小のパイロット

**手順(2 回とも共通)**: D1・D2・E2 を、それぞれ **1 回ずつ**(計 3 回)、中心の学習率 $2.4 \times 10^{-4}$・$T = 1024$・シード番号 92(実験のシード $0, 1, \dots$・較正のシード 90・計測のシード 91 と共有しない)で実行した。
**1 つのプロセスで逐次に** 実行し(並列には実行しない)、学習のたびにモデルを破棄してデバイスのキャッシュを空けた。索引の構築、他の学習率、他の $T$ は行っていない。本番と同じ`train_run()`(5.5 節)と、本番の水準の設定
($T = 1024$ の学習の組、較正の query 877 個(検証用の記事 100 本)、ブートストラップ 10,000 回)で、開始から 40 分を超えたら中断する条件で実行した。
測ったのは、対比量の向きの情報を含まない量(条件ごとの訓練損失と較正の query での指標、学習前のモデルの同じ量、ブートストラップの標準偏差、clipping の発動率)だけで、**条件どうしの差は計算も印字していない**。

パイロットは 2 回行った。**1 回目(6.2.6.1 節)は、改める前の設計での値** で、2 回目(6.2.6.2 節)は、6.2.7 節の 2 回目の修正を入れた後の値である。**6.1 節の現在の宣言に対応するのは 2 回目の値** である。
1 回目の値は、2 回目の修正の根拠(6.2.7 節)として残す。

##### 6.2.6.1 改める前の設計での値(1 回目)

学習に使う query は学習用の query のすべてで、正例は同じ記事の passage から一様に選び、同点のスコアは第 1 段の順位が先の候補を上位とする設計での値である(6.2.7 節の 2 回目の修正の前)。
所要は 21.3 分(40 分の上限内)で、1 ステップの時間(ローカルの MPS)は、D1 が 315 ms、D2 が 233 ms、E2 が 293 ms だった。

**学習の経過**(訓練損失はステップ区間の平均。一様な予測の損失は $\ln(1 + n) = 2.0794$):

| ステップ区間 | D1 | D2 | E2 |
|---|---|---|---|
| 0〜15 | 2.1292 | 4.5930 | 2.1291 |
| 16〜31 | 2.0671 | 4.1880 | 2.0630 |
| 32〜63 | 2.0992 | 2.9825 | 2.1037 |
| 64〜127 | 2.0820 | 2.1109 | 2.0816 |
| 128〜255 | 2.0769 | 2.0829 | 2.0201 |
| 256〜511 | 2.0026 | 2.0807 | 1.8259 |
| 512〜767 | 1.8711 | 2.0803 | 1.6910 |
| 768〜1023 | 1.8019 | 2.0801 | 1.5928 |

**前提条件 P1(a)**(最後の 51 ステップ。6.1 節の定義):

| 量 | D1 | D2 | E2 |
|---|---|---|---|
| 最後の区間の訓練損失 | 1.8257 | 2.0801 | 1.5924 |
| 同じ組での学習前のモデルの損失 | 2.1093 | 4.6674 | 2.1069 |
| 差の平均(標準誤差) | -0.2837(0.0434) | -2.5872(0.1404) | -0.5145(0.0448) |
| 差の平均 / 標準誤差(閾値 -2) | -6.5 | -18.4 | -11.5 |
| 最後の区間の訓練損失は $\ln(1 + n) = 2.0794$ 未満か | 未満 | **未満でない**(2.0794 を 0.0007 上回る) | 未満 |
| clipping の発動率(更新のスキップ) | 0.992(0) | 0.067(0) | 1.000(0) |
| **P1(a)** | **成立** | **不成立** | **成立** |

D2 は、学習前の損失が $\ln(1 + n)$ の約 2.2 倍(4.667)で、最初の 128 ステップで $\ln(1 + n)$ の近く(2.11)まで下がった後、2.08 に張り付いた(最後の区間の損失は 2.0801)。差の平均は標準誤差の 18.4 倍で、「学習前より下がった」側の基準は満たしている。
$\ln(1 + n)$ 未満の側の基準は満たしていない。**この不成立の原因は測っていない。**

**前提条件 P1(b)**(較正の query、検証用の記事 100 本の 877 個。ブートストラップの標準偏差は記事を単位とする 10,000 回の再標本。基準の条件は D2 と E2。D1 は参考として測った):

| 量 | D1(参考) | D2 | E2 |
|---|---|---|---|
| 学習前のモデルの Recall@10(標準偏差) | 0.1904(0.0206) | 0.2611(0.0250) | 0.1904(0.0206) |
| 学習後のモデルの Recall@10(標準偏差) | 0.1733(0.0177) | 0.2645(0.0253) | 0.2246(0.0235) |
| 学習前のモデルの平均逆順位(標準偏差) | 0.0710(0.0078) | 0.1521(0.0188) | 0.0710(0.0078) |
| 学習後のモデルの平均逆順位(標準偏差) | 0.0731(0.0098) | 0.1545(0.0189) | 0.1007(0.0121) |
| Recall@10 の増分(学習後 − 学習前、対応付き)(標準偏差) | -0.0171(0.0205) | +0.0034(0.0101) | +0.0342(0.0193) |
| 増分 / 標準偏差(閾値 2) | -0.8 | +0.3 | +1.8 |
| 平均逆順位の増分(標準偏差) | +0.0021(0.0108) | +0.0024(0.0062) | +0.0297(0.0110) |
| **P1(b)**(単一シードなので、シード間の項は 0) | 不成立 | **不成立** | **不成立** |

D1 と E2 の学習前のモデルは、ヘッドの初期値が同じシード番号で同一なので、学習前の値が一致している(整合の確認)。**P1(b)の標準偏差は、本番の判定と同じ式 $\sigma_g = \sqrt{\mathrm{Var}_s(g_s) / n + \sigma_{\mathrm{boot}}^2}$ のうちブートストラップの項だけ** で、
本番の 5 シード(または 3 シード)の平均ではシード間の項が加わる。したがって、この表の P1(b)は、本番の P1(b)の成否を直接示すものではない。

##### 6.2.6.2 改めた後の設計での値(2 回目)

6.2.7 節の 2 回目の修正(学習に使う query と正例、同点の扱い)を入れた後の値である(較正の候補の規則・実行計画・公開の条件は、このパイロットの学習の経路に関わらない)。
所要は 21.1 分(40 分の上限内)で、1 ステップの時間(ローカルの MPS)は、D1 が 298 ms、D2 が 233 ms、E2 が 294 ms だった。
学習に使える query は 39,303 個(学習用の query 89,803 個の 43.77%)で、$T = 1024$ は 8,192 個の query、使える query に対して 0.208 エポックにあたる。
参考として、較正の query(877 個)での **並べ替えなし**(第 1 段の順位のまま、$K = 50$)の値は、Recall@10 が 0.2725(ブートストラップの標準偏差 0.0233)、平均逆順位が 0.1611(同 0.0179)、coverage が 0.4675 である。

**学習の経過**(訓練損失はステップ区間の平均。一様な予測の損失は $\ln(1 + n) = 2.0794$):

| ステップ区間 | D1 | D2 | E2 |
|---|---|---|---|
| 0〜15 | 2.0859 | 2.1849 | 2.0891 |
| 16〜31 | 2.0968 | 1.9264 | 2.0968 |
| 32〜63 | 2.0761 | 2.0667 | 1.9680 |
| 64〜127 | 2.0963 | 1.9615 | 1.7835 |
| 128〜255 | 2.0812 | 1.9417 | 1.3760 |
| 256〜511 | 2.0755 | 1.8506 | 1.0190 |
| 512〜767 | 2.0786 | 1.7891 | 0.7409 |
| 768〜1023 | 2.0703 | 1.7866 | 0.6018 |

**前提条件 P1(a)**(最後の 51 ステップ):

| 量 | D1 | D2 | E2 |
|---|---|---|---|
| 最後の区間の訓練損失 | 2.0619 | 1.7423 | 0.5634 |
| 同じ組での学習前のモデルの損失 | 2.0988 | 1.9974 | 2.1016 |
| 差の平均(標準誤差) | -0.0370(0.0098) | -0.2552(0.0401) | -1.5382(0.0473) |
| 差の平均 / 標準誤差(閾値 -2) | -3.8 | -6.4 | -32.5 |
| 最後の区間の訓練損失は $\ln(1 + n) = 2.0794$ 未満か | 未満 | 未満 | 未満 |
| clipping の発動率(更新のスキップ) | 0.370(0) | 1.000(0) | 1.000(0) |
| 診断量: 学習後のモデルの、query ごとの候補 50 件のスコアの標準偏差の中央値(較正の query) | 0.0874 | 0.0383 | 1.3071 |
| **P1(a)** | **成立** | **成立** | **成立** |

D1 の最後の区間の訓練損失 2.0619 は、閾値 $\ln(1 + n) = 2.0794$ を 0.0175(= 2.0794 - 2.0619)下回るだけで、成立の余裕は小さい(差の平均は標準誤差の 3.8 倍)。

**前提条件 P1(b)**(較正の query、検証用の記事 100 本の 877 個。ブートストラップの標準偏差は記事を単位とする 10,000 回の再標本。基準の条件は D2 と E2。D1 は参考として測った):

| 量 | D1(参考) | D2 | E2 |
|---|---|---|---|
| 学習前のモデルの Recall@10(標準偏差) | 0.1870(0.0204) | 0.2588(0.0251) | 0.1870(0.0204) |
| 学習後のモデルの Recall@10(標準偏差) | 0.1893(0.0204) | 0.2805(0.0249) | 0.2121(0.0229) |
| 学習前のモデルの平均逆順位(標準偏差) | 0.0699(0.0078) | 0.1519(0.0188) | 0.0699(0.0078) |
| 学習後のモデルの平均逆順位(標準偏差) | 0.0655(0.0074) | 0.1699(0.0188) | 0.1019(0.0146) |
| Recall@10 の増分(学習後 − 学習前、対応付き)(標準偏差) | +0.0023(0.0192) | +0.0217(0.0104) | +0.0251(0.0176) |
| 増分 / 標準偏差(閾値 2) | +0.1 | +2.1 | +1.4 |
| 平均逆順位の増分(標準偏差) | -0.0044(0.0088) | +0.0180(0.0078) | +0.0320(0.0132) |
| **P1(b)**(単一シードなので、シード間の項は 0) | 不成立 | **成立** | **不成立** |

D1 と E2 の学習前のモデルは、ヘッドの初期値が同じシード番号で同一なので、学習前の値が一致している(整合の確認)。学習前のモデルの値が 6.2.6.1 節と少し異なる(例: D1 と E2 の Recall@10 が 0.1904 と 0.1870)のは、同点の扱いを改めたため
かもしれないが、確かめていない。P1(b)の標準偏差は、6.2.6.1 節と同じく、本番の判定と同じ式のうちブートストラップの項だけで、本番の 5 シード(または 3 シード)の平均ではシード間の項が加わる。
この表の P1(b)は、本番の P1(b)の成否を直接示すものではない。

##### 6.2.6.3 検出力の見込みと限界

**検出力の見込みの更新(実験 D・E)**: 単一の条件の較正の query での Recall@10 のブートストラップの標準偏差は 0.0177〜0.0253(6.2.6.1 節と 6.2.6.2 節の表の 12 値)で、検証用の記事は 100 本である。評価用の記事は 300 本なので、$\sqrt{100/300} = 0.577$ 倍して約 0.010〜0.015
(評価の query は記事あたり 5 個で、検証用の 10 個より少ないので、やや大きくなりうる)。2 条件の評価の変動が独立で打ち消しがまったく起きない場合の、対比量(シード平均の差)の評価の変動の項の上限は約 $\sqrt{2} \times 0.015 = 0.021$、閾値は約 0.042 である。
参考として、同じ条件の学習後と学習前のモデルの Recall@10 の対応付きの差の標準偏差(2 回のパイロットの 6 値で 0.0101〜0.0205)を 0.577 倍すると 0.006〜0.012 で、2 つのモデルの評価の変動が相関する場合の大きさの目安にあたる(ただし、2 つの学習したモデルの差ではない)。
**同じ条件の 2 回の学習の差の標準偏差は測っておらず、シード間の項も測っていない(測ったのは 1 シード)。** この見積もりは、評価の変動の項だけの範囲(約 0.006〜0.021)で、設計の判断は範囲の全体で行う(各実験の宣言)。

**限界と、設計を変えなかったこと**:

- 測ったのは、中心の学習率・$T = 1024$・1 シードだけである。**6.2.6.1 節の D2 の P1(a)の不成立が中心の学習率に特有か、較正の格子の他の学習率でも起こるかは測っていない**(格子は中心の $1/16$ 倍から 16 倍)。6.2.6.2 節では、D2 を含む 3 条件が P1(a)を満たしたが、
  D1 の余裕は小さく、基準の条件の P1(b)は D2 だけが満たした(E2 は標準偏差の 1.4 倍)。$T = 512, 256$ では測っていない。
  学習の成立の前提条件の閾値が、全ての $T$ の候補と較正で選ばれうる全ての学習率で成り立つかは、測っていない。
- **成り立たなかった前提条件があっても、判定基準・前提条件の定義・学習率の中心と格子は変えていない**(6.2.7 節で改めたものを除く)。 本番で前提条件が成り立たなければ、その実験は「前提不成立」と記録される(6.1 節)。
- **起こりうる結果**: 前提不成立(P0・P1。仮説に関する情報は得られない)、判定不能(効果が閾値に届かない)、支持・反証。**本番は 1 回で完結する** ので、前提不成立や判定不能の場合も再実行せず、結果をそのまま報告する
  (前提不成立の場合の、条件のみの修正を伴う再実行は、結果の方向に依存しない根拠がある場合に限る)。

##### 6.2.6.4 本番の値との対応(本番の実行の後に追記)

6.2.6.1〜6.2.6.3 節の記録は本番の前のままで、この項だけを本番の実行の後に加えた。パイロット(6.2.6.2 節。中心の学習率 $2.4 \times 10^{-4}$・$T = 1024$・1 シード)と、本番(較正で選ばれた学習率・$T = 1024$・5 シード。7.1 節・7.2 節)の、
学習の成立の成分の対応を示す。パイロットで測っていない学習率が、D1(中心の 4 倍)と D2(中心の 1/4 倍)で選ばれた。E2 は中心が選ばれた。

| 条件 | 学習率 | 最後の区間の訓練損失 | 成分 (a)(閾値: 差の平均が標準誤差の 2 倍以上低い、かつ $\ln 8 = 2.0794$ 未満) | 成分 (b)(増分 / 標準偏差、閾値 2) |
|---|---|---|---|---|
| D1 パイロット | $2.4 \times 10^{-4}$ | 2.0619 | 成立(差の平均 / 標準誤差 -3.8。最後の区間の損失と $\ln 8$ の差 0.0175) | +0.1(参考。基準の条件ではない) |
| D1 本番 | $9.6 \times 10^{-4}$ | 2.059〜2.078(平均 2.072) | シード 1・2 で不成立(差の平均 / 標準誤差の最小 1.2)。損失の最大 2.078 は閾値 2.0794 未満(差は約 0.001) | (基準の条件ではない) |
| D2 パイロット | $2.4 \times 10^{-4}$ | 1.7423 | 成立(-6.4) | +2.1(成立) |
| D2 本番 | $6 \times 10^{-5}$ | 1.821〜1.919(平均 1.868) | 5 シードとも成立 | 2.5(成立。シード平均の増分 +0.0180) |
| E2 パイロット | $2.4 \times 10^{-4}$ | 0.5634 | 成立(-32.5) | +1.4(不成立) |
| E2 本番 | $2.4 \times 10^{-4}$ | 0.437〜0.657(平均 0.547) | 5 シードとも成立 | 0.8(不成立。シード平均の増分 +0.0176) |

- **D1 の成分 (a)**: パイロットで余裕が小さいと記録した(閾値との差 0.0175)とおり、本番で、中心の 4 倍の学習率の 5 シードのうち 2 シードが、差の平均が標準誤差の 2 倍以上低いという基準を満たさなかった。
  パイロットは 1 シード・中心の学習率だけだったので、較正で選ばれた学習率での成立の余裕は測っていなかった。
- **E2 の成分 (b)**: パイロットの 1.4 は単一シードのブートストラップの項だけの値で、本番はシード間の項が加わって 0.8 になった(シードごとの増分は +0.0525 から -0.0251)。パイロットの時点で、基準の 2 を下回っていた。
- **D2 の成分 (b)**: パイロットの 2.1 に対して本番は 2.5 で、どちらも成立した。閾値 2 に対する余裕は小さい。
- **較正の結果**: 3 条件とも選ばれた学習率が格子の内点で、格子の拡張は起きなかった(6.5 節の出力。7.1 節)。P1(a)を満たさず候補から外れた学習率は、D1 の $1.5 \times 10^{-5}$ と D2 の $3.84 \times 10^{-3}$ の 2 点だった。
- 本番の計画は $T = 1024$ だったので、$T = 512, 256$ での閾値の成立は、本番でも測られていない。

#### 6.2.7 判定基準・前提条件・設計の変更の記録

この節は、6.1 節の宣言を改めた箇所の、旧基準・新基準・理由の記録である。改めたのは 2 回で、**どの変更も、観測した結果(条件の値・差・判定)の向きに依らない一般論によるもの** である。
判定基準(対比量の定義・閾値の導出式・期待する差の向き)は変えていない。変えたのは、前提条件の定義、学習率の較正の指標と候補、学習の正例の選び方、同点の扱い、実行計画、公開の条件である。

##### 1 回目の修正(6.2.6.1 節の測定の前)

| 項目 | 旧 | 新 | 理由 |
|---|---|---|---|
| P1(b)(学習の成立、基準の条件の較正の query での Recall@10 の増分) | 基準の条件の **すべてのシードで**、増分 $\mathrm{gain} > 0$ かつ $\mathrm{gain} \ge 2 \, \sigma_{\mathrm{boot}}$(単一シードのブートストラップの標準偏差) | 基準の条件の増分の **シード平均** $\bar{g} > 0$ かつ $\bar{g} \ge 2 \, \sigma_g$。$\sigma_g = \sqrt{\mathrm{Var}_s(g_s) / n + \sigma_{\mathrm{boot}}^2}$ は判定と同じ式(シード間の項 + ブートストラップの項) | 旧基準は「すべてのシードが閾値を超える」ことを要求するので、シード数 $n$ が増えるほど成立しにくくなる(各シードが独立に確率 $q < 1$ で成り立つなら、全シードで成り立つ確率は $q^n$)。シード数が多いほど検出力が上がるはずの設計で、前提条件だけが厳しくなるのは不整合である。判定の対比量はシード平均で判定するので、前提条件も同じ単位(シード平均)と同じ形の標準偏差で定義する |
| P1(a)(学習の成立、訓練損失) | 全条件・全シードで、最後の区間の損失の差の平均が $\bar{d} \le -2 \, \mathrm{SE}$(かつ丸めの床以下) | 左に加えて、**最後の区間の訓練損失の平均が一様な予測の損失 $\ln(1 + n)$ 未満** | 旧基準は「学習前のモデルの損失より下がった」ことだけを見る。学習前の損失が一様な予測の損失より高い条件(温度 0.05 の内積をスコアとする dual encoder は、スコアの範囲が広く、学習前に一様より高い損失になりうる)では、出力を定数に近づけて損失を $\ln(1 + n)$ に落とすだけでも成立しうる。$\ln(1 + n)$ 未満は、定数の出力では成立しない(学習前の損失が一様な予測より高いことは、6.2.6.1 節の D2 で確かめた: 4.667 対 2.079) |
| 学習率の較正の指標(6.5 節) | 較正の query の並べ替えの Recall@10 | 較正の query の、第 1 段の上位 50 件を並べ替えた後の **平均逆順位**。Recall@10 とそのブートストラップの標準偏差も印字する。判定の対比量は Recall@10 のまま | Recall@10 は query ごとに 0 か 1 の二値で、検証用の集合(100 記事、877 query)では、格子点どうしの差が小さいと最良がノイズで決まる。平均逆順位は連続量で、正例の順位が 1 つ上がるだけでも値が変わるので、同じ集合で、格子点どうしの差を二値の指標より捉えやすい。変えたのは較正の選択の量で、判定の対比量(Recall@10)の定義は変わらない |
| 実験 B の転置ファイルのリスト数(3.2 節・6.1 節・6.6 節) | $\sqrt{N}$ の $\{1/2, 1, 2\}$ 倍の 3 点。探索するリストの数 $p$ の格子は 512 まで | $\sqrt{N}$ の $\{1/2, 1, 2, 4, 8\}$ 倍の 5 点(公比 2)。$p$ の格子は 2048 まで(各 $K$ 以下の点だけを使う)。前提条件 P0(B)(費用が最小のリスト数が 5 点の内点)を追加 | 最小の点が選択肢の内点であることを確かめるには、最小の点の両側に点が要る。3 点では最適が端にある場合、範囲の外にあるかを区別できない。上側を広げたのは、理論上の最適が $K = \sqrt{pN}$($p \ge 1$)で、目標の recall に必要な $p$ は 1 より大きい(転置ファイルの $p = 1$ での recall が目標未満であることを、パイロットで確かめた。6.2.4 節)ため、最適が $\sqrt{N}$ より大きい側にあるはずだからである |

##### 2 回目の修正(6.2.6.1 節の値を見た後、6.2.6.2 節の測定の前)

2 回目の修正の理由は、1 回目のパイロットの条件間の大小ではなく、(a)学習時と推論時の分布の不一致、(b)崩れたモデルが第 1 段の順位を受け継ぐ、(c)学習が成立していない学習率の採用、(d)較正の時間の大きさ、(e)公開するモデルの品質、という一般論である。
どれも、条件の差の向きがどちらでも同じ修正になる。

| 項目 | 旧 | 新 | 理由 |
|---|---|---|---|
| 学習に使う query と正例(D1・D2・E2 共通。3.7 節) | query: 学習用の記事の query すべて(89,803 個)。正例: 同じ記事の、元の passage 以外の passage から一様 | query: 採掘の上位 50 件に、同じ記事の passage(元の passage を除く)が 1 つ以上ある query(39,303 個、43.77%)。正例: その上位 50 件の中の同じ記事の passage から一様。負例の選び方は変えない | 推論時に並べ替えモデルが見る正例は、必ず第 1 段の上位 50 件の中にある。旧設計では、上位 50 件に入る同じ記事の passage は 1 query あたり平均 1.25 個(111,961 件 / 89,803 query)で、正例(同じ記事の他の passage。1 記事あたり最大 11 個)の多くは上位 50 件の外にあり、困難な負例(上位 50 件から選ぶ)の条件では、「第 1 段のスコアが低い候補が正例」という、推論時には成り立たない手がかりで訓練損失を下げられた。学習時と推論時の分布を揃える規則を、負例だけでなく正例にも適用する |
| 同点のスコアの扱い(3.9 節・評価の実装) | 第 1 段の順位が先の候補を上位とする | **正例に不利に数える**(正例の順位 = 1 + スコアが最も高い正例以上の、正例でない候補の数)。第 1 段の順位は同点の解決に使わない。全候補に同じスコアを返すモデルで Recall@10 が 0 になることを単体テストで確かめる | 定数に近い出力に崩れたモデルが第 1 段の順位を受け継ぐと、指標が並べ替えモデルの識別の力ではなく第 1 段の力を測ってしまう。診断量「ランダムな候補の中での指標」の規則と揃える |
| 学習率の較正の候補(6.5 節) | 格子点すべてから、平均逆順位が最大の点を選ぶ | **P1(a)を満たさない学習率は候補から外し**、残りから平均逆順位が最大の点を選ぶ。すべて外れた条件は P0 を不成立とし(本番の学習には格子の中心を使う)、内点かどうかは外す前の格子の位置で判定する。外した学習率と理由を 6.5 節で印字する | 学習が成立していない学習率を選ぶと、その条件の比較が意味を持たない。前提条件は、較正で決める量の採用条件としても使い、選択と検査を整合させる |
| 実行計画(6.1 節・6.4 節) | 段階 0〜5 の 6 段階 × $T$ の 3 候補 = 18 通り | 段階 6「学習率の較正の格子を 5 点から 3 点(中心の $1/4, 1, 4$ 倍、拡張 1 点)にする」を、各 $T$ の段階 5 の後(次の $T$ の段階 0 の前)に追加。計画番号 = 7 × ($T$ の番号) + 段階、21 通り | 較正は並べ替えの側の時間の大きな部分を占める。$T$ を下げると、学習が飽和していない状態での比較になり、比べているものの意味が変わる。格子の点の数を削っても、比べているものの意味は変わらない。3 点の格子では最良が端に来やすく、P0 が不成立になりうることを、宣言に記した |
| Hub へのアップロード(6.16 節) | `UPLOAD_ARTIFACTS = True`ならアップロードする | `UPLOAD_ARTIFACTS = True`でも、読み込み直した D1 のシード 0 の、評価用の query での Recall@10 が、並べ替えなしの値を上回る場合だけアップロードする。満たさない場合は行わず、その旨と両方の値を印字する。**これは判定ではなく、公開するかどうかの条件である** | 並べ替えで第 1 段より悪くなるモデルを、後続のアプリの入力として公開しないため |
| 診断量(6.5・6.7・実験 D・E) | (なし) | 学習後のモデルごとに、query ごとの候補 50 件のスコアの標準偏差の中央値を印字する(崩壊の検出。判定には使わない) | 定数に近い出力への崩壊を、指標とは別の量で検出する |

**判定基準でない変更**(宣言の説明の追加・誤りの訂正):

- 3.7 節と実験 E の「解釈の注意」を、新しい正例の選び方に合わせて書き直した。ランダムな負例の条件(E2)では、正例が上位 50 件の中・負例が外になるので、第 1 段のスコアの高さが手がかりになりうることを、実験 E の「解釈の注意」と、対比量が閾値を超えうる状態に記した。
- 6.1 節の「学習量についての注意」を、学習に使える query の数(39,303 個)に対するエポック換算(0.208・0.104・0.052)に更新した。
- 実行計画の宣言に、段階 5 か 6($N_{\max} = 16{,}384$)が選ばれると、$N = 1024$ の水準の格子の最小点の recall(パイロット: 0.868)が目標の 0.9 に近く、P1(A)が不成立になりうることを記した。
- 学習率の格子の中心の記述を、中心の学習率で 3 条件を 1 回ずつ測ったパイロット(6.2.6 節)に合わせた。中心の値は変えていない。
- 表記の訂正(3.3.1 節の略語、実行計画の表の注の括弧)と、本番の実行が要求するコミット(`REQUIRED_ANCESTOR_COMMIT`)の設定。

### 6.3 スケーリングの計測と外挿(1 セッションの見積もり)

本番でデータ量や回数が大きくなる処理を、**実行するデバイスの上で** 計測する。別のトピックで測った環境間の速度比は流用しない。

- **HNSW の構築**(CPU の処理。Colab の CPU はローカルより遅い可能性がある): 挿入した要素の数 $N$ の 3 点(本番は 2048・4096・8192、スモークテストは縮小。5.2 節)で、先頭から挿入するのに要した累積の時間を測り、べき指数 $b$ を推定して、$N_{\max}$ まで外挿する
  (見積もりは外挿した累積の時間)。**挿入の時間は $N$ に対して超線形になりうる**(1 回の挿入の探索が $\log N$ に比例して増える)ので、比例ではなくべき乗則の外挿を使う。
- **HNSW の掃引**: 同じ 3 点で、各水準の索引を格子で掃引する時間を(一部の query で)測り、べき乗則で各水準の $N$ へ外挿して合計する。
- **転置ファイル**: 同じ 3 点で、リスト数 5 通りの k-means の学習の合計時間(query の数によらない)と、掃引の時間(1 query あたり)を別々に測り、$N_{\max}$ へ外挿する。**Product Quantization**: 同じ 3 点で、符号帳の学習と符号化の時間と、探索(非対称・対称・再採点)の時間(1 query あたり)を別々に測り、$N_{\max}$ へ外挿する。
- **並べ替えモデルの学習**(1 ステップの処理が一定の反復): 条件(D1・D2・E2)ごとに、ウォームアップ 8 ステップの後、8・16・32 ステップの時間を測る。本番と同じ関数(`train_reranker()`)・同じ精度・同じバッチで測る。
  見積もりは **定常状態の 1 ステップの時間(32 ステップの計測から)をステップ数に比例させる**。べき指数 $b$ も推定し、べき乗則の外挿値と比例の値の差を印字する。
- **並べ替えの評価**: 較正の query(検証用の記事)の一部で、cross-encoder と dual encoder の 1 組あたりのスコアの計算時間を測り、較正・本番の評価・ランダムな候補・実験 F の組の数に比例させる。
- **学習前のモデルの損失の測定**(学習 1 回ごとに最後の区間のステップ数だけ、更新なしの順伝播)、**学習の組の生成**(シードごとに 1 回。ステップ数に比例): 計測して学習 1 回の時間に足す。
- **データの準備**(コーパスの取得・符号化・埋め込み・採掘): 5.3 節で段階ごとの時間とピークのメモリを印字した。これらは本番の冒頭で 1 回だけ行い、6.4 節の経過時間に含まれる。
- **判定・ブートストラップ・図・アップロードの準備**: 固定の余裕(300 秒)に含める。

計測の結果は、較正・本番の見積もり(6.4 節)に使う。**計測は結果(対比量)を作らない**(色々な設定で、学習・探索の時間だけを測り、検索指標は使わない)。


```python
REPORTING_MARGIN_SECONDS = 300.0  # 判定・ブートストラップ・図・アップロードの準備の固定の余裕
RUN_OVERHEAD_SECONDS = 2.0  # 学習 1 回あたりのモデルの構築などの固定費の余裕
TIMING_QUERIES = 150  # 掃引の計測に使う query の数(時間は ANN_QUERY_INDEX の数に比例させて換算する)
SCALING_STEP_COUNTS = (8, 16, 32)
SCALING_WARMUP_STEPS = 8
_t0_scaling = time.time()

# --- HNSW の構築と掃引 ---
_timing_index = HierarchicalNavigableSmallWorld(PASSAGE_EMBEDDINGS_NP, HNSW_MAX_CONNECTIONS, HNSW_CONSTRUCTION_WIDTH, seed=HNSW_SEED_BASE + TIMING_SEED_INDEX)
_timing_queries = ANN_QUERIES_NP[:TIMING_QUERIES]
HNSW_BUILD_CUMULATIVE, HNSW_SWEEP_SECONDS_PER_QUERY = [], []
_cumulative = 0.0
for _n in TIMING_SIZES:
    _start = time.time()
    _timing_index.add(_n - _timing_index.num_inserted)
    _cumulative += time.time() - _start
    HNSW_BUILD_CUMULATIVE.append(_cumulative)
    _subset = _timing_index.inserted_elements()
    _truth, _, _excluded = ann_truth(TIMING_SEED_INDEX, _subset)
    _start = time.time()
    sweep_hnsw_search_width(_timing_index, _timing_queries, _excluded[:TIMING_QUERIES], _truth[:TIMING_QUERIES], HNSW_GRID, NEIGHBORS, STOP_RECALL)
    HNSW_SWEEP_SECONDS_PER_QUERY.append((time.time() - _start) / TIMING_QUERIES)
HNSW_BUILD_FIT = fit_power_law_exponent(TIMING_SIZES, HNSW_BUILD_CUMULATIVE)
HNSW_SWEEP_FIT = fit_power_law_exponent(TIMING_SIZES, HNSW_SWEEP_SECONDS_PER_QUERY)
del _timing_index


def power_law(fit, n: float) -> float:
    return fit.coefficient * n**fit.exponent


# --- 転置ファイルと Product Quantization(学習の時間と、query あたりの探索の時間を分けて測る)---
INVERTED_FILE_FIT_SECONDS, INVERTED_FILE_SWEEP_SECONDS_PER_QUERY, PRODUCT_QUANTIZATION_FIT_SECONDS, PRODUCT_QUANTIZATION_SEARCH_SECONDS_PER_QUERY = [], [], [], []
for _n in TIMING_SIZES:
    _subset = ann_ordering(TIMING_SEED_INDEX)[:_n]
    _truth, _local_excluded, _ = ann_truth(TIMING_SEED_INDEX, _subset)
    _vectors = PASSAGE_EMBEDDINGS[torch.from_numpy(_subset)]
    _fit_total, _sweep_total = 0.0, 0.0
    for _num_lists in inverted_file_list_counts_for(_n):
        _start = time.time()
        _inverted_file = InvertedFileIndex(_vectors, _num_lists, INVERTED_FILE_KMEANS_ITERATIONS, seed=INVERTED_FILE_SEED_BASE + TIMING_SEED_INDEX)
        _fit_total += time.time() - _start
        _start = time.time()
        sweep_inverted_file_probes(_inverted_file, ANN_QUERIES_TENSOR[:TIMING_QUERIES], _local_excluded[:TIMING_QUERIES], _subset, _truth[:TIMING_QUERIES], INVERTED_FILE_GRID, NEIGHBORS, STOP_RECALL)
        _sweep_total += time.time() - _start
    INVERTED_FILE_FIT_SECONDS.append(_fit_total)
    INVERTED_FILE_SWEEP_SECONDS_PER_QUERY.append(_sweep_total / TIMING_QUERIES)
    _start = time.time()
    _quantizer = ProductQuantizer(EMBEDDING_DIMENSION, PRODUCT_QUANTIZATION_SUBVECTORS, PRODUCT_QUANTIZATION_CODEWORDS)
    _quantizer.fit(_vectors, PRODUCT_QUANTIZATION_KMEANS_ITERATIONS, seed=PRODUCT_QUANTIZATION_SEED_BASE + TIMING_SEED_INDEX)
    _product_quantization_index = ProductQuantizationIndex(_quantizer, _vectors)
    PRODUCT_QUANTIZATION_FIT_SECONDS.append(time.time() - _start)
    _start = time.time()
    for _search in (_product_quantization_index.search_asymmetric, _product_quantization_index.search_symmetric):
        _search(ANN_QUERIES_TENSOR[:TIMING_QUERIES], NEIGHBORS, excluded=_local_excluded[:TIMING_QUERIES])
        for _rescored in PRODUCT_QUANTIZATION_RESCORE_COUNTS:
            _search(ANN_QUERIES_TENSOR[:TIMING_QUERIES], NEIGHBORS, excluded=_local_excluded[:TIMING_QUERIES], num_rescored=_rescored)
    PRODUCT_QUANTIZATION_SEARCH_SECONDS_PER_QUERY.append((time.time() - _start) / TIMING_QUERIES)
INVERTED_FILE_FIT_FIT = fit_power_law_exponent(TIMING_SIZES, INVERTED_FILE_FIT_SECONDS)
INVERTED_FILE_SWEEP_FIT = fit_power_law_exponent(TIMING_SIZES, INVERTED_FILE_SWEEP_SECONDS_PER_QUERY)
PRODUCT_QUANTIZATION_FIT_FIT = fit_power_law_exponent(TIMING_SIZES, PRODUCT_QUANTIZATION_FIT_SECONDS)
PRODUCT_QUANTIZATION_SEARCH_FIT = fit_power_law_exponent(TIMING_SIZES, PRODUCT_QUANTIZATION_SEARCH_SECONDS_PER_QUERY)
print(
    f"HNSW の構築(累積): {dict(zip(TIMING_SIZES, rounded(HNSW_BUILD_CUMULATIVE, 2), strict=True))} 秒、べき指数 b = {HNSW_BUILD_FIT.exponent:.3f}(標準誤差 {HNSW_BUILD_FIT.exponent_stderr:.3f})、"
    f"N_max = {N_MAX_CANDIDATES[0]} への外挿 {power_law(HNSW_BUILD_FIT, N_MAX_CANDIDATES[0]):.0f} 秒(1 シードあたり)"
)
print(
    f"HNSW の掃引(1 query あたり): {dict(zip(TIMING_SIZES, rounded(HNSW_SWEEP_SECONDS_PER_QUERY, 5), strict=True))} 秒、べき指数 b = {HNSW_SWEEP_FIT.exponent:.3f}、"
    f"N_max への外挿(1 query){power_law(HNSW_SWEEP_FIT, N_MAX_CANDIDATES[0]) * 1000:.2f} ms"
)
print(
    f"転置ファイル(リスト数 5 通り): k-means の学習のべき指数 b = {INVERTED_FILE_FIT_FIT.exponent:.3f}、掃引(1 query あたり)のべき指数 b = {INVERTED_FILE_SWEEP_FIT.exponent:.3f}。"
    f"Product Quantization: 符号帳の学習と符号化のべき指数 b = {PRODUCT_QUANTIZATION_FIT_FIT.exponent:.3f}、探索(1 query あたり)のべき指数 b = {PRODUCT_QUANTIZATION_SEARCH_FIT.exponent:.3f}。"
    f"N_max = {N_MAX_CANDIDATES[0]} への外挿(1 シード、query {len(ANN_QUERY_INDEX)} 個): 転置ファイル {power_law(INVERTED_FILE_FIT_FIT, N_MAX_CANDIDATES[0]) + power_law(INVERTED_FILE_SWEEP_FIT, N_MAX_CANDIDATES[0]) * len(ANN_QUERY_INDEX):.0f} 秒、"
    f"Product Quantization {power_law(PRODUCT_QUANTIZATION_FIT_FIT, N_MAX_CANDIDATES[0]) + power_law(PRODUCT_QUANTIZATION_SEARCH_FIT, N_MAX_CANDIDATES[0]) * len(ANN_QUERY_INDEX):.0f} 秒"
)
```

    HNSW の構築(累積): {2048: 6.89, 4096: 16.07, 8192: 32.96} 秒、べき指数 b = 1.129(標準誤差 0.053)、N_max = 65536 への外挿 352 秒(1 シードあたり)
    HNSW の掃引(1 query あたり): {2048: 0.00339, 4096: 0.00612, 8192: 0.00987} 秒、べき指数 b = 0.770、N_max への外挿(1 query)49.90 ms
    転置ファイル(リスト数 5 通り): k-means の学習のべき指数 b = 1.257、掃引(1 query あたり)のべき指数 b = 0.607。Product Quantization: 符号帳の学習と符号化のべき指数 b = 0.789、探索(1 query あたり)のべき指数 b = 0.379。N_max = 65536 への外挿(1 シード、query 890 個): 転置ファイル 53 秒、Product Quantization 45 秒



```python
# --- 並べ替えモデルの学習: 定常状態の 1 ステップの時間 ---
def measure_training(condition: str, step_counts, warmup_steps: int, batch_queries: int = BATCH_QUERIES) -> list[float]:
    spec = CONDITIONS[condition]
    model = build_model(condition, TIMING_SEED_INDEX).to(device)
    total = warmup_steps + sum(step_counts)
    schedule = sample_reranking_schedule(
        TRAIN_QUERY_INDEX, MINED_CANDIDATES, QUERY_ARTICLES, QUERY_SOURCES, PASSAGE_ARTICLES, TRAIN_PASSAGE_INDEX, total, batch_queries, NUM_NEGATIVES, seed=SCHEDULE_SEED_BASE + TIMING_SEED_INDEX
    )

    def run(steps: int) -> None:
        sub = type(schedule)(
            schedule.query[:steps], schedule.positive[:steps], schedule.hard_negatives[:steps], schedule.random_negatives[:steps], 0, 0, 0
        )
        train_reranker(
            model, spec["kind"], QUERY_TOKENS, PASSAGE_TOKENS, sub, spec["negatives"], peak_learning_rate=1e-4, warmup_steps=1, min_learning_rate=1e-6,
            weight_decay=WEIGHT_DECAY, gradient_clip_threshold=GRADIENT_CLIP_THRESHOLD, temperature=TEMPERATURE,
            use_fp16_autocast=USE_FP16_AUTOCAST, init_loss_scale=INIT_LOSS_SCALE, loss_scale_growth_interval=LOSS_SCALE_GROWTH_INTERVAL,
        )

    timed_call(lambda: run(warmup_steps))
    times = [timed_call(lambda n=n: run(n)) for n in step_counts]
    del model
    empty_device_cache()
    return times


STEP_SECONDS, SCALING_ROWS = {}, {}
for _condition in CONDITIONS:
    _times = measure_training(_condition, SCALING_STEP_COUNTS, SCALING_WARMUP_STEPS)
    STEP_SECONDS[_condition] = _times[-1] / SCALING_STEP_COUNTS[-1]
    _fit = fit_power_law_exponent(SCALING_STEP_COUNTS, _times)
    _t_max = STEP_CANDIDATES[0]
    _power = _times[-1] * (_t_max / SCALING_STEP_COUNTS[-1]) ** _fit.exponent
    SCALING_ROWS[_condition] = {"exponent": _fit.exponent, "power_minus_proportional": _power - STEP_SECONDS[_condition] * _t_max}
    print(
        f"{_condition}({CONDITIONS[_condition]['kind']}、{CONDITIONS[_condition]['negatives']} 負例): {dict(zip(SCALING_STEP_COUNTS, rounded(_times, 2), strict=True))} 秒、"
        f"1 ステップ {STEP_SECONDS[_condition] * 1000:.1f} ms、べき指数 b = {_fit.exponent:.3f}、T = {_t_max} でのべき乗則の外挿 - 比例の値 = {SCALING_ROWS[_condition]['power_minus_proportional']:+.1f} 秒"
    )

# 参考: 1 ステップの query 数を半分にしたときの 1 ステップの時間(固定費が支配的かの判断材料。見積もりには使わない)
for _condition in ("D1", "D2"):
    _half = measure_training(_condition, (SCALING_STEP_COUNTS[1],), SCALING_WARMUP_STEPS, batch_queries=BATCH_QUERIES // 2)[0] / SCALING_STEP_COUNTS[1]
    print(
        f"参考 {_condition}: query {BATCH_QUERIES // 2} 個の 1 ステップ {_half * 1000:.1f} ms、query {BATCH_QUERIES} 個は {STEP_SECONDS[_condition] * 1000:.1f} ms"
        f"(比 {STEP_SECONDS[_condition] / _half:.2f}。2 に近ければ時間はバッチに比例し、1 に近ければ固定費が支配的)"
    )

# --- 並べ替えの評価: 1 組あたりのスコアの時間(較正の query の先頭 100 個 x K)---
_sample_queries = CALIBRATION_QUERY_INDEX[:100]
_sample_candidates = CALIBRATION_CANDIDATE_IDS[:100]
PAIR_SECONDS = {}
for _condition in ("D1", "D2"):
    _model = build_model(_condition, TIMING_SEED_INDEX).to(device)
    evaluate_rerank(_model, CONDITIONS[_condition]["kind"], _sample_queries[:20], _sample_candidates[:20])  # ウォームアップ
    PAIR_SECONDS[CONDITIONS[_condition]["kind"]] = (
        timed_call(lambda: evaluate_rerank(_model, CONDITIONS[_condition]["kind"], _sample_queries, _sample_candidates)) / _sample_candidates.size
    )
    del _model
empty_device_cache()
# --- 学習前のモデルの損失の測定と、学習の組の生成 ---
_timing_schedule = get_schedule(TIMING_SEED_INDEX, 64)
_model = build_model("D1", TIMING_SEED_INDEX).to(device)
compute_reranking_losses(_model, "cross_encoder", QUERY_TOKENS, PASSAGE_TOKENS, _timing_schedule, "hard", range(8), TEMPERATURE, USE_FP16_AUTOCAST)  # ウォームアップ
INITIAL_LOSS_SECONDS_PER_STEP = timed_call(
    lambda: compute_reranking_losses(_model, "cross_encoder", QUERY_TOKENS, PASSAGE_TOKENS, _timing_schedule, "hard", range(32), TEMPERATURE, USE_FP16_AUTOCAST)
) / 32
del _model
empty_device_cache()
_schedule_times = []
for _steps in (128, 256, 512):
    _start = time.time()
    sample_reranking_schedule(TRAIN_QUERY_INDEX, MINED_CANDIDATES, QUERY_ARTICLES, QUERY_SOURCES, PASSAGE_ARTICLES, TRAIN_PASSAGE_INDEX, _steps, BATCH_QUERIES, NUM_NEGATIVES, seed=1)
    _schedule_times.append(time.time() - _start)
_schedule_fit = fit_power_law_exponent((128, 256, 512), _schedule_times)
SCHEDULE_SECONDS_PER_STEP = _schedule_times[-1] / 512
print(
    f"並べ替えの評価: cross-encoder {PAIR_SECONDS['cross_encoder'] * 1000:.3f} ms / 組、dual encoder {PAIR_SECONDS['dual_encoder'] * 1000:.3f} ms / 組"
    f"(候補 {RERANK_CANDIDATES} 件 x query 100 個の評価の時間から)"
)
print(
    f"学習前のモデルの損失の測定: 1 ステップあたり {INITIAL_LOSS_SECONDS_PER_STEP * 1000:.1f} ms(学習 1 回につき最後の区間のステップ数だけ)。"
    f"学習の組の生成: {dict(zip((128, 256, 512), rounded(_schedule_times, 3), strict=True))} 秒、べき指数 b = {_schedule_fit.exponent:.3f}、1 ステップあたり {SCHEDULE_SECONDS_PER_STEP * 1000:.2f} ms(見積もりは比例)"
)
print(
    "データの準備(5.3 節、1 回だけ): " + "、".join(f"{k} {v:.1f} 秒" for k, v in DATA_SECONDS_BY_STAGE.items())
    + f"。合計 {DATA_SECONDS:.1f} 秒(6.4 節の経過時間に含まれる)"
)
SCALING_SECONDS = time.time() - _t0_scaling
print(f"スケーリングの計測 {SCALING_SECONDS:.1f} 秒")
```

    D1(cross_encoder、hard 負例): {8: 0.6, 16: 1.2, 32: 2.4} 秒、1 ステップ 75.0 ms、べき指数 b = 0.997、T = 1024 でのべき乗則の外挿 - 比例の値 = -0.7 秒
    D2(dual_encoder、hard 負例): {8: 0.55, 16: 1.08, 32: 2.15} 秒、1 ステップ 67.2 ms、べき指数 b = 0.989、T = 1024 でのべき乗則の外挿 - 比例の値 = -2.5 秒
    E2(cross_encoder、random 負例): {8: 0.6, 16: 1.21, 32: 2.41} 秒、1 ステップ 75.5 ms、べき指数 b = 0.999、T = 1024 でのべき乗則の外挿 - 比例の値 = -0.2 秒
    参考 D1: query 4 個の 1 ステップ 42.8 ms、query 8 個は 75.0 ms(比 1.75。2 に近ければ時間はバッチに比例し、1 に近ければ固定費が支配的)
    参考 D2: query 4 個の 1 ステップ 60.8 ms、query 8 個は 67.2 ms(比 1.11。2 に近ければ時間はバッチに比例し、1 に近ければ固定費が支配的)
    並べ替えの評価: cross-encoder 0.380 ms / 組、dual encoder 0.220 ms / 組(候補 50 件 x query 100 個の評価の時間から)
    学習前のモデルの損失の測定: 1 ステップあたり 25.8 ms(学習 1 回につき最後の区間のステップ数だけ)。学習の組の生成: {128: 0.089, 256: 0.148, 512: 0.232} 秒、べき指数 b = 0.690、1 ステップあたり 0.45 ms(見積もりは比例)
    データの準備(5.3 節、1 回だけ): トークナイザの取得 5.3 秒、参照コーパスの取得 3.2 秒、コーパスのダウンロード 5.6 秒、コーパスの読み込み 13.3 秒、記事の境界の検証 0.0 秒、記事の符号化と検証 427.2 秒、分割 0.1 秒、コーパスの構成 4.2 秒、埋め込み 59.7 秒、第 1 段の候補と採掘 2.5 秒。合計 523.4 秒(6.4 節の経過時間に含まれる)
    スケーリングの計測 84.2 秒


### 6.4 実行計画の選択

6.3 節の計測から、21 通りの計画(6.1 節)のそれぞれの残りの実行時間を見積もる。

- **学習 1 回**(並べ替え)= 1 ステップの時間 × $T$ + 学習の組の生成(1 ステップあたりの時間 × $T$。安全側に毎回数える)+ 学習前のモデルの損失の測定(最後の区間のステップ数 × 1 ステップあたりの時間)+ 評価の時間 + 固定費の余裕 2 秒。
  評価の時間は、較正の学習では較正の query(検証用の記事)の並べ替え 1 回、本番の学習ではそれに加えて、並べ替えの query(評価用の記事)の並べ替えと、ランダムな候補の中での評価を 1 回ずつ
  (基準の条件の D2・E2 は、学習前のモデルの較正の query での評価 1 回を加える。D2 は 1 回だけ)。
- **較正** = 条件(D1・D2・E2)ごとに、格子 5 点 + 拡張 1 点の **最悪の場合の 6 回**(段階 6 は格子 3 点 + 拡張 1 点の最悪の場合の 4 回)。**本番** = 条件ごとのシード数 × 学習 1 回。
- **近似最近傍探索**(1 シードあたり): HNSW の構築(べき乗則で $N_{\max}$ まで外挿)+ 水準ごとの掃引の合計(べき乗則で各水準へ外挿。シード 0 は階層の診断のため掃引を 2 倍かかるとみなす)+ 転置ファイル + Product Quantization(それぞれべき乗則で外挿)。
  **シード数 × 上の時間**。
- **実験 F**(観察): 第 1 段 5 通り × 並べ替えの query(実験 F の query)× $K = 100$ の候補を cross-encoder で並べ替える時間。
- 判定・ブートストラップ・図・アップロードの準備の余裕 300 秒。

**経過時間 + 見積もりが予算(120 分)に収まる計画のうち、番号の最も小さいものを選ぶ。** どれも収まらなければ、索引の構築と学習の前に停止する。


```python
def run_seconds(condition: str, num_steps: int, main: bool) -> float:
    spec = CONDITIONS[condition]
    kind = spec["kind"]
    seconds = STEP_SECONDS[condition] * num_steps + RUN_OVERHEAD_SECONDS + SCHEDULE_SECONDS_PER_STEP * num_steps  # 学習の組の生成は(シード、T)ごとに 1 回だが、安全側に毎回数える
    seconds += INITIAL_LOSS_SECONDS_PER_STEP * final_loss_window(num_steps)  # 学習前のモデルの、最後の区間の組での損失(学習の成立 (a) の分母)
    seconds += PAIR_SECONDS[kind] * CALIBRATION_CANDIDATE_IDS.size  # 較正の query での並べ替え
    if not main:
        return seconds
    seconds += PAIR_SECONDS[kind] * RERANK_CANDIDATE_IDS.size  # 評価用の記事の query での並べ替え
    seconds += PAIR_SECONDS[kind] * RANDOM_CANDIDATE_IDS.size  # ランダムな候補の中での評価
    if condition in BASELINE_CONDITIONS:
        seconds += PAIR_SECONDS[kind] * CALIBRATION_CANDIDATE_IDS.size  # 学習前のモデルの較正の query での評価(D2 は 1 回だけだが、安全側に毎回数える)
    return seconds


def ann_seed_seconds(n_max: int, seed_index: int) -> float:
    sweep = sum(power_law(HNSW_SWEEP_FIT, n) for n in size_levels_for(n_max)) * len(ANN_QUERY_INDEX)
    diagnostic = sweep if seed_index == 0 else 0.0  # 階層の診断(シード 0 のみ。最下層だけの探索は同程度の時間とみなす)
    return (
        power_law(HNSW_BUILD_FIT, n_max) + sweep + diagnostic
        + power_law(INVERTED_FILE_FIT_FIT, n_max) + power_law(INVERTED_FILE_SWEEP_FIT, n_max) * len(ANN_QUERY_INDEX)
        + power_law(PRODUCT_QUANTIZATION_FIT_FIT, n_max) + power_law(PRODUCT_QUANTIZATION_SEARCH_FIT, n_max) * len(ANN_QUERY_INDEX)
    )




def estimate_plan(plan: dict, level: str = CURRENT_LEVEL_NAME) -> dict:
    settings = stage_settings(plan["stage"], level)
    steps = plan["num_steps"]
    calibration_points_worst = settings["grid_points"] + 1  # 格子 5 点(段階 6 は 3 点)+ 拡張 1 点の最悪の場合
    calibration = sum(calibration_points_worst * run_seconds(c, steps, main=False) for c in CONDITIONS)
    seeds = {"D1": settings["d_seeds"], "D2": settings["d_seeds"], "E2": settings["e2_seeds"]}
    main = sum(seeds[c] * run_seconds(c, steps, main=True) for c in CONDITIONS)
    ann = sum(ann_seed_seconds(settings["n_max"], s) for s in range(settings["ann_seeds"]))
    f_seconds = 5 * len(F_QUERY_INDEX) * max(RERANK_KS) * PAIR_SECONDS["cross_encoder"]
    return {"calibration": calibration, "main": main, "ann": ann, "f": f_seconds, "total": calibration + main + ann + f_seconds + REPORTING_MARGIN_SECONDS}


ELAPSED_AT_SELECTION = time.time() - NOTEBOOK_START_TIME
PLAN_ESTIMATES = {p["plan"]: estimate_plan(p) for p in PLANS}
print(f"経過時間 {ELAPSED_AT_SELECTION / 60:.1f} 分、予算 {SESSION_BUDGET_SECONDS / 60:.0f} 分")
for _p in PLANS:
    _e = PLAN_ESTIMATES[_p["plan"]]
    _settings = stage_settings(_p["stage"])
    _fits = ELAPSED_AT_SELECTION + _e["total"] <= SESSION_BUDGET_SECONDS
    print(
        f"  計画 {_p['plan']:>2}(T = {_p['num_steps']}、段階 {_p['stage']}: {json.dumps(_settings)}): 較正 {_e['calibration'] / 60:.1f} 分 + 本番 {_e['main'] / 60:.1f} 分 + 近似最近傍探索 {_e['ann'] / 60:.1f} 分"
        f" + 実験 F {_e['f'] / 60:.1f} 分 + 余裕 -> 残り {_e['total'] / 60:.1f} 分、終了の見込み {(ELAPSED_AT_SELECTION + _e['total']) / 60:.1f} 分({'収まる' if _fits else '超える'})"
    )
_feasible = [p for p in PLANS if ELAPSED_AT_SELECTION + PLAN_ESTIMATES[p["plan"]]["total"] <= SESSION_BUDGET_SECONDS]
AUTO_PLAN = _feasible[0] if _feasible else None
if FORCED_PLAN is not None:
    SELECTED_PLAN = PLANS[FORCED_PLAN]
    print(f"*** テスト専用の上書き: 計画 {FORCED_PLAN} を使う(見積もりによる選択は {AUTO_PLAN and AUTO_PLAN['plan']}) ***")
elif AUTO_PLAN is None:
    raise RuntimeError(
        f"最も下位の計画(計画 {len(PLANS) - 1})でも見積もりが予算を超えるため、索引の構築と学習の前に停止する。結果の情報は何も得ていない(学習も評価も行っていない)。"
        "実行条件(判定基準・水準・前提条件以外)を直して再実行すること。"
    )
else:
    SELECTED_PLAN = AUTO_PLAN
NUM_STEPS = SELECTED_PLAN["num_steps"]
STAGE = SELECTED_PLAN["stage"]
PLAN_SETTINGS = stage_settings(STAGE)
N_MAX = PLAN_SETTINGS["n_max"]
NUM_ANN_SEEDS = PLAN_SETTINGS["ann_seeds"]
NUM_SEEDS = {"D1": PLAN_SETTINGS["d_seeds"], "D2": PLAN_SETTINGS["d_seeds"], "E2": PLAN_SETTINGS["e2_seeds"]}
SEEDS_D = min(NUM_SEEDS["D1"], NUM_SEEDS["D2"])
SEEDS_E = min(NUM_SEEDS["D1"], NUM_SEEDS["E2"])
SIZE_LEVELS = size_levels_for(N_MAX)
print(
    f"選ばれた計画: {SELECTED_PLAN['plan']}(T = {NUM_STEPS}、段階 {STAGE}: {STAGE_CHANGES[STAGE]})。近似最近傍探索のシード数 {NUM_ANN_SEEDS}、N_max = {N_MAX}(N の水準 {SIZE_LEVELS})、"
    f"並べ替えのシード数 {json.dumps(NUM_SEEDS)}(実験 D は {SEEDS_D}、実験 E は {SEEDS_E} シード)、見積もり {PLAN_ESTIMATES[SELECTED_PLAN['plan']]['total'] / 60:.1f} 分"
)
print(f"HNSW の設定: M = {HNSW_MAX_CONNECTIONS}、efConstruction = {HNSW_CONSTRUCTION_WIDTH}。転置ファイルのリスト数(N_max): {inverted_file_list_counts_for(N_MAX)}")
```

    経過時間 10.8 分、予算 120 分
      計画  0(T = 1024、段階 0: {"ann_seeds": 5, "n_max": 65536, "e2_seeds": 5, "d_seeds": 5, "grid_points": 5}): 較正 27.7 分 + 本番 33.2 分 + 近似最近傍探索 47.5 分 + 実験 F 1.9 分 + 余裕 -> 残り 115.4 分、終了の見込み 126.2 分(超える)
      計画  1(T = 1024、段階 1: {"ann_seeds": 3, "n_max": 65536, "e2_seeds": 5, "d_seeds": 5, "grid_points": 5}): 較正 27.7 分 + 本番 33.2 分 + 近似最近傍探索 29.2 分 + 実験 F 1.9 分 + 余裕 -> 残り 97.1 分、終了の見込み 107.9 分(収まる)
      計画  2(T = 1024、段階 2: {"ann_seeds": 3, "n_max": 32768, "e2_seeds": 5, "d_seeds": 5, "grid_points": 5}): 較正 27.7 分 + 本番 33.2 分 + 近似最近傍探索 14.7 分 + 実験 F 1.9 分 + 余裕 -> 残り 82.6 分、終了の見込み 93.4 分(収まる)
      計画  3(T = 1024、段階 3: {"ann_seeds": 3, "n_max": 32768, "e2_seeds": 3, "d_seeds": 5, "grid_points": 5}): 較正 27.7 分 + 本番 28.2 分 + 近似最近傍探索 14.7 分 + 実験 F 1.9 分 + 余裕 -> 残り 77.5 分、終了の見込み 88.4 分(収まる)
      計画  4(T = 1024、段階 4: {"ann_seeds": 3, "n_max": 32768, "e2_seeds": 3, "d_seeds": 3, "grid_points": 5}): 較正 27.7 分 + 本番 19.9 分 + 近似最近傍探索 14.7 分 + 実験 F 1.9 分 + 余裕 -> 残り 69.3 分、終了の見込み 80.1 分(収まる)
      計画  5(T = 1024、段階 5: {"ann_seeds": 3, "n_max": 16384, "e2_seeds": 3, "d_seeds": 3, "grid_points": 5}): 較正 27.7 分 + 本番 19.9 分 + 近似最近傍探索 7.6 分 + 実験 F 1.9 分 + 余裕 -> 残り 62.2 分、終了の見込み 73.0 分(収まる)
      計画  6(T = 1024、段階 6: {"ann_seeds": 3, "n_max": 16384, "e2_seeds": 3, "d_seeds": 3, "grid_points": 3}): 較正 18.5 分 + 本番 19.9 分 + 近似最近傍探索 7.6 分 + 実験 F 1.9 分 + 余裕 -> 残り 52.9 分、終了の見込み 63.8 分(収まる)
      計画  7(T = 512、段階 0: {"ann_seeds": 5, "n_max": 65536, "e2_seeds": 5, "d_seeds": 5, "grid_points": 5}): 較正 16.3 分 + 本番 23.7 分 + 近似最近傍探索 47.5 分 + 実験 F 1.9 分 + 余裕 -> 残り 94.5 分、終了の見込み 105.3 分(収まる)
      計画  8(T = 512、段階 1: {"ann_seeds": 3, "n_max": 65536, "e2_seeds": 5, "d_seeds": 5, "grid_points": 5}): 較正 16.3 分 + 本番 23.7 分 + 近似最近傍探索 29.2 分 + 実験 F 1.9 分 + 余裕 -> 残り 76.1 分、終了の見込み 87.0 分(収まる)
      計画  9(T = 512、段階 2: {"ann_seeds": 3, "n_max": 32768, "e2_seeds": 5, "d_seeds": 5, "grid_points": 5}): 較正 16.3 分 + 本番 23.7 分 + 近似最近傍探索 14.7 分 + 実験 F 1.9 分 + 余裕 -> 残り 61.7 分、終了の見込み 72.5 分(収まる)
      計画 10(T = 512、段階 3: {"ann_seeds": 3, "n_max": 32768, "e2_seeds": 3, "d_seeds": 5, "grid_points": 5}): 較正 16.3 分 + 本番 20.0 分 + 近似最近傍探索 14.7 分 + 実験 F 1.9 分 + 余裕 -> 残り 57.9 分、終了の見込み 68.8 分(収まる)
      計画 11(T = 512、段階 4: {"ann_seeds": 3, "n_max": 32768, "e2_seeds": 3, "d_seeds": 3, "grid_points": 5}): 較正 16.3 分 + 本番 14.2 分 + 近似最近傍探索 14.7 分 + 実験 F 1.9 分 + 余裕 -> 残り 52.2 分、終了の見込み 63.0 分(収まる)
      計画 12(T = 512、段階 5: {"ann_seeds": 3, "n_max": 16384, "e2_seeds": 3, "d_seeds": 3, "grid_points": 5}): 較正 16.3 分 + 本番 14.2 分 + 近似最近傍探索 7.6 分 + 実験 F 1.9 分 + 余裕 -> 残り 45.1 分、終了の見込み 55.9 分(収まる)
      計画 13(T = 512、段階 6: {"ann_seeds": 3, "n_max": 16384, "e2_seeds": 3, "d_seeds": 3, "grid_points": 3}): 較正 10.9 分 + 本番 14.2 分 + 近似最近傍探索 7.6 分 + 実験 F 1.9 分 + 余裕 -> 残り 39.6 分、終了の見込み 50.5 分(収まる)
      計画 14(T = 256、段階 0: {"ann_seeds": 5, "n_max": 65536, "e2_seeds": 5, "d_seeds": 5, "grid_points": 5}): 較正 10.6 分 + 本番 19.0 分 + 近似最近傍探索 47.5 分 + 実験 F 1.9 分 + 余裕 -> 残り 84.0 分、終了の見込み 94.9 分(収まる)
      計画 15(T = 256、段階 1: {"ann_seeds": 3, "n_max": 65536, "e2_seeds": 5, "d_seeds": 5, "grid_points": 5}): 較正 10.6 分 + 本番 19.0 分 + 近似最近傍探索 29.2 分 + 実験 F 1.9 分 + 余裕 -> 残り 65.7 分、終了の見込み 76.5 分(収まる)
      計画 16(T = 256、段階 2: {"ann_seeds": 3, "n_max": 32768, "e2_seeds": 5, "d_seeds": 5, "grid_points": 5}): 較正 10.6 分 + 本番 19.0 分 + 近似最近傍探索 14.7 分 + 実験 F 1.9 分 + 余裕 -> 残り 51.2 分、終了の見込み 62.0 分(収まる)
      計画 17(T = 256、段階 3: {"ann_seeds": 3, "n_max": 32768, "e2_seeds": 3, "d_seeds": 5, "grid_points": 5}): 較正 10.6 分 + 本番 15.9 分 + 近似最近傍探索 14.7 分 + 実験 F 1.9 分 + 余裕 -> 残り 48.1 分、終了の見込み 59.0 分(収まる)
      計画 18(T = 256、段階 4: {"ann_seeds": 3, "n_max": 32768, "e2_seeds": 3, "d_seeds": 3, "grid_points": 5}): 較正 10.6 分 + 本番 11.4 分 + 近似最近傍探索 14.7 分 + 実験 F 1.9 分 + 余裕 -> 残り 43.6 分、終了の見込み 54.4 分(収まる)
      計画 19(T = 256、段階 5: {"ann_seeds": 3, "n_max": 16384, "e2_seeds": 3, "d_seeds": 3, "grid_points": 5}): 較正 10.6 分 + 本番 11.4 分 + 近似最近傍探索 7.6 分 + 実験 F 1.9 分 + 余裕 -> 残り 36.5 分、終了の見込み 47.3 分(収まる)
      計画 20(T = 256、段階 6: {"ann_seeds": 3, "n_max": 16384, "e2_seeds": 3, "d_seeds": 3, "grid_points": 3}): 較正 7.1 分 + 本番 11.4 分 + 近似最近傍探索 7.6 分 + 実験 F 1.9 分 + 余裕 -> 残り 33.0 分、終了の見込み 43.8 分(収まる)
    選ばれた計画: 1(T = 1024、段階 1: 実験 A・B・C のシード数を削る)。近似最近傍探索のシード数 3、N_max = 65536(N の水準 [4096, 8192, 16384, 32768, 65536])、並べ替えのシード数 {"D1": 5, "D2": 5, "E2": 5}(実験 D は 5、実験 E は 5 シード)、見積もり 97.1 分
    HNSW の設定: M = 5、efConstruction = 100。転置ファイルのリスト数(N_max): [128, 256, 512, 1024, 2048]




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/025_ann_search_and_reranking.ipynb
