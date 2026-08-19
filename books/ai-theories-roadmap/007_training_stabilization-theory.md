---
title: "学習の安定化(Training Stabilization)(理論編)"
---

この記事は前編(理論編)です。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/007_training_stabilization-practice-1)。

# 007. 学習の安定化(Training Stabilization)


## 1. 概要 / Overview

006(小型 GPT の事前学習)は、Adam(固定学習率、gradient clipping なし)という意図的に素朴な学習設定を採用し、実験 H でその限界を観測しようとした。しかしスモークテストのごく短い学習では明確な不安定化を引き出せず、「素朴な学習設定がどこで壊れるか」は未検証のまま残った。007 では、**正規化前置(Pre-Layer Normalization)/ 正規化後置(Post-Layer Normalization、002)と学習率を独立変数として意図的に不安定性を誘発し**、AdamW(Decoupled Weight Decay Regularization)・warmup + cosine スケジュール・gradient clipping(勾配クリッピング)という 3 つの安定化技術それぞれが、どの種類の不安定性に効くかを切り分けて検証する。mixed precision(混合精度学習)は数値精度(FP16 / BF16 の underflow/overflow)に起因する別種の問題であり、011(混合精度学習)に切り出す。


## 2. 参考論文 / References

| # | 著者 / Authors | タイトル / Title | 会議・媒体 / Venue | URL |
|---|---|---|---|---|
| [1] | Loshchilov, I., Hutter, F. | Decoupled Weight Decay Regularization | ICLR 2019 | https://arxiv.org/abs/1711.05101 |
| [2] | Kingma, D. P., Ba, J. | Adam: A Method for Stochastic Optimization | ICLR 2015 | https://arxiv.org/abs/1412.6980 |
| [3] | Goyal, P., et al. | Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour | 2017 | https://arxiv.org/abs/1706.02677 |
| [4] | Loshchilov, I., Hutter, F. | SGDR: Stochastic Gradient Descent with Warm Restarts | ICLR 2017 | https://arxiv.org/abs/1608.03983 |
| [5] | Brown, T. B., et al. | Language Models are Few-Shot Learners | NeurIPS 2020 | https://arxiv.org/abs/2005.14165 |
| [6] | Pascanu, R., Mikolov, T., Bengio, Y. | On the difficulty of training Recurrent Neural Networks | ICML 2013 | https://arxiv.org/abs/1211.5063 |
| [7] | Xiong, R., et al. | On Layer Normalization in the Transformer Architecture | ICML 2020 | https://arxiv.org/abs/2002.04745 |

本文中で各理論に言及する際は、対応する番号(例:「decoupled weight decay [1]」)を付す。


## 3. 理論 / Theory


### 3.1 動機・課題: なぜ「安定化」を独立トピックとして扱うか

**002 で観測した正規化後置の勾配の不均衡**: [002_transformer_block.ipynb](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/002_transformer_block-theory) は、正規化前置(Pre-Layer Normalization)と正規化後置(Post-Layer Normalization、Xiong et al. [7])を比較し、正規化後置では残差経路そのものが正規化を経由しない正規化前置と異なり、層を重ねるほど勾配のスケールが不均衡になりやすいことを確認した。

**004 で観測した正規化欠如条件でのシード間ばらつきの増大**: [004_normalization_and_activation.ipynb](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/004_normalization_and_activation-theory) は、正規化を欠いた条件で学習のシード間ばらつきが増大することを観測した。正規化の有無が、学習の再現性・安定性そのものに影響することを示唆する。

**006 実験 H は不安定性を引き出せなかった**: [006_pretraining_small_gpt.ipynb](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/006_pretraining_small_gpt-theory) の実験 H は、素朴な学習設定(Adam・固定学習率・gradient clipping なし)での勾配ノルムを記録する仕組みを用意したが、スモークテストの短い学習では明確なスパイクや発散を観測できなかった(006 7.4 節)。これは「安定化技術が不要である」ことを意味しない。**むしろ、不安定性が生じる条件そのものを意図的に作り出さなければ、安定化技術の効果を検証しようがない**、というのが 007 の出発点である。

**007 の設計方針**: 正規化前置 / 正規化後置(アーキテクチャ)と学習率(ハイパーパラメータ)を独立変数として直交させ、意図的に不安定性を誘発する条件(正規化後置 × 高学習率)を作った上で、AdamW・warmup + cosine スケジュール・gradient clipping のそれぞれがこの不安定性に効くかを個別に検証する(3.5 節・7 節)。


### 3.2 AdamW(Decoupled Weight Decay Regularization)

**記号 / Notation**:

- $\theta$: パラメータ
- $g_t$: ステップ $t$ の勾配($\partial \mathcal{L} / \partial \theta$)
- $\eta$: 学習率(learning rate)
- $\beta_1, \beta_2$: モーメント推定の指数移動平均の減衰率
- $m_t, v_t$: 1 次・2 次モーメント推定
- $\hat{m}_t, \hat{v}_t$: バイアス補正後のモーメント推定
- $\epsilon$: 数値安定化のための微小定数
- $\lambda$: 重み減衰係数(weight decay)

**Adam のモーメント推定 [2]**: Adam は勾配の 1 次モーメント(移動平均、方向)と 2 次モーメント(移動平均、スケール)を推定し、パラメータごとに更新幅を正規化する。

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}, \qquad
u_t = \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

$u_t$ は「勾配の典型的なスケールで正規化した更新方向」であり、$\theta \leftarrow \theta - \eta u_t$ がバニラ Adam の更新式である。

**L2 正則化を勾配に混ぜる方式(素朴な重み減衰)**: 重み減衰を実装する素朴な方法は、L2 正則化の勾配 $\lambda \theta$ を通常の勾配に加算することである。

$$
\tilde{g}_t = g_t + \lambda \theta_{t-1}
$$

この $\tilde{g}_t$ を Adam のモーメント推定にそのまま使う(`AdamWithL2Regularization`、`src/training/optimizer.py`)。**この方式には問題がある**: 重み減衰の寄与 $\lambda \theta$ が $\hat{v}_t$(2 次モーメント推定、勾配のスケールを反映)を経由してしまう。勾配 $g_t$ が恒常的に大きいパラメータ群では $\hat{v}_t \approx g_t^2 \gg (\lambda\theta)^2$ となり、重み減衰の寄与は $u_t$ の中でほぼ無視できるほど薄まる。逆に $g_t$ が恒常的に小さいパラメータ群では $\hat{v}_t \approx (\lambda\theta)^2$ となり、重み減衰由来の項が正規化によって $O(1)$ まで増幅される。**同一の名目上の $\lambda$ が、パラメータ群の勾配スケールに応じて全く異なる実効的な減衰強度をもたらしてしまう。**

**Decoupled weight decay(AdamW)[1]**: Loshchilov & Hutter は、重み減衰をモーメント推定(したがって $\hat{v}_t$ による正規化)から完全に切り離し、パラメータ更新式に別項として直接加えることを提案した。

$$
\theta_t = \theta_{t-1} - \eta \left( u_t + \lambda \theta_{t-1} \right)
= \theta_{t-1}(1 - \eta\lambda) - \eta u_t
$$

$u_t$ の計算(1 次・2 次モーメント推定)には $\tilde{g}_t$ ではなく元の勾配 $g_t$ のみを使う。重み減衰の項 $\eta\lambda\theta_{t-1}$ は $\hat{v}_t$ を一切経由しないため、実効的な減衰強度 $\eta\lambda$ はパラメータ群の勾配スケールに依存しない、**設計上の定数** になる。この違いを、勾配スケールが異なる複数のパラメータ群を用意した合成タスクで直接検証する(主張 6、6.8 節)。

**002・004 との関係**: AdamW 自体は不安定性の「スパイク」を直接抑える機構ではない(gradient clipping・warmup とは異なる役割)。むしろ、重み減衰という正則化がパラメータ群間で公平に働くことを保証する技術であり、正規化方式(002)によってパラメータ群間の勾配スケールの不均衡が生じる(3.1 節)ほど、AdamW と L2 正則化混入の実効的な差は大きくなりうる。


### 3.3 warmup + cosine スケジュール

学習率スケジュール(`compute_warmup_cosine_learning_rate`、`src/training/schedule.py`)は、学習序盤に学習率を線形に立ち上げる **warmup**(ステップ数 $w$)と、その後コサイン曲線に沿って減衰させる **cosine decay** を組み合わせる(GPT-3 [5] の Appendix D などで採用される標準的な構成)。

$$
\eta(s) =
\begin{cases}
\eta_{\text{peak}} \cdot s / w & (0 < s \le w) \\[4pt]
\eta_{\text{min}} + \dfrac{1}{2}(\eta_{\text{peak}} - \eta_{\text{min}})\left(1 + \cos(\pi p)\right), \quad p = \min\!\left(1, \dfrac{s - w}{T - w}\right) & (w < s)
\end{cases}
$$

**warmup が必要な理由**: 学習序盤はモーメント推定 $\hat{m}_t, \hat{v}_t$(3.2 節)がまだ十分なステップ数の移動平均に基づいておらず、特にバイアス補正直後の数ステップは推定が不安定である。この不安定な推定に基づいて最初から大きな学習率を適用すると、更新幅が過大になり発散しやすい(SGDR [4] の議論)。学習率を小さい値から徐々に立ち上げることで、モーメント推定が十分な観測数に基づくようになってから本来の学習率に到達させる。

**cosine decay が必要な理由**: 学習終盤に学習率を下げることで、パラメータが損失地形の狭い谷(sharp minima)に落ち着く前に更新幅を粗くしすぎず、より精密な収束を可能にする。コサイン曲線は終端に向けて減衰が滑らかに緩やかになる(線形減衰と異なり、終盤で学習率が急に変化しない)という性質を持つ。

**gradient clipping・AdamW との関係**: warmup は「学習序盤の不安定な大きい更新」を防ぐ点で gradient clipping と役割が重なる部分があるが、warmup は **学習率そのもの** を小さくすることで更新幅を抑えるのに対し、gradient clipping は **勾配のノルムに応じて事後的に** スケーリングする(3.4 節)。両者は異なるステップで異なる不安定性(warmup は序盤の推定不足、clipping は任意のタイミングで起きる突発的なスパイク)に効くと考えられ、主張 3・4 でこれを分離して検証する。


### 3.4 gradient clipping(勾配クリッピング)

グローバルノルムでのクリッピング(Pascanu et al. [6])は、全パラメータの勾配を連結した L2 ノルム $\|g\|_2$ が閾値 $c$ を超えた場合、勾配全体を同一のスケール係数で縮小する。

$$
\|g\|_2 = \sqrt{\textstyle\sum_i \|g_i\|_2^2}, \qquad
g \leftarrow g \cdot \min\!\left(1, \frac{c}{\|g\|_2}\right)
$$

**なぜ「グローバル」ノルムか**: パラメータごとに個別の閾値でクリッピングすると、勾配の方向が歪む(各パラメータの相対的な更新比率が変わってしまう)。全パラメータを 1 つのベクトルとみなしたグローバルノルムでスケーリングすることで、勾配の **方向は保ったまま大きさだけ** を抑える。

**何を防ぐか**: 損失地形が局所的に非常に急峻な領域(勾配爆発、gradient explosion)を通過するとき、1 ステップの更新幅が過大になり、パラメータが良い領域から弾き飛ばされることがある(Pascanu et al. [6] の議論、原論文は主に RNN の long-term dependency 学習を対象とするが、Transformer の深い残差スタックでも同様の急峻な領域が生じうる)。gradient clipping は、勾配ノルムが閾値を超えた **そのステップのみ** 更新幅を抑える対症療法であり、AdamW(3.2 節、重み減衰の実効強度の均一化)・warmup(3.3 節、序盤のモーメント推定不足への対処)とは異なる種類の問題に対処する。

**閾値 $c$ の決め方**: 閾値が小さすぎると通常の学習ステップまで過度に抑制してしまい、大きすぎるとクリッピングが実質的に機能しない。007 では、較正セル(6.1 節)で不安定な学習率における勾配ノルムの分布を実測し、その **`CLIP_QUANTILE` 分位点**($q$、`CLIP_QUANTILE`=0.85 を較正前に宣言、6.1 節。当初 0.90 から 2 回目の本番実行を踏まえ修正、6.10.2 節)を閾値 $c$ として決定する(固定の経験値を決め打ちしない)。

$$
c = \mathrm{quantile}_q(\{\|g\|_2^{(1)}, \dots, \|g\|_2^{(T)}\})
$$

**平均 + $k\sigma$ 方式からの変更(6.10.1 節)**: 当初は $c = \mu_g + k\sigma_g$($k$ は較正前に宣言する定数)を用いていたが、1 回目の本番実行(Google Colab T4、6.10.1 節)で、勾配ノルムの分布が右に長く歪んでいる(少数の大きなスパイクが平均・標準偏差を押し上げる)場合にこの方式では閾値が高くなりすぎ、本番のステップ数ではクリッピングがほぼ発動しないことが判明した(前提条件 P2 の不成立)。分位点方式では、較正実行時点での勾配ノルムの分布そのものから「上位 $1-q$ 割のステップで発動する」閾値を直接決定できるため、分布の形状(歪度)に依存せず、意図した発動頻度に近い挙動を保証しやすい。


### 3.5 実験設計: 正規化方式 × 学習率を独立変数とした不安定性の誘発

3.1 節の動機を踏まえ、**正規化前置 / 正規化後置**(アーキテクチャ変数)と **学習率の水準**(006 基準 / 較正で決定する「高め」)を独立変数とする 2×2 の基本枠組みに、Q4(正規化後置 × 高学習率、最も不安定になりやすいと予想される象限)でのみ安定化技術の水準を細分化した、以下の実験グリッドを用いる。

```mermaid
flowchart TB
    subgraph Q1["Q1: 正規化前置 × 006 基準学習率"]
        Q1N["なし"] --- Q1A["全部乗せ"]
    end
    subgraph Q2["Q2: 正規化前置 × 高学習率"]
        Q2N["なし"] --- Q2A["全部乗せ"]
    end
    subgraph Q3["Q3: 正規化後置 × 006 基準学習率"]
        Q3N["なし"] --- Q3A["全部乗せ"]
    end
    subgraph Q4["Q4(主軸): 正規化後置 × 高学習率"]
        Q4N["なし"] --- Q4W["warmup+cosine のみ"] --- Q4C["gradient clipping のみ"] --- Q4A["全部乗せ"]
    end
```

**図 1: 実験グリッド(全 10 条件)。** 「全部乗せ」は AdamW(重み減衰込み)+ warmup + cosine スケジュール + gradient clipping の 3 技術すべてを指す。Q4 のみ、warmup+cosine・gradient clipping を単独で有効化した中間水準を持つ(AdamW 単独の水準は設けない。AdamW の効果は言語モデルの学習曲線ではなく、3.2 節で述べた通り合成タスクで直接検証する、主張 6・6.8 節)。

**なぜ Q1〜Q3 は「なし」「全部乗せ」の 2 水準のみか**: Q1〜Q3(006 基準学習率、または正規化前置)は Q4 ほど不安定になることを想定していない対照条件であり、3 技術を個別に切り分ける価値は主に「不安定性が実際に生じている」Q4 で最大化される。Q1〜Q3 は「なし」と「全部乗せ」の差(主張 1・2a・2b の診断)だけを見れば、正規化方式・学習率が不安定性の誘発にどう効くかの主張には十分である。

**対比条件の設計(主張 1〜5、詳細は 7.2 節)**:

| # | 主張 | 対比条件(B vs A) | 直交性 |
|---|---|---|---|
| 1 | 正規化後置は正規化前置より不安定になりやすい | Q3 vs Q1 | 学習率・安定化技術の水準を固定し、正規化方式のみ変える |
| 2a | 正規化前置で学習率を上げると不安定性が増す | Q2 vs Q1 | 正規化方式・安定化技術の水準を固定し、学習率のみ変える |
| 2b | 正規化後置で学習率を上げると不安定性が増す | Q4(なし)vs Q3 | 同上(正規化後置側) |
| 3 | gradient clipping は損失の暴れを抑える | Q4(clipping のみ)vs Q4(なし) | 正規化方式・学習率を固定し、gradient clipping のみ変える |
| 4 | warmup + cosine は損失の暴れを抑える | Q4(warmup+cosine のみ)vs Q4(なし) | 同上(warmup + cosine 側) |
| 5 | 全部乗せは「なし」より収束を遅らせない | Q4(全部乗せ)vs Q4(なし) | 正規化方式・学習率を固定し、安定化技術の水準のみ変える |




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/02_pretraining/007_training_stabilization.ipynb
