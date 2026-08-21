---
title: "デコーディング戦略(Decoding Strategies)(理論編)"
---

この記事は前編(理論編)です。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/008_decoding_strategies-practice-1)。

# 008. デコーディング戦略(Decoding Strategies)


## 1. 概要 / Overview

自己回帰言語モデル(006・007 でスクラッチ実装した `GPTLanguageModel`)は、各ステップで次トークンの確率分布 $P(x_t \mid x_{<t})$ を出力するのみであり、その分布から実際にどのトークン列を生成するかは **デコーディング戦略(decoding strategy)** に委ねられる。本トピックでは、`GPTLanguageModel.generate()` が既に対応する貪欲法(greedy decoding)・temperature サンプリングに加え、top-k サンプリング・top-p サンプリング(nucleus sampling)・ビームサーチ(beam search)をスクラッチ実装し、生成されるテキストの性質(繰り返しの多さ・多様性・尤度)がデコーディング戦略によってどう変わるかを比較する。

比較には 006(小型 GPT の事前学習)・007(学習の安定化)で確立したアーキテクチャ・学習設定を用いて英語の小型 GPT モデルを新たに学習し、そのモデルを Hugging Face Hub に公開する(将来の他トピックで再利用する「標準モデル」の位置づけ)。


## 2. 参考論文 / References

- Fan, A., Lewis, M., Dauphin, Y., "Hierarchical Neural Story Generation", ACL 2018.
  https://arxiv.org/abs/1805.04833(top-k サンプリング)
- Holtzman, A., Buys, J., Du, L., Forbes, M., Choi, Y., "The Curious Case of Neural Text
  Degeneration", ICLR 2020. https://arxiv.org/abs/1904.09751(top-p サンプリング / nucleus
  sampling、退化現象(Degeneration)の理論)
- Wu, Y. et al., "Google's Neural Machine Translation System: Bridging the Gap between Human
  and Machine Translation", 2016. https://arxiv.org/abs/1609.08144(長さペナルティ付き
  ビームサーチ)
- Li, J., Galley, M., Brockett, C., Gao, J., Dolan, B., "A Diversity-Promoting Objective
  Function for Neural Conversation Models", NAACL 2016. https://arxiv.org/abs/1510.03055
  (distinct-n、多様性指標)
- Welleck, S. et al., "Neural Text Generation with Unlikelihood Training", ICLR 2020.
  https://arxiv.org/abs/1908.04319(seq-rep-n、n-gram 重複率)

本トピックのモデル学習は、006・007 の以下の論文にも依拠する(理論の再掲はしない)。

- Radford, A. et al., "Language Models are Unsupervised Multitask Learners", 2019(GPT-2、
  バイトレベル BPE)
- Su, J. et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021(RoPE)
- Loshchilov, I., Hutter, F., "Decoupled Weight Decay Regularization", ICLR 2019(AdamW)


## 3. 理論 / Theory


### 3.1 デコーディング戦略の位置づけ

自己回帰言語モデルは、系列 $x_{1:t-1}$ が与えられたとき次トークンの条件付き分布

$$
P(x_t \mid x_{1:t-1}) = \mathrm{softmax}(\mathrm{logits}_t)
$$

を出力する(`GPTLanguageModel.forward` の `logits`)。系列全体の生成は、この分布から
逐次的にトークンを選び、選んだトークンを文脈に追加して次のステップへ進める操作を
繰り返すことで行う。**モデルが学習するのは分布 $P$ のみであり、この分布から実際に
どのトークンを選ぶか(デコーディング戦略)はモデルの学習とは独立な、生成時の設計
判断である。** 同一のモデル・同一の分布からでも、デコーディング戦略が異なれば
生成されるテキストの性質(流暢さ・多様性・繰り返しの多さ)は大きく異なりうる。

以下、$V$ を語彙サイズ(vocabulary size)、$\mathrm{logits}_t \in \mathbb{R}^V$ を
ステップ $t$ の出力(`GPTLanguageModel.forward` の最終次元)とする。


### 3.2 貪欲法(Greedy Decoding)と temperature サンプリング(復習)

**貪欲法**: 各ステップで確率最大のトークンを決定的に選ぶ。

$$
x_t = \arg\max_{v} P(v \mid x_{<t})
$$

**temperature サンプリング**: 分布を温度 $T$ でスケーリングしてから多項分布サンプリングする。

$$
P_T(v \mid x_{<t}) = \frac{\exp(\mathrm{logits}_t[v] / T)}{\sum_{v'} \exp(\mathrm{logits}_t[v'] / T)}
$$

$T \to 0$ で貪欲法に一致し(最大値以外の項が相対的に無視できるほど小さくなる)、
$T=1$ で元の分布からのサンプリング、$T \to \infty$ で一様分布に近づく。いずれも
`GPTLanguageModel.generate()`(006)に実装済みである。


### 3.3 top-k サンプリング

Fan et al., 2018 が提案した手法。各ステップで確率上位 $k$ 個のトークンのみを候補集合
とし、それ以外の確率を 0 にしたうえで再正規化してサンプリングする。

$$
V^{(k)}_t = \{\, v \mid \mathrm{rank}(\mathrm{logits}_t[v]) \le k \,\}, \qquad
P_k(v \mid x_{<t}) = \begin{cases}
    P(v \mid x_{<t}) \big/ \sum_{v' \in V^{(k)}_t} P(v' \mid x_{<t}) & v \in V^{(k)}_t \\
    0 & \text{otherwise}
\end{cases}
$$

実装(`top_k_filter`、`src/generation/decoding.py`)では、候補集合の外側の logits を
$-\infty$ に置き換えることで、後段の `softmax` が自動的にこの再正規化を行う
(`softmax(-\infty) = 0`)。

**候補集合サイズは常に $k$ で固定** される。分布がどれだけ「自信を持っている」
(1 つのトークンに確率が集中している)かに関わらず、常に同じ個数の候補が残る点が
3.5 節で述べる top-p との本質的な違いになる(実験 B)。


### 3.4 top-p サンプリング(Nucleus Sampling)

Holtzman et al., 2020 が提案した手法。確率降順に並べたときの累積確率が $p$ に達する
までの最小の集合(nucleus)を候補集合とする。

$$
V^{(p)}_t = \min \left\{ V' \subseteq V \;\middle|\;
    \sum_{v \in V'} P(v \mid x_{<t}) \ge p \right\}
$$

（「最小の $V'$」は、確率降順に上位から要素を追加していき、累積和が初めて $p$ 以上に
達した時点の集合として一意に定まる。）候補集合の外側の確率を 0 にし再正規化してから
サンプリングする(`top_p_filter`、`src/generation/decoding.py` に数式レベルで対応する
実装)。

**候補集合サイズは分布の形状に応じて動的に変化する**。分布が 1 つのトークンに強く
集中している(エントロピーが低い、モデルが「自信を持っている」)場合、少数のトークンの
累積確率がすぐに $p$ に達するため $|V^{(p)}_t|$ は小さくなる。逆に分布が一様に近い
(エントロピーが高い、モデルが「迷っている」)場合、多くのトークンを合算しないと
累積確率が $p$ に達しないため $|V^{(p)}_t|$ は大きくなる。この「エントロピーに応じた
適応性」が top-k との本質的な違いであり、実験 B で直接検証する。

**退化現象(Degeneration)との関係**: Holtzman et al., 2020 は、貪欲法・ビームサーチの
ような **最大化に基づく(maximization-based)decoding** が、高頻度語の反復
(repetition)・不自然に平板なテキストを生成しやすいことを指摘した(退化現象)。
最大化に基づく decoding は各ステップで局所的に最も確率の高いトークンを選び続けるため、
一度「繰り返し」に入ると、直前の繰り返しパターンの継続がそれ自体高い確率を持つ
(言語モデルは訓練データ中の反復パターンを学習しており、文脈が繰り返しであるほど
次も同じ繰り返しになる確率を高く予測する傾向がある)ことから、繰り返しから抜け出す
確率の高い選択肢があってもそれを採用しない。一方、確率的な decoding(temperature
サンプリング・top-p サンプリング)は分布の裾からもサンプリングしうるため、この
自己強化的な繰り返しループに陥りにくいと理論的に予想される(実験 A で検証する)。


### 3.5 ビームサーチ(Beam Search)

各ステップで単一の最尤トークン列を追跡する貪欲法に対し、ビームサーチは複数
(ビーム数 $B$)の候補系列を並行して保持し、より広い探索を行うことで系列全体の
対数尤度が高い候補を探す近似的な探索アルゴリズムである。

各ステップで、現在保持する $B$ 個の候補系列それぞれについて次トークンの上位 $B$ 個を
展開し(最大 $B^2$ 個の候補)、系列全体のスコアが高い順に上位 $B$ 個を残す。系列
$Y = (y_1, \dots, y_{|Y|})$ の対数尤度の総和 $\log P(Y) = \sum_i \log P(y_i \mid y_{<i})$
は系列が長いほど(各項が非正であるため)単調に減少する傾向があり、そのまま比較すると
短い系列が不当に有利になる。Wu et al., 2016(GNMT)はこれを補正する長さペナルティ
$lp(Y)$ を導入し、正規化済みスコアで候補を評価する。

$$
lp(Y) = \frac{(5 + |Y|)^{\alpha}}{(5 + 1)^{\alpha}}, \qquad
\mathrm{score}(Y) = \frac{\log P(Y)}{lp(Y)}
$$

$\alpha$(`beam_search` の `length_penalty` 引数)は正規化の強さを制御するハイパー
パラメータであり、$\alpha = 0$ で $lp(Y) \equiv 1$(正規化なし、対数尤度の総和を
そのまま使う)になる。

**ビーム数 $B=1$ の場合、ビームサーチは貪欲法に一致する**(各ステップで残る候補が
常に 1 個 — その時点での最尤トークンを追加した系列 — になるため)。この関係は
`GPTLanguageModel.generate(temperature=0.0)` との一致として不変条件アサーション(4 節)
で直接検証する。

**ビーム数とトレードオフ**: ビーム数を増やすほどより広い探索により最良候補の尤度は
改善しうるが、ビーム内の複数候補は探索の初期に分岐した近傍の系列になりやすく、
候補間の多様性は下がる傾向がある(実験 D、定性的な観察)。


### 3.6 top-k と top-p の適応性の違い(実験 B の理論的根拠)

3.3・3.4 節で述べた通り、top-k は候補集合サイズが常に $k$ で一定であるのに対し、
top-p の候補集合サイズ $|V^{(p)}_t|$ は分布のエントロピー

$$
H(P(\cdot \mid x_{<t})) = -\sum_v P(v \mid x_{<t}) \log P(v \mid x_{<t})
$$

に応じて動的に変化することが理論的に予想される。エントロピーが高い(分布が
一様に近い)ステップほど、累積確率が $p$ に達するまでに必要なトークン数が多くなる
ため $|V^{(p)}_t|$ は増加し、エントロピーが低い(分布が集中している)ステップほど
$|V^{(p)}_t|$ は減少する。したがって $|V^{(p)}_t|$ とエントロピーの間には正の相関が
期待される。この相関を実験 B で直接測定する。




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/02_pretraining/008_decoding_strategies.ipynb
