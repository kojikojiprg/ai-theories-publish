---
title: "デコーディング戦略(Decoding Strategies)"
---

# 008. デコーディング戦略(Decoding Strategies)

## 1. 概要 / Overview

自己回帰言語モデル(006・007 でスクラッチ実装した`GPTLanguageModel`)は、各ステップで次トークンの確率分布 $P(x_t \mid x_{<t})$ を出力するのみであり、その分布から実際にどのトークン列を生成するかは **デコーディング戦略(decoding strategy)** に委ねられる。本トピックでは、`GPTLanguageModel.generate()`が既に対応する貪欲法(greedy decoding)・temperature サンプリングに加え、top-k サンプリング・top-p サンプリング(nucleus sampling)・ビームサーチ(beam search)をスクラッチ実装し、生成されるテキストの性質(繰り返しの多さ・多様性・尤度)がデコーディング戦略によってどう変わるかを比較する。

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

を出力する(`GPTLanguageModel.forward`の`logits`)。系列全体の生成は、この分布から
逐次的にトークンを選び、選んだトークンを文脈に追加して次のステップへ進める操作を
繰り返すことで行う。**モデルが学習するのは分布 $P$ のみであり、この分布から実際に
どのトークンを選ぶか(デコーディング戦略)はモデルの学習とは独立な、生成時の設計
判断である。** 同一のモデル・同一の分布からでも、デコーディング戦略が異なれば
生成されるテキストの性質(流暢さ・多様性・繰り返しの多さ)は大きく異なりうる。

以下、$V$ を語彙サイズ(vocabulary size)、$\mathrm{logits}_t \in \mathbb{R}^V$ を
ステップ $t$ の出力(`GPTLanguageModel.forward`の最終次元)とする。

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
$-\infty$ に置き換えることで、後段の`softmax`が自動的にこの再正規化を行う
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
サンプリングする(`top_p_filter`、`src/generation/decoding.py`に数式レベルで対応する
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

$\alpha$(`beam_search`の`length_penalty`引数)は正規化の強さを制御するハイパー
パラメータであり、$\alpha = 0$ で $lp(Y) \equiv 1$(正規化なし、対数尤度の総和を
そのまま使う)になる。

**ビーム数 $B=1$ の場合、ビームサーチは貪欲法に一致する**(各ステップで残る候補が
常に 1 個 — その時点での最尤トークンを追加した系列 — になるため)。この関係は
`GPTLanguageModel.generate(temperature=0.0)`との一致として不変条件アサーション(4 節)
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

## 4. 実装方針 / Implementation Policy

- top-k フィルタリング(`top_k_filter`)・top-p フィルタリング(`top_p_filter`)・
  ビームサーチ(`beam_search`)は`src/generation/decoding.py`にスクラッチ実装する
  (`torch.topk`・`torch.sort`などのテンソル演算プリミティブは使うが、フィルタリング・
  探索のロジック自体は既存ライブラリの高水準 API に委譲しない)。
- 生成品質の指標(n-gram 重複率・distinct-n)は`src/utils/statistics.py`に
  `compute_ngram_repetition_rate`・`compute_distinct_n`として追加する。
- モデル本体(`GPTLanguageModel`)・学習ループ(`train_language_model`)・
  optimizer(`AdamW`)・学習率スケジュール(`compute_warmup_cosine_learning_rate`)は
  006・007 の実装をそのまま再利用し、本トピックでは変更しない。
- モデルアーキテクチャ・トークナイザ条件は 006 の本番実行結果(実験 D・E・F・G)で
  確定した値(英語・バイトレベル BPE・語彙サイズ 8192・`D_MODEL=256`など)をそのまま
  採用する(2.1 節参照、改めて選定し直さない)。
- 学習は 007 の安定化技術(AdamW・warmup + cosine・gradient clipping)を適用するが、
  gradient clipping 閾値・学習率は 008 自身のモデル・データに対して本番スケールで
  改めて較正する(007 の較正結果を流用しない。較正は本番と同じスケールで行う)。
- Hugging Face Hub へのアップロードは関数として実装するが、スモークテスト(ローカル実行)では呼び出さない。認証情報(トークン)はコード中に一切書き込まず、
  Google Colab の Secrets(`google.colab.userdata.get("HF_TOKEN")`)から読み出す設計とする。

## 5. 実装 / Implementation

### 5.1 スケールの設定(スモークテスト)

`SMOKE_TEST=True`はローカルでの動作確認・本番実行可能性の検証用の縮小スケールであり、`SMOKE_TEST=False`(Google Colab T4)が本番設定である。model サイズ・トークナイザ語彙サイズは 006 の本番実行結果(2.1 節)を
そのまま使うため、`SMOKE_TEST`では変更しない。スケールする(縮小する)のはコーパス量・
学習ステップ数のみであり、本番設定との比率構造(等比刻みの語彙サイズは 008 では
単一条件のため対象外)は保つ。

### 5.2 コーパスの取得

006 と同じ`load_wikipedia_corpus`で英語 Wikipedia コーパス(`en_006_pretraining.json`、
20 MB 以上)を取得する。取得結果は`CACHE_DIR`(ランタイムのローカルファイルシステム)に
記事単位でキャッシュされ、同一セッション内でのセル再実行では再ダウンロードを省略する。
ただし Google Drive とは連携していないため、**ランタイムをリセットした場合(GitHub
経由でノートブックを開き直すたびに発生する)は、このキャッシュは失われ、実際に
再ダウンロードが走る**(固定の記事タイトル・リビジョン ID で取得するため、取得内容
自体は毎回同一で再現性は保たれる)。008 は英語のみを対象とするため、006 のように
複数言語・複数トークナイザ条件を並べる必要はない。

### 5.3 トークナイザを整数 ID 方式に適合させるラッパー

006 5.3 節と同じ理由で、`BPETokenizer.encode()`が返す部分語シンボル列(`list[str]`)を
整数 ID に変換するラッパーが必要である。`BPEIDTokenizer`は 006・008・009 で共通に使うため
`src/data/tokenizer.py`にモジュール化してあり、ここでは import して使う。

### 5.4 BPE 学習時間のスケーリング計測

BPE 学習時間のデータ量依存性を
3 点以上のデータ量で実測し、$\log t = \log a + b \log n$ のあてはめでべき指数 $b$ を
推定して本番データ量(`TOKENIZER_TRAIN_BYTES`)への外挿値を出す(006 5.4 節の
`measure_bpe_scaling`と同じ手法)。

### 5.5 符号化時間のスケーリング計測

学習データ全体(`LM_CORPUS_BYTES`)の符号化(encode)は 1 回だけ行われる処理だが、
超線形にコストが増加すると本番実行時間の見積もりを大きく崩すため、006 5.5 節と同じ
手法でスケーリングを計測する。

### 5.6 本番語彙でのトークナイザ取得・ラウンドトリップ検証

共有トークナイザリポジトリ`kojikojiprg/ai-theories-tokenizer-en`(バイトレベル BPE、
語彙サイズ 8192)から`BPEIDTokenizer`を取得する(`load_bpe_id_tokenizer_from_hub`、
`src/data/tokenizer.py`)。このリポジトリのトークナイザは、008 が以前に本番コーパスから
学習した結果をそのまま切り出したものであり内容は完全に同一である(取得元を切り替えても
トークン化が変わらないことは、ローカルで実際に BPE を学習して得たトークン ID 列と
比較し、`train_text`・`val_text`全体で完全一致することを事前に確認済み)。Hub からの
取得に失敗した場合のみ、同一設定(`VOCAB_SIZE=8192`、`byte_level=True`、
`max_chunk_bytes=MAX_CHUNK_BYTES`)で BPE を再学習するフォールバックを行う。
ラウンドトリップ(`decode(encode(text)) == text`)が英語コーパスで成立することを
確認する(4 節、005・006 と同様の検証)。

### 5.7 コーパスの符号化・評価窓の作成

### 5.8 model の構築ヘルパー

2.1 節で確定した構成(RoPE・RMSNorm・SwiGLU・正規化前置・重み共有)で`GPTLanguageModel`
を構築する。

### 5.9 較正セル: 学習率・gradient clipping 閾値・1 ステップあたりの実行時間

007 6.1 節と同じ手順(較正は本番と同じ`CALIBRATION_STEPS = NUM_STEPS`で行う)を踏む。
008 は正規化方式・学習率を独立変数とした不安定性誘発が目的ではなく、AdamW +
warmup + cosine + gradient clipping を **常に全部乗せで使う**(007 の結論、主張 5:
全部乗せは収束を遅らせない)ため、較正するのは (1) 発散しない学習率、(2) その学習率での
gradient clipping 閾値、の 2 つのみである。

**学習率の選定方針**: `LR_CALIBRATION_MULTIPLIERS`(006 基準学習率に対する倍率、等比刻み)
のうち、`CALIBRATION_STEPS`実行後に NaN/Inf に発散せず、かつ検証 bits-per-byte が
最小のものを採用する(008 は不安定性の誘発ではなく通常の学習が目的であるため、007と異なり
「最も不安定な候補」ではなく「最も良く学習できた候補」を選ぶ)。

**gradient clipping 閾値**: 採用した学習率での較正実行における勾配ノルムの
`CLIP_QUANTILE`分位点(007 3.4 節・6.1 節と同じ分位点方式)。

**1 ステップあたりの実行時間**: 007 6.1 節(3)と同様、ステップ数を変えた計測により
べき指数を推定し(モデル学習が線形時間であることを確認したうえで)、本番ステップ数
(`PRODUCTION_NUM_STEPS`)への外挿値を出す。

### 5.10 モデル学習の実行

**この学習は「主張を検証する実験」ではなく、7 節以降の生成品質比較実験のための
モデル準備である**(0 節)ため、判定基準の事前登録は不要である。検証用
bits-per-byte の推移は記録し、学習が発散していないことを確認する(前提条件 P0)。
結果の永続化(5.11 節)により、再実行時は既存の結果を再利用する。


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/008_decoding_strategies/output_22_0.png)
    


### 5.10 節の注記(後日の調査による): 印字した検証 bits-per-byte の時点

上のセルが「最終検証 bits-per-byte = 1.6707」として印字した値は、学習を最後まで終えた時点ではなく、評価の最後である 2,000 ステップ時点の値である(`NUM_STEPS = 2181`が`EVAL_INTERVAL = 200`で割り切れないため)。保存・アップロードした重み(2,181 ステップ時点)の同じ検証集合での bits-per-byte は **1.6681** であり、前提条件 P0 の成否も 7 節の生成実験も、この注記の影響を受けない。ノートブックの外での再現を含む詳細は付録 A.1。

### 5.11 生成ヘルパー: 各デコーディング戦略での生成

温度サンプリング・top-k・top-p は 1 ステップずつ手動でループし(`generate()`は
top-k・top-p に対応しないため)、`model.forward`を直接呼び出す。貪欲法は
`model.generate(temperature=0.0)`をそのまま使う。

### 5.12 Hugging Face Hub へのアップロード

**この節の関数は、スモークテスト(ローカル実行)では呼び出さない。** `SMOKE_TEST=False`
かつ Google Colab 実行時(`IN_COLAB=True`)にのみ実際にアップロードを行うガードを設ける。
トークンの値はコード中に一切書き込まず、Google Colab の Secrets(鍵アイコン)に
`HF_TOKEN`という名前で事前に登録しておいたものを`google.colab.userdata.get("HF_TOKEN")`
で読み出す。

## 6. 不変条件のアサーション / Invariant Assertions

実験の妥当性が依存する量をすべて
アサーションで確認する。

## 7. 実験 / Experiments

各実験は 1 実験 = 1 検証事項とし、実験 A(退化現象)・実験 B(top-k と
top-p の適応性の違い)を主軸(三値判定を厳密に適用)、実験 C(temperature のトレード
オフ)・実験 D(beam size のトレードオフ)を補助(定性的な観察、判定基準を設けない)
とする。

### 7.1 生成実験の共通設定

プロンプト・生成長・シード数は本番実行前に(結果を見る前に)ここで宣言する。

### 7.2 実験宣言セル: 検証すること・判定基準

**共通の対比量・標準偏差の導出(実験 A・B で共通)**: グループ A・グループ B をそれぞれ
$n_A$・$n_B$ 個のサンプル(実験 A ではプロンプト x 手法の組、実験 B ではプロンプトごとの
相関係数)で測定し、標本平均 $\bar{x}_A, \bar{x}_B$・標本標準偏差 $s_A, s_B$
(不偏標準偏差、`ddof=1`)を求める。

$$
\Delta = \bar{x}_B - \bar{x}_A \qquad \mathrm{SE}(\Delta) = \sqrt{\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}}
\qquad \text{threshold} = 2 \times \mathrm{SE}(\Delta)
$$

判定は **支持**($\Delta$ が期待方向に閾値超え)/ **反証**($\Delta$ が逆方向に閾値超え)/
**判定不能**($|\Delta| \le$ 閾値)の 3 分岐とする(007 6.2 節・6.4 節と同一の方式)。

| #   | 実験                 | 対比条件(B vs A)                             | 対比量                                        | 期待方向                   |
| --- | -------------------- | -------------------------------------------- | --------------------------------------------- | -------------------------- |
| A   | 退化現象             | {temperature, top-p}(B) vs {貪欲法, beam}(A) | REP_N-gram 重複率                             | $\Delta < 0$(B の方が低い) |
| B   | top-k/top-p の適応性 | 実測相関係数(B) vs ゼロ基準(A、全要素 0)     | エントロピーと top-p 候補集合サイズの相関係数 | $\Delta > 0$(正の相関)     |

**実験 A の対比量の直接性について**: 検証したい介入(デコーディング戦略の違い)が
直接作用するのは各ステップでの「候補集合の絞り込み方」であり、n-gram 重複率は
その結果として系列全体に現れる下流の量である。3.4 節の理論(退化現象は繰り返し
ループへの自己強化的な収束として説明される)に従えば、n-gram 重複率(特に短い n)は
繰り返しループの直接の観測量に近く、両者の間に他の要因が介在しにくいため、対比量として
採用する(Holtzman et al., 2020 自身も rep-n を主要な定量指標として用いている)。

**実験 B の対比量の直接性について**: 検証したい性質(top-p の候補集合サイズがエントロピーに
応じて動的に変化すること)そのものを対比量(相関係数)として直接測定しており、介入の
作用点との乖離はない。

**前提条件(pre-condition)**: 判定基準とは別に、検証したい仮説とは独立な量として
本番実行の前に宣言する。

- **前提条件 P0(学習の進行、5.10 節で確定済み)**: 最終検証 bits-per-byte が
  `PRECONDITION_BPB_THRESHOLD`以下であること。実験 A・B の両方に共通して適用する
  (モデルが十分に流暢なテキストを生成できないと、退化現象の比較(実験 A)も
  分布の形状に基づく適応性の比較(実験 B)も意味を持たない)。
- **前提条件 P1(分布の非退化、実験 B のみ)**: 各プロンプトのエントロピー系列
  (`GEN_LENGTH`ステップ分)の標準偏差が`PRECONDITION_ENTROPY_STD_MIN`を超えること。
  エントロピーが全ステップで一定であれば、エントロピーと候補集合サイズの相関係数は
  定義上不安定になり(分散 0 の変数との相関)、「適応性がない」ことの証拠にはならない
  (適応性を測る前提として、そもそもエントロピー自体に変動が必要という、検証したい
  仮説(top-p が適応的かどうか)とは独立な条件)。

### 7.3 判定関数

### 7.4 実験 A: 退化現象(Degeneration)の検証

貪欲法・ビームサーチ(グループ A、決定的)と temperature サンプリング・top-p
サンプリング(グループ B、確率的)で、各プロンプトから生成した継続テキストの
REP_N-gram 重複率を比較する。グループ A はプロンプトごとに 1 サンプル(貪欲法・
ビームサーチそれぞれ)、グループ B はプロンプト x シードごとに 1 サンプル
(temperature・top-p それぞれ)を生成する。

### 7.5 実験 B: top-k と top-p の適応性の違い

各プロンプトについて貪欲法で`GEN_LENGTH`ステップ生成する軌道に沿って、各ステップの
分布のエントロピーと、その分布に`top_p_filter(p=TOP_P_MAIN)`を適用したときの
候補集合サイズ(有限の logits を持つ要素数)を記録する。プロンプトごとに、
エントロピー系列と候補集合サイズ系列の間のピアソン相関係数を計算する
(プロンプト = 1 サンプルとして扱う)。診断として、`top_k_filter(k=TOP_K_MAIN)`の
候補集合サイズが全ステップで定数`TOP_K_MAIN`に一致することも確認する。

### 7.6 可視化: 実験 A・B


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/008_decoding_strategies/output_33_0.png)
    


### 7.7 実験 C(補助): temperature による多様性と一貫性のトレードオフ

**判定基準は設けない(定性的な観察のみである)。** temperature を等比刻みで複数水準
振り、各水準で生成したテキストの distinct-n(多様性)と、生成テキストを学習済み
モデル自身で評価した bits-per-byte 相当の指標(生成トークン列に対する平均負の対数
尤度、一貫性の代理指標)を観察する。


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/008_decoding_strategies/output_35_0.png)
    


### 7.8 実験 D(補助): beam size と生成確率・多様性のトレードオフ

**判定基準は設けない(定性的な観察のみである)。** beam size を公比 2 の等比刻みで
複数水準振り、各水準で best 候補の長さ正規化対数尤度(スコア)と、最終ビーム内候補間の
多様性(候補どうしを連結した系列の distinct-n、値が低いほど候補間で似た系列が多い)を
観察する。


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/008_decoding_strategies/output_37_0.png)
    


## 8. 結果・考察 / Results and Discussion

**実行環境についての注記**: 以下は Google Colab T4 GPU での本番実行(`SMOKE_TEST=False`、本番スケールのコーパス・語彙サイズ・ステップ数)の実際のセル出力に基づく。判定基準は本番実行後に変更していない(7.2 節で事前登録した基準・閾値の導出式をそのまま適用する)。

### 8.1 モデル学習

較正(5.9 節)では、006 基準学習率(3e-4)の x0.5・x1.0・x2.0・x4.0 のいずれも`CALIBRATION_STEPS=2181`ステップで発散せず(`diverged=False`)、最終検証 bits-per-byte が最小だった x4.0(学習率 1.2e-3)を採用した。

| 学習率倍率     |       学習率 | 発散  | 最終 bits-per-byte |
| -------------- | -----------: | ----- | -----------------: |
| x0.5           |     1.50e-04 | False |             2.2082 |
| x1.0           |     3.00e-04 | False |             1.9767 |
| x2.0           |     6.00e-04 | False |             1.7678 |
| **x4.0(採用)** | **1.20e-03** | False |         **1.6622** |

**注記(後日の調査による)**: 較正の評価の間隔は`max(1, num_steps // 2) = 1090`であり、上の表の「最終 bits-per-byte」は最終ステップ(2,181 ステップ)ではなく 2,180 ステップ時点の評価値である。全条件で同じ時点の値なので、学習率の選定には影響しない。

gradient clipping 閾値は、採用した学習率での較正実行における勾配ノルムの 90% 分位点として **0.6341** に決定した。

本番学習(2,181 ステップ)の実測時間は **359.1 秒** であり、5.9 節末尾のスケーリング外挿(5.8 分 ≈ 348 秒)とほぼ一致した。学習曲線(下図)では、訓練損失が約 9(ほぼ一様分布相当)から単調に低下し 2,181 ステップ時点で 3.5〜4 nats 付近まで下がっており、バッチ由来のノイズはあるものの発散の兆候はない。検証 bits-per-byte も最初の評価(200 ステップ、2.48)から最終評価(2,000 ステップ時点、1.6707)まで単調に減少し、前提条件 P0 の閾値`PRECONDITION_BPB_THRESHOLD=2.5`を最初の評価時点から一貫して下回っている。保存した 2,181 ステップ時点の重みでは 1.6681 である。

**訂正(後日の調査による)**: 当初、この段落は 1.6707 を「最終ステップ(2,181 ステップ)」の値と記していた。1.6707 は 200 ステップごとの評価の最後である 2,000 ステップ時点の値であり、2,181 ステップ時点の値ではない(5.10 節の注記を参照)。7 節の生成実験は 2,181 ステップ時点の重みを使っており、この訂正の影響を受けない。

**最終評価(2,000 ステップ時点)の検証 bits-per-byte = 1.6707 は閾値 2.5 を大きく下回り、前提条件 P0(学習の進行)は成立する。**

### 8.2 実験 A: 退化現象(Degeneration)

| グループ | 手法                 |   n |   mean |    std |
| -------- | -------------------- | --: | -----: | -----: |
| A        | 貪欲法・ビームサーチ |  16 | 0.5205 | 0.2101 |
| B        | temperature・top-p   |  80 | 0.0002 | 0.0018 |

$\Delta = \bar{x}_B - \bar{x}_A = 0.0002 - 0.5205 = -0.5203$、閾値 $2 \times \mathrm{SE}(\Delta) = 0.1050$。$\Delta$ は期待方向(負)に閾値を大きく超えており、**判定 = 支持** である。

**貪欲法・ビームサーチのような最大化に基づく decoding は、temperature・top-p サンプリングよりも明確に高い 4-gram 重複率を示した。** これは Holtzman et al., 2020 の理論的主張(3.4 節)を裏付ける結果である。5.11 節の生成ヘルパー動作確認セル(短い 1 プロンプト・10 トークンのみの簡易的な例であり、実験 A 本体の 8 プロンプト分の生成とは別のサンプルである点に注意)でも、同様の傾向を定性的に確認できる。

```
greedy: Conjugable words (verbed by the name of the same name)
beam  : Conjugable words (verbed into the name of the same name)
temp  : Conjugable words (verpended for the seat use) for the Dec
top-p : Conjugable words (verpended for the nativature), the name
```

greedy・beam はいずれも文末が「the name of the same name」「the name」で終わっており、短い例ながら同一フレーズの反復が観察できる。一方 temperature・top-p は同じ長さでもより多様な語が現れている。実験 A 本体の定量的な結果(グループ A の平均重複率 0.5205 はグループ B の 0.0002 の 2000 倍以上)が主たる根拠であり、この生成例はそれを補強する定性的な一例として位置づける。

### 8.3 実験 B: top-k と top-p の適応性の違い

前提条件 P1(全プロンプトのエントロピー標準偏差 > 0.05)は、8 プロンプト全てで標準偏差が 0.90〜1.85(`entropy_stds`、7.5 節のセル出力)と閾値を大きく上回っており **成立する**。

プロンプトごとの相関係数(エントロピー vs top-p 候補集合サイズ): [0.837, 0.864, 0.856, 0.854, 0.589, 0.878, 0.919, 0.890]。ゼロ基準との対比で $\Delta = 0.8358$、閾値 $2 \times \mathrm{SE}(\Delta) = 0.0726$。$\Delta$ は期待方向(正)に閾値を大きく超えており、**判定 = 支持** である。

7.6 節の散布図(実験 B、プロンプト 0)でも、top-p(青丸)の候補集合サイズはエントロピーが低い(0〜2 nats)領域ではほぼ 0〜数十個に留まる一方、エントロピーが高くなる(5〜6 nats)につれて候補集合サイズが数百〜1000 近くまで拡大しており、右肩上がりの関係が視覚的にも明瞭に確認できる。対照的に top-k(オレンジ x)の候補集合サイズはエントロピーによらず常に`TOP_K_MAIN=40`の水平な帯に張り付いている。**top-p が分布のエントロピーに応じて候補集合サイズを動的に変化させる一方、top-k は常に固定サイズであるという 3.6 節の理論的予想を、定量的な相関係数と散布図の両方が支持している。**

### 8.4 実験 C(補助・定性): temperature による多様性と一貫性のトレードオフ

**この実験には判定基準を設けていないため、以下は観察であり結論(支持・反証等)ではない。**

| temperature | distinct-3(多様性) | self-NLL(nats、一貫性の代理指標) |
| ----------- | -----------------: | -------------------------------: |
| 0.5         |             0.9194 |                           2.1466 |
| 0.7         |             0.9742 |                           2.7501 |
| 1.0         |             0.9988 |                           4.3752 |
| 1.4         |             1.0000 |                           6.7058 |
| 2.0         |             1.0000 |                           9.4008 |

temperature を上げると distinct-3 は単調に増加し、T=1.0 付近でほぼ上限(1.0 に近い値)に達して以降は頭打ちになる一方、self-NLL(モデル自身にとっての生成テキストの意外性、一貫性の代理指標)は上限に達したあとも単調に悪化し続けている。多様性が飽和したあとも一貫性の代理指標だけが悪化し続けるという非対称な関係が観察されており、3.6 節・7.7 節で述べた理論的な予想(多様性と一貫性のトレードオフ)とおおむね整合する観察である。

### 8.5 実験 D(補助・定性): beam size と生成確率・多様性のトレードオフ

**この実験には判定基準を設けていないため、以下は観察であり結論(支持・反証等)ではない。**

| beam size | best score(平均、長さ正規化対数尤度) | beam 内 distinct-3(多様性) |
| --------- | -----------------------------------: | -------------------------: |
| 1         |                              -6.3028 |    N/A(beam_size=1 のため) |
| 2         |                              -6.0012 |                     0.2440 |
| 4         |                              -5.8373 |                     0.1447 |
| 8         |                              -5.0329 |                     0.0868 |

beam size を増やすと best 候補のスコアは単調に改善する一方、beam 内候補間の多様性(distinct-3)は単調に低下しており、beam size を広げるほど探索の初期に分岐した近傍の系列が候補を占めるようになるという 3.5 節の理論的な予想と整合する観察である。

スモークテスト(ローカル・未学習に近いモデル)では、beam size を変えても best score がすべて同じ値になった。本番実行(学習済みモデル)では beam size 間で明確に差がつき、スモークテストの結果は本番スケールの結果を予測しなかった。詳細は付録 A.2。

### 8.6 Hugging Face Hub へのアップロード

モデルの重み・設定・モデルカードは [kojikojiprg/ai-theories-small-gpt-en](https://huggingface.co/kojikojiprg/ai-theories-small-gpt-en) に公開した(5.12 節のセル出力「アップロード完了」を確認済み)。**トークナイザ(`tokenizer.json`)はこのリポジトリに同梱していない。** トークナイザには [kojikojiprg/ai-theories-tokenizer-en](https://huggingface.co/kojikojiprg/ai-theories-tokenizer-en) を使用すること(モデルカードに明記済み)。

アップロード前にリポジトリを削除してから作り直しており(5.12 節のセル出力「削除してから作り直す: kojikojiprg/ai-theories-small-gpt-en」を確認済み)、過去のアップロードで同梱されていた`tokenizer.json`は残っていない。

アップロード前の不変条件アサーション(6 節(1)、state_dict の保存・再読み込みラウンドトリップで logits が完全一致すること)は、本番学習したモデルに対しても成立を確認している(6 節のセル出力「OK: state_dict 保存・再読み込みラウンドトリップで logits が完全一致」)。これにより、アップロードしたアーティファクト(state_dict)がネットワーク呼び出しを介さずに検証済みの正しいものであることを担保している。

付録(実行の手順・パイロット・設計の改訂の記録など)は、GitHub のノートブックを参照。

https://github.com/kojikojiprg/ai-theories/blob/main/theories/02_pretraining/008_decoding_strategies.ipynb


## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/02_pretraining/008_decoding_strategies.ipynb
