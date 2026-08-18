---
title: "はじめに"
---

## この本について

本書は [ai-theories](https://github.com/kojikojiprg/ai-theories) で学習している LLM(大規模言語モデル)・VLM(視覚言語モデル)の理論学習ロードマップです。古典的な機械学習理論(SVM、決定木など)は対象外とし、**LLM / VLMに関する理論に限定**しています。

- 理論的に本質的な部分は、可能な限り**PyTorchによるスクラッチ実装**で確認します(Attention計算、正規化、LoRAの低ランク分解など)。
- 学習・実験は**Google Colab無料枠(T4 GPUなど)で完結する規模**のモデル・データセットで構成しています。

## 本書の構成

以降の章は、[ai-theories](https://github.com/kojikojiprg/ai-theories) の各学習トピックのノートブックをそのまま章として収録したものです。下表は `theories/README.md` の推奨学習順序表と同じ構成(トピック名・カテゴリ・前提知識・扱う内容)の一覧です。トピック名がリンクになっているものは既に章として収録済みで、リンクをクリックすると該当の章に移動します。**まだ章になっていないトピックは「🚧 準備中」と表示**しています。

## 推奨学習順序 / Recommended Order

| # | トピック / Topic | カテゴリ / Category | 前提知識 / Prerequisites | 扱う内容 / Contents |
| --- | --- | --- | --- | --- |
| 001 | [注意機構(Attention Mechanism)](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/001_attention_mechanism-theory) | 01_foundations | なし / None | Query / Key / Value、Scaled Dot-Product Attention とスケーリング係数 $\sqrt{d_k}$ の導出、因果マスク、多頭注意機構(Multi-Head Attention)。実装は`src/layers/attention.py`にスクラッチ実装し、重みの可視化と copy task の学習で検証する |
| 002 | [Transformer Block](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/002_transformer_block-theory) | 01_foundations | 001 | 残差接続(Residual Connection)・層正規化(Layer Normalization)・順伝播ネットワーク(Feed-Forward Network)をスクラッチ実装し、正規化前置(Pre-Layer Normalization)と正規化後置(Post-Layer Normalization)の違いを扱う。001 で実装した多頭注意機構(Multi-Head Attention)を組み込んだ Encoder Block と、交差注意(cross-attention)を含む Decoder Block の両方を実装する。系列の順序情報を与えるため、正弦波(sinusoidal)方式の位置エンコーディング(Positional Encoding)を暫定的に導入する(各方式の比較は 003) |
| 003 | [位置エンコーディング / RoPE](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/003_positional_encoding_rope-theory) | 01_foundations | 002 | 002 で暫定導入した正弦波(sinusoidal)方式に加え、学習可能な絶対位置埋め込み(Learned Absolute Positional Embedding)・相対位置エンコーディング(Shaw et al. 方式・T5 の相対位置バイアス)・ALiBi(Attention with Linear Biases)を比較し、回転位置エンコーディング(RoPE: Rotary Position Embedding)の数学的導出と実装を扱う。実装は`src/layers/positional_encoding.py`にスクラッチ実装し、`MultiHeadAttention`に注入する形で組み込む。可変長 copy task による 7 条件の比較(学習長内の精度と学習長を超える外挿性能)および Attention 重みの定量分析で検証する |
| 004 | [正規化と活性化の系譜](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/004_normalization_and_activation-theory) | 01_foundations | 002 | 002 でスクラッチ実装した層正規化(Layer Normalization)を起点に RMSNorm(Root Mean Square Normalization)への変遷、ReLU から GELU、GLU(Gated Linear Unit)を経て SwiGLU に至る活性化関数の変遷を扱う。平均減算の有無・分散除算の有無による除去実験、常に負のユニット(always-negative unit)の測定、**乗法的相互作用の合成タスクとそれを含まない陰性対照(negative control)タスクの比較** によって各機構の寄与を検証する。学習には条件比較のための最小限の文字レベル言語モデリングを用い、条件間比較の評価ノイズを抑えるため固定した評価用バッチ集合を使う(本格的な自己回帰言語モデリングの事前学習は 006 で扱う) |
| 005 | [トークナイザと部分語分割](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/005_tokenizer-theory) | 02_pretraining | 001 | BPE(Byte Pair Encoding)の学習・符号化(タイブレーク規則を固定、空白をチャンク先頭に保持する可逆な事前分割)とバイトレベル BPE(byte-level BPE)をスクラッチ実装し、WordPiece のスコア関数を理論として位置づける。Unigram 言語モデル(Unigram Language Model)の Viterbi 最尤分割をスクラッチ実装し(語彙学習は sentencepiece に委譲)、SentencePiece が「アルゴリズムではなく実装である」ことを明確にする。英語・日本語(日本語版 Wikipedia、CC BY-SA 4.0、記事とリビジョン ID を固定して取得)・コードの 3 ドメインで語彙サイズ別の fertility を比較し、空白による事前分割(pre-tokenization)が日本語で機能しないことを英語・コードを陰性対照(negative control)として検証する。初期語彙方式(バイトレベル・文字レベル)と語彙サイズの相互作用を語彙サイズの掃引で検証し、語彙サイズと系列長・計算量のトレードオフを理論計算で示す |
| 006 | [小型 GPT の事前学習](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/006_pretraining_small_gpt-theory) | 02_pretraining | 003, 004, 005 | 001〜005 の部品(Transformer Block、RoPE、RMSNorm、SwiGLU、トークナイザ)を統合し、decoder-only な自己回帰言語モデル(`GPTLanguageModel`)の事前学習を初めて最後まで実行する。重み共有(weight tying)、perplexity と bits-per-byte の使い分け(トークナイザ間比較には bits-per-byte、言語間の絶対値比較はしない)を理論として扱う。日本語版・英語版 Wikipedia コーパス(段階 1 で取得、各 20 MB 以上)を用い、文字レベル・バイトレベル BPE(公比 2 の等比数列で 4 語彙サイズ)・Unigram 言語モデル(byte fallback により未知語による情報損失を排除、BPE の 1 条件と語彙サイズを厳密に一致させる)の計 5 トークナイザ条件を比較する(実験 A〜H)。ノイズ床を日英それぞれ 5 シードで測定し(標本標準偏差の過小バイアスを抑えるため)、対比量の誤差伝播に基づく判定閾値(実験 D・E・F)で支持 / 反証 / 判定不能の 3 値判定を行う。非埋め込みパラメータ数を条件間で揃えることで語彙サイズの影響を分離し、実験 G は model サイズを固定してステップ数のみを増やす。学習ステップ数は全言語・全トークナイザ条件の訓練トークン数を実測し、エポック上限(実験 C〜F は 1/2 エポック、実験 G は 1 エポック)から計算式で決定する。**本番実行の結果、部分語分割(BPE・Unigram)が文字レベルより優れるかは言語に依存することが判明した**(日本語は文字レベルが最良、英語は文字レベルが最悪で bits-per-byte の順序が言語間で反転する) |
| 007 | 学習の安定化(🚧 準備中) | 02_pretraining | 006 | 002 で観測した正規化後置(Post-Layer Normalization)の勾配の不均衡、および 004 で観測した正規化を欠いた条件のシード間のばらつきの増大を踏まえ、正規化前置(Pre-Layer Normalization)/ 正規化後置(Post-Layer Normalization)と学習率を独立変数として不安定性を意図的に誘発し、AdamW(Decoupled Weight Decay Regularization)・warmup + cosine スケジュール・gradient clipping(勾配クリッピング)が学習を安定化させる効果を検証する。較正を本番と同じステップ数で行い、学習率の水準・gradient clipping の閾値(勾配ノルムの分位点方式)・シード数を決定する(006 のスケーリング外挿の手法を踏襲)。学習が実際に進んでいること・gradient clipping が実際に発動していることを、検証したい仮説とは独立な前提条件(pre-condition)として本番実行の前に宣言し、前提条件が不成立の主張は支持 / 反証と判定せず「前提不成立」として記録する。勾配ノルムのピーク / 平均比率・最大単一ステップ損失上昇幅・固定ステップ数終了時点の損失値を対比量として、対比量の標準偏差に基づく判定閾値で支持 / 反証 / 判定不能 / 前提不成立の 4 値判定を行う。AdamW については、二次モーメント推定によるスケーリングから重み減衰を独立させる効果を、勾配の二次モーメントが異なるパラメータ群間の実効的な減衰強度の乖離として直接検証する合成タスクの実験を別途設ける(混合精度学習は 03_efficient_training の該当トピックに切り出す) |
| 008 | デコーディング戦略(🚧 準備中) | 02_pretraining | 006 | greedy / temperature / top-k / top-p / beam search など複数のデコーディング手法(Decoding Strategies)を実装し、生成品質を比較する |
| 009 | スケーリング則(🚧 準備中) | 02_pretraining | 007 | Kaplan らおよび Chinchilla のスケーリング則(Scaling Laws)を扱い、計算量最適(compute-optimal)なモデルサイズとデータ量の関係を導く |
| 010 | KV キャッシュと推論の計算量(🚧 準備中) | 03_efficient_training | 003, 008 | KV キャッシュ(KV Cache)のメモリ量、MQA(Multi-Query Attention)/ GQA(Grouped-Query Attention)、prefill と decode フェーズの違いを扱う。逐次生成時の位置インデックス(003 で導入した`positions`引数)の扱いも扱う |
| 011 | 混合精度学習(Mixed Precision Training)(🚧 準備中) | 03_efficient_training | 006 | FP16 / BF16 による数値精度の低減と、勾配の underflow を防ぐための loss scaling を扱う。勾配のスケールが小さい層で発生するアンダーフローの現象と、gradient scaler による対処を検証する |
| 012 | LoRA(🚧 準備中) | 03_efficient_training | 006 | 低ランク分解(Low-Rank Decomposition)による差分学習(LoRA: Low-Rank Adaptation)を rank・alpha のパラメータとともにスクラッチ実装する |
| 013 | 量子化の基礎(🚧 準備中) | 03_efficient_training | 012 | INT8 / NF4 などの量子化手法と量子化誤差(Quantization Error)を扱い、QLoRA の位置づけを整理する |
| 014 | Flash Attention(🚧 準備中) | 03_efficient_training | 010 | タイリング(Tiling)と online softmax によるメモリ帯域律速の解消を扱い、Flash Attention の計算手順を追う |
| 015 | 長文脈拡張(🚧 準備中) | 03_efficient_training | 003, 014 | RoPE のスケーリング手法(位置補間、NTK-aware スケーリング、YaRN など)による長文脈(Long Context)への拡張技術を扱う |
| 016 | SFT(指示チューニング)(🚧 準備中) | 04_alignment | 012 | 指示データ(Instruction Data)の形式と損失マスク(Loss Masking)を扱い、LoRA を用いた SFT(Supervised Fine-Tuning)を実装する |
| 017 | 報酬モデルと RLHF(🚧 準備中) | 04_alignment | 016 | 選好データ(Preference Data)と Bradley-Terry モデルによる報酬モデル(Reward Model)の学習、および PPO の枠組みを理論中心に扱う |
| 018 | DPO(🚧 準備中) | 04_alignment | 017 | RLHF の閉形式解(Closed-Form Solution)から DPO(Direct Preference Optimization)損失を導出し、スクラッチ実装する |
| 019 | ViT と画像パッチ埋め込み(🚧 準備中) | 05_vision_language | 002 | 画像のパッチ分割(Patch Embedding)と [CLS] トークンによる、Transformer の画像への適用(ViT: Vision Transformer)を扱う |
| 020 | CLIP と対照学習(🚧 準備中) | 05_vision_language | 019 | InfoNCE 損失による対照学習(Contrastive Learning)を用いて画像 encoder とテキスト encoder を共同学習する CLIP の zero-shot 分類を扱う |
| 021 | LLaVA 型 Vision-Language 連結(🚧 準備中) | 05_vision_language | 016, 020 | projection 層による視覚特徴の LLM 埋め込み空間への写像と、2 段階学習(事前学習 + 指示チューニング)による Vision-Language モデルの構築を扱う |
| 022 | Mixture of Experts (MoE)(🚧 準備中) | 06_architectures | 002 | ルーティング(Routing)、top-k gating、負荷分散損失(Load Balancing Loss)など、MoE(Mixture of Experts)アーキテクチャの仕組みを扱う |
| 023 | State Space Model / Mamba(🚧 準備中) | 06_architectures | 002 | 状態空間モデル(State Space Model)の離散化と選択的 SSM(Selective SSM)の仕組みを扱い、線形時間での系列モデリングを検証する |
| 024 | テキスト埋め込みと retriever(🚧 準備中) | 07_retrieval | 020 | 文埋め込み(Sentence Embedding)と対照学習による dense retriever を扱い、dual encoder 構造を実装する |
| 025 | ANN 検索とリランキング(🚧 準備中) | 07_retrieval | 024 | 近似最近傍探索(ANN: Approximate Nearest Neighbor、例: HNSW)と cross-encoder によるリランキング(Reranking)を扱う |

## コードのライセンス

本書内で参照するコード・ノートブックのライセンスは、[ai-theories](https://github.com/kojikojiprg/ai-theories) 本体の LICENSE に準拠します。

## リンク

- [ai-theories(GitHubリポジトリ)](https://github.com/kojikojiprg/ai-theories)
