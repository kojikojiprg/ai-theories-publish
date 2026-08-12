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
| 005 | トークナイザ(🚧 準備中) | 02_pretraining | なし / None | BPE(Byte Pair Encoding)の学習アルゴリズムと SentencePiece の仕組みを扱い、語彙サイズ(vocabulary size)と系列長のトレードオフを検討する |
| 006 | 小型 GPT の事前学習(🚧 準備中) | 02_pretraining | 003, 004, 005 | 自己回帰言語モデリング(Autoregressive Language Modeling)の学習ループを実装し、loss 曲線と perplexity で学習の進行を評価する |
| 007 | 学習の安定化(🚧 準備中) | 02_pretraining | 006 | 002 で観測した正規化後置の勾配の不均衡、および 004 で観測した正規化を欠いた条件のシード間のばらつきの増大を踏まえ、AdamW、warmup + cosine スケジュール、gradient clipping、mixed precision など、大規模言語モデルの学習を安定化させる技術を扱う |
| 008 | デコーディング戦略(🚧 準備中) | 02_pretraining | 006 | greedy / temperature / top-k / top-p / beam search など複数のデコーディング手法(Decoding Strategies)を実装し、生成品質を比較する |
| 009 | スケーリング則(🚧 準備中) | 02_pretraining | 007 | Kaplan らおよび Chinchilla のスケーリング則(Scaling Laws)を扱い、計算量最適(compute-optimal)なモデルサイズとデータ量の関係を導く |
| 010 | KV キャッシュと推論の計算量(🚧 準備中) | 03_efficient_training | 003, 008 | KV キャッシュ(KV Cache)のメモリ量、MQA(Multi-Query Attention)/ GQA(Grouped-Query Attention)、prefill と decode フェーズの違いを扱う。逐次生成時の位置インデックス(003 で導入した`positions`引数)の扱いも扱う |
| 011 | LoRA(🚧 準備中) | 03_efficient_training | 006 | 低ランク分解(Low-Rank Decomposition)による差分学習(LoRA: Low-Rank Adaptation)を rank・alpha のパラメータとともにスクラッチ実装する |
| 012 | 量子化の基礎(🚧 準備中) | 03_efficient_training | 011 | INT8 / NF4 などの量子化手法と量子化誤差(Quantization Error)を扱い、QLoRA の位置づけを整理する |
| 013 | Flash Attention(🚧 準備中) | 03_efficient_training | 010 | タイリング(Tiling)と online softmax によるメモリ帯域律速の解消を扱い、Flash Attention の計算手順を追う |
| 014 | 長文脈拡張(🚧 準備中) | 03_efficient_training | 003, 013 | RoPE のスケーリング手法(位置補間、NTK-aware スケーリング、YaRN など)による長文脈(Long Context)への拡張技術を扱う |
| 015 | SFT(指示チューニング)(🚧 準備中) | 04_alignment | 011 | 指示データ(Instruction Data)の形式と損失マスク(Loss Masking)を扱い、LoRA を用いた SFT(Supervised Fine-Tuning)を実装する |
| 016 | 報酬モデルと RLHF(🚧 準備中) | 04_alignment | 015 | 選好データ(Preference Data)と Bradley-Terry モデルによる報酬モデル(Reward Model)の学習、および PPO の枠組みを理論中心に扱う |
| 017 | DPO(🚧 準備中) | 04_alignment | 016 | RLHF の閉形式解(Closed-Form Solution)から DPO(Direct Preference Optimization)損失を導出し、スクラッチ実装する |
| 018 | ViT と画像パッチ埋め込み(🚧 準備中) | 05_vision_language | 002 | 画像のパッチ分割(Patch Embedding)と [CLS] トークンによる、Transformer の画像への適用(ViT: Vision Transformer)を扱う |
| 019 | CLIP と対照学習(🚧 準備中) | 05_vision_language | 018 | InfoNCE 損失による対照学習(Contrastive Learning)を用いて画像 encoder とテキスト encoder を共同学習する CLIP の zero-shot 分類を扱う |
| 020 | LLaVA 型 Vision-Language 連結(🚧 準備中) | 05_vision_language | 015, 019 | projection 層による視覚特徴の LLM 埋め込み空間への写像と、2 段階学習(事前学習 + 指示チューニング)による Vision-Language モデルの構築を扱う |
| 021 | Mixture of Experts (MoE)(🚧 準備中) | 06_architectures | 002 | ルーティング(Routing)、top-k gating、負荷分散損失(Load Balancing Loss)など、MoE(Mixture of Experts)アーキテクチャの仕組みを扱う |
| 022 | State Space Model / Mamba(🚧 準備中) | 06_architectures | 002 | 状態空間モデル(State Space Model)の離散化と選択的 SSM(Selective SSM)の仕組みを扱い、線形時間での系列モデリングを検証する |
| 023 | テキスト埋め込みと retriever(🚧 準備中) | 07_retrieval | 019 | 文埋め込み(Sentence Embedding)と対照学習による dense retriever を扱い、dual encoder 構造を実装する |
| 024 | ANN 検索とリランキング(🚧 準備中) | 07_retrieval | 023 | 近似最近傍探索(ANN: Approximate Nearest Neighbor、例: HNSW)と cross-encoder によるリランキング(Reranking)を扱う |

## コードのライセンス

本書内で参照するコード・ノートブックのライセンスは、[ai-theories](https://github.com/kojikojiprg/ai-theories) 本体の LICENSE に準拠します。

## リンク

- [ai-theories(GitHubリポジトリ)](https://github.com/kojikojiprg/ai-theories)
