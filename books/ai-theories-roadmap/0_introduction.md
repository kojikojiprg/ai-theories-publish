---
title: "はじめに"
---

## この本について

本書は [ai-theories](https://github.com/kojikojiprg/ai-theories) で学習している LLM(大規模言語モデル)・VLM(視覚言語モデル)の理論学習ロードマップです。古典的な機械学習理論(SVM、決定木など)は対象外とし、 **LLM / VLM に関する理論に限定** しています。

- 理論的に本質的な部分は、可能な限り **PyTorch によるスクラッチ実装** で確認します(Attention 計算、正規化、LoRA の低ランク分解など)。
- 学習・実験は **Google Colab 無料枠(T4 GPU など)で完結する規模** のモデル・データセットで構成しています。

## 本書の構成

以降の章は、[ai-theories](https://github.com/kojikojiprg/ai-theories) の各学習トピックのノートブックをそのまま章として収録したものです。この章には、トピックの全体像を示す図と推奨学習順序の表を、 `theories/README.md` から転載しています。表の「#」の列の番号をクリックすると、該当の章に移動します。 **まだ章になっていないトピックは「🚧 準備中」と表示** します。

## 全体像 / Overview

各トピックが、LLM(Large Language Model、大規模言語モデル)や AI エージェントのどの部分に当たるかを 2 つの図で示す。ノードの番号は下の学習順序表の番号に対応する。破線で薄く描いたノードは、まだ扱っていない部分である。Mermaid の図は GitHub 上で描画される。

### 図 1: 推論時の構成 / At Inference

AI エージェントの中で LLM がどう動くかを示す。

![図 1: 推論時の構成 / At Inference](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/introduction/overview-1-6181d336.png)

### 図 2: モデルを作る流れ / How Models Are Built

事前学習(Pretraining)から、適応(Adaptation)、選好への整合(Preference Alignment)、推論(Inference)へ進む段階に、各トピックを置く。

![図 2: モデルを作る流れ / How Models Are Built](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/introduction/overview-2-ea37a40d.png)

## 推奨学習順序 / Recommended Order

理論はカテゴリごとに整理していますが、**表の番号順に学習する** ことを推奨します。
The theories are organized by category, but I recommend **studying them in the numbered order shown in the table**.

| # | トピック / Topic | カテゴリ / Category | 前提知識 / Prerequisites | 何を学ぶか / What You Learn |
| --- | --- | --- | --- | --- |
| [001](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/001_attention_mechanism) | 注意機構(Attention Mechanism) | 01_foundations | なし / None | Query・Key・Value による注意機構と多頭注意機構の定式化、スケーリング係数の必要性を理解する |
| [002](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/002_transformer_block) | Transformer Block | 01_foundations | 001 | 残差接続・層正規化・順伝播ネットワークを組み合わせた Transformer Block の構成と、正規化の配置の違いを理解する |
| [003](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/003_positional_encoding_rope) | 位置エンコーディング / RoPE | 01_foundations | 002 | 位置情報を与える各方式を比較し、RoPE が相対位置を表す仕組みを導出して理解する |
| [004](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/004_normalization_and_activation) | 正規化と活性化の系譜 | 01_foundations | 002 | 層正規化から RMSNorm へ、ReLU から SwiGLU へ至る変遷の、何が本質的な変化かを切り分けて理解する |
| [005](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/005_tokenizer) | トークナイザと部分語分割 | 02_pretraining | 001 | 部分語分割(BPE・Unigram 言語モデル)の仕組みと、語彙サイズに伴う圧縮効率・計算量のトレードオフを理解する |
| [006](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/006_pretraining_small_gpt) | 小型 GPT の事前学習 | 02_pretraining | 003, 004, 005 | 部品を統合して自己回帰言語モデルを事前学習し、トークナイザの選び方が言語ごとにどう効くかを理解する |
| [007](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/007_training_stabilization) | 学習の安定化 | 02_pretraining | 006 | AdamW・warmup・gradient clipping が、どの種類の学習の不安定性に効くかを切り分けて理解する |
| [008](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/008_decoding_strategies) | デコーディング戦略 | 02_pretraining | 006, 007 | 貪欲法・サンプリング・ビームサーチなどのデコーディング戦略の違いと、退化現象の原因を理解する |
| [009](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/009_scaling_laws) | スケーリング則(Scaling Laws) | 02_pretraining | 006, 007 | スケーリング則と計算量最適なモデルサイズの考え方を、小さな規模で検証しながら理解する |
| [010](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/010_kv_cache_and_inference_compute) | KV キャッシュと推論の計算量 | 03_efficient_training | 003, 006, 008 | KV キャッシュによる生成の高速化、prefill と decode の性質の違い、Key / Value ヘッド数の削減を理解する |
| [011](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/011_mixed_precision_training-theory) | 混合精度学習(Mixed Precision Training) | 03_efficient_training | 006, 007 | FP16・BF16 の数値の性質と、損失スケーリング・FP32 マスター重みが混合精度学習を成り立たせる仕組みを理解する |
| [012](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/012_low_rank_adaptation) | LoRA(Low-Rank Adaptation) | 03_efficient_training | 006, 007, 008 | 事前学習済みの重みを凍結して低ランクの更新だけを学習する LoRA の仕組みと、それが効く理由を理解する |
| [013](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/013_quantization_basics) | 量子化の基礎(Quantization Basics) | 03_efficient_training | 010, 012 | 重みの量子化と誤差の理論値、NF4、量子化したモデルに LoRA を載せる QLoRA(Quantized LoRA)を理解する |
| [014](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/014_flash_attention) | Flash Attention | 03_efficient_training | 010 | Attention のメモリ量を系列長の 2 乗から下げる、Flash Attention のタイリングと online softmax を理解する |
| [015](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/015_long_context_extension-theory) | 長文脈拡張(Long Context Extension) | 03_efficient_training | 003, 008, 014 | 学習済みの RoPE モデルの文脈長を、周波数の変換(位置補間・YaRN など)で伸ばす方法を理解する |
| [016](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/016_supervised_fine_tuning) | SFT(指示チューニング) | 04_alignment | 008, 012 | 指示チューニングの目的と、損失マスクの扱い、必要なデータ量の考え方を理解する |
| [017](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/017_reward_model_and_rlhf-theory) | 報酬モデルと RLHF | 04_alignment | 016 | 報酬モデルの学習と、人間のフィードバックによる強化学習(RLHF)の 3 段階、報酬の過最適化を理解する |
| [018](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/018_direct_preference_optimization-theory) | DPO と IPO | 04_alignment | 017 | RLHF の目的関数から DPO の損失を導出し、IPO との違いと暗黙の報酬の性質を理解する |
| [019](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/019_vision_transformer-theory) | ViT と画像パッチ埋め込み | 05_vision_language | 002, 003 | 画像をパッチに分けて埋め込む ViT の仕組みと、位置埋め込み・帰納バイアスの役割を理解する |
| [020](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/020_clip_contrastive_learning) | CLIP と対照学習 | 05_vision_language | 019 | 画像とテキストを対照学習で結ぶ CLIP と、損失関数・負例の数の効き方を理解する |
| [021](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/021_llava_visual_instruction_tuning) | LLaVA 型 Vision-Language 連結と視覚指示チューニング | 05_vision_language | 016, 020 | 画像の特徴を言語モデルの入力につなぐ LLaVA 型の構成と、2 段階の学習を理解する |
| [022](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/022_mixture_of_experts-theory) | Mixture of Experts (MoE) | 06_architectures | 006, 007 | 順伝播ネットワークを複数のエキスパートとルーターに置き換える MoE の仕組みと、負荷分散を理解する |
| [023](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/023_state_space_model_mamba-theory) | State Space Model / Mamba | 06_architectures | 002, 008, 010 | 状態空間モデルと選択機構を持つ Mamba が、注意機構と計算量・長さの外挿でどう違うかを理解する |
| [024](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/024_text_embedding_and_retriever-theory) | テキスト埋め込みと retriever | 07_retrieval | 008, 020 | 言語モデルを埋め込みモデルに転用する仕組み(プーリング・対照学習)と、dense retriever による検索を理解する |
| [025](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/025_ann_search_and_reranking-theory) | ANN 検索とリランキング | 07_retrieval | 024, 008, 013 | 近似最近傍探索(転置ファイル・HNSW・積量子化)と、cross-encoder による並べ替えを理解する |
| [026](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/026_inverse_cloze_task_and_two_stage_retrieval) | Inverse Cloze Task と passage 単位の 2 段階検索 | 07_retrieval | 024, 025 | 検索器が何を関連とみなすかが学習で決まることと、Inverse Cloze Task、2 段階検索を理解する |

> 「未作成 / TBD」の行は今後追加予定のトピックです。追加のたびにこの表を更新します。
> Rows marked "未作成 / TBD" are planned topics; this table is updated whenever a topic is added.

## コードのライセンス

本書内で参照するコード・ノートブックのライセンスは、[ai-theories](https://github.com/kojikojiprg/ai-theories) 本体の LICENSE に準拠します。

## リンク

- [ai-theories(GitHub リポジトリ)](https://github.com/kojikojiprg/ai-theories)
