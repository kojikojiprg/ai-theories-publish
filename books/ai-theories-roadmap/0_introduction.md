---
title: "はじめに"
---

## この本について

本書は [ai-theories](https://github.com/kojikojiprg/ai-theories) で学習している LLM(大規模言語モデル)・VLM(視覚言語モデル)の理論学習ロードマップです。古典的な機械学習理論(SVM、決定木など)は対象外とし、**LLM / VLMに関する理論に限定**しています。

- 理論的に本質的な部分は、可能な限り**PyTorchによるスクラッチ実装**で確認します(Attention計算、正規化、LoRAの低ランク分解など)。
- 学習・実験は**Google Colab無料枠(T4 GPUなど)で完結する規模**のモデル・データセットで構成しています。

## 本書の構成

各章は、1つの学習トピックに対応しています。

- 章の本文には、トピックのカテゴリ・前提知識・扱う内容の概要を掲載します。
- 対応するノートブックがすでにZenn記事化されている場合は、章末にその記事へのリンクを掲載します。
- **記事がまだ準備中のトピックは、概要のみの掲載となります**(🚧 マークで示します)。準備が整い次第、記事へのリンクを追記します。

本書は [ai-theories](https://github.com/kojikojiprg/ai-theories) の `theories/README.md` と記事の生成状況をもとに自動生成しています。

## コードのライセンス

本書内で参照するコード・ノートブックのライセンスは、[ai-theories](https://github.com/kojikojiprg/ai-theories) 本体の LICENSE に準拠します。

## リンク

- [ai-theories(GitHubリポジトリ)](https://github.com/kojikojiprg/ai-theories)
