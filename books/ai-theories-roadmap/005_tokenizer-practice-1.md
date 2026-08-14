---
title: "トークナイザと部分語分割(実装・実験編 1/3)"
---

この記事は後編(実装・実験編 1/3)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/005_tokenizer-theory)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/005_tokenizer-practice-2)。

## 4. 実装方針 / Implementation Policy

CLAUDE.md のコーディング規約は、トークナイザの学習(BPE の学習など)を手段として使う場合は`tokenizers`などの既存ライブラリを活用してよいと述べている。しかし **本トピックはトークナイザそのものを主題とする** ため、この例外規定は適用せず、BPE の学習・符号化と Unigram 言語モデルの Viterbi 最尤分割はスクラッチ実装する。

| 機能 | 配置場所 | 実装/委譲 |
|---|---|---|
| BPE の学習(`learn_bpe`)・符号化(`BPETokenizer`)、バイトレベル初期化 | `src/data/tokenizer.py` | スクラッチ実装 |
| 事前分割(`pretokenize`、`chunk_split_mode`) | `src/data/tokenizer.py` | スクラッチ実装 |
| Unigram 言語モデルの Viterbi 最尤分割(`viterbi_segment`) | `src/data/tokenizer.py` | スクラッチ実装 |
| Unigram 言語モデルの語彙学習(EM ベースの反復的縮小)、`UnigramTokenizer`・`train_unigram_model` | `src/data/tokenizer.py` | **sentencepiece に委譲**(理由は下記) |
| WordPiece | (実装なし、3.4 節で理論のみ) | — |
| 英語・日本語・コードコーパスの取得(`load_tiny_shakespeare`・`load_japanese_corpus`・`load_code_corpus`) | `src/data/text.py` | スクラッチ実装(日本語は Wikipedia API、コードは本リポジトリ自身の`src/`) |
| fertility・未知語率・チャンク長分布・分割一致率の計算 | `src/utils/statistics.py` | スクラッチ実装 |
| 語彙サイズ×fertility 曲線、チャンク長のグループ棒グラフ、語彙サイズと計算量のトレードオフ図 | `src/utils/visualization.py` | スクラッチ実装(`plot_grouped_bar`・`plot_dual_axis_curves`を新規追加) |
| BPE と Unigram 言語モデルの分割比較・可視化 | 本ノートブック(実験セル) | ノートブック内に直接記述 |

**Unigram 言語モデルの語彙学習のみをライブラリに委ねる理由**: 3.5 節で述べた通り、語彙学習は EM アルゴリズムによる反復的なパラメータ推定と語彙削減からなる、実装がそれ自体で 1 つのトピックになりうる手続きである。本トピックが検証したいのは「Unigram 言語モデルが与えられた語彙のもとでどう分割を決めるか」(Viterbi 最尤分割というアルゴリズム)であり、「語彙をどう学習するか」ではない。そこで語彙学習は sentencepiece に委譲し、**その語彙(部分語ごとの対数確率)を受け取って最尤分割を求める部分だけをスクラッチ実装する**。この設計にはもう 1 つ利点がある。sentencepiece 自身も内部で Viterbi 最尤分割を行っており(学習した語彙で`encode`すると最尤分割が返る)、自作の`viterbi_segment`の出力と sentencepiece 自身の出力を **同一の語彙に対して** 比較できる。両者が一致すれば、それ自体が自作実装の正しさの証拠になる(実験6)。

**自作 BPE の可逆性についての注記**: 3.6 節で述べた通り、SentencePiece は空白を`▁`として扱うことで可逆な復号を実現する。本ノートブックの自作 BPE も、事前分割(`pretokenize`)の`chunk_split_mode="whitespace"`で空白をチャンクの先頭に保持する設計にしたことで、符号化結果を単純に連結するだけで元のテキストを完全に復元できる(可逆性の検証を 5.1 節に置く)。ただしこの可逆性は`chunk_split_mode="whitespace"`を選んだ場合に限られ、`chunk_split_mode="none"`は複数行にわたるテキストで改行が失われるため可逆ではない(実装上の単純化、3.2 節の`pretokenize`の docstring を参照)。

## 5. 実装 / Implementation

`src/` から必要な関数・クラスを import して使う。以下のセットアップセルは 004 までと同様、Google Colab 上でのみリポジトリを clone し、リポジトリルートをカレントディレクトリにする。


```python
# 環境セットアップ(Google Colab)
# Colab 上でのみリポジトリを clone し、リポジトリルートをカレントディレクトリにする。
# ローカル(Jupyter)実行時は、リポジトリルートで起動していればそのまま動く。
import sys

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    !git clone https://github.com/kojikojiprg/ai-theories.git
    %cd ai-theories
    !pip install uv -q
    !uv pip install --system -r requirements.txt
```

    Cloning into 'ai-theories'...
    remote: Enumerating objects: 262, done.[K
    remote: Counting objects: 100% (262/262), done.[K
    remote: Compressing objects: 100% (163/163), done.[K
    remote: Total 262 (delta 129), reused 210 (delta 85), pack-reused 0 (from 0)[K
    Receiving objects: 100% (262/262), 3.01 MiB | 10.03 MiB/s, done.
    Resolving deltas: 100% (129/129), done.
    /content/ai-theories
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m23.6/23.6 MB[0m [31m65.5 MB/s[0m eta [36m0:00:00[0m
    [?25h[2mUsing Python 3.12.13 environment at: /usr[0m
    [2K[2mResolved [1m41 packages[0m [2min 292ms[0m[0m
    [2K[2mPrepared [1m27 packages[0m [2min 47.96s[0m[0m
    [2mUninstalled [1m12 packages[0m [2min 1.07s[0m[0m
    [2K[2mInstalled [1m27 packages[0m [2min 507ms[0m[0m
     [31m-[39m [1mcuda-bindings[0m[2m==12.9.7[0m
     [32m+[39m [1mcuda-bindings[0m[2m==13.3.1[0m
     [31m-[39m [1mcuda-pathfinder[0m[2m==1.5.6[0m
     [32m+[39m [1mcuda-pathfinder[0m[2m==1.6.0[0m
     [31m-[39m [1mcuda-toolkit[0m[2m==12.8.1[0m
     [32m+[39m [1mcuda-toolkit[0m[2m==13.0.3.0[0m
     [31m-[39m [1mfilelock[0m[2m==3.29.7[0m
     [32m+[39m [1mfilelock[0m[2m==3.32.2[0m
     [31m-[39m [1mfsspec[0m[2m==2025.3.0[0m
     [32m+[39m [1mfsspec[0m[2m==2026.7.0[0m
     [31m-[39m [1mmatplotlib[0m[2m==3.10.0[0m
     [32m+[39m [1mmatplotlib[0m[2m==3.11.1[0m
     [31m-[39m [1mnumpy[0m[2m==2.0.2[0m
     [32m+[39m [1mnumpy[0m[2m==2.5.2[0m
     [32m+[39m [1mnvidia-cublas[0m[2m==13.1.1.3[0m
     [32m+[39m [1mnvidia-cuda-cupti[0m[2m==13.0.85[0m
     [32m+[39m [1mnvidia-cuda-nvrtc[0m[2m==13.0.88[0m
     [32m+[39m [1mnvidia-cuda-runtime[0m[2m==13.0.96[0m
     [32m+[39m [1mnvidia-cudnn-cu13[0m[2m==9.20.0.48[0m
     [32m+[39m [1mnvidia-cufft[0m[2m==12.0.0.61[0m
     [32m+[39m [1mnvidia-cufile[0m[2m==1.15.1.6[0m
     [32m+[39m [1mnvidia-curand[0m[2m==10.4.0.35[0m
     [32m+[39m [1mnvidia-cusolver[0m[2m==12.0.4.66[0m
     [32m+[39m [1mnvidia-cusparse[0m[2m==12.6.3.3[0m
     [32m+[39m [1mnvidia-cusparselt-cu13[0m[2m==0.8.1[0m
     [32m+[39m [1mnvidia-nccl-cu13[0m[2m==2.29.7[0m
     [32m+[39m [1mnvidia-nvjitlink[0m[2m==13.3.33[0m
     [32m+[39m [1mnvidia-nvshmem-cu13[0m[2m==3.4.5[0m
     [32m+[39m [1mnvidia-nvtx[0m[2m==13.0.85[0m
     [31m-[39m [1mpackaging[0m[2m==26.2[0m
     [32m+[39m [1mpackaging[0m[2m==26.3[0m
     [31m-[39m [1mpillow[0m[2m==11.3.0[0m
     [32m+[39m [1mpillow[0m[2m==12.3.0[0m
     [31m-[39m [1msetuptools[0m[2m==75.2.0[0m
     [32m+[39m [1msetuptools[0m[2m==84.0.0[0m
     [31m-[39m [1mtorch[0m[2m==2.11.0+cu128[0m
     [32m+[39m [1mtorch[0m[2m==2.13.0[0m
     [31m-[39m [1mtriton[0m[2m==3.6.0[0m
     [32m+[39m [1mtriton[0m[2m==3.7.1[0m



```python
# `from src...` の絶対 import が通るように、リポジトリルートを sys.path に追加する。
# (ノートブックを theories/02_pretraining/ から直接開いた場合の保険)
import os
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "src").exists():  # theories/02_pretraining/ から起動した場合
    ROOT = ROOT.parents[1]
    os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
print("repository root:", ROOT)
```

    repository root: /content/ai-theories



```python
import time

from src.data import (
    BPETokenizer,
    learn_bpe,
    load_code_corpus,
    load_japanese_corpus,
    load_tiny_shakespeare,
    pretokenize,
    train_unigram_model,
)
from src.utils import (
    compute_character_coverage,
    compute_chunk_length_statistics,
    compute_exact_match_rate,
    compute_fertility,
    compute_segmentation_agreement_rate,
    compute_unknown_rate,
    plot_dual_axis_curves,
    plot_grouped_bar,
)
```

### 5.1 可逆性の検証

事前分割(`pretokenize`)の`chunk_split_mode="whitespace"`は、空白の直前で分割し空白を次のチャンクの先頭に含める(3.2 節)。この結果、`BPETokenizer.encode()`の出力トークンを連結する(`byte_level=True`の場合は`BPETokenizer.decode()`で UTF-8 バイト列からデコードし直す)と、元のテキストが過不足なく復元できるはずである。`chunk_split_mode="none"`は改行のみ保持しないため、改行を含まないテキストでは同様に復元できるが、複数行にわたるテキストでは復元できない(`pretokenize`の docstring を参照)。この性質を、`byte_level`の真偽 × `chunk_split_mode`の 2 通り、計 4 通りの組み合わせで確認する。`byte_level=True`の検証をより強くするため、UTF-8 で 1 バイトの英語のみのサンプルに加え、3 バイトの日本語・4 バイトの絵文字を含むサンプルでも同様に確認する。


```python
# 可逆性の検証: BPETokenizer.encode() の出力を連結(byte_level=True の場合は decode())
# すると、入力テキストが復元できるか。改行を含まない文字列を使う
# (chunk_split_mode="none" は改行を保持しないため、改行を含む文字列では
# そもそも復元できないという 5.1 節で述べた既知の制約を切り分けるため)。
# byte_level=True の検証を強めるため、UTF-8 で 1 バイトの英語のみに加え、
# 3 バイトの日本語・4 バイトの絵文字を含むサンプルでも確認する。
roundtrip_samples = {
    "英語のみ": "  The quick, brown fox—jumps over the lazy dog!   Multiple   spaces here.  ",
    "日本語を含む": "  これは  日本語を含む   テスト文です。UTF-8で3バイトの文字を含む。  ",
    "絵文字を含む": "  Multiple emoji here: 😀🎉🚀 and more spaces after.  ",
}

for label, sample in roundtrip_samples.items():
    print(f"=== {label} ===")
    for byte_level in [False, True]:
        for mode in ["whitespace", "none"]:
            tokenizer = learn_bpe(
                sample, vocab_size=300, byte_level=byte_level, chunk_split_mode=mode
            )
            tokens = tokenizer.encode(sample)
            reconstructed = tokenizer.decode(tokens)
            match = reconstructed == sample
            print(f"  byte_level={byte_level!s:<5} chunk_split_mode={mode:<10} 復元一致={match}")
            assert match, (
                f"復元に失敗した: サンプル={label!r}, byte_level={byte_level}, mode={mode}"
            )
```

    === 英語のみ ===
      byte_level=False chunk_split_mode=whitespace 復元一致=True
      byte_level=False chunk_split_mode=none       復元一致=True
      byte_level=True  chunk_split_mode=whitespace 復元一致=True
      byte_level=True  chunk_split_mode=none       復元一致=True
    === 日本語を含む ===
      byte_level=False chunk_split_mode=whitespace 復元一致=True
      byte_level=False chunk_split_mode=none       復元一致=True
      byte_level=True  chunk_split_mode=whitespace 復元一致=True
      byte_level=True  chunk_split_mode=none       復元一致=True
    === 絵文字を含む ===
      byte_level=False chunk_split_mode=whitespace 復元一致=True
      byte_level=False chunk_split_mode=none       復元一致=True
      byte_level=True  chunk_split_mode=whitespace 復元一致=True
      byte_level=True  chunk_split_mode=none       復元一致=True


### 5.2 コーパスの取得

英語(`load_tiny_shakespeare`、004 と共通)・日本語(`load_japanese_corpus`、日本語版 Wikipedia の記事本文)・コード(`load_code_corpus`、本リポジトリ自身の`src/`配下の Python ソースコード)の 3 ドメインを取得する。

**日本語コーパスの出典とライセンス**: 日本語ドメインは、フリー百科事典『ウィキペディア(Wikipedia)』日本語版(https://ja.wikipedia.org/)の記事本文を、クリエイティブ・コモンズ 表示-継承 4.0 国際(CC BY-SA 4.0)ライセンスの下で使用する。記事本文は編集され続けるため、記事タイトルだけでは実行のたびに異なる文字列になりうる。そこで`load_japanese_corpus`は、記事タイトルと **リビジョン ID(特定時点の版を指す ID)を固定した組**(`src/data/text.py`の`_JAPANESE_WIKIPEDIA_REVISIONS`)から Wikimedia API(`action=parse`、`oldid`でリビジョンを指定)を用いて取得することで、実行時点によらず同一の入力を得る(6 節・実験の再現性についての注記も参照)。取得は Wikimedia API への複数回のリクエストを伴うため、初回の実行に数分かかることがある(2 回目以降はキャッシュから読むため一瞬で終わる)。


```python
CACHE_DIR = ROOT / ".cache" / "tokenizer"

t0 = time.time()
corpus_en = load_tiny_shakespeare(CACHE_DIR)
corpus_ja = load_japanese_corpus(CACHE_DIR)
corpus_code = load_code_corpus(ROOT)
print(f"取得時間: {time.time() - t0:.1f} 秒")

for name, corpus in [("English", corpus_en), ("Japanese", corpus_ja), ("Code", corpus_code)]:
    print(f"{name}: {len(corpus):,} 文字")
    print(repr(corpus[:80]))
    print()
```

    取得時間: 251.3 秒
    English: 1,115,394 文字
    'First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.'
    
    Japanese: 654,854 文字
    '日本国（にほんこく、にっぽん-）、または日本（にほん、にっぽん）は、東アジアに位置する立憲君主制国家。面積は 377,975.64 km2（平方キロメートル）、'
    
    Code: 102,057 文字
    '"""`theories/`・`apps/` の双方から再利用する共通モジュール。"""\n\n"""theories/・apps/ から再利用するデータ処理ユーテ'
    



```python
# 3 ドメインを同一サイズの学習用・ホールドアウト用に切り出す(ドメイン間の公平な比較のため)。
# コードコーパスが最も小さいため、その範囲に収まるサイズに揃える。
TRAIN_SIZE = 60_000
HOLDOUT_SIZE = 15_000

corpora = {"English": corpus_en, "Japanese": corpus_ja, "Code": corpus_code}
train_texts = {name: text[:TRAIN_SIZE] for name, text in corpora.items()}
holdout_texts = {
    name: text[TRAIN_SIZE : TRAIN_SIZE + HOLDOUT_SIZE] for name, text in corpora.items()
}

for name, text in corpora.items():
    assert len(text) >= TRAIN_SIZE + HOLDOUT_SIZE, f"{name} コーパスが小さすぎる: {len(text)} 文字"
print("学習・ホールドアウトの切り出しに成功")
```

    学習・ホールドアウトの切り出しに成功




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/02_pretraining/005_tokenizer.ipynb
