---
title: "小型 GPT の事前学習(Pretraining a Small GPT)(実装・実験編 1/4)"
---

この記事は後編(実装・実験編 1/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/006_pretraining_small_gpt-theory)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/006_pretraining_small_gpt-practice-2)。

## 4. 実装方針 / Implementation Policy

001〜005 でスクラッチ実装した部品(`MultiHeadAttention`・`DecoderBlock`・`RoPE`・`RMSNorm`・`SwiGLU`・トークナイザ)を **そのまま再利用** し、006 で新規に追加するのは「これらを 1 つの学習ループに統合する部分」のみである(段階 1 で `src/` に追加済み)。

| 機能 | 配置場所 | 実装/委譲 |
|---|---|---|
| decoder-only 言語モデル(`GPTLanguageModel`、トークン埋め込み・重み共有・`generate()`) | `src/models/gpt.py` | スクラッチ実装(新規、段階 1) |
| `DecoderBlock` の `use_cross_attention` 引数(交差注意を持たない decoder-only 構成) | `src/layers/transformer_block.py` | スクラッチ実装(段階 1、002・003・004 との後方互換性を検証済み) |
| 学習ループ(`train_language_model`、Adam・固定学習率・fp32)・評価(`evaluate_bits_per_byte`、非重複窓での逐次評価) | `src/training/trainer.py` | スクラッチ実装(新規、段階 1) |
| Wikipedia コーパス取得の一般化(`load_wikipedia_corpus`)、文字列段階での訓練・検証分割(`split_train_val_text`)、トークン化(`encode_corpus`)、評価窓の作成(`make_evaluation_windows`) | `src/data/text.py` | スクラッチ実装(段階 1) |
| bits-per-byte・perplexity・非埋め込みパラメータ数(`compute_bits_per_byte`・`compute_perplexity`・`count_non_embedding_parameters`) | `src/utils/statistics.py` | スクラッチ実装(段階 1) |
| ノイズ床の帯を表示するグループ棒グラフ(`plot_grouped_bar` の `noise_band` 引数) | `src/utils/visualization.py` | スクラッチ実装(段階 1) |
| BPE・Unigram トークナイザを整数 ID 方式のインターフェース(`encode(text) -> list[int]`・`decode(ids) -> str`)に適合させるラッパー | 本ノートブック(5.3 節) | ノートブック内に直接記述(下記の理由による) |

**ID 方式ラッパーを `src/` ではなく本ノートブックに置く理由**: `encode_corpus`(`src/data/text.py`)は `CharacterLevelTokenizer` と同じ「`encode(text) -> list[int]` で整数 ID を直接返す」インターフェースを想定している。しかし 005 の `BPETokenizer`・`UnigramTokenizer` の `encode()` は部分語シンボルの **文字列**(`list[str]`)を返し、シンボル → 整数 ID の対応を持たない(005 の関心が「どう分割するか」であり「ID をどう割り当てるか」ではなかったため)。この対応付けは実装上ごく薄い作業(語彙をソートして通し番号を振るだけ)であり、`src/` に切り出すほどの再利用価値のある抽象化を要さないため、本ノートブックに直接書く(`encode_corpus` の docstring で「整数 ID を割り当てる語彙ラッパーの追加は 006 のノートブック段階で扱う」と予告した通り)。なお `UnigramTokenizer` は内部で保持する `sentencepiece.SentencePieceProcessor`(`processor` 属性)が `encode(text) -> list[int]` / `decode(ids) -> str` を既に提供しているため、これはラップせずそのまま使う。

**実行環境についての注記**: Claude Code(本ノートブックの作成環境)は Google Colab を操作できない。5〜6 節のコードは Google Colab の T4 GPU での実行を前提に設計しているが、**本ノートブックの現在のセル出力はローカル(Apple Silicon、MPS または CPU)での縮小スケール(スモークテスト)実行によるもの** である。縮小の内容(コーパスサイズ・ステップ数など)は 5.1 節で具体的な値とともに明記し、Colab T4 での本番実行後に数値を差し替える必要がある箇所を 7 節末尾にまとめる。

## 5. 実装 / Implementation


```python
# 環境セットアップ(Google Colab)
import sys

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    !git clone https://github.com/kojikojiprg/ai-theories.git
    %cd ai-theories
    !pip install uv -q
    !uv pip install --system -r requirements.txt
# ローカル(Jupyter)実行時は、リポジトリルートで起動していればそのまま動く。
```

    Cloning into 'ai-theories'...
    remote: Enumerating objects: 349, done.[K
    remote: Counting objects: 100% (349/349), done.[K
    remote: Compressing objects: 100% (219/219), done.[K
    remote: Total 349 (delta 171), reused 274 (delta 112), pack-reused 0 (from 0)[K
    Receiving objects: 100% (349/349), 4.13 MiB | 27.83 MiB/s, done.
    Resolving deltas: 100% (171/171), done.
    /content/ai-theories
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m23.7/23.7 MB[0m [31m75.0 MB/s[0m eta [36m0:00:00[0m
    [?25h[2mUsing Python 3.12.13 environment at: /usr[0m
    [2K[2mResolved [1m41 packages[0m [2min 374ms[0m[0m
    [2K[2mPrepared [1m27 packages[0m [2min 51.13s[0m[0m
    [2mUninstalled [1m12 packages[0m [2min 1.82s[0m[0m
    [2K[2mInstalled [1m27 packages[0m [2min 648ms[0m[0m
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
import functools
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from src.data.text import (
    CharacterLevelTokenizer,
    encode_corpus,
    load_wikipedia_corpus,
    make_evaluation_windows,
    split_train_val_text,
)
from src.data.tokenizer import BPETokenizer, UnigramTokenizer, learn_bpe, train_unigram_model
from src.layers.normalization import RMSNorm
from src.layers.feedforward import SwiGLUFeedForwardNetwork
from src.layers.positional_encoding import RotaryPositionEmbedding
from src.models.gpt import GPTLanguageModel
from src.training.trainer import evaluate_bits_per_byte, train_language_model
from src.utils.statistics import (
    compute_bits_per_byte,
    compute_exact_match_rate,
    compute_perplexity,
    compute_unknown_rate,
    count_non_embedding_parameters,
)
from src.utils.visualization import plot_grouped_bar, plot_learning_curves_multi_seed

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"torch: {torch.__version__} / device: {device}")

ROOT = Path(".")
CACHE_DIR = ROOT / ".cache" / "006_corpus"

```

    torch: 2.13.0+cu130 / device: cuda


### 5.1 スケールの設定(スモークテスト)

**Claude Code は Google Colab を操作できないため、本ノートブックのセル出力はローカル(Apple Silicon、MPS または CPU)で実行したスモークテストによるものである。** 以下の `SMOKE_TEST = True` の分岐が、コーパスサイズ・語彙サイズ・学習ステップ数のすべてを、ローカルで数分〜十数分のうちに完走できる規模まで縮小する。`SMOKE_TEST = False` 側に、5〜6 節の理論・実験設計で述べた **本番(Google Colab T4 GPU)** で使うべき値をコメントとして残す。

**縮小した理由の内訳**:

- **コーパスサイズ**: 本番は日英とも 20 MB 以上(段階 1 で取得済み、`load_wikipedia_corpus`)。スモークテストでは、この一部(先頭バイト列)のみを使う。
- **BPE の学習コーパス**: 自作の `learn_bpe`(`src/data/tokenizer.py`)は BPE のマージ候補の再カウントを毎回全チャンクに対して行う素朴な実装であり、コーパスサイズに対して計算量が急増する(実測: 日本語コーパス先頭 40,000 バイトで語彙サイズ 1024 まで学習するのに約 2 秒だが、同じ語彙サイズでもコーパスサイズを数倍にすると数分〜十数分に達した)。そのため BPE・Unigram の **語彙学習** に使うコーパスと、学習後のトークナイザで実際に **符号化(encode)** して言語モデルの学習に使うコーパスを分離し、前者だけを特に小さくする(符号化そのものは学習済みのマージ規則を適用するだけなので高速)。
- **ステップ数・model サイズ**: 「数十ステップ」のスモークテストとして、7 節で述べる 1 エポック基準のステップ数決定は行わず、固定の小さいステップ数で全条件を回す。

以下の Markdown セルでは、判定基準・実験設計そのものは **本番のスケールを前提に** 記述する(スモークテストでの縮小はあくまで「ローカルで動作を確認する」ためであり、判定基準を満たす・満たさないの評価は本番実行後に行うべきものである)。各実験セルの直後に、スモークテストでの実測結果を記載し、本番との差異を明示する。

**本番でも `TOKENIZER_TRAIN_BYTES` を制限する理由(本番実行前の修正 14・17)**: `learn_bpe` の計算量の急増(上記)は本番スケール(コーパス全体、約 23 MB)でも問題になる。訓練分割全体を渡すと、Google Colab の 1 セッションの時間内に収まらない可能性がある(6 節の計測セルで見積もる)。そこで本番でも `TOKENIZER_TRAIN_BYTES` を制限する。**言語モデルの学習データ量(`LM_CORPUS_BYTES`)は制限しない**(コーパス全体を使う)。トークナイザの語彙学習に使うテキスト量を制限しても、**全トークナイザ条件が同一のテキストで語彙を学習している限り、条件間比較の妥当性は損なわれない**(比較の公平性は「同じ入力から学習したか」であり、「コーパス全体を使ったか」ではない)。

当初は 5 MB(語彙サイズの上限 $V{=}8192$ に対して 1 マージあたり約 610 バイト)を想定していたが、スモークテストの実測(日本語 40,000 バイト・語彙 4096 で 3.81 秒、べき指数 $b{=}1.280$)から外挿すると、5 MB では日英合計で約 130 分かかると見積もられた(修正 16 の全語彙サイズ分の補正込み)。日本語のべき指数が 1 を大きく超えるため、データ量に対して超線形に伸びることが主な要因である。そこで `TOKENIZER_TRAIN_BYTES = 2{,}000{,}000`(2 MB、外挿で日英合計約 41 分)に下げた。

**チャンク長の上限による性能改善(本番実行前の修正 21・23)**: しかし、Google Colab T4 でスモークテスト設定を実際に実行したところ、上記の外挿より実測値はさらに悪く、日本語の BPE 学習は 40,000 バイトで 10.4 秒(べき指数 $b{=}1.366$、`TOKENIZER_TRAIN_BYTES=2{,}000{,}000`への外挿で約 136 分)、符号化は 380,000 バイトで 16.4 秒 × 6 条件(べき指数 $b{=}1.273$、外挿で約 5 時間)に達した(英語はいずれも 1 分未満)。3.6 節で述べた通り、これは 005 で確認した「空白による事前分割が日本語で機能しない」ことの計算量側の帰結である。そこで`pretokenize()`に`max_chunk_bytes`引数(修正 21)を追加し、事前分割の各チャンクを UTF-8 の文字境界を壊さない位置でさらに分割してチャンク長に上限を設けた。あわせて`BPETokenizer.encode()`にチャンク単位のメモ化(修正 22)を導入した(`max_chunk_bytes`の導入によりチャンクの反復出現が増えるため、メモ化の効果が大きくなる)。

**`max_chunk_bytes`の値の選定**: 32〜128 バイトの範囲でローカル(Apple Silicon)の大規模コーパス(200,000〜800,000 バイト)を用いて符号化時間のべき指数を比較したところ、128 バイトで $b{=}0.510$、64 バイトで $b{=}0.505$、32 バイトで $b{=}0.469$ であり、64 バイト以下でべき指数がほぼ横ばいになった(英語は同じ範囲で $b{\approx}0.17$、いずれも制限なしの日本語の $b{=}0.539$ を大きく下回る)。64 バイトは、日本語の典型的な複合語・助詞の連なりを大きく損なわない程度の長さを保ちつつ、これ以上小さくしても速度上の追加の恩恵が小さいため、`MAX_CHUNK_BYTES = 64`を採用する(コード セルで定義)。

**再計測の結果(スモークテスト設定、修正 21・22・23 適用後)**: 5.4 節末尾・5.5 節の計測セル(`CALIBRATION_SIZES_BYTES = [10{,}000, 20{,}000, 40{,}000]`)を`max_chunk_bytes=64`導入後に再実行すると、学習は日本語 $b{=}1.233$(修正前 $b{=}1.366$)・英語 $b{=}1.424$、符号化は絶対時間が計測分解能(0.01 秒)以下になり($10\text{,}000$〜$40\text{,}000$ バイトいずれも 0.00 秒)、べき指数の推定自体が不安定になった(日本語 $b{=}0.874$・英語 $b{=}0.829$、参考値)。**この計測点の範囲(10 KB〜40 KB)は、そもそも`max_chunk_bytes=64`による分割が効き始めるかどうかの境界に近く、改善効果を過小評価する**(1 行・1 段落が 64 バイトを超える頻度が、コーパスが小さいうちは相対的に低いため)。そこでローカル(Apple Silicon)で、より本番に近いスケール(200,000〜800,000 バイト、Wikipedia コーパス由来)を使って別途補足計測したところ、日本語の符号化のべき指数は制限なし($b{=}0.539$)から`max_chunk_bytes=64`で $b{=}0.505$ に、学習のべき指数は制限なし($b{=}0.978$)から $b{=}0.874$ に下がり、英語の符号化のべき指数($b{\approx}0.17$〜$0.26$)と同程度の桁に近づいた(32 バイトまで下げると $b{=}0.469$ とさらに下がるが、日本語の典型的な複合語・助詞の連なりを損なう度合いが大きくなるため採用しない)。この補足計測は「既存の計測セルをスモークテスト設定のまま実行する」という手順そのものではないが、計測セルの計測点(10 KB〜40 KB)だけでは改善効果を過小評価するため、より大きなスケールでの検証として付記する。

**`TOKENIZER_TRAIN_BYTES`の見直し**: べき指数が下がったことで、`TOKENIZER_TRAIN_BYTES`を 2 MB から引き上げても Google Colab の 1 セッション時間内に収まる余地が生まれた。上記の大きいスケールでの補足計測(`max_chunk_bytes=64`)の指数・係数から`TOKENIZER_TRAIN_BYTES=8{,}000{,}000`バイトへ外挿すると、学習は日本語 4 語彙サイズ分で約 6.3 分・英語で約 0.6 分(日英合計 約 6.9 分)、`LM_CORPUS_BYTES`(コーパス全体、約 24〜25 MB)への符号化は日英合計で 1 分未満と見積もられ、model の学習(Google Colab T4 での実測: 約 15 分)と合わせても総所要時間は約 22〜23 分となり、60 分の予算に対して約 2.7 倍の余裕がある。この余裕は、Google Colab T4 とローカル(Apple Silicon)のハードウェアの違いによる速度差(べき指数は計算量の構造を反映するため環境に依存しにくいが、係数 $a$ は依存しうる)や、コーパス取得・Unigram 学習など計測に含めていない処理の分の安全マージンとして確保する。そこで`TOKENIZER_TRAIN_BYTES = 8{,}000{,}000`(8 MB)に引き上げる。8 MB は語彙サイズの上限に対して 1 マージあたり約 977 バイトになり、修正 18 で確認した「19.5 バイト/マージまでは語彙使用率が安定し、9.8 バイト/マージで崩壊する」という下限(100 バイト/マージ)に対して十分な余裕がある(6 節の診断セルで、この学習データ量が語彙品質に与える影響を改めて確認する)。

**BPE 語彙サイズを等比数列にする理由(本番実行前の修正)**: 語彙サイズを増やしたときの
bits-per-byte の改善が収穫逓減する(語彙サイズの対数に対してほぼ線形に減衰する)という仮説
(実験 E)を検証するには、比較する語彙サイズの **対数軸上の間隔を揃える** 必要がある。
対数軸上で等間隔(等比数列、公比 $r$)であれば、各区間の「情報量あたりの改善」を公平に比較
できる。当初案の $1024 \to 4096 \to 8192$ は倍率が $\times 4$ と $\times 2$ で揃っておらず、
改善が対数に対して減衰するという仮説が正しければ、この不揃いな刻み **だけ** で前半の改善幅が
大きく出ることがほぼ自動的に保証されてしまい、判定基準が実質的に何も検証しない。そこで
$1024 \to 2048 \to 4096 \to 8192$(公比 $r=2$ の 4 点)に変更する。

**エポック上限を実験 C〜F と実験 G で分ける理由(本番実行前の修正 9)**: 実験 G は選定条件で
「より多いステップ数」で学習し、短時間学習(実験 C〜F、`NUM_STEPS`)より改善するかを見る
(修正 7)。実験 C〜F と同じ 1 エポック上限を使うと、実験 G のステップ数は必然的にそれを
超えてしまう。そこで **実験 C〜F は高々 1/2 エポック、実験 G は高々 1 エポック** という
異なる上限を設ける。この差(1/2 エポックぶん)が、実験 G で「ステップ数を増やすと改善する」
ことを 1 エポック以内で示すための余地になる。


```python
SMOKE_TEST = False  # Claude Code はこの True 側のみ実行する(Colab T4 では False に切り替える)

# エポック上限(実験 C〜F と実験 G で共通の定数、本番実行前の修正 9)。
EPOCH_CAP_CF = 0.5
EPOCH_CAP_G = 1.0

# 実験 C(シード間ノイズ床の測定)で使うシード数(本番実行前の修正 11)。
NOISE_FLOOR_SEEDS = (0, 1, 2, 3, 4)

# 事前分割チャンクの最大バイト数(3.6 節・5.1 節、本番実行前の修正 21・23)。
# スモークテスト・本番の両方で同一の値を使う(スケールではなく計算量対策のための
# 設計上の定数であるため、SMOKE_TEST では分岐させない)。学習・符号化の両方で
# 必ず同一の値を使う必要があるため、共通の定数として一箇所で定義する。
MAX_CHUNK_BYTES = 64

if SMOKE_TEST:
    # --- コーパス ---
    LM_CORPUS_BYTES = 400_000  # 本番: 制限なし(取得済みコーパス全体、日英とも 20 MB 以上)
    TOKENIZER_TRAIN_BYTES = 40_000  # 本番: LM_CORPUS_BYTES の訓練分割全体を使う
    VALIDATION_RATIO = 0.05

    # --- トークナイザ語彙サイズ(本番: 1024 / 2048 / 4096 / 8192、公比 2 の等比数列) ---
    # スモークテストも同じ倍率構造(公比 2、4 点)に揃える。
    BPE_VOCAB_SIZES = [512, 1024, 2048, 4096]
    # Unigram は BPE のうち 1 条件(本番は V=4096、リストの 3 番目)と厳密に一致させる。
    # これにより Unigram と BPE の比較が「分割方式のみの対照」になり、語彙サイズの
    # 違いと交絡しない(本番実行前の修正 4)。
    UNIGRAM_VOCAB_SIZE = BPE_VOCAB_SIZES[2]

    # --- model(実験 C・D・E・F・G のすべてで共通。修正 7: 実験 G も model サイズは
    # 変えず、ステップ数のみ増やす) ---
    D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF = 64, 2, 4, 256
    SEQUENCE_LENGTH = 64
    DROPOUT = 0.0

    # --- 学習 ---
    BATCH_SIZE = 8
    LEARNING_RATE = 3e-4
    # NUM_STEPS・G_NUM_STEPS・EVAL_INTERVAL・G_EVAL_INTERVAL は 5.6 節で、
    # 実測したトークン数と EPOCH_CAP_CF・EPOCH_CAP_G から計算式で決定する
    # (本番実行前の修正 10。ハードコードしない)。
else:
    # --- コーパス ---
    LM_CORPUS_BYTES = 10**12  # 実質無制限(取得済みコーパス全体、日英とも 20 MB 以上)を使う
    # BPE・Unigram の語彙学習だけをコーパス全体(約 23〜25 MB)ではなく 8 MB に制限する
    # (本番実行前の修正 14・17)。言語モデルの学習データ量(LM_CORPUS_BYTES)は制限しない。
    # 3.6 節・5.1 節の MAX_CHUNK_BYTES 導入(修正 21・23)により学習・符号化の計算量が
    # コーパスサイズにほぼ比例するようになったため、修正 21 以前の 2 MB から 8 MB に
    # 引き上げた(5.1 節の再計測・外挿を参照)。
    TOKENIZER_TRAIN_BYTES = 8_000_000
    VALIDATION_RATIO = 0.05

    # --- トークナイザ語彙サイズ(公比 2 の等比数列、本番実行前の修正 3) ---
    BPE_VOCAB_SIZES = [1024, 2048, 4096, 8192]
    # Unigram は BPE のうち 1 条件(V=4096、リストの 3 番目)と厳密に一致させる(修正 4)。
    UNIGRAM_VOCAB_SIZE = BPE_VOCAB_SIZES[2]

    # --- model(実験 C・D・E・F・G のすべてで共通、修正 7・13) ---
    D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF = 256, 4, 8, 1024
    SEQUENCE_LENGTH = 256
    DROPOUT = 0.0

    # --- 学習 ---
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    # NUM_STEPS・G_NUM_STEPS・EVAL_INTERVAL・G_EVAL_INTERVAL は 5.6 節で計算式により決定する。

SWIGLU_D_FF = round((2 / 3) * D_FF)  # 004: 標準の順伝播ネットワークとパラメータ数を揃えるための丸め
print(f"SMOKE_TEST={SMOKE_TEST}")
print(f"D_MODEL={D_MODEL}, NUM_LAYERS={NUM_LAYERS}, NUM_HEADS={NUM_HEADS}, D_FF={D_FF}(SwiGLU: {SWIGLU_D_FF})")
print(f"SEQUENCE_LENGTH={SEQUENCE_LENGTH}, BATCH_SIZE={BATCH_SIZE}")
print(f"BPE_VOCAB_SIZES={BPE_VOCAB_SIZES}, UNIGRAM_VOCAB_SIZE={UNIGRAM_VOCAB_SIZE}")
print(f"EPOCH_CAP_CF={EPOCH_CAP_CF}, EPOCH_CAP_G={EPOCH_CAP_G}")
```

    SMOKE_TEST=False
    D_MODEL=256, NUM_LAYERS=4, NUM_HEADS=8, D_FF=1024(SwiGLU: 683)
    SEQUENCE_LENGTH=256, BATCH_SIZE=32
    BPE_VOCAB_SIZES=[1024, 2048, 4096, 8192], UNIGRAM_VOCAB_SIZE=4096
    EPOCH_CAP_CF=0.5, EPOCH_CAP_G=1.0


### model サイズについて: 計算最適ではないことの明記(本番実行前の修正 13)

本番の model サイズ($d_{\text{model}}{=}256$、4 層、8 ヘッド、非埋め込みパラメータ数は 6 節で実測して報告する)は、Kaplan et al. [6] や Chinchilla(Hoffmann et al., "Training Compute-Optimal Large Language Models", 2022)が示す計算最適(compute-optimal)の目安(非埋め込みパラメータ 1 個あたり約 20 トークン)には遠く及ばない。本番のコーパス規模・エポック上限(1/2 エポック、修正 9)から逆算すると、消化トークン数は非埋め込みパラメータ数 1 個あたり **約 2 トークン程度** にとどまる見込みである(6 節で実測値を報告する)。

**これは意図的な選択である。** 006 が主張するのは条件 **間** の比較(トークナイザ方式・語彙サイズによる bits-per-byte の違い)であり、その公平性は 3.3 節・6 節で述べた通り非埋め込みパラメータ数を全条件で揃えることによって担保されている。全条件が等しく学習不足(under-trained)であっても、条件間の順位や差の解釈そのものは損なわれない。model サイズ・データ量・計算量の最適な配分そのものは 009 の主題であり、006 では扱わない。

### 実行時間の較正(本番実行前の修正 13)

本番実行の前に、1 ステップあたりの学習時間を見積もる。コーパス・トークナイザの準備を待たずに測定できるよう、**ダミーのランダムトークン列** を使う(1 ステップの時間は model の形状(`D_MODEL`・`SEQUENCE_LENGTH`・`BATCH_SIZE` など)にのみ依存し、データの中身には依存しないため)。

**最初の 20 ステップ(warmup)を捨て、その後 30 ステップの中央値** で 1 ステップあたりの時間を測る。CUDA のカーネルコンパイルは初回のみ発生し、warmup を含めて測ると 1 ステップあたりの時間を大幅に過大評価する(004 での実測に基づく)。総ランのステップ数が決まる 5.6 節で、この較正結果を使って総実行時間を見積もる。


```python
calibration_vocab_size = max(BPE_VOCAB_SIZES)  # 最大語彙サイズで保守的に(やや多めに)見積もる
torch.manual_seed(0)
calibration_d_k = D_MODEL // NUM_HEADS
calibration_rope = RotaryPositionEmbedding(calibration_d_k, max_position=SEQUENCE_LENGTH)
calibration_model = GPTLanguageModel(
    vocabulary_size=calibration_vocab_size,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    max_sequence_length=SEQUENCE_LENGTH,
    positional_transform=calibration_rope,
    normalization_factory=RMSNorm,
    feed_forward_factory=functools.partial(SwiGLUFeedForwardNetwork, D_MODEL, SWIGLU_D_FF),
    tie_embeddings=True,
    dropout=DROPOUT,
).to(device)
calibration_optimizer = torch.optim.Adam(calibration_model.parameters(), lr=LEARNING_RATE)

NUM_CALIBRATION_STEPS = 50
NUM_CALIBRATION_WARMUP = 20
step_times = []
for _ in range(NUM_CALIBRATION_STEPS):
    x = torch.randint(0, calibration_vocab_size, (BATCH_SIZE, SEQUENCE_LENGTH), device=device)
    y = torch.randint(0, calibration_vocab_size, (BATCH_SIZE, SEQUENCE_LENGTH), device=device)
    t0 = time.time()
    logits = calibration_model(x)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, calibration_vocab_size), y.reshape(-1))
    calibration_optimizer.zero_grad()
    loss.backward()
    calibration_optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    step_times.append(time.time() - t0)

median_step_time = float(np.median(step_times[NUM_CALIBRATION_WARMUP:]))
print(f"1 ステップあたりの時間(warmup {NUM_CALIBRATION_WARMUP} 回を除いた中央値): {median_step_time * 1000:.1f} ms")
print("(総実行時間の見積もりは、総ステップ数が決まる 5.6 節で行う。見積もりが 2 時間を")
print(" 大きく超える場合は、BATCH_SIZE・D_MODEL を下げる方向で調整すること。)")

del calibration_model, calibration_optimizer  # 較正用の model は以降使わないため破棄する
```

    1 ステップあたりの時間(warmup 20 回を除いた中央値): 135.5 ms
    (総実行時間の見積もりは、総ステップ数が決まる 5.6 節で行う。見積もりが 2 時間を
     大きく超える場合は、BATCH_SIZE・D_MODEL を下げる方向で調整すること。)


### 5.2 コーパスの取得

日本語版・英語版 Wikipedia のコーパス(段階 1 で `load_wikipedia_corpus` により取得済み、日本語 24.58 MB・英語 24.33 MB、いずれも 20 MB 以上の目標を達成している)を読み込み、`LM_CORPUS_BYTES` バイトに切り詰める。バイト数を基準に切り詰めることで、日英で同じ「物理的な量」のコーパスを使う(3.5 節で述べた通り、文字数で揃えると日本語の方が少ない情報量になってしまう)。


```python
def load_corpus_prefix(language: str, max_bytes: int) -> str:
    """言語版 Wikipedia コーパスを取得し、UTF-8 で ``max_bytes`` バイト以内に
    切り詰める(マルチバイト文字の途中で切れないよう末尾を調整する)。"""
    text = load_wikipedia_corpus(language, CACHE_DIR)
    raw = text.encode("utf-8")[:max_bytes]
    while True:
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            raw = raw[:-1]


corpus_ja = load_corpus_prefix("ja", LM_CORPUS_BYTES)
corpus_en = load_corpus_prefix("en", LM_CORPUS_BYTES)
full_corpus = {"ja": corpus_ja, "en": corpus_en}
print(f"corpus_ja: {len(corpus_ja):,} 文字 / {len(corpus_ja.encode('utf-8')):,} バイト")
print(f"corpus_en: {len(corpus_en):,} 文字 / {len(corpus_en.encode('utf-8')):,} バイト")
```

    corpus_ja: 8,955,329 文字 / 24,575,245 バイト
    corpus_en: 24,214,546 文字 / 24,331,593 バイト



```python
# 文字列の段階で訓練用・検証用に分割する(全トークナイザ条件が同一の検証テキストを使うため、
# トークン化の前に分割することが必須。src/data/text.py の split_train_val_text の docstring を参照)。
train_text = {}
val_text = {}
for lang, corpus in [("ja", corpus_ja), ("en", corpus_en)]:
    train_text[lang], val_text[lang] = split_train_val_text(corpus, VALIDATION_RATIO)
    print(
        f"{lang}: train {len(train_text[lang].encode('utf-8')):,} バイト / "
        f"val {len(val_text[lang].encode('utf-8')):,} バイト"
    )
```

    ja: train 23,309,728 バイト / val 1,265,517 バイト
    en: train 23,117,476 バイト / val 1,214,117 バイト


### 5.3 トークナイザを整数 ID 方式に適合させるラッパー

4 節で述べた通り、`BPETokenizer.encode()` は部分語シンボルの文字列を返す。`encode_corpus`・`make_evaluation_windows`・`GPTLanguageModel` が要求する「整数 ID を返す `encode()` / ID から文字列に戻す `decode()`」のインターフェースに適合させるため、学習済み語彙をソートして通し番号を振る薄いラッパーを定義する。


```python
class BPEIDTokenizer:
    """BPETokenizer(部分語シンボル文字列を返す)を整数 ID 方式に適合させるラッパー。

    学習済み語彙(``vocab``、シンボル文字列の集合)をソートした順序で ID を
    割り当てる(学習コーパス・vocab_size が同じであれば毎回同一の対応になる)。
    """

    def __init__(self, bpe_tokenizer: BPETokenizer) -> None:
        self.bpe_tokenizer = bpe_tokenizer
        symbols = sorted(bpe_tokenizer.vocab)
        self.symbol_to_id = {symbol: i for i, symbol in enumerate(symbols)}
        self.id_to_symbol = symbols
        self.vocab_size = len(symbols)

    def encode(self, text: str) -> list[int]:
        return [self.symbol_to_id[symbol] for symbol in self.bpe_tokenizer.encode(text)]

    def decode(self, ids) -> str:
        if isinstance(ids, Tensor):
            ids = ids.tolist()
        return self.bpe_tokenizer.decode([self.id_to_symbol[i] for i in ids])


# UnigramTokenizer は内部の sentencepiece.SentencePieceProcessor(processor 属性)が
# encode(text) -> list[int] / decode(ids) -> str を既に提供しているため、ラップせずそのまま使う。
def get_vocab_size(tokenizer) -> int:
    """CharacterLevelTokenizer・BPEIDTokenizer(vocab_size 属性)と
    sentencepiece.SentencePieceProcessor(vocab_size はメソッドであり
    整数ではないため get_piece_size() を使う必要がある)の両方に対応する。"""
    v = getattr(tokenizer, "vocab_size", None)
    if isinstance(v, int):
        return v
    return tokenizer.get_piece_size()


print("BPEIDTokenizer 定義済み")
```

    BPEIDTokenizer 定義済み


### 5.4 トークナイザ条件の学習

各言語について、5 条件(文字レベル・バイトレベル BPE × 4 語彙サイズ・Unigram 言語モデル)のトークナイザを学習する。**BPE・Unigram の語彙学習は訓練テキストの先頭 `TOKENIZER_TRAIN_BYTES` バイトのみを使う**(5.1 節で述べた通り、自作 BPE の学習コストの都合による制限。本番実行前の修正 14 により、この制限はスモークテストに限らず本番でも行う。ただし言語モデルの学習データ量(`LM_CORPUS_BYTES`)は制限しない)。文字レベルの語彙は **コーパス全体(訓練 + 検証)** から構築する(文字レベルは語彙構築が軽量なため縮小の必要がなく、`CharacterLevelTokenizer` は未知文字への対応を持たないため、検証テキストのみに出現する文字で符号化が失敗しないよう訓練テキストだけでなくコーパス全体を使う。004 と同じ理由)。

**Unigram の `byte_fallback=True`・`character_coverage=1.0` について(本番実行前の修正 1)**: sentencepiece の既定(`character_coverage=0.9995`、`byte_fallback=False`)では、訓練コーパスの低頻度文字(改行を含む制御文字や、`character_coverage` の外側に落ちる希少文字)が語彙に入らず、符号化時にすべて単一の未知語トークン `<unk>` へ潰れてしまう。これは Unigram 条件だけが他の条件(バイトレベル BPE は 256 バイト初期語彙を持つため未知語が原理的に発生しない、文字レベルはコーパス全体から語彙を構築するため未知語が発生しない)と異なる問題を解くことを意味し、失われた文字の分だけ負の対数尤度が不当に下がってしまう。`byte_fallback=True`(語彙にない文字を UTF-8 バイト列へ退避させる)と `character_coverage=1.0`(全文字を被覆対象にする)を組み合わせることで、この情報損失を解消する(`train_unigram_model`、`src/data/tokenizer.py`。005 は `byte_fallback`・`character_coverage` を指定せず、変更後も既定値のままのため影響を受けない。005 を再実行し、結果が変わらないことを確認済み)。


```python
def build_tokenizers(lang: str) -> dict[str, object]:
    """言語 lang の 5 条件のトークナイザを学習して辞書で返す。"""
    train = train_text[lang]
    tokenizer_train_sample = train.encode("utf-8")[:TOKENIZER_TRAIN_BYTES]
    while True:
        try:
            tokenizer_train_sample = tokenizer_train_sample.decode("utf-8")
            break
        except UnicodeDecodeError:
            tokenizer_train_sample = tokenizer_train_sample[:-1]

    tokenizers = {}
    # 検証テキストのみに出現する文字で KeyError にならないよう、コーパス全体
    # (訓練 + 検証)から文字レベル語彙を構築する(CharacterLevelTokenizer は
    # 未知文字への対応を持たないため、004 と同じ理由でこの構築方法を取る)。
    tokenizers["character"] = CharacterLevelTokenizer(full_corpus[lang])

    for vocab_size in BPE_VOCAB_SIZES:
        bpe = learn_bpe(
            tokenizer_train_sample,
            vocab_size=vocab_size,
            byte_level=True,
            chunk_split_mode="whitespace",
            max_chunk_bytes=MAX_CHUNK_BYTES,
        )
        tokenizers[f"bpe_v{vocab_size}"] = BPEIDTokenizer(bpe)

    model_prefix = CACHE_DIR / "spm" / f"unigram_{lang}"
    unigram = train_unigram_model(
        tokenizer_train_sample,
        vocab_size=UNIGRAM_VOCAB_SIZE,
        model_prefix=model_prefix,
        byte_fallback=True,
        character_coverage=1.0,
    )
    tokenizers["unigram"] = unigram.processor

    return tokenizers
```

#### BPE 学習時間のスケーリング計測(本番実行前の修正 15)

`TOKENIZER_TRAIN_BYTES`(修正 14)での BPE 学習が Google Colab の 1 セッション時間内に収まるかを、実際に `build_tokenizers` を呼ぶ前に見積もる。訓練テキストの先頭の異なるデータ量(本番: 100 KB・200 KB・400 KB)のそれぞれで、`learn_bpe`(語彙サイズは `BPE_VOCAB_SIZES` の最大値)の実行時間を日英それぞれ実測する(日本語は空白による事前分割が効かないため、英語より遅くなる可能性がある、005 の議論)。

データ量 $n$(バイト)に対する実行時間 $t$ がべき乗則 $t = a n^{b}$ に従うと仮定し、3 点の対数 $\log t = \log a + b \log n$ を最小二乗法で当てはめて指数 $b$ を推定する。この $b$ を使って `TOKENIZER_TRAIN_BYTES` での所要時間を外挿し、日英合計が 30 分を超える場合は警告を表示する(処理は継続し、例外は送出しない)。

**全語彙サイズ分の補正(本番実行前の修正 16)**: この計測は `vocab_size=max(BPE_VOCAB_SIZES)` の 1 本だけを測っているが、`build_tokenizers()` は言語ごとに `BPE_VOCAB_SIZES` の **4 つすべて** を学習する。マージ回数(バイトレベル初期語彙 256 からのマージ数)の合計は、最大語彙 1 本だけのマージ回数の $\text{merge\_ratio} = \sum(\text{BPE\_VOCAB\_SIZES}) / \max(\text{BPE\_VOCAB\_SIZES})$ 倍になる。実行時間がマージ回数(≈ 語彙サイズ)にほぼ比例するという仮定のもとで、外挿値に `merge_ratio` を乗じて 4 語彙サイズ分の所要時間に補正する。


```python
def measure_bpe_scaling(lang: str, sizes_bytes: list[int]) -> tuple[list[int], list[float]]:
    """複数のデータ量で learn_bpe の実行時間を実測し、(データ量, 実行時間) を返す。"""
    full_bytes = train_text[lang].encode("utf-8")
    times = []
    for n_bytes in sizes_bytes:
        sample_bytes = full_bytes[:n_bytes]
        while True:
            try:
                sample = sample_bytes.decode("utf-8")
                break
            except UnicodeDecodeError:
                sample_bytes = sample_bytes[:-1]
        t0 = time.time()
        learn_bpe(
            sample,
            vocab_size=max(BPE_VOCAB_SIZES),
            byte_level=True,
            chunk_split_mode="whitespace",
            max_chunk_bytes=MAX_CHUNK_BYTES,
        )
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  {lang} {n_bytes:,} バイト: {elapsed:.2f}s")
    return sizes_bytes, times


# スモークテストでは訓練データ自体が小さい(TOKENIZER_TRAIN_BYTES=40,000)ため、
# 本番の 100 KB/200 KB/400 KB に対応する比率で計測点を縮小する。
CALIBRATION_SIZES_BYTES = [10_000, 20_000, 40_000] if SMOKE_TEST else [100_000, 200_000, 400_000]

# build_tokenizers() は言語ごとに BPE_VOCAB_SIZES の 4 つすべてを学習するが、この計測は
# max(BPE_VOCAB_SIZES) の 1 本のみを測っている。実行時間がマージ回数(≈ 語彙サイズ)に
# ほぼ比例するという仮定のもとで、4 語彙サイズ分の合計マージ回数の比を掛けて補正する
# (本番実行前の修正 16)。
merge_ratio = sum(BPE_VOCAB_SIZES) / max(BPE_VOCAB_SIZES)
print(f"merge_ratio = sum(BPE_VOCAB_SIZES) / max(BPE_VOCAB_SIZES) = {merge_ratio:.3f}\n")

scaling_results = {}
total_estimated_seconds_uncorrected = 0.0
total_estimated_seconds = 0.0
for lang in ("ja", "en"):
    print(f"{lang}:")
    sizes, times = measure_bpe_scaling(lang, CALIBRATION_SIZES_BYTES)
    # log(t) = log(a) + b * log(n) の最小二乗法によるあてはめ(1 次多項式のあてはめ)。
    log_n = np.log(sizes)
    log_t = np.log(times)
    exponent, log_a = np.polyfit(log_n, log_t, deg=1)
    coefficient = np.exp(log_a)
    extrapolated_seconds_uncorrected = float(coefficient * (TOKENIZER_TRAIN_BYTES**exponent))
    extrapolated_seconds = extrapolated_seconds_uncorrected * merge_ratio
    scaling_results[lang] = {"exponent": float(exponent), "extrapolated_seconds": extrapolated_seconds}
    total_estimated_seconds_uncorrected += extrapolated_seconds_uncorrected
    total_estimated_seconds += extrapolated_seconds
    print(
        f"  推定べき指数 b={exponent:.3f}  "
        f"TOKENIZER_TRAIN_BYTES={TOKENIZER_TRAIN_BYTES:,} バイトへの外挿"
        f"(語彙 1 本、補正前): {extrapolated_seconds_uncorrected:.1f}s  "
        f"(4 語彙サイズ分、補正後): {extrapolated_seconds:.1f}s\n"
    )

print(
    f"日英合計の外挿時間(語彙 1 本、補正前): {total_estimated_seconds_uncorrected:.1f}s "
    f"({total_estimated_seconds_uncorrected / 60:.1f} 分)"
)
print(
    f"日英合計の外挿時間(4 語彙サイズ分、補正後): {total_estimated_seconds:.1f}s "
    f"({total_estimated_seconds / 60:.1f} 分)"
)
if total_estimated_seconds > 30 * 60:
    print(
        "\n[警告] 外挿した BPE 学習時間の合計が 30 分を超えています。"
        " TOKENIZER_TRAIN_BYTES を下げることを検討してください。"
    )
```

    merge_ratio = sum(BPE_VOCAB_SIZES) / max(BPE_VOCAB_SIZES) = 1.875
    
    ja:
      ja 100,000 バイト: 21.20s
      ja 200,000 バイト: 41.62s
      ja 400,000 バイト: 73.12s
      推定べき指数 b=0.893  TOKENIZER_TRAIN_BYTES=8,000,000 バイトへの外挿(語彙 1 本、補正前): 1081.6s  (4 語彙サイズ分、補正後): 2027.9s
    
    en:
      en 100,000 バイト: 11.14s
      en 200,000 バイト: 15.67s
      en 400,000 バイト: 21.94s
      推定べき指数 b=0.489  TOKENIZER_TRAIN_BYTES=8,000,000 バイトへの外挿(語彙 1 本、補正前): 95.0s  (4 語彙サイズ分、補正後): 178.2s
    
    日英合計の外挿時間(語彙 1 本、補正前): 1176.6s (19.6 分)
    日英合計の外挿時間(4 語彙サイズ分、補正後): 2206.1s (36.8 分)
    
    [警告] 外挿した BPE 学習時間の合計が 30 分を超えています。 TOKENIZER_TRAIN_BYTES を下げることを検討してください。



```python
t0 = time.time()
tokenizers = {"ja": build_tokenizers("ja"), "en": build_tokenizers("en")}
print(f"トークナイザ学習完了: {time.time() - t0:.1f}s")
for lang in ("ja", "en"):
    for name, tok in tokenizers[lang].items():
        print(f"  {lang}/{name}: vocab_size={get_vocab_size(tok)}")
```

    トークナイザ学習完了: 2919.4s
      ja/character: vocab_size=4654
      ja/bpe_v1024: vocab_size=1024
      ja/bpe_v2048: vocab_size=2048
      ja/bpe_v4096: vocab_size=4096
      ja/bpe_v8192: vocab_size=8192
      ja/unigram: vocab_size=4096
      en/character: vocab_size=2394
      en/bpe_v1024: vocab_size=1024
      en/bpe_v2048: vocab_size=2048
      en/bpe_v4096: vocab_size=4096
      en/bpe_v8192: vocab_size=8192
      en/unigram: vocab_size=4096


#### 語彙品質の診断量(本番実行前の修正 18)

**これは判定基準ではなく、実験 E(語彙サイズ増加の収穫逓減)の結果を解釈するための補助情報(診断量)である**(`CLAUDE.md`: 診断量の追加は判定基準の変更にあたらない)。

トークナイザの学習データ量(`TOKENIZER_TRAIN_BYTES`)は、計算量の都合(5.1 節)により言語モデルの学習データ量(`LM_CORPUS_BYTES`、コーパス全体)より小さい。学習データを絞ると、**語彙サイズが大きい条件ほど不利になりうる**。例えば `TOKENIZER_TRAIN_BYTES=2{,}000{,}000` のとき、そこから 8192 回マージすると末尾のマージは低頻度のペアに基づくものになり、学習後の語彙のうち実際に使われる部分が少なくなる可能性がある。これは実験 E が測ろうとしている「語彙サイズ増加に伴う収穫逓減」と **同じ向き** に働くため、観測される収穫逓減の一部が、真の収穫逓減ではなく **トークナイザ学習データの不足で説明できてしまう可能性がある**。

以下の 3 つを BPE の 4 条件それぞれについて報告する。

1. **語彙使用率**: 訓練テキスト全体(`LM_CORPUS_BYTES` 規模)を符号化したときに、実際に 1 回以上出現した語彙の割合。語彙サイズを増やしても使われない語彙が増えるだけなら、この値が下がる。
2. **低頻度語彙の割合**: 訓練テキスト全体の符号化で出現回数が 10 回未満だった語彙(一度も出現しない語彙を含む)の割合。
3. **1 マージあたりの学習バイト数**: `TOKENIZER_TRAIN_BYTES / vocab_size`。

**語彙使用率が大きい語彙サイズで顕著に下がっている場合、実験 E の「収穫逓減」の一部はトークナイザ学習データの不足で説明されうる** ことに注意して 7.3 節の考察を読むこと。


```python
def compute_vocab_quality(lang: str, vocab_size: int) -> dict:
    """BPE 条件の語彙品質を診断する(訓練テキスト全体での符号化に基づく)。"""
    bpe_id_tokenizer = tokenizers[lang][f"bpe_v{vocab_size}"]
    symbols = bpe_id_tokenizer.bpe_tokenizer.encode(train_text[lang])
    counts = Counter(symbols)
    used_vocab = sum(1 for s in bpe_id_tokenizer.id_to_symbol if counts.get(s, 0) >= 1)
    low_freq_vocab = sum(1 for s in bpe_id_tokenizer.id_to_symbol if counts.get(s, 0) < 10)
    return {
        "usage_rate": used_vocab / vocab_size,
        "low_freq_rate": low_freq_vocab / vocab_size,
        "bytes_per_merge": TOKENIZER_TRAIN_BYTES / vocab_size,
    }


print(f"{'lang':4s} {'vocab_size':>10s} {'usage_rate':>11s} {'low_freq_rate':>14s} {'bytes/merge':>12s}")
for lang in ("ja", "en"):
    for vocab_size in BPE_VOCAB_SIZES:
        q = compute_vocab_quality(lang, vocab_size)
        print(
            f"{lang:4s} {vocab_size:10d} {q['usage_rate']:11.4f} "
            f"{q['low_freq_rate']:14.4f} {q['bytes_per_merge']:12.1f}"
        )
```

    lang vocab_size  usage_rate  low_freq_rate  bytes/merge
    ja         1024      0.9336         0.0820       7812.5
    ja         2048      0.9668         0.0420       3906.2
    ja         4096      0.9827         0.0225       1953.1
    ja         8192      0.9902         0.0175        976.6
    en         1024      0.9395         0.0713       7812.5
    en         2048      0.9663         0.0439       3906.2
    en         4096      0.9736         0.0396       1953.1
    en         8192      0.9822         0.0334        976.6


#### ラウンドトリップの完全一致の検証(本番実行前の修正 1)

各条件について、訓練テキスト・検証テキストの両方で `decode(encode(text)) == text` を確認する。一致しない条件があればその場で例外を送出する(未知語による情報損失が実際に解消されているかどうかの直接的な検証)。


```python
def check_roundtrip(tokenizer, text: str) -> bool:
    """decode(encode(text)) が text と完全に一致するかを確認する。"""
    ids = encode_corpus(tokenizer, text).tolist()
    return tokenizer.decode(ids) == text


tokenizer_conditions = ["character"] + [f"bpe_v{v}" for v in BPE_VOCAB_SIZES] + ["unigram"]

for lang in ("ja", "en"):
    for tokenizer_name in tokenizer_conditions:
        tokenizer = tokenizers[lang][tokenizer_name]
        for split_name, text in (("train", train_text[lang]), ("val", val_text[lang])):
            if not check_roundtrip(tokenizer, text):
                raise AssertionError(
                    f"ラウンドトリップ不一致: {lang}/{tokenizer_name}/{split_name}"
                )
        print(f"[OK] {lang}/{tokenizer_name}: train・val ともラウンドトリップ完全一致")

print("\n全条件でラウンドトリップの完全一致を確認した")
```

    [OK] ja/character: train・val ともラウンドトリップ完全一致
    [OK] ja/bpe_v1024: train・val ともラウンドトリップ完全一致
    [OK] ja/bpe_v2048: train・val ともラウンドトリップ完全一致
    [OK] ja/bpe_v4096: train・val ともラウンドトリップ完全一致
    [OK] ja/bpe_v8192: train・val ともラウンドトリップ完全一致
    [OK] ja/unigram: train・val ともラウンドトリップ完全一致
    [OK] en/character: train・val ともラウンドトリップ完全一致
    [OK] en/bpe_v1024: train・val ともラウンドトリップ完全一致
    [OK] en/bpe_v2048: train・val ともラウンドトリップ完全一致
    [OK] en/bpe_v4096: train・val ともラウンドトリップ完全一致
    [OK] en/bpe_v8192: train・val ともラウンドトリップ完全一致
    [OK] en/unigram: train・val ともラウンドトリップ完全一致
    
    全条件でラウンドトリップの完全一致を確認した


#### 未知トークン率の報告(本番実行前の修正 1)

バイトレベル BPE・文字レベル(コーパス全体から語彙を構築)は未知語の仕組みを持たないため未知トークン率は構造的に常に 0 である。Unigram は `byte_fallback=True` により未知語が発生しないことを実測で確認する(`compute_unknown_rate`、`src/utils/statistics.py`)。


```python
def compute_condition_unknown_rate(lang: str, tokenizer_name: str) -> float:
    tokenizer = tokenizers[lang][tokenizer_name]
    if tokenizer_name == "unigram":
        pieces = tokenizer.encode(val_text[lang], out_type=str)
        return compute_unknown_rate(pieces, unk_token="<unk>")
    # バイトレベル BPE(256 バイトの初期語彙を持つため未知語が原理的に発生しない)・
    # 文字レベル(コーパス全体から語彙を構築するため未知語が発生しない)は
    # 未知語の仕組み自体を持たないため、常に 0 として報告する。
    return 0.0


all_zero_unknown_rate = True
for lang in ("ja", "en"):
    for tokenizer_name in tokenizer_conditions:
        rate = compute_condition_unknown_rate(lang, tokenizer_name)
        all_zero_unknown_rate = all_zero_unknown_rate and (rate == 0.0)
        print(f"  {lang}/{tokenizer_name}: unknown_rate={rate:.6f}")

print(f"\n全条件で未知トークン率が 0 か: {all_zero_unknown_rate}")
```

      ja/character: unknown_rate=0.000000
      ja/bpe_v1024: unknown_rate=0.000000
      ja/bpe_v2048: unknown_rate=0.000000
      ja/bpe_v4096: unknown_rate=0.000000
      ja/bpe_v8192: unknown_rate=0.000000
      ja/unigram: unknown_rate=0.000000
      en/character: unknown_rate=0.000000
      en/bpe_v1024: unknown_rate=0.000000
      en/bpe_v2048: unknown_rate=0.000000
      en/bpe_v4096: unknown_rate=0.000000
      en/bpe_v8192: unknown_rate=0.000000
      en/unigram: unknown_rate=0.000000
    
    全条件で未知トークン率が 0 か: True


### 5.5 符号化時間のスケーリング計測(本番実行前の修正 20)

6 節冒頭の計測セルは `learn_bpe()` の **学習** 時間のみを測っており、学習後のトークナイザで実際にコーパス全体を **符号化(encode)** する時間を含んでいない。スクラッチ実装の byte-level BPE の符号化(`BPETokenizer.encode_chunk`)は、空白による事前分割が効かずチャンクが長くなる日本語のような場合、学習と同等以上に時間がかかる可能性があり、これを見積もりから除外すると本番の総実行時間を過小評価してしまう。

学習済みの $V{=}\max(\text{BPE\_VOCAB\_SIZES})$ のトークナイザを使い、`CALIBRATION_SIZES_BYTES`(実行時間の較正の節と同じ、本番: 100 KB・200 KB・400 KB)のテキストを符号化する時間を実測し、学習時間と同様にべき乗則をあてはめて **`LM_CORPUS_BYTES` の訓練分割全体**(トークナイザの学習データ量 `TOKENIZER_TRAIN_BYTES` ではなく、言語モデルの学習データ量であることに注意)への符号化時間を外挿する。この外挿値に、キャッシュ導入後(6 節、修正 19)の実際の符号化回数(言語ごとに 6 条件 = 文字レベル 1・BPE 4・Unigram 1)を掛けて、総符号化時間を見積もる。**語彙サイズによる符号化コストの差は無視できるものとして扱う**(BPE の符号化は学習済みのマージ規則を順に適用するだけの処理であり、コストは主にコーパスサイズに支配され、語彙サイズ(マージ規則の本数)への依存は小さいと仮定する)。

学習時間の外挿・符号化時間の外挿・両者の合計を分けて表示し、**合計が 30 分を超える場合に警告を出す**(学習時間のみを見ていた従来の警告から変更)。


```python
def measure_encode_scaling(lang: str, sizes_bytes: list[int]) -> tuple[list[int], list[float]]:
    """複数のデータ量で BPE の符号化時間を実測し、(データ量, 実行時間) を返す。"""
    full_bytes = train_text[lang].encode("utf-8")
    tokenizer = tokenizers[lang][f"bpe_v{max(BPE_VOCAB_SIZES)}"]
    times = []
    for n_bytes in sizes_bytes:
        sample_bytes = full_bytes[:n_bytes]
        while True:
            try:
                sample = sample_bytes.decode("utf-8")
                break
            except UnicodeDecodeError:
                sample_bytes = sample_bytes[:-1]
        t0 = time.time()
        tokenizer.encode(sample)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  {lang} {n_bytes:,} バイト: {elapsed:.2f}s")
    return sizes_bytes, times


# 修正 19 のキャッシュ導入後、条件 1 つあたり符号化は 1 回のみ行われる
# (文字レベル 1・BPE 4・Unigram 1 = 言語ごとに 6 回)。
NUM_ENCODE_CALLS_PER_LANG = 1 + len(BPE_VOCAB_SIZES) + 1

encode_scaling_results = {}
total_encode_time_estimate = 0.0
for lang in ("ja", "en"):
    print(f"{lang}:")
    sizes, times = measure_encode_scaling(lang, CALIBRATION_SIZES_BYTES)
    log_n = np.log(sizes)
    log_t = np.log(times)
    exponent, log_a = np.polyfit(log_n, log_t, deg=1)
    coefficient = np.exp(log_a)
    train_split_bytes = len(train_text[lang].encode("utf-8"))
    extrapolated_per_call = float(coefficient * (train_split_bytes**exponent))
    extrapolated_total = extrapolated_per_call * NUM_ENCODE_CALLS_PER_LANG
    encode_scaling_results[lang] = {"exponent": float(exponent), "extrapolated_total_seconds": extrapolated_total}
    total_encode_time_estimate += extrapolated_total
    print(
        f"  推定べき指数 b={exponent:.3f}  1 回(train_split={train_split_bytes:,} バイト)あたりの外挿: "
        f"{extrapolated_per_call:.1f}s  x{NUM_ENCODE_CALLS_PER_LANG} 回 = {extrapolated_total:.1f}s\n"
    )

total_train_time_estimate = sum(r["extrapolated_seconds"] for r in scaling_results.values())
total_time_estimate = total_train_time_estimate + total_encode_time_estimate
print(
    f"学習時間の外挿合計(4 語彙サイズ分、修正 16): {total_train_time_estimate:.1f}s "
    f"({total_train_time_estimate / 60:.1f} 分)"
)
print(f"符号化時間の外挿合計: {total_encode_time_estimate:.1f}s ({total_encode_time_estimate / 60:.1f} 分)")
print(f"学習 + 符号化の合計: {total_time_estimate:.1f}s ({total_time_estimate / 60:.1f} 分)")
if total_time_estimate > 30 * 60:
    print(
        "\n[警告] 学習時間と符号化時間の合計が 30 分を超えています。"
        " TOKENIZER_TRAIN_BYTES を下げることを検討してください。"
    )
```

    ja:
      ja 100,000 バイト: 0.01s
      ja 200,000 バイト: 0.02s
      ja 400,000 バイト: 0.04s
      推定べき指数 b=0.803  1 回(train_split=23,309,728 バイト)あたりの外挿: 1.2s  x6 回 = 6.9s
    
    en:
      en 100,000 バイト: 0.02s
      en 200,000 バイト: 0.03s
      en 400,000 バイト: 0.07s
      推定べき指数 b=1.095  1 回(train_split=23,117,476 バイト)あたりの外挿: 5.8s  x6 回 = 34.9s
    
    学習時間の外挿合計(4 語彙サイズ分、修正 16): 2206.1s (36.8 分)
    符号化時間の外挿合計: 41.8s (0.7 分)
    学習 + 符号化の合計: 2247.9s (37.5 分)
    
    [警告] 学習時間と符号化時間の合計が 30 分を超えています。 TOKENIZER_TRAIN_BYTES を下げることを検討してください。




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/02_pretraining/006_pretraining_small_gpt.ipynb
