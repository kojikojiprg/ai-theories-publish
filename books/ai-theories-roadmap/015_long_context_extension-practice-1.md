---
title: "長文脈拡張 / Long Context Extension(実装・実験編 1/4)"
---

この記事は後編(実装・実験編 1/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/015_long_context_extension-theory)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/015_long_context_extension-practice-2)。

## 4. 実装方針 / Implementation Policy

**`src/`に切り出す(スクラッチ実装)**: `src/layers/positional_encoding.py`(003 で作成した RoPE を拡張)

- `compute_rope_inverse_frequencies(d_k, base)`: $\theta_i = b^{-2i/d}$。既存の`RotaryPositionEmbedding`が内部で使っていた
  式を関数として切り出したもので、同じ式・同じ dtype で計算する。
- `RotaryFrequencyScaling`(抽象クラス)と、そのサブクラス`PositionInterpolation`・`NTKAwareScaling`・`NTKByPartsScaling`・
  `YaRNScaling`・`DynamicScaling`。いずれも`frequencies_and_factor(theta, d, b, l)`で **変換後の角周波数と cos・sin に
  掛ける倍率** を返す。位置補間も「角周波数を $1/s$ 倍にする」形で表す(3.3 節)。`DynamicScaling`は静的な手法を作る
  関数を受け取り、$s = \max(1, l/L)$ が 1 のときは変換を一切せず元の角周波数と倍率 1 を返す。このとき`is_identity(l)`が
  真になり、`RotaryPositionEmbedding`はスケーリングなしと同じ経路(事前計算した cos・sin のキャッシュ)を通るので、
  出力はスケーリングなしと bit 単位で一致する(cos・sin を計算するデバイスの違いによる丸めの差も生じない)。
- 補助関数`compute_ntk_aware_base()`($b'$)、`compute_yarn_ramp()`($\gamma$)、`compute_yarn_attention_factor()`
  ($\sqrt{1/t}$)。
- `RotaryPositionEmbedding`に`frequency_scaling`引数(既定値`None`)を追加した。`None`のときは 003〜014 と同じ経路
  (事前計算した cos・sin のキャッシュ)をそのまま通る。指定したときは、`apply`のたびに現在の系列長
  $l = \max(\text{positions}) + 1$ から角周波数と倍率を求め、要求された位置の cos・sin を直接計算する。
  `rotation_frequencies(l)`で、実際に使う角周波数と倍率を取り出せる(5.6 節の照合と診断量に使う)。
- **既定の挙動は変えない。** 変更前のコミットを`git worktree`で取得し、同じ環境のサブプロセスで実行した結果と、
  RoPE の出力・モデルの logits・勾配が bit 単位で一致することを 5.5 節で確かめる。

**`src/data/text.py`に追加(記事の境界)**: Hub のコーパスのアーティファクトは記事を改行 1 つで連結した 1 つのテキストであり、
記事の中にも改行があるので、テキストだけからは記事の境界を復元できない(5.6 節で確かめる)。実験 C のために評価窓を 1 つの
記事の中に収め、6.1 節のブートストラップで記事を単位にするため、境界を次の関数で得る。

- `load_wikipedia_articles()`: マニフェストの指定した番号の記事を 1 記事ずつ取得する(`load_wikipedia_corpus()`と同じ
  記事単位のキャッシュを使う)。
- `compute_wikipedia_article_offsets()`・`validate_wikipedia_article_offsets()`: 全記事を改行 1 つで連結した結果がコーパスと
  文字単位で一致することを確かめながら記事ごとの文字位置を求める / その整合性(記事数・単調増加・重なりなし・区切り)を確かめる。
- `locate_wikipedia_article_spans()`: コーパスのアーティファクトの`metadata.json`に記事ごとの文字位置(`article_offsets`)が
  あればそれを使い、なければ Wikipedia API から記事を取得してコーパスと照合する。どちらを使ったかを返す。
- `article_offsets`は`scripts/promote_canonical_corpora.py --article-offsets`が計算して`metadata.json`に書き込む
  (`corpus.txt`は変えない)。既存の`load_wikipedia_corpus_with_fallback()`は変更しておらず、`metadata.json`にキーが
  増えても返り値は変わらない。

**`src/utils/statistics.py`に追加**: `paired_cluster_bootstrap_ratio_of_sums()`(単位のまとまり(クラスタ)を復元抽出する
対応付きブートストラップ。013 の`paired_bootstrap_ratio_of_sums()`の単位を、記事などのクラスタに置き換えたもの)。

**ノートブック内に直接書く(015 固有)**:

- 手法の名前から周波数の変換を作る関数と、学習済みのモデルの全層の RoPE を差し替える関数(5.9 節)。
- 評価のハーネス: 位置ごとの損失(nats)の計算、位置の範囲ごとの bits-per-byte、文脈を直前の $L$ トークンに
  切り詰めた評価(実験 C)、Attention のエントロピーの記録(5.9 節・6.4 節)。
- 微調整のハーネス(5.10 節)。学習そのものは`src/training/trainer.py`の`train_language_model()`、optimizer は
  `src/training/optimizer.py`の`AdamW`、学習率スケジュールは`src/training/schedule.py`の warmup + cosine を使う
  ([007](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/007_training_stabilization-theory)・[012](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/012_low_rank_adaptation-theory) と同じ部品)。
- 判定の標準偏差は、記事を単位とする対応付きのクラスタブートストラップ(`paired_cluster_bootstrap_ratio_of_sums()`)で
  求める。評価窓を単位とする`paired_bootstrap_ratio_of_sums()`([013](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/013_quantization_basics-theory) で追加)は、旧基準の
  診断量にのみ使う(6.1 節)。

**Attention の実装**: 001 の標準の Attention(`MultiHeadAttention`)をそのまま使う。**KV キャッシュは使わない**
(3.7 節の dynamic scaling の問題を避けるため)。評価のたびに系列の全体を順伝播する。

**アップロード方針**: 微調整したモデルは条件比較のためのものであり、後続トピックの入力にも読者が単体で取得する対象にも
ならないので、保存もアップロードもしない。

## 5. 実装 / Implementation

### 5.1 環境セットアップ(Google Colab)と実行環境の記録

依存関係のインストールの後に、実行環境(GPU 名・compute capability・GPU の総メモリ・torch と CUDA・cuDNN の
バージョン・デバイス・コミット・実行日時)を印字する。結果・考察で実行環境に言及するときは、この印字を出典とする。


```python
# 環境セットアップ(Google Colab)
import sys
import time

NOTEBOOK_START_TIME = time.time()  # 6.2 節で、ここまでの実行時間を印字するために使う

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    !git clone https://github.com/kojikojiprg/ai-theories.git
    %cd ai-theories
    !pip install uv -q
    !uv pip install --system -r requirements.txt
# ローカル(Jupyter)実行時は、リポジトリルートで起動していればそのまま動く。

import torch  # noqa: E402

from src.utils.environment import print_execution_environment  # noqa: E402

device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
execution_environment = print_execution_environment(device)
```

    Cloning into 'ai-theories'...
    remote: Enumerating objects: 1104, done.[K
    remote: Counting objects: 100% (144/144), done.[K
    remote: Compressing objects: 100% (100/100), done.[K
    remote: Total 1104 (delta 80), reused 87 (delta 44), pack-reused 960 (from 1)[K
    Receiving objects: 100% (1104/1104), 12.29 MiB | 21.16 MiB/s, done.
    Resolving deltas: 100% (641/641), done.
    /content/ai-theories
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m20.5/20.5 MB[0m [31m30.5 MB/s[0m eta [36m0:00:00[0m
    [?25h[2mUsing Python 3.13.15 environment at: /usr[0m
    [2K[2mResolved [1m52 packages[0m [2min 374ms[0m[0m
    [2K[2mPrepared [1m31 packages[0m [2min 42.52s[0m[0m
    [2mUninstalled [1m17 packages[0m [2min 733ms[0m[0m
    [2K[2mInstalled [1m31 packages[0m [2min 315ms[0m[0m
     [31m-[39m [1mclick[0m[2m==8.5.0[0m
     [32m+[39m [1mclick[0m[2m==8.4.2[0m
     [31m-[39m [1mcuda-bindings[0m[2m==12.9.7[0m
     [32m+[39m [1mcuda-bindings[0m[2m==13.3.1[0m
     [31m-[39m [1mcuda-pathfinder[0m[2m==1.8.0[0m
     [32m+[39m [1mcuda-pathfinder[0m[2m==1.6.0[0m
     [31m-[39m [1mcuda-toolkit[0m[2m==12.8.1[0m
     [32m+[39m [1mcuda-toolkit[0m[2m==13.0.3.0[0m
     [31m-[39m [1mfilelock[0m[2m==3.32.5[0m
     [32m+[39m [1mfilelock[0m[2m==3.32.2[0m
     [31m-[39m [1mfonttools[0m[2m==4.64.0[0m
     [32m+[39m [1mfonttools[0m[2m==4.63.0[0m
     [31m-[39m [1mfsspec[0m[2m==2025.12.0[0m
     [32m+[39m [1mfsspec[0m[2m==2026.7.0[0m
     [31m-[39m [1mhuggingface-hub[0m[2m==1.29.0[0m
     [32m+[39m [1mhuggingface-hub[0m[2m==1.28.0[0m
     [31m-[39m [1mkiwisolver[0m[2m==1.5.1[0m
     [32m+[39m [1mkiwisolver[0m[2m==1.5.0[0m
     [31m-[39m [1mmatplotlib[0m[2m==3.10.0[0m
     [32m+[39m [1mmatplotlib[0m[2m==3.11.1[0m
     [31m-[39m [1mnumpy[0m[2m==2.1.3[0m
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
     [31m-[39m [1mnvidia-nccl-cu13[0m[2m==2.31.2[0m
     [32m+[39m [1mnvidia-nccl-cu13[0m[2m==2.29.7[0m
     [32m+[39m [1mnvidia-nvjitlink[0m[2m==13.3.33[0m
     [32m+[39m [1mnvidia-nvshmem-cu13[0m[2m==3.4.5[0m
     [32m+[39m [1mnvidia-nvtx[0m[2m==13.0.85[0m
     [31m-[39m [1mpillow[0m[2m==11.3.0[0m
     [32m+[39m [1mpillow[0m[2m==12.3.0[0m
     [31m-[39m [1msetuptools[0m[2m==80.10.2[0m
     [32m+[39m [1msetuptools[0m[2m==84.0.0[0m
     [31m-[39m [1mtorch[0m[2m==2.11.0+cu128[0m
     [32m+[39m [1mtorch[0m[2m==2.13.0[0m
     [31m-[39m [1mtqdm[0m[2m==4.67.3[0m
     [32m+[39m [1mtqdm[0m[2m==4.70.0[0m
     [31m-[39m [1mtriton[0m[2m==3.6.0[0m
     [32m+[39m [1mtriton[0m[2m==3.7.1[0m
    実行環境 / Execution environment
      Python                                 : 3.13.15
      OS / platform                          : Linux-6.6.122+-x86_64-with-glibc2.39
      torch                                  : 2.13.0+cu130
      torch のビルド時の CUDA                : 13.0
      cuDNN                                  : 92000
      デバイス / device                      : cuda
      GPU 名 / GPU name                      : Tesla T4
      compute capability                     : 7.5
      GPU の総メモリ (GiB)                   : 14.56
      コミット / git commit                  : e48131a7903a8b601deaa05c21a008c9a2a193ae
      未コミットの変更 / uncommitted changes : なし
      実行日時 (UTC)                         : 2026-09-25T03:35:06+00:00



```python
import copy
import functools
import hashlib
import inspect
import json
import math
import re
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from src.data.text import (
    encode_text_to_memmap,
    load_wikipedia_corpus_with_fallback,
    locate_wikipedia_article_spans,
    make_evaluation_windows,
    split_train_val_text,
)
from src.data.tokenizer import load_bpe_id_tokenizer_from_hub, load_bpe_id_tokenizer_json
from src.layers.attention import create_causal_mask
from src.layers.feedforward import SwiGLUFeedForwardNetwork
from src.layers.normalization import RMSNorm
from src.layers.positional_encoding import (
    DynamicScaling,
    NTKAwareScaling,
    NTKByPartsScaling,
    PositionInterpolation,
    RotaryPositionEmbedding,
    YaRNScaling,
    compute_ntk_aware_base,
    compute_yarn_attention_factor,
)
from src.models.gpt import GPTLanguageModel
from src.training.optimizer import AdamW
from src.training.schedule import compute_warmup_cosine_learning_rate
from src.training.trainer import evaluate_bits_per_byte, train_language_model
from src.utils.reporting import dumps_compact_json
from src.utils.statistics import (
    fit_power_law_exponent,
    paired_bootstrap_ratio_of_sums,
    paired_cluster_bootstrap_ratio_of_sums,
)

LOG2 = math.log(2.0)
ROOT = Path(".")
WIKIPEDIA_CACHE_DIR = ROOT / ".cache" / "wikipedia_en"  # 外部取得したコーパス・記事(データ源で命名)
ENCODED_CACHE_DIR = ROOT / ".cache" / "015_encoded"  # 015 の条件で符号化したトークン列
MODEL_REPO_ID = "kojikojiprg/ai-theories-small-gpt-en"
MODEL_REVISION = "main"
TOKENIZER_REPO_ID = "kojikojiprg/ai-theories-tokenizer-en"
CORPUS_REPO_ID = "kojikojiprg/ai-theories-corpus-en-pretraining"
MANIFEST_PATH = ROOT / "src" / "data" / "wikipedia_manifests" / "en_006_pretraining.json"
# scripts/promote_canonical_corpora.py --article-offsets の dry-run が書き出す metadata.json(ローカルでの照合用)
ARTICLE_OFFSETS_DRY_RUN_PATH = (
    ROOT / ".cache" / "promote_canonical_corpora" / "en_006" / "metadata_with_article_offsets.json"
)
VALIDATION_RATIO = 0.05  # 008 と同じ分割(コーパスの末尾 5% が検証)
MODEL_CARD_BITS_PER_BYTE = 1.668067  # モデルカードに記載した値(アップロードした重みを評価した値)


def sync_device() -> None:
    # 時間計測の直前・直後に、非同期に実行される GPU の処理の完了を待つ。
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def timed_call(fn) -> float:
    sync_device()
    t0 = time.time()
    fn()
    sync_device()
    return time.time() - t0


precondition_status: dict[str, bool] = {}  # 前提条件の成否(6.1 節で宣言、各実験の節で記録)
print(f"device: {device}")
```

    device: cuda


### 5.2 ベースモデル・トークナイザの取得(008 がアップロードしたもの)

ベースモデルは`kojikojiprg/ai-theories-small-gpt-en`の`main`ブランチ、トークナイザは`kojikojiprg/ai-theories-tokenizer-en`
から取得する。学習時の文脈長 $L$・ヘッドの次元 $d$・層数・ヘッド数は、取得した`config.json`から読む。

**底 $b$ は`config.json`に記録されていない。** 008 は`RotaryPositionEmbedding`を底を指定せずに構築しており、その既定値
(10000)で学習した。ここでは既定値を`inspect`で読み取って $b$ とし、008 の構築と同じ呼び出し(底を指定しない)で
モデルを組み立てる。底が異なれば 6.3 節の前提条件 P0(モデルカードの値の再現)が成立しないので、P0 が $b$ の確認を兼ねる。


```python
tokenizer, _tokenizer_from_hub = load_bpe_id_tokenizer_from_hub(TOKENIZER_REPO_ID)
assert _tokenizer_from_hub, "トークナイザを Hugging Face Hub から取得できなかった"
TOKENIZER_JSON_PATH = hf_hub_download(TOKENIZER_REPO_ID, "tokenizer.json")

_config_path = hf_hub_download(MODEL_REPO_ID, "config.json", revision=MODEL_REVISION)
MODEL_STATE_PATH = hf_hub_download(MODEL_REPO_ID, "model_state.pt", revision=MODEL_REVISION)
MODEL_CONFIG = json.loads(Path(_config_path).read_text(encoding="utf-8"))
print("MODEL_CONFIG:", json.dumps(MODEL_CONFIG, indent=2))
assert MODEL_CONFIG["positional_encoding"] == "rope"
assert MODEL_CONFIG["normalization"] == "rmsnorm" and MODEL_CONFIG["feed_forward"] == "swiglu"
assert MODEL_CONFIG["norm_first"] and MODEL_CONFIG["dropout"] == 0.0
assert tokenizer.vocab_size == MODEL_CONFIG["vocabulary_size"]

L = MODEL_CONFIG["sequence_length"]  # 学習時の文脈長 L
D_MODEL = MODEL_CONFIG["d_model"]
NUM_LAYERS = MODEL_CONFIG["num_layers"]
NUM_HEADS = MODEL_CONFIG["num_heads"]
VOCAB_SIZE = MODEL_CONFIG["vocabulary_size"]
HEAD_DIM = D_MODEL // NUM_HEADS  # d
ROPE_BASE = (
    inspect.signature(RotaryPositionEmbedding).parameters["base"].default
)  # b(008 が使った既定値)


def build_base_model() -> GPTLanguageModel:
    # 008 と同じ構成(底を指定しない RotaryPositionEmbedding)で組み立て、Hub の重みを読み込む。
    model = GPTLanguageModel(
        vocabulary_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=MODEL_CONFIG["d_ff"],
        max_sequence_length=L,
        positional_transform=RotaryPositionEmbedding(HEAD_DIM, max_position=L),
        normalization_factory=RMSNorm,
        feed_forward_factory=functools.partial(
            SwiGLUFeedForwardNetwork, D_MODEL, MODEL_CONFIG["swiglu_d_ff"]
        ),
        tie_embeddings=MODEL_CONFIG["tie_embeddings"],
    )
    _result = model.load_state_dict(torch.load(MODEL_STATE_PATH, map_location="cpu"))
    assert not _result.missing_keys and not _result.unexpected_keys
    return model.to(device).eval()


base_model = build_base_model()
TOTAL_PARAMETERS = sum(p.numel() for p in base_model.parameters())
print(
    f"L = {L}, d = {HEAD_DIM}, b = {ROPE_BASE}, 層数 = {NUM_LAYERS}, ヘッド数 = {NUM_HEADS}, "
    f"d_model = {D_MODEL}, 語彙サイズ = {VOCAB_SIZE}, パラメータ数 = {TOTAL_PARAMETERS:,}"
)
```


    tokenizer.json:   0%|          | 0.00/661k [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/305 [00:00<?, ?B/s]



    model_state.pt: reconstructing file:   0%|          |  0.00B / 21.0MB            



    model_state.pt: downloading bytes:           |  0.00B            


    MODEL_CONFIG: {
      "vocabulary_size": 8192,
      "d_model": 256,
      "num_layers": 4,
      "num_heads": 8,
      "d_ff": 1024,
      "swiglu_d_ff": 683,
      "sequence_length": 256,
      "dropout": 0.0,
      "tie_embeddings": true,
      "positional_encoding": "rope",
      "normalization": "rmsnorm",
      "feed_forward": "swiglu",
      "norm_first": true
    }
    L = 256, d = 32, b = 10000.0, 層数 = 4, ヘッド数 = 8, d_model = 256, 語彙サイズ = 8192, パラメータ数 = 5,246,208


### 5.3 スケールの設定(`SMOKE_TEST`の配線)とメモリの見積もり

水準の定義をこの 1 箇所(`LEVELS`)に集約する。`SMOKE_TEST = True`はローカルでのコード経路の確認用、
`SMOKE_TEST = False`は Google Colab T4 での本番実行用である。モデルは事前学習済みのものを使うので縮小しない。

**縮小の規則**:

- 評価長の水準 $\{2L, 4L, 8L\}$(公比 2 の等比)と、判定する水準($4L$)・微調整の長さ($4L$)は縮小しない。
- 評価窓の数・微調整のステップ数・シード数・較正用の窓の数・ブートストラップの反復回数・エントロピーを測る窓の数を
  縮小する。判定に使う条件のシード数は本番 5・スモークテスト 3、診断用の条件(底の調整 + 微調整、微調整した
  位置補間の切り詰めた評価)のシード数は本番・スモークテストとも 2 とし、「診断用 < 判定用」の順序関係を保つ
  (標本標準偏差には 2 以上が要る)。
- スモークテストの評価窓は 3 本の記事から 2 個ずつ選ぶ(5.6 節)。記事を単位とするブートストラップ(6.1 節)の経路を
  実行するため、2 本以上の記事から選ばれ、同じ記事に複数の窓があることを確かめる。
- 学習率のグリッド(公比 $\sqrt{10}$ の 3 水準)は縮小しない。ステップ数によらない構造(warmup の割合・最小学習率の比・
  gradient clipping の閾値の決め方)も共通にする。
- 判定に使う 2 条件と診断用の条件の関係(微調整の 3 手法、微調整なしの 5 手法)は縮小しない。

**バッチサイズとメモリ**: [014](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/014_flash_attention-theory) の閉形式(標準の実装の順伝播で $S$ と $P$ が同時に存在する、
$2 \cdot BHN^2 \cdot 4$ バイト)で、評価長 $8L$ の Attention のメモリを見積もる。これに logits($B N V \cdot 4$ バイト、$V$ は
語彙サイズ)を加えた量が、T4 の総メモリ(約 15 GiB)の 60% 以下に収まるバッチサイズを宣言する(60% は 014 と同じく、
アロケータのキャッシュ・断片化と作業領域の余裕)。微調整($4L$、逆伝播あり)は、各層で softmax の出力 $P$ を保存し
(層数 × $BHN^2 \cdot 4$ バイト)、順伝播の途中で 1 層分の $S$・マスク後のスコア・$P$ が同時に存在し($3 BHN^2 \cdot 4$ バイト)、
logits・損失の中間値・その勾配が $3 BNV \cdot 4$ バイト程度あるとして見積もる。


```python
SMOKE_TEST = False  # Claude Code はこの True 側のみ実行する(Colab T4 では False に切り替える)
_smoke_tag = "[動作確認のみ、結論ではない] " if SMOKE_TEST else ""
_plot_tag = (
    "[smoke test] " if SMOKE_TEST else ""
)  # 図のタイトル用(フォントに日本語がない環境がある)

# --- 全水準で共通の定数(本番実行前に宣言し、SMOKE_TEST で変えない) ---
EVAL_LENGTH_FACTORS = (2, 4, 8)  # 評価長 = 係数 x L(公比 2 の等比)。s = 係数
MAIN_FACTOR = 4  # 判定する水準(実験 A・D)
FINETUNE_FACTOR = 4  # 微調整の長さ = 4L、s = 4(実験 B・C)
MAX_FACTOR = max(EVAL_LENGTH_FACTORS)
EVAL_METHODS = ("none", "position_interpolation", "ntk_aware", "ntk_by_parts", "yarn")
FINETUNE_METHODS = ("position_interpolation", "yarn", "adjusted_base")  # 最後は診断用
JUDGED_FINETUNE_METHODS = ("position_interpolation", "yarn")
YARN_ALPHA, YARN_BETA = 1.0, 32.0  # YaRN の原論文の値に固定(評価集合で調整しない、6.1 節)
ADJUSTED_BASE = compute_ntk_aware_base(
    ROPE_BASE, FINETUNE_FACTOR, HEAD_DIM
)  # 底の調整の条件の底 b'
FINETUNE_BATCH_SIZE = 8  # 1 ステップ 8 x 4L = 8,192 トークン(008 の事前学習の 32 x L と同じ)
EVAL_BATCH_SIZE = 8
TRUNCATED_BATCH_SIZE = 64  # 実験 C の、文脈を L トークンに切り詰めた評価の 1 回の順伝播の系列数
WEIGHT_DECAY = 0.1  # 008 の事前学習と同じ(継続事前学習として同じ設定を使う)
WARMUP_RATIO = 0.1  # 007・008 と同じ
MIN_LEARNING_RATE_RATIO = 0.01  # 007・008 と同じ
CLIP_QUANTILE = 0.90  # gradient clipping の閾値 = 較正の実行の勾配ノルムの 90% 分位点(008 と同じ)
LEARNING_RATE_GRID_K = (-8, -7, -6)  # 学習率 = 10 ** (k / 2): 1e-4, 3.16e-4, 1e-3(公比 sqrt(10))
MAX_GRID_EXPANSIONS = 3  # 較正でグリッドの端が選ばれたときに拡張する最大回数(再実行で 1 -> 3。拡張後も端なら P-B2 不成立)
LEARNING_RATE_K_BOUNDS = (-11, -3)  # 拡張の範囲: 10 ** (-5.5) .. 10 ** (-1.5)
CALIBRATION_SEED = 1000  # 較正の実行のシード(本番のシード 0..4 と重ならない)
CALIBRATION_CHARS = 160_000  # 訓練テキストの末尾のこの文字数を較正用に取り分け、微調整には使わない
P0_TOLERANCE = 0.002  # 前提条件 P0 の許容誤差(6.1 節)
POSITION_BIN = L // 4  # 位置ごとの曲線の区間の幅
T4_MEMORY_GIB = 15.0
MEMORY_BUDGET_FRACTION = 0.6
SESSION_BUDGET_SECONDS = 2 * 60 * 60  # Google Colab 1 セッションの目安予算(006〜014 と同一)


def learning_rate_from_k(k: int) -> float:
    return 10.0 ** (k / 2)


# --- 水準の定義(この 1 箇所に集約する) ---
LEVELS = {
    "smoke": {
        "MAX_EVAL_WINDOWS": 6,  # 評価窓(長さ 8L)の数の上限(3 本の記事 x 2 窓、5.6 節)
        "NUM_SEEDS": 3,  # 判定に使う条件のシード数
        "NUM_DIAGNOSTIC_SEEDS": 2,  # 診断用の条件(底の調整、位置補間の切り詰めた評価)のシード数
        "FINETUNE_STEPS": 8,
        "CALIBRATION_WINDOWS": 2,
        "BOOTSTRAP_RESAMPLES": 1_000,
        "ENTROPY_WINDOWS": 1,
    },
    "prod": {
        "MAX_EVAL_WINDOWS": None,  # 作れる評価窓をすべて使う
        "NUM_SEEDS": 5,
        "NUM_DIAGNOSTIC_SEEDS": 2,
        "FINETUNE_STEPS": 400,
        "CALIBRATION_WINDOWS": 32,
        "BOOTSTRAP_RESAMPLES": 10_000,
        "ENTROPY_WINDOWS": 8,
    },
}
CURRENT_LEVEL_NAME = "smoke" if SMOKE_TEST else "prod"
CFG = LEVELS[CURRENT_LEVEL_NAME]
MAX_EVAL_WINDOWS = CFG["MAX_EVAL_WINDOWS"]
NUM_SEEDS = CFG["NUM_SEEDS"]
SEEDS = tuple(range(NUM_SEEDS))
DIAGNOSTIC_SEEDS = tuple(range(CFG["NUM_DIAGNOSTIC_SEEDS"]))
FINETUNE_STEPS = CFG["FINETUNE_STEPS"]
CALIBRATION_WINDOWS = CFG["CALIBRATION_WINDOWS"]
BOOTSTRAP_RESAMPLES = CFG["BOOTSTRAP_RESAMPLES"]
ENTROPY_WINDOWS = CFG["ENTROPY_WINDOWS"]

# --- 縮小規則の確認 ---
assert all(b == 2 * a for a, b in zip(EVAL_LENGTH_FACTORS, EVAL_LENGTH_FACTORS[1:], strict=False))
assert MAIN_FACTOR == FINETUNE_FACTOR and MAIN_FACTOR in EVAL_LENGTH_FACTORS[1:-1], (
    "判定する水準は等比の水準の内側で、微調整の長さと同じ"
)
_smoke, _prod = LEVELS["smoke"], LEVELS["prod"]
assert set(_smoke) == set(_prod)
assert 2 <= _smoke["NUM_SEEDS"] < _prod["NUM_SEEDS"]
for _level in LEVELS.values():  # 診断用の条件のシード数 < 判定に使う条件のシード数(順序関係を保つ)
    assert 2 <= _level["NUM_DIAGNOSTIC_SEEDS"] < _level["NUM_SEEDS"]
assert _smoke["FINETUNE_STEPS"] < _prod["FINETUNE_STEPS"]
assert max(1, round(WARMUP_RATIO * _smoke["FINETUNE_STEPS"])) < _smoke["FINETUNE_STEPS"]
assert _smoke["CALIBRATION_WINDOWS"] <= _prod["CALIBRATION_WINDOWS"]
assert _smoke["BOOTSTRAP_RESAMPLES"] < _prod["BOOTSTRAP_RESAMPLES"]
assert _smoke["ENTROPY_WINDOWS"] <= _prod["ENTROPY_WINDOWS"]
assert L % 4 == 0 and POSITION_BIN * 4 == L
assert (
    min(LEARNING_RATE_GRID_K) - MAX_GRID_EXPANSIONS,
    max(LEARNING_RATE_GRID_K) + MAX_GRID_EXPANSIONS,
) == LEARNING_RATE_K_BOUNDS, "拡張の範囲が上限の回数と合わない"


# --- メモリの見積もり(閉形式) ---
def estimate_eval_bytes(batch: int, length: int) -> int:
    attention = 2 * batch * NUM_HEADS * length**2 * 4  # 014 の閉形式(S と P、FP32)
    logits = batch * length * VOCAB_SIZE * 4
    return attention + logits


def estimate_finetune_bytes(batch: int, length: int) -> int:
    saved = NUM_LAYERS * batch * NUM_HEADS * length**2 * 4  # 各層の softmax の出力 P
    transient = 3 * batch * NUM_HEADS * length**2 * 4  # 1 層分の S・マスク後のスコア・P
    logits = 3 * batch * length * VOCAB_SIZE * 4  # logits・損失の中間値・その勾配
    return saved + transient + logits


_budget = MEMORY_BUDGET_FRACTION * T4_MEMORY_GIB * 2**30
_eval_bytes = estimate_eval_bytes(EVAL_BATCH_SIZE, MAX_FACTOR * L)
_finetune_bytes = estimate_finetune_bytes(FINETUNE_BATCH_SIZE, FINETUNE_FACTOR * L)
_truncated_bytes = estimate_eval_bytes(TRUNCATED_BATCH_SIZE, L)
assert _eval_bytes <= _budget and _finetune_bytes <= _budget and _truncated_bytes <= _budget

print(f"SMOKE_TEST={SMOKE_TEST}(現在の水準: {CURRENT_LEVEL_NAME!r})")
print(f"水準: {json.dumps(CFG)}")
print(
    f"評価長 = {[f * L for f in EVAL_LENGTH_FACTORS]}(s = {EVAL_LENGTH_FACTORS})、判定する水準 = {MAIN_FACTOR}L、"
    f"微調整の長さ = {FINETUNE_FACTOR}L"
)
print(
    f"YaRN: alpha = {YARN_ALPHA}, beta = {YARN_BETA}, sqrt(1/t) (s={MAIN_FACTOR}) = "
    f"{compute_yarn_attention_factor(MAIN_FACTOR):.4f}、底の調整の条件の底 b' = {ADJUSTED_BASE:,.1f}"
)
print(
    f"評価(8L = {MAX_FACTOR * L}、バッチ {EVAL_BATCH_SIZE})の見積もり: {_eval_bytes / 2**30:.2f} GiB "
    f"(うち Attention {2 * EVAL_BATCH_SIZE * NUM_HEADS * (MAX_FACTOR * L) ** 2 * 4 / 2**30:.2f} GiB、"
    f"1 系列あたり {2 * NUM_HEADS * (MAX_FACTOR * L) ** 2 * 4 / 2**20:.0f} MiB)"
)
print(
    f"微調整(4L = {FINETUNE_FACTOR * L}、バッチ {FINETUNE_BATCH_SIZE})の見積もり: {_finetune_bytes / 2**30:.2f} GiB、"
    f"切り詰めた評価(L、バッチ {TRUNCATED_BATCH_SIZE}): {_truncated_bytes / 2**30:.2f} GiB "
    f"/ 予算 {MEMORY_BUDGET_FRACTION:.0%} x {T4_MEMORY_GIB} GiB = {_budget / 2**30:.2f} GiB"
)
```

    SMOKE_TEST=False(現在の水準: 'prod')
    水準: {"MAX_EVAL_WINDOWS": null, "NUM_SEEDS": 5, "NUM_DIAGNOSTIC_SEEDS": 2, "FINETUNE_STEPS": 400, "CALIBRATION_WINDOWS": 32, "BOOTSTRAP_RESAMPLES": 10000, "ENTROPY_WINDOWS": 8}
    評価長 = [512, 1024, 2048](s = (2, 4, 8))、判定する水準 = 4L、微調整の長さ = 4L
    YaRN: alpha = 1.0, beta = 32.0, sqrt(1/t) (s=4) = 1.1386、底の調整の条件の底 b' = 43,873.0
    評価(8L = 2048、バッチ 8)の見積もり: 2.50 GiB (うち Attention 2.00 GiB、1 系列あたり 256 MiB)
    微調整(4L = 1024、バッチ 8)の見積もり: 2.50 GiB、切り詰めた評価(L、バッチ 64): 0.75 GiB / 予算 60% x 15.0 GiB = 9.00 GiB


### 5.4 既定の挙動の後方互換性(変更前のコミットとの bit 単位の一致)

`src/layers/positional_encoding.py`を変更する前のコミット(`81c6e95`、014 の最終コミット)を`git worktree`で取得し、
その worktree の`src`と現行の`src`の両方で同じスクリプトをサブプロセスとして実行する(`sys.executable`で起動するので
同一の環境)。スクリプトは CPU で、Hub の重みを読み込んだ小型 GPT を既定の`RotaryPositionEmbedding`(周波数の変換なし)で
組み立て、次の量の SHA-256 を出力する。**期待値をハードコードせず、2 つのサブプロセスの出力どうしを比べる。**

- 長さ $L$ の入力の logits、長さ $4L$ の入力の logits(cos・sin のキャッシュを再構築する経路)、位置を`positions`で
  ずらした入力の logits。
- `RotaryPositionEmbedding.apply`の出力、角周波数`inv_freq`、cos のキャッシュ。
- 長さ $L$ の入力での損失の、全パラメータの勾配。

参照側で読み込まれた`src`が worktree 配下のものであり、周波数の変換のクラスを持たない(変更前の実装である)ことも確かめる。


```python
_REFERENCE_COMMIT = (
    "81c6e95"  # src/layers/positional_encoding.py を 015 で変更する前の最後のコミット
)
_WORKTREE_DIR = ROOT / ".cache" / "_015_rope_backward_compat_worktree"

_compat_script = """
import functools
import hashlib
import json
import sys

import torch

import src.layers.positional_encoding as positional_encoding_module
from src.layers.feedforward import SwiGLUFeedForwardNetwork
from src.layers.normalization import RMSNorm
from src.layers.positional_encoding import RotaryPositionEmbedding
from src.models.gpt import GPTLanguageModel

print("SRC_FILE:" + __import__("src").__file__, file=sys.stderr)
print(
    "HAS_FREQUENCY_SCALING:" + str(hasattr(positional_encoding_module, "RotaryFrequencyScaling")),
    file=sys.stderr,
)
config = json.loads(open(sys.argv[1], encoding="utf-8").read())
length = config["sequence_length"]
d = config["d_model"] // config["num_heads"]
model = GPTLanguageModel(
    config["vocabulary_size"], config["d_model"], config["num_layers"], config["num_heads"],
    config["d_ff"], 4 * length,
    positional_transform=RotaryPositionEmbedding(d, max_position=length),
    normalization_factory=RMSNorm,
    feed_forward_factory=functools.partial(
        SwiGLUFeedForwardNetwork, config["d_model"], config["swiglu_d_ff"]
    ),
    tie_embeddings=config["tie_embeddings"],
)
model.load_state_dict(torch.load(sys.argv[2], map_location="cpu"))
model.eval()


def h(t):
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()


generator = torch.Generator().manual_seed(0)
tokens = torch.randint(0, config["vocabulary_size"], (2, 4 * length + 1), generator=generator)
out = {}
with torch.no_grad():
    out["logits_L"] = h(model(tokens[:, :length]))
    out["logits_4L"] = h(model(tokens[:, : 4 * length]))
    out["logits_shifted_positions"] = h(
        model(tokens[:, :length], positions=torch.arange(length, 2 * length))
    )
rope = RotaryPositionEmbedding(d, max_position=16)
q = torch.randn(2, 3, 40, d, generator=generator)
k = torch.randn(2, 3, 40, d, generator=generator)
rotated_q, rotated_k = rope.apply(q, k)
out["rope_apply"] = h(rotated_q) + h(rotated_k)
out["inv_freq"] = h(rope.inv_freq)
out["cos_cached"] = h(rope.cos_cached)
model.train()
logits = model(tokens[:, :length])
loss = torch.nn.functional.cross_entropy(
    logits.reshape(-1, config["vocabulary_size"]), tokens[:, 1 : length + 1].reshape(-1)
)
loss.backward()
out["gradients"] = {n: h(p.grad) for n, p in model.named_parameters() if p.grad is not None}
print(json.dumps(out))
"""

_config_file = str(Path(_config_path).resolve())
_state_file = str(Path(MODEL_STATE_PATH).resolve())
subprocess.run(["git", "worktree", "remove", "--force", str(_WORKTREE_DIR)], capture_output=True)
_r = subprocess.run(
    ["git", "worktree", "add", "--detach", str(_WORKTREE_DIR), _REFERENCE_COMMIT],
    capture_output=True,
    text=True,
)
assert _r.returncode == 0, _r.stderr
(_WORKTREE_DIR / "_compat_script.py").write_text(_compat_script)
(ROOT / "_compat_script.py").write_text(_compat_script)
try:
    _ref = subprocess.run(
        [sys.executable, "_compat_script.py", _config_file, _state_file],
        cwd=_WORKTREE_DIR,
        capture_output=True,
        text=True,
    )
    assert _ref.returncode == 0, _ref.stderr
    assert "SRC_FILE:" + str((_WORKTREE_DIR / "src" / "__init__.py").resolve()) in _ref.stderr, (
        _ref.stderr
    )
    assert "HAS_FREQUENCY_SCALING:False" in _ref.stderr, "参照側が変更後の実装を読み込んでいる"
    _new = subprocess.run(
        [sys.executable, "_compat_script.py", _config_file, _state_file],
        capture_output=True,
        text=True,
    )
    assert _new.returncode == 0, _new.stderr
    assert "SRC_FILE:" + str((ROOT / "src" / "__init__.py").resolve()) in _new.stderr, _new.stderr
    assert "HAS_FREQUENCY_SCALING:True" in _new.stderr, "現行側が変更後の実装を読み込んでいない"
    _ref_out, _new_out = json.loads(_ref.stdout), json.loads(_new.stdout)
    assert set(_ref_out) == set(_new_out)
    for _k in _ref_out:
        _match = _ref_out[_k] == _new_out[_k]
        _detail = f"({len(_ref_out[_k])} テンソル)" if isinstance(_ref_out[_k], dict) else ""
        print(f"  {_k}{_detail}: {'OK' if _match else 'MISMATCH'}")
    assert _ref_out == _new_out, "変更前の RotaryPositionEmbedding と数値的に一致しない"
    print(
        f"参照コミット {_REFERENCE_COMMIT}(worktree 配下の src を読み込んだことを確認)と現行の src で、"
        "既定の RoPE の出力・logits・勾配が bit 単位で一致: OK"
    )
finally:
    (ROOT / "_compat_script.py").unlink(missing_ok=True)
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(_WORKTREE_DIR)], capture_output=True
    )
```

      logits_L: OK
      logits_4L: OK
      logits_shifted_positions: OK
      rope_apply: OK
      inv_freq: OK
      cos_cached: OK
      gradients(38 テンソル): OK
    参照コミット 81c6e95(worktree 配下の src を読み込んだことを確認)と現行の src で、既定の RoPE の出力・logits・勾配が bit 単位で一致: OK


### 5.5 各手法の周波数と閉形式の照合

`RotaryPositionEmbedding.rotation_frequencies(l)`が返す角周波数と倍率を、3 節の式から **FP64 の NumPy で独立に計算した値**
と照合する(実装は FP32 で計算するので、相対誤差 $2 \times 10^{-6}$ 以内を一致とみなす。FP32 の丸めの単位は約
$6 \times 10^{-8}$ で、べき乗 $b^{-2i/d}$ の計算で指数 $\ln b \approx 9.2$ 倍に拡大されても $10^{-6}$ に届かない)。

- 位置補間・NTK-aware・NTK-by-parts・YaRN を $s \in \{2, 4, 8\}$ で。YaRN の倍率が $0.1 \ln s + 1$ であること。
- NTK-aware の両端($\theta'_0 = \theta_0$、$\theta'_{d/2-1} = \theta_{d/2-1}/s$)、NTK-by-parts の補間・外挿の領域。
- 位置補間の cos・sin が、位置を $m/s$ にしてスケーリングなしの式で計算した値と一致すること(3.3 節の等価性)。
- **dynamic scaling は $l \le L$ でスケーリングなしと完全に一致する**(`torch.equal`)。$l = 3L$ では静的な YaRN($s = 3$)と
  完全に一致する。
- YaRN の出力が NTK-by-parts の出力のちょうど $\sqrt{1/t}$ 倍であること(温度を cos・sin に掛ける実装)。
- 周波数を変換しても、Query と Key の内積が相対位置のみに依存すること(両方の位置を同じだけずらしても変わらない)。

あわせて、部分空間ごとの $\theta_i$・$\lambda_i$・$r_i$・$\gamma(r_i)$ と、$s = 4$ での各手法の $\theta'_i / \theta_i$ を表として印字する
(実験 A の診断量「各次元の $r_i$ と補間の度合い」)。


```python
def make_frequency_scaling(method: str, scale: float):
    # 手法の名前と倍率 s から周波数の変換を作る("none" は変換なし)。
    if method == "none":
        return None
    if method == "position_interpolation":
        return PositionInterpolation(scale)
    if method == "ntk_aware":
        return NTKAwareScaling(scale)
    if method == "ntk_by_parts":
        return NTKByPartsScaling(scale, L, YARN_ALPHA, YARN_BETA)
    if method == "yarn":
        return YaRNScaling(scale, L, YARN_ALPHA, YARN_BETA)
    if method == "dynamic_yarn":
        return DynamicScaling(lambda s: YaRNScaling(s, L, YARN_ALPHA, YARN_BETA), L)
    raise ValueError(method)


def make_rotary(method: str, scale: float, base: float = ROPE_BASE) -> RotaryPositionEmbedding:
    return RotaryPositionEmbedding(
        HEAD_DIM, base=base, max_position=L, frequency_scaling=make_frequency_scaling(method, scale)
    ).to(device)


# --- 閉形式(FP64、NumPy) ---
_index = np.arange(0, HEAD_DIM, 2) / HEAD_DIM  # 2i / d
THETA = ROPE_BASE**-_index
WAVELENGTH = 2 * np.pi / THETA
RATIO = L / WAVELENGTH  # r_i
RAMP = np.clip((RATIO - YARN_ALPHA) / (YARN_BETA - YARN_ALPHA), 0.0, 1.0)  # gamma(r_i)


def closed_form_frequencies(method: str, s: float) -> tuple[np.ndarray, float]:
    if method == "position_interpolation":
        return THETA / s, 1.0
    if method == "ntk_aware":
        return (ROPE_BASE * s ** (HEAD_DIM / (HEAD_DIM - 2))) ** -_index, 1.0
    if method == "ntk_by_parts":
        return (1 - RAMP) * THETA / s + RAMP * THETA, 1.0
    if method == "yarn":
        return (1 - RAMP) * THETA / s + RAMP * THETA, 0.1 * math.log(s) + 1
    raise ValueError(method)


FREQUENCY_RTOL = 2e-6
for _s in EVAL_LENGTH_FACTORS:
    for _method in EVAL_METHODS[1:]:
        _freq, _factor = make_rotary(_method, _s).rotation_frequencies(_s * L)
        _expected, _expected_factor = closed_form_frequencies(_method, _s)
        _freq = _freq.cpu().double().numpy()
        assert np.allclose(_freq, _expected, rtol=FREQUENCY_RTOL, atol=0.0), (_method, _s)
        assert math.isclose(_factor, _expected_factor, rel_tol=1e-12), (_method, _s)
    _ntk, _ = make_rotary("ntk_aware", _s).rotation_frequencies(_s * L)
    assert math.isclose(float(_ntk[0]), 1.0, rel_tol=1e-7)  # 最高周波は変わらない
    assert math.isclose(float(_ntk[-1]), THETA[-1] / _s, rel_tol=FREQUENCY_RTOL)  # 最低周波は 1/s
    _parts, _ = make_rotary("ntk_by_parts", _s).rotation_frequencies(_s * L)
    _parts = _parts.cpu().double().numpy()
    assert np.allclose(
        _parts[RATIO < YARN_ALPHA], THETA[RATIO < YARN_ALPHA] / _s, rtol=FREQUENCY_RTOL
    )
    assert np.allclose(_parts[RATIO > YARN_BETA], THETA[RATIO > YARN_BETA], rtol=FREQUENCY_RTOL)
print(
    f"4 手法 x s = {EVAL_LENGTH_FACTORS} の角周波数と倍率が閉形式と一致(相対誤差 {FREQUENCY_RTOL} 以内): OK"
)

# --- 位置補間 = 位置を m/s にすること ---
_positions = torch.arange(3 * L, device=device)
_rope_interp = make_rotary("position_interpolation", 4)
_cos, _sin = _rope_interp._scaled_cos_sin(_positions, 3 * L)
_angles = np.outer(np.arange(3 * L) / 4, THETA)
assert np.allclose(_cos[:, : HEAD_DIM // 2].cpu().double().numpy(), np.cos(_angles), atol=2e-5)
assert np.allclose(_sin[:, : HEAD_DIM // 2].cpu().double().numpy(), np.sin(_angles), atol=2e-5)
print("位置補間の cos・sin が、位置 m/s のスケーリングなしの cos・sin と一致: OK")

# --- dynamic scaling: l <= L でスケーリングなしと完全に一致 ---
_rope_none = make_rotary("none", 1.0)
_rope_dynamic = make_rotary("dynamic_yarn", 1.0)
_generator = torch.Generator().manual_seed(0)
for _length in (1, L // 2, L - 1, L):
    _freq, _factor = _rope_dynamic.rotation_frequencies(_length)
    assert torch.equal(_freq, _rope_none.inv_freq) and _factor == 1.0
    _q = torch.randn(2, NUM_HEADS, _length, HEAD_DIM, generator=_generator).to(device)
    _k = torch.randn(2, NUM_HEADS, _length, HEAD_DIM, generator=_generator).to(device)
    _a, _b = _rope_dynamic.apply(_q, _k), _rope_none.apply(_q, _k)
    assert torch.equal(_a[0], _b[0]) and torch.equal(_a[1], _b[1]), _length
_freq_dynamic, _factor_dynamic = _rope_dynamic.rotation_frequencies(3 * L)
_freq_static, _factor_static = make_rotary("yarn", 3).rotation_frequencies(3 * L)
assert torch.equal(_freq_dynamic, _freq_static) and _factor_dynamic == _factor_static
print(
    "dynamic YaRN: l <= L でスケーリングなしと完全一致(torch.equal)、l = 3L で静的な YaRN(s=3)と完全一致: OK"
)

# --- YaRN = NTK-by-parts の出力の sqrt(1/t) 倍 ---
_q = torch.randn(1, NUM_HEADS, 4 * L, HEAD_DIM, generator=_generator).to(device)
_k = torch.randn(1, NUM_HEADS, 4 * L, HEAD_DIM, generator=_generator).to(device)
_qy, _ky = make_rotary("yarn", 4).apply(_q, _k)
_qp, _kp = make_rotary("ntk_by_parts", 4).apply(_q, _k)
_f = compute_yarn_attention_factor(4)
assert torch.allclose(_qy, _f * _qp, rtol=1e-6, atol=1e-6) and torch.allclose(
    _ky, _f * _kp, rtol=1e-6, atol=1e-6
)
print(
    f"YaRN の出力 = NTK-by-parts の出力 x sqrt(1/t) = {_f:.4f}(logits は 1/t = {_f**2:.4f} 倍): OK"
)

# --- 相対位置のみへの依存(両方の位置を同じだけずらしても内積が変わらない) ---
for _method in EVAL_METHODS:
    _rope = make_rotary(_method, 4)
    _q1 = torch.randn(1, 1, 1, HEAD_DIM, generator=_generator).to(device)
    _k1 = torch.randn(1, 1, 1, HEAD_DIM, generator=_generator).to(device)
    _scores = []
    for _shift in (0, 300, 700):
        _qa, _ = _rope.apply(_q1, _q1, torch.tensor([_shift + 500], device=device))
        _, _ka = _rope.apply(_k1, _k1, torch.tensor([_shift + 20], device=device))
        _scores.append(float((_qa * _ka).sum()))
    assert np.allclose(_scores, _scores[0], rtol=1e-4, atol=1e-4), (_method, _scores)
print("全手法で Query・Key の内積が相対位置のみに依存: OK")

# --- 部分空間ごとの表(s = 4) ---
print(f"\n部分空間ごとの周波数(L = {L}, d = {HEAD_DIM}, b = {ROPE_BASE}, s = {MAIN_FACTOR})")
print(
    f"{'i':>3} | {'theta_i':>10} | {'lambda_i':>10} | {'r_i':>8} | {'gamma':>6} | "
    + " | ".join(f"{m[:14]:>14}" for m in EVAL_METHODS[1:])
)
_ratios_s4 = {m: closed_form_frequencies(m, MAIN_FACTOR)[0] / THETA for m in EVAL_METHODS[1:]}
for _i in range(HEAD_DIM // 2):
    print(
        f"{_i:>3} | {THETA[_i]:>10.3e} | {WAVELENGTH[_i]:>10.1f} | {RATIO[_i]:>8.3f} | {RAMP[_i]:>6.3f} | "
        + " | ".join(f"{_ratios_s4[m][_i]:>14.4f}" for m in EVAL_METHODS[1:])
    )
print(
    f"lambda_i < L の部分空間: {int((WAVELENGTH < L).sum())} 個、r_i > beta(外挿): {int((RATIO > YARN_BETA).sum())} 個、"
    f"alpha <= r_i <= beta(混合): {int(((RATIO >= YARN_ALPHA) & (RATIO <= YARN_BETA)).sum())} 個、"
    f"r_i < alpha(補間): {int((RATIO < YARN_ALPHA).sum())} 個"
)

fig, ax = plt.subplots(figsize=(7, 4))
for _method in EVAL_METHODS[1:]:
    ax.plot(range(HEAD_DIM // 2), _ratios_s4[_method], "o-", label=_method, markersize=4)
ax.axhline(1 / MAIN_FACTOR, color="gray", linestyle=":", label="1/s")
ax.set_xlabel("subspace index i (high frequency -> low frequency)")
ax.set_ylabel("theta'_i / theta_i")
ax.set_title(f"Frequency ratio per subspace (s = {MAIN_FACTOR}, L = {L}, d = {HEAD_DIM})")
ax.legend(fontsize=8)
plt.show()
```

    4 手法 x s = (2, 4, 8) の角周波数と倍率が閉形式と一致(相対誤差 2e-06 以内): OK
    位置補間の cos・sin が、位置 m/s のスケーリングなしの cos・sin と一致: OK
    dynamic YaRN: l <= L でスケーリングなしと完全一致(torch.equal)、l = 3L で静的な YaRN(s=3)と完全一致: OK
    YaRN の出力 = NTK-by-parts の出力 x sqrt(1/t) = 1.1386(logits は 1/t = 1.2965 倍): OK
    全手法で Query・Key の内積が相対位置のみに依存: OK
    
    部分空間ごとの周波数(L = 256, d = 32, b = 10000.0, s = 4)
      i |    theta_i |   lambda_i |      r_i |  gamma | position_inter |      ntk_aware |   ntk_by_parts |           yarn
      0 |  1.000e+00 |        6.3 |   40.744 |  1.000 |         0.2500 |         1.0000 |         1.0000 |         1.0000
      1 |  5.623e-01 |       11.2 |   22.912 |  0.707 |         0.2500 |         0.9117 |         0.7801 |         0.7801
      2 |  3.162e-01 |       19.9 |   12.884 |  0.383 |         0.2500 |         0.8312 |         0.5375 |         0.5375
      3 |  1.778e-01 |       35.3 |    7.245 |  0.201 |         0.2500 |         0.7579 |         0.4011 |         0.4011
      4 |  1.000e-01 |       62.8 |    4.074 |  0.099 |         0.2500 |         0.6910 |         0.3244 |         0.3244
      5 |  5.623e-02 |      111.7 |    2.291 |  0.042 |         0.2500 |         0.6300 |         0.2812 |         0.2812
      6 |  3.162e-02 |      198.7 |    1.288 |  0.009 |         0.2500 |         0.5743 |         0.2570 |         0.2570
      7 |  1.778e-02 |      353.3 |    0.725 |  0.000 |         0.2500 |         0.5236 |         0.2500 |         0.2500
      8 |  1.000e-02 |      628.3 |    0.407 |  0.000 |         0.2500 |         0.4774 |         0.2500 |         0.2500
      9 |  5.623e-03 |     1117.3 |    0.229 |  0.000 |         0.2500 |         0.4353 |         0.2500 |         0.2500
     10 |  3.162e-03 |     1986.9 |    0.129 |  0.000 |         0.2500 |         0.3969 |         0.2500 |         0.2500
     11 |  1.778e-03 |     3533.3 |    0.072 |  0.000 |         0.2500 |         0.3618 |         0.2500 |         0.2500
     12 |  1.000e-03 |     6283.2 |    0.041 |  0.000 |         0.2500 |         0.3299 |         0.2500 |         0.2500
     13 |  5.623e-04 |    11173.3 |    0.023 |  0.000 |         0.2500 |         0.3008 |         0.2500 |         0.2500
     14 |  3.162e-04 |    19869.2 |    0.013 |  0.000 |         0.2500 |         0.2742 |         0.2500 |         0.2500
     15 |  1.778e-04 |    35332.9 |    0.007 |  0.000 |         0.2500 |         0.2500 |         0.2500 |         0.2500
    lambda_i < L の部分空間: 7 個、r_i > beta(外挿): 1 個、alpha <= r_i <= beta(混合): 6 個、r_i < alpha(補間): 9 個



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/015_long_context_extension/output_26_1.png)
    




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/03_efficient_training/015_long_context_extension.ipynb
