---
title: "Flash Attention(実装・実験編 1/3)"
---

この記事は後編(実装・実験編 1/3)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/014_flash_attention-theory)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/014_flash_attention-practice-2)。

## 4. 実装方針 / Implementation Policy

**`src/`に切り出す(スクラッチ実装、本トピックで新規作成)**: `src/layers/flash_attention.py`

- `flash_attention_forward(query, key, value, block_size_query, block_size_key, causal=False, scale=None)`:
  3.5 節の擬似コードどおりのタイリングと online softmax による順伝播。出力 $O$ と行ごとの logsumexp $L$ を返す。
  入力が FP16 でも、統計量 $m, \ell, L$ と $\tilde{O}_i$ の累積は FP32 で行う(入力が FP64 の場合は FP64)。
  系列長がブロックの行数で割り切れない場合は最後のブロックを短くし、ブロックの行数が系列長以上の場合は
  ブロックが 1 つになる。
- `flash_attention_backward(...)`: 3.6 節の再計算による逆伝播($P_{ij}$ をブロックごとに再計算する)。
- `FlashAttentionFunction(torch.autograd.Function)`: 上の 2 つを順伝播・逆伝播とする。`ctx`に保存するのは
  $Q, K, V, O, L$ のみである。
- `count_block_pairs()`: 計算するブロックの組の数と全体の数(因果マスクで飛ばしたブロックの割合の計算に使う)。
  順伝播と同じ規則(`key_block_end()`)で数える。
- 入力の形状は`(batch, heads, sequence, head_dim)`で、001 の`scaled_dot_product_attention()`と揃える。バッチと
  ヘッドの次元はまとめてベクトル化し、系列方向のみをブロックに分ける。
- 既存の`src/layers/attention.py`・`MultiHeadAttention`は変更しない(小型 GPT には組み込まない)。

**ノートブック内に直接書く(014 固有)**:

- online softmax の 1 次元の例(5.4 節)。
- 実験のハーネス: 入力の生成、誤差の計算、ピークメモリ・保存テンソルの計測、SDPA のバックエンドの時間計測(5.6 節)。
- 標準の Attention の参照実装には 001 の`scaled_dot_product_attention()`をそのまま使う。FP64 の参照値もこの関数に
  FP64 の入力を与えて計算する。

**SDPA のバックエンドの指定**: `torch.nn.attention.sdpa_kernel`(インストールされている torch 2.13 の API)で
math・memory-efficient をそれぞれ 1 つだけ許可して呼ぶ。指定したバックエンドが使えない場合に例外になることを
5.5 節で確かめ、実験 E では加えて、実際に実行された演算の名前を profiler で確認する(5.6 節)。
**バックエンドの選択を無視して黙って別の実装で実行する環境がある** (第 1 段階で確認したローカルの MPS では、
memory-efficient・flash を指定しても例外にならなかった)ため、例外にならないことだけを「指定したバックエンドで
実行された」根拠にしない。

**デバイス**: CUDA が使える場合は CUDA、使えない場合は CPU を使う。MPS は FP64 に対応していないため
(実験 A・B の参照値に FP64 を使う)使わない。

**アップロード方針**: 学習もモデルもないため、Hugging Face Hub へのアップロードはない。生成物はセル出力
のみである。

## 5. 実装 / Implementation

### 5.1 環境セットアップ(Google Colab)と実行環境の記録

依存関係のインストールの後に、実行環境(GPU 名・compute capability・GPU の総メモリ・torch と CUDA・cuDNN の
バージョン・デバイス・コミット・実行日時)を印字する。結果・考察で実行環境に言及するときは、この印字を出典とする。


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

import torch  # noqa: E402

from src.utils.environment import print_execution_environment  # noqa: E402

# FP64 の参照値を使うため、MPS(FP64 非対応)は使わず、CUDA がなければ CPU で実行する
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
execution_environment = print_execution_environment(device)
```

    Cloning into 'ai-theories'...
    remote: Enumerating objects: 1058, done.[K
    remote: Counting objects: 100% (98/98), done.[K
    remote: Compressing objects: 100% (69/69), done.[K
    remote: Total 1058 (delta 52), reused 60 (delta 29), pack-reused 960 (from 1)[K
    Receiving objects: 100% (1058/1058), 10.87 MiB | 15.32 MiB/s, done.
    Resolving deltas: 100% (613/613), done.
    /content/ai-theories
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m20.5/20.5 MB[0m [31m41.0 MB/s[0m eta [36m0:00:00[0m
    [?25h[2mUsing Python 3.13.15 environment at: /usr[0m
    [2K[2mResolved [1m52 packages[0m [2min 464ms[0m[0m
    [2K[2mPrepared [1m31 packages[0m [2min 43.73s[0m[0m
    [2mUninstalled [1m17 packages[0m [2min 769ms[0m[0m
    [2K[2mInstalled [1m31 packages[0m [2min 303ms[0m[0m
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
      コミット / git commit                  : 055f6115350d697ea47f7a6792c207017861813b
      未コミットの変更 / uncommitted changes : なし
      実行日時 (UTC)                         : 2026-09-24T22:02:07+00:00



```python
import json
import math
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.multiprocessing.reductions import StorageWeakRef
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from src.layers.attention import create_causal_mask, scaled_dot_product_attention
from src.layers.flash_attention import (
    FlashAttentionFunction,
    count_block_pairs,
    flash_attention_backward,
    flash_attention_forward,
)
from src.utils.reporting import dumps_compact_json
from src.utils.statistics import fit_power_law_exponent

LOG2 = math.log(2.0)
HEAD_DIM = 64  # 全実験で共通の d


def sync_device() -> None:
    # 時間計測・メモリ計測の直前・直後に、非同期に実行される GPU の処理の完了を待つ。
    if device.type == "cuda":
        torch.cuda.synchronize()


precondition_status: dict[str, bool] = {}  # 前提条件の成否(6.1 節で宣言、各実験の節で記録)
print(f"device: {device}")
```

    device: cuda


### 5.2 スケールの設定(`SMOKE_TEST`の配線)

水準の定義をこの 1 箇所(`LEVELS`)に集約する。`SMOKE_TEST = True`はローカルでのコード経路の確認用、
`SMOKE_TEST = False`は Google Colab T4 での本番実行用である。

**縮小の規則**:

- 系列長の水準は、本番・スモークテストとも公比 2 の等比数列とする。スモークテストでは水準の数と最大値を縮小する。
  実験 A・B・D のスモークテストの水準は、本番の水準の先頭の部分列である。実験 C・E は、本番の水準がローカルの CPU では
  重すぎる(実験 C の標準の実装は最大の水準で約 8 GiB)ため、より小さい系列長から始まる公比 2 の数列とする。
- 実験 A・B のシード数は両実験で共通の値とし(本番 5、スモークテスト 2)、系列長の水準も両実験で共通にする。
- 実験 E の反復回数は本番 20、スモークテスト 5。ウォームアップの回数(3)は縮小しない。
- ブロックの行数・バッチ・ヘッド数・$d$・入力のスケール・dtype は縮小しない(全水準・両方の設定で共通)。

**実験 C の最大の系列長**: 標準の実装のピークメモリの増分の閉形式(6.1 節の実験 C)
$2 \cdot B H N^2 \cdot 4$ バイト(FP32 の $S$ と $P$ が同時に存在する)が、T4 の総メモリ(約 15 GiB)の 60% 以下に収まる
最大の公比 2 の水準を下のセルで求めて宣言する(60% は、CUDA のメモリアロケータのキャッシュ・断片化と行列積の作業領域の
余裕として置いた値である)。


```python
SMOKE_TEST = False  # Claude Code はこの True 側のみ実行する(Colab T4 では False に切り替える)
_smoke_tag = "[動作確認のみ、結論ではない] " if SMOKE_TEST else ""
_plot_tag = (
    "[smoke test] " if SMOKE_TEST else ""
)  # 図のタイトル用(フォントに日本語がない環境がある)

# --- 実験の条件(本番実行前に宣言し、SMOKE_TEST で変えない) ---
# 実験 A・B
BATCH_AB, HEADS_AB = 1, 2
BLOCK_AB = (64, 64)  # (B_r, B_c)
SIGMAS_AB = {"normal": 1.0, "large": 6.0}  # Q・K の各要素の標準偏差(スコアの標準偏差は sigma^2)
CAUSAL_AB = (False, True)
# 実験 C
BATCH_C, HEADS_C = 1, 4
BLOCK_C = (128, 128)
T4_MEMORY_GIB = 15.0
MEMORY_BUDGET_FRACTION_C = 0.6
# 実験 D
BATCH_D, HEADS_D = 1, 4
BLOCK_D = (128, 128)
# 実験 E
BATCH_E, HEADS_E = 1, 8
BLOCK_E = (128, 128)  # スクラッチ実装の時間(診断量)に使うブロックの行数
WARMUP_E = 3
EXPONENT_DIFF_THRESHOLD = 0.5  # 実験 C・D の判定の閾値
SESSION_BUDGET_SECONDS = 2 * 60 * 60  # Google Colab 1 セッションの目安予算(006〜013 と同一)


def geometric(start: int, count: int) -> tuple[int, ...]:
    return tuple(start * 2**k for k in range(count))


# --- 水準の定義(この 1 箇所に集約する) ---
LEVELS = {
    "smoke": {
        "SEQUENCE_LENGTHS_AB": geometric(80, 3),  # 80, 160, 320
        "NUM_SEEDS_AB": 2,
        "SEQUENCE_LENGTHS_C": geometric(256, 3),  # 256, 512, 1024
        "SEQUENCE_LENGTHS_D": geometric(512, 3),  # 512, 1024, 2048
        "SEQUENCE_LENGTHS_E": geometric(128, 3),  # 128, 256, 512
        "REPETITIONS_E": 5,
    },
    "prod": {
        "SEQUENCE_LENGTHS_AB": geometric(80, 5),  # 80, ..., 1280
        "NUM_SEEDS_AB": 5,
        "SEQUENCE_LENGTHS_C": geometric(1024, 5),  # 1024, ..., 16384
        "SEQUENCE_LENGTHS_D": geometric(512, 5),  # 512, ..., 8192
        "SEQUENCE_LENGTHS_E": geometric(256, 6),  # 256, ..., 8192
        "REPETITIONS_E": 20,
    },
}
CURRENT_LEVEL_NAME = "smoke" if SMOKE_TEST else "prod"
CFG = LEVELS[CURRENT_LEVEL_NAME]
SEQUENCE_LENGTHS_AB = CFG["SEQUENCE_LENGTHS_AB"]
NUM_SEEDS_AB = CFG["NUM_SEEDS_AB"]
SEEDS_AB = tuple(range(NUM_SEEDS_AB))
SEQUENCE_LENGTHS_C = CFG["SEQUENCE_LENGTHS_C"]
SEQUENCE_LENGTHS_D = CFG["SEQUENCE_LENGTHS_D"]
SEQUENCE_LENGTHS_E = CFG["SEQUENCE_LENGTHS_E"]
REPETITIONS_E = CFG["REPETITIONS_E"]


def estimate_standard_peak_bytes_c(n: int) -> int:
    # 実験 C の標準の実装のピークメモリの増分の閉形式(FP32 の S と P が同時に存在する)
    return 2 * BATCH_C * HEADS_C * n * n * 4


# --- 縮小規則の確認 ---
for _lv in LEVELS.values():
    for _key in (
        "SEQUENCE_LENGTHS_AB",
        "SEQUENCE_LENGTHS_C",
        "SEQUENCE_LENGTHS_D",
        "SEQUENCE_LENGTHS_E",
    ):
        _seq = _lv[_key]
        assert all(b == 2 * a for a, b in zip(_seq, _seq[1:], strict=False)), f"{_key} が公比 2 の等比でない"
        assert len(_seq) >= 3, "べき指数の標準誤差には 3 水準以上が必要"
    assert _lv["NUM_SEEDS_AB"] >= 2, "標本標準偏差には 2 シード以上が必要"
for _key in ("SEQUENCE_LENGTHS_AB", "SEQUENCE_LENGTHS_D"):
    _s, _p = LEVELS["smoke"][_key], LEVELS["prod"][_key]
    assert _p[: len(_s)] == _s, f"{_key}: スモークテストの水準が本番の先頭の部分列でない"
for _key in ("SEQUENCE_LENGTHS_C", "SEQUENCE_LENGTHS_E"):
    assert max(LEVELS["smoke"][_key]) <= max(LEVELS["prod"][_key])
assert LEVELS["smoke"]["REPETITIONS_E"] < LEVELS["prod"]["REPETITIONS_E"]
# 実験 A・B の水準は、ブロックの行数で割り切れない長さと割り切れる長さの両方を含む
for _lv in LEVELS.values():
    _seq = _lv["SEQUENCE_LENGTHS_AB"]
    assert any(n % BLOCK_AB[0] != 0 for n in _seq) and any(n % BLOCK_AB[0] == 0 for n in _seq)

# --- 実験 C の最大の系列長(閉形式の見積もりから決める) ---
_budget_c = MEMORY_BUDGET_FRACTION_C * T4_MEMORY_GIB * 2**30
_max_c = max(LEVELS["prod"]["SEQUENCE_LENGTHS_C"])
assert estimate_standard_peak_bytes_c(_max_c) <= _budget_c, "実験 C の最大の水準が予算を超える"
assert estimate_standard_peak_bytes_c(2 * _max_c) > _budget_c, (
    "実験 C の最大の水準をさらに 2 倍にできる"
)

print(f"SMOKE_TEST={SMOKE_TEST}(現在の水準: {CURRENT_LEVEL_NAME!r})")
print(
    f"実験 A・B: 系列長={SEQUENCE_LENGTHS_AB}, シード={SEEDS_AB}, (B_r, B_c)={BLOCK_AB}, "
    f"batch={BATCH_AB}, heads={HEADS_AB}, d={HEAD_DIM}, sigma={SIGMAS_AB}, causal={CAUSAL_AB}"
)
print(
    f"実験 C: 系列長={SEQUENCE_LENGTHS_C}, (B_r, B_c)={BLOCK_C}, batch={BATCH_C}, heads={HEADS_C}, FP32"
)
print(
    f"  標準の実装のピークメモリの増分の見積もり(本番の最大 N={_max_c}): "
    f"{estimate_standard_peak_bytes_c(_max_c) / 2**30:.2f} GiB"
    f"(予算 {MEMORY_BUDGET_FRACTION_C:.0%} x {T4_MEMORY_GIB} GiB = {_budget_c / 2**30:.2f} GiB、"
    f"N={2 * _max_c} では {estimate_standard_peak_bytes_c(2 * _max_c) / 2**30:.2f} GiB)"
)
print(
    f"実験 D: 系列長={SEQUENCE_LENGTHS_D}, (B_r, B_c)={BLOCK_D}, batch={BATCH_D}, heads={HEADS_D}, FP32"
)
print(
    f"実験 E: 系列長={SEQUENCE_LENGTHS_E}, 反復={REPETITIONS_E}, ウォームアップ={WARMUP_E}, "
    f"batch={BATCH_E}, heads={HEADS_E}, FP16"
)
```

    SMOKE_TEST=False(現在の水準: 'prod')
    実験 A・B: 系列長=(80, 160, 320, 640, 1280), シード=(0, 1, 2, 3, 4), (B_r, B_c)=(64, 64), batch=1, heads=2, d=64, sigma={'normal': 1.0, 'large': 6.0}, causal=(False, True)
    実験 C: 系列長=(1024, 2048, 4096, 8192, 16384), (B_r, B_c)=(128, 128), batch=1, heads=4, FP32
      標準の実装のピークメモリの増分の見積もり(本番の最大 N=16384): 8.00 GiB(予算 60% x 15.0 GiB = 9.00 GiB、N=32768 では 32.00 GiB)
    実験 D: 系列長=(512, 1024, 2048, 4096, 8192), (B_r, B_c)=(128, 128), batch=1, heads=4, FP32
    実験 E: 系列長=(256, 512, 1024, 2048, 4096, 8192), 反復=20, ウォームアップ=3, batch=1, heads=8, FP16


### 5.3 入力の生成

入力は乱数で作る。**CPU 上の FP64 で生成してから目的の dtype・デバイスに変換する** ので、同じシードからはデバイスに
よらず同じ値が得られる。Query・Key には標準偏差 $\sigma$ を掛ける(スコア $\tau q \cdot k$ の標準偏差は $\sigma^2$ になる)。
Value と出力の勾配 $dO$ は標準正規分布とする。


```python
def make_inputs(
    seed: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    sigma: float = 1.0,
    with_grad_output: bool = False,
) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tensors = [
        torch.randn(shape, generator=generator, dtype=torch.float64) * sigma,  # Q
        torch.randn(shape, generator=generator, dtype=torch.float64) * sigma,  # K
        torch.randn(shape, generator=generator, dtype=torch.float64),  # V
    ]
    if with_grad_output:
        tensors.append(torch.randn(shape, generator=generator, dtype=torch.float64))  # dO
    return [t.to(device=device, dtype=dtype) for t in tensors]


def relative_max_error(x: torch.Tensor, reference: torch.Tensor) -> float:
    # 相対誤差 ||x - ref||_inf / ||ref||_inf(要素ごとの最大絶対値によるノルム、FP64 で計算する)
    return float((x.double() - reference).abs().max() / reference.abs().max())


_a, _b = make_inputs(0, (1, 1, 4, 3), torch.float32), make_inputs(0, (1, 1, 4, 3), torch.float32)
assert all(torch.equal(x, y) for x, y in zip(_a, _b, strict=False)), "同じシードで同じ入力にならない"
```

### 5.4 online softmax の 1 次元の例

3.4 節の 3 つの計算を 1 次元のベクトルで比べる。

- 素朴な計算: $e^{x_i} / \sum_k e^{x_k}$(最大値を引かない)
- 最大値を引く 3 パスの計算
- online softmax(1 要素ずつの更新と、ブロックごとの統計量の合成)

通常のスケールの入力では 3 つとも`torch.softmax`と一致し、大きな値を含む入力では素朴な計算だけがオーバーフローして
NaN になることを確かめる(不変条件のアサーション)。大きな値の入力は標準偏差 300 とし、最大値が FP32 の境界(約 88.7)と
FP64 の境界(約 709.8)をともに超えるようにする。


```python
def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    e = torch.exp(x)
    return e / e.sum()


def three_pass_softmax(x: torch.Tensor) -> torch.Tensor:
    m = x.max()  # パス 1
    ell = torch.exp(x - m).sum()  # パス 2
    return torch.exp(x - m) / ell  # パス 3


def online_softmax_statistics(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 1 要素ずつ m_j = max(m_{j-1}, x_j), l_j = l_{j-1} exp(m_{j-1} - m_j) + exp(x_j - m_j)
    m = torch.tensor(float("-inf"), dtype=x.dtype)
    ell = torch.tensor(0.0, dtype=x.dtype)
    for value in x:
        m_new = torch.maximum(m, value)
        ell = ell * torch.exp(m - m_new) + torch.exp(value - m_new)
        m = m_new
    return m, ell


def blockwise_softmax_statistics(x: torch.Tensor, block: int) -> tuple[torch.Tensor, torch.Tensor]:
    # ブロックごとの統計量 (m^(b), l^(b)) を m = max, l = sum exp(m^(b) - m) l^(b) で合成する
    m = torch.tensor(float("-inf"), dtype=x.dtype)
    ell = torch.tensor(0.0, dtype=x.dtype)
    for start in range(0, len(x), block):
        chunk = x[start : start + block]
        m_chunk = chunk.max()
        ell_chunk = torch.exp(chunk - m_chunk).sum()
        m_new = torch.maximum(m, m_chunk)
        ell = ell * torch.exp(m - m_new) + ell_chunk * torch.exp(m_chunk - m_new)
        m = m_new
    return m, ell


_gen = torch.Generator().manual_seed(0)
for _label, _scale in (("通常のスケール", 1.0), ("大きなスケール", 300.0)):
    for _dtype in (torch.float64, torch.float32):
        _x = (torch.randn(1000, generator=_gen, dtype=torch.float64) * _scale).to(_dtype)
        _reference = torch.softmax(_x, dim=0)
        _m, _ell = online_softmax_statistics(_x)
        _mb, _ellb = blockwise_softmax_statistics(_x, block=64)
        _online = torch.exp(_x - _m) / _ell
        _blockwise = torch.exp(_x - _mb) / _ellb
        _naive = naive_softmax(_x)
        _tol = (
            dict(rtol=1e-12, atol=1e-15) if _dtype == torch.float64 else dict(rtol=1e-5, atol=1e-7)
        )
        assert _m == _x.max() and _mb == _x.max(), "online softmax の最大値が一致しない"
        assert torch.allclose(_online, _reference, **_tol), "online softmax(1 要素ずつ)が一致しない"
        assert torch.allclose(_blockwise, _reference, **_tol), (
            "online softmax(ブロック)が一致しない"
        )
        assert torch.allclose(three_pass_softmax(_x), _reference, **_tol)
        assert float(_ell) >= 1.0 and float(_ellb) >= 1.0, "正規化定数が 1 以上でない"
        _naive_ok = bool(torch.isfinite(_naive).all())
        if _scale == 1.0:
            assert _naive_ok and torch.allclose(_naive, _reference, **_tol)
        else:
            assert not _naive_ok, "大きなスケールで素朴な計算がオーバーフローしていない"
        print(
            f"{_label} ({str(_dtype).removeprefix('torch.')}): max(x)={float(_x.max()):.2f}, "
            f"online と softmax の最大差={float((_online - _reference).abs().max()):.2e}, "
            f"ブロック合成との最大差={float((_blockwise - _reference).abs().max()):.2e}, "
            f"ell={float(_ell):.4f}, 素朴な計算の NaN・無限大の数={int((~torch.isfinite(_naive)).sum())}"
        )
print("online softmax の 1 次元の例: 全アサーション OK")
```

    通常のスケール (float64): max(x)=2.83, online と softmax の最大差=3.47e-18, ブロック合成との最大差=8.67e-19, ell=95.6653, 素朴な計算の NaN・無限大の数=0
    通常のスケール (float32): max(x)=3.14, online と softmax の最大差=1.12e-08, ブロック合成との最大差=9.31e-10, ell=68.2426, 素朴な計算の NaN・無限大の数=0
    大きなスケール (float64): max(x)=889.63, online と softmax の最大差=1.06e-109, ブロック合成との最大差=1.06e-109, ell=1.0011, 素朴な計算の NaN・無限大の数=4
    大きなスケール (float32): max(x)=908.28, online と softmax の最大差=0.00e+00, ブロック合成との最大差=0.00e+00, ell=1.0000, 素朴な計算の NaN・無限大の数=418
    online softmax の 1 次元の例: 全アサーション OK


### 5.5 不変条件の確認(`src/layers/flash_attention.py`)

- **順伝播の正しさ(FP64)**: 系列長がブロックの行数で割り切れない場合・割り切れる場合・ブロックの行数が系列長以上の
  場合・$B_r \ne B_c$ の場合のそれぞれで、因果マスクの有無の両方について、001 の`scaled_dot_product_attention()`
  (FP64)との差が $10^{-12}$ 以下であること。logsumexp $L$ が`torch.logsumexp`と一致すること。
- **`gradcheck`(FP64)**: 同じ形状の組で`FlashAttentionFunction`の勾配が数値微分と一致すること。
- **逆伝播の正しさ(FP64)**: `flash_attention_backward()`の $dQ, dK, dV$ が、001 の関数を autograd で微分した値と
  $10^{-12}$ 以下で一致すること。
- **保存するテンソル**: `FlashAttentionFunction`が逆伝播のために保存するのが $Q, K, V, O, L$ の 5 つのみで
  あること(`saved_tensors_hooks`で記録したテンソルの記憶領域が、入力 3 つ・出力・$L$ と一致する)。
- **ブロックの組の数**: `count_block_pairs()`が、順伝播で実際に計算したブロックの組の数と一致すること。
- **SDPA のバックエンド**: 使えないバックエンドだけを許可すると例外になること。CUDA で compute capability が
  8.0 未満(T4 など)なら flash、CPU なら memory-efficient が使えないので、それを許可して例外を確かめる。


```python
_cases = [  # (N, B_r, B_c)
    (10, 4, 3),  # 割り切れない(B_r != B_c)
    (33, 8, 8),  # 割り切れない
    (32, 8, 8),  # 割り切れる
    (7, 16, 16),  # ブロックの行数が系列長以上
    (16, 16, 16),  # ブロックの行数 = 系列長
    (1, 1, 1),
]
for _n, _br, _bc in _cases:
    for _causal in (False, True):
        _q, _k, _v, _do = make_inputs(
            _n, (2, 3, _n, 8), torch.float64, sigma=2.0, with_grad_output=True
        )
        _mask = create_causal_mask(_n, device=device) if _causal else None
        _q.requires_grad_(), _k.requires_grad_(), _v.requires_grad_()
        _ref, _ = scaled_dot_product_attention(_q, _k, _v, _mask)
        _ref_grads = torch.autograd.grad(_ref, (_q, _k, _v), _do)
        with torch.no_grad():
            _out, _lse = flash_attention_forward(_q, _k, _v, _br, _bc, _causal)
            _scores = _q @ _k.transpose(-2, -1) / math.sqrt(8)
            if _causal:
                _scores = _scores.masked_fill(~_mask, float("-inf"))
            assert (_out - _ref).abs().max() <= 1e-12, (_n, _br, _bc, _causal)
            assert (_lse - torch.logsumexp(_scores, dim=-1)).abs().max() <= 1e-12
            _grads = flash_attention_backward(_q, _k, _v, _out, _lse, _do, _br, _bc, _causal)
            for _g, _rg in zip(_grads, _ref_grads, strict=False):
                assert (_g - _rg).abs().max() <= 1e-12, (_n, _br, _bc, _causal)
        assert torch.autograd.gradcheck(
            lambda q, k, v, br=_br, bc=_bc, c=_causal: FlashAttentionFunction.apply(
                q, k, v, br, bc, c, None
            ),
            (_q, _k, _v),
        )
print(f"順伝播・logsumexp・逆伝播(FP64)と gradcheck: {len(_cases) * 2} 通り OK")

# --- 保存するテンソルが Q, K, V, O, L のみであること ---
_saved = []
_q, _k, _v = [t.requires_grad_() for t in make_inputs(1, (1, 2, 50, 8), torch.float32)]
with torch.autograd.graph.saved_tensors_hooks(lambda t: (_saved.append(t), t)[1], lambda t: t):
    _out = FlashAttentionFunction.apply(_q, _k, _v, 16, 16, True, None)
assert len(_saved) == 5, f"保存したテンソルの数が 5 でない: {len(_saved)}"
_ptrs = [t.untyped_storage().data_ptr() for t in _saved]
assert _ptrs[:4] == [t.untyped_storage().data_ptr() for t in (_q, _k, _v, _out)], (
    "Q, K, V, O の順でない"
)
assert _saved[4].shape == (1, 2, 50) and _saved[4].dtype == torch.float32, "5 つ目が L でない"
assert len(_out.grad_fn.saved_tensors) == 5
print(
    "FlashAttentionFunction の保存テンソル: "
    + ", ".join(f"{name}{tuple(t.shape)}" for name, t in zip("QKVOL", _saved, strict=False))
)

# --- count_block_pairs が順伝播で実際に計算したブロックの組の数と一致すること ---
_orig_matmul = torch.matmul
for _n, _br, _bc in [(100, 16, 16), (100, 16, 8), (64, 16, 16), (10, 32, 32)]:
    for _causal in (False, True):
        _calls = 0

        def _counting_matmul(*args, **kwargs):
            global _calls
            _calls += 1
            return _orig_matmul(*args, **kwargs)

        torch.matmul = _counting_matmul
        try:
            with torch.no_grad():
                flash_attention_forward(
                    *make_inputs(0, (1, 1, _n, 8), torch.float32), _br, _bc, _causal
                )
        finally:
            torch.matmul = _orig_matmul
        _processed, _total = count_block_pairs(_n, _n, _br, _bc, _causal)
        assert _calls == 2 * _processed, (
            _n,
            _br,
            _bc,
            _causal,
        )  # ブロックの組 1 つあたり行列積 2 回
        assert _processed <= _total and (_causal or _processed == _total)
print("count_block_pairs と実際に計算したブロックの組の数の一致: OK")

# --- SDPA: 使えないバックエンドだけを許可すると例外になること ---
if device.type == "cuda" and torch.cuda.get_device_capability(device) < (8, 0):
    UNAVAILABLE_SDPA_BACKEND = SDPBackend.FLASH_ATTENTION
elif device.type == "cpu":
    UNAVAILABLE_SDPA_BACKEND = SDPBackend.EFFICIENT_ATTENTION
else:
    UNAVAILABLE_SDPA_BACKEND = None
if UNAVAILABLE_SDPA_BACKEND is not None:
    _dtype_probe = torch.float16 if device.type == "cuda" else torch.float32
    _q, _k, _v = make_inputs(0, (1, 2, 64, HEAD_DIM), _dtype_probe)
    _raised = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 使えない理由の警告は例外の確認には不要
        try:
            with sdpa_kernel(UNAVAILABLE_SDPA_BACKEND):
                F.scaled_dot_product_attention(_q, _k, _v)
        except RuntimeError as error:
            _raised = error
    assert _raised is not None, (
        f"{UNAVAILABLE_SDPA_BACKEND} を指定しても例外にならず、黙って実行された"
    )
    print(
        f"{UNAVAILABLE_SDPA_BACKEND.name} のみを許可 -> RuntimeError: {str(_raised).splitlines()[0][:120]}"
    )
else:
    print("このデバイスでは使えないバックエンドが既知でないため、例外の確認を省略した")
```

    /usr/local/lib/python3.13/dist-packages/torch/autograd/graph.py:979: UserWarning: Attempting to run cuBLAS, but there was no current CUDA context! Attempting to set the primary context... (Triggered internally at /__w/pytorch/pytorch/aten/src/ATen/cuda/CublasHandlePool.cpp:408.)
      return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass


    順伝播・logsumexp・逆伝播(FP64)と gradcheck: 12 通り OK
    FlashAttentionFunction の保存テンソル: Q(1, 2, 50, 8), K(1, 2, 50, 8), V(1, 2, 50, 8), O(1, 2, 50, 8), L(1, 2, 50)
    count_block_pairs と実際に計算したブロックの組の数の一致: OK
    FLASH_ATTENTION のみを許可 -> RuntimeError: No available kernel. Aborting execution.


### 5.6 実験のハーネス

- `run_sample_a()`・`run_sample_b()`: 実験 A・B の 1 標本(1 つの系列長・因果マスクの有無・スケール・シード)。
- `measure_forward_memory()`: 実験 C の順伝播(`torch.no_grad()`)のピークメモリ。CUDA では
  `torch.cuda.max_memory_allocated()`で測る。加えて、デバイスによらない診断量として、演算が作ったテンソルの記憶領域の
  バイト数の合計の最大値(生存中のテンソルのバイト数のピーク)を`LiveTensorBytesTracker`で数える。CUDA 以外
  (ローカルのスモークテスト)では後者のみが得られる。
- `measure_saved_tensor_bytes()`: 実験 D の、逆伝播のために保存されるテンソルのバイト数。`saved_tensors_hooks`で
  保存されるテンソルを記録し、**記憶領域(storage)ごとに 1 回だけ** 数える(転置などのビューや、softmax の出力と
  次の行列積の入力のように同じ記憶領域を複数回保存しても、実際に保持されるのは 1 つだけであるため)。
- `run_level_e()`: 実験 E の 1 つの系列長での SDPA の時間計測。math と memory-efficient を 1 回ずつ続けて実行する
  反復を繰り返し、反復ごとに実行の順序を入れ替える(偶数回目は math が先、奇数回目は memory-efficient が先)。
  それぞれ`torch.cuda.synchronize()`で挟んで時間を測る。時間計測とは別の 1 回の呼び出しを profiler で記録し、
  実行された演算の名前が指定したバックエンドのものであることを確かめる(前提条件 P-E2)。


```python
# ---------------- 実験 A・B ----------------
def standard_attention(q, k, v, causal):
    mask = create_causal_mask(q.size(-2), device=q.device) if causal else None
    return scaled_dot_product_attention(q, k, v, mask)[0]


def naive_softmax_nonfinite_row_fraction(q, k, causal) -> float:
    # 最大値を引かない素朴な softmax(FP32)で、非有限値を含む行の割合
    scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(q.size(-1))
    if causal:
        scores = scores.masked_fill(~create_causal_mask(q.size(-2), device=q.device), float("-inf"))
    e = torch.exp(scores)
    p = e / e.sum(dim=-1, keepdim=True)
    return float((~torch.isfinite(p).all(dim=-1)).double().mean())


def run_sample_a(n: int, causal: bool, sigma: float, seed: int) -> dict:
    shape = (BATCH_AB, HEADS_AB, n, HEAD_DIM)
    q, k, v = make_inputs(seed, shape, torch.float32, sigma)
    with torch.no_grad():
        reference = standard_attention(q.double(), k.double(), v.double(), causal)
        err_standard = relative_max_error(standard_attention(q, k, v, causal), reference)
        err_tiled = relative_max_error(
            flash_attention_forward(q, k, v, *BLOCK_AB, causal)[0], reference
        )
        # 診断量: FP16 の入力(参照値は FP16 に丸めた入力の FP64 での計算)
        q16, k16, v16 = (t.half() for t in (q, k, v))
        reference16 = standard_attention(q16.double(), k16.double(), v16.double(), causal)
        err_standard16 = relative_max_error(standard_attention(q16, k16, v16, causal), reference16)
        err_tiled16 = relative_max_error(
            flash_attention_forward(q16, k16, v16, *BLOCK_AB, causal)[0], reference16
        )
        naive_fraction = naive_softmax_nonfinite_row_fraction(q, k, causal)
    return {
        "err_tiled": err_tiled,
        "err_standard": err_standard,
        "err_tiled_fp16": err_tiled16,
        "err_standard_fp16": err_standard16,
        "naive_nonfinite_row_fraction": naive_fraction,
    }


GRADIENT_NAMES = ("dQ", "dK", "dV")


def run_sample_b(n: int, causal: bool, sigma: float, seed: int) -> dict:
    shape = (BATCH_AB, HEADS_AB, n, HEAD_DIM)
    q, k, v, grad_output = make_inputs(seed, shape, torch.float32, sigma, with_grad_output=True)
    q64, k64, v64 = (t.double().requires_grad_() for t in (q, k, v))
    reference = torch.autograd.grad(
        standard_attention(q64, k64, v64, causal), (q64, k64, v64), grad_output.double()
    )
    qs, ks, vs = (t.clone().requires_grad_() for t in (q, k, v))
    standard = torch.autograd.grad(
        standard_attention(qs, ks, vs, causal), (qs, ks, vs), grad_output
    )
    qt, kt, vt = (t.clone().requires_grad_() for t in (q, k, v))
    tiled = torch.autograd.grad(
        FlashAttentionFunction.apply(qt, kt, vt, *BLOCK_AB, causal, None), (qt, kt, vt), grad_output
    )
    return {
        "err_tiled": [relative_max_error(g, r) for g, r in zip(tiled, reference, strict=False)],
        "err_standard": [relative_max_error(g, r) for g, r in zip(standard, reference, strict=False)],
    }


# ---------------- 実験 C ----------------
class LiveTensorBytesTracker(TorchDispatchMode):
    # 演算が作ったテンソルの記憶領域を記録し、生存中のバイト数の合計の最大値(ピーク)を数える。
    # 計測開始前から存在するテンソル(入力)の記憶領域は数えない。記憶領域の解放は StorageWeakRef で検出する。

    def __init__(self, exclude: tuple[torch.Tensor, ...] = ()):
        super().__init__()
        self.excluded = {t.untyped_storage().data_ptr() for t in exclude}
        self.live: dict[int, tuple[StorageWeakRef, int]] = {}
        self.current = 0
        self.peak = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for key, (ref, nbytes) in list(self.live.items()):
            if ref.expired():
                del self.live[key]
                self.current -= nbytes
        for t in tree_flatten(out)[0]:
            if isinstance(t, torch.Tensor):
                storage = t.untyped_storage()
                key = storage.data_ptr()
                if storage.nbytes() > 0 and key not in self.live and key not in self.excluded:
                    self.live[key] = (StorageWeakRef(storage), storage.nbytes())
                    self.current += storage.nbytes()
                    self.peak = max(self.peak, self.current)
        return out


def attention_forward_c(implementation: str, q, k, v) -> torch.Tensor:
    # 実験 C の順伝播。返すのは Attention の出力 O のみ(標準の実装の P、タイリングの L は中間の量として数える)
    if implementation == "standard":
        return scaled_dot_product_attention(q, k, v)[0]
    return flash_attention_forward(q, k, v, *BLOCK_C)[0]


def measure_forward_memory(implementation: str, n: int) -> dict:
    shape = (BATCH_C, HEADS_C, n, HEAD_DIM)
    q, k, v = make_inputs(0, shape, torch.float32)
    record = {"shape": list(shape), "dtype": str(q.dtype), "completed": False}
    try:
        with torch.no_grad():
            if device.type == "cuda":
                sync_device()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                baseline = torch.cuda.memory_allocated(device)
            tracker = LiveTensorBytesTracker(exclude=(q, k, v))
            with tracker:
                output = attention_forward_c(implementation, q, k, v)
            sync_device()
            record["output_bytes"] = output.untyped_storage().nbytes()
            record["tracker_peak_bytes"] = tracker.peak
            if device.type == "cuda":
                record["cuda_peak_bytes"] = torch.cuda.max_memory_allocated(device)
                record["cuda_baseline_bytes"] = baseline
            record["input_bytes"] = sum(t.untyped_storage().nbytes() for t in (q, k, v))
            record["completed"] = True
            del output
    except torch.cuda.OutOfMemoryError:
        record["completed"] = False
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return record


def peak_increment_bytes(record: dict, source: str) -> int | None:
    # ピークメモリの増分。CUDA: max_memory_allocated - 呼び出し直前の memory_allocated(入力を含む) - 出力 O。
    # tracker: 入力を最初から数えないので、ピーク - 出力 O。
    if not record["completed"]:
        return None
    if source == "cuda_allocator":
        return record["cuda_peak_bytes"] - record["cuda_baseline_bytes"] - record["output_bytes"]
    return record["tracker_peak_bytes"] - record["output_bytes"]


# ---------------- 実験 D ----------------
def measure_saved_tensor_bytes(implementation: str, n: int) -> dict:
    shape = (BATCH_D, HEADS_D, n, HEAD_DIM)
    q, k, v = (t.requires_grad_() for t in make_inputs(0, shape, torch.float32))
    storages: dict[int, int] = {}

    def pack(t: torch.Tensor) -> torch.Tensor:
        storage = t.untyped_storage()
        storages[storage.data_ptr()] = max(storages.get(storage.data_ptr(), 0), storage.nbytes())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        if implementation == "standard":
            output = scaled_dot_product_attention(q, k, v)[0]
        else:
            output = FlashAttentionFunction.apply(q, k, v, *BLOCK_D, False, None)
    record = {
        "shape": list(shape),
        "saved_bytes": sum(storages.values()),
        "num_saved_storages": len(storages),
    }
    if device.type == "cuda":  # 診断量: 逆伝播の時間(3 回の中央値)
        grad_output = torch.randn_like(output)
        times = []
        for _ in range(3):
            sync_device()
            t0 = time.perf_counter()
            torch.autograd.grad(output, (q, k, v), grad_output, retain_graph=True)
            sync_device()
            times.append(time.perf_counter() - t0)
        record["backward_seconds_median"] = float(np.median(times))
    del output
    return record


# ---------------- 実験 E ----------------
if device.type == "cuda":
    SDPA_BACKENDS_E = {"math": SDPBackend.MATH, "fused": SDPBackend.EFFICIENT_ATTENTION}
    EXPECTED_SDPA_OPS = {
        "math": "aten::_scaled_dot_product_attention_math",
        "fused": "aten::_scaled_dot_product_efficient_attention",
    }
else:
    # CUDA 以外(ローカルのスモークテスト)ではコード経路の確認のため、CPU で使える flash バックエンドで代替する。
    # 前提条件 P-E1(CUDA であること)が不成立になるので、判定はしない。
    SDPA_BACKENDS_E = {"math": SDPBackend.MATH, "fused": SDPBackend.FLASH_ATTENTION}
    EXPECTED_SDPA_OPS = {
        "math": "aten::_scaled_dot_product_attention_math",
        "fused": "aten::_scaled_dot_product_flash_attention_for_cpu",
    }
DTYPE_E = torch.float16 if device.type == "cuda" else torch.float32  # CPU の代替経路は FP32


def sdpa_call(name: str, q, k, v) -> torch.Tensor:
    with sdpa_kernel(SDPA_BACKENDS_E[name]):
        return F.scaled_dot_product_attention(q, k, v)


def executed_sdpa_ops(name: str, q, k, v) -> list[str]:
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        sdpa_call(name, q, k, v)
        sync_device()
    return sorted(
        {e.key for e in prof.key_averages() if e.key.startswith("aten::_scaled_dot_product")}
    )


def timed_call(fn) -> float:
    sync_device()
    t0 = time.perf_counter()
    fn()
    sync_device()
    return time.perf_counter() - t0


def run_level_e(n: int, repetitions: int) -> dict:
    shape = (BATCH_E, HEADS_E, n, HEAD_DIM)
    q, k, v = make_inputs(0, shape, DTYPE_E)
    ops = {name: executed_sdpa_ops(name, q, k, v) for name in SDPA_BACKENDS_E}
    with torch.no_grad():
        for _ in range(WARMUP_E):
            for name in SDPA_BACKENDS_E:
                sdpa_call(name, q, k, v)
        times = {name: [] for name in SDPA_BACKENDS_E}
        names = list(SDPA_BACKENDS_E)  # ["math", "fused"]
        for repetition in range(repetitions):
            # 直前のカーネルの影響が一方にだけ系統的に乗らないよう、反復ごとに順序を入れ替える
            # (偶数回目は math が先、奇数回目は memory-efficient(fused)が先)
            order = names if repetition % 2 == 0 else names[::-1]
            for name in order:
                times[name].append(timed_call(lambda nm=name: sdpa_call(nm, q, k, v)))
        record = {
            "shape": list(shape),
            "dtype": str(q.dtype),
            "ops": ops,
            "times": times,
            # 診断量: スクラッチ実装(Python のループ)の時間。本来の性能を反映しない
            "scratch_seconds": timed_call(lambda: flash_attention_forward(q, k, v, *BLOCK_E)),
        }
        if device.type == "cuda":  # 診断量: 各バックエンドのピークメモリの増分
            record["cuda_peak_increment_bytes"] = {}
            for name in SDPA_BACKENDS_E:
                sync_device()
                torch.cuda.reset_peak_memory_stats(device)
                baseline = torch.cuda.memory_allocated(device)
                out = sdpa_call(name, q, k, v)
                sync_device()
                record["cuda_peak_increment_bytes"][name] = (
                    torch.cuda.max_memory_allocated(device)
                    - baseline
                    - out.untyped_storage().nbytes()
                )
                del out
    return record


# ハーネスの動作確認(小さな入力)
_probe_a = run_sample_a(80, True, 6.0, 0)
_probe_b = run_sample_b(80, True, 1.0, 0)
_probe_c = {impl: measure_forward_memory(impl, 256) for impl in ("standard", "tiled")}
assert _probe_c["standard"]["tracker_peak_bytes"] - _probe_c["standard"]["output_bytes"] >= (
    estimate_standard_peak_bytes_c(256)
), "tracker が標準の実装の S と P を数えていない"
_probe_d = {impl: measure_saved_tensor_bytes(impl, 256) for impl in ("standard", "tiled")}
assert _probe_d["tiled"]["num_saved_storages"] == 5
_probe_e = run_level_e(64, 2)
assert all(_probe_e["ops"][name] == [EXPECTED_SDPA_OPS[name]] for name in SDPA_BACKENDS_E), (
    _probe_e["ops"]
)
print("ハーネスの動作確認: OK")
print(f"  A: {_probe_a}")
print(f"  B: {_probe_b}")
print(
    f"  C(tracker、N=256): 標準 {peak_increment_bytes(_probe_c['standard'], 'tracker'):,} バイト、"
    f"タイリング {peak_increment_bytes(_probe_c['tiled'], 'tracker'):,} バイト"
)
print(
    f"  D(N=256): 標準 {_probe_d['standard']['saved_bytes']:,} バイト、"
    f"FlashAttentionFunction {_probe_d['tiled']['saved_bytes']:,} バイト"
)
print(f"  E: 実行された演算 {_probe_e['ops']}")
```

    ハーネスの動作確認: OK
      A: {'err_tiled': 4.382202330550769e-06, 'err_standard': 4.369672993286555e-06, 'err_tiled_fp16': 0.00023452719441837227, 'err_standard_fp16': 0.0046326815317523935, 'naive_nonfinite_row_fraction': 0.29375}
      B: {'err_tiled': [2.3566389316651865e-07, 3.5716842250513525e-07, 3.524777679063495e-07], 'err_standard': [2.5343782684073444e-07, 4.2634071710630957e-07, 3.524777679063495e-07]}
      C(tracker、N=256): 標準 2,097,152 バイト、タイリング 1,191,936 バイト
      D(N=256): 標準 1,835,008 バイト、FlashAttentionFunction 1,052,672 バイト
      E: 実行された演算 {'math': ['aten::_scaled_dot_product_attention_math'], 'fused': ['aten::_scaled_dot_product_efficient_attention']}


## 6. 実験 / Experiments



## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/03_efficient_training/014_flash_attention.ipynb
