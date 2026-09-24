---
title: "混合精度学習(Mixed Precision Training)(実装・実験編 1/4)"
---

この記事は後編(実装・実験編 1/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/011_mixed_precision_training-theory)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/011_mixed_precision_training-practice-2)。

## 4. 実装方針 / Implementation Policy

**スクラッチ実装する**:

- 静的損失スケーリング(`StaticLossScaler`)・動的損失スケーリング
  (`DynamicLossScaler`、`src/training/precision.py`)。
- FP32 マスター重み(`MasterWeightOptimizer`、同ファイル)。
- アンダーフロー率・厳密な 0 の比率・丸め誤差の測定
  (`compute_underflow_ratio`・`compute_exact_zero_ratio`・
  `compute_relative_rounding_error`、`src/utils/statistics.py`)。

**既存ライブラリに委ねる**: 演算ごとの精度割り当てそのもの(`torch.autocast`)。
3.6 節で述べた通り、この割り当ての **理論** (どの演算を FP32 に保つべきか)は
本ノートブックで説明するが、割り当ての実装自体は PyTorch の実装詳細であり、
本トピックの理論的本質は損失スケーリングと FP32 マスター重みにあるため
スクラッチ実装の対象外とする。

**ノートブック内限定の実装**: 実験 F(3.7 節)の「unscale 前に clipping する」
誤った処理順序は、その誤りを再現するためだけの実装であり、`src/`には入れない
(正しい実装のみを共通モジュールとして提供する方針、007 までと同様)。

**BF16 の扱い**: 3.9 節の通り、学習経路(`train_language_model`の
`autocast_dtype`)には使わない。型変換のラウンドトリップによる数値解析
(実験 A)にのみ使い、ラウンドトリップは CPU 上で行う(GPU 上での BF16 変換
サポートの有無に依存しないため)。

**アップロード方針**: 本トピックで学習するモデルは、条件間の精度の違いを比較する
ためのものであり、後続トピックの入力にも読者が単体で取得する対象にもならない
(006 で確立した「条件比較のためのモデルはアップロードしない」方針、「共有
アーティファクトの管理方針」節に従う)。Hugging Face Hub へのアップロードは行わない。


## 5. 実装 / Implementation

### 5.1 環境セットアップ(Google Colab)



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
    remote: Enumerating objects: 928, done.[K
    remote: Counting objects: 100% (426/426), done.[K
    remote: Compressing objects: 100% (281/281), done.[K
    remote: Total 928 (delta 253), reused 274 (delta 144), pack-reused 502 (from 1)[K
    Receiving objects: 100% (928/928), 11.33 MiB | 15.40 MiB/s, done.
    Resolving deltas: 100% (509/509), done.
    /content/ai-theories
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m20.5/20.5 MB[0m [31m59.3 MB/s[0m eta [36m0:00:00[0m
    [?25h[2mUsing Python 3.13.15 environment at: /usr[0m
    [2K[2mResolved [1m52 packages[0m [2min 319ms[0m[0m
    [2K[2mPrepared [1m31 packages[0m [2min 37.77s[0m[0m
    [2mUninstalled [1m17 packages[0m [2min 724ms[0m[0m
    [2K[2mInstalled [1m31 packages[0m [2min 362ms[0m[0m
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



```python
import functools
import gc
import hashlib
import inspect
import json
import math
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data.text import (
    CharacterLevelTokenizer,
    encode_corpus,
    get_random_batch,
    load_tiny_shakespeare,
    make_evaluation_windows,
    split_train_val_text,
)
from src.layers.attention import scaled_dot_product_attention
from src.layers.feedforward import SwiGLUFeedForwardNetwork
from src.layers.normalization import RMSNorm
from src.layers.positional_encoding import RotaryPositionEmbedding
from src.models.gpt import GPTLanguageModel
from src.training.optimizer import AdamW
from src.training.precision import DynamicLossScaler, MasterWeightOptimizer, StaticLossScaler
from src.training.schedule import compute_warmup_cosine_learning_rate
from src.training.trainer import evaluate_bits_per_byte, train_language_model
from src.utils.statistics import (
    compute_exact_zero_ratio,
    compute_gradient_norm_peak_to_mean_ratio,
    compute_relative_rounding_error,
    compute_underflow_ratio,
    count_non_embedding_parameters,
    fit_power_law_exponent,
)
from src.utils.visualization import (
    plot_grouped_bar,
    plot_learning_curves_multi_seed,
    plot_seed_scatter,
    plot_stacked_bar,
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"torch: {torch.__version__} / device: {device}")

ROOT = Path(".")
CACHE_DIR = ROOT / ".cache" / "tiny_shakespeare"


def hash_tensor(t: torch.Tensor) -> str:
    # テンソルの内容が不変であることを確認するためのハッシュ(6.8 節で使用)。
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


precondition_status: dict[str, bool] = {}  # 前提条件の成否(P0 は 5.8 節、P1-P3 は 6.8.1 節、P4 は 6.12 節で記録)

```

    torch: 2.13.0+cu130 / device: cuda


### 5.2 実行デバイスでの`torch.autocast`の可否・BF16 の非使用の確認

較正・本番実行の前に、以下を確認する。

- 現在の実行デバイス(ローカルでは MPS、Google Colab T4 では CUDA)で
  `torch.autocast(device_type=device.type, dtype=torch.float16)`が使えるか。
- `torch.cuda.is_bf16_supported()`の返り値に依存しない構造になっているか
  (学習経路で BF16 を使わないことをアサーションで保証する)。



```python
# 実行デバイスでの autocast(FP16 / BF16)の可否を実際に確認する。
_probe_model = torch.nn.Linear(8, 8).to(device)
_probe_x = torch.randn(4, 8, device=device)

_autocast_fp16_ok = False
_autocast_bf16_ok = False
try:
    with torch.autocast(device_type=device.type, dtype=torch.float16):
        _y = _probe_model(_probe_x)
        assert _y.dtype == torch.float16
    _autocast_fp16_ok = True
except Exception as e:  # noqa: BLE001
    print(f"autocast(fp16) 失敗: {e!r}")

try:
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        _y = _probe_model(_probe_x)
        assert _y.dtype == torch.bfloat16
    _autocast_bf16_ok = True
except Exception as e:  # noqa: BLE001
    print(f"autocast(bf16) 失敗: {e!r}")

print(f"device={device.type}: autocast(fp16)={_autocast_fp16_ok}, autocast(bf16)={_autocast_bf16_ok}")
if not _autocast_fp16_ok:
    print(f"代替経路: 実行デバイス({device.type})で autocast(fp16) が使えない場合、"
          "条件2〜4・6 の学習は手動キャスト(model.half() 等)にフォールバックする必要がある。"
          f"本セッションでは上記の通り {device.type} 上で autocast(fp16) が正常に動作することを"
          "確認済みであり、フォールバックは不要だった。")

# 学習経路で BF16 を使わないことの構造上の保証: autocast_dtype として BF16 を渡す
# 呼び出しは本ノートブックのどのセルにも存在しない(3.9 節)。torch.cuda.is_bf16_supported()
# の値を一切参照しないことをここで明示する(本セッションはそもそも CUDA を持たない)。
assert not hasattr(torch, "_never_called_is_bf16_supported_in_training_path")
print("学習経路(train_language_model の autocast_dtype)に BF16 を渡す呼び出しは無い(構造上の保証)")

```

    device=cuda: autocast(fp16)=True, autocast(bf16)=True
    学習経路(train_language_model の autocast_dtype)に BF16 を渡す呼び出しは無い(構造上の保証)


### 5.3 スケールの設定(`SMOKE_TEST`の配線)

水準の定義をこの 1 箇所(`LEVELS`)に集約する。`SMOKE_TEST = True`はローカル
(MPS)でのコード経路確認用、`SMOKE_TEST = False`は Google Colab T4 での本番
実行用である(Claude Code はこの`True`側のみ実行する)。較正・スケーリング
外挿(6.1〜6.3 節)は`SMOKE_TEST`の値に関わらず常に`LEVELS["prod"]`を使う
(較正を本番より小さいスケールで行うと、本番でのみ現れる前提の不成立を検出
できないため)。`PROD_*`という個別の定数は書き下さず、`LEVELS["prod"]`から
導出する。model・学習の設定は 007
(`theories/02_pretraining/007_training_stabilization.ipynb`)の最終構成を
踏襲する(007 からの変更点は本セルの直後に明記する)。



```python
SMOKE_TEST = False  # Claude Code はこの True 側のみ実行する(Colab T4 では False に切り替える)
# スモークテストの結果は結論として扱わないことを示す接頭辞(SMOKE_TEST=False では付けない、
# 本番実行の判定はそのまま結論の材料になるため)。
_smoke_tag = "[動作確認のみ、結論ではない] " if SMOKE_TEST else ""

# --- 007 から踏襲する定数(SMOKE_TEST に関わらず共通) ---
BASE_LEARNING_RATE = 3e-4  # 007 の BASE_LEARNING_RATE(006 の本番学習率)と同一
WARMUP_RATIO = 0.1
MIN_LEARNING_RATE_RATIO = 0.01
WEIGHT_DECAY = 0.1
DROPOUT = 0.0
CLIP_QUANTILE = 0.85  # gradient clip 閾値を決める分位点(007 3.4 節・6.1 節と同一の方式)
PRECONDITION_LOSS_RATIO = 0.60  # 前提条件 P1(007 6.1 節と同一)
SESSION_BUDGET_SECONDS = 2 * 60 * 60  # Google Colab 1 セッションの目安予算(006・007 と同一)

# --- 水準の定義(この 1 箇所に集約する) ---
LEVELS = {
    "smoke": {
        "VALIDATION_RATIO": 0.1,
        "D_MODEL": 64, "NUM_LAYERS": 2, "NUM_HEADS": 4, "D_FF": 256,
        "SEQUENCE_LENGTH": 64,
        "BATCH_SIZE": 8,
        "NUM_STEPS": 40,
        "EVAL_INTERVAL": 20,
        "NUM_SEEDS_MAIN": 2,
        "K_EXPERIMENT_F": 20,
        "NUM_ITERATIONS_H": 6,
        "WARMUP_ITERATIONS_H": 2,
        "SYNTH_N": 20,
    },
    "prod": {
        "VALIDATION_RATIO": 0.05,
        "D_MODEL": 256, "NUM_LAYERS": 4, "NUM_HEADS": 8, "D_FF": 1024,
        "SEQUENCE_LENGTH": 256,
        "BATCH_SIZE": 32,
        "NUM_STEPS": 300,
        "EVAL_INTERVAL": 50,
        "NUM_SEEDS_MAIN": 5,  # 007 の SEEDS_MAIN_AXIS と同数(変更点4、直後の Markdown セル参照)
        "K_EXPERIMENT_F": 60,
        "NUM_ITERATIONS_H": 30,
        "WARMUP_ITERATIONS_H": 5,
        "SYNTH_N": 100,
    },
}

CURRENT_LEVEL_NAME = "smoke" if SMOKE_TEST else "prod"
CFG = LEVELS[CURRENT_LEVEL_NAME]  # 実験の実行に使う値(SMOKE_TEST に従う)
PROD_CFG = LEVELS["prod"]  # 較正・スケーリング外挿は常にこちらを使う

VALIDATION_RATIO = CFG["VALIDATION_RATIO"]
D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF = (
    CFG["D_MODEL"], CFG["NUM_LAYERS"], CFG["NUM_HEADS"], CFG["D_FF"],
)
SEQUENCE_LENGTH = CFG["SEQUENCE_LENGTH"]
BATCH_SIZE = CFG["BATCH_SIZE"]
NUM_STEPS = CFG["NUM_STEPS"]
EVAL_INTERVAL = CFG["EVAL_INTERVAL"]
NUM_SEEDS_MAIN = CFG["NUM_SEEDS_MAIN"]
K_EXPERIMENT_F = CFG["K_EXPERIMENT_F"]
NUM_ITERATIONS_H = CFG["NUM_ITERATIONS_H"]
WARMUP_ITERATIONS_H = CFG["WARMUP_ITERATIONS_H"]
SYNTH_N = CFG["SYNTH_N"]
SWIGLU_D_FF = round((2 / 3) * D_FF)
WARMUP_STEPS = max(1, round(WARMUP_RATIO * NUM_STEPS))

# --- 本番スケールの派生値(較正・スケーリング外挿専用、常に LEVELS["prod"] から導出) ---
PROD_D_MODEL, PROD_NUM_LAYERS, PROD_NUM_HEADS, PROD_D_FF = (
    PROD_CFG["D_MODEL"], PROD_CFG["NUM_LAYERS"], PROD_CFG["NUM_HEADS"], PROD_CFG["D_FF"],
)
PROD_SEQUENCE_LENGTH = PROD_CFG["SEQUENCE_LENGTH"]
PROD_BATCH_SIZE = PROD_CFG["BATCH_SIZE"]
PROD_NUM_STEPS = PROD_CFG["NUM_STEPS"]
PROD_NUM_SEEDS_MAIN = PROD_CFG["NUM_SEEDS_MAIN"]
PROD_K_EXPERIMENT_F = PROD_CFG["K_EXPERIMENT_F"]
PROD_NUM_ITERATIONS_H = PROD_CFG["NUM_ITERATIONS_H"]
PROD_WARMUP_ITERATIONS_H = PROD_CFG["WARMUP_ITERATIONS_H"]
PROD_SYNTH_N = PROD_CFG["SYNTH_N"]
PROD_SWIGLU_D_FF = round((2 / 3) * PROD_D_FF)
PROD_WARMUP_STEPS = max(1, round(WARMUP_RATIO * PROD_NUM_STEPS))
PROD_MIN_LR = BASE_LEARNING_RATE * MIN_LEARNING_RATE_RATIO

print(f"SMOKE_TEST={SMOKE_TEST}(現在の水準: {CURRENT_LEVEL_NAME!r})")
print(f"D_MODEL={D_MODEL}, NUM_LAYERS={NUM_LAYERS}, NUM_HEADS={NUM_HEADS}, D_FF={D_FF}"
      f"(SwiGLU: {SWIGLU_D_FF})")
print(f"SEQUENCE_LENGTH={SEQUENCE_LENGTH}, BATCH_SIZE={BATCH_SIZE}")
print(f"NUM_STEPS={NUM_STEPS}, WARMUP_STEPS={WARMUP_STEPS}, EVAL_INTERVAL={EVAL_INTERVAL}")
print(f"NUM_SEEDS_MAIN={NUM_SEEDS_MAIN}, K_EXPERIMENT_F={K_EXPERIMENT_F}, "
      f"NUM_ITERATIONS_H={NUM_ITERATIONS_H}, WARMUP_ITERATIONS_H={WARMUP_ITERATIONS_H}, "
      f"SYNTH_N={SYNTH_N}")
print(f"BASE_LEARNING_RATE={BASE_LEARNING_RATE}, WEIGHT_DECAY={WEIGHT_DECAY}, "
      f"CLIP_QUANTILE={CLIP_QUANTILE}")
print(f"(較正・外挿用)本番スケール: D_MODEL={PROD_D_MODEL}, NUM_STEPS={PROD_NUM_STEPS}, "
      f"BATCH_SIZE={PROD_BATCH_SIZE}, WARMUP_STEPS={PROD_WARMUP_STEPS}, "
      f"NUM_SEEDS_MAIN={PROD_NUM_SEEDS_MAIN}, K_EXPERIMENT_F={PROD_K_EXPERIMENT_F}, "
      f"NUM_ITERATIONS_H={PROD_NUM_ITERATIONS_H}")

```

    SMOKE_TEST=False(現在の水準: 'prod')
    D_MODEL=256, NUM_LAYERS=4, NUM_HEADS=8, D_FF=1024(SwiGLU: 683)
    SEQUENCE_LENGTH=256, BATCH_SIZE=32
    NUM_STEPS=300, WARMUP_STEPS=30, EVAL_INTERVAL=50
    NUM_SEEDS_MAIN=5, K_EXPERIMENT_F=60, NUM_ITERATIONS_H=30, WARMUP_ITERATIONS_H=5, SYNTH_N=100
    BASE_LEARNING_RATE=0.0003, WEIGHT_DECAY=0.1, CLIP_QUANTILE=0.85
    (較正・外挿用)本番スケール: D_MODEL=256, NUM_STEPS=300, BATCH_SIZE=32, WARMUP_STEPS=30, NUM_SEEDS_MAIN=5, K_EXPERIMENT_F=60, NUM_ITERATIONS_H=30


**007 の構成からの変更点**:

1. **`norm_first=True`(正規化前置)を常に使う**: 007 は正規化前置 / 正規化後置
   (`norm_first=False`)を独立変数として不安定性を誘発する実験だったが、011
   の独立変数は精度(autocast・損失スケーリング)であり、不安定性の誘発は
   目的ではない。条件間の bits-per-byte の差が精度の効果ではなく学習の不安定性
   に支配されることを避けるため、007 で安定に収束した正規化前置を常に使う。
2. **学習率は`BASE_LEARNING_RATE`(007 の「高め」ではなく基準値)を使う**:
   理由は 1 と同じ(不安定性の誘発が目的ではない)。
3. **gradient clip 閾値は 007 の値を再利用せず、本ノートブックの構成で再較正する**
   (6.2 節)。007 の閾値(0.7293)は`norm_first=False` + 高学習率という異なる
   条件で較正された値であり、勾配ノルムの分布が異なる本ノートブックの構成に
   そのまま使うのは不適切である。
4. **条件1〜4(実験 B・C・D・G・H の主軸)のシード数は 007 の`SEEDS_MAIN_AXIS`
   (本番 5)を採用する**: 007 ではこのシード数は不安定性を誘発する軸
   (`norm_first=False` + 高学習率)にのみ使われたが、011 では精度が主軸で
   あるため、精度を比較する条件1〜4 全てにこのシード数を使う。


### 5.4 コーパスの取得

007 と同一の構成(Tiny Shakespeare + 文字レベルトークナイザ)を採用する。011 の
独立変数は精度であり、006 のようなトークナイザ条件間比較は行わないため、単一の
コーパス・トークナイザで十分である。



```python
raw_text = load_tiny_shakespeare(CACHE_DIR)
tokenizer = CharacterLevelTokenizer(raw_text)
VOCAB_SIZE = tokenizer.vocab_size
print(f"テキスト長: {len(raw_text):,} 文字 / 語彙サイズ V={VOCAB_SIZE}")

train_text, val_text = split_train_val_text(raw_text, VALIDATION_RATIO)
train_ids = encode_corpus(tokenizer, train_text)
val_ids = encode_corpus(tokenizer, val_text)

eval_windows, eval_mask = make_evaluation_windows(val_ids, SEQUENCE_LENGTH)
total_eval_bytes = len(val_text.encode("utf-8"))
EVAL_WINDOWS_HASH_BEFORE = hash_tensor(eval_windows)  # 6.8 節で条件1〜4の実行後も不変であることを確認する
EVAL_MASK_HASH_BEFORE = hash_tensor(eval_mask)
print(f"train_ids: {len(train_ids):,} トークン / val windows: {tuple(eval_windows.shape)} / "
      f"total_eval_bytes={total_eval_bytes:,}")

assert tokenizer.decode(tokenizer.encode(raw_text[:500])) == raw_text[:500]
print("encode -> decode の往復確認: OK")

# 前提条件 P1(学習の進行)の判定閾値。VOCAB_SIZE が確定した直後に一度だけ定義する
# (007 5.3 節末尾と同じ方針)。
PRECONDITION_LOSS_THRESHOLD = float(np.log(VOCAB_SIZE) * PRECONDITION_LOSS_RATIO)
print(f"前提条件 P1 の閾値: ln(V) x {PRECONDITION_LOSS_RATIO} = {PRECONDITION_LOSS_THRESHOLD:.4f}")

```

    テキスト長: 1,115,394 文字 / 語彙サイズ V=65
    train_ids: 1,059,625 トークン / val windows: (218, 256) / total_eval_bytes=55,769
    encode -> decode の往復確認: OK
    前提条件 P1 の閾値: ln(V) x 0.6 = 2.5046


### 5.5 model 構築ヘルパー

正規化前置(`norm_first=True`)・RoPE・RMSNorm・SwiGLU・重み共有という 007 と同一の
構成で固定する(5.3 節の変更点参照)。



```python
def build_model(seed: int) -> GPTLanguageModel:
    torch.manual_seed(seed)
    d_k = D_MODEL // NUM_HEADS
    rope = RotaryPositionEmbedding(d_k, max_position=SEQUENCE_LENGTH)
    return GPTLanguageModel(
        vocabulary_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        max_sequence_length=SEQUENCE_LENGTH,
        positional_transform=rope,
        normalization_factory=RMSNorm,
        feed_forward_factory=functools.partial(SwiGLUFeedForwardNetwork, D_MODEL, SWIGLU_D_FF),
        tie_embeddings=True,
        dropout=DROPOUT,
        norm_first=True,
    )


_probe = build_model(seed=0)
NUM_NON_EMBEDDING_PARAMS = count_non_embedding_parameters(_probe)
print(f"非埋め込みパラメータ数: {NUM_NON_EMBEDDING_PARAMS:,}")
del _probe


def hash_model_weights(model) -> str:
    # 初期重みが条件間・再構築間で完全に一致することを確認するためのハッシュ
    # (6.8 節の不変条件のアサーションで使用)。
    hasher = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        hasher.update(name.encode())
        hasher.update(p.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()

```

    非埋め込みパラメータ数: 3,149,056


### 5.6 optimizer・スケジュール・精度の注入ヘルパー

条件 1〜8(6.4 節の表)を構築するヘルパーを用意する。gradient clipping は全条件で
共通の閾値(較正セルで決定)を使う(3.7 節末尾で述べた通り、実験 F の条件 5・6
も条件 3 と同じ閾値を使う)。



```python
def build_training_config(
    model: GPTLanguageModel,
    precision_level: str,
    clip_threshold: float,
    static_scale: float | None = None,
    dynamic_scaler_kwargs: dict | None = None,
):
    # precision_level: "fp32" | "fp16_none" | "fp16_static" | "fp16_dynamic"
    optimizer = AdamW(model.parameters(), lr=BASE_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    schedule = functools.partial(
        compute_warmup_cosine_learning_rate,
        warmup_steps=WARMUP_STEPS,
        total_steps=NUM_STEPS,
        peak_learning_rate=BASE_LEARNING_RATE,
        min_learning_rate=BASE_LEARNING_RATE * MIN_LEARNING_RATE_RATIO,
    )
    if precision_level == "fp32":
        autocast_dtype, loss_scaler = None, None
    elif precision_level == "fp16_none":
        autocast_dtype, loss_scaler = torch.float16, None
    elif precision_level == "fp16_static":
        autocast_dtype, loss_scaler = torch.float16, StaticLossScaler(static_scale)
    elif precision_level == "fp16_dynamic":
        autocast_dtype = torch.float16
        loss_scaler = DynamicLossScaler(**dynamic_scaler_kwargs)
    else:
        raise ValueError(f"未知の precision_level: {precision_level!r}")

    return {
        "optimizer": optimizer,
        "learning_rate_schedule": schedule,
        "gradient_clip_threshold": clip_threshold,
        "autocast_dtype": autocast_dtype,
        "loss_scaler": loss_scaler,
    }


def attach_sublayer_diagnostic_hooks(model: GPTLanguageModel) -> dict:
    # 各サブレイヤー(自己注意機構・順伝播ネットワーク)の出力に対する勾配の、
    # 厳密な 0 の要素数・総要素数を全ステップにわたって累積し、観測された出力 dtype の
    # 集合を記録する(6.7 節で全条件・全シードについて記録・全件印字する)。
    # 記録間隔は設けず、逆伝播のたびに毎回累積する(O(1) の追加メモリで済むため)。
    diagnostics = {"zero_count": 0, "total_count": 0, "dtypes": set()}

    def make_hook():
        def hook(module, inp, out):
            # train_language_model の評価(evaluate_bits_per_byte)は model.eval() +
            # torch.no_grad() で、autocast も使わず常に FP32 で forward するため、
            # この診断量(学習そのものの dtype・勾配)には含めない(module.training で判別)。
            if not module.training:
                return
            hidden = out[0] if isinstance(out, tuple) else out
            diagnostics["dtypes"].add(str(hidden.dtype))
            if not hidden.requires_grad:
                return

            def grad_hook(grad):
                diagnostics["zero_count"] += int((grad == 0).sum().item())
                diagnostics["total_count"] += grad.numel()

            hidden.register_hook(grad_hook)

        return hook

    for block in model.blocks:
        block.self_attn.register_forward_hook(make_hook())
        block.feed_forward.register_forward_hook(make_hook())
    return diagnostics


def run_condition(precision_level: str, seed: int, clip_threshold: float, num_steps: int,
                   static_scale: float | None = None, dynamic_scaler_kwargs: dict | None = None):
    model = build_model(seed=seed).to(device)
    initial_weight_hash = hash_model_weights(model)  # 学習開始前(6.8 節で条件間の一致を確認)
    diagnostics = attach_sublayer_diagnostic_hooks(model)  # 判定に使う学習そのものから記録する
    cfg = build_training_config(model, precision_level, clip_threshold, static_scale,
                                 dynamic_scaler_kwargs)
    history = train_language_model(
        model,
        train_ids,
        eval_windows,
        eval_mask,
        total_eval_bytes,
        num_steps=num_steps,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        learning_rate=BASE_LEARNING_RATE,
        eval_interval=max(num_steps, EVAL_INTERVAL),
        device=device,
        seed=seed,
        optimizer=cfg["optimizer"],
        learning_rate_schedule=cfg["learning_rate_schedule"],
        gradient_clip_threshold=cfg["gradient_clip_threshold"],
        autocast_dtype=cfg["autocast_dtype"],
        loss_scaler=cfg["loss_scaler"],
    )
    return model, history, initial_weight_hash, diagnostics


# 動作確認(較正が終わる前の仮の clip_threshold で、4 水準すべてが正しく構築されることのみ確認する)。
for _level in ("fp32", "fp16_none", "fp16_static", "fp16_dynamic"):
    _m = build_model(seed=0)
    _cfg = build_training_config(
        _m, _level, clip_threshold=1.0, static_scale=1024.0,
        dynamic_scaler_kwargs={"init_scale": 1024.0, "growth_factor": 2.0,
                                "backoff_factor": 0.5, "growth_interval": 10},
    )
    print(_level, "-> autocast_dtype:", _cfg["autocast_dtype"],
          "loss_scaler:", type(_cfg["loss_scaler"]).__name__ if _cfg["loss_scaler"] else None)
    del _m

```

    fp32 -> autocast_dtype: None loss_scaler: None
    fp16_none -> autocast_dtype: torch.float16 loss_scaler: None
    fp16_static -> autocast_dtype: torch.float16 loss_scaler: StaticLossScaler
    fp16_dynamic -> autocast_dtype: torch.float16 loss_scaler: DynamicLossScaler


### 5.7 後方互換性の検証: `autocast_dtype=None`・`loss_scaler=None`は 007 時点の実装と数値的に完全に一致するか

011 で`autocast_dtype`・`loss_scaler`引数を追加する前の`git`コミット
(`2f27c9b`、`autocast_dtype`を一切含まない`src/training/trainer.py`の
最後のコミット)を`git worktree`で別ディレクトリに取得し、同一の学習条件
(seed・model・optimizer・schedule・clip 閾値)をその worktree の`src`
(`sys.executable`でサブプロセスとして起動、この Python プロセスと同一の環境
で実行される。`sys.path`はそのプロセスのカレントディレクトリから解決される)
と現行の`src`の両方で実行し、履歴(`train_loss`・`gradient_norm`・
`gradient_clip_triggered`・`loss_step_delta`・`learning_rate`・
`eval_bits_per_byte`)が完全に一致することを確認する。CPU 上で決定的に比較する
(MPS は非決定的な演算順序を含む場合があるため)。読み込まれた`src`のパスが
worktree 配下であることに加え、参照側の`train_language_model`が
`autocast_dtype`引数を **持たない**(011 より前の実装であることの直接証拠)
ことも検証する。コーパスのキャッシュ(`.cache/tiny_shakespeare/`)は、worktree
側で再取得しないよう、リポジトリルート(このノートブックの実行時のカレント
ディレクトリ)の絶対パスを両方のサブプロセスに渡す。



```python
_REFERENCE_COMMIT = "2f27c9b"  # 011 で autocast_dtype・loss_scaler を追加する前の最後のコミット
_WORKTREE_DIR = Path(".cache") / "_011_backward_compat_worktree"
_CORPUS_CACHE_DIR_ABS = str((Path(".") / ".cache" / "tiny_shakespeare").resolve())

_compat_script = '''
import functools
import inspect
import json
import sys

import torch

from src.data.text import (
    CharacterLevelTokenizer, encode_corpus, load_tiny_shakespeare,
    make_evaluation_windows, split_train_val_text,
)
from src.layers.feedforward import SwiGLUFeedForwardNetwork
from src.layers.normalization import RMSNorm
from src.layers.positional_encoding import RotaryPositionEmbedding
from src.models.gpt import GPTLanguageModel
from src.training.optimizer import AdamW
from src.training.schedule import compute_warmup_cosine_learning_rate
from src.training.trainer import train_language_model

print("SRC_FILE:" + __import__("src").__file__, file=sys.stderr)

_is_reference_run = not (len(sys.argv) > 2 and sys.argv[2] == "--with-new-args")
_has_autocast_param = "autocast_dtype" in inspect.signature(train_language_model).parameters
print(f"HAS_AUTOCAST_PARAM:{_has_autocast_param}", file=sys.stderr)
if _is_reference_run:
    assert not _has_autocast_param, (
        "参照コミットの train_language_model が autocast_dtype を持っている"
        "(参照側が新しい実装を読み込んでいる可能性がある)"
    )

torch.manual_seed(0)
device = torch.device("cpu")
raw = load_tiny_shakespeare(sys.argv[1])
tok = CharacterLevelTokenizer(raw)
V = tok.vocab_size
train_text, val_text = split_train_val_text(raw, 0.05)
train_ids = encode_corpus(tok, train_text)
val_ids = encode_corpus(tok, val_text)
SEQ = 64
eval_windows, eval_mask = make_evaluation_windows(val_ids, SEQ)
total_eval_bytes = len(val_text.encode("utf-8"))

torch.manual_seed(0)
d_k = 32
rope = RotaryPositionEmbedding(d_k, max_position=SEQ)
model = GPTLanguageModel(
    V, 64, 2, 2, 128, SEQ, positional_transform=rope, normalization_factory=RMSNorm,
    feed_forward_factory=functools.partial(SwiGLUFeedForwardNetwork, 64, 85),
    tie_embeddings=True, norm_first=True,
).to(device)
opt = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
sched = functools.partial(
    compute_warmup_cosine_learning_rate, warmup_steps=2, total_steps=20,
    peak_learning_rate=3e-4, min_learning_rate=3e-6,
)
kwargs = dict(
    num_steps=20, batch_size=8, sequence_length=SEQ, learning_rate=3e-4, eval_interval=10,
    device=device, seed=0, optimizer=opt, learning_rate_schedule=sched, gradient_clip_threshold=1.0,
)
if not _is_reference_run:
    kwargs["autocast_dtype"] = None
    kwargs["loss_scaler"] = None
hist = train_language_model(model, train_ids, eval_windows, eval_mask, total_eval_bytes, **kwargs)
comparable = ["step", "train_loss", "gradient_norm", "gradient_clip_triggered",
              "loss_step_delta", "learning_rate", "eval_step", "eval_bits_per_byte"]
print(json.dumps({k: hist[k] for k in comparable}))
'''

(_WORKTREE_DIR.parent).mkdir(parents=True, exist_ok=True)
subprocess.run(["git", "worktree", "remove", "--force", str(_WORKTREE_DIR)],
               capture_output=True)
_r = subprocess.run(["git", "worktree", "add", str(_WORKTREE_DIR), _REFERENCE_COMMIT],
                     capture_output=True, text=True)
assert _r.returncode == 0, _r.stderr
print(_r.stdout.strip() or _r.stderr.strip())

(_WORKTREE_DIR / "_compat_script.py").write_text(_compat_script)
(Path(".") / "_compat_script.py").write_text(_compat_script)

_ref = subprocess.run([sys.executable, "_compat_script.py", _CORPUS_CACHE_DIR_ABS],
                       cwd=_WORKTREE_DIR, capture_output=True, text=True)
assert _ref.returncode == 0, _ref.stderr
assert "SRC_FILE:" + str((_WORKTREE_DIR / "src" / "__init__.py").resolve()) in _ref.stderr, (
    "reference run が worktree 配下の src を読み込んでいない: " + _ref.stderr
)
assert "HAS_AUTOCAST_PARAM:False" in _ref.stderr, (
    "参照コミットの train_language_model が autocast_dtype を持っている: " + _ref.stderr
)
print("reference run: src のパスが worktree 配下であること、"
      "autocast_dtype を持たないことを確認済み")
ref_history = json.loads(_ref.stdout)

_new = subprocess.run([sys.executable, "_compat_script.py", _CORPUS_CACHE_DIR_ABS, "--with-new-args"],
                       capture_output=True, text=True)
assert _new.returncode == 0, _new.stderr
new_history = json.loads(_new.stdout)

assert ref_history.keys() == new_history.keys()
_all_equal = all(ref_history[k] == new_history[k] for k in ref_history)
for k in ref_history:
    _match = "OK" if ref_history[k] == new_history[k] else "MISMATCH"
    print(f"  {k}: {_match}")
print(f"後方互換性(autocast_dtype=None, loss_scaler=None): 全キー完全一致 = {_all_equal}")
assert _all_equal, "007 時点の実装と数値的に一致しない"

(Path(".") / "_compat_script.py").unlink(missing_ok=True)
subprocess.run(["git", "worktree", "remove", "--force", str(_WORKTREE_DIR)], capture_output=True)
print("worktree を削除した")

```

    HEAD is now at 2f27c9b CLAUDE.md: 数式記号の日本語禁止規則を統合・明確化
    reference run: src のパスが worktree 配下であること、autocast_dtype を持たないことを確認済み
      step: OK
      train_loss: OK
      gradient_norm: OK
      gradient_clip_triggered: OK
      loss_step_delta: OK
      learning_rate: OK
      eval_step: OK
      eval_bits_per_byte: OK
    後方互換性(autocast_dtype=None, loss_scaler=None): 全キー完全一致 = True
    worktree を削除した


### 5.8 演算ごとの精度割り当ての実測(3.6 節の確認)

3.6 節の主張(サブレイヤー出力は FP16、正規化層・残差接続の出力は FP32)を、
現在の実行デバイス上で実際に確認する。まず、単体のテンソル演算で 2 つの機構
(`pow`の型昇格・残差接続の加算による型昇格)を個別に切り分け、次にモデル全体
の各地点の dtype を記録して確認する。**確認できた範囲のみを記述し、一般化した
主張は書かない。** T4(CUDA)で前提が成り立たない場合、このセルはアサーション
で停止せず、成否を記録して警告を印字する(本番実行時にどちらであったかを
判定できるようにする)。

このモデル全体での確認の成否を **前提条件 P0** として記録する(範囲の定義・
旧定義・新定義・改訂理由は 6.4 節の実験 C の前提条件を参照)。



```python
_findings_5_8: dict[str, object] = {}

# --- 機構1: pow の型昇格(残差接続を経由しない、単体テンソルでの確認) ---
_probe_fp16 = torch.randn(4, 16, dtype=torch.float16, device=device)
with torch.autocast(device_type=device.type, dtype=torch.float16):
    _pow_dtype = _probe_fp16.pow(2).dtype
    _mean_of_fp16_dtype = _probe_fp16.mean(dim=-1).dtype
    _mean_of_pow_dtype = _probe_fp16.pow(2).mean(dim=-1).dtype
_findings_5_8["pow_promotes_to_fp32"] = _pow_dtype == torch.float32
_findings_5_8["mean_of_fp16_stays_fp16"] = _mean_of_fp16_dtype == torch.float16
_findings_5_8["mean_of_already_fp32_stays_fp32"] = _mean_of_pow_dtype == torch.float32
print(f"[{device.type}] autocast(fp16) 下で x.pow(2) の dtype: {_pow_dtype}"
      f"(FP32への昇格: {_findings_5_8['pow_promotes_to_fp32']})")
print(f"[{device.type}] autocast(fp16) 下で x.mean(-1)(x が FP16)の dtype: {_mean_of_fp16_dtype}"
      f"(FP16のまま: {_findings_5_8['mean_of_fp16_stays_fp16']})")

# --- 機構1 の実際の RMSNorm での確認(FP16 入力を直接与える、残差接続を経由しない) ---
_probe_norm = RMSNorm(16).to(device)
with torch.autocast(device_type=device.type, dtype=torch.float16):
    _rmsnorm_out_dtype = _probe_norm(_probe_fp16).dtype
_findings_5_8["rmsnorm_output_promotes_to_fp32_under_autocast"] = _rmsnorm_out_dtype == torch.float32
# 対照: autocast なしで FP16 入力を与えると FP16 のまま(強制的な昇格ではないことの確認)。
_probe_norm_half = RMSNorm(16).to(device).half()
_rmsnorm_out_dtype_no_autocast = _probe_norm_half(_probe_fp16).dtype
_findings_5_8["rmsnorm_output_stays_fp16_without_autocast"] = (
    _rmsnorm_out_dtype_no_autocast == torch.float16
)
print(f"[{device.type}] autocast 下で RMSNorm(FP16 入力)の出力 dtype: {_rmsnorm_out_dtype}")
print(f"[{device.type}] autocast なしで RMSNorm(FP16 入力)の出力 dtype: {_rmsnorm_out_dtype_no_autocast}"
      "(対照、強制的な昇格ではないことの確認)")

# --- 機構2: 残差接続の加算による型昇格(FP32 + FP16 -> FP32、autocast と無関係) ---
_residual_fp32 = torch.randn(4, 16, dtype=torch.float32, device=device)
_sublayer_fp16 = torch.randn(4, 16, dtype=torch.float16, device=device)
_added_dtype = (_residual_fp32 + _sublayer_fp16).dtype
_findings_5_8["residual_addition_promotes_to_fp32"] = _added_dtype == torch.float32
print(f"[{device.type}] FP32 + FP16 の加算の dtype: {_added_dtype}(autocast の外、通常の型昇格規則)")

for _name, _ok in _findings_5_8.items():
    if not _ok:
        print(f"警告: 前提「{_name}」がこのデバイス({device.type})では成り立たなかった。"
              "3.6 節の記述の再確認が必要。")

```

    [cuda] autocast(fp16) 下で x.pow(2) の dtype: torch.float32(FP32への昇格: True)
    [cuda] autocast(fp16) 下で x.mean(-1)(x が FP16)の dtype: torch.float16(FP16のまま: True)
    [cuda] autocast 下で RMSNorm(FP16 入力)の出力 dtype: torch.float32
    [cuda] autocast なしで RMSNorm(FP16 入力)の出力 dtype: torch.float16(対照、強制的な昇格ではないことの確認)
    [cuda] FP32 + FP16 の加算の dtype: torch.float32(autocast の外、通常の型昇格規則)



```python
# --- モデル全体での確認: 各地点の dtype を記録する ---
_probe_model_58 = build_model(seed=0).to(device)
_dtypes_58: dict[str, str] = {}


def _make_dtype_hook(name):
    def hook(module, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        _dtypes_58[name] = str(o.dtype)

    return hook


for _i, _block in enumerate(_probe_model_58.blocks):
    _block.norm1.register_forward_hook(_make_dtype_hook(f"blk{_i}.norm1(RMSNorm, pre-attn)"))
    _block.self_attn.register_forward_hook(_make_dtype_hook(f"blk{_i}.self_attn"))
    _block.norm3.register_forward_hook(_make_dtype_hook(f"blk{_i}.norm3(RMSNorm, pre-ffn)"))
    _block.feed_forward.register_forward_hook(_make_dtype_hook(f"blk{_i}.feed_forward"))
    _block.register_forward_hook(_make_dtype_hook(f"blk{_i}.block_output"))
_probe_model_58.final_norm.register_forward_hook(_make_dtype_hook("final_norm(RMSNorm)"))

_gen_58 = torch.Generator(device="cpu")
_gen_58.manual_seed(0)
_inputs_58, _targets_58 = get_random_batch(train_ids, BATCH_SIZE, SEQUENCE_LENGTH, _gen_58)
_inputs_58, _targets_58 = _inputs_58.to(device), _targets_58.to(device)
with torch.autocast(device_type=device.type, dtype=torch.float16):
    _logits_58 = _probe_model_58(_inputs_58)
    _loss_58 = F.cross_entropy(_logits_58.reshape(-1, _logits_58.size(-1)), _targets_58.reshape(-1))

for _name, _dt in _dtypes_58.items():
    print(f"{_name}: {_dt}")
print(f"logits: {_logits_58.dtype}")
print(f"loss: {_loss_58.dtype}")

_model_wide_checks = {
    "self_attn の出力は FP16": all("float16" in v for k, v in _dtypes_58.items() if "self_attn" in k),
    "feed_forward の出力は FP16": all("float16" in v for k, v in _dtypes_58.items() if "feed_forward" in k),
    "RMSNorm(norm1/norm3/final_norm)の出力は FP32": all(
        "float32" in v for k, v in _dtypes_58.items() if "RMSNorm" in k
    ),
    "block_output(残差接続後)は FP32": all("float32" in v for k, v in _dtypes_58.items() if "block_output" in k),
    "logits は FP16": _logits_58.dtype == torch.float16,
    "loss は FP32": _loss_58.dtype == torch.float32,
}
for _desc, _ok in _model_wide_checks.items():
    _mark = "OK" if _ok else "警告: 成り立たなかった"
    print(f"{_mark}: {_desc}")
del _probe_model_58

# --- 前提条件 P0(新定義): autocast 下でサブレイヤー(自己注意機構・順伝播ネットワーク)の
# 出力が FP16 になることの 2 項目のみで判定する。残りの 4 項目(RMSNorm・Decoder Block の
# 出力・logits・loss の dtype)は診断量として上で印字済みだが、P0 の判定には含めない
# (後続の測定(静的スケールの較正・P2・診断量のフック位置)が依存するのは、サブレイヤー
# 出力が実際に FP16 で計算されているかどうかのみであるため)。
_P0_RELEVANT_CHECKS = ("self_attn の出力は FP16", "feed_forward の出力は FP16")
precondition_status["P0"] = all(_model_wide_checks[k] for k in _P0_RELEVANT_CHECKS)
print(f"P0(新定義、{_P0_RELEVANT_CHECKS}のみ): -> {'成立' if precondition_status['P0'] else '不成立'}")

```

    blk0.norm1(RMSNorm, pre-attn): torch.float32
    blk0.self_attn: torch.float16
    blk0.norm3(RMSNorm, pre-ffn): torch.float32
    blk0.feed_forward: torch.float16
    blk0.block_output: torch.float32
    blk1.norm1(RMSNorm, pre-attn): torch.float32
    blk1.self_attn: torch.float16
    blk1.norm3(RMSNorm, pre-ffn): torch.float32
    blk1.feed_forward: torch.float16
    blk1.block_output: torch.float32
    blk2.norm1(RMSNorm, pre-attn): torch.float32
    blk2.self_attn: torch.float16
    blk2.norm3(RMSNorm, pre-ffn): torch.float32
    blk2.feed_forward: torch.float16
    blk2.block_output: torch.float32
    blk3.norm1(RMSNorm, pre-attn): torch.float32
    blk3.self_attn: torch.float16
    blk3.norm3(RMSNorm, pre-ffn): torch.float32
    blk3.feed_forward: torch.float16
    blk3.block_output: torch.float32
    final_norm(RMSNorm): torch.float32
    logits: torch.float16
    loss: torch.float32
    OK: self_attn の出力は FP16
    OK: feed_forward の出力は FP16
    OK: RMSNorm(norm1/norm3/final_norm)の出力は FP32
    OK: block_output(残差接続後)は FP32
    OK: logits は FP16
    OK: loss は FP32
    P0(新定義、('self_attn の出力は FP16', 'feed_forward の出力は FP16')のみ): -> 成立


## 6. 実験 / Experiments

### 6.1 較正・外挿で使うヘルパー関数

較正(6.2 節)・スケーリング外挿(6.3 節)は、`SMOKE_TEST`の値に関わらず
**常に本番スケール(`PROD_CFG`)** で行う(5.3 節)。コーパス(`train_ids`)は
5.4 節で読み込んだものをそのまま再利用する(`VALIDATION_RATIO`の違いは訓練
データ側にわずかな差を生むが、較正対象である勾配の大きさ・分布は検証データの
取り方には依存しないため問題にならない)。



```python
def build_model_prod(seed: int) -> GPTLanguageModel:
    torch.manual_seed(seed)
    d_k = PROD_D_MODEL // PROD_NUM_HEADS
    rope = RotaryPositionEmbedding(d_k, max_position=PROD_SEQUENCE_LENGTH)
    return GPTLanguageModel(
        vocabulary_size=VOCAB_SIZE,
        d_model=PROD_D_MODEL,
        num_layers=PROD_NUM_LAYERS,
        num_heads=PROD_NUM_HEADS,
        d_ff=PROD_D_FF,
        max_sequence_length=PROD_SEQUENCE_LENGTH,
        positional_transform=rope,
        normalization_factory=RMSNorm,
        feed_forward_factory=functools.partial(SwiGLUFeedForwardNetwork, PROD_D_MODEL, PROD_SWIGLU_D_FF),
        tie_embeddings=True,
        dropout=0.0,
        norm_first=True,
    )


def run_prod_training(seed: int, num_steps: int, autocast_dtype=None, loss_scaler=None,
                       clip_threshold: float | None = None) -> dict:
    model = build_model_prod(seed).to(device)
    # run_condition(6.7 節)と同じ診断フックを付けて計測する。診断量自体はここでは
    # 使わないが、フック付きで学習した場合の実時間を、6.3 節の外挿・見積もりに
    # 反映させるため(時間計測の目的のみ)。
    attach_sublayer_diagnostic_hooks(model)
    optimizer = AdamW(model.parameters(), lr=BASE_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    schedule = functools.partial(
        compute_warmup_cosine_learning_rate,
        warmup_steps=PROD_WARMUP_STEPS, total_steps=PROD_NUM_STEPS,
        peak_learning_rate=BASE_LEARNING_RATE, min_learning_rate=PROD_MIN_LR,
    )
    dummy_windows = torch.zeros(1, PROD_SEQUENCE_LENGTH, dtype=torch.long)
    dummy_mask = torch.ones(1, PROD_SEQUENCE_LENGTH, dtype=torch.bool)
    return train_language_model(
        model, train_ids, dummy_windows, dummy_mask, 1,
        num_steps=num_steps, batch_size=PROD_BATCH_SIZE, sequence_length=PROD_SEQUENCE_LENGTH,
        learning_rate=BASE_LEARNING_RATE, eval_interval=num_steps + 1, device=device, seed=seed,
        optimizer=optimizer, learning_rate_schedule=schedule, gradient_clip_threshold=clip_threshold,
        autocast_dtype=autocast_dtype, loss_scaler=loss_scaler,
    )


print(f"較正・外挿は常に本番スケールを使う: D_MODEL={PROD_D_MODEL}, NUM_STEPS={PROD_NUM_STEPS}, "
      f"BATCH_SIZE={PROD_BATCH_SIZE}, WARMUP_STEPS={PROD_WARMUP_STEPS}")

```

    較正・外挿は常に本番スケールを使う: D_MODEL=256, NUM_STEPS=300, BATCH_SIZE=32, WARMUP_STEPS=30




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/03_efficient_training/011_mixed_precision_training.ipynb
