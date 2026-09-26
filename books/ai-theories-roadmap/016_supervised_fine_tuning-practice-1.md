---
title: "SFT(指示チューニング) / Supervised Fine-Tuning (Instruction Tuning)(実装・実験編 1/4)"
---

この記事は後編(実装・実験編 1/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/016_supervised_fine_tuning-theory)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/016_supervised_fine_tuning-practice-2)。

## 4. 実装方針 / Implementation Policy

**`src/`に切り出す(スクラッチ実装、本トピックで新規作成)**:

- `src/data/instruction.py`: 合成の指示データ。課題(`TASK_NAMES`・`apply_task()`)、単語の語彙
  (`WORD_VOCABULARY`、単一トークンの確認は`check_single_token_words()`)、指示文(学習用の`SEEN_INSTRUCTIONS`、
  未見の`UNSEEN_INSTRUCTIONS`)と長い水準の前置き(`PREAMBLES`)、テンプレートの適用(`format_prompt()`・
  `format_response()`)、入力の生成(`sample_word_sequences()`)、学習用・評価用の事例の作成
  (`build_training_examples()`・`build_evaluation_examples()`、評価用は入力と課題の直積)、区切りごとの
  符号化(`encode_instruction_example()`)、右パディングと損失マスクの作成(`collate_instruction_batch()`)、
  $c^*$ の計算(`compute_instruction_agnostic_upper_bound()`)、採点(`score_generated_response()`)。
- `src/training/instruction_tuning.py`: 3.3 節の正規化の単位による損失(`compute_instruction_tuning_loss()`)、
  エポックごとの並べ替えによるミニバッチの添字(`make_epoch_batches()`)、SFT の学習ループ
  (`train_instruction_tuning()`)、教師強制(teacher forcing)による事例ごとの負の対数尤度の評価
  (`evaluate_instruction_negative_log_likelihood()`)。既存の`train_language_model()`は連続したトークン列から
  ランダムな区間を切り出す事前学習用のループで、事例の境界も損失マスクも扱わないため、拡張せずに新規に置いた
  (既存の関数・既存のファイルの挙動は変わらない)。
- `src/generation/stopping.py`: 終端記号の文字列で停止する貪欲法の生成(`greedy_generate_until_stop()`)。
  `GPTLanguageModel.generate()`は指定したトークン数だけ生成し、途中で止まらないため、新規に置いた。
  指示部分の長さごとにバッチにする(パディングを使わない)ので、1 事例ずつ生成する場合と位置・因果マスクが
  同じになる。既存の`generate()`は変更していない(5.5 節で、停止しない場合の出力が`generate()`と一致することを
  確認する)。

既存のモジュール(`train_language_model()`・`generate()`など)は変更していないため、後方互換性の検証
(変更前のコミットとの数値比較)は要らない。LoRA は 012 の`apply_lora()`、optimizer は 007 の`AdamW`、
学習率スケジュールは 007 の warmup + cosine をそのまま使う。

**ノートブック内に直接書く(016 固有)**: 条件の定義と学習・評価の呼び出し、
実験 A〜C の判定、コピー能力の診断、スケーリングの計測。

**アップロード方針**: 本トピックで学習する LoRA のアダプタは、すべて条件間の比較のためのものであり、後続トピックの
入力にも、読者が単体で取得する対象にもならない。保存もアップロードもしない(アップロードのセルも置かない)。

**生成物の置き場所**: `.cache/`には何も置かない(合成データは乱数シードから決定的に生成でき、
全体でも数秒で作れるため)。判定の記録は、各判定セル(6.6・6.7・6.10・6.12 節)の出力(対比量・標準偏差・判定の閾値・前提条件の値と成否・判定結果)である。外部から取得するのは、
Hugging Face Hub のモデル(`kojikojiprg/ai-theories-small-gpt-en`の`main`)・トークナイザ
(`kojikojiprg/ai-theories-tokenizer-en`)と、前提条件 P0 のための英語版 Wikipedia のコーパス
(`kojikojiprg/ai-theories-corpus-en-pretraining`、取得できない場合のみ Wikipedia から直接取得)である。

## 5. 実装 / Implementation

### 5.1 環境セットアップ(Google Colab)

`SMOKE_TEST`(スモークテストか本番か)はこのセルでのみ切り替える。実行環境はこのセルで 1 回だけ印字する。
再現性のため(6.1 節)、CUDA の cuBLAS が決定的な演算を使うための環境変数`CUBLAS_WORKSPACE_CONFIG`を
torch の読み込み前に設定し、`torch.use_deterministic_algorithms(True)`を有効にする(決定的な実装を持たない演算を
使うと例外になるので、5.6 節で学習・評価・生成の経路を一通り実行して確かめる)。


```python
# 環境セットアップ(Google Colab)
import os
import sys
import time

SMOKE_TEST = False  # Claude Code はこの True 側のみ実行する(Colab T4 では False に切り替える)

NOTEBOOK_START_TIME = time.time()
# cuBLAS を決定的にする(torch が CUDA を初期化する前に設定する必要がある)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    !git clone https://github.com/kojikojiprg/ai-theories.git
    %cd ai-theories
    !pip install uv -q
    !uv pip install --system -r requirements.txt
# ローカル(Jupyter)実行時は、リポジトリルートで起動していればそのまま動く。

import torch  # noqa: E402

from src.utils.environment import print_execution_environment  # noqa: E402

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
execution_environment = print_execution_environment(device)

print(
    f"SMOKE_TEST={SMOKE_TEST}、決定的な実行: {torch.are_deterministic_algorithms_enabled()}、"
    f"CUBLAS_WORKSPACE_CONFIG={os.environ['CUBLAS_WORKSPACE_CONFIG']}"
)
```

    Cloning into 'ai-theories'...
    remote: Enumerating objects: 1181, done.[K
    remote: Counting objects: 100% (221/221), done.[K
    remote: Compressing objects: 100% (146/146), done.[K
    remote: Total 1181 (delta 133), reused 146 (delta 75), pack-reused 960 (from 1)[K
    Receiving objects: 100% (1181/1181), 12.85 MiB | 10.66 MiB/s, done.
    Resolving deltas: 100% (694/694), done.
    /content/ai-theories
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m20.5/20.5 MB[0m [31m46.9 MB/s[0m eta [36m0:00:00[0m
    [?25h[2mUsing Python 3.13.15 environment at: /usr[0m
    [2K[2mResolved [1m52 packages[0m [2min 342ms[0m[0m
    [2K[2mPrepared [1m31 packages[0m [2min 1m 00s[0m[0m
    [2mUninstalled [1m17 packages[0m [2min 968ms[0m[0m
    [2K[2mInstalled [1m31 packages[0m [2min 419ms[0m[0m
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
      コミット / git commit                  : 4c13e5d34cac8499dacec4577e2f2db4306773e2
      未コミットの変更 / uncommitted changes : なし
      実行日時 (UTC)                         : 2026-09-26T04:14:38+00:00
    SMOKE_TEST=False、決定的な実行: True、CUBLAS_WORKSPACE_CONFIG=:4096:8



```python
import copy
import functools
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import nn

from src.data.instruction import (
    END_MARKER,
    PREAMBLES,
    SEEN_INSTRUCTIONS,
    TASK_NAMES,
    UNSEEN_INSTRUCTIONS,
    WORD_VOCABULARY,
    build_evaluation_examples,
    build_training_examples,
    check_single_token_words,
    collate_instruction_batch,
    compute_instruction_agnostic_upper_bound,
    encode_instruction_example,
    sample_word_sequences,
    score_generated_response,
    template_words,
)
from src.data.text import (
    load_wikipedia_corpus_with_fallback,
    make_evaluation_windows,
    split_train_val_text,
)
from src.data.tokenizer import load_bpe_id_tokenizer_from_hub
from src.generation.stopping import greedy_generate_until_stop
from src.layers.feedforward import SwiGLUFeedForwardNetwork
from src.layers.lora import apply_lora, compute_lora_parameter_count
from src.layers.normalization import RMSNorm
from src.layers.positional_encoding import RotaryPositionEmbedding
from src.models.gpt import GPTLanguageModel
from src.training.instruction_tuning import (
    compute_instruction_tuning_loss,
    evaluate_instruction_negative_log_likelihood,
    make_epoch_batches,
    train_instruction_tuning,
)
from src.training.optimizer import AdamW
from src.training.schedule import compute_warmup_cosine_learning_rate
from src.training.trainer import evaluate_bits_per_byte
from src.utils.reporting import dumps_compact_json
from src.utils.statistics import (
    fit_power_law_exponent,
    paired_bootstrap_ratio_of_sums,
    paired_cluster_bootstrap_ratio_of_sums,
)

ROOT = Path(".")
WIKIPEDIA_CACHE_DIR = (
    ROOT / ".cache" / "wikipedia_en"
)  # 外部取得したコーパス(データ源で命名、P0 のみに使う)
MODEL_REPO_ID = "kojikojiprg/ai-theories-small-gpt-en"
MODEL_REVISION = "main"
TOKENIZER_REPO_ID = "kojikojiprg/ai-theories-tokenizer-en"
CORPUS_REPO_ID = "kojikojiprg/ai-theories-corpus-en-pretraining"
MANIFEST_PATH = ROOT / "src" / "data" / "wikipedia_manifests" / "en_006_pretraining.json"


def sync_device() -> None:
    # 時間計測の直前・直後に、非同期に実行される GPU の処理の完了を待つ。
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def timed_call(fn) -> float:
    sync_device()
    start = time.time()
    fn()
    sync_device()
    return time.time() - start


def hash_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()[:16]


precondition_status: dict[str, bool] = {}  # 前提条件の成否(6.1 節で宣言、各節で記録)
```

### 5.2 スケールの設定(`SMOKE_TEST`の配線)

水準の定義をこの 1 箇所(`LEVELS`)に集約する。モデルは事前学習済みのものを使うので縮小しない。

**縮小の規則**:

- 学習ステップ数 $T$(本番 2048・スモークテスト 32)を縮小する。バッチサイズ $b = 32$ は縮小しない。
  最大データ数 $N_{\max} = T b$(1 エポックちょうど)という関係は両水準で保つ。
- 実験 C の水準は、両水準とも公比 4 の等比数列で $N_{\max}$ まで取る(段階 0 の最小水準は 16、段階 1 以降は 64。
  本番: 段階 0 で 16〜65536 の 7 水準、スモークテスト: 段階 0 で 16〜1024 の 4 水準)。本番の $N_{\max}$ が
  $16 \cdot 4^6$、スモークテストが $16 \cdot 4^3$ になるように $T$ を選んでいるので、等比の構造(公比・最小水準・
  最大水準 $= Tb$)は縮小後も保たれる。
- **削る段階の表**(6.1 節)は、本番とスモークテストで同じ構造にする。シード数は本番の 5 → 3 をスモークテストでは
  3 → 2 に縮小する(「削る前 > 削った後 $\ge 2$」の順序関係を保つ。標本標準偏差には 2 以上が要る)。最小水準の
  16 → 64 は両水準で同じ。
- 評価用の入力の数 $|X|$(本番 250・スモークテスト 16)、ブートストラップの反復回数、コピー能力の診断の系列数を
  縮小する。
- **テスト専用の上書き**: 環境変数`AI_THEORIES_FORCE_STAGE`(0〜3)が設定されているときのみ、6.4 節で見積もりによる
  段階の選択の代わりにその段階を使う(スモークテストで全段階の経路を確かめるため。上書きしたことを目立つ形で印字する)。
  本番(`SMOKE_TEST = False`)では受け付けず、このセルで例外で停止する。コミットする出力は上書きなしの実行のものである。
- 課題・テンプレート・単語の語彙・学習率・LoRA の設定・warmup の割合は縮小しない。したがって、長い水準の指示部分が短い水準の約 4 倍という比率(5.4 節で実測)は、縮小後も
  そのまま保たれる。


```python
# --- 全水準で共通の定数(本番実行前に宣言し、SMOKE_TEST で変えない) ---
TASK_COUNT = len(TASK_NAMES)  # K
MIN_WORDS, MAX_WORDS = 3, 8  # 入力の単語数
DATA_SEED = (
    16_016  # 合成データの乱数シード(学習のシードとは独立。パイロットのシードとも異なる、6.2 節)
)
TIMING_DATA_SEED = 16_099  # スケーリングの計測専用のデータ(本番の評価集合に触れずに計測するため)
BATCH_SIZE = 32  # b
LEARNING_RATE = 1e-2  # 6.2 節で 012 の 10^-1.5 から改訂
LORA_RANK, LORA_ALPHA = 8, 8.0  # 012 と同じ(alpha / r = 1)
LORA_TARGET_MODULES = ("w_q", "w_v")  # Query・Value 射影(012 と同じ)
WEIGHT_DECAY = 0.0  # 012 と同じ
WARMUP_RATIO = 0.1  # 007・008・012 と同じ
MIN_LEARNING_RATE_RATIO = 0.01  # 007・008・012 と同じ
# gradient clipping は使わない(012 の学習の部品のうち clipping のみ踏襲しない、6.2 節の改訂 5)
MAX_NEW_TOKENS = 32  # 生成の上限トークン数(応答部分の最大長 22 に余裕を持たせる、5.4 節で確認)
EVAL_BATCH_SIZE = 64  # 教師強制の評価の 1 回の順伝播の事例数(結果には影響しない)
GENERATION_BATCH_SIZE = 64  # 生成の 1 回の順伝播の事例数の上限
C_LEVEL_RATIO = 4  # 実験 C の水準の公比
PRECONDITION_L_RATIO = 0.5  # P-L: 応答部分の負の対数尤度 <= 微調整前の値 x 0.5
P0_REFERENCE_BITS_PER_BYTE = 1.668067  # 008 のモデルカードの値(アップロードした重みを評価した値)
P0_RELATIVE_TOLERANCE = 0.01
P0_VALIDATION_RATIO = 0.05  # 008 と同じ分割(コーパスの末尾 5% が検証)
LONG_PROMPT_RATIO_RANGE = (3.5, 4.5)  # 長い水準の指示部分のトークン数 / 短い水準(約 4 倍)
COPY_SEQUENCE_WORDS = 8  # コピー能力の診断の単語列の長さ(6.8 節)
BOOTSTRAP_SEED = 0
SESSION_BUDGET_SECONDS = (
    120 * 60
)  # 1 セッションの予算(T4 で 120 分、1.1 節)。削る段階の選択に使う(6.4 節)

# 条件の対応表(6.1 節)。C-N は条件 1 と同じ設定でデータ数だけを変える。
CONDITIONS = {
    1: {"mask": True, "long": False},
    2: {"mask": False, "long": False},
    3: {"mask": True, "long": True},
    4: {"mask": False, "long": True},
}

# --- 水準の定義(この 1 箇所に集約する) ---
LEVELS = {
    "smoke": {
        "NUM_STEPS": 32,
        "NUM_EVAL_INPUTS": 16,
        "BOOTSTRAP_RESAMPLES": 1_000,
        "NUM_COPY_SEQUENCES": 8,
    },
    "prod": {
        "NUM_STEPS": 2048,
        "NUM_EVAL_INPUTS": 250,
        "BOOTSTRAP_RESAMPLES": 10_000,
        "NUM_COPY_SEQUENCES": 64,
    },
}
CURRENT_LEVEL_NAME = "smoke" if SMOKE_TEST else "prod"

# テスト専用の上書き: 環境変数 AI_THEORIES_FORCE_STAGE が設定されているときのみ、6.4 節で見積もりによる段階の選択の
# 代わりにその段階を使う(スモークテストで全段階の経路を確かめるため)。本番では受け付けず、ここで停止する。
FORCED_STAGE_VALUE = os.environ.get("AI_THEORIES_FORCE_STAGE")
if FORCED_STAGE_VALUE is not None and not SMOKE_TEST:
    raise RuntimeError(
        f"本番(SMOKE_TEST=False)では段階の強制(AI_THEORIES_FORCE_STAGE={FORCED_STAGE_VALUE!r})を受け付けない。"
        "環境変数を削除して再実行すること。"
    )
FORCED_STAGE = None if FORCED_STAGE_VALUE is None else int(FORCED_STAGE_VALUE)
assert FORCED_STAGE is None or FORCED_STAGE in (0, 1, 2, 3), FORCED_STAGE
CFG = LEVELS[CURRENT_LEVEL_NAME]
NUM_STEPS = CFG["NUM_STEPS"]  # T
NUM_EVAL_INPUTS = CFG["NUM_EVAL_INPUTS"]  # |X|
BOOTSTRAP_RESAMPLES = CFG["BOOTSTRAP_RESAMPLES"]
NUM_COPY_SEQUENCES = CFG["NUM_COPY_SEQUENCES"]
N_MAX = NUM_STEPS * BATCH_SIZE  # 最大データ数(1 エポックちょうど)
WARMUP_STEPS = max(1, round(WARMUP_RATIO * NUM_STEPS))


def c_levels(min_level: int, n_max: int) -> tuple[int, ...]:
    levels = [min_level]
    while levels[-1] < n_max:
        levels.append(levels[-1] * C_LEVEL_RATIO)
    assert levels[-1] == n_max, f"N_max = {n_max} が {min_level} x {C_LEVEL_RATIO}^k にならない"
    return tuple(levels)


PROD_N_MAX = LEVELS["prod"]["NUM_STEPS"] * BATCH_SIZE

# --- 削る段階(6.1 節)。6.4 節で、見積もりのみから予算に収まる最小の段階を選ぶ ---
# 失う情報の小さい順: 実験 C の下端の分解能 -> 実験 C の精度(シード数)-> 実験 B の精度(シード数)
STAGES = {
    "prod": {
        0: {"C_MIN_LEVEL": 16, "NUM_SEEDS_C": 5, "NUM_SEEDS_AB": 5},
        1: {"C_MIN_LEVEL": 64, "NUM_SEEDS_C": 5, "NUM_SEEDS_AB": 5},
        2: {"C_MIN_LEVEL": 64, "NUM_SEEDS_C": 3, "NUM_SEEDS_AB": 5},
        3: {"C_MIN_LEVEL": 64, "NUM_SEEDS_C": 3, "NUM_SEEDS_AB": 3},
    },
    "smoke": {  # 本番と同じ構造(シード数 5 -> 3 を 3 -> 2 に縮小)
        0: {"C_MIN_LEVEL": 16, "NUM_SEEDS_C": 3, "NUM_SEEDS_AB": 3},
        1: {"C_MIN_LEVEL": 64, "NUM_SEEDS_C": 3, "NUM_SEEDS_AB": 3},
        2: {"C_MIN_LEVEL": 64, "NUM_SEEDS_C": 2, "NUM_SEEDS_AB": 3},
        3: {"C_MIN_LEVEL": 64, "NUM_SEEDS_C": 2, "NUM_SEEDS_AB": 2},
    },
}
ALL_C_LEVELS = c_levels(16, N_MAX)  # 段階 0 の水準(どの段階の水準もこの部分集合)

# --- 縮小規則の確認 ---
_smoke, _prod = LEVELS["smoke"], LEVELS["prod"]
assert set(_smoke) == set(_prod)
for _name, _level in LEVELS.items():
    _n_max = _level["NUM_STEPS"] * BATCH_SIZE
    _stages = STAGES[_name]
    assert set(_stages) == {0, 1, 2, 3}
    for _k, _stage in _stages.items():
        assert (
            c_levels(_stage["C_MIN_LEVEL"], _n_max)[-1] == _n_max
        )  # 公比 4 の等比で N_max = T b に届く
        assert (
            2 <= _stage["NUM_SEEDS_C"] <= _stage["NUM_SEEDS_AB"]
        )  # C のシードは A・B のシードの先頭部分
        if _k > 0:  # 段階が上がるほど、どの量も減るか変わらない(順序関係)
            _prev = _stages[_k - 1]
            assert _stage["C_MIN_LEVEL"] >= _prev["C_MIN_LEVEL"]
            assert (
                _stage["NUM_SEEDS_C"] <= _prev["NUM_SEEDS_C"]
                and _stage["NUM_SEEDS_AB"] <= _prev["NUM_SEEDS_AB"]
            )
    assert c_levels(64, _n_max) == c_levels(16, _n_max)[1:]
# 本番とスモークテストで、各段階で何が変わるかが同じ(構造を保つ縮小)
for _k in (1, 2, 3):
    for _key in ("C_MIN_LEVEL", "NUM_SEEDS_C", "NUM_SEEDS_AB"):
        assert (STAGES["prod"][_k][_key] == STAGES["prod"][_k - 1][_key]) == (
            STAGES["smoke"][_k][_key] == STAGES["smoke"][_k - 1][_key]
        )
assert (
    _smoke["NUM_STEPS"] < _prod["NUM_STEPS"]
    and _smoke["NUM_EVAL_INPUTS"] < _prod["NUM_EVAL_INPUTS"]
)
assert WARMUP_STEPS < NUM_STEPS

print(f"水準 {CURRENT_LEVEL_NAME!r}: {json.dumps(CFG)}")
print(
    f"T = {NUM_STEPS}、b = {BATCH_SIZE}、N_max = T b = {N_MAX:,}、warmup = {WARMUP_STEPS} ステップ、"
    f"評価用の入力 |X| = {NUM_EVAL_INPUTS}(事例 M = {NUM_EVAL_INPUTS * TASK_COUNT})"
)
print(f"削る段階({CURRENT_LEVEL_NAME!r}、6.4 節で選ぶ): {json.dumps(STAGES[CURRENT_LEVEL_NAME])}")
print(f"実験 C の水準(公比 {C_LEVEL_RATIO}、段階 0): {ALL_C_LEVELS}")
print(
    f"学習率 = {LEARNING_RATE}、LoRA r = {LORA_RANK}・alpha = {LORA_ALPHA}・対象 {LORA_TARGET_MODULES}、"
    f"重み減衰 = {WEIGHT_DECAY}、最小学習率の比 = {MIN_LEARNING_RATE_RATIO}、gradient clipping なし"
)
```

    水準 'prod': {"NUM_STEPS": 2048, "NUM_EVAL_INPUTS": 250, "BOOTSTRAP_RESAMPLES": 10000, "NUM_COPY_SEQUENCES": 64}
    T = 2048、b = 32、N_max = T b = 65,536、warmup = 205 ステップ、評価用の入力 |X| = 250(事例 M = 1000)
    削る段階('prod'、6.4 節で選ぶ): {"0": {"C_MIN_LEVEL": 16, "NUM_SEEDS_C": 5, "NUM_SEEDS_AB": 5}, "1": {"C_MIN_LEVEL": 64, "NUM_SEEDS_C": 5, "NUM_SEEDS_AB": 5}, "2": {"C_MIN_LEVEL": 64, "NUM_SEEDS_C": 3, "NUM_SEEDS_AB": 5}, "3": {"C_MIN_LEVEL": 64, "NUM_SEEDS_C": 3, "NUM_SEEDS_AB": 3}}
    実験 C の水準(公比 4、段階 0): (16, 64, 256, 1024, 4096, 16384, 65536)
    学習率 = 0.01、LoRA r = 8・alpha = 8.0・対象 ('w_q', 'w_v')、重み減衰 = 0.0、最小学習率の比 = 0.01、gradient clipping なし


### 5.3 ベースモデル・トークナイザの取得(008 がアップロードしたもの)

ベースモデルは`kojikojiprg/ai-theories-small-gpt-en`の`main`、トークナイザは`kojikojiprg/ai-theories-tokenizer-en`
から取得する。モデルの構成は取得した`config.json`から読む。各条件は、Hub から読み込み直す代わりに`base_model`の
複製から始める。


```python
tokenizer, _tokenizer_from_hub = load_bpe_id_tokenizer_from_hub(TOKENIZER_REPO_ID)
assert _tokenizer_from_hub, "トークナイザを Hugging Face Hub から取得できなかった"
_config_path = hf_hub_download(MODEL_REPO_ID, "config.json", revision=MODEL_REVISION)
MODEL_STATE_PATH = hf_hub_download(MODEL_REPO_ID, "model_state.pt", revision=MODEL_REVISION)
MODEL_CONFIG = json.loads(Path(_config_path).read_text(encoding="utf-8"))
assert MODEL_CONFIG["positional_encoding"] == "rope"
assert MODEL_CONFIG["normalization"] == "rmsnorm" and MODEL_CONFIG["feed_forward"] == "swiglu"
assert MODEL_CONFIG["norm_first"] and MODEL_CONFIG["dropout"] == 0.0
assert tokenizer.vocab_size == MODEL_CONFIG["vocabulary_size"]

CONTEXT_LENGTH = MODEL_CONFIG["sequence_length"]  # 008 のモデルの文脈長
D_MODEL = MODEL_CONFIG["d_model"]
NUM_LAYERS = MODEL_CONFIG["num_layers"]
NUM_HEADS = MODEL_CONFIG["num_heads"]
VOCAB_SIZE = MODEL_CONFIG["vocabulary_size"]


def build_base_model() -> GPTLanguageModel:
    model = GPTLanguageModel(
        vocabulary_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=MODEL_CONFIG["d_ff"],
        max_sequence_length=CONTEXT_LENGTH,
        positional_transform=RotaryPositionEmbedding(
            D_MODEL // NUM_HEADS, max_position=CONTEXT_LENGTH
        ),
        normalization_factory=RMSNorm,
        feed_forward_factory=functools.partial(
            SwiGLUFeedForwardNetwork, D_MODEL, MODEL_CONFIG["swiglu_d_ff"]
        ),
        tie_embeddings=MODEL_CONFIG["tie_embeddings"],
    )
    result = model.load_state_dict(torch.load(MODEL_STATE_PATH, map_location="cpu"))
    assert not result.missing_keys and not result.unexpected_keys
    return model.to(device).eval()


base_model = build_base_model()
for _p in base_model.parameters():
    _p.requires_grad_(False)  # base_model 自体は学習しない(各条件は複製から始める)
BASE_PARAMETERS = {n: p.detach().clone() for n, p in base_model.named_parameters()}
TOTAL_PARAMETERS = sum(p.numel() for p in base_model.parameters())
LORA_PARAMETERS = 2 * NUM_LAYERS * compute_lora_parameter_count(D_MODEL, D_MODEL, LORA_RANK)
print(
    f"層数 = {NUM_LAYERS}、d_model = {D_MODEL}、ヘッド数 = {NUM_HEADS}、語彙サイズ = {VOCAB_SIZE}、"
    f"文脈長 = {CONTEXT_LENGTH}、パラメータ数 = {TOTAL_PARAMETERS:,}、"
    f"LoRA の学習可能パラメータ数 = {LORA_PARAMETERS:,}(閉形式 4 L r d_model)"
)
```


    tokenizer.json:   0%|          | 0.00/661k [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/305 [00:00<?, ?B/s]



    model_state.pt: reconstructing file:   0%|          |  0.00B / 21.0MB            



    model_state.pt: downloading bytes:           |  0.00B            


    層数 = 4、d_model = 256、ヘッド数 = 8、語彙サイズ = 8192、文脈長 = 256、パラメータ数 = 5,246,208、LoRA の学習可能パラメータ数 = 32,768(閉形式 4 L r d_model)


### 5.4 合成の指示データ

**課題**($K = 4$、いずれも単語を単位とする並べ替え・複製・選択。文字単位の操作は BPE(Byte Pair Encoding)の部分語分割と相性が
悪いので避ける):

| 課題 | 応答(入力 $w_1 \dots w_n$ に対して) |
|---|---|
| 逆順(`reverse`) | $w_n \dots w_1$ |
| 各単語を 2 回ずつ(`repeat_twice`) | $w_1 w_1 w_2 w_2 \dots w_n w_n$ |
| 奇数番目のみ(`odd_positions`) | $w_1 w_3 w_5 \dots$ |
| 先頭を末尾へ移す回転(`rotate_left`) | $w_2 \dots w_n w_1$ |

候補にあった「先頭と末尾の単語を入れ替える」課題は、3 語の入力で逆順と応答が一致する($w_3 w_2 w_1$)。
これを採用すると、3 語の入力では 2 つの課題を区別できず、指示を読まなくても 2 課題に正解できるので
$c^*$(6.1 節)が上がる。回転は、互いに異なる 3 語以上の入力に対して、4 課題の応答がすべて異なる
(逆順と回転は長さが同じだが、先頭の単語が $w_n$ と $w_2$ で異なる。ほかの 2 課題とは長さが異なる)。
このため $c^* = 1/K$ になり、評価集合のすべての事例が「指示を読まなければ解けない」事例になる。

**入力**: 3〜8 語。単語は、先頭に空白を付けた形で 008 のトークナイザの **単一トークン** になる英単語 454 語
(`WORD_VOCABULARY`)から、1 つの入力の中で重複しないように選ぶ。単一トークンであること、テンプレートに
現れる単語と重ならないことをアサーションで確かめる。

**指示文**: 課題ごとに学習用の 4 個と、学習に使わない「未見テンプレート」2 個(実験 A の診断量)を用意する。
**短い水準** は指示文そのもの、**長い水準** は課題によらない冗長な前置き(4 種類)を指示文の前に加えたもので、
課題の意味を変えない。応答部分は両水準で同一である。

**データの構成**: 合成データの乱数シード(`DATA_SEED`)は学習のシードとは独立に固定し、全条件・全シードで
同じデータを使う。

- 学習データ: $N_{\max} = Tb$ 個の事例。先頭から $K$ 個ずつの区切りごとに 4 課題を 1 回ずつ割り当てるので、
  実験 C の入れ子の部分集合(先頭 $N$ 個)でも課題の数が均等になる。
- 評価集合: $|X|$ 個の入力と 4 課題の直積(事例 $M = K|X|$ 個)。評価用の入力は学習データの入力と重複しない。
- 未見テンプレートの評価集合: 同じ入力 $X$ と 4 課題の直積に、未見の指示文を割り当てたもの。
- **パイロットの評価入力との非重複**: 学習率の選択(6.2 節)にはパイロットの評価集合の完全一致率を使った。
  本番の評価集合がそれと重なると、学習率を選んだデータで評価することになるので、本番の評価集合はパイロットとは
  別の乱数シードで生成し、パイロットの評価入力(同じ関数と乱数シードで再生成する)と重複しないことを確かめる。


```python
_t0_data = time.time()
# --- 単語の語彙: 単一トークン・テンプレートの単語と重ならないこと ---
_not_single = check_single_token_words(tokenizer.encode, WORD_VOCABULARY)
assert not _not_single, f"単一トークンでない単語: {_not_single}"
assert len(set(WORD_VOCABULARY)) == len(WORD_VOCABULARY)
_overlap = set(WORD_VOCABULARY) & template_words()
assert not _overlap, f"テンプレートの単語と重なる: {_overlap}"
assert not any("#" in w for w in WORD_VOCABULARY)  # 区切り記号の文字を含まない

# --- パイロットの評価入力(6.2 節)を同じ関数・同じ乱数シードで再生成する ---
PILOT_EVALUATION_INPUT_SPECS = (  # (乱数シード, 入力の数, 語彙の大きさ, 最小語数, 最大語数)
    (2016, 250, len(WORD_VOCABULARY), 3, 8),  # パイロット 1(T と学習率の走査、T = 32〜512)
    (1, 100, len(WORD_VOCABULARY), 3, 8),  # パイロット 2(本番の課題、T = 1024・2048)
    (1, 100, len(WORD_VOCABULARY), 4, 8),  # パイロット 2(課題の候補の比較)
    (1, 100, 32, 3, 8),
    (1, 100, 32, 4, 8),
    (1, 100, 16, 3, 8),
)
PILOT_EVALUATION_INPUTS = set()
for _seed, _count, _vocab, _lo, _hi in PILOT_EVALUATION_INPUT_SPECS:
    PILOT_EVALUATION_INPUTS |= set(
        sample_word_sequences(
            np.random.default_rng(_seed), _count, WORD_VOCABULARY[:_vocab], _lo, _hi
        )
    )


def build_dataset(seed: int, num_eval_inputs: int, num_train: int) -> dict:
    rng = np.random.default_rng(seed)
    eval_inputs = sample_word_sequences(
        rng, num_eval_inputs, WORD_VOCABULARY, MIN_WORDS, MAX_WORDS, exclude=PILOT_EVALUATION_INPUTS
    )
    train_inputs = sample_word_sequences(
        rng,
        num_train,
        WORD_VOCABULARY,
        MIN_WORDS,
        MAX_WORDS,
        exclude=set(eval_inputs) | PILOT_EVALUATION_INPUTS,
    )
    return {
        "eval_inputs": eval_inputs,
        "train": build_training_examples(train_inputs, rng),
        "eval": build_evaluation_examples(eval_inputs, rng),
        "eval_unseen": build_evaluation_examples(eval_inputs, rng, unseen=True),
    }


def encode_examples(examples, long: bool) -> list:
    return [
        encode_instruction_example(tokenizer.encode, tokenizer.decode, e.prompt(long), e.response())
        for e in examples
    ]


DATASET = build_dataset(DATA_SEED, NUM_EVAL_INPUTS, N_MAX)
TRAIN_EXAMPLES, EVAL_EXAMPLES = DATASET["train"], DATASET["eval"]
EVAL_UNSEEN_EXAMPLES = DATASET["eval_unseen"]
EVAL_INPUTS = DATASET["eval_inputs"]
TRAIN_ENCODED = {
    False: encode_examples(TRAIN_EXAMPLES, False),
    True: encode_examples(TRAIN_EXAMPLES, True),
}
EVAL_ENCODED = {
    False: encode_examples(EVAL_EXAMPLES, False),
    True: encode_examples(EVAL_EXAMPLES, True),
}
EVAL_UNSEEN_ENCODED = encode_examples(EVAL_UNSEEN_EXAMPLES, False)
DATA_SECONDS = time.time() - _t0_data
EVAL_SIZE = len(EVAL_EXAMPLES)  # M
EVAL_INPUT_INDEX = np.repeat(np.arange(NUM_EVAL_INPUTS), TASK_COUNT)  # 事例 -> 入力 x(クラスタ)

# --- 不変条件 ---
# (1) 評価集合は入力と課題の直積であり、学習データ・パイロットの評価入力と重複しない
assert [(e.words, e.task) for e in EVAL_EXAMPLES] == [
    (x, t) for x in EVAL_INPUTS for t in TASK_NAMES
]
assert [(e.words, e.task) for e in EVAL_UNSEEN_EXAMPLES] == [
    (e.words, e.task) for e in EVAL_EXAMPLES
]
assert len(set(EVAL_INPUTS)) == NUM_EVAL_INPUTS
_train_inputs = {e.words for e in TRAIN_EXAMPLES}
assert len(_train_inputs) == N_MAX, "学習データの入力に重複がある"
assert not (set(EVAL_INPUTS) & _train_inputs), "評価用の入力が学習データと重複する"
assert not (set(EVAL_INPUTS) & PILOT_EVALUATION_INPUTS), (
    "評価用の入力がパイロットの評価入力と重複する"
)
assert all(not e.unseen for e in TRAIN_EXAMPLES) and all(e.unseen for e in EVAL_UNSEEN_EXAMPLES)
# (2) 学習データの先頭 N 個(実験 C の水準)で課題が均等
for _n in ALL_C_LEVELS:
    _counts = {t: sum(e.task == t for e in TRAIN_EXAMPLES[:_n]) for t in TASK_NAMES}
    assert len(set(_counts.values())) == 1, (_n, _counts)
# (3) 応答部分のトークン列が短い水準・長い水準で完全に同一(学習・評価とも)
for _encoded in (TRAIN_ENCODED, EVAL_ENCODED):
    assert all(
        a.response_ids == b.response_ids
        for a, b in zip(_encoded[False], _encoded[True], strict=True)
    )
assert all(
    a.response_ids == b.response_ids
    for a, b in zip(EVAL_ENCODED[False], EVAL_UNSEEN_ENCODED, strict=True)
)
# (4) 連結したトークン列の復号が全文と一致すること(encode_instruction_example 内で確認済み)と、
#     別々の符号化が全文を一度に符号化した結果とも一致すること(境界をまたぐマージがない)
for _examples, _encoded in ((TRAIN_EXAMPLES, TRAIN_ENCODED), (EVAL_EXAMPLES, EVAL_ENCODED)):
    for _long in (False, True):
        for _e, _x in zip(_examples, _encoded[_long], strict=True):
            assert tokenizer.decode(list(_x.token_ids)) == _e.prompt(_long) + _e.response()
            assert list(_x.token_ids) == tokenizer.encode(_e.prompt(_long) + _e.response())
# (5) 全系列長 + 生成の上限が文脈長以下、応答部分の最大長 + 余裕 <= 生成の上限
_all_encoded = [x for d in (TRAIN_ENCODED, EVAL_ENCODED) for v in d.values() for x in v]
_all_encoded += EVAL_UNSEEN_ENCODED
MAX_TOTAL_LENGTH = max(len(x.token_ids) for x in _all_encoded)
MAX_PROMPT_LENGTH = max(x.prompt_length for x in _all_encoded)
MAX_RESPONSE_LENGTH = max(len(x.response_ids) for x in _all_encoded)
assert MAX_TOTAL_LENGTH <= CONTEXT_LENGTH
assert MAX_PROMPT_LENGTH + MAX_NEW_TOKENS <= CONTEXT_LENGTH
assert MAX_RESPONSE_LENGTH + 8 <= MAX_NEW_TOKENS

# --- 指示部分の長さの比(長い水準 / 短い水準、学習データでの平均) ---
PROMPT_TOKENS = {
    long: np.array([x.prompt_length - 1 for x in TRAIN_ENCODED[long]]) for long in (False, True)
}  # 予測対象になる指示部分のトークン数 |P|(先頭のトークンを除く)
RESPONSE_TOKENS_MEAN = float(np.mean([len(x.response_ids) for x in TRAIN_ENCODED[False]]))
LONG_PROMPT_RATIO = float(PROMPT_TOKENS[True].mean() / PROMPT_TOKENS[False].mean())
assert LONG_PROMPT_RATIO_RANGE[0] <= LONG_PROMPT_RATIO <= LONG_PROMPT_RATIO_RANGE[1], (
    LONG_PROMPT_RATIO
)

# --- c*(6.1 節) ---
_answers_by_input = [
    [e.answer for e in EVAL_EXAMPLES[i * TASK_COUNT : (i + 1) * TASK_COUNT]]
    for i in range(NUM_EVAL_INPUTS)
]
C_STAR = compute_instruction_agnostic_upper_bound(_answers_by_input)
C_STAR_MAX_MULTIPLICITY_SUM = sum(
    max(sum(a == b for b in answers) for a in answers) for answers in _answers_by_input
)  # sum_x max_y n_x(y)(付録 A の再計算用)
assert math.isclose(
    C_STAR, C_STAR_MAX_MULTIPLICITY_SUM / (NUM_EVAL_INPUTS * TASK_COUNT), rel_tol=1e-12
)
assert C_STAR == 1 / TASK_COUNT  # 4 課題の応答はすべての入力で互いに異なる

print(
    f"単語の語彙: {len(WORD_VOCABULARY)} 語(すべて単一トークン、テンプレートの単語と重ならない)、"
    f"指示文: 学習用 {sum(map(len, SEEN_INSTRUCTIONS.values()))} 個・未見 "
    f"{sum(map(len, UNSEEN_INSTRUCTIONS.values()))} 個、前置き {len(PREAMBLES)} 個"
)
print(
    f"学習データ {len(TRAIN_EXAMPLES):,} 事例、評価集合 {EVAL_SIZE} 事例(|X| = {NUM_EVAL_INPUTS} x K = {TASK_COUNT})、"
    f"パイロットの評価入力 {len(PILOT_EVALUATION_INPUTS)} 個と重複なし、生成と符号化 {DATA_SECONDS:.1f}s"
)
print(
    f"指示部分の予測対象のトークン数 |P|(1 事例あたりの平均): 短い水準 {PROMPT_TOKENS[False].mean():.2f}、"
    f"長い水準 {PROMPT_TOKENS[True].mean():.2f}、比 {LONG_PROMPT_RATIO:.3f}"
    f"(応答部分のトークン数の平均 {RESPONSE_TOKENS_MEAN:.2f})"
)
print(
    f"最大長: 全系列 {MAX_TOTAL_LENGTH}、指示部分 {MAX_PROMPT_LENGTH}(+ 生成の上限 {MAX_NEW_TOKENS} <= "
    f"文脈長 {CONTEXT_LENGTH})、応答部分 {MAX_RESPONSE_LENGTH}(終端記号を含む)"
)
print(f"c* = {C_STAR:.4f}(sum_x max_y n_x(y) = {C_STAR_MAX_MULTIPLICITY_SUM}、|X| K = {EVAL_SIZE})")
print("\n--- 事例(評価集合の先頭の入力、短い水準)---")
for _e in EVAL_EXAMPLES[:TASK_COUNT]:
    print(repr(_e.prompt(False) + _e.response()))
print("\n--- 同じ事例の長い水準 ---")
print(EVAL_EXAMPLES[0].prompt(True) + EVAL_EXAMPLES[0].response())
print("\n--- 応答部分のトークン(先頭の事例)---")
print([tokenizer.decode([i]) for i in EVAL_ENCODED[False][0].response_ids])
```

    単語の語彙: 454 語(すべて単一トークン、テンプレートの単語と重ならない)、指示文: 学習用 16 個・未見 8 個、前置き 4 個
    学習データ 65,536 事例、評価集合 1000 事例(|X| = 250 x K = 4)、パイロットの評価入力 726 個と重複なし、生成と符号化 17.6s
    指示部分の予測対象のトークン数 |P|(1 事例あたりの平均): 短い水準 33.06、長い水準 132.37、比 4.004(応答部分のトークン数の平均 12.26)
    最大長: 全系列 165、指示部分 150(+ 生成の上限 32 <= 文脈長 256)、応答部分 22(終端記号を含む)
    c* = 0.2500(sum_x max_y n_x(y) = 250、|X| K = 1000)
    
    --- 事例(評価集合の先頭の入力、短い水準)---
    '### Instruction:\nList the words from the last one to the first one.\nWords: agreement green billion software genus member novel community\n### Response: community novel member genus software billion green agreement\n### End'
    '### Instruction:\nWrite every word two times in a row.\nWords: agreement green billion software genus member novel community\n### Response: agreement agreement green green billion billion software software genus genus member member novel novel community community\n### End'
    '### Instruction:\nKeep only the words in odd positions.\nWords: agreement green billion software genus member novel community\n### Response: agreement billion genus novel\n### End'
    '### Instruction:\nShift every word one place to the left and wrap the first word around to the end.\nWords: agreement green billion software genus member novel community\n### Response: green billion software genus member novel community agreement\n### End'
    
    --- 同じ事例の長い水準 ---
    ### Instruction:
    You are a careful helper who works with short lists of English words. In this exercise you will read a short description of an operation and a list of words. Please read the description slowly and make sure that you understand it before you begin to write. Your answer should contain only words from the list, written in lowercase letters and separated by single spaces, without extra comments or punctuation. List the words from the last one to the first one.
    Words: agreement green billion software genus member novel community
    ### Response: community novel member genus software billion green agreement
    ### End
    
    --- 応答部分のトークン(先頭の事例)---
    [' community', ' novel', ' member', ' genus', ' software', ' billion', ' green', ' agreement', '\n', '#', '#', '#', ' E', 'nd']


### 5.5 生成・採点・損失のハーネスの確認

- **生成**: `greedy_generate_until_stop()`で停止しない場合(停止文字列が現れない文字列を指定)の出力が、
  1 事例ずつ`GPTLanguageModel.generate(temperature=0.0, use_cache=True)`で生成した結果と一致すること。
  指示部分の長さごとにまとめたバッチでの生成は、行列積の形が変わるため浮動小数点の丸めが変わりうるので、
  1 事例ずつの生成との一致率を印字する(評価集合の先頭 256 事例。アサーションは一致率 0.98 以上)。
- **採点**: 形式の遵守・完全一致の判定を、手で作った文字列で確かめる(空白の正規化、区切り記号の混入、
  終端記号の欠落)。
- **損失**: `compute_instruction_tuning_loss()`の値を、トークンごとの負の対数尤度から直接計算した
  $\mathcal{L}_{\mathrm{mask}}$・$\mathcal{L}_{\mathrm{unmask}}$ と照合する。損失マスクなしの勾配が、損失マスクありの
  勾配と指示部分の項の和に分解されること(3.3 節)も確かめる。
- **008 のモデルの出力**: 微調整前のモデルに指示を与えた生成例を印字する(3.1 節)。


```python
_t0_harness = time.time()
# --- 生成: 停止しない場合に generate() と一致 ---
_probe = EVAL_ENCODED[False][:8]
_prompts = [list(x.token_ids[: x.prompt_length]) for x in _probe]
_no_stop = greedy_generate_until_stop(
    base_model, _prompts, tokenizer.decode, "\x00never\x00", 12, device, batch_size=1
)
for _p, _g in zip(_prompts, _no_stop, strict=True):
    _reference = base_model.generate(
        torch.tensor([_p], device=device), 12, temperature=0.0, use_cache=True
    )[0, len(_p) :].tolist()
    assert _g == _reference, "停止しない場合の出力が generate() と一致しない"
print(
    f"停止しない場合の出力が generate(temperature=0, use_cache=True) と一致({len(_probe)} 事例): OK"
)

_prompts_all = [list(x.token_ids[: x.prompt_length]) for x in EVAL_ENCODED[False][:256]]
_one_by_one = greedy_generate_until_stop(
    base_model, _prompts_all, tokenizer.decode, END_MARKER, MAX_NEW_TOKENS, device, batch_size=1
)
_batched = greedy_generate_until_stop(
    base_model,
    _prompts_all,
    tokenizer.decode,
    END_MARKER,
    MAX_NEW_TOKENS,
    device,
    GENERATION_BATCH_SIZE,
)
_agreement = float(np.mean([a == b for a, b in zip(_one_by_one, _batched, strict=True)]))
assert _agreement >= 0.98, _agreement
print(
    f"指示部分の長さごとのバッチでの生成と 1 事例ずつの生成の一致率(微調整前のモデル): {_agreement:.4f}"
)

# --- 採点 ---
_answer = ("cat", "dog")
for _text, _expected in (
    (" cat dog\n### End", (True, True)),
    ("  cat   dog ### End", (True, True)),  # 空白の正規化
    (" cat\ndog\n### End\n### Instruction:", (True, True)),  # 終端記号より後は見ない
    (" dog cat\n### End", (True, False)),
    (" cat dog", (False, False)),  # 終端記号がない
    (" cat dog\n### Response: x\n### End", (False, False)),  # 終端記号より前に他の区切り記号
    (" cat dog ##", (False, False)),
):
    assert score_generated_response(_text, _answer) == _expected, (_text, _expected)
print("採点の判定(空白の正規化・区切り記号の混入・終端記号の欠落): OK")

# --- 損失: 直接計算との照合と勾配の分解 ---
_batch = [TRAIN_ENCODED[True][i] for i in range(4)]
_tokens, _prompt_mask, _response_mask = collate_instruction_batch(_batch)
_tokens = _tokens.to(device)
_probe_model = copy.deepcopy(base_model)
torch.manual_seed(0)
apply_lora(_probe_model, LORA_TARGET_MODULES, rank=LORA_RANK, alpha=LORA_ALPHA)
with torch.no_grad():
    for _module in _probe_model.modules():
        if hasattr(_module, "lora_b"):
            _module.lora_b.normal_(
                0.0, 0.02
            )  # B = 0 のままだと A の勾配が 0 になるので乱数を入れる
_trainable = [p for p in _probe_model.parameters() if p.requires_grad]


def _gradients(include_prompt_loss: bool, part: str = "loss"):
    _probe_model.zero_grad(set_to_none=True)
    logits = _probe_model(_tokens)
    parts = compute_instruction_tuning_loss(
        logits, _tokens, _prompt_mask.to(device), _response_mask.to(device), include_prompt_loss
    )
    if part == "prompt":  # (1 / |R|) sum_{t in P} ell_t の勾配
        token_losses = F.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB_SIZE), _tokens[:, 1:].reshape(-1), reduction="none"
        ).view(_tokens[:, 1:].shape)
        value = (token_losses * _prompt_mask.to(device)).sum() / _response_mask.sum().to(device)
    else:
        value = parts["loss"]
    value.backward()
    return parts, [p.grad.detach().clone() for p in _trainable]


_mask_parts, _grad_mask = _gradients(False)
_unmask_parts, _grad_unmask = _gradients(True)
_, _grad_prompt = _gradients(True, part="prompt")
with torch.no_grad():
    _logits = _probe_model(_tokens)
    _direct = F.cross_entropy(
        _logits[:, :-1].reshape(-1, VOCAB_SIZE), _tokens[:, 1:].reshape(-1), reduction="none"
    ).view(_tokens[:, 1:].shape)
_r, _p = _response_mask.to(device), _prompt_mask.to(device)
torch.testing.assert_close(_mask_parts["loss"], (_direct * _r).sum() / _r.sum())
torch.testing.assert_close(
    _unmask_parts["loss"], ((_direct * _r).sum() + (_direct * _p).sum()) / _r.sum()
)
assert int(_r.sum()) == sum(len(x.response_ids) for x in _batch)
assert int(_p.sum()) == sum(x.prompt_length - 1 for x in _batch)
for _gu, _gm, _gp in zip(_grad_unmask, _grad_mask, _grad_prompt, strict=True):
    torch.testing.assert_close(_gu, _gm + _gp, atol=1e-6, rtol=1e-4)
print(
    "損失(mask・unmask)が直接計算と一致、|R|・|P| が事例のトークン数と一致、"
    "unmask の勾配 = mask の勾配 + (1/|R|) sum_P の勾配: OK"
)

# --- 微調整前のモデルの出力例 ---
print("\n--- 微調整前の 008 のモデルの生成例(短い水準)---")
for _x, _g in list(zip(EVAL_EXAMPLES, _batched, strict=False))[:TASK_COUNT]:
    print(f"[{_x.task}] 正解 {' '.join(_x.answer)!r} -> 生成 {tokenizer.decode(_g)!r}")
HARNESS_SECONDS = time.time() - _t0_harness
print(f"\nこのセルの実行時間: {HARNESS_SECONDS:.1f}s")
```

    停止しない場合の出力が generate(temperature=0, use_cache=True) と一致(8 事例): OK
    指示部分の長さごとのバッチでの生成と 1 事例ずつの生成の一致率(微調整前のモデル): 1.0000
    採点の判定(空白の正規化・区切り記号の混入・終端記号の欠落): OK
    損失(mask・unmask)が直接計算と一致、|R|・|P| が事例のトークン数と一致、unmask の勾配 = mask の勾配 + (1/|R|) sum_P の勾配: OK
    
    --- 微調整前の 008 のモデルの生成例(短い水準)---
    [reverse] 正解 'community novel member genus software billion green agreement' -> 生成 ' The first known record of the first record of the first time since the first time since the first time since the first time since the first time since the end of'
    [repeat_twice] 正解 'agreement agreement green green billion billion software software genus genus member member novel novel community community' -> 生成 ' The first known paper of the first known lesbian of the first major major general in the United States to be a major-color of the'
    [odd_positions] 正解 'agreement billion genus novel' -> 生成 ' The first known record of the first record of the first record of the first record of the first record of the first time since the first time since the first time'
    [rotate_left] 正解 'green billion software genus member novel community agreement' -> 生成 ' The first known paper of the first known lesbian of the first time in the first half of the decade was published in the United States by'
    
    このセルの実行時間: 49.9s


### 5.6 決定的な実行の確認

`torch.use_deterministic_algorithms(True)`のもとで、学習(順伝播・逆伝播・AdamW の更新)・
教師強制の評価・生成の経路を一通り実行し、決定的な実装を持たない演算で例外にならないこと、同じ設定で 2 回
実行した結果が bit 単位で一致することを確かめる(数ステップの短い学習で確かめる)。


```python
def _short_run(seed: int) -> tuple:
    model = copy.deepcopy(base_model)
    torch.manual_seed(seed)
    apply_lora(model, LORA_TARGET_MODULES, rank=LORA_RANK, alpha=LORA_ALPHA)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    history = train_instruction_tuning(
        model,
        TRAIN_ENCODED[True],
        make_epoch_batches(len(TRAIN_ENCODED[True]), 4, BATCH_SIZE, seed),
        optimizer,
        include_prompt_loss=True,
        device=device,
    )
    evaluation = evaluate_instruction_negative_log_likelihood(
        model, EVAL_ENCODED[True][:32], device
    )
    generated = greedy_generate_until_stop(
        model,
        [list(x.token_ids[: x.prompt_length]) for x in EVAL_ENCODED[True][:32]],
        tokenizer.decode,
        END_MARKER,
        MAX_NEW_TOKENS,
        device,
        GENERATION_BATCH_SIZE,
    )
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    return history, evaluation, generated, state


_first, _second = _short_run(0), _short_run(0)
assert (
    _first[0]["loss"] == _second[0]["loss"]
    and _first[0]["gradient_norm"] == _second[0]["gradient_norm"]
)
assert all(np.array_equal(_first[1][k], _second[1][k]) for k in _first[1])
assert _first[2] == _second[2]
assert all(torch.equal(_first[3][k], _second[3][k]) for k in _first[3])
print(
    f"決定的な実行({device}、torch.are_deterministic_algorithms_enabled() = "
    f"{torch.are_deterministic_algorithms_enabled()}): 学習 4 ステップ・教師強制の評価・"
    "生成を 2 回実行して、損失・勾配ノルム・LoRA の重み・評価値・生成が bit 単位で一致。例外なし: OK"
)
```

    決定的な実行(cuda、torch.are_deterministic_algorithms_enabled() = True): 学習 4 ステップ・教師強制の評価・生成を 2 回実行して、損失・勾配ノルム・LoRA の重み・評価値・生成が bit 単位で一致。例外なし: OK




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/04_alignment/016_supervised_fine_tuning.ipynb
