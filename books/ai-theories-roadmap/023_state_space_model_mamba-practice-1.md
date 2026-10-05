---
title: "State Space Model / Mamba / State Space Model and Mamba(実装・実験編 1/6)"
---

この記事は後編(実装・実験編 1/6)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/023_state_space_model_mamba-theory)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/023_state_space_model_mamba-practice-2)。

## 4. 実装方針 / Implementation Policy

**`src/`に切り出す(スクラッチ実装、本トピックで新規作成)**:

- `src/layers/mamba.py`: 離散化`discretize_state_space()`(ゼロ次ホールドと簡略形を引数で切り替え。対角の $A$ では要素ごとの式)、逐次ループによる走査`selective_scan()`(`Ā_t`と`B̄_t x_t`を全時刻ぶん一括で計算し、ループの中は状態の更新 $h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$ と出力 $y_t = C_t h_t$ だけ。計算は`torch.autocast`の中でも FP32 以上。勾配が要るときは逆伝播を手で書いた`torch.autograd.Function`で包む)、1 ステップ更新`selective_scan_step()`、`MambaBlock`(線形射影による拡大、因果的な depthwise の 1 次元畳み込み、SiLU、選択的な状態空間モデル、ゲートの分岐、出力の射影。引数`selective`で $\Delta$・$B$・$C$ が入力の関数か、入力によらない学習可能なパラメータかを切り替える。推論用の`step()`、状態`MambaState`の受け渡し、activation checkpointing)。
- `src/models/mamba.py`: `MambaLanguageModel`(埋め込み → [RMSNorm → `MambaBlock` → 残差接続] の積層 → RMSNorm → 出力層)と、状態を持ち回る生成。正規化前置・埋め込みと出力層の重み共有・埋め込みの初期化(標準偏差 0.02)は 008 の`GPTLanguageModel`に揃えた。**揃えなかった点**: 位置エンコーディングを持たない、順伝播ネットワークを持たない(`MambaBlock`が注意機構と順伝播ネットワークの役割を担う)、層の初期化は各層の既定のまま(公式の実装が行う出力射影の重みの縮小は行わない)。
- `src/data/synthetic_sequence.py`: selective copying と induction heads の事例の生成(シードから決定的に生成、系列長を引数に取る)。
- `src/training/sequence_task.py`: ステップごとに新しい事例を生成する学習ループ`train_sequence_task()`(007 の`AdamW`・warmup + cosine・gradient clipping を再利用)と、評価関数(正解の履歴を与える評価、貪欲な生成による採点。生成した答えそのものを返す)。コーパスからランダムな区間を切り出す既存の`train_language_model()`は、合成課題には使えないので別に用意した。

**既存モジュールの再利用(変更なし)**: 注意機構`MultiHeadAttention`・`create_causal_mask`(001)と KV キャッシュ`KeyValueCache`(010)は実験 B、RoPE・RMSNorm・SwiGLU の`GPTLanguageModel`(008 と同じ構成)は実験 C・D、`train_language_model()`と`evaluate_bits_per_byte()`は実験 D、`AdamW`・warmup + cosine・`DynamicLossScaler`、`fit_power_law_exponent()`、`paired_bootstrap_ratio_of_sums()`をそのまま使う。**既存モジュールのコードは変更していない**(新しいファイルの追加のみ)ので、021 までと 022 の出力は変わらない。

**ノートブック内に直接書く(023 固有)**: 条件の定義、評価集合の生成、単体テストと不変条件の確認、学習率の較正、実行計画の選択、判定、可視化、スケーリングの計測、実験 B の時間計測、実験 D のコーパスの取得と評価窓。

**アップロード方針**: 本トピックで学習するモデルは、すべて条件間の比較または動作確認のためのものであり、後続トピックの入力にも、読者が単体で取得する対象にもならない。保存もアップロードもしない(アップロードのセルも置かない)。

**生成物の置き場所**:

| 生成物 | 置き場所 |
|---|---|
| コーパス(Hub から取得できなかった場合の Wikipedia API からの取得結果) | `.cache/wikipedia_en/`(データ源で命名。決定的に再取得できる) |
| Hugging Face Hub から取得したコーパス・トークナイザ・008 のモデル | `huggingface_hub`の既定のキャッシュ(Colab ではセッションの終了とともに破棄される) |
| 合成課題の評価集合・学習データ | メモリ上のみ(シードから決定的に生成する。学習データは保存しない) |
| 符号化したトークン列・評価窓(実験 D) | メモリ上のみ |
| 学習したモデル(実験 A・C・D) | メモリ上のみ(評価の直後に破棄する) |
| 学習の履歴・生成した答え・正解の混同行列の元になる記録 | メモリ上のみ(判定と図に使う) |
| 判定の記録 | 判定と前提条件を計算した各セルの出力(6.7〜6.9・6.12 節) |

**外部からの取得**(Colab のセットアップセルのリポジトリの取得と依存関係のインストールを除く):

| 取得するもの | リポジトリ | 用途 |
|---|---|---|
| コーパス(`corpus.txt`・`metadata.json`) | `kojikojiprg/ai-theories-corpus-en-pretraining`(Dataset、356 記事) | 実験 D: 008 と同じ英語 Wikipedia のコーパス |
| トークナイザ(`tokenizer.json`) | `kojikojiprg/ai-theories-tokenizer-en` | 実験 D: 008 の英語の BPE(Byte Pair Encoding)トークナイザ(語彙サイズ 8192) |
| 008 の学習済みモデル(`config.json`・`model_state.pt`、`main`) | `kojikojiprg/ai-theories-small-gpt-en` | 実験 D: 比較の相手(学習し直さない) |

実験 A・B・C は外部のデータを使わない(合成課題と乱数の入力のみ)。実験 D を省く計画が選ばれた場合は、上の 3 つのいずれも取得しない。

## 5. 実装 / Implementation

### 5.1 環境セットアップ(Google Colab)

`SMOKE_TEST`(スモークテストか本番か)はこのセルでのみ切り替える。実行環境はこのセルで 1 回だけ印字する。

**精度と決定性**: 実験 A・C の学習と、すべての評価・実験 B の計測は FP32 で行う。実験 D の学習は、CUDA のときは FP16 の`torch.autocast`と動的損失スケーリング(011 の`DynamicLossScaler`)で行い、走査は`torch.autocast`の中でも FP32 で計算する(3.8 節。5.5 節で確かめる)。それ以外のデバイス(ローカルの MPS・CPU)では FP32 で行う。`torch.use_deterministic_algorithms(True)`は使わない。同じシードの学習を繰り返しても結果が bit 単位では一致しないことがあるが、条件間の対応付け(同じシードの条件どうしで学習データの流れを揃えること)は乱数の生成器によって決まり、演算の決定性には依存しない(5.5・6.11 節で確かめる)。


```python
# 環境セットアップ(Google Colab)
import os
import sys
import time

SMOKE_TEST = False  # Claude Code はこの True 側のみ実行する(Colab T4 では False に切り替える)

NOTEBOOK_START_TIME = time.time()

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not os.path.isdir("/content/ai-theories"):  # 2 回目の実行では clone し直さない
        !git clone https://github.com/kojikojiprg/ai-theories.git /content/ai-theories
    %cd /content/ai-theories
    !pip install uv -q
    !uv pip install --system -r requirements.txt

# インストールで版が入れ替わったパッケージの古い版がメモリに残っていないかを、torch などを import する前に確かめる。
# Colab では食い違いがあればカーネルを終了する(再接続後、もう一度「すべてのセルを実行」する)
from src.utils.environment import check_preloaded_package_versions  # noqa: E402

check_preloaded_package_versions(on_mismatch="restart" if IN_COLAB else "raise")

import torch  # noqa: E402

from src.utils.environment import print_execution_environment  # noqa: E402

device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
USE_FP16_AUTOCAST = device.type == "cuda"  # FP16 の autocast と動的損失スケーリングは、CUDA での実験 D の学習のみ
execution_environment = print_execution_environment(device)
print(
    f"SMOKE_TEST={SMOKE_TEST}、実験 A・C の学習の精度 FP32、実験 D の学習の精度 "
    f"{'FP16 の autocast + 動的損失スケーリング(走査は FP32)' if USE_FP16_AUTOCAST else 'FP32'}、評価は FP32、"
    f"決定的な演算の強制: {torch.are_deterministic_algorithms_enabled()}"
)
```

    /content/ai-theories
    [2mUsing Python 3.13.15 environment at: /usr[0m
    [2mChecked [1m60 packages[0m [2min 236ms[0m[0m
    読み込み済みのパッケージの版の検査(requirements.txt): 食い違いなし。検査した配布物 11 個(certifi==2026.7.22, cycler==0.12.1, fonttools==4.63.0, kiwisolver==1.5.0, matplotlib==3.11.1, numpy==2.5.2, packaging==26.3, pillow==12.3.0, pyparsing==3.3.2, python-dateutil==2.9.0.post0, six==1.17.0)、未読み込みのため検査しなかった配布物 44 個、__version__ がないため比べなかった配布物 0 個(なし)、表記の違いで誤検出するため比べなかった配布物 0 個(なし)
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
      コミット / git commit                  : d37a14f7e6f7ed75fe13db2fe262e68cf9a9b7e0
      未コミットの変更 / uncommitted changes : なし
      実行日時 (UTC)                         : 2026-10-05T03:15:25+00:00
    SMOKE_TEST=False、実験 A・C の学習の精度 FP32、実験 D の学習の精度 FP16 の autocast + 動的損失スケーリング(走査は FP32)、評価は FP32、決定的な演算の強制: False


### 5.2 インポートと共通の関数


```python
import functools
import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional

from src.data.synthetic_sequence import (
    IGNORE_INDEX,
    InductionHeadsTask,
    SelectiveCopyingTask,
    generate_induction_heads,
    generate_selective_copying,
    induction_heads_targets,
    make_rng,
)
from src.data.text import (
    encode_corpus,
    load_wikipedia_corpus_with_fallback,
    make_evaluation_windows,
    split_train_val_text,
)
from src.data.tokenizer import load_bpe_id_tokenizer_from_hub
from src.generation.cache import KeyValueCache
from src.layers.attention import MultiHeadAttention, create_causal_mask
from src.layers.feedforward import SwiGLUFeedForwardNetwork
from src.layers.mamba import MambaBlock, discretize_state_space, selective_scan
from src.layers.normalization import RMSNorm
from src.layers.positional_encoding import RotaryPositionEmbedding
from src.models.gpt import GPTLanguageModel
from src.models.mamba import MambaLanguageModel
from src.training.optimizer import AdamW
from src.training.precision import DynamicLossScaler
from src.training.schedule import compute_warmup_cosine_learning_rate
from src.training.sequence_task import (
    evaluate_induction_heads,
    evaluate_selective_copying,
    teacher_forced_evaluation,
    train_sequence_task,
)
from src.training.trainer import evaluate_bits_per_byte, train_language_model
from src.utils.reporting import dumps_compact_json
from src.utils.statistics import (
    count_non_embedding_parameters,
    fit_power_law_exponent,
    paired_bootstrap_ratio_of_sums,
)

ROOT = Path.cwd()
WIKIPEDIA_CACHE_DIR = ROOT / ".cache" / "wikipedia_en"  # 外部から取得したコーパス(データ源で命名)
CORPUS_REPO_ID = "kojikojiprg/ai-theories-corpus-en-pretraining"
TOKENIZER_REPO_ID = "kojikojiprg/ai-theories-tokenizer-en"
REFERENCE_MODEL_REPO_ID = "kojikojiprg/ai-theories-small-gpt-en"  # 008 の学習済みモデル(実験 D の比較の相手)
MANIFEST_PATH = ROOT / "src" / "data" / "wikipedia_manifests" / "en_006_pretraining.json"
RUN_TAG = "[スモークテスト] " if SMOKE_TEST else ""
PLOT_TAG = "[smoke test] " if SMOKE_TEST else ""


def sync_device() -> None:
    # 時間計測の直前・直後に、非同期に実行される GPU の処理の完了を待つ。
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_device_cache() -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def timed_call(fn) -> float:
    sync_device()
    start = time.time()
    fn()
    sync_device()
    return time.time() - start


def rounded(values, digits: int = 4):
    # 印字用: 入れ子のリスト・配列を丸めたリストにする
    return np.round(np.asarray(values, dtype=np.float64), digits).tolist()


precondition_status: dict[str, bool] = {}  # 前提条件の成否(6.1 節で宣言、各節で記録)
```

### 5.3 定数と水準の定義(`SMOKE_TEST`の配線)

水準の定義をこの 1 箇所に集約する。

**縮小の規則**: スモークテストは、本番と **同じモデル・同じ課題・同じ条件・同じ実行計画の構造** で、次の量だけを縮小する。

- **実験 C の学習ステップ数 $T_C$ と実験 A の学習ステップ数 $T_A$**: 本番は $T_C = 2000$、$T_A = 4000$(削る段階 6 では両方を半分にして 1000・2000)、スモークテストは $T_C = 24$、$T_A = 48$(同 12・24)。**$T_A / T_C = 2$ の比と、「削った後 = 半分」の関係を、本番とスモークテストで保つ。**
- **シード数**: 本番は 5(削った段階で 3)、スモークテストは 3(同 2)。「削る前 > 削った後 $\ge$ 2」の順序関係を保つ。
- **評価の事例数**: 本番は評価集合 1000 系列(系列長ごと)・較正用の集合 500 系列・途中の評価の部分集合 250 系列、スモークテストは 100・100・50。
- **実験 D の学習ステップ数の候補**: 本番 $(2181, 1090, 545)$、スモークテスト $(16, 8, 4)$。どちらも大きい順で、隣どうしの比が 2 の等比(本番は $2181 / 2^j$ を丸めた値)という構造を保つ。コーパスは本番が全体、スモークテストは先頭の 60 万文字。
- **実験 B の水準の数と反復**: 本番は系列長 $(512, 1024, 2048, 4096, 8192)$ の 5 水準・掃引 8 回・反復 8 回、スモークテストは先頭の 3 水準・掃引 3 回・反復 3 回。公比 2 の等比、下端が同じ(注意機構の二次の項が線形の項を上回る系列長以上、6.1 節)という構造を保つ。
- **ブートストラップの反復回数**: 本番 10,000 回、スモークテスト 1,000 回。

スモークテストはステップ数が極端に少ないので、学習は進まず、前提条件は成立しない見込みである(コードの経路の確認が目的)。**較正の格子の中心・前提条件の閾値・検出力の確認は、6.2 節のパイロットで本番と同じステップ数で行った。**

**縮小しないもの**: モデルの構成、課題の定義(語彙・系列長・データのトークンの数)、バッチサイズ、条件の対応表、学習率の較正の格子と拡張の規則、学習率のスケジュールの形、途中の評価の位置の決め方(学習ステップ数に対する割合)、実行計画の表の構造と優先順位、前提条件と判定の閾値、スケーリングの計測点、実験 C で評価する系列長。

**テスト専用の上書き**: 環境変数`AI_THEORIES_FORCE_PLAN`(0〜6)が設定されているときのみ、6.4 節で見積もりによる計画の選択の代わりにその計画を使う(スモークテストで下位の計画の経路を確かめるため)。本番では受け付けず、5.6 節のセルで停止する。このノートブックの出力は、上書きなしの実行のものである。

**実効水準の照合**: このセルで印字した水準(ステップ数・シード数・計画の表)が、実際の学習で使われた値と一致することを 6.11 節のアサーションで確かめる(`SMOKE_TEST`の配線漏れの検出)。


```python
# --- 全水準で共通の定数(本番実行前に宣言し、SMOKE_TEST で変えない) ---
# 合成課題(実験 A・C)のモデル。Mamba(条件 A1・A2・C1)と Transformer(条件 C2)は、層数 2、隠れ次元 64 で揃える
SYN_D_MODEL, SYN_NUM_LAYERS = 64, 2
MAMBA_STATE_DIM, MAMBA_EXPAND, MAMBA_CONV_KERNEL = 16, 2, 4  # Mamba の原論文の既定値
TRANSFORMER_NUM_HEADS = 4
TRANSFORMER_D_FF = 84  # SwiGLU の中間次元。非埋め込みパラメータ数を C1 の Mamba に揃える丸め(1 だけ増やすと 2 層で 384 個増える。5.4 節で確かめる)
# 実験 A: selective copying(原論文は系列長 4096・記憶するトークン 16 個・語彙 16。本ノートブックは縮小する、6.1 節)
TASK_A = SelectiveCopyingTask(num_symbols=8, num_data=8, input_length=40)
BATCH_A = 32
# 実験 C: induction heads(原論文は学習時の系列長 256。本ノートブックは縮小する、6.1 節)
TASK_C = InductionHeadsTask(num_symbols=16)
C_TRAIN_LENGTH = 32
C_LENGTH_MULTIPLIERS = (1, 2, 4, 8, 16)  # 評価する系列長 = L_train x 倍率(等比)
C_TEST_MULTIPLIER = 4  # 判定に使う L_test = 4 x L_train
BATCH_C = 64
# 学習(合成課題)。実験 A の学習ステップ数 T_A は実験 C の T_C の整数倍で、倍率は 6.2 節の追加パイロットで決めた
STEPS_RATIO_A_TO_C = 2
WARMUP_RATIO = 0.1
MIN_LEARNING_RATE_RATIO = 0.01
GRADIENT_CLIP_THRESHOLD = 1.0
SYN_WEIGHT_DECAY = 0.0
LOSS_WINDOW_FRACTION = 0.05  # 「最初・最後の区間」= 全ステップの 5%
# 学習率の較正(6.1 節): 中心は 6.2 節のパイロットで決めた。格子 = 中心 x {1/2, 1, 2}(公比 2)
LR_CENTER = {"A1": 4.0e-3, "A2": 1.0e-3, "C1": 3.2e-2, "C2": 8.0e-3}
LR_GRID_MULTIPLIERS = (0.5, 1.0, 2.0)
LR_GRID_RATIO = 2.0
CALIBRATION_POINTS_WORST = len(LR_GRID_MULTIPLIERS) + 1  # 格子 3 点 + 拡張 1 点(拡張は 1 回のみ)
# 途中の評価の位置(学習ステップ数に対する割合、初期に密な等比の間隔)
EVAL_FRACTIONS = (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
EVAL_TOKENS_PER_BATCH = 16384  # 評価の 1 バッチのトークン数(系列長に反比例して窓の数を決める)
# 前提条件と判定(6.1 節)
A_ROOM_MAX = 0.90  # 実験 A の P-A3: 条件 2 の正解率(シード平均)の上限(改善の余地)
C_MASTERY_MIN = 0.90  # 実験 C の P-C2: 学習時の系列長での正解率(シード平均)の下限
SIGMA_MULTIPLIER = 2.0  # 判定の閾値は対比量の標準偏差の 2 倍
# 実験 B(計算量のべき指数、6.1 節)
B_D_MODEL, B_NUM_HEADS = 128, 4
B_WARMUP_REPEATS = 3
P0A_MAX_DRIFT_SIGMA = 2.0  # 前提条件 P-B1: 先頭 3 反復と末尾 3 反復の平均の差 <= 全反復の標準偏差の 2 倍
P0B_MAX_RELATIVE_STDERR = 0.05  # 前提条件 P-B2: 平均の標準誤差 <= 平均の 5%
B_DECODE_CONTEXTS = (512, 2048, 8192)  # 診断量: 推論時の 1 トークンあたりの時間を測る文脈の長さ
# 実験 D(言語モデリングの動作確認): 008 のレシピ
D_VOCAB_SIZE, D_D_MODEL, D_NUM_LAYERS = 8192, 256, 7  # 非埋め込みパラメータ数を 008 のモデルにおおよそ揃える層数
D_SEQUENCE_LENGTH, D_BATCH_SIZE = 256, 32
D_LEARNING_RATE, D_GRADIENT_CLIP_THRESHOLD, D_WEIGHT_DECAY = 1.2e-3, 0.6341, 0.1  # 008 の較正の結果
D_VALIDATION_RATIO = 0.05
D_SEED = 42
D_INIT_LOSS_SCALE, D_LOSS_SCALE_GROWTH_INTERVAL = 2.0**16, 2000
D_PROMPTS = ("The history of", "In mathematics, a function", "The city was founded")
D_GENERATION_TOKENS = 48
# 条件の種類(A1: 選択的な Mamba、A2: 非選択の Mamba、C1: Mamba、C2: Transformer)
KINDS = ("A1", "A2", "C1", "C2")
# スケーリングの計測点(6.3 節)。水準によらず同じ
SCALING_STEP_COUNTS = (8, 16, 32)
SCALING_WARMUP_STEPS = 4
D_SCALING_STEP_COUNTS = (2, 4, 8)
D_SCALING_WARMUP_STEPS = 1
D_EVAL_FRACTIONS = (1 / 8, 1 / 4, 1 / 2, 1.0)  # 実験 D の途中の評価の位置(判定を伴わない観察)
# 乱数シード(学習のシード s: モデルの初期化は種類ごとの基準 + s、学習データの流れは共通の基準 + s)
INIT_SEED_BASE = {"A1": 23_100, "A2": 23_100, "C1": 23_300, "C2": 23_400}
DATA_SEED_BASE = {"A": 23_200, "C": 23_500}
EVAL_SEED_A, CALIBRATION_SEED_A = 23_700, 23_710
EVAL_SEED_C, CALIBRATION_SEED_C = 23_800, 23_810  # 評価集合は系列長ごとに EVAL_SEED_C + 倍率
BOOTSTRAP_SEED = 23_600
CALIBRATION_SEED_INDEX = 90  # 学習率の較正専用のシード(実験のシード 0〜4 と共有しない)
TIMING_SEED_INDEX = 91  # スケーリングの計測専用のシード
SESSION_BUDGET_SECONDS = 120 * 60  # 1 セッションの予算(T4 で 120 分)
REPORTING_MARGIN_SECONDS = 180.0  # 判定・ブートストラップ・図の作成・実験 D の準備と生成の固定の余裕
RUN_OVERHEAD_SECONDS = 2.0  # 学習 1 回あたりのモデルの構築などの固定費の余裕

# --- 水準(SMOKE_TEST で変わるもの) ---
LEVELS = {
    "smoke": {
        "STEPS_C": 24,  # 実験 C の学習ステップ数 T_C(削る段階 6 では半分)
        "STEPS_A": 24 * STEPS_RATIO_A_TO_C,  # 実験 A の学習ステップ数 T_A = 倍率 x T_C(本番と同じ倍率)
        "SEEDS_FULL": 3,
        "SEEDS_CUT": 2,
        "EVAL_SEQUENCES": 100,  # 実験 A の評価集合の系列数、実験 C の評価集合の系列数(系列長ごと)
        "CALIBRATION_SEQUENCES": 100,
        "CURVE_SEQUENCES": 50,  # 途中の評価に使う部分集合の系列数
        "B_LEVELS": (512, 1024, 2048),
        "B_SWEEPS": 3,
        "B_REPEATS": 3,
        "D_STEP_CANDIDATES": (16, 8, 4),
        "D_TRAIN_CHARACTERS": 600_000,
        "BOOTSTRAP_RESAMPLES": 1_000,
    },
    "prod": {
        "STEPS_C": 2000,
        "STEPS_A": 2000 * STEPS_RATIO_A_TO_C,
        "SEEDS_FULL": 5,
        "SEEDS_CUT": 3,
        "EVAL_SEQUENCES": 1000,
        "CALIBRATION_SEQUENCES": 500,
        "CURVE_SEQUENCES": 250,
        "B_LEVELS": (512, 1024, 2048, 4096, 8192),
        "B_SWEEPS": 8,
        "B_REPEATS": 8,
        "D_STEP_CANDIDATES": (2181, 1090, 545),  # 2181 は 008 と同じ(3 エポック相当)。以降は半分ずつ
        "D_TRAIN_CHARACTERS": None,
        "BOOTSTRAP_RESAMPLES": 10_000,
    },
}
CURRENT_LEVEL_NAME = "smoke" if SMOKE_TEST else "prod"
CFG = LEVELS[CURRENT_LEVEL_NAME]
STEPS_A_FULL, STEPS_C_FULL = CFG["STEPS_A"], CFG["STEPS_C"]
BOOTSTRAP_RESAMPLES = CFG["BOOTSTRAP_RESAMPLES"]
```

### 5.4 合成課題の評価集合

学習データは **ステップごとに新しく生成する**(同じ事例を繰り返さない)。評価集合は、学習とは別のシードで、実行の最初に固定して生成する。較正用の集合(学習率の選択だけに使う)も別のシードで生成する。

**Selective copying**(実験 A): 長さ 40 の入力の中のランダムな 8 か所にデータのトークン(8 種類)が置かれ、残りはノイズのトークンである。区切りのトークンの後に、データのトークンを **順序を保って** 出力する。系列全体は`[入力 (40), 区切り, 答え (8)]`の 49 トークンで、例えばデータのトークンを数字、ノイズを`.`、区切りを`|`と書くと、次の形である(8 個のうち 4 個だけを示す)。

```
入力: . . 3 . . . 5 . . 1 . . . . 6 . . . ...   区切り: |   答え: 3 5 1 6 ...
```

損失は答えのトークンの予測にだけ掛け、評価では区切りまでを与えて答えを **貪欲に 1 トークンずつ生成** し、正解と比べる(トークン単位の正解率を $a_i$ とする)。生成した答えそのものを記録する(6.7 節の混同行列)。答えのトークン 1 個を当てずっぽうで当てる確率は 1/8 = 0.125 である。

**Induction heads**(実験 C): ランダムなトークン(16 種類)の列の中に、特別なトークンが 1 回現れ、その直後のトークンが答えである。系列の末尾は再び特別なトークンで、その次に(最初の特別なトークンの直後にあったトークンを)出力する。最初の特別なトークンの位置は $[0, L - 3]$ の一様分布から引く。損失と評価は、末尾の位置の 1 回の予測にだけ掛ける。当てずっぽうで当てる確率は 1/16 = 0.0625 である。系列長 $L$ を引数に取るので、学習時の系列長 $L_{\mathrm{train}} = 32$ と、評価する系列長 $L = 32, 64, 128, 256, 512$($L_{\mathrm{train}}$ の 1・2・4・8・16 倍、等比)で同じ生成関数を使う。

**評価集合の大きさ**: 実験 A は 1000 系列(答えのトークン 8000 個)、実験 C は系列長ごとに 1000 系列。較正用の集合は 500 系列。


```python
# --- 合成課題の評価集合(学習とは別のシードで固定して生成する。学習データはステップごとに新しく生成する) ---
_num_eval = CFG["EVAL_SEQUENCES"]
_num_calibration = CFG["CALIBRATION_SEQUENCES"]
EVAL_A_TOKENS, EVAL_A_TARGETS, EVAL_A_DATA_POSITIONS = generate_selective_copying(TASK_A, _num_eval, make_rng(EVAL_SEED_A))
CALIBRATION_A_TOKENS, CALIBRATION_A_TARGETS, _ = generate_selective_copying(
    TASK_A, _num_calibration, make_rng(CALIBRATION_SEED_A)
)
C_LENGTHS = tuple(C_TRAIN_LENGTH * m for m in C_LENGTH_MULTIPLIERS)
C_TEST_LENGTH = C_TRAIN_LENGTH * C_TEST_MULTIPLIER
EVAL_C = {
    length: generate_induction_heads(TASK_C, length, _num_eval, make_rng(EVAL_SEED_C, m))
    for m, length in zip(C_LENGTH_MULTIPLIERS, C_LENGTHS, strict=True)
}  # 系列長 -> (トークン, 答え, 最初の特別なトークンの位置)
_calibration_c_tokens, _calibration_c_answers, _ = generate_induction_heads(
    TASK_C, C_TRAIN_LENGTH, _num_calibration, make_rng(CALIBRATION_SEED_C)
)
CALIBRATION_C_TOKENS = _calibration_c_tokens
CALIBRATION_C_TARGETS = induction_heads_targets(_calibration_c_tokens, _calibration_c_answers)
EVAL_C_TRAIN_TARGETS = induction_heads_targets(EVAL_C[C_TRAIN_LENGTH][0], EVAL_C[C_TRAIN_LENGTH][1])
CURVE_SEQUENCES = CFG["CURVE_SEQUENCES"]
```



## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/06_architectures/023_state_space_model_mamba.ipynb
