---
title: "ViT と画像パッチ埋め込み / Vision Transformer and Patch Embedding(実装・実験編 2/4)"
---

この記事は後編(実装・実験編 2/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/019_vision_transformer-practice-1)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/019_vision_transformer-practice-3)。

## 5. 実装 / Implementation

### 5.1 環境セットアップ(Google Colab)

`SMOKE_TEST`(スモークテストか本番か)はこのセルでのみ切り替える。本トピックは Hub にアップロードしないので、アップロードの
フラグはない。

**混合精度**: FP16 の`torch.autocast`と動的損失スケーリング(011 の`DynamicLossScaler`)は CUDA のときのみ有効にする。MPS・CPU
では FP32 で学習する。実効の設定をこのセルで印字する。評価は常に FP32 で行う。

**決定性**: 本トピックでは`torch.use_deterministic_algorithms(True)`を使わない。畳み込みの速度のために
`torch.backends.cudnn.benchmark = True`とし、決定的な演算を強制しない。そのため、同じシードの学習を
繰り返しても、CUDA の演算の丸めの順序の違いで結果が bit 単位では一致しないことがある。条件間の対応付け(同じシードの条件どうしで
初期化・データの順序・データ拡張を揃えること)は乱数の生成器によって決まり、演算の決定性には依存しない。


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
execution_environment = print_execution_environment(device)

# 混合精度(011): FP16 の autocast と動的損失スケーリングは CUDA のときのみ有効にする
USE_FP16_AUTOCAST = device.type == "cuda"
INIT_LOSS_SCALE = 2.0**16  # torch.amp.GradScaler の既定値と同じ
LOSS_SCALE_GROWTH_INTERVAL = 2000  # 同上
CNN_CHANNELS_LAST = device.type == "cuda"  # 畳み込みの高速化のためのメモリ配置(数値には影響しない)
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
print(
    f"SMOKE_TEST={SMOKE_TEST}、混合精度: FP16 autocast={USE_FP16_AUTOCAST}、"
    f"動的損失スケーリング={'有効' if USE_FP16_AUTOCAST else '無効(FP32)'}"
    f"(初期スケール {INIT_LOSS_SCALE:g}、growth_interval {LOSS_SCALE_GROWTH_INTERVAL}、backoff 0.5、growth 2.0)、"
    f"評価は FP32、CNN の channels_last={CNN_CHANNELS_LAST}、cudnn.benchmark={torch.backends.cudnn.benchmark}、"
    f"決定的な演算の強制: {torch.are_deterministic_algorithms_enabled()}"
)
```

    /content/ai-theories
    [2mUsing Python 3.13.15 environment at: /usr[0m
    [2mChecked [1m60 packages[0m [2min 104ms[0m[0m
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
      コミット / git commit                  : 18e70af43b0eb0397b34f3fe004eeb0a3efebfba
      未コミットの変更 / uncommitted changes : なし
      実行日時 (UTC)                         : 2026-09-30T01:49:07+00:00
    SMOKE_TEST=False、混合精度: FP16 autocast=True、動的損失スケーリング=有効(初期スケール 65536、growth_interval 2000、backoff 0.5、growth 2.0)、評価は FP32、CNN の channels_last=True、cudnn.benchmark=True、決定的な演算の強制: False



```python
import hashlib
import json
import math

import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.nn import functional

from src.data.image import (
    EpochShuffledBatchSampler,
    load_cifar10,
    make_nested_class_balanced_subsets,
    normalize_images,
    random_crop_and_flip,
    split_class_balanced_validation,
    to_normalized_float,
)
from src.layers.patch_embedding import (
    PatchEmbedding,
    extract_patches,
    permute_patches,
)
from src.models.vit import VisionTransformer, count_vit_non_embedding_parameters
from src.training.classification import (
    evaluate_image_classifier,
    logit,
    train_image_classifier,
    unscale_gradients_and_compute_norm,
)
from src.training.precision import DynamicLossScaler
from src.utils.statistics import fit_power_law_exponent


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


def sha256_of_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


precondition_status: dict[str, bool] = {}  # 前提条件の成否(6.1 節で宣言、各節で記録)
```

### 5.2 スケールの設定(`SMOKE_TEST`の配線)

水準の定義をこの 1 箇所に集約する。

**縮小の規則**:

- 縮小するのは **学習ステップ数 $T$ の候補** と **シード数** のみとする。$T$ の候補は本番 $\{3968, 1984, 992\}$、スモークテスト
  $\{128, 64, 32\}$ で、どちらも公比 2 の 3 点であり、本番とスモークテストの比(31 倍)を候補の間で揃える。スケーリングの計測点は、
  候補の最大値 $T_{\max}$ の $T_{\max}/32, T_{\max}/16, T_{\max}/8$ とする(本番 124・248・496、スモークテスト 4・8・16)。
- **縮小しないもの**: データ(訓練 45,000 枚・検証 5,000 枚・テスト 10,000 枚の全件)、実験 B のパッチサイズの 3 水準
  $P \in \{8, 4, 2\}$、実験 C のデータ量の 3 水準 $f \in \{1/16, 1/4, 1\}$(公比 4)、モデルの構成、バッチサイズ、学習率の較正の格子
  と拡張の規則、学習率のスケジュールの形(warmup の比率・最小学習率の比率)、途中の評価の位置($T$ に対する比率)、実行計画の表の
  構造と優先順位、前提条件・判定の閾値。
- **シード数**: 本番は 5(削った段階で 3)。スモークテストは 3(削った段階で 2)とする。スモークテストでも「削る前 > 削った後
  $\ge 2$」の順序関係を保ち、計画を強制指定したスモークテストで、削る処理が実際にシード数を減らすことを確かめるためである。
  すべての水準で標準条件 S($P = 4$・学習可能な 1 次元・全データ)のシードを実験 A・B・C で共有する構造を保つ。
- **テスト専用の上書き**: 環境変数`AI_THEORIES_FORCE_PLAN`(0〜11)が設定されているときのみ、6.4 節で見積もりによる計画の選択の
  代わりにその計画を使う(スモークテストで各計画の経路を確かめるため)。本番では受け付けず、このセルで停止する。


```python
# --- 全水準で共通の定数(本番実行前に宣言し、SMOKE_TEST で変えない) ---
IMAGE_SIZE, IN_CHANNELS, NUM_CLASSES = 32, 3, 10
D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF = 192, 6, 3, 768  # 標準の ViT
STANDARD_PATCH_SIZE = 4  # 標準条件 S(N = 64)
BATCH_SIZE = 128
WARMUP_RATIO = 0.1  # 最初の 10% のステップで線形 warmup
MIN_LEARNING_RATE_RATIO = 0.01  # cosine decay の下限 = 最大の学習率 x 0.01
WEIGHT_DECAY = 0.05  # 行列形の重みのみ(バイアス・正規化層・[CLS]・位置埋め込みには掛けない)
GRADIENT_CLIP_THRESHOLD = 1.0
CROP_PADDING = 4
VALIDATION_PER_CLASS = 500  # 検証集合 5,000 枚(クラス均衡)
DATA_FRACTIONS = (1 / 16, 1 / 4, 1.0)  # 実験 C のデータ量(入れ子、公比 4)
PATCH_SIZES = (8, 4, 2)  # 実験 B のパッチサイズ(N = 16, 64, 256)
POSITION_EMBEDDINGS = ("none", "learned", "sinusoidal_2d")  # 実験 A の条件 0, 1, 2
LR_GRID = {"vit": (5e-4, 1e-3, 2e-3), "cnn": (5e-4, 1e-3, 2e-3)}  # 学習率の較正の格子(公比 2)
LR_GRID_RATIO = 2.0
ACCURACY_MIN = 0.25  # 前提条件 P1: 一様な推測(0.1)の 2.5 倍
SIGMA_MULTIPLIER = 2.0  # 判定の閾値は対比量の標準偏差の 2 倍
INTERMEDIATE_EVAL_FRACTIONS = (0.25, 0.5, 0.75)  # 途中の検証集合の評価の位置(診断量、T に対する比率)
EVAL_BATCH_SIZE = 500
ATTENTION_DISTANCE_IMAGES = 1000  # 注意距離の観察に使う検証集合の画像の数

# 乱数シード(学習のシード s とは独立に固定するもの)
VALIDATION_SPLIT_SEED = 19_001
SUBSET_SEED = 19_002
PATCH_SHUFFLE_SEED = 19_003  # 位置情報への依存度の診断量で使うパッチの並べ替え
# 学習のシード s の初期化は torch.manual_seed(19100 + s)、データの順序・データ拡張は torch.Generator().manual_seed(19200 + s)
INIT_SEED_BASE = 19_100
DATA_SEED_BASE = 19_200
CALIBRATION_SEED_INDEX = 90  # 学習率の較正専用のシード(実験のシード 0〜4 と共有しない)
TIMING_SEED_INDEX = 91  # スケーリングの計測専用のシード
SESSION_BUDGET_SECONDS = 120 * 60  # 1 セッションの予算(T4 で 120 分)

# CIFAR-10 の内容の照合(torchvision が展開した配列の SHA-256。取得元が変わっていないことを確かめる)
CIFAR10_SHA256 = {
    "train_images": "9d5c2eddadb0deff02c14c350c0385aab7b1d5ff1edfeb4997c8dc8644960111",
    "train_labels": "f5cfe00b0f00968c0cc5ff3b1d2de51b10e33efa277c2986f4e0fa63e58c9f4f",
    "test_images": "177777c33ba79825320efa646ad04a94bdc3c80e4ebe3bd21724f8cdfd4b9d56",
    "test_labels": "cbb7365de8ed11f05cc4c3a1e7f78144127c5e851efd83762fb18202461230bb",
}

# 学習ステップ数 T の候補(公比 2、大きい順)。6.4 節で実行計画として自動選択する(6.1・6.2 節)
T_CANDIDATES = {
    "smoke": (128, 64, 32),
    "prod": (3968, 1984, 992),
}
# --- 削る段階(6.1 節)。順序: 実験 B の P = 2 -> 実験 C のシード数 -> 実験 A のシード数 ---
STAGES = {
    "prod": {
        0: {"SEEDS_A": 5, "SEEDS_B": 5, "SEEDS_C": 5, "B_PATCH_SIZES": (8, 4, 2)},
        1: {"SEEDS_A": 5, "SEEDS_B": 5, "SEEDS_C": 5, "B_PATCH_SIZES": (8, 4)},
        2: {"SEEDS_A": 5, "SEEDS_B": 5, "SEEDS_C": 3, "B_PATCH_SIZES": (8, 4)},
        3: {"SEEDS_A": 3, "SEEDS_B": 5, "SEEDS_C": 3, "B_PATCH_SIZES": (8, 4)},
    },
    "smoke": {  # 本番と同じ構造(シード数 5 -> 3 を 3 -> 2 に縮小)
        0: {"SEEDS_A": 3, "SEEDS_B": 3, "SEEDS_C": 3, "B_PATCH_SIZES": (8, 4, 2)},
        1: {"SEEDS_A": 3, "SEEDS_B": 3, "SEEDS_C": 3, "B_PATCH_SIZES": (8, 4)},
        2: {"SEEDS_A": 3, "SEEDS_B": 3, "SEEDS_C": 2, "B_PATCH_SIZES": (8, 4)},
        3: {"SEEDS_A": 2, "SEEDS_B": 3, "SEEDS_C": 2, "B_PATCH_SIZES": (8, 4)},
    },
}
SEED_KEYS = ("SEEDS_A", "SEEDS_B", "SEEDS_C")


def build_plans(level: str) -> list[dict]:
    # 実行計画: T の大きい順を第 1 キー、段階の番号の小さい順を第 2 キーとして番号を付ける(6.1 節)
    return [
        {"plan": i, "T": t, "stage": stage}
        for i, (t, stage) in enumerate((t, stage) for t in T_CANDIDATES[level] for stage in sorted(STAGES[level]))
    ]


PLANS = {level: build_plans(level) for level in T_CANDIDATES}

CURRENT_LEVEL_NAME = "smoke" if SMOKE_TEST else "prod"
FORCED_PLAN_VALUE = os.environ.get("AI_THEORIES_FORCE_PLAN")
if FORCED_PLAN_VALUE is not None and not SMOKE_TEST:
    raise RuntimeError(
        f"本番(SMOKE_TEST=False)では計画の強制(AI_THEORIES_FORCE_PLAN={FORCED_PLAN_VALUE!r})を受け付けない。"
        "環境変数を削除して再実行すること。"
    )
FORCED_PLAN = None if FORCED_PLAN_VALUE is None else int(FORCED_PLAN_VALUE)
assert FORCED_PLAN is None or 0 <= FORCED_PLAN < len(PLANS[CURRENT_LEVEL_NAME]), FORCED_PLAN

PROD_T_CANDIDATES = T_CANDIDATES["prod"]
T_MAX = T_CANDIDATES[CURRENT_LEVEL_NAME][0]
# TRAIN_STEPS(T)は 6.4 節で実行計画を選んだ後に決まる


def warmup_steps_for(num_steps: int) -> int:
    return max(1, round(WARMUP_RATIO * num_steps))


def intermediate_eval_steps_for(num_steps: int) -> tuple[int, ...]:
    return tuple(round(f * num_steps) for f in INTERMEDIATE_EVAL_FRACTIONS)


SCALING_STEP_COUNTS = (T_MAX // 32, T_MAX // 16, T_MAX // 8)

# --- 縮小規則と水準の構造の確認 ---
for _name, _candidates in T_CANDIDATES.items():
    assert _candidates[0] % 32 == 0, _name  # 計測点 T_max/32, T_max/16, T_max/8 が整数で公比 2
    assert len(_candidates) == 3 and all(a == 2 * b for a, b in zip(_candidates, _candidates[1:], strict=False))
    assert all(len(set(intermediate_eval_steps_for(t))) == 3 and warmup_steps_for(t) < t for t in _candidates)
    _plans = PLANS[_name]
    assert len(_plans) == 12 and [p["plan"] for p in _plans] == list(range(12))
    assert all(  # 優先順位: T の大きい順、同じ T では段階の小さい順
        (a["T"] > b["T"]) or (a["T"] == b["T"] and a["stage"] < b["stage"]) for a, b in zip(_plans, _plans[1:], strict=False)
    )
    _stages = STAGES[_name]
    assert set(_stages) == {0, 1, 2, 3}
    for _k in range(1, 4):  # 段階が上がるほど、どの量も減るか変わらない
        for _key in SEED_KEYS:
            assert _stages[_k][_key] <= _stages[_k - 1][_key]
        assert set(_stages[_k]["B_PATCH_SIZES"]) <= set(_stages[_k - 1]["B_PATCH_SIZES"])
    for _stage in _stages.values():
        assert min(_stage[k] for k in SEED_KEYS) >= 2
        assert STANDARD_PATCH_SIZE in _stage["B_PATCH_SIZES"] and len(_stage["B_PATCH_SIZES"]) >= 2
        assert set(_stage["B_PATCH_SIZES"]) <= set(PATCH_SIZES)
_ratios = {p / s for p, s in zip(T_CANDIDATES["prod"], T_CANDIDATES["smoke"], strict=True)}
assert len(_ratios) == 1 and _ratios.pop() > 1  # 本番とスモークテストの比が候補の間で揃う
assert [(p["T"] > 0, p["stage"]) for p in PLANS["prod"]] == [(p["T"] > 0, p["stage"]) for p in PLANS["smoke"]]
for _k in range(1, 4):  # 本番とスモークテストで、各段階で何が変わるかが同じ(構造を保つ縮小)
    for _key in (*SEED_KEYS, "B_PATCH_SIZES"):
        assert (STAGES["prod"][_k][_key] == STAGES["prod"][_k - 1][_key]) == (
            STAGES["smoke"][_k][_key] == STAGES["smoke"][_k - 1][_key]
        )
    assert STAGES["prod"][_k]["B_PATCH_SIZES"] == STAGES["smoke"][_k]["B_PATCH_SIZES"]
assert all(a / b == 2 for a, b in zip(PATCH_SIZES, PATCH_SIZES[1:], strict=False))  # N は公比 4
assert all(b / a == 4 for a, b in zip(DATA_FRACTIONS, DATA_FRACTIONS[1:], strict=False))
for _grid in LR_GRID.values():
    assert all(math.isclose(b / a, LR_GRID_RATIO) for a, b in zip(_grid, _grid[1:], strict=False))
assert D_MODEL % NUM_HEADS == 0 and D_MODEL % 4 == 0 and IMAGE_SIZE % max(PATCH_SIZES) == 0

print(
    f"水準 {CURRENT_LEVEL_NAME!r}: T の候補 {T_CANDIDATES[CURRENT_LEVEL_NAME]}(本番 {PROD_T_CANDIDATES})、"
    f"warmup {[warmup_steps_for(t) for t in T_CANDIDATES[CURRENT_LEVEL_NAME]]} ステップ"
)
print(
    f"ViT: D = {D_MODEL}、L = {NUM_LAYERS}、h = {NUM_HEADS}、d_ff = {D_FF}、標準の P = {STANDARD_PATCH_SIZE}、dropout なし、"
    f"位置埋め込みの条件 {POSITION_EMBEDDINGS}"
)
print(
    f"学習: バッチ {BATCH_SIZE}、AdamW(重み減衰 {WEIGHT_DECAY})、warmup {WARMUP_RATIO:.0%} + cosine(下限 x{MIN_LEARNING_RATE_RATIO})、"
    f"gradient clipping {GRADIENT_CLIP_THRESHOLD}、random crop(パディング {CROP_PADDING})+ 左右反転"
)
print(f"水準: P {PATCH_SIZES}、f {DATA_FRACTIONS}、学習率の格子 {LR_GRID}")
print(
    f"途中の評価のステップ(T の候補ごと) {[intermediate_eval_steps_for(t) for t in T_CANDIDATES[CURRENT_LEVEL_NAME]]}、"
    f"スケーリングの計測点 {SCALING_STEP_COUNTS} ステップ"
)
print(f"削る段階({CURRENT_LEVEL_NAME!r}): {json.dumps(STAGES[CURRENT_LEVEL_NAME])}")
print(f"実行計画({CURRENT_LEVEL_NAME!r}、6.4 節で選ぶ、番号の小さいほど優先):")
for _p in PLANS[CURRENT_LEVEL_NAME]:
    print(f"  計画 {_p['plan']:2d}: T = {_p['T']}、段階 {_p['stage']}")
```

    水準 'prod': T の候補 (3968, 1984, 992)(本番 (3968, 1984, 992))、warmup [397, 198, 99] ステップ
    ViT: D = 192、L = 6、h = 3、d_ff = 768、標準の P = 4、dropout なし、位置埋め込みの条件 ('none', 'learned', 'sinusoidal_2d')
    学習: バッチ 128、AdamW(重み減衰 0.05)、warmup 10% + cosine(下限 x0.01)、gradient clipping 1.0、random crop(パディング 4)+ 左右反転
    水準: P (8, 4, 2)、f (0.0625, 0.25, 1.0)、学習率の格子 {'vit': (0.0005, 0.001, 0.002), 'cnn': (0.0005, 0.001, 0.002)}
    途中の評価のステップ(T の候補ごと) [(992, 1984, 2976), (496, 992, 1488), (248, 496, 744)]、スケーリングの計測点 (124, 248, 496) ステップ
    削る段階('prod'): {"0": {"SEEDS_A": 5, "SEEDS_B": 5, "SEEDS_C": 5, "B_PATCH_SIZES": [8, 4, 2]}, "1": {"SEEDS_A": 5, "SEEDS_B": 5, "SEEDS_C": 5, "B_PATCH_SIZES": [8, 4]}, "2": {"SEEDS_A": 5, "SEEDS_B": 5, "SEEDS_C": 3, "B_PATCH_SIZES": [8, 4]}, "3": {"SEEDS_A": 3, "SEEDS_B": 5, "SEEDS_C": 3, "B_PATCH_SIZES": [8, 4]}}
    実行計画('prod'、6.4 節で選ぶ、番号の小さいほど優先):
      計画  0: T = 3968、段階 0
      計画  1: T = 3968、段階 1
      計画  2: T = 3968、段階 2
      計画  3: T = 3968、段階 3
      計画  4: T = 1984、段階 0
      計画  5: T = 1984、段階 1
      計画  6: T = 1984、段階 2
      計画  7: T = 1984、段階 3
      計画  8: T = 992、段階 0
      計画  9: T = 992、段階 1
      計画 10: T = 992、段階 2
      計画 11: T = 992、段階 3


### 5.3 データ: CIFAR-10 の取得・分割・部分集合

- CIFAR-10 を torchvision で取得し(キャッシュ`.cache/cifar10/`)、展開した配列の SHA-256 を 5.2 節の定数と照合する。
- 訓練集合 50,000 枚から、クラス均衡な **検証集合 5,000 枚**(各クラス 500 枚、乱数シード`19001`)を切り出す。残る 45,000 枚が
  学習に使う全データ($f = 1$)である。**検証集合は学習率の較正にのみ使い、テスト集合は較正に使わない。**
- 45,000 枚から、クラス均衡で入れ子の部分集合 $f = 1/16 \subset 1/4 \subset 1$ を作る(乱数シード`19002`、学習のシードとは独立)。
  各クラス 4,500 枚の $1/16$ は 281.25 枚で割り切れないので、各クラスの先頭 $\lfloor f \cdot 4500 \rfloor$ 枚をとる(1/16: 各クラス 281 枚、
  1/4: 1,125 枚)。実験 C の回帰には、名目の $f$ ではなく **実際の枚数の比** $\tilde{f} = \lvert S_f \rvert / \lvert S_1 \rvert$ の $\log_2$ を使う。
- 最小の部分集合 $S_{1/16}$ はすべての部分集合に含まれるので、学習の終わりの $S_{1/16}$ での正解率(データ拡張なし)を、データ量の
  異なる条件の間で比べられる「学習データでの正解率」の診断量とする(実験 C)。
- 画像は uint8 のまま GPU に載せ(訓練 50,000 枚で約 150 MB)、全条件・全シードで再利用する。


```python
_t0_data = time.time()
CIFAR10 = load_cifar10()
_data_hashes = {
    "train_images": sha256_of_tensor(CIFAR10.train_images),
    "train_labels": sha256_of_tensor(CIFAR10.train_labels),
    "test_images": sha256_of_tensor(CIFAR10.test_images),
    "test_labels": sha256_of_tensor(CIFAR10.test_labels),
}
for _name, _hash in _data_hashes.items():
    assert _hash == CIFAR10_SHA256[_name], (_name, _hash)
assert tuple(CIFAR10.train_images.shape) == (50_000, 3, 32, 32)
assert tuple(CIFAR10.test_images.shape) == (10_000, 3, 32, 32)
_train_labels_np = CIFAR10.train_labels.numpy()
_test_labels_np = CIFAR10.test_labels.numpy()

TRAIN_POOL, VALIDATION_INDICES = split_class_balanced_validation(
    _train_labels_np, VALIDATION_PER_CLASS, VALIDATION_SPLIT_SEED
)
SUBSETS = make_nested_class_balanced_subsets(
    _train_labels_np, TRAIN_POOL, DATA_FRACTIONS, SUBSET_SEED
)
FULL_SIZE = len(SUBSETS[1.0])
FRACTION_ACTUAL = {f: len(SUBSETS[f]) / FULL_SIZE for f in DATA_FRACTIONS}

# --- 分割・部分集合の不変条件 ---
assert len(VALIDATION_INDICES) == 10 * VALIDATION_PER_CLASS and len(TRAIN_POOL) == 45_000
assert np.bincount(_train_labels_np[VALIDATION_INDICES], minlength=10).tolist() == [VALIDATION_PER_CLASS] * 10
assert np.intersect1d(TRAIN_POOL, VALIDATION_INDICES).size == 0  # 検証集合は学習に使う事例と重ならない
assert np.array_equal(SUBSETS[1.0], TRAIN_POOL)
for _small, _large in zip(DATA_FRACTIONS, DATA_FRACTIONS[1:], strict=False):
    assert np.isin(SUBSETS[_small], SUBSETS[_large]).all()  # 入れ子
for _f in DATA_FRACTIONS:
    _counts = np.bincount(_train_labels_np[SUBSETS[_f]], minlength=10)
    assert (_counts == _counts[0]).all() and _counts[0] == math.floor(_f * 4500), (_f, _counts)  # クラス均衡
    assert np.intersect1d(SUBSETS[_f], VALIDATION_INDICES).size == 0
# テスト集合は別のファイル(添字の空間が別)なので、学習・検証の事例と添字では重ならない。
# 画素が完全に一致する画像の重複も数える(CIFAR-10 には訓練とテストの間に近い画像があることが知られている)。
_train_image_hashes = [hashlib.sha1(im.numpy().tobytes()).hexdigest() for im in CIFAR10.train_images]
_test_image_hashes = [hashlib.sha1(im.numpy().tobytes()).hexdigest() for im in CIFAR10.test_images]
_val_hash_set = {_train_image_hashes[i] for i in VALIDATION_INDICES}
_pool_hash_set = {_train_image_hashes[i] for i in TRAIN_POOL}
TEST_EXACT_DUPLICATES = {
    "検証集合": sum(h in _val_hash_set for h in _test_image_hashes),
    "学習に使う 45,000 枚": sum(h in _pool_hash_set for h in _test_image_hashes),
}
VALIDATION_EXACT_DUPLICATES_IN_POOL = sum(_train_image_hashes[i] in _pool_hash_set for i in VALIDATION_INDICES)
assert TEST_EXACT_DUPLICATES["検証集合"] == 0 and VALIDATION_EXACT_DUPLICATES_IN_POOL == 0

# --- GPU に載せる(全条件・全シードで再利用) ---
TRAIN_IMAGES = CIFAR10.train_images.to(device)
TRAIN_LABELS = CIFAR10.train_labels.to(device)
TEST_IMAGES = CIFAR10.test_images.to(device)
TEST_LABELS = CIFAR10.test_labels.to(device)
_validation_device = torch.from_numpy(VALIDATION_INDICES).to(device)
VALIDATION_IMAGES = TRAIN_IMAGES[_validation_device]
VALIDATION_LABELS = TRAIN_LABELS[_validation_device]
SUBSET_INDICES_DEVICE = {f: torch.from_numpy(SUBSETS[f]).to(device) for f in DATA_FRACTIONS}
_common = SUBSET_INDICES_DEVICE[DATA_FRACTIONS[0]]
COMMON_SUBSET_IMAGES = TRAIN_IMAGES[_common]  # S_{1/16}: すべての部分集合に含まれる
COMMON_SUBSET_LABELS = TRAIN_LABELS[_common]
# GPU 上のテンソル(キャッシュ)が元の配列と一致すること
assert torch.equal(TRAIN_IMAGES.cpu(), CIFAR10.train_images) and torch.equal(TEST_IMAGES.cpu(), CIFAR10.test_images)
assert torch.equal(VALIDATION_IMAGES.cpu(), CIFAR10.train_images[VALIDATION_INDICES])
TEST_IMAGES_SHA256 = sha256_of_tensor(TEST_IMAGES)
DATA_SECONDS = time.time() - _t0_data

print(f"CIFAR-10: 訓練 {len(CIFAR10.train_images):,} 枚・テスト {len(CIFAR10.test_images):,} 枚、SHA-256 が定数と一致: OK")
print(
    f"分割: 学習に使う {len(TRAIN_POOL):,} 枚・検証 {len(VALIDATION_INDICES):,} 枚(各クラス {VALIDATION_PER_CLASS})、"
    f"テスト {len(_test_labels_np):,} 枚(各クラス {np.bincount(_test_labels_np).tolist()[0]})"
)
for _f in DATA_FRACTIONS:
    print(
        f"  f = {_f:.4f}: {len(SUBSETS[_f]):,} 枚(各クラス {len(SUBSETS[_f]) // 10})、実際の比 {FRACTION_ACTUAL[_f]:.5f}、"
        f"log2 = {math.log2(FRACTION_ACTUAL[_f]):+.4f}、エポック数(T の候補ごと): "
        + "、".join(f"T = {t}: {t * BATCH_SIZE / len(SUBSETS[_f]):.1f}" for t in T_CANDIDATES[CURRENT_LEVEL_NAME])
        + "(本番 "
        + "、".join(f"T = {t}: {t * BATCH_SIZE / len(SUBSETS[_f]):.1f}" for t in PROD_T_CANDIDATES)
        + ")"
    )
print(
    "不変条件: 検証集合と学習の事例の添字が重ならない、部分集合が入れ子でクラス均衡、検証集合と部分集合が重ならない、"
    "GPU 上のテンソルが元の配列と一致: OK"
)
print(
    f"画素が完全に一致する画像の重複: テスト集合と {TEST_EXACT_DUPLICATES}、検証集合と学習に使う事例 "
    f"{VALIDATION_EXACT_DUPLICATES_IN_POOL} 枚"
)
print(f"データの準備: {DATA_SECONDS:.1f}s(ダウンロードを含む)")
```

    100%|██████████| 170M/170M [13:46<00:00, 206kB/s]


    CIFAR-10: 訓練 50,000 枚・テスト 10,000 枚、SHA-256 が定数と一致: OK
    分割: 学習に使う 45,000 枚・検証 5,000 枚(各クラス 500)、テスト 10,000 枚(各クラス 1000)
      f = 0.0625: 2,810 枚(各クラス 281)、実際の比 0.06244、log2 = -4.0013、エポック数(T の候補ごと): T = 3968: 180.7、T = 1984: 90.4、T = 992: 45.2(本番 T = 3968: 180.7、T = 1984: 90.4、T = 992: 45.2)
      f = 0.2500: 11,250 枚(各クラス 1125)、実際の比 0.25000、log2 = -2.0000、エポック数(T の候補ごと): T = 3968: 45.1、T = 1984: 22.6、T = 992: 11.3(本番 T = 3968: 45.1、T = 1984: 22.6、T = 992: 11.3)
      f = 1.0000: 45,000 枚(各クラス 4500)、実際の比 1.00000、log2 = +0.0000、エポック数(T の候補ごと): T = 3968: 11.3、T = 1984: 5.6、T = 992: 2.8(本番 T = 3968: 11.3、T = 1984: 5.6、T = 992: 2.8)
    不変条件: 検証集合と学習の事例の添字が重ならない、部分集合が入れ子でクラス均衡、検証集合と部分集合が重ならない、GPU 上のテンソルが元の配列と一致: OK
    画素が完全に一致する画像の重複: テスト集合と {'検証集合': 0, '学習に使う 45,000 枚': 0}、検証集合と学習に使う事例 0 枚
    データの準備: 846.9s(ダウンロードを含む)


### 5.4 モデルの構築と不変条件の確認

- **パラメータ数**: 実験 A の 3 条件・実験 B の 3 水準の ViT と、CNN(ResNet-56)のパラメータ数を印字する(ResNet-56 は約 0.85M になることを確かめる)。**実験 A・B の条件間で
  非埋め込みパラメータ数(パッチ埋め込み・[CLS]・位置埋め込みを除く)が完全に一致すること** をアサーションで確かめる。
- **パッチ埋め込みの等価性**(3.2.1 節): 畳み込みの出力と、`extract_patches()`で平坦化したパッチに $E$ を掛けた結果が一致すること
  (FP64、CPU)。
- **並べ替え不変性**(3.4.2 節): 位置埋め込みなしの ViT で、パッチを並べ替えた画像の logits が元の画像の logits と一致すること(FP32)。
  分類ヘッドは零で初期化されていて logits が恒等的に 0 になるので、この確認に限り分類ヘッドを乱数で初期化し直す。対照として、
  学習可能な 1 次元・2 次元正弦波の位置埋め込みの ViT では logits が変わることも確かめる(確認の検出力)。
- **データ拡張**: `random_crop_and_flip()`の出力が、同じ乱数で画像ごとにループで切り出した参照実装と bit 単位で一致すること。
  `EpochShuffledBatchSampler`が 1 エポックの中で各事例をちょうど 1 回ずつ使うこと。
- **損失スケーリングの unscale**: `unscale_gradients_and_compute_norm()`で除した勾配が、011 の`DynamicLossScaler.unscale_gradients()`で
  除した勾配と bit 単位で一致すること、非有限値を含む勾配でノルムが非有限になること(非有限値の検出)。
- **計算量の閉形式**(3.2.2 節): ViT の線形層の順伝播の演算数の閉形式が、モジュールのフック(`nn.Conv2d`・`nn.Linear`)で数えた値と
  一致すること。ResNet-56 の 1 枚あたりの積和演算の数の閉形式(畳み込みと全結合層。batch normalization・ReLU・加算・プーリングは
  数えない)も、フックで数えた値と一致すること。

**ResNet-56 の構成**(He et al. [7] の 4.2 節): $3 \times 3$ の畳み込み(16 チャネル)の後、チャネル数 16・32・64 の 3 つの段に、それぞれ
$n = 9$ 個の残差ブロック(畳み込み $3 \times 3$ → batch normalization → ReLU → 畳み込み $3 \times 3$ → batch normalization、ショートカットを
加えて ReLU)を置き、大域平均プーリングと全結合層で分類する(層数 $6n + 2 = 56$)。2・3 段目の先頭のブロックは stride 2 で空間を半分に
する。ショートカットは論文の **方式 A**: 次元が増えるところでは、入力を stride 2 で間引き、増えたチャネルを零で埋めた恒等写像とし、
パラメータを持たない。畳み込みの重みは He の初期化(fan_out)、畳み込みはバイアスを持たない。重み減衰は ViT と同じ規則で、
次元が 2 以上の重み(畳み込み・全結合)にのみ掛け、batch normalization のパラメータとバイアスには掛けない。

積和演算の数の閉形式は、段 $k$($k = 0, 1, 2$)の空間の一辺を $S_k = 32 / 2^k$、チャネル数を $c_k$ として

$$
M = 9 \cdot 3 c_0 S_0^2 + \sum_{k=0}^{2} \left[ 9 c_{k'} c_k S_k^2 + 9 c_k^2 S_k^2 + (n - 1) \cdot 2 \cdot 9 c_k^2 S_k^2 \right] + c_2 K
$$

である。ここで $c_{k'}$ は段 $k$ の先頭のブロックの入力のチャネル数($k = 0$ では $c_0$、それ以外は $c_{k-1}$)、$K$ はクラス数である。


```python
_t0_checks = time.time()


def build_vit(patch_size: int = STANDARD_PATCH_SIZE, position_embedding: str = "learned") -> VisionTransformer:
    return VisionTransformer(
        image_size=IMAGE_SIZE,
        patch_size=patch_size,
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        position_embedding=position_embedding,
    )


RESNET_DEPTH_N = 9  # 層数 6n + 2 = 56
RESNET_WIDTHS = (16, 32, 64)


class ResidualBlock(nn.Module):
    # He et al. の CIFAR-10 用の残差ブロック(ショートカットは方式 A: 間引きと零詰めによる恒等写像、パラメータなし)
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        self.extra_channels = out_channels - in_channels

    def shortcut(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1 and self.extra_channels == 0:
            return x
        x = x[:, :, :: self.stride, :: self.stride]  # stride 2 の間引き
        half = self.extra_channels // 2
        return functional.pad(x, (0, 0, 0, 0, half, self.extra_channels - half))  # 増えたチャネルを零で埋める

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return functional.relu(out + self.shortcut(x))


class CifarResNet(nn.Module):
    # ResNet-56(He et al. の 4.2 節): 3x3 畳み込み -> 3 段 x n ブロック(16・32・64 チャネル)-> 大域平均プーリング -> 全結合層
    def __init__(self, n: int = RESNET_DEPTH_N, widths=RESNET_WIDTHS, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.conv = nn.Conv2d(IN_CHANNELS, widths[0], 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(widths[0])
        blocks, in_channels = [], widths[0]
        for stage, width in enumerate(widths):
            for i in range(n):
                blocks.append(ResidualBlock(in_channels, width, 2 if (stage > 0 and i == 0) else 1))
                in_channels = width
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(widths[-1], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = functional.relu(self.bn(self.conv(x)))
        x = self.blocks(x)
        return self.fc(x.mean(dim=(2, 3)))  # 大域平均プーリング


def build_cnn() -> nn.Module:
    return CifarResNet()


def resnet_forward_macs(n: int = RESNET_DEPTH_N, widths=RESNET_WIDTHS) -> int:
    # 5.4 節の閉形式(1 枚の積和演算の数。畳み込みと全結合層のみ)
    total = 9 * IN_CHANNELS * widths[0] * IMAGE_SIZE**2
    for k, c in enumerate(widths):
        side = IMAGE_SIZE // 2**k
        c_in = widths[0] if k == 0 else widths[k - 1]
        total += 9 * c_in * c * side**2 + 9 * c * c * side**2 + (n - 1) * 2 * 9 * c * c * side**2
    return total + widths[-1] * NUM_CLASSES


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def vit_forward_flops(patch_size: int, include_attention: bool = True) -> int:
    # 3.2.2 節の閉形式(1 枚の画像の順伝播、積和を 2 回と数える)
    n_patches = (IMAGE_SIZE // patch_size) ** 2
    n = n_patches + 1
    patch = 2 * n_patches * (patch_size**2 * IN_CHANNELS) * D_MODEL
    linear = NUM_LAYERS * 2 * n * (4 * D_MODEL**2 + 2 * D_MODEL * D_FF)
    attention = NUM_LAYERS * 4 * n**2 * D_MODEL if include_attention else 0
    head = 2 * D_MODEL * NUM_CLASSES
    return patch + linear + attention + head


def count_module_forward_flops(model: nn.Module) -> int:
    # nn.Conv2d・nn.Linear の積和の数 x 2 をフックで数える(1 枚の画像、FP32、CPU)
    total = 0

    def conv_hook(module, inputs, output):
        nonlocal total
        k = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
        total += 2 * output.numel() * k

    def linear_hook(module, inputs, output):
        nonlocal total
        total += 2 * output.numel() * module.in_features

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
    with torch.no_grad():
        model.eval()(torch.zeros(1, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
    for h in handles:
        h.remove()
    return total


# --- パラメータ数と非埋め込みパラメータ数の一致(実験 A・B) ---
PARAMETER_TABLE = {}
for _pe in POSITION_EMBEDDINGS:
    _m = build_vit(STANDARD_PATCH_SIZE, _pe)
    PARAMETER_TABLE[f"A: P=4, {_pe}"] = (count_parameters(_m), count_vit_non_embedding_parameters(_m))
for _p in PATCH_SIZES:
    _m = build_vit(_p, "learned")
    PARAMETER_TABLE[f"B: P={_p}, learned"] = (count_parameters(_m), count_vit_non_embedding_parameters(_m))
NON_EMBEDDING_PARAMETERS = {v[1] for v in PARAMETER_TABLE.values()}
assert len(NON_EMBEDDING_PARAMETERS) == 1, PARAMETER_TABLE  # 実験 A・B の条件間で完全に一致
NON_EMBEDDING_PARAMETERS = NON_EMBEDDING_PARAMETERS.pop()
CNN_PARAMETERS = count_parameters(build_cnn())
print("パラメータ数(全体 / 非埋め込み):")
for _name, (_total, _non_embedding) in PARAMETER_TABLE.items():
    print(f"  ViT {_name}: {_total:,} / {_non_embedding:,}")
assert 0.84e6 < CNN_PARAMETERS < 0.86e6, CNN_PARAMETERS  # ResNet-56 は約 0.85M
print(
    f"  CNN(ResNet-56): {CNN_PARAMETERS:,}(ViT P=4・学習可能な 1 次元の全パラメータ数は CNN の "
    f"{PARAMETER_TABLE['A: P=4, learned'][0] / CNN_PARAMETERS:.2f} 倍)"
)
print("非埋め込みパラメータ数が実験 A・B の全条件で一致: OK")

# --- 計算量の閉形式の確認と CNN の演算数 ---
VIT_FORWARD_FLOPS = {}
for _p in PATCH_SIZES:
    _hooked = count_module_forward_flops(build_vit(_p, "learned"))
    assert _hooked == vit_forward_flops(_p, include_attention=False), (_p, _hooked)
    VIT_FORWARD_FLOPS[_p] = vit_forward_flops(_p)
CNN_FORWARD_FLOPS = count_module_forward_flops(build_cnn())
CNN_FORWARD_MACS = resnet_forward_macs()
assert CNN_FORWARD_FLOPS == 2 * CNN_FORWARD_MACS, (CNN_FORWARD_FLOPS, CNN_FORWARD_MACS)  # 閉形式とフックの一致
print("順伝播の演算数(1 枚、積和を 2 回と数える): 線形層の閉形式とフックの値が一致: OK")
for _p in PATCH_SIZES:
    print(f"  ViT P={_p}(N={(IMAGE_SIZE // _p) ** 2}): {VIT_FORWARD_FLOPS[_p] / 1e6:,.1f} MFLOP")
print(
    f"  CNN(ResNet-56): 積和演算 {CNN_FORWARD_MACS / 1e6:,.2f} M 回 = {CNN_FORWARD_FLOPS / 1e6:,.1f} MFLOP"
    f"(畳み込みと全結合層。閉形式とフックの値が一致: OK)"
)
print(
    f"  1 枚あたりの計算量の比 CNN / ViT(P=4): {CNN_FORWARD_FLOPS / VIT_FORWARD_FLOPS[STANDARD_PATCH_SIZE]:.2f}"
    f"(ViT は注意の行列積を含む閉形式 {VIT_FORWARD_FLOPS[STANDARD_PATCH_SIZE] / 1e6:,.1f} MFLOP)"
)

# --- パッチ埋め込みの等価性(FP64、CPU) ---
_x64 = to_normalized_float(CIFAR10.test_images[:16]).double()
for _p in PATCH_SIZES:
    torch.manual_seed(0)
    _layer = PatchEmbedding(IMAGE_SIZE, _p, IN_CHANNELS, D_MODEL).double()
    _weight, _bias = _layer.linear_projection_weight()
    _conv = _layer(_x64)
    _linear = extract_patches(_x64, _p) @ _weight + _bias
    _diff = (_conv - _linear).abs().max().item()
    assert _conv.shape == (16, (IMAGE_SIZE // _p) ** 2, D_MODEL) and _diff < 1e-12, (_p, _diff)
    print(f"パッチ埋め込み P={_p}: 畳み込みと平坦化 + 線形射影の最大絶対差 {_diff:.2e}(FP64): OK")

# --- 並べ替え不変性(FP32) ---
_generator = torch.Generator().manual_seed(PATCH_SHUFFLE_SEED)
SHUFFLE_PERMUTATION = torch.randperm((IMAGE_SIZE // STANDARD_PATCH_SIZE) ** 2, generator=_generator)
_x = to_normalized_float(TEST_IMAGES[:32])
_x_perm = permute_patches(_x, STANDARD_PATCH_SIZE, SHUFFLE_PERMUTATION)
assert torch.equal(
    extract_patches(_x_perm, STANDARD_PATCH_SIZE),
    extract_patches(_x, STANDARD_PATCH_SIZE)[:, SHUFFLE_PERMUTATION.to(device)],
)
assert not torch.equal(_x_perm, _x)
PERMUTATION_CHECK = {}
for _pe in POSITION_EMBEDDINGS:
    torch.manual_seed(1)
    _m = build_vit(STANDARD_PATCH_SIZE, _pe).to(device).eval()
    nn.init.normal_(_m.head.weight, std=1.0)  # 零の分類ヘッドでは logits が恒等的に 0 になるため(確認のみ)
    with torch.no_grad():
        _out, _out_perm = _m(_x), _m(_x_perm)
    PERMUTATION_CHECK[_pe] = ((_out - _out_perm).abs().max() / _out.abs().max()).item()
assert PERMUTATION_CHECK["none"] < 1e-5, PERMUTATION_CHECK
assert PERMUTATION_CHECK["learned"] > 1e-3 and PERMUTATION_CHECK["sinusoidal_2d"] > 1e-3, PERMUTATION_CHECK
print(
    "並べ替え不変性(logits の最大相対差、FP32): "
    + "、".join(f"{k} = {v:.2e}" for k, v in PERMUTATION_CHECK.items())
    + " -> 位置埋め込みなしは不変、他の 2 方式は変わる: OK"
)

# --- データ拡張の参照実装との一致 ---
_gen_a = torch.Generator().manual_seed(123)
_gen_b = torch.Generator().manual_seed(123)
_batch = TRAIN_IMAGES[:16]
_augmented = random_crop_and_flip(_batch, CROP_PADDING, _gen_a)
_offsets = torch.randint(0, 2 * CROP_PADDING + 1, (16, 2), generator=_gen_b)
_flips = torch.rand(16, generator=_gen_b) < 0.5
_reference = []
for _i in range(16):
    _padded = functional.pad(_batch[_i].float() / 255.0, (CROP_PADDING,) * 4)
    _oy, _ox = _offsets[_i].tolist()
    _crop = _padded[:, _oy : _oy + IMAGE_SIZE, _ox : _ox + IMAGE_SIZE]
    _reference.append(_crop.flip(-1) if _flips[_i] else _crop)
_reference = normalize_images(torch.stack(_reference))
assert torch.equal(_augmented, _reference)
assert torch.equal(_gen_a.get_state(), _gen_b.get_state())  # 乱数の消費量も一致
_sampler = EpochShuffledBatchSampler(1000, BATCH_SIZE, torch.Generator().manual_seed(0))
_drawn = torch.cat([_sampler.next_batch() for _ in range(3 * 1000 // BATCH_SIZE + 1)])
for _e in range(3):
    assert sorted(_drawn[_e * 1000 : (_e + 1) * 1000].tolist()) == list(range(1000))
print("データ拡張: 参照実装と bit 単位で一致、バッチの切り出しが各エポックで各事例を 1 回ずつ使う: OK")

# --- 損失スケーリングの unscale の一致(011 の DynamicLossScaler.unscale_gradients と比べる) ---
torch.manual_seed(2)
_m = build_vit(8, "learned").to(device)
nn.init.normal_(_m.head.weight, std=0.02)
_params = [p for p in _m.parameters()]
_loss = functional.cross_entropy(_m(_x), TEST_LABELS[:32])
(_loss * 1024.0).backward()
_raw = [p.grad.clone() for p in _params]
_scaler = DynamicLossScaler(1024.0)
assert _scaler.unscale_gradients(_params) is False
_expected = [p.grad.clone() for p in _params]
for p, g in zip(_params, _raw, strict=True):
    p.grad = g.clone()
_norm = unscale_gradients_and_compute_norm(_params, 1024.0)
assert all(torch.equal(p.grad, e) for p, e in zip(_params, _expected, strict=True))
assert torch.isfinite(_norm)
_params[3].grad.view(-1)[0] = float("inf")
assert not torch.isfinite(unscale_gradients_and_compute_norm(_params, 1.0))
_params[3].grad.view(-1)[0] = float("nan")
assert not torch.isfinite(unscale_gradients_and_compute_norm(_params, 1.0))
print("損失スケーリングの unscale: DynamicLossScaler.unscale_gradients と bit 単位で一致、Inf・NaN をノルムで検出: OK")
del _m, _params, _raw, _expected
CHECK_SECONDS = time.time() - _t0_checks
print(f"確認の実行時間: {CHECK_SECONDS:.1f}s")
```

    パラメータ数(全体 / 非埋め込み):
      ViT A: P=4, none: 2,676,490 / 2,666,890
      ViT A: P=4, learned: 2,688,970 / 2,666,890
      ViT A: P=4, sinusoidal_2d: 2,676,490 / 2,666,890
      ViT B: P=8, learned: 2,707,402 / 2,666,890
      ViT B: P=4, learned: 2,688,970 / 2,666,890
      ViT B: P=2, learned: 2,718,922 / 2,666,890
      CNN(ResNet-56): 853,018(ViT P=4・学習可能な 1 次元の全パラメータ数は CNN の 3.15 倍)
    非埋め込みパラメータ数が実験 A・B の全条件で一致: OK
    順伝播の演算数(1 枚、積和を 2 回と数える): 線形層の閉形式とフックの値が一致: OK
      ViT P=8(N=16): 92.8 MFLOP
      ViT P=4(N=64): 365.7 MFLOP
      ViT P=2(N=256): 1,669.8 MFLOP
      CNN(ResNet-56): 積和演算 125.49 M 回 = 251.0 MFLOP(畳み込みと全結合層。閉形式とフックの値が一致: OK)
      1 枚あたりの計算量の比 CNN / ViT(P=4): 0.69(ViT は注意の行列積を含む閉形式 365.7 MFLOP)
    パッチ埋め込み P=8: 畳み込みと平坦化 + 線形射影の最大絶対差 0.00e+00(FP64): OK
    パッチ埋め込み P=4: 畳み込みと平坦化 + 線形射影の最大絶対差 0.00e+00(FP64): OK
    パッチ埋め込み P=2: 畳み込みと平坦化 + 線形射影の最大絶対差 0.00e+00(FP64): OK
    並べ替え不変性(logits の最大相対差、FP32): none = 3.71e-07、learned = 2.61e-02、sinusoidal_2d = 1.31e-01 -> 位置埋め込みなしは不変、他の 2 方式は変わる: OK
    データ拡張: 参照実装と bit 単位で一致、バッチの切り出しが各エポックで各事例を 1 回ずつ使う: OK
    損失スケーリングの unscale: DynamicLossScaler.unscale_gradients と bit 単位で一致、Inf・NaN をノルムで検出: OK
    確認の実行時間: 1.3s


### 5.5 学習と評価のヘルパー

- **学習の鍵**: 1 つの学習を`(arch, P, 位置埋め込み, f, s)`の組で表す(CNN の`P`・位置埋め込みは`None`)。標準条件 S は
  `("vit", 4, "learned", 1.0, s)`であり、実験 A の条件 1・実験 B の $P = 4$・実験 C の ViT の $f = 1$ はすべてこの鍵になる。
  同じ鍵の学習は 1 回だけ行い、記録を共有する。
- **シード $s$ が決めるもの**: 初期化(`torch.manual_seed(19100 + s)`でモデルを構築)と、データの順序・データ拡張(CPU の生成器
  `19200 + s`)。同じ $s$ の学習どうしは、同じ部分集合を使う限り、条件によらず **同じ順序で同じ拡張をした同じ画像** を見る
  (6.11 節でアサーションにより確かめる)。
- **評価**(学習の最終ステップの重み、FP32、データ拡張なし): 検証集合の正解率(途中の $T/4, T/2, 3T/4$ と最後)、テスト集合の
  正解率と交差エントロピー、共通部分 $S_{1/16}$ での正解率。実験 A の鍵では、パッチを固定の置換で並べ替えたテスト画像の
  正解率(位置情報への依存度の診断量)を、シード 0 では注意距離と位置埋め込みの類似度(6.8 節の観察)を加える。
- **較正の学習**(`purpose="calibration"`)ではテスト集合を一切評価しない(記録にテストの鍵を持たないことを 6.11 節で確かめる)。
- 学習ステップ数`TRAIN_STEPS`と途中の評価のステップ`INTERMEDIATE_EVAL_STEPS`は、6.4 節で実行計画を選んだ後に決まる。


```python
def key_vit(patch_size=STANDARD_PATCH_SIZE, position_embedding="learned", fraction=1.0, seed=0) -> tuple:
    return ("vit", patch_size, position_embedding, fraction, seed)


def key_cnn(fraction=1.0, seed=0) -> tuple:
    return ("cnn", None, None, fraction, seed)


def is_a_key(key: tuple) -> bool:
    return key[0] == "vit" and key[1] == STANDARD_PATCH_SIZE and key[3] == 1.0


def build_model_for(key: tuple) -> nn.Module:
    torch.manual_seed(INIT_SEED_BASE + key[4])
    model = build_vit(key[1], key[2]) if key[0] == "vit" else build_cnn()
    model = model.to(device)
    if key[0] == "cnn" and CNN_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    return model


def evaluate(model, images, labels, arch, image_transform=None) -> dict:
    return evaluate_image_classifier(
        model,
        images,
        labels,
        EVAL_BATCH_SIZE,
        image_transform,
        channels_last=(arch == "cnn" and CNN_CHANNELS_LAST),
    )


def train(model, key, learning_rate, num_steps, evaluation_steps=(), evaluation_fn=None) -> dict:
    return train_image_classifier(
        model,
        TRAIN_IMAGES,
        TRAIN_LABELS,
        SUBSET_INDICES_DEVICE[key[3]],
        num_steps=num_steps,
        batch_size=BATCH_SIZE,
        peak_learning_rate=learning_rate,
        warmup_steps=warmup_steps_for(num_steps),
        min_learning_rate=learning_rate * MIN_LEARNING_RATE_RATIO,
        weight_decay=WEIGHT_DECAY,
        gradient_clip_threshold=GRADIENT_CLIP_THRESHOLD,
        data_seed=DATA_SEED_BASE + key[4],
        crop_padding=CROP_PADDING,
        use_fp16_autocast=USE_FP16_AUTOCAST,
        init_loss_scale=INIT_LOSS_SCALE,
        loss_scale_growth_interval=LOSS_SCALE_GROWTH_INTERVAL,
        channels_last=(key[0] == "cnn" and CNN_CHANNELS_LAST),
        evaluation_steps=evaluation_steps,
        evaluation_fn=evaluation_fn,
    )


def summary(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "correct"}


@torch.no_grad()
def mean_attention_distance(model: VisionTransformer, images: torch.Tensor) -> np.ndarray:
    # 層ごと・ヘッドごとの平均の注意距離(画素)。パッチ間の注意の重みを、パッチの中心間の距離で重み付けして平均する
    # (Query・Key ともにパッチのみ。[CLS] への重みを除いて行ごとに正規化し直す。原論文の図 7 右と同じ量)
    model.eval()
    patch = model.patch_embedding.patch_size
    grid = model.patch_embedding.grid_size
    rows, cols = np.divmod(np.arange(grid * grid), grid)
    distance = patch * np.sqrt((rows[:, None] - rows[None, :]) ** 2 + (cols[:, None] - cols[None, :]) ** 2)
    distance = torch.tensor(distance, dtype=torch.float32, device=device)
    total = torch.zeros(NUM_LAYERS, NUM_HEADS, device=device)
    for start in range(0, images.size(0), 250):
        _, weights = model(to_normalized_float(images[start : start + 250]), return_attention_weights=True)
        for layer, w in enumerate(weights):
            w = w[:, :, 1:, 1:]
            w = w / w.sum(dim=-1, keepdim=True)
            total[layer] += (w * distance).sum(dim=-1).mean(dim=-1).sum(dim=0)
    return (total / images.size(0)).cpu().numpy()


def position_embedding_cosine_similarity(model: VisionTransformer) -> np.ndarray | None:
    # パッチの位置どうしの位置埋め込みの余弦類似度(N x N)。[CLS] の位置を除く
    if model.position_embedding is None:
        return None
    e = model.position_embedding[0, 1:].detach().float()
    e = e / e.norm(dim=-1, keepdim=True)
    return (e @ e.t()).cpu().numpy()


def run(key: tuple, learning_rate: float, purpose: str) -> dict:
    assert purpose in ("calibration", "main")
    t0 = time.time()
    arch = key[0]
    model = build_model_for(key)
    history = train(
        model,
        key,
        learning_rate,
        TRAIN_STEPS,
        INTERMEDIATE_EVAL_STEPS,
        lambda m: {"val_accuracy": evaluate(m, VALIDATION_IMAGES, VALIDATION_LABELS, arch)["accuracy"]},
    )
    record = {
        "key": key,
        "purpose": purpose,
        "learning_rate": learning_rate,
        "train_steps": TRAIN_STEPS,
        "crop_padding": CROP_PADDING,
        "num_train_examples": int(SUBSET_INDICES_DEVICE[key[3]].numel()),
        "non_embedding_parameters": count_vit_non_embedding_parameters(model) if arch == "vit" else None,
        "history": history,
        "val": summary(evaluate(model, VALIDATION_IMAGES, VALIDATION_LABELS, arch)),  # 最終ステップの重み
    }
    if purpose == "main":  # テスト集合は本番の学習でのみ評価する(較正では評価しない)
        record["test"] = summary(evaluate(model, TEST_IMAGES, TEST_LABELS, arch))
        record["common_subset_train_accuracy"] = evaluate(
            model, COMMON_SUBSET_IMAGES, COMMON_SUBSET_LABELS, arch
        )["accuracy"]
        if is_a_key(key):
            record["shuffled_test"] = summary(
                evaluate(
                    model,
                    TEST_IMAGES,
                    TEST_LABELS,
                    arch,
                    lambda x: permute_patches(x, STANDARD_PATCH_SIZE, SHUFFLE_PERMUTATION),
                )
            )
            if key[4] == 0:
                record["attention_distance"] = mean_attention_distance(
                    model, VALIDATION_IMAGES[:ATTENTION_DISTANCE_IMAGES]
                )
                record["position_similarity"] = position_embedding_cosine_similarity(model)
    record["seconds"] = time.time() - t0
    del model
    empty_device_cache()
    return record


def describe(record: dict) -> str:
    key = record["key"]
    name = f"ViT P={key[1]} {key[2]}" if key[0] == "vit" else "CNN"
    text = f"{name} f={key[3]:.4f} s={key[4]} lr={record['learning_rate']:g}: val {record['val']['accuracy']:.4f}"
    if "test" in record:
        text += f"、test {record['test']['accuracy']:.4f}(CE {record['test']['cross_entropy']:.3f})"
    skipped = sum(record["history"]["step_skipped"])
    return text + f"、最終の訓練損失 {record['history']['loss'][-1]:.3f}、飛ばしたステップ {skipped}、{record['seconds']:.0f}s"


def ols_slope(x, y) -> np.ndarray:
    # 最後の軸について y を x に最小二乗で回帰した傾き(y は (..., L))
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc = x - x.mean()
    return (y - y.mean(axis=-1, keepdims=True)) @ xc / (xc @ xc)


def judge(value: float, sigma: float) -> str:
    # 支持 / 反証 / 判定不能(前提不成立は 6.12 節で前提条件から決める)
    if value > SIGMA_MULTIPLIER * sigma:
        return "支持"
    if value < -SIGMA_MULTIPLIER * sigma:
        return "反証"
    return "判定不能"


_tag = "[動作確認のみ、結論ではない] " if SMOKE_TEST else ""
_plot_tag = "[smoke test, not a result] " if SMOKE_TEST else ""  # 図のタイトル用(英字のみ)
```

## 6. 実験 / Experiments



## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/05_vision_language/019_vision_transformer.ipynb
