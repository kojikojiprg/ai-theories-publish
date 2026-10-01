---
title: "LLaVA 型 Vision-Language 連結と視覚指示チューニング / LLaVA-style Vision-Language Connection and Visual Instruction Tuning(実装・実験編 1/3)"
---

この記事は後編(実装・実験編 1/3)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/021_llava_visual_instruction_tuning-theory)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/021_llava_visual_instruction_tuning-practice-2)。

## 4. 実装方針 / Implementation Policy

**`src/`に切り出す(スクラッチ実装、本トピックで新規作成)**:

- `src/data/visual_instruction.py`: 視覚指示データの生成と符号化。シーンの意味から色・位置関係の質問と答えを作る
  `build_questions()`(テンプレートは画像と図形の添字から決定的に割り当てる)、第 1 段階のキャプションの事例`build_caption_examples()`、
  画像を見ない戦略の正解率の上界`language_prior_upper_bound()`、接頭部分・視覚トークンの枠・接尾部分・応答部分を別々に符号化して
  固定長のテンソルにする`encode_visual_examples()`(応答部分の予測対象のマスクを含む)、ステップごとのミニバッチの添字
  `make_step_batches()`。区切り記号と終端記号は 016 の`src/data/instruction.py`のものを使う。
- `src/models/llava.py`: 凍結した画像 encoder の指定した層のパッチトークンを取り出す`vision_patch_features()`、projection 層
  `VisualProjection`(線形写像。2 層の多層パーセプトロンも構築できる)、埋め込みの列から言語モデルを順伝播する
  `language_model_hidden_states()`(`GPTLanguageModel.forward()`の埋め込みの後の部分と同じ手順。既存の`forward()`は変更しない)、
  視覚トークンを枠に差し込んで言語モデルに通す`LlavaStyleModel`(パッチトークン全体 / 平均プールの切り替え)、視覚トークンを含む
  指示部分から終端記号まで貪欲法で生成する`greedy_generate_with_visual_tokens()`(016 の`greedy_generate_until_stop()`と同じ規則。
  prefill のみ埋め込みの列から行う)。
- `src/training/visual_instruction_tuning.py`: 応答部分のトークンだけの損失`response_loss()`(応答の位置の隠れ状態だけを語彙に
  射影する)、2 つの段階で共通の学習ループ`train_visual_instruction()`、教師強制の負の対数尤度の評価
  `evaluate_response_negative_log_likelihood()`、生成による評価`generate_answers()`(画像を差し替えた評価にも使う)。

**既存モジュールの変更**: なし。画像 encoder は 019 の`VisionTransformer`(020 の`CLIPDualEncoder`の`vision`)、言語モデルは 008 の
`GPTLanguageModel`、LoRA は 012 の`apply_lora()`、optimizer は 007・020 の`AdamW`(`foreach=True`)、学習率のスケジュールは 007 の
warmup + cosine、生成の採点は 016 の`score_generated_response()`、ブートストラップは 015 の`paired_cluster_bootstrap_ratio_of_sums()`を
そのまま使う。既存のファイルを変更しないので、後方互換性の確認(変更前のコミットとの数値比較)は要らない。

**ノートブック内に直接書く(021 固有)**: 条件の定義・学習率の較正・判定・可視化・スケーリングの計測、線形プローブ(実験 B の診断量)。

**アップロード方針**: 本トピックで学習するモデル(projection 層と LoRA)は、すべて条件間の比較のためのものであり、後続トピックの
入力にも、読者が単体で取得する対象にもならない。保存もアップロードもしない(アップロードのセルも置かない)。

**生成物の置き場所**:

| 生成物 | 置き場所 |
|---|---|
| 描画したシーン(学習用・検証用・評価用の画像)、画像 encoder のパッチ特徴、符号化した事例 | メモリ上のみ(規則とシードから決定的に再生成でき、全体でも数十秒で作れるため、`.cache/`にも置かない) |
| Hugging Face Hub から取得したモデル・トークナイザ | `huggingface_hub`の既定のキャッシュ(Colab ではセッションの終了とともに破棄される) |
| 学習したモデル(projection 層・LoRA) | メモリ上のみ(評価の直後に破棄する。第 1 段階の後の projection 層は、同じシードの第 2 段階の起点として学習の間だけ保持する) |
| 判定の記録 | 判定と前提条件を計算した各セルの出力(6.7・6.8・6.11 節) |

**外部からの取得**(Colab のセットアップセルのリポジトリの取得と依存関係のインストールを除く):

| 取得するもの | リポジトリ | 用途 |
|---|---|---|
| 画像 encoder(`model_state.pt`・`config.json`) | `kojikojiprg/ai-theories-clip-synthetic-scenes`(`main`、020 の NegCLIP・シード 0) | 凍結した画像 encoder $g$ |
| 言語モデル(`model_state.pt`・`config.json`) | `kojikojiprg/ai-theories-small-gpt-en`(`main`、008) | 凍結した言語モデル $f$ |
| トークナイザ(`tokenizer.json`) | `kojikojiprg/ai-theories-tokenizer-en` | 008 の英語の BPE(Byte Pair Encoding)トークナイザ |

画像とキャプションは`src/data/synthetic_scenes.py`の規則から生成する(020 の画像のミラーは使わない)。取得した 2 つのモデルの
`model_state.pt`の SHA-256 を 5.3 節で印字し、020 の画像 encoder はモデルカードに記載された SHA-256 と一致することを確かめる。

## 5. 実装 / Implementation

### 5.1 環境セットアップ(Google Colab)

`SMOKE_TEST`(スモークテストか本番か)はこのセルでのみ切り替える。実行環境はこのセルで 1 回だけ印字する。

**精度と決定性**: 学習・評価はすべて FP32 で行う。`torch.use_deterministic_algorithms(True)`は使わない(応答の位置の隠れ状態を
真偽値のマスクで取り出す演算の逆伝播などは、CUDA では演算の順序が非決定的になりうる)。同じシードの学習を繰り返しても結果が
bit 単位では一致しないことがあるが、条件間の対応付け(同じシードの条件どうしで projection 層・LoRA の初期値とミニバッチの順序を
揃えること)は乱数の生成器によって決まり、演算の決定性には依存しない(6.10 節で確かめる)。


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
print(f"SMOKE_TEST={SMOKE_TEST}、精度 FP32、決定的な演算の強制: {torch.are_deterministic_algorithms_enabled()}")
```

    /content/ai-theories
    [2mUsing Python 3.13.15 environment at: /usr[0m
    [2mChecked [1m60 packages[0m [2min 300ms[0m[0m
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
      コミット / git commit                  : 7cc0ceb2943cbba2c3c4075c557ddb3c4510d798
      未コミットの変更 / uncommitted changes : なし
      実行日時 (UTC)                         : 2026-10-01T08:21:21+00:00
    SMOKE_TEST=False、精度 FP32、決定的な演算の強制: False



```python
import copy
import functools
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download
from torch import nn
from torch.nn import functional

from src.data.instruction import END_MARKER, score_generated_response
from src.data.synthetic_scenes import (
    SHAPES,
    CaptionVocabulary,
    build_caption_universe,
    normalize_scene_images,
    render_scenes,
    repeat_caption_ids,
)
from src.data.tokenizer import load_bpe_id_tokenizer_from_hub
from src.data.visual_instruction import (
    ANSWER_VOCABULARY,
    COLOR_TEMPLATES,
    PROMPT_PREFIX,
    QUESTION_TYPES,
    SPATIAL_TEMPLATES,
    build_caption_examples,
    build_questions,
    encode_visual_examples,
    language_prior_upper_bound,
    make_step_batches,
)
from src.layers.feedforward import SwiGLUFeedForwardNetwork
from src.layers.lora import LoRALinear, apply_lora, compute_lora_parameter_count
from src.layers.normalization import RMSNorm
from src.layers.positional_encoding import RotaryPositionEmbedding
from src.models.clip import CausalTextTransformer, CLIPDualEncoder
from src.models.gpt import GPTLanguageModel
from src.models.llava import (
    LlavaStyleModel,
    VisualProjection,
    greedy_generate_with_visual_tokens,
    language_model_hidden_states,
    vision_patch_features,
)
from src.models.vit import VisionTransformer
from src.training.instruction_tuning import compute_instruction_tuning_loss
from src.training.optimizer import AdamW
from src.training.schedule import compute_warmup_cosine_learning_rate
from src.training.visual_instruction_tuning import (
    evaluate_response_negative_log_likelihood,
    generate_answers,
    response_loss,
    train_visual_instruction,
)
from src.utils.statistics import fit_power_law_exponent, paired_cluster_bootstrap_ratio_of_sums

ROOT = Path.cwd()
CLIP_REPO_ID = "kojikojiprg/ai-theories-clip-synthetic-scenes"
CLIP_REVISION = "main"
CLIP_STATE_SHA256 = "efc2ab9dcd4730f4fe4b74b13c66f33d803fe7c65d9fa75929a8a13f425f1c17"  # 020 のモデルカードの値
MODEL_REPO_ID = "kojikojiprg/ai-theories-small-gpt-en"
MODEL_REVISION = "main"
TOKENIZER_REPO_ID = "kojikojiprg/ai-theories-tokenizer-en"


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


def sha256_of_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_of_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


precondition_status: dict[str, bool] = {}  # 前提条件の成否(6.1 節で宣言、各節で記録)
```

### 5.2 スケールの設定(`SMOKE_TEST`の配線)

水準の定義をこの 1 箇所に集約する。

**縮小の規則**: スモークテストは、本番と **同じステップ数・同じデータ・同じモデル** で行う(本番で起きうる学習の崩壊や前提条件の
不成立を、本番の前に同じ規模で検出するため)。縮小するのは次の 2 つだけである。

- **シード数**: 本番は 5(削った段階で 3)、スモークテストは 3(削った段階で 2)。「削る前 > 削った後 $\ge 2$」の順序関係を保ち、
  各段階で何が変わるか(L の有無、どの実験のシード数か)は本番と同じにする。
- **ブートストラップの反復回数**: 本番 10,000 回、スモークテスト 1,000 回。

**縮小しないもの**: データ(学習用 11,552 枚・検証用 1,444 枚・評価用 1,444 枚)、ステップ数 $T_1$・$T_2$、バッチサイズ、学習率の
較正の格子と拡張の規則、学習率のスケジュールの形、途中の評価の位置、実行計画の表の構造と優先順位、前提条件と判定の閾値、
スケーリングの計測点、線形プローブの設定。

**テスト専用の上書き**: 環境変数`AI_THEORIES_FORCE_PLAN`(0〜7)が設定されているときのみ、6.4 節で見積もりによる計画の選択の代わりに
その計画を使う(スモークテストで下位の計画の経路を確かめるため)。本番では受け付けず、このセルで停止する。コミットする出力は
上書きなしの実行のものである。

**実効水準の照合**: このセルで印字した水準(ステップ数・シード数・計画の表)が、実際の学習で使われた値と一致することを 6.10 節の
アサーションで確かめる(`SMOKE_TEST`の配線漏れの検出)。


```python
# --- 全水準で共通の定数(本番実行前に宣言し、SMOKE_TEST で変えない) ---
TRAIN_COPIES = 8  # 学習用: 学習用のキャプション 1,444 個 x 8 枚 = 11,552 枚
VALIDATION_COPIES = 1  # 検証用(学習率の較正と前提条件 P1 にのみ使う)
EVAL_COPIES = 1  # 評価用(判定に使う)
TRAIN_RENDER_SEED, VALIDATION_RENDER_SEED, EVAL_RENDER_SEED = 21_001, 21_002, 21_003
SWAP_SEED = 21_004  # 前提条件 P2 の画像の差し替えの置換
BATCH_SIZE = 32
STAGE1_STEPS = 722  # T_1(第 1 段階のデータの 2 エポック、6.2 節)
STAGE2_STEPS = 1444  # T_2(第 2 段階のデータの 1 エポック、6.2 節)
STAGE1_LEARNING_RATE = 6.4e-2  # 第 1 段階の学習率(較正せず規則で固定する、6.2 節)
LORA_RANK, LORA_ALPHA = 8, 8.0  # 012・016 と同じ(alpha / r = 1)
LORA_TARGET_MODULES = ("w_q", "w_v")  # Query・Value 射影(012・016 と同じ)
WEIGHT_DECAY = 0.0  # 012・016 と同じ
WARMUP_RATIO = 0.1  # 007・008・012・016 と同じ
MIN_LEARNING_RATE_RATIO = 0.01
GRADIENT_CLIP_THRESHOLD = 1.0
PROJECTION_KIND = "linear"  # LLaVA の線形写像(3.3 節)
LR_CENTER = {  # 第 2 段階の学習率の格子の中心(段階, 条件)。6.2 節のパイロットで決めた
    ("stage2", "P"): 8e-3,
    ("stage2", "R"): 8e-3,
    ("stage2", "M"): 8e-3,
}
LR_GRID_MULTIPLIERS = (0.5, 1.0, 2.0)  # 格子 = 中心 x {1/2, 1, 2}(公比 2)
LR_GRID_RATIO = 2.0
MAX_NEW_TOKENS = 12  # 生成の上限トークン数(応答部分の最大長 9 に余裕を持たせる、5.4 節で確認)
INTERMEDIATE_EVAL_FRACTIONS = (0.25, 0.5, 0.75)  # 第 2 段階の途中の検証集合の評価の位置(診断量)
P1_NLL_RATIO = 0.5  # P1: 学習後の検証集合の負の対数尤度 <= 開始時の値 x 0.5
P2_SWAPPED_MAX = 0.5  # P2: 画像を差し替えたときの正解率 <= 0.5
P3_ROOM_MAX = 0.95  # P3: 基準の条件の正解率 <= 0.95
SIGMA_MULTIPLIER = 2.0  # 判定の閾値は対比量の標準偏差の 2 倍
PROBE_STEPS = 500  # 線形プローブ: 全バッチの Adam のステップ数
PROBE_LEARNING_RATE = 1e-2
PROBE_WEIGHT_DECAY = 1e-4
EVAL_BATCH_SIZE = 256  # 評価の 1 回の順伝播の事例数(結果には影響しない)

# 乱数シード(学習のシード s とは独立に固定するもの)
# 学習のシード s: projection 層の初期化 21100 + s、第 1 段階のミニバッチ 21200 + s、LoRA の初期化 21300 + s、第 2 段階のミニバッチ 21400 + s
PROJECTION_SEED_BASE, STAGE1_DATA_SEED_BASE, LORA_SEED_BASE, STAGE2_DATA_SEED_BASE = 21_100, 21_200, 21_300, 21_400
CALIBRATION_SEED_INDEX = 90  # 学習率の較正専用のシード(実験のシード 0〜4 と共有しない)
TIMING_SEED_INDEX = 91  # スケーリングの計測専用のシード
PROBE_SEED = 21_500
BOOTSTRAP_SEED = 21_600
SESSION_BUDGET_SECONDS = 120 * 60  # 1 セッションの予算(T4 で 120 分)

# 条件の対応表(6.1 節)。pooling: 視覚トークンの取り方、stage1: 第 1 段階の有無、stage2_steps: 第 2 段階のステップ数
CONDITIONS = {
    "P": {"pooling": "patch", "stage1": True, "stage2_steps": STAGE2_STEPS},
    "R": {"pooling": "patch", "stage1": False, "stage2_steps": STAGE2_STEPS},
    "L": {"pooling": "patch", "stage1": False, "stage2_steps": STAGE1_STEPS + STAGE2_STEPS},
    "M": {"pooling": "mean", "stage1": True, "stage2_steps": STAGE2_STEPS},
}

# --- 水準(SMOKE_TEST で変わるのはシード数とブートストラップの反復回数のみ) ---
LEVELS = {
    "smoke": {"BOOTSTRAP_RESAMPLES": 1_000},
    "prod": {"BOOTSTRAP_RESAMPLES": 10_000},
}
# 削る段階(6.1 節)。順序: L(診断量)-> 実験 A のシード数 -> 実験 B のシード数
STAGES = {
    "prod": {
        0: {"SEEDS_A": 5, "SEEDS_B": 5, "RUN_L": True},
        1: {"SEEDS_A": 5, "SEEDS_B": 5, "RUN_L": False},
        2: {"SEEDS_A": 3, "SEEDS_B": 5, "RUN_L": False},
        3: {"SEEDS_A": 3, "SEEDS_B": 3, "RUN_L": False},
    },
    "smoke": {  # 本番と同じ構造(シード数 5 -> 3 を 3 -> 2 に縮小)
        0: {"SEEDS_A": 3, "SEEDS_B": 3, "RUN_L": True},
        1: {"SEEDS_A": 3, "SEEDS_B": 3, "RUN_L": False},
        2: {"SEEDS_A": 2, "SEEDS_B": 3, "RUN_L": False},
        3: {"SEEDS_A": 2, "SEEDS_B": 2, "RUN_L": False},
    },
}
# スケーリングの計測点(ステップ数・評価の事例数、6.3 節)。水準によらず同じ
SCALING_STEP_COUNTS = (16, 32, 64)
SCALING_WARMUP_STEPS = 8
SCALING_EVAL_SIZES = (256, 512, 1024)


def build_plans() -> list[dict]:
    # 実行計画(6.1 節): 計画 0〜3 は較正の方式 "all" で段階 0〜3、計画 4〜7 は "representative" で段階 0〜3
    plans = [{"stage": k, "calibration_mode": mode} for mode in ("all", "representative") for k in range(4)]
    return [{"plan": i} | p for i, p in enumerate(plans)]


PLANS = build_plans()
CURRENT_LEVEL_NAME = "smoke" if SMOKE_TEST else "prod"
CFG = LEVELS[CURRENT_LEVEL_NAME]
BOOTSTRAP_RESAMPLES = CFG["BOOTSTRAP_RESAMPLES"]
FORCED_PLAN_VALUE = os.environ.get("AI_THEORIES_FORCE_PLAN")
if FORCED_PLAN_VALUE is not None and not SMOKE_TEST:
    raise RuntimeError(
        f"本番(SMOKE_TEST=False)では計画の強制(AI_THEORIES_FORCE_PLAN={FORCED_PLAN_VALUE!r})を受け付けない。"
        "環境変数を削除して再実行すること。"
    )
FORCED_PLAN = None if FORCED_PLAN_VALUE is None else int(FORCED_PLAN_VALUE)
assert FORCED_PLAN is None or 0 <= FORCED_PLAN < len(PLANS), FORCED_PLAN
# 段階・較正の方式・シード数は 6.4 節で実行計画を選んだ後に決まる


def warmup_steps_for(num_steps: int) -> int:
    return max(1, round(WARMUP_RATIO * num_steps))


def intermediate_eval_steps_for(num_steps: int) -> tuple[int, ...]:
    return tuple(round(f * num_steps) for f in INTERMEDIATE_EVAL_FRACTIONS)


def lr_grid_for(stage: str, condition: str) -> tuple[float, ...]:
    return tuple(LR_CENTER[(stage, condition)] * m for m in LR_GRID_MULTIPLIERS)


def seeds_for(condition: str, stage_cfg: dict) -> int:
    # 条件ごとのシード数(P は実験 A・B で共有するので max(S_A, S_B))
    return {
        "P": max(stage_cfg["SEEDS_A"], stage_cfg["SEEDS_B"]),
        "R": stage_cfg["SEEDS_A"],
        "L": stage_cfg["SEEDS_A"] if stage_cfg["RUN_L"] else 0,
        "M": stage_cfg["SEEDS_B"],
    }[condition]


# --- 縮小規則と水準の構造の確認 ---
assert set(LEVELS["smoke"]) == set(LEVELS["prod"])
assert LEVELS["smoke"]["BOOTSTRAP_RESAMPLES"] < LEVELS["prod"]["BOOTSTRAP_RESAMPLES"]
for _name, _stages in STAGES.items():
    assert set(_stages) == {0, 1, 2, 3}
    for _k in range(1, 4):  # 段階が上がるほど、どの量も減るか変わらない
        assert _stages[_k]["SEEDS_A"] <= _stages[_k - 1]["SEEDS_A"]
        assert _stages[_k]["SEEDS_B"] <= _stages[_k - 1]["SEEDS_B"]
        assert _stages[_k]["RUN_L"] <= _stages[_k - 1]["RUN_L"]
    for _stage in _stages.values():
        assert min(_stage["SEEDS_A"], _stage["SEEDS_B"]) >= 2
for _k in range(1, 4):  # 本番とスモークテストで、各段階で何が変わるかが同じ(構造を保つ縮小)
    for _key in ("SEEDS_A", "SEEDS_B", "RUN_L"):
        assert (STAGES["prod"][_k][_key] == STAGES["prod"][_k - 1][_key]) == (
            STAGES["smoke"][_k][_key] == STAGES["smoke"][_k - 1][_key]
        )
assert len(PLANS) == 8 and [p["plan"] for p in PLANS] == list(range(8))
assert [p["stage"] for p in PLANS] == [0, 1, 2, 3] * 2
assert [p["calibration_mode"] for p in PLANS] == ["all"] * 4 + ["representative"] * 4
assert all(b == 2 * a for a, b in itertools.pairwise(SCALING_STEP_COUNTS))
assert all(b == 2 * a for a, b in itertools.pairwise(SCALING_EVAL_SIZES))
for _key in LR_CENTER:
    _grid = lr_grid_for(*_key)
    assert all(math.isclose(b / a, LR_GRID_RATIO) for a, b in itertools.pairwise(_grid))
for _steps in (STAGE1_STEPS, STAGE2_STEPS, STAGE1_STEPS + STAGE2_STEPS):
    assert warmup_steps_for(_steps) < _steps and len(set(intermediate_eval_steps_for(_steps))) == 3
assert CONDITIONS["L"]["stage2_steps"] == CONDITIONS["P"]["stage2_steps"] + STAGE1_STEPS  # L は総ステップ数を P に揃える

print(f"水準 {CURRENT_LEVEL_NAME!r}: {json.dumps(CFG)}(シード数は削る段階で決まる)")
print(
    f"ステップ数: 第 1 段階 T_1 = {STAGE1_STEPS}(warmup {warmup_steps_for(STAGE1_STEPS)})、第 2 段階 T_2 = {STAGE2_STEPS}"
    f"(warmup {warmup_steps_for(STAGE2_STEPS)}、途中の評価 {intermediate_eval_steps_for(STAGE2_STEPS)})、"
    f"L の第 2 段階 {STAGE1_STEPS + STAGE2_STEPS}、バッチ {BATCH_SIZE}"
)
print(
    f"第 1 段階の学習率 {STAGE1_LEARNING_RATE}(固定)。学習: AdamW(重み減衰 {WEIGHT_DECAY}、foreach)、warmup {WARMUP_RATIO:.0%} + cosine(下限 x{MIN_LEARNING_RATE_RATIO})、"
    f"gradient clipping {GRADIENT_CLIP_THRESHOLD}、LoRA r = {LORA_RANK}・alpha = {LORA_ALPHA}・対象 {LORA_TARGET_MODULES}、"
    f"projection 層 {PROJECTION_KIND}"
)
for _key in LR_CENTER:
    print(f"  学習率の格子 {_key}: {tuple(float(f'{x:.3g}') for x in lr_grid_for(*_key))}")
print(f"条件: {json.dumps(CONDITIONS)}")
print(f"削る段階({CURRENT_LEVEL_NAME!r}): {json.dumps(STAGES[CURRENT_LEVEL_NAME])}")
print("実行計画(6.4 節で選ぶ、番号の小さいほど優先):")
for _p in PLANS:
    _cfg = STAGES[CURRENT_LEVEL_NAME][_p["stage"]]
    print(
        f"  計画 {_p['plan']}: 段階 {_p['stage']}、較正の方式 {_p['calibration_mode']!r}、"
        + "、".join(f"{c} {seeds_for(c, _cfg)} シード" for c in CONDITIONS)
    )
print(f"スケーリングの計測点: ステップ数 {SCALING_STEP_COUNTS}(ウォームアップ {SCALING_WARMUP_STEPS})、評価の事例数 {SCALING_EVAL_SIZES}")
if FORCED_PLAN is not None:
    print(f"*** テスト専用の上書き: AI_THEORIES_FORCE_PLAN = {FORCED_PLAN}(6.4 節で見積もりの代わりにこの計画を使う) ***")
```

    水準 'prod': {"BOOTSTRAP_RESAMPLES": 10000}(シード数は削る段階で決まる)
    ステップ数: 第 1 段階 T_1 = 722(warmup 72)、第 2 段階 T_2 = 1444(warmup 144、途中の評価 (361, 722, 1083))、L の第 2 段階 2166、バッチ 32
    第 1 段階の学習率 0.064(固定)。学習: AdamW(重み減衰 0.0、foreach)、warmup 10% + cosine(下限 x0.01)、gradient clipping 1.0、LoRA r = 8・alpha = 8.0・対象 ('w_q', 'w_v')、projection 層 linear
      学習率の格子 ('stage2', 'P'): (0.004, 0.008, 0.016)
      学習率の格子 ('stage2', 'R'): (0.004, 0.008, 0.016)
      学習率の格子 ('stage2', 'M'): (0.004, 0.008, 0.016)
    条件: {"P": {"pooling": "patch", "stage1": true, "stage2_steps": 1444}, "R": {"pooling": "patch", "stage1": false, "stage2_steps": 1444}, "L": {"pooling": "patch", "stage1": false, "stage2_steps": 2166}, "M": {"pooling": "mean", "stage1": true, "stage2_steps": 1444}}
    削る段階('prod'): {"0": {"SEEDS_A": 5, "SEEDS_B": 5, "RUN_L": true}, "1": {"SEEDS_A": 5, "SEEDS_B": 5, "RUN_L": false}, "2": {"SEEDS_A": 3, "SEEDS_B": 5, "RUN_L": false}, "3": {"SEEDS_A": 3, "SEEDS_B": 3, "RUN_L": false}}
    実行計画(6.4 節で選ぶ、番号の小さいほど優先):
      計画 0: 段階 0、較正の方式 'all'、P 5 シード、R 5 シード、L 5 シード、M 5 シード
      計画 1: 段階 1、較正の方式 'all'、P 5 シード、R 5 シード、L 0 シード、M 5 シード
      計画 2: 段階 2、較正の方式 'all'、P 5 シード、R 3 シード、L 0 シード、M 5 シード
      計画 3: 段階 3、較正の方式 'all'、P 3 シード、R 3 シード、L 0 シード、M 3 シード
      計画 4: 段階 0、較正の方式 'representative'、P 5 シード、R 5 シード、L 5 シード、M 5 シード
      計画 5: 段階 1、較正の方式 'representative'、P 5 シード、R 5 シード、L 0 シード、M 5 シード
      計画 6: 段階 2、較正の方式 'representative'、P 5 シード、R 3 シード、L 0 シード、M 5 シード
      計画 7: 段階 3、較正の方式 'representative'、P 3 シード、R 3 シード、L 0 シード、M 3 シード
    スケーリングの計測点: ステップ数 (16, 32, 64)(ウォームアップ 8)、評価の事例数 (256, 512, 1024)


### 5.3 画像 encoder・言語モデル・トークナイザの取得

- 画像 encoder: `kojikojiprg/ai-theories-clip-synthetic-scenes`の`main`(020 の NegCLIP・シード 0)。`config.json`から`CLIPDualEncoder`を
  構築して重みを読み込み、`vision`(019 の`VisionTransformer`)だけを使う。`model_state.pt`の SHA-256 が 020 のモデルカードの値と
  一致することを確かめる。画像 encoder のパラメータはすべて`requires_grad=False`にする。
- 言語モデル: `kojikojiprg/ai-theories-small-gpt-en`の`main`(008)。モデルの構成は取得した`config.json`から読む。各条件は
  Hub から読み込み直す代わりに、読み込んだ重みの複製から始める。
- トークナイザ: `kojikojiprg/ai-theories-tokenizer-en`。


```python
_t0_load = time.time()
tokenizer, _tokenizer_from_hub = load_bpe_id_tokenizer_from_hub(TOKENIZER_REPO_ID)
assert _tokenizer_from_hub, "トークナイザを Hugging Face Hub から取得できなかった"

# --- 画像 encoder(020) ---
CLIP_CONFIG = json.loads(Path(hf_hub_download(CLIP_REPO_ID, "config.json", revision=CLIP_REVISION)).read_text())
_clip_state_path = hf_hub_download(CLIP_REPO_ID, "model_state.pt", revision=CLIP_REVISION)
assert sha256_of_file(_clip_state_path) == CLIP_STATE_SHA256 == CLIP_CONFIG["model_state_sha256"]
VISION_CONFIG = CLIP_CONFIG["vision_config"]
_text_config = CLIP_CONFIG["text_config"]
_clip = CLIPDualEncoder(
    VisionTransformer(**VISION_CONFIG),
    CausalTextTransformer(
        _text_config["vocabulary_size"], _text_config["context_length"], _text_config["d_model"],
        _text_config["num_layers"], _text_config["num_heads"], _text_config["d_ff"], _text_config["end_token_id"],
    ),
    CLIP_CONFIG["embedding_dim"],
)
_result = _clip.load_state_dict(torch.load(_clip_state_path, map_location="cpu"))
assert not _result.missing_keys and not _result.unexpected_keys
VISION = _clip.vision.to(device).eval()
for _p in VISION.parameters():
    _p.requires_grad_(False)
del _clip
VISION_NUM_LAYERS = len(VISION.blocks)
VISION_LAYER = VISION_NUM_LAYERS - 1  # 最終層の一つ前の層(通すブロックの数、3.4 節)
VISION_DIM = VISION_CONFIG["d_model"]
NUM_PATCHES = VISION.num_patches  # N = HW / P^2
assert NUM_PATCHES == (VISION_CONFIG["image_size"] // VISION_CONFIG["patch_size"]) ** 2

# --- 言語モデル(008) ---
MODEL_CONFIG = json.loads(Path(hf_hub_download(MODEL_REPO_ID, "config.json", revision=MODEL_REVISION)).read_text())
MODEL_STATE_PATH = hf_hub_download(MODEL_REPO_ID, "model_state.pt", revision=MODEL_REVISION)
assert MODEL_CONFIG["positional_encoding"] == "rope"
assert MODEL_CONFIG["normalization"] == "rmsnorm" and MODEL_CONFIG["feed_forward"] == "swiglu"
assert MODEL_CONFIG["norm_first"] and MODEL_CONFIG["dropout"] == 0.0
assert tokenizer.vocab_size == MODEL_CONFIG["vocabulary_size"]
CONTEXT_LENGTH = MODEL_CONFIG["sequence_length"]  # 008 のモデルの文脈長 S_max
D_MODEL = MODEL_CONFIG["d_model"]
NUM_LAYERS = MODEL_CONFIG["num_layers"]
_BASE_LM_STATE = torch.load(MODEL_STATE_PATH, map_location="cpu")


def build_language_model() -> GPTLanguageModel:
    # 008 の構成で構築し、取得した重みを読み込む(全パラメータを凍結した状態で返す)
    model = GPTLanguageModel(
        vocabulary_size=MODEL_CONFIG["vocabulary_size"],
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=MODEL_CONFIG["num_heads"],
        d_ff=MODEL_CONFIG["d_ff"],
        max_sequence_length=CONTEXT_LENGTH,
        positional_transform=RotaryPositionEmbedding(D_MODEL // MODEL_CONFIG["num_heads"], max_position=CONTEXT_LENGTH),
        normalization_factory=RMSNorm,
        feed_forward_factory=functools.partial(SwiGLUFeedForwardNetwork, D_MODEL, MODEL_CONFIG["swiglu_d_ff"]),
        tie_embeddings=MODEL_CONFIG["tie_embeddings"],
    )
    result = model.load_state_dict(_BASE_LM_STATE)
    assert not result.missing_keys and not result.unexpected_keys
    for p in model.parameters():
        p.requires_grad_(False)
    return model


_lm = build_language_model()
LM_TOTAL_PARAMETERS = sum(p.numel() for p in _lm.parameters())
TOKEN_EMBEDDING_NORM = float(_lm.token_embedding.weight.norm(dim=1).mean())  # 語彙の埋め込みの平均ノルム(診断量の基準)
LORA_PARAMETERS = len(LORA_TARGET_MODULES) * NUM_LAYERS * compute_lora_parameter_count(D_MODEL, D_MODEL, LORA_RANK)
PROJECTION_PARAMETERS = sum(p.numel() for p in VisualProjection(VISION_DIM, D_MODEL, PROJECTION_KIND).parameters())
LOAD_SECONDS = time.time() - _t0_load
del _lm
print(
    f"画像 encoder: {VISION_CONFIG}、SHA-256 {CLIP_STATE_SHA256[:16]}...(020 のモデルカードと一致)、"
    f"視覚トークンに使う層: {VISION_LAYER} ブロックの後(全 {VISION_NUM_LAYERS} ブロック)、パッチの数 N = {NUM_PATCHES}"
)
print(
    f"言語モデル: 層数 {NUM_LAYERS}、d_model {D_MODEL}、ヘッド数 {MODEL_CONFIG['num_heads']}、語彙サイズ {MODEL_CONFIG['vocabulary_size']}、"
    f"文脈長 {CONTEXT_LENGTH}、パラメータ数 {LM_TOTAL_PARAMETERS:,}、SHA-256 {sha256_of_file(MODEL_STATE_PATH)[:16]}..."
)
print(
    f"学習するパラメータ数: projection 層 {PROJECTION_PARAMETERS:,}(第 1 段階)、projection 層 + LoRA "
    f"{PROJECTION_PARAMETERS + LORA_PARAMETERS:,}(第 2 段階。LoRA は閉形式 2 L r (d + d) = {LORA_PARAMETERS:,})"
)
print(f"言語モデルのトークン埋め込みの平均ノルム {TOKEN_EMBEDDING_NORM:.4f}、取得と構築 {LOAD_SECONDS:.1f}s")
```


    tokenizer.json:   0%|          | 0.00/661k [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/1.81k [00:00<?, ?B/s]



    model_state.pt: reconstructing file:   0%|          |  0.00B / 6.50MB            



    model_state.pt: downloading bytes:           |  0.00B            



    config.json:   0%|          | 0.00/305 [00:00<?, ?B/s]



    model_state.pt: reconstructing file:   0%|          |  0.00B / 21.0MB            



    model_state.pt: downloading bytes:           |  0.00B            


    画像 encoder: {'image_size': 32, 'patch_size': 4, 'in_channels': 3, 'num_classes': 1, 'd_model': 128, 'num_layers': 4, 'num_heads': 4, 'd_ff': 512, 'position_embedding': 'learned'}、SHA-256 efc2ab9dcd4730f4...(020 のモデルカードと一致)、視覚トークンに使う層: 3 ブロックの後(全 4 ブロック)、パッチの数 N = 64
    言語モデル: 層数 4、d_model 256、ヘッド数 8、語彙サイズ 8192、文脈長 256、パラメータ数 5,246,208、SHA-256 c3dfd9e14a42dbe2...
    学習するパラメータ数: projection 層 33,024(第 1 段階)、projection 層 + LoRA 65,792(第 2 段階。LoRA は閉形式 2 L r (d + d) = 32,768)
    言語モデルのトークン埋め込みの平均ノルム 0.8512、取得と構築 14.7s


### 5.4 データ: 合成のシーン・質問・符号化・パッチ特徴

- **シーン**: 020 の`src/data/synthetic_scenes.py`の生成器(`build_caption_universe()`・`render_scenes()`)を使う。020 の画像 encoder の
  学習データに合わせ、**学習用のキャプション 1,444 個**(020 で除外した (色, 形) の組を含まないもの)の意味だけを使う。学習用
  (各キャプション 8 枚、シード`21001`)・検証用(各 1 枚、シード`21002`)・評価用(各 1 枚、シード`21003`)は別のシードで描くので、
  同じ意味でも画素は異なる。いずれのシードも 020 の描画のシード(20001〜20003)と異なり、020 の画像 encoder が学習中に見た画像そのもの
  ではない。
- **質問**: 各画像から、色の質問 2 個・位置関係の質問 2 個(3.5 節)。テンプレートは質問の種類ごとに 2 個で、画像の添字と図形の添字から
  決定的に割り当てる。
- **テンプレートの既知性の分割**: 学習用と評価用は **画像で** 分け、テンプレート・答えの語彙・意味(キャプション)の集合は共通にする。
  各テンプレートは学習用・評価用のどちらでも同じ割合(各 1/2)で現れ、評価用のすべての答えは学習用にも現れる(アサーション)。
  020 の実験 B では、比べる 2 つの負例の一方だけが学習で見たキャプションであったため、「既知かどうか」が比較に交絡した。本トピック
  では、どの条件・どの質問の種類でも、評価の事例は「既知のテンプレート・既知の答えの語彙で、未知の画像に答える」ものに揃う。
- **画像を見ない戦略の上界**: 評価集合で、質問の文字列ごとに最も多い答えを返す戦略の正解率(`language_prior_upper_bound()`)を、
  全体と質問の種類ごとに印字する。
- **符号化**: 接頭部分`### Instruction:\n`・視覚トークンの枠・`\n{質問}\n### Response:`・応答部分` {答え}\n### End`を別々に符号化して
  連結し、各部分の復号が元の文字列と一致することを確かめる(5.5 節でも、枠の位置のトークン ID が出力に影響しないことを確かめる)。
  系列長は視覚トークンの取り方ごとに、全事例の最大長にそろえる(右側をパディング)。
- **視覚トークン数と文脈長**: 全事例の系列長 $S = S_t + N_v$、および指示部分の長さ + 生成の上限トークン数(12)が、言語モデルの
  文脈長 256 以下であることをアサーションで確かめる。
- **パッチ特徴**: 画像 encoder は凍結しているので、全画像の最終層の一つ前の層のパッチトークンを一度だけ計算し、学習・評価で再利用する。
  同じ手順で計算し直すと bit 単位で一致することを確かめる。パッチ特徴と言語モデルのトークン埋め込みの平均ノルムを印字する
  (projection 層が写す前の尺度の違い、実験 A の診断量の基準)。
- **画像の差し替え**(前提条件 P2): 評価用の画像の固定の置換(シード`21004`)。どの画像も、自分と異なる意味(キャプション)の画像に移る。


```python
_t0_data = time.time()
VOCABULARY = CaptionVocabulary()
UNIVERSE = build_caption_universe(VOCABULARY)
CAPTION_IDS = UNIVERSE.train_caption_ids  # 020 の画像 encoder の学習データと同じ意味の集合(1,444 個)
SPLIT_SPECS = {
    "train": (TRAIN_COPIES, TRAIN_RENDER_SEED),
    "validation": (VALIDATION_COPIES, VALIDATION_RENDER_SEED),
    "eval": (EVAL_COPIES, EVAL_RENDER_SEED),
}
SCENES, MEANINGS = {}, {}
for _split, (_copies, _seed) in SPLIT_SPECS.items():
    SCENES[_split] = render_scenes(UNIVERSE, repeat_caption_ids(CAPTION_IDS, _copies), _seed)
    MEANINGS[_split] = [UNIVERSE.meanings[i] for i in SCENES[_split].caption_ids]
    assert SCENES[_split].visible_pixels.min() >= 20  # 2 つの図形がどちらも見えている
assert len(set(UNIVERSE.holdout) & {o for m in MEANINGS["train"] for o in (m.first, m.second)}) == 0
for _a, _b in itertools.combinations(SCENES, 2):  # 分割の間で同じ画像がない
    _ha = {sha256_of_tensor(x) for x in SCENES[_a].images}
    assert not _ha & {sha256_of_tensor(x) for x in SCENES[_b].images}, (_a, _b)
NUM_IMAGES = {k: len(v.caption_ids) for k, v in SCENES.items()}

# --- 質問とキャプションの事例 ---
QUESTIONS = {k: build_questions(m) for k, m in MEANINGS.items()}
CAPTION_EXAMPLES = {k: build_caption_examples(m) for k, m in MEANINGS.items()}
for _split, _qs in QUESTIONS.items():
    assert len(_qs) == 4 * NUM_IMAGES[_split]
    for _qtype, _templates in (("color", COLOR_TEMPLATES), ("spatial", SPATIAL_TEMPLATES)):
        _counts = np.bincount([q.template_index for q in _qs if q.question_type == _qtype], minlength=len(_templates))
        assert (_counts == _counts[0]).all(), (_split, _qtype, _counts)  # テンプレートの割合が等しい
        assert {q.answer for q in _qs if q.question_type == _qtype} <= set(ANSWER_VOCABULARY[_qtype])
_train_answers = {(q.question_type, q.answer) for q in QUESTIONS["train"]}
assert {(q.question_type, q.answer) for q in QUESTIONS["eval"]} <= _train_answers  # 評価の答えはすべて学習用に現れる
assert {q.question for q in QUESTIONS["eval"]} <= {q.question for q in QUESTIONS["train"]}  # 評価の質問の文字列も既知
QUESTION_TYPE_OF_EVAL = np.array([q.question_type for q in QUESTIONS["eval"]])
EVAL_IMAGE_OF_QUESTION = np.array([q.image_index for q in QUESTIONS["eval"]])
EVAL_ANSWERS = [q.answer for q in QUESTIONS["eval"]]
PRIOR_UPPER_BOUND = {"all": language_prior_upper_bound(QUESTIONS["eval"])} | {
    t: language_prior_upper_bound([q for q in QUESTIONS["eval"] if q.question_type == t]) for t in QUESTION_TYPES
}

# --- 符号化(視覚トークンの取り方ごとに、全事例の最大長にそろえる) ---
NUM_VISUAL_TOKENS = {"patch": NUM_PATCHES, "mean": 1}
PLACEHOLDER_ID = PAD_ID = 0  # 枠の位置とパディングのトークン ID(出力に影響しないことを 5.5 節で確かめる)
_kinds = {"caption": CAPTION_EXAMPLES, "question": QUESTIONS}
SEQ_LEN, ENCODED = {}, {}
for _pooling, _nv in NUM_VISUAL_TOKENS.items():
    _probe = [
        encode_visual_examples(e[s], tokenizer.encode, tokenizer.decode, _nv, CONTEXT_LENGTH, PLACEHOLDER_ID, PAD_ID)
        for e in _kinds.values() for s in SCENES
    ]
    SEQ_LEN[_pooling] = max(int(d.lengths.max()) for d in _probe)
    for _kind, _examples in _kinds.items():
        for _split in SCENES:
            _d = encode_visual_examples(
                _examples[_split], tokenizer.encode, tokenizer.decode, _nv, SEQ_LEN[_pooling], PLACEHOLDER_ID, PAD_ID
            )
            assert _d.visual_start == len(tokenizer.encode(PROMPT_PREFIX))
            ENCODED[(_pooling, _kind, _split)] = _d.to(device)
VISUAL_START = ENCODED[("patch", "question", "train")].visual_start
_response_lengths = [int((d.lengths - d.prompt_lengths).max()) for (_, k, _), d in ENCODED.items() if k == "question"]
_prompt_lengths = {p: max(int(ENCODED[(p, k, s)].prompt_lengths.max()) for k in _kinds for s in SCENES) for p in NUM_VISUAL_TOKENS}
_text_lengths = {p: (ENCODED[(p, "question", "eval")].lengths - NUM_VISUAL_TOKENS[p]).cpu() for p in NUM_VISUAL_TOKENS}
assert torch.equal(_text_lengths["patch"], _text_lengths["mean"])  # テキストの部分 S_t は視覚トークンの取り方によらない
# 視覚トークン数と文脈長(3.6 節): 全事例の S = S_t + N_v、指示部分 + 生成の上限が文脈長以下
for _pooling in NUM_VISUAL_TOKENS:
    assert SEQ_LEN[_pooling] <= CONTEXT_LENGTH, ("系列長が文脈長を超える", _pooling, SEQ_LEN[_pooling])
    assert _prompt_lengths[_pooling] + MAX_NEW_TOKENS <= CONTEXT_LENGTH, ("生成の長さが文脈長を超える", _pooling)
assert max(_response_lengths) <= MAX_NEW_TOKENS  # 正解の応答部分は生成の上限トークン数に収まる
assert SEQ_LEN["patch"] - SEQ_LEN["mean"] == NUM_PATCHES - 1

# --- パッチ特徴(凍結した画像 encoder、一度だけ計算する) ---
FEATURE_CHUNK = 1024


def compute_patch_features(images_uint8: torch.Tensor) -> torch.Tensor:
    parts = [
        vision_patch_features(VISION, normalize_scene_images(images_uint8[s : s + FEATURE_CHUNK].to(device)), VISION_LAYER)
        for s in range(0, images_uint8.size(0), FEATURE_CHUNK)
    ]
    return torch.cat(parts)


FEATURES = {k: compute_patch_features(v.images) for k, v in SCENES.items()}
for _split, _f in FEATURES.items():
    assert _f.shape == (NUM_IMAGES[_split], NUM_PATCHES, VISION_DIM) and _f.dtype == torch.float32 and not _f.requires_grad
    _again = compute_patch_features(SCENES[_split].images)  # 同じ手順で計算し直す
    assert torch.equal(_f, _again), _split
    del _again
_single = torch.cat([compute_patch_features(SCENES["eval"].images[i : i + 1]) for i in range(8)])
FEATURE_CHUNK_DIFF = float((_single - FEATURES["eval"][:8]).abs().max())  # 一度に計算する枚数を変えたときの差(参考)
PATCH_FEATURE_NORM = float(FEATURES["train"].norm(dim=-1).mean())
MEAN_POOLED_NORM = float(FEATURES["train"].mean(dim=1).norm(dim=-1).mean())

# --- 画像の差し替え(前提条件 P2): どの画像も、自分と異なる意味の画像に移る固定の置換 ---
_rng = np.random.default_rng(SWAP_SEED)
_eval_captions = SCENES["eval"].caption_ids
while True:
    SWAP_PERMUTATION = _rng.permutation(NUM_IMAGES["eval"])
    if (_eval_captions[SWAP_PERMUTATION] != _eval_captions).all():
        break
SWAPPED_EVAL_IMAGE_INDEX = torch.as_tensor(SWAP_PERMUTATION[EVAL_IMAGE_OF_QUESTION], device=device)
DATA_SECONDS = time.time() - _t0_data

print(
    "画像: " + "、".join(f"{k} {NUM_IMAGES[k]:,} 枚(シード {SPLIT_SPECS[k][1]})" for k in SCENES)
    + "。除外した組を含まない、分割の間で同じ画像がない: OK"
)
print(
    "事例: " + "、".join(f"{k} 質問 {len(QUESTIONS[k]):,}・キャプション {len(CAPTION_EXAMPLES[k]):,}" for k in SCENES)
    + "。テンプレートの割合が各分割で等しい、評価の答えと質問の文字列はすべて学習用に現れる: OK"
)
print(f"画像を見ない戦略の正解率の上界(評価集合): {json.dumps({k: round(v, 4) for k, v in PRIOR_UPPER_BOUND.items()})}")
for _qtype in QUESTION_TYPES:
    _ans = [q.answer for q in QUESTIONS["eval"] if q.question_type == _qtype]
    print(f"  {_qtype} の答えの分布(評価集合): {dict(sorted(Counter(_ans).items()))}")
print(
    f"視覚トークンの枠の開始位置 {VISUAL_START}(接頭部分 {PROMPT_PREFIX!r} のトークン数)、テキストの部分 S_t: "
    f"最小 {int(_text_lengths['patch'].min())}・最大 {int(_text_lengths['patch'].max())}"
)
for _pooling, _nv in NUM_VISUAL_TOKENS.items():
    print(
        f"  {_pooling}: N_v = {_nv}、系列長 S の最大 {SEQ_LEN[_pooling]}、指示部分の最大 {_prompt_lengths[_pooling]} + 生成の上限 {MAX_NEW_TOKENS} "
        f"= {_prompt_lengths[_pooling] + MAX_NEW_TOKENS} <= 文脈長 {CONTEXT_LENGTH}: OK"
    )
print(f"質問の応答部分の最大トークン数 {max(_response_lengths)} <= 生成の上限 {MAX_NEW_TOKENS}: OK")
print(
    f"パッチ特徴: 形状 {tuple(FEATURES['train'].shape)}(学習用)、同じ手順で計算し直すと bit 単位で一致: OK、"
    f"一度に計算する枚数を変えたときの差の最大値 {FEATURE_CHUNK_DIFF:.1e}(参考。学習・評価は常に一度だけ計算した特徴を使う)"
)
print(
    f"平均ノルム: パッチトークン {PATCH_FEATURE_NORM:.2f}、平均プール {MEAN_POOLED_NORM:.2f}、言語モデルのトークン埋め込み "
    f"{TOKEN_EMBEDDING_NORM:.4f}(比 {PATCH_FEATURE_NORM / TOKEN_EMBEDDING_NORM:.1f})"
)
print(f"画像の差し替えの置換: 全 {NUM_IMAGES['eval']:,} 枚が自分と異なる意味の画像に移る: OK。データの準備 {DATA_SECONDS:.1f}s")

_fig, _axes = plt.subplots(1, 6, figsize=(13, 2.9))
for _k, _ax in enumerate(_axes):
    _i = _k * (NUM_IMAGES["eval"] // 6) + 1
    _ax.imshow(SCENES["eval"].images[_i].permute(1, 2, 0).numpy())
    _qs = [q for q in QUESTIONS["eval"] if q.image_index == _i]
    _ax.set_title("\n".join(f"{q.question[:34]} -> {q.answer}" for q in _qs[1:4:2]), fontsize=6)
    _ax.axis("off")
plt.tight_layout()
plt.show()
```

    画像: train 11,552 枚(シード 21001)、validation 1,444 枚(シード 21002)、eval 1,444 枚(シード 21003)。除外した組を含まない、分割の間で同じ画像がない: OK
    事例: train 質問 46,208・キャプション 11,552、validation 質問 5,776・キャプション 1,444、eval 質問 5,776・キャプション 1,444。テンプレートの割合が各分割で等しい、評価の答えと質問の文字列はすべて学習用に現れる: OK
    画像を見ない戦略の正解率の上界(評価集合): {"all": 0.2261, "color": 0.1801, "spatial": 0.2722}
      color の答えの分布(評価集合): {'blue': 360, 'cyan': 360, 'green': 360, 'magenta': 364, 'orange': 360, 'red': 360, 'white': 360, 'yellow': 364}
      spatial の答えの分布(評価集合): {'above': 722, 'below': 722, 'left': 722, 'right': 722}
    視覚トークンの枠の開始位置 7(接頭部分 '### Instruction:\n' のトークン数)、テキストの部分 S_t: 最小 31・最大 43
      patch: N_v = 64、系列長 S の最大 110、指示部分の最大 100 + 生成の上限 12 = 112 <= 文脈長 256: OK
      mean: N_v = 1、系列長 S の最大 47、指示部分の最大 37 + 生成の上限 12 = 49 <= 文脈長 256: OK
    質問の応答部分の最大トークン数 9 <= 生成の上限 12: OK
    パッチ特徴: 形状 (11552, 64, 128)(学習用)、同じ手順で計算し直すと bit 単位で一致: OK、一度に計算する枚数を変えたときの差の最大値 5.2e-06(参考。学習・評価は常に一度だけ計算した特徴を使う)
    平均ノルム: パッチトークン 31.13、平均プール 28.90、言語モデルのトークン埋め込み 0.8512(比 36.6)
    画像の差し替えの置換: 全 1,444 枚が自分と異なる意味の画像に移る: OK。データの準備 15.2s



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/021_llava_visual_instruction_tuning/output_20_1.png)
    




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/05_vision_language/021_llava_visual_instruction_tuning.ipynb
