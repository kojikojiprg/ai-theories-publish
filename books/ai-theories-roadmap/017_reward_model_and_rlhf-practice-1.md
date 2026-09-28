---
title: "報酬モデルと RLHF / Reward Models and RLHF(実装・実験編 1/4)"
---

この記事は後編(実装・実験編 1/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/017_reward_model_and_rlhf-theory)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/017_reward_model_and_rlhf-practice-2)。

## 4. 実装方針 / Implementation Policy

**`src/`に切り出す(スクラッチ実装、本トピックで新規作成または拡張)**:

- `src/data/preference.py`(新規): 真の報酬 $r^*$(`compute_true_reward()`。文字単位の編集距離
  `levenshtein_distance()`は numpy で 1 行ずつ更新する動的計画法)、Bradley-Terry モデルの確率
  (`compute_bradley_terry_probability()`)、$\kappa$ の決定(`calibrate_label_scale()`)、ラベルの抽選
  (`sample_preference_labels()`)、報酬モデルに入れる系列のバッチ化(`collate_scored_sequences()`)。016 の課題の定義
  (`src/data/instruction.py`)の上に置き、018 から再利用できるようにする。
- `src/models/reward_model.py`(新規): 最終正規化層の後の隠れ状態の取り出し(`compute_final_hidden_states()`。
  `GPTLanguageModel.forward()`の計算のうち語彙への射影の直前までを同じ手順で行う。既存の`forward()`は変更しない)、
  報酬モデル(`RewardModel`)、系列のスコアリング(`score_sequences()`)。
- `src/training/reward_modeling.py`(新規): Bradley-Terry の損失(`bradley_terry_loss()`、$-\log \sigma(z)$ を
  $\log(1 + e^{-z})$ として数値的に安定に計算する)、報酬モデルの学習ループ(`train_reward_model()`)。
- `src/training/ppo.py`(新規): トークン単位の KL ペナルティ付きの報酬(`compute_kl_penalized_rewards()`)、GAE
  (`compute_gae()`)、クリップ付き目的関数(`ppo_clipped_policy_loss()`)、価値関数の損失、本体を共有する価値ヘッド
  (`PolicyWithValueHead`)、1 回のロールアウトでの更新(`ppo_update()`)。
- `src/generation/best_of_n.py`(新規): 昇順の重み $w_i$(`best_of_n_order_weights()`)、期待値の推定量
  (`best_of_n_expected_value()`・`best_of_n_expected_values()`)、KL の上界(`best_of_n_kl_upper_bound()`)。
- `src/generation/stopping.py`(拡張): temperature sampling で終端記号まで生成する`sample_generate_until_stop()`を追加する
  (top-k・top-p なし)。016 の`greedy_generate_until_stop()`と生成ループを共有するため、内部の関数に「次のトークンの選び方」
  を渡せるようにした。貪欲法の選び方は変更前と同じ`argmax`であり、016 の貪欲法の出力は変わらない(5.5 節で、
  変更前のコミットの関数との出力の一致を確かめる)。

**ノートブック内に直接書く(017 固有)**: 016 の学習データの再生成と SFT の呼び出し、LoRA のマージ、プロンプトの分割、
応答プールと選好の組の生成、実験 A〜C の判定、PPO のロールアウトと学習ループ、スケーリングの計測、アップロード。

**アップロード方針**: SFT モデル(LoRA をマージした重み)は、018 の DPO が参照方策として使う(後続トピックの入力に
なる)ので、`kojikojiprg/ai-theories-small-gpt-en`の新ブランチ`sft-synthetic`にアップロードする。構造は`main`と同一なので
ブランチで管理する。報酬モデルはアップロードしない(スカラーのヘッドで構造が変わるうえ、学習のコストが小さく、018 で
必要なら`src/`から再学習できる)。PPO の方策も、判定を置かない動作確認のためのものなのでアップロードしない。

**生成物の置き場所**:

| 生成物 | 置き場所 |
|---|---|
| SFT モデル(参照方策 $\pi_{\mathrm{ref}}$) | Hugging Face Hub の`kojikojiprg/ai-theories-small-gpt-en`のブランチ`sft-synthetic`(`UPLOAD_ARTIFACTS = True`のときのみ) |
| 英語版 Wikipedia のコーパス(モデルカードの検証 bits-per-byte 用) | `.cache/wikipedia_en/`(外部取得、データ源で命名) |
| 合成データ・応答プール・選好の組・ラベル・報酬モデル・PPO の方策 | メモリ上のみ(同一セッション内。ファイルに保存しない) |
| 判定の記録 | 判定と前提条件を計算した各セルの出力 |

外部から取得するのは、Hugging Face Hub のモデル(`kojikojiprg/ai-theories-small-gpt-en`の`main`)・トークナイザ
(`kojikojiprg/ai-theories-tokenizer-en`)・英語版 Wikipedia のコーパス(`kojikojiprg/ai-theories-corpus-en-pretraining`、
取得できない場合のみ Wikipedia から直接取得)である。

## 5. 実装 / Implementation

### 5.1 環境セットアップ(Google Colab)

`SMOKE_TEST`(スモークテストか本番か)と`UPLOAD_ARTIFACTS`(SFT モデルを Hugging Face Hub にアップロードするか)は
このセルでのみ切り替える。2 つは独立であり、本番(`SMOKE_TEST = False`)でもアップロードは`UPLOAD_ARTIFACTS = True`に
したときだけ行う。実行環境はこのセルで 1 回だけ印字する。再現性のため、CUDA の cuBLAS が決定的な演算を使うための
環境変数`CUBLAS_WORKSPACE_CONFIG`を torch の読み込み前に設定し、`torch.use_deterministic_algorithms(True)`を有効にする
(決定的な実装を持たない演算を使うと例外になるので、5.6 節で学習・生成・スコアリング・PPO の経路を一通り実行して確かめる)。


```python
# 環境セットアップ(Google Colab)
import os
import sys
import time

SMOKE_TEST = False  # Claude Code はこの True 側のみ実行する(Colab T4 では False に切り替える)
UPLOAD_ARTIFACTS = True  # True のときのみ SFT モデルを Hub にアップロードする(SMOKE_TEST とは独立)

NOTEBOOK_START_TIME = time.time()
# cuBLAS を決定的にする(torch が CUDA を初期化する前に設定する必要がある)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not os.path.isdir("/content/ai-theories"):
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

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
execution_environment = print_execution_environment(device)

print(
    f"SMOKE_TEST={SMOKE_TEST}、UPLOAD_ARTIFACTS={UPLOAD_ARTIFACTS}、"
    f"決定的な実行: {torch.are_deterministic_algorithms_enabled()}、"
    f"CUBLAS_WORKSPACE_CONFIG={os.environ['CUBLAS_WORKSPACE_CONFIG']}"
)
```

    /content/ai-theories
    [2mUsing Python 3.13.15 environment at: /usr[0m
    [2mChecked [1m59 packages[0m [2min 104ms[0m[0m
    読み込み済みのパッケージの版の食い違い: なし
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
      コミット / git commit                  : 108c33b6ff2d364a7c982e48393038e75ecd6e51
      未コミットの変更 / uncommitted changes : なし
      実行日時 (UTC)                         : 2026-09-27T23:44:12+00:00
    SMOKE_TEST=False、UPLOAD_ARTIFACTS=True、決定的な実行: True、CUBLAS_WORKSPACE_CONFIG=:4096:8



```python
import copy
import functools
import hashlib
import json
import math
import subprocess
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import nn

from src.data.instruction import (
    END_MARKER,
    TASK_NAMES,
    WORD_VOCABULARY,
    build_evaluation_examples,
    build_training_examples,
    encode_instruction_example,
    sample_word_sequences,
    score_generated_response,
)
from src.data.preference import (
    calibrate_label_scale,
    compute_bradley_terry_probability,
    compute_true_reward,
    levenshtein_distance,
    sample_preference_labels,
)
from src.data.text import (
    load_wikipedia_corpus_with_fallback,
    make_evaluation_windows,
    split_train_val_text,
)
from src.data.tokenizer import load_bpe_id_tokenizer_from_hub
from src.generation.best_of_n import (
    best_of_n_expected_value,
    best_of_n_expected_values,
    best_of_n_kl_upper_bound,
    best_of_n_order_weights,
)
from src.generation.stopping import greedy_generate_until_stop, sample_generate_until_stop
from src.layers.feedforward import SwiGLUFeedForwardNetwork
from src.layers.lora import apply_lora
from src.layers.normalization import RMSNorm
from src.layers.positional_encoding import RotaryPositionEmbedding
from src.models.gpt import GPTLanguageModel
from src.models.reward_model import RewardModel, compute_final_hidden_states, score_sequences
from src.training.instruction_tuning import (
    evaluate_instruction_negative_log_likelihood,
    make_epoch_batches,
    train_instruction_tuning,
)
from src.training.optimizer import AdamW
from src.training.ppo import (
    PolicyWithValueHead,
    collate_rollout_sequences,
    compute_gae,
    compute_kl_penalized_rewards,
    gather_response_log_probs,
    gather_response_values,
    ppo_clipped_policy_loss,
    ppo_update,
    whiten_advantages,
)
from src.training.reward_modeling import bradley_terry_loss, train_reward_model
from src.training.schedule import compute_warmup_cosine_learning_rate
from src.training.trainer import evaluate_bits_per_byte
from src.utils.statistics import compute_spearman_correlation, fit_power_law_exponent

ROOT = Path(".")
WIKIPEDIA_CACHE_DIR = ROOT / ".cache" / "wikipedia_en"  # 外部取得したコーパス(データ源で命名)
MODEL_REPO_ID = "kojikojiprg/ai-theories-small-gpt-en"
MODEL_REVISION = "main"
SFT_BRANCH = "sft-synthetic"  # SFT モデルのアップロード先のブランチ(6.14 節)
TOKENIZER_REPO_ID = "kojikojiprg/ai-theories-tokenizer-en"
CORPUS_REPO_ID = "kojikojiprg/ai-theories-corpus-en-pretraining"
MANIFEST_PATH = ROOT / "src" / "data" / "wikipedia_manifests" / "en_006_pretraining.json"
COMMIT_BEFORE_017 = "09f142d"  # src/generation/stopping.py を 017 で変更する前のコミット(5.5 節)


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

- SFT の学習ステップ数 $T_{\mathrm{SFT}}$(本番 2048・スモークテスト 32)を縮小する。バッチサイズ 32 は縮小せず、
  データ数 $= T_{\mathrm{SFT}} \times 32$(1 エポックちょうど)という関係を両水準で保つ。
- 報酬モデルの学習データ量の水準は、両水準とも **公比 4 の 4 水準** で、最大の水準 $N_A$ が標準(実験 A・B)である
  (本番 $\{1024, 4096, 16384, 65536\}$、スモークテスト $\{16, 64, 256, 1024\}$)。学習ステップ数は全水準で
  $T_{\mathrm{RM}} = N_A / 32$(最大の水準で 1 エポックちょうど)に固定する。この関係は両水準で同じである。
- 応答プールの大きさ $M$(本番 512・スモークテスト 128)を縮小する。$n$ の水準は両水準とも $2^0, 2^1, \dots, M/4$ の
  等比で($n_{\max} = M/4$)、水準の数 $L$ は本番 8・スモークテスト 6 である。前半・後半の境界は両水準とも
  「最初の $L/2$ 水準と最後の $L/2$ 水準」とする(本番: $n \le 8$ と $n \ge 16$、スモークテスト: $n \le 4$ と $n \ge 8$)。
- 評価用の入力の数 $|X|$(本番 64・スモークテスト 16)、PPO の反復回数(本番 150・スモークテスト 4)、ブートストラップの
  反復回数、P0 の評価に使う 016 の評価集合の入力の数(本番 250 = 016 の評価集合の全体、スモークテスト 16)を縮小する。
- **削る段階の表**(6.1 節)は本番とスモークテストで同じ構造にする。シード数は本番の 5 → 3 をスモークテストでは 3 → 2 に
  縮小する(「削る前 > 削った後 $\ge 2$」の順序関係を保つ)。PPO の反復回数を半分にする段階も両水準で同じ。
- **テスト専用の上書き**: 環境変数`AI_THEORIES_FORCE_STAGE`(0〜4)が設定されているときのみ、6.4 節で見積もりによる
  段階の選択の代わりにその段階を使う(スモークテストで全段階の経路を確かめるため)。本番では受け付けず、このセルで停止する。
- 課題・テンプレート・単語の語彙・SFT の学習率・LoRA の設定・サンプリングの温度・$\kappa$ の決め方・報酬モデルの
  学習率とバッチサイズ・PPO の超パラメータ(反復回数を除く)は縮小しない。


```python
# --- 全水準で共通の定数(本番実行前に宣言し、SMOKE_TEST で変えない) ---
TASK_COUNT = len(TASK_NAMES)  # K
MIN_WORDS, MAX_WORDS = 3, 8  # 入力の単語数(016 と同じ)
MAX_NEW_TOKENS = 32  # 生成の上限トークン数(016 と同じ)
TEMPERATURE = 1.0  # pi_ref からのサンプリングの温度(top-k・top-p なし)
GENERATION_BATCH_SIZE = 512  # サンプリングの 1 回の順伝播の事例数の上限(結果の再現性にのみ影響)
SCORING_BATCH_SIZE = 256  # 報酬モデルのスコアリングの 1 回の順伝播の系列数(結果には影響しない)

# SFT(016 の条件 1 のレシピ。016 の本番で最も完全一致率が高かった条件)
DATA_SEED_016 = 16_016  # 016 の合成データの乱数シード(016 と同一の学習データ・評価集合を再生成する)
NUM_EVAL_INPUTS_016 = 250  # 016 の評価集合の入力の数(再生成に必要)
SFT_SEED = 0
SFT_BATCH_SIZE = 32
SFT_LEARNING_RATE = 1e-2
LORA_RANK, LORA_ALPHA = 8, 8.0
LORA_TARGET_MODULES = ("w_q", "w_v")
WARMUP_RATIO = 0.1
MIN_LEARNING_RATE_RATIO = 0.01

# 017 のプロンプト・応答プール・選好の組
PROMPT_SEED = 17_017  # 017 のプロンプト(評価用・報酬モデルの学習用・PPO 用の入力)の乱数シード
POOL_SEED = 17_018  # 応答プールのサンプリングの乱数シード(torch.Generator)
PAIR_SEED = 17_019  # 選好の組の応答のサンプリングの乱数シード
TIMING_SEED = 17_099  # スケーリングの計測専用
KAPPA_TARGET = 0.75  # median sigma(kappa |Delta r*|) の目標(3.3 節)
PILOT_EVALUATION_SEED_017 = (
    17_999  # 017 のパイロットの評価用の入力(本番の入力と重複させない、6.2 節)
)
PILOT_EVALUATION_COUNT_017 = 64

# 報酬モデル
RM_BATCH_PAIRS = 32  # 1 ステップの組の数(2 x 32 系列)
RM_LEARNING_RATE = (
    1e-4  # 6.2 節のパイロットで決めた(ホールドアウトの真の順序との一致率のみで選んだ)
)
RM_HEAD_INIT_STD = 0.02
RM_LEVEL_RATIO = 4
RM_INIT_SEED_BASE = 17_200  # シード s の報酬モデルのヘッドの初期化は torch.manual_seed(17200 + s)
LABEL_SEED_BASE = 17_300  # シード s のラベルの抽選は np.random.default_rng(17300 + s)
EVAL_LABEL_SEED = 17_399  # 実験 A の診断量(評価用の組のラベルとの一致率)のラベル

# PPO(判定なしの動作確認)
PPO_BETAS = (0.01, 0.1)  # KL 係数の 2 水準
PPO_BATCH_SIZE = 64  # 1 反復のロールアウトのプロンプト数
PPO_MINIBATCH_SIZE = PPO_BATCH_SIZE // 2  # 1 反復のロールアウトを 2 つのミニバッチに分ける
PPO_EPOCHS = 2
PPO_CLIP_EPSILON = 0.2
PPO_VALUE_COEFFICIENT = 0.5
PPO_GAMMA, PPO_LAMBDA = 1.0, 0.95
PPO_LEARNING_RATE = 1e-3
PPO_SEED = 0
PPO_TIME_CAP_SECONDS = 15 * 60  # PPO の 2 水準の合計の上限(6.1 節)

# 前提条件・判定
P0_REFERENCE_EXACT = 0.8040  # 016 の条件 1 の完全一致率の 5 シード平均(016 の本番のセル出力)
P0_REFERENCE_SEED_STD = 0.0289  # 同じ 5 シードの標本標準偏差 s_p(016 の本番のセル出力)
P0_TOLERANCE = 3 * P0_REFERENCE_SEED_STD
P0_MIN_FORMAT_RATE = 0.99  # 016 の条件 1 の形式の遵守率は全シードで 1.0000
P0_VALIDATION_RATIO = 0.05  # 008 と同じ分割(英語版 Wikipedia のコーパスの末尾 5%、モデルカード用)
BASE_REFERENCE_BITS_PER_BYTE = 1.668067  # 008 のモデルカードの値(読み込みの確認用)
A_EQUIVALENCE_MARGIN = 0.2  # 実験 A の同等性の幅 delta
BOOTSTRAP_SEED = 0
SESSION_BUDGET_SECONDS = 120 * 60  # 1 セッションの予算(T4 で 120 分、1.1 節)

# --- 水準の定義(この 1 箇所に集約する) ---
LEVELS = {
    "smoke": {
        "SFT_STEPS": 32,
        "NUM_P0_INPUTS": 16,
        "NUM_EVAL_INPUTS": 16,
        "POOL_SIZE": 128,
        "RM_PAIRS_MAX": 1024,
        "PPO_ITERATIONS": 4,
        "BOOTSTRAP_RESAMPLES": 1_000,
    },
    "prod": {
        "SFT_STEPS": 2048,
        "NUM_P0_INPUTS": 250,
        "NUM_EVAL_INPUTS": 64,
        "POOL_SIZE": 512,
        "RM_PAIRS_MAX": 65_536,
        "PPO_ITERATIONS": 150,
        "BOOTSTRAP_RESAMPLES": 10_000,
    },
}
CURRENT_LEVEL_NAME = "smoke" if SMOKE_TEST else "prod"

FORCED_STAGE_VALUE = os.environ.get("AI_THEORIES_FORCE_STAGE")
if FORCED_STAGE_VALUE is not None and not SMOKE_TEST:
    raise RuntimeError(
        f"本番(SMOKE_TEST=False)では段階の強制(AI_THEORIES_FORCE_STAGE={FORCED_STAGE_VALUE!r})を受け付けない。"
        "環境変数を削除して再実行すること。"
    )
FORCED_STAGE = None if FORCED_STAGE_VALUE is None else int(FORCED_STAGE_VALUE)
assert FORCED_STAGE is None or FORCED_STAGE in (0, 1, 2, 3, 4), FORCED_STAGE
CFG = LEVELS[CURRENT_LEVEL_NAME]
SFT_STEPS = CFG["SFT_STEPS"]
SFT_NUM_EXAMPLES = SFT_STEPS * SFT_BATCH_SIZE  # 1 エポックちょうど
NUM_EVAL_INPUTS = CFG["NUM_EVAL_INPUTS"]  # |X|
NUM_EVAL_PROMPTS = NUM_EVAL_INPUTS * TASK_COUNT  # P = K |X|
POOL_SIZE = CFG["POOL_SIZE"]  # M
RM_PAIRS_MAX = CFG["RM_PAIRS_MAX"]  # N_A(標準の水準)
RM_STEPS = RM_PAIRS_MAX // RM_BATCH_PAIRS  # T_RM(全水準で共通)
BOOTSTRAP_RESAMPLES = CFG["BOOTSTRAP_RESAMPLES"]
N_MAX_BON = POOL_SIZE // 4  # n_max = M / 4
BON_LEVELS = tuple(2**j for j in range(int(math.log2(N_MAX_BON)) + 1))  # n = 1, 2, 4, ..., n_max
BON_HALF = len(BON_LEVELS) // 2  # 前半 = 最初の L/2 水準、後半 = 最後の L/2 水準


def rm_levels(num_levels: int, n_max: int) -> tuple[int, ...]:
    # 公比 4 の等比で n_max を最大とする num_levels 水準(小さい水準から順に)
    return tuple(n_max // RM_LEVEL_RATIO**k for k in range(num_levels - 1, -1, -1))


PROD_CFG = LEVELS["prod"]

# --- 削る段階(6.1 節)。6.4 節で、見積もりのみから予算に収まる最小の段階を選ぶ ---
# 順序: 実験 C の水準数 -> 実験 C のシード数 -> PPO の反復回数 -> 実験 A・B のシード数
STAGES = {
    "prod": {
        0: {"C_NUM_LEVELS": 4, "NUM_SEEDS_C": 5, "PPO_FRACTION": 1.0, "NUM_SEEDS_AB": 5},
        1: {"C_NUM_LEVELS": 3, "NUM_SEEDS_C": 5, "PPO_FRACTION": 1.0, "NUM_SEEDS_AB": 5},
        2: {"C_NUM_LEVELS": 3, "NUM_SEEDS_C": 3, "PPO_FRACTION": 1.0, "NUM_SEEDS_AB": 5},
        3: {"C_NUM_LEVELS": 3, "NUM_SEEDS_C": 3, "PPO_FRACTION": 0.5, "NUM_SEEDS_AB": 5},
        4: {"C_NUM_LEVELS": 3, "NUM_SEEDS_C": 3, "PPO_FRACTION": 0.5, "NUM_SEEDS_AB": 3},
    },
    "smoke": {  # 本番と同じ構造(シード数 5 -> 3 を 3 -> 2 に縮小)
        0: {"C_NUM_LEVELS": 4, "NUM_SEEDS_C": 3, "PPO_FRACTION": 1.0, "NUM_SEEDS_AB": 3},
        1: {"C_NUM_LEVELS": 3, "NUM_SEEDS_C": 3, "PPO_FRACTION": 1.0, "NUM_SEEDS_AB": 3},
        2: {"C_NUM_LEVELS": 3, "NUM_SEEDS_C": 2, "PPO_FRACTION": 1.0, "NUM_SEEDS_AB": 3},
        3: {"C_NUM_LEVELS": 3, "NUM_SEEDS_C": 2, "PPO_FRACTION": 0.5, "NUM_SEEDS_AB": 3},
        4: {"C_NUM_LEVELS": 3, "NUM_SEEDS_C": 2, "PPO_FRACTION": 0.5, "NUM_SEEDS_AB": 2},
    },
}

# --- 縮小規則の確認 ---
assert set(LEVELS["smoke"]) == set(LEVELS["prod"])
for _name, _level in LEVELS.items():
    _m = _level["POOL_SIZE"]
    _levels = rm_levels(4, _level["RM_PAIRS_MAX"])
    assert _levels[-1] == _level["RM_PAIRS_MAX"] and _levels[0] >= RM_BATCH_PAIRS // 2
    assert all(
        b == a * RM_LEVEL_RATIO for a, b in zip(_levels, _levels[1:], strict=False)
    )  # 公比 4 の等比
    assert _level["RM_PAIRS_MAX"] % RM_BATCH_PAIRS == 0
    _n_max = _m // 4
    assert 2 ** int(math.log2(_n_max)) == _n_max  # n_max は 2 のべき(等比の刻みで届く)
    assert (int(math.log2(_n_max)) + 1) % 2 == 0  # L は偶数(前半・後半が同じ水準数)
    assert int(math.log2(_n_max)) + 1 >= 6  # 各半分に 3 水準以上(傾きの推定)
    _stages = STAGES[_name]
    assert set(_stages) == {0, 1, 2, 3, 4}
    for _k in range(1, 5):  # 段階が上がるほど、どの量も減るか変わらない
        for _key in ("C_NUM_LEVELS", "NUM_SEEDS_C", "PPO_FRACTION", "NUM_SEEDS_AB"):
            assert _stages[_k][_key] <= _stages[_k - 1][_key]
    for _stage in _stages.values():
        assert 2 <= _stage["NUM_SEEDS_C"] <= _stage["NUM_SEEDS_AB"]  # C のシードは A・B の先頭部分
        assert _stage["C_NUM_LEVELS"] >= 3  # 回帰の傾きの推定に 3 水準以上
for _k in range(1, 5):  # 本番とスモークテストで、各段階で何が変わるかが同じ(構造を保つ縮小)
    for _key in ("C_NUM_LEVELS", "NUM_SEEDS_C", "PPO_FRACTION", "NUM_SEEDS_AB"):
        assert (STAGES["prod"][_k][_key] == STAGES["prod"][_k - 1][_key]) == (
            STAGES["smoke"][_k][_key] == STAGES["smoke"][_k - 1][_key]
        )
for _key in ("SFT_STEPS", "NUM_P0_INPUTS", "NUM_EVAL_INPUTS", "POOL_SIZE", "RM_PAIRS_MAX"):
    assert LEVELS["smoke"][_key] < LEVELS["prod"][_key]
assert PROD_CFG["NUM_P0_INPUTS"] == NUM_EVAL_INPUTS_016  # 本番の P0 は 016 の評価集合の全体
assert N_MAX_BON <= POOL_SIZE // 4

print(f"水準 {CURRENT_LEVEL_NAME!r}: {json.dumps(CFG)}")
print(
    f"SFT: T = {SFT_STEPS}、データ数 {SFT_NUM_EXAMPLES:,}(1 エポック)、学習率 {SFT_LEARNING_RATE}、"
    f"LoRA r = {LORA_RANK}・alpha = {LORA_ALPHA}・対象 {LORA_TARGET_MODULES}、gradient clipping なし"
)
print(
    f"評価用の入力 |X| = {NUM_EVAL_INPUTS}(プロンプト P = {NUM_EVAL_PROMPTS})、プール M = {POOL_SIZE}、"
    f"n の水準 {BON_LEVELS}(前半 {BON_LEVELS[:BON_HALF]}・後半 {BON_LEVELS[BON_HALF:]})"
)
print(
    f"報酬モデル: データ量の水準(段階 0){rm_levels(4, RM_PAIRS_MAX)}、T_RM = {RM_STEPS}、"
    f"学習率 {RM_LEARNING_RATE}、バッチ {RM_BATCH_PAIRS} 組"
)
print(f"削る段階({CURRENT_LEVEL_NAME!r}、6.4 節で選ぶ): {json.dumps(STAGES[CURRENT_LEVEL_NAME])}")
```

    水準 'prod': {"SFT_STEPS": 2048, "NUM_P0_INPUTS": 250, "NUM_EVAL_INPUTS": 64, "POOL_SIZE": 512, "RM_PAIRS_MAX": 65536, "PPO_ITERATIONS": 150, "BOOTSTRAP_RESAMPLES": 10000}
    SFT: T = 2048、データ数 65,536(1 エポック)、学習率 0.01、LoRA r = 8・alpha = 8.0・対象 ('w_q', 'w_v')、gradient clipping なし
    評価用の入力 |X| = 64(プロンプト P = 256)、プール M = 512、n の水準 (1, 2, 4, 8, 16, 32, 64, 128)(前半 (1, 2, 4, 8)・後半 (16, 32, 64, 128))
    報酬モデル: データ量の水準(段階 0)(1024, 4096, 16384, 65536)、T_RM = 2048、学習率 0.0001、バッチ 32 組
    削る段階('prod'、6.4 節で選ぶ): {"0": {"C_NUM_LEVELS": 4, "NUM_SEEDS_C": 5, "PPO_FRACTION": 1.0, "NUM_SEEDS_AB": 5}, "1": {"C_NUM_LEVELS": 3, "NUM_SEEDS_C": 5, "PPO_FRACTION": 1.0, "NUM_SEEDS_AB": 5}, "2": {"C_NUM_LEVELS": 3, "NUM_SEEDS_C": 3, "PPO_FRACTION": 1.0, "NUM_SEEDS_AB": 5}, "3": {"C_NUM_LEVELS": 3, "NUM_SEEDS_C": 3, "PPO_FRACTION": 0.5, "NUM_SEEDS_AB": 5}, "4": {"C_NUM_LEVELS": 3, "NUM_SEEDS_C": 3, "PPO_FRACTION": 0.5, "NUM_SEEDS_AB": 3}}


### 5.3 ベースモデル・トークナイザの取得(008 がアップロードしたもの)

ベースモデルは`kojikojiprg/ai-theories-small-gpt-en`の`main`、トークナイザは`kojikojiprg/ai-theories-tokenizer-en`から
取得する。モデルの構成は取得した`config.json`から読む。SFT はこの`base_model`の複製から始める。


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

CONTEXT_LENGTH = MODEL_CONFIG["sequence_length"]
D_MODEL = MODEL_CONFIG["d_model"]
NUM_LAYERS = MODEL_CONFIG["num_layers"]
NUM_HEADS = MODEL_CONFIG["num_heads"]
VOCAB_SIZE = MODEL_CONFIG["vocabulary_size"]


def build_model(state_dict: dict) -> GPTLanguageModel:
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
    result = model.load_state_dict(state_dict)
    assert not result.missing_keys and not result.unexpected_keys
    return model.to(device).eval()


def freeze(model: nn.Module) -> nn.Module:
    for p in model.parameters():
        p.requires_grad_(False)
    return model


BASE_STATE = torch.load(MODEL_STATE_PATH, map_location="cpu")
base_model = freeze(build_model(BASE_STATE))
TOTAL_PARAMETERS = sum(p.numel() for p in base_model.parameters())
print(
    f"層数 = {NUM_LAYERS}、d_model = {D_MODEL}、ヘッド数 = {NUM_HEADS}、語彙サイズ = {VOCAB_SIZE}、"
    f"文脈長 = {CONTEXT_LENGTH}、パラメータ数 = {TOTAL_PARAMETERS:,}"
)
```


    tokenizer.json:   0%|          | 0.00/661k [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/305 [00:00<?, ?B/s]



    model_state.pt: reconstructing file:   0%|          |  0.00B / 21.0MB            



    model_state.pt: downloading bytes:           |  0.00B            


    層数 = 4、d_model = 256、ヘッド数 = 8、語彙サイズ = 8192、文脈長 = 256、パラメータ数 = 5,246,208


### 5.4 データ: 016 の学習データの再生成と、017 のプロンプトの分割

**016 の学習データ・評価集合の再生成**: SFT を 016 と同じデータで行い、P0 を 016 と同じ評価集合で測るため、016 と同じ関数・
同じ乱数シード(`16016`)・同じ除外集合(016 のパイロットの評価入力)で再生成する。016 の本番のセル出力と照合できる量
(評価集合の先頭の入力、パイロットの評価入力の数 726)をアサーションで確かめる。

**017 のプロンプト**: 016 の学習データ・評価集合・パイロットの評価入力、および 017 のパイロット(6.2 節)の評価用の入力と
重複しない入力を、1 つの乱数生成器(`PROMPT_SEED`)から順に引き、次の 3 つの互いに素な集合に分ける。

| 集合 | 入力の数 | 事例の作り方 | 用途 |
|---|---|---|---|
| 評価用 | $\lvert X \rvert$ | 入力と 4 課題の直積(016 の`build_evaluation_examples()`) | 応答プール(実験 A〜C) |
| 報酬モデルの学習用 | $N_A$ | 入力 1 つにつき 1 事例、4 事例ごとに 4 課題を 1 回ずつ(016 の`build_training_examples()`) | 選好の組 |
| PPO 用 | 反復回数 × 64 | 同上 | PPO のロールアウト |

プロンプトはすべて 016 の短い水準のテンプレート(学習に使った指示文 16 個、`SEEN_INSTRUCTIONS`)を使う。報酬モデルの学習用の
先頭 $N$ 個(実験 C の水準)でも課題の数が均等になる。


```python
_t0_data = time.time()
# --- 016 のパイロットの評価入力(016 の 5.4 節と同じ関数・同じ乱数シード) ---
PILOT_EVALUATION_INPUT_SPECS_016 = (
    (2016, 250, len(WORD_VOCABULARY), 3, 8),
    (1, 100, len(WORD_VOCABULARY), 3, 8),
    (1, 100, len(WORD_VOCABULARY), 4, 8),
    (1, 100, 32, 3, 8),
    (1, 100, 32, 4, 8),
    (1, 100, 16, 3, 8),
)
PILOT_EVALUATION_INPUTS_016 = set()
for _seed, _count, _vocab, _lo, _hi in PILOT_EVALUATION_INPUT_SPECS_016:
    PILOT_EVALUATION_INPUTS_016 |= set(
        sample_word_sequences(
            np.random.default_rng(_seed), _count, WORD_VOCABULARY[:_vocab], _lo, _hi
        )
    )

# --- 016 の学習データ・評価集合(016 の build_dataset と同じ手順) ---
_rng_016 = np.random.default_rng(DATA_SEED_016)
_eval_inputs_016 = sample_word_sequences(
    _rng_016,
    NUM_EVAL_INPUTS_016,
    WORD_VOCABULARY,
    MIN_WORDS,
    MAX_WORDS,
    exclude=PILOT_EVALUATION_INPUTS_016,
)
_train_inputs_016 = sample_word_sequences(
    _rng_016,
    LEVELS["prod"]["SFT_STEPS"] * SFT_BATCH_SIZE,  # 016 の N_max = 65536(乱数の消費を 016 と揃える)
    WORD_VOCABULARY,
    MIN_WORDS,
    MAX_WORDS,
    exclude=set(_eval_inputs_016) | PILOT_EVALUATION_INPUTS_016,
)
SFT_TRAIN_EXAMPLES = build_training_examples(_train_inputs_016, _rng_016)
EVAL_EXAMPLES_016 = build_evaluation_examples(_eval_inputs_016, _rng_016)
assert len(PILOT_EVALUATION_INPUTS_016) == 726  # 016 の 5.4 節の印字と一致
assert _eval_inputs_016[0] == (
    "agreement",
    "green",
    "billion",
    "software",
    "genus",
    "member",
    "novel",
    "community",
)  # 016 の 5.4 節の印字(評価集合の先頭の入力)と一致


def encode_examples(examples) -> list:
    return [
        encode_instruction_example(
            tokenizer.encode, tokenizer.decode, e.prompt(False), e.response()
        )
        for e in examples
    ]


SFT_TRAIN_ENCODED = encode_examples(SFT_TRAIN_EXAMPLES[:SFT_NUM_EXAMPLES])
P0_EXAMPLES = EVAL_EXAMPLES_016[: CFG["NUM_P0_INPUTS"] * TASK_COUNT]
P0_ENCODED = encode_examples(P0_EXAMPLES)

# --- 017 のパイロットの評価用の入力(6.2 節。本番の入力と重複させない) ---
_excluded_016 = set(_train_inputs_016) | set(_eval_inputs_016) | PILOT_EVALUATION_INPUTS_016
PILOT_EVALUATION_INPUTS_017 = set(
    sample_word_sequences(
        np.random.default_rng(PILOT_EVALUATION_SEED_017),
        PILOT_EVALUATION_COUNT_017,
        WORD_VOCABULARY,
        MIN_WORDS,
        MAX_WORDS,
        exclude=_excluded_016,
    )
)

# --- 017 のプロンプト: 評価用・報酬モデルの学習用・PPO 用(互いに素) ---
PPO_MAX_PROMPTS = CFG["PPO_ITERATIONS"] * PPO_BATCH_SIZE
_rng = np.random.default_rng(PROMPT_SEED)
_excluded = _excluded_016 | PILOT_EVALUATION_INPUTS_017
EVAL_INPUTS = sample_word_sequences(
    _rng, NUM_EVAL_INPUTS, WORD_VOCABULARY, MIN_WORDS, MAX_WORDS, exclude=_excluded
)
RM_TRAIN_INPUTS = sample_word_sequences(
    _rng, RM_PAIRS_MAX, WORD_VOCABULARY, MIN_WORDS, MAX_WORDS, exclude=_excluded | set(EVAL_INPUTS)
)
PPO_INPUTS = sample_word_sequences(
    _rng,
    PPO_MAX_PROMPTS,
    WORD_VOCABULARY,
    MIN_WORDS,
    MAX_WORDS,
    exclude=_excluded | set(EVAL_INPUTS) | set(RM_TRAIN_INPUTS),
)
EVAL_EXAMPLES = build_evaluation_examples(EVAL_INPUTS, _rng)
RM_TRAIN_EXAMPLES = build_training_examples(RM_TRAIN_INPUTS, _rng)
PPO_EXAMPLES = build_training_examples(PPO_INPUTS, _rng)
EVAL_PROMPT_IDS = [tokenizer.encode(e.prompt(False)) for e in EVAL_EXAMPLES]
RM_TRAIN_PROMPT_IDS = [tokenizer.encode(e.prompt(False)) for e in RM_TRAIN_EXAMPLES]
PPO_PROMPT_IDS = [tokenizer.encode(e.prompt(False)) for e in PPO_EXAMPLES]
EVAL_INPUT_INDEX = np.repeat(np.arange(NUM_EVAL_INPUTS), TASK_COUNT)  # プロンプト -> 入力 x
DATA_SECONDS = time.time() - _t0_data

# --- 不変条件 ---
_sets = {
    "評価用": set(EVAL_INPUTS),
    "報酬モデルの学習用": set(RM_TRAIN_INPUTS),
    "PPO 用": set(PPO_INPUTS),
    "016 の学習データ": set(_train_inputs_016),
    "016 の評価集合": set(_eval_inputs_016),
    "016 のパイロット": PILOT_EVALUATION_INPUTS_016,
    "017 のパイロット": PILOT_EVALUATION_INPUTS_017,
}
_names = list(_sets)
for _i, _a in enumerate(_names):
    for _b in _names[_i + 1 :]:
        if {_a, _b} <= {
            "016 の学習データ",
            "016 の評価集合",
            "016 のパイロット",
            "017 のパイロット",
        }:
            continue  # 016・017 のパイロットどうしの関係は 016 と上のサンプリングで保証済み
        assert not (_sets[_a] & _sets[_b]), f"{_a} と {_b} の入力が重複する"
assert len(set(EVAL_INPUTS)) == NUM_EVAL_INPUTS and len(set(RM_TRAIN_INPUTS)) == RM_PAIRS_MAX
assert [(e.words, e.task) for e in EVAL_EXAMPLES] == [
    (x, t) for x in EVAL_INPUTS for t in TASK_NAMES
]
for _n in rm_levels(4, RM_PAIRS_MAX):  # 報酬モデルの学習用の先頭 N 個で課題が均等
    _counts = {t: sum(e.task == t for e in RM_TRAIN_EXAMPLES[:_n]) for t in TASK_NAMES}
    assert len(set(_counts.values())) == 1, (_n, _counts)
assert all(not e.unseen for e in EVAL_EXAMPLES + RM_TRAIN_EXAMPLES + PPO_EXAMPLES)
_longest_prompt = max(map(len, EVAL_PROMPT_IDS + RM_TRAIN_PROMPT_IDS + PPO_PROMPT_IDS))
assert _longest_prompt + MAX_NEW_TOKENS <= CONTEXT_LENGTH

print(
    f"016 のデータを再生成: 学習データ {len(SFT_TRAIN_EXAMPLES):,} 事例(SFT に使うのは先頭 {SFT_NUM_EXAMPLES:,})、"
    f"評価集合 {len(EVAL_EXAMPLES_016)} 事例(P0 に使うのは先頭 {len(P0_EXAMPLES)})、"
    f"016 のパイロットの評価入力 {len(PILOT_EVALUATION_INPUTS_016)} 個、先頭の評価入力が 016 の印字と一致: OK"
)
print(
    f"017 のプロンプト: 評価用 {len(EVAL_EXAMPLES)}(|X| = {NUM_EVAL_INPUTS} x K = {TASK_COUNT})、"
    f"報酬モデルの学習用 {len(RM_TRAIN_EXAMPLES):,}、PPO 用 {len(PPO_EXAMPLES):,}。"
    f"互いに素で、016 のデータ・016 と 017 のパイロットの入力とも重複しない。最長のプロンプト {_longest_prompt} トークン"
    f"({DATA_SECONDS:.1f}s)"
)
print("\n--- 評価用のプロンプトの例 ---")
print(EVAL_EXAMPLES[0].prompt(False) + EVAL_EXAMPLES[0].response())
```

    016 のデータを再生成: 学習データ 65,536 事例(SFT に使うのは先頭 65,536)、評価集合 1000 事例(P0 に使うのは先頭 1000)、016 のパイロットの評価入力 726 個、先頭の評価入力が 016 の印字と一致: OK
    017 のプロンプト: 評価用 256(|X| = 64 x K = 4)、報酬モデルの学習用 65,536、PPO 用 9,600。互いに素で、016 のデータ・016 と 017 のパイロットの入力とも重複しない。最長のプロンプト 45 トークン(18.3s)
    
    --- 評価用のプロンプトの例 ---
    ### Instruction:
    List the words from the last one to the first one.
    Words: center graduate brother account inflation mission
    ### Response: mission inflation account brother graduate center
    ### End


### 5.5 真の報酬・生成・損失・推定量のハーネスの確認

- **真の報酬**: 編集距離を素朴な動的計画法と乱数の文字列で照合する。$r^*$ を手で作った文字列で確かめ、$r^* = 1$ が
  016 の完全一致と同値であることを、評価集合の正解の文字列で確かめる。
- **貪欲法の生成が 016 から変わっていないこと**: `src/generation/stopping.py`を 017 で変更する前のコミット
  (`09f142d`)の関数と、変更後の`greedy_generate_until_stop()`の出力を、同じモデル・同じプロンプトで照合する。
- **サンプリング**: 同じ乱数の状態から 2 回生成して一致すること。温度を小さくすると貪欲法の出力に一致すること。
- **隠れ状態の取り出し**: `model.lm_head(compute_final_hidden_states(model, x))`が`model(x)`と bit 単位で一致すること。
  報酬モデルのスコアが、1 系列ずつ計算した値と(パディングの有無によらず)一致すること。
- **Bradley-Terry の損失**: `-F.logsigmoid`との一致、報酬への定数の加算で変わらないこと。
- **best-of-n の推定量**: 小さな $M$ での全列挙(すべての部分集合の最大値の平均)と一致すること、重みの和が 1 であること。
- **GAE・クリップ付き目的関数・KL ペナルティ付きの報酬**: 手計算の小例と一致すること。


```python
_t0_harness = time.time()


# --- 編集距離: 素朴な動的計画法との照合 ---
def _levenshtein_reference(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        diagonal, row[0] = row[0], i
        for j, cb in enumerate(b, start=1):
            diagonal, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, diagonal + (ca != cb))
    return row[-1]


_rng_check = np.random.default_rng(0)
for _ in range(2000):
    _a = "".join(_rng_check.choice(list("ab c"), size=_rng_check.integers(0, 14)))
    _b = "".join(_rng_check.choice(list("ab c"), size=_rng_check.integers(0, 14)))
    assert levenshtein_distance(_a, _b) == _levenshtein_reference(_a, _b), (_a, _b)
print("編集距離が素朴な動的計画法と一致(乱数の文字列 2000 組): OK")

# --- 真の報酬 ---
_answer = ("cat", "dog")
for _text, _expected in (
    (" cat dog\n### End", 1.0),  # 完全一致
    ("  cat   dog ### End", 1.0),  # 空白の正規化
    (" cat dog", 0.5),  # 形式の違反(終端記号なし): s = 1, f = 0
    (" cat dgo\n### End", 0.5 * (1 - 2 / 7 + 1)),  # 置換 2 文字
    (" dog cat\n### End", 0.5 * (1 - 6 / 7 + 1)),
    (" cat dog\n### Response: x\n### End", 0.5),  # 最初の ### が終端記号でない
    ("", 0.0),
):
    assert math.isclose(compute_true_reward(_text, _answer), _expected), (_text, _expected)
for _e in EVAL_EXAMPLES:  # r* = 1 と 016 の完全一致は同値(正解・正解の変形で確かめる)
    for _text in (_e.response(), _e.response().replace(" ", "  "), _e.response()[:-1]):
        assert (compute_true_reward(_text, _e.answer) == 1.0) == score_generated_response(
            _text, _e.answer
        )[1]
print("真の報酬: 手で作った文字列の値、r* = 1 と 016 の完全一致の同値性: OK")

# --- 貪欲法の生成が 017 の変更前のコミットの関数と一致 ---
_old_source = subprocess.run(
    ["git", "show", f"{COMMIT_BEFORE_017}:src/generation/stopping.py"],
    capture_output=True,
    text=True,
    check=True,
).stdout
_old_stopping = types.ModuleType("stopping_before_017")
exec(compile(_old_source, "stopping_before_017", "exec"), _old_stopping.__dict__)
_probe_prompts = [list(x.token_ids[: x.prompt_length]) for x in P0_ENCODED[:64]]
_greedy_new = greedy_generate_until_stop(
    base_model, _probe_prompts, tokenizer.decode, END_MARKER, MAX_NEW_TOKENS, device, 64
)
_greedy_old = _old_stopping.greedy_generate_until_stop(
    base_model, _probe_prompts, tokenizer.decode, END_MARKER, MAX_NEW_TOKENS, device, 64
)
assert _greedy_new == _greedy_old
print(
    f"貪欲法の生成が変更前のコミット {COMMIT_BEFORE_017} の関数と一致({len(_probe_prompts)} 事例): OK"
)


# --- サンプリング: 再現性と、低温での貪欲法との一致 ---
def _sample(model, prompts, seed, temperature=TEMPERATURE, batch_size=GENERATION_BATCH_SIZE):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return sample_generate_until_stop(
        model,
        prompts,
        tokenizer.decode,
        END_MARKER,
        MAX_NEW_TOKENS,
        device,
        generator,
        temperature,
        batch_size,
    )


assert _sample(base_model, _probe_prompts, 0) == _sample(base_model, _probe_prompts, 0)
_low_temperature = _sample(base_model, _probe_prompts, 0, temperature=1e-4, batch_size=64)
_low_temperature_agreement = float(
    np.mean([a == b for a, b in zip(_low_temperature, _greedy_new, strict=True)])
)
assert _low_temperature_agreement >= 0.9, (
    _low_temperature_agreement
)  # logits の僅差の同点では一致しなくてよい
print(
    "サンプリング: 同じ乱数の状態で 2 回一致: OK / temperature = 1e-4 での生成が貪欲法の出力と一致する割合 "
    f"{_low_temperature_agreement:.4f}(>= 0.9。logits の差が 1e-4 程度の僅差の候補では抽選が分かれうる): OK"
)

# --- 隠れ状態の取り出しと報酬モデルのスコア ---
_tokens = torch.tensor([list(P0_ENCODED[i].token_ids[:24]) for i in range(4)], device=device)
with torch.no_grad():
    assert torch.equal(
        base_model.lm_head(compute_final_hidden_states(base_model, _tokens)), base_model(_tokens)
    )
torch.manual_seed(0)
_probe_rm = RewardModel(copy.deepcopy(base_model), RM_HEAD_INIT_STD).to(device)
_sequences = [list(x.token_ids) for x in P0_ENCODED[:16]]
_batched_scores = score_sequences(_probe_rm, _sequences, device, batch_size=16)
_single_scores = np.array([score_sequences(_probe_rm, [s], device)[0] for s in _sequences])
np.testing.assert_allclose(_batched_scores, _single_scores, rtol=1e-5, atol=1e-5)
print(
    "lm_head(compute_final_hidden_states(x)) == model(x)(bit 単位): OK / "
    f"報酬モデルのスコア: パディングありのバッチと 1 系列ずつが一致(最大差 {np.abs(_batched_scores - _single_scores).max():.2e}): OK"
)
del _probe_rm

# --- Bradley-Terry の損失 ---
_chosen, _rejected = torch.tensor([3.0, -50.0, 0.0, 40.0]), torch.tensor([0.0, 0.0, 0.0, -40.0])
torch.testing.assert_close(
    bradley_terry_loss(_chosen, _rejected), -F.logsigmoid(_chosen - _rejected).mean()
)
torch.testing.assert_close(
    bradley_terry_loss(_chosen + 7.0, _rejected + 7.0), bradley_terry_loss(_chosen, _rejected)
)
assert torch.isfinite(bradley_terry_loss(torch.tensor([-1e4]), torch.tensor([1e4])))
print("Bradley-Terry の損失: -logsigmoid と一致、定数の加算で不変、極端な差でも有限: OK")

# --- best-of-n の推定量: 全列挙との照合 ---
import itertools  # noqa: E402

for _m in range(1, 9):
    _proxy = _rng_check.normal(size=_m)
    _proxy[_rng_check.integers(_m)] = _proxy[0]  # 同点を含める
    _value = _rng_check.normal(size=_m)
    _rank = np.empty(_m, dtype=int)
    _rank[np.argsort(_proxy, kind="stable")] = np.arange(_m)
    for _n in range(1, _m + 1):
        _enumerated = np.mean(
            [_value[max(s, key=lambda i: _rank[i])] for s in itertools.combinations(range(_m), _n)]
        )
        assert math.isclose(
            best_of_n_expected_value(_proxy, _value, _n), _enumerated, rel_tol=1e-12, abs_tol=1e-12
        )
        assert math.isclose(best_of_n_order_weights(_m, _n).sum(), 1.0, rel_tol=1e-12)
_proxy, _value = _rng_check.normal(size=(5, 16)), _rng_check.normal(size=(5, 16))
np.testing.assert_allclose(
    best_of_n_expected_values(_proxy, _value, (1, 2, 4)),
    [[best_of_n_expected_value(_proxy[i], _value[i], n) for n in (1, 2, 4)] for i in range(5)],
    rtol=1e-12,
)
assert best_of_n_kl_upper_bound(1) == 0.0
print("best-of-n の推定量: M = 1〜8 の全 n で全列挙と一致(同点を含む)、重みの和が 1: OK")

# --- GAE・クリップ付き目的関数・KL ペナルティ付きの報酬: 手計算の小例 ---
_mask = torch.tensor([[True, True, True], [True, True, False]])
_advantages, _returns = compute_gae(
    torch.tensor([[0.0, 0.0, 1.0], [0.5, 2.0, 0.0]]),
    torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.6, 0.0]]),
    _mask,
    gamma=1.0,
    lam=0.5,
)
# 行 0: delta = (0.1, 0.1, 0.7) -> A_2 = 0.7, A_1 = 0.1 + 0.5 x 0.7 = 0.45, A_0 = 0.1 + 0.5 x 0.45 = 0.325
# 行 1: delta = (0.7, 1.4)      -> A_1 = 1.4, A_0 = 0.7 + 0.5 x 1.4 = 1.4(パディングの列は 0)
torch.testing.assert_close(_advantages, torch.tensor([[0.325, 0.45, 0.7], [1.4, 1.4, 0.0]]))
torch.testing.assert_close(_returns, torch.tensor([[0.425, 0.65, 1.0], [1.8, 2.0, 0.0]]))
_loss, _clip_fraction = ppo_clipped_policy_loss(
    torch.tensor([[0.0, -1.0]]),
    torch.tensor([[math.log(0.5), -1.0]]),
    torch.tensor([[1.0, -2.0]]),
    torch.ones(1, 2, dtype=torch.bool),
    clip_epsilon=0.2,
)
# トークン 0: 比 2、A = 1 -> min(2, 1.2) = 1.2 / トークン 1: 比 1、A = -2 -> -2 / 平均 -0.4 -> 損失 0.4
torch.testing.assert_close(_loss, torch.tensor(0.4))
assert float(_clip_fraction) == 0.5
_rewards = compute_kl_penalized_rewards(
    torch.tensor([[-1.0, -2.0, 0.0]]),
    torch.tensor([[-1.5, -1.0, 0.0]]),
    torch.tensor([3.0]),
    torch.tensor([[True, True, False]]),
    beta=0.1,
)
torch.testing.assert_close(_rewards, torch.tensor([[-0.05, 0.1 + 3.0, 0.0]]))
# 応答に揃えた対数確率・価値の取り出し(one-hot の積による実装)を、添字で直接取り出した値と照合する
_logits, _values = (
    torch.randn(2, 7, 11, generator=torch.Generator().manual_seed(0)),
    torch.arange(14.0).view(2, 7),
)
_token_ids = torch.randint(0, 11, (2, 7), generator=torch.Generator().manual_seed(1))
_prompt_lengths, _mask = (
    torch.tensor([2, 3]),
    torch.tensor([[True, True, True, False], [True, True, True, True]]),
)
_expected_log_probs, _expected_values = torch.zeros(2, 4), torch.zeros(2, 4)
for _b in range(2):
    for _t in range(int(_mask[_b].sum())):
        _pos = int(_prompt_lengths[_b]) - 1 + _t
        _expected_log_probs[_b, _t] = torch.log_softmax(_logits[_b, _pos], dim=-1)[
            _token_ids[_b, _pos + 1]
        ]
        _expected_values[_b, _t] = _values[_b, _pos]
torch.testing.assert_close(
    gather_response_log_probs(_logits, _token_ids, _prompt_lengths, _mask), _expected_log_probs
)
torch.testing.assert_close(
    gather_response_values(_values, _prompt_lengths, _mask), _expected_values
)
print(
    "GAE・クリップ付き目的関数・KL ペナルティ付きの報酬が手計算の小例と一致、応答に揃えた取り出しが添字での取り出しと一致: OK"
)
HARNESS_SECONDS = time.time() - _t0_harness
print(f"\nこのセルの実行時間: {HARNESS_SECONDS:.1f}s")
```

    編集距離が素朴な動的計画法と一致(乱数の文字列 2000 組): OK
    真の報酬: 手で作った文字列の値、r* = 1 と 016 の完全一致の同値性: OK
    貪欲法の生成が変更前のコミット 09f142d の関数と一致(64 事例): OK
    サンプリング: 同じ乱数の状態で 2 回一致: OK / temperature = 1e-4 での生成が貪欲法の出力と一致する割合 0.9844(>= 0.9。logits の差が 1e-4 程度の僅差の候補では抽選が分かれうる): OK
    lm_head(compute_final_hidden_states(x)) == model(x)(bit 単位): OK / 報酬モデルのスコア: パディングありのバッチと 1 系列ずつが一致(最大差 9.91e-07): OK
    Bradley-Terry の損失: -logsigmoid と一致、定数の加算で不変、極端な差でも有限: OK
    best-of-n の推定量: M = 1〜8 の全 n で全列挙と一致(同点を含む)、重みの和が 1: OK
    GAE・クリップ付き目的関数・KL ペナルティ付きの報酬が手計算の小例と一致、応答に揃えた取り出しが添字での取り出しと一致: OK
    
    このセルの実行時間: 16.6s




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/04_alignment/017_reward_model_and_rlhf.ipynb
