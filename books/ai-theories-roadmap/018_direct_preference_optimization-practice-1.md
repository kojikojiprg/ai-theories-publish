---
title: "DPO(Direct Preference Optimization) / Direct Preference Optimization(実装・実験編 1/4)"
---

この記事は後編(実装・実験編 1/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/018_direct_preference_optimization-theory)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/018_direct_preference_optimization-practice-2)。

## 4. 実装方針 / Implementation Policy

**`src/`に切り出す(スクラッチ実装、本トピックで新規作成または拡張)**:

- `src/training/direct_preference.py`(新規):
  - 応答部分の対数確率の和(`sequence_log_prob_sums()`は勾配を保つ版、`compute_response_log_prob_sums()`は評価用の
    勾配なしの版)。バッチ化は 017 の`collate_rollout_sequences()`(`src/training/ppo.py`)をそのまま使う。トークンの
    対数確率の取り出しは、017 の`gather_response_log_probs()`と同じ位置・同じ one-hot の積だが、語彙への射影を応答の位置に
    限って行う(017 の関数は全位置の logits を作るので、語彙 8192 のもとで学習のメモリと時間の大半を占めた。値が一致する
    ことを 5.6 節で照合する)。017 の関数は変更していない。
  - 参照方策の対数確率の事前計算`precompute_reference_log_probs()`(呼び出し側の辞書をキャッシュとして、条件・シードを
    またいで再利用する)。
  - 対数比のマージン`compute_log_ratio_margin()`、DPO 損失`dpo_loss()`($-\log \sigma(z)$ を`torch.logaddexp(0, -z)`で
    計算)、IPO 損失`ipo_loss()`。
  - 学習ループ`train_direct_preference()`(損失の種類を引数で切り替える。ステップごとの損失・マージン・暗黙の報酬による
    正解率・$\log \pi_\theta(y_w) - \log \pi_{\mathrm{ref}}(y_w)$ などを記録し、指定したステップ(実験 B の $T/2$ と $T$)で
    評価関数を呼んで結果を記録する)。
  - CHES スコアの計算`compute_ches_statistics()`・`ches_from_statistics()`(Razin et al. の Definition 2・3)。
- `src/data/preference.py`(017 の既存モジュールへの追加。既存の関数は変更しない):
  - $K$ 回独立に抽選したラベル`sample_preference_label_matrix()`、決定的なラベルを $K$ 回重複させる
    `deterministic_preference_label_matrix()`、事例の列への展開`flatten_preference_labels()`。
  - 正規化編集距離`normalized_edit_distance()`(既存の`levenshtein_distance()`を長い方の文字数で割る)と、層ごとに近い組・
    遠い組を同数選ぶ`select_similarity_stratified_pairs()`。

**ノートブック内に直接書く(018 固有)**: 016 の評価集合の再生成と参照方策の照合、プロンプトの分割、選好の組の生成、
学習の実行と条件の組み合わせ、**方策の評価**(方策からのサンプリングによる真の報酬の期待値と KL ダイバージェンスの推定。
017 の`sample_generate_until_stop()`と`compute_true_reward()`を呼ぶ)、$\beta$ の較正、判定、スケーリングの計測。
方策の評価は 016・017 の合成課題の真の報酬に依存するので、018 固有としてノートブックに置く。

**参照方策の扱い**: 参照方策 $\pi_{\mathrm{ref}}$ は凍結した`GPTLanguageModel`として 1 つだけメモリに持ち、方策はその
複製に LoRA(Low-Rank Adaptation、[012](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/012_low_rank_adaptation-theory))を Query・Value 射影に
$r = 8$ で掛けたものとする。LoRA の $B$ は 0 で初期化されるので、学習前の方策は参照方策と一致する(5.6 節で確かめる)。
学習データの応答の $\log \pi_{\mathrm{ref}}$ は学習中に変わらないので **事前計算してキャッシュ** し、全条件・全シードで
再利用する(参照方策の順伝播は学習データについて 1 回だけになる)。方策からサンプリングした評価用の応答は学習した方策ごとに
異なるので、その $\log \pi_{\mathrm{ref}}$ は保持している参照方策で計算する。アダプタを無効にして参照方策の代わりにする設計は
とらない。LoRA の層に無効化の機能を足す変更が必要になるうえ、本トピックのモデル(約 520 万パラメータ、約 21 MB)では
参照方策を別に持つメモリの負担が無視できるためである。

**アップロード方針**: Hugging Face Hub へのアップロードはしない。学習した方策は条件比較のためのもので、後続のトピックの
入力にならず、読者がノートブックの外で試すためのものでもない。

**生成物の置き場所**:

| 生成物 | 置き場所 |
|---|---|
| 参照方策・トークナイザ | Hugging Face Hub から取得(`kojikojiprg/ai-theories-small-gpt-en`のブランチ`sft-synthetic`、`kojikojiprg/ai-theories-tokenizer-en`)。018 では書き込まない |
| プロンプト・選好の組・ラベル・参照方策の対数確率(キャッシュ) | メモリ上のみ(同一セッション内。入力から決定的に再生成できる。ファイルに保存しない) |
| 学習した方策 | メモリ上のみ(評価の直後に破棄する) |
| 判定の記録 | 判定と前提条件を計算した各セルの出力 |

外部から取得するのは、Hugging Face Hub の参照方策(`sft-synthetic`のブランチの`config.json`・`model_state.pt`)と
トークナイザ(`tokenizer.json`)のみである。018 は英語版 Wikipedia のコーパスを使わない。

## 5. 実装 / Implementation

### 5.1 環境セットアップ(Google Colab)

`SMOKE_TEST`(スモークテストか本番か)はこのセルでのみ切り替える。本トピックは Hub にアップロードしないので、
アップロードのフラグはない。実行環境はこのセルで 1 回だけ印字する。再現性のため、CUDA の cuBLAS が決定的な演算を使うための
環境変数`CUBLAS_WORKSPACE_CONFIG`を torch の読み込み前に設定し、`torch.use_deterministic_algorithms(True)`を有効にする
(5.8 節で学習・生成の経路を一通り実行して確かめる)。


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

    /content/ai-theories
    [2mUsing Python 3.13.15 environment at: /usr[0m
    [2mChecked [1m59 packages[0m [2min 154ms[0m[0m
    読み込み済みのパッケージの版の検査(requirements.txt): 食い違いなし。検査した配布物 11 個(certifi==2026.7.22, cycler==0.12.1, fonttools==4.63.0, kiwisolver==1.5.0, matplotlib==3.11.1, numpy==2.5.2, packaging==26.3, pillow==12.3.0, pyparsing==3.3.2, python-dateutil==2.9.0.post0, six==1.17.0)、未読み込みのため検査しなかった配布物 43 個、__version__ がないため比べなかった配布物 0 個(なし)、表記の違いで誤検出するため比べなかった配布物 0 個(なし)
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
      コミット / git commit                  : ffa9f7bb70081ee3fd77e3140c50b8beebcb0422
      未コミットの変更 / uncommitted changes : なし
      実行日時 (UTC)                         : 2026-09-28T07:55:34+00:00
    SMOKE_TEST=False、決定的な実行: True、CUBLAS_WORKSPACE_CONFIG=:4096:8



```python
import copy
import functools
import hashlib
import itertools
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
    deterministic_preference_label_matrix,
    flatten_preference_labels,
    normalized_edit_distance,
    sample_preference_label_matrix,
    select_similarity_stratified_pairs,
)
from src.data.tokenizer import load_bpe_id_tokenizer_from_hub
from src.generation.stopping import greedy_generate_until_stop, sample_generate_until_stop
from src.layers.feedforward import SwiGLUFeedForwardNetwork
from src.layers.lora import LoRALinear, apply_lora
from src.layers.normalization import RMSNorm
from src.layers.positional_encoding import RotaryPositionEmbedding
from src.models.gpt import GPTLanguageModel
from src.models.reward_model import compute_final_hidden_states
from src.training.direct_preference import (
    ches_from_statistics,
    compute_ches_statistics,
    compute_log_ratio_margin,
    compute_response_log_prob_sums,
    dpo_loss,
    ipo_loss,
    precompute_reference_log_probs,
    sequence_log_prob_sums,
    train_direct_preference,
)
from src.training.instruction_tuning import (
    evaluate_instruction_negative_log_likelihood,
    make_epoch_batches,
)
from src.training.optimizer import AdamW
from src.training.ppo import collate_rollout_sequences, gather_response_log_probs
from src.utils.reporting import dumps_compact_json
from src.utils.statistics import fit_power_law_exponent

MODEL_REPO_ID = "kojikojiprg/ai-theories-small-gpt-en"
REFERENCE_REVISION = "sft-synthetic"  # 017 がアップロードした参照方策(LoRA をマージした SFT モデル)
TOKENIZER_REPO_ID = "kojikojiprg/ai-theories-tokenizer-en"


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

水準の定義をこの 1 箇所(`LEVELS`)に集約する。参照方策は 017 の学習済みのものを使うので縮小しない。

**縮小の規則**:

- 選好の組の候補の数 $N_{\mathrm{cand}}$(本番 8192・スモークテスト 256)と、実験 A〜C の学習用の組の数 $N$(本番 2048・
  スモークテスト 64)を縮小する。**$N_{\mathrm{cand}} = 4N$** の関係を両水準で保つ(実験 D は候補から近い組・遠い組を
  それぞれ約 1/4 ずつ選ぶので、実験 D のデータ量が実験 A〜C とほぼ同じになる)。
- 学習ステップ数 $T$(本番 768・スモークテスト 24)を縮小する。バッチサイズ 32 事例は縮小せず、
  **$32 T = 3 N K$(実験 A〜C のデータをちょうど 3 エポック)** の関係を両水準で保つ。
- 評価用の入力の数 $\lvert X \rvert$(本番 128・スモークテスト 8)、$\beta$ の較正用の入力の数(本番 32・スモークテスト 4)、
  ブートストラップの反復回数を縮小する。
- **縮小しないもの**: 重複の回数 $K = 4$、評価用のプロンプトあたりのサンプル数 $m = 8$、$\beta$ の較正の格子(3.2〜0.1、
  公比 $\sqrt{2}$ の 11 点)、実験 A の水準が格子の連続する 6 点であること(公比 $\sqrt{2}$)、標準の $\beta$ の決め方、崩壊の領域の
  観察の水準($\beta_6$ から格子を 2 点・4 点下る)、学習率・LoRA の設定・warmup の比率、実験 D の層の作り方
  (課題 4 × $\lvert \Delta r^* \rvert$ の 4 区間)と選ぶ割合 1/4、前提条件・判定の閾値。
- **削る段階の表**(6.1 節)は本番とスモークテストで同じ構造にする。シード数は本番の 5 → 3 をスモークテストでは 3 → 2 に
  縮小する(「削る前 > 削った後 $\ge 2$」の順序関係を保つ)。
- **テスト専用の上書き**: 環境変数`AI_THEORIES_FORCE_STAGE`(0〜3)が設定されているときのみ、6.4 節で見積もりによる
  段階の選択の代わりにその段階を使う(スモークテストで全段階の経路を確かめるため)。本番では受け付けず、このセルで停止する。


```python
# --- 全水準で共通の定数(本番実行前に宣言し、SMOKE_TEST で変えない) ---
TASK_COUNT = len(TASK_NAMES)  # 課題の数
MIN_WORDS, MAX_WORDS = 3, 8  # 入力の単語数(016・017 と同じ)
MAX_NEW_TOKENS = 32  # 生成の上限トークン数(016・017 と同じ)
TEMPERATURE = 1.0  # サンプリングの温度(top-k・top-p なし、017 と同じ)
GENERATION_BATCH_SIZE = 512  # サンプリングの 1 回の順伝播の事例数の上限(結果の再現性にのみ影響)
LOG_PROB_BATCH_SIZE = 256  # 評価用の対数確率の計算の 1 回の順伝播の系列数

# 参照方策の照合(017 の 6.5 節・6.14 節のセル出力から転記した独立な定数)
REFERENCE_SHA256_017 = "5814fbc024a27adc1624ee9a718fbbd5a7e28f5d20a0df40c8d4633126a91ab5"
REFERENCE_EXACT_MATCH_017 = 0.8180  # 016 の評価集合 1000 事例の貪欲法の完全一致率
REFERENCE_FORMAT_RATE_017 = 1.0000  # 同じ評価集合の形式の遵守率
REFERENCE_RESPONSE_NLL_017 = 0.6005  # 同じ評価集合の応答部分の負の対数尤度(nats / トークン)
REFERENCE_EXACT_TOLERANCE = 0.005  # 5 事例分(デバイスによる貪欲法の僅差の分かれを許す)
REFERENCE_NLL_TOLERANCE = 0.0005  # 017 の印字の丸め(0.00005)とデバイスの差を許す
DATA_SEED_016 = 16_016  # 016 の合成データの乱数シード(016 の評価集合を再生成する)
NUM_EVAL_INPUTS_016 = 250
SFT_NUM_EXAMPLES_016 = 65_536  # 016 の学習データの事例数(乱数の消費を 016 と揃える)

# 018 のプロンプト・選好の組・乱数シード
PROMPT_SEED = 18_018  # 評価用・較正用・学習用の入力(互いに素)
PAIR_SEED = 18_019  # 選好の組の応答のサンプリング(torch.Generator)
EVAL_PAIR_SEED = 18_020  # 評価用の組(実験 B の診断量)の応答のサンプリング
LABEL_SEED_BASE = 18_300  # シード s のラベルの抽選は np.random.default_rng(18300 + s)
LORA_INIT_SEED_BASE = 18_200  # シード s の LoRA の A の初期化は torch.manual_seed(18200 + s)
BATCH_SEED_BASE = 18_400  # シード s のミニバッチの順序は make_epoch_batches(..., 18400 + s)
EVAL_SAMPLE_SEED_BASE = (
    18_500  # シード s の方策の評価のサンプリング(全条件で共通の乱数の状態から始める)
)
CALIBRATION_SAMPLE_SEED = 18_600  # beta の較正の KL の推定のサンプリング
TIMING_SEED = 18_099  # スケーリングの計測専用
PILOT_INPUT_SPECS_018 = ((18_998, 4096), (18_997, 128))  # 6.2 節のパイロットの入力(本番から除く)

# 学習(6.2 節のパイロットで決めた値を含む)
NUM_DRAWS = 4  # K: 1 つの組を観測する回数(6.2 節。E[p_hat が (0, 1)] >= 0.5 となる最小の 2 のべき)
BATCH_SIZE = 32  # 1 ステップの事例の数(2 x 32 系列)
LEARNING_RATE = 1e-3  # 6.2 節のパイロット 3(学習の損失と発散の有無のみで決めた)
WARMUP_RATIO = 0.1  # 線形 warmup の後は一定(DPO の原論文の実装と同じ形)
LORA_RANK, LORA_ALPHA = 8, 8.0
LORA_TARGET_MODULES = ("w_q", "w_v")
CHECK_BETA = 0.1  # 5.6〜6.6 節の確認・計測だけに使う beta(判定には影響しない)
BETA_GRID = tuple(
    3.2 * 2.0 ** (-k / 2) for k in range(11)
)  # 較正の格子 3.2〜0.1(公比 sqrt(2)、11 点。6.2 節)
NUM_A_LEVELS = 6  # 実験 A の水準の数(格子の連続する 6 点、公比 sqrt(2)。最小の水準が標準の beta)
COLLAPSE_OFFSETS = (
    2,
    4,
)  # 崩壊の領域の観察: beta_6(実験 A の最小の水準)から格子を 2 点・4 点下った beta(判定なし、シード 0 のみ)
STANDARD_KL_MAX = 2.0  # 較正: 標準の beta(実験 B〜D、実験 A の最小の水準)は、KL がこの値以下にとどまる最小の beta(nats、6.1 節)
SAMPLES_PER_PROMPT = 8  # m: 評価用のプロンプトあたりのサンプル数
D_FRACTION = 0.25  # 実験 D: 各層で近い組・遠い組として選ぶ割合
D_REWARD_BINS = 4  # 実験 D: |Delta r*| の区間の数(候補の分位点で区切る)

# 前提条件・判定
ACCURACY_MIN = 0.55  # 学習の成立: 学習用の事例での暗黙の報酬による正解率
KL_RATIO_MIN = 4.0  # 実験 A: KL(最小の beta) / KL(最大の beta) の下限
MIXED_FRACTION_MIN = 0.4  # 実験 B: 確率的なラベルで p_hat が (0, 1) となる組の割合の下限
EXPECTED_MIXED_TARGET = 0.5  # K の決め方: E[p_hat が (0, 1)] の目標
KAPPA_TARGET = 0.75  # median sigma(kappa |Delta r*|) の目標(017 と同じ)
BOOTSTRAP_SEED = 0
SESSION_BUDGET_SECONDS = 120 * 60  # 1 セッションの予算(T4 で 120 分)

LEVELS = {
    "smoke": {
        "NUM_CANDIDATE_PAIRS": 256,
        "NUM_MAIN_PAIRS": 64,
        "TRAIN_STEPS": 24,
        "NUM_EVAL_INPUTS": 8,
        "NUM_CALIBRATION_INPUTS": 4,
        "BOOTSTRAP_RESAMPLES": 1_000,
    },
    "prod": {
        "NUM_CANDIDATE_PAIRS": 8192,
        "NUM_MAIN_PAIRS": 2048,
        "TRAIN_STEPS": 768,
        "NUM_EVAL_INPUTS": 128,
        "NUM_CALIBRATION_INPUTS": 32,
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
assert FORCED_STAGE is None or FORCED_STAGE in (0, 1, 2, 3), FORCED_STAGE

CFG = LEVELS[CURRENT_LEVEL_NAME]
PROD_CFG = LEVELS["prod"]
NUM_CANDIDATE_PAIRS = CFG["NUM_CANDIDATE_PAIRS"]
NUM_MAIN_PAIRS = CFG["NUM_MAIN_PAIRS"]  # N
TRAIN_STEPS = CFG["TRAIN_STEPS"]  # T
HALF_STEP = TRAIN_STEPS // 2  # 実験 B の T/2
NUM_EVAL_INPUTS = CFG["NUM_EVAL_INPUTS"]  # |X|
NUM_EVAL_PROMPTS = NUM_EVAL_INPUTS * TASK_COUNT
NUM_CALIBRATION_INPUTS = CFG["NUM_CALIBRATION_INPUTS"]
BOOTSTRAP_RESAMPLES = CFG["BOOTSTRAP_RESAMPLES"]

# --- 削る段階(6.1 節)。順序: 実験 D のシード数 -> 実験 A のシード数 -> 実験 B(と C)のシード数 ---
STAGES = {
    "prod": {
        0: {"NUM_SEEDS_D": 5, "NUM_SEEDS_A": 5, "NUM_SEEDS_B": 5},
        1: {"NUM_SEEDS_D": 3, "NUM_SEEDS_A": 5, "NUM_SEEDS_B": 5},
        2: {"NUM_SEEDS_D": 3, "NUM_SEEDS_A": 3, "NUM_SEEDS_B": 5},
        3: {"NUM_SEEDS_D": 3, "NUM_SEEDS_A": 3, "NUM_SEEDS_B": 3},
    },
    "smoke": {  # 本番と同じ構造(シード数 5 -> 3 を 3 -> 2 に縮小)
        0: {"NUM_SEEDS_D": 3, "NUM_SEEDS_A": 3, "NUM_SEEDS_B": 3},
        1: {"NUM_SEEDS_D": 2, "NUM_SEEDS_A": 3, "NUM_SEEDS_B": 3},
        2: {"NUM_SEEDS_D": 2, "NUM_SEEDS_A": 2, "NUM_SEEDS_B": 3},
        3: {"NUM_SEEDS_D": 2, "NUM_SEEDS_A": 2, "NUM_SEEDS_B": 2},
    },
}

# --- 縮小規則の確認 ---
assert set(LEVELS["smoke"]) == set(LEVELS["prod"])
for _name, _level in LEVELS.items():
    assert _level["NUM_CANDIDATE_PAIRS"] == 4 * _level["NUM_MAIN_PAIRS"]
    assert (
        BATCH_SIZE * _level["TRAIN_STEPS"] == 3 * _level["NUM_MAIN_PAIRS"] * NUM_DRAWS
    )  # 3 エポック
    assert _level["TRAIN_STEPS"] % 2 == 0  # T/2 が整数
    _stages = STAGES[_name]
    assert set(_stages) == {0, 1, 2, 3}
    for _k in range(1, 4):  # 段階が上がるほど、どの量も減るか変わらない
        for _key in ("NUM_SEEDS_D", "NUM_SEEDS_A", "NUM_SEEDS_B"):
            assert _stages[_k][_key] <= _stages[_k - 1][_key]
    assert all(min(s.values()) >= 2 for s in _stages.values())
for _k in range(1, 4):  # 本番とスモークテストで、各段階で何が変わるかが同じ(構造を保つ縮小)
    for _key in ("NUM_SEEDS_D", "NUM_SEEDS_A", "NUM_SEEDS_B"):
        assert (STAGES["prod"][_k][_key] == STAGES["prod"][_k - 1][_key]) == (
            STAGES["smoke"][_k][_key] == STAGES["smoke"][_k - 1][_key]
        )
for _key in LEVELS["prod"]:
    assert LEVELS["smoke"][_key] < LEVELS["prod"][_key], _key
assert all(
    math.isclose(a / b, math.sqrt(2.0)) for a, b in zip(BETA_GRID, BETA_GRID[1:], strict=False)
)  # 公比 sqrt(2)
assert math.isclose(BETA_GRID[0], 3.2) and math.isclose(BETA_GRID[-1], 0.1)
assert len(BETA_GRID) >= NUM_A_LEVELS

print(f"水準 {CURRENT_LEVEL_NAME!r}: {json.dumps(CFG)}")
print(
    f"学習: T = {TRAIN_STEPS}(T/2 = {HALF_STEP})、バッチ {BATCH_SIZE} 事例、学習率 {LEARNING_RATE}"
    f"(warmup {WARMUP_RATIO:.0%} の後は一定)、LoRA r = {LORA_RANK}・alpha = {LORA_ALPHA}・対象 {LORA_TARGET_MODULES}"
)
print(
    f"データ: 候補の組 N_cand = {NUM_CANDIDATE_PAIRS:,}、実験 A〜C の組 N = {NUM_MAIN_PAIRS:,}、K = {NUM_DRAWS}"
    f"(事例 {NUM_MAIN_PAIRS * NUM_DRAWS:,} = {BATCH_SIZE * TRAIN_STEPS // (NUM_MAIN_PAIRS * NUM_DRAWS)} エポック)"
)
print(
    f"評価: |X| = {NUM_EVAL_INPUTS}(プロンプト {NUM_EVAL_PROMPTS})x m = {SAMPLES_PER_PROMPT}、較正用の入力 "
    f"{NUM_CALIBRATION_INPUTS}、ブートストラップ {BOOTSTRAP_RESAMPLES:,} 回"
)
print(
    f"beta: 較正の格子 {[round(b, 6) for b in BETA_GRID]}、実験 A の水準の数 {NUM_A_LEVELS}(格子の連続する点)、"
    f"標準の beta の KL の閾値 {STANDARD_KL_MAX} nats、崩壊の領域の観察は beta_6 から格子を {COLLAPSE_OFFSETS} 点下る"
)
print(f"削る段階({CURRENT_LEVEL_NAME!r}、6.4 節で選ぶ): {json.dumps(STAGES[CURRENT_LEVEL_NAME])}")
```

    水準 'prod': {"NUM_CANDIDATE_PAIRS": 8192, "NUM_MAIN_PAIRS": 2048, "TRAIN_STEPS": 768, "NUM_EVAL_INPUTS": 128, "NUM_CALIBRATION_INPUTS": 32, "BOOTSTRAP_RESAMPLES": 10000}
    学習: T = 768(T/2 = 384)、バッチ 32 事例、学習率 0.001(warmup 10% の後は一定)、LoRA r = 8・alpha = 8.0・対象 ('w_q', 'w_v')
    データ: 候補の組 N_cand = 8,192、実験 A〜C の組 N = 2,048、K = 4(事例 8,192 = 3 エポック)
    評価: |X| = 128(プロンプト 512)x m = 8、較正用の入力 32、ブートストラップ 10,000 回
    beta: 較正の格子 [3.2, 2.262742, 1.6, 1.131371, 0.8, 0.565685, 0.4, 0.282843, 0.2, 0.141421, 0.1]、実験 A の水準の数 6(格子の連続する点)、標準の beta の KL の閾値 2.0 nats、崩壊の領域の観察は beta_6 から格子を (2, 4) 点下る
    削る段階('prod'、6.4 節で選ぶ): {"0": {"NUM_SEEDS_D": 5, "NUM_SEEDS_A": 5, "NUM_SEEDS_B": 5}, "1": {"NUM_SEEDS_D": 3, "NUM_SEEDS_A": 5, "NUM_SEEDS_B": 5}, "2": {"NUM_SEEDS_D": 3, "NUM_SEEDS_A": 3, "NUM_SEEDS_B": 5}, "3": {"NUM_SEEDS_D": 3, "NUM_SEEDS_A": 3, "NUM_SEEDS_B": 3}}


### 5.3 参照方策・トークナイザの取得(017 がアップロードしたもの)

参照方策 $\pi_{\mathrm{ref}}$ は`kojikojiprg/ai-theories-small-gpt-en`のブランチ`sft-synthetic`(017 で SFT し、LoRA をマージした
重み)、トークナイザは`kojikojiprg/ai-theories-tokenizer-en`から取得する。取得した`model_state.pt`の SHA-256 が、017 の
6.14 節のセル出力に記録された値(ノートブックに独立な定数として転記したもの)と一致することを確かめる。


```python
tokenizer, _tokenizer_from_hub = load_bpe_id_tokenizer_from_hub(TOKENIZER_REPO_ID)
assert _tokenizer_from_hub, "トークナイザを Hugging Face Hub から取得できなかった"
_config_path = hf_hub_download(MODEL_REPO_ID, "config.json", revision=REFERENCE_REVISION)
_state_path = hf_hub_download(MODEL_REPO_ID, "model_state.pt", revision=REFERENCE_REVISION)
REFERENCE_SHA256 = hashlib.sha256(Path(_state_path).read_bytes()).hexdigest()
assert REFERENCE_SHA256 == REFERENCE_SHA256_017, (REFERENCE_SHA256, REFERENCE_SHA256_017)
MODEL_CONFIG = json.loads(Path(_config_path).read_text(encoding="utf-8"))
assert MODEL_CONFIG["positional_encoding"] == "rope"
assert MODEL_CONFIG["normalization"] == "rmsnorm" and MODEL_CONFIG["feed_forward"] == "swiglu"
assert MODEL_CONFIG["norm_first"] and MODEL_CONFIG["dropout"] == 0.0
assert tokenizer.vocab_size == MODEL_CONFIG["vocabulary_size"]
CONTEXT_LENGTH = MODEL_CONFIG["sequence_length"]
D_MODEL, NUM_LAYERS, NUM_HEADS = (
    MODEL_CONFIG["d_model"],
    MODEL_CONFIG["num_layers"],
    MODEL_CONFIG["num_heads"],
)


def build_model(state_dict: dict) -> GPTLanguageModel:
    model = GPTLanguageModel(
        vocabulary_size=MODEL_CONFIG["vocabulary_size"],
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
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device).eval()


REFERENCE_STATE = torch.load(_state_path, map_location="cpu")
reference_policy = build_model(REFERENCE_STATE)
TOTAL_PARAMETERS = sum(p.numel() for p in reference_policy.parameters())
print(
    f"参照方策: {MODEL_REPO_ID}@{REFERENCE_REVISION}、SHA-256 {REFERENCE_SHA256}(017 の記録と一致: OK)"
)
print(
    f"層数 = {NUM_LAYERS}、d_model = {D_MODEL}、ヘッド数 = {NUM_HEADS}、語彙サイズ = {tokenizer.vocab_size}、"
    f"文脈長 = {CONTEXT_LENGTH}、パラメータ数 = {TOTAL_PARAMETERS:,}"
)
```


    tokenizer.json:   0%|          | 0.00/661k [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/305 [00:00<?, ?B/s]



    model_state.pt: reconstructing file:   0%|          |  0.00B / 21.0MB            



    model_state.pt: downloading bytes:           |  0.00B            


    参照方策: kojikojiprg/ai-theories-small-gpt-en@sft-synthetic、SHA-256 5814fbc024a27adc1624ee9a718fbbd5a7e28f5d20a0df40c8d4633126a91ab5(017 の記録と一致: OK)
    層数 = 4、d_model = 256、ヘッド数 = 8、語彙サイズ = 8192、文脈長 = 256、パラメータ数 = 5,246,208


### 5.4 データ: 016 の評価集合の再生成と、018 のプロンプトの分割

**016 の評価集合の再生成**: 参照方策の照合(5.5 節)に使うため、016 と同じ関数・同じ乱数シード(`16016`)・同じ除外集合
(016 のパイロットの評価入力)で、016 の学習データと評価集合を再生成する(017 の 5.4 節と同じ手順)。016 の本番のセル出力と
照合できる量(評価集合の先頭の入力、パイロットの評価入力の数 726)をアサーションで確かめる。

**018 のプロンプト**: 016 の学習データ・評価集合・パイロットの評価入力、および 018 のパイロット(6.2 節)の入力と重複しない
入力を、1 つの乱数生成器(`PROMPT_SEED`)から順に引き、次の 3 つの互いに素な集合に分ける。参照方策の SFT の学習データを
除くのは、参照方策が学習で見た入力で方策を学習・評価しないためである。

| 集合 | 入力の数 | 事例の作り方 | 用途 |
|---|---|---|---|
| 評価用 | $\lvert X \rvert$ | 入力と 4 課題の直積(016 の`build_evaluation_examples()`) | 方策の評価(真の報酬・KL)、評価用の組 |
| 較正用 | 本番 32 | 同上 | $\beta$ の較正(KL のみを測る) |
| 学習用 | $N_{\mathrm{cand}}$ | 入力 1 つにつき 1 事例、4 事例ごとに 4 課題を 1 回ずつ(016 の`build_training_examples()`) | 選好の組の候補 |

プロンプトはすべて 016 の短い水準のテンプレート(学習に使った指示文、`SEEN_INSTRUCTIONS`)を使う。


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
    SFT_NUM_EXAMPLES_016,
    WORD_VOCABULARY,
    MIN_WORDS,
    MAX_WORDS,
    exclude=set(_eval_inputs_016) | PILOT_EVALUATION_INPUTS_016,
)
build_training_examples(_train_inputs_016, _rng_016)  # 乱数の消費を 016 と揃える
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
EVAL_ENCODED_016 = [
    encode_instruction_example(tokenizer.encode, tokenizer.decode, e.prompt(False), e.response())
    for e in EVAL_EXAMPLES_016
]

# --- 018 のパイロットの入力(6.2 節。本番の入力と重複させない) ---
PILOT_INPUTS_018 = set()
for _seed, _count in PILOT_INPUT_SPECS_018:
    PILOT_INPUTS_018 |= set(
        sample_word_sequences(
            np.random.default_rng(_seed), _count, WORD_VOCABULARY, MIN_WORDS, MAX_WORDS
        )
    )

# --- 018 のプロンプト: 評価用・較正用・学習用(互いに素) ---
_excluded = (
    set(_train_inputs_016) | set(_eval_inputs_016) | PILOT_EVALUATION_INPUTS_016 | PILOT_INPUTS_018
)
_rng = np.random.default_rng(PROMPT_SEED)
EVAL_INPUTS = sample_word_sequences(
    _rng, NUM_EVAL_INPUTS, WORD_VOCABULARY, MIN_WORDS, MAX_WORDS, exclude=_excluded
)
CALIBRATION_INPUTS = sample_word_sequences(
    _rng,
    NUM_CALIBRATION_INPUTS,
    WORD_VOCABULARY,
    MIN_WORDS,
    MAX_WORDS,
    exclude=_excluded | set(EVAL_INPUTS),
)
TRAIN_INPUTS = sample_word_sequences(
    _rng,
    NUM_CANDIDATE_PAIRS,
    WORD_VOCABULARY,
    MIN_WORDS,
    MAX_WORDS,
    exclude=_excluded | set(EVAL_INPUTS) | set(CALIBRATION_INPUTS),
)
EVAL_EXAMPLES = build_evaluation_examples(EVAL_INPUTS, _rng)
CALIBRATION_EXAMPLES = build_evaluation_examples(CALIBRATION_INPUTS, _rng)
TRAIN_EXAMPLES = build_training_examples(TRAIN_INPUTS, _rng)
EVAL_PROMPT_IDS = [tokenizer.encode(e.prompt(False)) for e in EVAL_EXAMPLES]
CALIBRATION_PROMPT_IDS = [tokenizer.encode(e.prompt(False)) for e in CALIBRATION_EXAMPLES]
TRAIN_PROMPT_IDS = [tokenizer.encode(e.prompt(False)) for e in TRAIN_EXAMPLES]
DATA_SECONDS = time.time() - _t0_data

# --- 不変条件: 学習用と評価用(と較正用)のプロンプトの入力が重ならない ---
_sets = {"評価用": set(EVAL_INPUTS), "較正用": set(CALIBRATION_INPUTS), "学習用": set(TRAIN_INPUTS)}
for _a, _b in itertools.combinations(_sets, 2):
    assert not (_sets[_a] & _sets[_b]), f"{_a} と {_b} の入力が重複する"
for _name, _s in _sets.items():
    assert not (_s & _excluded), f"{_name} の入力が 016 のデータ・パイロットの入力と重複する"
assert len(set(TRAIN_INPUTS)) == NUM_CANDIDATE_PAIRS and len(set(EVAL_INPUTS)) == NUM_EVAL_INPUTS
assert [(e.words, e.task) for e in EVAL_EXAMPLES] == [
    (x, t) for x in EVAL_INPUTS for t in TASK_NAMES
]
assert all(not e.unseen for e in EVAL_EXAMPLES + CALIBRATION_EXAMPLES + TRAIN_EXAMPLES)
_longest_prompt = max(map(len, EVAL_PROMPT_IDS + CALIBRATION_PROMPT_IDS + TRAIN_PROMPT_IDS))
assert _longest_prompt + MAX_NEW_TOKENS <= CONTEXT_LENGTH
EVAL_INPUT_INDEX = np.repeat(np.arange(NUM_EVAL_INPUTS), TASK_COUNT)  # プロンプト -> 入力 x

print(
    f"016 の評価集合を再生成: {len(EVAL_EXAMPLES_016)} 事例、016 のパイロットの評価入力 {len(PILOT_EVALUATION_INPUTS_016)} 個、"
    "先頭の評価入力が 016 の印字と一致: OK"
)
print(
    f"018 のプロンプト: 評価用 {len(EVAL_EXAMPLES)}(|X| = {NUM_EVAL_INPUTS} x {TASK_COUNT} 課題)、較正用 "
    f"{len(CALIBRATION_EXAMPLES)}、学習用 {len(TRAIN_EXAMPLES):,}。互いに素で、016 のデータ・016 と 018 のパイロットの入力とも"
    f"重複しない: OK。最長のプロンプト {_longest_prompt} トークン({DATA_SECONDS:.1f}s)"
)
```

    016 の評価集合を再生成: 1000 事例、016 のパイロットの評価入力 726 個、先頭の評価入力が 016 の印字と一致: OK
    018 のプロンプト: 評価用 512(|X| = 128 x 4 課題)、較正用 128、学習用 8,192。互いに素で、016 のデータ・016 と 018 のパイロットの入力とも重複しない: OK。最長のプロンプト 45 トークン(6.9s)


### 5.5 参照方策の照合

読み込んだ参照方策で、017 が記録した評価値を再計算して照合する。期待値は 017 のセル出力から転記した定数(5.2 節の
`REFERENCE_*_017`)で、実測値は読み込んだ重みから計算するので、両者は同じ変数に由来しない。

- 016 の評価集合(250 入力 × 4 課題 = 1000 事例)での貪欲法の完全一致率(017: 0.8180、許容 ±0.005)と形式の遵守率
  (017: 1.0000、許容 ±0.005)。許容は、デバイスによって logits の僅差の候補で貪欲法の選択が分かれうる分(5 事例)である。
- 同じ評価集合の応答部分の負の対数尤度(教師強制、017: 0.6005 nats / トークン、許容 ±0.0005)。

照合に失敗したら、以降の実験は意味を持たないので停止する。


```python
_t0_refcheck = time.time()
_prompts_016 = [list(x.token_ids[: x.prompt_length]) for x in EVAL_ENCODED_016]
_generated_016 = greedy_generate_until_stop(
    reference_policy, _prompts_016, tokenizer.decode, END_MARKER, MAX_NEW_TOKENS, device, 64
)
_scores_016 = [
    score_generated_response(tokenizer.decode(g), e.answer)
    for g, e in zip(_generated_016, EVAL_EXAMPLES_016, strict=True)
]
REFERENCE_FORMAT_RATE = float(np.mean([s[0] for s in _scores_016]))
REFERENCE_EXACT_MATCH = float(np.mean([s[1] for s in _scores_016]))
_nll = evaluate_instruction_negative_log_likelihood(reference_policy, EVAL_ENCODED_016, device)
REFERENCE_RESPONSE_NLL = float(_nll["response_sum"].sum() / _nll["response_count"].sum())
REFCHECK_SECONDS = time.time() - _t0_refcheck
print(
    f"完全一致率 {REFERENCE_EXACT_MATCH:.4f}(017: {REFERENCE_EXACT_MATCH_017:.4f}、差 "
    f"{REFERENCE_EXACT_MATCH - REFERENCE_EXACT_MATCH_017:+.4f})、形式の遵守率 {REFERENCE_FORMAT_RATE:.4f}"
    f"(017: {REFERENCE_FORMAT_RATE_017:.4f})、応答部分の負の対数尤度 {REFERENCE_RESPONSE_NLL:.5f}"
    f"(017: {REFERENCE_RESPONSE_NLL_017:.4f}、差 {REFERENCE_RESPONSE_NLL - REFERENCE_RESPONSE_NLL_017:+.5f})"
)
assert abs(REFERENCE_EXACT_MATCH - REFERENCE_EXACT_MATCH_017) <= REFERENCE_EXACT_TOLERANCE
assert abs(REFERENCE_FORMAT_RATE - REFERENCE_FORMAT_RATE_017) <= REFERENCE_EXACT_TOLERANCE
assert abs(REFERENCE_RESPONSE_NLL - REFERENCE_RESPONSE_NLL_017) <= REFERENCE_NLL_TOLERANCE
print(f"参照方策の照合: 017 の記録と許容の範囲で一致: OK({REFCHECK_SECONDS:.1f}s)")
```

    完全一致率 0.8180(017: 0.8180、差 +0.0000)、形式の遵守率 1.0000(017: 1.0000)、応答部分の負の対数尤度 0.60045(017: 0.6005、差 -0.00005)
    参照方策の照合: 017 の記録と許容の範囲で一致: OK(4.1s)


### 5.6 損失・対数確率・ラベル・類似度・CHES スコアのハーネスの確認

- **応答部分の対数確率**: `sequence_log_prob_sums()`(応答の位置だけを語彙へ射影する)が、017 の
  `gather_response_log_probs()`(全位置の logits から取り出す)の和と一致すること(行列積の丸めの範囲、$10^{-4}$ nats 以内)。
  `compute_response_log_prob_sums()`(長さ順のバッチ)が 1 系列ずつの計算と一致すること。
- **学習前の方策と参照方策の一致**: LoRA を掛けた直後($B = 0$)の方策の対数確率が、同じバッチの参照方策の対数確率と
  **bit 単位で一致** し、そこから計算した DPO 損失が $\log 2$、IPO 損失が $(1/(2\tau))^2$ に **完全に一致** すること。
- **損失と勾配**: `dpo_loss()`が`-F.logsigmoid`と一致すること、極端なマージンでも有限であること。DPO 損失のマージンに
  ついての勾配が $-\beta \sigma(-\beta h) / n$(3.3 節、$n$ はバッチの事例数)に一致すること。`ipo_loss()`の手計算との一致。
- **ラベル**: 重複させたラベルの抽選頻度が Bradley-Terry モデルの確率に一致すること、決定的なラベルが同点で停止すること、
  事例の列への展開の順序。
- **正規化編集距離・層別の選択**: 手計算の例との一致。層ごとの件数が近い組・遠い組で一致し、互いに重ならず、各層で近い組の
  距離が遠い組の距離以下であること。
- **CHES スコア**: バッチでの計算が 1 系列ずつの素朴な計算と一致すること。
- **学習ループ**: 数ステップの学習で、LoRA 以外のパラメータが参照方策と bit 単位で一致したまま(凍結)であること。


```python
_t0_harness = time.time()
_rng_check = np.random.default_rng(0)
_probe_prompts = [TRAIN_PROMPT_IDS[i] for i in range(8)]
_probe_responses = sample_generate_until_stop(
    reference_policy,
    _probe_prompts * 2,
    tokenizer.decode,
    END_MARKER,
    MAX_NEW_TOKENS,
    device,
    torch.Generator(device=device).manual_seed(0),
    TEMPERATURE,
    GENERATION_BATCH_SIZE,
)

# --- 応答部分の対数確率: 017 の gather_response_log_probs() の和との照合 ---
_batch = [t.to(device) for t in collate_rollout_sequences(_probe_prompts * 2, _probe_responses)]
with torch.no_grad():
    _new = sequence_log_prob_sums(reference_policy, *_batch)
    _old = gather_response_log_probs(reference_policy(_batch[0]), *_batch).sum(dim=1)
LOG_PROB_PATH_MAX_DIFF = float((_new - _old).abs().max())
assert LOG_PROB_PATH_MAX_DIFF <= 1e-4, LOG_PROB_PATH_MAX_DIFF
_batched = compute_response_log_prob_sums(
    reference_policy, _probe_prompts * 2, _probe_responses, device, 5
)
_single = np.array(
    [
        compute_response_log_prob_sums(reference_policy, [p], [r], device)[0]
        for p, r in zip(_probe_prompts * 2, _probe_responses, strict=True)
    ]
)
assert np.abs(_batched - _single).max() <= 1e-4
print(
    f"応答部分の対数確率: 017 の gather_response_log_probs() の和との差の最大値 {LOG_PROB_PATH_MAX_DIFF:.2e}"
    f"(<= 1e-4)、長さ順のバッチと 1 系列ずつの差の最大値 {np.abs(_batched - _single).max():.2e}: OK"
)

# --- 学習前(LoRA の B = 0)の方策と参照方策の一致、初期の損失 ---
_policy = copy.deepcopy(reference_policy)
torch.manual_seed(LORA_INIT_SEED_BASE)
apply_lora(_policy, LORA_TARGET_MODULES, rank=LORA_RANK, alpha=LORA_ALPHA)
assert all(
    torch.count_nonzero(m.lora_b) == 0 for m in _policy.modules() if isinstance(m, LoRALinear)
)
with torch.no_grad():
    _policy_sums = sequence_log_prob_sums(_policy, *_batch)
    _reference_sums = sequence_log_prob_sums(reference_policy, *_batch)
assert torch.equal(_policy_sums, _reference_sums), (
    "B = 0 の方策と参照方策の対数確率が bit 単位で一致しない"
)
_margin0 = compute_log_ratio_margin(
    _policy_sums[:8], _policy_sums[8:], _reference_sums[:8], _reference_sums[8:]
)
assert torch.equal(_margin0, torch.zeros_like(_margin0))
# DPO 損失 = log 2: 丸めた値 float32(log 2) との差を印字する(logaddexp の実装の丸めで 1 ulp までずれうるので、
# 1 ulp 以内を許す)。IPO 損失 (0 - 1/(2 tau))^2 は正確に表せる値なので完全一致を求める。
_dpo_initial_loss = dpo_loss(_margin0, CHECK_BETA).cpu()
_dpo_initial_diff = float(
    (_dpo_initial_loss - torch.tensor(math.log(2.0), dtype=torch.float32)).abs()
)
assert _dpo_initial_diff <= torch.finfo(torch.float32).eps * math.log(2.0), _dpo_initial_diff
assert torch.equal(ipo_loss(_margin0, CHECK_BETA).cpu(), torch.tensor((1 / (2 * CHECK_BETA)) ** 2))
print(
    "学習前(B = 0)の方策と参照方策の対数確率が bit 単位で一致、マージン 0、DPO 損失と float32 の log 2 の差 "
    f"{_dpo_initial_diff:.1e}(1 ulp 以内)、IPO 損失 = (1/(2 tau))^2 = {(1 / (2 * CHECK_BETA)) ** 2:g}(完全に一致): OK"
)

# --- 損失と勾配 ---
_h = torch.tensor([3.0, -50.0, 0.0, 40.0, -1e4, 1e4], requires_grad=True)
torch.testing.assert_close(dpo_loss(_h, 0.5), -F.logsigmoid(0.5 * _h).mean())
dpo_loss(_h, 0.5).backward()
torch.testing.assert_close(_h.grad, -0.5 * torch.sigmoid(-0.5 * _h.detach()) / _h.numel())
assert torch.isfinite(dpo_loss(_h.detach(), 0.5))
torch.testing.assert_close(
    ipo_loss(torch.tensor([1.0, 7.0]), 0.1), torch.tensor(((1 - 5) ** 2 + (7 - 5) ** 2) / 2)
)
print(
    "DPO 損失: -logsigmoid と一致、勾配 = -beta sigma(-beta h) / n、極端なマージンでも有限。IPO 損失の手計算と一致: OK"
)

# --- ラベル ---
_first_r, _second_r = np.array([0.5, 0.6, 0.9]), np.array([0.6, 0.5, 0.1])
_matrix = sample_preference_label_matrix(
    _first_r, _second_r, 4.0, 100_000, np.random.default_rng(1)
)
np.testing.assert_allclose(
    _matrix.mean(axis=1), compute_bradley_terry_probability(_first_r - _second_r, 4.0), atol=0.01
)
assert deterministic_preference_label_matrix(_first_r, _second_r, 3).tolist() == [
    [False] * 3,
    [True] * 3,
    [True] * 3,
]
try:
    deterministic_preference_label_matrix(np.array([0.5]), np.array([0.5]), 2)
    raise AssertionError("同点で停止しなかった")
except ValueError:
    pass
_pairs, _chosen = flatten_preference_labels(np.array([[True, False], [False, False]]))
assert _pairs.tolist() == [0, 0, 1, 1] and _chosen.tolist() == [True, False, False, False]
print(
    "ラベル: 抽選頻度が Bradley-Terry の確率と一致(誤差 0.01 以内)、決定的なラベルと同点での停止、展開の順序: OK"
)

# --- 正規化編集距離・層別の選択 ---
assert math.isclose(normalized_edit_distance("abc", "abd"), 1 / 3)
assert normalized_edit_distance("", "") == 0.0 and normalized_edit_distance("a", "") == 1.0
_labels = [int(v) for v in _rng_check.integers(0, 5, size=500)]
_distances = _rng_check.random(500)
_near, _far, _counts = select_similarity_stratified_pairs(_labels, _distances, 0.25)
assert not (set(_near.tolist()) & set(_far.tolist()))
for _label, _c in _counts.items():
    _n = [i for i in _near if _labels[i] == _label]
    _f = [i for i in _far if _labels[i] == _label]
    assert len(_n) == len(_f) == _c["near"] == int(_c["size"] * 0.25)
    assert max(_distances[_n]) <= min(_distances[_f])
print("正規化編集距離の手計算、層別の選択(層ごとの件数の一致・重なりなし・近い組 <= 遠い組): OK")

# --- CHES スコア: 1 系列ずつの素朴な計算との照合 ---
_stats = compute_ches_statistics(
    reference_policy, _probe_prompts, _probe_responses[:8], _probe_responses[8:], device, 3
)


def _naive_hidden_sum(prompt, response):
    tokens = torch.tensor([list(prompt) + list(response)], device=device)
    with torch.no_grad():
        hidden = compute_final_hidden_states(reference_policy, tokens)[0]
    return (
        hidden[len(prompt) - 1 : len(prompt) - 1 + len(response)].sum(dim=0).cpu().double().numpy()
    )


for _i in range(8):
    _s1 = _naive_hidden_sum(_probe_prompts[_i], _probe_responses[_i])
    _s2 = _naive_hidden_sum(_probe_prompts[_i], _probe_responses[8 + _i])
    np.testing.assert_allclose(
        [_stats["inner"][_i], _stats["norm_first"][_i], _stats["norm_second"][_i]],
        [_s1 @ _s2, _s1 @ _s1, _s2 @ _s2],
        rtol=1e-4,
    )
_ches = ches_from_statistics(_stats, np.array([True, False] * 4))
np.testing.assert_allclose(_ches[1], _stats["inner"][1] - _stats["norm_second"][1])
print(
    "CHES スコア: バッチでの計算が 1 系列ずつの素朴な計算と一致(相対誤差 1e-4 以内)、向きの指定: OK"
)

# --- 学習ループ: LoRA 以外が凍結されていること ---
_reference_first = compute_response_log_prob_sums(
    reference_policy, _probe_prompts, _probe_responses[:8], device
)
_reference_second = compute_response_log_prob_sums(
    reference_policy, _probe_prompts, _probe_responses[8:], device
)
_history = train_direct_preference(
    _policy,
    _probe_prompts,
    _probe_responses[:8],
    _probe_responses[8:],
    _reference_first,
    _reference_second,
    [[0, 1, 2, 3], [4, 5, 6, 7], [0, 2, 4, 6]],
    AdamW([p for p in _policy.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=0.0),
    "dpo",
    CHECK_BETA,
    device,
)
_reference_parameters = dict(reference_policy.named_parameters())
for _name, _p in _policy.named_parameters():
    if _p.requires_grad:
        assert "lora_" in _name
    else:
        assert torch.equal(_p, _reference_parameters[_name.replace(".base_layer", "")]), _name
assert abs(_history["loss"][0] - math.log(2.0)) <= 1e-5 and len(_history["loss"]) == 3
del _policy
HARNESS_SECONDS = time.time() - _t0_harness
print(
    f"学習ループ: 3 ステップの後も LoRA 以外のパラメータが参照方策と bit 単位で一致、1 ステップ目の損失 "
    f"{_history['loss'][0]:.7f}(log 2 = {math.log(2):.7f}): OK"
)
print(f"\nこのセルの実行時間: {HARNESS_SECONDS:.1f}s")
```

    応答部分の対数確率: 017 の gather_response_log_probs() の和との差の最大値 0.00e+00(<= 1e-4)、長さ順のバッチと 1 系列ずつの差の最大値 1.14e-05: OK
    学習前(B = 0)の方策と参照方策の対数確率が bit 単位で一致、マージン 0、DPO 損失と float32 の log 2 の差 0.0e+00(1 ulp 以内)、IPO 損失 = (1/(2 tau))^2 = 25(完全に一致): OK
    DPO 損失: -logsigmoid と一致、勾配 = -beta sigma(-beta h) / n、極端なマージンでも有限。IPO 損失の手計算と一致: OK
    ラベル: 抽選頻度が Bradley-Terry の確率と一致(誤差 0.01 以内)、決定的なラベルと同点での停止、展開の順序: OK
    正規化編集距離の手計算、層別の選択(層ごとの件数の一致・重なりなし・近い組 <= 遠い組): OK
    CHES スコア: バッチでの計算が 1 系列ずつの素朴な計算と一致(相対誤差 1e-4 以内)、向きの指定: OK
    学習ループ: 3 ステップの後も LoRA 以外のパラメータが参照方策と bit 単位で一致、1 ステップ目の損失 0.6931472(log 2 = 0.6931472): OK
    
    このセルの実行時間: 1.2s




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/04_alignment/018_direct_preference_optimization.ipynb
