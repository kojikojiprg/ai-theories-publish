---
title: "Mixture of Experts(MoE) / Mixture of Experts(実装・実験編 1/5)"
---

この記事は後編(実装・実験編 1/5)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/022_mixture_of_experts-theory)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/022_mixture_of_experts-practice-2)。

## 4. 実装方針 / Implementation Policy

**`src/`に切り出す(スクラッチ実装、本トピックで新規作成)**:

- `src/layers/moe.py`: MoE 層`MixtureOfExpertsFeedForward`。ルーター(線形写像、バイアスなし、FP32 で計算)、top-k の選択とゲート
  (主構成は $k = 1$。$k = 2$ も動く)、容量でパディングした固定形状の振り分け・集約(形状 $(N, C, d_{\mathrm{model}})$ のテンソルと
  `torch.bmm`。エキスパートごとの Python のループは使わない)、容量を超えたトークンの破棄(評価時は破棄しない設定にできる)、負荷分散損失・router z-loss(係数を
  掛ける前の値)、診断量(エキスパートごとのトークンの割合 $f_i$ とルーターの確率の平均 $P_i$(どちらも破棄の前の割り当てで数える)、
  破棄されたトークンの割合、トークンごとのルーターの確率の最大値、ルーターのロジットのノルム)。容量の計算
  `compute_expert_capacity()`と枠の割り当て`compute_dispatch_slots()`は、単体テストのために関数として切り出す。

**既存モジュールの拡張**(既定の引数では、変更前のコミットと bit 単位で一致することを 5.4 節で確かめる):

- `src/models/gpt.py`: `GPTLanguageModel`に`auxiliary_losses()`を追加する(各層の順伝播ネットワークが直近の順伝播で記録した補助損失を、
  層について合計して返す)。MoE 層への差し替えは、004 で追加した既存の注入点`feed_forward_factory`をそのまま使う(`forward()`は
  変更しない)。
- `src/training/trainer.py`: `train_language_model()`に、補助損失に係数を掛けて加える`auxiliary_loss_coefficients`、検証するステップを
  明示する`eval_steps`、検証の関数を差し替えて診断量を記録する`evaluation_fn`を追加する(いずれも既定では無効)。評価窓ごとの負の
  対数尤度を返す`evaluate_window_negative_log_likelihoods()`を追加する(記事を単位とするブートストラップに使う)。
- `src/utils/statistics.py`: 正規化エントロピー`compute_normalized_entropy()`(実験 B の対比量)、正規化した相互情報量
  `compute_normalized_mutual_information()`(観察 D)を追加する。

**既存の部品をそのまま使うもの**: 小型 GPT の構成(RoPE・RMSNorm・SwiGLU・正規化前置・重み共有)は 006・008、optimizer は 007 の
`AdamW`(020 で追加した`foreach=True`)、学習率のスケジュールは 007 の warmup + cosine、混合精度は 011 の`DynamicLossScaler`と
`torch.autocast`、ブートストラップは 015 の`paired_cluster_bootstrap_ratio_of_sums()`、数値の印字は`dumps_compact_json()`。

**ノートブック内に直接書く(022 固有)**: 条件の定義、データの分割と評価窓、トークンの種類の分類(観察 D)、単体テストと不変条件の
確認、学習率の較正、実行計画の選択、判定、可視化、スケーリングの計測。

**アップロード方針**: 本トピックで学習するモデルは、すべて条件間の比較のためのものであり、後続トピックの入力にも、読者が単体で
取得する対象にもならない。保存もアップロードもしない(アップロードのセルも置かない)。

**生成物の置き場所**:

| 生成物 | 置き場所 |
|---|---|
| コーパス(Hub から取得できなかった場合の Wikipedia API からの取得結果) | `.cache/wikipedia_en/`(データ源で命名。決定的に再取得できる) |
| Hugging Face Hub から取得したコーパス・トークナイザ | `huggingface_hub`の既定のキャッシュ(Colab ではセッションの終了とともに破棄される) |
| 符号化したトークン列・評価窓 | メモリ上のみ(符号化は全体で数秒なので、`.cache/`にも置かない) |
| 学習したモデル | メモリ上のみ(評価の直後に破棄する) |
| 学習の履歴・評価窓ごとの負の対数尤度・割り当ての個数 | メモリ上のみ(判定と図に使う) |
| 判定の記録 | 判定と前提条件を計算した各セルの出力(6.7〜6.9・6.12 節) |

**外部からの取得**(Colab のセットアップセルのリポジトリの取得と依存関係のインストールを除く):

| 取得するもの | リポジトリ | 用途 |
|---|---|---|
| コーパス(`corpus.txt`・`metadata.json`) | `kojikojiprg/ai-theories-corpus-en-pretraining`(Dataset、356 記事) | 008 と同じ英語 Wikipedia のコーパスと、記事の境界(`article_offsets`) |
| トークナイザ(`tokenizer.json`) | `kojikojiprg/ai-theories-tokenizer-en` | 008 の英語の BPE(Byte Pair Encoding)トークナイザ(語彙サイズ 8192) |

008 の学習済みの重み(`kojikojiprg/ai-theories-small-gpt-en`)は使わない。密なモデルも MoE も、乱数の初期値から学習する。

## 5. 実装 / Implementation

### 5.1 環境セットアップ(Google Colab)

`SMOKE_TEST`(スモークテストか本番か)はこのセルでのみ切り替える。実行環境はこのセルで 1 回だけ印字する。

**本番のコミットの確認**: `SMOKE_TEST = False`のときだけ、(1)コミット`a835c21`(評価の手順の改訂を含むコミット)が HEAD の祖先で
あること、(2)HEAD が`a835c21`そのものではないこと、(3)追跡しているファイルに未コミットの変更がないこと、を確かめ、満たさなければ
学習の前に停止する。追跡外のファイルは停止の条件にせず、あれば一覧を参考として印字する。

**精度と決定性**: 学習は、CUDA のときは FP16 の`torch.autocast`と動的損失スケーリング(011 の`DynamicLossScaler`)で行い、それ以外の
デバイス(ローカルの MPS・CPU)では FP32 で行う。ルーターは常に FP32 で計算する(3.6 節)。評価は常に FP32 で行う。
`torch.use_deterministic_algorithms(True)`は使わない。同じシードの学習を繰り返しても結果が bit 単位では一致しないことがあるが、
条件間の対応付け(同じシードの条件どうしで初期値とミニバッチの順序を揃えること)は乱数の生成器によって決まり、演算の決定性には
依存しない(5.4・6.11 節で確かめる)。


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
USE_FP16_AUTOCAST = device.type == "cuda"  # FP16 の autocast と動的損失スケーリングは CUDA のときのみ
execution_environment = print_execution_environment(device)

# 本番(SMOKE_TEST = False)は、評価の手順の改訂(6.1 節)と汎化の差の診断量を含むコミットで実行する。満たさなければ学習の前に停止する
REQUIRED_ANCESTOR_COMMIT = "a835c21"  # 評価の手順の改訂を含むコミット。本番は、その子孫で、それ自身ではないコミットで行う
if not SMOKE_TEST:
    import subprocess

    _head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    _base = subprocess.run(["git", "rev-parse", REQUIRED_ANCESTOR_COMMIT], capture_output=True, text=True, check=True).stdout.strip()
    _is_descendant = subprocess.run(["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR_COMMIT, "HEAD"]).returncode == 0
    # 追跡しているファイルの変更だけを停止の条件にする。追跡外のファイル(キャッシュなど)は参考として印字する
    _status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True
    ).stdout.strip()
    _untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"], capture_output=True, text=True, check=True
    ).stdout.split()
    if not _is_descendant or _head == _base or _status:
        raise RuntimeError(
            f"本番の実行条件を満たさないため、学習の前に停止する(結果の情報は何も得ていない)。HEAD {_head[:7]}、"
            f"{REQUIRED_ANCESTOR_COMMIT} の子孫か: {_is_descendant}、{REQUIRED_ANCESTOR_COMMIT} そのものか: {_head == _base}、"
            f"追跡しているファイルの未コミットの変更: {_status or 'なし'}。リポジトリを最新の main に更新し、変更をなくして再実行すること。"
        )
    print(
        f"本番の実行条件: HEAD {_head[:7]} は {REQUIRED_ANCESTOR_COMMIT} の子孫で、それ自身ではなく、追跡しているファイルに未コミットの変更がない: OK"
        f"(参考: 追跡外のファイル {_untracked or 'なし'})"
    )
print(
    f"SMOKE_TEST={SMOKE_TEST}、学習の精度 {'FP16 の autocast + 動的損失スケーリング' if USE_FP16_AUTOCAST else 'FP32'}"
    f"(ルーターと評価は FP32)、決定的な演算の強制: {torch.are_deterministic_algorithms_enabled()}"
)
```

    /content/ai-theories
    [2mUsing Python 3.13.15 environment at: /usr[0m
    [2mChecked [1m60 packages[0m [2min 301ms[0m[0m
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
      コミット / git commit                  : 8f89fd3086ed7b22bb9dfae6bc926b9f49b6cc19
      未コミットの変更 / uncommitted changes : なし
      実行日時 (UTC)                         : 2026-10-02T03:56:56+00:00
    本番の実行条件: HEAD 8f89fd3 は a835c21 の子孫で、それ自身ではなく、追跡しているファイルに未コミットの変更がない: OK(参考: 追跡外のファイル なし)
    SMOKE_TEST=False、学習の精度 FP16 の autocast + 動的損失スケーリング(ルーターと評価は FP32)、決定的な演算の強制: False



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

from src.data.text import (
    encode_corpus,
    load_wikipedia_corpus_with_fallback,
    locate_wikipedia_article_spans,
    split_train_val_text,
)
from src.data.tokenizer import load_bpe_id_tokenizer_from_hub, try_decode_byte_level_symbol
from src.layers.feedforward import SwiGLUFeedForwardNetwork
from src.layers.moe import (
    LOAD_BALANCING_LOSS_NAME,
    ROUTER_Z_LOSS_NAME,
    MixtureOfExpertsFeedForward,
    compute_dispatch_slots,
    compute_expert_capacity,
    find_moe_layers,
    set_statistics_tracking,
)
from src.layers.normalization import RMSNorm
from src.layers.positional_encoding import RotaryPositionEmbedding
from src.models.gpt import GPTLanguageModel
from src.training.optimizer import AdamW
from src.training.precision import DynamicLossScaler
from src.training.schedule import compute_warmup_cosine_learning_rate
from src.training.trainer import (
    evaluate_bits_per_byte,
    evaluate_window_negative_log_likelihoods,
    train_language_model,
)
from src.utils.reporting import dumps_compact_json
from src.utils.statistics import (
    compute_normalized_entropy,
    compute_normalized_mutual_information,
    count_non_embedding_parameters,
    fit_power_law_exponent,
    paired_cluster_bootstrap_ratio_of_sums,
)

ROOT = Path.cwd()
WIKIPEDIA_CACHE_DIR = ROOT / ".cache" / "wikipedia_en"  # 外部から取得したコーパス(データ源で命名)
CORPUS_REPO_ID = "kojikojiprg/ai-theories-corpus-en-pretraining"
TOKENIZER_REPO_ID = "kojikojiprg/ai-theories-tokenizer-en"
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


def sha256_of_tensors(tensors) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def rounded(values, digits: int = 4):
    # 印字用: 入れ子のリスト・配列を丸めたリストにする
    return np.round(np.asarray(values, dtype=np.float64), digits).tolist()


precondition_status: dict[str, bool] = {}  # 前提条件の成否(6.1 節で宣言、各節で記録)
```

### 5.2 スケールの設定(`SMOKE_TEST`の配線)

水準の定義をこの 1 箇所に集約する。

**縮小の規則**: スモークテストは、本番と **同じモデル・同じ学習データ・同じ条件・同じ実行計画の構造** で、次の量だけを縮小する。

- **学習ステップ数 $T$ の候補**: 本番 $(2181, 1090, 545)$、スモークテスト $(16, 8, 4)$。どちらも大きい順で、隣どうしの比が 2 の
  等比(本番は $2181 / 2^j$ を丸めた値)という構造を保つ。
- **シード数**: 本番は 5(削った段階で 3)、スモークテストは 3(削った段階で 2)。「削る前 > 削った後 $\ge 2$」の順序関係を保ち、
  各段階で何が変わるかは本番と同じにする。
- **評価窓**: 本番は評価集合の全部の窓。スモークテストは、評価窓を 2 個以上持つ記事のそれぞれから先頭の 2 個(記事を単位とする
  ブートストラップの経路を、同じ記事に複数の窓があるクラスタを含めて実行するため)。較正用の集合は、本番は全部、スモークテストは
  先頭の 16 個。
- **ブートストラップの反復回数**: 本番 10,000 回、スモークテスト 1,000 回。

スモークテストはステップ数が極端に少ないので、学習は進まず、前提条件は成立しない見込みである(コードの経路の確認が目的)。
**較正・前提条件・検出力の確認は、6.2 節のパイロットで本番と同じステップ数で行った。**

**縮小しないもの**: モデルの構成、学習データ、バッチサイズ、系列長、条件の対応表、学習率の較正の格子と拡張の規則、学習率の
スケジュールの形、途中の評価の位置の決め方($T$ に対する割合)、実行計画の表の構造と優先順位、前提条件と判定の閾値、
スケーリングの計測点。

**テスト専用の上書き**: 環境変数`AI_THEORIES_FORCE_PLAN`(0〜29)が設定されているときのみ、6.4 節で見積もりによる計画の選択の代わりに
その計画を使う(スモークテストで下位の計画の経路を確かめるため)。本番では受け付けず、このセルで停止する。コミットする出力は
上書きなしの実行のものである。

**実効水準の照合**: このセルで印字した水準(ステップ数・シード数・計画の表)が、実際の学習で使われた値と一致することを 6.11 節の
アサーションで確かめる(`SMOKE_TEST`の配線漏れの検出)。


```python
# --- 全水準で共通の定数(本番実行前に宣言し、SMOKE_TEST で変えない) ---
# モデル(006・008 と同じ小型 GPT)
VOCAB_SIZE = 8192
D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF = 256, 4, 8, 1024
SEQUENCE_LENGTH = 256
SWIGLU_D_FF = round((2 / 3) * D_FF)  # 683(004: 標準の順伝播ネットワークとパラメータ数を揃える丸め)
WIDE_FACTOR = 8  # 条件 W: 順伝播ネットワークの中間次元を 8 倍にした密なモデル(総パラメータ数を M8 に揃える)
# データ
VALIDATION_RATIO = 0.05  # 008 と同じ(コーパスの末尾 5% が評価集合)
CALIBRATION_RATIO = 0.03  # 評価集合の直前の 3% を較正用の集合にする(学習に使わない)
# 学習(008 のレシピ)
BATCH_SIZE = 32
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.1
MIN_LEARNING_RATE_RATIO = 0.01
GRADIENT_CLIP_THRESHOLD = 1.0
INIT_LOSS_SCALE = 2.0**16  # 動的損失スケーリング(019〜021 と同じ)
LOSS_SCALE_GROWTH_INTERVAL = 2000
# MoE(係数と capacity factor は原論文の値、3.4〜3.6 節)
TOP_K = 1
LOAD_BALANCING_ALPHA = 1e-2  # Fedus et al. の alpha
ROUTER_Z_COEFFICIENT = 1e-3  # Zoph et al. の c_z
TRAIN_CAPACITY_FACTOR = 1.25  # Zoph et al.
EVAL_CAPACITY_FACTOR = None  # 評価時は破棄しない(容量 = バッチのトークン数、6.1 節の「評価の手順の改訂」)
REFERENCE_EVAL_CAPACITY_FACTOR = 2.0  # Zoph et al. の評価時の値。「破棄されたはずの割り当ての割合」(P2・診断量)の計算に使う
# 条件の対応表(6.1 節)
CONDITIONS = {
    "D": {"kind": "dense", "experts": 1, "d_ff": SWIGLU_D_FF, "alpha": None},
    "M2": {"kind": "moe", "experts": 2, "d_ff": SWIGLU_D_FF, "alpha": LOAD_BALANCING_ALPHA},
    "M4": {"kind": "moe", "experts": 4, "d_ff": SWIGLU_D_FF, "alpha": LOAD_BALANCING_ALPHA},
    "M8": {"kind": "moe", "experts": 8, "d_ff": SWIGLU_D_FF, "alpha": LOAD_BALANCING_ALPHA},
    "M16": {"kind": "moe", "experts": 16, "d_ff": SWIGLU_D_FF, "alpha": LOAD_BALANCING_ALPHA},
    "N8": {"kind": "moe", "experts": 8, "d_ff": SWIGLU_D_FF, "alpha": 0.0},
    "W": {"kind": "dense", "experts": 1, "d_ff": WIDE_FACTOR * SWIGLU_D_FF, "alpha": None},
}
TIMING_KIND = {"D": "D", "M2": "M2", "M4": "M4", "M8": "M8", "M16": "M16", "N8": "M8", "W": "W"}  # 時間の見積もりの対応
RUN_ORDER = ("D", "M8", "N8", "M4", "M16", "M2", "W")  # 本番の学習の順
# 学習率の較正(6.1 節)。中心は 6.2 節のパイロットで決めた
LR_CENTER = {"D": 2.4e-3, "M2": 2.4e-3, "M4": 2.4e-3, "M8": 2.4e-3, "M16": 2.4e-3, "N8": 2.4e-3, "W": 2.4e-3}
LR_GRID_MULTIPLIERS = (0.5, 1.0, 2.0)  # 格子 = 中心 x {1/2, 1, 2}(公比 2)
LR_GRID_RATIO = 2.0
REPRESENTATIVE_CALIBRATED = ("D", "M8")  # 較正の方式 "representative" で較正する条件
REPRESENTATIVE_RULE = {"M2": "M8", "M4": "M8", "M16": "M8", "N8": "M8", "W": "D"}  # 規則で決める条件 -> 値を借りる条件
# 途中の評価の位置(T に対する割合、初期に密な等比の間隔)
EVAL_FRACTIONS = (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
EVAL_BATCH_WINDOWS = 16  # 評価の 1 回の順伝播の窓の数(評価は破棄なしなので結果には影響しない。「破棄されたはずの割合」はこの区切りで数える)
# 前提条件と判定(6.1 節)
P1_LOSS_RATIO = 0.65  # P1: 最後の区間の訓練損失 <= ln(V) x この値
FINAL_LOSS_FRACTION = 0.05  # 「最後の区間」= 最後の 5% のステップ
P2_DROPPED_MAX = 0.15  # P2: 評価集合で、capacity factor 2.0 なら破棄されたはずの割り当ての割合(層の最大)<= この値
P2_LOAD_FACTOR = 3.0  # P2: 最大のエキスパートの負荷の割合 <= min(1, この値 / E)(一様な割り当ての 3 倍)
P3_MAX_PROBABILITY_FACTOR = 1.5  # P3: ルーターの確率の最大値の平均(層平均)>= この値 / E
RARE_EXPERT_FACTOR = 0.1  # 診断量: f_i < この値 / E のエキスパートを「ほとんど選ばれない」と数える
SIGMA_MULTIPLIER = 2.0  # 判定の閾値は対比量の標準偏差の 2 倍
# 乱数シード(学習のシード s: 密なモデルの初期化 22100 + s、順伝播ネットワークの差し替えの初期化 22300 + s、ミニバッチ 22200 + s)
INIT_SEED_BASE, DATA_SEED_BASE, FEED_FORWARD_SEED_BASE = 22_100, 22_200, 22_300
CALIBRATION_SEED_INDEX = 90  # 学習率の較正専用のシード(実験のシード 0〜4 と共有しない)
TIMING_SEED_INDEX = 91  # スケーリングの計測専用のシード
BOOTSTRAP_SEED = 22_600
SESSION_BUDGET_SECONDS = 120 * 60  # 1 セッションの予算(T4 で 120 分)
OBSERVATION_RUN = ("M8", 0)  # 観察 D に使う学習(実験 A の MoE、シード 0)

# --- 水準(SMOKE_TEST で変わるもの) ---
LEVELS = {
    "smoke": {
        "STEP_CANDIDATES": (16, 8, 4),
        "MAX_WINDOWS_PER_ARTICLE": 2,
        "MAX_CALIBRATION_WINDOWS": 16,
        "BOOTSTRAP_RESAMPLES": 1_000,
    },
    "prod": {
        "STEP_CANDIDATES": (2181, 1090, 545),  # 2181 は 008 と同じ(3 エポック相当)。以降は半分ずつ
        "MAX_WINDOWS_PER_ARTICLE": None,
        "MAX_CALIBRATION_WINDOWS": None,
        "BOOTSTRAP_RESAMPLES": 10_000,
    },
}
# 削る段階(6.1 節)。順序: W(診断量)-> M2 のシード数 -> N8 のシード数 -> M16 のシード数
STAGES = {
    "prod": {
        0: {"D": 5, "M8": 5, "N8": 5, "M4": 5, "M16": 5, "M2": 5, "W": 5},
        1: {"D": 5, "M8": 5, "N8": 5, "M4": 5, "M16": 5, "M2": 5, "W": 0},
        2: {"D": 5, "M8": 5, "N8": 5, "M4": 5, "M16": 5, "M2": 3, "W": 0},
        3: {"D": 5, "M8": 5, "N8": 3, "M4": 5, "M16": 5, "M2": 3, "W": 0},
        4: {"D": 5, "M8": 5, "N8": 3, "M4": 5, "M16": 3, "M2": 3, "W": 0},
    },
    "smoke": {  # 本番と同じ構造(シード数 5 -> 3 を 3 -> 2 に縮小)
        0: {"D": 3, "M8": 3, "N8": 3, "M4": 3, "M16": 3, "M2": 3, "W": 3},
        1: {"D": 3, "M8": 3, "N8": 3, "M4": 3, "M16": 3, "M2": 3, "W": 0},
        2: {"D": 3, "M8": 3, "N8": 3, "M4": 3, "M16": 3, "M2": 2, "W": 0},
        3: {"D": 3, "M8": 3, "N8": 2, "M4": 3, "M16": 3, "M2": 2, "W": 0},
        4: {"D": 3, "M8": 3, "N8": 2, "M4": 3, "M16": 2, "M2": 2, "W": 0},
    },
}
NUM_STAGES = 5
CALIBRATION_MODES = ("all", "representative")
# スケーリングの計測点(6.3 節)。水準によらず同じ
SCALING_STEP_COUNTS = (8, 16, 32)
SCALING_WARMUP_STEPS = 8
SCALING_ENCODE_CHARACTERS = (1_000_000, 2_000_000, 4_000_000)

CURRENT_LEVEL_NAME = "smoke" if SMOKE_TEST else "prod"
CFG = LEVELS[CURRENT_LEVEL_NAME]
STEP_CANDIDATES = CFG["STEP_CANDIDATES"]
BOOTSTRAP_RESAMPLES = CFG["BOOTSTRAP_RESAMPLES"]


def build_plans(step_candidates) -> list[dict]:
    # 実行計画(6.1 節): 学習ステップ数 T(大きい順)-> 較正の方式("all" -> "representative")-> 削る段階(0 -> 4)
    plans = [
        {"num_steps": t, "calibration_mode": mode, "stage": k}
        for t in step_candidates
        for mode in CALIBRATION_MODES
        for k in range(NUM_STAGES)
    ]
    return [{"plan": i} | p for i, p in enumerate(plans)]


PLANS = build_plans(STEP_CANDIDATES)
FORCED_PLAN_VALUE = os.environ.get("AI_THEORIES_FORCE_PLAN")
if FORCED_PLAN_VALUE is not None and not SMOKE_TEST:
    raise RuntimeError(
        f"本番(SMOKE_TEST=False)では計画の強制(AI_THEORIES_FORCE_PLAN={FORCED_PLAN_VALUE!r})を受け付けない。"
        "環境変数を削除して再実行すること。"
    )
FORCED_PLAN = None if FORCED_PLAN_VALUE is None else int(FORCED_PLAN_VALUE)
assert FORCED_PLAN is None or 0 <= FORCED_PLAN < len(PLANS), FORCED_PLAN
# T・較正の方式・段階・シード数は 6.4 節で実行計画を選んだ後に決まる


def warmup_steps_for(num_steps: int) -> int:
    return max(1, round(WARMUP_RATIO * num_steps))


def eval_steps_for(num_steps: int) -> tuple[int, ...]:
    # 途中の評価のステップ(重複は除く。最後は必ず T)
    return tuple(sorted({max(1, round(f * num_steps)) for f in EVAL_FRACTIONS}))


def final_loss_window(num_steps: int) -> int:
    return max(1, round(FINAL_LOSS_FRACTION * num_steps))


def lr_grid_for(condition: str) -> tuple[float, ...]:
    return tuple(LR_CENTER[condition] * m for m in LR_GRID_MULTIPLIERS)


def calibration_targets(mode: str, seeds: dict) -> list[str]:
    # 較正する条件。"all" は学習する全条件、"representative" は D と M8 のみ(6.1 節)
    if mode == "all":
        return [c for c in RUN_ORDER if seeds[c] > 0]
    return [c for c in REPRESENTATIVE_CALIBRATED if seeds[c] > 0]


# --- 縮小規則と水準の構造の確認 ---
assert set(LEVELS["smoke"]) == set(LEVELS["prod"])
for _name in LEVELS:
    _candidates = LEVELS[_name]["STEP_CANDIDATES"]
    assert len(_candidates) == 3 and list(_candidates) == sorted(_candidates, reverse=True)
    assert all(b == round(_candidates[0] / 2**j) for j, b in enumerate(_candidates))  # 等比(公比 1/2)
    for _t in _candidates:
        assert eval_steps_for(_t)[-1] == _t and warmup_steps_for(_t) < _t
assert LEVELS["smoke"]["BOOTSTRAP_RESAMPLES"] < LEVELS["prod"]["BOOTSTRAP_RESAMPLES"]
for _name, _stages in STAGES.items():
    assert set(_stages) == set(range(NUM_STAGES))
    for _k in range(NUM_STAGES):
        assert set(_stages[_k]) == set(CONDITIONS)
        assert all(v == 0 or v >= 2 for v in _stages[_k].values())
        assert _stages[_k]["D"] == _stages[_k]["M8"] == _stages[_k]["M4"] == _stages[0]["D"]  # 削らない条件
        if _k > 0:  # 段階が上がるほど、どの量も減るか変わらない。各段階で変わる条件は 1 つ
            assert all(_stages[_k][c] <= _stages[_k - 1][c] for c in CONDITIONS)
            assert sum(_stages[_k][c] != _stages[_k - 1][c] for c in CONDITIONS) == 1
for _k in range(1, NUM_STAGES):  # 本番とスモークテストで、各段階で何が変わるかが同じ(構造を保つ縮小)
    for _c in CONDITIONS:
        assert (STAGES["prod"][_k][_c] == STAGES["prod"][_k - 1][_c]) == (
            STAGES["smoke"][_k][_c] == STAGES["smoke"][_k - 1][_c]
        )
assert len(PLANS) == 30 and [p["plan"] for p in PLANS] == list(range(30))
assert all(b == 2 * a for a, b in itertools.pairwise(SCALING_STEP_COUNTS))
assert all(b == 2 * a for a, b in itertools.pairwise(SCALING_ENCODE_CHARACTERS))
assert all(math.isclose(b / a, 2.0) for a, b in itertools.pairwise(EVAL_FRACTIONS))  # 途中の評価の位置は等比
for _c in CONDITIONS:
    assert all(math.isclose(b / a, LR_GRID_RATIO) for a, b in itertools.pairwise(lr_grid_for(_c)))
assert [CONDITIONS[c]["experts"] for c in ("D", "M2", "M4", "M8", "M16")] == [1, 2, 4, 8, 16]  # 実験 C の水準は等比
assert set(REPRESENTATIVE_RULE) | set(REPRESENTATIVE_CALIBRATED) == set(CONDITIONS)

print(f"水準 {CURRENT_LEVEL_NAME!r}: {json.dumps(CFG)}")
print(
    f"モデル: {NUM_LAYERS} 層、d_model {D_MODEL}、ヘッド数 {NUM_HEADS}、SwiGLU の中間次元 {SWIGLU_D_FF}(W は {WIDE_FACTOR} 倍)、"
    f"系列長 {SEQUENCE_LENGTH}、バッチ {BATCH_SIZE}、語彙サイズ {VOCAB_SIZE}"
)
print(
    f"学習: AdamW(重み減衰 {WEIGHT_DECAY}、foreach)、warmup {WARMUP_RATIO:.0%} + cosine(下限 x{MIN_LEARNING_RATE_RATIO})、"
    f"gradient clipping {GRADIENT_CLIP_THRESHOLD}。MoE: top-{TOP_K}、alpha = {LOAD_BALANCING_ALPHA}、c_z = {ROUTER_Z_COEFFICIENT}、"
    f"capacity factor 学習時 {TRAIN_CAPACITY_FACTOR}・評価時は破棄なし(診断量の基準 {REFERENCE_EVAL_CAPACITY_FACTOR})"
)
print(f"条件: {dumps_compact_json(CONDITIONS)}")
print("学習率の格子: " + "、".join(f"{c} {tuple(float(f'{x:.3g}') for x in lr_grid_for(c))}" for c in CONDITIONS))
for _t in STEP_CANDIDATES:
    print(f"  T = {_t}: warmup {warmup_steps_for(_t)}、評価のステップ {eval_steps_for(_t)}、最後の区間 {final_loss_window(_t)} ステップ")
print(f"削る段階({CURRENT_LEVEL_NAME!r}、条件ごとのシード数): {dumps_compact_json(STAGES[CURRENT_LEVEL_NAME])}")
print(
    f"実行計画: {len(PLANS)} 通り(番号の小さいほど優先)。計画番号 = 10 x(T の番号)+ 5 x(較正の方式の番号)+ 段階。"
    f"T の候補 {STEP_CANDIDATES}、較正の方式 {CALIBRATION_MODES}、段階 0〜{NUM_STAGES - 1}"
)
print(f"スケーリングの計測点: ステップ数 {SCALING_STEP_COUNTS}(ウォームアップ {SCALING_WARMUP_STEPS})")
if FORCED_PLAN is not None:
    print(f"*** テスト専用の上書き: AI_THEORIES_FORCE_PLAN = {FORCED_PLAN}(6.4 節で見積もりの代わりにこの計画を使う) ***")
```

    水準 'prod': {"STEP_CANDIDATES": [2181, 1090, 545], "MAX_WINDOWS_PER_ARTICLE": null, "MAX_CALIBRATION_WINDOWS": null, "BOOTSTRAP_RESAMPLES": 10000}
    モデル: 4 層、d_model 256、ヘッド数 8、SwiGLU の中間次元 683(W は 8 倍)、系列長 256、バッチ 32、語彙サイズ 8192
    学習: AdamW(重み減衰 0.1、foreach)、warmup 10% + cosine(下限 x0.01)、gradient clipping 1.0。MoE: top-1、alpha = 0.01、c_z = 0.001、capacity factor 学習時 1.25・評価時は破棄なし(診断量の基準 2.0)
    条件: {
      "D": {"kind": "dense", "experts": 1, "d_ff": 683, "alpha": null},
      "M2": {"kind": "moe", "experts": 2, "d_ff": 683, "alpha": 0.01},
      "M4": {"kind": "moe", "experts": 4, "d_ff": 683, "alpha": 0.01},
      "M8": {"kind": "moe", "experts": 8, "d_ff": 683, "alpha": 0.01},
      "M16": {"kind": "moe", "experts": 16, "d_ff": 683, "alpha": 0.01},
      "N8": {"kind": "moe", "experts": 8, "d_ff": 683, "alpha": 0.0},
      "W": {"kind": "dense", "experts": 1, "d_ff": 5464, "alpha": null}
    }
    学習率の格子: D (0.0012, 0.0024, 0.0048)、M2 (0.0012, 0.0024, 0.0048)、M4 (0.0012, 0.0024, 0.0048)、M8 (0.0012, 0.0024, 0.0048)、M16 (0.0012, 0.0024, 0.0048)、N8 (0.0012, 0.0024, 0.0048)、W (0.0012, 0.0024, 0.0048)
      T = 2181: warmup 218、評価のステップ (34, 68, 136, 273, 545, 1090, 2181)、最後の区間 109 ステップ
      T = 1090: warmup 109、評価のステップ (17, 34, 68, 136, 272, 545, 1090)、最後の区間 54 ステップ
      T = 545: warmup 54、評価のステップ (9, 17, 34, 68, 136, 272, 545)、最後の区間 27 ステップ
    削る段階('prod'、条件ごとのシード数): {
      "0": {"D": 5, "M8": 5, "N8": 5, "M4": 5, "M16": 5, "M2": 5, "W": 5},
      "1": {"D": 5, "M8": 5, "N8": 5, "M4": 5, "M16": 5, "M2": 5, "W": 0},
      "2": {"D": 5, "M8": 5, "N8": 5, "M4": 5, "M16": 5, "M2": 3, "W": 0},
      "3": {"D": 5, "M8": 5, "N8": 3, "M4": 5, "M16": 5, "M2": 3, "W": 0},
      "4": {"D": 5, "M8": 5, "N8": 3, "M4": 5, "M16": 3, "M2": 3, "W": 0}
    }
    実行計画: 30 通り(番号の小さいほど優先)。計画番号 = 10 x(T の番号)+ 5 x(較正の方式の番号)+ 段階。T の候補 (2181, 1090, 545)、較正の方式 ('all', 'representative')、段階 0〜4
    スケーリングの計測点: ステップ数 (8, 16, 32)(ウォームアップ 8)


### 5.3 データ: コーパスの分割・符号化・評価窓・トークンの種類

**コーパスとトークナイザ**: 008 と同じ英語 Wikipedia のコーパス(356 記事、`kojikojiprg/ai-theories-corpus-en-pretraining`)と、
008 の BPE(Byte Pair Encoding)トークナイザ(語彙サイズ 8192、`kojikojiprg/ai-theories-tokenizer-en`)を使う。

**分割**(文字列の段階で、コーパスの並び順のまま分ける):

| 部分 | 範囲 | 用途 |
|---|---|---|
| 学習用 | 先頭の 92.15% | 学習(ランダムな連続区間のミニバッチ) |
| 較正用の集合 | 続く 2.85%(評価集合を除いた部分の末尾 $3/95$) | 学習率の較正にのみ使う |
| 評価集合 | 末尾の 5%(008 の検証の部分と同じ) | 判定と診断量(途中の評価を含む)に使う |

**学習用の部分の窓**(汎化の差の診断量): 学習用の部分からも、評価集合と同じ作り方(記事の中に収まる長さ 256 の重ならない窓)で
窓を作り、その中から評価集合と **同じ数** の窓を、全体から等間隔に選んで固定する。各学習の最終ステップで、この窓の bits-per-byte を
測り、評価集合の bits-per-byte との差(汎化の差)を診断量にする。学習のミニバッチはランダムな位置から切り出すので、これらの窓と
同じ区切りの系列を学習で見ているとは限らないが、同じテキストは学習で見ている。

**評価集合は学習率の較正に使わない。** 008 は末尾 5% 以外をすべて学習に使ったが、本トピックは学習率を条件ごとに選ぶので、選択に使う
集合を判定に使う集合から分ける。そのため学習用の部分は 008 より約 3% 少ない。

**評価窓**: 評価集合と較正用の集合を、記事ごとの区間(先頭の区間は記事の途中から始まりうる)に分け、区間ごとに別々に符号化する。
各区間の先頭から長さ $S = 256$ の重ならない窓を切り出し、端数は捨てる。**したがってどの窓も 1 つの記事の中に収まり、パディングは
ない。** 記事の境界は Hub の`metadata.json`の`article_offsets`(015 で追加)から得る。

**bits-per-byte**: 各窓 $[w_0, \dots, w_{S-1}]$ の位置 $1, \dots, S - 1$ のトークンの負の対数尤度(位置 0 のトークンは左文脈がないので
予測しない)の和を、予測対象のトークンの UTF-8 バイト数の和と $\ln 2$ で割る。トークナイザはバイトレベル BPE なので、各トークンの
バイト数は語彙の記号の長さで決まる。区間のトークンのバイト数の合計が区間の UTF-8 バイト数に等しいこと、復号が区間の文字列に
一致すること(可逆性)をアサーションで確かめる。全条件で同じ窓・同じ分母を使う。

**トークンの種類**(観察 D、語彙の各トークンを復号した文字列 $t$ から決める。`core`は $t$ から前後の空白・改行を除いたもの):

| 番号 | 種類 | 規則 |
|---|---|---|
| 0 | 数字 | `core`が数字を含み、英字を含まない |
| 1 | 句読点・記号・空白 | `core`が英数字を 1 文字も含まない(空白・改行だけのトークンを含む) |
| 2 | 機能語 | $t$ が空白(改行を含む)で始まり、`core`が英字だけで、小文字にしたものが下のセルの機能語の一覧にある |
| 3 | 内容語の先頭 | $t$ が空白(改行を含む)で始まり、`core`が英字だけで、機能語の一覧にない(単語全体のトークンと、単語の先頭の部分語) |
| 4 | 単語の途中の部分語 | $t$ が空白で始まらず、`core`が英字だけ(空白を伴わない文頭の単語の先頭もここに入る) |
| 5 | その他 | 上のどれでもない(英字と記号・数字の混在、ASCII 以外の文字、単独では復号できないバイト列) |


```python
_t0_data = time.time()
tokenizer, _tokenizer_from_hub = load_bpe_id_tokenizer_from_hub(TOKENIZER_REPO_ID)
assert _tokenizer_from_hub, "トークナイザを Hugging Face Hub から取得できなかった"
assert tokenizer.vocab_size == VOCAB_SIZE

corpus_text, corpus_metadata = load_wikipedia_corpus_with_fallback(
    "en", CORPUS_REPO_ID, WIKIPEDIA_CACHE_DIR, manifest_path=MANIFEST_PATH, return_metadata=True
)
assert len(corpus_text.encode("utf-8")) == corpus_metadata["raw_bytes"], "コーパスの取得が破損している"
_non_validation_text, validation_text = split_train_val_text(corpus_text, VALIDATION_RATIO)
VALIDATION_START = len(_non_validation_text)
train_text, calibration_text = split_train_val_text(_non_validation_text, CALIBRATION_RATIO / (1 - VALIDATION_RATIO))
CALIBRATION_START = len(train_text)
assert train_text + calibration_text + validation_text == corpus_text
del _non_validation_text

article_spans, ARTICLE_SPANS_SOURCE = locate_wikipedia_article_spans(
    corpus_text, "en", WIKIPEDIA_CACHE_DIR, MANIFEST_PATH, repo_id=CORPUS_REPO_ID, start_position=0
)
TOKEN_BYTE_LENGTHS = np.array([len(tokenizer.id_to_symbol[i]) for i in range(VOCAB_SIZE)], dtype=np.int64)


def build_windows(region_start: int, region_end: int) -> dict:
    # 文字位置 [region_start, region_end) を記事ごとの区間に分け、区間ごとに符号化して長さ S の重ならない窓を切り出す。
    segments = [
        (sp["manifest_index"], corpus_text[max(sp["start"], region_start) : min(sp["end"], region_end)])
        for sp in article_spans
        if max(sp["start"], region_start) < min(sp["end"], region_end)
    ]
    windows, articles, rows = [], [], []
    for index, text in segments:
        ids = tokenizer.encode(text)
        assert tokenizer.decode(ids) == text, "符号化が可逆でない"
        assert int(TOKEN_BYTE_LENGTHS[ids].sum()) == len(text.encode("utf-8")), "バイト数の合計が合わない"
        count = len(ids) // SEQUENCE_LENGTH
        for w in range(count):
            windows.append(ids[w * SEQUENCE_LENGTH : (w + 1) * SEQUENCE_LENGTH])
            articles.append(index)
        rows.append((index, len(text), len(ids), count))
    # 区間は領域を覆い、重ならない(区間の文字数の和 + 区切りの改行の数 = 領域の文字数)
    assert sum(len(t) for _, t in segments) + len(segments) - 1 == region_end - region_start, "区間が領域を覆っていない"
    return {"windows": torch.tensor(windows, dtype=torch.long), "articles": articles, "rows": rows}


def select_windows(articles: list[int], per_article: int | None) -> list[int]:
    # None ならすべて。そうでなければ、窓を per_article 個以上持つ記事のそれぞれから先頭の per_article 個
    if per_article is None:
        return list(range(len(articles)))
    by_article: dict[int, list[int]] = {}
    for i, a in enumerate(articles):
        by_article.setdefault(a, []).append(i)
    return sorted(i for indices in by_article.values() if len(indices) >= per_article for i in indices[:per_article])


_evaluation = build_windows(VALIDATION_START, len(corpus_text))
_calibration = build_windows(CALIBRATION_START, VALIDATION_START)
_train_region = build_windows(0, CALIBRATION_START)
_selected = select_windows(_evaluation["articles"], CFG["MAX_WINDOWS_PER_ARTICLE"])
EVAL_WINDOWS = _evaluation["windows"][_selected]
NUM_EVAL_WINDOWS_AVAILABLE = len(_evaluation["windows"])
EVAL_WINDOW_ARTICLES = [_evaluation["articles"][i] for i in _selected]  # 記事を単位とするブートストラップのクラスタ
EVAL_WINDOW_BYTES = TOKEN_BYTE_LENGTHS[EVAL_WINDOWS[:, 1:].numpy()].sum(axis=1)  # 窓ごとの予測対象のバイト数
CALIBRATION_WINDOWS = _calibration["windows"][: CFG["MAX_CALIBRATION_WINDOWS"]]
CALIBRATION_WINDOW_BYTES = TOKEN_BYTE_LENGTHS[CALIBRATION_WINDOWS[:, 1:].numpy()].sum(axis=1)
# 学習用の部分の窓: 評価集合と同じ数を、全体から等間隔に選んで固定する(汎化の差の診断量)
_train_selected = np.linspace(0, len(_train_region["windows"]) - 1, len(EVAL_WINDOWS)).round().astype(int)
assert len(set(_train_selected.tolist())) == len(EVAL_WINDOWS)
TRAIN_PROBE_WINDOWS = _train_region["windows"][_train_selected]
TRAIN_PROBE_WINDOW_BYTES = TOKEN_BYTE_LENGTHS[TRAIN_PROBE_WINDOWS[:, 1:].numpy()].sum(axis=1)
TRAIN_PROBE_ARTICLES = len({_train_region["articles"][i] for i in _train_selected})
assert TRAIN_PROBE_WINDOWS.shape == EVAL_WINDOWS.shape
NUM_EVAL_ARTICLES = len(set(EVAL_WINDOW_ARTICLES))
assert NUM_EVAL_ARTICLES >= 2, "評価窓が 2 本以上の記事から選ばれていない(クラスタブートストラップの経路を通らない)"
assert max(EVAL_WINDOW_ARTICLES.count(a) for a in set(EVAL_WINDOW_ARTICLES)) >= 2  # 複数の窓を持つクラスタを含む
assert len(CALIBRATION_WINDOWS) >= EVAL_BATCH_WINDOWS
EVAL_WINDOWS_HASH = hashlib.sha256(EVAL_WINDOWS.numpy().tobytes()).hexdigest()[:16]

# --- 学習用の部分の符号化(時間のスケーリングを 3 点で計測してから、全体を符号化する) ---
_encode_times = []
for _n in SCALING_ENCODE_CHARACTERS:
    _start = time.time()
    tokenizer.encode(train_text[:_n])
    _encode_times.append(time.time() - _start)
_encode_fit = fit_power_law_exponent(SCALING_ENCODE_CHARACTERS, _encode_times)
_encode_extrapolated = _encode_times[-1] * (len(train_text) / SCALING_ENCODE_CHARACTERS[-1]) ** _encode_fit.exponent
_encode_proportional = _encode_times[-1] * len(train_text) / SCALING_ENCODE_CHARACTERS[-1]
_start = time.time()
TRAIN_IDS = encode_corpus(tokenizer, train_text)
ENCODE_SECONDS = time.time() - _start
assert int(TRAIN_IDS.max()) < VOCAB_SIZE and TRAIN_IDS.dtype == torch.long
_sample = train_text[:50_000]
assert tokenizer.decode(tokenizer.encode(_sample)) == _sample, "トークナイザのラウンドトリップが一致しない"
TRAIN_TOKENS = len(TRAIN_IDS)
UNIFORM_LOSS = math.log(VOCAB_SIZE)  # 一様分布に相当する損失 ln(V)

# --- トークンの種類(観察 D) ---
FUNCTION_WORDS = frozenset(
    "the of and in to a an is was were are be been by for with as on at from that this these those it its he she they his her "
    "their which who whom or but not had has have also can could would should may might will than then there such into over "
    "after before between during under about through when where while if so no nor both each other some any all more most".split()
)
TOKEN_CATEGORIES = ("digit", "punctuation", "function", "content-start", "subword", "other")
TOKEN_CATEGORY_LABELS = ("数字", "句読点・記号・空白", "機能語", "内容語の先頭", "単語の途中の部分語", "その他")


def categorize_token(symbol: str) -> int:
    text = try_decode_byte_level_symbol(symbol)
    if text is None:
        return 5
    core = text.strip()
    if not any(ch.isalnum() for ch in core):
        return 1
    has_letter = any(ch.isalpha() for ch in core)
    if any(ch.isdigit() for ch in core) and not has_letter:
        return 0
    if core.isascii() and core.isalpha():
        if not text[0].isspace():
            return 4
        return 2 if core.lower() in FUNCTION_WORDS else 3
    return 5


TOKEN_CATEGORY = np.array([categorize_token(tokenizer.id_to_symbol[i]) for i in range(VOCAB_SIZE)])
_category_share = np.bincount(TOKEN_CATEGORY[EVAL_WINDOWS.numpy().reshape(-1)], minlength=len(TOKEN_CATEGORIES))
DATA_SECONDS = time.time() - _t0_data

print(
    f"コーパス: {len(corpus_text):,} 文字(取得元 {corpus_metadata['source']})。学習用 {len(train_text):,} 文字、較正用の集合 "
    f"{len(calibration_text):,} 文字(文字位置 {CALIBRATION_START:,} から)、評価集合 {len(validation_text):,} 文字(文字位置 {VALIDATION_START:,} から)"
)
print(f"記事の境界の取得元: {ARTICLE_SPANS_SOURCE}、記事 {len(article_spans)} 件")
print(
    f"符号化の時間: {dict(zip(SCALING_ENCODE_CHARACTERS, rounded(_encode_times, 3), strict=True))} 秒、べき指数 b = {_encode_fit.exponent:.3f}、"
    f"学習用の全体({len(train_text):,} 文字)への外挿: べき乗則 {_encode_extrapolated:.1f} 秒・比例 {_encode_proportional:.1f} 秒、"
    f"実測 {ENCODE_SECONDS:.1f} 秒(テンソルへの変換を含む。1 回だけ行い、全条件で共有する)"
)
print(f"学習用のトークン数 {TRAIN_TOKENS:,}(1 ステップ {BATCH_SIZE * SEQUENCE_LENGTH:,} トークン)、ln(V) = {UNIFORM_LOSS:.4f}")
for _t in STEP_CANDIDATES:
    print(f"  T = {_t}: 見るトークン数 {_t * BATCH_SIZE * SEQUENCE_LENGTH:,}(学習用の {_t * BATCH_SIZE * SEQUENCE_LENGTH / TRAIN_TOKENS:.2f} エポック相当)")
print(
    f"評価集合: 作れる窓 {len(_evaluation['windows'])} 個のうち {len(EVAL_WINDOWS)} 個を使う(水準 {CURRENT_LEVEL_NAME!r})、記事 {NUM_EVAL_ARTICLES} 本、"
    f"記事ごとの窓の数 {dict(sorted({a: EVAL_WINDOW_ARTICLES.count(a) for a in set(EVAL_WINDOW_ARTICLES)}.items()))}、"
    f"予測対象のバイト数 {int(EVAL_WINDOW_BYTES.sum()):,}、ハッシュ {EVAL_WINDOWS_HASH}"
)
print(
    f"較正用の集合: 作れる窓 {len(_calibration['windows'])} 個のうち {len(CALIBRATION_WINDOWS)} 個を使う、"
    f"予測対象のバイト数 {int(CALIBRATION_WINDOW_BYTES.sum()):,}"
)
print(
    f"学習用の部分の窓(汎化の差の診断量): 作れる窓 {len(_train_region['windows']):,} 個から等間隔に {len(TRAIN_PROBE_WINDOWS)} 個"
    f"(記事 {TRAIN_PROBE_ARTICLES} 本)、予測対象のバイト数 {int(TRAIN_PROBE_WINDOW_BYTES.sum()):,}"
)
print(
    "トークンの種類(語彙の中の数 / 評価窓のトークンに占める割合): "
    + "、".join(
        f"{label} {int((TOKEN_CATEGORY == i).sum())} / {_category_share[i] / _category_share.sum():.3f}"
        for i, label in enumerate(TOKEN_CATEGORY_LABELS)
    )
)
print(f"データの準備 {DATA_SECONDS:.1f} 秒")
```


    tokenizer.json:   0%|          | 0.00/661k [00:00<?, ?B/s]



    corpus.txt: reconstructing file:   0%|          |  0.00B / 24.3MB            



    corpus.txt: downloading bytes:           |  0.00B            



    metadata.json:   0%|          | 0.00/43.4k [00:00<?, ?B/s]


    コーパス取得元: kojikojiprg/ai-theories-corpus-en-pretraining(Hugging Face Hub)
    コーパス: 24,214,546 文字(取得元 hub)。学習用 22,277,383 文字、較正用の集合 726,436 文字(文字位置 22,277,383 から)、評価集合 1,210,727 文字(文字位置 23,003,819 から)
    記事の境界の取得元: metadata、記事 356 件
    符号化の時間: {1000000: 0.148, 2000000: 0.313, 4000000: 0.607} 秒、べき指数 b = 1.019、学習用の全体(22,277,383 文字)への外挿: べき乗則 3.5 秒・比例 3.4 秒、実測 9.4 秒(テンソルへの変換を含む。1 回だけ行い、全条件で共有する)
    学習用のトークン数 5,771,682(1 ステップ 8,192 トークン)、ln(V) = 9.0109
      T = 2181: 見るトークン数 17,866,752(学習用の 3.10 エポック相当)
      T = 1090: 見るトークン数 8,929,280(学習用の 1.55 エポック相当)
      T = 545: 見るトークン数 4,464,640(学習用の 0.77 エポック相当)
    評価集合: 作れる窓 1228 個のうち 1228 個を使う(水準 'prod')、記事 17 本、記事ごとの窓の数 {338: 51, 339: 58, 340: 150, 341: 47, 342: 128, 343: 92, 344: 2, 346: 18, 347: 66, 348: 1, 349: 91, 350: 1, 351: 133, 352: 7, 353: 34, 354: 2, 355: 347}、予測対象のバイト数 1,198,499、ハッシュ 6d965eb3ce29c97a
    較正用の集合: 作れる窓 714 個のうち 714 個を使う、予測対象のバイト数 716,250
    学習用の部分の窓(汎化の差の診断量): 作れる窓 22,384 個から等間隔に 1228 個(記事 210 本)、予測対象のバイト数 1,212,869
    トークンの種類(語彙の中の数 / 評価窓のトークンに占める割合): 数字 571 / 0.042、句読点・記号・空白 144 / 0.043、機能語 151 / 0.215、内容語の先頭 4019 / 0.335、単語の途中の部分語 2198 / 0.311、その他 1109 / 0.053
    データの準備 43.9 秒




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/06_architectures/022_mixture_of_experts.ipynb
