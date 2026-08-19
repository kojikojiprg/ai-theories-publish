---
title: "学習の安定化(Training Stabilization)(実装・実験編 1/3)"
---

この記事は後編(実装・実験編 1/3)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/007_training_stabilization-theory)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/007_training_stabilization-practice-2)。

## 4. 実装方針 / Implementation Policy

001〜006 でスクラッチ実装した部品(`GPTLanguageModel`・`DecoderBlock`・RoPE・RMSNorm・SwiGLU・`train_language_model`)を再利用し、007 で新規に追加するのは「AdamW・学習率スケジュール・gradient clipping と、それらを既存の学習ループへ注入する仕組み」のみである(段階 1 で `src/`に追加済み)。

| 機能 | 配置場所 | 実装/委譲 |
|---|---|---|
| AdamW・Adam(L2 正則化混入、比較対象)(`AdamW`・`AdamWithL2Regularization`) | `src/training/optimizer.py` | スクラッチ実装(新規、段階 1。`torch.optim.AdamW`との数値一致を検証済み、6.3 節) |
| warmup + cosine スケジュール(`compute_warmup_cosine_learning_rate`) | `src/training/schedule.py` | スクラッチ実装(新規、段階 1) |
| 学習ループへの optimizer・スケジュール・gradient clipping の注入(`train_language_model`の `optimizer`・`learning_rate_schedule`・`gradient_clip_threshold` 引数、`loss_step_delta`・`learning_rate`の記録) | `src/training/trainer.py` | 拡張(段階 1、006 との後方互換性を検証済み) |
| 正規化前置 / 正規化後置の切り替え(`GPTLanguageModel`の `norm_first` 引数) | `src/models/gpt.py` | 拡張(段階 1。`DecoderBlock`は 004 の時点で `norm_first`を持つため、`GPTLanguageModel` からの透過のみ追加。非埋め込みパラメータ数が正規化前置 / 正規化後置で一致することを検証済み) |
| 診断量(勾配ノルムのピーク / 平均比率・最大単一ステップ損失上昇幅・実効減衰強度の乖離) | `src/utils/statistics.py` | スクラッチ実装(新規、段階 1) |
| 勾配ノルムの時系列 + クリッピング閾値の可視化(`plot_gradient_norm_trace`) | `src/utils/visualization.py` | スクラッチ実装(新規、段階 1) |
| 条件グリッド構築・判定関数・較正セル | 本ノートブック(7 節) | ノートブック内に直接記述(007 固有の実験設計であり、再利用価値のある抽象化を要さないため) |

**実行環境についての注記**: 5〜7 節のコードは、ローカル(Apple Silicon、MPS または CPU)での縮小スケール実行(スモークテスト)と Google Colab T4 GPU での本番スケール実行の両方に対応する設計になっている。**どちらの実行段階であるかは、5.2 節の `SMOKE_TEST` の値(`True` ならスモークテスト、`False` なら本番)で判別できる**(この対応は本節で一度だけ説明し、以降の各節では実行段階を断定しない)。縮小の内容は 5.2 節で具体的な値とともに明記する。


## 5. 実装 / Implementation



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
    remote: Enumerating objects: 445, done.[K
    remote: Counting objects: 100% (445/445), done.[K
    remote: Compressing objects: 100% (289/289), done.[K
    remote: Total 445 (delta 224), reused 332 (delta 137), pack-reused 0 (from 0)[K
    Receiving objects: 100% (445/445), 5.51 MiB | 5.94 MiB/s, done.
    Resolving deltas: 100% (224/224), done.
    /content/ai-theories
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m23.7/23.7 MB[0m [31m86.9 MB/s[0m eta [36m0:00:00[0m
    [?25h[2mUsing Python 3.12.13 environment at: /usr[0m
    [2K[2mResolved [1m41 packages[0m [2min 493ms[0m[0m
    [2K[2mPrepared [1m25 packages[0m [2min 48.09s[0m[0m
    [2mUninstalled [1m11 packages[0m [2min 682ms[0m[0m
    [2K[2mInstalled [1m25 packages[0m [2min 281ms[0m[0m
     [31m-[39m [1mcuda-bindings[0m[2m==12.9.7[0m
     [32m+[39m [1mcuda-bindings[0m[2m==13.3.1[0m
     [31m-[39m [1mcuda-toolkit[0m[2m==12.8.1[0m
     [32m+[39m [1mcuda-toolkit[0m[2m==13.0.3.0[0m
     [31m-[39m [1mfilelock[0m[2m==3.32.3[0m
     [32m+[39m [1mfilelock[0m[2m==3.32.2[0m
     [31m-[39m [1mfsspec[0m[2m==2025.3.0[0m
     [32m+[39m [1mfsspec[0m[2m==2026.7.0[0m
     [31m-[39m [1mmatplotlib[0m[2m==3.10.0[0m
     [32m+[39m [1mmatplotlib[0m[2m==3.11.1[0m
     [31m-[39m [1mnumpy[0m[2m==2.0.2[0m
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
     [31m-[39m [1msetuptools[0m[2m==75.2.0[0m
     [32m+[39m [1msetuptools[0m[2m==84.0.0[0m
     [31m-[39m [1mtorch[0m[2m==2.11.0+cu128[0m
     [32m+[39m [1mtorch[0m[2m==2.13.0[0m
     [31m-[39m [1mtriton[0m[2m==3.6.0[0m
     [32m+[39m [1mtriton[0m[2m==3.7.1[0m



```python
import functools
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.text import (
    CharacterLevelTokenizer,
    encode_corpus,
    get_random_batch,
    load_tiny_shakespeare,
    make_evaluation_windows,
    split_train_val_text,
)
from src.layers.feedforward import SwiGLUFeedForwardNetwork
from src.layers.normalization import RMSNorm
from src.layers.positional_encoding import RotaryPositionEmbedding
from src.models.gpt import GPTLanguageModel
from src.training.optimizer import AdamW, AdamWithL2Regularization
from src.training.schedule import compute_warmup_cosine_learning_rate
from src.training.trainer import train_language_model
from src.utils.statistics import (
    compute_effective_decay_divergence,
    compute_gradient_norm_peak_to_mean_ratio,
    compute_loss_step_delta_std,
    compute_max_single_step_loss_increase,
    count_non_embedding_parameters,
)
from src.utils.visualization import (
    plot_gradient_norm_trace,
    plot_learning_curves_multi_seed,
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

```

    torch: 2.13.0+cu130 / device: cuda


### 5.1 AdamW スクラッチ実装の数値検証(本番実験に使う前の確認)

3.2 節で導出した AdamW を `torch.optim.AdamW`と比較し、数値的に一致することを確認する。あわせて、`AdamWithL2Regularization`(比較対象)が `weight_decay=0`のとき `AdamW`と一致すること(重み減衰が無ければ両者はバニラ Adam に帰着するはず)も確認する。



```python
def _check_adamw_matches_torch(num_steps: int = 50, lr: float = 1e-2, weight_decay: float = 0.1) -> float:
    torch.manual_seed(0)
    ref_param = torch.nn.Parameter(torch.randn(20, 7))
    my_param = torch.nn.Parameter(ref_param.detach().clone())

    ref_opt = torch.optim.AdamW([ref_param], lr=lr, weight_decay=weight_decay)
    my_opt = AdamW([my_param], lr=lr, weight_decay=weight_decay)

    for _ in range(num_steps):
        grad = torch.randn(20, 7)
        ref_param.grad = grad.clone()
        my_param.grad = grad.clone()
        ref_opt.step()
        my_opt.step()

    return (ref_param - my_param).abs().max().item()


max_diff = _check_adamw_matches_torch()
print(f"AdamW(スクラッチ) vs torch.optim.AdamW、{50} ステップ後の最大絶対差: {max_diff:.3e}")
assert max_diff < 1e-5, "AdamW のスクラッチ実装が torch.optim.AdamW と一致しない"
print("OK: AdamW は torch.optim.AdamW と数値的に一致する(float32 精度の範囲内)")


def _check_wd_zero_reduces_to_adam(num_steps: int = 20, lr: float = 1e-2) -> tuple[float, float]:
    torch.manual_seed(1)
    p_adamw = torch.nn.Parameter(torch.randn(4, 4))
    p_l2 = torch.nn.Parameter(p_adamw.detach().clone())
    p_ref = torch.nn.Parameter(p_adamw.detach().clone())

    opt_adamw = AdamW([p_adamw], lr=lr, weight_decay=0.0)
    opt_l2 = AdamWithL2Regularization([p_l2], lr=lr, weight_decay=0.0)
    opt_ref = torch.optim.Adam([p_ref], lr=lr)

    for _ in range(num_steps):
        g = torch.randn(4, 4)
        p_adamw.grad, p_l2.grad, p_ref.grad = g.clone(), g.clone(), g.clone()
        opt_adamw.step()
        opt_l2.step()
        opt_ref.step()

    return (
        (p_adamw - p_ref).abs().max().item(),
        (p_l2 - p_ref).abs().max().item(),
    )


diff_adamw, diff_l2 = _check_wd_zero_reduces_to_adam()
print(f"weight_decay=0 のとき: AdamW vs torch.optim.Adam の最大差 {diff_adamw:.3e}、"
      f"AdamWithL2Regularization vs torch.optim.Adam の最大差 {diff_l2:.3e}")
assert diff_adamw < 1e-5 and diff_l2 < 1e-5
print("OK: weight_decay=0 のとき両クラスともバニラ Adam に一致する")

```

    AdamW(スクラッチ) vs torch.optim.AdamW、50 ステップ後の最大絶対差: 2.384e-07
    OK: AdamW は torch.optim.AdamW と数値的に一致する(float32 精度の範囲内)
    weight_decay=0 のとき: AdamW vs torch.optim.Adam の最大差 1.192e-07、AdamWithL2Regularization vs torch.optim.Adam の最大差 1.192e-07
    OK: weight_decay=0 のとき両クラスともバニラ Adam に一致する


### 5.2 スケールの設定

**`SMOKE_TEST`(下記コードセル)が実行段階を切り替える**: `True` はローカル(Apple Silicon、MPS または CPU)での縮小スケール実行(スモークテスト)、`False` は Google Colab T4 GPU での本番スケール実行を意味する。以降の各節では、このセルの `SMOKE_TEST` の出力値でどちらの実行段階かを判別できるため、個々の節で「スモークテストによるものである」とは断定しない。006 と同じ方針(`SMOKE_TEST` 分岐)を踏襲する。model サイズ・系列長・バッチサイズは 006 の本番設定(`D_MODEL=256` など)をそのまま引き継ぐ(006 で Google Colab T4 上での実行時間・安定性が確認済みの構成であるため、007 で改めて model サイズを選定し直す必要はない)。007 固有の新しい定数(学習率倍率・clip 係数・warmup 比率など)は較正セル(6.1 節)で使う値をここで宣言する。

**`CLIP_QUANTILE`(gradient clipping 閾値の分位点)・`PRECONDITION_LOSS_RATIO`・`PRECONDITION_CLIP_RATE_MIN`(前提条件 P1・P2 の採用閾値)を較正前に宣言する理由**: これらの値は、較正の実測値を見てから決めると結果に迎合した値になりうる。そのため、いずれも較正実行の **前** にコード上の定数として固定する。gradient clipping 閾値の決め方は、当初「平均 $\mu_g$ + $k\sigma_g$」(旧 `CLIP_K`)方式だったが、1 回目の本番実行(6.10.1 節)で勾配ノルムの分布が右に長く歪んでいる場合に閾値が高くなりすぎ、gradient clipping がほぼ発動しない(前提条件 P2 の不成立)ことが判明したため、**勾配ノルムの `CLIP_QUANTILE` 分位点** を閾値とする方式に変更した(3.4 節)。



```python
SMOKE_TEST = False  # Claude Code はこの True 側のみ実行する(Colab T4 では False に切り替える)

# --- 較正に使う定数(較正の実測値を見る前に宣言する) ---
BASE_LEARNING_RATE = 3e-4  # 006 の本番学習率(006 5.1 節、LEARNING_RATE)
# 006 基準学習率に対する倍率(等比刻み)。旧実行(x2・x5・x10)は全倍率で正規化後置が
# 崩壊したため、より低い倍率を等比刻みを保ったまま追加した(
# 判定基準ではなく較正の手続きのみを修正する)。
LR_CALIBRATION_MULTIPLIERS = (1.25, 2.5, 5.0, 10.0)
# gradient clipping 閾値 = 採用した学習率での勾配ノルムの CLIP_QUANTILE 分位点
# (旧: 平均 + kσ 方式。分布が右に長く歪んでいる場合に閾値が高くなりすぎ、本番で
# ほぼ発動しなかったため分位点方式に変更した。3.4 節参照)。
#
# 値の根拠(0.85、6.10.2 節の 2 回目の本番実行の照合により当初の 0.90 から修正): 2 回目の
# 本番実行では、CLIP_QUANTILE=0.90 で決定した閾値(較正実行の勾配ノルムの 90% 分位点)
# により Q4_clip の発動比率が 0.0433〜0.0567 となり、前提条件 P2 の下限
# PRECONDITION_CLIP_RATE_MIN=0.05(6.2 節、変更しない)を 5 シード中 2 シードで下回った
# (gradient clipping を有効にすると勾配ノルムの分布そのものが変化するため、較正実行
# 時点の分布から見積もった発動比率が本番の Q4_clip でそのまま再現されるとは限らない)。
# 分位点を下げると閾値も下がり発動比率が上がるため、0.85 に引き下げる。0.85 は、
# Q4_clip の発動比率を前提条件 P2 の下限に対して余裕のある水準に引き上げつつ、
# clipping が常時発動して実質的な学習率スケーリングに変質してしまう水準(分位点を
# 極端に下げた場合)を避けるための値である。**この修正の根拠は「clipping の発動比率が
# 宣言した前提条件の下限に届かなかったから」という、gradient clipping が実際に
# 作動しているかどうかという検証したい仮説とは独立な量に基づくものであり、
# 「どの安定化技術に効果が出たか」という観測結果の方向には依存しない**
# 前提条件 P2 の閾値
# PRECONDITION_CLIP_RATE_MIN=0.05 自体は変更しない(前提条件は本番実行の前に宣言する
# ものであり、結果を見てから緩めるのではなく、前提が成立するよう実験条件のみを
# 修正するのが規約の趣旨である)。
CLIP_QUANTILE = 0.85
# 前提条件 P1(学習の進行、6.2 節)の採用閾値。
# 最終訓練損失が一様分布相当の損失 ln(V) のこの倍率以下であることを要求する。
# 検証したい仮説(どの安定化技術が効くか)とは独立な量として、較正の実測値を
# 見る前にここで宣言する。
#
# 値の根拠(0.60、6.10.1 節の照合により当初の 0.85 から修正): 1 回目の本番実行(6.10.1 節)を
# この閾値の候補と照合すると、正常に学習が進んだ 6 条件(Q1・Q2・Q3)の最終訓練損失は
# 1.5062〜1.9656(ln(V) に対する比で 0.36〜0.47)、学習が崩壊した Q4 の 4 水準は
# 3.2983〜3.3016(同 0.79)であった。当初の 0.85 では崩壊値(比 0.79)がすべて閾値を
# 下回ってしまい、旧実行と同一の崩壊が再現されても P1 が「成立」と判定されてしまう
# (この照合により発覚した)。0.60(閾値 ln(V) x 0.60 = 2.5046)は、正常条件の最悪値
# 1.9656 から 27% 上、崩壊値 3.30 から 24% 下に位置し、両側に余裕がある。この値の
# 決定は「学習が実際に進んでいるかどうか」という検証したい仮説とは独立な量に基づく
# ものであり、「どの安定化技術に効果が出たか」という観測結果の方向には依存しない
PRECONDITION_LOSS_RATIO = 0.60
# 前提条件 P2(gradient clipping の発動、6.2 節)の採用閾値。clipping を有効にした
# 条件で、発動したステップの比率がこの値以上であることを要求する。
PRECONDITION_CLIP_RATE_MIN = 0.05
WARMUP_RATIO = 0.1  # warmup ステップ数 = 全体ステップ数の 10%
MIN_LEARNING_RATE_RATIO = 0.01  # cosine decay の下限学習率 = peak_lr * この比率
WEIGHT_DECAY = 0.1  # AdamW / Adam(L2) の名目上の重み減衰係数(3.2 節)
SESSION_BUDGET_SECONDS = 2 * 60 * 60  # Google Colab 1 セッションの目安予算(006 5.1 節と同じ基準)

if SMOKE_TEST:
    # --- コーパス ---
    VALIDATION_RATIO = 0.1

    # --- model(006 のスモークテスト設定を踏襲) ---
    D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF = 64, 2, 4, 256
    SEQUENCE_LENGTH = 64
    DROPOUT = 0.0

    # --- 学習 ---
    BATCH_SIZE = 8
    NUM_STEPS = 40  # 主実験(10 条件)の学習ステップ数
    EVAL_INTERVAL = 20

    # --- 主張 6(合成タスク) ---
    SYNTH_STEPS = 60
    SYNTH_SEEDS = 5

    # --- シード数(較正セルで決定するまでの初期値。7.1 節で上書きする) ---
    SEEDS_MAIN_AXIS = 3  # Q4 の 4 条件
    SEEDS_OTHER = 2  # Q1〜Q3 の 6 条件
else:
    # --- コーパス ---
    VALIDATION_RATIO = 0.05

    # --- model(006 の本番設定、006 5.1 節) ---
    D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF = 256, 4, 8, 1024
    SEQUENCE_LENGTH = 256
    DROPOUT = 0.0

    # --- 学習 ---
    BATCH_SIZE = 32
    NUM_STEPS = 300
    EVAL_INTERVAL = 50

    # --- 主張 6(合成タスク) ---
    SYNTH_STEPS = 300
    SYNTH_SEEDS = 10

    # --- シード数(較正セルで決定するまでの初期値。7.1 節で上書きする) ---
    SEEDS_MAIN_AXIS = 5
    SEEDS_OTHER = 3

SWIGLU_D_FF = round((2 / 3) * D_FF)  # 004: 標準の順伝播ネットワークとパラメータ数を揃えるための丸め
WARMUP_STEPS = max(1, round(WARMUP_RATIO * NUM_STEPS))
# 較正のスケールが本番より小さいと本番でのみ現れる崩壊を検出できない
# ため、較正用ステップ数を
# 本番の NUM_STEPS と完全に同一にする(旧実行では CALIBRATION_STEPS=60 <
# NUM_STEPS=300 であったため、60 ステップでは崩壊が現れなかった)。
CALIBRATION_STEPS = NUM_STEPS
# 前提条件 P1 の判定に使う量(6.5 節末尾・6.1 節)を、最終ステップ単独ではなく直近
# P1_LOSS_WINDOW_STEPS 個のステップの訓練損失の平均にする(バッチのばらつきにより
# 閾値付近で判定が偶然左右されることを避けるため)。本番設定(NUM_STEPS=300)で 20
# ステップに相当する比率(1/15)をスモークテストにも同じ比率で適用する。
# **主張 5(固定ステップ数終了時点の損失値、判定基準)は引き続き train_loss[-1] を使い、
# この量とは別である**(6.2 節・6.5 節末尾に明記)。
P1_LOSS_WINDOW_STEPS = max(1, NUM_STEPS // 15)

print(f"SMOKE_TEST={SMOKE_TEST}")
print(f"D_MODEL={D_MODEL}, NUM_LAYERS={NUM_LAYERS}, NUM_HEADS={NUM_HEADS}, D_FF={D_FF}(SwiGLU: {SWIGLU_D_FF})")
print(f"SEQUENCE_LENGTH={SEQUENCE_LENGTH}, BATCH_SIZE={BATCH_SIZE}")
print(f"NUM_STEPS={NUM_STEPS}, CALIBRATION_STEPS={CALIBRATION_STEPS}, WARMUP_STEPS={WARMUP_STEPS}, EVAL_INTERVAL={EVAL_INTERVAL}")
print(f"BASE_LEARNING_RATE={BASE_LEARNING_RATE}, LR_CALIBRATION_MULTIPLIERS={LR_CALIBRATION_MULTIPLIERS}")
print(f"CLIP_QUANTILE={CLIP_QUANTILE}, WEIGHT_DECAY={WEIGHT_DECAY}")
print(f"PRECONDITION_LOSS_RATIO={PRECONDITION_LOSS_RATIO}, PRECONDITION_CLIP_RATE_MIN={PRECONDITION_CLIP_RATE_MIN}")
print(f"P1_LOSS_WINDOW_STEPS={P1_LOSS_WINDOW_STEPS}")

# 主張3'(6.2.1 節)専用の新規シード。既存の条件グリッド(6.5 節、Q4_none・Q4_clip は
# シード 0 .. SEEDS_MAIN_AXIS-1 を使う)と重複しないことを保証するため、SEEDS_MAIN_AXIS を
# 起点にオフセットする(既存シードの数値を判定基準の設定に使わないという事前登録の要請、
# 6.2.1 節)。SMOKE_TEST の値に応じて SEEDS_MAIN_AXIS 自体が変わるため、この構成は
# スモークテスト・本番のどちらでも既存シードと重複しない。
SEEDS_NEW_CLAIMS = list(range(SEEDS_MAIN_AXIS, 2 * SEEDS_MAIN_AXIS))
print(f"SEEDS_NEW_CLAIMS={SEEDS_NEW_CLAIMS}(既存シード 0..{SEEDS_MAIN_AXIS - 1} とは重複しない)")

```

    SMOKE_TEST=False
    D_MODEL=256, NUM_LAYERS=4, NUM_HEADS=8, D_FF=1024(SwiGLU: 683)
    SEQUENCE_LENGTH=256, BATCH_SIZE=32
    NUM_STEPS=300, CALIBRATION_STEPS=300, WARMUP_STEPS=30, EVAL_INTERVAL=50
    BASE_LEARNING_RATE=0.0003, LR_CALIBRATION_MULTIPLIERS=(1.25, 2.5, 5.0, 10.0)
    CLIP_QUANTILE=0.85, WEIGHT_DECAY=0.1
    PRECONDITION_LOSS_RATIO=0.6, PRECONDITION_CLIP_RATE_MIN=0.05
    P1_LOSS_WINDOW_STEPS=20
    SEEDS_NEW_CLAIMS=[5, 6, 7, 8, 9](既存シード 0..4 とは重複しない)


### 5.3 コーパスの取得

007 の独立変数は正規化方式・学習率・安定化技術であり、006 のようにトークナイザ条件を比較するわけではない。したがって単一のトークナイザ・単一のコーパスで十分であり、004 と同じ構成(Tiny Shakespeare + 文字レベルトークナイザ)を採用する(004 も「条件比較のための最小限の文字レベル言語モデリング」という同じ動機でこの構成を使った)。文字レベルトークナイザの語彙は全文(学習用・検証用の連結)から構築し、検証テキストに学習テキストにない文字が出現して未知文字エラーになることを避ける(006 のようなトークナイザ条件間比較を行わないため、006 のような分割前後の厳密な区別は本トピックでは不要)。



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
print(f"train_ids: {len(train_ids):,} トークン / val windows: {tuple(eval_windows.shape)} / "
      f"total_eval_bytes={total_eval_bytes:,}")

# ラウンドトリップの完全一致の検証(006 の方針を踏襲)。
assert tokenizer.decode(tokenizer.encode(raw_text[:500])) == raw_text[:500]
print("encode -> decode の往復確認: OK")

```

    テキスト長: 1,115,394 文字 / 語彙サイズ V=65
    train_ids: 1,059,625 トークン / val windows: (218, 256) / total_eval_bytes=55,769
    encode -> decode の往復確認: OK


**前提条件 P1 の閾値をここで一度だけ定義する理由**: `PRECONDITION_LOSS_THRESHOLD`(前提条件 P1 の判定閾値、`ln(V) x PRECONDITION_LOSS_RATIO`)は語彙サイズ `VOCAB_SIZE` に依存するため、5.2 節(定数宣言セル)では計算できない(`VOCAB_SIZE` は本セルで初めて確定する)。以前は 6.1 節(較正セル)・6.5 節末尾(前提条件確認セル)の 2 箇所で同じ式を重複して定義していたが、片方だけを変更したときに不整合が生じうるため、ここで一度だけ定義し、両方の箇所からこの変数を参照する形に統一する。**`PRECONDITION_LOSS_RATIO` 自体は 5.2 節で較正の実測値を見る前に宣言済みであり、定義位置がここになるのは `VOCAB_SIZE` の確定を待つ必要があるという実装上の都合のみで、値そのものは較正前に定まっている。**


```python
# 前提条件 P1(学習の進行、6.2 節)の判定閾値。VOCAB_SIZE が確定した直後に一度だけ
# 定義し、6.1 節(較正セル)・6.5 節末尾(前提条件確認セル)の両方からこの変数を
# 参照する(定義の重複による不整合を避ける)。
PRECONDITION_LOSS_THRESHOLD = float(np.log(VOCAB_SIZE) * PRECONDITION_LOSS_RATIO)
print(f"前提条件 P1 の閾値: ln(V) x PRECONDITION_LOSS_RATIO = "
      f"{np.log(VOCAB_SIZE):.4f} x {PRECONDITION_LOSS_RATIO} = {PRECONDITION_LOSS_THRESHOLD:.4f}")

```

    前提条件 P1 の閾値: ln(V) x PRECONDITION_LOSS_RATIO = 4.1744 x 0.6 = 2.5046


### 5.4 model の構築ヘルパーと `norm_first`の非埋め込みパラメータ数の一致確認

`GPTLanguageModel`は 006 と同じく RoPE(位置エンコーディング)・RMSNorm(正規化)・SwiGLU(順伝播ネットワーク)を注入する。`norm_first`のみが 007 で新たに条件間で振る変数である。まず、正規化前置・正規化後置で非埋め込みパラメータ数が完全に一致することをアサーションで確認する(不変条件、006 3.3 節の考え方を踏襲。語彙サイズは全条件で共通(単一トークナイザ)だが、`norm_first`の違いがパラメータ数に影響しないことも別途保証しておく必要がある)。



```python
def build_model(norm_first: bool, seed: int) -> GPTLanguageModel:
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
        norm_first=norm_first,
    )


_model_pre = build_model(norm_first=True, seed=0)
_model_post = build_model(norm_first=False, seed=0)
_n_pre = count_non_embedding_parameters(_model_pre)
_n_post = count_non_embedding_parameters(_model_post)
print(f"非埋め込みパラメータ数: 正規化前置={_n_pre:,} / 正規化後置={_n_post:,}")
assert _n_pre == _n_post, "正規化前置・正規化後置で非埋め込みパラメータ数が一致しない"
print("OK: 正規化前置・正規化後置で非埋め込みパラメータ数が完全に一致する")
del _model_pre, _model_post

```

    非埋め込みパラメータ数: 正規化前置=3,149,056 / 正規化後置=3,149,056
    OK: 正規化前置・正規化後置で非埋め込みパラメータ数が完全に一致する


### 5.5 optimizer・スケジュール・gradient clipping の注入ヘルパー

安定化技術の水準(`"none"` / `"warmup_cosine"` / `"clip"` / `"all"`)から、`train_language_model`にそのまま渡せる `(optimizer, learning_rate_schedule, gradient_clip_threshold, learning_rate)`の組を構築する。

- `"all"`(全部乗せ)のみ `AdamW`(重み減衰込み)を使う。それ以外(`"none"`・`"warmup_cosine"` 単独・`"clip"` 単独)は `optimizer=None`を渡し、`train_language_model`の既定である `torch.optim.Adam`(重み減衰なし)を使う。**AdamW 単独の水準を設けない理由は 3.5 節で述べた通り**(AdamW の効果は主張 6 の合成タスクで直接検証するため)。
- `"warmup_cosine"`・`"all"`は `learning_rate_schedule`を渡す(peak を `peak_lr`とする warmup + cosine)。それ以外は固定学習率(`learning_rate=peak_lr`)。
- `"clip"`・`"all"`は `gradient_clip_threshold`を渡す。



```python
def build_optimizer_config(
    model: GPTLanguageModel,
    stabilization_level: str,
    peak_lr: float,
    clip_threshold: float,
):
    """stabilization_level: "none" | "warmup_cosine" | "clip" | "all" """
    use_adamw = stabilization_level == "all"
    use_schedule = stabilization_level in ("warmup_cosine", "all")
    use_clip = stabilization_level in ("clip", "all")

    optimizer = AdamW(model.parameters(), lr=peak_lr, weight_decay=WEIGHT_DECAY) if use_adamw else None
    schedule = (
        functools.partial(
            compute_warmup_cosine_learning_rate,
            warmup_steps=WARMUP_STEPS,
            total_steps=NUM_STEPS,
            peak_learning_rate=peak_lr,
            min_learning_rate=peak_lr * MIN_LEARNING_RATE_RATIO,
        )
        if use_schedule
        else None
    )
    clip = clip_threshold if use_clip else None
    return {
        "optimizer": optimizer,
        "learning_rate_schedule": schedule,
        "gradient_clip_threshold": clip,
        "learning_rate": peak_lr,
    }


# 動作確認(較正が終わる前の仮の clip_threshold で、4 水準すべてが正しく構築されることのみ確認する)
for _level in ("none", "warmup_cosine", "clip", "all"):
    _cfg = build_optimizer_config(build_model(True, 0), _level, BASE_LEARNING_RATE, 1.0)
    print(_level, "-> optimizer:", type(_cfg["optimizer"]).__name__,
          "schedule:", _cfg["learning_rate_schedule"] is not None,
          "clip:", _cfg["gradient_clip_threshold"])

```

    none -> optimizer: NoneType schedule: False clip: None
    warmup_cosine -> optimizer: NoneType schedule: True clip: None
    clip -> optimizer: NoneType schedule: False clip: 1.0
    all -> optimizer: AdamW schedule: True clip: 1.0


## 6. 実験 / Experiments

**実行環境についての注記**: 5.2 節で述べた通り、以下の実験セルの出力がスモークテストと本番実行のどちらによるものかは `SMOKE_TEST` の値で判別できる。判定基準・実験設計自体は本番(Google Colab T4 GPU、`SMOKE_TEST=False`)を前提に事前登録しており、**宣言した判定基準は結果を見た後に変更しない**。**このノートブックの数値に基づいて結論(支持 / 反証 / 判定不能)を本文に記述しない**(結論は 7 節で、本番実行の数値と突き合わせたうえで記述する)。判定を行うコード自体は実装・実行し、動作確認の証跡としてセル出力に残すが、7 節(結果・考察)は見出しと箇条書きのみとする(6.10 節に記録した過去の本番実行の結果は例外的に本文として記述済みだが、これは事前登録した判定基準による結論の確定ではなく、前提不成立の記録である)。


### 6.1 較正セル: 学習率「高め」・gradient clipping 閾値・シード数の決定

**目的**: Q4(正規化後置・安定化技術なし)条件で、006 基準学習率の 1.25・2.5・5・10 倍(`LR_CALIBRATION_MULTIPLIERS`、等比刻み)を `CALIBRATION_STEPS`(= `NUM_STEPS`、本番と同じスケールで較正する。6.10.1 節で述べる 1 回目の本番実行の前提不成立を踏まえた修正)ステップだけ実行し、(1) 学習率「高め」の採用値、(2) gradient clipping 閾値、(3) シード数増加時の実行時間の外挿、の 3 つを決定する。

**(1) 学習率「高め」の選定方針(前提条件 P1 を採用条件に追加)**: 候補のうち、以下の **両方** を満たすものを採用候補とする。

- NaN / Inf に発散していないこと(較正用ステップ数の範囲で完全に崩壊していない)。
- **前提条件 P1(6.2 節)を満たすこと**: 較正実行の最終訓練損失が $\ln V \times$ `PRECONDITION_LOSS_RATIO` 以下であること(学習が実際に進んでいること)。

この 2 条件を満たす候補のうち、勾配ノルムのピーク / 平均比率(`compute_gradient_norm_peak_to_mean_ratio`)が最大の候補を選ぶ(「発散せず明確に不安定化する候補」という当初の設計意図を維持する)。どの候補も 2 条件を満たさない場合は、警告を出力したうえで最小倍率を採用し、**その旨をノートブックに明示的に記録する**(この場合、より低い倍率での再較正が必要になる)。

**(2) gradient clipping 閾値(分位点方式)**: 採用した学習率での較正実行における勾配ノルムの **`CLIP_QUANTILE` 分位点**(5.2 節で較正前に宣言済み、3.4 節で導出根拠を述べた分位点方式)。旧: 平均 $\mu_g$ + $k\sigma_g$ 方式(6.10.1 節参照、本番で閾値が高くなりすぎ前提条件 P2 が成立しなかった)。

**(3) シード数の外挿**: 較正用ステップ数ではなく **本番のステップ数**(`NUM_STEPS`)で Q4 全部乗せ条件(AdamW + warmup+cosine + gradient clipping、最も演算オーバーヘッドが大きい条件)を $n$ 個のシードで実行したときの累積実行時間を $n=1,2,3$ で実測し、$\log t = \log a + b \log n$ のあてはめでべき指数 $b$ を推定する。$b \approx 1$(線形)であることを確認したうえで、最も重い条件(Q4 全部乗せ)の 1 シードあたりの時間 $a$ を **全 10 条件で共通の保守的な(やや多めの)見積もり** として使い、条件×シードの全組み合わせ数への外挿値を計算する。この外挿値がセッション予算(`SESSION_BUDGET_SECONDS`)に収まるようにシード数(`SEEDS_MAIN_AXIS`・`SEEDS_OTHER`)を決定する(収まらない場合は警告を出し、シード数を保守的な値に落とす)。**較正が本番と同じ `NUM_STEPS` で 4 候補実行されるようになったため、較正自体の実行時間も無視できない**(6.3 節末尾の資源制約アサーションで、較正を含む総実行時間を確認する)。



```python
def _run_lm_condition(
    norm_first: bool,
    stabilization_level: str,
    peak_lr: float,
    clip_threshold: float | None,
    num_steps: int,
    seed: int,
) -> dict:
    model = build_model(norm_first=norm_first, seed=seed).to(device)
    cfg = build_optimizer_config(model, stabilization_level, peak_lr, clip_threshold or 0.0)
    history = train_language_model(
        model,
        train_ids,
        eval_windows,
        eval_mask,
        total_eval_bytes,
        num_steps=num_steps,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        learning_rate=cfg["learning_rate"],
        eval_interval=max(num_steps, EVAL_INTERVAL),
        device=device,
        seed=seed,
        optimizer=cfg["optimizer"],
        learning_rate_schedule=cfg["learning_rate_schedule"],
        gradient_clip_threshold=cfg["gradient_clip_threshold"],
    )
    return history


# --- (1)+(2) 学習率「高め」・gradient clipping 閾値の較正 ---
# 前提条件 P1 の閾値(PRECONDITION_LOSS_THRESHOLD)は 5.3 節末尾(VOCAB_SIZE 確定直後)
# で一度だけ定義済みであり、ここではそれをそのまま参照する(定義の重複を避ける)。

calibration_results = {}
for _mult in LR_CALIBRATION_MULTIPLIERS:
    _lr = BASE_LEARNING_RATE * _mult
    _t0 = time.time()
    _hist = _run_lm_condition(
        norm_first=False,
        stabilization_level="none",
        peak_lr=_lr,
        clip_threshold=None,
        num_steps=CALIBRATION_STEPS,
        seed=0,
    )
    _elapsed = time.time() - _t0
    _norms = np.array(_hist["gradient_norm"])
    _diverged = bool(np.isnan(_norms).any() or np.isinf(_norms).any() or np.isnan(_hist["train_loss"]).any())
    _final_loss = float("nan") if _diverged else float(_hist["train_loss"][-1])
    # 前提条件 P1 の判定量: 直近 P1_LOSS_WINDOW_STEPS 個のステップの訓練損失の平均
    # (バッチのばらつきにより閾値付近で判定が偶然左右されることを避けるため、
    # 最終ステップ単独ではなく窓平均を使う、5.3 節末尾で述べた通り)。
    # 主張 5(判定基準)は引き続き _final_loss(train_loss[-1])を使い、この量とは別である。
    _p1_loss = float("nan") if _diverged else float(np.mean(_hist["train_loss"][-P1_LOSS_WINDOW_STEPS:]))
    # 前提条件 P1: 較正実行の P1 判定量が PRECONDITION_LOSS_THRESHOLD 以下であること
    # (検証したい仮説とは独立な、学習が実際に進んでいるかどうかの条件)。
    _p1_ok = (not _diverged) and (_p1_loss <= PRECONDITION_LOSS_THRESHOLD)
    _peak_to_mean = float("nan") if _diverged else compute_gradient_norm_peak_to_mean_ratio(_norms)
    calibration_results[_mult] = {
        "lr": _lr,
        "history": _hist,
        "elapsed_seconds": _elapsed,
        "diverged": _diverged,
        "final_loss": _final_loss,
        "p1_loss": _p1_loss,
        "p1_ok": _p1_ok,
        "peak_to_mean": _peak_to_mean,
        "grad_norm_mean": float(_norms.mean()) if not _diverged else float("nan"),
        "grad_norm_std": float(_norms.std(ddof=1)) if not _diverged else float("nan"),
        "grad_norm_quantile": float(np.quantile(_norms, CLIP_QUANTILE)) if not _diverged else float("nan"),
    }
    print(f"lr={_lr:.2e}(x{_mult:.2f}): diverged={_diverged}, final_loss={_final_loss:.4f}, "
          f"p1_loss(直近{P1_LOSS_WINDOW_STEPS}ステップ平均)={_p1_loss:.4f}, "
          f"P1={'OK' if _p1_ok else 'NG'}, peak/mean={_peak_to_mean:.3f}, elapsed={_elapsed:.2f}s")

# 採用条件: (1) NaN/Inf に発散していない、かつ (2) 前提条件 P1 を満たす。
# この 2 条件を満たす候補のうち、peak/mean が最大のものを「発散せず明確に不安定化する候補」として採用する。
_survivors = {m: r for m, r in calibration_results.items() if r["p1_ok"]}
_calibration_fallback = False
if _survivors:
    HIGH_LR_MULTIPLIER = max(_survivors, key=lambda m: _survivors[m]["peak_to_mean"])
else:
    _calibration_fallback = True
    HIGH_LR_MULTIPLIER = min(LR_CALIBRATION_MULTIPLIERS)
    print()
    print("警告: 較正した全倍率で前提条件 P1(NaN/Inf に発散しない かつ 学習が実際に進んでいる)を"
          "満たさなかった。フォールバックとして最小倍率を採用するが、この場合はより低い倍率での"
          "再較正が必要である(この較正結果だけでは学習率『高め』の候補が 1 つも有効ではない)。")

HIGH_LEARNING_RATE = BASE_LEARNING_RATE * HIGH_LR_MULTIPLIER
_chosen = calibration_results[HIGH_LR_MULTIPLIER]
GRADIENT_CLIP_THRESHOLD = _chosen["grad_norm_quantile"]

print()
print(f"採用した学習率「高め」: {HIGH_LEARNING_RATE:.3e}(006 基準の x{HIGH_LR_MULTIPLIER:.2f})"
      f"{'(フォールバック: 前提条件 P1 を満たす候補が無かった)' if _calibration_fallback else ''}")
print(f"gradient clipping 閾値: {GRADIENT_CLIP_THRESHOLD:.4f} "
      f"(勾配ノルムの {CLIP_QUANTILE:.0%} 分位点、採用した学習率での較正実行より)")

```

    lr=3.75e-04(x1.25): diverged=False, final_loss=1.6992, p1_loss(直近20ステップ平均)=1.6767, P1=OK, peak/mean=5.798, elapsed=34.08s
    lr=7.50e-04(x2.50): diverged=False, final_loss=1.5827, p1_loss(直近20ステップ平均)=1.5604, P1=OK, peak/mean=9.447, elapsed=34.72s
    lr=1.50e-03(x5.00): diverged=False, final_loss=3.2713, p1_loss(直近20ステップ平均)=3.3166, P1=NG, peak/mean=13.580, elapsed=36.13s
    lr=3.00e-03(x10.00): diverged=False, final_loss=3.2717, p1_loss(直近20ステップ平均)=3.3171, P1=NG, peak/mean=13.527, elapsed=36.38s
    
    採用した学習率「高め」: 7.500e-04(006 基準の x2.50)
    gradient clipping 閾値: 0.7293 (勾配ノルムの 85% 分位点、採用した学習率での較正実行より)



```python
# --- (3) シード数増加時の実行時間の外挿 ---
_seed_scaling_ns = [1, 2, 3]
_seed_scaling_times = []
for _n in _seed_scaling_ns:
    _t0 = time.time()
    for _s in range(_n):
        _run_lm_condition(
            norm_first=False,
            stabilization_level="all",
            peak_lr=HIGH_LEARNING_RATE,
            clip_threshold=GRADIENT_CLIP_THRESHOLD,
            num_steps=NUM_STEPS,
            seed=100 + _s,
        )
    _elapsed = time.time() - _t0
    _seed_scaling_times.append(_elapsed)
    print(f"n_seeds={_n}: 累積実行時間={_elapsed:.2f}s")

_log_n = np.log(_seed_scaling_ns)
_log_t = np.log(_seed_scaling_times)
_seed_exponent, _seed_log_a = np.polyfit(_log_n, _log_t, deg=1)
_per_seed_seconds = float(np.exp(_seed_log_a))  # n=1 での時間(切片)= 1 シードあたりの時間
print(f"\nべき指数 b={_seed_exponent:.3f}(線形なら b≈1)、1 シードあたりの時間の推定 a={_per_seed_seconds:.2f}s")

TOTAL_CONDITION_SEED_PAIRS = 6 * SEEDS_OTHER + 4 * SEEDS_MAIN_AXIS
_lm_time_estimate = _per_seed_seconds * TOTAL_CONDITION_SEED_PAIRS

# 主張 6(合成タスク)の時間もダミー実行で見積もる(言語モデルより遥かに軽量なので概算でよい)。
_t0 = time.time()
_dummy_groups = [torch.nn.Parameter(torch.ones(1) * 2.0) for _ in range(4)]
_dummy_opt = AdamW(_dummy_groups, lr=1e-2, weight_decay=WEIGHT_DECAY)
for _ in range(SYNTH_STEPS):
    for p in _dummy_groups:
        p.grad = torch.randn(1)
    _dummy_opt.step()
_synth_per_run_seconds = time.time() - _t0
_synth_time_estimate = _synth_per_run_seconds * SYNTH_SEEDS * 2  # AdamW と Adam(L2)の2種類

total_time_estimate_seconds = _lm_time_estimate + _synth_time_estimate
print(f"\n条件×シードの組み合わせ数: {TOTAL_CONDITION_SEED_PAIRS}"
      f"(Q1〜Q3: 6 条件 x {SEEDS_OTHER} シード, Q4: 4 条件 x {SEEDS_MAIN_AXIS} シード)")
print(f"言語モデル実験(10 条件)の外挿実行時間: {_lm_time_estimate / 60:.1f} 分")
print(f"合成タスク実験(主張 6)の外挿実行時間: {_synth_time_estimate:.2f} 秒")
print(f"合計見積もり実行時間: {total_time_estimate_seconds / 60:.1f} 分"
      f"(予算 {SESSION_BUDGET_SECONDS / 60:.0f} 分)")

if total_time_estimate_seconds > SESSION_BUDGET_SECONDS:
    print("警告: 見積もり実行時間が予算を超えている。シード数(SEEDS_MAIN_AXIS・SEEDS_OTHER)を"
          "下げる方向で調整すること。")
else:
    print(f"OK: 見積もり実行時間は予算に対して "
          f"{SESSION_BUDGET_SECONDS / total_time_estimate_seconds:.1f} 倍の余裕がある。")

```

    n_seeds=1: 累積実行時間=36.74s
    n_seeds=2: 累積実行時間=74.00s
    n_seeds=3: 累積実行時間=110.84s
    
    べき指数 b=1.006(線形なら b≈1)、1 シードあたりの時間の推定 a=36.77s
    
    条件×シードの組み合わせ数: 38(Q1〜Q3: 6 条件 x 3 シード, Q4: 4 条件 x 5 シード)
    言語モデル実験(10 条件)の外挿実行時間: 23.3 分
    合成タスク実験(主張 6)の外挿実行時間: 1.31 秒
    合計見積もり実行時間: 23.3 分(予算 120 分)
    OK: 見積もり実行時間は予算に対して 5.1 倍の余裕がある。


**較正結果についての注記**: 上記セルの出力は生の較正数値であり、コミットしてよい。しかし、この数値の **解釈**(「学習率 x10 で明確に不安定化した」等)は結果・考察として書かない。以降の実験セルは、上記較正で決定した `HIGH_LEARNING_RATE`・`GRADIENT_CLIP_THRESHOLD`・`SEEDS_MAIN_AXIS`・`SEEDS_OTHER`をそのまま設定として使う、とだけ記載する。

**本修正について**: 本セルは、Google Colab T4 GPU での 1 回目の本番実行(6.10.1 節)で前提条件 P1 が成立しなかったこと、2 回目の本番実行(6.10.2 節)で前提条件 P2 が成立しなかったことを受けて較正手続きを修正したものである(CALIBRATION_STEPS を NUM_STEPS と同一にする・LR_CALIBRATION_MULTIPLIERS を 4 候補に拡張する・採用条件に前提条件 P1 を追加する・gradient clipping 閾値を分位点方式に変更し、さらに分位点の値を調整する)。**判定基準(対比量の定義・閾値の導出式・期待する差の方向)自体は一切変更していない**(6.2 節)。上記セルの出力が示す実行段階(スモークテストか本番か)は `SMOKE_TEST` の値(5.2 節)で判別できる。

### 6.2 実験宣言セル: 検証すること・判定基準

**共通の対比量・標準偏差の導出(全主張で共通)**: 条件 A・条件 B をそれぞれ $n_A$・$n_B$ シードで測定し、各シードの指標値の標本平均 $\bar{x}_A, \bar{x}_B$ ・標本標準偏差 $s_A, s_B$(不偏標準偏差、`ddof=1`)を求める。

$$
\Delta = \bar{x}_B - \bar{x}_A \qquad \mathrm{SE}(\Delta) = \sqrt{\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}} \qquad \text{閾値} = 2 \times \mathrm{SE}(\Delta)
$$

判定は **支持**($\Delta$ が期待方向に閾値超え)/ **反証**($\Delta$ が逆方向に閾値超え)/ **判定不能**($|\Delta| \le$ 閾値)の 3 分岐とする(`judge_three_way`、6.4 節)。

| # | 主張 | 対比条件(B vs A) | 診断量 | 期待方向 |
|---|---|---|---|---|
| 1 | 正規化後置は正規化前置より不安定になりやすい | Q3 vs Q1(なし条件、同一学習率) | 勾配ノルムのピーク / 平均比率 | $\Delta > 0$ |
| 2a | 正規化前置で学習率を上げると不安定性が増す | Q2 vs Q1(なし条件) | 勾配ノルムのピーク / 平均比率 | $\Delta > 0$ |
| 2b | 正規化後置で学習率を上げると不安定性が増す | Q4(なし)vs Q3(なし条件) | 勾配ノルムのピーク / 平均比率 | $\Delta > 0$ |
| 3 | gradient clipping は損失の暴れを抑える | Q4(clipping のみ)vs Q4(なし) | 最大単一ステップ損失上昇幅 | $\Delta < 0$(減少) |
| 4 | warmup + cosine は損失の暴れを抑える | Q4(warmup+cosine のみ)vs Q4(なし) | 最大単一ステップ損失上昇幅 | $\Delta < 0$(減少) |
| 5 | 全部乗せは「なし」より収束を遅らせない | Q4(全部乗せ)vs Q4(なし) | 固定ステップ数終了時点の損失値 | $\Delta \le 0$(悪化しない、$\Delta < 0$ を支持方向とする) |
| 6 | AdamW は実効減衰強度を二次モーメント推定から独立させる | Adam(L2 正則化混入)vs AdamW | パラメータ群間の実効減衰強度の乖離 | $\Delta > 0$(Adam(L2)の方が乖離が大きい) |

**主張 6 の対比の向き**: 「B vs A」を「Adam(L2 正則化混入)vs AdamW」とする(B=Adam(L2)、A=AdamW)。3.2 節の理論的予想は「Adam(L2)の方が群間の乖離が大きい」であるため、$\Delta = \bar{x}_{\text{L2}} - \bar{x}_{\text{AdamW}} > 0$ が支持方向になる(表内の記法を主張 1〜5 と揃えるため、実装上は「B $-$ A」の符号をそのまま使い、期待方向のみ主張ごとに指定する)。

**条件間の直交性(不変条件、6.3 節で検証)**: 上表の対比はいずれも、正規化方式・学習率・安定化技術の水準のうち **1 つだけ** を変え、他は固定する(3.5 節の表を参照)。


**前提条件(pre-condition)**: 上記の判定基準とは別に、「その実験が意味を持つために成立していなければならない状態」を、検証したい仮説とは独立な量として本番実行の前に宣言する。前提条件が不成立の主張は、支持 / 反証 / 判定不能のいずれとも判定せず「前提不成立」として記録する(6.5 節末尾で確認、6.6・6.8・6.9 節の判定に反映する)。

- **前提条件 P1(学習の進行)**: すべての条件・すべてのシードで、**直近 `P1_LOSS_WINDOW_STEPS`(本番設定でステップ数の 1/15、NUM_STEPS が確定した時点で宣言する)個のステップの訓練損失の平均** が一様分布相当の損失 $\ln V$(V は語彙サイズ)の `PRECONDITION_LOSS_RATIO` 倍以下であること。「学習が実際に進んでいるか」という、どの安定化技術が効くかという仮説とは独立な量である。**この判定量(窓平均)は、主張 5 の対比量である「固定ステップ数終了時点の損失値」(`train_loss[-1]`、判定基準)とは別の量である**: P1 は前提条件の判定のみに使う量であり、閾値付近でバッチのばらつきにより判定が偶然左右されることを避けるため窓平均を使うが、主張 5 自体の判定基準(対比量の定義)は変更していない。
- **前提条件 P2(gradient clipping の発動)**: gradient clipping を有効にした条件(Q4_clip・Q4_all)において、clipping が発動したステップの比率が `PRECONDITION_CLIP_RATE_MIN` 以上であること。「clipping が実際に発動しているか」という、clipping の効果(主張 3)そのものとは独立な量である(clipping が一度も発動していなければ、効果があってもなくても主張 3 の対比量 $\Delta$ は 0 に近づくため、$\Delta$ の値だけでは「効果がない」のか「そもそも発動していない」のかを区別できない)。

`PRECONDITION_LOSS_RATIO`・`PRECONDITION_CLIP_RATE_MIN` は 5.2 節で、較正の実測値を見る前に宣言済みである。前提条件の成否は主張ごとに、その主張の対比条件(上表の「対比条件」列)が全て P1 を満たすか、かつ clipping を含む条件については P2 も満たすかで判定する(主張 6 は言語モデルの学習を伴わない合成タスクであり、学習率・正規化方式の影響を受けないため前提条件を持たない)。

### 6.2.1 主張 3' の追加宣言: gradient clipping の勾配ノルムへの効果(新規シードによる事前登録)

**主張 3 との関係**: 主張 3(gradient clipping は損失の暴れを抑える、上表)は、対比条件 Q4(clipping のみ)vs Q4(なし)について「最大単一ステップ損失上昇幅」を対比量として判定した。3 回目の本番実行の結果、この対比量では判定不能となった(6.10.3 節)。原因は、対比量が gradient clipping の直接の作用点(勾配ノルム)から離れた下流の量だったことにあると考えられる。そこで、**gradient clipping が直接作用する量を対比量とする主張 3' を、主張 3 とは別の検証事項として追加する**(対比量が異なるものは別の実験として扱う)。**主張 3 の判定(判定不能)は変更せず、そのまま確定させる。** 主張 3' は主張 3 を置き換えるものではない。

**重要な制約: 既存 5 シードの数値を判定基準の設定に使わない**。既存シード 0〜4 による Q4_none・Q4_clip の勾配ノルムのピーク / 平均比率(3 回目の本番実行で 8.704・7.839、$\Delta = -0.865$)は **すでに観測済み** である。この数値を用いて主張 3' の判定基準($\Delta$・閾値)を設定すると、事前登録された主張ではなくなる。したがって主張 3' は、**既存シードと重複しない新規シード 5〜9(`SEEDS_NEW_CLAIMS`、5.2 節)** で Q4_none・Q4_clip を新たに実行し、**新規 5 シードのみ** で判定する。既存シード 0〜4 の値は 6.6.1 節で **診断として併記する** が、$\Delta$ の算出・標準偏差の推定・閾値の導出・判定のいずれにも使わない。この開示自体が誠実性の担保になる。

- **検証すること**: gradient clipping は、gradient clipping 適用前の勾配ノルムのピーク / 平均比率を抑える。
- **対比条件**: Q4_clip(B)vs Q4_none(A)、いずれも新規シード 5〜9(`SEEDS_NEW_CLAIMS`)。
- **対比量**: gradient clipping 適用前の勾配ノルムのピーク / 平均比率($\mathrm{peak/mean}$)。`history["gradient_norm"]`(`train_language_model()`が常に記録する clipping 適用前の値、`src/training/trainer.py`)から`compute_gradient_norm_peak_to_mean_ratio()`(`src/utils/statistics.py`)で算出する。
- **期待する差の方向**: $\Delta = \bar{x}_B - \bar{x}_A < 0$(gradient clipping を有効にすると比率が下がる)。
- **判定基準**: 6.2 節共通の公式をそのまま適用する。$\mathrm{SE}(\Delta) = \sqrt{s_A^2/n_A + s_B^2/n_B}$、閾値 $= 2 \times \mathrm{SE}(\Delta)$。$s_A$・$s_B$ は **新規 5 シードの実測のみ** から計算する(既存 5 シードのばらつきを閾値の設定に流用しない)。$\Delta$ が期待方向に閾値を超えれば支持、逆方向に閾値を超えれば反証、$|\Delta| \le$ 閾値なら判定不能とする(`judge_three_way`、6.4 節の共通実装をそのまま使う)。

**トートロジーでないことの明示**: 対比量は gradient clipping **適用前** の勾配ノルムから算出するため(`src/training/trainer.py`の記録タイミングの調査により確認済み)、「閾値でキャップしたから下がった」という定義上の帰結ではない。gradient clipping が学習の軌道を変えた結果として、後続ステップにおける勾配ノルムの暴れ自体が変化したかを測る。ただし、この量には「clipping が直接切った分」(切られたステップ自身の勾配ノルムが低下する効果)と「軌道の変化による間接的な効果」(その後のステップの勾配ノルムが変化する効果)の両方が含まれ、現在の対比量からはこの 2 つを分離できない。007 の目的(gradient clipping が学習を安定化させるかを検証すること)においてはこの分離は必要ないため、両者を合わせた効果を主張 3' の対比量として扱う。

**前提条件**: 主張 3 と同じ前提条件 P1(学習の進行)・P2(gradient clipping の発動)を適用する。新規シード 5〜9 についても、6.5.1 節で 6.5 節末尾と同じ形式で P1・P2 の成否を確認する。前提条件が不成立の場合、主張 3 と同様に「前提不成立」として記録し、支持 / 反証 / 判定不能のいずれとも判定しない。

### 6.2.2 主張 4' の追加宣言: warmup + cosine の損失差分標準偏差への効果(新規シードによる事前登録)

**主張 4 との関係**: 主張 4(warmup + cosine は損失の暴れを抑える、6.2 節)は、対比条件 Q4(warmup + cosine のみ)vs Q4(なし)について「最大単一ステップ損失上昇幅」を対比量として判定した。3 回目の本番実行の結果、この対比量では判定不能となった($\Delta = -0.0400$、閾値 $= 0.1351$)。主張 3 と同じく、対比量が極値統計であるためシード間のばらつきが大きく、効果があっても検出しづらい構造になっている。そこで、**損失差分(直前ステップとの損失の差)系列全体のばらつきを対比量とする主張 4' を、主張 4 とは別の検証事項として追加する**(対比量が異なるものは別の実験として扱う)。**主張 4 の判定(判定不能)は変更せず、そのまま確定させる。** 主張 4' は主張 4 を置き換えるものではない。

**重要な制約: 既存 5 シードの数値を判定基準の設定に使わない**。主張 3' と同じ制約を課す。既存シード 0〜4 の`loss_step_delta`標準偏差は 6.6.1 節で **診断として併記する** が、$\Delta$ の算出・標準偏差の推定・閾値の導出・判定のいずれにも使わない。主張 4' は、**既存シードと重複しない新規シード 5〜9(`SEEDS_NEW_CLAIMS`、5.2 節)** で Q4_warmup_cosine・Q4_none を実行し、**新規 5 シードのみ** で判定する。**Q4_none の新規シード実行は主張 3' と共有し、二重実行しない**(6.5.1 節)。

- **検証すること**: warmup + cosine スケジュールは、損失差分(直前ステップとの損失の差)系列の標準偏差を抑える。
- **対比条件**: Q4_warmup_cosine(B)vs Q4_none(A)、いずれも新規シード 5〜9(`SEEDS_NEW_CLAIMS`)。
- **対比量**: `loss_step_delta`系列の標準偏差(`compute_loss_step_delta_std()`、`src/utils/statistics.py`)。
- **期待する差の方向**: $\Delta = \bar{x}_B - \bar{x}_A < 0$(warmup + cosine を有効にすると標準偏差が下がる)。
- **判定基準**: 6.2 節共通の公式をそのまま適用する。$\mathrm{SE}(\Delta) = \sqrt{s_A^2/n_A + s_B^2/n_B}$、閾値 $= 2 \times \mathrm{SE}(\Delta)$。$s_A$・$s_B$ は **新規 5 シードの実測のみ** から計算する(既存 5 シードのばらつきを閾値の設定に流用しない)。$\Delta$ が期待方向に閾値を超えれば支持、逆方向に閾値を超えれば反証、$|\Delta| \le$ 閾値なら判定不能とする(`judge_three_way`、6.4 節の共通実装をそのまま使う)。

**ピーク / 平均比率を使わない理由**: 主張 3' では gradient clipping の直接の作用点である勾配ノルムのピーク / 平均比率を対比量とした。warmup + cosine には同じ処方が使えない。warmup は学習序盤の学習率を意図的に下げる技術であり、序盤の勾配ノルムの平均が下がることで、ピーク / 平均比率という指標そのものが見かけ上大きくなるアーティファクトが疑われる(3 回目の本番実行で Q4_warmup_cosine のピーク / 平均比率は 11.307 と、Q4_none の 8.704 より大きい、6.10.3 節)。このため主張 4' では、勾配ノルムではなく「損失の暴れ」という検証事項をそのまま捉える損失差分の分布を対比量とする。最大値という極値統計ではなく標準偏差(分布全体の統計量)を使うことで、極値統計を対比量に用いる場合に生じる検出力の低下を避ける設計になっている。

**前提条件**: 主張 4 と同じ前提条件 P1(学習の進行)を適用する。P2(gradient clipping の発動)は gradient clipping 固有の前提条件であり、主張 4' には適用しない。新規シード 5〜9 についても、6.5.1 節で P1 の成否を確認する。前提条件が不成立の場合、主張 4 と同様に「前提不成立」として記録し、支持 / 反証 / 判定不能のいずれとも判定しない。

### 6.3 不変条件のアサーション

- 全条件で非埋め込みパラメータ数が一致すること(5.4 節で確認済み)。
- 評価バッチ集合(`eval_windows`・`eval_mask`・`total_eval_bytes`)が全条件・シードで共有されていること(同一のオブジェクトを使い回すことで保証する)。
- 全条件でステップ数(`NUM_STEPS`)が同一であること。
- 較正で決定した gradient clipping 閾値・学習率・シード数が資源制約(セッション時間)を満たすこと(6.1 節で確認済み)。



```python
assert count_non_embedding_parameters(build_model(True, 0)) == count_non_embedding_parameters(
    build_model(False, 0)
)
print("OK: 非埋め込みパラメータ数の一致(再確認)")

_eval_windows_snapshot = eval_windows.clone()
_eval_mask_snapshot = eval_mask.clone()
_total_eval_bytes_snapshot = total_eval_bytes
# 全条件のループ(6.5 節)で同一の eval_windows / eval_mask / total_eval_bytes を渡すことを、
# ループ後にこのスナップショットと比較して確認する(不変条件アサーション、6.5 節末尾で実行)。
print("評価バッチ集合のスナップショットを保存した(6.5 節末尾で一致を確認する)")

# SMOKE_TEST=True(較正・動作確認が目的)では超過しても警告のみで停止しない。
# SMOKE_TEST=False(本番、Google Colab T4 GPU)では、超過時に実際に AssertionError を
# 送出して停止する(GPU セッション時間を無駄にしないため。SEEDS_MAIN_AXIS・SEEDS_OTHER を
# 下げてこのノートブックを再実行する必要があることを明示する)。
_within_budget = total_time_estimate_seconds <= SESSION_BUDGET_SECONDS
assert SMOKE_TEST or _within_budget, (
    f"見積もり実行時間({total_time_estimate_seconds / 60:.1f}分)がセッション予算"
    f"({SESSION_BUDGET_SECONDS / 60:.0f}分)を超過しています。SEEDS_MAIN_AXIS・SEEDS_OTHER を"
    f"下げてこのノートブックを再実行してください。"
)
print(f"資源制約の確認: 見積もり実行時間 {total_time_estimate_seconds / 60:.1f} 分 / "
      f"予算 {SESSION_BUDGET_SECONDS / 60:.0f} 分 "
      f"({'OK' if _within_budget else ('警告: 超過(SMOKE_TEST=True のため続行)' if SMOKE_TEST else '超過')})")

```

    OK: 非埋め込みパラメータ数の一致(再確認)
    評価バッチ集合のスナップショットを保存した(6.5 節末尾で一致を確認する)
    資源制約の確認: 見積もり実行時間 23.3 分 / 予算 120 分 (OK)




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/02_pretraining/007_training_stabilization.ipynb
