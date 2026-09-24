---
title: "LoRA(Low-Rank Adaptation)(実装・実験編 1/3)"
---

この記事は後編(実装・実験編 1/3)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/012_low_rank_adaptation-theory)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/012_low_rank_adaptation-practice-2)。

## 4. 実装方針 / Implementation Policy

**`src/`に切り出す(スクラッチ実装)**: `src/layers/lora.py`(本トピックで新規作成)。

- `LoRALinear`: ベース層を合成(composition)で包み、$h = W_0 x + \frac{\alpha}{r} B A x$ を
  計算する。ベース層は`nn.Linear`に限定せず、`forward(x)`と入出力次元を持つ任意の
  モジュールを受け付ける(013 で量子化したベース層に差し替えるため)。スケーリングは
  $\alpha / r$ のみを実装する(rsLoRA の $\alpha / \sqrt{r}$ は実験で使わないため実装しない)。
  `merge()`・`unmerge()`・`delta_weight()`を持ち、マージはベース層が`.weight`を持つ
  場合のみ許可する(持たない場合は`TypeError`を送出する)。
- `apply_lora(model, target_module_names, rank, alpha)`: 対象モジュールを置換し、LoRA 以外の
  全パラメータを凍結する。置換したモジュールの完全修飾名を返す。
- `compute_lora_parameter_count(d_in, d_out, rank)`: 閉形式の学習可能パラメータ数
  $r(d_{\mathrm{in}} + d_{\mathrm{out}})$。

016(SFT)でも使うため、使い方を docstring に明記している。既存の`src/`モジュールは
変更しない(`src/layers/__init__.py`から上記 3 つを公開する行を追加したのみで、既存の
関数・クラスの挙動は変わらない)。`MultiHeadAttention`の Query・Key・Value の射影は
独立した`nn.Linear`(`w_q`・`w_k`・`w_v`)なので、`w_q`・`w_v`をそのまま置換できる。

**ノートブック内に直接書く(012 固有)**:

- 実験 B の対照となるランダムマスク疎微調整層(`RandomMaskSparseLinear`)。
  更新量を長さ $k$ のパラメータベクトルとして持ち、固定したインデックスへ scatter する
  ことで、学習可能パラメータ数が文字どおり $k$ になるようにする。
- 実験の学習・評価ループ。学習そのものは`src/training/trainer.py`の
  `train_language_model()`を再利用し、optimizer は`src/training/optimizer.py`の`AdamW`、
  学習率スケジュールは`src/training/schedule.py`の warmup + cosine を使う(007・008 と同じ部品)。
- 出力例の top-p サンプリングのループ(フィルタは`src/generation/decoding.py`の
  `top_p_filter()`を使う)。

**アップロード方針**: 本トピックで学習する LoRA アダプタ・全パラメータ微調整の重みは、
条件間の比較のためのものであり、後続トピックの入力にも読者が単体で取得する対象にもならない。
保存もアップロードもしない。

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
    remote: Enumerating objects: 949, done.[K
    remote: Counting objects: 100% (447/447), done.[K
    remote: Compressing objects: 100% (296/296), done.[K
    remote: Total 949 (delta 265), reused 287 (delta 150), pack-reused 502 (from 1)[K
    Receiving objects: 100% (949/949), 11.73 MiB | 9.95 MiB/s, done.
    Resolving deltas: 100% (521/521), done.
    /content/ai-theories
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m20.5/20.5 MB[0m [31m29.6 MB/s[0m eta [36m0:00:00[0m
    [?25h[2mUsing Python 3.13.15 environment at: /usr[0m
    [2K[2mResolved [1m52 packages[0m [2min 334ms[0m[0m
    [2K[2mPrepared [1m31 packages[0m [2min 42.14s[0m[0m
    [2mUninstalled [1m17 packages[0m [2min 724ms[0m[0m
    [2K[2mInstalled [1m31 packages[0m [2min 311ms[0m[0m
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
import copy
import functools
import hashlib
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import nn

from src.data.text import (
    encode_corpus,
    encode_text_to_memmap,
    load_tiny_shakespeare,
    make_evaluation_windows,
)
from src.data.tokenizer import load_bpe_id_tokenizer_from_hub
from src.generation.decoding import top_p_filter
from src.layers.feedforward import SwiGLUFeedForwardNetwork
from src.layers.lora import LoRALinear, apply_lora, compute_lora_parameter_count
from src.layers.normalization import RMSNorm
from src.layers.positional_encoding import RotaryPositionEmbedding
from src.models.gpt import GPTLanguageModel
from src.training.optimizer import AdamW
from src.training.schedule import compute_warmup_cosine_learning_rate
from src.training.trainer import evaluate_bits_per_byte, train_language_model
from src.utils.statistics import fit_power_law_exponent
from src.utils.visualization import plot_learning_curves_multi_seed, plot_seed_scatter

device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"torch: {torch.__version__} / device: {device}")

ROOT = Path(".")
CORPUS_CACHE_DIR = ROOT / ".cache" / "tiny_shakespeare"  # 外部取得したコーパス(データ源で命名)
ENCODED_CACHE_DIR = ROOT / ".cache" / "012_encoded"  # 012 の条件で符号化したトークン列

MODEL_REPO_ID = "kojikojiprg/ai-theories-small-gpt-en"
MODEL_REVISION = "main"
TOKENIZER_REPO_ID = "kojikojiprg/ai-theories-tokenizer-en"


def hash_tensor(t: torch.Tensor) -> str:
    # テンソルの内容が条件間・実行前後で同一であることを確認するためのハッシュ。
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def sync_device() -> None:
    # 時間計測の直前・直後に、非同期に実行される GPU の処理の完了を待つ。
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


precondition_status: dict[str, bool] = {}  # 前提条件の成否(6.1 節で宣言、各実験の節で記録)
```

    torch: 2.13.0+cu130 / device: cuda


### 5.2 スケールの設定(`SMOKE_TEST`の配線)

水準の定義をこの 1 箇所(`LEVELS`)に集約する。`SMOKE_TEST = True`はローカルでのコード経路の
確認用、`SMOKE_TEST = False`は Google Colab T4 での本番実行用である。モデルは事前学習済みの
ものを使うため縮小しない。スモークテストでは、訓練テキストを先頭の一部に切り詰めることで
ステップ数を縮小する(エポック数 $E$ とステップ数の決め方(6.1 節)は本番と共通なので、
「総学習トークン数 = 訓練トークン数 × $E$」という構造は保たれる)。

縮小の規則:

- **rank の集合は縮小しない**(本番・スモークテストとも $\{1,2,4,8,16,32,64\}$ = 公比 2 の等比)。
  中間の rank(2・4・16・32)を描画用のシード数で学習するコード経路を、スモークテストでも実際に
  通すためである。
- 学習率のグリッドは等比のまま縮小する(本番: 公比 $\sqrt{10}$ の 5 水準、スモークテスト:
  そこから 1 つおきに取った公比 10 の 3 水準)。グリッドを拡張するときの刻みも、それぞれの公比に揃える。
- ステップ数・シード数は全条件で共通の規則で縮小する(判定に使う条件のシード数 > 曲線の描画用の
  中間の rank のシード数、という順序関係を保つ)。
- LoRA と疎微調整の学習可能パラメータ数の一致は、モデルを縮小しないので縮小後もそのまま成り立つ。

較正(6.3 節)は`SMOKE_TEST`に従う水準のステップ数で行う。本番(`SMOKE_TEST = False`)では
本番と同じステップ数で較正することになる。スモークテストでの較正は動作確認のみを目的とする。


```python
SMOKE_TEST = False  # Claude Code はこの True 側のみ実行する(Colab T4 では False に切り替える)
_smoke_tag = "[動作確認のみ、結論ではない] " if SMOKE_TEST else ""

# --- 全水準で共通の定数(本番実行前に宣言し、SMOKE_TEST で変えない) ---
EPOCHS = 2  # E: 総学習トークン数 = 訓練トークン数 x E(6.1 節)
TRAIN_RATIO, CALIBRATION_RATIO, EVALUATION_RATIO = 0.90, 0.05, 0.05
BATCH_SIZE = 32  # 008 と同一
WARMUP_RATIO = 0.1  # 007・008 と同一
MIN_LEARNING_RATE_RATIO = 0.01  # 007・008 と同一
WEIGHT_DECAY = 0.0  # 全条件で 0(理由は 6.1 節)
CLIP_QUANTILE = 0.90  # gradient clipping の閾値の決め方(008 と同一、6.3 節)
LORA_TARGET_MODULES = ("w_q", "w_v")  # Query・Value 射影(3.6 節)
LORA_ALPHA = 8.0  # 全 rank 共通。r=8 のとき alpha / r = 1
MAIN_RANK = 8  # 実験 A・B の LoRA の rank、実験 B の疎微調整のパラメータ数の基準
JUDGED_RANKS = (1, 8, 64)  # 実験 C の判定に使う rank
DELTA_A = 0.9  # 実験 A の基準 delta(6.1 節)
PRECONDITION_A1_FACTOR = 10.0  # 前提条件 A1: b0 - b1 >= 10 sigma_b1
PRECONDITION_TRAIN_LOSS_DROP = 0.05  # 前提条件 B1・C1: 訓練損失が 5% 以上低下
TRAIN_LOSS_WINDOWS = 64  # 訓練損失の測定に使う、訓練集合から等間隔に選ぶ窓の数(6.1 節)
EVAL_POINTS = 8  # 評価集合の学習曲線の記録点数の目安(最終ステップは必ず記録する)
LR_GRID_RATIO_EXPONENT_STEP = 0.5  # 学習率 = 10 ** (k * 0.5)(公比 sqrt(10))
MAX_GRID_EXPANSIONS = 4  # 較正でグリッドの端が選ばれたときに拡張する最大回数
BOOTSTRAP_RESAMPLES = 10_000
SESSION_BUDGET_SECONDS = 2 * 60 * 60  # Google Colab 1 セッションの目安予算(006〜011 と同一)

# 学習率のグリッド(k の整数列、学習率 = 10 ** (k / 2))。本番の中央の水準は、予備掃引
# (6.3 節の Markdown に記録)から端にならない位置に置いた。端が選ばれた場合は 6.3 節で自動的に拡張する。
PROD_LR_GRID_K = {
    "full": (-7, -6, -5, -4, -3),  # 3.16e-4 .. 3.16e-2
    "lora": (-5, -4, -3, -2, -1),  # 3.16e-3 .. 3.16e-1
    "sparse": (-4, -3, -2, -1, 0),  # 1e-2 .. 1
}

# --- 水準の定義(この 1 箇所に集約する) ---
LEVELS = {
    "smoke": {
        "TRAIN_TEXT_FRACTION": 0.15,  # 訓練テキストの先頭から使う割合
        "RANKS": (1, 2, 4, 8, 16, 32, 64),  # 本番と同一(縮小しない、公比 2 の等比)
        "LR_GRID_K": {m: g[::2] for m, g in PROD_LR_GRID_K.items()},  # 公比 10 の 3 水準
        "LR_GRID_K_STEP": 2,  # グリッド拡張の刻み(公比 10)
        "NUM_SEEDS_JUDGED": 2,
        "NUM_SEEDS_INTERMEDIATE": 1,
    },
    "prod": {
        "TRAIN_TEXT_FRACTION": 1.0,
        "RANKS": (1, 2, 4, 8, 16, 32, 64),  # 公比 2 の等比
        "LR_GRID_K": dict(PROD_LR_GRID_K),  # 公比 sqrt(10) の 5 水準
        "LR_GRID_K_STEP": 1,
        "NUM_SEEDS_JUDGED": 5,
        "NUM_SEEDS_INTERMEDIATE": 2,
    },
}

CURRENT_LEVEL_NAME = "smoke" if SMOKE_TEST else "prod"
CFG = LEVELS[CURRENT_LEVEL_NAME]
TRAIN_TEXT_FRACTION = CFG["TRAIN_TEXT_FRACTION"]
RANKS = CFG["RANKS"]
LR_GRID_K = CFG["LR_GRID_K"]
LR_GRID_K_STEP = CFG["LR_GRID_K_STEP"]
NUM_SEEDS_JUDGED = CFG["NUM_SEEDS_JUDGED"]
NUM_SEEDS_INTERMEDIATE = CFG["NUM_SEEDS_INTERMEDIATE"]
SEEDS_JUDGED = tuple(range(NUM_SEEDS_JUDGED))
SEEDS_INTERMEDIATE = tuple(range(NUM_SEEDS_INTERMEDIATE))


def learning_rate_from_k(k: int) -> float:
    return 10.0 ** (k * LR_GRID_RATIO_EXPONENT_STEP)


# 縮小規則の確認(5.2 節の Markdown セル)。
assert set(JUDGED_RANKS) <= set(RANKS) and MAIN_RANK in RANKS
_rank_ratios = {RANKS[i + 1] / RANKS[i] for i in range(len(RANKS) - 1)}
assert len(_rank_ratios) == 1, f"rank の集合が等比でない: {RANKS}"
for _m, _g in LR_GRID_K.items():
    assert len({_g[i + 1] - _g[i] for i in range(len(_g) - 1)}) == 1, f"{_m} のグリッドが等比でない"
    assert _g[1] - _g[0] == LR_GRID_K_STEP, "拡張の刻みがグリッドの公比と一致しない"
assert NUM_SEEDS_JUDGED > NUM_SEEDS_INTERMEDIATE >= 1

print(f"SMOKE_TEST={SMOKE_TEST}(現在の水準: {CURRENT_LEVEL_NAME!r})")
print(f"EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, TRAIN_TEXT_FRACTION={TRAIN_TEXT_FRACTION}")
print(f"RANKS={RANKS}, JUDGED_RANKS={JUDGED_RANKS}, MAIN_RANK={MAIN_RANK}, LORA_ALPHA={LORA_ALPHA}")
print(f"NUM_SEEDS_JUDGED={NUM_SEEDS_JUDGED}, NUM_SEEDS_INTERMEDIATE={NUM_SEEDS_INTERMEDIATE}")
for _m, _g in LR_GRID_K.items():
    print(
        f"学習率グリッド[{_m}]: {[f'{learning_rate_from_k(k):.2e}' for k in _g]}"
        f"(拡張の刻み k={LR_GRID_K_STEP})"
    )
print(
    f"WEIGHT_DECAY={WEIGHT_DECAY}, WARMUP_RATIO={WARMUP_RATIO}, "
    f"MIN_LEARNING_RATE_RATIO={MIN_LEARNING_RATE_RATIO}, CLIP_QUANTILE={CLIP_QUANTILE}"
)
```

    SMOKE_TEST=False(現在の水準: 'prod')
    EPOCHS=2, BATCH_SIZE=32, TRAIN_TEXT_FRACTION=1.0
    RANKS=(1, 2, 4, 8, 16, 32, 64), JUDGED_RANKS=(1, 8, 64), MAIN_RANK=8, LORA_ALPHA=8.0
    NUM_SEEDS_JUDGED=5, NUM_SEEDS_INTERMEDIATE=2
    学習率グリッド[full]: ['3.16e-04', '1.00e-03', '3.16e-03', '1.00e-02', '3.16e-02'](拡張の刻み k=1)
    学習率グリッド[lora]: ['3.16e-03', '1.00e-02', '3.16e-02', '1.00e-01', '3.16e-01'](拡張の刻み k=1)
    学習率グリッド[sparse]: ['1.00e-02', '3.16e-02', '1.00e-01', '3.16e-01', '1.00e+00'](拡張の刻み k=1)
    WEIGHT_DECAY=0.0, WARMUP_RATIO=0.1, MIN_LEARNING_RATE_RATIO=0.01, CLIP_QUANTILE=0.9


### 5.3 ベースモデル・トークナイザの取得(008 がアップロードしたもの)

ベースモデルは`kojikojiprg/ai-theories-small-gpt-en`の`main`ブランチ、トークナイザは
`kojikojiprg/ai-theories-tokenizer-en`から取得する。モデルの構成(層数・$d_{\mathrm{model}}$・
文脈長など)はノートブックに書き込まず、取得した`config.json`から読む。


```python
tokenizer, _tokenizer_loaded_from_hub = load_bpe_id_tokenizer_from_hub(TOKENIZER_REPO_ID)
assert _tokenizer_loaded_from_hub, "トークナイザを Hugging Face Hub から取得できなかった"

_config_path = hf_hub_download(
    repo_id=MODEL_REPO_ID, filename="config.json", revision=MODEL_REVISION
)
_state_path = hf_hub_download(
    repo_id=MODEL_REPO_ID, filename="model_state.pt", revision=MODEL_REVISION
)
MODEL_CONFIG = json.loads(Path(_config_path).read_text(encoding="utf-8"))
print("MODEL_CONFIG:", json.dumps(MODEL_CONFIG, indent=2))
assert tokenizer.vocab_size == MODEL_CONFIG["vocabulary_size"]
assert MODEL_CONFIG["positional_encoding"] == "rope"
assert MODEL_CONFIG["normalization"] == "rmsnorm" and MODEL_CONFIG["feed_forward"] == "swiglu"

D_MODEL = MODEL_CONFIG["d_model"]
NUM_LAYERS = MODEL_CONFIG["num_layers"]
NUM_HEADS = MODEL_CONFIG["num_heads"]
SWIGLU_D_FF = MODEL_CONFIG["swiglu_d_ff"]
VOCAB_SIZE = MODEL_CONFIG["vocabulary_size"]
SEQUENCE_LENGTH = MODEL_CONFIG["sequence_length"]  # 学習・評価の系列長はモデルの文脈長に揃える


def build_base_model_from_hub() -> GPTLanguageModel:
    d_k = D_MODEL // NUM_HEADS
    model = GPTLanguageModel(
        vocabulary_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=MODEL_CONFIG["d_ff"],
        max_sequence_length=SEQUENCE_LENGTH,
        positional_transform=RotaryPositionEmbedding(d_k, max_position=SEQUENCE_LENGTH),
        normalization_factory=RMSNorm,
        feed_forward_factory=functools.partial(SwiGLUFeedForwardNetwork, D_MODEL, SWIGLU_D_FF),
        tie_embeddings=MODEL_CONFIG["tie_embeddings"],
        dropout=MODEL_CONFIG["dropout"],
        norm_first=MODEL_CONFIG["norm_first"],
    )
    model.load_state_dict(torch.load(_state_path, map_location="cpu"))
    return model.eval()


base_model = build_base_model_from_hub().to(device)
for _p in base_model.parameters():
    _p.requires_grad_(False)  # base_model 自体は学習しない(各条件は deepcopy から始める)
BASE_STATE_HASH = hash_tensor(
    torch.cat([p.detach().flatten().cpu() for p in base_model.parameters()])
)

# 各条件は、Hub から読み込み直す代わりに base_model の deepcopy から始める(冗長な再読み込みの排除)。
# deepcopy が読み込み直したモデルと完全に同一であることを確認する(重み共有も保たれること)。
_fresh = build_base_model_from_hub().to(device)
_copied = copy.deepcopy(base_model)
_fresh_state, _copied_state = _fresh.state_dict(), _copied.state_dict()
assert _fresh_state.keys() == _copied_state.keys()
assert all(torch.equal(_fresh_state[k], _copied_state[k]) for k in _fresh_state)
assert _copied.lm_head.weight is _copied.token_embedding.weight  # 重み共有が deepcopy 後も保たれる
del _fresh, _copied, _fresh_state, _copied_state

TOTAL_PARAMETERS = sum(p.numel() for p in base_model.parameters())  # 重み共有は 1 回だけ数える
# 閉形式: V d + L (4 d^2 + 3 d d_ff' + 2 d) + d(埋め込み(出力層と共有)、各層の注意機構の 4 射影・
# SwiGLU の 3 射影・RMSNorm 2 個、最終 RMSNorm。いずれもバイアスなし)
_closed_form_total = (
    VOCAB_SIZE * D_MODEL
    + NUM_LAYERS * (4 * D_MODEL**2 + 3 * D_MODEL * SWIGLU_D_FF + 2 * D_MODEL)
    + D_MODEL
)
assert _closed_form_total == TOTAL_PARAMETERS, (TOTAL_PARAMETERS, _closed_form_total)
print(f"全パラメータ数 P = {TOTAL_PARAMETERS:,}(閉形式と一致)")
print(
    f"L={NUM_LAYERS}, d_model={D_MODEL}, h={NUM_HEADS}, d_ff'(SwiGLU)={SWIGLU_D_FF}, "
    f"V={VOCAB_SIZE}, 文脈長={SEQUENCE_LENGTH}"
)
```


    tokenizer.json:   0%|          | 0.00/661k [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/305 [00:00<?, ?B/s]



    model_state.pt: reconstructing file:   0%|          |  0.00B / 21.0MB            



    model_state.pt: downloading bytes:           |  0.00B            


    MODEL_CONFIG: {
      "vocabulary_size": 8192,
      "d_model": 256,
      "num_layers": 4,
      "num_heads": 8,
      "d_ff": 1024,
      "swiglu_d_ff": 683,
      "sequence_length": 256,
      "dropout": 0.0,
      "tie_embeddings": true,
      "positional_encoding": "rope",
      "normalization": "rmsnorm",
      "feed_forward": "swiglu",
      "norm_first": true
    }
    全パラメータ数 P = 5,246,208(閉形式と一致)
    L=4, d_model=256, h=8, d_ff'(SwiGLU)=683, V=8192, 文脈長=256


### 5.4 コーパスの取得・分割・符号化

Tiny Shakespeare を **文字列の段階で** 連続区間に分割する(先頭から訓練 90%・較正用検証 5%・
評価 5%)。学習率の較正は較正用検証集合で行い、判定はすべて評価集合で行う。

符号化したトークン列は`.cache/012_encoded/`に`uint16`の memmap としてキャッシュする
(同一セッション内の再実行で再符号化しない)。キャッシュから読んだトークン列が、キャッシュを
経由せず直接符号化したトークン列と完全に一致することを確認する。


```python
raw_text = load_tiny_shakespeare(CORPUS_CACHE_DIR)
_n_chars = len(raw_text)
_n_calibration = int(_n_chars * CALIBRATION_RATIO)
_n_evaluation = int(_n_chars * EVALUATION_RATIO)
_n_train_full = _n_chars - _n_calibration - _n_evaluation
TRAIN_START, CALIBRATION_START, EVALUATION_START = 0, _n_train_full, _n_train_full + _n_calibration

train_text_full = raw_text[:CALIBRATION_START]
calibration_text = raw_text[CALIBRATION_START:EVALUATION_START]
evaluation_text = raw_text[EVALUATION_START:]
# スモークテストでは訓練テキストの先頭のみを使う(較正用検証・評価の集合は本番と同一)。
train_text = train_text_full[: int(len(train_text_full) * TRAIN_TEXT_FRACTION)]
assert train_text_full + calibration_text + evaluation_text == raw_text
print(f"raw_text: {_n_chars:,} 文字 / {len(raw_text.encode('utf-8')):,} バイト")
print(
    f"訓練: {len(train_text):,} 文字(全体の訓練区間は {len(train_text_full):,} 文字)/ "
    f"較正用検証: {len(calibration_text):,} 文字 / 評価: {len(evaluation_text):,} 文字"
)


def encode_split_with_cache(name: str, text: str) -> torch.Tensor:
    memmap = encode_text_to_memmap(tokenizer, text, ENCODED_CACHE_DIR / f"{name}.u16")
    return torch.from_numpy(np.asarray(memmap, dtype=np.int64))


_t0 = time.time()
train_ids = encode_split_with_cache(f"train_{TRAIN_TEXT_FRACTION}", train_text)
train_ids_full = encode_split_with_cache("train_1.0", train_text_full)
calibration_ids = encode_split_with_cache("calibration", calibration_text)
evaluation_ids = encode_split_with_cache("evaluation", evaluation_text)
print(f"符号化(キャッシュ経由): {time.time() - _t0:.2f}s")

# --- キャッシュの前後で数値が完全に一致すること(キャッシュを経由しない直接の符号化と比較) ---
for _name, _text, _ids in (
    ("train", train_text, train_ids),
    ("train_full", train_text_full, train_ids_full),
    ("calibration", calibration_text, calibration_ids),
    ("evaluation", evaluation_text, evaluation_ids),
):
    _direct = encode_corpus(tokenizer, _text)
    assert torch.equal(_direct, _ids), (
        f"{_name}: キャッシュ経由のトークン列が直接の符号化と一致しない"
    )
    # 可逆であるべき変換(符号化と復号)がラウンドトリップで完全一致すること
    assert tokenizer.decode(_ids.tolist()) == _text, (
        f"{_name}: 符号化 -> 復号のラウンドトリップが一致しない"
    )
print("キャッシュ経由と直接の符号化の完全一致・符号化 -> 復号のラウンドトリップ: OK(4 分割すべて)")

calibration_windows, calibration_mask = make_evaluation_windows(calibration_ids, SEQUENCE_LENGTH)
evaluation_windows, evaluation_mask = make_evaluation_windows(evaluation_ids, SEQUENCE_LENGTH)
calibration_bytes = len(calibration_text.encode("utf-8"))
evaluation_bytes = len(evaluation_text.encode("utf-8"))

# 評価の分母の期待値を、評価ループに渡す変数とは独立に、生テキストと分割位置から直接計算する。
EXPECTED_EVALUATION_BYTES = len(raw_text.encode("utf-8")) - len(
    raw_text[:EVALUATION_START].encode("utf-8")
)
EXPECTED_CALIBRATION_BYTES = len(raw_text[:EVALUATION_START].encode("utf-8")) - len(
    raw_text[:CALIBRATION_START].encode("utf-8")
)
assert (
    evaluation_bytes == EXPECTED_EVALUATION_BYTES
    and calibration_bytes == EXPECTED_CALIBRATION_BYTES
)
EVALUATION_WINDOWS_HASH = hash_tensor(evaluation_windows)
EVALUATION_MASK_HASH = hash_tensor(evaluation_mask)

# 訓練損失の測定用: 訓練集合の非重複窓から、パディングを含まない窓を等間隔に選ぶ(6.1 節)。
_train_windows_all, _train_mask_all = make_evaluation_windows(train_ids, SEQUENCE_LENGTH)
_full_windows = torch.nonzero(_train_mask_all.all(dim=1)).flatten()
_selected = _full_windows[
    torch.linspace(0, len(_full_windows) - 1, min(TRAIN_LOSS_WINDOWS, len(_full_windows)))
    .round()
    .long()
]
train_loss_windows = _train_windows_all[_selected]
print(f"訓練損失の測定窓: {tuple(train_loss_windows.shape)}")

# --- ステップ数(6.1 節): 総学習トークン数 = 訓練トークン数 x E ---
TOKENS_PER_STEP = BATCH_SIZE * SEQUENCE_LENGTH


def steps_for(num_train_tokens: int) -> int:
    return (EPOCHS * num_train_tokens) // TOKENS_PER_STEP


NUM_STEPS = steps_for(len(train_ids))
PROD_NUM_STEPS = steps_for(len(train_ids_full))  # 外挿(6.2 節)の目標値。SMOKE_TEST によらない
WARMUP_STEPS = max(1, round(WARMUP_RATIO * NUM_STEPS))
EVAL_INTERVAL = max(1, NUM_STEPS // EVAL_POINTS)
# 資源制約: 全条件がエポック上限 E を満たす(全条件で同じ NUM_STEPS を使う、6.5 節で再確認)
assert EPOCHS * len(train_ids) >= NUM_STEPS * TOKENS_PER_STEP
print(
    f"訓練トークン数: {len(train_ids):,}(全体 {len(train_ids_full):,})/ 較正用検証: "
    f"{len(calibration_ids):,} / 評価: {len(evaluation_ids):,}"
)
print(
    f"NUM_STEPS={NUM_STEPS}(本番 {PROD_NUM_STEPS})、WARMUP_STEPS={WARMUP_STEPS}、"
    f"EVAL_INTERVAL={EVAL_INTERVAL}、学習トークン数 = {NUM_STEPS * TOKENS_PER_STEP:,} "
    f"<= E x 訓練トークン数 = {EPOCHS * len(train_ids):,}"
)
print(
    f"評価の分母: {evaluation_bytes:,} バイト(評価窓 {tuple(evaluation_windows.shape)})/ "
    f"較正の分母: {calibration_bytes:,} バイト(窓 {tuple(calibration_windows.shape)})"
)
```

    raw_text: 1,115,394 文字 / 1,115,394 バイト
    訓練: 1,003,856 文字(全体の訓練区間は 1,003,856 文字)/ 較正用検証: 55,769 文字 / 評価: 55,769 文字
    .cache/012_encoded/train_1.0.u16: 360,004 トークンを uint16 memmap として書き出した
    [キャッシュ] .cache/012_encoded/train_1.0.u16 を再利用する(360,004 トークン、再符号化しない)
    .cache/012_encoded/calibration.u16: 20,781 トークンを uint16 memmap として書き出した
    .cache/012_encoded/evaluation.u16: 20,881 トークンを uint16 memmap として書き出した
    符号化(キャッシュ経由): 0.94s
    キャッシュ経由と直接の符号化の完全一致・符号化 -> 復号のラウンドトリップ: OK(4 分割すべて)
    訓練損失の測定窓: (64, 256)
    訓練トークン数: 360,004(全体 360,004)/ 較正用検証: 20,781 / 評価: 20,881
    NUM_STEPS=87(本番 87)、WARMUP_STEPS=9、EVAL_INTERVAL=10、学習トークン数 = 712,704 <= E x 訓練トークン数 = 720,008
    評価の分母: 55,769 バイト(評価窓 (82, 256))/ 較正の分母: 55,769 バイト(窓 (82, 256))


### 5.5 LoRA の不変条件の確認(`src/layers/lora.py`)

- `apply_lora`直後の出力が、ベースモデルの出力と`torch.equal`で完全に一致すること($B = 0$)。
- 学習可能パラメータ数が閉形式 $r(d_{\mathrm{in}} + d_{\mathrm{out}})$ と一致すること。
- マージ前後の出力が fp32 で許容誤差内で一致し、`unmerge()`後に元の重みへ戻ること。許容誤差は、
  マージが直接作用する各 LoRA 層の出力に`atol=1e-5`を課し、モデル全体の logits には、層を重ねる
  ことによる fp32 の丸め誤差の累積を見込んで`atol=1e-4`を課す。
  学習直後の $B = 0$ のままではマージが自明になるため、ここでは $B$ に乱数を入れて確認し、
  学習後のモデルでも 6.5 節で同じ確認を行う。
- `.weight`を持たないベース層ではマージが例外になること。


```python
_probe_tokens = evaluation_windows[:4].to(device)
with torch.no_grad():
    _base_logits = base_model(_probe_tokens)

_lora_probe = copy.deepcopy(base_model)
torch.manual_seed(0)
_replaced = apply_lora(_lora_probe, LORA_TARGET_MODULES, rank=MAIN_RANK, alpha=LORA_ALPHA)
print("置換したモジュール:", _replaced)
assert len(_replaced) == 2 * NUM_LAYERS
with torch.no_grad():
    _lora_logits = _lora_probe(_probe_tokens)
assert torch.equal(_lora_logits, _base_logits), (
    "apply_lora 直後の出力がベースモデルと完全一致しない"
)
print("apply_lora 直後の出力 == ベースモデルの出力(torch.equal): OK")

_trainable = [p for p in _lora_probe.parameters() if p.requires_grad]
_expected_lora_params = sum(
    compute_lora_parameter_count(
        _lora_probe.get_submodule(n).in_features,
        _lora_probe.get_submodule(n).out_features,
        MAIN_RANK,
    )
    for n in _replaced
)
assert (
    sum(p.numel() for p in _trainable)
    == _expected_lora_params
    == 4 * NUM_LAYERS * MAIN_RANK * D_MODEL
)
print(
    f"LoRA(r={MAIN_RANK})の学習可能パラメータ数: {_expected_lora_params:,}"
    f"(閉形式 4 L r d_model と一致、全パラメータ数の {_expected_lora_params / TOTAL_PARAMETERS:.4%})"
)

MERGE_LAYER_ATOL = 1e-5  # 各 LoRA 層の出力(マージが直接作用する量)の許容誤差
MERGE_MODEL_ATOL = 1e-4  # モデル全体の logits の許容誤差(L 層分の fp32 の丸め誤差の累積を見込む)
UNMERGE_WEIGHT_ATOL = 1e-6  # unmerge 後の重みの許容誤差


def check_merge_roundtrip(model: nn.Module, replaced: list[str], tokens: torch.Tensor) -> dict:
    # マージ前後で、各 LoRA 層の出力とモデル全体の logits が許容誤差内で一致し、
    # unmerge 後に元の重みへ戻ることを確認する。最大絶対誤差を返す。
    generator = torch.Generator().manual_seed(0)
    layer_inputs = {
        n: torch.randn(
            4, SEQUENCE_LENGTH, model.get_submodule(n).in_features, generator=generator
        ).to(device)
        for n in replaced
    }
    with torch.no_grad():
        logits_before = model(tokens)
        layer_before = {n: model.get_submodule(n)(layer_inputs[n]) for n in replaced}
        weights_before = {n: model.get_submodule(n).base_layer.weight.clone() for n in replaced}
        for n in replaced:
            model.get_submodule(n).merge()
        layer_after = {n: model.get_submodule(n)(layer_inputs[n]) for n in replaced}
        logits_after = model(tokens)
        for n in replaced:
            torch.testing.assert_close(
                layer_after[n], layer_before[n], atol=MERGE_LAYER_ATOL, rtol=0
            )
        torch.testing.assert_close(logits_after, logits_before, atol=MERGE_MODEL_ATOL, rtol=0)
        for n in replaced:
            model.get_submodule(n).unmerge()
            torch.testing.assert_close(
                model.get_submodule(n).base_layer.weight,
                weights_before[n],
                atol=UNMERGE_WEIGHT_ATOL,
                rtol=0,
            )
            assert not model.get_submodule(n).merged
    return {
        "max_layer_error": max(
            float((layer_after[n] - layer_before[n]).abs().max()) for n in replaced
        ),
        "max_logits_error": float((logits_after - logits_before).abs().max()),
        "logits_changed_from_base": not torch.equal(logits_before, _base_logits),
    }


# B に乱数を入れて、マージが自明(Delta W = 0)にならないようにする
with torch.no_grad():
    for _n in _replaced:
        _lora_probe.get_submodule(_n).lora_b.normal_(0.0, 0.02)
_merge_check = check_merge_roundtrip(_lora_probe, _replaced, _probe_tokens)
assert _merge_check["logits_changed_from_base"]
print(
    f"マージ前後の一致(各層の出力 atol={MERGE_LAYER_ATOL}、logits atol={MERGE_MODEL_ATOL})・"
    f"unmerge 後に元の重みへ戻ること(atol={UNMERGE_WEIGHT_ATOL}): OK {_merge_check}"
)


class _NoWeightLayer(nn.Module):
    in_features, out_features = 4, 4

    def forward(self, x):
        return x


try:
    LoRALinear(_NoWeightLayer(), rank=2, alpha=2.0).merge()
    raise AssertionError(".weight を持たないベース層のマージが例外にならなかった")
except TypeError as _e:
    print(f".weight を持たないベース層のマージは例外になる: OK({_e})")
del _lora_probe
```

    置換したモジュール: ['blocks.0.self_attn.w_q', 'blocks.0.self_attn.w_v', 'blocks.1.self_attn.w_q', 'blocks.1.self_attn.w_v', 'blocks.2.self_attn.w_q', 'blocks.2.self_attn.w_v', 'blocks.3.self_attn.w_q', 'blocks.3.self_attn.w_v']
    apply_lora 直後の出力 == ベースモデルの出力(torch.equal): OK
    LoRA(r=8)の学習可能パラメータ数: 32,768(閉形式 4 L r d_model と一致、全パラメータ数の 0.6246%)
    マージ前後の一致(各層の出力 atol=1e-05、logits atol=0.0001)・unmerge 後に元の重みへ戻ること(atol=1e-06): OK {'max_layer_error': 5.245208740234375e-06, 'max_logits_error': 1.2874603271484375e-05, 'logits_changed_from_base': True}
    .weight を持たないベース層のマージは例外になる: OK(マージにはベース層が .weight(Tensor)を持つ必要がある: _NoWeightLayer)


### 5.6 ランダムマスク疎微調整層(実験 B の対照、ノートブック内の実装)

対象の行列 $W_0 \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}$ ごとに、
$d_{\mathrm{out}} d_{\mathrm{in}}$ 個の要素から $k = r(d_{\mathrm{in}} + d_{\mathrm{out}})$ 個
($r = 8$、LoRA と同数)の位置 $\mathcal{I}$ をシードごとに固定の乱数で選ぶ。更新量は長さ $k$ の
ベクトル $\delta \in \mathbb{R}^k$(0 で初期化)として持ち、順伝播では

$$
h = \left(W_0 + \mathrm{scatter}_{\mathcal{I}}(\delta)\right) x
$$

を計算する。$\mathrm{scatter}_{\mathcal{I}}(\delta)$ は、位置 $\mathcal{I}$ に $\delta$ の要素を置き、
それ以外を 0 とした $d_{\mathrm{out}} \times d_{\mathrm{in}}$ の行列である。学習可能パラメータは
$\delta$ のみなので、学習可能パラメータ数は文字どおり $k$ になる(マスクを掛けた密な行列を
学習するのではない)。


```python
class RandomMaskSparseLinear(nn.Module):
    # 固定したランダムな位置のみを更新する疎な微調整層(実験 B の対照)。
    def __init__(self, base_layer: nn.Linear, num_updated: int, generator: torch.Generator) -> None:
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)
        numel = base_layer.weight.numel()
        # 位置は CPU の乱数生成器で選ぶ(device によらず同じ位置になる)。
        indices = torch.randperm(numel, generator=generator)[:num_updated].sort().values
        self.register_buffer("indices", indices.to(base_layer.weight.device))
        self.delta = nn.Parameter(torch.zeros(num_updated, device=base_layer.weight.device))

    def delta_weight(self) -> torch.Tensor:
        flat = torch.zeros(
            self.base_layer.weight.numel(), device=self.delta.device, dtype=self.delta.dtype
        )
        return flat.index_add(0, self.indices, self.delta).view_as(self.base_layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.base_layer.weight + self.delta_weight(), self.base_layer.bias)


def apply_random_mask_sparse(
    model: nn.Module, target_module_names, rank: int, seed: int
) -> list[str]:
    # 各対象行列に、LoRA(rank)と同数 k = rank (d_in + d_out) の更新位置を割り当てる。
    generator = torch.Generator().manual_seed(seed)
    targets = set(target_module_names)
    matches = [n for n, m in model.named_modules() if n.rsplit(".", 1)[-1] in targets]
    for p in model.parameters():
        p.requires_grad_(False)
    for name in matches:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        layer = getattr(parent, attr)
        k = compute_lora_parameter_count(layer.in_features, layer.out_features, rank)
        setattr(parent, attr, RandomMaskSparseLinear(layer, k, generator))
    return matches


_sparse_probe = copy.deepcopy(base_model)
_sparse_replaced = apply_random_mask_sparse(_sparse_probe, LORA_TARGET_MODULES, MAIN_RANK, seed=0)
with torch.no_grad():
    assert torch.equal(_sparse_probe(_probe_tokens), _base_logits), (
        "疎微調整層の初期出力がベースと一致しない"
    )
# 学習可能パラメータ数が LoRA(r=8)と行列ごとに完全一致すること
for _n in _sparse_replaced:
    _layer = _sparse_probe.get_submodule(_n)
    assert _layer.delta.numel() == compute_lora_parameter_count(D_MODEL, D_MODEL, MAIN_RANK)
    assert _layer.indices.unique().numel() == _layer.delta.numel()  # 位置の重複がない
assert (
    sum(p.numel() for p in _sparse_probe.parameters() if p.requires_grad) == _expected_lora_params
)
print(
    f"疎微調整: 行列ごとの k = {_sparse_probe.get_submodule(_sparse_replaced[0]).delta.numel():,}"
    f"(密度 {compute_lora_parameter_count(D_MODEL, D_MODEL, MAIN_RANK) / D_MODEL**2:.4%})、"
    f"合計 {_expected_lora_params:,}(LoRA r={MAIN_RANK} と行列ごとに一致)/ 初期出力 == ベース: OK"
)
del _sparse_probe
```

    疎微調整: 行列ごとの k = 4,096(密度 6.2500%)、合計 32,768(LoRA r=8 と行列ごとに一致)/ 初期出力 == ベース: OK


### 5.7 1 条件の学習・評価を行うヘルパー

1 回の学習(1 方式 × 1 シード × 1 学習率)を実行し、判定と診断に必要な量を記録して返す。
モデル自体は(出力例のために残す 2 つを除き)返さず、学習済みの重みは保存しない。

- 学習は`train_language_model()`で行い、評価集合(較正では較正用検証集合)の bits-per-byte を
  `EVAL_INTERVAL`ごとに記録する。最終ステップが記録点に含まれない場合は、学習後に最終ステップの
  値を追加で評価する(判定は最終ステップの値で行う)。
- 訓練損失は、訓練集合から等間隔に選んだ固定の窓(5.4 節)での平均損失(nats / トークン)として、
  学習の前後で同じ関数により測る(6.1 節)。
- 学習後に、凍結したパラメータがすべて学習前(ベースモデル)と bit 単位で一致すること、
  凍結したパラメータの`.grad`が`None`であることを確認する。


```python
def mean_loss_nats(model: nn.Module, windows: torch.Tensor, batch_size: int = 16) -> float:
    # 固定の窓(パディングなし)での平均の次トークン予測損失(nats / トークン)。
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, windows.size(0), batch_size):
            batch = windows[start : start + batch_size].to(device)
            logits = model(batch)[:, :-1, :]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1), reduction="sum"
            )
            total += loss.item()
            count += batch[:, 1:].numel()
    return total / count


BASE_TRAIN_LOSS = mean_loss_nats(base_model, train_loss_windows)  # 学習開始時の訓練損失(全条件共通)
print(f"学習開始時の訓練損失(ベースモデル、固定の訓練窓): {BASE_TRAIN_LOSS:.4f} nats / トークン")

_base_parameter_by_name = dict(base_model.named_parameters())


def _base_name(name: str) -> str:
    # LoRALinear・RandomMaskSparseLinear で包んだ層のパラメータ名を、ベースモデルの名前に戻す。
    return name.replace(".base_layer", "")


def relative_update_norms(model: nn.Module, method: str) -> list[float]:
    # Query・Value の各行列について ||Delta W||_F / ||W_0||_F を返す(層順、Query・Value の順)。
    ratios = []
    for layer_index in range(NUM_LAYERS):
        for attr in LORA_TARGET_MODULES:
            name = f"blocks.{layer_index}.self_attn.{attr}"
            w0 = _base_parameter_by_name[f"{name}.weight"]
            module = model.get_submodule(name)
            if method == "full":
                delta = module.weight.detach() - w0
            else:
                delta = module.delta_weight().detach()
            ratios.append(float(delta.norm() / w0.norm()))
    return ratios


def top_rank_energy(model: nn.Module, rank: int) -> list[float]:
    # 全パラメータ微調整の Delta W について、上位 rank 個の特異値が占めるエネルギー(二乗和)の割合。
    energies = []
    for layer_index in range(NUM_LAYERS):
        for attr in LORA_TARGET_MODULES:
            name = f"blocks.{layer_index}.self_attn.{attr}"
            delta = (
                model.get_submodule(name).weight.detach()
                - _base_parameter_by_name[f"{name}.weight"]
            )
            singular_values = torch.linalg.svdvals(delta.float().cpu())
            energies.append(float((singular_values[:rank] ** 2).sum() / (singular_values**2).sum()))
    return energies


def build_condition_model(method: str, seed: int, rank: int) -> tuple[nn.Module, list[str]]:
    model = copy.deepcopy(base_model)
    if method == "full":
        for p in model.parameters():
            p.requires_grad_(True)
        return model, []
    if method == "lora":
        torch.manual_seed(seed)  # A の初期化
        return model, apply_lora(model, LORA_TARGET_MODULES, rank=rank, alpha=LORA_ALPHA)
    if method == "sparse":
        return model, apply_random_mask_sparse(model, LORA_TARGET_MODULES, MAIN_RANK, seed=seed)
    raise ValueError(f"未知の method: {method!r}")


def run_condition(
    method: str,
    seed: int,
    learning_rate: float,
    clip_threshold: float | None,
    rank: int = MAIN_RANK,
    split: str = "evaluation",
    num_steps: int | None = None,
    keep_model: bool = False,
    measure_diagnostics: bool = True,
    eval_interval: int | None = None,
) -> dict:
    num_steps = NUM_STEPS if num_steps is None else num_steps
    # 既定では学習曲線を EVAL_POINTS 点程度記録する。較正では最終ステップのみ評価する(6.3 節)。
    eval_interval = max(1, num_steps // EVAL_POINTS) if eval_interval is None else eval_interval
    windows, mask, total_bytes = {
        "evaluation": (evaluation_windows, evaluation_mask, evaluation_bytes),
        "calibration": (calibration_windows, calibration_mask, calibration_bytes),
    }[split]
    model, replaced = build_condition_model(method, seed, rank)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=learning_rate, weight_decay=WEIGHT_DECAY)
    schedule = functools.partial(
        compute_warmup_cosine_learning_rate,
        warmup_steps=max(1, round(WARMUP_RATIO * num_steps)),
        total_steps=num_steps,
        peak_learning_rate=learning_rate,
        min_learning_rate=learning_rate * MIN_LEARNING_RATE_RATIO,
    )
    history = train_language_model(
        model,
        train_ids,
        windows,
        mask,
        total_bytes,
        num_steps=num_steps,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        learning_rate=learning_rate,
        eval_interval=eval_interval,
        device=device,
        seed=seed,
        optimizer=optimizer,
        learning_rate_schedule=schedule,
        gradient_clip_threshold=clip_threshold,
    )
    if not history["eval_step"] or history["eval_step"][-1] != num_steps:
        history["eval_step"].append(num_steps)
        history["eval_bits_per_byte"].append(
            evaluate_bits_per_byte(model, windows, mask, total_bytes, device)
        )

    record = {
        "method": method,
        "seed": seed,
        "rank": rank if method == "lora" else None,
        "learning_rate": learning_rate,
        "clip_threshold": clip_threshold,
        "split": split,
        "num_steps": num_steps,
        "history_length": len(history["step"]),
        "history_keys": sorted(history.keys()),
        "num_trainable_parameters": sum(p.numel() for p in trainable),
        "evaluation_denominator_bytes": total_bytes,
        "evaluation_windows_hash": hash_tensor(windows),
        "eval_step": list(history["eval_step"]),
        "eval_bits_per_byte": list(history["eval_bits_per_byte"]),
        "final_bits_per_byte": history["eval_bits_per_byte"][-1],
        "history_first_train_loss": history["train_loss"][0],
        "history_last_train_loss": history["train_loss"][-1],
        "gradient_norm": list(history["gradient_norm"]),
        "clip_trigger_ratio": float(np.mean(history["gradient_clip_triggered"])),
    }

    # 凍結したパラメータが学習前と bit 単位で一致し、.grad が None であること(全条件で確認)
    frozen_checked = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            continue
        assert p.grad is None, f"凍結パラメータ {name} の .grad が None でない"
        assert torch.equal(p, _base_parameter_by_name[_base_name(name)]), (
            f"凍結パラメータ {name} が変化した"
        )
        frozen_checked += 1
    record["num_frozen_tensors_checked"] = frozen_checked

    if measure_diagnostics:
        record["initial_train_loss"] = BASE_TRAIN_LOSS
        record["final_train_loss"] = mean_loss_nats(model, train_loss_windows)
        record["relative_update_norms"] = relative_update_norms(model, method)
        if method == "full":
            record["top_rank_energy"] = top_rank_energy(model, MAIN_RANK)
    if keep_model:
        record["_model"] = model
        record["_replaced"] = replaced
    else:
        del model
    return record


def public(record: dict) -> dict:
    # 印字・保存用(モデル本体を除く)
    return {k: v for k, v in record.items() if not k.startswith("_")}


_t0 = time.time()
_smoke_record = run_condition("lora", seed=0, learning_rate=1e-3, clip_threshold=None, num_steps=2)
print(
    f"run_condition の動作確認(LoRA、2 ステップ): {time.time() - _t0:.2f}s, "
    f"final_bits_per_byte={_smoke_record['final_bits_per_byte']:.4f}, "
    f"凍結テンソルの確認数={_smoke_record['num_frozen_tensors_checked']}"
)
```

    学習開始時の訓練損失(ベースモデル、固定の訓練窓): 6.2466 nats / トークン
    run_condition の動作確認(LoRA、2 ステップ): 0.75s, final_bits_per_byte=3.3886, 凍結テンソルの確認数=38


## 6. 実験 / Experiments



## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/03_efficient_training/012_low_rank_adaptation.ipynb
