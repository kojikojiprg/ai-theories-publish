---
title: "量子化の基礎(Quantization Basics)(実装・実験編 1/5)"
---

この記事は後編(実装・実験編 1/5)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/013_quantization_basics-theory)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/013_quantization_basics-practice-2)。

## 4. 実装方針 / Implementation Policy

**`src/`に切り出す(スクラッチ実装、本トピックで新規作成)**:

- `src/quantization/quantize.py`
  - `quantize_weight()`: absmax 方式(対称)・ゼロ点付き方式(非対称)の一様量子化と NF4 を、
    テンソル単位・チャネル単位・ブロック単位の粒度で行い、`QuantizedWeight`(符号・スケールなど
    格納するテンソル一式)を返す。`QuantizedWeight.dequantize()`で逆量子化する。
  - `compute_nf4_codebook()`: NF4 の 16 個の符号語を標準正規分布の分位点(`torch.special.ndtri`)から
    導出する。一次情報から転記した参照値`NF4_CODEBOOK_REFERENCE`(QLoRA の Appendix E、公式実装の
    `bitsandbytes/functional.py`)を定数として持つ。
  - `double_quantize_scale()`: 二重量子化(平均を引いて 8 ビット・ブロック 256)。
  - `pack_4bit()`・`unpack_4bit()`: 4 ビットの符号を 2 個ずつ 1 バイトに詰める・戻す。
  - `QuantizedWeight.storage_bytes()`: 格納しているテンソルの実測バイト数。
    `compute_quantized_storage_bytes()`・`compute_effective_bits_per_parameter()`: 閉形式。
- `src/quantization/quantized_linear.py`
  - `QuantizedLinear`: パック済みの符号とスケールを buffer として保持し(重みは凍結)、順伝播で
    FP32 に逆量子化して行列積を行う。
  - `quantize_linear_layers()`・`quantize_linear_weights()`: モジュール内の`nn.Linear`を置換する /
    量子化済みの重みをキャッシュとして作る。
- `src/utils/statistics.py`に追加: `compute_excess_kurtosis()`(尖度)、`compute_spearman_correlation()`・
  `compute_stratified_spearman_correlation()`(層別 Spearman 相関)、
  `bootstrap_stratified_spearman_correlation()`(層内で再標本化するブートストラップ)、
  `paired_bootstrap_ratio_of_sums()`(評価窓の対応付きブートストラップ)。
- `src/layers/lora.py`の変更(012 で作成): `LoRALinear`は、ベース層がパラメータを持たない場合に限り、
  浮動小数点の buffer から $A$・$B$ の device・dtype を決めるようにした(`QuantizedLinear`の重みは
  buffer であるため)。パラメータを持つベース層(`nn.Linear`)の挙動は変わらない。012 の時点のコミットを
  `git worktree`で取得したサブプロセスとの数値比較で、既定の挙動が bit 単位で同一であることを確認する
  (5.7 節)。

**ノートブック内に直接書く(013 固有)**:

- 実験の条件定義、評価窓ごとの負の対数尤度の計算、QLoRA の学習・評価ループ(学習そのものは
  `src/training/trainer.py`の`train_language_model()`、optimizer は`src/training/optimizer.py`の`AdamW`、
  学習率スケジュールは`src/training/schedule.py`の warmup + cosine を使う。012 と同じ部品)。
- 外れ値の特徴次元の観察(forward pre-hook による入力活性化の記録)。

**量子化の対象**: 各 Decoder Block 内の`nn.Linear`の重み(注意機構の Query・Key・Value・出力の射影、
SwiGLU の 3 つの行列)。**埋め込み行列(出力層と重みを共有)と正規化層(RMSNorm)の重みは FP32 のまま
残す。** 対象の行列数はモデルの構造から数えて 5.4 節で印字する。スケールは、二重量子化なしの条件では
FP32 とする(QLoRA に合わせる)。

**アップロード方針**: 量子化した重みは`kojikojiprg/ai-theories-small-gpt-en`の`main`から決定的に
再現でき、QLoRA のアダプタは条件比較のためのものであるため、いずれも保存・アップロードしない。

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
    remote: Enumerating objects: 1001, done.[K
    remote: Counting objects: 100% (41/41), done.[K
    remote: Compressing objects: 100% (30/30), done.[K
    remote: Total 1001 (delta 17), reused 29 (delta 11), pack-reused 960 (from 1)[K
    Receiving objects: 100% (1001/1001), 10.07 MiB | 22.82 MiB/s, done.
    Resolving deltas: 100% (578/578), done.
    /content/ai-theories
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m20.5/20.5 MB[0m [31m45.8 MB/s[0m eta [36m0:00:00[0m
    [?25h[2mUsing Python 3.13.15 environment at: /usr[0m
    [2K[2mResolved [1m52 packages[0m [2min 326ms[0m[0m
    [2K[2mPrepared [1m31 packages[0m [2min 44.28s[0m[0m
    [2mUninstalled [1m17 packages[0m [2min 764ms[0m[0m
    [2K[2mInstalled [1m31 packages[0m [2min 330ms[0m[0m
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
import subprocess
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
    load_wikipedia_corpus_with_fallback,
    make_evaluation_windows,
    split_train_val_text,
)
from src.data.tokenizer import load_bpe_id_tokenizer_from_hub
from src.layers.feedforward import SwiGLUFeedForwardNetwork
from src.layers.lora import apply_lora, compute_lora_parameter_count
from src.layers.normalization import RMSNorm
from src.layers.positional_encoding import RotaryPositionEmbedding
from src.models.gpt import GPTLanguageModel
from src.quantization import (
    NF4_CODEBOOK_REFERENCE,
    NF4_OFFSET,
    NF4_THRESHOLDS_REFERENCE,
    QuantizedLinear,
    compute_codebook_thresholds,
    compute_effective_bits_per_parameter,
    compute_nf4_codebook,
    compute_nf4_offset,
    compute_quantized_storage_bytes,
    double_quantize_scale,
    nearest_codebook_index,
    pack_4bit,
    quantize_linear_layers,
    quantize_linear_weights,
    quantize_weight,
    unpack_4bit,
)
from src.training.optimizer import AdamW
from src.training.schedule import compute_warmup_cosine_learning_rate
from src.training.trainer import evaluate_bits_per_byte, train_language_model
from src.utils.reporting import dumps_compact_json
from src.utils.statistics import (
    bootstrap_stratified_spearman_correlation,
    compute_excess_kurtosis,
    compute_stratified_spearman_correlation,
    fit_power_law_exponent,
    paired_bootstrap_ratio_of_sums,
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"torch: {torch.__version__} / device: {device}")

ROOT = Path(".")
TINY_SHAKESPEARE_CACHE_DIR = ROOT / ".cache" / "tiny_shakespeare"  # 外部取得(データ源で命名)
WIKIPEDIA_CACHE_DIR = ROOT / ".cache" / "wikipedia_en"  # Hub 取得に失敗した場合の直接取得先
ENCODED_CACHE_DIR = ROOT / ".cache" / "013_encoded"  # 013 の条件で符号化したトークン列

MODEL_REPO_ID = "kojikojiprg/ai-theories-small-gpt-en"
MODEL_REVISION = "main"
TOKENIZER_REPO_ID = "kojikojiprg/ai-theories-tokenizer-en"
# 006・008 の英語コーパス(356 記事、en_006_pretraining.json)。009 用の
# kojikojiprg/ai-theories-corpus-en(9826 記事)とは別のリポジトリである。
EN_PRETRAINING_CORPUS_REPO_ID = "kojikojiprg/ai-theories-corpus-en-pretraining"
EN_PRETRAINING_MANIFEST = Path("src/data/wikipedia_manifests/en_006_pretraining.json")
EXPECTED_CORPUS_BYTES = 24_331_593  # 010 のセル出力で確認済みの corpus.txt のバイト数
EXPECTED_MANIFEST_ARTICLE_COUNT = 356
LN2 = math.log(2.0)


def hash_tensor(t: torch.Tensor) -> str:
    # テンソルの内容が条件間・実行前後で同一であることを確認するためのハッシュ。
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def hash_state(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(hash_tensor(state[key]).encode())
    return digest.hexdigest()


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

水準の定義をこの 1 箇所(`LEVELS`)に集約する。`SMOKE_TEST = True`はローカルでのコード経路の確認用、
`SMOKE_TEST = False`は Google Colab T4 での本番実行用である。モデル・量子化の条件・評価窓は縮小しない
(実験 A〜C と診断量は重みそのものと英語 Wikipedia の検証集合全体に対する決定的な計算であり、
本番と同じ規模でもローカルで数分で終わるため)。縮小するのは次の 3 つのみである。

- 実験 D の訓練テキスト: 先頭の一部に切り詰める(012 のスモークテストと同じ割合)。エポック数 $E$ と
  ステップ数の決め方は本番と共通なので、「総学習トークン数 = 訓練トークン数 × $E$」という構造は保たれる。
- 実験 D のシード数: 本番 5、スモークテスト 2。条件 1・2 で共通(条件間で揃える値は縮小後も揃える)。
- ブートストラップの反復回数: 本番 10,000、スモークテスト 1,000。すべてのブートストラップで共通。

ブロックサイズの掃引(実験 B の診断量)は本番・スモークテストとも公比 2 の等比($16, 32, \dots, 65536$)
で縮小しない。


```python
SMOKE_TEST = False  # Claude Code はこの True 側のみ実行する(Colab T4 では False に切り替える)
_smoke_tag = "[動作確認のみ、結論ではない] " if SMOKE_TEST else ""

# --- 量子化の条件(本番実行前に宣言し、SMOKE_TEST で変えない) ---
BLOCK_SIZE = 64  # 実験 B・C・D のブロックサイズ(QLoRA と同じ)
BITS_JUDGED_A = (4, 5, 6, 7, 8)  # 実験 A の判定に使うビット幅
BITS_DIAGNOSTIC_A = (2, 3)  # 実験 A の診断量(低ビットで理論から外れることの観察)
EQUIVALENCE_MARGIN_A = math.log(1.1)  # 実験 A の同等性の範囲 [-log 1.1, log 1.1]
BLOCK_SWEEP_B = tuple(2**k for k in range(4, 17))  # 16, 32, ..., 65536(公比 2)
PRECONDITION_B_MIN_GROUPS = 2  # 前提条件 P-B: 形状の群が 2 つ以上
PRECONDITION_B_MIN_GROUP_SIZE = 3  # 前提条件 P-B: 各群の行列数が 3 以上
PRECONDITION_C_MAX_MEDIAN_ABS_KURTOSIS = 1.0  # 前提条件 P-C
P0_REFERENCE_BITS_PER_BYTE = (
    1.6707  # 008 5.10 節のセル出力「最終検証 bits-per-byte = 1.6707」を転記
)
P0_RELATIVE_TOLERANCE = 0.01  # 前提条件 P0 の許容相対誤差(6.1 節)
VALIDATION_RATIO_008 = 0.05  # 008 の訓練・検証の分割比(008 5.1 節)
OUTLIER_MAGNITUDE = 6.0  # LLM.int8() の外れ値の基準(大きさ 6 以上)
OUTLIER_LAYER_FRACTION = 0.25  # 同(全層の 25% 以上)
OUTLIER_POSITION_FRACTION = 0.06  # 同(系列位置の 6% 以上)
OBSERVATION_WINDOWS = 64  # 外れ値の観察に使う評価窓の数(英語 Wikipedia の検証集合の先頭から)

# --- 012 から流用する設定(012 のセル出力から転記。条件 2 のために較正し直さない) ---
EPOCHS = 2
TRAIN_RATIO, CALIBRATION_RATIO, EVALUATION_RATIO = 0.90, 0.05, 0.05
BATCH_SIZE = 32
WARMUP_RATIO = 0.1
MIN_LEARNING_RATE_RATIO = 0.01
WEIGHT_DECAY = 0.0
LORA_TARGET_MODULES = ("w_q", "w_v")
LORA_RANK = 8
LORA_ALPHA = 8.0
LORA_LEARNING_RATE = 0.03162277660168379  # 012 6.3 節の較正結果(10 ** -1.5)
LORA_CLIP_THRESHOLD = 0.28605208992958076  # 012 6.3 節の gradient clipping の閾値
PRECONDITION_TRAIN_LOSS_DROP = 0.05  # 012 の前提条件 B1・C1 と同じ定義(P-D2)
TRAIN_LOSS_WINDOWS = 64
EVAL_POINTS = 8
SESSION_BUDGET_SECONDS = 2 * 60 * 60  # Google Colab 1 セッションの目安予算(006〜012 と同一)

# --- 水準の定義(この 1 箇所に集約する) ---
LEVELS = {
    "smoke": {"TRAIN_TEXT_FRACTION": 0.15, "NUM_SEEDS": 2, "BOOTSTRAP_RESAMPLES": 1_000},
    "prod": {"TRAIN_TEXT_FRACTION": 1.0, "NUM_SEEDS": 5, "BOOTSTRAP_RESAMPLES": 10_000},
}
CURRENT_LEVEL_NAME = "smoke" if SMOKE_TEST else "prod"
CFG = LEVELS[CURRENT_LEVEL_NAME]
TRAIN_TEXT_FRACTION = CFG["TRAIN_TEXT_FRACTION"]
NUM_SEEDS = CFG["NUM_SEEDS"]
BOOTSTRAP_RESAMPLES = CFG["BOOTSTRAP_RESAMPLES"]
SEEDS = tuple(range(NUM_SEEDS))

# 縮小規則の確認(5.2 節)
assert len({BLOCK_SWEEP_B[i + 1] / BLOCK_SWEEP_B[i] for i in range(len(BLOCK_SWEEP_B) - 1)}) == 1
assert BLOCK_SIZE in BLOCK_SWEEP_B
assert all(lv["NUM_SEEDS"] >= 2 for lv in LEVELS.values()), (
    "シード間の標準偏差には 2 シード以上が必要"
)
assert LEVELS["smoke"]["NUM_SEEDS"] < LEVELS["prod"]["NUM_SEEDS"]

print(f"SMOKE_TEST={SMOKE_TEST}(現在の水準: {CURRENT_LEVEL_NAME!r})")
print(
    f"TRAIN_TEXT_FRACTION={TRAIN_TEXT_FRACTION}, NUM_SEEDS={NUM_SEEDS}(SEEDS={SEEDS}), "
    f"BOOTSTRAP_RESAMPLES={BOOTSTRAP_RESAMPLES:,}"
)
print(
    f"BLOCK_SIZE={BLOCK_SIZE}, BITS_JUDGED_A={BITS_JUDGED_A}, BITS_DIAGNOSTIC_A={BITS_DIAGNOSTIC_A}"
)
print(f"BLOCK_SWEEP_B={BLOCK_SWEEP_B}")
print(
    f"LoRA: targets={LORA_TARGET_MODULES}, r={LORA_RANK}, alpha={LORA_ALPHA}, "
    f"lr={LORA_LEARNING_RATE:.6e}, clip={LORA_CLIP_THRESHOLD:.6f}(012 から転記)"
)
print(
    f"P0: 期待値 {P0_REFERENCE_BITS_PER_BYTE}(008 から転記)、許容相対誤差 {P0_RELATIVE_TOLERANCE}"
)
```

    SMOKE_TEST=False(現在の水準: 'prod')
    TRAIN_TEXT_FRACTION=1.0, NUM_SEEDS=5(SEEDS=(0, 1, 2, 3, 4)), BOOTSTRAP_RESAMPLES=10,000
    BLOCK_SIZE=64, BITS_JUDGED_A=(4, 5, 6, 7, 8), BITS_DIAGNOSTIC_A=(2, 3)
    BLOCK_SWEEP_B=(16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
    LoRA: targets=('w_q', 'w_v'), r=8, alpha=8.0, lr=3.162278e-02, clip=0.286052(012 から転記)
    P0: 期待値 1.6707(008 から転記)、許容相対誤差 0.01


### 5.3 基盤モデル・トークナイザの取得(008 がアップロードしたもの)

基盤モデルは`kojikojiprg/ai-theories-small-gpt-en`の`main`ブランチ、トークナイザは
`kojikojiprg/ai-theories-tokenizer-en`から取得する。モデルの構成は取得した`config.json`から読む。


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
SEQUENCE_LENGTH = MODEL_CONFIG["sequence_length"]


def build_base_model_from_hub() -> GPTLanguageModel:
    model = GPTLanguageModel(
        vocabulary_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=MODEL_CONFIG["d_ff"],
        max_sequence_length=SEQUENCE_LENGTH,
        positional_transform=RotaryPositionEmbedding(
            D_MODEL // NUM_HEADS, max_position=SEQUENCE_LENGTH
        ),
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
    _p.requires_grad_(False)  # base_model 自体は変更しない(各条件は deepcopy から始める)
BASE_STATE_HASH = hash_state(base_model.state_dict())
TOTAL_PARAMETERS = sum(p.numel() for p in base_model.parameters())
print(f"全パラメータ数 P = {TOTAL_PARAMETERS:,}")
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
    全パラメータ数 P = 5,246,208


### 5.4 量子化の対象となる行列

各 Decoder Block 内の`nn.Linear`を、モデルの構造から列挙する(出力層`lm_head`は埋め込み行列と重みを
共有しており、`blocks`の外にあるので含まれない)。


```python
TARGET_NAMES = [
    name for name, module in base_model.blocks.named_modules() if isinstance(module, nn.Linear)
]  # base_model.blocks からの相対名(例: "0.self_attn.w_q")
TARGET_WEIGHTS = {
    name: base_model.blocks.get_submodule(name).weight.detach().cpu().clone()
    for name in TARGET_NAMES
}
TARGET_SHAPES = {name: tuple(w.shape) for name, w in TARGET_WEIGHTS.items()}
assert all(base_model.blocks.get_submodule(n).bias is None for n in TARGET_NAMES)
NUM_TARGET_MATRICES = len(TARGET_NAMES)
TARGET_NUM_ELEMENTS = sum(w.numel() for w in TARGET_WEIGHTS.values())

_by_shape: dict[tuple[int, int], list[str]] = {}
for _n in TARGET_NAMES:
    _by_shape.setdefault(TARGET_SHAPES[_n], []).append(_n)
print(f"量子化の対象: {NUM_TARGET_MATRICES} 行列(要素数の合計 {TARGET_NUM_ELEMENTS:,})")
for _shape, _names in _by_shape.items():
    print(f"  形状 {_shape}: {len(_names)} 行列 例: {_names[:3]}")
# 構造からの期待値: 各層に注意機構の 4 射影 + SwiGLU の 3 行列
assert NUM_TARGET_MATRICES == NUM_LAYERS * (4 + 3), NUM_TARGET_MATRICES

# 量子化しないパラメータ(FP32 のまま残す)
NON_TARGET_PARAMETER_NAMES = [
    n for n, _ in base_model.named_parameters() if not n.startswith("blocks.") or "norm" in n
]
_non_target_count = sum(
    dict(base_model.named_parameters())[n].numel() for n in NON_TARGET_PARAMETER_NAMES
)
assert _non_target_count + TARGET_NUM_ELEMENTS == TOTAL_PARAMETERS
assert base_model.lm_head.weight is base_model.token_embedding.weight
print(
    f"FP32 のまま残すパラメータ: {len(NON_TARGET_PARAMETER_NAMES)} テンソル・{_non_target_count:,} 要素"
    f"(埋め込み行列(出力層と共有)と RMSNorm の重み): {NON_TARGET_PARAMETER_NAMES}"
)
```

    量子化の対象: 28 行列(要素数の合計 3,146,752)
      形状 (256, 256): 16 行列 例: ['0.self_attn.w_q', '0.self_attn.w_k', '0.self_attn.w_v']
      形状 (683, 256): 8 行列 例: ['0.feed_forward.w', '0.feed_forward.v', '1.feed_forward.w']
      形状 (256, 683): 4 行列 例: ['0.feed_forward.w2', '1.feed_forward.w2', '2.feed_forward.w2']
    FP32 のまま残すパラメータ: 10 テンソル・2,099,456 要素(埋め込み行列(出力層と共有)と RMSNorm の重み): ['token_embedding.weight', 'blocks.0.norm1.gamma', 'blocks.0.norm3.gamma', 'blocks.1.norm1.gamma', 'blocks.1.norm3.gamma', 'blocks.2.norm1.gamma', 'blocks.2.norm3.gamma', 'blocks.3.norm1.gamma', 'blocks.3.norm3.gamma', 'final_norm.gamma']


### 5.5 コーパスの取得・分割・符号化

- **英語 Wikipedia の検証集合(前提条件 P0・実験 A〜C の診断量・観察)**: 008 と同じコーパス
  (356 記事、`kojikojiprg/ai-theories-corpus-en-pretraining`)を 008 と同じ比率(末尾 5%)で文字列の段階で
  分割した検証集合。008 と同じ非重複窓(系列長 256)で評価する。
- **Tiny Shakespeare(実験 D)**: 012 と同じく文字列の段階で連続区間に分割する(先頭から訓練 90%・
  較正用検証 5%・評価 5%)。本トピックでは較正を行わない(学習率・gradient clipping の閾値は 012 の
  較正結果を使う)ので、較正用検証集合は使わない。判定はすべて評価集合で行う。

符号化したトークン列は`.cache/013_encoded/`に`uint16`の memmap としてキャッシュし、キャッシュを経由しない
直接の符号化と完全に一致すること、符号化 → 復号のラウンドトリップが一致することを確認する。

**評価窓ごとのバイト数**: トークナイザはバイトレベル BPE なので、各トークンが表す UTF-8 バイト数は
語彙の記号の長さ(1 文字 = 1 バイト)で決まる。評価窓ごとのバイト数(ブートストラップの分母)を
窓内の実トークンのバイト数の和として求め、その総和が評価テキスト全体の UTF-8 バイト数に一致することを
確認する。


```python
# --- 英語 Wikipedia(008 の検証集合) ---
corpus_en, _corpus_meta = load_wikipedia_corpus_with_fallback(
    "en",
    EN_PRETRAINING_CORPUS_REPO_ID,
    WIKIPEDIA_CACHE_DIR,
    manifest_path=EN_PRETRAINING_MANIFEST,
    return_metadata=True,
)
assert _corpus_meta["source"] == "hub", "Hub からの取得に失敗し直接取得にフォールバックした"
assert _corpus_meta["manifest_article_count"] == EXPECTED_MANIFEST_ARTICLE_COUNT
assert _corpus_meta["fetched_article_count"] == EXPECTED_MANIFEST_ARTICLE_COUNT
assert not _corpus_meta["skipped_articles"]
assert len(corpus_en.encode("utf-8")) == EXPECTED_CORPUS_BYTES
assert _corpus_meta["validation_ratio"] == VALIDATION_RATIO_008
_, wiki_val_text = split_train_val_text(corpus_en, VALIDATION_RATIO_008)
del corpus_en
wiki_bytes = len(wiki_val_text.encode("utf-8"))
print(f"英語 Wikipedia の検証集合: {len(wiki_val_text):,} 文字 / {wiki_bytes:,} バイト")

# --- Tiny Shakespeare(012 と同じ分割) ---
raw_text = load_tiny_shakespeare(TINY_SHAKESPEARE_CACHE_DIR)
_n_chars = len(raw_text)
_n_calibration = int(_n_chars * CALIBRATION_RATIO)
_n_evaluation = int(_n_chars * EVALUATION_RATIO)
CALIBRATION_START = _n_chars - _n_calibration - _n_evaluation
EVALUATION_START = CALIBRATION_START + _n_calibration
train_text_full = raw_text[:CALIBRATION_START]
evaluation_text = raw_text[EVALUATION_START:]
train_text = train_text_full[: int(len(train_text_full) * TRAIN_TEXT_FRACTION)]
print(
    f"Tiny Shakespeare: 訓練 {len(train_text):,} 文字(全体の訓練区間は {len(train_text_full):,} 文字)/ "
    f"評価 {len(evaluation_text):,} 文字"
)


def encode_with_cache(name: str, text: str) -> torch.Tensor:
    memmap = encode_text_to_memmap(tokenizer, text, ENCODED_CACHE_DIR / f"{name}.u16")
    return torch.from_numpy(np.asarray(memmap, dtype=np.int64))


_t0 = time.time()
wiki_val_ids = encode_with_cache("wikipedia_validation", wiki_val_text)
train_ids = encode_with_cache(f"shakespeare_train_{TRAIN_TEXT_FRACTION}", train_text)
train_ids_full = encode_with_cache("shakespeare_train_1.0", train_text_full)
evaluation_ids = encode_with_cache("shakespeare_evaluation", evaluation_text)
print(f"符号化(キャッシュ経由): {time.time() - _t0:.2f}s")
for _name, _text, _ids in (
    ("wikipedia_validation", wiki_val_text, wiki_val_ids),
    ("shakespeare_train", train_text, train_ids),
    ("shakespeare_train_full", train_text_full, train_ids_full),
    ("shakespeare_evaluation", evaluation_text, evaluation_ids),
):
    assert torch.equal(encode_corpus(tokenizer, _text), _ids), (
        f"{_name}: キャッシュと直接の符号化が不一致"
    )
    assert tokenizer.decode(_ids.tolist()) == _text, (
        f"{_name}: 符号化 -> 復号のラウンドトリップが不一致"
    )
print("キャッシュ経由と直接の符号化の完全一致・ラウンドトリップ: OK(4 分割すべて)")

wiki_windows, wiki_mask = make_evaluation_windows(wiki_val_ids, SEQUENCE_LENGTH)
evaluation_windows, evaluation_mask = make_evaluation_windows(evaluation_ids, SEQUENCE_LENGTH)
evaluation_bytes = len(evaluation_text.encode("utf-8"))
# 分母の期待値は、評価ループに渡す変数とは独立に、生テキストと分割位置から直接計算する
EXPECTED_EVALUATION_BYTES = len(raw_text.encode("utf-8")) - len(
    raw_text[:EVALUATION_START].encode("utf-8")
)
assert evaluation_bytes == EXPECTED_EVALUATION_BYTES
WIKI_WINDOWS_HASH = hash_tensor(wiki_windows)
EVALUATION_WINDOWS_HASH = hash_tensor(evaluation_windows)

# --- 評価窓ごとの UTF-8 バイト数(ブートストラップの分母) ---
assert tokenizer.bpe_tokenizer.byte_level, "バイトレベル BPE であることを前提とする"
TOKEN_BYTE_LENGTHS = torch.tensor(
    [len(tokenizer.id_to_symbol[i]) for i in range(VOCAB_SIZE)], dtype=torch.long
)


def window_byte_counts(windows: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    return (TOKEN_BYTE_LENGTHS[windows] * mask).sum(dim=1).numpy().astype(np.float64)


wiki_window_bytes = window_byte_counts(wiki_windows, wiki_mask)
evaluation_window_bytes = window_byte_counts(evaluation_windows, evaluation_mask)
assert int(wiki_window_bytes.sum()) == wiki_bytes
assert int(evaluation_window_bytes.sum()) == evaluation_bytes
print(
    f"評価窓: 英語 Wikipedia {tuple(wiki_windows.shape)}({wiki_bytes:,} バイト)/ "
    f"Tiny Shakespeare {tuple(evaluation_windows.shape)}({evaluation_bytes:,} バイト)"
)
print("評価窓ごとのバイト数の総和 == 評価テキスト全体の UTF-8 バイト数: OK(2 集合とも)")

# --- 実験 D のステップ数(012 と同じ決め方) ---
TOKENS_PER_STEP = BATCH_SIZE * SEQUENCE_LENGTH
NUM_STEPS = (EPOCHS * len(train_ids)) // TOKENS_PER_STEP
PROD_NUM_STEPS = (EPOCHS * len(train_ids_full)) // TOKENS_PER_STEP
EVAL_INTERVAL = max(1, NUM_STEPS // EVAL_POINTS)
assert EPOCHS * len(train_ids) >= NUM_STEPS * TOKENS_PER_STEP
print(f"NUM_STEPS={NUM_STEPS}(本番 {PROD_NUM_STEPS})、EVAL_INTERVAL={EVAL_INTERVAL}")

# 訓練損失の測定用: 訓練集合の非重複窓から、パディングを含まない窓を等間隔に選ぶ(012 と同じ)
_train_windows_all, _train_mask_all = make_evaluation_windows(train_ids, SEQUENCE_LENGTH)
_full_windows = torch.nonzero(_train_mask_all.all(dim=1)).flatten()
train_loss_windows = _train_windows_all[
    _full_windows[
        torch.linspace(0, len(_full_windows) - 1, min(TRAIN_LOSS_WINDOWS, len(_full_windows)))
        .round()
        .long()
    ]
]
print(f"訓練損失の測定窓: {tuple(train_loss_windows.shape)}")
```


    corpus.txt: reconstructing file:   0%|          |  0.00B / 24.3MB            



    corpus.txt: downloading bytes:           |  0.00B            



    metadata.json:   0%|          | 0.00/270 [00:00<?, ?B/s]


    コーパス取得元: kojikojiprg/ai-theories-corpus-en-pretraining(Hugging Face Hub)
    英語 Wikipedia の検証集合: 1,210,727 文字 / 1,214,117 バイト
    Tiny Shakespeare: 訓練 1,003,856 文字(全体の訓練区間は 1,003,856 文字)/ 評価 55,769 文字
    .cache/013_encoded/wikipedia_validation.u16: 317,201 トークンを uint16 memmap として書き出した
    .cache/013_encoded/shakespeare_train_1.0.u16: 360,004 トークンを uint16 memmap として書き出した
    [キャッシュ] .cache/013_encoded/shakespeare_train_1.0.u16 を再利用する(360,004 トークン、再符号化しない)
    .cache/013_encoded/shakespeare_evaluation.u16: 20,881 トークンを uint16 memmap として書き出した
    符号化(キャッシュ経由): 5.88s
    キャッシュ経由と直接の符号化の完全一致・ラウンドトリップ: OK(4 分割すべて)
    評価窓: 英語 Wikipedia (1240, 256)(1,214,117 バイト)/ Tiny Shakespeare (82, 256)(55,769 バイト)
    評価窓ごとのバイト数の総和 == 評価テキスト全体の UTF-8 バイト数: OK(2 集合とも)
    NUM_STEPS=87(本番 87)、EVAL_INTERVAL=10
    訓練損失の測定窓: (64, 256)


### 5.6 量子化の不変条件の確認(`src/quantization/`)

- **NF4 の符号語**: `compute_nf4_codebook()`が導出した 16 値が、一次情報から転記した参照値
  (QLoRA の Appendix E、公式実装の`get_4bit_type("nf4")`)と`torch.equal`で一致すること。符号語が
  正側 8 個・負側 7 個・0 が 1 個であること。$\delta$ の閉形式を小数第 7 位で丸めると公式実装の既定値になること。
- **パック・アンパック**: 256 通りのバイト値すべてと乱数の符号列で、ラウンドトリップが完全一致すること。
- **最近傍の符号語**: 二分探索(`nearest_codebook_index`)の 15 個の閾値(隣り合う符号語の中点を FP64 で
  計算して FP32 に丸めた値)が、公式実装の`dQuantizeNF4`(`csrc/kernels.cu`)の閾値と`torch.equal`で一致すること。
  境界の規則も公式実装と同じ(閾値より大きければ上側の符号語、閾値ちょうどは下側)であること。乱数の値で、
  全符号語との距離の`argmin`と一致すること(中点ちょうどの値では、FP32 の距離の計算の丸めにより`argmin`が
  1 ulp の差で上側を選ぶことがあるため、`argmin`との比較は乱数の値に限る)。
- **ベクトル化した量子化が、ブロックごとのループで書いた素朴な参照実装と一致すること**(対象の行列で確認)。
- **格納バイト数**: `QuantizedLinear.storage_bytes()`(実測)が、手で計算して定数として書いた期待値と、
  閉形式`compute_quantized_storage_bytes()`の両方に一致すること(期待値は実測値と同じ変数から計算しない)。

| 条件 | 符号 | スケール |
|---|---|---|
| INT8・チャネル単位 | $N$ バイト | $4 d_{\mathrm{out}}$ バイト |
| INT8・テンソル単位 | $N$ | $4$ |
| INT4・テンソル単位 | $N/2$ | $4$ |
| INT4・ブロック 64 / NF4・ブロック 64 | $N/2$ | $4 \lceil N/64 \rceil$ |
| NF4・ブロック 64 + 二重量子化 | $N/2$ | $\lceil N/64 \rceil + 4 \lceil N/(64 \cdot 256) \rceil + 4$ |

(二重量子化の $+4$ はテンソルごとの平均 $\mu$、$\lceil \cdot \rceil$ の第 2 項は第 2 段の FP32 スケール。)


```python
# --- NF4 の符号語 ---
NF4_CODEBOOK = compute_nf4_codebook()
assert torch.equal(NF4_CODEBOOK, torch.tensor(NF4_CODEBOOK_REFERENCE, dtype=torch.float32)), (
    "導出した NF4 の符号語が参照値と一致しない"
)
assert round(compute_nf4_offset(), 7) == NF4_OFFSET
assert int((NF4_CODEBOOK > 0).sum()) == 8 and int((NF4_CODEBOOK < 0).sum()) == 7
assert int((NF4_CODEBOOK == 0).sum()) == 1 and NF4_CODEBOOK.min() == -1 and NF4_CODEBOOK.max() == 1
print(f"delta の閉形式 {compute_nf4_offset():.10f} -> 小数第 7 位で丸めて {NF4_OFFSET}")
print("NF4 の符号語(導出値、参照値と torch.equal で一致):")
print("  " + ", ".join(f"{v:+.6f}" for v in NF4_CODEBOOK.tolist()))

# --- パック・アンパックのラウンドトリップ ---
_all_bytes = torch.arange(256, dtype=torch.uint8)
assert torch.equal(pack_4bit(unpack_4bit(_all_bytes, 512)), _all_bytes)
_gen = torch.Generator().manual_seed(0)
for _n in (1, 2, 1001, 65536):
    _codes = torch.randint(0, 16, (_n,), generator=_gen).to(torch.uint8)
    assert torch.equal(unpack_4bit(pack_4bit(_codes), _n), _codes)
    assert pack_4bit(_codes).numel() == -(-_n // 2)
print("パック・アンパックのラウンドトリップ(256 通りのバイト値・乱数の符号列、奇数長を含む): OK")

# --- 最近傍の符号語: 閾値が公式実装と一致し、境界の規則・乱数での argmin との一致を確認 ---
_thresholds = compute_codebook_thresholds(NF4_CODEBOOK)
assert torch.equal(_thresholds, torch.tensor(NF4_THRESHOLDS_REFERENCE, dtype=torch.float32))
assert torch.equal(
    nearest_codebook_index(_thresholds, NF4_CODEBOOK), torch.arange(15)
)  # 閾値ちょうど -> 下側
_above = torch.nextafter(_thresholds, torch.tensor(2.0))
assert torch.equal(
    nearest_codebook_index(_above, NF4_CODEBOOK), torch.arange(1, 16)
)  # 1 ulp 上 -> 上側
assert torch.equal(nearest_codebook_index(NF4_CODEBOOK, NF4_CODEBOOK), torch.arange(16))
_values = torch.rand(1_000_000, generator=_gen) * 2 - 1
assert torch.equal(
    nearest_codebook_index(_values, NF4_CODEBOOK),
    (_values[:, None] - NF4_CODEBOOK).abs().argmin(dim=1),
)
print(
    "最近傍の符号語: 15 個の閾値が公式実装(dQuantizeNF4)と一致、閾値ちょうどは下側・1 ulp 上は上側、"
    "乱数 1,000,000 値で argmin と一致: OK"
)


# --- ベクトル化した量子化 vs ブロックごとのループの参照実装(対象の各形状の先頭の行列で確認) ---
def reference_block_dequantize(w: torch.Tensor, method: str, bits: int, block: int) -> torch.Tensor:
    flat, out = w.flatten(), torch.empty(w.numel())
    for s in range(0, flat.numel(), block):
        blk = flat[s : s + block]
        m = blk.abs().max()
        if method == "absmax":
            d = m / (2 ** (bits - 1) - 1)
            out[s : s + block] = (
                torch.round(blk / d).clamp(-(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1) * d
            )
        else:
            out[s : s + block] = (
                NF4_CODEBOOK[((blk / m)[:, None] - NF4_CODEBOOK).abs().argmin(1)] * m
            )
    return out.view(w.shape)


for _names in _by_shape.values():
    _w = TARGET_WEIGHTS[_names[0]]
    for _method, _bits in (("absmax", 4), ("absmax", 8), ("nf4", 4)):
        _q = quantize_weight(
            _w, method=_method, bits=_bits, granularity="block", block_size=BLOCK_SIZE
        )
        torch.testing.assert_close(
            _q.dequantize(),
            reference_block_dequantize(_w, _method, _bits, BLOCK_SIZE),
            rtol=0,
            atol=1e-7,
        )
    _scale = quantize_weight(
        _w, method="nf4", bits=4, granularity="block", block_size=BLOCK_SIZE
    ).scale
    _dq = double_quantize_scale(_scale)
    _mu, _ref = _scale.mean(), torch.empty_like(_scale)
    for s in range(0, _scale.numel(), 256):
        _blk = _scale[s : s + 256] - _mu
        _d = _blk.abs().max() / 127
        _ref[s : s + 256] = torch.round(_blk / _d).clamp(-127, 127) * _d + _mu
    torch.testing.assert_close(_dq.dequantize(), _ref, rtol=0, atol=1e-8)
print("ベクトル化した量子化(absmax 4/8 ビット・NF4・二重量子化)と素朴な参照実装の一致: OK(3 形状)")
```

    delta の閉形式 0.9677083333 -> 小数第 7 位で丸めて 0.9677083
    NF4 の符号語(導出値、参照値と torch.equal で一致):
      -1.000000, -0.696193, -0.525073, -0.394917, -0.284441, -0.184773, -0.091050, +0.000000, +0.079580, +0.160930, +0.246112, +0.337915, +0.440710, +0.562617, +0.722957, +1.000000
    パック・アンパックのラウンドトリップ(256 通りのバイト値・乱数の符号列、奇数長を含む): OK
    最近傍の符号語: 15 個の閾値が公式実装(dQuantizeNF4)と一致、閾値ちょうどは下側・1 ulp 上は上側、乱数 1,000,000 値で argmin と一致: OK
    ベクトル化した量子化(absmax 4/8 ビット・NF4・二重量子化)と素朴な参照実装の一致: OK(3 形状)



```python
# --- 量子化の条件(実験 A〜C の診断量・実験 D) ---
QUANT_CONDITIONS = {
    "q0": None,  # FP32(量子化なし)
    "q1": dict(method="absmax", bits=8, granularity="channel"),
    "q2": dict(method="absmax", bits=8, granularity="tensor"),
    "q3": dict(method="absmax", bits=4, granularity="tensor"),
    "q4": dict(method="absmax", bits=4, granularity="block", block_size=BLOCK_SIZE),
    "q5": dict(method="nf4", bits=4, granularity="block", block_size=BLOCK_SIZE),
    "q6": dict(
        method="nf4", bits=4, granularity="block", block_size=BLOCK_SIZE, double_quantization=True
    ),
}
QUANT_LABELS = {
    "q0": "FP32",
    "q1": "INT8 (per-channel)",
    "q2": "INT8 (per-tensor)",
    "q3": "INT4 (per-tensor)",
    "q4": "INT4 (block 64)",
    "q5": "NF4 (block 64)",
    "q6": "NF4 (block 64 + DQ)",
}

# --- 格納バイト数の期待値(手で計算した定数。実測値・閉形式と同じ変数から計算しない) ---
# 形状 (256, 256): N = 65536、(683, 256): N = 174848、(256, 683): N = 174848
EXPECTED_STORAGE_BYTES = {
    "q0": {(256, 256): 262_144, (683, 256): 699_392, (256, 683): 699_392},  # FP32: 4N
    "q1": {(256, 256): 66_560, (683, 256): 177_580, (256, 683): 175_872},  # N + 4 d_out
    "q2": {(256, 256): 65_540, (683, 256): 174_852, (256, 683): 174_852},  # N + 4
    "q3": {(256, 256): 32_772, (683, 256): 87_428, (256, 683): 87_428},  # N/2 + 4
    "q4": {(256, 256): 36_864, (683, 256): 98_352, (256, 683): 98_352},  # N/2 + 4 N/64
    "q5": {(256, 256): 36_864, (683, 256): 98_352, (256, 683): 98_352},  # 同上
    # N/2 + N/64 + 4 ceil(N/64/256) + 4: 32768 + 1024 + 16 + 4 / 87424 + 2732 + 44 + 4
    "q6": {(256, 256): 33_812, (683, 256): 90_204, (256, 683): 90_204},
}
assert set(TARGET_SHAPES.values()) == set(EXPECTED_STORAGE_BYTES["q0"]), "想定と異なる形状がある"

# 量子化済みの重みのキャッシュ(条件・シードをまたいで再利用する、5.8 節で一致を確認)
_t0 = time.time()
QUANTIZED_CACHE = {
    cid: quantize_linear_weights(base_model.blocks, **cfg)
    for cid, cfg in QUANT_CONDITIONS.items()
    if cfg is not None
}
print(f"量子化(6 条件 x {NUM_TARGET_MATRICES} 行列): {time.time() - _t0:.2f}s")

storage_table: dict[str, int] = {}
for _cid, _cfg in QUANT_CONDITIONS.items():
    _total = 0
    for _name in TARGET_NAMES:
        _shape = TARGET_SHAPES[_name]
        _n = _shape[0] * _shape[1]
        if _cfg is None:
            _measured = TARGET_WEIGHTS[_name].numel() * TARGET_WEIGHTS[_name].element_size()
        else:
            _measured = QuantizedLinear(QUANTIZED_CACHE[_cid][_name]).storage_bytes()
            _q = QUANTIZED_CACHE[_cid][_name]
            _closed = compute_quantized_storage_bytes(
                _n,
                _q.bits,
                _q.num_groups,
                double_quantization_block_size=256 if _cfg.get("double_quantization") else None,
            )
            assert _closed == EXPECTED_STORAGE_BYTES[_cid][_shape], (_cid, _shape, _closed)
        assert _measured == EXPECTED_STORAGE_BYTES[_cid][_shape], (_cid, _name, _measured)
        _total += _measured
    storage_table[_cid] = _total
print("格納バイト数: 実測値 == 手計算の期待値 == 閉形式(6 条件 x 28 行列): OK")
print(f"{'条件':<22} | {'格納バイト数':>12} | {'FP32 比':>8} | {'実効ビット数 / パラメータ':>24}")
for _cid, _total in storage_table.items():
    print(
        f"{QUANT_LABELS[_cid]:<22} | {_total:>12,} | {_total / storage_table['q0']:>8.4f} | "
        f"{8 * _total / TARGET_NUM_ELEMENTS:>24.4f}"
    )

# 実効ビット数の閉形式(3.4 節・3.7 節)
assert compute_effective_bits_per_parameter(4, 64) == 4.5
assert (
    compute_effective_bits_per_parameter(4, 64, double_quantization_block_size=256) == 4.126953125
)
print("実効ビット数の閉形式: ブロック 64・FP32 スケール 4.5、二重量子化 4.126953125: OK")
```

    量子化(6 条件 x 28 行列): 0.31s
    格納バイト数: 実測値 == 手計算の期待値 == 閉形式(6 条件 x 28 行列): OK
    条件                     |       格納バイト数 |   FP32 比 |           実効ビット数 / パラメータ
    FP32                   |   12,587,008 |   1.0000 |                  32.0000
    INT8 (per-channel)     |    3,189,088 |   0.2534 |                   8.1076
    INT8 (per-tensor)      |    3,146,864 |   0.2500 |                   8.0003
    INT4 (per-tensor)      |    1,573,488 |   0.1250 |                   4.0003
    INT4 (block 64)        |    1,770,048 |   0.1406 |                   4.5000
    NF4 (block 64)         |    1,770,048 |   0.1406 |                   4.5000
    NF4 (block 64 + DQ)    |    1,623,440 |   0.1290 |                   4.1273
    実効ビット数の閉形式: ブロック 64・FP32 スケール 4.5、二重量子化 4.126953125: OK


### 5.7 `LoRALinear`の変更の後方互換性(012 の時点の実装との bit 単位の一致)

`src/layers/lora.py`を変更する前のコミット(`5143cc4`、012 の最終コミット)を`git worktree`で取得し、
同一の条件(小型 GPT・`apply_lora`・AdamW・warmup + cosine・gradient clipping・CPU)で、その worktree の
`src`と現行の`src`の両方をサブプロセスとして実行する(`sys.executable`で起動するので同一の環境)。
学習履歴・学習後の LoRA の全テンソル・学習後の logits が完全に一致することを確認する。参照側で
読み込まれた`src`のパスが worktree 配下であること、参照側の`LoRALinear.__init__`が buffer を
参照しない(変更前の実装である)ことも確認する。


```python
_REFERENCE_COMMIT = "5143cc4"  # src/layers/lora.py を 013 で変更する前の最後のコミット
_WORKTREE_DIR = ROOT / ".cache" / "_013_lora_backward_compat_worktree"
_CORPUS_CACHE_DIR_ABS = str(TINY_SHAKESPEARE_CACHE_DIR.resolve())

_compat_script = """
import functools
import hashlib
import inspect
import json
import sys

import torch

from src.data.text import (
    CharacterLevelTokenizer, encode_corpus, load_tiny_shakespeare,
    make_evaluation_windows, split_train_val_text,
)
from src.layers.feedforward import SwiGLUFeedForwardNetwork
from src.layers.lora import LoRALinear, apply_lora
from src.layers.normalization import RMSNorm
from src.layers.positional_encoding import RotaryPositionEmbedding
from src.models.gpt import GPTLanguageModel
from src.training.optimizer import AdamW
from src.training.schedule import compute_warmup_cosine_learning_rate
from src.training.trainer import train_language_model

print("SRC_FILE:" + __import__("src").__file__, file=sys.stderr)
print("USES_BUFFERS:" + str("buffers()" in inspect.getsource(LoRALinear.__init__)), file=sys.stderr)

device = torch.device("cpu")
raw = load_tiny_shakespeare(sys.argv[1])
tok = CharacterLevelTokenizer(raw)
train_text, val_text = split_train_val_text(raw, 0.05)
train_ids = encode_corpus(tok, train_text)
eval_windows, eval_mask = make_evaluation_windows(encode_corpus(tok, val_text), 64)
total_eval_bytes = len(val_text.encode("utf-8"))

torch.manual_seed(0)
model = GPTLanguageModel(
    tok.vocab_size, 64, 2, 2, 128, 64,
    positional_transform=RotaryPositionEmbedding(32, max_position=64),
    normalization_factory=RMSNorm,
    feed_forward_factory=functools.partial(SwiGLUFeedForwardNetwork, 64, 85),
    tie_embeddings=True,
).to(device)
torch.manual_seed(1)
replaced = apply_lora(model, ("w_q", "w_v"), rank=4, alpha=4.0)
trainable = [p for p in model.parameters() if p.requires_grad]
opt = AdamW(trainable, lr=1e-2, weight_decay=0.0)
sched = functools.partial(
    compute_warmup_cosine_learning_rate, warmup_steps=2, total_steps=20,
    peak_learning_rate=1e-2, min_learning_rate=1e-4,
)
hist = train_language_model(
    model, train_ids, eval_windows, eval_mask, total_eval_bytes, num_steps=20, batch_size=8,
    sequence_length=64, learning_rate=1e-2, eval_interval=10, device=device, seed=0,
    optimizer=opt, learning_rate_schedule=sched, gradient_clip_threshold=0.5,
)
with torch.no_grad():
    logits = model(eval_windows[:2])
def h(t):
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()
out = {k: hist[k] for k in ("train_loss", "gradient_norm", "gradient_clip_triggered", "eval_bits_per_byte")}
out["replaced"] = replaced
out["lora_state"] = {k: h(v) for k, v in model.state_dict().items() if "lora_" in k}
out["full_state"] = {k: h(v) for k, v in model.state_dict().items()}
out["logits"] = h(logits)
print(json.dumps(out))
"""

subprocess.run(["git", "worktree", "remove", "--force", str(_WORKTREE_DIR)], capture_output=True)
_r = subprocess.run(
    ["git", "worktree", "add", "--detach", str(_WORKTREE_DIR), _REFERENCE_COMMIT],
    capture_output=True,
    text=True,
)
assert _r.returncode == 0, _r.stderr
(_WORKTREE_DIR / "_compat_script.py").write_text(_compat_script)
(ROOT / "_compat_script.py").write_text(_compat_script)
try:
    _ref = subprocess.run(
        [sys.executable, "_compat_script.py", _CORPUS_CACHE_DIR_ABS],
        cwd=_WORKTREE_DIR,
        capture_output=True,
        text=True,
    )
    assert _ref.returncode == 0, _ref.stderr
    assert "SRC_FILE:" + str((_WORKTREE_DIR / "src" / "__init__.py").resolve()) in _ref.stderr, (
        _ref.stderr
    )
    assert "USES_BUFFERS:False" in _ref.stderr, "参照側が変更後の LoRALinear を読み込んでいる"
    _new = subprocess.run(
        [sys.executable, "_compat_script.py", _CORPUS_CACHE_DIR_ABS], capture_output=True, text=True
    )
    assert _new.returncode == 0, _new.stderr
    assert "SRC_FILE:" + str((ROOT / "src" / "__init__.py").resolve()) in _new.stderr, _new.stderr
    assert "USES_BUFFERS:True" in _new.stderr, "現行側が変更後の LoRALinear を読み込んでいない"
    _ref_out, _new_out = json.loads(_ref.stdout), json.loads(_new.stdout)
    for _k in _ref_out:
        print(f"  {_k}: {'OK' if _ref_out[_k] == _new_out[_k] else 'MISMATCH'}")
    assert _ref_out == _new_out, "012 の時点の LoRALinear と数値的に一致しない"
    print(
        f"参照コミット {_REFERENCE_COMMIT}(worktree 配下の src を読み込んだことを確認)と現行の src で、"
        "学習履歴・LoRA のテンソル・全 state・logits が完全一致: OK"
    )
finally:
    (ROOT / "_compat_script.py").unlink(missing_ok=True)
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(_WORKTREE_DIR)], capture_output=True
    )
```

      train_loss: OK
      gradient_norm: OK
      gradient_clip_triggered: OK
      eval_bits_per_byte: OK
      replaced: OK
      lora_state: OK
      full_state: OK
      logits: OK
    参照コミット 5143cc4(worktree 配下の src を読み込んだことを確認)と現行の src で、学習履歴・LoRA のテンソル・全 state・logits が完全一致: OK




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/03_efficient_training/013_quantization_basics.ipynb
