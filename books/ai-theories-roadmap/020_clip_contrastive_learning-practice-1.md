---
title: "CLIP と対照学習 / CLIP and Contrastive Learning(実装・実験編 1/5)"
---

この記事は後編(実装・実験編 1/5)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/020_clip_contrastive_learning-theory)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/020_clip_contrastive_learning-practice-2)。

## 4. 実装方針 / Implementation Policy

**`src/`に切り出す(スクラッチ実装、本トピックで新規作成)**:

- `src/data/synthetic_scenes.py`: シーンの意味(`SceneMeaning`)と正準形のキャプション(`caption_text()`)、キャプションを意味に戻す
  `caption_meaning()`(`right of`・`below`の言い換えも正準形に直す。同義の言い換えの判定に使う)、有効な意味の全列挙、未見の組み合わせの
  分割(`holdout_pairs()`・`build_caption_universe()`)、困難な負例の生成規則(`swap_attributes()`・`swap_order()`)、単語単位の
  トークナイザ(`CaptionVocabulary`)、numpy でベクトル化した描画(`render_scenes()`)、バッチ内のキャプションが互いに異なるように
  切り出すサンプラ(`DistinctCaptionBatchSampler`)。
- `src/models/clip.py`: 因果マスクつきのテキスト encoder(`CausalTextTransformer`。002 の`DecoderBlock`を`use_cross_attention=False`で
  積み、終端トークンの位置の特徴を返す)、二重 encoder(`CLIPDualEncoder`。019 の`VisionTransformer`の`forward_features()`を画像側に使い、
  共通の埋め込み次元への線形射影・L2 正規化・温度とバイアスのパラメータを持つ)。
- `src/training/contrastive.py`: 対称 InfoNCE 損失`softmax_contrastive_loss()`、sigmoid 損失`sigmoid_contrastive_loss()`、NegCLIP の損失
  `negclip_loss()`、バッチ内のマージン`in_batch_margin()`、学習ループ`train_contrastive_model()`(困難な負例として使ってよい
  キャプションを`hard_negative_allowed`で指定し、除外した組を含むキャプションはテキスト encoder に入れない。学習中にテキスト encoder に
  入れたキャプションの集合を記録する)、評価(`evaluate_retrieval()`・
  `evaluate_two_alternative()`・`modality_gap()`)。

**既存モジュールの変更**(いずれも既定の挙動は変えない):

- `src/models/vit.py`: `VisionTransformer`に、分類ヘッドの直前の表現 $y = \mathrm{LN}(z_L^0)$ を返す`forward_features()`を追加し、
  `forward()`はその出力に分類ヘッドを掛ける形にした。変更前のコミットを`git worktree`で取得し、同一環境のサブプロセスで出力・注意の
  重み・勾配・`state_dict`が bit 単位で一致することを 5.4 節で確かめる(期待値はハードコードしない)。
- `src/training/optimizer.py`: `AdamW`に`foreach`引数(既定値`False`)を追加した。`True`のとき、同じ更新式を`torch._foreach_*`で
  全パラメータにまとめて適用する。本トピックのモデルは小さく、パラメータの数だけ小さな演算を起動する既定の経路では、起動の待ち時間が
  1 ステップの時間の大半を占めるためである。既定の経路が変更前と bit 単位で一致すること(上の`git worktree`の比較に含める)と、
  `foreach=True`の結果が既定の経路と CPU で bit 単位で一致することを 5.4 節で確かめる。

- `src/training/contrastive.py`(同じトピックの中での変更、停止した本番の実行への対応): 学習全体のバッチの添字を事前に作る
  `prepare_batch_indices()`、NegCLIP の困難な負例の固定長化(`negclip_loss()`の`hard_negative_mask`)、CUDA graph による順伝播・逆伝播・
  clipping の再生(`train_contrastive_model(use_cuda_graph=True)`)。変更前のコミット(`81adad2`)との学習の一致を 5.4 節で確かめる。

**ノートブック内に直接書く(020 固有)**: 描画したシーンのディスクキャッシュ(`.cache/020_synthetic_scenes/`)、条件の定義・学習率の
較正・判定・可視化・スケーリングの計測、アップロードのセル。

**アップロード方針**: 実験 C の NegCLIP・シード 0 のモデルのみを`kojikojiprg/ai-theories-clip-synthetic-scenes`にアップロードする
([021](https://github.com/kojikojiprg/ai-theories/blob/main/theories/05_vision_language/021_llava_visual_instruction_tuning.ipynb) の入力になりうるため)。既存のモデルのリポジトリ(小型 GPT など)とモデルの構造が異なるので、別のリポジトリとする。対象は結果を
見て選ばず、この指定で固定する。その他の条件のモデルは条件比較のためのものであり、アップロードしない。トークナイザとデータは
`src/data/synthetic_scenes.py`の規則から決定的に再生成できるので、リポジトリには同梱せず、モデルカードに再生成の方法を書く。

**生成物の置き場所**:

| 生成物 | 置き場所 |
|---|---|
| 描画したシーン(学習用・既知の組み合わせの検証用・未見の組み合わせの評価用の画像) | `.cache/020_synthetic_scenes/`(規則とシードから決定的に再生成できる。セッション内でのみ再利用する) |
| 符号化したキャプション、ランダムな負例の添字、GPU 上の画像のテンソル | メモリ上のみ(固定のシードから決定的に再生成できる) |
| 学習したモデル(アップロードの対象以外) | メモリ上のみ(評価の直後に破棄する) |
| NegCLIP・シード 0 のモデル(`model_state.pt`・`config.json`・`README.md`) | Hugging Face Hub の`kojikojiprg/ai-theories-clip-synthetic-scenes`(`UPLOAD_ARTIFACTS = True`のときのみ) |
| 判定の記録 | 判定と前提条件を計算した各セルの出力 |

**外部からの取得**: なし。データはすべてノートブック内で規則から生成する(Colab のセットアップセルのリポジトリの取得と依存関係の
インストール、6.13 節のアップロードを除き、ネットワークを使わない)。

## 5. 実装 / Implementation

### 5.1 環境セットアップ(Google Colab)

`SMOKE_TEST`(スモークテストか本番か)と`UPLOAD_ARTIFACTS`(Hub へのアップロードをするか)はこのセルでのみ切り替える。2 つは独立で、
本番(`SMOKE_TEST = False`)でも、アップロードは`UPLOAD_ARTIFACTS = True`にしたときだけ行う。

**実行の方式(精度と CUDA graph)**: 学習の 1 ステップの実行の方式を、次の候補から 6.4 節で **時間の見積もりのみに基づいて** 自動で選ぶ
(6.3 節で各方式の時間を計測する)。選ばれた方式は全条件・全シードで同じものを使い、6.4 節で印字する。

| 方式 | 内容 | 候補になる環境 |
|---|---|---|
| `fp16_eager` | encoder の順伝播を FP16 の`torch.autocast`で行い、動的損失スケーリング(011 の`DynamicLossScaler`)を使う。非有限値の検査のため、毎ステップ 1 回ホストとデバイスが同期する | CUDA |
| `fp32_eager` | FP32。1 ステップの中に同期がない | すべて |
| `fp32_graph` | FP32。順伝播・逆伝播・gradient clipping を CUDA graph に記録して再生する(optimizer の更新は記録しない) | CUDA(6.3 節の等価性の確認を通った場合のみ) |

どの方式でも、埋め込みの正規化・類似度・損失は FP32 で計算し、評価は常に FP32 で行う。MPS・CPU では`fp32_eager`のみが候補になる。
方式は計算の丸めを変えうるが、全条件で同じ方式を使うので条件間の比較には影響しない。

**決定性**: `torch.use_deterministic_algorithms(True)`は使わない。同じシードの学習を繰り返しても、CUDA の演算の丸めの順序の違いで
結果が bit 単位では一致しないことがある。条件間の対応付け(同じシードの条件どうしで初期化とバッチの順序を揃えること)は乱数の
生成器によって決まり、演算の決定性には依存しない(6.11 節で確かめる)。


```python
# 環境セットアップ(Google Colab)
import os
import sys
import time

SMOKE_TEST = False  # Claude Code はこの True 側のみ実行する(Colab T4 では False に切り替える)
UPLOAD_ARTIFACTS = True  # True のときのみ NegCLIP・シード 0 のモデルを Hub にアップロードする(SMOKE_TEST とは独立)

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

# 実行の方式の候補(方式 -> (FP16 の autocast と動的損失スケーリングを使うか, CUDA graph を使うか))。6.4 節で選ぶ
EXECUTION_MODES = {"fp16_eager": (True, False), "fp32_eager": (False, False), "fp32_graph": (False, True)}
EXECUTION_MODE_CANDIDATES = list(EXECUTION_MODES) if device.type == "cuda" else ["fp32_eager"]
INIT_LOSS_SCALE = 2.0**16  # torch.amp.GradScaler の既定値と同じ(fp16_eager のみ)
LOSS_SCALE_GROWTH_INTERVAL = 2000  # 同上
print(
    f"SMOKE_TEST={SMOKE_TEST}、UPLOAD_ARTIFACTS={UPLOAD_ARTIFACTS}、実行の方式の候補 {EXECUTION_MODE_CANDIDATES}(6.4 節で選ぶ)、"
    f"動的損失スケーリングは fp16_eager のみ(初期スケール {INIT_LOSS_SCALE:g}、growth_interval {LOSS_SCALE_GROWTH_INTERVAL}、"
    f"backoff 0.5、growth 2.0)、評価は FP32、決定的な演算の強制: {torch.are_deterministic_algorithms_enabled()}"
)
```

    /content/ai-theories
    [2mUsing Python 3.13.15 environment at: /usr[0m
    [2mChecked [1m60 packages[0m [2min 124ms[0m[0m
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
      コミット / git commit                  : 65257e0090da96c671f57d7313872e8a78e53685
      未コミットの変更 / uncommitted changes : なし
      実行日時 (UTC)                         : 2026-09-30T12:51:27+00:00
    SMOKE_TEST=False、UPLOAD_ARTIFACTS=True、実行の方式の候補 ['fp16_eager', 'fp32_eager', 'fp32_graph'](6.4 節で選ぶ)、動的損失スケーリングは fp16_eager のみ(初期スケール 65536、growth_interval 2000、backoff 0.5、growth 2.0)、評価は FP32、決定的な演算の強制: False



```python
import hashlib
import itertools
import json
import math
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.nn import functional

from src.data.synthetic_scenes import (
    COLOR_NAMES,
    RELATIONS,
    SHAPES,
    CaptionVocabulary,
    DistinctCaptionBatchSampler,
    SceneSet,
    build_caption_universe,
    caption_meaning,
    is_valid_meaning,
    render_scenes,
    repeat_caption_ids,
)
from src.models.clip import CausalTextTransformer, CLIPDualEncoder, count_parameters_by_part
from src.models.vit import VisionTransformer
from src.training.contrastive import (
    encode_images,
    encode_texts,
    evaluate_retrieval,
    evaluate_two_alternative,
    in_batch_margin,
    modality_gap,
    negclip_loss,
    sigmoid_contrastive_loss,
    softmax_contrastive_loss,
    train_contrastive_model,
)
from src.training.optimizer import AdamW
from src.utils.statistics import fit_power_law_exponent

ROOT = Path.cwd()


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


def sha256_of_state(state: dict, exclude: tuple[str, ...] = ()) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        if name not in exclude:
            digest.update(name.encode())
            digest.update(state[name].detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


precondition_status: dict[str, bool] = {}  # 前提条件の成否(6.1 節で宣言、各節で記録)
```

### 5.2 スケールの設定(`SMOKE_TEST`の配線)

水準の定義をこの 1 箇所に集約する。

**縮小の規則**:

- **縮小するのは、学習で見る事例数 $E$ の候補、シード数、スケーリングの計測点のみとする。**$E$ の候補は本番
  $\{2^{19}, 2^{18}\}$、スモークテスト $\{2^{13}, 2^{12}\}$ で、どちらも公比 2 の 2 点であり、本番とスモークテストの比
  ($2^6 = 64$ 倍)を候補の間で揃える。すべての候補がバッチサイズの最大値 256 で割り切れる(ステップ数 $T_N = E/N$ が整数)。
- **縮小しないもの**: データ(学習用 1,444 キャプション × 32 = 46,208 シーン、既知の組み合わせの検証集合 2,888 枚、未見の組み合わせの
  評価集合 3,184 枚の全件)、バッチサイズの水準 $N \in \{16, 64, 256\}$(公比 4)、損失の水準、モデルの構成、学習率の較正の格子と
  拡張の規則、学習率のスケジュールの形(warmup の比率・最小学習率の比率)、途中の評価の位置(ステップ数に対する比率)、実行計画の表の
  構造と優先順位、前提条件・判定の閾値。
- **シード数**: 本番は 5(削った段階で 3)。スモークテストは 3(削った段階で 2)とする。「削る前 > 削った後 $\ge 2$」の順序関係を保つ。
  標準条件 S(softmax・$N = 256$)は、実験 A の $N = 256$ の softmax と実験 B・C の標準のモデルで共有する。
- **スケーリングの計測点**: 学習は各条件で、eager の方式は本番 32・64・128 ステップ、スモークテスト 8・16・32 ステップ、CUDA graph の方式は
  本番 128・256・512 ステップ(1 ステップが短く、記録の固定費を薄めるため)、スモークテスト 32・64・128 ステップ(どれも公比 2)。
- **テスト専用の上書き**: 環境変数`AI_THEORIES_FORCE_PLAN`(0〜11)が設定されているときのみ、6.4 節で見積もりによる計画の選択の
  代わりにその計画を使う(スモークテストで各計画の経路を確かめるため)。本番では受け付けず、このセルで停止する。

**学習率の較正の方式**(`CALIBRATION_MODE`): 固定の設定ではなく、**選ばれた実行計画から決まる**(6.1 節)。計画 0〜7 は`"all"`
(実験 A で学習する全条件(損失 × バッチサイズ)を本番と同じ規模でそれぞれ較正する)、計画 8〜11 は`"representative"`($N = 256$ の
2 条件(softmax・sigmoid)のみを較正し、他のバッチサイズの学習率をべき乗の規則 $\eta_N = \eta_{256} (N / 256)^{3/4}$ で決める)。


```python
# --- 全水準で共通の定数(本番実行前に宣言し、SMOKE_TEST で変えない) ---
VISION_CONFIG = {
    "image_size": 32, "patch_size": 4, "in_channels": 3, "num_classes": 1,
    "d_model": 128, "num_layers": 4, "num_heads": 4, "d_ff": 512, "position_embedding": "learned",
}
TEXT_CONFIG = {"context_length": 10, "d_model": 128, "num_layers": 4, "num_heads": 4, "d_ff": 512}
EMBEDDING_DIM = 64  # 共通の埋め込みの次元 d_e(3.2 節の W_I・W_T の列数)
TRAIN_COPIES = 32  # 学習用のキャプションあたりのシーンの数
VALIDATION_COPIES = 2  # 既知の組み合わせの検証集合(学習率の較正と前提条件 P1 にのみ使う)
UNSEEN_COPIES = 4  # 未見の組み合わせの評価集合(判定に使う)
BATCH_SIZES = (16, 64, 256)  # 実験 A のバッチサイズ N(公比 4)
STANDARD_BATCH_SIZE = 256  # 実験 B・C と標準条件 S のバッチサイズ
A_LOSSES = ("softmax", "sigmoid")
WARMUP_RATIO = 0.1  # 最初の 10% のステップで線形 warmup
MIN_LEARNING_RATE_RATIO = 0.01  # cosine decay の下限 = 最大の学習率 x 0.01
WEIGHT_DECAY = 0.1  # 行列形の重みのみ(バイアス・正規化層・[CLS]・位置埋め込み・温度・バイアスには掛けない)
GRADIENT_CLIP_THRESHOLD = 1.0
INIT_LOGIT_SCALE = {"softmax": math.log(1 / 0.07), "negclip": math.log(1 / 0.07), "sigmoid": math.log(10.0)}
INIT_LOGIT_BIAS = -10.0  # sigmoid の損失のみが使う(SigLIP の初期値)
LOGIT_SCALE_MAX = {"softmax": math.log(100.0), "negclip": math.log(100.0), "sigmoid": None}  # CLIP の切り詰め
LR_CENTER_AT_STANDARD = {"softmax": 2e-3, "sigmoid": 1e-3}  # N = 256 の格子の中心(損失ごと。6.2 節の改訂 1・改訂 3)
LR_BATCH_EXPONENT = 0.75  # 格子の中心の N への依存: 中心 x (N / 256)^0.75(6.2 節のパイロットで決めた)
LR_GRID_MULTIPLIERS = (0.5, 1.0, 2.0)  # 格子 = 中心 x (N / 256)^0.75 x {1/2, 1, 2}(公比 2)
LR_GRID_RATIO = 2.0
SEEN_RETRIEVAL_MIN = 0.1  # 前提条件 P1: 既知の組み合わせの検証集合の検索の正解率(チャンス水準 1/1444 の約 144 倍)
RANDOM_NEGATIVES_PER_IMAGE = 2  # ランダムな負例との 2 択の数(困難な負例の 2 種類と数を揃える)
SIGMA_MULTIPLIER = 2.0  # 判定の閾値は対比量の標準偏差の 2 倍
INTERMEDIATE_EVAL_FRACTIONS = (0.25, 0.5, 0.75)  # 途中の検証集合の評価の位置(診断量、ステップ数に対する比率)
MARGIN_TAIL_FRACTION = 0.1  # 学習中のバッチ内のマージンを平均する末尾のステップの割合(診断量)
PROJECTION_IMAGES = 400  # modality gap の射影図に使う未見の組み合わせの評価集合の画像の数(シード 0 のみ)

# 乱数シード(学習のシード s とは独立に固定するもの)
TRAIN_RENDER_SEED = 20_001
VALIDATION_RENDER_SEED = 20_002
UNSEEN_RENDER_SEED = 20_003
RANDOM_NEGATIVE_SEED = 20_004
# 学習のシード s の初期化は torch.manual_seed(20100 + s)、バッチの順序は torch.Generator().manual_seed(20200 + s)
INIT_SEED_BASE = 20_100
DATA_SEED_BASE = 20_200
CALIBRATION_SEED_INDEX = 90  # 学習率の較正専用のシード(実験のシード 0〜4 と共有しない)
TIMING_SEED_INDEX = 91  # スケーリングの計測専用のシード
SESSION_BUDGET_SECONDS = 120 * 60  # 1 セッションの予算(T4 で 120 分)
UPLOAD_KEY_SEED = 0  # アップロードするモデル: 実験 C の NegCLIP・シード 0(結果を見て選ばない)
HUB_REPO_ID = "kojikojiprg/ai-theories-clip-synthetic-scenes"

# 学習で見る事例数 E の候補(公比 2、大きい順)。6.4 節で実行計画として自動選択する(6.1・6.2 節)
EXAMPLE_CANDIDATES = {
    "smoke": (2**13, 2**12),
    "prod": (2**19, 2**18),
}
# --- 削る段階(6.1 節)。順序: 実験 A の N = 64(診断のみ)-> 実験 A のシード数 -> 実験 B・C のシード数 ---
STAGES = {
    "prod": {
        0: {"SEEDS_A": 5, "A_BATCH_SIZES": (16, 64, 256), "SEEDS_BC": 5},
        1: {"SEEDS_A": 5, "A_BATCH_SIZES": (16, 256), "SEEDS_BC": 5},
        2: {"SEEDS_A": 3, "A_BATCH_SIZES": (16, 256), "SEEDS_BC": 5},
        3: {"SEEDS_A": 3, "A_BATCH_SIZES": (16, 256), "SEEDS_BC": 3},
    },
    "smoke": {  # 本番と同じ構造(シード数 5 -> 3 を 3 -> 2 に縮小)
        0: {"SEEDS_A": 3, "A_BATCH_SIZES": (16, 64, 256), "SEEDS_BC": 3},
        1: {"SEEDS_A": 3, "A_BATCH_SIZES": (16, 256), "SEEDS_BC": 3},
        2: {"SEEDS_A": 2, "A_BATCH_SIZES": (16, 256), "SEEDS_BC": 3},
        3: {"SEEDS_A": 2, "A_BATCH_SIZES": (16, 256), "SEEDS_BC": 2},
    },
}
# スケーリングの計測点(ステップ数、6.3 節)。eager と CUDA graph で別
SCALING_STEP_COUNTS_BY_LEVEL = {
    "eager": {"smoke": (8, 16, 32), "prod": (32, 64, 128)},
    "graph": {"smoke": (32, 64, 128), "prod": (128, 256, 512)},
}


def build_plans(level: str) -> list[dict]:
    # 実行計画(6.1 節): 計画 0〜7 は較正の方式 "all" で、E の大きい順を第 1 キー、段階の番号の小さい順を第 2 キーとする。
    # 計画 8〜11 は下位の計画で、E の小さい方の候補・較正の方式 "representative"・段階 0〜3
    plans = [
        {"E": e, "stage": stage, "calibration_mode": "all"} for e in EXAMPLE_CANDIDATES[level] for stage in sorted(STAGES[level])
    ]
    plans += [
        {"E": min(EXAMPLE_CANDIDATES[level]), "stage": stage, "calibration_mode": "representative"}
        for stage in sorted(STAGES[level])
    ]
    return [{"plan": i} | p for i, p in enumerate(plans)]


PLANS = {level: build_plans(level) for level in EXAMPLE_CANDIDATES}

CURRENT_LEVEL_NAME = "smoke" if SMOKE_TEST else "prod"
FORCED_PLAN_VALUE = os.environ.get("AI_THEORIES_FORCE_PLAN")
if FORCED_PLAN_VALUE is not None and not SMOKE_TEST:
    raise RuntimeError(
        f"本番(SMOKE_TEST=False)では計画の強制(AI_THEORIES_FORCE_PLAN={FORCED_PLAN_VALUE!r})を受け付けない。"
        "環境変数を削除して再実行すること。"
    )
FORCED_PLAN = None if FORCED_PLAN_VALUE is None else int(FORCED_PLAN_VALUE)
assert FORCED_PLAN is None or 0 <= FORCED_PLAN < len(PLANS[CURRENT_LEVEL_NAME]), FORCED_PLAN

PROD_EXAMPLE_CANDIDATES = EXAMPLE_CANDIDATES["prod"]
SCALING_STEP_COUNTS = {kind: counts[CURRENT_LEVEL_NAME] for kind, counts in SCALING_STEP_COUNTS_BY_LEVEL.items()}
# EXAMPLES(E)・CALIBRATION_MODE・EXECUTION_MODE は 6.4 節で実行計画と実行の方式を選んだ後に決まる


def steps_for(examples: int, batch_size: int) -> int:
    assert examples % batch_size == 0, (examples, batch_size)
    return examples // batch_size


def warmup_steps_for(num_steps: int) -> int:
    return max(1, round(WARMUP_RATIO * num_steps))


def intermediate_eval_steps_for(num_steps: int) -> tuple[int, ...]:
    return tuple(round(f * num_steps) for f in INTERMEDIATE_EVAL_FRACTIONS)


def lr_grid_for(loss_type: str, batch_size: int) -> tuple[float, ...]:
    center = LR_CENTER_AT_STANDARD[loss_type] * (batch_size / STANDARD_BATCH_SIZE) ** LR_BATCH_EXPONENT
    return tuple(center * m for m in LR_GRID_MULTIPLIERS)


# --- 縮小規則と水準の構造の確認 ---
for _name, _candidates in EXAMPLE_CANDIDATES.items():
    assert len(_candidates) == 2 and all(a == 2 * b for a, b in zip(_candidates, _candidates[1:], strict=False))
    for _e in _candidates:
        for _n in BATCH_SIZES:
            _t = steps_for(_e, _n)
            assert len(set(intermediate_eval_steps_for(_t))) == 3 and warmup_steps_for(_t) < _t, (_e, _n)
    _plans = PLANS[_name]
    assert len(_plans) == 12 and [p["plan"] for p in _plans] == list(range(12))
    _main, _lower = _plans[:8], _plans[8:]
    assert all(p["calibration_mode"] == "all" for p in _main) and all(p["calibration_mode"] == "representative" for p in _lower)
    assert all(  # 優先順位(計画 0〜7): E の大きい順、同じ E では段階の小さい順
        (a["E"] > b["E"]) or (a["E"] == b["E"] and a["stage"] < b["stage"]) for a, b in zip(_main, _main[1:], strict=False)
    )
    assert [p["stage"] for p in _lower] == [0, 1, 2, 3] and all(p["E"] == min(_candidates) for p in _lower)
    _stages = STAGES[_name]
    assert set(_stages) == {0, 1, 2, 3}
    for _k in range(1, 4):  # 段階が上がるほど、どの量も減るか変わらない
        for _key in ("SEEDS_A", "SEEDS_BC"):
            assert _stages[_k][_key] <= _stages[_k - 1][_key]
        assert set(_stages[_k]["A_BATCH_SIZES"]) <= set(_stages[_k - 1]["A_BATCH_SIZES"])
    for _stage in _stages.values():
        assert min(_stage["SEEDS_A"], _stage["SEEDS_BC"]) >= 2
        assert {min(BATCH_SIZES), STANDARD_BATCH_SIZE} <= set(_stage["A_BATCH_SIZES"]) <= set(BATCH_SIZES)
_ratios = {p / s for p, s in zip(EXAMPLE_CANDIDATES["prod"], EXAMPLE_CANDIDATES["smoke"], strict=True)}
assert len(_ratios) == 1 and _ratios.pop() > 1  # 本番とスモークテストの比が候補の間で揃う
for _k in range(1, 4):  # 本番とスモークテストで、各段階で何が変わるかが同じ(構造を保つ縮小)
    for _key in ("SEEDS_A", "SEEDS_BC", "A_BATCH_SIZES"):
        assert (STAGES["prod"][_k][_key] == STAGES["prod"][_k - 1][_key]) == (
            STAGES["smoke"][_k][_key] == STAGES["smoke"][_k - 1][_key]
        )
    assert STAGES["prod"][_k]["A_BATCH_SIZES"] == STAGES["smoke"][_k]["A_BATCH_SIZES"]
for _by_level in SCALING_STEP_COUNTS_BY_LEVEL.values():
    for _counts in _by_level.values():
        assert all(b == 2 * a for a, b in zip(_counts, _counts[1:], strict=False))
    assert len({p / s for p, s in zip(_by_level["prod"], _by_level["smoke"], strict=True)}) == 1
assert all(b / a == 4 for a, b in zip(BATCH_SIZES, BATCH_SIZES[1:], strict=False))
for _n in BATCH_SIZES:
    for _loss in A_LOSSES:
        _grid = lr_grid_for(_loss, _n)
        assert all(math.isclose(b / a, LR_GRID_RATIO) for a, b in zip(_grid, _grid[1:], strict=False))

print(
    f"水準 {CURRENT_LEVEL_NAME!r}: E の候補 {EXAMPLE_CANDIDATES[CURRENT_LEVEL_NAME]}(本番 {PROD_EXAMPLE_CANDIDATES})"
)
for _e in EXAMPLE_CANDIDATES[CURRENT_LEVEL_NAME]:
    print(
        f"  E = {_e:,}: ステップ数 T_N = E/N = "
        + "、".join(f"N={n}: {steps_for(_e, n):,}(warmup {warmup_steps_for(steps_for(_e, n))})" for n in BATCH_SIZES)
    )
print(f"画像 encoder(ViT): {VISION_CONFIG}")
print(f"テキスト encoder(因果マスク): {TEXT_CONFIG}、共通の埋め込みの次元 d_e = {EMBEDDING_DIM}")
print(
    f"学習: AdamW(重み減衰 {WEIGHT_DECAY}、foreach)、warmup {WARMUP_RATIO:.0%} + cosine(下限 x{MIN_LEARNING_RATE_RATIO})、"
    f"gradient clipping {GRADIENT_CLIP_THRESHOLD}、温度の初期値 {json.dumps({k: round(v, 4) for k, v in INIT_LOGIT_SCALE.items()})}、"
    f"sigmoid のバイアスの初期値 {INIT_LOGIT_BIAS}、倍率の上限(log){json.dumps({k: (round(v, 4) if v else None) for k, v in LOGIT_SCALE_MAX.items()})}"
)
for _loss in A_LOSSES:
    print(f"学習率の格子({_loss}、N ごと): " + "、".join(f"N={n}: {tuple(float(f'{x:.3g}') for x in lr_grid_for(_loss, n))}" for n in BATCH_SIZES))
print(f"スケーリングの計測点(ステップ数) {SCALING_STEP_COUNTS}")
print(f"削る段階({CURRENT_LEVEL_NAME!r}): {json.dumps(STAGES[CURRENT_LEVEL_NAME])}")
print(f"実行計画({CURRENT_LEVEL_NAME!r}、6.4 節で選ぶ、番号の小さいほど優先):")
for _p in PLANS[CURRENT_LEVEL_NAME]:
    print(f"  計画 {_p['plan']:2d}: E = {_p['E']:,}、段階 {_p['stage']}、較正の方式 {_p['calibration_mode']!r}")
```

    水準 'prod': E の候補 (524288, 262144)(本番 (524288, 262144))
      E = 524,288: ステップ数 T_N = E/N = N=16: 32,768(warmup 3277)、N=64: 8,192(warmup 819)、N=256: 2,048(warmup 205)
      E = 262,144: ステップ数 T_N = E/N = N=16: 16,384(warmup 1638)、N=64: 4,096(warmup 410)、N=256: 1,024(warmup 102)
    画像 encoder(ViT): {'image_size': 32, 'patch_size': 4, 'in_channels': 3, 'num_classes': 1, 'd_model': 128, 'num_layers': 4, 'num_heads': 4, 'd_ff': 512, 'position_embedding': 'learned'}
    テキスト encoder(因果マスク): {'context_length': 10, 'd_model': 128, 'num_layers': 4, 'num_heads': 4, 'd_ff': 512}、共通の埋め込みの次元 d_e = 64
    学習: AdamW(重み減衰 0.1、foreach)、warmup 10% + cosine(下限 x0.01)、gradient clipping 1.0、温度の初期値 {"softmax": 2.6593, "negclip": 2.6593, "sigmoid": 2.3026}、sigmoid のバイアスの初期値 -10.0、倍率の上限(log){"softmax": 4.6052, "negclip": 4.6052, "sigmoid": null}
    学習率の格子(softmax、N ごと): N=16: (0.000125, 0.00025, 0.0005)、N=64: (0.000354, 0.000707, 0.00141)、N=256: (0.001, 0.002, 0.004)
    学習率の格子(sigmoid、N ごと): N=16: (6.25e-05, 0.000125, 0.00025)、N=64: (0.000177, 0.000354, 0.000707)、N=256: (0.0005, 0.001, 0.002)
    スケーリングの計測点(ステップ数) {'eager': (32, 64, 128), 'graph': (128, 256, 512)}
    削る段階('prod'): {"0": {"SEEDS_A": 5, "A_BATCH_SIZES": [16, 64, 256], "SEEDS_BC": 5}, "1": {"SEEDS_A": 5, "A_BATCH_SIZES": [16, 256], "SEEDS_BC": 5}, "2": {"SEEDS_A": 3, "A_BATCH_SIZES": [16, 256], "SEEDS_BC": 5}, "3": {"SEEDS_A": 3, "A_BATCH_SIZES": [16, 256], "SEEDS_BC": 3}}
    実行計画('prod'、6.4 節で選ぶ、番号の小さいほど優先):
      計画  0: E = 524,288、段階 0、較正の方式 'all'
      計画  1: E = 524,288、段階 1、較正の方式 'all'
      計画  2: E = 524,288、段階 2、較正の方式 'all'
      計画  3: E = 524,288、段階 3、較正の方式 'all'
      計画  4: E = 262,144、段階 0、較正の方式 'all'
      計画  5: E = 262,144、段階 1、較正の方式 'all'
      計画  6: E = 262,144、段階 2、較正の方式 'all'
      計画  7: E = 262,144、段階 3、較正の方式 'all'
      計画  8: E = 262,144、段階 0、較正の方式 'representative'
      計画  9: E = 262,144、段階 1、較正の方式 'representative'
      計画 10: E = 262,144、段階 2、較正の方式 'representative'
      計画 11: E = 262,144、段階 3、較正の方式 'representative'


### 5.3 データ: 合成のシーンとキャプション

- **語彙**: 特殊トークン 3 個(`<pad>`・`<sot>`・`<eot>`)、機能語 4 個(`a`・`left`・`of`・`above`)、色 8 個、形 5 個の計 20 個の単語単位の
  トークナイザ。系列は`[<sot>, 単語..., <eot>, <pad>...]`で長さ 10 にそろえる。**全キャプションで符号化と復号がラウンドトリップで
  一致すること** を確かめる。
- **キャプションの空間**: 2 つの図形の (色, 形) が色も形も異なる組を、左右(`left of`)・上下(`above`)の 2 通りに並べた全 2,240 個
  (3.7 節の注意と 5.3 節の出力)。キャプションは意味から一意に決まる正準形で、`right of`・`below`は使わない。
- **未見の組み合わせ**: (色, 形) の 40 組のうち 8 組(20%、色 $i$ に形 $i \bmod 5$ を対応させる)を学習データの **どちらの図形にも出さず**、
  それを 1 つ以上含むキャプション 796 個を未見の組み合わせの評価集合に回す。残る 1,444 個が学習用のキャプションである。除外が
  守られていること(学習・検証のシーンのどちらの図形にも除外した組がないこと、各色・各形は学習データに現れること)をアサーションで
  確かめる。
- **困難な負例**: 色の入れ替えと順序の入れ替え(3.7 節)。いずれも (1)元のキャプションと **意味が異なる**(`caption_meaning()`で
  正準形の意味に戻して比べる。`B right of A`のような言い換えも正準形に直してから比べるので、同義の言い換えが負例に混ざらない)、
  (2)単語の多重集合が元と同じ、(3)有効なキャプション(色も形も異なる)であることを確かめる。
- **描画**: 学習用(各キャプション 32 枚、シード`20001`)、既知の組み合わせの検証集合(各 2 枚、シード`20002`)、未見の組み合わせの
  評価集合(各 4 枚、シード`20003`)。**描画した画像は`.cache/020_synthetic_scenes/`にキャッシュする。** キャッシュの鍵は
  `src/data/synthetic_scenes.py`のソースのハッシュ・キャプションの番号・シードで、規則を変えると別の鍵になる。**キャッシュから読んだ
  画像、ディスクから読み直した画像、キャッシュを使わずに描き直した画像(一度に描く数を変えて描く)の 3 つが bit 単位で一致すること**
  を確かめる。
- **学習中の困難な負例の除外**: 学習用のキャプションの色の入れ替えのうち、除外した組を含むものを数えて印字する。NegCLIP の学習では
  これらを困難な負例に使わない(3.7 節)。順序の入れ替えが常に学習用のキャプションであることを確かめる。
- **ランダムな負例**(実験 B、シード`20004`): 未見の組み合わせの評価集合の各画像に 2 個を選ぶ。1 個目は順序の入れ替えに対応させ、
  正例以外の未見の組み合わせのキャプションから一様に選ぶ。2 個目は色の入れ替えに対応させ、その画像の色の入れ替えのキャプションと
  同じ集合(学習用 / 未見の組み合わせ)から、正例とその困難な負例を除いて一様に選ぶ(6.1 節)。診断量のために、旧定義(2 個とも
  未見の組み合わせから、以前と同じ乱数の列)と、学習用のキャプションから一様に選んだ 1 個も用意する。
- 画像は uint8 のまま GPU に載せ、全条件・全シードで再利用する。


```python
_t0_data = time.time()
VOCABULARY = CaptionVocabulary(TEXT_CONFIG["context_length"])
UNIVERSE = build_caption_universe(VOCABULARY)
NUM_CAPTIONS = UNIVERSE.num_captions
TRAIN_CAPTION_IDS = UNIVERSE.train_caption_ids
UNSEEN_CAPTION_IDS = UNIVERSE.unseen_caption_ids

# --- 語彙と符号化のラウンドトリップ ---
assert len(VOCABULARY) == 3 + 4 + len(COLOR_NAMES) + len(SHAPES) == 20
_token_rows = UNIVERSE.token_ids.tolist()
for _caption, _ids in zip(UNIVERSE.captions, _token_rows, strict=True):
    assert VOCABULARY.encode(_caption) == _ids and VOCABULARY.decode(_ids) == _caption
    assert _ids.count(VOCABULARY.end_id) == 1 and _ids[0] == VOCABULARY.start_id
    _end = _ids.index(VOCABULARY.end_id)
    assert all(t == VOCABULARY.pad_id for t in _ids[_end + 1 :]) and VOCABULARY.pad_id not in _ids[:_end]
assert len(set(UNIVERSE.captions)) == NUM_CAPTIONS  # キャプションの文字列が互いに異なる

# --- 意味と正準形 ---
assert NUM_CAPTIONS == len(RELATIONS) * 40 * (7 * 4) == 2240
for _m, _c in zip(UNIVERSE.meanings, UNIVERSE.captions, strict=True):
    assert caption_meaning(_c) == _m and is_valid_meaning(_m)
_example = UNIVERSE.meanings[0]
_c1, _c2 = (f"{COLOR_NAMES[o[0]]} {SHAPES[o[1]]}" for o in (_example.first, _example.second))
assert caption_meaning(f"a {_c2} right of a {_c1}") == _example  # 同義の言い換えは同じ意味に戻る
_vertical = next(m for m in UNIVERSE.meanings if m.relation == RELATIONS.index("above"))
_v1, _v2 = (f"{COLOR_NAMES[o[0]]} {SHAPES[o[1]]}" for o in (_vertical.first, _vertical.second))
assert caption_meaning(f"a {_v2} below a {_v1}") == _vertical

# --- 未見の組み合わせの分割 ---
HOLDOUT = UNIVERSE.holdout
_all_objects = set(itertools.product(range(len(COLOR_NAMES)), range(len(SHAPES))))
assert len(HOLDOUT) == 8 and HOLDOUT < _all_objects and len(HOLDOUT) / len(_all_objects) == 0.2
_train_objects = _all_objects - HOLDOUT
assert {c for c, _ in _train_objects} == set(range(len(COLOR_NAMES)))  # 各色は学習データに現れる
assert {s for _, s in _train_objects} == set(range(len(SHAPES)))  # 各形は学習データに現れる
assert np.intersect1d(TRAIN_CAPTION_IDS, UNSEEN_CAPTION_IDS).size == 0
assert len(TRAIN_CAPTION_IDS) + len(UNSEEN_CAPTION_IDS) == NUM_CAPTIONS
for _i in TRAIN_CAPTION_IDS:
    _m = UNIVERSE.meanings[_i]
    assert _m.first not in HOLDOUT and _m.second not in HOLDOUT
for _i in UNSEEN_CAPTION_IDS:
    _m = UNIVERSE.meanings[_i]
    assert _m.first in HOLDOUT or _m.second in HOLDOUT

# --- 困難な負例: 意味が異なる・単語の多重集合が同じ・有効 ---
for _rule, _neg_ids in UNIVERSE.hard_negative_ids.items():
    for _i, _j in enumerate(_neg_ids):
        _orig, _neg = UNIVERSE.captions[_i], UNIVERSE.captions[_j]
        assert caption_meaning(_neg) != caption_meaning(_orig), (_rule, _orig, _neg)  # 同義の言い換えではない
        assert sorted(_neg.split()) == sorted(_orig.split()), (_rule, _orig, _neg)  # 単語の袋としては同じ
        assert is_valid_meaning(UNIVERSE.meanings[_j])
assert (UNIVERSE.hard_negative_ids["attribute_swap"] != UNIVERSE.hard_negative_ids["order_swap"]).all()
# 順序の入れ替えは未見 / 既知の分割を保つ(同じ 2 つの図形を使う)
assert np.isin(UNIVERSE.hard_negative_ids["order_swap"][UNSEEN_CAPTION_IDS], UNSEEN_CAPTION_IDS).all()
_attr_neg_unseen = np.isin(UNIVERSE.hard_negative_ids["attribute_swap"][UNSEEN_CAPTION_IDS], UNSEEN_CAPTION_IDS)

# --- 描画とキャッシュ ---
SCENE_CACHE_DIR = ROOT / ".cache" / "020_synthetic_scenes"
_SCENE_SOURCE_HASH = hashlib.sha256((ROOT / "src" / "data" / "synthetic_scenes.py").read_bytes()).hexdigest()


def scene_cache_path(name: str, caption_ids: np.ndarray, seed: int) -> Path:
    digest = hashlib.sha256(_SCENE_SOURCE_HASH.encode() + np.asarray(caption_ids).tobytes() + str(seed).encode())
    return SCENE_CACHE_DIR / f"{name}_{digest.hexdigest()[:16]}.npz"


def load_scene_file(path: Path) -> SceneSet:
    with np.load(path) as data:
        return SceneSet(torch.from_numpy(data["images"]), data["caption_ids"], data["visible_pixels"])


def load_or_render(name: str, caption_ids: np.ndarray, seed: int) -> tuple[SceneSet, str]:
    path = scene_cache_path(name, caption_ids, seed)
    if path.exists():
        return load_scene_file(path), "キャッシュから読み込み"
    scenes = render_scenes(UNIVERSE, caption_ids, seed)
    SCENE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, images=scenes.images.numpy(), caption_ids=scenes.caption_ids, visible_pixels=scenes.visible_pixels)
    return scenes, "描画してキャッシュに保存"


SCENE_SPECS = {
    "train": (repeat_caption_ids(TRAIN_CAPTION_IDS, TRAIN_COPIES), TRAIN_RENDER_SEED),
    "validation": (repeat_caption_ids(TRAIN_CAPTION_IDS, VALIDATION_COPIES), VALIDATION_RENDER_SEED),
    "unseen": (repeat_caption_ids(UNSEEN_CAPTION_IDS, UNSEEN_COPIES), UNSEEN_RENDER_SEED),
}
SCENES: dict[str, SceneSet] = {}
SCENE_SHA256: dict[str, str] = {}
for _name, (_ids, _seed) in SCENE_SPECS.items():
    SCENES[_name], _source = load_or_render(_name, _ids, _seed)
    _reloaded = load_scene_file(scene_cache_path(_name, _ids, _seed))  # ディスクから読み直す
    _fresh = render_scenes(UNIVERSE, _ids, _seed, chunk_size=1000)  # キャッシュを使わず、一度に描く数を変えて描き直す
    assert torch.equal(SCENES[_name].images, _reloaded.images) and torch.equal(SCENES[_name].images, _fresh.images)
    assert np.array_equal(SCENES[_name].caption_ids, _ids) and np.array_equal(_fresh.caption_ids, _ids)
    assert np.array_equal(SCENES[_name].visible_pixels, _fresh.visible_pixels)
    SCENE_SHA256[_name] = sha256_of_tensor(SCENES[_name].images)
    print(
        f"{_name}: {len(_ids):,} 枚({_source})。キャッシュ・ディスクから読み直し・描き直しが bit 単位で一致: OK、"
        f"SHA-256 {SCENE_SHA256[_name][:16]}...、見えている画素数の最小値 {SCENES[_name].visible_pixels.min()}"
    )
    assert SCENES[_name].visible_pixels.min() >= 20  # 2 つの図形がどちらも見えている
    for _i in np.unique(_ids):  # 除外した組がどちらの図形にも出ない(学習・検証)/ 1 つ以上出る(評価)
        _m = UNIVERSE.meanings[_i]
        assert (_m.first in HOLDOUT or _m.second in HOLDOUT) == (_name == "unseen")
    del _reloaded, _fresh
assert np.array_equal(np.bincount(SCENES["unseen"].caption_ids, minlength=NUM_CAPTIONS)[UNSEEN_CAPTION_IDS], np.full(len(UNSEEN_CAPTION_IDS), UNSEEN_COPIES))

# --- 学習中の困難な負例の除外(NegCLIP): 除外した組を含むキャプションは困難な負例に使わない ---
TRAIN_CAPTION_MASK = np.zeros(NUM_CAPTIONS, dtype=bool)
TRAIN_CAPTION_MASK[TRAIN_CAPTION_IDS] = True
_train_attr_ok = TRAIN_CAPTION_MASK[UNIVERSE.hard_negative_ids["attribute_swap"][TRAIN_CAPTION_IDS]]
assert TRAIN_CAPTION_MASK[UNIVERSE.hard_negative_ids["order_swap"][TRAIN_CAPTION_IDS]].all()  # 順序の入れ替えは常に学習用
ATTRIBUTE_SWAP_TRAIN_FRACTION = float(_train_attr_ok.mean())

# --- ランダムな負例(実験 B) ---
_rng = np.random.default_rng(RANDOM_NEGATIVE_SEED)
_unseen_positive = SCENES["unseen"].caption_ids
_candidate_positions = {c: i for i, c in enumerate(UNSEEN_CAPTION_IDS)}
_positive_positions = np.array([_candidate_positions[c] for c in _unseen_positive])


def _uniform_unseen_except_positive(rng, size: int) -> np.ndarray:
    draw = rng.integers(0, len(UNSEEN_CAPTION_IDS) - 1, size=(len(_unseen_positive), size))
    return UNSEEN_CAPTION_IDS[draw + (draw >= _positive_positions[:, None])]  # 正例の位置を飛ばす


# 旧定義(診断量): 2 個とも、正例以外の未見の組み合わせのキャプションから一様に選ぶ(以前と同じ乱数の列)
RANDOM_NEGATIVE_IDS_OLD = _uniform_unseen_except_positive(_rng, RANDOM_NEGATIVES_PER_IMAGE)
# 新定義: 1 個目は順序の入れ替えに対応させて未見の組み合わせから(正例を除く)、2 個目は色の入れ替えに対応させ、
# その画像の色の入れ替えのキャプションと同じ集合(学習用 / 未見の組み合わせ)から、正例とその困難な負例を除いて選ぶ
_new_first = _uniform_unseen_except_positive(_rng, 1)[:, 0]
_attr_swap_of_positive = UNIVERSE.hard_negative_ids["attribute_swap"][_unseen_positive]
_order_swap_of_positive = UNIVERSE.hard_negative_ids["order_swap"][_unseen_positive]
_new_second = np.empty(len(_unseen_positive), dtype=np.int64)
for _i, (_p, _a, _o) in enumerate(zip(_unseen_positive, _attr_swap_of_positive, _order_swap_of_positive, strict=True)):
    _pool = TRAIN_CAPTION_IDS if TRAIN_CAPTION_MASK[_a] else UNSEEN_CAPTION_IDS
    while True:  # 除く 3 個以外から一様に選ぶ(棄却法)
        _c = _pool[_rng.integers(0, len(_pool))]
        if _c not in (_p, _a, _o):
            break
    _new_second[_i] = _c
RANDOM_NEGATIVE_IDS = np.stack([_new_first, _new_second], axis=1)  # (枚数, 2)
# 既知性の偏りの診断量: 未見の正例 vs 学習用のキャプション(一様)の 2 択と、未見の正例 vs 未見の組み合わせのキャプション
# (新定義の 1 個目)の 2 択
RANDOM_TRAIN_IDS = TRAIN_CAPTION_IDS[_rng.integers(0, len(TRAIN_CAPTION_IDS), size=len(_unseen_positive))]
assert (RANDOM_NEGATIVE_IDS != _unseen_positive[:, None]).all() and (RANDOM_NEGATIVE_IDS_OLD != _unseen_positive[:, None]).all()
assert np.isin(RANDOM_NEGATIVE_IDS_OLD, UNSEEN_CAPTION_IDS).all() and np.isin(_new_first, UNSEEN_CAPTION_IDS).all()
assert (TRAIN_CAPTION_MASK[_new_second] == TRAIN_CAPTION_MASK[_attr_swap_of_positive]).all()  # 既知 / 未見の状態が対応する
assert ((_new_second != _attr_swap_of_positive) & (_new_second != _order_swap_of_positive)).all()
assert TRAIN_CAPTION_MASK[RANDOM_TRAIN_IDS].all()

# --- GPU に載せる(全条件・全シードで再利用) ---
TOKENS = UNIVERSE.token_ids.to(device)
TRAIN_IMAGES = SCENES["train"].images.to(device)
VALIDATION_IMAGES = SCENES["validation"].images.to(device)
UNSEEN_IMAGES = SCENES["unseen"].images.to(device)
TRAIN_SCENE_CAPTION_IDS = SCENES["train"].caption_ids
TRAIN_CANDIDATES = torch.as_tensor(TRAIN_CAPTION_IDS, device=device)
UNSEEN_CANDIDATES = torch.as_tensor(UNSEEN_CAPTION_IDS, device=device)
_train_pos = {c: i for i, c in enumerate(TRAIN_CAPTION_IDS)}
VALIDATION_TRUE_INDEX = torch.as_tensor([_train_pos[c] for c in SCENES["validation"].caption_ids], device=device)
UNSEEN_TRUE_INDEX = torch.as_tensor(_positive_positions, device=device)
UNSEEN_POSITIVE_IDS = torch.as_tensor(_unseen_positive, device=device)
VALIDATION_POSITIVE_IDS = torch.as_tensor(SCENES["validation"].caption_ids, device=device)
HARD_NEGATIVE_IDS_DEVICE = {k: torch.as_tensor(v, device=device) for k, v in UNIVERSE.hard_negative_ids.items()}
RANDOM_NEGATIVE_SETS_DEVICE = {  # 名前 -> (枚数, k) の負例のキャプションの番号
    "random": torch.as_tensor(RANDOM_NEGATIVE_IDS, device=device),  # 判定に使う(新定義)
    "random_old": torch.as_tensor(RANDOM_NEGATIVE_IDS_OLD, device=device),  # 旧定義(診断量)
    "random_train": torch.as_tensor(RANDOM_TRAIN_IDS[:, None], device=device),  # 既知性の偏り(診断量)
    "random_unseen": torch.as_tensor(_new_first[:, None], device=device),  # 既知性の偏り(診断量)
}
assert torch.equal(TRAIN_IMAGES.cpu(), SCENES["train"].images) and torch.equal(UNSEEN_IMAGES.cpu(), SCENES["unseen"].images)
UNSEEN_IMAGES_SHA256 = sha256_of_tensor(UNSEEN_IMAGES)
DATA_SECONDS = time.time() - _t0_data

print(
    f"語彙 {len(VOCABULARY)} 語 {VOCABULARY.tokens}、系列長 {VOCABULARY.context_length}。"
    f"全 {NUM_CAPTIONS:,} キャプションで符号化・復号のラウンドトリップが一致: OK"
)
print(
    f"除外した (色, 形) の組 H(40 組のうち {len(HOLDOUT)} 組): "
    + "、".join(f"{COLOR_NAMES[c]} {SHAPES[s]}" for c, s in sorted(HOLDOUT))
)
print(
    f"キャプション: 全 {NUM_CAPTIONS:,} = 学習用 {len(TRAIN_CAPTION_IDS):,} + 未見の組み合わせ {len(UNSEEN_CAPTION_IDS):,}。"
    "除外した組は学習・検証のどちらの図形にも出ない、各色・各形は学習データに現れる: OK"
)
print(
    "困難な負例(色の入れ替え・順序の入れ替え): 全キャプションで意味が異なり(同義の言い換えなし)、単語の多重集合が同じ、有効: OK。"
    f"未見の組み合わせのキャプションの色の入れ替えが未見の組み合わせのキャプションになる割合 {_attr_neg_unseen.mean():.3f}"
    "(順序の入れ替えは常に未見の組み合わせのまま)"
)
print(
    f"学習用のキャプションの色の入れ替えのうち、除外した組を含むもの: {int((~_train_attr_ok).sum())} / {len(TRAIN_CAPTION_IDS):,} 個"
    f"({1 - ATTRIBUTE_SWAP_TRAIN_FRACTION:.3f})。NegCLIP の学習ではこれらを困難な負例に使わないので、色の入れ替えの困難な負例を持つ正例は"
    f" {ATTRIBUTE_SWAP_TRAIN_FRACTION:.3f}。順序の入れ替えは常に学習用のキャプション: OK"
)
_second_is_train = TRAIN_CAPTION_MASK[_new_second].mean()
print(
    f"ランダムな負例(新定義): 各画像に 2 個。1 個目は未見の組み合わせから、2 個目は色の入れ替えと同じ集合から(学習用 {_second_is_train:.3f}・"
    f"未見 {1 - _second_is_train:.3f})。既知 / 未見の状態が色の入れ替えと対応し、正例と困難な負例を含まない: OK。"
    "旧定義(すべて未見の組み合わせから)と既知性の偏りの診断用の負例も用意した"
)
print("バッチ内に、ある正例の困難な負例のキャプションが別の正例として偶然現れる確率(3.7 節):")
for _n in BATCH_SIZES:
    _p_order = (_n - 1) / (len(TRAIN_CAPTION_IDS) - 1)
    print(
        f"  N = {_n}: 順序の入れ替え {_p_order:.4f}、色の入れ替え {_p_order * ATTRIBUTE_SWAP_TRAIN_FRACTION:.4f}"
        f"(学習用として存在する割合 {ATTRIBUTE_SWAP_TRAIN_FRACTION:.3f} を掛けた平均)"
    )
# 3.7 節の比較: 6 色 x 4 種類(除外 5 組)にした場合の学習用のキャプションの数と同じ確率(N = 256)
_objects_64 = [o for o in itertools.product(range(6), range(4)) if o not in {(i, i % 4) for i in range(5)}]
_captions_64 = 2 * sum(1 for a in _objects_64 for b in _objects_64 if a[0] != b[0] and a[1] != b[1])
print(f"  (参考)6 色 x 4 種類なら学習用のキャプション {_captions_64} 個、N = 256 で {(256 - 1) / (_captions_64 - 1):.4f}")
print(f"データの準備: {DATA_SECONDS:.1f}s")

_fig, _axes = plt.subplots(2, 6, figsize=(13, 5.4))
for _k, _ax in enumerate(_axes.flat):
    _set = "train" if _k < 6 else "unseen"
    _j = (_k % 6) * (len(SCENES[_set].caption_ids) // 6) + 3
    _ax.imshow(SCENES[_set].images[_j].permute(1, 2, 0).numpy())
    _ax.set_title(("[train] " if _set == "train" else "[unseen] ") + UNIVERSE.captions[SCENES[_set].caption_ids[_j]][2:], fontsize=7)
    _ax.axis("off")
plt.tight_layout()
plt.show()
```

    train: 46,208 枚(描画してキャッシュに保存)。キャッシュ・ディスクから読み直し・描き直しが bit 単位で一致: OK、SHA-256 2e89f47c75dc1a60...、見えている画素数の最小値 29
    validation: 2,888 枚(描画してキャッシュに保存)。キャッシュ・ディスクから読み直し・描き直しが bit 単位で一致: OK、SHA-256 b994bfd499633c25...、見えている画素数の最小値 32
    unseen: 3,184 枚(描画してキャッシュに保存)。キャッシュ・ディスクから読み直し・描き直しが bit 単位で一致: OK、SHA-256 c390c156be2ab3b7...、見えている画素数の最小値 30
    語彙 20 語 ('<pad>', '<sot>', '<eot>', 'a', 'left', 'of', 'above', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'white', 'circle', 'square', 'triangle', 'diamond', 'cross')、系列長 10。全 2,240 キャプションで符号化・復号のラウンドトリップが一致: OK
    除外した (色, 形) の組 H(40 組のうち 8 組): red circle、green square、blue triangle、yellow diamond、magenta cross、cyan circle、orange square、white triangle
    キャプション: 全 2,240 = 学習用 1,444 + 未見の組み合わせ 796。除外した組は学習・検証のどちらの図形にも出ない、各色・各形は学習データに現れる: OK
    困難な負例(色の入れ替え・順序の入れ替え): 全キャプションで意味が異なり(同義の言い換えなし)、単語の多重集合が同じ、有効: OK。未見の組み合わせのキャプションの色の入れ替えが未見の組み合わせのキャプションになる割合 0.121(順序の入れ替えは常に未見の組み合わせのまま)
    学習用のキャプションの色の入れ替えのうち、除外した組を含むもの: 700 / 1,444 個(0.485)。NegCLIP の学習ではこれらを困難な負例に使わないので、色の入れ替えの困難な負例を持つ正例は 0.515。順序の入れ替えは常に学習用のキャプション: OK
    ランダムな負例(新定義): 各画像に 2 個。1 個目は未見の組み合わせから、2 個目は色の入れ替えと同じ集合から(学習用 0.879・未見 0.121)。既知 / 未見の状態が色の入れ替えと対応し、正例と困難な負例を含まない: OK。旧定義(すべて未見の組み合わせから)と既知性の偏りの診断用の負例も用意した
    バッチ内に、ある正例の困難な負例のキャプションが別の正例として偶然現れる確率(3.7 節):
      N = 16: 順序の入れ替え 0.0104、色の入れ替え 0.0054(学習用として存在する割合 0.515 を掛けた平均)
      N = 64: 順序の入れ替え 0.0437、色の入れ替え 0.0225(学習用として存在する割合 0.515 を掛けた平均)
      N = 256: 順序の入れ替え 0.1767、色の入れ替え 0.0910(学習用として存在する割合 0.515 を掛けた平均)
      (参考)6 色 x 4 種類なら学習用のキャプション 456 個、N = 256 で 0.5604
    データの準備: 35.5s



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/020_clip_contrastive_learning/output_20_1.png)
    




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/05_vision_language/020_clip_contrastive_learning.ipynb
