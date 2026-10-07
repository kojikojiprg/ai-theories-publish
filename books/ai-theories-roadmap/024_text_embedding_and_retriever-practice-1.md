---
title: "テキスト埋め込みと retriever / Text Embedding and Retriever(実装・実験編 1/7)"
---

この記事は後編(実装・実験編 1/7)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/024_text_embedding_and_retriever-theory)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/024_text_embedding_and_retriever-practice-2)。

## 4. 実装方針 / Implementation Policy

**`src/`に切り出す(スクラッチ実装、本トピックで新規作成)**:

- `src/models/text_embedding.py`: 小型 GPT を包む埋め込みモデル`TextEmbeddingModel`(プーリング(平均・終端位置)、注意マスクの切り替え(因果・双方向)、
  L2 正規化)。最終正規化層の後の隠れ状態`compute_hidden_states()`、次元の切り詰めと再正規化`truncate_and_normalize()`、固定長のトークン列を FP32 で符号化する
  `encode_tokens()`(切り詰める次元の一覧と、指定した位置の出力の埋め込みも一度に返す)。既存の`GPTLanguageModel`は変更しない。
- `src/data/retrieval.py`: 記事を単位とする学習用・検証用・評価用の分割`split_articles()`(使ってよい記事の候補の中から決定的に選ぶ)、使う記事だけの記事ごとの符号化、評価用の索引と query の構成
  `build_retrieval_set()`(記事ごとの passage 数の上限つき、決定的)、学習の組の切り出し`sample_pair_schedule()`(重ならない 2 つの区間、1 つのバッチに同じ記事を入れない)。
  [025](https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/025_ann_search_and_reranking.ipynb) から同じ評価用の passage の集合を再構成できる。
- `src/training/contrastive_text.py`: in-batch negatives の InfoNCE`info_nce_loss()`、Matryoshka Representation Learning の損失`matryoshka_representation_loss()`、
  学習ループ`train_text_embedding()`、同じ組での損失の測定`compute_pair_losses()`(更新なしの損失)。020 の`src/training/contrastive.py`の`softmax_contrastive_loss()`は参照実装として単体テストで使い(温度を固定した対称な損失と一致する)、
  学習の部品(007 の`AdamW`・warmup + cosine・gradient clipping、011 の`DynamicLossScaler`、019 の`split_weight_decay_parameters()`・
  `unscale_gradients_and_compute_norm()`)は再利用する。
- `src/retrieval/bm25.py`: BM25 のスクラッチ実装`BM25Index`(転置索引、トークン ID の列を入力とする)。
- `src/retrieval/evaluation.py`: 全件の内積による厳密な探索での Recall@k・MRR・nDCG@10`rank_candidates()`・`evaluate_embeddings()`、ランダムな順位の期待値
  `expected_random_recall()`、異方性`mean_pairwise_cosine()`・`alignment()`・`uniformity()`。近似最近傍探索は使わない。

**既存モジュールの扱い**: `src/`は新しいファイルの追加のみである。`scripts/`では、コーパスのアーティファクトの運用スクリプト`scripts/promote_canonical_corpora.py`の英語のスペックに、
記事の境界(`article_offsets`)の追加の対象にするフラグを足した。したがって既存の挙動が変わらないことは、変更前のコミットとの差分が、そのスクリプトの追加の行と説明の 1 行の書き換えだけであることで確かめる
(5.4 節)。因果マスクを外す経路は、`GPTLanguageModel.forward()`を変更せず、`TextEmbeddingModel`が各層の`DecoderBlock`を直接呼んで実現する
(`attention="causal"`のとき`src/models/reward_model.py`の`compute_final_hidden_states()`と bit 単位で一致する)。

**ノートブック内に直接書く(024 固有)**: 条件の定義、データの分割の確認、単体テストと不変条件の確認、学習率の較正、実行計画の選択、判定、可視化、スケーリングの計測、Hub へのアップロード。

**生成物の置き場所**:

| 生成物 | 置き場所 |
|---|---|
| Hugging Face Hub から取得したコーパス・トークナイザ・008 の学習済みモデル | `huggingface_hub`の既定のキャッシュ(Colab ではセッションの終了とともに破棄される) |
| 符号化したトークン列・評価用の索引と query | メモリ上のみ(`.cache/`には置かない。同一セッションの中で全条件・全シードが共有する) |
| 学習したモデル(C5 のシード 0 を除く) | メモリ上のみ(評価の直後に破棄する) |
| C5 のシード 0 の学習後の重み | メモリ上に保持し、6.14 節でアップロードの対象にする(`.cache/`には置かない) |
| 学習の履歴・query ごとの順位 | メモリ上のみ(判定と図に使う) |
| 判定の記録 | 判定と前提条件を計算した各セルの出力(6.7〜6.10・6.13 節) |
| アップロードしたモデル | Hugging Face Hub の`kojikojiprg/ai-theories-text-embedding-en`(新規、public。`UPLOAD_ARTIFACTS`が`True`のときのみ) |

**外部からの取得**(Colab のセットアップセルのリポジトリの取得と依存関係のインストールを除く):

| 取得するもの | リポジトリ | 用途 |
|---|---|---|
| コーパス(`corpus.txt`・`metadata.json`) | `kojikojiprg/ai-theories-corpus-en`(Dataset、9,826 記事、約 546 MB) | 学習・評価の英語 Wikipedia のコーパスと、記事の境界(`article_offsets`)。Wikipedia API にはフォールバックしない |
| 参照コーパス(`corpus.txt`) | `kojikojiprg/ai-theories-corpus-en-pretraining`(Dataset、356 記事、約 24 MB) | 008 の事前学習とトークナイザの学習に使ったコーパス。上のコーパスの先頭と同一であることの確認と、008 が使った範囲の特定 |
| トークナイザ(`tokenizer.json`) | `kojikojiprg/ai-theories-tokenizer-en` | 008 の英語の BPE(Byte Pair Encoding)トークナイザ(語彙サイズ 8192) |
| 008 の学習済みモデル(`config.json`・`model_state.pt`、`main`) | `kojikojiprg/ai-theories-small-gpt-en` | 条件 C1・C2・C3・C5 の初期値 |

**アップロード方針**: アップロードするのは C5(Matryoshka Representation Learning)のシード 0 のモデルだけで、結果を見て選ばない(後続の [025](https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/025_ann_search_and_reranking.ipynb) の入力になるため)。
トークナイザは同梱せず、モデルカードで`kojikojiprg/ai-theories-tokenizer-en`を参照する。モデルカードの指標は、アップロードする重みそのものを読み込み直して評価した値とする。

## 5. 実装 / Implementation

### 5.1 環境セットアップ(Google Colab)

`SMOKE_TEST`(スモークテストか本番か)はこのセルでのみ切り替える。実行環境はこのセルで 1 回だけ印字する。

**本番のコミットの確認**: `SMOKE_TEST = False`のときだけ、(1)追跡しているファイルに未コミットの変更がないこと、(2)`REQUIRED_ANCESTOR_COMMIT`が指定されていれば、
それが HEAD の祖先であることを確かめ、満たさなければ学習の前に停止する。追跡外のファイルは停止の条件にせず、あれば一覧を参考として印字する。

**精度と決定性**: 学習は、CUDA のときは FP16 の`torch.autocast`と動的損失スケーリング(011 の`DynamicLossScaler`)で行い、それ以外のデバイス(ローカルの MPS(Metal Performance Shaders、Apple Silicon の GPU 用のバックエンド)・CPU)では FP32 で行う。
埋め込みの正規化・類似度・損失・評価は常に FP32 で行う。`torch.use_deterministic_algorithms(True)`は使わない。同じシードの学習を繰り返しても結果が bit 単位では
一致しないことがあるが、条件間の対応付け(同じシードの条件どうしで学習の組と順序を揃えること)は乱数の生成器によって決まり、演算の決定性には依存しない(5.5・6.12 節で確かめる)。


```python
# 環境セットアップ(Google Colab)
import os
import sys
import time

SMOKE_TEST = False  # Claude Code はこの True 側のみ実行する(Colab T4 では False に切り替える)
UPLOAD_ARTIFACTS = False  # 6.14 節。既定は False(Colab Secrets の HF_TOKEN を使ってアップロードするときだけ True)

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

# 本番(SMOKE_TEST = False)は、追跡しているファイルに未コミットの変更がない状態で実行する。第 1 段階の完了の後に修正を行った場合は、
# その修正を含むコミットを REQUIRED_ANCESTOR_COMMIT に指定する(HEAD がその子孫であることを確かめる)
REQUIRED_ANCESTOR_COMMIT = None
if not SMOKE_TEST:
    import subprocess

    _head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    _status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True
    ).stdout.strip()
    _untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"], capture_output=True, text=True, check=True
    ).stdout.split()
    _is_descendant = REQUIRED_ANCESTOR_COMMIT is None or (
        subprocess.run(["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR_COMMIT, "HEAD"]).returncode == 0
    )
    if not _is_descendant or _status:
        raise RuntimeError(
            f"本番の実行条件を満たさないため、学習の前に停止する(結果の情報は何も得ていない)。HEAD {_head[:7]}、"
            f"{REQUIRED_ANCESTOR_COMMIT} の子孫か: {_is_descendant}、追跡しているファイルの未コミットの変更: {_status or 'なし'}。"
            "リポジトリを最新の main に更新し、変更をなくして再実行すること。"
        )
    print(
        f"本番の実行条件: HEAD {_head[:7]}、追跡しているファイルに未コミットの変更がない"
        f"、必要な祖先のコミット {REQUIRED_ANCESTOR_COMMIT}: OK(参考: 追跡外のファイル {_untracked or 'なし'})"
    )
print(
    f"SMOKE_TEST={SMOKE_TEST}、学習の精度 {'FP16 の autocast + 動的損失スケーリング' if USE_FP16_AUTOCAST else 'FP32'}"
    f"(埋め込みの正規化・損失・評価は FP32)、決定的な演算の強制: {torch.are_deterministic_algorithms_enabled()}"
)
```

    /content/ai-theories
    [2mUsing Python 3.13.15 environment at: /usr[0m
    [2mChecked [1m60 packages[0m [2min 507ms[0m[0m
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
      コミット / git commit                  : b68fceb94bb90538fc1cdf52dcd45c8293014d65
      未コミットの変更 / uncommitted changes : なし
      実行日時 (UTC)                         : 2026-10-06T08:26:07+00:00
    本番の実行条件: HEAD b68fceb、追跡しているファイルに未コミットの変更がない、必要な祖先のコミット None: OK(参考: 追跡外のファイル なし)
    SMOKE_TEST=False、学習の精度 FP16 の autocast + 動的損失スケーリング(埋め込みの正規化・損失・評価は FP32)、決定的な演算の強制: False



```python
import functools
import gc
import hashlib
import itertools
import json
import math
import resource
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.retrieval import (
    build_retrieval_set,
    compute_article_sampling_weights,
    concatenate_articles,
    encode_articles,
    find_articles_starting_at_or_after,
    sample_pair_schedule,
    select_evenly,
    split_articles,
)
from src.data.text import locate_wikipedia_article_spans, split_train_val_text
from src.data.tokenizer import load_bpe_id_tokenizer_from_hub
from src.layers.feedforward import SwiGLUFeedForwardNetwork
from src.layers.normalization import RMSNorm
from src.layers.positional_encoding import RotaryPositionEmbedding
from src.models.gpt import GPTLanguageModel
from src.models.reward_model import compute_final_hidden_states
from src.models.text_embedding import (
    TextEmbeddingModel,
    compute_hidden_states,
    encode_tokens,
    truncate_and_normalize,
)
from src.retrieval.bm25 import BM25Index
from src.retrieval.evaluation import (
    alignment,
    evaluate_embeddings,
    expected_random_recall,
    mean_pairwise_cosine,
    rank_candidates,
    uniformity,
)
from src.training.contrastive import softmax_contrastive_loss
from src.training.contrastive_text import (
    compute_pair_losses,
    info_nce_loss,
    matryoshka_representation_loss,
    train_text_embedding,
)
from src.utils.reporting import dumps_compact_json
from src.utils.statistics import fit_power_law_exponent, paired_cluster_bootstrap_ratio_of_sums

ROOT = Path.cwd()
WIKIPEDIA_CACHE_DIR = ROOT / ".cache" / "wikipedia_en"  # locate_wikipedia_article_spans() の引数(記事の境界は metadata.json から得るので、Wikipedia API には接続しない)
CORPUS_REPO_ID = "kojikojiprg/ai-theories-corpus-en"  # 9,826 記事。使うのは先頭の 356 記事を除く 9,470 記事
REFERENCE_CORPUS_REPO_ID = "kojikojiprg/ai-theories-corpus-en-pretraining"  # 008 の事前学習と英語のトークナイザの学習に使ったコーパス(CORPUS_REPO_ID の先頭の 356 記事と同一)
TOKENIZER_REPO_ID = "kojikojiprg/ai-theories-tokenizer-en"
REFERENCE_MODEL_REPO_ID = "kojikojiprg/ai-theories-small-gpt-en"  # 008 の学習済みモデル(条件 C1・C2・C3・C5 の初期値)
UPLOAD_REPO_ID = "kojikojiprg/ai-theories-text-embedding-en"
MANIFEST_PATH = ROOT / "src" / "data" / "wikipedia_manifests" / "en_009_scaling.json"
REFERENCE_MANIFEST_PATH = ROOT / "src" / "data" / "wikipedia_manifests" / "en_006_pretraining.json"
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


def timed_call(function) -> float:
    sync_device()
    start = time.time()
    function()
    sync_device()
    return time.time() - start


def rounded(values, digits: int = 4):
    # 印字用: 入れ子のリスト・配列を丸めたリストにする
    return np.round(np.asarray(values, dtype=np.float64), digits).tolist()


precondition_status: dict[str, bool] = {}  # 前提条件の成否(6.1 節で宣言、各節で記録)
```

### 5.2 スケールの設定(`SMOKE_TEST`の配線)

水準の定義をこの 1 箇所に集約する。

**縮小の規則**: スモークテストは、本番と **同じモデル・同じデータ・同じ条件・同じ実行計画の構造** で、次の量だけを縮小する。

- **学習ステップ数 $T$ の候補**: 本番 $(1024, 512, 256)$、スモークテスト $(16, 8, 4)$。どちらも大きい順で、隣どうしの比が 2 の等比という構造を保つ。
- **シード数**: 本番は 5(削った段階で 3)、スモークテストは 3(削った段階で 2)。「削る前 > 削った後 $\ge 2$」の順序関係を保つ。
- **ブートストラップの反復回数**: 本番 10,000 回、スモークテスト 1,000 回。

評価用の索引・query・検証用の集合は縮小しない(評価は数秒で終わるため、本番と同じ集合でコードの経路を確かめる)。スモークテストはステップ数が極端に少ないので、
学習は進まず、前提条件は成立しない見込みである(コードの経路の確認が目的)。**較正・前提条件・検出力の確認は、6.2 節のパイロットで本番と同じステップ数で行った。**

**縮小しないもの**: モデルの構成、データの分割、バッチサイズ、query と passage の長さ、条件の対応表、学習率の較正の格子と拡張の規則、学習率のスケジュールの形、
途中の評価の位置の決め方($T$ に対する割合)、実行計画の表の構造と優先順位、前提条件と判定の閾値、スケーリングの計測点。

**テスト専用の上書き**: 環境変数`AI_THEORIES_FORCE_PLAN`が設定されているときのみ、6.4 節で見積もりによる計画の選択の代わりにその計画を使う(スモークテストで下位の計画の経路を
確かめるため)。本番では受け付けず、このセルで停止する。コミットする出力は上書きなしの実行のものである。

**実効水準の照合**: このセルで印字した水準(ステップ数・シード数・計画の表)が、実際の学習で使われた値と一致することを 6.12 節のアサーションで確かめる(`SMOKE_TEST`の配線漏れの検出)。


```python
# --- 全水準で共通の定数(本番実行前に宣言し、SMOKE_TEST で変えない) ---
# モデル(008 と同じ小型 GPT。構成は Hub の config.json から読む)
EMBEDDING_DIMENSION = 256
MATRYOSHKA_DIMENSIONS = (16, 32, 64, 128, 256)  # 入れ子の次元の集合 M(等比、公比 2)
# データ(5.3 節)
QUERY_LENGTH, PASSAGE_LENGTH = 32, 128  # L_q、L_p(固定長、パディングなし)
MAX_PASSAGES_PER_ARTICLE = 12  # 記事ごとの索引の passage 数の上限(学習で記事を選ぶ重みにも同じ上限を使う)
MAX_QUERIES_PER_ARTICLE = 12
NUM_VALIDATION_ARTICLES, NUM_EVALUATION_ARTICLES, NUM_TRAIN_PROBE_ARTICLES = 100, 300, 120  # 学習用は候補の残りの全記事
SPLIT_SEED, EVALUATION_SET_SEED, VALIDATION_SET_SEED, TRAIN_PROBE_SEED = 24_001, 24_002, 24_003, 24_004
REFERENCE_VALIDATION_RATIO = 0.05  # 008 の事前学習と英語のトークナイザの学習は、参照コーパスの末尾 5% を使っていない
COLAB_RAM_GIB = 12.7  # Colab 無料枠の RAM(メモリ使用量の見込みの判定に使う)
RAM_FRACTION_LIMIT = 0.75  # ピークのメモリ使用量が RAM のこの割合以下であること
LIST_LIKE_TITLE_PREFIXES = ("List of", "Lists of", "Timeline of", "Outline of", "Index of", "Glossary of", "Bibliography of", "Comparison of", "Discography", "Filmography", "Deaths in", "Births in")
EVAL_ENCODE_BATCH = 256  # 評価の 1 回の順伝播の系列の数(結果には影響しない)
# 学習(008 のレシピに準じる)
BATCH_SIZE = 64  # N
TEMPERATURE = 0.05  # tau(固定)
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.1
MIN_LEARNING_RATE_RATIO = 0.01
GRADIENT_CLIP_THRESHOLD = 1.0
INIT_LOSS_SCALE = 2.0**16  # 動的損失スケーリング(019〜022 と同じ)
LOSS_SCALE_GROWTH_INTERVAL = 2000
# 条件の対応表(6.1 節)
CONDITIONS = {
    "C1": {"init": "pretrained", "attention": "causal", "pooling": "last", "loss": "infonce", "track": False},
    "C2": {"init": "pretrained", "attention": "causal", "pooling": "mean", "loss": "infonce", "track": True},
    "C3": {"init": "pretrained", "attention": "bidirectional", "pooling": "mean", "loss": "infonce", "track": True},
    "C4": {"init": "random", "attention": "causal", "pooling": "mean", "loss": "infonce", "track": True},
    "C5": {"init": "pretrained", "attention": "causal", "pooling": "mean", "loss": "matryoshka", "track": False},
}
RUN_ORDER = ("C2", "C1", "C3", "C5", "C4")  # 本番の学習の順
# 学習率の較正(6.1 節)。中心は 6.2 節のパイロットで決めた
LEARNING_RATE_CENTER = {"C1": 0.00024, "C2": 0.00048, "C3": 0.00024, "C4": 0.0006, "C5": 0.00048}
LEARNING_RATE_GRID_MULTIPLIERS = (0.25, 1.0, 4.0)  # 格子 = 中心 x {1/4, 1, 4}(公比 4。6.1 節)
LEARNING_RATE_GRID_RATIO = 4.0
REPRESENTATIVE_CALIBRATED = ("C2", "C4")  # 較正の方式 "representative" で較正する条件
REPRESENTATIVE_RULE = {"C1": "C2", "C3": "C2", "C5": "C2"}  # 規則で決める条件 -> 基にする条件(C2 の較正で選ばれた値 x その条件の格子の中心 / C2 の格子の中心)
# 途中の評価の位置(T に対する割合、初期に密な等比の間隔)。検証用の集合の Recall@10 を測る(C2・C3・C4 のみ)
EVAL_FRACTIONS = (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
# 位置ごとの診断量(実験 A): 系列の長さに対する割合。位置 = ceil(割合 x 長さ) - 1
POSITION_FRACTIONS = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
# 前提条件と判定(6.1 節)
P1_LOSS_RATIO = 0.9  # P1(a): 008 の重みから始める条件の最後の区間の訓練損失 <= 同じ組での学習前のモデルの損失 x この値
FINAL_LOSS_FRACTION = 0.05  # 「最初の区間」「最後の区間」= 最初・最後の 5% のステップ
P1_RECALL_GAIN = 0.04  # P1(b): C2 の検証用の集合の Recall@10 >= 学習前の 008(C2 の構成)の検証用の Recall@10 + この値
P2_RECALL_CEILING = 0.95  # P2(改善の余地): 基準となる条件の Recall@10 <= この値
SIGMA_MULTIPLIER = 2.0  # 判定の閾値は対比量の標準偏差の 2 倍
# 乱数シード(学習のシード s: 学習の組 24100 + s、ランダム初期化 24200 + s)
PAIR_SEED_BASE, INIT_SEED_BASE = 24_100, 24_200
CALIBRATION_SEED_INDEX = 90  # 学習率の較正専用のシード(実験のシード 0〜4 と共有しない)
TIMING_SEED_INDEX = 91  # スケーリングの計測専用のシード
BOOTSTRAP_SEED = 24_600
SESSION_BUDGET_SECONDS = 120 * 60  # 1 セッションの予算(T4 で 120 分)
OBSERVATION_RUN = ("C2", 0)  # 観察 F の「後」に使う学習(C2、シード 0)
UPLOAD_RUN = ("C5", 0)  # Hub にアップロードする学習(結果を見て選ばない)

# --- 水準(SMOKE_TEST で変わるもの) ---
LEVELS = {
    "smoke": {"STEP_CANDIDATES": (16, 8, 4), "BOOTSTRAP_RESAMPLES": 1_000},
    "prod": {"STEP_CANDIDATES": (1024, 512, 256), "BOOTSTRAP_RESAMPLES": 10_000},
}
# 削る段階(6.1 節)。順序: 較正の方式 -> C5 のシード数 -> C4 のシード数(T を下げるのは最後)
STAGES = {
    "prod": {
        0: {"mode": "all", "seeds": {"C1": 5, "C2": 5, "C3": 5, "C4": 5, "C5": 5}},
        1: {"mode": "representative", "seeds": {"C1": 5, "C2": 5, "C3": 5, "C4": 5, "C5": 5}},
        2: {"mode": "representative", "seeds": {"C1": 5, "C2": 5, "C3": 5, "C4": 5, "C5": 3}},
        3: {"mode": "representative", "seeds": {"C1": 5, "C2": 5, "C3": 5, "C4": 3, "C5": 3}},
    },
    "smoke": {  # 本番と同じ構造(シード数 5 -> 3 を 3 -> 2 に縮小)
        0: {"mode": "all", "seeds": {"C1": 3, "C2": 3, "C3": 3, "C4": 3, "C5": 3}},
        1: {"mode": "representative", "seeds": {"C1": 3, "C2": 3, "C3": 3, "C4": 3, "C5": 3}},
        2: {"mode": "representative", "seeds": {"C1": 3, "C2": 3, "C3": 3, "C4": 3, "C5": 2}},
        3: {"mode": "representative", "seeds": {"C1": 3, "C2": 3, "C3": 3, "C4": 2, "C5": 2}},
    },
}
NUM_STAGES = 4
# スケーリングの計測点(6.3 節)。水準によらず同じ
SCALING_STEP_COUNTS = (8, 16, 32)
SCALING_WARMUP_STEPS = 8
SCALING_ENCODE_CHARACTERS = (8_000_000, 16_000_000, 32_000_000)  # 使う記事の先頭から、重ならない区間を順に符号化して計測する

CURRENT_LEVEL_NAME = "smoke" if SMOKE_TEST else "prod"
LEVEL_SETTINGS = LEVELS[CURRENT_LEVEL_NAME]
STEP_CANDIDATES = LEVEL_SETTINGS["STEP_CANDIDATES"]
BOOTSTRAP_RESAMPLES = LEVEL_SETTINGS["BOOTSTRAP_RESAMPLES"]


def build_plans(step_candidates) -> list[dict]:
    # 実行計画(6.1 節): 学習ステップ数 T(大きい順)-> 削る段階(0 -> 3。段階が較正の方式とシード数を決める)
    plans = [{"num_steps": t, "stage": k} for t in step_candidates for k in range(NUM_STAGES)]
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


def learning_rate_grid_for(condition: str) -> tuple[float, ...]:
    return tuple(LEARNING_RATE_CENTER[condition] * m for m in LEARNING_RATE_GRID_MULTIPLIERS)


def calibration_targets(mode: str, seeds: dict) -> list[str]:
    # 較正する条件。"all" は学習する全条件、"representative" は C2 と C4 のみ(6.1 節)
    if mode == "all":
        return [c for c in RUN_ORDER if seeds[c] > 0]
    return [c for c in REPRESENTATIVE_CALIBRATED if seeds[c] > 0]


def position_for(fraction: float, length: int) -> int:
    return math.ceil(fraction * length) - 1


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
        assert set(_stages[_k]["seeds"]) == set(CONDITIONS) and all(v >= 2 for v in _stages[_k]["seeds"].values())
        assert _stages[_k]["seeds"]["C2"] == _stages[0]["seeds"]["C2"]  # 基準の条件は削らない
        assert _stages[_k]["mode"] == ("all" if _k == 0 else "representative")
        if _k > 0:  # 段階が上がるほど、較正の方式かシード数が 1 つだけ削られる
            _changes = int(_stages[_k]["mode"] != _stages[_k - 1]["mode"]) + sum(
                _stages[_k]["seeds"][c] != _stages[_k - 1]["seeds"][c] for c in CONDITIONS
            )
            assert _changes == 1 and all(_stages[_k]["seeds"][c] <= _stages[_k - 1]["seeds"][c] for c in CONDITIONS)
for _k in range(1, NUM_STAGES):  # 本番とスモークテストで、各段階で何が変わるかが同じ(構造を保つ縮小)
    for _c in CONDITIONS:
        assert (STAGES["prod"][_k]["seeds"][_c] == STAGES["prod"][_k - 1]["seeds"][_c]) == (
            STAGES["smoke"][_k]["seeds"][_c] == STAGES["smoke"][_k - 1]["seeds"][_c]
        )
assert len(PLANS) == 3 * NUM_STAGES and [p["plan"] for p in PLANS] == list(range(len(PLANS)))
assert all(b == 2 * a for a, b in itertools.pairwise(SCALING_STEP_COUNTS))
assert all(b == 2 * a for a, b in itertools.pairwise(SCALING_ENCODE_CHARACTERS))
assert all(math.isclose(b / a, 2.0) for a, b in itertools.pairwise(EVAL_FRACTIONS))  # 途中の評価の位置は等比
assert all(math.isclose(b / a, 2.0) for a, b in itertools.pairwise(POSITION_FRACTIONS))  # 位置の診断量の水準は等比
assert all(math.isclose(b / a, 2.0) for a, b in itertools.pairwise(MATRYOSHKA_DIMENSIONS))  # 入れ子の次元は等比
assert MATRYOSHKA_DIMENSIONS[-1] == EMBEDDING_DIMENSION
for _c in CONDITIONS:
    assert all(math.isclose(b / a, LEARNING_RATE_GRID_RATIO) for a, b in itertools.pairwise(learning_rate_grid_for(_c)))
assert set(REPRESENTATIVE_RULE) | set(REPRESENTATIVE_CALIBRATED) == set(CONDITIONS)
assert set(RUN_ORDER) == set(CONDITIONS)
assert QUERY_LENGTH <= PASSAGE_LENGTH <= 256  # passage は 008 の学習長(256)以内

print(f"水準 {CURRENT_LEVEL_NAME!r}: {json.dumps(LEVEL_SETTINGS)}")
print(
    f"データ: query {QUERY_LENGTH} トークン、passage {PASSAGE_LENGTH} トークン(固定長、パディングなし)、記事ごとの passage 数の上限 {MAX_PASSAGES_PER_ARTICLE}、"
    f"検証用 {NUM_VALIDATION_ARTICLES} 記事・評価用 {NUM_EVALUATION_ARTICLES} 記事"
)
print(
    f"学習: バッチ {BATCH_SIZE} 組(in-batch negatives)、温度 {TEMPERATURE}(固定)、AdamW(重み減衰 {WEIGHT_DECAY}、foreach)、"
    f"warmup {WARMUP_RATIO:.0%} + cosine(下限 x{MIN_LEARNING_RATE_RATIO})、gradient clipping {GRADIENT_CLIP_THRESHOLD}。"
    f"入れ子の次元 {MATRYOSHKA_DIMENSIONS}"
)
print(f"条件: {dumps_compact_json(CONDITIONS)}")
print("学習率の格子: " + "、".join(f"{c} {tuple(float(f'{x:.3g}') for x in learning_rate_grid_for(c))}" for c in CONDITIONS))
for _t in STEP_CANDIDATES:
    print(f"  T = {_t}: warmup {warmup_steps_for(_t)}、途中の評価のステップ {eval_steps_for(_t)}、最後の区間 {final_loss_window(_t)} ステップ")
print(f"削る段階({CURRENT_LEVEL_NAME!r}): {dumps_compact_json(STAGES[CURRENT_LEVEL_NAME])}")
print(
    f"実行計画: {len(PLANS)} 通り(番号の小さいほど優先)。計画番号 = {NUM_STAGES} x(T の番号)+ 段階。"
    f"T の候補 {STEP_CANDIDATES}、段階 0〜{NUM_STAGES - 1}"
)
print(f"スケーリングの計測点: ステップ数 {SCALING_STEP_COUNTS}(ウォームアップ {SCALING_WARMUP_STEPS})")
if FORCED_PLAN is not None:
    print(f"*** テスト専用の上書き: AI_THEORIES_FORCE_PLAN = {FORCED_PLAN}(6.4 節で見積もりの代わりにこの計画を使う) ***")
```

    水準 'prod': {"STEP_CANDIDATES": [1024, 512, 256], "BOOTSTRAP_RESAMPLES": 10000}
    データ: query 32 トークン、passage 128 トークン(固定長、パディングなし)、記事ごとの passage 数の上限 12、検証用 100 記事・評価用 300 記事
    学習: バッチ 64 組(in-batch negatives)、温度 0.05(固定)、AdamW(重み減衰 0.1、foreach)、warmup 10% + cosine(下限 x0.01)、gradient clipping 1.0。入れ子の次元 (16, 32, 64, 128, 256)
    条件: {
      "C1": {
        "init": "pretrained",
        "attention": "causal",
        "pooling": "last",
        "loss": "infonce",
        "track": false
      },
      "C2": {
        "init": "pretrained",
        "attention": "causal",
        "pooling": "mean",
        "loss": "infonce",
        "track": true
      },
      "C3": {
        "init": "pretrained",
        "attention": "bidirectional",
        "pooling": "mean",
        "loss": "infonce",
        "track": true
      },
      "C4": {
        "init": "random",
        "attention": "causal",
        "pooling": "mean",
        "loss": "infonce",
        "track": true
      },
      "C5": {
        "init": "pretrained",
        "attention": "causal",
        "pooling": "mean",
        "loss": "matryoshka",
        "track": false
      }
    }
    学習率の格子: C1 (6e-05, 0.00024, 0.00096)、C2 (0.00012, 0.00048, 0.00192)、C3 (6e-05, 0.00024, 0.00096)、C4 (0.00015, 0.0006, 0.0024)、C5 (0.00012, 0.00048, 0.00192)
      T = 1024: warmup 102、途中の評価のステップ (16, 32, 64, 128, 256, 512, 1024)、最後の区間 51 ステップ
      T = 512: warmup 51、途中の評価のステップ (8, 16, 32, 64, 128, 256, 512)、最後の区間 26 ステップ
      T = 256: warmup 26、途中の評価のステップ (4, 8, 16, 32, 64, 128, 256)、最後の区間 13 ステップ
    削る段階('prod'): {
      "0": {
        "mode": "all",
        "seeds": {"C1": 5, "C2": 5, "C3": 5, "C4": 5, "C5": 5}
      },
      "1": {
        "mode": "representative",
        "seeds": {"C1": 5, "C2": 5, "C3": 5, "C4": 5, "C5": 5}
      },
      "2": {
        "mode": "representative",
        "seeds": {"C1": 5, "C2": 5, "C3": 5, "C4": 5, "C5": 3}
      },
      "3": {
        "mode": "representative",
        "seeds": {"C1": 5, "C2": 5, "C3": 5, "C4": 3, "C5": 3}
      }
    }
    実行計画: 12 通り(番号の小さいほど優先)。計画番号 = 4 x(T の番号)+ 段階。T の候補 (1024, 512, 256)、段階 0〜3
    スケーリングの計測点: ステップ数 (8, 16, 32)(ウォームアップ 8)




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/024_text_embedding_and_retriever.ipynb
