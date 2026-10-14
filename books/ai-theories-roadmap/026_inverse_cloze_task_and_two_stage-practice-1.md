---
title: "Inverse Cloze Task と passage 単位の 2 段階検索 / Inverse Cloze Task and Two-Stage Retrieval at Passage Level(実装・実験編 1/8)"
---

この記事は後編(実装・実験編 1/8)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/026_inverse_cloze_task_and_two_stage-theory)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/026_inverse_cloze_task_and_two_stage-practice-2)。

## 4. 実装方針 / Implementation Policy

**`src/`に切り出す(スクラッチ実装、本トピックで新規作成)**:

- `src/data/inverse_cloze.py`: 文の分割の表`build_sentence_tables()`と文の境界`sentence_boundaries()`、抜く文の選択`choose_cloze_sentence()`、窓から文を抜く`remove_span()`・`extract_span()`、
  学習の組`sample_cloze_pair_batches()`(Inverse Cloze Task。1 つの記事から重ならない 2 つの窓を引き、同じバッチに入れる。右側をパディングしたトークンと長さの配列)、passage 単位の評価の索引`build_cloze_index()`(抜く文を決定的に選び、除いた形を索引にする)。
- `src/models/variable_length.py`: 可変長の入力の平均プール`pooled_with_lengths()`、可変長の符号化`encode_tokens_with_lengths()`、可変長の cross-encoder`VariableLengthCrossEncoder`(列を左から詰める)。
  既存の`TextEmbeddingModel`・`CrossEncoder`は変更しない。
- `src/training/contrastive_text_variable.py`: 学習の組を受け取る第 1 段の学習ループ`train_text_embedding_from_batches()`と、同じ組での損失の測定`compute_pair_losses_from_batches()`
  (024 の`train_text_embedding()`と同じ部品・同じ演算。すべての行が満長で、除く組がないときは同じ損失の系列になる)。同じ記事の組を負例から除くマスクつきの損失`masked_info_nce_loss()`・`masked_matryoshka_representation_loss()`を持つ。
- `src/training/reranking_variable.py`: cross-encoder の学習の組`sample_cloze_reranking_schedule()`と学習ループ`train_reranker_variable()`、同じ組での損失の測定`compute_reranking_losses_variable()`。
- `src/retrieval/passage_level.py`: 単一の正例の順位`rank_single_positive()`(同点は正例に不利)、同じ索引・同じ query の記事単位のラベルの順位`rank_article_label()`、並べ替えの順位`rerank_single_positive()`、可変長の cross-encoder による候補のスコア`score_candidates_variable()`。

**既存モジュールの扱い**: `src/`は新しいファイルの追加のみで、既存のファイルを変更しない(5.4 節で、変更前のコミットとの差分で確かめる)。
024 の`src/data/retrieval.py`・`src/retrieval/evaluation.py`・`src/models/text_embedding.py`・`src/training/contrastive_text.py`と、025 の`src/retrieval/`・`src/models/cross_encoder.py`・`src/data/reranking.py`・`src/training/reranking.py`は、そのまま再利用する。

**ノートブック内に直接書く(026 固有)**: 条件の定義、データの分割・コーパスの構成の確認、単体テストと不変条件の確認、スケーリングの計測、パイロット、実行計画の選択、学習率の較正、判定、可視化、Hub へのアップロード。

**生成物の置き場所**:

| 生成物 | 置き場所 |
|---|---|
| Hugging Face Hub から取得したコーパス・トークナイザ・008 のモデル・024 のモデル | `huggingface_hub`の既定のキャッシュ(Colab ではセッションの終了とともに破棄される) |
| 符号化した記事のトークン列 | `.cache/026_encoded/`(入力から決定的に再生成できるもの。同一セッション内の高速化のみを目的とする) |
| 文の分割の表・コーパスの passage の索引・Inverse Cloze Task の索引・学習の組 | メモリ上のみ(いずれも入力とシードから決定的に再生成できる) |
| 学習したモデル(R2 と cross-encoder のシード 0 を除く) | メモリ上のみ(評価の直後に破棄する) |
| R2 のシード 0・cross-encoder のシード 0 の学習後の重み、R1・R2 のシード 0 の索引の埋め込み | メモリ上に保持し、観察 D・E と 6.14 節のアップロードに使う(`.cache/`には置かない) |
| 学習の履歴・query ごとの順位・較正で決めた値・パイロットの結果 | メモリ上のみ(判定と図に使う) |
| 判定の記録 | 判定と前提条件を計算した各セルの出力(6.8〜6.13 節) |
| Hub へアップロードするモデル(R2 のシード 0、cross-encoder のシード 0) | `kojikojiprg/ai-theories-text-embedding-en`の`inverse-cloze-task`ブランチ、`kojikojiprg/ai-theories-cross-encoder-en`(公開の条件を満たし、`UPLOAD_ARTIFACTS`が`True`のときのみ) |

**外部からの取得**(Colab のセットアップセルのリポジトリの取得と依存関係のインストールを除く):

| 取得するもの | リポジトリ | 用途 |
|---|---|---|
| コーパス(`corpus.txt`・`metadata.json`) | `kojikojiprg/ai-theories-corpus-en`(Dataset、9,826 記事、約 546 MB) | 検索の対象のコーパスと記事の境界(`article_offsets`)。Wikipedia API にはフォールバックしない |
| 参照コーパス(`corpus.txt`) | `kojikojiprg/ai-theories-corpus-en-pretraining`(Dataset、356 記事、約 24 MB) | 008 の事前学習とトークナイザの学習に使った範囲の特定(先頭の 356 記事を使わないことの確認) |
| トークナイザ(`tokenizer.json`) | `kojikojiprg/ai-theories-tokenizer-en` | 008 の英語の BPE(Byte Pair Encoding)トークナイザ(語彙サイズ 8192) |
| 008 の学習済みモデル(`config.json`・`model_state.pt`、`main`) | `kojikojiprg/ai-theories-small-gpt-en` | R1・R2 と cross-encoder の起点 |
| 024 の学習済みモデル(`config.json`・`model_state.pt`、`main`) | `kojikojiprg/ai-theories-text-embedding-en` | R1 の再現の確認(診断量)だけに使う。学習の起点にも評価の対象にもしない |

**アップロード方針**: アップロードするのは R2 のシード 0 と cross-encoder のシード 0 のモデルだけで、結果を見て選ばない(後続のアプリ(`apps/`)の入力になるため)。
トークナイザは同梱せず、モデルカードで`kojikojiprg/ai-theories-tokenizer-en`を参照する。モデルカードの指標は、アップロードする重みそのものを読み込み直して評価した値とする。

## 5. 実装 / Implementation

### 5.1 環境セットアップ(Google Colab)

`SMOKE_TEST`(動作確認か本番か)はこのセルでのみ切り替える。実行環境はこのセルで 1 回だけ印字する。

**本番のコミットの確認**: `SMOKE_TEST = False`のときだけ、(1)追跡しているファイルに未コミットの変更がないこと、(2)`REQUIRED_ANCESTOR_COMMIT`が指定されていれば、
それが HEAD の祖先であることを確かめ、満たさなければ学習の前に停止する。追跡外のファイルは停止の条件にせず、あれば一覧を参考として印字する。

**精度と決定性**: 学習は、CUDA のときは FP16 の`torch.autocast`と動的損失スケーリング(011 の`DynamicLossScaler`)で行い、それ以外のデバイス(ローカルの MPS(Metal Performance Shaders、Apple Silicon の GPU 用のバックエンド)・CPU)では FP32 で行う。
埋め込みの正規化・類似度・損失・評価は常に FP32 で行う。`torch.use_deterministic_algorithms(True)`は使わない。同じシードの学習を繰り返しても結果が bit 単位では一致しないことがあるが、
条件間の対応付け(同じシードの条件どうしで記事の並びとバッチを揃えること)は乱数の生成器によって決まり、演算の決定性には依存しない(6.12 節で確かめる)。


```python
# 環境セットアップ(Google Colab)
import os
import sys
import time

SMOKE_TEST = False  # True: 縮小スケールでの動作確認、False: 本番(Colab T4 の 1 セッション)
UPLOAD_ARTIFACTS = True  # 6.14 節。既定は False(Colab Secrets の HF_TOKEN を使ってアップロードするときだけ True)

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

# 本番(SMOKE_TEST = False)は、追跡しているファイルに未コミットの変更がない状態で実行する。特定の修正を含むコミットで実行する場合は、
# そのコミットを REQUIRED_ANCESTOR_COMMIT に指定する(HEAD がその子孫であることを確かめる)
REQUIRED_ANCESTOR_COMMIT = "90b3dde"  # 学習の組の生成の修正の直前のコミット(HEAD はその子孫で、そのコミット自身ではない)
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
        and subprocess.run(["git", "rev-parse", REQUIRED_ANCESTOR_COMMIT + "^{commit}"], capture_output=True, text=True).stdout.strip() != _head
    )
    if not _is_descendant or _status:
        raise RuntimeError(
            f"本番の実行条件を満たさないため、学習の前に停止する(結果の情報は何も得ていない)。HEAD {_head[:7]}、"
            f"{REQUIRED_ANCESTOR_COMMIT} の子孫(自身を除く)か: {_is_descendant}、追跡しているファイルの未コミットの変更: {_status or 'なし'}。"
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
    [2mChecked [1m60 packages[0m [2min 237ms[0m[0m
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
      コミット / git commit                  : 47f25ba2e2087ffc1d4afaca63646ecae1d1bf7e
      未コミットの変更 / uncommitted changes : なし
      実行日時 (UTC)                         : 2026-10-14T04:11:03+00:00
    本番の実行条件: HEAD 47f25ba、追跡しているファイルに未コミットの変更がない、必要な祖先のコミット 90b3dde: OK(参考: 追跡外のファイル なし)
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

from src.data.inverse_cloze import (
    _eligible_window_positions,
    build_cloze_index,
    build_sentence_tables,
    choose_cloze_sentence,
    extract_span,
    pairable_articles,
    remove_span,
    sample_cloze_pair_batches,
    sentence_boundaries,
    sentence_structure,
    splitmix64_uniform,
)
from src.data.reranking import encode_articles_with_cache
from src.data.retrieval import (
    build_retrieval_set,
    concatenate_articles,
    compute_article_sampling_weights,
    find_articles_starting_at_or_after,
    sample_pair_schedule,
    select_evenly,
    split_articles,
)
from src.data.text import locate_wikipedia_article_spans, split_train_val_text
from src.data.tokenizer import load_bpe_id_tokenizer_from_hub, try_decode_byte_level_symbol
from src.layers.feedforward import SwiGLUFeedForwardNetwork
from src.layers.normalization import RMSNorm
from src.layers.positional_encoding import RotaryPositionEmbedding
from src.models.cross_encoder import CrossEncoder, build_cross_encoder_inputs
from src.models.gpt import GPTLanguageModel
from src.models.text_embedding import TextEmbeddingModel, compute_hidden_states, encode_tokens
from src.models.variable_length import (
    VariableLengthCrossEncoder,
    build_variable_cross_encoder_inputs,
    encode_tokens_with_lengths,
    pooled_with_lengths,
)
from src.retrieval.cost_curve import cluster_bootstrap_weights
from src.retrieval.evaluation import evaluate_embeddings, expected_random_recall
from src.retrieval.exact_search import exact_search, remove_excluded_and_truncate
from src.retrieval.hnsw import HierarchicalNavigableSmallWorld
from src.retrieval.passage_level import (
    rank_article_label,
    rank_single_positive,
    rerank_single_positive,
    score_candidates_variable,
)
from src.retrieval.reranking import rank_reranked_candidates
from src.training.contrastive_text import (
    compute_pair_losses,
    info_nce_loss,
    train_text_embedding,
)
from src.training.contrastive_text_variable import (
    VariableBatches,
    compute_pair_losses_from_batches,
    masked_info_nce_loss,
    train_text_embedding_from_batches,
)
from src.training.reranking import softmax_cross_entropy_over_candidates
from src.training.reranking_variable import (
    VariableTables,
    candidate_scores,
    compute_reranking_losses_variable,
    sample_cloze_reranking_schedule,
    train_reranker_variable,
    usable_cloze_queries,
)
from src.utils.reporting import dumps_compact_json
from src.utils.statistics import fit_power_law_exponent, paired_cluster_bootstrap_ratio_of_sums

ROOT = Path.cwd()
WIKIPEDIA_CACHE_DIR = ROOT / ".cache" / "wikipedia_en"  # locate_wikipedia_article_spans() の引数(記事の境界は metadata.json から得るので、Wikipedia API には接続しない)
ENCODED_CACHE_DIR = ROOT / ".cache" / "026_encoded"  # 符号化した記事のトークン列(入力から決定的に再生成できる。同一セッション内の高速化のみが目的)
CORPUS_REPO_ID = "kojikojiprg/ai-theories-corpus-en"  # 9,826 記事。使うのは先頭の 356 記事を除く 9,470 記事
REFERENCE_CORPUS_REPO_ID = "kojikojiprg/ai-theories-corpus-en-pretraining"  # 008 の事前学習と英語のトークナイザの学習に使ったコーパス
TOKENIZER_REPO_ID = "kojikojiprg/ai-theories-tokenizer-en"
REFERENCE_MODEL_REPO_ID = "kojikojiprg/ai-theories-small-gpt-en"  # 008 の学習済みモデル(R1・R2・cross-encoder の起点)
EMBEDDING_REPO_ID = "kojikojiprg/ai-theories-text-embedding-en"  # 024 の学習済みモデル(R1 の再現の確認だけに使う)/ R2 のアップロード先(ブランチ)
EMBEDDING_UPLOAD_BRANCH = "inverse-cloze-task"
CROSS_ENCODER_REPO_ID = "kojikojiprg/ai-theories-cross-encoder-en"
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

**縮小の規則**: 動作確認(`SMOKE_TEST = True`)は、本番と **同じモデル・同じデータの分割・同じ条件・同じ実行計画の構造** で、次の量だけを縮小する。コードの経路が最後まで通ることの確認だけが目的で、数値には意味を持たせない
(学習は進まず、前提条件は成立しない見込みである)。

- **第 1 段の学習ステップ数 $T_1$**: 本番 1024、動作確認 16。
- **cross-encoder の学習ステップ数の候補 $T_2$**: 本番 $(2048, 4096, 8192)$、動作確認 $(16, 32, 64)$。どちらも隣どうしの比が 2 の等比という構造を保つ。
- **シード数**: 本番は $(5, 3, 2)$(削る前、中、低)、動作確認は $(4, 3, 2)$。「削る前 > 中 > 低 $\ge 2$」の順序関係を保つ。
- **コーパスに入れる学習用の記事の数**: 本番はすべて(9,070 記事)、動作確認は 80 記事(検証用 100 記事・評価用 300 記事は縮小しない)。コーパス全体の索引の大きさと、それを使う埋め込みの計算・探索の時間が縮む。
- **観察 D の索引の件数**: 本番 $(16384, 8192)$、動作確認 $(2048, 1024)$(隣どうしの比が 2 の等比)。観察 D の query の数は、本番 600、動作確認 60。
- **cross-encoder の評価に使う query の数**(検証用・評価用それぞれ): 本番はすべて(検証用 約 1,000 個、評価用 約 3,200 個)、動作確認は等間隔に選んだ 64 個(候補 50 件との組の順伝播の回数が query の数に比例して多いため)。
- **ブートストラップの反復回数**: 本番 10,000 回、動作確認 1,000 回。

**縮小しないもの**: モデルの構成、データの分割(024 と同じ)、バッチサイズ、query と passage の長さ、文の分割の規則、条件の対応表、学習率の較正の格子と選択の規則、学習率のスケジュールの形、
cross-encoder の負例の数と候補の数、実行計画の表の構造と優先順位、前提条件と判定の閾値、スケーリングの計測点。第 1 段の評価の query(評価用の約 3,200 個)は縮小しない。

**テスト専用の上書き**: 環境変数`AI_THEORIES_FORCE_STAGE`が設定されているときのみ、6.4 節で見積もりによる計画の選択の代わりにその段階を使う(動作確認で下位の計画の経路を確かめるため)。本番では受け付けず、このセルで停止する。

**実効水準の照合**: このセルで印字した水準が、実際の学習で使われた値と一致することを 6.12 節のアサーションで確かめる(`SMOKE_TEST`の配線漏れの検出)。


```python
# --- 全水準で共通の定数(本番実行前に宣言し、SMOKE_TEST で変えない) ---
# モデル(008 と同じ小型 GPT。構成は Hub の config.json から読む)
EMBEDDING_DIMENSION = 256
MATRYOSHKA_DIMENSIONS = (16, 32, 64, 128, 256)  # 入れ子の次元の集合(024 の C5 と同じ。R1 と R2 で揃える)
# データ(5.3 節)
QUERY_LENGTH, PASSAGE_LENGTH = 32, 128  # L_q(上限)、L_p
MIN_QUERY_TOKENS, MAX_REMOVED_TOKENS = 4, 96  # 抜く文の長さの範囲(文脈は 32 トークン以上残る)
KEEP_PROBABILITY = 0.1  # 抜いた文を文脈に残す確率 r(原論文は 0.1)
MAX_PASSAGES_PER_ARTICLE = 12  # 記事ごとの索引の passage 数の上限(024・025 と同じ)
MAX_QUERIES_PER_ARTICLE = 12
NUM_VALIDATION_ARTICLES, NUM_EVALUATION_ARTICLES = 100, 300
SPLIT_SEED, EVALUATION_SET_SEED, VALIDATION_SET_SEED = 24_001, 24_002, 24_003  # 024 と同じ分割と集合
SPLIT_DIGEST_024, EVALUATION_DIGEST_024, VALIDATION_DIGEST_024 = "716cfe33d05a2615", "da3e16b96714c0a4", "acc9ec62c2a091b0"  # 024 の本番の出力に記録された値
REFERENCE_RECALL_024 = 0.6477  # 024 の C5 のシード 0(Hub の重み)の、評価用の索引での Recall@10(全次元。モデルカード)
CLOZE_SEED = 26_001  # passage 単位の評価の索引で、抜く文を選ぶ乱数のシード
REFERENCE_VALIDATION_RATIO = 0.05  # 008 の事前学習と英語のトークナイザの学習は、参照コーパスの末尾 5% を使っていない
LIST_LIKE_TITLE_PREFIXES = ("List of", "Lists of", "Timeline of", "Outline of", "Index of", "Glossary of", "Bibliography of", "Comparison of", "Discography", "Filmography", "Deaths in", "Births in")
COLAB_RAM_GIB = 12.7  # Colab 無料枠の RAM(メモリ使用量の見込みの判定に使う)
RAM_FRACTION_LIMIT = 0.75  # ピークのメモリ使用量が RAM のこの割合以下であること
EMBED_BATCH = 512  # 埋め込みの計算の 1 回の順伝播の系列の数(結果には影響しない)
SEPARATOR_BYTE, TERMINAL_BYTE = 0x1F, 0x1E  # cross-encoder の区切りと終端に使う単一バイトのトークン(制御文字。語彙を増やさない)
# 第 1 段の学習(024 の条件 C5 に準じる)
BATCH_SIZE = 64  # N(1 バッチの組の数)
PAIRS_PER_ARTICLE = 2  # 1 つの記事から同じバッチに入れる組の数(窓は互いに重ならない)
ARTICLES_PER_BATCH = BATCH_SIZE // PAIRS_PER_ARTICLE  # 1 バッチの記事の数(32)
TRAINING_SAFETY_FACTOR = 1.1  # 見積もりで、学習の時間に掛ける安全係数
TEMPERATURE = 0.05  # tau(固定)
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.1
MIN_LEARNING_RATE_RATIO = 0.01
GRADIENT_CLIP_THRESHOLD = 1.0
INIT_LOSS_SCALE = 2.0**16
LOSS_SCALE_GROWTH_INTERVAL = 2000
FIRST_STAGE_CONDITIONS = ("R1", "R2")  # R1: 記事単位の定義、R2: passage 単位の定義(Inverse Cloze Task)
FIRST_STAGE_LEARNING_RATE_CENTER = {"R1": 4.8e-4, "R2": 4.8e-4}  # 024 の C5 の較正で選ばれた値(T = 1024)を、どちらの条件の格子の中心にも使う(組の作り方が違うので、最適の保証はない)
LEARNING_RATE_GRID_MULTIPLIERS = (0.25, 1.0, 4.0)  # 格子 = 中心 x {1/4, 1, 4}(公比 4)
LEARNING_RATE_GRID_RATIO = 4.0
R1_RULE_LEARNING_RATE = 4.8e-4  # 削る段階 4 以降: R1 の学習率は較正せず、024 の C5(同じ構成・同じ T_1。組の作り方は違う)の較正で選ばれた値にする
# cross-encoder(025 と同じ構成)
BATCH_QUERIES, NUM_NEGATIVES = 8, 7  # B、n(1 ステップ = B 個の query x (正例 1 + 負例 n))
RERANK_CANDIDATES = 50  # K(並べ替える候補の数 = 採掘の深さ)
CROSS_ENCODER_LEARNING_RATE_CENTER = 2.4e-4  # 025 の D1 のパイロットで、最後の区間の訓練損失が最も低かった学習率
CROSS_ENCODER_RULE_LEARNING_RATE = CROSS_ENCODER_LEARNING_RATE_CENTER  # 削る段階 5 以降: cross-encoder の学習率は較正せず、中心にする
HEAD_INIT_STD = 0.02
SCORE_BATCH = 512  # 並べ替えの推論の 1 回の順伝播の系列の数(結果には影響しない)
# 前提条件と判定(6.1 節)
FINAL_LOSS_FRACTION = 0.05  # 「最後の区間」= 最後の 5% のステップ
P1_LOSS_RATIO = 0.9  # 第 1 段の P1(a): 最後の区間の訓練損失 <= 同じ組での学習前のモデルの損失 x この値
P1_VALIDATION_GAIN = 0.04  # 第 1 段の P1(b): 自分の定義の検証用の Recall@10 >= 学習前の 008 の値 + この値
P1_SIGMA_MULTIPLIER = 2.0  # cross-encoder の P1: 差の平均が標準誤差(標準偏差)のこの倍数以上
P2_FLOOR, P2_CEILING = 0.02, 0.95  # 実験 A・B の P2: 基準の条件の Recall@10 が [床, 天井] の内側
P2_ROOM = 0.05  # 実験 C の P2: R2 だけの Recall@10 <= coverage - この値
P3_MIN_COVERAGE, P3_MIN_HEADROOM = 0.35, 0.10  # 実験 C の P3: coverage の下限と、並べ替えの余地(coverage - R2 だけの Recall@10)の下限
SIGMA_MULTIPLIER = 2.0  # 判定の閾値は対比量の標準偏差の 2 倍
SELECTION_SIGMA_MULTIPLIER = 1.0  # 較正の選択: 最良との差がこの倍数の標準偏差以内の候補の中では、学習の成立の余裕が最も大きいものを選ぶ
# 乱数シード(学習のシード s: 第 1 段の組(記事の並びを含む。R1 と R2 で共通)26100 + s、cross-encoder の組 26500 + s、ヘッドの初期化 26200 + s)
CLOZE_BATCH_SEED_BASE, INIT_SEED_BASE, CROSS_ENCODER_SCHEDULE_SEED_BASE = 26_100, 26_200, 26_500
CALIBRATION_SEED_INDEX = 90  # 学習率の較正とパイロット専用のシード(実験のシード 0〜 と共有しない)
TIMING_SEED_INDEX = 91  # スケーリングの計測専用のシード
BOOTSTRAP_SEED = 26_600
SESSION_BUDGET_SECONDS = 120 * 60  # 1 セッションの予算(T4 で 120 分)
REPORTING_MARGIN_SECONDS = 300.0  # 判定・ブートストラップ・図・アップロードの準備の固定の余裕
RUN_OVERHEAD_SECONDS = 2.0  # 学習 1 回あたりのモデルの構築などの固定費の余裕
UPLOAD_SEED = 0  # Hub にアップロードする学習のシード(結果を見て選ばない)
UNIFORM_LOSS_CROSS_ENCODER = math.log(1 + NUM_NEGATIVES)  # cross-encoder の一様な予測の損失 ln(1 + n)
# 観察 D(2 段階の検索の費用)
HNSW_MAX_CONNECTIONS, HNSW_CONSTRUCTION_WIDTH = 5, 100  # M、efConstruction(025 と同じ)
OBSERVATION_RERANK_KS = (10, 20, 50, 100)  # 並べ替える候補の数 K
OBSERVATION_SEARCH_WIDTHS = (101, 127, 160, 202, 254)  # HNSW の探索の幅 ef の格子(公比 2^(1/3)、最小は max(K) + 1)
OBSERVATION_TARGET_RECALL = 0.9  # 第 1 段の近似の目標(全探索の上位 100 件との一致率)
OBSERVATION_SEED = 26_900

# --- 水準(SMOKE_TEST で変わるもの) ---
LEVELS = {
    "smoke": {
        "FIRST_STAGE_STEPS": 16, "CROSS_ENCODER_STEP_CANDIDATES": (16, 32, 64), "SEEDS": (4, 3, 2), "CORPUS_TRAIN_ARTICLES": 80,
        "OBSERVATION_N_CANDIDATES": (2048, 1024), "OBSERVATION_QUERIES": 60, "RERANK_QUERIES": 64, "BOOTSTRAP_RESAMPLES": 1_000, "HNSW_TIMING_SIZES": (128, 256, 512),
    },
    "prod": {
        "FIRST_STAGE_STEPS": 1024, "CROSS_ENCODER_STEP_CANDIDATES": (2048, 4096, 8192), "SEEDS": (5, 3, 2), "CORPUS_TRAIN_ARTICLES": None,
        "OBSERVATION_N_CANDIDATES": (16384, 8192), "OBSERVATION_QUERIES": 600, "RERANK_QUERIES": None, "BOOTSTRAP_RESAMPLES": 10_000, "HNSW_TIMING_SIZES": (1024, 2048, 4096),
    },
}
CURRENT_LEVEL_NAME = "smoke" if SMOKE_TEST else "prod"
LEVEL_SETTINGS = LEVELS[CURRENT_LEVEL_NAME]
FIRST_STAGE_STEPS = LEVEL_SETTINGS["FIRST_STAGE_STEPS"]  # T_1(第 1 段の学習ステップ数。全条件・全段階で共通)
CROSS_ENCODER_STEP_CANDIDATES = LEVEL_SETTINGS["CROSS_ENCODER_STEP_CANDIDATES"]  # T_2 の候補(パイロットで選ぶ)
CORPUS_TRAIN_ARTICLES = LEVEL_SETTINGS["CORPUS_TRAIN_ARTICLES"]
RERANK_QUERIES = LEVEL_SETTINGS["RERANK_QUERIES"]  # cross-encoder の評価(検証用・評価用)に使う query の数(None はすべて)
BOOTSTRAP_RESAMPLES = LEVEL_SETTINGS["BOOTSTRAP_RESAMPLES"]

# 削る段階(6.1 節)。各段階は前の段階に 1 つだけ変更を加える(累積)
STAGE_CHANGES = (
    "(削らない)",
    "観察 D の索引の件数を 1 段下げる",
    "cross-encoder のシード数を「中」にする",
    "第 1 段(R1・R2)のシード数を「中」にする",
    "R1 の学習率の較正をやめ、024 の値にする",
    "cross-encoder の学習率の較正をやめ、中心の値にする",
    "cross-encoder のシード数を「低」にする",
    "第 1 段(R1・R2)のシード数を「低」にする",
)
NUM_STAGES = len(STAGE_CHANGES)


def stage_settings(stage: int, level: str = CURRENT_LEVEL_NAME) -> dict:
    # 段階 stage の実効の設定(累積)
    full, mid, low = LEVELS[level]["SEEDS"]
    return {
        "observation_n": LEVELS[level]["OBSERVATION_N_CANDIDATES"][1 if stage >= 1 else 0],
        "cross_encoder_seeds": low if stage >= 6 else mid if stage >= 2 else full,
        "first_stage_seeds": low if stage >= 7 else mid if stage >= 3 else full,
        "calibrate_r1": stage < 4,
        "calibrate_ce": stage < 5,
    }


FORCED_STAGE_VALUE = os.environ.get("AI_THEORIES_FORCE_STAGE")
if FORCED_STAGE_VALUE is not None and not SMOKE_TEST:
    raise RuntimeError(
        f"本番(SMOKE_TEST=False)では段階の強制(AI_THEORIES_FORCE_STAGE={FORCED_STAGE_VALUE!r})を受け付けない。環境変数を削除して再実行すること。"
    )
FORCED_STAGE = None if FORCED_STAGE_VALUE is None else int(FORCED_STAGE_VALUE)
assert FORCED_STAGE is None or 0 <= FORCED_STAGE < NUM_STAGES, FORCED_STAGE


def warmup_steps_for(num_steps: int) -> int:
    return max(1, round(WARMUP_RATIO * num_steps))


def final_loss_window(num_steps: int) -> int:
    return max(1, round(FINAL_LOSS_FRACTION * num_steps))


def first_stage_grid(condition: str) -> tuple[float, ...]:
    return tuple(FIRST_STAGE_LEARNING_RATE_CENTER[condition] * m for m in LEARNING_RATE_GRID_MULTIPLIERS)


def cross_encoder_grid() -> tuple[float, ...]:
    return tuple(CROSS_ENCODER_LEARNING_RATE_CENTER * m for m in LEARNING_RATE_GRID_MULTIPLIERS)


# --- 縮小規則と水準の構造の確認 ---
assert set(LEVELS["smoke"]) == set(LEVELS["prod"])
for _name, _settings in LEVELS.items():
    _candidates = _settings["CROSS_ENCODER_STEP_CANDIDATES"]
    assert len(_candidates) == 3 and all(b == 2 * a for a, b in itertools.pairwise(_candidates))  # 等比(公比 2)
    _full, _mid, _low = _settings["SEEDS"]
    assert _full > _mid > _low >= 2  # 「削る前 > 中 > 低 >= 2」
    assert len(_settings["OBSERVATION_N_CANDIDATES"]) == 2 and _settings["OBSERVATION_N_CANDIDATES"][0] == 2 * _settings["OBSERVATION_N_CANDIDATES"][1]
    assert len(_settings["HNSW_TIMING_SIZES"]) == 3 and all(b == 2 * a for a, b in itertools.pairwise(_settings["HNSW_TIMING_SIZES"]))
    for _t in (_settings["FIRST_STAGE_STEPS"], *_candidates):
        assert warmup_steps_for(_t) < _t
assert LEVELS["smoke"]["BOOTSTRAP_RESAMPLES"] < LEVELS["prod"]["BOOTSTRAP_RESAMPLES"] and LEVELS["smoke"]["FIRST_STAGE_STEPS"] < LEVELS["prod"]["FIRST_STAGE_STEPS"]
assert LEVELS["smoke"]["OBSERVATION_QUERIES"] < LEVELS["prod"]["OBSERVATION_QUERIES"] and LEVELS["prod"]["CORPUS_TRAIN_ARTICLES"] is None and LEVELS["prod"]["RERANK_QUERIES"] is None
for _level in LEVELS:  # 段階ごとの変更が累積し、各段階で 1 つだけ変わる(非増加)。cross-encoder のシード数 <= 第 1 段のシード数
    for _k in range(1, NUM_STAGES):
        _before, _after = stage_settings(_k - 1, _level), stage_settings(_k, _level)
        _changed = [key for key in _before if _before[key] != _after[key]]
        assert len(_changed) == 1, (_level, _k, _changed)
        assert all(int(_after[key]) <= int(_before[key]) for key in _before)  # 数値は減る、真偽は True -> False
        assert _after["cross_encoder_seeds"] <= _after["first_stage_seeds"]
for _k in range(1, NUM_STAGES):  # 本番と動作確認で、各段階で何が変わるかが同じ
    assert [key for key in stage_settings(_k, "smoke") if stage_settings(_k, "smoke")[key] != stage_settings(_k - 1, "smoke")[key]] == [
        key for key in stage_settings(_k, "prod") if stage_settings(_k, "prod")[key] != stage_settings(_k - 1, "prod")[key]
    ]
assert all(math.isclose(b / a, LEARNING_RATE_GRID_RATIO) for a, b in itertools.pairwise(LEARNING_RATE_GRID_MULTIPLIERS))
assert R1_RULE_LEARNING_RATE == FIRST_STAGE_LEARNING_RATE_CENTER["R1"] and CROSS_ENCODER_RULE_LEARNING_RATE == CROSS_ENCODER_LEARNING_RATE_CENTER
assert BATCH_SIZE % PAIRS_PER_ARTICLE == 0 and PAIRS_PER_ARTICLE == 2 and TRAINING_SAFETY_FACTOR >= 1.0
assert QUERY_LENGTH <= PASSAGE_LENGTH and QUERY_LENGTH + 1 + PASSAGE_LENGTH + 1 <= 256  # cross-encoder の列は 008 の学習長(256)以内
assert MIN_QUERY_TOKENS <= QUERY_LENGTH and PASSAGE_LENGTH - MAX_REMOVED_TOKENS >= QUERY_LENGTH  # 文脈は query の長さ以上残る
assert max(OBSERVATION_RERANK_KS) + 1 <= OBSERVATION_SEARCH_WIDTHS[0] and 10 in OBSERVATION_RERANK_KS and RERANK_CANDIDATES in OBSERVATION_RERANK_KS
assert NUM_NEGATIVES + 1 <= RERANK_CANDIDATES
assert MATRYOSHKA_DIMENSIONS[-1] == EMBEDDING_DIMENSION and all(math.isclose(b / a, 2.0) for a, b in itertools.pairwise(MATRYOSHKA_DIMENSIONS))

print(f"水準 {CURRENT_LEVEL_NAME!r}: {json.dumps({k: list(v) if isinstance(v, tuple) else v for k, v in LEVEL_SETTINGS.items()})}")
print(
    f"データ: query {QUERY_LENGTH} トークン(上限)、passage {PASSAGE_LENGTH} トークン(窓)、記事ごとの passage 数の上限 {MAX_PASSAGES_PER_ARTICLE}、検証用 {NUM_VALIDATION_ARTICLES} 記事・評価用 {NUM_EVALUATION_ARTICLES} 記事(024 と同じ分割)。"
    f"Inverse Cloze Task: 抜く文の長さ {MIN_QUERY_TOKENS}〜{MAX_REMOVED_TOKENS} トークン、抜いた文を文脈に残す確率 {KEEP_PROBABILITY}"
)
print(
    f"第 1 段の学習: バッチ {BATCH_SIZE} 組({ARTICLES_PER_BATCH} 記事 x {PAIRS_PER_ARTICLE} 組、in-batch negatives)、温度 {TEMPERATURE}(固定)、Matryoshka Representation Learning {MATRYOSHKA_DIMENSIONS}、AdamW(重み減衰 {WEIGHT_DECAY}、foreach)、"
    f"warmup {WARMUP_RATIO:.0%} + cosine(下限 x{MIN_LEARNING_RATE_RATIO})、gradient clipping {GRADIENT_CLIP_THRESHOLD}、T_1 = {FIRST_STAGE_STEPS}"
)
print("第 1 段の学習率の格子: " + "、".join(f"{c} {tuple(float(f'{x:.3g}') for x in first_stage_grid(c))}" for c in FIRST_STAGE_CONDITIONS))
print(
    f"cross-encoder: 1 ステップ {BATCH_QUERIES} query x (正例 1 + 負例 {NUM_NEGATIVES})、並べ替える候補 K = {RERANK_CANDIDATES}、T_2 の候補 {CROSS_ENCODER_STEP_CANDIDATES}、"
    f"学習率の格子 {tuple(float(f'{x:.3g}') for x in cross_encoder_grid())}"
)
print(f"削る段階({CURRENT_LEVEL_NAME!r}): " + "、".join(f"段階 {k}({STAGE_CHANGES[k]}): {json.dumps(stage_settings(k))}" for k in range(NUM_STAGES)))
if FORCED_STAGE is not None:
    print(f"*** テスト専用の上書き: AI_THEORIES_FORCE_STAGE = {FORCED_STAGE}(6.4 節で見積もりの代わりにこの段階を使う) ***")
```

    水準 'prod': {"FIRST_STAGE_STEPS": 1024, "CROSS_ENCODER_STEP_CANDIDATES": [2048, 4096, 8192], "SEEDS": [5, 3, 2], "CORPUS_TRAIN_ARTICLES": null, "OBSERVATION_N_CANDIDATES": [16384, 8192], "OBSERVATION_QUERIES": 600, "RERANK_QUERIES": null, "BOOTSTRAP_RESAMPLES": 10000, "HNSW_TIMING_SIZES": [1024, 2048, 4096]}
    データ: query 32 トークン(上限)、passage 128 トークン(窓)、記事ごとの passage 数の上限 12、検証用 100 記事・評価用 300 記事(024 と同じ分割)。Inverse Cloze Task: 抜く文の長さ 4〜96 トークン、抜いた文を文脈に残す確率 0.1
    第 1 段の学習: バッチ 64 組(32 記事 x 2 組、in-batch negatives)、温度 0.05(固定)、Matryoshka Representation Learning (16, 32, 64, 128, 256)、AdamW(重み減衰 0.1、foreach)、warmup 10% + cosine(下限 x0.01)、gradient clipping 1.0、T_1 = 1024
    第 1 段の学習率の格子: R1 (0.00012, 0.00048, 0.00192)、R2 (0.00012, 0.00048, 0.00192)
    cross-encoder: 1 ステップ 8 query x (正例 1 + 負例 7)、並べ替える候補 K = 50、T_2 の候補 (2048, 4096, 8192)、学習率の格子 (6e-05, 0.00024, 0.00096)
    削る段階('prod'): 段階 0((削らない)): {"observation_n": 16384, "cross_encoder_seeds": 5, "first_stage_seeds": 5, "calibrate_r1": true, "calibrate_ce": true}、段階 1(観察 D の索引の件数を 1 段下げる): {"observation_n": 8192, "cross_encoder_seeds": 5, "first_stage_seeds": 5, "calibrate_r1": true, "calibrate_ce": true}、段階 2(cross-encoder のシード数を「中」にする): {"observation_n": 8192, "cross_encoder_seeds": 3, "first_stage_seeds": 5, "calibrate_r1": true, "calibrate_ce": true}、段階 3(第 1 段(R1・R2)のシード数を「中」にする): {"observation_n": 8192, "cross_encoder_seeds": 3, "first_stage_seeds": 3, "calibrate_r1": true, "calibrate_ce": true}、段階 4(R1 の学習率の較正をやめ、024 の値にする): {"observation_n": 8192, "cross_encoder_seeds": 3, "first_stage_seeds": 3, "calibrate_r1": false, "calibrate_ce": true}、段階 5(cross-encoder の学習率の較正をやめ、中心の値にする): {"observation_n": 8192, "cross_encoder_seeds": 3, "first_stage_seeds": 3, "calibrate_r1": false, "calibrate_ce": false}、段階 6(cross-encoder のシード数を「低」にする): {"observation_n": 8192, "cross_encoder_seeds": 2, "first_stage_seeds": 3, "calibrate_r1": false, "calibrate_ce": false}、段階 7(第 1 段(R1・R2)のシード数を「低」にする): {"observation_n": 8192, "cross_encoder_seeds": 2, "first_stage_seeds": 2, "calibrate_r1": false, "calibrate_ce": false}




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/026_inverse_cloze_task_and_two_stage_retrieval.ipynb
