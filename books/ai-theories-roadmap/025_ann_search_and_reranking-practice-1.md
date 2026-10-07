---
title: "ANN 検索とリランキング / ANN Search and Reranking(実装・実験編 1/8)"
---

この記事は後編(実装・実験編 1/8)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/025_ann_search_and_reranking-theory)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/025_ann_search_and_reranking-practice-2)。

## 4. 実装方針 / Implementation Policy

**`src/`に切り出す(スクラッチ実装、本トピックで新規作成)**:

- `src/retrieval/exact_search.py`: 内積による全探索`exact_search()`(query の元の passage を候補から除く)、距離計算の回数を種類別に数える`CostCounter`、返した近傍と正解の一致率`recall_against_ground_truth()`、
  近似探索の結果から除くベクトルを取り除く`remove_excluded_and_truncate()`。
- `src/retrieval/kmeans.py`: Lloyd のアルゴリズムによる k-means`fit_kmeans()`(転置ファイルと Product Quantization で共用。初期化と反復回数は固定)。
- `src/retrieval/inverted_file.py`: 転置ファイル`InvertedFileIndex`(重心との比較と走査したベクトルを別々に数える。評価用に、複数の探索するリストの数を一括で評価する`search_batch_by_probe_counts()`を持ち、
  1 つずつ探索する`search()`と結果・費用が一致することを単体テストで確かめる)。
- `src/retrieval/hnsw.py`: HNSW`HierarchicalNavigableSmallWorld`(NumPy。Algorithm 1〜5、近傍の選択のヒューリスティック。1 つの頂点の近傍への距離は行列積で 1 回にまとめる。層の番号と挿入の順序はシードで固定する)。
- `src/retrieval/product_quantization.py`: `ProductQuantizer`(符号帳の学習・符号化・復元)と`ProductQuantizationIndex`(非対称距離計算・対称距離計算・再採点。参照表の作成と表引きを別々に数える)。
- `src/retrieval/cost_curve.py`: recall と費用の曲線(格子の掃引、目標の recall での費用の補間、ブートストラップの再標本での一括の補間、べき指数の傾き)。
- `src/retrieval/reranking.py`: 並べ替えの推論と評価(cross-encoder と dual encoder の候補のスコア、並べ替えた後の順位と指標)。
- `src/models/cross_encoder.py`: cross-encoder`CrossEncoder`(008 の小型 GPT の本体 + 終端の位置の出力へのスカラーのヘッド)。
- `src/data/reranking.py`: 並べ替えの学習の組`sample_reranking_schedule()`、第 1 段の上位の採掘`mine_candidate_lists()`、符号化済みの記事のキャッシュ`encode_articles_with_cache()`。
- `src/training/reranking.py`: 並べ替えモデルの学習ループ`train_reranker()`(cross-encoder と dual encoder で共通)、同じ組での損失の測定`compute_reranking_losses()`。

**既存モジュールの扱い**: `src/`は新しいファイルの追加のみで、既存のファイルを変更しない(単体テストの前後比較で確かめる。5.4 節)。024 の`src/data/retrieval.py`・`src/retrieval/evaluation.py`・`src/models/text_embedding.py`は、
そのまま再利用する。

**ノートブック内に直接書く(025 固有)**: 実験の水準と条件の定義、データの分割・コーパスの構成の確認、単体テストと不変条件の確認、スケーリングの計測と実行計画の選択、学習率の較正、索引の構築と掃引、
判定、可視化、Hub へのアップロード。

**生成物の置き場所**:

| 生成物 | 置き場所 |
|---|---|
| Hugging Face Hub から取得したコーパス・トークナイザ・008 と 024 のモデル | `huggingface_hub`の既定のキャッシュ(Colab ではセッションの終了とともに破棄される) |
| 符号化した記事のトークン列、コーパスの passage と query の埋め込み | `.cache/025_encoded/`・`.cache/025_embeddings/`(入力から決定的に再生成できるもの。同一セッション内の高速化のみを目的とする) |
| 索引(HNSW・転置ファイル・Product Quantization)・k-means の重心・符号帳 | メモリ上のみ(評価の直後に破棄する。実験 F の第 1 段のため、シード 0 の最大の $N$ の索引だけを保持する) |
| 学習したモデル(D1 のシード 0 を除く) | メモリ上のみ(評価の直後に破棄する) |
| D1 のシード 0 の学習後の重み | メモリ上に保持し、実験 F の並べ替えと 6.16 節のアップロードに使う(`.cache/`には置かない) |
| 学習の履歴・query ごとの順位や recall と費用 | メモリ上のみ(判定と図に使う) |
| 判定の記録 | 判定と前提条件を計算した各セルの出力(6.8〜6.12 節) |
| Hub へアップロードするモデル(D1 のシード 0) | 公開の条件(6.16 節)を満たし、`UPLOAD_ARTIFACTS`が`True`のときのみ。**本番では公開の条件を満たさず、アップロードは行われなかった(7.10 節)** |

**外部からの取得**(Colab のセットアップセルのリポジトリの取得と依存関係のインストールを除く):

| 取得するもの | リポジトリ | 用途 |
|---|---|---|
| コーパス(`corpus.txt`・`metadata.json`) | `kojikojiprg/ai-theories-corpus-en`(Dataset、9,826 記事、約 546 MB) | 検索の対象のコーパスと記事の境界(`article_offsets`) |
| 参照コーパス(`corpus.txt`) | `kojikojiprg/ai-theories-corpus-en-pretraining`(Dataset、356 記事、約 24 MB) | 008 の事前学習とトークナイザの学習に使った範囲の特定(先頭の 356 記事を使わないことの確認) |
| トークナイザ(`tokenizer.json`) | `kojikojiprg/ai-theories-tokenizer-en` | 008 の英語の BPE(Byte Pair Encoding)トークナイザ(語彙サイズ 8192) |
| 008 の学習済みモデル(`config.json`・`model_state.pt`、`main`) | `kojikojiprg/ai-theories-small-gpt-en` | cross-encoder と dual encoder の起点 |
| 024 の学習済みモデル(`config.json`・`model_state.pt`) | `kojikojiprg/ai-theories-text-embedding-en` | 第 1 段の埋め込み(凍結) |

**アップロード方針**: アップロードの対象は D1(困難な負例で学習した cross-encoder)のシード 0 のモデルだけで、結果を見て選ばない(後続のアプリ(`apps/`)の入力になるため)。トークナイザは同梱せず、
モデルカードで`kojikojiprg/ai-theories-tokenizer-en`を参照する。モデルカードの指標は、アップロードする重みそのものを読み込み直して評価した値とする。**本番では公開の条件を満たさず、アップロードは行われなかった(7.10 節)。**

## 5. 実装 / Implementation

### 5.1 環境セットアップ(Google Colab)

`SMOKE_TEST`(スモークテストか本番か)はこのセルでのみ切り替える。実行環境はこのセルで 1 回だけ印字する。

**本番のコミットの確認**: `SMOKE_TEST = False`のときだけ、(1)追跡しているファイルに未コミットの変更がないこと、(2)`REQUIRED_ANCESTOR_COMMIT`が指定されていれば、
それが HEAD の祖先であることを確かめ、満たさなければ索引の構築と学習の前に停止する。追跡外のファイルは停止の条件にせず、あれば一覧を参考として印字する。

**精度と決定性**: 並べ替えモデルの学習と評価は、CUDA のときは本体の順伝播を FP16 の`torch.autocast`と動的損失スケーリング(011 の`DynamicLossScaler`)で行い、
それ以外のデバイス(ローカルの MPS(Metal Performance Shaders、Apple Silicon の GPU 用のバックエンド)・CPU)では FP32 で行う。スコアの計算・損失・指標は常に FP32 で行う。
第 1 段の埋め込み(024 のモデル)の計算は常に FP32 である。近似最近傍探索の索引(HNSW・転置ファイル・Product Quantization)は CPU で、FP32 で計算する。
`torch.use_deterministic_algorithms(True)`は使わない。同じシードの学習を繰り返しても結果が bit 単位では一致しないことがあるが、条件間の対応付け(同じシードの条件どうしで学習の組と順序を揃えること)は
乱数の生成器によって決まり、演算の決定性には依存しない(6.14 節で確かめる)。


```python
# 環境セットアップ(Google Colab)
import os
import sys
import time

SMOKE_TEST = False  # Claude Code はこの True 側のみ実行する(Colab T4 では False に切り替える)
UPLOAD_ARTIFACTS = True  # 6.16 節。既定は False(Colab Secrets の HF_TOKEN を使ってアップロードするときだけ True)

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
REQUIRED_ANCESTOR_COMMIT = "df661bc"  # 第 1 段階の追加の修正の直前のコミット(HEAD がその子孫であることを確かめる)
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
            f"本番の実行条件を満たさないため、索引の構築と学習の前に停止する(結果の情報は何も得ていない)。HEAD {_head[:7]}、"
            f"{REQUIRED_ANCESTOR_COMMIT} の子孫か: {_is_descendant}、追跡しているファイルの未コミットの変更: {_status or 'なし'}。"
            "リポジトリを最新の main に更新し、変更をなくして再実行すること。"
        )
    print(
        f"本番の実行条件: HEAD {_head[:7]}、追跡しているファイルに未コミットの変更がない"
        f"、必要な祖先のコミット {REQUIRED_ANCESTOR_COMMIT}: OK(参考: 追跡外のファイル {_untracked or 'なし'})"
    )
print(
    f"SMOKE_TEST={SMOKE_TEST}、並べ替えモデルの精度 {'FP16 の autocast + 動的損失スケーリング' if USE_FP16_AUTOCAST else 'FP32'}"
    f"(スコア・損失・指標は FP32、索引は CPU の FP32)、決定的な演算の強制: {torch.are_deterministic_algorithms_enabled()}"
)
```

    /content/ai-theories
    [2mUsing Python 3.13.15 environment at: /usr[0m
    [2mChecked [1m60 packages[0m [2min 219ms[0m[0m
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
      コミット / git commit                  : 2d508485e52b78d7786dc9475fb27cc59bbdd906
      未コミットの変更 / uncommitted changes : なし
      実行日時 (UTC)                         : 2026-10-07T07:28:55+00:00
    本番の実行条件: HEAD 2d50848、追跡しているファイルに未コミットの変更がない、必要な祖先のコミット df661bc: OK(参考: 追跡外のファイル なし)
    SMOKE_TEST=False、並べ替えモデルの精度 FP16 の autocast + 動的損失スケーリング(スコア・損失・指標は FP32、索引は CPU の FP32)、決定的な演算の強制: False



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

from src.data.reranking import (
    article_passage_ranges,
    encode_articles_with_cache,
    mine_candidate_lists,
    sample_reranking_schedule,
    usable_training_queries,
)
from src.data.retrieval import (
    build_retrieval_set,
    find_articles_starting_at_or_after,
    select_evenly,
    split_articles,
)
from src.data.text import locate_wikipedia_article_spans, split_train_val_text
from src.data.tokenizer import load_bpe_id_tokenizer_from_hub
from src.layers.feedforward import SwiGLUFeedForwardNetwork
from src.layers.normalization import RMSNorm
from src.layers.positional_encoding import RotaryPositionEmbedding
from src.models.cross_encoder import CrossEncoder, build_cross_encoder_inputs
from src.models.gpt import GPTLanguageModel
from src.models.text_embedding import TextEmbeddingModel, encode_tokens
from src.retrieval.cost_curve import (
    cluster_bootstrap_weights,
    fit_log_log_slope,
    geometric_grid,
    interpolate_log_cost_batch,
    positions_in_subset,
    sweep_hnsw_search_width,
    sweep_inverted_file_probes,
    weighted_curve,
)
from src.retrieval.evaluation import evaluate_embeddings, expected_random_recall
from src.retrieval.exact_search import (
    CostCounter,
    exact_search,
    recall_against_ground_truth,
    remove_excluded_and_truncate,
)
from src.retrieval.hnsw import HierarchicalNavigableSmallWorld, draw_levels_and_insertion_order
from src.retrieval.inverted_file import InvertedFileIndex
from src.retrieval.kmeans import fit_kmeans, inertia
from src.retrieval.product_quantization import ProductQuantizationIndex, ProductQuantizer
from src.retrieval.reranking import RerankingResult, rank_reranked_candidates, score_candidates
from src.training.reranking import (
    build_reranker,
    compute_candidate_scores,
    compute_reranking_losses,
    softmax_cross_entropy_over_candidates,
    train_reranker,
)
from src.utils.reporting import dumps_compact_json
from src.utils.statistics import fit_power_law_exponent, paired_cluster_bootstrap_ratio_of_sums

ROOT = Path.cwd()
WIKIPEDIA_CACHE_DIR = ROOT / ".cache" / "wikipedia_en"  # locate_wikipedia_article_spans() の引数(記事の境界は metadata.json から得るので、Wikipedia API には接続しない)
ENCODED_CACHE_DIR = ROOT / ".cache" / "025_encoded"  # 符号化した記事のトークン列(入力から決定的に再生成できる。同一セッション内の高速化のみが目的)
EMBEDDING_CACHE_DIR = ROOT / ".cache" / "025_embeddings"  # 第 1 段の埋め込み(同上)
CORPUS_REPO_ID = "kojikojiprg/ai-theories-corpus-en"  # 9,826 記事。使うのは先頭の 356 記事を除く 9,470 記事
REFERENCE_CORPUS_REPO_ID = "kojikojiprg/ai-theories-corpus-en-pretraining"  # 008 の事前学習と英語のトークナイザの学習に使ったコーパス
TOKENIZER_REPO_ID = "kojikojiprg/ai-theories-tokenizer-en"
REFERENCE_MODEL_REPO_ID = "kojikojiprg/ai-theories-small-gpt-en"  # 008 の学習済みモデル(cross-encoder と dual encoder の起点)
FIRST_STAGE_REPO_ID = "kojikojiprg/ai-theories-text-embedding-en"  # 024 の学習済みモデル(第 1 段の埋め込み。凍結)
UPLOAD_REPO_ID = "kojikojiprg/ai-theories-cross-encoder-en"
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

**スモークテストの目的は、コードの経路が最後まで通る動作確認だけである。重い処理は走らせない。** 時間を決める値は、本番と同じ構造を保ったまま十分に縮小し、数値に意味を持たせない(前提条件は成立しない見込みで、判定の結果は意味を持たない)。
本番の規模での確認は、6.3 節のスケーリングの計測と、6.2 節のパイロット(並べ替えモデルの学習は最小の 3 回)で行った。

**縮小の規則**: スモークテストは、本番と **同じモデル・同じデータ・同じ条件・同じ実行計画の構造** で、次の量だけを縮小する。

- **学習ステップ数 $T$ の候補**: 本番 $(1024, 512, 256)$、スモークテスト $(16, 8, 4)$。どちらも大きい順で、隣どうしの比が 2 の等比という構造を保つ。
- **索引の最大の件数 $N_{\max}$ の候補**: 本番 $(65536, 32768, 16384)$、スモークテスト $(4096, 2048, 1024)$。どちらも大きい順で、隣どうしの比が 2 の等比という構造を保つ。
  $N$ の水準は $N_{\max}$ を最大とする公比 2 の 5 点($N_{\max} / 16, \dots, N_{\max}$)で、水準の数も保つ。**計測の水準**(6.3 節の $N$ の 3 点)も、本番 $(2048, 4096, 8192)$、スモークテスト $(256, 512, 1024)$ と、公比 2 の 3 点を保って縮小する。
- **シード数**: 本番は 5(削った段階で 3)、スモークテストは 3(削った段階で 2)。「削る前 > 削った後 $\ge 2$」の順序関係を保つ。
- **評価の query の数**(評価用の記事 1 本あたり): 本番は 近似最近傍探索 3・並べ替え 5・実験 F 2、スモークテストは 1・1・1。検証用の記事 1 本あたりは、本番 10、スモークテスト 1。
  「ランダムな候補の中での指標」の query 数の上限は本番 500、スモークテスト 100。並べ替えの query 数の上限は、スモークテストだけ 100、実験 F の query 数の上限は、スモークテストだけ 50。
- **ブートストラップの反復回数**: 本番 10,000 回、スモークテスト 1,000 回。

**縮小しないもの**: モデルの構成、データの分割とコーパスの構成、バッチ(1 ステップの query 数と負例の数)、query と passage の長さ、並べ替える候補の数 $K$、条件の対応表、HNSW の設定($M$・$efConstruction$・格子)、
転置ファイルのリスト数の倍率と格子、Product Quantization の設定、学習率の較正の格子と拡張の規則、学習率のスケジュールの形、実行計画の表の構造と優先順位、前提条件と判定の閾値、スケーリングの計測点の数。

**テスト専用の上書き**: 環境変数`AI_THEORIES_FORCE_PLAN`が設定されているときのみ、6.4 節で見積もりによる計画の選択の代わりにその計画を使う(スモークテストで下位の計画の経路を確かめるため)。
本番では受け付けず、このセルで停止する。コミットする出力は上書きなしの実行のものである。

**実効水準の照合**: このセルで印字した水準(学習ステップ数の候補・$N_{\max}$ の候補・計測の水準・シード数・計画の表)が、実際に使われた値と一致することを 6.14 節のアサーションで確かめる(`SMOKE_TEST`の配線漏れの検出)。


```python
# --- 全水準で共通の定数(本番実行前に宣言し、SMOKE_TEST で変えない) ---
# データ(5.3 節)
EMBEDDING_DIMENSION = 256
QUERY_LENGTH, PASSAGE_LENGTH = 32, 128  # L_q、L_p(固定長、パディングなし。024 と同じ)
MAX_PASSAGES_PER_ARTICLE = 12  # 記事ごとの索引の passage 数の上限(024 と同じ)
MAX_QUERIES_PER_ARTICLE = 12
NUM_VALIDATION_ARTICLES, NUM_EVALUATION_ARTICLES = 100, 300
SPLIT_SEED, EVALUATION_SET_SEED = 24_001, 24_002  # 024 と同じ分割と評価用の集合
SPLIT_DIGEST_024, EVALUATION_DIGEST_024 = "716cfe33d05a2615", "da3e16b96714c0a4"  # 024 の本番の出力(モデルカード)に記録された値
REFERENCE_RECALL_024 = 0.6477  # 024 の C5 のシード 0(アップロードした重み)の評価用の索引での Recall@10(全次元。モデルカード)
REFERENCE_VALIDATION_RATIO = 0.05  # 008 の事前学習と英語のトークナイザの学習は、参照コーパスの末尾 5% を使っていない
LIST_LIKE_TITLE_PREFIXES = ("List of", "Lists of", "Timeline of", "Outline of", "Index of", "Glossary of", "Bibliography of", "Comparison of", "Discography", "Filmography", "Deaths in", "Births in")
COLAB_RAM_GIB = 12.7  # Colab 無料枠の RAM(メモリ使用量の見込みの判定に使う)
RAM_FRACTION_LIMIT = 0.75  # ピークのメモリ使用量が RAM のこの割合以下であること
EMBED_BATCH = 512  # 埋め込みの計算の 1 回の順伝播の系列の数(結果には影響しない)
SEPARATOR_BYTE, TERMINAL_BYTE = 0x1F, 0x1E  # cross-encoder の区間の区切りと終端に使う単一バイトのトークン(制御文字。語彙を増やさない)
# 近似最近傍探索(実験 A・B・C)
NEIGHBORS = 10  # k(recall は上位 10 件の一致率)
TARGET_RECALL = 0.9  # 目標の recall
STOP_RECALL = 0.95  # 掃引を止める recall(格子点の recall の平均がこの値以上になったら止める。ブートストラップの再標本が目標を挟めるための余裕)
NUM_SIZE_LEVELS = 5  # N の水準の数(N_max / 16, ..., N_max。公比 2)
HNSW_MAX_CONNECTIONS, HNSW_CONSTRUCTION_WIDTH = 5, 100  # M、efConstruction(全水準で固定)
HNSW_GRID = geometric_grid(11, 2 ** (1 / 3), 14)  # 探索の幅 ef の格子(公比 2^(1/3)。最小は k + 1 = 11)
INVERTED_FILE_LIST_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0, 8.0)  # リスト数 = round(sqrt(N) x 倍率)(公比 2 の 5 点)
INVERTED_FILE_KMEANS_ITERATIONS = 20
INVERTED_FILE_GRID = geometric_grid(1, 2**0.5, 23)  # 探索するリストの数 p の格子(公比 sqrt(2))
PRODUCT_QUANTIZATION_SUBVECTORS, PRODUCT_QUANTIZATION_CODEWORDS, PRODUCT_QUANTIZATION_KMEANS_ITERATIONS = 32, 256, 15  # m、k*(符号 32 バイト、圧縮率 32 倍)
PRODUCT_QUANTIZATION_RESCORE_COUNTS = (10, 20, 50, 100, 200)  # 再採点する候補の数 R
HNSW_SEED_BASE, INVERTED_FILE_SEED_BASE, PRODUCT_QUANTIZATION_SEED_BASE = 25_100, 25_500, 25_700  # 乱数シード(シード s は + s)
ANN_QUERY_SEED = 25_900
# 並べ替え(実験 D・E・F)
BATCH_QUERIES, NUM_NEGATIVES = 8, 7  # B、n(1 ステップ = B 個の query x (正例 1 + 負例 n))
RERANK_CANDIDATES = 50  # K(並べ替える候補の数)
MINING_DEPTH = RERANK_CANDIDATES  # K_mine(困難な負例を採掘する第 1 段の上位の件数。K と同じ)
RERANK_KS = (10, 20, 50, 100)  # 実験 F で掃引する並べ替える候補の数 K
RANDOM_CANDIDATES = 50  # 診断量「ランダムな候補の中での指標」の候補の数(正例 1 + ランダムな負例 49)
TEMPERATURE = 0.05  # dual encoder の温度(024 と同じ。固定)
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.1
MIN_LEARNING_RATE_RATIO = 0.01
GRADIENT_CLIP_THRESHOLD = 1.0
HEAD_INIT_STD = 0.02  # cross-encoder のスカラーのヘッドの重みの初期化の標準偏差
INIT_LOSS_SCALE = 2.0**16  # 動的損失スケーリング
LOSS_SCALE_GROWTH_INTERVAL = 2000
SCORE_BATCH = 512  # 並べ替えの推論の 1 回の順伝播の系列の数(結果には影響しない)
# 条件の対応表(6.1 節)。kind はモデルの種類、negatives は負例の種類
CONDITIONS = {
    "D1": {"kind": "cross_encoder", "negatives": "hard"},
    "D2": {"kind": "dual_encoder", "negatives": "hard"},
    "E2": {"kind": "cross_encoder", "negatives": "random"},
}
RUN_ORDER = ("D1", "D2", "E2")  # 本番の学習の順
# 学習率の較正(6.1 節)。中心は、024 の C2(dual encoder、4.8e-4)を起点にした仮置きの値で、中心以外の学習率での値は測っていない(6.2.6 節)。
# 中心の誤りを吸収するため、格子は中心の {1/16, 1/4, 1, 4, 16} 倍の 5 点(公比 4)に広げる(6.1 節)
LEARNING_RATE_CENTER = {"D1": 2.4e-4, "D2": 2.4e-4, "E2": 2.4e-4}
LEARNING_RATE_GRID_MULTIPLIERS = (1 / 16, 1 / 4, 1.0, 4.0, 16.0)
LEARNING_RATE_GRID_MULTIPLIERS_BY_POINTS = {5: LEARNING_RATE_GRID_MULTIPLIERS, 3: (1 / 4, 1.0, 4.0)}  # 削る段階の最後で 5 点から 3 点にする
LEARNING_RATE_GRID_RATIO = 4.0
# 前提条件と判定(6.1 節)
# 学習の成立(6.1 節): 学習前のモデルとの差が、そのばらつき(標準誤差・標準偏差)の P1_SIGMA_MULTIPLIER 倍以上であること(統計的な基準)。
# (a)は、さらに最後の区間の訓練損失が一様な予測の損失 ln(1 + n) 未満であること
FINAL_LOSS_FRACTION = 0.05  # 「最後の区間」= 最後の 5% のステップ
P1_SIGMA_MULTIPLIER = 2.0
UNIFORM_LOSS = math.log(1 + NUM_NEGATIVES)  # 一様な予測の損失 ln(1 + n)
P2_ROOM = 0.05  # 改善の余地: 基準の条件の Recall@10 <= coverage - この値
P3_MIN_COVERAGE, P3_MIN_HEADROOM = 0.35, 0.10  # 第 1 段の上位 K 件に正例が入る割合の下限と、並べ替えの余地(coverage - 並べ替えなしの Recall@10)の下限
SIGMA_MULTIPLIER = 2.0  # 判定の閾値は対比量の標準偏差の 2 倍
# 乱数シード(学習のシード s: 学習の組 25000 + s、初期化 25200 + s)
SCHEDULE_SEED_BASE, INIT_SEED_BASE = 25_000, 25_200
CALIBRATION_SEED_INDEX = 90  # 学習率の較正専用のシード(実験のシード 0〜4 と共有しない)
TIMING_SEED_INDEX = 91  # スケーリングの計測専用のシード
BOOTSTRAP_SEED = 25_600
SESSION_BUDGET_SECONDS = 120 * 60  # 1 セッションの予算(T4 で 120 分)
UPLOAD_RUN = ("D1", 0)  # Hub にアップロードする学習(結果を見て選ばない)

# --- 水準(SMOKE_TEST で変わるもの) ---
LEVELS = {
    "smoke": {
        "STEP_CANDIDATES": (16, 8, 4),
        "N_MAX_CANDIDATES": (4096, 2048, 1024),
        "TIMING_SIZES": (256, 512, 1024),
        "SEEDS_FULL": 3,
        "SEEDS_CUT": 2,
        "ANN_QUERIES_PER_ARTICLE": 1,
        "RERANK_QUERIES_PER_ARTICLE": 1,
        "RERANK_QUERIES_PER_ARTICLE_F": 1,
        "VALIDATION_QUERIES_PER_ARTICLE": 1,
        "RANDOM_CANDIDATE_QUERIES": 100,
        "RERANK_MAX_QUERIES": 100,
        "F_MAX_QUERIES": 50,
        "BOOTSTRAP_RESAMPLES": 1_000,
    },
    "prod": {
        "STEP_CANDIDATES": (1024, 512, 256),
        "N_MAX_CANDIDATES": (65536, 32768, 16384),
        "TIMING_SIZES": (2048, 4096, 8192),
        "SEEDS_FULL": 5,
        "SEEDS_CUT": 3,
        "ANN_QUERIES_PER_ARTICLE": 3,
        "RERANK_QUERIES_PER_ARTICLE": 5,
        "RERANK_QUERIES_PER_ARTICLE_F": 2,
        "VALIDATION_QUERIES_PER_ARTICLE": 10,
        "RANDOM_CANDIDATE_QUERIES": 500,
        "RERANK_MAX_QUERIES": 100_000,
        "F_MAX_QUERIES": 100_000,
        "BOOTSTRAP_RESAMPLES": 10_000,
    },
}
CURRENT_LEVEL_NAME = "smoke" if SMOKE_TEST else "prod"
LEVEL_SETTINGS = LEVELS[CURRENT_LEVEL_NAME]
STEP_CANDIDATES = LEVEL_SETTINGS["STEP_CANDIDATES"]
N_MAX_CANDIDATES = LEVEL_SETTINGS["N_MAX_CANDIDATES"]
RANDOM_CANDIDATE_QUERIES = LEVEL_SETTINGS["RANDOM_CANDIDATE_QUERIES"]  # 診断量「ランダムな候補の中での指標」で評価する query の数の上限
TIMING_SIZES = LEVEL_SETTINGS["TIMING_SIZES"]  # HNSW・転置ファイル・Product Quantization のスケーリングの計測の水準(3 点)
BOOTSTRAP_RESAMPLES = LEVEL_SETTINGS["BOOTSTRAP_RESAMPLES"]

# 削る段階(6.1 節)。各段階は前の段階に 1 つだけ変更を加える(累積)。順序:
# 近似最近傍探索のシード数 -> N_max(1 段下げる)-> 実験 E の E2 のシード数 -> 実験 D の D1・D2 のシード数 -> N_max(もう 1 段下げる)-> 学習率の較正の格子(5 点から 3 点)。T を下げるのは最後
STAGE_CHANGES = (
    "(削らない)",
    "実験 A・B・C のシード数を削る",
    "N_max を 1 段下げる",
    "E2 のシード数を削る",
    "D1・D2 のシード数を削る",
    "N_max をもう 1 段下げる",
    "学習率の較正の格子を 5 点から 3 点にする",
)
NUM_STAGES = len(STAGE_CHANGES)


def stage_settings(stage: int, level: str = CURRENT_LEVEL_NAME) -> dict:
    # 段階 stage の実効の設定(累積)。シード数は SEEDS_FULL か SEEDS_CUT
    full, cut = LEVELS[level]["SEEDS_FULL"], LEVELS[level]["SEEDS_CUT"]
    return {
        "ann_seeds": cut if stage >= 1 else full,
        "n_max": LEVELS[level]["N_MAX_CANDIDATES"][(1 if stage >= 2 else 0) + (1 if stage >= 5 else 0)],
        "e2_seeds": cut if stage >= 3 else full,
        "d_seeds": cut if stage >= 4 else full,
        "grid_points": 3 if stage >= 6 else 5,
    }


def build_plans(step_candidates) -> list[dict]:
    # 実行計画(6.1 節): 学習ステップ数 T(大きい順)-> 削る段階(0 -> 6)
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
# T・N_max・シード数は 6.4 節で実行計画を選んだ後に決まる


def warmup_steps_for(num_steps: int) -> int:
    return max(1, round(WARMUP_RATIO * num_steps))


def final_loss_window(num_steps: int) -> int:
    return max(1, round(FINAL_LOSS_FRACTION * num_steps))


def learning_rate_grid_for(condition: str, grid_points: int = 5) -> tuple[float, ...]:
    # 較正の格子: 中心の {1/16, 1/4, 1, 4, 16} 倍(5 点)。削る段階の最後(段階 6)では {1/4, 1, 4} 倍(3 点)
    multipliers = LEARNING_RATE_GRID_MULTIPLIERS_BY_POINTS[grid_points]
    return tuple(LEARNING_RATE_CENTER[condition] * m for m in multipliers)


def best_eligible_index(points: list[float], eligible: list[bool], values: list[float]) -> int | None:
    # 較正の選択: P1(a) を満たす(eligible)学習率のうち、指標(平均逆順位)が最大の点の位置(同点なら小さい学習率)。候補がなければ None
    candidates = [i for i, ok in enumerate(eligible) if ok]
    if not candidates:
        return None
    return max(candidates, key=lambda i: (values[i], -points[i]))


def size_levels_for(n_max: int) -> list[int]:
    # N の水準: N_max を最大とする公比 2 の NUM_SIZE_LEVELS 点(昇順)
    return [n_max // 2 ** (NUM_SIZE_LEVELS - 1 - j) for j in range(NUM_SIZE_LEVELS)]


def inverted_file_list_counts_for(n: int) -> list[int]:
    return [max(2, round(math.sqrt(n) * m)) for m in INVERTED_FILE_LIST_MULTIPLIERS]


# --- 縮小規則と水準の構造の確認 ---
assert set(LEVELS["smoke"]) == set(LEVELS["prod"])
assert all(LEVELS["smoke"][k] != LEVELS["prod"][k] for k in ("STEP_CANDIDATES", "N_MAX_CANDIDATES", "TIMING_SIZES", "BOOTSTRAP_RESAMPLES"))
for _name, _settings in LEVELS.items():
    _steps, _sizes = _settings["STEP_CANDIDATES"], _settings["N_MAX_CANDIDATES"]
    assert len(_steps) == 3 and list(_steps) == sorted(_steps, reverse=True)
    assert all(b == round(_steps[0] / 2**j) for j, b in enumerate(_steps))  # 等比(公比 1/2)
    assert all(b == _sizes[0] // 2**j for j, b in enumerate(_sizes)) and len(_sizes) == 3  # N_max の候補も等比(公比 1/2)
    assert len(_settings["TIMING_SIZES"]) == 3 and all(b == 2 * a for a, b in itertools.pairwise(_settings["TIMING_SIZES"]))  # 計測の水準は公比 2 の 3 点
    assert _settings["SEEDS_FULL"] > _settings["SEEDS_CUT"] >= 2  # 「削る前 > 削った後 >= 2」
    for _t in _steps:
        assert warmup_steps_for(_t) < _t
assert LEVELS["smoke"]["BOOTSTRAP_RESAMPLES"] < LEVELS["prod"]["BOOTSTRAP_RESAMPLES"]
for _key in ("ANN_QUERIES_PER_ARTICLE", "RERANK_QUERIES_PER_ARTICLE", "RERANK_QUERIES_PER_ARTICLE_F", "VALIDATION_QUERIES_PER_ARTICLE"):
    assert LEVELS["smoke"][_key] <= LEVELS["prod"][_key]
for _level in LEVELS:  # 段階ごとの変更が累積し、各段階で 1 つだけ変わる。本番とスモークテストで、各段階で何が変わるかが同じ
    for _k in range(1, NUM_STAGES):
        _before, _after = stage_settings(_k - 1, _level), stage_settings(_k, _level)
        assert sum(_before[key] != _after[key] for key in _before) == 1 and all(_after[key] <= _before[key] for key in _before), (_level, _k)
for _k in range(1, NUM_STAGES):
    _changed = lambda level: {key for key in stage_settings(_k, level) if stage_settings(_k, level)[key] != stage_settings(_k - 1, level)[key]}  # noqa: E731
    assert _changed("smoke") == _changed("prod")
assert len(PLANS) == 3 * NUM_STAGES and [p["plan"] for p in PLANS] == list(range(len(PLANS)))
assert all(b == 2 * a for a, b in itertools.pairwise(size_levels_for(N_MAX_CANDIDATES[0]))) and len(size_levels_for(N_MAX_CANDIDATES[0])) == NUM_SIZE_LEVELS
assert HNSW_GRID[0] == NEIGHBORS + 1 and all(b > a for a, b in itertools.pairwise(HNSW_GRID))
assert all(math.isclose(b / a, 2 ** (1 / 3), rel_tol=0.1) for a, b in itertools.pairwise(HNSW_GRID[3:]))  # 丸めの範囲で等比
assert INVERTED_FILE_GRID[0] == 1 and all(b > a for a, b in itertools.pairwise(INVERTED_FILE_GRID))
assert set(LEARNING_RATE_GRID_MULTIPLIERS_BY_POINTS[3]) <= set(LEARNING_RATE_GRID_MULTIPLIERS_BY_POINTS[5]) and all(  # 3 点の格子は 5 点の格子の部分(公比 4 のまま)
    math.isclose(b / a, LEARNING_RATE_GRID_RATIO) for m in LEARNING_RATE_GRID_MULTIPLIERS_BY_POINTS.values() for a, b in itertools.pairwise(m)
)
for _level in LEVELS:  # 探索するリストの数の格子が、どのリスト数にも届く(最大のリスト数以上まである)。掃引ではリスト数以下の点だけを使う
    for _n_max in LEVELS[_level]["N_MAX_CANDIDATES"]:
        _list_counts = inverted_file_list_counts_for(_n_max)
        assert INVERTED_FILE_GRID[-1] >= max(_list_counts) and _list_counts == sorted(_list_counts) and len(set(_list_counts)) == len(INVERTED_FILE_LIST_MULTIPLIERS), (_level, _n_max, _list_counts)
assert all(math.isclose(b / a, 2.0) for a, b in itertools.pairwise(INVERTED_FILE_LIST_MULTIPLIERS))  # リスト数の倍率は公比 2
assert EMBEDDING_DIMENSION % PRODUCT_QUANTIZATION_SUBVECTORS == 0 and PRODUCT_QUANTIZATION_CODEWORDS <= 256
assert QUERY_LENGTH <= PASSAGE_LENGTH and QUERY_LENGTH + 1 + PASSAGE_LENGTH + 1 <= 256  # cross-encoder の列は 008 の学習長(256)以内
assert MINING_DEPTH == RERANK_CANDIDATES and RERANK_CANDIDATES in RERANK_KS and 10 in RERANK_KS

print(f"水準 {CURRENT_LEVEL_NAME!r}: {json.dumps({k: list(v) if isinstance(v, tuple) else v for k, v in LEVEL_SETTINGS.items()})}")
print(
    f"データ: query {QUERY_LENGTH} トークン、passage {PASSAGE_LENGTH} トークン(固定長、パディングなし)、記事ごとの passage 数の上限 {MAX_PASSAGES_PER_ARTICLE}、"
    f"検証用 {NUM_VALIDATION_ARTICLES} 記事・評価用 {NUM_EVALUATION_ARTICLES} 記事(024 と同じ分割)"
)
print(
    f"近似最近傍探索: k = {NEIGHBORS}、目標の recall {TARGET_RECALL}(掃引は recall の平均が {STOP_RECALL} 以上で止める)、N の水準 {NUM_SIZE_LEVELS} 点(N_max / 16, ..., N_max)。"
    f"HNSW: M = {HNSW_MAX_CONNECTIONS}、efConstruction = {HNSW_CONSTRUCTION_WIDTH}、探索の幅の格子 {HNSW_GRID}"
)
print(
    f"転置ファイル: リスト数 = round(sqrt(N) x {INVERTED_FILE_LIST_MULTIPLIERS})、k-means {INVERTED_FILE_KMEANS_ITERATIONS} 回、p の格子 {INVERTED_FILE_GRID}。"
    f"Product Quantization: m = {PRODUCT_QUANTIZATION_SUBVECTORS}、k* = {PRODUCT_QUANTIZATION_CODEWORDS}(符号 {PRODUCT_QUANTIZATION_SUBVECTORS} バイト、{EMBEDDING_DIMENSION * 4 // PRODUCT_QUANTIZATION_SUBVECTORS} 倍の圧縮)、k-means {PRODUCT_QUANTIZATION_KMEANS_ITERATIONS} 回、再採点 R {PRODUCT_QUANTIZATION_RESCORE_COUNTS}"
)
print(
    f"並べ替え: 1 ステップ {BATCH_QUERIES} query x (正例 1 + 負例 {NUM_NEGATIVES})、並べ替える候補 K = {RERANK_CANDIDATES}(採掘の深さ {MINING_DEPTH})、"
    f"温度 {TEMPERATURE}(dual encoder、固定)、AdamW(重み減衰 {WEIGHT_DECAY}、foreach)、warmup {WARMUP_RATIO:.0%} + cosine(下限 x{MIN_LEARNING_RATE_RATIO})、gradient clipping {GRADIENT_CLIP_THRESHOLD}"
)
print(f"条件: {dumps_compact_json(CONDITIONS)}")
for _t in STEP_CANDIDATES:
    print(f"  T = {_t}: warmup {warmup_steps_for(_t)}、最後の区間 {final_loss_window(_t)} ステップ")
print(f"N_max の候補: {N_MAX_CANDIDATES}。N の水準(N_max = {N_MAX_CANDIDATES[0]}): {size_levels_for(N_MAX_CANDIDATES[0])}")
print(f"削る段階({CURRENT_LEVEL_NAME!r}): " + "、".join(f"段階 {k}({STAGE_CHANGES[k]}): {dumps_compact_json(stage_settings(k))}" for k in range(NUM_STAGES)))
print(f"実行計画: {len(PLANS)} 通り(番号の小さいほど優先)。計画番号 = {NUM_STAGES} x(T の番号)+ 段階。T の候補 {STEP_CANDIDATES}、段階 0〜{NUM_STAGES - 1}")
if FORCED_PLAN is not None:
    print(f"*** テスト専用の上書き: AI_THEORIES_FORCE_PLAN = {FORCED_PLAN}(6.4 節で見積もりの代わりにこの計画を使う) ***")
```

    水準 'prod': {"STEP_CANDIDATES": [1024, 512, 256], "N_MAX_CANDIDATES": [65536, 32768, 16384], "TIMING_SIZES": [2048, 4096, 8192], "SEEDS_FULL": 5, "SEEDS_CUT": 3, "ANN_QUERIES_PER_ARTICLE": 3, "RERANK_QUERIES_PER_ARTICLE": 5, "RERANK_QUERIES_PER_ARTICLE_F": 2, "VALIDATION_QUERIES_PER_ARTICLE": 10, "RANDOM_CANDIDATE_QUERIES": 500, "RERANK_MAX_QUERIES": 100000, "F_MAX_QUERIES": 100000, "BOOTSTRAP_RESAMPLES": 10000}
    データ: query 32 トークン、passage 128 トークン(固定長、パディングなし)、記事ごとの passage 数の上限 12、検証用 100 記事・評価用 300 記事(024 と同じ分割)
    近似最近傍探索: k = 10、目標の recall 0.9(掃引は recall の平均が 0.95 以上で止める)、N の水準 5 点(N_max / 16, ..., N_max)。HNSW: M = 5、efConstruction = 100、探索の幅の格子 [11, 14, 17, 22, 28, 35, 44, 55, 70, 88, 111, 140, 176, 222]
    転置ファイル: リスト数 = round(sqrt(N) x (0.5, 1.0, 2.0, 4.0, 8.0))、k-means 20 回、p の格子 [1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512, 724, 1024, 1448, 2048]。Product Quantization: m = 32、k* = 256(符号 32 バイト、32 倍の圧縮)、k-means 15 回、再採点 R (10, 20, 50, 100, 200)
    並べ替え: 1 ステップ 8 query x (正例 1 + 負例 7)、並べ替える候補 K = 50(採掘の深さ 50)、温度 0.05(dual encoder、固定)、AdamW(重み減衰 0.1、foreach)、warmup 10% + cosine(下限 x0.01)、gradient clipping 1.0
    条件: {
      "D1": {"kind": "cross_encoder", "negatives": "hard"},
      "D2": {"kind": "dual_encoder", "negatives": "hard"},
      "E2": {"kind": "cross_encoder", "negatives": "random"}
    }
      T = 1024: warmup 102、最後の区間 51 ステップ
      T = 512: warmup 51、最後の区間 26 ステップ
      T = 256: warmup 26、最後の区間 13 ステップ
    N_max の候補: (65536, 32768, 16384)。N の水準(N_max = 65536): [4096, 8192, 16384, 32768, 65536]
    削る段階('prod'): 段階 0((削らない)): {"ann_seeds": 5, "n_max": 65536, "e2_seeds": 5, "d_seeds": 5, "grid_points": 5}、段階 1(実験 A・B・C のシード数を削る): {"ann_seeds": 3, "n_max": 65536, "e2_seeds": 5, "d_seeds": 5, "grid_points": 5}、段階 2(N_max を 1 段下げる): {"ann_seeds": 3, "n_max": 32768, "e2_seeds": 5, "d_seeds": 5, "grid_points": 5}、段階 3(E2 のシード数を削る): {"ann_seeds": 3, "n_max": 32768, "e2_seeds": 3, "d_seeds": 5, "grid_points": 5}、段階 4(D1・D2 のシード数を削る): {"ann_seeds": 3, "n_max": 32768, "e2_seeds": 3, "d_seeds": 3, "grid_points": 5}、段階 5(N_max をもう 1 段下げる): {"ann_seeds": 3, "n_max": 16384, "e2_seeds": 3, "d_seeds": 3, "grid_points": 5}、段階 6(学習率の較正の格子を 5 点から 3 点にする): {"ann_seeds": 3, "n_max": 16384, "e2_seeds": 3, "d_seeds": 3, "grid_points": 3}
    実行計画: 21 通り(番号の小さいほど優先)。計画番号 = 7 x(T の番号)+ 段階。T の候補 (1024, 512, 256)、段階 0〜6




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/025_ann_search_and_reranking.ipynb
