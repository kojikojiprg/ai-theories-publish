---
title: "デコーディング戦略(Decoding Strategies)(実装・実験編 1/2)"
---

この記事は後編(実装・実験編 1/2)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/008_decoding_strategies-theory)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/008_decoding_strategies-practice-2)。

## 4. 実装方針 / Implementation Policy

- top-k フィルタリング(`top_k_filter`)・top-p フィルタリング(`top_p_filter`)・
  ビームサーチ(`beam_search`)は `src/generation/decoding.py` にスクラッチ実装する
  (`torch.topk`・`torch.sort` などのテンソル演算プリミティブは使うが、フィルタリング・
  探索のロジック自体は既存ライブラリの高水準 API に委譲しない)。
- 生成品質の指標(n-gram 重複率・distinct-n)は `src/utils/statistics.py` に
  `compute_ngram_repetition_rate`・`compute_distinct_n` として追加する。
- モデル本体(`GPTLanguageModel`)・学習ループ(`train_language_model`)・
  optimizer(`AdamW`)・学習率スケジュール(`compute_warmup_cosine_learning_rate`)は
  006・007 の実装をそのまま再利用し、本トピックでは変更しない。
- モデルアーキテクチャ・トークナイザ条件は 006 の本番実行結果(実験 D・E・F・G)で
  確定した値(英語・バイトレベル BPE・語彙サイズ 8192・`D_MODEL=256` など)をそのまま
  採用する(2.1 節参照、改めて選定し直さない)。
- 学習は 007 の安定化技術(AdamW・warmup + cosine・gradient clipping)を適用するが、
  gradient clipping 閾値・学習率は 008 自身のモデル・データに対して本番スケールで
  改めて較正する(007 の較正結果を流用しない、CLAUDE.md「較正は本番と同じスケールで
  行う」規約)。
- Hugging Face Hub へのアップロードは関数として実装するが、Claude Code(第 1 段階、
  ローカル実行)では呼び出さない。認証情報(トークン)はコード中に一切書き込まず、
  Google Colab の Secrets(`google.colab.userdata.get("HF_TOKEN")`)から読み出す設計とする。


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
    remote: Enumerating objects: 471, done.[K
    remote: Counting objects: 100% (471/471), done.[K
    remote: Compressing objects: 100% (310/310), done.[K
    remote: Total 471 (delta 239), reused 346 (delta 142), pack-reused 0 (from 0)[K
    Receiving objects: 100% (471/471), 5.90 MiB | 359.00 KiB/s, done.
    Resolving deltas: 100% (239/239), done.
    /content/ai-theories
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m23.7/23.7 MB[0m [31m72.1 MB/s[0m eta [36m0:00:00[0m
    [?25h[2mUsing Python 3.12.13 environment at: /usr[0m
    [2K[2mResolved [1m52 packages[0m [2min 352ms[0m[0m
    [2K[2mPrepared [1m28 packages[0m [2min 1m 02s[0m[0m
    [2mUninstalled [1m14 packages[0m [2min 974ms[0m[0m
    [2K[2mInstalled [1m28 packages[0m [2min 502ms[0m[0m
     [31m-[39m [1mcuda-bindings[0m[2m==12.9.7[0m
     [32m+[39m [1mcuda-bindings[0m[2m==13.3.1[0m
     [31m-[39m [1mcuda-toolkit[0m[2m==12.8.1[0m
     [32m+[39m [1mcuda-toolkit[0m[2m==13.0.3.0[0m
     [31m-[39m [1mfilelock[0m[2m==3.32.3[0m
     [32m+[39m [1mfilelock[0m[2m==3.32.2[0m
     [31m-[39m [1mfsspec[0m[2m==2025.3.0[0m
     [32m+[39m [1mfsspec[0m[2m==2026.7.0[0m
     [31m-[39m [1mhuggingface-hub[0m[2m==1.27.0[0m
     [32m+[39m [1mhuggingface-hub[0m[2m==1.28.0[0m
     [31m-[39m [1midna[0m[2m==3.18[0m
     [32m+[39m [1midna[0m[2m==3.19[0m
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
     [31m-[39m [1mtqdm[0m[2m==4.67.3[0m
     [32m+[39m [1mtqdm[0m[2m==4.70.0[0m
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
    encode_corpus,
    load_wikipedia_corpus,
    make_evaluation_windows,
    split_train_val_text,
)
from src.data.tokenizer import BPETokenizer, learn_bpe
from src.generation.decoding import beam_search, top_k_filter, top_p_filter
from src.layers.feedforward import SwiGLUFeedForwardNetwork
from src.layers.normalization import RMSNorm
from src.layers.positional_encoding import RotaryPositionEmbedding
from src.models.gpt import GPTLanguageModel
from src.training.optimizer import AdamW
from src.training.schedule import compute_warmup_cosine_learning_rate
from src.training.trainer import train_language_model
from src.utils.statistics import (
    compute_distinct_n,
    compute_ngram_repetition_rate,
    count_non_embedding_parameters,
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"torch: {torch.__version__} / device: {device}")

ROOT = Path(".")
# 006 が取得済みの英語 Wikipedia コーパス(同一のマニフェスト en_006_pretraining.json)を
# 共有する。同一言語・同一記事集合であるため、008 専用に再取得する必要はない。
CACHE_DIR = ROOT / ".cache" / "006_corpus"
```

    torch: 2.13.0+cu130 / device: cuda


### 5.1 スケールの設定(スモークテスト)

`SMOKE_TEST=True` はローカル(Claude Code、第 1 段階)での動作確認・本番実行可能性の
検証用の縮小スケールであり、`SMOKE_TEST=False`(Google Colab T4、第 2 段階)が本番
設定である。model サイズ・トークナイザ語彙サイズは 006 の本番実行結果(2.1 節)を
そのまま使うため、`SMOKE_TEST` では変更しない。スケールする(縮小する)のはコーパス量・
学習ステップ数のみであり、本番設定との比率構造(等比刻みの語彙サイズは 008 では
単一条件のため対象外)は保つ。



```python
SMOKE_TEST = False  # Claude Code はこの True 側のみ実行する(Colab T4 では False に切り替える)

# --- トークナイザ・model(006 の本番実行結果、2.1 節で確定済み。SMOKE_TEST で変更しない) ---
VOCAB_SIZE = 8192
D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF = 256, 4, 8, 1024
SEQUENCE_LENGTH = 256
DROPOUT = 0.0
TIE_EMBEDDINGS = True
SWIGLU_D_FF = round((2 / 3) * D_FF)  # 004: 標準の順伝播ネットワークとパラメータ数を揃えるための丸め
MAX_CHUNK_BYTES = 64  # 006 3.6 節・5.1 節と同一(事前分割チャンクの最大バイト数)

# --- 学習(2.2〜2.3 節) ---
BATCH_SIZE = 32  # 006 踏襲
TOKENS_PER_STEP = BATCH_SIZE * SEQUENCE_LENGTH
# 006 実測の英語・bpe_v8192 訓練トークン数(2.3 節の出典): 5,956,436 -> 1 エポック ~= 727 ステップ
EN_BPE_V8192_TRAIN_TOKENS_006 = 5_956_436
STEPS_PER_EPOCH_006 = EN_BPE_V8192_TRAIN_TOKENS_006 / TOKENS_PER_STEP
PRODUCTION_EPOCHS = 3.0
PRODUCTION_NUM_STEPS = round(
    PRODUCTION_EPOCHS * STEPS_PER_EPOCH_006
)  # 推薦値 ~= 2181 ステップ(2.3 節)

BASE_LEARNING_RATE = 3e-4  # 006 の本番学習率(006 5.1 節、007 6.1 節と同じ出発点)
# 較正(6.1 節)で使う倍率(等比刻み、007 3.2 節と同じ考え方)。
LR_CALIBRATION_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0)
CLIP_QUANTILE = (
    0.90  # gradient clipping 閾値 = 較正実行の勾配ノルムの CLIP_QUANTILE 分位点(007 3.4 節)
)
WARMUP_RATIO = 0.1  # 007 と同じ
MIN_LEARNING_RATE_RATIO = 0.01  # 007 と同じ
WEIGHT_DECAY = 0.1  # 007 と同じ
SESSION_BUDGET_SECONDS = 2 * 60 * 60  # Google Colab 1 セッションの目安予算(006・007 と同じ基準)

# 前提条件 P1(学習の進行、6 節)の採用閾値。最終検証 bits-per-byte がこの絶対値以下で
# あることを要求する(検証したい仮説 — どのデコーディング戦略が繰り返しを起こしやすいか
# — とは独立な、モデルが実際に流暢なテキストを生成できる水準まで学習できているかの条件)。
# 値の根拠: 006 実験 G(bpe_v8192, en, 636 ステップ)の本番実行結果で検証 bits-per-byte は
# 2.19 に達している(006 7 節)。008 は同一条件で 2181 ステップ(636 の約 3.4 倍)学習するため
# さらに低い値が期待されるが、余裕を持って 2.19 よりやや高い 2.5 を閾値とする。
PRECONDITION_BPB_THRESHOLD = 2.5

if SMOKE_TEST:
    # --- コーパス ---
    LM_CORPUS_BYTES = 400_000  # 本番: 制限なし(取得済みコーパス全体)
    TOKENIZER_TRAIN_BYTES = 40_000  # 本番: 8,000,000(006 と同一)
    VALIDATION_RATIO = 0.05

    # --- 学習ステップ数: 本番との比率構造(3 エポック相当)を保って縮小する ---
    SMOKE_STEPS_PER_EPOCH = max(
        1, round(LM_CORPUS_BYTES * (1 - VALIDATION_RATIO) / 4 / SEQUENCE_LENGTH / BATCH_SIZE)
    )
    NUM_STEPS = max(10, round(PRODUCTION_EPOCHS * SMOKE_STEPS_PER_EPOCH))
    EVAL_INTERVAL = max(1, NUM_STEPS // 4)
else:
    # --- コーパス ---
    LM_CORPUS_BYTES = 10**12  # 実質無制限(取得済みコーパス全体を使う)
    TOKENIZER_TRAIN_BYTES = 8_000_000  # 006 と同一
    VALIDATION_RATIO = 0.05

    NUM_STEPS = PRODUCTION_NUM_STEPS
    EVAL_INTERVAL = 200

WARMUP_STEPS = max(1, round(WARMUP_RATIO * NUM_STEPS))
CALIBRATION_STEPS = NUM_STEPS  # 較正は本番と同じスケールで行う(007 の反省を踏まえた規約)

print(f"SMOKE_TEST={SMOKE_TEST}")
print(
    f"VOCAB_SIZE={VOCAB_SIZE}, D_MODEL={D_MODEL}, NUM_LAYERS={NUM_LAYERS}, NUM_HEADS={NUM_HEADS}, "
    f"D_FF={D_FF}(SwiGLU: {SWIGLU_D_FF})"
)
print(
    f"SEQUENCE_LENGTH={SEQUENCE_LENGTH}, BATCH_SIZE={BATCH_SIZE}, TOKENS_PER_STEP={TOKENS_PER_STEP}"
)
print(f"PRODUCTION_NUM_STEPS(3 epochs, 006実測より逆算)={PRODUCTION_NUM_STEPS}")
print(
    f"NUM_STEPS={NUM_STEPS}, CALIBRATION_STEPS={CALIBRATION_STEPS}, WARMUP_STEPS={WARMUP_STEPS}, "
    f"EVAL_INTERVAL={EVAL_INTERVAL}"
)
print(f"LM_CORPUS_BYTES={LM_CORPUS_BYTES}, TOKENIZER_TRAIN_BYTES={TOKENIZER_TRAIN_BYTES}")
print(f"PRECONDITION_BPB_THRESHOLD={PRECONDITION_BPB_THRESHOLD}")
```

    SMOKE_TEST=False
    VOCAB_SIZE=8192, D_MODEL=256, NUM_LAYERS=4, NUM_HEADS=8, D_FF=1024(SwiGLU: 683)
    SEQUENCE_LENGTH=256, BATCH_SIZE=32, TOKENS_PER_STEP=8192
    PRODUCTION_NUM_STEPS(3 epochs, 006実測より逆算)=2181
    NUM_STEPS=2181, CALIBRATION_STEPS=2181, WARMUP_STEPS=218, EVAL_INTERVAL=200
    LM_CORPUS_BYTES=1000000000000, TOKENIZER_TRAIN_BYTES=8000000
    PRECONDITION_BPB_THRESHOLD=2.5


### 5.2 コーパスの取得

006 と同じ `load_wikipedia_corpus` で英語 Wikipedia コーパス(`en_006_pretraining.json`、
20 MB 以上)を取得する。取得結果は `CACHE_DIR`(ランタイムのローカルファイルシステム)に
記事単位でキャッシュされ、同一セッション内でのセル再実行では再ダウンロードを省略する。
ただし Google Drive とは連携していないため、**ランタイムをリセットした場合(GitHub
経由でノートブックを開き直すたびに発生する)は、このキャッシュは失われ、実際に
再ダウンロードが走る**(固定の記事タイトル・リビジョン ID で取得するため、取得内容
自体は毎回同一で再現性は保たれる)。008 は英語のみを対象とするため、006 のように
複数言語・複数トークナイザ条件を並べる必要はない。



```python
raw_text_full = load_wikipedia_corpus("en", CACHE_DIR)
raw_text = (
    raw_text_full[:LM_CORPUS_BYTES] if len(raw_text_full) > LM_CORPUS_BYTES else raw_text_full
)
print(f"取得したコーパス全体: {len(raw_text_full):,} 文字 / 使用する範囲: {len(raw_text):,} 文字")

train_text, val_text = split_train_val_text(raw_text, VALIDATION_RATIO)
total_eval_bytes = len(val_text.encode("utf-8"))
print(
    f"train_text: {len(train_text):,} 文字, val_text: {len(val_text):,} 文字 "
    f"({total_eval_bytes:,} UTF-8 バイト)"
)
```

    取得したコーパス全体: 24,214,546 文字 / 使用する範囲: 24,214,546 文字
    train_text: 23,003,819 文字, val_text: 1,210,727 文字 (1,214,117 UTF-8 バイト)


### 5.3 トークナイザを整数 ID 方式に適合させるラッパー

006 5.3 節と同じ理由で、`BPETokenizer.encode()` が返す部分語シンボル列(`list[str]`)を
整数 ID に変換するラッパーが必要である。



```python
class BPEIDTokenizer:
    """BPETokenizer をラップし、encode() が整数 ID の列を返すようにする(006 5.3 節と同一の設計)。"""

    def __init__(self, bpe_tokenizer: BPETokenizer):
        self.bpe_tokenizer = bpe_tokenizer
        symbols = sorted(bpe_tokenizer.vocab)
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.id_to_symbol = dict(enumerate(symbols))
        self.vocab_size = len(symbols)

    def encode(self, text: str) -> list[int]:
        return [self.symbol_to_id[s] for s in self.bpe_tokenizer.encode(text)]

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.bpe_tokenizer.decode([self.id_to_symbol[i] for i in ids])
```

### 5.4 BPE 学習時間のスケーリング計測

CLAUDE.md「スケーリングの計測と外挿」節に従い、BPE 学習時間のデータ量依存性を
3 点以上のデータ量で実測し、$\log t = \log a + b \log n$ のあてはめでべき指数 $b$ を
推定して本番データ量(`TOKENIZER_TRAIN_BYTES`)への外挿値を出す(006 5.4 節の
`measure_bpe_scaling` と同じ手法)。



```python
def measure_bpe_scaling(text: str, sizes_bytes: list[int]) -> tuple[list[int], list[float]]:
    """複数のコーパスサイズで learn_bpe の実行時間を計測する(006 5.4 節と同一の手法)。"""
    text_bytes = text.encode("utf-8")
    actual_sizes, times = [], []
    for size in sizes_bytes:
        prefix_bytes = text_bytes[:size]
        prefix_text = prefix_bytes.decode("utf-8", errors="ignore")
        actual_sizes.append(len(prefix_text.encode("utf-8")))
        t0 = time.time()
        learn_bpe(prefix_text, VOCAB_SIZE, byte_level=True, max_chunk_bytes=MAX_CHUNK_BYTES)
        times.append(time.time() - t0)
    return actual_sizes, times


_bpe_scaling_sizes = [
    size for size in (10_000, 20_000, 40_000) if size <= len(train_text.encode("utf-8"))
]
if len(_bpe_scaling_sizes) < 3:
    _bpe_scaling_sizes = [
        len(train_text.encode("utf-8")) // 4,
        len(train_text.encode("utf-8")) // 2,
        len(train_text.encode("utf-8")),
    ]

_bpe_sizes, _bpe_times = measure_bpe_scaling(train_text, _bpe_scaling_sizes)
for s, t in zip(_bpe_sizes, _bpe_times, strict=True):
    print(f"BPE 学習: {s:,} bytes -> {t:.3f} s")

_log_n = np.log(_bpe_sizes)
_log_t = np.log(_bpe_times)
_bpe_exponent, _bpe_log_a = np.polyfit(_log_n, _log_t, deg=1)
_bpe_a = float(np.exp(_bpe_log_a))
_bpe_extrapolated = _bpe_a * TOKENIZER_TRAIN_BYTES**_bpe_exponent
print(f"\nべき指数 b={_bpe_exponent:.3f}(線形なら b~=1)")
print(
    f"本番データ量({TOKENIZER_TRAIN_BYTES:,} bytes)への外挿: {_bpe_extrapolated:.2f} s"
    f"(= {_bpe_extrapolated / 60:.2f} 分、条件数 x シード数 = 1 回のみ実行)"
)
```

    BPE 学習: 10,000 bytes -> 0.613 s
    BPE 学習: 20,000 bytes -> 1.303 s
    BPE 学習: 40,000 bytes -> 2.718 s
    
    べき指数 b=1.074(線形なら b~=1)
    本番データ量(8,000,000 bytes)への外挿: 807.58 s(= 13.46 分、条件数 x シード数 = 1 回のみ実行)


### 5.5 符号化時間のスケーリング計測

学習データ全体(`LM_CORPUS_BYTES`)の符号化(encode)は 1 回だけ行われる処理だが、
超線形にコストが増加すると本番実行時間の見積もりを大きく崩すため、006 5.5 節と同じ
手法でスケーリングを計測する。



```python
def measure_encode_scaling(
    tokenizer, text: str, sizes_chars: list[int]
) -> tuple[list[int], list[float]]:
    """複数のテキスト長で BPETokenizer.encode の実行時間を計測する(006 5.5 節と同一の手法)。"""
    sizes, times = [], []
    for size in sizes_chars:
        prefix = text[:size]
        sizes.append(len(prefix))
        t0 = time.time()
        tokenizer.encode(prefix)
        times.append(time.time() - t0)
    return sizes, times


# 較正には、本番と同じ語彙サイズ・条件で学習した小規模トークナイザを使う
# (符号化コストは学習済みマージ規則の数にも依存するため)。
_calibration_tokenizer = learn_bpe(
    train_text[: min(len(train_text), 200_000)],
    VOCAB_SIZE,
    byte_level=True,
    max_chunk_bytes=MAX_CHUNK_BYTES,
)
_encode_scaling_sizes = [size for size in (20_000, 50_000, 100_000) if size <= len(train_text)]
if len(_encode_scaling_sizes) < 3:
    _encode_scaling_sizes = [len(train_text) // 4, len(train_text) // 2, len(train_text)]

_enc_sizes, _enc_times = measure_encode_scaling(
    _calibration_tokenizer, train_text, _encode_scaling_sizes
)
for s, t in zip(_enc_sizes, _enc_times, strict=True):
    print(f"符号化: {s:,} chars -> {t:.4f} s")

_log_n = np.log(_enc_sizes)
_log_t = np.log(_enc_times)
_enc_exponent, _enc_log_a = np.polyfit(_log_n, _log_t, deg=1)
_enc_a = float(np.exp(_enc_log_a))
_enc_extrapolated = _enc_a * len(raw_text) ** _enc_exponent
print(f"\nべき指数 b={_enc_exponent:.3f}(線形なら b~=1)")
print(
    f"本番データ量({len(raw_text):,} 文字、学習用+検証用の符号化 x2 回)への外挿: "
    f"{2 * _enc_extrapolated:.2f} s"
)
del _calibration_tokenizer
```

    符号化: 20,000 chars -> 0.0260 s
    符号化: 50,000 chars -> 0.0467 s
    符号化: 100,000 chars -> 0.0696 s
    
    べき指数 b=0.612(線形なら b~=1)
    本番データ量(24,214,546 文字、学習用+検証用の符号化 x2 回)への外挿: 4.04 s


### 5.6 本番語彙での BPE 学習・ラウンドトリップ検証

`VOCAB_SIZE=8192` で BPE を学習し、整数 ID ラッパーで包む。ラウンドトリップ
(`decode(encode(text)) == text`)が英語コーパスで成立することを確認する(4 節、
005・006 と同様の検証)。



```python
t0 = time.time()
bpe_tokenizer = learn_bpe(
    train_text[:TOKENIZER_TRAIN_BYTES], VOCAB_SIZE, byte_level=True, max_chunk_bytes=MAX_CHUNK_BYTES
)
tokenizer = BPEIDTokenizer(bpe_tokenizer)
print(f"BPE 学習(実測、語彙サイズ={tokenizer.vocab_size}): {time.time() - t0:.2f} s")

_roundtrip_sample = val_text[: min(len(val_text), 50_000)]
_roundtrip_ok = bpe_tokenizer.decode(bpe_tokenizer.encode(_roundtrip_sample)) == _roundtrip_sample
assert _roundtrip_ok, "トークナイザのラウンドトリップが一致しない"
print(f"OK: ラウンドトリップ一致(検証テキスト先頭 {len(_roundtrip_sample):,} 文字)")
```

    BPE 学習(実測、語彙サイズ=8192): 154.68 s
    OK: ラウンドトリップ一致(検証テキスト先頭 50,000 文字)


### 5.7 コーパスの符号化・評価窓の作成



```python
train_ids = encode_corpus(tokenizer, train_text)
val_ids = encode_corpus(tokenizer, val_text)
eval_windows, eval_mask = make_evaluation_windows(val_ids, SEQUENCE_LENGTH)
print(
    f"train_ids: {len(train_ids):,} トークン, val_ids: {len(val_ids):,} トークン, "
    f"eval_windows: {tuple(eval_windows.shape)}"
)
```

    train_ids: 5,956,594 トークン, val_ids: 317,201 トークン, eval_windows: (1240, 256)


### 5.8 model の構築ヘルパー

2.1 節で確定した構成(RoPE・RMSNorm・SwiGLU・正規化前置・重み共有)で `GPTLanguageModel`
を構築する。



```python
def build_model(seed: int) -> GPTLanguageModel:
    torch.manual_seed(seed)
    d_k = D_MODEL // NUM_HEADS
    rope = RotaryPositionEmbedding(d_k, max_position=SEQUENCE_LENGTH)
    return GPTLanguageModel(
        vocabulary_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        max_sequence_length=SEQUENCE_LENGTH,
        positional_transform=rope,
        normalization_factory=RMSNorm,
        feed_forward_factory=functools.partial(SwiGLUFeedForwardNetwork, D_MODEL, SWIGLU_D_FF),
        tie_embeddings=TIE_EMBEDDINGS,
        dropout=DROPOUT,
    )


_probe_model = build_model(seed=0)
print(f"非埋め込みパラメータ数: {count_non_embedding_parameters(_probe_model):,}")
del _probe_model
```

    非埋め込みパラメータ数: 3,149,056


### 5.9 較正セル: 学習率・gradient clipping 閾値・1 ステップあたりの実行時間

007 6.1 節と同じ手順(較正は本番と同じ `CALIBRATION_STEPS = NUM_STEPS` で行う)を踏む。
008 は正規化方式・学習率を独立変数とした不安定性誘発が目的ではなく、AdamW +
warmup + cosine + gradient clipping を **常に全部乗せで使う**(007 の結論、主張 5:
全部乗せは収束を遅らせない)ため、較正するのは (1) 発散しない学習率、(2) その学習率での
gradient clipping 閾値、の 2 つのみである。

**学習率の選定方針**: `LR_CALIBRATION_MULTIPLIERS`(006 基準学習率に対する倍率、等比刻み)
のうち、`CALIBRATION_STEPS` 実行後に NaN/Inf に発散せず、かつ検証 bits-per-byte が
最小のものを採用する(008 は不安定性の誘発ではなく通常の学習が目的であるため、007と異なり
「最も不安定な候補」ではなく「最も良く学習できた候補」を選ぶ)。

**gradient clipping 閾値**: 採用した学習率での較正実行における勾配ノルムの
`CLIP_QUANTILE` 分位点(007 3.4 節・6.1 節と同じ分位点方式)。

**1 ステップあたりの実行時間**: 007 6.1 節(3)と同様、ステップ数を変えた計測により
べき指数を推定し(モデル学習が線形時間であることを確認したうえで)、本番ステップ数
(`PRODUCTION_NUM_STEPS`)への外挿値を出す。



```python
def _run_calibration(peak_lr: float, num_steps: int, seed: int = 0) -> dict:
    model = build_model(seed=seed).to(device)
    optimizer = AdamW(model.parameters(), lr=peak_lr, weight_decay=WEIGHT_DECAY)
    schedule = functools.partial(
        compute_warmup_cosine_learning_rate,
        warmup_steps=min(WARMUP_STEPS, max(1, num_steps // 10)),
        total_steps=num_steps,
        peak_learning_rate=peak_lr,
        min_learning_rate=peak_lr * MIN_LEARNING_RATE_RATIO,
    )
    history = train_language_model(
        model,
        train_ids,
        eval_windows,
        eval_mask,
        total_eval_bytes,
        num_steps=num_steps,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        learning_rate=peak_lr,
        eval_interval=max(1, num_steps // 2),
        device=device,
        seed=seed,
        optimizer=optimizer,
        learning_rate_schedule=schedule,
        gradient_clip_threshold=None,
    )
    return history


calibration_results = {}
for _mult in LR_CALIBRATION_MULTIPLIERS:
    _lr = BASE_LEARNING_RATE * _mult
    _t0 = time.time()
    _hist = _run_calibration(_lr, CALIBRATION_STEPS)
    _elapsed = time.time() - _t0
    _norms = np.array(_hist["gradient_norm"])
    _diverged = bool(
        np.isnan(_norms).any() or np.isinf(_norms).any() or np.isnan(_hist["train_loss"]).any()
    )
    _final_bpb = float("nan") if _diverged else _hist["eval_bits_per_byte"][-1]
    calibration_results[_mult] = {
        "lr": _lr,
        "elapsed_seconds": _elapsed,
        "diverged": _diverged,
        "final_bpb": _final_bpb,
        "grad_norm_quantile": float(np.quantile(_norms, CLIP_QUANTILE))
        if not _diverged
        else float("nan"),
    }
    print(
        f"lr={_lr:.2e}(x{_mult:.2f}): diverged={_diverged}, final_bpb={_final_bpb}, "
        f"elapsed={_elapsed:.2f}s"
    )

_survivors = {m: r for m, r in calibration_results.items() if not r["diverged"]}
assert _survivors, "全ての学習率候補が発散した。LR_CALIBRATION_MULTIPLIERS を下げて再較正が必要"
CHOSEN_LR_MULTIPLIER = min(_survivors, key=lambda m: _survivors[m]["final_bpb"])
PEAK_LEARNING_RATE = BASE_LEARNING_RATE * CHOSEN_LR_MULTIPLIER
GRADIENT_CLIP_THRESHOLD = calibration_results[CHOSEN_LR_MULTIPLIER]["grad_norm_quantile"]
print(f"\n採用した学習率: {PEAK_LEARNING_RATE:.3e}(006 基準の x{CHOSEN_LR_MULTIPLIER:.2f})")
print(
    f"gradient clipping 閾値: {GRADIENT_CLIP_THRESHOLD:.4f}"
    f"(勾配ノルムの {CLIP_QUANTILE:.0%} 分位点)"
)
```

    lr=1.50e-04(x0.50): diverged=False, final_bpb=2.2082103525643877, elapsed=343.23s
    lr=3.00e-04(x1.00): diverged=False, final_bpb=1.976747404947222, elapsed=358.19s
    lr=6.00e-04(x2.00): diverged=False, final_bpb=1.7678492952691964, elapsed=358.29s
    lr=1.20e-03(x4.00): diverged=False, final_bpb=1.6621888441031092, elapsed=358.34s
    
    採用した学習率: 1.200e-03(006 基準の x4.00)
    gradient clipping 閾値: 0.6341(勾配ノルムの 90% 分位点)



```python
# --- 1 ステップあたりの実行時間のスケーリング計測(モデル学習が線形時間であることの確認) ---
_step_scaling_steps = sorted(
    {
        max(5, CALIBRATION_STEPS // 8),
        max(10, CALIBRATION_STEPS // 4),
        max(15, CALIBRATION_STEPS // 2),
    }
)
_step_scaling_times = []
for _n in _step_scaling_steps:
    _t0 = time.time()
    _run_calibration(PEAK_LEARNING_RATE, _n, seed=1)
    _elapsed = time.time() - _t0
    _step_scaling_times.append(_elapsed)
    print(f"num_steps={_n}: 実行時間={_elapsed:.2f}s")

_log_n = np.log(_step_scaling_steps)
_log_t = np.log(_step_scaling_times)
_step_exponent, _step_log_a = np.polyfit(_log_n, _log_t, deg=1)
# log-log の切片 a は n=1 での外挿時間にあたるが、n=1 付近はモデル構築などの固定
# オーバーヘッドが相対的に大きく1ステップの実コストを過大評価しうる。そのため
# 1 ステップあたりの時間は、最も大きい計測点(固定オーバーヘッドの相対比率が最小)
# における時間 / ステップ数の比から求める。
_per_step_seconds = _step_scaling_times[-1] / _step_scaling_steps[-1]
_production_time_estimate = _per_step_seconds * PRODUCTION_NUM_STEPS

print(f"\nべき指数 b={_step_exponent:.3f}(線形なら b~=1)")
print(f"1 ステップあたりの時間(最大計測点からの実測比): {_per_step_seconds * 1000:.1f} ms")
print(
    f"本番ステップ数({PRODUCTION_NUM_STEPS:,})への外挿(学習 1 回分): "
    f"{_production_time_estimate / 60:.1f} 分(予算 {SESSION_BUDGET_SECONDS / 60:.0f} 分)"
)
if _production_time_estimate > SESSION_BUDGET_SECONDS:
    print(
        "警告: 本番 1 回の学習だけでセッション予算を超える見積もり。BATCH_SIZE や生成実験の"
        "規模を見直す必要がある。"
    )
else:
    _budget_ratio = SESSION_BUDGET_SECONDS / _production_time_estimate
    print(f"OK: 見積もり実行時間は予算に対して {_budget_ratio:.1f} 倍の余裕がある。")
```

    num_steps=272: 実行時間=48.51s
    num_steps=545: 実行時間=92.75s
    num_steps=1090: 実行時間=181.44s
    
    べき指数 b=0.950(線形なら b~=1)
    1 ステップあたりの時間(最大計測点からの実測比): 166.5 ms
    本番ステップ数(2,181)への外挿(学習 1 回分): 6.1 分(予算 120 分)
    OK: 見積もり実行時間は予算に対して 19.8 倍の余裕がある。


### 5.10 モデル学習の実行

**この学習は「主張を検証する実験」ではなく、7 節以降の生成品質比較実験のための
モデル準備である**(0 節)ため、判定基準の事前登録は不要である。検証用
bits-per-byte の推移は記録し、学習が発散していないことを確認する(前提条件 P0)。
結果の永続化(5.11 節)により、再実行時は既存の結果を再利用する。



```python
RESULTS_CACHE_PATH = ROOT / ".cache" / "008_model_cache.json"
MODEL_STATE_PATH = ROOT / ".cache" / "008_model_state.pt"


def train_production_model(seed: int = SEED) -> tuple[GPTLanguageModel, dict]:
    model = build_model(seed=seed).to(device)
    optimizer = AdamW(model.parameters(), lr=PEAK_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    schedule = functools.partial(
        compute_warmup_cosine_learning_rate,
        warmup_steps=WARMUP_STEPS,
        total_steps=NUM_STEPS,
        peak_learning_rate=PEAK_LEARNING_RATE,
        min_learning_rate=PEAK_LEARNING_RATE * MIN_LEARNING_RATE_RATIO,
    )
    history = train_language_model(
        model,
        train_ids,
        eval_windows,
        eval_mask,
        total_eval_bytes,
        num_steps=NUM_STEPS,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        learning_rate=PEAK_LEARNING_RATE,
        eval_interval=EVAL_INTERVAL,
        device=device,
        seed=seed,
        optimizer=optimizer,
        learning_rate_schedule=schedule,
        gradient_clip_threshold=GRADIENT_CLIP_THRESHOLD,
    )
    return model, history


t0 = time.time()
model, train_history = train_production_model()
_final_bpb = train_history["eval_bits_per_byte"][-1]
print(f"学習完了: {time.time() - t0:.1f} s, 最終検証 bits-per-byte = {_final_bpb:.4f}")

RESULTS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULTS_CACHE_PATH.write_text(json.dumps(train_history), encoding="utf-8")
torch.save(model.state_dict(), MODEL_STATE_PATH)
print(f"学習履歴を {RESULTS_CACHE_PATH} に、state_dict を {MODEL_STATE_PATH} に保存した")
```

    学習完了: 376.9 s, 最終検証 bits-per-byte = 1.6707
    学習履歴を .cache/008_model_cache.json に、state_dict を .cache/008_model_state.pt に保存した



```python
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
axes[0].plot(train_history["step"], train_history["train_loss"], linewidth=0.8, alpha=0.7)
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Train loss (nats)")
axes[0].set_title("Training loss")
axes[0].grid(alpha=0.3)

axes[1].plot(train_history["eval_step"], train_history["eval_bits_per_byte"], marker="o")
axes[1].axhline(
    PRECONDITION_BPB_THRESHOLD,
    color="tab:red",
    linestyle="--",
    label=f"P0 threshold ({PRECONDITION_BPB_THRESHOLD})",
)
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Validation bits-per-byte")
axes[1].set_title("Validation bits-per-byte")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)
fig.tight_layout()
plt.show()

FINAL_EVAL_BPB = train_history["eval_bits_per_byte"][-1]
PRECONDITION_P0_OK = FINAL_EVAL_BPB <= PRECONDITION_BPB_THRESHOLD
print(
    f"前提条件 P0(学習の進行): 最終検証 bits-per-byte = {FINAL_EVAL_BPB:.4f} "
    f"<= {PRECONDITION_BPB_THRESHOLD} ? {'OK' if PRECONDITION_P0_OK else 'NG'}"
)
if SMOKE_TEST:
    print(
        "(SMOKE_TEST=True: スモークスケールでの動作確認のみが目的であり、P0 の成否は"
        "参考情報である。本番実行後にあらためて確認する。)"
    )
```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/008_decoding_strategies/output_35_0.png)
    


    前提条件 P0(学習の進行): 最終検証 bits-per-byte = 1.6707 <= 2.5 ? OK


### 5.11 生成ヘルパー: 各デコーディング戦略での生成

温度サンプリング・top-k・top-p は 1 ステップずつ手動でループし(`generate()` は
top-k・top-p に対応しないため)、`model.forward` を直接呼び出す。貪欲法は
`model.generate(temperature=0.0)` をそのまま使う。



```python
@torch.no_grad()
def generate_with_filter(
    model: GPTLanguageModel,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """temperature サンプリングに top-k・top-p フィルタを組み合わせた生成(008)。

    ``top_k``・``top_p`` のいずれも None の場合、素の temperature サンプリングになる
    (``GPTLanguageModel.generate()`` と等価)。両方指定された場合は top-k を先に適用してから
    top-p を適用する(一般的な実装上の慣習、Holtzman et al. 2020 の実験でも同様の併用例がある)。
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device=prompt_ids.device)
        generator.manual_seed(seed)

    was_training = model.training
    model.eval()
    token_ids = prompt_ids
    try:
        for _ in range(max_new_tokens):
            context = token_ids[:, -model.max_sequence_length :]
            logits = model(context)
            next_logits = logits[:, -1, :]
            if top_k is not None:
                next_logits = top_k_filter(next_logits, top_k)
            if top_p is not None:
                next_logits = top_p_filter(next_logits, top_p)
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)
            token_ids = torch.cat([token_ids, next_token], dim=1)
    finally:
        model.train(was_training)
    return token_ids


# 動作確認(較正用の小さいモデルではなく、本番学習した model で 1 回ずつ実行できることを確認)。
_probe_prompt = train_ids[:8].unsqueeze(0).to(device)
model.to(device)
_greedy_probe = model.generate(_probe_prompt.clone(), max_new_tokens=10, temperature=0.0)
_temp_probe = generate_with_filter(model, _probe_prompt.clone(), 10, temperature=1.0, seed=0)
_topk_probe = generate_with_filter(
    model, _probe_prompt.clone(), 10, temperature=1.0, top_k=40, seed=0
)
_topp_probe = generate_with_filter(
    model, _probe_prompt.clone(), 10, temperature=1.0, top_p=0.9, seed=0
)
_beam_probe = beam_search(
    model, _probe_prompt.clone(), beam_size=4, max_new_tokens=10, length_penalty=1.0
)
print("greedy:", tokenizer.decode(_greedy_probe[0].cpu())[:120])
print("temp  :", tokenizer.decode(_temp_probe[0].cpu())[:120])
print("top-k :", tokenizer.decode(_topk_probe[0].cpu())[:120])
print("top-p :", tokenizer.decode(_topp_probe[0].cpu())[:120])
print("beam  :", tokenizer.decode(_beam_probe[0][0][0].cpu())[:120])
```

    greedy: Conjugable words (verbed by the name of the same name)
    temp  : Conjugable words (verpended for the seat use) for the Dec
    top-k : Conjugable words (verpended for the nativature), the name
    top-p : Conjugable words (verpended for the nativature), the name
    beam  : Conjugable words (verbed into the name of the same name)


### 5.12 Hugging Face Hub へのアップロード

**この節の関数は Claude Code(ローカル、第 1 段階)では呼び出さない。** `SMOKE_TEST=False`
かつ Google Colab 実行時(`IN_COLAB=True`)にのみ実際にアップロードを行うガードを設ける。
トークンの値はコード中に一切書き込まず、Google Colab の Secrets(鍵アイコン)に
`HF_TOKEN` という名前で事前に登録しておいたものを `google.colab.userdata.get("HF_TOKEN")`
で読み出す。



```python
HF_REPO_ID = "kojikojiprg/ai-theories-small-gpt-en"  # アップロード先(public)


def upload_model_to_hub(model: GPTLanguageModel, tokenizer: BPEIDTokenizer, repo_id: str) -> None:
    """学習済みモデル・トークナイザ・モデルカードを Hugging Face Hub にアップロードする。

    Google Colab の Secrets に登録した HF_TOKEN を使う(トークンの値はここには書かない)。
    呼び出し前に、アップロードするアーティファクトの正しさをネットワーク呼び出しなしで
    検証しておくこと(5.13 節の state_dict ラウンドトリップ検証)。
    """
    import tempfile

    from google.colab import userdata
    from huggingface_hub import HfApi

    token = userdata.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        torch.save(model.state_dict(), tmpdir / "model_state.pt")

        config = {
            "vocabulary_size": tokenizer.vocab_size,
            "d_model": D_MODEL,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
            "d_ff": D_FF,
            "swiglu_d_ff": SWIGLU_D_FF,
            "sequence_length": SEQUENCE_LENGTH,
            "dropout": DROPOUT,
            "tie_embeddings": TIE_EMBEDDINGS,
            "positional_encoding": "rope",
            "normalization": "rmsnorm",
            "feed_forward": "swiglu",
            "norm_first": True,
        }
        (tmpdir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

        tokenizer_data = {
            "merges": tokenizer.bpe_tokenizer.merges,
            "vocab": sorted(tokenizer.bpe_tokenizer.vocab),
            "byte_level": tokenizer.bpe_tokenizer.byte_level,
            "chunk_split_mode": tokenizer.bpe_tokenizer.chunk_split_mode,
            "max_chunk_bytes": tokenizer.bpe_tokenizer.max_chunk_bytes,
            "symbol_to_id": tokenizer.symbol_to_id,
        }
        (tmpdir / "tokenizer.json").write_text(
            json.dumps(tokenizer_data, indent=2), encoding="utf-8"
        )

        model_card = f"""---
language: en
license: mit
tags:
- ai-theories
- gpt
- scratch-implementation
---

# ai-theories 標準小型 GPT モデル(英語)

`ai-theories`(https://github.com/kojikojiprg/ai-theories)プロジェクトの成果物。
Transformer decoder-only の GPT スタイル言語モデルを PyTorch でスクラッチ実装し、
英語 Wikipedia コーパスの一部で事前学習したもの({NUM_STEPS} ステップ、
バッチサイズ {BATCH_SIZE}、系列長 {SEQUENCE_LENGTH})。

研究・教育目的のモデルであり、品質保証は行っていない。商用・実運用での利用は想定しない。

## 関連ノートブック

- [006. 小型 GPT の事前学習](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/006_pretraining_small_gpt-theory)
- [007. 学習の安定化](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/007_training_stabilization-theory)
- [008. デコーディング戦略](https://github.com/kojikojiprg/ai-theories/blob/main/theories/02_pretraining/008_decoding_strategies.ipynb)

## 構成

`config.json` を参照。トークナイザはバイトレベル BPE(語彙サイズ {tokenizer.vocab_size}、
`tokenizer.json`)。
"""
        (tmpdir / "README.md").write_text(model_card, encoding="utf-8")

        api.upload_folder(repo_id=repo_id, folder_path=str(tmpdir), repo_type="model")


if not SMOKE_TEST and IN_COLAB:
    upload_model_to_hub(model, tokenizer, HF_REPO_ID)
    print(f"アップロード完了: https://huggingface.co/{HF_REPO_ID}")
else:
    print(
        "アップロードをスキップした(SMOKE_TEST=True またはローカル実行のため)。"
        "本番実行は Google Colab で SMOKE_TEST=False の状態で行うこと。"
    )
```

    アップロード完了: https://huggingface.co/kojikojiprg/ai-theories-small-gpt-en


## 6. 不変条件のアサーション / Invariant Assertions

CLAUDE.md「不変条件のアサーション」節に従い、実験の妥当性が依存する量をすべて
アサーションで確認する。



```python
# --- (1) state_dict の保存・再読み込みラウンドトリップで出力が完全一致すること(2.4 節) ---
_probe_input = train_ids[:SEQUENCE_LENGTH].unsqueeze(0).to(device)
model.eval()
with torch.no_grad():
    _logits_before = model(_probe_input).clone()

_tmp_state_path = ROOT / ".cache" / "008_roundtrip_check.pt"
torch.save(model.state_dict(), _tmp_state_path)
_reloaded_model = build_model(seed=999).to(device)  # 異なるシードで初期化した別インスタンス
_reloaded_model.load_state_dict(torch.load(_tmp_state_path, map_location=device))
_reloaded_model.eval()
with torch.no_grad():
    _logits_after = _reloaded_model(_probe_input)

assert torch.equal(_logits_before, _logits_after), "state_dict のラウンドトリップで出力が一致しない"
print("OK: state_dict 保存・再読み込みラウンドトリップで logits が完全一致")
_tmp_state_path.unlink()
del _reloaded_model, _logits_before, _logits_after
```

    OK: state_dict 保存・再読み込みラウンドトリップで logits が完全一致



```python
# --- (2) beam_search(beam_size=1) が generate() の貪欲法出力と完全一致すること ---
_bs1_prompt = train_ids[:12].unsqueeze(0).to(device)
_greedy_check = model.generate(_bs1_prompt.clone(), max_new_tokens=8, temperature=0.0)
_beam1_check = beam_search(
    model, _bs1_prompt.clone(), beam_size=1, max_new_tokens=8, length_penalty=1.0
)
assert torch.equal(_greedy_check, _beam1_check[0][0]), "beam_size=1 が貪欲法と一致しない"
print("OK: beam_search(beam_size=1) が model.generate(temperature=0.0) と完全一致")
```

    OK: beam_search(beam_size=1) が model.generate(temperature=0.0) と完全一致



```python
# --- (3) top_k_filter(k=vocab_size) がフィルタなしの元の logits と一致すること ---
_probe_logits = torch.randn(4, tokenizer.vocab_size)
_filtered_full_k = top_k_filter(_probe_logits, tokenizer.vocab_size)
assert torch.allclose(_filtered_full_k, _probe_logits), (
    "top_k_filter(k=V) が元の logits と一致しない"
)
print("OK: top_k_filter(k=vocab_size) はフィルタなしと一致")

# --- (4) top_p_filter(p=1.0) がフィルタなしの元の logits と(数値誤差の範囲で)一致すること ---
_filtered_full_p = top_p_filter(_probe_logits, 1.0)
assert torch.allclose(_filtered_full_p, _probe_logits, atol=1e-4), (
    "top_p_filter(p=1.0) が元の logits と一致しない"
)
print("OK: top_p_filter(p=1.0) はフィルタなしと(数値誤差の範囲で)一致")
```

    OK: top_k_filter(k=vocab_size) はフィルタなしと一致
    OK: top_p_filter(p=1.0) はフィルタなしと(数値誤差の範囲で)一致



```python
# --- (5) トークナイザのラウンドトリップ(decode(encode(text)) == text)が成立すること ---
# 5.6 節で既に検証済みだが、独立した検証テキスト(学習コーパスの別の範囲)で再確認する。
_roundtrip_sample_2 = (
    train_text[100_000 : 100_000 + 20_000] if len(train_text) > 120_000 else train_text
)
assert bpe_tokenizer.decode(bpe_tokenizer.encode(_roundtrip_sample_2)) == _roundtrip_sample_2
print(f"OK: トークナイザのラウンドトリップ一致(独立サンプル、{len(_roundtrip_sample_2):,} 文字)")
```

    OK: トークナイザのラウンドトリップ一致(独立サンプル、20,000 文字)



```python
# --- (6) compute_ngram_repetition_rate が完全反復系列で高い値・完全ユニーク系列で 0 になること ---
_repeated_seq = [1, 2, 3, 1, 2, 3, 1, 2, 3]
_unique_seq = [1, 2, 3, 4, 5, 6, 7, 8, 9]
_rep_repeated = compute_ngram_repetition_rate(_repeated_seq, n=3)
_rep_unique = compute_ngram_repetition_rate(_unique_seq, n=3)
assert _rep_unique == 0.0, f"完全ユニーク系列の重複率が 0 でない: {_rep_unique}"
assert _rep_repeated > 0.5, f"完全反復系列の重複率が十分に高くない: {_rep_repeated}"
print(
    f"OK: compute_ngram_repetition_rate(完全反復)={_rep_repeated:.4f}, "
    f"compute_ngram_repetition_rate(完全ユニーク)={_rep_unique:.4f}"
)
```

    OK: compute_ngram_repetition_rate(完全反復)=0.5714, compute_ngram_repetition_rate(完全ユニーク)=0.0000




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/02_pretraining/008_decoding_strategies.ipynb
