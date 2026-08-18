---
title: "正規化と活性化の系譜(実装・実験編 1/3)"
---

この記事は後編(実装・実験編 1/3)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/004_normalization_and_activation-theory)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/004_normalization_and_activation-practice-2)。

## 4. 実装方針 / Implementation Plan

**`src/`に切り出すもの**(理論的に本質的で、他トピック・`apps/`からも再利用する部品):

- 正規化層: `src/layers/normalization.py`に`LayerNormalization`(`center`・`scale`引数を追加)と`RMSNorm`を実装する。
- 活性化関数: `src/layers/activation.py`に`gelu_exact`・`gelu_tanh_approximation`・`swish`をスクラッチ実装する。
- 順伝播ネットワークの変種: `src/layers/feedforward.py`に`SwiGLUFeedForwardNetwork`を追加する。`FeedForwardNetwork`は活性化関数の指定方法を`activation_fn`(Callable)引数 1 つに一本化する(既定値`None`のとき ReLU、001・002・003 と同一の挙動)。
- ブロックへの注入点: `src/layers/transformer_block.py`の`EncoderBlock`・`DecoderBlock`に`normalization_factory`・`feed_forward_factory`・`activation_fn`を追加する。003 で`MultiHeadAttention`に位置エンコーディングを注入したのと同じ設計方針で、注入点がブロック内部にあるため、ラッパークラスを増やすのではなく optional 引数で拡張する。
- データ処理: `src/data/text.py`に文字レベルの`CharacterLevelTokenizer`と Tiny Shakespeare のロード関数を実装する。
- 測定関数: `src/utils/statistics.py`に、隠れ状態の平均/RMS 比・常に負のユニットの割合・ユニット群ごとの勾配ノルム・層ごとの勾配ノルムを計算する関数を実装する。
- 可視化: `src/utils/visualization.py`に、複数シードの散布図・層ごとの棒グラフ・関数曲線の描画関数を追加する。

**ノートブック内に直接書くもの**:

- 条件比較用の最小構成の言語モデル(`CausalCharacterLevelLanguageModel`、下記参照)と学習ループ。
- 実験ごとのデータ収集・統計量集計のグルーコード(隠れ状態やユニット活性化前の値を集めるための forward hook など)。
- 実験の実行・可視化コード。

**モデル本体と学習ループを`src/`に出さない理由**: 本格的な自己回帰言語モデリング(Autoregressive Language Modeling)の事前学習ループは 006(小型 GPT の事前学習)で扱う。004 の目的はあくまで正規化・活性化という **条件の比較** であり、汎用的な学習基盤の整備ではないため、比較に必要な最小構成のモデル・学習ループをこのノートブック内に閉じて実装する。

**decoder-only 構成の作り方**: 言語モデリングには、各トークンが自分より前のトークンだけを参照する decoder-only な自己回帰モデルが必要になる。002 の`DecoderBlock`は Encoder の出力(memory)を参照する交差注意(cross-attention)を持つため、そのままでは言語モデリングに使えない(memory が存在しない)。本ノートブックでは、**`EncoderBlock`に因果マスク(causal mask、`create_causal_mask()`)を渡すことで decoder-only 構成を作る**。`EncoderBlock`はもともと自己注意のみの 2 サブレイヤー構成であり、マスクを渡すだけで「各位置が自分より前しか見えない自己回帰モデル」になる。名前に反して「Encoder」を使うのは紛らわしいが、`DecoderBlock`という名前のクラスを使わないのが正しい選択であることを、誤解を避けるためにここで明示しておく。

**正規化の配置**: 002 の実験 2 の結論([002](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/002_transformer_block-theory) 6.2 節)に従い、全条件で **正規化前置(Pre-Layer Normalization、`norm_first=True`)に固定** する。正規化前置・正規化後置の比較は 002 で既に扱っているため、004 では正規化そのものの中身(平均減算・分散除算の有無)の比較に集中する。

**評価データを固定する理由**: 学習ループの検証損失を、学習の各評価タイミングで`train_data`からランダムに引いた 1 バッチだけで計算すると、評価バッチ自体のサンプリングノイズが条件・シードごとに異なる形で最終検証損失に混入する。これは、6 節冒頭で定める「5 シードの最小値〜最大値の区間が重なるかどうか」という判定基準の信頼性を直接損なう(区間の重なりが、条件間の実質的な差ではなく評価ノイズによって生じている可能性を排除できない)。そこで、ノートブック上部で **評価用バッチの集合を独立した乱数生成器で 1 度だけ固定生成し、以降の全条件・全シードでこの同じ集合を使い回す**。これにより、最終検証損失の変動要因を「学習過程の違い(乱数シードと条件)」のみに限定できる。

**`CausalCharacterLevelLanguageModel`の設計**: `token_embedding`・学習可能な絶対位置埋め込み(`position_embedding`、[003](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/003_positional_encoding_rope-theory)で導入した学習可能絶対位置埋め込みの最小版)・`EncoderBlock`を`n_layers`個積んだ`blocks`・最終正規化・出力射影(`lm_head`)から成る。`normalization_factory`・`feed_forward_factory`を`EncoderBlock`にそのまま透過させることで、正規化の種類(実験 C)と活性化・順伝播ネットワークの種類(実験 E)を、モデルの他の部分を変えずに差し替えられるようにする。

## 5. 実装 / Implementation

### 5.1 環境セットアップ


```python
# 環境セットアップ(Google Colab)
# Colab 上でのみリポジトリを clone し、リポジトリルートをカレントディレクトリにする。
# ローカル(Jupyter)実行時は、リポジトリルートで起動していればそのまま動く。
import sys

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    !git clone https://github.com/kojikojiprg/ai-theories.git
    %cd ai-theories
    !pip install uv -q
    !uv pip install --system -r requirements.txt
```

    Cloning into 'ai-theories'...
    remote: Enumerating objects: 163, done.[K
    remote: Counting objects: 100% (163/163), done.[K
    remote: Compressing objects: 100% (112/112), done.[K
    remote: Total 163 (delta 75), reused 127 (delta 45), pack-reused 0 (from 0)[K
    Receiving objects: 100% (163/163), 2.58 MiB | 20.01 MiB/s, done.
    Resolving deltas: 100% (75/75), done.
    /content/ai-theories
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m22.3/22.3 MB[0m [31m70.0 MB/s[0m eta [36m0:00:00[0m
    [?25h[2mUsing Python 3.12.13 environment at: /usr[0m
    [2K[2mResolved [1m40 packages[0m [2min 353ms[0m[0m
    [2K[2mPrepared [1m27 packages[0m [2min 46.28s[0m[0m
    [2mUninstalled [1m12 packages[0m [2min 764ms[0m[0m
    [2K[2mInstalled [1m27 packages[0m [2min 319ms[0m[0m
     [31m-[39m [1mcuda-bindings[0m[2m==12.9.7[0m
     [32m+[39m [1mcuda-bindings[0m[2m==13.3.1[0m
     [31m-[39m [1mcuda-pathfinder[0m[2m==1.5.6[0m
     [32m+[39m [1mcuda-pathfinder[0m[2m==1.6.0[0m
     [31m-[39m [1mcuda-toolkit[0m[2m==12.8.1[0m
     [32m+[39m [1mcuda-toolkit[0m[2m==13.0.3.0[0m
     [31m-[39m [1mfilelock[0m[2m==3.29.7[0m
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
     [32m+[39m [1mnvidia-nccl-cu13[0m[2m==2.29.7[0m
     [32m+[39m [1mnvidia-nvjitlink[0m[2m==13.3.33[0m
     [32m+[39m [1mnvidia-nvshmem-cu13[0m[2m==3.4.5[0m
     [32m+[39m [1mnvidia-nvtx[0m[2m==13.0.85[0m
     [31m-[39m [1mpackaging[0m[2m==26.2[0m
     [32m+[39m [1mpackaging[0m[2m==26.3[0m
     [31m-[39m [1mpillow[0m[2m==11.3.0[0m
     [32m+[39m [1mpillow[0m[2m==12.3.0[0m
     [31m-[39m [1msetuptools[0m[2m==75.2.0[0m
     [32m+[39m [1msetuptools[0m[2m==84.0.0[0m
     [31m-[39m [1mtorch[0m[2m==2.11.0+cu128[0m
     [32m+[39m [1mtorch[0m[2m==2.13.0[0m
     [31m-[39m [1mtriton[0m[2m==3.6.0[0m
     [32m+[39m [1mtriton[0m[2m==3.7.1[0m



```python
# `from src...` の絶対 import が通るように、リポジトリルートを sys.path に追加する。
# (ノートブックを theories/01_foundations/ から直接開いた場合の保険)
import os
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "src").exists():  # theories/01_foundations/ から起動した場合
    ROOT = ROOT.parents[1]
    os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
print("repository root:", ROOT)
```

    repository root: /content/ai-theories



```python
import inspect
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.data import (
    CharacterLevelTokenizer,
    get_random_batch,
    load_tiny_shakespeare,
    split_train_val,
)
from src.layers import (
    EncoderBlock,
    FeedForwardNetwork,
    LayerNormalization,
    RMSNorm,
    SwiGLUFeedForwardNetwork,
    create_causal_mask,
    gelu_exact,
    gelu_tanh_approximation,
    swish,
)
from src.utils import (
    compute_always_negative_unit_ratio,
    compute_gradient_norm_by_unit_group,
    compute_gradient_norm_per_layer,
    compute_mean_to_rms_ratio,
    plot_bar_by_layer,
    plot_function_curves,
    plot_learning_curves,
    plot_learning_curves_multi_seed,
    plot_seed_scatter,
)

# 再現性のためのシード固定
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
)
print(f"torch: {torch.__version__} / device: {device}")
```

    torch: 2.13.0+cu130 / device: cuda



```python
if IN_COLAB:
    !nvidia-smi
else:
    print("Colab 環境ではないため nvidia-smi の出力は省略する。")
```

    Wed Aug 12 07:54:38 2026       
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
    +-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
    | N/A   54C    P8             14W /   70W |       3MiB /  15360MiB |      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+
    
    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
    |  No running processes found                                                             |
    +-----------------------------------------------------------------------------------------+


### 5.2 正規化層の実装確認: `center`・`scale`と`RMSNorm`

`LayerNormalization`の`center=True, scale=True`(既定値)は 002 と完全に同一の計算になっていることを確認し、続けて`center=False, scale=True`(RMSNorm と数学的に同じ計算)が初期化直後(`beta=0`)に`RMSNorm`と一致することを確認する(これは実験 B でも扱う相互検証だが、実装確認の時点でも確かめておく)。


```python
print(inspect.getsource(LayerNormalization.forward))
```

        def forward(self, x: Tensor) -> Tensor:
            """層正規化を適用する。
    
            Args:
                x: 形状 ``(..., d_model)`` の入力。
    
            Returns:
                x と同じ形状の正規化済みテンソル。
            """
            if self.center and self.scale:
                # center=True, scale=True(既定値)は 002 と完全に同一の計算にする。
                mean = x.mean(dim=-1, keepdim=True)
                # unbiased=False: 分散は 1/d_model の標本分散を使う(nn.LayerNorm と同じ規約)。
                var = x.var(dim=-1, keepdim=True, unbiased=False)
                x_hat = (x - mean) / torch.sqrt(var + self.eps)
                return self.gamma * x_hat + self.beta
    
            x_hat = x
            if self.center:
                x_hat = x_hat - x_hat.mean(dim=-1, keepdim=True)
            if self.scale:
                # center=False の場合、x_hat = x に対する 2 乗平均(平均減算前の分散)になる。
                mean_square = x_hat.pow(2).mean(dim=-1, keepdim=True)
                x_hat = x_hat / torch.sqrt(mean_square + self.eps)
            return self.gamma * x_hat + self.beta
    



```python
torch.manual_seed(SEED)
D_MODEL_CHECK = 32
x_check = torch.randn(4, 6, D_MODEL_CHECK)

ln_default = LayerNormalization(D_MODEL_CHECK)
mean = x_check.mean(dim=-1, keepdim=True)
var = x_check.var(dim=-1, keepdim=True, unbiased=False)
expected = (x_check - mean) / torch.sqrt(var + ln_default.eps)
print(
    "LayerNormalization(既定値)が手計算と一致:",
    torch.allclose(ln_default(x_check), expected, atol=1e-6),
)

ln_scale_only = LayerNormalization(D_MODEL_CHECK, center=False, scale=True)
rms_check = RMSNorm(D_MODEL_CHECK)
print(
    "center=False, scale=True と RMSNorm が初期化直後に一致:",
    torch.allclose(ln_scale_only(x_check), rms_check(x_check), atol=1e-6),
)

for center, scale, label in [
    (True, True, "層正規化(center=True, scale=True)"),
    (False, True, "分散除算のみ(center=False, scale=True)"),
    (True, False, "平均減算のみ(center=True, scale=False)"),
    (False, False, "正規化なし(center=False, scale=False)"),
]:
    m = LayerNormalization(D_MODEL_CHECK, center=center, scale=scale)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"  {label:38s}: パラメータ数 = {n_params}(4 条件とも d_model の 2 倍で共通)")
```

    LayerNormalization(既定値)が手計算と一致: True
    center=False, scale=True と RMSNorm が初期化直後に一致: True
      層正規化(center=True, scale=True)         : パラメータ数 = 64(4 条件とも d_model の 2 倍で共通)
      分散除算のみ(center=False, scale=True)      : パラメータ数 = 64(4 条件とも d_model の 2 倍で共通)
      平均減算のみ(center=True, scale=False)      : パラメータ数 = 64(4 条件とも d_model の 2 倍で共通)
      正規化なし(center=False, scale=False)      : パラメータ数 = 64(4 条件とも d_model の 2 倍で共通)


### 5.3 活性化関数の実装確認: スクラッチ実装と`torch.nn.functional`の数値一致

GELU・Swish は`torch.nn.functional`を使わずスクラッチ実装した(`src/layers/activation.py`)。正しさの確認として、対応する組み込み実装との最大絶対誤差を計測する。


```python
x_act = torch.linspace(-6.0, 6.0, 2001)

diff_gelu_exact = (gelu_exact(x_act) - F.gelu(x_act)).abs().max().item()
diff_gelu_tanh = (
    (gelu_tanh_approximation(x_act) - F.gelu(x_act, approximate="tanh")).abs().max().item()
)
diff_swish = (swish(x_act, beta=1.0) - F.silu(x_act)).abs().max().item()
diff_gelu_exact_vs_tanh = (gelu_exact(x_act) - gelu_tanh_approximation(x_act)).abs().max().item()

print(f"gelu_exact            vs F.gelu                    最大絶対誤差: {diff_gelu_exact:.3e}")
print(f"gelu_tanh_approximation vs F.gelu(approximate=tanh) 最大絶対誤差: {diff_gelu_tanh:.3e}")
print(f"swish(beta=1)          vs F.silu                    最大絶対誤差: {diff_swish:.3e}")
print(
    f"gelu_exact             vs gelu_tanh_approximation    "
    f"最大絶対誤差: {diff_gelu_exact_vs_tanh:.3e}"
)
```

    gelu_exact            vs F.gelu                    最大絶対誤差: 9.537e-07
    gelu_tanh_approximation vs F.gelu(approximate=tanh) 最大絶対誤差: 2.384e-07
    swish(beta=1)          vs F.silu                    最大絶対誤差: 4.768e-07
    gelu_exact             vs gelu_tanh_approximation    最大絶対誤差: 4.733e-04


### 5.4 順伝播ネットワークの変種の実装確認: `SwiGLUFeedForwardNetwork`

3.7 節で導出したパラメータ数を揃える中間次元 $d_{\text{ff}}' = \mathrm{round}(2/3 \times d_{\text{ff}})$ を、共通ハイパーパラメータ($d_{\text{model}}=256$、$d_{\text{ff}}=1024$)で実際に計算し、理論値どおりのパラメータ数になっていることを確認する。`FeedForwardNetwork`は`activation_fn`(Callable)引数に一本化してあるため(004 で、文字列引数との併存を解消)、GELU を使いたい場合は`activation_fn=gelu_exact`を渡す(既定値`None`は ReLU、001・002・003 と同一の挙動)。


```python
D_MODEL_COMMON, N_LAYERS_COMMON, N_HEADS_COMMON, D_FF_COMMON, SEQ_LEN_COMMON = 256, 4, 4, 1024, 128
D_FF_SWIGLU_MATCHED = round((2 / 3) * D_FF_COMMON)

feed_forward_standard = FeedForwardNetwork(D_MODEL_COMMON, D_FF_COMMON, activation_fn=gelu_exact)
feed_forward_swiglu_matched = SwiGLUFeedForwardNetwork(D_MODEL_COMMON, D_FF_SWIGLU_MATCHED)
feed_forward_swiglu_naive = SwiGLUFeedForwardNetwork(D_MODEL_COMMON, D_FF_COMMON)

n_standard = sum(p.numel() for p in feed_forward_standard.parameters())
n_matched = sum(p.numel() for p in feed_forward_swiglu_matched.parameters())
n_naive = sum(p.numel() for p in feed_forward_swiglu_naive.parameters())

print(f"標準(GELU, d_ff={D_FF_COMMON})                             パラメータ数: {n_standard:,}")
print(
    f"SwiGLU(パラメータ数を揃えた条件, d_ff'={D_FF_SWIGLU_MATCHED})   パラメータ数: {n_matched:,}"
)
print(
    f"SwiGLU(素朴な置換, d_ff={D_FF_COMMON})                       "
    f"パラメータ数: {n_naive:,}({n_naive / n_standard:.3f} 倍)"
)

x_feed_forward_check = torch.randn(2, 5, D_MODEL_COMMON)
print("SwiGLU 出力形状:", tuple(feed_forward_swiglu_matched(x_feed_forward_check).shape))
```

    標準(GELU, d_ff=1024)                             パラメータ数: 525,568
    SwiGLU(パラメータ数を揃えた条件, d_ff'=683)   パラメータ数: 524,544
    SwiGLU(素朴な置換, d_ff=1024)                       パラメータ数: 786,432(1.496 倍)
    SwiGLU 出力形状: (2, 5, 256)


### 5.5 `EncoderBlock`への注入点の実装確認

`normalization_factory`・`feed_forward_factory`を指定しない場合、002・003 と完全に同じ`LayerNormalization` + `FeedForwardNetwork`(ReLU)が使われることを確認する。


```python
torch.manual_seed(SEED)
encoder_block_default = EncoderBlock(D_MODEL_COMMON, N_HEADS_COMMON, D_FF_COMMON)
print("既定のnormalization:", type(encoder_block_default.norm1).__name__)
print("既定のfeed_forward:", type(encoder_block_default.feed_forward).__name__)
print("既定の活性化関数:", type(encoder_block_default.feed_forward.activation).__name__)

encoder_block_injected = EncoderBlock(
    D_MODEL_COMMON,
    N_HEADS_COMMON,
    D_FF_COMMON,
    normalization_factory=lambda d: RMSNorm(d),
    feed_forward_factory=lambda: SwiGLUFeedForwardNetwork(D_MODEL_COMMON, D_FF_SWIGLU_MATCHED),
)
print("注入後のnormalization:", type(encoder_block_injected.norm1).__name__)
print("注入後のfeed_forward:", type(encoder_block_injected.feed_forward).__name__)

x_block_check = torch.randn(2, SEQ_LEN_COMMON, D_MODEL_COMMON)
mask_block_check = create_causal_mask(SEQ_LEN_COMMON)
out_block_check, attn_block_check = encoder_block_injected(x_block_check, mask_block_check)
print("因果マスクつき EncoderBlock 出力形状:", tuple(out_block_check.shape))
```

    既定のnormalization: LayerNormalization
    既定のfeed_forward: FeedForwardNetwork
    既定の活性化関数: ReLU
    注入後のnormalization: RMSNorm
    注入後のfeed_forward: SwiGLUFeedForwardNetwork
    因果マスクつき EncoderBlock 出力形状: (2, 128, 256)


### 5.6 データ: Tiny Shakespeare の文字レベル分割

`src/data/text.py`の`CharacterLevelTokenizer`(文字単位の最小限の処理。サブワード分割は 005 で扱う)で、Andrej Karpathy の char-rnn リポジトリで配布されている Tiny Shakespeare データセットを文字 ID 列に変換し、学習用・検証用に分割する。


```python
CACHE_DIR = ROOT / ".cache" / "tiny_shakespeare"
raw_text = load_tiny_shakespeare(CACHE_DIR)
print(f"テキスト長: {len(raw_text):,} 文字")
print(raw_text[:200])
```

    テキスト長: 1,115,394 文字
    First Citizen:
    Before we proceed any further, hear me speak.
    
    All:
    Speak, speak.
    
    First Citizen:
    You are all resolved rather to die than to famish?
    
    All:
    Resolved. resolved.
    
    First Citizen:
    First, you



```python
tokenizer = CharacterLevelTokenizer(raw_text)
VOCAB_SIZE = tokenizer.vocab_size
print(f"語彙サイズ V = {VOCAB_SIZE}")

token_ids = tokenizer.encode(raw_text)
train_data, val_data = split_train_val(token_ids, val_ratio=0.1)
print(f"学習用: {len(train_data):,} 文字 / 検証用: {len(val_data):,} 文字")

# encode/decode の往復確認
sample_text = raw_text[:100]
assert tokenizer.decode(tokenizer.encode(sample_text)) == sample_text
print("encode -> decode の往復確認: OK")

x_batch_demo, y_batch_demo = get_random_batch(
    train_data, batch_size=4, seq_len=16, generator=torch.Generator().manual_seed(SEED)
)
print("バッチ形状 (inputs, targets):", tuple(x_batch_demo.shape), tuple(y_batch_demo.shape))
print("入力例:", repr(tokenizer.decode(x_batch_demo[0])))
print("targetは1つ右にずれている:", repr(tokenizer.decode(y_batch_demo[0])))
```

    語彙サイズ V = 65
    学習用: 1,003,855 文字 / 検証用: 111,539 文字
    encode -> decode の往復確認: OK
    バッチ形状 (inputs, targets): (4, 16) (4, 16)
    入力例: '\nNot in a grave,'
    targetは1つ右にずれている: 'Not in a grave,\n'


### 5.7 `CausalCharacterLevelLanguageModel`: 条件比較用の最小構成の言語モデル

4 節で述べた通り、本格的な事前学習ループは 006 で扱うため、ここでは条件比較に必要な最小構成のみを実装する。`normalization_factory`・`feed_forward_factory`を通じて、正規化・順伝播ネットワークの実装を差し替えられるようにしてある。`record_layer_outputs=True`を渡すと、各`EncoderBlock`の出力に`retain_grad()`を呼んで`self.layer_outputs`に記録する(実験 D で`compute_gradient_norm_per_layer()`と組み合わせて使う)。


```python
class CausalCharacterLevelLanguageModel(nn.Module):
    """条件比較用の最小構成の decoder-only 文字レベル言語モデル。

    `EncoderBlock` に因果マスクを与えることで decoder-only 構成にする(4 節参照)。
    本格的な事前学習ループ(データローダの効率化、KV キャッシュ、学習率のチューニングなど)は
    006 で扱うため、本クラスは 004 の条件比較(正規化・活性化の差し替え)に
    必要な最小構成に留める。

    Args:
        vocab_size: 語彙サイズ V。
        d_model, n_layers, n_heads, d_ff: Transformer Block の次元・層数・ヘッド数・中間次元。
        seq_len: 学習可能絶対位置埋め込みが対応する最大系列長。
        normalization_factory: `EncoderBlock` にそのまま渡す正規化層の factory。
        feed_forward_factory: `EncoderBlock` にそのまま渡す順伝播ネットワークの factory。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        seq_len: int,
        normalization_factory=None,
        feed_forward_factory=None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    d_model,
                    n_heads,
                    d_ff,
                    normalization_factory=normalization_factory,
                    feed_forward_factory=feed_forward_factory,
                )
                for _ in range(n_layers)
            ]
        )
        norm_fn = normalization_factory or LayerNormalization
        self.final_norm = norm_fn(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: Tensor, record_layer_outputs: bool = False) -> Tensor:
        _, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)[None]
        mask = create_causal_mask(seq_len, device=input_ids.device)

        if record_layer_outputs:
            self.layer_outputs = []
        for block in self.blocks:
            hidden, _ = block(hidden, mask)
            if record_layer_outputs:
                hidden.retain_grad()
                self.layer_outputs.append(hidden)

        hidden = self.final_norm(hidden)
        return self.lm_head(hidden)


torch.manual_seed(SEED)
language_model_check = CausalCharacterLevelLanguageModel(VOCAB_SIZE, 32, 2, 4, 64, 16)
x_language_model_check = torch.randint(0, VOCAB_SIZE, (3, 16))
print(
    "CausalCharacterLevelLanguageModel 出力形状:",
    tuple(language_model_check(x_language_model_check).shape),
    "(期待値: (3, 16, V))",
)
```

    CausalCharacterLevelLanguageModel 出力形状: (3, 16, 65) (期待値: (3, 16, V))


### 5.8 評価用バッチの固定(評価ノイズの除去)

4 節で述べた通り、最終検証損失に評価バッチのサンプリングノイズが混入しないよう、**評価用バッチの集合を独立した乱数生成器で 1 度だけ固定生成し、以降の全条件・全シードでこの同じ集合を使い回す**。評価は固定集合全体にわたるトークン数で重み付けした平均損失とする。


```python
N_EVALUATION_BATCHES = 20
BATCH_SIZE_LANGUAGE_MODEL = 32

# 学習バッチの生成(条件・シードごとに異なる)とは独立した乱数生成器を使う。
evaluation_batch_generator = torch.Generator().manual_seed(12345)
evaluation_batches = [
    get_random_batch(
        val_data, BATCH_SIZE_LANGUAGE_MODEL, SEQ_LEN_COMMON, generator=evaluation_batch_generator
    )
    for _ in range(N_EVALUATION_BATCHES)
]
print(
    f"固定評価バッチ: {N_EVALUATION_BATCHES} バッチ "
    f"× {BATCH_SIZE_LANGUAGE_MODEL} × {SEQ_LEN_COMMON} "
    f"= {N_EVALUATION_BATCHES * BATCH_SIZE_LANGUAGE_MODEL * SEQ_LEN_COMMON:,} トークン"
)


# 固定評価バッチ集合全体にわたる、トークン数で重み付けした平均損失を計算する。
def evaluate_language_model(model: nn.Module, evaluation_batches, device: torch.device) -> float:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss_sum = 0.0
    total_tokens = 0
    with torch.no_grad():
        for inputs, targets in evaluation_batches:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            total_loss_sum += loss_fn(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            ).item()
            total_tokens += targets.numel()
    return total_loss_sum / total_tokens
```

    固定評価バッチ: 20 バッチ × 32 × 128 = 81,920 トークン


### 5.9 学習ループ

線形 warmup + cosine 減衰の学習率スケジュール、勾配クリッピングを備えた最小限の学習ループを実装する。損失が非有限(NaN / Inf)になった場合は打ち切り、`diverged=True`として発散した事実をそのまま記録する(実験 C の陽性対照条件で使う。ハイパーパラメータは条件によらず共通のものを使う)。検証損失の計算は 5.8 節で固定した`evaluation_batches`を使う。


```python
def train_character_level_language_model(
    model: nn.Module,
    train_data: Tensor,
    evaluation_batches,
    steps: int,
    batch_size: int,
    seq_len: int,
    learning_rate: float,
    warmup_steps: int,
    seed: int,
    device: torch.device,
    evaluate_every: int = 50,
    grad_clip: float = 1.0,
) -> tuple[dict, bool]:
    """文字レベル言語モデルの最小限の学習ループ。

    Returns:
        (history, diverged) のタプル。history は {"step", "train_loss", "evaluation_loss"} を
        持つ辞書。diverged は損失が非有限になり途中で打ち切った場合に True。
    """
    torch.manual_seed(seed)
    batch_generator = torch.Generator().manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = nn.CrossEntropyLoss()
    history = {"step": [], "train_loss": [], "evaluation_loss": []}
    diverged = False

    for step in range(steps):
        inputs, targets = get_random_batch(
            train_data, batch_size, seq_len, generator=batch_generator
        )
        inputs, targets = inputs.to(device), targets.to(device)

        model.train()
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        if not torch.isfinite(loss):
            diverged = True
            history["step"].append(step)
            history["train_loss"].append(float("nan"))
            history["evaluation_loss"].append(float("nan"))
            break

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        if step % evaluate_every == 0 or step == steps - 1:
            evaluation_loss = evaluate_language_model(model, evaluation_batches, device)
            history["step"].append(step)
            history["train_loss"].append(loss.item())
            history["evaluation_loss"].append(evaluation_loss)

    return history, diverged
```

## 6. 実験 / Experiments

> **実行環境について**: 以下の実験は、Google Colab の T4 GPU 上で実行して得た実測値である(実行時の`device`は`cuda`。`nvidia-smi`の出力で GPU が **Tesla T4** であることを確認済み)。文字レベル言語モデリングの学習(実験 C・E)は、6.3 節の較正セルで実測した 1 step あたりの所要時間に安全係数(2.0 倍)を掛けたうえで、合計 35 run の目標時間(70 分)から採用 step 数を **950**(warmup 95 step)と決定した。較正セルは、短い較正が GPU カーネルの初回コンパイル等の一度きりのコストを過大に見積もる不具合(ウォームアップ区間を計測から分離していなかったことが原因)を修正済みであり、修正後の較正の実測精度(約 15% の過大評価)については 7 節末尾で述べる。

**判定基準(実験の実行前に定める)**: 複数シードを使う実験(実験 C・E・G)では、各条件を **5 個(実験 C・E)または 10 個(実験 G)の乱数シード(seed)** で実行し、平均・標準偏差に加えて **個々のシードの値をすべてプロットに重ねる**(`plot_seed_scatter`)。条件間で 5(または 10)シードの最小値〜最大値の区間が重なる場合、優劣を主張せず「この規模では差を検出できなかった」と記述する。この基準は実験の実行前に定めたものであり、結果を見てから閾値を調整することはしない。最終評価損失の計算には 5.8 節で固定した評価用バッチ集合を全条件・全シードで共通に使うため、評価バッチのサンプリングノイズはこの判定基準の信頼性を損なわない。

### 6.1 実験 A: 活性化関数の形状(学習不要)

ReLU・GELU(厳密形・tanh 近似形)・Swish(複数の$\beta$)の関数値と導関数を描画し、3.5 節の理論(GELU の 2 つの近似の一致度、Swish の$\beta \to \infty$での ReLU への収束)を数値で確認する。


```python
x_plot = torch.linspace(-4.0, 4.0, 400).requires_grad_(True)


def grad_of(f, x):
    y = f(x)
    (grad,) = torch.autograd.grad(y.sum(), x, create_graph=False)
    return y.detach(), grad.detach()


y_relu, g_relu = grad_of(lambda x: F.relu(x), x_plot)
y_gelu_exact, g_gelu_exact = grad_of(gelu_exact, x_plot)
y_gelu_tanh, g_gelu_tanh = grad_of(gelu_tanh_approximation, x_plot)

betas = [0.5, 1.0, 2.0, 8.0]
swish_curves = {}
swish_grads = {}
for beta in betas:
    y_b, g_b = grad_of(lambda x, b=beta: swish(x, beta=b), x_plot)
    swish_curves[f"Swish (beta={beta})"] = y_b
    swish_grads[f"Swish (beta={beta})"] = g_b

x_np = x_plot.detach()

fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))

plot_function_curves(
    x_np,
    {"ReLU": y_relu, "GELU (exact)": y_gelu_exact, "GELU (tanh approx.)": y_gelu_tanh},
    title="ReLU vs GELU (value)",
    ax=axes[0, 0],
)
plot_function_curves(
    x_np,
    {"ReLU": g_relu, "GELU (exact)": g_gelu_exact, "GELU (tanh approx.)": g_gelu_tanh},
    title="ReLU vs GELU (derivative)",
    ylabel="f'(x)",
    ax=axes[0, 1],
)
plot_function_curves(
    x_np,
    {**{"ReLU": y_relu}, **swish_curves},
    title="Swish for several beta, approaching ReLU",
    ax=axes[1, 0],
)
plot_function_curves(
    x_np,
    {**{"ReLU": g_relu}, **swish_grads},
    title="Swish derivative for several beta",
    ylabel="f'(x)",
    ax=axes[1, 1],
)
fig.tight_layout()
plt.show()

max_diff_gelu = (y_gelu_exact - y_gelu_tanh).abs().max().item()
print(f"GELU 厳密形と tanh 近似形の最大絶対差(x in [-4, 4]): {max_diff_gelu:.3e}")
for beta in betas:
    diff = (swish_curves[f"Swish (beta={beta})"] - y_relu).abs().max().item()
    print(f"Swish(beta={beta:>4}) と ReLU の最大絶対差: {diff:.4f}")

relu_grad_at_zero_left = grad_of(lambda x: F.relu(x), torch.tensor([-1e-4], requires_grad=True))[
    1
].item()
relu_grad_at_zero_right = grad_of(lambda x: F.relu(x), torch.tensor([1e-4], requires_grad=True))[
    1
].item()
relu_grad_negative = grad_of(lambda x: F.relu(x), torch.tensor([-2.0], requires_grad=True))[
    1
].item()
print(
    f"ReLU の導関数: x=-1e-4 で {relu_grad_at_zero_left}, x=+1e-4 で"
    f" {relu_grad_at_zero_right}(原点付近で不連続)"
)
print(f"ReLU の導関数: x=-2.0 で {relu_grad_negative}(負領域で恒等的に 0)")
```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/004_normalization_and_activation/output_39_0.png)
    


    GELU 厳密形と tanh 近似形の最大絶対差(x in [-4, 4]): 4.733e-04
    Swish(beta= 0.5) と ReLU の最大絶対差: 0.5569
    Swish(beta= 1.0) と ReLU の最大絶対差: 0.2785
    Swish(beta= 2.0) と ReLU の最大絶対差: 0.1392
    Swish(beta= 8.0) と ReLU の最大絶対差: 0.0347
    ReLU の導関数: x=-1e-4 で 0.0, x=+1e-4 で 1.0(原点付近で不連続)
    ReLU の導関数: x=-2.0 で 0.0(負領域で恒等的に 0)


#### 実験 A の結果・考察

実測値: GELU の厳密形と tanh 近似形の最大絶対差は $4.734 \times 10^{-4}$($x \in [-4, 4]$)であり、3.5 節で述べた近似が実用上ほぼ無視できる精度であることを裏づけた。Swish($\beta$)と ReLU の最大絶対差は $\beta=0.5$ で $0.557$、$\beta=1.0$ で $0.279$、$\beta=2.0$ で $0.139$、$\beta=8.0$ で $0.035$ と、$\beta$ を大きくするにつれて単調に減少しており、3.5 節で導出した $\beta \to \infty$ での ReLU への収束を数値的に確認した。ReLU の導関数は $x=-10^{-4}$ で $0$、$x=+10^{-4}$ で $1$ となり原点付近で不連続であること、$x=-2.0$ で $0$ となり負領域で恒等的に $0$ であることも実測値で確認できた。

### 6.2 実験 B: 正規化の不変性の数値検証(学習不要)

3.3 節で導出した不変性を数値で検証する。ランダムな入力$a$に対して$a \mapsto \alpha a + b \mathbf{1}$の変換を施し、層正規化(Layer Normalization)と RMSNorm の出力がどれだけ変化するかを測る。あわせて、初期化直後($\beta = 0$)において`RMSNorm`と`LayerNormalization(center=False, scale=True)`の出力が一致することを検証し、2 つの実装の相互検証とする(5.2 節でも簡単に確認したが、ここでは複数のシードで再確認する)。


```python
torch.manual_seed(SEED)
D_MODEL_INVARIANCE = 64
N_TRIALS_INVARIANCE = 8

layer_norm_invariance = LayerNormalization(D_MODEL_INVARIANCE)
rms_norm_invariance = RMSNorm(D_MODEL_INVARIANCE)

alpha_values = [0.1, 0.5, 2.0, 10.0]
b_values = [0.0, 1.0, 5.0, -3.0]

layer_norm_alpha_diffs, layer_norm_b_diffs = [], []
rms_norm_alpha_diffs, rms_norm_b_diffs = [], []

for _trial in range(N_TRIALS_INVARIANCE):
    a = torch.randn(4, D_MODEL_INVARIANCE)

    # alpha のみを変える(b=0)
    for alpha in alpha_values:
        a_scaled = alpha * a
        layer_norm_alpha_diffs.append(
            (layer_norm_invariance(a_scaled) - layer_norm_invariance(a)).abs().max().item()
        )
        rms_norm_alpha_diffs.append(
            (rms_norm_invariance(a_scaled) - rms_norm_invariance(a)).abs().max().item()
        )

    # b のみを変える(alpha=1)
    for b in b_values:
        a_shifted = a + b
        layer_norm_b_diffs.append(
            (layer_norm_invariance(a_shifted) - layer_norm_invariance(a)).abs().max().item()
        )
        rms_norm_b_diffs.append(
            (rms_norm_invariance(a_shifted) - rms_norm_invariance(a)).abs().max().item()
        )

print(
    f"層正規化: alpha を変えたときの出力差(最大絶対値、"
    f"{N_TRIALS_INVARIANCE * len(alpha_values)} 件の最大): {max(layer_norm_alpha_diffs):.3e}"
)
print(
    f"層正規化: b を変えたときの出力差(最大絶対値、"
    f"{N_TRIALS_INVARIANCE * len(b_values)} 件の最大):     {max(layer_norm_b_diffs):.3e}"
)
print(
    "RMSNorm : alpha を変えたときの出力差(最大絶対値):                          "
    f"{max(rms_norm_alpha_diffs):.3e}"
)
print(
    "RMSNorm : b を変えたときの出力差(最大絶対値):                              "
    f"{max(rms_norm_b_diffs):.3e}"
)
```

    層正規化: alpha を変えたときの出力差(最大絶対値、32 件の最大): 2.092e-03
    層正規化: b を変えたときの出力差(最大絶対値、32 件の最大):     8.345e-07
    RMSNorm : alpha を変えたときの出力差(最大絶対値):                          2.086e-03
    RMSNorm : b を変えたときの出力差(最大絶対値):                              4.023e+00



```python
# RMSNorm と LayerNormalization(center=False, scale=True) の相互検証(初期化直後 beta=0)
torch.manual_seed(SEED)
cross_check_diffs = []
layer_norm_scale_only = LayerNormalization(D_MODEL_INVARIANCE, center=False, scale=True)
for _trial in range(N_TRIALS_INVARIANCE):
    a = torch.randn(4, D_MODEL_INVARIANCE) * 3.0 + 1.5  # 平均が 0 でないランダム入力
    diff = (layer_norm_scale_only(a) - rms_norm_invariance(a)).abs().max().item()
    cross_check_diffs.append(diff)
print(
    "LayerNormalization(center=False, scale=True) と RMSNorm の出力差"
    f"(最大絶対値): {max(cross_check_diffs):.3e}"
)
```

    LayerNormalization(center=False, scale=True) と RMSNorm の出力差(最大絶対値): 0.000e+00


#### 実験 B の結果・考察

不変性の検証は 3.3 節の理論的な予測と一致した。層正規化は $\alpha$ を変えたときの出力差(最大絶対値)が $2.092 \times 10^{-3}$、$b$ を変えたときの出力差は $8.345 \times 10^{-7}$ であり、$b$ に対してはほぼ完全に不変(数値誤差の水準)、$\alpha$ に対してもごく小さい残差しか出なかった(この $\alpha$ 側の残差は数値安定化のための $\varepsilon$ に起因し、3.3 節で置いた理想化された極限 $\varepsilon \to 0$ からのずれである)。一方 RMSNorm は $\alpha$ を変えたときの出力差が $2.086 \times 10^{-3}$ と層正規化の $\alpha$ 側の残差とほぼ同水準(同じ $\varepsilon$ に由来する誤差)だったのに対し、$b$ を変えたときの出力差は $4.023$ と **3 桁以上大きい**。これは、3.3 節で導出した「RMSNorm は平行移動不変性を失う」という結論を、$\varepsilon$ に由来する数値誤差と明確に区別できる規模で裏づけている。また、初期化直後($\beta=0$)における`LayerNormalization(center=False, scale=True)`と`RMSNorm`の出力差は $0$(完全一致)であり、3.2 節で述べた関係、および 2 つの実装の相互検証が成立していることを確認した。



## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/01_foundations/004_normalization_and_activation.ipynb
