---
title: "Transformer Block(実装・実験編 1/2)"
---

この記事は後編(実装・実験編 1/2)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/002_transformer_block-theory)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/002_transformer_block-practice-2)。

## 4. 実装方針 / Implementation Plan

**`src/`に切り出すもの**(以降のトピック・アプリからも再利用する汎用部品):

| ファイル | 内容 |
|---|---|
| `src/layers/normalization.py` | `LayerNormalization`(`nn.Module`) |
| `src/layers/feedforward.py` | `FeedForwardNetwork`(`nn.Module`) |
| `src/layers/positional_encoding.py` | `SinusoidalPositionalEncoding`(`nn.Module`) |
| `src/layers/transformer_block.py` | `EncoderBlock`、`DecoderBlock`(いずれも`norm_first: bool`で正規化前置・正規化後置を切り替え) |

- 001 で実装済みの`src/layers/attention.py`の`MultiHeadAttention`、`create_causal_mask`、`create_padding_mask`を **そのまま再利用** する。API 変更は行っていない(001 のノートブックへの影響はない)。
- `LayerNormalization`は`nn.LayerNorm`を使わずスクラッチ実装するが、5.2 節で`nn.LayerNorm`との数値一致を確認する。
- `EncoderBlock` / `DecoderBlock`は、正規化前置(Pre-Layer Normalization)をデフォルト(`norm_first=True`)とする。現代の大規模言語モデルの多くが正規化前置を採用しており(3.5 節の議論の通り深い層で学習が安定するため)、以降のトピックでもこちらを標準として使う想定のため。

**ノートブックに直接書くもの**(この実験に固有で再利用しないもの):

- 実験 1・実験 3 で使う induction task(2 ホップの参照を要する copy task の変種)のデータ生成・モデル・学習ループ
- 実験 2 で使う勾配伝播計測専用のブロック(`DepthProbeBlock`)と計測関数
- 実験 4 で使う Encoder-Decoder copy task のデータ生成・モデル・学習ループ

学習ループは 001 と同様、各実験に固有の設定(データ生成・モデル定義)と一体になっているため`src/training/`へは切り出さず、ノートブック内の関数として定義する(001 の`train_copy_model`と同じ方針)。

**方針**:

- 理論の本質部分(層正規化の平均・分散計算、順伝播ネットワークの線形変換、残差接続、正規化前置・正規化後置の切り替え)は **すべてスクラッチ実装** する。
- 図中のラベル・タイトルは、Colab に日本語フォントが無く文字化けするため **英語** で書く(説明は Markdown セルで日本語)。

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

    repository root: /Users/koji/projects/ai-theories



```python
import inspect
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src.layers import (
    DecoderBlock,
    EncoderBlock,
    FeedForwardNetwork,
    LayerNormalization,
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    create_causal_mask,
)
from src.utils import (
    plot_attention_heatmap,
    plot_learning_curves,
    plot_multi_head_attention,
)

# 再現性のためのシード固定
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch: {torch.__version__} / device: {device}")
```

    torch: 2.13.0 / device: cpu


### 5.2 層正規化(Layer Normalization)の実装確認

`src/layers/normalization.py`の実装を確認し、`torch.nn.LayerNorm`と数値が一致することを確認する。


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
            mean = x.mean(dim=-1, keepdim=True)
            # unbiased=False: 分散は 1/d_model の標本分散を使う(nn.LayerNorm と同じ規約)。
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            x_hat = (x - mean) / torch.sqrt(var + self.eps)
            return self.gamma * x_hat + self.beta
    



```python
torch.manual_seed(SEED)

D_MODEL_CHECK = 64
x_ln = torch.randn(4, 10, D_MODEL_CHECK) * 3.0 + 1.0  # 平均・分散を意図的に偏らせる

ln_custom = LayerNormalization(D_MODEL_CHECK)
ln_ref = nn.LayerNorm(D_MODEL_CHECK)
# パラメータを合わせて比較する(初期値はどちらも gamma=1, beta=0 だが念のため揃える)
with torch.no_grad():
    ln_ref.weight.copy_(ln_custom.gamma)
    ln_ref.bias.copy_(ln_custom.beta)

out_custom = ln_custom(x_ln)
out_ref = ln_ref(x_ln)
max_diff = (out_custom - out_ref).abs().max().item()
print(f"LayerNormalization vs nn.LayerNorm 最大差: {max_diff:.3e}")

# 正規化後、各位置の特徴次元方向の平均・分散が 0 / 1 に近いことも確認する
mean_after = out_custom.mean(dim=-1)
var_after = out_custom.var(dim=-1, unbiased=False)
print(f"正規化後の平均(絶対値の最大): {mean_after.abs().max().item():.3e}")
print(f"正規化後の分散(1.0 からの最大差): {(var_after - 1.0).abs().max().item():.3e}")
```

    LayerNormalization vs nn.LayerNorm 最大差: 4.768e-07
    正規化後の平均(絶対値の最大): 6.706e-08
    正規化後の分散(1.0 からの最大差): 1.669e-06


### 5.3 順伝播ネットワーク(Feed-Forward Network)の実装確認


```python
print(inspect.getsource(FeedForwardNetwork.forward))
```

        def forward(self, x: Tensor) -> Tensor:
            """順伝播。
    
            Args:
                x: 形状 ``(..., d_model)`` の入力。
    
            Returns:
                x と同じ形状 ``(..., d_model)`` の出力。
            """
            return self.linear2(self.dropout(self.activation(self.linear1(x))))
    



```python
D_MODEL_FF, D_FF_DEMO = 64, 256  # d_ff = 4 * d_model(原論文の設定)
ffn_demo = FeedForwardNetwork(D_MODEL_FF, D_FF_DEMO)
x_ff = torch.randn(2, 5, D_MODEL_FF)
out_ff = ffn_demo(x_ff)
print(f"入力形状: {tuple(x_ff.shape)} / 出力形状: {tuple(out_ff.shape)}")

n_params_ff = sum(p.numel() for p in ffn_demo.parameters())
expected = 2 * D_MODEL_FF * D_FF_DEMO + D_FF_DEMO + D_MODEL_FF
print(f"FeedForwardNetwork パラメータ数: {n_params_ff:,}(理論値 {expected:,})")

n_params_mha = sum(p.numel() for p in MultiHeadAttention(D_MODEL_FF, 4).parameters())
print(f"MultiHeadAttention パラメータ数(バイアスなし): {n_params_mha:,}")
print(f"比率(FeedForwardNetwork / MultiHeadAttention): {n_params_ff / n_params_mha:.2f}")
```

    入力形状: (2, 5, 64) / 出力形状: (2, 5, 64)
    FeedForwardNetwork パラメータ数: 33,088(理論値 33,088)
    MultiHeadAttention パラメータ数(bias なし): 16,384
    比率(FeedForwardNetwork / MultiHeadAttention): 2.02


### 5.4 正弦波位置エンコーディングの実装確認


```python
print(inspect.getsource(SinusoidalPositionalEncoding.__init__))
```

        def __init__(self, d_model: int, max_len: int = 5000) -> None:
            super().__init__()
    
            position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
            # 10000^(2i/d_model) を exp/log で計算(オーバーフロー回避のため対数空間で計算)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
            )  # (d_model / 2,)
    
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
    
            # 学習パラメータではないが device / dtype をモデルと合わせたいので buffer として登録する。
            # (1, max_len, d_model)
            self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
    



```python
D_MODEL_POS, MAX_LEN_POS = 64, 50
pos_enc_demo = SinusoidalPositionalEncoding(D_MODEL_POS, max_len=MAX_LEN_POS)
pos_enc_matrix = pos_enc_demo.pe[0]  # (max_len, d_model)
print(f"位置エンコーディング行列の形状: {tuple(pos_enc_matrix.shape)}")

fig, ax = plt.subplots(figsize=(7.0, 4.5))
plot_attention_heatmap(
    pos_enc_matrix,
    title="Sinusoidal positional encoding PE(pos, dim)",
    ax=ax,
    cmap="RdBu",
    colorbar=True,
    vmin=-1.0,
    vmax=1.0,
)
ax.set_xlabel("Dimension i")
ax.set_ylabel("Position (pos)")
fig.tight_layout()
plt.show()

# 加算方式の確認: 埋め込みに加えても形状は変わらない
x_zero = torch.zeros(1, 12, D_MODEL_POS)
out_pos_enc = pos_enc_demo(x_zero)
print(
    f"\n加算後の形状: {tuple(out_pos_enc.shape)}"
    "(埋め込みがゼロなので出力はそのまま位置エンコーディング)"
)
print(f"位置 0 のベクトルのノルム: {out_pos_enc[0, 0].norm().item():.4f}")
print(f"位置 1 のベクトルのノルム: {out_pos_enc[0, 1].norm().item():.4f}")
```

    位置エンコーディング行列の形状: (50, 64)



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/002_transformer_block/output_23_1.png)
    


    
    加算後の形状: (1, 12, 64)(埋め込みがゼロなので出力はそのまま位置エンコーディング)
    位置 0 のベクトルのノルム: 5.6569
    位置 1 のベクトルのノルム: 5.6569


**確認できたこと**: ヒートマップでは、次元インデックスが小さい(左側の)列ほど位置方向に細かく振動し(高周波)、次元インデックスが大きい(右側の)列ほどゆっくり変化する(低周波)ことが分かる。これは 3.6 節で述べた「次元ペアごとに周波数 $1/10000^{2i/d_{\text{model}}}$ が異なる」という定義通りの挙動である。

### 5.5 Encoder Block / Decoder Block の実装確認

`norm_first`の切り替え、および Decoder Block の交差注意で $S_q \ne S_k$(Query 側と Key/Value 側で系列長が異なる)場合の形状を確認する。


```python
print(inspect.getsource(EncoderBlock.forward))
```

        def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
            """Encoder Block の順伝播。
    
            Args:
                x: 形状 ``(B, S, d_model)`` の入力。
                mask: 自己注意に渡すマスク(省略可)。パディングマスクなどを想定する。
    
            Returns:
                (output, attn_weights) のタプル。
                output は ``(B, S, d_model)``、attn_weights は ``(B, h, S, S)``。
            """
            if self.norm_first:
                normed = self.norm1(x)
                attn_out, attn_weights = self.self_attn(normed, normed, normed, mask)
                x = x + self.dropout1(attn_out)
                x = x + self.dropout2(self.feed_forward(self.norm2(x)))
            else:
                attn_out, attn_weights = self.self_attn(x, x, x, mask)
                x = self.norm1(x + self.dropout1(attn_out))
                x = self.norm2(x + self.dropout2(self.feed_forward(x)))
            return x, attn_weights
    



```python
print(inspect.getsource(DecoderBlock.forward))
```

        def forward(
            self,
            x: Tensor,
            memory: Tensor,
            tgt_mask: Tensor | None = None,
            memory_mask: Tensor | None = None,
        ) -> tuple[Tensor, Tensor, Tensor]:
            """Decoder Block の順伝播。
    
            Args:
                x: 形状 ``(B, S_tgt, d_model)`` の Decoder 側入力。
                memory: 形状 ``(B, S_src, d_model)`` の Encoder 出力。
                    交差注意の Key / Value になる。
                tgt_mask: 自己注意に渡すマスク。自己回帰生成では因果マスクを渡す。
                memory_mask: 交差注意に渡すマスク(Encoder 側のパディングマスクなど)。
    
            Returns:
                (output, self_attn_weights, cross_attn_weights) のタプル。
                output は ``(B, S_tgt, d_model)``、
                self_attn_weights は ``(B, h, S_tgt, S_tgt)``、
                cross_attn_weights は ``(B, h, S_tgt, S_src)``。
            """
            if self.norm_first:
                normed = self.norm1(x)
                self_out, self_attn_weights = self.self_attn(normed, normed, normed, tgt_mask)
                x = x + self.dropout1(self_out)
    
                normed = self.norm2(x)
                cross_out, cross_attn_weights = self.cross_attn(normed, memory, memory, memory_mask)
                x = x + self.dropout2(cross_out)
    
                x = x + self.dropout3(self.feed_forward(self.norm3(x)))
            else:
                self_out, self_attn_weights = self.self_attn(x, x, x, tgt_mask)
                x = self.norm1(x + self.dropout1(self_out))
    
                cross_out, cross_attn_weights = self.cross_attn(x, memory, memory, memory_mask)
                x = self.norm2(x + self.dropout2(cross_out))
    
                x = self.norm3(x + self.dropout3(self.feed_forward(x)))
            return x, self_attn_weights, cross_attn_weights
    



```python
torch.manual_seed(SEED)

B_CHK, S_SRC, S_TGT, D_MODEL_CHK, H_CHK, D_FF_CHK = 2, 7, 5, 32, 4, 128

x_enc = torch.randn(B_CHK, S_SRC, D_MODEL_CHK)
x_dec = torch.randn(B_CHK, S_TGT, D_MODEL_CHK)

for norm_first in (True, False):
    label = "正規化前置" if norm_first else "正規化後置"
    encoder_block = EncoderBlock(D_MODEL_CHK, H_CHK, D_FF_CHK, norm_first=norm_first)
    enc_out, enc_attn = encoder_block(x_enc)
    assert enc_out.shape == x_enc.shape
    assert enc_attn.shape == (B_CHK, H_CHK, S_SRC, S_SRC)

    decoder_block = DecoderBlock(D_MODEL_CHK, H_CHK, D_FF_CHK, norm_first=norm_first)
    causal = create_causal_mask(S_TGT)
    dec_out, self_attn, cross_attn = decoder_block(x_dec, enc_out, tgt_mask=causal)
    assert dec_out.shape == x_dec.shape
    assert self_attn.shape == (B_CHK, H_CHK, S_TGT, S_TGT)
    assert cross_attn.shape == (B_CHK, H_CHK, S_TGT, S_SRC)  # Query 側 S_TGT, Key/Value 側 S_SRC

    print(
        f"[{label}] Encoder out: {tuple(enc_out.shape)} / "
        f"Decoder out: {tuple(dec_out.shape)} / "
        f"cross_attn: {tuple(cross_attn.shape)}(= B, h, S_tgt, S_src)"
    )

print("\nすべての形状アサーションを通過した。")
```

    [正規化前置] Encoder out: (2, 7, 32) / Decoder out: (2, 5, 32) / cross_attn: (2, 4, 5, 7)(= B, h, S_tgt, S_src)
    [正規化後置] Encoder out: (2, 7, 32) / Decoder out: (2, 5, 32) / cross_attn: (2, 4, 5, 7)(= B, h, S_tgt, S_src)
    
    すべての形状アサーションを通過した。


交差注意の重み`cross_attn`の形状が`(B, h, S_tgt, S_src)`になっている通り、Query 側の系列長($S_{\text{tgt}}=5$)と Key/Value 側の系列長($S_{\text{src}}=7$)が異なっていても Decoder Block は正しく動作する。これは 001 の`MultiHeadAttention`が最初から $S_q \ne S_k$ に対応する設計だったためで、Decoder Block の実装で変更は不要だった。



## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/01_foundations/002_transformer_block.ipynb
