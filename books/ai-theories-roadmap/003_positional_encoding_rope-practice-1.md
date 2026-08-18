---
title: "位置エンコーディング(Positional Encoding)/ RoPE(実装・実験編 1/3)"
---

この記事は後編(実装・実験編 1/3)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/003_positional_encoding_rope-theory)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/003_positional_encoding_rope-practice-2)。

## 4. 実装方針 / Implementation Plan

### `src/`に切り出すもの

`src/layers/positional_encoding.py`に、注入点に応じた 2 種類の抽象基底クラスと、各方式の実装クラスを追加する(`SinusoidalPositionalEncoding`はそのまま残す)。

| クラス | 役割 | 注入点(3.2 節) |
|---|---|---|
| `QueryKeyPositionalTransform`(抽象基底クラス) | `apply(query, key, positions=None) -> (query, key)` | Query・Key の変換 |
| `RotaryPositionEmbedding` | 上記の実装。RoPE(3.8 節) | 同上 |
| `AttentionScoreBias`(抽象基底クラス) | `bias(query_length, key_length, device, dtype) -> Tensor` | スコアへの加算 |
| `T5RelativePositionBias` | 上記の実装。T5(3.6 節) | 同上 |
| `ALiBiPositionBias` | 上記の実装。ALiBi(3.7 節) | 同上 |
| `ShawRelativePositionBias` | `AttentionScoreBias` を継承するが、`bias()` ではなく `relative_vectors()` を主に使う(下記の設計判断を参照) | 同上(ただし特別扱い) |
| `LearnedAbsolutePositionalEmbedding` | `forward(x, positions=None) -> x` | 入力埋め込みへの加算 |

`positions`引数はいずれも、指定しない場合は $0$ から系列長 $-1$ までの連番として扱う任意引数である。これを外部から渡せるようにしている理由は、KV キャッシュを用いた逐次推論(トピック 010)では、生成の各ステップで Query の絶対位置がキャッシュ長だけずれるため、位置インデックスを明示的に指定できる必要があるからである。

**`ShawRelativePositionBias`に関する設計判断**: 3.5 節で見た通り、Shaw et al. 方式の Key 側の項 $a^K_{mn}$ は Query の内容 $q_m$ と内積を取るため、位置のみに依存する`bias(query_length, key_length, device, dtype)`という統一インターフェース(T5・ALiBi と共通)では計算できない。そこで`ShawRelativePositionBias`は`AttentionScoreBias`を継承しつつ、`bias()`は使用不可であることを示す例外を送出し、代わりに相対位置ベクトル $a^K_{mn}$ のみを返す`relative_vectors(query_length, key_length, device, dtype) -> Tensor`(形状`(S_q, S_k, d_k)`)を提供する。`MultiHeadAttention`側で`isinstance`判定を行い、このクラスの場合だけ Query との内積 $q_m \cdot a^K_{mn}$ を直接計算する特別扱いを 1 箇所だけ追加する。

`src/layers/attention.py`の`MultiHeadAttention`は、既存の引数・挙動を変えずに拡張する:

- コンストラクタに`positional_transform: QueryKeyPositionalTransform | None = None`と`attention_score_bias: AttentionScoreBias | None = None`を追加する(両方とも既定値`None`)。
- `forward`に`positions: Tensor | None = None`を追加する。
- 引数が渡されたときだけ、対応する処理を Attention 計算の該当箇所(Query・Key の線形射影の直後、スコアの softmax の直前)に挟む。
- `scaled_dot_product_attention()`にも`bias: Tensor | None = None`を追加し、スケーリング後のスコアに加算できるようにする。

**後方互換性**: これらの引数をすべて省略した場合、001・002 のノートブックは一切変更なく、同一 seed で完全に同一の数値を出力する。実際に両ノートブックを再実行して数値を比較し、5.1 節末尾で結果を報告する。

`src/layers/transformer_block.py`(`EncoderBlock`)は変更しない。実験 B では`EncoderBlock`を構築した後、その`self_attn`属性を`positional_transform`/`attention_score_bias`を指定した`MultiHeadAttention`に差し替えることで、`EncoderBlock`の残差接続・層正規化・順伝播ネットワークの構造をそのまま再利用する。

### ノートブック内に直接書くもの

- Shaw et al. 方式の **Value 側の項** $a^V_{mn}$(3.5 節)。スコアへのバイアスという枠組みに収まらず(Value の集約に直接介入するため)、`src/`のインターフェースからは意図的に外している。5.5 節で短い検証コードとして直接実装し、Key 側のみの場合との数値的な差を確認する。
- 実験 B・C 用の可変長 copy task のデータ生成、モデル定義、学習ループ、評価コード。

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
    !uv pip install --system -r requirements.txt -q
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
```


```python
import inspect
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src.layers import (
    ALiBiPositionBias,
    EncoderBlock,
    LearnedAbsolutePositionalEmbedding,
    MultiHeadAttention,
    RotaryPositionEmbedding,
    ShawRelativePositionBias,
    SinusoidalPositionalEncoding,
    T5RelativePositionBias,
    create_causal_mask,
)
from src.utils.visualization import plot_attention_heatmap

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}")
```

    device: cpu


**001・002 の後方互換性の確認について**:`MultiHeadAttention`に`positional_transform`/`attention_score_bias`/`positions`を追加するにあたり、これらをすべて省略した場合に 001・002 が変更前と完全に同一の数値を出力することを、本ノートブックの作成過程で別途確認済みである(`jupyter nbconvert --execute`で両ノートブックを変更前後で再実行し、損失・精度・パラメータ数などの全出力を比較した。壁時計時間の実測値のみ実行環境依存で異なり、それ以外の数値はすべて一致した)。作業報告にもこの確認結果を明記する。

### 5.2 RoPE(Rotary Position Embedding)の実装確認


```python
print(inspect.getsource(RotaryPositionEmbedding.apply))
```

        def apply(
            self, query: Tensor, key: Tensor, positions: Tensor | None = None
        ) -> tuple[Tensor, Tensor]:
            seq_len = query.size(-2)
            if positions is None:
                positions = torch.arange(seq_len, device=query.device)
    
            max_pos = int(positions.max().item()) + 1
            if max_pos > self.max_position:
                self._build_cache(max_pos)
    
            cos = self.cos_cached[positions].to(dtype=query.dtype, device=query.device)
            sin = self.sin_cached[positions].to(dtype=query.dtype, device=query.device)
    
            q = query * cos + self._rotate_half(query) * sin
            k = key * cos + self._rotate_half(key) * sin
            return q, k
    



```python
D_K_DEMO = 8
rope_demo = RotaryPositionEmbedding(D_K_DEMO, max_position=32)
print(f"cos_cached の形状: {tuple(rope_demo.cos_cached.shape)}")

q_demo = torch.randn(1, 1, 5, D_K_DEMO)
k_demo = torch.randn(1, 1, 5, D_K_DEMO)
q_rot, k_rot = rope_demo.apply(q_demo, k_demo)
print(f"適用後の形状: query={tuple(q_rot.shape)}, key={tuple(k_rot.shape)}")
print(
    f"位置 0 では回転角が 0 なので、適用前後で一致する: "
    f"{torch.allclose(q_demo[:, :, 0], q_rot[:, :, 0])}"
)
```

    cos_cached の形状: (32, 8)
    適用後の形状: query=(1, 1, 5, 8), key=(1, 1, 5, 8)
    位置 0 では回転角が 0 なので、適用前後で一致する: True


### 5.3 相対位置バイアス(Shaw et al. 方式・T5・ALiBi)の実装確認


```python
print(inspect.getsource(ALiBiPositionBias._compute_slopes))
```

        @staticmethod
        def _compute_slopes(num_heads: int) -> list[float]:
            """m_h = 2^(-8h/H) の幾何数列を計算する。
    
            H が 2 のべき乗でない場合は、原論文の公式実装に従い、直近の 2 のべき乗
            で計算した数列を補間して残りのヘッド分を埋める。
            """
    
            def _power_of_2_slopes(n: int) -> list[float]:
                start = 2.0 ** (-8.0 / n)
                return [start ** (h + 1) for h in range(n)]
    
            if math.log2(num_heads).is_integer():
                return _power_of_2_slopes(num_heads)
    
            closest_pow2 = 2 ** math.floor(math.log2(num_heads))
            base_slopes = _power_of_2_slopes(closest_pow2)
            extra_slopes = _power_of_2_slopes(2 * closest_pow2)[0::2][: num_heads - closest_pow2]
            return base_slopes + extra_slopes
    



```python
for h in (4, 8):
    slopes = ALiBiPositionBias(h).slopes
    print(f"H={h}: slopes = {[round(s, 5) for s in slopes.tolist()]}")
```

    H=4: slopes = [0.25, 0.0625, 0.01562, 0.00391]
    H=8: slopes = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.01562, 0.00781, 0.00391]



```python
torch.manual_seed(SEED)
t5_demo = T5RelativePositionBias(num_heads=4, num_buckets=16, max_distance=32, bidirectional=False)
bias_demo = t5_demo.bias(query_length=6, key_length=6, device="cpu", dtype=torch.float32)
print(f"T5 相対位置バイアスの形状: {tuple(bias_demo.shape)}")
print("head=0 のバイアス行列(下三角、因果的なので n <= m のみ意味を持つ):")
print(bias_demo[0].detach().numpy().round(3))
```

    T5 相対位置バイアスの形状: (4, 6, 6)
    head=0 のバイアス行列(下三角、因果的なので n <= m のみ意味を持つ):
    [[-1.126 -1.126 -1.126 -1.126 -1.126 -1.126]
     [ 0.849 -1.126 -1.126 -1.126 -1.126 -1.126]
     [ 0.322  0.849 -1.126 -1.126 -1.126 -1.126]
     [ 0.12   0.322  0.849 -1.126 -1.126 -1.126]
     [-1.353  0.12   0.322  0.849 -1.126 -1.126]
     [ 0.599 -1.353  0.12   0.322  0.849 -1.126]]


因果的(`bidirectional=False`)なバケット化のため、$n > m$(未来を参照する側)はすべて同じバケット(バケット 0)に潰れており、値も同一になっている。これは因果マスクによってどのみち参照されない領域なので実害はない。

### 5.4 位置エンコーディングを統合した`MultiHeadAttention`の確認


```python
print(inspect.getsource(MultiHeadAttention.forward))
```

        def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            mask: Tensor | None = None,
            positions: Tensor | None = None,
        ) -> tuple[Tensor, Tensor]:
            """Multi-Head Attention の順伝播。
    
            Args:
                query: 形状 ``(B, S_q, d_model)``。
                key: 形状 ``(B, S_k, d_model)``。
                value: 形状 ``(B, S_k, d_model)``。
                    自己注意(self-attention)では query = key = value を渡す。
                mask: True が「参加させる」を表す bool マスク。
                    形状は ``(S_q, S_k)`` / ``(B, S_q, S_k)`` / ``(B, h, S_q, S_k)``。
                positions: Query 側の絶対位置インデックス(形状 ``(S_q,)``)。
                    ``positional_transform`` に渡される。``None`` のときは 0 から
                    S_q - 1 までの連番として扱う。KV キャッシュを用いた逐次推論
                    (トピック 010)では、生成の各ステップで Query の絶対位置が
                    キャッシュ長だけずれるため、これを外部から指定できるようにしている。
    
            Returns:
                (output, attn_weights) のタプル。
                output は ``(B, S_q, d_model)``、attn_weights は ``(B, h, S_q, S_k)``。
            """
            # 1. 線形射影(全ヘッド分をまとめて計算)
            q = self._split_heads(self.w_q(query))  # (B, h, S_q, d_k)
            k = self._split_heads(self.w_k(key))  # (B, h, S_k, d_k)
            v = self._split_heads(self.w_v(value))  # (B, h, S_k, d_v)
    
            # 1.5. Query・Key の位置変換(例: RoPE)。指定がなければ従来通り何もしない。
            if self.positional_transform is not None:
                q, k = self.positional_transform.apply(q, k, positions)
    
            if mask is not None:
                mask = self._expand_mask(mask)
    
            # 1.6. Attention スコアへの位置バイアス(例: Shaw et al. 方式・T5・ALiBi)。
            # 指定がなければ従来通り何も加算しない。
            score_bias = None
            if self.attention_score_bias is not None:
                s_q, s_k = q.size(-2), k.size(-2)
                if isinstance(self.attention_score_bias, ShawRelativePositionBias):
                    # a^K_mn は Query に内容依存するため、専用の relative_vectors() から
                    # 相対位置ベクトルのみを取得し、ここで Query との内積を直接計算する。
                    relative_vectors = self.attention_score_bias.relative_vectors(
                        s_q, s_k, q.device, q.dtype
                    )  # (S_q, S_k, d_k)
                    score_bias = torch.einsum("bhqd,qkd->bhqk", q, relative_vectors) / math.sqrt(
                        self.d_k
                    )
                else:
                    score_bias = self.attention_score_bias.bias(s_q, s_k, q.device, q.dtype).unsqueeze(
                        0
                    )  # (1, h, S_q, S_k) -> バッチ方向へブロードキャスト
    
            # 2. 各ヘッドで Scaled Dot-Product Attention
            head_outputs, attn_weights = scaled_dot_product_attention(
                q, k, v, mask, self.dropout, score_bias
            )
    
            # 3. ヘッドを連結して出力射影 W^O を適用
            concatenated = self._merge_heads(head_outputs)  # (B, S_q, d_model)
            output = self.w_o(concatenated)
            return output, attn_weights
    



```python
torch.manual_seed(SEED)
B_CHK, S_CHK, D_MODEL_CHK, H_CHK = 2, 6, 32, 4
x_chk = torch.randn(B_CHK, S_CHK, D_MODEL_CHK)

# 5 通りの組み合わせがすべて動作し、出力形状が変わらないことを確認する
configs = {
    "なし(既定値)": {},
    "RoPE": {"positional_transform": RotaryPositionEmbedding(D_MODEL_CHK // H_CHK)},
    "Shaw et al. 方式": {"attention_score_bias": ShawRelativePositionBias(D_MODEL_CHK // H_CHK)},
    "T5": {"attention_score_bias": T5RelativePositionBias(H_CHK)},
    "ALiBi": {"attention_score_bias": ALiBiPositionBias(H_CHK)},
}
for name, kwargs in configs.items():
    mha_chk = MultiHeadAttention(D_MODEL_CHK, H_CHK, **kwargs)
    out_chk, attn_chk = mha_chk(x_chk, x_chk, x_chk, create_causal_mask(S_CHK))
    row_sums_ok = torch.allclose(attn_chk.sum(-1), torch.ones_like(attn_chk.sum(-1)), atol=1e-5)
    print(
        f"{name:16s}: output={tuple(out_chk.shape)}  attn={tuple(attn_chk.shape)}"
        f"  attn 各行の和 ≈ 1: {row_sums_ok}"
    )
```

    なし(既定値)         : output=(2, 6, 32)  attn=(2, 4, 6, 6)  attn 各行の和 ≈ 1: True
    RoPE            : output=(2, 6, 32)  attn=(2, 4, 6, 6)  attn 各行の和 ≈ 1: True
    Shaw et al. 方式  : output=(2, 6, 32)  attn=(2, 4, 6, 6)  attn 各行の和 ≈ 1: True
    T5              : output=(2, 6, 32)  attn=(2, 4, 6, 6)  attn 各行の和 ≈ 1: True
    ALiBi           : output=(2, 6, 32)  attn=(2, 4, 6, 6)  attn 各行の和 ≈ 1: True


### 5.5 Shaw et al. 方式の Value 側の項(ノートブック内実装)

3.5 節で述べた通り、Value 側の項 $a^V_{mn}$ は Attention 重みによる Value の集約 $z_m = \sum_n a_{mn} (v_n + a^V_{mn})$ に直接介入するため、`src/`の`AttentionScoreBias`インターフェースには収まらない。ここでは小規模な数値例で、Key 側のみの場合との差を確認する。


```python
torch.manual_seed(SEED)
S_V, D_K_V = 5, 4
k_clip_v = 2

value_v = torch.randn(S_V, D_K_V)
attn_weights_v = torch.softmax(torch.randn(S_V, S_V), dim=-1)  # 適当な Attention 重み(各行の和が 1)
a_v = torch.randn(2 * k_clip_v + 1, D_K_V) * 0.1  # 相対位置ごとの Value 側ベクトル w^V

m_idx = torch.arange(S_V)[:, None]
n_idx = torch.arange(S_V)[None, :]
rel_v = torch.clamp(n_idx - m_idx, -k_clip_v, k_clip_v) + k_clip_v  # (S_V, S_V)
a_v_mn = a_v[rel_v]  # (S_V, S_V, D_K_V) = a^V_{mn}

# Key 側のみ(通常の Attention): z_m = sum_n a_mn * v_n
z_key_only = attn_weights_v @ value_v

# Key 側 + Value 側: z_m = sum_n a_mn * (v_n + a^V_mn)
z_with_value_term = z_key_only + torch.einsum("mn,mnd->md", attn_weights_v, a_v_mn)

diff = (z_with_value_term - z_key_only).norm(dim=-1)
print("Value 側の項を加えた場合と加えない場合の出力の差(位置ごとの L2 ノルム):")
print(diff.detach().numpy().round(4))
print(f"\n相対差(平均): {(diff / z_key_only.norm(dim=-1)).mean().item():.4f}")
```

    Value 側の項を加えた場合と加えない場合の出力の差(位置ごとの L2 ノルム):
    [0.113  0.1152 0.0612 0.0606 0.123 ]
    
    相対差(平均): 0.0854


**確認できたこと**: Value 側の項を加えると、出力ベクトルは Key 側のみの場合から無視できない差(相対差にして数 % 〜 十数 %程度)を持つ。原論文が Value 側の項を持つ定式化であること自体は数式として無視できないが、後続研究でこの項が省略されることが多いのは、(a) この項がスコアへのバイアスという単純な枠組みに収まらず実装が煩雑になること、(b) 経験的にこの項を省略しても性能への影響が小さいことの両方が理由とされる。本ノートブックの実験 B では、`src/`の設計判断に合わせて Key 側のみの`ShawRelativePositionBias`を使用する。

## 6. 実験 / Experiments

> **実行環境について**: 以下の実験で報告する実測値は、本ノートブックを CPU 上で実行して得たものである(GPU は使用していない)。損失・精度は乱数シードを固定しているため同一環境では再現するはずだが、学習時間などハードウェアに強く依存する数値は実行環境によって変わる。Google Colab の T4 GPU で実行する場合、学習時間はここでの CPU の値とは一致しない。

### 6.1 実験 A: RoPE の理論的性質の数値検証

学習を伴わない検証。3.8 節で導出した性質が数値的に成り立つことを確認する。


```python
torch.manual_seed(SEED)
D_K_A, MAX_POS_A = 16, 64
rope_a = RotaryPositionEmbedding(D_K_A, max_position=MAX_POS_A)


def block_diag_rotation_half_split(position: int, inv_freq: torch.Tensor) -> torch.Tensor:
    """3.8.7 節の rotate_half に対応するペアの取り方(i 番目と i+half 番目)で
    ブロック対角の回転行列を明示的に構築する(検証専用)。"""
    d_k = inv_freq.numel() * 2
    r = torch.zeros(d_k, d_k)
    half = d_k // 2
    for i, theta_i in enumerate(inv_freq.tolist()):
        angle = position * theta_i
        c, s = math.cos(angle), math.sin(angle)
        r[i, i] = c
        r[i, i + half] = -s
        r[i + half, i] = s
        r[i + half, i + half] = c
    return r


def block_diag_rotation_interleaved(position: int, inv_freq: torch.Tensor) -> torch.Tensor:
    """3.8.4 節の定義通り、2i 番目と 2i+1 番目を組にするペアの取り方(原論文の
    隣接ペア)でブロック対角の回転行列 R_{Theta,m} を明示的に構築する(検証専用)。"""
    d_k = inv_freq.numel() * 2
    r = torch.zeros(d_k, d_k)
    for i, theta_i in enumerate(inv_freq.tolist()):
        angle = position * theta_i
        c, s = math.cos(angle), math.sin(angle)
        r[2 * i, 2 * i] = c
        r[2 * i, 2 * i + 1] = -s
        r[2 * i + 1, 2 * i] = s
        r[2 * i + 1, 2 * i + 1] = c
    return r


inv_freq_a = rope_a.inv_freq
print(f"theta_i (i=0..{D_K_A // 2 - 1}): {inv_freq_a.tolist()}")
```

    theta_i (i=0..7): [1.0, 0.3162277638912201, 0.10000000149011612, 0.03162277489900589, 0.009999999776482582, 0.003162277629598975, 0.0010000000474974513, 0.0003162277571391314]


**検証 A-1**: $R_{\Theta,m}^{\top} R_{\Theta,n} = R_{\Theta,\, n-m}$ が機械精度で成り立つことを確認する。


```python
m_a1, n_a1 = 5, 13
r_m = block_diag_rotation_half_split(m_a1, inv_freq_a)
r_n = block_diag_rotation_half_split(n_a1, inv_freq_a)
r_diff = block_diag_rotation_half_split(n_a1 - m_a1, inv_freq_a)

lhs_a1 = r_m.T @ r_n
max_err_a1 = (lhs_a1 - r_diff).abs().max().item()
print(f"R_m^T R_n vs R_(n-m) の最大要素誤差: {max_err_a1:.3e}")
print(f"機械精度で一致: {torch.allclose(lhs_a1, r_diff, atol=1e-5)}")
```

    R_m^T R_n vs R_(n-m) の最大要素誤差: 5.960e-08
    機械精度で一致: True


**検証 A-2**: 回転を適用した Query・Key の内積が、絶対位置をずらしても相対位置が同じであれば不変であることを確認する(例: $(m,n)=(3,7)$ と $(10,14)$ で内積が一致する、いずれも相対位置は $n-m=4$)。


```python
torch.manual_seed(SEED)
q_a2 = torch.randn(1, 1, 1, D_K_A)
k_a2 = torch.randn(1, 1, 1, D_K_A)

pairs_a2 = [(3, 7), (10, 14), (0, 4), (50, 54)]
dots_a2 = []
for m_pos, n_pos in pairs_a2:
    q_rot, _ = rope_a.apply(q_a2, q_a2, positions=torch.tensor([m_pos]))
    _, k_rot = rope_a.apply(k_a2, k_a2, positions=torch.tensor([n_pos]))
    dot = (q_rot * k_rot).sum().item()
    dots_a2.append(dot)
    print(f"(m, n) = ({m_pos:2d}, {n_pos:2d})  相対位置 n-m = {n_pos - m_pos}  内積 = {dot:.6f}")

max_diff_a2 = max(dots_a2) - min(dots_a2)
print(f"\n相対位置が同じ 4 組の内積の最大差: {max_diff_a2:.3e}(機械精度なら 0 に近い)")
```

    (m, n) = ( 3,  7)  相対位置 n-m = 4  内積 = -0.693269
    (m, n) = (10, 14)  相対位置 n-m = 4  内積 = -0.693269
    (m, n) = ( 0,  4)  相対位置 n-m = 4  内積 = -0.693269
    (m, n) = (50, 54)  相対位置 n-m = 4  内積 = -0.693270
    
    相対位置が同じ 4 組の内積の最大差: 1.192e-06(機械精度なら 0 に近い)


**検証 A-3a**: ブロック対角行列との明示的な行列積による実装(`block_diag_rotation_half_split`、3.8.7 節の rotate_half に対応するペアの取り方)と、`rotate_half`+ 要素ごとの積による効率的な実装(`RotaryPositionEmbedding.apply`)が機械精度で一致することを確認する。


```python
torch.manual_seed(SEED)
q_a3 = torch.randn(D_K_A)
m_a3 = 17

r_a3 = block_diag_rotation_half_split(m_a3, inv_freq_a)
q_explicit_a3 = r_a3 @ q_a3

q_batched_a3 = q_a3.view(1, 1, 1, D_K_A)
q_efficient_a3, _ = rope_a.apply(q_batched_a3, q_batched_a3, positions=torch.tensor([m_a3]))
q_efficient_a3 = q_efficient_a3.view(D_K_A)

max_err_a3a = (q_explicit_a3 - q_efficient_a3).abs().max().item()
print(f"明示的な行列積 vs rotate_half 実装の最大要素誤差: {max_err_a3a:.3e}")
print(f"機械精度で一致: {torch.allclose(q_explicit_a3, q_efficient_a3, atol=1e-5)}")
```

    明示的な行列積 vs rotate_half 実装の最大要素誤差: 2.384e-07
    機械精度で一致: True


**検証 A-3b**: 隣接ペア版(`block_diag_rotation_interleaved`、3.8.4 節の定義通り)と前半・後半ペア版(`block_diag_rotation_half_split`)は、次元ペアの取り方が異なる。同一の $q, k$ に対して、2 つの取り方が **一致しない** 内積を与えることを数値で確認する。


```python
torch.manual_seed(SEED)
q_a3b = torch.randn(D_K_A)
k_a3b = torch.randn(D_K_A)
m_a3b, n_a3b = 3, 7

r_m_half = block_diag_rotation_half_split(m_a3b, inv_freq_a)
r_n_half = block_diag_rotation_half_split(n_a3b, inv_freq_a)
dot_half_a3b = (r_m_half @ q_a3b) @ (r_n_half @ k_a3b)

r_m_inter = block_diag_rotation_interleaved(m_a3b, inv_freq_a)
r_n_inter = block_diag_rotation_interleaved(n_a3b, inv_freq_a)
dot_inter_a3b = (r_m_inter @ q_a3b) @ (r_n_inter @ k_a3b)

print(f"前半・後半ペア版の内積: {dot_half_a3b.item():.4f}")
print(f"隣接ペア版の内積: {dot_inter_a3b.item():.4f}")
print(f"一致する: {torch.allclose(dot_half_a3b, dot_inter_a3b, atol=1e-5)}")
```

    前半・後半ペア版の内積: -0.6933
    隣接ペア版の内積: -1.3158
    一致する: False


**検証 A-3c**: 2 つの取り方は、座標の固定置換 $P$(前半・後半ペアの添字 $i \to$ 隣接ペアの添字 $2i$、$i+d_k/2 \to 2i+1$ に対応させる置換行列、$P^{\top}=P^{-1}$)によって厳密に結ばれている。以下を確認する。

1. $R^{\text{half}}_{\Theta,m} = P^{\top} R^{\text{interleaved}}_{\Theta,m} P$ が機械精度で成り立つこと。
2. $q, k$ に同じ置換 $P$(半空間表現から隣接ペア表現への埋め込み)を適用すれば、隣接ペア版の内積が前半・後半ペア版と機械精度で一致すること(A-3b で「一致しない」ことを確認した設定に、置換だけを追加する)。


```python
half_a3c = D_K_A // 2
perm_p = torch.zeros(D_K_A, D_K_A)
for i in range(half_a3c):
    perm_p[2 * i, i] = 1.0
    perm_p[2 * i + 1, i + half_a3c] = 1.0

# (1) R^half_{Theta,m} = P^T R^interleaved_{Theta,m} P
r_half_check = block_diag_rotation_half_split(m_a3b, inv_freq_a)
r_inter_check = block_diag_rotation_interleaved(m_a3b, inv_freq_a)
lhs_a3c = perm_p.T @ r_inter_check @ perm_p
max_err_a3c1 = (lhs_a3c - r_half_check).abs().max().item()
print(f"P^T R^interleaved P vs R^half の最大要素誤差: {max_err_a3c1:.3e}")
print(f"機械精度で一致: {torch.allclose(lhs_a3c, r_half_check, atol=1e-5)}")

# (2) 同じ置換 P を q, k に適用すれば、隣接ペア版の内積が前半・後半ペア版と一致する
q_a3c_permuted = perm_p @ q_a3b
k_a3c_permuted = perm_p @ k_a3b
dot_inter_permuted_a3c = (r_m_inter @ q_a3c_permuted) @ (r_n_inter @ k_a3c_permuted)

print(f"\n前半・後半ペア版の内積(A-3b と同じ): {dot_half_a3b.item():.4f}")
print(f"置換後の隣接ペア版の内積: {dot_inter_permuted_a3c.item():.4f}")
print(f"機械精度で一致: {torch.allclose(dot_half_a3b, dot_inter_permuted_a3c, atol=1e-5)}")
```

    P^T R^interleaved P vs R^half の最大要素誤差: 0.000e+00
    機械精度で一致: True
    
    前半・後半ペア版の内積(A-3b と同じ): -0.6933
    置換後の隣接ペア版の内積: -0.6933
    機械精度で一致: True


**検証 A-4**: 遠距離減衰(long-term decay、3.8.6 節)は、同一の Query ベクトルを固定し、相対距離 $t$ だけ離れた位置における自己相関 $\langle R_{\Theta,0}\,q,\ R_{\Theta,t}\,q \rangle$ が $t$ とともに減衰する傾向として観察できる。3.8.4 節の $\theta_i$ は $i$ が大きい(低次元インデックスではなく低周波の)部分空間ほど回転が遅く、$d_k=16$・$\mathrm{base}=10000$ では最も遅い周波数が 1 周するのに $2\pi/\theta_{\max i} \approx 20000$ ステップ以上を要する。そのため、独立にサンプルした Query・Key の内積を数十〜数百程度の狭い相対距離レンジで平均しても、低周波成分がほとんど回転しないまま内積の平均値に一定のオフセットを与え続け、減衰が観測できない。ここでは相対距離のレンジを $0$〜$2000$ 程度まで広げ、かつ同一ベクトルの自己相関を見ることで、この減衰傾向を可視化する。


```python
torch.manual_seed(SEED)
N_TRIALS_A4 = 300
MAX_REL_DIST_A4 = 2000
rope_a4 = RotaryPositionEmbedding(D_K_A, max_position=MAX_REL_DIST_A4 + 1)
rel_distances_a4 = list(range(0, MAX_REL_DIST_A4, 20))

q_a4 = torch.randn(N_TRIALS_A4, 1, 1, D_K_A)
q_rot0_a4, _ = rope_a4.apply(q_a4, q_a4, positions=torch.tensor([0]))

mean_abs_dots_a4 = []
for rel_dist in rel_distances_a4:
    q_rot_t_a4, _ = rope_a4.apply(q_a4, q_a4, positions=torch.tensor([rel_dist]))
    dots_a4 = (q_rot0_a4 * q_rot_t_a4).sum(dim=-1).squeeze()
    mean_abs_dots_a4.append(dots_a4.abs().mean().item())

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(rel_distances_a4, mean_abs_dots_a4, linewidth=1.0)
ax.set_xlabel("Relative distance t")
ax.set_ylabel("Mean |<R_0 q, R_t q>| (300 random trials)")
ax.set_title("RoPE: long-term decay of the self-correlation")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"相対距離 0 での平均絶対内積: {mean_abs_dots_a4[0]:.4f}")
window = 10  # 末尾付近の移動平均で振動をならして傾向を見る
tail_mean_a4 = sum(mean_abs_dots_a4[-window:]) / window
print(f"相対距離 {rel_distances_a4[-1]} 近傍(末尾 {window} 点)の平均絶対内積: {tail_mean_a4:.4f}")
```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/003_positional_encoding_rope/output_37_0.png)
    


    相対距離 0 での平均絶対内積: 15.7103
    相対距離 1980 近傍(末尾 10 点)の平均絶対内積: 4.5793


#### 実験 A の結果・考察

- **A-1**: $R_{\Theta,m}^{\top} R_{\Theta,n}$ と $R_{\Theta,\,n-m}$ の最大要素誤差は $5.960 \times 10^{-8}$ であり、機械精度で一致した。3.8.5 節で証明したブロック対角性による関係が数値的にも成立している。
- **A-2**: 相対位置が同じ 4 組 $(m,n) \in \{(3,7), (10,14), (0,4), (50,54)\}$($n-m=4$ で共通)の内積はいずれも $-0.6933$ 付近に一致し(最大差 $1.192 \times 10^{-6}$)、3.8.1 節の要請(内積が相対位置のみに依存する)が実際に満たされていることを確認できた。
- **A-3a**: 明示的な行列積による実装(前半・後半ペア版、3.8.7 節)と`rotate_half`による効率的な実装の最大要素誤差は $2.384 \times 10^{-7}$ であり、機械精度で一致した。
- **A-3b**: 同一の $q, k$ に対し、前半・後半ペア版(3.8.7 節)の内積は $-0.6933$、隣接ペア版(3.8.4 節)の内積は $-1.3158$ であり、**一致しなかった**。次元ペアの取り方が異なれば、同一のベクトルに対する数値そのものは変わる。
- **A-3c**: 座標の固定置換 $P$ について、(1) $R^{\text{half}}_{\Theta,m} = P^{\top} R^{\text{interleaved}}_{\Theta,m} P$ の最大要素誤差は $0.000 \times 10^{0}$(完全に一致)であり、(2) $q, k$ に同じ置換 $P$ を適用すると隣接ペア版の内積は $-0.6933$ となり、A-3b の前半・後半ペア版の値($-0.6933$)と機械精度で一致することを確認した。この 2 つの取り方は座標の固定置換で厳密に移り合う関係にあり、この置換は $W^Q, W^K$ の再パラメータ化(列の並べ替え)に吸収されるため、**モデルの表現力としては等価** である(A-3b で見た「一致しない」という結果は、置換を揃えていない同一ベクトルを比較したことによるものであり、表現力の違いを意味しない)。これが 3.8.7 節で述べた「Query と Key に同一の置換を適用するため、相対位置のみへの依存という性質は保たれる」ことの数値的な裏付けになる。
- **A-4**: 相対距離 $t=0$ での平均絶対自己相関は $15.71$ であったのに対し、$t \approx 1980$〜$2000$ 近傍(末尾 10 点の平均)では $4.58$ まで減衰した。3.8.6 節で述べた通り、$d_k=16$・$\mathrm{base}=10000$ では最も低い周波数 $\theta_{7} \approx 3.16\times10^{-4}$ が 1 周するのに $2\pi/\theta_7 \approx 20000$ ステップを要するため、独立サンプルの Query・Key を数十〜数百程度の相対距離で平均しただけでは低周波成分がほとんど回転せず減衰が観測できない。相対距離のレンジを $0$〜$2000$ まで広げ、同一 Query の自己相関を見ることで、遠距離減衰の傾向を可視化できた。



## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/01_foundations/003_positional_encoding_rope.ipynb
