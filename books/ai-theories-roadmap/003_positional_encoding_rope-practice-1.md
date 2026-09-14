---
title: "位置エンコーディング(Positional Encoding)/ RoPE(実装・実験編 1/2)"
---

この記事は後編(実装・実験編 1/2)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/003_positional_encoding_rope-theory)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/003_positional_encoding_rope-practice-2)。

## 4. 実装方針 / Implementation Plan

### `src/`に切り出すもの

`src/layers/positional_encoding.py`に、注入点に応じた 2 種類の抽象基底クラスと各方式の実装クラスを追加する(`SinusoidalPositionalEncoding`はそのまま残す)。

| クラス | 役割 | 注入点(3.2 節) |
|---|---|---|
| `QueryKeyPositionalTransform`(抽象基底クラス) | `apply(query, key, positions=None) -> (query, key)` | Query・Key の変換 |
| `RotaryPositionEmbedding` | 上記の実装。RoPE(3.8 節) | 同上 |
| `AttentionScoreBias`(抽象基底クラス) | `bias(query_length, key_length, device, dtype) -> Tensor` | スコアへの加算 |
| `T5RelativePositionBias` | 上記の実装。T5(3.6 節) | 同上 |
| `ALiBiPositionBias` | 上記の実装。ALiBi(3.7 節) | 同上 |
| `ShawRelativePositionBias` | `AttentionScoreBias` を継承するが、`bias()` ではなく `relative_vectors()` を主に使う(下記の設計判断を参照) | 同上(ただし特別扱い) |
| `LearnedAbsolutePositionalEmbedding` | `forward(x, positions=None) -> x` | 入力埋め込みへの加算 |

`positions`引数は任意引数で、省略時は $0$ から系列長 $-1$ までの連番として扱う。外部から渡せるようにしている理由は、KV キャッシュを用いた逐次推論(トピック 010)では生成の各ステップで Query の絶対位置がキャッシュ長だけずれるため、位置インデックスを明示的に指定できる必要があるからである。

**`ShawRelativePositionBias`に関する設計判断**: 3.5 節の通り、Shaw et al. 方式の Key 側の項 $a^K_{mn}$ は Query の内容 $q_m$ と内積を取るため、位置のみに依存する`bias(query_length, key_length, device, dtype)`という統一インターフェース(T5・ALiBi と共通)では計算できない。そこで`ShawRelativePositionBias`は`AttentionScoreBias`を継承しつつ`bias()`は例外を送出し、代わりに相対位置ベクトル $a^K_{mn}$ のみを返す`relative_vectors(query_length, key_length, device, dtype) -> Tensor`(形状`(S_q, S_k, d_k)`)を提供し、`MultiHeadAttention`側の`isinstance`判定でこのクラスの場合だけ Query との内積 $q_m \cdot a^K_{mn}$ を直接計算する特別扱いを追加する。

`src/layers/attention.py`の`MultiHeadAttention`は、既存の引数・挙動を変えずに拡張する:

- コンストラクタに`positional_transform: QueryKeyPositionalTransform | None = None`と`attention_score_bias: AttentionScoreBias | None = None`を追加(既定値`None`)。
- `forward`に`positions: Tensor | None = None`を追加する。
- 引数が渡されたときだけ、対応する処理を Attention 計算の該当箇所(Query・Key の線形射影の直後、スコアの softmax の直前)に挟む。
- `scaled_dot_product_attention()`にも`bias: Tensor | None = None`を追加し、スケーリング後のスコアに加算できるようにする。

**後方互換性**: これらの引数をすべて省略した場合、001・002 は変更前と同一 seed で完全に同一の数値を出力する。両ノートブックを再実行して数値を比較した結果は 5.1 節末尾で報告する。

`src/layers/transformer_block.py`(`EncoderBlock`)は変更しない。実験 B では構築後に`self_attn`属性を`positional_transform`/`attention_score_bias`を指定した`MultiHeadAttention`に差し替え、残差接続・層正規化・順伝播ネットワークの構造をそのまま再利用する。

### ノートブック内に直接書くもの

- Shaw et al. 方式の **Value 側の項** $a^V_{mn}$(3.5 節)。スコアへのバイアスという枠組みに収まらず(Value の集約に直接介入するため)、`src/`のインターフェースからは意図的に外す。5.5 節で直接実装し、Key 側のみの場合との数値的な差を確認する。
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


**001・002 の後方互換性の確認について**:`MultiHeadAttention`に`positional_transform`/`attention_score_bias`/`positions`を追加するにあたり、これらをすべて省略した場合に 001・002 が変更前と完全に同一の数値を出力することを別途確認済みである(`jupyter nbconvert --execute`で両ノートブックを変更前後で再実行し、損失・精度・パラメータ数などの全出力を比較した。壁時計時間の実測値のみ実行環境依存で異なり、それ以外の数値はすべて一致した)。

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


**確認できたこと**: Value 側の項を加えると、出力ベクトルは Key 側のみの場合から無視できない差(相対差にして数 % 〜 十数 %程度)を持つ。原論文が Value 側の項を持つ定式化であること自体は無視できないが、後続研究で省略されることが多いのは、(a) スコアへのバイアスという単純な枠組みに収まらず実装が煩雑になること、(b) 経験的に性能への影響が小さいことの両方が理由とされる。本ノートブックの実験 B では、`src/`の設計判断に合わせて Key 側のみの`ShawRelativePositionBias`を使用する。

## 6. 実験 / Experiments

> **実行環境について**: 以下の実測値は本ノートブックを CPU 上で実行して得たものである(GPU 未使用)。損失・精度は乱数シードを固定しているため同一環境では再現するが、学習時間などハードウェア依存の数値は実行環境によって変わる(Google Colab の T4 GPU では一致しない)。

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


**検証 A-4**: 遠距離減衰(long-term decay、3.8.6 節)は、同一の Query ベクトルを固定し、相対距離 $t$ だけ離れた位置における自己相関 $\langle R_{\Theta,0}\,q,\ R_{\Theta,t}\,q \rangle$ が $t$ とともに減衰する傾向として観察できる。3.8.4 節の通り $\theta_i$ は $i$ が大きいほど回転が遅く、$d_k=16$・$\mathrm{base}=10000$ では最も遅い周波数が 1 周するのに $2\pi/\theta_{\max i} \approx 20000$ ステップ以上を要するため、狭い相対距離レンジで平均すると低周波成分がほとんど回転せず減衰が観測できない。ここでは相対距離のレンジを $0$〜$2000$ 程度まで広げ、同一ベクトルの自己相関を見ることで、この減衰傾向を可視化する。


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

- **A-1**: $R_{\Theta,m}^{\top} R_{\Theta,n}$ と $R_{\Theta,\,n-m}$ の最大要素誤差は $5.960 \times 10^{-8}$ で機械精度一致し、3.8.5 節の証明が数値的にも成立した。
- **A-2**: 相対位置が同じ 4 組 $(m,n) \in \{(3,7), (10,14), (0,4), (50,54)\}$($n-m=4$ で共通)の内積はいずれも $-0.6933$ 付近に一致(最大差 $1.192 \times 10^{-6}$)し、3.8.1 節の要請(内積が相対位置のみに依存する)を確認した。
- **A-3a**: 明示的な行列積による実装(前半・後半ペア版、3.8.7 節)と`rotate_half`による効率的な実装の最大要素誤差は $2.384 \times 10^{-7}$ で機械精度一致した。
- **A-3b**: 同一の $q, k$ に対し、前半・後半ペア版(3.8.7 節)の内積は $-0.6933$、隣接ペア版(3.8.4 節)の内積は $-1.3158$ で **一致しなかった**(次元ペアの取り方が異なれば数値そのものは変わる)。
- **A-3c**: 座標の固定置換 $P$ について、(1) $R^{\text{half}}_{\Theta,m} = P^{\top} R^{\text{interleaved}}_{\Theta,m} P$ の最大要素誤差は $0.000 \times 10^{0}$(完全に一致)、(2) $q, k$ に同じ置換 $P$ を適用すると隣接ペア版の内積は $-0.6933$ となり、A-3b の前半・後半ペア版の値と機械精度で一致した。3.8.7 節で述べた「両者はモデルの表現力として等価」であることの数値的な裏付けになる(A-3b の「一致しない」という結果は、置換を揃えていないことによるもので表現力の違いを意味しない)。
- **A-4**: 相対距離 $t=0$ での平均絶対自己相関は $15.71$ であったのに対し、$t \approx 1980$〜$2000$ 近傍(末尾 10 点の平均)では $4.58$ まで減衰した。最も低い周波数 $\theta_{7} \approx 3.16\times10^{-4}$ の周期が約 $20000$ ステップ(3.8.6 節・6.1 節参照)であることと整合し、相対距離のレンジを広げたことで遠距離減衰の傾向を可視化できた。

### 6.2 実験 B: 位置エンコーディング方式の比較(学習を伴う)

**タスク設計についての注記**: 006「小型 GPT の事前学習」は本トピック(003)を前提とするため、ここで言語モデリングを扱うと前提関係が逆転する。そのため、位置情報が本質的に効く合成タスクとして **可変長 copy task** を用いる。

**タスク**: 系列 $[x_1, \dots, x_L, \mathrm{SEP}, x_1, \dots, x_L]$ を構成し、因果マスク付き Decoder のみの構成(`EncoderBlock`を因果マスクで使い、cross-attention を持たない GPT スタイル、002・実験 1 と同じ手法)で後半部分(2 回目の $x_1, \dots, x_L$)を予測し、損失・精度は後半部分のみで計算する。系列長 $L$ を学習の各ステップでランダムに変え、モデルが「相対的な参照パターン」を学習するよう促す。

**語彙サイズについての設計判断**: 語彙サイズを大きく取ると、系列内で重複がほとんど発生しないため、「直前に出た同じトークンの次を見る」という **内容に基づく induction 機構**(位置情報なしで学習できる)だけでタスクがある程度解けてしまい、方式間の差が出にくい。そこで語彙サイズを $6$ と小さく設定し、同じトークンが複数回出現するようにした。これにより induction 機構だけでは参照先が曖昧になり、位置情報(絶対位置または相対位置)に基づく参照が本質的に必要になる。


```python
CONTENT_VOCAB_SIZE = 6  # 小さい語彙サイズで内容ベース induction のみでの解決を妨げる(上記参照)
SEP_TOKEN_ID = CONTENT_VOCAB_SIZE
VOCAB_SIZE_B = CONTENT_VOCAB_SIZE + 1

D_MODEL_B, N_HEADS_B, D_FF_B, N_LAYERS_B = 64, 4, 256, 2
L_MIN_TRAIN, L_MAX_TRAIN = 4, 16  # 学習時の系列長 L の範囲
EXTRAPOLATION_RATIOS = [1.5, 2.0, 3.0, 4.0]
EXTRAPOLATION_LENGTHS = [int(L_MAX_TRAIN * r) for r in EXTRAPOLATION_RATIOS]
MAX_EVAL_LEN_B = 2 * max(EXTRAPOLATION_LENGTHS) + 1  # cos/sin キャッシュなどのバッファ用

TRAIN_STEPS_B = 2500
BATCH_SIZE_B = 64
LR_B = 3e-4
SEEDS_B = [0, 1, 2]  # 条件間の差を seed 間のばらつきと区別するため、複数 seed で学習する

T5_NUM_BUCKETS = 32
T5_MAX_DISTANCE = 64

print(f"学習時の L の範囲: [{L_MIN_TRAIN}, {L_MAX_TRAIN}]")
print(f"外挿評価の L: {EXTRAPOLATION_LENGTHS} (学習長 {L_MAX_TRAIN} の {EXTRAPOLATION_RATIOS} 倍)")
print(f"学習・評価に用いる seed: {SEEDS_B}")
```

    学習時の L の範囲: [4, 16]
    外挿評価の L: [24, 32, 48, 64] (学習長 16 の [1.5, 2.0, 3.0, 4.0] 倍)
    学習・評価に用いる seed: [0, 1, 2]



```python
def make_copy_batch(
    batch_size: int, seq_len_l: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """可変長 copy task のバッチを生成する。

    系列は [x_1, ..., x_L, SEP, x_1, ..., x_L] の形で、次のトークン予測用に
    (入力, 目標, 損失を計算する位置のマスク)を返す。損失マスクは後半部分
    (2 回目に現れる x_1, ..., x_L を予測する位置)のみ True になる。
    """
    first_half = torch.randint(0, CONTENT_VOCAB_SIZE, (batch_size, seq_len_l), device=device)
    sep = torch.full((batch_size, 1), SEP_TOKEN_ID, device=device)
    full_sequence = torch.cat([first_half, sep, first_half], dim=1)  # (B, 2L+1)

    input_ids = full_sequence[:, :-1]
    target_ids = full_sequence[:, 1:]
    total_len = input_ids.size(1)
    # 入力位置 t(0-indexed)の目標が後半部分の予測に対応するのは t >= L のとき
    loss_mask = torch.arange(total_len, device=device) >= seq_len_l
    return input_ids, target_ids, loss_mask


# 動作確認
_ids, _tgt, _mask = make_copy_batch(2, 4, "cpu")
print("input :", _ids[0].tolist())
print("target:", _tgt[0].tolist())
print("mask  :", _mask.tolist())
```

    input : [0, 5, 3, 0, 6, 0, 5, 3]
    target: [5, 3, 0, 6, 0, 5, 3, 0]
    mask  : [False, False, False, False, True, True, True, True]


**モデル**: 002 の`EncoderBlock`(因果マスク付き、正規化前置)を 2 層積んで使う。加算方式(正弦波・学習可能な絶対位置埋め込み)は埋め込み層に、RoPE・Shaw et al. 方式・T5・ALiBi は各層`self_attn`を対応する`MultiHeadAttention`に差し替えて組み込む。`EncoderBlock`自体は変更しない。


```python
class PositionalEncodingCopyModel(nn.Module):
    """可変長 copy task 用のモデル(比較する 7 条件を 1 つのクラスで表現する)。

    embedding_mode: 埋め込みへの加算方式("none" / "sinusoidal" / "learned")。
    positional_transform_factory: 各層の self_attn に渡す QueryKeyPositionalTransform
        のファクトリ(例: RoPE)。None なら使わない。
    score_bias_factory: 各層の self_attn に渡す AttentionScoreBias のファクトリ
        (例: Shaw et al. 方式・T5・ALiBi)。None なら使わない。
    """

    def __init__(
        self,
        embedding_mode: str,
        positional_transform_factory=None,
        score_bias_factory=None,
        max_learned_len: int = 2 * L_MAX_TRAIN + 1,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE_B, D_MODEL_B)

        self.pos_embedding: nn.Module | None
        if embedding_mode == "sinusoidal":
            self.pos_embedding = SinusoidalPositionalEncoding(D_MODEL_B, max_len=MAX_EVAL_LEN_B)
        elif embedding_mode == "learned":
            # 学習時の最大系列長ちょうどに max_len を設定する(3.4 節: 外挿できないことを
            # 明示的に示すため、学習長を超える位置を要求すると ValueError が送出される)。
            self.pos_embedding = LearnedAbsolutePositionalEmbedding(
                D_MODEL_B, max_len=max_learned_len
            )
        else:
            self.pos_embedding = None

        self.blocks = nn.ModuleList()
        for _ in range(N_LAYERS_B):
            block = EncoderBlock(D_MODEL_B, N_HEADS_B, D_FF_B, norm_first=True)
            if positional_transform_factory is not None or score_bias_factory is not None:
                block.self_attn = MultiHeadAttention(
                    D_MODEL_B,
                    N_HEADS_B,
                    positional_transform=(
                        positional_transform_factory() if positional_transform_factory else None
                    ),
                    attention_score_bias=(score_bias_factory() if score_bias_factory else None),
                )
            self.blocks.append(block)

        self.output_proj = nn.Linear(D_MODEL_B, VOCAB_SIZE_B)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """順伝播。

        Returns:
            (logits, attn_weights_per_layer) のタプル。attn_weights_per_layer は
            層ごとの Attention 重み(形状 (B, h, S, S))のリストで、リストの長さは
            self.blocks の層数。最終層だけでなく全層を観測できるようにするため、
            上書きせずリストに積む(6.3 節: copy 機構がどの層・どのヘッドに
            載るかを層をまたいで調べるために必要)。
        """
        h = self.token_embedding(input_ids)
        if self.pos_embedding is not None:
            h = self.pos_embedding(h)
        seq_len = input_ids.size(1)
        mask = create_causal_mask(seq_len, device=input_ids.device)
        attn_weights_per_layer = []
        for block in self.blocks:
            h, attn_weights = block(h, mask)
            attn_weights_per_layer.append(attn_weights)
        logits = self.output_proj(h)
        return logits, attn_weights_per_layer


D_K_B = D_MODEL_B // N_HEADS_B

MODEL_FACTORIES = {
    "none": lambda: PositionalEncodingCopyModel("none"),
    "sinusoidal": lambda: PositionalEncodingCopyModel("sinusoidal"),
    "learned": lambda: PositionalEncodingCopyModel("learned"),
    "shaw": lambda: PositionalEncodingCopyModel(
        "none",
        score_bias_factory=lambda: ShawRelativePositionBias(
            D_K_B, max_relative_position=L_MAX_TRAIN
        ),
    ),
    "t5": lambda: PositionalEncodingCopyModel(
        "none",
        score_bias_factory=lambda: T5RelativePositionBias(
            N_HEADS_B, num_buckets=T5_NUM_BUCKETS, max_distance=T5_MAX_DISTANCE, bidirectional=False
        ),
    ),
    "alibi": lambda: PositionalEncodingCopyModel(
        "none", score_bias_factory=lambda: ALiBiPositionBias(N_HEADS_B)
    ),
    "rope": lambda: PositionalEncodingCopyModel(
        "none",
        positional_transform_factory=lambda: RotaryPositionEmbedding(
            D_K_B, max_position=MAX_EVAL_LEN_B
        ),
    ),
}
CONDITION_LABELS = {
    "none": "None (baseline)",
    "sinusoidal": "Sinusoidal",
    "learned": "Learned Absolute",
    "shaw": "Shaw et al.",
    "t5": "T5 relative bias",
    "alibi": "ALiBi",
    "rope": "RoPE",
}
print(f"比較する条件: {list(MODEL_FACTORIES.keys())}")
```

    比較する条件: ['none', 'sinusoidal', 'learned', 'shaw', 't5', 'alibi', 'rope']



```python
def evaluate_copy_task(
    model: nn.Module, seq_len_l: int, n_batches: int = 4, batch_size: int = 64, device: str = "cpu"
) -> tuple[float, float]:
    """指定した L で評価し、(平均損失, 精度) を返す。max_len を超える場合は (nan, nan)。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for _ in range(n_batches):
            input_ids, target_ids, loss_mask = make_copy_batch(batch_size, seq_len_l, device)
            try:
                logits, _ = model(input_ids)
            except ValueError:
                return float("nan"), float("nan")
            log_probs = torch.log_softmax(logits, dim=-1)
            masked_log_probs = log_probs[:, loss_mask].reshape(-1, VOCAB_SIZE_B)
            masked_targets = target_ids[:, loss_mask].reshape(-1)
            loss = nn.functional.nll_loss(masked_log_probs, masked_targets)
            total_loss += loss.item()
            pred = logits[:, loss_mask].argmax(dim=-1)
            correct += (pred == target_ids[:, loss_mask]).sum().item()
            total += pred.numel()
    return total_loss / n_batches, correct / total


def train_copy_model(name: str, seed: int = SEEDS_B[0]) -> tuple[nn.Module, float]:
    """copy task で 1 条件を学習し、(モデル, 学習時間[秒]) を返す。"""
    torch.manual_seed(seed)
    model = MODEL_FACTORIES[name]().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_B)

    start = time.time()
    for _ in range(TRAIN_STEPS_B):
        seq_len_l = torch.randint(L_MIN_TRAIN, L_MAX_TRAIN + 1, (1,)).item()
        input_ids, target_ids, loss_mask = make_copy_batch(BATCH_SIZE_B, seq_len_l, DEVICE)
        logits, _ = model(input_ids)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = nn.functional.nll_loss(
            log_probs[:, loss_mask].reshape(-1, VOCAB_SIZE_B), target_ids[:, loss_mask].reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    elapsed = time.time() - start
    return model, elapsed
```


```python
def _nan_safe_stat(arr: np.ndarray, func) -> list[float]:
    """列(系列長)ごとに集計する。全 seed が nan の列(学習可能な絶対位置埋め込みが
    max_len を超えて評価不能になった場合)は nan をそのまま返し、`np.nanmean`
    などを全 nan 列に対して呼んで出る RuntimeWarning を避ける。
    """
    result = []
    for col in arr.T:
        if np.all(np.isnan(col)):
            result.append(float("nan"))
        else:
            result.append(float(func(col)))
    return result


results_exp_b = {}
trained_models_b = {}
eval_lengths_b = [L_MAX_TRAIN] + EXTRAPOLATION_LENGTHS

for name in MODEL_FACTORIES:
    per_seed_losses, per_seed_accs, per_seed_times = [], [], []
    trained_models_b[name] = {}

    for seed in SEEDS_B:
        model_b, elapsed_b = train_copy_model(name, seed=seed)
        trained_models_b[name][seed] = model_b
        per_seed_times.append(elapsed_b)

        losses_b, accs_b = [], []
        for l_eval in eval_lengths_b:
            loss_b, acc_b = evaluate_copy_task(model_b, l_eval, device=DEVICE)
            losses_b.append(loss_b)
            accs_b.append(acc_b)
        per_seed_losses.append(losses_b)
        per_seed_accs.append(accs_b)

    losses_arr = np.array(per_seed_losses)  # (len(SEEDS_B), len(eval_lengths_b))
    accs_arr = np.array(per_seed_accs)

    results_exp_b[name] = {
        "lengths": eval_lengths_b,
        "losses_mean": _nan_safe_stat(losses_arr, np.nanmean),
        "losses_min": _nan_safe_stat(losses_arr, np.nanmin),
        "losses_max": _nan_safe_stat(losses_arr, np.nanmax),
        "accs_mean": _nan_safe_stat(accs_arr, np.nanmean),
        "accs_min": _nan_safe_stat(accs_arr, np.nanmin),
        "accs_max": _nan_safe_stat(accs_arr, np.nanmax),
        "time_mean": sum(per_seed_times) / len(per_seed_times),
        "per_seed_accs": per_seed_accs,
    }

    print(
        f"{CONDITION_LABELS[name]:20s} time={results_exp_b[name]['time_mean']:5.1f}s(seed 平均)  "
        + "  ".join(
            f"L={l_eval}:acc={mean:.3f}[{lo:.3f},{hi:.3f}]"
            for l_eval, mean, lo, hi in zip(
                eval_lengths_b,
                results_exp_b[name]["accs_mean"],
                results_exp_b[name]["accs_min"],
                results_exp_b[name]["accs_max"],
                strict=True,
            )
        )
    )
```

    None (baseline)      time= 21.7s(seed 平均)  L=16:acc=0.634[0.587,0.668]  L=24:acc=0.478[0.448,0.501]  L=32:acc=0.390[0.352,0.424]  L=48:acc=0.308[0.284,0.342]  L=64:acc=0.266[0.246,0.298]


    Sinusoidal           time= 21.8s(seed 平均)  L=16:acc=0.986[0.982,0.991]  L=24:acc=0.221[0.187,0.248]  L=32:acc=0.238[0.214,0.269]  L=48:acc=0.242[0.230,0.254]  L=64:acc=0.213[0.203,0.225]


    Learned Absolute     time= 21.8s(seed 平均)  L=16:acc=0.813[0.787,0.848]  L=24:acc=nan[nan,nan]  L=32:acc=nan[nan,nan]  L=48:acc=nan[nan,nan]  L=64:acc=nan[nan,nan]


    Shaw et al.          time= 23.7s(seed 平均)  L=16:acc=0.959[0.929,0.975]  L=24:acc=0.891[0.798,0.947]  L=32:acc=0.813[0.674,0.904]  L=48:acc=0.642[0.489,0.756]  L=64:acc=0.505[0.392,0.595]


    T5 relative bias     time= 22.2s(seed 平均)  L=16:acc=0.725[0.576,0.862]  L=24:acc=0.376[0.344,0.424]  L=32:acc=0.451[0.269,0.631]  L=48:acc=0.275[0.258,0.286]  L=64:acc=0.277[0.229,0.323]


    ALiBi                time= 22.1s(seed 平均)  L=16:acc=0.967[0.964,0.969]  L=24:acc=0.920[0.913,0.925]  L=32:acc=0.855[0.849,0.863]  L=48:acc=0.711[0.692,0.722]  L=64:acc=0.603[0.578,0.615]


    RoPE                 time= 23.8s(seed 平均)  L=16:acc=0.988[0.983,0.992]  L=24:acc=0.330[0.280,0.367]  L=32:acc=0.311[0.299,0.329]  L=48:acc=0.173[0.168,0.177]  L=64:acc=0.167[0.163,0.175]


#### 実験 B-1: 学習長内での性能

学習時と同じ範囲(ここでは $L = L_{\max} = 16$)での精度を比較する。ここで測っているのは **位置情報の表現力そのもの** である(学習長を超えないため、外挿性能は問わない)。


```python
fig, ax = plt.subplots(figsize=(7.5, 4.5))
names_b1 = list(MODEL_FACTORIES.keys())
means_b1 = np.array(
    [results_exp_b[name]["accs_mean"][0] for name in names_b1]
)  # 先頭が L_MAX_TRAIN
mins_b1 = np.array([results_exp_b[name]["accs_min"][0] for name in names_b1])
maxs_b1 = np.array([results_exp_b[name]["accs_max"][0] for name in names_b1])
yerr_b1 = np.vstack([means_b1 - mins_b1, maxs_b1 - means_b1])

colors_b1 = plt.cm.tab10(np.linspace(0, 1, len(names_b1)))
ax.bar(
    [CONDITION_LABELS[n] for n in names_b1],
    means_b1,
    yerr=yerr_b1,
    capsize=4,
    color=colors_b1,
)
ax.set_ylabel(f"Accuracy at L = {L_MAX_TRAIN} (within training range)")
ax.set_title(f"Experiment B-1: in-distribution accuracy (mean, min-max over {len(SEEDS_B)} seeds)")
ax.set_ylim(0, 1.05)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

for name in names_b1:
    r = results_exp_b[name]
    print(
        f"{CONDITION_LABELS[name]:20s} L={L_MAX_TRAIN}: "
        f"loss={r['losses_mean'][0]:.4f}  "
        f"acc(mean)={r['accs_mean'][0]:.4f}  "
        f"acc(min-max)=[{r['accs_min'][0]:.4f}, {r['accs_max'][0]:.4f}]"
    )
```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/003_positional_encoding_rope/output_47_0.png)
    


    None (baseline)      L=16: loss=0.9653  acc(mean)=0.6340  acc(min-max)=[0.5869, 0.6677]
    Sinusoidal           L=16: loss=0.0356  acc(mean)=0.9863  acc(min-max)=[0.9822, 0.9907]
    Learned Absolute     L=16: loss=0.4737  acc(mean)=0.8129  acc(min-max)=[0.7874, 0.8477]
    Shaw et al.          L=16: loss=0.1126  acc(mean)=0.9591  acc(min-max)=[0.9292, 0.9753]
    T5 relative bias     L=16: loss=0.7192  acc(mean)=0.7253  acc(min-max)=[0.5757, 0.8616]
    ALiBi                L=16: loss=0.0940  acc(mean)=0.9666  acc(min-max)=[0.9639, 0.9692]
    RoPE                 L=16: loss=0.0348  acc(mean)=0.9878  acc(min-max)=[0.9829, 0.9917]


#### 実験 B-2: 学習長を超える外挿性能

学習時の最大系列長($L_{\max}=16$)の 1.5, 2, 3, 4 倍の複数の長さで評価する。ここで測っているのは **外挿(length extrapolation)性能** である。


```python
fig, ax = plt.subplots(figsize=(8.5, 5.0))
for name in MODEL_FACTORIES:
    r = results_exp_b[name]
    lengths_b2 = r["lengths"]
    means_b2 = r["accs_mean"]
    mins_b2 = r["accs_min"]
    maxs_b2 = r["accs_max"]
    (line,) = ax.plot(lengths_b2, means_b2, marker="o", label=CONDITION_LABELS[name])
    ax.fill_between(lengths_b2, mins_b2, maxs_b2, color=line.get_color(), alpha=0.15)

ax.axvline(
    L_MAX_TRAIN,
    color="gray",
    linestyle="--",
    linewidth=1,
    label=f"Training max length (L={L_MAX_TRAIN})",
)
ax.text(
    EXTRAPOLATION_LENGTHS[1],
    0.05,
    "Learned Absolute: not evaluable beyond max_len (all-nan)",
    fontsize=8,
    color="gray",
    ha="center",
)
ax.set_xlabel("Evaluation sequence length L")
ax.set_ylabel("Accuracy")
ax.set_title(f"Experiment B-2: length extrapolation (mean, min-max band over {len(SEEDS_B)} seeds)")
ax.set_ylim(-0.05, 1.05)
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/003_positional_encoding_rope/output_49_0.png)
    


**T5 の相対距離バケットの補足分析**: T5(`num_buckets=32`・`max_distance=64`・`bidirectional=False`)が学習中に実際に受け取る相対距離の範囲と、評価時に受け取る範囲を比較する。学習時の入力系列長は最大でも $2 L_{\max} = 32$ であり、causal な自己注意で意味を持つ相対距離(Key が Query 以前にある場合)は高々 $31$ である。


```python
t5_probe = T5RelativePositionBias(
    N_HEADS_B, num_buckets=T5_NUM_BUCKETS, max_distance=T5_MAX_DISTANCE, bidirectional=False
)

max_rel_dist_train = 2 * L_MAX_TRAIN - 1  # 学習中に現れる最大の相対距離(n - m の絶対値)
train_rel_distances = torch.arange(0, max_rel_dist_train + 1)
train_buckets = t5_probe.relative_position_bucket(-train_rel_distances)
max_trained_bucket = train_buckets.max().item()

all_buckets = torch.arange(T5_NUM_BUCKETS)
untrained_buckets = (all_buckets > max_trained_bucket).sum().item()

eval_rel_dist_l64 = 2 * EXTRAPOLATION_LENGTHS[-1] - 1  # L=64 での最大相対距離
eval_bucket_l64 = t5_probe.relative_position_bucket(torch.tensor([-eval_rel_dist_l64])).item()

print(f"学習中に現れる最大相対距離: {max_rel_dist_train} -> バケット {max_trained_bucket}")
print(f"学習中に一度も勾配を受けないバケット数: {untrained_buckets} / {T5_NUM_BUCKETS}")
eval_l64 = EXTRAPOLATION_LENGTHS[-1]
print(f"評価 L={eval_l64} での最大相対距離: {eval_rel_dist_l64} -> バケット {eval_bucket_l64}")
```

    学習中に現れる最大相対距離: 31 -> バケット 23
    学習中に一度も勾配を受けないバケット数: 8 / 32
    評価 L=64 での最大相対距離: 127 -> バケット 31


#### 実験 B の結果・考察

実測値(6.2 節の実行結果より、$3$ seed(0, 1, 2)の平均値、括弧内は最小値・最大値)は以下の通り。

**実験 B-1(学習長内 $L=16$、3.2〜3.8 節との対応)**:

| 条件 | 損失(平均) | 精度(平均 [最小, 最大]) |
|---|---:|---|
| なし(対照群) | 0.9653 | 0.6340 [0.5869, 0.6677] |
| 正弦波方式 | 0.0356 | 0.9863 [0.9822, 0.9907] |
| 学習可能な絶対位置埋め込み | 0.4737 | 0.8129 [0.7874, 0.8477] |
| Shaw et al. 方式 | 0.1126 | 0.9591 [0.9292, 0.9753] |
| T5 の相対位置バイアス | 0.7192 | 0.7253 [0.5757, 0.8616] |
| ALiBi | 0.0940 | 0.9666 [0.9639, 0.9692] |
| RoPE | 0.0348 | 0.9878 [0.9829, 0.9917] |

$3$ seed の範囲で比較すると、**RoPE(0.988)・正弦波方式(0.986)** の最小値(0.983・0.982)が **ALiBi(0.967)・Shaw et al. 方式(0.959)** の最大値(0.969・0.975)を上回り、他条件よりはっきり高い精度に達した。ALiBi と Shaw et al. 方式は範囲が重なり($[0.964, 0.969] \subset [0.929, 0.975]$)$3$ seed では区別できない。**学習可能な絶対位置埋め込み(0.813)** はこの 2 条件より明確に低く(最大値 0.848 が Shaw et al. 方式の最小値 0.929 を下回る)、**T5(0.725)・対照群(0.634)** も範囲が重なり($[0.587, 0.668] \subset [0.576, 0.862]$)区別できない。ただし T5 は分散が非常に大きく(範囲 $0.29$)、seed によって対照群並みから学習可能な絶対位置埋め込みに匹敵する精度まで大きくばらついた。

**対照群がチャンスレベル($\approx 0.167$)を大きく上回る理由**: 3.1 節の通り因果マスクの下では位置 $m$ の Query が参照できる Key の集合が $\{0, \dots, m\}$ に制限され、この非対称性自体が暗黙の位置情報として働くためである。加えて内容ベースの induction 機構(位置エンコーディング不要)も一定の精度に寄与しうる(語彙サイズ $6$ に絞ったことで単独では解けないが、部分的な寄与までは排除できていない)。Haviv et al. [7] の知見と整合する結果である。

**実験 B-2(学習長超、3.4・3.7・3.8 節との対応)**:

| 条件 | L=24 | L=32 | L=48 | L=64($4\times$) |
|---|---|---|---|---|
| なし | 0.478 [0.448, 0.501] | 0.390 [0.352, 0.424] | 0.308 [0.284, 0.342] | 0.266 [0.246, 0.298] |
| 正弦波方式 | 0.221 [0.187, 0.248] | 0.238 [0.214, 0.269] | 0.242 [0.230, 0.254] | 0.213 [0.203, 0.225] |
| 学習可能な絶対位置埋め込み | nan | nan | nan | nan |
| Shaw et al. 方式 | 0.891 [0.798, 0.947] | 0.813 [0.674, 0.904] | 0.642 [0.489, 0.756] | 0.505 [0.392, 0.595] |
| T5 | 0.376 [0.344, 0.424] | 0.451 [0.269, 0.631] | 0.275 [0.258, 0.286] | 0.277 [0.229, 0.323] |
| ALiBi | 0.920 [0.913, 0.925] | 0.855 [0.849, 0.863] | 0.711 [0.692, 0.722] | 0.603 [0.578, 0.615] |
| RoPE | 0.330 [0.280, 0.367] | 0.311 [0.299, 0.329] | 0.173 [0.168, 0.177] | 0.167 [0.163, 0.175] |

学習可能な絶対位置埋め込みは、3.4 節の通り`max_len`($=2L_{\max}+1=33$)を超える位置で`ValueError`が送出され評価は全域`nan`となった。「性能が劣化する」のではなく「原理的に動作しない」という 3.4 節の議論をそのまま裏付ける。

実験前の想定(「$L=48$・$L=64$ あたりで対照群を下回る」程度の緩やかな劣化)より厳しく、**正弦波方式・RoPE はいずれも $1.5$ 倍長($L=24$)の時点ですでに対照群を下回り、$4$ つの外挿長すべてで下回り続けた**(全長で $3$ seed の範囲に重なりなし。例えば $L=24$: $0.221$/$0.330$ vs 対照群 $0.478$、$L=64$: $0.213$/$0.167$ vs $0.266$、上表参照)。T5 も $L=24$ で対照群を下回り(重なりなし)、$L=32$〜$L=64$ では区別できない水準まで落ち込んだ。この結果は 3.1 節の議論と整合する:因果マスクだけが与える暗黙の位置情報(「位置 $m$ は $m+1$ 個の Key しか参照できない」)は学習長を超えても意味を失わないのに対し、正弦波方式・RoPE・T5 が学習した位置パターンは学習長の範囲でのみ意味を持つよう最適化されており、学習長を超えた位置には **学習時に見たことのない、意味をなさない信号** を Attention に持ち込む。この信号が、暗黙の位置情報だけに頼るよりも有害に働きうる、というのが今回の結果である。

対照的に、**ALiBi(0.603 @ $L=64$)・Shaw et al. 方式(0.505 @ $L=64$)は最も緩やかな劣化を示し、全外挿長で対照群を明確に上回り続けた**(両者の範囲は重なり、$3$ seed では優劣を確定できない)。ALiBi は分散が小さく安定して高い精度を維持し、Shaw et al. 方式も $k_{\text{clip}}=16$ のクリップ(3.5 節)のおかげでクリップ範囲を超えた遠距離で新しい情報が増えず対照群を上回った。これは 3.7 節の通り **ALiBi が外挿を明示的な設計目標としている手法である**([5])ことの帰結であり、RoPE が素の状態では学習長を大きく超えると急激に劣化しやすいという既知の傾向と整合する(**「ALiBi の方が優れた手法である」ことを意味しない** — 実験 B-1 の表現力では RoPE・正弦波方式が ALiBi を上回っており、測っている軸が異なる、7 節)。一方 T5 は、5.3 節の補足分析(6.2 節末尾)によると学習中に一度も勾配を受けないバケットが存在し(全 $32$ 個中 $8$ 個)、評価時 $L=64$ で相対距離が未学習のオーバーフローバケットに落ちることが、T5 の外挿性能の低さ(むしろ $L=24$ では対照群も下回る)の具体的な要因だと考えられる。



## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/01_foundations/003_positional_encoding_rope.ipynb
