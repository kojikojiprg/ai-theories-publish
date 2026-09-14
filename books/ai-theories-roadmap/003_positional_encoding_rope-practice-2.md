---
title: "位置エンコーディング(Positional Encoding)/ RoPE(実装・実験編 2/2)"
---

この記事は後編(実装・実験編 2/2)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/003_positional_encoding_rope-practice-1)。

### 6.3 実験 C: Attention 重みの可視化

正弦波方式・ALiBi・RoPE(および対照群)について、copy task が「参照すべき位置に注目できているか」を定量指標で確認し、代表例をヒートマップで可視化する。学習長内($L=16$)と学習長を超える系列($L=32$、学習長の 2 倍)の両方で評価し、外挿時に何が壊れるかを確認する。モデルは 2 層(`N_LAYERS_B`)・ヘッド数 $4$(`N_HEADS_B`)であり、指標は特定の層・ヘッドに決め打ちせず **全層・全ヘッド** を対象に計算する。

**参照すべき位置のオフセットについて**:`make_copy_batch`(6.2 節)が構成する系列は`[x_1, ..., x_L, SEP, x_1, ..., x_L]`であり、`input_ids`はこの系列の末尾 1 トークンを除いたもの(長さ $2L$、0-indexed で位置 $0$〜$2L-1$)である。位置 $L$ が`SEP`、位置 $L+1+j$($j=0,\dots,L-1$)が 2 回目の $x_{j+1}$ に対応する。後半の位置 $m$($m \ge L$)の次のトークンを予測するための正解は`target_ids[m] = full_sequence[m+1]`であり、$m=L+j$ のとき`full_sequence[L+j+1] = x_{j+1} = \text{full\_sequence}[j]`(前半の位置 $j = m - L$ の値)と一致する。したがって、位置 $m$ が参照すべき前半の位置は $m - L$ である(以下の指標計算はこの値を用いる)。


```python
def compute_chance_level(seq_len_l: int, device: str = "cpu") -> float:
    """因果マスク下で一様な Attention 重みを仮定した場合のチャンスレベルを計算する。

    位置 m(m = L, ..., 2L-1)は m+1 個の Key(位置 0..m)に一様に注意を割り振ると
    仮定すると、参照位置 m-L への重みの期待値は 1/(m+1) である。指標は後半の
    全位置で平均するため、チャンスレベルは (1/L) * sum_{m=L}^{2L-1} 1/(m+1)。
    argmax が参照位置と一致する確率も、一様分布上ではどの位置も等確率で
    argmax になりうるとみなせるため、同じ 1/(m+1) の平均に一致する。
    """
    m_positions = torch.arange(seq_len_l, 2 * seq_len_l, device=device)
    return (1.0 / (m_positions + 1).float()).mean().item()


def compute_attention_concentration_all_heads(
    model: nn.Module, seq_len_l: int, device: str = "cpu", n_samples: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """全層・全ヘッドについて、参照位置への重みと argmax 一致率を計算する。

    Returns:
        (weight_at_ref, argmax_match) のタプル。いずれも形状 (n_layers, h) の
        Tensor(n_samples 個のバッチ・後半全位置で平均した値)。
    """
    model.eval()
    weight_sum: torch.Tensor | None = None
    match_sum: torch.Tensor | None = None
    with torch.no_grad():
        for _ in range(n_samples):
            input_ids, _, _ = make_copy_batch(1, seq_len_l, device)
            _, attn_weights_per_layer = model(input_ids)
            m_positions = torch.arange(seq_len_l, 2 * seq_len_l, device=device)
            ref_positions = m_positions - seq_len_l

            layer_weights, layer_matches = [], []
            for attn_weights in attn_weights_per_layer:
                attn = attn_weights[0]  # (h, S, S)
                w = attn[:, m_positions, ref_positions]  # (h, len(m_positions))
                layer_weights.append(w.mean(dim=-1))  # (h,)
                argmax_pred = attn[:, m_positions, :].argmax(dim=-1)  # (h, len(m_positions))
                match = (argmax_pred == ref_positions[None, :]).float().mean(dim=-1)  # (h,)
                layer_matches.append(match)

            layer_weights_t = torch.stack(layer_weights)  # (n_layers, h)
            layer_matches_t = torch.stack(layer_matches)
            if weight_sum is None:
                weight_sum, match_sum = layer_weights_t, layer_matches_t
            else:
                weight_sum = weight_sum + layer_weights_t
                match_sum = match_sum + layer_matches_t
    return weight_sum / n_samples, match_sum / n_samples


METRIC_CONDITIONS_C = ["none", "sinusoidal", "alibi", "rope"]
concentration_full = {}

for name in METRIC_CONDITIONS_C:
    for l_vis in (L_MAX_TRAIN, 2 * L_MAX_TRAIN):
        weight_per_seed, match_per_seed = [], []
        best_layer_per_seed, best_head_per_seed = [], []
        for seed in SEEDS_B:
            torch.manual_seed(seed)
            w, m = compute_attention_concentration_all_heads(
                trained_models_b[name][seed], l_vis, DEVICE
            )
            weight_per_seed.append(w)
            match_per_seed.append(m)
            best_layer, best_head = divmod(w.argmax().item(), w.size(1))
            best_layer_per_seed.append(best_layer)
            best_head_per_seed.append(best_head)
        concentration_full[(name, l_vis)] = {
            "weight_per_seed": weight_per_seed,  # 各 seed: (n_layers, h)
            "match_per_seed": match_per_seed,
            "best_layer_per_seed": best_layer_per_seed,  # 各 seed: 重み最大の層
            "best_head_per_seed": best_head_per_seed,  # 各 seed: 重み最大のヘッド
        }

# 表 1(主表): 条件 x 系列長ごとに、重み(weight_at_ref)が最大となる (層, ヘッド) を
# seed ごとに求め、その (層, ヘッド) における重みと argmax 一致率の両方を集計する。
# 2 つの指標を別々に最大化すると異なる (層, ヘッド) 由来の値が並びうるため、
# 同一の (層, ヘッド) の値を揃えて表にする。
print("=== 表 1: 重みが最大の (層, ヘッド) における重み・argmax 一致率(seed 平均[最小,最大]) ===")
table1_rows = {}
for name in METRIC_CONDITIONS_C:
    for l_vis in (L_MAX_TRAIN, 2 * L_MAX_TRAIN):
        r = concentration_full[(name, l_vis)]
        weight_at_best_per_seed = []
        match_at_best_per_seed = []
        for seed_idx in range(len(SEEDS_B)):
            bl, bh = r["best_layer_per_seed"][seed_idx], r["best_head_per_seed"][seed_idx]
            weight_at_best_per_seed.append(r["weight_per_seed"][seed_idx][bl, bh].item())
            match_at_best_per_seed.append(r["match_per_seed"][seed_idx][bl, bh].item())
        chance = compute_chance_level(l_vis)
        table1_rows[(name, l_vis)] = {
            "weight_mean": sum(weight_at_best_per_seed) / len(weight_at_best_per_seed),
            "weight_min": min(weight_at_best_per_seed),
            "weight_max": max(weight_at_best_per_seed),
            "match_mean": sum(match_at_best_per_seed) / len(match_at_best_per_seed),
            "match_min": min(match_at_best_per_seed),
            "match_max": max(match_at_best_per_seed),
            "chance": chance,
        }
        t = table1_rows[(name, l_vis)]
        w_str = f"{t['weight_mean']:.4f}[{t['weight_min']:.4f},{t['weight_max']:.4f}]"
        m_str = f"{t['match_mean']:.4f}[{t['match_min']:.4f},{t['match_max']:.4f}]"
        label = f"{CONDITION_LABELS[name]:16s} L={l_vis:3d}"
        print(f"{label}  weight={w_str}  argmax_match={m_str}  chance={chance:.4f}")

# 表 2(補助表): 条件 x 系列長ごとの層・ヘッドごとの重み(seed 平均)
print("\n=== 表 2: 層・ヘッドごとの参照位置への重み(seed 平均) ===")
table2_rows = {}
for name in METRIC_CONDITIONS_C:
    for l_vis in (L_MAX_TRAIN, 2 * L_MAX_TRAIN):
        r = concentration_full[(name, l_vis)]
        weight_layerhead_mean = torch.stack(r["weight_per_seed"]).mean(dim=0)  # (n_layers, h)
        table2_rows[(name, l_vis)] = weight_layerhead_mean
        label = f"{CONDITION_LABELS[name]:16s} L={l_vis:3d}"
        for layer_idx in range(weight_layerhead_mean.size(0)):
            head_values = ", ".join(
                f"h{h_idx}={v:.4f}"
                for h_idx, v in enumerate(weight_layerhead_mean[layer_idx].tolist())
            )
            print(f"{label}  layer={layer_idx}  {head_values}")

# 表 3(補足): seed ごとに重みが最大となる (層, ヘッド) と、比較用の最終層 head 0 の値。
# 「copy 機構が載る (層, ヘッド) が seed によって異なるか」を直接検証するための参考データ。
print("\n=== 表 3: seed ごとの最大値 (層, ヘッド) と最終層 head 0 の値 ===")
last_layer_idx = N_LAYERS_B - 1
for name in METRIC_CONDITIONS_C:
    for l_vis in (L_MAX_TRAIN, 2 * L_MAX_TRAIN):
        r = concentration_full[(name, l_vis)]
        label = f"{CONDITION_LABELS[name]:16s} L={l_vis:3d}"
        for seed_idx, seed in enumerate(SEEDS_B):
            bl, bh = r["best_layer_per_seed"][seed_idx], r["best_head_per_seed"][seed_idx]
            best_val = r["weight_per_seed"][seed_idx][bl, bh].item()
            head0_val = r["weight_per_seed"][seed_idx][last_layer_idx, 0].item()
            best_str = f"best=(layer={bl},head={bh})={best_val:.4f}"
            head0_str = f"head0(layer={last_layer_idx})={head0_val:.4f}"
            print(f"{label}  seed={seed}  {best_str}  {head0_str}")
```

    === 表 1: 重みが最大の (層, ヘッド) における重み・argmax 一致率(seed 平均[最小,最大]) ===
    None (baseline)  L= 16  weight=0.1553[0.1410,0.1767]  argmax_match=0.2500[0.2188,0.2969]  chance=0.0424
    None (baseline)  L= 32  weight=0.0559[0.0551,0.0566]  argmax_match=0.0977[0.0703,0.1250]  chance=0.0214
    Sinusoidal       L= 16  weight=0.5760[0.5529,0.6058]  argmax_match=0.9010[0.8594,0.9375]  chance=0.0424
    Sinusoidal       L= 32  weight=0.0583[0.0434,0.0674]  argmax_match=0.0859[0.0625,0.1055]  chance=0.0214
    ALiBi            L= 16  weight=0.5971[0.5725,0.6222]  argmax_match=0.8594[0.8438,0.8906]  chance=0.0424
    ALiBi            L= 32  weight=0.4815[0.4739,0.4945]  argmax_match=0.8307[0.8203,0.8477]  chance=0.0214
    RoPE             L= 16  weight=0.7018[0.6635,0.7396]  argmax_match=0.9193[0.8984,0.9453]  chance=0.0424
    RoPE             L= 32  weight=0.1553[0.1000,0.2132]  argmax_match=0.2161[0.1406,0.2891]  chance=0.0214
    
    === 表 2: 層・ヘッドごとの参照位置への重み(seed 平均) ===
    None (baseline)  L= 16  layer=0  h0=0.0389, h1=0.0378, h2=0.0364, h3=0.0415
    None (baseline)  L= 16  layer=1  h0=0.0836, h1=0.1553, h2=0.0392, h3=0.0633
    None (baseline)  L= 32  layer=0  h0=0.0204, h1=0.0199, h2=0.0192, h3=0.0214
    None (baseline)  L= 32  layer=1  h0=0.0364, h1=0.0559, h2=0.0224, h3=0.0280
    Sinusoidal       L= 16  layer=0  h0=0.0278, h1=0.0413, h2=0.0289, h3=0.0282
    Sinusoidal       L= 16  layer=1  h0=0.2819, h1=0.4162, h2=0.2218, h3=0.0888
    Sinusoidal       L= 32  layer=0  h0=0.0223, h1=0.0257, h2=0.0238, h3=0.0253
    Sinusoidal       L= 32  layer=1  h0=0.0338, h1=0.0510, h2=0.0545, h3=0.0361
    ALiBi            L= 16  layer=0  h0=0.0040, h1=0.0265, h2=0.0341, h3=0.0397
    ALiBi            L= 16  layer=1  h0=0.0138, h1=0.2730, h2=0.3427, h3=0.4546
    ALiBi            L= 32  layer=0  h0=0.0001, h1=0.0083, h2=0.0164, h3=0.0198
    ALiBi            L= 32  layer=1  h0=0.0003, h1=0.1482, h2=0.2510, h3=0.3588
    RoPE             L= 16  layer=0  h0=0.0170, h1=0.0213, h2=0.0571, h3=0.0252
    RoPE             L= 16  layer=1  h0=0.4930, h1=0.6784, h2=0.5339, h3=0.2617
    RoPE             L= 32  layer=0  h0=0.0037, h1=0.0089, h2=0.0090, h3=0.0037
    RoPE             L= 32  layer=1  h0=0.0851, h1=0.1164, h2=0.1004, h3=0.0473
    
    === 表 3: seed ごとの最大値 (層, ヘッド) と最終層 head 0 の値 ===
    None (baseline)  L= 16  seed=0  best=(layer=1,head=1)=0.1482  head0(layer=1)=0.0791
    None (baseline)  L= 16  seed=1  best=(layer=1,head=1)=0.1410  head0(layer=1)=0.1056
    None (baseline)  L= 16  seed=2  best=(layer=1,head=1)=0.1767  head0(layer=1)=0.0662
    None (baseline)  L= 32  seed=0  best=(layer=1,head=1)=0.0566  head0(layer=1)=0.0338
    None (baseline)  L= 32  seed=1  best=(layer=1,head=1)=0.0551  head0(layer=1)=0.0383
    None (baseline)  L= 32  seed=2  best=(layer=1,head=1)=0.0562  head0(layer=1)=0.0370
    Sinusoidal       L= 16  seed=0  best=(layer=1,head=0)=0.5691  head0(layer=1)=0.5691
    Sinusoidal       L= 16  seed=1  best=(layer=1,head=1)=0.6058  head0(layer=1)=0.1920
    Sinusoidal       L= 16  seed=2  best=(layer=1,head=1)=0.5529  head0(layer=1)=0.0845
    Sinusoidal       L= 32  seed=0  best=(layer=1,head=2)=0.0434  head0(layer=1)=0.0433
    Sinusoidal       L= 32  seed=1  best=(layer=1,head=1)=0.0674  head0(layer=1)=0.0285
    Sinusoidal       L= 32  seed=2  best=(layer=1,head=2)=0.0642  head0(layer=1)=0.0297
    ALiBi            L= 16  seed=0  best=(layer=1,head=3)=0.5725  head0(layer=1)=0.0107
    ALiBi            L= 16  seed=1  best=(layer=1,head=2)=0.6222  head0(layer=1)=0.0140
    ALiBi            L= 16  seed=2  best=(layer=1,head=3)=0.5967  head0(layer=1)=0.0167
    ALiBi            L= 32  seed=0  best=(layer=1,head=3)=0.4761  head0(layer=1)=0.0003
    ALiBi            L= 32  seed=1  best=(layer=1,head=2)=0.4945  head0(layer=1)=0.0003
    ALiBi            L= 32  seed=2  best=(layer=1,head=3)=0.4739  head0(layer=1)=0.0003
    RoPE             L= 16  seed=0  best=(layer=1,head=1)=0.7025  head0(layer=1)=0.0993
    RoPE             L= 16  seed=1  best=(layer=1,head=0)=0.7396  head0(layer=1)=0.7396
    RoPE             L= 16  seed=2  best=(layer=1,head=2)=0.6635  head0(layer=1)=0.6403
    RoPE             L= 32  seed=0  best=(layer=1,head=1)=0.1000  head0(layer=1)=0.0427
    RoPE             L= 32  seed=1  best=(layer=1,head=1)=0.2132  head0(layer=1)=0.0618
    RoPE             L= 32  seed=2  best=(layer=1,head=2)=0.1528  head0(layer=1)=0.1508



```python
def get_attention_weights(
    model: nn.Module, seq_len_l: int, layer_idx: int, head_idx: int, device: str = "cpu"
) -> torch.Tensor:
    """1 サンプルぶんの入力に対する、指定した層・ヘッドの Attention 重みを取得する。"""
    model.eval()
    input_ids, _, _ = make_copy_batch(1, seq_len_l, device)
    with torch.no_grad():
        _, attn_weights_per_layer = model(input_ids)
    return attn_weights_per_layer[layer_idx][0, head_idx]  # (S, S)


VIS_CONDITIONS_C = ["sinusoidal", "alibi", "rope"]
fig, axes = plt.subplots(2, len(VIS_CONDITIONS_C), figsize=(5.0 * len(VIS_CONDITIONS_C), 9.0))

# 可視化は代表として先頭の seed(SEEDS_B[0])で学習したモデルを使い、条件・系列長ごとに
# 表 2 の元データ(weight_per_seed[0] が SEEDS_B[0] に対応)から最も集中度が高い
# (層, ヘッド)を選んで描く。
for col, name in enumerate(VIS_CONDITIONS_C):
    model_c = trained_models_b[name][SEEDS_B[0]]
    for row, l_vis in enumerate((L_MAX_TRAIN, 2 * L_MAX_TRAIN)):
        r = concentration_full[(name, l_vis)]
        best_layer, best_head = r["best_layer_per_seed"][0], r["best_head_per_seed"][0]
        best_value = r["weight_per_seed"][0][best_layer, best_head].item()

        torch.manual_seed(SEED)
        attn_c = get_attention_weights(model_c, l_vis, best_layer, best_head, DEVICE)
        length_note = "  = train length" if l_vis == L_MAX_TRAIN else "  = 2x train length"
        title = (
            f"{CONDITION_LABELS[name]}  (L={l_vis}{length_note})\n"
            f"layer={best_layer}, head={best_head} (weight_at_ref={best_value:.3f})"
        )
        plot_attention_heatmap(
            attn_c, title=title, ax=axes[row, col], colorbar=(col == len(VIS_CONDITIONS_C) - 1)
        )

plt.tight_layout()
plt.show()
```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/003_positional_encoding_rope/output_55_0.png)
    


#### 実験 C の結果・考察

**表 1(重みが最大の (層, ヘッド) における重み・argmax 一致率、6.3 節の実行結果より、$3$ seed の平均 [最小, 最大])**: 参照位置への重みが最大となる (層, ヘッド) を seed ごとに特定し、同一の (層, ヘッド) における重みと argmax 一致率を並べている(2 つの指標を別々に最大化すると異なる (層, ヘッド) が選ばれうるため)。

| 条件 | $L$ | 参照位置への重み(最大の (層, ヘッド)) | argmax 一致率(同一の (層, ヘッド)) | チャンスレベル |
|---|---:|---|---|---:|
| なし(対照群) | 16 | 0.1553 [0.1410, 0.1767] | 0.2500 [0.2188, 0.2969] | 0.0424 |
| なし(対照群) | 32 | 0.0559 [0.0551, 0.0566] | 0.0977 [0.0703, 0.1250] | 0.0214 |
| 正弦波方式 | 16 | 0.5760 [0.5529, 0.6058] | 0.9010 [0.8594, 0.9375] | 0.0424 |
| 正弦波方式 | 32 | 0.0583 [0.0434, 0.0674] | 0.0859 [0.0625, 0.1055] | 0.0214 |
| ALiBi | 16 | 0.5971 [0.5725, 0.6222] | 0.8594 [0.8438, 0.8906] | 0.0424 |
| ALiBi | 32 | 0.4815 [0.4739, 0.4945] | 0.8307 [0.8203, 0.8477] | 0.0214 |
| RoPE | 16 | 0.7018 [0.6635, 0.7396] | 0.9193 [0.8984, 0.9453] | 0.0424 |
| RoPE | 32 | 0.1553 [0.1000, 0.2132] | 0.2161 [0.1406, 0.2891] | 0.0214 |

チャンスレベルは、位置 $m$ が因果マスクの下で参照可能な $m+1$ 個の Key に一様な重みを置く場合の値($\mathrm{chance}(L) = (1/L)\sum_{m=L}^{2L-1} 1/(m+1)$、$L=16$ で $0.0424$、$L=32$ で $0.0214$)。表 2(層・ヘッド別、$3$ seed 平均)では、いずれの条件も 1 層目(`layer=0`)はチャンスレベル前後にとどまり(最も高い RoPE の`layer=0`head 2 でも約 $1.35$ 倍)、copy に関わる集中は 2 層目(最終層)に現れる。

**表 1 を主指標とする理由**: 正弦波方式・RoPE は精度が seed 間でほぼ一定(6.2 節末尾の表)にもかかわらず、$L=16$ で重みが最大となる (層, ヘッド) は seed ごとに異なり(表 3。RoPE は $3$ seed すべてで異なるヘッドが最大で、最終層 head 0 単体の値も $[0.0993, 0.7396]$ と大きく振れる)、対照群(`none`)は $L=16$・$L=32$ とも $3$ seed すべてが同じ (層, ヘッド) で一致する。すなわち copy 機構がどの (層, ヘッド) に載るかは **seed にも依存しうる**。以降はこの依存を避けるため、seed ごとに重み最大の (層, ヘッド) を特定して集計した表 1 を主指標とする。

**正弦波方式・RoPE**: 学習長内($L=16$)では最大値・argmax 一致率ともにチャンスレベルの $13$〜$22$ 倍に達し、明確な集中を示した(ヒートマップでも SEP からのオフセット分ずれた対角線として観察できる)。学習長の 2 倍($L=32$)では正弦波方式が約 $2.7$〜$4.0$ 倍、RoPE が約 $7.3$〜$10.1$ 倍まで低下し、$L=16$ の水準(十数倍以上)から大きく後退した。RoPE の方が倍率を高く保っており、実験 B-2 の精度($L=32$ で正弦波方式 $0.238$・RoPE $0.311$、6.2 節)と整合する。

**ALiBi**: 学習長内で重み・一致率ともにチャンスレベルの約 $14$〜$20$ 倍に達し、正弦波方式・RoPE と同水準の集中を示した。$L=32$ でも約 $22$〜$39$ 倍とほとんど低下せず、正弦波方式・RoPE と対照的に集中を維持し、実験 B-2 の緩やかな精度低下($L=16$ で $0.967$、$L=32$ で $0.855$)と整合する。表 2 より`layer=1`の各ヘッドの重み($3$ seed 平均、$L=16$)は head 0 の $0.0138$ から head 3 の $0.4546$ まで、傾き $m_h$(3.7 節、$[0.25, 0.0625, 0.015625, 0.00390625]$)が小さいヘッドほど単調に大きい($L=32$ でも同じ順序)。チャンスレベル比では head 0 のみ約 $0.33$ 倍で **下回り**、他の 3 ヘッドは約 $6.4$〜$10.7$ 倍で明確に上回る。傾きが大きい head 0・head 1 が最大となる seed は $3$ seed × $2$ 系列長のどの組み合わせにもなく、常に head 2・head 3 のいずれかが最大となる(どちらかは seed に依存、表 3)。これは ALiBi の傾きが **学習可能パラメータを持たない固定値**(3.7 節)であることによる構造的な帰結であり、「傾きが小さいヘッドが長距離を担当する」という設計上の帰結は実測で裏付けられるが、そのうちどちらが最大かは seed に依存する。

**対照群との比較**: $L=16$ での対照群は重み・一致率ともにチャンスレベルの約 $3.7$〜$5.9$ 倍に達し、**明示的な位置エンコーディングがなくても参照位置への集中はチャンスレベルを明確に上回る**。これは 3.1 節の「因果マスク自体が暗黙の位置情報を与える」という議論の裏付けであり、実験 B-1 で対照群の精度が $0.634$ に達したこと(6.2 節)とも整合する。$L=32$ では正弦波方式が対照群と **ほぼ同水準**(argmax 一致率はむしろ下回る)まで低下し、実験 B-2 で正弦波方式が全外挿長で対照群を下回ったこと(6.2 節)と整合する。一方 RoPE は $L=32$ でも対照群を明確に上回るにもかかわらず精度では下回る(6.2 節)。すなわち **参照位置への集中度の高さと精度は単調に対応しない**。この乖離は本実験の結果だけからは断定できないが、本指標が重みが最大となる $1$ つの (層, ヘッド) しか見ていないこと、また copy には注目だけでなく参照した値を正しく出力に反映する処理(値の読み出し・変換)も必要であることが可能性として考えられる。

可視化に用いたモデルは代表として seed $0$ で学習したものであり、上記の定量指標は $3$ seed の平均値である点に注意(個別の seed の値は表 3 が示す通り大きく異なりうる)。

## 7. 結果・考察 / Results and Discussion

各実験の詳細は実験セクション末尾(6.1・6.2・6.3 節)に記載した。ここでは実験 B-1・B-2 と実験 C の結果を統合し、位置エンコーディング方式全体の位置づけを整理する。

**1. 学習長内での表現力(実験 B-1、3.2〜3.8 節)**: 位置エンコーディングを持たない対照群(精度 $0.634$)が最も低く、RoPE($0.988$)・正弦波方式($0.986$)が最も高い精度に達した($3$ seed でも明確に区別できる差、6.2 節)。これは Attention の内積構造(RoPE)または埋め込み(正弦波)に密な位置情報を与えられることの帰結だと考えられる。対照群の精度もチャンスレベル($\approx 0.167$)を大きく上回るのは因果マスクそのものが暗黙の位置情報を与えるためであり(3.1 節)、実験 C でも対照群の Attention 集中度がチャンスレベルの約 $3.7$〜$5.9$ 倍に達することでこの議論を挙動レベルで裏付けている(6.3 節)。

**2. 外挿性能(実験 B-2、3.4・3.7・3.8 節)**: 学習可能な絶対位置埋め込みは原理的に外挿できず(3.4 節)実験でもそのまま動作しなかった。RoPE・正弦波方式は学習長内の高い表現力とは対照的に $4$ つの外挿長すべてで対照群を下回り、ALiBi・Shaw et al. 方式は最も緩やかな劣化を示し全外挿長で対照群を上回り続けた(両者の優劣は $3$ seed では確定できない、6.2 節)。これは ALiBi が外挿を明示的な設計目標としている手法である([5])ことの帰結であり、**「ALiBi の方が優れた手法である」という結論には直結しない**(実験 B-1 は表現力、実験 B-2 は外挿性能という異なる軸を測っている)。

**3. 統合的な考察**: RoPE は 3.8.5 節で証明した通り相対位置を Attention の内積構造そのものに埋め込む設計であり、これが学習長内での高い精度(実験 B-1)の帰結だと考えられる一方、外挿性能は保証せず学習長を超えると急激に劣化する(実験 B-2)。この不足は、角周波数 $\theta_i$(3.8.4 節)のスケールを調整する **RoPE のスケーリング**(NTK-aware スケーリング・YaRN・位置補間、トピック 014「長文脈拡張」)で対処されることが多く、現代の多くの大規模言語モデルが RoPE を採用しつつ長文脈への対応を別の技術で行っている理由につながる。

**4. 想定との整合性**: 実験前の想定(「RoPE は表現力に優れる一方、素の状態では外挿に劣り ALiBi が優位」)はこの傾向を裏付けたが、想定より厳しく、**RoPE・正弦波方式は $1.5$ 倍長($L=24$)の時点ですでに対照群を下回った**(想定は $L=48$・$L=64$ あたりでの劣化だった、6.2 節)。暗黙の位置情報(因果マスク由来)は学習長を超えても意味を失わないのに対し、明示的な位置信号は学習長を超えると意味をなさなくなり有害に働きうるという 3.1 節の議論と整合する。T5 の外挿性能の低さは、学習中に一度も勾配を受けないバケットの存在(全 $32$ 個中 $8$ 個)で説明でき、学習長内の精度のばらつきも大きく(範囲 $[0.576, 0.862]$)、本タスク・本設定との相性が悪かったと考えられる。

**5. 実験 C の結果**: 想定外だった結果が $2$ 件、実験 B・3.1 節の議論を Attention 挙動レベルで裏付ける結果が $1$ 件得られた(6.3 節)。想定外の $1$ つ目は、精度が seed 間でほぼ一定な条件(正弦波方式・RoPE)でも copy 機構がどの (層, ヘッド) に載るかは seed に依存しうる点、$2$ つ目は ALiBi の copy 機構が傾きの小さい $2$ ヘッド(head 2・head 3)のいずれかに載るが、どちらに載るかは seed に依存する点である(3.7 節の「傾きが小さいヘッドが長距離を担う」という帰結自体は裏付けられた)。想定を裏付ける結果としては、$L=32$ で正弦波方式の Attention 集中度が対照群とほぼ同水準まで低下したことが実験 B-2 の精度低下と対応し、「意味をなさない明示的な位置信号が有害に働きうる」という主張を集中度の面からも裏付けている。ALiBi は学習長内・学習長超のいずれでも高い集中度を維持し、実験 B-2 の緩やかな精度低下と対応する結果が得られたが、この対応関係は観測された相関であり、集中度が精度を直接決定する因果関係までは実測値から結論づけられない。


## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/01_foundations/003_positional_encoding_rope.ipynb
