---
title: "長文脈拡張 / Long Context Extension(実装・実験編 3/4)"
---

この記事は後編(実装・実験編 3/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/015_long_context_extension-practice-2)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/015_long_context_extension-practice-4)。

### 6.3 共通の前提条件 P0: モデルカードの値の再現

モデルカードと同じ評価集合(008 の検証の部分全体を長さ $L$ の重ならない窓に区切ったもの、`make_evaluation_windows()`)・
同じ関数(`evaluate_bits_per_byte()`)で、スケーリングなしのモデルの bits-per-byte を計算する。


```python
set_rotary(base_model, "none", 1)
P0_BITS_PER_BYTE = evaluate_bits_per_byte(
    base_model, P0_WINDOWS, P0_MASK, len(validation_text.encode("utf-8")), device
)
precondition_status["P0"] = abs(P0_BITS_PER_BYTE - MODEL_CARD_BITS_PER_BYTE) <= P0_TOLERANCE
print(
    f"P0: b_P0 = {P0_BITS_PER_BYTE:.6f}、モデルカードの値 {MODEL_CARD_BITS_PER_BYTE}、"
    f"差 {P0_BITS_PER_BYTE - MODEL_CARD_BITS_PER_BYTE:+.6f}(許容 {P0_TOLERANCE})-> "
    f"{'成立' if precondition_status['P0'] else '不成立'}"
)
```

    P0: b_P0 = 1.668067、モデルカードの値 1.668067、差 +0.000000(許容 0.002)-> 成立


### 6.4 実験 A: 微調整なしでの外挿


```python
def three_way_verdict(value: float, sigma: float) -> str:
    if not (math.isfinite(value) and math.isfinite(sigma)):
        return "判定不能"
    if value - 2 * sigma > 0:
        return "支持"
    if value + 2 * sigma < 0:
        return "反証"
    return "判定不能"


def verdict_label(computed: str, preconditions: list[str]) -> str:
    return computed if all(precondition_status.get(p) for p in preconditions) else "前提不成立"


def bootstrap_difference_std(
    rows_a: list, rows_b: list, byte_counts: np.ndarray, seed: int, clusters: list
) -> float:
    # 記事を単位とする対応付きのクラスタブートストラップで、mean_k(比 a_k) - mean_k(比 b_k) の標本標準偏差を返す
    # (全行に同じ再標本。判定に使う新基準)。
    k = len(rows_a)
    samples = paired_cluster_bootstrap_ratio_of_sums(
        np.stack(list(rows_a) + list(rows_b)), byte_counts, clusters, BOOTSTRAP_RESAMPLES, seed
    )
    return float((samples[:, :k].mean(axis=1) - samples[:, k:].mean(axis=1)).std(ddof=1))


def window_bootstrap_difference_std(
    rows_a: list, rows_b: list, byte_counts: np.ndarray, seed: int
) -> float:
    # 評価窓を単位とする対応付きブートストラップでの同じ量(旧基準、判定に使わない診断量)。
    k = len(rows_a)
    samples = paired_bootstrap_ratio_of_sums(
        np.stack(list(rows_a) + list(rows_b)), byte_counts, BOOTSTRAP_RESAMPLES, seed
    )
    return float((samples[:, :k].mean(axis=1) - samples[:, k:].mean(axis=1)).std(ddof=1))


def print_per_article(label: str, rows_a: list, rows_b: list, byte_counts: np.ndarray) -> None:
    # 記事ごとの mean_k(比 a_k) - mean_k(比 b_k) と記事ごとの評価窓の数(診断量)。
    articles = np.array(WINDOW_ARTICLES)
    parts = []
    for article in dict.fromkeys(WINDOW_ARTICLES):
        mask = articles == article
        diff = np.mean(
            [
                bits_per_byte(a[mask], byte_counts[mask])
                - bits_per_byte(b[mask], byte_counts[mask])
                for a, b in zip(rows_a, rows_b, strict=True)
            ]
        )
        parts.append(f"{article}: {diff:+.4f}({int(mask.sum())} 窓)")
    print(f"(診断量)記事ごとの {label}: " + "、".join(parts))


# クラスタブートストラップの確認: 全クラスタが 1 単位なら、単位を復元抽出するブートストラップと完全に一致する
_rng = np.random.default_rng(0)
_num, _den = _rng.random((2, 9)), _rng.random(9) + 1
assert np.allclose(  # 乱数の抽出は同一。行列積の配置の違いによる丸め誤差のみを許す
    paired_cluster_bootstrap_ratio_of_sums(_num, _den, list(range(9)), 200, 5),
    paired_bootstrap_ratio_of_sums(_num, _den, 200, 5),
    rtol=1e-12,
    atol=0.0,
)
print(
    "クラスタが 1 単位ずつのとき、記事単位と窓単位のブートストラップが一致(相対誤差 1e-12 以内): OK"
)

_t0 = time.time()
for _factor in EVAL_LENGTH_FACTORS:
    for _method in EVAL_METHODS:
        base_losses(_method, _factor)
print(
    f"実験 A の評価: {len(EVAL_METHODS)} 手法 x {len(EVAL_LENGTH_FACTORS)} 長さ、{time.time() - _t0:.1f}s"
)

# --- 判定(4L、位置 [L, 4L)) ---
BYTES_A = range_bytes(TARGET_BYTES, L, MAIN_FACTOR * L)
BITS_A = {m: range_bits(base_losses(m, MAIN_FACTOR), L, MAIN_FACTOR * L) for m in EVAL_METHODS}
DELTA_A = bits_per_byte(BITS_A["position_interpolation"], BYTES_A) - bits_per_byte(
    BITS_A["yarn"], BYTES_A
)
SIGMA_A = bootstrap_difference_std(
    [BITS_A["position_interpolation"]], [BITS_A["yarn"]], BYTES_A, seed=0, clusters=WINDOW_ARTICLES
)
SIGMA_A_WINDOW = window_bootstrap_difference_std(
    [BITS_A["position_interpolation"]], [BITS_A["yarn"]], BYTES_A, seed=0
)
verdict_A = three_way_verdict(DELTA_A, SIGMA_A)
print(
    f"{_smoke_tag}Delta_A = b_interp - b_yarn = {DELTA_A:+.4f}, sigma_A = {SIGMA_A:.4f}, "
    f"[Delta - 2 sigma, Delta + 2 sigma] = [{DELTA_A - 2 * SIGMA_A:+.4f}, {DELTA_A + 2 * SIGMA_A:+.4f}] "
    f"-> 判定関数の結果: {verdict_A}"
)
print(f"(診断量)旧基準(評価窓を単位とするブートストラップ)の sigma_A = {SIGMA_A_WINDOW:.4f}")
print_per_article(
    "b_interp - b_yarn(4L、[L, 4L))", [BITS_A["position_interpolation"]], [BITS_A["yarn"]], BYTES_A
)

# --- 診断量: 手法 x 評価長の bits-per-byte([0, L) と [L, sL))と、2L・8L での同じ対比量 ---
print(f"\n{'手法':>24} | " + " | ".join(f"{f}L: [0,L) / [L,{f}L)" for f in EVAL_LENGTH_FACTORS))
for _method in EVAL_METHODS:
    _cells = []
    for _factor in EVAL_LENGTH_FACTORS:
        _losses = base_losses(_method, _factor)
        _in = bits_per_byte(range_bits(_losses, 0, L), range_bytes(TARGET_BYTES, 0, L))
        _out = bits_per_byte(
            range_bits(_losses, L, _factor * L), range_bytes(TARGET_BYTES, L, _factor * L)
        )
        _cells.append(f"{_in:.4f} / {_out:.4f}")
    print(f"{_method:>24} | " + " | ".join(f"{c:>17}" for c in _cells))
for _factor in EVAL_LENGTH_FACTORS:
    _bytes = range_bytes(TARGET_BYTES, L, _factor * L)
    _bi = range_bits(base_losses("position_interpolation", _factor), L, _factor * L)
    _by = range_bits(base_losses("yarn", _factor), L, _factor * L)
    _delta = bits_per_byte(_bi, _bytes) - bits_per_byte(_by, _bytes)
    _sigma = bootstrap_difference_std([_bi], [_by], _bytes, seed=_factor, clusters=WINDOW_ARTICLES)
    print(f"(診断量) {_factor}L: b_interp - b_yarn = {_delta:+.4f}, sigma = {_sigma:.4f}")

# --- 診断量: 位置の区間ごとの bits-per-byte の曲線 ---
fig, axes = plt.subplots(1, len(EVAL_LENGTH_FACTORS), figsize=(16, 4.2), sharey=True)
for _ax, _factor in zip(axes, EVAL_LENGTH_FACTORS, strict=True):
    _edges = list(range(0, _factor * L + 1, POSITION_BIN))
    for _method in EVAL_METHODS:
        _losses = base_losses(_method, _factor)
        _curve = [
            bits_per_byte(range_bits(_losses, a, c), range_bytes(TARGET_BYTES, a, c))
            for a, c in zip(_edges[:-1], _edges[1:], strict=True)
        ]
        _ax.plot(
            [(a + c) / 2 for a, c in zip(_edges[:-1], _edges[1:], strict=True)],
            _curve,
            label=_method,
        )
    _ax.axvline(L, color="gray", linestyle=":")
    _ax.set_title(f"{_plot_tag}eval length {_factor}L (s = {_factor})")
    _ax.set_xlabel("token position j")
axes[0].set_ylabel("bits-per-byte (bin width L/4)")
axes[-1].legend(fontsize=8)
fig.tight_layout()
plt.show()


# --- 診断量: Attention のエントロピーの位置依存性(温度の補正の有無) ---
def attention_entropy_by_position(model: GPTLanguageModel, windows: torch.Tensor) -> np.ndarray:
    # 各 Query の位置での注意の分布のエントロピー(nats)を、層・ヘッド・窓で平均する。形状 (層, 長さ)。
    sums = [None] * len(model.blocks)
    handles = []
    for layer_index, block in enumerate(model.blocks):

        def hook(module, inputs, output, i=layer_index):
            weights = output[1]  # (B, h, S, S)
            entropy = -torch.special.xlogy(weights, weights).sum(-1).mean(dim=(0, 1))  # (S,)
            sums[i] = entropy if sums[i] is None else sums[i] + entropy

        handles.append(block.self_attn.register_forward_hook(hook))
    try:
        model.eval()
        with torch.no_grad():
            for window in windows:
                model(window[None].to(device))
    finally:
        for handle in handles:
            handle.remove()
    return torch.stack(sums).float().cpu().numpy().astype(np.float64) / len(windows)


ENTROPY_METHODS = ("none", "ntk_by_parts", "yarn")
ENTROPY = {}
for _method in ENTROPY_METHODS:
    set_rotary(base_model, _method, MAX_FACTOR)
    ENTROPY[_method] = attention_entropy_by_position(base_model, EVAL_WINDOWS[:ENTROPY_WINDOWS])
_query_positions = np.arange(MAX_FACTOR * L)
fig, ax = plt.subplots(figsize=(7, 4))
for _method in ENTROPY_METHODS:
    ax.plot(_query_positions + 1, ENTROPY[_method].mean(axis=0), label=_method)
ax.plot(
    _query_positions + 1,
    np.log(_query_positions + 1),
    color="gray",
    linestyle=":",
    label="log(m + 1) (uniform)",
)
ax.axvline(L, color="gray", linestyle="--", linewidth=0.8)
ax.set_xscale("log", base=2)
ax.set_xlabel("query position m + 1")
ax.set_ylabel("attention entropy (nats, mean over layers/heads)")
ax.set_title(f"{_plot_tag}Attention entropy, eval length {MAX_FACTOR}L (s = {MAX_FACTOR})")
ax.legend(fontsize=8)
plt.show()
for _method in ENTROPY_METHODS:
    _mean = ENTROPY[_method].mean(axis=0)
    print(
        f"(診断量) エントロピー {_method:>13}: Query の位置 [L/2, L) の平均 {_mean[L // 2 : L].mean():.4f}、"
        f"[4L, 8L) の平均 {_mean[4 * L : 8 * L].mean():.4f} nats"
    )
```

    クラスタが 1 単位ずつのとき、記事単位と窓単位のブートストラップが一致(相対誤差 1e-12 以内): OK
    実験 A の評価: 5 手法 x 3 長さ、44.5s
    Delta_A = b_interp - b_yarn = +0.1587, sigma_A = 0.0049, [Delta - 2 sigma, Delta + 2 sigma] = [+0.1490, +0.1685] -> 判定関数の結果: 支持
    (診断量)旧基準(評価窓を単位とするブートストラップ)の sigma_A = 0.0026
    (診断量)記事ごとの b_interp - b_yarn(4L、[L, 4L)): 338: +0.1337(6 窓)、339: +0.1671(7 窓)、340: +0.1338(18 窓)、341: +0.1533(5 窓)、342: +0.1712(16 窓)、343: +0.1473(11 窓)、346: +0.1694(2 窓)、347: +0.1638(8 窓)、349: +0.1764(11 窓)、351: +0.1645(16 窓)、353: +0.1352(4 窓)、355: +0.1668(43 窓)
    
                          手法 | 2L: [0,L) / [L,2L) | 4L: [0,L) / [L,4L) | 8L: [0,L) / [L,8L)
                        none |   1.6552 / 1.8147 |   1.6552 / 1.9914 |   1.6552 / 2.1466
      position_interpolation |   1.7265 / 1.7290 |   1.8419 / 1.8570 |   1.9523 / 1.9862
                   ntk_aware |   1.6559 / 1.6813 |   1.6608 / 1.7531 |   1.6698 / 1.8810
                ntk_by_parts |   1.6661 / 1.6681 |   1.6926 / 1.7056 |   1.7237 / 1.7478
                        yarn |   1.6676 / 1.6666 |   1.6951 / 1.6983 |   1.7275 / 1.7277
    (診断量) 2L: b_interp - b_yarn = +0.0624, sigma = 0.0030
    (診断量) 4L: b_interp - b_yarn = +0.1587, sigma = 0.0049
    (診断量) 8L: b_interp - b_yarn = +0.2585, sigma = 0.0084



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/015_long_context_extension/output_43_1.png)
    



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/015_long_context_extension/output_43_2.png)
    


    (診断量) エントロピー          none: Query の位置 [L/2, L) の平均 2.0973、[4L, 8L) の平均 4.0129 nats
    (診断量) エントロピー  ntk_by_parts: Query の位置 [L/2, L) の平均 2.7479、[4L, 8L) の平均 3.7003 nats
    (診断量) エントロピー          yarn: Query の位置 [L/2, L) の平均 1.9767、[4L, 8L) の平均 2.6161 nats


### 6.5 学習率の較正と gradient clipping の閾値(実験 B)

**再実行**: 旧実行(コミット`982ff84`)では、YaRN の採用した学習率が拡張の上限(1 回)に達してもグリッドの下端
($10^{-4.5}$)のままで、前提条件 P-B2 が成立しなかった。拡張の上限を最大 3 回(下側は $10^{-5.5}$、上側は $10^{-1.5}$ まで)に
戻して再実行する。根拠・理由・変えないものは 6.1 節の実験 B の「再実行の記録」のとおりである。

採用の規則は`calibrate_learning_rate()`に切り出し、実際の学習の前に、合成した評価関数で拡張の経路(複数回の拡張の後に内点に
達する場合、上限まで拡張しても端のままの場合)を確かめる。

6.1 節の手順どおり、判定に使う 2 手法ごとに学習率のグリッドを本番と同じステップ数で掃引する。診断用の底の調整の条件には、
YaRN で採用した学習率と閾値を使う。


```python
def calibrate_learning_rate(evaluate, grid_k, max_expansions: int, bounds: tuple[int, int]) -> dict:
    # 学習率の較正の採用規則(6.1 節)。evaluate(k) は {"calibration_bits_per_byte", "all_finite", ...} を返す。
    # 採用した k がグリッドの端なら、同じ公比でその側に 1 水準拡張する(合計で最大 max_expansions 回、bounds の範囲内)。
    results: dict[int, dict] = {}
    grid = list(grid_k)
    expansions = 0
    while True:
        for k in grid:
            if k not in results:
                results[k] = evaluate(k)
        finite = {
            k: r
            for k, r in results.items()
            if r["all_finite"] and math.isfinite(r["calibration_bits_per_byte"])
        }
        assert finite, "全ての学習率で損失が非有限"
        best = min(finite, key=lambda k: finite[k]["calibration_bits_per_byte"])
        if expansions < max_expansions and best == min(grid):
            grid.insert(0, min(grid) - 1)
            expansions += 1
            continue
        if expansions < max_expansions and best == max(grid):
            grid.append(max(grid) + 1)
            expansions += 1
            continue
        break
    assert bounds[0] <= min(grid) and max(grid) <= bounds[1], "拡張が範囲を超えた"
    return {
        "best": best,
        "grid": grid,
        "results": results,
        "expansions": expansions,
        "is_interior": min(grid) < best < max(grid),
    }


# --- 拡張の経路の確認(合成した評価関数、学習はしない) ---
def _synthetic(minimum_k: float):
    return lambda k: {"calibration_bits_per_byte": (k - minimum_k) ** 2, "all_finite": True}


_bounds = LEARNING_RATE_K_BOUNDS
_case = calibrate_learning_rate(
    _synthetic(-9.2), LEARNING_RATE_GRID_K, MAX_GRID_EXPANSIONS, _bounds
)
assert (
    _case["expansions"] == 2 and _case["best"] == -9 and _case["is_interior"]
)  # 下側に 2 回拡張して内点に達する
_case = calibrate_learning_rate(_synthetic(-20), LEARNING_RATE_GRID_K, MAX_GRID_EXPANSIONS, _bounds)
assert (
    _case["expansions"] == MAX_GRID_EXPANSIONS
    and _case["best"] == _bounds[0]
    and not _case["is_interior"]
)
_case = calibrate_learning_rate(_synthetic(5), LEARNING_RATE_GRID_K, MAX_GRID_EXPANSIONS, _bounds)
assert (
    _case["expansions"] == MAX_GRID_EXPANSIONS
    and _case["best"] == _bounds[1]
    and not _case["is_interior"]
)
_case = calibrate_learning_rate(_synthetic(-7), LEARNING_RATE_GRID_K, MAX_GRID_EXPANSIONS, _bounds)
assert _case["expansions"] == 0 and _case["best"] == -7 and _case["is_interior"]  # 拡張しない
_case = calibrate_learning_rate(_synthetic(-9.2), LEARNING_RATE_GRID_K, 1, _bounds)
assert (
    _case["expansions"] == 1 and _case["best"] == -9 and not _case["is_interior"]
)  # 同じ関数でも上限 1 回では端のまま
print(
    f"拡張の経路(合成した評価関数): 2 回の拡張で内点 / 上限 {MAX_GRID_EXPANSIONS} 回で下端・上端のまま(内点でない、"
    f"P-B2 不成立になる)/ 拡張なし / 上限 1 回では端のまま: OK(範囲 10^({_bounds[0] / 2}) .. 10^({_bounds[1] / 2}))"
)

calibration_log: dict[str, dict] = {}
LEARNING_RATE: dict[str, float] = {}
CLIP_THRESHOLD: dict[str, float] = {}
_t0_calibration = time.time()
for _method in JUDGED_FINETUNE_METHODS:

    def _evaluate(k: int, method: str = _method) -> dict:
        rec = run_finetune(
            method, CALIBRATION_SEED, learning_rate_from_k(k), None, FINETUNE_STEPS, evaluate=False
        )
        print(
            f"[{method}] lr = {learning_rate_from_k(k):.2e}: 較正用の窓の [L, 4L) の bits-per-byte = "
            f"{rec['calibration_bits_per_byte']:.4f}"
        )
        return {
            "learning_rate": learning_rate_from_k(k),
            "calibration_bits_per_byte": rec["calibration_bits_per_byte"],
            "gradient_norm_quantile": float(np.quantile(rec["gradient_norm"], CLIP_QUANTILE)),
            "all_finite": rec["all_finite"],
        }

    _calibration = calibrate_learning_rate(
        _evaluate, LEARNING_RATE_GRID_K, MAX_GRID_EXPANSIONS, LEARNING_RATE_K_BOUNDS
    )
    _best, _results = _calibration["best"], _calibration["results"]
    calibration_log[_method] = {
        "grid": {
            f"{learning_rate_from_k(k):.6g}": _results[k]["calibration_bits_per_byte"]
            for k in sorted(_results)
        },
        "chosen_learning_rate": learning_rate_from_k(_best),
        "chosen_is_interior": _calibration["is_interior"],
        "expansions": _calibration["expansions"],
        "clip_threshold": _results[_best]["gradient_norm_quantile"],
    }
    LEARNING_RATE[_method] = learning_rate_from_k(_best)
    CLIP_THRESHOLD[_method] = _results[_best]["gradient_norm_quantile"]
    print(
        f"[{_method}] 採用: lr = {LEARNING_RATE[_method]:.2e}(内点: {_calibration['is_interior']}、"
        f"拡張 {_calibration['expansions']} 回)、gradient clipping の閾値 = {CLIP_THRESHOLD[_method]:.4f}"
    )
LEARNING_RATE["adjusted_base"] = LEARNING_RATE["yarn"]  # 診断用の条件は較正しない(6.1 節)
CLIP_THRESHOLD["adjusted_base"] = CLIP_THRESHOLD["yarn"]
print(
    f"[adjusted_base] YaRN の値を使う: lr = {LEARNING_RATE['adjusted_base']:.2e}、閾値 = {CLIP_THRESHOLD['adjusted_base']:.4f}"
)
print(f"較正の合計時間: {time.time() - _t0_calibration:.1f}s")
```

    拡張の経路(合成した評価関数): 2 回の拡張で内点 / 上限 3 回で下端・上端のまま(内点でない、P-B2 不成立になる)/ 拡張なし / 上限 1 回では端のまま: OK(範囲 10^(-5.5) .. 10^(-1.5))
    [position_interpolation] lr = 1.00e-04: 較正用の窓の [L, 4L) の bits-per-byte = 1.4407
    [position_interpolation] lr = 3.16e-04: 較正用の窓の [L, 4L) の bits-per-byte = 1.4424
    [position_interpolation] lr = 1.00e-03: 較正用の窓の [L, 4L) の bits-per-byte = 1.4727
    [position_interpolation] lr = 3.16e-05: 較正用の窓の [L, 4L) の bits-per-byte = 1.4739
    [position_interpolation] 採用: lr = 1.00e-04(内点: True、拡張 1 回)、gradient clipping の閾値 = 0.9976
    [yarn] lr = 1.00e-04: 較正用の窓の [L, 4L) の bits-per-byte = 1.4180
    [yarn] lr = 3.16e-04: 較正用の窓の [L, 4L) の bits-per-byte = 1.4327
    [yarn] lr = 1.00e-03: 較正用の窓の [L, 4L) の bits-per-byte = 1.4673
    [yarn] lr = 3.16e-05: 較正用の窓の [L, 4L) の bits-per-byte = 1.4161
    [yarn] lr = 1.00e-05: 較正用の窓の [L, 4L) の bits-per-byte = 1.4229
    [yarn] 採用: lr = 3.16e-05(内点: True、拡張 2 回)、gradient clipping の閾値 = 0.7955
    [adjusted_base] YaRN の値を使う: lr = 3.16e-05、閾値 = 0.7955
    較正の合計時間: 1076.1s


### 6.6 実験 B: 短い微調整の後の比較


```python
finetune_records: dict[str, list[dict]] = {m: [] for m in FINETUNE_METHODS}
_t0 = time.time()
for _method in FINETUNE_METHODS:
    for _seed in SEEDS if _method in JUDGED_FINETUNE_METHODS else DIAGNOSTIC_SEEDS:
        _rec = run_finetune(
            _method, _seed, LEARNING_RATE[_method], CLIP_THRESHOLD[_method], FINETUNE_STEPS
        )
        finetune_records[_method].append(_rec)
        print(
            f"[{_method}] seed={_seed}: 較正用の窓の損失 {_rec['initial_loss']:.4f} -> {_rec['final_loss']:.4f} nats、"
            f"clipping の発動率 {_rec['clip_trigger_ratio']:.2f}"
        )
print(f"本番の微調整の合計時間: {time.time() - _t0:.1f}s")

_all_finetune = [r for rs in finetune_records.values() for r in rs]
precondition_status["P-B1"] = all(r["final_loss"] < r["initial_loss"] for r in _all_finetune)
precondition_status["P-B2"] = all(r["all_finite"] for r in _all_finetune) and all(
    calibration_log[m]["chosen_is_interior"] for m in JUDGED_FINETUNE_METHODS
)
print(
    f"前提条件 P-B1(全条件・全シードで微調整後の損失 < 微調整前): {'成立' if precondition_status['P-B1'] else '不成立'}、"
    f"P-B2(損失が有限・学習率が内点): {'成立' if precondition_status['P-B2'] else '不成立'}"
)


def seed_bits(method: str, start: int, end: int, key: str = "eval_losses") -> list[np.ndarray]:
    return [range_bits(r[key], start, end) for r in finetune_records[method]]


def seed_paired_contrast(
    rows_a: list, rows_b: list, byte_counts: np.ndarray, seed: int, clusters: list
) -> dict:
    # シードで対応をとった差 d_k = b(a_k) - b(b_k) の平均と、sigma = sqrt(sigma_seed^2 + sigma_window^2)。
    # sigma_window は記事を単位とするクラスタブートストラップ(新基準)。旧基準の値も診断量として返す。
    differences = np.array(
        [
            bits_per_byte(a, byte_counts) - bits_per_byte(b, byte_counts)
            for a, b in zip(rows_a, rows_b, strict=True)
        ]
    )
    sigma_seed = float(differences.std(ddof=1) / math.sqrt(len(differences)))
    sigma_window = bootstrap_difference_std(rows_a, rows_b, byte_counts, seed, clusters)
    sigma_window_old = window_bootstrap_difference_std(rows_a, rows_b, byte_counts, seed)
    return {
        "delta": float(differences.mean()),
        "differences": differences.tolist(),
        "sigma_seed": sigma_seed,
        "sigma_window": sigma_window,
        "sigma": math.sqrt(sigma_seed**2 + sigma_window**2),
        "sigma_window_old": sigma_window_old,
        "sigma_old": math.sqrt(sigma_seed**2 + sigma_window_old**2),
    }


BYTES_B = range_bytes(TARGET_BYTES, L, FINETUNE_FACTOR * L)
CONTRAST_B = seed_paired_contrast(
    seed_bits("position_interpolation", L, FINETUNE_FACTOR * L),
    seed_bits("yarn", L, FINETUNE_FACTOR * L),
    BYTES_B,
    seed=10,
    clusters=WINDOW_ARTICLES,
)
verdict_B = three_way_verdict(CONTRAST_B["delta"], CONTRAST_B["sigma"])
_yarn_values = [bits_per_byte(b, BYTES_B) for b in seed_bits("yarn", L, FINETUNE_FACTOR * L)]
NOISE_FLOOR_B = float(np.std(_yarn_values, ddof=1))
print(
    f"{_smoke_tag}d_k = {np.round(CONTRAST_B['differences'], 4).tolist()}、Delta_B = {CONTRAST_B['delta']:+.4f}、"
    f"sigma_seed = {CONTRAST_B['sigma_seed']:.4f}、sigma_window = {CONTRAST_B['sigma_window']:.4f}、"
    f"sigma_B = {CONTRAST_B['sigma']:.4f} -> 判定関数の結果: {verdict_B}"
)
print(
    f"{_smoke_tag}ノイズ床(YaRN + 微調整の bits-per-byte のシード間の標本標準偏差): {NOISE_FLOOR_B:.4f}"
)
print(
    f"(診断量)旧基準の sigma_window = {CONTRAST_B['sigma_window_old']:.4f}、"
    f"sigma_B = {CONTRAST_B['sigma_old']:.4f}"
)
print_per_article(
    "シード平均の b_interp - b_yarn(微調整後、[L, 4L))",
    seed_bits("position_interpolation", L, FINETUNE_FACTOR * L),
    seed_bits("yarn", L, FINETUNE_FACTOR * L),
    BYTES_B,
)

# --- 診断量 ---
print(f"\n{'手法 + 微調整':>24} | {'[0, L)':>15} | {'[L, 4L)':>15}(シード平均 ± 標本標準偏差)")
for _method in FINETUNE_METHODS:
    _cells = []
    for _a, _c in ((0, L), (L, FINETUNE_FACTOR * L)):
        _values = [
            bits_per_byte(b, range_bytes(TARGET_BYTES, _a, _c)) for b in seed_bits(_method, _a, _c)
        ]
        _cells.append(f"{np.mean(_values):.4f} ± {np.std(_values, ddof=1):.4f}")
    print(f"{_method:>24} | " + " | ".join(f"{c:>15}" for c in _cells))
for _method in ("none", "position_interpolation", "yarn"):
    _losses = base_losses(_method, FINETUNE_FACTOR)
    print(
        f"(比較)微調整なし {_method:>22}: [0, L) {bits_per_byte(range_bits(_losses, 0, L), range_bytes(TARGET_BYTES, 0, L)):.4f}、"
        f"[L, 4L) {bits_per_byte(range_bits(_losses, L, 4 * L), BYTES_B):.4f}"
    )
fig, ax = plt.subplots(figsize=(7, 4))
for _method in FINETUNE_METHODS:
    for _i, _r in enumerate(finetune_records[_method]):
        ax.plot(
            _r["train_loss"],
            color=f"C{FINETUNE_METHODS.index(_method)}",
            alpha=0.5,
            label=_method if _i == 0 else None,
        )
ax.set_xlabel("step")
ax.set_ylabel("train loss (nats / token)")
ax.set_title(f"{_plot_tag}Fine-tuning at length {FINETUNE_FACTOR}L")
ax.legend(fontsize=8)
plt.show()
```

    [position_interpolation] seed=0: 較正用の窓の損失 4.4623 -> 3.8831 nats、clipping の発動率 0.11
    [position_interpolation] seed=1: 較正用の窓の損失 4.4623 -> 3.8863 nats、clipping の発動率 0.12
    [position_interpolation] seed=2: 較正用の窓の損失 4.4623 -> 3.8821 nats、clipping の発動率 0.12
    [position_interpolation] seed=3: 較正用の窓の損失 4.4623 -> 3.8814 nats、clipping の発動率 0.12
    [position_interpolation] seed=4: 較正用の窓の損失 4.4623 -> 3.8901 nats、clipping の発動率 0.10
    [yarn] seed=0: 較正用の窓の損失 3.9248 -> 3.8216 nats、clipping の発動率 0.12
    [yarn] seed=1: 較正用の窓の損失 3.9248 -> 3.8228 nats、clipping の発動率 0.12
    [yarn] seed=2: 較正用の窓の損失 3.9248 -> 3.8210 nats、clipping の発動率 0.11
    [yarn] seed=3: 較正用の窓の損失 3.9248 -> 3.8224 nats、clipping の発動率 0.09
    [yarn] seed=4: 較正用の窓の損失 3.9248 -> 3.8245 nats、clipping の発動率 0.12
    [adjusted_base] seed=0: 較正用の窓の損失 4.0251 -> 3.8299 nats、clipping の発動率 0.17
    [adjusted_base] seed=1: 較正用の窓の損失 4.0251 -> 3.8314 nats、clipping の発動率 0.18
    本番の微調整の合計時間: 1853.9s
    前提条件 P-B1(全条件・全シードで微調整後の損失 < 微調整前): 成立、P-B2(損失が有限・学習率が内点): 成立
    d_k = [0.0071, 0.0071, 0.0072, 0.0075, 0.0071]、Delta_B = +0.0072、sigma_seed = 0.0001、sigma_window = 0.0012、sigma_B = 0.0012 -> 判定関数の結果: 支持
    ノイズ床(YaRN + 微調整の bits-per-byte のシード間の標本標準偏差): 0.0003
    (診断量)旧基準の sigma_window = 0.0007、sigma_B = 0.0007
    (診断量)記事ごとの シード平均の b_interp - b_yarn(微調整後、[L, 4L)): 338: +0.0122(6 窓)、339: +0.0039(7 窓)、340: +0.0054(18 窓)、341: +0.0015(5 窓)、342: +0.0147(16 窓)、343: +0.0038(11 窓)、346: -0.0211(2 窓)、347: +0.0104(8 窓)、349: +0.0060(11 窓)、351: +0.0038(16 窓)、353: +0.0187(4 窓)、355: +0.0077(43 窓)
    
                    手法 + 微調整 |          [0, L) |         [L, 4L)(シード平均 ± 標本標準偏差)
      position_interpolation | 1.6758 ± 0.0002 | 1.6737 ± 0.0003
                        yarn | 1.6654 ± 0.0002 | 1.6665 ± 0.0003
               adjusted_base | 1.6592 ± 0.0003 | 1.6708 ± 0.0003
    (比較)微調整なし                   none: [0, L) 1.6552、[L, 4L) 1.9914
    (比較)微調整なし position_interpolation: [0, L) 1.8419、[L, 4L) 1.8570
    (比較)微調整なし                   yarn: [0, L) 1.6951、[L, 4L) 1.6983



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/015_long_context_extension/output_47_1.png)
    


### 6.7 実験 C: 伸ばした文脈を実際に使っているか


```python
precondition_status["P-C1"] = WINDOWS_WITHIN_ARTICLE
_start_c, _end_c = (FINETUNE_FACTOR - 1) * L, FINETUNE_FACTOR * L
BYTES_C = range_bytes(TARGET_BYTES, _start_c, _end_c)


def truncated_bits(losses: np.ndarray) -> np.ndarray:
    return losses.sum(axis=1) / LOG2


CONTRAST_C = seed_paired_contrast(
    [truncated_bits(r["truncated_losses"]) for r in finetune_records["yarn"]],
    seed_bits("yarn", _start_c, _end_c),
    BYTES_C,
    seed=20,
    clusters=WINDOW_ARTICLES,
)
verdict_C = three_way_verdict(CONTRAST_C["delta"], CONTRAST_C["sigma"])
print(
    f"前提条件 P-C1(全評価窓が 1 つの記事の中): {'成立' if precondition_status['P-C1'] else '不成立'}"
)
print(
    f"{_smoke_tag}c_k = {np.round(CONTRAST_C['differences'], 4).tolist()}、Delta_C = {CONTRAST_C['delta']:+.4f}、"
    f"sigma_seed = {CONTRAST_C['sigma_seed']:.4f}、sigma_window = {CONTRAST_C['sigma_window']:.4f}、"
    f"sigma_C = {CONTRAST_C['sigma']:.4f} -> 判定関数の結果: {verdict_C}"
)
print(
    f"(診断量)旧基準の sigma_window = {CONTRAST_C['sigma_window_old']:.4f}、"
    f"sigma_C = {CONTRAST_C['sigma_old']:.4f}"
)
print_per_article(
    "シード平均の b_truncated - b_full(YaRN + 微調整、[3L, 4L))",
    [truncated_bits(r["truncated_losses"]) for r in finetune_records["yarn"]],
    seed_bits("yarn", _start_c, _end_c),
    BYTES_C,
)

# --- 診断量: 微調整なしの各手法(s = 4)と、微調整した位置補間 ---
print("\n(診断量)位置 [3L, 4L) の b_truncated - b_full")
for _method in ("none", "position_interpolation", "ntk_aware", "yarn"):

    def _compute(m=_method):
        set_rotary(base_model, m, FINETUNE_FACTOR)
        return truncated_token_losses(base_model, EVAL_WINDOWS, _start_c, _end_c, L)

    _truncated = cached_losses(("base_truncated", _method, FINETUNE_FACTOR), _compute)
    _bits_full = range_bits(base_losses(_method, FINETUNE_FACTOR), _start_c, _end_c)
    _bits_truncated = truncated_bits(_truncated)
    _delta = bits_per_byte(_bits_truncated, BYTES_C) - bits_per_byte(_bits_full, BYTES_C)
    _sigma = bootstrap_difference_std(
        [_bits_truncated], [_bits_full], BYTES_C, seed=21, clusters=WINDOW_ARTICLES
    )
    print(
        f"  微調整なし {_method:>22}: full {bits_per_byte(_bits_full, BYTES_C):.4f}、"
        f"truncated {bits_per_byte(_bits_truncated, BYTES_C):.4f}、差 {_delta:+.4f}(sigma {_sigma:.4f})"
    )
_interp_diagnostic = [
    r for r in finetune_records["position_interpolation"] if "truncated_losses" in r
]
_contrast_interp = seed_paired_contrast(
    [truncated_bits(r["truncated_losses"]) for r in _interp_diagnostic],
    [range_bits(r["eval_losses"], _start_c, _end_c) for r in _interp_diagnostic],
    BYTES_C,
    seed=22,
    clusters=WINDOW_ARTICLES,
)
print(
    f"  位置補間 + 微調整(シード {[r['seed'] for r in _interp_diagnostic]}): 差のシード平均 {_contrast_interp['delta']:+.4f}(sigma {_contrast_interp['sigma']:.4f})、"
    f"シードごと {np.round(_contrast_interp['differences'], 4).tolist()}"
)
```

    前提条件 P-C1(全評価窓が 1 つの記事の中): 成立
    c_k = [0.0024, 0.0025, 0.0024, 0.0023, 0.0024]、Delta_C = +0.0024、sigma_seed = 0.0000、sigma_window = 0.0007、sigma_C = 0.0007 -> 判定関数の結果: 支持
    (診断量)旧基準の sigma_window = 0.0006、sigma_C = 0.0006
    (診断量)記事ごとの シード平均の b_truncated - b_full(YaRN + 微調整、[3L, 4L)): 338: +0.0091(6 窓)、339: -0.0018(7 窓)、340: +0.0038(18 窓)、341: -0.0014(5 窓)、342: +0.0001(16 窓)、343: +0.0005(11 窓)、346: +0.0066(2 窓)、347: +0.0026(8 窓)、349: +0.0000(11 窓)、351: +0.0033(16 窓)、353: -0.0030(4 窓)、355: +0.0038(43 窓)
    
    (診断量)位置 [3L, 4L) の b_truncated - b_full
      微調整なし                   none: full 2.1297、truncated 1.6806、差 -0.4491(sigma 0.0108)
      微調整なし position_interpolation: full 1.8797、truncated 1.8688、差 -0.0109(sigma 0.0014)
      微調整なし              ntk_aware: full 1.8801、truncated 1.6851、差 -0.1950(sigma 0.0091)
      微調整なし                   yarn: full 1.7153、truncated 1.7190、差 +0.0037(sigma 0.0011)
      位置補間 + 微調整(シード [0, 1]): 差のシード平均 +0.0014(sigma 0.0008)、シードごと [0.0013, 0.0015]


### 6.8 実験 D: 静的なスケーリングの短い系列でのコスト


```python
CHUNKS = EVAL_WINDOWS.reshape(-1, L)  # (窓 x 8, L): 各窓を 8 個の長さ L の区間に分ける
_chunk_target_bytes = TOKEN_BYTE_LENGTHS[CHUNKS[:, 1:].numpy()]
BYTES_D = (
    _chunk_target_bytes.reshape(NUM_EVAL_WINDOWS, MAX_FACTOR, L - 1)
    .sum(axis=(1, 2))
    .astype(np.float64)
)


def chunk_losses(method: str, scale: float) -> np.ndarray:
    def compute():
        set_rotary(base_model, method, scale)
        return token_losses(base_model, CHUNKS)

    return cached_losses(("base_chunks", method, scale), compute)


def chunk_bits_per_window(losses: np.ndarray) -> np.ndarray:
    # 窓(長さ 8L)を単位として、その 8 区間の損失の合計(ビット)
    return losses.reshape(NUM_EVAL_WINDOWS, MAX_FACTOR, L - 1).sum(axis=(1, 2)) / LOG2


BITS_D = {
    "none": chunk_bits_per_window(chunk_losses("none", 1)),
    "yarn": chunk_bits_per_window(chunk_losses("yarn", MAIN_FACTOR)),
    "position_interpolation": chunk_bits_per_window(
        chunk_losses("position_interpolation", MAIN_FACTOR)
    ),
}
DELTA_D = bits_per_byte(BITS_D["yarn"], BYTES_D) - bits_per_byte(BITS_D["none"], BYTES_D)
SIGMA_D = bootstrap_difference_std(
    [BITS_D["yarn"]], [BITS_D["none"]], BYTES_D, seed=30, clusters=WINDOW_ARTICLES
)
SIGMA_D_WINDOW = window_bootstrap_difference_std(
    [BITS_D["yarn"]], [BITS_D["none"]], BYTES_D, seed=30
)
verdict_D = three_way_verdict(DELTA_D, SIGMA_D)
print(
    f"{_smoke_tag}Delta_D = b_yarn(s=4) - b_none = {DELTA_D:+.4f}, sigma_D = {SIGMA_D:.4f} "
    f"-> 判定関数の結果: {verdict_D}"
)
print(f"(診断量)旧基準(評価窓を単位とするブートストラップ)の sigma_D = {SIGMA_D_WINDOW:.4f}")
print_per_article("b_yarn(s=4) - b_none(長さ L)", [BITS_D["yarn"]], [BITS_D["none"]], BYTES_D)

# --- dynamic YaRN は長さ L の区間でスケーリングなしと完全に一致する(アサーション) ---
_dynamic_losses = chunk_losses("dynamic_yarn", 1)
assert np.array_equal(_dynamic_losses, chunk_losses("none", 1)), (
    "dynamic YaRN が l <= L でスケーリングなしと一致しない"
)
set_rotary(base_model, "dynamic_yarn", 1)
with torch.no_grad():
    _logits_dynamic = base_model(CHUNKS[:4].to(device))
set_rotary(base_model, "none", 1)
with torch.no_grad():
    _logits_none = base_model(CHUNKS[:4].to(device))
assert torch.equal(_logits_dynamic, _logits_none)
print("dynamic YaRN: 長さ L の全区間の損失と logits がスケーリングなしと完全一致: OK")

# --- 診断量 ---
_delta_interp = bits_per_byte(BITS_D["position_interpolation"], BYTES_D) - bits_per_byte(
    BITS_D["none"], BYTES_D
)
print(
    f"(診断量)長さ L: b_none = {bits_per_byte(BITS_D['none'], BYTES_D):.4f}、b_yarn(s=4) = "
    f"{bits_per_byte(BITS_D['yarn'], BYTES_D):.4f}、b_interp(s=4) = {bits_per_byte(BITS_D['position_interpolation'], BYTES_D):.4f}"
    f"(b_interp - b_none = {_delta_interp:+.4f}、sigma "
    f"{bootstrap_difference_std([BITS_D['position_interpolation']], [BITS_D['none']], BYTES_D, seed=31, clusters=WINDOW_ARTICLES):.4f})"
)
print(
    f"(診断量)P0 との比較: 記事の中の長さ L の区間でのスケーリングなし {bits_per_byte(BITS_D['none'], BYTES_D):.4f}、"
    f"P0 の評価集合 {P0_BITS_PER_BYTE:.4f}"
)


def dynamic_prefix_losses() -> np.ndarray:
    # 位置 j in [(c-1)L, cL) を、先頭から cL トークンの系列(dynamic により s = c)で予測する(c = 1..4)。
    set_rotary(base_model, "dynamic_yarn", 1)
    out = np.empty((NUM_EVAL_WINDOWS, FINETUNE_FACTOR * L - 1))
    for c in range(1, FINETUNE_FACTOR + 1):
        losses = token_losses(base_model, EVAL_WINDOWS[:, : c * L])
        lo, hi = max((c - 1) * L, 1) - 1, c * L - 1
        out[:, lo:hi] = losses[:, lo:hi]
    return out


_dynamic_prefix = cached_losses(("base_dynamic_prefix", FINETUNE_FACTOR), dynamic_prefix_losses)
assert np.allclose(
    _dynamic_prefix[:, : L - 1], base_losses("none", FINETUNE_FACTOR)[:, : L - 1], atol=1e-4
)
_bytes = range_bytes(TARGET_BYTES, L, FINETUNE_FACTOR * L)
_bits_dynamic = range_bits(_dynamic_prefix, L, FINETUNE_FACTOR * L)
_bits_static = range_bits(base_losses("yarn", FINETUNE_FACTOR), L, FINETUNE_FACTOR * L)
print(
    f"(診断量)位置 [L, 4L): dynamic YaRN(前方からの評価){bits_per_byte(_bits_dynamic, _bytes):.4f}、"
    f"静的な YaRN(s=4){bits_per_byte(_bits_static, _bytes):.4f}、差 "
    f"{bits_per_byte(_bits_dynamic, _bytes) - bits_per_byte(_bits_static, _bytes):+.4f}"
    f"(sigma {bootstrap_difference_std([_bits_dynamic], [_bits_static], _bytes, seed=32, clusters=WINDOW_ARTICLES):.4f})"
)
```

    Delta_D = b_yarn(s=4) - b_none = +0.0407, sigma_D = 0.0018 -> 判定関数の結果: 支持
    (診断量)旧基準(評価窓を単位とするブートストラップ)の sigma_D = 0.0008
    (診断量)記事ごとの b_yarn(s=4) - b_none(長さ L): 338: +0.0418(6 窓)、339: +0.0447(7 窓)、340: +0.0382(18 窓)、341: +0.0433(5 窓)、342: +0.0369(16 窓)、343: +0.0370(11 窓)、346: -0.0058(2 窓)、347: +0.0437(8 窓)、349: +0.0453(11 窓)、351: +0.0408(16 窓)、353: +0.0159(4 窓)、355: +0.0457(43 窓)
    dynamic YaRN: 長さ L の全区間の損失と logits がスケーリングなしと完全一致: OK
    (診断量)長さ L: b_none = 1.6752、b_yarn(s=4) = 1.7160、b_interp(s=4) = 1.8619(b_interp - b_none = +0.1867、sigma 0.0049)
    (診断量)P0 との比較: 記事の中の長さ L の区間でのスケーリングなし 1.6752、P0 の評価集合 1.6681
    (診断量)位置 [L, 4L): dynamic YaRN(前方からの評価)1.6854、静的な YaRN(s=4)1.6983、差 -0.0129(sigma 0.0006)


### 6.9 不変条件のアサーションと`SMOKE_TEST`の配線

- 全ての微調整の記録で、評価窓(ハッシュ)・ステップ数・学習履歴の長さが同一であり、トークン数がエポックの上限を超えない。
- 判定に使う量の分母(バイト数)は、条件によらず評価窓のトークンだけから計算した同じ配列である。実験 A・B で共通の範囲の
  分母が一致すること。
- 再利用した損失(`LOSS_CACHE`)が、キャッシュを経由せずに計算し直した値と完全に一致すること(冗長な再計算の排除の前後で
  数値が変わらない)。
- ベースモデルの重みが、全ての実験の後も Hub から読み込んだ重みと完全に一致すること(微調整は複製に対して行った)。
- `SMOKE_TEST`の配線: 実際に使った評価窓の数・シード数・ステップ数・反復回数が、5.3 節の実効水準と一致すること。


```python
for _r in _all_finetune:
    assert _r["eval_windows_hash"] == EVAL_WINDOWS_HASH, "評価窓が条件間で異なる"
    assert _r["num_steps"] == FINETUNE_STEPS == _r["history_length"], "ステップ数が条件間で異なる"
    assert _r["num_steps"] * TOKENS_PER_STEP <= len(finetune_ids), "エポックの上限を超えた"
    assert _r["eval_losses"].shape == (NUM_EVAL_WINDOWS, FINETUNE_FACTOR * L - 1)
for _r in _all_finetune:  # 切り詰めた評価は YaRN の全シードと位置補間の診断用のシードのみ
    _expected_truncated = _r["method"] == "yarn" or (
        _r["method"] == "position_interpolation" and _r["seed"] in DIAGNOSTIC_SEEDS
    )
    assert ("truncated_losses" in _r) == _expected_truncated
    if _expected_truncated:
        assert _r["truncated_losses"].shape == (NUM_EVAL_WINDOWS, L)
assert np.array_equal(BYTES_A, BYTES_B), "実験 A・B の分母が一致しない"
assert all(
    base_losses(m, f).shape == (NUM_EVAL_WINDOWS, f * L - 1)
    for m in EVAL_METHODS
    for f in EVAL_LENGTH_FACTORS
)
print(
    f"微調整の全 {len(_all_finetune)} 回: 評価窓・ステップ数({FINETUNE_STEPS})・エポックの上限・形状: OK"
)

# --- 冗長な再計算の排除の前後で数値が変わらないこと ---
set_rotary(base_model, "yarn", MAIN_FACTOR)
assert np.array_equal(
    token_losses(base_model, EVAL_WINDOWS[:, : MAIN_FACTOR * L]), base_losses("yarn", MAIN_FACTOR)
)
set_rotary(base_model, "none", 1)
assert np.array_equal(token_losses(base_model, CHUNKS), chunk_losses("none", 1))
print(f"キャッシュした損失({len(LOSS_CACHE)} 件)を計算し直した値と完全一致: OK")

# --- ベースモデルが変化していないこと ---
_hub_state = torch.load(MODEL_STATE_PATH, map_location="cpu")
assert all(torch.equal(v.cpu(), _hub_state[k]) for k, v in base_model.state_dict().items())
print("ベースモデルの重みが Hub から読み込んだ重みと完全一致: OK")

# --- SMOKE_TEST の配線 ---
_expected_windows = (
    len(EVAL_WINDOWS_ALL)
    if MAX_EVAL_WINDOWS is None
    else min(MAX_EVAL_WINDOWS, len(EVAL_WINDOWS_ALL))
)
assert NUM_EVAL_WINDOWS == _expected_windows == len(BYTES_A) == len(BYTES_D)
assert all([r["seed"] for r in finetune_records[m]] == list(SEEDS) for m in JUDGED_FINETUNE_METHODS)
assert [r["seed"] for r in finetune_records["adjusted_base"]] == list(DIAGNOSTIC_SEEDS)
assert NUM_EVAL_ARTICLES == len(set(WINDOW_ARTICLES)) >= 2
assert all(
    len(calibration_log[m]["grid"]) <= len(LEARNING_RATE_GRID_K) + MAX_GRID_EXPANSIONS
    for m in calibration_log
)
assert len(SEEDS) == CFG["NUM_SEEDS"] and CFG["FINETUNE_STEPS"] == FINETUNE_STEPS
assert len(calibration_windows) == CFG["CALIBRATION_WINDOWS"]
assert all(len(ENTROPY[m]) == NUM_LAYERS for m in ENTROPY_METHODS)
_probe = paired_bootstrap_ratio_of_sums(np.stack([BITS_A["yarn"]]), BYTES_A, BOOTSTRAP_RESAMPLES, 0)
assert _probe.shape == (CFG["BOOTSTRAP_RESAMPLES"], 1)
print(
    f"SMOKE_TEST の配線: 水準 {CURRENT_LEVEL_NAME!r}(評価窓 {NUM_EVAL_WINDOWS}・記事 {NUM_EVAL_ARTICLES} 本、"
    f"シード {len(SEEDS)}(診断用 {len(DIAGNOSTIC_SEEDS)})、"
    f"ステップ {FINETUNE_STEPS}、較正用の窓 {len(calibration_windows)}、反復 {BOOTSTRAP_RESAMPLES})が実効水準と一致: OK"
)
```

    微調整の全 12 回: 評価窓・ステップ数(400)・エポックの上限・形状: OK
    キャッシュした損失(24 件)を計算し直した値と完全一致: OK
    ベースモデルの重みが Hub から読み込んだ重みと完全一致: OK
    SMOKE_TEST の配線: 水準 'prod'(評価窓 147・記事 12 本、シード 5(診断用 2)、ステップ 400、較正用の窓 32、反復 10000)が実効水準と一致: OK


### 6.10 判定結果の一覧

前提条件が 1 つでも不成立の実験は、判定関数の結果に関わらず「前提不成立」とする(判定不能とは区別する)。判定関数の結果は
参考として別欄に残す。標準偏差は記事を単位とするクラスタブートストラップによる(改訂後の基準)。旧基準(評価窓を単位とする
ブートストラップ)による標準偏差も、判定に使わない参考値として並べる。


```python
_experiment_preconditions = {
    "A": ["P0"],
    "B": ["P0", "P-B1", "P-B2"],
    "C": ["P0", "P-B1", "P-B2", "P-C1"],
    "D": ["P0"],
}
_verdicts = [
    ("A", f"Delta={DELTA_A:+.4f}、sigma={SIGMA_A:.4f}(旧基準 {SIGMA_A_WINDOW:.4f})", verdict_A),
    (
        "B",
        f"Delta={CONTRAST_B['delta']:+.4f}、sigma={CONTRAST_B['sigma']:.4f}(旧基準 {CONTRAST_B['sigma_old']:.4f})",
        verdict_B,
    ),
    (
        "C",
        f"Delta={CONTRAST_C['delta']:+.4f}、sigma={CONTRAST_C['sigma']:.4f}(旧基準 {CONTRAST_C['sigma_old']:.4f})",
        verdict_C,
    ),
    ("D", f"Delta={DELTA_D:+.4f}、sigma={SIGMA_D:.4f}(旧基準 {SIGMA_D_WINDOW:.4f})", verdict_D),
]
print(
    f"{_smoke_tag}実験 | 対比量(sigma は記事単位、括弧内は参考) | 前提条件の成否 | 判定関数の結果(参考) | 最終判定"
)
for _name, _stat, _computed in _verdicts:
    _pre = ", ".join(f"{p}={precondition_status.get(p)}" for p in _experiment_preconditions[_name])
    print(
        f"{_smoke_tag}{_name} | {_stat} | {_pre} | {_computed} | "
        f"{verdict_label(_computed, _experiment_preconditions[_name])}"
    )
```

    実験 | 対比量(sigma は記事単位、括弧内は参考) | 前提条件の成否 | 判定関数の結果(参考) | 最終判定
    A | Delta=+0.1587、sigma=0.0049(旧基準 0.0026) | P0=True | 支持 | 支持
    B | Delta=+0.0072、sigma=0.0012(旧基準 0.0007) | P0=True, P-B1=True, P-B2=True | 支持 | 支持
    C | Delta=+0.0024、sigma=0.0007(旧基準 0.0006) | P0=True, P-B1=True, P-B2=True, P-C1=True | 支持 | 支持
    D | Delta=+0.0407、sigma=0.0018(旧基準 0.0008) | P0=True | 支持 | 支持




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/03_efficient_training/015_long_context_extension.ipynb
