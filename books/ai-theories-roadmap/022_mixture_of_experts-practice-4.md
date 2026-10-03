---
title: "Mixture of Experts(MoE) / Mixture of Experts(実装・実験編 4/5)"
---

この記事は後編(実装・実験編 4/5)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/022_mixture_of_experts-practice-3)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/022_mixture_of_experts-practice-5)。

### 6.6 本番の学習と評価

選ばれた計画の条件とシードをすべて学習する。D と M8 は実験 A・B・C で共有し、学習は 1 回だけ行う。評価は評価集合で、途中の評価の
位置ごとに行う。


```python
_t0_main = time.time()
RUNS = {}  # (条件, シード) -> 記録
for _c in RUN_ORDER:
    for _s in range(NUM_SEEDS[_c]):
        RUNS[(_c, _s)] = train_run(
            _c, _s, LEARNING_RATE[_c], NUM_STEPS, EVAL_WINDOWS, EVAL_WINDOW_BYTES, EVAL_STEPS,
            collect_tokens=(_c, _s) == OBSERVATION_RUN, measure_train_probe=True,
        )
        _r = RUNS[(_c, _s)]
        _line = (
            f"{_c} シード {_s}: bits-per-byte {_r['bits_per_byte']:.4f}(学習用の部分の窓 {_r['train_bits_per_byte']:.4f})、最後の区間の訓練損失 {_r['final_train_loss']:.3f}"
            f"(ln V の {_r['final_train_loss'] / UNIFORM_LOSS:.3f} 倍)、clipping の発動 {_r['clip_rate']:.2f}、更新のスキップ {_r['skipped_steps']}、"
            f"{_r['seconds']:.0f} 秒"
        )
        if "routing" in _r:
            _line += (
                f"、h {_r['routing']['h']:.4f}、最大の負荷 {max(_r['routing']['max_load']):.3f}、破棄 {max(_r['routing']['dropped']):.4f}、"
                f"確率の最大値 {np.mean(_r['routing']['max_probability']):.3f}"
            )
        print(_line)
MAIN_SECONDS = time.time() - _t0_main
print(f"本番の学習と評価 {MAIN_SECONDS / 60:.1f} 分(見積もり {PLAN_ESTIMATES[SELECTED_PLAN['plan']]['main'] / 60:.1f} 分)")


def bits_per_byte_of(condition: str, seeds: int) -> np.ndarray:
    return np.array([RUNS[(condition, s)]["bits_per_byte"] for s in range(seeds)])


def generalization_gap_line(condition: str) -> str:
    # 汎化の差(診断量): 評価集合の bits-per-byte - 学習用の部分の窓の bits-per-byte(シード平均と各シードの値)
    records = [RUNS[(condition, s)] for s in range(NUM_SEEDS[condition])]
    train = np.array([r["train_bits_per_byte"] for r in records])
    gap = np.array([r["bits_per_byte"] for r in records]) - train
    line = (
        f"{condition}(総パラメータ数 {TOTAL_PARAMETERS[condition]:,}、{len(records)} シード): 学習用の部分の窓 {train.mean():.4f}、"
        f"差(評価集合 - 学習用)の平均 {gap.mean():+.4f}、各シード {rounded(gap)}"
    )
    if condition != "D":  # 同じシードの D との差(共通するシードのみ)
        common = min(NUM_SEEDS[condition], NUM_SEEDS["D"])
        dense = np.array([RUNS[("D", s)]["bits_per_byte"] - RUNS[("D", s)]["train_bits_per_byte"] for s in range(common)])
        paired = gap[:common] - dense
        line += f"。同じシードの D との差(共通する {common} シード)の平均 {paired.mean():+.4f}、各シード {rounded(paired)}"
    return line


def bootstrap_bits_per_byte(keys: list[tuple[str, int]]) -> np.ndarray:
    # 記事をクラスタとして復元抽出する対応付きブートストラップ(全行に同じ再標本)。形状 (反復, 行)
    numerators = np.stack([RUNS[k]["nll"] for k in keys]) / math.log(2)
    return paired_cluster_bootstrap_ratio_of_sums(
        numerators, EVAL_WINDOW_BYTES, EVAL_WINDOW_ARTICLES, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
    )


_ARTICLE_LABELS = sorted(set(EVAL_WINDOW_ARTICLES))
_ARTICLE_OF_WINDOW = np.array([_ARTICLE_LABELS.index(a) for a in EVAL_WINDOW_ARTICLES])


def bootstrap_normalized_entropy(keys: list[tuple[str, int]]) -> np.ndarray:
    # 記事を復元抽出し、選ばれた記事の割り当ての個数を足し合わせて h(正規化エントロピーの層平均)を作り直す。形状 (反復, 行)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    num_articles = len(_ARTICLE_LABELS)
    weights = rng.multinomial(num_articles, np.full(num_articles, 1.0 / num_articles), size=BOOTSTRAP_RESAMPLES).astype(np.float64)
    columns = []
    for key in keys:
        counts = RUNS[key]["counts"].astype(np.float64)  # (窓, 層, E)
        per_article = np.zeros((num_articles, *counts.shape[1:]))
        np.add.at(per_article, _ARTICLE_OF_WINDOW, counts)
        resampled = np.einsum("ra,ale->rle", weights, per_article)
        fractions = resampled / resampled.sum(axis=-1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(fractions > 0, fractions * np.log(fractions), 0.0)
        columns.append((-terms.sum(axis=-1) / math.log(counts.shape[-1])).mean(axis=1))
    return np.stack(columns, axis=1)


# ブートストラップの関数の確認: 再標本をしない場合(全記事を 1 回ずつ)の値が、記録した値と一致する
_key = OBSERVATION_RUN
_counts = RUNS[_key]["counts"].sum(axis=0).astype(np.float64)
assert math.isclose(
    float(np.mean([compute_normalized_entropy(f) for f in _counts])), RUNS[_key]["routing"]["h"], rel_tol=1e-9
), "窓ごとの割り当ての個数から作った h が、層の積算から作った h と一致しない"
assert math.isclose(
    float(RUNS[_key]["nll"].sum() / math.log(2) / EVAL_WINDOW_BYTES.sum()), RUNS[_key]["bits_per_byte"], rel_tol=1e-12
)
```

    D シード 0: bits-per-byte 1.7721(学習用の部分の窓 1.4726)、最後の区間の訓練損失 3.953(ln V の 0.439 倍)、clipping の発動 0.00、更新のスキップ 0、115 秒
    D シード 1: bits-per-byte 1.7734(学習用の部分の窓 1.4788)、最後の区間の訓練損失 3.958(ln V の 0.439 倍)、clipping の発動 0.00、更新のスキップ 0、115 秒
    D シード 2: bits-per-byte 1.7793(学習用の部分の窓 1.4860)、最後の区間の訓練損失 4.007(ln V の 0.445 倍)、clipping の発動 0.00、更新のスキップ 0、115 秒
    D シード 3: bits-per-byte 1.7613(学習用の部分の窓 1.4664)、最後の区間の訓練損失 3.910(ln V の 0.434 倍)、clipping の発動 0.00、更新のスキップ 0、115 秒
    D シード 4: bits-per-byte 1.7766(学習用の部分の窓 1.4835)、最後の区間の訓練損失 3.988(ln V の 0.443 倍)、clipping の発動 0.00、更新のスキップ 0、115 秒
    M8 シード 0: bits-per-byte 1.7815(学習用の部分の窓 1.4725)、最後の区間の訓練損失 3.961(ln V の 0.440 倍)、clipping の発動 0.00、更新のスキップ 0、194 秒、h 0.9845、最大の負荷 0.205、破棄 0.1162、確率の最大値 0.311
    M8 シード 1: bits-per-byte 1.7763(学習用の部分の窓 1.4739)、最後の区間の訓練損失 3.949(ln V の 0.438 倍)、clipping の発動 0.00、更新のスキップ 0、188 秒、h 0.9871、最大の負荷 0.178、破棄 0.0403、確率の最大値 0.325
    M8 シード 2: bits-per-byte 1.7774(学習用の部分の窓 1.4800)、最後の区間の訓練損失 4.000(ln V の 0.444 倍)、clipping の発動 0.00、更新のスキップ 0、188 秒、h 0.9877、最大の負荷 0.199、破棄 0.0529、確率の最大値 0.321
    M8 シード 3: bits-per-byte 1.7770(学習用の部分の窓 1.4777)、最後の区間の訓練損失 3.948(ln V の 0.438 倍)、clipping の発動 0.00、更新のスキップ 0、188 秒、h 0.9931、最大の負荷 0.168、破棄 0.0291、確率の最大値 0.316
    M8 シード 4: bits-per-byte 1.7847(学習用の部分の窓 1.4824)、最後の区間の訓練損失 3.988(ln V の 0.443 倍)、clipping の発動 0.00、更新のスキップ 0、188 秒、h 0.9926、最大の負荷 0.179、破棄 0.0689、確率の最大値 0.321
    N8 シード 0: bits-per-byte 1.7965(学習用の部分の窓 1.4944)、最後の区間の訓練損失 4.053(ln V の 0.450 倍)、clipping の発動 0.00、更新のスキップ 0、188 秒、h 0.8835、最大の負荷 0.257、破棄 0.1120、確率の最大値 0.316
    N8 シード 1: bits-per-byte 1.8007(学習用の部分の窓 1.5025)、最後の区間の訓練損失 4.060(ln V の 0.451 倍)、clipping の発動 0.00、更新のスキップ 0、188 秒、h 0.8897、最大の負荷 0.294、破棄 0.1259、確率の最大値 0.340
    N8 シード 2: bits-per-byte 1.8222(学習用の部分の窓 1.5311)、最後の区間の訓練損失 4.171(ln V の 0.463 倍)、clipping の発動 0.00、更新のスキップ 0、187 秒、h 0.8901、最大の負荷 0.246、破棄 0.1295、確率の最大値 0.328
    N8 シード 3: bits-per-byte 1.7964(学習用の部分の窓 1.5017)、最後の区間の訓練損失 4.037(ln V の 0.448 倍)、clipping の発動 0.00、更新のスキップ 0、188 秒、h 0.9112、最大の負荷 0.254、破棄 0.1181、確率の最大値 0.322
    N8 シード 4: bits-per-byte 1.8062(学習用の部分の窓 1.5102)、最後の区間の訓練損失 4.092(ln V の 0.454 倍)、clipping の発動 0.00、更新のスキップ 0、188 秒、h 0.9182、最大の負荷 0.227、破棄 0.0674、確率の最大値 0.349
    M4 シード 0: bits-per-byte 1.7691(学習用の部分の窓 1.4583)、最後の区間の訓練損失 3.917(ln V の 0.435 倍)、clipping の発動 0.00、更新のスキップ 0、167 秒、h 0.9878、最大の負荷 0.345、破棄 0.0216、確率の最大値 0.501
    M4 シード 1: bits-per-byte 1.7727(学習用の部分の窓 1.4671)、最後の区間の訓練損失 3.926(ln V の 0.436 倍)、clipping の発動 0.00、更新のスキップ 0、167 秒、h 0.9960、最大の負荷 0.302、破棄 0.0030、確率の最大値 0.514
    M4 シード 2: bits-per-byte 1.7824(学習用の部分の窓 1.4845)、最後の区間の訓練損失 4.006(ln V の 0.445 倍)、clipping の発動 0.00、更新のスキップ 0、167 秒、h 0.9867、最大の負荷 0.364、破棄 0.0260、確率の最大値 0.508
    M4 シード 3: bits-per-byte 1.7636(学習用の部分の窓 1.4601)、最後の区間の訓練損失 3.895(ln V の 0.432 倍)、clipping の発動 0.00、更新のスキップ 0、167 秒、h 0.9951、最大の負荷 0.291、破棄 0.0163、確率の最大値 0.502
    M4 シード 4: bits-per-byte 1.7804(学習用の部分の窓 1.4845)、最後の区間の訓練損失 3.992(ln V の 0.443 倍)、clipping の発動 0.00、更新のスキップ 0、167 秒、h 0.9951、最大の負荷 0.325、破棄 0.0342、確率の最大値 0.516
    M16 シード 0: bits-per-byte 1.7969(学習用の部分の窓 1.4926)、最後の区間の訓練損失 4.019(ln V の 0.446 倍)、clipping の発動 0.00、更新のスキップ 0、233 秒、h 0.9876、最大の負荷 0.102、破棄 0.0509、確率の最大値 0.180
    M16 シード 1: bits-per-byte 1.8044(学習用の部分の窓 1.5086)、最後の区間の訓練損失 4.051(ln V の 0.450 倍)、clipping の発動 0.00、更新のスキップ 0、233 秒、h 0.9858、最大の負荷 0.118、破棄 0.0957、確率の最大値 0.181
    M16 シード 2: bits-per-byte 1.8075(学習用の部分の窓 1.5093)、最後の区間の訓練損失 4.090(ln V の 0.454 倍)、clipping の発動 0.00、更新のスキップ 0、233 秒、h 0.9828、最大の負荷 0.127、破棄 0.1047、確率の最大値 0.183
    M16 シード 3: bits-per-byte 1.7935(学習用の部分の窓 1.4904)、最後の区間の訓練損失 3.985(ln V の 0.442 倍)、clipping の発動 0.00、更新のスキップ 0、233 秒、h 0.9871、最大の負荷 0.135、破棄 0.0977、確率の最大値 0.178
    M16 シード 4: bits-per-byte 1.7831(学習用の部分の窓 1.4869)、最後の区間の訓練損失 4.006(ln V の 0.445 倍)、clipping の発動 0.00、更新のスキップ 0、233 秒、h 0.9855、最大の負荷 0.116、破棄 0.1104、確率の最大値 0.179
    M2 シード 0: bits-per-byte 1.7679(学習用の部分の窓 1.4599)、最後の区間の訓練損失 3.920(ln V の 0.435 倍)、clipping の発動 0.00、更新のスキップ 0、155 秒、h 0.9877、最大の負荷 0.587、破棄 0.0000、確率の最大値 0.733
    M2 シード 1: bits-per-byte 1.7681(学習用の部分の窓 1.4646)、最後の区間の訓練損失 3.919(ln V の 0.435 倍)、clipping の発動 0.00、更新のスキップ 0、155 秒、h 0.9977、最大の負荷 0.544、破棄 0.0000、確率の最大値 0.739
    M2 シード 2: bits-per-byte 1.7694(学習用の部分の窓 1.4769)、最後の区間の訓練損失 3.985(ln V の 0.442 倍)、clipping の発動 0.00、更新のスキップ 0、155 秒、h 0.9897、最大の負荷 0.618、破棄 0.0000、確率の最大値 0.731
    M2 シード 3: bits-per-byte 1.7560(学習用の部分の窓 1.4598)、最後の区間の訓練損失 3.891(ln V の 0.432 倍)、clipping の発動 0.00、更新のスキップ 0、156 秒、h 0.9847、最大の負荷 0.628、破棄 0.0000、確率の最大値 0.739
    M2 シード 4: bits-per-byte 1.7660(学習用の部分の窓 1.4703)、最後の区間の訓練損失 3.946(ln V の 0.438 倍)、clipping の発動 0.00、更新のスキップ 0、155 秒、h 0.9954、最大の負荷 0.574、破棄 0.0000、確率の最大値 0.742
    本番の学習と評価 87.2 分(見積もり 84.5 分)


### 6.7 実験 A: 計算量を揃えた密なモデルとの比較


```python
_seeds = list(range(SEEDS_A))
B_D, B_M8 = bits_per_byte_of("D", SEEDS_A), bits_per_byte_of("M8", SEEDS_A)
A_PER_SEED = B_D - B_M8
_boot = bootstrap_bits_per_byte([("D", s) for s in _seeds] + [("M8", s) for s in _seeds])
_boot_contrast = _boot[:, :SEEDS_A].mean(axis=1) - _boot[:, SEEDS_A:].mean(axis=1)
DELTA_A = float(A_PER_SEED.mean())
SIGMA_A = combined_sigma(A_PER_SEED, _boot_contrast)

_p1 = {(c, s): learning_precondition(RUNS[(c, s)]) for c in ("D", "M8") for s in _seeds}
_p2 = {s: load_precondition(RUNS[("M8", s)]) for s in _seeds}
precondition_status["P1(A)"] = all(_p1.values())
precondition_status["P2(A)"] = all(_p2.values())
A_PRECONDITIONS = ["P0(A)", "P1(A)", "P2(A)"]
A_COMPUTED = judge(DELTA_A, SIGMA_A["sigma"])
A_VERDICT = A_COMPUTED if all(precondition_status[k] for k in A_PRECONDITIONS) else "前提不成立"

print(f"{RUN_TAG}実験 A(シード {_seeds}、T = {NUM_STEPS})")
print(f"  bits-per-byte b_D: {rounded(B_D)}、b_M8: {rounded(B_M8)}")
print(f"  d_s = b_D - b_M8: {rounded(A_PER_SEED)}")
print(
    f"  Delta_A = {DELTA_A:+.4f}、sigma_A = {SIGMA_A['sigma']:.4f}(シード間 {SIGMA_A['seed_term']:.4f}・ブートストラップ "
    f"{SIGMA_A['bootstrap_term']:.4f}、反復 {BOOTSTRAP_RESAMPLES:,})、閾値 2 sigma_A = {SIGMA_MULTIPLIER * SIGMA_A['sigma']:.4f}"
)
print(
    f"  前提条件: P0 {precondition_status['P0(A)']}、P1 {precondition_status['P1(A)']}"
    f"(不成立の学習 {[k for k, v in _p1.items() if not v] or 'なし'}、閾値 {P1_LOSS_RATIO} x ln V = {P1_LOSS_RATIO * UNIFORM_LOSS:.3f})、"
    f"P2 {precondition_status['P2(A)']}(M8 の破棄の最大 {max(max(RUNS[('M8', s)]['routing']['dropped']) for s in _seeds):.4f} <= {P2_DROPPED_MAX}、"
    f"最大の負荷 {max(max(RUNS[('M8', s)]['routing']['max_load']) for s in _seeds):.3f} <= {P2_LOAD_FACTOR / 8:.3f})"
)
print(f"  判定関数の結果 {A_COMPUTED} -> {RUN_TAG}最終判定: {A_VERDICT}")

# 診断量(判定なし)
if NUM_SEEDS["W"] > 0:
    _b_w = bits_per_byte_of("W", NUM_SEEDS["W"])
    print(
        f"  診断量: 総パラメータ数を M8 に揃えた密なモデル W(計算量は揃わない)の bits-per-byte {rounded(_b_w)}、平均 {_b_w.mean():.4f}"
        f"(D の平均 {B_D.mean():.4f}、M8 の平均 {B_M8.mean():.4f})"
    )
else:
    print("  診断量: W(総パラメータ数を揃えた密なモデル)は、選ばれた段階では学習していない")
print(f"  診断量: 総パラメータ数 {dumps_compact_json({c: TOTAL_PARAMETERS[c] for c in ('D', 'M8', 'W')})}")
print("  診断量: 評価集合の bits-per-byte の推移(シード平均、評価のステップ " + f"{EVAL_STEPS})")
for _c in ("D", "M8", "W"):
    if NUM_SEEDS[_c] > 0:
        print(f"    {_c}: {rounded(np.mean([RUNS[(_c, s)]['eval_bits_per_byte'] for s in range(NUM_SEEDS[_c])], axis=0))}")
print("  診断量: 汎化の差(評価集合と学習用の部分の窓は別の記事なので、条件間での差の違いを読む)")
for _c in ("D", "M8", "W"):
    if NUM_SEEDS[_c] > 0:
        print(f"    {generalization_gap_line(_c)}")
print(
    f"  診断量: M8 の学習時の破棄の割合(評価と評価の間の学習ステップ、層とシードの最大): "
    f"{rounded(np.max([[max(d['train_dropped']) for d in RUNS[('M8', s)]['eval_diagnostics']] for s in _seeds], axis=0))}"
)

_fig, _axes = plt.subplots(1, 2, figsize=(11, 3.6))
for _c, _marker in (("D", "o"), ("M8", "s"), ("W", "^")):
    if NUM_SEEDS[_c] > 0:
        _axes[0].plot(range(NUM_SEEDS[_c]), bits_per_byte_of(_c, NUM_SEEDS[_c]), _marker, label=_c)
        _curves = np.array([RUNS[(_c, s)]["eval_bits_per_byte"] for s in range(NUM_SEEDS[_c])])
        _axes[1].plot(EVAL_STEPS, _curves.mean(axis=0), marker=_marker, label=_c)
_axes[0].set_xlabel("seed")
_axes[0].set_ylabel("bits-per-byte (evaluation set)")
_axes[0].set_title(f"{PLOT_TAG}Experiment A: bits-per-byte per seed")
_axes[0].legend()
_axes[1].set_xscale("log", base=2)
_axes[1].set_xlabel("step")
_axes[1].set_ylabel("bits-per-byte (seed mean)")
_axes[1].set_title(f"{PLOT_TAG}bits-per-byte during training")
_axes[1].legend()
plt.tight_layout()
plt.show()
```

    実験 A(シード [0, 1, 2, 3, 4]、T = 1090)
      bits-per-byte b_D: [1.7721, 1.7734, 1.7793, 1.7613, 1.7766]、b_M8: [1.7815, 1.7763, 1.7774, 1.777, 1.7847]
      d_s = b_D - b_M8: [-0.0095, -0.0029, 0.0018, -0.0157, -0.0082]
      Delta_A = -0.0069、sigma_A = 0.0038(シード間 0.0030・ブートストラップ 0.0024、反復 10,000)、閾値 2 sigma_A = 0.0076
      前提条件: P0 True、P1 True(不成立の学習 なし、閾値 0.65 x ln V = 5.857)、P2 True(M8 の破棄の最大 0.1162 <= 0.15、最大の負荷 0.205 <= 0.375)
      判定関数の結果 判定不能 -> 最終判定: 判定不能
      診断量: W(総パラメータ数を揃えた密なモデル)は、選ばれた段階では学習していない
      診断量: 総パラメータ数 {"D": 5246208, "M8": 19941632, "W": 19933440}
      診断量: 評価集合の bits-per-byte の推移(シード平均、評価のステップ (17, 34, 68, 136, 272, 545, 1090))
        D: [3.1961, 2.868, 2.82, 2.4954, 2.2044, 1.9251, 1.7725]
        M8: [3.211, 2.8726, 2.8362, 2.5058, 2.2242, 1.9508, 1.7794]
      診断量: 汎化の差(評価集合と学習用の部分の窓は別の記事なので、条件間での差の違いを読む)
        D(総パラメータ数 5,246,208、5 シード): 学習用の部分の窓 1.4775、差(評価集合 - 学習用)の平均 +0.2950、各シード [0.2995, 0.2946, 0.2932, 0.2949, 0.2931]
        M8(総パラメータ数 19,941,632、5 シード): 学習用の部分の窓 1.4773、差(評価集合 - 学習用)の平均 +0.3021、各シード [0.309, 0.3025, 0.2975, 0.2993, 0.3024]。同じシードの D との差(共通する 5 シード)の平均 +0.0071、各シード [0.0096, 0.0079, 0.0042, 0.0044, 0.0093]
      診断量: M8 の学習時の破棄の割合(評価と評価の間の学習ステップ、層とシードの最大): [0.2605, 0.492, 0.5462, 0.2417, 0.1134, 0.0557, 0.0189]



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/022_mixture_of_experts/output_36_1.png)
    


### 6.8 実験 B: 負荷分散損失と負荷の偏り


```python
_seeds = list(range(SEEDS_B))
H_M8 = np.array([RUNS[("M8", s)]["routing"]["h"] for s in _seeds])
H_N8 = np.array([RUNS[("N8", s)]["routing"]["h"] for s in _seeds])
B_PER_SEED = H_M8 - H_N8
_boot = bootstrap_normalized_entropy([("M8", s) for s in _seeds] + [("N8", s) for s in _seeds])
_boot_contrast = _boot[:, :SEEDS_B].mean(axis=1) - _boot[:, SEEDS_B:].mean(axis=1)
DELTA_B = float(B_PER_SEED.mean())
SIGMA_B = combined_sigma(B_PER_SEED, _boot_contrast)

_p1 = {(c, s): learning_precondition(RUNS[(c, s)]) for c in ("M8", "N8") for s in _seeds}
_max_probability = np.array([np.mean(RUNS[("M8", s)]["routing"]["max_probability"]) for s in _seeds])
precondition_status["P1(B)"] = all(_p1.values())
precondition_status["P3(B)"] = bool((_max_probability >= P3_MAX_PROBABILITY_FACTOR / 8).all())
B_PRECONDITIONS = ["P0(B)", "P1(B)", "P3(B)"]
B_COMPUTED = judge(DELTA_B, SIGMA_B["sigma"])
B_VERDICT = B_COMPUTED if all(precondition_status[k] for k in B_PRECONDITIONS) else "前提不成立"

print(f"{RUN_TAG}実験 B(シード {_seeds}、T = {NUM_STEPS})")
print(f"  h_M8(負荷分散損失あり): {rounded(H_M8)}、h_N8(なし): {rounded(H_N8)}")
print(f"  d_s = h_M8 - h_N8: {rounded(B_PER_SEED)}")
print(
    f"  Delta_B = {DELTA_B:+.4f}、sigma_B = {SIGMA_B['sigma']:.4f}(シード間 {SIGMA_B['seed_term']:.4f}・ブートストラップ "
    f"{SIGMA_B['bootstrap_term']:.4f}、反復 {BOOTSTRAP_RESAMPLES:,})、閾値 2 sigma_B = {SIGMA_MULTIPLIER * SIGMA_B['sigma']:.4f}"
)
print(
    f"  前提条件: P0 {precondition_status['P0(B)']}、P1 {precondition_status['P1(B)']}(不成立の学習 {[k for k, v in _p1.items() if not v] or 'なし'})、"
    f"P3 {precondition_status['P3(B)']}(M8 のルーターの確率の最大値の平均(層平均){rounded(_max_probability, 3)} >= {P3_MAX_PROBABILITY_FACTOR / 8:.3f})"
)
print(f"  判定関数の結果 {B_COMPUTED} -> {RUN_TAG}最終判定: {B_VERDICT}")

# 診断量(判定なし)
print(f"  診断量: h の推移(シード平均、評価のステップ {EVAL_STEPS})")
for _c in ("M8", "N8"):
    _recs = [RUNS[(_c, s)] for s in _seeds]
    print(f"    {_c}: {rounded(np.mean([[d['h'] for d in r['eval_diagnostics']] for r in _recs], axis=0))}")
for _c in ("M8", "N8"):
    _recs = [RUNS[(_c, s)] for s in _seeds]
    print(
        f"    {_c}(学習の終わり、層ごとのシード平均): 最大の負荷 {rounded(np.mean([r['routing']['max_load'] for r in _recs], axis=0), 3)}、"
        f"ほとんど選ばれないエキスパートの数 {rounded(np.mean([r['routing']['rare_experts'] for r in _recs], axis=0), 2)}、"
        f"評価集合で capacity factor 2.0 なら破棄されたはずの割合 {rounded(np.mean([r['routing']['dropped'] for r in _recs], axis=0))}、"
        f"学習時の破棄(最後の区間){rounded(np.mean([r['eval_diagnostics'][-1]['train_dropped'] for r in _recs], axis=0))}、"
        f"確率の最大値の平均 {rounded(np.mean([r['routing']['max_probability'] for r in _recs], axis=0), 3)}、"
        f"ロジットのノルム {rounded(np.mean([r['routing']['logit_norm'] for r in _recs], axis=0), 2)}"
    )
    print(
        f"    {_c}: 補助損失(最後の区間の平均、係数を掛ける前、4 層の合計。シード平均): "
        f"{json.dumps({k: round(float(np.mean([r['auxiliary'][k] for r in _recs])), 4) for k in _recs[0]['auxiliary']})}"
    )
_b_m8, _b_n8 = bits_per_byte_of("M8", SEEDS_B), bits_per_byte_of("N8", SEEDS_B)
print(f"  診断量: bits-per-byte b_M8 {rounded(_b_m8)}、b_N8 {rounded(_b_n8)}、差 b_N8 - b_M8 の平均 {float((_b_n8 - _b_m8).mean()):+.4f}")

_fig, _axes = plt.subplots(1, 3, figsize=(15, 3.6))
for _c, _marker in (("M8", "s"), ("N8", "x")):
    for _s in _seeds:
        _axes[0].plot(EVAL_STEPS, [d["h"] for d in RUNS[(_c, _s)]["eval_diagnostics"]], marker=_marker, color="C0" if _c == "M8" else "C3",
                      alpha=0.6, label=_c if _s == 0 else None)
_axes[0].set_xscale("log", base=2)
_axes[0].set_xlabel("step")
_axes[0].set_ylabel("normalized entropy of load (layer mean)")
_axes[0].set_title(f"{PLOT_TAG}Experiment B: load entropy during training")
_axes[0].legend()
for _axis, _c in zip(_axes[1:], ("M8", "N8"), strict=True):
    _fractions = RUNS[(_c, 0)]["counts"].sum(axis=0).astype(np.float64)
    _fractions = _fractions / _fractions.sum(axis=1, keepdims=True)
    _image = _axis.imshow(_fractions, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(0.5, float(_fractions.max())))
    _axis.set_xlabel("expert")
    _axis.set_ylabel("layer")
    _axis.set_title(f"{PLOT_TAG}{_c} seed 0: load fraction f_i")
    plt.colorbar(_image, ax=_axis)
plt.tight_layout()
plt.show()
```

    実験 B(シード [0, 1, 2, 3, 4]、T = 1090)
      h_M8(負荷分散損失あり): [0.9845, 0.9871, 0.9877, 0.9931, 0.9926]、h_N8(なし): [0.8835, 0.8897, 0.8901, 0.9112, 0.9182]
      d_s = h_M8 - h_N8: [0.101, 0.0974, 0.0976, 0.082, 0.0744]
      Delta_B = +0.0905、sigma_B = 0.0069(シード間 0.0052・ブートストラップ 0.0045、反復 10,000)、閾値 2 sigma_B = 0.0138
      前提条件: P0 True、P1 True(不成立の学習 なし)、P3 True(M8 のルーターの確率の最大値の平均(層平均)[0.311, 0.325, 0.321, 0.316, 0.321] >= 0.188)
      判定関数の結果 支持 -> 最終判定: 支持
      診断量: h の推移(シード平均、評価のステップ (17, 34, 68, 136, 272, 545, 1090))
        M8: [0.8025, 0.6525, 0.685, 0.8978, 0.9532, 0.9869, 0.989]
        N8: [0.4194, 0.2801, 0.3693, 0.7126, 0.7833, 0.8653, 0.8985]
        M8(学習の終わり、層ごとのシード平均): 最大の負荷 [0.164, 0.174, 0.162, 0.179]、ほとんど選ばれないエキスパートの数 [0.0, 0.0, 0.0, 0.0]、評価集合で capacity factor 2.0 なら破棄されたはずの割合 [0.0257, 0.0253, 0.0285, 0.04]、学習時の破棄(最後の区間)[0.0079, 0.008, 0.0074, 0.0095]、確率の最大値の平均 [0.227, 0.283, 0.36, 0.404]、ロジットのノルム [5.74, 6.34, 6.76, 6.74]
        M8: 補助損失(最後の区間の平均、係数を掛ける前、4 層の合計。シード平均): {"load_balancing": 4.0357, "router_z": 1.1946}
        N8(学習の終わり、層ごとのシード平均): 最大の負荷 [0.202, 0.231, 0.22, 0.233]、ほとんど選ばれないエキスパートの数 [0.4, 0.6, 1.2, 1.2]、評価集合で capacity factor 2.0 なら破棄されたはずの割合 [0.0383, 0.0546, 0.0679, 0.0797]、学習時の破棄(最後の区間)[0.0904, 0.134, 0.1244, 0.1517]、確率の最大値の平均 [0.235, 0.313, 0.378, 0.398]、ロジットのノルム [6.02, 6.62, 7.04, 6.96]
        N8: 補助損失(最後の区間の平均、係数を掛ける前、4 層の合計。シード平均): {"load_balancing": 4.4957, "router_z": 0.918}
      診断量: bits-per-byte b_M8 [1.7815, 1.7763, 1.7774, 1.777, 1.7847]、b_N8 [1.7965, 1.8007, 1.8222, 1.7964, 1.8062]、差 b_N8 - b_M8 の平均 +0.0250



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/022_mixture_of_experts/output_38_1.png)
    


### 6.9 実験 C: エキスパート数の等比スケーリング


```python
_seeds = list(range(SEEDS_C))
_levels = {"D": 1, "M2": 2, "M4": 4, "M8": 8, "M16": 16}
B1, B4, B16 = (bits_per_byte_of(c, SEEDS_C) for c in ("D", "M4", "M16"))
G_LOW, G_HIGH = (B1 - B4) / 2, (B4 - B16) / 2
C_PER_SEED = G_LOW - G_HIGH
_boot = bootstrap_bits_per_byte([(c, s) for c in ("D", "M4", "M16") for s in _seeds])
_b1, _b4, _b16 = (_boot[:, i * SEEDS_C : (i + 1) * SEEDS_C].mean(axis=1) for i in range(3))
_boot_contrast = (_b1 - 2 * _b4 + _b16) / 2
DELTA_C = float(C_PER_SEED.mean())
SIGMA_C = combined_sigma(C_PER_SEED, _boot_contrast)
assert math.isclose(DELTA_C, float((B1 - 2 * B4 + B16).mean() / 2), abs_tol=1e-12)

_c_conditions = [c for c in _levels if NUM_SEEDS[c] > 0]
_p1 = {(c, s): learning_precondition(RUNS[(c, s)]) for c in ("D", "M4", "M16") for s in _seeds}
_p2 = {(c, s): load_precondition(RUNS[(c, s)]) for c in ("M4", "M16") for s in _seeds}
precondition_status["P1(C)"] = all(_p1.values())
precondition_status["P2(C)"] = all(_p2.values())
C_PRECONDITIONS = ["P0(C)", "P1(C)", "P2(C)"]
C_COMPUTED = judge(DELTA_C, SIGMA_C["sigma"])
C_VERDICT = C_COMPUTED if all(precondition_status[k] for k in C_PRECONDITIONS) else "前提不成立"

print(f"{RUN_TAG}実験 C(シード {_seeds}、T = {NUM_STEPS})")
print(f"  bits-per-byte b_1(D): {rounded(B1)}、b_4(M4): {rounded(B4)}、b_16(M16): {rounded(B16)}")
print(f"  g_low = (b_1 - b_4) / 2: {rounded(G_LOW)}、g_high = (b_4 - b_16) / 2: {rounded(G_HIGH)}")
print(f"  d_s = g_low - g_high: {rounded(C_PER_SEED)}")
print(
    f"  Delta_C = {DELTA_C:+.4f}、sigma_C = {SIGMA_C['sigma']:.4f}(シード間 {SIGMA_C['seed_term']:.4f}・ブートストラップ "
    f"{SIGMA_C['bootstrap_term']:.4f}、反復 {BOOTSTRAP_RESAMPLES:,})、閾値 2 sigma_C = {SIGMA_MULTIPLIER * SIGMA_C['sigma']:.4f}"
)
print(
    f"  前提条件: P0 {precondition_status['P0(C)']}、P1 {precondition_status['P1(C)']}(不成立の学習 {[k for k, v in _p1.items() if not v] or 'なし'})、"
    f"P2 {precondition_status['P2(C)']}(不成立の学習 {[k for k, v in _p2.items() if not v] or 'なし'})"
)
print(f"  判定関数の結果 {C_COMPUTED} -> {RUN_TAG}最終判定: {C_VERDICT}")

# 診断量(判定なし)
_g_low, _g_high = combined_sigma(G_LOW, (_b1 - _b4) / 2), combined_sigma(G_HIGH, (_b4 - _b16) / 2)
print(
    f"  診断量: g_low の平均 {G_LOW.mean():+.4f}(標準偏差 {_g_low['sigma']:.4f})、g_high の平均 {G_HIGH.mean():+.4f}(標準偏差 {_g_high['sigma']:.4f})"
)
print(
    "  診断量: 全水準(E、シード数、bits-per-byte の平均とシード間の標準偏差、h・最大の負荷 x E・"
    "capacity factor 2.0 なら破棄されたはずの割合の層とシードの最大)"
)
_means, _stds = {}, {}
for _c in _c_conditions:
    _b = bits_per_byte_of(_c, NUM_SEEDS[_c])
    _means[_c], _stds[_c] = float(_b.mean()), float(_b.std(ddof=1))
    _line = f"    E = {_levels[_c]:>2}({_c}、{NUM_SEEDS[_c]} シード、学習率 {LEARNING_RATE[_c]:.3g}): {_means[_c]:.4f} ± {_stds[_c]:.4f}"
    if _c != "D":
        _recs = [RUNS[(_c, s)] for s in range(NUM_SEEDS[_c])]
        _line += (
            f"、h {np.mean([r['routing']['h'] for r in _recs]):.4f}、最大の負荷 x E {max(max(r['routing']['max_load']) for r in _recs) * _levels[_c]:.2f}、"
            f"破棄 {max(max(r['routing']['dropped']) for r in _recs):.4f}、"
            f"学習時の破棄(最後の区間){max(max(r['eval_diagnostics'][-1]['train_dropped']) for r in _recs):.4f}"
        )
    print(_line)
print("  診断量: 汎化の差(評価集合と学習用の部分の窓は別の記事なので、条件間での差の違いを読む)")
for _c in _c_conditions:
    print(f"    {generalization_gap_line(_c)}")

_fig, _axis = plt.subplots(figsize=(5.5, 3.8))
_x = [math.log2(_levels[c]) for c in _c_conditions]
_axis.errorbar(_x, [_means[c] for c in _c_conditions], yerr=[_stds[c] for c in _c_conditions], marker="o", capsize=3)
_axis.set_xticks(_x)
_axis.set_xticklabels([str(_levels[c]) for c in _c_conditions])
_axis.set_xlabel("number of experts E (log scale)")
_axis.set_ylabel("bits-per-byte (mean ± seed std)")
_axis.set_title(f"{PLOT_TAG}Experiment C: bits-per-byte vs number of experts")
plt.tight_layout()
plt.show()
```

    実験 C(シード [0, 1, 2, 3, 4]、T = 1090)
      bits-per-byte b_1(D): [1.7721, 1.7734, 1.7793, 1.7613, 1.7766]、b_4(M4): [1.7691, 1.7727, 1.7824, 1.7636, 1.7804]、b_16(M16): [1.7969, 1.8044, 1.8075, 1.7935, 1.7831]
      g_low = (b_1 - b_4) / 2: [0.0015, 0.0003, -0.0016, -0.0011, -0.0019]、g_high = (b_4 - b_16) / 2: [-0.0139, -0.0159, -0.0126, -0.015, -0.0013]
      d_s = g_low - g_high: [0.0154, 0.0162, 0.011, 0.0138, -0.0006]
      Delta_C = +0.0112、sigma_C = 0.0035(シード間 0.0031・ブートストラップ 0.0016、反復 10,000)、閾値 2 sigma_C = 0.0069
      前提条件: P0 True、P1 True(不成立の学習 なし)、P2 True(不成立の学習 なし)
      判定関数の結果 支持 -> 最終判定: 支持
      診断量: g_low の平均 -0.0006(標準偏差 0.0013)、g_high の平均 -0.0117(標準偏差 0.0027)
      診断量: 全水準(E、シード数、bits-per-byte の平均とシード間の標準偏差、h・最大の負荷 x E・capacity factor 2.0 なら破棄されたはずの割合の層とシードの最大)
        E =  1(D、5 シード、学習率 0.0024): 1.7725 ± 0.0069
        E =  2(M2、5 シード、学習率 0.0024): 1.7655 ± 0.0055、h 0.9910、最大の負荷 x E 1.26、破棄 0.0000、学習時の破棄(最後の区間)0.0001
        E =  4(M4、5 シード、学習率 0.0024): 1.7736 ± 0.0078、h 0.9921、最大の負荷 x E 1.46、破棄 0.0342、学習時の破棄(最後の区間)0.0039
        E =  8(M8、5 シード、学習率 0.0024): 1.7794 ± 0.0036、h 0.9890、最大の負荷 x E 1.64、破棄 0.1162、学習時の破棄(最後の区間)0.0189
        E = 16(M16、5 シード、学習率 0.0024): 1.7971 ± 0.0097、h 0.9858、最大の負荷 x E 2.16、破棄 0.1104、学習時の破棄(最後の区間)0.0384
      診断量: 汎化の差(評価集合と学習用の部分の窓は別の記事なので、条件間での差の違いを読む)
        D(総パラメータ数 5,246,208、5 シード): 学習用の部分の窓 1.4775、差(評価集合 - 学習用)の平均 +0.2950、各シード [0.2995, 0.2946, 0.2932, 0.2949, 0.2931]
        M2(総パラメータ数 7,346,432、5 シード): 学習用の部分の窓 1.4663、差(評価集合 - 学習用)の平均 +0.2992、各シード [0.308, 0.3035, 0.2925, 0.2962, 0.2957]。同じシードの D との差(共通する 5 シード)の平均 +0.0041、各シード [0.0085, 0.009, -0.0007, 0.0013, 0.0027]
        M4(総パラメータ数 11,544,832、5 シード): 学習用の部分の窓 1.4709、差(評価集合 - 学習用)の平均 +0.3027、各シード [0.3108, 0.3056, 0.2979, 0.3034, 0.2959]。同じシードの D との差(共通する 5 シード)の平均 +0.0077、各シード [0.0113, 0.011, 0.0047, 0.0086, 0.0028]
        M8(総パラメータ数 19,941,632、5 シード): 学習用の部分の窓 1.4773、差(評価集合 - 学習用)の平均 +0.3021、各シード [0.309, 0.3025, 0.2975, 0.2993, 0.3024]。同じシードの D との差(共通する 5 シード)の平均 +0.0071、各シード [0.0096, 0.0079, 0.0042, 0.0044, 0.0093]
        M16(総パラメータ数 36,735,232、5 シード): 学習用の部分の窓 1.4976、差(評価集合 - 学習用)の平均 +0.2995、各シード [0.3043, 0.2959, 0.2982, 0.3031, 0.2962]。同じシードの D との差(共通する 5 シード)の平均 +0.0045、各シード [0.0048, 0.0013, 0.005, 0.0082, 0.0031]



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/022_mixture_of_experts/output_40_1.png)
    


### 6.10 観察 D: エキスパートの専門化(判定基準を設けない)

実験 A の MoE(M8、シード 0)の学習の最終ステップの重みで、評価集合の各位置の **入力トークンの種類**(5.3 節の 6 種類)と、その位置の
表現が送られたエキスパート(破棄の前の割り当て)の対応を、層ごとに数える。層ごとに、種類を行・エキスパートを列とする分割表を、
行ごとに割合にして印字する(行の和が 1)。あわせて、種類とエキスパートの正規化した相互情報量
$I(X; Y) / \min(H(X), H(Y))$ を層ごとに示す($X$ はトークンの種類、$Y$ は割り当て先のエキスパート。独立なら 0、一方が他方の
関数なら 1)。

**読み方の注意**: これは 1 つの学習(1 シード)の観察であり、判定基準を設けない。層 1 以降の入力は、注意機構によって文脈が混ざった
表現であり、「その位置のトークンの種類」は割り当てを説明しうる要因の 1 つにすぎない。種類の分類は語彙の文字列だけから決めた
粗い規則である。


```python
_record = RUNS[OBSERVATION_RUN]
_token_category = TOKEN_CATEGORY[EVAL_WINDOWS.numpy()]  # (窓, 位置)
_num_experts = CONDITIONS[OBSERVATION_RUN[0]]["experts"]
SPECIALIZATION_TABLES, SPECIALIZATION_NMI = [], []
print(f"{RUN_TAG}観察 D({OBSERVATION_RUN[0]}・シード {OBSERVATION_RUN[1]}、評価集合の {_token_category.size:,} トークン)")
for _layer in range(NUM_LAYERS):
    _table = np.zeros((len(TOKEN_CATEGORIES), _num_experts))
    np.add.at(_table, (_token_category.reshape(-1), _record["token_routing"][:, _layer].reshape(-1)), 1)
    assert _table.sum() == _token_category.size
    assert np.array_equal(_table.sum(axis=0), _record["counts"][:, _layer].sum(axis=0))  # 評価の割り当ての個数と一致
    SPECIALIZATION_TABLES.append(_table)
    SPECIALIZATION_NMI.append(compute_normalized_mutual_information(_table))
    print(f"  層 {_layer}: 正規化した相互情報量 {SPECIALIZATION_NMI[-1]:.4f}。種類ごとの割り当ての割合(列はエキスパート 0〜{_num_experts - 1})")
    for _i, _label in enumerate(TOKEN_CATEGORY_LABELS):
        _row = _table[_i] / max(_table[_i].sum(), 1.0)
        print(f"    {_label}(n = {int(_table[_i].sum()):,}): {rounded(_row, 2)}")

_fig, _axes = plt.subplots(1, NUM_LAYERS, figsize=(4.2 * NUM_LAYERS, 3.4), sharey=True)
for _layer, _axis in enumerate(_axes):
    _rows = SPECIALIZATION_TABLES[_layer] / np.maximum(SPECIALIZATION_TABLES[_layer].sum(axis=1, keepdims=True), 1.0)
    _image = _axis.imshow(_rows, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    _axis.set_xlabel("expert")
    _axis.set_title(f"{PLOT_TAG}layer {_layer} (NMI {SPECIALIZATION_NMI[_layer]:.3f})")
    _axis.set_yticks(range(len(TOKEN_CATEGORIES)))
    _axis.set_yticklabels(TOKEN_CATEGORIES)
plt.colorbar(_image, ax=_axes, label="share of the token category routed to the expert")
plt.show()
```

    観察 D(M8・シード 0、評価集合の 314,368 トークン)
      層 0: 正規化した相互情報量 0.1235。種類ごとの割り当ての割合(列はエキスパート 0〜7)
        数字(n = 13,160): [0.31, 0.01, 0.02, 0.27, 0.06, 0.03, 0.16, 0.15]
        句読点・記号・空白(n = 13,627): [0.17, 0.03, 0.06, 0.15, 0.13, 0.06, 0.02, 0.38]
        機能語(n = 67,721): [0.11, 0.09, 0.09, 0.29, 0.08, 0.17, 0.02, 0.16]
        内容語の先頭(n = 105,316): [0.07, 0.16, 0.2, 0.12, 0.16, 0.12, 0.08, 0.1]
        単語の途中の部分語(n = 97,841): [0.01, 0.07, 0.17, 0.04, 0.11, 0.15, 0.4, 0.06]
        その他(n = 16,703): [0.15, 0.11, 0.04, 0.14, 0.09, 0.12, 0.1, 0.25]
      層 1: 正規化した相互情報量 0.1490。種類ごとの割り当ての割合(列はエキスパート 0〜7)
        数字(n = 13,160): [0.46, 0.0, 0.03, 0.06, 0.0, 0.06, 0.2, 0.18]
        句読点・記号・空白(n = 13,627): [0.04, 0.14, 0.08, 0.03, 0.0, 0.13, 0.42, 0.15]
        機能語(n = 67,721): [0.13, 0.41, 0.1, 0.06, 0.03, 0.12, 0.09, 0.06]
        内容語の先頭(n = 105,316): [0.15, 0.12, 0.04, 0.19, 0.15, 0.18, 0.07, 0.1]
        単語の途中の部分語(n = 97,841): [0.02, 0.15, 0.25, 0.1, 0.24, 0.07, 0.01, 0.15]
        その他(n = 16,703): [0.05, 0.07, 0.1, 0.13, 0.02, 0.05, 0.41, 0.17]
      層 2: 正規化した相互情報量 0.1828。種類ごとの割り当ての割合(列はエキスパート 0〜7)
        数字(n = 13,160): [0.04, 0.08, 0.22, 0.04, 0.23, 0.01, 0.18, 0.21]
        句読点・記号・空白(n = 13,627): [0.03, 0.02, 0.04, 0.01, 0.24, 0.01, 0.61, 0.05]
        機能語(n = 67,721): [0.49, 0.13, 0.04, 0.03, 0.14, 0.0, 0.11, 0.05]
        内容語の先頭(n = 105,316): [0.09, 0.11, 0.11, 0.12, 0.08, 0.24, 0.06, 0.19]
        単語の途中の部分語(n = 97,841): [0.02, 0.09, 0.17, 0.18, 0.16, 0.21, 0.02, 0.16]
        その他(n = 16,703): [0.08, 0.02, 0.1, 0.07, 0.32, 0.04, 0.33, 0.05]
      層 3: 正規化した相互情報量 0.0782。種類ごとの割り当ての割合(列はエキスパート 0〜7)
        数字(n = 13,160): [0.6, 0.01, 0.04, 0.01, 0.0, 0.24, 0.05, 0.04]
        句読点・記号・空白(n = 13,627): [0.2, 0.04, 0.11, 0.06, 0.01, 0.26, 0.04, 0.29]
        機能語(n = 67,721): [0.14, 0.1, 0.16, 0.06, 0.1, 0.25, 0.03, 0.17]
        内容語の先頭(n = 105,316): [0.13, 0.19, 0.08, 0.2, 0.15, 0.14, 0.03, 0.08]
        単語の途中の部分語(n = 97,841): [0.02, 0.08, 0.08, 0.15, 0.22, 0.23, 0.04, 0.18]
        その他(n = 16,703): [0.1, 0.1, 0.13, 0.06, 0.05, 0.22, 0.04, 0.3]



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/022_mixture_of_experts/output_42_1.png)
    


### 6.11 不変条件のアサーションと`SMOKE_TEST`の配線

- 実効水準の照合: 実際に学習した条件・シード・ステップ数・評価のステップが、5.2 節で印字した水準と 6.4 節で選ばれた計画に一致する。
- 対応のある比較: 同じシードの条件どうしで、順伝播ネットワーク以外の初期値が一致する(異なるシードでは異なる)。
- 全条件で、学習ステップ数・バッチサイズ・系列長(したがって見たトークン数)・評価窓・bits-per-byte の分母が同じである。
- D と M8 の学習は 1 回だけで、実験 A・B・C が同じ記録を共有している。
- 学習率が、較正した値または規則で決めた値に一致する。
- 各対比量と診断量が何個のシードで計算されたかを印字し、選ばれた段階の条件ごとのシード数(6.1 節の削る段階の表)と一致する
  ことを確かめる。
- 評価は破棄なしで行われ(`evaluate_model()`の中で、層が数えた破棄が 0 であることを毎回確かめている)、P2 と診断量の
  「破棄されたはずの割合」が、記録した割り当ての個数から作り直した値と一致する。


```python
# --- 実効水準の照合(SMOKE_TEST の配線) ---
assert SELECTED_PLAN in PLANS and NUM_STEPS in STEP_CANDIDATES and NUM_SEEDS == STAGES[CURRENT_LEVEL_NAME][STAGE]
assert set(RUNS) == {(c, s) for c in CONDITIONS for s in range(NUM_SEEDS[c])}
for (_c, _s), _r in RUNS.items():
    assert _r["num_steps"] == NUM_STEPS == len(_r["train_loss"]), (_c, _s)
    assert tuple(_r["eval_step"]) == EVAL_STEPS and len(_r["eval_bits_per_byte"]) == len(EVAL_STEPS)
    assert _r["nll"].shape == (len(EVAL_WINDOWS),) and math.isfinite(_r["train_bits_per_byte"])
    assert _r["learning_rate"] == LEARNING_RATE[_c] == CALIBRATION[LEARNING_RATE_SOURCE[_c]]["chosen"]
    if CONDITIONS[_c]["kind"] == "moe":
        assert _r["counts"].shape == (len(EVAL_WINDOWS), NUM_LAYERS, CONDITIONS[_c]["experts"])
        assert (_r["counts"].sum(axis=2) == SEQUENCE_LENGTH).all()  # どの窓・層でも、全位置が 1 個ずつ割り当てられている
        # 「破棄されたはずの割合」は、記録した割り当ての個数から作り直した値と一致する
        assert _r["routing"]["dropped"] == reference_dropped_fraction(_r["counts"], SEQUENCE_LENGTH)
assert SMOKE_TEST or len(EVAL_WINDOWS) == NUM_EVAL_WINDOWS_AVAILABLE  # 本番は評価集合の全部の窓
assert BOOTSTRAP_RESAMPLES == LEVELS[CURRENT_LEVEL_NAME]["BOOTSTRAP_RESAMPLES"]
assert set(CALIBRATION) == set(calibration_targets(CALIBRATION_MODE, NUM_SEEDS))
for _c, _r in CALIBRATION.items():
    assert len(_r["grid"]) in (3, 4) and (len(_r["grid"]) == 4) == (_r["extended"] is not None)

# --- 対応のある比較: 同じシードの条件どうしで順伝播ネットワーク以外の初期値が同じ、異なるシードでは異なる ---
for _s in range(max(NUM_SEEDS.values())):
    _hashes = {RUNS[(c, _s)]["trunk_hash"] for c in CONDITIONS if NUM_SEEDS[c] > _s}
    assert len(_hashes) == 1, _s
assert len({RUNS[("D", s)]["trunk_hash"] for s in range(NUM_SEEDS["D"])}) == NUM_SEEDS["D"]

# --- 実験 A・B・C が同じ D・M8 の記録を使っている(学習を重複させていない) ---
assert np.array_equal(B_D[:SEEDS_C], B1) and np.array_equal(B_M8[:SEEDS_B], bits_per_byte_of("M8", SEEDS_B))
# --- 量ごとのシード数(削る段階でシード数を減らした条件が入る量は、共通するシードだけで計算する) ---
SEED_COUNTS = {
    "Delta_A(D・M8)": len(A_PER_SEED),
    "Delta_B(M8・N8)": len(B_PER_SEED),
    "Delta_C(D・M4・M16)": len(C_PER_SEED),
    "汎化の差の D との対応(条件: 共通するシード数)": {c: min(NUM_SEEDS[c], NUM_SEEDS["D"]) for c in CONDITIONS if c != "D" and NUM_SEEDS[c] > 0},
    "実験 C の図の水準(条件: シード数)": {c: NUM_SEEDS[c] for c in ("D", "M2", "M4", "M8", "M16")},
    "W(実験 A の診断量)": NUM_SEEDS["W"],
}
assert SEED_COUNTS["Delta_A(D・M8)"] == SEEDS_A == min(NUM_SEEDS["D"], NUM_SEEDS["M8"])
assert SEED_COUNTS["Delta_B(M8・N8)"] == SEEDS_B == NUM_SEEDS["N8"] <= NUM_SEEDS["M8"]
assert SEED_COUNTS["Delta_C(D・M4・M16)"] == SEEDS_C == NUM_SEEDS["M16"] <= NUM_SEEDS["M4"] == NUM_SEEDS["D"]
assert len(H_M8) == len(H_N8) == SEEDS_B and len(B1) == len(B4) == len(B16) == SEEDS_C
print(f"量ごとのシード数(段階 {STAGE}、条件ごとのシード数 {json.dumps(NUM_SEEDS)}): {json.dumps(SEED_COUNTS, ensure_ascii=False)}")
_trained = sum(NUM_SEEDS.values())
_calibrated = sum(len(r["grid"]) for r in CALIBRATION.values())
print(
    f"実効水準: 水準 {CURRENT_LEVEL_NAME!r}、計画 {SELECTED_PLAN['plan']}(T = {NUM_STEPS}、較正の方式 {CALIBRATION_MODE!r}、段階 {STAGE})、"
    f"本番の学習 {_trained} 回・較正の学習 {_calibrated} 回、評価窓 {len(EVAL_WINDOWS)} 個・記事 {NUM_EVAL_ARTICLES} 本、"
    f"ブートストラップ {BOOTSTRAP_RESAMPLES:,} 回"
)
print("不変条件(実効水準、対応のある初期値、ステップ数・評価窓・分母の一致、記録の共有、学習率の出どころ): OK")
```

    量ごとのシード数(段階 1、条件ごとのシード数 {"D": 5, "M8": 5, "N8": 5, "M4": 5, "M16": 5, "M2": 5, "W": 0}): {"Delta_A(D・M8)": 5, "Delta_B(M8・N8)": 5, "Delta_C(D・M4・M16)": 5, "汎化の差の D との対応(条件: 共通するシード数)": {"M2": 5, "M4": 5, "M8": 5, "M16": 5, "N8": 5}, "実験 C の図の水準(条件: シード数)": {"D": 5, "M2": 5, "M4": 5, "M8": 5, "M16": 5}, "W(実験 A の診断量)": 0}
    実効水準: 水準 'prod'、計画 16(T = 1090、較正の方式 'representative'、段階 1)、本番の学習 30 回・較正の学習 6 回、評価窓 1228 個・記事 17 本、ブートストラップ 10,000 回
    不変条件(実効水準、対応のある初期値、ステップ数・評価窓・分母の一致、記録の共有、学習率の出どころ): OK


### 6.12 判定結果の一覧


```python
print(f"{RUN_TAG}判定結果(計画 {SELECTED_PLAN['plan']}、T = {NUM_STEPS}、較正の方式 {CALIBRATION_MODE!r}、段階 {STAGE})")
for _name, _delta, _sigma, _computed, _verdict, _preconditions in (
    ("実験 A(計算量を揃えた密なモデルとの比較)", DELTA_A, SIGMA_A["sigma"], A_COMPUTED, A_VERDICT, A_PRECONDITIONS),
    ("実験 B(負荷分散損失と負荷の偏り)", DELTA_B, SIGMA_B["sigma"], B_COMPUTED, B_VERDICT, B_PRECONDITIONS),
    ("実験 C(エキスパート数の等比スケーリング)", DELTA_C, SIGMA_C["sigma"], C_COMPUTED, C_VERDICT, C_PRECONDITIONS),
):
    print(
        f"  {_name}: 対比量 {_delta:+.4f}、標準偏差 {_sigma:.4f}、閾値 {SIGMA_MULTIPLIER * _sigma:.4f}、"
        f"前提条件 {json.dumps({k: precondition_status[k] for k in _preconditions}, ensure_ascii=False)}、"
        f"判定関数の結果 {_computed} -> 最終判定 {_verdict}"
    )
print(f"\nノートブック全体の実行時間: {(time.time() - NOTEBOOK_START_TIME) / 60:.1f} 分")
```

    判定結果(計画 16、T = 1090、較正の方式 'representative'、段階 1)
      実験 A(計算量を揃えた密なモデルとの比較): 対比量 -0.0069、標準偏差 0.0038、閾値 0.0076、前提条件 {"P0(A)": true, "P1(A)": true, "P2(A)": true}、判定関数の結果 判定不能 -> 最終判定 判定不能
      実験 B(負荷分散損失と負荷の偏り): 対比量 +0.0905、標準偏差 0.0069、閾値 0.0138、前提条件 {"P0(B)": true, "P1(B)": true, "P3(B)": true}、判定関数の結果 支持 -> 最終判定 支持
      実験 C(エキスパート数の等比スケーリング): 対比量 +0.0112、標準偏差 0.0035、閾値 0.0069、前提条件 {"P0(C)": true, "P1(C)": true, "P2(C)": true}、判定関数の結果 支持 -> 最終判定 支持
    
    ノートブック全体の実行時間: 103.1 分




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/06_architectures/022_mixture_of_experts.ipynb
