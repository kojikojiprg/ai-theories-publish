---
title: "SFT(指示チューニング) / Supervised Fine-Tuning (Instruction Tuning)(実装・実験編 3/4)"
---

この記事は後編(実装・実験編 3/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/016_supervised_fine_tuning-practice-2)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/016_supervised_fine_tuning-practice-4)。

### 6.5 本番の学習: 条件 1〜4 × シード

各条件を $N_{\max}$ 事例・$T$ ステップで学習し(gradient clipping なし)、評価集合で評価する。シードは選ばれた段階の
実験 A・B のシード(6.4 節)。条件 1 の学習の記録は、実験 C の $N_{\max}$ の水準(C-$N_{\max}$)としても使う
(6.9 節)。コピー能力の診断(6.8 節)のため、条件 1 のシード 0 のモデルのみメモリ上に残す(保存はしない)。
前提条件 P-L(実験 A・B)をここで記録する。


```python
MAIN_RECORDS = {c: [] for c in CONDITIONS}
_t0_main = time.time()
for _condition, _spec in CONDITIONS.items():
    for _seed in SEEDS_AB:
        _record = run_condition(
            _condition, _seed, N_MAX, keep_model=(_condition == 1 and _seed == 0)
        )
        if "_model" in _record:
            KEPT_CONDITION_1_MODEL = _record.pop("_model")
        MAIN_RECORDS[_condition].append(_record)
        print(
            f"条件 {_condition}(マスク{'あり' if _spec['mask'] else 'なし'}・"
            f"{'長い' if _spec['long'] else '短い'}水準)seed={_seed}: 学習 {_record['train_seconds']:.1f}s・"
            f"評価 {_record['eval_seconds']:.1f}s"
        )
print(f"本番の学習と評価の合計: {time.time() - _t0_main:.1f}s")

# --- 前提条件 P-L(条件 1〜4 の全シード) ---
_pl_rows = []
for _condition, _spec in CONDITIONS.items():
    for _record in MAIN_RECORDS[_condition]:
        _limit = PRECONDITION_L_RATIO * BASE_RESPONSE_LOSS[_spec["long"]]
        _pl_rows.append(_record["evaluation"]["response_loss"] <= _limit)
precondition_status["P-L(A・B)"] = bool(all(_pl_rows))
print(
    f"P-L(条件 1〜4 の全 {len(_pl_rows)} 実行で、応答部分の負の対数尤度 <= {PRECONDITION_L_RATIO} x 微調整前): "
    f"{'成立' if precondition_status['P-L(A・B)'] else '不成立'}(満たした実行 {sum(_pl_rows)} / {len(_pl_rows)})"
)
```

    条件 1(マスクあり・短い水準)seed=0: 学習 61.3s・評価 7.0s
    条件 1(マスクあり・短い水準)seed=1: 学習 61.0s・評価 5.9s
    条件 1(マスクあり・短い水準)seed=2: 学習 60.8s・評価 6.5s
    条件 1(マスクあり・短い水準)seed=3: 学習 61.4s・評価 6.1s
    条件 1(マスクあり・短い水準)seed=4: 学習 61.5s・評価 6.1s
    条件 2(マスクなし・短い水準)seed=0: 学習 61.6s・評価 3.9s
    条件 2(マスクなし・短い水準)seed=1: 学習 61.4s・評価 3.6s
    条件 2(マスクなし・短い水準)seed=2: 学習 61.4s・評価 3.6s
    条件 2(マスクなし・短い水準)seed=3: 学習 61.6s・評価 4.2s
    条件 2(マスクなし・短い水準)seed=4: 学習 61.6s・評価 3.6s
    条件 3(マスクあり・長い水準)seed=0: 学習 170.7s・評価 5.6s
    条件 3(マスクあり・長い水準)seed=1: 学習 170.7s・評価 5.5s
    条件 3(マスクあり・長い水準)seed=2: 学習 170.9s・評価 5.8s
    条件 3(マスクあり・長い水準)seed=3: 学習 171.0s・評価 6.1s
    条件 3(マスクあり・長い水準)seed=4: 学習 171.0s・評価 5.8s
    条件 4(マスクなし・長い水準)seed=0: 学習 177.6s・評価 6.2s
    条件 4(マスクなし・長い水準)seed=1: 学習 177.2s・評価 5.5s
    条件 4(マスクなし・長い水準)seed=2: 学習 177.5s・評価 5.5s
    条件 4(マスクなし・長い水準)seed=3: 学習 177.6s・評価 6.0s
    条件 4(マスクなし・長い水準)seed=4: 学習 177.4s・評価 6.1s
    本番の学習と評価の合計: 2463.6s
    P-L(条件 1〜4 の全 20 実行で、応答部分の負の対数尤度 <= 0.5 x 微調整前): 成立(満たした実行 20 / 20)


### 6.6 実験 A: 指示への依存


```python
def experiment_a_statistics(counts_by_seed: list, counts_by_input: list) -> dict:
    # counts_by_seed: シードごとの正解数(M 事例のうち)、counts_by_input: 入力ごとの正解数(S シード x K 課題のうち)
    p = np.array(counts_by_seed, dtype=np.float64) / EVAL_SIZE
    a = np.array(counts_by_input, dtype=np.float64) / (NUM_SEEDS_AB * TASK_COUNT)
    p_bar = float(p.mean())
    s_p = float(p.std(ddof=1))
    variance_by_input = float(a.var(ddof=1))
    binomial = p_bar * (1 - p_bar) / EVAL_SIZE  # 旧基準の第 2 項(診断量)
    return {
        "p_bar": p_bar,
        "s_p": s_p,
        "variance_by_input": variance_by_input,
        "sigma": math.sqrt(s_p**2 / NUM_SEEDS_AB + variance_by_input / NUM_EVAL_INPUTS),
        "sigma_old": math.sqrt(s_p**2 / NUM_SEEDS_AB + binomial),
        "binomial": binomial,
        "mean_of_inputs": float(a.mean()),
    }


_smoke_tag = "[動作確認のみ、結論ではない] " if SMOKE_TEST else ""
_condition_1 = MAIN_RECORDS[1]
_exact = np.stack([r["evaluation"]["exact"] for r in _condition_1]).astype(np.int64)  # (S, M)
EXACT_COUNTS_A = [int(v) for v in _exact.sum(axis=1)]
EXACT_COUNTS_BY_INPUT_A = [int(v) for v in by_input(_exact).sum(axis=0)]  # sum_s sum_k c_{x,k,s}
STATISTICS_A = experiment_a_statistics(EXACT_COUNTS_A, EXACT_COUNTS_BY_INPUT_A)
P_BAR, SIGMA_A = STATISTICS_A["p_bar"], STATISTICS_A["sigma"]
assert math.isclose(STATISTICS_A["mean_of_inputs"], P_BAR, rel_tol=1e-12, abs_tol=1e-15)
CONTRAST_A = P_BAR - C_STAR
verdict_A = three_way_verdict(CONTRAST_A, SIGMA_A)

print(
    f"{_smoke_tag}c* = {C_STAR:.4f}、M = {EVAL_SIZE}、|X| = {NUM_EVAL_INPUTS}、S = {NUM_SEEDS_AB}"
)
print(
    f"{_smoke_tag}条件 1 の完全一致率(シード順): "
    + ", ".join(f"{c / EVAL_SIZE:.4f}" for c in EXACT_COUNTS_A)
)
print(
    f"{_smoke_tag}p_bar = {P_BAR:.4f}、s_p = {STATISTICS_A['s_p']:.4f}、Var_x(a_x) = "
    f"{STATISTICS_A['variance_by_input']:.4e}、sigma_A = {SIGMA_A:.4f}(第 1 項 "
    f"{STATISTICS_A['s_p'] ** 2 / NUM_SEEDS_AB:.2e}、第 2 項 {STATISTICS_A['variance_by_input'] / NUM_EVAL_INPUTS:.2e})"
)
print(f"{_smoke_tag}対比量 p_bar - c* = {CONTRAST_A:+.4f}、2 sigma_A = {2 * SIGMA_A:.4f}")
print(
    f"{_smoke_tag}前提条件: P0={precondition_status['P0']}、P-A={precondition_status['P-A']}、"
    f"P-L={precondition_status['P-L(A・B)']}"
)
print(f"{_smoke_tag}判定関数の結果: {verdict_A}")
print(
    f"{_smoke_tag}診断量: 旧基準の第 2 項 p_bar (1 - p_bar) / M = {STATISTICS_A['binomial']:.2e}"
    f"(旧基準の sigma_A = {STATISTICS_A['sigma_old']:.4f}、旧基準の判定関数の結果: "
    f"{three_way_verdict(CONTRAST_A, STATISTICS_A['sigma_old'])})"
)
print(
    f"{_smoke_tag}診断量: 条件 1 の応答部分の負の対数尤度(シード順): "
    + ", ".join(f"{v:.4f}" for v in response_loss_by_seed(_condition_1))
    + f"(微調整前 {BASE_RESPONSE_LOSS[False]:.4f})"
)
_unseen = [float(r["evaluation"]["unseen_exact"].mean()) for r in _condition_1]
print(
    f"{_smoke_tag}診断量: 未見テンプレートでの完全一致率(シード順): "
    + ", ".join(f"{v:.4f}" for v in _unseen)
)
_task_index = np.array([TASK_NAMES.index(e.task) for e in EVAL_EXAMPLES])
_per_task = {
    t: [float(r["evaluation"]["exact"][_task_index == k].mean()) for r in _condition_1]
    for k, t in enumerate(TASK_NAMES)
}
for _task, _values in _per_task.items():
    print(
        f"{_smoke_tag}診断量: 課題別の完全一致率 [{_task}](シード順): "
        + ", ".join(f"{v:.4f}" for v in _values)
    )
print(
    f"{_smoke_tag}診断量: 条件 1 の形式の遵守率(シード順): "
    + ", ".join(f"{r['evaluation']['format_ok'].mean():.4f}" for r in _condition_1)
)
print("\n--- 条件 1(シード 0)の生成例 ---")
for _e, _text in list(zip(EVAL_EXAMPLES, _condition_1[0]["evaluation"]["texts"], strict=True))[
    :TASK_COUNT
]:
    print(
        f"[{_e.task}] 入力 {' '.join(_e.words)!r}、正解 {' '.join(_e.answer)!r} -> 生成 {_text!r}"
    )

_fig, _ax = plt.subplots(figsize=(8, 4))
_x = np.arange(TASK_COUNT + 1)
_labels = [*TASK_NAMES, "unseen templates"]
_values = [_per_task[t] for t in TASK_NAMES] + [_unseen]
_ax.bar(_x, [np.mean(v) for v in _values], color="tab:blue", alpha=0.6, label="mean over seeds")
for _i, _v in enumerate(_values):
    _ax.scatter([_i] * len(_v), _v, color="black", s=12, zorder=3)
_ax.axhline(C_STAR, color="tab:red", linestyle="--", label="c* (instruction-agnostic upper bound)")
_ax.axhline(BASE_EXACT_MATCH, color="gray", linestyle=":", label="before fine-tuning")
_ax.set_xticks(_x, _labels, rotation=15)
_ax.set_ylim(0, 1.05)
_ax.set_ylabel("exact match")
_ax.set_title(("[smoke test] " if SMOKE_TEST else "") + "Condition 1: exact match by task")
_ax.legend(fontsize=8)
_fig.tight_layout()
plt.show()
```

    c* = 0.2500、M = 1000、|X| = 250、S = 5
    条件 1 の完全一致率(シード順): 0.8180, 0.7600, 0.8070, 0.8380, 0.7970
    p_bar = 0.8040、s_p = 0.0289、Var_x(a_x) = 4.4361e-02、sigma_A = 0.0186(第 1 項 1.67e-04、第 2 項 1.77e-04)
    対比量 p_bar - c* = +0.5540、2 sigma_A = 0.0371
    前提条件: P0=True、P-A=True、P-L=True
    判定関数の結果: 支持
    診断量: 旧基準の第 2 項 p_bar (1 - p_bar) / M = 1.58e-04(旧基準の sigma_A = 0.0180、旧基準の判定関数の結果: 支持)
    診断量: 条件 1 の応答部分の負の対数尤度(シード順): 0.6005, 0.6473, 0.6011, 0.5797, 0.6095(微調整前 8.2724)
    診断量: 未見テンプレートでの完全一致率(シード順): 0.0230, 0.0220, 0.0490, 0.0780, 0.0440
    診断量: 課題別の完全一致率 [reverse](シード順): 0.7120, 0.6640, 0.6880, 0.7600, 0.6680
    診断量: 課題別の完全一致率 [repeat_twice](シード順): 0.8080, 0.7480, 0.8200, 0.8320, 0.8280
    診断量: 課題別の完全一致率 [odd_positions](シード順): 0.9120, 0.8640, 0.9040, 0.9320, 0.9040
    診断量: 課題別の完全一致率 [rotate_left](シード順): 0.8400, 0.7640, 0.8160, 0.8280, 0.7880
    診断量: 条件 1 の形式の遵守率(シード順): 1.0000, 1.0000, 1.0000, 1.0000, 1.0000
    
    --- 条件 1(シード 0)の生成例 ---
    [reverse] 入力 'agreement green billion software genus member novel community'、正解 'community novel member genus software billion green agreement' -> 生成 ' community novel member genus global green green agreement\n### End'
    [repeat_twice] 入力 'agreement green billion software genus member novel community'、正解 'agreement agreement green green billion billion software software genus genus member member novel novel community community' -> 生成 ' agreement agreement green green billion billion software software genus genus member member novel novel community community\n### End'
    [odd_positions] 入力 'agreement green billion software genus member novel community'、正解 'agreement billion genus novel' -> 生成 ' agreement billion genus novel\n### End'
    [rotate_left] 入力 'agreement green billion software genus member novel community'、正解 'green billion software genus member novel community agreement' -> 生成 ' green billion software genus member novel community agreement\n### End'



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/016_supervised_fine_tuning/output_32_1.png)
    


### 6.7 実験 B: 損失マスクの効果と指示の長さの交互作用

評価集合の標本誤差の項は、入力を 4 条件・全シードで共通に復元抽出するブートストラップで求める(6.1 節)。
入力ごとに $K$ 事例の和をとった分子・分母を単位として復元抽出する。これは、事例を入力ごとのクラスタとして復元抽出する
ブートストラップ(015 の方式)と同じ再標本を与えることを、アサーションで確かめる。


```python
def paired_contrast(numerators_by_condition: dict, counts: np.ndarray, weights: dict) -> dict:
    # シードごとの対比量 sum_i weights[i] * (合計_{i,s} / 分母) と、その標準偏差
    # sqrt(s^2 / S + Var_boot)。Var_boot は入力を全条件・全シードで共通に復元抽出するブートストラップ(判定)。
    # 事例を単位とするブートストラップ(旧基準)は診断量として返す。
    totals = {c: [float(row.sum()) for row in rows] for c, rows in numerators_by_condition.items()}
    denominator = int(counts.sum())
    per_seed = np.array(
        [
            sum(w * totals[c][s] / denominator for c, w in weights.items())
            for s in range(NUM_SEEDS_AB)
        ]
    )
    per_example = sum(w * np.mean(numerators_by_condition[c], axis=0) for c, w in weights.items())
    per_input = by_input(per_example)
    counts_by_input = by_input(counts)
    boot = paired_bootstrap_ratio_of_sums(
        per_input, counts_by_input, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
    )
    cluster = paired_cluster_bootstrap_ratio_of_sums(
        per_example, counts, EVAL_INPUT_INDEX, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
    )
    assert np.allclose(boot, cluster, rtol=1e-9, atol=1e-12), (
        "入力単位とクラスタ単位の再標本が一致しない"
    )
    example = paired_bootstrap_ratio_of_sums(
        per_example, counts, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
    )
    seed_std = float(per_seed.std(ddof=1))
    return {
        "per_seed": per_seed,
        "mean": float(per_seed.mean()),
        "seed_std": seed_std,
        "var_boot": float(boot.var(ddof=1)),
        "var_boot_example": float(example.var(ddof=1)),
        "sigma": math.sqrt(seed_std**2 / NUM_SEEDS_AB + float(boot.var(ddof=1))),
        "sigma_example": math.sqrt(seed_std**2 / NUM_SEEDS_AB + float(example.var(ddof=1))),
        "per_input": per_input,
        "counts_by_input": counts_by_input,
        "totals": totals,
    }


_smoke_tag = "[動作確認のみ、結論ではない] " if SMOKE_TEST else ""
# 応答部分のトークン数が全条件・全シードで同一であること(トークンあたりの比較の前提)
for _records in MAIN_RECORDS.values():
    for _r in _records:
        assert np.array_equal(_r["evaluation"]["response_count"], RESPONSE_TOKEN_COUNTS)
_response_sums = {
    c: np.stack([r["evaluation"]["response_sum"] for r in rs]) for c, rs in MAIN_RECORDS.items()
}
CONTRAST_B = paired_contrast(_response_sums, RESPONSE_TOKEN_COUNTS, {4: 1, 3: -1, 2: -1, 1: 1})
D_B, SIGMA_D = CONTRAST_B["mean"], CONTRAST_B["sigma"]
verdict_B = three_way_verdict(-D_B, SIGMA_D)  # 期待する方向は D < 0

# 前提条件 P-B: 指示部分の負の対数尤度(マスクあり - マスクなし)> 2 sigma を両方の長さの水準で
PROMPT_CONTRASTS = {}
for _level, (_a, _u), _long in (("short", (1, 2), False), ("long", (3, 4), True)):
    for _r in MAIN_RECORDS[_a] + MAIN_RECORDS[_u]:
        assert np.array_equal(_r["evaluation"]["prompt_count"], PROMPT_TOKEN_COUNTS[_long])
    _prompt_sums = {
        c: np.stack([r["evaluation"]["prompt_sum"] for r in MAIN_RECORDS[c]]) for c in (_a, _u)
    }
    PROMPT_CONTRASTS[_level] = paired_contrast(
        _prompt_sums, PROMPT_TOKEN_COUNTS[_long], {_a: 1, _u: -1}
    )
precondition_status["P-B"] = bool(
    all(v["mean"] > 2 * v["sigma"] for v in PROMPT_CONTRASTS.values())
)

_losses = {c: [t / RESPONSE_TOKEN_TOTAL for t in CONTRAST_B["totals"][c]] for c in CONDITIONS}
for _c in CONDITIONS:
    print(
        f"{_smoke_tag}L_{_c},s(応答部分、nats / トークン、シード順): "
        + ", ".join(f"{v:.5f}" for v in _losses[_c])
    )
print(f"{_smoke_tag}D_s(シード順): " + ", ".join(f"{v:+.5f}" for v in CONTRAST_B["per_seed"]))
print(
    f"{_smoke_tag}D = {D_B:+.5f}、s_D = {CONTRAST_B['seed_std']:.5f}、Var_boot(D)(入力単位)= "
    f"{CONTRAST_B['var_boot']:.3e}、sigma_D = {SIGMA_D:.5f}、2 sigma_D = {2 * SIGMA_D:.5f}"
)
for _level, _v in PROMPT_CONTRASTS.items():
    print(
        f"{_smoke_tag}P-B [{_level}]: 指示部分の負の対数尤度の差(マスクあり - なし)= {_v['mean']:+.4f}、"
        f"sigma = {_v['sigma']:.4f} -> {'>' if _v['mean'] > 2 * _v['sigma'] else '<='} 2 sigma"
    )
print(
    f"{_smoke_tag}前提条件: P0={precondition_status['P0']}、P-L={precondition_status['P-L(A・B)']}、"
    f"P-B={precondition_status['P-B']}"
)
print(f"{_smoke_tag}判定関数の結果: {verdict_B}")
print(
    f"{_smoke_tag}診断量: 事例を単位とするブートストラップ(旧基準)の Var_boot(D) = "
    f"{CONTRAST_B['var_boot_example']:.3e}(入力単位との比 "
    f"{CONTRAST_B['var_boot'] / CONTRAST_B['var_boot_example']:.2f})、旧基準の sigma_D = "
    f"{CONTRAST_B['sigma_example']:.5f}、旧基準の判定関数の結果: "
    f"{three_way_verdict(-D_B, CONTRAST_B['sigma_example'])}"
)
for _c in CONDITIONS:
    _records = MAIN_RECORDS[_c]
    print(
        f"{_smoke_tag}診断量 条件 {_c}: 完全一致率 "
        + ", ".join(f"{r['evaluation']['exact'].mean():.4f}" for r in _records)
        + "、指示部分の負の対数尤度 "
        + ", ".join(f"{r['evaluation']['prompt_loss']:.3f}" for r in _records)
    )

_fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
for _c, _marker in zip(CONDITIONS, "osD^", strict=True):
    _axes[0].plot(SEEDS_AB, _losses[_c], _marker + "-", label=f"condition {_c}")
_axes[0].set_xlabel("seed")
_axes[0].set_ylabel("response negative log-likelihood (nats / token)")
_axes[0].set_title(("[smoke test] " if SMOKE_TEST else "") + "Response loss by condition")
_axes[0].legend(fontsize=8)
_short = np.array(_losses[2]) - np.array(_losses[1])
_long = np.array(_losses[4]) - np.array(_losses[3])
for _s in SEEDS_AB:
    _axes[1].plot([0, 1], [_short[_s], _long[_s]], "o-", color="gray", alpha=0.6)
_axes[1].axhline(0, color="black", linewidth=0.8)
_axes[1].set_xticks([0, 1], ["short: L2 - L1", "long: L4 - L3"])
_axes[1].set_ylabel("unmasked - masked (nats / token)")
_axes[1].set_title("Advantage of loss masking per seed")
_fig.tight_layout()
plt.show()
```

    L_1,s(応答部分、nats / トークン、シード順): 0.60045, 0.64732, 0.60105, 0.57972, 0.60952
    L_2,s(応答部分、nats / トークン、シード順): 1.82754, 1.52110, 1.85497, 1.58622, 2.20648
    L_3,s(応答部分、nats / トークン、シード順): 1.29383, 0.70892, 0.81696, 0.87589, 1.10478
    L_4,s(応答部分、nats / トークン、シード順): 3.42677, 3.42573, 3.39077, 3.41170, 3.45042
    D_s(シード順): +0.90585, +1.84303, +1.31989, +1.52931, +0.74868
    D = +1.26935、s_D = 0.44790、Var_boot(D)(入力単位)= 2.192e-04、sigma_D = 0.20085、2 sigma_D = 0.40171
    P-B [short]: 指示部分の負の対数尤度の差(マスクあり - なし)= +6.8455、sigma = 0.1346 -> > 2 sigma
    P-B [long]: 指示部分の負の対数尤度の差(マスクあり - なし)= +8.4725、sigma = 0.0830 -> > 2 sigma
    前提条件: P0=True、P-L=True、P-B=True
    判定関数の結果: 反証
    診断量: 事例を単位とするブートストラップ(旧基準)の Var_boot(D) = 1.059e-03(入力単位との比 0.21)、旧基準の sigma_D = 0.20294、旧基準の判定関数の結果: 反証
    診断量 条件 1: 完全一致率 0.8180, 0.7600, 0.8070, 0.8380, 0.7970、指示部分の負の対数尤度 7.945, 8.389, 8.706, 8.050, 8.301
    診断量 条件 2: 完全一致率 0.0570, 0.0760, 0.0780, 0.0590, 0.0200、指示部分の負の対数尤度 1.432, 1.436, 1.432, 1.434, 1.430
    診断量 条件 3: 完全一致率 0.2730, 0.6670, 0.5690, 0.5340, 0.3460、指示部分の負の対数尤度 8.874, 8.798, 9.186, 8.728, 8.757
    診断量 条件 4: 完全一致率 0.0000, 0.0000, 0.0000, 0.0000, 0.0000、指示部分の負の対数尤度 0.395, 0.396, 0.398, 0.397, 0.395



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/016_supervised_fine_tuning/output_34_1.png)
    


### 6.8 コピー能力の診断(判定基準を設けない観察)

学習データに含まれない 8 語の単語列(テンプレートなし)を 2 回繰り返した系列で、1 回目と 2 回目の単語の負の対数尤度の
平均を測る(6.1 節)。対象は微調整前のモデルと条件 1 のシード 0(C-$N_{\max}$ のシード 0 と同一の学習)。
**定性的な観察であり、判定基準を設けない。**


```python
COPY_RESULTS = {
    "base": print_copy_losses("微調整前のモデル", base_model),
    "condition_1_seed_0": print_copy_losses(
        "条件 1・シード 0(= C-N_max・シード 0)", KEPT_CONDITION_1_MODEL
    ),
}
```

    微調整前のモデル: 1 回目の単語 11.426 nats、2 回目の単語 10.691 nats(差 +0.735、64 系列 x 8 語)
    条件 1・シード 0(= C-N_max・シード 0): 1 回目の単語 9.911 nats、2 回目の単語 8.984 nats(差 +0.928、64 系列 x 8 語)


### 6.9 実験 C の学習(C-$N_{\max}$ は条件 1 と共有)

選ばれた段階の実験 C の水準・シード(6.4 節)で学習する(gradient clipping なし)。C-$N_{\max}$ は条件 1 と同一の設定
なので学習を重複して実行せず、条件 1 の同じシードの記録をそのまま使う(共有していることをアサーションで確かめる)。
前提条件 P-L は C-$N_{\max}$ の全シードについて記録し、$N < N_{\max}$ の水準の同じ量は診断量として印字する(6.1 節)。


```python
C_RECORDS = {n: [] for n in C_LEVELS}
_t0_c = time.time()
for _n in C_LEVELS:
    for _seed in SEEDS_C:
        if _n == N_MAX:  # C-N_max は条件 1 と共有する(同じ学習を重複して実行しない)
            C_RECORDS[_n].append(MAIN_RECORDS[1][SEEDS_AB.index(_seed)])
            continue
        _record = run_condition(1, _seed, _n)
        C_RECORDS[_n].append(_record)
        print(
            f"C-{_n}(エポック {_record['max_epochs']:g})seed={_seed}: 学習 {_record['train_seconds']:.1f}s・"
            f"評価 {_record['eval_seconds']:.1f}s"
        )
print(f"実験 C の学習と評価の合計: {time.time() - _t0_c:.1f}s")

# --- C-N_max と条件 1 の共有の確認 ---
for _seed, _record in zip(SEEDS_C, C_RECORDS[N_MAX], strict=True):
    assert _record is MAIN_RECORDS[1][SEEDS_AB.index(_seed)], "C-N_max が条件 1 の記録と同一でない"
    assert (
        _record["condition"] == 1 and _record["num_examples"] == N_MAX and _record["seed"] == _seed
    )
print(
    f"C-N_max(シード {SEEDS_C})は条件 1 の同じシードの学習の記録と同一(共有、再学習していない): OK"
)

# --- 前提条件 P-L(C-N_max の全シードのみ、6.2 節の改訂 7) ---
_limit = PRECONDITION_L_RATIO * BASE_RESPONSE_LOSS[False]
_pl_rows = [r["evaluation"]["response_loss"] <= _limit for r in C_RECORDS[N_MAX]]
precondition_status["P-L(C-N_max)"] = bool(all(_pl_rows))
print(
    f"P-L(C-N_max の全 {len(_pl_rows)} シード): {'成立' if precondition_status['P-L(C-N_max)'] else '不成立'}"
    f"(満たしたシード {sum(_pl_rows)} / {len(_pl_rows)})"
)
# 診断量(判定には使わない): N < N_max の水準で、応答部分の負の対数尤度が微調整前の 1/2 以下か
for _n in C_LEVELS[:-1]:
    print(
        f"診断量 C-{_n}: 応答部分の負の対数尤度 <= {PRECONDITION_L_RATIO} x 微調整前(シード順): "
        + ", ".join(str(bool(r["evaluation"]["response_loss"] <= _limit)) for r in C_RECORDS[_n])
    )
```

    C-16(エポック 4096)seed=0: 学習 60.8s・評価 10.5s
    C-16(エポック 4096)seed=1: 学習 60.8s・評価 11.3s
    C-16(エポック 4096)seed=2: 学習 61.6s・評価 11.8s
    C-16(エポック 4096)seed=3: 学習 61.2s・評価 10.7s
    C-16(エポック 4096)seed=4: 学習 61.0s・評価 11.3s
    C-64(エポック 1024)seed=0: 学習 61.0s・評価 10.3s
    C-64(エポック 1024)seed=1: 学習 60.9s・評価 8.9s
    C-64(エポック 1024)seed=2: 学習 60.7s・評価 9.9s
    C-64(エポック 1024)seed=3: 学習 60.7s・評価 10.3s
    C-64(エポック 1024)seed=4: 学習 60.7s・評価 10.0s
    C-256(エポック 256)seed=0: 学習 61.1s・評価 9.5s
    C-256(エポック 256)seed=1: 学習 61.5s・評価 9.2s
    C-256(エポック 256)seed=2: 学習 61.0s・評価 9.6s
    C-256(エポック 256)seed=3: 学習 61.2s・評価 9.2s
    C-256(エポック 256)seed=4: 学習 61.2s・評価 8.3s
    C-1024(エポック 64)seed=0: 学習 61.6s・評価 6.2s
    C-1024(エポック 64)seed=1: 学習 61.3s・評価 6.9s
    C-1024(エポック 64)seed=2: 学習 62.0s・評価 6.0s
    C-1024(エポック 64)seed=3: 学習 61.2s・評価 6.9s
    C-1024(エポック 64)seed=4: 学習 61.0s・評価 6.2s
    C-4096(エポック 16)seed=0: 学習 61.3s・評価 6.9s
    C-4096(エポック 16)seed=1: 学習 61.5s・評価 6.0s
    C-4096(エポック 16)seed=2: 学習 62.4s・評価 7.3s
    C-4096(エポック 16)seed=3: 学習 62.4s・評価 6.2s
    C-4096(エポック 16)seed=4: 学習 61.6s・評価 6.5s
    C-16384(エポック 4)seed=0: 学習 61.4s・評価 6.2s
    C-16384(エポック 4)seed=1: 学習 61.9s・評価 6.4s
    C-16384(エポック 4)seed=2: 学習 61.6s・評価 6.3s
    C-16384(エポック 4)seed=3: 学習 61.7s・評価 6.3s
    C-16384(エポック 4)seed=4: 学習 61.7s・評価 6.6s
    実験 C の学習と評価の合計: 2088.0s
    C-N_max(シード (0, 1, 2, 3, 4))は条件 1 の同じシードの学習の記録と同一(共有、再学習していない): OK
    P-L(C-N_max の全 5 シード): 成立(満たしたシード 5 / 5)
    診断量 C-16: 応答部分の負の対数尤度 <= 0.5 x 微調整前(シード順): False, False, False, False, False
    診断量 C-64: 応答部分の負の対数尤度 <= 0.5 x 微調整前(シード順): False, False, False, False, False
    診断量 C-256: 応答部分の負の対数尤度 <= 0.5 x 微調整前(シード順): False, False, False, False, False
    診断量 C-1024: 応答部分の負の対数尤度 <= 0.5 x 微調整前(シード順): True, True, True, True, True
    診断量 C-4096: 応答部分の負の対数尤度 <= 0.5 x 微調整前(シード順): True, True, True, True, True
    診断量 C-16384: 応答部分の負の対数尤度 <= 0.5 x 微調整前(シード順): True, True, True, True, True


### 6.10 実験 C: 形式の習得と課題の習得の速さ

評価集合の標本誤差の項 $\mathrm{Var}_{\mathrm{boot}}(\Delta)$ は、入力を復元抽出するブートストラップで求める。各反復で、
入力ごとの正解数(形式の遵守・完全一致、水準 × シード × 入力)に同じ重みを掛けて全水準・両指標・全シードの率を
計算し直し、シードごとの $N_{90}$ と $\Delta$ を求める(6.1 節)。


```python
def n90(values, levels: tuple[int, ...]) -> tuple[float, bool]:
    # 飽和値(最大水準の値)の 90% に初めて達するデータ数の log2 と、打ち切りの有無(6.1 節)
    threshold = 0.9 * values[-1]
    logs = [math.log2(n) for n in levels]
    if values[0] >= threshold:
        return logs[0], True
    for j in range(1, len(levels)):
        if values[j] >= threshold:
            fraction = (threshold - values[j - 1]) / (values[j] - values[j - 1])
            return logs[j - 1] + fraction * (logs[j] - logs[j - 1]), False
    raise AssertionError("最大水準の値は必ず閾値以上になる")


def delta_from_rates(
    format_rates: np.ndarray, exact_rates: np.ndarray, levels
) -> tuple[float, list]:
    # format_rates・exact_rates: 形状 (水準, シード)。シードごとの N90 と、その差の平均 Delta を返す。
    rows = []
    for s in range(format_rates.shape[1]):
        log_format, censored_format = n90(list(format_rates[:, s]), levels)
        log_task, censored_task = n90(list(exact_rates[:, s]), levels)
        rows.append(
            {
                "log2_n90_format": log_format,
                "log2_n90_task": log_task,
                "censored_format": censored_format,
                "censored_task": censored_task,
                "difference": log_task - log_format,
            }
        )
    return float(np.mean([r["difference"] for r in rows])), rows


def experiment_c_statistics(
    format_by_input, exact_by_input, levels, num_tasks: int, resamples: int, seed: int
) -> dict:
    # format_by_input・exact_by_input: 形状 (水準, シード, 入力) の正解数(各 0..K)。
    format_by_input = np.asarray(format_by_input, dtype=np.float64)
    exact_by_input = np.asarray(exact_by_input, dtype=np.float64)
    num_inputs = format_by_input.shape[2]
    denominator = num_inputs * num_tasks
    format_rates = format_by_input.sum(axis=2) / denominator
    exact_rates = exact_by_input.sum(axis=2) / denominator
    delta, rows = delta_from_rates(format_rates, exact_rates, levels)
    num_seeds = format_rates.shape[1]
    seed_std = float(np.std([r["difference"] for r in rows], ddof=1))
    # 入力を復元抽出するブートストラップ: 各反復で全水準・両指標・全シードに同じ重みを使う
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        num_inputs, np.full(num_inputs, 1.0 / num_inputs), size=resamples
    ).astype(np.float64)
    format_boot = (
        np.einsum("rx,lsx->rls", weights, format_by_input) / denominator
    )  # sum_x w_x = |X|
    exact_boot = np.einsum("rx,lsx->rls", weights, exact_by_input) / denominator
    deltas = np.array(
        [delta_from_rates(format_boot[r], exact_boot[r], levels)[0] for r in range(resamples)]
    )
    var_boot = float(deltas.var(ddof=1))
    return {
        "rows": rows,
        "delta": delta,
        "seed_std": seed_std,
        "var_boot": var_boot,
        "sigma": math.sqrt(seed_std**2 / num_seeds + var_boot),
        "sigma_old": seed_std / math.sqrt(num_seeds),
        "saturation_exact": float(exact_rates[-1].mean()),
        "saturation_format": float(format_rates[-1].mean()),
        "format_rates": format_rates,
        "exact_rates": exact_rates,
    }


_smoke_tag = "[動作確認のみ、結論ではない] " if SMOKE_TEST else ""
FORMAT_BY_INPUT_C = [
    [
        [int(v) for v in by_input(r["evaluation"]["format_ok"].astype(np.int64))]
        for r in C_RECORDS[n]
    ]
    for n in C_LEVELS
]
EXACT_BY_INPUT_C = [
    [[int(v) for v in by_input(r["evaluation"]["exact"].astype(np.int64))] for r in C_RECORDS[n]]
    for n in C_LEVELS
]
STATISTICS_C = experiment_c_statistics(
    FORMAT_BY_INPUT_C, EXACT_BY_INPUT_C, C_LEVELS, TASK_COUNT, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
)
for _j, _n in enumerate(C_LEVELS):  # 入力ごとの和から求めた率が、事例ごとの平均と一致すること
    for _s, _r in enumerate(C_RECORDS[_n]):
        assert math.isclose(
            STATISTICS_C["format_rates"][_j, _s], float(_r["evaluation"]["format_ok"].mean())
        )
        assert math.isclose(
            STATISTICS_C["exact_rates"][_j, _s], float(_r["evaluation"]["exact"].mean())
        )
DELTA_C, SIGMA_C = STATISTICS_C["delta"], STATISTICS_C["sigma"]
verdict_C = three_way_verdict(DELTA_C, SIGMA_C)
precondition_status["P-C"] = bool(
    STATISTICS_C["saturation_exact"] >= 2 * C_STAR and STATISTICS_C["saturation_format"] >= 0.5
)

print(f"{_smoke_tag}水準 N: {C_LEVELS}")
print(
    f"{_smoke_tag}{'N':>6} | {'形式の遵守率(シード順)':<40} | {'完全一致率(シード順)':<40} | 応答部分の負の対数尤度"
)
for _j, _n in enumerate(C_LEVELS):
    print(
        f"{_smoke_tag}{_n:>6} | "
        + ", ".join(f"{v:.3f}" for v in STATISTICS_C["format_rates"][_j]).ljust(40)
        + " | "
        + ", ".join(f"{v:.3f}" for v in STATISTICS_C["exact_rates"][_j]).ljust(40)
        + " | "
        + ", ".join(f"{v:.3f}" for v in response_loss_by_seed(C_RECORDS[_n]))
    )
for _s, _row in enumerate(STATISTICS_C["rows"]):
    print(
        f"{_smoke_tag}seed={_s}: log2 N90(形式)= {_row['log2_n90_format']:.3f}"
        f"{'(打ち切り)' if _row['censored_format'] else ''}、log2 N90(課題)= {_row['log2_n90_task']:.3f}"
        f"{'(打ち切り)' if _row['censored_task'] else ''}、差 {_row['difference']:+.3f}"
    )
print(
    f"{_smoke_tag}Delta = {DELTA_C:+.4f}、s_Delta = {STATISTICS_C['seed_std']:.4f}、Var_boot(Delta)(入力単位)= "
    f"{STATISTICS_C['var_boot']:.4e}、sigma_Delta = {SIGMA_C:.4f}、2 sigma_Delta = {2 * SIGMA_C:.4f}"
)
print(
    f"{_smoke_tag}P-C: N_max での完全一致率 {STATISTICS_C['saturation_exact']:.4f} >= 2c* = {2 * C_STAR:.2f}、"
    f"形式の遵守率 {STATISTICS_C['saturation_format']:.4f} >= 0.5 -> {precondition_status['P-C']}"
)
print(
    f"{_smoke_tag}前提条件: P0={precondition_status['P0']}、P-L(C-N_max)={precondition_status['P-L(C-N_max)']}、"
    f"P-C={precondition_status['P-C']}"
)
print(f"{_smoke_tag}判定関数の結果: {verdict_C}")
print(
    f"{_smoke_tag}診断量: 旧基準の sigma_Delta = s_Delta / sqrt(S) = {STATISTICS_C['sigma_old']:.4f}"
    f"(旧基準の判定関数の結果: {three_way_verdict(DELTA_C, STATISTICS_C['sigma_old'])})"
)

_fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
for _label, _rates, _color in (
    ("format compliance", STATISTICS_C["format_rates"], "tab:green"),
    ("exact match", STATISTICS_C["exact_rates"], "tab:blue"),
):
    _axes[0].plot(C_LEVELS, _rates.mean(axis=1), "o-", color=_color, label=f"{_label} (mean)")
    _axes[0].fill_between(C_LEVELS, _rates.min(axis=1), _rates.max(axis=1), color=_color, alpha=0.2)
_axes[0].axhline(C_STAR, color="tab:red", linestyle="--", label="c*")
_axes[0].set_xscale("log", base=2)
_axes[0].set_xlabel("number of training examples N")
_axes[0].set_ylabel("rate on the evaluation set")
_axes[0].set_title(
    ("[smoke test] " if SMOKE_TEST else "") + "Format vs task (mean and range over seeds)"
)
_axes[0].legend(fontsize=8)
_losses_c = np.array([response_loss_by_seed(C_RECORDS[n]) for n in C_LEVELS])
_axes[1].plot(C_LEVELS, _losses_c.mean(axis=1), "o-")
_axes[1].fill_between(C_LEVELS, _losses_c.min(axis=1), _losses_c.max(axis=1), alpha=0.2)
_axes[1].axhline(BASE_RESPONSE_LOSS[False], color="gray", linestyle=":", label="before fine-tuning")
_axes[1].set_xscale("log", base=2)
_axes[1].set_xlabel("number of training examples N")
_axes[1].set_ylabel("response negative log-likelihood (nats / token)")
_axes[1].set_title("Response loss by N")
_axes[1].legend(fontsize=8)
_fig.tight_layout()
plt.show()
```

    水準 N: (16, 64, 256, 1024, 4096, 16384, 65536)
         N | 形式の遵守率(シード順)                             | 完全一致率(シード順)                              | 応答部分の負の対数尤度
        16 | 0.901, 0.842, 0.862, 0.945, 0.943        | 0.000, 0.000, 0.000, 0.000, 0.000        | 5.126, 5.169, 4.977, 4.969, 5.050
        64 | 0.984, 0.987, 0.975, 0.987, 0.963        | 0.000, 0.000, 0.000, 0.000, 0.000        | 4.995, 4.996, 5.061, 4.994, 4.929
       256 | 0.992, 0.983, 0.982, 0.974, 0.991        | 0.000, 0.000, 0.000, 0.000, 0.000        | 4.328, 4.341, 4.383, 4.298, 4.376
      1024 | 1.000, 1.000, 1.000, 1.000, 1.000        | 0.480, 0.418, 0.446, 0.334, 0.517        | 0.929, 0.997, 0.986, 1.194, 0.917
      4096 | 1.000, 1.000, 1.000, 1.000, 1.000        | 0.607, 0.740, 0.681, 0.564, 0.795        | 0.779, 0.664, 0.681, 0.783, 0.634
     16384 | 1.000, 1.000, 1.000, 1.000, 1.000        | 0.749, 0.824, 0.773, 0.791, 0.835        | 0.644, 0.602, 0.619, 0.623, 0.584
     65536 | 1.000, 1.000, 1.000, 1.000, 1.000        | 0.818, 0.760, 0.807, 0.838, 0.797        | 0.600, 0.647, 0.601, 0.580, 0.610
    seed=0: log2 N90(形式)= 4.000(打ち切り)、log2 N90(課題)= 13.820、差 +9.820
    seed=1: log2 N90(形式)= 4.800、log2 N90(課題)= 11.652、差 +6.852
    seed=2: log2 N90(形式)= 4.673、log2 N90(課題)= 12.985、差 +8.312
    seed=3: log2 N90(形式)= 4.000(打ち切り)、log2 N90(課題)= 13.676、差 +9.676
    seed=4: log2 N90(形式)= 4.000(打ち切り)、log2 N90(課題)= 11.441、差 +7.441
    Delta = +8.4202、s_Delta = 1.3195、Var_boot(Delta)(入力単位)= 1.6789e-02、sigma_Delta = 0.6042、2 sigma_Delta = 1.2083
    P-C: N_max での完全一致率 0.8040 >= 2c* = 0.50、形式の遵守率 1.0000 >= 0.5 -> True
    前提条件: P0=True、P-L(C-N_max)=True、P-C=True
    判定関数の結果: 支持
    診断量: 旧基準の sigma_Delta = s_Delta / sqrt(S) = 0.5901(旧基準の判定関数の結果: 支持)



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/016_supervised_fine_tuning/output_40_1.png)
    


### 6.11 不変条件のアサーションと`SMOKE_TEST`の配線

- 全ての実行で、ステップ数・学習履歴の長さが $T$、学習可能パラメータ数が閉形式 $4 L r d_{\mathrm{model}}$ と一致し、
  gradient clipping が一度も発動していない(無効)。
- 条件 1〜4: データ数が $N_{\max}$(1 エポックちょうど)。同じシードでは条件 1〜4 のミニバッチの添字が同一、
  異なるシードでは異なる。評価の配列の長さが $M$。
- 実験 C: データの部分集合が入れ子であり(各水準は学習データの先頭 $N$ 個)、エポック数が $Tb/N$、ミニバッチの添字の
  ハッシュが同じ引数で作り直した値と一致する。C-$N_{\max}$ は条件 1 の記録と同一(6.9 節)。
- 評価集合の事例ごとの応答部分のトークン数が全実行で同一。
- ベースモデルの重みが、全ての学習の後も Hub から読み込んだ重みと完全に一致する。
- `SMOKE_TEST`の配線と削る段階: 実際に使ったシード・水準・ステップ数・データ数・評価集合の大きさ・ブートストラップの
  反復回数が、5.2 節の水準と 6.4 節で選ばれた段階と一致する。


```python
_all = [r for rs in MAIN_RECORDS.values() for r in rs]
for _r in _all:
    assert _r["num_steps"] == _r["history_length"] == NUM_STEPS, "ステップ数が T と一致しない"
    assert _r["num_examples"] == N_MAX and _r["max_epochs"] == 1.0
    assert _r["num_trainable_parameters"] == LORA_PARAMETERS
    assert not _r["any_clip_triggered"], "gradient clipping が発動した"
    assert len(_r["evaluation"]["exact"]) == EVAL_SIZE
for _s in SEEDS_AB:
    assert len({MAIN_RECORDS[c][_s]["batches_hash"] for c in CONDITIONS}) == 1, (
        "条件間でデータ順序が異なる"
    )
assert len({MAIN_RECORDS[1][s]["batches_hash"] for s in SEEDS_AB}) == NUM_SEEDS_AB
for _n, _records in C_RECORDS.items():
    for _s, _r in zip(SEEDS_C, _records, strict=True):
        assert _r["num_steps"] == _r["history_length"] == NUM_STEPS, "ステップ数が T と一致しない"
        assert _r["num_examples"] == _n and _r["seed"] == _s and not _r["any_clip_triggered"]
        assert _r["num_trainable_parameters"] == LORA_PARAMETERS
        assert _r["max_epochs"] == NUM_STEPS * BATCH_SIZE / _n
        assert _r["batches_hash"] == hash_json(make_epoch_batches(_n, NUM_STEPS, BATCH_SIZE, _s))
        assert np.array_equal(_r["evaluation"]["response_count"], RESPONSE_TOKEN_COUNTS)
assert all(
    max(i for b in make_epoch_batches(n, NUM_STEPS, BATCH_SIZE, 0) for i in b) == n - 1
    for n in C_LEVELS
)
assert all(C_RECORDS[N_MAX][i] is MAIN_RECORDS[1][i] for i in range(NUM_SEEDS_C))
_hub_state = torch.load(MODEL_STATE_PATH, map_location="cpu")
assert all(torch.equal(v.cpu(), _hub_state[k]) for k, v in base_model.state_dict().items())
# SMOKE_TEST の配線と削る段階
assert all([r["seed"] for r in rs] == list(SEEDS_AB) for rs in MAIN_RECORDS.values())
assert (
    tuple(range(STAGE["NUM_SEEDS_AB"])) == SEEDS_AB
    and tuple(range(STAGE["NUM_SEEDS_C"])) == SEEDS_C
)
assert (
    tuple(C_RECORDS) == C_LEVELS == c_levels(STAGE["C_MIN_LEVEL"], N_MAX) and C_LEVELS[-1] == N_MAX
)
assert STAGES[CURRENT_LEVEL_NAME][SELECTED_STAGE] == STAGE and CFG["NUM_STEPS"] == NUM_STEPS
assert CFG["NUM_EVAL_INPUTS"] * TASK_COUNT == EVAL_SIZE and CFG["NUM_STEPS"] * BATCH_SIZE == N_MAX
assert paired_bootstrap_ratio_of_sums(
    np.ones(NUM_EVAL_INPUTS), np.ones(NUM_EVAL_INPUTS), BOOTSTRAP_RESAMPLES, 0
).shape == (CFG["BOOTSTRAP_RESAMPLES"],)
_num_c_runs = sum(len(rs) for rs in C_RECORDS.values())
print(
    f"条件 1〜4 の全 {len(_all)} 実行と実験 C の全 {_num_c_runs} 実行(うち C-N_max の {NUM_SEEDS_C} 実行は条件 1 と共有): "
    "ステップ数・データ数・学習可能パラメータ数・gradient clipping の不発動・データ順序・入れ子・応答部分のトークン数: OK"
)
print("ベースモデルの重みが Hub から読み込んだ重みと完全一致: OK")
print(
    f"SMOKE_TEST の配線と削る段階: 水準 {CURRENT_LEVEL_NAME!r}・段階 {SELECTED_STAGE}(実験 A・B のシード {SEEDS_AB}、"
    f"実験 C のシード {SEEDS_C}、水準 {C_LEVELS}、T {NUM_STEPS}、M {EVAL_SIZE}、反復 {BOOTSTRAP_RESAMPLES})が一致: OK"
)
```

    条件 1〜4 の全 20 実行と実験 C の全 35 実行(うち C-N_max の 5 実行は条件 1 と共有): ステップ数・データ数・学習可能パラメータ数・gradient clipping の不発動・データ順序・入れ子・応答部分のトークン数: OK
    ベースモデルの重みが Hub から読み込んだ重みと完全一致: OK
    SMOKE_TEST の配線と削る段階: 水準 'prod'・段階 0(実験 A・B のシード (0, 1, 2, 3, 4)、実験 C のシード (0, 1, 2, 3, 4)、水準 (16, 64, 256, 1024, 4096, 16384, 65536)、T 2048、M 1000、反復 10000)が一致: OK


### 6.12 判定結果の一覧

前提条件が 1 つでも不成立の実験は、判定関数の結果に関わらず「前提不成立」とする(判定不能とは区別する)。
判定関数の結果は参考として別欄に残す。標準偏差は入力を単位とする新基準による。旧基準(事例を単位とする)の
標準偏差も、判定に使わない参考値として並べる。選ばれた段階と、判定に実際に使ったシード数・水準を印字する。


```python
def verdict_label(computed: str, preconditions: list[str]) -> str:
    return computed if all(precondition_status.get(p) for p in preconditions) else "前提不成立"


_smoke_tag = "[動作確認のみ、結論ではない] " if SMOKE_TEST else ""
print(f"{_smoke_tag}段階の選択: {STAGE_SELECTION_MESSAGE}")
print(
    f"{_smoke_tag}判定に使った値: 実験 A・B は S = {NUM_SEEDS_AB}(シード {SEEDS_AB})、実験 C は S = {NUM_SEEDS_C}"
    f"(シード {SEEDS_C})・水準 {C_LEVELS}(N_min = {C_LEVELS[0]})"
)
PRECONDITIONS = {
    "A": ["P0", "P-L(A・B)", "P-A"],
    "B": ["P0", "P-L(A・B)", "P-B"],
    "C": ["P0", "P-L(C-N_max)", "P-C"],
}
print(
    f"{_smoke_tag}実験 | 対比量(sigma は入力単位、括弧内は旧基準) | 前提条件の成否 | 判定関数の結果(参考) | 最終判定"
)
for _name, _stat, _computed in (
    (
        "A",
        f"p_bar - c* = {CONTRAST_A:+.4f}、sigma_A = {SIGMA_A:.4f}({STATISTICS_A['sigma_old']:.4f})",
        verdict_A,
    ),
    ("B", f"D = {D_B:+.5f}、sigma_D = {SIGMA_D:.5f}({CONTRAST_B['sigma_example']:.5f})", verdict_B),
    (
        "C",
        f"Delta = {DELTA_C:+.4f}、sigma_Delta = {SIGMA_C:.4f}({STATISTICS_C['sigma_old']:.4f})",
        verdict_C,
    ),
):
    _pre = ", ".join(f"{p}={precondition_status.get(p)}" for p in PRECONDITIONS[_name])
    print(
        f"{_smoke_tag}{_name} | {_stat} | {_pre} | {_computed} | {verdict_label(_computed, PRECONDITIONS[_name])}"
    )
if SMOKE_TEST:
    print(
        "上記はスモークテストによるコードの動作確認であり、本番実行の後に数値が変わるため、結論としては扱わない。"
    )
```

    段階の選択: 予算 120 分に収まる最小の段階として、段階 0 を選んだ(見積もり 79.1 分、cuda 基準)
    判定に使った値: 実験 A・B は S = 5(シード (0, 1, 2, 3, 4))、実験 C は S = 5(シード (0, 1, 2, 3, 4))・水準 (16, 64, 256, 1024, 4096, 16384, 65536)(N_min = 16)
    実験 | 対比量(sigma は入力単位、括弧内は旧基準) | 前提条件の成否 | 判定関数の結果(参考) | 最終判定
    A | p_bar - c* = +0.5540、sigma_A = 0.0186(0.0180) | P0=True, P-L(A・B)=True, P-A=True | 支持 | 支持
    B | D = +1.26935、sigma_D = 0.20085(0.20294) | P0=True, P-L(A・B)=True, P-B=True | 反証 | 反証
    C | Delta = +8.4202、sigma_Delta = 0.6042(0.5901) | P0=True, P-L(C-N_max)=True, P-C=True | 支持 | 支持




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/04_alignment/016_supervised_fine_tuning.ipynb
