---
title: "DPO(Direct Preference Optimization) / Direct Preference Optimization(実装・実験編 4/4)"
---

この記事は後編(実装・実験編 4/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/018_direct_preference_optimization-practice-3)。

### 6.12 実験 C: 尤度の置き換わりの存在

実験 B の条件 2(DPO・確率的なラベル・標準の $\beta$)の記録を使う(学習を再実行しない)。


```python
_chosen = np.array(
    [b_contribution(2, s, TRAIN_STEPS, "chosen_log_ratio") for s in SEEDS_B]
)  # (S, N)
C_SEEDS = _chosen.mean(axis=1)
_boot_c = (PAIR_BOOTSTRAP_COUNTS @ _chosen.mean(axis=0).astype(np.float32)) / NUM_MAIN_PAIRS
CONTRAST_C = {
    "value": float(C_SEEDS.mean()),
    "seed_values": C_SEEDS.tolist(),
    "var_boot": float(_boot_c.astype(np.float64).var(ddof=1)),
}
CONTRAST_C["sigma"] = math.sqrt(
    float(np.std(C_SEEDS, ddof=1)) ** 2 / len(SEEDS_B) + CONTRAST_C["var_boot"]
)
verdict_C = three_way_verdict(-CONTRAST_C["value"], CONTRAST_C["sigma"])  # 負の向きが支持
_margins = _h_bar[(2, TRAIN_STEPS)]
precondition_status["PC"] = bool(np.all(_margins > 0))
_dataset = dataset_for("main", "stochastic", 0)
_len_first = np.array([len(r) for r in _dataset["first"]], dtype=np.float64)
_len_second = np.array([len(r) for r in _dataset["second"]], dtype=np.float64)
_per_token = [
    float(
        _chosen[i].sum()
        / (
            dataset_for("main", "stochastic", s)["p_hat"] * _len_first
            + (1 - dataset_for("main", "stochastic", s)["p_hat"]) * _len_second
        ).sum()
    )
    for i, s in enumerate(SEEDS_B)
]
_rejected = np.array(
    [b_contribution(2, s, TRAIN_STEPS, "rejected_log_ratio").mean() for s in SEEDS_B]
)
print(
    f"選好された応答の対数比の平均 Delta_w(T)(系列単位、nats)= {CONTRAST_C['value']:+.4f}、シードごと "
    + ", ".join(f"{v:+.4f}" for v in C_SEEDS)
    + f"、Var_boot = {CONTRAST_C['var_boot']:.3e}、sigma = {CONTRAST_C['sigma']:.4f}、2 sigma = {2 * CONTRAST_C['sigma']:.4f}"
)
print(
    "PC: マージンの平均 h(T)(シード順)"
    + ", ".join(f"{v:+.4f}" for v in _margins)
    + f" > 0 -> {'成立' if precondition_status['PC'] else '不成立'}。学習の成立(B・C): {precondition_status['学習の成立(B・C)']}"
)
print(f"{_tag}判定関数の結果: {verdict_C}")
print(
    "診断量(シード平均): "
    + dumps_compact_json(
        {
            "chosen_log_ratio_per_token": round(float(np.mean(_per_token)), 5),
            "rejected_log_ratio": round(float(_rejected.mean()), 4),
            "margin": round(float(_margins.mean()), 4),
        }
    )
)
_fig, _ax = plt.subplots(figsize=(7, 4))
for _key, _label in (
    ("chosen_log_ratio", "log pi(y_w) - log pi_ref(y_w)"),
    ("rejected_log_ratio", "log pi(y_l) - log pi_ref(y_l)"),
    ("mean_margin", "margin h"),
):
    _curves = np.array([RECORDS[key_b(2, s)]["history"][_key] for s in SEEDS_B]).mean(axis=0)
    _window = max(1, TRAIN_STEPS // 32)
    _ax.plot(
        np.arange(len(_curves) - _window + 1) + _window,
        np.convolve(_curves, np.ones(_window) / _window, mode="valid"),
        label=_label,
    )
_ax.axhline(0, color="gray", linewidth=0.8)
_ax.set_xlabel("step")
_ax.set_ylabel("batch mean (nats, moving average)")
_ax.set_title(f"DPO, stochastic labels, beta = {STANDARD_BETA:g} (experiment B condition 2)")
_ax.legend()
plt.tight_layout()
plt.show()
```

    選好された応答の対数比の平均 Delta_w(T)(系列単位、nats)= -0.0750、シードごと -0.0503, -0.0382, -0.0801, -0.1626, -0.0435、Var_boot = 1.191e-04、sigma = 0.0255、2 sigma = 0.0511
    PC: マージンの平均 h(T)(シード順)+1.3696, +1.3842, +1.4538, +1.4721, +1.3735 > 0 -> 成立。学習の成立(B・C): True
    判定関数の結果: 支持
    診断量(シード平均): {"chosen_log_ratio_per_token": -0.00596, "rejected_log_ratio": -1.4856, "margin": 1.4107}



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/018_direct_preference_optimization/output_47_1.png)
    


### 6.13 実験 D: 尤度の置き換わりの類似度依存性

ブートストラップは、各層の中で近い組・遠い組をそれぞれ復元抽出する(層ごとの件数を保つ)。再標本は全シードで共通である。
CHES スコアは参照方策の隠れ状態で計算し、シードごとのラベルの向き($\hat{p}$)で重み付けして平均する。


```python
def stratified_bootstrap_counts(rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # 各層の中で復元抽出した抽出回数の行列 (B, len(rows))
    counts = np.zeros((BOOTSTRAP_RESAMPLES, len(rows)), dtype=np.float32)
    labels = [D_STRATA[r] for r in rows]
    for label in D_COUNTS:
        columns = [j for j, value in enumerate(labels) if value == label]
        if columns:
            counts[:, columns] = rng.multinomial(
                len(columns), np.full(len(columns), 1.0 / len(columns)), size=BOOTSTRAP_RESAMPLES
            )
    return counts


_rng_d = np.random.default_rng(BOOTSTRAP_SEED + 2)
D_BOOTSTRAP_COUNTS = {
    "near": stratified_bootstrap_counts(NEAR_ROWS, _rng_d),
    "far": stratified_bootstrap_counts(FAR_ROWS, _rng_d),
}
_d_chosen = {
    name: np.array(
        [
            pair_contributions(
                RECORDS[key_d(name, s)]["history"]["evaluations"][TRAIN_STEPS],
                dataset_for(name, "stochastic", s)["p_hat"],
            )["chosen_log_ratio"]
            for s in SEEDS_D
        ]
    )
    for name in ("near", "far")
}  # 各 (S, n)
D_SEEDS = _d_chosen["near"].mean(axis=1) - _d_chosen["far"].mean(axis=1)
_boot_e = (D_BOOTSTRAP_COUNTS["near"] @ _d_chosen["near"].mean(axis=0).astype(np.float32)) / len(
    NEAR_ROWS
) - (D_BOOTSTRAP_COUNTS["far"] @ _d_chosen["far"].mean(axis=0).astype(np.float32)) / len(FAR_ROWS)
CONTRAST_D = {
    "value": float(D_SEEDS.mean()),
    "seed_values": D_SEEDS.tolist(),
    "var_boot": float(_boot_e.astype(np.float64).var(ddof=1)),
}
CONTRAST_D["sigma"] = math.sqrt(
    float(np.std(D_SEEDS, ddof=1)) ** 2 / len(SEEDS_D) + CONTRAST_D["var_boot"]
)
verdict_D = three_way_verdict(-CONTRAST_D["value"], CONTRAST_D["sigma"])  # 負の向きが支持
_d_margin = {
    n: np.array([RECORDS[key_d(n, s)]["final"]["margin"] for s in SEEDS_D]) for n in ("near", "far")
}
precondition_status["PD"] = bool(all(np.all(v > 0) for v in _d_margin.values()))

# --- 診断量: CHES スコア・応答の長さ・正規化編集距離 ---
_t0 = time.time()
D_DIAGNOSTICS = {}
for _name in ("near", "far"):
    _ds = dataset_for(_name, "stochastic", SEEDS_D[0])
    _stats = compute_ches_statistics(
        reference_policy, _ds["prompts"], _ds["first"], _ds["second"], device, LOG_PROB_BATCH_SIZE
    )
    _ches, _ches_norm = [], []
    for _s in SEEDS_D:
        _p = dataset_for(_name, "stochastic", _s)["p_hat"]
        _ches.append(
            float(
                np.mean(
                    _p * ches_from_statistics(_stats, np.ones_like(_p, bool))
                    + (1 - _p) * ches_from_statistics(_stats, np.zeros_like(_p, bool))
                )
            )
        )
        _ches_norm.append(
            float(
                np.mean(
                    _p * ches_from_statistics(_stats, np.ones_like(_p, bool), True)
                    + (1 - _p) * ches_from_statistics(_stats, np.zeros_like(_p, bool), True)
                )
            )
        )
    _rows = NEAR_ROWS if _name == "near" else FAR_ROWS
    _lengths = np.concatenate([_stats["length_first"], _stats["length_second"]])
    D_DIAGNOSTICS[_name] = {
        "pairs": len(_rows),
        "normalized_edit_distance": round(
            float(CANDIDATES["distance"][VALID_INDEX[_rows]].mean()), 4
        ),
        "abs_delta_r": round(float(np.abs(VALID_DELTA[_rows]).mean()), 4),
        "ches": round(float(np.mean(_ches)), 2),
        "ches_length_normalized": round(float(np.mean(_ches_norm)), 4),
        "response_tokens_quantiles": np.quantile(_lengths, [0.1, 0.5, 0.9]).tolist(),
        "margin": round(float(_d_margin[_name].mean()), 4),
        "final_accuracy": round(
            float(np.mean([RECORDS[key_d(_name, s)]["final"]["accuracy"] for s in SEEDS_D])), 4
        ),
    }
CHES_SECONDS = time.time() - _t0
print(f"S_D = {len(SEEDS_D)}、近い組 {len(NEAR_ROWS):,}・遠い組 {len(FAR_ROWS):,}")
print(
    f"E = Delta_w(近い組) - Delta_w(遠い組) = {CONTRAST_D['value']:+.4f}、シードごと "
    + ", ".join(f"{v:+.4f}" for v in D_SEEDS)
    + f"、Var_boot = {CONTRAST_D['var_boot']:.3e}、sigma = {CONTRAST_D['sigma']:.4f}、2 sigma = {2 * CONTRAST_D['sigma']:.4f}"
)
print(
    "PD: マージンの平均 h(T) > 0(近い組・遠い組、シード順)"
    + "; ".join(f"{n}: " + ", ".join(f"{v:+.3f}" for v in _d_margin[n]) for n in _d_margin)
    + f" -> {'成立' if precondition_status['PD'] else '不成立'}。学習の成立(D): {precondition_status['学習の成立(D)']}"
)
print(f"{_tag}判定関数の結果: {verdict_D}")
print(
    f"診断量(CHES スコアは参照方策で計算、{CHES_SECONDS:.1f}s): "
    + dumps_compact_json(D_DIAGNOSTICS)
)
```

    S_D = 5、近い組 1,990・遠い組 1,990
    E = Delta_w(近い組) - Delta_w(遠い組) = -0.0292、シードごと +0.0070, -0.0324, -0.0616, -0.0274, -0.0318、Var_boot = 2.691e-04、sigma = 0.0197、2 sigma = 0.0394
    PD: マージンの平均 h(T) > 0(近い組・遠い組、シード順)near: +0.908, +0.941, +0.850, +0.820, +0.928; far: +1.900, +1.941, +2.100, +1.999, +1.961 -> 成立。学習の成立(D): True
    判定関数の結果: 判定不能
    診断量(CHES スコアは参照方策で計算、2.3s): {
      "near": {
        "pairs": 1990,
        "normalized_edit_distance": 0.3887,
        "abs_delta_r": 0.113,
        "ches": -19.43,
        "ches_length_normalized": -2.509,
        "response_tokens_quantiles": [8.0, 11.0, 16.0],
        "margin": 0.8893,
        "final_accuracy": 0.7392
      },
      "far": {
        "pairs": 1990,
        "normalized_edit_distance": 0.6872,
        "abs_delta_r": 0.1594,
        "ches": 1492.32,
        "ches_length_normalized": -8.636,
        "response_tokens_quantiles": [9.0, 13.0, 22.0],
        "margin": 1.9804,
        "final_accuracy": 0.7661
      }
    }


### 6.14 不変条件のアサーションと`SMOKE_TEST`の配線

- すべての学習の記録で、ステップ数が $T$、記録されたキーが同じで、学習用の組の評価がステップ $T/2$ と $T$ で行われている。
- 同じシード・同じ大きさのデータの学習は、損失・ラベルの種類・$\beta$ によらず、同じミニバッチの順序を使った(条件間で
  揃えた量)。実験 D の近い組・遠い組は同じ大きさである。
- すべての評価で、同じ評価用のプロンプト・同じサンプル数($m$)を使った。
- 実験 A の水準は格子の連続する 6 点で公比 $\sqrt{2}$、崩壊の領域の観察の 2 学習(シード 0)があり、実験 B の 4 条件と実験 D の 2 データセットがそろっている。
- 参照方策の重みが、すべての学習の後も読み込み直後と bit 単位で一致する(学習が参照方策を書き換えていない)。
- **`SMOKE_TEST`の配線**: 5.2 節で印字した実効水準(シード数・$\beta$ の水準の数・ステップ数・$K$・データ量)と、実際に使われた
  値(history の長さ・ラベルの行列の形・学習の数)が一致する。


```python
_keys = set(RECORDS[next(iter(RECORDS))]["history"])
for _key, _r in RECORDS.items():
    _h = _r["history"]
    assert set(_h) == _keys, _key
    assert all(len(_h[k]) == TRAIN_STEPS for k in _keys if k != "evaluations"), _key
    assert set(_h["evaluations"]) == {HALF_STEP, TRAIN_STEPS}, _key
    assert _r["eval"]["count_per_input"] == SAMPLES_PER_PROMPT * TASK_COUNT
    assert len(_r["eval"]["true_reward_by_input"]) == NUM_EVAL_INPUTS
_by_seed_size: dict = {}
for _key, _r in RECORDS.items():
    _size = len(dataset_for(_r["dataset"], _r["label_kind"], _r["seed"])["instances"]["prompts"])
    _by_seed_size.setdefault((_r["seed"], _size), set()).add(_r["batches_hash"])
assert all(len(v) == 1 for v in _by_seed_size.values()), (
    "同じシード・同じ大きさで異なるミニバッチの順序がある"
)
assert len(NEAR_ROWS) == len(FAR_ROWS)
assert len(A_BETAS) == NUM_A_LEVELS and len(B_CONDITIONS) == 4
assert all(math.isclose(a / b, math.sqrt(2.0)) for a, b in zip(A_BETAS, A_BETAS[1:], strict=False))
assert all(
    b <= min(A_BETAS) for b in COLLAPSE_BETAS
)  # 格子の下端で打ち切った点は beta_6 と一致しうる
_expected_runs = len(
    {key_a(b, s) for b in A_BETAS for s in SEEDS_A}
    | {key_a(b, 0) for b in COLLAPSE_BETAS}
    | {key_b(c, s) for c in B_CONDITIONS for s in SEEDS_B}
    | {key_d(n, s) for n in ("near", "far") for s in SEEDS_D}
)
assert len(RECORDS) == _expected_runs, (len(RECORDS), _expected_runs)
assert all(v.shape == (len(VALID_INDEX), NUM_DRAWS) for v in STOCHASTIC_LABELS.values())
assert DETERMINISTIC_LABELS.shape == (len(VALID_INDEX), NUM_DRAWS)
assert all(len(v["pair_indices"]) == NUM_MAIN_PAIRS for k, v in _datasets.items() if k[0] == "main")
_state_after = reference_policy.state_dict()
assert all(torch.equal(_state_after[k].cpu(), REFERENCE_STATE[k]) for k in REFERENCE_STATE), (
    "参照方策の重みが変わった"
)
print(
    f"不変条件: 全 {len(RECORDS)} 学習の history の長さ {TRAIN_STEPS}・キー {sorted(_keys)}・評価のステップ {HALF_STEP}, {TRAIN_STEPS}、"
    "同じシード・同じ大きさのミニバッチの順序が共通、評価のプロンプトとサンプル数が共通、参照方策の重みが不変: OK"
)
_effective = {
    "level": CURRENT_LEVEL_NAME,
    "seeds": {"A": len(SEEDS_A), "B": len(SEEDS_B), "D": len(SEEDS_D)},
    "a_levels": len(A_BETAS),
    "train_steps": TRAIN_STEPS,
    "num_draws": NUM_DRAWS,
    "main_pairs": NUM_MAIN_PAIRS,
    "candidate_pairs": NUM_CANDIDATE_PAIRS,
    "eval_inputs": NUM_EVAL_INPUTS,
}
_used = {
    "level": CURRENT_LEVEL_NAME,
    "seeds": {
        "A": len({s for s in SEEDS_A if all(key_a(b, s) in RECORDS for b in A_BETAS)}),
        "B": len({k[4] for k in RECORDS if k[0] == "ipo"}),
        "D": len({k[4] for k in RECORDS if k[2] == "near"}),
    },
    "a_levels": len({b for b in A_BETAS if all(key_a(b, s) in RECORDS for s in SEEDS_A)}),
    "train_steps": len(next(iter(RECORDS.values()))["history"]["loss"]),
    "num_draws": STOCHASTIC_LABELS[0].shape[1],
    "main_pairs": len(dataset_for("main", "stochastic", 0)["pair_indices"]),
    "candidate_pairs": len(CANDIDATES["first"]),
    "eval_inputs": len(next(iter(RECORDS.values()))["eval"]["true_reward_by_input"]),
}
assert _used == _effective, (_used, _effective)
assert (CURRENT_LEVEL_NAME == "smoke") == SMOKE_TEST
print(
    f"SMOKE_TEST={SMOKE_TEST} の配線: 実効水準 {json.dumps(_effective)} と実際に使われた値が一致: OK"
)
```

    不変条件: 全 57 学習の history の長さ 768・キー ['accuracy', 'chosen_log_ratio', 'evaluations', 'learning_rate', 'loss', 'mean_margin', 'rejected_log_ratio', 'step']・評価のステップ 384, 768、同じシード・同じ大きさのミニバッチの順序が共通、評価のプロンプトとサンプル数が共通、参照方策の重みが不変: OK
    SMOKE_TEST=False の配線: 実効水準 {"level": "prod", "seeds": {"A": 5, "B": 5, "D": 5}, "a_levels": 6, "train_steps": 768, "num_draws": 4, "main_pairs": 2048, "candidate_pairs": 8192, "eval_inputs": 128} と実際に使われた値が一致: OK


### 6.15 判定結果の一覧

前提条件が 1 つでも不成立の実験は、判定関数の結果に関わらず「前提不成立」とする(判定不能とは区別する)。判定関数の結果は
参考として別欄に残す。選ばれた段階と、判定に実際に使ったシード数・水準を印字する。


```python
def verdict_label(computed: str, preconditions: list[str]) -> str:
    return computed if all(precondition_status.get(p) for p in preconditions) else "前提不成立"


print(f"{_tag}段階の選択: {STAGE_SELECTION_MESSAGE}")
print(
    f"{_tag}判定に使った値: S_A = {len(SEEDS_A)}、S_B = {len(SEEDS_B)}、S_D = {len(SEEDS_D)}、beta の水準 "
    f"{[round(b, 6) for b in A_BETAS]}、T = {TRAIN_STEPS}、K = {NUM_DRAWS}、N = {NUM_MAIN_PAIRS}"
)
_rows = [
    (
        "A",
        f"g1 = {CONTRAST_A['g1']['value']:+.5f}(sigma {CONTRAST_A['g1']['sigma']:.5f})、g2 = {CONTRAST_A['g2']['value']:+.5f}(sigma {CONTRAST_A['g2']['sigma']:.5f})",
        ["学習の成立(A)", "PA"],
        verdict_A,
    ),
    (
        "B",
        f"D = {CONTRAST_B['value']:+.4f}(sigma {CONTRAST_B['sigma']:.4f})",
        ["学習の成立(B・C)", "PB"],
        verdict_B,
    ),
    (
        "C",
        f"Delta_w(T) = {CONTRAST_C['value']:+.4f}(sigma {CONTRAST_C['sigma']:.4f})",
        ["学習の成立(B・C)", "PC"],
        verdict_C,
    ),
    (
        "D",
        f"E = {CONTRAST_D['value']:+.4f}(sigma {CONTRAST_D['sigma']:.4f})",
        ["学習の成立(D)", "PD"],
        verdict_D,
    ),
]
print(f"{_tag}実験 | 対比量 | 前提条件 | 判定関数の結果 | 最終判定")
for _name, _value, _pre, _computed in _rows:
    print(
        f"{_tag}{_name} | {_value} | "
        + ", ".join(f"{p}={precondition_status.get(p)}" for p in _pre)
        + f" | {_computed} | {verdict_label(_computed, _pre)}"
    )
print(f"\nノートブック全体の実行時間: {(time.time() - NOTEBOOK_START_TIME) / 60:.1f} 分")
```

    段階の選択: 予算 120 分に収まる最小の段階として、段階 0 を選んだ(見積もり 74.6 分、cuda 基準)
    判定に使った値: S_A = 5、S_B = 5、S_D = 5、beta の水準 [3.2, 2.262742, 1.6, 1.131371, 0.8, 0.565685]、T = 768、K = 4、N = 2048
    実験 | 対比量 | 前提条件 | 判定関数の結果 | 最終判定
    A | g1 = +0.00474(sigma 0.00182)、g2 = -0.01482(sigma 0.00261) | 学習の成立(A)=True, PA=True | 支持 | 支持
    B | D = +1.2413(sigma 0.0830) | 学習の成立(B・C)=True, PB=True | 支持 | 支持
    C | Delta_w(T) = -0.0750(sigma 0.0255) | 学習の成立(B・C)=True, PC=True | 支持 | 支持
    D | E = -0.0292(sigma 0.0197) | 学習の成立(D)=True, PD=True | 判定不能 | 判定不能
    
    ノートブック全体の実行時間: 59.0 分


## 7. 結果・考察 / Results and Discussion

本番実行(Google Colab T4)のセル出力に基づいて記す。判定は 6.1 節で事前に宣言した基準のみから導く(7.3〜7.6 節)。結果を見た後に立てた解釈は 7.7 節に分けて記し、検証済みの結論としては扱わない。

### 7.1 実行の概要

- **実行環境**(5.1 節の印字): Tesla T4(compute capability 7.5、総メモリ 14.56 GiB)、Python 3.13.15、torch 2.13.0+cu130(ビルド時の CUDA 13.0、cuDNN 92000)、コミット`ffa9f7b`(未コミットの変更なし)、実行日時 2026-09-28T07:55:34(UTC)。
- **参照方策の照合**(5.5 節): T4 でも、016 の評価集合での完全一致率 0.8180(017: 0.8180)、形式の遵守率 1.0000、応答部分の負の対数尤度 0.60045(017: 0.6005、差 −0.00005)で、許容の範囲に入った。`model_state.pt`の SHA-256 も 017 の記録と一致した(5.3 節)。
- **決定的な実行**(5.8 節): CUDA で、サンプリング・参照方策の対数確率・DPO と IPO の学習・評価を 2 回実行して bit 単位で一致した。キャッシュの確認(6.6 節)も、選好の組の生成し直し・キャッシュなしの参照方策の対数確率・24 ステップの学習と評価のすべてで bit 単位で一致した。
- **段階と実行時間**(6.4 節): 段階 0 の見積もりは 74.6 分(予算の 62.2%)で、段階 0(全実験 5 シード)が選ばれた。ノートブック全体の実行時間は 59.0 分だった(6.15 節)。
- **選好の組**(6.5 節): 候補 8,192 組のうち、$y_1 = y_2$ の 0.0039 と、同一でない応答の同点の 0.0223 を除外し、7,977 組が残った(実験 A〜C は先頭 2,048 組)。$\kappa = 12.8373$($\lvert \Delta r^* \rvert$ の中央値 0.0856)。$K = 4$ での $\hat{p} \in (0, 1)$ の期待値は 0.5709 で、実際の割合はシードごとに 0.5630〜0.5845 だった(6.11 節)。実験 D の層は 16(課題 × $\lvert \Delta r^* \rvert$ の区間。区切り 0.0392・0.0856・0.1647)で、近い組・遠い組はそれぞれ 1,990 組だった。
- **較正**(6.7 節): 較正の格子の $\widehat{\mathrm{KL}}$(較正用のプロンプト)は次のとおりだった。

| $\beta$ | 3.2 | 2.26 | 1.6 | 1.13 | 0.8 | 0.566 | 0.4 | 0.283 | 0.2 | 0.141 | 0.1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $\widehat{\mathrm{KL}}$(nats) | 0.042 | 0.068 | 0.117 | 0.214 | 0.374 | 0.596 | 3.98 | 15.1 | 28.9 | 39.8 | 61.4 |
| 応答の平均トークン数 | 12.96 | 12.93 | 13.02 | 13.18 | 13.13 | 13.23 | 16.11 | 21.71 | 26.09 | 26.95 | 30.64 |
| 終端記号で止まった割合 | 0.987 | 0.989 | 0.983 | 0.981 | 0.982 | 0.974 | 0.836 | 0.565 | 0.327 | 0.265 | 0.089 |

  6.1 節の規則により、標準の $\beta$ は **0.565685**(大きい $\beta$ から見て $\widehat{\mathrm{KL}} \le 2$ nats が続く最小の格子点)、実験 A の水準は $3.2, 2.263, 1.6, 1.131, 0.8, 0.566$($\beta_6$ = 標準の $\beta$)、崩壊の領域の観察は $\beta = 0.283$・$0.141$ になった。標準の $\beta$ は実験 A の水準に含まれるので、実験 B の条件 2 と実験 C は実験 A の $\beta = 0.566$ の学習を共有した。
- **パイロットとの違い**: 6.2 節のパイロット(MPS)では $\beta = 0.4$ の $\widehat{\mathrm{KL}}$ が 1.71 nats だったが、T4 の較正では 3.98 nats になり、6.2 節で予想した 0.4 または 0.283 ではなく、0.566 が標準の $\beta$ に選ばれた。この違いの原因は特定していない。
- **旧規則との比較**: 公比 2 の旧規則($\widehat{\mathrm{KL}} \ge 0.1$ nats となる最大の $\beta$ を $\beta_1$ とする)を上の表に当てはめると $\beta_1 = 1.6$ で、水準は $1.6, 0.8, 0.4, 0.2, 0.1, 0.05$ になる。後半の $0.2$・$0.1$ の $\widehat{\mathrm{KL}}$ は 28.9・61.4 nats であり、後半の水準は崩壊の領域に入っていた。
- **較正の診断量**: $\beta$ を下げると、終端記号で止まった割合は 0.99 付近(3.2〜0.8)から 0.089($\beta = 0.1$)まで下がり、応答の平均トークン数は約 13 から 30.64 に伸びた。$\widehat{\mathrm{KL}}$ の急増($\beta = 0.566$ の 0.596 から $0.4$ の 3.98)は、終端記号で止まる割合の低下と応答の長さの増加を伴っている。

### 7.2 前提条件

すべての前提条件が成立した。

| 前提条件 | 値 | 基準 | 成否 |
|---|---|---|---|
| 学習の成立(実験 A) | $a(T)$ の最小値 0.7411(30 学習) | 0.55 以上 | 成立 |
| 学習の成立(実験 B・C) | $a(T)$ の最小値 0.7411(4 条件 × 5 シード) | 0.55 以上 | 成立 |
| 学習の成立(実験 D) | $a(T)$ の最小値 0.7343(2 データセット × 5 シード) | 0.55 以上 | 成立 |
| PA(KL の範囲) | $\overline{\widehat{\mathrm{KL}}}(\beta_6) = 0.8077$、$\overline{\widehat{\mathrm{KL}}}(\beta_1) = 0.0435$(比 18.6) | 比 4 以上 | 成立 |
| PB(重複の効果) | $\hat{p} \in (0, 1)$ の組の割合 0.5630〜0.5845(5 シード) | 全シードで 0.4 以上 | 成立 |
| PC(マージンの拡大) | $\bar{h}(T) = 1.3696$〜$1.4721$(5 シード) | 全シードで正 | 成立 |
| PD(マージンの拡大) | 近い組 $\bar{h}(T) = 0.820$〜$0.941$、遠い組 $1.900$〜$2.100$(5 シード) | 全シードで正 | 成立 |

### 7.3 実験 A: 直接選好最適化の過最適化

**判定: 支持。** 前半の傾き $g_1 = +0.00474$($\sigma_{g_1} = 0.00182$、$2\sigma_{g_1} = 0.00364$)は閾値を超えて正、後半の傾き $g_2 = -0.01482$($\sigma_{g_2} = 0.00261$、$2\sigma_{g_2} = 0.00521$)は閾値を超えて負だった。シードごとの傾きも、$g_1$ は 5 シードとも正(+0.00177〜+0.00863)、$g_2$ は 5 シードとも負(−0.02262〜−0.00999)だった。

**効果量は小さい。** 真の報酬の平均(シード平均)は、$\beta_1$ から順に 0.7112・0.7125・0.7160・0.7141・0.7083・0.6993 で、6 水準を通して 0.6993〜0.7160 の範囲でしか動かなかった。最大は $\beta = 1.6$ の 0.7160 で、$\beta_1 = 3.2$ との差は約 0.005 である。後半の低下は最大から $\beta_6$ まで約 0.017 だった。図(6.9 節)でも、どのシードの曲線も前半で上がるか横ばいになり、後半で下がる。真の報酬 対 $\widehat{\mathrm{KL}}$ の図では、真の報酬は $\widehat{\mathrm{KL}}$ が約 0.11〜0.19 nats($\beta = 1.6$・$1.131$)のところで最大になり、その後 $\widehat{\mathrm{KL}}$ が 0.81 nats まで増えるにつれて下がった。

**診断量**(シード平均): 評価用の組での、暗黙の報酬の符号と真の報酬の順序の一致率は 0.5389〜0.5544 で、どの水準でも 0.54 前後にとどまった。一方、評価用の組での真の報酬の順序に向けたマージンは、$\beta$ を下げるとともに 0.054 から 0.366 へ大きくなった。応答の平均トークン数は 13.076 から 13.519 へわずかに増え、完全一致の割合は 0.0187〜0.0215 だった。

**限界**:

- 参照方策そのものの真の報酬を、同じ評価条件(評価用のプロンプト・temperature 1 のサンプリング)では測っていない。$\beta_1$ の方策($\widehat{\mathrm{KL}} = 0.0435$ nats)は参照方策の近似にとどまるので、**DPO が参照方策(SFT)を改善したかは、本トピックの出力からは言えない。**
- 評価は temperature 1 のサンプリングで行っており、完全一致の割合(約 0.02)は、貪欲法で測った参照方策の照合値(0.818、5.5 節)とは比べられない。

**崩壊の領域の観察**(6.10 節、判定なし、シード 0 のみ):

| $\beta$ | 真の報酬の平均 | $\widehat{\mathrm{KL}}$(nats) | 応答の平均トークン数 | 終端記号で止まった割合 | 完全一致の割合 | $\bar{\Delta}_w(T)$ |
|---|---|---|---|---|---|---|
| 0.566($\beta_6$、参考) | 0.7013 | 0.692 | 13.46 | 0.9648 | 0.0178 | −0.050 |
| 0.283 | 0.4925 | 14.304 | 21.33 | 0.582 | 0.0103 | −0.987 |
| 0.141 | 0.2846 | 40.311 | 26.94 | 0.2715 | 0.0037 | −4.875 |

$\beta_6$ から格子を 2 点下るだけで、$\widehat{\mathrm{KL}}$ は 0.692 から 14.3 nats へ、真の報酬は 0.70 から 0.49 へ変わった。崩壊は、終端記号を出さずに生成し続けること(終端記号で止まった割合の低下と、応答の長さの増加)として現れ、$\widehat{\mathrm{KL}}$ の急増は応答の長さの増加を伴っている。学習用の事例での正解率 $a(T)$ は 0.7419・0.6775 で、学習自体は進んでいた(6.8 節)。

**017 との定性的な比較**: 017 の best-of-n で真の報酬が最大になったのは $n = 4$〜8 で、その KL の上界 $\log n - (n - 1)/n$ は約 0.64〜1.20 nats だった。018 で真の報酬が最大になった $\widehat{\mathrm{KL}}$ は約 0.11〜0.19 nats である。ただし、best-of-n の値は KL の **上界** であり(017 の 3.6 節)、018 の値はサンプリングによる **モンテカルロ推定** なので、推定量が異なり、数値を直接比べることはできない。017 の PPO では、KL 係数が小さい条件で代理報酬(報酬モデルのスコア)が上がる一方で真の報酬が下がった。018 では、評価用の組での暗黙の報酬のマージンが $\beta$ を下げるにつれて上がり続ける一方で、後半では真の報酬が下がった。代理の指標が上がり続けても真の報酬が下がるという形は共通している。

### 7.4 実験 B: 決定的なラベルでの過適合

**判定: 支持。** 4 条件の学習用の組でのマージンの平均(シード平均)は次のとおりだった。

| 条件 | 損失・ラベル | $\bar{h}(T/2)$ | $\bar{h}(T)$ | 伸び $g$(シードごと) |
|---|---|---|---|---|
| 1 | DPO・決定的 | 2.5221 | 4.2924 | +1.5719〜+1.8496 |
| 2 | DPO・確率的 | 0.9797 | 1.4107 | +0.2812〜+0.5144 |
| 3 | IPO・決定的 | 0.4210 | 0.5890 | +0.1290〜+0.1897 |
| 4 | IPO・確率的 | 0.1815 | 0.2513 | +0.0571〜+0.0833 |

差の差 $D = (g_1 - g_2) - (g_3 - g_4) = +1.2413$($\sigma_D = 0.0830$、$2\sigma_D = 0.1659$)は閾値を大きく超えて正で、5 シードとも正(+0.9973〜+1.4661)だった。DPO では決定的なラベルと確率的なラベルの伸びの差が約 1.3 nats あるのに対し、IPO では約 0.1 nats である。マージンの推移の図(6.11 節)でも、決定的なラベルの DPO のバッチのマージンは学習の最後まで上がり続け、ステップ 768 で約 4 に達した。

- **KL ダイバージェンス**: 決定的なラベルの DPO の $\widehat{\mathrm{KL}}$ はシード平均で 17.28 nats(シードごとに 9.5〜24.1)に達した。標準の $\beta$ のままでも、崩壊の領域の観察($\beta = 0.283$ で 14.3 nats)と同程度の水準である。IPO の $\widehat{\mathrm{KL}}$ は 0.073(決定的)・0.032(確率的)にとどまった。
- **IPO の目標値との比較**: IPO の目標のマージンは $1/(2\tau) = 1/(2 \times 0.566) \approx 0.884$ nats である。決定的なラベルの IPO の $\bar{h}(T) = 0.5890$ は目標に届いておらず、後半にも +0.17 伸びている。したがって「有界な値に収束した」とは言えず、言えるのは **DPO と比べて伸びが小さい** ことまでである。
- **DPO の確率的なラベルの条件も後半に伸びている**(+0.43)。3.5 節の理論からは、$K = 4$ でも $\hat{p} \in \{0, 1\}$ の組(約 43%)では最適なマージンが発散するので、伸び続けることが予想される。ただしこれは理論からの読みであり、$\hat{p} \in (0, 1)$ の組とそれ以外の組に分けた検証はしていない。

### 7.5 実験 C: 尤度の置き換わりの存在

**判定: 支持。** 学習用の事例での選好された応答の対数比の平均は $\bar{\Delta}_w(T) = -0.0750$ nats($\sigma = 0.0255$、$2\sigma = 0.0511$)で、閾値を超えて負だった。5 シードとも負(−0.1626〜−0.0382)で、前提条件のマージンの平均は 1.3696〜1.4721 と正に広がっていた。マージンが広がる一方で、選好された応答の対数確率は参照方策より下がった。

**効果量は小さい。** 選好されなかった応答の対数比 −1.4856 と比べて、選好された応答の低下は約 1/20 で、トークンあたりでは −0.00596 nats である。マージンの約 1.41 nats の大部分は、選好されなかった応答の確率を下げることで作られている。対数比の推移の図(6.12 節)でも、選好された応答の対数比は学習を通して 0 からわずかに下(約 −0.2 まで)の範囲にとどまり、選好されなかった応答の対数比が約 −1.4 まで下がった。

**結論の範囲**: この支持は、標準の $\beta = 0.566$・確率的なラベル・本トピックの学習の設定での結論である。

### 7.6 実験 D: 尤度の置き換わりの類似度依存性

**判定: 判定不能。** $E = \bar{\Delta}_w^{\mathrm{near}} - \bar{\Delta}_w^{\mathrm{far}} = -0.0292$($\sigma_E = 0.0197$、$2\sigma_E = 0.0394$)で、閾値に届かなかった。シードごとの差は +0.0070・−0.0324・−0.0616・−0.0274・−0.0318 で、5 シード中 4 シードで負だった。

**診断量**:

- 課題 × $\lvert \Delta r^* \rvert$ の区間で層別して件数を揃えたにもかかわらず、$\lvert \Delta r^* \rvert$ の平均は近い組 0.113、遠い組 0.1594 と異なった。区間の中での分布は揃っていない。
- マージンの平均は近い組 0.8893、遠い組 1.9804 で、応答のトークン数の分位点(10%・50%・90%)も近い組 8・11・16、遠い組 9・13・22 と異なった。正規化編集距離の平均は 0.3887・0.6872 だった。
- **CHES スコア**(参照方策で計算): 生の値は遠い組のほうが大きい(1492.32 対 −19.43)が、長さで正規化した値は近い組のほうが大きく(−2.509 対 −8.636)、編集距離と同じ順序(近い組のほうが似ている)になった。生の CHES スコアはトークンについての隠れ状態の和から作るので、応答の長さに強く依存する。

### 7.7 事後的な解釈(検証済みの結論ではない)

**以下はすべて、結果を見た後に立てた解釈であり、検証していない。** 事前に宣言した判定(7.3〜7.6 節)とは区別して読むこと。

1. **実験 A の後半の低下と崩壊の連続性**: $\beta$ を $\beta_1$ から $\beta_6$ へ下げるにつれて、応答の平均トークン数は 13.076 から 13.519 へ(シード平均、6.9 節の診断量)、終端記号で止まった割合は 0.9858 から 0.9648 へ(シード 0、6.10 節の表)変わった。変化の大部分は後半($\beta_4$ から $\beta_6$)で起きており、トークン数は 13.102 → 13.519、終端記号で止まった割合は 0.9846 → 0.9648 である。前半の割合は 0.9812〜0.9858 でほぼ横ばいである。崩壊の観察での割合(0.58・0.27)と連続しているように見える。このことから、実験 A の範囲での真の報酬の低下は、終端の失敗の初期段階を反映している可能性がある。これは、有限の選好データへの過適合による過最適化(3.7 節)とは別の機構かもしれず、両者は本トピックの実験では区別できていない。
2. **尤度の置き換わりと KL ダイバージェンスの対応**: 実験 A の出力(6.8 節)では、$\bar{\Delta}_w(T)$ が $\beta \ge 0.8$ では 5 シードとも正、$\beta = 0.566$ では 5 シードとも負になった。崩壊の観察では −0.987・−4.875、決定的なラベルの DPO ではシード平均で約 −1.42(シードごとに −0.67〜−1.90)である。方策が参照方策から離れるほど置き換わりが大きいように見え、置き換えられた確率質量の行き先の候補として、終端記号を出さない長い応答が考えられる。Razin et al. [4] が述べた、意図しない出力への確率質量の移動に対応するかもしれない。
3. **実験 D の交絡**: 近い組は $\lvert \Delta r^* \rvert$ が小さく、ラベルの雑音が大きいので、達成されるマージンが小さく、マージンを広げる圧力そのものが弱かった可能性がある。その場合、$\bar{\Delta}_w$ の差は、類似度の効果とマージンの差の効果の両方を含む。分位点の区間での層別は、区間の中の偏りを残しうる。マッチング(最近傍の組どうしの組み合わせ)や、交絡する変数を回帰で除く設計が、次の検証の候補である。また、近い組の学習はマージンが小さいのに $\widehat{\mathrm{KL}}$ が大きかった(6.8 節のシードごとの出力の平均で、近い組 0.90、遠い組 0.46)。
4. **パイロットと本番の KL の違い**: 崩壊の手前では、$\widehat{\mathrm{KL}}$ が $\beta$ に対して急峻に変化する(較正の表で、$\beta = 0.566$ の 0.596 から $0.4$ の 3.98)。そのため、デバイスの違いによる小さな数値差でも、較正で選ばれる格子点が変わりうる。

### 7.8 まとめ

| 実験 | 対比量 | 判定 |
|---|---|---|
| A: 直接選好最適化の過最適化 | $g_1 = +0.00474$($\sigma$ 0.00182)、$g_2 = -0.01482$($\sigma$ 0.00261) | 支持(効果量は小さい) |
| B: 決定的なラベルでの過適合 | $D = +1.2413$($\sigma$ 0.0830) | 支持 |
| C: 尤度の置き換わりの存在 | $\bar{\Delta}_w(T) = -0.0750$($\sigma$ 0.0255) | 支持(効果量は小さい) |
| D: 尤度の置き換わりの類似度依存性 | $E = -0.0292$($\sigma$ 0.0197) | 判定不能 |

- **本トピックの設定から言えること**:
  - 決定的なラベルに対する DPO と IPO の振る舞いの違い(実験 B)は、明確に観測された。決定的なラベルの DPO はマージンが学習の後半も伸び続け、$\widehat{\mathrm{KL}}$ は約 17 nats に達した。
  - 過最適化(実験 A)と尤度の置き換わり(実験 C)は、事前の基準を満たした。ただし効果量は小さい(真の報酬の変化は 0.02 以下、選好された応答の対数確率の低下は 0.075 nats)。
  - 標準の $\beta$ は、KL ダイバージェンスのみの較正で 0.566 に決まった。それより小さい $\beta$ では、終端記号を出さずに生成し続ける崩壊が観察された(判定なし)。
- **言えないこと**: DPO が参照方策(SFT)の真の報酬を改善したか(同じ評価条件で参照方策を測っていない)。尤度の置き換わりの類似度依存性(実験 D は判定不能で、類似度と $\lvert \Delta r^* \rvert$・マージンの交絡が残る)。


## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/04_alignment/018_direct_preference_optimization.ipynb
