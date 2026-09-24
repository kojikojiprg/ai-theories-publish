---
title: "量子化の基礎(Quantization Basics)(実装・実験編 3/5)"
---

この記事は後編(実装・実験編 3/5)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/013_quantization_basics-practice-2)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/013_quantization_basics-practice-4)。

### 6.5 実験 B: 粒度の効果と外れ値(尖度)の関係


```python
experiment_b = {}
for _name in TARGET_NAMES:
    _w = TARGET_WEIGHTS[_name]
    _err_tensor = reconstruction_error(
        _w, quantize_weight(_w, method="absmax", bits=4, granularity="tensor")
    )
    _err_block = reconstruction_error(
        _w, quantize_weight(_w, method="absmax", bits=4, granularity="block", block_size=BLOCK_SIZE)
    )
    experiment_b[_name] = {
        "shape": list(TARGET_SHAPES[_name]),
        "err_tensor": _err_tensor,
        "err_block64": _err_block,
        "log_rho": math.log(_err_tensor / _err_block),
        "excess_kurtosis": compute_excess_kurtosis(_w),
        "max_abs_over_std": float(_w.abs().max() / _w.std()),
    }

_groups = [tuple(experiment_b[n]["shape"]) for n in TARGET_NAMES]
_group_sizes = {g: _groups.count(g) for g in dict.fromkeys(_groups)}
precondition_status["P-B"] = bool(
    len(_group_sizes) >= PRECONDITION_B_MIN_GROUPS
    and min(_group_sizes.values()) >= PRECONDITION_B_MIN_GROUP_SIZE
)
print(
    f"前提条件 P-B: 形状の群 {_group_sizes} -> {'成立' if precondition_status['P-B'] else '不成立'}"
)

_log_rho = np.array([experiment_b[n]["log_rho"] for n in TARGET_NAMES])
_kurtosis = np.array([experiment_b[n]["excess_kurtosis"] for n in TARGET_NAMES])
stratified_spearman_B = compute_stratified_spearman_correlation(_log_rho, _kurtosis, _groups)
_boot_b = bootstrap_stratified_spearman_correlation(
    _log_rho, _kurtosis, _groups, BOOTSTRAP_RESAMPLES, seed=0
)
assert len(_boot_b) == BOOTSTRAP_RESAMPLES
interval_B = tuple(float(x) for x in np.percentile(_boot_b, [2.5, 97.5]))
verdict_B = "支持" if interval_B[0] > 0 else "反証" if interval_B[1] < 0 else "判定不能"
for _g in _group_sizes:
    _mask = [g == _g for g in _groups]
    _rho_g = compute_stratified_spearman_correlation(
        _log_rho[_mask], _kurtosis[_mask], [_g] * sum(_mask)
    )
    print(f"{_smoke_tag}群 {_g}(n={sum(_mask)}): Spearman 相関 = {_rho_g:+.4f}")
print(
    f"{_smoke_tag}層別 Spearman 相関(群の大きさで重み付け)= {stratified_spearman_B:+.4f}、"
    f"ブートストラップ 95% 区間({BOOTSTRAP_RESAMPLES:,} 回)= [{interval_B[0]:+.4f}, {interval_B[1]:+.4f}]"
)
print(f"{_smoke_tag}実験 B の判定関数の結果: {verdict_B}")
_max_over_std = np.array([experiment_b[n]["max_abs_over_std"] for n in TARGET_NAMES])
for _g in _group_sizes:
    _mask = np.array([g == _g for g in _groups])
    print(
        f"{_smoke_tag}診断量: 群 {_g}: log(rho) の最小 / 中央値 / 最大 = "
        f"{_log_rho[_mask].min():.4f} / {np.median(_log_rho[_mask]):.4f} / {_log_rho[_mask].max():.4f}、"
        f"最大絶対値 / 標準偏差 = {_max_over_std[_mask].min():.2f} / "
        f"{np.median(_max_over_std[_mask]):.2f} / {_max_over_std[_mask].max():.2f}"
    )

# 診断量: ブロックサイズの掃引と実効ビット数
sweep_B = {}
for _block in (*BLOCK_SWEEP_B, "tensor"):
    _errs = []
    for _name in TARGET_NAMES:
        _w = TARGET_WEIGHTS[_name]
        _kw = (
            dict(granularity="tensor")
            if _block == "tensor"
            else dict(granularity="block", block_size=_block)
        )
        _errs.append(reconstruction_error(_w, quantize_weight(_w, method="absmax", bits=4, **_kw)))
    _num_scales = sum(
        1 if _block == "tensor" else -(-w.numel() // _block) for w in TARGET_WEIGHTS.values()
    )
    sweep_B[str(_block)] = {
        "mean_log_error": float(np.mean(np.log(_errs))),
        "effective_bits": 4 + 32 * _num_scales / TARGET_NUM_ELEMENTS,
    }
print(f"\n{_smoke_tag}診断量: ブロックサイズの掃引(INT4 absmax、28 行列)")
for _block, _row in sweep_B.items():
    print(
        f"  block={_block:>6}: 平均 log(誤差) = {_row['mean_log_error']:.4f}、実効ビット数 = {_row['effective_bits']:.4f}"
    )

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for _g in _group_sizes:
    _mask = np.array([g == _g for g in _groups])
    axes[0].scatter(_kurtosis[_mask], _log_rho[_mask], label=f"shape {_g}")
axes[0].set_xlabel("excess kurtosis")
axes[0].set_ylabel("log(err_tensor / err_block64)")
axes[0].set_title("Experiment B: gain from block-wise scales vs kurtosis")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)
_eff = [r["effective_bits"] for r in sweep_B.values()]
_err = [r["mean_log_error"] for r in sweep_B.values()]
axes[1].plot(_eff, _err, marker="o")
for _block, _x, _y in zip(sweep_B, _eff, _err, strict=True):
    axes[1].annotate(_block, (_x, _y), fontsize=7, textcoords="offset points", xytext=(3, 3))
axes[1].set_xscale("log")
axes[1].set_xlabel("effective bits per parameter (log scale)")
axes[1].set_ylabel("mean log reconstruction error")
axes[1].set_title("Block-size sweep (INT4 absmax)")
axes[1].grid(alpha=0.3)
fig.tight_layout()
plt.show()
```

    前提条件 P-B: 形状の群 {(256, 256): 16, (683, 256): 8, (256, 683): 4} -> 成立
    群 (256, 256)(n=16): Spearman 相関 = +0.7559
    群 (683, 256)(n=8): Spearman 相関 = +0.2619
    群 (256, 683)(n=4): Spearman 相関 = -0.8000
    層別 Spearman 相関(群の大きさで重み付け)= +0.3925、ブートストラップ 95% 区間(10,000 回)= [+0.0735, +0.6878]
    実験 B の判定関数の結果: 支持
    診断量: 群 (256, 256): log(rho) の最小 / 中央値 / 最大 = 0.7216 / 0.9192 / 1.3988、最大絶対値 / 標準偏差 = 3.21 / 3.60 / 4.86
    診断量: 群 (683, 256): log(rho) の最小 / 中央値 / 最大 = 1.0154 / 1.2097 / 1.3023、最大絶対値 / 標準偏差 = 4.08 / 4.67 / 4.90
    診断量: 群 (256, 683): log(rho) の最小 / 中央値 / 最大 = 1.1772 / 1.2264 / 1.3099、最大絶対値 / 標準偏差 = 4.72 / 4.81 / 5.06
    
    診断量: ブロックサイズの掃引(INT4 absmax、28 行列)
      block=    16: 平均 log(誤差) = -10.7045、実効ビット数 = 6.0000
      block=    32: 平均 log(誤差) = -10.4733、実効ビット数 = 5.0000
      block=    64: 平均 log(誤差) = -10.2893、実効ビット数 = 4.5000
      block=   128: 平均 log(誤差) = -10.1366、実効ビット数 = 4.2500
      block=   256: 平均 log(誤差) = -10.0059、実効ビット数 = 4.1250
      block=   512: 平均 log(誤差) = -9.8811、実効ビット数 = 4.0626
      block=  1024: 平均 log(誤差) = -9.7670、実効ビット数 = 4.0313
      block=  2048: 平均 log(誤差) = -9.6632、実効ビット数 = 4.0157
      block=  4096: 平均 log(誤差) = -9.5707、実効ビット数 = 4.0079
      block=  8192: 平均 log(誤差) = -9.4845、実効ビット数 = 4.0040
      block= 16384: 平均 log(誤差) = -9.3841、実効ビット数 = 4.0020
      block= 32768: 平均 log(誤差) = -9.3001、実効ビット数 = 4.0011
      block= 65536: 平均 log(誤差) = -9.2317、実効ビット数 = 4.0005
      block=tensor: 平均 log(誤差) = -9.1831、実効ビット数 = 4.0003



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/013_quantization_basics/output_34_1.png)
    


### 6.6 実験 C: NF4 と INT4 の再構成誤差


```python
experiment_c = {}
for _name in TARGET_NAMES:
    _w = TARGET_WEIGHTS[_name]
    experiment_c[_name] = {
        "err_nf4": reconstruction_error(_w, QUANTIZED_CACHE["q5"][_name]),
        "err_int4": reconstruction_error(_w, QUANTIZED_CACHE["q4"][_name]),
        "err_nf4_double_quantized": reconstruction_error(_w, QUANTIZED_CACHE["q6"][_name]),
        "excess_kurtosis": experiment_b[_name]["excess_kurtosis"],
    }
_median_abs_kurtosis = float(np.median(np.abs(_kurtosis)))
precondition_status["P-C"] = bool(_median_abs_kurtosis <= PRECONDITION_C_MAX_MEDIAN_ABS_KURTOSIS)
print(
    f"前提条件 P-C: |尖度| の中央値 = {_median_abs_kurtosis:.4f}(閾値 {PRECONDITION_C_MAX_MEDIAN_ABS_KURTOSIS})"
    f" -> {'成立' if precondition_status['P-C'] else '不成立'}"
)
_ell = np.array([math.log(v["err_nf4"] / v["err_int4"]) for v in experiment_c.values()])
ell_mean_C, ell_se_C = mean_and_standard_error(_ell)
verdict_C = (
    "支持"
    if ell_mean_C + 2 * ell_se_C < 0
    else "反証"
    if ell_mean_C - 2 * ell_se_C > 0
    else "判定不能"
)
print(
    f"{_smoke_tag}l_bar = mean log(err_NF4 / err_INT4) = {ell_mean_C:+.5f}、SE = {ell_se_C:.5f}、"
    f"[l_bar - 2SE, l_bar + 2SE] = [{ell_mean_C - 2 * ell_se_C:+.5f}, {ell_mean_C + 2 * ell_se_C:+.5f}]"
)
print(f"{_smoke_tag}実験 C の判定関数の結果: {verdict_C}")
_ELL_BAR_SMOKE = -0.18560  # 第 1 段階のスモークテストの出力から転記(小数第 5 位まで)
print(
    f"診断量: l_bar - スモークテストの転記値 = {ell_mean_C - _ELL_BAR_SMOKE:+.2e}"
    f"(転記値 {_ELL_BAR_SMOKE:+.5f}、丸めの範囲は 5e-6)"
)
_dq = np.array(
    [math.log(v["err_nf4_double_quantized"] / v["err_nf4"]) for v in experiment_c.values()]
)
print(
    f"{_smoke_tag}診断量: mean log(err_q6 / err_q5)(二重量子化による誤差の変化)= {_dq.mean():+.6f}"
)
print(
    f"{_smoke_tag}診断量: 下流の bits-per-byte の差 b_q5 - b_q4 = "
    f"{wiki_bits_per_byte['q5'] - wiki_bits_per_byte['q4']:+.5f}、b_q6 - b_q5 = "
    f"{wiki_bits_per_byte['q6'] - wiki_bits_per_byte['q5']:+.5f}"
)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(_ell, bins=14)
ax.axvline(0, color="black", linewidth=1)
ax.axvline(ell_mean_C, color="tab:red", linestyle="--", label=f"mean = {ell_mean_C:+.3f}")
ax.set_xlabel("log(err_NF4 / err_INT4) per matrix")
ax.set_ylabel("count")
ax.set_title("Experiment C: NF4 vs INT4 (block 64)")
ax.legend(fontsize=8)
plt.show()
```

    前提条件 P-C: |尖度| の中央値 = 0.3154(閾値 1.0) -> 成立
    l_bar = mean log(err_NF4 / err_INT4) = -0.18560、SE = 0.02105、[l_bar - 2SE, l_bar + 2SE] = [-0.22770, -0.14351]
    実験 C の判定関数の結果: 支持
    診断量: l_bar - スモークテストの転記値 = -3.69e-06(転記値 -0.18560、丸めの範囲は 5e-6)
    診断量: mean log(err_q6 / err_q5)(二重量子化による誤差の変化)= +0.000234
    診断量: 下流の bits-per-byte の差 b_q5 - b_q4 = -0.00050、b_q6 - b_q5 = -0.00000



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/013_quantization_basics/output_36_1.png)
    


### 6.7 実験 D: QLoRA による量子化劣化の縮小

微調整前の基盤モデル(条件 1: FP32、条件 2: NF4 + 二重量子化)の評価集合での bits-per-byte と、評価窓の
対応付きブートストラップによる $\sigma_{\mathrm{before}}$ を求めてから、両条件を判定用のシード数で学習する。


```python
_nll_before = {
    cond: per_window_nll(build_quantized_model(cid), evaluation_windows, evaluation_mask)
    for cond, cid in FINETUNE_BASE.items()
}
b_before = {
    cond: evaluate_bits_per_byte(
        build_quantized_model(cid), evaluation_windows, evaluation_mask, evaluation_bytes, device
    )
    for cond, cid in FINETUNE_BASE.items()
}
for _cond in FINETUNE_BASE:
    assert math.isclose(
        bits_per_byte_from_nll(_nll_before[_cond], evaluation_bytes), b_before[_cond], rel_tol=1e-6
    )
DELTA_BEFORE = b_before["nf4"] - b_before["fp32"]
_boot_before = paired_bootstrap_ratio_of_sums(
    np.stack([_nll_before["fp32"], _nll_before["nf4"]]) / LN2,
    evaluation_window_bytes,
    BOOTSTRAP_RESAMPLES,
    seed=1,
)
SIGMA_BEFORE = float((_boot_before[:, 1] - _boot_before[:, 0]).std(ddof=1))
precondition_status["P-D1"] = bool(DELTA_BEFORE > 2 * SIGMA_BEFORE)
print(
    f"{_smoke_tag}微調整前: b(FP32 の基盤)= {b_before['fp32']:.6f}、b(NF4 の基盤)= {b_before['nf4']:.6f}、"
    f"Delta_before = {DELTA_BEFORE:+.6f}、sigma_before = {SIGMA_BEFORE:.6f}"
)
print(
    f"前提条件 P-D1(Delta_before > 2 sigma_before = {2 * SIGMA_BEFORE:.6f}): "
    f"{'成立' if precondition_status['P-D1'] else '不成立'}"
)

finetune_records: dict[str, list[dict]] = {"fp32": [], "nf4": []}
_t0_main = time.time()
for _cond in ("fp32", "nf4"):
    for _seed in SEEDS:
        _t0 = time.time()
        _rec = run_finetune(_cond, _seed)
        finetune_records[_cond].append(_rec)
        print(
            f"[{_cond}] seed={_seed}: b={_rec['final_bits_per_byte']:.4f}, 訓練損失 "
            f"{_rec['initial_train_loss']:.4f} -> {_rec['final_train_loss']:.4f}, "
            f"clipping 発動率={_rec['clip_trigger_ratio']:.2f}({time.time() - _t0:.1f}s)"
        )
print(f"実験 D の学習(2 条件 x {NUM_SEEDS} シード)の合計時間: {time.time() - _t0_main:.1f}s")
```

    微調整前: b(FP32 の基盤)= 3.392751、b(NF4 の基盤)= 3.400214、Delta_before = +0.007463、sigma_before = 0.001035
    前提条件 P-D1(Delta_before > 2 sigma_before = 0.002069): 成立
    [fp32] seed=0: b=2.6077, 訓練損失 6.2466 -> 4.6590, clipping 発動率=0.10(14.3s)
    [fp32] seed=1: b=2.6179, 訓練損失 6.2466 -> 4.6642, clipping 発動率=0.15(14.4s)
    [fp32] seed=2: b=2.6007, 訓練損失 6.2466 -> 4.6437, clipping 発動率=0.10(14.4s)
    [fp32] seed=3: b=2.6163, 訓練損失 6.2466 -> 4.6788, clipping 発動率=0.20(14.3s)
    [fp32] seed=4: b=2.6138, 訓練損失 6.2466 -> 4.6821, clipping 発動率=0.20(14.3s)
    [nf4] seed=0: b=2.6150, 訓練損失 6.2553 -> 4.6745, clipping 発動率=0.14(14.4s)
    [nf4] seed=1: b=2.6176, 訓練損失 6.2553 -> 4.6706, clipping 発動率=0.14(14.4s)
    [nf4] seed=2: b=2.6164, 訓練損失 6.2553 -> 4.6668, clipping 発動率=0.11(14.4s)
    [nf4] seed=3: b=2.6126, 訓練損失 6.2553 -> 4.6777, clipping 発動率=0.18(14.5s)
    [nf4] seed=4: b=2.6225, 訓練損失 6.2553 -> 4.6909, clipping 発動率=0.21(14.4s)
    実験 D の学習(2 条件 x 5 シード)の合計時間: 143.8s


**実験 D の不変条件**: 両条件の全シードで、評価窓(ハッシュ)・評価の分母・ステップ数・学習履歴の長さ・
LoRA の置換先・学習可能パラメータ数が同一であること、エポック上限を満たすこと、`SMOKE_TEST`の水準どおりの
シードで実行されたこと、基盤モデル自体が変化していないことを確認する。


```python
_all_d = finetune_records["fp32"] + finetune_records["nf4"]
_expected_lora = 2 * NUM_LAYERS * compute_lora_parameter_count(D_MODEL, D_MODEL, LORA_RANK)
for _r in _all_d:
    assert _r["evaluation_windows_hash"] == EVALUATION_WINDOWS_HASH
    assert _r["evaluation_denominator_bytes"] == EXPECTED_EVALUATION_BYTES
    assert _r["num_steps"] == NUM_STEPS and _r["history_length"] == NUM_STEPS
    assert _r["eval_step"][-1] == NUM_STEPS
    assert _r["num_trainable_parameters"] == _expected_lora
    assert _r["replaced"] == finetune_records["fp32"][0]["replaced"]
    assert _r["num_steps"] * TOKENS_PER_STEP <= EPOCHS * len(train_ids)
for _cond, _rs in finetune_records.items():
    assert [r["seed"] for r in _rs] == list(SEEDS), "シードが実効水準と一致しない"
    assert all(r["initial_train_loss"] == INITIAL_TRAIN_LOSS[_cond] for r in _rs)
assert hash_state(base_model.state_dict()) == BASE_STATE_HASH
print(
    f"実験 D の不変条件(評価窓・分母 {EXPECTED_EVALUATION_BYTES:,} バイト・ステップ数 {NUM_STEPS}・"
    f"LoRA のパラメータ数 {_expected_lora:,}・エポック上限・シード {SEEDS}・基盤モデルの不変): OK"
)

_b1 = np.array([r["final_bits_per_byte"] for r in finetune_records["fp32"]])
_b2 = np.array([r["final_bits_per_byte"] for r in finetune_records["nf4"]])
_n = len(SEEDS)
assert [r["seed"] for r in finetune_records["fp32"]] == [r["seed"] for r in finetune_records["nf4"]]
_d = _b2 - _b1  # シードごとの差 d_s(同じシードどうしの対応)
DELTA_AFTER = float(_d.mean())
assert math.isclose(DELTA_AFTER, float(_b2.mean() - _b1.mean()), rel_tol=1e-9, abs_tol=1e-12)
C_D = DELTA_BEFORE - DELTA_AFTER

# 新基準: sigma_c = sqrt(s_d^2 / n + sigma_win_c^2)
S_D = float(_d.std(ddof=1))
_mean_nll_after = {
    cond: np.mean([r["evaluation_window_nll"] for r in finetune_records[cond]], axis=0)
    for cond in ("fp32", "nf4")
}
_four_numerators = (
    np.stack(
        [_nll_before["fp32"], _nll_before["nf4"], _mean_nll_after["fp32"], _mean_nll_after["nf4"]]
    )
    / LN2
)
# 全標本で計算した c^* は c と一致する(シード平均した窓ごとの負の対数尤度の比 = シード平均の bits-per-byte)
_c_full = (
    (_four_numerators[1].sum() - _four_numerators[0].sum())
    - (_four_numerators[3].sum() - _four_numerators[2].sum())
) / evaluation_window_bytes.sum()
# c は evaluate_bits_per_byte(バッチごとの FP32 の和)の値から、_c_full は窓ごとの和から作るので、
# 4 つの bits-per-byte それぞれの相対誤差 1e-6(約 3.5e-6)の和を許容誤差とする。
assert math.isclose(_c_full, C_D, rel_tol=0, abs_tol=1.5e-5), (_c_full, C_D)
_boot_c = paired_bootstrap_ratio_of_sums(
    _four_numerators, evaluation_window_bytes, BOOTSTRAP_RESAMPLES, seed=2
)
assert _boot_c.shape == (BOOTSTRAP_RESAMPLES, 4)
_c_star = (_boot_c[:, 1] - _boot_c[:, 0]) - (_boot_c[:, 3] - _boot_c[:, 2])
SIGMA_WIN_C = float(_c_star.std(ddof=1))
SIGMA_C_D = math.sqrt(S_D**2 / _n + SIGMA_WIN_C**2)
verdict_D = "支持" if C_D > 2 * SIGMA_C_D else "反証" if C_D < -2 * SIGMA_C_D else "判定不能"

# 旧基準(参考、判定には使わない): sigma_c = sqrt(sigma_1^2 / n + sigma_2^2 / n + sigma_before^2)
SIGMA_1, SIGMA_2 = float(_b1.std(ddof=1)), float(_b2.std(ddof=1))
SIGMA_C_D_OLD = math.sqrt(SIGMA_1**2 / _n + SIGMA_2**2 / _n + SIGMA_BEFORE**2)
verdict_D_old = (
    "支持" if C_D > 2 * SIGMA_C_D_OLD else "反証" if C_D < -2 * SIGMA_C_D_OLD else "判定不能"
)

precondition_status["P-D2"] = all(
    r["final_train_loss"] <= (1 - PRECONDITION_TRAIN_LOSS_DROP) * r["initial_train_loss"]
    for r in _all_d
)
print(
    f"{_smoke_tag}b1(FP32 + LoRA)= {_b1.mean():.6f}(sigma_1 = {SIGMA_1:.6f})、"
    f"b2(NF4 + LoRA)= {_b2.mean():.6f}(sigma_2 = {SIGMA_2:.6f})、n = {_n}"
)
print(f"{_smoke_tag}シードごとの差 d_s = b2_s - b1_s: " + ", ".join(f"{x:+.6f}" for x in _d))
print(
    f"{_smoke_tag}Delta_before = {DELTA_BEFORE:+.6f}、Delta_after = mean(d_s) = {DELTA_AFTER:+.6f}、"
    f"c = {C_D:+.6f}"
)
print(
    f"{_smoke_tag}新基準: s_d = {S_D:.6f}、sigma_win_c = {SIGMA_WIN_C:.6f}"
    f"(ブートストラップ {BOOTSTRAP_RESAMPLES:,} 回)、sigma_c = {SIGMA_C_D:.6f}、"
    f"2 sigma_c = {2 * SIGMA_C_D:.6f} -> 判定関数の結果: {verdict_D}"
)
print(
    f"{_smoke_tag}旧基準(参考、判定には使わない): sigma_c = {SIGMA_C_D_OLD:.6f}、"
    f"2 sigma_c = {2 * SIGMA_C_D_OLD:.6f} -> 判定関数の結果: {verdict_D_old}"
)
print(
    "前提条件 P-D2(全シードで最終訓練損失 <= 0.95 x 学習開始時): "
    f"{'成立' if precondition_status['P-D2'] else '不成立'}"
)

# 診断量
_recovery = 1 - DELTA_AFTER / DELTA_BEFORE if DELTA_BEFORE != 0 else float("nan")
print(f"{_smoke_tag}診断量: 回復率 1 - Delta_after / Delta_before = {_recovery:.4f}")
print(
    f"{_smoke_tag}診断量: 基盤の重み(量子化対象の 28 行列)の格納バイト数 FP32 {storage_table['q0']:,} / "
    f"NF4 + 二重量子化 {storage_table['q6']:,}(比 {storage_table['q6'] / storage_table['q0']:.4f})"
)
if device.type == "cuda":
    for _cond, _rs in finetune_records.items():
        _peaks = [r["peak_gpu_memory_bytes"] / 2**20 for r in _rs]
        print(
            f"{_smoke_tag}診断量: 学習中の最大 GPU メモリ [{_cond}]: "
            + ", ".join(f"{p:.1f} MB" for p in _peaks)
        )
else:
    print(f"CUDA 環境ではない({device})ため、学習中の最大 GPU メモリは記録していない。")
# 012 のセル出力(6.7 節の判定の一次情報)から転記した、LoRA(r = 8)のシード 0〜4 の値と FP32 の基盤の b0
_B2_012 = [
    2.607666127234919,
    2.617892988033689,
    2.600680119165922,
    2.616290870893274,
    2.6137570138035047,
]
_B0_012 = 3.3927508811831832
print(
    f"{_smoke_tag}診断量: FP32 の基盤の b = {b_before['fp32']:.10f}(012 の b0 = {_B0_012:.10f}、差 {b_before['fp32'] - _B0_012:+.2e})"
)
for _r in finetune_records["fp32"]:
    print(
        f"{_smoke_tag}診断量: 条件 1 seed={_r['seed']}: b = {_r['final_bits_per_byte']:.10f}"
        f"(012 の記録値 {_B2_012[_r['seed']]:.10f}、差 {_r['final_bits_per_byte'] - _B2_012[_r['seed']]:+.2e})"
    )

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for _cond, _color, _label in (
    ("fp32", "tab:blue", "FP32 base + LoRA"),
    ("nf4", "tab:orange", "NF4 (DQ) base + LoRA"),
):
    for _i, _r in enumerate(finetune_records[_cond]):
        axes[0].plot(
            [0] + _r["eval_step"],
            [b_before[_cond]] + _r["eval_bits_per_byte"],
            color=_color,
            alpha=0.6,
            label=_label if _i == 0 else None,
        )
axes[0].set_xlabel("step")
axes[0].set_ylabel("evaluation bits-per-byte")
axes[0].set_title("Experiment D: fine-tuning curves (step 0 = base model)")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)
axes[1].scatter(np.zeros(_n), _b1, label="condition 1 (FP32)")
axes[1].scatter(np.ones(_n), _b2, label="condition 2 (NF4)")
axes[1].set_xticks([0, 1], ["FP32 + LoRA", "NF4 + LoRA"])
axes[1].set_ylabel("final evaluation bits-per-byte")
axes[1].set_title("Final bits-per-byte per seed")
axes[1].grid(alpha=0.3)
fig.tight_layout()
plt.show()
```

    実験 D の不変条件(評価窓・分母 55,769 バイト・ステップ数 87・LoRA のパラメータ数 32,768・エポック上限・シード (0, 1, 2, 3, 4)・基盤モデルの不変): OK
    b1(FP32 + LoRA)= 2.611257(sigma_1 = 0.007078)、b2(NF4 + LoRA)= 2.616811(sigma_2 = 0.003656)、n = 5
    シードごとの差 d_s = b2_s - b1_s: +0.007357, -0.000326, +0.015692, -0.003666, +0.008710
    Delta_before = +0.007463、Delta_after = mean(d_s) = +0.005553、c = +0.001910
    新基準: s_d = 0.007674、sigma_win_c = 0.001541(ブートストラップ 10,000 回)、sigma_c = 0.003762、2 sigma_c = 0.007524 -> 判定関数の結果: 判定不能
    旧基準(参考、判定には使わない): sigma_c = 0.003710、2 sigma_c = 0.007420 -> 判定関数の結果: 判定不能
    前提条件 P-D2(全シードで最終訓練損失 <= 0.95 x 学習開始時): 成立
    診断量: 回復率 1 - Delta_after / Delta_before = 0.2559
    診断量: 基盤の重み(量子化対象の 28 行列)の格納バイト数 FP32 12,587,008 / NF4 + 二重量子化 1,623,440(比 0.1290)
    診断量: 学習中の最大 GPU メモリ [fp32]: 1886.3 MB, 1886.3 MB, 1886.3 MB, 1886.3 MB, 1886.3 MB
    診断量: 学習中の最大 GPU メモリ [nf4]: 1887.1 MB, 1887.1 MB, 1887.1 MB, 1887.1 MB, 1887.1 MB
    診断量: FP32 の基盤の b = 3.3927508812(012 の b0 = 3.3927508812、差 +0.00e+00)
    診断量: 条件 1 seed=0: b = 2.6076661272(012 の記録値 2.6076661272、差 +0.00e+00)
    診断量: 条件 1 seed=1: b = 2.6178929880(012 の記録値 2.6178929880、差 +0.00e+00)
    診断量: 条件 1 seed=2: b = 2.6006801192(012 の記録値 2.6006801192、差 +0.00e+00)
    診断量: 条件 1 seed=3: b = 2.6162908709(012 の記録値 2.6162908709、差 +0.00e+00)
    診断量: 条件 1 seed=4: b = 2.6137570138(012 の記録値 2.6137570138、差 +0.00e+00)



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/013_quantization_basics/output_40_1.png)
    


### 6.8 観察: LLM.int8() の外れ値の特徴次元(判定基準を設けない)

FP32 の基盤モデルの各層で、量子化対象の行列への入力を forward pre-hook で記録する。同じ入力を受け取る行列
(Query・Key・Value の射影、SwiGLU の 2 つの入力側の行列)は 1 回だけ記録する。


```python
_hooked = {}  # 隠れ状態の名前 -> 記録するモジュールの名前
for _layer in range(NUM_LAYERS):
    _hooked[f"L{_layer}.attention_input"] = f"{_layer}.self_attn.w_q"
    _hooked[f"L{_layer}.attention_output_projection_input"] = f"{_layer}.self_attn.w_o"
    _hooked[f"L{_layer}.feed_forward_input"] = f"{_layer}.feed_forward.w"
    _hooked[f"L{_layer}.feed_forward_output_projection_input"] = f"{_layer}.feed_forward.w2"
_captured: dict[str, list[torch.Tensor]] = {k: [] for k in _hooked}
_handles = [
    base_model.blocks.get_submodule(module_name).register_forward_pre_hook(
        lambda _m, inputs, key=key: _captured[key].append(inputs[0].detach().float().cpu())
    )
    for key, module_name in _hooked.items()
]
_obs_windows, _obs_mask = wiki_windows[:OBSERVATION_WINDOWS], wiki_mask[:OBSERVATION_WINDOWS]
try:
    with torch.no_grad():
        for _s in range(0, OBSERVATION_WINDOWS, 16):
            base_model(_obs_windows[_s : _s + 16].to(device))
finally:
    for _h in _handles:
        _h.remove()
_token_mask = _obs_mask.reshape(-1)
hidden_states = {
    k: torch.cat(v).reshape(-1, v[0].size(-1))[_token_mask] for k, v in _captured.items()
}
assert all(h.size(0) == int(_token_mask.sum()) for h in hidden_states.values())

observation = {}
for _key, _h in hidden_states.items():
    _max_abs = _h.abs().amax(dim=0)
    observation[_key] = {
        "num_features": _h.size(1),
        "max_abs": float(_max_abs.max()),
        "num_features_at_least_6": int((_max_abs >= OUTLIER_MAGNITUDE).sum()),
        "top5_feature_max_abs": [float(x) for x in _max_abs.topk(5).values],
    }
    print(
        f"{_key:<42} 特徴次元 {_h.size(1)}: 最大絶対値 {observation[_key]['max_abs']:.2f}、"
        f"大きさ 6 以上の値を持つ特徴次元 {observation[_key]['num_features_at_least_6']}"
    )

# 原論文の基準: d_model 次元の隠れ状態(注意機構の入力・SwiGLU の入力、各層 2 つ)について
_criterion_keys = [
    k for k in hidden_states if k.endswith(("attention_input", "feed_forward_input"))
]
_stack = torch.stack(
    [hidden_states[k].abs() >= OUTLIER_MAGNITUDE for k in _criterion_keys]
)  # (H, T, d)
_layer_fraction = (
    _stack.any(dim=1).float().mean(dim=0)
)  # 特徴次元ごと: その次元が 6 以上になる隠れ状態の割合
_position_fraction = (
    _stack.any(dim=0).float().mean(dim=0)
)  # 特徴次元ごと: いずれかの隠れ状態で 6 以上の位置の割合
_outlier_features = torch.nonzero(
    (_layer_fraction >= OUTLIER_LAYER_FRACTION) & (_position_fraction >= OUTLIER_POSITION_FRACTION)
).flatten()
observation["criterion"] = {
    "hidden_states": _criterion_keys,
    "num_tokens": int(_token_mask.sum()),
    "outlier_features": _outlier_features.tolist(),
    "layer_fraction_of_outlier_features": [float(_layer_fraction[i]) for i in _outlier_features],
    "position_fraction_of_outlier_features": [
        float(_position_fraction[i]) for i in _outlier_features
    ],
    "max_layer_fraction": float(_layer_fraction.max()),
    "max_position_fraction": float(_position_fraction.max()),
}
print(
    f"\n原論文の基準(大きさ >= {OUTLIER_MAGNITUDE}、隠れ状態の >= {OUTLIER_LAYER_FRACTION:.0%}、"
    f"系列位置の >= {OUTLIER_POSITION_FRACTION:.0%})を満たす特徴次元: {observation['criterion']['outlier_features']}"
    f"(隠れ状態の割合の最大 {observation['criterion']['max_layer_fraction']:.3f}、"
    f"系列位置の割合の最大 {observation['criterion']['max_position_fraction']:.4f})"
)

fig, ax = plt.subplots(figsize=(13, 3.8))
_matrix = torch.stack([hidden_states[k].abs().amax(dim=0) for k in _criterion_keys]).numpy()
_im = ax.imshow(_matrix, aspect="auto", cmap="viridis")
ax.set_yticks(range(len(_criterion_keys)), _criterion_keys, fontsize=7)
ax.set_xlabel("feature dimension")
ax.set_title(
    "Max |activation| per feature dimension (inputs to quantized matrices, FP32 base model)"
)
fig.colorbar(_im, ax=ax)
plt.show()
```

    L0.attention_input                         特徴次元 256: 最大絶対値 4.58、大きさ 6 以上の値を持つ特徴次元 0
    L0.attention_output_projection_input       特徴次元 256: 最大絶対値 6.20、大きさ 6 以上の値を持つ特徴次元 2
    L0.feed_forward_input                      特徴次元 256: 最大絶対値 4.07、大きさ 6 以上の値を持つ特徴次元 0
    L0.feed_forward_output_projection_input    特徴次元 683: 最大絶対値 16.64、大きさ 6 以上の値を持つ特徴次元 219
    L1.attention_input                         特徴次元 256: 最大絶対値 5.13、大きさ 6 以上の値を持つ特徴次元 0
    L1.attention_output_projection_input       特徴次元 256: 最大絶対値 4.18、大きさ 6 以上の値を持つ特徴次元 0
    L1.feed_forward_input                      特徴次元 256: 最大絶対値 4.75、大きさ 6 以上の値を持つ特徴次元 0
    L1.feed_forward_output_projection_input    特徴次元 683: 最大絶対値 22.93、大きさ 6 以上の値を持つ特徴次元 497
    L2.attention_input                         特徴次元 256: 最大絶対値 3.90、大きさ 6 以上の値を持つ特徴次元 0
    L2.attention_output_projection_input       特徴次元 256: 最大絶対値 4.54、大きさ 6 以上の値を持つ特徴次元 0
    L2.feed_forward_input                      特徴次元 256: 最大絶対値 4.91、大きさ 6 以上の値を持つ特徴次元 0
    L2.feed_forward_output_projection_input    特徴次元 683: 最大絶対値 23.89、大きさ 6 以上の値を持つ特徴次元 623
    L3.attention_input                         特徴次元 256: 最大絶対値 3.96、大きさ 6 以上の値を持つ特徴次元 0
    L3.attention_output_projection_input       特徴次元 256: 最大絶対値 5.11、大きさ 6 以上の値を持つ特徴次元 0
    L3.feed_forward_input                      特徴次元 256: 最大絶対値 5.18、大きさ 6 以上の値を持つ特徴次元 0
    L3.feed_forward_output_projection_input    特徴次元 683: 最大絶対値 28.88、大きさ 6 以上の値を持つ特徴次元 658
    
    原論文の基準(大きさ >= 6.0、隠れ状態の >= 25%、系列位置の >= 6%)を満たす特徴次元: [](隠れ状態の割合の最大 0.000、系列位置の割合の最大 0.0000)



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/013_quantization_basics/output_42_1.png)
    




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/03_efficient_training/013_quantization_basics.ipynb
