---
title: "混合精度学習(Mixed Precision Training)(実装・実験編 3/4)"
---

この記事は後編(実装・実験編 3/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/011_mixed_precision_training-practice-2)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/011_mixed_precision_training-practice-4)。

### 6.8.1 前提条件の評価(P0〜P3、非致命的に記録する)

前提条件が成立しなかった場合でも`assert`で停止せず、成否を`precondition_status`
(5.1 節で初期化済み)に記録し、6.15 節の判定表に反映する(前提不成立の実験は
判定不能と区別して扱う)。P0 は 5.8 節で既に記録済み。P1 は条件1の全シードの
最終訓練損失で評価する。P4(実験 F に固有の前提条件)は 6.12 節で、実験 F を
実際に実行した後にそこで記録する。

**P2 の評価対象の改訂(旧定義・新定義・理由)**: 前回までは、P2(条件2の活性化
勾配の厳密な 0 の比率が条件1の同じ量を上回ること)を較正セル(6.2 節、seed=0
の 1 回のみの実行)の結果で評価していた(旧定義)。これを、6.7 節で記録した
**本番の条件1・条件2の全シードの診断量**(判定に使う学習そのものから記録した
もの)のシード平均の比較に改める(新定義)。この改訂は、前提条件は判定する
実行そのもので成立していなければならないという一般論に基づくものであり、
観測結果の方向には依存しない(閾値`PRECONDITION_P2_MIN_MARGIN`の値は変えない)。



```python
# --- P1: 条件1の全シードの最終訓練損失が閾値以下であること(現在の水準、全シード) ---
_p1_final_losses = [h["train_loss"][-1] for h in condition_histories["1"]]
precondition_status["P1"] = all(v <= PRECONDITION_LOSS_THRESHOLD for v in _p1_final_losses)
print(f"P1: 条件1の最終訓練損失(シードごと)={['%.4f' % v for v in _p1_final_losses]}, "
      f"閾値={PRECONDITION_LOSS_THRESHOLD:.4f} -> {'成立' if precondition_status['P1'] else '不成立'}")

# --- P2(新定義): 本番の条件1・条件2の全シードの診断量(6.7 節)のシード平均を比較する ---
_p2_ratios_cond1 = [
    d["zero_count"] / d["total_count"] for d in sublayer_diagnostics["1"] if d["total_count"] > 0
]
_p2_ratios_cond2 = [
    d["zero_count"] / d["total_count"] for d in sublayer_diagnostics["2"] if d["total_count"] > 0
]
_p2_mean_cond1 = float(np.mean(_p2_ratios_cond1))
_p2_mean_cond2 = float(np.mean(_p2_ratios_cond2))
_p2_margin = _p2_mean_cond2 - _p2_mean_cond1
precondition_status["P2"] = _p2_margin >= PRECONDITION_P2_MIN_MARGIN
print(f"P2(新定義、条件1〜2の全シード): 条件1平均={_p2_mean_cond1:.6f}, 条件2平均={_p2_mean_cond2:.6f}, "
      f"マージン={_p2_margin:.6f}, 閾値={PRECONDITION_P2_MIN_MARGIN} -> "
      f"{'成立' if precondition_status['P2'] else '不成立'}")
print(f"P2 の較正実行(旧定義、6.2 節、参考値): マージン={_p2_margin_calibrated:.6f}")

# --- P3: 条件4の全シードで backoff・growth がそれぞれ1回以上起きていること ---
_p3_per_seed_ok = []
for _h4 in condition_histories["4"]:
    _scale_series = np.array(_h4["loss_scale"])
    _n_backoff = int(np.sum(np.diff(_scale_series) < 0))
    _n_growth = int(np.sum(np.diff(_scale_series) > 0))
    _p3_per_seed_ok.append(_n_backoff >= 1 and _n_growth >= 1)
    print(f"  条件4 seed: backoff={_n_backoff}, growth={_n_growth}")
precondition_status["P3"] = all(_p3_per_seed_ok)
print(f"P3: 全シードで backoff>=1 かつ growth>=1 -> "
      f"{'成立' if precondition_status['P3'] else '不成立'}"
      f"({sum(_p3_per_seed_ok)}/{len(_p3_per_seed_ok)} シードで成立)")

print()
print(f"前提条件の成否一覧(この時点、P4 は 6.12 節で追加される): {precondition_status}")

```

    P1: 条件1の最終訓練損失(シードごと)=['1.9452', '1.9166', '1.9116', '1.9276', '1.9037'], 閾値=2.5046 -> 成立
    P2(新定義、条件1〜2の全シード): 条件1平均=0.000000, 条件2平均=0.017641, マージン=0.017641, 閾値=0.005 -> 成立
    P2 の較正実行(旧定義、6.2 節、参考値): マージン=0.017184
      条件4 seed: backoff=2, growth=7
      条件4 seed: backoff=1, growth=7
      条件4 seed: backoff=2, growth=7
      条件4 seed: backoff=2, growth=7
      条件4 seed: backoff=2, growth=7
    P3: 全シードで backoff>=1 かつ growth>=1 -> 成立(5/5 シードで成立)
    
    前提条件の成否一覧(この時点、P4 は 6.12 節で追加される): {'P0': True, 'P1': True, 'P2': True, 'P3': True}


### 6.9 実験 C: 静的損失スケーリングの効果



```python
def summarize_final_bpb(histories: list[dict]) -> tuple[float, float, int]:
    # シードごとの最終 bits-per-byte から (平均, 標準誤差, 非有限値のシード数) を計算する。
    # 非有限値のシードは平均・標準誤差の計算から除外する(6.4節の事前宣言の通り)。
    values = [h["eval_bits_per_byte"][-1] for h in histories]
    finite_values = [v for v in values if np.isfinite(v)]
    n_nonfinite = len(values) - len(finite_values)
    if len(finite_values) < 2:
        return float("nan"), float("nan"), n_nonfinite
    arr = np.array(finite_values)
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr))), n_nonfinite


def judge_three_way(delta: float, threshold: float) -> str:
    if delta > threshold:
        return "支持"
    if delta < -threshold:
        return "反証"
    return "判定不能"


b1, se1, nf1 = summarize_final_bpb(condition_histories["1"])
b2, se2, nf2 = summarize_final_bpb(condition_histories["2"])
b3, se3, nf3 = summarize_final_bpb(condition_histories["3"])
b4, se4, nf4 = summarize_final_bpb(condition_histories["4"])
print(f"条件1: b1={b1:.4f}, se1={se1:.4f}, 非有限値シード数={nf1}")
print(f"条件2: b2={b2:.4f}, se2={se2:.4f}, 非有限値シード数={nf2}")
print(f"条件3: b3={b3:.4f}, se3={se3:.4f}, 非有限値シード数={nf3}")
print(f"条件4: b4={b4:.4f}, se4={se4:.4f}, 非有限値シード数={nf4}")

if nf2 > 0 or nf3 > 0:
    print(f"[前提条件確認] 条件2で非有限値のシードが {nf2} 件、条件3で {nf3} 件あり、"
          f"6.4節の事前宣言により実験Cの判定は判定不能として扱う")
    D_C, threshold_C, verdict_C = float("nan"), float("nan"), "判定不能(非有限値)"
else:
    D_C = b2 - b3
    threshold_C = 2 * math.sqrt(se2**2 + se3**2)
    verdict_C = judge_three_way(D_C, threshold_C)
print(f"{_smoke_tag}実験C: D_C={D_C}, 閾値={threshold_C}, 判定={verdict_C}")

```

    条件1: b1=2.8454, se1=0.0074, 非有限値シード数=0
    条件2: b2=2.8912, se2=0.0086, 非有限値シード数=0
    条件3: b3=2.8454, se3=0.0074, 非有限値シード数=0
    条件4: b4=2.8458, se4=0.0073, 非有限値シード数=0
    実験C: D_C=0.045814454298842566, 閾値=0.022744028219419263, 判定=支持


条件1〜4 の最終 bits-per-byte を、シードごとの散布図で示す(`plot_seed_scatter`)。
条件1(FP32)の平均を基準線として重ねる。



```python
_final_bpb_by_condition = {
    f"Condition {_c}": [h["eval_bits_per_byte"][-1] for h in condition_histories[_c]]
    for _c in ("1", "2", "3", "4")
}
_ax_bpb_scatter = plot_seed_scatter(
    _final_bpb_by_condition,
    title="Final bits-per-byte by condition",
    ylabel="bits-per-byte",
)
_ax_bpb_scatter.axhline(
    b1, color="tab:blue", linestyle="--", linewidth=1.2, label="Condition 1 (baseline mean)"
)
_ax_bpb_scatter.legend(fontsize=8)
plt.show()

```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/011_mixed_precision_training/output_53_0.png)
    


### 6.10 実験 D: 動的損失スケーリングの効果



```python
if nf2 > 0 or nf4 > 0:
    print(f"[前提条件確認] 条件2で非有限値のシードが {nf2} 件、条件4で {nf4} 件あり、"
          f"6.4節の事前宣言により実験Dの判定は判定不能として扱う")
    D_D, threshold_D, verdict_D = float("nan"), float("nan"), "判定不能(非有限値)"
else:
    D_D = b2 - b4
    threshold_D = 2 * math.sqrt(se2**2 + se4**2)
    verdict_D = judge_three_way(D_D, threshold_D)
print(f"{_smoke_tag}実験D: D_D={D_D}, 閾値={threshold_D}, 判定={verdict_D}")

# 前提条件 P3: 条件4の全シードで backoff・growth がそれぞれ1回以上起きていること。
for _seed_idx, _h4 in enumerate(condition_histories["4"]):
    _scale_series = np.array(_h4["loss_scale"])
    _n_backoff = int(np.sum(np.diff(_scale_series) < 0))
    _n_growth = int(np.sum(np.diff(_scale_series) > 0))
    _skip_ratio = float(np.mean(_h4["step_skipped"]))
    print(f"条件4 seed={_seed_idx}: backoff回数={_n_backoff}, growth回数={_n_growth}, "
          f"更新スキップ比率={_skip_ratio:.4f}, スケール値の範囲=[{_scale_series.min():.1f}, "
          f"{_scale_series.max():.1f}]")

```

    実験D: D_D=0.04544379323877168, 閾値=0.02259154642509119, 判定=支持
    条件4 seed=0: backoff回数=2, growth回数=7, 更新スキップ比率=0.0067, スケール値の範囲=[65536.0, 4194304.0]
    条件4 seed=1: backoff回数=1, growth回数=7, 更新スキップ比率=0.0033, スケール値の範囲=[65536.0, 4194304.0]
    条件4 seed=2: backoff回数=2, growth回数=7, 更新スキップ比率=0.0067, スケール値の範囲=[65536.0, 4194304.0]
    条件4 seed=3: backoff回数=2, growth回数=7, 更新スキップ比率=0.0067, スケール値の範囲=[65536.0, 4194304.0]
    条件4 seed=4: backoff回数=2, growth回数=7, 更新スキップ比率=0.0067, スケール値の範囲=[65536.0, 4194304.0]


条件4 の損失スケール値の時系列を、シードごとに重ね描きする
(`plot_learning_curves_multi_seed`)。更新をスキップしたステップは、
上に重ねた散布図の×印で示す。



```python
_ax_scale_timeline = plot_learning_curves_multi_seed(
    {"Condition 4": condition_histories["4"]}, step_key="step", value_key="loss_scale",
    title="Loss scale over training (condition 4)",
    xlabel="Step", ylabel="Loss scale", log_scale=True,
)
for _seed_idx, _h4 in enumerate(condition_histories["4"]):
    _steps_arr = np.array(_h4["step"])
    _scale_arr = np.array(_h4["loss_scale"])
    _skipped_arr = np.array(_h4["step_skipped"], dtype=bool)
    if _skipped_arr.any():
        _ax_scale_timeline.scatter(
            _steps_arr[_skipped_arr], _scale_arr[_skipped_arr],
            color="tab:red", marker="x", s=25, zorder=4,
            label="skipped update" if _seed_idx == 0 else None,
        )
_ax_scale_timeline.legend(fontsize=8)
plt.show()

```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/011_mixed_precision_training/output_57_0.png)
    


### 6.11 実験 E: FP32 マスター重みの必要性(合成タスク)



```python
def run_synthetic_master_weight_experiment(u_over_w0: float, num_steps: int) -> dict:
    # w0=1 に更新量 u を num_steps 回加算し、FP16直接更新(条件7)と
    # FP32マスター重み更新(条件8)を比較する。期待値は理論式から独立に計算する。
    w0 = 1.0
    u = u_over_w0 * w0

    # 条件7: FP16 で直接更新
    w_fp16 = torch.tensor([w0], dtype=torch.float16)
    u_fp16 = torch.tensor([u], dtype=torch.float16)
    for _ in range(num_steps):
        w_fp16 = w_fp16 + u_fp16
    cond7_final = w_fp16.item()

    # 条件8: FP32 マスター重みで更新(MasterWeightOptimizer)
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[w0]]))
    model = model.half()
    master_opt = MasterWeightOptimizer(model, functools.partial(torch.optim.SGD, lr=1.0))
    x = torch.tensor([[1.0]], dtype=torch.float16)
    for _ in range(num_steps):
        master_opt.zero_grad()
        y = model(x)
        y.sum().backward()
        for p in model.parameters():
            p.grad = torch.tensor([[-u]], dtype=torch.float16)  # 更新量が -lr*grad=u になるよう設定
        master_opt.sync_gradients_to_master()
        master_opt.step()
    # 条件8で比較する対象は FP32 マスター重み自身である(FP32マスター重みの主張は
    # 「正本(master)側で更新が正しく蓄積される」ことであり、model.weight は毎ステップ
    # FP16へキャストし直した「写し」に過ぎず、そこだけを見ると常にFP16粒度の量子化誤差が
    # 乗る。model.weight(FP16写し)も診断量として併記する)。
    cond8_final = master_opt.master_params[0].item()
    cond8_fp16_copy = model.weight.item()

    # 期待値は理論式から独立に計算する(実行結果からは作らない)。
    theoretical_expected = w0 + num_steps * u
    theory_update_lost = u_over_w0 <= 2.0**-11  # 3.5節: 加算方向の境界

    return {
        "u_over_w0": u_over_w0,
        "cond7_final": cond7_final,
        "cond8_final": cond8_final,
        "cond8_fp16_copy": cond8_fp16_copy,
        "theoretical_expected": theoretical_expected,
        "theory_update_lost": theory_update_lost,
    }


_u_levels = [2.0**-k for k in range(8, 15)]  # 2^-8 .. 2^-14
_results_e = [run_synthetic_master_weight_experiment(u, SYNTH_N) for u in _u_levels]
assert SYNTH_N == LEVELS[CURRENT_LEVEL_NAME]["SYNTH_N"]

# FP32 の累積丸め誤差の上界(単純化した見積もり: 各ステップの丸め誤差の上界 unit_roundoff_fp32
# x 現在値、の SYNTH_N 回の合計)。
_UNIT_ROUNDOFF_FP32 = torch.finfo(torch.float32).eps / 2
_cond8_error_bound = SYNTH_N * _UNIT_ROUNDOFF_FP32 * 2.0  # 値がおよそ [1,2) の範囲に留まるため

_all_match_theory = True
for r in _results_e:
    _observed_lost = abs(r["cond7_final"] - 1.0) < 1e-9
    _matches_cond7 = _observed_lost == r["theory_update_lost"]
    _cond8_error = abs(r["cond8_final"] - r["theoretical_expected"])
    _matches_cond8 = _cond8_error <= _cond8_error_bound
    _all_match_theory = _all_match_theory and _matches_cond7 and _matches_cond8
    print(f"u/w0=2^{{{math.log2(r['u_over_w0']):.0f}}}: cond7_final={r['cond7_final']:.10f} "
          f"(更新消失予測={r['theory_update_lost']}, 観測消失={_observed_lost}, 一致={_matches_cond7}), "
          f"cond8_final(FP32マスター)={r['cond8_final']:.10f} vs 理論値={r['theoretical_expected']:.10f} "
          f"(誤差={_cond8_error:.2e} <= 上界{_cond8_error_bound:.2e}: {_matches_cond8}), "
          f"cond8のFP16写し(診断量)={r['cond8_fp16_copy']:.10f}")

print(f"{_smoke_tag}実験E: 全水準で理論と一致={_all_match_theory}")

```

    u/w0=2^{-8}: cond7_final=1.3906250000 (更新消失予測=False, 観測消失=False, 一致=True), cond8_final(FP32マスター)=1.3906250000 vs 理論値=1.3906250000 (誤差=0.00e+00 <= 上界1.19e-05: True), cond8のFP16写し(診断量)=1.3906250000
    u/w0=2^{-9}: cond7_final=1.1953125000 (更新消失予測=False, 観測消失=False, 一致=True), cond8_final(FP32マスター)=1.1953125000 vs 理論値=1.1953125000 (誤差=0.00e+00 <= 上界1.19e-05: True), cond8のFP16写し(診断量)=1.1953125000
    u/w0=2^{-10}: cond7_final=1.0976562500 (更新消失予測=False, 観測消失=False, 一致=True), cond8_final(FP32マスター)=1.0976562500 vs 理論値=1.0976562500 (誤差=0.00e+00 <= 上界1.19e-05: True), cond8のFP16写し(診断量)=1.0976562500
    u/w0=2^{-11}: cond7_final=1.0000000000 (更新消失予測=True, 観測消失=True, 一致=True), cond8_final(FP32マスター)=1.0488281250 vs 理論値=1.0488281250 (誤差=0.00e+00 <= 上界1.19e-05: True), cond8のFP16写し(診断量)=1.0488281250
    u/w0=2^{-12}: cond7_final=1.0000000000 (更新消失予測=True, 観測消失=True, 一致=True), cond8_final(FP32マスター)=1.0244140625 vs 理論値=1.0244140625 (誤差=0.00e+00 <= 上界1.19e-05: True), cond8のFP16写し(診断量)=1.0244140625
    u/w0=2^{-13}: cond7_final=1.0000000000 (更新消失予測=True, 観測消失=True, 一致=True), cond8_final(FP32マスター)=1.0122070312 vs 理論値=1.0122070312 (誤差=0.00e+00 <= 上界1.19e-05: True), cond8のFP16写し(診断量)=1.0117187500
    u/w0=2^{-14}: cond7_final=1.0000000000 (更新消失予測=True, 観測消失=True, 一致=True), cond8_final(FP32マスター)=1.0061035156 vs 理論値=1.0061035156 (誤差=0.00e+00 <= 上界1.19e-05: True), cond8のFP16写し(診断量)=1.0058593750
    実験E: 全水準で理論と一致=True


### 6.12 実験 F: 損失スケーリングと gradient clipping の順序



```python
_flags5_seeds = []
_flags6_seeds = []
_unscaled_norm_ratios_seeds = []  # 診断量: unscale後の勾配ノルムの条件6/条件5の比
_t0 = time.time()
_sched_f = functools.partial(
    compute_warmup_cosine_learning_rate, warmup_steps=WARMUP_STEPS, total_steps=NUM_STEPS,
    peak_learning_rate=BASE_LEARNING_RATE, min_learning_rate=BASE_LEARNING_RATE * MIN_LEARNING_RATE_RATIO,
)
for _seed in range(NUM_SEEDS_MAIN):
    model5 = build_model(_seed).to(device)
    opt5 = AdamW(model5.parameters(), lr=BASE_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    dummy_w = torch.zeros(1, SEQUENCE_LENGTH, dtype=torch.long)
    dummy_m = torch.ones(1, SEQUENCE_LENGTH, dtype=torch.bool)
    h5 = train_language_model(
        model5, train_ids, dummy_w, dummy_m, 1, num_steps=K_EXPERIMENT_F,
        batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LENGTH, learning_rate=BASE_LEARNING_RATE,
        eval_interval=NUM_STEPS + 1, device=device, seed=_seed, optimizer=opt5,
        learning_rate_schedule=_sched_f, gradient_clip_threshold=GRADIENT_CLIP_THRESHOLD,
        autocast_dtype=torch.float16, loss_scaler=StaticLossScaler(STATIC_SCALE),
    )
    _flags5_seeds.append(np.array(h5["gradient_clip_triggered"], dtype=bool))
    _unscaled_norm5 = np.array(h5["gradient_norm"])

    model6 = build_model(_seed).to(device)
    opt6 = AdamW(model6.parameters(), lr=BASE_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler6 = StaticLossScaler(STATIC_SCALE)
    torch.manual_seed(_seed)
    gen6 = torch.Generator(device="cpu")
    gen6.manual_seed(_seed)
    triggered6 = []
    unscaled_norm6 = []
    for step in range(1, K_EXPERIMENT_F + 1):
        model6.train()
        opt6.set_learning_rate(_sched_f(step))
        inputs, targets = get_random_batch(train_ids, BATCH_SIZE, SEQUENCE_LENGTH, gen6)
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model6(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        scaled_loss = scaler6.scale_loss(loss)
        opt6.zero_grad()
        scaled_loss.backward()
        scaled_norm = float(sum(
            p.grad.detach().float().pow(2).sum() for p in model6.parameters() if p.grad is not None
        ) ** 0.5)
        clip_triggered = scaled_norm > GRADIENT_CLIP_THRESHOLD
        if clip_triggered:
            for p in model6.parameters():
                if p.grad is not None:
                    p.grad.detach().mul_(GRADIENT_CLIP_THRESHOLD / scaled_norm)
        found_inf = scaler6.unscale_gradients(model6.parameters())
        # 診断量: unscale 後(誤った順序でも、clip 適用後に unscale はする)の勾配ノルム。
        unscaled_norm6.append(float(sum(
            p.grad.detach().float().pow(2).sum() for p in model6.parameters() if p.grad is not None
        ) ** 0.5))
        if not found_inf:
            opt6.step()
        triggered6.append(clip_triggered)
    _flags6_seeds.append(np.array(triggered6, dtype=bool))
    _unscaled_norm_ratios_seeds.append(
        float(np.mean(unscaled_norm6)) / float(np.mean(_unscaled_norm5).clip(min=1e-12))
    )

assert len(_flags5_seeds) == LEVELS[CURRENT_LEVEL_NAME]["NUM_SEEDS_MAIN"]
assert len(_flags6_seeds) == LEVELS[CURRENT_LEVEL_NAME]["NUM_SEEDS_MAIN"]
assert all(len(f) == LEVELS[CURRENT_LEVEL_NAME]["K_EXPERIMENT_F"] for f in _flags5_seeds)
assert all(len(f) == LEVELS[CURRENT_LEVEL_NAME]["K_EXPERIMENT_F"] for f in _flags6_seeds)
_q5_per_seed = np.array([f.mean() for f in _flags5_seeds])
_q6_per_seed = np.array([f.mean() for f in _flags6_seeds])
print(f"elapsed={time.time() - _t0:.2f}s")
print(f"q5(シードごと)={_q5_per_seed}, q6(シードごと)={_q6_per_seed}")
print(f"診断量(unscale後の勾配ノルムの条件6/条件5比、シードごと): "
      f"{['%.4e' % v for v in _unscaled_norm_ratios_seeds]}")

D_F = float(_q6_per_seed.mean() - _q5_per_seed.mean())

# 新基準: 実験C・Dと同じ形(各条件のシード平均の標準誤差から誤差伝播で合成)。
if len(_q5_per_seed) >= 2:
    _se_q5 = float(_q5_per_seed.std(ddof=1) / np.sqrt(len(_q5_per_seed)))
    _se_q6 = float(_q6_per_seed.std(ddof=1) / np.sqrt(len(_q6_per_seed)))
    sigma_DF = float(np.sqrt(_se_q5**2 + _se_q6**2))
else:
    sigma_DF = float("nan")
verdict_F = judge_three_way(D_F, 2 * sigma_DF) if np.isfinite(sigma_DF) else "判定不能(シード数不足)"

# 旧基準(参考値): シードごとの q5・q6 の標本標準偏差から直接 sigma_DF を導出した場合。
if len(_q5_per_seed) >= 2:
    sigma_DF_old = float(np.sqrt(_q5_per_seed.std(ddof=1) ** 2 + _q6_per_seed.std(ddof=1) ** 2))
    verdict_F_old = judge_three_way(D_F, 2 * sigma_DF_old)
else:
    sigma_DF_old, verdict_F_old = float("nan"), "判定不能(シード数不足)"

_p4_ok = all(PRECONDITION_P4_MIN <= q <= PRECONDITION_P4_MAX for q in _q5_per_seed)
precondition_status["P4"] = bool(_p4_ok)  # 新定義: 実験Fの実際の実行(条件5の全シードのq5)で評価する
print(f"{_smoke_tag}実験F(新基準): D_F={D_F:.4f}, sigma_DF={sigma_DF:.4f}, "
      f"判定={verdict_F}, 前提条件P4成立={_p4_ok}")
print(f"[参考値、旧基準] sigma_DF_old={sigma_DF_old:.4f}, 判定={verdict_F_old}")
print(f"P4(新定義、実験Fの全シード): q5(シードごと)={_q5_per_seed}, "
      f"範囲=[{PRECONDITION_P4_MIN}, {PRECONDITION_P4_MAX}] -> "
      f"{'成立' if precondition_status['P4'] else '不成立'}")
print(f"P4 の較正実行(旧定義、6.2 節、参考値): q5={_q5_calibrated:.4f}")

```

    elapsed=48.23s
    q5(シードごと)=[0.31666667 0.28333333 0.31666667 0.3        0.3       ], q6(シードごと)=[1. 1. 1. 1. 1.]
    診断量(unscale後の勾配ノルムの条件6/条件5比、シードごと): ['1.2714e-05', '1.3778e-05', '1.2917e-05', '1.3988e-05', '1.3672e-05']
    実験F(新基準): D_F=0.6967, sigma_DF=0.0062, 判定=支持, 前提条件P4成立=True
    [参考値、旧基準] sigma_DF_old=0.0139, 判定=支持
    P4(新定義、実験Fの全シード): q5(シードごと)=[0.31666667 0.28333333 0.31666667 0.3        0.3       ], 範囲=[0.02, 0.98] -> 成立
    P4 の較正実行(旧定義、6.2 節、参考値): q5=0.3167


### 6.13 実験 G: 活性化メモリの削減

**本番水準でのピークメモリの事前見積もり**: 系列長 1024(3 水準のうち最大)・
バッチサイズ 32(本番)での FP32 の 1 ステップのピークメモリを、Attention の
スコア行列($B \times h \times T^2$、softmax の出力を含む)・QKV 射影・
Feed-Forward Network の中間活性化・正規化前の残差ストリームの複製を含めた
簡易な見積もりで概算し、T4 のメモリ量(15 GB)に収まるかを確認する。収まらない
場合は、実験 G の系列長掃引に限りバッチサイズのみを縮小する(実験 C・D の
条件は変えない。この判断は、見積もり自体が本番結果に依存しない理論計算である
ため、観測結果の方向に依存しない)。



```python
def estimate_activation_memory_bytes(
    d_model: int, num_layers: int, num_heads: int, d_ff: int,
    batch_size: int, seq_len: int, bytes_per_element: int,
) -> int:
    # 順伝播で保持し逆伝播に使う主要な活性化テンソルの簡易見積もり(層ごとの合計)。
    # Attention のスコア行列(B x h x T x T)が系列長について支配的な項になる。
    per_layer = (
        3 * batch_size * seq_len * d_model  # Q, K, V 射影
        + batch_size * num_heads * seq_len * seq_len  # attention scores / softmax
        + batch_size * seq_len * d_model  # attention 出力
        + 2 * batch_size * seq_len * d_ff  # SwiGLU の中間活性化(up + gate の2系統)
        + 2 * batch_size * seq_len * d_model  # 正規化前の残差ストリームの複製(pre-attn, pre-ffn)
    )
    return num_layers * per_layer * bytes_per_element


T4_MEMORY_BYTES = 15 * 1024**3  # Google Colab 無料枠の T4(約15GB)
_max_seq_len_g = PROD_SEQUENCE_LENGTH * 4
_estimated_bytes = estimate_activation_memory_bytes(
    PROD_D_MODEL, PROD_NUM_LAYERS, PROD_NUM_HEADS, PROD_D_FF,
    PROD_BATCH_SIZE, _max_seq_len_g, bytes_per_element=4,
)
print(f"本番水準(系列長={_max_seq_len_g}, バッチサイズ={PROD_BATCH_SIZE})でのFP32ピークメモリ見積もり: "
      f"{_estimated_bytes / 1024**3:.2f} GB(T4: {T4_MEMORY_BYTES / 1024**3:.0f} GB)")
EXPERIMENT_G_BATCH_SIZE = PROD_BATCH_SIZE
if _estimated_bytes > T4_MEMORY_BYTES * 0.7:  # 7割を安全マージンとする(パラメータ側・断片化も考慮)
    EXPERIMENT_G_BATCH_SIZE = max(1, PROD_BATCH_SIZE // 2)
    print(f"見積もりが T4 メモリの70%を超えるため、実験Gに限りバッチサイズを "
          f"{EXPERIMENT_G_BATCH_SIZE} に縮小する(実験C・Dの条件は変更しない)")
else:
    print(f"見積もりは T4 メモリの70%以内に収まるため、実験Gもバッチサイズ {EXPERIMENT_G_BATCH_SIZE} のまま実行する")

```

    本番水準(系列長=1024, バッチサイズ=32)でのFP32ピークメモリ見積もり: 5.75 GB(T4: 15 GB)
    見積もりは T4 メモリの70%以内に収まるため、実験Gもバッチサイズ 32 のまま実行する



```python
def measure_peak_memory_one_step(precision_level: str, seq_len: int, batch_size: int,
                                  seed: int = 0) -> float | None:
    # 1ステップ(順伝播+逆伝播)のピークメモリを測る。CUDA以外では None を返す
    # (MPS にはCUDAと同等のピークメモリ統計APIが無いため、コード経路の確認のみ行う)。
    torch.manual_seed(seed)
    d_k = D_MODEL // NUM_HEADS
    rope = RotaryPositionEmbedding(d_k, max_position=max(seq_len, SEQUENCE_LENGTH))
    model = GPTLanguageModel(
        VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, max(seq_len, SEQUENCE_LENGTH),
        positional_transform=rope, normalization_factory=RMSNorm,
        feed_forward_factory=functools.partial(SwiGLUFeedForwardNetwork, D_MODEL, SWIGLU_D_FF),
        tie_embeddings=True, norm_first=True,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=BASE_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    autocast_dtype = torch.float16 if precision_level == "fp16_dynamic" else None
    loss_scaler = DynamicLossScaler(**DYNAMIC_SCALER_KWARGS) if precision_level == "fp16_dynamic" else None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    inputs, targets = get_random_batch(train_ids, batch_size, seq_len, generator)
    inputs, targets = inputs.to(device), targets.to(device)

    if autocast_dtype is not None:
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    else:
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    scaled_loss = loss_scaler.scale_loss(loss) if loss_scaler is not None else loss
    optimizer.zero_grad()
    scaled_loss.backward()

    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else None

    # 次の計測に影響しないよう、このステップのモデル・optimizer・勾配を解放する。
    del model, optimizer, logits, loss, scaled_loss, inputs, targets
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return peak


_seq_lengths = [SEQUENCE_LENGTH, SEQUENCE_LENGTH * 2, SEQUENCE_LENGTH * 4]
print(f"実験Gの系列長水準(公比2の等比、5.3節のSEQUENCE_LENGTHの1・2・4倍): {_seq_lengths}")
print(f"実験Gのバッチサイズ: {EXPERIMENT_G_BATCH_SIZE}"
      f"{'(縮小済み)' if EXPERIMENT_G_BATCH_SIZE != PROD_BATCH_SIZE else '(本番バッチサイズのまま)'}")

_memory_results = {}
for _cond_name, _level in [("1", "fp32"), ("4", "fp16_dynamic")]:
    for _L in _seq_lengths:
        _mem = measure_peak_memory_one_step(_level, _L, EXPERIMENT_G_BATCH_SIZE)
        _memory_results[(_cond_name, _L)] = _mem
        _mem_str = f"{_mem:,} bytes" if _mem is not None else "N/A(MPS、コード経路のみ確認)"
        print(f"条件{_cond_name}, L={_L}: ピークメモリ={_mem_str}")

experiment_g_rho: float | None = None

if device.type == "cuda":
    experiment_g_rho = ((_memory_results[("4", _seq_lengths[2])] - _memory_results[("4", _seq_lengths[0])]) /
                         (_memory_results[("1", _seq_lengths[2])] - _memory_results[("1", _seq_lengths[0])]))
    print(f"{_smoke_tag}rho={experiment_g_rho:.4f}(閾値0.75で支持/反証を判定)")
else:
    print("MPS環境のためピークメモリは測定できない(コード経路の確認のみ完了、6.4節の記載の通り)")

```

    実験Gの系列長水準(公比2の等比、5.3節のSEQUENCE_LENGTHの1・2・4倍): [256, 512, 1024]
    実験Gのバッチサイズ: 32(本番バッチサイズのまま)
    条件1, L=256: ピークメモリ=2,222,404,096 bytes
    条件1, L=512: ピークメモリ=4,271,092,224 bytes
    条件1, L=1024: ピークメモリ=11,209,191,936 bytes
    条件4, L=256: ピークメモリ=2,069,788,160 bytes
    条件4, L=512: ピークメモリ=4,116,953,600 bytes
    条件4, L=1024: ピークメモリ=11,427,297,792 bytes
    rho=1.0413(閾値0.75で支持/反証を判定)


#### 実験 G: パラメータ側メモリ確認の測定方法の修正(第1回本番実行後)

**第1回本番実行(Google Colab T4)の値**: 理論値 = 48.3 MB に対し、実測値は
1079.3 MB(許容誤差20%以内 = False)であった。

**誤りの内容**: 修正前のコードは、パラメータ側メモリ(重み・勾配・optimizer
状態)を確認するモデルを構築した **直後** に`torch.cuda.memory_allocated()`
の絶対値を 1 回だけ測定していた。この絶対値には、本セルより前に実行済みの
セルに由来する確保量が混入する。実際に 6.12 節(実験 F)のコードを確認すると、
シードのループ内で構築した`model5`・`opt5`・`model6`・`opt6`のうち、最後の
シードの分がループ終了後も`del`されずにグローバル名前空間に残っていた。
このように、測定対象(パラメータ側メモリを確認するモデル 1 つ分)以外の確保量が
絶対値に混入し、理論値との比較が成立していなかった。

**修正内容**: モデルを構築する **直前** と、モデル構築・順伝播・逆伝播・
`AdamW.step()`を終えた **直後** の 2 時点で`torch.cuda.memory_allocated()`を
測定し、その **差分(delta)** をパラメータ側メモリの実測値とする。差分を取る
ことで、それ以前に何が確保されたまま残っていたかに依存せず、このセルが
新たに確保した量だけを測定できる。また、「直後」の測定前に、順伝播・逆伝播で
使った入力・ロジット・損失といった活性化側のテンソルを`del`と`gc.collect()`で
解放し、解放を試みたテンソルの内訳(名前・バイト数)を印字する(逆伝播に
`retain_graph=True`を渡していないため、計算グラフ自体は逆伝播の時点で
解放済みであり、解放を妨げる外部参照は存在しない)。許容誤差(20%)は変更しない。

**この修正は、観測結果の方向と無関係に特定できる測定の実装の誤りの修正である。**
判定基準(実験 G の $\rho$ に対する閾値 0.75)・前提条件の定義・閾値は変更しない。



```python
experiment_g_param_memory_ok: bool | None = None  # CUDA でのみ判定、MPS では経路確認(None)

if device.type == "cuda":
    # パラメータ側のメモリが3.8節の内訳(FP32学習: 重み4 + 勾配4 + AdamWのm・v 8 = 16バイト/パラメータ)
    # と一致することを確認する(許容誤差20%、optimizer実装の細部・メモリアロケータの
    # 断片化による差を許容するため事前に宣言する)。本体の学習と同じ AdamW
    # (src/training/optimizer.py)で測る。
    # 直前・直後の2時点の差分(delta)で測定する(修正の経緯は直前のMarkdownセルを参照)。
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _before_construct_bytes = torch.cuda.memory_allocated()

    _param_check_model = build_model(seed=0).to(device)
    _param_check_optimizer = AdamW(_param_check_model.parameters(), lr=BASE_LEARNING_RATE,
                                    weight_decay=WEIGHT_DECAY)
    _gen_check = torch.Generator(device="cpu")
    _gen_check.manual_seed(0)
    _inputs_check, _targets_check = get_random_batch(train_ids, BATCH_SIZE, SEQUENCE_LENGTH, _gen_check)
    _inputs_check, _targets_check = _inputs_check.to(device), _targets_check.to(device)
    _logits_check = _param_check_model(_inputs_check)
    _loss_check = F.cross_entropy(
        _logits_check.reshape(-1, _logits_check.size(-1)), _targets_check.reshape(-1)
    )
    _loss_check.backward()
    _param_check_optimizer.step()  # AdamW の m・v を確保させる

    # 「直後」の測定前に、活性化側のテンソル(パラメータ側メモリの対象外)を解放する。
    _activation_bytes_check = {
        "inputs": _inputs_check.element_size() * _inputs_check.nelement(),
        "targets": _targets_check.element_size() * _targets_check.nelement(),
        "logits": _logits_check.element_size() * _logits_check.nelement(),
        "loss": _loss_check.element_size() * _loss_check.nelement(),
    }
    del _inputs_check, _targets_check, _logits_check, _loss_check
    gc.collect()
    _after_construct_bytes = torch.cuda.memory_allocated()
    _observed_param_related_bytes = _after_construct_bytes - _before_construct_bytes

    _num_params_check = sum(p.numel() for p in _param_check_model.parameters())
    _expected_param_bytes = _num_params_check * 16  # 3.8節: 重み4 + 勾配4 + m・v 8
    _EXPERIMENT_G_PARAM_MEMORY_TOLERANCE = 0.20
    experiment_g_param_memory_ok = bool(
        abs(_observed_param_related_bytes - _expected_param_bytes) / _expected_param_bytes
        <= _EXPERIMENT_G_PARAM_MEMORY_TOLERANCE
    )
    print(f"パラメータ側メモリ(直前直後の差分): 理論値={_expected_param_bytes / 1024**2:.1f}MB, "
          f"実測={_observed_param_related_bytes / 1024**2:.1f}MB, "
          f"許容誤差{_EXPERIMENT_G_PARAM_MEMORY_TOLERANCE:.0%}以内={experiment_g_param_memory_ok}")
    print("解放を試みた活性化側テンソルの内訳(参考、理論値には含めない):")
    for _name, _nbytes in _activation_bytes_check.items():
        print(f"  {_name}: {_nbytes / 1024**2:.4f} MB")

    del _param_check_model, _param_check_optimizer
    gc.collect()
    torch.cuda.empty_cache()
else:
    print("MPS環境のためパラメータ側メモリは測定できない(コード経路の確認のみ完了)")

```

    パラメータ側メモリ(直前直後の差分): 理論値=48.3MB, 実測=48.4MB, 許容誤差20%以内=True
    解放を試みた活性化側テンソルの内訳(参考、理論値には含めない):
      inputs: 0.0625 MB
      targets: 0.0625 MB
      logits: 2.0312 MB
      loss: 0.0000 MB


#### 実験 G: 事後的な診断(逆伝播のために保存されるテンソルの dtype 内訳)

**この節は事後的な診断であり、判定基準(実験 G の $\rho$ に対する閾値 0.75)を
変更するものではない。** 3.8 節の理論(Attention のスコア行列・確率行列が
系列長について支配的な項であること)を、実際に`torch.autograd`が逆伝播のために
保存するテンソルから直接確認する診断量を追加する。

`torch.autograd.graph.saved_tensors_hooks`を使うと、順伝播 1 回で PyTorch が
逆伝播用に保存すると判断したテンソルを、`pack_hook`の発火という形で捕捉できる
(CUDA を必要としないため、ローカル(MPS)でも実測値が得られる)。診断量の
記録自体は順伝播の時点で完了しているが、この後で述べる理由により、実装上は
逆伝播も呼んでいる。

条件1(FP32)・条件4(自動混合精度)について、6.13 節の3系列長水準それぞれで、
保存されたテンソルの合計バイト数を dtype 別に、また最終 2 次元が
系列長×系列長であるテンソル(Attention のスコア行列・確率行列に相当)と
それ以外のテンソルとに分けて dtype 別に集計する。

**系列長256の水準における注意点**: 本番水準の $d_{\mathrm{model}} = 256$
(5.3 節)は、3系列長水準のうち最小の $L=256$ と偶然一致する。この水準では、
`(B, L, d_{\mathrm{model}})`の形をとる通常の活性化(埋め込み出力・残差
ストリームなど)の最終 2 次元が`(256, 256)`となり、Attention のスコア行列・
確率行列と同じ「最終 2 次元が系列長×系列長」という分類条件を満たしてしまう。
そのため、$L=256$ の水準における「Attention スコア/確率行列形状」の集計には、
Attention に無関係な活性化テンソルが混入している可能性がある。集計コードの
分類条件自体は、第 1 回・第 2 回の出力と比較可能な状態を保つため変更しない。
この混入は $d_{\mathrm{model}} \neq L$ となる $L=512$・$L=1024$ の水準では
発生しないため、解釈にはこれら 2 水準を用いる。

**ストレージを共有するテンソルの重複計上を避ける**: `tensor.untyped_storage()`
の`data_ptr()`(記憶領域の先頭アドレス)が同一のテンソルは同じ物理メモリを
指すため、集計は最初に観測したストレージ 1 個につき 1 回のみとし、以降同じ
ストレージを指すテンソルはバイト数に加算しない。`data_ptr()`は記憶領域
(ストレージ)自体のアドレスであり、順伝播 1 回の間、生存しているテンソルの
間では一意であるため、この用途では正しく働く。実装・確認結果は次の
コードセルで印字する。

**この診断量の実装修正の記録**: 重複判定の鍵に、当初`tensor.untyped_storage()`
の`id()`(Python オブジェクトのアドレス)を使っていたが、これは判定に使わない
診断量そのものの実装上の誤りであり、実験 G の判定基準($\rho$ に対する閾値
0.75)・前提条件・他の実験の判定には影響しない。`id()`は、その呼び出しが
返す Python オブジェクトが解放され、別のオブジェクトに同じアドレスが再利用
されると、共有していないテンソルまで誤って重複と判定してしまう一般的な
リスクがある。ローカル実行(`SMOKE_TEST=True`)で、この実行環境の
PyTorch(2.13.0)では`untyped_storage()`が呼び出しのたびに同一の
Python オブジェクトを返すキャッシュを持つことを別途確認しており、その結果
今回のローカル実行では修正前後の実測値に差は生じなかった。このキャッシュの
有無は PyTorch のバージョン・実装に依存する詳細であり将来にわたって保証
されないため、`data_ptr()`という、キャッシュの有無によらず正しく働く鍵に
変更した。修正前後の実測値・確認内容は完了報告に記載する。

**GPU メモリ不足(CUDA out of memory)の修正の記録**: 第2回本番実行
(Google Colab T4)で、本節のセルが GPU メモリ不足で中断した。原因は、
このセルが逆伝播を呼ばずに順伝播の計算グラフを構築するだけであったこと。
`del`と`gc.collect()`だけでは計算グラフが確実には解放されず、系列長水準・
条件を切り替えて呼び出すたびに前回分のメモリが解放されないまま積み上がり、
数回目の呼び出しで GPU メモリ不足に至った。これも判定に使わない診断量の
実装上の誤りであり、実験 G の判定基準・前提条件・他の実験の判定には影響
しない。値を使わないダミーの逆伝播(`logits.sum().backward()`)を、診断量を
記録した後に呼ぶことで、計算グラフを確実に解放するよう修正した。

さらに、5.8 節と同じ切り分け方針で、`scaled_dot_product_attention`
(`src/layers/attention.py`)を単体で呼び出し、autocast 下での softmax の
出力(Attention の確率行列、`attn_weights`)自体の dtype と、それを入力とする
次の行列積(Value との積)の出力 dtype を直接確認する。**`attn_weights`は
Python レベルでは行列積にそのまま渡す同一のテンソルだが、autocast は行列積
(matmul)を FP16 にキャストして実行する演算として扱うため、`attn_weights`
自身の dtype(呼び出し元から見える値)と、行列積の内部で実際に使われる
dtype は異なりうる。** 5.8 節と同様、確認できた範囲のみを記述し、一般化した
主張はしない。



```python
def measure_saved_tensor_bytes_by_dtype(precision_level: str, seq_len: int, batch_size: int,
                                         seed: int = 0) -> dict:
    # 1回の順伝播でautogradが逆伝播用に保存するテンソルを saved_tensors_hooks で捕捉し、
    # dtype別・形状別のバイト数を集計する(事後的な診断量、判定には使わない)。
    torch.manual_seed(seed)
    d_k = D_MODEL // NUM_HEADS
    rope = RotaryPositionEmbedding(d_k, max_position=max(seq_len, SEQUENCE_LENGTH))
    model = GPTLanguageModel(
        VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, max(seq_len, SEQUENCE_LENGTH),
        positional_transform=rope, normalization_factory=RMSNorm,
        feed_forward_factory=functools.partial(SwiGLUFeedForwardNetwork, D_MODEL, SWIGLU_D_FF),
        tie_embeddings=True, norm_first=True,
    ).to(device)
    autocast_dtype = torch.float16 if precision_level == "fp16_dynamic" else None

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    inputs, _ = get_random_batch(train_ids, batch_size, seq_len, generator)
    inputs = inputs.to(device)

    # ストレージを共有するテンソルの重複計上を避ける(data_ptr()を鍵にする。
    # id(tensor.untyped_storage())は untyped_storage() が呼び出しのたびに新しい
    # Python オブジェクトを返すため、直後にそのラッパーオブジェクトが解放されて
    # 別のオブジェクトに同じアドレスが再利用されると、共有していないテンソルまで
    # 誤って重複と判定してしまう不具合があった。data_ptr() は記憶領域自体の
    # アドレスであり、順伝播 1 回の間、生存しているテンソルの間では一意である)。
    _seen_data_ptrs: set[int] = set()
    _num_skipped_as_duplicate = 0
    _records: list[tuple[str, int, tuple[int, ...]]] = []

    def _pack_hook(tensor):
        data_ptr = tensor.untyped_storage().data_ptr()
        if data_ptr in _seen_data_ptrs:
            nonlocal _num_skipped_as_duplicate
            _num_skipped_as_duplicate += 1
        else:
            _seen_data_ptrs.add(data_ptr)
            _records.append((str(tensor.dtype), tensor.untyped_storage().nbytes(), tuple(tensor.shape)))
        return tensor

    def _unpack_hook(tensor):
        return tensor

    model.train()
    with torch.autograd.graph.saved_tensors_hooks(_pack_hook, _unpack_hook):
        if autocast_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                logits = model(inputs)
        else:
            logits = model(inputs)

    # 診断量の記録(pack_hook の発火)は順伝播の時点で完了しているが、計算グラフ
    # 自体は逆伝播を呼ぶまで保持され続ける。値を使わないダミーの逆伝播を呼び、
    # 計算グラフを確実に解放する(第2回本番実行での修正、経緯は直前の
    # Markdown セルを参照)。
    logits.sum().backward()

    # 次の呼び出し(より長い系列長)に影響しないよう、確保したメモリを解放する
    # (measure_peak_memory_one_step と同じ後始末)。
    del model, inputs, logits
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    _total_by_dtype: dict[str, int] = {}
    _attn_shaped_by_dtype: dict[str, int] = {}
    _other_by_dtype: dict[str, int] = {}
    for _dtype_str, _nbytes, _shape in _records:
        _total_by_dtype[_dtype_str] = _total_by_dtype.get(_dtype_str, 0) + _nbytes
        _is_attn_shaped = len(_shape) >= 2 and _shape[-1] == seq_len and _shape[-2] == seq_len
        _target = _attn_shaped_by_dtype if _is_attn_shaped else _other_by_dtype
        _target[_dtype_str] = _target.get(_dtype_str, 0) + _nbytes

    return {
        "total_by_dtype": _total_by_dtype,
        "attn_score_shaped_by_dtype": _attn_shaped_by_dtype,
        "other_by_dtype": _other_by_dtype,
        "num_unique_tensors": len(_records),
        "num_skipped_as_duplicate_storage": _num_skipped_as_duplicate,
    }


print("--- 実験Gの事後的な診断: 逆伝播のために保存されるテンソルのdtype内訳(全件印字) ---")
_g_diag_results: dict[tuple[str, int], dict] = {}  # 可視化セルで再利用する
for _cond_name, _level in [("1", "fp32"), ("4", "fp16_dynamic")]:
    for _L in _seq_lengths:
        _diag = measure_saved_tensor_bytes_by_dtype(_level, _L, EXPERIMENT_G_BATCH_SIZE)
        _g_diag_results[(_cond_name, _L)] = _diag
        print(f"条件{_cond_name}, L={_L}: "
              f"合計(dtype別バイト数)={_diag['total_by_dtype']}, "
              f"Attentionスコア/確率行列形状(系列長x系列長、dtype別)={_diag['attn_score_shaped_by_dtype']}, "
              f"その他(dtype別)={_diag['other_by_dtype']}, "
              f"一意なテンソル数={_diag['num_unique_tensors']}, "
              f"ストレージ共有により重複計上を避けた件数={_diag['num_skipped_as_duplicate_storage']}")

# --- 単体呼び出しによる切り分け: softmaxの出力dtypeと、次の行列積に入力される時点のdtype ---
_probe_batch, _probe_heads, _probe_seq, _probe_dk = 2, NUM_HEADS, SEQUENCE_LENGTH, D_MODEL // NUM_HEADS
_probe_q = torch.randn(_probe_batch, _probe_heads, _probe_seq, _probe_dk, dtype=torch.float16, device=device)
_probe_k = torch.randn(_probe_batch, _probe_heads, _probe_seq, _probe_dk, dtype=torch.float16, device=device)
_probe_v = torch.randn(_probe_batch, _probe_heads, _probe_seq, _probe_dk, dtype=torch.float16, device=device)
with torch.autocast(device_type=device.type, dtype=torch.float16):
    _probe_output, _probe_attn_weights = scaled_dot_product_attention(_probe_q, _probe_k, _probe_v)
_dtype_matches = _probe_attn_weights.dtype == _probe_output.dtype
print(f"[{device.type}] autocast(fp16) 下での scaled_dot_product_attention: "
      f"softmax出力(確率行列)自体のdtype={_probe_attn_weights.dtype}, "
      f"それを入力とする次の行列積(Valueとの積)の出力dtype={_probe_output.dtype}, "
      f"両者が一致={_dtype_matches}"
      "(一致しない場合、Python変数としては同一のテンソルでも、"
      "autocastが行列積の内部でFP16へキャストしてから演算していることを示す)")

```

    --- 実験Gの事後的な診断: 逆伝播のために保存されるテンソルのdtype内訳(全件印字) ---
    条件1, L=256: 合計(dtype別バイト数)={'torch.int64': 65536, 'torch.float32': 1089976320, 'torch.bool': 262144}, Attentionスコア/確率行列形状(系列長x系列長、dtype別)={'torch.float32': 423624704, 'torch.bool': 262144}, その他(dtype別)={'torch.int64': 65536, 'torch.float32': 666351616}, 一意なテンソル数=127, ストレージ共有により重複計上を避けた件数=46
    条件1, L=512: 合計(dtype別バイト数)={'torch.int64': 131072, 'torch.float32': 2704160768, 'torch.bool': 1048576}, Attentionスコア/確率行列形状(系列長x系列長、dtype別)={'torch.bool': 1048576, 'torch.float32': 1073741824}, その他(dtype別)={'torch.int64': 131072, 'torch.float32': 1630418944}, 一意なテンソル数=127, ストレージ共有により重複計上を避けた件数=46
    条件1, L=1024: 合計(dtype別バイト数)={'torch.int64': 262144, 'torch.float32': 7543142400, 'torch.bool': 4194304}, Attentionスコア/確率行列形状(系列長x系列長、dtype別)={'torch.bool': 4194304, 'torch.float32': 4294967296}, その他(dtype別)={'torch.int64': 262144, 'torch.float32': 3248175104}, 一意なテンソル数=127, ストレージ共有により重複計上を避けた件数=46
    条件4, L=256: 合計(dtype別バイト数)={'torch.int64': 65536, 'torch.float32': 419734528, 'torch.float16': 519670272, 'torch.bool': 262144}, Attentionスコア/確率行列形状(系列長x系列長、dtype別)={'torch.float32': 419430400, 'torch.float16': 136314880, 'torch.bool': 262144}, その他(dtype別)={'torch.int64': 65536, 'torch.float32': 304128, 'torch.float16': 383355392}, 一意なテンソル数=143, ストレージ共有により重複計上を避けた件数=30
    条件4, L=512: 合計(dtype別バイト数)={'torch.int64': 131072, 'torch.float32': 1376330752, 'torch.float16': 1301449216, 'torch.bool': 1048576}, Attentionスコア/確率行列形状(系列長x系列長、dtype別)={'torch.bool': 1048576, 'torch.float32': 1073741824, 'torch.float16': 536870912}, その他(dtype別)={'torch.int64': 131072, 'torch.float32': 302588928, 'torch.float16': 764578304}, 一意なテンソル数=143, ストレージ共有により重複計上を避けた件数=30
    条件4, L=1024: 合計(dtype別バイト数)={'torch.int64': 262144, 'torch.float32': 4900135936, 'torch.float16': 3670313472, 'torch.bool': 4194304}, Attentionスコア/確率行列形状(系列長x系列長、dtype別)={'torch.bool': 4194304, 'torch.float32': 4294967296, 'torch.float16': 2147483648}, その他(dtype別)={'torch.int64': 262144, 'torch.float32': 605168640, 'torch.float16': 1522829824}, 一意なテンソル数=143, ストレージ共有により重複計上を避けた件数=30
    [cuda] autocast(fp16) 下での scaled_dot_product_attention: softmax出力(確率行列)自体のdtype=torch.float32, それを入力とする次の行列積(Valueとの積)の出力dtype=torch.float16, 両者が一致=False(一致しない場合、Python変数としては同一のテンソルでも、autocastが行列積の内部でFP16へキャストしてから演算していることを示す)


系列長ごとの保存テンソルのバイト数を、条件(1・4)× dtype ×分類(Attention 形状 /
その他)で積み上げた棒グラフを描画する(`plot_stacked_bar`)。あわせて、実測した
ピークメモリ(条件1・4の3水準、6.13 節)も併せて示す。$L=256$ の水準の
Attention 形状の分類には、直前の Markdown セルで述べた混入がある点に注意する。



```python
_g_categories = [f"cond{_c}, L={_L}" for _c in ("1", "4") for _L in _seq_lengths]

_g_segment_keys: set[tuple[str, str]] = set()
for _diag in _g_diag_results.values():
    _g_segment_keys.update(("attn", _d) for _d in _diag["attn_score_shaped_by_dtype"])
    _g_segment_keys.update(("other", _d) for _d in _diag["other_by_dtype"])

_g_segment_values: dict[str, list[float]] = {}
for _kind, _dtype_str in sorted(_g_segment_keys):
    _kind_label = "Attention-shaped" if _kind == "attn" else "Other"
    _segment_label = f"{_kind_label} ({_dtype_str.replace('torch.', '')})"
    _values = []
    for _c in ("1", "4"):
        for _L in _seq_lengths:
            _diag = _g_diag_results[(_c, _L)]
            _source = _diag["attn_score_shaped_by_dtype"] if _kind == "attn" else _diag["other_by_dtype"]
            _values.append(_source.get(_dtype_str, 0) / 1024**2)  # MB 単位
    _g_segment_values[_segment_label] = _values

plot_stacked_bar(
    _g_categories, _g_segment_values,
    title="Saved tensor bytes by dtype and shape category",
    ylabel="Bytes (MB)",
)
plt.show()

if device.type == "cuda":
    _g_peak_memory_series = {
        "Condition 1 (FP32)": {f"L={_L}": _memory_results[("1", _L)] / 1024**2 for _L in _seq_lengths},
        "Condition 4 (Mixed precision)": {
            f"L={_L}": _memory_results[("4", _L)] / 1024**2 for _L in _seq_lengths
        },
    }
    plot_grouped_bar(
        _g_peak_memory_series,
        title="Peak memory per training step",
        ylabel="Peak memory (MB)",
        xlabel="Sequence length",
    )
    plt.show()
else:
    print("MPS環境のためピークメモリの実測値が無く、この図は描画できない(6.13節の記載の通り)")

```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/011_mixed_precision_training/output_70_0.png)
    



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/011_mixed_precision_training/output_70_1.png)
    




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/03_efficient_training/011_mixed_precision_training.ipynb
