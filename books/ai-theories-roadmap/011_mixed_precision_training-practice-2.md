---
title: "混合精度学習(Mixed Precision Training)(実装・実験編 2/4)"
---

この記事は後編(実装・実験編 2/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/011_mixed_precision_training-practice-1)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/011_mixed_precision_training-practice-3)。

### 6.2 較正セル

以下を本番スケール(`PROD_NUM_STEPS`)で較正する。

1. **gradient clip 閾値**: 条件1(FP32、autocast なし)の較正実行における勾配
   ノルムの`CLIP_QUANTILE=0.85`分位点(007 3.4 節の方式を踏襲、5.3 節の
   変更点3の通り値自体は再較正する)。
2. **静的損失スケール $S$(条件3)**: 3.6 節で確認した通り、`torch.autocast`
   下で実際に FP16 のまま計算される活性化は各サブレイヤー(自己注意機構・
   順伝播ネットワーク)の出力である。これらの出力に対する勾配の絶対値の
   高い分位点(99.9%、最大値は使わない)を全層・全ステップから集め、その
   最大値を $A_{\text{high}}$ とする。オーバーフローの実害は活性化勾配
   そのものではなく、重み勾配 $\partial L/\partial W = (\partial L/\partial
   \text{output})^\top \cdot \text{input}$ が バッチ x 系列長 $= B \times T$
   個の項の和であることに起因する。これは次のセルの探索的検証(スケールの
   候補を実際に試し、非有限な勾配が生じる境界を探す)で確認する。安全マージンを
   $2 \times B \times T$(この和の項数の 2 倍)として、

   $$
   S = 2^{\left\lfloor \log_2 \left( \dfrac{\text{FP16\_MAX}}{A_{\text{high}}
   \times 2 B T} \right) \right\rfloor}
   $$

   で決める規則を採用する。
3. **動的損失スケーリングの設定(条件4)**: 初期値を静的スケールと同じ $S$ とし、
   growth 係数 2・backoff 係数 0.5・growth 間隔は本番ステップ数の中で growth が
   複数回起こりうる値(40)とする。
4. **前提条件 P2 の閾値**: 条件2(FP16・損失スケーリングなし)の活性化勾配の
   厳密な 0 の比率が、条件1(FP32)の同じ量を上回るべき最小マージン。この
   マージンは、条件1・条件2 **それぞれの較正実行 1 回(seed=0)** で評価する
   (本番実行では、6.7 節で全シードの活性化勾配ゼロ比率を診断量として記録し、
   この閾値との関係を確認する)。
5. **前提条件 P4 の閾値**: 実験 F の条件5(正しい順序)の clipping 発動率 $q_5$
   が入るべき範囲。
6. **実験 F の $K$**: 条件5・6を較正して $q_5$ が閾値内に入ることを確認した
   ステップ数(`PROD_CFG["K_EXPERIMENT_F"]`として 5.3 節で宣言済みの値を、
   このセルで確認する)。



```python
# --- (1) gradient clip 閾値: 条件1(FP32)の較正実行の勾配ノルム分位点 ---
_t0 = time.time()
_calib_history_cond1 = run_prod_training(seed=0, num_steps=PROD_NUM_STEPS, clip_threshold=None)
_t_cond1_calibration = time.time() - _t0
_calib_norms = np.array(_calib_history_cond1["gradient_norm"])
_calib_final_loss = _calib_history_cond1["train_loss"][-1]
print(f"条件1 較正実行: elapsed={_t_cond1_calibration:.2f}s, final_loss={_calib_final_loss:.4f}, "
      f"ln(V) x {PRECONDITION_LOSS_RATIO}={PRECONDITION_LOSS_THRESHOLD:.4f}")
_p1_calibration_ok = _calib_final_loss <= PRECONDITION_LOSS_THRESHOLD
print(f"前提条件 P1(較正実行、参考値。本番判定は 6.7 節で条件1の全シードにより行う): "
      f"{'成立' if _p1_calibration_ok else '不成立'}")

GRADIENT_CLIP_THRESHOLD = float(np.quantile(_calib_norms, CLIP_QUANTILE))
print(f"GRADIENT_CLIP_THRESHOLD(分位点={CLIP_QUANTILE}) = {GRADIENT_CLIP_THRESHOLD:.4f}")
print(f"勾配ノルム: mean={_calib_norms.mean():.4f}, std={_calib_norms.std(ddof=1):.4f}, "
      f"peak/mean={compute_gradient_norm_peak_to_mean_ratio(_calib_norms):.4f}")

```

    条件1 較正実行: elapsed=34.62s, final_loss=2.0224, ln(V) x 0.6=2.5046
    前提条件 P1(較正実行、参考値。本番判定は 6.7 節で条件1の全シードにより行う): 成立
    GRADIENT_CLIP_THRESHOLD(分位点=0.85) = 1.1187
    勾配ノルム: mean=0.9299, std=0.5758, peak/mean=4.7289



```python
# --- (2a) 静的損失スケール S の探索的検証: どのスケールから非有限勾配が生じるかを実測する ---
# 条件1の較正実行(直前のセル)と同じモデル構築(seed=0)を使い、1 ステップだけ
# 損失を候補スケールで倍にして逆伝播し、パラメータ勾配に非有限値が出るかを確認する。
def _has_non_finite_grad_at_scale(scale: float, seed: int = 0) -> bool:
    model = build_model_prod(seed).to(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    inputs, targets = get_random_batch(train_ids, PROD_BATCH_SIZE, PROD_SEQUENCE_LENGTH, generator)
    inputs, targets = inputs.to(device), targets.to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16):
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    scaled_loss = loss * scale
    model.zero_grad()
    scaled_loss.backward()
    return any(
        (not torch.isfinite(p.grad).all().item()) for p in model.parameters() if p.grad is not None
    )


_EXPLORATORY_SCALE_CANDIDATES = [2.0**k for k in (14, 15, 16, 17, 18, 19, 20)]
_exploratory_results = {
    int(math.log2(s)): _has_non_finite_grad_at_scale(s) for s in _EXPLORATORY_SCALE_CANDIDATES
}
for _k_exp, _bad in _exploratory_results.items():
    print(f"S=2^{_k_exp}={2.0**_k_exp:.3e} -> 非有限な勾配: {_bad}")

```

    S=2^14=1.638e+04 -> 非有限な勾配: False
    S=2^15=3.277e+04 -> 非有限な勾配: False
    S=2^16=6.554e+04 -> 非有限な勾配: False
    S=2^17=1.311e+05 -> 非有限な勾配: False
    S=2^18=2.621e+05 -> 非有限な勾配: True
    S=2^19=5.243e+05 -> 非有限な勾配: True
    S=2^20=1.049e+06 -> 非有限な勾配: True



```python
# --- (2b) 静的損失スケール S: サブレイヤー出力の活性化勾配の高い分位点から決める ---
STEP_PERCENTILE = 99.9
FP16_MAX = float(torch.finfo(torch.float16).max)
SAFETY_MARGIN = 2 * PROD_BATCH_SIZE * PROD_SEQUENCE_LENGTH


def _measure_activation_gradient_high_percentile(seed: int, num_steps: int) -> float:
    # 条件1(FP32)相当の学習で、各層の自己注意機構・順伝播ネットワーク出力に対する
    # 勾配の絶対値の、ステップごとの高い分位点(STEP_PERCENTILE)の最大値を測定する。
    model = build_model_prod(seed).to(device)
    optimizer = AdamW(model.parameters(), lr=BASE_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    schedule = functools.partial(
        compute_warmup_cosine_learning_rate,
        warmup_steps=PROD_WARMUP_STEPS, total_steps=PROD_NUM_STEPS,
        peak_learning_rate=BASE_LEARNING_RATE, min_learning_rate=PROD_MIN_LR,
    )
    captured: list[torch.Tensor] = []

    def make_hook():
        def hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out

            def grad_hook(grad):
                captured.append(grad.detach().abs())

            hidden.register_hook(grad_hook)

        return hook

    for block in model.blocks:
        block.self_attn.register_forward_hook(make_hook())
        block.feed_forward.register_forward_hook(make_hook())

    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    per_step_high = []
    for step in range(1, num_steps + 1):
        model.train()
        optimizer.set_learning_rate(schedule(step))
        inputs, targets = get_random_batch(train_ids, PROD_BATCH_SIZE, PROD_SEQUENCE_LENGTH, generator)
        inputs, targets = inputs.to(device), targets.to(device)
        captured.clear()
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_abs = torch.cat([g.reshape(-1) for g in captured]).float()
        per_step_high.append(torch.quantile(all_abs, STEP_PERCENTILE / 100.0).item())
    return float(np.max(per_step_high))


_t0 = time.time()
A_HIGH = _measure_activation_gradient_high_percentile(seed=0, num_steps=PROD_NUM_STEPS)
_t_activation_gradient_measurement = time.time() - _t0
print(f"elapsed={_t_activation_gradient_measurement:.2f}s, A_HIGH={A_HIGH:.6e}")

_max_scale = FP16_MAX / (A_HIGH * SAFETY_MARGIN)
_k = math.floor(math.log2(_max_scale))
STATIC_SCALE = 2.0**_k
print(f"FP16_MAX={FP16_MAX}, SAFETY_MARGIN={SAFETY_MARGIN}, max_scale={_max_scale:.3e}")
print(f"STATIC_SCALE = 2^{_k} = {STATIC_SCALE}")
assert not _exploratory_results.get(_k, False), (
    "探索的検証によれば採用した STATIC_SCALE で非有限な勾配が生じる"
)
print(f"採用した STATIC_SCALE(2^{_k})は直前の探索的検証で非有限な勾配を生じさせないことを確認済み")

# --- (3) 動的損失スケーリングの設定 ---
DYNAMIC_SCALER_KWARGS = {
    "init_scale": STATIC_SCALE,
    "growth_factor": 2.0,
    "backoff_factor": 0.5,
    "growth_interval": 40,
}
print(f"DYNAMIC_SCALER_KWARGS={DYNAMIC_SCALER_KWARGS}")

```

    elapsed=38.02s, A_HIGH=4.104892e-05
    FP16_MAX=65504.0, SAFETY_MARGIN=16384, max_scale=9.740e+04
    STATIC_SCALE = 2^16 = 65536.0
    採用した STATIC_SCALE(2^16)は直前の探索的検証で非有限な勾配を生じさせないことを確認済み
    DYNAMIC_SCALER_KWARGS={'init_scale': 65536.0, 'growth_factor': 2.0, 'backoff_factor': 0.5, 'growth_interval': 40}



```python
# --- (4) 前提条件 P2 の閾値・(5) P4 の閾値・(6) 実験 F の K ---
PRECONDITION_P2_MIN_MARGIN = 0.005  # 条件2の厳密な0の比率が条件1をこの値以上上回ること
PRECONDITION_P4_MIN, PRECONDITION_P4_MAX = 0.02, 0.98  # 条件5の発動率 q5 がこの範囲内であること


def _measure_p2_zero_ratio(use_autocast: bool, seed: int, num_steps: int) -> float:
    model = build_model_prod(seed).to(device)
    optimizer = AdamW(model.parameters(), lr=BASE_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    schedule = functools.partial(
        compute_warmup_cosine_learning_rate,
        warmup_steps=PROD_WARMUP_STEPS, total_steps=PROD_NUM_STEPS,
        peak_learning_rate=BASE_LEARNING_RATE, min_learning_rate=PROD_MIN_LR,
    )
    captured: list[torch.Tensor] = []

    def make_hook():
        def hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out

            def grad_hook(grad):
                captured.append(grad.detach())

            hidden.register_hook(grad_hook)

        return hook

    for block in model.blocks:
        block.self_attn.register_forward_hook(make_hook())
        block.feed_forward.register_forward_hook(make_hook())

    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    per_step_ratio = []
    for step in range(1, num_steps + 1):
        model.train()
        optimizer.set_learning_rate(schedule(step))
        inputs, targets = get_random_batch(train_ids, PROD_BATCH_SIZE, PROD_SEQUENCE_LENGTH, generator)
        inputs, targets = inputs.to(device), targets.to(device)
        captured.clear()
        if use_autocast:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(inputs)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        else:
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        if not torch.isfinite(loss).item():
            per_step_ratio.append(float("nan"))
            continue
        optimizer.step()
        all_vals = torch.cat([g.reshape(-1) for g in captured])
        per_step_ratio.append(compute_exact_zero_ratio(all_vals))
    return float(np.nanmean(per_step_ratio))


_t0 = time.time()
_p2_cond1 = _measure_p2_zero_ratio(use_autocast=False, seed=0, num_steps=PROD_NUM_STEPS)
_p2_cond2 = _measure_p2_zero_ratio(use_autocast=True, seed=0, num_steps=PROD_NUM_STEPS)
_t_p2_measurement = time.time() - _t0
print(f"elapsed={_t_p2_measurement:.2f}s")
print(f"条件1 exact_zero_ratio(平均)={_p2_cond1:.6f}, 条件2 exact_zero_ratio(平均)={_p2_cond2:.6f}")
_p2_margin_calibrated = _p2_cond2 - _p2_cond1
print(f"較正で観測したマージン(条件2-条件1) = {_p2_margin_calibrated:.6f}")
print(f"宣言する前提条件 P2 の最小マージン: {PRECONDITION_P2_MIN_MARGIN}")
print(f"P2 が本番で成立する見込み: {_p2_margin_calibrated >= PRECONDITION_P2_MIN_MARGIN}")

```

    elapsed=58.97s
    条件1 exact_zero_ratio(平均)=0.000000, 条件2 exact_zero_ratio(平均)=0.017184
    較正で観測したマージン(条件2-条件1) = 0.017184
    宣言する前提条件 P2 の最小マージン: 0.005
    P2 が本番で成立する見込み: True



```python
# --- 実験Fの較正: 条件5(正しい順序)・条件6(誤った順序)を K ステップ実行して q5・q6 を確認する ---
def _run_condition5_clip_flags(num_steps: int, seed: int = 0) -> np.ndarray:
    model = build_model_prod(seed).to(device)
    optimizer = AdamW(model.parameters(), lr=BASE_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    schedule = functools.partial(
        compute_warmup_cosine_learning_rate,
        warmup_steps=PROD_WARMUP_STEPS, total_steps=PROD_NUM_STEPS,
        peak_learning_rate=BASE_LEARNING_RATE, min_learning_rate=PROD_MIN_LR,
    )
    dummy_windows = torch.zeros(1, PROD_SEQUENCE_LENGTH, dtype=torch.long)
    dummy_mask = torch.ones(1, PROD_SEQUENCE_LENGTH, dtype=torch.bool)
    scaler = StaticLossScaler(STATIC_SCALE)
    history = train_language_model(
        model, train_ids, dummy_windows, dummy_mask, 1,
        num_steps=num_steps, batch_size=PROD_BATCH_SIZE, sequence_length=PROD_SEQUENCE_LENGTH,
        learning_rate=BASE_LEARNING_RATE, eval_interval=num_steps + 1, device=device, seed=seed,
        optimizer=optimizer, learning_rate_schedule=schedule,
        gradient_clip_threshold=GRADIENT_CLIP_THRESHOLD,
        autocast_dtype=torch.float16, loss_scaler=scaler,
    )
    return np.array(history["gradient_clip_triggered"], dtype=bool)


def _run_condition6_clip_flags(num_steps: int, seed: int = 0) -> np.ndarray:
    # 誤った順序(clip -> unscale)。ノートブック内でのみ実装する(4.節・3.7節)。
    model = build_model_prod(seed).to(device)
    optimizer = AdamW(model.parameters(), lr=BASE_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    schedule = functools.partial(
        compute_warmup_cosine_learning_rate,
        warmup_steps=PROD_WARMUP_STEPS, total_steps=PROD_NUM_STEPS,
        peak_learning_rate=BASE_LEARNING_RATE, min_learning_rate=PROD_MIN_LR,
    )
    scaler = StaticLossScaler(STATIC_SCALE)
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    triggered = []
    for step in range(1, num_steps + 1):
        model.train()
        optimizer.set_learning_rate(schedule(step))
        inputs, targets = get_random_batch(train_ids, PROD_BATCH_SIZE, PROD_SEQUENCE_LENGTH, generator)
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        scaled_loss = scaler.scale_loss(loss)
        optimizer.zero_grad()
        scaled_loss.backward()
        # 誤った順序: unscale する前の(scale 倍された)勾配ノルムで clipping する。
        scaled_norm_sq = sum(
            p.grad.detach().float().pow(2).sum() for p in model.parameters() if p.grad is not None
        )
        scaled_norm = float(scaled_norm_sq**0.5)
        clip_triggered = scaled_norm > GRADIENT_CLIP_THRESHOLD
        if clip_triggered:
            clip_scale = GRADIENT_CLIP_THRESHOLD / scaled_norm
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.detach().mul_(clip_scale)
        found_inf = scaler.unscale_gradients(model.parameters())
        if not found_inf:
            optimizer.step()
        triggered.append(clip_triggered)
    return np.array(triggered, dtype=bool)


_t0 = time.time()
_flags5 = _run_condition5_clip_flags(PROD_K_EXPERIMENT_F)
_t5_calibration = time.time() - _t0
_t0 = time.time()
_flags6 = _run_condition6_clip_flags(PROD_K_EXPERIMENT_F)
_t6_calibration = time.time() - _t0
_q5_calibrated = _flags5.mean()
_q6_calibrated = _flags6.mean()
print(f"K={PROD_K_EXPERIMENT_F}: 条件5 elapsed={_t5_calibration:.2f}s, q5={_q5_calibrated:.4f} "
      f"({_flags5.sum()}/{PROD_K_EXPERIMENT_F})")
print(f"条件6 elapsed={_t6_calibration:.2f}s, q6={_q6_calibrated:.4f} ({_flags6.sum()}/{PROD_K_EXPERIMENT_F})")
print(f"D_F(較正)= q6-q5 = {_q6_calibrated - _q5_calibrated:.4f}")
_p4_calibration_ok = bool(PRECONDITION_P4_MIN <= _q5_calibrated <= PRECONDITION_P4_MAX)
print(f"前提条件 P4(q5 が [{PRECONDITION_P4_MIN}, {PRECONDITION_P4_MAX}] の範囲内、較正実行): "
      f"{'成立' if _p4_calibration_ok else '不成立'}")
print(f"K_EXPERIMENT_F={PROD_K_EXPERIMENT_F} を採用する(較正で確認済み)")

```

    K=60: 条件5 elapsed=4.83s, q5=0.3167 (19/60)
    条件6 elapsed=4.66s, q6=1.0000 (60/60)
    D_F(較正)= q6-q5 = 0.6833
    前提条件 P4(q5 が [0.02, 0.98] の範囲内、較正実行): 成立
    K_EXPERIMENT_F=60 を採用する(較正で確認済み)


### 6.3 スケーリングの計測と外挿

同じ処理を $n$ 回繰り返したときの累積時間から推定するべき指数 $b$ は、
繰り返し回数についての(ほぼ自明に $b \approx 1$ になる)線形性の確認にしか
ならず、**データ量(問題の規模)に対するスケーリングの検証にはならない**。
本ノートブックで実際に重い処理(条件1〜4の学習・実験 B のフック処理)は、
いずれも本番スケール(`PROD_NUM_STEPS`)で直接計測できているため、これらは
「直接計測」として扱い、外挿は行わない。べき指数のあてはめ(3 点以上のデータ量
から本番スケールへの外挿)は、**ステップ数を実際に変えた場合の処理時間**
(実験 B のフック処理を例に)にのみ用い、直接計測できる値の検証にも使う。



```python
# --- べき指数のあてはめ(ステップ数を実際に変える、実験Bのフック処理を例に) ---
def _time_hook_instrumented_training(num_steps: int, seed: int = 0) -> float:
    _t0 = time.time()
    _measure_activation_gradient_high_percentile(seed=seed, num_steps=num_steps)
    return time.time() - _t0


_scaling_step_counts = [30, 60, 120]
_scaling_times = [_time_hook_instrumented_training(n) for n in _scaling_step_counts]
for _n, _t in zip(_scaling_step_counts, _scaling_times, strict=True):
    print(f"num_steps={_n}: elapsed={_t:.2f}s")

_fit = fit_power_law_exponent(_scaling_step_counts, _scaling_times)
print(f"べき指数 b={_fit.exponent:.3f}(標準誤差 {_fit.exponent_stderr:.3f}), "
      f"係数 a={_fit.coefficient:.3f}, R^2={_fit.r_squared:.4f}")
_extrapolated_at_prod_steps = _fit.coefficient * PROD_NUM_STEPS**_fit.exponent
print(f"外挿値(num_steps={PROD_NUM_STEPS}での予測): {_extrapolated_at_prod_steps:.2f}s")
print(f"直接計測値(6.2 節、静的損失スケール較正セルの A_HIGH 測定): "
      f"{_t_activation_gradient_measurement:.2f}s")
print(f"外挿値と直接計測値の比: {_extrapolated_at_prod_steps / _t_activation_gradient_measurement:.3f}"
      "(1 に近いほど、べき指数によるステップ数スケーリングの外挿が妥当なことを示す)")

```

    num_steps=30: elapsed=3.91s
    num_steps=60: elapsed=7.81s
    num_steps=120: elapsed=15.57s
    べき指数 b=0.997(標準誤差 0.001), 係数 a=0.131, R^2=1.0000
    外挿値(num_steps=300での予測): 38.86s
    直接計測値(6.2 節、静的損失スケール較正セルの A_HIGH 測定): 38.02s
    外挿値と直接計測値の比: 1.022(1 に近いほど、べき指数によるステップ数スケーリングの外挿が妥当なことを示す)



```python
# --- 本番実行の合計時間の見積もり(直接計測値の合計、内訳を印字する) ---
_t0 = time.time()
_direct_cond2 = run_prod_training(seed=0, num_steps=PROD_NUM_STEPS, autocast_dtype=torch.float16,
                                   clip_threshold=GRADIENT_CLIP_THRESHOLD)
_t_cond2_direct = time.time() - _t0

_t0 = time.time()
_direct_cond3 = run_prod_training(seed=0, num_steps=PROD_NUM_STEPS, autocast_dtype=torch.float16,
                                   loss_scaler=StaticLossScaler(STATIC_SCALE),
                                   clip_threshold=GRADIENT_CLIP_THRESHOLD)
_t_cond3_direct = time.time() - _t0

_t0 = time.time()
_direct_cond4 = run_prod_training(seed=0, num_steps=PROD_NUM_STEPS, autocast_dtype=torch.float16,
                                   loss_scaler=DynamicLossScaler(**DYNAMIC_SCALER_KWARGS),
                                   clip_threshold=GRADIENT_CLIP_THRESHOLD)
_t_cond4_direct = time.time() - _t0

# --- P3 が本番スケールで成立する見込みの確認(本番スケール・単一シード、参考値) ---
# 6.3 節の時間見積もりのために実行している本番スケールの条件4(_direct_cond4)の
# 学習記録をそのまま使う(この目的のためだけの追加学習は行わない)。
_prod_scale_series = np.array(_direct_cond4["loss_scale"])
_prod_n_backoff = int(np.sum(np.diff(_prod_scale_series) < 0))
_prod_n_growth = int(np.sum(np.diff(_prod_scale_series) > 0))
_prod_skip_ratio = float(np.mean(_direct_cond4["step_skipped"]))
_p3_likely_ok = _prod_n_backoff >= 1 and _prod_n_growth >= 1
print(f"本番スケール条件4(単一シード、参考値): backoff回数={_prod_n_backoff}, "
      f"growth回数={_prod_n_growth}, スケール値の範囲=[{_prod_scale_series.min():.1f}, "
      f"{_prod_scale_series.max():.1f}], 更新スキップ比率={_prod_skip_ratio:.4f}")
print(f"P3(前提条件、backoff>=1 かつ growth>=1)が本番で成立する見込み: {_p3_likely_ok}")

print(f"条件1(較正セルで既に測定済み): {_t_cond1_calibration:.2f}s")
print(f"条件2: {_t_cond2_direct:.2f}s")
print(f"条件3: {_t_cond3_direct:.2f}s")
print(f"条件4: {_t_cond4_direct:.2f}s")

_per_condition_seconds = {
    "1": _t_cond1_calibration, "2": _t_cond2_direct, "3": _t_cond3_direct, "4": _t_cond4_direct,
}
_main_grid_seconds = sum(_per_condition_seconds.values()) * PROD_NUM_SEEDS_MAIN

_calibration_seconds = (
    _t_cond1_calibration + _t_activation_gradient_measurement + _t_p2_measurement
    + _t5_calibration + _t6_calibration
)
_scaling_fit_seconds = sum(_scaling_times)  # 6.3 節のべき指数のあてはめ自体(30・60・120ステップの実行)
_experiment_b_seconds = _t_activation_gradient_measurement  # 実験B自体も同形の1回のフック付き学習
_experiment_f_seconds = (_t5_calibration + _t6_calibration) * PROD_NUM_SEEDS_MAIN
_experiment_e_seconds = 2.0  # 合成タスク、極めて軽量(較正なしで直接見積もる)
_experiment_g_seconds = sum(_per_condition_seconds[c] for c in ("1", "4")) * 3  # 系列長3水準 x 2条件
# 実験Gの事後的な診断(6.13節、saved_tensors_hooksによる保存テンソルのdtype内訳):
# 順伝播のみ(逆伝播・optimizer.stepを含まない)のため、実験G本体(順伝播+逆伝播)の
# 所要時間の半分程度で近似する(系列長3水準 x 2条件、直接計測はしていない見積もり)。
_experiment_g_diagnostic_seconds = _experiment_g_seconds * 0.5
_experiment_h_seconds = (
    (_per_condition_seconds["1"] + _per_condition_seconds["4"])
    / PROD_NUM_STEPS * (PROD_NUM_ITERATIONS_H + PROD_WARMUP_ITERATIONS_H) * 2
)
_backward_compat_seconds = 20.0  # 5.7 節、2 回のサブプロセス起動(短い学習)の実測に基づく見積もり
_visualization_seconds = 5.0  # 追加した図(matplotlib、5枚程度)の描画。軽量なため実測・外挿はしない

_total_seconds = (
    _calibration_seconds + _scaling_fit_seconds + _main_grid_seconds + _experiment_b_seconds
    + _experiment_f_seconds + _experiment_e_seconds + _experiment_g_seconds + _experiment_g_diagnostic_seconds
    + _experiment_h_seconds + _backward_compat_seconds + _visualization_seconds
)

print()
print("--- 本番実行の合計時間の見積もり(内訳) ---")
print(f"較正セル一式: {_calibration_seconds:.1f}s")
print(f"6.3節のべき指数のあてはめ(30・60・120ステップの実行): {_scaling_fit_seconds:.1f}s")
print(f"条件1〜4 x {PROD_NUM_SEEDS_MAIN}シード: {_main_grid_seconds:.1f}s")
print(f"実験B(フック込み、較正の A_HIGH 測定と同形): {_experiment_b_seconds:.1f}s")
print(f"実験F(2条件 x {PROD_NUM_SEEDS_MAIN}シード x K={PROD_K_EXPERIMENT_F}ステップ): "
      f"{_experiment_f_seconds:.1f}s")
print(f"実験E(合成タスク): {_experiment_e_seconds:.1f}s")
print(f"実験G(系列長3水準 x 2条件): {_experiment_g_seconds:.1f}s")
print(f"実験Gの事後的な診断(系列長3水準 x 2条件、順伝播のみの近似): "
      f"{_experiment_g_diagnostic_seconds:.1f}s")
print(f"実験H({PROD_NUM_ITERATIONS_H + PROD_WARMUP_ITERATIONS_H}反復 x 2条件): {_experiment_h_seconds:.1f}s")
print(f"後方互換性の検証: {_backward_compat_seconds:.1f}s")
print(f"可視化セル(追加した図の描画): {_visualization_seconds:.1f}s")
print(f"合計: {_total_seconds:.1f}s({_total_seconds / 60:.1f}分) / "
      f"セッション予算 {SESSION_BUDGET_SECONDS / 60:.0f}分")
print(f"予算に対する余裕: {SESSION_BUDGET_SECONDS / _total_seconds:.2f}倍")
if _total_seconds > SESSION_BUDGET_SECONDS:
    print("警告: 見積もり実行時間がセッション予算を超える。シード数・実験Gの水準数を見直す必要がある。")
print("注記: 上記は全てローカル(MPS)基準の値であり、Google Colab T4 での実際の実行時間とは異なる。")

```

    本番スケール条件4(単一シード、参考値): backoff回数=2, growth回数=7, スケール値の範囲=[65536.0, 4194304.0], 更新スキップ比率=0.0067
    P3(前提条件、backoff>=1 かつ growth>=1)が本番で成立する見込み: True
    条件1(較正セルで既に測定済み): 34.62s
    条件2: 22.73s
    条件3: 24.41s
    条件4: 24.35s
    
    --- 本番実行の合計時間の見積もり(内訳) ---
    較正セル一式: 141.1s
    6.3節のべき指数のあてはめ(30・60・120ステップの実行): 27.3s
    条件1〜4 x 5シード: 530.6s
    実験B(フック込み、較正の A_HIGH 測定と同形): 38.0s
    実験F(2条件 x 5シード x K=60ステップ): 47.4s
    実験E(合成タスク): 2.0s
    実験G(系列長3水準 x 2条件): 176.9s
    実験Gの事後的な診断(系列長3水準 x 2条件、順伝播のみの近似): 88.5s
    実験H(35反復 x 2条件): 13.8s
    後方互換性の検証: 20.0s
    可視化セル(追加した図の描画): 5.0s
    合計: 1090.6s(18.2分) / セッション予算 120分
    予算に対する余裕: 6.60倍
    注記: 上記は全てローカル(MPS)基準の値であり、Google Colab T4 での実際の実行時間とは異なる。


**セル出力末尾の注記についての補足**: 直前のセルの出力の末尾に印字される「上記は全て
ローカル(MPS)基準の値であり、Google Colab T4 での実際の実行時間とは異なる」という
注記は、第 1 段階(ローカルでの本番実行可能性の検証)の時点で書いた固定の文言であり、
実行環境に応じて切り替わらない。本番実行(Google Colab T4、`SMOKE_TEST=False`)の
出力に印字されている各処理の時間と見積もりの合計は、T4 上の実測値に基づく値である。

### 6.4 実験宣言セル: 検証すること・判定基準・前提条件

条件の対応:

| 条件 | 内容 | 使う実験 |
|---|---|---|
| 1 | FP32 | B, C, D, G, H |
| 2 | FP16(autocast)、損失スケーリングなし | C, D |
| 3 | FP16(autocast)、静的損失スケーリング | C, F |
| 4 | FP16(autocast)、動的損失スケーリング | D, G, H |
| 5 | 条件3と同じ(unscale -> clipping の正しい順序) | F |
| 6 | 条件3と同じだが clipping を unscale 前に行う | F |
| 7 | 合成タスク: FP16 で重みを保持して更新 | E |
| 8 | 合成タスク: FP32 マスター重みで更新 | E |

#### 実験 A: 浮動小数点形式の表現特性

**検証すること**: FP32・FP16・BF16 の最小正規数・最小非正規数・最大値・unit
roundoff の実測(`torch.finfo`とビット操作)と、値の大きさに対する相対丸め誤差
の曲線。

**判定基準**: 定性的観察のため判定基準は設けない。理論値との一致はアサーション
で確認する。

#### 実験 B: 活性化の勾配のアンダーフロー率

**検証すること**: 条件1(FP32)の学習中、記録ステップごとに各サブレイヤー
(自己注意機構・順伝播ネットワーク)の出力に対する勾配を FP16・BF16 にキャスト
したときのアンダーフロー率 $r_{\mathrm{FP16}}$・$r_{\mathrm{BF16}}$ を測定し、
FP16 のほうが BF16 よりアンダーフローしやすいことを検証する。

**判定基準**: 全層・全記録ステップで $r_{\mathrm{FP16}} \ge r_{\mathrm{BF16}}$、
かつ全層・全記録ステップの合計で $r_{\mathrm{FP16}} > r_{\mathrm{BF16}}$ ならば
支持、それ以外は反証。

**診断量**: 非正規数に落ちる要素の比率、パラメータの勾配についての同じ量。

#### 実験 C: 静的損失スケーリングの効果

損失スケーリングが **直接作用する量** は活性化の勾配のアンダーフロー(厳密な
0 の比率、実験 B・6.7 節の診断量)であり、最終 bits-per-byte はその下流の量
である(勾配のアンダーフローが実際に更新の質を落とすまでに、多数のステップの
最適化・重み共有・正規化を経由するため)。それでも最終 bits-per-byte を対比量に
するのは、実験 B・診断量がアンダーフローの **有無** を検証するのに対し、
実験 C・D が検証したい問いは「損失スケーリングは学習の質(最終的な予測性能)を
改善するか」であり、その問い自体の対比量は bits-per-byte でなければならない
ためである(アンダーフローの有無の検証と、その下流への影響の検証は別の問い)。

条件 $i$ のシード平均の最終 bits-per-byte を $b_i$、その標準誤差を $\sigma_i$
とする。

**対比量**: $D_C = (b_2 - b_1) - (b_3 - b_1) = b_2 - b_3$($b_1$ が相殺する)、
$\sigma_{D_C} = \sqrt{\sigma_2^2 + \sigma_3^2}$。

**判定基準**: $D_C > 2\sigma_{D_C}$ で支持、$D_C < -2\sigma_{D_C}$ で反証、
それ以外は判定不能。

**前提条件 P0 の範囲の改訂(旧定義・新定義・理由)**: 以前は、5.8 節のモデル
全体での dtype 確認の 6 項目(自己注意機構・順伝播ネットワークの出力が FP16、
RMSNorm・Decoder Block の出力が FP32、logits が FP16、loss が FP32)全てを
P0 の判定に含めていた(旧定義)。これを、**autocast 下でサブレイヤー
(自己注意機構・順伝播ネットワーク)の出力が FP16 になることの 2 項目のみ**
で判定するよう改める(新定義)。残りの 4 項目は診断量として引き続き印字するが、
P0 の判定には含めない。静的スケール`STATIC_SCALE`の較正(6.2 節、活性化勾配の
高い分位点から決める)・前提条件 P2(6.8.1 節)・実験 C・D の診断量(6.7 節、
サブレイヤー出力の勾配の厳密な 0 の比率)が依存しているのは「サブレイヤー出力が
実際に FP16 で計算されている」ことのみであり、それ以外の 4 項目(RMSNorm・
残差接続・logits・loss の dtype)には依存しない。依存しない項目を P0 に含めると、
検証したい仮説とも後続の測定の妥当性とも無関係な理由で前提不成立になりうる。
この改訂は観測結果の方向に依存しない一般論によるものである。

**前提条件**:

- P0: autocast 下でサブレイヤー(自己注意機構・順伝播ネットワーク)の出力が
  実際に FP16(条件2〜4)・FP32(条件1)で計算されていること(5.8 節)。
  P2 の診断量は P0 が成立していなければ意図した量を測っていない。
- P1: 条件1の最終訓練損失が $\ln V \times$ `PRECONDITION_LOSS_RATIO`(=0.60)
  以下(条件1の全シードで評価する、6.8.1 節)。
- P2: 条件2の活性化勾配の厳密な 0 の比率のシード平均が、条件1の同じ量の
  シード平均を`PRECONDITION_P2_MIN_MARGIN`以上上回ること(判定に使う
  学習そのものから記録した、本番の条件1・条件2の全シードの診断量(6.7 節)で
  評価する、6.8.1 節)。

条件2で損失または評価値が非有限値になったシードは、支持の根拠に数えず、
該当シードの数とステップを記録した上で、その実験の判定を判定不能とする
(アンダーフローとは別の機構(オーバーフロー)による非有限値のため)。条件3・4
(静的・動的損失スケーリングそのものを使う条件)で損失または評価値が非有限値に
なったシードも同様に扱う(該当シードの数とステップを記録し、支持の根拠に
数えない)。条件3・4では損失スケーリングが適切に機能していればオーバーフローは
起きにくいはずであり、非有限値が生じた場合はその事実自体を診断量として報告する。

#### 実験 D: 動的損失スケーリングの効果

実験 C と同じ理由により、対比量は最終 bits-per-byte(損失スケーリングの直接の
作用点である活性化勾配のアンダーフローの下流の量)とする。

**対比量**: $D_D = b_2 - b_4$、$\sigma_{D_D} = \sqrt{\sigma_2^2 + \sigma_4^2}$。
判定は実験 C と同じ形。

**前提条件**: P0・P1・P2(実験 C と共通)、P3: 条件4の全シードでスケール値の
backoff と growth がそれぞれ 1 回以上起きていること(6.8.1 節で評価する。
本番スケールでの成立見込みは、6.3 節の時間見積もりのために実行している
本番スケール条件4(単一シード)の学習記録から事前に確認する)。

**診断量**: スケール値の時系列、更新をスキップしたステップの比率。

#### 実験 E: FP32 マスター重みの必要性(合成タスク)

$w_0=1$ の重みに一定の更新量 $u$ を加算方向に $N$ ステップ加える。$u/w_0$ を
$2^{-8}$ から $2^{-14}$ まで公比 $1/2$ の等比で振る。

**検証すること**: 理論(3.5 節)で導出した境界通りに、条件7(FP16 で直接更新)
では更新が失われ、条件8(FP32 マスター重み)では失われないこと。

**判定基準**: 全水準で、条件7の重みの変化の有無が理論の予測と一致し、かつ
条件8の **FP32 マスター重み自身** が理論値 $w_0 + Nu$ と、FP32 の累積丸め誤差の
上界以内で一致すれば支持、1 つでも外れれば反証。期待値は理論式から独立に計算する
(実行結果からは作らない)。**条件8で比較する対象は FP32 マスター重み自身であり、
FP16 のモデル重み(FP32マスターを毎ステップ FP16 へキャストし直した写し)ではない**。
FP16 の写しは、マスターの更新が正しく蓄積されていても、読み出すたびに FP16 粒度の
量子化誤差が乗る(診断量として併記する)。

#### 実験 F: 損失スケーリングと gradient clipping の順序

条件5・6について、同じ先頭 $K$(=`K_EXPERIMENT_F`)ステップで clipping の
発動率 $q_5$・$q_6$ を測定する。

**対比量**: $D_F = q_6 - q_5$。

**$\sigma_{D_F}$ の導出(旧基準・新基準・改訂理由)**: 第1段階のスモークテストで
最初に実装した基準では、$\sigma_{D_F}$ をシードごとの $q_5$・$q_6$ の系列
(2 値からなる小標本)の標本標準偏差から直接求めていた(旧基準)。これを、
実験 C・D と同じ形(各条件の **シード平均の標準誤差** から誤差伝播で合成する、
$\sigma_{D_F} = \sqrt{\mathrm{se}_5^2 + \mathrm{se}_6^2}$)に改める(新基準)。
この改訂は、対比量が複数の測定値(シード平均)から作られる場合、誤差伝播は
個々の観測値の標本標準偏差ではなく **平均の標準誤差** から行うべきという
誤差伝播の一般論に基づくものであり、改訂の根拠はスモークテストで観測された
$D_F$ の符号や大きさに依存しない(このノートブックの実験 C・D で既に採用して
いる方式と揃えることが目的であり、スモークテストの結果を見て閾値を緩めた・
厳しくしたものではない)。スモークテストの実行結果(6.12 節)には、旧基準
($\sigma_{D_F}$ を $q_5$・$q_6$ のシード間標本標準偏差から直接導出した場合)
の判定も参考値として残す。

**判定基準**: $D_F > 2\sigma_{D_F}$ で支持、$D_F < -2\sigma_{D_F}$ で反証、
それ以外は判定不能。

**前提条件 P4 の評価対象の改訂(旧定義・新定義・理由)**: 前回までは、P4(条件5の
発動率 $q_5$ が`PRECONDITION_P4_MIN`〜`PRECONDITION_P4_MAX`の範囲内にあること)を
較正セル(6.2 節、seed=0 の 1 回のみの実行)の結果で評価していた(旧定義)。
これを、**本セルで実際に実行する実験 F(条件5の全シードの $q_5$、6.12 節)**
で評価するよう改める(新定義)。この改訂も P2 と同じ理由(前提条件は判定する
実行そのもので成立していなければならないという一般論)によるものであり、
観測結果の方向には依存しない(閾値`PRECONDITION_P4_MIN`・`PRECONDITION_P4_MAX`
の値は変えない)。P4 は 6.12 節で、実験 F を実行した後にそこで記録する。

**診断量**: unscale 後の勾配ノルムの条件6と条件5の比。

#### 実験 G: 活性化メモリの削減

条件1と条件4について、系列長を`SEQUENCE_LENGTH`の 1・2・4 倍で振り、1 ステップ
(順伝播 + 逆伝播)のピークメモリ $M_i(L)$ を測定する。

**対比量**: $\rho = \dfrac{M_4(4L) - M_4(L)}{M_1(4L) - M_1(L)}$。

**判定基準**: $\rho < 0.75$ で支持、それ以外は反証(理論セクション 3.8 節の
期待に基づく閾値、本番実行前に宣言)。

パラメータ側のメモリが理論の内訳と一致することはアサーションで確認する。
ピークメモリは T4 上で`torch.cuda.reset_peak_memory_stats()` /
`torch.cuda.max_memory_allocated()`で測る。**ローカル(MPS)ではコード経路の
確認のみを行い、数値は判定に用いない**(MPS には同等の CUDA メモリ統計 API が
無いため)。

#### 実験 H: 学習スループット

条件1と条件4の 1 ステップあたりの時間 $t_1$・$t_4$ を測定し、対比量を
$\tau = t_4/t_1$ とする。

**判定基準**: $\tau < 1 - 2\sigma_\tau$ で支持、$\tau > 1 + 2\sigma_\tau$ で
反証、それ以外は判定不能。

**前提条件**:

- P5-a(ウォームアップの担保): 先頭と末尾の数反復の平均の差でドリフトが
  ないこと。
- P5-b(判定精度の担保): 平均の標準誤差が平均の宣言割合以下であること。

反復回数は本番実行前に宣言して固定する。条件1と条件4の反復は交互に実行する。

**診断量**: モデルの各次元が 8 の倍数か(T4 の Tensor Core の利用条件)。


### 6.5 実験 A: 浮動小数点形式の表現特性



```python
def smallest_subnormal(dtype: torch.dtype) -> float:
    # 最小非正規数をビット操作で求める(仮数部の最下位ビットのみが1、指数部が全て0)。
    finfo = torch.finfo(dtype)
    itemsize = finfo.bits // 8
    int_dtype = {2: torch.int16, 4: torch.int32}[itemsize]
    bits = torch.tensor([1], dtype=int_dtype)  # 仮数部の最下位ビットのみ1
    return bits.view(dtype).item()


_results_a = {}
for _dtype, _name in [(torch.float32, "FP32"), (torch.float16, "FP16"), (torch.bfloat16, "BF16")]:
    _finfo = torch.finfo(_dtype)
    _subnormal = smallest_subnormal(_dtype)
    _results_a[_name] = {
        "smallest_normal": _finfo.smallest_normal,
        "smallest_subnormal": _subnormal,
        "max": _finfo.max,
        "eps": _finfo.eps,
        "unit_roundoff": _finfo.eps / 2,
    }
    print(f"{_name}: bits={_finfo.bits}, smallest_normal={_finfo.smallest_normal:.6e}, "
          f"smallest_subnormal={_subnormal:.6e}, max={_finfo.max:.6e}, "
          f"unit_roundoff={_finfo.eps / 2:.6e}")

# 理論値との一致をアサーションで確認する: FP16 は e=5,m=10、最小正規数 2^(1-15)=2^-14、
# 最小非正規数 2^-14 * 2^-10 = 2^-24。
assert abs(_results_a["FP16"]["smallest_normal"] - 2.0**-14) < 1e-10
assert abs(_results_a["FP16"]["smallest_subnormal"] - 2.0**-24) < 1e-12
assert abs(_results_a["FP16"]["unit_roundoff"] - 2.0**-11) < 1e-8
print("FP16 の理論値(最小正規数 2^-14・最小非正規数 2^-24・unit roundoff 2^-11)と一致")

# BF16 は e=8,m=7 で FP32 と同じ指数幅 -> 最小正規数は FP32 と同じ 2^-126。
assert abs(_results_a["BF16"]["smallest_normal"] - _results_a["FP32"]["smallest_normal"]) < 1e-40
print("BF16 の最小正規数は FP32 と一致(指数部が同じ8ビットのため)")

```

    FP32: bits=32, smallest_normal=1.175494e-38, smallest_subnormal=1.401298e-45, max=3.402823e+38, unit_roundoff=5.960464e-08
    FP16: bits=16, smallest_normal=6.103516e-05, smallest_subnormal=5.960464e-08, max=6.550400e+04, unit_roundoff=4.882812e-04
    BF16: bits=16, smallest_normal=1.175494e-38, smallest_subnormal=9.183550e-41, max=3.389531e+38, unit_roundoff=3.906250e-03
    FP16 の理論値(最小正規数 2^-14・最小非正規数 2^-24・unit roundoff 2^-11)と一致
    BF16 の最小正規数は FP32 と一致(指数部が同じ8ビットのため)



```python
# 値の大きさに対する相対丸め誤差の曲線(compute_relative_rounding_error を使用)。
_magnitudes = torch.logspace(-8, 4, 100)
fig, ax = plt.subplots(figsize=(7, 4.5))
for _dtype, _name in [(torch.float16, "FP16"), (torch.bfloat16, "BF16")]:
    _err = compute_relative_rounding_error(_magnitudes, _dtype)
    ax.plot(_magnitudes.numpy(), _err.numpy(), label=_name, alpha=0.8)
ax.set_xscale("log")
ax.set_xlabel("Absolute value of original input (log scale)")
ax.set_ylabel("Relative rounding error")
ax.set_title("Experiment A: relative rounding error vs. magnitude (round-trip cast)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/011_mixed_precision_training/output_39_0.png)
    


### 6.6 実験 B: 活性化の勾配のアンダーフロー率



```python
def run_experiment_b(seed: int, num_steps: int, record_interval: int) -> dict:
    # 条件1(FP32)の学習を行い、記録ステップごとに各サブレイヤー出力の勾配に
    # 対する FP16・BF16 のアンダーフロー率を測定する。
    model = build_model(seed).to(device)
    optimizer = AdamW(model.parameters(), lr=BASE_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    schedule = functools.partial(
        compute_warmup_cosine_learning_rate,
        warmup_steps=WARMUP_STEPS, total_steps=NUM_STEPS,
        peak_learning_rate=BASE_LEARNING_RATE, min_learning_rate=BASE_LEARNING_RATE * MIN_LEARNING_RATE_RATIO,
    )
    captured: dict[str, torch.Tensor] = {}

    def make_hook(name):
        def hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out

            def grad_hook(grad):
                captured[name] = grad.detach()

            hidden.register_hook(grad_hook)

        return hook

    layer_names = []
    for i, block in enumerate(model.blocks):
        block.self_attn.register_forward_hook(make_hook(f"blk{i}.attn"))
        block.feed_forward.register_forward_hook(make_hook(f"blk{i}.ffn"))
        layer_names += [f"blk{i}.attn", f"blk{i}.ffn"]

    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    fp16_ratios, bf16_ratios, recorded_steps = [], [], []
    for step in range(1, num_steps + 1):
        model.train()
        optimizer.set_learning_rate(schedule(step))
        inputs, targets = get_random_batch(train_ids, BATCH_SIZE, SEQUENCE_LENGTH, generator)
        inputs, targets = inputs.to(device), targets.to(device)
        captured.clear()
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % record_interval == 0:
            per_layer_fp16 = [compute_underflow_ratio(captured[n], torch.float16) for n in layer_names]
            # BF16 のラウンドトリップは CPU 上で行う(4 節の実装方針と一致させる)。
            per_layer_bf16 = [
                compute_underflow_ratio(captured[n].cpu(), torch.bfloat16) for n in layer_names
            ]
            fp16_ratios.append(per_layer_fp16)
            bf16_ratios.append(per_layer_bf16)
            recorded_steps.append(step)

    return {
        "layer_names": layer_names,
        "recorded_steps": recorded_steps,
        "fp16_ratios": np.array(fp16_ratios),  # (num_recorded, num_layers)
        "bf16_ratios": np.array(bf16_ratios),
    }


_result_b = run_experiment_b(seed=0, num_steps=NUM_STEPS, record_interval=max(1, NUM_STEPS // 5))
print(f"記録ステップ: {_result_b['recorded_steps']}")
print(f"FP16 アンダーフロー率(全層平均、ステップごと): {_result_b['fp16_ratios'].mean(axis=1)}")
print(f"BF16 アンダーフロー率(全層平均、ステップごと): {_result_b['bf16_ratios'].mean(axis=1)}")

_fp16_ge_bf16_everywhere = bool(np.all(_result_b["fp16_ratios"] >= _result_b["bf16_ratios"]))
_fp16_sum_gt_bf16_sum = bool(_result_b["fp16_ratios"].sum() > _result_b["bf16_ratios"].sum())
print(f"{_smoke_tag}全層全ステップでFP16>=BF16: {_fp16_ge_bf16_everywhere}, "
      f"合計でFP16>BF16: {_fp16_sum_gt_bf16_sum}")

```

    記録ステップ: [60, 120, 180, 240, 300]
    FP16 アンダーフロー率(全層平均、ステップごと): [0.02024859 0.01698381 0.01432586 0.01418108 0.013506  ]
    BF16 アンダーフロー率(全層平均、ステップごと): [0. 0. 0. 0. 0.]
    全層全ステップでFP16>=BF16: True, 合計でFP16>BF16: True


記録ステップに対する FP16・BF16 のアンダーフロー率の推移をサブレイヤーごとに
描画する(`plot_learning_curves_multi_seed`を再利用し、条件をFP16・BF16、
「シード」の位置にサブレイヤーを当てはめる)。



```python
_result_b_histories_by_format = {
    "FP16": [
        {"step": _result_b["recorded_steps"], "underflow_ratio": _result_b["fp16_ratios"][:, _i].tolist()}
        for _i in range(_result_b["fp16_ratios"].shape[1])
    ],
    "BF16": [
        {"step": _result_b["recorded_steps"], "underflow_ratio": _result_b["bf16_ratios"][:, _i].tolist()}
        for _i in range(_result_b["bf16_ratios"].shape[1])
    ],
}
plot_learning_curves_multi_seed(
    _result_b_histories_by_format, step_key="step", value_key="underflow_ratio",
    title="Underflow ratio over training, by format (per sublayer)",
    xlabel="Step", ylabel="Underflow ratio",
)
plt.show()

```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/011_mixed_precision_training/output_43_0.png)
    


### 6.7 条件1〜4 のシード集合の実行(実験 C・D で共有)

条件1〜4 を`NUM_SEEDS_MAIN`シードで学習し、学習記録(最終 bits-per-byte 等)を
実験 C・D で共有する(6.4 節の条件対応表の通り)。この時点の`STATIC_SCALE`・
`GRADIENT_CLIP_THRESHOLD`・`DYNAMIC_SCALER_KWARGS`は較正セル(6.2 節)で本番
スケールで決定した実際の値を使う(スモークテストの model サイズでは活性化勾配の
絶対値のスケールが異なりうるが、第1段階ではコード経路の確認を優先し、
スケール値自体は本番較正値をそのまま使う。数値上、静的スケールが過大・過小に
なり非有限値が出ても、それ自体が経路の動作確認になる)。

前提条件 P1・P2 は、本セルで実行する条件1〜4 の全シードの実際の結果を使って
評価する(6.8.1 節)。診断量(サブレイヤー出力に対する勾配の厳密な 0 の比率・
観測された dtype)は、`run_condition`が判定に使う学習そのもの(`train_language_model`
に渡すモデルに直接フックを登録する)から記録する。全シードについて全件印字する。



```python
_t0 = time.time()
condition_histories: dict[str, list[dict]] = {"1": [], "2": [], "3": [], "4": []}
initial_weight_hashes: dict[str, list[str]] = {"1": [], "2": [], "3": [], "4": []}
sublayer_diagnostics: dict[str, list[dict]] = {"1": [], "2": [], "3": [], "4": []}
non_embedding_param_counts: dict[str, list[int]] = {"1": [], "2": [], "3": [], "4": []}
for _seed in range(NUM_SEEDS_MAIN):
    _m1, h1, _hash1, _diag1 = run_condition("fp32", _seed, GRADIENT_CLIP_THRESHOLD, NUM_STEPS)
    condition_histories["1"].append(h1)
    initial_weight_hashes["1"].append(_hash1)
    sublayer_diagnostics["1"].append(_diag1)
    non_embedding_param_counts["1"].append(count_non_embedding_parameters(_m1))
    del _m1

    _m2, h2, _hash2, _diag2 = run_condition("fp16_none", _seed, GRADIENT_CLIP_THRESHOLD, NUM_STEPS)
    condition_histories["2"].append(h2)
    initial_weight_hashes["2"].append(_hash2)
    sublayer_diagnostics["2"].append(_diag2)
    non_embedding_param_counts["2"].append(count_non_embedding_parameters(_m2))
    del _m2

    _m3, h3, _hash3, _diag3 = run_condition("fp16_static", _seed, GRADIENT_CLIP_THRESHOLD, NUM_STEPS,
                                             static_scale=STATIC_SCALE)
    condition_histories["3"].append(h3)
    initial_weight_hashes["3"].append(_hash3)
    sublayer_diagnostics["3"].append(_diag3)
    non_embedding_param_counts["3"].append(count_non_embedding_parameters(_m3))
    del _m3

    _m4, h4, _hash4, _diag4 = run_condition("fp16_dynamic", _seed, GRADIENT_CLIP_THRESHOLD, NUM_STEPS,
                                             dynamic_scaler_kwargs=DYNAMIC_SCALER_KWARGS)
    condition_histories["4"].append(h4)
    initial_weight_hashes["4"].append(_hash4)
    sublayer_diagnostics["4"].append(_diag4)
    non_embedding_param_counts["4"].append(count_non_embedding_parameters(_m4))
    del _m4
print(f"条件1〜4 x {NUM_SEEDS_MAIN}シード 実行完了、elapsed={time.time() - _t0:.2f}s")

for _cond, _histories in condition_histories.items():
    _finals = [h["eval_bits_per_byte"][-1] for h in _histories]
    _n_nonfinite = sum(1 for v in _finals if not np.isfinite(v))
    print(f"条件{_cond}: 最終bits-per-byte(シードごと)={['%.4f' % v for v in _finals]}, "
          f"非有限値のシード数={_n_nonfinite}")

print()
print("--- 診断量: サブレイヤー出力の勾配の厳密な0の比率・観測されたdtype(全シード、全件印字) ---")
for _cond, _diags in sublayer_diagnostics.items():
    for _seed_idx, _d in enumerate(_diags):
        _ratio = _d["zero_count"] / _d["total_count"] if _d["total_count"] > 0 else float("nan")
        print(f"条件{_cond} seed={_seed_idx}: 厳密な0の比率={_ratio:.6f} "
              f"({_d['zero_count']}/{_d['total_count']}), 観測dtype={sorted(_d['dtypes'])}")

# --- サブレイヤー出力の dtype: 条件1はFP32のみ、条件2〜4はFP16のみであること ---
for _seed_idx, _d in enumerate(sublayer_diagnostics["1"]):
    assert _d["dtypes"] == {"torch.float32"}, (
        f"条件1 seed={_seed_idx} のサブレイヤー出力dtypeがFP32のみでない: {_d['dtypes']}"
    )
for _cond in ("2", "3", "4"):
    for _seed_idx, _d in enumerate(sublayer_diagnostics[_cond]):
        assert _d["dtypes"] == {"torch.float16"}, (
            f"条件{_cond} seed={_seed_idx} のサブレイヤー出力dtypeがFP16のみでない: {_d['dtypes']}"
        )
print("サブレイヤー出力のdtypeは、条件1で全シードFP32のみ、条件2〜4で全シードFP16のみ"
      "(FP16の経路が実際に使われていることの確認)")

```

    条件1〜4 x 5シード 実行完了、elapsed=546.44s
    条件1: 最終bits-per-byte(シードごと)=['2.8529', '2.8247', '2.8687', '2.8427', '2.8377'], 非有限値のシード数=0
    条件2: 最終bits-per-byte(シードごと)=['2.8811', '2.8702', '2.9206', '2.8856', '2.8988'], 非有限値のシード数=0
    条件3: 最終bits-per-byte(シードごと)=['2.8531', '2.8247', '2.8687', '2.8427', '2.8380'], 非有限値のシード数=0
    条件4: 最終bits-per-byte(シードごと)=['2.8517', '2.8258', '2.8695', '2.8433', '2.8386'], 非有限値のシード数=0
    
    --- 診断量: サブレイヤー出力の勾配の厳密な0の比率・観測されたdtype(全シード、全件印字) ---
    条件1 seed=0: 厳密な0の比率=0.000000 (15/5033164800), 観測dtype=['torch.float32']
    条件1 seed=1: 厳密な0の比率=0.000000 (12/5033164800), 観測dtype=['torch.float32']
    条件1 seed=2: 厳密な0の比率=0.000000 (9/5033164800), 観測dtype=['torch.float32']
    条件1 seed=3: 厳密な0の比率=0.000000 (20/5033164800), 観測dtype=['torch.float32']
    条件1 seed=4: 厳密な0の比率=0.000000 (14/5033164800), 観測dtype=['torch.float32']
    条件2 seed=0: 厳密な0の比率=0.017057 (85848756/5033164800), 観測dtype=['torch.float16']
    条件2 seed=1: 厳密な0の比率=0.016907 (85095864/5033164800), 観測dtype=['torch.float16']
    条件2 seed=2: 厳密な0の比率=0.018209 (91648563/5033164800), 観測dtype=['torch.float16']
    条件2 seed=3: 厳密な0の比率=0.018233 (91768856/5033164800), 観測dtype=['torch.float16']
    条件2 seed=4: 厳密な0の比率=0.017798 (89578602/5033164800), 観測dtype=['torch.float16']
    条件3 seed=0: 厳密な0の比率=0.000000 (1336/5033164800), 観測dtype=['torch.float16']
    条件3 seed=1: 厳密な0の比率=0.000000 (1325/5033164800), 観測dtype=['torch.float16']
    条件3 seed=2: 厳密な0の比率=0.000000 (1334/5033164800), 観測dtype=['torch.float16']
    条件3 seed=3: 厳密な0の比率=0.000000 (1341/5033164800), 観測dtype=['torch.float16']
    条件3 seed=4: 厳密な0の比率=0.000000 (1268/5033164800), 観測dtype=['torch.float16']
    条件4 seed=0: 厳密な0の比率=0.000000 (427/5033164800), 観測dtype=['torch.float16']
    条件4 seed=1: 厳密な0の比率=0.000000 (385/5033164800), 観測dtype=['torch.float16']
    条件4 seed=2: 厳密な0の比率=0.000000 (420/5033164800), 観測dtype=['torch.float16']
    条件4 seed=3: 厳密な0の比率=0.000000 (428/5033164800), 観測dtype=['torch.float16']
    条件4 seed=4: 厳密な0の比率=0.000000 (466/5033164800), 観測dtype=['torch.float16']
    サブレイヤー出力のdtypeは、条件1で全シードFP32のみ、条件2〜4で全シードFP16のみ(FP16の経路が実際に使われていることの確認)


### 6.8 不変条件のアサーション

条件1〜4 の実際の学習記録(6.7 節)を使って、以下が条件間で完全に一致することを
確認する: 評価の分母(UTF-8 バイト数)・評価窓の内容のハッシュ・学習履歴の長さ・
シード集合・学習開始前の初期重みのハッシュ・非埋め込みパラメータ数。



```python
# --- 評価の分母(UTF-8 バイト数)が全条件で同一であること ---
# 4 条件とも run_condition 経由で同じ total_eval_bytes(グローバル変数、単一の値)を
# train_language_model に渡しているため、この値自体が「全条件で同一の分母」である。
assert total_eval_bytes > 0
print(f"評価の分母(total_eval_bytes)は全条件で同一: {total_eval_bytes:,}")

# --- 評価窓(eval_windows・eval_mask)の内容のハッシュが全条件の実行後も不変であること ---
_eval_windows_hash_after = hash_tensor(eval_windows)
_eval_mask_hash_after = hash_tensor(eval_mask)
assert _eval_windows_hash_after == EVAL_WINDOWS_HASH_BEFORE
assert _eval_mask_hash_after == EVAL_MASK_HASH_BEFORE
print(f"評価窓の内容のハッシュは条件1〜4の実行前後で不変: "
      f"eval_windows={_eval_windows_hash_after[:12]}..., eval_mask={_eval_mask_hash_after[:12]}...")

# --- 学習履歴の長さが宣言した NUM_STEPS と一致すること(期待値は LEVELS の宣言値から) ---
_expected_num_steps = LEVELS[CURRENT_LEVEL_NAME]["NUM_STEPS"]
for _cond, _histories in condition_histories.items():
    for _h in _histories:
        assert len(_h["step"]) == _expected_num_steps, (
            f"条件{_cond}の学習履歴の長さが期待値と異なる: {len(_h['step'])} != {_expected_num_steps}"
        )
print(f"学習履歴の長さは全条件・全シードで NUM_STEPS(={_expected_num_steps})と一致")

# --- シード集合が宣言した NUM_SEEDS_MAIN と一致すること ---
_expected_num_seeds = LEVELS[CURRENT_LEVEL_NAME]["NUM_SEEDS_MAIN"]
for _cond, _histories in condition_histories.items():
    assert len(_histories) == _expected_num_seeds, (
        f"条件{_cond}のシード数が期待値と異なる: {len(_histories)} != {_expected_num_seeds}"
    )
print(f"シード集合の大きさは全条件で NUM_SEEDS_MAIN(={_expected_num_seeds})と一致")

# --- 学習開始前の初期重みのハッシュが、シードごとに条件間で一致すること ---
for _seed_idx in range(_expected_num_seeds):
    _hashes_this_seed = [initial_weight_hashes[_cond][_seed_idx] for _cond in ("1", "2", "3", "4")]
    assert len(set(_hashes_this_seed)) == 1, (
        f"seed={_seed_idx} の初期重みハッシュが条件間で一致しない: {_hashes_this_seed}"
    )
print("学習開始前の初期重みのハッシュは、シードごとに条件1〜4間で完全に一致する"
      "(precision_level はモデル初期化に影響しないことの確認、RNG 状態の漏れも無いことの確認)")

# --- 非埋め込みパラメータ数が精度条件によらず一定であること(重みは常に FP32 で保持される) ---
# 6.7 節で実際に学習した各モデルから数えた値(non_embedding_param_counts)を使う
# (新たに build_model() を呼んで再構築したモデルではなく、判定に使ったモデル自身で確認する)。
_all_param_counts = [c for _counts in non_embedding_param_counts.values() for c in _counts]
assert len(set(_all_param_counts)) == 1, non_embedding_param_counts
print(f"非埋め込みパラメータ数は全精度条件・全シードで共通: {_all_param_counts[0]:,}")

# --- 学習経路で BF16 が使われていないこと(5.2 節・5.8 節の構造上の保証の再確認) ---
_trainer_source = inspect.getsource(train_language_model)
assert "bfloat16" not in _trainer_source
print("train_language_model のソースに bfloat16 の参照は無い(学習経路でBF16を使わない構造上の保証)")

```

    評価の分母(total_eval_bytes)は全条件で同一: 55,769
    評価窓の内容のハッシュは条件1〜4の実行前後で不変: eval_windows=72a9b18f7a62..., eval_mask=b785f7fe7115...
    学習履歴の長さは全条件・全シードで NUM_STEPS(=300)と一致
    シード集合の大きさは全条件で NUM_SEEDS_MAIN(=5)と一致
    学習開始前の初期重みのハッシュは、シードごとに条件1〜4間で完全に一致する(precision_level はモデル初期化に影響しないことの確認、RNG 状態の漏れも無いことの確認)
    非埋め込みパラメータ数は全精度条件・全シードで共通: 3,149,056
    train_language_model のソースに bfloat16 の参照は無い(学習経路でBF16を使わない構造上の保証)




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/03_efficient_training/011_mixed_precision_training.ipynb
