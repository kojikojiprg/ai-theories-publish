---
title: "State Space Model / Mamba / State Space Model and Mamba(実装・実験編 2/6)"
---

この記事は後編(実装・実験編 2/6)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/023_state_space_model_mamba-practice-1)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/023_state_space_model_mamba-practice-3)。

### 5.5 モデルの構築・学習と評価のヘルパー・単体テスト・不変条件の確認

**モデルの構築**: 条件の種類 A1・A2(実験 A)、C1・C2(実験 C)のモデルは、`torch.manual_seed(基準 + シード番号)`の後に構築する。層数 2、$d_{\mathrm{model}} = 64$。Mamba は状態の次元 $N = 16$、拡大率 $E = 2$、カーネル幅 $K = 4$(原論文の既定値)。Transformer は 008 と同じ構成(RoPE、RMSNorm、SwiGLU、正規化前置、重み共有)で、ヘッド数 4、SwiGLU の中間次元 84 とし、**非埋め込みパラメータ数が C1 の Mamba に 0.5% 以内で揃う** ように選んだ(中間次元を 1 だけ増やすと 2 層で 384 個増える)。

**学習と評価のヘルパー**: `run_task_a()`・`run_task_c()`は、種類・シード・学習率・ステップ数を受け取って学習し、途中の評価(5.3 節で定めた位置。正解の履歴を与えた負の対数尤度(Negative Log-Likelihood)と正解率を、評価集合の先頭の部分集合で測る)と、学習の最終ステップの重みでの評価(貪欲な生成による採点。実験 C は全系列長)を記録する。較正のとき(`calibration=True`)は、較正用の集合の負の対数尤度だけを測る(評価集合は使わない)。

**単体テスト**(CPU、FP64、小さな次元)。計算結果を実装と独立な経路で計算した参照値と比べる。**許容の誤差は、実際に計算される型(ここでは FP64)の丸めの単位 $\varepsilon$ と値の大きさ、および「同じ経路の 2 回の差」と「同じ計算を別の実行の順序で行った差」の実測から導く**(固定の絶対値は使わない)。FP64 の入力は FP64 のまま計算されることもアサーションで確かめる(走査は入力が FP16・BF16・FP32 なら FP32、FP64 なら FP64 で計算する)。

1. 因果的な畳み込みが`torch.nn.functional.conv1d`(`groups` = チャネル数、左に $K-1$ 個のゼロを足す)と一致する。
2. ゼロ次ホールドの離散化が、定数入力に対する連続時間の系の厳密解と一致する。参照値は、拡大した行列の行列指数関数`torch.linalg.matrix_exp`から $\bar{A}$ と $\bar{B}$ を読んだもので、3.3 節の式を使わない。許容の誤差は、行列指数関数自身の精度(`exp(M)`と`exp(M/2)`の 2 乗の差)と丸めの単位 × 値の大きさの大きい方の 16 倍。
3. 逐次ループの走査が、数式をそのまま書いた独立な参照実装(行列指数関数で離散化し、要素ごとのスカラーのループで再帰を回す)と一致する(選択的・非選択の $B$・$C$ の形、ゼロ次ホールド・簡略形)。
   3b. 非選択の構成の走査が、畳み込みカーネル $K_k = C \bar{A}^k \bar{B}$ との畳み込みに一致し、入力を遅らせると出力も遅れるだけである(時不変)。
4. 因果性: 時刻 $t$ より後の入力を変えても、時刻 $t$ までの出力が変わらない(後ろの出力は変わる)。ブロックだけでなく言語モデル全体で、選択的・非選択の両方。
5. 1 ステップ更新を繰り返した出力・最後の状態が、系列全体の順伝播と一致する。状態を持ち回る貪欲な生成が、文脈全体を再計算する貪欲な生成と一致する。系列を途中で区切り、状態を受け渡して続きを処理しても、一括の処理と一致する。
6. Theorem 1: $N = 1$、$A = -1$、$B = 1$、$\Delta = \mathrm{softplus}$ のとき、$\bar{A} = 1 - g$、$\bar{B} = g$ と、走査の出力がゲートつき再帰 $h_t = (1 - g_t) h_{t-1} + g_t x_t$ に一致する(走査とは独立なスカラーのループ)。
7. $\Delta \to 0$ で、簡略形とゼロ次ホールドの $\bar{B}$ の相対差が単調に減り、相対差 / $(\Delta \lvert A \rvert / 2)$ が 1 に近づく($\Delta \lvert A \rvert \le 3 \times 10^{-2}$ では 1% 以内。3.3 節の級数の 2 項目から)。
8. 走査の手で書いた逆伝播が、自動微分の勾配と一致し(出力行列 $C$ の形 4 通り)、`torch.autograd.gradcheck`も通る。
9. ブロック全体の入力についての`gradcheck`が通り、activation checkpointing の有無で勾配が一致する。
10. 走査は`torch.autocast`の中でも FP32 で計算される(出力・最後の状態が FP32 で、FP32 に上げた入力の走査と bit 単位で一致する)。
11. 選択的な構成だけが $\Delta$・$B$・$C$ を入力の関数にする。1 ブロックのパラメータ数と差の内訳を印字する。
12. 合成課題の事例の生成(同じシードで同じ事例、構造、損失を掛ける位置)と評価関数の採点(正解を知っている参照モデルは 1.0、答えをずらすと 0.0、生成した答えそのものが記録される)。同じシードの条件どうしは、学習データの流れが同じである。
13. 非埋め込みパラメータ数(C1 と C2 の差が 0.5% 以内)、層数、Transformer が評価する最大の系列長まで入力できること。


```python
def build_model(kind: str, seed_index: int):
    # 条件 A1: 選択的な Mamba、A2: 非選択の Mamba(課題 A の語彙)、C1: 選択的な Mamba、C2: Transformer(課題 C の語彙)
    torch.manual_seed(INIT_SEED_BASE[kind] + seed_index)
    if kind in ("A1", "A2", "C1"):
        return MambaLanguageModel(
            TASK_A.vocabulary_size if kind in ("A1", "A2") else TASK_C.vocabulary_size,
            SYN_D_MODEL,
            SYN_NUM_LAYERS,
            state_dim=MAMBA_STATE_DIM,
            expand=MAMBA_EXPAND,
            conv_kernel=MAMBA_CONV_KERNEL,
            selective=kind != "A2",
            discretization="zero_order_hold",
        )
    max_length = max(C_LENGTHS)
    return GPTLanguageModel(
        vocabulary_size=TASK_C.vocabulary_size,
        d_model=SYN_D_MODEL,
        num_layers=SYN_NUM_LAYERS,
        num_heads=TRANSFORMER_NUM_HEADS,
        d_ff=TRANSFORMER_D_FF,
        max_sequence_length=max_length,
        positional_transform=RotaryPositionEmbedding(SYN_D_MODEL // TRANSFORMER_NUM_HEADS, max_position=max_length),
        normalization_factory=RMSNorm,
        feed_forward_factory=functools.partial(SwiGLUFeedForwardNetwork, SYN_D_MODEL, TRANSFORMER_D_FF),
        tie_embeddings=True,
        dropout=0.0,
    )


def make_batch_a(rng):
    tokens, targets, _ = generate_selective_copying(TASK_A, BATCH_A, rng)
    return tokens, targets


def make_batch_c(rng):
    tokens, answers, _ = generate_induction_heads(TASK_C, C_TRAIN_LENGTH, BATCH_C, rng)
    return tokens, induction_heads_targets(tokens, answers)


def evaluation_batch_size(length: int) -> int:
    return max(1, min(512, EVAL_TOKENS_PER_BATCH // length))


def eval_steps_for(num_steps: int) -> tuple[int, ...]:
    # 途中の評価のステップ(重複は除く。最後は必ず num_steps)
    return tuple(sorted({max(1, round(f * num_steps)) for f in EVAL_FRACTIONS}))


def loss_window(num_steps: int) -> int:
    return max(1, round(LOSS_WINDOW_FRACTION * num_steps))


def lr_grid_for(kind: str) -> tuple[float, ...]:
    return tuple(LR_CENTER[kind] * m for m in LR_GRID_MULTIPLIERS)


def probe_delta(model, tokens: torch.Tensor, task: SelectiveCopyingTask, num_sequences: int = 200) -> dict:
    # 選択的な Mamba の Δ_t(チャネルについての平均)を、入力部分のデータのトークンの位置とノイズのトークンの位置で平均する
    # (層ごと)。区切りと答えの位置の平均も添える。作用点に近い診断量(6.1 節)
    tokens = tokens[:num_sequences]
    model.set_delta_tracking(True)
    model.eval()
    with torch.no_grad():
        model(tokens.to(device))
    layers = [block.last_delta_mean.float().cpu().numpy() for block in model.blocks]  # (系列, 位置)
    model.set_delta_tracking(False)
    is_data = (tokens[:, : task.input_length] < task.num_symbols).numpy()
    is_noise = ~is_data
    result = {"data": [], "noise": [], "separator": [], "answer": [], "example": [], "example_is_data": is_data[0]}
    for delta in layers:
        inputs = delta[:, : task.input_length]
        result["data"].append(float(inputs[is_data].mean()))
        result["noise"].append(float(inputs[is_noise].mean()))
        result["separator"].append(float(delta[:, task.input_length].mean()))
        result["answer"].append(float(delta[:, task.input_length + 1 :].mean()))
        result["example"].append(delta[0])
    return result


def summarize_training(history: dict, num_steps: int) -> dict:
    window = loss_window(num_steps)
    loss = history["train_loss"]
    return {
        "train_loss": loss.astype(np.float32),
        "finite": bool(np.isfinite(loss).all()),
        "initial_loss": float(loss[:window].mean()),
        "final_loss": float(loss[-window:].mean()),
        "clip_rate": float(history["clip_triggered"].mean()),
        "seconds": history["seconds"],
    }


def run_task_a(kind: str, seed_index: int, learning_rate: float, num_steps: int, calibration: bool = False) -> dict:
    # 実験 A の学習 1 回。calibration=True なら較正用の集合の 負の対数尤度 だけを測る(評価集合は使わない)
    model = build_model(kind, seed_index)
    curve_n = CURVE_SEQUENCES

    def curve_fn(m) -> dict:
        return teacher_forced_evaluation(
            m, EVAL_A_TOKENS[:curve_n], EVAL_A_TARGETS[:curve_n], device, evaluation_batch_size(TASK_A.total_length)
        )

    history = train_sequence_task(
        model, make_batch_a, num_steps, learning_rate, device, DATA_SEED_BASE["A"] + seed_index,
        warmup_ratio=WARMUP_RATIO, min_learning_rate_ratio=MIN_LEARNING_RATE_RATIO, weight_decay=SYN_WEIGHT_DECAY,
        gradient_clip_threshold=GRADIENT_CLIP_THRESHOLD,
        eval_steps=() if calibration else eval_steps_for(num_steps), evaluation_fn=curve_fn,
    )
    record = {"kind": kind, "seed": seed_index, "learning_rate": learning_rate, "num_steps": num_steps}
    record |= summarize_training(history, num_steps)
    if calibration:
        record["calibration_nll"] = teacher_forced_evaluation(
            model, CALIBRATION_A_TOKENS, CALIBRATION_A_TARGETS, device, evaluation_batch_size(TASK_A.total_length)
        )["nll"]
    else:
        record["eval_step"] = history["eval_step"]
        record["curve"] = history["eval_results"]  # 途中の評価(正解の履歴を与えた 負の対数尤度 と正解率、評価集合の先頭の部分集合)
        record["final"] = evaluate_selective_copying(
            model, EVAL_A_TOKENS, TASK_A, device, evaluation_batch_size(TASK_A.total_length)
        )
        if kind == "A1":
            record["delta_probe"] = probe_delta(model, EVAL_A_TOKENS, TASK_A)
    del model
    empty_device_cache()
    return record


def run_task_c(kind: str, seed_index: int, learning_rate: float, num_steps: int, calibration: bool = False) -> dict:
    # 実験 C の学習 1 回。calibration=True なら、学習時の系列長の較正用の集合の 負の対数尤度 だけを測る
    model = build_model(kind, seed_index)
    curve_n = CURVE_SEQUENCES

    def curve_fn(m) -> dict:
        tokens, answers, _ = EVAL_C[C_TRAIN_LENGTH]
        return {
            "accuracy": evaluate_induction_heads(
                m, tokens[:curve_n], answers[:curve_n], device, evaluation_batch_size(C_TRAIN_LENGTH)
            )["accuracy"]
        }

    history = train_sequence_task(
        model, make_batch_c, num_steps, learning_rate, device, DATA_SEED_BASE["C"] + seed_index,
        warmup_ratio=WARMUP_RATIO, min_learning_rate_ratio=MIN_LEARNING_RATE_RATIO, weight_decay=SYN_WEIGHT_DECAY,
        gradient_clip_threshold=GRADIENT_CLIP_THRESHOLD,
        eval_steps=() if calibration else eval_steps_for(num_steps), evaluation_fn=curve_fn,
    )
    record = {"kind": kind, "seed": seed_index, "learning_rate": learning_rate, "num_steps": num_steps}
    record |= summarize_training(history, num_steps)
    if calibration:
        record["calibration_nll"] = teacher_forced_evaluation(
            model, CALIBRATION_C_TOKENS, CALIBRATION_C_TARGETS, device, evaluation_batch_size(C_TRAIN_LENGTH)
        )["nll"]
    else:
        record["eval_step"] = history["eval_step"]
        record["curve"] = history["eval_results"]
        record["by_length"] = {
            length: evaluate_induction_heads(model, tokens, answers, device, evaluation_batch_size(length))
            for length, (tokens, answers, _) in EVAL_C.items()
        }
    del model
    empty_device_cache()
    return record


def learning_precondition(record: dict) -> bool:
    # 学習の数値的な健全性: 損失がすべてのステップで有限で、最後の区間の損失が最初の区間の損失を下回る
    return record["finite"] and record["final_loss"] < record["initial_loss"]


def judge(delta: float, sigma: float) -> str:
    threshold = SIGMA_MULTIPLIER * sigma
    if delta > threshold:
        return "支持"
    if delta < -threshold:
        return "反証"
    return "判定不能"


def combined_sigma(per_seed: np.ndarray, bootstrap_contrast: np.ndarray) -> dict:
    # 対比量(シード平均)の標準偏差: シード間の分散 / n と、評価の事例のブートストラップの分散の合成
    seed_variance = float(np.var(per_seed, ddof=1)) / len(per_seed)
    bootstrap_variance = float(np.var(bootstrap_contrast, ddof=1))
    return {
        "sigma": math.sqrt(seed_variance + bootstrap_variance),
        "seed_term": math.sqrt(seed_variance),
        "bootstrap_term": math.sqrt(bootstrap_variance),
    }
```


```python
_t0_checks = time.time()
EPS64 = float(torch.finfo(torch.float64).eps)
EPS32 = float(torch.finfo(torch.float32).eps)


def max_abs_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).abs().max())


with torch.no_grad():
    # --- 許容の閾値の基準(FP64・CPU): 同じ経路の 2 回の差と、同じ計算を別の実行の順序で行った差 ---
    torch.manual_seed(0)
    _b, _l, _d, _n = 3, 24, 5, 4
    _x = torch.randn(_b, _l, _d, dtype=torch.float64)
    _delta = torch.rand(_b, _l, _d, dtype=torch.float64) * 0.5 + 0.01
    _a = -torch.arange(1, _n + 1, dtype=torch.float64).repeat(_d, 1)
    _bm = torch.randn(_b, _l, 1, _n, dtype=torch.float64)
    _cm = torch.randn(_b, _l, 1, _n, dtype=torch.float64)
    _skip = torch.randn(_d, dtype=torch.float64)
    _y, _ = selective_scan(_x, _delta, _a, _bm, _cm, _skip)
    _repeat = max_abs_difference(_y, selective_scan(_x, _delta, _a, _bm, _cm, _skip)[0])
    _per_sequence = torch.cat(
        [selective_scan(_x[i : i + 1], _delta[i : i + 1], _a, _bm[i : i + 1], _cm[i : i + 1], _skip)[0] for i in range(_b)]
    )
    _reorder = max_abs_difference(_y, _per_sequence)
    _unit = EPS64 * float(_y.abs().max())
    TOLERANCE_64 = 16 * max(_repeat, _reorder, _unit)
    print(
        f"許容の閾値の基準(FP64): 同じ経路の 2 回の差 {_repeat:.2e}、バッチ全体と 1 系列ずつの差 {_reorder:.2e}、"
        f"丸めの単位 x 出力の最大値 {_unit:.2e} -> 閾値 {TOLERANCE_64:.2e}(出力の最大値 {float(_y.abs().max()):.2f})"
    )
    assert _y.dtype == torch.float64  # 走査は入力の型(FP64)のまま計算される

    # --- 1. 因果的な depthwise の畳み込みが nn.Conv1d(左に K - 1 個の 0 を足す)と一致する ---
    _block64 = MambaBlock(16, state_dim=4).double()
    _u = torch.randn(2, _block64.d_inner, 11, dtype=torch.float64)
    _padded = torch.cat([torch.zeros(2, _block64.d_inner, MAMBA_CONV_KERNEL - 1, dtype=torch.float64), _u], dim=-1)
    _reference_conv = functional.conv1d(_padded, _block64.conv1d.weight, _block64.conv1d.bias, groups=_block64.d_inner)
    _difference = max_abs_difference(_block64.causal_convolution(_padded), _reference_conv)
    assert _difference <= 16 * max(_reorder, EPS64 * float(_reference_conv.abs().max())), _difference
    print(f"1. 因果的な畳み込みが functional.conv1d(groups = チャネル数)と一致: 差 {_difference:.2e}: OK")

    # --- 2. ゼロ次ホールドの離散化が、定数入力に対する連続時間の系の厳密解(行列指数関数)と一致する ---
    # 区間 [0, Δ] で入力 x = 1 が一定のとき、h' = A h + B は拡大した行列 M = [[A, B], [0, 0]] の指数関数で解ける:
    # exp(Δ M) = [[exp(ΔA), A^{-1}(exp(ΔA) - 1) B], [0, 1]]。上の式を使わず torch.linalg.matrix_exp から Ā・B̄ を読む
    torch.manual_seed(1)
    _delta2 = torch.rand(2, 3, 4, dtype=torch.float64) * 1.9 + 0.1  # Δ in [0.1, 2]
    _a2 = -torch.arange(1, 5, dtype=torch.float64).repeat(4, 1)  # (チャネル 4, 状態 4)
    _b2 = torch.randn(2, 3, 1, 4, dtype=torch.float64)
    _abar, _bbar = discretize_state_space(_delta2, _a2, _b2, "zero_order_hold")

    def augmented_exponential(scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        matrix = torch.zeros(2, 3, 4, 4, 2, 2, dtype=torch.float64)
        matrix[..., 0, 0] = _a2
        matrix[..., 0, 1] = _b2.expand(2, 3, 4, 4)
        exponential = torch.linalg.matrix_exp(matrix * (_delta2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * scale))
        return exponential[..., 0, 0], exponential[..., 0, 1]

    _exact_a, _exact_b = augmented_exponential(1.0)
    # 行列指数関数自身の精度の基準: exp(M) を exp(M / 2) の 2 乗として求めたときの差
    _matrix_half = torch.zeros(2, 3, 4, 4, 2, 2, dtype=torch.float64)
    _matrix_half[..., 0, 0] = _a2
    _matrix_half[..., 0, 1] = _b2.expand(2, 3, 4, 4)
    _half_exp = torch.linalg.matrix_exp(_matrix_half * (_delta2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * 0.5))
    _squared = _half_exp @ _half_exp
    _self_consistency = max(max_abs_difference(_squared[..., 0, 0], _exact_a), max_abs_difference(_squared[..., 0, 1], _exact_b))
    _scale = max(float(_exact_a.abs().max()), float(_exact_b.abs().max()), 1.0)
    _tolerance_exact = 16 * max(_self_consistency, EPS64 * _scale)
    _difference_a = max_abs_difference(_abar, _exact_a)
    _difference_b = max_abs_difference(_bbar, _exact_b)
    assert _difference_a <= _tolerance_exact and _difference_b <= _tolerance_exact, (_difference_a, _difference_b)
    print(
        f"2. ゼロ次ホールドの離散化が行列指数関数の厳密解と一致: Ā の差 {_difference_a:.2e}、B̄ の差 {_difference_b:.2e}"
        f"(閾値 {_tolerance_exact:.2e}。基準: 行列指数関数の exp(M) と exp(M/2)^2 の差 {_self_consistency:.2e}、"
        f"丸めの単位 x 値の大きさ {EPS64 * _scale:.2e}): OK"
    )

    # --- 3. 逐次ループの走査が、数式をそのまま書いた独立な参照実装(行列指数関数で離散化し、要素ごとのループ)と一致する ---
    def reference_scan(x, delta, a, b, c, skip_coefficient, method, h0=None):
        batch, length, channels = x.shape
        state = a.size(-1)
        y = torch.zeros(batch, length, channels, dtype=torch.float64)
        for i in range(batch):
            for d in range(channels):
                h = torch.zeros(state, dtype=torch.float64) if h0 is None else h0[i, d].clone()
                for t in range(length):
                    for n in range(state):
                        step = float(delta[i, t, d])
                        a_value = float(a[d, n])
                        b_value = float(b[min(i, b.size(0) - 1), min(t, b.size(1) - 1), min(d, b.size(2) - 1), n])
                        if method == "zero_order_hold":
                            matrix = torch.tensor([[a_value, b_value], [0.0, 0.0]], dtype=torch.float64)
                            exponential = torch.linalg.matrix_exp(matrix * step)
                            a_bar, b_bar = float(exponential[0, 0]), float(exponential[0, 1])
                        else:
                            a_bar, b_bar = math.exp(step * a_value), step * b_value
                        h[n] = a_bar * h[n] + b_bar * float(x[i, t, d])
                    c_row = c[min(i, c.size(0) - 1), min(t, c.size(1) - 1), min(d, c.size(2) - 1)]
                    y[i, t, d] = float((h * c_row).sum()) + float(skip_coefficient[d]) * float(x[i, t, d])
        return y

    _small = (2, 9, 3, 4)
    torch.manual_seed(2)
    _xs = torch.randn(_small[0], _small[1], _small[2], dtype=torch.float64)
    _ds = torch.rand(_small[0], _small[1], _small[2], dtype=torch.float64) * 1.5 + 0.05
    _as = -torch.arange(1, _small[3] + 1, dtype=torch.float64).repeat(_small[2], 1)
    _skip_small = torch.randn(_small[2], dtype=torch.float64)
    _scan_rows = []
    for _layout, _b_shape, _c_shape in (
        ("選択的な構成の形(B・C が入力ごと、チャネルで共通)", (2, 9, 1, 4), (2, 9, 1, 4)),
        ("非選択の構成の形(B・C がチャネルごと、入力によらない)", (1, 1, 3, 4), (1, 1, 3, 4)),
    ):
        _bs = torch.randn(*_b_shape, dtype=torch.float64)
        _cs = torch.randn(*_c_shape, dtype=torch.float64)
        for _method in ("zero_order_hold", "euler"):
            _y_impl, _ = selective_scan(_xs, _ds, _as, _bs, _cs, _skip_small, _method)
            _y_ref = reference_scan(_xs, _ds, _as, _bs, _cs, _skip_small, _method)
            _difference = max_abs_difference(_y_impl, _y_ref)
            _limit = 16 * max(_unit, EPS64 * float(_y_ref.abs().max()), _self_consistency)
            assert _difference <= _limit, (_layout, _method, _difference, _limit)
            _scan_rows.append(f"{_layout[:6]}・{_method}: 差 {_difference:.1e}")
    print("3. 逐次ループの走査が独立な参照実装(行列指数関数 + 要素ごとのループ)と一致: " + "、".join(_scan_rows) + ": OK")

    # --- 3b. 非選択の構成(時不変)は畳み込みで表せる: y = K * x、K_t = C Ā^t B̄(S4 の畳み込み表現)---
    _bs = torch.randn(1, 1, _small[2], _small[3], dtype=torch.float64)
    _cs = torch.randn(1, 1, _small[2], _small[3], dtype=torch.float64)
    _delta_constant = torch.rand(_small[2], dtype=torch.float64) * 0.5 + 0.1
    _delta_time = _delta_constant.expand(_small[0], _small[1], -1)
    _y_scan, _ = selective_scan(_xs, _delta_time, _as, _bs, _cs)
    _a_bar_c, _b_bar_c = discretize_state_space(_delta_constant.view(1, 1, -1), _as, _bs)
    _kernel = torch.stack(
        [(_cs[0, 0] * _a_bar_c[0, 0] ** t * _b_bar_c[0, 0]).sum(dim=-1) for t in range(_small[1])], dim=0
    )  # (時間 t, チャネル)
    _y_conv = torch.zeros_like(_y_scan)
    for _t in range(_small[1]):
        for _s in range(_t + 1):
            _y_conv[:, _t] += _kernel[_t - _s] * _xs[:, _s]
    _difference = max_abs_difference(_y_scan, _y_conv)
    assert _difference <= 16 * max(_unit, EPS64 * float(_y_scan.abs().max())), _difference
    # 時不変性: 入力を tau ステップ遅らせる(前に 0 を足す)と、出力も tau ステップ遅れるだけで形が変わらない
    _tau = 4
    _y_shift, _ = selective_scan(
        torch.cat([torch.zeros(_small[0], _tau, _small[2], dtype=torch.float64), _xs], dim=1),
        _delta_constant.expand(_small[0], _small[1] + _tau, -1), _as, _bs, _cs,
    )
    assert max_abs_difference(_y_shift[:, _tau:], _y_scan) <= 16 * max(_unit, EPS64 * float(_y_scan.abs().max()))
    print(f"3b. 非選択の構成の走査は畳み込み K_t = C Ā^t B̄ と一致(差 {_difference:.1e})し、時不変(入力を {_tau} ステップ遅らせると出力も遅れるだけ): OK")

    # --- 4. 因果性: 時刻 t より後の入力を変えても、時刻 t までの出力は変わらない(ブロックと言語モデル、選択的・非選択) ---
    for _selective in (True, False):
        torch.manual_seed(3)
        _model = MambaLanguageModel(20, 16, 2, state_dim=4, selective=_selective).double().eval()
        _ids = torch.randint(0, 20, (2, 30))
        _changed = _ids.clone()
        _changed[:, 17:] = torch.randint(0, 20, (2, 13))
        _out_a, _out_b = _model(_ids), _model(_changed)
        _repeat_model = max_abs_difference(_out_a, _model(_ids))
        _limit = 16 * max(_repeat_model, EPS64 * float(_out_a.abs().max()))
        assert max_abs_difference(_out_a[:, :17], _out_b[:, :17]) <= _limit
        assert max_abs_difference(_out_a[:, 17:], _out_b[:, 17:]) > _limit  # 検出力: 後ろの出力は実際に変わる
    print("4. 因果性(位置 17 以降の入力を変えても、位置 0〜16 の出力は変わらない。後ろの出力は変わる): 選択的・非選択とも OK")

    # --- 5. 1 ステップ更新を繰り返した出力が、系列全体の順伝播と一致する。状態を持ち回る生成が文脈の再計算と一致する ---
    for _selective in (True, False):
        torch.manual_seed(4)
        _model = MambaLanguageModel(20, 16, 2, state_dim=4, selective=_selective).double().eval()
        _ids = torch.randint(0, 20, (3, 21))
        _full, _states_full = _model(_ids, return_states=True)
        _states = _model.initial_states(3, torch.device("cpu"))
        assert _states[0].recurrent_state.dtype == torch.float32  # 状態は FP32 で持つ(FP64 の入力では FP64 に昇格する)
        _states = [type(s)(s.conv_buffer.double(), s.recurrent_state.double()) for s in _states]
        _stepped = []
        for _t in range(_ids.size(1)):
            _logits, _states = _model.step(_ids[:, _t], _states)
            _stepped.append(_logits)
        _stepped = torch.stack(_stepped, dim=1)
        _limit = 16 * max(_reorder, EPS64 * float(_full.abs().max()))
        _difference = max_abs_difference(_stepped, _full)
        assert _difference <= _limit, (_selective, _difference, _limit)
        assert all(
            max_abs_difference(s.recurrent_state, f.recurrent_state) <= _limit and max_abs_difference(s.conv_buffer, f.conv_buffer) <= _limit
            for s, f in zip(_states, _states_full, strict=True)
        )
        _prompt = _ids[:, :6]
        assert torch.equal(
            _model.generate(_prompt, 12, temperature=0.0, use_state=True),
            _model.generate(_prompt, 12, temperature=0.0, use_state=False),
        )
        # 系列の続きを状態から処理しても、一括の処理と一致する
        _first, _state_mid = _model(_ids[:, :9], return_states=True)
        _second, _ = _model(_ids[:, 9:], states=_state_mid, return_states=True)
        assert max_abs_difference(torch.cat([_first, _second], dim=1), _full) <= _limit
        print(f"5. 1 ステップ更新 vs 全体の順伝播({'選択的' if _selective else '非選択'}): 差 {_difference:.2e}(閾値 {_limit:.2e})。状態を持ち回る貪欲な生成 = 文脈の再計算、状態からの続きの処理 = 一括の処理: OK")

    # --- 6. Theorem 1: N = 1、A = -1、B = 1 のとき、ゲートつき再帰 h_t = (1 - g_t) h_{t-1} + g_t x_t になる ---
    torch.manual_seed(5)
    _length = 40
    _logit = torch.randn(1, _length, 1, dtype=torch.float64) * 2  # s_Delta = Linear(x) の出力(スカラー)
    _x1 = torch.randn(1, _length, 1, dtype=torch.float64)
    _delta1 = functional.softplus(_logit)  # tau_Delta = softplus
    _a1, _b1, _c1 = -torch.ones(1, 1, dtype=torch.float64), torch.ones(1, 1, 1, 1, dtype=torch.float64), torch.ones(1, 1, 1, 1, dtype=torch.float64)
    _y1, _ = selective_scan(_x1, _delta1, _a1, _b1, _c1)
    _a_bar1, _b_bar1 = discretize_state_space(_delta1, _a1, _b1, "zero_order_hold")
    _gate = torch.sigmoid(_logit)
    _h = 0.0
    _expected = []
    for _t in range(_length):  # ゲートつき再帰を、走査とは独立にスカラーのループで書く
        _h = (1.0 - float(_gate[0, _t, 0])) * _h + float(_gate[0, _t, 0]) * float(_x1[0, _t, 0])
        _expected.append(_h)
    _expected = torch.tensor(_expected, dtype=torch.float64).view(1, _length, 1)
    _limit = 16 * EPS64 * max(1.0, float(_expected.abs().max())) * 4
    assert max_abs_difference(_y1, _expected) <= _limit
    assert max_abs_difference(_a_bar1[..., 0], 1.0 - _gate) <= _limit and max_abs_difference(_b_bar1[..., 0], _gate) <= _limit
    print(
        f"6. Theorem 1(N = 1、A = -1、B = 1、Δ = softplus): Ā = 1 - g と B̄ = g(差 "
        f"{max(max_abs_difference(_a_bar1[..., 0], 1.0 - _gate), max_abs_difference(_b_bar1[..., 0], _gate)):.1e})、"
        f"走査の出力がゲートつき再帰と一致(差 {max_abs_difference(_y1, _expected):.1e}、閾値 {_limit:.1e}): OK"
    )

    # --- 7. Δ -> 0 で、簡略形 B̄ = ΔB とゼロ次ホールドの B̄ の相対差が 0 に近づく(相対差 ~ Δ|A| / 2) ---
    _a_one = -torch.tensor([[3.0]], dtype=torch.float64)
    _ratios, _relative_differences = [], []
    for _exponent in range(1, 6):
        _step = torch.full((1, 1, 1), 10.0**-_exponent, dtype=torch.float64)
        _unit_b = torch.ones(1, 1, 1, 1, dtype=torch.float64)
        _zero_order_hold = discretize_state_space(_step, _a_one, _unit_b, "zero_order_hold")[1]
        _euler = discretize_state_space(_step, _a_one, _unit_b, "euler")[1]
        _relative = float((_euler - _zero_order_hold).abs() / _zero_order_hold.abs())
        _leading = float(_step) * 3.0 / 2.0  # Δ|A| / 2
        _relative_differences.append(_relative)
        _ratios.append(_relative / _leading)
    assert all(b < a for a, b in itertools.pairwise(_relative_differences)), _relative_differences
    # (exp(u) - 1) / u = 1 + u/2 + u^2/6 + ... なので、相対差 / (u/2) = 1 + u/3 + ...。|u| <= 3e-2 では 1% 以内
    assert all(abs(r - 1.0) <= 0.01 for r in _ratios[1:]), _ratios
    print(f"7. Δ -> 0 で簡略形とゼロ次ホールドの B̄ の相対差が小さくなる(Δ = 1e-1〜1e-5): {[f'{v:.2e}' for v in _relative_differences]}、相対差 / (Δ|A|/2) = {[f'{r:.4f}' for r in _ratios]}: OK")

    # --- 8. 走査の勾配: 手で書いた逆伝播が自動微分の勾配と一致する(gradcheck も通る) ---
    with torch.enable_grad():
        from src.layers.mamba import _scan_forward, _SelectiveScanFunction

        torch.manual_seed(6)
        _grad_ok = []
        for _c_shape in ((2, 6, 1, 3), (1, 1, 3, 3), (2, 6, 3, 3), (1, 6, 3, 3)):
            _inputs = [
                torch.rand(2, 6, 3, 3, dtype=torch.float64).requires_grad_(),
                torch.randn(2, 6, 3, 3, dtype=torch.float64).requires_grad_(),
                torch.randn(*_c_shape, dtype=torch.float64).requires_grad_(),
                torch.randn(2, 3, 3, dtype=torch.float64).requires_grad_(),
            ]
            _wy, _wh = torch.randn(2, 6, 3, dtype=torch.float64), torch.randn(2, 3, 3, dtype=torch.float64)

            def _loss(fn, inputs=_inputs, wy=_wy, wh=_wh):
                y, h = fn(*inputs)
                return (y * wy).sum() + (h * wh).sum()

            _g_ref = torch.autograd.grad(_loss(lambda *a: _scan_forward(*a, None)), _inputs)
            _g_fn = torch.autograd.grad(_loss(_SelectiveScanFunction.apply), _inputs)
            _grad_ok.append(max(max_abs_difference(r, f) for r, f in zip(_g_ref, _g_fn, strict=True)))
        assert max(_grad_ok) <= 16 * max(_unit, EPS64 * 10), _grad_ok
        assert torch.autograd.gradcheck(_SelectiveScanFunction.apply, tuple(t.detach().requires_grad_() for t in _inputs))
        print(f"8. 走査の逆伝播が自動微分と一致(C の形 4 通り、最大の差 {max(_grad_ok):.1e})し、gradcheck も通る: OK")

# --- 9. 離散化を含む全体の勾配(ブロック全体を自動微分で数値微分と比べる)と、activation checkpointing で値と勾配が変わらない ---
torch.manual_seed(7)
_block_a = MambaBlock(8, state_dim=3, expand=2).double()
_block_b = MambaBlock(8, state_dim=3, expand=2, use_checkpoint=True).double()
_block_b.load_state_dict(_block_a.state_dict())
_xin = torch.randn(2, 10, 8, dtype=torch.float64, requires_grad=True)
assert torch.autograd.gradcheck(lambda t: _block_a(t), (_xin,), atol=1e-6)
_grads = []
for _block in (_block_a, _block_b):
    _block.train()
    _block.zero_grad()
    _out = _block(_xin)
    _out.square().sum().backward()
    _grads.append([p.grad.clone() for p in _block.parameters()])
_checkpoint_difference = max(max_abs_difference(a, b) for a, b in zip(_grads[0], _grads[1], strict=True))
assert _checkpoint_difference <= 16 * max(TOLERANCE_64, EPS64 * 100)
print(f"9. ブロック全体の入力についての gradcheck が通る。activation checkpointing の有無で勾配が一致(最大の差 {_checkpoint_difference:.1e}): OK")

# --- 10. 走査は torch.autocast の中でも FP32 で計算される ---
_autocast_device = "cuda" if device.type == "cuda" else "cpu"
_autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
torch.manual_seed(8)
_x16 = torch.randn(2, 12, 6, device=_autocast_device).to(_autocast_dtype)
_d16 = (torch.rand(2, 12, 6, device=_autocast_device) * 0.5 + 0.05).to(_autocast_dtype)
_a16 = -torch.arange(1, 4, dtype=torch.float32, device=_autocast_device).repeat(6, 1)
_b16 = torch.randn(2, 12, 1, 3, device=_autocast_device).to(_autocast_dtype)
_c16 = torch.randn(2, 12, 1, 3, device=_autocast_device).to(_autocast_dtype)
with torch.autocast(device_type=_autocast_device, dtype=_autocast_dtype):
    _y_autocast, _h_autocast = selective_scan(_x16, _d16, _a16, _b16, _c16)
_y_fp32, _ = selective_scan(*(t.float() for t in (_x16, _d16, _a16, _b16, _c16)))
assert _y_autocast.dtype == torch.float32 and _h_autocast.dtype == torch.float32
assert torch.equal(_y_autocast.cpu(), _y_fp32.cpu())  # autocast の中でも、FP32 に上げた入力を autocast の外で計算した結果と同じ
with torch.autocast(device_type=_autocast_device, dtype=_autocast_dtype):
    _block_autocast = MambaBlock(16, state_dim=4).to(_autocast_device)
    _out_autocast = _block_autocast(torch.randn(2, 9, 16, device=_autocast_device))
assert bool(torch.isfinite(_out_autocast).all())
print(f"10. autocast({_autocast_device}、{_autocast_dtype})の中で、走査の出力と最後の状態が FP32 で、FP32 の入力の走査と bit 単位で一致: OK")

# --- 11. 選択的な構成だけが Δ・B・C を入力の関数にする。非選択の構成の Δ・B・C は入力によらない ---
torch.manual_seed(9)
_selective_block, _plain_block = MambaBlock(16, state_dim=4), MambaBlock(16, state_dim=4, selective=False)
_u1, _u2 = torch.randn(2, 7, _selective_block.d_inner), torch.randn(2, 7, _selective_block.d_inner)
_s1, _s2 = _selective_block.selection_parameters(_u1), _selective_block.selection_parameters(_u2)
_p1, _p2 = _plain_block.selection_parameters(_u1), _plain_block.selection_parameters(_u2)
assert all(not torch.equal(a, b) for a, b in zip(_s1, _s2, strict=True))
assert all(torch.equal(a, b) for a, b in zip(_p1, _p2, strict=True))
_count_selective = sum(p.numel() for p in _selective_block.parameters())
_count_plain = sum(p.numel() for p in _plain_block.parameters())
_only_selective = {n: p.numel() for n, p in _selective_block.named_parameters() if n not in dict(_plain_block.named_parameters())}
_only_plain = {n: p.numel() for n, p in _plain_block.named_parameters() if n not in dict(_selective_block.named_parameters())}
print(
    f"11. 選択的な構成だけが Δ・B・C を入力の関数にする: OK。1 ブロックのパラメータ数(d_model 16)は選択的 {_count_selective}・非選択 {_count_plain}。"
    f"選択的だけが持つ {_only_selective}、非選択だけが持つ {_only_plain}"
)

# --- 12. 合成課題の事例の生成: 決定性・構造・評価関数 ---
_task, _rng_a, _rng_b = TASK_A, make_rng(1, 2), make_rng(1, 2)
_ta, _ya, _pa = generate_selective_copying(_task, 64, _rng_a)
_tb, _yb, _pb = generate_selective_copying(_task, 64, _rng_b)
assert torch.equal(_ta, _tb) and torch.equal(_ya, _yb) and torch.equal(_pa, _pb)  # 同じシードで同じ事例
assert not torch.equal(_ta, generate_selective_copying(_task, 64, make_rng(1, 3))[0])  # ステップが違えば別の事例
assert _ta.shape == (64, _task.total_length) and int(_ta.max()) < _task.vocabulary_size
_inputs_part, _answers_part = _ta[:, : _task.input_length], _ta[:, _task.input_length + 1 :]
assert bool((_ta[:, _task.input_length] == _task.separator_token).all())
assert bool(((_inputs_part < _task.num_symbols).sum(dim=1) == _task.num_data).all())  # データのトークンはちょうど num_data 個
assert bool((_pa[:, 1:] > _pa[:, :-1]).all())  # 位置は昇順で重複しない
assert torch.equal(torch.gather(_inputs_part, 1, _pa), _answers_part)  # 答え = 入力のデータのトークンを順序を保って並べたもの
assert bool((_ya[:, _task.input_length : _task.input_length + _task.num_data] == _answers_part).all())
assert int((_ya != IGNORE_INDEX).sum()) == 64 * _task.num_data  # 損失を掛けるのは答えのトークンの予測だけ
_ti, _ai, _fi = generate_induction_heads(TASK_C, 32, 200, make_rng(2, 1))
assert bool(((_ti == TASK_C.special_token).sum(dim=1) == 2).all()) and bool((_ti[:, -1] == TASK_C.special_token).all())
assert torch.equal(_ti[torch.arange(200), _fi], torch.full((200,), TASK_C.special_token))
assert torch.equal(_ti[torch.arange(200), _fi + 1], _ai) and bool((_ai < TASK_C.num_symbols).all())
assert int(_fi.min()) >= 0 and int(_fi.max()) <= 32 - 3
_targets_i = induction_heads_targets(_ti, _ai)
assert int((_targets_i != IGNORE_INDEX).sum()) == 200 and torch.equal(_targets_i[:, -1], _ai)


class OracleCopyingModel(torch.nn.Module):
    # 正解を知っている参照モデル: 入力部分のデータのトークンを順に出力する(評価関数の採点を確かめる)
    def __init__(self, task: SelectiveCopyingTask, flip: bool = False) -> None:
        super().__init__()
        self.task, self.flip = task, flip

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        task = self.task
        logits = torch.zeros(tokens.size(0), tokens.size(1), task.vocabulary_size)
        for i in range(tokens.size(0)):
            data = tokens[i, : task.input_length][tokens[i, : task.input_length] < task.num_symbols]
            k = tokens.size(1) - task.input_length - 1  # 生成済みの答えの数
            if k < task.num_data:
                answer = int(data[k])
                logits[i, -1, (answer + 1) % task.num_symbols if self.flip else answer] = 5.0
        return logits


class OracleInductionModel(torch.nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(tokens.size(0), tokens.size(1), TASK_C.vocabulary_size)
        for i in range(tokens.size(0)):
            first = int((tokens[i] == TASK_C.special_token).nonzero()[0])
            logits[i, -1, int(tokens[i, first + 1])] = 5.0
        return logits


_oracle = evaluate_selective_copying(OracleCopyingModel(TASK_A), _ta, TASK_A, torch.device("cpu"), 16)
_flipped = evaluate_selective_copying(OracleCopyingModel(TASK_A, flip=True), _ta, TASK_A, torch.device("cpu"), 16)
assert _oracle["token_accuracy"] == 1.0 and _oracle["exact_match"] == 1.0 and bool(_oracle["correct"].all())
assert torch.equal(torch.from_numpy(_oracle["answers"]), _answers_part)  # 生成した答えそのものが記録される
assert _flipped["token_accuracy"] == 0.0 and _flipped["exact_match"] == 0.0
_induction_oracle = evaluate_induction_heads(OracleInductionModel(), _ti, _ai, torch.device("cpu"), 16)
assert _induction_oracle["accuracy"] == 1.0 and _induction_oracle["predictions"].tolist() == _ai.tolist()


class ShiftModel(torch.nn.Module):
    # 次のトークンを入力から直接読み出す参照モデル(正解の履歴を与える評価の採点を確かめる)
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(tokens.size(0), tokens.size(1), TASK_A.vocabulary_size)
        logits[:, :-1].scatter_(2, tokens[:, 1:].unsqueeze(-1), 10.0)
        return logits


_teacher_forced = teacher_forced_evaluation(ShiftModel(), _ta, _ya, torch.device("cpu"), 16)
assert _teacher_forced["accuracy"] == 1.0 and _teacher_forced["nll"] < 0.05
print(
    "12. 合成課題の事例の生成(決定性・構造・損失の位置)と評価関数の採点(正解を知っている参照モデルは 1.0、答えをずらすと 0.0、"
    "生成した答えそのものを記録): OK"
)
_stream_a = {kind: make_batch_a(make_rng(DATA_SEED_BASE["A"] + 3, 7))[0] for kind in ("A1", "A2")}
assert torch.equal(_stream_a["A1"], _stream_a["A2"])  # 同じシードの A1 と A2 は同じ学習データの流れ(対応のある比較)
_stream_c = [make_batch_c(make_rng(DATA_SEED_BASE["C"] + 3, 7))[0] for _ in range(2)]
assert torch.equal(_stream_c[0], _stream_c[1])  # C1 と C2 も同じ学習データの流れ
assert not any(
    torch.equal(make_batch_a(make_rng(DATA_SEED_BASE["A"] + s, step))[0][:8], EVAL_A_TOKENS[:8]) for s in range(3) for step in range(1, 4)
)  # 評価集合の事例は学習データの流れに現れない(少なくとも先頭の 8 事例)

# --- 13. モデルの大きさと対応 ---
_models = {kind: build_model(kind, 0) for kind in ("A1", "A2", "C1", "C2")}
_non_embedding = {kind: count_non_embedding_parameters(m) for kind, m in _models.items()}
_total = {kind: sum(p.numel() for p in m.parameters()) for kind, m in _models.items()}
_gap = _non_embedding["C2"] - _non_embedding["C1"]
assert abs(_gap) <= 0.005 * _non_embedding["C1"], (_non_embedding, _gap)  # Transformer と Mamba の非埋め込みパラメータ数の差は 0.5% 以内
assert _non_embedding["A1"] > _non_embedding["A2"]  # 選択的な構成の方が 1 ブロックあたり (selection_projection + delta_projection の重み) - (B と C のパラメータ) だけ多い
assert _models["C2"].max_sequence_length >= max(C_LENGTHS)  # Transformer は評価する最大の系列長まで入力できる
assert len(_models["C2"].blocks) == len(_models["C1"].blocks) == SYN_NUM_LAYERS  # 層数は同じ
print(f"13. 非埋め込みパラメータ数: {dumps_compact_json(_non_embedding)}(C2 - C1 = {_gap:+d}、C1 の {abs(_gap) / _non_embedding['C1']:.2%})、総数 {dumps_compact_json(_total)}: OK")
del _models

print(f"単体テストと不変条件の確認 {time.time() - _t0_checks:.1f} 秒")
```

    許容の閾値の基準(FP64): 同じ経路の 2 回の差 0.00e+00、バッチ全体と 1 系列ずつの差 0.00e+00、丸めの単位 x 出力の最大値 1.30e-15 -> 閾値 2.09e-14(出力の最大値 5.87)
    1. 因果的な畳み込みが functional.conv1d(groups = チャネル数)と一致: 差 4.44e-16: OK
    2. ゼロ次ホールドの離散化が行列指数関数の厳密解と一致: Ā の差 3.33e-16、B̄ の差 5.55e-16(閾値 8.88e-15。基準: 行列指数関数の exp(M) と exp(M/2)^2 の差 5.55e-16、丸めの単位 x 値の大きさ 2.98e-16): OK
    3. 逐次ループの走査が独立な参照実装(行列指数関数 + 要素ごとのループ)と一致: 選択的な構成・zero_order_hold: 差 4.4e-16、選択的な構成・euler: 差 4.4e-16、非選択の構成・zero_order_hold: 差 8.9e-16、非選択の構成・euler: 差 2.2e-16: OK
    3b. 非選択の構成の走査は畳み込み K_t = C Ā^t B̄ と一致(差 4.4e-16)し、時不変(入力を 4 ステップ遅らせると出力も遅れるだけ): OK
    4. 因果性(位置 17 以降の入力を変えても、位置 0〜16 の出力は変わらない。後ろの出力は変わる): 選択的・非選択とも OK
    5. 1 ステップ更新 vs 全体の順伝播(選択的): 差 1.53e-16(閾値 1.11e-15)。状態を持ち回る貪欲な生成 = 文脈の再計算、状態からの続きの処理 = 一括の処理: OK
    5. 1 ステップ更新 vs 全体の順伝播(非選択): 差 2.22e-16(閾値 1.14e-15)。状態を持ち回る貪欲な生成 = 文脈の再計算、状態からの続きの処理 = 一括の処理: OK
    6. Theorem 1(N = 1、A = -1、B = 1、Δ = softplus): Ā = 1 - g と B̄ = g(差 1.7e-16)、走査の出力がゲートつき再帰と一致(差 2.2e-16、閾値 2.3e-14): OK
    7. Δ -> 0 で簡略形とゼロ次ホールドの B̄ の相対差が小さくなる(Δ = 1e-1〜1e-5): ['1.57e-01', '1.51e-02', '1.50e-03', '1.50e-04', '1.50e-05']、相対差 / (Δ|A|/2) = ['1.0499', '1.0050', '1.0005', '1.0000', '1.0000']: OK
    8. 走査の逆伝播が自動微分と一致(C の形 4 通り、最大の差 0.0e+00)し、gradcheck も通る: OK
    9. ブロック全体の入力についての gradcheck が通る。activation checkpointing の有無で勾配が一致(最大の差 0.0e+00): OK
    10. autocast(cuda、torch.float16)の中で、走査の出力と最後の状態が FP32 で、FP32 の入力の走査と bit 単位で一致: OK
    11. 選択的な構成だけが Δ・B・C を入力の関数にする: OK。1 ブロックのパラメータ数(d_model 16)は選択的 2208・非選択 2144。選択的だけが持つ {'selection_projection.weight': 288, 'delta_projection.weight': 32, 'delta_projection.bias': 32}、非選択だけが持つ {'delta_bias': 32, 'input_matrix': 128, 'output_matrix': 128}
    12. 合成課題の事例の生成(決定性・構造・損失の位置)と評価関数の採点(正解を知っている参照モデルは 1.0、答えをずらすと 0.0、生成した答えそのものを記録): OK
    13. 非埋め込みパラメータ数: {"A1": 65472, "A2": 63424, "C1": 65472, "C2": 65344}(C2 - C1 = -128、C1 の 0.20%)、総数 {"A1": 66112, "A2": 64064, "C1": 66560, "C2": 66432}: OK
    単体テストと不変条件の確認 16.5 秒


### 5.6 実行計画の定義と水準の構造の確認

実行計画(6.1 節)の表を作り、縮小の規則(5.3 節)が保たれていること(本番とスモークテストの水準の構造、各計画が前の計画に変更を 1 つだけ加えること、水準の刻み)をアサーションで確かめる。


```python
def build_plans() -> list[dict]:
    # 実行計画(6.1 節): 番号が大きいほど多く削る。各計画は前の計画に変更を 1 つだけ加える
    # 順序: 実験 D の学習ステップ数を下げる -> 実験 D を省く -> 実験 C のシード数 -> 実験 A のシード数 -> 実験 A・C の学習ステップ数(ともに半分)
    full, cut, d = CFG["SEEDS_FULL"], CFG["SEEDS_CUT"], CFG["D_STEP_CANDIDATES"]
    steps_a, steps_c = CFG["STEPS_A"], CFG["STEPS_C"]
    specs = [
        (full, full, (steps_a, steps_c), d[0]),
        (full, full, (steps_a, steps_c), d[1]),
        (full, full, (steps_a, steps_c), d[2]),
        (full, full, (steps_a, steps_c), None),
        (full, cut, (steps_a, steps_c), None),
        (cut, cut, (steps_a, steps_c), None),
        (cut, cut, (steps_a // 2, steps_c // 2), None),
    ]
    return [
        {"plan": i, "seeds_a": a, "seeds_c": c, "steps_a": s[0], "steps_c": s[1], "d_steps": ds}
        for i, (a, c, s, ds) in enumerate(specs)
    ]


PLANS = build_plans()
FORCED_PLAN_VALUE = os.environ.get("AI_THEORIES_FORCE_PLAN")
if FORCED_PLAN_VALUE and not SMOKE_TEST:
    raise RuntimeError(
        f"本番(SMOKE_TEST=False)では計画の強制(AI_THEORIES_FORCE_PLAN={FORCED_PLAN_VALUE!r})を受け付けない。"
        "環境変数を削除して再実行すること。"
    )
FORCED_PLAN = None if not FORCED_PLAN_VALUE else int(FORCED_PLAN_VALUE)
assert FORCED_PLAN is None or 0 <= FORCED_PLAN < len(PLANS), FORCED_PLAN
# --- 縮小規則と水準の構造の確認 ---
assert set(LEVELS["smoke"]) == set(LEVELS["prod"])
for _name, _cfg in LEVELS.items():
    assert _cfg["SEEDS_FULL"] > _cfg["SEEDS_CUT"] >= 2  # 「削る前 > 削った後 >= 2」の順序関係を保つ
    _candidates = _cfg["D_STEP_CANDIDATES"]
    assert len(_candidates) == 3 and list(_candidates) == sorted(_candidates, reverse=True)
    assert all(b == round(_candidates[0] / 2**j) for j, b in enumerate(_candidates))  # 等比(公比 1/2)
    assert _cfg["STEPS_A"] == STEPS_RATIO_A_TO_C * _cfg["STEPS_C"]  # T_A と T_C の比は本番とスモークテストで同じ
    for _steps in (_cfg["STEPS_A"], _cfg["STEPS_C"]):
        assert _steps % 2 == 0 and eval_steps_for(_steps)[-1] == _steps and eval_steps_for(_steps // 2)[-1] == _steps // 2
    _levels_b = _cfg["B_LEVELS"]
    assert len(_levels_b) >= 3 and all(b == 2 * a for a, b in itertools.pairwise(_levels_b))  # 等比(公比 2)
    assert _levels_b[0] == LEVELS["prod"]["B_LEVELS"][0]  # 下端は縮小しない(注意機構の二次の項が支配的になる系列長以上、6.1 節)
    assert _cfg["B_SWEEPS"] * _cfg["B_REPEATS"] >= 6  # 前提条件 P-B1(先頭 3 反復と末尾 3 反復の比較)が定義できる
assert [p["plan"] for p in PLANS] == list(range(len(PLANS)))
assert all(b == 2 * a for a, b in itertools.pairwise(SCALING_STEP_COUNTS)) and all(
    b == 2 * a for a, b in itertools.pairwise(D_SCALING_STEP_COUNTS)
)
assert all(math.isclose(b / a, 2.0) for a, b in itertools.pairwise(EVAL_FRACTIONS))  # 途中の評価の位置は等比
assert all(math.isclose(b / a, LR_GRID_RATIO) for kind in KINDS for a, b in itertools.pairwise(lr_grid_for(kind)))
assert C_TEST_MULTIPLIER in C_LENGTH_MULTIPLIERS and all(b == 2 * a for a, b in itertools.pairwise(C_LENGTH_MULTIPLIERS))
for _a, _b in itertools.pairwise(PLANS):  # 各計画は前の計画に変更を 1 つだけ加え、どの量も減るか変わらない
    _steps_changed = (_a["steps_a"], _a["steps_c"]) != (_b["steps_a"], _b["steps_c"])  # T_A と T_C を半分にするのは 1 つの変更
    _changes = sum(_a[k] != _b[k] for k in ("seeds_a", "seeds_c", "d_steps")) + int(_steps_changed)
    assert _changes == 1, (_a, _b)
    assert _b["seeds_a"] <= _a["seeds_a"] and _b["seeds_c"] <= _a["seeds_c"]
    assert _b["steps_a"] <= _a["steps_a"] and _b["steps_c"] <= _a["steps_c"]
    assert _b["steps_a"] == STEPS_RATIO_A_TO_C * _b["steps_c"]  # どの計画でも T_A と T_C の比は同じ
    assert (_b["d_steps"] is None) or (_a["d_steps"] is not None and _b["d_steps"] <= _a["d_steps"])

print(f"水準 {CURRENT_LEVEL_NAME!r}: {dumps_compact_json({k: v for k, v in CFG.items()})}")
print(
    f"合成課題のモデル: {SYN_NUM_LAYERS} 層、d_model {SYN_D_MODEL}。Mamba: 状態の次元 {MAMBA_STATE_DIM}、拡大率 {MAMBA_EXPAND}、カーネル幅 {MAMBA_CONV_KERNEL}。"
    f"Transformer: ヘッド数 {TRANSFORMER_NUM_HEADS}、SwiGLU の中間次元 {TRANSFORMER_D_FF}、RoPE、RMSNorm"
)
print(
    f"実験 A: 語彙 {TASK_A.vocabulary_size}(データ {TASK_A.num_symbols}・ノイズ・区切り)、入力 {TASK_A.input_length}、データのトークン "
    f"{TASK_A.num_data} 個、系列全体 {TASK_A.total_length}、バッチ {BATCH_A}、偶然の水準 {TASK_A.chance_accuracy:.3f}。"
    f"実験 C: 語彙 {TASK_C.vocabulary_size}、L_train = {C_TRAIN_LENGTH}、評価する系列長 {C_LENGTHS}、L_test = {C_TEST_LENGTH}、"
    f"バッチ {BATCH_C}、偶然の水準 {TASK_C.chance_accuracy:.4f}"
)
print("学習率の格子: " + "、".join(f"{k} {tuple(float(f'{x:.3g}') for x in lr_grid_for(k))}" for k in KINDS))
for _t in sorted({p[k] for p in PLANS for k in ("steps_a", "steps_c")}, reverse=True):
    print(f"  ステップ数 {_t}: warmup {max(1, round(WARMUP_RATIO * _t))}、評価のステップ {eval_steps_for(_t)}、損失の区間 {loss_window(_t)} ステップ")
print(f"実行計画 {len(PLANS)} 通り(番号の小さいほど優先): " + dumps_compact_json(PLANS))
print(f"実験 B: 系列長 {CFG['B_LEVELS']}、掃引 {CFG['B_SWEEPS']} 回 x 反復 {CFG['B_REPEATS']} 回、d_model {B_D_MODEL}")
if FORCED_PLAN is not None:
    print(f"*** テスト専用の上書き: AI_THEORIES_FORCE_PLAN = {FORCED_PLAN}(6.4 節で見積もりの代わりにこの計画を使う) ***")
```

    水準 'prod': {
      "STEPS_C": 2000,
      "STEPS_A": 4000,
      "SEEDS_FULL": 5,
      "SEEDS_CUT": 3,
      "EVAL_SEQUENCES": 1000,
      "CALIBRATION_SEQUENCES": 500,
      "CURVE_SEQUENCES": 250,
      "B_LEVELS": [512, 1024, 2048, 4096, 8192],
      "B_SWEEPS": 8,
      "B_REPEATS": 8,
      "D_STEP_CANDIDATES": [2181, 1090, 545],
      "D_TRAIN_CHARACTERS": null,
      "BOOTSTRAP_RESAMPLES": 10000
    }
    合成課題のモデル: 2 層、d_model 64。Mamba: 状態の次元 16、拡大率 2、カーネル幅 4。Transformer: ヘッド数 4、SwiGLU の中間次元 84、RoPE、RMSNorm
    実験 A: 語彙 10(データ 8・ノイズ・区切り)、入力 40、データのトークン 8 個、系列全体 49、バッチ 32、偶然の水準 0.125。実験 C: 語彙 17、L_train = 32、評価する系列長 (32, 64, 128, 256, 512)、L_test = 128、バッチ 64、偶然の水準 0.0625
    学習率の格子: A1 (0.002, 0.004, 0.008)、A2 (0.0005, 0.001, 0.002)、C1 (0.016, 0.032, 0.064)、C2 (0.004, 0.008, 0.016)
      ステップ数 4000: warmup 400、評価のステップ (62, 125, 250, 500, 1000, 2000, 4000)、損失の区間 200 ステップ
      ステップ数 2000: warmup 200、評価のステップ (31, 62, 125, 250, 500, 1000, 2000)、損失の区間 100 ステップ
      ステップ数 1000: warmup 100、評価のステップ (16, 31, 62, 125, 250, 500, 1000)、損失の区間 50 ステップ
    実行計画 7 通り(番号の小さいほど優先): [
      {"plan": 0, "seeds_a": 5, "seeds_c": 5, "steps_a": 4000, "steps_c": 2000, "d_steps": 2181},
      {"plan": 1, "seeds_a": 5, "seeds_c": 5, "steps_a": 4000, "steps_c": 2000, "d_steps": 1090},
      {"plan": 2, "seeds_a": 5, "seeds_c": 5, "steps_a": 4000, "steps_c": 2000, "d_steps": 545},
      {"plan": 3, "seeds_a": 5, "seeds_c": 5, "steps_a": 4000, "steps_c": 2000, "d_steps": null},
      {"plan": 4, "seeds_a": 5, "seeds_c": 3, "steps_a": 4000, "steps_c": 2000, "d_steps": null},
      {"plan": 5, "seeds_a": 3, "seeds_c": 3, "steps_a": 4000, "steps_c": 2000, "d_steps": null},
      {"plan": 6, "seeds_a": 3, "seeds_c": 3, "steps_a": 2000, "steps_c": 1000, "d_steps": null}
    ]
    実験 B: 系列長 (512, 1024, 2048, 4096, 8192)、掃引 8 回 x 反復 8 回、d_model 128


## 6. 実験 / Experiments



## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/06_architectures/023_state_space_model_mamba.ipynb
