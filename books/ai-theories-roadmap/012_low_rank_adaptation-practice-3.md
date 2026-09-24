---
title: "LoRA(Low-Rank Adaptation)(実装・実験編 3/3)"
---

この記事は後編(実装・実験編 3/3)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/012_low_rank_adaptation-practice-2)。

### 6.9 実験 B: 同じパラメータ数の対照との比較


```python
d_B = b3_mean - b2_mean
sigma_d_B = math.sqrt(sigma_b3**2 + sigma_b2**2)
verdict_B = three_way_verdict(d_B, sigma_d_B)


def train_loss_dropped(records: list[dict]) -> bool:
    return all(
        r["final_train_loss"] <= (1 - PRECONDITION_TRAIN_LOSS_DROP) * r["initial_train_loss"]
        for r in records
    )


precondition_status["B1"] = train_loss_dropped(main_records["lora"]) and train_loss_dropped(
    main_records["sparse"]
)
_final_loss_lora, _ = seed_mean_and_standard_error(main_records["lora"], "final_train_loss")
_final_loss_sparse, _ = seed_mean_and_standard_error(main_records["sparse"], "final_train_loss")

print(
    f"{_smoke_tag}b3={b3_mean:.4f}(標準誤差 {sigma_b3:.4f}), b2={b2_mean:.4f}(標準誤差 {sigma_b2:.4f})"
)
print(
    f"{_smoke_tag}d = b3 - b2 = {d_B:.4f}, sigma_d = {sigma_d_B:.4f}, 2 sigma_d = {2 * sigma_d_B:.4f}"
)
print(
    f"{_smoke_tag}前提条件 B1(両方式の全シードで訓練損失が 5% 以上低下): {precondition_status['B1']}"
)
print(f"{_smoke_tag}判定関数の結果: {verdict_B}")
print(
    f"{_smoke_tag}診断量: 最終訓練損失の差(疎微調整 - LoRA)= {_final_loss_sparse - _final_loss_lora:+.4f} nats"
    f"(疎微調整 {_final_loss_sparse:.4f}、LoRA {_final_loss_lora:.4f})"
)
for _method in ("lora", "sparse"):
    print(
        f"{_smoke_tag}診断量: 学習曲線の最終区間の傾き(シード平均、bits-per-byte / ステップ)"
        f"[{_method}]: {mean_final_slope(main_records[_method]):+.6f}"
    )
_ratios_sparse = mean_over_records(main_records["sparse"], "relative_update_norms")
print(
    f"{_smoke_tag}診断量 ||Delta W||_F / ||W0||_F [sparse]: "
    + ", ".join(f"{x:.4f}" for x in _ratios_sparse)
)
```

    b3=2.6391(標準誤差 0.0050), b2=2.6113(標準誤差 0.0032)
    d = b3 - b2 = 0.0279, sigma_d = 0.0059, 2 sigma_d = 0.0119
    前提条件 B1(両方式の全シードで訓練損失が 5% 以上低下): True
    判定関数の結果: 支持
    診断量: 最終訓練損失の差(疎微調整 - LoRA)= -0.0092 nats(疎微調整 4.6563、LoRA 4.6656)
    診断量: 学習曲線の最終区間の傾き(シード平均、bits-per-byte / ステップ)[lora]: -0.000178
    診断量: 学習曲線の最終区間の傾き(シード平均、bits-per-byte / ステップ)[sparse]: -0.000145
    診断量 ||Delta W||_F / ||W0||_F [sparse]: 1.3281, 1.0830, 2.0418, 1.7238, 2.0019, 1.8546, 2.1152, 2.0841


### 6.10 実験 C: rank の飽和


```python
rank_summary = {
    r: dict(zip(("mean", "sigma"), seed_mean_and_standard_error(rs), strict=True))
    for r, rs in rank_records.items()
}
c_C = rank_summary[1]["mean"] - 2 * rank_summary[8]["mean"] + rank_summary[64]["mean"]
sigma_c_C = math.sqrt(
    rank_summary[1]["sigma"] ** 2
    + 4 * rank_summary[8]["sigma"] ** 2
    + rank_summary[64]["sigma"] ** 2
)
verdict_C = three_way_verdict(c_C, sigma_c_C)
precondition_status["C1"] = all(train_loss_dropped(rs) for rs in rank_records.values())

for _r, _s in rank_summary.items():
    _update = float(np.mean([r["relative_update_norms"] for r in rank_records[_r]]))
    print(
        f"{_smoke_tag}r={_r:>2}: b={_s['mean']:.4f}(標準誤差 {_s['sigma']:.4f}、{len(rank_records[_r])} シード)、"
        f"診断量 ||(alpha/r) B A||_F / ||W0||_F の平均 = {_update:.4f}、"
        f"学習曲線の最終区間の傾き(シード平均)= {mean_final_slope(rank_records[_r]):+.6f}"
    )
print(
    f"{_smoke_tag}Delta_low = b(r=1) - b(r=8) = {rank_summary[1]['mean'] - rank_summary[8]['mean']:.4f}, "
    f"Delta_high = b(r=8) - b(r=64) = {rank_summary[8]['mean'] - rank_summary[64]['mean']:.4f}"
)
print(f"{_smoke_tag}c = {c_C:.4f}, sigma_c = {sigma_c_C:.4f}, 2 sigma_c = {2 * sigma_c_C:.4f}")
print(
    f"{_smoke_tag}前提条件 C1(全 rank の全シードで訓練損失が 5% 以上低下): {precondition_status['C1']}"
)
print(f"{_smoke_tag}判定関数の結果: {verdict_C}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
_rs = list(rank_summary)
axes[0].errorbar(
    _rs,
    [rank_summary[r]["mean"] for r in _rs],
    yerr=[
        2 * rank_summary[r]["sigma"] if math.isfinite(rank_summary[r]["sigma"]) else 0 for r in _rs
    ],
    fmt="o-",
    capsize=4,
    label="LoRA (mean +/- 2 standard errors)",
)
for _r in _rs:
    axes[0].scatter(
        [_r] * len(rank_records[_r]),
        [x["final_bits_per_byte"] for x in rank_records[_r]],
        color="tab:gray",
        s=12,
        alpha=0.6,
    )
axes[0].axhline(b1_mean, color="tab:red", linestyle="--", label="full fine-tuning (mean)")
axes[0].set_xscale("log", base=2)
axes[0].set_xlabel("rank r")
axes[0].set_ylabel("final evaluation bits-per-byte")
axes[0].set_title("Rank sweep (alpha fixed, scaling alpha / r)")
axes[0].legend(fontsize=8)
axes[1].plot(
    _rs, [np.mean([r["relative_update_norms"] for r in rank_records[x]]) for x in _rs], "o-"
)
axes[1].set_xscale("log", base=2)
axes[1].set_xlabel("rank r")
axes[1].set_ylabel("||(alpha / r) B A||_F / ||W0||_F")
axes[1].set_title("Relative update norm (Query and Value, mean)")
fig.tight_layout()
plt.show()
```

    r= 1: b=2.7792(標準誤差 0.0030、5 シード)、診断量 ||(alpha/r) B A||_F / ||W0||_F の平均 = 0.9982、学習曲線の最終区間の傾き(シード平均)= -0.000186
    r= 2: b=2.7172(標準誤差 0.0053、2 シード)、診断量 ||(alpha/r) B A||_F / ||W0||_F の平均 = 1.0291、学習曲線の最終区間の傾き(シード平均)= -0.000185
    r= 4: b=2.6622(標準誤差 0.0054、2 シード)、診断量 ||(alpha/r) B A||_F / ||W0||_F の平均 = 1.0357、学習曲線の最終区間の傾き(シード平均)= -0.000112
    r= 8: b=2.6113(標準誤差 0.0032、5 シード)、診断量 ||(alpha/r) B A||_F / ||W0||_F の平均 = 0.9640、学習曲線の最終区間の傾き(シード平均)= -0.000178
    r=16: b=2.5863(標準誤差 0.0025、2 シード)、診断量 ||(alpha/r) B A||_F / ||W0||_F の平均 = 0.9095、学習曲線の最終区間の傾き(シード平均)= -0.000171
    r=32: b=2.5677(標準誤差 0.0009、2 シード)、診断量 ||(alpha/r) B A||_F / ||W0||_F の平均 = 0.8685、学習曲線の最終区間の傾き(シード平均)= -0.000204
    r=64: b=2.5635(標準誤差 0.0014、5 シード)、診断量 ||(alpha/r) B A||_F / ||W0||_F の平均 = 0.8444、学習曲線の最終区間の傾き(シード平均)= -0.000160
    Delta_low = b(r=1) - b(r=8) = 0.1680, Delta_high = b(r=8) - b(r=64) = 0.0478
    c = 0.1202, sigma_c = 0.0071, 2 sigma_c = 0.0143
    前提条件 C1(全 rank の全シードで訓練損失が 5% 以上低下): True
    判定関数の結果: 支持



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/012_low_rank_adaptation/output_40_1.png)
    


### 6.11 効率の測定(判定基準を設けない)

閉形式の表(3.1 節)を計算し、実際に数えた値と一致することをアサーションで確認する。
勾配のバイト数は逆伝播後の`.grad`、AdamW の状態のバイト数は 1 回の`step()`の後に optimizer が
保持する $m$・$v$ から数える。CUDA 環境では、1 ステップのピークメモリと 1 ステップあたりの時間も測る。


```python
def count_training_state_bytes(method: str) -> dict:
    model, _ = build_condition_model(method, seed=0, rank=MAIN_RANK)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=1e-4, weight_decay=0.0)
    inputs = train_loss_windows[:2].to(device)
    logits = model(inputs)[:, :-1, :]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), inputs[:, 1:].reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    grad_bytes = sum(
        p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None
    )
    optimizer.step()
    state_bytes = sum(
        t.numel() * t.element_size() for t in [*optimizer._m.values(), *optimizer._v.values()]
    )
    return {
        "trainable": sum(p.numel() for p in trainable),
        "gradient_bytes": grad_bytes,
        "adamw_state_bytes": state_bytes,
    }


_closed_form = {
    "full": TOTAL_PARAMETERS,
    "lora": 2 * NUM_LAYERS * compute_lora_parameter_count(D_MODEL, D_MODEL, MAIN_RANK),
}
efficiency_table = {}
for _method, _p_train in _closed_form.items():
    _counted = count_training_state_bytes(_method)
    _row = {
        "trainable_parameters": _p_train,
        "gradient_bytes": 4 * _p_train,
        "adamw_state_bytes": 8 * _p_train,
        "total_without_activations_bytes": 4 * TOTAL_PARAMETERS + 12 * _p_train,
    }
    assert _counted["trainable"] == _row["trainable_parameters"]
    assert _counted["gradient_bytes"] == _row["gradient_bytes"]
    assert _counted["adamw_state_bytes"] == _row["adamw_state_bytes"]
    efficiency_table[_method] = _row
print("閉形式と実際に数えた値(学習可能パラメータ数・勾配・AdamW の状態のバイト数)の一致: OK")
print(
    f"{'方式':<6} | {'学習可能パラメータ数':>18} | {'勾配 (MB)':>10} | {'AdamW 状態 (MB)':>15} | "
    f"{'活性化を除く合計 (MB)':>20}"
)
for _method, _row in efficiency_table.items():
    print(
        f"{_method:<6} | {_row['trainable_parameters']:>18,} | {_row['gradient_bytes'] / 2**20:>10.3f} | "
        f"{_row['adamw_state_bytes'] / 2**20:>15.3f} | {_row['total_without_activations_bytes'] / 2**20:>20.3f}"
    )


def measure_step_memory_and_time(method: str, num_warmup: int = 3, num_measure: int = 20) -> dict:
    model, _ = build_condition_model(method, seed=0, rank=MAIN_RANK)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=0.0)
    generator = torch.Generator().manual_seed(0)

    def one_step():
        starts = torch.randint(
            0, len(train_ids) - SEQUENCE_LENGTH - 1, (BATCH_SIZE,), generator=generator
        )
        batch = torch.stack([train_ids[s : s + SEQUENCE_LENGTH + 1] for s in starts]).to(device)
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for _ in range(num_warmup):
        one_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(num_measure):
        one_step()
    torch.cuda.synchronize()
    return {
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "seconds_per_step": (time.time() - t0) / num_measure,
    }


if device.type == "cuda":
    measured_efficiency = {}
    for _method in ("full", "lora"):
        measured_efficiency[_method] = measure_step_memory_and_time(_method)
        torch.cuda.empty_cache()
    print(f"{'方式':<6} | {'ピークメモリ (MB)':>16} | {'1 ステップの時間 (ms)':>20}")
    for _method, _row in measured_efficiency.items():
        print(
            f"{_method:<6} | {_row['peak_memory_bytes'] / 2**20:>16.1f} | {_row['seconds_per_step'] * 1000:>20.1f}"
        )
else:
    measured_efficiency = None
    print(f"CUDA 環境ではない({device})ため、ピークメモリ・1 ステップの時間の実測をスキップした。")
```

    閉形式と実際に数えた値(学習可能パラメータ数・勾配・AdamW の状態のバイト数)の一致: OK
    方式     |         学習可能パラメータ数 |    勾配 (MB) |   AdamW 状態 (MB) |        活性化を除く合計 (MB)
    full   |          5,246,208 |     20.013 |          40.025 |               80.051
    lora   |             32,768 |      0.125 |           0.250 |               20.388
    方式     |      ピークメモリ (MB) |       1 ステップの時間 (ms)
    full   |           2273.0 |                161.8
    lora   |           1995.8 |                135.1


### 6.12 出力例(定性的な観察)

同じプロンプト・同じシードで、ベースモデル・全パラメータ微調整・LoRA($r=8$)から top-p
サンプリング($p = 0.9$、temperature 1.0)で生成する。サンプリングの乱数は CPU の
`torch.Generator`で固定し、デバイスによらず同じ乱数列を使う。


```python
SAMPLE_PROMPT = "ROMEO:\n"
SAMPLE_NEW_TOKENS = 80
SAMPLE_TOP_P = 0.9
SAMPLE_SEED = 0


@torch.no_grad()
def sample_top_p(model: nn.Module, prompt: str) -> str:
    model.eval()
    token_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generator = torch.Generator().manual_seed(SAMPLE_SEED)
    for _ in range(SAMPLE_NEW_TOKENS):
        logits = model(token_ids[:, -SEQUENCE_LENGTH:])[:, -1, :].float().cpu()
        probs = torch.softmax(top_p_filter(logits, SAMPLE_TOP_P), dim=-1)
        next_token = torch.multinomial(probs, num_samples=1, generator=generator).to(device)
        token_ids = torch.cat([token_ids, next_token], dim=1)
    return tokenizer.decode(token_ids[0].tolist())


for _label, _model in (
    ("ベースモデル", base_model),
    ("全パラメータ微調整(seed 0)", kept_models["full"]["model"]),
    (f"LoRA r={MAIN_RANK}(seed 0)", kept_models["lora"]["model"]),
):
    print(f"===== {_label} =====")
    print(sample_top_p(_model, SAMPLE_PROMPT))
    print()
```

    ===== ベースモデル =====
    ROMEO:
    QOMORESZ-Co vs. DaSGZ UTON
    Clayer Chiefs Joint Marines
    
    Truce Kommendalley (2008–2011)
    
    Miss Press
    Garley Fox (Milan), historian, director of Sputtical PBS Conference
    Government Tow Restes
    
    U.S. administration
    Group Order (2011–2014)
    Ukrainian
    
    ===== 全パラメータ微調整(seed 0) =====
    ROMEO:
    'Tis he would; my judge for what should do;
    But not ask how no child 'baith, I know'st,
    Your households to slew his fought!
    
    LEONTES:
    I am your brother toil; I pray it,
    At is good I; look on him;
    This time is like mad on m
    
    ===== LoRA r=8(seed 0) =====
    ROMEO:
    'O:?
    It, I now what should have lport whate like I no child 'IAEM:
    UpruiCrect "The First sources'd afterce if she did a rural men,
    Cropry upon Hehyce-bliffen,’d jow Rome;
    Tadruply nothing actually", and was our textm
    


### 6.13 判定結果一覧

前提条件が 1 つでも不成立の実験は、判定関数の結果に関わらず「前提不成立」とする(判定不能とは
区別する)。判定関数の結果は参考として別欄に残す。


```python
_experiment_preconditions = {"A": ["A1"], "B": ["B1"], "C": ["C1"]}


def final_verdict(name: str, computed: str) -> str:
    if not all(precondition_status.get(n) for n in _experiment_preconditions[name]):
        return "前提不成立"
    return computed


_verdicts = [
    ("A", f"rho={rho:.4f}, sigma_rho={sigma_rho:.4f}, delta={DELTA_A}", verdict_A),
    ("B", f"d={d_B:.4f}, sigma_d={sigma_d_B:.4f}", verdict_B),
    ("C", f"c={c_C:.4f}, sigma_c={sigma_c_C:.4f}", verdict_C),
]
print("実験 | 対比量 | 前提条件の成否 | 判定関数の結果(参考) | 最終判定")
for _name, _stat, _computed in _verdicts:
    _pre = ", ".join(
        f"{n}={'成立' if precondition_status.get(n) else '不成立'}"
        for n in _experiment_preconditions[_name]
    )
    print(f"  {_name}: {_stat} | {_pre} | {_computed} | {final_verdict(_name, _computed)}")
print(f"\n前提条件・採用条件の成否一覧: {precondition_status}")
if SMOKE_TEST:
    print(
        "上記はスモークテスト(SMOKE_TEST=True)によるコードの動作確認結果であり、本番実行"
        "(SMOKE_TEST=False、Google Colab T4)後に数値が変わるため、結論としては扱わない。"
    )
```

    実験 | 対比量 | 前提条件の成否 | 判定関数の結果(参考) | 最終判定
      A: rho=0.6848, sigma_rho=0.0041, delta=0.9 | A1=成立 | 反証 | 反証
      B: d=0.0279, sigma_d=0.0059 | B1=成立 | 支持 | 支持
      C: c=0.1202, sigma_c=0.0071 | C1=成立 | 支持 | 支持
    
    前提条件・採用条件の成否一覧: {'calibration_interior_full': True, 'calibration_interior_lora': True, 'calibration_interior_sparse': True, 'A1': True, 'B1': True, 'C1': True}


## 7. 結果・考察 / Results and Discussion

本番実行(Google Colab T4、torch 2.13.0+cu130、`SMOKE_TEST=False`)のセル出力に基づいて記す。
判定は 6.1 節で事前に宣言した基準のみから導く。結果を見た後に立てた解釈は「事後的な解釈」として
節または段落を分けて記し、検証済みの結論としては扱わない。

### 7.1 較正結果

較正はすべて本番と同じ 87 ステップ・シード 0・gradient clipping なしで行った(6.3 節)。

| 方式 | 学習率: 較正用検証集合の最終 bits-per-byte | 採用した学習率 | gradient clipping の閾値 | 本番での発動率(5 シードの範囲) |
|---|---|---|---|---|
| 全パラメータ微調整 | $10^{-3.5}$: 2.3148 / $10^{-3}$: 2.1481 / $10^{-2.5}$: **2.0548** / $10^{-2}$: 2.1382 / $10^{-1.5}$: 2.9292 | $10^{-2.5}$ | 0.9471 | 0.10〜0.13 |
| LoRA($r=8$) | $10^{-2.5}$: 2.7779 / $10^{-2}$: 2.6410 / $10^{-1.5}$: **2.5648** / $10^{-1}$: 3.3538 / $10^{-0.5}$: 3.6256 | $10^{-1.5}$ | 0.2861 | 0.10〜0.20 |
| ランダムマスク疎微調整 | $10^{-2}$: 2.6999 / $10^{-1.5}$: 2.6183 / $10^{-1}$: **2.6004** / $10^{-0.5}$: 2.6967 / $10^{0}$: 3.2072 | $10^{-1}$ | 0.1004 | 0.09〜0.13 |

3 方式とも、採用した水準はグリッドの内点であり(採用条件は成立)、グリッドの拡張は起きなかった。
採用した学習率(全パラメータ微調整 $10^{-2.5}$、LoRA $10^{-1.5}$、疎微調整 $10^{-1}$)は、6.3 節に
記録した予備掃引(ローカルの MPS)で最良だった水準と一致した。gradient clipping の閾値は、勾配ノルムが
学習可能なパラメータのみから計算されるため方式ごとに大きく異なる。本番の学習での発動率は、
おおむね 0.1〜0.2 であった。

### 7.2 前提条件の成否

宣言したすべての前提条件が成立した。7.3〜7.5 節の判定に「前提不成立」の実験はない。

| 前提条件 | 評価に使った量 | 閾値 | 成否 |
|---|---|---|---|
| A1 | $b_0 - \bar{b}_1 = 1.1412$ | $10\,\sigma_{\bar{b}_1} = 10 \times 0.0051 = 0.051$ | 成立 |
| B1 | 最終訓練損失 / 学習開始時の訓練損失(LoRA・疎微調整の全シードの最大値)= 約 0.75 | 0.95 以下 | 成立 |
| C1 | 同じ比(全 rank の全シードの最大値、$r = 1$)= 約 0.82 | 0.95 以下 | 成立 |

学習開始時の訓練損失(ベースモデル、固定の訓練窓)は 6.2466 nats / トークンで、全条件で共通である。

### 7.3 実験 A(全パラメータ微調整との比較)の判定結果

**判定: 反証。**

| 条件 | 評価集合の最終 bits-per-byte(シード 0〜4) | シード平均 | 標準誤差 |
|---|---|---|---|
| $b_0$(ベースモデル) | 3.3928(決定的) | 3.3928 | 0 |
| $b_1$(全パラメータ微調整) | 2.2618・2.2410・2.2610・2.2377・2.2566 | 2.2516 | 0.0051 |
| $b_2$(LoRA、$r=8$) | 2.6077・2.6179・2.6007・2.6163・2.6138 | 2.6113 | 0.0032 |

$$
\rho = \frac{3.3928 - 2.6113}{3.3928 - 2.2516} = 0.685, \qquad \sigma_\rho = 0.004, \qquad
\rho + 2\sigma_\rho = 0.693 < \delta = 0.9
$$

$\rho + 2\sigma_\rho < \delta$ なので、事前に宣言した基準により **反証** となる。診断量として求めた
ブートストラップの 95% 区間は $[0.678, 0.692]$ であり、これも $\delta$ を大きく下回る。

**まとめ**: $T = 87$ ステップの固定の学習予算のもとで、Query・Value への LoRA($r=8$)は、
全パラメータ微調整による bits-per-byte の改善幅の約 7 割を回復したが、9 割には届かなかった。

**診断量の読み取り**:

- **学習曲線の最終区間の傾き $g$**: 全パラメータ微調整 $-3.08 \times 10^{-4}$、LoRA
  $-1.78 \times 10^{-4}$、疎微調整 $-1.45 \times 10^{-4}$(bits-per-byte / ステップ、シード平均)で、
  全方式で負だった。最終ステップの時点で、どの方式もまだ改善していた。ただし、最後の記録区間
  (ステップ 80 → 87)は cosine スケジュールの末尾にあたり、学習率はピークの約 3% から 1% まで
  下がっている。$g$ はこの小さな学習率での値なので、方式間の大小の比較や、ステップ数を増やした場合の
  外挿には使えない。
- **最良値 − 最終値**: 全方式・全シードで 0 だった。記録点の中で最終ステップが常に最良であり、
  評価集合での過学習は観測されなかった。
- **$\lVert \Delta W \rVert_F / \lVert W_0 \rVert_F$(Query・Value の 8 行列)**: LoRA は 0.59〜1.24、
  全パラメータ微調整は 0.16〜0.28 であった。Query・Value の重みの変化は、LoRA のほうが大きい。
- **全パラメータ微調整の $\Delta W$ の上位 8 特異値のエネルギー割合**: 0.29〜0.41 であった。
  256 個の特異値が均等にエネルギーを持つ場合の $8 / 256 \approx 0.031$ に比べれば上位の方向に
  集中しているが、エネルギーの 59〜71% は上位 8 方向の外にある。同じ形のランダム行列での参照値は
  計算していないため、ランダムな更新との比較はここでは行わない。

#### 7.3.1 事後的な解釈(事前宣言した判定とは独立)

**この小節は、結果を見た後に立てた事後的な解釈であり、検証済みの結論ではない。** 7.3 節の判定
(反証)はこの小節の内容によって変わらない。

実験 C の数値(7.5 節)から同じ式で回復率を計算すると、$r = 64$ でも
$(3.3928 - 2.5635) / (3.3928 - 2.2516) \approx 0.73$ にとどまる($r = 1$ で約 0.54、$r = 8$ で
0.685)。$d_{\mathrm{model}} = 256$ に対して rank 64 の制約は緩いので、全パラメータ微調整との差の
大部分は rank の制約ではなく、適応させる対象を Query・Value に限ったことに由来する可能性がある。
全パラメータ微調整は、Query・Value に加えて、Key・出力の射影、埋め込み(出力層と重みを共有)、
順伝播ネットワーク(Feed-Forward Network)、正規化層も更新している。

これは事前に宣言した量ではなく、本トピックでは検証していない仮説である。LoRA の適用対象を広げた
比較(例えば順伝播ネットワークや出力層にも掛ける)によって確かめうる。LoRA を使う後続のトピック
(016 など)で扱える。

3.8 節では、小規模かつドメイン差が大きい設定では LoRA が全パラメータ微調整に届かない可能性がある
(Biderman et al. 2024 [6])と事前に注記した。今回の反証はこの注記と矛盾しない。ただし、
Biderman et al. が扱ったのは大規模なデータでの継続事前学習・指示チューニングであり、本トピックの
設定(小型モデル、87 ステップ)とは規模が大きく異なる。同論文が指摘する「全パラメータ微調整の
更新量は高い rank を持つ」ことと、上記のエネルギー割合の診断量(上位 8 方向の外にエネルギーの
6〜7 割がある)は同じ方向を示しているが、ランダム行列の参照値がないため、この対応も検証済みでは
ない。

### 7.4 実験 B(同じパラメータ数の対照との比較)の判定結果

**判定: 支持。**

| 条件 | 評価集合の最終 bits-per-byte(シード 0〜4) | シード平均 | 標準誤差 |
|---|---|---|---|
| $b_2$(LoRA、$r=8$) | 2.6077・2.6179・2.6007・2.6163・2.6138 | 2.6113 | 0.0032 |
| $b_3$(ランダムマスク疎微調整) | 2.6396・2.6362・2.6353・2.6271・2.6574 | 2.6391 | 0.0050 |

$$
d = \bar{b}_3 - \bar{b}_2 = 0.028, \qquad \sigma_d = 0.006, \qquad 2\sigma_d = 0.012
$$

$d > 2\sigma_d$ なので、事前に宣言した基準により **支持** となる。$T = 87$ ステップの固定の学習予算の
もとで、学習可能パラメータ数を揃えたとき、低ランク構造の更新(LoRA)のほうが、ランダムに選んだ
要素だけを更新する疎な微調整よりも、評価集合の bits-per-byte が低かった。

**診断量の読み取り**:

- **最終訓練損失の差(疎微調整 − LoRA)**: $-0.009$ nats / トークン(疎微調整 4.6563、LoRA 4.6656)。
  シードごとの値は、LoRA が 4.6437〜4.6821(標本標準偏差 0.016)、疎微調整が 4.6505〜4.6676
  (同 0.007)で、2 方式の範囲は重なっている。差の標準誤差は約 0.008 であり、差はシード間の
  ばらつきと同程度である。したがって、訓練データへの当てはまりは 2 方式で同程度であり、評価集合で
  のみ LoRA が良かった。差は最適化のしやすさではなく、汎化の側に現れていると読める。なお、訓練損失は
  nats / トークン、判定に使った量は bits-per-byte であり、単位が異なるので、2 つの差の大きさを
  直接比べることはできない。
- **学習曲線の最終区間の傾き $g$**: LoRA $-1.78 \times 10^{-4}$、疎微調整 $-1.45 \times 10^{-4}$
  (bits-per-byte / ステップ)で、どちらも最終ステップの時点でまだ改善していた。7.3 節と同じ理由
  (最後の区間の学習率がピークの約 1〜3%)により、この値から方式間の収束の度合いの違いは読み取らない。
- **疎微調整の $\lVert \Delta W \rVert_F / \lVert W_0 \rVert_F$**: 1.08〜2.12 であった。学習可能な
  要素は各行列の 6.25% のみなので、少数の要素に大きな更新が集中している。

学習曲線の途中では、2 方式の大小が最終ステップと逆だった区間がある(7.8 節)。

### 7.5 実験 C(rank の飽和)の判定結果

**判定: 支持。**

| rank $r$ | シード数 | $\bar{b}_r$ | 標準誤差 | 次の rank(2 倍)への改善幅 | $\lVert \frac{\alpha}{r} B A \rVert_F / \lVert W_0 \rVert_F$ | 最終区間の傾き $g$ |
|---|---|---|---|---|---|---|
| 1 | 5 | 2.7792 | 0.0030 | 0.062 | 0.998 | $-1.86 \times 10^{-4}$ |
| 2 | 2 | 2.7172 | 0.0053 | 0.055 | 1.029 | $-1.85 \times 10^{-4}$ |
| 4 | 2 | 2.6622 | 0.0054 | 0.051 | 1.036 | $-1.12 \times 10^{-4}$ |
| 8 | 5 | 2.6113 | 0.0032 | 0.025 | 0.964 | $-1.78 \times 10^{-4}$ |
| 16 | 2 | 2.5863 | 0.0025 | 0.019 | 0.910 | $-1.71 \times 10^{-4}$ |
| 32 | 2 | 2.5677 | 0.0009 | 0.004 | 0.869 | $-2.04 \times 10^{-4}$ |
| 64 | 5 | 2.5635 | 0.0014 | - | 0.844 | $-1.60 \times 10^{-4}$ |

$$
\Delta_{\mathrm{low}} = 2.7792 - 2.6113 = 0.168, \qquad
\Delta_{\mathrm{high}} = 2.6113 - 2.5635 = 0.048
$$

$$
c = \Delta_{\mathrm{low}} - \Delta_{\mathrm{high}} = 0.120, \qquad \sigma_c = 0.007, \qquad 2\sigma_c = 0.014
$$

$c > 2\sigma_c$ なので、事前に宣言した基準により **支持** となる。rank を 1 → 8 に上げたときの改善幅は、
8 → 64 に上げたときの改善幅の約 3.5 倍だった。

**診断量と交絡**:

- rank を 2 倍にするごとの改善幅は、0.062 → 0.055 → 0.051 と推移した後、$r = 8$ を境に
  0.025 → 0.019 → 0.004 と小さくなった。中間の rank(2・4・16・32)は 2 シードなので、この推移は
  判定に使わない記述である。$\Delta_{\mathrm{high}}$ も正であり、$r = 8$ 以降も改善は続いている。
  したがって、rank の効果は「頭打ち」ではなく **「逓減」** と書くのが正確である。
- 更新量の比 $\lVert \frac{\alpha}{r} B A \rVert_F / \lVert W_0 \rVert_F$ は、$r = 1$ の約 1.00 から
  $r = 64$ の約 0.84 へ、約 15% 縮んだ。単調ではなく、$r = 4$(約 1.04)を最大として、それより大きい
  rank で縮んでいる。rsLoRA が指摘する $\alpha / r$ による更新量の縮小は観測されたが、その程度は穏やか
  であった。
- rank ごとの最終区間の傾き $g$ は $-1.1 \times 10^{-4}$〜$-2.0 \times 10^{-4}$ で、rank に対する
  一貫した傾向はなかった。ただし 7.3 節と同じ理由(最後の区間の学習率が小さい)により、$g$ は
  「大きい rank ほど収束から遠い」かどうかの強い手がかりにはならない。
- 6.1 節で宣言したとおり、**支持の判定は「低ランクで十分である」ことの検証とはみなさない。**
  更新量の縮小は穏やかだったとはいえ存在しており、この実験だけでは、改善の逓減の原因が低ランク性
  (小さい rank で必要な更新を表現しきれている)なのか、大きい rank での更新量の縮小なのかを
  区別できない。

### 7.6 効率の測定

判定基準を設けない測定である(6.1 節)。

**閉形式(活性化を除く、3.1 節)**:

| 方式 | 学習可能パラメータ数 | 勾配 | AdamW の状態 | 活性化を除く合計 |
|---|---|---|---|---|
| 全パラメータ微調整 | 5,246,208 | 20.0 MB | 40.0 MB | 80.1 MB |
| LoRA($r=8$) | 32,768 | 0.125 MB | 0.250 MB | 20.4 MB |

閉形式の値は、実際のパラメータ・勾配・AdamW の状態から数えた値と一致した(アサーションで確認)。
閉形式による削減は約 60 MB である。

**T4 での実測(バッチサイズ 32、系列長 256、1 ステップ)**:

| 方式 | ピークメモリ | 1 ステップの時間 |
|---|---|---|
| 全パラメータ微調整 | 2273.0 MB | 161.8 ms |
| LoRA($r=8$) | 1995.8 MB | 135.1 ms |

ピークメモリの削減は 277 MB で、閉形式による削減(約 60 MB)を大きく上回った。これは 3.1 節で述べた
仕組みと整合する。LoRA の条件では、重みを凍結した線形層のうち、出力の射影・順伝播ネットワークの
3 つの射影・出力層(埋め込みと重みを共有)は、逆伝播のために入力を保存する必要がなく、活性化の
メモリも一部減る。これらの入力(注意の重み付き和を連結したもの、正規化層の出力、SwiGLU のゲートとの
積、最終正規化層の出力)は、モデルの構造上、ほかの学習可能な層や非線形の演算には保存されない。
一方、Key の射影は凍結していてもこの例に含まれない。Key の射影は Query・Value の射影と同じ入力
テンソル(正規化層の出力)を受け取り、学習可能な LoRA の $A$ がその入力を保存するからである。
入力を共有する層のうち 1 つでも学習可能であれば、その入力の保存は残る。ただし、
本ノートブックでは保存テンソルの内訳を測っていないので、277 MB のうちどれだけがこの仕組みによるかは
分解していない。1 ステップの時間が約 16.5% 短くなったことも、凍結した線形層で重みの勾配の行列積を
省けるという 3.1 節の説明と整合する。

一方、ピークメモリの総量に対する削減率は約 12% にとどまった。学習可能パラメータ数は約 160 分の 1
になったが、LoRA の条件でも重み・勾配・AdamW の状態の合計は閉形式で 20.4 MB にすぎず、約 2 GB の
ピークメモリの大部分は活性化などの、パラメータ以外のメモリが占めている(その内訳は測っていない)。
活性化のメモリが支配的であるという、011 の実験 G で見た構図と整合する。

### 7.7 出力例

同じプロンプト(`ROMEO:`)・同じシードでの top-p サンプリングの生成例(6.12 節)を比べると、次の違いが
見られた。

- **ベースモデル**: 記事の見出しや年号の範囲(「(2008–2011)」など)が並び、英語版 Wikipedia 風の
  文を生成している。
- **全パラメータ微調整**: 話者名(`LEONTES:`)・台詞ごとの改行・古語風の語彙(`'Tis`、`know'st`)
  など、戯曲の形式をおおむね再現している。
- **LoRA($r=8$)**: 戯曲の書き出しを部分的に再現するが、途中で崩れ、Wikipedia 風の断片
  (`The First sources`、`rural men`、`actually`など)が混じる。

この違いは、LoRA が全パラメータ微調整の改善幅の約 7 割しか回復しなかった実験 A の結果と定性的に
整合する。ただし各条件 1 サンプルのみの観察であり、判定の根拠ではない。

### 7.8 可視化

**図: 評価集合の学習曲線と最終 bits-per-byte の散布図(6.8 節)**

- 全パラメータ微調整は、最初の記録点(ステップ 10)の時点で bits-per-byte を約 2.55 まで下げており、
  LoRA・疎微調整(約 2.8〜2.95)より立ち上がりが大きく速い。以降は 3 方式ともゆるやかに下がり続け、
  最終ステップまで改善が続く(7.3 節の $g$ と整合)。
- LoRA と疎微調整の学習曲線は交差している。図から読み取る限り、ステップ 10〜20 では疎微調整の
  ほうが低く、ステップ 30 前後で並び、それ以降は LoRA のほうが低い。学習曲線の途中の値はセル出力に
  数値として印字していないため、交差の位置は図からの読み取りである。
- 散布図では、3 方式の最終値のシード間のばらつきは方式間の差に比べて小さく、方式どうしの範囲は
  重ならない。

**図: rank の掃引と更新量の比(6.10 節)**

- 最終 bits-per-byte は rank とともに単調に下がり、$r = 8$ 以降は傾きが小さくなる(7.5 節の逓減)。
  すべての rank で、全パラメータ微調整の平均(破線)から大きく離れている。
- 更新量の比は $r = 4$ で最大となり、それより大きい rank で縮む(7.5 節)。

**事後的な解釈(学習曲線の交差について)**: 以下は結果を見た後の解釈であり、7.4 節の判定(支持)を
変えるものではない。LoRA と疎微調整の大小が学習の途中で入れ替わっていることから、実験 B の結論は
学習予算 $T$ に依存しうる。事前に宣言した $T = 87$ のもとでは LoRA が良かったが、より少ないステップ数
では逆の結論になりえた。6.1 節で明記したとおり、実験 B の判定は固定の学習予算のもとでの比較である。
LoRA の立ち上がりが遅いことについては、最初のステップで $A$ が更新されないこと(3.3 節)だけでは、
ステップ 10〜20 まで続く差を説明できない。考えられる説明は、更新量が積 $\frac{\alpha}{r} B A$ で
表され、$B = 0$ から始まるため、学習の初期は $\Delta W$ の実効的な変化の速さが小さく、$B$ が 0 から
離れるにつれて大きくなる、というものである。これも検証していない説明である。

### 7.9 全体としての考察

**判定の一覧**(前提条件はすべて成立、7.2 節):

| 実験 | 検証したこと | 判定 |
|---|---|---|
| A | Query・Value への LoRA($r=8$)が全パラメータ微調整の改善幅の 9 割以上を回復する | 反証($\rho = 0.685$) |
| B | 同じパラメータ数なら、低ランク構造の更新がランダムな疎な更新より評価集合で良い | 支持($d = 0.028$) |
| C | rank 1 → 8 の改善幅が 8 → 64 の改善幅より大きい | 支持($c = 0.120$) |

**言えること**(いずれも $T = 87$ ステップの固定の学習予算のもとで):

- Query・Value への LoRA($r=8$、学習可能パラメータは全体の 0.62%)は、全パラメータ微調整の
  改善幅の約 7 割を回復する。9 割には届かない。
- 学習可能パラメータ数を揃えると、低ランク構造の更新はランダムな疎な更新より評価集合の
  bits-per-byte が低い。訓練損失は同程度であり、差は汎化の側に現れている(診断量の読み取り)。
- rank を上げたときの改善効果は、$r = 8$ 付近から逓減する。

**言えないこと**:

- 各方式が収束した状態での比較。全方式が最終ステップの時点でまだ改善しており、判定は固定の学習予算の
  もとでの比較である。
- 「低ランクで十分である」こと。実験 C の支持は、rsLoRA が指摘する更新量の縮小という交絡を分離
  していない。
- 実験 A の差の原因が、適応させる対象を Query・Value に限ったことにある、ということ(7.3.1 節の
  事後的な仮説)。

**今後確かめるべきこと**(事後的に立てた仮説であり、本トピックの結論ではない):

- LoRA の適用対象を Query・Value 以外(順伝播ネットワーク・出力層など)に広げると、実験 A の回復率が
  どこまで上がるか(7.3.1 節)。
- 実験 B の結論が学習予算 $T$ にどう依存するか(7.8 節の学習曲線の交差)。
- rank の効果の逓減のうち、低ランク性によるものと更新量の縮小によるものを分けること(例えば
  rsLoRA の $\alpha / \sqrt{r}$ スケーリングで同じ掃引を行う)。


## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/03_efficient_training/012_low_rank_adaptation.ipynb
