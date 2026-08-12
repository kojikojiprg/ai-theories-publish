---
title: "正規化と活性化の系譜(実装・実験編 2/3)"
---

この記事は後編(実装・実験編 2/3)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/004_normalization_and_activation-practice-1)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/004_normalization_and_activation-practice-3)。

### 6.3 実験 C: 正規化の除去実験(文字レベル言語モデリング)

以下の 4 条件(いずれも活性化関数は GELU、正規化の配置は正規化前置(Pre-Layer Normalization)に固定)を、共通ハイパーパラメータ($d_{\text{model}}=256$、4 層、ヘッド数 4、$d_{\text{ff}}=1024$、系列長 128)、学習率$10^{-2}$で比較する。**この学習率は AdamW を用いる Transformer としてはやや高めであり、これは正規化なし条件を破綻させる十分な負荷を確保するために選んだものである。したがって、以下で観測される正規化なし・平均減算のみの条件の劣化には、「この学習率に耐えられない」という側面が含まれることに留意する** (より小さい学習率でも同様に劣化するかどうかは本ノートブックでは確認していない)。

1. 層正規化(Layer Normalization)
2. RMSNorm
3. 平均減算のみ(分散除算なし、`LayerNormalization(center=True, scale=False)`)
4. 正規化なし(アフィン変換のみ、`LayerNormalization(center=False, scale=False)`)

条件 3・4 は明確に劣化または発散することが期待される **陽性対照(positive control)** であり、測定系(この学習ループ・この規模のモデルで、正規化の欠如を検出できること)の感度を示す役割を持つ。発散した場合は損失が発散した事実そのものを結果として記録し、ハイパーパラメータを条件ごとに変更することはしない。

条件 1(層正規化)と条件 2(RMSNorm)の間に差が出ない場合、それは「差を検出できなかった」ではなく、**この学習率($10^{-2}$)のもとでは平均減算(re-centering)が不要であることを示した** と解釈する。条件 3・4 で明確な劣化が観測されることが、この解釈(測定系に十分な感度がある)を支える根拠になる。

まず、1 run の所要時間を実測し、実験 C・実験 E 合わせて 35 run が CLAUDE.md の目標(70 分以内)に収まる step 数を決定する。


```python
SEEDS = [0, 1, 2, 3, 4]
LEARNING_RATE_LANGUAGE_MODEL = 1e-2
N_TOTAL_LANGUAGE_MODEL_RUNS = (
    35  # 実験 C(4条件x5seed=20)+実験 E(4条件x5seed=20、RMSNorm+GELUの5runは共有)
)


def make_gelu_feed_forward_factory():
    return lambda: FeedForwardNetwork(D_MODEL_COMMON, D_FF_COMMON, activation_fn=gelu_exact)


def build_language_model(normalization_factory, feed_forward_factory):
    return CausalCharacterLevelLanguageModel(
        VOCAB_SIZE,
        D_MODEL_COMMON,
        N_LAYERS_COMMON,
        N_HEADS_COMMON,
        D_FF_COMMON,
        SEQ_LEN_COMMON,
        normalization_factory=normalization_factory,
        feed_forward_factory=feed_forward_factory,
    )


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    # CPU は同期実行のため、明示的な同期操作は存在しない。
```

**較正セルについて**: 1 step あたりの所要時間は、GPU 上の演算が非同期に実行されるため、計測の前後で明示的に同期を取らないと不正確になりうる。加えて、較正の step 数が短すぎると、CUDA カーネルの初回コンパイルや cuDNN のオートチューニングといった **一度きりのウォームアップコスト** が計測全体に均等に按分され、定常状態の速度を大きく過大評価してしまう(実際、50 step だけの較正では 1 step あたり実測 $284.9\,\text{ms}$ と見積もったのに対し、後続の実験 C の実測は 1 step あたり約 $55\,\text{ms}$ しかなく、5 倍以上の過大評価だった)。

そこで較正は 2 段階に分ける。まず`CALIBRATION_WARMUP_STEPS`だけ学習を実行してウォームアップコストをここで払い(この区間は計測しない)、続けて`CALIBRATION_TIMED_STEPS`にわたって改めて計測する。ウォームアップの影響を除いた定常状態の 1 step あたりの時間に、安全係数(2 倍)を掛けたうえで採用 step 数を決める。`TARGET_MINUTES`は CLAUDE.md の定める 70 分の目標値としてある。


```python
CALIBRATION_WARMUP_STEPS = 20  # 計測対象外。CUDA カーネルの初回コンパイル等のコストをここで払う
CALIBRATION_TIMED_STEPS = 150  # 実際に時間を計測する step 数
CALIBRATION_EVALUATE_EVERY = 25

torch.manual_seed(0)
calibration_model = build_language_model(
    lambda d: LayerNormalization(d), make_gelu_feed_forward_factory()
)
calibration_model = calibration_model.to(device)

# ウォームアップ(計測しない): この区間で一度きりのコストを払っておく。
train_character_level_language_model(
    calibration_model,
    train_data,
    evaluation_batches,
    steps=CALIBRATION_WARMUP_STEPS,
    batch_size=BATCH_SIZE_LANGUAGE_MODEL,
    seq_len=SEQ_LEN_COMMON,
    learning_rate=LEARNING_RATE_LANGUAGE_MODEL,
    warmup_steps=5,
    seed=0,
    device=device,
    evaluate_every=CALIBRATION_WARMUP_STEPS,
)

synchronize_device(device)
calibration_start = time.time()
train_character_level_language_model(
    calibration_model,
    train_data,
    evaluation_batches,
    steps=CALIBRATION_TIMED_STEPS,
    batch_size=BATCH_SIZE_LANGUAGE_MODEL,
    seq_len=SEQ_LEN_COMMON,
    learning_rate=LEARNING_RATE_LANGUAGE_MODEL,
    warmup_steps=5,
    seed=1,
    device=device,
    evaluate_every=CALIBRATION_EVALUATE_EVERY,
)
synchronize_device(device)
calibration_elapsed = time.time() - calibration_start
per_step_time_measured = calibration_elapsed / CALIBRATION_TIMED_STEPS

SAFETY_FACTOR = 2.0
per_step_time_with_safety = per_step_time_measured * SAFETY_FACTOR
print(
    f"ウォームアップ {CALIBRATION_WARMUP_STEPS} step(計測対象外)の後、"
    f"{CALIBRATION_TIMED_STEPS} step で計測"
)
print(f"1 step あたりの所要時間(評価込み、実測): {per_step_time_measured * 1000:.1f} ms")
print(f"安全係数({SAFETY_FACTOR}倍)適用後: {per_step_time_with_safety * 1000:.1f} ms")

TARGET_MINUTES = 70.0  # CLAUDE.md の定める目標値
MINIMUM_STEPS_TARGET = 1000

budget_seconds_per_run = TARGET_MINUTES * 60 / N_TOTAL_LANGUAGE_MODEL_RUNS
STEPS_LANGUAGE_MODEL = int(budget_seconds_per_run / per_step_time_with_safety // 50 * 50)
STEPS_LANGUAGE_MODEL = max(200, min(4000, STEPS_LANGUAGE_MODEL))
WARMUP_STEPS_LANGUAGE_MODEL = max(10, STEPS_LANGUAGE_MODEL // 10)

print(f"採用する step 数(実験 C・実験 E で共通): {STEPS_LANGUAGE_MODEL}")
if STEPS_LANGUAGE_MODEL < MINIMUM_STEPS_TARGET:
    print(f"注意: {MINIMUM_STEPS_TARGET} step という目安を下回っている。")
print(f"warmup step 数: {WARMUP_STEPS_LANGUAGE_MODEL}")
total_estimated_minutes = (
    per_step_time_with_safety * STEPS_LANGUAGE_MODEL * N_TOTAL_LANGUAGE_MODEL_RUNS / 60
)
total_estimated_minutes_str = f"{total_estimated_minutes:.1f}"
print(
    f"見積もり合計時間({N_TOTAL_LANGUAGE_MODEL_RUNS} run、安全係数込み): "
    f"{total_estimated_minutes_str} 分"
)
```

    ウォームアップ 20 step(計測対象外)の後、150 step で計測
    1 step あたりの所要時間(評価込み、実測): 61.6 ms
    安全係数(2.0倍)適用後: 123.1 ms
    採用する step 数(実験 C・実験 E で共通): 950
    注意: 1000 step という目安を下回っている。
    warmup step 数: 95
    見積もり合計時間(35 run、安全係数込み): 68.2 分



```python
normalization_conditions = {
    "Layer Normalization": lambda d: LayerNormalization(d),
    "RMSNorm": lambda d: RMSNorm(d),
    "Mean subtraction only": lambda d: LayerNormalization(d, center=True, scale=False),
    "No normalization": lambda d: LayerNormalization(d, center=False, scale=False),
}

exp_c_histories = {name: [] for name in normalization_conditions}
exp_c_diverged = {name: [] for name in normalization_conditions}
exp_c_final_evaluation_loss = {name: [] for name in normalization_conditions}
exp_c_elapsed_by_condition = {}  # 実験 C: 条件ごとの累積経過時間(6.3 節の考察で使う)
exp_c_models = {"Layer Normalization": {}, "RMSNorm": {}}  # 実験 D で使うため学習済みモデルを保持

start_time = time.time()
for name, norm_factory in normalization_conditions.items():
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = build_language_model(norm_factory, make_gelu_feed_forward_factory())
        history, diverged = train_character_level_language_model(
            model,
            train_data,
            evaluation_batches,
            steps=STEPS_LANGUAGE_MODEL,
            batch_size=BATCH_SIZE_LANGUAGE_MODEL,
            seq_len=SEQ_LEN_COMMON,
            learning_rate=LEARNING_RATE_LANGUAGE_MODEL,
            warmup_steps=WARMUP_STEPS_LANGUAGE_MODEL,
            seed=seed,
            device=device,
        )
        exp_c_histories[name].append(history)
        exp_c_diverged[name].append(diverged)
        exp_c_final_evaluation_loss[name].append(history["evaluation_loss"][-1])
        if name in exp_c_models:
            exp_c_models[name][seed] = model.to("cpu")
    elapsed = time.time() - start_time
    exp_c_elapsed_by_condition[name] = elapsed
    print(f"[実験 C] {name:22s} 完了(5 seed)  累積経過時間: {elapsed:.1f}s")

print(f"実験 C 合計時間: {time.time() - start_time:.1f}s")
```

    [実験 C] Layer Normalization    完了(5 seed)  累積経過時間: 271.8s
    [実験 C] RMSNorm                完了(5 seed)  累積経過時間: 533.9s
    [実験 C] Mean subtraction only  完了(5 seed)  累積経過時間: 778.5s
    [実験 C] No normalization       完了(5 seed)  累積経過時間: 1016.4s
    実験 C 合計時間: 1016.4s



```python
fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0))

plot_learning_curves_multi_seed(
    exp_c_histories,
    title="Evaluation loss over training (5 seeds overlaid per condition, log scale)",
    xlabel="Step",
    ylabel="Evaluation loss",
    ax=axes[0],
    log_scale=True,
)

plot_seed_scatter(
    exp_c_final_evaluation_loss,
    title="Final evaluation loss across 5 seeds",
    ylabel="Final evaluation loss",
    ax=axes[1],
)
fig.tight_layout()
plt.show()

print(f"{'条件':22s}  {'平均':>8s}  {'標準偏差':>8s}  {'最小':>8s}  {'最大':>8s}  発散したseed数")
for name, values in exp_c_final_evaluation_loss.items():
    arr = np.array(values)
    n_diverged = sum(exp_c_diverged[name])
    print(
        f"{name:22s}  {arr.mean():8.4f}  {arr.std():8.4f}  {arr.min():8.4f}  "
        f"{arr.max():8.4f}  {n_diverged}/{len(values)}"
    )

print()
print("条件ごとの累積経過時間(6 節冒頭の実行環境についての記述、および 6.3 節末尾の考察で参照):")
previous_elapsed = 0.0
for name, elapsed in exp_c_elapsed_by_condition.items():
    condition_elapsed = elapsed - previous_elapsed
    print(
        f"  {name:22s}: 累積 {elapsed:8.1f}s / この条件のみ {condition_elapsed:8.1f}s "
        f"/ 1 run あたり {condition_elapsed / len(SEEDS):6.1f}s"
    )
    previous_elapsed = elapsed
```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/004_normalization_and_activation/output_50_0.png)
    


    条件                            平均      標準偏差        最小        最大  発散したseed数
    Layer Normalization       1.7159    0.0151    1.7069    1.7459  0/5
    RMSNorm                   1.7190    0.0100    1.7015    1.7277  0/5
    Mean subtraction only     2.4190    0.4652    1.6171    2.8813  0/5
    No normalization          2.2895    0.5466    1.5984    2.9009  0/5
    
    条件ごとの累積経過時間(6 節冒頭の実行環境についての記述、および 6.3 節末尾の考察で参照):
      Layer Normalization   : 累積    271.8s / この条件のみ    271.8s / 1 run あたり   54.4s
      RMSNorm               : 累積    533.9s / この条件のみ    262.1s / 1 run あたり   52.4s
      Mean subtraction only : 累積    778.5s / この条件のみ    244.6s / 1 run あたり   48.9s
      No normalization      : 累積   1016.4s / この条件のみ    237.9s / 1 run あたり   47.6s


#### 実験 C の結果・考察

**較正セルの不具合とその修正**: 当初の較正セル(50 step のみで計測)は 1 step あたり実測 $284.9\,\text{ms}$(評価込み)と見積もっていた。この不具合(短い較正では GPU カーネルの初回コンパイル等の一度きりのウォームアップコストが計測全体に大きな割合を占め、定常状態の速度を大きく過大評価してしまう)は、ウォームアップ区間を計測から分離し、計測対象の step 数を増やす形で修正済みである。修正後の較正は 1 step あたり $61.6\,\text{ms}$ と見積もり、安全係数(2 倍)適用後の $123.1\,\text{ms}$ から、70 分の予算をもとに採用 step 数を **950**(warmup 95 step)と決定した。**以下の実験 C はこの採用 step 数 950 で実行したものである**(較正の実測精度については 7 節末尾を参照)。

4 条件 × 5 seed の最終評価損失は以下の通り(この語彙サイズでのランダム推測の損失は $\ln 65 \approx 4.174$)。

| 条件 | 平均 ± 標準偏差 | [最小, 最大] |
|---|---:|---:|
| 層正規化 | $1.7159 \pm 0.0151$ | $[1.7069,\ 1.7459]$ |
| RMSNorm | $1.7190 \pm 0.0100$ | $[1.7015,\ 1.7277]$ |
| 平均減算のみ | $2.4190 \pm 0.4652$ | $[1.6171,\ 2.8813]$ |
| 正規化なし | $2.2895 \pm 0.5466$ | $[1.5984,\ 2.9009]$ |

**観察できたこと**

1. **判定基準(最小値〜最大値の区間の重なり)に厳密に従う限り、陽性対照(条件 3・4)は「明確に機能した」とは言えない。** 平均減算のみの範囲 $[1.6171, 2.8813]$・正規化なしの範囲 $[1.5984, 2.9009]$ は、層正規化の範囲 $[1.7069, 1.7459]$・RMSNorm の範囲 $[1.7015, 1.7277]$ を完全に包含している。正規化なし条件の最小値 $1.5984$ は、層正規化の 5 seed のどの値よりも小さい(良い)。200 step の実行では陽性対照が判定基準上も明確に機能していたが(本節末尾を参照)、950 step ではこの結論は変わった。
2. **ただし平均で見れば差は依然として明確である。** 平均減算のみ($2.4190$)・正規化なし($2.2895$)は、層正規化・RMSNorm の平均($1.7159$・$1.7190$)より大幅に悪い。区間が重なる原因は、正規化を欠いた条件の **シード間のばらつきが極端に大きいこと** にある。標準偏差は平均減算のみ $0.4652$・正規化なし $0.5466$ であり、正規化条件の標準偏差($0.0151$・$0.0100$)の $30$〜$55$ 倍に達する。
3. **したがって 950 step で観測されたのは「正規化がないと学習が破綻する」ではなく、「正規化がないと学習が不安定になる」という現象である。** シードによっては正規化ありと同等以上の損失に達する一方、別のシードでは大きく劣る。**ただしシード間の分散は 6 節冒頭で事前に定めた判定指標ではないため、これは事後的な観察であり、本ノートブックでは正式な主張としては扱わない。**
4. **方法論上の知見**: 最小値〜最大値による判定基準は、条件間で分散が桁違いに異なる場合には適さない。分散が近い条件どうしでは非重複が意味のある差を示す一方、分散が極端に大きい条件が含まれると、平均に明確な差があっても区間の重なりだけでは検出できない。ただし 6 節冒頭で定めた判定基準そのものは実験前に固定したものであり、本ノートブックでは変更しない。
5. **前方参照**: この「正規化の欠如による学習の不安定化」は、007 で扱う学習の安定化技術の主題そのものである。

**200 step の実行時との比較(参考)**: 以前(採用 step 数 200)の実行では、平均減算のみ・正規化なしの条件はそれぞれ平均損失 $21.3$・$26.0$ まで完全に破綻し、層正規化・RMSNorm の範囲と全く重ならなかった。**同じ実験設定でも採用 step 数によって定性的な挙動が変わりうる** ことを示す実例として記録しておく。**この 200 step の数値は以前の実行時の値であり、本ノートブックの現在の出力(950 step)には対応するものがない。**

**条件ごとの所要時間**:

| 実行順 | 条件 | 累積経過時間 | この条件のみ | 1 run あたり |
|---|---|---:|---:|---:|
| 1 | 層正規化 | $271.8\,\text{s}$ | $271.8\,\text{s}$ | $54.4\,\text{s}$ |
| 2 | RMSNorm | $533.9\,\text{s}$ | $262.1\,\text{s}$ | $52.4\,\text{s}$ |
| 3 | 平均減算のみ | $778.5\,\text{s}$ | $244.6\,\text{s}$ | $48.9\,\text{s}$ |
| 4 | 正規化なし | $1016.4\,\text{s}$ | $237.9\,\text{s}$ | $47.6\,\text{s}$ |

1 run あたりの所要時間は $47.6$〜$54.4\,\text{s}$ の範囲に収まっており、条件間の変動は小さい(最大でも約 $14\%$)。Google Colab の T4 GPU では持続負荷下でも比較的安定した速度が観測された(ただし 3.4 節で述べた通り、本ノートブックは実行時間の優劣を主張する目的の測定を行っていない)。

### 6.4 実験 D: 隠れ状態の統計量(学習不要、実験 C の学習済みモデルを使用)

実験 C の条件 1(層正規化)・条件 2(RMSNorm)の学習済みモデル(5 seed 分)について、検証データ上で (a) 各層の隠れ状態の平均の絶対値と二乗平均平方根(RMS)の比、(b) 層ごとの勾配ノルムを測定する。

**測定箇所についての注記**: ここで言う「各層の隠れ状態」は、各`EncoderBlock`の出力(正規化前置の構成では、次のブロックの正規化層への入力に相当する)を指す。最初のブロックの正規化層への入力(トークン埋め込み + 位置埋め込みの出力そのもの)は測定対象に含まれていない。

(a) の測定の主眼は **RMSNorm 条件の値** にある。層正規化条件は各ブロック内部で平均を明示的に除去する計算を経ているため、その隠れ状態の |mean|/RMS 比が小さいこと自体はほぼ自明であり、これは「平均除去を明示的に行った場合の下限」に相当する **比較のための参照点** にすぎない。議論の実質を担うのは、平均をそろえる操作を一切行っていない RMSNorm 条件の隠れ状態が、学習後にどれだけゼロ平均から外れているか(あるいは自然にゼロ平均に近いままか)という測定値の方である。これが、3.3 節で導出した「RMSNorm が失った平行移動不変性」が実際の学習では実害にならない理由の根拠になる。

(b) は 002(`theories/01_foundations/002_transformer_block.ipynb`実験 2)で導入した、「最終層を基準にした相対勾配」の手法(`compute_gradient_norm_per_layer`として再実装)を学習済みモデルに適用する。


```python
# 各 EncoderBlock の出力(post-block の隠れ状態)を forward hook で集める。
def collect_hidden_states(model, data, n_batches, batch_size, seq_len, device, seed=0):
    model = model.to(device)
    model.eval()
    captured = [[] for _ in range(len(model.blocks))]

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            hidden = output[0]
            captured[layer_idx].append(hidden.detach().reshape(-1, hidden.size(-1)).cpu())

        return hook

    hooks = [block.register_forward_hook(make_hook(i)) for i, block in enumerate(model.blocks)]
    batch_generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _ in range(n_batches):
            x, _ = get_random_batch(data, batch_size, seq_len, generator=batch_generator)
            model(x.to(device))
    for hook in hooks:
        hook.remove()
    return [torch.cat(layer_chunks, dim=0) for layer_chunks in captured]


# 学習済みモデルで、最終層出力の 2 乗和を損失とした 1 回の逆伝播から層ごとの勾配ノルムを測る。
def measure_trained_gradient_norms(model, data, batch_size, seq_len, device, seed):
    model = model.to(device)
    batch_generator = torch.Generator().manual_seed(seed)
    x, _ = get_random_batch(data, batch_size, seq_len, generator=batch_generator)
    model.zero_grad(set_to_none=True)
    model(x.to(device), record_layer_outputs=True)
    loss = model.layer_outputs[-1].pow(2).sum()
    loss.backward()
    raw_norms, relative_norms = compute_gradient_norm_per_layer(model)
    model.zero_grad(set_to_none=True)
    return raw_norms, relative_norms
```


```python
N_BATCHES_DIAGNOSTIC = 10
exp_d_mean_rms_ratio = {}  # {条件: (n_seeds, n_layers) の ndarray}
exp_d_relative_grad = {}

for name in ["Layer Normalization", "RMSNorm"]:
    ratio_per_seed = []
    grad_per_seed = []
    for seed in SEEDS:
        model = exp_c_models[name][seed]
        hidden_states = collect_hidden_states(
            model,
            val_data,
            N_BATCHES_DIAGNOSTIC,
            BATCH_SIZE_LANGUAGE_MODEL,
            SEQ_LEN_COMMON,
            device="cpu",
            seed=seed,
        )
        ratio_per_seed.append(compute_mean_to_rms_ratio(hidden_states))

        _, relative = measure_trained_gradient_norms(
            model, val_data, BATCH_SIZE_LANGUAGE_MODEL, SEQ_LEN_COMMON, device="cpu", seed=seed
        )
        grad_per_seed.append(relative)

    exp_d_mean_rms_ratio[name] = np.array(ratio_per_seed)
    exp_d_relative_grad[name] = np.array(grad_per_seed)

fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0))
plot_bar_by_layer(
    {name: arr.mean(axis=0).tolist() for name, arr in exp_d_mean_rms_ratio.items()},
    title="Mean(|mean(h)|) / RMS(h) per layer (averaged over 5 seeds)",
    ylabel="|mean| / RMS",
    ax=axes[0],
)
plot_bar_by_layer(
    {name: arr.mean(axis=0).tolist() for name, arr in exp_d_relative_grad.items()},
    title="Relative gradient norm per layer (averaged over 5 seeds)",
    ylabel="Gradient norm / final-layer norm",
    ax=axes[1],
)
fig.tight_layout()
plt.show()

for name in ["Layer Normalization", "RMSNorm"]:
    ratios = exp_d_mean_rms_ratio[name]
    mean_str = np.array2string(ratios.mean(axis=0), precision=4)
    print(f"{name:22s} |mean|/RMS  層ごとの平均(5 seed): {mean_str}")
    std_str = np.array2string(ratios.std(axis=0), precision=4)
    print(f"{name:22s} |mean|/RMS  層ごとの標準偏差(5 seed): {std_str}")
for name in ["Layer Normalization", "RMSNorm"]:
    grads = exp_d_relative_grad[name]
    grad_mean_str = np.array2string(grads.mean(axis=0), precision=4)
    print(f"{name:22s} 相対勾配ノルム 層ごとの平均(5 seed): {grad_mean_str}")
```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/004_normalization_and_activation/output_54_0.png)
    


    Layer Normalization    |mean|/RMS  層ごとの平均(5 seed): [0.0122 0.0104 0.01   0.0099]
    Layer Normalization    |mean|/RMS  層ごとの標準偏差(5 seed): [0.0009 0.0007 0.0003 0.0004]
    RMSNorm                |mean|/RMS  層ごとの平均(5 seed): [0.0546 0.0531 0.0532 0.0552]
    RMSNorm                |mean|/RMS  層ごとの標準偏差(5 seed): [0.0065 0.0081 0.0064 0.0066]
    Layer Normalization    相対勾配ノルム 層ごとの平均(5 seed): [10.4961  3.8678  2.0291  1.    ]
    RMSNorm                相対勾配ノルム 層ごとの平均(5 seed): [12.3004  4.1546  2.0666  1.    ]


#### 実験 D の結果・考察

層 1〜4(入力に近い順)の |mean|/RMS 比(5 seed 平均)は、層正規化で $[0.0122,\ 0.0104,\ 0.0100,\ 0.0099]$、RMSNorm で $[0.0546,\ 0.0531,\ 0.0532,\ 0.0552]$ であった。

層正規化条件の値が小さいこと自体は、各ブロック内部で平均を明示的に除去しているため測定するまでもなくほぼ自明であり、これは「平均除去を明示的に行った場合の下限」に相当する **比較のための参照点** にすぎない。議論の実質を担うのは RMSNorm 条件の値である: **平均をそろえる操作を一切行っていない RMSNorm 条件でも、学習後のモデルの隠れ状態は自然に、層正規化の参照値のおよそ 5 倍という範囲に収まった |mean|/RMS 比(最大でも $5.5\%$)にとどまり、ゼロ平均から大きくは外れていなかった。** これが、3.3 節で導出した「RMSNorm が失った平行移動不変性(shift invariance)」が実際の学習では実害にならない理由だと考えられる。

なお、ここでの測定対象は各`EncoderBlock`の出力(正規化前置の構成では、次のブロックの正規化層への入力に相当する)であり、最初のブロックの正規化層への入力(トークン埋め込み + 位置埋め込みの出力そのもの)は測定していない。

層ごとの相対勾配ノルム(最終層を 1 として正規化、5 seed 平均)は、層正規化で $[10.4961,\ 3.8678,\ 2.0291,\ 1.0000]$、RMSNorm で $[12.3004,\ 4.1546,\ 2.0666,\ 1.0000]$ とほぼ同じ形状のパターンを示した(いずれも入力に近い層ほど勾配が大きく、[002](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/002_transformer_block-theory) の実験 2 で見た正規化前置の傾向と整合する)。層 1 の相対勾配ノルムは 200 step の実行時(層正規化で $5.39$)より大きくなっている($10.50$)が、本ノートブックはこの値の大小自体を主張の根拠にしていないため深追いしない。RMSNorm へ切り替えても勾配伝播の基本的な挙動は層正規化と変わらないことが確認できた。

### 6.5 実験 E: 活性化関数の比較(文字レベル言語モデリング)

正規化は RMSNorm に固定し、以下の 4 条件を比較する。

1. ReLU
2. GELU(RMSNorm 固定という条件が実験 C の条件 2 と完全に一致するため、その 5 run をそのまま再利用する)
3. SwiGLU(パラメータ数を揃えた中間次元、$d_{\text{ff}}' = 683$)
4. SwiGLU(中間次元をそのままにした素朴な置換、$d_{\text{ff}} = 1024$、パラメータ数 約 1.5 倍)

条件 4 を含める理由は、条件 3(揃えた条件)だけでは「パラメータを揃えたら差が消えた」としか言えず、逆に条件 4 だけでは gating の寄与とパラメータ数の寄与が交絡してしまうためである。両方を並べることで、この 2 つの要因を切り分けられる。


```python
D_FF_SWIGLU_NAIVE = D_FF_COMMON

activation_condition_factories = {
    "ReLU": lambda: FeedForwardNetwork(D_MODEL_COMMON, D_FF_COMMON),
    "GELU": make_gelu_feed_forward_factory(),
    "SwiGLU (matched)": lambda: SwiGLUFeedForwardNetwork(D_MODEL_COMMON, D_FF_SWIGLU_MATCHED),
    "SwiGLU (naive)": lambda: SwiGLUFeedForwardNetwork(D_MODEL_COMMON, D_FF_SWIGLU_NAIVE),
}

print(f"{'条件':18s}  {'中間次元':>8s}  {'モデル全体のパラメータ数':>14s}")
for name, factory in activation_condition_factories.items():
    m = build_language_model(lambda d: RMSNorm(d), factory)
    n_params = sum(p.numel() for p in m.parameters())
    d_ff_used = (
        D_FF_SWIGLU_MATCHED
        if name == "SwiGLU (matched)"
        else D_FF_SWIGLU_NAIVE
        if "SwiGLU" in name
        else D_FF_COMMON
    )
    print(f"{name:18s}  {d_ff_used:8d}  {n_params:14,d}")
```

    条件                      中間次元    モデル全体のパラメータ数
    ReLU                    1024       3,219,265
    GELU                    1024       3,219,265
    SwiGLU (matched)         683       3,215,169
    SwiGLU (naive)          1024       4,262,721



```python
exp_e_histories = {"GELU": exp_c_histories["RMSNorm"]}
exp_e_diverged = {"GELU": exp_c_diverged["RMSNorm"]}
exp_e_final_evaluation_loss = {"GELU": exp_c_final_evaluation_loss["RMSNorm"]}
exp_e_models = {"GELU": exp_c_models["RMSNorm"], "ReLU": {}}
print("GELU 条件(RMSNorm + GELU)は実験 C の 5 run をそのまま再利用する。")

start_time = time.time()
for name, feed_forward_factory in activation_condition_factories.items():
    if name == "GELU":
        continue
    exp_e_histories[name] = []
    exp_e_diverged[name] = []
    exp_e_final_evaluation_loss[name] = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = build_language_model(lambda d: RMSNorm(d), feed_forward_factory)
        history, diverged = train_character_level_language_model(
            model,
            train_data,
            evaluation_batches,
            steps=STEPS_LANGUAGE_MODEL,
            batch_size=BATCH_SIZE_LANGUAGE_MODEL,
            seq_len=SEQ_LEN_COMMON,
            learning_rate=LEARNING_RATE_LANGUAGE_MODEL,
            warmup_steps=WARMUP_STEPS_LANGUAGE_MODEL,
            seed=seed,
            device=device,
        )
        exp_e_histories[name].append(history)
        exp_e_diverged[name].append(diverged)
        exp_e_final_evaluation_loss[name].append(history["evaluation_loss"][-1])
        if name == "ReLU":
            exp_e_models[name][seed] = model.to("cpu")
    elapsed = time.time() - start_time
    print(f"[実験 E] {name:18s} 完了(5 seed)  経過時間: {elapsed:.1f}s")

print(f"実験 E 追加所要時間(GELU の 5 run を除く 15 run): {time.time() - start_time:.1f}s")
```

    GELU 条件(RMSNorm + GELU)は実験 C の 5 run をそのまま再利用する。
    [実験 E] ReLU               完了(5 seed)  経過時間: 218.6s
    [実験 E] SwiGLU (matched)   完了(5 seed)  経過時間: 469.9s
    [実験 E] SwiGLU (naive)     完了(5 seed)  経過時間: 768.2s
    実験 E 追加所要時間(GELU の 5 run を除く 15 run): 768.2s



```python
fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0))

plot_learning_curves_multi_seed(
    exp_e_histories,
    title="Evaluation loss over training (5 seeds overlaid per condition, log scale)",
    xlabel="Step",
    ylabel="Evaluation loss",
    ax=axes[0],
    log_scale=True,
)
plot_seed_scatter(
    exp_e_final_evaluation_loss,
    title="Final evaluation loss across 5 seeds",
    ylabel="Final evaluation loss",
    ax=axes[1],
)
fig.tight_layout()
plt.show()

print(f"{'条件':18s}  {'平均':>8s}  {'標準偏差':>8s}  {'最小':>8s}  {'最大':>8s}")
for name, values in exp_e_final_evaluation_loss.items():
    arr = np.array(values)
    print(f"{name:18s}  {arr.mean():8.4f}  {arr.std():8.4f}  {arr.min():8.4f}  {arr.max():8.4f}")
```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/004_normalization_and_activation/output_59_0.png)
    


    条件                        平均      標準偏差        最小        最大
    GELU                  1.7190    0.0100    1.7015    1.7277
    ReLU                  1.7314    0.0178    1.7021    1.7557
    SwiGLU (matched)      1.7123    0.0380    1.6606    1.7631
    SwiGLU (naive)        1.6883    0.0366    1.6534    1.7471


#### 実験 E の結果・考察

最終評価損失(5 seed、GELU 条件は実験 C の RMSNorm 条件を再利用):

| 条件 | 平均 ± 標準偏差 | [最小, 最大] |
|---|---:|---:|
| GELU | $1.7190 \pm 0.0100$ | $[1.7015,\ 1.7277]$ |
| ReLU | $1.7314 \pm 0.0178$ | $[1.7021,\ 1.7557]$ |
| SwiGLU(パラメータ数を揃えた条件) | $1.7123 \pm 0.0380$ | $[1.6606,\ 1.7631]$ |
| SwiGLU(素朴な置換) | $1.6883 \pm 0.0366$ | $[1.6534,\ 1.7471]$ |

**6 通りの条件ペアすべてで最小値〜最大値の区間が重なっている。** 200 step の実行で観測された ReLU vs SwiGLU(素朴な置換)の 1 ペアの非重複は、950 step では再現しなかった。

**多重比較についての注記(結果を踏まえた振り返り)**: 200 step の考察では、4 条件・6 通りのペアを同時に比較しているため、そのうち 1 ペアが偶然に非重複となる確率は 1 ペアだけを比較する場合より高い、という留保を付けたうえで ReLU vs SwiGLU(素朴な置換)の非重複を記録していた。950 step の結果はこの非重複が再現しなかったことを示しており、**6 通りの同時比較で見つかった単独の非重複に強い解釈を与えなかった判断が妥当だった** ことが確認された。この多重比較についての注記自体は、複数条件を同時比較する実験一般に有効な留意点として残す。

結論は「この規模・この採用 step 数(950)では、4 つの活性化関数の間に評価損失の差を検出できなかった」となる。これは Narang et al. [7] が報告した「小規模な比較実験ではアーキテクチャ変更の優劣が既存の指標に表れないことがある」という観察と整合する。

**プラトー仮説の検証結果(棄却)**: 200 step の実行では ReLU の標準偏差($0.0059$)が他 3 条件($0.095$〜$0.096$)より際立って小さく、「ReLU が採用 step 数の時点で既にプラトー(頭打ち)に達しており、他条件はまだ下降中でシード間のばらつきが大きい」という仮説を立てた。ローカル環境(MPS)での暫定的な再現実行では、ReLU の標準偏差が学習後半を通じて拡大しない一方、他条件は終盤にかけて標準偏差が拡大するという、仮説を支持する所見が得られていた。**しかし 950 step の結果では、ReLU の標準偏差($0.0178$)はもはや外れ値ではなく、むしろ GELU($0.0100$)より大きい。** 「ReLU の標準偏差だけが他条件より 1 桁小さい」という 200 step 特有のパターンは消えており、**プラトー仮説は 950 step の結果によって支持されなかった。** 仮説を立てて検証し、より長い学習では支持されなかったという経緯そのものは、200 step の結果を過大解釈しなかった根拠として記録しておく(ローカル再現実行の詳細な数値は 200 step 特有の現象についての補足だったため、ここでは割愛する)。

実験 F との関係も整理し直す。200 step の考察では、ReLU の標準偏差の小ささを、実験 F で観測した ReLU の常に負のユニット比率の高さと、プラトー仮説を介して整合する仮説として結びつけていた。950 step ではプラトー仮説自体が棄却され、かつ実験 F で観測される常に負のユニット比率も大きく減少・変化している(6.6 節)。**したがって、標準偏差・プラトー・常に負のユニット比率を結びつけるこの解釈は撤回する。** 実験 F で機構として確認されている「ReLU は常に負のユニットの勾配を厳密に遮断する」という事実自体は本節の結論に影響しないが、それが実験 E の評価損失のばらつきに表れているという主張はしない。

### 6.6 実験 F: 常に負のユニット(always-negative unit)の割合と、それが受け取る勾配(実験 E の学習済みモデルを使用)

実験 E の条件 1(ReLU)・条件 2(GELU)の学習済みモデル(5 seed 分)について、順伝播ネットワークの中間層(`feed_forward.linear1`の出力、活性化関数を通す前の値)を検証データ全体にわたって集め、`compute_always_negative_unit_ratio()`で「一度も正にならなかったユニット」の割合を層ごとに集計する。

**この比率が意味することは活性化関数によって異なる。** ReLU は負領域で導関数が恒等的に 0 になるため、常に負のユニットは実際に勾配を受け取れない(いわゆる「死んだユニット(dead unit)」)。GELU は負領域でも導関数が非零であるため、同じ「常に負」という条件を満たしていても勾配は伝わり続ける。したがって、この比率の値そのものを ReLU と GELU の間で比較しても、それだけでは「どちらがより深刻か」を意味しない。

そこで、比率という代理指標に加えて、**実際に勾配が伝わっているかどうかを直接測定する**。学習済みモデルに検証バッチを 1 つ通して逆伝播し、`feed_forward.linear1`の重み勾配を、常に負のユニットに対応する行とそれ以外の行に分けて、それぞれのノルムを`compute_gradient_norm_by_unit_group()`で計算する。ReLU 条件では常に負のユニットに対応する行の勾配ノルムがほぼ厳密に 0 になることが期待される。


```python
# feed_forward.linear1 の出力(活性化前の値)を forward hook で集める。
def collect_feed_forward_preactivations(
    model, data, n_batches, batch_size, seq_len, device, seed=0
):
    model = model.to(device)
    model.eval()
    captured = [[] for _ in range(len(model.blocks))]

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            captured[layer_idx].append(output.detach().reshape(-1, output.size(-1)).cpu())

        return hook

    hooks = [
        block.feed_forward.linear1.register_forward_hook(make_hook(i))
        for i, block in enumerate(model.blocks)
    ]
    batch_generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _ in range(n_batches):
            x, _ = get_random_batch(data, batch_size, seq_len, generator=batch_generator)
            model(x.to(device))
    for hook in hooks:
        hook.remove()
    return [torch.cat(layer_chunks, dim=0) for layer_chunks in captured]


# 実際の言語モデリング損失を 1 バッチ分逆伝播し、常に負のユニットに対応する行の勾配ノルムを
# それ以外の行と比較する。
def measure_gradient_norm_by_unit_group_all_layers(
    model, always_negative_masks, data, batch_size, seq_len, device, seed
):
    model = model.to(device)
    batch_generator = torch.Generator().manual_seed(seed)
    inputs, targets = get_random_batch(data, batch_size, seq_len, generator=batch_generator)
    inputs, targets = inputs.to(device), targets.to(device)
    model.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    loss.backward()

    always_negative_norms, other_norms = [], []
    for block, mask in zip(model.blocks, always_negative_masks, strict=True):
        weight_grad = block.feed_forward.linear1.weight.grad
        always_negative_norm, other_norm = compute_gradient_norm_by_unit_group(weight_grad, mask)
        always_negative_norms.append(always_negative_norm)
        other_norms.append(other_norm)
    model.zero_grad(set_to_none=True)
    return always_negative_norms, other_norms
```


```python
exp_f_always_negative_ratio = {}
exp_f_sample_preactivations = {}
exp_f_gradient_by_group = {}

for name in ["ReLU", "GELU"]:
    ratio_per_seed = []
    always_negative_norm_per_seed = []
    other_norm_per_seed = []
    for seed in SEEDS:
        model = exp_e_models[name][seed]
        preactivations = collect_feed_forward_preactivations(
            model,
            val_data,
            N_BATCHES_DIAGNOSTIC,
            BATCH_SIZE_LANGUAGE_MODEL,
            SEQ_LEN_COMMON,
            device="cpu",
            seed=seed,
        )
        ratio_per_seed.append(compute_always_negative_unit_ratio(preactivations))
        if seed == SEEDS[0]:
            exp_f_sample_preactivations[name] = preactivations

        always_negative_masks = [pre.max(dim=0).values <= 0.0 for pre in preactivations]
        always_negative_norms, other_norms = measure_gradient_norm_by_unit_group_all_layers(
            model,
            always_negative_masks,
            val_data,
            BATCH_SIZE_LANGUAGE_MODEL,
            SEQ_LEN_COMMON,
            device="cpu",
            seed=seed,
        )
        always_negative_norm_per_seed.append(always_negative_norms)
        other_norm_per_seed.append(other_norms)

    exp_f_always_negative_ratio[name] = np.array(ratio_per_seed)
    exp_f_gradient_by_group[name] = {
        "always_negative": np.array(always_negative_norm_per_seed),
        "other": np.array(other_norm_per_seed),
    }

fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.0))
plot_bar_by_layer(
    {name: arr.mean(axis=0).tolist() for name, arr in exp_f_always_negative_ratio.items()},
    title="Always-negative unit ratio per layer (averaged over 5 seeds)",
    ylabel="Fraction of units never positive on val set",
    ax=axes[0],
)

bins = np.linspace(-8, 8, 81)
for name in ["ReLU", "GELU"]:
    values = exp_f_sample_preactivations[name][0].flatten().numpy()
    axes[1].hist(values, bins=bins, alpha=0.5, density=True, label=name)
axes[1].axvline(0.0, color="gray", linestyle="--", linewidth=1.0)
axes[1].set_title("Pre-activation distribution at layer 1 (seed=0)")
axes[1].set_xlabel("pre-activation value")
axes[1].set_ylabel("density")
axes[1].legend()

gradient_by_group_for_plot = {
    f"{name} (always-negative units)": data["always_negative"].mean(axis=0).tolist()
    for name, data in exp_f_gradient_by_group.items()
} | {
    f"{name} (other units)": data["other"].mean(axis=0).tolist()
    for name, data in exp_f_gradient_by_group.items()
}
plot_bar_by_layer(
    gradient_by_group_for_plot,
    title="Weight gradient norm by unit group (averaged over 5 seeds)",
    ylabel="||linear1.weight.grad|| for the unit group",
    ax=axes[2],
)
fig.tight_layout()
plt.show()

for name in ["ReLU", "GELU"]:
    ratios = exp_f_always_negative_ratio[name]
    mean_str = np.array2string(ratios.mean(axis=0), precision=4)
    print(f"{name:6s} always-negative unit ratio 層ごとの平均(5 seed): {mean_str}")
    std_str = np.array2string(ratios.std(axis=0), precision=4)
    print(f"{name:6s} always-negative unit ratio 層ごとの標準偏差(5 seed): {std_str}")
for name in ["ReLU", "GELU"]:
    an_norms = exp_f_gradient_by_group[name]["always_negative"]
    other_norms = exp_f_gradient_by_group[name]["other"]
    an_str = np.array2string(an_norms.mean(axis=0), precision=6)
    other_str = np.array2string(other_norms.mean(axis=0), precision=4)
    print(f"{name:6s} 常に負のユニットの勾配ノルム 層ごとの平均(5 seed): {an_str}")
    print(f"{name:6s} それ以外のユニットの勾配ノルム 層ごとの平均(5 seed): {other_str}")
```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/004_normalization_and_activation/output_63_0.png)
    


    ReLU   always-negative unit ratio 層ごとの平均(5 seed): [0.     0.     0.0148 0.0049]
    ReLU   always-negative unit ratio 層ごとの標準偏差(5 seed): [0.     0.     0.0014 0.0026]
    GELU   always-negative unit ratio 層ごとの平均(5 seed): [0.     0.     0.0027 0.001 ]
    GELU   always-negative unit ratio 層ごとの標準偏差(5 seed): [0.     0.     0.0019 0.0011]
    ReLU   常に負のユニットの勾配ノルム 層ごとの平均(5 seed): [0. 0. 0. 0.]
    ReLU   それ以外のユニットの勾配ノルム 層ごとの平均(5 seed): [0.0701 0.0269 0.0194 0.02  ]
    GELU   常に負のユニットの勾配ノルム 層ごとの平均(5 seed): [0.000000e+00 0.000000e+00 2.162407e-05 8.844595e-06]
    GELU   それ以外のユニットの勾配ノルム 層ごとの平均(5 seed): [0.0734 0.0332 0.0228 0.023 ]


#### 実験 F の結果・考察

常に負のユニット(always-negative unit)の割合(層 1〜4、5 seed 平均)は、ReLU で $[0,\ 0,\ 0.0148,\ 0.0049]$、GELU で $[0,\ 0,\ 0.0027,\ 0.0010]$ であった。**200 step の実行では層 4 で ReLU $15.16\%$・GELU $6.33\%$ に達し、入力から遠い層ほど割合が増える傾向を示していたが、950 step ではいずれも 1.5% 以下まで減少し(層 4 で ReLU $0.49\%$(200 step 時点の約 30 分の 1)・GELU $0.10\%$(同じく約 60 分の 1))、層 4 の割合が層 3 より低くなる(両条件とも)など、深さに対する単調な増加傾向も消えている。** 常に負のユニットは、学習が十分に進んだこの設定では **学習初期に生じる一時的な現象という側面が強く**、ReLU の恒常的な病理として提示するのは適切でない。

**一方、決定的な測定である勾配ノルムの結論は変わらない。** 実際の言語モデリング損失を 1 バッチ逆伝播した結果、

- **ReLU**: 常に負のユニットの勾配ノルムは層 1〜4 すべてで **厳密に $0$**($[0,\ 0,\ 0,\ 0]$)。それ以外のユニットの勾配ノルムは $[0.0701,\ 0.0269,\ 0.0194,\ 0.0200]$ と非零。
- **GELU**: 常に負のユニットの勾配ノルムは $[0,\ 0,\ 2.16\times 10^{-5},\ 8.84\times 10^{-6}]$ と、桁は小さいものの **明確に非零**(層 1・2 で $0$ なのは、その層に常に負のユニットが 1 つもなかったため)。それ以外のユニットの勾配ノルムは $[0.0734,\ 0.0332,\ 0.0228,\ 0.0230]$。

これは、代理指標(比率)の比較を、直接の検証に置き換えた結果である。**ReLU と GELU はいずれも常に負のユニットを一定割合生むが、そのユニットが勾配を受け取れるかどうかが決定的に異なる、という主張 3 の機構的な核は保たれている。** ReLU では常に負のユニットの勾配が厳密に $0$ になり、そのユニット自身の重み行を通した勾配経路からは回復できない。GELU では負領域でも導関数が非零であるため、常に負のユニットであっても非零の勾配を受け取り続け、この経路は閉じない。**ただし、実際に影響を受けるユニットの割合は 1.5% 以下にとどまることから、機構としては明確に存在するが、この設定での実質的な影響はごく限定的であると評価を調整する。**

200 step の考察では、「他層の学習によって当該層への入力分布が変化すれば、いずれにせよ『常に負』という状態自体が変わりうる」という留保を付けていた。**200 step から 950 step にかけて常に負のユニットの割合が約 30 分の 1 に減ったことは、この復活が実際に起きていることの傍証と言える。** ただし 200 step の実行と 950 step の実行は採用 step 数が異なる別々の学習実行であり、同一の学習を 2 時点で観測したものではないため、この解釈は断定しない。

**この機構の差は、実験 E の評価損失には検出されなかった**(6.5 節、950 step では 6 通りの条件ペアすべてで区間が重なった)。常に負のユニットの割合がごく一部(最大でも層 3 の $1.48\%$)にとどまることを踏まえると、これは自然な結果である。「機構は確かに存在するが、この規模・この指標にはほとんど転写されない」という 6.5 節の結論と整合する。

### 6.7 実験 G: 乗法的相互作用の合成タスクと陰性対照(negative control)タスク(小規模回帰)

実験 E は文字レベル言語モデリングという 1 つの最終指標に集約されるため、SwiGLU のゲート機構そのものが寄与しているかどうかを直接には切り分けられない。そこで、gating が有利になるように意図的に設計した小規模な回帰タスクで、機構の存在を最終指標から独立に確認する。

**乗法的相互作用タスク**: 入力$x \in \mathbb{R}^{d}$($d = 32$、標準正規分布からサンプリング)に対し、固定したランダムベクトル$w_{1j}, w_{2j} \in \mathbb{R}^d$($j = 1, \dots, d$)を使って

$$
y_j = (w_{1j}^\top x)(w_{2j}^\top x), \qquad j = 1, \dots, d
$$

を目標とする回帰タスク(目標は成分ごとに標準偏差で正規化する)。**この目標関数は、SwiGLU の関数形$(\mathrm{Swish}(xW) \odot xV)W_2$にほぼそのまま含まれる形をしている**(ゲート側とバリュー側の 2 本の線形射影を要素ごとに掛け合わせるという構造が、目標そのものの構造と一致する)。したがって、このタスク単体での SwiGLU の優位は、「SwiGLU の仮説クラスに元々含まれる関数を、SwiGLU がうまく表現できた」という以上の主張をするには弱い。

**陰性対照タスク**: 乗法的相互作用を含まない対照として、固定したランダムな射影$w_j \in \mathbb{R}^d$を使った

$$
y_j = \tanh(w_j^\top x), \qquad j = 1, \dots, d
$$

を用意する(こちらも目標は成分ごとに標準偏差で正規化する)。単一の線形射影に要素ごとの非線形性をかけただけであり、乗法的相互作用を必要としない。入力次元・データ数・step 数・シード数・中間次元の水準は乗法的相互作用タスクと完全に揃える。**乗法的相互作用タスクで SwiGLU が優位を示し、陰性対照タスクで優位を示さない場合に限り、「gating は乗法的相互作用を必要とするタスクでのみ低いパラメータコストで優位を示す」と主張できる。** 陰性対照タスクでも SwiGLU が優位を示した場合は、その事実をそのまま記録し、主張を「乗法的相互作用に限定した優位」とは述べない。

**条件**: 標準の順伝播ネットワーク(GELU)と`SwiGLUFeedForwardNetwork`(パラメータ数を揃えた条件)を、中間次元$d_{\text{ff}} \in \{8, 16, 32, 64\}$(SwiGLU 側は 3.7 節の$2/3$倍を丸めた値)の 4 水準、2 つのタスクで比較する。各条件・各水準・各タスクで 10 シード、合計$2 \times 4 \times 2 \times 10 = 160$ run。

**評価指標**: 検証データ上の平均二乗誤差を目標の分散で正規化した値(正規化 MSE。1.0 でモデルが目標の分散を全く説明できていない水準、0.0 で完全に説明できている水準に対応する)。


```python
D_TASK = 32
N_TRAIN_TASK, N_VALIDATION_TASK = 2000, 500
STEPS_TASK, LEARNING_RATE_TASK = 800, 3e-3
D_FF_LEVELS_TASK = [8, 16, 32, 64]
N_SEEDS_TASK = 10


def make_multiplicative_interaction_task(task_seed):
    generator = torch.Generator().manual_seed(task_seed)
    w1 = torch.randn(D_TASK, D_TASK, generator=generator)
    w2 = torch.randn(D_TASK, D_TASK, generator=generator)

    def sample(n, sample_seed):
        sample_generator = torch.Generator().manual_seed(sample_seed)
        x = torch.randn(n, D_TASK, generator=sample_generator)
        y = (x @ w1) * (x @ w2)
        y = y / y.std(dim=0, keepdim=True)
        return x, y

    return sample


def make_negative_control_task(task_seed):
    generator = torch.Generator().manual_seed(task_seed)
    projection = torch.randn(D_TASK, D_TASK, generator=generator)

    def sample(n, sample_seed):
        sample_generator = torch.Generator().manual_seed(sample_seed)
        x = torch.randn(n, D_TASK, generator=sample_generator)
        y = torch.tanh(x @ projection)
        y = y / y.std(dim=0, keepdim=True)
        return x, y

    return sample


tasks = {
    "Multiplicative interaction: y_j = (w_1j^T x)(w_2j^T x)": make_multiplicative_interaction_task,
    "Negative control: y_j = tanh(w_j^T x)": make_negative_control_task,
}


def train_regression(
    model, x_train, y_train, x_validation, y_validation, steps, learning_rate, seed
):
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        validation_mse = loss_fn(model(x_validation), y_validation).item()
        target_variance = y_validation.var().item()
    return validation_mse / target_variance
```


```python
exp_g_normalized_mse = {}  # {task_name: {条件名: {d_ff: [シードごとの正規化MSE]}}}
exp_g_params = {}

start_time = time.time()
for task_name, task_factory in tasks.items():
    task_sampler = task_factory(task_seed=0)
    x_train_task, y_train_task = task_sampler(N_TRAIN_TASK, sample_seed=1)
    x_validation_task, y_validation_task = task_sampler(N_VALIDATION_TASK, sample_seed=2)

    exp_g_normalized_mse[task_name] = {
        "GELU (standard feed-forward)": {d_ff: [] for d_ff in D_FF_LEVELS_TASK},
        "SwiGLU": {d_ff: [] for d_ff in D_FF_LEVELS_TASK},
    }
    exp_g_params[task_name] = {"GELU (standard feed-forward)": {}, "SwiGLU": {}}

    for d_ff in D_FF_LEVELS_TASK:
        d_ff_swiglu = round((2 / 3) * d_ff)
        for seed in range(N_SEEDS_TASK):
            torch.manual_seed(seed)
            model_gelu = FeedForwardNetwork(D_TASK, d_ff, activation_fn=gelu_exact)
            error_gelu = train_regression(
                model_gelu,
                x_train_task,
                y_train_task,
                x_validation_task,
                y_validation_task,
                STEPS_TASK,
                LEARNING_RATE_TASK,
                seed,
            )
            exp_g_normalized_mse[task_name]["GELU (standard feed-forward)"][d_ff].append(error_gelu)

            torch.manual_seed(seed)
            model_swiglu = SwiGLUFeedForwardNetwork(D_TASK, d_ff_swiglu)
            error_swiglu = train_regression(
                model_swiglu,
                x_train_task,
                y_train_task,
                x_validation_task,
                y_validation_task,
                STEPS_TASK,
                LEARNING_RATE_TASK,
                seed,
            )
            exp_g_normalized_mse[task_name]["SwiGLU"][d_ff].append(error_swiglu)

        exp_g_params[task_name]["GELU (standard feed-forward)"][d_ff] = sum(
            p.numel() for p in model_gelu.parameters()
        )
        exp_g_params[task_name]["SwiGLU"][d_ff] = sum(p.numel() for p in model_swiglu.parameters())

n_runs_exp_g = 2 * len(tasks) * len(D_FF_LEVELS_TASK) * N_SEEDS_TASK
print(f"実験 G 合計時間({n_runs_exp_g} run): {time.time() - start_time:.1f}s")
```

    実験 G 合計時間(160 run): 260.4s



```python
fig, axes = plt.subplots(1, len(tasks), figsize=(8.0 * len(tasks), 5.0), sharey=True)
rng = np.random.default_rng(0)

for ax, (task_name, results) in zip(axes, exp_g_normalized_mse.items(), strict=True):
    for name, color in [("GELU (standard feed-forward)", "tab:blue"), ("SwiGLU", "tab:orange")]:
        xs, ys, means = [], [], []
        for d_ff in D_FF_LEVELS_TASK:
            values = results[name][d_ff]
            jitter = rng.uniform(-1.5, 1.5, size=len(values))
            xs.extend([d_ff] * len(values) + jitter)
            ys.extend(values)
            means.append(np.mean(values))
        ax.scatter(xs, ys, alpha=0.5, s=18, color=color, label=f"{name} (seeds)")
        ax.plot(
            D_FF_LEVELS_TASK, means, color=color, marker="o", linewidth=2.0, label=f"{name} (mean)"
        )
    ax.set_xlabel("Intermediate dimension (standard feed-forward: d_ff, SwiGLU: matched d_ff')")
    ax.set_xscale("log", base=2)
    ax.set_xticks(D_FF_LEVELS_TASK)
    ax.set_xticklabels(D_FF_LEVELS_TASK)
    ax.set_title(task_name, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel("Normalized validation MSE")
fig.tight_layout()
plt.show()

for task_name, results in exp_g_normalized_mse.items():
    print(f"=== {task_name} ===")
    cols = [
        "d_ff",
        "GELU 平均",
        "GELU 標準偏差",
        "SwiGLU 平均",
        "SwiGLU 標準偏差",
        "GELU params",
        "SwiGLU params",
    ]
    widths = [6, 12, 14, 12, 14, 12, 13]
    print("  ".join(f"{c:>{w}s}" for c, w in zip(cols, widths, strict=True)))
    for d_ff in D_FF_LEVELS_TASK:
        gelu_values = np.array(results["GELU (standard feed-forward)"][d_ff])
        swiglu_values = np.array(results["SwiGLU"][d_ff])
        print(
            f"{d_ff:6d}  {gelu_values.mean():12.4f}  {gelu_values.std():14.4f}  "
            f"{swiglu_values.mean():12.4f}  {swiglu_values.std():14.4f}  "
            f"{exp_g_params[task_name]['GELU (standard feed-forward)'][d_ff]:12,d}  "
            f"{exp_g_params[task_name]['SwiGLU'][d_ff]:13,d}"
        )
    print()
```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/004_normalization_and_activation/output_68_0.png)
    


    === Multiplicative interaction: y_j = (w_1j^T x)(w_2j^T x) ===
      d_ff       GELU 平均       GELU 標準偏差     SwiGLU 平均     SwiGLU 標準偏差   GELU params  SwiGLU params
         8        0.8642          0.0054        0.8318          0.0026           552            480
        16        0.7629          0.0061        0.6498          0.0061         1,072          1,056
        32        0.5774          0.0069        0.3568          0.0039         2,112          2,016
        64        0.2395          0.0050        0.0315          0.0014         4,192          4,128
    
    === Negative control: y_j = tanh(w_j^T x) ===
      d_ff       GELU 平均       GELU 標準偏差     SwiGLU 平均     SwiGLU 標準偏差   GELU params  SwiGLU params
         8        0.5648          0.0013        0.8736          0.0035           552            480
        16        0.3595          0.0016        0.7605          0.0075         1,072          1,056
        32        0.2853          0.0020        0.6109          0.0035         2,112          2,016
        64        0.2643          0.0101        0.4806          0.0055         4,192          4,128
    


#### 実験 G の結果・考察

**乗法的相互作用タスク** では、4 つの中間次元水準すべてで、SwiGLU(パラメータ数を揃えた条件)が標準の順伝播ネットワーク(GELU)を明確に上回った。

| 中間次元 | GELU 平均 ± 標準偏差 | SwiGLU 平均 ± 標準偏差 |
|---:|---|---|
| 8  | $0.864 \pm 0.005$ | $0.832 \pm 0.003$ |
| 16 | $0.763 \pm 0.006$ | $0.650 \pm 0.006$ |
| 32 | $0.577 \pm 0.007$ | $0.357 \pm 0.004$ |
| 64 | $0.240 \pm 0.005$ | $0.032 \pm 0.001$ |

**陰性対照タスク($y_j = \tanh(w_j^\top x)$)では、結果が逆転した。** SwiGLU は標準の順伝播ネットワークに優位を示さないどころか、**4 水準すべてで明確に下回った**(標準誤差に対して大きな差)。

| 中間次元 | GELU 平均 ± 標準偏差 | SwiGLU 平均 ± 標準偏差 |
|---:|---|---|
| 8  | $0.565 \pm 0.001$ | $0.874 \pm 0.004$ |
| 16 | $0.360 \pm 0.002$ | $0.761 \pm 0.008$ |
| 32 | $0.285 \pm 0.002$ | $0.611 \pm 0.004$ |
| 64 | $0.264 \pm 0.010$ | $0.481 \pm 0.005$ |

これは「差を検出できなかった」ではなく、**SwiGLU が陰性対照タスクにおいて標準の順伝播ネットワークより明確に劣る** という積極的な結果である。単一の線形射影に要素ごとの非線形性をかけるだけの目標に対しては、ゲート機構(2 本の線形射影の積)を持つ SwiGLU の構造がむしろ最適化・表現の面で不利に働いたと考えられる(標準の順伝播ネットワークは、この目標を近似するのに十分な素直な構造を持つ)。

この対比は 6.7 節冒頭で述べた「乗法的相互作用タスクの目標関数は SwiGLU の関数形にほぼそのまま含まれる」という懸念に対する直接の反証になる。もし乗法的相互作用タスクでの SwiGLU の優位が単に「目標が SwiGLU の仮説クラスに含まれているから」だけで説明されるなら、SwiGLU は陰性対照タスクでも(不利にはならず)少なくとも標準の順伝播ネットワークと同程度には機能するはずである。しかし実際には陰性対照タスクで明確に劣っており、**SwiGLU の優位性は「その特定のタスクを表現できるかどうか」ではなく「そのタスクが乗法的相互作用を必要とするかどうか」に紐づいている** ことを、より強い形で裏づけている。



## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/01_foundations/004_normalization_and_activation.ipynb
