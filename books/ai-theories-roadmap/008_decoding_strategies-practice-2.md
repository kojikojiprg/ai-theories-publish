---
title: "デコーディング戦略(Decoding Strategies)(実装・実験編 2/2)"
---

この記事は後編(実装・実験編 2/2)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/008_decoding_strategies-practice-1)。

## 7. 実験 / Experiments

CLAUDE.md「1 実験 = 1 検証事項」規約により、実験 A(退化現象)・実験 B(top-k と
top-p の適応性の違い)を主軸(三値判定を厳密に適用)、実験 C(temperature のトレード
オフ)・実験 D(beam size のトレードオフ)を補助(定性的な観察、判定基準を設けない)
とする。

### 7.1 生成実験の共通設定

プロンプト・生成長・シード数は本番実行前に(結果を見る前に)ここで宣言する。



```python
if SMOKE_TEST:
    N_PROMPTS = 4
    PROMPT_LENGTH = 12
    GEN_LENGTH = 24
    N_SEEDS_STOCHASTIC = 3
else:
    N_PROMPTS = 8
    PROMPT_LENGTH = 24
    GEN_LENGTH = 64
    N_SEEDS_STOCHASTIC = 5

REP_N = 4  # n-gram 重複率の n(Holtzman et al. 2020 の rep-4 慣習)
DISTINCT_N = 3  # distinct-n の n
BEAM_SIZE_MAIN = 4  # 実験 A で使うビームサーチのビーム数
TOP_P_MAIN = 0.9  # 実験 A・B で使う top-p の p
TOP_K_MAIN = 40  # 実験 B の診断で使う top-k の k
TEMPERATURE_MAIN = 1.0  # 実験 A で使う temperature サンプリングの温度

# 検証テキストからプロンプトを均等な間隔で切り出す(乱数に依存させず再現性を持たせる)。
_val_len = val_ids.size(0)
_stride = max(1, (_val_len - PROMPT_LENGTH - GEN_LENGTH) // N_PROMPTS)
prompt_ids_list = [
    val_ids[i * _stride : i * _stride + PROMPT_LENGTH].unsqueeze(0).to(device)
    for i in range(N_PROMPTS)
    if i * _stride + PROMPT_LENGTH <= _val_len
]
N_PROMPTS = len(prompt_ids_list)  # 端数により減った場合はここで確定させる
print(
    f"N_PROMPTS={N_PROMPTS}, PROMPT_LENGTH={PROMPT_LENGTH}, GEN_LENGTH={GEN_LENGTH}, "
    f"N_SEEDS_STOCHASTIC={N_SEEDS_STOCHASTIC}"
)
print(
    f"REP_N={REP_N}, DISTINCT_N={DISTINCT_N}, BEAM_SIZE_MAIN={BEAM_SIZE_MAIN}, "
    f"TOP_P_MAIN={TOP_P_MAIN}, TOP_K_MAIN={TOP_K_MAIN}, TEMPERATURE_MAIN={TEMPERATURE_MAIN}"
)
```

    N_PROMPTS=8, PROMPT_LENGTH=24, GEN_LENGTH=64, N_SEEDS_STOCHASTIC=5
    REP_N=4, DISTINCT_N=3, BEAM_SIZE_MAIN=4, TOP_P_MAIN=0.9, TOP_K_MAIN=40, TEMPERATURE_MAIN=1.0


### 7.2 実験宣言セル: 検証すること・判定基準

**共通の対比量・標準偏差の導出(実験 A・B で共通)**: グループ A・グループ B をそれぞれ
$n_A$・$n_B$ 個のサンプル(実験 A ではプロンプト x 手法の組、実験 B ではプロンプトごとの
相関係数)で測定し、標本平均 $\bar{x}_A, \bar{x}_B$・標本標準偏差 $s_A, s_B$
(不偏標準偏差、`ddof=1`)を求める。

$$
\Delta = \bar{x}_B - \bar{x}_A \qquad \mathrm{SE}(\Delta) = \sqrt{\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}}
\qquad \text{閾値} = 2 \times \mathrm{SE}(\Delta)
$$

判定は **支持**($\Delta$ が期待方向に閾値超え)/ **反証**($\Delta$ が逆方向に閾値超え)/
**判定不能**($|\Delta| \le$ 閾値)の 3 分岐とする(007 6.2 節・6.4 節と同一の方式)。

| #   | 実験                 | 対比条件(B vs A)                             | 対比量                                        | 期待方向                   |
| --- | -------------------- | -------------------------------------------- | --------------------------------------------- | -------------------------- |
| A   | 退化現象             | {temperature, top-p}(B) vs {貪欲法, beam}(A) | REP_N-gram 重複率                             | $\Delta < 0$(B の方が低い) |
| B   | top-k/top-p の適応性 | 実測相関係数(B) vs ゼロ基準(A、全要素 0)     | エントロピーと top-p 候補集合サイズの相関係数 | $\Delta > 0$(正の相関)     |

**実験 A の対比量の直接性について**: 検証したい介入(デコーディング戦略の違い)が
直接作用するのは各ステップでの「候補集合の絞り込み方」であり、n-gram 重複率は
その結果として系列全体に現れる下流の量である。3.4 節の理論(退化現象は繰り返し
ループへの自己強化的な収束として説明される)に従えば、n-gram 重複率(特に短い n)は
繰り返しループの直接の観測量に近く、両者の間に他の要因が介在しにくいため、対比量として
採用する(Holtzman et al., 2020 自身も rep-n を主要な定量指標として用いている)。

**実験 B の対比量の直接性について**: 検証したい性質(top-p の候補集合サイズがエントロピーに
応じて動的に変化すること)そのものを対比量(相関係数)として直接測定しており、介入の
作用点との乖離はない。

**前提条件(pre-condition)**: 判定基準とは別に、検証したい仮説とは独立な量として
本番実行の前に宣言する。

- **前提条件 P0(学習の進行、5.10 節で確定済み)**: 最終検証 bits-per-byte が
  `PRECONDITION_BPB_THRESHOLD` 以下であること。実験 A・B の両方に共通して適用する
  (モデルが十分に流暢なテキストを生成できないと、退化現象の比較(実験 A)も
  分布の形状に基づく適応性の比較(実験 B)も意味を持たない)。
- **前提条件 P1(分布の非退化、実験 B のみ)**: 各プロンプトのエントロピー系列
  (`GEN_LENGTH` ステップ分)の標準偏差が `PRECONDITION_ENTROPY_STD_MIN` を超えること。
  エントロピーが全ステップで一定であれば、エントロピーと候補集合サイズの相関係数は
  定義上不安定になり(分散 0 の変数との相関)、「適応性がない」ことの証拠にはならない
  (適応性を測る前提として、そもそもエントロピー自体に変動が必要という、検証したい
  仮説(top-p が適応的かどうか)とは独立な条件)。



```python
PRECONDITION_ENTROPY_STD_MIN = 0.05  # 前提条件 P1 の採用閾値(本番実行前に宣言、結果非依存)
```

### 7.3 判定関数



```python
def judge_three_way(delta: float, threshold: float, expect_positive: bool) -> str:
    """7.2 節の共通公式による 3 分岐判定(007 6.4 節と同一の実装)。"""
    if expect_positive:
        if delta > threshold:
            return "支持"
        elif delta < -threshold:
            return "反証"
        return "判定不能"
    else:
        if delta < -threshold:
            return "支持"
        elif delta > threshold:
            return "反証"
        return "判定不能"


def contrast_stats(values_a, values_b) -> dict:
    a, b = np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)
    mean_a, mean_b = a.mean(), b.mean()
    std_a = a.std(ddof=1) if len(a) > 1 else 0.0
    std_b = b.std(ddof=1) if len(b) > 1 else 0.0
    delta = mean_b - mean_a
    se = float(np.sqrt(std_a**2 / len(a) + std_b**2 / len(b)))
    return {
        "mean_a": float(mean_a),
        "mean_b": float(mean_b),
        "std_a": float(std_a),
        "std_b": float(std_b),
        "n_a": len(a),
        "n_b": len(b),
        "delta": float(delta),
        "se": se,
        "threshold": 2 * se,
    }


def judge_with_precondition(precondition_ok: bool, judge_fn, *args) -> tuple[str, dict | None]:
    """前提条件が成立している場合のみ judge_fn を呼ぶ(007 6.4 節と同一の実装)。"""
    if not precondition_ok:
        return "前提不成立", None
    return judge_fn(*args)


def judge_experiment_a(values_deterministic, values_stochastic) -> tuple[str, dict]:
    s = contrast_stats(values_deterministic, values_stochastic)
    return judge_three_way(s["delta"], s["threshold"], expect_positive=False), s


def judge_experiment_b(values_zero_baseline, values_correlation) -> tuple[str, dict]:
    s = contrast_stats(values_zero_baseline, values_correlation)
    return judge_three_way(s["delta"], s["threshold"], expect_positive=True), s


# --- 4 分岐到達性の確認(ダミー入力、本番データを使う前に実施) ---
_dummy_a = [1.0, 1.02, 0.98]
_dummy_cases = {
    "delta明確に正": [3.0, 3.05, 2.95],
    "delta明確に負": [-1.0, -0.95, -1.05],
    "deltaゼロ近辺": [1.0, 1.03, 0.97],
}
for _name, _judge_fn in {
    "experiment_a": judge_experiment_a,
    "experiment_b": judge_experiment_b,
}.items():
    _reached = set()
    for _case_name, _dummy_b in _dummy_cases.items():
        _verdict, _ = judge_with_precondition(True, _judge_fn, _dummy_a, _dummy_b)
        _reached.add(_verdict)
    print(f"{_name}(前提成立): 到達した判定 = {sorted(_reached)}")
    assert _reached == {"支持", "反証", "判定不能"}, (
        f"{_name} が 3 分岐すべてに到達しない: {_reached}"
    )
    _verdict_np, _stats_np = judge_with_precondition(
        False, _judge_fn, _dummy_a, _dummy_cases["delta明確に正"]
    )
    assert _verdict_np == "前提不成立" and _stats_np is None
print("OK: 実験 A・B の判定関数が、支持 / 反証 / 判定不能 / 前提不成立の 4 分岐すべてに到達する")
```

    experiment_a(前提成立): 到達した判定 = ['判定不能', '反証', '支持']
    experiment_b(前提成立): 到達した判定 = ['判定不能', '反証', '支持']
    OK: 実験 A・B の判定関数が、支持 / 反証 / 判定不能 / 前提不成立の 4 分岐すべてに到達する


### 7.4 実験 A: 退化現象(Degeneration)の検証

貪欲法・ビームサーチ(グループ A、決定的)と temperature サンプリング・top-p
サンプリング(グループ B、確率的)で、各プロンプトから生成した継続テキストの
REP_N-gram 重複率を比較する。グループ A はプロンプトごとに 1 サンプル(貪欲法・
ビームサーチそれぞれ)、グループ B はプロンプト x シードごとに 1 サンプル
(temperature・top-p それぞれ)を生成する。



```python
def generate_all_methods_for_prompt(prompt: torch.Tensor) -> dict[str, list[torch.Tensor]]:
    """1 プロンプトについて、実験 A で使う全手法の生成トークン列(新規生成分のみ)を返す。"""
    results: dict[str, list[torch.Tensor]] = {
        "greedy": [],
        "beam": [],
        "temperature": [],
        "top_p": [],
    }

    greedy_out = model.generate(prompt.clone(), max_new_tokens=GEN_LENGTH, temperature=0.0)
    results["greedy"].append(greedy_out[0, prompt.size(1) :].cpu())

    beam_out = beam_search(
        model,
        prompt.clone(),
        beam_size=BEAM_SIZE_MAIN,
        max_new_tokens=GEN_LENGTH,
        length_penalty=1.0,
    )
    best_beam_seq = beam_out[0][0]
    results["beam"].append(best_beam_seq[0, prompt.size(1) :].cpu())

    for seed in range(N_SEEDS_STOCHASTIC):
        temp_out = generate_with_filter(
            model, prompt.clone(), GEN_LENGTH, temperature=TEMPERATURE_MAIN, seed=seed
        )
        results["temperature"].append(temp_out[0, prompt.size(1) :].cpu())

        topp_out = generate_with_filter(
            model, prompt.clone(), GEN_LENGTH, temperature=1.0, top_p=TOP_P_MAIN, seed=seed
        )
        results["top_p"].append(topp_out[0, prompt.size(1) :].cpu())

    return results


experiment_a_generations = [generate_all_methods_for_prompt(p) for p in prompt_ids_list]

rep_rates_deterministic = []  # greedy + beam, プロンプトごと
rep_rates_stochastic = []  # temperature + top_p, プロンプト x シードごと
for gen in experiment_a_generations:
    rep_rates_deterministic.append(compute_ngram_repetition_rate(gen["greedy"][0], REP_N))
    rep_rates_deterministic.append(compute_ngram_repetition_rate(gen["beam"][0], REP_N))
    for seq in gen["temperature"]:
        rep_rates_stochastic.append(compute_ngram_repetition_rate(seq, REP_N))
    for seq in gen["top_p"]:
        rep_rates_stochastic.append(compute_ngram_repetition_rate(seq, REP_N))

_std_det = np.std(rep_rates_deterministic, ddof=1)
_std_sto = np.std(rep_rates_stochastic, ddof=1)
print(
    f"グループ A(貪欲法・beam、n={len(rep_rates_deterministic)}): "
    f"mean={np.mean(rep_rates_deterministic):.4f}, std={_std_det:.4f}"
)
print(
    f"グループ B(temperature・top-p、n={len(rep_rates_stochastic)}): "
    f"mean={np.mean(rep_rates_stochastic):.4f}, std={_std_sto:.4f}"
)
```

    グループ A(貪欲法・beam、n=16): mean=0.5205, std=0.2101
    グループ B(temperature・top-p、n=80): mean=0.0002, std=0.0018



```python
# 実験 A の判定コード(実装・実行するが、判定結果を結論として本文には書かない。8 節参照)。
verdict_a, stats_a = judge_with_precondition(
    PRECONDITION_P0_OK, judge_experiment_a, rep_rates_deterministic, rep_rates_stochastic
)
print(f"実験 A: 判定 = {verdict_a}")
if stats_a is not None:
    print(
        f"  delta={stats_a['delta']:.4f}, threshold={stats_a['threshold']:.4f}, "
        f"mean_a={stats_a['mean_a']:.4f}, mean_b={stats_a['mean_b']:.4f}"
    )
```

    実験 A: 判定 = 支持
      delta=-0.5203, threshold=0.1050, mean_a=0.5205, mean_b=0.0002


### 7.5 実験 B: top-k と top-p の適応性の違い

各プロンプトについて貪欲法で `GEN_LENGTH` ステップ生成する軌道に沿って、各ステップの
分布のエントロピーと、その分布に `top_p_filter(p=TOP_P_MAIN)` を適用したときの
候補集合サイズ(有限の logits を持つ要素数)を記録する。プロンプトごとに、
エントロピー系列と候補集合サイズ系列の間のピアソン相関係数を計算する
(プロンプト = 1 サンプルとして扱う)。診断として、`top_k_filter(k=TOP_K_MAIN)` の
候補集合サイズが全ステップで定数 `TOP_K_MAIN` に一致することも確認する。



```python
@torch.no_grad()
def trace_entropy_and_nucleus_size(prompt: torch.Tensor) -> dict[str, np.ndarray]:
    """貪欲法の生成軌道に沿って、各ステップのエントロピー・top-p 候補集合サイズ・
    top-k 候補集合サイズを記録する(実験 B)。"""
    entropies, topp_sizes, topk_sizes = [], [], []
    token_ids = prompt.clone()
    was_training = model.training
    model.eval()
    for _ in range(GEN_LENGTH):
        context = token_ids[:, -model.max_sequence_length :]
        logits = model(context)[:, -1, :]  # (1, V)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).item()
        entropies.append(entropy)

        filtered_p = top_p_filter(logits, TOP_P_MAIN)
        topp_sizes.append(int(torch.isfinite(filtered_p).sum().item()))

        filtered_k = top_k_filter(logits, TOP_K_MAIN)
        topk_sizes.append(int(torch.isfinite(filtered_k).sum().item()))

        next_token = logits.argmax(dim=-1, keepdim=True)
        token_ids = torch.cat([token_ids, next_token], dim=1)
    model.train(was_training)
    return {
        "entropy": np.array(entropies),
        "topp_size": np.array(topp_sizes),
        "topk_size": np.array(topk_sizes),
    }


experiment_b_traces = [trace_entropy_and_nucleus_size(p) for p in prompt_ids_list]

correlations_topp = []
entropy_stds = []
topk_size_all = []
for trace in experiment_b_traces:
    entropy_stds.append(float(np.std(trace["entropy"], ddof=1)))
    if np.std(trace["entropy"]) > 0 and np.std(trace["topp_size"]) > 0:
        r = float(np.corrcoef(trace["entropy"], trace["topp_size"])[0, 1])
    else:
        r = 0.0
    correlations_topp.append(r)
    topk_size_all.extend(trace["topk_size"].tolist())

print(
    f"プロンプトごとの相関係数(entropy vs top-p 候補集合サイズ): "
    f"{[round(r, 3) for r in correlations_topp]}"
)
print(f"エントロピー系列の標準偏差(プロンプトごと): {[round(s, 4) for s in entropy_stds]}")
print(
    f"top-k 候補集合サイズ(全ステップ x 全プロンプト、診断): "
    f"min={min(topk_size_all)}, max={max(topk_size_all)}(TOP_K_MAIN={TOP_K_MAIN} と一致するはず)"
)
assert all(s == TOP_K_MAIN for s in topk_size_all), "top-k 候補集合サイズが固定値と一致しない"
print("OK: top-k 候補集合サイズは全ステップで定数(TOP_K_MAIN)に一致(=非適応性の直接確認)")
```

    プロンプトごとの相関係数(entropy vs top-p 候補集合サイズ): [0.837, 0.864, 0.856, 0.854, 0.589, 0.878, 0.919, 0.89]
    エントロピー系列の標準偏差(プロンプトごと): [1.5828, 1.5677, 1.6034, 1.6881, 1.0812, 0.8966, 0.9856, 1.8494]
    top-k 候補集合サイズ(全ステップ x 全プロンプト、診断): min=40, max=40(TOP_K_MAIN=40 と一致するはず)
    OK: top-k 候補集合サイズは全ステップで定数(TOP_K_MAIN)に一致(=非適応性の直接確認)



```python
PRECONDITION_P1_OK = all(s > PRECONDITION_ENTROPY_STD_MIN for s in entropy_stds)
print(
    f"前提条件 P1(分布の非退化): 全プロンプトのエントロピー標準偏差 > "
    f"{PRECONDITION_ENTROPY_STD_MIN} ? {'OK' if PRECONDITION_P1_OK else 'NG'}"
)
PRECONDITION_B_OK = PRECONDITION_P0_OK and PRECONDITION_P1_OK

# 実験 B の判定コード(実装・実行するが、判定結果を結論として本文には書かない。8 節参照)。
_zero_baseline = [0.0] * len(correlations_topp)
verdict_b, stats_b = judge_with_precondition(
    PRECONDITION_B_OK, judge_experiment_b, _zero_baseline, correlations_topp
)
print(f"実験 B: 判定 = {verdict_b}")
if stats_b is not None:
    print(f"  delta(=mean相関係数)={stats_b['delta']:.4f}, threshold={stats_b['threshold']:.4f}")
```

    前提条件 P1(分布の非退化): 全プロンプトのエントロピー標準偏差 > 0.05 ? OK
    実験 B: 判定 = 支持
      delta(=mean相関係数)=0.8358, threshold=0.0726


### 7.6 可視化: 実験 A・B



```python
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

axes[0].boxplot(
    [rep_rates_deterministic, rep_rates_stochastic], tick_labels=["greedy+beam", "temp+top-p"]
)
axes[0].set_ylabel(f"{REP_N}-gram repetition rate")
axes[0].set_title("Experiment A: repetition rate by method group")
axes[0].grid(alpha=0.3, axis="y")

_trace0 = experiment_b_traces[0]
axes[1].scatter(_trace0["entropy"], _trace0["topp_size"], s=14, alpha=0.7, label="top-p (adaptive)")
axes[1].scatter(
    _trace0["entropy"], _trace0["topk_size"], s=14, alpha=0.7, marker="x", label="top-k (fixed)"
)
axes[1].set_xlabel("Entropy (nats)")
axes[1].set_ylabel("Candidate set size")
axes[1].set_title("Experiment B: entropy vs candidate set size (prompt 0)")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

fig.tight_layout()
plt.show()
```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/008_decoding_strategies/output_59_0.png)
    


### 7.7 実験 C(補助): temperature による多様性と一貫性のトレードオフ

**判定基準は設けない(定性的な観察のみである)。** temperature を等比刻みで複数水準
振り、各水準で生成したテキストの distinct-n(多様性)と、生成テキストを学習済み
モデル自身で評価した bits-per-byte 相当の指標(生成トークン列に対する平均負の対数
尤度、一貫性の代理指標)を観察する。



```python
TEMPERATURES = [0.5, 0.7, 1.0, 1.4, 2.0]  # 公比約 1.4 の等比刻み

experiment_c_results = {"temperature": [], "distinct_n_mean": [], "self_nll_mean": []}
for temperature in TEMPERATURES:
    distinct_ns, self_nlls = [], []
    for prompt in prompt_ids_list:
        for seed in range(N_SEEDS_STOCHASTIC):
            out = generate_with_filter(
                model, prompt.clone(), GEN_LENGTH, temperature=temperature, seed=seed
            )
            new_tokens = out[0, prompt.size(1) :]
            distinct_ns.append(compute_distinct_n(new_tokens.cpu(), DISTINCT_N))

            with torch.no_grad():
                context_for_eval = out[:, :-1]
                logits = model(context_for_eval[:, -model.max_sequence_length :])
                target = out[:, -new_tokens.size(0) :]
                nll = torch.nn.functional.cross_entropy(
                    logits[:, -new_tokens.size(0) :, :].reshape(-1, tokenizer.vocab_size),
                    target.reshape(-1),
                )
                self_nlls.append(nll.item())

    experiment_c_results["temperature"].append(temperature)
    experiment_c_results["distinct_n_mean"].append(float(np.mean(distinct_ns)))
    experiment_c_results["self_nll_mean"].append(float(np.mean(self_nlls)))
    print(
        f"T={temperature}: distinct-{DISTINCT_N}(mean)={np.mean(distinct_ns):.4f}, "
        f"self-NLL(mean, nats)={np.mean(self_nlls):.4f}"
    )
```

    T=0.5: distinct-3(mean)=0.9194, self-NLL(mean, nats)=2.1466
    T=0.7: distinct-3(mean)=0.9742, self-NLL(mean, nats)=2.7501
    T=1.0: distinct-3(mean)=0.9988, self-NLL(mean, nats)=4.3752
    T=1.4: distinct-3(mean)=1.0000, self-NLL(mean, nats)=6.7058
    T=2.0: distinct-3(mean)=1.0000, self-NLL(mean, nats)=9.4008



```python
fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
ax2 = ax1.twinx()
ax1.plot(
    experiment_c_results["temperature"],
    experiment_c_results["distinct_n_mean"],
    "o-",
    color="tab:blue",
    label=f"distinct-{DISTINCT_N} (left)",
)
ax2.plot(
    experiment_c_results["temperature"],
    experiment_c_results["self_nll_mean"],
    "s--",
    color="tab:red",
    label="self-NLL (right)",
)
ax1.set_xlabel("Temperature")
ax1.set_ylabel(f"distinct-{DISTINCT_N} (diversity)")
ax2.set_ylabel("self-NLL (nats, consistency proxy)")
ax1.set_title("Experiment C: temperature vs diversity / consistency (qualitative)")
ax1.grid(alpha=0.3)
fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88), fontsize=8)
fig.tight_layout()
plt.show()
```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/008_decoding_strategies/output_62_0.png)
    


### 7.8 実験 D(補助): beam size と生成確率・多様性のトレードオフ

**判定基準は設けない(定性的な観察のみである)。** beam size を公比 2 の等比刻みで
複数水準振り、各水準で best 候補の長さ正規化対数尤度(スコア)と、最終ビーム内候補間の
多様性(候補どうしを連結した系列の distinct-n、値が低いほど候補間で似た系列が多い)を
観察する。



```python
BEAM_SIZES = [1, 2, 4, 8]  # 公比 2 の等比刻み

experiment_d_results = {"beam_size": [], "best_score_mean": [], "beam_diversity_mean": []}
for beam_size in BEAM_SIZES:
    best_scores, diversities = [], []
    for prompt in prompt_ids_list:
        beams = beam_search(
            model,
            prompt.clone(),
            beam_size=beam_size,
            max_new_tokens=GEN_LENGTH,
            length_penalty=1.0,
        )
        best_scores.append(beams[0][1])  # 正規化済み対数尤度(降順ソート済みの先頭が best)

        if beam_size > 1:
            concatenated = torch.cat([seq[0, prompt.size(1) :] for seq, _ in beams]).cpu()
            diversities.append(compute_distinct_n(concatenated, DISTINCT_N))
        else:
            diversities.append(float("nan"))  # beam_size=1 では候補間多様性は定義できない

    diversity_mean = float(np.mean(diversities)) if beam_size > 1 else float("nan")
    experiment_d_results["beam_size"].append(beam_size)
    experiment_d_results["best_score_mean"].append(float(np.mean(best_scores)))
    experiment_d_results["beam_diversity_mean"].append(diversity_mean)
    _diversity_str = f"{diversity_mean:.4f}" if beam_size > 1 else "N/A(beam_size=1)"
    print(
        f"beam_size={beam_size}: best_score(mean)={np.mean(best_scores):.4f}, "
        f"beam内distinct-{DISTINCT_N}(mean)={_diversity_str}"
    )
```

    beam_size=1: best_score(mean)=-6.3028, beam内distinct-3(mean)=N/A(beam_size=1)
    beam_size=2: best_score(mean)=-6.0012, beam内distinct-3(mean)=0.2440
    beam_size=4: best_score(mean)=-5.8373, beam内distinct-3(mean)=0.1447
    beam_size=8: best_score(mean)=-5.0329, beam内distinct-3(mean)=0.0868



```python
fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
ax2 = ax1.twinx()
ax1.plot(
    experiment_d_results["beam_size"],
    experiment_d_results["best_score_mean"],
    "o-",
    color="tab:blue",
    label="best score (left)",
)
ax2.plot(
    experiment_d_results["beam_size"][1:],
    experiment_d_results["beam_diversity_mean"][1:],
    "s--",
    color="tab:red",
    label="beam diversity (right)",
)
ax1.set_xscale("log", base=2)
ax1.set_xticks(BEAM_SIZES)
ax1.set_xticklabels(BEAM_SIZES)
ax1.set_xlabel("Beam size")
ax1.set_ylabel("Best candidate length-normalized log-likelihood")
ax2.set_ylabel(f"distinct-{DISTINCT_N} across beam candidates")
ax1.set_title("Experiment D: beam size vs likelihood / diversity (qualitative)")
ax1.grid(alpha=0.3)
fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88), fontsize=8)
fig.tight_layout()
plt.show()
```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/008_decoding_strategies/output_65_0.png)
    


## 8. 結果・考察 / Results and Discussion

**実行環境についての注記**: 以下は Google Colab T4 GPU での本番実行(`SMOKE_TEST=False`、本番スケールのコーパス・語彙サイズ・ステップ数)の実際のセル出力に基づく。判定基準は本番実行後に変更していない(7.2 節で事前登録した基準・閾値の導出式をそのまま適用する)。

### 8.1 モデル学習

較正(5.9 節)では、006 基準学習率(3e-4)の x0.5・x1.0・x2.0・x4.0 のいずれも `CALIBRATION_STEPS=2181` ステップで発散せず(`diverged=False`)、最終検証 bits-per-byte が最小だった x4.0(学習率 1.2e-3)を採用した。

| 学習率倍率     |       学習率 | 発散  | 最終 bits-per-byte |
| -------------- | -----------: | ----- | -----------------: |
| x0.5           |     1.50e-04 | False |             2.2082 |
| x1.0           |     3.00e-04 | False |             1.9767 |
| x2.0           |     6.00e-04 | False |             1.7678 |
| **x4.0(採用)** | **1.20e-03** | False |         **1.6622** |

gradient clipping 閾値は、採用した学習率での較正実行における勾配ノルムの 90% 分位点として **0.6341** に決定した。

本番学習(2,181 ステップ)の実測時間は **376.9 秒** であり、5.9 節末尾のスケーリング外挿(6.1 分 ≈ 366 秒)とほぼ一致した。学習曲線(下図)では、訓練損失が約 9(ほぼ一様分布相当)から単調に低下し 2,181 ステップ時点で 3.5〜4 nats 付近まで下がっており、バッチ由来のノイズはあるものの発散の兆候はない。検証 bits-per-byte も最初の評価(200 ステップ、2.48)から最終ステップ(2,181 ステップ、1.6707)まで単調に減少し、前提条件 P0 の閾値 `PRECONDITION_BPB_THRESHOLD=2.5` を最初の評価時点から一貫して下回っている。

**最終検証 bits-per-byte = 1.6707 は閾値 2.5 を大きく下回り、前提条件 P0(学習の進行)は成立する。**

### 8.2 実験 A: 退化現象(Degeneration)

| グループ | 手法                 |   n |   mean |    std |
| -------- | -------------------- | --: | -----: | -----: |
| A        | 貪欲法・ビームサーチ |  16 | 0.5205 | 0.2101 |
| B        | temperature・top-p   |  80 | 0.0002 | 0.0018 |

$\Delta = \bar{x}_B - \bar{x}_A = 0.0002 - 0.5205 = -0.5203$、閾値 $2 \times \mathrm{SE}(\Delta) = 0.1050$。$\Delta$ は期待方向(負)に閾値を大きく超えており、**判定 = 支持** である。

**貪欲法・ビームサーチのような最大化に基づく decoding は、temperature・top-p サンプリングよりも明確に高い 4-gram 重複率を示した。** これは Holtzman et al., 2020 の理論的主張(3.4 節)を裏付ける結果である。5.11 節の生成ヘルパー動作確認セル(短い 1 プロンプト・10 トークンのみの簡易的な例であり、実験 A 本体の 8 プロンプト分の生成とは別のサンプルである点に注意)でも、同様の傾向を定性的に確認できる。

```
greedy: Conjugable words (verbed by the name of the same name)
beam  : Conjugable words (verbed into the name of the same name)
temp  : Conjugable words (verpended for the seat use) for the Dec
top-p : Conjugable words (verpended for the nativature), the name
```

greedy・beam はいずれも文末が「the name of the same name」「the name」で終わっており、短い例ながら同一フレーズの反復が観察できる。一方 temperature・top-p は同じ長さでもより多様な語が現れている。実験 A 本体の定量的な結果(グループ A の平均重複率 0.5205 はグループ B の 0.0002 の 2000 倍以上)が主たる根拠であり、この生成例はそれを補強する定性的な一例として位置づける。

### 8.3 実験 B: top-k と top-p の適応性の違い

前提条件 P1(全プロンプトのエントロピー標準偏差 > 0.05)は、8 プロンプト全てで標準偏差が 0.90〜1.85(`entropy_stds`、7.5 節のセル出力)と閾値を大きく上回っており **成立する**。

プロンプトごとの相関係数(エントロピー vs top-p 候補集合サイズ): [0.837, 0.864, 0.856, 0.854, 0.589, 0.878, 0.919, 0.890]。ゼロ基準との対比で $\Delta = 0.8358$、閾値 $2 \times \mathrm{SE}(\Delta) = 0.0726$。$\Delta$ は期待方向(正)に閾値を大きく超えており、**判定 = 支持** である。

7.6 節の散布図(実験 B、プロンプト 0)でも、top-p(青丸)の候補集合サイズはエントロピーが低い(0〜2 nats)領域ではほぼ 0〜数十個に留まる一方、エントロピーが高くなる(5〜6 nats)につれて候補集合サイズが数百〜1000 近くまで拡大しており、右肩上がりの関係が視覚的にも明瞭に確認できる。対照的に top-k(オレンジ x)の候補集合サイズはエントロピーによらず常に `TOP_K_MAIN=40` の水平な帯に張り付いている。**top-p が分布のエントロピーに応じて候補集合サイズを動的に変化させる一方、top-k は常に固定サイズであるという 3.6 節の理論的予想を、定量的な相関係数と散布図の両方が支持している。**

### 8.4 実験 C(補助・定性): temperature による多様性と一貫性のトレードオフ

**この実験には判定基準を設けていないため、以下は観察であり結論(支持・反証等)ではない。**

| temperature | distinct-3(多様性) | self-NLL(nats、一貫性の代理指標) |
| ----------- | -----------------: | -------------------------------: |
| 0.5         |             0.9194 |                           2.1466 |
| 0.7         |             0.9742 |                           2.7501 |
| 1.0         |             0.9988 |                           4.3752 |
| 1.4         |             1.0000 |                           6.7058 |
| 2.0         |             1.0000 |                           9.4008 |

temperature を上げると distinct-3 は単調に増加し、T=1.0 付近でほぼ上限(1.0 に近い値)に達して以降は頭打ちになる一方、self-NLL(モデル自身にとっての生成テキストの意外性、一貫性の代理指標)は上限に達したあとも単調に悪化し続けている。多様性が飽和したあとも一貫性の代理指標だけが悪化し続けるという非対称な関係が観察されており、3.6 節・7.7 節で述べた理論的な予想(多様性と一貫性のトレードオフ)とおおむね整合する観察である。

### 8.5 実験 D(補助・定性): beam size と生成確率・多様性のトレードオフ

**この実験には判定基準を設けていないため、以下は観察であり結論(支持・反証等)ではない。**

| beam size | best score(平均、長さ正規化対数尤度) | beam 内 distinct-3(多様性) |
| --------- | -----------------------------------: | -------------------------: |
| 1         |                              -6.3028 |    N/A(beam_size=1 のため) |
| 2         |                              -6.0012 |                     0.2440 |
| 4         |                              -5.8373 |                     0.1447 |
| 8         |                              -5.0329 |                     0.0868 |

beam size を増やすと best 候補のスコアは単調に改善する一方、beam 内候補間の多様性(distinct-3)は単調に低下しており、beam size を広げるほど探索の初期に分岐した近傍の系列が候補を占めるようになるという 3.5 節の理論的な予想と整合する観察である。

**スモークテスト(第 1 段階、ローカル・未学習に近いモデル)との比較についての注記(事後的な観察である)**: 008 のスモークテストでは、beam size を 1・2・4・8 と変えても best score がすべて `-12.4529` で完全に一致するという結果だった。これは判定基準による結論ではなく、本番実行後に振り返って気づいた観察である。考えられる説明としては、スモークテストのモデルは学習がほとんど進んでおらず(グリーディ生成が "the the the the ..." のように単一トークンへ収束するほど分布が極端に偏っていた)、この場合どの beam size で探索しても貪欲法の経路が他のどの候補よりも圧倒的に高い尤度を持ち続けるため、ビームを広げても best 候補が入れ替わらなかった、というものが考えられる(この説明は事後的な解釈であり、追加の検証は行っていない)。本番実行(学習済みモデル)では beam size 間で明確に差がつく結果になっており、**スモークテストの結果が本番スケールの結果を予測しない一例** である(006 7 節と同様の教訓)。

### 8.6 Hugging Face Hub へのアップロード

学習済みモデル・トークナイザ・モデルカードは [kojikojiprg/ai-theories-small-gpt-en](https://huggingface.co/kojikojiprg/ai-theories-small-gpt-en) に公開した(5.12 節のセル出力「アップロード完了」を確認済み)。

アップロード前の不変条件アサーション(6 節(1)、state_dict の保存・再読み込みラウンドトリップで logits が完全一致すること)は、本番学習したモデルに対しても成立を確認している(6 節のセル出力「OK: state_dict 保存・再読み込みラウンドトリップで logits が完全一致」)。これにより、アップロードしたアーティファクト(state_dict)がネットワーク呼び出しを介さずに検証済みの正しいものであることを担保している。



## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/02_pretraining/008_decoding_strategies.ipynb
