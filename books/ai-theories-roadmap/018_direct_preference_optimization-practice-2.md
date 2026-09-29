---
title: "DPO(Direct Preference Optimization) / Direct Preference Optimization(実装・実験編 2/4)"
---

この記事は後編(実装・実験編 2/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/018_direct_preference_optimization-practice-1)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/018_direct_preference_optimization-practice-3)。

### 5.7 学習と評価のヘルパー

- `sample_responses()`: 方策から temperature 1.0(top-k・top-p なし)で終端記号まで生成する(017 と同じ)。
- `true_rewards()`: 応答を復号して $r^*$ を計算する(同じ文字列・同じ正解の組はキャッシュする)。
- `reference_log_probs()`: 学習データの応答の $\log \pi_{\mathrm{ref}}$ を`precompute_reference_log_probs()`で求める。
  鍵(参照方策の SHA-256・プロンプトと応答のハッシュ)ごとにキャッシュし、全条件・全シードで再利用する。
- `build_dataset()`: 組の添字とラベルの行列 $(N, K)$ から、学習の事例(プロンプト・$y_w$・$y_l$・その $\log \pi_{\mathrm{ref}}$)を
  作る。組の中の向きは、1 つ目の応答 $y_1$ を先にした $\hat{p}$($y_1$ が選ばれた割合)で持つ。
- `pair_statistics()`: 組ごとの $\log \pi_\theta - \log \pi_{\mathrm{ref}}$(`first`・`second`)と $\hat{p}$ から、事例単位の量
  (事例の平均。組 $i$ は事例 $K$ 個に対応するので、$\hat{p}_i$ で向きを重み付けした組の平均と等しい)を求める。
  $h_{1,i} = \Delta_{1,i} - \Delta_{2,i}$($\Delta$ は $\log \pi_\theta - \log \pi_{\mathrm{ref}}$、$y_1$ を選好とした向きのマージン)として
  - マージンの平均 $\bar{h} = \frac{1}{N} \sum_i (2\hat{p}_i - 1) h_{1,i}$
  - 暗黙の報酬による正解率 $\frac{1}{N} \sum_i \left[ \hat{p}_i \mathbb{1}(h_{1,i} > 0) + (1 - \hat{p}_i) \mathbb{1}(h_{1,i} < 0) \right]$
  - 選好された応答の対数比の平均 $\frac{1}{N} \sum_i \left[ \hat{p}_i \Delta_{1,i} + (1 - \hat{p}_i) \Delta_{2,i} \right]$(実験 C・D)
- `run_training()`: 1 つの条件・シードの学習と評価。LoRA の初期化(`torch.manual_seed(18200 + s)`)・ミニバッチの順序
  (`make_epoch_batches(K N, T, 32, 18400 + s)`)はシード $s$ で決まり、同じシードなら損失・ラベルが違っても共通である。
  学習中のステップ $T/2$ と $T$ で学習用の組全体の $\Delta_1, \Delta_2$ を記録する。学習の後、評価用のプロンプトごとに
  $m$ 個の応答をサンプリングし(乱数の状態はシードごとに全条件で共通)、真の報酬と KL ダイバージェンスの推定値を入力 $x$
  ごとの和として記録する(応答のトークン数と、終端記号で止まった応答の数も記録する)。$\beta$ の較正では、較正用のプロンプトで **KL ダイバージェンスのみ** を測り、真の報酬を計算しない。


```python
def sample_responses(
    model: nn.Module, prompts: list, seed: int, stop_text: str = END_MARKER
) -> list:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return sample_generate_until_stop(
        model,
        prompts,
        tokenizer.decode,
        stop_text,
        MAX_NEW_TOKENS,
        device,
        generator,
        TEMPERATURE,
        GENERATION_BATCH_SIZE,
    )


TRUE_REWARD_CACHE: dict[tuple[str, tuple[str, ...]], float] = {}


def true_rewards(responses: list, examples: list) -> np.ndarray:
    values = np.empty(len(responses), dtype=np.float64)
    for i, (response, example) in enumerate(zip(responses, examples, strict=True)):
        key = (tokenizer.decode(response), example.answer)
        if key not in TRUE_REWARD_CACHE:
            TRUE_REWARD_CACHE[key] = compute_true_reward(*key)
        values[i] = TRUE_REWARD_CACHE[key]
    return values


REFERENCE_LOG_PROB_CACHE: dict = {}  # 参照方策の対数確率のキャッシュ(条件・シードをまたいで再利用)
CACHE_STATS = {"hit": 0, "miss": 0}


def reference_log_probs(prompts: list, responses: list, cache: dict | None) -> np.ndarray:
    key = (REFERENCE_SHA256, hash_json([prompts, responses]))
    values, hit = precompute_reference_log_probs(
        reference_policy,
        prompts,
        responses,
        device,
        LOG_PROB_BATCH_SIZE,
        cache,
        key if cache is not None else None,
    )
    CACHE_STATS["hit" if hit else "miss"] += 1
    return values


def learning_rate_schedule(num_steps: int):
    warmup = max(1, round(WARMUP_RATIO * num_steps))
    return lambda step: LEARNING_RATE * min(1.0, step / warmup)


def build_dataset(
    pairs: dict, pair_indices: np.ndarray, label_matrix: np.ndarray, cache: dict | None
) -> dict:
    # pairs: {"prompts", "first", "second"}(候補の組)。label_matrix[i, k] は組 pair_indices[i] の k 回目で y_1 が選ばれたか
    pair_indices = np.asarray(pair_indices)
    assert label_matrix.shape == (len(pair_indices), NUM_DRAWS)
    prompts = [pairs["prompts"][i] for i in pair_indices]
    first = [pairs["first"][i] for i in pair_indices]
    second = [pairs["second"][i] for i in pair_indices]
    reference_first = reference_log_probs(prompts, first, cache)
    reference_second = reference_log_probs(prompts, second, cache)
    local, first_chosen = flatten_preference_labels(label_matrix)
    return {
        "pair_indices": pair_indices,
        "prompts": prompts,
        "first": first,
        "second": second,
        "reference_first": reference_first,
        "reference_second": reference_second,
        "p_hat": label_matrix.mean(axis=1),  # y_1 が選ばれた割合
        "instances": {
            "prompts": [prompts[i] for i in local],
            "chosen": [
                first[i] if c else second[i] for i, c in zip(local, first_chosen, strict=True)
            ],
            "rejected": [
                second[i] if c else first[i] for i, c in zip(local, first_chosen, strict=True)
            ],
            "reference_chosen": np.where(
                first_chosen, reference_first[local], reference_second[local]
            ),
            "reference_rejected": np.where(
                first_chosen, reference_second[local], reference_first[local]
            ),
        },
    }


def pair_log_ratios(model: nn.Module, dataset: dict) -> dict:
    # 組ごとの log pi_theta - log pi_ref(学習用の組全体)。train_direct_preference の evaluate に渡す
    first = compute_response_log_prob_sums(
        model, dataset["prompts"], dataset["first"], device, LOG_PROB_BATCH_SIZE
    )
    second = compute_response_log_prob_sums(
        model, dataset["prompts"], dataset["second"], device, LOG_PROB_BATCH_SIZE
    )
    return {
        "first": first - dataset["reference_first"],
        "second": second - dataset["reference_second"],
    }


def pair_contributions(ratios: dict, p_hat: np.ndarray) -> dict:
    # 組ごとの寄与(組の平均をとると事例単位の平均になる)
    h1 = ratios["first"] - ratios["second"]
    return {
        "margin": (2 * p_hat - 1) * h1,
        "accuracy": p_hat * (h1 > 0) + (1 - p_hat) * (h1 < 0),
        "chosen_log_ratio": p_hat * ratios["first"] + (1 - p_hat) * ratios["second"],
        "rejected_log_ratio": p_hat * ratios["second"] + (1 - p_hat) * ratios["first"],
    }


def pair_statistics(ratios: dict, p_hat: np.ndarray) -> dict:
    return {k: float(v.mean()) for k, v in pair_contributions(ratios, p_hat).items()}


def build_policy(seed: int) -> GPTLanguageModel:
    policy = copy.deepcopy(reference_policy)
    torch.manual_seed(LORA_INIT_SEED_BASE + seed)
    apply_lora(policy, LORA_TARGET_MODULES, rank=LORA_RANK, alpha=LORA_ALPHA)
    return policy


def evaluate_policy(
    policy: nn.Module, prompts: list, examples: list | None, seed: int, stop_text: str = END_MARKER
) -> dict:
    # 方策から m 個ずつサンプリングし、KL の推定値(と examples があれば真の報酬)を入力 x ごとの和で返す
    flat_prompts = [p for p in prompts for _ in range(SAMPLES_PER_PROMPT)]
    responses = sample_responses(policy, flat_prompts, seed, stop_text)
    policy_lp = compute_response_log_prob_sums(
        policy, flat_prompts, responses, device, LOG_PROB_BATCH_SIZE
    )
    reference_lp = compute_response_log_prob_sums(
        reference_policy, flat_prompts, responses, device, LOG_PROB_BATCH_SIZE
    )
    per_input = SAMPLES_PER_PROMPT * TASK_COUNT  # 入力 1 つあたりの応答の数
    result = {
        "kl_by_input": (policy_lp - reference_lp).reshape(-1, per_input).sum(axis=1),
        "length_by_input": np.array([len(r) for r in responses], dtype=np.float64)
        .reshape(-1, per_input)
        .sum(axis=1),
        "stopped_by_input": np.array(
            [END_MARKER in tokenizer.decode(r) for r in responses], dtype=np.float64
        )
        .reshape(-1, per_input)
        .sum(axis=1),  # 終端記号で止まった応答の数
        "count_per_input": per_input,
    }
    if examples is not None:  # 真の報酬(beta の較正では計算しない)
        flat_examples = [e for e in examples for _ in range(SAMPLES_PER_PROMPT)]
        rewards = true_rewards(responses, flat_examples)
        result["true_reward_by_input"] = rewards.reshape(-1, per_input).sum(axis=1)
        result["exact_by_input"] = (
            (rewards == 1.0).reshape(-1, per_input).sum(axis=1).astype(np.float64)
        )
        result["example_response"] = tokenizer.decode(responses[0])
    total = per_input * len(result["kl_by_input"])
    result["kl"] = float(result["kl_by_input"].sum() / total)
    result["mean_tokens"] = float(result["length_by_input"].sum() / total)
    result["stop_rate"] = float(result["stopped_by_input"].sum() / total)
    return result


def run_training(
    dataset: dict,
    loss_type: str,
    beta: float,
    seed: int,
    evaluation: str,
    num_steps: int | None = None,
) -> dict:
    # evaluation: "full"(評価用のプロンプトで真の報酬と KL)、"kl_only"(較正用のプロンプトで KL のみ)、"none"
    num_steps = TRAIN_STEPS if num_steps is None else num_steps
    t0 = time.time()
    policy = build_policy(seed)
    instances = dataset["instances"]
    batches = make_epoch_batches(
        len(instances["prompts"]), num_steps, BATCH_SIZE, BATCH_SEED_BASE + seed
    )
    history = train_direct_preference(
        policy,
        instances["prompts"],
        instances["chosen"],
        instances["rejected"],
        instances["reference_chosen"],
        instances["reference_rejected"],
        batches,
        AdamW(
            [p for p in policy.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=0.0
        ),
        loss_type,
        beta,
        device,
        learning_rate_schedule(num_steps),
        evaluation_steps=(num_steps // 2, num_steps),
        evaluate=lambda model: pair_log_ratios(model, dataset),
    )
    train_seconds = time.time() - t0
    record = {
        "loss_type": loss_type,
        "beta": beta,
        "seed": seed,
        "history": history,
        "batches_hash": hash_json(batches),
        "train_seconds": train_seconds,
        "final": pair_statistics(history["evaluations"][num_steps], dataset["p_hat"]),
    }
    t0 = time.time()
    if evaluation == "full":
        record["eval"] = evaluate_policy(
            policy, EVAL_PROMPT_IDS, EVAL_EXAMPLES, EVAL_SAMPLE_SEED_BASE + seed
        )
        record["eval_pairs"] = pair_log_ratios(policy, EVAL_PAIR_DATASET)
    elif evaluation == "kl_only":
        record["eval"] = evaluate_policy(
            policy, CALIBRATION_PROMPT_IDS, None, CALIBRATION_SAMPLE_SEED
        )
    else:
        assert evaluation == "none", evaluation
    record["eval_seconds"] = time.time() - t0
    del policy
    return record
```

### 5.8 決定的な実行の確認

`torch.use_deterministic_algorithms(True)`のもとで、選好の組のサンプリング・参照方策の対数確率の計算・DPO / IPO の学習
(数ステップ)・方策の評価(サンプリングと KL の推定)を 2 回実行し、決定的な実装を持たない演算で例外にならないこと、
2 回の結果が bit 単位で一致することを確かめる。LoRA 以外のパラメータ(埋め込みを含む)は凍結しているので、017 で MPS に
あった埋め込みの逆伝播の非決定性は、本トピックの学習には現れない。


```python
def _short_determinism_run() -> tuple:
    prompts = TRAIN_PROMPT_IDS[:8]
    flat = sample_responses(reference_policy, [p for p in prompts for _ in range(2)], 1)
    pairs = {"prompts": prompts, "first": flat[0::2], "second": flat[1::2]}
    labels = np.ones((8, NUM_DRAWS), dtype=bool)
    dataset = build_dataset(pairs, np.arange(8), labels, cache=None)
    records = []
    for loss_type in ("dpo", "ipo"):
        record = run_training(dataset, loss_type, CHECK_BETA, 0, "kl_only", num_steps=4)
        for key in ("train_seconds", "eval_seconds"):
            record.pop(key)
        records.append(record)
    return flat, dataset["reference_first"], records


def _equal(a, b) -> bool:
    # 入れ子の辞書・リスト・numpy の配列を bit 単位で比べる
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return a == b


_first_run, _second_run = _short_determinism_run(), _short_determinism_run()
assert _equal(_first_run, _second_run), "2 回の実行が bit 単位で一致しない"
print(
    f"決定的な実行({device}、torch.are_deterministic_algorithms_enabled() = {torch.are_deterministic_algorithms_enabled()}): "
    "サンプリング・参照方策の対数確率・DPO と IPO の 4 ステップの学習・評価のサンプリングと KL を 2 回実行して、"
    "bit 単位で一致。例外なし: OK"
)
```

    決定的な実行(cuda、torch.are_deterministic_algorithms_enabled() = True): サンプリング・参照方策の対数確率・DPO と IPO の 4 ステップの学習・評価のサンプリングと KL を 2 回実行して、bit 単位で一致。例外なし: OK


## 6. 実験 / Experiments

### 6.1 実験宣言セル: 共通の設定・検証すること・判定基準・前提条件

**この節の内容は本番実行の前に確定させ、結果を見た後に変更しない。** パイロット(6.2 節)を受けて決めた設定は、
本番実行の前に決めたものであり、6.2 節に経緯を記録した。

#### 共通の設定

- **参照方策 $\pi_{\mathrm{ref}}$**: 017 の SFT モデル(`kojikojiprg/ai-theories-small-gpt-en`のブランチ`sft-synthetic`、5.3・5.5 節)。
- **方策**: $\pi_{\mathrm{ref}}$ の複製の Query・Value 射影に LoRA($r = 8$、$\alpha = 8$)を掛け、LoRA のみを学習する。AdamW
  (重み減衰 0)、学習率 $10^{-3}$(6.2 節)、最初の 10% のステップで線形 warmup し、その後は一定。gradient clipping なし。
  バッチは 32 事例、ステップ数は **全条件で共通の $T = 768$**。
- **選好の組の候補**: 学習用の各プロンプトについて、$\pi_{\mathrm{ref}}$ から 2 応答を抽選する(temperature 1.0、top-k・top-p
  なし、乱数シード`18019`)。$y_1 = y_2$(文字列として同一、正規化編集距離 0)の組と、$\Delta r^* = r^*(y_1) - r^*(y_2) = 0$ の組を
  除外する。**除外は組だけで決まり、ラベルの種類によらない**(確率的・決定的なラベルの両条件で組の集合は同一。アサーションで
  確かめる)。
- **$\kappa$**: 除外後の候補の組の $\Delta r^*$ から、017 と同じ式 $\kappa = \log 3 / \operatorname{median}(\lvert \Delta r^* \rvert)$ で決める。
- **ラベル**: 1 つの組を $K = 4$ 回観測したデータとする。
  - **確率的なラベル**: シード $s$ ごとに、候補の組すべてについて $K$ 回独立に尺度 $\kappa$ の Bradley-Terry モデルから
    抽選する(乱数シード $18300 + s$)。
  - **決定的なラベル**: 真の報酬の高い方を選好とするラベルを $K$ 回重複させる(事例数を確率的なラベルと揃える)。
- **実験 A〜C の学習データ**: 除外後の候補の組の先頭 $N = 2048$ 組(事例 $NK = 8192$、$T$ ステップでちょうど 3 エポック)。
  **実験 A と実験 B の DPO・確率的なラベルの条件は同じデータ** であり、標準の $\beta$(下記)が実験 A の水準に含まれる場合は、
  同じシードの学習を共有する(実験 C も共有する)。
- **標準の $\beta$**(実験 B〜D。IPO の $\tau$ も同じ値 $\tau = \beta$ とし、目標のマージンは $1 / (2\tau)$): 6.7 節の較正の
  $\widehat{\mathrm{KL}}$ **のみ** から、格子を大きい $\beta$ から順に見て $\widehat{\mathrm{KL}} \le 2$ nats が続く最小の格子点とする
  (最大の格子点で超える場合は最大の格子点)。**規則の理由**: 実験 B〜D は学習用の組のマージンと尤度の振る舞いを調べる実験で
  あり、方策が参照方策から大きく離れて崩れた状態ではなく、参照方策の近傍にとどまる範囲で最も正則化の弱い設定を標準にする。
  2 nats は、017 の best-of-n で調べた KL の上界の範囲(0〜3.86 nats)の内側にある。DPO の原論文でよく使われる $\beta = 0.1$ は、
  全パラメータを小さな学習率で 1 エポック程度学習する設定での値であり、本トピックの LoRA・学習率 $10^{-3}$・3 エポックの
  設定では、6.2 節のパイロットで $\beta \le 0.2$ の KL が数十 nats に達した(KL のみを見た判断である)。
- **シード**: シード $s$ が決めるのは、LoRA の $A$ の初期化(`torch.manual_seed(18200 + s)`)・ミニバッチの順序
  (`make_epoch_batches(..., 18400 + s)`)・確率的なラベルの抽選・評価のサンプリング(乱数シード $18500 + s$)である。
  同じシードの条件どうしはこれらを共有する(対応のある比較)。シードは先頭から $0, 1, \dots$ を使い、実験 A・B・D の
  シード数 $S_A$・$S_B$・$S_D$ は削る段階で決まる(段階 0 ですべて 5)。
- **評価**: 評価用の入力 $\lvert X \rvert = 128$ と 4 課題の直積のプロンプト($P = 512$)ごとに、学習した方策から $m = 8$ 個の応答を
  サンプリングし(temperature 1.0)、真の報酬の平均 $R$ と KL ダイバージェンスの推定値
  $\widehat{\mathrm{KL}} = \frac{1}{Pm} \sum \left[ \log \pi_\theta(y \mid x) - \log \pi_{\mathrm{ref}}(y \mid x) \right]$(3.7 節)を求める。
- **決定的な実行**: `torch.use_deterministic_algorithms(True)`と`CUBLAS_WORKSPACE_CONFIG`を設定する(5.1 節)。

#### 記号

- $i = 1, \dots, N$: 学習用の組。$\hat{p}_{i,s}$: シード $s$ のラベルで組 $i$ の $y_1$ が選ばれた割合(決定的なラベルでは
  $\mathbb{1}(\Delta r^*_i > 0)$)。
- $\Delta_{1,i}(t)$・$\Delta_{2,i}(t)$: ステップ $t$ の方策の $\log \pi_\theta(y_{i,1} \mid x_i) - \log \pi_{\mathrm{ref}}(y_{i,1} \mid x_i)$ と、
  $y_{i,2}$ についての同じ量(nats)。$h_{1,i} = \Delta_{1,i} - \Delta_{2,i}$($y_1$ を選好とした向きのマージン)。
- **事例単位の平均**(5.7 節): 組 $i$ は事例 $K$ 個に対応するので、事例の平均は組の平均で書ける。
  - マージンの平均 $\bar{h}(t) = \frac{1}{N} \sum_i (2\hat{p}_i - 1) h_{1,i}(t)$
  - 選好された応答の対数比の平均 $\bar{\Delta}_w(t) = \frac{1}{N} \sum_i \left[ \hat{p}_i \Delta_{1,i}(t) + (1 - \hat{p}_i) \Delta_{2,i}(t) \right]$
  - 暗黙の報酬による正解率 $a(t) = \frac{1}{N} \sum_i \left[ \hat{p}_i \mathbb{1}(h_{1,i} > 0) + (1 - \hat{p}_i) \mathbb{1}(h_{1,i} < 0) \right]$
- $\beta_1 > \beta_2 > \dots > \beta_6$: 実験 A の水準($\beta_{j+1} = \beta_j / \sqrt{2}$)。$u_j = \log_2(1 / \beta_j)$(隣接する水準の間隔は 0.5)。
- $R_s(\beta)$・$\widehat{\mathrm{KL}}_s(\beta)$: シード $s$ の方策の真の報酬の平均と KL の推定値。上線はシード平均。

#### 共通の前提条件

- **学習の成立**: 各実験の学習したすべての方策(全条件・全シード)で、学習用の事例での暗黙の報酬による正解率 $a(T)$ が
  $0.55$ 以上であること。ラベルは確率的なので正解率の上限は 1 より小さい(017 のラベルの真の順序との一致率は約 0.75)。
  0.55 は、学習が 0.5(学習前の $h = 0$ の状態)から明らかに進んだことを表す下限であり、真の報酬や $\log \pi_\theta(y_w)$ とは
  独立な量である。

**標準偏差の単位**: 各対比量の標準偏差は、シード間の分散と、評価の単位(実験 A は評価用の入力 $x$、実験 B〜D は学習用の
組)を復元抽出する対応付きのブートストラップ(反復 10,000 回)の分散の和とする。ブートストラップは、全シード・全条件で
**同じ再標本** を使うので、条件間・シード間の相関が保たれる。**閾値は対比量の標準偏差の 2 倍** とする。

#### 削る段階(実行時間の予算に応じた自動選択)

| 段階 | 実験 D のシード数 $S_D$ | 実験 A のシード数 $S_A$ | 実験 B(と C)のシード数 $S_B$ |
|---|---|---|---|
| 0 | 5 | 5 | 5 |
| 1 | 3 | 5 | 5 |
| 2 | 3 | 3 | 5 |
| 3 | 3 | 3 | 3 |

- **選択の規則**: 本番の学習を始める前に(6.4 節)、スケーリングの計測(6.3 節)から 1 セッション全体の実行時間を段階ごとに
  見積もり、予算(T4 で 120 分)以内に収まる **最小の段階** を選ぶ。段階 3 でも超える場合は、学習の前に例外で停止する。
  **選択は見積もりのみに基づき、どの実験の結果も参照しない。** $\beta$ の水準は等比の構造を崩すので削らない。
- **段階の順序の理由**: 実験 D は他の実験と学習を共有しないうえ、類似度の効果は条件間の差であり、対応のあるシードで
  ばらつきが打ち消し合う。次に、学習の数が最も多い実験 A(6 水準)を削る。実験 B は差の差を対比量とし誤差が大きくなりやすく、
  実験 C もこれを共有するので最後に削る。

#### 実験 A: 直接選好最適化の過最適化

**検証すること**: DPO の $\beta$ を公比 $\sqrt{2}$ の等比で小さくしていくと、方策の真の報酬の期待値は、はじめ上がり、やがて下がる
(3.7 節、Rafailov et al. [3])。

**条件**: DPO・確率的なラベル・実験 A〜C の学習データ・$T$ ステップで、$\beta \in \{\beta_1, \dots, \beta_6\}$($\beta_{j+1} = \beta_j / \sqrt{2}$)。

**$\beta$ の較正(本番の冒頭、6.7 節)**: 較正の格子 $\{3.2 \times 2^{-k/2} : k = 0, 1, \dots, 10\}$(3.2〜0.1、公比 $\sqrt{2}$ の
11 点)の各 $\beta$ で、本番と同じデータ・同じステップ数・シード 0 で DPO を学習し、**較正用のプロンプト**(評価用と重ならない
32 入力 × 4 課題、各 $m = 8$ 応答)で $\widehat{\mathrm{KL}}$ **のみ** を測る(真の報酬は計算しない)。

- **水準**: $\beta_6$ を標準の $\beta$(上記の「$\widehat{\mathrm{KL}} \le 2$ nats」の規則で決まる格子点)とし、
  $\beta_j = \beta_6 \cdot 2^{(6-j)/2}$($j = 1, \dots, 6$、公比 $\sqrt{2}$ の等比、格子の連続する 6 点)とする。
  $\beta_1 = \beta_6 \cdot 2^{5/2}$ が格子に含まれない場合(標準の $\beta$ が格子の上から 6 番目より大きい場合)は、格子の上端の
  6 点を水準とする(このとき標準の $\beta$ は実験 A の水準に含まれず、実験 B の条件 2 は別に学習する)。
- **規則の理由**: 実験 A は、方策が参照方策の近傍にとどまる範囲(KL ダイバージェンスが 0 から崩壊の手前まで)で、有限の
  選好データへの過適合による過最適化(3.7 節)を調べる。そこで水準の下端 $\beta_6$ を標準の $\beta$ と同じ
  「$\widehat{\mathrm{KL}} \le 2$ nats」の規則で決め、それより小さい $\beta$(方策が崩れる領域)を判定の水準に含めない。017 の
  best-of-n で真の報酬が最大になった $n = 4$〜8 の KL の上界($\log n - (n-1)/n$ で約 0.64〜1.20 nats)はこの範囲に含まれる。
  公比 $\sqrt{2}$ は、6 水準をこの範囲に収めるための刻みである。較正は KL のみを見るので、真の報酬の曲線の形(仮説の方向)に
  依存しない。
- **旧規則(廃止)**: 公比 2 の格子 $\{0.1 \times 2^k : k = -5, \dots, 5\}$ の上で、$\widehat{\mathrm{KL}} \ge 0.1$ nats となる最大の
  $\beta$ を $\beta_1$ とし、公比 2 で 6 水準をとる。改訂の経緯は 6.2 節に記録した(本番実行の前の改訂)。

**対比量**: 水準の前半 $j \in \{1, 2, 3\}$ での $\bar{R}(\beta_j)$ の $u_j$ に対する最小二乗の傾き $g_1$ と、後半
$j \in \{4, 5, 6\}$ での同じ傾き $g_2$(017 の実験 B と同じ形。argmax などの極値統計は使わない)。

**標準偏差の導出**: 傾きは $\bar{R}(\beta_j)$ の線形結合なので、$\sigma_{g_k}^2 = s_{g_k}^2 / S_A + \mathrm{Var}_{\mathrm{boot}}(g_k)$。
$s_{g_k}$ はシードごとの傾き $g_{k,s}$ の標本標準偏差、$\mathrm{Var}_{\mathrm{boot}}$ は評価用の入力 $x$ を復元抽出する
クラスタブートストラップ(入力ごとの真の報酬の和を再標本の重みで集計する)の分散である。評価のサンプリングのばらつきは、
入力の内側のばらつきとしてブートストラップに含まれ、シード間の分散にも含まれる(二重に数えるので安全側である)。

**判定**:

- **支持**: $g_1 > 2\sigma_{g_1}$ かつ $g_2 < -2\sigma_{g_2}$。
- **反証**: $g_2 > 2\sigma_{g_2}$(範囲内では過最適化が起きず、真の報酬が上がり続ける)。
- **判定不能**: それ以外。

**前提条件**: 学習の成立(実験 A の全水準・全シード)。**PA(KL の範囲)**: $\overline{\widehat{\mathrm{KL}}}(\beta_6) > 0$ かつ
$\overline{\widehat{\mathrm{KL}}}(\beta_6) \ge 4 \max(\overline{\widehat{\mathrm{KL}}}(\beta_1), 0)$(評価用のプロンプトでのシード平均)。
方策の移動量の範囲が狭いと、「上がってから下がる」形を検証できないためである。

**作用点の記述**: $\beta$ が直接作用するのは正則化の強さ、すなわち方策が $\pi_{\mathrm{ref}}$ からどれだけ離れるか
($\widehat{\mathrm{KL}}$)である。真の報酬は、そこから暗黙の報酬と真の報酬のずれを経て決まる下流の量だが、検証したい主張
(過最適化)そのものを表す量なので対比量に選んだ。$\widehat{\mathrm{KL}}$ を診断量として併記し、真の報酬を
$\widehat{\mathrm{KL}}$ に対して描いた図も出す。

**診断量**: 水準ごとの $\overline{\widehat{\mathrm{KL}}}$、$\bar{R}$ のシードごとの曲線、完全一致($r^* = 1$)の割合、応答の平均
トークン数、評価用の組(評価用のプロンプトごとに $\pi_{\mathrm{ref}}$ から抽選した 1 組。$y_1 = y_2$ と同点を除く)での、真の報酬の
順序に向けたマージンの平均と、その符号が真の順序と一致する割合。

#### 崩壊の領域の観察(判定基準を設けない)

実験 A の直後に、$\beta_6 / 2$ と $\beta_6 / 4$(較正の格子を 2 点・4 点下った点。格子の下端を下回る場合は格子の下端とし、同じ点になった場合は 1 つにまとめる。
$\beta_6$ が格子の下端の場合は $\beta_6$ 自体になり、実験 A の学習を共有する)で、
DPO・確率的なラベル・**シード 0 のみ** を学習し、実験 A と同じ評価をする。真の報酬の平均、$\widehat{\mathrm{KL}}$、応答の
平均トークン数、終端記号で止まった応答の割合、完全一致の割合を印字し、真の報酬 対 $\widehat{\mathrm{KL}}$ の図に別の記号で
重ねる。**判定基準を設けない観察であり、実験 A の $g_2$ の計算に含めない。** 1 シードのみなので、削る段階によらず実行する。

#### 実験 B: 決定的なラベルでの過適合(DPO と IPO)

**検証すること**: ラベルが決定的なとき、DPO の対数比のマージンは学習の後半も伸び続け(3.5 節、最適解が無限大)、IPO では
そうならない(最適解が有限)。確率的なラベルでは $\hat{p} \in (0, 1)$ の組の最適なマージンが両損失とも有限である。

**条件**(実験 A〜C の学習データ、同じ組の集合・同じ事例数、標準の $\beta = \tau$、$T$ ステップ):

| 条件 | 損失 | ラベル |
|---|---|---|
| 1 | DPO | 決定的 |
| 2 | DPO | 確率的(標準の $\beta$ が実験 A の水準に含まれれば実験 A と同じ学習) |
| 3 | IPO | 決定的 |
| 4 | IPO | 確率的 |

**$K$ の決め方**: $K$ が小さいと $\hat{p} \in (0, 1)$ の組が少なく、確率的なラベルの条件も決定的なラベルに近づく。6.2 節の
パイロットの組の $\Delta r^*$ と $\kappa$ から、組ごとの確率 $p_i = \sigma(\kappa \Delta r^*_i)$ で
$\mathbb{E}[\hat{p} \in (0, 1)] = \frac{1}{N}\sum_i (1 - p_i^K - (1 - p_i)^K) \ge 0.5$ となる最小の 2 のべきとして $K = 4$ とした
(本番でも同じ期待値を印字する)。

**対比量**: 条件 $c$・シード $s$ の学習用の組でのマージンの平均の、ステップ $T/2$ から $T$ までの伸び
$g_{c,s} = \bar{h}_{c,s}(T) - \bar{h}_{c,s}(T/2)$ を求め、差の差

$$
D_s = (g_{1,s} - g_{2,s}) - (g_{3,s} - g_{4,s}), \qquad D = \frac{1}{S_B} \sum_s D_s
$$

を対比量とする。DPO での決定的と確率的の差が、IPO での同じ差より大きいことを表す。

**標準偏差の導出**: $D$ はシードごとに対応のある 4 条件から作るので、シードごとの $D_s$ の標本標準偏差 $s_D$ を使い、
$\sigma_D^2 = s_D^2 / S_B + \mathrm{Var}_{\mathrm{boot}}(D)$ とする。$\mathrm{Var}_{\mathrm{boot}}$ は学習用の組を復元抽出するブートストラップ
(組ごとの寄与 $(2\hat{p}_i - 1) h_{1,i}$ を再標本の重みで平均し直す。全条件・全シード・2 つのステップで同じ再標本)の分散で
ある。4 つの伸びの差の差なので、単一の条件の伸びより分散が大きいことに注意する(誤差伝播は再標本・シードごとに $D$ を作る
ことで自動的に含まれる)。

**判定**: $D > 2\sigma_D$ なら支持、$D < -2\sigma_D$ なら反証、それ以外は判定不能。

**前提条件**: 学習の成立(4 条件・全シード)。**PB(重複の効果)**: 確率的なラベルで $\hat{p}_i \in (0, 1)$ となる組の割合が、
全シードで 0.4 以上であること($K = 4$ での期待値は約 0.59。これを大きく下回ると、確率的なラベルの条件が決定的なラベルと
区別できない)。

**作用点の記述**: 損失関数とラベルが直接作用するのはマージン $h$ そのものであり、対比量は作用点に一致する。
診断量として、評価用の組でのマージンと $\widehat{\mathrm{KL}}$、ステップごとのバッチのマージンの推移を併記する。

#### 実験 C: 尤度の置き換わりの存在

**検証すること**: DPO の学習後、マージンは正に広がっている一方で、選好された応答の対数確率は参照方策より下がっている
(3.6 節)。

**条件・共有**: 実験 B の条件 2(DPO・確率的なラベル・標準の $\beta$)の学習結果をそのまま使い、**同じ学習を再実行しない**。

**対比量**: 学習用の事例での選好された応答の対数比の平均 $\bar{\Delta}_w(T)$ のシード平均(系列単位、nats)。

**標準偏差の導出**: $\sigma^2 = s^2 / S_B + \mathrm{Var}_{\mathrm{boot}}$($s$ はシードごとの $\bar{\Delta}_{w,s}(T)$ の標本標準偏差、
$\mathrm{Var}_{\mathrm{boot}}$ は実験 B と同じ組の再標本による分散)。

**判定**: 対比量 $< -2\sigma$ なら支持、$> 2\sigma$ なら反証、それ以外は判定不能。

**前提条件**: **PC(マージンの拡大)**: 全シードで $\bar{h}(T) > 0$(損失が実際にマージンを広げていること)。加えて学習の成立
(実験 B と共通)。

**作用点の記述**: DPO 損失が直接作用するのはマージンであり、$\bar{\Delta}_w$ はその 2 つの成分の一方である。主張そのものが
「差が広がっても成分が下がる」ことなので、成分を対比量とする。診断量として、トークンあたりの値
($\sum_i [\hat{p}_i \Delta_{1,i} + (1-\hat{p}_i) \Delta_{2,i}] / \sum_i [\hat{p}_i \lvert y_{i,1} \rvert + (1-\hat{p}_i) \lvert y_{i,2} \rvert]$)、
選好されなかった応答の対数比の平均、マージンの平均を併記する。

#### 実験 D: 尤度の置き換わりの類似度依存性

**検証すること**: $y_w$ と $y_l$ が似ている組で学習するほど、選好された応答の対数確率の低下が大きい(3.6 節、Pal et al. [5]、
Razin et al. [4])。

**条件**: 除外後の候補の組全体を、課題(4 通り)× $\lvert \Delta r^* \rvert$ の区間(候補の分位点で 4 区間)の 16 の層に分け、
各層で応答の文字列の正規化編集距離の小さい順に並べて、先頭の $\lfloor n_{\mathrm{layer}} / 4 \rfloor$ 組を **近い組**、末尾の同じ数を
**遠い組** とする($n_{\mathrm{layer}}$ は層の大きさ)。層ごとの件数が一致するので(アサーションで確かめる)、課題と真の報酬の差の
分布が両データセットで揃う。それぞれのデータセットで、DPO・確率的なラベル(同じシードのラベルの行列の該当する行)・
標準の $\beta$・$T$ ステップで学習する。

**対比量**: $E = \frac{1}{S_D} \sum_s \left( \bar{\Delta}_{w,s}^{\mathrm{near}}(T) - \bar{\Delta}_{w,s}^{\mathrm{far}}(T) \right)$(近い組 − 遠い組)。

**標準偏差の導出**: $\sigma_E^2 = s_E^2 / S_D + \mathrm{Var}_{\mathrm{boot}}(E)$。$s_E$ はシードごとの差の標本標準偏差。
$\mathrm{Var}_{\mathrm{boot}}$ は、各層の中で近い組・遠い組をそれぞれ復元抽出するブートストラップ(層ごとの件数を保つ。
2 つのデータセットの組は異なるので、組どうしの対応はない)の分散である。

**判定**: $E < -2\sigma_E$ なら支持、$E > 2\sigma_E$ なら反証、それ以外は判定不能。

**前提条件**: 学習の成立(両データセット・全シード)。**PD(マージンの拡大)**: 両データセットの全シードで $\bar{h}(T) > 0$。

**作用点の記述**: 類似度が作用するのは、$y_w$ と $y_l$ の勾配の共有の度合いである(3.6 節)。Razin et al. の CHES スコア
(隠れ状態の類似度)がそれに近い量であり、文字列の編集距離はその **代理** にすぎない。両データセットの CHES スコア
(参照方策の隠れ状態で計算し、事例の向きで重み付けした平均。長さで正規化した版も)を診断量として併記する。応答の長さの
分布(近い組は長さが揃いやすい)と正規化編集距離の平均も診断量として比べる。

### 6.2 パイロットによる設定の決定

本番実行の前に、ノートブックの外(ローカルの Apple M4 の MPS、スクリプトで実行)で、本番と同じスケールのパイロットを行った。
以下はその記録である。パイロットの入力(学習用: 乱数シード`18998`の 4096 入力、評価用: 乱数シード`18997`の入力)は本番の入力と
重複しない(5.4 節でアサーションにより確認)。**パイロットで見た量は、参照方策の応答の組の統計(真の報酬の差・編集距離)、
学習の損失、KL ダイバージェンスの推定値、実行時間だけである。学習した方策の真の報酬と、選好された応答の対数確率の変化
(実験 C・D の量)、実験 B のマージンの伸びは計算していない。**

#### パイロット 1: 選好の組の統計と $K$

参照方策から、パイロットの学習用のプロンプト 4096 個について 2 応答ずつ抽選した(temperature 1.0)。

| 量 | 値 |
|---|---|
| 除外後に残る組の割合($y_1 = y_2$ 0.0037、同一でない応答の同点 0.0212 を除外) | 0.9751 |
| $\kappa$(除外後の $\lvert \Delta r^* \rvert$ の中央値 0.0889) | 12.36 |
| 正規化編集距離の分位点(5%・25%・50%・75%・95%) | 0.297・0.480・0.571・0.640・0.724 |
| 応答の平均トークン数 | 12.8 |

$\mathbb{E}[\hat{p} \in (0, 1)] = \frac{1}{N}\sum_i (1 - p_i^K - (1 - p_i)^K)$($p_i = \sigma(\kappa \Delta r^*_i)$)は、$K = 1, 2, 4, 8, 16$ で
それぞれ 0、0.327、0.588、0.757、0.852 だった。目標 0.5 以上となる最小の 2 のべきとして **$K = 4$** とした(6.1 節の規則)。

#### パイロット 2: 実行時間と実装の変更

DPO の 1 ステップ(32 事例、64 系列)は MPS で約 0.13 秒(事例の数にほぼ比例)、評価用のサンプリング(4096 応答)は約 12 秒、
その対数確率の計算(方策と参照方策)は約 6 秒だった。最初の実装では、017 の`gather_response_log_probs()`に全位置の logits
(系列長 × 語彙 8192)を渡しており、4096 系列の対数確率の計算に 25 秒かかった(64 事例のバッチではメモリの逼迫で 1 ステップ
1.3 秒)。応答の位置の隠れ状態だけを語彙へ射影する`sequence_log_prob_sums()`に替えて 6 秒になった(損失の値は変わらない。
5.6 節で照合する)。学習データの大きさ($N = 2048$、$K = 4$、3 エポック)とステップ数($T = 768$)は、この実行時間から
1 セッションの予算に収まるように決めた(6.3 節の見積もり)。
6.3・6.4 節のセルを本番の水準(`SMOKE_TEST = False`)でローカルの MPS で実行した見積もりは、段階 0 が 189.0 分、段階 3 が
128.0 分(崩壊の領域の観察の 2 学習を含む。いずれも MPS 基準で予算を超え、学習の前に停止した)だった。017 の本番での T4 と MPS の比(学習で約 2.1 倍、生成で
約 1.5 倍速い)で換算すると、T4 では段階 0 が約 97 分の見込みである。本番の段階は、本番の冒頭の T4 での計測のみから選ぶ。

#### パイロット 3: 学習率

**規則**(このパイロットの前に決めた): DPO・確率的なラベル(シード 0)・$\beta = 0.1$・本番と同じデータ量とステップ数で
学習率の格子を学習し、**非有限値がなく、学習の最後の 10% の平均損失が学習前の値(DPO は $\log 2$)を下回る** 学習率のうち、
最後の 10% の平均損失が最小のものを選ぶ。IPO($\tau = 0.1$)は、選んだ学習率で同じ条件を満たすことを確かめる。学習の損失のみを
見る規則である。$\beta = 0.1$ はこの時点の標準の $\beta$ の予定値である(パイロット 4 の後に改めた。下記)。

| 学習率 | DPO の損失(最初の 10% → 最後の 10%) | IPO の損失(同) |
|---|---|---|
| $10^{-4}$ | 0.6931 → 0.6527 | 24.99 → 22.91 |
| $3 \times 10^{-4}$ | 0.6929 → 0.6356 | 24.96 → 21.59 |
| $10^{-3}$ | 0.6919 → **0.6170** | 24.87 → 20.62 |
| $3 \times 10^{-3}$ | 0.6941 → 0.6247(途中で 0.7405 に上昇) | 24.83 → 22.52(途中で 28.12 に上昇) |
| $10^{-2}$ | 0.7015 → 0.8833($\log 2$ を上回る) | 35.84 → 180.8 |

**学習率 $10^{-3}$** を選んだ($10^{-4}$〜$10^{-3}$ で損失は学習率とともに単調に下がり、$3 \times 10^{-3}$ 以上で上昇と発散が
現れた)。

**標準の $\beta$ の規則を改めた後も、学習率は維持した(選び直さない)。** 学習率は旧い標準の $\beta = 0.1$ での学習の損失で選んだ
ものである。選択の規則は最適化の安定性(非有限値・損失の上昇・発散)に関するもので、改めた規則のもとでの標準の $\beta$ の付近
(パイロット 4 の $\beta = 0.4$〜0.8)でも、学習の最後の 10% の損失は 0.584〜0.587 と学習前の $\log 2$ から正常に下がっていた
(パイロット 4 の表)。

#### パイロット 4: $\beta$ の較正、標準の $\beta$ の決め方の変更と、実験 A の水準の規則の改訂

較正を、パイロットの入力で本番と同じスケールで行った(KL のみを測り、真の報酬は計算していない。較正用のプロンプトは
パイロットの評価用の入力 32 × 4 課題、各 8 応答)。この表は **旧い格子(公比 2、0.003125〜3.2)で測った記録** である。

| $\beta$ | 3.2 | 1.6 | 0.8 | 0.4 | 0.2 | 0.1 | 0.05 | 0.025 | 0.0125 | 0.00625 | 0.003125 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $\widehat{\mathrm{KL}}$(nats) | 0.023 | 0.142 | 0.303 | 1.71 | 49.2 | 55.7 | 221 | 242 | 249 | 258 | 256 |
| 学習の最後の 10% の損失 | 0.611 | 0.592 | 0.584 | 0.587 | 0.600 | 0.617 | 0.628 | 0.629 | 0.638 | 0.652 | 0.666 |

- **旧規則での実験 A の水準**: $\widehat{\mathrm{KL}} \ge 0.1$ nats となる最大の $\beta$ を $\beta_1$ とする旧規則では、パイロットの値で
  $\beta_1 = 1.6$、水準は $1.6, 0.8, 0.4, 0.2, 0.1, 0.05$ になる。
- **標準の $\beta$ の決め方の変更**: 当初は実験 B〜D の標準の $\beta$ を DPO の原論文でよく使われる 0.1 に固定する予定だった。
  この設定では KL が約 56 nats に達し、方策が参照方策の近傍から大きく外れるので、6.1 節の規則(大きい $\beta$ から見て
  $\widehat{\mathrm{KL}} \le 2$ nats が続く最小の格子点。パイロットの値では 0.4)に改めた。**根拠は KL の大きさのみであり、
  実験 B〜D の対比量(マージンの伸び・選好された応答の対数確率の変化)や真の報酬は計算していない。** 改めたのは本番実行の前である。
- **実験 A の水準の規則の改訂**(本番実行の前): $\widehat{\mathrm{KL}}$ は $\beta = 0.4$ の 1.71 nats から $\beta = 0.2$ の 49.2 nats へ
  1 オクターブで急増した。公比 2 の旧規則では、6 水準のうち後半の 3 水準がこの領域に入り、後半の傾き $g_2$ が、有限の選好データへの
  過適合による緩やかな過最適化(3.7 節)ではなく、方策の崩壊という別の現象で決まってしまう。検証したい主張が表れうる範囲
  (KL ダイバージェンスが 0 から崩壊の手前まで)に 6 水準を収めるため、較正の格子を公比 $\sqrt{2}$ の
  $\{3.2 \times 2^{-k/2} : k = 0, \dots, 10\}$(3.2〜0.1、学習の回数は旧格子と同じ 11)に改め、水準を $\beta_6$ = 標準の $\beta$ から
  公比 $\sqrt{2}$ で上へ 6 点とする規則に改めた(6.1 節)。崩壊の領域は、判定に含めない観察(シード 0 のみ)として別に記録する。
  **根拠は KL ダイバージェンスの値と水準の刻みの妥当性のみであり、真の報酬と実験 B〜D の対比量は計算していない。**
- **新規則をパイロットの値に当てはめると**: 旧格子の点(3.2, 1.6, 0.8, 0.4, 0.2, 0.1)は新しい格子に含まれる。
  $\widehat{\mathrm{KL}}(0.4) = 1.71 \le 2$、$\widehat{\mathrm{KL}}(0.2) = 49.2 > 2$ なので、標準の $\beta$(= $\beta_6$)は、間の格子点
  $0.4 / \sqrt{2} \approx 0.283$ の KL(パイロットでは測っていない)によって 0.4 か 0.283 のどちらかになる。0.4 なら水準は
  $2.26, 1.6, 1.13, 0.8, 0.566, 0.4$、観察は $0.2$ と $0.1$、0.283 なら水準は $1.6, 1.13, 0.8, 0.566, 0.4, 0.283$、観察は
  $0.141$ と $0.1$(格子の下端)になる。確定は本番の較正に委ねる。

### 6.3 スケーリングの計測と外挿(1 セッションの見積もり)

本番でデータ量がスモークテストの何倍にもなる重い処理について、3 点のデータ量で実行時間を実測し、
$\log t = \log a + b \log n$ をあてはめてべき指数 $b$ を推定し、本番のデータ量へ外挿する。本番では、この計測を **本番の
実行の冒頭に T4 上で** 行い、その値のみから削る段階を選ぶ(6.4 節)。外挿値と、最大の計測点の実測値を比例で伸ばした値の
大きいほうを見積もりとする(固定費があると $b < 1$ となり、外挿値が過小になりうるため)。

| 処理 | 計測するデータ量 | 外挿先(本番) | 本番での回数 |
|---|---|---|---|
| 選好の組の生成(サンプリング) | 組の数 $N_{\mathrm{cand}}/32, /16, /8$ | $N_{\mathrm{cand}} + P$(候補の組と評価用の組) | 2(6.5 節と 6.6 節の再生成) |
| 真の報酬の計算 | 応答の数(上で生成した応答の 1/4・1/2・全部) | 候補と評価用の組の応答 $2(N_{\mathrm{cand}} + P)$ | 2 |
| 正規化編集距離 | 組の数(同上) | $N_{\mathrm{cand}} + P$ | 1 |
| 参照方策の対数確率の事前計算 | 系列の数 $N/2, N, 2N$ | 実験 A〜C と D のデータ・評価用の組 $6N + 2P$ | 1(キャッシュ)+ 6.6 節の再計算 |
| DPO / IPO の学習 | ステップ数 $T/16, T/8, T/4$ | $T$ | 学習の数(段階で決まる) |
| 学習用の組の対数比の評価 | 組の数 $N/4, N/2, N$ | $N$ | 学習の数 × 2($T/2$ と $T$) |
| 方策の評価(サンプリング・対数確率・真の報酬) | プロンプトの数 $P/4, P/2, P$(各 $m$ 応答) | $P$(較正は較正用のプロンプトの数に比例で換算) | 学習の数 |
| CHES スコア | 組の数 $N/4, N/2, N$ | 近い組と遠い組 $\approx 2N$ | 1 |

**学習の数**: 較正 11(格子の点の数)+ 崩壊の領域の観察 2(段階によらない)+ 実験 A $6 S_A$ + 実験 B $4 S_B$ + 実験 D $2 S_D$。実験 B の条件 2 は実験 A と共有されうるが、
標準の $\beta$ が実験 A の水準に含まれるかは較正で決まるので、見積もりでは共有しないとみなす(安全側)。6.6 節のキャッシュの確認の
短い学習と、それ以前のセル(データの準備・参照方策の照合・ハーネス・決定的な実行の確認)とスケーリングの計測自体は実測値を使う。

生成の計測は、終端記号で止めない(上限の 32 トークンまで生成する。学習した方策が長い応答を出す場合にも見積もりが過小に
ならない安全側)。計測専用のプロンプト(乱数シード`18099`)を使い、本番のプロンプトに触れない。**スモークテストの出力の段階の
見積もりは、スモークテストの計測点が小さく 1 バッチの固定費が支配的なので、T4 での本番の目安にならない。**


```python
NEVER_STOP = "\x00"  # 復号した文字列に現れないので、上限の 32 トークンまで生成する(安全側)


def fit_and_extrapolate(label: str, sizes, times, target: float) -> float:
    fit = fit_power_law_exponent(sizes, times)
    extrapolated = fit.coefficient * target**fit.exponent
    proportional = times[-1] * target / sizes[-1]
    detail = ", ".join(f"n={n:,}: {t:.2f}s" for n, t in zip(sizes, times, strict=True))
    print(
        f"[{label}] {detail} -> b={fit.exponent:.3f}(標準誤差 {fit.exponent_stderr:.3f}), "
        f"R^2={fit.r_squared:.4f}, n={target:,.0f} での外挿値 {extrapolated:.1f}s、比例 {proportional:.1f}s"
    )
    return max(extrapolated, proportional)


_t0_scaling = time.time()
_prod_n = PROD_CFG["NUM_MAIN_PAIRS"]
_prod_cand = PROD_CFG["NUM_CANDIDATE_PAIRS"]
_prod_prompts = PROD_CFG["NUM_EVAL_INPUTS"] * TASK_COUNT
_prod_cal_prompts = PROD_CFG["NUM_CALIBRATION_INPUTS"] * TASK_COUNT
_prod_steps = PROD_CFG["TRAIN_STEPS"]
_rng_timing = np.random.default_rng(TIMING_SEED)
_timing_examples = build_training_examples(
    sample_word_sequences(_rng_timing, 4096, WORD_VOCABULARY, MIN_WORDS, MAX_WORDS), _rng_timing
)
_timing_prompts = [tokenizer.encode(e.prompt(False)) for e in _timing_examples]
ESTIMATE: dict[str, float] = {}

# --- 選好の組の生成(組ごとに 2 応答) ---
_sizes = [NUM_CANDIDATE_PAIRS // 32, NUM_CANDIDATE_PAIRS // 16, NUM_CANDIDATE_PAIRS // 8]
if (
    not SMOKE_TEST
):  # 最小の計測点の応答数が 1 バッチ以上(1 バッチの固定費が支配的だと比例の見積もりが過大になる)
    assert 2 * _sizes[0] >= GENERATION_BATCH_SIZE, _sizes
_timing_pairs = {}


def _time_pairs(n: int) -> None:
    out = sample_responses(
        reference_policy,
        [p for p in _timing_prompts[:n] for _ in range(2)],
        TIMING_SEED,
        NEVER_STOP,
    )
    _timing_pairs.update({"prompts": _timing_prompts[:n], "first": out[0::2], "second": out[1::2]})


ESTIMATE["pairs"] = fit_and_extrapolate(
    "選好の組の生成(組の数)",
    _sizes,
    [timed_call(lambda n=n: _time_pairs(n)) for n in _sizes],
    _prod_cand + _prod_prompts,
)

# --- 真の報酬(キャッシュを使わずに計算する: 安全側)と正規化編集距離 ---
_texts = [tokenizer.decode(r) for r in _timing_pairs["first"] + _timing_pairs["second"]]
_answers = [_timing_examples[i % len(_timing_pairs["first"])].answer for i in range(len(_texts))]
_sizes = [len(_texts) // 4, len(_texts) // 2, len(_texts)]
ESTIMATE["true_reward"] = fit_and_extrapolate(
    "真の報酬の計算(応答の数)",
    _sizes,
    [
        timed_call(
            lambda n=n: [
                compute_true_reward(t, a) for t, a in zip(_texts[:n], _answers[:n], strict=True)
            ]
        )
        for n in _sizes
    ],
    2 * (_prod_cand + _prod_prompts),
)
_half = len(_texts) // 2
_sizes = [_half // 4, _half // 2, _half]
ESTIMATE["distance"] = fit_and_extrapolate(
    "正規化編集距離(組の数)",
    _sizes,
    [
        timed_call(
            lambda n=n: [
                normalized_edit_distance(a, b)
                for a, b in zip(_texts[:n], _texts[_half : _half + n], strict=True)
            ]
        )
        for n in _sizes
    ],
    _prod_cand + _prod_prompts,
)

# --- 参照方策の対数確率の事前計算(系列の数) ---
_seq_prompts = _timing_pairs["prompts"] * 2
_seq_responses = _timing_pairs["first"] + _timing_pairs["second"]
_sizes = [len(_seq_prompts) // 4, len(_seq_prompts) // 2, len(_seq_prompts)]
ESTIMATE["reference_log_probs"] = fit_and_extrapolate(
    "参照方策の対数確率(系列の数)",
    _sizes,
    [
        timed_call(lambda n=n: reference_log_probs(_seq_prompts[:n], _seq_responses[:n], None))
        for n in _sizes
    ],
    6 * _prod_n + 2 * _prod_prompts,
)

# --- 学習(ステップ数、1 つの学習あたり)。計測用の組を N 組に切り詰めた決定的なラベルで学習する ---
_n_timing = min(NUM_MAIN_PAIRS, len(_timing_pairs["first"]))
_timing_labels = np.tile(np.arange(_n_timing)[:, None] % 2 == 0, (1, NUM_DRAWS))
_timing_dataset = build_dataset(_timing_pairs, np.arange(_n_timing), _timing_labels, cache=None)
_sizes = [max(2, TRAIN_STEPS // 16), max(4, TRAIN_STEPS // 8), max(8, TRAIN_STEPS // 4)]
ESTIMATE["train"] = fit_and_extrapolate(
    "DPO の学習(ステップ数)",
    _sizes,
    [
        timed_call(
            lambda n=n: run_training(_timing_dataset, "dpo", CHECK_BETA, 0, "none", num_steps=n)
        )
        for n in _sizes
    ],
    _prod_steps,
)

# --- 学習用の組の対数比の評価(組の数)と CHES スコア(組の数) ---
_sizes = [_n_timing // 4, _n_timing // 2, _n_timing]


def _subset(n: int) -> dict:
    return build_dataset(_timing_pairs, np.arange(n), _timing_labels[:n], cache=None)


_subsets = {n: _subset(n) for n in _sizes}
ESTIMATE["pair_eval"] = fit_and_extrapolate(
    "学習用の組の対数比の評価(組の数)",
    _sizes,
    [timed_call(lambda n=n: pair_log_ratios(reference_policy, _subsets[n])) for n in _sizes],
    _prod_n,
)
ESTIMATE["ches"] = fit_and_extrapolate(
    "CHES スコア(組の数)",
    _sizes,
    [
        timed_call(
            lambda n=n: compute_ches_statistics(
                reference_policy,
                _subsets[n]["prompts"],
                _subsets[n]["first"],
                _subsets[n]["second"],
                device,
                LOG_PROB_BATCH_SIZE,
            )
        )
        for n in _sizes
    ],
    2 * _prod_n,
)

# --- 方策の評価(プロンプトの数、各 m 応答。終端記号で止めない) ---
_eval_timing_examples = build_evaluation_examples(
    sample_word_sequences(
        _rng_timing, max(1, NUM_EVAL_INPUTS), WORD_VOCABULARY, MIN_WORDS, MAX_WORDS
    ),
    _rng_timing,
)
_eval_timing_prompts = [tokenizer.encode(e.prompt(False)) for e in _eval_timing_examples]
_sizes = [NUM_EVAL_PROMPTS // 4, NUM_EVAL_PROMPTS // 2, NUM_EVAL_PROMPTS]
ESTIMATE["policy_eval"] = fit_and_extrapolate(
    "方策の評価(プロンプトの数 x m)",
    _sizes,
    [
        timed_call(
            lambda n=n: evaluate_policy(
                reference_policy,
                _eval_timing_prompts[:n],
                _eval_timing_examples[:n],
                TIMING_SEED,
                NEVER_STOP,
            )
        )
        for n in _sizes
    ],
    _prod_prompts,
)
ESTIMATE["calibration_eval"] = (
    ESTIMATE["policy_eval"] * _prod_cal_prompts / _prod_prompts
)  # 比例で換算(真の報酬の分は過大で安全側)
SCALING_SECONDS = time.time() - _t0_scaling
print(f"スケーリングの計測自体: {SCALING_SECONDS:.1f}s")
```

    [選好の組の生成(組の数)] n=256: 4.55s, n=512: 4.18s, n=1,024: 5.01s -> b=0.070(標準誤差 0.111), R^2=0.2844, n=8,704 での外挿値 5.6s、比例 42.6s
    [真の報酬の計算(応答の数)] n=512: 0.38s, n=1,024: 0.78s, n=2,048: 0.96s -> b=0.680(標準誤差 0.215), R^2=0.9089, n=17,408 での外挿値 4.5s、比例 8.2s
    [正規化編集距離(組の数)] n=256: 0.27s, n=512: 0.57s, n=1,024: 1.09s -> b=1.009(標準誤差 0.038), R^2=0.9986, n=8,704 での外挿値 9.6s、比例 9.3s
    [参照方策の対数確率(系列の数)] n=512: 0.22s, n=1,024: 0.33s, n=2,048: 0.62s -> b=0.761(標準誤差 0.083), R^2=0.9883, n=13,312 での外挿値 2.5s、比例 4.0s
    [DPO の学習(ステップ数)] n=48: 3.82s, n=96: 6.42s, n=192: 11.54s -> b=0.797(標準誤差 0.028), R^2=0.9988, n=768 での外挿値 34.4s、比例 46.2s
    [学習用の組の対数比の評価(組の数)] n=256: 0.17s, n=512: 0.33s, n=1,024: 0.62s -> b=0.921(標準誤差 0.004), R^2=1.0000, n=2,048 での外挿値 1.2s、比例 1.2s
    [CHES スコア(組の数)] n=256: 0.16s, n=512: 0.31s, n=1,024: 0.61s -> b=0.954(標準誤差 0.021), R^2=0.9995, n=4,096 での外挿値 2.3s、比例 2.4s
    [方策の評価(プロンプトの数 x m)] n=128: 4.96s, n=256: 7.70s, n=512: 10.53s -> b=0.543(標準誤差 0.053), R^2=0.9907, n=512 での外挿値 10.8s、比例 10.5s
    スケーリングの計測自体: 68.3s


### 6.4 削る段階の選択

6.3 節の外挿値から、削る段階ごとに 1 セッション全体の見積もりを出し、予算(T4 で 120 分)に収まる最小の段階を選ぶ(6.1 節)。
**選択は見積もりのみに基づき、どの実験の結果も参照しない。** この時点では、選好の組の生成も学習も行っていない。


```python
PER_RUN_SECONDS = ESTIMATE["train"] + 2 * ESTIMATE["pair_eval"] + ESTIMATE["policy_eval"]
CALIBRATION_RUN_SECONDS = (
    ESTIMATE["train"] + 2 * ESTIMATE["pair_eval"] + ESTIMATE["calibration_eval"]
)


def num_full_runs(stage: dict) -> int:
    # 実験 A(6 水準)・実験 B(4 条件、条件 2 を共有しないとみなす: 安全側)・実験 D(2 データセット)
    return NUM_A_LEVELS * stage["NUM_SEEDS_A"] + 4 * stage["NUM_SEEDS_B"] + 2 * stage["NUM_SEEDS_D"]


COMMON_ESTIMATE = {
    "選好の組の生成(外挿 x 2: 6.5 節と 6.6 節の再生成)": 2 * ESTIMATE["pairs"],
    "真の報酬の計算(外挿 x 2)": 2 * ESTIMATE["true_reward"],
    "正規化編集距離(外挿)": ESTIMATE["distance"],
    "参照方策の対数確率(外挿 x 2: キャッシュと 6.6 節の再計算)": 2
    * ESTIMATE["reference_log_probs"],
    "CHES スコア(外挿)": ESTIMATE["ches"],
    f"beta の較正({len(BETA_GRID)} 学習)": len(BETA_GRID) * CALIBRATION_RUN_SECONDS,
    f"崩壊の領域の観察({len(COLLAPSE_OFFSETS)} 学習、段階によらない)": len(COLLAPSE_OFFSETS)
    * PER_RUN_SECONDS,
    "キャッシュの確認の短い学習と評価(外挿の比例、6.6 節)": 2
    * (ESTIMATE["train"] * 24 / _prod_steps + ESTIMATE["policy_eval"] + 2 * ESTIMATE["pair_eval"]),
    "データの準備(5.4 節、実測)": DATA_SECONDS,
    "参照方策の照合(5.5 節、実測)": REFCHECK_SECONDS,
    "ハーネスの確認(5.6 節、実測)": HARNESS_SECONDS,
    "スケーリングの計測自体(実測)": SCALING_SECONDS,
}


def estimate_stage(stage: dict) -> dict:
    runs = num_full_runs(stage)
    return {**COMMON_ESTIMATE, f"学習と評価({runs} 学習)": runs * PER_RUN_SECONDS}


class StageBudgetExceededError(RuntimeError):
    pass


def select_stage(totals: dict[int, float], budget_seconds: float) -> int:
    for stage in sorted(totals):
        if totals[stage] <= budget_seconds:
            return stage
    raise StageBudgetExceededError(
        f"段階 {max(totals)} でも見積もり {totals[max(totals)] / 60:.1f} 分が予算 {budget_seconds / 60:.1f} 分を超える"
    )


STAGE_ESTIMATES = {k: estimate_stage(v) for k, v in STAGES["prod"].items()}
STAGE_TOTALS = {k: sum(v.values()) for k, v in STAGE_ESTIMATES.items()}
print(f"1 つの学習と評価の見積もり: {PER_RUN_SECONDS:.1f}s(較正は {CALIBRATION_RUN_SECONDS:.1f}s)")
print(f"--- 段階 0 の本番の見積もりの内訳({device} 基準)---")
for _k, _v in STAGE_ESTIMATES[0].items():
    print(f"  {_k}: {_v:,.1f}s")
print(
    f"\n--- 削る段階ごとの 1 セッション全体の見積もり({device} 基準、予算 {SESSION_BUDGET_SECONDS / 60:.0f} 分)---"
)
for _k, _total in STAGE_TOTALS.items():
    _stage = STAGES["prod"][_k]
    print(
        f"  段階 {_k}(S_D = {_stage['NUM_SEEDS_D']}、S_A = {_stage['NUM_SEEDS_A']}、S_B = {_stage['NUM_SEEDS_B']}、"
        f"{num_full_runs(_stage)} 学習): {_total:,.1f}s = {_total / 60:.1f} 分(予算の {_total / SESSION_BUDGET_SECONDS:.1%})"
    )
assert all(STAGE_TOTALS[k] >= STAGE_TOTALS[k + 1] for k in range(3)), (
    "段階が上がると見積もりが減るはず"
)

try:
    SELECTED_STAGE = select_stage(STAGE_TOTALS, SESSION_BUDGET_SECONDS)
    STAGE_SELECTION_MESSAGE = (
        f"予算 {SESSION_BUDGET_SECONDS / 60:.0f} 分に収まる最小の段階として、段階 {SELECTED_STAGE} を選んだ"
        f"(見積もり {STAGE_TOTALS[SELECTED_STAGE] / 60:.1f} 分、{device} 基準)"
    )
except StageBudgetExceededError as _error:
    if not SMOKE_TEST:
        print(f"\n警告: {_error}。本番の学習を始める前に停止する。計画を見直すこと。")
        raise
    SELECTED_STAGE = max(STAGE_TOTALS)
    STAGE_SELECTION_MESSAGE = f"{_error}(本番なら学習の前に停止する)。スモークテストのため停止せず、段階 {SELECTED_STAGE} で動作確認を続ける"
if FORCED_STAGE is not None:  # テスト専用の上書き(5.2 節。スモークテストでのみ有効)
    assert SMOKE_TEST
    print("\n" + "!" * 100)
    print(
        f"!!! テスト専用の上書き: AI_THEORIES_FORCE_STAGE={FORCED_STAGE} により、見積もりによる選択(段階 {SELECTED_STAGE})の"
        f"代わりに段階 {FORCED_STAGE} を使う(スモークテストのみ。この出力は通常の実行の記録ではない)"
    )
    print("!" * 100)
    STAGE_SELECTION_MESSAGE = (
        f"テスト専用の上書きで段階 {FORCED_STAGE} を強制(見積もりによる選択は段階 {SELECTED_STAGE})"
    )
    SELECTED_STAGE = FORCED_STAGE
print(f"\n段階の選択: {STAGE_SELECTION_MESSAGE}")

if SMOKE_TEST:  # 予算を人為的に小さくして、選択の規則の分岐を確かめる(予算の定数は変えない)
    _budget_between = (STAGE_TOTALS[0] + STAGE_TOTALS[1]) / 2
    assert select_stage(STAGE_TOTALS, _budget_between) == 1
    try:
        select_stage(STAGE_TOTALS, 0.5 * STAGE_TOTALS[3])
        raise AssertionError("どの段階でも超える予算で停止しなかった")
    except StageBudgetExceededError as _error:
        _stopped = str(_error)
    print(
        f"選択の規則の確認(人為的に小さい予算、確認のみ): 予算 {_budget_between / 60:.1f} 分 -> 段階 1: OK / "
        f"予算 {0.5 * STAGE_TOTALS[3] / 60:.1f} 分 -> 停止する({_stopped}): OK。予算の定数は {SESSION_BUDGET_SECONDS / 60:.0f} 分のまま"
    )
    assert SESSION_BUDGET_SECONDS == 120 * 60

# --- 選ばれた段階のシード数(以降のすべてのセルがこれを使う) ---
STAGE = STAGES[CURRENT_LEVEL_NAME][SELECTED_STAGE]
SEEDS_A = tuple(range(STAGE["NUM_SEEDS_A"]))
SEEDS_B = tuple(range(STAGE["NUM_SEEDS_B"]))
SEEDS_D = tuple(range(STAGE["NUM_SEEDS_D"]))
ALL_SEEDS = tuple(range(max(len(SEEDS_A), len(SEEDS_B), len(SEEDS_D))))
print(
    f"このノートブックで使う値(段階 {SELECTED_STAGE}、水準 {CURRENT_LEVEL_NAME!r}): 実験 A のシード {SEEDS_A}、実験 B・C のシード {SEEDS_B}、実験 D のシード {SEEDS_D}"
)
if device.type != "cuda":
    print("注意: CUDA 以外での見積もりであり、T4 での時間とは異なる")
```

    1 つの学習と評価の見積もり: 59.4s(較正は 51.3s)
    --- 段階 0 の本番の見積もりの内訳(cuda 基準)---
      選好の組の生成(外挿 x 2: 6.5 節と 6.6 節の再生成): 85.2s
      真の報酬の計算(外挿 x 2): 16.4s
      正規化編集距離(外挿): 9.6s
      参照方策の対数確率(外挿 x 2: キャッシュと 6.6 節の再計算): 8.0s
      CHES スコア(外挿): 2.4s
      beta の較正(11 学習): 564.5s
      崩壊の領域の観察(2 学習、段階によらない): 118.8s
      キャッシュの確認の短い学習と評価(外挿の比例、6.6 節): 29.3s
      データの準備(5.4 節、実測): 6.9s
      参照方策の照合(5.5 節、実測): 4.1s
      ハーネスの確認(5.6 節、実測): 1.2s
      スケーリングの計測自体(実測): 68.3s
      学習と評価(60 学習): 3,563.1s
    
    --- 削る段階ごとの 1 セッション全体の見積もり(cuda 基準、予算 120 分)---
      段階 0(S_D = 5、S_A = 5、S_B = 5、60 学習): 4,477.8s = 74.6 分(予算の 62.2%)
      段階 1(S_D = 3、S_A = 5、S_B = 5、56 学習): 4,240.3s = 70.7 分(予算の 58.9%)
      段階 2(S_D = 3、S_A = 3、S_B = 5、44 学習): 3,527.6s = 58.8 分(予算の 49.0%)
      段階 3(S_D = 3、S_A = 3、S_B = 3、36 学習): 3,052.6s = 50.9 分(予算の 42.4%)
    
    段階の選択: 予算 120 分に収まる最小の段階として、段階 0 を選んだ(見積もり 74.6 分、cuda 基準)
    このノートブックで使う値(段階 0、水準 'prod'): 実験 A のシード (0, 1, 2, 3, 4)、実験 B・C のシード (0, 1, 2, 3, 4)、実験 D のシード (0, 1, 2, 3, 4)




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/04_alignment/018_direct_preference_optimization.ipynb
