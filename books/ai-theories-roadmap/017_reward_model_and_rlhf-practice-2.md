---
title: "報酬モデルと RLHF / Reward Models and RLHF(実装・実験編 2/4)"
---

この記事は後編(実装・実験編 2/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/017_reward_model_and_rlhf-practice-1)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/017_reward_model_and_rlhf-practice-3)。

### 5.6 SFT・生成・報酬モデル・PPO のヘルパー

- `train_sft()`: 016 の条件 1 のレシピ(損失マスクあり・短い水準・学習率 $10^{-2}$・$T = 2048$・バッチサイズ 32・LoRA
  $r = 8$ を Query・Value 射影に・AdamW(重み減衰 0)・warmup 10% + cosine(最小はピークの 1%)・gradient clipping なし・
  シード 0)で、`base_model`の複製を学習する。
- `merge_lora()`: LoRA を $W = W_0 + \frac{\alpha}{r} B A$ としてベース層に畳み込み、`main`と同じ構造(同じ`state_dict`の
  キー)の`GPTLanguageModel`を作る。
- `sample_responses()`: $\pi$ から temperature 1.0(top-k・top-p なし)で終端記号まで生成する。
- `true_rewards()`: 応答を復号して $r^*$ を計算する(同じ文字列・同じ正解の組はキャッシュする)。
- `train_one_reward_model()`: シード $s$ の報酬モデルを、選好の組の先頭 $N$ 個で $T_{\mathrm{RM}}$ ステップ学習する。
  ヘッドの初期化は`torch.manual_seed(17200 + s)`、組の順序は`make_epoch_batches(N, T_RM, 32, s)`、ラベルはシード $s$ の
  抽選の先頭 $N$ 個(6.6 節)。本体は $\pi_{\mathrm{ref}}$ の複製で、全パラメータを学習する(AdamW、重み減衰 0、warmup 10% +
  cosine、gradient clipping なし)。
- `run_ppo_iterations()`: 3.7 節の PPO を指定の反復回数だけ行い、反復ごとの KL ダイバージェンスの推定値・代理報酬・
  真の報酬などを記録する。KL ダイバージェンスの推定値は、ロールアウトの応答について
  $\sum_t (\log \pi_{\mathrm{old}}(y_t \mid s_t) - \log \pi_{\mathrm{ref}}(y_t \mid s_t))$ をバッチで平均したもの
  ($\mathrm{KL}(\pi_{\mathrm{old}} \| \pi_{\mathrm{ref}})$ の不偏推定量)である。


```python
def learning_rate_schedule(peak_learning_rate: float, num_steps: int):
    return functools.partial(
        compute_warmup_cosine_learning_rate,
        warmup_steps=max(1, round(WARMUP_RATIO * num_steps)),
        total_steps=num_steps,
        peak_learning_rate=peak_learning_rate,
        min_learning_rate=peak_learning_rate * MIN_LEARNING_RATE_RATIO,
    )


def train_sft(num_steps: int, encoded: list) -> tuple[GPTLanguageModel, list, dict]:
    model = copy.deepcopy(base_model)
    torch.manual_seed(SFT_SEED)  # LoRA の A の初期化
    replaced = apply_lora(model, LORA_TARGET_MODULES, rank=LORA_RANK, alpha=LORA_ALPHA)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=SFT_LEARNING_RATE, weight_decay=0.0
    )
    history = train_instruction_tuning(
        model,
        encoded,
        make_epoch_batches(len(encoded), num_steps, SFT_BATCH_SIZE, SFT_SEED),
        optimizer,
        include_prompt_loss=False,  # 損失マスクあり(016 の条件 1)
        device=device,
        learning_rate_schedule=learning_rate_schedule(SFT_LEARNING_RATE, num_steps),
        gradient_clip_threshold=None,
    )
    return model, replaced, history


def merge_lora(lora_model: nn.Module, replaced: list) -> tuple[GPTLanguageModel, dict]:
    # LoRA を畳み込み、main と同じキーの state_dict(CPU)と、それを読み込んだ GPTLanguageModel を返す。
    merged = copy.deepcopy(lora_model)
    for name in replaced:
        merged.get_submodule(name).merge()
    state = {}
    for key, value in merged.state_dict().items():
        if ".lora_" in key:
            continue
        state[key.replace(".base_layer", "")] = value.detach().cpu().clone()
    assert set(state) == set(BASE_STATE), "マージ後の state_dict のキーが main と一致しない"
    return freeze(build_model(state)), state


def sample_responses(model: nn.Module, prompts: list, seed: int) -> list[list[int]]:
    return _sample(model, prompts, seed)


TRUE_REWARD_CACHE: dict[tuple[str, tuple[str, ...]], float] = {}


def true_rewards(responses: list, examples: list) -> np.ndarray:
    values = np.empty(len(responses), dtype=np.float64)
    for i, (response, example) in enumerate(zip(responses, examples, strict=True)):
        key = (tokenizer.decode(response), example.answer)
        if key not in TRUE_REWARD_CACHE:
            TRUE_REWARD_CACHE[key] = compute_true_reward(*key)
        values[i] = TRUE_REWARD_CACHE[key]
    return values


def train_one_reward_model(
    reference: GPTLanguageModel, pairs: dict, labels: np.ndarray, num_pairs: int, seed: int
):
    torch.manual_seed(RM_INIT_SEED_BASE + seed)
    reward_model = RewardModel(copy.deepcopy(reference), RM_HEAD_INIT_STD).to(device)
    for p in reward_model.parameters():
        p.requires_grad_(True)  # 全パラメータを学習する
    first, second = pairs["first"][:num_pairs], pairs["second"][:num_pairs]
    chosen = [
        a if label else b for a, b, label in zip(first, second, labels[:num_pairs], strict=True)
    ]
    rejected = [
        b if label else a for a, b, label in zip(first, second, labels[:num_pairs], strict=True)
    ]
    batches = make_epoch_batches(num_pairs, RM_STEPS, RM_BATCH_PAIRS, seed)
    history = train_reward_model(
        reward_model,
        pairs["prompts"][:num_pairs],
        chosen,
        rejected,
        batches,
        AdamW(list(reward_model.parameters()), lr=RM_LEARNING_RATE, weight_decay=0.0),
        device,
        learning_rate_schedule(RM_LEARNING_RATE, RM_STEPS),
    )
    history["batches_hash"] = hash_json(batches)
    return freeze(reward_model).eval(), history


def run_ppo_iterations(
    reference: GPTLanguageModel,
    reward_model: RewardModel,
    beta: float,
    num_iterations: int,
    batch_size: int = PPO_BATCH_SIZE,
) -> dict[str, list]:
    policy = copy.deepcopy(reference)
    torch.manual_seed(PPO_SEED)
    apply_lora(policy, LORA_TARGET_MODULES, rank=LORA_RANK, alpha=LORA_ALPHA)
    agent = PolicyWithValueHead(policy, value_head_init_std=0.0).to(device)
    trainable = [p for p in agent.parameters() if p.requires_grad]  # LoRA と価値ヘッド
    optimizer = AdamW(trainable, lr=PPO_LEARNING_RATE, weight_decay=0.0)
    generator = torch.Generator(device=device)
    generator.manual_seed(PPO_SEED)
    rng = np.random.default_rng(PPO_SEED)
    history: dict[str, list] = {
        key: []
        for key in (
            "kl",
            "proxy_reward",
            "true_reward",
            "response_length",
            "policy_loss",
            "value_loss",
            "clip_fraction",
            "approx_kl",
            "seconds",
            "example",
        )
    }
    for iteration in range(num_iterations):
        t0 = time.time()
        examples = PPO_EXAMPLES[iteration * batch_size : (iteration + 1) * batch_size]
        prompts = PPO_PROMPT_IDS[iteration * batch_size : (iteration + 1) * batch_size]
        responses = sample_generate_until_stop(
            agent.policy,
            prompts,
            tokenizer.decode,
            END_MARKER,
            MAX_NEW_TOKENS,
            device,
            generator,
            TEMPERATURE,
            batch_size,
        )
        token_ids, prompt_lengths, mask = (
            t.to(device) for t in collate_rollout_sequences(prompts, responses)
        )
        with torch.no_grad():
            logits, values = agent(token_ids)
            old_log_probs = gather_response_log_probs(logits, token_ids, prompt_lengths, mask)
            token_values = gather_response_values(values, prompt_lengths, mask)
            reference_log_probs = gather_response_log_probs(
                reference(token_ids), token_ids, prompt_lengths, mask
            )
            scores = reward_model(token_ids, prompt_lengths + mask.sum(dim=1) - 1)
        rewards = compute_kl_penalized_rewards(
            old_log_probs, reference_log_probs, scores, mask, beta
        )
        advantages, returns = compute_gae(rewards, token_values, mask, PPO_GAMMA, PPO_LAMBDA)
        rollout = {
            "token_ids": token_ids,
            "prompt_lengths": prompt_lengths,
            "response_mask": mask,
            "old_log_probs": old_log_probs,
            "advantages": whiten_advantages(advantages, mask),
            "returns": returns,
        }
        stats = ppo_update(
            agent,
            rollout,
            optimizer,
            PPO_EPOCHS,
            batch_size // 2,
            PPO_CLIP_EPSILON,
            PPO_VALUE_COEFFICIENT,
            rng,
        )
        history["kl"].append(
            float(((old_log_probs - reference_log_probs) * mask).sum(dim=1).mean())
        )
        history["proxy_reward"].append(float(scores.mean()))
        history["true_reward"].append(float(true_rewards(responses, examples).mean()))
        history["response_length"].append(float(mask.sum(dim=1).float().mean()))
        for key in ("policy_loss", "value_loss", "clip_fraction", "approx_kl"):
            history[key].append(stats[key])
        history["example"].append(tokenizer.decode(responses[0]))
        history["seconds"].append(time.time() - t0)
    return history
```

### 5.7 決定的な実行の確認

`torch.use_deterministic_algorithms(True)`のもとで、SFT の学習・サンプリング・報酬モデルの学習(全パラメータ、埋め込みを
含む)とスコアリング・PPO の 1 反復を数ステップずつ 2 回実行し、決定的な実装を持たない演算で例外にならないこと、2 回の
結果が bit 単位で一致することを確かめる。

**MPS(Apple Silicon)での例外**: MPS の埋め込み層の逆伝播(同じトークンの勾配の累積)は決定的でなく、しかも
`torch.use_deterministic_algorithms(True)`で検出されない(第 1 段階のローカルの確認で判明した)。報酬モデルは埋め込みも
学習するので、MPS では報酬モデルの学習の結果が実行ごとにわずかに変わる。本番の CUDA では、報酬モデルの学習についても
bit 単位の一致をアサーションで確かめる。MPS で決定的な実装を持たない`gather`・高度なインデックスの逆伝播は、値が同じ
one-hot の積に置き換えた(`src/models/reward_model.py`・`src/training/ppo.py`)。


```python
def _short_reward_model_run():
    prompts = [p for p in EVAL_PROMPT_IDS[:8] for _ in range(4)]
    responses = sample_responses(base_model, prompts, 1)
    pairs = {"prompts": prompts[0::2], "first": responses[0::2], "second": responses[1::2]}
    return train_one_reward_model(base_model, pairs, np.arange(16) % 2 == 0, num_pairs=16, seed=0)


def _short_determinism_run(reward_model: RewardModel) -> tuple:
    sft, _, sft_history = train_sft(4, SFT_TRAIN_ENCODED[: 4 * SFT_BATCH_SIZE])
    prompts = [p for p in EVAL_PROMPT_IDS[:8] for _ in range(4)]
    responses = sample_responses(sft, prompts, 1)
    scores = score_sequences(
        reward_model, [p + r for p, r in zip(prompts, responses, strict=True)], device
    )
    ppo_history = run_ppo_iterations(
        base_model, reward_model, beta=0.1, num_iterations=1, batch_size=8
    )
    ppo_history.pop("seconds")
    return sft_history["loss"], responses, scores, ppo_history


_saved_rm_steps = RM_STEPS
RM_STEPS = 4  # 確認のためだけに短くする(このセルの最後で元に戻す)
(_rm_first, _rm_history_first), (_rm_second, _rm_history_second) = (
    _short_reward_model_run(),
    _short_reward_model_run(),
)
RM_STEPS = _saved_rm_steps
# SFT・サンプリング・スコアリング・PPO: 同じ報酬モデルで 2 回実行して bit 単位で一致すること(全デバイス)
_first_run, _second_run = _short_determinism_run(_rm_first), _short_determinism_run(_rm_first)
assert _first_run[0] == _second_run[0] and _first_run[1] == _second_run[1]
assert np.array_equal(_first_run[2], _second_run[2]) and _first_run[3] == _second_run[3]
# 報酬モデルの学習(埋め込みを含む全パラメータ): 2 回の学習の重みが bit 単位で一致すること
_rm_max_diff = max(
    float((a - b).abs().max())
    for a, b in zip(_rm_first.parameters(), _rm_second.parameters(), strict=True)
)
if device.type == "mps":
    # MPS の埋め込みの逆伝播(同じトークンの勾配の累積)は決定的でなく、use_deterministic_algorithms でも
    # 検出されない(6.2 節の注記)。本番の CUDA では下の assert で bit 単位の一致を確かめる。
    print(
        f"注意(MPS のみ): 報酬モデルの学習の 2 回の重みの差の最大値 {_rm_max_diff:.2e}(埋め込みの逆伝播が決定的でない)"
    )
else:
    assert _rm_max_diff == 0.0 and _rm_history_first["loss"] == _rm_history_second["loss"], (
        _rm_max_diff
    )
    print(
        "報酬モデルの学習(全パラメータ、4 ステップ)を 2 回実行して、損失・重みが bit 単位で一致: OK"
    )
del _rm_first, _rm_second
print(
    f"決定的な実行({device}、torch.are_deterministic_algorithms_enabled() = "
    f"{torch.are_deterministic_algorithms_enabled()}): SFT 4 ステップ・サンプリング・スコアリング・PPO 1 反復を"
    "同じ報酬モデルで 2 回実行して、損失・生成・スコア・PPO の記録が bit 単位で一致。例外なし: OK"
)
```

    報酬モデルの学習(全パラメータ、4 ステップ)を 2 回実行して、損失・重みが bit 単位で一致: OK
    決定的な実行(cuda、torch.are_deterministic_algorithms_enabled() = True): SFT 4 ステップ・サンプリング・スコアリング・PPO 1 反復を同じ報酬モデルで 2 回実行して、損失・生成・スコア・PPO の記録が bit 単位で一致。例外なし: OK


## 6. 実験 / Experiments

### 6.1 実験宣言セル: 共通の設定・検証すること・判定基準・前提条件

**この節の内容は本番実行の前に確定させ、結果を見た後に変更しない。** パイロット(6.2 節)を受けて決めた設定は、
本番実行の前に決めたものであり、6.2 節に経緯を記録した。実験 A の対比量はスモークテストの後(本番実行の前)に改訂した
(6.2 節の改訂 1。旧基準・新基準・理由・旧基準によるスモークテストの結果を記録した)。

#### 共通の設定

- **参照方策 $\pi_{\mathrm{ref}}$**: 008 のモデル(`kojikojiprg/ai-theories-small-gpt-en`の`main`)から、016 の条件 1 の
  レシピ(損失マスクあり・短い水準・016 と同一の学習データ 65,536 事例・$T = 2048$・シード 0。5.6 節)で SFT し、LoRA を
  マージした重み。報酬モデルの本体と PPO の方策の初期値もこの重みとする。
- **サンプリング**: temperature 1.0、top-k・top-p なし、終端記号`### End`まで(上限 32 トークン)。best-of-n の参照方策を
  厳密に $\pi_{\mathrm{ref}}$ にするため、分布を切り詰めない。
- **プロンプト**: 評価用(入力 $|X| = 64$ と 4 課題の直積、$P = 256$)・報酬モデルの学習用($N_A = 65536$)・PPO 用は互いに素
  (5.4 節)。
- **応答プール**: 評価用の各プロンプトについて、$\pi_{\mathrm{ref}}$ から $M = 512$ 個の応答を 1 度だけ抽選する(乱数シード
  `17018`)。**全シード・全条件で同じプールを使う。**
- **選好の組**: 報酬モデルの学習用の各プロンプトについて、$\pi_{\mathrm{ref}}$ から 2 応答を抽選する(乱数シード`17019`)。
  データ量の水準 $N$ の報酬モデルは、先頭 $N$ 組を使う(入れ子)。
- **$\kappa$**: 選好の組($N_A$ 組すべて)の $\Delta r^*$ から、3.3 節の式で決める(学習した報酬モデルの結果を見ずに決まる)。
- **ラベル**: シード $s$ ごとに、$N_A$ 組すべてのラベルを尺度 $\kappa$ の Bradley-Terry モデルから抽選する(乱数シード
  $17300 + s$)。水準 $N$ ではその先頭 $N$ 個を使う。
- **報酬モデル**: 本体は $\pi_{\mathrm{ref}}$ の複製で全パラメータを学習する。学習率 $10^{-4}$(6.2 節)、バッチ 32 組、
  ステップ数は全水準で $T_{\mathrm{RM}} = N_A / 32 = 2048$(標準の水準で 1 エポック、水準 $N$ では $N_A / N$ エポック)。
  **学習の計算量(ステップ数)を全水準で揃える。** このため水準 $N$ では、同じ組を $N_A / N$ エポック反復して学習する。
  実験 C が操作する変数は「同じ計算量のもとでの、異なる組の数」であり、組の数の減少と反復回数の増加(過適合)の効果は
  分離できない。
- **シード**: 変動させるのは報酬モデルのヘッドの初期化・組の順序とラベルの抽選。実験 A・B は $S_{\mathrm{AB}}$ 個、
  実験 C は $S_{\mathrm{C}}$ 個で、削る段階で決まる(段階 0 で両方 5)。シードは先頭から $0, 1, \dots$ を使う。
- **標準の水準の共有**: 実験 C の最大の水準 $N_A$ は、実験 A・B の報酬モデルと同じ設定なので、同じ学習を重複して実行せず
  共有する($S_{\mathrm{C}} \le S_{\mathrm{AB}}$ なので実験 C のシードは実験 A・B のシードの先頭部分)。
- **決定的な実行**: `torch.use_deterministic_algorithms(True)`と`CUBLAS_WORKSPACE_CONFIG`を設定する(5.1 節)。

#### 記号

- $x$: 入力(単語の列)。$X$: 評価用の入力の集合($|X| = 64$)。$K = 4$: 課題の数。$P = K|X| = 256$: 評価用の
  プロンプトの数。$p$: 評価用のプロンプト。
- $M = 512$: プロンプトごとの応答プールの大きさ。$y_{p,k}$: プロンプト $p$ の $k$ 番目の応答($k = 1, \dots, M$)。
- $s \in \{0, \dots, S-1\}$: シード。$S$ は実験 A・B の式では $S_{\mathrm{AB}}$、実験 C の式では $S_{\mathrm{C}}$。
- $r_{\phi_{N,s}}$: データ量の水準 $N$・シード $s$ の報酬モデル。標準の水準 $N_A$ の報酬モデルを $r_{\phi_s}$ と書く。
- $n \in \{2^0, 2^1, \dots, 2^7\}$: best-of-n の $n$($n_{\max} = M/4 = 128$、水準の数 $L = 8$)。$j = \log_2 n$。
- $\mathcal{N} = \{1024, 4096, 16384, 65536\}$: 報酬モデルのデータ量の水準(段階で決まる。段階 1 以降は最小の水準を除く)。

#### 共通の前提条件

- **P0(SFT の成立)**: $\pi_{\mathrm{ref}}$(マージ後の重み)の、016 の評価集合(250 入力 × 4 課題、016 と同一)での貪欲法の
  完全一致率 $\hat{p}$ と形式の遵守率 $\hat{f}$ が、$|\hat{p} - 0.8040| \le 3 \times 0.0289 = 0.0867$ かつ $\hat{f} \ge 0.99$
  を満たすこと。0.8040 は 016 の条件 1 の完全一致率の 5 シード平均、0.0289 はそのシード間の標本標準偏差であり、許容誤差は
  シード間のばらつきの 3 倍とした(017 の SFT は 1 シードなので、016 のシードのばらつきの範囲に入っていればよい)。
  016 の条件 1 の形式の遵守率は全シードで 1.0000 だった。全実験に適用する。
- **P1(報酬モデルの学習の成立)**: 報酬モデルの、評価用の組(下記)のうち $\Delta r^* \ne 0$ の組での、真の報酬の順序との
  一致率 $a$ が、$a - 0.5 > 2\sigma_a$ を満たすこと($\sigma_a$ は評価用の入力を単位とするクラスタブートストラップの
  標準偏差)。**実験 A・B・C のいずれにも、標準の水準 $N_A$ の報酬モデル(全シード)にのみ適用する。** 小さい水準の
  報酬モデルの一致率は、実験 C が操作する変数(データ量)の帰結そのものなので、前提条件にせず診断量として印字する
  (前提条件は操作する変数から独立でなければならない)。
- **P2(プールの多様性)**: プロンプトごとの異なり応答数の比率(異なるトークン列の数 / $M$)の、評価用のプロンプトでの平均が
  $n_{\max} / M = 0.25$ 以上であること。平均で $n_{\max}$ 個以上の異なる応答がないと、$n$ を増やしても同じ応答を選ぶだけに
  なり、best-of-n が早く飽和する。実験 B・C に適用する。

**評価用の組**: 各プロンプトのプールを先頭から 2 つずつ組にした $(y_{p,2k-1}, y_{p,2k})$($k = 1, \dots, M/2$、計
$P \cdot M / 2 = 65536$ 組)。プールは $\pi_{\mathrm{ref}}$ からの独立な抽選なので、これは報酬モデルの学習データと同じ分布の、
学習に使っていないプロンプトの組である。

**評価集合の標本誤差の単位**: 016 と同じく、独立な標本の単位は入力 $x$ とする(同じ入力の 4 プロンプトは同じ単語の並びを
共有するので独立でない)。評価用の入力を復元抽出するクラスタブートストラップ(反復 10,000 回。全シード・全水準・全 $n$ で
**同じ再標本** を使う)の分散を、シード間の分散に加える。

#### 削る段階(実行時間の予算に応じた自動選択)

| 段階 | 実験 C の水準 $\mathcal{N}$ | 実験 C のシード数 $S_{\mathrm{C}}$ | PPO の反復回数 | 実験 A・B のシード数 $S_{\mathrm{AB}}$ |
|---|---|---|---|---|
| 0 | $\{1024, 4096, 16384, 65536\}$(4 水準) | 5 | 150 | 5 |
| 1 | $\{4096, 16384, 65536\}$(3 水準) | 5 | 150 | 5 |
| 2 | $\{4096, 16384, 65536\}$(3 水準) | 3 | 150 | 5 |
| 3 | $\{4096, 16384, 65536\}$(3 水準) | 3 | 75 | 5 |
| 4 | $\{4096, 16384, 65536\}$(3 水準) | 3 | 75 | 3 |

- **選択の規則**: 本番の学習を始める前に(6.4 節)、スケーリングの計測(6.3 節)から 1 セッション全体の実行時間を段階ごとに
  見積もり、予算(T4 で 120 分)以内に収まる **最小の段階** を選ぶ。段階 4 でも超える場合は、学習の前に例外で停止する。
  **選択は見積もりのみに基づき、どの実験の結果も参照しない。**
- **段階の順序の理由**: 報酬モデルの学習(全パラメータ・$T_{\mathrm{RM}} = 2048$ ステップ × 報酬モデルの数)が最も重く
  (6.3 節)、その数を減らす段階を先に置いた。そのうえで失う情報の小さい順に、実験 C の最小の水準(回帰の範囲の下端)→
  実験 C の精度(シード数)→ 判定を置かない PPO の観察の長さ → 実験 A・B の精度(シード数)とした。
- **PPO の実行時間の上限**: PPO(2 水準の合計)の見積もりが 15 分を超える場合は、反復回数を
  $\lfloor C / (2 t) \rfloor$ に減らす(見積もりのみに基づく。$C$ は上限 15 分、$t$ は 1 反復あたりの見積もり)。

#### 実験 A: 報酬の尺度の較正(Bradley-Terry の最尤推定による尺度の識別)

**検証すること**: 標準のデータ量で学習した報酬モデルの報酬の差は、ラベルを生成した対数オッズの尺度 $\kappa \Delta r^*$ に
較正(calibration)されている(3.3 節の線形ヘッドの一階条件)。すなわち較正の傾き $\rho$ が 1 から $\pm 20\%$ 以内である。

**対比量(較正の傾き)**: シード $s$ ごとに、評価用の組(同点の組を含む)で $\hat{\Delta}_s = r_{\phi_s}(x, y_1) - r_{\phi_s}(x, y_2)$
と目標確率 $p = \sigma(\kappa \Delta r^*)$($\Delta r^* = r^*(x, y_1) - r^*(x, y_2)$)を求め、1 変数の凸な問題

$$
\hat{a}_s = \arg\min_a \sum_{\mathrm{pairs}} \left[ -p \log \sigma(a \hat{\Delta}_s) - (1 - p) \log \sigma(-a \hat{\Delta}_s) \right]
$$

の解 $\hat{a}_s$ をニュートン法で求める(目的関数は $a$ について凸で、2 階微分は $\sum \sigma(a\hat{\Delta})\sigma(-a\hat{\Delta}) \hat{\Delta}^2 > 0$)。
そのシード平均

$$
\rho = \frac{1}{S} \sum_s \hat{a}_s
$$

を対比量とする。**切片は入れない**: 組の 2 応答を入れ替えると $\hat{\Delta}$ と $\Delta r^*$ の符号がそろって反転し、目的関数の
各項は変わらないので、切片(応答の順序への偏り)が入る余地がない。報酬そのものではなく差 $\hat{\Delta}$ を使うのは、報酬が定数の
差を除いてしか識別できないため(3.2 節)である。ラベルを抽選せずに目標確率 $p$ を直接使うので、評価用の組のラベルの抽選による
ばらつきは入らない。

**標準偏差の導出**: 016 と同じく、シードと評価集合の 2 つの独立な源を考える。

1. シード: $\hat{a}_s$ のシード間の標本標準偏差 $s_\rho$ から、シード平均の分散は $s_\rho^2 / S$。
2. 評価集合: 評価用の入力のクラスタブートストラップの各反復で、同じ再標本から全シードの $\hat{a}_s$ を求め直して $\rho$ を
   計算し、その分散を $\mathrm{Var}_{\mathrm{boot}}(\rho)$ とする。再標本ごとの $\hat{a}_s$ は、ニュートン法を全データの推定値
   $\hat{a}_s$ から始めて 1 ステップ進めた値で近似する(影響関数による近似。入力ごとの勾配 $g_x$・2 階微分 $h_x$ を
   $\hat{a}_s$ で 1 度だけ計算し、再標本の重み $w_x$ で $\hat{a}_s^* = \hat{a}_s - \sum_x w_x g_x / \sum_x w_x h_x$ とする)。
   **これは近似である**(再標本ごとに最適化を解き直す値とは、2 次の項だけ異なる)。

$$
\sigma_\rho^2 = \frac{s_\rho^2}{S} + \mathrm{Var}_{\mathrm{boot}}(\rho)
$$

$\kappa$ は報酬モデルの学習用の組から決めた定数として扱う(評価集合の標本誤差に含めない)。

**判定(同等性の検定の形)**: 区間 $[\rho - 2\sigma_\rho, \rho + 2\sigma_\rho]$ が

- $[1 - \delta, 1 + \delta]$($\delta = 0.2$)に **完全に含まれれば支持**、
- $[1 - \delta, 1 + \delta]$ と **共通部分を持たなければ反証**、
- それ以外は判定不能。

「差が見えないこと」を支持としない形である(区間が広ければ、$\rho$ が 1 に近くても判定不能になる)。

**前提条件**: P0・P1(標準の水準の全シード)。

**作用点の記述**: Bradley-Terry の損失が直接作用するのは、同じプロンプトの応答どうしの報酬の差の、対数オッズとしての
大きさである(3.2・3.3 節)。較正の傾きはその量そのものである。

**診断量**: 報酬モデルの予測の忠実度 $b_s / \kappa$($\hat{\Delta}_s$ を $\Delta r^*$ に最小二乗で回帰した傾き $b_s$(切片あり)の
$\kappa$ に対する比。回帰の希釈を受けるので、報酬モデルが真の報酬を表現しきれない分だけ 1 より小さくなる。3.3 節)と $b_s$ の
切片、真の報酬との順位相関(プロンプトごとの Spearman の順位相関を、$r^*$ が定数でないプロンプトで平均したもの)、真の順序との
一致率(P1 の量)、評価用の組に同じ Bradley-Terry モデルで抽選したラベル(乱数シード`17399`)との一致率と、その
**ベイズ最適な一致率の上限** $\mathbb{E}[\max(p, 1 - p)]$(ラベルが確率的なので、真の報酬を完全に知っていてもこれを超えられない)。

#### 実験 B: best-of-n による過最適化

**検証すること**: 報酬モデルに対する最適化の圧力($n$)を強めると、真の報酬の期待値は、はじめ上がり、やがて下がる
(過最適化、Gao et al. [8])。

**量**: シード $s$ の報酬モデル $r_{\phi_s}$ を代理報酬とし、プロンプト $p$ ごとに 3.6 節の推定量で
$R_s(p, n) = \sum_i w_i(n, M) \, r^*(y_{p,(i)})$ を求め、プロンプトで平均して $R_s(n) = \frac{1}{P} \sum_p R_s(p, n)$ とする。
シード平均 $\bar{R}(n) = \frac{1}{S} \sum_s R_s(n)$。

**対比量**: 前半の水準 $j \in \{0, 1, 2, 3\}$($n \le 8$)での $\bar{R}$ の $j = \log_2 n$ に対する最小二乗の傾き $g_1$ と、
後半の水準 $j \in \{4, 5, 6, 7\}$($n \ge 16$)での同じ傾き $g_2$。境界は本番実行の前に固定した(水準の前半と後半を等分)。
argmax などの極値統計は判定に使わない。

**標準偏差の導出**: 傾きは $\bar{R}(n)$ の線形結合なので、実験 A と同じく $\sigma_{g_k}^2 = s_{g_k}^2 / S + \mathrm{Var}_{\mathrm{boot}}(g_k)$
($s_{g_k}$ はシードごとの傾き $g_{k,s}$ の標本標準偏差、$\mathrm{Var}_{\mathrm{boot}}$ は入力のクラスタブートストラップの分散。
全シード・全 $n$ で同じ再標本を使うので、$n$ 間・シード間の相関が保たれる)。

**判定**:

- **支持**: $g_1 > 2\sigma_{g_1}$ かつ $g_2 < -2\sigma_{g_2}$。
- **反証**: $g_2 > 2\sigma_{g_2}$(範囲内では過最適化が起きず、真の報酬が上がり続ける)。
- **判定不能**: それ以外。

**前提条件**: P0・P1(標準の水準の全シード)・P2。

**作用点の記述**: $n$ が直接作用するのは、選ばれる応答の代理報酬(代理報酬の順位での上位への集中)である。代理報酬の期待値は
$n$ について数学的に単調に増える(部分集合の最大値)ので判定に使わず、観察として図示する。真の報酬はその下流で、代理報酬と
真の報酬のずれを通じて決まる量であり、検証したい主張(過最適化)そのものを表す量なので対比量に選んだ。横軸は $\log_2 n$ とし、
KL ダイバージェンスの上界 $\log n - (n-1)/n$ を参考として併記する。

**診断量**: 代理報酬の期待値の曲線、$\bar{R}(n)$ の曲線(シードごと)、KL の上界、完全一致($r^* = 1$)の確率の曲線。

#### 実験 C: 選好データ量による過最適化の緩和

**検証すること**: 報酬モデルの学習データを増やすと、強い最適化の圧力($n_{\max}$)のもとでの真の報酬が高くなる(過最適化が
緩和される、Gao et al. [8])。

**量**: 水準 $N \in \mathcal{N}$・シード $s$ の報酬モデルで、実験 B と同じく $R_{N,s}(n_{\max})$ を求める。

**対比量**: シード平均 $\bar{R}_N(n_{\max})$ の $\log_2 N$ に対する最小二乗の傾き $c$。

**標準偏差の導出**: $\sigma_c^2 = s_c^2 / S + \mathrm{Var}_{\mathrm{boot}}(c)$($s_c$ はシードごとの傾き $c_s$ の標本標準偏差。
シード $s$ の報酬モデルは水準間でヘッドの初期化・ラベルの抽選の乱数を共有する(ラベルは入れ子)ので、$c_s$ を
シードごとに作ることで水準間の相関を含めた誤差伝播になる)。

**判定**: $c > 2\sigma_c$ なら支持、$c < -2\sigma_c$ なら反証、それ以外は判定不能。

**前提条件**: P0・P1(標準の水準の全シード)・P2。小さい水準の P1 の量は診断量(上記)。

**作用点の記述**: データ量が直接作用するのは報酬モデルの推定の精度(真の報酬の差の回復の度合い)であり、実験 A の診断量
(真の順序との一致率)がそれに近い。ただし学習の計算量を揃えているため、小さい水準では同じ組を多くのエポック反復しており、
組の数の減少と過適合の効果が交絡している(共通の設定を参照)。対比量はそこから best-of-n の選択を経た下流の量である。水準ごとの真の順序との一致率を、
直接の作用点に近い診断量として併記する。

**診断量**: 水準ごとの $\bar{R}_N(n)$ の曲線、各曲線の $\log_2 n$ への放物線のあてはめの頂点(下に凸でない場合は「頂点なし」)、
水準ごとの真の順序との一致率。

#### PPO の動作確認(判定基準を設けない観察)

KL 係数 $\beta \in \{0.01, 0.1\}$ の 2 水準で、標準の水準・シード 0 の報酬モデル $r_{\phi_0}$ を報酬として、固定の反復回数
(段階 0〜2 で 150 回、段階 3・4 で 75 回。各反復 64 プロンプト)だけ PPO を行い、反復ごとの KL ダイバージェンスの推定値・
代理報酬・真の報酬の推移を記録・図示する。**判定基準を設けない定性的な観察である。** 実行時間の上限は 2 水準の合計で
15 分(見積もり)とする(上記)。

### 6.2 パイロットによる設定の決定

本番実行の前に、ノートブックの外(ローカルの MPS、スクリプトで実行)で、本番と同じスケールのパイロットを行った。以下は
その記録である。パイロットの評価用の入力(乱数シード`17999`の 64 入力)は、本番の入力と重複しない(5.4 節でアサーションに
より確認)。**パイロットで決めたのは、報酬モデルの学習データ量・ステップ数・学習率、応答プールの大きさ $M$、PPO の学習率・
KL 係数・反復回数であり、いずれも実験 A〜C の対比量を計算せずに決めた。** スモークテストの後に、実験 A の対比量を 1 度
改訂した(本節の最後の「改訂 1」。本番実行の前の改訂である)。

#### パイロット 1: SFT の再現(本番と同じ設定)

016 のレシピで SFT を行い(ローカルの MPS で 120 秒)、016 の評価集合での完全一致率 0.805、形式の遵守率 1.000 を得た
(016 の条件 1 の 5 シード平均 0.804)。LoRA のマージの前後で、完全一致率・形式の遵守率は一致した。

#### パイロット 2: 応答プールの統計

パイロットの評価用のプロンプト 256 個(64 入力 × 4 課題)について、SFT モデルから $M = 256$ 個ずつ応答を抽選した
(temperature 1.0)。

| 量 | 値 |
|---|---|
| 異なり応答数の比率の平均(最小) | 0.941(0.406) |
| 抽選した応答の完全一致($r^* = 1$)の割合 | 0.034 |
| $r^*$ の平均(5%・50%・95% 分位点) | 0.730(0.558・0.737・0.920) |
| 組の同点($\Delta r^* = 0$)の割合 | 0.024(うち両方が完全一致 0.005) |
| 同一でない応答の組の同点の割合 | 0.019 |
| $\kappa$(差が 0 でない組の $\lvert \Delta r^* \rvert$ の中央値 0.0883) | 12.44 |
| 応答の平均トークン数(上限 32 に達した割合) | 12.7(0.014) |
| 生成の速度(MPS) | 1.66 ms / 応答 |

温度 1.0 のサンプリングでは完全一致はまれで(貪欲法の完全一致率 0.805 に対して 0.034)、応答は多様である。同点の組は少ない。
P2 の閾値 0.25 を大きく上回るので、プールの大きさを $M = 512$($n_{\max} = 128$)とした(生成の時間は $M$ に比例する)。プール全体は報酬モデルごとにスコアリングし直す(報酬モデルの数 × $PM$ 系列)ので、評価用の入力は $|X| = 64$($P = 256$)とし、プールを $256 \times 512 = 131072$ 応答に抑えた(スモークテストの見積もりで、報酬モデルの学習とスコアリングが実行時間の大半を占めたため。実験の結果は参照していない)。

#### パイロット 3: 報酬モデルの学習データ量・ステップ数・学習率

報酬モデルの学習用のプロンプト(パイロット専用、乱数シード`17998`)の組で学習し、パイロット 2 のプールの組(128 プロンプト ×
32 組、$\Delta r^* \ne 0$ の組)での **真の順序との一致率のみ** で比べた(実験 A の対比量(旧基準の $b / \kappa$・改訂後の較正の傾き)はいずれも計算していない)。
ラベルのシードは 5、ヘッドの初期化のシードは 0。

| 組の数 | ステップ数(エポック) | 学習率 | ホールドアウトの一致率 | 学習の最後の 50 ステップの損失・ラベルとの一致率 | MPS の時間 |
|---|---|---|---|---|---|
| 16384 | 512(1) | $10^{-5}$ | 0.571 | 0.672・0.551 | 55 s |
| 16384 | 512(1) | $3 \times 10^{-5}$ | 0.592 | 0.665・0.571 | 54 s |
| 16384 | 512(1) | $10^{-4}$ | 0.589 | 0.661・0.570 | 55 s |
| 16384 | 512(1) | $3 \times 10^{-4}$ | 0.592 | 0.664・0.562 | 54 s |
| 16384 | 2048(4) | $3 \times 10^{-5}$ | 0.632 | 0.591・0.657 | 227 s |
| 16384 | 2048(4) | $10^{-4}$ | 0.646 | 0.440・0.772 | 254 s |
| 65536 | 2048(1) | $10^{-4}$ | **0.671** | 0.629・0.630 | 228 s |
| 65536 | 2048(1) | $3 \times 10^{-4}$ | 0.673 | 0.631・0.606 | 218 s |

同じステップ数(2048)なら、同じ組を 4 エポック使うより、4 倍の組を 1 エポック使うほうがホールドアウトの一致率が高く、学習データ
への過適合(ラベルとの一致率 0.772 が、ラベルのばらつきから見て高すぎる)も起きなかった。$10^{-4}$ と $3 \times 10^{-4}$ は同等
だったので小さいほうを選び、**標準の水準 $N_A = 65536$ 組・$T_{\mathrm{RM}} = 2048$(1 エポック)・学習率 $10^{-4}$** とした。
報酬モデルは弱い(一致率 0.67)が、P1 を満たす水準にある。

#### パイロット 4: PPO の動作確認

パイロット 3 の報酬モデル(65536 組・$10^{-4}$)で、$\beta = 0.05$・60 反復(各 64 プロンプト)の PPO を、学習率 $10^{-3}$ と
$3 \times 10^{-3}$ で実行した(MPS で 1 反復約 2 秒)。$10^{-3}$ では KL ダイバージェンスの推定値が 60 反復で約 0.9 まで増え、
代理報酬は 0.25 から 0.7 前後に上がった。$3 \times 10^{-3}$ では KL が約 5 まで増え、真の報酬は途中から下がった。学習率は
$10^{-3}$ とし、$\beta$ は KL の増え方の違いが見えるよう 10 倍離した 2 水準 $\{0.01, 0.1\}$、反復回数は 150 とした。

#### 改訂 1: 実験 A の対比量(回帰の傾きの比 → 較正の傾き。スモークテストの後、本番実行の前)

**旧基準**: シード $s$ ごとに、評価用の組の $\hat{\Delta}_s$ を $\Delta r^*$ に最小二乗で回帰した傾き $b_s$(切片あり)を求め、
$\rho = \bar{b} / \kappa$ を対比量とする。標準偏差はシード間の分散と入力のクラスタブートストラップの分散の和。判定は同等性の
検定の形($\delta = 0.2$)。

**新基準**: 較正の傾き $\rho = \frac{1}{S} \sum_s \hat{a}_s$(6.1 節)。**変えないもの**: 同等性の検定の形(区間と
$[1 - \delta, 1 + \delta]$ の包含・共通部分による支持 / 反証 / 判定不能)、$\delta = 0.2$、前提条件(P0・P1)、標準偏差を
シード間の分散と入力のクラスタブートストラップの分散の和とすること。旧対比量 $b_s / \kappa$ は、報酬モデルの予測の忠実度を表す
診断量として残す(散布図も残す)。

**改訂の理由**:

- **旧対比量の問題**: $\hat{\Delta}$ を $\Delta r^*$ に回帰した傾きは、報酬モデルが真の報酬を完全に表現できない限り、回帰の希釈に
  よって $\kappa$ より小さくなる。較正された予測は、真の値に対して平均へ縮むためである(3.3 節)。したがって旧対比量は、
  尺度が識別されたかではなく報酬モデルの正確さを測る。正確さは、一致率・順位相関の診断量ですでに測っている。
- **新対比量の理論的な裏付け**: 報酬モデルのヘッドは線形なので、$a \, r_\phi$ も族に含まれる。そのため学習の目的関数の
  最適点では、$a$ についての一階条件から、学習データの分布の上で較正の傾きが 1 になる。これはモデルの族が $\kappa r^*$ を
  表現できなくても成り立つ。評価用の組は学習データと同じ分布の、学習に使っていない組なので、1 からのずれは汎化の分だけで
  ある(3.3 節)。これが「確率的なラベルから尺度が識別される」という主張の、表現能力に依存しない形である。
- **改訂の方向と時期**: この根拠は回帰の希釈と線形ヘッドの一階条件という一般論であり、観測結果の方向には依存しない。
  スモークテストの結果がどうであっても同じ改訂になる。改訂はスモークテストの後、本番実行の前に行った。

**旧基準によるスモークテストの結果**(縮小スケール: SFT 32 ステップ・報酬モデルの学習 32 ステップ・評価用の入力 16・2 シード。
動作確認のみで結論ではない): $\rho = 0.1909$、$\sigma_\rho = 0.0051$、区間 $[0.1807, 0.2012]$、判定関数の結果「反証」。
前提条件 P0 が不成立(縮小スケールの SFT は課題を学習しない)のため、最終判定は「前提不成立」だった。

### 6.3 スケーリングの計測と外挿(1 セッションの見積もり)

本番でデータ量がスモークテストの何倍にもなる重い処理について、3 点のデータ量で実行時間を実測し、
$\log t = \log a + b \log n$ をあてはめてべき指数 $b$ を推定し、本番のデータ量へ外挿する。本番では、この計測を **本番の
実行の冒頭に T4 上で** 行い、その値のみから削る段階を選ぶ(6.4 節)。外挿値と、最大の計測点の実測値を比例で伸ばした値の
大きいほうを見積もりとする(固定費があると $b < 1$ となり、外挿値が過小になりうるため)。

| 処理 | 計測するデータ量 | 外挿先 | 本番での回数 |
|---|---|---|---|
| SFT の学習 | ステップ数 $T/16, T/8, T/4$ | $T = 2048$ | 1 |
| 応答プールの生成 | プロンプト数 $P/64, P/32, P/16$(各 $M$ 応答) | $P M = 256 \times 512$ | 1 |
| 選好の組の生成 | 組の数 $N_A/128, N_A/64, N_A/32$ | $N_A = 65536$ | 1 |
| 真の報酬の計算 | 応答の数(上のプールの生成で得た応答の 1/4・1/2・全部) | プール + 組の応答 + PPO の応答 | 1 |
| 報酬モデルの学習 | ステップ数 $T_{\mathrm{RM}}/32, /16, /8$ | $T_{\mathrm{RM}} = 2048$ | 報酬モデルの数(段階で決まる) |
| 全応答のスコアリング | 系列の数(上のプールの生成で得た系列の 1/4・1/2・全部) | $PM$ | 報酬モデルの数 |
| PPO | 反復回数 1・2・4 | 反復回数 | 2(KL 係数の水準) |
| P0 の評価(貪欲法の生成) | 事例の数 64・128・256 | 1000 | 1 |

生成は、終端記号で止まらない`base_model`で計測する(上限 32 トークンまで生成するので、SFT 後の平均約 13 トークンより長く、
安全側)。計測専用のプロンプト(乱数シード`17099`)を使い、本番の評価用のプロンプトに触れない。モデルカード用の
検証 bits-per-byte の評価(コーパスの取得・符号化・評価)は、ここで`base_model`について実測し(読み込んだ重みが 008 の
モデルカードの値 1.668067 を再現することもあわせて確かめる)、SFT モデルの評価も同じ時間とみなす。スモークテストでは、
本番の見積もり(スモークテストを実行したデバイスでの外挿値)で段階の選択のコードを動かし、予算を人為的に小さくした場合に
「段階 0 以外が選ばれる」ことと「どの段階でも超えて停止する」ことを確かめる。

生成の計測では、最小の計測点の応答数が生成の 1 バッチ(`GENERATION_BATCH_SIZE` = 512)以上であることを、本番でアサーションにより
確かめる(1 バッチに満たない点では 1 バッチの固定費が支配的になり、比例で伸ばした見積もりが過大になるため)。スモークテストの
計測点はこの条件を満たさない(特に選好の組の生成の見積もりが過大になる)ので、**スモークテストの出力の段階の見積もりは、
T4 での本番の目安にならない。**


```python
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
_prod_prompts = PROD_CFG["NUM_EVAL_INPUTS"] * TASK_COUNT
_prod_pool = PROD_CFG["POOL_SIZE"]
_prod_pairs = PROD_CFG["RM_PAIRS_MAX"]
_prod_rm_steps = _prod_pairs // RM_BATCH_PAIRS
_prod_sft_steps = PROD_CFG["SFT_STEPS"]
_rng_timing = np.random.default_rng(TIMING_SEED)
_timing_examples = build_training_examples(
    sample_word_sequences(_rng_timing, 4096, WORD_VOCABULARY, MIN_WORDS, MAX_WORDS), _rng_timing
)
_timing_prompts = [tokenizer.encode(e.prompt(False)) for e in _timing_examples]
ESTIMATE: dict[str, float] = {}

# --- SFT の学習 ---
train_sft(2, SFT_TRAIN_ENCODED[: 2 * SFT_BATCH_SIZE])  # 初回のオーバーヘッドを除く
_sizes = [max(2, SFT_STEPS // 16), max(4, SFT_STEPS // 8), max(8, SFT_STEPS // 4)]
ESTIMATE["sft"] = fit_and_extrapolate(
    "SFT の学習(ステップ数)",
    _sizes,
    [timed_call(lambda n=n: train_sft(n, SFT_TRAIN_ENCODED[: n * SFT_BATCH_SIZE])) for n in _sizes],
    _prod_sft_steps,
)

# --- 応答プールの生成(各プロンプト M 応答、base_model は終端記号で止まらないので安全側) ---
_sizes = [
    max(1, NUM_EVAL_PROMPTS // 64),
    max(2, NUM_EVAL_PROMPTS // 32),
    max(4, NUM_EVAL_PROMPTS // 16),
]
_timing_texts = []


def _time_pool(n: int) -> None:
    out = sample_responses(
        base_model, [p for p in _timing_prompts[:n] for _ in range(POOL_SIZE)], TIMING_SEED
    )
    _timing_texts.extend(out)


def check_generation_sizes(label: str, smallest_responses: int) -> None:
    # 生成を計測する最小の点の応答数が 1 バッチ以上であること(1 バッチの固定費が支配的だと比例の見積もりが過大になる)
    if not SMOKE_TEST:
        assert smallest_responses >= GENERATION_BATCH_SIZE, (label, smallest_responses)
    elif smallest_responses < GENERATION_BATCH_SIZE:
        print(
            f"注意(スモークテスト): {label}の最小の計測点の応答数 {smallest_responses} が GENERATION_BATCH_SIZE = "
            f"{GENERATION_BATCH_SIZE} 未満(1 バッチの固定費が支配的なので、比例の見積もりが過大になる。本番では満たすことを確かめる)"
        )


check_generation_sizes("応答プールの生成", _sizes[0] * POOL_SIZE)
ESTIMATE["pool"] = fit_and_extrapolate(
    "応答プールの生成(プロンプト数 x M)",
    [n * POOL_SIZE for n in _sizes],
    [timed_call(lambda n=n: _time_pool(n)) for n in _sizes],
    _prod_prompts * _prod_pool,
)

# --- 選好の組の生成(組ごとに異なるプロンプト x 2 応答) ---
_sizes = [max(4, RM_PAIRS_MAX // 128), max(8, RM_PAIRS_MAX // 64), max(16, RM_PAIRS_MAX // 32)]
_timing_pairs = {}


def _time_pairs(n: int) -> None:
    out = sample_responses(
        base_model, [p for p in _timing_prompts[:n] for _ in range(2)], TIMING_SEED
    )
    _timing_pairs.update({"prompts": _timing_prompts[:n], "first": out[0::2], "second": out[1::2]})


check_generation_sizes("選好の組の生成", 2 * _sizes[0])
ESTIMATE["pairs"] = fit_and_extrapolate(
    "選好の組の生成(組の数)",
    _sizes,
    [timed_call(lambda n=n: _time_pairs(n)) for n in _sizes],
    _prod_pairs,
)

# --- 真の報酬の計算(キャッシュを使わずに計算する: 安全側) ---
_texts = [tokenizer.decode(r) for r in _timing_texts]
_answers = [_timing_examples[i % len(_timing_examples)].answer for i in range(len(_texts))]
_sizes = [len(_texts) // 4, len(_texts) // 2, len(_texts)]
_prod_ppo_responses = 2 * PROD_CFG["PPO_ITERATIONS"] * PPO_BATCH_SIZE
_true_reward_count = _prod_prompts * _prod_pool + 2 * _prod_pairs + _prod_ppo_responses
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
    _true_reward_count,
)

# --- 報酬モデルの学習(ステップ数、1 モデルあたり) ---
_saved_rm_steps = RM_STEPS
_timing_labels = np.arange(len(_timing_pairs["first"])) % 2 == 0
_sizes = [
    max(2, _saved_rm_steps // 32),
    max(4, _saved_rm_steps // 16),
    max(8, _saved_rm_steps // 8),
]
_times = []
for _n in _sizes:
    RM_STEPS = _n
    _times.append(
        timed_call(
            lambda: train_one_reward_model(
                base_model, _timing_pairs, _timing_labels, len(_timing_labels), 0
            )
        )
    )
RM_STEPS = _saved_rm_steps
ESTIMATE["rm_train"] = fit_and_extrapolate(
    "報酬モデルの学習(ステップ数)", _sizes, _times, _prod_rm_steps
)

# --- 全応答のスコアリング(系列の数、1 モデルあたり) ---
torch.manual_seed(0)
_timing_rm = freeze(RewardModel(copy.deepcopy(base_model), RM_HEAD_INIT_STD).to(device))
_timing_sequences = [
    _timing_prompts[(i // POOL_SIZE) % len(_timing_prompts)] + r
    for i, r in enumerate(_timing_texts)
]
_sizes = [len(_timing_sequences) // 4, len(_timing_sequences) // 2, len(_timing_sequences)]
ESTIMATE["rm_score"] = fit_and_extrapolate(
    "全応答のスコアリング(系列の数)",
    _sizes,
    [
        timed_call(
            lambda n=n: score_sequences(
                _timing_rm, _timing_sequences[:n], device, SCORING_BATCH_SIZE
            )
        )
        for n in _sizes
    ],
    _prod_prompts * _prod_pool,
)

# --- PPO(反復回数、1 反復あたりに換算) ---
_sizes = [1, 2, 4]
_times = [
    timed_call(lambda n=n: run_ppo_iterations(base_model, _timing_rm, 0.1, n)) for n in _sizes
]
_fit = fit_power_law_exponent(_sizes, _times)
PPO_SECONDS_PER_ITERATION = max(
    _times[-1] / _sizes[-1], _fit.coefficient * 100**_fit.exponent / 100
)
print(
    "[PPO(反復回数)] "
    + ", ".join(f"n={n}: {t:.2f}s" for n, t in zip(_sizes, _times, strict=True))
    + f" -> b={_fit.exponent:.3f}、1 反復あたりの見積もり {PPO_SECONDS_PER_ITERATION:.2f}s"
)

# --- P0 の評価(貪欲法の生成) ---
_sizes = [64, 128, 256]
ESTIMATE["p0_generation"] = fit_and_extrapolate(
    "P0 の評価(貪欲法の生成、事例の数)",
    _sizes,
    [
        timed_call(
            lambda n=n: greedy_generate_until_stop(
                base_model,
                [_timing_prompts[i % len(_timing_prompts)] for i in range(n)],
                tokenizer.decode,
                END_MARKER,
                MAX_NEW_TOKENS,
                device,
                64,
            )
        )
        for n in _sizes
    ],
    PROD_CFG["NUM_P0_INPUTS"] * TASK_COUNT,
)

# --- 検証 bits-per-byte(モデルカード用。base_model で実測し、読み込みも確かめる) ---
_t0 = time.time()
_corpus_text, _corpus_metadata = load_wikipedia_corpus_with_fallback(
    "en", CORPUS_REPO_ID, WIKIPEDIA_CACHE_DIR, manifest_path=MANIFEST_PATH, return_metadata=True
)
assert len(_corpus_text.encode("utf-8")) == _corpus_metadata["raw_bytes"], (
    "コーパスの取得が破損している"
)
_, _validation_text = split_train_val_text(_corpus_text, P0_VALIDATION_RATIO)
VALIDATION_BYTES = len(_validation_text.encode("utf-8"))
VALIDATION_WINDOWS, VALIDATION_MASK = make_evaluation_windows(
    torch.tensor(tokenizer.encode(_validation_text), dtype=torch.long), CONTEXT_LENGTH
)
del _corpus_text, _validation_text
BASE_BITS_PER_BYTE = evaluate_bits_per_byte(
    base_model, VALIDATION_WINDOWS, VALIDATION_MASK, VALIDATION_BYTES, device
)
ESTIMATE["bits_per_byte"] = 2 * (
    time.time() - _t0
)  # base_model(実測)+ SFT モデル(同じ時間とみなす)
assert abs(BASE_BITS_PER_BYTE - BASE_REFERENCE_BITS_PER_BYTE) / BASE_REFERENCE_BITS_PER_BYTE <= 0.01
print(
    f"検証 bits-per-byte(base_model、取得元 {_corpus_metadata['source']})= {BASE_BITS_PER_BYTE:.6f}"
    f"(008 のモデルカードの値 {BASE_REFERENCE_BITS_PER_BYTE}、相対誤差 1% 以内: OK)"
)
SCALING_SECONDS = time.time() - _t0_scaling
print(f"スケーリングの計測自体: {SCALING_SECONDS:.1f}s")
```

    [SFT の学習(ステップ数)] n=128: 3.52s, n=256: 7.44s, n=512: 14.38s -> b=1.016(標準誤差 0.037), R^2=0.9987, n=2,048 での外挿値 59.7s、比例 57.5s
    [応答プールの生成(プロンプト数 x M)] n=2,048: 1.68s, n=4,096: 3.52s, n=8,192: 7.37s -> b=1.065(標準誤差 0.000), R^2=1.0000, n=131,072 での外挿値 141.2s、比例 117.9s
    [選好の組の生成(組の数)] n=512: 4.04s, n=1,024: 5.25s, n=2,048: 5.89s -> b=0.272(標準誤差 0.061), R^2=0.9519, n=65,536 での外挿値 15.5s、比例 188.5s
    [真の報酬の計算(応答の数)] n=3,584: 4.49s, n=7,168: 7.97s, n=14,336: 14.20s -> b=0.830(標準誤差 0.002), R^2=1.0000, n=281,344 での外挿値 167.8s、比例 278.6s
    [報酬モデルの学習(ステップ数)] n=64: 3.78s, n=128: 7.46s, n=256: 15.28s -> b=1.008(標準誤差 0.015), R^2=0.9998, n=2,048 での外挿値 123.5s、比例 122.3s
    [全応答のスコアリング(系列の数)] n=3,584: 1.02s, n=7,168: 2.04s, n=14,336: 4.14s -> b=1.009(標準誤差 0.005), R^2=1.0000, n=131,072 での外挿値 38.5s、比例 37.8s
    [PPO(反復回数)] n=1: 3.70s, n=2: 7.47s, n=4: 14.08s -> b=0.964、1 反復あたりの見積もり 3.52s
    [P0 の評価(貪欲法の生成、事例の数)] n=64: 2.48s, n=128: 3.38s, n=256: 3.76s -> b=0.302(標準誤差 0.085), R^2=0.9265, n=1,000 での外挿値 5.9s、比例 14.7s



    corpus.txt: reconstructing file:   0%|          |  0.00B / 24.3MB            



    corpus.txt: downloading bytes:           |  0.00B            



    metadata.json:   0%|          | 0.00/43.4k [00:00<?, ?B/s]


    コーパス取得元: kojikojiprg/ai-theories-corpus-en-pretraining(Hugging Face Hub)
    検証 bits-per-byte(base_model、取得元 hub)= 1.668067(008 のモデルカードの値 1.668067、相対誤差 1% 以内: OK)
    スケーリングの計測自体: 155.0s


### 6.4 削る段階の選択

6.3 節の外挿値から、削る段階ごとに 1 セッション全体の見積もりを出し、予算(T4 で 120 分)に収まる最小の段階を選ぶ(6.1 節)。
**選択は見積もりのみに基づき、どの実験の結果も参照しない。** この時点では SFT も報酬モデルの学習も行っていない。


```python
def ppo_iterations_for(stage: dict, level: dict) -> int:
    # 段階の反復回数を、PPO の実行時間の上限(2 水準の合計で 15 分)で打ち切る(見積もりのみに基づく)
    iterations = max(1, round(level["PPO_ITERATIONS"] * stage["PPO_FRACTION"]))
    cap = int(PPO_TIME_CAP_SECONDS // (len(PPO_BETAS) * PPO_SECONDS_PER_ITERATION))
    return max(1, min(iterations, cap))


def num_reward_models(stage: dict) -> int:
    # 標準の水準は実験 A・B のシード数、それ以外の水準は実験 C のシード数(標準の水準は共有)
    return stage["NUM_SEEDS_AB"] + (stage["C_NUM_LEVELS"] - 1) * stage["NUM_SEEDS_C"]


COMMON_ESTIMATE = {
    "SFT の学習(外挿)": ESTIMATE["sft"],
    "応答プールの生成(外挿)": ESTIMATE["pool"],
    "選好の組の生成(外挿)": ESTIMATE["pairs"],
    "真の報酬の計算(外挿)": ESTIMATE["true_reward"],
    "P0 の評価(外挿)": ESTIMATE["p0_generation"],
    "検証 bits-per-byte(実測 x 2)": ESTIMATE["bits_per_byte"],
    "スケーリングの計測自体(実測)": SCALING_SECONDS,
    "ハーネスの確認(5.5 節、実測)": HARNESS_SECONDS,
}


def estimate_stage(stage: dict) -> dict:
    count = num_reward_models(stage)
    iterations = ppo_iterations_for(stage, PROD_CFG)
    return {
        **COMMON_ESTIMATE,
        f"報酬モデルの学習とスコアリング({count} モデル)": count
        * (ESTIMATE["rm_train"] + ESTIMATE["rm_score"]),
        f"PPO({len(PPO_BETAS)} 水準 x {iterations} 反復)": len(PPO_BETAS)
        * iterations
        * PPO_SECONDS_PER_ITERATION,
    }


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
print(f"--- 段階 0 の本番の見積もりの内訳({device} 基準)---")
for _k, _v in STAGE_ESTIMATES[0].items():
    print(f"  {_k}: {_v:,.1f}s")
print(
    f"\n--- 削る段階ごとの 1 セッション全体の見積もり({device} 基準、予算 {SESSION_BUDGET_SECONDS / 60:.0f} 分)---"
)
for _k, _total in STAGE_TOTALS.items():
    _stage = STAGES["prod"][_k]
    print(
        f"  段階 {_k}(実験 C の水準 {rm_levels(_stage['C_NUM_LEVELS'], PROD_CFG['RM_PAIRS_MAX'])}、S_C = "
        f"{_stage['NUM_SEEDS_C']}、PPO {ppo_iterations_for(_stage, PROD_CFG)} 反復、S_AB = {_stage['NUM_SEEDS_AB']}): "
        f"{_total:,.1f}s = {_total / 60:.1f} 分(予算の {_total / SESSION_BUDGET_SECONDS:.1%})"
    )
assert all(STAGE_TOTALS[k] >= STAGE_TOTALS[k + 1] for k in range(4)), (
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
        select_stage(STAGE_TOTALS, 0.5 * STAGE_TOTALS[4])
        raise AssertionError("どの段階でも超える予算で停止しなかった")
    except StageBudgetExceededError as _error:
        _stopped = str(_error)
    print(
        f"選択の規則の確認(人為的に小さい予算、確認のみ): 予算 {_budget_between / 60:.1f} 分 -> 段階 1: OK / "
        f"予算 {0.5 * STAGE_TOTALS[4] / 60:.1f} 分 -> 停止する({_stopped}): OK。予算の定数は {SESSION_BUDGET_SECONDS / 60:.0f} 分のまま"
    )
    assert SESSION_BUDGET_SECONDS == 120 * 60

# --- 選ばれた段階の水準・シード数(以降のすべてのセルがこれを使う) ---
STAGE = STAGES[CURRENT_LEVEL_NAME][SELECTED_STAGE]
NUM_SEEDS_AB, NUM_SEEDS_C = STAGE["NUM_SEEDS_AB"], STAGE["NUM_SEEDS_C"]
SEEDS_AB, SEEDS_C = tuple(range(NUM_SEEDS_AB)), tuple(range(NUM_SEEDS_C))
C_LEVELS = rm_levels(STAGE["C_NUM_LEVELS"], RM_PAIRS_MAX)
PPO_ITERATIONS = ppo_iterations_for(STAGE, CFG)
assert set(SEEDS_C) <= set(SEEDS_AB) and C_LEVELS[-1] == RM_PAIRS_MAX
assert len(PPO_EXAMPLES) >= PPO_ITERATIONS * PPO_BATCH_SIZE
print(
    f"このノートブックで使う値(段階 {SELECTED_STAGE}、水準 {CURRENT_LEVEL_NAME!r}): 実験 A・B のシード {SEEDS_AB}、"
    f"実験 C のシード {SEEDS_C}・水準 {C_LEVELS}、PPO の反復回数 {PPO_ITERATIONS}"
)
if device.type != "cuda":
    print("注意: CUDA 以外での見積もりであり、T4 での時間とは異なる")
```

    --- 段階 0 の本番の見積もりの内訳(cuda 基準)---
      SFT の学習(外挿): 59.7s
      応答プールの生成(外挿): 141.2s
      選好の組の生成(外挿): 188.5s
      真の報酬の計算(外挿): 278.6s
      P0 の評価(外挿): 14.7s
      検証 bits-per-byte(実測 x 2): 12.1s
      スケーリングの計測自体(実測): 155.0s
      ハーネスの確認(5.5 節、実測): 16.6s
      報酬モデルの学習とスコアリング(20 モデル): 3,240.4s
      PPO(2 水準 x 127 反復): 893.9s
    
    --- 削る段階ごとの 1 セッション全体の見積もり(cuda 基準、予算 120 分)---
      段階 0(実験 C の水準 (1024, 4096, 16384, 65536)、S_C = 5、PPO 127 反復、S_AB = 5): 5,000.7s = 83.3 分(予算の 69.5%)
      段階 1(実験 C の水準 (4096, 16384, 65536)、S_C = 5、PPO 127 反復、S_AB = 5): 4,190.6s = 69.8 分(予算の 58.2%)
      段階 2(実験 C の水準 (4096, 16384, 65536)、S_C = 3、PPO 127 反復、S_AB = 5): 3,542.5s = 59.0 分(予算の 49.2%)
      段階 3(実験 C の水準 (4096, 16384, 65536)、S_C = 3、PPO 75 反復、S_AB = 5): 3,176.5s = 52.9 分(予算の 44.1%)
      段階 4(実験 C の水準 (4096, 16384, 65536)、S_C = 3、PPO 75 反復、S_AB = 3): 2,852.5s = 47.5 分(予算の 39.6%)
    
    段階の選択: 予算 120 分に収まる最小の段階として、段階 0 を選んだ(見積もり 83.3 分、cuda 基準)
    このノートブックで使う値(段階 0、水準 'prod'): 実験 A・B のシード (0, 1, 2, 3, 4)、実験 C のシード (0, 1, 2, 3, 4)・水準 (1024, 4096, 16384, 65536)、PPO の反復回数 127




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/04_alignment/017_reward_model_and_rlhf.ipynb
