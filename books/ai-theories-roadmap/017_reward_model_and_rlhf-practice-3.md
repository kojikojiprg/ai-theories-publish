---
title: "報酬モデルと RLHF / Reward Models and RLHF(実装・実験編 3/4)"
---

この記事は後編(実装・実験編 3/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/017_reward_model_and_rlhf-practice-2)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/017_reward_model_and_rlhf-practice-4)。

### 6.5 SFT(参照方策 $\pi_{\mathrm{ref}}$ の作成)と前提条件 P0

016 の条件 1 のレシピで SFT を行い、LoRA をマージする。

- **マージの前後で出力が一致すること**: 同じ入力の logits の差の最大値が $10^{-3}$ 以下で、016 の評価集合での貪欲法の生成が
  99% 以上の事例で一致すること(畳み込みは浮動小数点の演算の順序を変えるので、bit 単位の一致は求めない)。
- 凍結したパラメータが`base_model`と bit 単位で一致すること(LoRA 以外が学習されていないこと)。
- **P0**: マージ後の重み(アップロードする重みそのもの)で、016 の評価集合の完全一致率・形式の遵守率を測る(6.1 節)。
  あわせて、応答部分の負の対数尤度(教師強制)と、英語版 Wikipedia の検証 bits-per-byte を測る(モデルカード用)。


```python
_t0 = time.time()
_sft_lora, _replaced, SFT_HISTORY = train_sft(SFT_STEPS, SFT_TRAIN_ENCODED)
SFT_SECONDS = time.time() - _t0
for _name, _p in _sft_lora.named_parameters():  # 凍結したパラメータが base_model と一致
    if not _p.requires_grad:
        assert torch.equal(_p.cpu(), BASE_STATE[_name.replace(".base_layer", "")]), _name
reference_policy, REFERENCE_STATE = merge_lora(_sft_lora, _replaced)

# --- マージの前後の一致 ---
_probe = torch.tensor([list(x.token_ids[:24]) for x in P0_ENCODED[:32]], device=device)
with torch.no_grad():
    MERGE_MAX_LOGIT_DIFF = float((reference_policy(_probe) - _sft_lora.eval()(_probe)).abs().max())
assert MERGE_MAX_LOGIT_DIFF <= 1e-3, MERGE_MAX_LOGIT_DIFF
_p0_prompts = [list(x.token_ids[: x.prompt_length]) for x in P0_ENCODED]
_generated_merged = greedy_generate_until_stop(
    reference_policy, _p0_prompts, tokenizer.decode, END_MARKER, MAX_NEW_TOKENS, device, 64
)
_generated_lora = greedy_generate_until_stop(
    _sft_lora, _p0_prompts, tokenizer.decode, END_MARKER, MAX_NEW_TOKENS, device, 64
)
MERGE_GENERATION_AGREEMENT = float(
    np.mean([a == b for a, b in zip(_generated_merged, _generated_lora, strict=True)])
)
assert MERGE_GENERATION_AGREEMENT >= 0.99, MERGE_GENERATION_AGREEMENT
del _sft_lora

# --- P0(マージ後の重み) ---
_scores = [
    score_generated_response(tokenizer.decode(g), e.answer)
    for g, e in zip(_generated_merged, P0_EXAMPLES, strict=True)
]
SFT_FORMAT_RATE = float(np.mean([s[0] for s in _scores]))
SFT_EXACT_MATCH = float(np.mean([s[1] for s in _scores]))
_nll = evaluate_instruction_negative_log_likelihood(reference_policy, P0_ENCODED, device)
SFT_RESPONSE_NLL = float(_nll["response_sum"].sum() / _nll["response_count"].sum())
SFT_BITS_PER_BYTE = evaluate_bits_per_byte(
    reference_policy, VALIDATION_WINDOWS, VALIDATION_MASK, VALIDATION_BYTES, device
)
precondition_status["P0"] = bool(
    abs(SFT_EXACT_MATCH - P0_REFERENCE_EXACT) <= P0_TOLERANCE
    and SFT_FORMAT_RATE >= P0_MIN_FORMAT_RATE
)
print(
    f"SFT: {SFT_STEPS} ステップ({SFT_SECONDS:.1f}s)、学習の最後の 100 ステップの損失の平均 "
    f"{np.mean(SFT_HISTORY['loss'][-100:]):.4f}"
)
print(
    f"LoRA のマージ: logits の差の最大値 {MERGE_MAX_LOGIT_DIFF:.2e}(<= 1e-3)、貪欲法の生成の一致率 "
    f"{MERGE_GENERATION_AGREEMENT:.4f}(>= 0.99)、凍結したパラメータが base_model と bit 単位で一致: OK"
)
print(
    f"P0(マージ後の重み、016 の評価集合 {len(P0_EXAMPLES)} 事例): 完全一致率 {SFT_EXACT_MATCH:.4f}"
    f"(016 の 5 シード平均 {P0_REFERENCE_EXACT}、許容 +/-{P0_TOLERANCE:.4f})、形式の遵守率 {SFT_FORMAT_RATE:.4f}"
    f"(>= {P0_MIN_FORMAT_RATE})-> {'成立' if precondition_status['P0'] else '不成立'}"
)
print(
    f"モデルカード用: 応答部分の負の対数尤度 {SFT_RESPONSE_NLL:.4f} nats / トークン、"
    f"検証 bits-per-byte {SFT_BITS_PER_BYTE:.6f}(SFT 前 {BASE_BITS_PER_BYTE:.6f})"
)
print("\n--- π_ref の貪欲法の生成例 ---")
for _e, _g in list(zip(P0_EXAMPLES, _generated_merged, strict=True))[:TASK_COUNT]:
    print(f"[{_e.task}] 正解 {' '.join(_e.answer)!r} -> 生成 {tokenizer.decode(_g)!r}")
```

    SFT: 2048 ステップ(60.7s)、学習の最後の 100 ステップの損失の平均 0.5956
    LoRA のマージ: logits の差の最大値 3.67e-05(<= 1e-3)、貪欲法の生成の一致率 1.0000(>= 0.99)、凍結したパラメータが base_model と bit 単位で一致: OK
    P0(マージ後の重み、016 の評価集合 1000 事例): 完全一致率 0.8180(016 の 5 シード平均 0.804、許容 +/-0.0867)、形式の遵守率 1.0000(>= 0.99)-> 成立
    モデルカード用: 応答部分の負の対数尤度 0.6005 nats / トークン、検証 bits-per-byte 3.151869(SFT 前 1.668067)
    
    --- π_ref の貪欲法の生成例 ---
    [reverse] 正解 'community novel member genus software billion green agreement' -> 生成 ' community novel member genus global green green agreement\n### End'
    [repeat_twice] 正解 'agreement agreement green green billion billion software software genus genus member member novel novel community community' -> 生成 ' agreement agreement green green billion billion software software genus genus member member novel novel community community\n### End'
    [odd_positions] 正解 'agreement billion genus novel' -> 生成 ' agreement billion genus novel\n### End'
    [rotate_left] 正解 'green billion software genus member novel community agreement' -> 生成 ' green billion software genus member novel community agreement\n### End'


### 6.6 応答プール・選好の組・$\kappa$・ラベルと、前提条件 P2

- 評価用の各プロンプトについて、$\pi_{\mathrm{ref}}$ から $M$ 個の応答を抽選する(乱数シード`17018`)。
- 報酬モデルの学習用の各プロンプトについて、$\pi_{\mathrm{ref}}$ から 2 応答を抽選する(乱数シード`17019`)。
- 選好の組の $\Delta r^*$ から $\kappa$ を決め(3.3 節)、シードごとにラベルを抽選する。
- **P2**: プロンプトごとの異なり応答数の比率の平均 $\ge n_{\max} / M$。


```python
_t0 = time.time()
_flat_pool = sample_responses(
    reference_policy, [p for p in EVAL_PROMPT_IDS for _ in range(POOL_SIZE)], POOL_SEED
)
POOL = [_flat_pool[i * POOL_SIZE : (i + 1) * POOL_SIZE] for i in range(NUM_EVAL_PROMPTS)]
POOL_TRUE = true_rewards(_flat_pool, [e for e in EVAL_EXAMPLES for _ in range(POOL_SIZE)]).reshape(
    NUM_EVAL_PROMPTS, POOL_SIZE
)
POOL_SECONDS = time.time() - _t0
POOL_HASH = hash_json(POOL)
_t0 = time.time()
_flat_pairs = sample_responses(
    reference_policy, [p for p in RM_TRAIN_PROMPT_IDS for _ in range(2)], PAIR_SEED
)
PAIRS = {"prompts": RM_TRAIN_PROMPT_IDS, "first": _flat_pairs[0::2], "second": _flat_pairs[1::2]}
PAIR_TRUE_FIRST = true_rewards(PAIRS["first"], RM_TRAIN_EXAMPLES)
PAIR_TRUE_SECOND = true_rewards(PAIRS["second"], RM_TRAIN_EXAMPLES)
PAIRS_SECONDS = time.time() - _t0
del _flat_pool, _flat_pairs

# --- kappa とラベル ---
_pair_delta = PAIR_TRUE_FIRST - PAIR_TRUE_SECOND
KAPPA = calibrate_label_scale(_pair_delta, KAPPA_TARGET)
LABELS = {
    s: sample_preference_labels(
        PAIR_TRUE_FIRST, PAIR_TRUE_SECOND, KAPPA, np.random.default_rng(LABEL_SEED_BASE + s)
    )
    for s in SEEDS_AB
}
_nonzero = _pair_delta != 0
_probabilities = compute_bradley_terry_probability(_pair_delta[_nonzero], KAPPA)
assert math.isclose(  # sigma(kappa x 中央値) = 目標(3.3 節の式)
    float(compute_bradley_terry_probability(np.median(np.abs(_pair_delta[_nonzero])), KAPPA)),
    KAPPA_TARGET,
    rel_tol=1e-9,
)

# --- 評価用の組(プールを先頭から 2 つずつ) ---
EVAL_DELTA_TRUE = POOL_TRUE[:, 0::2] - POOL_TRUE[:, 1::2]  # (P, M/2)

# --- P2 と、プール・組の統計 ---
POOL_DISTINCT_RATIO = np.array(
    [len({tuple(r) for r in responses}) / POOL_SIZE for responses in POOL]
)
P2_THRESHOLD = N_MAX_BON / POOL_SIZE
precondition_status["P2"] = bool(POOL_DISTINCT_RATIO.mean() >= P2_THRESHOLD)
_identical_pairs = np.array(
    [[tuple(r[k]) == tuple(r[k + 1]) for k in range(0, POOL_SIZE, 2)] for r in POOL]
)
print(
    f"応答プール: {NUM_EVAL_PROMPTS} プロンプト x M = {POOL_SIZE}({POOL_SECONDS:.1f}s)、選好の組 {len(PAIRS['first']):,}({PAIRS_SECONDS:.1f}s)"
)
print(
    f"P2: 異なり応答数の比率の平均 {POOL_DISTINCT_RATIO.mean():.4f}(最小 {POOL_DISTINCT_RATIO.min():.4f})"
    f" >= n_max / M = {P2_THRESHOLD:.4f} -> {'成立' if precondition_status['P2'] else '不成立'}"
)
print(
    f"プールの r*: 平均 {POOL_TRUE.mean():.4f}、完全一致(r* = 1)の割合 {np.mean(POOL_TRUE == 1.0):.4f}、"
    f"分位点(5%・50%・95%){np.round(np.quantile(POOL_TRUE, [0.05, 0.5, 0.95]), 4).tolist()}"
)
print(
    f"評価用の組: {EVAL_DELTA_TRUE.size:,} 組、同点(Delta r* = 0)の割合 {np.mean(EVAL_DELTA_TRUE == 0):.4f}"
    f"(同一の応答の組 {_identical_pairs.mean():.4f}、同一でない応答の同点 "
    f"{np.mean((EVAL_DELTA_TRUE == 0) & ~_identical_pairs):.4f})"
)
print(
    f"kappa = log 3 / median|Delta r*| = {KAPPA:.4f}(選好の組の差が 0 でない組 {_nonzero.mean():.4f}、"
    f"|Delta r*| の中央値 {np.median(np.abs(_pair_delta[_nonzero])):.4f})"
)
print(
    "ラベルが真の順序と一致する割合(差が 0 でない組、シード順): "
    + ", ".join(
        f"{np.mean(LABELS[s][_nonzero] == (_pair_delta[_nonzero] > 0)):.4f}" for s in SEEDS_AB
    )
    + f"(期待値 E[max(p, 1 - p)] = {np.mean(np.maximum(_probabilities, 1 - _probabilities)):.4f})"
)
print("\n--- プールの応答の例(評価用の先頭のプロンプト)---")
print(f"正解: {' '.join(EVAL_EXAMPLES[0].answer)!r}")
for _k in range(6):
    print(f"  r* = {POOL_TRUE[0, _k]:.4f}: {tokenizer.decode(POOL[0][_k])!r}")
```

    応答プール: 256 プロンプト x M = 512(144.3s)、選好の組 65,536(153.5s)
    P2: 異なり応答数の比率の平均 0.9351(最小 0.3926) >= n_max / M = 0.2500 -> 成立
    プールの r*: 平均 0.7257、完全一致(r* = 1)の割合 0.0332、分位点(5%・50%・95%)[0.5515, 0.734, 0.9167]
    評価用の組: 65,536 組、同点(Delta r* = 0)の割合 0.0259(同一の応答の組 0.0069、同一でない応答の同点 0.0190)
    kappa = log 3 / median|Delta r*| = 12.4746(選好の組の差が 0 でない組 0.9765、|Delta r*| の中央値 0.0881)
    ラベルが真の順序と一致する割合(差が 0 でない組、シード順): 0.7544, 0.7543, 0.7523, 0.7552, 0.7538(期待値 E[max(p, 1 - p)] = 0.7546)
    
    --- プールの応答の例(評価用の先頭のプロンプト)---
    正解: 'mission inflation account brother graduate center'
      r* = 0.7449: ' mission sit amid assistant graduateherit\n### End'
      r* = 0.8163: ' plan incre injur brother graduate center\n### End'
      r* = 0.8878: ' mission train account brother graduate network\n### End'
      r* = 0.7959: ' mission funding course brother professor center\n### End'
      r* = 0.7755: ' mission climatesue murder graduate zone\n### End'
      r* = 0.6735: ' fac data st friend graduate train\n### End'


### 6.7 報酬モデルの学習とスコアリング、前提条件 P1

標準の水準 $N_A$ を実験 A・B のシード、それ以外の水準を実験 C のシードで学習し、プールの全応答をスコアリングする(同じ応答は
1 回だけスコアリングする)。PPO(6.11 節)に使う標準の水準・シード 0 の報酬モデルのみメモリ上に残す(保存はしない)。

**P1**: 標準の水準の全シードで、評価用の組($\Delta r^* \ne 0$)の真の順序との一致率 $a$ が $a - 0.5 > 2\sigma_a$。
$\sigma_a$ は評価用の入力を単位とするクラスタブートストラップ(全実験で共通の再標本)の標準偏差。小さい水準の同じ量は診断量。


```python
# --- 全実験で共通のクラスタブートストラップの再標本(評価用の入力の抽出回数) ---
BOOTSTRAP_COUNTS = (
    np.random.default_rng(BOOTSTRAP_SEED)
    .multinomial(
        NUM_EVAL_INPUTS, np.full(NUM_EVAL_INPUTS, 1.0 / NUM_EVAL_INPUTS), size=BOOTSTRAP_RESAMPLES
    )
    .astype(np.float64)
)  # (B, |X|)


def by_input(values: np.ndarray) -> np.ndarray:
    # 軸 -2 がプロンプト(P)の配列を、入力ごと(4 プロンプト)の和にする: (..., P, k) -> (..., |X|, k)
    values = np.asarray(values)
    return values.reshape(*values.shape[:-2], NUM_EVAL_INPUTS, TASK_COUNT, values.shape[-1]).sum(
        axis=-2
    )


# --- プールの一意な系列(同じ応答は 1 回だけスコアリングする) ---
_unique_index: dict[tuple, int] = {}
POOL_UNIQUE_SEQUENCES = []
POOL_SEQUENCE_INDEX = np.empty((NUM_EVAL_PROMPTS, POOL_SIZE), dtype=np.int64)
for _p, _responses in enumerate(POOL):
    for _k, _response in enumerate(_responses):
        _key = (_p, tuple(_response))
        if _key not in _unique_index:
            _unique_index[_key] = len(POOL_UNIQUE_SEQUENCES)
            POOL_UNIQUE_SEQUENCES.append(EVAL_PROMPT_IDS[_p] + list(_response))
        POOL_SEQUENCE_INDEX[_p, _k] = _unique_index[_key]


def score_pool(reward_model: RewardModel) -> np.ndarray:
    return score_sequences(reward_model, POOL_UNIQUE_SEQUENCES, device, SCORING_BATCH_SIZE)[
        POOL_SEQUENCE_INDEX
    ]


def true_order_agreement(pool_scores: np.ndarray) -> dict:
    # 評価用の組(Delta r* != 0)で、報酬の差の符号が真の報酬の差の符号と一致する割合と、そのブートストラップの標準偏差
    delta_hat = pool_scores[:, 0::2] - pool_scores[:, 1::2]
    nonzero = EVAL_DELTA_TRUE != 0
    agree = (np.sign(delta_hat) == np.sign(EVAL_DELTA_TRUE)) & nonzero
    agree_x = by_input(agree.sum(axis=1, keepdims=True))[:, 0]
    count_x = by_input(nonzero.sum(axis=1, keepdims=True))[:, 0]
    boot = (BOOTSTRAP_COUNTS @ agree_x) / (BOOTSTRAP_COUNTS @ count_x)
    return {"agreement": float(agree_x.sum() / count_x.sum()), "sigma": float(boot.std(ddof=1))}


RM_JOBS = [(RM_PAIRS_MAX, s) for s in SEEDS_AB] + [(n, s) for n in C_LEVELS[:-1] for s in SEEDS_C]
RM_RECORDS: dict[tuple[int, int], dict] = {}
_t0_rm = time.time()
for _n, _s in RM_JOBS:
    _t0 = time.time()
    _reward_model, _history = train_one_reward_model(reference_policy, PAIRS, LABELS[_s], _n, _s)
    _train_seconds = time.time() - _t0
    _t0 = time.time()
    _scores = score_pool(_reward_model)
    RM_RECORDS[(_n, _s)] = {
        "history": _history,
        "pool_scores": _scores,
        "train_seconds": _train_seconds,
        "score_seconds": time.time() - _t0,
        **true_order_agreement(_scores),
    }
    if (_n, _s) == (RM_PAIRS_MAX, 0):
        PPO_REWARD_MODEL = _reward_model
    del _reward_model
    _r = RM_RECORDS[(_n, _s)]
    print(
        f"報酬モデル N = {_n:,}(エポック {RM_STEPS * RM_BATCH_PAIRS / _n:g})seed={_s}: 学習 {_r['train_seconds']:.1f}s・"
        f"スコアリング {_r['score_seconds']:.1f}s、最後の 100 ステップの損失 {np.mean(_history['loss'][-100:]):.4f}・"
        f"ラベルとの一致率 {np.mean(_history['accuracy'][-100:]):.4f}"
    )
print(
    f"報酬モデルの学習とスコアリングの合計: {time.time() - _t0_rm:.1f}s(プールの一意な系列 {len(POOL_UNIQUE_SEQUENCES):,})"
)

# --- P1(標準の水準の全シード)と、小さい水準の診断量 ---
_p1_rows = []
for _s in SEEDS_AB:
    _r = RM_RECORDS[(RM_PAIRS_MAX, _s)]
    _p1_rows.append(_r["agreement"] - 0.5 > 2 * _r["sigma"])
    print(
        f"P1 [N = {RM_PAIRS_MAX:,}, seed={_s}]: 真の順序との一致率 {_r['agreement']:.4f}、sigma {_r['sigma']:.4f} -> {'> 2 sigma' if _p1_rows[-1] else '<= 2 sigma'}"
    )
precondition_status["P1"] = bool(all(_p1_rows))
print(
    f"P1(標準の水準の全 {len(_p1_rows)} シード): {'成立' if precondition_status['P1'] else '不成立'}"
)
for _n in C_LEVELS[:-1]:
    print(
        f"診断量 N = {_n:,}: 真の順序との一致率(シード順)"
        + ", ".join(f"{RM_RECORDS[(_n, s)]['agreement']:.4f}" for s in SEEDS_C)
    )
```

    報酬モデル N = 65,536(エポック 1)seed=0: 学習 108.8s・スコアリング 24.4s、最後の 100 ステップの損失 0.6259・ラベルとの一致率 0.6231
    報酬モデル N = 65,536(エポック 1)seed=1: 学習 107.9s・スコアリング 24.1s、最後の 100 ステップの損失 0.6332・ラベルとの一致率 0.6022
    報酬モデル N = 65,536(エポック 1)seed=2: 学習 108.0s・スコアリング 24.2s、最後の 100 ステップの損失 0.6455・ラベルとの一致率 0.5803
    報酬モデル N = 65,536(エポック 1)seed=3: 学習 108.2s・スコアリング 24.2s、最後の 100 ステップの損失 0.6229・ラベルとの一致率 0.6316
    報酬モデル N = 65,536(エポック 1)seed=4: 学習 108.3s・スコアリング 24.2s、最後の 100 ステップの損失 0.6359・ラベルとの一致率 0.6141
    報酬モデル N = 1,024(エポック 64)seed=0: 学習 107.5s・スコアリング 24.3s、最後の 100 ステップの損失 0.0001・ラベルとの一致率 1.0000
    報酬モデル N = 1,024(エポック 64)seed=1: 学習 107.8s・スコアリング 24.3s、最後の 100 ステップの損失 0.0001・ラベルとの一致率 1.0000
    報酬モデル N = 1,024(エポック 64)seed=2: 学習 107.8s・スコアリング 24.3s、最後の 100 ステップの損失 0.0001・ラベルとの一致率 1.0000
    報酬モデル N = 1,024(エポック 64)seed=3: 学習 107.5s・スコアリング 24.3s、最後の 100 ステップの損失 0.0001・ラベルとの一致率 1.0000
    報酬モデル N = 1,024(エポック 64)seed=4: 学習 107.5s・スコアリング 24.3s、最後の 100 ステップの損失 0.0001・ラベルとの一致率 1.0000
    報酬モデル N = 4,096(エポック 16)seed=0: 学習 108.3s・スコアリング 24.3s、最後の 100 ステップの損失 0.0064・ラベルとの一致率 0.9953
    報酬モデル N = 4,096(エポック 16)seed=1: 学習 108.2s・スコアリング 24.4s、最後の 100 ステップの損失 0.0053・ラベルとの一致率 0.9962
    報酬モデル N = 4,096(エポック 16)seed=2: 学習 108.2s・スコアリング 24.2s、最後の 100 ステップの損失 0.0069・ラベルとの一致率 0.9941
    報酬モデル N = 4,096(エポック 16)seed=3: 学習 107.8s・スコアリング 24.2s、最後の 100 ステップの損失 0.0052・ラベルとの一致率 0.9947
    報酬モデル N = 4,096(エポック 16)seed=4: 学習 108.5s・スコアリング 24.3s、最後の 100 ステップの損失 0.0086・ラベルとの一致率 0.9947
    報酬モデル N = 16,384(エポック 4)seed=0: 学習 108.3s・スコアリング 24.2s、最後の 100 ステップの損失 0.4305・ラベルとの一致率 0.7897
    報酬モデル N = 16,384(エポック 4)seed=1: 学習 107.9s・スコアリング 24.2s、最後の 100 ステップの損失 0.4579・ラベルとの一致率 0.7672
    報酬モデル N = 16,384(エポック 4)seed=2: 学習 108.2s・スコアリング 24.3s、最後の 100 ステップの損失 0.4818・ラベルとの一致率 0.7591
    報酬モデル N = 16,384(エポック 4)seed=3: 学習 108.3s・スコアリング 24.4s、最後の 100 ステップの損失 0.5237・ラベルとの一致率 0.7241
    報酬モデル N = 16,384(エポック 4)seed=4: 学習 108.5s・スコアリング 24.3s、最後の 100 ステップの損失 0.4854・ラベルとの一致率 0.7431
    報酬モデルの学習とスコアリングの合計: 2647.1s(プールの一意な系列 122,565)
    P1 [N = 65,536, seed=0]: 真の順序との一致率 0.6802、sigma 0.0059 -> > 2 sigma
    P1 [N = 65,536, seed=1]: 真の順序との一致率 0.6601、sigma 0.0053 -> > 2 sigma
    P1 [N = 65,536, seed=2]: 真の順序との一致率 0.6235、sigma 0.0045 -> > 2 sigma
    P1 [N = 65,536, seed=3]: 真の順序との一致率 0.6789、sigma 0.0050 -> > 2 sigma
    P1 [N = 65,536, seed=4]: 真の順序との一致率 0.6568、sigma 0.0052 -> > 2 sigma
    P1(標準の水準の全 5 シード): 成立
    診断量 N = 1,024: 真の順序との一致率(シード順)0.5711, 0.5588, 0.5903, 0.5588, 0.5744
    診断量 N = 4,096: 真の順序との一致率(シード順)0.5892, 0.5934, 0.6021, 0.5970, 0.5815
    診断量 N = 16,384: 真の順序との一致率(シード順)0.6314, 0.6346, 0.6301, 0.6398, 0.6403


### 6.8 実験 A: 報酬の尺度の較正

較正の傾き $\hat{a}_s$ をニュートン法で求める(6.1 節)。解法の確認として、人工データ $\hat{\Delta} = c \, \kappa \Delta r^*$
で $\hat{a} = 1/c$ が回復することをアサーションで確かめる(このとき $a = 1/c$ で $\sigma(a \hat{\Delta}) = p$ となり、交差エントロピーが
最小値に達する)。クラスタブートストラップの再標本ごとの値は 1 ステップのニュートン法による近似(影響関数による近似)であり、
先頭の再標本の一部について、最適化を解き直した値との差を印字する。


```python
def three_way_verdict(value: float, sigma: float) -> str:
    # value > 2 sigma: 支持、value < -2 sigma: 反証、それ以外: 判定不能(value は期待する向きが正になるよう渡す)
    if not (math.isfinite(value) and math.isfinite(sigma)):
        return "判定不能"
    if value > 2 * sigma:
        return "支持"
    if value < -2 * sigma:
        return "反証"
    return "判定不能"


def equivalence_verdict(value: float, sigma: float, delta: float) -> str:
    # [value - 2 sigma, value + 2 sigma] が [1 - delta, 1 + delta] に含まれれば支持、共通部分がなければ反証
    if not (math.isfinite(value) and math.isfinite(sigma)):
        return "判定不能"
    low, high = value - 2 * sigma, value + 2 * sigma
    if 1 - delta <= low and high <= 1 + delta:
        return "支持"
    if high < 1 - delta or low > 1 + delta:
        return "反証"
    return "判定不能"


def slope_from_sums(n, sum_x, sum_y, sum_xx, sum_xy):
    # 最小二乗の傾き(切片あり)を十分統計量から計算する
    return (sum_xy - sum_x * sum_y / n) / (sum_xx - sum_x**2 / n)


def ols_slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # y の最後の軸を x に回帰した傾き(切片あり)
    x = np.asarray(x, dtype=np.float64)
    xc = x - x.mean()
    return (np.asarray(y) - np.asarray(y).mean(axis=-1, keepdims=True)) @ xc / (xc @ xc)


def calibration_terms(
    a: float, delta_hat: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # 目的関数 -p log sigma(a d) - (1 - p) log sigma(-a d) の、組ごとの a についての 1 階微分・2 階微分
    q = 1.0 / (1.0 + np.exp(-a * delta_hat))
    return (q - target) * delta_hat, q * (1.0 - q) * delta_hat**2


def calibration_loss(
    a: float, delta_hat: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> float:
    z = a * delta_hat
    return float(
        np.sum(weights * (target * np.logaddexp(0.0, -z) + (1.0 - target) * np.logaddexp(0.0, z)))
    )


def fit_calibration_slope(
    delta_hat: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None
) -> float:
    # 較正の傾き a = argmin_a sum w [-p log sigma(a d) - (1 - p) log sigma(-a d)](切片なし)をニュートン法で求める。
    # 目的関数は a について凸。損失が増える場合はステップを半分にする(減衰付きのニュートン法)。
    weights = np.ones_like(delta_hat) if weights is None else weights
    a = 1.0
    for _ in range(100):
        gradient, hessian = calibration_terms(a, delta_hat, target)
        step = float(np.sum(weights * gradient) / np.sum(weights * hessian))
        current = calibration_loss(a, delta_hat, target, weights)
        while calibration_loss(a - step, delta_hat, target, weights) > current + 1e-12 * abs(
            current
        ):
            step /= 2.0
        a -= step
        if abs(step) < 1e-12 * max(1.0, abs(a)):
            break
    return float(a)


# --- 解法の確認: 人工データ Delta_hat = c kappa Delta r* で a = 1 / c が回復する ---
_rng_synthetic = np.random.default_rng(0)
_synthetic_delta = _rng_synthetic.normal(0.0, 0.1, size=5000)
_synthetic_target = compute_bradley_terry_probability(_synthetic_delta, KAPPA)
for _c in (0.25, 0.5, 2.0, 4.0):
    _a = fit_calibration_slope(_c * KAPPA * _synthetic_delta, _synthetic_target)
    assert math.isclose(_a, 1.0 / _c, rel_tol=1e-8), (_c, _a)
print(
    "較正の傾きの解法: 人工データ Delta_hat = c kappa Delta r*(c = 0.25, 0.5, 2, 4)で a = 1 / c を回復(相対誤差 1e-8 以内): OK"
)

STANDARD_SCORES = np.stack(
    [RM_RECORDS[(RM_PAIRS_MAX, s)]["pool_scores"] for s in SEEDS_AB]
)  # (S, P, M)
_delta_hat = STANDARD_SCORES[:, :, 0::2] - STANDARD_SCORES[:, :, 1::2]  # (S, P, M/2)
_x = EVAL_DELTA_TRUE  # (P, M/2)
_target = compute_bradley_terry_probability(_x, KAPPA)  # 目標確率 p(同点の組は 0.5)

# --- 対比量: 較正の傾き ---
A_HAT_SEEDS = np.array([fit_calibration_slope(_delta_hat[s], _target) for s in range(NUM_SEEDS_AB)])
RHO = float(A_HAT_SEEDS.mean())
RHO_SEED_STD = float(A_HAT_SEEDS.std(ddof=1))
# 評価集合の項: 入力ごとの 1 階微分・2 階微分(a_hat_s で評価)から、1 ステップのニュートン法で再標本ごとの値を近似する
_gradient_x, _hessian_x = [], []
for _s in range(NUM_SEEDS_AB):
    _g, _h = calibration_terms(A_HAT_SEEDS[_s], _delta_hat[_s], _target)
    _gradient_x.append(by_input(_g.sum(axis=1, keepdims=True))[:, 0])
    _hessian_x.append(by_input(_h.sum(axis=1, keepdims=True))[:, 0])
_gradient_x, _hessian_x = np.stack(_gradient_x), np.stack(_hessian_x)  # (S, |X|)
_a_boot = A_HAT_SEEDS[None, :] - (BOOTSTRAP_COUNTS @ _gradient_x.T) / (
    BOOTSTRAP_COUNTS @ _hessian_x.T
)  # (B, S)
RHO_VAR_BOOT = float(_a_boot.mean(axis=1).var(ddof=1))
SIGMA_RHO = math.sqrt(RHO_SEED_STD**2 / NUM_SEEDS_AB + RHO_VAR_BOOT)
verdict_A = equivalence_verdict(RHO, SIGMA_RHO, A_EQUIVALENCE_MARGIN)
# 近似の確認(診断): 先頭の 20 再標本について、シード 0 の値を解き直した値と比べる
_weights_pairs = (
    np.repeat(BOOTSTRAP_COUNTS[:20], TASK_COUNT, axis=1)[:, :, None] * np.ones_like(_x)[None]
)  # (20, P, M/2)
_exact = np.array([fit_calibration_slope(_delta_hat[0], _target, w) for w in _weights_pairs])
_approximation_error = float(np.abs(_exact - _a_boot[:20, 0]).max())
_tag = "[動作確認のみ、結論ではない] " if SMOKE_TEST else ""
print(
    f"kappa = {KAPPA:.4f}、評価用の組 {_x.size:,}(|X| = {NUM_EVAL_INPUTS}、同点の組を含む)、S = {NUM_SEEDS_AB}"
)
print("較正の傾き a_hat_s(シード順): " + ", ".join(f"{a:.4f}" for a in A_HAT_SEEDS))
print(
    f"rho = mean_s a_hat_s = {RHO:.4f}、s_rho = {RHO_SEED_STD:.4f}、Var_boot(rho) = {RHO_VAR_BOOT:.3e}"
    f"(1 ステップのニュートン法による近似)、sigma_rho = {SIGMA_RHO:.4f}"
)
print(
    f"区間 [rho - 2 sigma, rho + 2 sigma] = [{RHO - 2 * SIGMA_RHO:.4f}, {RHO + 2 * SIGMA_RHO:.4f}]、"
    f"同等性の範囲 [{1 - A_EQUIVALENCE_MARGIN}, {1 + A_EQUIVALENCE_MARGIN}]"
)
print(f"前提条件: P0={precondition_status['P0']}、P1={precondition_status['P1']}")
print(f"{_tag}判定関数の結果: {verdict_A}")
print(
    f"ブートストラップの近似の確認(シード 0、先頭 20 再標本): 解き直した値との差の最大値 {_approximation_error:.2e}"
    f"(再標本の値のばらつき(標準偏差){_a_boot[:, 0].std(ddof=1):.2e} と比べる)"
)

# --- 診断量: 報酬モデルの予測の忠実度 b_s / kappa(回帰の希釈を受ける、旧対比量) ---
_n_x = by_input(np.full((NUM_EVAL_PROMPTS, 1), _x.shape[1], dtype=np.float64))[:, 0]
_sx_x = by_input(_x.sum(axis=1, keepdims=True))[:, 0]
_sxx_x = by_input((_x**2).sum(axis=1, keepdims=True))[:, 0]
_sy_x = by_input(_delta_hat.sum(axis=2)[..., None])[..., 0]  # (S, |X|)
_sxy_x = by_input((_delta_hat * _x).sum(axis=2)[..., None])[..., 0]
B_SEEDS = slope_from_sums(
    _n_x.sum(), _sx_x.sum(), _sy_x.sum(axis=1), _sxx_x.sum(), _sxy_x.sum(axis=1)
)
INTERCEPTS = (_sy_x.sum(axis=1) - B_SEEDS * _sx_x.sum()) / _n_x.sum()
FIDELITY_SEEDS = B_SEEDS / KAPPA
print(
    "診断量: 予測の忠実度 b_s / kappa(回帰の傾きの比、回帰の希釈を受ける、シード順): "
    + ", ".join(f"{r:.4f}" for r in FIDELITY_SEEDS)
    + f"、平均 {FIDELITY_SEEDS.mean():.4f}(傾き b_s: "
    + ", ".join(f"{b:.4f}" for b in B_SEEDS)
    + ")"
)

# --- 診断量 ---
_spearman = []
for _s in range(NUM_SEEDS_AB):
    _values = [
        compute_spearman_correlation(STANDARD_SCORES[_s, p], POOL_TRUE[p])
        for p in range(NUM_EVAL_PROMPTS)
        if np.ptp(POOL_TRUE[p]) > 0
    ]
    _spearman.append(float(np.mean(_values)))
_eval_labels = sample_preference_labels(
    POOL_TRUE[:, 0::2].ravel(),
    POOL_TRUE[:, 1::2].ravel(),
    KAPPA,
    np.random.default_rng(EVAL_LABEL_SEED),
)
_label_agreement = [
    float(np.mean((_delta_hat[s].ravel() > 0) == _eval_labels)) for s in range(NUM_SEEDS_AB)
]
_p = compute_bradley_terry_probability(_x.ravel(), KAPPA)
BAYES_LABEL_AGREEMENT = float(np.mean(np.maximum(_p, 1 - _p)))
print(
    "診断量: 真の報酬とのプロンプト内の Spearman の順位相関の平均(シード順): "
    + ", ".join(f"{v:.4f}" for v in _spearman)
)
print(
    "診断量: 真の順序との一致率(P1 の量、シード順): "
    + ", ".join(f"{RM_RECORDS[(RM_PAIRS_MAX, s)]['agreement']:.4f}" for s in SEEDS_AB)
)
print(
    "診断量: 評価用の組に抽選したラベルとの一致率(シード順): "
    + ", ".join(f"{v:.4f}" for v in _label_agreement)
    + f"、ベイズ最適な一致率の上限 E[max(p, 1 - p)] = {BAYES_LABEL_AGREEMENT:.4f}"
)
print("診断量: 回帰の切片(シード順): " + ", ".join(f"{v:+.4f}" for v in INTERCEPTS))

_fig, _ax = plt.subplots(figsize=(6, 5))
_sub = np.random.default_rng(0).choice(_x.size, size=min(4000, _x.size), replace=False)
_ax.scatter(_x.ravel()[_sub], _delta_hat[0].ravel()[_sub], s=3, alpha=0.3, label="pairs (seed 0)")
_grid = np.linspace(_x.min(), _x.max(), 50)
_ax.plot(
    _grid, INTERCEPTS[0] + B_SEEDS[0] * _grid, color="C1", label=f"OLS fit (b = {B_SEEDS[0]:.2f})"
)
_ax.plot(
    _grid,
    KAPPA * _grid,
    color="C2",
    linestyle="--",
    label=f"kappa x Delta r* (kappa = {KAPPA:.2f})",
)
_ax.set_xlabel("Delta r* (true reward difference)")
_ax.set_ylabel("Delta hat (reward model difference)")
_ax.set_title("Experiment A diagnostic: fidelity (regression slope)")
_ax.legend()
plt.tight_layout()
plt.show()
```

    較正の傾きの解法: 人工データ Delta_hat = c kappa Delta r*(c = 0.25, 0.5, 2, 4)で a = 1 / c を回復(相対誤差 1e-8 以内): OK
    kappa = 12.4746、評価用の組 65,536(|X| = 64、同点の組を含む)、S = 5
    較正の傾き a_hat_s(シード順): 1.0586, 1.0634, 1.0674, 1.0363, 1.0264
    rho = mean_s a_hat_s = 1.0504、s_rho = 0.0180、Var_boot(rho) = 5.813e-04(1 ステップのニュートン法による近似)、sigma_rho = 0.0254
    区間 [rho - 2 sigma, rho + 2 sigma] = [0.9996, 1.1013]、同等性の範囲 [0.8, 1.2]
    前提条件: P0=True、P1=True
    判定関数の結果: 支持
    ブートストラップの近似の確認(シード 0、先頭 20 再標本): 解き直した値との差の最大値 1.81e-03(再標本の値のばらつき(標準偏差)2.81e-02 と比べる)
    診断量: 予測の忠実度 b_s / kappa(回帰の傾きの比、回帰の希釈を受ける、シード順): 0.3379, 0.2889, 0.2827, 0.3241, 0.3167、平均 0.3101(傾き b_s: 4.2148, 3.6040, 3.5269, 4.0434, 3.9511)
    診断量: 真の報酬とのプロンプト内の Spearman の順位相関の平均(シード順): 0.4918, 0.4462, 0.3393, 0.5007, 0.4401
    診断量: 真の順序との一致率(P1 の量、シード順): 0.6802, 0.6601, 0.6235, 0.6789, 0.6568
    診断量: 評価用の組に抽選したラベルとの一致率(シード順): 0.6248, 0.6154, 0.5895, 0.6272, 0.6106、ベイズ最適な一致率の上限 E[max(p, 1 - p)] = 0.7533
    診断量: 回帰の切片(シード順): +0.0044, +0.0013, +0.0008, -0.0004, +0.0012



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/017_reward_model_and_rlhf/output_38_1.png)
    


### 6.9 実験 B: best-of-n による過最適化


```python
def curves_by_input(scores: np.ndarray, values: np.ndarray) -> np.ndarray:
    # 1 つの報酬モデルのプールのスコア (P, M) から、入力ごとの best-of-n の期待値の和 (|X|, L) を返す
    return by_input(best_of_n_expected_values(scores, values, BON_LEVELS))


def bootstrap_curves(per_input: np.ndarray) -> np.ndarray:
    # (S, |X|, L) の入力ごとの和から、再標本ごとのシード平均の曲線 (B, L) を返す
    weights = BOOTSTRAP_COUNTS / (TASK_COUNT * BOOTSTRAP_COUNTS.sum(axis=1, keepdims=True))
    return np.einsum("bx,sxl->bl", weights, per_input) / per_input.shape[0]


BON_J = np.log2(BON_LEVELS)
_first, _second = slice(0, BON_HALF), slice(BON_HALF, None)
BON_TRUE_BY_INPUT = np.stack(
    [curves_by_input(STANDARD_SCORES[s], POOL_TRUE) for s in range(NUM_SEEDS_AB)]
)  # (S, |X|, L)
BON_PROXY = np.stack(
    [
        best_of_n_expected_values(STANDARD_SCORES[s], STANDARD_SCORES[s], BON_LEVELS).mean(axis=0)
        for s in range(NUM_SEEDS_AB)
    ]
)
BON_EXACT = np.stack(
    [
        best_of_n_expected_values(
            STANDARD_SCORES[s], (POOL_TRUE == 1.0).astype(float), BON_LEVELS
        ).mean(axis=0)
        for s in range(NUM_SEEDS_AB)
    ]
)
BON_TRUE_SEEDS = BON_TRUE_BY_INPUT.sum(axis=1) / NUM_EVAL_PROMPTS  # (S, L) = R_s(n)
BON_TRUE_MEAN = BON_TRUE_SEEDS.mean(axis=0)
_boot_curves = bootstrap_curves(BON_TRUE_BY_INPUT)  # (B, L)
CONTRAST_B = {}
for _name, _part in (("g1", _first), ("g2", _second)):
    _seed_slopes = ols_slope(BON_J[_part], BON_TRUE_SEEDS[:, _part])
    _var_boot = float(ols_slope(BON_J[_part], _boot_curves[:, _part]).var(ddof=1))
    CONTRAST_B[_name] = {
        "value": float(ols_slope(BON_J[_part], BON_TRUE_MEAN[_part])),
        "seed_slopes": _seed_slopes,
        "var_boot": _var_boot,
        "sigma": math.sqrt(float(_seed_slopes.std(ddof=1)) ** 2 / NUM_SEEDS_AB + _var_boot),
    }
_g1, _g2 = CONTRAST_B["g1"], CONTRAST_B["g2"]
if _g1["value"] > 2 * _g1["sigma"] and _g2["value"] < -2 * _g2["sigma"]:
    verdict_B = "支持"
elif _g2["value"] > 2 * _g2["sigma"]:
    verdict_B = "反証"
else:
    verdict_B = "判定不能"
print(
    f"n の水準: {BON_LEVELS}(前半 {BON_LEVELS[_first]}・後半 {BON_LEVELS[_second]})、M = {POOL_SIZE}、S = {NUM_SEEDS_AB}"
)
print("真の報酬の期待値 R(n)(シード平均): " + ", ".join(f"{v:.4f}" for v in BON_TRUE_MEAN))
for _name, _label in (("g1", "前半"), ("g2", "後半")):
    _c = CONTRAST_B[_name]
    print(
        f"{_name}({_label}の傾き、log2 n あたり)= {_c['value']:+.5f}、シードごと "
        + ", ".join(f"{v:+.5f}" for v in _c["seed_slopes"])
        + f"、Var_boot = {_c['var_boot']:.3e}、sigma = {_c['sigma']:.5f}、2 sigma = {2 * _c['sigma']:.5f}"
    )
print(
    f"前提条件: P0={precondition_status['P0']}、P1={precondition_status['P1']}、P2={precondition_status['P2']}"
)
print(f"{_tag}判定関数の結果: {verdict_B}")
print(
    "診断量: 代理報酬の期待値(シード平均): " + ", ".join(f"{v:.4f}" for v in BON_PROXY.mean(axis=0))
)
print(
    "診断量: 完全一致(r* = 1)の確率(シード平均): "
    + ", ".join(f"{v:.4f}" for v in BON_EXACT.mean(axis=0))
)
print(
    "参考: KL の上界 log n - (n-1)/n: "
    + ", ".join(f"{best_of_n_kl_upper_bound(n):.3f}" for n in BON_LEVELS)
)

_fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
for _s in range(NUM_SEEDS_AB):
    _axes[0].plot(BON_J, BON_TRUE_SEEDS[_s], color="C0", alpha=0.3)
_axes[0].plot(BON_J, BON_TRUE_MEAN, color="C0", marker="o", label="true reward r* (seed mean)")
_axes[0].axvline(BON_J[BON_HALF] - 0.5, color="gray", linestyle=":", label="first / second half")
_axes[0].set_xlabel("log2 n")
_axes[0].set_ylabel("E[r*] of best-of-n")
_top = _axes[0].secondary_xaxis("top", functions=(lambda j: j, lambda j: j))
_top.set_xticks(BON_J)
_top.set_xticklabels([f"{best_of_n_kl_upper_bound(n):.2f}" for n in BON_LEVELS])
_top.set_xlabel("KL upper bound log n - (n-1)/n (nats)")
_axes[0].legend()
_axes[1].plot(BON_J, BON_PROXY.mean(axis=0), color="C1", marker="o")
_axes[1].set_xlabel("log2 n")
_axes[1].set_ylabel("E[proxy reward] of best-of-n")
_axes[1].set_title("proxy reward (monotone by construction)")
plt.tight_layout()
plt.show()
```

    n の水準: (1, 2, 4, 8, 16, 32, 64, 128)(前半 (1, 2, 4, 8)・後半 (16, 32, 64, 128))、M = 512、S = 5
    真の報酬の期待値 R(n)(シード平均): 0.7257, 0.7621, 0.7735, 0.7736, 0.7673, 0.7591, 0.7515, 0.7444
    g1(前半の傾き、log2 n あたり)= +0.01552、シードごと +0.01939, +0.01621, +0.00709, +0.01925, +0.01565、Var_boot = 6.745e-07、sigma = 0.00239、2 sigma = 0.00477
    g2(後半の傾き、log2 n あたり)= -0.00762、シードごと -0.00710, -0.00784, -0.00795, -0.00722, -0.00798、Var_boot = 2.808e-07、sigma = 0.00056、2 sigma = 0.00112
    前提条件: P0=True、P1=True、P2=True
    判定関数の結果: 支持
    診断量: 代理報酬の期待値(シード平均): 1.7270, 1.9950, 2.1022, 2.1684, 2.2194, 2.2628, 2.3018, 2.3378
    診断量: 完全一致(r* = 1)の確率(シード平均): 0.0332, 0.0474, 0.0547, 0.0484, 0.0326, 0.0174, 0.0076, 0.0026
    参考: KL の上界 log n - (n-1)/n: 0.000, 0.193, 0.636, 1.204, 1.835, 2.497, 3.175, 3.860



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/017_reward_model_and_rlhf/output_40_1.png)
    


### 6.10 実験 C: 選好データ量による過最適化の緩和


```python
C_TRUE_BY_INPUT = {
    n: np.stack([curves_by_input(RM_RECORDS[(n, s)]["pool_scores"], POOL_TRUE) for s in SEEDS_C])
    for n in C_LEVELS
}  # 水準ごとに (S_C, |X|, L)
C_CURVES = {n: v.sum(axis=1) / NUM_EVAL_PROMPTS for n, v in C_TRUE_BY_INPUT.items()}  # (S_C, L)
_log_n = np.log2(C_LEVELS)
_at_max = np.stack([C_CURVES[n][:, -1] for n in C_LEVELS], axis=1)  # (S_C, 水準)
C_SEED_SLOPES = ols_slope(_log_n, _at_max)
C_SLOPE = float(ols_slope(_log_n, _at_max.mean(axis=0)))
_boot_at_max = np.stack(
    [bootstrap_curves(C_TRUE_BY_INPUT[n])[:, -1] for n in C_LEVELS], axis=1
)  # (B, 水準)
C_VAR_BOOT = float(ols_slope(_log_n, _boot_at_max).var(ddof=1))
SIGMA_C = math.sqrt(float(C_SEED_SLOPES.std(ddof=1)) ** 2 / NUM_SEEDS_C + C_VAR_BOOT)
verdict_C = three_way_verdict(C_SLOPE, SIGMA_C)
print(f"データ量の水準 N: {C_LEVELS}、シード {SEEDS_C}、n_max = {N_MAX_BON}")
for _i, _n in enumerate(C_LEVELS):
    print(
        f"  N = {_n:>6,}: R(n_max)(シード順)"
        + ", ".join(f"{v:.4f}" for v in _at_max[:, _i])
        + f"、平均 {_at_max[:, _i].mean():.4f}"
    )
print(
    f"c(R(n_max) の log2 N に対する傾き)= {C_SLOPE:+.5f}、シードごと "
    + ", ".join(f"{v:+.5f}" for v in C_SEED_SLOPES)
    + f"、Var_boot = {C_VAR_BOOT:.3e}、sigma_c = {SIGMA_C:.5f}、2 sigma_c = {2 * SIGMA_C:.5f}"
)
print(
    f"前提条件: P0={precondition_status['P0']}、P1={precondition_status['P1']}、P2={precondition_status['P2']}"
)
print(f"{_tag}判定関数の結果: {verdict_C}")
for _n in C_LEVELS:  # 診断量: 放物線のあてはめの頂点と、真の順序との一致率
    _mean = C_CURVES[_n].mean(axis=0)
    _a2, _a1, _a0 = np.polyfit(BON_J, _mean, 2)
    _vertex = f"log2 n = {-_a1 / (2 * _a2):.2f}" if _a2 < 0 else "頂点なし(上に凸でない)"
    print(
        f"診断量 N = {_n:,}: R(n)(シード平均)"
        + ", ".join(f"{v:.4f}" for v in _mean)
        + f"、放物線の頂点 {_vertex}、真の順序との一致率の平均 {np.mean([RM_RECORDS[(_n, s)]['agreement'] for s in SEEDS_C]):.4f}"
    )

_fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
for _i, _n in enumerate(C_LEVELS):
    _axes[0].plot(BON_J, C_CURVES[_n].mean(axis=0), marker="o", color=f"C{_i}", label=f"N = {_n:,}")
_axes[0].set_xlabel("log2 n")
_axes[0].set_ylabel("E[r*] of best-of-n (seed mean)")
_axes[0].legend()
for _s in range(NUM_SEEDS_C):
    _axes[1].plot(_log_n, _at_max[_s], color="C0", alpha=0.3)
_axes[1].plot(_log_n, _at_max.mean(axis=0), color="C0", marker="o")
_axes[1].set_xlabel("log2 N (preference pairs)")
_axes[1].set_ylabel(f"E[r*] at n_max = {N_MAX_BON}")
plt.tight_layout()
plt.show()
```

    データ量の水準 N: (1024, 4096, 16384, 65536)、シード (0, 1, 2, 3, 4)、n_max = 128
      N =  1,024: R(n_max)(シード順)0.7095, 0.7057, 0.7210, 0.7094, 0.7098、平均 0.7111
      N =  4,096: R(n_max)(シード順)0.7156, 0.7183, 0.7263, 0.7198, 0.7134、平均 0.7187
      N = 16,384: R(n_max)(シード順)0.7276, 0.7350, 0.7307, 0.7334, 0.7279、平均 0.7309
      N = 65,536: R(n_max)(シード順)0.7610, 0.7467, 0.7106, 0.7601, 0.7437、平均 0.7444
    c(R(n_max) の log2 N に対する傾き)= +0.00562、シードごと +0.00833, +0.00700, -0.00134, +0.00828, +0.00582、Var_boot = 1.648e-07、sigma_c = 0.00185、2 sigma_c = 0.00369
    前提条件: P0=True、P1=True、P2=True
    判定関数の結果: 支持
    診断量 N = 1,024: R(n)(シード平均)0.7257, 0.7446, 0.7457, 0.7388, 0.7304, 0.7230, 0.7167, 0.7111、放物線の頂点 log2 n = 2.14、真の順序との一致率の平均 0.5707
    診断量 N = 4,096: R(n)(シード平均)0.7257, 0.7502, 0.7534, 0.7472, 0.7386, 0.7309, 0.7244, 0.7187、放物線の頂点 log2 n = 2.64、真の順序との一致率の平均 0.5926
    診断量 N = 16,384: R(n)(シード平均)0.7257, 0.7579, 0.7666, 0.7644, 0.7567, 0.7476, 0.7390, 0.7309、放物線の頂点 log2 n = 3.23、真の順序との一致率の平均 0.6353
    診断量 N = 65,536: R(n)(シード平均)0.7257, 0.7621, 0.7735, 0.7736, 0.7673, 0.7591, 0.7515, 0.7444、放物線の頂点 log2 n = 3.56、真の順序との一致率の平均 0.6599



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/017_reward_model_and_rlhf/output_42_1.png)
    


### 6.11 PPO の動作確認(判定なし)

標準の水準・シード 0 の報酬モデルを報酬とし、KL 係数 $\beta \in \{0.01, 0.1\}$ で PPO を行う。判定基準を設けない定性的な
観察である。反復ごとの値はバッチ(64 プロンプト)の平均なので、図には 10 反復の移動平均を重ねる。


```python
PPO_HISTORIES = {}
_t0_ppo = time.time()
for _beta in PPO_BETAS:
    _t0 = time.time()
    PPO_HISTORIES[_beta] = run_ppo_iterations(
        reference_policy, PPO_REWARD_MODEL, _beta, PPO_ITERATIONS
    )
    _h = PPO_HISTORIES[_beta]
    _k = max(1, min(10, PPO_ITERATIONS // 4))
    print(
        f"beta = {_beta}: {PPO_ITERATIONS} 反復({time.time() - _t0:.1f}s)。最初 / 最後の {_k} 反復の平均: "
        f"KL {np.mean(_h['kl'][:_k]):.3f} / {np.mean(_h['kl'][-_k:]):.3f}、"
        f"代理報酬 {np.mean(_h['proxy_reward'][:_k]):.3f} / {np.mean(_h['proxy_reward'][-_k:]):.3f}、"
        f"真の報酬 {np.mean(_h['true_reward'][:_k]):.4f} / {np.mean(_h['true_reward'][-_k:]):.4f}、"
        f"応答の長さ {np.mean(_h['response_length'][:_k]):.1f} / {np.mean(_h['response_length'][-_k:]):.1f}、"
        f"クリップの割合(平均){np.mean(_h['clip_fraction']):.3f}"
    )
    print(f"  最後の反復の応答の例: {_h['example'][-1]!r}")
PPO_SECONDS = time.time() - _t0_ppo
print(
    f"PPO の合計: {PPO_SECONDS:.1f}s(上限の宣言 {PPO_TIME_CAP_SECONDS / 60:.0f} 分は見積もりに対するもの)"
)


def moving_average(values, window: int = 10) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    window = max(1, min(window, len(values)))
    return np.convolve(values, np.ones(window) / window, mode="valid")


_fig, _axes = plt.subplots(1, 3, figsize=(15, 4))
for _i, (_beta, _h) in enumerate(PPO_HISTORIES.items()):
    for _ax, _key, _label in zip(
        _axes,
        ("kl", "proxy_reward", "true_reward"),
        ("KL(pi || pi_ref) estimate (nats)", "proxy reward r_phi", "true reward r*"),
        strict=True,
    ):
        _ax.plot(_h[_key], color=f"C{_i}", alpha=0.25)
        _smoothed = moving_average(_h[_key])
        _ax.plot(
            np.arange(len(_smoothed)) + (len(_h[_key]) - len(_smoothed)),
            _smoothed,
            color=f"C{_i}",
            label=f"beta = {_beta}",
        )
        _ax.set_xlabel("PPO iteration")
        _ax.set_ylabel(_label)
_axes[0].legend()
plt.tight_layout()
plt.show()
```

    beta = 0.01: 127 反復(248.9s)。最初 / 最後の 10 反復の平均: KL 0.057 / 4.427、代理報酬 1.446 / 1.656、真の報酬 0.7165 / 0.6594、応答の長さ 12.9 / 12.6、クリップの割合(平均)0.004
      最後の反復の応答の例: ' degree security director coach share compan\n### End'
    beta = 0.1: 127 反復(245.7s)。最初 / 最後の 10 反復の平均: KL 0.044 / 0.534、代理報酬 1.456 / 1.542、真の報酬 0.7162 / 0.7136、応答の長さ 12.8 / 12.7、クリップの割合(平均)0.002
      最後の反復の応答の例: ' degree gas director judge web industry\n### End'
    PPO の合計: 494.7s(上限の宣言 15 分は見積もりに対するもの)



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/017_reward_model_and_rlhf/output_44_1.png)
    


### 6.12 不変条件のアサーションと`SMOKE_TEST`の配線

- 応答プールが、全ての報酬モデルのスコアリングの後も生成直後と同一(ハッシュが一致)であり、全シード・全水準で同じプール・
  同じ評価用のプロンプトを使った。
- 報酬モデルの学習用・評価用・PPO 用のプロンプトの入力が互いに素(5.4 節で確認済みのものを再確認)。
- 全ての報酬モデルの学習ステップ数が $T_{\mathrm{RM}}$ で、組の順序のハッシュが同じ引数で作り直した値と一致し、データ量の
  水準が入れ子(学習用の先頭 $N$ 組)である。
- 参照方策の重みが、全ての実験の後もマージ直後と bit 単位で一致する(報酬モデル・PPO は複製を学習した)。ベースモデルの重みが
  Hub から読み込んだ重みと一致する。
- `SMOKE_TEST`の配線と削る段階: 実際に使ったシード・水準・$M$・$n$ の水準・PPO の反復回数・ステップ数・ブートストラップの
  反復回数が、5.2 節の水準と 6.4 節で選ばれた段階と一致する。


```python
assert hash_json(POOL) == POOL_HASH, "応答プールが変わった"
assert STANDARD_SCORES.shape == (NUM_SEEDS_AB, NUM_EVAL_PROMPTS, POOL_SIZE)
assert all(r["pool_scores"].shape == (NUM_EVAL_PROMPTS, POOL_SIZE) for r in RM_RECORDS.values())
assert not (set(EVAL_INPUTS) & set(RM_TRAIN_INPUTS)) and not (set(EVAL_INPUTS) & set(PPO_INPUTS))
assert not (set(RM_TRAIN_INPUTS) & set(PPO_INPUTS))
for (_n, _s), _r in RM_RECORDS.items():
    assert len(_r["history"]["step"]) == RM_STEPS
    assert _r["history"]["batches_hash"] == hash_json(
        make_epoch_batches(_n, RM_STEPS, RM_BATCH_PAIRS, _s)
    )
    assert (
        max(i for b in make_epoch_batches(_n, RM_STEPS, RM_BATCH_PAIRS, _s) for i in b) == _n - 1
    )  # 先頭 N 組のみ
assert sorted(RM_RECORDS) == sorted(RM_JOBS) and len(RM_RECORDS) == num_reward_models(STAGE)
assert all(len(LABELS[s]) == RM_PAIRS_MAX for s in SEEDS_AB)
_reference_now = {k: v.detach().cpu() for k, v in reference_policy.state_dict().items()}
assert all(torch.equal(_reference_now[k], REFERENCE_STATE[k]) for k in REFERENCE_STATE), (
    "参照方策の重みが変わった"
)
_hub_state = torch.load(MODEL_STATE_PATH, map_location="cpu")
assert all(torch.equal(v.cpu(), _hub_state[k]) for k, v in base_model.state_dict().items())
# SMOKE_TEST の配線と削る段階
assert STAGES[CURRENT_LEVEL_NAME][SELECTED_STAGE] == STAGE
assert tuple(range(STAGE["NUM_SEEDS_AB"])) == SEEDS_AB and tuple(
    range(STAGE["NUM_SEEDS_C"])
) == SEEDS_C
assert rm_levels(STAGE["C_NUM_LEVELS"], CFG["RM_PAIRS_MAX"]) == C_LEVELS
assert CFG["RM_PAIRS_MAX"] // RM_BATCH_PAIRS == RM_STEPS and CFG["SFT_STEPS"] == SFT_STEPS
assert len(SFT_HISTORY["step"]) == CFG["SFT_STEPS"]
assert CFG["POOL_SIZE"] == POOL_SIZE and all(len(r) == POOL_SIZE for r in POOL)
assert (
    CFG["NUM_EVAL_INPUTS"] * TASK_COUNT == NUM_EVAL_PROMPTS
    and len(P0_EXAMPLES) == CFG["NUM_P0_INPUTS"] * TASK_COUNT
)
assert BON_LEVELS[-1] == CFG["POOL_SIZE"] // 4 and BOOTSTRAP_COUNTS.shape == (
    CFG["BOOTSTRAP_RESAMPLES"],
    NUM_EVAL_INPUTS,
)
assert ppo_iterations_for(STAGE, CFG) == PPO_ITERATIONS
assert all(len(h["kl"]) == PPO_ITERATIONS for h in PPO_HISTORIES.values())
print(
    f"応答プールの同一性、プロンプトの分割、全 {len(RM_RECORDS)} 報酬モデルのステップ数・組の順序・入れ子、"
    "参照方策とベースモデルの重みの不変性: OK"
)
print(
    f"SMOKE_TEST の配線と削る段階: 水準 {CURRENT_LEVEL_NAME!r}・段階 {SELECTED_STAGE}(実験 A・B のシード {SEEDS_AB}、"
    f"実験 C のシード {SEEDS_C}・水準 {C_LEVELS}、M {POOL_SIZE}、n {BON_LEVELS}、T_SFT {SFT_STEPS}、T_RM {RM_STEPS}、"
    f"PPO {PPO_ITERATIONS} 反復、反復 {BOOTSTRAP_RESAMPLES})が一致: OK"
)
```

    応答プールの同一性、プロンプトの分割、全 20 報酬モデルのステップ数・組の順序・入れ子、参照方策とベースモデルの重みの不変性: OK
    SMOKE_TEST の配線と削る段階: 水準 'prod'・段階 0(実験 A・B のシード (0, 1, 2, 3, 4)、実験 C のシード (0, 1, 2, 3, 4)・水準 (1024, 4096, 16384, 65536)、M 512、n (1, 2, 4, 8, 16, 32, 64, 128)、T_SFT 2048、T_RM 2048、PPO 127 反復、反復 10000)が一致: OK


### 6.13 判定結果の一覧

前提条件が 1 つでも不成立の実験は、判定関数の結果に関わらず「前提不成立」とする(判定不能とは区別する)。判定関数の結果は
参考として別欄に残す。選ばれた段階と、判定に実際に使ったシード数・水準を印字する。


```python
def verdict_label(computed: str, preconditions: list[str]) -> str:
    return computed if all(precondition_status.get(p) for p in preconditions) else "前提不成立"


print(f"{_tag}段階の選択: {STAGE_SELECTION_MESSAGE}")
print(
    f"{_tag}判定に使った値: 実験 A・B は S = {NUM_SEEDS_AB}(シード {SEEDS_AB})、実験 C は S = {NUM_SEEDS_C}"
    f"(シード {SEEDS_C})・水準 {C_LEVELS}、M = {POOL_SIZE}、n = {BON_LEVELS}"
)
PRECONDITIONS = {"A": ["P0", "P1"], "B": ["P0", "P1", "P2"], "C": ["P0", "P1", "P2"]}
print(f"{_tag}実験 | 対比量 | 前提条件の成否 | 判定関数の結果(参考) | 最終判定")
for _name, _stat, _computed in (
    (
        "A",
        f"較正の傾き rho = {RHO:.4f}、sigma = {SIGMA_RHO:.4f}、区間 [{RHO - 2 * SIGMA_RHO:.4f}, {RHO + 2 * SIGMA_RHO:.4f}]",
        verdict_A,
    ),
    (
        "B",
        f"g1 = {_g1['value']:+.5f}(sigma {_g1['sigma']:.5f})、g2 = {_g2['value']:+.5f}(sigma {_g2['sigma']:.5f})",
        verdict_B,
    ),
    ("C", f"c = {C_SLOPE:+.5f}、sigma = {SIGMA_C:.5f}", verdict_C),
):
    _pre = ", ".join(f"{p}={precondition_status.get(p)}" for p in PRECONDITIONS[_name])
    print(
        f"{_tag}{_name} | {_stat} | {_pre} | {_computed} | {verdict_label(_computed, PRECONDITIONS[_name])}"
    )
if SMOKE_TEST:
    print(
        "上記はスモークテストによるコードの動作確認であり、本番実行の後に数値が変わるため、結論としては扱わない。"
    )
print(f"\nノートブック全体の実行時間: {(time.time() - NOTEBOOK_START_TIME) / 60:.1f} 分")
```

    段階の選択: 予算 120 分に収まる最小の段階として、段階 0 を選んだ(見積もり 83.3 分、cuda 基準)
    判定に使った値: 実験 A・B は S = 5(シード (0, 1, 2, 3, 4))、実験 C は S = 5(シード (0, 1, 2, 3, 4))・水準 (1024, 4096, 16384, 65536)、M = 512、n = (1, 2, 4, 8, 16, 32, 64, 128)
    実験 | 対比量 | 前提条件の成否 | 判定関数の結果(参考) | 最終判定
    A | 較正の傾き rho = 1.0504、sigma = 0.0254、区間 [0.9996, 1.1013] | P0=True, P1=True | 支持 | 支持
    B | g1 = +0.01552(sigma 0.00239)、g2 = -0.00762(sigma 0.00056) | P0=True, P1=True, P2=True | 支持 | 支持
    C | c = +0.00562、sigma = 0.00185 | P0=True, P1=True, P2=True | 支持 | 支持
    
    ノートブック全体の実行時間: 62.3 分




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/04_alignment/017_reward_model_and_rlhf.ipynb
