---
title: "DPO(Direct Preference Optimization) / Direct Preference Optimization(実装・実験編 3/4)"
---

この記事は後編(実装・実験編 3/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/018_direct_preference_optimization-practice-2)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/018_direct_preference_optimization-practice-4)。

### 6.5 選好の組・$\kappa$・ラベル・実験 D の層

- 学習用の各プロンプトについて $\pi_{\mathrm{ref}}$ から 2 応答を抽選し(乱数シード`18019`)、$y_1 = y_2$ と $\Delta r^* = 0$ の組を除外する。
  実験 A〜C の学習データは除外後の先頭 $N$ 組。
- $\kappa$ を決め、シードごとに確率的なラベル($K$ 回の独立な抽選)を、全シード共通の決定的なラベルを作る。
- 実験 D: 課題 × $\lvert \Delta r^* \rvert$ の区間の層ごとに、近い組・遠い組を同数選ぶ。
- 評価用の組: 評価用の各プロンプトについて $\pi_{\mathrm{ref}}$ から 2 応答を抽選し(乱数シード`18020`)、同じ除外をする
  (実験 B の診断量。真の報酬の高い方を選好とした向きで測る)。


```python
def generate_candidates(
    prompts: list, examples: list, seed: int, reward_cache: bool = True
) -> dict:
    flat = sample_responses(reference_policy, [p for p in prompts for _ in range(2)], seed)
    pairs = {"prompts": prompts, "first": flat[0::2], "second": flat[1::2]}
    if reward_cache:
        first_rewards, second_rewards = (
            true_rewards(pairs["first"], examples),
            true_rewards(pairs["second"], examples),
        )
    else:  # キャッシュを使わずに計算する(6.6 節の確認用)
        first_rewards = np.array(
            [
                compute_true_reward(tokenizer.decode(r), e.answer)
                for r, e in zip(pairs["first"], examples, strict=True)
            ]
        )
        second_rewards = np.array(
            [
                compute_true_reward(tokenizer.decode(r), e.answer)
                for r, e in zip(pairs["second"], examples, strict=True)
            ]
        )
    pairs["first_reward"], pairs["second_reward"] = first_rewards, second_rewards
    pairs["distance"] = np.array(
        [
            normalized_edit_distance(tokenizer.decode(a), tokenizer.decode(b))
            for a, b in zip(pairs["first"], pairs["second"], strict=True)
        ]
    )
    pairs["valid"] = (pairs["distance"] > 0) & (first_rewards != second_rewards)
    return pairs


_t0 = time.time()
CANDIDATES = generate_candidates(TRAIN_PROMPT_IDS, TRAIN_EXAMPLES, PAIR_SEED)
VALID_INDEX = np.flatnonzero(CANDIDATES["valid"])  # 除外後の候補(候補の組の添字)
assert len(VALID_INDEX) >= NUM_MAIN_PAIRS, (len(VALID_INDEX), NUM_MAIN_PAIRS)
MAIN_ROWS = np.arange(NUM_MAIN_PAIRS)  # 実験 A〜C: 除外後の先頭 N 組(VALID_INDEX の行)
VALID_DELTA = CANDIDATES["first_reward"][VALID_INDEX] - CANDIDATES["second_reward"][VALID_INDEX]
KAPPA = calibrate_label_scale(VALID_DELTA, KAPPA_TARGET)
_p_main = compute_bradley_terry_probability(VALID_DELTA[MAIN_ROWS], KAPPA)
EXPECTED_MIXED_FRACTION = float(np.mean(1 - _p_main**NUM_DRAWS - (1 - _p_main) ** NUM_DRAWS))
STOCHASTIC_LABELS = {  # シード s の確率的なラベル(除外後の候補すべて、(len(VALID_INDEX), K))
    s: sample_preference_label_matrix(
        CANDIDATES["first_reward"][VALID_INDEX],
        CANDIDATES["second_reward"][VALID_INDEX],
        KAPPA,
        NUM_DRAWS,
        np.random.default_rng(LABEL_SEED_BASE + s),
    )
    for s in ALL_SEEDS
}
DETERMINISTIC_LABELS = deterministic_preference_label_matrix(
    CANDIDATES["first_reward"][VALID_INDEX], CANDIDATES["second_reward"][VALID_INDEX], NUM_DRAWS
)

# --- 実験 D の層と、近い組・遠い組 ---
_abs_delta = np.abs(VALID_DELTA)
D_BIN_EDGES = np.quantile(_abs_delta, [q / D_REWARD_BINS for q in range(1, D_REWARD_BINS)])
_bins = np.searchsorted(D_BIN_EDGES, _abs_delta, side="right")
D_STRATA = [(TRAIN_EXAMPLES[i].task, int(b)) for i, b in zip(VALID_INDEX, _bins, strict=True)]
NEAR_ROWS, FAR_ROWS, D_COUNTS = select_similarity_stratified_pairs(
    D_STRATA, CANDIDATES["distance"][VALID_INDEX], D_FRACTION
)
assert all(c["near"] == c["far"] for c in D_COUNTS.values()) and len(NEAR_ROWS) == len(FAR_ROWS)
assert not (set(NEAR_ROWS.tolist()) & set(FAR_ROWS.tolist()))
assert len(D_COUNTS) == TASK_COUNT * D_REWARD_BINS
for _label in D_COUNTS:  # 層ごとの件数の一致(課題と |Delta r*| の区間の分布が揃う)
    assert sum(D_STRATA[r] == _label for r in NEAR_ROWS) == sum(
        D_STRATA[r] == _label for r in FAR_ROWS
    )

# --- 評価用の組(実験 B の診断量) ---
EVAL_PAIRS = generate_candidates(EVAL_PROMPT_IDS, EVAL_EXAMPLES, EVAL_PAIR_SEED)
_eval_valid = np.flatnonzero(EVAL_PAIRS["valid"])
_eval_labels = deterministic_preference_label_matrix(
    EVAL_PAIRS["first_reward"][_eval_valid], EVAL_PAIRS["second_reward"][_eval_valid], 1
)
PAIRS_SECONDS = time.time() - _t0
CANDIDATES_HASH = hash_json([CANDIDATES["first"], CANDIDATES["second"]])

print(f"選好の組の候補: {NUM_CANDIDATE_PAIRS:,} 組({PAIRS_SECONDS:.1f}s)")
print(
    f"除外: y_1 = y_2 {np.mean(CANDIDATES['distance'] == 0):.4f}、同一でない応答の同点 "
    f"{np.mean((CANDIDATES['distance'] > 0) & (CANDIDATES['first_reward'] == CANDIDATES['second_reward'])):.4f} "
    f"-> 除外後 {len(VALID_INDEX):,} 組(実験 A〜C は先頭 {NUM_MAIN_PAIRS:,} 組)"
)
print(
    f"kappa = log 3 / median|Delta r*| = {KAPPA:.4f}(|Delta r*| の中央値 {np.median(_abs_delta):.4f})"
)
print(
    f"K = {NUM_DRAWS}: 実験 A〜C の組での E[p_hat が (0, 1)] = {EXPECTED_MIXED_FRACTION:.4f}(目標 >= {EXPECTED_MIXED_TARGET})"
)
print(
    f"実験 D: 層 {len(D_COUNTS)}(課題 x |Delta r*| の区間、区切り {np.round(D_BIN_EDGES, 4).tolist()})、"
    f"近い組 {len(NEAR_ROWS):,}・遠い組 {len(FAR_ROWS):,}(層ごとの件数が一致: OK)"
)
print(f"評価用の組: {len(EVAL_PAIRS['first'])} 組のうち除外後 {len(_eval_valid)} 組")
```

    選好の組の候補: 8,192 組(30.6s)
    除外: y_1 = y_2 0.0039、同一でない応答の同点 0.0223 -> 除外後 7,977 組(実験 A〜C は先頭 2,048 組)
    kappa = log 3 / median|Delta r*| = 12.8373(|Delta r*| の中央値 0.0856)
    K = 4: 実験 A〜C の組での E[p_hat が (0, 1)] = 0.5709(目標 >= 0.5)
    実験 D: 層 16(課題 x |Delta r*| の区間、区切り [0.0392, 0.0856, 0.1647])、近い組 1,990・遠い組 1,990(層ごとの件数が一致: OK)
    評価用の組: 512 組のうち除外後 504 組


### 6.6 キャッシュの確認と、学習データの不変条件

選好の組(サンプリングと真の報酬)と参照方策の対数確率は、1 度だけ計算して全条件・全シードで再利用する(キャッシュ)。
キャッシュの導入前後で数値が完全に一致することを、次の手順で確かめる。

1. 選好の組の候補を同じ乱数シードで **生成し直し**、真の報酬をキャッシュを使わずに計算し直して、6.5 節の結果と
   bit 単位で一致すること。
2. 実験 A〜C の学習データ(シード 0 の確率的なラベル)を、参照方策の対数確率のキャッシュを使って作った場合と、使わずに
   計算し直した場合で、事例と $\log \pi_{\mathrm{ref}}$ が bit 単位で一致すること。
3. 2 つのデータで DPO の学習(24 ステップ)と方策の評価を行い、記録(損失・マージン・対数比・評価の値)が bit 単位で
   一致すること。

あわせて、学習データの不変条件を確かめる: 確率的なラベルと決定的なラベルの学習データで、組の集合と事例数が同一であること。


```python
_t0_cache = time.time()
EVAL_PAIR_DATASET = build_dataset(
    EVAL_PAIRS, _eval_valid, _eval_labels.repeat(NUM_DRAWS, axis=1), REFERENCE_LOG_PROB_CACHE
)

# 1. 選好の組の生成し直し(キャッシュなし)
_regenerated = generate_candidates(TRAIN_PROMPT_IDS, TRAIN_EXAMPLES, PAIR_SEED, reward_cache=False)
assert _equal(_regenerated, CANDIDATES), "生成し直した選好の組が一致しない"


def main_dataset(label_kind: str, seed: int, cache: dict | None) -> dict:
    matrix = STOCHASTIC_LABELS[seed] if label_kind == "stochastic" else DETERMINISTIC_LABELS
    return build_dataset(CANDIDATES, VALID_INDEX[MAIN_ROWS], matrix[MAIN_ROWS], cache)


def d_dataset(name: str, seed: int, cache: dict | None) -> dict:
    rows = NEAR_ROWS if name == "near" else FAR_ROWS
    return build_dataset(CANDIDATES, VALID_INDEX[rows], STOCHASTIC_LABELS[seed][rows], cache)


# 2. 参照方策の対数確率: キャッシュあり(2 回目はキャッシュから)とキャッシュなし
_with_cache = main_dataset("stochastic", 0, REFERENCE_LOG_PROB_CACHE)
_hits_before = CACHE_STATS["hit"]
_with_cache_again = main_dataset("stochastic", 0, REFERENCE_LOG_PROB_CACHE)
assert CACHE_STATS["hit"] == _hits_before + 2, "2 回目がキャッシュから返されていない"
_without_cache = main_dataset("stochastic", 0, None)
assert _equal(_with_cache, _without_cache) and _equal(_with_cache_again, _without_cache)

# 3. 学習と評価
_cache_steps = min(TRAIN_STEPS, 24)
_record_cached = run_training(
    _with_cache_again, "dpo", CHECK_BETA, 0, "full", num_steps=_cache_steps
)
_record_uncached = run_training(
    _without_cache, "dpo", CHECK_BETA, 0, "full", num_steps=_cache_steps
)
for _r in (_record_cached, _record_uncached):
    _r.pop("train_seconds"), _r.pop("eval_seconds")
assert _equal(_record_cached, _record_uncached), "キャッシュの有無で学習・評価の記録が一致しない"

# 学習データの不変条件: 確率的・決定的なラベルで組の集合と事例数が同一
_deterministic = main_dataset("deterministic", 0, REFERENCE_LOG_PROB_CACHE)
assert np.array_equal(_deterministic["pair_indices"], _with_cache["pair_indices"])
assert (
    len(_deterministic["instances"]["prompts"])
    == len(_with_cache["instances"]["prompts"])
    == NUM_MAIN_PAIRS * NUM_DRAWS
)
assert (
    _deterministic["prompts"] == _with_cache["prompts"]
    and _deterministic["first"] == _with_cache["first"]
)
CACHE_CHECK_SECONDS = time.time() - _t0_cache
print(
    f"キャッシュの確認: 選好の組の生成し直し・参照方策の対数確率(キャッシュなし)・DPO の {_cache_steps} ステップの学習と"
    f"評価が、キャッシュを使った場合と bit 単位で一致: OK({CACHE_CHECK_SECONDS:.1f}s)"
)
print(
    f"学習データ: 確率的なラベルと決定的なラベルで組の集合({len(_with_cache['pair_indices']):,} 組)と事例数"
    f"({NUM_MAIN_PAIRS * NUM_DRAWS:,})が同一: OK"
)
del _regenerated, _with_cache, _with_cache_again, _without_cache, _deterministic
```

    キャッシュの確認: 選好の組の生成し直し・参照方策の対数確率(キャッシュなし)・DPO の 24 ステップの学習と評価が、キャッシュを使った場合と bit 単位で一致: OK(54.3s)
    学習データ: 確率的なラベルと決定的なラベルで組の集合(2,048 組)と事例数(8,192)が同一: OK


### 6.7 $\beta$ の較正(KL ダイバージェンスのみ)

6.1 節の規則で、標準の $\beta$ と実験 A の水準を決める。各格子点で、実験 A と同じデータ(シード 0 の確率的なラベル)・同じ
ステップ数・シード 0 で DPO を学習し、較正用のプロンプトで $\widehat{\mathrm{KL}}$ **のみ** を測る。真の報酬はこのセルでは
計算しない。学習の進みの確認のため、学習用の事例での損失と正解率も印字する(仮説に関わる量ではない)。診断量として、
応答の平均トークン数と終端記号で止まった応答の割合も印字する(水準の規則には使わない。KL の急増が、終端記号を出さずに
最大長まで生成することによる推定値の膨張かどうかを読むため)。


```python
_t0_calibration = time.time()
_calibration_dataset = main_dataset("stochastic", 0, REFERENCE_LOG_PROB_CACHE)
CALIBRATION_RECORDS = {}
for _beta in sorted(BETA_GRID, reverse=True):
    _r = run_training(_calibration_dataset, "dpo", _beta, 0, "kl_only")
    assert "true_reward_by_input" not in _r["eval"]  # 較正では真の報酬を計算しない
    CALIBRATION_RECORDS[_beta] = _r
    print(
        f"beta = {_beta:.6f}: KL(較正用のプロンプト)= {_r['eval']['kl']:.4f} nats、学習の最後の 10% の損失 "
        f"{np.mean(_r['history']['loss'][-max(1, TRAIN_STEPS // 10) :]):.4f}、正解率 a(T) = {_r['final']['accuracy']:.4f}、"
        f"診断量: 応答の平均トークン数 {_r['eval']['mean_tokens']:.2f}・終端記号で止まった割合 {_r['eval']['stop_rate']:.4f}"
        f"({_r['train_seconds'] + _r['eval_seconds']:.1f}s)"
    )


def choose_standard_beta(kl_by_beta: dict[float, float]) -> tuple[float, str]:
    # 大きい beta から順に見て、KL が初めて STANDARD_KL_MAX を超える直前の格子点(雑音による非単調性に頑健)
    chosen = None
    for b in sorted(kl_by_beta, reverse=True):
        if kl_by_beta[b] > STANDARD_KL_MAX:
            break
        chosen = b
    if chosen is None:
        return max(kl_by_beta), f"最大の格子点でも KL > {STANDARD_KL_MAX} のため、最大の格子点"
    return chosen, f"大きい beta から見て KL <= {STANDARD_KL_MAX} が続く最小の格子点"


def grid_index(beta: float) -> int:
    grid = sorted(BETA_GRID, reverse=True)  # 降順
    index = min(range(len(grid)), key=lambda i: abs(grid[i] - beta))
    assert math.isclose(grid[index], beta)
    return index


def choose_beta_levels(standard_beta: float) -> tuple[tuple[float, ...], str]:
    # beta_6 = 標準の beta、beta_j = beta_6 2^((6 - j)/2)(格子の連続する 6 点)。beta_1 が格子の外なら上端の 6 点
    grid, index = sorted(BETA_GRID, reverse=True), grid_index(standard_beta)
    if index >= NUM_A_LEVELS - 1:
        return tuple(
            grid[index - NUM_A_LEVELS + 1 : index + 1]
        ), "beta_6 = 標準の beta から公比 sqrt(2) で 6 点"
    return tuple(grid[:NUM_A_LEVELS]), "beta_1 = beta_6 2^(5/2) が格子の外なので、格子の上端の 6 点"


def collapse_betas(beta_6: float) -> tuple[float, ...]:
    # 崩壊の領域の観察: 実験 A の最小の水準 beta_6 から格子を 2 点・4 点下った beta(beta_6 / 2・beta_6 / 4。
    # 格子の下端を下回る場合は下端)。起点は標準の beta ではなく beta_6(格子の上端の 6 点の経路では両者が異なる)
    grid, index = sorted(BETA_GRID, reverse=True), grid_index(beta_6)
    # 打ち切りで同じ点になった場合は 1 つにまとめる(beta_6 自体と一致する場合は実験 A の学習を共有する)
    return tuple(dict.fromkeys(grid[min(index + k, len(grid) - 1)] for k in COLLAPSE_OFFSETS))


# 規則の確認(人工の KL・人工の標準の beta で)
assert choose_standard_beta({b: 1.0 / b for b in BETA_GRID})[0] == min(
    b for b in BETA_GRID if 1.0 / b <= STANDARD_KL_MAX
)
assert (
    choose_standard_beta({1.0: 0.5, 0.5: 3.0, 0.25: 1.0})[0] == 1.0
)  # 一度超えたら、それより小さい beta は選ばない
_grid_desc = sorted(BETA_GRID, reverse=True)
assert choose_beta_levels(_grid_desc[7])[0] == tuple(_grid_desc[2:8])  # beta_6 = 標準の beta
assert choose_beta_levels(_grid_desc[5])[0] == tuple(_grid_desc[0:6])  # ちょうど上端の 6 点
assert choose_beta_levels(_grid_desc[2])[0] == tuple(
    _grid_desc[0:6]
)  # beta_1 が格子の外: 上端の 6 点
assert collapse_betas(_grid_desc[5]) == (_grid_desc[7], _grid_desc[9])
assert collapse_betas(_grid_desc[9]) == (_grid_desc[10],)  # 下端で打ち切り、同じ点は 1 つにまとめる
assert collapse_betas(_grid_desc[10]) == (
    _grid_desc[10],
)  # beta_6 が下端: beta_6 自体(実験 A と共有)
# 上端の 6 点の経路(標準の beta = _grid_desc[2]): 観察の起点は beta_6 = _grid_desc[5] で、実験 A の水準と重ならない
_levels_top = choose_beta_levels(_grid_desc[2])[0]
assert collapse_betas(_levels_top[-1]) == (_grid_desc[7], _grid_desc[9])
assert not set(collapse_betas(_levels_top[-1])) & set(_levels_top)

STANDARD_BETA, STANDARD_BETA_RULE = choose_standard_beta(
    {b: r["eval"]["kl"] for b, r in CALIBRATION_RECORDS.items()}
)
A_BETAS, A_BETA_RULE = choose_beta_levels(STANDARD_BETA)
assert len(A_BETAS) == NUM_A_LEVELS and all(
    math.isclose(a / b, math.sqrt(2.0)) for a, b in zip(A_BETAS, A_BETAS[1:], strict=False)
)
_levels_index = [grid_index(b) for b in A_BETAS]
assert _levels_index == list(
    range(_levels_index[0], _levels_index[0] + NUM_A_LEVELS)
)  # 格子の連続する 6 点
COLLAPSE_BETAS = collapse_betas(A_BETAS[-1])  # 起点は beta_6(実験 A の最小の水準)
assert all(b <= A_BETAS[-1] for b in COLLAPSE_BETAS)
assert not set(COLLAPSE_BETAS) & set(
    A_BETAS[:-1]
)  # 重なりうるのは beta_6 自体(格子の下端で打ち切った場合)のみ
STANDARD_IN_A = any(math.isclose(b, STANDARD_BETA) for b in A_BETAS)
CALIBRATION_SECONDS = time.time() - _t0_calibration
print(f"\n実験 B〜D の標準の beta(IPO の tau): {STANDARD_BETA:.6g}({STANDARD_BETA_RULE})")
print(f"実験 A の水準(beta_1 > ... > beta_6): {[round(b, 6) for b in A_BETAS]}({A_BETA_RULE})")
print(f"崩壊の領域の観察(判定なし、シード 0 のみ): beta = {[round(b, 6) for b in COLLAPSE_BETAS]}")
print(
    f"標準の beta = {STANDARD_BETA:.6g} は実験 A の水準に{'含まれる(実験 B の条件 2 と実験 C は実験 A の学習を共有する)' if STANDARD_IN_A else '含まれない(実験 B の条件 2 は別に学習する)'}"
    f"。較正の合計 {CALIBRATION_SECONDS:.1f}s"
)
```

    beta = 3.200000: KL(較正用のプロンプト)= 0.0415 nats、学習の最後の 10% の損失 0.6020、正解率 a(T) = 0.7668、診断量: 応答の平均トークン数 12.96・終端記号で止まった割合 0.9873(43.3s)
    beta = 2.262742: KL(較正用のプロンプト)= 0.0684 nats、学習の最後の 10% の損失 0.6017、正解率 a(T) = 0.7644、診断量: 応答の平均トークン数 12.93・終端記号で止まった割合 0.9893(43.4s)
    beta = 1.600000: KL(較正用のプロンプト)= 0.1168 nats、学習の最後の 10% の損失 0.5714、正解率 a(T) = 0.7659、診断量: 応答の平均トークン数 13.02・終端記号で止まった割合 0.9834(44.5s)
    beta = 1.131371: KL(較正用のプロンプト)= 0.2139 nats、学習の最後の 10% の損失 0.5707、正解率 a(T) = 0.7615、診断量: 応答の平均トークン数 13.18・終端記号で止まった割合 0.9814(44.0s)
    beta = 0.800000: KL(較正用のプロンプト)= 0.3742 nats、学習の最後の 10% の損失 0.5654、正解率 a(T) = 0.7598、診断量: 応答の平均トークン数 13.13・終端記号で止まった割合 0.9824(44.6s)
    beta = 0.565685: KL(較正用のプロンプト)= 0.5958 nats、学習の最後の 10% の損失 0.5732、正解率 a(T) = 0.7505、診断量: 応答の平均トークン数 13.23・終端記号で止まった割合 0.9736(44.7s)
    beta = 0.400000: KL(較正用のプロンプト)= 3.9775 nats、学習の最後の 10% の損失 0.5740、正解率 a(T) = 0.7422、診断量: 応答の平均トークン数 16.11・終端記号で止まった割合 0.8359(45.4s)
    beta = 0.282843: KL(較正用のプロンプト)= 15.1080 nats、学習の最後の 10% の損失 0.5797、正解率 a(T) = 0.7419、診断量: 応答の平均トークン数 21.71・終端記号で止まった割合 0.5654(46.3s)
    beta = 0.200000: KL(較正用のプロンプト)= 28.8673 nats、学習の最後の 10% の損失 0.5805、正解率 a(T) = 0.7307、診断量: 応答の平均トークン数 26.09・終端記号で止まった割合 0.3271(45.6s)
    beta = 0.141421: KL(較正用のプロンプト)= 39.8408 nats、学習の最後の 10% の損失 0.6066、正解率 a(T) = 0.6775、診断量: 応答の平均トークン数 26.95・終端記号で止まった割合 0.2646(45.9s)
    beta = 0.100000: KL(較正用のプロンプト)= 61.3698 nats、学習の最後の 10% の損失 0.6022、正解率 a(T) = 0.6987、診断量: 応答の平均トークン数 30.64・終端記号で止まった割合 0.0889(46.7s)
    
    実験 B〜D の標準の beta(IPO の tau): 0.565685(大きい beta から見て KL <= 2.0 が続く最小の格子点)
    実験 A の水準(beta_1 > ... > beta_6): [3.2, 2.262742, 1.6, 1.131371, 0.8, 0.565685](beta_6 = 標準の beta から公比 sqrt(2) で 6 点)
    崩壊の領域の観察(判定なし、シード 0 のみ): beta = [0.282843, 0.141421]
    標準の beta = 0.565685 は実験 A の水準に含まれる(実験 B の条件 2 と実験 C は実験 A の学習を共有する)。較正の合計 494.5s


### 6.8 学習と評価(実験 A・B・C・D)

すべての学習を 1 つの辞書`RECORDS`に、鍵(損失・ラベル・データセット・$\beta$・シード)で記録する。同じ鍵の学習は 1 回だけ行う
実験 A の直後に、崩壊の領域の観察(6.10 節)の 2 学習を行う。(標準の $\beta$ が実験 A の水準に含まれる場合、その水準の実験 A と実験 B の条件 2 は同じ鍵になり、共有される。実験 C は実験 B の条件 2 の記録を使う)。学習した方策は
評価の直後に破棄する。


```python
RECORDS: dict[tuple, dict] = {}
_datasets: dict[tuple, dict] = {}


def dataset_for(name: str, label_kind: str, seed: int) -> dict:
    key = (name, label_kind, seed if label_kind == "stochastic" else None)
    if key not in _datasets:
        if name == "main":
            _datasets[key] = main_dataset(label_kind, seed, REFERENCE_LOG_PROB_CACHE)
        else:
            assert label_kind == "stochastic"
            _datasets[key] = d_dataset(name, seed, REFERENCE_LOG_PROB_CACHE)
    return _datasets[key]


def run_condition(loss_type: str, label_kind: str, name: str, beta: float, seed: int) -> dict:
    key = (loss_type, label_kind, name, round(beta, 9), seed)
    if key not in RECORDS:
        record = run_training(dataset_for(name, label_kind, seed), loss_type, beta, seed, "full")
        record["label_kind"], record["dataset"] = label_kind, name
        RECORDS[key] = record
        f, e = record["final"], record["eval"]
        print(
            f"{loss_type.upper()}・{label_kind}・{name}・beta={beta:.6g}・seed={seed}: 損失 {record['history']['loss'][-1]:.4f}、"
            f"a(T) {f['accuracy']:.4f}、h(T) {f['margin']:+.3f}、Delta_w(T) {f['chosen_log_ratio']:+.3f}、"
            f"KL {e['kl']:.3f}({record['train_seconds'] + record['eval_seconds']:.1f}s)"
        )
    return RECORDS[key]


_t0_runs = time.time()
print("--- 実験 A ---")
for _s in SEEDS_A:
    for _beta in A_BETAS:
        run_condition("dpo", "stochastic", "main", _beta, _s)
print("--- 崩壊の領域の観察(判定なし、シード 0 のみ)---")
for _beta in COLLAPSE_BETAS:
    run_condition("dpo", "stochastic", "main", _beta, 0)
print("--- 実験 B(と C)---")
B_CONDITIONS = {
    1: ("dpo", "deterministic"),
    2: ("dpo", "stochastic"),
    3: ("ipo", "deterministic"),
    4: ("ipo", "stochastic"),
}
for _s in SEEDS_B:
    for _c, (_loss, _kind) in B_CONDITIONS.items():
        run_condition(_loss, _kind, "main", STANDARD_BETA, _s)
print("--- 実験 D ---")
for _s in SEEDS_D:
    for _name in ("near", "far"):
        run_condition("dpo", "stochastic", _name, STANDARD_BETA, _s)
RUNS_SECONDS = time.time() - _t0_runs
print(
    f"\n学習と評価の合計: {len(RECORDS)} 学習、{RUNS_SECONDS:.1f}s(参照方策の対数確率のキャッシュ: "
    f"計算 {CACHE_STATS['miss']} 回・再利用 {CACHE_STATS['hit']} 回)"
)


def key_a(beta: float, seed: int) -> tuple:
    return ("dpo", "stochastic", "main", round(beta, 9), seed)


def key_b(condition: int, seed: int) -> tuple:
    loss_type, label_kind = B_CONDITIONS[condition]
    return (loss_type, label_kind, "main", round(STANDARD_BETA, 9), seed)


def key_d(name: str, seed: int) -> tuple:
    return ("dpo", "stochastic", name, round(STANDARD_BETA, 9), seed)


# --- 学習の成立(共通の前提条件)を実験ごとに記録する ---
def learning_holds(keys: list) -> bool:
    return all(RECORDS[k]["final"]["accuracy"] >= ACCURACY_MIN for k in keys)


precondition_status["学習の成立(A)"] = learning_holds(
    [key_a(b, s) for b in A_BETAS for s in SEEDS_A]
)
precondition_status["学習の成立(B・C)"] = learning_holds(
    [key_b(c, s) for c in B_CONDITIONS for s in SEEDS_B]
)
precondition_status["学習の成立(D)"] = learning_holds(
    [key_d(n, s) for n in ("near", "far") for s in SEEDS_D]
)
for _name in ("A", "B・C", "D"):
    print(
        f"学習の成立({_name}): a(T) >= {ACCURACY_MIN} -> {'成立' if precondition_status[f'学習の成立({_name})'] else '不成立'}"
    )
```

    --- 実験 A ---
    DPO・stochastic・main・beta=3.2・seed=0: 損失 0.4179、a(T) 0.7668、h(T) +0.359、Delta_w(T) +0.069、KL 0.046(50.0s)
    DPO・stochastic・main・beta=2.26274・seed=0: 損失 0.4163、a(T) 0.7644、h(T) +0.486、Delta_w(T) +0.076、KL 0.090(49.5s)
    DPO・stochastic・main・beta=1.6・seed=0: 損失 0.4408、a(T) 0.7659、h(T) +0.636、Delta_w(T) +0.080、KL 0.136(48.5s)
    DPO・stochastic・main・beta=1.13137・seed=0: 損失 0.4554、a(T) 0.7615、h(T) +0.822、Delta_w(T) +0.074、KL 0.178(49.2s)
    DPO・stochastic・main・beta=0.8・seed=0: 損失 0.4288、a(T) 0.7598、h(T) +1.066、Delta_w(T) +0.032、KL 0.374(49.4s)
    DPO・stochastic・main・beta=0.565685・seed=0: 損失 0.4440、a(T) 0.7505、h(T) +1.370、Delta_w(T) -0.050、KL 0.692(49.7s)
    DPO・stochastic・main・beta=3.2・seed=1: 損失 0.5335、a(T) 0.7621、h(T) +0.318、Delta_w(T) +0.058、KL 0.046(50.5s)
    DPO・stochastic・main・beta=2.26274・seed=1: 損失 0.5173、a(T) 0.7604、h(T) +0.455、Delta_w(T) +0.069、KL 0.062(49.4s)
    DPO・stochastic・main・beta=1.6・seed=1: 損失 0.5541、a(T) 0.7582、h(T) +0.550、Delta_w(T) +0.068、KL 0.104(49.0s)
    DPO・stochastic・main・beta=1.13137・seed=1: 損失 0.5456、a(T) 0.7540、h(T) +0.726、Delta_w(T) +0.062、KL 0.188(50.7s)
    DPO・stochastic・main・beta=0.8・seed=1: 損失 0.5470、a(T) 0.7467、h(T) +1.007、Delta_w(T) +0.023、KL 0.352(49.4s)
    DPO・stochastic・main・beta=0.565685・seed=1: 損失 0.5334、a(T) 0.7411、h(T) +1.384、Delta_w(T) -0.038、KL 0.766(49.7s)
    DPO・stochastic・main・beta=3.2・seed=2: 損失 0.5775、a(T) 0.7711、h(T) +0.364、Delta_w(T) +0.076、KL 0.038(50.4s)
    DPO・stochastic・main・beta=2.26274・seed=2: 損失 0.6716、a(T) 0.7728、h(T) +0.480、Delta_w(T) +0.084、KL 0.071(48.9s)
    DPO・stochastic・main・beta=1.6・seed=2: 損失 0.5778、a(T) 0.7689、h(T) +0.657、Delta_w(T) +0.088、KL 0.113(50.3s)
    DPO・stochastic・main・beta=1.13137・seed=2: 損失 0.5736、a(T) 0.7672、h(T) +0.797、Delta_w(T) +0.085、KL 0.192(49.7s)
    DPO・stochastic・main・beta=0.8・seed=2: 損失 0.6994、a(T) 0.7570、h(T) +1.085、Delta_w(T) +0.034、KL 0.370(50.5s)
    DPO・stochastic・main・beta=0.565685・seed=2: 損失 0.6326、a(T) 0.7550、h(T) +1.454、Delta_w(T) -0.080、KL 0.879(50.0s)
    DPO・stochastic・main・beta=3.2・seed=3: 損失 0.6354、a(T) 0.7672、h(T) +0.350、Delta_w(T) +0.066、KL 0.042(50.4s)
    DPO・stochastic・main・beta=2.26274・seed=3: 損失 0.6404、a(T) 0.7701、h(T) +0.471、Delta_w(T) +0.082、KL 0.072(48.9s)
    DPO・stochastic・main・beta=1.6・seed=3: 損失 0.6030、a(T) 0.7653、h(T) +0.639、Delta_w(T) +0.084、KL 0.111(50.2s)
    DPO・stochastic・main・beta=1.13137・seed=3: 損失 0.5894、a(T) 0.7670、h(T) +0.864、Delta_w(T) +0.074、KL 0.219(48.9s)
    DPO・stochastic・main・beta=0.8・seed=3: 損失 0.6678、a(T) 0.7533、h(T) +1.096、Delta_w(T) +0.002、KL 0.383(50.0s)
    DPO・stochastic・main・beta=0.565685・seed=3: 損失 0.8665、a(T) 0.7433、h(T) +1.472、Delta_w(T) -0.163、KL 0.809(49.7s)
    DPO・stochastic・main・beta=3.2・seed=4: 損失 0.7666、a(T) 0.7584、h(T) +0.340、Delta_w(T) +0.065、KL 0.046(50.1s)
    DPO・stochastic・main・beta=2.26274・seed=4: 損失 0.7426、a(T) 0.7621、h(T) +0.434、Delta_w(T) +0.075、KL 0.076(48.6s)
    DPO・stochastic・main・beta=1.6・seed=4: 損失 0.6633、a(T) 0.7611、h(T) +0.578、Delta_w(T) +0.088、KL 0.108(50.9s)
    DPO・stochastic・main・beta=1.13137・seed=4: 損失 0.6478、a(T) 0.7555、h(T) +0.738、Delta_w(T) +0.076、KL 0.171(50.8s)
    DPO・stochastic・main・beta=0.8・seed=4: 損失 0.6358、a(T) 0.7555、h(T) +1.078、Delta_w(T) +0.055、KL 0.407(50.7s)
    DPO・stochastic・main・beta=0.565685・seed=4: 損失 0.6535、a(T) 0.7452、h(T) +1.374、Delta_w(T) -0.044、KL 0.893(49.6s)
    --- 崩壊の領域の観察(判定なし、シード 0 のみ)---
    DPO・stochastic・main・beta=0.282843・seed=0: 損失 0.4776、a(T) 0.7419、h(T) +2.425、Delta_w(T) -0.987、KL 14.304(51.1s)
    DPO・stochastic・main・beta=0.141421・seed=0: 損失 0.5205、a(T) 0.6775、h(T) +3.786、Delta_w(T) -4.875、KL 40.311(52.6s)
    --- 実験 B(と C)---
    DPO・deterministic・main・beta=0.565685・seed=0: 損失 0.2698、a(T) 0.9580、h(T) +4.374、Delta_w(T) -1.480、KL 16.968(51.6s)
    IPO・deterministic・main・beta=0.565685・seed=0: 損失 0.3555、a(T) 0.9575、h(T) +0.611、Delta_w(T) +0.168、KL 0.080(49.2s)
    IPO・stochastic・main・beta=0.565685・seed=0: 損失 0.4529、a(T) 0.7573、h(T) +0.253、Delta_w(T) +0.059、KL 0.040(48.6s)
    DPO・deterministic・main・beta=0.565685・seed=1: 損失 0.2889、a(T) 0.9595、h(T) +4.307、Delta_w(T) -0.669、KL 9.514(51.1s)
    IPO・deterministic・main・beta=0.565685・seed=1: 損失 0.2394、a(T) 0.9521、h(T) +0.586、Delta_w(T) +0.158、KL 0.076(49.8s)
    IPO・stochastic・main・beta=0.565685・seed=1: 損失 0.5598、a(T) 0.7582、h(T) +0.245、Delta_w(T) +0.055、KL 0.026(48.6s)
    DPO・deterministic・main・beta=0.565685・seed=2: 損失 0.2549、a(T) 0.9595、h(T) +4.411、Delta_w(T) -1.379、KL 17.661(51.4s)
    IPO・deterministic・main・beta=0.565685・seed=2: 損失 0.2311、a(T) 0.9590、h(T) +0.604、Delta_w(T) +0.171、KL 0.067(49.5s)
    IPO・stochastic・main・beta=0.565685・seed=2: 損失 0.6328、a(T) 0.7655、h(T) +0.262、Delta_w(T) +0.067、KL 0.035(48.5s)
    DPO・deterministic・main・beta=0.565685・seed=3: 損失 0.3304、a(T) 0.9360、h(T) +3.964、Delta_w(T) -1.897、KL 18.129(51.5s)
    IPO・deterministic・main・beta=0.565685・seed=3: 損失 0.2990、a(T) 0.9487、h(T) +0.584、Delta_w(T) +0.166、KL 0.076(48.6s)
    IPO・stochastic・main・beta=0.565685・seed=3: 損失 0.6714、a(T) 0.7621、h(T) +0.260、Delta_w(T) +0.060、KL 0.033(48.4s)
    DPO・deterministic・main・beta=0.565685・seed=4: 損失 0.3011、a(T) 0.9580、h(T) +4.406、Delta_w(T) -1.654、KL 24.142(52.0s)
    IPO・deterministic・main・beta=0.565685・seed=4: 損失 0.2593、a(T) 0.9487、h(T) +0.561、Delta_w(T) +0.152、KL 0.066(49.9s)
    IPO・stochastic・main・beta=0.565685・seed=4: 損失 0.7536、a(T) 0.7596、h(T) +0.237、Delta_w(T) +0.058、KL 0.026(49.0s)
    --- 実験 D ---
    DPO・stochastic・near・beta=0.565685・seed=0: 損失 0.5412、a(T) 0.7343、h(T) +0.908、Delta_w(T) -0.019、KL 0.834(44.0s)
    DPO・stochastic・far・beta=0.565685・seed=0: 損失 0.4314、a(T) 0.7574、h(T) +1.900、Delta_w(T) -0.027、KL 0.399(50.8s)
    DPO・stochastic・near・beta=0.565685・seed=1: 損失 0.5697、a(T) 0.7466、h(T) +0.941、Delta_w(T) -0.074、KL 1.203(44.7s)
    DPO・stochastic・far・beta=0.565685・seed=1: 損失 0.4100、a(T) 0.7686、h(T) +1.941、Delta_w(T) -0.042、KL 0.417(51.7s)
    DPO・stochastic・near・beta=0.565685・seed=2: 損失 0.6258、a(T) 0.7347、h(T) +0.850、Delta_w(T) -0.093、KL 0.818(44.6s)
    DPO・stochastic・far・beta=0.565685・seed=2: 損失 0.4637、a(T) 0.7682、h(T) +2.100、Delta_w(T) -0.032、KL 0.527(51.9s)
    DPO・stochastic・near・beta=0.565685・seed=3: 損失 0.6006、a(T) 0.7384、h(T) +0.820、Delta_w(T) -0.066、KL 0.737(44.0s)
    DPO・stochastic・far・beta=0.565685・seed=3: 損失 0.4483、a(T) 0.7665、h(T) +1.999、Delta_w(T) -0.039、KL 0.461(51.0s)
    DPO・stochastic・near・beta=0.565685・seed=4: 損失 0.5666、a(T) 0.7420、h(T) +0.928、Delta_w(T) -0.069、KL 0.904(44.9s)
    DPO・stochastic・far・beta=0.565685・seed=4: 損失 0.3490、a(T) 0.7700、h(T) +1.961、Delta_w(T) -0.037、KL 0.475(52.0s)
    
    学習と評価の合計: 57 学習、2827.1s(参照方策の対数確率のキャッシュ: 計算 25 回・再利用 34 回)
    学習の成立(A): a(T) >= 0.55 -> 成立
    学習の成立(B・C): a(T) >= 0.55 -> 成立
    学習の成立(D): a(T) >= 0.55 -> 成立


### 6.9 実験 A: 直接選好最適化の過最適化

評価用の入力を復元抽出するクラスタブートストラップの再標本(抽出回数の行列)は、全水準・全シードで共通である。


```python
def ols_slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # 最後の軸について y を x に最小二乗で回帰した傾き(y は (..., L))
    x = np.asarray(x, dtype=np.float64)
    xc = x - x.mean()
    return (np.asarray(y) - np.asarray(y).mean(axis=-1, keepdims=True)) @ xc / (xc @ xc)


def three_way_verdict(value: float, sigma: float) -> str:
    # value > 2 sigma: 支持、value < -2 sigma: 反証、それ以外: 判定不能(value は期待する向きが正になるよう渡す)
    if not (math.isfinite(value) and math.isfinite(sigma)):
        return "判定不能"
    if value > 2 * sigma:
        return "支持"
    if value < -2 * sigma:
        return "反証"
    return "判定不能"


_tag = "[スモークテスト・結論ではない] " if SMOKE_TEST else ""
EVAL_BOOTSTRAP_COUNTS = (
    np.random.default_rng(BOOTSTRAP_SEED)
    .multinomial(
        NUM_EVAL_INPUTS, np.full(NUM_EVAL_INPUTS, 1.0 / NUM_EVAL_INPUTS), size=BOOTSTRAP_RESAMPLES
    )
    .astype(np.float64)
)  # (B, |X|)
_per_input = SAMPLES_PER_PROMPT * TASK_COUNT
A_U = np.log2(1.0 / np.array(A_BETAS))  # u_j(昇順)
A_TRUE_BY_INPUT = np.array(
    [[RECORDS[key_a(b, s)]["eval"]["true_reward_by_input"] for b in A_BETAS] for s in SEEDS_A]
)  # (S, L, |X|)
A_KL = np.array([[RECORDS[key_a(b, s)]["eval"]["kl"] for b in A_BETAS] for s in SEEDS_A])  # (S, L)
A_R = A_TRUE_BY_INPUT.sum(axis=2) / (_per_input * NUM_EVAL_INPUTS)  # (S, L) = R_s(beta_j)
A_R_MEAN = A_R.mean(axis=0)
_boot = np.einsum("bx,slx->bl", EVAL_BOOTSTRAP_COUNTS, A_TRUE_BY_INPUT) / (
    _per_input * EVAL_BOOTSTRAP_COUNTS.sum(axis=1)[:, None] * len(SEEDS_A)
)  # (B, L): 再標本ごとのシード平均
_first, _second = slice(0, NUM_A_LEVELS // 2), slice(NUM_A_LEVELS // 2, None)
CONTRAST_A = {}
for _name, _part in (("g1", _first), ("g2", _second)):
    _seed_slopes = ols_slope(A_U[_part], A_R[:, _part])
    _var_boot = float(ols_slope(A_U[_part], _boot[:, _part]).var(ddof=1))
    CONTRAST_A[_name] = {
        "value": float(ols_slope(A_U[_part], A_R_MEAN[_part])),
        "seed_slopes": _seed_slopes.tolist(),
        "var_boot": _var_boot,
        "sigma": math.sqrt(float(np.std(_seed_slopes, ddof=1)) ** 2 / len(SEEDS_A) + _var_boot),
    }
_g1, _g2 = CONTRAST_A["g1"], CONTRAST_A["g2"]
if _g1["value"] > 2 * _g1["sigma"] and _g2["value"] < -2 * _g2["sigma"]:
    verdict_A = "支持"
elif _g2["value"] > 2 * _g2["sigma"]:
    verdict_A = "反証"
else:
    verdict_A = "判定不能"
A_KL_MEAN = A_KL.mean(axis=0)
precondition_status["PA"] = bool(
    A_KL_MEAN[-1] > 0 and A_KL_MEAN[-1] >= KL_RATIO_MIN * max(A_KL_MEAN[0], 0.0)
)
print(
    f"水準 beta_j: {[round(b, 6) for b in A_BETAS]}、u_j = log2(1/beta_j): {A_U.tolist()}、S_A = {len(SEEDS_A)}"
)
print("真の報酬の平均 R(beta_j)(シード平均): " + ", ".join(f"{v:.4f}" for v in A_R_MEAN))
for _name, _label in (("g1", "前半"), ("g2", "後半")):
    _c = CONTRAST_A[_name]
    print(
        f"{_name}({_label}の傾き、u あたり)= {_c['value']:+.5f}、シードごと "
        + ", ".join(f"{v:+.5f}" for v in _c["seed_slopes"])
        + f"、Var_boot = {_c['var_boot']:.3e}、sigma = {_c['sigma']:.5f}、2 sigma = {2 * _c['sigma']:.5f}"
    )
print(
    f"PA: KL(beta_6) = {A_KL_MEAN[-1]:.4f}、KL(beta_1) = {A_KL_MEAN[0]:.4f}、比の下限 {KL_RATIO_MIN} -> "
    f"{'成立' if precondition_status['PA'] else '不成立'}。学習の成立(A): {precondition_status['学習の成立(A)']}"
)
print(f"{_tag}判定関数の結果: {verdict_A}")
_exact = np.array(
    [
        [
            RECORDS[key_a(b, s)]["eval"]["exact_by_input"].sum() / (_per_input * NUM_EVAL_INPUTS)
            for b in A_BETAS
        ]
        for s in SEEDS_A
    ]
)
_length = np.array(
    [
        [
            RECORDS[key_a(b, s)]["eval"]["length_by_input"].sum() / (_per_input * NUM_EVAL_INPUTS)
            for b in A_BETAS
        ]
        for s in SEEDS_A
    ]
)
_eval_pair = [
    [
        pair_statistics(RECORDS[key_a(b, s)]["eval_pairs"], EVAL_PAIR_DATASET["p_hat"])
        for b in A_BETAS
    ]
    for s in SEEDS_A
]
print(
    "診断量(シード平均): "
    + dumps_compact_json(
        {
            "KL": A_KL_MEAN.round(4).tolist(),
            "exact_match_rate": _exact.mean(axis=0).round(4).tolist(),
            "response_tokens": _length.mean(axis=0).round(3).tolist(),
            "eval_pair_margin": np.mean([[p["margin"] for p in row] for row in _eval_pair], axis=0)
            .round(3)
            .tolist(),
            "eval_pair_true_order_agreement": np.mean(
                [[p["accuracy"] for p in row] for row in _eval_pair], axis=0
            )
            .round(4)
            .tolist(),
        }
    )
)

_fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
for _s in range(len(SEEDS_A)):
    _axes[0].plot(A_U, A_R[_s], color="C0", alpha=0.3)
    _axes[1].plot(A_KL[_s], A_R[_s], color="C1", alpha=0.3, marker=".")
_axes[0].plot(A_U, A_R_MEAN, color="C0", marker="o", label="true reward (seed mean)")
_axes[0].axvline(
    (A_U[NUM_A_LEVELS // 2 - 1] + A_U[NUM_A_LEVELS // 2]) / 2,
    color="gray",
    linestyle=":",
    label="first / second half",
)
_axes[0].set_xlabel("log2(1 / beta)")
_axes[0].set_ylabel("E[r*] under the DPO policy")
_axes[0].legend()
_axes[1].plot(A_KL_MEAN, A_R_MEAN, color="C1", marker="o")
_axes[1].set_xlabel("KL estimate (nats)")
_axes[1].set_ylabel("E[r*] under the DPO policy")
_axes[1].set_title("true reward vs KL (each point = one beta)")
plt.tight_layout()
plt.show()
```

    水準 beta_j: [3.2, 2.262742, 1.6, 1.131371, 0.8, 0.565685]、u_j = log2(1/beta_j): [-1.6780719051126376, -1.178071905112638, -0.6780719051126377, -0.17807190511263796, 0.32192809488736235, 0.8219280948873621]、S_A = 5
    真の報酬の平均 R(beta_j)(シード平均): 0.7112, 0.7125, 0.7160, 0.7141, 0.7083, 0.6993
    g1(前半の傾き、u あたり)= +0.00474、シードごと +0.00237, +0.00177, +0.00863, +0.00791, +0.00304、Var_boot = 1.180e-06、sigma = 0.00182、2 sigma = 0.00364
    g2(後半の傾き、u あたり)= -0.01482、シードごと -0.01366, -0.00999, -0.02262, -0.01108, -0.01676、Var_boot = 1.627e-06、sigma = 0.00261、2 sigma = 0.00521
    PA: KL(beta_6) = 0.8077、KL(beta_1) = 0.0435、比の下限 4.0 -> 成立。学習の成立(A): True
    判定関数の結果: 支持
    診断量(シード平均): {
      "KL": [0.0435, 0.074, 0.1144, 0.1896, 0.3773, 0.8077],
      "exact_match_rate": [0.0195, 0.0198, 0.0215, 0.0203, 0.0193, 0.0187],
      "response_tokens": [13.076, 13.058, 13.068, 13.102, 13.247, 13.519],
      "eval_pair_margin": [0.054, 0.083, 0.091, 0.144, 0.237, 0.366],
      "eval_pair_true_order_agreement": [0.5389, 0.5544, 0.5401, 0.5448, 0.5433, 0.5425]
    }



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/018_direct_preference_optimization/output_41_1.png)
    


### 6.10 崩壊の領域の観察(判定なし)

**判定基準を設けない観察であり、実験 A の $g_2$ の計算に含めない。** $\beta_6$(実験 A の最小の水準)から較正の格子を 2 点・4 点下った
$\beta$(格子の下端を下回る場合は下端。同じ点は 1 つにまとめ、$\beta_6$ と一致する場合は実験 A の学習を共有する)で、DPO・確率的なラベル・シード 0 のみを学習し、実験 A と同じ評価をした(6.8 節)。
KL ダイバージェンスが急増する領域で、真の報酬・応答の長さ・終端記号で止まる割合がどう変わるかを記録する。


```python
_rows = []
for _beta, _kind in [(b, "A") for b in A_BETAS] + [(b, "collapse") for b in COLLAPSE_BETAS]:
    _e = RECORDS[key_a(_beta, 0)]["eval"]
    _total = _e["count_per_input"] * NUM_EVAL_INPUTS
    _rows.append(
        {
            "beta": round(_beta, 6),
            "kind": _kind,
            "same_run_as_A": _kind == "collapse"
            and math.isclose(_beta, A_BETAS[-1]),  # beta_6 自体との一致
            "true_reward": round(float(_e["true_reward_by_input"].sum() / _total), 4),
            "KL": round(_e["kl"], 3),
            "mean_tokens": round(_e["mean_tokens"], 2),
            "stop_rate": round(_e["stop_rate"], 4),
            "exact_match_rate": round(float(_e["exact_by_input"].sum() / _total), 4),
        }
    )
print("シード 0 の実験 A の水準と、崩壊の領域の観察(判定なし): " + dumps_compact_json(_rows))
_fig, _ax = plt.subplots(figsize=(7, 4))
_ax.plot(A_KL_MEAN, A_R_MEAN, color="C1", marker="o", label="experiment A (seed mean)")
_collapse = [r for r in _rows if r["kind"] == "collapse"]
_ax.scatter(
    [r["KL"] for r in _collapse],
    [r["true_reward"] for r in _collapse],
    color="C3",
    marker="x",
    s=80,
    label="collapse region (seed 0, no verdict)",
)
_ax.set_xscale("symlog", linthresh=1.0)
_ax.set_xlabel("KL estimate (nats, symlog)")
_ax.set_ylabel("E[r*] under the DPO policy")
_ax.legend()
plt.tight_layout()
plt.show()
```

    シード 0 の実験 A の水準と、崩壊の領域の観察(判定なし): [
      {
        "beta": 3.2,
        "kind": "A",
        "same_run_as_A": false,
        "true_reward": 0.7125,
        "KL": 0.046,
        "mean_tokens": 13.0,
        "stop_rate": 0.9858,
        "exact_match_rate": 0.0195
      },
      {
        "beta": 2.262742,
        "kind": "A",
        "same_run_as_A": false,
        "true_reward": 0.7124,
        "KL": 0.09,
        "mean_tokens": 13.08,
        "stop_rate": 0.9812,
        "exact_match_rate": 0.0198
      },
      {
        "beta": 1.6,
        "kind": "A",
        "same_run_as_A": false,
        "true_reward": 0.7149,
        "KL": 0.136,
        "mean_tokens": 13.06,
        "stop_rate": 0.9839,
        "exact_match_rate": 0.0208
      },
      {
        "beta": 1.131371,
        "kind": "A",
        "same_run_as_A": false,
        "true_reward": 0.715,
        "KL": 0.178,
        "mean_tokens": 13.04,
        "stop_rate": 0.9846,
        "exact_match_rate": 0.0176
      },
      {
        "beta": 0.8,
        "kind": "A",
        "same_run_as_A": false,
        "true_reward": 0.7093,
        "KL": 0.374,
        "mean_tokens": 13.2,
        "stop_rate": 0.9766,
        "exact_match_rate": 0.0161
      },
      {
        "beta": 0.565685,
        "kind": "A",
        "same_run_as_A": false,
        "true_reward": 0.7013,
        "KL": 0.692,
        "mean_tokens": 13.46,
        "stop_rate": 0.9648,
        "exact_match_rate": 0.0178
      },
      {
        "beta": 0.282843,
        "kind": "collapse",
        "same_run_as_A": false,
        "true_reward": 0.4925,
        "KL": 14.304,
        "mean_tokens": 21.33,
        "stop_rate": 0.582,
        "exact_match_rate": 0.0103
      },
      {
        "beta": 0.141421,
        "kind": "collapse",
        "same_run_as_A": false,
        "true_reward": 0.2846,
        "KL": 40.311,
        "mean_tokens": 26.94,
        "stop_rate": 0.2715,
        "exact_match_rate": 0.0037
      }
    ]



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/018_direct_preference_optimization/output_43_1.png)
    


### 6.11 実験 B: 決定的なラベルでの過適合(DPO と IPO)

学習用の組を復元抽出するブートストラップの再標本(抽出回数の行列)は、実験 B・C で共通である。


```python
PAIR_BOOTSTRAP_COUNTS = (
    np.random.default_rng(BOOTSTRAP_SEED + 1)
    .multinomial(
        NUM_MAIN_PAIRS, np.full(NUM_MAIN_PAIRS, 1.0 / NUM_MAIN_PAIRS), size=BOOTSTRAP_RESAMPLES
    )
    .astype(np.float32)
)  # (B, N)


def b_contribution(condition: int, seed: int, step: int, quantity: str) -> np.ndarray:
    loss_type, label_kind = B_CONDITIONS[condition]
    record = RECORDS[key_b(condition, seed)]
    p_hat = dataset_for("main", label_kind, seed)["p_hat"]
    return pair_contributions(record["history"]["evaluations"][step], p_hat)[quantity]


# g_{c,s} の組ごとの寄与 (2 p_hat - 1)(h_1(T) - h_1(T/2)) を、条件・シードごとに並べる: (4, S, N)
_growth = np.array(
    [
        [
            b_contribution(c, s, TRAIN_STEPS, "margin") - b_contribution(c, s, HALF_STEP, "margin")
            for s in SEEDS_B
        ]
        for c in B_CONDITIONS
    ]
)
B_GROWTH = _growth.mean(axis=2)  # (4, S) = g_{c,s}
B_D_SEEDS = (B_GROWTH[0] - B_GROWTH[1]) - (B_GROWTH[2] - B_GROWTH[3])  # D_s
_d_pairs = ((_growth[0] - _growth[1]) - (_growth[2] - _growth[3])).mean(
    axis=0
)  # 組ごとの D のシード平均への寄与 (N,)
_boot_d = (PAIR_BOOTSTRAP_COUNTS @ _d_pairs.astype(np.float32)) / NUM_MAIN_PAIRS
CONTRAST_B = {
    "value": float(B_D_SEEDS.mean()),
    "seed_values": B_D_SEEDS.tolist(),
    "var_boot": float(_boot_d.astype(np.float64).var(ddof=1)),
}
CONTRAST_B["sigma"] = math.sqrt(
    float(np.std(B_D_SEEDS, ddof=1)) ** 2 / len(SEEDS_B) + CONTRAST_B["var_boot"]
)
verdict_B = three_way_verdict(CONTRAST_B["value"], CONTRAST_B["sigma"])
B_MIXED = {
    s: float(
        np.mean(
            (dataset_for("main", "stochastic", s)["p_hat"] > 0)
            & (dataset_for("main", "stochastic", s)["p_hat"] < 1)
        )
    )
    for s in SEEDS_B
}
precondition_status["PB"] = bool(min(B_MIXED.values()) >= MIXED_FRACTION_MIN)
_h_bar = {
    (c, step): np.array([b_contribution(c, s, step, "margin").mean() for s in SEEDS_B])
    for c in B_CONDITIONS
    for step in (HALF_STEP, TRAIN_STEPS)
}
print(f"S_B = {len(SEEDS_B)}、T/2 = {HALF_STEP}、T = {TRAIN_STEPS}、beta = tau = {STANDARD_BETA}")
for _c, (_loss, _kind) in B_CONDITIONS.items():
    print(
        f"条件 {_c}({_loss.upper()}・{_kind}): h(T/2) = {_h_bar[(_c, HALF_STEP)].mean():+.4f}、h(T) = "
        f"{_h_bar[(_c, TRAIN_STEPS)].mean():+.4f}、伸び g(シードごと)"
        + ", ".join(f"{v:+.4f}" for v in B_GROWTH[_c - 1])
    )
print(
    f"D = (g1 - g2) - (g3 - g4) = {CONTRAST_B['value']:+.4f}、シードごと "
    + ", ".join(f"{v:+.4f}" for v in B_D_SEEDS)
    + f"、Var_boot = {CONTRAST_B['var_boot']:.3e}、sigma = {CONTRAST_B['sigma']:.4f}、2 sigma = {2 * CONTRAST_B['sigma']:.4f}"
)
print(
    "PB: p_hat が (0, 1) の組の割合(シード順)"
    + ", ".join(f"{v:.4f}" for v in B_MIXED.values())
    + f"(下限 {MIXED_FRACTION_MIN}、期待値 {EXPECTED_MIXED_FRACTION:.4f})-> {'成立' if precondition_status['PB'] else '不成立'}。"
    f"学習の成立(B・C): {precondition_status['学習の成立(B・C)']}"
)
print(f"{_tag}判定関数の結果: {verdict_B}")
print(
    "診断量(シード平均): "
    + dumps_compact_json(
        {
            f"condition_{c}": {
                "KL": round(
                    float(np.mean([RECORDS[key_b(c, s)]["eval"]["kl"] for s in SEEDS_B])), 4
                ),
                "eval_pair_margin": round(
                    float(
                        np.mean(
                            [
                                pair_statistics(
                                    RECORDS[key_b(c, s)]["eval_pairs"], EVAL_PAIR_DATASET["p_hat"]
                                )["margin"]
                                for s in SEEDS_B
                            ]
                        )
                    ),
                    4,
                ),
                "final_accuracy": round(
                    float(np.mean([RECORDS[key_b(c, s)]["final"]["accuracy"] for s in SEEDS_B])), 4
                ),
            }
            for c in B_CONDITIONS
        }
    )
)
_fig, _ax = plt.subplots(figsize=(7, 4))
for _c, (_loss, _kind) in B_CONDITIONS.items():
    _curves = np.array([RECORDS[key_b(_c, s)]["history"]["mean_margin"] for s in SEEDS_B])
    _window = max(1, TRAIN_STEPS // 32)
    _smooth = np.convolve(_curves.mean(axis=0), np.ones(_window) / _window, mode="valid")
    _ax.plot(np.arange(len(_smooth)) + _window, _smooth, label=f"{_c}: {_loss.upper()}, {_kind}")
_ax.axvline(HALF_STEP, color="gray", linestyle=":")
_ax.set_xlabel("step")
_ax.set_ylabel("batch mean margin h (nats, moving average)")
_ax.legend()
plt.tight_layout()
plt.show()
```

    S_B = 5、T/2 = 384、T = 768、beta = tau = 0.5656854249492381
    条件 1(DPO・deterministic): h(T/2) = +2.5221、h(T) = +4.2924、伸び g(シードごと)+1.8015, +1.8189, +1.8496, +1.5719, +1.8099
    条件 2(DPO・stochastic): h(T/2) = +0.9797、h(T) = +1.4107、伸び g(シードごと)+0.4989, +0.4053, +0.5144, +0.4547, +0.2812
    条件 3(IPO・deterministic): h(T/2) = +0.4210、h(T) = +0.5890、伸び g(シードごと)+0.1692, +0.1752, +0.1897, +0.1769, +0.1290
    条件 4(IPO・stochastic): h(T/2) = +0.1815、h(T) = +0.2513、伸び g(シードごと)+0.0723, +0.0700, +0.0833, +0.0571, +0.0664
    D = (g1 - g2) - (g3 - g4) = +1.2413、シードごと +1.2057, +1.3084, +1.2288, +0.9973, +1.4661、Var_boot = 1.084e-03、sigma = 0.0830、2 sigma = 0.1659
    PB: p_hat が (0, 1) の組の割合(シード順)0.5771, 0.5801, 0.5630, 0.5679, 0.5845(下限 0.4、期待値 0.5709)-> 成立。学習の成立(B・C): True
    判定関数の結果: 支持
    診断量(シード平均): {
      "condition_1": {"KL": 17.2828, "eval_pair_margin": 0.808, "final_accuracy": 0.9542},
      "condition_2": {"KL": 0.8077, "eval_pair_margin": 0.3656, "final_accuracy": 0.747},
      "condition_3": {"KL": 0.0731, "eval_pair_margin": 0.0729, "final_accuracy": 0.9532},
      "condition_4": {"KL": 0.032, "eval_pair_margin": 0.0375, "final_accuracy": 0.7605}
    }



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/018_direct_preference_optimization/output_45_1.png)
    




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/04_alignment/018_direct_preference_optimization.ipynb
