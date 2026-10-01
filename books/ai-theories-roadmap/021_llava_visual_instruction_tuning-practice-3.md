---
title: "LLaVA 型 Vision-Language 連結と視覚指示チューニング / LLaVA-style Vision-Language Connection and Visual Instruction Tuning(実装・実験編 3/3)"
---

この記事は後編(実装・実験編 3/3)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/021_llava_visual_instruction_tuning-practice-2)。

### 6.4 実行計画の選択

6.3 節の計測から、各計画の残りの実行時間を見積もる。

- 学習 1 回: 第 1 段階 $=$ 1 ステップの時間 $\times T_1$ + 検証集合のキャプションの評価 2 回(前後)。第 2 段階 $=$ 1 ステップの時間
  $\times$ ステップ数 + 検証集合の質問の評価 5 回(開始時・途中 3 回・終わり)。本番の学習は、加えて生成による評価 2 回(通常・画像の
  差し替え)。
- 較正: 起点の第 1 段階(視覚トークンの取り方ごとに 1 回)と、較正する条件ごとに、格子の 3 点に拡張の 1 点を加えた **4 点**
  (拡張が起きる最悪の場合)の第 2 段階の学習。
- 線形プローブ 4 回と、判定・図の作成の固定の余裕 60 秒。

これに、ノートブックの開始からの経過時間を足して予算(120 分)と比べ、収まる計画のうち番号の最も小さいものを選ぶ。計画 7 でも
超える場合は、学習の前に停止する。


```python
REPORTING_MARGIN_SECONDS = 60.0  # 判定・ブートストラップ・図の作成の固定の余裕
CALIBRATION_POINTS_WORST = len(LR_GRID_MULTIPLIERS) + 1  # 格子 3 点 + 拡張 1 点


def stage1_seconds(pooling: str) -> float:
    return STEP_SECONDS[("stage1", pooling)] * STAGE1_STEPS + 2 * ESTIMATE_NLL[(pooling, "caption")]


def stage2_seconds(pooling: str, steps: int) -> float:
    return STEP_SECONDS[("stage2", pooling)] * steps + (2 + len(INTERMEDIATE_EVAL_FRACTIONS)) * ESTIMATE_NLL[(pooling, "question")]


def calibration_targets(mode: str) -> list[tuple[str, str]]:
    # 較正する (段階, 条件)。"all" は第 2 段階の P・R・M、"representative" は第 2 段階の P のみ(6.1 節)
    if mode == "all":
        return [("stage2", "P"), ("stage2", "R"), ("stage2", "M")]
    return [("stage2", "P")]


def stage1_poolings_for_calibration(mode: str) -> list[str]:
    # 第 2 段階の較正の起点にする、較正のシードの第 1 段階の学習(視覚トークンの取り方ごとに 1 回)
    return sorted({CONDITIONS[c]["pooling"] for _, c in calibration_targets(mode) if CONDITIONS[c]["stage1"]})


def estimate_plan(plan: dict) -> dict:
    stage_cfg = STAGES[CURRENT_LEVEL_NAME][plan["stage"]]
    calibration = sum(stage1_seconds(p) for p in stage1_poolings_for_calibration(plan["calibration_mode"]))
    for _, condition in calibration_targets(plan["calibration_mode"]):
        calibration += CALIBRATION_POINTS_WORST * stage2_seconds(CONDITIONS[condition]["pooling"], STAGE2_STEPS)
    main = 0.0
    for condition, spec in CONDITIONS.items():
        pooling = spec["pooling"]
        one = (stage1_seconds(pooling) if spec["stage1"] else 0.0) + stage2_seconds(pooling, spec["stage2_steps"])
        main += seeds_for(condition, stage_cfg) * (one + 2 * ESTIMATE_GENERATION[pooling])
    total = calibration + main + ESTIMATE_PROBE + REPORTING_MARGIN_SECONDS
    return {"calibration": calibration, "main": main, "probe": ESTIMATE_PROBE, "total": total}


ELAPSED_AT_SELECTION = time.time() - NOTEBOOK_START_TIME
PLAN_ESTIMATES = {p["plan"]: estimate_plan(p) for p in PLANS}
print(f"経過時間 {ELAPSED_AT_SELECTION / 60:.1f} 分、予算 {SESSION_BUDGET_SECONDS / 60:.0f} 分")
for _p in PLANS:
    _e = PLAN_ESTIMATES[_p["plan"]]
    _fits = ELAPSED_AT_SELECTION + _e["total"] <= SESSION_BUDGET_SECONDS
    print(
        f"  計画 {_p['plan']}(段階 {_p['stage']}、較正 {_p['calibration_mode']!r}): 較正 {_e['calibration'] / 60:.1f} 分 + 本番 "
        f"{_e['main'] / 60:.1f} 分 + プローブ {_e['probe'] / 60:.1f} 分 + 余裕 -> 残り {_e['total'] / 60:.1f} 分、"
        f"終了の見込み {(ELAPSED_AT_SELECTION + _e['total']) / 60:.1f} 分({'収まる' if _fits else '超える'})"
    )
_feasible = [p for p in PLANS if ELAPSED_AT_SELECTION + PLAN_ESTIMATES[p["plan"]]["total"] <= SESSION_BUDGET_SECONDS]
AUTO_PLAN = _feasible[0] if _feasible else None
if FORCED_PLAN is not None:
    SELECTED_PLAN = PLANS[FORCED_PLAN]
    print(f"*** テスト専用の上書き: 計画 {FORCED_PLAN} を使う(見積もりによる選択は {AUTO_PLAN and AUTO_PLAN['plan']}) ***")
elif AUTO_PLAN is None:
    raise RuntimeError(
        "最も下位の計画(計画 7)でも見積もりが予算を超えるため、学習の前に停止する。結果の情報は何も得ていない。"
        "実行条件(判定基準・水準・前提条件以外)を直して再実行すること。"
    )
else:
    SELECTED_PLAN = AUTO_PLAN
STAGE = SELECTED_PLAN["stage"]
CALIBRATION_MODE = SELECTED_PLAN["calibration_mode"]
STAGE_CFG = STAGES[CURRENT_LEVEL_NAME][STAGE]
SEEDS_A, SEEDS_B = STAGE_CFG["SEEDS_A"], STAGE_CFG["SEEDS_B"]
NUM_SEEDS = {c: seeds_for(c, STAGE_CFG) for c in CONDITIONS}
print(
    f"選ばれた計画: {SELECTED_PLAN['plan']}(段階 {STAGE}、較正の方式 {CALIBRATION_MODE!r})、実験 A のシード数 {SEEDS_A}、"
    f"実験 B のシード数 {SEEDS_B}、条件ごとのシード数 {NUM_SEEDS}、見積もり {PLAN_ESTIMATES[SELECTED_PLAN['plan']]['total'] / 60:.1f} 分"
)
```

    経過時間 1.6 分、予算 120 分
      計画 0(段階 0、較正 'all'): 較正 13.8 分 + 本番 33.0 分 + プローブ 0.2 分 + 余裕 -> 残り 47.9 分、終了の見込み 49.5 分(収まる)
      計画 1(段階 1、較正 'all'): 較正 13.8 分 + 本番 23.4 分 + プローブ 0.2 分 + 余裕 -> 残り 38.3 分、終了の見込み 39.9 分(収まる)
      計画 2(段階 2、較正 'all'): 較正 13.8 分 + 本番 20.4 分 + プローブ 0.2 分 + 余裕 -> 残り 35.3 分、終了の見込み 36.9 分(収まる)
      計画 3(段階 3、較正 'all'): 較正 13.8 分 + 本番 14.0 分 + プローブ 0.2 分 + 余裕 -> 残り 29.0 分、終了の見込み 30.5 分(収まる)
      計画 4(段階 0、較正 'representative'): 較正 5.4 分 + 本番 33.0 分 + プローブ 0.2 分 + 余裕 -> 残り 39.5 分、終了の見込み 41.1 分(収まる)
      計画 5(段階 1、較正 'representative'): 較正 5.4 分 + 本番 23.4 分 + プローブ 0.2 分 + 余裕 -> 残り 29.9 分、終了の見込み 31.5 分(収まる)
      計画 6(段階 2、較正 'representative'): 較正 5.4 分 + 本番 20.4 分 + プローブ 0.2 分 + 余裕 -> 残り 26.9 分、終了の見込み 28.5 分(収まる)
      計画 7(段階 3、較正 'representative'): 較正 5.4 分 + 本番 14.0 分 + プローブ 0.2 分 + 余裕 -> 残り 20.6 分、終了の見込み 22.1 分(収まる)
    選ばれた計画: 0(段階 0、較正の方式 'all')、実験 A のシード数 5、実験 B のシード数 5、条件ごとのシード数 {'P': 5, 'R': 5, 'L': 5, 'M': 5}、見積もり 47.9 分


### 6.5 学習率の較正(第 2 段階)

6.1 節の規則のとおり。まず較正のシードで、固定の学習率の第 1 段階を視覚トークンの取り方ごとに 1 回学習し、P・M の第 2 段階の較正の
起点にする。較正の学習は評価集合を評価しない。


```python
_t0_calibration = time.time()
CALIBRATION: dict[tuple[str, str], dict] = {}

# 第 2 段階の較正の起点: 較正のシードで、固定の学習率の第 1 段階を視覚トークンの取り方ごとに 1 回学習する
STAGE1_CALIBRATION_STATE = {}
for _pooling in stage1_poolings_for_calibration(CALIBRATION_MODE):
    _model = build_llava(_pooling, CALIBRATION_SEED_INDEX)
    _h = train_stage1(_model, CALIBRATION_SEED_INDEX, STAGE1_LEARNING_RATE)
    STAGE1_CALIBRATION_STATE[_pooling] = {k: v.detach().cpu().clone() for k, v in _model.projection.state_dict().items()}
    print(f"{_tag}較正の起点の第 1 段階 {_pooling}: 最終の訓練損失 {np.mean(_h['loss'][-20:]):.4f}、検証集合のキャプションの負の対数尤度 {validation_nll(_model, 'caption'):.4f}")
    del _model
    empty_device_cache()


def calibrate(condition: str) -> dict:
    # 第 2 段階の学習率の較正(6.1 節): 格子の各点を本番と同じステップ数で学習し、検証集合の答えの負の対数尤度が最小のものを選ぶ
    spec = CONDITIONS[condition]
    base = STAGE1_CALIBRATION_STATE[spec["pooling"]] if spec["stage1"] else None
    nll, seconds = {}, {}

    def train_point(lr: float) -> None:
        record = run_condition(condition, CALIBRATION_SEED_INDEX, None, lr, "calibration", stage1_projection=base)
        assert "correct" not in record  # 較正では評価集合を評価しない
        finite = all(math.isfinite(x) for x in record["stage2_history"]["loss"])
        value = record["stage2_nll_end"]
        nll[lr] = value if (finite and math.isfinite(value)) else math.inf
        seconds[lr] = record["seconds"]

    grid = list(lr_grid_for("stage2", condition))
    for lr in grid:
        train_point(lr)
    best = min(sorted(nll), key=lambda lr: nll[lr])  # 同点なら小さい方
    extended = None
    if best in (min(grid), max(grid)):  # 端なら、その方向に公比 2 で 1 点だけ拡張する(1 回のみ)
        extended = best / LR_GRID_RATIO if best == min(grid) else best * LR_GRID_RATIO
        train_point(extended)
        best = min(sorted(nll), key=lambda lr: nll[lr])
    interior = min(nll) < best < max(nll)
    return {"grid": grid, "nll": nll, "chosen": best, "extended": extended, "interior": interior, "seconds": seconds}


for _stage, _condition in calibration_targets(CALIBRATION_MODE):
    CALIBRATION[(_stage, _condition)] = calibrate(_condition)
    _c = CALIBRATION[(_stage, _condition)]
    _extension = "なし" if _c["extended"] is None else f"{_c['extended']:.3g}"
    print(
        f"{_tag}較正 {_stage} {_condition}: 検証集合の負の対数尤度 "
        + "、".join(f"{lr:.3g}: {v:.4f}" for lr, v in sorted(_c["nll"].items()))
        + f" -> 選んだ学習率 {_c['chosen']:.3g}(拡張 {_extension}、内点 {_c['interior']})、{sum(_c['seconds'].values()):.0f}s"
    )


def learning_rates_for(condition: str) -> tuple[float | None, float]:
    # 条件の (第 1 段階, 第 2 段階) の学習率。第 1 段階は固定、第 2 段階で較正していない条件は規則で決める(6.1 節)
    stage2_source = {"P": "P", "R": "R", "L": "R", "M": "M"} if CALIBRATION_MODE == "all" else dict.fromkeys(CONDITIONS, "P")
    lr1 = STAGE1_LEARNING_RATE if CONDITIONS[condition]["stage1"] else None
    return lr1, CALIBRATION[("stage2", stage2_source[condition])]["chosen"]


LEARNING_RATES = {c: learning_rates_for(c) for c in CONDITIONS}
# P0(較正): 第 2 段階の較正のうち、各実験の条件の学習率を決めたもの(6.1 節)
P0_TARGETS = {"A": [("stage2", "P"), ("stage2", "R")], "B": [("stage2", "P"), ("stage2", "M")]}
for _exp, _targets in P0_TARGETS.items():
    _checked = [k for k in _targets if k in CALIBRATION]
    _excluded = [k for k in _targets if k not in CALIBRATION]
    precondition_status[f"P0({_exp})"] = all(CALIBRATION[k]["interior"] for k in _checked)
    print(
        f"P0(実験 {_exp}): 較正した {_checked} の内点 {[CALIBRATION[k]['interior'] for k in _checked]} -> "
        f"{precondition_status[f'P0({_exp})']}" + (f"。規則で決めたため対象外: {_excluded}" if _excluded else "")
    )
CALIBRATION_SECONDS = time.time() - _t0_calibration
print(f"条件ごとの学習率 (第 1 段階, 第 2 段階): {LEARNING_RATES}")
print(f"較正: {CALIBRATION_SECONDS / 60:.1f} 分")
```

    較正の起点の第 1 段階 mean: 最終の訓練損失 4.5988、検証集合のキャプションの負の対数尤度 4.5903
    較正の起点の第 1 段階 patch: 最終の訓練損失 2.5774、検証集合のキャプションの負の対数尤度 2.5867
    較正 stage2 P: 検証集合の負の対数尤度 0.004: 0.0971、0.008: 0.0793、0.016: 0.0935 -> 選んだ学習率 0.008(拡張 なし、内点 True)、238s
    較正 stage2 R: 検証集合の負の対数尤度 0.004: 0.0879、0.008: 0.0806、0.016: 0.1012 -> 選んだ学習率 0.008(拡張 なし、内点 True)、245s
    較正 stage2 M: 検証集合の負の対数尤度 0.004: 0.1006、0.008: 0.0912、0.016: 0.1012 -> 選んだ学習率 0.008(拡張 なし、内点 True)、124s
    P0(実験 A): 較正した [('stage2', 'P'), ('stage2', 'R')] の内点 [True, True] -> True
    P0(実験 B): 較正した [('stage2', 'P'), ('stage2', 'M')] の内点 [True, True] -> True
    条件ごとの学習率 (第 1 段階, 第 2 段階): {'P': (0.064, 0.008), 'R': (None, 0.008), 'L': (None, 0.008), 'M': (0.064, 0.008)}
    較正: 10.8 分


### 6.6 本番の学習と評価(実験 A・B)

シードの順に、各条件を学習する(条件ごとのシード数は 6.4 節で選ばれた段階で決まる)。


```python
_t0_training = time.time()
RECORDS: dict[tuple[str, int], dict] = {}
for _seed in range(max(NUM_SEEDS.values())):
    for _condition in CONDITIONS:
        if _seed < NUM_SEEDS[_condition]:
            _lr1, _lr2 = LEARNING_RATES[_condition]
            RECORDS[(_condition, _seed)] = run_condition(_condition, _seed, _lr1, _lr2, "main")
            print(_tag + describe(RECORDS[(_condition, _seed)]))
TRAINING_SECONDS = time.time() - _t0_training
print(f"本番の学習と評価: {TRAINING_SECONDS / 60:.1f} 分(見積もり {PLAN_ESTIMATES[SELECTED_PLAN['plan']]['main'] / 60:.1f} 分)")
```

    P s=0 lr1=0.064 lr2=0.008: 第 1 段階の検証 NLL 6.343 -> 2.574、 第 2 段階の検証 NLL 3.240 -> 0.0818、正解率 0.7448(色 0.8331・位置関係 0.6565)、差し替え 0.2014、119s
    R s=0 lr1=None lr2=0.008: 第 2 段階の検証 NLL 6.765 -> 0.0735、正解率 0.7940(色 0.8570・位置関係 0.7310)、差し替え 0.1996、89s
    L s=0 lr1=None lr2=0.008: 第 2 段階の検証 NLL 6.765 -> 0.0542、正解率 0.8627(色 0.8930・位置関係 0.8324)、差し替え 0.2000、118s
    M s=0 lr1=0.064 lr2=0.008: 第 1 段階の検証 NLL 6.146 -> 4.591、 第 2 段階の検証 NLL 4.799 -> 0.0918、正解率 0.6667(色 0.8092・位置関係 0.5242)、差し替え 0.2029、61s
    P s=1 lr1=0.064 lr2=0.008: 第 1 段階の検証 NLL 6.319 -> 2.573、 第 2 段階の検証 NLL 3.273 -> 0.0859、正解率 0.6776(色 0.8348・位置関係 0.5204)、差し替え 0.1998、120s
    R s=1 lr1=None lr2=0.008: 第 2 段階の検証 NLL 6.742 -> 0.0779、正解率 0.7235(色 0.8833・位置関係 0.5637)、差し替え 0.2033、89s
    L s=1 lr1=None lr2=0.008: 第 2 段階の検証 NLL 6.742 -> 0.0666、正解率 0.7741(色 0.9141・位置関係 0.6340)、差し替え 0.2017、119s
    M s=1 lr1=0.064 lr2=0.008: 第 1 段階の検証 NLL 6.153 -> 4.591、 第 2 段階の検証 NLL 4.799 -> 0.0900、正解率 0.6671(色 0.8061・位置関係 0.5280)、差し替え 0.2000、61s
    P s=2 lr1=0.064 lr2=0.008: 第 1 段階の検証 NLL 6.465 -> 2.591、 第 2 段階の検証 NLL 3.247 -> 0.0806、正解率 0.7699(色 0.7995・位置関係 0.7403)、差し替え 0.2031、119s
    R s=2 lr1=None lr2=0.008: 第 2 段階の検証 NLL 7.040 -> 0.0790、正解率 0.7232(色 0.8781・位置関係 0.5682)、差し替え 0.2055、89s
    L s=2 lr1=None lr2=0.008: 第 2 段階の検証 NLL 7.040 -> 0.0714、正解率 0.7415(色 0.9100・位置関係 0.5731)、差し替え 0.1998、119s
    M s=2 lr1=0.064 lr2=0.008: 第 1 段階の検証 NLL 6.153 -> 4.591、 第 2 段階の検証 NLL 4.802 -> 0.0912、正解率 0.6660(色 0.8106・位置関係 0.5215)、差し替え 0.1975、61s
    P s=3 lr1=0.064 lr2=0.008: 第 1 段階の検証 NLL 6.440 -> 2.576、 第 2 段階の検証 NLL 3.233 -> 0.0837、正解率 0.7336(色 0.8296・位置関係 0.6375)、差し替え 0.2041、119s
    R s=3 lr1=None lr2=0.008: 第 2 段階の検証 NLL 7.112 -> 0.0794、正解率 0.7377(色 0.8653・位置関係 0.6101)、差し替え 0.2015、90s
    L s=3 lr1=None lr2=0.008: 第 2 段階の検証 NLL 7.112 -> 0.0669、正解率 0.7812(色 0.9020・位置関係 0.6603)、差し替え 0.2003、118s
    M s=3 lr1=0.064 lr2=0.008: 第 1 段階の検証 NLL 6.158 -> 4.591、 第 2 段階の検証 NLL 4.803 -> 0.0950、正解率 0.6459(色 0.7832・位置関係 0.5087)、差し替え 0.2029、61s
    P s=4 lr1=0.064 lr2=0.008: 第 1 段階の検証 NLL 6.323 -> 2.559、 第 2 段階の検証 NLL 3.252 -> 0.0794、正解率 0.7071(色 0.8733・位置関係 0.5409)、差し替え 0.1988、120s
    R s=4 lr1=None lr2=0.008: 第 2 段階の検証 NLL 6.903 -> 0.0830、正解率 0.7126(色 0.8577・位置関係 0.5675)、差し替え 0.2033、89s
    L s=4 lr1=None lr2=0.008: 第 2 段階の検証 NLL 6.903 -> 0.0571、正解率 0.8570(色 0.8996・位置関係 0.8144)、差し替え 0.2038、119s
    M s=4 lr1=0.064 lr2=0.008: 第 1 段階の検証 NLL 6.154 -> 4.591、 第 2 段階の検証 NLL 4.797 -> 0.0913、正解率 0.6648(色 0.8096・位置関係 0.5201)、差し替え 0.2024、66s
    本番の学習と評価: 32.5 分(見積もり 33.0 分)


### 6.7 実験 A: 第 1 段階(特徴の整列)の効果


```python
def learning_preconditions(record: dict) -> dict:
    # P1(学習の成立)の各項目(6.1 節): 損失が有限、第 2 段階の検証集合の負の対数尤度が開始時の 0.5 倍以下、第 1 段階は開始時より小さい
    result = {
        "stage2_finite": all(math.isfinite(x) for x in record["stage2_history"]["loss"]),
        "stage2_nll": record["stage2_nll_end"] <= P1_NLL_RATIO * record["stage2_nll_start"],
    }
    if "stage1_history" in record:
        result["stage1_finite"] = all(math.isfinite(x) for x in record["stage1_history"]["loss"])
        result["stage1_nll"] = record["stage1_nll_after"] < record["stage1_nll_before"]  # 下がり幅は問わない(6.2 節の改訂)
    return result


def bootstrap_accuracies(correct_rows: list[np.ndarray], question_type: str | None = None) -> np.ndarray:
    # 画像をクラスタとして復元抽出する対応付きブートストラップ(全行に同じ再標本)。形状 (反復, 行)
    mask = np.ones(len(EVAL_ANSWERS)) if question_type is None else (QUESTION_TYPE_OF_EVAL == question_type).astype(np.float64)
    numerators = np.stack([row.astype(np.float64) * mask for row in correct_rows])
    return paired_cluster_bootstrap_ratio_of_sums(
        numerators, mask, EVAL_IMAGE_OF_QUESTION.tolist(), BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
    )


def combined_sigma(per_seed: np.ndarray, bootstrap_contrast: np.ndarray) -> dict:
    seed_variance = float(np.var(per_seed, ddof=1)) / len(per_seed)
    bootstrap_variance = float(np.var(bootstrap_contrast, ddof=1))
    return {
        "sigma": math.sqrt(seed_variance + bootstrap_variance),
        "seed_term": math.sqrt(seed_variance),
        "bootstrap_term": math.sqrt(bootstrap_variance),
    }


_seeds_a = list(range(SEEDS_A))
ACC = {c: np.array([RECORDS[(c, s)]["accuracy"]["all"] for s in range(NUM_SEEDS[c])]) for c in CONDITIONS if NUM_SEEDS[c] > 0}
A_PER_SEED = ACC["P"][:SEEDS_A] - ACC["R"]
_boot = bootstrap_accuracies([RECORDS[("P", s)]["correct"] for s in _seeds_a] + [RECORDS[("R", s)]["correct"] for s in _seeds_a])
_boot_contrast = _boot[:, :SEEDS_A].mean(axis=1) - _boot[:, SEEDS_A:].mean(axis=1)
DELTA_A = float(A_PER_SEED.mean())
SIGMA_A = combined_sigma(A_PER_SEED, _boot_contrast)

# 前提条件(6.1 節)
_p1 = {(c, s): learning_preconditions(RECORDS[(c, s)]) for c in ("P", "R") for s in _seeds_a}
precondition_status["P1(A)"] = all(all(v.values()) for v in _p1.values())
_swapped_a = {c: float(np.mean([RECORDS[(c, s)]["accuracy_swapped"]["all"] for s in _seeds_a])) for c in ("P", "R")}
precondition_status["P2(A)"] = all(v <= P2_SWAPPED_MAX for v in _swapped_a.values())
_base_a = float(ACC["R"].mean())
precondition_status["P3(A)"] = _base_a <= P3_ROOM_MAX
A_PRECONDITIONS = ["P0(A)", "P1(A)", "P2(A)", "P3(A)"]
A_COMPUTED = judge(DELTA_A, SIGMA_A["sigma"])
A_VERDICT = A_COMPUTED if all(precondition_status[k] for k in A_PRECONDITIONS) else "前提不成立"

print(f"{_tag}実験 A(シード {_seeds_a})")
print(f"  正解率 acc_P: {np.round(ACC['P'][:SEEDS_A], 4).tolist()}、acc_R: {np.round(ACC['R'], 4).tolist()}")
print(f"  d_s = acc_P - acc_R: {np.round(A_PER_SEED, 4).tolist()}")
print(
    f"  Delta_A = {DELTA_A:+.4f}、sigma_A = {SIGMA_A['sigma']:.4f}(シード間 {SIGMA_A['seed_term']:.4f}・ブートストラップ "
    f"{SIGMA_A['bootstrap_term']:.4f}、反復 {BOOTSTRAP_RESAMPLES:,})、閾値 2 sigma_A = {SIGMA_MULTIPLIER * SIGMA_A['sigma']:.4f}"
)
print(
    f"  前提条件: P0 {precondition_status['P0(A)']}、P1 {precondition_status['P1(A)']}"
    f"(不成立の学習 {[k for k, v in _p1.items() if not all(v.values())] or 'なし'})、"
    f"P2 {precondition_status['P2(A)']}(画像を差し替えた正解率 {json.dumps({k: round(v, 4) for k, v in _swapped_a.items()})} <= {P2_SWAPPED_MAX})、"
    f"P3 {precondition_status['P3(A)']}(acc_R の平均 {_base_a:.4f} <= {P3_ROOM_MAX})"
)
print(f"  判定関数の結果 {A_COMPUTED} -> {_tag}最終判定: {A_VERDICT}")

# 診断量(判定なし)
_fractions = (0.0, *INTERMEDIATE_EVAL_FRACTIONS, 1.0)


def nll_curve(record: dict) -> list[float]:
    evaluations = record["stage2_history"]["evaluations"]
    middle = [evaluations[s]["validation_nll"] for s in sorted(evaluations)]
    return [record["stage2_nll_start"], *middle, record["stage2_nll_end"]]


print("  診断量: 第 2 段階の検証集合の負の対数尤度(開始時・25%・50%・75%・終わり、シード平均)")
for _c in ("P", "R", "L", "M"):
    if NUM_SEEDS[_c] > 0:
        _curves = np.array([nll_curve(RECORDS[(_c, s)]) for s in range(NUM_SEEDS[_c])])
        print(f"    {_c}: {np.round(_curves.mean(axis=0), 4).tolist()}")
for _c in ("P", "R", "L", "M"):
    if NUM_SEEDS[_c] > 0:
        _recs = [RECORDS[(_c, s)] for s in range(NUM_SEEDS[_c])]
        _norms = {k: float(np.mean([r[k] for r in _recs])) for k in ("norm_init", "norm_after_stage1", "norm_after_stage2") if k in _recs[0]}
        print(
            f"    視覚トークンの平均ノルム {_c}: "
            + "、".join(f"{k} {v:.3f}(埋め込みの {v / TOKEN_EMBEDDING_NORM:.1f} 倍)" for k, v in _norms.items())
        )
if NUM_SEEDS["L"] > 0:
    print(f"    L(総ステップ数を P に揃えた第 2 段階のみ)の正解率: {np.round(ACC['L'], 4).tolist()}、平均 {ACC['L'].mean():.4f}")

_fig, _axes = plt.subplots(1, 2, figsize=(11, 3.6))
for _c, _marker in (("P", "o"), ("R", "s"), ("L", "^")):
    if NUM_SEEDS[_c] > 0:
        _axes[0].plot(range(NUM_SEEDS[_c]), ACC[_c], _marker, label=_c)
        _curves = np.array([nll_curve(RECORDS[(_c, s)]) for s in range(NUM_SEEDS[_c])])
        _x = np.array(_fractions) * CONDITIONS[_c]["stage2_steps"]
        _axes[1].plot(_x, _curves.mean(axis=0), marker=_marker, label=_c)
_axes[0].set_xlabel("seed")
_axes[0].set_ylabel("exact-match accuracy (eval)")
_axes[0].set_title(f"{_plot_tag}Experiment A: accuracy per seed")
_axes[0].legend()
_axes[1].set_xlabel("stage-2 step")
_axes[1].set_ylabel("validation NLL per answer token")
_axes[1].set_yscale("log")
_axes[1].set_title(f"{_plot_tag}stage-2 validation NLL (seed mean)")
_axes[1].legend()
plt.tight_layout()
plt.show()
```

    実験 A(シード [0, 1, 2, 3, 4])
      正解率 acc_P: [0.7448, 0.6776, 0.7699, 0.7336, 0.7071]、acc_R: [0.794, 0.7235, 0.7232, 0.7377, 0.7126]
      d_s = acc_P - acc_R: [-0.0492, -0.0459, 0.0467, -0.0042, -0.0055]
      Delta_A = -0.0116、sigma_A = 0.0177(シード間 0.0174・ブートストラップ 0.0030、反復 10,000)、閾値 2 sigma_A = 0.0354
      前提条件: P0 True、P1 True(不成立の学習 なし)、P2 True(画像を差し替えた正解率 {"P": 0.2014, "R": 0.2026} <= 0.5)、P3 True(acc_R の平均 0.7382 <= 0.95)
      判定関数の結果 判定不能 -> 最終判定: 判定不能
      診断量: 第 2 段階の検証集合の負の対数尤度(開始時・25%・50%・75%・終わり、シード平均)
        P: [3.249, 0.1443, 0.1134, 0.0917, 0.0823]
        R: [6.9123, 0.1441, 0.1106, 0.0875, 0.0786]
        L: [6.9123, 0.1351, 0.0969, 0.0738, 0.0633]
        M: [4.8001, 0.1549, 0.1233, 0.1001, 0.0919]
        視覚トークンの平均ノルム P: norm_init 25.607(埋め込みの 30.1 倍)、norm_after_stage1 251.349(埋め込みの 295.3 倍)、norm_after_stage2 280.049(埋め込みの 329.0 倍)
        視覚トークンの平均ノルム R: norm_init 25.607(埋め込みの 30.1 倍)、norm_after_stage2 89.063(埋め込みの 104.6 倍)
        視覚トークンの平均ノルム L: norm_init 25.607(埋め込みの 30.1 倍)、norm_after_stage2 102.015(埋め込みの 119.8 倍)
        視覚トークンの平均ノルム M: norm_init 23.823(埋め込みの 28.0 倍)、norm_after_stage1 418.990(埋め込みの 492.2 倍)、norm_after_stage2 389.150(埋め込みの 457.2 倍)
        L(総ステップ数を P に揃えた第 2 段階のみ)の正解率: [0.8627, 0.7741, 0.7415, 0.7812, 0.857]、平均 0.8033



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/021_llava_visual_instruction_tuning/output_39_1.png)
    


### 6.8 実験 B: 視覚トークンの取り方 × 質問の種類


```python
_seeds_b = list(range(SEEDS_B))
ACC_TYPE = {
    (c, t): np.array([RECORDS[(c, s)]["accuracy"][t] for s in _seeds_b]) for c in ("P", "M") for t in QUESTION_TYPES
}
B_COLOR = ACC_TYPE[("P", "color")] - ACC_TYPE[("M", "color")]  # c_s
B_SPATIAL = ACC_TYPE[("P", "spatial")] - ACC_TYPE[("M", "spatial")]  # p_s
B_PER_SEED = B_SPATIAL - B_COLOR  # d_s
_rows = [RECORDS[("P", s)]["correct"] for s in _seeds_b] + [RECORDS[("M", s)]["correct"] for s in _seeds_b]
_boot_type = {t: bootstrap_accuracies(_rows, t) for t in QUESTION_TYPES}  # 同じシードなので再標本は共通
_boot_diff = {t: _boot_type[t][:, :SEEDS_B].mean(axis=1) - _boot_type[t][:, SEEDS_B:].mean(axis=1) for t in QUESTION_TYPES}
_boot_contrast_b = _boot_diff["spatial"] - _boot_diff["color"]
DELTA_B = float(B_PER_SEED.mean())
SIGMA_B = combined_sigma(B_PER_SEED, _boot_contrast_b)

_p1b = {(c, s): learning_preconditions(RECORDS[(c, s)]) for c in ("P", "M") for s in _seeds_b}
precondition_status["P1(B)"] = all(all(v.values()) for v in _p1b.values())
_swapped_b = {c: float(np.mean([RECORDS[(c, s)]["accuracy_swapped"]["all"] for s in _seeds_b])) for c in ("P", "M")}
precondition_status["P2(B)"] = all(v <= P2_SWAPPED_MAX for v in _swapped_b.values())
_base_b = float(ACC_TYPE[("M", "spatial")].mean())
precondition_status["P3(B)"] = _base_b <= P3_ROOM_MAX
B_PRECONDITIONS = ["P0(B)", "P1(B)", "P2(B)", "P3(B)"]
B_COMPUTED = judge(DELTA_B, SIGMA_B["sigma"])
B_VERDICT = B_COMPUTED if all(precondition_status[k] for k in B_PRECONDITIONS) else "前提不成立"

print(f"{_tag}実験 B(シード {_seeds_b})")
for (_c, _t), _v in ACC_TYPE.items():
    print(f"  acc^{_t}_{_c}: {np.round(_v, 4).tolist()}")
print(f"  c_s(色の差): {np.round(B_COLOR, 4).tolist()}、p_s(位置関係の差): {np.round(B_SPATIAL, 4).tolist()}")
print(f"  d_s = p_s - c_s: {np.round(B_PER_SEED, 4).tolist()}")
print(
    f"  Delta_B = {DELTA_B:+.4f}、sigma_B = {SIGMA_B['sigma']:.4f}(シード間 {SIGMA_B['seed_term']:.4f}・ブートストラップ "
    f"{SIGMA_B['bootstrap_term']:.4f}、反復 {BOOTSTRAP_RESAMPLES:,})、閾値 2 sigma_B = {SIGMA_MULTIPLIER * SIGMA_B['sigma']:.4f}"
)
print(
    f"  前提条件: P0 {precondition_status['P0(B)']}、P1 {precondition_status['P1(B)']}"
    f"(不成立の学習 {[k for k, v in _p1b.items() if not all(v.values())] or 'なし'})、"
    f"P2 {precondition_status['P2(B)']}(画像を差し替えた正解率 {json.dumps({k: round(v, 4) for k, v in _swapped_b.items()})} <= {P2_SWAPPED_MAX})、"
    f"P3 {precondition_status['P3(B)']}(acc^spatial_M の平均 {_base_b:.4f} <= {P3_ROOM_MAX})"
)
print(f"  判定関数の結果 {B_COMPUTED} -> {_tag}最終判定: {B_VERDICT}")
# 診断量: 色の差と位置関係の差それぞれのシード平均(交互作用の成分)
print(f"  診断量: 色の差の平均 {B_COLOR.mean():+.4f}、位置関係の差の平均 {B_SPATIAL.mean():+.4f}")
_by_template = {}
for _c in ("P", "M"):
    for _t, _templates in (("color", COLOR_TEMPLATES), ("spatial", SPATIAL_TEMPLATES)):
        for _k in range(len(_templates)):
            _mask = np.array([q.question_type == _t and q.template_index == _k for q in QUESTIONS["eval"]])
            _by_template[(_c, _t, _k)] = float(np.mean([RECORDS[(_c, s)]["correct"][_mask].mean() for s in _seeds_b]))
print("  診断量: テンプレートごとの正解率(シード平均) " + "、".join(f"{k[0]} {k[1]} #{k[2]}: {v:.4f}" for k, v in _by_template.items()))

_fig, _ax = plt.subplots(figsize=(6, 3.6))
_width = 0.35
for _i, _c in enumerate(("P", "M")):
    _means = [ACC_TYPE[(_c, t)].mean() for t in QUESTION_TYPES]
    _ax.bar(np.arange(len(QUESTION_TYPES)) + (_i - 0.5) * _width, _means, _width, label=f"{_c} ({'patch grid' if _c == 'P' else 'mean pool'})")
    for _j, _t in enumerate(QUESTION_TYPES):
        _ax.plot(np.full(SEEDS_B, _j + (_i - 0.5) * _width), ACC_TYPE[(_c, _t)], "k.", markersize=4)
_ax.set_xticks(range(len(QUESTION_TYPES)), QUESTION_TYPES)
_ax.set_ylabel("exact-match accuracy (eval)")
_ax.set_title(f"{_plot_tag}Experiment B: visual tokens x question type")
_ax.legend()
plt.tight_layout()
plt.show()
```

    実験 B(シード [0, 1, 2, 3, 4])
      acc^color_P: [0.8331, 0.8348, 0.7995, 0.8296, 0.8733]
      acc^spatial_P: [0.6565, 0.5204, 0.7403, 0.6375, 0.5409]
      acc^color_M: [0.8092, 0.8061, 0.8106, 0.7832, 0.8096]
      acc^spatial_M: [0.5242, 0.528, 0.5215, 0.5087, 0.5201]
      c_s(色の差): [0.0239, 0.0287, -0.0111, 0.0464, 0.0637]、p_s(位置関係の差): [0.1323, -0.0076, 0.2188, 0.1288, 0.0208]
      d_s = p_s - c_s: [0.1084, -0.0364, 0.2299, 0.0824, -0.0429]
      Delta_B = +0.0683、sigma_B = 0.0510(シード間 0.0506・ブートストラップ 0.0064、反復 10,000)、閾値 2 sigma_B = 0.1020
      前提条件: P0 True、P1 True(不成立の学習 なし)、P2 True(画像を差し替えた正解率 {"P": 0.2014, "M": 0.2011} <= 0.5)、P3 True(acc^spatial_M の平均 0.5205 <= 0.95)
      判定関数の結果 判定不能 -> 最終判定: 判定不能
      診断量: 色の差の平均 +0.0303、位置関係の差の平均 +0.0986
      診断量: テンプレートごとの正解率(シード平均) P color #0: 0.8320、P color #1: 0.8361、P spatial #0: 0.6100、P spatial #1: 0.6283、M color #0: 0.7960、M color #1: 0.8115、M spatial #0: 0.5227、M spatial #1: 0.5183



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/021_llava_visual_instruction_tuning/output_41_1.png)
    


### 6.9 診断量: 線形プローブ(判定なし)

5.7 節の線形プローブを、2 つの標的 × 2 つの入力で学習・評価する。実験 B の結果の解釈(特に支持以外の結果が出たときの「なぜそうなったか」の
考察)の材料とする。


```python
_t0_probe = time.time()
PROBE_RESULTS = {(t, i): run_probe(t, i) for t in PROBE_TARGETS for i in PROBE_INPUTS}
PROBE_SECONDS = time.time() - _t0_probe
for (_target, _inputs), _r in PROBE_RESULTS.items():
    print(
        f"{_tag}線形プローブ 標的 {_target}・入力 {_inputs}: 評価用の正解率 {_r['eval']:.4f}(学習用 {_r['train']:.4f}、"
        f"最終の訓練損失 {_r['final_loss']:.4f}、チャンス 1/{_r['classes']} = {1 / _r['classes']:.3f}、最頻のクラス {_r['majority']:.3f})"
    )
print(f"線形プローブ: {PROBE_SECONDS:.1f}s(見積もり {ESTIMATE_PROBE:.1f}s)")
```

    線形プローブ 標的 layout・入力 patch: 評価用の正解率 0.8774(学習用 1.0000、最終の訓練損失 0.0056、チャンス 1/10 = 0.100、最頻のクラス 0.107)
    線形プローブ 標的 layout・入力 mean: 評価用の正解率 0.8012(学習用 0.8031、最終の訓練損失 0.5302、チャンス 1/10 = 0.100、最頻のクラス 0.107)
    線形プローブ 標的 shape_pair・入力 patch: 評価用の正解率 0.6641(学習用 1.0000、最終の訓練損失 0.0371、チャンス 1/10 = 0.100、最頻のクラス 0.119)
    線形プローブ 標的 shape_pair・入力 mean: 評価用の正解率 0.6281(学習用 0.6605、最終の訓練損失 0.9191、チャンス 1/10 = 0.100、最頻のクラス 0.119)
    線形プローブ: 7.7s(見積もり 10.8s)


### 6.10 不変条件のアサーションと`SMOKE_TEST`の配線

- 5.2 節で印字した水準(ステップ数・シード数・計画)と、実際の学習の記録が一致すること。
- 同じシードの条件どうしで、projection 層と LoRA の初期値、第 2 段階のミニバッチの順序が一致すること(L は第 2 段階が長いので、
  ミニバッチの添字の列の先頭 $T_2$ ステップが一致することを 5.5 節で確かめた)。第 1 段階を行う P・M の第 1 段階のミニバッチの順序が
  一致すること。
- 学習率が 6.5 節の規則のとおりであること。較正の学習の記録が評価集合の鍵を持たないこと。
- 評価の事例数・質問の種類の事例数が全条件で同じであること。


```python
# --- 実効水準の照合(SMOKE_TEST の配線) ---
assert BOOTSTRAP_RESAMPLES == LEVELS[CURRENT_LEVEL_NAME]["BOOTSTRAP_RESAMPLES"]
assert STAGE_CFG == STAGES[CURRENT_LEVEL_NAME][STAGE] and SELECTED_PLAN in PLANS
for _c in CONDITIONS:
    _seeds = sorted(s for (c, s) in RECORDS if c == _c)
    assert _seeds == list(range(NUM_SEEDS[_c])), (_c, _seeds)
    assert NUM_SEEDS[_c] == seeds_for(_c, STAGES[CURRENT_LEVEL_NAME][STAGE])
for (_c, _s), _r in RECORDS.items():
    _spec = CONDITIONS[_c]
    assert _r["stage2_history"]["steps"] == _spec["stage2_steps"] == len(_r["stage2_history"]["loss"])
    assert ("stage1_history" in _r) == _spec["stage1"]
    if _spec["stage1"]:
        assert _r["stage1_history"]["steps"] == STAGE1_STEPS == len(_r["stage1_history"]["loss"])
        assert all(n.startswith("projection.") for n in _r["stage1_history"]["trainable_names"])  # 第 1 段階は projection 層のみ
    _names2 = _r["stage2_history"]["trainable_names"]  # 第 2 段階は projection 層と LoRA のみ
    assert all(n.startswith("projection.") or ".lora_" in n for n in _names2)
    assert sum(".lora_" in n for n in _names2) == 2 * len(LORA_TARGET_MODULES) * NUM_LAYERS
    assert (_r["lr1"], _r["lr2"]) == LEARNING_RATES[_c]
    assert len(_r["correct"]) == len(_r["correct_swapped"]) == len(EVAL_ANSWERS)
# --- 対応のある比較: 同じシードの条件どうしで初期値・ミニバッチの順序が一致する ---
for _s in range(max(NUM_SEEDS.values())):
    _present = [c for c in CONDITIONS if (c, _s) in RECORDS]
    assert len({RECORDS[(c, _s)]["initial_trainable_sha256"] for c in _present}) == 1
    assert len({RECORDS[(c, _s)]["lora_init_sha256"] for c in _present}) == 1
    _same_length = [c for c in _present if CONDITIONS[c]["stage2_steps"] == STAGE2_STEPS]
    assert len({RECORDS[(c, _s)]["stage2_history"]["batch_sha256"] for c in _same_length}) == 1
    _with_stage1 = [c for c in _present if CONDITIONS[c]["stage1"]]
    assert len({RECORDS[(c, _s)]["stage1_history"]["batch_sha256"] for c in _with_stage1}) <= 1
# 異なるシードどうしでは初期値が異なる(確認の検出力)
if NUM_SEEDS["P"] >= 2:
    assert RECORDS[("P", 0)]["initial_trainable_sha256"] != RECORDS[("P", 1)]["initial_trainable_sha256"]
# --- 較正 ---
assert set(CALIBRATION) == set(calibration_targets(CALIBRATION_MODE))
for _key, _c in CALIBRATION.items():
    assert _c["chosen"] in _c["nll"] and len(_c["nll"]) in (3, 4)
# --- 評価集合の事例数は全条件で同じ ---
assert len(EVAL_ANSWERS) == 4 * NUM_IMAGES["eval"]
assert all(int((QUESTION_TYPE_OF_EVAL == t).sum()) == 2 * NUM_IMAGES["eval"] for t in QUESTION_TYPES)
print(
    f"実効水準の照合: 計画 {SELECTED_PLAN['plan']}・段階 {STAGE}・較正 {CALIBRATION_MODE!r}・シード数 {NUM_SEEDS}・"
    f"T_1 = {STAGE1_STEPS}・T_2 = {STAGE2_STEPS}・ブートストラップ {BOOTSTRAP_RESAMPLES:,} 回が、実際の学習の記録と一致: OK"
)
print("同じシードの条件どうしで projection 層・LoRA の初期値とミニバッチの順序が一致、学習率は 6.5 節の規則どおり、較正は評価集合を評価していない: OK")
```

    実効水準の照合: 計画 0・段階 0・較正 'all'・シード数 {'P': 5, 'R': 5, 'L': 5, 'M': 5}・T_1 = 722・T_2 = 1444・ブートストラップ 10,000 回が、実際の学習の記録と一致: OK
    同じシードの条件どうしで projection 層・LoRA の初期値とミニバッチの順序が一致、学習率は 6.5 節の規則どおり、較正は評価集合を評価していない: OK


### 6.11 判定結果の一覧


```python
print(f"{_tag}判定結果(計画 {SELECTED_PLAN['plan']}、段階 {STAGE}、較正の方式 {CALIBRATION_MODE!r})")
for _name, _delta, _sigma, _computed, _verdict, _preconditions in (
    ("実験 A(第 1 段階の効果)", DELTA_A, SIGMA_A["sigma"], A_COMPUTED, A_VERDICT, A_PRECONDITIONS),
    ("実験 B(視覚トークンの取り方 x 質問の種類)", DELTA_B, SIGMA_B["sigma"], B_COMPUTED, B_VERDICT, B_PRECONDITIONS),
):
    print(
        f"  {_name}: 対比量 {_delta:+.4f}、標準偏差 {_sigma:.4f}、閾値 {SIGMA_MULTIPLIER * _sigma:.4f}、"
        f"前提条件 {json.dumps({k: precondition_status[k] for k in _preconditions}, ensure_ascii=False)}、"
        f"判定関数の結果 {_computed} -> 最終判定 {_verdict}"
    )
print(f"\nノートブック全体の実行時間: {(time.time() - NOTEBOOK_START_TIME) / 60:.1f} 分")
```

    判定結果(計画 0、段階 0、較正の方式 'all')
      実験 A(第 1 段階の効果): 対比量 -0.0116、標準偏差 0.0177、閾値 0.0354、前提条件 {"P0(A)": true, "P1(A)": true, "P2(A)": true, "P3(A)": true}、判定関数の結果 判定不能 -> 最終判定 判定不能
      実験 B(視覚トークンの取り方 x 質問の種類): 対比量 +0.0683、標準偏差 0.0510、閾値 0.1020、前提条件 {"P0(B)": true, "P1(B)": true, "P2(B)": true, "P3(B)": true}、判定関数の結果 判定不能 -> 最終判定 判定不能
    
    ノートブック全体の実行時間: 45.1 分


## 7. 結果・考察 / Results and Discussion

判定(7.3・7.4 節)は 6.1 節で事前に宣言した基準のみから導く。結果を見た後に立てた解釈は 7.6 節に分けて記し、検証済みの結論としては
扱わない。数値はすべて、このノートブックのセル出力(本番実行)のものである。

### 7.1 実行の概要

**実行環境**(5.1 節の印字): Google Colab の Tesla T4(compute capability 7.5、総メモリ 14.56 GiB)、Python 3.13.15、torch 2.13.0+cu130
(ビルド時の CUDA 13.0、cuDNN 92000)、コミット`7cc0ceb`(未コミットの変更なし)、実行日時 2026-10-01T08:21:21 UTC、`SMOKE_TEST = False`、
FP32。

**準備と計測の時間**: モデルの取得と構築 14.7 秒、データの準備 15.2 秒、不変条件の確認 14.0 秒、スケーリングの計測 30.1 秒。計画の選択の
時点の経過時間は 1.6 分だった。

**定常状態の 1 ステップの時間**(6.3 節、バッチ 32):

| 処理 | 1 ステップ | べき指数 $b$ | べき乗則の外挿と比例の値の差 |
|---|---|---|---|
| 第 1 段階・パッチトークン全体 | 35.0 ms | 0.993 | $T = 722$ で −0.3 秒 |
| 第 1 段階・平均プール | 17.4 ms | 0.962 | $T = 722$ で −0.9 秒 |
| 第 2 段階・パッチトークン全体 | 36.8 ms | 0.999 | $T = 1{,}444$ で −0.1 秒 |
| 第 2 段階・平均プール | 28.3 ms | 1.166 | $T = 1{,}444$ で +25.8 秒 |

第 2 段階・平均プールだけ、べき指数が 1 から離れ、べき乗則の外挿値(66.7 秒)が比例の値(40.9 秒)を大きく上回った。16 ステップの計測
(0.34 秒)が 32・64 ステップ(0.85・1.70 秒)に比べて短く、短い区間の計測の揺らぎの可能性がある。見積もりは比例の値を使っており、
実際の M の学習 1 回(評価を含む)は約 61 秒だった。第 2 段階・パッチトークン全体の 1 ステップは、バッチ 16 で 20.1 ms、バッチ 32 で
36.8 ms(比 1.83)で、時間はバッチサイズにほぼ比例した(固定費は支配的ではない)。

**計画の選択と実行時間**: 8 通りの計画はすべて予算に収まり、**計画 0**(段階 0・較正の方式`"all"`、P・R・L・M とも 5 シード)が選ばれた
(残りの見積もり 47.9 分、終了の見込み 49.5 分)。

| 区分 | 見積もり | 実際 |
|---|---|---|
| 学習率の較正 | 13.8 分 | 10.8 分 |
| 本番の学習と評価 | 33.0 分 | 32.5 分 |
| 線形プローブ | 10.8 秒 | 7.7 秒 |
| ノートブック全体 | 49.5 分(終了の見込み) | 45.1 分 |

較正の見積もりは、格子の拡張が起きる最悪の場合(各条件 4 点)を仮定している。実際には拡張が起きず各条件 3 点で済んだので、
実際の時間は見積もりより短かった。

**学習率の較正**(6.5 節、検証集合の答えのトークンあたりの負の対数尤度、較正専用のシード):

| 条件 | $4 \times 10^{-3}$ | $8 \times 10^{-3}$ | $1.6 \times 10^{-2}$ | 拡張 | 選んだ学習率 |
|---|---|---|---|---|---|
| P | 0.0971 | 0.0793 | 0.0935 | なし | $8 \times 10^{-3}$(内点) |
| R | 0.0879 | 0.0806 | 0.1012 | なし | $8 \times 10^{-3}$(内点) |
| M | 0.1006 | 0.0912 | 0.1012 | なし | $8 \times 10^{-3}$(内点) |

3 条件とも格子の中心が選ばれた。L は規則により R の値($8 \times 10^{-3}$)を使った。第 1 段階の学習率は固定の $6.4 \times 10^{-2}$ である。

### 7.2 前提条件

| 前提条件 | 実験 A | 実験 B |
|---|---|---|
| P0(較正、選んだ学習率が内点) | 成立(P・R とも内点) | 成立(P・M とも内点) |
| P1(学習の成立) | 成立(P・R の全 10 回、不成立の学習なし) | 成立(P・M の全 10 回、不成立の学習なし) |
| P2(画像を差し替えた正解率 $\le 0.5$) | 成立(P 0.2014、R 0.2026) | 成立(P 0.2014、M 0.2011) |
| P3(基準の条件の正解率 $\le 0.95$) | 成立($\overline{\mathrm{acc}}_{R} = 0.7382$) | 成立($\overline{\mathrm{acc}}^{\mathrm{spatial}}_{M} = 0.5205$) |

前提条件は両実験ですべて成立した。P1 の内訳として、第 2 段階の検証集合の負の対数尤度は、どの学習でも開始時(3.2〜7.1)から終わり
(0.05〜0.10)まで 0.5 倍を大きく下回って下がった。第 1 段階のキャプションの負の対数尤度は、P で 6.3〜6.5 から約 2.57、M で約 6.15 から
4.59 に下がった。

画像を差し替えたときの正解率(約 0.20)は、画像を見ずに質問の文字列だけから答える戦略の上界(5.4 節、全体 0.2261)を下回る。
差し替えない場合の正解率(0.65〜0.86)との差は、モデルが画像を使って答えていることを示す。

### 7.3 実験 A: 第 1 段階(特徴の整列)の効果

**最終判定: 判定不能**(前提条件はすべて成立)。

| シード $s$ | $\mathrm{acc}_{P,s}$ | $\mathrm{acc}_{R,s}$ | $d_s = \mathrm{acc}_{P,s} - \mathrm{acc}_{R,s}$ |
|---|---|---|---|
| 0 | 0.7448 | 0.7940 | −0.0492 |
| 1 | 0.6776 | 0.7235 | −0.0459 |
| 2 | 0.7699 | 0.7232 | +0.0467 |
| 3 | 0.7336 | 0.7377 | −0.0042 |
| 4 | 0.7071 | 0.7126 | −0.0055 |
| 平均 | 0.7266 | 0.7382 | −0.0116 |

対比量は $\Delta_A = -0.0116$、標準偏差は $\sigma_A = 0.0177$(シード間 0.0174、ブートストラップ 0.0030、反復 10,000 回)、閾値は
$2\sigma_A = 0.0354$ である。$\lvert \Delta_A \rvert = 0.0116$ は閾値 0.0354 より小さく、$\Delta_A > 2\sigma_A$(支持)も $\Delta_A < -2\sigma_A$(反証)も
満たさない。「第 1 段階が、第 2 段階のステップ数を揃えたときの正解率を上げる」という仮説は、支持も反証もされなかった。標準偏差の
ほとんどはシード間のばらつきによる。

**診断量**(判定なし):

第 2 段階の検証集合の答えのトークンあたりの負の対数尤度(シード平均):

| 条件 | 開始時 | 25% | 50% | 75% | 終わり |
|---|---|---|---|---|---|
| P | 3.2490 | 0.1443 | 0.1134 | 0.0917 | 0.0823 |
| R | 6.9123 | 0.1441 | 0.1106 | 0.0875 | 0.0786 |
| L | 6.9123 | 0.1351 | 0.0969 | 0.0738 | 0.0633 |
| M | 4.8001 | 0.1549 | 0.1233 | 0.1001 | 0.0919 |

(L は第 2 段階が 2,166 ステップなので、同じ割合の時点のステップ数は P・R・M の 1.5 倍である。)

視覚トークンの平均ノルム(シード平均。括弧内は言語モデルのトークン埋め込みの平均ノルム 0.8512 に対する比):

| 条件 | 初期値 | 第 1 段階の後 | 第 2 段階の後 |
|---|---|---|---|
| P | 25.61(30.1 倍) | 251.35(295.3 倍) | 280.05(329.0 倍) |
| R | 25.61(30.1 倍) | (第 1 段階なし) | 89.06(104.6 倍) |
| L | 25.61(30.1 倍) | (第 1 段階なし) | 102.02(119.8 倍) |
| M | 23.82(28.0 倍) | 418.99(492.2 倍) | 389.15(457.2 倍) |

L(第 1 段階なしで、第 2 段階を総ステップ数 $T_1 + T_2 = 2{,}166$ だけ学習)の正解率は、シード順に 0.8627・0.7741・0.7415・0.7812・0.8570、
平均 0.8033 だった。P の平均 0.7266、R の平均 0.7382 より高い。L は判定を置かない診断量である。

### 7.4 実験 B: 視覚トークンの取り方 × 質問の種類

**最終判定: 判定不能**(前提条件はすべて成立)。

| シード $s$ | $\mathrm{acc}^{\mathrm{color}}_{P,s}$ | $\mathrm{acc}^{\mathrm{color}}_{M,s}$ | $c_s$ | $\mathrm{acc}^{\mathrm{spatial}}_{P,s}$ | $\mathrm{acc}^{\mathrm{spatial}}_{M,s}$ | $p_s$ | $d_s = p_s - c_s$ |
|---|---|---|---|---|---|---|---|
| 0 | 0.8331 | 0.8092 | +0.0239 | 0.6565 | 0.5242 | +0.1323 | +0.1084 |
| 1 | 0.8348 | 0.8061 | +0.0287 | 0.5204 | 0.5280 | −0.0076 | −0.0364 |
| 2 | 0.7995 | 0.8106 | −0.0111 | 0.7403 | 0.5215 | +0.2188 | +0.2299 |
| 3 | 0.8296 | 0.7832 | +0.0464 | 0.6375 | 0.5087 | +0.1288 | +0.0824 |
| 4 | 0.8733 | 0.8096 | +0.0637 | 0.5409 | 0.5201 | +0.0208 | −0.0429 |
| 平均 | 0.8341 | 0.8037 | +0.0303 | 0.6191 | 0.5205 | +0.0986 | +0.0683 |

対比量は $\Delta_B = +0.0683$、標準偏差は $\sigma_B = 0.0510$(シード間 0.0506、ブートストラップ 0.0064、反復 10,000 回)、閾値は
$2\sigma_B = 0.1020$ である。$\Delta_B = 0.0683$ は閾値 0.1020 より小さく、支持の条件 $\Delta_B > 2\sigma_B$ を満たさない。反証の条件
$\Delta_B < -2\sigma_B$ も満たさない。「平均プールによる正解率の低下は、色の質問より位置関係の質問で大きい」という仮説は、支持も反証も
されなかった。$d_s$ は 5 シードのうち 3 つで正、2 つで負であり、標準偏差のほとんどはシード間のばらつきによる。

**診断量**(判定なし):

- 交互作用の成分: 色の差の平均 $\bar{c} = +0.0303$、位置関係の差の平均 $\bar{p} = +0.0986$。
- テンプレートごとの正解率(シード平均):

| 条件 | 色・テンプレート 0 | 色・テンプレート 1 | 位置関係・テンプレート 0 | 位置関係・テンプレート 1 |
|---|---|---|---|---|
| P | 0.8320 | 0.8361 | 0.6100 | 0.6283 |
| M | 0.7960 | 0.8115 | 0.5227 | 0.5183 |

  同じ質問の種類の 2 つのテンプレートの差は 0.02 以内で、質問の種類の間の差(0.2〜0.3)に比べて小さい。

### 7.5 診断量: 線形プローブ

| 標的 | 入力 | 評価用の正解率 | 学習用の正解率 | 最終の訓練損失 | チャンス | 最頻のクラス |
|---|---|---|---|---|---|---|
| 配置 | 格子全体(8,192 次元) | 0.8774 | 1.0000 | 0.0056 | 0.100 | 0.107 |
| 配置 | 平均プール(128 次元) | 0.8012 | 0.8031 | 0.5302 | 0.100 | 0.107 |
| 形の組 | 格子全体(8,192 次元) | 0.6641 | 1.0000 | 0.0371 | 0.100 | 0.119 |
| 形の組 | 平均プール(128 次元) | 0.6281 | 0.6605 | 0.9191 | 0.100 | 0.119 |

**手順の制約**: 2 つの入力のプローブは、容量と最適化の到達点が違う。

- 格子全体の入力のプローブは、学習用の正解率が 1.0000 に達し、評価用(配置 0.8774、形の組 0.6641)との差が大きい(過学習)。入力の次元
  8,192 は学習用の画像の数 11,552 に近く、学習データを暗記できる。
- 平均プールの入力のプローブは、学習用の正解率が評価用とほぼ同じ(配置で 0.8031 と 0.8012)で、最終の訓練損失も 0.53・0.92 と大きい。
  500 ステップでは学習データにも当てはまりきっていない(学習不足)可能性がある。
- したがって、**2 つの入力のプローブの正解率の差(配置で 0.076、形の組で 0.036)を、特徴の持つ情報の量の差と解釈することはできない。**
- 一方、プローブの評価用の正解率は、その特徴から線形に読み出せる情報の量の **下界** とみなせる(より良く学習したプローブは、これ以上の
  正解率に達しうる)。平均プールの特徴から、配置(位置関係の種類と、左または上の図形の形)が少なくとも 0.80 の正解率で線形に読み出せる。

形の組は、位置によらない内容の対照として置いたが、どちらの入力でも配置より正解率が低かった(0.63〜0.66)。形の組が「対照として
より易しい」という想定は成り立たなかった。

### 7.6 事後的な解釈(検証済みの結論ではない)

**この節の内容はすべて、結果を見た後に立てた解釈である。事前に宣言した基準による結論(7.3・7.4 節の「判定不能」)とは別であり、
検証済みの結論として扱わない。** 診断量は判定を置いていない量であり、ここでの比較にも判定の閾値はない。

#### 実験 A が判定不能になった理由の候補

**候補 1: 第 1 段階の効果は作用点では大きいが、第 2 段階の初めのうちに消える。**

- 原因の候補: projection 層(33,024 パラメータ)の整列は、第 2 段階の中でも短いステップ数で学習でき、第 1 段階で先に済ませておく利点が
  残らない。
- 根拠: 第 2 段階の開始時の検証集合の負の対数尤度は、P 3.2490・R 6.9123 と大きく違う(介入の作用点に近い量)。しかし 25% の時点
  (361 ステップ)では 0.1443・0.1441 とほぼ同じになり、終わりでは R のほうが低い(P 0.0823・R 0.0786)。途中の評価は 25% より前には
  ないので、差がいつ消えたかはこの出力からは分からない。
- 確かめる方法: 第 2 段階の初めの数十〜数百ステップで検証集合の負の対数尤度を細かく記録し、P と R の差が消えるステップを測る。
- 教訓: 作用点に近い量を診断量として記録したことで、「作用点では差があり、下流で消えた」ことまでは読み取れた。途中の評価の位置は、
  差が消えうる初期に密に置くべきだった。

**候補 2: 規則で決めた第 1 段階の学習率が、P に不利に働いた。**

- 原因の候補: 第 1 段階の学習率は較正せず $6.4 \times 10^{-2}$ に固定した。6.1 節で、この値が最適からずれたときの偏りは実験 A では負の向き
  (第 1 段階の効果を過小に見る側)と宣言していた。
- 根拠: 第 1 段階の後の P の視覚トークンの平均ノルムは、トークン埋め込みの 295.3 倍で、初期値(30.1 倍)の約 10 倍に増えた。第 1 段階を
  行わない R の第 2 段階の後は 104.6 倍である。第 1 段階は視覚トークンを言語モデルの埋め込みの尺度に近づけず、逆に遠ざけており、P は
  その状態から第 2 段階を始めている。ただし、ノルムが大きいことが第 2 段階の学習を妨げたことを直接示す数値はない。
- 確かめる方法: 第 1 段階の学習率を掃引し、第 1 段階の指標ではなく第 2 段階の後の正解率で比べる。projection 層の出力のノルムを
  トークン埋め込みのノルムに合わせた初期化(または出力の正規化)を加えた条件と比べる。
- 教訓: 規則で決めた値による偏りの向きを事前に宣言していたので、この候補は結果を見る前から挙がっていた。較正の指標(第 1 段階の
  キャプションの負の対数尤度)が平坦だったことは、第 1 段階の学習率が下流に影響しないことを意味しない。

**候補 3: この規模では、同じステップ数を第 2 段階に回すほうが有利だった(学習不足)。**

- 原因の候補: $T_2 = 1{,}444$ では第 2 段階の学習が飽和しておらず、正解率は第 2 段階のステップ数で決まる部分が大きい。
- 根拠: L の正解率の平均 0.8033 は、R(0.7382)より 0.065、P(0.7266)より 0.077 高い。P と L は総ステップ数が同じ(2,166)で、違いは
  先頭の 722 ステップを第 1 段階に使うか第 2 段階に使うかである。検証集合の負の対数尤度も、P・R・L とも 75% から終わりまで下がり続けて
  いる(R で 0.0875 → 0.0786)。L は判定を置かない診断量であり、L と P・R の差に閾値との比較はない。
- 確かめる方法: $T_2$ を延ばした水準(例えば 2 倍・4 倍)で P・R・L を比べ、学習が飽和した後も順位が変わらないかを見る。
- 教訓: 総ステップ数を揃えた条件を診断量として置いたことで、「第 1 段階の有無」と「学習量」の交絡の向きが読み取れた。

**原論文との違い**: 原論文の言語モデルは本トピックの 2,000 倍以上のパラメータを持ち、第 2 段階では全パラメータを更新し、第 1 段階の
データも数十万組ある。本トピックでは、言語モデルが小さく、第 2 段階で動かせるのは Query・Value への LoRA と projection 層だけで、
課題も 2 種類の質問に限られる。第 1 段階が「言語モデルを壊さずに視覚トークンを配置する」ことの利点は、第 2 段階で言語モデルが大きく
動く設定で現れやすいと考えられるが、本トピックの出力からは確かめられない。

#### 実験 B が判定不能になった理由の候補

**候補 1: ボトルネックは視覚の情報ではなく、言語モデル側の学習にある。**

- 原因の候補: 視覚トークンは位置関係に答えるのに必要な情報を持っているが、言語モデル(LoRA と projection 層)がそれを使うことを
  $T_2$ の範囲で学習しきれていない。
- 根拠: 平均プールの特徴から、配置(位置関係の種類に加えて、左または上の図形の形まで)が線形プローブで 0.8012 の正解率で読み出せる
  (7.5 節)。それに対して、M の位置関係の質問の正解率は 0.5205 である。格子全体でも、プローブ 0.8774 に対して P の位置関係の正解率は
  0.6191(シード平均)にとどまる。7.5 節のとおり、プローブの正解率は線形に読み出せる情報の量の下界なので、プローブの学習不足・過学習は
  この解釈を弱める向きには働かない(より良いプローブなら、差はさらに開く)。ただし、プローブの標的(10 クラスの配置)と質問(指定された
  2 つの形の相対位置を 4 択で答える)は同じ課題ではなく、厳密には直接比べられない。
- 3.6 節の非対称の仮定との関係: 3.6 節では「位置の情報が系列の中の位置だけで表されているなら、平均を取ると失われる」と述べた。
  平均プールの特徴から配置が 0.80 で線形に読み出せたので、位置と内容の組の情報は、系列の中の位置だけでなく、パッチトークンの
  ベクトルの中身にも含まれている。ただし、各パッチトークンが自分の位置にある内容を表しているのか、3 層の自己注意を経て画像全体の
  配置の要約を持っているのか(3.4 節のとおり、最終層の一つ前の層のパッチトークンは [CLS] トークンに画像の情報を渡すように学習されて
  いる)は、この出力からは区別できない。
  平均を取っても、位置の情報は少なくとも配置を 0.80 の正解率で線形に読み出せる程度には残っており、平均で位置の情報が失われるという
  想定は、少なくともその大部分については成り立っていなかった。格子全体のプローブ(0.8774)との差は、7.5 節のとおり手順の違いを含むので、
  平均でどれだけの情報が失われたかはこの出力からは分からない。
- 確かめる方法: $T_2$ を延ばして P・M の位置関係の正解率がプローブの正解率に近づくかを見る。LoRA の対象や rank を広げた条件、
  言語モデルの全パラメータを更新する条件と比べる。パッチトークンが何を表しているかについては、パッチトークンを 1 個ずつ入力にして、
  その位置にある図形の形・色を当てる線形プローブ(位置ごとのプローブ)を学習し、各トークンが自分の位置の内容を保っているかを測る。
- 教訓: 介入の作用点に近い量(視覚トークンから読み出せる情報)を、本番の前のパイロットで測っておけば、「平均プールで位置の情報が
  失われる」という仮説の前提が成り立たないことを、実験の設計の段階で知ることができた。

**候補 2: M は位置関係の「軸」は分かるが「向き」が分からない状態にとどまっている。**

- 原因の候補: 位置関係の答えは 4 択(`left`・`right`・`above`・`below`)で、軸(左右か上下か)だけを正しく選び、向きを当て推量すると
  正解率は 0.5 になる。
- 根拠: M の位置関係の正解率は 5 シードとも 0.5087〜0.5280 の狭い範囲にある(色の質問は 0.78〜0.81)。画像を見ない戦略の上界は位置関係で
  0.2722 なので、0.52 は画像を使った結果である。ただし、今回記録したのは事例ごとの正誤のみで、生成した答えそのものは残していないので、
  誤答が同じ軸の反対の向きに集中しているかどうかは確かめられない。
- 確かめる方法: 生成した答えを記録し、正解と答えの混同行列を作る。
- 教訓: 閉じた語彙の答えを生成で採点する評価では、正誤だけでなく生成した答えそのものを記録しておく。

**候補 3: シード間のばらつきが大きく、検出力が足りなかった。**

- 原因の候補: 位置関係の向きの獲得が、シードによって $T_2$ までに起きたり起きなかったりする(二峰的な学習)。
- 根拠: P の位置関係の正解率はシードによって 0.5204〜0.7403 と大きく違い、0.52〜0.54 のシード(1・4)と 0.64〜0.74 のシード(0・2・3)に
  分かれる。R も 0.7310 のシード 0 と 0.56〜0.61 の他のシードに、L も 0.83・0.81 のシード(0・4)と 0.57〜0.66 のシード(1・2・3)に分かれる。
  第 2 段階が長い L で高いシードが多いことは、向きの獲得が学習の途中で起き、学習が長いほど起きやすいという見方と整合する。
  $\sigma_B = 0.0510$ のうちシード間が 0.0506 で、ブートストラップ(評価集合の抽出)は 0.0064 にすぎない。
- 確かめる方法: シード数を増やす。位置関係の正解率を第 2 段階の途中で記録し、シードごとに 0.5 付近から離れる時点があるかを見る。
- 教訓: **検出力の事前確認。** 本番の前のパイロットで、基準となる条件の対比量のシード間のばらつきを測り、そのシード数で検出できる
  最小の効果量を判定基準とともに宣言する。今回のパイロットは、基準の条件が上限に届くかの真偽だけを確かめており、ばらつきの大きさは
  見ていなかった。シード間の標準偏差が 0.11($d_s$ の標本標準偏差 $0.0506 \times \sqrt{5}$)のとき、5 シードで検出できる差は約 0.10 以上で
  あり、観測された $\Delta_B = 0.068$ はそれより小さい。

#### 共通: 結論の一般化の制約

本トピックの構成は、原論文と、言語モデルの規模(約 525 万パラメータ)、第 2 段階の更新の範囲(LoRA と projection 層)、画像 encoder
(合成のシーンで学習した小さな ViT、$N = 64$)、指示データ(規則で生成した 2 種類の質問)のすべてで異なる。ここでの「判定不能」は、
この構成と学習量のもとで、事前に宣言した基準では差を検出できなかったことを意味し、原論文の設定で第 1 段階やパッチトークンの格子に
効果がないことを意味しない。


## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/05_vision_language/021_llava_visual_instruction_tuning.ipynb
