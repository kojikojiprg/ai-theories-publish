---
title: "テキスト埋め込みと retriever / Text Embedding and Retriever(実装・実験編 5/7)"
---

この記事は後編(実装・実験編 5/7)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/024_text_embedding_and_retriever-practice-4)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/024_text_embedding_and_retriever-practice-6)。

### 6.4 実行計画の選択

6.3 節の計測から、12 通りの計画(6.1 節)のそれぞれの残りの実行時間を見積もる。

- 学習 1 回 = 1 ステップの時間 × $T$ + 学習の組の生成の時間(1 ステップあたりの時間 × $T$。安全側に毎回数える)+ 学習前のモデルの損失の測定の時間(最後の区間のステップ数 × 1 ステップあたりの時間)+ 評価の時間 + 固定費の余裕 2 秒。評価の時間は、較正の学習では検証用の集合の評価 1 回、本番の学習では最終ステップの評価
  (検証用 + 評価用 + 学習用の部分)1 回に、追跡する条件(C2・C3・C4)では途中の評価の位置の数だけ検証用の集合の評価を加えたもの。
- 較正 = 較正する条件ごとに、格子 3 点 + 拡張 1 点の **最悪の場合の 4 回**。
- 本番 = 条件ごとのシード数 × 学習 1 回。
- 判定・ブートストラップ・観察・図・アップロードの準備の余裕 150 秒と、BM25 の時間。

**経過時間 + 見積もりが予算(120 分)に収まる計画のうち、番号の最も小さいものを選ぶ。** どれも収まらなければ、学習の前に停止する。


```python
def run_seconds(condition: str, num_steps: int, main: bool) -> float:
    spec = CONDITIONS[condition]
    seconds = STEP_SECONDS[condition] * num_steps + RUN_OVERHEAD_SECONDS + SCHEDULE_SECONDS_PER_STEP * num_steps  # 学習の組の生成は(シード、T)ごとに 1 回だが、安全側に毎回数える
    seconds += INITIAL_LOSS_SECONDS_PER_STEP * final_loss_window(num_steps)  # 学習前のモデルの、最後の区間の組での損失(P1(a)の分母)
    if not main:
        return seconds + CALIBRATION_EVAL_SECONDS
    tracked = len(eval_steps_for(num_steps)) if spec["track"] else 0
    return seconds + EVAL_SECONDS["final"] + tracked * EVAL_SECONDS["validation"]


CALIBRATION_POINTS_WORST = len(LEARNING_RATE_GRID_MULTIPLIERS) + 1  # 格子 3 点 + 拡張 1 点


def estimate_plan(plan: dict, level: str = CURRENT_LEVEL_NAME) -> dict:
    stage = STAGES[level][plan["stage"]]
    steps = plan["num_steps"]
    calibration = sum(
        CALIBRATION_POINTS_WORST * run_seconds(c, steps, main=False) for c in calibration_targets(stage["mode"], stage["seeds"])
    )
    main = sum(stage["seeds"][c] * run_seconds(c, steps, main=True) for c in CONDITIONS)
    return {"calibration": calibration, "main": main, "total": calibration + main + REPORTING_MARGIN_SECONDS + BM25_SECONDS}


ELAPSED_AT_SELECTION = time.time() - NOTEBOOK_START_TIME
PLAN_ESTIMATES = {p["plan"]: estimate_plan(p) for p in PLANS}
print(f"経過時間 {ELAPSED_AT_SELECTION / 60:.1f} 分、予算 {SESSION_BUDGET_SECONDS / 60:.0f} 分")
for _p in PLANS:
    _e = PLAN_ESTIMATES[_p["plan"]]
    _stage = STAGES[CURRENT_LEVEL_NAME][_p["stage"]]
    _fits = ELAPSED_AT_SELECTION + _e["total"] <= SESSION_BUDGET_SECONDS
    print(
        f"  計画 {_p['plan']:>2}(T = {_p['num_steps']}、段階 {_p['stage']}: 較正 {_stage['mode']!r}、シード数 {dict(_stage['seeds'])}): "
        f"較正 {_e['calibration'] / 60:.1f} 分 + 本番 {_e['main'] / 60:.1f} 分 + 余裕 -> 残り {_e['total'] / 60:.1f} 分、終了の見込み "
        f"{(ELAPSED_AT_SELECTION + _e['total']) / 60:.1f} 分({'収まる' if _fits else '超える'})"
    )
_feasible = [p for p in PLANS if ELAPSED_AT_SELECTION + PLAN_ESTIMATES[p["plan"]]["total"] <= SESSION_BUDGET_SECONDS]
AUTO_PLAN = _feasible[0] if _feasible else None
if FORCED_PLAN is not None:
    SELECTED_PLAN = PLANS[FORCED_PLAN]
    print(f"*** テスト専用の上書き: 計画 {FORCED_PLAN} を使う(見積もりによる選択は {AUTO_PLAN and AUTO_PLAN['plan']}) ***")
elif AUTO_PLAN is None:
    raise RuntimeError(
        f"最も下位の計画(計画 {len(PLANS) - 1})でも見積もりが予算を超えるため、学習の前に停止する。結果の情報は何も得ていない。"
        "実行条件(判定基準・水準・前提条件以外)を直して再実行すること。"
    )
else:
    SELECTED_PLAN = AUTO_PLAN
if SMOKE_TEST:  # 参考: このデバイスの計測値で、本番水準(T = 1024/512/256、シード数 5)の 12 通りを見積もる(T4 の値ではない。本番では印字しない)
    print(f"[参考] {device.type} の計測値による本番水準の計画の見積もり(T4 の値ではない。経過時間を含まない総和):")
    for _p in build_plans(LEVELS["prod"]["STEP_CANDIDATES"]):
        _e = estimate_plan(_p, "prod")
        print(
            f"  計画 {_p['plan']:>2}(T = {_p['num_steps']}、段階 {_p['stage']}): 較正 {_e['calibration'] / 60:.1f} 分 + 本番 {_e['main'] / 60:.1f} 分 + 余裕 -> {_e['total'] / 60:.1f} 分"
            f"({'収まる' if _e['total'] <= SESSION_BUDGET_SECONDS else '超える'})"
        )
NUM_STEPS = SELECTED_PLAN["num_steps"]
STAGE = SELECTED_PLAN["stage"]
CALIBRATION_MODE = STAGES[CURRENT_LEVEL_NAME][STAGE]["mode"]
NUM_SEEDS = dict(STAGES[CURRENT_LEVEL_NAME][STAGE]["seeds"])
EVAL_STEPS = eval_steps_for(NUM_STEPS)
SEEDS_A = min(NUM_SEEDS["C1"], NUM_SEEDS["C2"])
SEEDS_B = min(NUM_SEEDS["C2"], NUM_SEEDS["C3"])
SEEDS_C = min(NUM_SEEDS["C2"], NUM_SEEDS["C4"])
SEEDS_D = min(NUM_SEEDS["C2"], NUM_SEEDS["C5"])
print(
    f"選ばれた計画: {SELECTED_PLAN['plan']}(T = {NUM_STEPS}、較正の方式 {CALIBRATION_MODE!r}、段階 {STAGE})、条件ごとのシード数 {json.dumps(NUM_SEEDS)}、"
    f"実験 A・B・C・D のシード数 {SEEDS_A}・{SEEDS_B}・{SEEDS_C}・{SEEDS_D}、途中の評価のステップ {EVAL_STEPS}、見積もり {PLAN_ESTIMATES[SELECTED_PLAN['plan']]['total'] / 60:.1f} 分"
)
print(
    f"学習のエポック数(組を作れる学習用の記事 {TRAIN_TOKENS_PAIRABLE:,} トークンに対する、1 ステップ {BATCH_SIZE * (QUERY_LENGTH + PASSAGE_LENGTH):,} トークン): "
    f"{NUM_STEPS * BATCH_SIZE * (QUERY_LENGTH + PASSAGE_LENGTH) / TRAIN_TOKENS_PAIRABLE:.2f}"
)
```

    経過時間 8.4 分、予算 120 分
      計画  0(T = 1024、段階 0: 較正 'all'、シード数 {'C1': 5, 'C2': 5, 'C3': 5, 'C4': 5, 'C5': 5}): 較正 26.4 分 + 本番 36.0 分 + 余裕 -> 残り 65.0 分、終了の見込み 73.3 分(収まる)
      計画  1(T = 1024、段階 1: 較正 'representative'、シード数 {'C1': 5, 'C2': 5, 'C3': 5, 'C4': 5, 'C5': 5}): 較正 10.7 分 + 本番 36.0 分 + 余裕 -> 残り 49.2 分、終了の見込み 57.6 分(収まる)
      計画  2(T = 1024、段階 2: 較正 'representative'、シード数 {'C1': 5, 'C2': 5, 'C3': 5, 'C4': 5, 'C5': 3}): 較正 10.7 分 + 本番 33.1 分 + 余裕 -> 残り 46.3 分、終了の見込み 54.7 分(収まる)
      計画  3(T = 1024、段階 3: 較正 'representative'、シード数 {'C1': 5, 'C2': 5, 'C3': 5, 'C4': 3, 'C5': 3}): 較正 10.7 分 + 本番 30.2 分 + 余裕 -> 残り 43.4 分、終了の見込み 51.8 分(収まる)
      計画  4(T = 512、段階 0: 較正 'all'、シード数 {'C1': 5, 'C2': 5, 'C3': 5, 'C4': 5, 'C5': 5}): 較正 13.7 分 + 本番 20.0 分 + 余裕 -> 残り 36.2 分、終了の見込み 44.6 分(収まる)
      計画  5(T = 512、段階 1: 較正 'representative'、シード数 {'C1': 5, 'C2': 5, 'C3': 5, 'C4': 5, 'C5': 5}): 較正 5.5 分 + 本番 20.0 分 + 余裕 -> 残り 28.1 分、終了の見込み 36.4 分(収まる)
      計画  6(T = 512、段階 2: 較正 'representative'、シード数 {'C1': 5, 'C2': 5, 'C3': 5, 'C4': 5, 'C5': 3}): 較正 5.5 分 + 本番 18.5 分 + 余裕 -> 残り 26.5 分、終了の見込み 34.9 分(収まる)
      計画  7(T = 512、段階 3: 較正 'representative'、シード数 {'C1': 5, 'C2': 5, 'C3': 5, 'C4': 3, 'C5': 3}): 較正 5.5 分 + 本番 16.8 分 + 余裕 -> 残り 24.8 分、終了の見込み 33.2 分(収まる)
      計画  8(T = 256、段階 0: 較正 'all'、シード数 {'C1': 5, 'C2': 5, 'C3': 5, 'C4': 5, 'C5': 5}): 較正 7.3 分 + 本番 12.0 分 + 余裕 -> 残り 21.8 分、終了の見込み 30.2 分(収まる)
      計画  9(T = 256、段階 1: 較正 'representative'、シード数 {'C1': 5, 'C2': 5, 'C3': 5, 'C4': 5, 'C5': 5}): 較正 2.9 分 + 本番 12.0 分 + 余裕 -> 残り 17.5 分、終了の見込み 25.8 分(収まる)
      計画 10(T = 256、段階 2: 較正 'representative'、シード数 {'C1': 5, 'C2': 5, 'C3': 5, 'C4': 5, 'C5': 3}): 較正 2.9 分 + 本番 11.1 分 + 余裕 -> 残り 16.6 分、終了の見込み 24.9 分(収まる)
      計画 11(T = 256、段階 3: 較正 'representative'、シード数 {'C1': 5, 'C2': 5, 'C3': 5, 'C4': 3, 'C5': 3}): 較正 2.9 分 + 本番 10.1 分 + 余裕 -> 残り 15.6 分、終了の見込み 23.9 分(収まる)
    選ばれた計画: 0(T = 1024、較正の方式 'all'、段階 0)、条件ごとのシード数 {"C1": 5, "C2": 5, "C3": 5, "C4": 5, "C5": 5}、実験 A・B・C・D のシード数 5・5・5・5、途中の評価のステップ (16, 32, 64, 128, 256, 512, 1024)、見積もり 65.0 分
    学習のエポック数(組を作れる学習用の記事 141,126,568 トークンに対する、1 ステップ 10,240 トークン): 0.07


### 6.5 学習率の較正

選ばれた計画の $T$・較正の方式で、6.1 節の規則のとおりに学習率を決める。指標は **検証用の集合** の Recall@10(学習の最終ステップの重み、較正専用のシード)である。
**評価用の集合は較正に使わない。** 各格子点について、検証用の集合の Recall@10(指標)と最後の区間の訓練損失を印字する。

あわせて、前提条件 P1(b)(6.1 節)の基準として、**学習前の 008(C2 と同じ構成: 平均プール・因果マスク)の検証用の集合の Recall@10** を測る(本番と同じ集合・同じ手順で、学習は行わない)。


```python
_t0_calibration = time.time()
# P1(b)の基準: 学習前の 008(C2 の構成)の、検証用の集合の Recall@10(学習のない評価。シードによらない)
_untrained_model = build_model("C2", 0).to(device)
UNTRAINED_VALIDATION_RECALL = evaluate_set(_untrained_model, "validation")["by_dimension"][EMBEDDING_DIMENSION].recall_at(10)
del _untrained_model
empty_device_cache()
print(f"学習前の 008(C2 の構成)の検証用の集合の Recall@10: {UNTRAINED_VALIDATION_RECALL:.4f}(P1(b)の基準。ランダムな順位の期待値 {RANDOM_RECALL['validation']:.4f})")
CALIBRATION = {}  # 条件 -> {"grid", "values", "chosen", "interior", "extended"}


def calibrate(condition: str) -> dict:
    grid = list(learning_rate_grid_for(condition))
    records = {}

    def value(learning_rate: float) -> float:
        if learning_rate not in records:
            records[learning_rate] = train_run(condition, CALIBRATION_SEED_INDEX, learning_rate, NUM_STEPS, main=False)
        record = records[learning_rate]
        return record["validation_recall"] if record["finite"] else -math.inf

    def best(points: list[float]) -> int:
        values = [value(learning_rate) for learning_rate in points]
        return max(range(len(points)), key=lambda i: (values[i], -points[i]))  # 同点なら小さい学習率

    index = best(grid)
    extended = None
    if index == 0:  # 最良が端: その方向に公比 4 で 1 点だけ拡張する(1 回のみ)
        extended = "lower"
        grid = [grid[0] / LEARNING_RATE_GRID_RATIO] + grid
    elif index == len(grid) - 1:
        extended = "upper"
        grid = grid + [grid[-1] * LEARNING_RATE_GRID_RATIO]
    index = best(grid)
    return {
        "grid": grid,
        "values": [value(learning_rate) for learning_rate in grid],
        "chosen": grid[index],
        "interior": 0 < index < len(grid) - 1,
        "extended": extended,
        "final_train_loss": [records[learning_rate]["final_train_loss"] for learning_rate in grid],
    }


for _c in calibration_targets(CALIBRATION_MODE, NUM_SEEDS):
    CALIBRATION[_c] = calibrate(_c)
    _r = CALIBRATION[_c]
    print(
        f"較正 {_c}: 学習率 {[float(f'{x:.3g}') for x in _r['grid']]} -> 検証用の集合の Recall@10 {rounded(_r['values'])}、"
        f"最後の区間の訓練損失 {rounded(_r['final_train_loss'], 3)}、拡張 {_r['extended'] or 'なし'}、選んだ学習率 {_r['chosen']:.3g}"
        f"({'内点' if _r['interior'] else '端'})"
    )

# 学習率の割り当て: 較正した条件はその値。規則で決める条件は、基にする条件の較正で選ばれた値 x(その条件の格子の中心 / 基にする条件の格子の中心)(6.1 節)
LEARNING_RATE, LEARNING_RATE_SOURCE, GRID_MULTIPLIER = {}, {}, {}
for _c in CONDITIONS:
    _source = _c if _c in CALIBRATION else REPRESENTATIVE_RULE[_c]
    if _source == _c:
        LEARNING_RATE[_c] = CALIBRATION[_c]["chosen"]
    else:
        LEARNING_RATE[_c] = CALIBRATION[_source]["chosen"] * (LEARNING_RATE_CENTER[_c] / LEARNING_RATE_CENTER[_source])
    LEARNING_RATE_SOURCE[_c] = _source
    GRID_MULTIPLIER[_c] = LEARNING_RATE[_c] / LEARNING_RATE_CENTER[_c]  # 自分の格子の中心の何倍か(基にする条件の格子での位置と同じ)
# P0(較正): 各実験の対比量に入る条件のうち、自分自身を較正した条件で、選んだ学習率が内点であること
P0_TARGETS = {"A": ("C1", "C2"), "B": ("C2", "C3"), "C": ("C2", "C4"), "D": ("C2", "C5")}
P0_DETAIL = {}
for _experiment, _conditions in P0_TARGETS.items():
    _calibrated = [c for c in _conditions if LEARNING_RATE_SOURCE[c] == c]
    P0_DETAIL[_experiment] = {
        "calibrated": {c: CALIBRATION[c]["interior"] for c in _calibrated},
        "by_rule": [c for c in _conditions if LEARNING_RATE_SOURCE[c] != c],  # 規則で決めた条件(P0 の対象外)
    }
    precondition_status[f"P0({_experiment})"] = all(P0_DETAIL[_experiment]["calibrated"].values())
CALIBRATION_SECONDS = time.time() - _t0_calibration
print(
    f"学習率: {json.dumps({c: float(f'{v:.3g}') for c, v in LEARNING_RATE.items()})}、値の出どころ {json.dumps(LEARNING_RATE_SOURCE)}、"
    f"自分の格子の中心に対する倍率 {json.dumps({c: float(f'{v:.3g}') for c, v in GRID_MULTIPLIER.items()})}"
)
print(
    f"{RUN_TAG}P0(較正): "
    + "、".join(
        f"実験 {e}: {precondition_status[f'P0({e})']}(較正した条件の内点 {json.dumps(d['calibrated'])}、規則で決めた条件(対象外){d['by_rule'] or 'なし'})"
        for e, d in P0_DETAIL.items()
    )
)
print(f"較正 {CALIBRATION_SECONDS / 60:.1f} 分(見積もり {PLAN_ESTIMATES[SELECTED_PLAN['plan']]['calibration'] / 60:.1f} 分、最悪の場合)")
```

    学習前の 008(C2 の構成)の検証用の集合の Recall@10: 0.6919(P1(b)の基準。ランダムな順位の期待値 0.0962)
    較正 C2: 学習率 [0.00012, 0.00048, 0.00192] -> 検証用の集合の Recall@10 [0.8135, 0.8449, 0.7831]、最後の区間の訓練損失 [2.332, 2.055, 2.253]、拡張 なし、選んだ学習率 0.00048(内点)
    較正 C1: 学習率 [6e-05, 0.00024, 0.00096] -> 検証用の集合の Recall@10 [0.7458, 0.7713, 0.7596]、最後の区間の訓練損失 [2.704, 2.359, 2.242]、拡張 なし、選んだ学習率 0.00024(内点)
    較正 C3: 学習率 [6e-05, 0.00024, 0.00096] -> 検証用の集合の Recall@10 [0.8008, 0.843, 0.8175]、最後の区間の訓練損失 [2.379, 2.059, 1.972]、拡張 なし、選んだ学習率 0.00024(内点)
    較正 C5: 学習率 [0.00012, 0.00048, 0.00192] -> 検証用の集合の Recall@10 [0.7929, 0.8234, 0.7772]、最後の区間の訓練損失 [2.467, 2.166, 2.315]、拡張 なし、選んだ学習率 0.00048(内点)
    較正 C4: 学習率 [0.00015, 0.0006, 0.0024] -> 検証用の集合の Recall@10 [0.5466, 0.6546, 0.5486]、最後の区間の訓練損失 [3.173, 2.682, 2.903]、拡張 なし、選んだ学習率 0.0006(内点)
    学習率: {"C1": 0.00024, "C2": 0.00048, "C3": 0.00024, "C4": 0.0006, "C5": 0.00048}、値の出どころ {"C1": "C1", "C2": "C2", "C3": "C3", "C4": "C4", "C5": "C5"}、自分の格子の中心に対する倍率 {"C1": 1.0, "C2": 1.0, "C3": 1.0, "C4": 1.0, "C5": 1.0}
    P0(較正): 実験 A: True(較正した条件の内点 {"C1": true, "C2": true}、規則で決めた条件(対象外)なし)、実験 B: True(較正した条件の内点 {"C2": true, "C3": true}、規則で決めた条件(対象外)なし)、実験 C: True(較正した条件の内点 {"C2": true, "C4": true}、規則で決めた条件(対象外)なし)、実験 D: True(較正した条件の内点 {"C2": true, "C5": true}、規則で決めた条件(対象外)なし)
    較正 20.0 分(見積もり 26.4 分、最悪の場合)


### 6.6 本番の学習と評価

選ばれた計画の条件とシードをすべて学習する。C2 は実験 A・B・C・D で共有し、学習は 1 回だけ行う。評価は評価用の集合で、最終ステップの重みについて行う(切り詰める次元ごと・query ごとの順位を記録する)。
追跡する条件(C2・C3・C4)は、検証用の集合の Recall@10 を途中の評価の位置ごとに記録する。


```python
_t0_main = time.time()
RUNS = {}  # (条件, シード) -> 記録
for _c in RUN_ORDER:
    for _s in range(NUM_SEEDS[_c]):
        RUNS[(_c, _s)] = train_run(_c, _s, LEARNING_RATE[_c], NUM_STEPS, main=True)
        _r = RUNS[(_c, _s)]
        print(
            f"{_c} シード {_s}(学習率 {_r['learning_rate']:.3g}): 評価用 Recall@10 {_r['hits'][EMBEDDING_DIMENSION].mean():.4f}(検証用 {_r['validation_recall']:.4f}、"
            f"学習用の部分 {_r['train_probe_recall']:.4f})、最後の区間の訓練損失 {_r['final_train_loss']:.3f}(ln N の {_r['final_train_loss'] / UNIFORM_LOSS:.3f} 倍)、"
            f"同じ組での学習前のモデルの損失 {_r['initial_model_loss_on_final_window']:.3f}(比 {_r['final_train_loss'] / _r['initial_model_loss_on_final_window']:.3f})、"
            f"clipping の発動 {_r['clip_rate']:.2f}、更新のスキップ {_r['skipped_steps']}、{_r['seconds']:.0f} 秒"
        )
MAIN_SECONDS = time.time() - _t0_main
print(f"本番の学習と評価 {MAIN_SECONDS / 60:.1f} 分(見積もり {PLAN_ESTIMATES[SELECTED_PLAN['plan']]['main'] / 60:.1f} 分)")


def check_learning(conditions: tuple[str, ...], seeds: list[int]) -> dict:
    # 前提条件 P1(6.1 節)。(a)全条件・全シードの訓練損失が有限で、008 の重みから始める条件では、最後の区間の訓練損失が同じ組での学習前のモデルの損失の P1_LOSS_RATIO 倍以下。
    # (b)C2 の検証用の Recall@10(最終ステップの重み)が、学習前の 008(C2 の構成)の値より P1_RECALL_GAIN 以上高い(全シード)
    pretrained = [c for c in conditions if CONDITIONS[c]["init"] == "pretrained"]
    loss_ok = {(c, s): learning_precondition(RUNS[(c, s)], strict=c in pretrained) for c in conditions for s in seeds}
    ratios = {(c, s): RUNS[(c, s)]["final_train_loss"] / RUNS[(c, s)]["initial_model_loss_on_final_window"] for c in pretrained for s in seeds}
    gains = np.array([RUNS[("C2", s)]["validation_recall"] for s in seeds]) - UNTRAINED_VALIDATION_RECALL
    gain_ok = bool((gains >= P1_RECALL_GAIN).all())
    return {"ok": all(loss_ok.values()) and gain_ok, "loss_failed": [k for k, v in loss_ok.items() if not v], "ratio_max": max(ratios.values()), "gains": gains, "gain_ok": gain_ok}


def describe_learning(result: dict) -> str:
    return (
        f"(損失の不成立 {result['loss_failed'] or 'なし'}、最後の区間の訓練損失 / 同じ組での学習前のモデルの損失の最大 {result['ratio_max']:.3f} <= {P1_LOSS_RATIO}。"
        f"C2 の検証用の Recall@10 の学習前からの増分の最小 {result['gains'].min():+.4f} >= {P1_RECALL_GAIN}: {result['gain_ok']})"
    )


def recall_of(condition: str, seeds: int, dimension: int = EMBEDDING_DIMENSION) -> np.ndarray:
    return np.array([RUNS[(condition, s)]["hits"][dimension].mean() for s in range(seeds)])


def hits_matrix(keys: list[tuple[str, int]], dimension: int = EMBEDDING_DIMENSION) -> np.ndarray:
    return np.stack([RUNS[k]["hits"][dimension] for k in keys]).astype(np.float64)


def bootstrap_recall(keys: list[tuple[str, int]], dimension: int = EMBEDDING_DIMENSION) -> np.ndarray:
    # 記事をクラスタとして復元抽出する対応付きブートストラップ(全行に同じ再標本)。形状 (反復, 行)
    numerators = hits_matrix(keys, dimension)
    return paired_cluster_bootstrap_ratio_of_sums(
        numerators, np.ones(numerators.shape[1]), EVALUATION_SET.query_articles, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
    )


# ブートストラップの関数の確認: 再標本の分布が、記録した Recall@10 を中心にしていて、標準偏差が正で有限である
_key = OBSERVATION_RUN
_observed = float(RUNS[_key]["hits"][EMBEDDING_DIMENSION].mean())
_resampled = bootstrap_recall([_key])[:, 0]
assert _resampled.shape == (BOOTSTRAP_RESAMPLES,) and 0.0 < _resampled.std() < 0.5
assert abs(float(_resampled.mean()) - _observed) < _resampled.std(), (float(_resampled.mean()), _observed, float(_resampled.std()))
assert math.isclose(float((RUNS[_key]["rank"][EMBEDDING_DIMENSION] <= 10).mean()), _observed, rel_tol=1e-12)
print(
    f"ブートストラップ(記事を単位): {_key} の Recall@10 {_observed:.4f} に対し、再標本 {BOOTSTRAP_RESAMPLES:,} 回の平均 {_resampled.mean():.4f}・標準偏差 {_resampled.std(ddof=1):.4f}: OK"
)
```

    C2 シード 0(学習率 0.00048): 評価用 Recall@10 0.6692(検証用 0.8381、学習用の部分 0.7740)、最後の区間の訓練損失 2.121(ln N の 0.510 倍)、同じ組での学習前のモデルの損失 3.357(比 0.632)、clipping の発動 1.00、更新のスキップ 0、89 秒
    C2 シード 1(学習率 0.00048): 評価用 Recall@10 0.6665(検証用 0.8292、学習用の部分 0.7927)、最後の区間の訓練損失 2.014(ln N の 0.484 倍)、同じ組での学習前のモデルの損失 3.285(比 0.613)、clipping の発動 1.00、更新のスキップ 0、89 秒
    C2 シード 2(学習率 0.00048): 評価用 Recall@10 0.6714(検証用 0.8391、学習用の部分 0.7903)、最後の区間の訓練損失 2.066(ln N の 0.497 倍)、同じ組での学習前のモデルの損失 3.317(比 0.623)、clipping の発動 1.00、更新のスキップ 0、89 秒
    C2 シード 3(学習率 0.00048): 評価用 Recall@10 0.6723(検証用 0.8440、学習用の部分 0.7919)、最後の区間の訓練損失 2.062(ln N の 0.496 倍)、同じ組での学習前のモデルの損失 3.319(比 0.621)、clipping の発動 1.00、更新のスキップ 0、89 秒
    C2 シード 4(学習率 0.00048): 評価用 Recall@10 0.6708(検証用 0.8518、学習用の部分 0.7997)、最後の区間の訓練損失 2.049(ln N の 0.493 倍)、同じ組での学習前のモデルの損失 3.274(比 0.626)、clipping の発動 1.00、更新のスキップ 0、89 秒
    C1 シード 0(学習率 0.00024): 評価用 Recall@10 0.5712(検証用 0.7655、学習用の部分 0.7108)、最後の区間の訓練損失 2.393(ln N の 0.575 倍)、同じ組での学習前のモデルの損失 6.387(比 0.375)、clipping の発動 1.00、更新のスキップ 0、82 秒
    C1 シード 1(学習率 0.00024): 評価用 Recall@10 0.5789(検証用 0.7831、学習用の部分 0.7108)、最後の区間の訓練損失 2.297(ln N の 0.552 倍)、同じ組での学習前のモデルの損失 6.326(比 0.363)、clipping の発動 1.00、更新のスキップ 0、82 秒
    C1 シード 2(学習率 0.00024): 評価用 Recall@10 0.5660(検証用 0.7763、学習用の部分 0.7186)、最後の区間の訓練損失 2.339(ln N の 0.562 倍)、同じ組での学習前のモデルの損失 6.255(比 0.374)、clipping の発動 1.00、更新のスキップ 0、82 秒
    C1 シード 3(学習率 0.00024): 評価用 Recall@10 0.5647(検証用 0.7684、学習用の部分 0.7217)、最後の区間の訓練損失 2.349(ln N の 0.565 倍)、同じ組での学習前のモデルの損失 6.316(比 0.372)、clipping の発動 1.00、更新のスキップ 0、82 秒
    C1 シード 4(学習率 0.00024): 評価用 Recall@10 0.5700(検証用 0.7713、学習用の部分 0.7116)、最後の区間の訓練損失 2.342(ln N の 0.563 倍)、同じ組での学習前のモデルの損失 6.384(比 0.367)、clipping の発動 1.00、更新のスキップ 0、82 秒
    C3 シード 0(学習率 0.00024): 評価用 Recall@10 0.6773(検証用 0.8322、学習用の部分 0.8020)、最後の区間の訓練損失 2.099(ln N の 0.505 倍)、同じ組での学習前のモデルの損失 3.469(比 0.605)、clipping の発動 1.00、更新のスキップ 0、83 秒
    C3 シード 1(学習率 0.00024): 評価用 Recall@10 0.6769(検証用 0.8292、学習用の部分 0.7701)、最後の区間の訓練損失 2.004(ln N の 0.482 倍)、同じ組での学習前のモデルの損失 3.332(比 0.601)、clipping の発動 1.00、更新のスキップ 0、83 秒
    C3 シード 2(学習率 0.00024): 評価用 Recall@10 0.6763(検証用 0.8204、学習用の部分 0.7903)、最後の区間の訓練損失 2.030(ln N の 0.488 倍)、同じ組での学習前のモデルの損失 3.374(比 0.602)、clipping の発動 1.00、更新のスキップ 0、83 秒
    C3 シード 3(学習率 0.00024): 評価用 Recall@10 0.6877(検証用 0.8400、学習用の部分 0.8036)、最後の区間の訓練損失 2.039(ln N の 0.490 倍)、同じ組での学習前のモデルの損失 3.435(比 0.594)、clipping の発動 1.00、更新のスキップ 0、83 秒
    C3 シード 4(学習率 0.00024): 評価用 Recall@10 0.6711(検証用 0.8283、学習用の部分 0.7989)、最後の区間の訓練損失 2.050(ln N の 0.493 倍)、同じ組での学習前のモデルの損失 3.381(比 0.606)、clipping の発動 1.00、更新のスキップ 0、83 秒
    C5 シード 0(学習率 0.00048): 評価用 Recall@10 0.6477(検証用 0.8204、学習用の部分 0.7545)、最後の区間の訓練損失 2.232(ln N の 0.537 倍)、同じ組での学習前のモデルの損失 4.059(比 0.550)、clipping の発動 1.00、更新のスキップ 0、87 秒
    C5 シード 1(学習率 0.00048): 評価用 Recall@10 0.6495(検証用 0.8292、学習用の部分 0.7599)、最後の区間の訓練損失 2.149(ln N の 0.517 倍)、同じ組での学習前のモデルの損失 4.015(比 0.535)、clipping の発動 1.00、更新のスキップ 0、87 秒
    C5 シード 2(学習率 0.00048): 評価用 Recall@10 0.6486(検証用 0.8263、学習用の部分 0.7529)、最後の区間の訓練損失 2.177(ln N の 0.523 倍)、同じ組での学習前のモデルの損失 4.015(比 0.542)、clipping の発動 1.00、更新のスキップ 0、87 秒
    C5 シード 3(学習率 0.00048): 評価用 Recall@10 0.6563(検証用 0.8361、学習用の部分 0.7623)、最後の区間の訓練損失 2.197(ln N の 0.528 倍)、同じ組での学習前のモデルの損失 4.028(比 0.545)、clipping の発動 1.00、更新のスキップ 0、87 秒
    C5 シード 4(学習率 0.00048): 評価用 Recall@10 0.6517(検証用 0.8351、学習用の部分 0.7654)、最後の区間の訓練損失 2.165(ln N の 0.521 倍)、同じ組での学習前のモデルの損失 3.978(比 0.544)、clipping の発動 1.00、更新のスキップ 0、87 秒
    C4 シード 0(学習率 0.0006): 評価用 Recall@10 0.4556(検証用 0.6673、学習用の部分 0.6111)、最後の区間の訓練損失 2.727(ln N の 0.656 倍)、同じ組での学習前のモデルの損失 7.205(比 0.378)、clipping の発動 1.00、更新のスキップ 0、87 秒
    C4 シード 1(学習率 0.0006): 評価用 Recall@10 0.4630(検証用 0.6889、学習用の部分 0.5994)、最後の区間の訓練損失 2.669(ln N の 0.642 倍)、同じ組での学習前のモデルの損失 7.078(比 0.377)、clipping の発動 1.00、更新のスキップ 0、87 秒
    C4 シード 2(学習率 0.0006): 評価用 Recall@10 0.4698(検証用 0.6605、学習用の部分 0.6267)、最後の区間の訓練損失 2.691(ln N の 0.647 倍)、同じ組での学習前のモデルの損失 7.261(比 0.371)、clipping の発動 1.00、更新のスキップ 0、87 秒
    C4 シード 3(学習率 0.0006): 評価用 Recall@10 0.4516(検証用 0.6801、学習用の部分 0.6080)、最後の区間の訓練損失 2.742(ln N の 0.659 倍)、同じ組での学習前のモデルの損失 7.354(比 0.373)、clipping の発動 1.00、更新のスキップ 0、87 秒
    C4 シード 4(学習率 0.0006): 評価用 Recall@10 0.4738(検証用 0.6791、学習用の部分 0.6220)、最後の区間の訓練損失 2.695(ln N の 0.648 倍)、同じ組での学習前のモデルの損失 7.113(比 0.379)、clipping の発動 1.00、更新のスキップ 0、87 秒
    本番の学習と評価 35.7 分(見積もり 36.0 分)
    ブートストラップ(記事を単位): ('C2', 0) の Recall@10 0.6692 に対し、再標本 10,000 回の平均 0.6694・標準偏差 0.0124: OK


### 6.7 実験 A: 因果マスクでのプーリング(終端位置 と 平均)


```python
_seeds = list(range(SEEDS_A))
R_C1, R_C2_A = recall_of("C1", SEEDS_A), recall_of("C2", SEEDS_A)
A_PER_SEED = R_C1 - R_C2_A
_boot = bootstrap_recall([("C1", s) for s in _seeds] + [("C2", s) for s in _seeds])
_boot_contrast = _boot[:, :SEEDS_A].mean(axis=1) - _boot[:, SEEDS_A:].mean(axis=1)
DELTA_A = float(A_PER_SEED.mean())
SIGMA_A = combined_sigma(A_PER_SEED, _boot_contrast)

_p1 = check_learning(("C1", "C2"), _seeds)
precondition_status["P1(A)"] = _p1["ok"]
precondition_status["P2(A)"] = bool(R_C2_A.mean() <= P2_RECALL_CEILING)
A_PRECONDITIONS = ["P0(A)", "P1(A)", "P2(A)"]
A_COMPUTED = judge(DELTA_A, SIGMA_A["sigma"])
A_VERDICT = A_COMPUTED if all(precondition_status[k] for k in A_PRECONDITIONS) else "前提不成立"

print(f"{RUN_TAG}実験 A(シード {_seeds}、T = {NUM_STEPS})")
print(f"  Recall@10 R(C1): {rounded(R_C1)}、R(C2): {rounded(R_C2_A)}")
print(f"  d_s = R(C1) - R(C2): {rounded(A_PER_SEED)}")
print(
    f"  Delta_A = {DELTA_A:+.4f}、sigma_A = {SIGMA_A['sigma']:.4f}(シード間 {SIGMA_A['seed_term']:.4f}・ブートストラップ "
    f"{SIGMA_A['bootstrap_term']:.4f}、反復 {BOOTSTRAP_RESAMPLES:,})、閾値 2 sigma_A = {SIGMA_MULTIPLIER * SIGMA_A['sigma']:.4f}"
)
print(
    f"  前提条件: P0 {precondition_status['P0(A)']}、P1 {precondition_status['P1(A)']}{describe_learning(_p1)}、"
    f"P2 {precondition_status['P2(A)']}(R(C2) の平均 {R_C2_A.mean():.4f} <= {P2_RECALL_CEILING})"
)
print(f"  判定関数の結果 {A_COMPUTED} -> {RUN_TAG}最終判定: {A_VERDICT}")

# 診断量(判定なし)
print("  診断量: 作用点に近い量。C2 のモデルで、位置 t の出力だけを埋め込みにしたときの Recall@10(学習を伴わない評価。query の位置 / passage の位置は系列の長さに対する割合から決める)")
_position_means = {}
for _fraction, _tq, _tp in zip(POSITION_FRACTIONS, POSITIONS_QUERY, POSITIONS_PASSAGE, strict=True):
    _values = np.array([RUNS[("C2", s)]["position_hits"][_fraction].mean() for s in range(NUM_SEEDS["C2"])])
    _position_means[_fraction] = (float(_values.mean()), float(_values.std(ddof=1)))
    print(f"    割合 {_fraction:.4f}(query の位置 {_tq}、passage の位置 {_tp}): {_values.mean():.4f}(シード間の標準偏差 {_values.std(ddof=1):.4f})")
print(f"    参照: C2 の平均プール {recall_of('C2', NUM_SEEDS['C2']).mean():.4f}、C1 の終端位置 {recall_of('C1', NUM_SEEDS['C1']).mean():.4f}")
for _c in ("C1", "C2"):
    _mrr = np.array([(1 / RUNS[(_c, s)]["rank"][EMBEDDING_DIMENSION]).mean() for s in range(NUM_SEEDS[_c])])
    _ndcg = np.array([RUNS[(_c, s)]["ndcg_at_10"].mean() for s in range(NUM_SEEDS[_c])])
    _loss = np.array([RUNS[(_c, s)]["final_train_loss"] for s in range(NUM_SEEDS[_c])])
    print(
        f"  診断量: {_c} の MRR {_mrr.mean():.4f}・nDCG@10 {_ndcg.mean():.4f}・最後の区間の訓練損失 {_loss.mean():.3f}"
        f"・学習率 {LEARNING_RATE[_c]:.3g}(出どころ {LEARNING_RATE_SOURCE[_c]})・clipping の発動 {np.mean([RUNS[(_c, s)]['clip_rate'] for s in range(NUM_SEEDS[_c])]):.2f}"
    )

_fig, _axes = plt.subplots(1, 2, figsize=(11, 3.6))
for _c, _marker in (("C1", "o"), ("C2", "s")):
    _axes[0].plot(range(NUM_SEEDS[_c]), recall_of(_c, NUM_SEEDS[_c]), _marker, label=_c)
_axes[0].axhline(RANDOM_RECALL["evaluation"], color="gray", linestyle=":", label="random ranking")
_axes[0].set_xlabel("seed")
_axes[0].set_ylabel("Recall@10 (evaluation set)")
_axes[0].set_title(f"{PLOT_TAG}Experiment A: Recall@10 per seed")
_axes[0].legend()
_x = list(_position_means)
_axes[1].errorbar(_x, [_position_means[f][0] for f in _x], yerr=[_position_means[f][1] for f in _x], marker="o", capsize=3, label="C2, single position")
_axes[1].axhline(recall_of("C2", NUM_SEEDS["C2"]).mean(), color="C1", linestyle="--", label="C2, mean pooling")
_axes[1].set_xscale("log", base=2)
_axes[1].set_xlabel("position as a fraction of the sequence length")
_axes[1].set_ylabel("Recall@10")
_axes[1].set_title(f"{PLOT_TAG}Recall@10 using only the output at one position")
_axes[1].legend()
plt.tight_layout()
plt.show()
```

    実験 A(シード [0, 1, 2, 3, 4]、T = 1024)
      Recall@10 R(C1): [0.5712, 0.5789, 0.566, 0.5647, 0.57]、R(C2): [0.6692, 0.6665, 0.6714, 0.6723, 0.6708]
      d_s = R(C1) - R(C2): [-0.098, -0.0875, -0.1054, -0.1076, -0.1008]
      Delta_A = -0.0999、sigma_A = 0.0075(シード間 0.0035・ブートストラップ 0.0066、反復 10,000)、閾値 2 sigma_A = 0.0150
      前提条件: P0 True、P1 True(損失の不成立 なし、最後の区間の訓練損失 / 同じ組での学習前のモデルの損失の最大 0.632 <= 0.9。C2 の検証用の Recall@10 の学習前からの増分の最小 +0.1374 >= 0.04: True)、P2 True(R(C2) の平均 0.6700 <= 0.95)
      判定関数の結果 反証 -> 最終判定: 反証
      診断量: 作用点に近い量。C2 のモデルで、位置 t の出力だけを埋め込みにしたときの Recall@10(学習を伴わない評価。query の位置 / passage の位置は系列の長さに対する割合から決める)
        割合 0.0312(query の位置 0、passage の位置 3): 0.0809(シード間の標準偏差 0.0019)
        割合 0.0625(query の位置 1、passage の位置 7): 0.0999(シード間の標準偏差 0.0024)
        割合 0.1250(query の位置 3、passage の位置 15): 0.1233(シード間の標準偏差 0.0026)
        割合 0.2500(query の位置 7、passage の位置 31): 0.1387(シード間の標準偏差 0.0021)
        割合 0.5000(query の位置 15、passage の位置 63): 0.1586(シード間の標準偏差 0.0022)
        割合 1.0000(query の位置 31、passage の位置 127): 0.1729(シード間の標準偏差 0.0039)
        参照: C2 の平均プール 0.6700、C1 の終端位置 0.5702
      診断量: C1 の MRR 0.3589・nDCG@10 0.2024・最後の区間の訓練損失 2.344・学習率 0.00024(出どころ C1)・clipping の発動 1.00
      診断量: C2 の MRR 0.4610・nDCG@10 0.2694・最後の区間の訓練損失 2.062・学習率 0.00048(出どころ C2)・clipping の発動 1.00



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/024_text_embedding_and_retriever/output_37_1.png)
    


### 6.8 実験 B: 平均プールでの注意マスク(双方向 と 因果)


```python
_seeds = list(range(SEEDS_B))
R_C2_B, R_C3 = recall_of("C2", SEEDS_B), recall_of("C3", SEEDS_B)
B_PER_SEED = R_C3 - R_C2_B
_boot = bootstrap_recall([("C3", s) for s in _seeds] + [("C2", s) for s in _seeds])
_boot_contrast = _boot[:, :SEEDS_B].mean(axis=1) - _boot[:, SEEDS_B:].mean(axis=1)
DELTA_B = float(B_PER_SEED.mean())
SIGMA_B = combined_sigma(B_PER_SEED, _boot_contrast)

_p1 = check_learning(("C2", "C3"), _seeds)
precondition_status["P1(B)"] = _p1["ok"]
precondition_status["P2(B)"] = bool(R_C2_B.mean() <= P2_RECALL_CEILING)
B_PRECONDITIONS = ["P0(B)", "P1(B)", "P2(B)"]
B_COMPUTED = judge(DELTA_B, SIGMA_B["sigma"])
B_VERDICT = B_COMPUTED if all(precondition_status[k] for k in B_PRECONDITIONS) else "前提不成立"

print(f"{RUN_TAG}実験 B(シード {_seeds}、T = {NUM_STEPS})")
print(f"  Recall@10 R(C3): {rounded(R_C3)}、R(C2): {rounded(R_C2_B)}")
print(f"  d_s = R(C3) - R(C2): {rounded(B_PER_SEED)}")
print(
    f"  Delta_B = {DELTA_B:+.4f}、sigma_B = {SIGMA_B['sigma']:.4f}(シード間 {SIGMA_B['seed_term']:.4f}・ブートストラップ "
    f"{SIGMA_B['bootstrap_term']:.4f}、反復 {BOOTSTRAP_RESAMPLES:,})、閾値 2 sigma_B = {SIGMA_MULTIPLIER * SIGMA_B['sigma']:.4f}"
)
print(
    f"  前提条件: P0 {precondition_status['P0(B)']}、P1 {precondition_status['P1(B)']}{describe_learning(_p1)}、"
    f"P2 {precondition_status['P2(B)']}(R(C2) の平均 {R_C2_B.mean():.4f} <= {P2_RECALL_CEILING})"
)
print(f"  判定関数の結果 {B_COMPUTED} -> {RUN_TAG}最終判定: {B_VERDICT}")

# 診断量(判定なし): 学習の初期に密な位置での検証用の集合の Recall@10(シード平均)
print(f"  診断量: 検証用の集合の Recall@10 の推移(シード平均、途中の評価のステップ {EVAL_STEPS})")
for _c in ("C2", "C3"):
    print(f"    {_c}: {rounded(np.mean([RUNS[(_c, s)]['eval_validation_recall'] for s in range(NUM_SEEDS[_c])], axis=0))}")
for _c in ("C2", "C3"):
    print(
        f"  診断量: {_c} の最後の区間の訓練損失 {np.mean([RUNS[(_c, s)]['final_train_loss'] for s in range(NUM_SEEDS[_c])]):.3f}・"
        f"学習率 {LEARNING_RATE[_c]:.3g}(出どころ {LEARNING_RATE_SOURCE[_c]})・MRR "
        f"{np.mean([(1 / RUNS[(_c, s)]['rank'][EMBEDDING_DIMENSION]).mean() for s in range(NUM_SEEDS[_c])]):.4f}"
    )

_fig, _axes = plt.subplots(1, 2, figsize=(11, 3.6))
for _c, _marker in (("C3", "o"), ("C2", "s")):
    _axes[0].plot(range(NUM_SEEDS[_c]), recall_of(_c, NUM_SEEDS[_c]), _marker, label=_c)
_axes[0].set_xlabel("seed")
_axes[0].set_ylabel("Recall@10 (evaluation set)")
_axes[0].set_title(f"{PLOT_TAG}Experiment B: Recall@10 per seed")
_axes[0].legend()
for _c, _marker in (("C3", "o"), ("C2", "s")):
    _curves = np.array([RUNS[(_c, s)]["eval_validation_recall"] for s in range(NUM_SEEDS[_c])])
    _axes[1].plot(EVAL_STEPS, _curves.mean(axis=0), marker=_marker, label=_c)
_axes[1].set_xscale("log", base=2)
_axes[1].set_xlabel("step")
_axes[1].set_ylabel("Recall@10 (validation set, seed mean)")
_axes[1].set_title(f"{PLOT_TAG}Recall@10 during training")
_axes[1].legend()
plt.tight_layout()
plt.show()
```

    実験 B(シード [0, 1, 2, 3, 4]、T = 1024)
      Recall@10 R(C3): [0.6773, 0.6769, 0.6763, 0.6877, 0.6711]、R(C2): [0.6692, 0.6665, 0.6714, 0.6723, 0.6708]
      d_s = R(C3) - R(C2): [0.008, 0.0105, 0.0049, 0.0154, 0.0003]
      Delta_B = +0.0078、sigma_B = 0.0049(シード間 0.0025・ブートストラップ 0.0042、反復 10,000)、閾値 2 sigma_B = 0.0098
      前提条件: P0 True、P1 True(損失の不成立 なし、最後の区間の訓練損失 / 同じ組での学習前のモデルの損失の最大 0.632 <= 0.9。C2 の検証用の Recall@10 の学習前からの増分の最小 +0.1374 >= 0.04: True)、P2 True(R(C2) の平均 0.6700 <= 0.95)
      判定関数の結果 判定不能 -> 最終判定: 判定不能
      診断量: 検証用の集合の Recall@10 の推移(シード平均、途中の評価のステップ (16, 32, 64, 128, 256, 512, 1024))
        C2: [0.6981, 0.7199, 0.7425, 0.757, 0.7851, 0.8108, 0.8404]
        C3: [0.7172, 0.7327, 0.7558, 0.7678, 0.787, 0.8132, 0.83]
      診断量: C2 の最後の区間の訓練損失 2.062・学習率 0.00048(出どころ C2)・MRR 0.4610
      診断量: C3 の最後の区間の訓練損失 2.044・学習率 0.00024(出どころ C3)・MRR 0.4699



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/024_text_embedding_and_retriever/output_39_1.png)
    


### 6.9 実験 C: 事前学習の効果(008 の重み と ランダム初期化)


```python
_seeds = list(range(SEEDS_C))
R_C2_C, R_C4 = recall_of("C2", SEEDS_C), recall_of("C4", SEEDS_C)
C_PER_SEED = R_C2_C - R_C4
_keys = [("C2", s) for s in _seeds] + [("C4", s) for s in _seeds]
_boot = bootstrap_recall(_keys)
_boot_contrast = _boot[:, :SEEDS_C].mean(axis=1) - _boot[:, SEEDS_C:].mean(axis=1)
DELTA_C = float(C_PER_SEED.mean())
SIGMA_C = combined_sigma(C_PER_SEED, _boot_contrast)

_p1 = check_learning(("C2", "C4"), _seeds)  # C4 は損失が有限であることのみ
precondition_status["P1(C)"] = _p1["ok"]
precondition_status["P2(C)"] = bool(R_C4.mean() <= P2_RECALL_CEILING)
C_PRECONDITIONS = ["P0(C)", "P1(C)", "P2(C)"]
C_COMPUTED = judge(DELTA_C, SIGMA_C["sigma"])
C_VERDICT = C_COMPUTED if all(precondition_status[k] for k in C_PRECONDITIONS) else "前提不成立"

print(f"{RUN_TAG}実験 C(シード {_seeds}、T = {NUM_STEPS})")
print(f"  Recall@10 R(C2): {rounded(R_C2_C)}、R(C4): {rounded(R_C4)}")
print(f"  d_s = R(C2) - R(C4): {rounded(C_PER_SEED)}")
print(
    f"  Delta_C = {DELTA_C:+.4f}、sigma_C = {SIGMA_C['sigma']:.4f}(シード間 {SIGMA_C['seed_term']:.4f}・ブートストラップ "
    f"{SIGMA_C['bootstrap_term']:.4f}、反復 {BOOTSTRAP_RESAMPLES:,})、閾値 2 sigma_C = {SIGMA_MULTIPLIER * SIGMA_C['sigma']:.4f}"
)
print(
    f"  前提条件: P0 {precondition_status['P0(C)']}、P1 {precondition_status['P1(C)']}{describe_learning(_p1)}、"
    f"P2 {precondition_status['P2(C)']}(R(C4) の平均 {R_C4.mean():.4f} <= {P2_RECALL_CEILING})"
)
print(f"  判定関数の結果 {C_COMPUTED} -> {RUN_TAG}最終判定: {C_VERDICT}")

# 診断量(判定なし)
print(f"  診断量: 検証用の集合の Recall@10 の推移(シード平均、途中の評価のステップ {EVAL_STEPS})")
for _c in ("C2", "C4"):
    print(f"    {_c}: {rounded(np.mean([RUNS[(_c, s)]['eval_validation_recall'] for s in range(NUM_SEEDS[_c])], axis=0))}")
print("  診断量: 汎化の差(学習用の部分の索引の Recall@10 - 評価用の Recall@10。別の記事・別の索引の大きさなので、条件間での差の違いを読む)")
_gap = {}
for _c in ("C2", "C4"):
    _train = np.array([RUNS[(_c, s)]["train_probe_recall"] for s in _seeds])
    _gap[_c] = _train - recall_of(_c, SEEDS_C)
    print(f"    {_c}: 学習用の部分 {_train.mean():.4f}、差の平均 {_gap[_c].mean():+.4f}、各シード {rounded(_gap[_c])}")
print(f"    同じシードの C2 との差(C4 の差 - C2 の差)の平均 {(_gap['C4'] - _gap['C2']).mean():+.4f}、各シード {rounded(_gap['C4'] - _gap['C2'])}")
print(
    f"  診断量: 最後の区間の訓練損失 C2 {np.mean([RUNS[('C2', s)]['final_train_loss'] for s in _seeds]):.3f}・C4 {np.mean([RUNS[('C4', s)]['final_train_loss'] for s in _seeds]):.3f}、"
    f"学習率 C2 {LEARNING_RATE['C2']:.3g}・C4 {LEARNING_RATE['C4']:.3g}"
)

_fig, _axes = plt.subplots(1, 2, figsize=(11, 3.6))
for _c, _marker in (("C2", "s"), ("C4", "x")):
    _axes[0].plot(range(NUM_SEEDS[_c]), recall_of(_c, NUM_SEEDS[_c]), _marker, label=_c)
_axes[0].axhline(RANDOM_RECALL["evaluation"], color="gray", linestyle=":", label="random ranking")
_axes[0].set_xlabel("seed")
_axes[0].set_ylabel("Recall@10 (evaluation set)")
_axes[0].set_title(f"{PLOT_TAG}Experiment C: Recall@10 per seed")
_axes[0].legend()
for _c, _marker in (("C2", "s"), ("C4", "x")):
    _curves = np.array([RUNS[(_c, s)]["eval_validation_recall"] for s in range(NUM_SEEDS[_c])])
    _axes[1].plot(EVAL_STEPS, _curves.mean(axis=0), marker=_marker, label=_c)
_axes[1].set_xscale("log", base=2)
_axes[1].set_xlabel("step")
_axes[1].set_ylabel("Recall@10 (validation set, seed mean)")
_axes[1].set_title(f"{PLOT_TAG}Recall@10 during training")
_axes[1].legend()
plt.tight_layout()
plt.show()
```

    実験 C(シード [0, 1, 2, 3, 4]、T = 1024)
      Recall@10 R(C2): [0.6692, 0.6665, 0.6714, 0.6723, 0.6708]、R(C4): [0.4556, 0.463, 0.4698, 0.4516, 0.4738]
      d_s = R(C2) - R(C4): [0.2136, 0.2035, 0.2016, 0.2207, 0.197]
      Delta_C = +0.2073、sigma_C = 0.0092(シード間 0.0043・ブートストラップ 0.0081、反復 10,000)、閾値 2 sigma_C = 0.0183
      前提条件: P0 True、P1 True(損失の不成立 なし、最後の区間の訓練損失 / 同じ組での学習前のモデルの損失の最大 0.632 <= 0.9。C2 の検証用の Recall@10 の学習前からの増分の最小 +0.1374 >= 0.04: True)、P2 True(R(C4) の平均 0.4628 <= 0.95)
      判定関数の結果 支持 -> 最終判定: 支持
      診断量: 検証用の集合の Recall@10 の推移(シード平均、途中の評価のステップ (16, 32, 64, 128, 256, 512, 1024))
        C2: [0.6981, 0.7199, 0.7425, 0.757, 0.7851, 0.8108, 0.8404]
        C4: [0.2012, 0.2092, 0.2279, 0.3034, 0.42, 0.5843, 0.6752]
      診断量: 汎化の差(学習用の部分の索引の Recall@10 - 評価用の Recall@10。別の記事・別の索引の大きさなので、条件間での差の違いを読む)
        C2: 学習用の部分 0.7897、差の平均 +0.1197、各シード [0.1047, 0.1262, 0.1189, 0.1196, 0.1289]
        C4: 学習用の部分 0.6134、差の平均 +0.1506、各シード [0.1555, 0.1364, 0.1569, 0.1563, 0.1482]
        同じシードの C2 との差(C4 の差 - C2 の差)の平均 +0.0310、各シード [0.0507, 0.0102, 0.0379, 0.0368, 0.0193]
      診断量: 最後の区間の訓練損失 C2 2.062・C4 2.705、学習率 C2 0.00048・C4 0.0006



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/024_text_embedding_and_retriever/output_41_1.png)
    


### 6.10 実験 D: Matryoshka Representation Learning(先頭 16 次元)


```python
_seeds = list(range(SEEDS_D))
SMALLEST = MATRYOSHKA_DIMENSIONS[0]  # 16
R16_C2 = recall_of("C2", SEEDS_D, SMALLEST)
R16_C5 = recall_of("C5", SEEDS_D, SMALLEST)
D_PER_SEED = R16_C5 - R16_C2
_keys5 = [("C5", s) for s in _seeds]
_keys2 = [("C2", s) for s in _seeds]
_boot = paired_cluster_bootstrap_ratio_of_sums(
    np.concatenate([hits_matrix(_keys5, SMALLEST), hits_matrix(_keys2, SMALLEST)]), np.ones(EVALUATION_SET.num_queries),
    EVALUATION_SET.query_articles, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED,
)
_boot_contrast = _boot[:, :SEEDS_D].mean(axis=1) - _boot[:, SEEDS_D:].mean(axis=1)
DELTA_D = float(D_PER_SEED.mean())
SIGMA_D = combined_sigma(D_PER_SEED, _boot_contrast)

_p1 = check_learning(("C2", "C5"), _seeds)
precondition_status["P1(D)"] = _p1["ok"]
precondition_status["P2(D)"] = bool(R16_C2.mean() <= P2_RECALL_CEILING)
D_PRECONDITIONS = ["P0(D)", "P1(D)", "P2(D)"]
D_COMPUTED = judge(DELTA_D, SIGMA_D["sigma"])
D_VERDICT = D_COMPUTED if all(precondition_status[k] for k in D_PRECONDITIONS) else "前提不成立"

print(f"{RUN_TAG}実験 D(シード {_seeds}、T = {NUM_STEPS})")
print(f"  先頭 {SMALLEST} 次元の Recall@10 R_{SMALLEST}(C5): {rounded(R16_C5)}、R_{SMALLEST}(C2): {rounded(R16_C2)}")
print(f"  d_s = R_{SMALLEST}(C5) - R_{SMALLEST}(C2): {rounded(D_PER_SEED)}")
print(
    f"  Delta_D = {DELTA_D:+.4f}、sigma_D = {SIGMA_D['sigma']:.4f}(シード間 {SIGMA_D['seed_term']:.4f}・ブートストラップ "
    f"{SIGMA_D['bootstrap_term']:.4f}、反復 {BOOTSTRAP_RESAMPLES:,})、閾値 2 sigma_D = {SIGMA_MULTIPLIER * SIGMA_D['sigma']:.4f}"
)
print(
    f"  前提条件: P0 {precondition_status['P0(D)']}、P1 {precondition_status['P1(D)']}{describe_learning(_p1)}、"
    f"P2 {precondition_status['P2(D)']}(R_{SMALLEST}(C2) の平均 {R16_C2.mean():.4f} <= {P2_RECALL_CEILING})"
)
print(f"  判定関数の結果 {D_COMPUTED} -> {RUN_TAG}最終判定: {D_VERDICT}")

# 診断量(判定なし): 全ての m で、両条件の Recall@10。「全次元では劣化しない」は別の検証事項で、判定に含めない
print("  診断量: 入れ子の次元 m ごとの Recall@10(シード平均 ± シード間の標準偏差)と、同じシードの差 C5 - C2(標準偏差は判定と同じ構成)")
_dimension_table = {}
for _m in MATRYOSHKA_DIMENSIONS:
    _a, _b = recall_of("C5", SEEDS_D, _m), recall_of("C2", SEEDS_D, _m)
    _boot_m = paired_cluster_bootstrap_ratio_of_sums(
        np.concatenate([hits_matrix(_keys5, _m), hits_matrix(_keys2, _m)]), np.ones(EVALUATION_SET.num_queries),
        EVALUATION_SET.query_articles, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED,
    )
    _sigma_m = combined_sigma(_a - _b, _boot_m[:, :SEEDS_D].mean(axis=1) - _boot_m[:, SEEDS_D:].mean(axis=1))
    _dimension_table[_m] = (float(_a.mean()), float(_a.std(ddof=1)), float(_b.mean()), float(_b.std(ddof=1)))
    print(
        f"    m = {_m:>3}: C5 {_a.mean():.4f} ± {_a.std(ddof=1):.4f}、C2 {_b.mean():.4f} ± {_b.std(ddof=1):.4f}、差 {(_a - _b).mean():+.4f}"
        f"(標準偏差 {_sigma_m['sigma']:.4f}、2 倍 {SIGMA_MULTIPLIER * _sigma_m['sigma']:.4f})"
    )
print(
    f"  診断量: C5 の最後の区間の次元ごとの訓練損失 {rounded(np.mean([RUNS[('C5', s)]['final_dimension_losses'] for s in _seeds], axis=0), 3)}"
    f"(m = {MATRYOSHKA_DIMENSIONS})、C5 の最後の区間の訓練損失(平均)の平均 {np.mean([RUNS[('C5', s)]['final_train_loss'] for s in _seeds]):.3f}、"
    f"clipping の発動 C5 {np.mean([RUNS[('C5', s)]['clip_rate'] for s in _seeds]):.2f}・C2 {np.mean([RUNS[('C2', s)]['clip_rate'] for s in _seeds]):.2f}"
)
print(f"  診断量: 索引のメモリ(FP32、評価用の passage {EVALUATION_SET.num_passages} 個): " + "、".join(f"m = {m}: {4 * EVALUATION_SET.num_passages * m / 1024:.0f} KiB" for m in MATRYOSHKA_DIMENSIONS))

_fig, _axis = plt.subplots(figsize=(5.8, 3.8))
for _label, _column, _marker in (("C5 (Matryoshka Representation Learning)", 0, "o"), ("C2 (InfoNCE)", 2, "s")):
    _axis.errorbar(
        [math.log2(m) for m in MATRYOSHKA_DIMENSIONS], [_dimension_table[m][_column] for m in MATRYOSHKA_DIMENSIONS],
        yerr=[_dimension_table[m][_column + 1] for m in MATRYOSHKA_DIMENSIONS], marker=_marker, capsize=3, label=_label,
    )
_axis.set_xticks([math.log2(m) for m in MATRYOSHKA_DIMENSIONS])
_axis.set_xticklabels([str(m) for m in MATRYOSHKA_DIMENSIONS])
_axis.set_xlabel("truncated dimension m")
_axis.set_ylabel("Recall@10 (evaluation set)")
_axis.set_title(f"{PLOT_TAG}Experiment D: Recall@10 vs dimension")
_axis.legend()
plt.tight_layout()
plt.show()
```

    実験 D(シード [0, 1, 2, 3, 4]、T = 1024)
      先頭 16 次元の Recall@10 R_16(C5): [0.5672, 0.5712, 0.5749, 0.5647, 0.5678]、R_16(C2): [0.3782, 0.39, 0.3847, 0.3678, 0.3711]
      d_s = R_16(C5) - R_16(C2): [0.189, 0.1813, 0.1902, 0.197, 0.1967]
      Delta_D = +0.1908、sigma_D = 0.0074(シード間 0.0029・ブートストラップ 0.0068、反復 10,000)、閾値 2 sigma_D = 0.0147
      前提条件: P0 True、P1 True(損失の不成立 なし、最後の区間の訓練損失 / 同じ組での学習前のモデルの損失の最大 0.632 <= 0.9。C2 の検証用の Recall@10 の学習前からの増分の最小 +0.1374 >= 0.04: True)、P2 True(R_16(C2) の平均 0.3784 <= 0.95)
      判定関数の結果 支持 -> 最終判定: 支持
      診断量: 入れ子の次元 m ごとの Recall@10(シード平均 ± シード間の標準偏差)と、同じシードの差 C5 - C2(標準偏差は判定と同じ構成)
        m =  16: C5 0.5692 ± 0.0040、C2 0.3784 ± 0.0092、差 +0.1908(標準偏差 0.0074、2 倍 0.0147)
        m =  32: C5 0.6055 ± 0.0035、C2 0.5088 ± 0.0052、差 +0.0967(標準偏差 0.0065、2 倍 0.0130)
        m =  64: C5 0.6256 ± 0.0048、C2 0.5846 ± 0.0036、差 +0.0410(標準偏差 0.0054、2 倍 0.0108)
        m = 128: C5 0.6376 ± 0.0045、C2 0.6359 ± 0.0058、差 +0.0017(標準偏差 0.0046、2 倍 0.0092)
        m = 256: C5 0.6507 ± 0.0034、C2 0.6700 ± 0.0023、差 -0.0193(標準偏差 0.0034、2 倍 0.0068)
      診断量: C5 の最後の区間の次元ごとの訓練損失 [2.293, 2.197, 2.157, 2.134, 2.138](m = (16, 32, 64, 128, 256))、C5 の最後の区間の訓練損失(平均)の平均 2.184、clipping の発動 C5 1.00・C2 1.00
      診断量: 索引のメモリ(FP32、評価用の passage 3244 個): m = 16: 203 KiB、m = 32: 406 KiB、m = 64: 811 KiB、m = 128: 1622 KiB、m = 256: 3244 KiB



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/024_text_embedding_and_retriever/output_43_1.png)
    




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/024_text_embedding_and_retriever.ipynb
