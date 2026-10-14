---
title: "Inverse Cloze Task と passage 単位の 2 段階検索 / Inverse Cloze Task and Two-Stage Retrieval at Passage Level(実装・実験編 6/8)"
---

この記事は後編(実装・実験編 6/8)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/026_inverse_cloze_task_and_two_stage-practice-5)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/026_inverse_cloze_task_and_two_stage-practice-7)。

### 6.3 パイロット(第 1 段の R2 と cross-encoder の学習ステップ数)

6.1 節で宣言した規則のとおり、本番と同じセッションの冒頭で、学習を伴うパイロットを行う。**測るのは単一の条件の量だけ** で、条件間の比較や対比量の計算はしない。評価に使うのは較正専用のシード(90)と検証用の query だけで、評価用の集合は使わない。

1. **第 1 段のパイロット**: R2 を、シード 90・$T_1$・中心の学習率で学習し、第 1 段の上位 $K = 50$ 件を求める(学習用・検証用・評価用の query)。この学習は、6.5 節で R2 の較正の格子の中心の点として使い回す。
   あわせて、各集合の query での coverage(上位 $K$ 件に正例が入る割合)を印字する。学習用の query での coverage が、検証用・評価用の query より大きければ、第 1 段が学習用の passage を見ていることによる学習時と推論時の分布の違いを示す。
2. **cross-encoder のパイロット**: $T_2 = 2048, 4096, 8192$(動作確認では 16, 32, 64)の順に、中心の学習率で学習し、(i)学習の成立 P1(a)、(ii)検証用の query での並べ替えの Recall@10 の増分(学習前の cross-encoder との差)が増分の標準偏差の 2 倍以上、
   (iii)検証用の query での並べ替えの Recall@10 が R2 だけの値を上回る、の 3 つがすべて成立した最初の $T_2$ で止める。どれも成立しなければ最大の候補を使う。選ばれた $T_2$ の最後の学習は、6.5 節で cross-encoder の較正の格子の中心の点として使い回す。


```python
_t0_pilot = time.time()
CALIBRATION_RECORDS: dict[tuple[str, float], dict] = {}  # (条件, 学習率) -> 第 1 段の較正専用シードの記録
CROSS_ENCODER_CALIBRATION_RECORDS: dict[tuple[int, float], dict] = {}  # (T_2, 学習率) -> cross-encoder の較正専用シードの記録


def learning_rate_key(learning_rate: float) -> float:
    return float(f"{learning_rate:.10g}")


# --- 1. 第 1 段のパイロット: R2(較正専用のシード、T_1、中心の学習率)---
_center = FIRST_STAGE_LEARNING_RATE_CENTER["R2"]
CALIBRATION_RECORDS[("R2", learning_rate_key(_center))] = train_first_stage("R2", CALIBRATION_SEED_INDEX, _center, FIRST_STAGE_STEPS, main=False, mine=True)
_pilot_r2 = CALIBRATION_RECORDS[("R2", learning_rate_key(_center))]
print(
    f"第 1 段のパイロット: R2(シード {CALIBRATION_SEED_INDEX}、T_1 = {FIRST_STAGE_STEPS}、学習率 {_center:.3g})、最後の区間の訓練損失 {_pilot_r2['final_train_loss']:.3f}"
    f"(同じ組での学習前のモデルの損失 {_pilot_r2['initial_model_loss_on_final_window']:.3f}、比 {_pilot_r2['final_train_loss'] / _pilot_r2['initial_model_loss_on_final_window']:.3f})、"
    f"検証用の Recall@10: passage 単位 {_pilot_r2['validation']['passage_hits'].mean():.4f}・記事単位 {_pilot_r2['validation']['article_hits'].mean():.4f}、{_pilot_r2['seconds']:.0f} 秒"
)
_coverage_rows = {}
for _name, _entries in (("train", TRAIN_ENTRIES), ("validation", VALIDATION_ENTRIES), ("evaluation", EVALUATION_ENTRIES)):
    _coverage_rows[_name] = float(usable_cloze_queries(MINED[CALIBRATION_SEED_INDEX][_name], _entries).mean())
PILOT_USABLE_QUERIES = int(usable_cloze_queries(MINED[CALIBRATION_SEED_INDEX]["train"], TRAIN_ENTRIES).sum())
print(
    f"第 1 段 R2 の上位 {RERANK_CANDIDATES} 件に正例が入る割合(coverage): 学習用の query {_coverage_rows['train']:.4f}・検証用 {_coverage_rows['validation']:.4f}・評価用 {_coverage_rows['evaluation']:.4f}。"
    f"R2 だけの Recall@10: 検証用 {MINED[CALIBRATION_SEED_INDEX]['validation_hits'].mean():.4f}・評価用 {MINED[CALIBRATION_SEED_INDEX]['evaluation_hits'].mean():.4f}。"
    f"学習に使える query(正例が上位 {RERANK_CANDIDATES} 件に入る学習用の query): {PILOT_USABLE_QUERIES:,} 個(学習用の query {len(TRAIN_ENTRIES):,} 個の {PILOT_USABLE_QUERIES / len(TRAIN_ENTRIES):.1%})"
)

# --- 2. cross-encoder のパイロット ---
def pilot_criteria(record: dict) -> dict:
    reasons = cross_encoder_failure_reasons(record)
    difference = record["validation_hits"].astype(np.float64) - record["untrained_validation_hits"].astype(np.float64)
    gain = float(difference.mean())
    sigma = float(bootstrap_mean(RERANK_VALIDATION_WEIGHTS, difference).std(ddof=1))
    no_rerank = float(record["validation_no_rerank_hits"].mean())
    return {
        "learning": not reasons, "learning_reasons": reasons, "gain": gain, "gain_sigma": sigma, "gain_ok": bool(gain > 0 and gain >= P1_SIGMA_MULTIPLIER * sigma),
        "above_no_rerank": bool(record["validation_recall"] > no_rerank), "no_rerank": no_rerank,
    }


PILOT_ROWS = []
CROSS_ENCODER_STEPS, PILOT_SUCCEEDED = max(CROSS_ENCODER_STEP_CANDIDATES), False
for _steps in CROSS_ENCODER_STEP_CANDIDATES:
    _record = train_cross_encoder(CALIBRATION_SEED_INDEX, CROSS_ENCODER_LEARNING_RATE_CENTER, _steps, main=False)
    CROSS_ENCODER_CALIBRATION_RECORDS[(_steps, learning_rate_key(CROSS_ENCODER_LEARNING_RATE_CENTER))] = _record
    _criteria = pilot_criteria(_record)
    _statistics = learning_loss_statistics(_record)
    PILOT_ROWS.append({"steps": _steps, "criteria": _criteria, "record": _record})
    print(
        f"cross-encoder のパイロット T_2 = {_steps}(学習率 {CROSS_ENCODER_LEARNING_RATE_CENTER:.3g}、シード {CALIBRATION_SEED_INDEX}): 最後の区間の訓練損失 {_record['final_train_loss']:.4f}"
        f"(学習前のモデルの損失 {_record['initial_model_loss_on_final_window']:.4f}、差の平均 / 標準誤差 {_statistics['mean'] / _statistics['standard_error'] if _statistics['standard_error'] > 0 else float('nan'):.1f}、一様な予測 {UNIFORM_LOSS_CROSS_ENCODER:.4f})、"
        f"検証用の query の並べ替えの Recall@10 {_record['validation_recall']:.4f}(学習前 {float(_record['untrained_validation_hits'].mean()):.4f}、増分 {_criteria['gain']:+.4f}、標準偏差 {_criteria['gain_sigma']:.4f}、R2 だけ {_criteria['no_rerank']:.4f}、coverage {_record['validation_coverage']:.4f})、"
        f"スコアの標準偏差の中央値 {_record['validation_score_std_median']:.4f}、clipping の発動 {_record['clip_rate']:.2f}、{_record['seconds']:.0f} 秒"
    )
    print(
        f"  (i)学習の成立 P1(a): {_criteria['learning']} {_criteria['learning_reasons'] or ''}、(ii)増分 >= {P1_SIGMA_MULTIPLIER:g} x 標準偏差: {_criteria['gain_ok']}、(iii)R2 だけを上回る: {_criteria['above_no_rerank']}"
    )
    if _criteria["learning"] and _criteria["gain_ok"] and _criteria["above_no_rerank"]:
        CROSS_ENCODER_STEPS, PILOT_SUCCEEDED = _steps, True
        break
_usable_cross = PILOT_USABLE_QUERIES
print(
    f"パイロットの結果: 3 条件がすべて成立した最小の T_2 = {CROSS_ENCODER_STEPS}" if PILOT_SUCCEEDED else f"パイロットの結果: どの T_2 でも 3 条件がすべては成立しなかった。規則どおり、最大の候補 T_2 = {CROSS_ENCODER_STEPS} を使う(実験 C は前提条件の成否で記録する)"
)
print(
    f"選ばれた T_2 = {CROSS_ENCODER_STEPS} が見る query 数 {BATCH_QUERIES * CROSS_ENCODER_STEPS:,} 個(学習に使える query {_usable_cross:,} 個に対して {BATCH_QUERIES * CROSS_ENCODER_STEPS / max(_usable_cross, 1):.2f} エポック)"
)
PILOT_SECONDS = time.time() - _t0_pilot
print(f"パイロット {PILOT_SECONDS / 60:.1f} 分")
```

    第 1 段のパイロット: R2(シード 90、T_1 = 1024、学習率 0.00048)、最後の区間の訓練損失 1.959(同じ組での学習前のモデルの損失 3.571、比 0.548)、検証用の Recall@10: passage 単位 0.5108・記事単位 0.7028、130 秒
    第 1 段 R2 の上位 50 件に正例が入る割合(coverage): 学習用の query 0.2305・検証用 0.2333・評価用 0.2312。R2 だけの Recall@10: 検証用 0.1339・評価用 0.1271。学習に使える query(正例が上位 50 件に入る学習用の query): 20,686 個(学習用の query 89,752 個の 23.0%)
    cross-encoder のパイロット T_2 = 2048(学習率 0.00024、シード 90): 最後の区間の訓練損失 0.4307(学習前のモデルの損失 2.1115、差の平均 / 標準誤差 -55.8、一様な予測 2.0794)、検証用の query の並べ替えの Recall@10 0.2264(学習前 0.0394、増分 +0.1870、標準偏差 0.0168、R2 だけ 0.1339、coverage 0.2333)、スコアの標準偏差の中央値 4.1989、clipping の発動 1.00、186 秒
      (i)学習の成立 P1(a): True 、(ii)増分 >= 2 x 標準偏差: True、(iii)R2 だけを上回る: True
    パイロットの結果: 3 条件がすべて成立した最小の T_2 = 2048
    選ばれた T_2 = 2048 が見る query 数 16,384 個(学習に使える query 20,686 個に対して 0.79 エポック)
    パイロット 5.3 分


### 6.4 実行計画の選択

6.2・6.3 節の計測とパイロットの後の経過時間から、8 通りの計画(6.1 節)のそれぞれの残りの実行時間を見積もる。

- 第 1 段の学習 1 回 = 1 ステップの時間 × $T_1$ + 学習の組の生成の時間(1 ステップあたりの時間 × $T_1$)+ 学習前のモデルの損失の測定の時間(最後の区間のステップ数 × 1 ステップあたりの時間)+ 評価の時間 + 固定費の余裕 2 秒。
  評価の時間は、較正の学習では検証用の評価 1 回、本番の学習では本番の評価(passage 単位のコーパス全体 + 記事単位)に、R2 では第 1 段の上位 $K$ 件の採掘を加えたもの。
- 第 1 段の較正 = 較正する条件(R2 は常に、R1 は段階 3 まで)ごとに、格子 3 点 + 拡張 1 点の **最悪の場合の 4 回**(パイロットで学習した R2 の中心の点は使い回すので、R2 は 3 回)。
- cross-encoder の学習 1 回 = 1 ステップの時間 × $T_2$ + 学習の組の生成の時間 + 学習前のモデルの損失の測定の時間(順伝播だけなので 1 ステップの時間の 1/3 として見積もる)+ 検証用の query の並べ替えの評価(学習前と学習後)+ 本番の学習ではさらに評価用の query の並べ替えの評価。
  cross-encoder の較正 = 最悪の場合の 4 回(パイロットで学習した中心の点は使い回すので 3 回。段階 5 以降はしない)。
- 本番 = 条件ごとのシード数 × 学習 1 回。観察 D の時間(HNSW の構築の外挿 + 探索 + 並べ替え)と、判定・ブートストラップ・図・アップロードの準備の余裕 300 秒。

**経過時間 + 見積もりが予算(120 分)に収まる計画のうち、番号の最も小さいものを選ぶ。** どれも収まらなければ、本番の学習の前に停止する。


```python
def first_stage_run_seconds(condition: str, kind: str) -> float:
    seconds = (STEP_SECONDS[condition] * TRAINING_SAFETY_FACTOR + BATCH_SECONDS_PER_STEP[condition]) * FIRST_STAGE_STEPS + RUN_OVERHEAD_SECONDS
    seconds += INITIAL_LOSS_SECONDS_PER_STEP * final_loss_window(FIRST_STAGE_STEPS) + CALIBRATION_EVALUATION_SECONDS
    if kind == "main":
        seconds += MAIN_EVALUATION_SECONDS + (MINING_SECONDS if condition == "R2" else 0.0)
    return seconds


def cross_encoder_run_seconds(kind: str, steps: int = CROSS_ENCODER_STEPS) -> float:
    seconds = (CROSS_ENCODER_STEP_SECONDS * TRAINING_SAFETY_FACTOR + CROSS_ENCODER_SCHEDULE_SECONDS_PER_STEP) * steps + RUN_OVERHEAD_SECONDS + CROSS_ENCODER_STEP_SECONDS / 3 * final_loss_window(steps)
    pairs_validation = len(RERANK_VALIDATION_ENTRIES) * RERANK_CANDIDATES
    seconds += 2 * pairs_validation * CROSS_ENCODER_PAIR_SECONDS  # 学習前と学習後の検証用の query の並べ替え
    if kind == "main":
        seconds += len(RERANK_EVALUATION_ENTRIES) * RERANK_CANDIDATES * CROSS_ENCODER_PAIR_SECONDS
    return seconds


def observation_seconds(observation_n: int) -> float:
    queries = LEVEL_SETTINGS["OBSERVATION_QUERIES"]
    search = queries * HNSW_SEARCH_SECONDS_PER_QUERY * len(OBSERVATION_SEARCH_WIDTHS) * 1.5  # 索引が大きいほど遅くなる分の余裕
    rerank = 2 * queries * max(OBSERVATION_RERANK_KS) * CROSS_ENCODER_PAIR_SECONDS  # 全探索と HNSW の上位 100 件
    return hnsw_build_seconds(observation_n) + search + rerank + 30.0


CALIBRATION_POINTS_WORST = len(LEARNING_RATE_GRID_MULTIPLIERS) + 1  # 格子 3 点 + 拡張 1 点


def estimate_stage(stage: int) -> dict:
    settings = stage_settings(stage)
    first_stage_calibration = (CALIBRATION_POINTS_WORST - 1) * first_stage_run_seconds("R2", "calibration")  # パイロットの R2 の中心の点は使い回す
    if settings["calibrate_r1"]:
        first_stage_calibration += CALIBRATION_POINTS_WORST * first_stage_run_seconds("R1", "calibration")
    cross_encoder_calibration = (CALIBRATION_POINTS_WORST - 1) * cross_encoder_run_seconds("calibration") if settings["calibrate_ce"] else 0.0
    seeds = settings["first_stage_seeds"]
    first_stage_main = seeds * (first_stage_run_seconds("R1", "main") + first_stage_run_seconds("R2", "main"))
    cross_encoder_main = settings["cross_encoder_seeds"] * cross_encoder_run_seconds("main")
    observation = observation_seconds(settings["observation_n"])
    pregeneration = (seeds + 1) * BATCH_SECONDS_PER_STEP["R1"] * FIRST_STAGE_STEPS + 2 * FIRST_SCAN_SECONDS  # 学習の組の事前生成(較正のシード + 本番のシード。6.4.1 節)
    total = first_stage_calibration + cross_encoder_calibration + first_stage_main + cross_encoder_main + observation + pregeneration + REPORTING_MARGIN_SECONDS
    return {
        "first_stage_calibration": first_stage_calibration, "cross_encoder_calibration": cross_encoder_calibration, "first_stage_main": first_stage_main,
        "cross_encoder_main": cross_encoder_main, "observation": observation, "pregeneration": pregeneration, "total": total,
    }


ELAPSED_AT_SELECTION = time.time() - NOTEBOOK_START_TIME
STAGE_ESTIMATES = {k: estimate_stage(k) for k in range(NUM_STAGES)}
print(f"経過時間 {ELAPSED_AT_SELECTION / 60:.1f} 分(パイロットを含む)、予算 {SESSION_BUDGET_SECONDS / 60:.0f} 分。T_1 = {FIRST_STAGE_STEPS}、T_2 = {CROSS_ENCODER_STEPS}。学習の時間の見積もりには安全係数 {TRAINING_SAFETY_FACTOR} を掛ける")
for _k in range(NUM_STAGES):
    _e = STAGE_ESTIMATES[_k]
    _fits = ELAPSED_AT_SELECTION + _e["total"] <= SESSION_BUDGET_SECONDS
    print(
        f"  計画 {_k}({STAGE_CHANGES[_k]}): 第 1 段の較正 {_e['first_stage_calibration'] / 60:.1f} 分 + cross-encoder の較正 {_e['cross_encoder_calibration'] / 60:.1f} 分 + 第 1 段の本番 {_e['first_stage_main'] / 60:.1f} 分"
        f" + cross-encoder の本番 {_e['cross_encoder_main'] / 60:.1f} 分 + 観察 D {_e['observation'] / 60:.1f} 分 + 組の事前生成 {_e['pregeneration'] / 60:.2f} 分 + 余裕 -> 残り {_e['total'] / 60:.1f} 分、終了の見込み {(ELAPSED_AT_SELECTION + _e['total']) / 60:.1f} 分({'収まる' if _fits else '超える'})"
    )
_feasible = [k for k in range(NUM_STAGES) if ELAPSED_AT_SELECTION + STAGE_ESTIMATES[k]["total"] <= SESSION_BUDGET_SECONDS]
AUTO_STAGE = _feasible[0] if _feasible else None
if FORCED_STAGE is not None:
    STAGE = FORCED_STAGE
    print(f"*** テスト専用の上書き: 計画 {FORCED_STAGE} を使う(見積もりによる選択は {AUTO_STAGE}) ***")
elif AUTO_STAGE is None:
    raise RuntimeError(
        f"最も下位の計画(計画 {NUM_STAGES - 1})でも見積もりが予算を超えるため、本番の学習の前に停止する。判定の対象の結果は何も得ていない。実行条件(判定基準・水準・前提条件以外)を直して再実行すること。"
    )
else:
    STAGE = AUTO_STAGE
SETTINGS = stage_settings(STAGE)
FIRST_STAGE_SEEDS, CROSS_ENCODER_SEEDS, OBSERVATION_N = SETTINGS["first_stage_seeds"], SETTINGS["cross_encoder_seeds"], SETTINGS["observation_n"]
SEEDS_A = SEEDS_B = FIRST_STAGE_SEEDS
SEEDS_C = CROSS_ENCODER_SEEDS
print(
    f"選ばれた計画: {STAGE}({STAGE_CHANGES[STAGE]})、第 1 段のシード数 {FIRST_STAGE_SEEDS}(実験 A・B)、cross-encoder のシード数 {CROSS_ENCODER_SEEDS}(実験 C)、R1 の較正 {SETTINGS['calibrate_r1']}、cross-encoder の較正 {SETTINGS['calibrate_ce']}、"
    f"観察 D の索引の件数 {OBSERVATION_N}、見積もり {STAGE_ESTIMATES[STAGE]['total'] / 60:.1f} 分"
)
```

    経過時間 15.4 分(パイロットを含む)、予算 120 分。T_1 = 1024、T_2 = 2048。学習の時間の見積もりには安全係数 1.1 を掛ける
      計画 0((削らない)): 第 1 段の較正 10.5 分 + cross-encoder の較正 10.1 分 + 第 1 段の本番 22.1 分 + cross-encoder の本番 21.0 分 + 観察 D 2.4 分 + 組の事前生成 0.12 分 + 余裕 -> 残り 71.3 分、終了の見込み 86.7 分(収まる)
      計画 1(観察 D の索引の件数を 1 段下げる): 第 1 段の較正 10.5 分 + cross-encoder の較正 10.1 分 + 第 1 段の本番 22.1 分 + cross-encoder の本番 21.0 分 + 観察 D 1.9 分 + 組の事前生成 0.12 分 + 余裕 -> 残り 70.8 分、終了の見込み 86.2 分(収まる)
      計画 2(cross-encoder のシード数を「中」にする): 第 1 段の較正 10.5 分 + cross-encoder の較正 10.1 分 + 第 1 段の本番 22.1 分 + cross-encoder の本番 12.6 分 + 観察 D 1.9 分 + 組の事前生成 0.12 分 + 余裕 -> 残り 62.4 分、終了の見込み 77.8 分(収まる)
      計画 3(第 1 段(R1・R2)のシード数を「中」にする): 第 1 段の較正 10.5 分 + cross-encoder の較正 10.1 分 + 第 1 段の本番 13.2 分 + cross-encoder の本番 12.6 分 + 観察 D 1.9 分 + 組の事前生成 0.09 分 + 余裕 -> 残り 53.5 分、終了の見込み 68.9 分(収まる)
      計画 4(R1 の学習率の較正をやめ、024 の値にする): 第 1 段の較正 4.5 分 + cross-encoder の較正 10.1 分 + 第 1 段の本番 13.2 分 + cross-encoder の本番 12.6 分 + 観察 D 1.9 分 + 組の事前生成 0.09 分 + 余裕 -> 残り 47.5 分、終了の見込み 62.9 分(収まる)
      計画 5(cross-encoder の学習率の較正をやめ、中心の値にする): 第 1 段の較正 4.5 分 + cross-encoder の較正 0.0 分 + 第 1 段の本番 13.2 分 + cross-encoder の本番 12.6 分 + 観察 D 1.9 分 + 組の事前生成 0.09 分 + 余裕 -> 残り 37.4 分、終了の見込み 52.8 分(収まる)
      計画 6(cross-encoder のシード数を「低」にする): 第 1 段の較正 4.5 分 + cross-encoder の較正 0.0 分 + 第 1 段の本番 13.2 分 + cross-encoder の本番 8.4 分 + 観察 D 1.9 分 + 組の事前生成 0.09 分 + 余裕 -> 残り 33.2 分、終了の見込み 48.6 分(収まる)
      計画 7(第 1 段(R1・R2)のシード数を「低」にする): 第 1 段の較正 4.5 分 + cross-encoder の較正 0.0 分 + 第 1 段の本番 8.8 分 + cross-encoder の本番 8.4 分 + 観察 D 1.9 分 + 組の事前生成 0.08 分 + 余裕 -> 残り 28.7 分、終了の見込み 44.1 分(収まる)
    選ばれた計画: 0((削らない))、第 1 段のシード数 5(実験 A・B)、cross-encoder のシード数 5(実験 C)、R1 の較正 True、cross-encoder の較正 True、観察 D の索引の件数 16384、見積もり 71.3 分


#### 6.4.1 学習の組の事前生成

選んだ計画が使う学習の組(較正のシードと、計画の第 1 段のシード数ぶんの本番のシード。それぞれ $T_1$ ステップ)の生成を、学習の前に済ませて確かめる。確かめる内容は次のとおり。

- 例外なく最後まで終わること。
- 選ばれた記事の重みがすべて正で、同じ記事の 2 組の窓が重ならず、query の長さが範囲に入っていること。
- 生成時間が、6.2 節の計測(ステップ数に比例)の見積もりの 2 倍(と固定の余裕 5 秒)に収まること。

この生成の時間は 6.4 節の見積もりに含める。生成した組は保持せず(メモリのため)、学習のときに同じシードから再び生成する(生成は決定的)。cross-encoder の学習の組は、学習済みの第 1 段の上位の候補から作るため、学習の前には生成できない(パイロットで、$T_2$ の候補のすべてについて生成を確かめている)。


```python
# 本番で使う学習の組(較正のシードと本番のシード、T_1 ステップ)の生成を、学習の前に済ませて確かめる
_t0_pregeneration = time.time()
PREGENERATION_ROWS = []
for _seed in [CALIBRATION_SEED_INDEX, *range(FIRST_STAGE_SEEDS)]:
    PAIR_BATCHES.clear()
    _start = time.time()
    _batches = pair_batches(_seed, FIRST_STAGE_STEPS)  # 例外なく最後まで終わること
    _seconds = time.time() - _start
    assert _batches.articles.shape == (FIRST_STAGE_STEPS, BATCH_SIZE)
    assert (SAMPLING_WEIGHTS[_batches.articles] > 0).all(), "重みが 0 の記事が選ばれた"
    _starts_pair = _batches.window_starts.reshape(FIRST_STAGE_STEPS, ARTICLES_PER_BATCH, PAIRS_PER_ARTICLE)
    assert (np.abs(_starts_pair[:, :, 0] - _starts_pair[:, :, 1]) >= PASSAGE_LENGTH).all(), "同じ記事の 2 組の窓が重なっている"
    assert (_batches.query_lengths >= MIN_QUERY_TOKENS).all() and (_batches.query_lengths <= QUERY_LENGTH).all()
    PREGENERATION_ROWS.append((_seed, _seconds, _batches.num_attempts, _batches.num_exhaustive, float(_batches.truncated.mean())))
PAIR_BATCHES.clear()
PREGENERATION_SECONDS = time.time() - _t0_pregeneration
_per_step = np.array([row[1] for row in PREGENERATION_ROWS]) / FIRST_STAGE_STEPS
_estimated = BATCH_SECONDS_PER_STEP["R1"] * FIRST_STAGE_STEPS * len(PREGENERATION_ROWS)
print(
    f"学習の組の事前生成: シード {[row[0] for row in PREGENERATION_ROWS]}、T_1 = {FIRST_STAGE_STEPS} ステップを例外なく生成した。合計 {PREGENERATION_SECONDS:.1f} 秒(見積もり {_estimated:.1f} 秒)、"
    f"1 ステップあたり {_per_step.min() * 1000:.2f}〜{_per_step.max() * 1000:.2f} ms(6.2 節の計測は {BATCH_SECONDS_PER_STEP['R1'] * 1000:.2f} ms)。"
    f"窓の組を引き直した回数 {[row[2] for row in PREGENERATION_ROWS]}、記事のすべての窓から選んだ組 {[row[3] for row in PREGENERATION_ROWS]}、切り詰めた query の割合 {[round(row[4], 3) for row in PREGENERATION_ROWS]}"
)
assert PREGENERATION_SECONDS <= 2 * _estimated + 5.0, "学習の組の生成が、見積もり(ステップ数に比例)の 2 倍を超えている"
```

    学習の組の事前生成: シード [90, 0, 1, 2, 3, 4]、T_1 = 1024 ステップを例外なく生成した。合計 5.0 秒(見積もり 4.9 秒)、1 ステップあたり 0.79〜0.83 ms(6.2 節の計測は 0.79 ms)。窓の組を引き直した回数 [40521, 39894, 40405, 40611, 39827, 40230]、記事のすべての窓から選んだ組 [50, 38, 44, 46, 38, 47]、切り詰めた query の割合 [0.042, 0.044, 0.043, 0.043, 0.043, 0.042]


### 6.5 学習率の較正

選ばれた計画の規則(6.1 節)のとおりに学習率を決める。第 1 段は $T_1$、cross-encoder は選ばれた $T_2$ で、較正専用のシード(90)の 1 シードで各格子点を学習する。**評価用の集合は較正に使わない。**
各格子点について、指標(第 1 段は検証用の passage 単位と記事単位の Recall@10 の平均 $M$、cross-encoder は検証用の query の並べ替えの Recall@10)とその標準偏差(記事を単位とするブートストラップ)、最後の区間の訓練損失、学習の成立の余裕を印字する。
パイロットで学習した R2 の中心の点と、選ばれた $T_2$ の cross-encoder の中心の点は、同じ設定であることをアサーションで確かめたうえで使い回す。

選択の規則: 指標が最大の点から標準偏差の 1 倍以内の候補(学習の成立の前提条件 P1(a) を満たす点だけ)のうち、学習の成立の余裕が最大の点(同点なら小さい学習率)。
最良が格子の端に来た場合は、その方向に公比 4 で 1 点だけ拡張して学習し、改めて同じ規則で選ぶ。


```python
_t0_calibration = time.time()
FIRST_STAGE_CALIBRATION: dict[str, dict] = {}
CROSS_ENCODER_CALIBRATION: dict = {}


def calibrate_first_stage(condition: str) -> dict:
    grid = list(first_stage_grid(condition))
    records: dict[float, dict] = {}

    def record_for(learning_rate: float) -> dict:
        key = learning_rate_key(learning_rate)
        if key not in records:
            if (condition, key) in CALIBRATION_RECORDS:  # パイロットの学習の使い回し(同じ設定であることを確かめる)
                reused = CALIBRATION_RECORDS[(condition, key)]
                assert reused["seed"] == CALIBRATION_SEED_INDEX and reused["num_steps"] == FIRST_STAGE_STEPS and math.isclose(reused["learning_rate"], learning_rate)
                records[key] = reused
            else:
                records[key] = train_first_stage(condition, CALIBRATION_SEED_INDEX, learning_rate, FIRST_STAGE_STEPS, main=False)
                CALIBRATION_RECORDS[(condition, key)] = records[key]
        return records[key]

    def choose(points: list[float]) -> tuple[int | None, list[dict]]:
        rows = [record_for(p) for p in points]
        values = [calibration_metric(r["validation"]) for r in rows]
        deviations = [calibration_metric_deviation(r["validation"]) for r in rows]
        margins = [1.0 - r["final_train_loss"] / r["initial_model_loss_on_final_window"] for r in rows]
        eligible = [not first_stage_failure_reasons(r) for r in rows]
        index = select_calibrated_index(points, values, deviations, margins, eligible)
        if index is None:  # P1(a) を満たす点がない: 余裕が最大の点を選び、P1 の不成立として進める
            index = max(range(len(points)), key=lambda i: (margins[i], -points[i]))
        return index, rows

    index, rows = choose(grid)
    extended = None
    if index == 0:
        extended = "lower"
        grid = [grid[0] / LEARNING_RATE_GRID_RATIO] + grid
        index, rows = choose(grid)
    elif index == len(grid) - 1:
        extended = "upper"
        grid = grid + [grid[-1] * LEARNING_RATE_GRID_RATIO]
        index, rows = choose(grid)
    return {
        "grid": grid, "chosen": grid[index], "interior": 0 < index < len(grid) - 1, "extended": extended,
        "metric": [calibration_metric(r["validation"]) for r in rows], "metric_deviation": [calibration_metric_deviation(r["validation"]) for r in rows],
        "passage": [float(r["validation"]["passage_hits"].mean()) for r in rows], "article": [float(r["validation"]["article_hits"].mean()) for r in rows],
        "final_train_loss": [r["final_train_loss"] for r in rows], "initial_loss": [r["initial_model_loss_on_final_window"] for r in rows],
        "p1a": [not first_stage_failure_reasons(r) for r in rows],
    }


def calibrate_cross_encoder() -> dict:
    grid = list(cross_encoder_grid())
    records: dict[float, dict] = {}

    def record_for(learning_rate: float) -> dict:
        key = learning_rate_key(learning_rate)
        if key not in records:
            if (CROSS_ENCODER_STEPS, key) in CROSS_ENCODER_CALIBRATION_RECORDS:  # パイロットの学習の使い回し
                reused = CROSS_ENCODER_CALIBRATION_RECORDS[(CROSS_ENCODER_STEPS, key)]
                assert reused["seed"] == CALIBRATION_SEED_INDEX and reused["num_steps"] == CROSS_ENCODER_STEPS and math.isclose(reused["learning_rate"], learning_rate)
                records[key] = reused
            else:
                records[key] = train_cross_encoder(CALIBRATION_SEED_INDEX, learning_rate, CROSS_ENCODER_STEPS, main=False)
                CROSS_ENCODER_CALIBRATION_RECORDS[(CROSS_ENCODER_STEPS, key)] = records[key]
        return records[key]

    def choose(points: list[float]) -> tuple[int, list[dict], bool]:
        rows = [record_for(p) for p in points]
        values = [r["validation_recall"] for r in rows]
        deviations = [float(bootstrap_mean(RERANK_VALIDATION_WEIGHTS, r["validation_hits"]).std(ddof=1)) for r in rows]
        statistics = [learning_loss_statistics(r) for r in rows]
        margins = [-s["mean"] / s["standard_error"] if s["standard_error"] > 0 else -math.inf for s in statistics]
        eligible = [not cross_encoder_failure_reasons(r) for r in rows]
        index = select_calibrated_index(points, values, deviations, margins, eligible)
        all_failed = index is None
        if index is None:  # P1(a) を満たす点がない: 余裕が最大の点を選び、P1 の不成立として進める
            index = max(range(len(points)), key=lambda i: (margins[i], -points[i]))
        return index, rows, all_failed

    index, rows, all_failed = choose(grid)
    extended = None
    if index == 0:
        extended = "lower"
        grid = [grid[0] / LEARNING_RATE_GRID_RATIO] + grid
        index, rows, all_failed = choose(grid)
    elif index == len(grid) - 1:
        extended = "upper"
        grid = grid + [grid[-1] * LEARNING_RATE_GRID_RATIO]
        index, rows, all_failed = choose(grid)
    return {
        "grid": grid, "chosen": grid[index], "interior": 0 < index < len(grid) - 1, "extended": extended, "all_failed": all_failed,
        "recall": [r["validation_recall"] for r in rows], "recall_deviation": [float(bootstrap_mean(RERANK_VALIDATION_WEIGHTS, r["validation_hits"]).std(ddof=1)) for r in rows],
        "final_train_loss": [r["final_train_loss"] for r in rows], "initial_loss": [r["initial_model_loss_on_final_window"] for r in rows],
        "z": [-learning_loss_statistics(r)["mean"] / learning_loss_statistics(r)["standard_error"] if learning_loss_statistics(r)["standard_error"] > 0 else float("nan") for r in rows],
        "p1a": [not cross_encoder_failure_reasons(r) for r in rows], "score_std": [r["validation_score_std_median"] for r in rows],
    }


def describe_first_stage(condition: str, result: dict) -> str:
    return (
        f"較正 {condition}: 学習率 {[float(f'{x:.3g}') for x in result['grid']]} -> 指標 M {rounded(result['metric'])}(標準偏差 {rounded(result['metric_deviation'])})、"
        f"passage 単位 {rounded(result['passage'])}・記事単位 {rounded(result['article'])}、最後の区間の訓練損失 {rounded(result['final_train_loss'], 3)}"
        f"(学習前のモデルの損失 {rounded(result['initial_loss'], 3)})、P1(a) {result['p1a']}、拡張 {result['extended'] or 'なし'}、選んだ学習率 {result['chosen']:.3g}({'内点' if result['interior'] else '端'})"
    )


LEARNING_RATE, LEARNING_RATE_SOURCE = {}, {}
FIRST_STAGE_CALIBRATION["R2"] = calibrate_first_stage("R2")
print(describe_first_stage("R2", FIRST_STAGE_CALIBRATION["R2"]))
LEARNING_RATE["R2"], LEARNING_RATE_SOURCE["R2"] = FIRST_STAGE_CALIBRATION["R2"]["chosen"], "較正"
if SETTINGS["calibrate_r1"]:
    FIRST_STAGE_CALIBRATION["R1"] = calibrate_first_stage("R1")
    print(describe_first_stage("R1", FIRST_STAGE_CALIBRATION["R1"]))
    LEARNING_RATE["R1"], LEARNING_RATE_SOURCE["R1"] = FIRST_STAGE_CALIBRATION["R1"]["chosen"], "較正"
else:
    LEARNING_RATE["R1"], LEARNING_RATE_SOURCE["R1"] = R1_RULE_LEARNING_RATE, "規則(024 の C5 の較正の値)"
    print(f"R1 の学習率: 較正しない。024 の C5 の較正で選ばれた値 {R1_RULE_LEARNING_RATE:.3g} を使う(P0 の対象外)")
if SETTINGS["calibrate_ce"]:
    CROSS_ENCODER_CALIBRATION = calibrate_cross_encoder()
    _r = CROSS_ENCODER_CALIBRATION
    print(
        f"較正 cross-encoder(T_2 = {CROSS_ENCODER_STEPS}): 学習率 {[float(f'{x:.3g}') for x in _r['grid']]} -> 検証用の並べ替えの Recall@10 {rounded(_r['recall'])}(標準偏差 {rounded(_r['recall_deviation'])})、"
        f"最後の区間の訓練損失 {rounded(_r['final_train_loss'], 3)}(学習前のモデルの損失 {rounded(_r['initial_loss'], 3)})、損失の差の平均 / 標準誤差の符号反転 {rounded(_r['z'], 1)}、P1(a) {_r['p1a']}、"
        f"スコアの標準偏差の中央値 {rounded(_r['score_std'], 4)}、拡張 {_r['extended'] or 'なし'}、選んだ学習率 {_r['chosen']:.3g}({'内点' if _r['interior'] else '端'})"
        + ("。すべての格子点で P1(a) が不成立: 余裕が最大の点を選び、実験 C は P1 の不成立として進める" if _r["all_failed"] else "")
    )
    LEARNING_RATE["cross_encoder"], LEARNING_RATE_SOURCE["cross_encoder"] = _r["chosen"], "較正"
else:
    LEARNING_RATE["cross_encoder"], LEARNING_RATE_SOURCE["cross_encoder"] = CROSS_ENCODER_RULE_LEARNING_RATE, "規則(中心の値)"
    print(f"cross-encoder の学習率: 較正しない。中心の値 {CROSS_ENCODER_RULE_LEARNING_RATE:.3g} を使う(P0 の対象外)")

# P0(較正): 各実験の対比量に入る条件のうち、自分自身を較正した条件で、選んだ学習率が内点であること
P0_DETAIL = {}
for _experiment, _conditions in {"A": ("R1", "R2"), "B": ("R1", "R2"), "C": ("R2", "cross_encoder")}.items():
    _calibrated = {}
    for _c in _conditions:
        if _c in ("R1", "R2") and _c in FIRST_STAGE_CALIBRATION:
            _calibrated[_c] = FIRST_STAGE_CALIBRATION[_c]["interior"]
        elif _c == "cross_encoder" and CROSS_ENCODER_CALIBRATION:
            _calibrated[_c] = CROSS_ENCODER_CALIBRATION["interior"]
    P0_DETAIL[_experiment] = {"calibrated": _calibrated, "by_rule": [c for c in _conditions if c not in _calibrated]}
    precondition_status[f"P0({_experiment})"] = all(_calibrated.values())
CALIBRATION_SECONDS = time.time() - _t0_calibration
print(f"学習率: {json.dumps({c: float(f'{v:.3g}') for c, v in LEARNING_RATE.items()})}、値の出どころ {json.dumps(LEARNING_RATE_SOURCE, ensure_ascii=False)}")
print(
    f"{RUN_TAG}P0(較正): "
    + "、".join(f"実験 {e}: {precondition_status[f'P0({e})']}(較正した条件の内点 {json.dumps(d['calibrated'])}、規則で決めた条件(対象外){d['by_rule'] or 'なし'})" for e, d in P0_DETAIL.items())
)
print(f"較正 {CALIBRATION_SECONDS / 60:.1f} 分(見積もり {(STAGE_ESTIMATES[STAGE]['first_stage_calibration'] + STAGE_ESTIMATES[STAGE]['cross_encoder_calibration']) / 60:.1f} 分、最悪の場合)")
```

    較正 R2: 学習率 [0.00012, 0.00048, 0.00192] -> 指標 M [0.5586, 0.6068, 0.5556](標準偏差 [0.017, 0.0161, 0.0185])、passage 単位 [0.4587, 0.5108, 0.4577]・記事単位 [0.6585, 0.7028, 0.6535]、最後の区間の訓練損失 [2.182, 1.959, 2.103](学習前のモデルの損失 [3.571, 3.571, 3.571])、P1(a) [True, True, True]、拡張 なし、選んだ学習率 0.00048(内点)
    較正 R1: 学習率 [0.00012, 0.00048, 0.00192] -> 指標 M [0.5315, 0.5605, 0.4734](標準偏差 [0.0188, 0.0183, 0.0208])、passage 単位 [0.4252, 0.4557, 0.3593]・記事単位 [0.6378, 0.6654, 0.5876]、最後の区間の訓練損失 [2.918, 2.711, 2.881](学習前のモデルの損失 [4.734, 4.734, 4.734])、P1(a) [True, True, True]、拡張 なし、選んだ学習率 0.00048(内点)
    較正 cross-encoder(T_2 = 2048): 学習率 [6e-05, 0.00024, 0.00096, 0.00384] -> 検証用の並べ替えの Recall@10 [0.2156, 0.2264, 0.2126, 0.2018](標準偏差 [0.0185, 0.0186, 0.0173, 0.0176])、最後の区間の訓練損失 [0.639, 0.431, 0.519, 0.791](学習前のモデルの損失 [2.112, 2.112, 2.112, 2.112])、損失の差の平均 / 標準誤差の符号反転 [44.5, 55.8, 56.8, 32.6]、P1(a) [True, True, True, True]、スコアの標準偏差の中央値 [3.2306, 4.1989, 2.8169, 2.0704]、拡張 upper、選んだ学習率 0.00096(内点)
    学習率: {"R2": 0.00048, "R1": 0.00048, "cross_encoder": 0.00096}、値の出どころ {"R2": "較正", "R1": "較正", "cross_encoder": "較正"}
    P0(較正): 実験 A: True(較正した条件の内点 {"R1": true, "R2": true}、規則で決めた条件(対象外)なし)、実験 B: True(較正した条件の内点 {"R1": true, "R2": true}、規則で決めた条件(対象外)なし)、実験 C: True(較正した条件の内点 {"R2": true, "cross_encoder": true}、規則で決めた条件(対象外)なし)
    較正 15.4 分(見積もり 20.6 分、最悪の場合)


### 6.6 第 1 段の本番の学習と評価

選ばれた計画のシード数の R1 と R2 をすべて学習する(同じシードの R1 と R2 は、記事の並びが同じ)。評価は評価用の集合で、最終ステップの重みについて行う(passage 単位はコーパス全体の索引、記事単位は 024 の評価用の索引)。
R2 は、cross-encoder の学習と評価に使う第 1 段の上位 $K$ 件も求める。1 行あたりの負例の数(R1 は同じ記事の組を除いて 62 個、R2 は 63 個)、文を文脈に残した組の割合、切り詰めた query の割合を、学習ごとに印字する。
診断量として、024 と同じ形式の記事単位の評価の Recall@10 を R1・R2・024 の Hub のモデルで測り、R1 のシード 0 と 024 の値の対応を印字する(組の作り方が違うので、一致は期待しない。判定には使わない)。


```python
_t0_main = time.time()
RUNS: dict[tuple[str, int], dict] = {}
for _s in range(FIRST_STAGE_SEEDS):
    for _c in FIRST_STAGE_CONDITIONS:
        RUNS[(_c, _s)] = train_first_stage(_c, _s, LEARNING_RATE[_c], FIRST_STAGE_STEPS, main=True)
        _r = RUNS[(_c, _s)]
        print(
            f"{_c} シード {_s}(学習率 {_r['learning_rate']:.3g}): 評価用 Recall@10 passage 単位のラベル {_r['passage_hits'].mean():.4f}・記事単位のラベル {_r['article_hits'].mean():.4f}"
            f"(024 と同じ形式の記事単位の評価 {_r['format024_hits'].mean():.4f}。検証用 {_r['validation']['passage_hits'].mean():.4f}・{_r['validation']['article_hits'].mean():.4f})、最後の区間の訓練損失 {_r['final_train_loss']:.3f}"
            f"(学習前のモデルの損失 {_r['initial_model_loss_on_final_window']:.3f}、比 {_r['final_train_loss'] / _r['initial_model_loss_on_final_window']:.3f})、clipping の発動 {_r['clip_rate']:.2f}、更新のスキップ {_r['skipped_steps']}、"
            f"1 行あたりの負例の数 {_r['negatives_per_row']:.1f}、文を文脈に残した組の割合 {_r['kept_share']:.3f}、切り詰めた query の割合 {_r['truncated_query_share']:.3f}、{_r['seconds']:.0f} 秒"
        )
    assert RUNS[("R1", _s)]["article_hash"] == RUNS[("R2", _s)]["article_hash"], "同じシードの R1 と R2 の記事の並びが一致しない"
FIRST_STAGE_MAIN_SECONDS = time.time() - _t0_main
print(f"第 1 段の本番の学習と評価 {FIRST_STAGE_MAIN_SECONDS / 60:.1f} 分(見積もり {STAGE_ESTIMATES[STAGE]['first_stage_main'] / 60:.1f} 分)")

# --- 診断量: 024 と同じ形式の記事単位の評価(R1・R2・024 の Hub の重みの 3 つ。判定には使わない)と、R1 の再現の確認 ---
from huggingface_hub import hf_hub_download  # noqa: E402

_hub_config = json.loads(Path(hf_hub_download(EMBEDDING_REPO_ID, "config.json")).read_text(encoding="utf-8"))
_hub_backbone = build_gpt(_hub_config)
_hub_backbone.load_state_dict(torch.load(hf_hub_download(EMBEDDING_REPO_ID, "model_state.pt"), map_location="cpu"))
_hub_model = TextEmbeddingModel(_hub_backbone, pooling=_hub_config["pooling"], attention=_hub_config["attention"]).to(device).eval()
_hub_queries, _hub_passages = embed_article_set(_hub_model, "evaluation")
HUB_FORMAT_024_RECALL = evaluate_embeddings(_hub_queries, _hub_passages, EVALUATION_SET).recall_at(10)
del _hub_model, _hub_backbone, _hub_queries, _hub_passages
empty_device_cache()
print(f"診断量: 024 と同じ形式の記事単位の評価(024 の評価用の索引と query。固定長)の Recall@10(シード平均): R1 {np.mean([RUNS[('R1', s)]['format024_hits'].mean() for s in range(FIRST_STAGE_SEEDS)]):.4f}、R2 {np.mean([RUNS[('R2', s)]['format024_hits'].mean() for s in range(FIRST_STAGE_SEEDS)]):.4f}、024 の Hub の重み(C5 のシード 0)を読み込み直した値 {HUB_FORMAT_024_RECALL:.4f}(024 のモデルカードの値 {REFERENCE_RECALL_024})。判定には使わない")
_r1 = RUNS[("R1", 0)]
print(
    f"R1 の再現の確認(診断量): 本トピックの R1 は、024 の C5 と入力の形式(query は抜いた文、passage は文を除いた形)・正例の作り方・バッチ(同じ記事の組を含む)が違うので、"
    f"ビット単位の一致も、024 の値との一致も期待しない。同じ形式の評価での R1 のシード 0 の値は {_r1['format024_hits'].mean():.4f}、024 の Hub の重みは {HUB_FORMAT_024_RECALL:.4f}(差 {_r1['format024_hits'].mean() - HUB_FORMAT_024_RECALL:+.4f}。"
    f"024 の C5 の記事単位 Recall@10 のシード間の標準偏差の目安は 0.002〜0.009)"
)
```

    R1 シード 0(学習率 0.00048): 評価用 Recall@10 passage 単位のラベル 0.0964・記事単位のラベル 0.1546(024 と同じ形式の記事単位の評価 0.6205。検証用 0.4557・0.6811)、最後の区間の訓練損失 2.847(学習前のモデルの損失 4.851、比 0.587)、clipping の発動 1.00、更新のスキップ 0、1 行あたりの負例の数 62.0、文を文脈に残した組の割合 0.099、切り詰めた query の割合 0.044、125 秒
    R2 シード 0(学習率 0.00048): 評価用 Recall@10 passage 単位のラベル 0.1348・記事単位のラベル 0.1661(024 と同じ形式の記事単位の評価 0.6338。検証用 0.5118・0.6870)、最後の区間の訓練損失 2.030(学習前のモデルの損失 3.579、比 0.567)、clipping の発動 1.00、更新のスキップ 0、1 行あたりの負例の数 63.0、文を文脈に残した組の割合 0.099、切り詰めた query の割合 0.044、133 秒
    R1 シード 1(学習率 0.00048): 評価用 Recall@10 passage 単位のラベル 0.0970・記事単位のラベル 0.1518(024 と同じ形式の記事単位の評価 0.6122。検証用 0.4596・0.6752)、最後の区間の訓練損失 2.801(学習前のモデルの損失 4.760、比 0.588)、clipping の発動 1.00、更新のスキップ 0、1 行あたりの負例の数 62.0、文を文脈に残した組の割合 0.100、切り詰めた query の割合 0.043、125 秒
    R2 シード 1(学習率 0.00048): 評価用 Recall@10 passage 単位のラベル 0.1236・記事単位のラベル 0.1726(024 と同じ形式の記事単位の評価 0.6403。検証用 0.4892・0.6900)、最後の区間の訓練損失 1.998(学習前のモデルの損失 3.570、比 0.560)、clipping の発動 1.00、更新のスキップ 0、1 行あたりの負例の数 63.0、文を文脈に残した組の割合 0.100、切り詰めた query の割合 0.043、132 秒
    R1 シード 2(学習率 0.00048): 評価用 Recall@10 passage 単位のラベル 0.0964・記事単位のラベル 0.1481(024 と同じ形式の記事単位の評価 0.6057。検証用 0.4321・0.6900)、最後の区間の訓練損失 2.760(学習前のモデルの損失 4.702、比 0.587)、clipping の発動 1.00、更新のスキップ 0、1 行あたりの負例の数 62.0、文を文脈に残した組の割合 0.101、切り詰めた query の割合 0.043、125 秒
    R2 シード 2(学習率 0.00048): 評価用 Recall@10 passage 単位のラベル 0.1320・記事単位のラベル 0.1624(024 と同じ形式の記事単位の評価 0.6430。検証用 0.5148・0.6929)、最後の区間の訓練損失 1.990(学習前のモデルの損失 3.537、比 0.563)、clipping の発動 1.00、更新のスキップ 0、1 行あたりの負例の数 63.0、文を文脈に残した組の割合 0.101、切り詰めた query の割合 0.043、132 秒
    R1 シード 3(学習率 0.00048): 評価用 Recall@10 passage 単位のラベル 0.1001・記事単位のラベル 0.1518(024 と同じ形式の記事単位の評価 0.6122。検証用 0.4596・0.6722)、最後の区間の訓練損失 2.714(学習前のモデルの損失 4.695、比 0.578)、clipping の発動 1.00、更新のスキップ 0、1 行あたりの負例の数 62.0、文を文脈に残した組の割合 0.100、切り詰めた query の割合 0.043、125 秒
    R2 シード 3(学習率 0.00048): 評価用 Recall@10 passage 単位のラベル 0.1360・記事単位のラベル 0.1695(024 と同じ形式の記事単位の評価 0.6335。検証用 0.5197・0.6821)、最後の区間の訓練損失 1.974(学習前のモデルの損失 3.475、比 0.568)、clipping の発動 1.00、更新のスキップ 0、1 行あたりの負例の数 63.0、文を文脈に残した組の割合 0.100、切り詰めた query の割合 0.043、132 秒
    R1 シード 4(学習率 0.00048): 評価用 Recall@10 passage 単位のラベル 0.1004・記事単位のラベル 0.1528(024 と同じ形式の記事単位の評価 0.6116。検証用 0.4449・0.6663)、最後の区間の訓練損失 2.708(学習前のモデルの損失 4.686、比 0.578)、clipping の発動 1.00、更新のスキップ 0、1 行あたりの負例の数 62.0、文を文脈に残した組の割合 0.101、切り詰めた query の割合 0.042、125 秒
    R2 シード 4(学習率 0.00048): 評価用 Recall@10 passage 単位のラベル 0.1255・記事単位のラベル 0.1726(024 と同じ形式の記事単位の評価 0.6307。検証用 0.5059・0.7047)、最後の区間の訓練損失 2.006(学習前のモデルの損失 3.556、比 0.564)、clipping の発動 1.00、更新のスキップ 0、1 行あたりの負例の数 63.0、文を文脈に残した組の割合 0.101、切り詰めた query の割合 0.042、133 秒
    第 1 段の本番の学習と評価 21.5 分(見積もり 22.1 分)



    config.json:   0%|          | 0.00/732 [00:00<?, ?B/s]



    model_state.pt: reconstructing file:   0%|          |  0.00B / 29.4MB            



    model_state.pt: downloading bytes:           |  0.00B            


    診断量: 024 と同じ形式の記事単位の評価(024 の評価用の索引と query。固定長)の Recall@10(シード平均): R1 0.6125、R2 0.6363、024 の Hub の重み(C5 のシード 0)を読み込み直した値 0.6477(024 のモデルカードの値 0.6477)。判定には使わない
    R1 の再現の確認(診断量): 本トピックの R1 は、024 の C5 と入力の形式(query は抜いた文、passage は文を除いた形)・正例の作り方・バッチ(同じ記事の組を含む)が違うので、ビット単位の一致も、024 の値との一致も期待しない。同じ形式の評価での R1 のシード 0 の値は 0.6205、024 の Hub の重みは 0.6477(差 -0.0271。024 の C5 の記事単位 Recall@10 のシード間の標準偏差の目安は 0.002〜0.009)


### 6.7 cross-encoder の本番の学習と評価

選ばれた計画のシード数の cross-encoder を学習する。シード $s$ の cross-encoder は、シード $s$ の R2 の上位 $K$ 件で学習し、評価する。評価は、R2 の上位 $K$ 件を並べ替えた後の Recall@10 を、評価用の query で測る。


```python
_t0_ce = time.time()
CROSS_ENCODER_RUNS: dict[int, dict] = {}
for _s in range(CROSS_ENCODER_SEEDS):
    CROSS_ENCODER_RUNS[_s] = train_cross_encoder(_s, LEARNING_RATE["cross_encoder"], CROSS_ENCODER_STEPS, main=True)
    _r = CROSS_ENCODER_RUNS[_s]
    _statistics = learning_loss_statistics(_r)
    print(
        f"cross-encoder シード {_s}(T_2 = {CROSS_ENCODER_STEPS}、学習率 {_r['learning_rate']:.3g}): 並べ替えた Recall@10 {_r['hits'].mean():.4f}(R2 だけ {_r['no_rerank_hits'].mean():.4f}、coverage {_r['coverage']:.4f})、"
        f"最後の区間の訓練損失 {_r['final_train_loss']:.3f}(学習前のモデルの損失 {_r['initial_model_loss_on_final_window']:.3f}、差の平均 / 標準誤差 {_statistics['mean'] / _statistics['standard_error'] if _statistics['standard_error'] > 0 else float('nan'):.1f})、"
        f"検証用の並べ替えの Recall@10 {_r['validation_recall']:.4f}(学習前 {float(_r['untrained_validation_hits'].mean()):.4f})、スコアの標準偏差の中央値 {_r['score_std_median']:.4f}、clipping の発動 {_r['clip_rate']:.2f}、{_r['seconds']:.0f} 秒"
    )
CROSS_ENCODER_MAIN_SECONDS = time.time() - _t0_ce
print(f"cross-encoder の本番の学習と評価 {CROSS_ENCODER_MAIN_SECONDS / 60:.1f} 分(見積もり {STAGE_ESTIMATES[STAGE]['cross_encoder_main'] / 60:.1f} 分)")
```

    cross-encoder シード 0(T_2 = 2048、学習率 0.00096): 並べ替えた Recall@10 0.2268(R2 だけ 0.1348、coverage 0.2315)、最後の区間の訓練損失 0.307(学習前のモデルの損失 2.107、差の平均 / 標準誤差 -59.4)、検証用の並べ替えの Recall@10 0.2382(学習前 0.0433)、スコアの標準偏差の中央値 3.6727、clipping の発動 0.63、235 秒
    cross-encoder シード 1(T_2 = 2048、学習率 0.00096): 並べ替えた Recall@10 0.2135(R2 だけ 0.1236、coverage 0.2216)、最後の区間の訓練損失 0.437(学習前のモデルの損失 2.134、差の平均 / 標準誤差 -42.9)、検証用の並べ替えの Recall@10 0.2431(学習前 0.0394)、スコアの標準偏差の中央値 3.4640、clipping の発動 0.93、235 秒
    cross-encoder シード 2(T_2 = 2048、学習率 0.00096): 並べ替えた Recall@10 0.2197(R2 だけ 0.1320、coverage 0.2250)、最後の区間の訓練損失 0.369(学習前のモデルの損失 2.093、差の平均 / 標準誤差 -53.1)、検証用の並べ替えの Recall@10 0.2421(学習前 0.0650)、スコアの標準偏差の中央値 3.0315、clipping の発動 0.55、235 秒
    cross-encoder シード 3(T_2 = 2048、学習率 0.00096): 並べ替えた Recall@10 0.2240(R2 だけ 0.1360、coverage 0.2281)、最後の区間の訓練損失 0.349(学習前のモデルの損失 2.138、差の平均 / 標準誤差 -59.8)、検証用の並べ替えの Recall@10 0.2392(学習前 0.0335)、スコアの標準偏差の中央値 3.7827、clipping の発動 0.82、235 秒
    cross-encoder シード 4(T_2 = 2048、学習率 0.00096): 並べ替えた Recall@10 0.2194(R2 だけ 0.1255、coverage 0.2240)、最後の区間の訓練損失 0.278(学習前のモデルの損失 2.114、差の平均 / 標準誤差 -69.8)、検証用の並べ替えの Recall@10 0.2303(学習前 0.0423)、スコアの標準偏差の中央値 3.3261、clipping の発動 0.65、235 秒
    cross-encoder の本番の学習と評価 19.6 分(見積もり 21.0 分)


### 6.8 実験 A: passage 単位の評価での、関連性の定義の効果


```python
def first_stage_learning_check(conditions: tuple[str, ...], seeds: list[int]) -> dict:
    # 第 1 段の P1(6.1 節): (a)全シードで訓練損失が有限、最後の区間の訓練損失 <= 同じ組での学習前のモデルの損失 x P1_LOSS_RATIO、
    # (b)自分の定義の検証用の Recall@10(R1: 記事単位、R2: passage 単位) >= 学習前の 008 の値 + P1_VALIDATION_GAIN
    untrained = {"R1": float(UNTRAINED_VALIDATION_FIRST_STAGE["article_hits"].mean()), "R2": float(UNTRAINED_VALIDATION_FIRST_STAGE["passage_hits"].mean())}
    own_key = {"R1": "article_hits", "R2": "passage_hits"}
    failed_a, gains, ratios = [], {}, []
    for c in conditions:
        for s in seeds:
            if first_stage_failure_reasons(RUNS[(c, s)]):
                failed_a.append((c, s))
            ratios.append(RUNS[(c, s)]["final_train_loss"] / RUNS[(c, s)]["initial_model_loss_on_final_window"])
        gains[c] = np.array([float(RUNS[(c, s)]["validation"][own_key[c]].mean()) - untrained[c] for s in seeds])
    gain_ok = all(bool((g >= P1_VALIDATION_GAIN).all()) for g in gains.values())
    return {"ok": not failed_a and gain_ok, "failed_a": failed_a, "ratio_max": max(ratios), "gains": gains, "gain_ok": gain_ok}


def describe_first_stage_learning(result: dict) -> str:
    return (
        f"((a)の不成立 {result['failed_a'] or 'なし'}、最後の区間の訓練損失 / 同じ組での学習前のモデルの損失の最大 {result['ratio_max']:.3f} <= {P1_LOSS_RATIO}。"
        f"(b)自分の定義の検証用の Recall@10 の学習前からの増分の最小 " + "、".join(f"{c} {g.min():+.4f}" for c, g in result["gains"].items()) + f" >= {P1_VALIDATION_GAIN}: {result['gain_ok']})"
    )


def passage_recall(condition: str, seeds: int) -> np.ndarray:
    return np.array([RUNS[(condition, s)]["passage_hits"].mean() for s in range(seeds)])


def article_recall(condition: str, seeds: int) -> np.ndarray:
    return np.array([RUNS[(condition, s)]["article_hits"].mean() for s in range(seeds)])


def format024_recall(condition: str, seeds: int) -> np.ndarray:
    return np.array([RUNS[(condition, s)]["format024_hits"].mean() for s in range(seeds)])


def contrast_bootstrap(rows: list[np.ndarray], clusters: np.ndarray, num_first: int) -> np.ndarray:
    # 行 = 条件 x シードの query ごとの 0/1。先頭の num_first 行の平均 - 残りの行の平均の、記事を単位とする対応付きブートストラップの再標本の分布
    hits = np.stack(rows).astype(np.float64)
    boot = paired_cluster_bootstrap_ratio_of_sums(hits, np.ones(hits.shape[1]), clusters, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    return boot[:, :num_first].mean(axis=1) - boot[:, num_first:].mean(axis=1)


_seeds = list(range(SEEDS_A))
R_A_R1, R_A_R2 = passage_recall("R1", SEEDS_A), passage_recall("R2", SEEDS_A)
A_PER_SEED = R_A_R2 - R_A_R1
_boot_a = contrast_bootstrap([RUNS[("R2", s)]["passage_hits"] for s in _seeds] + [RUNS[("R1", s)]["passage_hits"] for s in _seeds], EVALUATION_QUERY_ARTICLES, SEEDS_A)
DELTA_A = float(A_PER_SEED.mean())
SIGMA_A = combined_sigma(A_PER_SEED, _boot_a)
_p1_a = first_stage_learning_check(("R1", "R2"), _seeds)
precondition_status["P1(A)"] = _p1_a["ok"]
precondition_status["P2(A)"] = bool(P2_FLOOR <= R_A_R1.mean() <= P2_CEILING)
A_PRECONDITIONS = ["P0(A)", "P1(A)", "P2(A)"]
A_COMPUTED = judge(DELTA_A, SIGMA_A["sigma"])
A_VERDICT = A_COMPUTED if all(precondition_status[k] for k in A_PRECONDITIONS) else "前提不成立"

print(f"{RUN_TAG}実験 A(シード {_seeds}、T_1 = {FIRST_STAGE_STEPS}、評価用の query {len(EVALUATION_ENTRIES)} 個、索引 {NUM_INDEX:,} passage、ラベル: passage 単位)")
print(f"  passage 単位の Recall@10 R(R1): {rounded(R_A_R1)}、R(R2): {rounded(R_A_R2)}")
print(f"  d_s = R(R2) - R(R1): {rounded(A_PER_SEED)}")
print(
    f"  Delta_A = {DELTA_A:+.4f}、sigma_A = {SIGMA_A['sigma']:.4f}(シード間 {SIGMA_A['seed_term']:.4f}・ブートストラップ {SIGMA_A['bootstrap_term']:.4f}、反復 {BOOTSTRAP_RESAMPLES:,})、閾値 2 sigma_A = {SIGMA_MULTIPLIER * SIGMA_A['sigma']:.4f}"
)
print(
    f"  前提条件: P0 {precondition_status['P0(A)']}、P1 {precondition_status['P1(A)']}{describe_first_stage_learning(_p1_a)}、"
    f"P2 {precondition_status['P2(A)']}(R(R1) の平均 {R_A_R1.mean():.4f} が [{P2_FLOOR}, {P2_CEILING}])"
)
print(f"  判定関数の結果 {A_COMPUTED} -> {RUN_TAG}最終判定: {A_VERDICT}")
print("  診断量(判定なし):")
for _c in ("R1", "R2"):
    _runs = [RUNS[(_c, s)] for s in range(SEEDS_A)]
    print(
        f"    {_c}: 上位 10 件のうち正例と同じ記事の passage の割合 {np.mean([r['same_article_top10_share'] for r in _runs]):.4f}、1 位が同じ記事の別の passage の割合 {np.mean([r['top1_same_article_wrong'] for r in _runs]):.4f}、"
        f"平均逆順位 {np.mean([r['passage_mean_reciprocal_rank'] for r in _runs]):.4f}、Recall@10 のシード間の標準偏差 {passage_recall(_c, SEEDS_A).std(ddof=1) if SEEDS_A >= 2 else float('nan'):.4f}、"
        f"最後の区間の訓練損失 {np.mean([r['final_train_loss'] for r in _runs]):.3f}、学習率 {LEARNING_RATE[_c]:.3g}(出どころ {LEARNING_RATE_SOURCE[_c]})、clipping の発動 {np.mean([r['clip_rate'] for r in _runs]):.2f}"
    )
print(
    f"    参照: 学習前の 008 の passage 単位の Recall@10(評価用) {UNTRAINED_MAIN['passage_hits'].mean():.4f}、ランダムな順位の期待値 {RANDOM_RECALL_PASSAGE:.6f}。"
    f"R1・R2 の記事単位のラベルの Recall@10(実験 B の量): R1 {article_recall('R1', SEEDS_A).mean():.4f}、R2 {article_recall('R2', SEEDS_A).mean():.4f}"
)

_fig, _ax = plt.subplots(figsize=(5.5, 4))
for _x, _c in enumerate(("R1", "R2")):
    _values = passage_recall(_c, SEEDS_A)
    _ax.scatter([_x] * len(_values), _values, alpha=0.6)
    _ax.plot([_x - 0.2, _x + 0.2], [_values.mean()] * 2, "k-")
_ax.axhline(UNTRAINED_MAIN["passage_hits"].mean(), color="gray", linestyle=":", label="untrained 008")
_ax.set_xticks([0, 1])
_ax.set_xticklabels(["R1 (article-level)", "R2 (Inverse Cloze Task)"])
_ax.set_ylabel("Recall@10 (passage-level evaluation)")
_ax.set_title(f"{PLOT_TAG}Experiment A")
_ax.legend(fontsize=7)
plt.tight_layout()
plt.show()
```

    実験 A(シード [0, 1, 2, 3, 4]、T_1 = 1024、評価用の query 3227 個、索引 93,995 passage、ラベル: passage 単位)
      passage 単位の Recall@10 R(R1): [0.0964, 0.097, 0.0964, 0.1001, 0.1004]、R(R2): [0.1348, 0.1236, 0.132, 0.136, 0.1255]
      d_s = R(R2) - R(R1): [0.0384, 0.0267, 0.0356, 0.0359, 0.0251]
      Delta_A = +0.0324、sigma_A = 0.0041(シード間 0.0027・ブートストラップ 0.0031、反復 10,000)、閾値 2 sigma_A = 0.0082
      前提条件: P0 True、P1 True((a)の不成立 なし、最後の区間の訓練損失 / 同じ組での学習前のモデルの損失の最大 0.588 <= 0.9。(b)自分の定義の検証用の Recall@10 の学習前からの増分の最小 R1 +0.1398、R2 +0.1398 >= 0.04: True)、P2 True(R(R1) の平均 0.0980 が [0.02, 0.95])
      判定関数の結果 支持 -> 最終判定: 支持
      診断量(判定なし):
        R1: 上位 10 件のうち正例と同じ記事の passage の割合 0.0403、1 位が同じ記事の別の passage の割合 0.0474、平均逆順位 0.0531、Recall@10 のシード間の標準偏差 0.0020、最後の区間の訓練損失 2.766、学習率 0.00048(出どころ 較正)、clipping の発動 1.00
        R2: 上位 10 件のうち正例と同じ記事の passage の割合 0.0477、1 位が同じ記事の別の passage の割合 0.0554、平均逆順位 0.0740、Recall@10 のシード間の標準偏差 0.0056、最後の区間の訓練損失 2.000、学習率 0.00048(出どころ 較正)、clipping の発動 1.00
        参照: 学習前の 008 の passage 単位の Recall@10(評価用) 0.0762、ランダムな順位の期待値 0.000106。R1・R2 の記事単位のラベルの Recall@10(実験 B の量): R1 0.1518、R2 0.1686



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/026_inverse_cloze_task_and_two_stage_retrieval/output_38_1.png)
    




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/026_inverse_cloze_task_and_two_stage_retrieval.ipynb
