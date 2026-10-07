---
title: "ANN 検索とリランキング / ANN Search and Reranking(実装・実験編 6/8)"
---

この記事は後編(実装・実験編 6/8)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/025_ann_search_and_reranking-practice-5)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/025_ann_search_and_reranking-practice-7)。

### 6.5 学習率の較正

選ばれた計画の $T$ で、6.1 節の規則のとおりに、条件(D1・D2・E2)ごとに学習率を決める。格子は、計画の段階に応じて中心の $\{1/16, 1/4, 1, 4, 16\}$ 倍(5 点)または $\{1/4, 1, 4\}$ 倍(3 点)である。
**候補**: 格子点のうち、学習の成立の前提条件 P1(a)(訓練損失がすべてのステップで有限、最後の区間の損失の差の平均が標準誤差の 2 倍以上低い、最後の区間の訓練損失が $\ln(1 + n)$ 未満)を満たさない学習率は、候補から外す。
**指標** は、残った候補の、較正の query(検証用の記事)の、第 1 段の上位 50 件を並べ替えた後の **平均逆順位(Mean Reciprocal Rank)**(学習の最終ステップの重み、較正専用のシード)で、これが最大の点を選ぶ。
判定の対比量である Recall@10 は選択に使わない。**評価用の記事の query は較正に使わない。** すべての格子点が外れた条件は、P0 を不成立とし、本番の学習には格子の中心の学習率を使う(P0 が不成立なので、その条件を含む実験は前提不成立になる)。
内点かどうかは、外す前の格子(拡張した場合は拡張後)の位置で判定する。

各格子点について、較正の query の平均逆順位(指標)とそのブートストラップの標準偏差、Recall@10 とそのブートストラップの標準偏差、最後の区間の訓練損失と、同じ組での学習前のモデルの損失、診断量として
学習後のモデルの、query ごとの候補 50 件のスコアの標準偏差の中央値(崩壊の検出。判定には使わない)を印字する。候補から外した学習率とその理由も印字する。


```python
_t0_calibration = time.time()
CALIBRATION = {}  # 条件 -> {"grid", "values"(平均逆順位)、"recalls"、"value_sds"、"recall_sds"、"eligible"、"excluded"、"chosen", "chosen_index", "interior", "extended", "fallback", ...}


def calibrate(condition: str) -> dict:
    grid = list(learning_rate_grid_for(condition, PLAN_SETTINGS["grid_points"]))
    records = {}

    def record_of(learning_rate: float) -> dict:
        if learning_rate not in records:
            records[learning_rate] = train_run(condition, CALIBRATION_SEED_INDEX, learning_rate, NUM_STEPS, main=False)
        return records[learning_rate]

    def best(points: list[float]) -> int | None:
        # P1(a) を満たす候補のうち、較正の query の平均逆順位が最大の点(同点なら小さい学習率)。候補がなければ None
        eligible = [learning_precondition(record_of(learning_rate)) for learning_rate in points]
        return best_eligible_index(points, eligible, [records[learning_rate]["validation_mean_reciprocal_rank"] for learning_rate in points])

    index = best(grid)
    extended = None
    if index is not None and index == 0:  # 最良が端: その方向に公比 4 で 1 点だけ拡張する(1 回のみ)
        extended = "lower"
        grid = [grid[0] / LEARNING_RATE_GRID_RATIO] + grid
        index = best(grid)
    elif index is not None and index == len(grid) - 1:
        extended = "upper"
        grid = grid + [grid[-1] * LEARNING_RATE_GRID_RATIO]
        index = best(grid)
    fallback = index is None  # すべての格子点が外れた: 格子の中心を使う(P0 は不成立)
    chosen = LEARNING_RATE_CENTER[condition] if fallback else grid[index]
    excluded = {learning_rate: learning_failure_reasons(records[learning_rate]) for learning_rate in grid if learning_failure_reasons(records[learning_rate])}
    return {
        "grid": grid,
        "values": [records[learning_rate]["validation_mean_reciprocal_rank"] for learning_rate in grid],
        "value_sds": [bootstrap_sd_of_mean(records[learning_rate]["validation_reciprocal_ranks"]) for learning_rate in grid],
        "recalls": [records[learning_rate]["validation_recall"] for learning_rate in grid],
        "recall_sds": [bootstrap_sd_of_mean(records[learning_rate]["validation_hits"]) for learning_rate in grid],
        "score_stds": [records[learning_rate]["validation_score_std_median"] for learning_rate in grid],
        "eligible": [learning_rate not in excluded for learning_rate in grid],
        "excluded": excluded,
        "chosen": chosen,
        "chosen_index": index,
        "interior": index is not None and 0 < index < len(grid) - 1,
        "extended": extended,
        "fallback": fallback,
        "final_train_loss": [records[learning_rate]["final_train_loss"] for learning_rate in grid],
        "initial_loss": [records[learning_rate]["initial_model_loss_on_final_window"] for learning_rate in grid],
    }


for _c in RUN_ORDER:
    CALIBRATION[_c] = calibrate(_c)
    _r = CALIBRATION[_c]
    print(
        f"較正 {_c}: 学習率 {[float(f'{x:.3g}') for x in _r['grid']]} -> 較正の query の平均逆順位(指標){rounded(_r['values'])}(ブートストラップの標準偏差 {rounded(_r['value_sds'])})、"
        f"Recall@10 {rounded(_r['recalls'])}(同 {rounded(_r['recall_sds'])})、"
        f"最後の区間の訓練損失 {rounded(_r['final_train_loss'], 3)}(学習前のモデルの同じ組での損失 {rounded(_r['initial_loss'], 3)})、"
        f"候補 50 件のスコアの標準偏差の中央値(診断量){rounded(_r['score_stds'], 4)}、拡張 {_r['extended'] or 'なし'}"
    )
    for _learning_rate, _reasons in _r["excluded"].items():
        print(f"  候補から外した学習率 {_learning_rate:.3g}: " + "、".join(_reasons))
    print(
        f"  -> 選んだ学習率 {_r['chosen']:.3g}("
        + ("すべての格子点が外れたので、格子の中心を使う。P0 は不成立" if _r["fallback"] else "内点" if _r["interior"] else "端")
        + ")"
    )
LEARNING_RATE = {c: CALIBRATION[c]["chosen"] for c in CONDITIONS}
GRID_MULTIPLIER = {c: LEARNING_RATE[c] / LEARNING_RATE_CENTER[c] for c in CONDITIONS}  # 自分の格子の中心の何倍か
# 較正に関する前提条件: その実験の対比量に入る条件で、選んだ学習率が格子の内点であること(実験 D: D1・D2、実験 E: D1・E2)。候補がない条件は不成立
P0_TARGETS = {"D": ("D1", "D2"), "E": ("D1", "E2")}
for _experiment, _conditions in P0_TARGETS.items():
    precondition_status[f"P0({_experiment})"] = all(CALIBRATION[c]["interior"] for c in _conditions)
CALIBRATION_SECONDS = time.time() - _t0_calibration
print(
    f"学習率: {json.dumps({c: float(f'{v:.3g}') for c, v in LEARNING_RATE.items()})}、自分の格子の中心に対する倍率 {json.dumps({c: float(f'{v:.3g}') for c, v in GRID_MULTIPLIER.items()})}"
)
print(
    f"{RUN_TAG}較正に関する前提条件: "
    + "、".join(f"実験 {e}: {precondition_status[f'P0({e})']}(内点 {json.dumps({c: CALIBRATION[c]['interior'] for c in cs})})" for e, cs in P0_TARGETS.items())
)
print(f"較正 {CALIBRATION_SECONDS / 60:.1f} 分(見積もり {PLAN_ESTIMATES[SELECTED_PLAN['plan']]['calibration'] / 60:.1f} 分、最悪の場合)")
```

    較正 D1: 学習率 [1.5e-05, 6e-05, 0.00024, 0.00096, 0.00384] -> 較正の query の平均逆順位(指標)[0.0697, 0.0704, 0.09, 0.096, 0.0835](ブートストラップの標準偏差 [0.0088, 0.0097, 0.0141, 0.0121, 0.0108])、Recall@10 [0.187, 0.1688, 0.2007, 0.203, 0.2064](同 [0.0218, 0.0207, 0.0232, 0.0211, 0.0231])、最後の区間の訓練損失 [2.082, 2.062, 2.056, 2.076, 2.078](学習前のモデルの同じ組での損失 [2.127, 2.127, 2.127, 2.127, 2.127])、候補 50 件のスコアの標準偏差の中央値(診断量)[0.0978, 0.1248, 0.1868, 0.0455, 0.0281]、拡張 なし
      候補から外した学習率 1.5e-05: 最後の区間の訓練損失 2.0818 が ln(1 + n) = 2.0794 未満でない
      -> 選んだ学習率 0.00096(内点)
    較正 D2: 学習率 [1.5e-05, 6e-05, 0.00024, 0.00096, 0.00384] -> 較正の query の平均逆順位(指標)[0.1549, 0.1669, 0.1659, 0.1353, 0.0817](ブートストラップの標準偏差 [0.019, 0.0196, 0.0191, 0.0177, 0.0085])、Recall@10 [0.2725, 0.2771, 0.2714, 0.2497, 0.1984](同 [0.0241, 0.024, 0.0238, 0.0248, 0.0187])、最後の区間の訓練損失 [1.971, 1.858, 1.754, 1.94, 2.057](学習前のモデルの同じ組での損失 [2.137, 2.137, 2.137, 2.137, 2.137])、候補 50 件のスコアの標準偏差の中央値(診断量)[0.034, 0.0301, 0.037, 0.0218, 0.0059]、拡張 なし
      候補から外した学習率 0.00384: 損失の差の平均 -0.080 が -2 x 標準誤差(0.115)以下でない
      -> 選んだ学習率 6e-05(内点)
    較正 E2: 学習率 [1.5e-05, 6e-05, 0.00024, 0.00096, 0.00384] -> 較正の query の平均逆順位(指標)[0.0904, 0.0942, 0.1037, 0.0897, 0.0917](ブートストラップの標準偏差 [0.0143, 0.0138, 0.0161, 0.0117, 0.0115])、Recall@10 [0.1995, 0.2064, 0.2018, 0.1927, 0.2212](同 [0.024, 0.0244, 0.0246, 0.021, 0.0228])、最後の区間の訓練損失 [1.499, 0.915, 0.527, 1.168, 1.835](学習前のモデルの同じ組での損失 [2.071, 2.071, 2.071, 2.071, 2.071])、候補 50 件のスコアの標準偏差の中央値(診断量)[0.5687, 1.0159, 1.3015, 0.9273, 0.1885]、拡張 なし
      -> 選んだ学習率 0.00024(内点)
    学習率: {"D1": 0.00096, "D2": 6e-05, "E2": 0.00024}、自分の格子の中心に対する倍率 {"D1": 4.0, "D2": 0.25, "E2": 1.0}
    較正に関する前提条件: 実験 D: True(内点 {"D1": true, "D2": true})、実験 E: True(内点 {"D1": true, "E2": true})
    較正 23.1 分(見積もり 27.7 分、最悪の場合)


### 6.6 近似最近傍探索の索引の構築と掃引(実験 A・B・C)

選ばれた計画の $N_{\max}$・シード数で、シードごとに HNSW(挿入の順序の先頭から $N$ の水準ごとに格子を掃引)・転置ファイル(最大の $N$ でリスト数 5 通り)・Product Quantization(最大の $N$)を実行する。
シード 0 では、**階層の効果の診断** として、各水準で最下層だけをランダムな入口から探索する掃引も行い、**実験 F のために、最大の $N$ の索引(HNSW・転置ファイル・Product Quantization)を保持する**。
壁時計時間は判定に使わず、観察として印字する。


```python
_t0_ann = time.time()
HNSW_RUNS, INVERTED_FILE_RUNS, PRODUCT_QUANTIZATION_RUNS = {}, {}, {}
for _s in range(NUM_ANN_SEEDS):
    HNSW_RUNS[_s] = run_hnsw_seed(_s, N_MAX, flat_diagnostic=_s == 0, keep_index=_s == 0)
    INVERTED_FILE_RUNS[_s] = run_inverted_file_seed(_s, N_MAX, keep_index=_s == 0)
    PRODUCT_QUANTIZATION_RUNS[_s] = run_product_quantization_seed(_s, N_MAX, keep_index=_s == 0)
    _h = HNSW_RUNS[_s]
    print(
        f"シード {_s}: HNSW 構築 {_h['build_seconds']:.1f} 秒・掃引 {_h['sweep_seconds']:.1f} 秒、層の大きさ(N_max){_h['layer_sizes'][N_MAX]}、構築の距離計算 {_h['construction_distances'][N_MAX]:,} 回。"
        f"転置ファイル {sum(INVERTED_FILE_RUNS[_s]['fit_seconds'].values()):.1f} 秒(リスト数 {list(INVERTED_FILE_RUNS[_s]['num_lists'])}、空のクラスタの置き直し {sum(INVERTED_FILE_RUNS[_s]['empty_events'].values())} 回)。"
        f"Product Quantization 符号帳の学習 {PRODUCT_QUANTIZATION_RUNS[_s]['fit_seconds']:.1f} 秒(符号 {PRODUCT_QUANTIZATION_RUNS[_s]['code_bytes']} バイト)"
    )
ANN_SECONDS = time.time() - _t0_ann
print(f"近似最近傍探索の索引の構築と掃引 {ANN_SECONDS / 60:.1f} 分(見積もり {PLAN_ESTIMATES[SELECTED_PLAN['plan']]['ann'] / 60:.1f} 分)")
```

    シード 0: HNSW 構築 283.2 秒・掃引 63.5 秒、層の大きさ(N_max)[65536, 13221, 2575, 508, 108, 17, 4]、構築の距離計算 48,670,745 回。転置ファイル 82.3 秒(リスト数 [128, 256, 512, 1024, 2048]、空のクラスタの置き直し 0 回)。Product Quantization 符号帳の学習 40.1 秒(符号 32 バイト)
    シード 1: HNSW 構築 286.0 秒・掃引 58.8 秒、層の大きさ(N_max)[65536, 13069, 2582, 507, 116, 29, 6, 1, 1, 1]、構築の距離計算 48,645,659 回。転置ファイル 85.4 秒(リスト数 [128, 256, 512, 1024, 2048]、空のクラスタの置き直し 2 回)。Product Quantization 符号帳の学習 40.0 秒(符号 32 バイト)
    シード 2: HNSW 構築 280.0 秒・掃引 61.4 秒、層の大きさ(N_max)[65536, 13064, 2637, 484, 103, 21, 4, 1, 1]、構築の距離計算 48,698,384 回。転置ファイル 82.4 秒(リスト数 [128, 256, 512, 1024, 2048]、空のクラスタの置き直し 0 回)。Product Quantization 符号帳の学習 39.2 秒(符号 32 バイト)
    近似最近傍探索の索引の構築と掃引 29.4 分(見積もり 29.2 分)


### 6.7 並べ替えモデルの学習と評価(実験 D・E)

選ばれた計画の条件とシードをすべて学習する。D1(cross-encoder、困難な負例)は実験 D と実験 E で共有し、学習は 1 回だけ行う。同じシードの D1・D2・E2 は、**同じ query と正例を同じ順序で見る**
(D1 と D2 は負例も同じ。E2 だけ、負例がランダム)。評価は並べ替えの query(評価用の記事)で、最終ステップの重みについて行う。


```python
_t0_main = time.time()
RUNS = {}  # (条件, シード) -> 記録
for _c in RUN_ORDER:
    for _s in range(NUM_SEEDS[_c]):
        RUNS[(_c, _s)] = train_run(_c, _s, LEARNING_RATE[_c], NUM_STEPS, main=True)
        _r = RUNS[(_c, _s)]
        _removed = get_schedule(_s, NUM_STEPS)
        print(
            f"{_c} シード {_s}(学習率 {_r['learning_rate']:.3g}): 並べ替えの Recall@10 {_r['hits'].mean():.4f}(較正の query: Recall@10 {_r['validation_recall']:.4f}・平均逆順位 {_r['validation_mean_reciprocal_rank']:.4f})、"
            f"最後の区間の訓練損失 {_r['final_train_loss']:.3f}(同じ組での学習前のモデルの損失 {_r['initial_model_loss_on_final_window']:.3f}、比 {_r['final_train_loss'] / _r['initial_model_loss_on_final_window']:.3f})、"
            f"clipping の発動 {_r['clip_rate']:.2f}、更新のスキップ {_r['skipped_steps']}、候補 50 件のスコアの標準偏差の中央値(診断量){_r['score_std_median']:.4f}、{_r['seconds']:.0f} 秒"
        )
MAIN_SECONDS = time.time() - _t0_main
print(f"本番の学習と評価 {MAIN_SECONDS / 60:.1f} 分(見積もり {PLAN_ESTIMATES[SELECTED_PLAN['plan']]['main'] / 60:.1f} 分)")
_removed_totals = [get_schedule(s, NUM_STEPS) for s in range(max(NUM_SEEDS.values()))]
print(
    f"負例の採掘(T = {NUM_STEPS}、{BATCH_QUERIES} query x {NUM_STEPS} ステップ x 上位 {MINING_DEPTH} 件 / シード): 同じ記事の passage として除いた件数は "
    + "、".join(f"シード {s}: {sc.num_same_article_removed:,} / {sc.num_mined_considered:,}({sc.num_same_article_removed / sc.num_mined_considered:.2%})" for s, sc in enumerate(_removed_totals))
    + f"。除くと負例が {NUM_NEGATIVES} 個に足りずランダムな負例で補った組: " + "、".join(f"{sc.num_fallback_queries}" for sc in _removed_totals)
)
```

    D1 シード 0(学習率 0.00096): 並べ替えの Recall@10 0.1708(較正の query: Recall@10 0.1847・平均逆順位 0.0781)、最後の区間の訓練損失 2.078(同じ組での学習前のモデルの損失 2.114、比 0.983)、clipping の発動 0.08、更新のスキップ 0、候補 50 件のスコアの標準偏差の中央値(診断量)0.0457、136 秒
    D1 シード 1(学習率 0.00096): 並べ替えの Recall@10 0.1791(較正の query: Recall@10 0.1984・平均逆順位 0.0870)、最後の区間の訓練損失 2.078(同じ組での学習前のモデルの損失 2.092、比 0.993)、clipping の発動 0.18、更新のスキップ 1、候補 50 件のスコアの標準偏差の中央値(診断量)0.0696、137 秒
    D1 シード 2(学習率 0.00096): 並べ替えの Recall@10 0.1722(較正の query: Recall@10 0.2189・平均逆順位 0.0983)、最後の区間の訓練損失 2.072(同じ組での学習前のモデルの損失 2.086、比 0.993)、clipping の発動 0.38、更新のスキップ 0、候補 50 件のスコアの標準偏差の中央値(診断量)0.0721、136 秒
    D1 シード 3(学習率 0.00096): 並べ替えの Recall@10 0.1584(較正の query: Recall@10 0.2030・平均逆順位 0.0851)、最後の区間の訓練損失 2.075(同じ組での学習前のモデルの損失 2.111、比 0.983)、clipping の発動 0.31、更新のスキップ 2、候補 50 件のスコアの標準偏差の中央値(診断量)0.0486、136 秒
    D1 シード 4(学習率 0.00096): 並べ替えの Recall@10 0.1757(較正の query: Recall@10 0.2166・平均逆順位 0.0909)、最後の区間の訓練損失 2.059(同じ組での学習前のモデルの損失 2.109、比 0.976)、clipping の発動 0.26、更新のスキップ 0、候補 50 件のスコアの標準偏差の中央値(診断量)0.1197、136 秒
    D2 シード 0(学習率 6e-05): 並べ替えの Recall@10 0.2351(較正の query: Recall@10 0.2794・平均逆順位 0.1645)、最後の区間の訓練損失 1.919(同じ組での学習前のモデルの損失 2.253、比 0.852)、clipping の発動 1.00、更新のスキップ 0、候補 50 件のスコアの標準偏差の中央値(診断量)0.0296、107 秒
    D2 シード 1(学習率 6e-05): 並べ替えの Recall@10 0.2317(較正の query: Recall@10 0.2714・平均逆順位 0.1741)、最後の区間の訓練損失 1.854(同じ組での学習前のモデルの損失 2.061、比 0.899)、clipping の発動 1.00、更新のスキップ 0、候補 50 件のスコアの標準偏差の中央値(診断量)0.0304、99 秒
    D2 シード 2(学習率 6e-05): 並べ替えの Recall@10 0.2296(較正の query: Recall@10 0.2737・平均逆順位 0.1655)、最後の区間の訓練損失 1.821(同じ組での学習前のモデルの損失 2.072、比 0.879)、clipping の発動 1.00、更新のスキップ 0、候補 50 件のスコアの標準偏差の中央値(診断量)0.0294、99 秒
    D2 シード 3(学習率 6e-05): 並べ替えの Recall@10 0.2344(較正の query: Recall@10 0.2816・平均逆順位 0.1682)、最後の区間の訓練損失 1.853(同じ組での学習前のモデルの損失 2.052、比 0.903)、clipping の発動 1.00、更新のスキップ 0、候補 50 件のスコアの標準偏差の中央値(診断量)0.0301、99 秒
    D2 シード 4(学習率 6e-05): 並べ替えの Recall@10 0.2379(較正の query: Recall@10 0.2782・平均逆順位 0.1671)、最後の区間の訓練損失 1.895(同じ組での学習前のモデルの損失 2.148、比 0.882)、clipping の発動 1.00、更新のスキップ 0、候補 50 件のスコアの標準偏差の中央値(診断量)0.0294、99 秒
    E2 シード 0(学習率 0.00024): 並べ替えの Recall@10 0.1701(較正の query: Recall@10 0.2144・平均逆順位 0.1086)、最後の区間の訓練損失 0.510(同じ組での学習前のモデルの損失 2.168、比 0.235)、clipping の発動 1.00、更新のスキップ 2、候補 50 件のスコアの標準偏差の中央値(診断量)1.3394、153 秒
    E2 シード 1(学習率 0.00024): 並べ替えの Recall@10 0.1812(較正の query: Recall@10 0.2166・平均逆順位 0.0994)、最後の区間の訓練損失 0.493(同じ組での学習前のモデルの損失 2.069、比 0.239)、clipping の発動 1.00、更新のスキップ 2、候補 50 件のスコアの標準偏差の中央値(診断量)1.2784、153 秒
    E2 シード 2(学習率 0.00024): 並べ替えの Recall@10 0.1736(較正の query: Recall@10 0.2030・平均逆順位 0.0986)、最後の区間の訓練損失 0.657(同じ組での学習前のモデルの損失 2.139、比 0.307)、clipping の発動 1.00、更新のスキップ 3、候補 50 件のスコアの標準偏差の中央値(診断量)1.2577、153 秒
    E2 シード 3(学習率 0.00024): 並べ替えの Recall@10 0.1750(較正の query: Recall@10 0.1881・平均逆順位 0.0937)、最後の区間の訓練損失 0.636(同じ組での学習前のモデルの損失 2.119、比 0.300)、clipping の発動 1.00、更新のスキップ 2、候補 50 件のスコアの標準偏差の中央値(診断量)1.2114、153 秒
    E2 シード 4(学習率 0.00024): 並べ替えの Recall@10 0.1812(較正の query: Recall@10 0.1961・平均逆順位 0.0888)、最後の区間の訓練損失 0.437(同じ組での学習前のモデルの損失 2.158、比 0.203)、clipping の発動 1.00、更新のスキップ 2、候補 50 件のスコアの標準偏差の中央値(診断量)1.2822、153 秒
    本番の学習と評価 32.5 分(見積もり 33.2 分)
    負例の採掘(T = 1024、8 query x 1024 ステップ x 上位 50 件 / シード): 同じ記事の passage として除いた件数は シード 0: 23,251 / 409,600(5.68%)、シード 1: 23,350 / 409,600(5.70%)、シード 2: 22,866 / 409,600(5.58%)、シード 3: 23,197 / 409,600(5.66%)、シード 4: 23,541 / 409,600(5.75%)。除くと負例が 7 個に足りずランダムな負例で補った組: 0、0、0、0、0


### 6.8 実験 A: HNSW の探索の費用の件数への依存性


```python
# ブートストラップの重み(記事を単位とする。全シード・全水準・全索引に同じ再標本を使う)
ANN_BOOTSTRAP_WEIGHTS = cluster_bootstrap_weights(ANN_QUERY_ARTICLES, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
_ONES = np.ones((1, len(ANN_QUERY_INDEX)))


def log_cost_of(sweep: dict, weights: np.ndarray) -> np.ndarray:
    # 掃引の結果から、再標本ごとの目標の recall での費用の対数(補間できなければ NaN)
    recalls, costs = weighted_curve(sweep["recall"], sweep["cost"], weights)
    return interpolate_log_cost_batch(recalls, costs, TARGET_RECALL)


_seeds = list(range(NUM_ANN_SEEDS))
A_LOG_COST = np.array([[log_cost_of(HNSW_RUNS[s]["sweeps"][n], _ONES)[0] for n in SIZE_LEVELS] for s in _seeds])  # (シード, 水準)
A_SLOPES = fit_log_log_slope(SIZE_LEVELS, A_LOG_COST)  # シードごとのべき指数 b
A_PER_SEED = 1.0 - A_SLOPES  # d_s = 1 - b_s
DELTA_A = float(A_PER_SEED.mean()) if not np.isnan(A_LOG_COST).any() else float("nan")
_boot_log_cost = np.stack(
    [np.stack([log_cost_of(HNSW_RUNS[s]["sweeps"][n], ANN_BOOTSTRAP_WEIGHTS) for n in SIZE_LEVELS], axis=1) for s in _seeds], axis=1
)  # (再標本, シード, 水準)
A_BOOT_NAN_FRACTION = float(np.isnan(_boot_log_cost).any(axis=(1, 2)).mean())
_boot_delta = 1.0 - fit_log_log_slope(SIZE_LEVELS, _boot_log_cost).mean(axis=1)  # 再標本ごとの対比量(シード平均)
A_BOOT_DELTA = _boot_delta[~np.isnan(_boot_delta)]
SIGMA_A = combined_sigma(A_PER_SEED, A_BOOT_DELTA) if len(A_BOOT_DELTA) > 1 and len(_seeds) > 1 else {"sigma": float("nan"), "seed_term": float("nan"), "bootstrap_term": float("nan")}

# 前提条件 P1(A): すべてのシード・水準で、格子が目標の recall を挟む(補間できる)。再標本で補間できない割合が 1% 以下
_bracketed = not np.isnan(A_LOG_COST).any()
precondition_status["P1(A)"] = bool(_bracketed and A_BOOT_NAN_FRACTION <= 0.01)
A_PRECONDITIONS = ["P1(A)"]
A_COMPUTED = judge(DELTA_A, SIGMA_A["sigma"]) if _bracketed else "判定不能"
A_VERDICT = A_COMPUTED if all(precondition_status[k] for k in A_PRECONDITIONS) else "前提不成立"

print(f"{RUN_TAG}実験 A(シード {_seeds}、N の水準 {SIZE_LEVELS}、N_max = {N_MAX}、query {len(ANN_QUERY_INDEX)} 個、目標の recall {TARGET_RECALL})")
print("  目標の recall での費用 c(1 query あたりの距離計算の回数、シード平均の幾何平均): " + "、".join(f"N = {n}: {math.exp(float(np.mean(A_LOG_COST[:, k]))):.1f}" for k, n in enumerate(SIZE_LEVELS)))
print(f"  シードごとのべき指数 b_s: {rounded(A_SLOPES)}、d_s = 1 - b_s: {rounded(A_PER_SEED)}")
print(
    f"  Delta_A = {DELTA_A:+.4f}、sigma_A = {SIGMA_A['sigma']:.4f}(シード間 {SIGMA_A['seed_term']:.4f}・ブートストラップ {SIGMA_A['bootstrap_term']:.4f}、反復 {BOOTSTRAP_RESAMPLES:,}、補間できない再標本の割合 {A_BOOT_NAN_FRACTION:.4f})、"
    f"閾値 2 sigma_A = {SIGMA_MULTIPLIER * SIGMA_A['sigma']:.4f}"
)
print(f"  前提条件: P1(A) {precondition_status['P1(A)']}(すべてのシード・水準で補間できる: {_bracketed}、再標本で補間できない割合 {A_BOOT_NAN_FRACTION:.4f} <= 0.01)")
print(f"  判定関数の結果 {A_COMPUTED} -> {RUN_TAG}最終判定: {A_VERDICT}")

# 診断量(判定なし)
_cost_by_level = np.exp(A_LOG_COST)
_residual = A_LOG_COST - (A_LOG_COST.mean(axis=1, keepdims=True) + A_SLOPES[:, None] * (np.log(SIZE_LEVELS) - np.log(SIZE_LEVELS).mean())[None, :])
print("  診断量(判定なし):")
print("    全探索の費用 N との比 c / N(シード平均): " + "、".join(f"N = {n}: {float(np.mean(_cost_by_level[:, k]) / n):.4f}" for k, n in enumerate(SIZE_LEVELS)))
print(f"    べき乗則のあてはめの残差(対数の費用の、あてはめとの差の二乗平均平方根。シードごと): {rounded(np.sqrt((_residual**2).mean(axis=1)))}")
print("    HNSW の構築の距離計算(シード 0、累積): " + "、".join(f"N = {n}: {HNSW_RUNS[0]['construction_distances'][n]:,}" for n in SIZE_LEVELS) + f"、層の大きさ(N_max){HNSW_RUNS[0]['layer_sizes'][N_MAX]}")
_flat_rows = []
for _k, _n in enumerate(SIZE_LEVELS):
    _flat = log_cost_of(HNSW_RUNS[0]["flat_sweeps"][_n], _ONES)[0]
    _flat_rows.append(f"N = {_n}: " + (f"{math.exp(_flat):.1f}(階層ありの {math.exp(A_LOG_COST[0, _k]):.1f} の {math.exp(_flat) / math.exp(A_LOG_COST[0, _k]):.2f} 倍)" if not np.isnan(_flat) else "格子の範囲で目標の recall に届かない(または最小で超える)"))
print("    階層の効果(シード 0): 最下層だけをランダムな入口から探索したときの目標の recall での費用: " + "、".join(_flat_rows))
print(
    f"    壁時計時間(観察): HNSW の構築 " + "、".join(f"シード {s}: {HNSW_RUNS[s]['build_seconds']:.1f} 秒" for s in _seeds) + "。判定には使わない"
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for _s in _seeds:
    axes[0].plot(SIZE_LEVELS, np.exp(A_LOG_COST[_s]), "o-", label=f"seed {_s} (b = {A_SLOPES[_s]:.3f})")
axes[0].plot(SIZE_LEVELS, SIZE_LEVELS, "k--", label="exhaustive search (cost = N)")
axes[0].set_xscale("log", base=2)
axes[0].set_yscale("log", base=2)
axes[0].set_xlabel("N")
axes[0].set_ylabel("distance computations per query at recall 0.9")
axes[0].set_title(f"{PLOT_TAG}HNSW: cost at target recall")
axes[0].legend(fontsize=7)
for _n in SIZE_LEVELS:
    _sweep = HNSW_RUNS[0]["sweeps"][_n]
    axes[1].plot(_sweep["recall"].mean(axis=1), _sweep["cost"].mean(axis=1), "o-", markersize=3, label=f"N = {_n}")
axes[1].axvline(TARGET_RECALL, color="gray", linestyle=":")
axes[1].set_yscale("log")
axes[1].set_xlabel("recall (seed 0)")
axes[1].set_ylabel("distance computations per query")
axes[1].set_title(f"{PLOT_TAG}recall-cost curves")
axes[1].legend(fontsize=7)
plt.tight_layout()
plt.show()
```

    実験 A(シード [0, 1, 2]、N の水準 [4096, 8192, 16384, 32768, 65536]、N_max = 65536、query 890 個、目標の recall 0.9)
      目標の recall での費用 c(1 query あたりの距離計算の回数、シード平均の幾何平均): N = 4096: 156.8、N = 8192: 197.3、N = 16384: 244.5、N = 32768: 308.4、N = 65536: 399.4
      シードごとのべき指数 b_s: [0.3342, 0.334, 0.3346]、d_s = 1 - b_s: [0.6658, 0.666, 0.6654]
      Delta_A = +0.6657、sigma_A = 0.0102(シード間 0.0002・ブートストラップ 0.0102、反復 10,000、補間できない再標本の割合 0.0000)、閾値 2 sigma_A = 0.0204
      前提条件: P1(A) True(すべてのシード・水準で補間できる: True、再標本で補間できない割合 0.0000 <= 0.01)
      判定関数の結果 支持 -> 最終判定: 支持
      診断量(判定なし):
        全探索の費用 N との比 c / N(シード平均): N = 4096: 0.0383、N = 8192: 0.0241、N = 16384: 0.0149、N = 32768: 0.0094、N = 65536: 0.0061
        べき乗則のあてはめの残差(対数の費用の、あてはめとの差の二乗平均平方根。シードごと): [0.0129, 0.0121, 0.0156]
        HNSW の構築の距離計算(シード 0、累積): N = 4096: 2,453,609、N = 8192: 5,239,088、N = 16384: 11,070,203、N = 32768: 23,232,288、N = 65536: 48,670,745、層の大きさ(N_max)[65536, 13221, 2575, 508, 108, 17, 4]
        階層の効果(シード 0): 最下層だけをランダムな入口から探索したときの目標の recall での費用: N = 4096: 161.0(階層ありの 156.9 の 1.03 倍)、N = 8192: 208.9(階層ありの 202.4 の 1.03 倍)、N = 16384: 255.3(階層ありの 252.0 の 1.01 倍)、N = 32768: 327.8(階層ありの 310.3 の 1.06 倍)、N = 65536: 414.6(階層ありの 403.5 の 1.03 倍)
        壁時計時間(観察): HNSW の構築 シード 0: 283.2 秒、シード 1: 286.0 秒、シード 2: 280.0 秒。判定には使わない



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/025_ann_search_and_reranking/output_44_1.png)
    


### 6.9 実験 B: HNSW と転置ファイルの比較


```python
_inverted_file_options = sorted(INVERTED_FILE_RUNS[0]["num_lists"])


def best_option(row: np.ndarray) -> int:
    # 費用が最小の選択肢の位置(すべて補間できなければ中央)
    return int(np.nanargmin(row)) if not np.isnan(row).all() else len(row) // 2
B_IVF_LOG_COST = np.array([[log_cost_of(INVERTED_FILE_RUNS[s]["sweeps"][k], _ONES)[0] for k in _inverted_file_options] for s in _seeds])  # (シード, リスト数の選択肢)
B_HNSW_LOG_COST = A_LOG_COST[:, -1]  # N_max での HNSW(実験 A と同じ設定。調整しない)
_bracketed_b = bool(not np.isnan(B_IVF_LOG_COST).any() and not np.isnan(B_HNSW_LOG_COST).any())
_inverted_file_all_nan = np.isnan(B_IVF_LOG_COST).all(axis=1)
B_PER_SEED = np.where(_inverted_file_all_nan, np.nan, np.nanmin(np.where(np.isnan(B_IVF_LOG_COST), np.inf, B_IVF_LOG_COST), axis=1)) - B_HNSW_LOG_COST  # d_s = log(転置ファイルの最小の費用)- log(HNSW の費用)。転置ファイルはリスト数の選択肢の中で費用が最小のもの
DELTA_B = float(B_PER_SEED.mean()) if not np.isnan(B_PER_SEED).any() else float("nan")
_inverted_file_boot = np.stack([np.stack([log_cost_of(INVERTED_FILE_RUNS[s]["sweeps"][k], ANN_BOOTSTRAP_WEIGHTS) for k in _inverted_file_options], axis=1) for s in _seeds], axis=1)  # (再標本, シード, 選択肢)
_hnsw_boot = _boot_log_cost[:, :, -1]  # (再標本, シード)
B_BOOT_NAN_FRACTION = float((np.isnan(_inverted_file_boot).any(axis=(1, 2)) | np.isnan(_hnsw_boot).any(axis=1)).mean())
_boot_delta_b = (np.where(np.isnan(_inverted_file_boot).all(axis=2), np.nan, np.nanmin(np.where(np.isnan(_inverted_file_boot), np.inf, _inverted_file_boot), axis=2)) - _hnsw_boot).mean(axis=1)  # 再標本ごとに、リスト数の最小をとり直す
B_BOOT_DELTA = _boot_delta_b[~np.isnan(_boot_delta_b)]
SIGMA_B = combined_sigma(B_PER_SEED, B_BOOT_DELTA) if len(_seeds) > 1 else {"sigma": float("nan"), "seed_term": float("nan"), "bootstrap_term": float("nan")}
precondition_status["P1(B)"] = bool(_bracketed_b and B_BOOT_NAN_FRACTION <= 0.01)
# P0(B): 目標の recall での費用の対数のシード平均が最小のリスト数が、選択肢の端でない(内点である)こと
_mean_log_cost_by_lists = B_IVF_LOG_COST.mean(axis=0) if not np.isnan(B_IVF_LOG_COST).any() else None
_best_lists_by_mean = int(np.argmin(_mean_log_cost_by_lists)) if _mean_log_cost_by_lists is not None else None
precondition_status["P0(B)"] = bool(_best_lists_by_mean is not None and 0 < _best_lists_by_mean < len(_inverted_file_options) - 1)
B_PRECONDITIONS = ["P0(B)", "P1(B)"]
B_COMPUTED = judge(DELTA_B, SIGMA_B["sigma"]) if _bracketed_b else "判定不能"
B_VERDICT = B_COMPUTED if all(precondition_status[k] for k in B_PRECONDITIONS) else "前提不成立"

print(f"{RUN_TAG}実験 B(シード {_seeds}、N = N_max = {N_MAX}、転置ファイルのリスト数 {_inverted_file_options}、query {len(ANN_QUERY_INDEX)} 個)")
print(
    f"  目標の recall での費用(シード平均の幾何平均): HNSW {math.exp(float(B_HNSW_LOG_COST.mean())):.1f}、転置ファイル "
    + "、".join(f"リスト数 {k}: {math.exp(float(np.nanmean(B_IVF_LOG_COST[:, j]))):.1f}" for j, k in enumerate(_inverted_file_options))
    + f"。転置ファイルの値(シードごとに最小のリスト数を選んだもの): {rounded(np.exp(B_PER_SEED + B_HNSW_LOG_COST), 1)}"
)
print(f"  d_s = log c_1 - log c_2(c_1: 転置ファイル、c_2: HNSW): {rounded(B_PER_SEED)}、シードごとに選ばれたリスト数 {[_inverted_file_options[best_option(row)] for row in B_IVF_LOG_COST]}")
print(
    f"  Delta_B = {DELTA_B:+.4f}、sigma_B = {SIGMA_B['sigma']:.4f}(シード間 {SIGMA_B['seed_term']:.4f}・ブートストラップ {SIGMA_B['bootstrap_term']:.4f}、反復 {BOOTSTRAP_RESAMPLES:,}、補間できない再標本の割合 {B_BOOT_NAN_FRACTION:.4f})、"
    f"閾値 2 sigma_B = {SIGMA_MULTIPLIER * SIGMA_B['sigma']:.4f}"
)
print(
    f"  前提条件: P0(B) {precondition_status['P0(B)']}(費用のシード平均が最小のリスト数 "
    + (f"{_inverted_file_options[_best_lists_by_mean]}(5 点のうち {_best_lists_by_mean + 1} 番目。内点は 2〜4 番目)" if _best_lists_by_mean is not None else "未定義(補間できない選択肢がある)")
    + ")"
)
print(f"  前提条件: P1(B) {precondition_status['P1(B)']}(HNSW・転置ファイルのすべての選択肢・シードで補間できる: {_bracketed_b}、再標本で補間できない割合 {B_BOOT_NAN_FRACTION:.4f} <= 0.01)")
print(f"  判定関数の結果 {B_COMPUTED} -> {RUN_TAG}最終判定: {B_VERDICT}")
print("  診断量(判定なし):")
print(
    "    転置ファイルの走査したベクトルの割合(目標の recall での費用 / N): " + "、".join(f"リスト数 {k}: {math.exp(float(np.nanmean(B_IVF_LOG_COST[:, j]))) / N_MAX:.4f}" for j, k in enumerate(_inverted_file_options))
    + f"、HNSW: {math.exp(float(B_HNSW_LOG_COST.mean())) / N_MAX:.4f}"
)
print("    リストの大きさ(最小・平均・最大。シード 0): " + "、".join(f"リスト数 {k}: {INVERTED_FILE_RUNS[0]['list_size_stats'][k][0]}・{INVERTED_FILE_RUNS[0]['list_size_stats'][k][1]:.1f}・{INVERTED_FILE_RUNS[0]['list_size_stats'][k][2]}" for k in _inverted_file_options))
print(
    "    壁時計時間(観察、判定には使わない): 転置ファイルの学習と掃引 " + "、".join(f"シード {s}: {sum(INVERTED_FILE_RUNS[s]['fit_seconds'].values()):.1f} 秒(学習のみ)" for s in _seeds) + "。HNSW の構築 "
    + "、".join(f"シード {s}: {HNSW_RUNS[s]['build_seconds']:.1f} 秒" for s in _seeds)
)

fig, ax = plt.subplots(figsize=(6, 4))
for j, k in enumerate(_inverted_file_options):
    _sweep = INVERTED_FILE_RUNS[0]["sweeps"][k]
    ax.plot(_sweep["recall"].mean(axis=1), _sweep["cost"].mean(axis=1), "o-", markersize=3, label=f"inverted file, {k} lists")
_sweep = HNSW_RUNS[0]["sweeps"][N_MAX]
ax.plot(_sweep["recall"].mean(axis=1), _sweep["cost"].mean(axis=1), "s-", markersize=3, color="black", label="HNSW")
ax.axvline(TARGET_RECALL, color="gray", linestyle=":")
ax.set_yscale("log")
ax.set_xlabel("recall (seed 0)")
ax.set_ylabel("distance computations per query")
ax.set_title(f"{PLOT_TAG}N = {N_MAX}: recall-cost curves")
ax.legend(fontsize=7)
plt.tight_layout()
plt.show()
```

    実験 B(シード [0, 1, 2]、N = N_max = 65536、転置ファイルのリスト数 [128, 256, 512, 1024, 2048]、query 890 個)
      目標の recall での費用(シード平均の幾何平均): HNSW 399.4、転置ファイル リスト数 128: 2624.5、リスト数 256: 1946.0、リスト数 512: 1675.4、リスト数 1024: 1854.4、リスト数 2048: 2674.5。転置ファイルの値(シードごとに最小のリスト数を選んだもの): [1664.2, 1714.5, 1648.1]
      d_s = log c_1 - log c_2(c_1: 転置ファイル、c_2: HNSW): [1.417, 1.4732, 1.4114]、シードごとに選ばれたリスト数 [512, 512, 512]
      Delta_B = +1.4339、sigma_B = 0.0307(シード間 0.0198・ブートストラップ 0.0234、反復 10,000、補間できない再標本の割合 0.0000)、閾値 2 sigma_B = 0.0613
      前提条件: P0(B) True(費用のシード平均が最小のリスト数 512(5 点のうち 3 番目。内点は 2〜4 番目))
      前提条件: P1(B) True(HNSW・転置ファイルのすべての選択肢・シードで補間できる: True、再標本で補間できない割合 0.0000 <= 0.01)
      判定関数の結果 支持 -> 最終判定: 支持
      診断量(判定なし):
        転置ファイルの走査したベクトルの割合(目標の recall での費用 / N): リスト数 128: 0.0400、リスト数 256: 0.0297、リスト数 512: 0.0256、リスト数 1024: 0.0283、リスト数 2048: 0.0408、HNSW: 0.0061
        リストの大きさ(最小・平均・最大。シード 0): リスト数 128: 63・512.0・1543、リスト数 256: 29・256.0・895、リスト数 512: 5・128.0・419、リスト数 1024: 1・64.0・380、リスト数 2048: 1・32.0・143
        壁時計時間(観察、判定には使わない): 転置ファイルの学習と掃引 シード 0: 82.3 秒(学習のみ)、シード 1: 85.4 秒(学習のみ)、シード 2: 82.4 秒(学習のみ)。HNSW の構築 シード 0: 283.2 秒、シード 1: 286.0 秒、シード 2: 280.0 秒



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/025_ann_search_and_reranking/output_46_1.png)
    


### 6.10 実験 C: Product Quantization の非対称距離計算と対称距離計算


```python
C_ADC = np.stack([PRODUCT_QUANTIZATION_RUNS[s]["recall"]["asymmetric"] for s in _seeds])  # (シード, query)
C_SDC = np.stack([PRODUCT_QUANTIZATION_RUNS[s]["recall"]["symmetric"] for s in _seeds])
C_PER_SEED = C_ADC.mean(axis=1) - C_SDC.mean(axis=1)  # d_s = recall(非対称)- recall(対称)
DELTA_C = float(C_PER_SEED.mean())
_boot_delta_c = ((ANN_BOOTSTRAP_WEIGHTS @ (C_ADC - C_SDC).T) / ANN_BOOTSTRAP_WEIGHTS.sum(axis=1, keepdims=True)).mean(axis=1)
SIGMA_C = combined_sigma(C_PER_SEED, _boot_delta_c) if len(_seeds) > 1 else {"sigma": float("nan"), "seed_term": float("nan"), "bootstrap_term": float("nan")}
C_SDC_MEAN = float(C_SDC.mean())
precondition_status["P2(C)"] = bool(0.10 <= C_SDC_MEAN <= 0.90)  # 改善の余地: 対称距離計算の recall が床にも天井にも達していない
C_PRECONDITIONS = ["P2(C)"]
C_COMPUTED = judge(DELTA_C, SIGMA_C["sigma"])
C_VERDICT = C_COMPUTED if all(precondition_status[k] for k in C_PRECONDITIONS) else "前提不成立"

print(f"{RUN_TAG}実験 C(シード {_seeds}、N = N_max = {N_MAX}、m = {PRODUCT_QUANTIZATION_SUBVECTORS}、k* = {PRODUCT_QUANTIZATION_CODEWORDS}: 符号 {PRODUCT_QUANTIZATION_SUBVECTORS} バイト = {EMBEDDING_DIMENSION * 4} バイトの {PRODUCT_QUANTIZATION_SUBVECTORS} 分の 1、query {len(ANN_QUERY_INDEX)} 個)")
print(f"  recall(上位 10 件の一致率)非対称距離計算: {rounded(C_ADC.mean(axis=1))}、対称距離計算: {rounded(C_SDC.mean(axis=1))}")
print(f"  d_s = recall(非対称)- recall(対称): {rounded(C_PER_SEED)}")
print(
    f"  Delta_C = {DELTA_C:+.4f}、sigma_C = {SIGMA_C['sigma']:.4f}(シード間 {SIGMA_C['seed_term']:.4f}・ブートストラップ {SIGMA_C['bootstrap_term']:.4f}、反復 {BOOTSTRAP_RESAMPLES:,})、"
    f"閾値 2 sigma_C = {SIGMA_MULTIPLIER * SIGMA_C['sigma']:.4f}"
)
print(f"  前提条件: P2(C) {precondition_status['P2(C)']}(対称距離計算の recall の平均 {C_SDC_MEAN:.4f} が 0.10 以上 0.90 以下)")
print(f"  判定関数の結果 {C_COMPUTED} -> {RUN_TAG}最終判定: {C_VERDICT}")
print("  診断量(判定なし):")
_diag = {k: np.mean([PRODUCT_QUANTIZATION_RUNS[s]["diagnostics"][k] for s in _seeds]) for k in PRODUCT_QUANTIZATION_RUNS[0]["diagnostics"]}
print(
    f"    距離の二乗誤差(20,000 組の平均、シード平均): 量子化の平均二乗誤差 D = {_diag['D']:.4f}、query の量子化の誤差 E||e_x||^2 = {_diag['query_error']:.4f}(D の {_diag['query_error'] / _diag['D']:.2f} 倍)。"
    f"推定の偏り(推定 - 真の二乗距離の平均): 非対称 {_diag['bias_asymmetric']:+.4f}(理論 {-_diag['D']:+.4f} = -D)、対称 {_diag['bias_symmetric']:+.4f}(理論 {-(_diag['query_error'] + _diag['D']):+.4f} = -(E||e_x||^2 + D))。"
    f"平均二乗誤差: 非対称 {_diag['mse_asymmetric']:.5f}、対称 {_diag['mse_symmetric']:.5f}(比 {_diag['mse_symmetric'] / _diag['mse_asymmetric']:.2f})"
)
print("    再採点(上位 R 件を元のベクトルで再採点したときの recall、シード平均): " + "、".join(
    f"{name} R = {r}: {np.mean([PRODUCT_QUANTIZATION_RUNS[s]['rescored_recall'][name][r].mean() for s in _seeds]):.4f}" for name in ("asymmetric", "symmetric") for r in PRODUCT_QUANTIZATION_RESCORE_COUNTS
))
_cost = PRODUCT_QUANTIZATION_RUNS[0]["cost"]["asymmetric"]
_cost_symmetric = PRODUCT_QUANTIZATION_RUNS[0]["cost"]["symmetric"]
print(
    f"    費用(1 query あたり): 非対称 参照表の作成 {_cost['table_entries']:.0f}・表引き {_cost['table_lookups']:.0f}、対称 query の量子化 {_cost_symmetric['table_entries']:.0f}・表引き {_cost_symmetric['table_lookups']:.0f}。"
    f"1 ベクトルあたりのバイト数 {PRODUCT_QUANTIZATION_RUNS[0]['code_bytes']}(元は {EMBEDDING_DIMENSION * 4}、圧縮率 {EMBEDDING_DIMENSION * 4 // PRODUCT_QUANTIZATION_RUNS[0]['code_bytes']} 倍)"
)

fig, ax = plt.subplots(figsize=(6, 4))
_x = np.arange(len(PRODUCT_QUANTIZATION_RESCORE_COUNTS) + 1)
for name, marker in (("asymmetric", "o"), ("symmetric", "s")):
    _values = [C_ADC.mean() if name == "asymmetric" else C_SDC.mean()] + [np.mean([PRODUCT_QUANTIZATION_RUNS[s]["rescored_recall"][name][r].mean() for s in _seeds]) for r in PRODUCT_QUANTIZATION_RESCORE_COUNTS]
    ax.plot(_x, _values, marker + "-", label=name)
ax.set_xticks(_x)
ax.set_xticklabels(["none"] + [str(r) for r in PRODUCT_QUANTIZATION_RESCORE_COUNTS])
ax.set_xlabel("number of re-scored candidates R")
ax.set_ylabel("recall (top 10 vs exact)")
ax.set_title(f"{PLOT_TAG}Product Quantization: asymmetric vs symmetric")
ax.legend()
plt.tight_layout()
plt.show()
```

    実験 C(シード [0, 1, 2]、N = N_max = 65536、m = 32、k* = 256: 符号 32 バイト = 1024 バイトの 32 分の 1、query 890 個)
      recall(上位 10 件の一致率)非対称距離計算: [0.516, 0.5121, 0.5148]、対称距離計算: [0.3861, 0.3825, 0.3856]
      d_s = recall(非対称)- recall(対称): [0.1299, 0.1297, 0.1292]
      Delta_C = +0.1296、sigma_C = 0.0033(シード間 0.0002・ブートストラップ 0.0033、反復 10,000)、閾値 2 sigma_C = 0.0067
      前提条件: P2(C) True(対称距離計算の recall の平均 0.3847 が 0.10 以上 0.90 以下)
      判定関数の結果 支持 -> 最終判定: 支持
      診断量(判定なし):
        距離の二乗誤差(20,000 組の平均、シード平均): 量子化の平均二乗誤差 D = 0.1014、query の量子化の誤差 E||e_x||^2 = 0.1749(D の 1.73 倍)。推定の偏り(推定 - 真の二乗距離の平均): 非対称 -0.1017(理論 -0.1014 = -D)、対称 -0.3157(理論 -0.2763 = -(E||e_x||^2 + D))。平均二乗誤差: 非対称 0.01541、対称 0.11299(比 7.33)
        再採点(上位 R 件を元のベクトルで再採点したときの recall、シード平均): asymmetric R = 10: 0.5143、asymmetric R = 20: 0.7025、asymmetric R = 50: 0.8897、asymmetric R = 100: 0.9625、asymmetric R = 200: 0.9916、symmetric R = 10: 0.3847、symmetric R = 20: 0.5406、symmetric R = 50: 0.7430、symmetric R = 100: 0.8643、symmetric R = 200: 0.9392
        費用(1 query あたり): 非対称 参照表の作成 8192・表引き 2097152、対称 query の量子化 8192・表引き 2097152。1 ベクトルあたりのバイト数 32(元は 1024、圧縮率 32 倍)



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/025_ann_search_and_reranking/output_48_1.png)
    


### 6.11 実験 D: cross-encoder の相互作用の効果


```python
RERANK_BOOTSTRAP_WEIGHTS = cluster_bootstrap_weights(QUERY_ARTICLES[RERANK_QUERY_INDEX], BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)


def recall_of(condition: str, seeds: int) -> np.ndarray:
    return np.array([RUNS[(condition, s)]["hits"].mean() for s in range(seeds)])


def hits_of(condition: str, seeds: int) -> np.ndarray:
    return np.stack([RUNS[(condition, s)]["hits"] for s in range(seeds)]).astype(np.float64)  # (シード, query)


def check_learning(conditions: tuple[str, ...], baseline: str, seeds: int) -> dict:
    # 学習の成立(6.1 節): (a)全条件・全シードで、訓練損失がすべてのステップで有限、最後の区間の訓練損失が学習前のモデルの同じ組での損失より標準誤差の 2 倍以上低く、
    # かつ一様な予測の損失 ln(1 + n) 未満。(b)基準の条件の、較正の query での Recall@10 の増分(学習前のモデルとの差)のシード平均が、
    # 判定と同じ式の標準偏差(シード間の項 + ブートストラップの項)の 2 倍以上
    loss_ok = {(c, s): learning_precondition(RUNS[(c, s)]) for c in conditions for s in range(seeds)}
    statistics = {(c, s): learning_loss_statistics(RUNS[(c, s)]) for c in conditions for s in range(seeds)}
    gain = validation_gain_statistics([RUNS[(baseline, s)] for s in range(seeds)])
    gain_ok = validation_gain_precondition([RUNS[(baseline, s)] for s in range(seeds)])
    return {
        "ok": all(loss_ok.values()) and gain_ok,
        "loss_failed": [k for k, v in loss_ok.items() if not v],
        "gain_ok": gain_ok,
        "loss_margin_min": min(
            (abs(v["mean"]) / v["standard_error"] if v["standard_error"] > 0 else float("inf")) if v["mean"] < 0 else 0.0 for v in statistics.values()
        ),
        "ratio_range": (min(v["ratio"] for v in statistics.values()), max(v["ratio"] for v in statistics.values())),
        "final_loss_max": max(v["final_loss"] for v in statistics.values()),
        "gain": gain["gain"],
        "gain_sigma": gain["sigma"],
        "gain_seed_term": gain["seed_term"],
        "gain_bootstrap_term": gain["bootstrap_term"],
        "per_seed_gain": gain["per_seed_gain"],
    }


def describe_learning(result: dict) -> str:
    return (
        f"((a)の不成立 {result['loss_failed'] or 'なし'}、最後の区間の損失の差の平均 / 標準誤差の最小 {result['loss_margin_min']:.1f}(閾値 {P1_SIGMA_MULTIPLIER})、"
        f"最後の区間の訓練損失の最大 {result['final_loss_max']:.3f}(閾値 ln(1 + n) = {UNIFORM_LOSS:.3f})、参考として最後の区間の訓練損失 / 学習前のモデルの損失は {result['ratio_range'][0]:.3f}〜{result['ratio_range'][1]:.3f}。"
        f"(b)基準の条件の較正の query での Recall@10 の増分のシード平均 {result['gain']:+.4f}(シードごと {rounded(result['per_seed_gain'])})、標準偏差 {result['gain_sigma']:.4f}"
        f"(シード間 {result['gain_seed_term']:.4f}・ブートストラップ {result['gain_bootstrap_term']:.4f})、増分 / 標準偏差 {result['gain'] / result['gain_sigma'] if result['gain_sigma'] > 0 else float('inf'):.1f}"
        f"(閾値 {P1_SIGMA_MULTIPLIER}): {result['gain_ok']})"
    )


def precondition_first_stage() -> bool:
    # 第 1 段の候補(基準の条件の結果に依らない): coverage >= 下限、かつ coverage - 並べ替えなしの Recall@10 >= 余地の下限
    return bool(COVERAGE >= P3_MIN_COVERAGE and COVERAGE - FIRST_STAGE_RECALL >= P3_MIN_HEADROOM)


_seeds_d = list(range(SEEDS_D))
R_D1, R_D2 = recall_of("D1", SEEDS_D), recall_of("D2", SEEDS_D)
D_PER_SEED = R_D1 - R_D2
_hit_difference = hits_of("D1", SEEDS_D) - hits_of("D2", SEEDS_D)
_boot_d = ((RERANK_BOOTSTRAP_WEIGHTS @ _hit_difference.T) / RERANK_BOOTSTRAP_WEIGHTS.sum(axis=1, keepdims=True)).mean(axis=1)
DELTA_D = float(D_PER_SEED.mean())
SIGMA_D = combined_sigma(D_PER_SEED, _boot_d)
_p1_d = check_learning(("D1", "D2"), "D2", SEEDS_D)
precondition_status["P1(D)"] = _p1_d["ok"]
precondition_status["P2(D)"] = bool(R_D2.mean() <= COVERAGE - P2_ROOM)
precondition_status["P3(D)"] = precondition_first_stage()
D_PRECONDITIONS = ["P0(D)", "P1(D)", "P2(D)", "P3(D)"]
D_COMPUTED = judge(DELTA_D, SIGMA_D["sigma"])
D_VERDICT = D_COMPUTED if all(precondition_status[k] for k in D_PRECONDITIONS) else "前提不成立"

print(f"{RUN_TAG}実験 D(シード {_seeds_d}、T = {NUM_STEPS}、K = {RERANK_CANDIDATES}、並べ替えの query {len(RERANK_QUERY_INDEX)} 個)")
print(f"  並べ替えの Recall@10 R(D1): {rounded(R_D1)}、R(D2): {rounded(R_D2)}")
print(f"  d_s = R(D1) - R(D2): {rounded(D_PER_SEED)}")
print(
    f"  Delta_D = {DELTA_D:+.4f}、sigma_D = {SIGMA_D['sigma']:.4f}(シード間 {SIGMA_D['seed_term']:.4f}・ブートストラップ {SIGMA_D['bootstrap_term']:.4f}、反復 {BOOTSTRAP_RESAMPLES:,})、"
    f"閾値 2 sigma_D = {SIGMA_MULTIPLIER * SIGMA_D['sigma']:.4f}"
)
print(
    f"  前提条件: P0 {precondition_status['P0(D)']}、P1 {precondition_status['P1(D)']}{describe_learning(_p1_d)}、"
    f"P2 {precondition_status['P2(D)']}(R(D2) の平均 {R_D2.mean():.4f} <= coverage {COVERAGE:.4f} - {P2_ROOM})、"
    f"P3 {precondition_status['P3(D)']}(coverage {COVERAGE:.4f} >= {P3_MIN_COVERAGE}、coverage - 並べ替えなしの Recall@10 {FIRST_STAGE_RECALL:.4f} = {COVERAGE - FIRST_STAGE_RECALL:.4f} >= {P3_MIN_HEADROOM})"
)
print(f"  判定関数の結果 {D_COMPUTED} -> {RUN_TAG}最終判定: {D_VERDICT}")
print("  診断量(判定なし):")
print(
    f"    第 1 段の順位のまま(並べ替えなし)の指標(K = {RERANK_CANDIDATES}): Recall@10 {FIRST_STAGE_RECALL:.4f}、平均逆順位 {FIRST_STAGE_RERANK[RERANK_CANDIDATES].mean_reciprocal_rank():.4f}、"
    f"正規化割引累積利得(上位 10 件){FIRST_STAGE_RERANK[RERANK_CANDIDATES].mean_normalized_discounted_cumulative_gain_at_10():.4f}、coverage(並べ替えの上限){COVERAGE:.4f}"
)
for _c in ("D1", "D2"):
    _mean_reciprocal_rank = np.array([RUNS[(_c, s)]["mean_reciprocal_rank"] for s in _seeds_d])
    _normalized_discounted_cumulative_gain = np.array([RUNS[(_c, s)]["normalized_discounted_cumulative_gain"] for s in _seeds_d])
    print(f"    {_c}: Recall@10 {recall_of(_c, SEEDS_D).mean():.4f}(シード間の標準偏差 {recall_of(_c, SEEDS_D).std(ddof=1):.4f})、平均逆順位 {_mean_reciprocal_rank.mean():.4f}、正規化割引累積利得(上位 10 件){_normalized_discounted_cumulative_gain.mean():.4f}")
print(
    f"    並べ替えによる Recall@10 の変化(並べ替えた後 - 第 1 段の順位のまま。追加の学習量と交絡するので判定の対象にしない): D1 {R_D1.mean() - FIRST_STAGE_RECALL:+.4f}、D2 {R_D2.mean() - FIRST_STAGE_RECALL:+.4f}"
)
print(
    f"    1 query あたりの順伝播の回数: cross-encoder(D1){RERANK_CANDIDATES} 回(組ごと)、dual encoder(D2)1 回(query。passage の埋め込みは事前に計算でき、{RERANK_CANDIDATES} 回の内積)。"
    f"最後の区間の訓練損失: D1 {np.mean([RUNS[('D1', s)]['final_train_loss'] for s in _seeds_d]):.3f}、D2 {np.mean([RUNS[('D2', s)]['final_train_loss'] for s in _seeds_d]):.3f}(一様な予測の損失 ln(1 + n) = {math.log(1 + NUM_NEGATIVES):.3f})"
)

print(
    "    学習後のモデルの、query ごとの候補 50 件のスコアの標準偏差の中央値(崩壊の検出。判定には使わない): "
    + "、".join(f"{_c} {rounded([RUNS[(_c, s)]['score_std_median'] for s in _seeds_d], 4)}" for _c in ("D1", "D2"))
)

fig, ax = plt.subplots(figsize=(5.5, 4))
for _x, _c in enumerate(("D1", "D2")):
    _values = recall_of(_c, SEEDS_D)
    ax.scatter([_x] * len(_values), _values, alpha=0.6)
    ax.plot([_x - 0.2, _x + 0.2], [_values.mean()] * 2, "k-")
ax.axhline(FIRST_STAGE_RECALL, color="gray", linestyle=":", label="first stage (no rerank)")
ax.axhline(COVERAGE, color="gray", linestyle="--", label="coverage (upper bound)")
ax.set_xticks([0, 1])
ax.set_xticklabels(["D1 cross-encoder", "D2 dual encoder"])
ax.set_ylabel(f"Recall@10 after reranking top-{RERANK_CANDIDATES}")
ax.set_title(f"{PLOT_TAG}Experiment D")
ax.legend(fontsize=7)
plt.tight_layout()
plt.show()
```

    実験 D(シード [0, 1, 2, 3, 4]、T = 1024、K = 50、並べ替えの query 1446 個)
      並べ替えの Recall@10 R(D1): [0.1708, 0.1791, 0.1722, 0.1584, 0.1757]、R(D2): [0.2351, 0.2317, 0.2296, 0.2344, 0.2379]
      d_s = R(D1) - R(D2): [-0.0643, -0.0526, -0.0574, -0.0761, -0.0622]
      Delta_D = -0.0625、sigma_D = 0.0084(シード間 0.0040・ブートストラップ 0.0074、反復 10,000)、閾値 2 sigma_D = 0.0168
      前提条件: P0 True、P1 False((a)の不成立 [('D1', 1), ('D1', 2)]、最後の区間の損失の差の平均 / 標準誤差の最小 1.2(閾値 2.0)、最後の区間の訓練損失の最大 2.078(閾値 ln(1 + n) = 2.079)、参考として最後の区間の訓練損失 / 学習前のモデルの損失は 0.852〜0.993。(b)基準の条件の較正の query での Recall@10 の増分のシード平均 +0.0180(シードごと [0.0205, 0.0125, 0.0148, 0.0228, 0.0194])、標準偏差 0.0072(シード間 0.0019・ブートストラップ 0.0070)、増分 / 標準偏差 2.5(閾値 2.0): True)、P2 True(R(D2) の平均 0.2337 <= coverage 0.4094 - 0.05)、P3 True(coverage 0.4094 >= 0.35、coverage - 並べ替えなしの Recall@10 0.2199 = 0.1895 >= 0.1)
      判定関数の結果 反証 -> 最終判定: 前提不成立
      診断量(判定なし):
        第 1 段の順位のまま(並べ替えなし)の指標(K = 50): Recall@10 0.2199、平均逆順位 0.1306、正規化割引累積利得(上位 10 件)0.0611、coverage(並べ替えの上限)0.4094
        D1: Recall@10 0.1712(シード間の標準偏差 0.0079)、平均逆順位 0.0753、正規化割引累積利得(上位 10 件)0.0308
        D2: Recall@10 0.2337(シード間の標準偏差 0.0032)、平均逆順位 0.1407、正規化割引累積利得(上位 10 件)0.0651
        並べ替えによる Recall@10 の変化(並べ替えた後 - 第 1 段の順位のまま。追加の学習量と交絡するので判定の対象にしない): D1 -0.0487、D2 +0.0138
        1 query あたりの順伝播の回数: cross-encoder(D1)50 回(組ごと)、dual encoder(D2)1 回(query。passage の埋め込みは事前に計算でき、50 回の内積)。最後の区間の訓練損失: D1 2.072、D2 1.868(一様な予測の損失 ln(1 + n) = 2.079)
        学習後のモデルの、query ごとの候補 50 件のスコアの標準偏差の中央値(崩壊の検出。判定には使わない): D1 [0.0457, 0.0696, 0.0721, 0.0486, 0.1197]、D2 [0.0296, 0.0304, 0.0294, 0.0301, 0.0294]



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/025_ann_search_and_reranking/output_50_1.png)
    




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/025_ann_search_and_reranking.ipynb
