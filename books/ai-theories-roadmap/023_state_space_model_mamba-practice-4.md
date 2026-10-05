---
title: "State Space Model / Mamba / State Space Model and Mamba(実装・実験編 4/6)"
---

この記事は後編(実装・実験編 4/6)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/023_state_space_model_mamba-practice-3)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/023_state_space_model_mamba-practice-5)。

### 6.5 学習率の較正

選ばれた計画の学習ステップ数(実験 A の種類は $T_A$、実験 C の種類は $T_C$)で、6.1 節の規則のとおりに学習率を決める。指標は **較正用の集合** の負の対数尤度(最終ステップの重み、較正専用のシード)である。


```python
_t0_calibration = time.time()
RUNNER = {"A1": run_task_a, "A2": run_task_a, "C1": run_task_c, "C2": run_task_c}
CALIBRATION = {}  # 種類 -> {"grid", "values", "chosen", "interior", "extended", "final_loss", "finite"}


def calibrate(kind: str) -> dict:
    grid = list(lr_grid_for(kind))
    records = {}

    def value(lr: float) -> float:
        if lr not in records:
            records[lr] = RUNNER[kind](kind, CALIBRATION_SEED_INDEX, lr, NUM_STEPS[kind[0]], calibration=True)
        record = records[lr]
        return record["calibration_nll"] if record["finite"] and math.isfinite(record["calibration_nll"]) else math.inf

    def best(points: list[float]) -> int:
        values = [value(lr) for lr in points]
        return min(range(len(points)), key=lambda i: (values[i], points[i]))  # 同点なら小さい学習率

    index = best(grid)
    extended = None
    if index == 0:  # 最良が端: その方向に公比 2 で 1 点だけ拡張する(上限は 1 回)
        extended = "lower"
        grid = [grid[0] / LR_GRID_RATIO] + grid
    elif index == len(grid) - 1:
        extended = "upper"
        grid = grid + [grid[-1] * LR_GRID_RATIO]
    index = best(grid)
    return {
        "grid": grid,
        "values": [value(lr) for lr in grid],
        "chosen": grid[index],
        "interior": 0 < index < len(grid) - 1,
        "extended": extended,
        "final_loss": [records[lr]["final_loss"] for lr in grid],
        "finite": [records[lr]["finite"] for lr in grid],
        "num_steps": NUM_STEPS[kind[0]],
    }


for _kind in KINDS:
    CALIBRATION[_kind] = calibrate(_kind)
    _r = CALIBRATION[_kind]
    print(
        f"較正 {_kind}: 学習率 {[float(f'{x:.3g}') for x in _r['grid']]} -> 較正用の集合の 負の対数尤度 {rounded(_r['values'])}、"
        f"最後の区間の訓練損失 {rounded(_r['final_loss'], 3)}、拡張 {_r['extended'] or 'なし'}、選んだ学習率 {_r['chosen']:.3g}"
        f"({'内点' if _r['interior'] else '端'})"
    )
LEARNING_RATE = {kind: CALIBRATION[kind]["chosen"] for kind in KINDS}
# P0(較正): その実験の対比量に入る両方の条件で、選んだ学習率が格子の内点であること(拡張後も最良が端なら不成立)
precondition_status["P0(A)"] = CALIBRATION["A1"]["interior"] and CALIBRATION["A2"]["interior"]
precondition_status["P0(C)"] = CALIBRATION["C1"]["interior"] and CALIBRATION["C2"]["interior"]
CALIBRATION_SECONDS = time.time() - _t0_calibration
print(f"学習率: {json.dumps({k: float(f'{v:.3g}') for k, v in LEARNING_RATE.items()})}")
print(f"{RUN_TAG}P0(較正): 実験 A: {precondition_status['P0(A)']}、実験 C: {precondition_status['P0(C)']}")
print(f"較正 {CALIBRATION_SECONDS / 60:.1f} 分(見積もり {PLAN_ESTIMATES[SELECTED_PLAN['plan']]['calibration'] / 60:.1f} 分、最悪の場合)")
```

    較正 A1: 学習率 [0.002, 0.004, 0.008] -> 較正用の集合の 負の対数尤度 [0.0004, 0.0001, 0.0001]、最後の区間の訓練損失 [0.001, 0.0, 0.0]、拡張 なし、選んだ学習率 0.004(内点)
    較正 A2: 学習率 [0.0005, 0.001, 0.002, 0.004] -> 較正用の集合の 負の対数尤度 [1.0946, 0.5225, 0.2478, 0.6001]、最後の区間の訓練損失 [1.099, 0.522, 0.258, 0.598]、拡張 upper、選んだ学習率 0.002(内点)
    較正 C1: 学習率 [0.016, 0.032, 0.064] -> 較正用の集合の 負の対数尤度 [0.0, 0.0, 2.7709]、最後の区間の訓練損失 [0.0, 0.0, 2.772]、拡張 なし、選んだ学習率 0.032(内点)
    較正 C2: 学習率 [0.004, 0.008, 0.016] -> 較正用の集合の 負の対数尤度 [0.0002, 0.0001, 2.2711]、最後の区間の訓練損失 [0.0, 0.0, 2.31]、拡張 なし、選んだ学習率 0.008(内点)
    学習率: {"A1": 0.004, "A2": 0.002, "C1": 0.032, "C2": 0.008}
    P0(較正): 実験 A: True、実験 C: True
    較正 17.9 分(見積もり 25.0 分、最悪の場合)


### 6.6 実験 A・C の本番の学習と評価

選ばれた計画の種類とシードをすべて学習する。学習のたびに、最終ステップの重みで評価集合の評価(実験 A は貪欲な生成による採点、実験 C は全系列長の予測)を行い、途中の評価の結果と生成した答えそのものを記録する。同じシードの A1 と A2、C1 と C2 は、学習データの流れが同じである。


```python
_t0_main = time.time()
RUNS = {}  # (種類, シード) -> 記録
for _s in range(SEEDS_A):
    for _kind in ("A1", "A2"):
        RUNS[(_kind, _s)] = run_task_a(_kind, _s, LEARNING_RATE[_kind], NUM_STEPS["A"])
        _r = RUNS[(_kind, _s)]
        print(
            f"{_kind} シード {_s}: 正解率(トークン単位、貪欲な生成) {_r['final']['token_accuracy']:.3f}、系列全体の一致 {_r['final']['exact_match']:.3f}、"
            f"訓練損失 {_r['initial_loss']:.3f} -> {_r['final_loss']:.3f}、clipping の発動 {_r['clip_rate']:.2f}、{_r['seconds']:.0f} 秒"
        )
for _s in range(SEEDS_C):
    for _kind in ("C1", "C2"):
        RUNS[(_kind, _s)] = run_task_c(_kind, _s, LEARNING_RATE[_kind], NUM_STEPS["C"])
        _r = RUNS[(_kind, _s)]
        print(
            f"{_kind} シード {_s}: 正解率 L_train = {C_TRAIN_LENGTH}: {_r['by_length'][C_TRAIN_LENGTH]['accuracy']:.3f}、"
            f"L_test = {C_TEST_LENGTH}: {_r['by_length'][C_TEST_LENGTH]['accuracy']:.3f}、"
            f"訓練損失 {_r['initial_loss']:.3f} -> {_r['final_loss']:.3f}、clipping の発動 {_r['clip_rate']:.2f}、{_r['seconds']:.0f} 秒"
        )
MAIN_SECONDS = time.time() - _t0_main
print(f"実験 A・C の学習と評価 {MAIN_SECONDS / 60:.1f} 分(見積もり {PLAN_ESTIMATES[SELECTED_PLAN['plan']]['main'] / 60:.1f} 分)")
```

    A1 シード 0: 正解率(トークン単位、貪欲な生成) 1.000、系列全体の一致 1.000、訓練損失 2.137 -> 0.000、clipping の発動 0.37、124 秒
    A2 シード 0: 正解率(トークン単位、貪欲な生成) 0.828、系列全体の一致 0.568、訓練損失 2.156 -> 0.206、clipping の発動 0.96、122 秒
    A1 シード 1: 正解率(トークン単位、貪欲な生成) 0.999、系列全体の一致 0.999、訓練損失 2.133 -> 0.001、clipping の発動 0.47、124 秒
    A2 シード 1: 正解率(トークン単位、貪欲な生成) 0.468、系列全体の一致 0.001、訓練損失 2.132 -> 0.944、clipping の発動 0.97、124 秒
    A1 シード 2: 正解率(トークン単位、貪欲な生成) 1.000、系列全体の一致 1.000、訓練損失 2.134 -> 0.000、clipping の発動 0.47、127 秒
    A2 シード 2: 正解率(トークン単位、貪欲な生成) 0.824、系列全体の一致 0.589、訓練損失 2.155 -> 0.183、clipping の発動 0.94、124 秒
    A1 シード 3: 正解率(トークン単位、貪欲な生成) 1.000、系列全体の一致 1.000、訓練損失 2.134 -> 0.000、clipping の発動 0.42、124 秒
    A2 シード 3: 正解率(トークン単位、貪欲な生成) 0.809、系列全体の一致 0.565、訓練損失 2.152 -> 0.197、clipping の発動 0.93、124 秒
    A1 シード 4: 正解率(トークン単位、貪欲な生成) 1.000、系列全体の一致 0.998、訓練損失 2.134 -> 0.001、clipping の発動 0.47、125 秒
    A2 シード 4: 正解率(トークン単位、貪欲な生成) 0.543、系列全体の一致 0.027、訓練損失 2.154 -> 0.651、clipping の発動 0.94、124 秒
    C1 シード 0: 正解率 L_train = 32: 1.000、L_test = 128: 1.000、訓練損失 1.957 -> 0.000、clipping の発動 0.04、48 秒
    C2 シード 0: 正解率 L_train = 32: 1.000、L_test = 128: 0.235、訓練損失 2.350 -> 0.000、clipping の発動 0.10、21 秒
    C1 シード 1: 正解率 L_train = 32: 1.000、L_test = 128: 1.000、訓練損失 1.840 -> 0.000、clipping の発動 0.06、47 秒
    C2 シード 1: 正解率 L_train = 32: 1.000、L_test = 128: 0.208、訓練損失 2.535 -> 0.000、clipping の発動 0.16、21 秒
    C1 シード 2: 正解率 L_train = 32: 0.064、L_test = 128: 0.070、訓練損失 1.864 -> 2.766、clipping の発動 0.13、47 秒
    C2 シード 2: 正解率 L_train = 32: 1.000、L_test = 128: 0.337、訓練損失 2.511 -> 0.000、clipping の発動 0.21、21 秒
    C1 シード 3: 正解率 L_train = 32: 1.000、L_test = 128: 1.000、訓練損失 1.741 -> 0.000、clipping の発動 0.04、48 秒
    C2 シード 3: 正解率 L_train = 32: 1.000、L_test = 128: 0.301、訓練損失 2.554 -> 0.000、clipping の発動 0.09、22 秒
    C1 シード 4: 正解率 L_train = 32: 1.000、L_test = 128: 1.000、訓練損失 1.979 -> 0.000、clipping の発動 0.04、47 秒
    C2 シード 4: 正解率 L_train = 32: 1.000、L_test = 128: 0.208、訓練損失 2.469 -> 0.000、clipping の発動 0.17、21 秒
    実験 A・C の学習と評価 26.8 分(見積もり 31.6 分)


### 6.7 実験 A: 選択性(selective copying)


```python
_seeds = list(range(SEEDS_A))
ACCURACY_A1 = np.array([RUNS[("A1", s)]["final"]["token_accuracy"] for s in _seeds])
ACCURACY_A2 = np.array([RUNS[("A2", s)]["final"]["token_accuracy"] for s in _seeds])
A_PER_SEED = ACCURACY_A1 - ACCURACY_A2
# 評価の事例(系列)を再標本化するブートストラップ。系列ごとの正解数 / 答えのトークン数 の比を、全条件・全シードに同じ再標本で作る
_correct_counts = np.stack(
    [RUNS[(kind, s)]["final"]["correct"].sum(axis=1) for kind in ("A1", "A2") for s in _seeds]
).astype(np.float64)
_denominators = np.full(_correct_counts.shape[1], float(TASK_A.num_data))
_boot = paired_bootstrap_ratio_of_sums(_correct_counts, _denominators, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
_boot_contrast = _boot[:, :SEEDS_A].mean(axis=1) - _boot[:, SEEDS_A:].mean(axis=1)
DELTA_A = float(A_PER_SEED.mean())
SIGMA_A = combined_sigma(A_PER_SEED, _boot_contrast)

_health = {(kind, s): learning_precondition(RUNS[(kind, s)]) for kind in ("A1", "A2") for s in _seeds}
precondition_status["P1(A)"] = all(_health.values())
precondition_status["P2(A)"] = float(ACCURACY_A2.mean()) <= A_ROOM_MAX
A_PRECONDITIONS = ["P0(A)", "P1(A)", "P2(A)"]
A_COMPUTED = judge(DELTA_A, SIGMA_A["sigma"])
A_VERDICT = A_COMPUTED if all(precondition_status[k] for k in A_PRECONDITIONS) else "前提不成立"

print(f"{RUN_TAG}実験 A(シード {_seeds}、学習 {NUM_STEPS['A']} ステップ)")
print(f"  正解率 a_1(選択的): {rounded(ACCURACY_A1)}、a_2(非選択): {rounded(ACCURACY_A2)}")
print(f"  d_s = a_1 - a_2: {rounded(A_PER_SEED)}")
print(
    f"  Delta_A = {DELTA_A:+.4f}、sigma_A = {SIGMA_A['sigma']:.4f}(シード間 {SIGMA_A['seed_term']:.4f}・ブートストラップ "
    f"{SIGMA_A['bootstrap_term']:.4f}、反復 {BOOTSTRAP_RESAMPLES:,})、閾値 2 sigma_A = {SIGMA_MULTIPLIER * SIGMA_A['sigma']:.4f}"
)
print(
    f"  前提条件: P0 {precondition_status['P0(A)']}、P1 {precondition_status['P1(A)']}(不成立の学習 {[k for k, v in _health.items() if not v] or 'なし'})、"
    f"P2(改善の余地) {precondition_status['P2(A)']}(a_2 のシード平均 {ACCURACY_A2.mean():.4f} <= {A_ROOM_MAX})"
)
print(f"  判定関数の結果 {A_COMPUTED} -> {RUN_TAG}最終判定: {A_VERDICT}")

# --- 診断量(判定なし) ---
print(
    f"  診断量: 系列全体の一致の割合 A1 {rounded([RUNS[('A1', s)]['final']['exact_match'] for s in _seeds])}、"
    f"A2 {rounded([RUNS[('A2', s)]['final']['exact_match'] for s in _seeds])}(偶然の水準: トークン単位 {TASK_A.chance_accuracy:.3f})"
)
for _kind in ("A1", "A2"):
    _position = np.mean([RUNS[(_kind, s)]["final"]["position_accuracy"] for s in _seeds], axis=0)
    print(f"  診断量: {_kind} の答えの位置ごとの正解率(シード平均、1 番目〜{TASK_A.num_data} 番目) {rounded(_position, 3)}")
print(
    f"  診断量: 非埋め込みパラメータ数 A1 {count_non_embedding_parameters(build_model('A1', 0)):,}、"
    f"A2 {count_non_embedding_parameters(build_model('A2', 0)):,}(差の内訳は 5.5 節の出力 11: 選択的は selection_projection と delta_projection の重みを持ち、"
    f"非選択は B・C のパラメータと delta_bias を持つ)"
)


def confusion_matrix(kind: str) -> np.ndarray:
    # 行: 正解のデータのトークン、列: 生成したトークン(データ 8 種類 + ノイズ + 区切り)。全シード・全評価事例・全位置を合計して行ごとに正規化
    matrix = np.zeros((TASK_A.num_symbols, TASK_A.vocabulary_size))
    targets = EVAL_A_TOKENS[:, TASK_A.input_length + 1 :].numpy()
    for s in _seeds:
        generated = RUNS[(kind, s)]["final"]["answers"]
        np.add.at(matrix, (targets.reshape(-1), generated.reshape(-1)), 1)
    return matrix / matrix.sum(axis=1, keepdims=True)


CONFUSION = {kind: confusion_matrix(kind) for kind in ("A1", "A2")}
for _kind in ("A1", "A2"):
    _matrix = CONFUSION[_kind]
    _off = _matrix[:, : TASK_A.num_symbols].copy()
    np.fill_diagonal(_off, 0.0)
    print(
        f"  診断量: {_kind} の混同行列(正解 -> 生成): 対角の平均 {np.mean(np.diag(_matrix)):.3f}、データ以外のトークン(ノイズ・区切り)を生成した割合 "
        f"{_matrix[:, TASK_A.num_symbols :].sum(axis=1).mean():.4f}、データのトークンどうしの誤りで最大の 1 セル {_off.max():.3f}"
    )

# 作用点に近い診断量: 条件 A1 の Δ_t(チャネルについての平均)の、入力部分のデータのトークンの位置とノイズのトークンの位置での平均
_probe = [RUNS[("A1", s)]["delta_probe"] for s in _seeds]
for _layer in range(SYN_NUM_LAYERS):
    _data = np.array([p["data"][_layer] for p in _probe])
    _noise = np.array([p["noise"][_layer] for p in _probe])
    _sep = np.array([p["separator"][_layer] for p in _probe])
    _ans = np.array([p["answer"][_layer] for p in _probe])
    print(
        f"  診断量(作用点に近い量): A1 の層 {_layer} の Δ の平均 データ {_data.mean():.4f}(各シード {rounded(_data, 4)})、"
        f"ノイズ {_noise.mean():.4f}(各シード {rounded(_noise, 4)})、データ / ノイズ = {np.mean(_data / _noise):.2f}、"
        f"区切り {_sep.mean():.4f}、答えの区間 {_ans.mean():.4f}"
    )

_fig, _axes = plt.subplots(1, 4, figsize=(18, 3.8))
_axes[0].plot(_seeds, ACCURACY_A1, "o", label="A1 (selective)")
_axes[0].plot(_seeds, ACCURACY_A2, "s", label="A2 (non-selective)")
_axes[0].axhline(TASK_A.chance_accuracy, color="gray", linestyle=":", label="chance")
_axes[0].set(xlabel="seed", ylabel="token accuracy (greedy generation)", title=f"{PLOT_TAG}Exp. A: accuracy per seed", ylim=(0, 1.02))
_axes[0].legend(fontsize=7)
for _kind, _marker in (("A1", "o"), ("A2", "s")):
    _curve = np.array([[c["accuracy"] for c in RUNS[(_kind, s)]["curve"]] for s in _seeds])
    _axes[1].plot(RUNS[(_kind, 0)]["eval_step"], _curve.mean(axis=0), marker=_marker, label=_kind)
_axes[1].set(xscale="log", xlabel="step", ylabel="teacher-forced token accuracy", title=f"{PLOT_TAG}accuracy during training")
_axes[1].legend(fontsize=7)
_example = RUNS[("A1", 0)]["delta_probe"]
for _layer in range(SYN_NUM_LAYERS):
    _axes[2].plot(_example["example"][_layer][: TASK_A.input_length], label=f"layer {_layer}")
_data_positions = np.flatnonzero(_example["example_is_data"])
_axes[2].scatter(_data_positions, np.zeros(len(_data_positions)), marker="|", color="k", s=200, label="data token")
_axes[2].set(xlabel="position (input part)", ylabel="mean Delta over channels", title=f"{PLOT_TAG}A1 seed 0: Delta_t")
_axes[2].legend(fontsize=7)
_axes[3].imshow(CONFUSION["A2"], vmin=0, vmax=1, cmap="viridis")
_axes[3].set(xlabel="generated token", ylabel="correct data token", title=f"{PLOT_TAG}A2 confusion matrix")
plt.tight_layout()
plt.show()
```

    実験 A(シード [0, 1, 2, 3, 4]、学習 4000 ステップ)
      正解率 a_1(選択的): [1.0, 0.9992, 1.0, 1.0, 0.9995]、a_2(非選択): [0.8282, 0.4675, 0.8235, 0.8089, 0.543]
      d_s = a_1 - a_2: [0.1717, 0.5317, 0.1765, 0.1911, 0.4565]
      Delta_A = +0.3055、sigma_A = 0.0781(シード間 0.0780・ブートストラップ 0.0041、反復 10,000)、閾値 2 sigma_A = 0.1562
      前提条件: P0 True、P1 True(不成立の学習 なし)、P2(改善の余地) True(a_2 のシード平均 0.6942 <= 0.9)
      判定関数の結果 支持 -> 最終判定: 支持
      診断量: 系列全体の一致の割合 A1 [1.0, 0.999, 1.0, 1.0, 0.998]、A2 [0.568, 0.001, 0.589, 0.565, 0.027](偶然の水準: トークン単位 0.125)
      診断量: A1 の答えの位置ごとの正解率(シード平均、1 番目〜8 番目) [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
      診断量: A2 の答えの位置ごとの正解率(シード平均、1 番目〜8 番目) [0.803, 0.744, 0.688, 0.635, 0.627, 0.649, 0.669, 0.739]
      診断量: 非埋め込みパラメータ数 A1 65,472、A2 63,424(差の内訳は 5.5 節の出力 11: 選択的は selection_projection と delta_projection の重みを持ち、非選択は B・C のパラメータと delta_bias を持つ)
      診断量: A1 の混同行列(正解 -> 生成): 対角の平均 1.000、データ以外のトークン(ノイズ・区切り)を生成した割合 0.0000、データのトークンどうしの誤りで最大の 1 セル 0.000
      診断量: A2 の混同行列(正解 -> 生成): 対角の平均 0.694、データ以外のトークン(ノイズ・区切り)を生成した割合 0.0000、データのトークンどうしの誤りで最大の 1 セル 0.121
      診断量(作用点に近い量): A1 の層 0 の Δ の平均 データ 0.0502(各シード [0.0423, 0.0791, 0.0301, 0.024, 0.0755])、ノイズ 0.0712(各シード [0.0503, 0.0993, 0.0459, 0.0303, 0.1304])、データ / ノイズ = 0.73、区切り 0.0798、答えの区間 0.0673
      診断量(作用点に近い量): A1 の層 1 の Δ の平均 データ 0.0887(各シード [0.0361, 0.0927, 0.0536, 0.0401, 0.2212])、ノイズ 0.0974(各シード [0.0358, 0.0868, 0.0621, 0.0521, 0.25])、データ / ノイズ = 0.92、区切り 0.1555、答えの区間 0.0566



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/023_state_space_model_mamba/output_32_1.png)
    


### 6.8 実験 B: 計算量のべき指数

6.3 節で定義した層・水準・計測の関数を使い、掃引 $R$ 回 × 反復 $K$ 回を計測する。**ウォームアップは、この節の計測の直前に行う**(アロケータのキャッシュを解放した直後に、全水準・両条件を 3 回ずつ実行する)。6.3 節のウォームアップだけでは、その後の実験 A・C の学習の関数が学習のたびにキャッシュを解放するため、掃引の先頭の反復でバッファの確保のし直しが起こり、先頭の反復だけが極端に遅くなる(6.1 節の「ウォームアップの手順の改訂」)。


```python
_t0_b = time.time()
# ウォームアップは計測の直前に行う。6.3 節のウォームアップの後に実験 A・C を学習すると、学習の関数が学習のたびに
# empty_device_cache() でアロケータのキャッシュを解放するため、そのままでは掃引の先頭の反復でバッファの確保のし直しが
# 起こり、先頭の反復だけが極端に遅くなる。そこで、キャッシュを解放した直後に、全水準・両条件を B_WARMUP_REPEATS 回ずつ実行する。
empty_device_cache()
with torch.no_grad():
    for _length in B_LEVELS:
        for _ in range(B_WARMUP_REPEATS):
            for _condition in ("mamba", "attention"):
                time_layer(_condition, _length)
print(f"実験 B のウォームアップ(キャッシュの解放の直後、全水準・両条件を {B_WARMUP_REPEATS} 回ずつ): {time.time() - _t0_b:.1f} 秒")
B_RAW = {c: {length: [[] for _ in range(B_SWEEPS)] for length in B_LEVELS} for c in ("mamba", "attention")}
for _sweep in range(B_SWEEPS):
    for _repeat in range(B_REPEATS):
        _times = run_b_pass(_sweep * B_REPEATS + _repeat)
        for _c, _by_length in _times.items():
            for _length, _seconds in _by_length.items():
                B_RAW[_c][_length][_sweep].append(_seconds)
B_MEASURE_SECONDS = time.time() - _t0_b


def series_in_time_order(condition: str, length: int) -> np.ndarray:
    return np.array([t for sweep in B_RAW[condition][length] for t in sweep])


def compute_p0a(values) -> dict:
    # P-B1(ウォームアップの担保): 先頭 3 反復の平均と末尾 3 反復の平均の差が、全反復の標準偏差(ddof=1)の 2 倍以内か
    arr = np.asarray(values)
    diff = abs(float(arr[-3:].mean()) - float(arr[:3].mean()))
    threshold = P0A_MAX_DRIFT_SIGMA * float(arr.std(ddof=1))
    return {"head_mean": float(arr[:3].mean()), "tail_mean": float(arr[-3:].mean()), "diff": diff, "threshold": threshold, "holds": bool(diff <= threshold)}


def compute_p0b(values) -> dict:
    # P-B2(判定精度の担保): 平均の標準誤差(標準偏差 ddof=1 / sqrt(反復回数))が平均の 5% 以下か
    arr = np.asarray(values)
    stderr = float(arr.std(ddof=1)) / math.sqrt(len(arr))
    return {"mean": float(arr.mean()), "stderr": stderr, "ratio": stderr / float(arr.mean()), "holds": bool(stderr / float(arr.mean()) <= P0B_MAX_RELATIVE_STDERR)}


P0A = {(c, L): compute_p0a(series_in_time_order(c, L)) for c in B_RAW for L in B_LEVELS}
P0B = {(c, L): compute_p0b(series_in_time_order(c, L)) for c in B_RAW for L in B_LEVELS}
precondition_status["P-B1"] = all(v["holds"] for v in P0A.values())
precondition_status["P-B2"] = all(v["holds"] for v in P0B.values())

# 掃引ごとのべき指数: 水準ごとの中央値の対数に対する回帰(水準が独立な単位。擬似反復を避ける)
B_FITS = {c: [] for c in B_RAW}
for _sweep in range(B_SWEEPS):
    for _c in B_RAW:
        _medians = [float(np.median(B_RAW[_c][L][_sweep])) for L in B_LEVELS]
        B_FITS[_c].append(fit_power_law_exponent(B_LEVELS, _medians))
EXPONENT_MAMBA = np.array([f.exponent for f in B_FITS["mamba"]])
EXPONENT_ATTENTION = np.array([f.exponent for f in B_FITS["attention"]])
B_PER_SWEEP = EXPONENT_ATTENTION - EXPONENT_MAMBA
DELTA_B = float(B_PER_SWEEP.mean())
SIGMA_B = float(B_PER_SWEEP.std(ddof=1)) / math.sqrt(B_SWEEPS)
B_PRECONDITIONS = ["P-B1", "P-B2"]
B_COMPUTED = judge(DELTA_B, SIGMA_B)
B_VERDICT = B_COMPUTED if all(precondition_status[k] for k in B_PRECONDITIONS) else "前提不成立"

print(f"{RUN_TAG}実験 B(系列長 {B_LEVELS}、掃引 {B_SWEEPS} 回 x 反復 {B_REPEATS} 回、デバイス {device.type})")
print(f"  掃引ごとのべき指数 b_1(Mamba の層): {rounded(EXPONENT_MAMBA, 3)}、b_2(注意機構の層): {rounded(EXPONENT_ATTENTION, 3)}")
print(f"  掃引ごとの Delta_B = b_2 - b_1: {rounded(B_PER_SWEEP, 3)}")
print(
    f"  Delta_B = {DELTA_B:+.4f}、sigma_B = {SIGMA_B:.4f}(掃引ごとの Delta_B の標準偏差 {B_PER_SWEEP.std(ddof=1):.4f} / sqrt({B_SWEEPS}))、"
    f"閾値 2 sigma_B = {SIGMA_MULTIPLIER * SIGMA_B:.4f}"
)
_bad_a = [k for k, v in P0A.items() if not v["holds"]]
_bad_b = [k for k, v in P0B.items() if not v["holds"]]
print(
    f"  前提条件: P-B1(ドリフトなし) {precondition_status['P-B1']}(不成立 {_bad_a or 'なし'})、P-B2(平均の標準誤差 <= 平均の "
    f"{P0B_MAX_RELATIVE_STDERR:.0%}) {precondition_status['P-B2']}(不成立 {_bad_b or 'なし'}、比の最大 {max(v['ratio'] for v in P0B.values()):.4f})"
)
print(f"  判定関数の結果 {B_COMPUTED} -> {RUN_TAG}最終判定: {B_VERDICT}")

# --- 診断量(判定なし) ---
print(
    f"  診断量: b_1 = {EXPONENT_MAMBA.mean():.3f}(理論値 1 との差 {EXPONENT_MAMBA.mean() - 1:+.3f})、"
    f"b_2 = {EXPONENT_ATTENTION.mean():.3f}(理論値 2 との差 {EXPONENT_ATTENTION.mean() - 2:+.3f})。掃引ごとの回帰の標準誤差の平均 "
    f"b_1 {np.mean([f.exponent_stderr for f in B_FITS['mamba']]):.3f}・b_2 {np.mean([f.exponent_stderr for f in B_FITS['attention']]):.3f}"
)
print("  診断量: 水準ごとの時間の中央値(全掃引・全反復、ミリ秒):")
for _c in B_RAW:
    print(f"    {_c}: " + "、".join(f"{L}: {1000 * float(np.median(series_in_time_order(_c, L))):.2f}" for L in B_LEVELS))
print("  診断量: 水準ごとの P-B1 の先頭 3 反復 / 末尾 3 反復の平均(ミリ秒)と P-B2 の標準誤差 / 平均:")
for _c in B_RAW:
    print(f"    {_c}: " + "、".join(
        f"{L}: {1000 * P0A[(_c, L)]['head_mean']:.2f}/{1000 * P0A[(_c, L)]['tail_mean']:.2f}・{P0B[(_c, L)]['ratio']:.4f}" for L in B_LEVELS
    ))

# 最大メモリの系列長に対するべき指数(決定的な量なので判定には使わない。CUDA でのみ測る)
if device.type == "cuda":
    _peaks = {c: [] for c in B_RAW}
    for _length in B_LEVELS:
        for _c in B_RAW:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            _base = torch.cuda.memory_allocated()
            time_layer(_c, _length)
            _peaks[_c].append(torch.cuda.max_memory_allocated() - _base)
    _memory_fits = {c: fit_power_law_exponent(B_LEVELS, _peaks[c]) for c in _peaks}
    print(
        "  診断量: 最大メモリの増分(MiB) "
        + "、".join(f"{c}: {[round(p / 2**20, 1) for p in _peaks[c]]}(べき指数 {_memory_fits[c].exponent:.3f})" for c in _peaks)
    )
else:
    print("  診断量: 最大メモリは CUDA でのみ測る(このデバイスでは未計測)")
_analytic_mamba = [2 * L * B_MAMBA.d_inner * MAMBA_STATE_DIM * 4 for L in B_LEVELS]  # Ā と B̄x の 2 枚(FP32)
_analytic_attention = [3 * B_NUM_HEADS * L**2 * 4 for L in B_LEVELS]  # スコア・マスク後・softmax の 3 枚(FP32)
print(
    f"  診断量(閉形式の下限): 一時テンソルのバイト数のべき指数 Mamba {fit_power_law_exponent(B_LEVELS, _analytic_mamba).exponent:.3f}、"
    f"注意機構 {fit_power_law_exponent(B_LEVELS, _analytic_attention).exponent:.3f}"
)

# 推論時の 1 トークンあたりの時間と状態のサイズ: Mamba の 1 ステップ更新(定数) と、KV キャッシュつきの注意機構(文脈の長さに比例)
DECODE_ROWS = []
with torch.no_grad():
    _state = B_MAMBA.initial_state(1, device, torch.float32)
    _token = torch.randn(1, B_D_MODEL, device=device)
    _mamba_step = lambda: B_MAMBA.step(_token, _state)  # noqa: E731
    timed_call(_mamba_step)
    _mamba_ms = 1000 * float(np.median([timed_call(_mamba_step) for _ in range(20)]))
    _state_bytes = sum(t.numel() * t.element_size() for t in _state)
    for _context in B_DECODE_CONTEXTS:
        _cache = KeyValueCache(1)
        _d_k = B_D_MODEL // B_NUM_HEADS
        _cache.update(0, torch.randn(1, B_NUM_HEADS, _context, _d_k, device=device), torch.randn(1, B_NUM_HEADS, _context, _d_k, device=device))
        _new = torch.randn(1, 1, B_D_MODEL, device=device)
        _attention_step = lambda: B_ATTENTION(_new, _new, _new, kv_cache=_cache, layer_idx=0)  # noqa: E731
        timed_call(_attention_step)
        _attention_ms = 1000 * float(np.median([timed_call(_attention_step) for _ in range(20)]))
        DECODE_ROWS.append((_context, _attention_ms, _cache.memory_bytes()))
print(
    f"  診断量: 推論時の 1 トークンあたり — Mamba の 1 ステップ更新 {_mamba_ms:.3f} ms、状態 {_state_bytes:,} バイト(文脈の長さによらず一定)。"
    + "注意機構(KV キャッシュつき、1 層): "
    + "、".join(f"文脈 {c}: {ms:.3f} ms・KV キャッシュ {b:,} バイト" for c, ms, b in DECODE_ROWS)
    + "(逐次ループの実装の絶対時間は公式実装を代表しない)"
)

_fig, _axes = plt.subplots(1, 2, figsize=(11, 3.8))
for _c, _marker in (("mamba", "o"), ("attention", "s")):
    _medians = [float(np.median(series_in_time_order(_c, L))) for L in B_LEVELS]
    _axes[0].plot(B_LEVELS, _medians, marker=_marker, label=f"{_c} (b = {np.mean([f.exponent for f in B_FITS[_c]]):.2f})")
_axes[0].set(xscale="log", yscale="log", xlabel="sequence length L", ylabel="forward time of one layer [s]", title=f"{PLOT_TAG}Experiment B: time per layer")
_axes[0].legend()
_axes[1].plot(range(B_SWEEPS), EXPONENT_MAMBA, "o", label="b_1 (Mamba)")
_axes[1].plot(range(B_SWEEPS), EXPONENT_ATTENTION, "s", label="b_2 (attention)")
_axes[1].axhline(1, color="gray", linestyle=":")
_axes[1].axhline(2, color="gray", linestyle=":")
_axes[1].set(xlabel="sweep", ylabel="exponent of time vs L", title=f"{PLOT_TAG}exponent per sweep (dotted: theory 1 and 2)")
_axes[1].legend()
plt.tight_layout()
plt.show()
print(f"実験 B の計測 {B_MEASURE_SECONDS / 60:.1f} 分(見積もり {B_TOTAL_SECONDS / 60:.1f} 分)")
```

    実験 B のウォームアップ(キャッシュの解放の直後、全水準・両条件を 3 回ずつ): 2.3 秒
    実験 B(系列長 (512, 1024, 2048, 4096, 8192)、掃引 8 回 x 反復 8 回、デバイス cuda)
      掃引ごとのべき指数 b_1(Mamba の層): [1.003, 0.992, 1.037, 0.984, 0.967, 0.992, 0.981, 0.986]、b_2(注意機構の層): [1.552, 1.571, 1.561, 1.574, 1.557, 1.594, 1.58, 1.581]
      掃引ごとの Delta_B = b_2 - b_1: [0.549, 0.579, 0.525, 0.59, 0.59, 0.602, 0.6, 0.595]
      Delta_B = +0.5788、sigma_B = 0.0098(掃引ごとの Delta_B の標準偏差 0.0276 / sqrt(8))、閾値 2 sigma_B = 0.0195
      前提条件: P-B1(ドリフトなし) True(不成立 なし)、P-B2(平均の標準誤差 <= 平均の 5%) True(不成立 なし、比の最大 0.0348)
      判定関数の結果 支持 -> 最終判定: 支持
      診断量: b_1 = 0.993(理論値 1 との差 -0.007)、b_2 = 1.571(理論値 2 との差 -0.429)。掃引ごとの回帰の標準誤差の平均 b_1 0.014・b_2 0.157
      診断量: 水準ごとの時間の中央値(全掃引・全反復、ミリ秒):
        mamba: 512: 22.43、1024: 43.73、2048: 86.34、4096: 172.55、8192: 349.91
        attention: 512: 0.76、1024: 1.20、2048: 3.64、4096: 13.18、8192: 54.12
      診断量: 水準ごとの P-B1 の先頭 3 反復 / 末尾 3 反復の平均(ミリ秒)と P-B2 の標準誤差 / 平均:
        mamba: 512: 22.45/22.18・0.0252、1024: 43.91/43.58・0.0257、2048: 87.37/86.07・0.0348、4096: 169.65/229.68・0.0264、8192: 413.43/345.14・0.0219
        attention: 512: 0.71/0.74・0.0212、1024: 1.18/1.21・0.0211、2048: 3.65/3.60・0.0022、4096: 13.18/13.17・0.0010、8192: 54.14/53.98・0.0006
      診断量: 最大メモリの増分(MiB) mamba: [34.6, 69.7, 138.3, 276.6, 553.6](べき指数 0.999)、attention: [9.0, 34.5, 135.0, 534.0, 2124.0](べき指数 1.972)
      診断量(閉形式の下限): 一時テンソルのバイト数のべき指数 Mamba 1.000、注意機構 2.000
      診断量: 推論時の 1 トークンあたり — Mamba の 1 ステップ更新 0.932 ms、状態 19,456 バイト(文脈の長さによらず一定)。注意機構(KV キャッシュつき、1 層): 文脈 512: 0.600 ms・KV キャッシュ 545,792 バイト、文脈 2048: 0.594 ms・KV キャッシュ 2,118,656 バイト、文脈 8192: 0.609 ms・KV キャッシュ 8,410,112 バイト(逐次ループの実装の絶対時間は公式実装を代表しない)



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/023_state_space_model_mamba/output_34_1.png)
    


    実験 B の計測 0.9 分(見積もり 1.0 分)


### 6.9 実験 C: 長さの外挿(induction heads)


```python
_seeds = list(range(SEEDS_C))


def accuracy_at(kind: str, length: int) -> np.ndarray:
    return np.array([RUNS[(kind, s)]["by_length"][length]["accuracy"] for s in _seeds])


ACCURACY_TRAIN = {kind: accuracy_at(kind, C_TRAIN_LENGTH) for kind in ("C1", "C2")}
ACCURACY_TEST = {kind: accuracy_at(kind, C_TEST_LENGTH) for kind in ("C1", "C2")}
DROP = {kind: ACCURACY_TRAIN[kind] - ACCURACY_TEST[kind] for kind in ("C1", "C2")}  # d_i,s
C_PER_SEED = DROP["C2"] - DROP["C1"]
# 評価の事例を再標本化するブートストラップ。系列長ごとに別の評価集合なので、再標本は系列長ごとに独立に作る。
# 各系列長で、全条件・全シードに同じ再標本を使う
_runs = [(kind, s) for kind in ("C1", "C2") for s in _seeds]
_boot = {}
for _offset, _length in enumerate((C_TRAIN_LENGTH, C_TEST_LENGTH)):
    _correct = np.stack([RUNS[key]["by_length"][_length]["correct"].astype(np.float64) for key in _runs])  # (実行, 事例)
    _boot[_length] = paired_bootstrap_ratio_of_sums(
        _correct, np.ones(_correct.shape[1]), BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED + 1 + _offset
    )  # (反復, 実行)
_index = {key: i for i, key in enumerate(_runs)}


def bootstrap_drop(kind: str) -> np.ndarray:
    columns = [_index[(kind, s)] for s in _seeds]
    return (_boot[C_TRAIN_LENGTH][:, columns] - _boot[C_TEST_LENGTH][:, columns]).mean(axis=1)


_boot_contrast = bootstrap_drop("C2") - bootstrap_drop("C1")
DELTA_C = float(C_PER_SEED.mean())
SIGMA_C = combined_sigma(C_PER_SEED, _boot_contrast)

_health = {(kind, s): learning_precondition(RUNS[(kind, s)]) for kind in ("C1", "C2") for s in _seeds}
precondition_status["P1(C)"] = all(_health.values())
precondition_status["P2(C)"] = all(float(ACCURACY_TRAIN[kind].mean()) >= C_MASTERY_MIN for kind in ("C1", "C2"))
C_PRECONDITIONS = ["P0(C)", "P1(C)", "P2(C)"]
C_COMPUTED = judge(DELTA_C, SIGMA_C["sigma"])
C_VERDICT = C_COMPUTED if all(precondition_status[k] for k in C_PRECONDITIONS) else "前提不成立"

print(f"{RUN_TAG}実験 C(シード {_seeds}、学習 {NUM_STEPS['C']} ステップ、L_train = {C_TRAIN_LENGTH}、L_test = {C_TEST_LENGTH})")
for _kind in ("C1", "C2"):
    print(
        f"  {_kind}: a(L_train) {rounded(ACCURACY_TRAIN[_kind])}、a(L_test) {rounded(ACCURACY_TEST[_kind])}、"
        f"d = a(L_train) - a(L_test) {rounded(DROP[_kind])}"
    )
print(f"  d_s(条件 2 - 条件 1) = {rounded(C_PER_SEED)}")
print(
    f"  Delta_C = {DELTA_C:+.4f}、sigma_C = {SIGMA_C['sigma']:.4f}(シード間 {SIGMA_C['seed_term']:.4f}・ブートストラップ "
    f"{SIGMA_C['bootstrap_term']:.4f}、反復 {BOOTSTRAP_RESAMPLES:,})、閾値 2 sigma_C = {SIGMA_MULTIPLIER * SIGMA_C['sigma']:.4f}"
)
print(
    f"  前提条件: P0 {precondition_status['P0(C)']}、P1 {precondition_status['P1(C)']}(不成立の学習 {[k for k, v in _health.items() if not v] or 'なし'})、"
    f"P2(習得) {precondition_status['P2(C)']}(a(L_train) のシード平均 C1 {ACCURACY_TRAIN['C1'].mean():.4f}・C2 {ACCURACY_TRAIN['C2'].mean():.4f} >= {C_MASTERY_MIN})"
)
print(f"  判定関数の結果 {C_COMPUTED} -> {RUN_TAG}最終判定: {C_VERDICT}")

# --- 診断量(判定なし)。「Mamba が正解率を保つ」こと自体は判定の対象ではなく、ここで読む ---
print(f"  診断量: 偶然の水準 {TASK_C.chance_accuracy:.4f}。系列長ごとの正解率(シード平均 ± シード間の標準偏差):")
for _kind in ("C1", "C2"):
    print(
        f"    {_kind}: "
        + "、".join(
            f"{length // C_TRAIN_LENGTH}x({length}): {accuracy_at(_kind, length).mean():.3f} ± {accuracy_at(_kind, length).std(ddof=1):.3f}"
            for length in C_LENGTHS
        )
    )
print(f"  診断量: L_test = {C_TEST_LENGTH} の正解率の各シードの値(二峰かどうかが分かる形、昇順): " + "、".join(
    f"{_kind} {sorted(rounded(ACCURACY_TEST[_kind], 3))}" for _kind in ("C1", "C2")
))
print(f"  診断量: L_train の各シードの値(昇順): " + "、".join(f"{_kind} {sorted(rounded(ACCURACY_TRAIN[_kind], 3))}" for _kind in ("C1", "C2")))
print(
    f"  診断量: 非埋め込みパラメータ数 C1 {count_non_embedding_parameters(build_model('C1', 0)):,}、C2 {count_non_embedding_parameters(build_model('C2', 0)):,}"
)

_fig, _axes = plt.subplots(1, 2, figsize=(11, 3.8))
for _kind, _marker, _label in (("C1", "o", "C1 Mamba"), ("C2", "s", "C2 Transformer (RoPE)")):
    _per_length = np.array([accuracy_at(_kind, length) for length in C_LENGTHS])  # (長さ, シード)
    _axes[0].plot(C_LENGTHS, _per_length.mean(axis=1), marker=_marker, label=_label)
    for _s in range(_per_length.shape[1]):
        _axes[0].plot(C_LENGTHS, _per_length[:, _s], color=f"C{0 if _kind == 'C1' else 1}", alpha=0.25, linewidth=0.8)
_axes[0].axhline(TASK_C.chance_accuracy, color="gray", linestyle=":", label="chance")
_axes[0].axvline(C_TRAIN_LENGTH, color="k", linestyle="--", linewidth=0.8)
_axes[0].axvline(C_TEST_LENGTH, color="k", linestyle=":", linewidth=0.8)
_axes[0].set(xscale="log", xlabel="sequence length", ylabel="accuracy", ylim=(0, 1.02), title=f"{PLOT_TAG}Experiment C: accuracy vs length (thin: seeds)")
_axes[0].legend(fontsize=7)
for _kind, _marker in (("C1", "o"), ("C2", "s")):
    _curve = np.array([[c["accuracy"] for c in RUNS[(_kind, s)]["curve"]] for s in _seeds])
    _axes[1].plot(RUNS[(_kind, 0)]["eval_step"], _curve.mean(axis=0), marker=_marker, label=_kind)
_axes[1].set(xscale="log", xlabel="step", ylabel=f"accuracy at L_train (first {CURVE_SEQUENCES} eval sequences)", title=f"{PLOT_TAG}accuracy during training (seed mean)")
_axes[1].legend()
plt.tight_layout()
plt.show()
```

    実験 C(シード [0, 1, 2, 3, 4]、学習 2000 ステップ、L_train = 32、L_test = 128)
      C1: a(L_train) [1.0, 1.0, 0.064, 1.0, 1.0]、a(L_test) [1.0, 1.0, 0.07, 1.0, 1.0]、d = a(L_train) - a(L_test) [0.0, 0.0, -0.006, 0.0, 0.0]
      C2: a(L_train) [1.0, 1.0, 1.0, 1.0, 1.0]、a(L_test) [0.235, 0.208, 0.337, 0.301, 0.208]、d = a(L_train) - a(L_test) [0.765, 0.792, 0.663, 0.699, 0.792]
      d_s(条件 2 - 条件 1) = [0.765, 0.792, 0.669, 0.699, 0.792]
      Delta_C = +0.7434、sigma_C = 0.0269(シード間 0.0252・ブートストラップ 0.0095、反復 10,000)、閾値 2 sigma_C = 0.0538
      前提条件: P0 True、P1 False(不成立の学習 [('C1', 2)])、P2(習得) False(a(L_train) のシード平均 C1 0.8128・C2 1.0000 >= 0.9)
      判定関数の結果 支持 -> 最終判定: 前提不成立
      診断量: 偶然の水準 0.0625。系列長ごとの正解率(シード平均 ± シード間の標準偏差):
        C1: 1x(32): 0.813 ± 0.419、2x(64): 0.813 ± 0.418、4x(128): 0.814 ± 0.416、8x(256): 0.811 ± 0.424、16x(512): 0.810 ± 0.424
        C2: 1x(32): 1.000 ± 0.000、2x(64): 0.620 ± 0.023、4x(128): 0.258 ± 0.058、8x(256): 0.122 ± 0.044、16x(512): 0.080 ± 0.028
      診断量: L_test = 128 の正解率の各シードの値(二峰かどうかが分かる形、昇順): C1 [0.07, 1.0, 1.0, 1.0, 1.0]、C2 [0.208, 0.208, 0.235, 0.301, 0.337]
      診断量: L_train の各シードの値(昇順): C1 [0.064, 1.0, 1.0, 1.0, 1.0]、C2 [1.0, 1.0, 1.0, 1.0, 1.0]
      診断量: 非埋め込みパラメータ数 C1 65,472、C2 65,344



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/023_state_space_model_mamba/output_36_1.png)
    


### 6.10 実験 D: 言語モデリングの動作確認(判定基準を設けない)

計画で選ばれた学習ステップ数で、Mamba の言語モデルを 008 のレシピで 1 シード学習し、008 の学習済みモデルと同じ評価窓で検証 bits-per-byte を測る。優劣や同等性について結論を書かない。


```python
_t0_d = time.time()
D_RESULT = None
if D_STEPS is None:
    print(f"{RUN_TAG}実験 D: 選ばれた計画(計画 {SELECTED_PLAN['plan']})では省略した(実行時間の見積もりのみによる選択、6.1 節)")
else:
    # --- コーパス・トークナイザ・評価窓(008 と同じ英語 Wikipedia のコーパスの末尾 5% を検証の部分とする) ---
    d_tokenizer, _loaded = load_bpe_id_tokenizer_from_hub(TOKENIZER_REPO_ID)
    assert _loaded, "トークナイザを Hugging Face Hub から取得できなかった"
    assert d_tokenizer.vocab_size == D_VOCAB_SIZE
    d_corpus, d_corpus_metadata = load_wikipedia_corpus_with_fallback(
        "en", CORPUS_REPO_ID, WIKIPEDIA_CACHE_DIR, manifest_path=MANIFEST_PATH, return_metadata=True
    )
    assert len(d_corpus.encode("utf-8")) == d_corpus_metadata["raw_bytes"], "コーパスの取得が破損している"
    assert d_corpus_metadata["source"] == "hub", f"コーパスの取得元が Hub でない: {d_corpus_metadata['source']}"
    if CFG["D_TRAIN_CHARACTERS"] is not None:  # スモークテストはコーパスの先頭だけを使う
        d_corpus = d_corpus[: CFG["D_TRAIN_CHARACTERS"]]
    d_train_text, d_val_text = split_train_val_text(d_corpus, D_VALIDATION_RATIO)
    d_total_bytes = len(d_val_text.encode("utf-8"))
    d_train_ids, d_val_ids = encode_corpus(d_tokenizer, d_train_text), encode_corpus(d_tokenizer, d_val_text)
    assert d_tokenizer.decode(d_tokenizer.encode(d_train_text[:50_000])) == d_train_text[:50_000], "トークナイザのラウンドトリップが一致しない"
    d_windows, d_mask = make_evaluation_windows(d_val_ids, D_SEQUENCE_LENGTH)
    d_windows_hash = hashlib.sha256(d_windows.numpy().tobytes()).hexdigest()[:16]
    print(
        f"コーパス: {len(d_corpus):,} 文字(取得元 {d_corpus_metadata['source']})。学習用 {len(d_train_text):,} 文字({len(d_train_ids):,} トークン)、"
        f"検証 {len(d_val_text):,} 文字({d_total_bytes:,} バイト、評価窓 {len(d_windows)} 個、ハッシュ {d_windows_hash})"
    )
    print(
        f"実験 D の学習: {D_STEPS} ステップ x バッチ {D_BATCH_SIZE} x 系列長 {D_SEQUENCE_LENGTH} = {D_STEPS * D_BATCH_SIZE * D_SEQUENCE_LENGTH:,} トークン"
        f"(学習用の {D_STEPS * D_BATCH_SIZE * D_SEQUENCE_LENGTH / len(d_train_ids):.2f} エポック相当)。008 は 2,181 ステップ"
        f"({'同じ学習量' if D_STEPS == LEVELS['prod']['D_STEP_CANDIDATES'][0] else '同じ学習量ではない'})"
    )

    # --- 比較の相手: 008 の学習済みモデル(Hub の main)。同じ評価窓・同じ評価の関数(FP32)で測る ---
    from huggingface_hub import hf_hub_download

    _reference_config = json.loads(Path(hf_hub_download(REFERENCE_MODEL_REPO_ID, "config.json")).read_text(encoding="utf-8"))
    _reference_state = hf_hub_download(REFERENCE_MODEL_REPO_ID, "model_state.pt")
    assert _reference_config["vocabulary_size"] == d_tokenizer.vocab_size
    reference_model = GPTLanguageModel(
        vocabulary_size=_reference_config["vocabulary_size"],
        d_model=_reference_config["d_model"],
        num_layers=_reference_config["num_layers"],
        num_heads=_reference_config["num_heads"],
        d_ff=_reference_config["d_ff"],
        max_sequence_length=_reference_config["sequence_length"],
        positional_transform=RotaryPositionEmbedding(
            _reference_config["d_model"] // _reference_config["num_heads"], max_position=_reference_config["sequence_length"]
        ),
        normalization_factory=RMSNorm,
        feed_forward_factory=functools.partial(SwiGLUFeedForwardNetwork, _reference_config["d_model"], _reference_config["swiglu_d_ff"]),
        tie_embeddings=_reference_config["tie_embeddings"],
        dropout=_reference_config["dropout"],
    )
    reference_model.load_state_dict(torch.load(_reference_state, map_location="cpu"))
    reference_model = reference_model.to(device).eval()
    reference_bpb = evaluate_bits_per_byte(reference_model, d_windows, d_mask, d_total_bytes, device)
    reference_parameters = (count_non_embedding_parameters(reference_model), sum(p.numel() for p in reference_model.parameters()))

    # --- Mamba 言語モデルを 008 のレシピで 1 シード学習する ---
    torch.manual_seed(D_SEED)
    d_model_lm = MambaLanguageModel(
        D_VOCAB_SIZE, D_D_MODEL, D_NUM_LAYERS, state_dim=MAMBA_STATE_DIM, expand=MAMBA_EXPAND, conv_kernel=MAMBA_CONV_KERNEL,
        selective=True, discretization="zero_order_hold", use_checkpoint=True,
    )
    d_parameters = (count_non_embedding_parameters(d_model_lm), sum(p.numel() for p in d_model_lm.parameters()))
    assert abs(d_parameters[0] / reference_parameters[0] - 1) <= 0.05, (d_parameters, reference_parameters)  # 非埋め込みパラメータ数は 008 の 5% 以内
    d_eval_steps = tuple(sorted({max(1, round(f * D_STEPS)) for f in D_EVAL_FRACTIONS}))
    d_warmup = max(1, round(WARMUP_RATIO * D_STEPS))
    d_optimizer = AdamW(d_model_lm.parameters(), lr=D_LEARNING_RATE, weight_decay=D_WEIGHT_DECAY, foreach=True)
    d_schedule = functools.partial(
        compute_warmup_cosine_learning_rate, warmup_steps=d_warmup, total_steps=D_STEPS,
        peak_learning_rate=D_LEARNING_RATE, min_learning_rate=D_LEARNING_RATE * MIN_LEARNING_RATE_RATIO,
    )
    d_history = train_language_model(
        d_model_lm, d_train_ids, d_windows, d_mask, d_total_bytes, num_steps=D_STEPS, batch_size=D_BATCH_SIZE,
        sequence_length=D_SEQUENCE_LENGTH, learning_rate=D_LEARNING_RATE, eval_interval=D_STEPS, device=device, seed=D_SEED,
        optimizer=d_optimizer, learning_rate_schedule=d_schedule, gradient_clip_threshold=D_GRADIENT_CLIP_THRESHOLD,
        autocast_dtype=torch.float16 if USE_FP16_AUTOCAST else None,
        loss_scaler=DynamicLossScaler(D_INIT_LOSS_SCALE, growth_interval=D_LOSS_SCALE_GROWTH_INTERVAL) if USE_FP16_AUTOCAST else None,
        evaluate_at_final_step=True, eval_steps=d_eval_steps,
    )
    assert d_history["eval_step"][-1] == D_STEPS == len(d_history["train_loss"]) and d_history["eval_step"] == list(d_eval_steps)
    mamba_bpb = d_history["eval_bits_per_byte"][-1]  # 学習の最終ステップの重みで評価した値

    print(f"{RUN_TAG}実験 D(判定基準を設けない観察)")
    print(
        f"  非埋め込みパラメータ数 / 総数: Mamba {d_parameters[0]:,} / {d_parameters[1]:,}、008 のモデル {reference_parameters[0]:,} / {reference_parameters[1]:,}"
        f"(Mamba は 008 の {d_parameters[0] / reference_parameters[0]:.3f} 倍)"
    )
    print(
        f"  検証 bits-per-byte(同じ評価窓・同じ評価の関数): Mamba(最終ステップ {D_STEPS}) {mamba_bpb:.4f}、008 のモデル(Hub の main) {reference_bpb:.4f}。"
        f"途中の Mamba: " + ", ".join(f"{s}: {v:.3f}" for s, v in zip(d_history["eval_step"], d_history["eval_bits_per_byte"], strict=True))
    )
    print(
        f"  学習: 損失 {np.mean(d_history['train_loss'][:5]):.3f} -> {np.mean(d_history['train_loss'][-max(5, D_STEPS // 20):]):.3f}、"
        f"clipping の発動 {np.mean(d_history['gradient_clip_triggered']):.2f}、更新のスキップ {int(sum(d_history['step_skipped']))}、"
        f"最後の損失スケール {d_history['loss_scale'][-1]:g}"
    )
    # 状態を持ち回る生成によるサンプル(温度 0.8。比較のため 008 のモデルの同じ温度のサンプルも示す)
    d_model_lm = d_model_lm.to(device)
    D_SAMPLES = []
    for _k, _prompt in enumerate(D_PROMPTS):
        _ids = torch.tensor([d_tokenizer.encode(_prompt)], device=device)
        _mamba_out = d_model_lm.generate(_ids, D_GENERATION_TOKENS, temperature=0.8, seed=_k, use_state=True)
        _ref_out = reference_model.generate(_ids, D_GENERATION_TOKENS, temperature=0.8, seed=_k)
        D_SAMPLES.append((_prompt, d_tokenizer.decode(_mamba_out[0].tolist()), d_tokenizer.decode(_ref_out[0].tolist())))
    for _prompt, _mamba_text, _ref_text in D_SAMPLES:
        print(f"  サンプル(プロンプト {_prompt!r})\n    Mamba(状態を持ち回る生成): {_mamba_text!r}\n    008 のモデル: {_ref_text!r}")
    # 診断量: 状態を持ち回る貪欲な生成と文脈を再計算する貪欲な生成の一致(学習後の FP32 のモデル。FP64 での厳密な一致は 5.4 節で確かめた)
    _probe_ids = torch.tensor([d_tokenizer.encode(D_PROMPTS[0])], device=device)
    _with_state = d_model_lm.generate(_probe_ids, 24, temperature=0.0, use_state=True)
    _without_state = d_model_lm.generate(_probe_ids, 24, temperature=0.0, use_state=False)
    _match = float((_with_state == _without_state).float().mean())
    print(f"  診断量: 状態を持ち回る貪欲な生成と文脈を再計算する貪欲な生成の一致したトークンの割合 {_match:.3f}(FP32、24 トークン)")
    D_RESULT = {"history": d_history, "mamba_bpb": mamba_bpb, "reference_bpb": reference_bpb, "parameters": d_parameters, "samples": D_SAMPLES}

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.6))
    _loss = np.asarray(d_history["train_loss"])
    _kernel = np.ones(min(20, len(_loss))) / min(20, len(_loss))
    _axes[0].plot(_loss, alpha=0.3, label="train loss")
    _axes[0].plot(np.arange(len(_kernel) - 1, len(_loss)), np.convolve(_loss, _kernel, mode="valid"), label="moving average (20)")
    _axes[0].set(xlabel="step", ylabel="cross entropy [nats/token]", title=f"{PLOT_TAG}Experiment D: training loss (Mamba LM)")
    _axes[0].legend()
    _axes[1].plot(d_history["eval_step"], d_history["eval_bits_per_byte"], marker="o", label="Mamba LM")
    _axes[1].axhline(reference_bpb, color="gray", linestyle="--", label="008 GPT (2181 steps)")
    _axes[1].set(xscale="log", xlabel="step", ylabel="validation bits-per-byte", title=f"{PLOT_TAG}validation bits-per-byte")
    _axes[1].legend()
    plt.tight_layout()
    plt.show()
    del d_model_lm, d_optimizer, reference_model
    empty_device_cache()
D_SECONDS = time.time() - _t0_d
print(f"実験 D {D_SECONDS / 60:.1f} 分(見積もり {PLAN_ESTIMATES[SELECTED_PLAN['plan']]['d'] / 60:.1f} 分)")
```


    tokenizer.json:   0%|          | 0.00/661k [00:00<?, ?B/s]



    corpus.txt: reconstructing file:   0%|          |  0.00B / 24.3MB            



    corpus.txt: downloading bytes:           |  0.00B            



    metadata.json:   0%|          | 0.00/43.4k [00:00<?, ?B/s]


    コーパス取得元: kojikojiprg/ai-theories-corpus-en-pretraining(Hugging Face Hub)
    コーパス: 24,214,546 文字(取得元 hub)。学習用 23,003,819 文字(5,956,594 トークン)、検証 1,210,727 文字(1,214,117 バイト、評価窓 1240 個、ハッシュ 91484c872b8945c0)
    実験 D の学習: 2181 ステップ x バッチ 32 x 系列長 256 = 17,866,752 トークン(学習用の 3.00 エポック相当)。008 は 2,181 ステップ(同じ学習量)



    config.json:   0%|          | 0.00/305 [00:00<?, ?B/s]



    model_state.pt: reconstructing file:   0%|          |  0.00B / 21.0MB            



    model_state.pt: downloading bytes:           |  0.00B            


    実験 D(判定基準を設けない観察)
      非埋め込みパラメータ数 / 総数: Mamba 3,066,368 / 5,163,520、008 のモデル 3,149,056 / 5,246,208(Mamba は 008 の 0.974 倍)
      検証 bits-per-byte(同じ評価窓・同じ評価の関数): Mamba(最終ステップ 2181) 1.6525、008 のモデル(Hub の main) 1.6681。途中の Mamba: 273: 2.133, 545: 1.871, 1090: 1.725, 2181: 1.652
      学習: 損失 9.061 -> 3.461、clipping の発動 0.55、更新のスキップ 0、最後の損失スケール 131072
      サンプル(プロンプト 'The history of')
        Mamba(状態を持ち回る生成): 'The history of sovereignty, and in the Decusus was a partial tarkhine of the same name, the newly formed American relationship with the Allitional Night. Deblie Driver and its V'
        008 のモデル: 'The history of sounds of the song "Part Deck", was described as "emprokhine" for "new was disguisive and “inite like a sea thrill to be polarizated'
      サンプル(プロンプト 'In mathematics, a function')
        Mamba(状態を持ち回る生成): 'In mathematics, a functionic technology based on vebrogenation and biochemist groups and all manufacturer interstellar arts for both individuals. Nazi participants in the daily years of the Mune Dlema'
        008 のモデル: 'In mathematics, a function of the firearms of the Aliyaho Marianuč site. Nigeria and approximately 40,000 families can be displayed by China, was reported that Israel was dying a bombing in Sogneha. B'
      サンプル(プロンプト 'The city was founded')
        Mamba(状態を持ち回る生成): 'The city was founded by the National People Act of 1971, and it was considered their exception to the title of Mauritius (two of the four four people) on the period, as the producers of the Behind Spider-M'
        008 のモデル: 'The city was founded by the holidays. The idea was later considered their exception to be legally known as a kit.\n\nThere is not the result of the initiative, as the result was frisible. The website for the'
      診断量: 状態を持ち回る貪欲な生成と文脈を再計算する貪欲な生成の一致したトークンの割合 1.000(FP32、24 トークン)



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/023_state_space_model_mamba/output_38_9.png)
    


    実験 D 37.2 分(見積もり 38.9 分)




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/06_architectures/023_state_space_model_mamba.ipynb
