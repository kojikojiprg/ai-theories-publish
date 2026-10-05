---
title: "State Space Model / Mamba / State Space Model and Mamba(実装・実験編 5/6)"
---

この記事は後編(実装・実験編 5/6)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/023_state_space_model_mamba-practice-4)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/023_state_space_model_mamba-practice-6)。

### 6.11 不変条件のアサーションと`SMOKE_TEST`の配線

- 実効水準の照合: 実際に学習した種類・シード・ステップ数・評価のステップが、5.3 節で印字した水準と 6.4 節で選ばれた計画と一致する。較正も本番と同じ学習ステップ数で行われた。
- 条件間で揃えた量: 較正・計測のシードが実験のシードと共有されない。評価集合・較正用の集合が互いに別である。
- 判定の記録の整合: 各実験の前提条件がすべて記録されている。


```python
# --- 実効水準の照合(SMOKE_TEST の配線): 5.2 節で印字した水準と、選ばれた計画が、実際の学習・評価で使われた値と一致する ---
assert SELECTED_PLAN in PLANS and NUM_STEPS == {"A": SELECTED_PLAN["steps_a"], "C": SELECTED_PLAN["steps_c"]}
assert EVAL_STEPS == {key: eval_steps_for(steps) for key, steps in NUM_STEPS.items()}
assert NUM_STEPS["A"] in (CFG["STEPS_A"], CFG["STEPS_A"] // 2) and NUM_STEPS["C"] in (CFG["STEPS_C"], CFG["STEPS_C"] // 2)
assert NUM_STEPS["A"] == STEPS_RATIO_A_TO_C * NUM_STEPS["C"]  # T_A と T_C の比は本番と同じ
assert SEEDS_A in (CFG["SEEDS_FULL"], CFG["SEEDS_CUT"]) and SEEDS_C in (CFG["SEEDS_FULL"], CFG["SEEDS_CUT"])
assert len(EVAL_A_TOKENS) == CFG["EVAL_SEQUENCES"] and all(len(v[0]) == CFG["EVAL_SEQUENCES"] for v in EVAL_C.values())
assert len(CALIBRATION_A_TOKENS) == len(CALIBRATION_C_TOKENS) == CFG["CALIBRATION_SEQUENCES"]
assert BOOTSTRAP_RESAMPLES == CFG["BOOTSTRAP_RESAMPLES"] and tuple(B_LEVELS) == tuple(CFG["B_LEVELS"])
assert set(RUNS) == {(kind, s) for kind in ("A1", "A2") for s in range(SEEDS_A)} | {(kind, s) for kind in ("C1", "C2") for s in range(SEEDS_C)}
for (_kind, _seed), _record in RUNS.items():
    _steps = NUM_STEPS[_kind[0]]
    assert _record["num_steps"] == _steps and len(_record["train_loss"]) == _steps and _record["eval_step"] == list(EVAL_STEPS[_kind[0]])
    assert _record["learning_rate"] == LEARNING_RATE[_kind] and _record["kind"] == _kind and _record["seed"] == _seed
for _kind in KINDS:
    assert CALIBRATION[_kind]["num_steps"] == NUM_STEPS[_kind[0]]  # 較正は本番と同じ学習ステップ数で行った
    assert CALIBRATION[_kind]["chosen"] in CALIBRATION[_kind]["grid"]
assert all(len(B_RAW[c][L]) == B_SWEEPS and all(len(sweep) == B_REPEATS for sweep in B_RAW[c][L]) for c in B_RAW for L in B_LEVELS)
if D_STEPS is not None:
    assert D_RESULT is not None and len(D_RESULT["history"]["train_loss"]) == D_STEPS
else:
    assert D_RESULT is None
# --- 条件間で揃えた量 ---
assert min(CALIBRATION_SEED_INDEX, TIMING_SEED_INDEX) >= max(SEEDS_A, SEEDS_C) and CALIBRATION_SEED_INDEX != TIMING_SEED_INDEX  # 較正・計測のシードは実験のシードと共有しない
assert DATA_SEED_BASE["A"] != DATA_SEED_BASE["C"] and EVAL_SEED_A != CALIBRATION_SEED_A and EVAL_SEED_C != CALIBRATION_SEED_C
_evaluation_hashes = {
    "A": hashlib.sha256(EVAL_A_TOKENS.numpy().tobytes()).hexdigest()[:12],
    "C": {L: hashlib.sha256(v[0].numpy().tobytes()).hexdigest()[:12] for L, v in EVAL_C.items()},
}
assert not torch.equal(CALIBRATION_A_TOKENS, EVAL_A_TOKENS) and not torch.equal(CALIBRATION_C_TOKENS, EVAL_C[C_TRAIN_LENGTH][0])  # 較正用の集合は評価集合と別
# --- 判定の記録の整合 ---
for _name, _keys in (("A", A_PRECONDITIONS), ("B", B_PRECONDITIONS), ("C", C_PRECONDITIONS)):
    assert all(k in precondition_status for k in _keys), (_name, _keys)
print(
    f"{RUN_TAG}実効水準の照合: 計画 {SELECTED_PLAN['plan']}(A のシード {SEEDS_A}・C のシード {SEEDS_C}・学習 A {NUM_STEPS['A']}・C {NUM_STEPS['C']} ステップ・"
    f"D {D_STEPS})が、学習した全記録({len(RUNS)} 回)・較正({len(KINDS)} 条件)・実験 B の記録({B_SWEEPS} x {B_REPEATS} 回)・実験 D と一致: OK"
)
print(f"評価集合のハッシュ: {json.dumps(_evaluation_hashes)}")
```

    実効水準の照合: 計画 0(A のシード 5・C のシード 5・学習 A 4000・C 2000 ステップ・D 2181)が、学習した全記録(20 回)・較正(4 条件)・実験 B の記録(8 x 8 回)・実験 D と一致: OK
    評価集合のハッシュ: {"A": "3850f5298f83", "C": {"32": "e53637bd342d", "64": "2859d2df2044", "128": "e28616daba9d", "256": "a4479369abe0", "512": "101f797f0c28"}}


### 6.12 判定結果の一覧

この節は、本番の実行(6.3〜6.11 節)の判定結果である。条件を直して再実行した実験 B・C の判定は 6.13 節に記す。


```python
print(f"{RUN_TAG}判定結果(計画 {SELECTED_PLAN['plan']}、学習 A {NUM_STEPS['A']}・C {NUM_STEPS['C']} ステップ、実験 A・C のシード {SEEDS_A}・{SEEDS_C})")
print(
    f"  実験 A: {A_VERDICT}(判定関数の結果 {A_COMPUTED}、Delta_A = {DELTA_A:+.4f}、閾値 {SIGMA_MULTIPLIER * SIGMA_A['sigma']:.4f}、"
    f"前提条件 {json.dumps({k: precondition_status[k] for k in A_PRECONDITIONS})})"
)
print(
    f"  実験 B: {B_VERDICT}(判定関数の結果 {B_COMPUTED}、Delta_B = {DELTA_B:+.4f}、閾値 {SIGMA_MULTIPLIER * SIGMA_B:.4f}、"
    f"前提条件 {json.dumps({k: precondition_status[k] for k in B_PRECONDITIONS})})"
)
print(
    f"  実験 C: {C_VERDICT}(判定関数の結果 {C_COMPUTED}、Delta_C = {DELTA_C:+.4f}、閾値 {SIGMA_MULTIPLIER * SIGMA_C['sigma']:.4f}、"
    f"前提条件 {json.dumps({k: precondition_status[k] for k in C_PRECONDITIONS})})"
)
print("  実験 D: 判定基準を設けない観察(上の 6.10 節の出力)" if D_STEPS is not None else "  実験 D: 省略(選ばれた計画)")
_total = time.time() - NOTEBOOK_START_TIME
print(
    f"全体の経過時間 {_total / 60:.1f} 分(予算 {SESSION_BUDGET_SECONDS / 60:.0f} 分、6.4 節の見積もり "
    f"{(ELAPSED_AT_SELECTION + PLAN_ESTIMATES[SELECTED_PLAN['plan']]['total']) / 60:.1f} 分)"
)
```

    判定結果(計画 0、学習 A 4000・C 2000 ステップ、実験 A・C のシード 5・5)
      実験 A: 支持(判定関数の結果 支持、Delta_A = +0.3055、閾値 0.1562、前提条件 {"P0(A)": true, "P1(A)": true, "P2(A)": true})
      実験 B: 支持(判定関数の結果 支持、Delta_B = +0.5788、閾値 0.0195、前提条件 {"P-B1": true, "P-B2": true})
      実験 C: 前提不成立(判定関数の結果 支持、Delta_C = +0.7434、閾値 0.0538、前提条件 {"P0(C)": true, "P1(C)": false, "P2(C)": false})
      実験 D: 判定基準を設けない観察(上の 6.10 節の出力)
    全体の経過時間 84.0 分(予算 120 分、6.4 節の見積もり 100.6 分)


### 6.13 実験 B・C の再実行(条件を直した再実行)

6.3〜6.12 節が本番の実行、この 6.13 節が条件を直した再実行(実験 B・C)である。6.13 節は本番とは別のセッションで実行した。5.1 節の実行環境の印字は、6.13 節のセッションのものである。実験 A と実験 D は再実行していない。

- **実験 C**: 本番の実行(6.5・6.6・6.9 節)で前提不成立になったので、判定基準と前提条件の定義・閾値を変えずに、学習率の採用の規則だけを直して再実行した。6.13 節のセル出力に印字される「初回」の値は、本番の実行の出力(6.5・6.6・6.9 節)の値と同じである。ただし、所要時間の見込みに使った実測(C1 48 秒、C2 22 秒)は、6.6 節の出力(C1 47〜48 秒、C2 21〜22 秒)を丸めた値である。
- **実験 B**: 本番の実行(6.8 節)では前提条件が成立し、支持だった。一方、**同じ条件の別の実行では、P-B2 が Mamba の層の系列長 1024 の 1 水準で不成立だった(標準誤差と平均の比 0.0823、閾値 0.05)。** 計測のばらつきが閾値の近くにあるため、反復回数を増やし、ガベージコレクションを止めた条件でも計測した。6.13 節のセル出力の「実験 B: 初回の最終判定 前提不成立」は、この別の実行を指す。

#### 6.13.1 実験 C が前提不成立になった根拠(本番の実行の出力の値)

- **6.6 節の出力**: C1 のシード 2 で学習が崩壊した(訓練損失 1.864 → 2.766、学習時の系列長 $L_{\mathrm{train}}$ の正解率 0.064)。ほかの 4 シードは $L_{\mathrm{train}}$ の正解率 1.000 だった。
- **6.9 節の出力**: P1(C)は不成立(不成立の学習は C1 のシード 2)。P2(C)も不成立(C1 の $L_{\mathrm{train}}$ の正解率のシード平均 0.8128 で、閾値 0.90 未満。C2 は 1.0000)。
- **6.5 節の出力(較正)**: C1 の較正で選ばれた学習率 $3.2 \times 10^{-2}$ の 1 つ上の格子点 $6.4 \times 10^{-2}$ は、較正用の集合の負の対数尤度が 2.7709 で、偶然の水準($\ln 16 \approx 2.7726$)とほぼ同じ(学習が崩壊していた)。C2 も、選ばれた学習率 $8 \times 10^{-3}$ の 1 つ上の格子点 $1.6 \times 10^{-2}$ の負の対数尤度が 2.2711 だった。一方、選ばれた学習率とその 1 つ下の格子点の負の対数尤度には差がなく(C1 は 0.0 と 0.0、C2 は 0.0001 と 0.0002)、較正の指標では安定性の余裕を区別できなかった。

前提不成立の実験について、判定関数が返した参考値は、結論として扱わない。

#### 6.13.2 修正の内容と理由(宣言)

**この節の内容は再実行の前に確定させ、再実行の結果を見た後に変更しない。** 変更するのは実験条件だけである。判定基準(対比量の定義・閾値の導出式・期待する差の方向)と前提条件の定義・閾値は変更しない。

##### 実験 C: 学習率の採用の規則

**変更しないもの**: シード(0〜4)、学習ステップ数 $T_C = 2000$、バッチ(64 系列)、モデル、課題、評価集合(6.13.4 節で本番の実行のハッシュと照合する)、評価する系列長、$L_{\mathrm{test}} = 128$、学習率の格子と拡張の規則(6.1 節「学習率の較正」)、較正専用のシード(90)、P0(C)・P1(C)・P2(C)の定義と閾値、対比量 $\Delta_C$ と閾値の導出。

**変更するもの**: 較正で最良の学習率を求めた後の **採用の規則** だけである。次の規則を **両条件(C1・C2)に同じように** 適用する。

- 最良の格子点の 1 つ上の格子点が「崩壊」しているときは、最良の 1 つ下の格子点を採用する(崩壊する学習率から、公比 1 つ分の余裕を取る)。
- 1 つ上の格子点が崩壊していないときは、最良の格子点をそのまま採用する。最良の格子点が格子の端で、1 つ上または 1 つ下の格子点がないときも、最良の格子点を採用する。
- 「崩壊」の定義: (a)学習の健全性(P1(C)と同じ定義。訓練損失がすべてのステップで有限であり、最後の 5% のステップの平均が最初の 5% の平均を下回ること)が不成立、または、(b)較正用の集合の負の対数尤度が、偶然の水準の負の対数尤度 $\ln 16$ の 1/2 以上である。

**P0(C)の読み方**: 6.1 節の P0(C)は「選んだ学習率が格子の内点であること」であり、ここでの「選ぶ」は、較正の指標(較正用の集合の負の対数尤度)が最小の格子点を求める手順を指す。再実行でも、P0(C)は **最良の格子点の位置**(拡張後の格子の内点かどうか)で判定し、採用する学習率(最良の 1 つ下)の位置では判定しない。したがって、採用する学習率が格子の下端になっても、P0(C)は成立しうる。この読み方は 6.1 節の文言と矛盾しない(6.1 節の括弧書き「拡張後も最良が端なら不成立」も、最良の格子点の位置を見ている)ので、解釈は変えていない。

**改訂の理由(観測結果の方向に依存しない)**: 本番の実行の較正では、最良の格子点とその 1 つ下の格子点の負の対数尤度に差がなく(どちらも約 0)、較正の指標では安定性の余裕を区別できなかった。最良の格子点の 1 つ上の格子点は学習が崩壊しており、最良の格子点は崩壊する学習率の隣にあった。その学習率で、本番の 5 シードのうち 1 つ(C1 のシード 2)が崩壊した(6.13.1 節)。理由は、学習の崩壊と、較正の指標が余裕を区別できないことである。**対比量の向きや大きさ、条件間の差を理由にしていない。**

**較正に関する前提条件の対象外になるものと、偏りの向き**:

- 採用の規則で決めた学習率(最良の 1 つ下)は、較正の指標が最小の格子点ではないので、較正に関する前提条件 P0(C)(最良の格子点の位置を見る)の対象外である。
- 採用する学習率が最良からずれることで、対比量 $\Delta_C$ がどちらの向きに偏るかは **断定できない。** 学習率が下がると学習が遅くなる方向に働き、学習時の系列長での習得(P2(C))が遅れうるが、その影響の大きさは条件ごとに異なりうるので、$\Delta_C$ の向きは決められない。習得が遅れて P2(C)が不成立になれば、前提不成立として記録する。

##### 実験 B: 計測のばらつき

**変更しないもの**: 系列長の水準、$d_{\mathrm{model}} = 128$、掃引の回数(8)、実行の順序の入れ替え、計測の直前のウォームアップの手順、P-B1・P-B2 の定義と閾値、べき指数の推定方法、対比量 $\Delta_B$ と $\sigma_B$ の導出。

**変更するもの**:

1. **掃引あたりの反復回数を 8 から 32 にする。** 各条件・各水準の反復の総数は 64 から 256 になる。P-B2 の標準誤差は反復の総数の平方根に反比例するので、標準偏差が同じなら、標準誤差は半分になる。6.1 節の「反復回数は本番の前に宣言して固定し、前提が満たされるまで反復を増やすことはしない」は本番の実行の宣言であり、再実行ではこの 32 回を再実行の前に固定する。再実行の後に反復をさらに増やすことはしない。
2. **計測の区間ではガベージコレクション(Garbage Collection)を止める**(`gc.disable()`)。各掃引の前に`gc.collect()`を呼び、計測の後に必ず元に戻す。

**改訂の理由(観測結果の方向に依存しない)**: 本番の実行と同じ条件の別の実行で、P-B2 が 10 個の(条件、系列長)の組のうち 1 つ(Mamba の層の系列長 1024)で不成立だった。本番の実行(6.8 節)でも、P-B2 の標準誤差と平均の比の最大は 0.0348 で、閾値 0.05 に近かった。これは計測の精度の問題であり、対比量の向きにも大きさにも依存しない。反復回数を増やせば平均の標準誤差は小さくなる。ガベージコレクションは、計測の区間に不定期な停止を持ち込みうる一般的な要因である。

**注意**: P-B2 が不成立になった原因は特定できておらず、この修正で P-B2 が成立する保証はない。再び不成立になれば、前提不成立として記録する。

**追加する診断量(判定には使わない)**: 水準・条件ごとに、時間が中央値の 2 倍を超えた反復の数(全反復の数に対する)と、最大値と中央値の比を印字する。

##### 修正の根拠が観測結果の方向に依存しないこと

- 実験 C の修正の根拠は、学習の崩壊と、較正の指標が余裕を区別できなかったことである。実験 B の修正の根拠は、計測の精度(P-B2)である。
- 対比量の向きや大きさ(判定関数が返した参考値を含む)を、どちらの修正の理由にもしていない。「ある技術に効果が出なかったから条件を変える」という理由は使っていない。
- 再実行は **1 回限り** とする。再び前提不成立になった場合は、そのまま記録し、さらに再実行はしない。実験 A と実験 D は再実行しない。

##### 修正の一覧

| 項目 | 本番の実行 | 再実行 | 理由 |
|---|---|---|---|
| 実験 C: 採用する学習率 | 最良の格子点 | 最良の 1 つ上の格子点が崩壊していれば、最良の 1 つ下の格子点(C1・C2 とも) | 学習の崩壊を避ける余裕を取る。較正の指標では余裕を区別できなかった |
| 実験 B: 掃引あたりの反復回数 | 8 | 32 | P-B2 の標準誤差を小さくする |
| 実験 B: 計測の区間のガベージコレクション | 有効 | 無効(各掃引の前に`gc.collect()`、計測の後に元に戻す) | 計測の区間への不定期な停止を避ける |
| 上記以外のすべて | — | 変更しない | — |

#### 6.13.3 パイロットの記録(実験 C の再実行の前、ローカル)

再実行で採用する見込みの C2(Transformer)の学習率 $4 \times 10^{-3}$ で、学習時の系列長 $L_{\mathrm{train}}$ を習得できるかを、再実行の前にローカル(Apple M4、MPS、FP32)で確かめた。

- **条件**: C2、学習率 $4 \times 10^{-3}$、2000 ステップ、5 シード(パイロット用のシード番号 100〜104。実験のシード 0〜4・較正のシード 90 と共有しない)。モデル・課題・バッチ・評価集合は本番と同じ。
- **出力した量**: $L_{\mathrm{train}}$ の正解率のシード平均が 0.90 以上か(真偽)、$L_{\mathrm{train}}$ の正解率のシード間の標準偏差、学習の健全性(P1(C)と同じ定義)。**$L_{\mathrm{test}}$ を含む他の系列長は評価していない。** C1 と C2 の比較は見ていない。
- **結果**: 平均が 0.90 以上か: **真**。シード間の標準偏差: 0.0000。学習の健全性: 5 シードすべてで成立。
- C1 の学習率 $1.6 \times 10^{-2}$ は、6.2.1 節の旧パイロット(2000 ステップ、シード 100〜104)で、$L_{\mathrm{train}}$ の正解率のシード平均が 0.90 以上か(真)、シード間の標準偏差 0.0000、学習の健全性(すべての格子点で成立)を確認済みなので、再実行していない。

パイロットが出力したのは、単一の条件の値だけで、対比量の向きの情報を含まない。採用の規則(6.13.2 節)は、このパイロットの結果で変えていない。

#### 6.13.4 再実行の確認と見込み

最初のコードセルは、学習も計測もせずに、次を行う。識別子`[RERUN_BC]`は、6.13 節のすべてのコードセルの出力の冒頭に印字する。

- 必要な名前(5.1〜5.5 節のセルが定義するもの)が定義済みであることを確かめる。
- 実行環境を印字する。`SMOKE_TEST`が偽でデバイスが`cuda`でなければ、例外で停止する。
- 再実行の水準(実験 C はシード 5・$T_C = 2000$、実験 B は掃引 8 回 x 反復 32 回)を定め、実験 C の評価集合のハッシュが本番の実行(6.11 節)の値と一致することを確かめる。
- 判定・前提条件・診断量・図と、実験 B の層・計測の関数・ウォームアップは、本番の実行のコードセル(6.3・6.8・6.9 節)の本文を読み込み、そのまま実行する(関数を複製して書き換えない)。読み込んだ本文の SHA-256 が本番の実行のものと一致することを確かめる。
- 本番の実行の実測(6.6 節・6.8 節の出力)から、再実行の所要時間の見込みを印字する。計画の自動選択は設けない。


```python
import ast

RERUN_TAG = "[RERUN_BC] "
print(f"{RERUN_TAG}実験 B・C の再実行(前提不成立への対処)。このセルは、実行環境の記録と、再実行できる状態かの確認だけを行う(学習も計測もしない)")

# --- 6.13 節のセルが使う名前のうち、5.1〜5.5 節のセルが定義するもの(6.3〜6.12 節の変数には依存しない) ---
_REQUIRED_NAMES = (  # noqa: SIM905  名前を 1 行に 1 つずつ書くと長くなるため、空白区切りの文字列にしている
    # 5.1 節(セットアップ)
    "SMOKE_TEST device print_execution_environment time torch "
    # 5.2 節(インポートと共通の関数)
    "KeyValueCache MambaBlock MultiHeadAttention PLOT_TAG ROOT RUN_TAG count_non_embedding_parameters "
    "create_causal_mask empty_device_cache fit_power_law_exponent hashlib json math np "
    "paired_bootstrap_ratio_of_sums plt precondition_status rounded timed_call "
    # 5.3 節(定数と水準)
    "BOOTSTRAP_RESAMPLES BOOTSTRAP_SEED B_DECODE_CONTEXTS B_D_MODEL B_NUM_HEADS B_WARMUP_REPEATS "
    "CALIBRATION_POINTS_WORST CALIBRATION_SEED_INDEX CFG C_MASTERY_MIN C_TRAIN_LENGTH LEVELS LR_GRID_MULTIPLIERS "
    "LR_GRID_RATIO MAMBA_CONV_KERNEL MAMBA_EXPAND MAMBA_STATE_DIM P0A_MAX_DRIFT_SIGMA P0B_MAX_RELATIVE_STDERR "
    "REPORTING_MARGIN_SECONDS RUN_OVERHEAD_SECONDS SESSION_BUDGET_SECONDS SIGMA_MULTIPLIER TASK_C TIMING_SEED_INDEX "
    # 5.4 節(評価集合)
    "CURVE_SEQUENCES C_LENGTHS C_TEST_LENGTH EVAL_C "
    # 5.5 節(ヘルパー)
    "build_model combined_sigma eval_steps_for judge learning_precondition lr_grid_for run_task_c"
).split()
_missing = [name for name in _REQUIRED_NAMES if name not in globals()]
if _missing:
    raise RuntimeError(
        f"{RERUN_TAG}定義されていない名前がある: {_missing}。6.13.1 節の手順どおり、5.1〜5.5 節のコードセルを上から順に実行してから、"
        "このセルを実行すること。学習も計測もしていない。"
    )
if "SELECTED_PLAN" in globals():  # 6.4 節で定義される変数。6.3〜6.12 節のセルがこのセッションで実行されている
    raise RuntimeError(
        f"{RERUN_TAG}6.3〜6.12 節のセルがこのセッションで実行されている。初回の出力を上書きしないよう、新しいセッションで、"
        "6.13.1 節の手順どおりに実行し直すこと。学習も計測もしていない。"
    )

# --- 実行環境の記録(初回の印字は 6.13.2 節に文章で残してある。5.1 節のセルの出力はこのセッションのものに置き換わる) ---
print_execution_environment(device)
if not SMOKE_TEST and device.type != "cuda":
    raise RuntimeError(
        f"{RERUN_TAG}SMOKE_TEST が False なのにデバイスが {device.type} である。ランタイムのタイプを T4 GPU にして、"
        "セッションを開き直し、6.13.1 節の手順どおりに 5.1 節のセルから実行し直すこと。学習も計測もしていない。"
    )

# --- 再実行の水準(計画の自動選択は設けない。削る段階は適用しない) ---
RERUN_SEEDS_C = CFG["SEEDS_FULL"]  # 実験 C のシード数 5(スモークテストは 3)
RERUN_STEPS_C = CFG["STEPS_C"]  # 実験 C の学習ステップ数 T_C = 2000(スモークテストは 24)
B_REPEATS_MULTIPLIER = 4  # 実験 B の掃引あたりの反復回数の、初回に対する倍率(スモークテストも同じ倍率)
B_REPEATS_RERUN = B_REPEATS_MULTIPLIER * CFG["B_REPEATS"]  # 初回の 8 の 4 倍 = 32(スモークテストは 3 の 4 倍 = 12)
assert SMOKE_TEST or (RERUN_SEEDS_C, RERUN_STEPS_C, CFG["B_SWEEPS"], B_REPEATS_RERUN) == (5, 2000, 8, 32)
print(
    f"{RERUN_TAG}水準: 実験 C のシード 0〜{RERUN_SEEDS_C - 1}({RERUN_SEEDS_C} 個)・学習 {RERUN_STEPS_C} ステップ、"
    f"実験 B の系列長 {CFG['B_LEVELS']}・掃引 {CFG['B_SWEEPS']} 回 x 反復 {B_REPEATS_RERUN} 回(初回は {CFG['B_REPEATS']} 回)"
)

# --- 実験 C の評価集合が、初回と同じであること(初回の 6.11 節の出力のハッシュと照合する) ---
FIRST_RUN_EVAL_HASHES_C = {  # 6.11 節の出力(初回)
    32: "e53637bd342d",
    64: "2859d2df2044",
    128: "e28616daba9d",
    256: "a4479369abe0",
    512: "101f797f0c28",
}
_eval_hashes_c = {length: hashlib.sha256(tokens.numpy().tobytes()).hexdigest()[:12] for length, (tokens, _, _) in EVAL_C.items()}
if SMOKE_TEST:
    print(f"{RERUN_TAG}実験 C の評価集合のハッシュ: {_eval_hashes_c}(スモークテストは評価の系列数が初回と違うので、照合しない)")
else:
    assert _eval_hashes_c == FIRST_RUN_EVAL_HASHES_C, (_eval_hashes_c, FIRST_RUN_EVAL_HASHES_C)
    print(f"{RERUN_TAG}実験 C の評価集合のハッシュが、初回(6.11 節の出力)と一致: OK {_eval_hashes_c}")

# --- 初回のコードセルの本文の読み込み ---
# 判定・前提条件・診断量・計測の手順を初回と完全に同じコードで計算するため、初回のコードセル(6.3・6.8・6.9 節)の本文を
# ノートブックのファイルから読み込み、そのまま実行する(複製して書き換えない)。本文の SHA-256 の先頭 16 文字が、初回の
# 本番実行のコードセルのものと一致しなければ、学習の前に停止する。
FIRST_RUN_NOTEBOOK = ROOT / "theories" / "06_architectures" / "023_state_space_model_mamba.ipynb"


def first_run_cell_source(first_line: str, expected_digest: str) -> str:
    cells = json.loads(FIRST_RUN_NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    matches = ["".join(c["source"]) for c in cells if c["cell_type"] == "code" and "".join(c["source"]).startswith(first_line)]
    if len(matches) != 1:
        raise RuntimeError(f"{RERUN_TAG}初回のコードセル({first_line!r} で始まるもの)が {len(matches)} 個見つかった(1 個のはず): {FIRST_RUN_NOTEBOOK}")
    digest = hashlib.sha256(matches[0].encode("utf-8")).hexdigest()[:16]
    if digest != expected_digest:
        raise RuntimeError(f"{RERUN_TAG}初回のコードセル({first_line!r} で始まるもの)の本文が変更されている: SHA-256 {digest}(初回 {expected_digest})。学習の前に停止する。")
    return matches[0]


_source_6_3 = first_run_cell_source("# --- 実験 B の層・水準・計測の関数", "3efe9d2c0204f79c")  # 6.3 節
_source_6_8 = first_run_cell_source("_t0_b = time.time()", "188d09e72385875d")  # 6.8 節(実験 B)
FIRST_RUN_SOURCE_C = first_run_cell_source("_seeds = list(range(SEEDS_C))", "33b11cd858888f7e")  # 6.9 節(実験 C)

# 6.3 節: 実験 B の層・水準・計測の関数の定義(最初の with 文、つまりウォームアップの前まで)
_lines_6_3 = _source_6_3.splitlines()
_first_with = next(node for node in ast.parse(_source_6_3).body if isinstance(node, ast.With))
FIRST_RUN_B_DEFINITIONS = "\n".join(_lines_6_3[: _first_with.lineno - 1])
# 6.8 節: 計測の直前のウォームアップ(B_RAW の定義の前まで)と、計測の後の前提条件・判定・診断量・図(series_in_time_order の定義から最後まで)
FIRST_RUN_B_WARMUP = _source_6_8[: _source_6_8.index("B_RAW = {")]
FIRST_RUN_B_ANALYSIS = _source_6_8[_source_6_8.index("def series_in_time_order") :]
print(f"{RERUN_TAG}初回のコードセル(6.3・6.8・6.9 節)の本文を読み込み、SHA-256 が初回と一致することを確認した")

# --- 再実行の所要時間の見込み(初回の実測から。計画の自動選択は設けない) ---
FIRST_RUN_TRAIN_SECONDS = {"C1": 48.0, "C2": 22.0}  # 1 学習あたり。6.6 節の出力(初回)
FIRST_RUN_B_MINUTES = 0.9  # 実験 B の計測(掃引 8 回 x 反復 8 回、ウォームアップと診断量を含む)。6.8 節の出力(初回)
_prod = LEVELS["prod"]
_per_pair = sum(FIRST_RUN_TRAIN_SECONDS.values()) + 2 * RUN_OVERHEAD_SECONDS  # C1 と C2 を 1 回ずつ学習する時間
ESTIMATE_SECONDS = {
    "calibration_best": len(LR_GRID_MULTIPLIERS) * _per_pair,  # 格子 3 点 x 2 条件
    "calibration_worst": CALIBRATION_POINTS_WORST * _per_pair,  # 拡張が起きた場合(格子 3 点 + 拡張 1 点)
    "main": _prod["SEEDS_FULL"] * _per_pair,
    "b": FIRST_RUN_B_MINUTES * 60 * B_REPEATS_MULTIPLIER,  # 反復回数に比例
    "margin": REPORTING_MARGIN_SECONDS,
}
ESTIMATE_TOTAL_SECONDS = ESTIMATE_SECONDS["calibration_worst"] + ESTIMATE_SECONDS["main"] + ESTIMATE_SECONDS["b"] + ESTIMATE_SECONDS["margin"]
assert ESTIMATE_TOTAL_SECONDS <= SESSION_BUDGET_SECONDS
print(
    f"{RERUN_TAG}再実行の所要時間の見込み(本番の設定。初回の実測 C1 {FIRST_RUN_TRAIN_SECONDS['C1']:.0f} 秒・C2 {FIRST_RUN_TRAIN_SECONDS['C2']:.0f} 秒・"
    f"実験 B {FIRST_RUN_B_MINUTES} 分から): 実験 C の較正 {ESTIMATE_SECONDS['calibration_best'] / 60:.1f}〜{ESTIMATE_SECONDS['calibration_worst'] / 60:.1f} 分、"
    f"実験 C の学習と評価 {ESTIMATE_SECONDS['main'] / 60:.1f} 分、実験 B {ESTIMATE_SECONDS['b'] / 60:.1f} 分、判定・図などの余裕 {ESTIMATE_SECONDS['margin'] / 60:.1f} 分"
    f" -> 合計 最大 {ESTIMATE_TOTAL_SECONDS / 60:.1f} 分(予算 {SESSION_BUDGET_SECONDS / 60:.0f} 分。セットアップを除く)。計画の自動選択は設けない"
)
RERUN_START_TIME = time.time()
```

    [RERUN_BC] 実験 B・C の再実行(前提不成立への対処)。このセルは、実行環境の記録と、再実行できる状態かの確認だけを行う(学習も計測もしない)
    実行環境 / Execution environment
      Python                                 : 3.13.15
      OS / platform                          : Linux-6.6.122+-x86_64-with-glibc2.39
      torch                                  : 2.13.0+cu130
      torch のビルド時の CUDA                : 13.0
      cuDNN                                  : 92000
      デバイス / device                      : cuda
      GPU 名 / GPU name                      : Tesla T4
      compute capability                     : 7.5
      GPU の総メモリ (GiB)                   : 14.56
      コミット / git commit                  : d37a14f7e6f7ed75fe13db2fe262e68cf9a9b7e0
      未コミットの変更 / uncommitted changes : なし
      実行日時 (UTC)                         : 2026-10-05T03:16:00+00:00
    [RERUN_BC] 水準: 実験 C のシード 0〜4(5 個)・学習 2000 ステップ、実験 B の系列長 (512, 1024, 2048, 4096, 8192)・掃引 8 回 x 反復 32 回(初回は 8 回)
    [RERUN_BC] 実験 C の評価集合のハッシュが、初回(6.11 節の出力)と一致: OK {32: 'e53637bd342d', 64: '2859d2df2044', 128: 'e28616daba9d', 256: 'a4479369abe0', 512: '101f797f0c28'}
    [RERUN_BC] 初回のコードセル(6.3・6.8・6.9 節)の本文を読み込み、SHA-256 が初回と一致することを確認した
    [RERUN_BC] 再実行の所要時間の見込み(本番の設定。初回の実測 C1 48 秒・C2 22 秒・実験 B 0.9 分から): 実験 C の較正 3.7〜4.9 分、実験 C の学習と評価 6.2 分、実験 B 3.6 分、判定・図などの余裕 3.0 分 -> 合計 最大 17.7 分(予算 120 分。セットアップを除く)。計画の自動選択は設けない


#### 6.13.5 実験 C: 学習率の採用の規則(定義と検算)

6.13.2 節で宣言した採用の規則を関数として定義し、学習の前に検算する。検算は、(1)本番の実行の較正の値(6.5 節の出力、4 桁に丸めた値)に当てはめた場合に、採用される学習率が C1 は $1.6 \times 10^{-2}$、C2 は $4 \times 10^{-3}$ になること、(2)規則の場合分け(崩壊していない場合、健全性の不成立による崩壊、基準ちょうどの値、有限でない値、最良が格子の端の場合など)を、合成した値でアサーションとして確かめることである。学習はしない。


```python
print(f"{RERUN_TAG}実験 C の学習率の採用の規則(定義と、初回の較正の値への当てはめによる検算。学習はしない)")

CHANCE_LOSS_C = math.log(TASK_C.num_symbols)  # 偶然の水準の負の対数尤度 ln 16
COLLAPSE_LOSS_C = CHANCE_LOSS_C / 2  # 崩壊とみなす較正用の集合の負の対数尤度の下限


def is_collapsed(calibration_loss: float, healthy: bool) -> bool:
    # 「崩壊」: 学習の健全性(P1 と同じ定義)が不成立、または較正用の集合の負の対数尤度が偶然の水準の 1/2 以上。
    # 有限でない値は比較が偽になるので、崩壊とみなされる
    return (not healthy) or not (calibration_loss < COLLAPSE_LOSS_C)


def best_grid_index(grid: list[float], calibration_losses: list[float]) -> int:
    # 初回の較正(6.5 節)と同じ規則: 較正用の集合の負の対数尤度が最小の格子点。同点なら小さい学習率。有限でない学習は選ばない
    return min(range(len(grid)), key=lambda i: (calibration_losses[i], grid[i]))


def adopted_grid_index(best: int, calibration_losses: list[float], healthy: list[bool]) -> tuple[int, bool | None]:
    # 採用する格子点(両条件 C1・C2 に同じ規則): 最良の 1 つ上の格子点が崩壊していれば、最良の 1 つ下の格子点。
    # 崩壊していなければ(または 1 つ上・1 つ下の格子点がなければ)、最良の格子点。戻り値は (採用する添字, 1 つ上の崩壊の真偽または None)
    if best + 1 >= len(calibration_losses):
        return best, None
    above_collapsed = is_collapsed(calibration_losses[best + 1], healthy[best + 1])
    return (best - 1 if above_collapsed and best >= 1 else best), above_collapsed


# --- 検算 1: 初回の較正の値(6.5 節の出力、4 桁に丸めた値)に当てはめる。最良の格子点は、初回の出力の「選んだ学習率」を使う。
# 健全性は出力にないので成立と仮定するが、1 つ上の格子点は負の対数尤度だけで崩壊と判定される ---
_FIRST_RUN_CALIBRATION_C = {  # 種類 -> (学習率の格子, 較正用の集合の負の対数尤度, 初回に選んだ学習率)
    "C1": ([0.016, 0.032, 0.064], [0.0, 0.0, 2.7709], 0.032),
    "C2": ([0.004, 0.008, 0.016], [0.0002, 0.0001, 2.2711], 0.008),
}
for _kind, (_grid, _losses, _first_run_choice) in _FIRST_RUN_CALIBRATION_C.items():
    _best = _grid.index(_first_run_choice)
    _adopted, _above_collapsed = adopted_grid_index(_best, _losses, [True] * len(_grid))
    print(
        f"  初回の較正の値への当てはめ {_kind}: 最良 {_grid[_best]:.3g}(内点)、1 つ上の {_grid[_best + 1]:.3g} の負の対数尤度 {_losses[_best + 1]} "
        f">= {COLLAPSE_LOSS_C:.4f}(偶然の水準 {CHANCE_LOSS_C:.4f} の 1/2): 崩壊 {_above_collapsed} -> 採用 {_grid[_adopted]:.3g}"
    )
    assert _above_collapsed and _grid[_adopted] == _grid[_best - 1]
assert COLLAPSE_LOSS_C < 2.2711 < 2.7709  # 初回の C1・C2 の最良の 1 つ上の値は、どちらも崩壊の基準を超えている

# --- 検算 2: 規則の場合分け(合成した値) ---
_nan = float("nan")
assert best_grid_index([1.0, 2.0, 4.0], [0.5, 0.1, 0.9]) == 1
assert best_grid_index([1.0, 2.0, 4.0], [0.1, 0.1, 0.9]) == 0  # 同点なら小さい学習率
assert best_grid_index([1.0, 2.0, 4.0], [math.inf, 0.1, math.inf]) == 1  # 有限でない学習は選ばれない
assert adopted_grid_index(1, [0.1, 0.0, 0.3], [True] * 3) == (1, False)  # 1 つ上が崩壊していなければ最良
assert adopted_grid_index(1, [0.1, 0.0, 2.0], [True] * 3) == (0, True)  # 負の対数尤度による崩壊
assert adopted_grid_index(1, [0.1, 0.0, 0.3], [True, True, False]) == (0, True)  # 学習の健全性の不成立による崩壊
assert adopted_grid_index(1, [0.1, 0.0, COLLAPSE_LOSS_C], [True] * 3) == (0, True)  # 基準ちょうどは崩壊(以上)
assert adopted_grid_index(1, [0.1, 0.0, COLLAPSE_LOSS_C - 1e-9], [True] * 3) == (1, False)
assert adopted_grid_index(1, [0.1, 0.0, math.inf], [True] * 3) == (0, True) and adopted_grid_index(1, [0.1, 0.0, _nan], [True] * 3) == (0, True)
assert adopted_grid_index(2, [0.3, 0.2, 0.0], [True] * 3) == (2, None)  # 最良が上端: 1 つ上がない
assert adopted_grid_index(0, [0.0, 2.0, 2.0], [True] * 3) == (0, True)  # 最良が下端: 1 つ下がないので最良のまま
print(f"{RERUN_TAG}採用の規則の検算(初回の較正の値への当てはめ 2 件、場合分け 11 件): OK")
```

    [RERUN_BC] 実験 C の学習率の採用の規則(定義と、初回の較正の値への当てはめによる検算。学習はしない)
      初回の較正の値への当てはめ C1: 最良 0.032(内点)、1 つ上の 0.064 の負の対数尤度 2.7709 >= 1.3863(偶然の水準 2.7726 の 1/2): 崩壊 True -> 採用 0.016
      初回の較正の値への当てはめ C2: 最良 0.008(内点)、1 つ上の 0.016 の負の対数尤度 2.2711 >= 1.3863(偶然の水準 2.7726 の 1/2): 崩壊 True -> 採用 0.004
    [RERUN_BC] 採用の規則の検算(初回の較正の値への当てはめ 2 件、場合分け 11 件): OK


#### 6.13.6 実験 C: 学習率の較正

C1・C2 の学習率の格子(中心 $3.2 \times 10^{-2}$ と $8 \times 10^{-3}$ の $\{1/2, 1, 2\}$ 倍)を、本番の実行と同じ手順(本番と同じ学習ステップ数、較正専用のシード 90、較正用の集合の負の対数尤度、拡張は 1 回 1 点まで)で評価し、最良の格子点を求めた後に、6.13.5 節の採用の規則を適用する。P0(C)は最良の格子点の位置で判定する。


```python
print(f"{RERUN_TAG}実験 C の学習率の較正(C1・C2、本番と同じ学習ステップ数 {RERUN_STEPS_C}、較正専用のシード {CALIBRATION_SEED_INDEX})")
_t0_calibration = time.time()
CALIBRATION_C = {}  # 種類 -> {"grid", "values", "final_loss", "healthy", "collapsed", "extended", "best", "interior", "above_collapsed", "adopted", "num_steps"}


def calibrate_c(kind: str) -> dict:
    # 格子と拡張の規則は初回(6.5 節)と同じ。違いは、最良の格子点を求めた後に採用の規則(上のセル)を適用する点だけ
    grid = list(lr_grid_for(kind))
    records = {}

    def record(learning_rate: float) -> dict:
        if learning_rate not in records:
            records[learning_rate] = run_task_c(kind, CALIBRATION_SEED_INDEX, learning_rate, RERUN_STEPS_C, calibration=True)
        return records[learning_rate]

    def evaluate(points: list[float]) -> tuple[list[float], list[bool]]:
        values = []
        for learning_rate in points:
            r = record(learning_rate)
            values.append(r["calibration_nll"] if r["finite"] and math.isfinite(r["calibration_nll"]) else math.inf)
        return values, [bool(learning_precondition(record(learning_rate))) for learning_rate in points]

    values, healthy = evaluate(grid)
    best = best_grid_index(grid, values)
    extended = None
    if best == 0:  # 最良が端: その方向に公比 2 で 1 点だけ拡張する(上限は 1 回)
        extended = "lower"
        grid = [grid[0] / LR_GRID_RATIO] + grid
    elif best == len(grid) - 1:
        extended = "upper"
        grid = grid + [grid[-1] * LR_GRID_RATIO]
    if extended is not None:
        values, healthy = evaluate(grid)
        best = best_grid_index(grid, values)
    adopted, above_collapsed = adopted_grid_index(best, values, healthy)
    return {
        "grid": grid,
        "values": values,
        "final_loss": [records[learning_rate]["final_loss"] for learning_rate in grid],
        "healthy": healthy,
        "collapsed": [is_collapsed(v, h) for v, h in zip(values, healthy, strict=True)],
        "extended": extended,
        "best": best,
        "interior": 0 < best < len(grid) - 1,  # P0(C) は最良の格子点の位置で判定する(採用する格子点の位置ではない)
        "above_collapsed": above_collapsed,
        "adopted": adopted,
        "num_steps": RERUN_STEPS_C,
    }


for _kind in ("C1", "C2"):
    CALIBRATION_C[_kind] = calibrate_c(_kind)
    _r = CALIBRATION_C[_kind]
    print(
        f"較正 {_kind}: 学習率 {[float(f'{x:.3g}') for x in _r['grid']]} -> 較正用の集合の負の対数尤度 {rounded(_r['values'])}、"
        f"最後の区間の訓練損失 {rounded(_r['final_loss'], 3)}、健全性 {_r['healthy']}、崩壊 {_r['collapsed']}、拡張 {_r['extended'] or 'なし'}、"
        f"最良 {_r['grid'][_r['best']]:.3g}({'内点' if _r['interior'] else '端'})、最良の 1 つ上の格子点の崩壊 {_r['above_collapsed']} "
        f"-> 採用する学習率 {_r['grid'][_r['adopted']]:.3g}"
    )
RERUN_LEARNING_RATE = {kind: CALIBRATION_C[kind]["grid"][CALIBRATION_C[kind]["adopted"]] for kind in ("C1", "C2")}
# P0(較正): 対比量に入る両方の条件で、最良の格子点が格子の内点であること(拡張後も最良が端なら不成立)。初回と同じ定義
precondition_status["P0(C)"] = CALIBRATION_C["C1"]["interior"] and CALIBRATION_C["C2"]["interior"]
for _kind in ("C1", "C2"):
    assert CALIBRATION_C[_kind]["num_steps"] == RERUN_STEPS_C  # 較正は本番と同じ学習ステップ数で行った
    assert CALIBRATION_C[_kind]["adopted"] in (CALIBRATION_C[_kind]["best"], CALIBRATION_C[_kind]["best"] - 1)
print(f"採用する学習率: {json.dumps({k: float(f'{v:.3g}') for k, v in RERUN_LEARNING_RATE.items()})}")
print(f"{RERUN_TAG}P0(較正): 実験 C: {precondition_status['P0(C)']}")
CALIBRATION_SECONDS = time.time() - _t0_calibration
print(
    f"較正 {CALIBRATION_SECONDS / 60:.1f} 分(見積もり {ESTIMATE_SECONDS['calibration_best'] / 60:.1f}〜"
    f"{ESTIMATE_SECONDS['calibration_worst'] / 60:.1f} 分、本番の設定での値)"
)
```

    [RERUN_BC] 実験 C の学習率の較正(C1・C2、本番と同じ学習ステップ数 2000、較正専用のシード 90)
    較正 C1: 学習率 [0.016, 0.032, 0.064] -> 較正用の集合の負の対数尤度 [0.0, 0.0, 2.7709]、最後の区間の訓練損失 [0.0, 0.0, 2.772]、健全性 [True, True, False]、崩壊 [False, False, True]、拡張 なし、最良 0.032(内点)、最良の 1 つ上の格子点の崩壊 True -> 採用する学習率 0.016
    較正 C2: 学習率 [0.004, 0.008, 0.016] -> 較正用の集合の負の対数尤度 [0.0002, 0.0001, 2.2711]、最後の区間の訓練損失 [0.0, 0.0, 2.31]、健全性 [True, True, False]、崩壊 [False, False, True]、拡張 なし、最良 0.008(内点)、最良の 1 つ上の格子点の崩壊 True -> 採用する学習率 0.004
    採用する学習率: {"C1": 0.016, "C2": 0.004}
    [RERUN_BC] P0(較正): 実験 C: True
    較正 3.5 分(見積もり 3.7〜4.9 分、本番の設定での値)


#### 6.13.7 実験 C: 学習と評価

採用した学習率で、C1・C2 を、シード 0〜4、2000 ステップで学習し、本番の実行と同じ評価集合で評価する。学習の関数(`run_task_c`)は本番の実行と同じである。


```python
print(
    f"{RERUN_TAG}実験 C の学習と評価(シード 0〜{RERUN_SEEDS_C - 1}、{RERUN_STEPS_C} ステップ、採用した学習率 "
    f"{json.dumps({k: float(f'{v:.3g}') for k, v in RERUN_LEARNING_RATE.items()})})"
)
_t0_main = time.time()
RUNS = {}  # (種類, シード) -> 記録
for _s in range(RERUN_SEEDS_C):
    for _kind in ("C1", "C2"):
        RUNS[(_kind, _s)] = run_task_c(_kind, _s, RERUN_LEARNING_RATE[_kind], RERUN_STEPS_C)
        _r = RUNS[(_kind, _s)]
        print(
            f"{_kind} シード {_s}: 正解率 L_train = {C_TRAIN_LENGTH}: {_r['by_length'][C_TRAIN_LENGTH]['accuracy']:.3f}、"
            f"L_test = {C_TEST_LENGTH}: {_r['by_length'][C_TEST_LENGTH]['accuracy']:.3f}、"
            f"訓練損失 {_r['initial_loss']:.3f} -> {_r['final_loss']:.3f}、clipping の発動 {_r['clip_rate']:.2f}、{_r['seconds']:.0f} 秒"
        )
MAIN_SECONDS = time.time() - _t0_main

# --- 不変条件: 実際に学習した種類・シード・ステップ数・評価のステップ・学習率が、宣言した水準と採用した学習率に一致する ---
assert set(RUNS) == {(kind, s) for kind in ("C1", "C2") for s in range(RERUN_SEEDS_C)}
for (_kind, _seed), _record in RUNS.items():
    assert _record["num_steps"] == RERUN_STEPS_C and len(_record["train_loss"]) == RERUN_STEPS_C
    assert _record["eval_step"] == list(eval_steps_for(RERUN_STEPS_C))
    assert _record["learning_rate"] == RERUN_LEARNING_RATE[_kind] and _record["kind"] == _kind and _record["seed"] == _seed
assert min(CALIBRATION_SEED_INDEX, TIMING_SEED_INDEX) >= RERUN_SEEDS_C  # 較正・計測のシードは実験のシードと共有しない
print(
    f"{RERUN_TAG}実験 C の学習と評価 {MAIN_SECONDS / 60:.1f} 分(見積もり {ESTIMATE_SECONDS['main'] / 60:.1f} 分、本番の設定での値)。"
    f"学習した全記録({len(RUNS)} 回)が、宣言した水準と採用した学習率に一致: OK"
)
```

    [RERUN_BC] 実験 C の学習と評価(シード 0〜4、2000 ステップ、採用した学習率 {"C1": 0.016, "C2": 0.004})
    C1 シード 0: 正解率 L_train = 32: 1.000、L_test = 128: 1.000、訓練損失 2.542 -> 0.000、clipping の発動 0.05、47 秒
    C2 シード 0: 正解率 L_train = 32: 1.000、L_test = 128: 0.294、訓練損失 2.565 -> 0.000、clipping の発動 0.09、21 秒
    C1 シード 1: 正解率 L_train = 32: 1.000、L_test = 128: 1.000、訓練損失 2.306 -> 0.000、clipping の発動 0.07、47 秒
    C2 シード 1: 正解率 L_train = 32: 1.000、L_test = 128: 0.318、訓練損失 2.753 -> 0.000、clipping の発動 0.14、22 秒
    C1 シード 2: 正解率 L_train = 32: 1.000、L_test = 128: 1.000、訓練損失 2.289 -> 0.000、clipping の発動 0.05、47 秒
    C2 シード 2: 正解率 L_train = 32: 1.000、L_test = 128: 0.324、訓練損失 2.682 -> 0.000、clipping の発動 0.13、21 秒
    C1 シード 3: 正解率 L_train = 32: 1.000、L_test = 128: 1.000、訓練損失 2.259 -> 0.000、clipping の発動 0.04、47 秒
    C2 シード 3: 正解率 L_train = 32: 1.000、L_test = 128: 0.363、訓練損失 2.744 -> 0.000、clipping の発動 0.12、21 秒
    C1 シード 4: 正解率 L_train = 32: 1.000、L_test = 128: 1.000、訓練損失 2.548 -> 0.000、clipping の発動 0.07、47 秒
    C2 シード 4: 正解率 L_train = 32: 1.000、L_test = 128: 0.352、訓練損失 2.712 -> 0.000、clipping の発動 0.13、21 秒
    [RERUN_BC] 実験 C の学習と評価 6.0 分(見積もり 6.2 分、本番の設定での値)。学習した全記録(10 回)が、宣言した水準と採用した学習率に一致: OK


#### 6.13.8 実験 C: 前提条件と判定

本番の実行の 6.9 節のコードセルの本文を、そのまま実行する。対比量 $\Delta_C$、標準偏差 $\sigma_C$、閾値、前提条件 P0(C)・P1(C)・P2(C)、診断量、図は、本番の実行と同じである。**前提条件が成立しない場合は「前提不成立」として記録し、判定関数の参考値を結論として扱わない。**


```python
print(f"{RERUN_TAG}実験 C の前提条件と判定(初回の 6.9 節のコードセルの本文を、そのまま実行する。対比量・標準偏差・閾値・前提条件・診断量・図は初回と同じ)")
# 初回の 6.9 節の本文が使う名前に、再実行の水準を与える(本文は変更しない)
SEEDS_C = RERUN_SEEDS_C
NUM_STEPS = {"C": RERUN_STEPS_C}
exec(compile(FIRST_RUN_SOURCE_C, "初回の 6.9 節のコードセル", "exec"), globals())
```

    [RERUN_BC] 実験 C の前提条件と判定(初回の 6.9 節のコードセルの本文を、そのまま実行する。対比量・標準偏差・閾値・前提条件・診断量・図は初回と同じ)
    実験 C(シード [0, 1, 2, 3, 4]、学習 2000 ステップ、L_train = 32、L_test = 128)
      C1: a(L_train) [1.0, 1.0, 1.0, 1.0, 1.0]、a(L_test) [1.0, 1.0, 1.0, 1.0, 1.0]、d = a(L_train) - a(L_test) [0.0, 0.0, 0.0, 0.0, 0.0]
      C2: a(L_train) [1.0, 1.0, 1.0, 1.0, 1.0]、a(L_test) [0.294, 0.318, 0.324, 0.363, 0.352]、d = a(L_train) - a(L_test) [0.706, 0.682, 0.676, 0.637, 0.648]
      d_s(条件 2 - 条件 1) = [0.706, 0.682, 0.676, 0.637, 0.648]
      Delta_C = +0.6698、sigma_C = 0.0173(シード間 0.0123・ブートストラップ 0.0122、反復 10,000)、閾値 2 sigma_C = 0.0347
      前提条件: P0 True、P1 True(不成立の学習 なし)、P2(習得) True(a(L_train) のシード平均 C1 1.0000・C2 1.0000 >= 0.9)
      判定関数の結果 支持 -> 最終判定: 支持
      診断量: 偶然の水準 0.0625。系列長ごとの正解率(シード平均 ± シード間の標準偏差):
        C1: 1x(32): 1.000 ± 0.000、2x(64): 1.000 ± 0.000、4x(128): 1.000 ± 0.000、8x(256): 1.000 ± 0.000、16x(512): 1.000 ± 0.000
        C2: 1x(32): 1.000 ± 0.000、2x(64): 0.670 ± 0.027、4x(128): 0.330 ± 0.028、8x(256): 0.172 ± 0.024、16x(512): 0.096 ± 0.012
      診断量: L_test = 128 の正解率の各シードの値(二峰かどうかが分かる形、昇順): C1 [1.0, 1.0, 1.0, 1.0, 1.0]、C2 [0.294, 0.318, 0.324, 0.352, 0.363]
      診断量: L_train の各シードの値(昇順): C1 [1.0, 1.0, 1.0, 1.0, 1.0]、C2 [1.0, 1.0, 1.0, 1.0, 1.0]
      診断量: 非埋め込みパラメータ数 C1 65,472、C2 65,344



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/023_state_space_model_mamba/output_56_1.png)
    


#### 6.13.9 実験 B: 計測と判定

本番の実行の 6.3 節の層・水準・計測の関数の定義と、6.8 節の計測の直前のウォームアップを、そのまま実行する。計測は、本番の実行と同じ手順で、掃引 8 回 x 反復 32 回(変更点 1)を、ガベージコレクションを止めて(変更点 2)行う。計測の後の前提条件 P-B1・P-B2、べき指数、判定、診断量、図は、本番の実行の 6.8 節の本文をそのまま実行する。最後に、6.13.2 節で宣言した追加の診断量(時間が中央値の 2 倍を超えた反復の数と、最大値と中央値の比)を印字する。**前提条件が成立しない場合は「前提不成立」として記録し、判定関数の参考値を結論として扱わない。**


```python
import gc

print(
    f"{RERUN_TAG}実験 B の再実行(系列長 {CFG['B_LEVELS']}、掃引 {CFG['B_SWEEPS']} 回 x 反復 {B_REPEATS_RERUN} 回。初回は反復 {CFG['B_REPEATS']} 回)。"
    "層・水準・計測の関数、ウォームアップ、前提条件・判定・診断量・図は、初回のコードセルの本文をそのまま実行する"
)
# 初回の 6.3 節の、層・水準・計測の関数の定義(ウォームアップの前まで)。同じ乱数シード(B_SEED)で同じ重みと入力を作る
exec(compile(FIRST_RUN_B_DEFINITIONS, "初回の 6.3 節のコードセル(定義の部分)", "exec"), globals())
B_REPEATS = B_REPEATS_RERUN  # 変更点 1: 掃引あたりの反復回数 8 -> 32(平均の標準誤差は反復回数の平方根に反比例する)
B_TOTAL_SECONDS = ESTIMATE_SECONDS["b"]  # 初回の 6.8 節の本文の最後の印字が使う見積もり

# 初回の 6.8 節の、計測の直前のウォームアップ(キャッシュを解放した直後に、全水準・両条件を B_WARMUP_REPEATS 回ずつ)
exec(compile(FIRST_RUN_B_WARMUP, "初回の 6.8 節のコードセル(ウォームアップの部分)", "exec"), globals())

# 計測: 初回と同じ手順(全水準・両条件を 1 反復ずつ、反復と水準ごとに実行の順序を入れ替える。掃引の数は 8 のまま)。
# 変更点 2: 計測の区間ではガベージコレクション(Garbage Collection)を止める。各掃引の前に gc.collect() を呼び、計測の後に必ず元に戻す
B_RAW = {c: {length: [[] for _ in range(B_SWEEPS)] for length in B_LEVELS} for c in ("mamba", "attention")}
_gc_was_enabled = gc.isenabled()
gc.disable()
try:
    for _sweep in range(B_SWEEPS):
        gc.collect()
        for _repeat in range(B_REPEATS):
            _times = run_b_pass(_sweep * B_REPEATS + _repeat)
            for _c, _by_length in _times.items():
                for _length, _seconds in _by_length.items():
                    B_RAW[_c][_length][_sweep].append(_seconds)
finally:
    if _gc_was_enabled:
        gc.enable()
B_MEASURE_SECONDS = time.time() - _t0_b

# --- 不変条件: 変えないと宣言したものが変わっていない。ガベージコレクションが元の状態に戻っている ---
assert gc.isenabled() == _gc_was_enabled
assert CFG["B_SWEEPS"] == B_SWEEPS and tuple(CFG["B_LEVELS"]) == tuple(B_LEVELS) and B_WARMUP_REPEATS == 3 and B_D_MODEL == 128
assert SMOKE_TEST or (B_SWEEPS, tuple(B_LEVELS)) == (8, (512, 1024, 2048, 4096, 8192))  # 掃引の回数・系列長の水準は初回のまま
assert (P0A_MAX_DRIFT_SIGMA, P0B_MAX_RELATIVE_STDERR) == (2.0, 0.05)  # P-B1・P-B2 の閾値(6.1 節の宣言)
assert all(len(B_RAW[c][L]) == B_SWEEPS and all(len(sweep) == B_REPEATS for sweep in B_RAW[c][L]) for c in B_RAW for L in B_LEVELS)
print(f"{RERUN_TAG}計測を終えた: 掃引 {B_SWEEPS} 回 x 反復 {B_REPEATS} 回(各条件・各水準で {B_SWEEPS * B_REPEATS} 反復)、ガベージコレクションは元の状態({'有効' if _gc_was_enabled else '無効'})に戻した")

# 初回の 6.8 節の、計測の後の前提条件 P-B1・P-B2、べき指数、判定、診断量、図
exec(compile(FIRST_RUN_B_ANALYSIS, "初回の 6.8 節のコードセル(計測の後の部分)", "exec"), globals())

# --- 追加する診断量(判定には使わない): 水準・条件ごとの、時間が中央値の 2 倍を超えた反復の数 / 全反復の数と、最大値 / 中央値 ---
print(f"{RERUN_TAG}診断量(判定には使わない): 水準ごとの「時間が中央値の 2 倍を超えた反復の数 / 全反復の数・最大値 / 中央値」:")
for _c in B_RAW:
    _parts = []
    for _length in B_LEVELS:
        _series = series_in_time_order(_c, _length)
        _median = float(np.median(_series))
        _parts.append(f"{_length}: {int((_series > 2 * _median).sum())}/{len(_series)}・{float(_series.max()) / _median:.2f}")
    print(f"    {_c}: " + "、".join(_parts))
```

    [RERUN_BC] 実験 B の再実行(系列長 (512, 1024, 2048, 4096, 8192)、掃引 8 回 x 反復 32 回。初回は反復 8 回)。層・水準・計測の関数、ウォームアップ、前提条件・判定・診断量・図は、初回のコードセルの本文をそのまま実行する
    実験 B のウォームアップ(キャッシュの解放の直後、全水準・両条件を 3 回ずつ): 2.3 秒
    [RERUN_BC] 計測を終えた: 掃引 8 回 x 反復 32 回(各条件・各水準で 256 反復)、ガベージコレクションは元の状態(有効)に戻した
    実験 B(系列長 (512, 1024, 2048, 4096, 8192)、掃引 8 回 x 反復 32 回、デバイス cuda)
      掃引ごとのべき指数 b_1(Mamba の層): [0.989, 0.989, 0.993, 0.99, 0.989, 0.992, 0.987, 0.991]、b_2(注意機構の層): [1.597, 1.583, 1.589, 1.591, 1.595, 1.596, 1.586, 1.59]
      掃引ごとの Delta_B = b_2 - b_1: [0.608, 0.594, 0.596, 0.601, 0.606, 0.604, 0.599, 0.6]
      Delta_B = +0.6009、sigma_B = 0.0017(掃引ごとの Delta_B の標準偏差 0.0047 / sqrt(8))、閾値 2 sigma_B = 0.0033
      前提条件: P-B1(ドリフトなし) True(不成立 なし)、P-B2(平均の標準誤差 <= 平均の 5%) True(不成立 なし、比の最大 0.0116)
      判定関数の結果 支持 -> 最終判定: 支持
      診断量: b_1 = 0.990(理論値 1 との差 -0.010)、b_2 = 1.591(理論値 2 との差 -0.409)。掃引ごとの回帰の標準誤差の平均 b_1 0.007・b_2 0.147
      診断量: 水準ごとの時間の中央値(全掃引・全反復、ミリ秒):
        mamba: 512: 22.21、1024: 42.98、2048: 85.40、4096: 171.04、8192: 344.82
        attention: 512: 0.73、1024: 1.19、2048: 3.63、4096: 13.18、8192: 54.06
      診断量: 水準ごとの P-B1 の先頭 3 反復 / 末尾 3 反復の平均(ミリ秒)と P-B2 の標準誤差 / 平均:
        mamba: 512: 28.10/23.87・0.0112、1024: 54.63/43.00・0.0116、2048: 114.28/89.71・0.0109、4096: 230.46/171.56・0.0103、8192: 440.39/340.78・0.0095
        attention: 512: 0.91/0.71・0.0102、1024: 1.29/1.20・0.0031、2048: 3.69/3.61・0.0013、4096: 13.28/13.14・0.0004、8192: 53.46/53.61・0.0004
      診断量: 最大メモリの増分(MiB) mamba: [34.6, 70.0, 138.3, 276.6, 553.9](べき指数 0.998)、attention: [9.0, 34.5, 135.0, 534.0, 2124.0](べき指数 1.972)
      診断量(閉形式の下限): 一時テンソルのバイト数のべき指数 Mamba 1.000、注意機構 2.000
      診断量: 推論時の 1 トークンあたり — Mamba の 1 ステップ更新 0.737 ms、状態 19,456 バイト(文脈の長さによらず一定)。注意機構(KV キャッシュつき、1 層): 文脈 512: 0.493 ms・KV キャッシュ 545,792 バイト、文脈 2048: 0.454 ms・KV キャッシュ 2,118,656 バイト、文脈 8192: 0.480 ms・KV キャッシュ 8,410,112 バイト(逐次ループの実装の絶対時間は公式実装を代表しない)



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/023_state_space_model_mamba/output_58_1.png)
    


    実験 B の計測 3.4 分(見積もり 3.6 分)
    [RERUN_BC] 診断量(判定には使わない): 水準ごとの「時間が中央値の 2 倍を超えた反復の数 / 全反復の数・最大値 / 中央値」:
        mamba: 512: 0/256・1.84、1024: 1/256・2.03、2048: 0/256・1.87、4096: 0/256・1.73、8192: 0/256・1.61
        attention: 512: 0/256・1.96、1024: 0/256・1.19、2048: 0/256・1.08、4096: 0/256・1.03、8192: 0/256・1.01


#### 6.13.10 判定のまとめ

実験 B・C について、出力に「初回」と印字される最終判定と、再実行の最終判定を並べて印字する。実験 C の「初回」は本番の実行を指し、実験 B の「初回」は 6.13 節の冒頭に記した別の実行を指す。再実行の結果は、再び前提不成立であっても、そのまま記録する。


```python
print(f"{RERUN_TAG}実験 B・C の再実行の判定のまとめ(初回の最終判定と再実行の最終判定)")
# 初回の値: 初回のセル出力から書き写した定数(6.12 節の出力。実験 B は 6.8 節、実験 C は 6.9 節の出力と同じ)
FIRST_RUN_RESULTS = {
    "B": {"verdict": "前提不成立", "preconditions": {"P-B1": True, "P-B2": False}},
    "C": {"verdict": "前提不成立", "preconditions": {"P0(C)": True, "P1(C)": False, "P2(C)": False}},
}
RERUN_RESULTS = {
    "B": {"verdict": B_VERDICT, "computed": B_COMPUTED, "delta": DELTA_B, "threshold": SIGMA_MULTIPLIER * SIGMA_B, "preconditions": {k: precondition_status[k] for k in B_PRECONDITIONS}},
    "C": {"verdict": C_VERDICT, "computed": C_COMPUTED, "delta": DELTA_C, "threshold": SIGMA_MULTIPLIER * SIGMA_C["sigma"], "preconditions": {k: precondition_status[k] for k in C_PRECONDITIONS}},
}
assert set(FIRST_RUN_RESULTS["B"]["preconditions"]) == set(B_PRECONDITIONS) and set(FIRST_RUN_RESULTS["C"]["preconditions"]) == set(C_PRECONDITIONS)
for _experiment in ("B", "C"):
    _first, _again = FIRST_RUN_RESULTS[_experiment], RERUN_RESULTS[_experiment]
    print(
        f"  実験 {_experiment}: 初回の最終判定 {_first['verdict']}(前提条件 {json.dumps(_first['preconditions'])}) -> "
        f"再実行の最終判定 {_again['verdict']}(判定関数の結果 {_again['computed']}、Delta = {_again['delta']:+.4f}、閾値 {_again['threshold']:.4f}、"
        f"前提条件 {json.dumps(_again['preconditions'])})"
    )
print("  実験 A・D: 再実行していない(初回の結果が記録)。判定基準・前提条件の定義と閾値は初回のまま。再実行は 1 回限りで、結果はそのまま記録する")
_elapsed = time.time() - RERUN_START_TIME
print(f"{RERUN_TAG}再実行の所要時間 {_elapsed / 60:.1f} 分(見積もり 最大 {ESTIMATE_TOTAL_SECONDS / 60:.1f} 分、本番の設定での値。セットアップを除く)")
```

    [RERUN_BC] 実験 B・C の再実行の判定のまとめ(初回の最終判定と再実行の最終判定)
      実験 B: 初回の最終判定 前提不成立(前提条件 {"P-B1": true, "P-B2": false}) -> 再実行の最終判定 支持(判定関数の結果 支持、Delta = +0.6009、閾値 0.0033、前提条件 {"P-B1": true, "P-B2": true})
      実験 C: 初回の最終判定 前提不成立(前提条件 {"P0(C)": true, "P1(C)": false, "P2(C)": false}) -> 再実行の最終判定 支持(判定関数の結果 支持、Delta = +0.6698、閾値 0.0347、前提条件 {"P0(C)": true, "P1(C)": true, "P2(C)": true})
      実験 A・D: 再実行していない(初回の結果が記録)。判定基準・前提条件の定義と閾値は初回のまま。再実行は 1 回限りで、結果はそのまま記録する
    [RERUN_BC] 再実行の所要時間 13.1 分(見積もり 最大 17.7 分、本番の設定での値。セットアップを除く)




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/06_architectures/023_state_space_model_mamba.ipynb
