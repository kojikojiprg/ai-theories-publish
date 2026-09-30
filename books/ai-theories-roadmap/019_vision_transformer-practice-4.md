---
title: "ViT と画像パッチ埋め込み / Vision Transformer and Patch Embedding(実装・実験編 4/4)"
---

この記事は後編(実装・実験編 4/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/019_vision_transformer-practice-3)。

### 6.10 実験 C: 帰納バイアスとデータ量


```python
C_LOG2_F = np.array([math.log2(FRACTION_ACTUAL[f]) for f in DATA_FRACTIONS])
C_KEYS = {
    "ViT": lambda f, s: key_vit(STANDARD_PATCH_SIZE, "learned", f, s),
    "CNN": lambda f, s: key_cnn(f, s),
}
C_ACCURACY = {m: np.array([[test_accuracy(k(f, s)) for f in DATA_FRACTIONS] for s in SEEDS_C]) for m, k in C_KEYS.items()}
C_SLOPES = {m: ols_slope(C_LOG2_F, a) for m, a in C_ACCURACY.items()}


def slope_contrast(slopes: dict) -> dict:
    n = len(SEEDS_C)
    value = slopes["ViT"].mean() - slopes["CNN"].mean()
    sigma = math.sqrt(slopes["ViT"].var(ddof=1) / n + slopes["CNN"].var(ddof=1) / n)
    return {"value": float(value), "sigma": float(sigma)}


CONTRAST_C = slope_contrast(C_SLOPES)
verdict_C = judge(CONTRAST_C["value"], CONTRAST_C["sigma"])
precondition_status["P1(C)"] = bool(all((a.mean(axis=0) >= ACCURACY_MIN).all() for a in C_ACCURACY.values()))
_logit_slopes = {m: ols_slope(C_LOG2_F, logit(a)) for m, a in C_ACCURACY.items()}
_logit_contrast = slope_contrast(_logit_slopes)

print(f"{_tag}log2 f(実際の比)= {np.round(C_LOG2_F, 4).tolist()}")
for _m, _k in C_KEYS.items():
    for _j, _f in enumerate(DATA_FRACTIONS):
        _train_acc = np.array([RECORDS[_k(_f, s)]["common_subset_train_accuracy"] for s in SEEDS_C])
        _ce = np.mean([RECORDS[_k(_f, s)]["test"]["cross_entropy"] for s in SEEDS_C])
        print(
            f"{_tag}  {_m} f = {_f:.4f}: テスト正解率 {np.round(C_ACCURACY[_m][:, _j], 4).tolist()} 平均 {C_ACCURACY[_m][:, _j].mean():.4f}、"
            f"交差エントロピー平均 {_ce:.4f}、S_1/16 での正解率 平均 {_train_acc.mean():.4f}"
            f"(テストとの差 {(_train_acc - C_ACCURACY[_m][:, _j]).mean():+.4f})"
        )
    print(f"{_tag}  {_m} のシードごとの傾き gamma: {np.round(C_SLOPES[_m], 5).tolist()} 平均 {C_SLOPES[_m].mean():+.5f}")
print(
    f"{_tag}対比量 Delta_C = gamma_ViT - gamma_CNN = {CONTRAST_C['value']:+.5f}、"
    f"sigma_C = sqrt(s_ViT^2/{len(SEEDS_C)} + s_CNN^2/{len(SEEDS_C)}) = {CONTRAST_C['sigma']:.5f}、"
    f"閾値 {2 * CONTRAST_C['sigma']:.5f} -> 判定関数の結果: {verdict_C}"
)
print(
    f"{_tag}診断量(判定なし、天井効果の確認): ロジットの傾き ViT {_logit_slopes['ViT'].mean():+.4f}・CNN {_logit_slopes['CNN'].mean():+.4f}、"
    f"差 {_logit_contrast['value']:+.4f}(同じ式の sigma {_logit_contrast['sigma']:.4f}、比 {_logit_contrast['value'] / max(_logit_contrast['sigma'], 1e-12):+.2f})"
)
print(
    f"{_tag}前提条件: P0(ViT) = {precondition_status['P0(ViT)']}、P0(CNN) = {precondition_status['P0(CNN)']}、"
    f"P1(C) = {precondition_status['P1(C)']}"
)
print(
    f"交絡(5.4 節): パラメータ数 ViT {PARAMETER_TABLE['A: P=4, learned'][0]:,}・CNN {CNN_PARAMETERS:,}、"
    f"順伝播 ViT {VIT_FORWARD_FLOPS[4] / 1e6:,.0f} MFLOP・CNN {CNN_FORWARD_FLOPS / 1e6:,.0f} MFLOP / 枚"
)

_fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
for _i, _m in enumerate(C_KEYS):
    for _s_i in range(len(SEEDS_C)):
        _axes[0].plot(C_LOG2_F, C_ACCURACY[_m][_s_i], color=f"C{_i}", alpha=0.3)
    _axes[0].plot(C_LOG2_F, C_ACCURACY[_m].mean(axis=0), marker="o", color=f"C{_i}", lw=2, label=_m)
    _axes[1].plot(C_LOG2_F, logit(C_ACCURACY[_m]).mean(axis=0), marker="o", color=f"C{_i}", lw=2, label=_m)
_axes[0].set_ylabel("test accuracy")
_axes[1].set_ylabel("logit(test accuracy) (diagnostic)")
for _ax in _axes:
    _ax.set_xlabel("log2 (fraction of training data)")
    _ax.legend()
_axes[0].set_title(f"{_plot_tag}Experiment C: data size (thin = seeds)")
_axes[1].set_title("logit scale (ceiling-effect diagnostic)")
plt.tight_layout()
plt.show()
```

    log2 f(実際の比)= [-4.0013, -2.0, 0.0]
      ViT f = 0.0625: テスト正解率 [0.4771, 0.4744, 0.4767] 平均 0.4761、交差エントロピー平均 1.9399、S_1/16 での正解率 平均 0.9081(テストとの差 +0.4320)
      ViT f = 0.2500: テスト正解率 [0.5735, 0.5717, 0.5757] 平均 0.5736、交差エントロピー平均 1.2011、S_1/16 での正解率 平均 0.6204(テストとの差 +0.0468)
      ViT f = 1.0000: テスト正解率 [0.5832, 0.5889, 0.5693] 平均 0.5805、交差エントロピー平均 1.1670、S_1/16 での正解率 平均 0.5877(テストとの差 +0.0072)
      ViT のシードごとの傾き gamma: [0.02652, 0.02862, 0.02315] 平均 +0.02609
      CNN f = 0.0625: テスト正解率 [0.6998, 0.7041, 0.6932] 平均 0.6990、交差エントロピー平均 1.7648、S_1/16 での正解率 平均 0.9995(テストとの差 +0.3005)
      CNN f = 0.2500: テスト正解率 [0.8104, 0.8035, 0.8048] 平均 0.8062、交差エントロピー平均 0.6354、S_1/16 での正解率 平均 0.9116(テストとの差 +0.1054)
      CNN f = 1.0000: テスト正解率 [0.8253, 0.8213, 0.8287] 平均 0.8251、交差エントロピー平均 0.5124、S_1/16 での正解率 平均 0.8463(テストとの差 +0.0212)
      CNN のシードごとの傾き gamma: [0.03137, 0.02929, 0.03387] 平均 +0.03151
    対比量 Delta_C = gamma_ViT - gamma_CNN = -0.00541、sigma_C = sqrt(s_ViT^2/3 + s_CNN^2/3) = 0.00207、閾値 0.00414 -> 判定関数の結果: 反証
    診断量(判定なし、天井効果の確認): ロジットの傾き ViT +0.1051・CNN +0.1771、差 -0.0720(同じ式の sigma 0.0099、比 -7.28)
    前提条件: P0(ViT) = True、P0(CNN) = False、P1(C) = True
    交絡(5.4 節): パラメータ数 ViT 2,688,970・CNN 853,018、順伝播 ViT 366 MFLOP・CNN 251 MFLOP / 枚



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/019_vision_transformer/output_39_1.png)
    


### 6.11 不変条件のアサーションと`SMOKE_TEST`の配線

- すべての学習(較正を含む)で、ステップ数が $T$、学習率のスケジュールの形(最大値で割った列)が同一、データ拡張のパディングが同一。
- 本番の学習はすべて同じテスト集合 10,000 枚で評価した(テスト集合のテンソルが読み込み直後と一致)。較正の学習はテスト集合を
  評価していない。
- 同じシード・同じ部分集合の学習は、モデル・条件によらず、同じデータの順序・同じデータ拡張を使った(データの乱数の列のハッシュが
  一致)。
- 実験 A・B の ViT の非埋め込みパラメータ数が全学習で一致。
- 標準条件 S の記録を、実験 A の条件 1・実験 B の $P = 4$・実験 C の ViT の $f = 1$ で共有した(同じ鍵)。
- **`SMOKE_TEST`の配線**: 5.2 節・6.4 節で決まった実効値(水準・計画・$T$・段階・シード数・パッチサイズの水準・データ量の水準)が
  計画の表どおりであり、実際に使われた値(記録の鍵・学習の記録の長さ)と一致する。


```python
_all_records = list(RECORDS.values()) + [r for c in CALIBRATION.values() for r in c["records"].values()]
_shape = None
for _r in _all_records:
    _h = _r["history"]
    assert _r["train_steps"] == TRAIN_STEPS and len(_h["loss"]) == TRAIN_STEPS and len(_h["learning_rate"]) == TRAIN_STEPS
    assert set(_h["evaluations"]) == set(INTERMEDIATE_EVAL_STEPS)
    assert _r["crop_padding"] == CROP_PADDING
    _normalized = np.array(_h["learning_rate"]) / _r["learning_rate"]
    _shape = _normalized if _shape is None else _shape
    assert np.allclose(_normalized, _shape, rtol=1e-12, atol=0), "学習率のスケジュールの形が異なる"
for _r in RECORDS.values():
    assert _r["purpose"] == "main" and _r["test"]["num_examples"] == len(TEST_LABELS)
for _c in CALIBRATION.values():
    assert all(r["purpose"] == "calibration" and "test" not in r for r in _c["records"].values())
assert sha256_of_tensor(TEST_IMAGES) == TEST_IMAGES_SHA256
_streams: dict = {}
for _r in RECORDS.values():
    _streams.setdefault((_r["key"][4], _r["key"][3]), set()).add(_r["history"]["data_stream_hash"])
assert all(len(v) == 1 for v in _streams.values()), "同じシード・同じ部分集合でデータの順序・拡張が異なる"
_vit_non_embedding = {r["non_embedding_parameters"] for r in RECORDS.values() if r["key"][0] == "vit"}
assert _vit_non_embedding == {NON_EMBEDDING_PARAMETERS}
for _s in SEEDS_S:  # S は 1 つの記録を共有する
    assert key_vit(STANDARD_PATCH_SIZE, "learned", 1.0, _s) in RECORDS
assert set(SEEDS_S) >= set(SEEDS_A) | set(SEEDS_B) | set(SEEDS_C)
print(
    f"不変条件: 全 {len(_all_records)} 学習(本番 {len(RECORDS)}・較正 {len(_all_records) - len(RECORDS)})の T = {TRAIN_STEPS}・"
    f"学習率のスケジュールの形・パディングが同一、本番はすべて同じテスト集合 {len(TEST_LABELS):,} 枚で評価、較正はテスト集合を評価していない、"
    f"同じシード・同じ部分集合のデータの順序と拡張が共通({len(_streams)} 組)、ViT の非埋め込みパラメータ数が一致、S を共有: OK"
)

_table = PLANS[CURRENT_LEVEL_NAME][SELECTED_PLAN]
assert _table["plan"] == SELECTED_PLAN and _table["T"] == TRAIN_STEPS and _table["stage"] == SELECTED_STAGE
assert TRAIN_STEPS == T_CANDIDATES[CURRENT_LEVEL_NAME][SELECTED_PLAN // 4] and SELECTED_STAGE == SELECTED_PLAN % 4
_effective = {
    "level": CURRENT_LEVEL_NAME,
    "plan": SELECTED_PLAN,
    "stage": SELECTED_STAGE,
    "train_steps": TRAIN_STEPS,
    "seeds": {k[-1]: STAGES[CURRENT_LEVEL_NAME][_table["stage"]][k] for k in SEED_KEYS},
    "b_patch_sizes": list(STAGES[CURRENT_LEVEL_NAME][_table["stage"]]["B_PATCH_SIZES"]),
    "fractions": list(DATA_FRACTIONS),
    "runs": len(plan_runs(STAGES[CURRENT_LEVEL_NAME][_table["stage"]])),
}
_history_lengths = {len(r["history"]["loss"]) for r in _all_records}
assert len(_history_lengths) == 1
_used = {
    "level": CURRENT_LEVEL_NAME,
    "plan": SELECTED_PLAN,
    "stage": SELECTED_STAGE,
    "train_steps": _history_lengths.pop(),
    "seeds": {
        "A": len({k[4] for k in RECORDS if k[0] == "vit" and k[2] == "none"}),
        "B": len({k[4] for k in RECORDS if k[0] == "vit" and k[1] == max(B_PATCH_SIZES)}),
        "C": len({k[4] for k in RECORDS if k[0] == "cnn"}),
    },
    "b_patch_sizes": sorted({k[1] for k in RECORDS if k[0] == "vit"}, reverse=True),
    "fractions": sorted({k[3] for k in RECORDS}),
    "runs": len(RECORDS),
}
assert _used == _effective, (_used, _effective)
assert (CURRENT_LEVEL_NAME == "smoke") == SMOKE_TEST
assert A_ACCURACY.shape == (3, len(SEEDS_A)) and B_ACCURACY.shape == (len(SEEDS_B), len(B_PATCH_SIZES))
assert all(a.shape == (len(SEEDS_C), len(DATA_FRACTIONS)) for a in C_ACCURACY.values())
print(f"SMOKE_TEST={SMOKE_TEST} の配線: 実効値 {json.dumps(_effective)} と実際に使われた値が一致: OK")
```

    不変条件: 全 43 学習(本番 35・較正 8)の T = 1984・学習率のスケジュールの形・パディングが同一、本番はすべて同じテスト集合 10,000 枚で評価、較正はテスト集合を評価していない、同じシード・同じ部分集合のデータの順序と拡張が共通(11 組)、ViT の非埋め込みパラメータ数が一致、S を共有: OK
    SMOKE_TEST=False の配線: 実効値 {"level": "prod", "plan": 6, "stage": 2, "train_steps": 1984, "seeds": {"A": 5, "B": 5, "C": 3}, "b_patch_sizes": [8, 4], "fractions": [0.0625, 0.25, 1.0], "runs": 35} と実際に使われた値が一致: OK


### 6.12 判定結果の一覧

前提条件が 1 つでも不成立の実験は、判定関数の結果に関わらず「前提不成立」とする(判定不能とは区別する)。判定関数の結果は
参考として別欄に残す。選ばれた計画($T$・段階)と、判定に実際に使ったシード数・水準を印字する。


```python
def verdict_label(computed: str, preconditions: list[str]) -> str:
    return computed if all(precondition_status.get(p) for p in preconditions) else "前提不成立"


print(f"{_tag}計画の選択: {PLAN_SELECTION_MESSAGE}")
print(
    f"{_tag}判定に使った値: 計画 {SELECTED_PLAN}(T = {TRAIN_STEPS}、段階 {SELECTED_STAGE})、学習率 ViT {LEARNING_RATE['vit']:g}・CNN {LEARNING_RATE['cnn']:g}、"
    f"n_A = {len(SEEDS_A)}、n_B = {len(SEEDS_B)}(P {B_PATCH_SIZES})、n_C = {len(SEEDS_C)}(f {DATA_FRACTIONS})"
)
_rows = [
    ("A", f"Delta_A = {CONTRAST_A['value']:+.4f}(sigma {CONTRAST_A['sigma']:.4f})", ["P0(ViT)", "P1(A)"], verdict_A),
    ("B", f"beta_bar = {CONTRAST_B['value']:+.5f}(sigma {CONTRAST_B['sigma']:.5f})", ["P0(ViT)", "P1(B)"], verdict_B),
    ("C", f"Delta_C = {CONTRAST_C['value']:+.5f}(sigma {CONTRAST_C['sigma']:.5f})", ["P0(ViT)", "P0(CNN)", "P1(C)"], verdict_C),
]
print(f"{_tag}実験 | 対比量 | 前提条件 | 判定関数の結果 | 最終判定")
for _name, _value, _pre, _computed in _rows:
    print(
        f"{_tag}{_name} | {_value} | "
        + ", ".join(f"{p}={precondition_status.get(p)}" for p in _pre)
        + f" | {_computed} | {verdict_label(_computed, _pre)}"
    )
print(f"\nノートブック全体の実行時間: {(time.time() - NOTEBOOK_START_TIME) / 60:.1f} 分")
```

    計画の選択: 残りの予算 98.8 分に収まる番号の最も小さい計画として、計画 6 を選んだ(見積もり 82.7 分、cuda 基準)
    判定に使った値: 計画 6(T = 1984、段階 2)、学習率 ViT 0.0005・CNN 0.004、n_A = 5、n_B = 5(P (8, 4))、n_C = 3(f (0.0625, 0.25, 1.0))
    実験 | 対比量 | 前提条件 | 判定関数の結果 | 最終判定
    A | Delta_A = +0.0393(sigma 0.0036) | P0(ViT)=True, P1(A)=True | 支持 | 支持
    B | beta_bar = +0.05156(sigma 0.00308) | P0(ViT)=True, P1(B)=True | 支持 | 支持
    C | Delta_C = -0.00541(sigma 0.00207) | P0(ViT)=True, P0(CNN)=False, P1(C)=True | 反証 | 前提不成立
    
    ノートブック全体の実行時間: 106.6 分


## 7. 結果・考察 / Results and Discussion

本番実行(Google Colab T4、1 回)のセル出力に基づいて記す。判定(7.2〜7.5 節)は 6.1 節で事前に宣言した基準のみから導く。
結果を見た後に立てた解釈はすべて 7.6 節に分けて記し、検証済みの結論としては扱わない。

### 7.1 実行の概要

- **実行環境**(5.1 節の印字): Tesla T4(compute capability 7.5、総メモリ 14.56 GiB)、Python 3.13.15、torch 2.13.0+cu130
  (ビルド時の CUDA 13.0、cuDNN 92000)、コミット`18e70af`、未コミットの変更なし、実行日時 2026-09-30T01:49:07(UTC)。
  混合精度の実効設定は、FP16 の autocast と動的損失スケーリング(初期スケール 65536、growth_interval 2000、backoff 0.5、growth 2.0)が
  有効、評価は FP32、CNN は channels_last、`cudnn.benchmark = True`、決定的な演算の強制なし。
- **データの準備**(5.3 節): CIFAR-10 のダウンロードに 13 分 46 秒(約 206 kB/s)かかり、データの準備全体で 846.9 秒だった。
  ノートブックの開始から計画の選択までの経過時間は 21.2 分(データの準備 847 秒・確認 1 秒・スケーリングの計測 411 秒)で、
  残りの予算は 98.8 分になった。
  - 6.4 節の表では、計画 5($T = 1984$・段階 1)の見積もりが 101.7 分、計画 6 が 82.7 分、計画 4 が 148.6 分である。ダウンロードの
    13 分 46 秒(826 秒)がなければ残りの予算は約 98.8 + 13.8 = 112.6 分となり、計画 5 が予算内に入っていた(計画 4 は入らない)。
    ダウンロードの時間によって、選ばれる計画が 1 つ後ろにずれたことになる(段階 1 ではなく段階 2 となり、実験 C のシード数が
    5 から 3 に減った)。これは実行時間についての事実の記述であり、判定には影響しない。
- **T4 の 1 ステップの時間**(6.3 節、最大の計測点 496 ステップ)と、第 1 段階のスモークテスト(ローカルの Apple M4 の MPS、FP32。
  コミット`18e70af`の出力の 6.3 節)の値の比は次のとおり。

  | モデル | T4(FP16 の autocast) | MPS(FP32) | MPS / T4 |
  |---|---|---|---|
  | ViT $P = 8$ | 36.6 ms | 67.8 ms | 1.85 |
  | ViT $P = 4$ | 60.7 ms | 227.5 ms | 3.75 |
  | ViT $P = 2$ | 253.0 ms | 1041.2 ms | 4.12 |
  | CNN(ResNet-56) | 47.5 ms | 100.6 ms | 2.12 |

  旧版(6.2 節の「旧版の経緯」)では「T4 は MPS の 3 倍速い」と仮定して $T$ を決めていた。実際の比はモデルによって 1.85〜4.12 倍と
  幅があり、1 ステップの計算量の小さいモデル(ViT $P = 8$、ResNet-56)では 3 倍を下回り、大きいモデル(ViT $P = 4$、$P = 2$)では
  上回った。1 つの倍率では、モデルごとの T4 の速さを表せなかった。
- **選ばれた計画**(6.4 節): 残りの予算 98.8 分に収まる番号の最も小さい計画として、**計画 6($T = 1984$・段階 2)** が選ばれた
  (見積もり 82.7 分)。計画 0〜5 はいずれも見積もりが残りの予算を超えた。段階 2 なので、実験 B の $P = 2$ は学習しておらず、
  実験 A・B のシード数は 5、実験 C のシード数は 3 である。$T = 1984$ は、全データで約 5.6 エポック、$f = 1/4$ で約 22.6 エポック、
  $f = 1/16$ で約 90.4 エポックにあたる。
- **実際の実行時間**: 較正 15.5 分、本番の学習と評価(35 学習)69.8 分、ノートブック全体で 106.6 分(予算 120 分の内側)。
  計画の選択の後に実際にかかった時間は約 106.6 − 21.2 = 85.4 分で、見積もり 82.7 分をわずかに上回った。
- **学習率の較正**(6.5 節、検証集合の正解率):

  | 学習率 | ViT | CNN |
  |---|---|---|
  | 0.00025(ViT の拡張) | 0.5616 | 学習していない |
  | 0.0005 | 0.5884 | 0.7314 |
  | 0.001 | 0.5870 | 0.7926 |
  | 0.002 | 0.4434 | 0.8274 |
  | 0.004(CNN の拡張) | 学習していない | 0.8384 |

  ViT は、元の格子で最良が下端の 0.0005 だったので下方向に 1 点拡張し(0.00025 は 0.5616)、0.0005 が内点の最良となった
  (選んだ学習率 0.0005、P0(ViT) 成立)。CNN は 0.0005 → 0.001 → 0.002 と単調に上がり、上端の 0.002 から上方向に拡張した 0.004 が
  最良(0.8384)で、拡張後も格子の端のままだった(選んだ学習率 0.004、P0(CNN) 不成立)。
- **損失スケーリングで飛ばしたステップ**(6.5・6.6 節の各行): ViT の学習(較正 4 回・本番 26 回)はすべて 0 回。CNN の学習は、
  較正の 4 回と本番の 8 回が各 1 回、本番の $f = 1$・シード 2 の 1 回のみ 2 回だった。

### 7.2 前提条件

| 前提条件 | 値 | 成否 | 前提とする実験 |
|---|---|---|---|
| P0(ViT) | 選んだ学習率 0.0005。拡張後の格子 {0.00025, 0.0005, 0.001, 0.002} の内点 | 成立 | A・B・C |
| P0(CNN) | 選んだ学習率 0.004。拡張後の格子 {0.0005, 0.001, 0.002, 0.004} の上端 | **不成立** | C |
| P1(A) | シード平均のテスト正解率: 条件 0 が 0.5445、条件 1 が 0.5838、条件 2 が 0.6197(すべて 0.25 以上) | 成立 | A |
| P1(B) | $P = 8$ が 0.4807、$P = 4$ が 0.5838 | 成立 | B |
| P1(C) | ViT が 0.4761・0.5736・0.5805、CNN が 0.6990・0.8062・0.8251($f$ = 1/16・1/4・1) | 成立 | C |

P0(CNN) が成立しなかったので、実験 C の最終判定は、判定関数の結果によらず **前提不成立** である(6.1 節の宣言)。

### 7.3 実験 A: 位置埋め込みの寄与(支持)

| シード | 0 | 1 | 2 | 3 | 4 | 平均 |
|---|---|---|---|---|---|---|
| 条件 0(なし) | 0.5410 | 0.5496 | 0.5420 | 0.5440 | 0.5461 | 0.5445 |
| 条件 1(学習可能な 1 次元) | 0.5832 | 0.5889 | 0.5693 | 0.5935 | 0.5841 | 0.5838 |
| 条件 2(2 次元正弦波) | 0.6225 | 0.6149 | 0.6175 | 0.6215 | 0.6220 | 0.6197 |
| $d_s = a_{1,s} - a_{0,s}$ | 0.0422 | 0.0393 | 0.0273 | 0.0495 | 0.0380 | 0.0393 |

- 対比量 $\Delta_A = +0.0393$、$\sigma_A = \mathrm{sd}(d_s)/\sqrt{5} = 0.0036$、閾値 $2\sigma_A = 0.0072$。$\Delta_A > 2\sigma_A$ で、
  前提条件 P0(ViT)・P1(A) が成立しているので、**支持**。5 シードすべてで $d_s > 0$ だった。
- **診断量**(判定なし):
  - パッチを固定の置換で並べ替えたテスト画像での正解率(シード平均): 条件 0 は 0.5445(低下幅 0.0000。3.4.2 節の並べ替え不変性の
    とおり)、条件 1 は 0.5089(低下幅 0.0749)、条件 2 は 0.4002(低下幅 0.2195)。
  - テストの交差エントロピー(シード平均): 条件 0 が 1.2716、条件 1 が 1.1607、条件 2 が 1.0551。
  - 条件 2 − 条件 1 = +0.0359(シードごとに +0.0393・+0.0260・+0.0482・+0.0280・+0.0379)。
  - 検証正解率の推移(6.7 節の右の図、シード平均): ステップ 496 から $T = 1984$ まで、どの時点でも条件 2 > 条件 1 > 条件 0 の順で、
    3 条件とも $T$ の時点でまだ上がり続けていた(最後の区間でも増加している)。ただし、学習率は warmup の後に cosine で最大値の
    0.01 倍まで減衰するので、最後の区間の上昇は学習率を下げきることによる効果(アニーリング)としても起こり、十分に学習した
    モデルでも見られる。この観察だけでは学習不足の根拠にならない。
- **観察**(6.8 節、判定基準を設けない、シード 0):
  - 学習可能な 1 次元の位置埋め込みの余弦類似度: どのパッチも自分の位置で最大になる。格子の中央付近(おおむね内側の 4 × 4)の
    パッチの埋め込みは、中央の領域のパッチの埋め込みどうしで類似度が高く、縁の領域とは低い。縁のパッチの埋め込みは、縁の
    パッチの埋め込みと類似度が高い。特に、上端の行のパッチは上端の行と、下端の行のパッチは下端の行と、左端・右端の列の
    パッチはそれぞれ同じ列と高い。内部の行・列については、同じ行・同じ列で類似度が高いという構造ははっきりしない。
  - 2 次元正弦波の位置埋め込み(学習しない): どの位置の組でも類似度が高く(図の色の範囲 −1〜1 のうち上側に集中する)、位置の差が
    大きいほど緩やかに下がる。
  - 平均の注意距離(検証集合の 1,000 枚): 3 条件・6 層・3 ヘッドのすべてで 12.83〜17.71 画素の範囲にあり、全パッチの組の平均距離
    16.55 画素の近くに分布した。第 1・2 層では平均距離より小さいヘッドが多く(条件 2 の第 1・2 層で 12.83〜14.98 画素、条件 1 の
    第 1 層で 14.39〜16.21 画素、条件 0 の第 1 層で 14.57〜15.10 画素)、第 4〜6 層では 15.86〜17.41 画素だった。

### 7.4 実験 B: パッチサイズと系列長(支持)

段階 2 が選ばれたので $P = 2$($N = 256$)は学習しておらず、**$P \in \{8, 4\}$($N \in \{16, 64\}$)の 2 点の比較** である。

| シード | 0 | 1 | 2 | 3 | 4 | 平均 |
|---|---|---|---|---|---|---|
| $P = 8$($N = 16$) | 0.4816 | 0.4784 | 0.4819 | 0.4712 | 0.4903 | 0.4807 |
| $P = 4$($N = 64$、S) | 0.5832 | 0.5889 | 0.5693 | 0.5935 | 0.5841 | 0.5838 |
| $\beta_s$ | 0.05080 | 0.05525 | 0.04370 | 0.06115 | 0.04690 | 0.05156 |

- 対比量 $\bar{\beta} = +0.05156$($\log_2 N$ あたりの正解率。2 点なので正解率の差 0.1031 を間隔 2 で割った値)、
  $\sigma_B = \mathrm{sd}(\beta_s)/\sqrt{5} = 0.00308$、閾値 0.00616。$\bar{\beta} > 2\sigma_B$ で、前提条件 P0(ViT)・P1(B) が成立して
  いるので、**支持**。5 シードすべてで $\beta_s > 0$ だった。テストの交差エントロピーは $P = 8$ が 1.4314、$P = 4$ が 1.1607。
- **交絡**(6.1 節で宣言したとおり): 1 枚あたりの順伝播の計算量は $P = 4$ が 365.7 MFLOP、$P = 8$ が 92.8 MFLOP で約 4 倍
  (3.94 倍)、T4 での 1 ステップの時間は 60.7 ms と 36.6 ms で約 1.66 倍だった。同じ $T$ でも $P = 4$ の方が多くの計算をしており、
  「系列が長いこと」と「計算量が多いこと」の効果を分離できない。判定が支持するのは「同じ $T$ で学習したとき、$P = 4$ の方が
  $P = 8$ より正解率が高い」ことであり、その原因をどちらかに帰することはできない。

### 7.5 実験 C: 帰納バイアスとデータ量(前提不成立)

**最終判定は前提不成立である。** 前提条件 P0(CNN) が成立しなかったため、仮説(データ量の対数に対するテスト正解率の傾きが、
ViT の方が CNN より大きい)について、支持とも反証とも言えない。6.12 節の表の「判定関数の結果: 反証」は、前提条件と切り離して
計算した参考値にすぎず、結論ではない。

以下は記録として示す値である。**前提不成立のため、これらの値から結論は導かない。**

| データ量 $f$ | 1/16 | 1/4 | 1 |
|---|---|---|---|
| ViT のテスト正解率(3 シードの平均) | 0.4761 | 0.5736 | 0.5805 |
| CNN のテスト正解率(3 シードの平均) | 0.6990 | 0.8062 | 0.8251 |

- 傾きの平均: $\bar{\gamma}_{\mathrm{ViT}} = +0.02609$、$\bar{\gamma}_{\mathrm{CNN}} = +0.03151$。$\Delta_C = -0.00541$、
  $\sigma_C = 0.00207$、閾値 0.00414。ロジットの傾きの差(診断量)は −0.0720(同じ式の標準偏差 0.0099)。
- ViT の $f = 1$ は標準条件 S のシード 0〜2 の記録であり、実験 A・B の 5 シードの平均(0.5838)とは異なる。
- **前提が成立しなかった理由**: CNN の較正で、検証正解率が 0.0005 → 0.001 → 0.002 → 0.004 と単調に上がり、1 回の拡張の後も
  最良の学習率が格子の上端に来た(7.1 節)。したがって、CNN の学習率は最適値より低い可能性があり、CNN の各データ量の正解率が
  どの程度過小になっているかは分からない。学習率が変われば CNN の傾き $\gamma_{\mathrm{CNN}}$ も変わりうるので、$\Delta_C$ の符号も
  この結果からは定まらない。

### 7.6 事後的な解釈(検証済みの結論ではない)

**この節の内容は、すべて結果を見た後に立てた解釈であり、検証していない。** 事前に宣言した判定(7.2〜7.5 節)とは区別する。
各項目で根拠にした数値は、6 節のセル出力の値である。

#### 実験 C について

実験 C は前提不成立であり(7.5 節)、以下の 2 つの問いを分けて扱う。どちらも判定を変えるものではなく、判定関数の参考値
(反証の向き)を結論として扱わないことも変わらない。

**問い 1: なぜ前提条件 P0(CNN) が成立しなかったか**

- **原因の候補(設計)**:
  - 学習率の較正の格子 $\{5 \times 10^{-4}, 10^{-3}, 2 \times 10^{-3}\}$ を ViT と CNN で共通にしていた。batch normalization を持つ
    ResNet は、正規化によって各層の入力の尺度が保たれるので、ViT より大きな学習率に耐えることが多い(He et al. [7] の CIFAR-10 の
    設定は、SGD(確率的勾配降下法、Stochastic Gradient Descent)で学習率 0.1。AdamW とは学習率の尺度が異なるので、数値を
    直接は比べられない)。この候補を直接示す数値(CNN と ViT で学習率への耐性を比べた値)は、セル出力にはない。
  - 格子の拡張は、端の方向に 1 点・1 回に限っていた。最適値が元の格子の上端から 2 倍以上離れていれば、拡張後も端に残る。
- **根拠(診断量)**: CNN の検証正解率は、格子の全点で単調に増加した(0.7314 → 0.7926 → 0.8274 → 0.8384、6.5 節)。学習率を 2 倍に
  するごとの増分は +0.0612、+0.0348、+0.0110 と縮小していた。このことから、最適値は 0.004 のすぐ上(次の 1〜2 点の範囲)にある
  可能性がある。そうであれば、CNN の各データ量の正解率の過小評価は小さかった可能性がある。ただし、増分の縮小が続くかどうかは
  この 4 点からは分からず、0.004 より上で正解率がどこまで上がるかを示す数値はない。
- **確かめる方法**: CNN の格子を上に広げた較正(例: $\{0.004, 0.008, 0.016\}$ を加える)で、検証正解率が内点で最大になるかを見る。
  本トピックでは行っていない。

**問い 2: なぜ判定関数の参考値が「ViT の傾きの方が小さい」向きに出たか**

傾き $\gamma$ を、$f = 1/16 \to 1/4$ と $f = 1/4 \to 1$ の 2 つの区間のテスト正解率の伸び(3 シードの平均。6.10 節の値から計算)に
分けると、次のようになる。

| 区間 | ViT の伸び | CNN の伸び | 差(ViT − CNN) | ViT / CNN |
|---|---|---|---|---|
| $f = 1/16 \to 1/4$ | +0.0975 | +0.1072 | −0.0097 | 0.91 |
| $f = 1/4 \to 1$ | +0.0069 | +0.0189 | −0.0120 | 0.37 |
| $f = 1/16 \to 1$(全体) | +0.1044 | +0.1261 | −0.0217 | 0.83 |

- 伸びの差の絶対値は 2 つの区間にほぼ半分ずつ(−0.0097 と −0.0120)分かれており、どちらか一方の区間だけから来ているのではない。
  一方、比で見ると、前半の区間では ViT は CNN の 91% の伸びがあるのに対し、後半の区間では 37% にとどまる。ViT の伸びは、後半の
  区間でほとんど止まっていた(+0.0975 から +0.0069 へ)。
- **原因の候補(現象)**: 全データの ViT は、学習データの共通部分 $S_{1/16}$ での正解率(0.5877)とテスト正解率(0.5805)の差が
  0.0072 しかなく、学習データにすら当てはまっていない(6.10 節)。データを 4 倍に増やしても学習データへの当てはまりが上がらない
  状態では、テスト正解率も上がりようがない。したがって、全データ付近の ViT は **データ量ではなく学習ステップ数に律速されていた**
  可能性がある。この解釈の主な根拠は、上の差 0.0072 の小ささである。$T = 1984$ は全データで約 5.6 エポックにあたる。
  対照として、CNN の同じ差は 0.8463 − 0.8251 = 0.0212 で、ViT の約 3 倍あり、CNN は全データでも学習データに ViT より強く
  当てはまっていた。$f = 1/16$ では、ViT も $S_{1/16}$ での正解率 0.9081(テストとの差 0.4320)と、少ないデータには当てはまっている。
  補助的な観察として、実験 A の検証正解率は $T$ の時点でまだ上がり続けていた(7.3 節)。ただし、これは cosine の減衰で学習率を
  下げきることによる効果(アニーリング)としても起こり、十分に学習したモデルでも見られるので、単独では学習不足の根拠にならない。
- **原因の候補(設計)**: 問い 1 のとおり CNN の学習率は最適値より低い可能性があり、学習率が変われば CNN の各データ量の正解率、
  したがって $\gamma_{\mathrm{CNN}}$ も変わりうる。ただし、学習率がデータ量ごとの伸びにどちらの向きに効くかを示す数値はセル出力に
  なく、この候補が参考値の向きに寄与したかどうかは分からない。
- **確かめる方法**: $T$ を伸ばしたとき(例: 3968 やそれ以上)に、全データの ViT で $S_{1/16}$ での正解率とテスト正解率の差が開き、
  $f = 1/4 \to 1$ のテスト正解率の伸びが大きくなるかを見る。本トピックでは行っていない。
- **位置づけ**: 仮に P0(CNN) が成立していても、$T = 1984$ では、この参考値は原論文の主張(十分に学習した状態での、データ量による
  汎化の差)とは別の量、すなわち **学習の速さ**(同じステップ数でどこまで当てはまるか)を反映していた可能性が高い。

**教訓: 実験 C を意味のある形で行うための条件**(上の 2 つの問いから導かれる、次のトピックの設計に持ち越せる一般論): 第一に、比べる全モデルが学習データに十分に
当てはまるまでの学習量が要る。汎化の差を測るには、全データでも学習データでの正解率とテスト正解率の差が開いている状態が必要で
あり、それを満たさない学習量では学習の速さの比較になる。第二に、学習率の較正の格子は、最適値の尺度が異なりうるモデルごとに
設計する(正規化の有無や最適化手法の既定値の違いを考慮して格子の中心を変え、端に来たときの拡張も最適値を内側に挟むまで
許す)。どちらも、学習量と格子を本番の前に宣言し、見積もりのみから選ぶという枠組みの中で満たせる。

#### 実験 A・B の観察について

1. **2 次元正弦波の位置埋め込みが、学習可能な 1 次元より正解率が高かったこと**(条件 2 − 条件 1 = +0.0359、5 シードすべてで正)。
   - 短い学習では、2 次元の格子の構造を事前に与える方が有利である可能性がある。学習可能な方式は、その構造を学習で獲得する
     必要があり、$T = 1984$ ではそれが間に合っていなかったのかもしれない。
   - 並べ替えたテスト画像での正解率の低下幅が、2 次元正弦波で 0.2195、学習可能な 1 次元で 0.0749 と、2 次元正弦波の方が大きい
     ことは、2 次元正弦波のモデルが位置の情報をより強く使っていることと整合する。
2. **学習された位置埋め込みの類似度の構造**(7.3 節の観察)。
   - 原論文の図 7 中央に見られる「同じ行・同じ列のパッチどうしが似る」構造よりも、「中央のパッチどうし・縁のパッチどうしが似る」
     同心円状の構造が目立った(上下左右の端の行・列については、同じ行・同じ列で似る構造も見える)。
   - 仮説: データ拡張の random crop(パディング 4)により、縁のパッチは零で埋めた画素を含むことが多い。そのため、位置埋め込みが
     「縁からの距離」を表すように学習された可能性がある。
   - 検証には、パディングなしのデータ拡張との比較が要る。本トピックでは行っていない。
3. **平均の注意距離がほぼ一様だったこと**(7.3 節の観察)。
   - どの条件・層・ヘッドも 12.83〜17.71 画素で、全パッチの組の平均距離 16.55 画素に近い。近くのパッチだけに注意する局所的な
     ヘッド(平均距離が 1 パッチ分の 4 画素に近いもの)は見られなかった。
   - これは Raghu et al. [5] の「データが少ないと、下位層が局所的な注意を学習しない」という指摘と整合するが、本トピックの規模では
     原因(データ量・学習量・格子の小ささ)を切り分けられない。
   - $8 \times 8$ の格子では、パッチの中心間の距離は 0〜約 39.6 画素($4 \times 7\sqrt{2}$)の範囲にしかなく、局所的な注意と大域的な
     注意の差が距離の値に現れにくい。
   - 位置埋め込みのない条件 0 でも、第 1 層の平均の注意距離は 14.57〜15.10 画素と全パッチの組の平均距離より小さかった。位置を区別できない
     モデルでも、内容の似たパッチ(近くのパッチは内容も似ていることが多い)に注意が向けば距離は短くなりうるので、この量は位置の
     情報の利用だけを表すものではない。

### 7.7 まとめ

| 実験 | 対比量 | 最終判定 |
|---|---|---|
| A: 位置埋め込みの寄与 | $\Delta_A = +0.0393$($\sigma_A = 0.0036$) | 支持 |
| B: パッチサイズと系列長 | $\bar{\beta} = +0.05156$($\sigma_B = 0.00308$)、$P \in \{8, 4\}$ の 2 点 | 支持 |
| C: 帰納バイアスとデータ量 | (P0(CNN) 不成立のため判定しない) | 前提不成立 |

- **言えること**: CIFAR-10 で、本トピックの ViT($D = 192$、6 層)を $T = 1984$(全データで約 5.6 エポック)学習した時点では、
  (A)学習可能な 1 次元の位置埋め込みは、位置埋め込みなしに比べてテスト正解率を上げた。(B)同じ $T$ で学習したとき、$P = 4$
  ($N = 64$)は $P = 8$($N = 16$)よりテスト正解率が高かった。ただし、B は計算量の違いと分離できず、$P = 2$ を含む 3 水準での
  傾向は調べていない。
- **言えないこと**: 実験 C の仮説(データを減らしたときの正解率の落ち方が ViT の方が大きい)については、支持とも反証とも言えない。
  原論文の大規模な事前学習の主張(データ量とともに ViT が CNN を追い越す)についても、本トピックの結果からは何も言えない。
  また、A・B の結論は学習の途中の比較である可能性があり、十分に学習した状態で同じ差が残るかは分からない。根拠は、同じ構成
  (標準条件 S)の全データの ViT が、学習データの共通部分 $S_{1/16}$ での正解率とテスト正解率の差が 0.0072 しかなく、学習データに
  ほとんど当てはまっていなかったことである(7.6 節の問い 2。この判断自体も事後的な解釈である)。
- **今後の課題**: 実験 C を意味のある形で行うための条件(十分な学習量と、モデルごとの較正の格子)は、7.6 節の「実験 C について」の末尾の教訓にまとめた。
  本トピックでは再実行しない。

本番実行の後に更新が必要だった箇所(7 節の本文、1 節の概要への結果の要約、`theories/README.md`の 019 の行、選ばれた計画・$T$・
段階の記載)は、本番のセル出力に基づいて更新した。6.2 節の本文は、本番前に書いた根拠として変更していない。


## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/05_vision_language/019_vision_transformer.ipynb
