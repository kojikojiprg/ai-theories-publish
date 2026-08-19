---
title: "学習の安定化(Training Stabilization)(実装・実験編 2/3)"
---

この記事は後編(実装・実験編 2/3)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/007_training_stabilization-practice-1)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/007_training_stabilization-practice-3)。

### 6.4 判定関数と 4 分岐到達性の確認

6.2 節の共通公式を実装した `judge_three_way`を、主張 1〜6 それぞれのラッパー関数(`judge_claim_1` 〜 `judge_claim_6`)から呼び出す。**ダミー入力(差が明確に正・負・ゼロ近辺の 3 パターン)** を与えて、支持 / 反証 / 判定不能のすべてに到達できることを、本番のシードデータを使う前に確認する。さらに、`judge_with_precondition`(前提条件が不成立の場合に `judge_claim_*` を呼ばず 「前提不成立」を返すラッパー、6.2 節の前提条件宣言に対応)を経由した場合に、**支持 / 反証 / 判定不能 / 前提不成立の 4 分岐すべて** に到達できることも確認する(`judge_claim_1` 〜 `judge_claim_6` 自体の中身は変更しない)。



```python
def judge_three_way(delta: float, threshold: float, expect_positive: bool) -> str:
    """6.2 節の共通公式による 3 分岐判定。

    expect_positive=True の場合、delta > threshold で「支持」、delta < -threshold で「反証」。
    expect_positive=False の場合はこの向きが反転する(delta < -threshold で「支持」)。
    """
    if expect_positive:
        if delta > threshold:
            return "支持"
        elif delta < -threshold:
            return "反証"
        return "判定不能"
    else:
        if delta < -threshold:
            return "支持"
        elif delta > threshold:
            return "反証"
        return "判定不能"


def contrast_stats(values_a, values_b) -> dict:
    a, b = np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)
    mean_a, mean_b = a.mean(), b.mean()
    std_a = a.std(ddof=1) if len(a) > 1 else 0.0
    std_b = b.std(ddof=1) if len(b) > 1 else 0.0
    delta = mean_b - mean_a
    se = float(np.sqrt(std_a**2 / len(a) + std_b**2 / len(b)))
    return {
        "mean_a": float(mean_a), "mean_b": float(mean_b), "std_a": float(std_a), "std_b": float(std_b),
        "n_a": len(a), "n_b": len(b), "delta": float(delta), "se": se, "threshold": 2 * se,
    }


def judge_claim_1(values_q1, values_q3) -> tuple[str, dict]:
    s = contrast_stats(values_q1, values_q3)
    return judge_three_way(s["delta"], s["threshold"], expect_positive=True), s


def judge_claim_2a(values_q1, values_q2) -> tuple[str, dict]:
    s = contrast_stats(values_q1, values_q2)
    return judge_three_way(s["delta"], s["threshold"], expect_positive=True), s


def judge_claim_2b(values_q3, values_q4_none) -> tuple[str, dict]:
    s = contrast_stats(values_q3, values_q4_none)
    return judge_three_way(s["delta"], s["threshold"], expect_positive=True), s


def judge_claim_3(values_q4_none, values_q4_clip) -> tuple[str, dict]:
    s = contrast_stats(values_q4_none, values_q4_clip)
    return judge_three_way(s["delta"], s["threshold"], expect_positive=False), s


def judge_claim_4(values_q4_none, values_q4_warmup) -> tuple[str, dict]:
    s = contrast_stats(values_q4_none, values_q4_warmup)
    return judge_three_way(s["delta"], s["threshold"], expect_positive=False), s


def judge_claim_5(values_q4_none, values_q4_all) -> tuple[str, dict]:
    s = contrast_stats(values_q4_none, values_q4_all)
    return judge_three_way(s["delta"], s["threshold"], expect_positive=False), s


def judge_claim_6(values_adamw, values_l2) -> tuple[str, dict]:
    """B=Adam(L2 正則化混入)、A=AdamW。delta = mean(L2) - mean(AdamW)。
    L2 の方が群間の乖離が大きい(delta > 0)ことが支持方向。"""
    s = contrast_stats(values_adamw, values_l2)
    return judge_three_way(s["delta"], s["threshold"], expect_positive=True), s



def judge_with_precondition(precondition_ok: bool, judge_fn, *args) -> tuple[str, dict | None]:
    """前提条件(precondition)が成立している場合のみ ``judge_fn`` を呼ぶ。

    ``judge_three_way``・``contrast_stats``・``judge_claim_1`` 〜 ``judge_claim_6`` の
    中身は一切変更しない(判定基準そのものは変更しないため)。
    前提条件が不成立の場合、``judge_fn`` を呼び出さずに ``("前提不成立", None)`` を返す。
    """
    if not precondition_ok:
        return "前提不成立", None
    return judge_fn(*args)

```


```python
def judge_claim_3_prime(values_q4_none_new, values_q4_clip_new) -> tuple[str, dict]:
    """主張3': gradient clipping 適用前の勾配ノルムのピーク / 平均比率を対比量とする、
    主張3とは別の事前登録(6.2.1 節)。新規シード5〜9(SEEDS_NEW_CLAIMS)の
    Q4_none・Q4_clipのみを入力とし、既存シード0〜4のデータは含まない。
    既存の judge_three_way・contrast_stats は変更せずそのまま呼び出す。"""
    s = contrast_stats(values_q4_none_new, values_q4_clip_new)
    return judge_three_way(s["delta"], s["threshold"], expect_positive=False), s
```


```python
def judge_claim_4_prime(values_q4_none_new, values_q4_warmup_new) -> tuple[str, dict]:
    """主張4': warmup + cosine スケジュールによる損失差分(loss_step_delta)系列の
    標準偏差の抑制を対比量とする、主張4とは別の事前登録(6.2.2 節)。新規シード5〜9
    (SEEDS_NEW_CLAIMS)のQ4_none・Q4_warmup_cosineのみを入力とし、既存シード0〜4の
    データは含まない。既存の judge_three_way・contrast_stats は変更せずそのまま呼び出す。"""
    s = contrast_stats(values_q4_none_new, values_q4_warmup_new)
    return judge_three_way(s["delta"], s["threshold"], expect_positive=False), s
```


```python
# 4 分岐到達性の確認(ダミー入力、本番データを使う前に実施)。
# 条件 A の値を固定し、条件 B の値をずらすことで delta を明確に正・負・ゼロ近辺にする。
_dummy_a = [1.0, 1.02, 0.98]
_dummy_cases = {
    "delta明確に正": [3.0, 3.05, 2.95],
    "delta明確に負": [-1.0, -0.95, -1.05],
    "deltaゼロ近辺": [1.0, 1.03, 0.97],
}

_claim_judges = {
    "claim_1": judge_claim_1,
    "claim_2a": judge_claim_2a,
    "claim_2b": judge_claim_2b,
    "claim_3": judge_claim_3,
    "claim_4": judge_claim_4,
    "claim_5": judge_claim_5,
    "claim_6": judge_claim_6,
    "claim_3_prime": judge_claim_3_prime,
    "claim_4_prime": judge_claim_4_prime,
}

# (1) 前提条件が成立している場合: judge_with_precondition(True, ...) は judge_claim_* を
#     そのまま呼び出すため、支持 / 反証 / 判定不能の 3 分岐に到達できることを確認する
#     (judge_claim_* 自体への到達性確認を兼ねる)。
for _name, _judge_fn in _claim_judges.items():
    _reached = set()
    for _case_name, _dummy_b in _dummy_cases.items():
        _verdict, _ = judge_with_precondition(True, _judge_fn, _dummy_a, _dummy_b)
        _reached.add(_verdict)
    print(f"{_name}(前提成立): 到達した判定 = {sorted(_reached)}")
    assert _reached == {"支持", "反証", "判定不能"}, f"{_name} が 3 分岐すべてに到達しない: {_reached}"

# (2) 前提条件が不成立の場合: judge_claim_* を呼ばずに「前提不成立」を返すことを確認する。
for _name, _judge_fn in _claim_judges.items():
    _verdict, _stats = judge_with_precondition(False, _judge_fn, _dummy_a, _dummy_cases["delta明確に正"])
    assert _verdict == "前提不成立", f"{_name} が前提不成立時に正しく分岐しない: {_verdict}"
    assert _stats is None, f"{_name} が前提不成立時に judge_fn を呼び出してしまっている(stats が None でない)"
print("全 claim: 前提条件不成立時 -> 前提不成立(judge_claim_* は呼び出されない)")

print("\nOK: 主張 1〜6・3'・4' のすべての判定関数が、支持 / 反証 / 判定不能 / 前提不成立の 4 分岐すべてに到達する")

```

    claim_1(前提成立): 到達した判定 = ['判定不能', '反証', '支持']
    claim_2a(前提成立): 到達した判定 = ['判定不能', '反証', '支持']
    claim_2b(前提成立): 到達した判定 = ['判定不能', '反証', '支持']
    claim_3(前提成立): 到達した判定 = ['判定不能', '反証', '支持']
    claim_4(前提成立): 到達した判定 = ['判定不能', '反証', '支持']
    claim_5(前提成立): 到達した判定 = ['判定不能', '反証', '支持']
    claim_6(前提成立): 到達した判定 = ['判定不能', '反証', '支持']
    claim_3_prime(前提成立): 到達した判定 = ['判定不能', '反証', '支持']
    claim_4_prime(前提成立): 到達した判定 = ['判定不能', '反証', '支持']
    全 claim: 前提条件不成立時 -> 前提不成立(judge_claim_* は呼び出されない)
    
    OK: 主張 1〜6・3'・4' のすべての判定関数が、支持 / 反証 / 判定不能 / 前提不成立の 4 分岐すべてに到達する


### 6.5 条件グリッドの実行(全 10 条件)

3.5 節の実験グリッドを `CONDITIONS`として定義し、各条件を対応するシード数だけ実行する。結果は設定のハッシュをキーに JSON へ永続化し(006 5.8 節の方針を踏襲)、ノートブックを再実行する際の冗長な再計算を避ける。



```python
CONDITIONS = [
    {"name": "Q1_none", "norm_first": True, "lr_level": "base", "stab": "none", "n_seeds": SEEDS_OTHER},
    {"name": "Q1_all", "norm_first": True, "lr_level": "base", "stab": "all", "n_seeds": SEEDS_OTHER},
    {"name": "Q2_none", "norm_first": True, "lr_level": "high", "stab": "none", "n_seeds": SEEDS_OTHER},
    {"name": "Q2_all", "norm_first": True, "lr_level": "high", "stab": "all", "n_seeds": SEEDS_OTHER},
    {"name": "Q3_none", "norm_first": False, "lr_level": "base", "stab": "none", "n_seeds": SEEDS_OTHER},
    {"name": "Q3_all", "norm_first": False, "lr_level": "base", "stab": "all", "n_seeds": SEEDS_OTHER},
    {"name": "Q4_none", "norm_first": False, "lr_level": "high", "stab": "none", "n_seeds": SEEDS_MAIN_AXIS},
    {"name": "Q4_warmup_cosine", "norm_first": False, "lr_level": "high", "stab": "warmup_cosine", "n_seeds": SEEDS_MAIN_AXIS},
    {"name": "Q4_clip", "norm_first": False, "lr_level": "high", "stab": "clip", "n_seeds": SEEDS_MAIN_AXIS},
    {"name": "Q4_all", "norm_first": False, "lr_level": "high", "stab": "all", "n_seeds": SEEDS_MAIN_AXIS},
]
assert sum(c["n_seeds"] for c in CONDITIONS) == TOTAL_CONDITION_SEED_PAIRS

LR_BY_LEVEL = {"base": BASE_LEARNING_RATE, "high": HIGH_LEARNING_RATE}
print(f"条件数: {len(CONDITIONS)} / シード合計: {sum(c['n_seeds'] for c in CONDITIONS)}")
for c in CONDITIONS:
    print(f"  {c['name']:20s} norm_first={c['norm_first']!s:5s} lr={c['lr_level']:4s} "
          f"stab={c['stab']:14s} n_seeds={c['n_seeds']}")

```

    条件数: 10 / シード合計: 38
      Q1_none              norm_first=True  lr=base stab=none           n_seeds=3
      Q1_all               norm_first=True  lr=base stab=all            n_seeds=3
      Q2_none              norm_first=True  lr=high stab=none           n_seeds=3
      Q2_all               norm_first=True  lr=high stab=all            n_seeds=3
      Q3_none              norm_first=False lr=base stab=none           n_seeds=3
      Q3_all               norm_first=False lr=base stab=all            n_seeds=3
      Q4_none              norm_first=False lr=high stab=none           n_seeds=5
      Q4_warmup_cosine     norm_first=False lr=high stab=warmup_cosine  n_seeds=5
      Q4_clip              norm_first=False lr=high stab=clip           n_seeds=5
      Q4_all               norm_first=False lr=high stab=all            n_seeds=5



```python
RESULTS_CACHE_PATH = ROOT / ".cache" / "007_results_cache.json"


def build_results_cache_key() -> dict:
    return {
        "SMOKE_TEST": SMOKE_TEST,
        "NUM_STEPS": NUM_STEPS,
        "WARMUP_STEPS": WARMUP_STEPS,
        "D_MODEL": D_MODEL,
        "NUM_LAYERS": NUM_LAYERS,
        "NUM_HEADS": NUM_HEADS,
        "D_FF": D_FF,
        "SEQUENCE_LENGTH": SEQUENCE_LENGTH,
        "BATCH_SIZE": BATCH_SIZE,
        "BASE_LEARNING_RATE": BASE_LEARNING_RATE,
        "HIGH_LEARNING_RATE": HIGH_LEARNING_RATE,
        "GRADIENT_CLIP_THRESHOLD": GRADIENT_CLIP_THRESHOLD,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "MIN_LEARNING_RATE_RATIO": MIN_LEARNING_RATE_RATIO,
        "conditions": [(c["name"], c["norm_first"], c["lr_level"], c["stab"], c["n_seeds"]) for c in CONDITIONS],
    }


def load_results_cache(current_key: dict) -> dict | None:
    if not RESULTS_CACHE_PATH.exists():
        return None
    with open(RESULTS_CACHE_PATH) as f:
        payload = json.load(f)
    if payload.get("cache_key") != current_key:
        print(f"[キャッシュ] 設定が変更されているため無効化します: {RESULTS_CACHE_PATH}")
        return None
    print(f"[キャッシュ] {RESULTS_CACHE_PATH} から読み込みました(設定が完全一致)")
    return payload["results"]


def save_results_cache(results: dict, cache_key: dict) -> None:
    RESULTS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CACHE_PATH, "w") as f:
        json.dump({"cache_key": cache_key, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"[キャッシュ] 実験結果を保存しました: {RESULTS_CACHE_PATH}")


_cache_key = build_results_cache_key()
_cached = load_results_cache(_cache_key)

if _cached is not None:
    condition_results = _cached
else:
    condition_results = {}
    for c in CONDITIONS:
        peak_to_mean_list, max_loss_increase_list, final_loss_list, histories = [], [], [], []
        for seed_idx in range(c["n_seeds"]):
            hist = _run_lm_condition(
                norm_first=c["norm_first"],
                stabilization_level=c["stab"],
                peak_lr=LR_BY_LEVEL[c["lr_level"]],
                clip_threshold=GRADIENT_CLIP_THRESHOLD,
                num_steps=NUM_STEPS,
                seed=seed_idx,
            )
            assert hist["step"][-1] == NUM_STEPS  # 不変条件: 全条件でステップ数が同一
            peak_to_mean_list.append(compute_gradient_norm_peak_to_mean_ratio(hist["gradient_norm"]))
            max_loss_increase_list.append(compute_max_single_step_loss_increase(hist["loss_step_delta"]))
            final_loss_list.append(hist["train_loss"][-1])
            histories.append(hist)
        condition_results[c["name"]] = {
            "peak_to_mean": peak_to_mean_list,
            "max_loss_increase": max_loss_increase_list,
            "final_loss": final_loss_list,
            "histories": histories,
        }
        print(f"{c['name']:20s} peak/mean={np.mean(peak_to_mean_list):.3f} "
              f"max_loss_increase={np.mean(max_loss_increase_list):.4f} "
              f"final_loss={np.mean(final_loss_list):.4f}")
    save_results_cache(condition_results, _cache_key)

# 不変条件: 評価バッチ集合が全条件・シードで変更されていないことを確認する。
assert torch.equal(eval_windows, _eval_windows_snapshot)
assert torch.equal(eval_mask, _eval_mask_snapshot)
assert total_eval_bytes == _total_eval_bytes_snapshot
print("\nOK: 評価バッチ集合は全条件・シードを通じて変更されていない")

```

    Q1_none              peak/mean=5.035 max_loss_increase=0.0810 final_loss=1.6925
    Q1_all               peak/mean=5.233 max_loss_increase=0.0684 final_loss=1.9060
    Q2_none              peak/mean=6.770 max_loss_increase=0.0986 final_loss=1.5265
    Q2_all               peak/mean=6.577 max_loss_increase=0.0917 final_loss=1.6447
    Q3_none              peak/mean=5.975 max_loss_increase=0.0800 final_loss=1.7140
    Q3_all               peak/mean=6.186 max_loss_increase=0.0649 final_loss=1.9200
    Q4_none              peak/mean=8.706 max_loss_increase=0.1626 final_loss=1.5674
    Q4_warmup_cosine     peak/mean=11.307 max_loss_increase=0.1224 final_loss=1.8008
    Q4_clip              peak/mean=7.831 max_loss_increase=0.1024 final_loss=1.5229
    Q4_all               peak/mean=7.484 max_loss_increase=0.0911 final_loss=1.6730
    [キャッシュ] 実験結果を保存しました: .cache/007_results_cache.json
    
    OK: 評価バッチ集合は全条件・シードを通じて変更されていない


### 前提条件 P1・P2 の確認

6.2 節で宣言した前提条件 P1(学習の進行)・P2(gradient clipping の発動)を、6.5 節で得た全条件・全シードの結果に照らして確認する。この結果は、続く 6.6・6.8・6.9 節の判定で、主張ごとに判定を行うか「前提不成立」とするかを決める入力になる。


```python
# 前提条件 P1 の閾値(PRECONDITION_LOSS_THRESHOLD)は 5.3 節末尾(VOCAB_SIZE 確定直後)
# で一度だけ定義済みであり、ここではそれをそのまま参照する(定義の重複を避ける)。
print(f"前提条件 P1(学習の進行): 直近 {P1_LOSS_WINDOW_STEPS} ステップの訓練損失の平均 "
      f"<= ln(V) x PRECONDITION_LOSS_RATIO = {PRECONDITION_LOSS_THRESHOLD:.4f}"
      f"(ln(V)={np.log(VOCAB_SIZE):.4f}, PRECONDITION_LOSS_RATIO={PRECONDITION_LOSS_RATIO})\n")
print("(主張 5 の対比量である固定ステップ数終了時点の損失値 final_loss = train_loss[-1] とは"
      "別の量である点に注意。P1 はバッチのばらつきの影響を抑えるため窓平均を使う。)\n")

precondition_p1_ok: dict[str, bool] = {}
print(f"{'condition':20s} {'seed':>4s} {'p1_loss':>12s} {'threshold':>10s} {'P1':>6s}")
for c in CONDITIONS:
    name = c["name"]
    _seed_ok = []
    for seed_idx, hist in enumerate(condition_results[name]["histories"]):
        p1_loss = float(np.mean(hist["train_loss"][-P1_LOSS_WINDOW_STEPS:]))
        ok = p1_loss <= PRECONDITION_LOSS_THRESHOLD
        _seed_ok.append(ok)
        print(f"{name:20s} {seed_idx:4d} {p1_loss:12.4f} {PRECONDITION_LOSS_THRESHOLD:10.4f} "
              f"{'OK' if ok else 'NG':>6s}")
    precondition_p1_ok[name] = all(_seed_ok)

print(f"\n前提条件 P2(gradient clipping の発動): 発動ステップ比率 >= PRECONDITION_CLIP_RATE_MIN = "
      f"{PRECONDITION_CLIP_RATE_MIN}\n")

precondition_p2_ok: dict[str, bool] = {}
CLIP_ENABLED_CONDITIONS = ("Q4_clip", "Q4_all")
print(f"{'condition':20s} {'seed':>4s} {'clip_rate':>10s} {'P2':>6s}")
for name in CLIP_ENABLED_CONDITIONS:
    _seed_ok = []
    for seed_idx, hist in enumerate(condition_results[name]["histories"]):
        rate = float(np.mean(hist["gradient_clip_triggered"]))
        ok = rate >= PRECONDITION_CLIP_RATE_MIN
        _seed_ok.append(ok)
        print(f"{name:20s} {seed_idx:4d} {rate:10.4f} {'OK' if ok else 'NG':>6s}")
    precondition_p2_ok[name] = all(_seed_ok)

# 主張ごとの前提条件: 対比条件(6.2 節の表)が全て P1 を満たし、かつ clipping を含む
# 条件については P2 も満たすことを要求する。主張 6(合成タスク)は言語モデルの学習を
# 伴わないため前提条件を持たない(常に成立、そもそも仮説に依存しない
# 性質から前提条件の対象外とする)。
CLAIM_PRECONDITION_CONDITIONS = {
    "claim_1": ["Q1_none", "Q3_none"],
    "claim_2a": ["Q1_none", "Q2_none"],
    "claim_2b": ["Q3_none", "Q4_none"],
    "claim_3": ["Q4_none", "Q4_clip"],
    "claim_4": ["Q4_none", "Q4_warmup_cosine"],
    "claim_5": ["Q4_none", "Q4_all"],
    "claim_6": [],
}

precondition_ok_by_claim: dict[str, bool] = {}
print("\n主張ごとの前提条件の成否:")
for claim, conds in CLAIM_PRECONDITION_CONDITIONS.items():
    ok = all(precondition_p1_ok[c] for c in conds)
    ok = ok and all(precondition_p2_ok[c] for c in conds if c in CLIP_ENABLED_CONDITIONS)
    precondition_ok_by_claim[claim] = ok
    print(f"  {claim}: {'前提成立' if ok else '前提不成立'}")

if not all(precondition_ok_by_claim.values()):
    _failed = [c for c, ok in precondition_ok_by_claim.items() if not ok]
    print(f"\n前提不成立の主張: {_failed}(6.6・6.8・6.9 節で判定を行わず「前提不成立」として記録する)")
else:
    print("\nOK: すべての主張で前提条件が成立している")

```

    前提条件 P1(学習の進行): 直近 20 ステップの訓練損失の平均 <= ln(V) x PRECONDITION_LOSS_RATIO = 2.5046(ln(V)=4.1744, PRECONDITION_LOSS_RATIO=0.6)
    
    (主張 5 の対比量である固定ステップ数終了時点の損失値 final_loss = train_loss[-1] とは別の量である点に注意。P1 はバッチのばらつきの影響を抑えるため窓平均を使う。)
    
    condition            seed      p1_loss  threshold     P1
    Q1_none                 0       1.7063     2.5046     OK
    Q1_none                 1       1.6857     2.5046     OK
    Q1_none                 2       1.7313     2.5046     OK
    Q1_all                  0       1.9132     2.5046     OK
    Q1_all                  1       1.8845     2.5046     OK
    Q1_all                  2       1.9257     2.5046     OK
    Q2_none                 0       1.5265     2.5046     OK
    Q2_none                 1       1.5299     2.5046     OK
    Q2_none                 2       1.5533     2.5046     OK
    Q2_all                  0       1.6284     2.5046     OK
    Q2_all                  1       1.6136     2.5046     OK
    Q2_all                  2       1.6680     2.5046     OK
    Q3_none                 0       1.7312     2.5046     OK
    Q3_none                 1       1.7217     2.5046     OK
    Q3_none                 2       1.7390     2.5046     OK
    Q3_all                  0       1.9162     2.5046     OK
    Q3_all                  1       1.9210     2.5046     OK
    Q3_all                  2       1.9264     2.5046     OK
    Q4_none                 0       1.5601     2.5046     OK
    Q4_none                 1       1.5632     2.5046     OK
    Q4_none                 2       1.5806     2.5046     OK
    Q4_none                 3       1.5852     2.5046     OK
    Q4_none                 4       1.5660     2.5046     OK
    Q4_warmup_cosine        0       1.8199     2.5046     OK
    Q4_warmup_cosine        1       1.8214     2.5046     OK
    Q4_warmup_cosine        2       1.7761     2.5046     OK
    Q4_warmup_cosine        3       1.8097     2.5046     OK
    Q4_warmup_cosine        4       1.7663     2.5046     OK
    Q4_clip                 0       1.5200     2.5046     OK
    Q4_clip                 1       1.5174     2.5046     OK
    Q4_clip                 2       1.5182     2.5046     OK
    Q4_clip                 3       1.5422     2.5046     OK
    Q4_clip                 4       1.5195     2.5046     OK
    Q4_all                  0       1.6549     2.5046     OK
    Q4_all                  1       1.6502     2.5046     OK
    Q4_all                  2       1.6659     2.5046     OK
    Q4_all                  3       1.6915     2.5046     OK
    Q4_all                  4       1.6618     2.5046     OK
    
    前提条件 P2(gradient clipping の発動): 発動ステップ比率 >= PRECONDITION_CLIP_RATE_MIN = 0.05
    
    condition            seed  clip_rate     P2
    Q4_clip                 0     0.0800     OK
    Q4_clip                 1     0.0767     OK
    Q4_clip                 2     0.1167     OK
    Q4_clip                 3     0.0767     OK
    Q4_clip                 4     0.0800     OK
    Q4_all                  0     0.1133     OK
    Q4_all                  1     0.0900     OK
    Q4_all                  2     0.0800     OK
    Q4_all                  3     0.1133     OK
    Q4_all                  4     0.1167     OK
    
    主張ごとの前提条件の成否:
      claim_1: 前提成立
      claim_2a: 前提成立
      claim_2b: 前提成立
      claim_3: 前提成立
      claim_4: 前提成立
      claim_5: 前提成立
      claim_6: 前提成立
    
    OK: すべての主張で前提条件が成立している


### 6.5.1 主張 3'・4' 専用: 新規シード(5〜9)による追加実行

6.2.1・6.2.2 節の事前登録に従い、Q4_none・Q4_clip・Q4_warmup_cosine を新規シード`SEEDS_NEW_CLAIMS`(既存の条件グリッドが使うシード 0..`SEEDS_MAIN_AXIS`-1 とは重複しない、5.2 節)で追加実行する。主張 3'・4' はいずれも Q4_none を対照条件(A)として共有するため、**Q4_none の新規シード実行は 1 回のみ行い、両主張の判定で同じ結果を再利用する**(二重実行しない)。既存の条件グリッド実行の仕組み(`_run_lm_condition()`、6.1 節)をそのまま再利用し、シードのリストのみを変更する。結果は 6.5 節と同じキャッシュ方式(設定のハッシュをキーとした JSON 永続化)で、**既存の条件グリッドのキャッシュ(`RESULTS_CACHE_PATH`)とは別のファイル** に保存する(新規シードの追加実行が既存シードのキャッシュを破壊しないため)。

実行の前に、**モデルの初期化とバッチ順序の両方がシードによって支配されていること**(新規シードが既存シードと実質的に独立であるための前提)をアサーションで確認する。片方だけがシード依存だと、新規シードが既存シードと実質的に独立でなくなる。この確認は`build_model()`・バッチ生成のいずれも`stabilization_level`(`none`・`clip`・`warmup_cosine`)を引数に取らないため、Q4_warmup_cosine を含む全条件に共通して成り立つ。個別に再確認する必要はない。


```python
# シードがモデル初期化・バッチ順序の両方を支配していることの確認
# (新規シードが既存シードと実質的に独立であることの前提、6.5.1 節)。

# (a) 異なるシードでは初期パラメータが異なる。
_seed_check_a, _seed_check_b = SEEDS_NEW_CLAIMS[0], SEEDS_NEW_CLAIMS[1]
_m_a = build_model(norm_first=False, seed=_seed_check_a)
_m_b = build_model(norm_first=False, seed=_seed_check_b)
_p_a = next(_m_a.parameters()).detach().clone()
_p_b = next(_m_b.parameters()).detach().clone()
assert not torch.equal(_p_a, _p_b), "異なるシードで初期パラメータが一致してしまっている"
print(f"OK: シード{_seed_check_a}とシード{_seed_check_b}で初期パラメータが異なる")

# (b) 異なるシードではバッチの切り出し開始位置(インデックス列)が異なる。
_gen_a = torch.Generator(device="cpu")
_gen_a.manual_seed(_seed_check_a)
_gen_b = torch.Generator(device="cpu")
_gen_b.manual_seed(_seed_check_b)
_batch_a, _ = get_random_batch(train_ids, BATCH_SIZE, SEQUENCE_LENGTH, _gen_a)
_batch_b, _ = get_random_batch(train_ids, BATCH_SIZE, SEQUENCE_LENGTH, _gen_b)
assert not torch.equal(_batch_a, _batch_b), "異なるシードでバッチ順序が一致してしまっている"
print(f"OK: シード{_seed_check_a}とシード{_seed_check_b}でバッチの切り出し開始位置が異なる")

# (c) 同一シードで 2 回実行すると、モデル初期化・バッチ順序とも完全に再現される。
_m_c1 = build_model(norm_first=False, seed=_seed_check_a)
_m_c2 = build_model(norm_first=False, seed=_seed_check_a)
assert torch.equal(next(_m_c1.parameters()), next(_m_c2.parameters())), "同一シードで初期パラメータが再現されない"
_gen_c1 = torch.Generator(device="cpu")
_gen_c1.manual_seed(_seed_check_a)
_gen_c2 = torch.Generator(device="cpu")
_gen_c2.manual_seed(_seed_check_a)
_batch_c1, _ = get_random_batch(train_ids, BATCH_SIZE, SEQUENCE_LENGTH, _gen_c1)
_batch_c2, _ = get_random_batch(train_ids, BATCH_SIZE, SEQUENCE_LENGTH, _gen_c2)
assert torch.equal(_batch_c1, _batch_c2), "同一シードでバッチ順序が再現されない"
print(f"OK: 同一シード(シード{_seed_check_a})でモデル初期化・バッチ順序とも完全に再現される")
del _m_a, _m_b, _m_c1, _m_c2
```

    OK: シード5とシード6で初期パラメータが異なる
    OK: シード5とシード6でバッチの切り出し開始位置が異なる
    OK: 同一シード(シード5)でモデル初期化・バッチ順序とも完全に再現される



```python
NEW_SEED_CONDITIONS = [
    {"name": "Q4_none_new", "norm_first": False, "lr_level": "high", "stab": "none"},
    {"name": "Q4_clip_new", "norm_first": False, "lr_level": "high", "stab": "clip"},
    {"name": "Q4_warmup_cosine_new", "norm_first": False, "lr_level": "high", "stab": "warmup_cosine"},
]
NEW_SEED_CACHE_PATH = ROOT / ".cache" / "007_new_seed_cache.json"
# 既存の条件グリッドのキャッシュ(RESULTS_CACHE_PATH)とは別ファイルにする
# (新規シードの追加実行が既存シードのキャッシュを破壊しないことを保証するため)。
# 主張3'(Q4_none_new・Q4_clip_new)・主張4'(Q4_none_new・Q4_warmup_cosine_new)が
# このキャッシュを共有する。Q4_none_new は両主張で共通の対照条件(A)であり、
# ここで 1 回だけ実行した結果を両方の判定で再利用する(二重実行しない)。


def build_new_seed_cache_key() -> dict:
    return {
        "SMOKE_TEST": SMOKE_TEST,
        "NUM_STEPS": NUM_STEPS,
        "D_MODEL": D_MODEL,
        "NUM_LAYERS": NUM_LAYERS,
        "NUM_HEADS": NUM_HEADS,
        "D_FF": D_FF,
        "SEQUENCE_LENGTH": SEQUENCE_LENGTH,
        "BATCH_SIZE": BATCH_SIZE,
        "HIGH_LEARNING_RATE": HIGH_LEARNING_RATE,
        "GRADIENT_CLIP_THRESHOLD": GRADIENT_CLIP_THRESHOLD,
        "conditions": [(c["name"], c["norm_first"], c["lr_level"], c["stab"]) for c in NEW_SEED_CONDITIONS],
        "seeds": SEEDS_NEW_CLAIMS,
    }


def load_new_seed_cache(current_key: dict) -> dict | None:
    if not NEW_SEED_CACHE_PATH.exists():
        return None
    with open(NEW_SEED_CACHE_PATH) as f:
        payload = json.load(f)
    if payload.get("cache_key") != current_key:
        print(f"[キャッシュ] 設定が変更されているため無効化します: {NEW_SEED_CACHE_PATH}")
        return None
    print(f"[キャッシュ] {NEW_SEED_CACHE_PATH} から読み込みました(設定が完全一致)")
    return payload["results"]


def save_new_seed_cache(results: dict, cache_key: dict) -> None:
    NEW_SEED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NEW_SEED_CACHE_PATH, "w") as f:
        json.dump({"cache_key": cache_key, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"[キャッシュ] 実験結果を保存しました: {NEW_SEED_CACHE_PATH}")


_new_seed_cache_key = build_new_seed_cache_key()
_new_seed_cached = load_new_seed_cache(_new_seed_cache_key)

if _new_seed_cached is not None:
    new_seed_results = _new_seed_cached
else:
    new_seed_results = {}
    for c in NEW_SEED_CONDITIONS:
        peak_to_mean_list, max_loss_increase_list, loss_step_delta_std_list, final_loss_list, histories = (
            [], [], [], [], []
        )
        for seed_idx in SEEDS_NEW_CLAIMS:
            hist = _run_lm_condition(
                norm_first=c["norm_first"],
                stabilization_level=c["stab"],
                peak_lr=LR_BY_LEVEL[c["lr_level"]],
                clip_threshold=GRADIENT_CLIP_THRESHOLD,
                num_steps=NUM_STEPS,
                seed=seed_idx,
            )
            assert hist["step"][-1] == NUM_STEPS  # 不変条件: 既存の条件グリッドと同一のステップ数
            peak_to_mean_list.append(compute_gradient_norm_peak_to_mean_ratio(hist["gradient_norm"]))
            max_loss_increase_list.append(compute_max_single_step_loss_increase(hist["loss_step_delta"]))
            loss_step_delta_std_list.append(compute_loss_step_delta_std(hist["loss_step_delta"]))
            final_loss_list.append(hist["train_loss"][-1])
            histories.append(hist)
        new_seed_results[c["name"]] = {
            "peak_to_mean": peak_to_mean_list,
            "max_loss_increase": max_loss_increase_list,
            "loss_step_delta_std": loss_step_delta_std_list,
            "final_loss": final_loss_list,
            "histories": histories,
        }
        print(f"{c['name']:20s} peak/mean={np.mean(peak_to_mean_list):.3f} "
              f"max_loss_increase={np.mean(max_loss_increase_list):.4f} "
              f"loss_step_delta_std={np.mean(loss_step_delta_std_list):.4f} "
              f"final_loss={np.mean(final_loss_list):.4f}")
    save_new_seed_cache(new_seed_results, _new_seed_cache_key)

# 不変条件: 主張3'・4'専用の実行でも評価バッチ集合が既存の条件グリッドと変更されていないこと。
assert torch.equal(eval_windows, _eval_windows_snapshot)
assert torch.equal(eval_mask, _eval_mask_snapshot)
assert total_eval_bytes == _total_eval_bytes_snapshot
print("\nOK: 主張3'・4'専用の実行でも評価バッチ集合は変更されていない")
```

    Q4_none_new          peak/mean=9.218 max_loss_increase=0.1250 loss_step_delta_std=0.0548 final_loss=1.5658
    Q4_clip_new          peak/mean=7.383 max_loss_increase=0.1068 loss_step_delta_std=0.0552 final_loss=1.5166
    Q4_warmup_cosine_new peak/mean=7.668 max_loss_increase=0.1143 loss_step_delta_std=0.0358 final_loss=1.7933
    [キャッシュ] 実験結果を保存しました: .cache/007_new_seed_cache.json
    
    OK: 主張3'・4'専用の実行でも評価バッチ集合は変更されていない



```python
# 前提条件 P1(主張3'・4'共通)・P2(主張3'のみに適用)の確認(新規シード5〜9)。
# 6.5節末尾と同じ形式で行う。
precondition_p1_ok_new_seeds: dict[str, bool] = {}
print(f"{'condition':20s} {'seed':>4s} {'p1_loss':>12s} {'threshold':>10s} {'P1':>6s}")
for name in ("Q4_none_new", "Q4_clip_new", "Q4_warmup_cosine_new"):
    _seed_ok = []
    for seed_idx, hist in zip(SEEDS_NEW_CLAIMS, new_seed_results[name]["histories"], strict=True):
        p1_loss = float(np.mean(hist["train_loss"][-P1_LOSS_WINDOW_STEPS:]))
        ok = p1_loss <= PRECONDITION_LOSS_THRESHOLD
        _seed_ok.append(ok)
        print(f"{name:20s} {seed_idx:4d} {p1_loss:12.4f} {PRECONDITION_LOSS_THRESHOLD:10.4f} "
              f"{'OK' if ok else 'NG':>6s}")
    precondition_p1_ok_new_seeds[name] = all(_seed_ok)

print(f"\n前提条件 P2(gradient clipping の発動、新規シード、主張3'のみに適用): 発動ステップ比率 >= "
      f"PRECONDITION_CLIP_RATE_MIN = {PRECONDITION_CLIP_RATE_MIN}\n")
precondition_p2_ok_new_seeds: dict[str, bool] = {}
print(f"{'condition':20s} {'seed':>4s} {'clip_rate':>10s} {'P2':>6s}")
for name in ("Q4_clip_new",):
    _seed_ok = []
    for seed_idx, hist in zip(SEEDS_NEW_CLAIMS, new_seed_results[name]["histories"], strict=True):
        rate = float(np.mean(hist["gradient_clip_triggered"]))
        ok = rate >= PRECONDITION_CLIP_RATE_MIN
        _seed_ok.append(ok)
        print(f"{name:20s} {seed_idx:4d} {rate:10.4f} {'OK' if ok else 'NG':>6s}")
    precondition_p2_ok_new_seeds[name] = all(_seed_ok)

# 主張3': P1(Q4_none_new・Q4_clip_new)とP2(Q4_clip_new)の両方を要求する。
precondition_ok_claim_3_prime = (
    precondition_p1_ok_new_seeds["Q4_none_new"]
    and precondition_p1_ok_new_seeds["Q4_clip_new"]
    and precondition_p2_ok_new_seeds["Q4_clip_new"]
)
print(f"\n主張3'の前提条件: {'前提成立' if precondition_ok_claim_3_prime else '前提不成立'}")

# 主張4': P1(Q4_none_new・Q4_warmup_cosine_new)のみを要求する
# (P2 は gradient clipping 固有の前提条件であり、warmup + cosine には適用しない、6.2.2 節)。
precondition_ok_claim_4_prime = (
    precondition_p1_ok_new_seeds["Q4_none_new"]
    and precondition_p1_ok_new_seeds["Q4_warmup_cosine_new"]
)
print(f"主張4'の前提条件: {'前提成立' if precondition_ok_claim_4_prime else '前提不成立'}")
```

    condition            seed      p1_loss  threshold     P1
    Q4_none_new             5       1.5473     2.5046     OK
    Q4_none_new             6       1.6019     2.5046     OK
    Q4_none_new             7       1.6003     2.5046     OK
    Q4_none_new             8       1.5740     2.5046     OK
    Q4_none_new             9       1.5647     2.5046     OK
    Q4_clip_new             5       1.5268     2.5046     OK
    Q4_clip_new             6       1.5296     2.5046     OK
    Q4_clip_new             7       1.5475     2.5046     OK
    Q4_clip_new             8       1.5241     2.5046     OK
    Q4_clip_new             9       1.5187     2.5046     OK
    Q4_warmup_cosine_new    5       1.8310     2.5046     OK
    Q4_warmup_cosine_new    6       1.8291     2.5046     OK
    Q4_warmup_cosine_new    7       1.7457     2.5046     OK
    Q4_warmup_cosine_new    8       1.7779     2.5046     OK
    Q4_warmup_cosine_new    9       1.7825     2.5046     OK
    
    前提条件 P2(gradient clipping の発動、新規シード、主張3'のみに適用): 発動ステップ比率 >= PRECONDITION_CLIP_RATE_MIN = 0.05
    
    condition            seed  clip_rate     P2
    Q4_clip_new             5     0.0600     OK
    Q4_clip_new             6     0.0733     OK
    Q4_clip_new             7     0.0533     OK
    Q4_clip_new             8     0.0733     OK
    Q4_clip_new             9     0.0900     OK
    
    主張3'の前提条件: 前提成立
    主張4'の前提条件: 前提成立


### 6.6 主張 1〜5 の判定(結論としては書かない)

以下は 6.4 節の判定関数を実際のシードデータに適用した結果である。**この判定結果を「結果・考察」として本文で結論づけない**(7 節は見出しと箇条書きのみ)。ここでは判定関数が正しく動作することの確認としてのみ出力する。前提条件(6.5 節末尾で確認)が不成立の主張は、`judge_with_precondition` により判定関数を呼び出さず「前提不成立」と出力する。



```python
def _v(name, key):
    return condition_results[name][key]


def _judge(claim_key, judge_fn, *args):
    return judge_with_precondition(precondition_ok_by_claim[claim_key], judge_fn, *args)


verdict_1, stats_1 = _judge("claim_1", judge_claim_1, _v("Q1_none", "peak_to_mean"), _v("Q3_none", "peak_to_mean"))
verdict_2a, stats_2a = _judge("claim_2a", judge_claim_2a, _v("Q1_none", "peak_to_mean"), _v("Q2_none", "peak_to_mean"))
verdict_2b, stats_2b = _judge("claim_2b", judge_claim_2b, _v("Q3_none", "peak_to_mean"), _v("Q4_none", "peak_to_mean"))
verdict_3, stats_3 = _judge("claim_3", judge_claim_3, _v("Q4_none", "max_loss_increase"), _v("Q4_clip", "max_loss_increase"))
verdict_4, stats_4 = _judge("claim_4", judge_claim_4, _v("Q4_none", "max_loss_increase"), _v("Q4_warmup_cosine", "max_loss_increase"))
verdict_5, stats_5 = _judge("claim_5", judge_claim_5, _v("Q4_none", "final_loss"), _v("Q4_all", "final_loss"))

for _name, _verdict, _stats in [
    ("主張1(正規化後置 > 正規化前置の不安定性)", verdict_1, stats_1),
    ("主張2a(正規化前置、学習率↑で不安定性↑)", verdict_2a, stats_2a),
    ("主張2b(正規化後置、学習率↑で不安定性↑)", verdict_2b, stats_2b),
    ("主張3(gradient clippingが損失の暴れを抑える)", verdict_3, stats_3),
    ("主張4(warmup+cosineが損失の暴れを抑える)", verdict_4, stats_4),
    ("主張5(全部乗せは収束を遅らせない)", verdict_5, stats_5),
]:
    if _stats is None:
        print(f"{_name}: 判定={_verdict}(前提条件が不成立のため判定を行っていない)")
    else:
        print(f"{_name}: delta={_stats['delta']:+.4f}, threshold={_stats['threshold']:.4f}, 判定={_verdict}")

```

    主張1(正規化後置 > 正規化前置の不安定性): delta=+0.9406, threshold=1.0187, 判定=判定不能
    主張2a(正規化前置、学習率↑で不安定性↑): delta=+1.7347, threshold=1.2065, 判定=支持
    主張2b(正規化後置、学習率↑で不安定性↑): delta=+2.7306, threshold=1.3087, 判定=支持
    主張3(gradient clippingが損失の暴れを抑える): delta=-0.0602, threshold=0.1199, 判定=判定不能
    主張4(warmup+cosineが損失の暴れを抑える): delta=-0.0403, threshold=0.1350, 判定=判定不能
    主張5(全部乗せは収束を遅らせない): delta=+0.1055, threshold=0.0442, 判定=反証



```python
verdict_3_prime, stats_3_prime = judge_with_precondition(
    precondition_ok_claim_3_prime,
    judge_claim_3_prime,
    new_seed_results["Q4_none_new"]["peak_to_mean"],
    new_seed_results["Q4_clip_new"]["peak_to_mean"],
)
if stats_3_prime is None:
    print(f"主張3'(gradient clippingが勾配ノルムのピーク/平均比率を抑える、新規シード5〜9): "
          f"判定={verdict_3_prime}(前提条件が不成立のため判定を行っていない)")
else:
    print(f"主張3'(gradient clippingが勾配ノルムのピーク/平均比率を抑える、新規シード5〜9): "
          f"delta={stats_3_prime['delta']:+.4f}, threshold={stats_3_prime['threshold']:.4f}, "
          f"判定={verdict_3_prime}")
```

    主張3'(gradient clippingが勾配ノルムのピーク/平均比率を抑える、新規シード5〜9): delta=-1.8346, threshold=2.6899, 判定=判定不能



```python
verdict_4_prime, stats_4_prime = judge_with_precondition(
    precondition_ok_claim_4_prime,
    judge_claim_4_prime,
    new_seed_results["Q4_none_new"]["loss_step_delta_std"],
    new_seed_results["Q4_warmup_cosine_new"]["loss_step_delta_std"],
)
if stats_4_prime is None:
    print(f"主張4'(warmup+cosineが損失差分系列の標準偏差を抑える、新規シード5〜9): "
          f"判定={verdict_4_prime}(前提条件が不成立のため判定を行っていない)")
else:
    print(f"主張4'(warmup+cosineが損失差分系列の標準偏差を抑える、新規シード5〜9): "
          f"delta={stats_4_prime['delta']:+.4f}, threshold={stats_4_prime['threshold']:.4f}, "
          f"判定={verdict_4_prime}")
```

    主張4'(warmup+cosineが損失差分系列の標準偏差を抑える、新規シード5〜9): delta=-0.0191, threshold=0.0020, 判定=支持


### 6.6.1 診断: 既存シードによる参考値(主張 3'・4'、判定には使用しない)

既存シード 0〜4 による診断値を表示する: 主張 3' は Q4_none・Q4_clip の勾配ノルムのピーク / 平均比率(3 回目の本番実行、6.10.3 節)、主張 4' は Q4_none・Q4_warmup_cosine の損失差分(`loss_step_delta`)系列の標準偏差である。**6.2.1・6.2.2 節で明記した通り、これらの値はそれぞれの主張の $\Delta$・標準偏差・閾値・判定のいずれにも使わない。** 新規シードによる判定と同じ方向を示すかどうかは、結論の頑健性を示す参考情報として並べて表示する。


```python
# 診断(参考、判定には使用しない): 既存シード0〜4によるQ4_none・Q4_clipの
# 勾配ノルムのピーク/平均比率。3回目の本番実行で得られた値であり、主張3'の
# Δ・標準偏差・閾値・判定のいずれにも使っていない(6.2.1節)。
_diagnostic_stats_existing_seeds = contrast_stats(
    condition_results["Q4_none"]["peak_to_mean"], condition_results["Q4_clip"]["peak_to_mean"]
)
print("[参考(判定には使用しない)] 既存シード0〜4による診断値:")
print(f"  Q4_none(平均)={_diagnostic_stats_existing_seeds['mean_a']:.3f}, "
      f"Q4_clip(平均)={_diagnostic_stats_existing_seeds['mean_b']:.3f}, "
      f"delta={_diagnostic_stats_existing_seeds['delta']:+.4f}")

if stats_3_prime is not None:
    _same_direction = (stats_3_prime["delta"] < 0) == (_diagnostic_stats_existing_seeds["delta"] < 0)
    print(f"\n新規シード5〜9による判定のdeltaの符号: {'負' if stats_3_prime['delta'] < 0 else '非負'}")
    print(f"既存シード0〜4による診断値のdeltaの符号: {'負' if _diagnostic_stats_existing_seeds['delta'] < 0 else '非負'}")
    print(f"符号が一致: {_same_direction}(判定そのものは新規シードのみに基づく。一致は結論の頑健性を示す参考情報)")
else:
    print("\n主張3'は前提不成立のため、新規シードとの符号比較は行わない")
```

    [参考(判定には使用しない)] 既存シード0〜4による診断値:
      Q4_none(平均)=8.706, Q4_clip(平均)=7.831, delta=-0.8748
    
    新規シード5〜9による判定のdeltaの符号: 負
    既存シード0〜4による診断値のdeltaの符号: 負
    符号が一致: True(判定そのものは新規シードのみに基づく。一致は結論の頑健性を示す参考情報)



```python
# 診断(参考、判定には使用しない): 既存シード0〜4によるQ4_none・Q4_warmup_cosineの
# 損失差分(loss_step_delta)系列の標準偏差。3回目の本番実行で得られた値であり、
# 主張4'のΔ・標準偏差・閾値・判定のいずれにも使っていない(6.2.2節)。
_diagnostic_loss_std_none = [
    compute_loss_step_delta_std(h["loss_step_delta"]) for h in condition_results["Q4_none"]["histories"]
]
_diagnostic_loss_std_warmup = [
    compute_loss_step_delta_std(h["loss_step_delta"]) for h in condition_results["Q4_warmup_cosine"]["histories"]
]
_diagnostic_stats_existing_seeds_4prime = contrast_stats(_diagnostic_loss_std_none, _diagnostic_loss_std_warmup)
print("[参考(判定には使用しない)] 既存シード0〜4による診断値:")
print(f"  Q4_none(平均)={_diagnostic_stats_existing_seeds_4prime['mean_a']:.4f}, "
      f"Q4_warmup_cosine(平均)={_diagnostic_stats_existing_seeds_4prime['mean_b']:.4f}, "
      f"delta={_diagnostic_stats_existing_seeds_4prime['delta']:+.4f}")

if stats_4_prime is not None:
    _same_direction_4prime = (
        (stats_4_prime["delta"] < 0) == (_diagnostic_stats_existing_seeds_4prime["delta"] < 0)
    )
    print(f"\n新規シード5〜9による判定のdeltaの符号: {'負' if stats_4_prime['delta'] < 0 else '非負'}")
    print(f"既存シード0〜4による診断値のdeltaの符号: "
          f"{'負' if _diagnostic_stats_existing_seeds_4prime['delta'] < 0 else '非負'}")
    print(f"符号が一致: {_same_direction_4prime}(判定そのものは新規シードのみに基づく。一致は結論の頑健性を示す参考情報)")
else:
    print("\n主張4'は前提不成立のため、新規シードとの符号比較は行わない")
```

    [参考(判定には使用しない)] 既存シード0〜4による診断値:
      Q4_none(平均)=0.0563, Q4_warmup_cosine(平均)=0.0359, delta=-0.0204
    
    新規シード5〜9による判定のdeltaの符号: 負
    既存シード0〜4による診断値のdeltaの符号: 負
    符号が一致: True(判定そのものは新規シードのみに基づく。一致は結論の頑健性を示す参考情報)


### 6.7 可視化

勾配ノルムの時系列(Q4 の 4 水準、gradient clipping 閾値を重畳表示)と、学習曲線(Q1〜Q4 の「なし」条件を比較)を描画する。



```python
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

_grad_norm_conditions = ["Q4_none", "Q4_warmup_cosine", "Q4_clip", "Q4_all"]
for _name in _grad_norm_conditions:
    _first_seed_history = condition_results[_name]["histories"][0]
    axes[0].plot(
        _first_seed_history["step"], _first_seed_history["gradient_norm"], alpha=0.8, linewidth=1.2, label=_name
    )
axes[0].axhline(GRADIENT_CLIP_THRESHOLD, color="tab:red", linestyle="--", linewidth=1.2, label="clip threshold")
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Gradient norm (pre-clip)")
axes[0].set_title("Q4: gradient norm trace by stabilization level (seed 0)")
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=8)

_none_conditions = ["Q1_none", "Q2_none", "Q3_none", "Q4_none"]
_curves_for_plot = {
    _name: [
        {"step": h["step"], "train_loss": h["train_loss"]}
        for h in condition_results[_name]["histories"]
    ]
    for _name in _none_conditions
}
plot_learning_curves_multi_seed(
    _curves_for_plot,
    step_key="step",
    value_key="train_loss",
    title="Learning curves: 'none' conditions (Q1-Q4)",
    ylabel="Train loss",
    ax=axes[1],
)
fig.tight_layout()
plt.show()

```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/007_training_stabilization/output_57_0.png)
    


### 6.8 主張 6: AdamW の合成タスク実験

3.2 節で導出した通り、AdamW は重み減衰の実効強度を 2 次モーメント推定から独立させる設計になっている。これを言語モデルの学習曲線からではなく、**勾配の典型的なスケールが異なる複数のパラメータ群を用意した合成タスク** で直接検証する。

**設計**: パラメータ群を 4 つ用意し(`SYNTH_GROUP_SIGMAS`、等比数列 $0.01, 0.1, 1.0, 10.0$)、各群 $i$ に標準偏差 $\sigma_i$ の平均 0 のガウス雑音を勾配として与える(真の学習信号を持たない、重み減衰のみを観測するための構成)。全群を同一の初期値 $\theta_0=2.0$ から開始し、同一の名目上の重み減衰係数 `WEIGHT_DECAY`で `SYNTH_STEPS` ステップ更新した後、群ごとの実効減衰率 $d_i = (\theta_i^{(0)} - \theta_i^{(T)}) / \theta_i^{(0)}$ と、その群間の乖離(`compute_effective_decay_divergence`)を計算する。この手続きを `SYNTH_SEEDS` 個の乱数シードで繰り返し、AdamW と Adam(L2 正則化混入)それぞれの乖離の分布を得る。



```python
SYNTH_GROUP_SIGMAS = (0.01, 0.1, 1.0, 10.0)
SYNTH_THETA0 = 2.0
SYNTH_LR = 1e-2


def run_synthetic_decay_experiment(optimizer_cls, seed: int) -> dict:
    torch.manual_seed(seed)
    params = [torch.nn.Parameter(torch.ones(1) * SYNTH_THETA0) for _ in SYNTH_GROUP_SIGMAS]
    initial_values = [p.item() for p in params]
    optimizer = optimizer_cls(params, lr=SYNTH_LR, weight_decay=WEIGHT_DECAY)
    for _ in range(SYNTH_STEPS):
        for p, sigma in zip(params, SYNTH_GROUP_SIGMAS, strict=True):
            p.grad = torch.randn(1) * sigma
        optimizer.step()
    final_values = [p.item() for p in params]
    return compute_effective_decay_divergence(initial_values, final_values)


adamw_divergences = []
l2_divergences = []
for _seed in range(SYNTH_SEEDS):
    adamw_divergences.append(run_synthetic_decay_experiment(AdamW, _seed)["divergence"])
    l2_divergences.append(run_synthetic_decay_experiment(AdamWithL2Regularization, _seed)["divergence"])

print(f"AdamW の群間乖離: 平均={np.mean(adamw_divergences):.4f}, 標準偏差={np.std(adamw_divergences, ddof=1):.4f}")
print(f"Adam(L2) の群間乖離: 平均={np.mean(l2_divergences):.4f}, 標準偏差={np.std(l2_divergences, ddof=1):.4f}")

verdict_6, stats_6 = _judge("claim_6", judge_claim_6, adamw_divergences, l2_divergences)
print(f"\n主張6(AdamWはL2正則化混入より群間の実効減衰強度の乖離が小さい): "
      f"delta={stats_6['delta']:+.4f}, threshold={stats_6['threshold']:.4f}, "
      f"判定={verdict_6}")

```

    AdamW の群間乖離: 平均=0.1372, 標準偏差=0.0725
    Adam(L2) の群間乖離: 平均=0.8682, 標準偏差=0.0711
    
    主張6(AdamWはL2正則化混入より群間の実効減衰強度の乖離が小さい): delta=+0.7311, threshold=0.0642, 判定=支持


### 6.9 判定結果一覧

以下は判定関数が正しく動作していることの動作確認としての一覧表である(このセルの出力がスモークテストと本番実行のどちらによるものかは `SMOKE_TEST` の値、5.2 節で判別できる)。前提条件が不成立の主張は delta・threshold を計算せず「前提不成立」と表示する。



```python
_verdict_table = [
    ("1", "正規化後置 > 正規化前置の不安定性", stats_1, verdict_1),
    ("2a", "正規化前置、学習率↑で不安定性↑", stats_2a, verdict_2a),
    ("2b", "正規化後置、学習率↑で不安定性↑", stats_2b, verdict_2b),
    ("3", "gradient clippingが損失の暴れを抑える", stats_3, verdict_3),
    ("3'", "gradient clippingが勾配ノルムのピーク/平均比率を抑える(新規シード)", stats_3_prime, verdict_3_prime),
    ("4", "warmup+cosineが損失の暴れを抑える", stats_4, verdict_4),
    ("4'", "warmup+cosineが損失差分系列の標準偏差を抑える(新規シード)", stats_4_prime, verdict_4_prime),
    ("5", "全部乗せは収束を遅らせない", stats_5, verdict_5),
    ("6", "AdamWはL2正則化混入より乖離が小さい", stats_6, verdict_6),
]
print(f"{'#':4s} {'主張':38s} {'delta':>10s} {'threshold':>10s} {'判定':>10s}")
for _num, _label, _stats, _verdict in _verdict_table:
    if _stats is None:
        print(f"{_num:4s} {_label:38s} {'':>10s} {'':>10s} {_verdict:>10s}")
    else:
        print(f"{_num:4s} {_label:38s} {_stats['delta']:10.4f} {_stats['threshold']:10.4f} {_verdict:>10s}")

```

    #    主張                                          delta  threshold         判定
    1    正規化後置 > 正規化前置の不安定性                         0.9406     1.0187       判定不能
    2a   正規化前置、学習率↑で不安定性↑                           1.7347     1.2065         支持
    2b   正規化後置、学習率↑で不安定性↑                           2.7306     1.3087         支持
    3    gradient clippingが損失の暴れを抑える               -0.0602     0.1199       判定不能
    3'   gradient clippingが勾配ノルムのピーク/平均比率を抑える(新規シード)    -1.8346     2.6899       判定不能
    4    warmup+cosineが損失の暴れを抑える                   -0.0403     0.1350       判定不能
    4'   warmup+cosineが損失差分系列の標準偏差を抑える(新規シード)      -0.0191     0.0020         支持
    5    全部乗せは収束を遅らせない                              0.1055     0.0442         反証
    6    AdamWはL2正則化混入より乖離が小さい                      0.7311     0.0642         支持


### 6.10 旧本番実行の結果(前提不成立の記録)

**本節はコードセルではなく Markdown セルとして記録する。** 旧実行(Google Colab T4 GPU での本番実行)の出力は、本ノートブックを較正手続きの修正のために再実行した時点で失われている。セルの出力を手で書き込まないという方針が禁じているのは **セルの outputs 領域に実行していない結果を注入すること** であり、Markdown セルの本文に過去の実行結果を記録として書くことはこの方針の対象外である(この区別はここに明記する)。以下の数値・判定は削除せず、実行のたびに追記していく形で保持する(1 回目: 6.10.1、2 回目: 6.10.2)。



## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/02_pretraining/007_training_stabilization.ipynb
