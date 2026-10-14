---
title: "Inverse Cloze Task と passage 単位の 2 段階検索 / Inverse Cloze Task and Two-Stage Retrieval at Passage Level(実装・実験編 7/8)"
---

この記事は後編(実装・実験編 7/8)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/026_inverse_cloze_task_and_two_stage_retrieval-practice-6)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/026_inverse_cloze_task_and_two_stage_retrieval-practice-8)。

### 6.9 実験 B: 記事単位の評価での、関連性の定義の効果


```python
_seeds = list(range(SEEDS_B))
R_B_R1, R_B_R2 = article_recall("R1", SEEDS_B), article_recall("R2", SEEDS_B)
B_PER_SEED = R_B_R1 - R_B_R2
_boot_b = contrast_bootstrap([RUNS[("R1", s)]["article_hits"] for s in _seeds] + [RUNS[("R2", s)]["article_hits"] for s in _seeds], EVALUATION_QUERY_ARTICLES, SEEDS_B)
DELTA_B = float(B_PER_SEED.mean())
SIGMA_B = combined_sigma(B_PER_SEED, _boot_b)
_p1_b = first_stage_learning_check(("R1", "R2"), _seeds)
precondition_status["P1(B)"] = _p1_b["ok"]
precondition_status["P2(B)"] = bool(P2_FLOOR <= R_B_R1.mean() <= P2_CEILING)
B_PRECONDITIONS = ["P0(B)", "P1(B)", "P2(B)"]
B_COMPUTED = judge(DELTA_B, SIGMA_B["sigma"])
B_VERDICT = B_COMPUTED if all(precondition_status[k] for k in B_PRECONDITIONS) else "前提不成立"

print(f"{RUN_TAG}実験 B(シード {_seeds}、T_1 = {FIRST_STAGE_STEPS}、評価用の query {len(EVALUATION_ENTRIES)} 個、索引 {NUM_INDEX:,} passage(実験 A と同じ)、ラベル: 記事単位。ランダムな順位の Recall@10 の期待値 {RANDOM_RECALL_ARTICLE_LABEL:.5f})")
print(f"  記事単位の Recall@10 R(R1): {rounded(R_B_R1)}、R(R2): {rounded(R_B_R2)}")
print(f"  d_s = R(R1) - R(R2): {rounded(B_PER_SEED)}")
print(
    f"  Delta_B = {DELTA_B:+.4f}、sigma_B = {SIGMA_B['sigma']:.4f}(シード間 {SIGMA_B['seed_term']:.4f}・ブートストラップ {SIGMA_B['bootstrap_term']:.4f}、反復 {BOOTSTRAP_RESAMPLES:,})、閾値 2 sigma_B = {SIGMA_MULTIPLIER * SIGMA_B['sigma']:.4f}"
)
print(
    f"  前提条件: P0 {precondition_status['P0(B)']}、P1 {precondition_status['P1(B)']}{describe_first_stage_learning(_p1_b)}、"
    f"P2 {precondition_status['P2(B)']}(R(R1) の平均 {R_B_R1.mean():.4f} が [{P2_FLOOR}, {P2_CEILING}])"
)
print(f"  判定関数の結果 {B_COMPUTED} -> {RUN_TAG}最終判定: {B_VERDICT}")
print("  診断量(判定なし):")
for _c in ("R1", "R2"):
    _runs = [RUNS[(_c, s)] for s in range(SEEDS_B)]
    print(
        f"    {_c}: 記事単位の平均逆順位 {np.mean([r['article_mean_reciprocal_rank'] for r in _runs]):.4f}、正規化割引累積利得(上位 10 件){np.mean([r['article_normalized_discounted_cumulative_gain'].mean() for r in _runs]):.4f}、"
        f"Recall@10 のシード間の標準偏差 {article_recall(_c, SEEDS_B).std(ddof=1) if SEEDS_B >= 2 else float('nan'):.4f}"
    )
print(f"    参照: 学習前の 008 の記事単位のラベルの Recall@10(評価用) {UNTRAINED_MAIN['article_hits'].mean():.4f}")
print(
    f"    診断量(判定なし): 024 と同じ形式の記事単位の評価(024 の評価用の索引と query、固定長)の Recall@10: R1 {format024_recall('R1', SEEDS_B).mean():.4f}、R2 {format024_recall('R2', SEEDS_B).mean():.4f}、"
    f"024 の Hub の重み {HUB_FORMAT_024_RECALL:.4f}、学習前の 008 {UNTRAINED_MAIN['format024_hits'].mean():.4f}。R1 と R2 の差(R1 - R2)のシードごと: {rounded(format024_recall('R1', SEEDS_B) - format024_recall('R2', SEEDS_B))}"
)

# 実験 A と B の判定の組み合わせの読み方(6.1 節で宣言した表。どちらかが前提不成立のときは適用しない)
COMBINATION_READING = {
    ("支持", "支持"): "交差: 各定義の検索器が、その定義のラベルの評価で優れる(同じ索引・同じ query・同じ入力の形式で、ラベルだけが違う)",
    ("支持", "判定不能"): "R2 の利点は passage 単位のラベルで示されたが、R1 の記事単位のラベルでの優位は示されない。交差とは言えない",
    ("支持", "反証"): "R2 が両方のラベルで優れる。定義の交差ではなく、passage 単位の定義の全面的な優位",
    ("判定不能", "支持"): "R1 の利点は記事単位のラベルで示されたが、R2 の passage 単位のラベルでの優位は示されない。交差とは言えない",
    ("判定不能", "判定不能"): "どちらのラベルでも差はノイズ床以内。この学習量・この設定で、定義の違いは検出されない",
    ("判定不能", "反証"): "R2 は記事単位のラベルでも優れ、passage 単位のラベルでは差がない",
    ("反証", "支持"): "R1 が両方のラベルで優れる。定義の交差ではなく、記事単位の定義の全面的な優位",
    ("反証", "判定不能"): "R1 は passage 単位のラベルで優れ、記事単位のラベルでは差がない",
    ("反証", "反証"): "逆の交差: 各定義の検索器が、その定義と反対のラベルの評価で優れる",
}
if "前提不成立" in (A_VERDICT, B_VERDICT):
    print(f"  A と B の組み合わせ: 実験 A {A_VERDICT}、実験 B {B_VERDICT} のため、組み合わせの読み方を適用しない")
else:
    print(f"  A と B の組み合わせ(実験 A {A_VERDICT}、実験 B {B_VERDICT}): {COMBINATION_READING[(A_VERDICT, B_VERDICT)]}")
    if not SETTINGS["calibrate_r1"]:
        print("  (段階 4 以降: R1 の学習率は規則で決めたので、実験 A の支持は、R1 の学習率が最適からずれた分の偏り(仮説に有利な向き)を含みうる。実験 B の反証・判定不能は、R1 に不利な向きの偏りを含みうる)")

_fig, _axes = plt.subplots(1, 2, figsize=(10, 3.8))
for _ax, _function, _title in ((_axes[0], passage_recall, "A: passage-label"), (_axes[1], article_recall, "B: article-label")):
    for _x, _c in enumerate(("R1", "R2")):
        _values = _function(_c, SEEDS_A)
        _ax.scatter([_x] * len(_values), _values, alpha=0.6)
        _ax.plot([_x - 0.2, _x + 0.2], [_values.mean()] * 2, "k-")
    _ax.set_xticks([0, 1])
    _ax.set_xticklabels(["R1", "R2"])
    _ax.set_ylabel("Recall@10")
    _ax.set_title(f"{PLOT_TAG}{_title}")
plt.tight_layout()
plt.show()
```

    実験 B(シード [0, 1, 2, 3, 4]、T_1 = 1024、評価用の query 3227 個、索引 93,995 passage(実験 A と同じ)、ラベル: 記事単位。ランダムな順位の Recall@10 の期待値 0.00112)
      記事単位の Recall@10 R(R1): [0.1546, 0.1518, 0.1481, 0.1518, 0.1528]、R(R2): [0.1661, 0.1726, 0.1624, 0.1695, 0.1726]
      d_s = R(R1) - R(R2): [-0.0115, -0.0208, -0.0143, -0.0177, -0.0198]
      Delta_B = -0.0168、sigma_B = 0.0036(シード間 0.0017・ブートストラップ 0.0032、反復 10,000)、閾値 2 sigma_B = 0.0073
      前提条件: P0 True、P1 True((a)の不成立 なし、最後の区間の訓練損失 / 同じ組での学習前のモデルの損失の最大 0.588 <= 0.9。(b)自分の定義の検証用の Recall@10 の学習前からの増分の最小 R1 +0.1398、R2 +0.1398 >= 0.04: True)、P2 True(R(R1) の平均 0.1518 が [0.02, 0.95])
      判定関数の結果 反証 -> 最終判定: 反証
      診断量(判定なし):
        R1: 記事単位の平均逆順位 0.0888、正規化割引累積利得(上位 10 件)0.0376、Recall@10 のシード間の標準偏差 0.0024
        R2: 記事単位の平均逆順位 0.1026、正規化割引累積利得(上位 10 件)0.0436、Recall@10 のシード間の標準偏差 0.0044
        参照: 学習前の 008 の記事単位のラベルの Recall@10(評価用) 0.1054
        診断量(判定なし): 024 と同じ形式の記事単位の評価(024 の評価用の索引と query、固定長)の Recall@10: R1 0.6125、R2 0.6363、024 の Hub の重み 0.6477、学習前の 008 0.4834。R1 と R2 の差(R1 - R2)のシードごと: [-0.0133, -0.0281, -0.0373, -0.0213, -0.0191]
      A と B の組み合わせ(実験 A 支持、実験 B 反証): R2 が両方のラベルで優れる。定義の交差ではなく、passage 単位の定義の全面的な優位



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/026_inverse_cloze_task_and_two_stage_retrieval/output_40_1.png)
    


### 6.10 実験 C: 2 段階の検索の効果(アプリの構成の判断)


```python
_seeds = list(range(SEEDS_C))
R_C_RERANK = np.array([CROSS_ENCODER_RUNS[s]["hits"].mean() for s in _seeds])
R_C_FIRST = np.array([CROSS_ENCODER_RUNS[s]["no_rerank_hits"].mean() for s in _seeds])
for _s in _seeds:  # R2 だけの hits は、実験 A の R2 の passage 単位の hits と一致する(同じ順位の定義)
    assert np.array_equal(CROSS_ENCODER_RUNS[_s]["no_rerank_hits"], RUNS[("R2", _s)]["passage_hits"][RERANK_EVALUATION_POSITIONS]) if _s < SEEDS_A else True
C_PER_SEED = R_C_RERANK - R_C_FIRST
_boot_c = contrast_bootstrap([CROSS_ENCODER_RUNS[s]["hits"] for s in _seeds] + [CROSS_ENCODER_RUNS[s]["no_rerank_hits"] for s in _seeds], RERANK_EVALUATION_ARTICLES, SEEDS_C)
DELTA_C = float(C_PER_SEED.mean())
SIGMA_C = combined_sigma(C_PER_SEED, _boot_c)
COVERAGE_C = float(np.mean([CROSS_ENCODER_RUNS[s]["coverage"] for s in _seeds]))
# P1: R2(実験 A・B と同じ第 1 段の P1 のうち R2 の分)と cross-encoder の P1((a)全シードの訓練損失、(b)検証用の query の並べ替えの Recall@10 の増分)
_p1_r2 = first_stage_learning_check(("R2",), _seeds)
_failed_a = [s for s in _seeds if cross_encoder_failure_reasons(CROSS_ENCODER_RUNS[s])]
_gain = bootstrap_rerank_gain([CROSS_ENCODER_RUNS[s] for s in _seeds])
_gain_ok = bool(_gain["gain"] > 0 and _gain["gain"] >= P1_SIGMA_MULTIPLIER * _gain["sigma"])
precondition_status["P1(C)"] = bool(_p1_r2["ok"] and not _failed_a and _gain_ok)
precondition_status["P2(C)"] = bool(R_C_FIRST.mean() <= COVERAGE_C - P2_ROOM)
precondition_status["P3(C)"] = bool(COVERAGE_C >= P3_MIN_COVERAGE and COVERAGE_C - R_C_FIRST.mean() >= P3_MIN_HEADROOM)
C_PRECONDITIONS = ["P0(C)", "P1(C)", "P2(C)", "P3(C)"]
C_COMPUTED = judge(DELTA_C, SIGMA_C["sigma"])
C_VERDICT = C_COMPUTED if all(precondition_status[k] for k in C_PRECONDITIONS) else "前提不成立"

print(f"{RUN_TAG}実験 C(シード {_seeds}、T_2 = {CROSS_ENCODER_STEPS}、K = {RERANK_CANDIDATES}、並べ替えの query {len(RERANK_EVALUATION_ENTRIES)} 個)")
print(f"  Recall@10 R(R3): {rounded(R_C_RERANK)}、R(R2 だけ): {rounded(R_C_FIRST)}")
print(f"  d_s = R(R3) - R(R2): {rounded(C_PER_SEED)}")
print(
    f"  Delta_C = {DELTA_C:+.4f}、sigma_C = {SIGMA_C['sigma']:.4f}(シード間 {SIGMA_C['seed_term']:.4f}・ブートストラップ {SIGMA_C['bootstrap_term']:.4f}、反復 {BOOTSTRAP_RESAMPLES:,})、閾値 2 sigma_C = {SIGMA_MULTIPLIER * SIGMA_C['sigma']:.4f}"
)
print(
    f"  前提条件: P0 {precondition_status['P0(C)']}、P1 {precondition_status['P1(C)']}"
    f"(R2 の P1 {_p1_r2['ok']}、cross-encoder の(a)の不成立 {_failed_a or 'なし'}、(b)検証用の query での並べ替えの Recall@10 の増分のシード平均 {_gain['gain']:+.4f}(シードごと {rounded(_gain['per_seed_gain'])})、"
    f"標準偏差 {_gain['sigma']:.4f}(シード間 {_gain['seed_term']:.4f}・ブートストラップ {_gain['bootstrap_term']:.4f})、増分 / 標準偏差 {_gain['gain'] / _gain['sigma'] if _gain['sigma'] > 0 else float('inf'):.1f}(閾値 {P1_SIGMA_MULTIPLIER}): {_gain_ok})、"
    f"P2 {precondition_status['P2(C)']}(R(R2 だけ) の平均 {R_C_FIRST.mean():.4f} <= coverage {COVERAGE_C:.4f} - {P2_ROOM})、"
    f"P3 {precondition_status['P3(C)']}(coverage {COVERAGE_C:.4f} >= {P3_MIN_COVERAGE}、coverage - R2 だけ {COVERAGE_C - R_C_FIRST.mean():.4f} >= {P3_MIN_HEADROOM})"
)
print(f"  判定関数の結果 {C_COMPUTED} -> {RUN_TAG}最終判定: {C_VERDICT}")
print("  診断量(判定なし):")
_mean_reciprocal_rank_first = np.array([float((1.0 / RUNS[("R2", s)]["passage_rank"][RERANK_EVALUATION_POSITIONS]).mean()) for s in _seeds if s < SEEDS_A])
print(
    f"    R2 だけ(索引全体の順位): Recall@10 {R_C_FIRST.mean():.4f}、平均逆順位 {_mean_reciprocal_rank_first.mean() if len(_mean_reciprocal_rank_first) else float('nan'):.4f}、coverage(上位 {RERANK_CANDIDATES} 件に正例が入る割合){COVERAGE_C:.4f}"
)
print(
    f"    R3(R2 + cross-encoder): Recall@10 {R_C_RERANK.mean():.4f}(シード間の標準偏差 {R_C_RERANK.std(ddof=1) if SEEDS_C >= 2 else float('nan'):.4f})、平均逆順位 {np.mean([CROSS_ENCODER_RUNS[s]['mean_reciprocal_rank'] for s in _seeds]):.4f}、"
    f"正規化割引累積利得(上位 10 件){np.mean([CROSS_ENCODER_RUNS[s]['normalized_discounted_cumulative_gain'] for s in _seeds]):.4f}、最後の区間の訓練損失 {np.mean([CROSS_ENCODER_RUNS[s]['final_train_loss'] for s in _seeds]):.3f}(一様な予測 {UNIFORM_LOSS_CROSS_ENCODER:.3f})"
)
print(f"    並べ替えによる Recall@10 の変化(追加の学習量と交絡するので、実用の問いとして読む): {DELTA_C:+.4f}。1 query あたりの順伝播の回数 {RERANK_CANDIDATES} 回(cross-encoder は組ごと)")
print("    候補 50 件のスコアの標準偏差の中央値(崩壊の検出。判定には使わない): " + rounded([CROSS_ENCODER_RUNS[s]["score_std_median"] for s in _seeds], 4).__repr__())
print(
    "    学習時と推論時の分布の違いの検出: 第 1 段 R2 の上位 50 件に正例が入る割合(coverage) シードごとの 学習用 / 検証用 / 評価用: "
    + "、".join(
        f"{s}: {usable_cloze_queries(MINED[s]['train'], TRAIN_ENTRIES).mean():.3f} / {usable_cloze_queries(MINED[s]['validation'], VALIDATION_ENTRIES).mean():.3f} / {usable_cloze_queries(MINED[s]['evaluation'], EVALUATION_ENTRIES).mean():.3f}"
        for s in _seeds
    )
)
print(
    f"    学習に使える query: " + "、".join(f"シード {s}: {CROSS_ENCODER_SCHEDULES[(s, CROSS_ENCODER_STEPS)].num_usable:,} 個(見た query {BATCH_QUERIES * CROSS_ENCODER_STEPS:,} 個、{BATCH_QUERIES * CROSS_ENCODER_STEPS / CROSS_ENCODER_SCHEDULES[(s, CROSS_ENCODER_STEPS)].num_usable:.2f} エポック)" for s in _seeds)
)

_fig, _ax = plt.subplots(figsize=(5.5, 4))
for _x, (_label, _values) in enumerate((("R2 only", R_C_FIRST), ("R2 + cross-encoder", R_C_RERANK))):
    _ax.scatter([_x] * len(_values), _values, alpha=0.6)
    _ax.plot([_x - 0.2, _x + 0.2], [_values.mean()] * 2, "k-")
_ax.axhline(COVERAGE_C, color="gray", linestyle="--", label=f"coverage (K = {RERANK_CANDIDATES})")
_ax.set_xticks([0, 1])
_ax.set_xticklabels(["R2 only", "R2 + cross-encoder"])
_ax.set_ylabel(f"Recall@10 (passage-level, top-{RERANK_CANDIDATES} reranked)")
_ax.set_title(f"{PLOT_TAG}Experiment C")
_ax.legend(fontsize=7)
plt.tight_layout()
plt.show()
```

    実験 C(シード [0, 1, 2, 3, 4]、T_2 = 2048、K = 50、並べ替えの query 3227 個)
      Recall@10 R(R3): [0.2268, 0.2135, 0.2197, 0.224, 0.2194]、R(R2 だけ): [0.1348, 0.1236, 0.132, 0.136, 0.1255]
      d_s = R(R3) - R(R2): [0.092, 0.0899, 0.0877, 0.088, 0.0939]
      Delta_C = +0.0903、sigma_C = 0.0042(シード間 0.0012・ブートストラップ 0.0040、反復 10,000)、閾値 2 sigma_C = 0.0084
      前提条件: P0 True、P1 True(R2 の P1 True、cross-encoder の(a)の不成立 なし、(b)検証用の query での並べ替えの Recall@10 の増分のシード平均 +0.1939(シードごと [0.1949, 0.2037, 0.1772, 0.2057, 0.188])、標準偏差 0.0157(シード間 0.0053・ブートストラップ 0.0148)、増分 / 標準偏差 12.3(閾値 2.0): True)、P2 True(R(R2 だけ) の平均 0.1304 <= coverage 0.2260 - 0.05)、P3 False(coverage 0.2260 >= 0.35、coverage - R2 だけ 0.0956 >= 0.1)
      判定関数の結果 支持 -> 最終判定: 前提不成立
      診断量(判定なし):
        R2 だけ(索引全体の順位): Recall@10 0.1304、平均逆順位 0.0740、coverage(上位 50 件に正例が入る割合)0.2260
        R3(R2 + cross-encoder): Recall@10 0.2207(シード間の標準偏差 0.0051)、平均逆順位 0.1406、正規化割引累積利得(上位 10 件)0.1602、最後の区間の訓練損失 0.348(一様な予測 2.079)
        並べ替えによる Recall@10 の変化(追加の学習量と交絡するので、実用の問いとして読む): +0.0903。1 query あたりの順伝播の回数 50 回(cross-encoder は組ごと)
        候補 50 件のスコアの標準偏差の中央値(崩壊の検出。判定には使わない): [3.6727, 3.464, 3.0315, 3.7827, 3.3261]
        学習時と推論時の分布の違いの検出: 第 1 段 R2 の上位 50 件に正例が入る割合(coverage) シードごとの 学習用 / 検証用 / 評価用: 0: 0.231 / 0.239 / 0.231、1: 0.230 / 0.248 / 0.222、2: 0.233 / 0.245 / 0.225、3: 0.233 / 0.240 / 0.228、4: 0.230 / 0.231 / 0.224
        学習に使える query: シード 0: 20,754 個(見た query 16,384 個、0.79 エポック)、シード 1: 20,653 個(見た query 16,384 個、0.79 エポック)、シード 2: 20,917 個(見た query 16,384 個、0.78 エポック)、シード 3: 20,878 個(見た query 16,384 個、0.78 エポック)、シード 4: 20,601 個(見た query 16,384 個、0.80 エポック)



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/026_inverse_cloze_task_and_two_stage_retrieval/output_42_1.png)
    


### 6.11 観察 D・E(判定なし)

**観察 D(2 段階の検索の費用と指標)**: R2 のシード 0 の索引の埋め込みの部分集合(件数 $N_{\mathrm{obs}}$。評価用の query の passage をすべて含み、残りは索引からランダムに選ぶ)で、第 1 段を全探索と HNSW(025 の実装、$M = 5$、$\mathrm{efConstruction} = 100$)にし、
上位 100 件の先頭 $K = 10, 20, 50, 100$ 件を、cross-encoder のシード 0 で並べ替える。HNSW の探索の幅は、全探索の上位 100 件との一致率が 0.9 に初めて届いた値(格子 $101, 127, 160, 202, 254$ のうち)。
費用は、1 query あたりの第 1 段の距離計算の回数と、第 2 段の順伝播の回数 $K$。関連性は部分集合の中の passage 単位(正解は query の passage だけ)で、Recall@10 の絶対値は実験 A・C(コーパス全体)と比べない。

**観察 E(検索結果の例)**: 評価用の query の並び順で、R1 と R2(シード 0)の一方だけが正例を上位 10 件に入れた query を、R2 だけが入れたもの 2 件、R1 だけが入れたもの 1 件選ぶ(決定的な規則)。


```python
_t0_obs = time.time()
assert ("R2", UPLOAD_SEED) in SAVED_STATES and ("cross_encoder", UPLOAD_SEED) in SAVED_STATES and "R2" in KEPT_EMBEDDINGS and "R1" in KEPT_EMBEDDINGS
_rng = np.random.default_rng(OBSERVATION_SEED)
_query_positions = select_evenly(len(EVALUATION_ENTRIES), LEVEL_SETTINGS["OBSERVATION_QUERIES"])
_query_entries = EVALUATION_ENTRIES[_query_positions]
_others = np.setdiff1d(np.arange(NUM_INDEX), _query_entries)
_extra = _rng.choice(_others, size=min(len(_others), max(OBSERVATION_N - len(_query_entries), 0)), replace=False)
OBSERVATION_SUBSET = np.sort(np.concatenate([_query_entries, _extra]))  # 部分集合の passage の添字(索引の中の番号。昇順)
_position_in_subset = np.searchsorted(OBSERVATION_SUBSET, _query_entries)
assert (OBSERVATION_SUBSET[_position_in_subset] == _query_entries).all() and len(np.unique(OBSERVATION_SUBSET)) == len(OBSERVATION_SUBSET)
_contexts = KEPT_EMBEDDINGS["R2"]["contexts"]
_queries = KEPT_EMBEDDINGS["R2"]["queries"][_query_positions]
_subset_vectors = _contexts[torch.from_numpy(OBSERVATION_SUBSET)]
_depth = max(OBSERVATION_RERANK_KS)

# 第 1 段: 全探索
_exact_local, _ = exact_search(_queries.to(device), _subset_vectors.to(device), _depth)
_exact_candidates = OBSERVATION_SUBSET[_exact_local]
# 第 1 段: HNSW
_start = time.time()
_hnsw = HierarchicalNavigableSmallWorld(_subset_vectors.numpy(), HNSW_MAX_CONNECTIONS, HNSW_CONSTRUCTION_WIDTH, seed=OBSERVATION_SEED)
_hnsw.add(len(OBSERVATION_SUBSET))
HNSW_BUILD_SECONDS = time.time() - _start
_hnsw_rows = []
for _width in OBSERVATION_SEARCH_WIDTHS:
    _ids, _costs = _hnsw.search_batch(_queries.numpy(), _depth, _width)
    _recall = float(np.mean([len(set(_ids[i].tolist()) & set(_exact_local[i].tolist())) / _depth for i in range(len(_ids))]))
    _hnsw_rows.append((_width, _recall, float(_costs.mean()), _ids))
    if _recall >= OBSERVATION_TARGET_RECALL:
        break
OBSERVATION_WIDTH, OBSERVATION_HNSW_RECALL, OBSERVATION_HNSW_COST, _hnsw_ids = _hnsw_rows[-1][0], _hnsw_rows[-1][1], _hnsw_rows[-1][2], _hnsw_rows[-1][3]
assert (_hnsw_ids >= 0).all(), "HNSW の候補が足りない query がある"
_hnsw_candidates = OBSERVATION_SUBSET[_hnsw_ids]

_model = build_cross_encoder(UPLOAD_SEED).to(device)
_model.load_state_dict({k: v.to(device) for k, v in SAVED_STATES[("cross_encoder", UPLOAD_SEED)].items()})
OBSERVATION_TABLE = {}
for _name, _candidates, _cost in (("exact search", _exact_candidates, float(len(OBSERVATION_SUBSET))), (f"HNSW (search width = {OBSERVATION_WIDTH})", _hnsw_candidates, OBSERVATION_HNSW_COST)):
    _scores = score_candidates_variable(_model, CLOZE_QUERY_TOKENS, CLOZE_QUERY_LENGTHS, CONTEXT_TOKENS, CONTEXT_LENGTHS, _query_entries, _candidates, SCORE_BATCH, USE_FP16_AUTOCAST)
    OBSERVATION_TABLE[_name] = {"cost": _cost}
    for _k in OBSERVATION_RERANK_KS:
        _after = rerank_single_positive(_scores[:, :_k], _candidates[:, :_k], _query_entries, NUM_INDEX)
        _before = rerank_single_positive(np.broadcast_to(-np.arange(_k, dtype=np.float64), (len(_query_entries), _k)), _candidates[:, :_k], _query_entries, NUM_INDEX)  # 第 1 段の順位のまま
        OBSERVATION_TABLE[_name][_k] = {"after": _after.recall_at(10), "before": _before.recall_at(10), "coverage": _after.coverage()}
del _model
empty_device_cache()
print(
    f"{RUN_TAG}観察 D(判定なし。部分集合 N = {len(OBSERVATION_SUBSET)}、query {len(_query_entries)} 個、R2 のシード 0 の埋め込み、cross-encoder のシード 0 で並べ替え、ランダムな順位の Recall@10 の期待値 {10 / len(OBSERVATION_SUBSET):.4f})"
)
print(
    f"  HNSW の構築 {HNSW_BUILD_SECONDS:.1f} 秒(M = {HNSW_MAX_CONNECTIONS}、efConstruction = {HNSW_CONSTRUCTION_WIDTH})。探索の幅の格子と全探索の上位 {_depth} 件との一致率: "
    + "、".join(f"ef = {w}: {r:.3f}" for w, r, _, _ in _hnsw_rows)
    + f"。使う探索の幅 {OBSERVATION_WIDTH}"
)
for _name, _row in OBSERVATION_TABLE.items():
    print(f"  第 1 段: {_name}、第 1 段の距離計算の回数(1 query あたり){_row['cost']:,.0f}")
    for _k in OBSERVATION_RERANK_KS:
        _m = _row[_k]
        print(f"    K = {_k:>3}: 第 2 段の順伝播 {_k} 回、coverage {_m['coverage']:.4f}、Recall@10: 並べ替えなし {_m['before']:.4f} -> cross-encoder で並べ替え {_m['after']:.4f}")

_fig, _ax = plt.subplots(figsize=(6.5, 4))
for _name in OBSERVATION_TABLE:
    _ax.plot(OBSERVATION_RERANK_KS, [OBSERVATION_TABLE[_name][_k]["after"] for _k in OBSERVATION_RERANK_KS], "o-", label=f"{_name}: reranked")
    _ax.plot(OBSERVATION_RERANK_KS, [OBSERVATION_TABLE[_name][_k]["before"] for _k in OBSERVATION_RERANK_KS], "x--", label=f"{_name}: first-stage order")
_ax.set_xscale("log")
_ax.set_xlabel("K (candidates reranked)")
_ax.set_ylabel("Recall@10 (passage-level, subset)")
_ax.set_title(f"{PLOT_TAG}Observation D (N = {len(OBSERVATION_SUBSET)})")
_ax.legend(fontsize=6)
plt.tight_layout()
plt.show()

# --- 観察 E: 検索結果の例 ---
_rank_r1, _rank_r2 = RUNS[("R1", 0)]["passage_rank"], RUNS[("R2", 0)]["passage_rank"]
_only_r2 = np.nonzero((_rank_r2 <= 10) & (_rank_r1 > 10))[0][:2]
_only_r1 = np.nonzero((_rank_r1 <= 10) & (_rank_r2 > 10))[0][:1]
_selected = list(_only_r2) + list(_only_r1)


def snippet(entry: int, limit: int = 160) -> str:
    return tokenizer.decode(CLOZE.context_tokens[entry, : CLOZE.context_lengths[entry]].tolist())[:limit].replace("\n", "\\n")


print(f"{RUN_TAG}観察 E(判定なし。評価用の query のうち、R2 だけが正例を上位 10 件に入れた {len(_only_r2)} 件、R1 だけが入れた {len(_only_r1)} 件。シード 0)")
for _position in _selected:
    _entry = int(EVALUATION_ENTRIES[_position])
    _query_text = tokenizer.decode(CLOZE.query_tokens[_entry, : CLOZE.query_lengths[_entry]].tolist()).replace("\n", "\\n")
    print(f"  query(抜いた文): {_query_text[:200]!r}  正例の passage: {snippet(_entry, 100)!r}")
    for _name in ("R1", "R2"):
        _scores = (KEPT_EMBEDDINGS[_name]["queries"][_position] @ KEPT_EMBEDDINGS[_name]["contexts"].T).numpy()
        _top = np.argsort(-_scores)[:3]
        _rank = (_rank_r1 if _name == "R1" else _rank_r2)[_position]
        print(f"    {_name}(正例の順位 {_rank}): " + " / ".join(f"{'[正例]' if int(t) == _entry else ('[同じ記事]' if ENTRY_ARTICLES[int(t)] == ENTRY_ARTICLES[_entry] else '')}{snippet(int(t), 60)!r}" for t in _top))
OBSERVATION_SECONDS = time.time() - _t0_obs
print(f"観察 D・E {OBSERVATION_SECONDS / 60:.1f} 分(見積もり {STAGE_ESTIMATES[STAGE]['observation'] / 60:.1f} 分)")
```

    観察 D(判定なし。部分集合 N = 16384、query 600 個、R2 のシード 0 の埋め込み、cross-encoder のシード 0 で並べ替え、ランダムな順位の Recall@10 の期待値 0.0006)
      HNSW の構築 53.8 秒(M = 5、efConstruction = 100)。探索の幅の格子と全探索の上位 100 件との一致率: ef = 101: 0.835、ef = 127: 0.872、ef = 160: 0.901。使う探索の幅 160
      第 1 段: exact search、第 1 段の距離計算の回数(1 query あたり)16,384
        K =  10: 第 2 段の順伝播 10 回、coverage 0.2200、Recall@10: 並べ替えなし 0.2200 -> cross-encoder で並べ替え 0.2200
        K =  20: 第 2 段の順伝播 20 回、coverage 0.2817、Recall@10: 並べ替えなし 0.2200 -> cross-encoder で並べ替え 0.2800
        K =  50: 第 2 段の順伝播 50 回、coverage 0.3733、Recall@10: 並べ替えなし 0.2200 -> cross-encoder で並べ替え 0.3667
        K = 100: 第 2 段の順伝播 100 回、coverage 0.4583、Recall@10: 並べ替えなし 0.2200 -> cross-encoder で並べ替え 0.4333
      第 1 段: HNSW (search width = 160)、第 1 段の距離計算の回数(1 query あたり)736
        K =  10: 第 2 段の順伝播 10 回、coverage 0.2117、Recall@10: 並べ替えなし 0.2117 -> cross-encoder で並べ替え 0.2117
        K =  20: 第 2 段の順伝播 20 回、coverage 0.2733、Recall@10: 並べ替えなし 0.2117 -> cross-encoder で並べ替え 0.2717
        K =  50: 第 2 段の順伝播 50 回、coverage 0.3650、Recall@10: 並べ替えなし 0.2117 -> cross-encoder で並べ替え 0.3567
        K = 100: 第 2 段の順伝播 100 回、coverage 0.4450、Recall@10: 並べ替えなし 0.2117 -> cross-encoder で並べ替え 0.4217



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/026_inverse_cloze_task_and_two_stage_retrieval/output_44_1.png)
    


    観察 E(判定なし。評価用の query のうち、R2 だけが正例を上位 10 件に入れた 2 件、R1 だけが入れた 1 件。シード 0)
      query(抜いた文): '\\nStarring Jodie Whittaker as the Thirteenth Doctor.'  正例の passage: '-ray from 2010 to 2014.\\n\\nTwelfth Doctor\\nStarring Peter Capaldi as the Twelfth Doctor. All Twelfth Do'
        R1(正例の順位 154): '\\nStephen Lobo as Jim Corrigan\\nTom Ellis as Lucifer Morningst' / "\\nReu and Harper's Wonder World\\nRoarsome Dinosaurs\\nThe Secret" / ' Panda Patrol\\nJourney to the Heart of World\\nThe Kids from Ro'
        R2(正例の順位 2): 'ittle Helper\\nSeed of Chucky\\nShaun of the Dead\\nTo Catch a Vir' / [正例]'-ray from 2010 to 2014.\\n\\nTwelfth Doctor\\nStarring Peter Capal' / ' serving Canary Wharf tube station are officially opened.\\nTh'
      query(抜いた文): ' It was released on DVD in Region 2 on 12 November 2007, Region 4 on 5 June 2008 and Region 1 on 18 November 2008.'  正例の passage: ' end.\\nDreamland is a six-part animated serial (1 × 12 min., 5 × 6 min.). They were broadcast daily o'
        R1(正例の順位 33): ' announced on 5 June 2023. The final 23-player squad was ann' / 'land announced their 15-member squad on 5 January 2015. The ' / 'quad was announced on 31 May.\\n\\nPoland\\nCoach: Adam Nawałka\\n\\nP'
        R2(正例の順位 1): [正例]' end.\\nDreamland is a six-part animated serial (1 × 12 min., ' / ' Carraro was also chairman of Sofipa from 28 January 2000 to' / ' March 2021. It existed continuously from 25 June 2013 to 13'
      query(抜いた文): ' A limited edition Blu-ray box-set containing Series 1-7 of Doctor Who  was released 5 November 2013.'  正例の passage: ' 2013 with exclusive bonus features. The 1996 telemovie received a Blu-ray release on 19 September 2'
        R1(正例の順位 1): [正例]' 2013 with exclusive bonus features. The 1996 telemovie rece' / "DV's financial troubles.\\n\\nA DVD set from New Video, containi" / ' The release also includes 45 minutes of deleted scenes and '
        R2(正例の順位 163): " the band rose to number two on Spotify Mexico's Daily Top A" / ' on the Billboard Hot 100 Chart, number 23 on the Billboard ' / ' The release also includes 45 minutes of deleted scenes and '
    観察 D・E 1.6 分(見積もり 2.4 分)


### 6.12 不変条件のアサーションと`SMOKE_TEST`の配線

- 実効水準の照合: 実際の学習の長さ・シード数・ブートストラップの反復回数が、5.2 節と 6.4 節で印字した値と一致する。
- 同じシードの R1 と R2 の記事の並びが一致する(6.6 節)。R2 の組はシードごとに違う。評価の順位の長さが評価用の query の数と一致する。
- 総パラメータ数が R1 と R2 で同じで、cross-encoder は本体のパラメータ数 + ヘッド(257 個)である。
- 保持した埋め込み(R2 のシード 0)から順位を再計算すると、記録した順位と一致する。第 1 段の上位 $K$ 件の先頭が、全探索の最上位と一致する。
- アップロード用に保持した重み(R2 と cross-encoder のシード 0)が存在し、トークナイザなど同梱しないものを含まない。


```python
# --- 実効水準の照合(SMOKE_TEST の配線)---
assert CURRENT_LEVEL_NAME == ("smoke" if SMOKE_TEST else "prod")
for (_c, _s), _r in RUNS.items():
    assert _r["num_steps"] == FIRST_STAGE_STEPS == len(_r["train_loss"]) == LEVELS[CURRENT_LEVEL_NAME]["FIRST_STAGE_STEPS"], (_c, _s)
    assert len(_r["passage_rank"]) == len(_r["article_rank"]) == len(EVALUATION_ENTRIES) and len(_r["format024_hits"]) == EVALUATION_SET.num_queries
assert {s for (_c, s) in RUNS} == set(range(FIRST_STAGE_SEEDS)) and set(CROSS_ENCODER_RUNS) == set(range(CROSS_ENCODER_SEEDS))
for _s, _r in CROSS_ENCODER_RUNS.items():
    assert _r["num_steps"] == CROSS_ENCODER_STEPS == len(_r["train_loss"]) and CROSS_ENCODER_STEPS in LEVELS[CURRENT_LEVEL_NAME]["CROSS_ENCODER_STEP_CANDIDATES"], _s
    assert len(_r["hits"]) == len(RERANK_EVALUATION_ENTRIES)
assert BOOTSTRAP_RESAMPLES == LEVELS[CURRENT_LEVEL_NAME]["BOOTSTRAP_RESAMPLES"] and _boot_a.shape == (BOOTSTRAP_RESAMPLES,)
assert CROSS_ENCODER_SEEDS <= FIRST_STAGE_SEEDS and set(SETTINGS) == {"observation_n", "cross_encoder_seeds", "first_stage_seeds", "calibrate_r1", "calibrate_ce"}
assert SETTINGS == stage_settings(STAGE) and OBSERVATION_N == LEVELS[CURRENT_LEVEL_NAME]["OBSERVATION_N_CANDIDATES"][1 if STAGE >= 1 else 0]
print(
    f"実効水準: 水準 {CURRENT_LEVEL_NAME!r}、T_1 = {FIRST_STAGE_STEPS}(全 {len(RUNS)} 回の第 1 段の学習で一致)、T_2 = {CROSS_ENCODER_STEPS}(全 {len(CROSS_ENCODER_RUNS)} 回の cross-encoder の学習で一致)、"
    f"第 1 段のシード数 {FIRST_STAGE_SEEDS}・cross-encoder のシード数 {CROSS_ENCODER_SEEDS}、ブートストラップ {BOOTSTRAP_RESAMPLES:,} 回、計画 {STAGE}: OK"
)

# --- 記事の並び・組・パラメータ数 ---
for _s in range(FIRST_STAGE_SEEDS):
    assert RUNS[("R1", _s)]["article_hash"] == RUNS[("R2", _s)]["article_hash"], _s
assert len({RUNS[("R2", s)]["schedule_hash"] for s in range(FIRST_STAGE_SEEDS)}) == FIRST_STAGE_SEEDS and len({RUNS[("R1", s)]["schedule_hash"] for s in range(FIRST_STAGE_SEEDS)}) == FIRST_STAGE_SEEDS
assert RUNS[("R1", 0)]["schedule_hash"] != RUNS[("R2", 0)]["schedule_hash"], "R1 と R2 の組が同じ(介入がない)"
# R1 と R2 の組の関係: 同じバッチの query と passage のトークンの集合は同じで、正例の対応(R1: 同じ記事のもう一方の組の窓)だけが違う。R1 だけが同じ記事の組を負例から除く
_q2, _ql2, _c2, _cl2 = pair_arrays("R2", 0, 8)
_q1, _ql1, _c1, _cl1 = pair_arrays("R1", 0, 8)
_swap = np.arange(BATCH_SIZE) ^ 1
assert np.array_equal(_q1, _q2) and np.array_equal(_ql1, _ql2) and np.array_equal(_c1, _c2[:, _swap]) and np.array_equal(_cl1, _cl2[:, _swap]), "R1 と R2 の組の対応が設計と違う"
_articles_check = pair_batches(0, 8).articles
assert (_articles_check[:, _swap] == _articles_check).all(), "入れ替える相手が同じ記事でない"
_batches_r1, _batches_r2 = first_stage_batches("R1", 0, 8), first_stage_batches("R2", 0, 8)
assert _batches_r1.mask_same_article and not _batches_r2.mask_same_article
assert (_batches_r1.negatives_per_row() == BATCH_SIZE - 2).all() and (_batches_r2.negatives_per_row() == BATCH_SIZE - 1).all()
assert int(_batches_r1.ignore(0).sum()) == BATCH_SIZE and bool((_batches_r1.ignore(0).diagonal() == 0).all())
NEGATIVES_PER_ROW = {"R1": float(_batches_r1.negatives_per_row().mean()), "R2": float(_batches_r2.negatives_per_row().mean())}
del _batches_r1, _batches_r2
_base_parameters = sum(p.numel() for p in build_embedding_model().parameters())
_cross_encoder_parameters = sum(p.numel() for p in build_cross_encoder(0).parameters())
assert _cross_encoder_parameters == _base_parameters + EMBEDDING_DIMENSION + 1, (_cross_encoder_parameters, _base_parameters)
print(f"記事の並び: 同じシードの R1 と R2 で一致({FIRST_STAGE_SEEDS} シード)、組はシードごと・条件ごとに違う。R1 の passage は R2 の文脈を同じ記事の組の間で入れ替えたもの(query は同じ)で、R1 だけが同じ記事の組を負例から除く(1 行あたりの負例の数: R1 {NEGATIVES_PER_ROW['R1']:.0f}、R2 {NEGATIVES_PER_ROW['R2']:.0f})。総パラメータ数: R1・R2 {_base_parameters:,}(同じ構造)、cross-encoder {_cross_encoder_parameters:,}(本体 + ヘッド {EMBEDDING_DIMENSION + 1}): OK")

# --- 保持した埋め込みからの再計算・第 1 段の候補 ---
_kept = KEPT_EMBEDDINGS["R2"]
_recomputed = rank_single_positive(_kept["queries"].to(device), _kept["contexts"].to(device), EVALUATION_ENTRIES)
assert np.array_equal(_recomputed, RUNS[("R2", UPLOAD_SEED)]["passage_rank"]), "保持した埋め込みから再計算した順位が、記録した順位と一致しない"
for _s in sorted(MINED):
    for _name in ("train", "validation", "evaluation"):
        assert (MINED[_s][_name] >= 0).all() and MINED[_s][_name].shape[1] == RERANK_CANDIDATES
assert (MINED[UPLOAD_SEED]["evaluation"][:, 0] == exact_search(_kept["queries"].to(device), _kept["contexts"].to(device), 1)[0][:, 0]).all(), "第 1 段の上位 K 件の先頭が全探索の最上位と一致しない"
print("保持した埋め込み: 再計算した順位が記録した順位と一致(R2 のシード 0)、第 1 段の上位 K 件の先頭が全探索の最上位と一致: OK")
assert set(SAVED_STATES) == {("R2", UPLOAD_SEED), ("cross_encoder", UPLOAD_SEED)}, set(SAVED_STATES)
print(f"アップロード用に保持した重み: {sorted(SAVED_STATES)}(トークナイザなどは保持しない): OK")
TOTAL_ELAPSED_SECONDS = time.time() - NOTEBOOK_START_TIME
print(f"ここまでの経過時間 {TOTAL_ELAPSED_SECONDS / 60:.1f} 分(予算 {SESSION_BUDGET_SECONDS / 60:.0f} 分、選択時の見積もりの終了 {(ELAPSED_AT_SELECTION + STAGE_ESTIMATES[STAGE]['total']) / 60:.1f} 分)")
```

    実効水準: 水準 'prod'、T_1 = 1024(全 10 回の第 1 段の学習で一致)、T_2 = 2048(全 5 回の cross-encoder の学習で一致)、第 1 段のシード数 5・cross-encoder のシード数 5、ブートストラップ 10,000 回、計画 0: OK
    記事の並び: 同じシードの R1 と R2 で一致(5 シード)、組はシードごと・条件ごとに違う。R1 の passage は R2 の文脈を同じ記事の組の間で入れ替えたもの(query は同じ)で、R1 だけが同じ記事の組を負例から除く(1 行あたりの負例の数: R1 62、R2 63)。総パラメータ数: R1・R2 5,246,208(同じ構造)、cross-encoder 5,246,465(本体 + ヘッド 257): OK
    保持した埋め込み: 再計算した順位が記録した順位と一致(R2 のシード 0)、第 1 段の上位 K 件の先頭が全探索の最上位と一致: OK
    アップロード用に保持した重み: [('R2', 0), ('cross_encoder', 0)](トークナイザなどは保持しない): OK
    ここまでの経過時間 73.8 分(予算 120 分、選択時の見積もりの終了 86.7 分)


### 6.13 判定結果の一覧


```python
print(f"{RUN_TAG}判定結果(計画 {STAGE}: {STAGE_CHANGES[STAGE]}、T_1 = {FIRST_STAGE_STEPS}、T_2 = {CROSS_ENCODER_STEPS})")
for _name, _delta, _sigma, _computed, _verdict, _keys in (
    ("A", DELTA_A, SIGMA_A, A_COMPUTED, A_VERDICT, A_PRECONDITIONS),
    ("B", DELTA_B, SIGMA_B, B_COMPUTED, B_VERDICT, B_PRECONDITIONS),
    ("C", DELTA_C, SIGMA_C, C_COMPUTED, C_VERDICT, C_PRECONDITIONS),
):
    print(
        f"  実験 {_name}: 対比量 {_delta:+.4f}、標準偏差 {_sigma['sigma']:.4f}、閾値 {SIGMA_MULTIPLIER * _sigma['sigma']:.4f}、前提条件 "
        + "、".join(f"{k} {precondition_status[k]}" for k in _keys)
        + f"、判定関数の結果 {_computed}、最終判定 {_verdict}"
    )
print(f"  学習率: {json.dumps({c: float(f'{v:.3g}') for c, v in LEARNING_RATE.items()})}、出どころ {json.dumps(LEARNING_RATE_SOURCE, ensure_ascii=False)}")
```

    判定結果(計画 0: (削らない)、T_1 = 1024、T_2 = 2048)
      実験 A: 対比量 +0.0324、標準偏差 0.0041、閾値 0.0082、前提条件 P0(A) True、P1(A) True、P2(A) True、判定関数の結果 支持、最終判定 支持
      実験 B: 対比量 -0.0168、標準偏差 0.0036、閾値 0.0073、前提条件 P0(B) True、P1(B) True、P2(B) True、判定関数の結果 反証、最終判定 反証
      実験 C: 対比量 +0.0903、標準偏差 0.0042、閾値 0.0084、前提条件 P0(C) True、P1(C) True、P2(C) True、P3(C) False、判定関数の結果 支持、最終判定 前提不成立
      学習率: {"R2": 0.00048, "R1": 0.00048, "cross_encoder": 0.00096}、出どころ {"R2": "較正", "R1": "較正", "cross_encoder": "較正"}


### 6.14 Hugging Face Hub へのアップロード(R2 のシード 0・cross-encoder のシード 0)

後続のアプリ(`apps/`)の入力にする 2 つのモデルを、`UPLOAD_ARTIFACTS`が`True`のときだけアップロードする。アップロードの前に、書き出した重みを新しいモデルに読み込み直して評価し、モデルカードの指標にはその値を使う。
**公開の条件(判定ではない)** を満たさない場合はアップロードしない。

- R2 のシード 0 → `kojikojiprg/ai-theories-text-embedding-en`の新しいブランチ`inverse-cloze-task`(`main`の 024 のモデルと同じ構造なので、同じリポジトリのブランチにする)。公開の条件: 読み込み直した重みの passage 単位の評価での Recall@10 が、R1 のシード 0 の値を上回ること。
- cross-encoder のシード 0 → 新しいリポジトリ`kojikojiprg/ai-theories-cross-encoder-en`。公開の条件: 読み込み直した重みで、R2 のシード 0 の上位 $K$ 件を並べ替えた Recall@10 が、並べ替えなしの値を上回ること(評価用の query)。

`HF_TOKEN`は Colab Secrets から取得する(取得できなければ、停止せずスキップする)。トークンの値は印字・記録しない。アップロードの後、`list_repo_files()`で意図したファイルが存在することを確認する。


```python
_t0_upload = time.time()
import tempfile  # noqa: E402

EMBEDDING_CARD_TEMPLATE = '''---
language: en
license: mit
tags:
- ai-theories
- text-embedding
- retrieval
- inverse-cloze-task
- matryoshka-representation-learning
- scratch-implementation
---

# ai-theories テキスト埋め込みモデル(英語、Inverse Cloze Task、`{branch}` ブランチ)

`ai-theories`(https://github.com/kojikojiprg/ai-theories)プロジェクトの成果物。`main` ブランチ(024 の記事単位の定義で学習したモデル)と **同じ構造** で、**学習の組の作り方が違う** モデルである。
008 の小型 GPT(`kojikojiprg/ai-theories-small-gpt-en`)を encoder として、Inverse Cloze Task(passage から 1 文を抜いて query にし、その文を除いた残りを正例の文脈にする)と
Matryoshka Representation Learning(入れ子の次元 {matryoshka_dimensions})で、全パラメータを更新したもの。スクラッチ実装であり、研究・教育目的のモデルである。品質保証は行っていない。商用・実運用での利用は想定しない。

## 由来

- [026. Inverse Cloze Task と passage 単位の 2 段階検索](https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/026_inverse_cloze_task_and_two_stage_retrieval.ipynb)の
  条件 R2 のシード 0 の学習後の重み。結果を見て選んだものではなく、事前に固定した条件とシードである。
- 学習: {num_steps} ステップ、バッチ {batch_size} 組、温度 {temperature}、学習率 {learning_rate:.3g}、AdamW(重み減衰 {weight_decay})、warmup + cosine、gradient clipping {clip}。
  1 バッチは {articles_per_batch} 記事 x 2 組で、同じ記事の組を意図的に同じバッチに入れる(同じ記事の組の passage も負例として数える)。
  Inverse Cloze Task: 長さ {passage_length} トークンの窓から、長さ {min_query}〜{max_removed} トークンの文を 1 つ選んで query(先頭 {query_length} トークンまで)にし、残りを文脈にする。確率 {keep_probability} で文を文脈に残す。
  学習データは `kojikojiprg/ai-theories-corpus-en` の、先頭の 356 記事を除く記事のうち学習用に分けた {num_train_articles} 記事のみ(検証用・評価用の記事は含まない)。

## 使うために必要な他のアーティファクト

- **トークナイザは同梱していない。** `kojikojiprg/ai-theories-tokenizer-en` を使用してください(語彙サイズ 8192、特殊トークンなし)。
- 本体の構成は `config.json` を参照。

## 埋め込みの作り方

`main` ブランチと同じ(因果マスク、平均プール、L2 正規化、先頭 m 次元の切り詰めと再正規化)。**違い**: 学習で見た入力が可変長である(query は抜いた文で 4〜{query_length} トークン、文脈は 32〜{passage_length} トークン)。
可変長の入力は右側をパディングして長さを渡す(因果マスクのもとでは、右側のパディングは手前の位置の出力に影響しない。平均プールはパディングを除く)。

```python
import json
import torch
from huggingface_hub import hf_hub_download
from src.models.text_embedding import TextEmbeddingModel
from src.models.variable_length import encode_tokens_with_lengths

config = json.load(open(hf_hub_download("{repo_id}", "config.json", revision="{branch}")))
# GPTLanguageModel の構築は 008(kojikojiprg/ai-theories-small-gpt-en)の config.json と同じ(026 のノートブックの build_gpt() を参照)
backbone = build_gpt(config)
backbone.load_state_dict(torch.load(hf_hub_download("{repo_id}", "model_state.pt", revision="{branch}"), map_location="cpu"))
model = TextEmbeddingModel(backbone, pooling="mean", attention="causal").eval()
embeddings = encode_tokens_with_lengths(model, padded_token_ids, lengths, batch_size=256)[256]  # 右側をパディングしたトークンと有効な長さ
```

## 評価(アップロードした重みそのものを読み込み直して測った値)

評価用の記事(300 記事)の passage から作った query({num_queries} 個。passage から抜いた文)を、コーパス全体の索引({num_index} passage。すべて「1 文を除いた形」)に対して、全件の内積で検索した。正例は query の passage だけ。

| 評価 | このモデル(R2 のシード 0) | R1 のシード 0(記事単位の定義、`main` と同じ構造) |
|---|---|---|
| passage 単位の Recall@10 | {recall_passage:.4f} | {recall_passage_r1:.4f} |
| 記事単位のラベルの Recall@10(同じ索引・同じ query。正解は同じ記事の他の passage) | {recall_article:.4f} | {recall_article_r1:.4f} |
| 024 と同じ形式の記事単位の評価の Recall@10(024 の評価用の索引と query) | {recall_format024:.4f} | {recall_format024_r1:.4f} |

- ランダムな順位の passage 単位の Recall@10 の期待値: {random_recall:.6f}。
- 上の値は {smoke_note}

## 注意

- 学習データは英語 Wikipedia の長大な記事(Special:LongPages の長い順に選んだ記事)に限られ、汎用の検索器として使えるものではない。
- passage 単位の評価の索引と query は、ノートブックの実行時点のコーパス(`article_offsets` を含む `metadata.json`)に依存する。
'''

CROSS_ENCODER_CARD_TEMPLATE = '''---
language: en
license: mit
tags:
- ai-theories
- cross-encoder
- reranking
- retrieval
- scratch-implementation
---

# ai-theories cross-encoder(英語、passage 単位の並べ替えモデル)

`ai-theories`(https://github.com/kojikojiprg/ai-theories)プロジェクトの成果物。008 の小型 GPT(`kojikojiprg/ai-theories-small-gpt-en`)を起点に、query と passage を連結して符号化し、
終端の位置の出力から並べ替えのスコアを出す cross-encoder として、全パラメータを更新して学習したもの。第 1 段の検索器(`kojikojiprg/ai-theories-text-embedding-en` の `inverse-cloze-task` ブランチ)の
上位 {rerank_candidates} 件から採掘した正例・負例で学習した。スクラッチ実装であり、研究・教育目的のモデルである。品質保証は行っていない。商用・実運用での利用は想定しない。

## 由来

- [026. Inverse Cloze Task と passage 単位の 2 段階検索](https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/026_inverse_cloze_task_and_two_stage_retrieval.ipynb)の
  cross-encoder のシード 0 の学習後の重み。結果を見て選んだものではなく、事前に固定したシードである。
- 学習: {num_steps} ステップ、1 ステップ {batch_queries} query x(正例 1 + 負例 {num_negatives})、学習率 {learning_rate:.3g}、AdamW(重み減衰 {weight_decay})、warmup + cosine、gradient clipping {clip}。
  損失は 1 個の正例と {num_negatives} 個の負例の softmax 交差エントロピー。query は passage から抜いた文(先頭 {query_length} トークンまで)、正例はその文を除いた passage、負例は第 1 段の上位 {rerank_candidates} 件から正例を除いて選んだ passage。
  学習データは `kojikojiprg/ai-theories-corpus-en` の、先頭の 356 記事を除く記事のうち学習用に分けた {num_train_articles} 記事のみ(検証用・評価用の記事は含まない)。

## 使うために必要な他のアーティファクト

- **トークナイザは同梱していない。** `kojikojiprg/ai-theories-tokenizer-en` を使用してください(語彙サイズ 8192、特殊トークンなし)。
- **第 1 段の検索器は同梱していない。** 並べ替える候補は、`kojikojiprg/ai-theories-text-embedding-en` の `inverse-cloze-task` ブランチの全探索または近似最近傍探索の上位 {rerank_candidates} 件を想定している。
- 本体の構成は `config.json` を参照。

## 入力とスコア

入力の列は `[query のトークン] [区切り] [passage のトークン] [終端]` を **左から詰めて** 並べたもの(右側をパディング。長さの配列を渡す)。008 のトークナイザには特殊トークンがないので、語彙を増やさず、
コーパスに一度も現れない単一バイトのトークンを使う: 区切りは 0x1F(ID {separator_id})、終端は 0x1E(ID {terminal_id})。スコアは、因果マスクのもとで query 全体と passage 全体を見ている **終端の位置** の、
最終正規化層の後の隠れ状態に、線形の層(`head`)を掛けたスカラーである。

```python
import json
import torch
from huggingface_hub import hf_hub_download
from src.models.variable_length import VariableLengthCrossEncoder

config = json.load(open(hf_hub_download("{repo_id}", "config.json")))
backbone = build_gpt(config)  # 026 のノートブックの build_gpt() を参照
model = VariableLengthCrossEncoder(backbone, config["separator_token_id"], config["terminal_token_id"])
model.load_state_dict(torch.load(hf_hub_download("{repo_id}", "model_state.pt"), map_location="cpu"))
model.eval()
with torch.no_grad():
    scores = model(query_tokens, passage_tokens, query_lengths, passage_lengths)  # 形状 (B,)。トークンは右側をパディング
```

## 評価(アップロードした重みそのものを読み込み直して測った値)

評価用の記事の passage から作った query({num_queries} 個)について、コーパス全体({num_index} passage)に対する第 1 段(R2 のシード 0 の全探索)の上位 {rerank_candidates} 件を並べ替えて、Recall@10 を測った(正例は query の passage だけ)。

| 指標 | 並べ替えなし(第 1 段の順位) | このモデルで並べ替え |
|---|---|---|
| Recall@10 | {first_stage_recall:.4f} | {recall:.4f} |
| 平均逆順位(Mean Reciprocal Rank) | {first_stage_mrr:.4f} | {mrr:.4f} |

- 並べ替えで到達できる Recall@10 の上限(上位 {rerank_candidates} 件に正例が入る query の割合): {coverage:.4f}。
- 上の値は {smoke_note}



## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/026_inverse_cloze_task_and_two_stage_retrieval.ipynb
