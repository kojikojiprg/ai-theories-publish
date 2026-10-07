---
title: "ANN 検索とリランキング / ANN Search and Reranking(実装・実験編 7/8)"
---

この記事は後編(実装・実験編 7/8)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/025_ann_search_and_reranking-practice-6)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/025_ann_search_and_reranking-practice-8)。

### 6.12 実験 E: 困難な負例の効果


```python
_seeds_e = list(range(SEEDS_E))
R_E1, R_E2 = recall_of("D1", SEEDS_E), recall_of("E2", SEEDS_E)  # E1 は実験 D の D1(同じ学習)
E_PER_SEED = R_E1 - R_E2
_hit_difference_e = hits_of("D1", SEEDS_E) - hits_of("E2", SEEDS_E)
_boot_e = ((RERANK_BOOTSTRAP_WEIGHTS @ _hit_difference_e.T) / RERANK_BOOTSTRAP_WEIGHTS.sum(axis=1, keepdims=True)).mean(axis=1)
DELTA_E = float(E_PER_SEED.mean())
SIGMA_E = combined_sigma(E_PER_SEED, _boot_e)
_p1_e = check_learning(("D1", "E2"), "E2", SEEDS_E)
precondition_status["P1(E)"] = _p1_e["ok"]
precondition_status["P2(E)"] = bool(R_E2.mean() <= COVERAGE - P2_ROOM)
precondition_status["P3(E)"] = precondition_first_stage()
E_PRECONDITIONS = ["P0(E)", "P1(E)", "P2(E)", "P3(E)"]
E_COMPUTED = judge(DELTA_E, SIGMA_E["sigma"])
E_VERDICT = E_COMPUTED if all(precondition_status[k] for k in E_PRECONDITIONS) else "前提不成立"

print(f"{RUN_TAG}実験 E(シード {_seeds_e}、T = {NUM_STEPS}、K = {RERANK_CANDIDATES}、並べ替えの query {len(RERANK_QUERY_INDEX)} 個。E1 = D1)")
print(f"  並べ替えの Recall@10 R(E1): {rounded(R_E1)}、R(E2): {rounded(R_E2)}")
print(f"  d_s = R(E1) - R(E2): {rounded(E_PER_SEED)}")
print(
    f"  Delta_E = {DELTA_E:+.4f}、sigma_E = {SIGMA_E['sigma']:.4f}(シード間 {SIGMA_E['seed_term']:.4f}・ブートストラップ {SIGMA_E['bootstrap_term']:.4f}、反復 {BOOTSTRAP_RESAMPLES:,})、"
    f"閾値 2 sigma_E = {SIGMA_MULTIPLIER * SIGMA_E['sigma']:.4f}"
)
print(
    f"  前提条件: P0 {precondition_status['P0(E)']}、P1 {precondition_status['P1(E)']}{describe_learning(_p1_e)}、"
    f"P2 {precondition_status['P2(E)']}(R(E2) の平均 {R_E2.mean():.4f} <= coverage {COVERAGE:.4f} - {P2_ROOM})、P3 {precondition_status['P3(E)']}(実験 D と同じ)"
)
print(f"  判定関数の結果 {E_COMPUTED} -> {RUN_TAG}最終判定: {E_VERDICT}")
print("  診断量(判定なし):")
for _c in ("D1", "E2", "D2"):
    _rc = [RUNS[(_c, s)]["random_candidates"] for s in range(NUM_SEEDS[_c])]
    print(
        f"    ランダムな候補の中での指標(正例 1 個 + ランダムな負例 {RANDOM_CANDIDATES - 1} 個、query {len(RANDOM_CANDIDATE_QUERIES_INDEX)} 個): {_c}({'困難な負例' if CONDITIONS[_c]['negatives'] == 'hard' else 'ランダムな負例'}で学習)"
        f" Recall@1 {np.mean([r['recall_at_1'] for r in _rc]):.4f}・Recall@10 {np.mean([r['recall_at_10'] for r in _rc]):.4f}"
    )
_window = final_loss_window(NUM_STEPS)
for _c in ("D1", "E2"):
    _first = np.mean([RUNS[(_c, s)]["train_loss"][:_window].mean() for s in range(NUM_SEEDS[_c])])
    _last = np.mean([RUNS[(_c, s)]["final_train_loss"] for s in range(NUM_SEEDS[_c])])
    print(f"    学習の損失の推移: {_c} 最初の {_window} ステップの平均 {_first:.3f} -> 最後の {_window} ステップの平均 {_last:.3f}(一様な予測の損失 ln(1 + n) = {math.log(1 + NUM_NEGATIVES):.3f})")

print(
    "    学習後のモデルの、query ごとの候補 50 件のスコアの標準偏差の中央値(崩壊の検出。判定には使わない): "
    + "、".join(f"{_c} {rounded([RUNS[(_c, s)]['score_std_median'] for s in _seeds_e], 4)}" for _c in ("D1", "E2"))
)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for _x, (_c, _label) in enumerate((("D1", "E1 (hard negatives)"), ("E2", "E2 (random negatives)"))):
    _values = recall_of(_c, SEEDS_E)
    axes[0].scatter([_x] * len(_values), _values, alpha=0.6)
    axes[0].plot([_x - 0.2, _x + 0.2], [_values.mean()] * 2, "k-")
axes[0].axhline(COVERAGE, color="gray", linestyle="--", label="coverage (upper bound)")
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["E1 hard", "E2 random"])
axes[0].set_ylabel(f"Recall@10 after reranking top-{RERANK_CANDIDATES}")
axes[0].set_title(f"{PLOT_TAG}Experiment E")
axes[0].legend(fontsize=7)
for _c in ("D1", "E2"):
    _curves = np.stack([np.convolve(RUNS[(_c, s)]["train_loss"], np.ones(max(1, NUM_STEPS // 32)) / max(1, NUM_STEPS // 32), mode="valid") for s in range(NUM_SEEDS[_c])])
    axes[1].plot(_curves.mean(axis=0), label=_c)
axes[1].axhline(math.log(1 + NUM_NEGATIVES), color="gray", linestyle=":", label="uniform prediction")
axes[1].set_xlabel("step")
axes[1].set_ylabel("training loss (moving average)")
axes[1].set_title(f"{PLOT_TAG}training loss")
axes[1].legend(fontsize=7)
plt.tight_layout()
plt.show()
```

    実験 E(シード [0, 1, 2, 3, 4]、T = 1024、K = 50、並べ替えの query 1446 個。E1 = D1)
      並べ替えの Recall@10 R(E1): [0.1708, 0.1791, 0.1722, 0.1584, 0.1757]、R(E2): [0.1701, 0.1812, 0.1736, 0.175, 0.1812]
      d_s = R(E1) - R(E2): [0.0007, -0.0021, -0.0014, -0.0166, -0.0055]
      Delta_E = -0.0050、sigma_E = 0.0076(シード間 0.0031・ブートストラップ 0.0069、反復 10,000)、閾値 2 sigma_E = 0.0151
      前提条件: P0 True、P1 False((a)の不成立 [('D1', 1), ('D1', 2)]、最後の区間の損失の差の平均 / 標準誤差の最小 1.2(閾値 2.0)、最後の区間の訓練損失の最大 2.078(閾値 ln(1 + n) = 2.079)、参考として最後の区間の訓練損失 / 学習前のモデルの損失は 0.203〜0.993。(b)基準の条件の較正の query での Recall@10 の増分のシード平均 +0.0176(シードごと [0.0525, 0.0525, 0.0228, -0.0148, -0.0251])、標準偏差 0.0213(シード間 0.0163・ブートストラップ 0.0137)、増分 / 標準偏差 0.8(閾値 2.0): False)、P2 True(R(E2) の平均 0.1762 <= coverage 0.4094 - 0.05)、P3 True(実験 D と同じ)
      判定関数の結果 判定不能 -> 最終判定: 前提不成立
      診断量(判定なし):
        ランダムな候補の中での指標(正例 1 個 + ランダムな負例 49 個、query 500 個): D1(困難な負例で学習) Recall@1 0.0188・Recall@10 0.2156
        ランダムな候補の中での指標(正例 1 個 + ランダムな負例 49 個、query 500 個): E2(ランダムな負例で学習) Recall@1 0.1904・Recall@10 0.6644
        ランダムな候補の中での指標(正例 1 個 + ランダムな負例 49 個、query 500 個): D2(困難な負例で学習) Recall@1 0.2892・Recall@10 0.6612
        学習の損失の推移: D1 最初の 51 ステップの平均 2.098 -> 最後の 51 ステップの平均 2.072(一様な予測の損失 ln(1 + n) = 2.079)
        学習の損失の推移: E2 最初の 51 ステップの平均 2.069 -> 最後の 51 ステップの平均 0.547(一様な予測の損失 ln(1 + n) = 2.079)
        学習後のモデルの、query ごとの候補 50 件のスコアの標準偏差の中央値(崩壊の検出。判定には使わない): D1 [0.0457, 0.0696, 0.0721, 0.0486, 0.1197]、E2 [1.3394, 1.2784, 1.2577, 1.2114, 1.2822]



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/025_ann_search_and_reranking/output_52_1.png)
    


### 6.13 実験 F: 2 段階の検索の全体(判定基準なし)

**観察であり、判定基準を設けない。** 第 1 段を全探索・HNSW・転置ファイル・Product Quantization(再採点なし・あり)に替え、それぞれの上位 100 件を、D1 のシード 0 のモデルで並べ替える。
並べ替える候補の数 $K = 10, 20, 50, 100$ は、第 1 段の上位 100 件の先頭 $K$ 件を使う。表の費用は、1 query あたりの第 1 段の距離計算の回数(Product Quantization は参照表の作成・表引き・再採点の距離計算を別々に)と、
第 2 段の順伝播の回数($K$)である。

**対象の索引**: 6.6 節で保持した、シード 0 の最大の $N$($N_{\max}$)の部分集合の索引(HNSW・転置ファイル・Product Quantization)。全探索も同じ部分集合である。**関連性は、部分集合の中での同じ記事の passage**
(部分集合は passage をランダムに間引くので、正解の数が実験 D・E のコーパス全体より少ない。正解が 1 つもない query は除く)。そのため、この表の Recall@10 の絶対値は、実験 D・E(コーパス全体)と比べない。
**第 1 段の設定**: HNSW は、実験 A のシード 0 の $N_{\max}$ の掃引で、目標の recall に初めて届いた探索の幅(上位 100 件を返すため、探索の幅は 101 以上にする)。転置ファイルは、実験 B のシード 0 で費用が最小だったリスト数と、
その掃引で目標の recall に初めて届いた探索するリストの数。Product Quantization は非対称距離計算で、再採点ありは上位 200 件を再採点する。


```python
_t0_f = time.time()
F_SUBSET = ann_ordering(0)[:N_MAX]  # 6.6 節のシード 0 の索引と同じ部分集合
_positions = positions_in_subset(F_SUBSET, NUM_PASSAGES)
_subset_article_counts = np.bincount(PASSAGE_ARTICLES[F_SUBSET], minlength=int(PASSAGE_ARTICLES.max()) + 1)
_f_in_subset = _positions[QUERY_SOURCES[F_QUERY_INDEX]] >= 0
_f_num_positives = _subset_article_counts[QUERY_ARTICLES[F_QUERY_INDEX]] - _f_in_subset.astype(int)  # 部分集合の中の同じ記事の passage の数(元の passage を除く)
_f_keep = _f_num_positives >= 1
F_QUERIES = F_QUERY_INDEX[_f_keep]
F_NUM_POSITIVES = _f_num_positives[_f_keep]
_f_local_excluded = _positions[QUERY_SOURCES[F_QUERIES]]
_f_excluded = np.where(_f_local_excluded >= 0, QUERY_SOURCES[F_QUERIES], -1)
_f_query_embeddings = QUERY_EMBEDDINGS[F_QUERIES]
_f_depth = max(RERANK_KS)
F_FIRST_STAGE = {}  # 方法 -> (候補 (Q, 100)、費用の辞書)
# 全探索
_local, _ = exact_search(_f_query_embeddings, PASSAGE_EMBEDDINGS[torch.from_numpy(F_SUBSET)], _f_depth, excluded=_f_local_excluded)
F_FIRST_STAGE["exact search"] = (F_SUBSET[_local], {"distance computations": float(N_MAX)})
# HNSW
_sweep0 = HNSW_RUNS[0]["sweeps"][N_MAX]
_search_width_target = next((w for w, r in zip(_sweep0["grid"], _sweep0["recall"].mean(axis=1), strict=True) if r >= TARGET_RECALL), _sweep0["grid"][-1])
_search_width = max(_search_width_target, _f_depth + 1)
_ids, _costs = HNSW_RUNS[0]["index"].search_batch(QUERY_EMBEDDINGS_NP[F_QUERIES], _f_depth + 1, _search_width)
F_FIRST_STAGE[f"HNSW (search width = {_search_width})"] = (remove_excluded_and_truncate(_ids, _f_excluded, _f_depth), {"distance computations": float(_costs.mean())})
# 転置ファイル(実験 B のシード 0 で費用が最小のリスト数、その掃引で目標の recall に初めて届いた p)
_best_lists = _inverted_file_options[best_option(B_IVF_LOG_COST[0])]
_sweep_inverted_file = INVERTED_FILE_RUNS[0]["sweeps"][_best_lists]
_p_target = next((p for p, r in zip(_sweep_inverted_file["grid"], _sweep_inverted_file["recall"].mean(axis=1), strict=True) if r >= TARGET_RECALL), _sweep_inverted_file["grid"][-1])
_result = INVERTED_FILE_RUNS[0]["indices"][_best_lists].search_batch_by_probe_counts(_f_query_embeddings, _f_depth, [_p_target], excluded=_f_local_excluded)[_p_target]
F_FIRST_STAGE[f"inverted file ({_best_lists} lists, p = {_p_target})"] = (
    np.where(_result[0] >= 0, F_SUBSET[np.maximum(_result[0], 0)], -1), {"distance computations": float(_result[1].mean())}
)
# Product Quantization(非対称距離計算。再採点なし・あり)
_product_quantization_index = PRODUCT_QUANTIZATION_RUNS[0]["index"]
for _rescored in (0, 200):
    _ids, _costs = _product_quantization_index.search_asymmetric(_f_query_embeddings, _f_depth, excluded=_f_local_excluded, num_rescored=_rescored)
    F_FIRST_STAGE["Product Quantization" + (f" + re-scoring (R = {_rescored})" if _rescored else "")] = (
        F_SUBSET[_ids], {"table entries": float(_costs["table_entries"].mean()), "table lookups": float(_costs["table_lookups"].mean()), "re-scored distance computations": float(_costs["rescored_vectors"].mean())}
    )
assert all((ids >= 0).all() for ids, _ in F_FIRST_STAGE.values()), "候補が足りない query がある"
assert all(np.isin(ids, F_SUBSET).all() and (ids != _f_excluded[:, None]).all() for ids, _ in F_FIRST_STAGE.values())

# D1 のシード 0(アップロードするモデルと同じ重み)で、各方法の上位 100 件を並べ替える
_f_model = build_model("D1", 0).to(device)
_f_model.load_state_dict(SAVED_STATES[UPLOAD_RUN])
F_TABLE = {}
for _name, (_candidates, _cost) in F_FIRST_STAGE.items():
    _scores = score_candidates(_f_model, "cross_encoder", QUERY_TOKENS[torch.as_tensor(F_QUERIES, device=device)], PASSAGE_TOKENS, _candidates, SCORE_BATCH, USE_FP16_AUTOCAST)
    F_TABLE[_name] = {"cost": _cost}
    for _k in RERANK_KS:
        _after = rank_reranked_candidates(_scores[:, :_k], _candidates[:, :_k], QUERY_ARTICLES[F_QUERIES], PASSAGE_ARTICLES, F_NUM_POSITIVES)
        _before = rank_reranked_candidates(
            np.broadcast_to(FIRST_STAGE_ORDER_SCORES[:, :_k], (len(F_QUERIES), _k)), _candidates[:, :_k], QUERY_ARTICLES[F_QUERIES], PASSAGE_ARTICLES, F_NUM_POSITIVES
        )
        F_TABLE[_name][_k] = {"recall_after": _after.recall_at(10), "coverage": _after.coverage(), "recall_before": _before.recall_at(10)}
del _f_model
empty_device_cache()

print(
    f"{RUN_TAG}実験 F(判定基準なしの観察。部分集合 N = {N_MAX}(シード 0)、query {len(F_QUERIES)} 個(正解がなく除いた query {int((~_f_keep).sum())} 個)、"
    f"部分集合内の正解の数の中央値 {int(np.median(F_NUM_POSITIVES))}、D1 のシード 0 で並べ替え)"
)
for _name, _row in F_TABLE.items():
    print(f"  第 1 段: {_name}、費用(1 query あたり): " + "、".join(f"{k} {v:,.0f}" for k, v in _row["cost"].items()))
    for _k in RERANK_KS:
        _m = _row[_k]
        print(f"    K = {_k:>3}: 順伝播 {_k} 回、coverage {_m['coverage']:.4f}、Recall@10: 並べ替えなし {_m['recall_before']:.4f} -> D1 で並べ替え {_m['recall_after']:.4f}")
F_SECONDS = time.time() - _t0_f
print(f"実験 F {F_SECONDS:.1f} 秒")

fig, ax = plt.subplots(figsize=(7, 4))
for _name in F_TABLE:
    ax.plot(RERANK_KS, [F_TABLE[_name][_k]["recall_after"] for _k in RERANK_KS], "o-", label=_name)
ax.set_xscale("log")
ax.set_xlabel("K (candidates reranked by D1 seed 0)")
ax.set_ylabel("Recall@10 after reranking")
ax.set_title(f"{PLOT_TAG}Experiment F: first stage x K (N = {N_MAX})")
ax.legend(fontsize=6)
plt.tight_layout()
plt.show()
```

    実験 F(判定基準なしの観察。部分集合 N = 65536(シード 0)、query 594 個(正解がなく除いた query 6 個)、部分集合内の正解の数の中央値 7、D1 のシード 0 で並べ替え)
      第 1 段: exact search、費用(1 query あたり): distance computations 65,536
        K =  10: 順伝播 10 回、coverage 0.1481、Recall@10: 並べ替えなし 0.1481 -> D1 で並べ替え 0.1481
        K =  20: 順伝播 20 回、coverage 0.2104、Recall@10: 並べ替えなし 0.1481 -> D1 で並べ替え 0.1195
        K =  50: 順伝播 50 回、coverage 0.2997、Recall@10: 並べ替えなし 0.1481 -> D1 で並べ替え 0.1128
        K = 100: 順伝播 100 回、coverage 0.4259、Recall@10: 並べ替えなし 0.1481 -> D1 で並べ替え 0.0960
      第 1 段: HNSW (search width = 101)、費用(1 query あたり): distance computations 533
        K =  10: 順伝播 10 回、coverage 0.1465、Recall@10: 並べ替えなし 0.1465 -> D1 で並べ替え 0.1465
        K =  20: 順伝播 20 回、coverage 0.2071、Recall@10: 並べ替えなし 0.1465 -> D1 で並べ替え 0.1246
        K =  50: 順伝播 50 回、coverage 0.2929、Recall@10: 並べ替えなし 0.1465 -> D1 で並べ替え 0.1128
        K = 100: 順伝播 100 回、coverage 0.4024、Recall@10: 並べ替えなし 0.1465 -> D1 で並べ替え 0.0808
      第 1 段: inverted file (512 lists, p = 11)、費用(1 query あたり): distance computations 1,907
        K =  10: 順伝播 10 回、coverage 0.1313、Recall@10: 並べ替えなし 0.1313 -> D1 で並べ替え 0.1313
        K =  20: 順伝播 20 回、coverage 0.1902、Recall@10: 並べ替えなし 0.1313 -> D1 で並べ替え 0.1178
        K =  50: 順伝播 50 回、coverage 0.2811、Recall@10: 並べ替えなし 0.1313 -> D1 で並べ替え 0.1061
        K = 100: 順伝播 100 回、coverage 0.3889、Recall@10: 並べ替えなし 0.1313 -> D1 で並べ替え 0.0859
      第 1 段: Product Quantization、費用(1 query あたり): table entries 8,192、table lookups 2,097,152、re-scored distance computations 0
        K =  10: 順伝播 10 回、coverage 0.1229、Recall@10: 並べ替えなし 0.1229 -> D1 で並べ替え 0.1229
        K =  20: 順伝播 20 回、coverage 0.1919、Recall@10: 並べ替えなし 0.1229 -> D1 で並べ替え 0.1111
        K =  50: 順伝播 50 回、coverage 0.2980、Recall@10: 並べ替えなし 0.1229 -> D1 で並べ替え 0.1178
        K = 100: 順伝播 100 回、coverage 0.3788、Recall@10: 並べ替えなし 0.1229 -> D1 で並べ替え 0.0909
      第 1 段: Product Quantization + re-scoring (R = 200)、費用(1 query あたり): table entries 8,192、table lookups 2,097,152、re-scored distance computations 200
        K =  10: 順伝播 10 回、coverage 0.1481、Recall@10: 並べ替えなし 0.1481 -> D1 で並べ替え 0.1481
        K =  20: 順伝播 20 回、coverage 0.2104、Recall@10: 並べ替えなし 0.1481 -> D1 で並べ替え 0.1229
        K =  50: 順伝播 50 回、coverage 0.2963、Recall@10: 並べ替えなし 0.1481 -> D1 で並べ替え 0.1094
        K = 100: 順伝播 100 回、coverage 0.4091、Recall@10: 並べ替えなし 0.1481 -> D1 で並べ替え 0.0960
    実験 F 126.0 秒



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/025_ann_search_and_reranking/output_54_1.png)
    


### 6.14 不変条件のアサーションと`SMOKE_TEST`の配線

- 実効水準の照合: 実際に使われた学習ステップ数・$N_{\max}$・シード数・条件・学習した回数が、5.2 節で印字した水準と 6.4 節で選ばれた計画に一致する。
- 同じシードの条件どうしで、query と正例の系列(ハッシュ)が一致し、D1 と D2 は負例も一致する(E2 だけ負例が異なる)。異なるシードでは異なる。
- 学習に使った query がすべて採掘の上位 50 件に同じ記事の passage がある query で、正例がその上位 50 件の中にある(使い回している学習の組のすべてで確かめる)。
- 学習率が、較正で選ばれた値(すべての格子点が候補から外れた条件は格子の中心)に一致する。
- 全条件で、学習ステップ数・バッチ・query と passage の長さ・並べ替えの候補・評価の query が同じである。
- 学習率が、較正した値に一致する。記録した順位から再計算した Recall@10 が、記録した hits の平均と一致する。
- 近似最近傍探索: すべての索引で、正解が部分集合での全探索の上位 10 件であること(再計算で一致)、HNSW・転置ファイル・Product Quantization の部分集合が同じシードで同じであること。
- キャッシュの導入前後の一致: 使い回している学習の組(`SCHEDULES`)が、毎回作り直した場合と完全に一致する。符号化したトークンと埋め込みのキャッシュは 5.3 節で、キャッシュの再計算との一致を確かめた。
- 評価用の索引・query の記事が、すべて 008 の事前学習とトークナイザの学習の範囲(参照コーパス)の外にある。


```python
# --- 実効水準の照合(SMOKE_TEST の配線) ---
assert CURRENT_LEVEL_NAME == ("smoke" if SMOKE_TEST else "prod")
assert STEP_CANDIDATES == LEVELS[CURRENT_LEVEL_NAME]["STEP_CANDIDATES"] and NUM_STEPS in STEP_CANDIDATES
assert N_MAX_CANDIDATES == LEVELS[CURRENT_LEVEL_NAME]["N_MAX_CANDIDATES"] and N_MAX in N_MAX_CANDIDATES
assert BOOTSTRAP_RESAMPLES == LEVELS[CURRENT_LEVEL_NAME]["BOOTSTRAP_RESAMPLES"] and ANN_BOOTSTRAP_WEIGHTS.shape[0] == RERANK_BOOTSTRAP_WEIGHTS.shape[0] == BOOTSTRAP_RESAMPLES
assert SELECTED_PLAN == PLANS[SELECTED_PLAN["plan"]] and PLAN_SETTINGS == stage_settings(STAGE) and N_MAX == PLAN_SETTINGS["n_max"]
assert set(RUNS) == {(c, s) for c in CONDITIONS for s in range(NUM_SEEDS[c])}, "学習した条件・シードが選ばれた計画と一致しない"
assert set(HNSW_RUNS) == set(INVERTED_FILE_RUNS) == set(PRODUCT_QUANTIZATION_RUNS) == set(range(NUM_ANN_SEEDS))
assert all(r["num_steps"] == NUM_STEPS and len(r["train_loss"]) == NUM_STEPS for r in RUNS.values())
assert all(r["learning_rate"] == LEARNING_RATE[r["condition"]] == CALIBRATION[r["condition"]]["chosen"] for r in RUNS.values())
assert all(h["sizes"] == SIZE_LEVELS and set(h["sweeps"]) == set(SIZE_LEVELS) for h in HNSW_RUNS.values())
assert all(set(r["num_lists"]) == set(inverted_file_list_counts_for(N_MAX)) for r in INVERTED_FILE_RUNS.values())
assert len(ANN_QUERY_INDEX) == len(ANN_QUERY_ARTICLES) and (UPLOAD_RUN in SAVED_STATES) and set(SAVED_STATES) == {UPLOAD_RUN}
# --- 対応のある比較: 同じシードの条件どうしで、query と正例の系列が一致する。D1・D2 は負例も一致する。異なるシードでは異なる ---
for _s in range(min(NUM_SEEDS.values())):
    _schedule = get_schedule(_s, NUM_STEPS)
    assert len({RUNS[(c, _s)]["schedule_hash"] for c in ("D1", "D2")}) == 1, f"シード {_s} で D1 と D2 の学習の組が一致しない"
    assert RUNS[("E2", _s)]["schedule_hash"] != RUNS[("D1", _s)]["schedule_hash"]  # E2 だけ負例が異なる(ランダム)
assert len({get_schedule(s, NUM_STEPS).digest() for s in range(NUM_SEEDS["D1"])}) == NUM_SEEDS["D1"], "異なるシードで学習の組が同じ"
assert all(get_schedule(0, NUM_STEPS).digest() == get_schedule(0, NUM_STEPS).digest() for _ in [0])
# --- キャッシュの導入前後の一致: 使い回している学習の組(SCHEDULES)が、作り直した場合と完全に一致する ---
for (_seed_index, _steps), _cached in SCHEDULES.items():
    _fresh = sample_reranking_schedule(
        TRAIN_QUERY_INDEX, MINED_CANDIDATES, QUERY_ARTICLES, QUERY_SOURCES, PASSAGE_ARTICLES, TRAIN_PASSAGE_INDEX, _steps, BATCH_QUERIES, NUM_NEGATIVES, seed=SCHEDULE_SEED_BASE + _seed_index
    )
    assert all(np.array_equal(getattr(_cached, n), getattr(_fresh, n)) for n in ("query", "positive", "hard_negatives", "random_negatives")), (_seed_index, _steps)
print(f"学習の組のキャッシュ({len(SCHEDULES)} 通り)が、作り直した場合と完全に一致: OK")
# --- 学習に使った query は採掘の上位に同じ記事の passage がある query で、正例はその上位の中にある ---
_usable_query_set_check = set(TRAIN_QUERY_INDEX[USABLE_TRAIN_QUERY_MASK].tolist())
_mined_of_query_check = {int(q): set(MINED_CANDIDATES[i].tolist()) for i, q in enumerate(TRAIN_QUERY_INDEX)}
for (_seed_index, _steps), _cached in SCHEDULES.items():
    assert all(int(q) in _usable_query_set_check for q in _cached.query.ravel()), (_seed_index, _steps, "学習に使えない query が入っている")
    assert all(int(p) in _mined_of_query_check[int(q)] for q, p in zip(_cached.query.ravel(), _cached.positive.ravel(), strict=True)), (_seed_index, _steps, "正例が採掘の上位の中にない")
print(f"学習に使った query はすべて学習に使える query({NUM_USABLE_TRAIN_QUERIES:,} 個のうち)で、正例はすべて採掘の上位 {MINING_DEPTH} 件の中にある(学習の組 {len(SCHEDULES)} 通り): OK")
# --- 全条件で評価の分母が同じ ---
assert all(r["hits"].shape == (len(RERANK_QUERY_INDEX),) and r["rank"].shape == (len(RERANK_QUERY_INDEX),) for r in RUNS.values())
for _record in RUNS.values():
    assert np.array_equal(_record["rank"] <= 10, _record["hits"])
    assert _record["rank"].min() >= 1 and _record["rank"].max() <= RERANK_CANDIDATES + 1
assert RERANK_CANDIDATE_IDS.shape == (len(RERANK_QUERY_INDEX), RERANK_CANDIDATES) and CALIBRATION_CANDIDATE_IDS.shape == (len(CALIBRATION_QUERY_INDEX), RERANK_CANDIDATES)
# --- 近似最近傍探索: 正解の再計算と、同じシードの部分集合の一致 ---
for _s in _seeds:
    _subset = ann_ordering(_s)[:N_MAX]
    _truth, _, _ = ann_truth(_s, _subset)
    _positions_check = positions_in_subset(_subset, NUM_PASSAGES)
    _local = _positions_check[QUERY_SOURCES[ANN_QUERY_INDEX]]
    _again, _ = exact_search(ANN_QUERIES_TENSOR, PASSAGE_EMBEDDINGS[torch.from_numpy(_subset)], NEIGHBORS, excluded=_local)
    assert np.array_equal(_subset[_again], _truth), "正解が再計算と一致しない"
    assert np.array_equal(HNSW_RUNS[_s]["sweeps"][N_MAX]["recall"].shape[1:], (len(ANN_QUERY_INDEX),))
    assert np.array_equal(np.sort(INVERTED_FILE_RUNS[_s]["subset"]), np.sort(_subset)) and np.array_equal(np.sort(PRODUCT_QUANTIZATION_RUNS[_s]["subset"]), np.sort(_subset))
    _hnsw_subset = np.sort(HNSW_RUNS[0]["index"].inserted_elements()) if _s == 0 else None
    if _s == 0:
        assert np.array_equal(_hnsw_subset, np.sort(_subset)), "HNSW の挿入済みの要素が、転置ファイル・Product Quantization の部分集合と一致しない"
for _s in _seeds:  # 掃引の recall が 0 以上 1 以下で、費用が正
    for _sweep in list(HNSW_RUNS[_s]["sweeps"].values()) + list(INVERTED_FILE_RUNS[_s]["sweeps"].values()):
        assert (_sweep["recall"] >= 0).all() and (_sweep["recall"] <= 1).all() and (_sweep["cost"] > 0).all()
# --- 学習の成立の前提条件が、更新がない(学習率 0)の学習で不成立になる(基準の条件 D2・E2、32 ステップ)---
# 更新がなければ、(a)の損失の差と(b)の増分は 0(丸めの範囲)になる。(a)の「最後の区間の訓練損失 < ln(1 + n)」は、学習前のモデルの損失が一様な予測より高いか低いかに依るので、値を印字する
_zero_checks = {}
for _condition in BASELINE_CONDITIONS:
    _record = train_run(_condition, CALIBRATION_SEED_INDEX, 0.0, 32, main=False, with_untrained=True)
    _zero_checks[_condition] = (learning_loss_statistics(_record), validation_gain_statistics([_record]))
    assert not learning_precondition(_record), f"学習率 0 の学習で学習の成立 (a) が成立した({_condition})"
    assert not validation_gain_precondition([_record]), f"学習率 0 の学習で学習の成立 (b) が成立した({_condition})"
print(
    "学習の成立の前提条件は、学習率 0(更新なし)の学習で不成立になる: "
    + "、".join(
        f"{c}: (a) 損失の差の平均 {st['mean']:+.2e}・標準誤差 {st['standard_error']:.2e}・最後の区間の損失 {st['final_loss']:.3f}(ln(1 + n) = {UNIFORM_LOSS:.3f})、(b) 増分 {g['gain']:+.4f}"
        for c, (st, g) in _zero_checks.items()
    )
)
# --- 008 とトークナイザの学習の範囲の外 ---
_evaluated_articles = set(QUERY_ARTICLES[RERANK_QUERY_INDEX].tolist()) | set(QUERY_ARTICLES[ANN_QUERY_INDEX].tolist()) | set(PASSAGE_ARTICLES.tolist())
assert min(_evaluated_articles) >= NUM_REFERENCE_ARTICLES
assert all(r["finite"] for r in RUNS.values()), "有限でない損失の学習がある"
print(
    f"実効水準: T = {NUM_STEPS}、計画 {SELECTED_PLAN['plan']}(段階 {STAGE})、N_max = {N_MAX}、近似最近傍探索のシード数 {NUM_ANN_SEEDS}、並べ替えのシード数 {json.dumps(NUM_SEEDS)}、"
    f"学習 {len(RUNS)} 回、バッチ {BATCH_QUERIES} query x (正例 1 + 負例 {NUM_NEGATIVES})、K = {RERANK_CANDIDATES}、並べ替えの query {len(RERANK_QUERY_INDEX)} 個、近似最近傍探索の query {len(ANN_QUERY_INDEX)} 個、"
    f"ブートストラップ {BOOTSTRAP_RESAMPLES:,} 回: 照合 OK"
)
print(f"同じシードの D1 と D2 で学習の組(query・正例・負例)が一致、E2 は query と正例が同じで負例だけ異なる、異なるシードでは異なる(シード {min(NUM_SEEDS.values())} 個で確認): OK")
```

    学習の組のキャッシュ(7 通り)が、作り直した場合と完全に一致: OK
    学習に使った query はすべて学習に使える query(39,303 個のうち)で、正例はすべて採掘の上位 50 件の中にある(学習の組 7 通り): OK
    学習の成立の前提条件は、学習率 0(更新なし)の学習で不成立になる: D2: (a) 損失の差の平均 +0.00e+00・標準誤差 0.00e+00・最後の区間の損失 2.521(ln(1 + n) = 2.079)、(b) 増分 +0.0000、E2: (a) 損失の差の平均 +0.00e+00・標準誤差 0.00e+00・最後の区間の損失 2.201(ln(1 + n) = 2.079)、(b) 増分 +0.0000
    実効水準: T = 1024、計画 1(段階 1)、N_max = 65536、近似最近傍探索のシード数 3、並べ替えのシード数 {"D1": 5, "D2": 5, "E2": 5}、学習 15 回、バッチ 8 query x (正例 1 + 負例 7)、K = 50、並べ替えの query 1446 個、近似最近傍探索の query 890 個、ブートストラップ 10,000 回: 照合 OK
    同じシードの D1 と D2 で学習の組(query・正例・負例)が一致、E2 は query と正例が同じで負例だけ異なる、異なるシードでは異なる(シード 5 個で確認): OK


### 6.15 判定結果の一覧


```python
print(f"{RUN_TAG}判定結果(計画 {SELECTED_PLAN['plan']}、T = {NUM_STEPS}、段階 {STAGE}: {STAGE_CHANGES[STAGE]})")
VERDICTS = {
    "A": (DELTA_A, SIGMA_A["sigma"], A_COMPUTED, A_VERDICT, A_PRECONDITIONS),
    "B": (DELTA_B, SIGMA_B["sigma"], B_COMPUTED, B_VERDICT, B_PRECONDITIONS),
    "C": (DELTA_C, SIGMA_C["sigma"], C_COMPUTED, C_VERDICT, C_PRECONDITIONS),
    "D": (DELTA_D, SIGMA_D["sigma"], D_COMPUTED, D_VERDICT, D_PRECONDITIONS),
    "E": (DELTA_E, SIGMA_E["sigma"], E_COMPUTED, E_VERDICT, E_PRECONDITIONS),
}
for _experiment, (_delta, _sigma, _computed, _verdict, _preconditions) in VERDICTS.items():
    print(
        f"  実験 {_experiment}: 対比量 {_delta:+.4f}、標準偏差 {_sigma:.4f}、閾値 {SIGMA_MULTIPLIER * _sigma:.4f}、判定関数の結果 {_computed}、"
        f"前提条件 {json.dumps({k: precondition_status[k] for k in _preconditions})} -> {RUN_TAG}最終判定: {_verdict}"
    )
TOTAL_SECONDS = time.time() - NOTEBOOK_START_TIME
print(f"全体の経過時間 {TOTAL_SECONDS / 60:.1f} 分(予算 {SESSION_BUDGET_SECONDS / 60:.0f} 分、計画の見積もり {PLAN_ESTIMATES[SELECTED_PLAN['plan']]['total'] / 60:.1f} 分 + 選択時の経過 {ELAPSED_AT_SELECTION / 60:.1f} 分)")
```

    判定結果(計画 1、T = 1024、段階 1: 実験 A・B・C のシード数を削る)
      実験 A: 対比量 +0.6657、標準偏差 0.0102、閾値 0.0204、判定関数の結果 支持、前提条件 {"P1(A)": true} -> 最終判定: 支持
      実験 B: 対比量 +1.4339、標準偏差 0.0307、閾値 0.0613、判定関数の結果 支持、前提条件 {"P0(B)": true, "P1(B)": true} -> 最終判定: 支持
      実験 C: 対比量 +0.1296、標準偏差 0.0033、閾値 0.0067、判定関数の結果 支持、前提条件 {"P2(C)": true} -> 最終判定: 支持
      実験 D: 対比量 -0.0625、標準偏差 0.0084、閾値 0.0168、判定関数の結果 反証、前提条件 {"P0(D)": true, "P1(D)": false, "P2(D)": true, "P3(D)": true} -> 最終判定: 前提不成立
      実験 E: 対比量 -0.0050、標準偏差 0.0076、閾値 0.0151、判定関数の結果 判定不能、前提条件 {"P0(E)": true, "P1(E)": false, "P2(E)": true, "P3(E)": true} -> 最終判定: 前提不成立
    全体の経過時間 98.9 分(予算 120 分、計画の見積もり 97.1 分 + 選択時の経過 10.8 分)


### 6.16 Hugging Face Hub へのアップロード(D1 のシード 0)

後続のアプリ(`apps/`)の入力にするため、**D1(困難な負例で学習した cross-encoder)のシード 0** の学習後の重みを、公開の条件(下)を満たす場合に、`kojikojiprg/ai-theories-cross-encoder-en`(新規、public)にアップロードする設計だった。
アップロードするモデルは結果を見て選ばない(条件とシードを事前に固定している)。**本番では公開の条件を満たさず、アップロードは行われなかった(7.10 節)。**

- **アップロードのセルは`UPLOAD_ARTIFACTS`で守る。既定は`False`。** `SMOKE_TEST`とは独立のフラグで、本番(`SMOKE_TEST = False`)でも、アップロードは`UPLOAD_ARTIFACTS`を別途`True`にしたときだけ行う。
- **公開の条件**: `UPLOAD_ARTIFACTS = True`でも、アップロードは **読み込み直した D1 のシード 0 の、評価用の query での Recall@10 が、並べ替えなしの Recall@10(第 1 段の順位のまま)を上回る場合だけ** 行う。
  満たさない場合はアップロードを行わず、その旨と両方の値を印字する(`HF_TOKEN`の取得も行わない)。**これは実験の判定ではなく、公開するかどうかの条件である**(判定の対比量にも、前提条件にも使わない)。
  並べ替えで第 1 段より悪くなるモデルを、後続のアプリの入力として公開しないためである。
- Colab Secrets の`HF_TOKEN`が取得できない場合は、例外で停止せず、アップロードをスキップしてその旨を印字する。取得した値は印字・記録・加工しない。
- アップロード後、`list_repo_files()`でリポジトリに意図したファイルが実際に存在することを確かめる。`upload_file`が例外を送出しなかったことだけを根拠にしない。
- **このセルの前半(ファイルの書き出し・読み込み直し・評価・モデルカードの組み立て)は、`UPLOAD_ARTIFACTS`によらず毎回実行する**(アップロードしなくてもモデルカードの下書きを確認できるように)。
  モデルカードの指標は、書き出した重みを新しいモデルに読み込み直して評価した値とする(学習中の途中の評価値は使わない)。
- トークナイザと第 1 段のモデルは同梱しない。モデルカードで`kojikojiprg/ai-theories-tokenizer-en`・`kojikojiprg/ai-theories-text-embedding-en`を参照する。索引・埋め込み・他の条件のモデルはアップロードしない。


```python
_t0_upload = time.time()
MODEL_CARD_TEMPLATE = '''---
language: en
license: mit
tags:
- ai-theories
- cross-encoder
- reranking
- retrieval
- scratch-implementation
---

# ai-theories cross-encoder(英語、困難な負例で学習した並べ替えモデル)

`ai-theories`(https://github.com/kojikojiprg/ai-theories)プロジェクトの成果物。
008 の小型 GPT(`kojikojiprg/ai-theories-small-gpt-en`)を起点に、query と passage を連結して符号化し、終端の位置の出力から並べ替えのスコアを出す cross-encoder として、
全パラメータを更新して学習したもの。第 1 段の検索器(`kojikojiprg/ai-theories-text-embedding-en`)の上位から採掘した困難な負例で学習した。
スクラッチ実装であり、研究・教育目的のモデルである。品質保証は行っていない。商用・実運用での利用は想定しない。

## 由来

- [025. ANN 検索とリランキング](https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/025_ann_search_and_reranking.ipynb)の
  条件 D1(困難な負例で学習した cross-encoder)のシード 0 の学習後の重み。結果を見て選んだものではなく、事前に固定した条件とシードである。
- 学習: {num_steps} ステップ、1 ステップ {batch_queries} query x(正例 1 + 負例 {num_negatives})、query {query_length} トークン・passage {passage_length} トークン、学習率 {learning_rate:.3g}、
  AdamW(重み減衰 {weight_decay})、warmup + cosine、gradient clipping {clip}。損失は、1 個の正例と {num_negatives} 個の負例の softmax 交差エントロピー。
  学習データは `kojikojiprg/ai-theories-corpus-en` の、先頭の 356 記事を除く記事のうち学習用に分けた {num_train_articles} 記事のみ(検証用・評価用の記事は含まない)。
  負例は、第 1 段の検索器の、学習用の記事の passage だけを対象にした全探索の上位 {mining_depth} 件から、query と同じ記事の passage を除いて一様に選んだ。

## 使うために必要な他のアーティファクト

- **トークナイザは同梱していない。** `kojikojiprg/ai-theories-tokenizer-en` を使用してください(語彙サイズ 8192、特殊トークンなし)。
- **第 1 段の検索器は同梱していない。** 並べ替える候補は、`kojikojiprg/ai-theories-text-embedding-en`(024 の Matryoshka Representation Learning で学習した埋め込みモデル。平均プール・因果マスク)の
  全探索または近似最近傍探索の上位 {rerank_candidates} 件を想定している。
- 本体の構成は `config.json`(008 の `config.json` と同じ構成に、cross-encoder の設定を加えたもの)を参照。

## 入力とスコア

入力の列は `[query のトークン {query_length} 個] [区切り] [passage のトークン {passage_length} 個] [終端]` の {sequence_length} トークン(パディングなし)。
008 のトークナイザには特殊トークンがないので、語彙を増やさず、コーパスに一度も現れない単一バイトのトークンを使う: 区切りは 0x1F(ID {separator_id})、終端は 0x1E(ID {terminal_id})。
スコアは、因果マスクのもとで query 全体と passage 全体を見ている **最後の位置(終端)** の、最終正規化層の後の隠れ状態に、線形の層(`head`)を掛けたスカラーである。

```python
import json
import torch
from huggingface_hub import hf_hub_download
from src.models.cross_encoder import CrossEncoder

config = json.load(open(hf_hub_download("{repo_id}", "config.json")))
# GPTLanguageModel の構築は 008(kojikojiprg/ai-theories-small-gpt-en)の config.json と同じ(025 のノートブックの build_gpt() を参照)
backbone = build_gpt(config)
model = CrossEncoder(backbone, config["separator_token_id"], config["terminal_token_id"])
model.load_state_dict(torch.load(hf_hub_download("{repo_id}", "model_state.pt"), map_location="cpu"))
model.eval()
with torch.no_grad():
    scores = model(query_token_ids, passage_token_ids)  # 形状 (B,)。query は (B, {query_length})、passage は (B, {passage_length}) のトークン ID
```

## 評価(アップロードした重みそのものを読み込み直して測った値)

評価用の記事の query {num_queries} 個(024 の評価用の集合と同じ手順で切り出した query から、記事ごとに等間隔に選んだもの)を、コーパス全体({num_passages} passage)に対する第 1 段(024 のモデルの全探索)の
上位 {rerank_candidates} 件を並べ替えて、Recall@10(上位 10 件に同じ記事の別の passage が 1 つ以上ある query の割合)などを測った。

| 指標 | 並べ替えなし(第 1 段の順位) | このモデルで並べ替え |
|---|---|---|
| Recall@10 | {first_stage_recall:.4f} | {recall:.4f} |
| 平均逆順位(Mean Reciprocal Rank) | {first_stage_mean_reciprocal_rank:.4f} | {mean_reciprocal_rank:.4f} |

- 並べ替えで到達できる Recall@10 の上限(上位 {rerank_candidates} 件に正例が入る query の割合): {coverage:.4f}。ランダムな順位の Recall@10 の期待値: {random_recall:.4f}。
- 上の値は {smoke_note}

## 注意

- 学習データは英語 Wikipedia の長大な記事(Special:LongPages の長い順に選んだ記事。約 {list_like_share} がリスト系の記事)に限られ、汎用の並べ替えモデルとして使えるものではない。
- 関連性は「query と同じ記事の別の passage」と定義した評価であり、内容が関連する別の記事の passage は負例として扱われる。
- 評価用の集合は、ノートブックの実行時点のコーパス(`article_offsets` を含む `metadata.json`)に依存する。
'''


def build_card_and_files(directory: Path) -> dict:
    # 学習後の重みをファイルに書き出し、新しいモデルに読み込み直して評価した値でモデルカードを組み立てる
    state = SAVED_STATES[UPLOAD_RUN]
    torch.save(state, directory / "model_state.pt")
    config = dict(REFERENCE_CONFIG) | {
        "model_type": "cross_encoder",
        "separator_token_id": SEPARATOR_TOKEN_ID,
        "terminal_token_id": TERMINAL_TOKEN_ID,
        "head": "linear(d_model -> 1) applied to the final-normalized hidden state of the last position",
        "query_length": QUERY_LENGTH,
        "passage_length": PASSAGE_LENGTH,
        "rerank_candidates": RERANK_CANDIDATES,
        "tokenizer_repo": TOKENIZER_REPO_ID,
        "base_model_repo": REFERENCE_MODEL_REPO_ID,
        "first_stage_repo": FIRST_STAGE_REPO_ID,
        "source_notebook": "theories/07_retrieval/025_ann_search_and_reranking.ipynb",
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.manual_seed(0)
    reloaded = CrossEncoder(build_gpt(REFERENCE_CONFIG), SEPARATOR_TOKEN_ID, TERMINAL_TOKEN_ID, HEAD_INIT_STD)
    reloaded.load_state_dict(torch.load(directory / "model_state.pt", map_location="cpu"))
    assert all(torch.equal(a, b) for a, b in zip(reloaded.state_dict().values(), state.values(), strict=True)), "書き出した重みが一致しない"
    reloaded = reloaded.to(device)
    evaluated, _ = evaluate_rerank(reloaded, "cross_encoder", RERANK_QUERY_INDEX, RERANK_CANDIDATE_IDS)
    del reloaded
    empty_device_cache()
    # 学習直後の値(記録した順位)と、読み込み直した重みでの値が一致する(同じ関数・同じデバイス。バッチの大きさは結果に影響しない)
    assert abs(evaluated.recall_at(10) - float(RUNS[UPLOAD_RUN]["hits"].mean())) <= 2 / len(RERANK_QUERY_INDEX), evaluated.recall_at(10)
    card = MODEL_CARD_TEMPLATE.format(
        num_steps=NUM_STEPS, batch_queries=BATCH_QUERIES, num_negatives=NUM_NEGATIVES, query_length=QUERY_LENGTH, passage_length=PASSAGE_LENGTH, learning_rate=LEARNING_RATE["D1"],
        weight_decay=WEIGHT_DECAY, clip=GRADIENT_CLIP_THRESHOLD, num_train_articles=f"{len(SPLIT.train):,}", mining_depth=MINING_DEPTH, rerank_candidates=RERANK_CANDIDATES,
        sequence_length=QUERY_LENGTH + 1 + PASSAGE_LENGTH + 1, separator_id=SEPARATOR_TOKEN_ID, terminal_id=TERMINAL_TOKEN_ID, repo_id=UPLOAD_REPO_ID,
        num_queries=len(RERANK_QUERY_INDEX), num_passages=f"{NUM_PASSAGES:,}", first_stage_recall=FIRST_STAGE_RECALL, recall=evaluated.recall_at(10),
        first_stage_mean_reciprocal_rank=FIRST_STAGE_RERANK[RERANK_CANDIDATES].mean_reciprocal_rank(), mean_reciprocal_rank=evaluated.mean_reciprocal_rank(), coverage=evaluated.coverage(), random_recall=RANDOM_RECALL,
        smoke_note="スモークテストの値であり、意味を持たない。" if SMOKE_TEST else "本番の実行の値である。", list_like_share=f"{len(LIST_LIKE_ARTICLES) / len(CANDIDATE_ARTICLES):.0%}",
    )
    (directory / "README.md").write_text(card, encoding="utf-8")
    return {"evaluated": evaluated, "card": card, "files": sorted(p.name for p in directory.iterdir())}


with tempfile.TemporaryDirectory() as _temporary_directory:
    _built = build_card_and_files(Path(_temporary_directory))
    UPLOAD_FILES = _built["files"]
    assert UPLOAD_FILES == ["README.md", "config.json", "model_state.pt"], "トークナイザなど、同梱しないファイルが含まれている"
    print(
        f"{RUN_TAG}モデルカードの下書きと書き出したファイル {UPLOAD_FILES} を作成し、重みを読み込み直して評価した"
        f"(並べ替えの Recall@10 {_built['evaluated'].recall_at(10):.4f}、記録した値 {float(RUNS[UPLOAD_RUN]['hits'].mean()):.4f})"
    )
    print("--- モデルカードの先頭 ---")
    print("\n".join(_built["card"].splitlines()[:22]))

    UPLOAD_RECALL, UPLOAD_FIRST_STAGE_RECALL = _built["evaluated"].recall_at(10), FIRST_STAGE_RECALL
    UPLOAD_CONDITION_MET = bool(UPLOAD_RECALL > UPLOAD_FIRST_STAGE_RECALL)  # 公開の条件(判定ではない): 並べ替えた後の Recall@10 > 並べ替えなしの Recall@10
    print(
        f"{RUN_TAG}公開の条件(判定ではない): 読み込み直した D1 のシード 0 の Recall@10 {UPLOAD_RECALL:.4f} > 並べ替えなしの Recall@10 {UPLOAD_FIRST_STAGE_RECALL:.4f}: {UPLOAD_CONDITION_MET}"
    )
    if not UPLOAD_ARTIFACTS:
        print("UPLOAD_ARTIFACTS = False のため、アップロードは行わない(モデルカードの下書きの確認のみ)")
    elif not UPLOAD_CONDITION_MET:
        print(
            f"公開の条件を満たさないため、アップロードを行わない: 読み込み直した D1 のシード 0 の Recall@10 {UPLOAD_RECALL:.4f} が、並べ替えなしの Recall@10 {UPLOAD_FIRST_STAGE_RECALL:.4f} を上回らない"
        )
    else:
        try:
            from google.colab import userdata  # noqa: E402

            _token = userdata.get("HF_TOKEN")
        except Exception:  # noqa: BLE001  # Colab 以外・Secrets 未設定・取得失敗のいずれでも、停止せずスキップする
            _token = None
        if not _token:
            print("Colab Secrets から HF_TOKEN を取得できないため、アップロードをスキップする")
        else:
            from huggingface_hub import HfApi

            _api = HfApi(token=_token)
            _api.create_repo(UPLOAD_REPO_ID, repo_type="model", exist_ok=True, private=False)
            for _name in UPLOAD_FILES:
                _api.upload_file(path_or_fileobj=str(Path(_temporary_directory) / _name), path_in_repo=_name, repo_id=UPLOAD_REPO_ID, repo_type="model")
            _remote = sorted(_api.list_repo_files(UPLOAD_REPO_ID, repo_type="model"))
            _missing = [n for n in UPLOAD_FILES if n not in _remote]
            assert not _missing, f"アップロードしたはずのファイルがリポジトリにない: {_missing}"
            print(f"アップロード完了: {UPLOAD_REPO_ID} のファイル {_remote}(意図したファイルがすべて存在することを list_repo_files() で確認)")
print(f"アップロードの準備 {time.time() - _t0_upload:.1f} 秒")
```

    モデルカードの下書きと書き出したファイル ['README.md', 'config.json', 'model_state.pt'] を作成し、重みを読み込み直して評価した(並べ替えの Recall@10 0.1708、記録した値 0.1708)
    --- モデルカードの先頭 ---
    ---
    language: en
    license: mit
    tags:
    - ai-theories
    - cross-encoder
    - reranking
    - retrieval
    - scratch-implementation
    ---
    
    # ai-theories cross-encoder(英語、困難な負例で学習した並べ替えモデル)
    
    `ai-theories`(https://github.com/kojikojiprg/ai-theories)プロジェクトの成果物。
    008 の小型 GPT(`kojikojiprg/ai-theories-small-gpt-en`)を起点に、query と passage を連結して符号化し、終端の位置の出力から並べ替えのスコアを出す cross-encoder として、
    全パラメータを更新して学習したもの。第 1 段の検索器(`kojikojiprg/ai-theories-text-embedding-en`)の上位から採掘した困難な負例で学習した。
    スクラッチ実装であり、研究・教育目的のモデルである。品質保証は行っていない。商用・実運用での利用は想定しない。
    
    ## 由来
    
    - [025. ANN 検索とリランキング](https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/025_ann_search_and_reranking.ipynb)の
      条件 D1(困難な負例で学習した cross-encoder)のシード 0 の学習後の重み。結果を見て選んだものではなく、事前に固定した条件とシードである。
    公開の条件(判定ではない): 読み込み直した D1 のシード 0 の Recall@10 0.1708 > 並べ替えなしの Recall@10 0.2199: False
    公開の条件を満たさないため、アップロードを行わない: 読み込み直した D1 のシード 0 の Recall@10 0.1708 が、並べ替えなしの Recall@10 0.2199 を上回らない
    アップロードの準備 28.6 秒




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/025_ann_search_and_reranking.ipynb
