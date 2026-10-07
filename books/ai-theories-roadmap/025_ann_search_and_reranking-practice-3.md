---
title: "ANN 検索とリランキング / ANN Search and Reranking(実装・実験編 3/8)"
---

この記事は後編(実装・実験編 3/8)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/025_ann_search_and_reranking-practice-2)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/025_ann_search_and_reranking-practice-4)。

### 5.4 モデルの構築・単体テスト・不変条件の確認

**モデルの構築**: 並べ替えモデルは、`torch.manual_seed(25200 + s)`の後に、008 と同じ構成の小型 GPT(4 層、$d_{\mathrm{model}} = 256$、ヘッド数 8、RoPE、RMSNorm、SwiGLU、正規化前置、
埋め込みと出力層の重み共有、系列長 256、語彙サイズ 8192)を作り、008 の学習済みの重み(`kojikojiprg/ai-theories-small-gpt-en`の`main`)を読み込む。cross-encoder(D1・E2)はこれにスカラーのヘッド
(重みは標準偏差 0.02 の正規分布、バイアスは 0)を付け、dual encoder(D2)は 024 の C5 と同じ構成(平均プール・因果マスク。射影層なし)で包む。**全パラメータを更新する。**
本体の初期値はシードによらず同じで、シードが動かすのは学習の組と順序(とヘッドの初期値)だけである。

**単体テスト**(CPU、小さな次元。許容の誤差は、実際に計算される型の丸めの単位 x 値の大きさ から導き、固定の絶対値は使わない。比べる値がその型で計算されていることもアサーションで確かめる):

- k-means: 反復ごとの二乗誤差の和 $J$ が単調に減少する。割り当てが素朴な計算と一致する。空のクラスタの置き直しが働く。
- 全探索: 素朴な計算(完全な類似度行列の並べ替え)と一致する。
- 転置ファイル: **すべてのリストを探索すると全探索と一致する**。一括で評価する`search_batch_by_probe_counts()`が、1 つずつ探索する`search()`と、結果も数えた費用(重心との比較・走査したベクトル)も一致する。
- Product Quantization: **非対称距離計算の値が、復元したベクトルとの距離を直接計算した値と一致する**(対称距離計算も同様)。再採点で全候補を調べると全探索と一致する。
- HNSW: **探索の幅を件数いっぱいにすると、全探索と一致する**。接続の数の上限・自己ループなし・接続先がその層に属する・入口が最上層、という不変条件。層の番号の分布が $P(l \ge j) = M^{-j}$ と統計的に一致する。
  **原論文の Algorithm 2 をそのまま書いた素朴な参照実装と、探索の結果も数えた距離計算の回数も一致する**(`float64`の索引)。
- 並べ替えの指標: 順位の逆数(逆順位)が手で計算した値と一致し、候補に正例がない query は 0 になる(平均逆順位は、学習率の較正の指標)。**同点のスコアは正例に不利に数える**(正例と同点の負例が上位になる。全候補に同じスコアを返すモデルで、Recall@10 が 0 になる)。
- cross-encoder: 区切りと終端を含む入力の組み立て。因果マスクにより、スコアが終端の位置の出力だけで決まり、query の後ろ側の passage の途中の位置の出力が passage の後ろのトークンに依存しない。
- 学習の組: 同じシードで決定的、query と正例の系列が負例の種類によらず同じ、負例が同じ記事の passage を含まない、困難な負例が採掘した候補の中にある、正例が同じ記事の元の passage 以外。
- 学習ループ: 1 ステップ目の損失が、同じ組から手で計算した損失と一致する。学習率 0 の学習(更新がない)の訓練損失が、同じ組での学習前のモデルの損失(`compute_reranking_losses()`)と一致し、
  `compute_reranking_losses()`はモデルを変更しない。評価の関数が、組ごとの 1 回の順伝播(cross-encoder)・passage の事前計算(dual encoder)のどちらでも、バッチの大きさに依らない。

**不変条件**: 分割・コーパス・埋め込みが 024 の本番の値と一致する(5.3 節)、新しい`src/`のファイルが追加のみで、既存のファイルを変更・削除していない。


```python
_t0_checks = time.time()
# 並べ替えの指標: 順位 1・2・5 の逆順位が 1・1/2・1/5、候補に正例がない(順位 K + 1)query は 0(平均逆順位の元になる量。較正の指標)
_reciprocal_check = RerankingResult(rank=np.array([1, 2, 5, 51]), normalized_discounted_cumulative_gain_at_10=np.zeros(4), num_candidates=50)
assert np.allclose(_reciprocal_check.reciprocal_ranks(), [1.0, 0.5, 0.2, 0.0], rtol=0, atol=4 * np.finfo(np.float64).eps)
assert math.isclose(_reciprocal_check.mean_reciprocal_rank(), (1.0 + 0.5 + 0.2) / 4, rel_tol=4 * np.finfo(np.float64).eps)


class _ConstantScoreModel(torch.nn.Module):
    # 全候補に同じスコアを返すモデル(崩れたモデルの極端な場合)
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, queries: torch.Tensor, passages: torch.Tensor) -> torch.Tensor:
        return torch.zeros(queries.size(0), device=queries.device)


# 同点は正例に不利に数える: 全候補に同じスコアを返すモデルでは、第 1 段の順位が同点の解決に使われず、Recall@10 が 0 になる(1 query の正例は最大 11 個 < K - 9 = 41)
_constant_queries = RERANK_QUERY_INDEX[:200]
_constant_scores = score_candidates(_ConstantScoreModel(), "cross_encoder", QUERY_TOKENS_CPU[_constant_queries], PASSAGE_TOKENS_CPU, RERANK_CANDIDATE_IDS[:200], 64, False)
assert (_constant_scores == _constant_scores[:, :1]).all()
_constant_result = rank_reranked_candidates(
    _constant_scores, RERANK_CANDIDATE_IDS[:200], QUERY_ARTICLES[_constant_queries], PASSAGE_ARTICLES, NUM_POSITIVES_BY_QUERY[_constant_queries]
)
assert _constant_result.coverage() > 0 and _constant_result.recall_at(10) == 0.0, "同点を正例に不利に数えていない"
# 較正の選択: P1(a) を満たさない学習率は、指標が最大でも選ばない。同点は小さい学習率。候補がなければ None
assert best_eligible_index([1.0, 2.0, 4.0], [False, True, True], [0.9, 0.2, 0.3]) == 2
assert best_eligible_index([1.0, 2.0, 4.0], [True, False, True], [0.5, 0.9, 0.5]) == 0
assert best_eligible_index([1.0, 2.0, 4.0], [False, False, False], [0.9, 0.2, 0.3]) is None
# 段階 6 の格子(3 点)は、5 点の格子の中心の 1/4・1・4 倍
assert all(math.isclose(a, b) for a, b in zip(learning_rate_grid_for("D1", 3), (6e-5, 2.4e-4, 9.6e-4), strict=True)) and len(learning_rate_grid_for("D1", 5)) == 5
# 手で作る例: 正例が先頭の候補と同点の負例 1 つ -> 順位 2(同点の負例が上位)
_tie_check = rank_reranked_candidates(np.array([[1.0, 1.0, 0.0]]), np.array([[0, 1, 2]]), np.array([0]), np.array([0, 1, 1]), np.array([1]))
assert _tie_check.rank.tolist() == [2], _tie_check.rank
EPSILON_FP16 = float(torch.finfo(torch.float16).eps)


def max_abs_difference(a, b) -> float:
    return float((torch.as_tensor(a).detach().cpu().double() - torch.as_tensor(b).detach().cpu().double()).abs().max())  # MPS は FP64 を扱えないので CPU で計算する


def build_model(condition: str, seed_index: int):
    # 008 の重みから始める並べ替えモデル(条件のモデルの種類。初期値の乱数: ヘッドの初期化)
    torch.manual_seed(INIT_SEED_BASE + seed_index)
    backbone = build_gpt(REFERENCE_CONFIG)
    backbone.load_state_dict(REFERENCE_STATE)
    return build_reranker(CONDITIONS[condition]["kind"], backbone, SEPARATOR_TOKEN_ID, TERMINAL_TOKEN_ID, HEAD_INIT_STD)


# --- k-means ---
_rng = np.random.default_rng(0)
_centers = _rng.normal(size=(12, 16))
_points = torch.from_numpy(np.concatenate([c + 0.4 * _rng.normal(size=(50, 16)) for c in _centers]).astype(np.float32))
_result = fit_kmeans(_points, 12, 15, seed=1)
assert all(b <= a * (1 + 4 * EPSILON_FP32) for a, b in itertools.pairwise(_result.inertia_history)), _result.inertia_history  # 単調減少(丸めの範囲)
_brute = torch.cdist(_points.double(), _result.centroids.double()).pow(2)
assert torch.equal(_brute.argmin(1), _result.assignments)
assert abs(inertia(_points, _result.centroids) - float(_brute.min(1).values.sum())) <= 1e-4 * float(_brute.min(1).values.sum())
assert len(torch.unique(fit_kmeans(_points[:20], 20, 3, seed=0).assignments)) >= 15  # 空のクラスタの置き直しで、ほぼ全部の重心が使われる
print(f"k-means: 二乗誤差の和が単調に減少({_result.inertia_history[0]:.1f} -> {_result.inertia_history[-1]:.1f})、割り当てが素朴な計算と一致、空のクラスタの置き直しが働く: OK")

# --- 全探索・転置ファイル・Product Quantization ---
_data = torch.nn.functional.normalize(_points, dim=1)
_queries = torch.nn.functional.normalize(_points[::7][:40] + 0.1 * torch.randn(40, 16, generator=torch.Generator().manual_seed(2)), dim=1)
_source = np.array([int(torch.argmax(_queries[i] @ _data.t())) for i in range(len(_queries))])  # query の元の passage にあたるもの(最も近いもの)
_truth, _ = exact_search(_queries, _data, 10, excluded=_source)
_similarity = _queries @ _data.t()
_similarity[torch.arange(len(_queries)), torch.from_numpy(_source)] = float("-inf")
assert (_truth == _similarity.topk(10, dim=1).indices.numpy()).all()
assert recall_against_ground_truth(_truth, _truth).min() == 1.0
_inverted_file = InvertedFileIndex(_data, 10, num_iterations=10, seed=3)
_batch = _inverted_file.search_batch_by_probe_counts(_queries, 10, [1, 3, 10], excluded=_source)
assert (_batch[10][0] == _truth).all(), "すべてのリストを探索しても全探索と一致しない"
for _p in (1, 3):
    for _i in range(len(_queries)):
        _ids, _centroid_comparisons, _scanned = _inverted_file.search(_queries[_i], 10, _p, excluded=int(_source[_i]), count_cost=False)
        assert (_ids == _batch[_p][0][_i]).all() and _centroid_comparisons + _scanned == _batch[_p][1][_i], (_p, _i)
print("転置ファイル: すべてのリストを探索すると全探索と一致、一括の評価が 1 つずつの探索と結果・費用(重心との比較 + 走査したベクトル)で一致: OK")
_quantizer = ProductQuantizer(16, 4, 16)
_quantizer.fit(_data, 10, seed=5)
_product_quantization_index = ProductQuantizationIndex(_quantizer, _data)
for _asymmetric in (True, False):
    _reference = _product_quantization_index.reconstructed_distances(_queries, _asymmetric)  # 復元したベクトルとの距離を直接計算した値(FP32)
    _tables = _quantizer.asymmetric_table(_queries) if _asymmetric else torch.stack([_quantizer.symmetric_table[j][_quantizer.encode(_queries).long()[:, j]] for j in range(4)], dim=1)
    _estimated = _product_quantization_index._estimated_distances(_tables)
    assert _estimated.dtype == _reference.dtype == torch.float32  # 比べる値が FP32 で計算されている
    _tolerance = 4 * 4 * EPSILON_FP32 * float(_reference.max())  # 和の項数(m = 4)x 丸めの単位 x 値の大きさ x 4
    assert max_abs_difference(_estimated, _reference) <= _tolerance, (_asymmetric, max_abs_difference(_estimated, _reference), _tolerance)
_ids_all, _ = _product_quantization_index.search_asymmetric(_queries, 10, excluded=_source, num_rescored=len(_data) - 1)
assert (_ids_all == _truth).all(), "再採点で全候補を調べても全探索と一致しない"
assert _product_quantization_index.counter.get("construction_table_entries") == 4 * 16 * 16
print("Product Quantization: 非対称・対称距離計算の値が、復元したベクトルとの距離を直接計算した値と一致(FP32、許容は 項数 x 丸めの単位 x 値の大きさの 4 倍)、再採点で全候補を調べると全探索と一致: OK")

# --- HNSW ---
_vectors = _data.numpy().astype(np.float64)
_queries64 = _queries.numpy().astype(np.float64)
_HNSW_REFERENCE_MISMATCHES = {}
for _heuristic in (True, False):
    _hnsw = HierarchicalNavigableSmallWorld(_vectors, max_connections=5, construction_search_width=40, seed=0, use_heuristic=_heuristic, dtype=np.float64)
    _hnsw.add(len(_vectors))
    assert _hnsw.num_inserted == len(_vectors) and sorted(_hnsw.insertion_order.tolist()) == list(range(len(_vectors)))
    for _layer in range(_hnsw.top_level + 1):
        assert (_hnsw.degree[_layer] <= _hnsw.capacity(_layer)).all()
        for _node in np.nonzero(_hnsw.levels >= _layer)[0][:300]:
            _neighbors = _hnsw.neighbors(_layer, int(_node))
            assert len(set(_neighbors.tolist())) == len(_neighbors) and int(_node) not in _neighbors.tolist()  # 重複なし・自己ループなし
            assert (_hnsw.levels[_neighbors] >= _layer).all()  # 接続先がその層に属する
    assert _hnsw.levels[_hnsw.entry_point] == _hnsw.top_level == _hnsw.levels.max()
    _mismatch = 0
    for _i in range(len(_queries)):  # 探索の幅を件数いっぱいにすると、全探索と一致する
        _ids, _, _ = _hnsw.search(_queries64[_i], 11, len(_vectors))
        _mismatch += int(not np.array_equal(np.sort(_ids[_ids != _source[_i]][:10]), np.sort(_truth[_i])))
    assert _mismatch == 0, f"探索の幅を件数いっぱいにしても全探索と一致しない query が {_mismatch} 個ある"

    def _naive_search_layer(index, q, entries, width, layer):  # 原論文の Algorithm 2 をそのまま書いた参照実装(集合とヒープ)
        import heapq

        visited = {e for e, _ in entries}
        candidates = [(d, e) for e, d in entries]
        heapq.heapify(candidates)
        found = [(-d, e) for e, d in entries]
        heapq.heapify(found)
        count = 0
        while candidates:
            distance_c, c = heapq.heappop(candidates)
            if distance_c > -found[0][0]:
                break
            for e in index.neighbors(layer, c).tolist():
                if e in visited:
                    continue
                visited.add(e)
                distance_e = 1.0 - float(np.dot(index.vectors[e], q))
                count += 1
                if distance_e < -found[0][0] or len(found) < width:
                    heapq.heappush(candidates, (distance_e, e))
                    heapq.heappush(found, (-distance_e, e))
                    if len(found) > width:
                        heapq.heappop(found)
        return sorted((-nd, e) for nd, e in found), count

    for _i in range(10):
        _q = _queries64[_i]
        _ep = _hnsw.entry_point
        _entries, _total = [(_ep, 1.0 - float(np.dot(_hnsw.vectors[_ep], _q)))], 1
        for _layer in range(_hnsw.top_level, 0, -1):
            _found, _count = _naive_search_layer(_hnsw, _q, _entries, 1, _layer)
            _total += _count
            _entries = [(_found[0][1], _found[0][0])]
        _found, _count = _naive_search_layer(_hnsw, _q, _entries, 30, 0)
        _total += _count
        _ids, _, _cost = _hnsw.search(_q, 10, 30)
        assert [e for _, e in _found[:10]] == _ids.tolist() and _total == _cost, (_heuristic, _i, _total, _cost)
    _HNSW_REFERENCE_MISMATCHES[_heuristic] = 0
_level_samples = HierarchicalNavigableSmallWorld(np.zeros((200_000, 2), dtype=np.float32), max_connections=5, seed=1).levels
for _j in (1, 2, 3):
    _p_hat, _p = float((_level_samples >= _j).mean()), 5.0 ** (-_j)
    assert abs(_p_hat - _p) <= 4 * math.sqrt(_p * (1 - _p) / len(_level_samples)), (_j, _p_hat, _p)  # 二項分布の標準誤差の 4 倍以内
print(
    "HNSW(ヒューリスティックあり・なしの両方): 探索の幅 = 件数で全探索と一致、接続の数の上限・自己ループなし・接続先がその層に属する・入口が最上層、"
    "原論文の Algorithm 2 の素朴な参照実装と探索の結果・距離計算の回数が一致(float64)、層の番号の分布が P(l >= j) = M^-j と一致(二項分布の標準誤差の 4 倍以内): OK"
)

# --- cross-encoder と並べ替えの学習・評価 ---
torch.manual_seed(7)
_backbone = build_gpt(REFERENCE_CONFIG)
_backbone.load_state_dict(REFERENCE_STATE)
_cross = CrossEncoder(_backbone, SEPARATOR_TOKEN_ID, TERMINAL_TOKEN_ID, HEAD_INIT_STD)
_q_tokens = QUERY_TOKENS_CPU[RERANK_QUERY_INDEX[:3]]
_p_tokens = PASSAGE_TOKENS_CPU[RERANK_CANDIDATE_IDS[:3, 0]]
_inputs = build_cross_encoder_inputs(_q_tokens, _p_tokens, SEPARATOR_TOKEN_ID, TERMINAL_TOKEN_ID)
assert _inputs.shape == (3, QUERY_LENGTH + 1 + PASSAGE_LENGTH + 1) and (_inputs[:, QUERY_LENGTH] == SEPARATOR_TOKEN_ID).all() and (_inputs[:, -1] == TERMINAL_TOKEN_ID).all()
assert torch.equal(_inputs[:, :QUERY_LENGTH], _q_tokens) and torch.equal(_inputs[:, QUERY_LENGTH + 1 : -1], _p_tokens)
with torch.no_grad():
    _scores = _cross(_q_tokens, _p_tokens)
    assert _scores.dtype == torch.float32 and _scores.shape == (3,)
    from src.models.text_embedding import compute_hidden_states  # noqa: E402

    _hidden = compute_hidden_states(_backbone, _inputs, "causal")
    assert torch.equal(_scores, _cross.head(_hidden[:, -1, :]).squeeze(-1)), "スコアが終端の位置の出力だけから計算されていない"
    _changed = _inputs.clone()
    _changed[:, QUERY_LENGTH + 1 + 64 :] = torch.randint(0, 8192, (3, PASSAGE_LENGTH + 1 - 64), generator=torch.Generator().manual_seed(3))
    _hidden_changed = compute_hidden_states(_backbone, _changed, "causal")
    assert torch.equal(_hidden[:, : QUERY_LENGTH + 1 + 64], _hidden_changed[:, : QUERY_LENGTH + 1 + 64]), "因果マスクで passage の後ろのトークンに依存している"
    assert not torch.equal(_hidden[:, -1], _hidden_changed[:, -1]), "終端の位置の出力が passage の後ろのトークンに依存しない"
    _query_changed = _inputs.clone()
    _query_changed[:, 5] = (_query_changed[:, 5] + 1) % 8192
    assert not torch.equal(compute_hidden_states(_backbone, _query_changed, "causal")[:, -1], _hidden[:, -1]), "終端の位置の出力が query に依存しない"
print("cross-encoder: 入力の組み立て(query・区切り・passage・終端)、スコアが終端の位置の出力だけから計算される、因果マスクで終端以外の位置は後ろのトークンに依存しない、終端の位置は query と passage の両方に依存する: OK")
```

    k-means: 二乗誤差の和が単調に減少(6467.5 -> 2460.4)、割り当てが素朴な計算と一致、空のクラスタの置き直しが働く: OK
    転置ファイル: すべてのリストを探索すると全探索と一致、一括の評価が 1 つずつの探索と結果・費用(重心との比較 + 走査したベクトル)で一致: OK
    Product Quantization: 非対称・対称距離計算の値が、復元したベクトルとの距離を直接計算した値と一致(FP32、許容は 項数 x 丸めの単位 x 値の大きさの 4 倍)、再採点で全候補を調べると全探索と一致: OK
    HNSW(ヒューリスティックあり・なしの両方): 探索の幅 = 件数で全探索と一致、接続の数の上限・自己ループなし・接続先がその層に属する・入口が最上層、原論文の Algorithm 2 の素朴な参照実装と探索の結果・距離計算の回数が一致(float64)、層の番号の分布が P(l >= j) = M^-j と一致(二項分布の標準誤差の 4 倍以内): OK
    cross-encoder: 入力の組み立て(query・区切り・passage・終端)、スコアが終端の位置の出力だけから計算される、因果マスクで終端以外の位置は後ろのトークンに依存しない、終端の位置は query と passage の両方に依存する: OK



```python
# --- 学習の組(並べ替え) ---
_small_schedule = sample_reranking_schedule(
    TRAIN_QUERY_INDEX, MINED_CANDIDATES, QUERY_ARTICLES, QUERY_SOURCES, PASSAGE_ARTICLES, TRAIN_PASSAGE_INDEX, 50, BATCH_QUERIES, NUM_NEGATIVES, seed=5
)
_again = sample_reranking_schedule(
    TRAIN_QUERY_INDEX, MINED_CANDIDATES, QUERY_ARTICLES, QUERY_SOURCES, PASSAGE_ARTICLES, TRAIN_PASSAGE_INDEX, 50, BATCH_QUERIES, NUM_NEGATIVES, seed=5
)
_other = sample_reranking_schedule(
    TRAIN_QUERY_INDEX, MINED_CANDIDATES, QUERY_ARTICLES, QUERY_SOURCES, PASSAGE_ARTICLES, TRAIN_PASSAGE_INDEX, 50, BATCH_QUERIES, NUM_NEGATIVES, seed=6
)
for _name in ("query", "positive", "hard_negatives", "random_negatives"):
    assert np.array_equal(getattr(_small_schedule, _name), getattr(_again, _name)), f"学習の組がシードで決定的でない({_name})"
assert _small_schedule.digest() == _again.digest() != _other.digest()
assert (_small_schedule.query.shape, _small_schedule.hard_negatives.shape) == ((50, BATCH_QUERIES), (50, BATCH_QUERIES, NUM_NEGATIVES))
_train_query_set = set(TRAIN_QUERY_INDEX.tolist())
assert all(int(q) in _train_query_set for q in _small_schedule.query.ravel()), "学習用の記事の外の query が入っている"
_usable_query_set = set(TRAIN_QUERY_INDEX[USABLE_TRAIN_QUERY_MASK].tolist())
assert all(int(q) in _usable_query_set for q in _small_schedule.query.ravel()), "採掘の上位に同じ記事の passage がない query(学習に使えない query)が入っている"
assert (_small_schedule.num_usable_queries, _small_schedule.num_train_queries) == (NUM_USABLE_TRAIN_QUERIES, len(TRAIN_QUERY_INDEX))
_articles = QUERY_ARTICLES[_small_schedule.query]  # (T, B)
assert (PASSAGE_ARTICLES[_small_schedule.positive] == _articles).all() and (_small_schedule.positive != QUERY_SOURCES[_small_schedule.query]).all(), "正例が同じ記事の元の passage 以外でない"
for _kind in ("hard", "random"):
    _negatives = _small_schedule.negatives(_kind)
    assert (PASSAGE_ARTICLES[_negatives] != _articles[:, :, None]).all(), f"{_kind} の負例に同じ記事の passage が入っている"
    assert all(_train_passage_set[_negatives.ravel()]), "負例が学習用の記事の passage の外にある"
    assert all(len(set(row)) == NUM_NEGATIVES for row in _negatives.reshape(-1, NUM_NEGATIVES).tolist()), "負例に重複がある"
_mined_of_query = {int(q): MINED_CANDIDATES[i] for i, q in enumerate(TRAIN_QUERY_INDEX)}
assert all(
    set(_small_schedule.hard_negatives[t, b].tolist()) <= set(_mined_of_query[int(_small_schedule.query[t, b])].tolist())
    for t in range(_small_schedule.query.shape[0]) for b in range(BATCH_QUERIES)
) or _small_schedule.num_fallback_queries > 0, "困難な負例が採掘した候補の中にない"
assert all(
    int(_small_schedule.positive[t, b]) in set(_mined_of_query[int(_small_schedule.query[t, b])].tolist())
    for t in range(_small_schedule.query.shape[0]) for b in range(BATCH_QUERIES)
), "正例が採掘の上位 K 件の中にない(推論時の候補の分布と揃っていない)"
# query と正例の系列は、負例の種類によらず同じ(負例の乱数の生成器が別)
assert (_small_schedule.num_mined_considered, _small_schedule.num_same_article_removed) == (50 * BATCH_QUERIES * MINING_DEPTH, _small_schedule.num_same_article_removed)
print(
    f"学習の組: 50 ステップ x {BATCH_QUERIES} query で、シードで決定的(異なるシードでは異なる)、query は学習に使える query(採掘の上位に同じ記事の passage がある)、正例は採掘の上位の中の同じ記事の元の passage 以外、"
    f"負例(困難・ランダムとも)は同じ記事の passage を含まず学習用の記事の passage で重複なし、困難な負例は採掘した候補の中。"
    f"採掘の上位 {MINING_DEPTH} 件から除いた同じ記事の passage: {_small_schedule.num_same_article_removed:,} / {_small_schedule.num_mined_considered:,} 件、"
    f"除くと足りずにランダムな負例で補った組: {_small_schedule.num_fallback_queries}: OK"
)

# --- 学習ループ: 1 ステップ目の損失が、同じ組から手で計算した損失と一致する。学習率 0 の学習の損失が、学習前のモデルの損失と一致する(CPU、FP32)---
_tiny_schedule = sample_reranking_schedule(
    TRAIN_QUERY_INDEX, MINED_CANDIDATES, QUERY_ARTICLES, QUERY_SOURCES, PASSAGE_ARTICLES, TRAIN_PASSAGE_INDEX, 3, 2, 3, seed=11
)
for _condition in CONDITIONS:
    _kind, _negative_kind = CONDITIONS[_condition]["kind"], CONDITIONS[_condition]["negatives"]
    _model = build_model(_condition, 0)
    _queries = QUERY_TOKENS_CPU[_tiny_schedule.query[0]]
    _candidates = PASSAGE_TOKENS_CPU[np.concatenate([_tiny_schedule.positive[0][:, None], _tiny_schedule.negatives(_negative_kind)[0]], axis=1)]
    _model.eval()
    with torch.no_grad():  # 手で計算: 組ごとにスコアを求め、softmax 交差エントロピーを閉形式で計算する
        if _kind == "cross_encoder":
            _manual_scores = torch.stack([_model(_queries[i : i + 1].expand(4, -1), _candidates[i]) for i in range(2)])
        else:
            _q = torch.nn.functional.normalize(_model.pooled(_queries), dim=-1)
            _manual_scores = torch.stack([torch.nn.functional.normalize(_model.pooled(_candidates[i]), dim=-1) @ _q[i] for i in range(2)]) / TEMPERATURE
        _manual = float(torch.stack([-(_manual_scores[i, 0] - torch.logsumexp(_manual_scores[i], dim=0)) for i in range(2)]).mean())
    _model.train()
    _before_hash = state_hash(_model.state_dict())
    _measured = compute_reranking_losses(_model, _kind, QUERY_TOKENS_CPU, PASSAGE_TOKENS_CPU, _tiny_schedule, _negative_kind, range(3), TEMPERATURE)
    assert state_hash(_model.state_dict()) == _before_hash and _model.training, "compute_reranking_losses() がモデルまたはモードを変更した"
    assert _measured.shape == (3,) and _measured.dtype == np.float64
    assert abs(_measured[0] - _manual) <= 16 * EPSILON_FP32 * abs(_manual) * 8, (_condition, _measured[0], _manual)  # exp・log の合成 + softmax の実装の差(約 8 個の演算)
    _history = train_reranker(
        _model, _kind, QUERY_TOKENS_CPU, PASSAGE_TOKENS_CPU, _tiny_schedule, _negative_kind, 0.0, 1, 0.0, WEIGHT_DECAY, GRADIENT_CLIP_THRESHOLD, TEMPERATURE
    )
    assert state_hash(_model.state_dict()) == _before_hash, "学習率 0 の学習でモデルが変化した"
    assert np.allclose(_history["loss"], _measured, rtol=16 * EPSILON_FP32, atol=0.0), (_condition, _history["loss"], _measured)
    assert math.isfinite(_history["loss"][0])
print("学習ループ: 1 ステップ目の損失が、同じ組から手で計算した閉形式の損失と一致(3 つの条件)、学習率 0 の学習(更新がない)の訓練損失が学習前のモデルの損失(compute_reranking_losses())と一致し、モデルとモードを変更しない: OK")

# --- 評価: スコアの計算がバッチの大きさに依らない(実行するデバイス)/ cross-encoder は passage の事前計算が要らず、dual encoder は候補の passage の重複を 1 回にまとめても同じ ---
_check_queries = QUERY_TOKENS[torch.as_tensor(RERANK_QUERY_INDEX[:24], device=device)]
_check_candidates = RERANK_CANDIDATE_IDS[:24, :10]
for _condition in ("D1", "D2"):
    _kind = CONDITIONS[_condition]["kind"]
    _model = build_model(_condition, 0).to(device)
    _wide = score_candidates(_model, _kind, _check_queries, PASSAGE_TOKENS, _check_candidates, 240, USE_FP16_AUTOCAST)
    _repeat = float(np.abs(_wide - score_candidates(_model, _kind, _check_queries, PASSAGE_TOKENS, _check_candidates, 240, USE_FP16_AUTOCAST)).max())
    _narrow = float(np.abs(_wide - score_candidates(_model, _kind, _check_queries, PASSAGE_TOKENS, _check_candidates, 7, USE_FP16_AUTOCAST)).max())
    _unit = (EPSILON_FP16 if USE_FP16_AUTOCAST else EPSILON_FP32) * float(np.abs(_wide).max())
    _tolerance = 16 * max(_repeat, _unit)
    assert _wide.dtype == np.float32 and _narrow <= _tolerance, (_condition, _narrow, _tolerance)
    print(f"  {_condition}: スコアのバッチの大きさへの依存(240 と 7 の最大差 {_narrow:.2e}、同じ経路の 2 回の差 {_repeat:.2e}、丸めの単位 x 値の大きさ {_unit:.2e}、許容 {_tolerance:.2e}): OK")
    del _model
empty_device_cache()

# --- 既存モジュールの後方互換性: 参照コミット(024 の完成)以降、src/・scripts/ の既存のファイルに変更も削除もない(追加のみ) ---
_REFERENCE_COMMIT = "b5cded4"  # 025 の変更を始める前の最後のコミット
_changed = subprocess.run(["git", "diff", "--name-only", "--diff-filter=MD", _REFERENCE_COMMIT, "--", "src", "scripts"], capture_output=True, text=True)
assert _changed.returncode == 0, _changed.stderr
assert _changed.stdout.strip() == "", f"既存のファイルが変更・削除されている: {_changed.stdout}"
_existing = subprocess.run(["git", "ls-tree", "-r", "--name-only", _REFERENCE_COMMIT, "src", "scripts"], capture_output=True, text=True).stdout.split()
print(f"既存モジュールの後方互換性: 参照コミット {_REFERENCE_COMMIT} の src/・scripts/ の既存のファイル {len(_existing)} 個に、変更も削除もない(025 は新しいファイルの追加のみ): OK")
CHECK_SECONDS = time.time() - _t0_checks
print(f"単体テストと不変条件の確認 {CHECK_SECONDS:.1f} 秒")
```

    学習の組: 50 ステップ x 8 query で、シードで決定的(異なるシードでは異なる)、query は学習に使える query(採掘の上位に同じ記事の passage がある)、正例は採掘の上位の中の同じ記事の元の passage 以外、負例(困難・ランダムとも)は同じ記事の passage を含まず学習用の記事の passage で重複なし、困難な負例は採掘した候補の中。採掘の上位 50 件から除いた同じ記事の passage: 1,145 / 20,000 件、除くと足りずにランダムな負例で補った組: 0: OK
    学習ループ: 1 ステップ目の損失が、同じ組から手で計算した閉形式の損失と一致(3 つの条件)、学習率 0 の学習(更新がない)の訓練損失が学習前のモデルの損失(compute_reranking_losses())と一致し、モデルとモードを変更しない: OK
      D1: スコアのバッチの大きさへの依存(240 と 7 の最大差 1.59e-03、同じ経路の 2 回の差 0.00e+00、丸めの単位 x 値の大きさ 9.67e-04、許容 1.55e-02): OK
      D2: スコアのバッチの大きさへの依存(240 と 7 の最大差 7.74e-05、同じ経路の 2 回の差 0.00e+00、丸めの単位 x 値の大きさ 9.22e-04、許容 1.48e-02): OK
    既存モジュールの後方互換性: 参照コミット b5cded4 の src/・scripts/ の既存のファイル 71 個に、変更も削除もない(025 は新しいファイルの追加のみ): OK
    単体テストと不変条件の確認 12.9 秒


### 5.5 索引の構築・並べ替えの学習と評価のヘルパー

- **近似最近傍探索**: シード $s$ ごとに、`draw_levels_and_insertion_order()`(HNSW が使うものと同じ)でコーパスの passage の並べ替え(挿入の順序)を引き、先頭 $N$ 個を $N$ の部分集合とする
  (入れ子。同じシードの HNSW・転置ファイル・Product Quantization は、最大の $N$ で **同じ部分集合** を索引にする)。正解は、その部分集合での全探索の上位 10 件(query の元の passage を除く)。
  HNSW は先頭から挿入していき、$N$ の水準ごとに、その時点の索引を格子で掃引する(`run_hnsw_seed()`)。転置ファイルは最大の $N$ で、リスト数 5 通りのそれぞれを掃引する(`run_inverted_file_seed()`)。
  Product Quantization は最大の $N$ で、非対称・対称距離計算の recall を測る(`run_product_quantization_seed()`)。
- **並べ替え**: `train_run()`は、条件・シード・学習率・ステップ数を受け取って学習し、較正の query(検証用の記事)での並べ替えの Recall@10 と平均逆順位を記録する。本番の学習(`main=True`)は、
  並べ替えの query(評価用の記事)での query ごとの順位と、診断量(ランダムな候補の中での指標など)も記録する。モデルは評価の直後に破棄する(D1 のシード 0 の学習後の重みを除く)。


```python
# ---------------- 近似最近傍探索 ----------------
ANN_QUERIES_NP = QUERY_EMBEDDINGS_NP[ANN_QUERY_INDEX]
ANN_QUERIES_TENSOR = torch.from_numpy(ANN_QUERIES_NP)
ANN_QUERY_ARTICLES = QUERY_ARTICLES[ANN_QUERY_INDEX]
ANN_TRUTH_CACHE: dict[tuple[int, int], tuple] = {}


def ann_ordering(seed_index: int) -> np.ndarray:
    # シード s のコーパスの passage の並べ替え(HNSW の挿入の順序と同じ。入れ子の部分集合は先頭から取る)
    return draw_levels_and_insertion_order(NUM_PASSAGES, HNSW_MAX_CONNECTIONS, HNSW_SEED_BASE + seed_index)[1]


def ann_truth(seed_index: int, subset: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 部分集合での全探索の上位 10 件(要素の番号)、部分集合の中での query の元の passage の位置、部分集合に含まれる query の元の passage の要素の番号(含まれなければ -1)
    key = (seed_index, len(subset))
    if key not in ANN_TRUTH_CACHE:
        positions = positions_in_subset(subset, NUM_PASSAGES)
        local_excluded = positions[QUERY_SOURCES[ANN_QUERY_INDEX]]
        truth_local, _ = exact_search(ANN_QUERIES_TENSOR, PASSAGE_EMBEDDINGS[torch.from_numpy(subset)], NEIGHBORS, excluded=local_excluded)
        excluded = np.where(local_excluded >= 0, QUERY_SOURCES[ANN_QUERY_INDEX], -1)
        ANN_TRUTH_CACHE[key] = (subset[truth_local], local_excluded, excluded)
    return ANN_TRUTH_CACHE[key]


def run_hnsw_seed(seed_index: int, n_max: int, flat_diagnostic: bool = False, keep_index: bool = False) -> dict:
    # シード s の HNSW を、挿入の順序の先頭から N の水準ごとに挿入しながら、各水準で格子を掃引する
    sizes = size_levels_for(n_max)
    index = HierarchicalNavigableSmallWorld(
        PASSAGE_EMBEDDINGS_NP, HNSW_MAX_CONNECTIONS, HNSW_CONSTRUCTION_WIDTH, seed=HNSW_SEED_BASE + seed_index
    )
    result = {"sizes": sizes, "sweeps": {}, "flat_sweeps": {}, "layer_sizes": {}, "construction_distances": {}, "build_seconds": 0.0, "sweep_seconds": 0.0}
    entry_rng = np.random.default_rng([ANN_QUERY_SEED, seed_index])
    for n in sizes:
        start = time.time()
        index.add(n - index.num_inserted)
        result["build_seconds"] += time.time() - start
        subset = index.inserted_elements()
        truth, _, excluded = ann_truth(seed_index, subset)
        start = time.time()
        result["sweeps"][n] = sweep_hnsw_search_width(index, ANN_QUERIES_NP, excluded, truth, HNSW_GRID, NEIGHBORS, STOP_RECALL)
        result["sweep_seconds"] += time.time() - start
        if flat_diagnostic:  # 階層の効果の診断: 最下層だけを、ランダムな入口(挿入済みの要素)から探索する
            entries = subset[entry_rng.integers(0, len(subset), size=len(ANN_QUERIES_NP))]
            result["flat_sweeps"][n] = sweep_hnsw_search_width(
                index, ANN_QUERIES_NP, excluded, truth, HNSW_GRID, NEIGHBORS, STOP_RECALL, use_hierarchy=False, entry_elements=entries
            )
        result["layer_sizes"][n] = index.layer_sizes()
        result["construction_distances"][n] = index.counter.get("construction")
    result["index"] = index if keep_index else None
    return result


def run_inverted_file_seed(seed_index: int, n_max: int, keep_index: bool = False) -> dict:
    # 最大の N の部分集合(HNSW のシード s と同じ)で、リスト数 5 通りのそれぞれの転置ファイルを掃引する
    subset = ann_ordering(seed_index)[:n_max]
    truth, local_excluded, _ = ann_truth(seed_index, subset)
    vectors = PASSAGE_EMBEDDINGS[torch.from_numpy(subset)]
    result = {"num_lists": {}, "sweeps": {}, "fit_seconds": {}, "list_size_stats": {}, "empty_events": {}, "indices": {}, "subset": subset}
    for num_lists in inverted_file_list_counts_for(n_max):
        start = time.time()
        index = InvertedFileIndex(vectors, num_lists, INVERTED_FILE_KMEANS_ITERATIONS, seed=INVERTED_FILE_SEED_BASE + seed_index)
        result["fit_seconds"][num_lists] = time.time() - start
        result["sweeps"][num_lists] = sweep_inverted_file_probes(index, ANN_QUERIES_TENSOR, local_excluded, subset, truth, INVERTED_FILE_GRID, NEIGHBORS, STOP_RECALL)
        assert max(result["sweeps"][num_lists]["grid"]) <= num_lists, "探索するリストの数がリスト数を超えている"  # 5 点のリスト数のどれでも p <= K
        sizes = index.list_sizes.numpy()
        result["list_size_stats"][num_lists] = (int(sizes.min()), float(sizes.mean()), int(sizes.max()))
        result["empty_events"][num_lists] = index.num_empty_cluster_events
        result["num_lists"][num_lists] = num_lists
        if keep_index:
            result["indices"][num_lists] = index
    return result


def product_quantization_distance_error_diagnostics(product_quantization_index: ProductQuantizationIndex, subset_vectors: torch.Tensor, query_vectors: torch.Tensor, num_pairs: int = 20_000) -> dict:
    # 距離の二乗誤差の診断量: 非対称・対称距離計算の推定の二乗距離と真の二乗距離の差(平均 = 偏り、平均二乗誤差)と、量子化の平均二乗誤差 D
    rng = np.random.default_rng(0)
    q_ids, y_ids = rng.integers(0, len(query_vectors), num_pairs), rng.integers(0, len(subset_vectors), num_pairs)
    x, y = query_vectors[q_ids], subset_vectors[y_ids]
    exact = (x - y).pow(2).sum(dim=1).double()
    quantizer = product_quantization_index.quantizer
    q_x, q_y = quantizer.decode(quantizer.encode(x)), quantizer.decode(quantizer.encode(y))
    asymmetric = (x - q_y).pow(2).sum(dim=1).double()
    symmetric = (q_x - q_y).pow(2).sum(dim=1).double()
    return {
        "D": quantizer.quantization_mean_squared_error(subset_vectors),
        "query_error": float((x - q_x).double().pow(2).sum(dim=1).mean()),  # E ||e_x||^2(query の量子化の誤差)
        "bias_asymmetric": float((asymmetric - exact).mean()), "bias_symmetric": float((symmetric - exact).mean()),
        "mse_asymmetric": float((asymmetric - exact).pow(2).mean()), "mse_symmetric": float((symmetric - exact).pow(2).mean()),
    }


def run_product_quantization_seed(seed_index: int, n_max: int, keep_index: bool = False) -> dict:
    # 最大の N の部分集合で符号帳を学習し、非対称・対称距離計算の recall(全件を走査、再採点なし)と、再採点の recall を測る
    subset = ann_ordering(seed_index)[:n_max]
    truth, local_excluded, _ = ann_truth(seed_index, subset)
    vectors = PASSAGE_EMBEDDINGS[torch.from_numpy(subset)]
    start = time.time()
    quantizer = ProductQuantizer(EMBEDDING_DIMENSION, PRODUCT_QUANTIZATION_SUBVECTORS, PRODUCT_QUANTIZATION_CODEWORDS)
    quantizer.fit(vectors, PRODUCT_QUANTIZATION_KMEANS_ITERATIONS, seed=PRODUCT_QUANTIZATION_SEED_BASE + seed_index)
    product_quantization_index = ProductQuantizationIndex(quantizer, vectors)
    result = {"fit_seconds": time.time() - start, "recall": {}, "rescored_recall": {}, "cost": {}, "code_bytes": quantizer.code_bytes}
    for name, search in (("asymmetric", product_quantization_index.search_asymmetric), ("symmetric", product_quantization_index.search_symmetric)):
        ids, costs = search(ANN_QUERIES_TENSOR, NEIGHBORS, excluded=local_excluded)
        result["recall"][name] = recall_against_ground_truth(subset[ids], truth)  # query ごと
        result["cost"][name] = {k: float(v.mean()) for k, v in costs.items()}
        result["rescored_recall"][name] = {}
        for rescored in PRODUCT_QUANTIZATION_RESCORE_COUNTS:
            ids_r, _ = search(ANN_QUERIES_TENSOR, NEIGHBORS, excluded=local_excluded, num_rescored=rescored)
            result["rescored_recall"][name][rescored] = recall_against_ground_truth(subset[ids_r], truth)
    result["diagnostics"] = product_quantization_distance_error_diagnostics(product_quantization_index, vectors, ANN_QUERIES_TENSOR)
    result["index"], result["subset"] = (product_quantization_index, subset) if keep_index else (None, subset)
    return result


# ---------------- 並べ替え ----------------
SCHEDULES: dict[tuple[int, int], object] = {}


def get_schedule(seed_index: int, num_steps: int):
    # 学習の組は条件によらずシードとステップ数だけで決まる(query と正例は負例の種類にも依らない)。同じものを使い回す
    key = (seed_index, num_steps)
    if key not in SCHEDULES:
        SCHEDULES[key] = sample_reranking_schedule(
            TRAIN_QUERY_INDEX, MINED_CANDIDATES, QUERY_ARTICLES, QUERY_SOURCES, PASSAGE_ARTICLES, TRAIN_PASSAGE_INDEX,
            num_steps, BATCH_QUERIES, NUM_NEGATIVES, seed=SCHEDULE_SEED_BASE + seed_index,
        )
    return SCHEDULES[key]


def evaluate_rerank(model, kind: str, query_indices: np.ndarray, candidate_ids: np.ndarray):
    scores = score_candidates(model, kind, QUERY_TOKENS[torch.as_tensor(query_indices, device=device)], PASSAGE_TOKENS, candidate_ids, SCORE_BATCH, USE_FP16_AUTOCAST)
    return rank_reranked_candidates(scores, candidate_ids, QUERY_ARTICLES[query_indices], PASSAGE_ARTICLES, NUM_POSITIVES_BY_QUERY[query_indices]), scores


# 診断量「ランダムな候補の中での指標」の候補: 先頭の RANDOM_CANDIDATE_QUERIES 個の並べ替えの query の、正例 1 個(同じ記事の元の passage 以外から)+ ランダムな負例 49 個(他の記事の passage)
def build_random_candidates() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(ANN_QUERY_SEED + 1)
    queries = RERANK_QUERY_INDEX[:RANDOM_CANDIDATE_QUERIES]
    candidates = np.empty((len(queries), RANDOM_CANDIDATES), dtype=np.int64)
    for row, query in enumerate(queries):
        first, last = PASSAGES_BY_ARTICLE[int(QUERY_ARTICLES[query])]
        members = np.array([m for m in range(first, last) if m != QUERY_SOURCES[query]])
        candidates[row, 0] = members[rng.integers(0, len(members))]
        chosen: set[int] = set()
        while len(chosen) < RANDOM_CANDIDATES - 1:
            draw = int(rng.integers(0, NUM_PASSAGES))
            if PASSAGE_ARTICLES[draw] != QUERY_ARTICLES[query]:
                chosen.add(draw)
        candidates[row, 1:] = sorted(chosen)
    return queries, candidates


RANDOM_CANDIDATE_QUERIES_INDEX, RANDOM_CANDIDATE_IDS = build_random_candidates()
assert (PASSAGE_ARTICLES[RANDOM_CANDIDATE_IDS[:, 0]] == QUERY_ARTICLES[RANDOM_CANDIDATE_QUERIES_INDEX]).all() and (PASSAGE_ARTICLES[RANDOM_CANDIDATE_IDS[:, 1:]] != QUERY_ARTICLES[RANDOM_CANDIDATE_QUERIES_INDEX][:, None]).all()

BASELINE_CONDITIONS = ("D2", "E2")  # 実験 D の基準は D2、実験 E の基準は E2(本番では、学習前のモデルの検証用の値を測る)
UNTRAINED_VALIDATION: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}  # (条件, シード) -> 学習前のモデルの較正の query ごとの (Recall@10 の 0/1、逆順位)。D2 はシードによらず同じ
SAVED_STATES: dict[tuple[str, int], dict] = {}  # アップロード用(D1 のシード 0)


def median_score_std(scores: np.ndarray) -> float:
    # 診断量: query ごとの、候補 K 件のスコアの標準偏差の中央値(定数に近い出力に崩れたモデルでは 0 に近づく。判定には使わない)
    return float(np.median(np.asarray(scores, dtype=np.float64).std(axis=1)))


def train_run(condition: str, seed_index: int, learning_rate: float, num_steps: int, main: bool, with_untrained: bool | None = None) -> dict:
    spec = CONDITIONS[condition]
    kind, negative_kind = spec["kind"], spec["negatives"]
    start = time.time()
    model = build_model(condition, seed_index).to(device)
    schedule = get_schedule(seed_index, num_steps)
    window = final_loss_window(num_steps)
    # 学習前のモデルの、最後の区間の組での損失(学習の成立 (a) の分母。同じ組での比較なので、バッチごとの難しさが打ち消される)
    initial_model_losses = compute_reranking_losses(
        model, kind, QUERY_TOKENS, PASSAGE_TOKENS, schedule, negative_kind, range(num_steps - window, num_steps), TEMPERATURE, USE_FP16_AUTOCAST
    )
    if with_untrained is None:  # 本番では基準の条件だけ。True を指定すればどの条件でも測る
        with_untrained = main and condition in BASELINE_CONDITIONS
    untrained = None
    if with_untrained:
        key = (condition, 0 if kind == "dual_encoder" else seed_index)
        if key not in UNTRAINED_VALIDATION:
            untrained_result, _ = evaluate_rerank(model, kind, CALIBRATION_QUERY_INDEX, CALIBRATION_CANDIDATE_IDS)
            UNTRAINED_VALIDATION[key] = (untrained_result.hits(10), untrained_result.reciprocal_ranks())
        untrained = UNTRAINED_VALIDATION[key]
    train_start = time.time()
    history = train_reranker(
        model, kind, QUERY_TOKENS, PASSAGE_TOKENS, schedule, negative_kind,
        peak_learning_rate=learning_rate, warmup_steps=warmup_steps_for(num_steps), min_learning_rate=learning_rate * MIN_LEARNING_RATE_RATIO,
        weight_decay=WEIGHT_DECAY, gradient_clip_threshold=GRADIENT_CLIP_THRESHOLD, temperature=TEMPERATURE,
        use_fp16_autocast=USE_FP16_AUTOCAST, init_loss_scale=INIT_LOSS_SCALE, loss_scale_growth_interval=LOSS_SCALE_GROWTH_INTERVAL,
    )
    train_seconds = time.time() - train_start
    assert len(history["loss"]) == num_steps
    train_loss = np.array(history["loss"], dtype=np.float64)
    record = {
        "condition": condition, "seed": seed_index, "learning_rate": learning_rate, "num_steps": num_steps, "schedule_hash": history["data_stream_hash"],
        "train_loss": train_loss.astype(np.float32), "initial_train_loss": float(train_loss[:window].mean()), "final_train_loss": float(train_loss[-window:].mean()),
        "initial_model_loss_on_final_window": float(initial_model_losses.mean()), "initial_model_losses_by_step": initial_model_losses, "finite": bool(np.isfinite(train_loss).all()),
        "skipped_steps": int(sum(history["step_skipped"])), "clip_rate": float(np.mean(history["gradient_clip_triggered"])), "final_loss_scale": float(history["loss_scale"][-1]),
        "untrained_validation_hits": None if untrained is None else untrained[0], "untrained_validation_reciprocal_ranks": None if untrained is None else untrained[1],
        "train_seconds": train_seconds,
    }
    validation, validation_scores = evaluate_rerank(model, kind, CALIBRATION_QUERY_INDEX, CALIBRATION_CANDIDATE_IDS)
    record["validation_score_std_median"] = median_score_std(validation_scores)  # 診断量(崩壊の検出。判定には使わない)
    record["validation_recall"] = validation.recall_at(10)
    record["validation_hits"] = validation.hits(10)
    record["validation_mean_reciprocal_rank"] = validation.mean_reciprocal_rank()
    record["validation_reciprocal_ranks"] = validation.reciprocal_ranks()
    if main:
        final, final_scores = evaluate_rerank(model, kind, RERANK_QUERY_INDEX, RERANK_CANDIDATE_IDS)
        record["hits"], record["rank"] = final.hits(10), final.rank
        record["mean_reciprocal_rank"], record["normalized_discounted_cumulative_gain"] = final.mean_reciprocal_rank(), final.mean_normalized_discounted_cumulative_gain_at_10()
        record["score_std_median"] = median_score_std(final_scores)
        random_scores = score_candidates(
            model, kind, QUERY_TOKENS[torch.as_tensor(RANDOM_CANDIDATE_QUERIES_INDEX, device=device)], PASSAGE_TOKENS, RANDOM_CANDIDATE_IDS, SCORE_BATCH, USE_FP16_AUTOCAST
        )
        positive_rank = 1 + (random_scores[:, 1:] >= random_scores[:, :1]).sum(axis=1)  # 同点は正例に不利に数える
        record["random_candidates"] = {"recall_at_1": float((positive_rank == 1).mean()), "recall_at_10": float((positive_rank <= 10).mean())}
        if (condition, seed_index) == UPLOAD_RUN:
            SAVED_STATES[UPLOAD_RUN] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    record["seconds"] = time.time() - start
    del model
    empty_device_cache()
    return record


def learning_loss_statistics(record: dict) -> dict:
    # 学習の成立 (a)(6.1 節)の量: 最後の区間の各ステップの、訓練損失と、同じ組での学習前のモデルの損失の差 d_t(学習前より下がれば負)。
    # 差の平均 mean と標準誤差 standard_error = (d_t の標準偏差) / sqrt(区間のステップ数)。更新がなければ d_t = 0
    window = len(record["initial_model_losses_by_step"])
    differences = record["train_loss"][-window:].astype(np.float64) - record["initial_model_losses_by_step"]
    mean = float(differences.mean())
    standard_error = float(differences.std(ddof=1) / math.sqrt(window)) if window >= 2 else float("nan")
    return {
        "mean": mean, "standard_error": standard_error, "ratio": record["final_train_loss"] / record["initial_model_loss_on_final_window"],
        "final_loss": record["final_train_loss"], "initial_loss": record["initial_model_loss_on_final_window"],
    }


def learning_failure_reasons(record: dict) -> list[str]:
    # 学習の成立 (a) を満たさない理由(空なら成立)。訓練損失がすべてのステップで有限で、(i)差の平均 <= -max(P1_SIGMA_MULTIPLIER x 標準誤差、数値の丸めの床)、かつ
    # (ii)最後の区間の訓練損失 < 一様な予測の損失 ln(1 + n)((i)だけだと、学習前に一様より高い損失を、定数に近い出力へ崩すことでも成り立ちうるため)。
    # 数値の丸めの床 = 16 x 実際に計算される型(FP16 の autocast なら FP16、そうでなければ FP32)の丸めの単位 x 学習前のモデルの損失の大きさ。
    # 更新がなければ差は 0(丸めの範囲)なので、(i)は成り立たない
    statistics = learning_loss_statistics(record)
    rounding = 16 * (EPSILON_FP16 if USE_FP16_AUTOCAST else EPSILON_FP32) * record["initial_model_loss_on_final_window"]
    threshold = max(P1_SIGMA_MULTIPLIER * statistics["standard_error"], rounding)
    reasons = []
    if not record["finite"]:
        reasons.append("訓練損失が有限でない")
    if not statistics["mean"] <= -threshold:
        reasons.append(f"損失の差の平均 {statistics['mean']:+.3f} が -{P1_SIGMA_MULTIPLIER:g} x 標準誤差({threshold:.3f})以下でない")
    if not record["final_train_loss"] < UNIFORM_LOSS:
        reasons.append(f"最後の区間の訓練損失 {record['final_train_loss']:.4f} が ln(1 + n) = {UNIFORM_LOSS:.4f} 未満でない")
    return reasons


def learning_precondition(record: dict) -> bool:
    return not learning_failure_reasons(record)


CALIBRATION_BOOTSTRAP_WEIGHTS = cluster_bootstrap_weights(QUERY_ARTICLES[CALIBRATION_QUERY_INDEX], BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED + 1)


def bootstrap_sd_of_mean(values: np.ndarray) -> float:
    # 較正の query ごとの値の平均の、記事を単位とするブートストラップの標準偏差
    resampled = (CALIBRATION_BOOTSTRAP_WEIGHTS @ np.asarray(values, dtype=np.float64)) / CALIBRATION_BOOTSTRAP_WEIGHTS.sum(axis=1)
    return float(resampled.std(ddof=1))


def combined_sigma(per_seed: np.ndarray, bootstrap_contrast: np.ndarray) -> dict:
    # シードが 1 つのときは、シード間の項を 0 とする(学習率 0 の確認など)
    seed_variance = float(np.var(per_seed, ddof=1)) / len(per_seed) if len(per_seed) >= 2 else 0.0
    bootstrap_variance = float(np.var(bootstrap_contrast, ddof=1))
    return {"sigma": math.sqrt(seed_variance + bootstrap_variance), "seed_term": math.sqrt(seed_variance), "bootstrap_term": math.sqrt(bootstrap_variance)}


def validation_gain_statistics(records: list[dict]) -> dict:
    # 学習の成立 (b)(6.1 節)の量(基準の条件、全シード): 較正の query での Recall@10 の、学習前のモデルからの増分 g_s(シードごと)のシード平均 gain と、
    # 判定と同じ式の標準偏差 sigma = sqrt(Var_s(g_s) / n + sigma_boot^2)(sigma_boot はシード平均の増分の、記事を単位とするブートストラップの標準偏差)
    difference = np.stack([r["validation_hits"].astype(np.float64) - r["untrained_validation_hits"].astype(np.float64) for r in records])  # (シード, query)
    per_seed = difference.mean(axis=1)
    resampled = (CALIBRATION_BOOTSTRAP_WEIGHTS @ difference.mean(axis=0)) / CALIBRATION_BOOTSTRAP_WEIGHTS.sum(axis=1)
    sigma = combined_sigma(per_seed, resampled)
    return {"gain": float(per_seed.mean()), "per_seed_gain": per_seed, **sigma}


def validation_gain_precondition(records: list[dict]) -> bool:
    # 学習の成立 (b): gain > 0 かつ gain >= P1_SIGMA_MULTIPLIER x sigma
    statistics = validation_gain_statistics(records)
    return bool(statistics["gain"] > 0 and statistics["gain"] >= P1_SIGMA_MULTIPLIER * statistics["sigma"])


def judge(delta: float, sigma: float) -> str:
    threshold = SIGMA_MULTIPLIER * sigma
    if delta > threshold:
        return "支持"
    if delta < -threshold:
        return "反証"
    return "判定不能"
```

## 6. 実験 / Experiments



## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/025_ann_search_and_reranking.ipynb
