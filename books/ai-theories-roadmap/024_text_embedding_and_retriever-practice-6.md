---
title: "テキスト埋め込みと retriever / Text Embedding and Retriever(実装・実験編 6/7)"
---

この記事は後編(実装・実験編 6/7)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/024_text_embedding_and_retriever-practice-5)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/024_text_embedding_and_retriever-practice-7)。

### 6.11 観察 E・F(判定基準を設けない)

**観察 E: BM25 との比較。** 評価用の同じ索引・同じ query で BM25 の Recall@10・MRR・nDCG@10 を測り、各条件(全次元)の値と並べる。優劣は判定しない。BM25 は学習を伴わず、シードを持たない。

**観察 F: 異方性と alignment・uniformity。** 対照学習の前(008 の重みのまま、平均プール・因果マスク)と後(C2 のシード 0)で、評価用の passage の埋め込みについて、
ランダムな 2 つの組のコサイン類似度の平均(異方性)、alignment(正例の組 = 各 query と、その記事の次の窓の passage)、uniformity、Recall@10 を測る。
定性的な観察であり、判定基準を設けない。


```python
# --- 観察 E: BM25 ---
_t0_bm25 = time.time()
BM25_INDEX = BM25Index(EVALUATION_SET.passages, tokenizer.vocab_size)
BM25_RESULT = rank_candidates(
    torch.from_numpy(BM25_INDEX.score_batch(EVALUATION_SET.queries)), EVALUATION_SET.query_articles, EVALUATION_SET.passage_articles, EVALUATION_SET.query_sources
)
_bm25_boot = paired_cluster_bootstrap_ratio_of_sums(
    BM25_RESULT.hits(10).astype(np.float64), np.ones(EVALUATION_SET.num_queries), EVALUATION_SET.query_articles, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
)
print(
    f"{RUN_TAG}観察 E(評価用の索引 {EVALUATION_SET.num_passages} passage・query {EVALUATION_SET.num_queries} 個、ランダムな順位の Recall@10 の期待値 {RANDOM_RECALL['evaluation']:.4f}、"
    f"BM25 k1 = {BM25_INDEX.k1}・b = {BM25_INDEX.b}、{time.time() - _t0_bm25:.1f} 秒)"
)
print(
    f"  BM25: Recall@10 {BM25_RESULT.recall_at(10):.4f}(記事を単位とするブートストラップの標準偏差 {_bm25_boot.std(ddof=1):.4f})・MRR {BM25_RESULT.mean_reciprocal_rank():.4f}・"
    f"nDCG@10 {BM25_RESULT.mean_ndcg_at_10():.4f}"
)
OBSERVATION_E = {"BM25": (BM25_RESULT.recall_at(10), BM25_RESULT.mean_reciprocal_rank(), BM25_RESULT.mean_ndcg_at_10())}
for _c in RUN_ORDER:
    _recalls = recall_of(_c, NUM_SEEDS[_c])
    _mrr = np.array([(1 / RUNS[(_c, s)]["rank"][EMBEDDING_DIMENSION]).mean() for s in range(NUM_SEEDS[_c])])
    _ndcg = np.array([RUNS[(_c, s)]["ndcg_at_10"].mean() for s in range(NUM_SEEDS[_c])])
    OBSERVATION_E[_c] = (float(_recalls.mean()), float(_mrr.mean()), float(_ndcg.mean()))
    print(
        f"  {_c}({NUM_SEEDS[_c]} シード): Recall@10 {_recalls.mean():.4f}(シード間の標準偏差 {_recalls.std(ddof=1):.4f})・MRR {_mrr.mean():.4f}・nDCG@10 {_ndcg.mean():.4f}"
    )

# --- 観察 F: 対照学習の前後 ---
_reference_model = build_model("C2", 0).to(device)  # 008 の重みのまま(平均プール・因果マスク)
_before = evaluate_set(_reference_model, "evaluation", keep_embeddings=True)
del _reference_model
empty_device_cache()


def positive_passage_index(retrieval_set) -> np.ndarray:
    # 各 query の正例の passage: 同じ記事の次の窓(最後の窓なら、同じ記事の最初の窓)。元の passage 自身は除く
    result = np.empty(retrieval_set.num_queries, dtype=np.int64)
    for i, (article, source) in enumerate(zip(retrieval_set.query_articles, retrieval_set.query_sources, strict=True)):
        members = np.nonzero(retrieval_set.passage_articles == article)[0]
        members = members[members != source]
        later = members[members > source]
        result[i] = later[0] if len(later) else members[0]
    return result


_positive = positive_passage_index(EVALUATION_SET)
assert (EVALUATION_SET.passage_articles[_positive] == EVALUATION_SET.query_articles).all() and not (_positive == EVALUATION_SET.query_sources).any()
GEOMETRY = {}
for _label, _embeddings in (
    ("008 の重みのまま", {"passages": _before["passage_embeddings"], "queries": _before["query_embeddings"], "recall": _before["by_dimension"][EMBEDDING_DIMENSION]}),
    ("対照学習の後(C2、シード 0)", {"passages": OBSERVATION_EMBEDDINGS["passages"], "queries": OBSERVATION_EMBEDDINGS["queries"], "recall": None}),
):
    _p, _q = _embeddings["passages"], _embeddings["queries"]
    GEOMETRY[_label] = {
        "mean_cosine": mean_pairwise_cosine(_p),
        "alignment": alignment(_q, _p[_positive]),
        "uniformity": uniformity(_p),
        "recall": _embeddings["recall"].recall_at(10) if _embeddings["recall"] is not None else float(RUNS[OBSERVATION_RUN]["hits"][EMBEDDING_DIMENSION].mean()),
        "mrr": _embeddings["recall"].mean_reciprocal_rank() if _embeddings["recall"] is not None else float((1 / RUNS[OBSERVATION_RUN]["rank"][EMBEDDING_DIMENSION]).mean()),
    }
print(f"{RUN_TAG}観察 F(評価用の passage {EVALUATION_SET.num_passages} 個の埋め込み。alignment は {EVALUATION_SET.num_queries} 組の正例)")
for _label, _g in GEOMETRY.items():
    print(
        f"  {_label}: ランダムな組のコサイン類似度の平均 {_g['mean_cosine']:.4f}・alignment {_g['alignment']:.4f}・uniformity {_g['uniformity']:.4f}・"
        f"Recall@10 {_g['recall']:.4f}・MRR {_g['mrr']:.4f}"
    )
```

    観察 E(評価用の索引 3244 passage・query 3244 個、ランダムな順位の Recall@10 の期待値 0.0321、BM25 k1 = 1.2・b = 0.75、1.7 秒)
      BM25: Recall@10 0.7398(記事を単位とするブートストラップの標準偏差 0.0114)・MRR 0.5751・nDCG@10 0.3281
      C2(5 シード): Recall@10 0.6700(シード間の標準偏差 0.0023)・MRR 0.4610・nDCG@10 0.2694
      C1(5 シード): Recall@10 0.5702(シード間の標準偏差 0.0056)・MRR 0.3589・nDCG@10 0.2024
      C3(5 シード): Recall@10 0.6779(シード間の標準偏差 0.0061)・MRR 0.4699・nDCG@10 0.2765
      C5(5 シード): Recall@10 0.6507(シード間の標準偏差 0.0034)・MRR 0.4397・nDCG@10 0.2556
      C4(5 シード): Recall@10 0.4628(シード間の標準偏差 0.0093)・MRR 0.2885・nDCG@10 0.1545
    観察 F(評価用の passage 3244 個の埋め込み。alignment は 3244 組の正例)
      008 の重みのまま: ランダムな組のコサイン類似度の平均 0.4659・alignment 0.7604・uniformity -1.8987・Recall@10 0.4834・MRR 0.3201
      対照学習の後(C2、シード 0): ランダムな組のコサイン類似度の平均 0.4590・alignment 0.6998・uniformity -2.0079・Recall@10 0.6692・MRR 0.4634


### 6.12 不変条件のアサーションと`SMOKE_TEST`の配線

- 実効水準の照合: 実際に学習した条件・シード・ステップ数・途中の評価のステップが、5.2 節で印字した水準と 6.4 節で選ばれた計画に一致する。
- 同じシードの条件どうしで、学習の組(ハッシュ)が一致する。異なるシードでは異なる。
- 全条件で、学習ステップ数・バッチサイズ・query と passage の長さ・評価用の索引と query が同じである。
- 学習率が、較正した値または規則で決めた値に一致する。
- 記録した順位から再計算した Recall@10 が、記録した hits の平均と一致する。
- 評価用の索引と query の記事が、すべて 008 の事前学習とトークナイザの学習の範囲(参照コーパス)の外にある。


```python
# --- 実効水準の照合(SMOKE_TEST の配線) ---
assert CURRENT_LEVEL_NAME == ("smoke" if SMOKE_TEST else "prod")
assert STEP_CANDIDATES == LEVELS[CURRENT_LEVEL_NAME]["STEP_CANDIDATES"] and NUM_STEPS in STEP_CANDIDATES
assert BOOTSTRAP_RESAMPLES == LEVELS[CURRENT_LEVEL_NAME]["BOOTSTRAP_RESAMPLES"]
assert NUM_SEEDS == STAGES[CURRENT_LEVEL_NAME][STAGE]["seeds"] and SELECTED_PLAN == PLANS[SELECTED_PLAN["plan"]]
assert set(RUNS) == {(c, s) for c in CONDITIONS for s in range(NUM_SEEDS[c])}, "学習した条件・シードが選ばれた計画と一致しない"
assert all(r["num_steps"] == NUM_STEPS and len(r["train_loss"]) == NUM_STEPS for r in RUNS.values())
assert all(r["eval_step"] == (list(EVAL_STEPS) if CONDITIONS[r["condition"]]["track"] else []) for r in RUNS.values())
assert all(r["learning_rate"] == LEARNING_RATE[r["condition"]] for r in RUNS.values())
for _c in CONDITIONS:  # 較正した条件はその値、規則で決める条件は「基にする条件の値 x 格子の中心の比」(丸めの誤差の範囲で一致)
    _source = LEARNING_RATE_SOURCE[_c]
    if _source == _c:
        assert LEARNING_RATE[_c] == CALIBRATION[_c]["chosen"]
    else:
        assert math.isclose(LEARNING_RATE[_c], CALIBRATION[_source]["chosen"] * LEARNING_RATE_CENTER[_c] / LEARNING_RATE_CENTER[_source], rel_tol=8 * EPSILON_FP64)
        assert math.isclose(GRID_MULTIPLIER[_c], CALIBRATION[_source]["chosen"] / LEARNING_RATE_CENTER[_source], rel_tol=8 * EPSILON_FP64), "格子の中での位置が基にする条件と異なる"
assert all(len(r["eval_validation_recall"]) == len(r["eval_step"]) for r in RUNS.values())
# --- P1 の基準: 学習前のモデルの損失は全ての学習で正で有限、同じ手順で再現できる。学習前の 008 の検証用の値は、ランダムな順位の期待値より高い ---
assert all(math.isfinite(r["initial_model_loss_on_final_window"]) and r["initial_model_loss_on_final_window"] > 0 for r in RUNS.values())
_key = OBSERVATION_RUN
_again = float(
    compute_pair_losses(
        build_model(*_key).to(device), TRAIN_STREAM, TRAIN_STARTS, get_schedule(_key[1], NUM_STEPS), range(NUM_STEPS - final_loss_window(NUM_STEPS), NUM_STEPS),
        QUERY_LENGTH, PASSAGE_LENGTH, TEMPERATURE, matryoshka_dimensions=MATRYOSHKA_DIMENSIONS if CONDITIONS[_key[0]]["loss"] == "matryoshka" else None,
        use_fp16_autocast=USE_FP16_AUTOCAST,
    ).mean()
)
assert math.isclose(_again, RUNS[_key]["initial_model_loss_on_final_window"], rel_tol=16 * (EPSILON_FP16 if USE_FP16_AUTOCAST else EPSILON_FP32)), (_again, RUNS[_key]["initial_model_loss_on_final_window"])
assert UNTRAINED_VALIDATION_RECALL > RANDOM_RECALL["validation"]
print(f"P1 の基準: 学習前のモデルの損失は全 {len(RUNS)} 回の学習で正で有限、{_key} で同じ手順の再計算と一致。学習前の 008 の検証用の Recall@10 {UNTRAINED_VALIDATION_RECALL:.4f} > ランダム {RANDOM_RECALL['validation']:.4f}: OK")
# --- 対応のある比較: 同じシードの条件どうしで学習の組が一致する / 異なるシードでは異なる ---
for _s in range(min(NUM_SEEDS.values())):
    assert len({RUNS[(c, _s)]["schedule_hash"] for c in CONDITIONS}) == 1, f"シード {_s} で学習の組が条件間で一致しない"
assert len({RUNS[("C2", s)]["schedule_hash"] for s in range(NUM_SEEDS["C2"])}) == NUM_SEEDS["C2"], "異なるシードで学習の組が同じ"
# --- キャッシュの導入前後の一致: 使い回している学習の組(SCHEDULES)が、毎回作り直した場合と完全に一致する ---
for (_seed_index, _steps), _cached in SCHEDULES.items():
    _fresh = sample_pair_schedule(TRAIN_TOKEN_COUNTS, SAMPLING_WEIGHTS, QUERY_LENGTH, PASSAGE_LENGTH, _steps, BATCH_SIZE, seed=PAIR_SEED_BASE + _seed_index)
    assert set(_cached) == set(_fresh) and all(np.array_equal(_cached[k], _fresh[k]) for k in _fresh), (_seed_index, _steps)
    _digest = hashlib.sha256()
    for _key in ("article", "query_start", "passage_start"):
        _digest.update(np.ascontiguousarray(_fresh[_key], dtype=np.int64).tobytes())
    for _record in RUNS.values():
        if _record["seed"] == _seed_index and _record["num_steps"] == _steps:
            assert _record["schedule_hash"] == _digest.hexdigest(), "学習に使われた組が、作り直した組と一致しない"
print(f"学習の組のキャッシュ({len(SCHEDULES)} 通り)が、作り直した場合と完全に一致(配列の完全一致と、学習の記録のハッシュの一致): OK")
# --- 全条件で評価の分母が同じ ---
assert all(
    r["hits"][m].shape == (EVALUATION_SET.num_queries,) and r["rank"][m].shape == (EVALUATION_SET.num_queries,) for r in RUNS.values() for m in MATRYOSHKA_DIMENSIONS
)
assert all(set(r["hits"]) == set(MATRYOSHKA_DIMENSIONS) for r in RUNS.values())
_rebuilt = make_sets()
assert [s.digest() for s in RETRIEVAL_SETS.values()] == [s.digest() for s in _rebuilt], "評価用の索引と query が実行の途中で変わった"
del _rebuilt
# --- 記録した順位から再計算した値の整合 ---
for _record in RUNS.values():
    for _m in MATRYOSHKA_DIMENSIONS:
        assert np.array_equal(_record["rank"][_m] <= 10, _record["hits"][_m])
        assert _record["rank"][_m].min() >= 1 and _record["rank"][_m].max() <= EVALUATION_SET.num_passages - 1
# --- 条件の定義が実際の学習で使われた ---
assert all(("final_dimension_losses" in r) == (CONDITIONS[r["condition"]]["loss"] == "matryoshka") for r in RUNS.values())
assert all(("position_hits" in r) == (r["condition"] == "C2") for r in RUNS.values())
assert UPLOAD_RUN in SAVED_STATES and set(SAVED_STATES) == {UPLOAD_RUN}
assert OBSERVATION_RUN in RUNS
# --- 008 とトークナイザの学習の範囲の外: 評価用の索引と query の記事は、すべて参照コーパスの外に始まる ---
_evaluated_articles = set(EVALUATION_SET.query_articles.tolist()) | set(EVALUATION_SET.passage_articles.tolist())
assert min(_evaluated_articles) >= NUM_REFERENCE_ARTICLES
assert all(article_spans[a]["start"] >= REFERENCE_CORPUS_CHARACTERS > REFERENCE_PRETRAINING_END for a in _evaluated_articles)
assert all(r["finite"] for r in RUNS.values()), "有限でない損失の学習がある"
print(
    f"{RUN_TAG}実効水準: T = {NUM_STEPS}、計画 {SELECTED_PLAN['plan']}(段階 {STAGE}、較正の方式 {CALIBRATION_MODE!r})、シード数 {json.dumps(NUM_SEEDS)}、学習 {len(RUNS)} 回、"
    f"バッチ {BATCH_SIZE}、query {QUERY_LENGTH}・passage {PASSAGE_LENGTH}、評価用の query {EVALUATION_SET.num_queries} 個、ブートストラップ {BOOTSTRAP_RESAMPLES:,} 回: 照合 OK"
)
print(f"同じシードの条件どうしで学習の組が一致(シード {min(NUM_SEEDS.values())} 個で確認)、異なるシードでは異なる: OK")
```

    P1 の基準: 学習前のモデルの損失は全 25 回の学習で正で有限、('C2', 0) で同じ手順の再計算と一致。学習前の 008 の検証用の Recall@10 0.6919 > ランダム 0.0962: OK
    学習の組のキャッシュ(6 通り)が、作り直した場合と完全に一致(配列の完全一致と、学習の記録のハッシュの一致): OK
    実効水準: T = 1024、計画 0(段階 0、較正の方式 'all')、シード数 {"C1": 5, "C2": 5, "C3": 5, "C4": 5, "C5": 5}、学習 25 回、バッチ 64、query 32・passage 128、評価用の query 3244 個、ブートストラップ 10,000 回: 照合 OK
    同じシードの条件どうしで学習の組が一致(シード 5 個で確認)、異なるシードでは異なる: OK


### 6.13 判定結果の一覧


```python
print(f"{RUN_TAG}判定結果(計画 {SELECTED_PLAN['plan']}、T = {NUM_STEPS}、較正の方式 {CALIBRATION_MODE!r}、段階 {STAGE})")
VERDICTS = {
    "A": (DELTA_A, SIGMA_A["sigma"], A_COMPUTED, A_VERDICT, A_PRECONDITIONS),
    "B": (DELTA_B, SIGMA_B["sigma"], B_COMPUTED, B_VERDICT, B_PRECONDITIONS),
    "C": (DELTA_C, SIGMA_C["sigma"], C_COMPUTED, C_VERDICT, C_PRECONDITIONS),
    "D": (DELTA_D, SIGMA_D["sigma"], D_COMPUTED, D_VERDICT, D_PRECONDITIONS),
}
for _experiment, (_delta, _sigma, _computed, _verdict, _preconditions) in VERDICTS.items():
    print(
        f"  実験 {_experiment}: 対比量 {_delta:+.4f}、標準偏差 {_sigma:.4f}、閾値 {SIGMA_MULTIPLIER * _sigma:.4f}、判定関数の結果 {_computed}、"
        f"前提条件 {json.dumps({k: precondition_status[k] for k in _preconditions})} -> {RUN_TAG}最終判定: {_verdict}"
    )
TOTAL_SECONDS = time.time() - NOTEBOOK_START_TIME
print(f"全体の経過時間 {TOTAL_SECONDS / 60:.1f} 分(予算 {SESSION_BUDGET_SECONDS / 60:.0f} 分、計画の見積もり {PLAN_ESTIMATES[SELECTED_PLAN['plan']]['total'] / 60:.1f} 分 + 選択時の経過 {ELAPSED_AT_SELECTION / 60:.1f} 分)")
```

    判定結果(計画 0、T = 1024、較正の方式 'all'、段階 0)
      実験 A: 対比量 -0.0999、標準偏差 0.0075、閾値 0.0150、判定関数の結果 反証、前提条件 {"P0(A)": true, "P1(A)": true, "P2(A)": true} -> 最終判定: 反証
      実験 B: 対比量 +0.0078、標準偏差 0.0049、閾値 0.0098、判定関数の結果 判定不能、前提条件 {"P0(B)": true, "P1(B)": true, "P2(B)": true} -> 最終判定: 判定不能
      実験 C: 対比量 +0.2073、標準偏差 0.0092、閾値 0.0183、判定関数の結果 支持、前提条件 {"P0(C)": true, "P1(C)": true, "P2(C)": true} -> 最終判定: 支持
      実験 D: 対比量 +0.1908、標準偏差 0.0074、閾値 0.0147、判定関数の結果 支持、前提条件 {"P0(D)": true, "P1(D)": true, "P2(D)": true} -> 最終判定: 支持
    全体の経過時間 64.3 分(予算 120 分、計画の見積もり 65.0 分 + 選択時の経過 8.4 分)


### 6.14 Hugging Face Hub へのアップロード(C5 のシード 0)

後続のトピック([025](https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/025_ann_search_and_reranking.ipynb))の入力にするため、**条件 C5(Matryoshka Representation Learning)のシード 0** の学習後の重みを`kojikojiprg/ai-theories-text-embedding-en`(新規、public)にアップロードする。
アップロードするモデルは結果を見て選ばない(条件とシードを事前に固定している)。

- **アップロードのセルは`UPLOAD_ARTIFACTS`で守る。既定は`False`。** `SMOKE_TEST`とは独立のフラグで、本番(`SMOKE_TEST = False`)でも、アップロードは`UPLOAD_ARTIFACTS`を別途`True`にしたときだけ行う。
- Colab Secrets の`HF_TOKEN`が取得できない場合は、例外で停止せず、アップロードをスキップしてその旨を印字する。取得した値は印字・記録・加工しない。
- アップロード後、`list_repo_files()`でリポジトリに意図したファイルが実際に存在することを確かめる。`upload_file`が例外を送出しなかったことだけを根拠にしない。
- **このセルの前半(ファイルの書き出し・読み込み直し・評価・モデルカードの組み立て)は、`UPLOAD_ARTIFACTS`によらず毎回実行する**(アップロードしなくてもモデルカードの下書きを確認できるように)。
  モデルカードの指標は、書き出した重みを新しいモデルに読み込み直して評価した値とする(学習中の途中の評価値は使わない)。
- トークナイザは同梱しない。モデルカードで`kojikojiprg/ai-theories-tokenizer-en`を参照する。


```python
_t0_upload = time.time()
MODEL_CARD_TEMPLATE = """---
language: en
license: mit
tags:
- ai-theories
- text-embedding
- retrieval
- matryoshka-representation-learning
- scratch-implementation
---

# ai-theories テキスト埋め込みモデル(英語、Matryoshka Representation Learning)

`ai-theories`(https://github.com/kojikojiprg/ai-theories)プロジェクトの成果物。
008 の小型 GPT(`kojikojiprg/ai-theories-small-gpt-en`)を encoder として、英語 Wikipedia の同じ記事から切り出した重ならない 2 つの区間を正例の組にする
教師なしの対照学習(in-batch negatives の InfoNCE)と Matryoshka Representation Learning(入れ子の次元 {matryoshka_dimensions})で、全パラメータを更新したもの。
スクラッチ実装であり、研究・教育目的のモデルである。品質保証は行っていない。商用・実運用での利用は想定しない。

## 由来

- [024. テキスト埋め込みと retriever](https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/024_text_embedding_and_retriever.ipynb)の
  条件 C5(Matryoshka Representation Learning)のシード 0 の学習後の重み。結果を見て選んだものではなく、事前に固定した条件とシードである。
- 学習: {num_steps} ステップ、バッチ {batch_size} 組、query {query_length} トークン・passage {passage_length} トークン、温度 {temperature}、学習率 {learning_rate:.3g}、
  AdamW(重み減衰 {weight_decay})、warmup + cosine、gradient clipping {clip}。学習データは `kojikojiprg/ai-theories-corpus-en` の、先頭の 356 記事を除く記事のうち学習用に分けた {num_train_articles} 記事のみ(検証用・評価用の記事は含まない)。

## 使うために必要な他のアーティファクト

- **トークナイザは同梱していない。** `kojikojiprg/ai-theories-tokenizer-en` を使用してください(語彙サイズ 8192、特殊トークンなし)。
- 本体の構成は `config.json`(008 の `config.json` と同じ構成に、埋め込みの設定を加えたもの)を参照。

## 埋め込みの作り方(プーリング・注意マスク・切り詰め)

- 注意マスク: **因果マスク**(008 と同じ)。
- プーリング: **平均プール**(最終正規化層の後の隠れ状態を、全位置で平均する)。射影層はない。
- 埋め込み: プーリングしたベクトルを L2 正規化する(次元 256)。類似度は内積(= コサイン類似度)。
- 切り詰め: 先頭 m 次元を取り出して **L2 正規化し直す**(m は {matryoshka_dimensions} のいずれか)。
- query は {query_length} トークン、passage は {passage_length} トークンの固定長で学習した(パディングは使わない)。

```python
import json
import torch
from huggingface_hub import hf_hub_download
from src.models.gpt import GPTLanguageModel
from src.models.text_embedding import TextEmbeddingModel, truncate_and_normalize

config = json.load(open(hf_hub_download("{repo_id}", "config.json")))
# GPTLanguageModel の構築は 008(kojikojiprg/ai-theories-small-gpt-en)の config.json と同じ(024 のノートブックの build_gpt() を参照)
backbone = build_gpt(config)
backbone.load_state_dict(torch.load(hf_hub_download("{repo_id}", "model_state.pt"), map_location="cpu"))
model = TextEmbeddingModel(backbone, pooling="mean", attention="causal").eval()
with torch.no_grad():
    embedding = model(token_ids)                  # 次元 256
    embedding_16 = model(token_ids, dimension=16)  # 先頭 16 次元を切り詰めて再正規化
```

## 評価(アップロードした重みそのものを読み込み直して測った値)

評価用の索引と query(`src/data/retrieval.py` の `build_retrieval_set()`)で、全件の内積による厳密な探索。query を含む passage は候補から除く。

| 次元 m | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|
{evaluation_rows}

- ランダムな順位の Recall@10 の期待値: {random_recall:.4f}、BM25 の Recall@10: {bm25_recall:.4f}、評価用の query {num_queries} 個、索引 {num_passages} passage。
- 上の値は {smoke_note}

## 評価用の passage の集合の再構成(025 など後続のトピック向け)

評価用の索引と query は、次の手順で決定的に再構成できる(`src/data/retrieval.py`)。

1. コーパス `kojikojiprg/ai-theories-corpus-en`(9,826 記事)と、記事の境界 `metadata.json` の `article_offsets` を取得する。
   先頭の {num_reference_articles} 記事は 008 の事前学習とトークナイザの学習に使われたコーパス(`kojikojiprg/ai-theories-corpus-en-pretraining`)と同一で、学習・評価のどれにも使わない。
   使うのは、参照コーパスの全体({reference_characters} 文字)の外に始まる記事(`find_articles_starting_at_or_after()`)で、{num_candidates} 記事ある。
2. トークナイザ `kojikojiprg/ai-theories-tokenizer-en` で、使う記事を記事ごとに別々に符号化する(`encode_articles()`)。
3. `split_articles(記事ごとのトークン数, {min_tokens}, 使う記事, {num_validation}, {num_evaluation}, {split_seed})` で記事を分割し、評価用の記事を得る。
4. `build_retrieval_set(記事ごとのトークン, 評価用の記事, {passage_length}, {query_length}, {max_passages}, {max_queries}, {evaluation_seed})` で索引と query を得る。
   ダイジェスト(`RetrievalSet.digest()`)は `{evaluation_digest}`、分割のダイジェスト(`ArticleSplit.digest()`)は `{split_digest}`。

## 注意

- 学習データは英語 Wikipedia の長大な記事(Special:LongPages の長い順に選んだ記事。約 {list_like_share} がリスト系の記事)の {total_tokens} トークンに限られ、汎用の埋め込みモデルとして使えるものではない。
- 評価用の集合は、ノートブックの実行時点のコーパス(`article_offsets` を含む `metadata.json`)に依存する。
"""


def build_card_and_files(directory: Path) -> dict:
    # 学習後の重みをファイルに書き出し、新しいモデルに読み込み直して評価した値でモデルカードを組み立てる
    state = SAVED_STATES[UPLOAD_RUN]
    torch.save(state, directory / "model_state.pt")
    config = dict(REFERENCE_CONFIG) | {
        "embedding_dimension": EMBEDDING_DIMENSION,
        "pooling": "mean",
        "attention": "causal",
        "matryoshka_dimensions": list(MATRYOSHKA_DIMENSIONS),
        "temperature": TEMPERATURE,
        "query_length": QUERY_LENGTH,
        "passage_length": PASSAGE_LENGTH,
        "tokenizer_repo": TOKENIZER_REPO_ID,
        "base_model_repo": REFERENCE_MODEL_REPO_ID,
        "source_notebook": "theories/07_retrieval/024_text_embedding_and_retriever.ipynb",
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    reloaded = TextEmbeddingModel(build_gpt(REFERENCE_CONFIG), pooling="mean", attention="causal")
    reloaded.backbone.load_state_dict(torch.load(directory / "model_state.pt", map_location="cpu"))
    assert all(torch.equal(a, b) for a, b in zip(reloaded.backbone.state_dict().values(), state.values(), strict=True)), "書き出した重みが一致しない"
    reloaded = reloaded.to(device)
    evaluated = evaluate_set(reloaded, "evaluation", MATRYOSHKA_DIMENSIONS)["by_dimension"]
    del reloaded
    empty_device_cache()
    # 学習直後の値(記録した順位)と、読み込み直した重みでの値が一致する(同じ関数・同じデバイス、符号化のバッチの大きさは結果に影響しない)
    for m, result in evaluated.items():
        assert abs(result.recall_at(10) - float(RUNS[UPLOAD_RUN]["hits"][m].mean())) <= 2 / EVALUATION_SET.num_queries, (m, result.recall_at(10))
    rows = "\n".join(
        f"| {m} | {r.recall_at(10):.4f} | {r.mean_reciprocal_rank():.4f} | {r.mean_ndcg_at_10():.4f} |" for m, r in evaluated.items()
    )
    card = MODEL_CARD_TEMPLATE.format(
        matryoshka_dimensions=", ".join(str(m) for m in MATRYOSHKA_DIMENSIONS),
        num_steps=NUM_STEPS, batch_size=BATCH_SIZE, query_length=QUERY_LENGTH, passage_length=PASSAGE_LENGTH,
        temperature=TEMPERATURE, learning_rate=LEARNING_RATE["C5"], weight_decay=WEIGHT_DECAY, clip=GRADIENT_CLIP_THRESHOLD,
        repo_id=UPLOAD_REPO_ID, evaluation_rows=rows, random_recall=RANDOM_RECALL["evaluation"], bm25_recall=BM25_RESULT.recall_at(10),
        num_queries=EVALUATION_SET.num_queries, num_passages=EVALUATION_SET.num_passages,
        smoke_note="スモークテストの値であり、意味を持たない。" if SMOKE_TEST else "本番の実行の値である。",
        min_tokens=MIN_TOKENS_FOR_RETRIEVAL, num_validation=NUM_VALIDATION_ARTICLES, num_evaluation=NUM_EVALUATION_ARTICLES, split_seed=SPLIT_SEED,
        max_passages=MAX_PASSAGES_PER_ARTICLE, max_queries=MAX_QUERIES_PER_ARTICLE, evaluation_seed=EVALUATION_SET_SEED,
        evaluation_digest=EVALUATION_SET.digest(), split_digest=SPLIT.digest(), total_tokens=f"約 {CANDIDATE_TOKENS / 1e6:.0f} 百万", num_train_articles=f"{len(SPLIT.train):,}", num_reference_articles=NUM_REFERENCE_ARTICLES,
        reference_characters=f"{REFERENCE_CORPUS_CHARACTERS:,}", num_candidates=f"{len(CANDIDATE_ARTICLES):,}",
        list_like_share=f"{len(LIST_LIKE_ARTICLES) / len(CANDIDATE_ARTICLES):.0%}",
    )
    (directory / "README.md").write_text(card, encoding="utf-8")
    return {"evaluated": evaluated, "card": card, "files": sorted(p.name for p in directory.iterdir())}


with tempfile.TemporaryDirectory() as _temporary_directory:
    _built = build_card_and_files(Path(_temporary_directory))
    UPLOAD_FILES = _built["files"]
    assert UPLOAD_FILES == ["README.md", "config.json", "model_state.pt"], "トークナイザなど、同梱しないファイルが含まれている"
    print(f"{RUN_TAG}モデルカードの下書きと書き出したファイル {UPLOAD_FILES} を作成し、重みを読み込み直して評価した(Recall@10 {', '.join(f'm={m}: {r.recall_at(10):.4f}' for m, r in _built['evaluated'].items())})")
    print("--- モデルカードの先頭 ---")
    print("\n".join(_built["card"].splitlines()[:24]))

    if not UPLOAD_ARTIFACTS:
        print("UPLOAD_ARTIFACTS = False のため、アップロードは行わない(モデルカードの下書きの確認のみ)")
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

    モデルカードの下書きと書き出したファイル ['README.md', 'config.json', 'model_state.pt'] を作成し、重みを読み込み直して評価した(Recall@10 m=16: 0.5672, m=32: 0.6008, m=64: 0.6208, m=128: 0.6319, m=256: 0.6477)
    --- モデルカードの先頭 ---
    ---
    language: en
    license: mit
    tags:
    - ai-theories
    - text-embedding
    - retrieval
    - matryoshka-representation-learning
    - scratch-implementation
    ---
    
    # ai-theories テキスト埋め込みモデル(英語、Matryoshka Representation Learning)
    
    `ai-theories`(https://github.com/kojikojiprg/ai-theories)プロジェクトの成果物。
    008 の小型 GPT(`kojikojiprg/ai-theories-small-gpt-en`)を encoder として、英語 Wikipedia の同じ記事から切り出した重ならない 2 つの区間を正例の組にする
    教師なしの対照学習(in-batch negatives の InfoNCE)と Matryoshka Representation Learning(入れ子の次元 16, 32, 64, 128, 256)で、全パラメータを更新したもの。
    スクラッチ実装であり、研究・教育目的のモデルである。品質保証は行っていない。商用・実運用での利用は想定しない。
    
    ## 由来
    
    - [024. テキスト埋め込みと retriever](https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/024_text_embedding_and_retriever.ipynb)の
      条件 C5(Matryoshka Representation Learning)のシード 0 の学習後の重み。結果を見て選んだものではなく、事前に固定した条件とシードである。
    - 学習: 1024 ステップ、バッチ 64 組、query 32 トークン・passage 128 トークン、温度 0.05、学習率 0.00048、
      AdamW(重み減衰 0.1)、warmup + cosine、gradient clipping 1.0。学習データは `kojikojiprg/ai-theories-corpus-en` の、先頭の 356 記事を除く記事のうち学習用に分けた 9,070 記事のみ(検証用・評価用の記事は含まない)。



    Processing Files (0 / 0)      : |          |  0.00B /  0.00B            



    New Data Upload               : |          |  0.00B /  0.00B            



      ...mptm96zlu3/model_state.pt:   3%|2         |  780kB / 29.4MB            


    アップロード完了: kojikojiprg/ai-theories-text-embedding-en のファイル ['.gitattributes', 'README.md', 'config.json', 'model_state.pt'](意図したファイルがすべて存在することを list_repo_files() で確認)
    アップロードの準備 7.1 秒




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/024_text_embedding_and_retriever.ipynb
