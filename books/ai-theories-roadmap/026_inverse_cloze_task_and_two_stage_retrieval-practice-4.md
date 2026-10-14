---
title: "Inverse Cloze Task と passage 単位の 2 段階検索 / Inverse Cloze Task and Two-Stage Retrieval at Passage Level(実装・実験編 4/8)"
---

この記事は後編(実装・実験編 4/8)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/026_inverse_cloze_task_and_two_stage_retrieval-practice-3)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/026_inverse_cloze_task_and_two_stage_retrieval-practice-5)。

### 5.5 学習と評価のヘルパー

- `evaluate_calibration()`: 検証用の記事の passage だけを索引にし、同じ索引・同じ query に、passage 単位のラベルと記事単位のラベルを当てて Recall@10 を測る。較正の指標とパイロットに使う。
- `evaluate_main()`: 評価用の集合で、コーパス全体の索引と評価用の記事の query に、passage 単位のラベル(実験 A)と記事単位のラベル(実験 B)を当てて、query ごとの順位を記録する。診断量(上位 10 件のうち同じ記事の passage の割合、
  1 位が同じ記事の別の passage である query の割合、024 と同じ形式の記事単位の評価)も測る。評価は FP32 で、全件の内積。
- `train_first_stage()`: 条件(R1 または R2)・シード・学習率・ステップ数で第 1 段を学習する。組は両条件で共通(32 記事 × 2 組、シード $26100 + s$)で、R2 は各窓の文脈を正例にし、
  R1 は同じ記事のもう一方の窓の文脈を正例にして、同じ記事の組を負例から除く。学習前のモデルの損失(最後の区間の組で)、最後の区間の訓練損失、検証用の Recall@10 を記録し、本番の学習(`main=True`)は評価用の集合の順位も記録する。
  R2 は、cross-encoder の学習と評価に使う第 1 段の上位 $K$ 件(学習用・検証用・評価用の query)も求める。
- `train_cross_encoder()`: 学習の組(`sample_cloze_reranking_schedule()`)で cross-encoder を学習する。学習前のモデルと学習後のモデルで検証用の query の並べ替えを評価し(学習の成立の前提条件)、
  本番の学習は評価用の query の並べ替えの順位も記録する。


```python
# ---------- 埋め込みの計算と評価 ----------
EVALUATION_BATCH = EMBED_BATCH


def embed_contexts(model, entries: np.ndarray | None = None) -> torch.Tensor:
    # 索引の passage(文を除いた形)の埋め込み(FP32、デバイス上)。entries が None なら索引全体
    if entries is None:
        tokens, lengths = CONTEXT_TOKENS, CONTEXT_LENGTHS
    else:
        tokens, lengths = CONTEXT_TOKENS[torch.as_tensor(entries)], CONTEXT_LENGTHS[entries]
    return encode_tokens_with_lengths(model, tokens, lengths, EVALUATION_BATCH, (None,))[EMBEDDING_DIMENSION]


def embed_cloze_queries(model, entries: np.ndarray) -> torch.Tensor:
    return encode_tokens_with_lengths(model, CLOZE_QUERY_TOKENS[torch.as_tensor(entries)], CLOZE_QUERY_LENGTHS[entries], EVALUATION_BATCH, (None,))[EMBEDDING_DIMENSION]


def embed_article_set(model, set_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    retrieval_set = ARTICLE_LEVEL_SETS[set_name]
    queries = encode_tokens(model, torch.from_numpy(retrieval_set.queries), EVALUATION_BATCH)["by_dimension"][EMBEDDING_DIMENSION]
    passages = encode_tokens(model, torch.from_numpy(retrieval_set.passages), EVALUATION_BATCH)["by_dimension"][EMBEDDING_DIMENSION]
    return queries, passages


def evaluate_calibration(model) -> dict:
    # 検証用の集合の Recall@10(query ごとの 0/1)。索引は検証用の記事の passage だけ(文を除いた形)で、同じ索引・同じ query に 2 つのラベルを当てる:
    # passage 単位(正解は query の passage だけ)と、記事単位(正解は同じ記事の他の passage。query の passage は候補から除く)
    contexts = embed_contexts(model, VALIDATION_INDEX_ENTRIES)
    queries = embed_cloze_queries(model, VALIDATION_ENTRIES)
    positions = np.searchsorted(VALIDATION_INDEX_ENTRIES, VALIDATION_ENTRIES)
    passage_ranks = rank_single_positive(queries, contexts, positions)
    article = rank_article_label(queries, contexts, positions, ENTRY_ARTICLES[VALIDATION_INDEX_ENTRIES])
    return {"passage_hits": passage_ranks <= 10, "article_hits": article.rank <= 10}


def evaluate_main(model, keep_embeddings: bool = False) -> dict:
    # 評価用の集合: コーパス全体の索引と評価用の記事の query に、2 つのラベルを当てる(実験 A: passage 単位、実験 B: 記事単位)。
    # 診断量として、024 と同じ形式の記事単位の評価(024 の評価用の索引と query。固定長)も測る
    contexts = embed_contexts(model)
    queries = embed_cloze_queries(model, EVALUATION_ENTRIES)
    ranks = rank_single_positive(queries, contexts, EVALUATION_ENTRIES)
    article = rank_article_label(queries, contexts, EVALUATION_ENTRIES, ENTRY_ARTICLES)
    top_ids, _ = exact_search(queries, contexts, 10)
    same_article = ENTRY_ARTICLES[top_ids] == EVALUATION_QUERY_ARTICLES[:, None]
    result = {
        "passage_rank": ranks,
        "passage_hits": ranks <= 10,
        "article_rank": article.rank,
        "article_hits": article.rank <= 10,
        "same_article_top10_share": float(same_article.mean()),
        "top1_same_article_wrong": float((same_article[:, 0] & (top_ids[:, 0] != EVALUATION_ENTRIES)).mean()),
        "passage_mean_reciprocal_rank": float((1.0 / ranks).mean()),
        "article_mean_reciprocal_rank": article.mean_reciprocal_rank(),
        "article_normalized_discounted_cumulative_gain": article.ndcg_at_10,
    }
    format_queries, format_passages = embed_article_set(model, "evaluation")
    format_024 = evaluate_embeddings(format_queries, format_passages, EVALUATION_SET)
    result.update({"format024_hits": format_024.rank <= 10, "format024_mean_reciprocal_rank": format_024.mean_reciprocal_rank()})
    if keep_embeddings:
        result["contexts"], result["queries"] = contexts, queries
    return result


# ---------- 第 1 段の学習 ----------
PAIR_BATCHES: dict[tuple[int, int], object] = {}
ELIGIBLE_CACHE: dict[int, np.ndarray] = {}  # 記事のすべての窓から選ぶ処理の走査の結果(結果は変わらず、時間だけが変わる)


def pair_batches(seed_index: int, num_steps: int):
    # Inverse Cloze Task の組(32 記事 x 2 組、同じ記事の組は窓が重ならない)。R1 と R2 で共通(シードとステップ数だけで決まる)
    key = (seed_index, num_steps)
    if key not in PAIR_BATCHES:
        PAIR_BATCHES.clear()  # 同時に保持するのは 1 つだけ(メモリ)
        PAIR_BATCHES[key] = sample_cloze_pair_batches(
            TRAIN_STREAM, np.asarray(TRAIN_STARTS), TRAIN_TOKEN_COUNTS, SAMPLING_WEIGHTS, SENTENCE_TABLES, num_steps, ARTICLES_PER_BATCH, PAIRS_PER_ARTICLE,
            PASSAGE_LENGTH, QUERY_LENGTH, MIN_QUERY_TOKENS, MAX_REMOVED_TOKENS, KEEP_PROBABILITY, seed=CLOZE_BATCH_SEED_BASE + seed_index, eligible_cache=ELIGIBLE_CACHE,
        )
    return PAIR_BATCHES[key]


def pair_arrays(condition: str, seed_index: int, num_steps: int):
    # R2: 組の文脈をそのまま正例にする。R1: 同じ記事のもう一方の組の窓(文を除いた形)を正例にする(バッチの中で隣り合う 2 組の文脈を入れ替える)。
    # 同じバッチの query と passage のトークンの集合は R1 と R2 で同じで、正例の対応だけが違う
    batches = pair_batches(seed_index, num_steps)
    context_tokens, context_lengths = batches.context_tokens, batches.context_lengths
    if condition == "R1":
        swap = np.arange(BATCH_SIZE) ^ 1
        context_tokens, context_lengths = context_tokens[:, swap], context_lengths[:, swap]
    return batches.query_tokens, batches.query_lengths, context_tokens, context_lengths


def first_stage_batches(condition: str, seed_index: int, num_steps: int) -> VariableBatches:
    batches = pair_batches(seed_index, num_steps)
    return VariableBatches(*pair_arrays(condition, seed_index, num_steps), device, article_ids=batches.articles, mask_same_article=condition == "R1")


SAVED_STATES: dict[tuple[str, int], dict] = {}  # アップロード用(R2 と cross-encoder のシード 0)
KEPT_EMBEDDINGS: dict[str, dict] = {}  # 観察 D・E 用(R1・R2 のシード 0 の、索引と評価用の query の埋め込み。CPU)
MINED: dict[int, dict] = {}  # シード -> R2 の上位 K 件(学習用・検証用・評価用の query)と、R2 だけの Recall@10 の hits


def mine_candidates(model, contexts: torch.Tensor) -> dict:
    # R2 の上位 K 件(索引全体の全探索)。学習用・検証用・評価用の query のそれぞれについて
    mined = {}
    for name, entries in (("train", TRAIN_ENTRIES), ("validation", VALIDATION_ENTRIES), ("evaluation", EVALUATION_ENTRIES)):
        queries = embed_cloze_queries(model, entries)
        ids, _ = exact_search(queries, contexts, RERANK_CANDIDATES, chunk_size=2048)
        mined[name] = ids
        if name != "train":
            mined[name + "_hits"] = rank_single_positive(queries, contexts, entries) <= 10  # R2 だけ(並べ替えなし)の Recall@10 の hits
        del queries
    return mined


def train_first_stage(condition: str, seed_index: int, learning_rate: float, num_steps: int, main: bool, mine: bool | None = None) -> dict:
    start = time.time()
    model = build_embedding_model(seed_index).to(device)
    batches = first_stage_batches(condition, seed_index, num_steps)
    window = final_loss_window(num_steps)
    initial_losses = compute_pair_losses_from_batches(model, batches, range(num_steps - window, num_steps), TEMPERATURE, MATRYOSHKA_DIMENSIONS, USE_FP16_AUTOCAST)
    history = train_text_embedding_from_batches(
        model, batches, peak_learning_rate=learning_rate, warmup_steps=warmup_steps_for(num_steps), min_learning_rate=learning_rate * MIN_LEARNING_RATE_RATIO,
        weight_decay=WEIGHT_DECAY, gradient_clip_threshold=GRADIENT_CLIP_THRESHOLD, temperature=TEMPERATURE, matryoshka_dimensions=MATRYOSHKA_DIMENSIONS,
        use_fp16_autocast=USE_FP16_AUTOCAST, init_loss_scale=INIT_LOSS_SCALE, loss_scale_growth_interval=LOSS_SCALE_GROWTH_INTERVAL,
    )
    assert len(history["loss"]) == num_steps
    train_loss = np.array(history["loss"], dtype=np.float64)
    record = {
        "condition": condition, "seed": seed_index, "learning_rate": learning_rate, "num_steps": num_steps, "schedule_hash": history["data_stream_hash"],
        "article_hash": hashlib.sha256(np.ascontiguousarray(pair_batches(seed_index, num_steps).articles).tobytes()).hexdigest(),
        "negatives_per_row": float(batches.negatives_per_row().mean()), "truncated_query_share": float(pair_batches(seed_index, num_steps).truncated.mean()),
        "kept_share": float(pair_batches(seed_index, num_steps).kept.mean()),
        "train_loss": train_loss.astype(np.float32), "initial_train_loss": float(train_loss[:window].mean()), "final_train_loss": float(train_loss[-window:].mean()),
        "initial_model_loss_on_final_window": float(initial_losses.mean()), "finite": bool(np.isfinite(train_loss).all()),
        "skipped_steps": int(sum(history["step_skipped"])), "clip_rate": float(np.mean(history["gradient_clip_triggered"])), "final_loss_scale": float(history["loss_scale"][-1]),
    }
    record["validation"] = evaluate_calibration(model)
    do_mine = (main and condition == "R2") if mine is None else mine
    if main:
        final = evaluate_main(model, keep_embeddings=do_mine or seed_index == UPLOAD_SEED)
        contexts, eval_queries = final.pop("contexts", None), final.pop("queries", None)
        record.update(final)
        if seed_index == UPLOAD_SEED:
            KEPT_EMBEDDINGS[condition] = {"contexts": contexts.cpu(), "queries": eval_queries.cpu()}
            if condition == "R2":
                SAVED_STATES[("R2", UPLOAD_SEED)] = {k: v.detach().cpu().clone() for k, v in model.backbone.state_dict().items()}
        if do_mine:
            MINED[seed_index] = mine_candidates(model, contexts)
        del contexts, eval_queries
    elif do_mine:
        MINED[seed_index] = mine_candidates(model, embed_contexts(model))
    record["seconds"] = time.time() - start
    del model, batches
    empty_device_cache()
    return record


def first_stage_failure_reasons(record: dict) -> list[str]:
    # 第 1 段の P1(a)(6.1 節): 訓練損失がすべてのステップで有限で、最後の区間の訓練損失が、同じ組での学習前のモデルの損失の P1_LOSS_RATIO 倍以下
    reasons = []
    if not record["finite"]:
        reasons.append("訓練損失が有限でない")
    ratio = record["final_train_loss"] / record["initial_model_loss_on_final_window"]
    if not ratio <= P1_LOSS_RATIO:
        reasons.append(f"最後の区間の訓練損失 / 学習前のモデルの損失 {ratio:.3f} が {P1_LOSS_RATIO} 以下でない")
    return reasons


# ---------- cross-encoder の学習と並べ替えの評価 ----------
CROSS_ENCODER_TABLES = VariableTables(CLOZE.query_tokens, CLOZE.query_lengths, CLOZE.context_tokens, CLOZE.context_lengths, device)
CROSS_ENCODER_SCHEDULES: dict[tuple[int, int], object] = {}
UNTRAINED_VALIDATION: dict[int, np.ndarray] = {}  # シード -> 学習前の cross-encoder の検証用の query ごとの hits


def median_score_std(scores: np.ndarray) -> float:
    # 診断量: query ごとの、候補 K 件のスコアの標準偏差の中央値(定数に近い出力に崩れたモデルでは 0 に近づく。判定には使わない)
    return float(np.median(np.asarray(scores, dtype=np.float64).std(axis=1)))


def evaluate_rerank(model, entries: np.ndarray, candidate_ids: np.ndarray):
    scores = score_candidates_variable(
        model, CLOZE_QUERY_TOKENS, CLOZE_QUERY_LENGTHS, CONTEXT_TOKENS, CONTEXT_LENGTHS, entries, candidate_ids, SCORE_BATCH, USE_FP16_AUTOCAST
    )
    return rerank_single_positive(scores, candidate_ids, entries, NUM_INDEX), scores


def cross_encoder_schedule(seed_index: int, num_steps: int):
    key = (seed_index, num_steps)
    if key not in CROSS_ENCODER_SCHEDULES:
        CROSS_ENCODER_SCHEDULES[key] = sample_cloze_reranking_schedule(TRAIN_ENTRIES, MINED[seed_index]["train"], num_steps, BATCH_QUERIES, NUM_NEGATIVES, seed=CROSS_ENCODER_SCHEDULE_SEED_BASE + seed_index)
    return CROSS_ENCODER_SCHEDULES[key]


def _rerank_positions(count: int) -> np.ndarray:
    return np.arange(count) if RERANK_QUERIES is None else select_evenly(count, RERANK_QUERIES)


RERANK_EVALUATION_POSITIONS = _rerank_positions(len(EVALUATION_ENTRIES))  # cross-encoder の評価に使う query(評価用の query の中の位置。本番はすべて)
RERANK_VALIDATION_POSITIONS = _rerank_positions(len(VALIDATION_ENTRIES))
RERANK_EVALUATION_ENTRIES = EVALUATION_ENTRIES[RERANK_EVALUATION_POSITIONS]
RERANK_VALIDATION_ENTRIES = VALIDATION_ENTRIES[RERANK_VALIDATION_POSITIONS]
RERANK_EVALUATION_ARTICLES = ENTRY_ARTICLES[RERANK_EVALUATION_ENTRIES]  # ブートストラップのクラスタ(実験 C)


def train_cross_encoder(seed_index: int, learning_rate: float, num_steps: int, main: bool) -> dict:
    start = time.time()
    model = build_cross_encoder(seed_index).to(device)
    schedule = cross_encoder_schedule(seed_index, num_steps)
    window = final_loss_window(num_steps)
    initial_losses = compute_reranking_losses_variable(model, CROSS_ENCODER_TABLES, schedule, range(num_steps - window, num_steps), USE_FP16_AUTOCAST)
    if seed_index not in UNTRAINED_VALIDATION:  # 学習前の cross-encoder(ヘッドの初期化がシードで決まる)の検証用の query の hits
        UNTRAINED_VALIDATION[seed_index] = evaluate_rerank(model, RERANK_VALIDATION_ENTRIES, MINED[seed_index]["validation"][RERANK_VALIDATION_POSITIONS])[0].hits(10)
    train_start = time.time()
    history = train_reranker_variable(
        model, CROSS_ENCODER_TABLES, schedule, peak_learning_rate=learning_rate, warmup_steps=warmup_steps_for(num_steps), min_learning_rate=learning_rate * MIN_LEARNING_RATE_RATIO,
        weight_decay=WEIGHT_DECAY, gradient_clip_threshold=GRADIENT_CLIP_THRESHOLD, use_fp16_autocast=USE_FP16_AUTOCAST,
        init_loss_scale=INIT_LOSS_SCALE, loss_scale_growth_interval=LOSS_SCALE_GROWTH_INTERVAL,
    )
    train_seconds = time.time() - train_start
    assert len(history["loss"]) == num_steps
    train_loss = np.array(history["loss"], dtype=np.float64)
    record = {
        "seed": seed_index, "learning_rate": learning_rate, "num_steps": num_steps, "schedule_hash": history["data_stream_hash"],
        "train_loss": train_loss.astype(np.float32), "initial_train_loss": float(train_loss[:window].mean()), "final_train_loss": float(train_loss[-window:].mean()),
        "initial_model_loss_on_final_window": float(initial_losses.mean()), "initial_model_losses_by_step": initial_losses, "finite": bool(np.isfinite(train_loss).all()),
        "skipped_steps": int(sum(history["step_skipped"])), "clip_rate": float(np.mean(history["gradient_clip_triggered"])), "final_loss_scale": float(history["loss_scale"][-1]),
        "untrained_validation_hits": UNTRAINED_VALIDATION[seed_index], "train_seconds": train_seconds,
    }
    validation, validation_scores = evaluate_rerank(model, RERANK_VALIDATION_ENTRIES, MINED[seed_index]["validation"][RERANK_VALIDATION_POSITIONS])
    record["validation_hits"] = validation.hits(10)
    record["validation_recall"] = validation.recall_at(10)
    record["validation_coverage"] = validation.coverage()
    record["validation_score_std_median"] = median_score_std(validation_scores)
    record["validation_no_rerank_hits"] = MINED[seed_index]["validation_hits"][RERANK_VALIDATION_POSITIONS]
    if main:
        final, final_scores = evaluate_rerank(model, RERANK_EVALUATION_ENTRIES, MINED[seed_index]["evaluation"][RERANK_EVALUATION_POSITIONS])
        record["hits"], record["rank"] = final.hits(10), final.rank
        record["coverage"] = final.coverage()
        record["mean_reciprocal_rank"], record["normalized_discounted_cumulative_gain"] = final.mean_reciprocal_rank(), final.mean_normalized_discounted_cumulative_gain_at_10()
        record["score_std_median"] = median_score_std(final_scores)
        record["no_rerank_hits"] = MINED[seed_index]["evaluation_hits"][RERANK_EVALUATION_POSITIONS]
        if seed_index == UPLOAD_SEED:
            SAVED_STATES[("cross_encoder", UPLOAD_SEED)] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    record["seconds"] = time.time() - start
    del model
    empty_device_cache()
    return record


def learning_loss_statistics(record: dict) -> dict:
    # cross-encoder の P1(a)(6.1 節)の量: 最後の区間の各ステップの、訓練損失と、同じ組での学習前のモデルの損失の差 d_t(学習前より下がれば負)。
    # 差の平均 mean と標準誤差 standard_error = (d_t の標準偏差) / sqrt(区間のステップ数)。更新がなければ d_t = 0
    window = len(record["initial_model_losses_by_step"])
    differences = record["train_loss"][-window:].astype(np.float64) - record["initial_model_losses_by_step"]
    mean = float(differences.mean())
    standard_error = float(differences.std(ddof=1) / math.sqrt(window)) if window >= 2 else float("nan")
    return {"mean": mean, "standard_error": standard_error, "ratio": record["final_train_loss"] / record["initial_model_loss_on_final_window"], "final_loss": record["final_train_loss"]}


def cross_encoder_failure_reasons(record: dict) -> list[str]:
    # P1(a) を満たさない理由(空なら成立): 訓練損失が有限で、(i)差の平均 <= -max(2 x 標準誤差、数値の丸めの床)、かつ(ii)最後の区間の訓練損失 < 一様な予測の損失 ln(1 + n)。
    # 数値の丸めの床 = 16 x 実際に計算される型(FP16 の autocast なら FP16、そうでなければ FP32)の丸めの単位 x 学習前のモデルの損失の大きさ。更新がなければ差は 0 なので (i) は成り立たない
    statistics = learning_loss_statistics(record)
    rounding = 16 * (EPSILON_FP16 if USE_FP16_AUTOCAST else EPSILON_FP32) * record["initial_model_loss_on_final_window"]
    threshold = max(P1_SIGMA_MULTIPLIER * statistics["standard_error"], rounding)
    reasons = []
    if not record["finite"]:
        reasons.append("訓練損失が有限でない")
    if not statistics["mean"] <= -threshold:
        reasons.append(f"損失の差の平均 {statistics['mean']:+.3f} が -{P1_SIGMA_MULTIPLIER:g} x 標準誤差({threshold:.3f})以下でない")
    if not record["final_train_loss"] < UNIFORM_LOSS_CROSS_ENCODER:
        reasons.append(f"最後の区間の訓練損失 {record['final_train_loss']:.4f} が ln(1 + n) = {UNIFORM_LOSS_CROSS_ENCODER:.4f} 未満でない")
    return reasons


# ---------- 判定と統計 ----------
def judge(delta: float, sigma: float) -> str:
    threshold = SIGMA_MULTIPLIER * sigma
    if delta > threshold:
        return "支持"
    if delta < -threshold:
        return "反証"
    return "判定不能"


def combined_sigma(per_seed: np.ndarray, bootstrap_contrast: np.ndarray) -> dict:
    # シードが 1 つのときは、シード間の項を 0 とする
    seed_variance = float(np.var(per_seed, ddof=1)) / len(per_seed) if len(per_seed) >= 2 else 0.0
    bootstrap_variance = float(np.var(bootstrap_contrast, ddof=1))
    return {"sigma": math.sqrt(seed_variance + bootstrap_variance), "seed_term": math.sqrt(seed_variance), "bootstrap_term": math.sqrt(bootstrap_variance)}


VALIDATION_WEIGHTS = cluster_bootstrap_weights(ENTRY_ARTICLES[VALIDATION_ENTRIES], BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED + 1)  # 検証用の query(2 つのラベルで共通)
RERANK_VALIDATION_WEIGHTS = cluster_bootstrap_weights(ENTRY_ARTICLES[RERANK_VALIDATION_ENTRIES], BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED + 3)  # cross-encoder の検証用の query


def bootstrap_mean(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    # query ごとの値の平均の、記事を単位とするクラスタブートストラップの再標本の分布 (反復,)
    return (weights @ np.asarray(values, dtype=np.float64)) / weights.sum(axis=1)


def calibration_metric(validation: dict) -> float:
    # 第 1 段の較正の指標 M: 検証用の passage 単位と記事単位の Recall@10 の平均(どちらの定義にも偏らない)
    return 0.5 * (float(validation["passage_hits"].mean()) + float(validation["article_hits"].mean()))


def calibration_metric_deviation(validation: dict) -> float:
    boot = 0.5 * (bootstrap_mean(VALIDATION_WEIGHTS, validation["passage_hits"]) + bootstrap_mean(VALIDATION_WEIGHTS, validation["article_hits"]))
    return float(boot.std(ddof=1))


# ---------- 較正の選択(6.1 節) ----------
def select_calibrated_index(points: list[float], values: list[float], deviations: list[float], margins: list[float], eligible: list[bool]) -> int | None:
    # 指標が最大の点から SELECTION_SIGMA_MULTIPLIER x 標準偏差以内の候補(P1 を満たす点だけ)のうち、学習の成立の余裕(margins。大きいほど余裕がある)が最大の点。
    # 同点なら小さい学習率。候補がなければ None
    candidates = [i for i, ok in enumerate(eligible) if ok]
    if not candidates:
        return None
    best = max(candidates, key=lambda i: (values[i], -points[i]))
    near = [i for i in candidates if values[best] - values[i] <= SELECTION_SIGMA_MULTIPLIER * deviations[best]]
    return max(near, key=lambda i: (margins[i], -points[i]))


def bootstrap_rerank_gain(records: list[dict]) -> dict:
    # cross-encoder の P1(b): 検証用の query での Recall@10 の、学習前の cross-encoder からの増分(シードごと g_s)のシード平均 gain と、
    # 判定と同じ式の標準偏差 sigma = sqrt(Var_s(g_s) / n + sigma_boot^2)
    difference = np.stack([r["validation_hits"].astype(np.float64) - r["untrained_validation_hits"].astype(np.float64) for r in records])
    per_seed = difference.mean(axis=1)
    resampled = bootstrap_mean(RERANK_VALIDATION_WEIGHTS, difference.mean(axis=0))
    sigma = combined_sigma(per_seed, resampled)
    return {"gain": float(per_seed.mean()), "per_seed_gain": per_seed, **sigma}
```

## 6. 実験 / Experiments



## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/026_inverse_cloze_task_and_two_stage_retrieval.ipynb
