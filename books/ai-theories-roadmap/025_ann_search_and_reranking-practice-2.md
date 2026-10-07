---
title: "ANN 検索とリランキング / ANN Search and Reranking(実装・実験編 2/8)"
---

この記事は後編(実装・実験編 2/8)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/025_ann_search_and_reranking-practice-1)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/025_ann_search_and_reranking-practice-3)。

### 5.3 データ: コーパス・passage の索引・埋め込み・第 1 段の候補

**コーパスと分割**: 024 と同じ、英語 Wikipedia のコーパス(`kojikojiprg/ai-theories-corpus-en`、9,826 記事)から、008 の事前学習とトークナイザの学習に使った先頭の 356 記事を除く **9,470 記事** を使う。
記事の境界は`metadata.json`の`article_offsets`から得る。記事を単位とする学習用(9,070 記事)・検証用(100 記事)・評価用(300 記事)の分割(`split_articles()`、シード 24001)は 024 と同じで、
**分割のダイジェストが 024 の本番の値(`716cfe33d05a2615`)と一致することを確かめる**。008 もトークナイザも、使う記事のどれも一度も見ていない(文字位置の範囲のアサーションで確かめる)。

**コーパス(検索の対象)**: 使う 9,470 記事すべてから、024 と同じ手順(`build_retrieval_set()`、シード 24002。記事を重ならない長さ 128 トークンの passage に分け、記事ごとに最大 12 個を等間隔に選ぶ。
query は長さ 32 トークンで、記事ごとに最大 12 個、元の passage の中の位置はシード付きの乱数で決める)で切り出した passage と query を作る。評価用の記事の部分は 024 の評価用の索引と query と **完全に同じ**
(ダイジェスト`da3e16b96714c0a4`が 024 の本番の値と一致することを確かめる)。関連性の定義は 024 と同じで、query の正例は **同じ記事の、元の passage 以外の passage** である。

**第 1 段の埋め込み**: 024 の学習済みモデル(`kojikojiprg/ai-theories-text-embedding-en`、平均プール・因果マスク、全次元 256、**凍結**)で、コーパスの passage と query を FP32 で埋め込んだ単位ベクトルを、
**1 回だけ計算** して`.cache/025_embeddings/`に置く(入力のダイジェストとモデルの重みのハッシュをキーにする。キャッシュを読んだ場合は、一部を計算し直して一致を確かめる)。
**024 の評価用の索引(3,244 passage)での Recall@10 が、024 の本番の値(0.6477)と一致する** ことを確かめて、取得したモデルとコーパスが 024 のものであることの確認とする。

**評価用の query の選び方**: 評価用の記事 1 本あたり、query のうち等間隔に選んだ最大 $q$ 個を使う(`select_evenly()`)。近似最近傍探索(実験 A〜C)は $q = 3$、並べ替え(実験 D・E)は $q = 5$、実験 F は $q = 2$、
検証用の記事(学習率の較正と前提条件)は $q = 10$ である(スモークテストの値は 5.2 節)。同じ記事の query は互いに独立でないので、ブートストラップは記事を単位とする。

**並べ替えの第 1 段の候補**: 第 1 段(024 のモデルによる、コーパス全体の全探索)の上位 $K = 50$ 件(query の元の passage を除く)を、評価用・検証用の query ごとに求める。
**採掘**: 学習用の記事の query ごとに、**学習用の記事の passage だけ** を対象にした全探索の上位 50 件を求め、並べ替えの学習の困難な負例の候補とする(`mine_candidate_lists()`)。
**学習に使う query と正例**: 採掘の上位 50 件に、同じ記事の passage(query の元の passage を除く)が 1 つ以上ある query だけを学習に使い、正例は、その上位 50 件の中の同じ記事の passage から一様に選ぶ(`usable_training_queries()`)。
推論時に並べ替えモデルが見る正例は、必ず第 1 段の上位 50 件の中にあるので、学習時の正例も同じ範囲から選ぶ。使える query の数と割合、エポック換算はこのセルの出力にある。

**符号化したトークンのキャッシュ**: 記事のトークン化は時間がかかるので、`.cache/025_encoded/`にキャッシュする(コーパスの長さ・記事の境界・トークナイザの指紋をキーにする)。
キャッシュを読んだ場合は、等間隔に選んだ記事を符号化し直して完全に一致することを確かめる。


```python
_t0_data = time.time()
DATA_SECONDS_BY_STAGE: dict[str, float] = {}
PEAK_MEMORY_GIB_BY_STAGE: dict[str, float] = {}


def peak_memory_gib() -> float:
    # このプロセスのこれまでのピークのメモリ使用量(macOS は単位がバイト、Linux は KiB)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024**3 if sys.platform == "darwin" else 1024**2)


def total_memory_gib() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (ValueError, OSError, AttributeError):
        return None


def end_stage(name: str, started: float) -> None:
    DATA_SECONDS_BY_STAGE[name] = time.time() - started
    PEAK_MEMORY_GIB_BY_STAGE[name] = peak_memory_gib()


from huggingface_hub import hf_hub_download  # noqa: E402

# --- トークナイザ: 区切りと終端に使う単一バイトのトークン(語彙を増やさない) ---
_stage = time.time()
tokenizer, _tokenizer_from_hub = load_bpe_id_tokenizer_from_hub(TOKENIZER_REPO_ID)
assert _tokenizer_from_hub, "トークナイザを Hugging Face Hub から取得できなかった"
assert tokenizer.vocab_size == 8192
_bpe = tokenizer.bpe_tokenizer
assert tokenizer.vocab_size == 256 + len(_bpe.merges) and len(_bpe.vocab) == tokenizer.vocab_size  # 特殊トークンがない
_separator_ids, _terminal_ids = tokenizer.encode(chr(SEPARATOR_BYTE)), tokenizer.encode(chr(TERMINAL_BYTE))
assert len(_separator_ids) == len(_terminal_ids) == 1 and _separator_ids != _terminal_ids
SEPARATOR_TOKEN_ID, TERMINAL_TOKEN_ID = int(_separator_ids[0]), int(_terminal_ids[0])
end_stage("トークナイザの取得", _stage)

# --- 参照コーパス(008 の事前学習と英語のトークナイザの学習に使ったコーパス)---
_stage = time.time()
_reference_text = Path(hf_hub_download(REFERENCE_CORPUS_REPO_ID, "corpus.txt", repo_type="dataset")).read_text(encoding="utf-8")
REFERENCE_CORPUS_CHARACTERS = len(_reference_text)
_reference_train_text, _ = split_train_val_text(_reference_text, REFERENCE_VALIDATION_RATIO)
REFERENCE_PRETRAINING_END = len(_reference_train_text)  # 008 とトークナイザが使った範囲の終わり(文字位置)
del _reference_train_text
_manifest_items = list(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")).items())
_reference_items = list(json.loads(REFERENCE_MANIFEST_PATH.read_text(encoding="utf-8")).items())
NUM_REFERENCE_ARTICLES = len(_reference_items)
assert _manifest_items[:NUM_REFERENCE_ARTICLES] == _reference_items, "先頭の記事のタイトルとリビジョン ID が参照コーパスのマニフェストと一致しない"
end_stage("参照コーパスの取得", _stage)

# --- コーパス(Hub から取得する。Wikipedia API にはフォールバックしない)---
_stage = time.time()
_corpus_path = hf_hub_download(CORPUS_REPO_ID, "corpus.txt", repo_type="dataset")
_metadata_path = hf_hub_download(CORPUS_REPO_ID, "metadata.json", repo_type="dataset")
end_stage("コーパスのダウンロード", _stage)
_stage = time.time()
corpus_metadata = json.loads(Path(_metadata_path).read_text(encoding="utf-8"))
assert corpus_metadata["manifest"] == MANIFEST_PATH.name and corpus_metadata["manifest_article_count"] == len(_manifest_items)
assert "article_offsets" in corpus_metadata, "metadata.json に article_offsets がない"
assert os.path.getsize(_corpus_path) == corpus_metadata["raw_bytes"], "コーパスの取得が破損している"
corpus_text = Path(_corpus_path).read_text(encoding="utf-8")
assert corpus_text.startswith(_reference_text) and corpus_text[REFERENCE_CORPUS_CHARACTERS] == "\n", "先頭が参照コーパスと一致しない"
del _reference_text
end_stage("コーパスの読み込み", _stage)

_stage = time.time()
article_spans, ARTICLE_SPANS_SOURCE = locate_wikipedia_article_spans(
    corpus_text, "en", WIKIPEDIA_CACHE_DIR, MANIFEST_PATH, metadata_path=_metadata_path, start_position=0
)
assert ARTICLE_SPANS_SOURCE == "metadata", "article_offsets を metadata.json から取得できなかった"
NUM_ARTICLES = len(article_spans)
assert NUM_ARTICLES == len(_manifest_items)
end_stage("記事の境界の検証", _stage)

# --- 使う記事: 参照コーパスの外に始まる記事だけ ---
CANDIDATE_ARTICLES = find_articles_starting_at_or_after(article_spans, REFERENCE_CORPUS_CHARACTERS)
assert CANDIDATE_ARTICLES == list(range(NUM_REFERENCE_ARTICLES, NUM_ARTICLES)), "使う記事が先頭の記事を除く全記事と一致しない"
assert article_spans[NUM_REFERENCE_ARTICLES - 1]["end"] == REFERENCE_CORPUS_CHARACTERS, "先頭の記事の末尾が参照コーパスの末尾と一致しない"
assert article_spans[CANDIDATE_ARTICLES[0]]["start"] == REFERENCE_CORPUS_CHARACTERS + 1 > REFERENCE_PRETRAINING_END

# --- 記事ごとの符号化(キャッシュあり。キャッシュを読んだ場合は一部を符号化し直して一致を確かめる)---
_stage = time.time()
_candidate_characters = article_spans[-1]["end"] - article_spans[CANDIDATE_ARTICLES[0]]["start"]
ARTICLE_TOKENS, ENCODED_FROM_CACHE = encode_articles_with_cache(
    tokenizer, corpus_text, article_spans, CANDIDATE_ARTICLES, ENCODED_CACHE_DIR / "tokens.npz"
)
ARTICLE_TOKEN_COUNTS = np.array([0 if t is None else len(t) for t in ARTICLE_TOKENS])  # 使わない記事は 0
CANDIDATE_TOKENS = int(ARTICLE_TOKEN_COUNTS.sum())
assert all(ARTICLE_TOKENS[a] is None for a in range(NUM_REFERENCE_ARTICLES)) and all(ARTICLE_TOKENS[a] is not None for a in CANDIDATE_ARTICLES)
TOKEN_BYTE_LENGTHS = np.array([len(tokenizer.id_to_symbol[i]) for i in range(tokenizer.vocab_size)], dtype=np.int64)
for _article in CANDIDATE_ARTICLES:  # 可逆性とバイト数の整合
    _tokens, _span = ARTICLE_TOKENS[_article], article_spans[_article]
    assert int(_tokens.max()) < tokenizer.vocab_size and _tokens.dtype == np.int32
    assert int(TOKEN_BYTE_LENGTHS[_tokens].sum()) == len(corpus_text[_span["start"] : _span["end"]].encode("utf-8"))
_first = article_spans[CANDIDATE_ARTICLES[0]]
_sample = corpus_text[_first["start"] : _first["start"] + 50_000]
assert tokenizer.decode(tokenizer.encode(_sample)) == _sample, "トークナイザのラウンドトリップが一致しない"
assert tokenizer.decode(ARTICLE_TOKENS[CANDIDATE_ARTICLES[-1]].tolist()) == corpus_text[article_spans[-1]["start"] : article_spans[-1]["end"]]
# 区切りと終端のトークンがコーパスに一度も現れない(cross-encoder の入力の区切りと終端が、コーパスの内容と衝突しない)
SEPARATOR_OCCURRENCES = sum(int((ARTICLE_TOKENS[a] == SEPARATOR_TOKEN_ID).sum()) for a in CANDIDATE_ARTICLES)
TERMINAL_OCCURRENCES = sum(int((ARTICLE_TOKENS[a] == TERMINAL_TOKEN_ID).sum()) for a in CANDIDATE_ARTICLES)
assert SEPARATOR_OCCURRENCES == 0 and TERMINAL_OCCURRENCES == 0, "区切り・終端のトークンがコーパスに現れる"
end_stage("記事の符号化と検証", _stage)

_titles = list(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))
LIST_LIKE_ARTICLES = frozenset(a for a in CANDIDATE_ARTICLES if _titles[a].startswith(LIST_LIKE_TITLE_PREFIXES))
CORPUS_CHARACTERS = len(corpus_text)
del corpus_text, _sample
gc.collect()

# --- 記事を単位とする分割(024 と同じ。ダイジェストが 024 の本番の値と一致する)---
_stage = time.time()
MIN_TOKENS_FOR_RETRIEVAL = 2 * PASSAGE_LENGTH
SPLIT = split_articles(ARTICLE_TOKEN_COUNTS, MIN_TOKENS_FOR_RETRIEVAL, CANDIDATE_ARTICLES, NUM_VALIDATION_ARTICLES, NUM_EVALUATION_ARTICLES, SPLIT_SEED)
assert SPLIT.digest() == SPLIT_DIGEST_024, f"分割が 024 の本番の分割と一致しない: {SPLIT.digest()}"
_parts = [set(SPLIT.train.tolist()), set(SPLIT.validation.tolist()), set(SPLIT.evaluation.tolist())]
assert not (_parts[0] & _parts[1]) and not (_parts[0] & _parts[2]) and not (_parts[1] & _parts[2]), "分割が重なっている"
assert _parts[0] | _parts[1] | _parts[2] == set(CANDIDATE_ARTICLES) and sum(len(p) for p in _parts) == len(CANDIDATE_ARTICLES)
assert len(SPLIT.train) == 9070 and len(SPLIT.validation) == NUM_VALIDATION_ARTICLES and len(SPLIT.evaluation) == NUM_EVALUATION_ARTICLES
_used_starts = np.array([article_spans[a]["start"] for a in sorted(_parts[0] | _parts[1] | _parts[2])])
assert _used_starts.min() >= REFERENCE_CORPUS_CHARACTERS > REFERENCE_PRETRAINING_END, "008 の事前学習の範囲にかかる記事を使っている"
end_stage("分割", _stage)

# --- コーパス: 使う記事すべてから 024 と同じ手順で切り出した passage と query ---
_stage = time.time()
CORPUS = build_retrieval_set(
    ARTICLE_TOKENS, CANDIDATE_ARTICLES, PASSAGE_LENGTH, QUERY_LENGTH, MAX_PASSAGES_PER_ARTICLE, MAX_QUERIES_PER_ARTICLE, EVALUATION_SET_SEED
)
_evaluation_set = build_retrieval_set(
    ARTICLE_TOKENS, SPLIT.evaluation, PASSAGE_LENGTH, QUERY_LENGTH, MAX_PASSAGES_PER_ARTICLE, MAX_QUERIES_PER_ARTICLE, EVALUATION_SET_SEED
)
assert _evaluation_set.digest() == EVALUATION_DIGEST_024, f"評価用の索引と query が 024 の本番の集合と一致しない: {_evaluation_set.digest()}"
NUM_PASSAGES, NUM_CORPUS_QUERIES = CORPUS.num_passages, CORPUS.num_queries
PASSAGE_ARTICLES, QUERY_ARTICLES, QUERY_SOURCES = CORPUS.passage_articles, CORPUS.query_articles, CORPUS.query_sources
PASSAGES_BY_ARTICLE = article_passage_ranges(PASSAGE_ARTICLES)  # 記事 -> (先頭の passage, 末尾 + 1)(同じ記事の passage は連続している)
assert all(PASSAGE_ARTICLES[a:b].tolist() == [art] * (b - a) for art, (a, b) in PASSAGES_BY_ARTICLE.items())
assert (CORPUS.num_positives() >= 1).all() and CORPUS.passages.shape[1] == PASSAGE_LENGTH and CORPUS.queries.shape[1] == QUERY_LENGTH
_offsets = CORPUS.query_offsets[:, None] + np.arange(QUERY_LENGTH)[None, :]
assert np.array_equal(CORPUS.queries, np.take_along_axis(CORPUS.passages[CORPUS.query_sources], _offsets, axis=1)), "query が元の passage の一部でない"
assert np.array_equal(PASSAGE_ARTICLES[CORPUS.query_sources], QUERY_ARTICLES)
# 評価用の記事の部分が 024 の評価用の索引・query と一致する(コーパスの passage・query の並びは記事の昇順)
EVALUATION_PASSAGE_INDEX = np.nonzero(np.isin(PASSAGE_ARTICLES, SPLIT.evaluation))[0]
EVALUATION_QUERY_INDEX = np.nonzero(np.isin(QUERY_ARTICLES, SPLIT.evaluation))[0]
assert np.array_equal(CORPUS.passages[EVALUATION_PASSAGE_INDEX], _evaluation_set.passages) and np.array_equal(CORPUS.queries[EVALUATION_QUERY_INDEX], _evaluation_set.queries)
assert np.array_equal(PASSAGE_ARTICLES[EVALUATION_PASSAGE_INDEX], _evaluation_set.passage_articles)
VALIDATION_QUERY_INDEX = np.nonzero(np.isin(QUERY_ARTICLES, SPLIT.validation))[0]
TRAIN_QUERY_INDEX = np.nonzero(np.isin(QUERY_ARTICLES, SPLIT.train))[0]
TRAIN_PASSAGE_INDEX = np.nonzero(np.isin(PASSAGE_ARTICLES, SPLIT.train))[0]  # 負例の採掘とランダムな負例の対象(学習用の記事の passage だけ)
assert len(TRAIN_QUERY_INDEX) + len(VALIDATION_QUERY_INDEX) + len(EVALUATION_QUERY_INDEX) == NUM_CORPUS_QUERIES
NUM_POSITIVES_BY_QUERY = np.bincount(PASSAGE_ARTICLES)[QUERY_ARTICLES] - 1  # query ごとの正解の数(コーパス全体での、同じ記事の他の passage の数)


def select_queries_per_article(query_indices: np.ndarray, per_article: int) -> np.ndarray:
    # 記事ごとに、query のうち等間隔に選んだ最大 per_article 個(コーパスの query の添字。昇順)
    chosen = []
    articles = QUERY_ARTICLES[query_indices]
    for article in np.unique(articles):
        members = query_indices[articles == article]
        chosen.extend(members[select_evenly(len(members), per_article)].tolist())
    return np.array(sorted(chosen), dtype=np.int64)


ANN_QUERY_INDEX = select_queries_per_article(EVALUATION_QUERY_INDEX, LEVEL_SETTINGS["ANN_QUERIES_PER_ARTICLE"])
RERANK_QUERY_INDEX = select_queries_per_article(EVALUATION_QUERY_INDEX, LEVEL_SETTINGS["RERANK_QUERIES_PER_ARTICLE"])[: LEVEL_SETTINGS["RERANK_MAX_QUERIES"]]
F_QUERY_INDEX = select_queries_per_article(RERANK_QUERY_INDEX, LEVEL_SETTINGS["RERANK_QUERIES_PER_ARTICLE_F"])[: LEVEL_SETTINGS["F_MAX_QUERIES"]]  # 並べ替えの query の部分集合
CALIBRATION_QUERY_INDEX = select_queries_per_article(VALIDATION_QUERY_INDEX, LEVEL_SETTINGS["VALIDATION_QUERIES_PER_ARTICLE"])
assert set(F_QUERY_INDEX.tolist()) <= set(RERANK_QUERY_INDEX.tolist()) and set(RERANK_QUERY_INDEX.tolist()) <= set(EVALUATION_QUERY_INDEX.tolist())
end_stage("コーパスの構成", _stage)

# --- 並べ替えの学習・評価に使うトークン(コーパスの passage と query。デバイスに置く)---
PASSAGE_TOKENS_CPU = torch.from_numpy(CORPUS.passages)  # int64
QUERY_TOKENS_CPU = torch.from_numpy(CORPUS.queries)
PASSAGE_TOKENS = PASSAGE_TOKENS_CPU.to(device)
QUERY_TOKENS = QUERY_TOKENS_CPU.to(device)
for _article in range(NUM_ARTICLES):  # 記事のトークンは passage と query に取り出したので、解放する(passage の切り出しは済んだ)
    ARTICLE_TOKENS[_article] = None
gc.collect()
DATA_SECONDS = time.time() - _t0_data
```


    tokenizer.json:   0%|          | 0.00/661k [00:00<?, ?B/s]



    corpus.txt: reconstructing file:   0%|          |  0.00B / 24.3MB            



    corpus.txt: downloading bytes:           |  0.00B            



    corpus.txt: reconstructing file:   0%|          |  0.00B /  546MB            



    corpus.txt: downloading bytes:           |  0.00B            



    metadata.json:   0%|          | 0.00/1.22M [00:00<?, ?B/s]



```python
# --- 008(cross-encoder と dual encoder の起点)と 024(第 1 段)のモデルの構成と重み ---
_stage = time.time()
REFERENCE_CONFIG = json.loads(Path(hf_hub_download(REFERENCE_MODEL_REPO_ID, "config.json")).read_text(encoding="utf-8"))
REFERENCE_STATE = {k: v.clone() for k, v in torch.load(hf_hub_download(REFERENCE_MODEL_REPO_ID, "model_state.pt"), map_location="cpu").items()}
FIRST_STAGE_CONFIG = json.loads(Path(hf_hub_download(FIRST_STAGE_REPO_ID, "config.json")).read_text(encoding="utf-8"))
FIRST_STAGE_STATE = {k: v.clone() for k, v in torch.load(hf_hub_download(FIRST_STAGE_REPO_ID, "model_state.pt"), map_location="cpu").items()}
assert REFERENCE_CONFIG["vocabulary_size"] == tokenizer.vocab_size == 8192 and REFERENCE_CONFIG["d_model"] == EMBEDDING_DIMENSION
assert REFERENCE_CONFIG["sequence_length"] >= QUERY_LENGTH + 1 + PASSAGE_LENGTH + 1 and REFERENCE_CONFIG["tie_embeddings"]
assert FIRST_STAGE_CONFIG["pooling"] == "mean" and FIRST_STAGE_CONFIG["attention"] == "causal" and FIRST_STAGE_CONFIG["embedding_dimension"] == EMBEDDING_DIMENSION
assert FIRST_STAGE_CONFIG["query_length"] == QUERY_LENGTH and FIRST_STAGE_CONFIG["passage_length"] == PASSAGE_LENGTH


def build_gpt(config: dict) -> GPTLanguageModel:
    # 006・008・024 と同じ構成(RoPE・RMSNorm・SwiGLU・正規化前置・重み共有)
    return GPTLanguageModel(
        vocabulary_size=config["vocabulary_size"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        max_sequence_length=config["sequence_length"],
        positional_transform=RotaryPositionEmbedding(config["d_model"] // config["num_heads"], max_position=config["sequence_length"]),
        normalization_factory=RMSNorm,
        feed_forward_factory=functools.partial(SwiGLUFeedForwardNetwork, config["d_model"], config["swiglu_d_ff"]),
        tie_embeddings=config["tie_embeddings"],
        dropout=config["dropout"],
    )


def state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for _, value in sorted(state.items()):
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def build_first_stage_model() -> TextEmbeddingModel:
    backbone = build_gpt(FIRST_STAGE_CONFIG)
    backbone.load_state_dict(FIRST_STAGE_STATE)
    return TextEmbeddingModel(backbone, pooling="mean", attention="causal").eval()


FIRST_STAGE_HASH = state_hash(FIRST_STAGE_STATE)
REFERENCE_HASH = state_hash(REFERENCE_STATE)
EPSILON_FP32, EPSILON_FP64 = float(torch.finfo(torch.float32).eps), float(torch.finfo(torch.float64).eps)

# --- コーパスの passage と query の埋め込み(第 1 段。FP32。1 回だけ計算して .cache/ に置く)---
EMBEDDING_KEY = hashlib.sha256((CORPUS.digest() + FIRST_STAGE_HASH).encode()).hexdigest()[:24]
EMBEDDING_CACHE_PATH = EMBEDDING_CACHE_DIR / f"embeddings_{EMBEDDING_KEY}.npz"
_first_stage_model = build_first_stage_model().to(device)


def embed_tokens(tokens: np.ndarray) -> np.ndarray:
    return encode_tokens(_first_stage_model, torch.from_numpy(tokens), EMBED_BATCH)["by_dimension"][EMBEDDING_DIMENSION].cpu().numpy()


EMBEDDINGS_FROM_CACHE = EMBEDDING_CACHE_PATH.exists()
if not EMBEDDINGS_FROM_CACHE:
    _passages_computed, _queries_computed = embed_tokens(CORPUS.passages), embed_tokens(CORPUS.queries)
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(EMBEDDING_CACHE_PATH, passages=_passages_computed, queries=_queries_computed)
    del _passages_computed, _queries_computed
with np.load(EMBEDDING_CACHE_PATH) as _stored:  # 計算した場合も、書き出したキャッシュを読み戻して使う(以降の数値はすべて読み戻した値から作る)
    PASSAGE_EMBEDDINGS_NP, QUERY_EMBEDDINGS_NP = _stored["passages"], _stored["queries"]
assert PASSAGE_EMBEDDINGS_NP.shape == (NUM_PASSAGES, EMBEDDING_DIMENSION) and QUERY_EMBEDDINGS_NP.shape == (NUM_CORPUS_QUERIES, EMBEDDING_DIMENSION)
assert PASSAGE_EMBEDDINGS_NP.dtype == QUERY_EMBEDDINGS_NP.dtype == np.float32
# キャッシュの検証: 等間隔に選んだ一部を計算し直して、キャッシュと一致する(許容は、同じ経路の 2 回の差と丸めの単位 x 値の大きさから導く)
_sample_passages = np.linspace(0, NUM_PASSAGES - 1, 256).astype(int)
_again = embed_tokens(CORPUS.passages[_sample_passages])
_repeat = float(np.abs(_again - embed_tokens(CORPUS.passages[_sample_passages])).max())
EMBEDDING_TOLERANCE = 16 * max(_repeat, EPSILON_FP32 * float(np.abs(_again).max()))
_cache_difference = float(np.abs(_again - PASSAGE_EMBEDDINGS_NP[_sample_passages]).max())
assert _cache_difference <= EMBEDDING_TOLERANCE, (_cache_difference, EMBEDDING_TOLERANCE)
_norm_error = float(np.abs(np.linalg.norm(PASSAGE_EMBEDDINGS_NP, axis=1) - 1).max())
assert _norm_error <= 16 * EPSILON_FP32 and float(np.abs(np.linalg.norm(QUERY_EMBEDDINGS_NP, axis=1) - 1).max()) <= 16 * EPSILON_FP32
del _first_stage_model
empty_device_cache()
PASSAGE_EMBEDDINGS, QUERY_EMBEDDINGS = torch.from_numpy(PASSAGE_EMBEDDINGS_NP), torch.from_numpy(QUERY_EMBEDDINGS_NP)
PASSAGE_EMBEDDINGS_DEVICE, QUERY_EMBEDDINGS_DEVICE = PASSAGE_EMBEDDINGS.to(device), QUERY_EMBEDDINGS.to(device)
end_stage("埋め込み", _stage)

# --- 024 の評価用の索引(3,244 passage)での Recall@10 が、024 の本番の値と一致する ---
_reproduced = evaluate_embeddings(
    QUERY_EMBEDDINGS[EVALUATION_QUERY_INDEX], PASSAGE_EMBEDDINGS[EVALUATION_PASSAGE_INDEX], _evaluation_set
).recall_at(10)
assert abs(_reproduced - REFERENCE_RECALL_024) <= 0.005, f"024 の評価用の索引での Recall@10 が 024 の本番の値と一致しない: {_reproduced:.4f}"

# --- 第 1 段の候補(並べ替える候補)と、採掘(学習用の記事の query ごとに、学習用の記事の passage だけを対象にした全探索の上位)---
_stage = time.time()
MAX_RERANK_K = max(RERANK_KS)


def first_stage_candidates(query_indices: np.ndarray, k: int) -> np.ndarray:
    # コーパス全体の全探索(024 のモデルの内積)の上位 k 件(query の元の passage を除く)
    ids, _ = exact_search(QUERY_EMBEDDINGS_DEVICE[torch.as_tensor(query_indices, device=device)], PASSAGE_EMBEDDINGS_DEVICE, k, excluded=QUERY_SOURCES[query_indices])
    return ids


RERANK_CANDIDATE_IDS_FULL = first_stage_candidates(RERANK_QUERY_INDEX, MAX_RERANK_K)  # 実験 F の K の掃引のため、最大の K まで求める
CALIBRATION_CANDIDATE_IDS_FULL = first_stage_candidates(CALIBRATION_QUERY_INDEX, MAX_RERANK_K)
RERANK_CANDIDATE_IDS = RERANK_CANDIDATE_IDS_FULL[:, :RERANK_CANDIDATES]
CALIBRATION_CANDIDATE_IDS = CALIBRATION_CANDIDATE_IDS_FULL[:, :RERANK_CANDIDATES]
assert NUM_PASSAGES > MAX_RERANK_K and (RERANK_CANDIDATE_IDS != QUERY_SOURCES[RERANK_QUERY_INDEX][:, None]).all()
MINED_CANDIDATES = mine_candidate_lists(
    QUERY_EMBEDDINGS_DEVICE[torch.as_tensor(TRAIN_QUERY_INDEX, device=device)], PASSAGE_EMBEDDINGS_DEVICE, TRAIN_PASSAGE_INDEX, QUERY_SOURCES[TRAIN_QUERY_INDEX], MINING_DEPTH
)
_train_passage_set = np.zeros(NUM_PASSAGES, dtype=bool)
_train_passage_set[TRAIN_PASSAGE_INDEX] = True
assert _train_passage_set[MINED_CANDIDATES].all() and (MINED_CANDIDATES != QUERY_SOURCES[TRAIN_QUERY_INDEX][:, None]).all(), "採掘した候補に学習用の記事の外の passage、または query の元の passage が入っている"
MINED_SAME_ARTICLE = int((PASSAGE_ARTICLES[MINED_CANDIDATES] == QUERY_ARTICLES[TRAIN_QUERY_INDEX][:, None]).sum())  # 採掘の上位のうち、同じ記事の passage(負例から除く。正例の候補になる)
# 学習に使える query: 採掘の上位 K 件に、同じ記事の passage(元の passage を除く)が 1 つ以上ある query(正例は、その上位の中の同じ記事の passage から選ぶ)
USABLE_TRAIN_QUERY_MASK = usable_training_queries(TRAIN_QUERY_INDEX, MINED_CANDIDATES, QUERY_ARTICLES, QUERY_SOURCES, PASSAGE_ARTICLES)
NUM_USABLE_TRAIN_QUERIES = int(USABLE_TRAIN_QUERY_MASK.sum())
assert NUM_USABLE_TRAIN_QUERIES > BATCH_QUERIES * 2, "学習に使える query が少なすぎる"
gc.collect()
empty_device_cache()  # 採掘の途中の大きな中間のテンソルのキャッシュを解放する(以降の学習のメモリのため)
end_stage("第 1 段の候補と採掘", _stage)

# 第 1 段の順位のまま(並べ替えなし)の指標と、上位 K 件に正例が入る割合(coverage)
FIRST_STAGE_ORDER_SCORES = -np.arange(MAX_RERANK_K, dtype=np.float64)[None, :]


def first_stage_result(query_indices: np.ndarray, candidate_ids_full: np.ndarray, k: int):
    ids = candidate_ids_full[:, :k]
    return rank_reranked_candidates(
        np.broadcast_to(FIRST_STAGE_ORDER_SCORES[:, :k], ids.shape), ids, QUERY_ARTICLES[query_indices], PASSAGE_ARTICLES, NUM_POSITIVES_BY_QUERY[query_indices]
    )


FIRST_STAGE_RERANK = {k: first_stage_result(RERANK_QUERY_INDEX, RERANK_CANDIDATE_IDS_FULL, k) for k in RERANK_KS}
FIRST_STAGE_CALIBRATION = {k: first_stage_result(CALIBRATION_QUERY_INDEX, CALIBRATION_CANDIDATE_IDS_FULL, k) for k in RERANK_KS}
COVERAGE = FIRST_STAGE_RERANK[RERANK_CANDIDATES].coverage()  # 並べ替えで到達できる Recall@10 の上限(K = 50)
FIRST_STAGE_RECALL = FIRST_STAGE_RERANK[RERANK_CANDIDATES].recall_at(10)  # 並べ替えなしの Recall@10(K によらない: 上位 10 件は同じ)
assert math.isclose(FIRST_STAGE_RERANK[10].recall_at(10), FIRST_STAGE_RECALL, abs_tol=1e-12)
RANDOM_RECALL = expected_random_recall(NUM_POSITIVES_BY_QUERY[RERANK_QUERY_INDEX], NUM_PASSAGES - 1, 10)
DATA_SECONDS = time.time() - _t0_data
DATA_PEAK_MEMORY_GIB = peak_memory_gib()
_total_memory = total_memory_gib()
_memory_limit = RAM_FRACTION_LIMIT * min(COLAB_RAM_GIB, _total_memory if _total_memory else COLAB_RAM_GIB)
assert DATA_PEAK_MEMORY_GIB <= _memory_limit, f"データの準備のピークのメモリ {DATA_PEAK_MEMORY_GIB:.1f} GiB が Colab の RAM に収まる見込みを超える({_memory_limit:.1f} GiB)"

print(
    f"コーパス: {CORPUS_CHARACTERS:,} 文字(取得元 Hugging Face Hub、{os.path.getsize(_corpus_path) / 1e6:,.0f} MB)、記事 {NUM_ARTICLES:,} 件(境界の取得元 {ARTICLE_SPANS_SOURCE})。"
    f"先頭の {NUM_REFERENCE_ARTICLES} 件は参照コーパス(008 の事前学習とトークナイザの学習に使ったコーパス、{REFERENCE_CORPUS_CHARACTERS:,} 文字)と同一で、使わない"
)
print(
    f"使う記事: {len(CANDIDATE_ARTICLES):,} 件({_candidate_characters:,} 文字、{CANDIDATE_TOKENS:,} トークン。記事の長さの中央値 {int(np.median(ARTICLE_TOKEN_COUNTS[CANDIDATE_ARTICLES])):,})。"
    f"リスト系の記事の割合 {len(LIST_LIKE_ARTICLES) / len(CANDIDATE_ARTICLES):.1%}。符号化のキャッシュ: {'読み込み' if ENCODED_FROM_CACHE else '新規に作成'}"
)
print(
    f"トークナイザ: 語彙サイズ {tokenizer.vocab_size}(特殊トークンなし)。区切り: 単一バイト 0x{SEPARATOR_BYTE:02X}(ID {SEPARATOR_TOKEN_ID})、終端: 0x{TERMINAL_BYTE:02X}(ID {TERMINAL_TOKEN_ID})。"
    f"コーパスでの出現回数は 区切り {SEPARATOR_OCCURRENCES}・終端 {TERMINAL_OCCURRENCES}(使用する記事のすべて)"
)
print(
    f"分割(ダイジェスト {SPLIT.digest()}、024 の本番の値と一致): 学習用 {len(SPLIT.train):,} 記事・検証用 {len(SPLIT.validation)} 記事・評価用 {len(SPLIT.evaluation)} 記事。"
    f"使う記事の開始位置の最小 {int(_used_starts.min()):,} は参照コーパスの全体({REFERENCE_CORPUS_CHARACTERS:,} 文字)の外"
)
print(
    f"コーパス(ダイジェスト {CORPUS.digest()}): passage {NUM_PASSAGES:,} 個(記事 {len(PASSAGES_BY_ARTICLE):,} 本、1 記事あたり平均 {NUM_PASSAGES / len(PASSAGES_BY_ARTICLE):.2f} 個)、query {NUM_CORPUS_QUERIES:,} 個。"
    f"評価用の記事の部分は 024 の評価用の索引・query と一致(passage {len(EVALUATION_PASSAGE_INDEX):,}・query {len(EVALUATION_QUERY_INDEX):,}、ダイジェスト {_evaluation_set.digest()} は 024 の本番の値と一致)。"
    f"学習用の passage {len(TRAIN_PASSAGE_INDEX):,}・query {len(TRAIN_QUERY_INDEX):,}、検証用の query {len(VALIDATION_QUERY_INDEX):,}"
)
print(
    f"評価に使う query: 近似最近傍探索 {len(ANN_QUERY_INDEX):,}、並べ替え {len(RERANK_QUERY_INDEX):,}(実験 F は {len(F_QUERY_INDEX):,})、較正 {len(CALIBRATION_QUERY_INDEX):,}。"
    f"正解の数(コーパス全体)の中央値: 並べ替えの query {int(np.median(NUM_POSITIVES_BY_QUERY[RERANK_QUERY_INDEX]))}"
)
print(
    f"第 1 段の埋め込み(024 のモデル、重みのハッシュ {FIRST_STAGE_HASH}、008 のハッシュ {REFERENCE_HASH}): キャッシュ {'読み込み' if EMBEDDINGS_FROM_CACHE else '新規に作成'}"
    f"(キャッシュの再計算との最大差 {_cache_difference:.2e} <= 許容 {EMBEDDING_TOLERANCE:.2e}、ノルムの誤差 {_norm_error:.1e})。"
    f"024 の評価用の索引での Recall@10 {_reproduced:.4f}(024 の本番の値 {REFERENCE_RECALL_024}、差 {abs(_reproduced - REFERENCE_RECALL_024):.4f} <= 0.005: OK)"
)
print(
    f"第 1 段(コーパス全体の全探索)の並べ替えの query での値: ランダムな順位の Recall@10 の期待値 {RANDOM_RECALL:.4f}、並べ替えなしの Recall@10 {FIRST_STAGE_RECALL:.4f}。"
    "上位 K 件に正例が入る割合(coverage): " + "、".join(f"K = {k}: {FIRST_STAGE_RERANK[k].coverage():.4f}" for k in RERANK_KS)
    + "。較正の query: " + "、".join(f"K = {k}: {FIRST_STAGE_CALIBRATION[k].coverage():.4f}" for k in RERANK_KS)
)
print(
    f"採掘: 学習用の query {len(TRAIN_QUERY_INDEX):,} 個の、学習用の passage だけを対象にした全探索の上位 {MINING_DEPTH} 件のうち、同じ記事の passage(負例から除く)は "
    f"{MINED_SAME_ARTICLE:,} 件({MINED_SAME_ARTICLE / MINED_CANDIDATES.size:.2%})"
)
print(
    f"学習に使える query(採掘の上位 {MINING_DEPTH} 件に、同じ記事の passage が 1 つ以上ある): {NUM_USABLE_TRAIN_QUERIES:,} 個(学習用の query {len(TRAIN_QUERY_INDEX):,} 個の {NUM_USABLE_TRAIN_QUERIES / len(TRAIN_QUERY_INDEX):.2%})。"
    "見る事例数(8 T 個の query)の、使える query に対するエポック換算: "
    + "、".join(f"T = {t}: {BATCH_QUERIES * t:,} 個で {BATCH_QUERIES * t / NUM_USABLE_TRAIN_QUERIES:.3f} エポック" for t in STEP_CANDIDATES)
)
print("データの準備の時間(段階ごと): " + "、".join(f"{k} {v:.1f} 秒" for k, v in DATA_SECONDS_BY_STAGE.items()) + f"。合計 {DATA_SECONDS:.1f} 秒、ピークのメモリ {DATA_PEAK_MEMORY_GIB:.2f} GiB(Colab 無料枠 {COLAB_RAM_GIB} GiB の {RAM_FRACTION_LIMIT:.0%} 以下: OK)")
```


    config.json:   0%|          | 0.00/305 [00:00<?, ?B/s]



    model_state.pt: reconstructing file:   0%|          |  0.00B / 21.0MB            



    model_state.pt: downloading bytes:           |  0.00B            



    config.json:   0%|          | 0.00/732 [00:00<?, ?B/s]



    model_state.pt: reconstructing file:   0%|          |  0.00B / 29.4MB            



    model_state.pt: downloading bytes:           |  0.00B            


    コーパス: 542,892,375 文字(取得元 Hugging Face Hub、546 MB)、記事 9,826 件(境界の取得元 metadata)。先頭の 356 件は参照コーパス(008 の事前学習とトークナイザの学習に使ったコーパス、24,214,546 文字)と同一で、使わない
    使う記事: 9,470 件(518,677,828 文字、147,610,040 トークン。記事の長さの中央値 14,267)。リスト系の記事の割合 30.2%。符号化のキャッシュ: 新規に作成
    トークナイザ: 語彙サイズ 8192(特殊トークンなし)。区切り: 単一バイト 0x1F(ID 3551)、終端: 0x1E(ID 3550)。コーパスでの出現回数は 区切り 0・終端 0(使用する記事のすべて)
    分割(ダイジェスト 716cfe33d05a2615、024 の本番の値と一致): 学習用 9,070 記事・検証用 100 記事・評価用 300 記事。使う記事の開始位置の最小 24,214,547 は参照コーパスの全体(24,214,546 文字)の外
    コーパス(ダイジェスト 1d8d778c8c3776d1): passage 94,486 個(記事 9,158 本、1 記事あたり平均 10.32 個)、query 94,066 個。評価用の記事の部分は 024 の評価用の索引・query と一致(passage 3,244・query 3,244、ダイジェスト da3e16b96714c0a4 は 024 の本番の値と一致)。学習用の passage 90,223・query 89,803、検証用の query 1,019
    評価に使う query: 近似最近傍探索 890、並べ替え 1,446(実験 F は 600)、較正 877。正解の数(コーパス全体)の中央値: 並べ替えの query 11
    第 1 段の埋め込み(024 のモデル、重みのハッシュ 7adcb36e87887b8a、008 のハッシュ 9e2cba765a2ae1e2): キャッシュ 新規に作成(キャッシュの再計算との最大差 0.00e+00 <= 許容 7.92e-07、ノルムの誤差 1.2e-07)。024 の評価用の索引での Recall@10 0.6477(024 の本番の値 0.6477、差 0.0000 <= 0.005: OK)
    第 1 段(コーパス全体の全探索)の並べ替えの query での値: ランダムな順位の Recall@10 の期待値 0.0011、並べ替えなしの Recall@10 0.2199。上位 K 件に正例が入る割合(coverage): K = 10: 0.2199、K = 20: 0.2988、K = 50: 0.4094、K = 100: 0.4959。較正の query: K = 10: 0.2725、K = 20: 0.3615、K = 50: 0.4675、K = 100: 0.5644
    採掘: 学習用の query 89,803 個の、学習用の passage だけを対象にした全探索の上位 50 件のうち、同じ記事の passage(負例から除く)は 111,961 件(2.49%)
    学習に使える query(採掘の上位 50 件に、同じ記事の passage が 1 つ以上ある): 39,303 個(学習用の query 89,803 個の 43.77%)。見る事例数(8 T 個の query)の、使える query に対するエポック換算: T = 1024: 8,192 個で 0.208 エポック、T = 512: 4,096 個で 0.104 エポック、T = 256: 2,048 個で 0.052 エポック
    データの準備の時間(段階ごと): トークナイザの取得 5.3 秒、参照コーパスの取得 3.2 秒、コーパスのダウンロード 5.6 秒、コーパスの読み込み 13.3 秒、記事の境界の検証 0.0 秒、記事の符号化と検証 427.2 秒、分割 0.1 秒、コーパスの構成 4.2 秒、埋め込み 59.7 秒、第 1 段の候補と採掘 2.5 秒。合計 523.4 秒、ピークのメモリ 5.48 GiB(Colab 無料枠 12.7 GiB の 75% 以下: OK)




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/025_ann_search_and_reranking.ipynb
