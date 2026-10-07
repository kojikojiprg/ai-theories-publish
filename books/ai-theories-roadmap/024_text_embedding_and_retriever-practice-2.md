---
title: "テキスト埋め込みと retriever / Text Embedding and Retriever(実装・実験編 2/7)"
---

この記事は後編(実装・実験編 2/7)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/024_text_embedding_and_retriever-practice-1)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/024_text_embedding_and_retriever-practice-3)。

### 5.3 データ: 記事の分割・符号化・評価用の索引と query

**コーパスとトークナイザ**: 英語 Wikipedia のコーパス(`kojikojiprg/ai-theories-corpus-en`、9,826 記事)と、008 の BPE(Byte Pair Encoding)トークナイザ
(語彙サイズ 8192、`kojikojiprg/ai-theories-tokenizer-en`)を使う。記事の境界は`metadata.json`の`article_offsets`から得て、`corpus.txt`との整合を検証する。
トークナイザに特殊トークンがないこと(終端トークンの有無)を確かめて印字する。

**使う記事と、008・トークナイザの学習範囲との関係**: このコーパスの先頭の 356 記事は、008 の事前学習とトークナイザの学習に使ったコーパス
(`kojikojiprg/ai-theories-corpus-en-pretraining`、以下「参照コーパス」)と同一である(タイトル・リビジョン ID・本文を確かめる)。008 は参照コーパスの先頭 95% だけで事前学習し、
トークナイザも同じ範囲で学習している。**先頭の 356 記事は、学習用・検証用・評価用のどれにも使わない。** 使うのは、参照コーパスの全体の外に始まる 9,470 記事で、
008 もトークナイザも、そのどの記事も一度も見ていない(使う記事の開始位置の最小が参照コーパスの長さ以上であることを、文字位置の範囲のアサーションで確かめる)。
そのため、C2(008 の重みから始める)が事前学習で記事の内容を記憶していることによる有利さは、評価に混ざらない。

追加の記事は、Wikipedia の長大記事の一覧(Special:LongPages)から長い順に選んだもので、平均的な記事ではない。**約 30% がリスト系の記事**(タイトルが`List of`・`Timeline of`・
`Deaths in`などで始まる。実際の割合は下のセルの出力に印字する)である。リスト系の記事は、列挙の構造が繰り返されるため、同じ記事の区間どうしの語の重なりが通常の記事と異なりうる
(本トピックでは測っていない)。結果の一般化の制約として 7 節で扱う。

**記事を単位とする分割**(`src/data/retrieval.py`の`split_articles()`、決定的。使う記事の中から選ぶ):

| 部分 | 記事の数 | 用途 |
|---|---|---|
| 学習用 | 使う記事の残りの全部(検索に使えない短い記事を含む) | 対照学習の組の切り出し(組を作れない短い記事は使われない) |
| 検証用 | 100 | 学習率の較正と、学習の途中の評価(実験 B・C の診断量)にのみ使う |
| 評価用 | 300 | 判定と診断量に使う |

検証用・評価用の記事は、passage を 2 個以上持つ記事(トークン数が $2 L_p$ 以上)から、固定したシードの並べ替えの先頭から選ぶ(評価用が先頭の 300、検証用が次の 100)。
3 つの部分は互いに素で、和集合が使う記事になる(このセルのアサーションと、分割のダイジェストの印字で確かめる)。

**評価用の索引と query**: 評価用の記事を、重ならない長さ $L_p = 128$ の passage に分けて索引にする(記事の全体から等間隔に選び、記事ごとに最大 12 個)。
query は、索引の passage のうち記事ごとに等間隔に選んだ最大 12 個の一部(長さ $L_q = 32$、passage の中の位置はシード付きの乱数で決める)で、**その query を含む passage は、
その query の候補から除く**。正解は同じ記事の他の passage である。記事ごとの passage 数に上限を設けるのは、記事の長さが極端に偏る(中央値 約 1.4 万トークン、最長 約 12.5 万トークン)ので、
上限がないと最長の記事 1 本が索引の多くを占め、正解が大量にあって検索が易しくなるためである。検証用の集合も同じ構成で作る。

**学習の組**: 学習用の記事から、長さ $L_q = 32$ の query 側の区間と長さ $L_p = 128$ の passage 側の区間を、**重ならないように** 切り出す。Contriever(independent cropping)は
2 つの区間が重なることを許すが、重なると語の一致だけで正例を当てられるので、本トピックは重ならないようにする。1 つのバッチ(64 組)には同じ記事の組を 2 つ以上入れない。
記事は、長さ $L_p$ の窓の数(最大 12 で頭打ち)に比例する確率で非復元抽出する。

**学習用の部分の索引(汎化の差の診断量)**: 学習用の記事のうち固定した 120 本で、評価用と同じ構成の索引と query(`train_probe`)を作り、各学習の最終ステップで Recall@10 を測る。
学習用の記事は学習で見ているので、評価用との差は汎化の差と分布の違い(索引の大きさの違いを含む)の両方を含む。そのため差の絶対値ではなく、**条件間での差の違い** を読む。


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


_stage = time.time()
tokenizer, _tokenizer_from_hub = load_bpe_id_tokenizer_from_hub(TOKENIZER_REPO_ID)
assert _tokenizer_from_hub, "トークナイザを Hugging Face Hub から取得できなかった"
assert tokenizer.vocab_size == 8192

# 終端の特殊トークンの有無: 語彙は 256 個のバイトの記号と、マージで作った記号だけである(特殊トークンの記号を持たない)
_bpe = tokenizer.bpe_tokenizer
HAS_SPECIAL_TOKENS = len(_bpe.vocab) != 256 + len(_bpe.merges) or any(
    s.startswith("<") and s.endswith(">") and len(s) > 2 for s in tokenizer.symbol_to_id
)
assert not HAS_SPECIAL_TOKENS and tokenizer.vocab_size == 256 + len(_bpe.merges)
end_stage("トークナイザの取得", _stage)

# --- 参照コーパス(008 の事前学習と英語のトークナイザの学習に使ったコーパス)---
from huggingface_hub import hf_hub_download  # noqa: E402

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

# --- 使う記事: 参照コーパスの外に始まる記事だけ(先頭の記事は学習用・検証用・評価用のどれにも使わない)---
CANDIDATE_ARTICLES = find_articles_starting_at_or_after(article_spans, REFERENCE_CORPUS_CHARACTERS)
assert CANDIDATE_ARTICLES == list(range(NUM_REFERENCE_ARTICLES, NUM_ARTICLES)), "使う記事が先頭の記事を除く全記事と一致しない"
assert article_spans[NUM_REFERENCE_ARTICLES - 1]["end"] == REFERENCE_CORPUS_CHARACTERS, "先頭の記事の末尾が参照コーパスの末尾と一致しない"
assert article_spans[CANDIDATE_ARTICLES[0]]["start"] == REFERENCE_CORPUS_CHARACTERS + 1 > REFERENCE_PRETRAINING_END

# --- 記事ごとの符号化(時間のスケーリングを 3 点で計測してから、使う記事を符号化する)---
_stage = time.time()
_candidate_start = article_spans[CANDIDATE_ARTICLES[0]]["start"]
_candidate_characters = article_spans[-1]["end"] - _candidate_start
_encode_times, _encode_position = [], _candidate_start
for _n in SCALING_ENCODE_CHARACTERS:  # 区間は重ならない(同じ文章を繰り返し符号化して、語の単位の記憶で速くなる効果を避ける)
    _start = time.time()
    tokenizer.encode(corpus_text[_encode_position : _encode_position + _n])
    _encode_times.append(time.time() - _start)
    _encode_position += _n
_encode_fit = fit_power_law_exponent(SCALING_ENCODE_CHARACTERS, _encode_times)
_encode_extrapolated = _encode_times[-1] * (_candidate_characters / SCALING_ENCODE_CHARACTERS[-1]) ** _encode_fit.exponent
_encode_proportional = _encode_times[-1] * _candidate_characters / SCALING_ENCODE_CHARACTERS[-1]
_start = time.time()
ARTICLE_TOKENS = encode_articles(tokenizer, corpus_text, article_spans, CANDIDATE_ARTICLES)
ENCODE_SECONDS = time.time() - _start
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
end_stage("記事の符号化と検証", _stage)

_titles = list(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))
LIST_LIKE_ARTICLES = frozenset(a for a in CANDIDATE_ARTICLES if _titles[a].startswith(LIST_LIKE_TITLE_PREFIXES))  # タイトルの接頭辞で判定するリスト系の記事
CORPUS_CHARACTERS = len(corpus_text)
del corpus_text, _sample
gc.collect()

# --- 記事を単位とする分割(決定的) ---
_stage = time.time()
MIN_TOKENS_FOR_RETRIEVAL = 2 * PASSAGE_LENGTH  # passage を 2 個以上持つ記事


def make_split():
    return split_articles(
        ARTICLE_TOKEN_COUNTS, MIN_TOKENS_FOR_RETRIEVAL, CANDIDATE_ARTICLES, NUM_VALIDATION_ARTICLES, NUM_EVALUATION_ARTICLES, SPLIT_SEED
    )


SPLIT = make_split()
_parts = [set(SPLIT.train.tolist()), set(SPLIT.validation.tolist()), set(SPLIT.evaluation.tolist())]
assert not (_parts[0] & _parts[1]) and not (_parts[0] & _parts[2]) and not (_parts[1] & _parts[2]), "分割が重なっている"
assert _parts[0] | _parts[1] | _parts[2] == set(CANDIDATE_ARTICLES) and sum(len(p) for p in _parts) == len(CANDIDATE_ARTICLES)
assert len(SPLIT.validation) == NUM_VALIDATION_ARTICLES and len(SPLIT.evaluation) == NUM_EVALUATION_ARTICLES
assert all(ARTICLE_TOKEN_COUNTS[a] >= MIN_TOKENS_FOR_RETRIEVAL for a in list(SPLIT.validation) + list(SPLIT.evaluation))
assert make_split().digest() == SPLIT.digest(), "分割が決定的でない"
# 008 の事前学習とトークナイザの学習の範囲(参照コーパスの先頭から REFERENCE_PRETRAINING_END の手前まで)の外: 使う記事はすべて、参照コーパスの全体の外に始まる
_used_starts = np.array([article_spans[a]["start"] for a in sorted(_parts[0] | _parts[1] | _parts[2])])
assert _used_starts.min() >= REFERENCE_CORPUS_CHARACTERS > REFERENCE_PRETRAINING_END, "008 の事前学習の範囲にかかる記事を使っている"
assert not (set(range(NUM_REFERENCE_ARTICLES)) & (_parts[0] | _parts[1] | _parts[2])), "先頭の記事を使っている"
end_stage("分割", _stage)

# --- 評価用・検証用の索引と query、学習用の部分の索引と query(汎化の差の診断量) ---
_stage = time.time()
def make_sets():
    evaluation = build_retrieval_set(
        ARTICLE_TOKENS, SPLIT.evaluation, PASSAGE_LENGTH, QUERY_LENGTH, MAX_PASSAGES_PER_ARTICLE, MAX_QUERIES_PER_ARTICLE, EVALUATION_SET_SEED
    )
    validation = build_retrieval_set(
        ARTICLE_TOKENS, SPLIT.validation, PASSAGE_LENGTH, QUERY_LENGTH, MAX_PASSAGES_PER_ARTICLE, MAX_QUERIES_PER_ARTICLE, VALIDATION_SET_SEED
    )
    _probe_pool = [a for a in SPLIT.train if ARTICLE_TOKEN_COUNTS[a] >= MIN_TOKENS_FOR_RETRIEVAL]
    _probe_articles = [_probe_pool[i] for i in select_evenly(len(_probe_pool), NUM_TRAIN_PROBE_ARTICLES)]
    probe = build_retrieval_set(
        ARTICLE_TOKENS, _probe_articles, PASSAGE_LENGTH, QUERY_LENGTH, MAX_PASSAGES_PER_ARTICLE, MAX_QUERIES_PER_ARTICLE, TRAIN_PROBE_SEED
    )
    return evaluation, validation, probe


EVALUATION_SET, VALIDATION_SET, TRAIN_PROBE_SET = make_sets()
_again = make_sets()
assert [s.digest() for s in _again] == [s.digest() for s in (EVALUATION_SET, VALIDATION_SET, TRAIN_PROBE_SET)], "索引と query が決定的でない"
del _again
RETRIEVAL_SETS = {"evaluation": EVALUATION_SET, "validation": VALIDATION_SET, "train_probe": TRAIN_PROBE_SET}
for _name, _retrieval_set in RETRIEVAL_SETS.items():
    _offsets = _retrieval_set.query_offsets[:, None] + np.arange(QUERY_LENGTH)[None, :]
    assert np.array_equal(_retrieval_set.queries, np.take_along_axis(_retrieval_set.passages[_retrieval_set.query_sources], _offsets, axis=1)), "query が元の passage の一部でない"
    assert np.array_equal(_retrieval_set.passage_articles[_retrieval_set.query_sources], _retrieval_set.query_articles)
    assert (_retrieval_set.num_positives() >= 1).all() and _retrieval_set.passages.shape[1] == PASSAGE_LENGTH and _retrieval_set.queries.shape[1] == QUERY_LENGTH
    _per_article = np.bincount(_retrieval_set.passage_articles)
    assert _per_article.max() <= MAX_PASSAGES_PER_ARTICLE and (_per_article[_per_article > 0] >= 1).all()
# 3 つの集合の記事は互いに素(学習用の部分の索引は学習用の記事だけ)
_sets_articles = {n: set(s.passage_articles.tolist()) for n, s in RETRIEVAL_SETS.items()}
assert not (_sets_articles["evaluation"] & _sets_articles["validation"]) and not (_sets_articles["evaluation"] & _sets_articles["train_probe"])
assert not (_sets_articles["validation"] & _sets_articles["train_probe"])
assert _sets_articles["train_probe"] <= set(SPLIT.train.tolist()) and _sets_articles["evaluation"] <= set(SPLIT.evaluation.tolist())
assert _sets_articles["validation"] <= set(SPLIT.validation.tolist())
RANDOM_RECALL = {n: expected_random_recall(s.num_positives(), s.num_passages - 1, 10) for n, s in RETRIEVAL_SETS.items()}
end_stage("評価用の索引と query の構築", _stage)

# --- 学習用の記事(組の切り出し) ---
_stage = time.time()
TRAIN_ARTICLES = SPLIT.train
TRAIN_TOKEN_COUNTS = ARTICLE_TOKEN_COUNTS[TRAIN_ARTICLES]
SAMPLING_WEIGHTS = compute_article_sampling_weights(TRAIN_TOKEN_COUNTS, QUERY_LENGTH, PASSAGE_LENGTH, MAX_PASSAGES_PER_ARTICLE)
_flat, TRAIN_STARTS = concatenate_articles(ARTICLE_TOKENS, TRAIN_ARTICLES)
TRAIN_STREAM = torch.from_numpy(_flat).to(device)
del _flat
_probe_articles_kept = set(TRAIN_PROBE_SET.passage_articles.tolist())  # 学習用の部分の索引を作り直せるように、その記事のトークンは残す
for _article in TRAIN_ARTICLES.tolist():  # 学習用の記事のトークンは TRAIN_STREAM に移したので、CPU 側の配列を解放する
    if _article not in _probe_articles_kept:
        ARTICLE_TOKENS[_article] = None
gc.collect()
NUM_PAIRABLE_TRAIN_ARTICLES = int((SAMPLING_WEIGHTS > 0).sum())
assert NUM_PAIRABLE_TRAIN_ARTICLES >= BATCH_SIZE
TRAIN_TOKENS_PAIRABLE = int(TRAIN_TOKEN_COUNTS[SAMPLING_WEIGHTS > 0].sum())
UNIFORM_LOSS = math.log(BATCH_SIZE)  # ln(N): 類似度が一様なときの in-batch negatives の InfoNCE 損失
end_stage("学習用の記事の連結", _stage)
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
    f"使う記事: {len(CANDIDATE_ARTICLES):,} 件({_candidate_characters:,} 文字、{CANDIDATE_TOKENS:,} トークン。記事の長さ: 中央値 {int(np.median(ARTICLE_TOKEN_COUNTS[CANDIDATE_ARTICLES])):,}、"
    f"最長 {int(ARTICLE_TOKEN_COUNTS.max()):,})。リスト系の記事(タイトルが {', '.join(LIST_LIKE_TITLE_PREFIXES)} で始まる): "
    f"{len(LIST_LIKE_ARTICLES):,} 件({len(LIST_LIKE_ARTICLES) / len(CANDIDATE_ARTICLES):.1%})"
)
print(
    f"符号化の時間: {dict(zip(SCALING_ENCODE_CHARACTERS, rounded(_encode_times, 3), strict=True))} 秒、べき指数 b = {_encode_fit.exponent:.3f}、"
    f"使う記事の全体({_candidate_characters:,} 文字)への外挿: べき乗則 {_encode_extrapolated:.1f} 秒・比例 {_encode_proportional:.1f} 秒、実測 {ENCODE_SECONDS:.1f} 秒"
)
print(
    f"トークナイザ: 語彙サイズ {tokenizer.vocab_size} = 256 個のバイト + {len(_bpe.merges)} 個のマージ。特殊トークン(終端トークンを含む)の有無: "
    f"{'あり' if HAS_SPECIAL_TOKENS else 'なし'}。終端位置のプーリングは入力の最後の位置の出力を使う"
)
print(
    f"008 の事前学習とトークナイザの学習の範囲: 参照コーパスの文字位置 {REFERENCE_PRETRAINING_END:,} の手前まで(参照コーパスの全体は {REFERENCE_CORPUS_CHARACTERS:,} 文字)。"
    f"使う記事の開始位置の最小は {int(_used_starts.min()):,} で、範囲の外: OK"
)
print(
    f"分割(ダイジェスト {SPLIT.digest()}): 学習用 {len(SPLIT.train):,} 記事({int(TRAIN_TOKEN_COUNTS.sum()):,} トークン、うち組を作れる記事 {NUM_PAIRABLE_TRAIN_ARTICLES:,} 件・"
    f"{TRAIN_TOKENS_PAIRABLE:,} トークン)、検証用 {len(SPLIT.validation)} 記事({int(ARTICLE_TOKEN_COUNTS[SPLIT.validation].sum()):,} トークン)、"
    f"評価用 {len(SPLIT.evaluation)} 記事({int(ARTICLE_TOKEN_COUNTS[SPLIT.evaluation].sum()):,} トークン)。リスト系の記事の割合: "
    f"学習用 {len(LIST_LIKE_ARTICLES & set(SPLIT.train.tolist())) / len(SPLIT.train):.1%}、検証用 {len(LIST_LIKE_ARTICLES & set(SPLIT.validation.tolist())) / len(SPLIT.validation):.1%}、"
    f"評価用 {len(LIST_LIKE_ARTICLES & set(SPLIT.evaluation.tolist())) / len(SPLIT.evaluation):.1%}"
)
for _name, _retrieval_set in RETRIEVAL_SETS.items():
    print(
        f"  {_name}: 索引 {_retrieval_set.num_passages} passage・記事 {len(set(_retrieval_set.passage_articles.tolist()))} 本、query {_retrieval_set.num_queries} 個、正解の数の中央値 "
        f"{int(np.median(_retrieval_set.num_positives()))}、ランダムな順位の Recall@10 の期待値 {RANDOM_RECALL[_name]:.4f}、ダイジェスト {_retrieval_set.digest()}"
    )
print(
    f"学習: 1 ステップ {BATCH_SIZE} 組 x(query {QUERY_LENGTH} + passage {PASSAGE_LENGTH})= {BATCH_SIZE * (QUERY_LENGTH + PASSAGE_LENGTH):,} トークン。"
    f"組を作れる学習用の記事のトークン数に対するエポック数: " + "、".join(
        f"T = {_t}: {_t * BATCH_SIZE * (QUERY_LENGTH + PASSAGE_LENGTH) / TRAIN_TOKENS_PAIRABLE:.3f}" for _t in STEP_CANDIDATES
    ) + f"。ln(N) = {UNIFORM_LOSS:.4f}"
)
print(
    "データの準備の時間(段階ごと): " + "、".join(f"{k} {v:.1f} 秒" for k, v in DATA_SECONDS_BY_STAGE.items()) + f"。合計 {DATA_SECONDS:.1f} 秒"
)
print(
    f"データの準備のピークのメモリ: {DATA_PEAK_MEMORY_GIB:.2f} GiB(段階ごとの累積のピーク: "
    + "、".join(f"{k} {v:.2f}" for k, v in PEAK_MEMORY_GIB_BY_STAGE.items())
    + f")。このマシンの RAM {_total_memory:.1f} GiB、Colab 無料枠 {COLAB_RAM_GIB} GiB の {RAM_FRACTION_LIMIT:.0%} = {_memory_limit:.1f} GiB 以下: OK"
    if _total_memory
    else f"データの準備のピークのメモリ: {DATA_PEAK_MEMORY_GIB:.2f} GiB(Colab 無料枠 {COLAB_RAM_GIB} GiB の {RAM_FRACTION_LIMIT:.0%} 以下: OK)"
)
```


    tokenizer.json:   0%|          | 0.00/661k [00:00<?, ?B/s]



    corpus.txt: reconstructing file:   0%|          |  0.00B / 24.3MB            



    corpus.txt: downloading bytes:           |  0.00B            



    corpus.txt: reconstructing file:   0%|          |  0.00B /  546MB            



    corpus.txt: downloading bytes:           |  0.00B            



    metadata.json:   0%|          | 0.00/1.22M [00:00<?, ?B/s]


    コーパス: 542,892,375 文字(取得元 Hugging Face Hub、546 MB)、記事 9,826 件(境界の取得元 metadata)。先頭の 356 件は参照コーパス(008 の事前学習とトークナイザの学習に使ったコーパス、24,214,546 文字)と同一で、使わない
    使う記事: 9,470 件(518,677,828 文字、147,610,040 トークン。記事の長さ: 中央値 14,267、最長 125,195)。リスト系の記事(タイトルが List of, Lists of, Timeline of, Outline of, Index of, Glossary of, Bibliography of, Comparison of, Discography, Filmography, Deaths in, Births in で始まる): 2,864 件(30.2%)
    符号化の時間: {8000000: 5.339, 16000000: 9.876, 32000000: 21.949} 秒、べき指数 b = 1.020、使う記事の全体(518,677,828 文字)への外挿: べき乗則 375.9 秒・比例 355.8 秒、実測 350.5 秒
    トークナイザ: 語彙サイズ 8192 = 256 個のバイト + 7936 個のマージ。特殊トークン(終端トークンを含む)の有無: なし。終端位置のプーリングは入力の最後の位置の出力を使う
    008 の事前学習とトークナイザの学習の範囲: 参照コーパスの文字位置 23,003,819 の手前まで(参照コーパスの全体は 24,214,546 文字)。使う記事の開始位置の最小は 24,214,547 で、範囲の外: OK
    分割(ダイジェスト 716cfe33d05a2615): 学習用 9,070 記事(141,169,592 トークン、うち組を作れる記事 8,639 件・141,126,568 トークン)、検証用 100 記事(1,434,284 トークン)、評価用 300 記事(5,006,164 トークン)。リスト系の記事の割合: 学習用 30.3%、検証用 32.0%、評価用 27.0%
      evaluation: 索引 3244 passage・記事 300 本、query 3244 個、正解の数の中央値 11、ランダムな順位の Recall@10 の期待値 0.0321、ダイジェスト da3e16b96714c0a4
      validation: 索引 1019 passage・記事 100 本、query 1019 個、正解の数の中央値 11、ランダムな順位の Recall@10 の期待値 0.0962、ダイジェスト acc9ec62c2a091b0
      train_probe: 索引 1283 passage・記事 120 本、query 1283 個、正解の数の中央値 11、ランダムな順位の Recall@10 の期待値 0.0796、ダイジェスト 88644431fad40387
    学習: 1 ステップ 64 組 x(query 32 + passage 128)= 10,240 トークン。組を作れる学習用の記事のトークン数に対するエポック数: T = 1024: 0.074、T = 512: 0.037、T = 256: 0.019。ln(N) = 4.1589
    データの準備の時間(段階ごと): トークナイザの取得 6.2 秒、参照コーパスの取得 2.2 秒、コーパスのダウンロード 5.6 秒、コーパスの読み込み 7.5 秒、記事の境界の検証 0.0 秒、記事の符号化と検証 389.4 秒、分割 0.1 秒、評価用の索引と query の構築 0.5 秒、学習用の記事の連結 1.0 秒。合計 412.8 秒
    データの準備のピークのメモリ: 5.48 GiB(段階ごとの累積のピーク: トークナイザの取得 0.80、参照コーパスの取得 1.01、コーパスのダウンロード 1.20、コーパスの読み込み 5.48、記事の境界の検証 5.48、記事の符号化と検証 5.48、分割 5.48、評価用の索引と query の構築 5.48、学習用の記事の連結 5.48)。このマシンの RAM 12.7 GiB、Colab 無料枠 12.7 GiB の 75% = 9.5 GiB 以下: OK




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/024_text_embedding_and_retriever.ipynb
