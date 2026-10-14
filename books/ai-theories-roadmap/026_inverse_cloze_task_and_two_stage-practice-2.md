---
title: "Inverse Cloze Task と passage 単位の 2 段階検索 / Inverse Cloze Task and Two-Stage Retrieval at Passage Level(実装・実験編 2/8)"
---

この記事は後編(実装・実験編 2/8)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/026_inverse_cloze_task_and_two_stage-practice-1)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/026_inverse_cloze_task_and_two_stage-practice-3)。

### 5.3 データ: コーパス・記事の分割・Inverse Cloze Task の索引

**コーパスとトークナイザ**: 英語 Wikipedia のコーパス(`kojikojiprg/ai-theories-corpus-en`、9,826 記事)と、008 の BPE(Byte Pair Encoding)トークナイザ(語彙サイズ 8192)を使う。
記事の境界は`metadata.json`の`article_offsets`から得る。コーパスの先頭の 356 記事は 008 の事前学習とトークナイザの学習に使ったコーパスと同一なので使わず、使うのは、そのどちらも一度も見ていない 9,470 記事である。
記事の分割([024](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/024_text_embedding_and_retriever-theory) と同じ。学習用 9,070 記事・検証用 100 記事・評価用 300 記事)と、評価用・検証用の記事単位の索引・query は、024 の本番の値(ダイジェスト)と一致することを確かめる。

**コーパスの passage の索引**: 使う記事すべてを、重ならない長さ $L_p = 128$ の窓に分けて passage にする(記事ごとに最大 12 個を等間隔に選ぶ。024・025 と同じ手順)。
評価用の記事の部分は、024 の評価用の索引と一致する。**評価の索引**(3.5 節)は、このコーパスの passage から、抜く文を決定的に 1 つ選んで「文を除いた形」にしたもので、
対象でない passage(文が 2 つ未満、または長さ $4 \le \ell \le 96$ の文がない passage)は索引に入れない。除いた passage の数を印字する。索引の passage の数が $P$ である。
抜く文は、長さ 32 トークン以内に収まる文を優先して選ぶ(収まる文がない passage だけ長い文を選んで query の先頭を切り詰める)。切り詰めた query の割合を、評価の索引と学習の組のそれぞれで印字する。

| 部分 | 記事 | 評価での役割 |
|---|---|---|
| 学習用(9,070 記事) | 索引に入る(検索の対象の一部)。cross-encoder の学習の query もここから作る | 検索の対象のコーパスを大きくする(困難な負例の候補になる) |
| 検証用(100 記事) | 索引に入る。較正は検証用の記事の passage だけを索引にした小さい索引で行う | 学習率の較正と前提条件に使う。判定には使わない |
| 評価用(300 記事) | 索引に入る。query はこの部分の passage から作る | 判定と診断量に使う |

**同じ索引・同じ query、2 つのラベル**: 実験 A・B と較正は、同じ索引・同じ query で、passage 単位のラベルと記事単位のラベルを当てる。query は、記事単位のラベルの正解(同じ記事の他の passage)が索引に 1 つ以上ある query に限る(除いた数を印字する)。
024 と同じ形式の記事単位の評価(024 の評価用の索引と query。固定長)は、診断量として別に測る。

**文の分割**: 3.4 節の規則を、トークンごとの表で実装する(トークンのテキストは、バイトレベルの BPE のシンボルを UTF-8 にデコードして得る。デコードできないトークンは、空白でも句点でもないものとして扱う)。

**第 1 段の学習の組**(5.5 節): 1 つのバッチは、32 記事 × 各記事 2 組の 64 組である。各記事から重ならない 2 つの窓を引き、それぞれから抜く文を選び、文脈を作る(3.2 節)。
R2 は各窓の文脈をその窓の query の正例にし、R1 は同じ記事のもう一方の窓の文脈を正例にする(バッチの中で隣り合う 2 組の文脈を入れ替える)。記事の並び・窓・抜く文・文を残す組は R1 と R2 で同一である(6.12 節のアサーションで確かめる)。
記事は、窓を重ならずに 2 つ取れる長さ($2 L_p = 256$ トークン以上)の記事から、窓の数(最大 12 で頭打ち)に比例する確率で非復元抽出する。

**cross-encoder の学習の query**: 索引の passage のうち学習用の記事のもの(索引の形の文を除いた passage と、抜いた文)から作る。正例は、その query の passage である。


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

# --- トークナイザ: 区切りと終端に使う単一バイトのトークン(語彙を増やさない)と、文の分割の表 ---
_stage = time.time()
tokenizer, _tokenizer_from_hub = load_bpe_id_tokenizer_from_hub(TOKENIZER_REPO_ID)
assert _tokenizer_from_hub, "トークナイザを Hugging Face Hub から取得できなかった"
assert tokenizer.vocab_size == 8192
_bpe = tokenizer.bpe_tokenizer
assert tokenizer.vocab_size == 256 + len(_bpe.merges) and len(_bpe.vocab) == tokenizer.vocab_size  # 特殊トークンがない
_separator_ids, _terminal_ids = tokenizer.encode(chr(SEPARATOR_BYTE)), tokenizer.encode(chr(TERMINAL_BYTE))
assert len(_separator_ids) == len(_terminal_ids) == 1 and _separator_ids != _terminal_ids
SEPARATOR_TOKEN_ID, TERMINAL_TOKEN_ID = int(_separator_ids[0]), int(_terminal_ids[0])
TOKEN_TEXTS = [try_decode_byte_level_symbol(tokenizer.id_to_symbol[i]) for i in range(tokenizer.vocab_size)]
SENTENCE_TABLES = build_sentence_tables(TOKEN_TEXTS)
NUM_UNDECODABLE_TOKENS = sum(t is None for t in TOKEN_TEXTS)
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

# --- 記事単位の評価用・検証用の索引と query(024 と同じ。ダイジェストが 024 の本番の値と一致する)---
_stage = time.time()
EVALUATION_SET = build_retrieval_set(
    ARTICLE_TOKENS, SPLIT.evaluation, PASSAGE_LENGTH, QUERY_LENGTH, MAX_PASSAGES_PER_ARTICLE, MAX_QUERIES_PER_ARTICLE, EVALUATION_SET_SEED
)
VALIDATION_SET = build_retrieval_set(
    ARTICLE_TOKENS, SPLIT.validation, PASSAGE_LENGTH, QUERY_LENGTH, MAX_PASSAGES_PER_ARTICLE, MAX_QUERIES_PER_ARTICLE, VALIDATION_SET_SEED
)
assert EVALUATION_SET.digest() == EVALUATION_DIGEST_024, f"評価用の索引と query が 024 の本番の集合と一致しない: {EVALUATION_SET.digest()}"
assert VALIDATION_SET.digest() == VALIDATION_DIGEST_024, f"検証用の索引と query が 024 の本番の集合と一致しない: {VALIDATION_SET.digest()}"
ARTICLE_LEVEL_SETS = {"evaluation": EVALUATION_SET, "validation": VALIDATION_SET}
# --- コーパスの passage(使う記事すべて。動作確認では、学習用の記事を一部に絞る)---
_pairable_train = [a for a in SPLIT.train.tolist() if ARTICLE_TOKEN_COUNTS[a] >= MIN_TOKENS_FOR_RETRIEVAL]
CORPUS_ARTICLES = (
    list(CANDIDATE_ARTICLES)
    if CORPUS_TRAIN_ARTICLES is None
    else sorted(set(SPLIT.evaluation.tolist()) | set(SPLIT.validation.tolist()) | {_pairable_train[i] for i in select_evenly(len(_pairable_train), CORPUS_TRAIN_ARTICLES)})
)
CORPUS = build_retrieval_set(
    ARTICLE_TOKENS, CORPUS_ARTICLES, PASSAGE_LENGTH, QUERY_LENGTH, MAX_PASSAGES_PER_ARTICLE, MAX_QUERIES_PER_ARTICLE, EVALUATION_SET_SEED
)
_corpus_evaluation = np.nonzero(np.isin(CORPUS.passage_articles, SPLIT.evaluation))[0]
assert np.array_equal(CORPUS.passages[_corpus_evaluation], EVALUATION_SET.passages), "コーパスの評価用の記事の部分が 024 の評価用の索引と一致しない"
NUM_CORPUS_PASSAGES = CORPUS.num_passages
# --- passage 単位の評価の索引(3.5 節): 抜く文を決定的に選び、文を除いた形にする ---
CLOZE = build_cloze_index(
    CORPUS.passages, CORPUS.passage_articles, CORPUS.passage_windows, SENTENCE_TABLES, QUERY_LENGTH, MIN_QUERY_TOKENS, MAX_REMOVED_TOKENS, CLOZE_SEED
)
NUM_INDEX = CLOZE.num_passages  # P
ENTRY_ARTICLES = CORPUS.passage_articles[CLOZE.passage_ids]  # 索引の各 passage の記事
_entries_per_article = np.bincount(ENTRY_ARTICLES)
NUM_MATES = _entries_per_article[ENTRY_ARTICLES] - 1  # 索引の中の、同じ記事の他の passage の数(記事単位のラベルの正解の数)
HAS_MATE = NUM_MATES >= 1
EVALUATION_INDEX_ENTRIES = np.nonzero(np.isin(ENTRY_ARTICLES, SPLIT.evaluation))[0]
VALIDATION_INDEX_ENTRIES = np.nonzero(np.isin(ENTRY_ARTICLES, SPLIT.validation))[0]  # 検証用の記事の passage(較正の索引)
TRAIN_ENTRIES = np.nonzero(np.isin(ENTRY_ARTICLES, SPLIT.train))[0]
assert len(EVALUATION_INDEX_ENTRIES) + len(VALIDATION_INDEX_ENTRIES) + len(TRAIN_ENTRIES) == NUM_INDEX
# 実験 A・B の評価の query: 2 つのラベルで同じ query を使うため、記事単位のラベルの正解(同じ記事の他の passage)が 1 つ以上ある query に限る
EVALUATION_ENTRIES = EVALUATION_INDEX_ENTRIES[HAS_MATE[EVALUATION_INDEX_ENTRIES]]
VALIDATION_ENTRIES = VALIDATION_INDEX_ENTRIES[HAS_MATE[VALIDATION_INDEX_ENTRIES]]
CONTEXT_TOKENS = torch.from_numpy(CLOZE.context_tokens)  # 索引に入れる形(右側を 0 でパディング)
CONTEXT_LENGTHS = CLOZE.context_lengths
CLOZE_QUERY_TOKENS = torch.from_numpy(CLOZE.query_tokens)
CLOZE_QUERY_LENGTHS = CLOZE.query_lengths
EVALUATION_QUERY_ARTICLES = ENTRY_ARTICLES[EVALUATION_ENTRIES]  # ブートストラップのクラスタ(実験 A・C)
RANDOM_RECALL_PASSAGE = 10 / NUM_INDEX  # passage 単位のラベルの、ランダムな順位の Recall@10 の期待値(正例は 1 つ)
RANDOM_RECALL_ARTICLE_LABEL = expected_random_recall(NUM_MATES[EVALUATION_ENTRIES], NUM_INDEX - 1, 10)  # 記事単位のラベル(正解は同じ記事の他の passage)
end_stage("コーパスと索引の構成", _stage)

# --- 学習用の記事(第 1 段の組の切り出し): 連結した 1 本のトークン列(int32)---
_stage = time.time()
TRAIN_ARTICLES = SPLIT.train
TRAIN_TOKEN_COUNTS = ARTICLE_TOKEN_COUNTS[TRAIN_ARTICLES]
SAMPLING_WEIGHTS = compute_article_sampling_weights(TRAIN_TOKEN_COUNTS, QUERY_LENGTH, PASSAGE_LENGTH, MAX_PASSAGES_PER_ARTICLE)
SAMPLING_WEIGHTS[TRAIN_TOKEN_COUNTS < PAIRS_PER_ARTICLE * PASSAGE_LENGTH] = 0.0  # 窓を重ならずに 2 つ取れない短い記事は選ばない
_flat, TRAIN_STARTS = concatenate_articles(ARTICLE_TOKENS, TRAIN_ARTICLES)
TRAIN_STREAM = _flat.astype(np.int32)
del _flat
for _article in range(NUM_ARTICLES):  # 記事ごとのトークンは、passage・連結した列に取り出したので解放する
    ARTICLE_TOKENS[_article] = None
gc.collect()
# 重ならない対象の窓の組(2 個)が実際に取れるかを、記事ごとに厳密に判定する。取れない記事(対象の窓が互いに近い位置にしかない記事など)の重みは 0 にする
_stage = time.time()
_length_ok = SAMPLING_WEIGHTS > 0
_pairable = pairable_articles(
    TRAIN_STREAM, np.asarray(TRAIN_STARTS), TRAIN_TOKEN_COUNTS, _length_ok, PASSAGE_LENGTH, SENTENCE_TABLES, MIN_QUERY_TOKENS, MAX_REMOVED_TOKENS
)
UNPAIRABLE_TRAIN_ARTICLES = np.nonzero(_length_ok & ~_pairable)[0]  # 学習用の記事の中での添字
SAMPLING_WEIGHTS[UNPAIRABLE_TRAIN_ARTICLES] = 0.0
end_stage("組が取れる記事の判定", _stage)
NUM_PAIRABLE_TRAIN_ARTICLES = int((SAMPLING_WEIGHTS > 0).sum())
assert NUM_PAIRABLE_TRAIN_ARTICLES >= ARTICLES_PER_BATCH
print(
    f"学習用の記事 {len(TRAIN_ARTICLES):,} 件のうち、トークン数が {PAIRS_PER_ARTICLE * PASSAGE_LENGTH} 以上で重みが正の記事 {int(_length_ok.sum()):,} 件から、"
    f"重ならない対象の窓の組が取れない記事 {len(UNPAIRABLE_TRAIN_ARTICLES)} 件(学習用の記事の中の添字 {UNPAIRABLE_TRAIN_ARTICLES.tolist()[:10]}、トークン数 {TRAIN_TOKEN_COUNTS[UNPAIRABLE_TRAIN_ARTICLES].tolist()[:10]})の重みを 0 にした。組を作れる記事 {NUM_PAIRABLE_TRAIN_ARTICLES:,} 件"
)
TRAIN_TOKENS_PAIRABLE = int(TRAIN_TOKEN_COUNTS[SAMPLING_WEIGHTS > 0].sum())
UNIFORM_LOSS = math.log(BATCH_SIZE)  # ln(N): 類似度が一様なときの in-batch negatives の InfoNCE 損失
end_stage("学習用の記事の連結", _stage)
DATA_SECONDS = time.time() - _t0_data
DATA_PEAK_MEMORY_GIB = peak_memory_gib()
_total_memory = total_memory_gib()
_memory_limit = RAM_FRACTION_LIMIT * min(COLAB_RAM_GIB, _total_memory if _total_memory else COLAB_RAM_GIB)
assert DATA_PEAK_MEMORY_GIB <= _memory_limit, f"データの準備のピークのメモリ {DATA_PEAK_MEMORY_GIB:.1f} GiB が Colab の RAM に収まる見込みを超える({_memory_limit:.1f} GiB)"

print(
    f"コーパス: {CORPUS_CHARACTERS:,} 文字(取得元 Hugging Face Hub)、記事 {NUM_ARTICLES:,} 件。使う記事 {len(CANDIDATE_ARTICLES):,} 件({CANDIDATE_TOKENS:,} トークン)、"
    f"リスト系の記事の割合 {len(LIST_LIKE_ARTICLES) / len(CANDIDATE_ARTICLES):.1%}。符号化のキャッシュ: {'読み込み' if ENCODED_FROM_CACHE else '新規に作成'}"
)
print(
    f"トークナイザ: 語彙サイズ {tokenizer.vocab_size}(特殊トークンなし)、UTF-8 として不正なトークン {NUM_UNDECODABLE_TOKENS} 個。文の分割の表: 空白で始まる {int(SENTENCE_TABLES.starts_with_whitespace.sum())}、"
    f"改行で始まる {int(SENTENCE_TABLES.starts_with_newline.sum())}、文の終わりで終わる {int(SENTENCE_TABLES.ends_sentence.sum())}、閉じ記号だけ {int(SENTENCE_TABLES.closer_only.sum())} トークン。"
    f"区切り: 単一バイト 0x1F(ID {SEPARATOR_TOKEN_ID})、終端: 0x1E(ID {TERMINAL_TOKEN_ID})。コーパスでの出現回数は 区切り {SEPARATOR_OCCURRENCES}・終端 {TERMINAL_OCCURRENCES}"
)
print(
    f"分割(ダイジェスト {SPLIT.digest()}、024 の本番の値と一致): 学習用 {len(SPLIT.train):,} 記事(うち組を作れる記事 {NUM_PAIRABLE_TRAIN_ARTICLES:,} 件・{TRAIN_TOKENS_PAIRABLE:,} トークン)、"
    f"検証用 {len(SPLIT.validation)} 記事、評価用 {len(SPLIT.evaluation)} 記事。使う記事の開始位置の最小 {int(_used_starts.min()):,} は参照コーパスの全体({REFERENCE_CORPUS_CHARACTERS:,} 文字)の外"
)
for _name, _set in ARTICLE_LEVEL_SETS.items():
    print(f"  記事単位の{_name}: 索引 {_set.num_passages} passage・query {_set.num_queries} 個、ダイジェスト {_set.digest()}(024 の本番の値と一致)")
print(
    f"コーパス: passage {NUM_CORPUS_PASSAGES:,} 個(記事 {len(set(CORPUS.passage_articles.tolist())):,} 本、動作確認では学習用の記事を絞る: {CORPUS_TRAIN_ARTICLES})。"
    f"passage 単位の評価の索引(Inverse Cloze Task の対象): {NUM_INDEX:,} 個。除いた passage {CLOZE.num_excluded:,} 個"
    f"(うち文が 2 つ未満の passage {CLOZE.num_single_sentence:,} 個。残りは長さ {MIN_QUERY_TOKENS}〜{MAX_REMOVED_TOKENS} トークンの文がない passage)"
)
print(
    f"索引の内訳: 学習用 {len(TRAIN_ENTRIES):,}・検証用 {len(VALIDATION_INDEX_ENTRIES):,}・評価用 {len(EVALUATION_INDEX_ENTRIES):,}。評価・較正の query は、記事単位のラベルの正解が 1 つ以上ある query に限る: "
    f"評価用 {len(EVALUATION_ENTRIES):,} 個(正解の数の中央値 {int(np.median(NUM_MATES[EVALUATION_ENTRIES]))}、除いた {len(EVALUATION_INDEX_ENTRIES) - len(EVALUATION_ENTRIES)} 個)、検証用 {len(VALIDATION_ENTRIES):,} 個。"
    f"ランダムな順位の Recall@10 の期待値: passage 単位のラベル {RANDOM_RECALL_PASSAGE:.6f}、記事単位のラベル {RANDOM_RECALL_ARTICLE_LABEL:.5f}。文脈の長さ: 平均 {CONTEXT_LENGTHS.mean():.1f}、10・50・90 パーセンタイル {np.percentile(CONTEXT_LENGTHS, [10, 50, 90]).tolist()}。"
    f"query の長さ(抜いた文の先頭 {QUERY_LENGTH} トークンまで): 平均 {CLOZE_QUERY_LENGTHS.mean():.1f}、50 パーセンタイル {float(np.median(CLOZE_QUERY_LENGTHS)):.0f}。"
    f"評価の索引で、抜いた文が {QUERY_LENGTH} トークンを超えて切られた query の割合 {float((CLOZE.removed_length > QUERY_LENGTH).mean()):.1%}"
)
print(
    f"学習: 1 ステップ {ARTICLES_PER_BATCH} 記事 x {PAIRS_PER_ARTICLE} 組 = {BATCH_SIZE} 組(窓 {BATCH_SIZE} 個 = {BATCH_SIZE * PASSAGE_LENGTH:,} トークン)。組を作れる学習用の記事のトークン数に対する、窓が見るトークン数のエポック数: "
    f"T_1 = {FIRST_STAGE_STEPS}: {FIRST_STAGE_STEPS * BATCH_SIZE * PASSAGE_LENGTH / TRAIN_TOKENS_PAIRABLE:.3f}。ln(N) = {UNIFORM_LOSS:.4f}(同じ記事の組を負例から除く R1 では ln(N - 1) = {math.log(BATCH_SIZE - 1):.4f})"
)
print("データの準備の時間(段階ごと): " + "、".join(f"{k} {v:.1f} 秒" for k, v in DATA_SECONDS_BY_STAGE.items()) + f"。合計 {DATA_SECONDS:.1f} 秒")
print(
    f"データの準備のピークのメモリ: {DATA_PEAK_MEMORY_GIB:.2f} GiB(Colab 無料枠 {COLAB_RAM_GIB} GiB の {RAM_FRACTION_LIMIT:.0%} = {_memory_limit:.1f} GiB 以下: OK)"
)
```


    tokenizer.json:   0%|          | 0.00/661k [00:00<?, ?B/s]



    corpus.txt: reconstructing file:   0%|          |  0.00B / 24.3MB            



    corpus.txt: downloading bytes:           |  0.00B            



    corpus.txt: reconstructing file:   0%|          |  0.00B /  546MB            



    corpus.txt: downloading bytes:           |  0.00B            



    metadata.json:   0%|          | 0.00/1.22M [00:00<?, ?B/s]


    学習用の記事 9,070 件のうち、トークン数が 256 以上で重みが正の記事 8,338 件から、重ならない対象の窓の組が取れない記事 1 件(学習用の記事の中の添字 [8646]、トークン数 [277])の重みを 0 にした。組を作れる記事 8,337 件
    コーパス: 542,892,375 文字(取得元 Hugging Face Hub)、記事 9,826 件。使う記事 9,470 件(147,610,040 トークン)、リスト系の記事の割合 30.2%。符号化のキャッシュ: 新規に作成
    トークナイザ: 語彙サイズ 8192(特殊トークンなし)、UTF-8 として不正なトークン 137 個。文の分割の表: 空白で始まる 4952、改行で始まる 352、文の終わりで終わる 223、閉じ記号だけ 7 トークン。区切り: 単一バイト 0x1F(ID 3551)、終端: 0x1E(ID 3550)。コーパスでの出現回数は 区切り 0・終端 0
    分割(ダイジェスト 716cfe33d05a2615、024 の本番の値と一致): 学習用 9,070 記事(うち組を作れる記事 8,337 件・141,064,514 トークン)、検証用 100 記事、評価用 300 記事。使う記事の開始位置の最小 24,214,547 は参照コーパスの全体(24,214,546 文字)の外
      記事単位のevaluation: 索引 3244 passage・query 3244 個、ダイジェスト da3e16b96714c0a4(024 の本番の値と一致)
      記事単位のvalidation: 索引 1019 passage・query 1019 個、ダイジェスト acc9ec62c2a091b0(024 の本番の値と一致)
    コーパス: passage 94,486 個(記事 9,158 本、動作確認では学習用の記事を絞る: None)。passage 単位の評価の索引(Inverse Cloze Task の対象): 93,995 個。除いた passage 491 個(うち文が 2 つ未満の passage 374 個。残りは長さ 4〜96 トークンの文がない passage)
    索引の内訳: 学習用 89,752・検証用 1,016・評価用 3,227。評価・較正の query は、記事単位のラベルの正解が 1 つ以上ある query に限る: 評価用 3,227 個(正解の数の中央値 11、除いた 0 個)、検証用 1,016 個。ランダムな順位の Recall@10 の期待値: passage 単位のラベル 0.000106、記事単位のラベル 0.00112。文脈の長さ: 平均 109.7、10・50・90 パーセンタイル [98.0, 111.0, 123.0]。query の長さ(抜いた文の先頭 32 トークンまで): 平均 17.4、50 パーセンタイル 17。評価の索引で、抜いた文が 32 トークンを超えて切られた query の割合 4.5%
    学習: 1 ステップ 32 記事 x 2 組 = 64 組(窓 64 個 = 8,192 トークン)。組を作れる学習用の記事のトークン数に対する、窓が見るトークン数のエポック数: T_1 = 1024: 0.059。ln(N) = 4.1589(同じ記事の組を負例から除く R1 では ln(N - 1) = 4.1431)
    データの準備の時間(段階ごと): トークナイザの取得 2.1 秒、参照コーパスの取得 3.1 秒、コーパスのダウンロード 7.7 秒、コーパスの読み込み 8.2 秒、記事の境界の検証 0.0 秒、記事の符号化と検証 332.0 秒、分割 0.1 秒、コーパスと索引の構成 4.9 秒、組が取れる記事の判定 19.2 秒、学習用の記事の連結 19.2 秒。合計 379.1 秒
    データの準備のピークのメモリ: 5.48 GiB(Colab 無料枠 12.7 GiB の 75% = 9.5 GiB 以下: OK)




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/026_inverse_cloze_task_and_two_stage_retrieval.ipynb
