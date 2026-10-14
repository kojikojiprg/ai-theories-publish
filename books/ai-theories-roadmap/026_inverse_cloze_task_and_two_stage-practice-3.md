---
title: "Inverse Cloze Task と passage 単位の 2 段階検索 / Inverse Cloze Task and Two-Stage Retrieval at Passage Level(実装・実験編 3/8)"
---

この記事は後編(実装・実験編 3/8)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/026_inverse_cloze_task_and_two_stage-practice-2)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/026_inverse_cloze_task_and_two_stage-practice-4)。

### 5.4 モデルの構築・単体テスト・不変条件の確認

**モデルの構築**:

1. 008 と同じ構成の小型 GPT(4 層、$d_{\mathrm{model}} = 256$、ヘッド数 8、RoPE、RMSNorm、SwiGLU、正規化前置、埋め込みと出力層の重み共有、系列長 256、語彙サイズ 8192。構成は Hub の`config.json`から読む)を作り、
   008 の学習済みの重み(`kojikojiprg/ai-theories-small-gpt-en`の`main`)を読み込む。R1・R2・cross-encoder のすべてがこの重みから始まる。
2. R1・R2 は、平均プール・因果マスクの`TextEmbeddingModel`で包む(024 の条件 C5 と同じ)。cross-encoder は、`VariableLengthCrossEncoder`(区切り・終端のトークンと、標準偏差 0.02 で初期化したスカラーのヘッド)で包む。
   ヘッドの初期化はシード(`torch.manual_seed(26200 + s)`)で決まる。R1・R2 の初期値は、シードによらず 008 の重みで同一である。

**単体テスト**(CPU、FP32。許容の誤差は、実際に計算される型の丸めの単位 × 値の大きさから導き、固定の絶対値は使わない。比べる値がその型で計算されていることもアサーションで確かめる):

- 文の分割: 手で作った文章で、文の区切りが規則どおり(改行、`.`・`!`・`?`、閉じ引用符・閉じ括弧、略語での既知の誤り、小数点で分割しないこと)。窓の端は文の端。
- 抜く文の選択: 抜く文が範囲の長さで文の先頭から始まる、2 つの対象の文があるとき一様に選ぶ、文を抜いた文脈に抜いた文を戻すと元の窓に一致する、パディングが 0。
- passage 単位の評価の索引: 抜く文の選択が passage の並びに依らず決定的、すべての索引の passage が「1 文を除いた形」(文脈の長さ + 抜いた文の長さ = 窓の長さ)、抜いた文を戻すと元の passage に一致する。
- 学習の組(Inverse Cloze Task): シードで決定的、32 記事 × 2 組で同じ記事の 2 組が隣り合い窓が重ならない、1 つのバッチに同じ記事が 1 回だけ、文を文脈に残す割合が約 0.1、長さの範囲、パディング、ストリームの窓と文脈・抜いた文の整合、query の長さ以内に収まる文の優先(切り詰める割合が下がること)。
- 可変長のモデル: 右側のパディングが出力に影響しない(パディングなしの出力と一致)、満長のとき`TextEmbeddingModel.pooled()`と bit 単位で一致、バッチの大きさに依らない、cross-encoder の列の組み立てと終端の位置、
  パディングの内容がスコアに影響しない、パディングなしの 025 の`CrossEncoder`と同じスコア。
- 指標: 単一の正例の順位(passage 単位のラベル)と、最もスコアの高い正解の順位(記事単位のラベル。query の passage 自身は候補から除く)が素朴な実装と一致、**同点は正例・正解に不利**(全候補に同じスコアを返すモデルの Recall@10 が 0 になる)。
- マスクつきの in-batch negatives: 同じ記事の組の passage を負例から除いた損失が閉形式の値と一致、除かないとき通常の InfoNCE と一致。
- 学習ループ: 満長の組で 024 の`train_text_embedding()`と同じ損失の系列になる、学習率 0 の学習(更新がない)の訓練損失が同じ組での学習前のモデルの損失と一致し、モデルが変化しない(第 1 段の R1・R2 の組と cross-encoder)。R1 の組では各行で同じ記事のもう一方の組が負例から除かれ、負例の数が R2 より 1 つ少ない。
- cross-encoder の学習の組: 正例が先頭で上位の候補の中にある、負例が正例を含まず重複しない、決定的。

**既存モジュールの後方互換性**: 本トピックの`src/`は新しいファイルの追加だけである。変更前のコミット(`8ca43d1`)と比べて、`src/`・`scripts/`の既存のファイルに変更も削除もないことを`git diff`で確かめる。


```python
_t0_checks = time.time()
from huggingface_hub import hf_hub_download  # noqa: E402, F811

EPSILON_FP16, EPSILON_FP32, EPSILON_FP64 = (float(torch.finfo(t).eps) for t in (torch.float16, torch.float32, torch.float64))


def max_abs_difference(a, b) -> float:
    return float((torch.as_tensor(a).detach().cpu().double() - torch.as_tensor(b).detach().cpu().double()).abs().max())  # MPS は FP64 を扱えないので CPU で計算する


# --- 008 の構成と重み(Hub の main) ---
REFERENCE_CONFIG = json.loads(Path(hf_hub_download(REFERENCE_MODEL_REPO_ID, "config.json")).read_text(encoding="utf-8"))
REFERENCE_STATE = {k: v.clone() for k, v in torch.load(hf_hub_download(REFERENCE_MODEL_REPO_ID, "model_state.pt"), map_location="cpu").items()}
assert REFERENCE_CONFIG["vocabulary_size"] == tokenizer.vocab_size == 8192 and REFERENCE_CONFIG["d_model"] == EMBEDDING_DIMENSION
assert REFERENCE_CONFIG["sequence_length"] >= QUERY_LENGTH + 1 + PASSAGE_LENGTH + 1 and REFERENCE_CONFIG["tie_embeddings"]


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


def state_hash(model_or_state) -> str:
    state = model_or_state.state_dict() if hasattr(model_or_state, "state_dict") else model_or_state
    digest = hashlib.sha256()
    for _, value in sorted(state.items()):
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def build_embedding_model(seed_index: int = 0) -> TextEmbeddingModel:
    # R1・R2: 008 の重みから始める平均プール・因果マスクのモデル(初期値はシードによらない。024 の C5 と同じ)
    torch.manual_seed(INIT_SEED_BASE + seed_index)
    backbone = build_gpt(REFERENCE_CONFIG)
    backbone.load_state_dict(REFERENCE_STATE)
    return TextEmbeddingModel(backbone, pooling="mean", attention="causal")


def build_cross_encoder(seed_index: int) -> VariableLengthCrossEncoder:
    # cross-encoder: 008 の重み + 区切り・終端 + シードで初期化したヘッド
    torch.manual_seed(INIT_SEED_BASE + seed_index)
    backbone = build_gpt(REFERENCE_CONFIG)
    backbone.load_state_dict(REFERENCE_STATE)
    return VariableLengthCrossEncoder(backbone, SEPARATOR_TOKEN_ID, TERMINAL_TOKEN_ID, HEAD_INIT_STD)


def build_format024_batches(schedule: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # 024 の学習の組(記事の連結から切り出した固定長の区間)。すべての行が満長。学習ループの等価性の確認だけに使う
    starts = np.asarray(TRAIN_STARTS)[schedule["article"]]
    query_index = (starts + schedule["query_start"])[:, :, None] + np.arange(QUERY_LENGTH)[None, None, :]
    passage_index = (starts + schedule["passage_start"])[:, :, None] + np.arange(PASSAGE_LENGTH)[None, None, :]
    shape = schedule["article"].shape
    return (
        TRAIN_STREAM[query_index].astype(np.int64),
        np.full(shape, QUERY_LENGTH, dtype=np.int64),
        TRAIN_STREAM[passage_index].astype(np.int64),
        np.full(shape, PASSAGE_LENGTH, dtype=np.int64),
    )


def small_pair_batches(num_steps: int, articles_per_batch: int, seed: int):
    return sample_cloze_pair_batches(
        TRAIN_STREAM, np.asarray(TRAIN_STARTS), TRAIN_TOKEN_COUNTS, SAMPLING_WEIGHTS, SENTENCE_TABLES, num_steps, articles_per_batch, PAIRS_PER_ARTICLE,
        PASSAGE_LENGTH, QUERY_LENGTH, MIN_QUERY_TOKENS, MAX_REMOVED_TOKENS, KEEP_PROBABILITY, seed=seed,
    )


def segment_texts(token_ids: np.ndarray) -> list[str]:
    # 文の分割の確認用: 1 本のトークン列を文ごとのテキストに戻す
    boundary = sentence_boundaries(token_ids[None, :], SENTENCE_TABLES)[0]
    pieces, current = [], []
    for token, end in zip(token_ids.tolist(), boundary.tolist(), strict=True):
        current.append(token)
        if end:
            pieces.append(tokenizer.decode(current))
            current = []
    return pieces


# --- 文の分割 ---
_document = "The cat sat on the mat. It was a very nice mat indeed! Is it? \"Yes.\" He said (ok). New line here\n\nNext para is long enough to count as a sentence.\n* item one is here\n* item two is there"
_expected = ["The cat sat on the mat.", " It was a very nice mat indeed!", " Is it?", ' "Yes."', " He said (ok).", " New line here", "\n\nNext para is long enough to count as a sentence.", "\n* item one is here", "\n* item two is there"]
assert segment_texts(np.array(tokenizer.encode(_document))) == _expected, segment_texts(np.array(tokenizer.encode(_document)))
_known_error = segment_texts(np.array(tokenizer.encode("He met Dr. Smith. Pi is 3.14 exactly.")))
assert _known_error == ["He met Dr.", " Smith.", " Pi is 3.14 exactly."], _known_error  # 略語の後の空白で分割する(既知の誤り)。小数点では分割しない
_no_period = segment_texts(np.array(tokenizer.encode("alpha beta gamma delta")))
assert _no_period == ["alpha beta gamma delta"]  # 区切りがなければ窓全体が 1 つの文(窓の端は文の端)
print("文の分割: 手で作った文章で規則どおり(改行・文末記号・閉じ引用符/括弧)、略語の後の空白では分割する(既知の誤り)、小数点では分割しない、区切りがなければ窓全体が 1 文: OK")

# --- 抜く文の選択・文を抜く操作(コーパスの実際の窓で) ---
_windows = CORPUS.passages[:2000]
_uniform = np.random.default_rng(0).random(len(_windows))
_valid, _start, _length = choose_cloze_sentence(_windows, SENTENCE_TABLES, _uniform, MIN_QUERY_TOKENS, MAX_REMOVED_TOKENS)
_is_start, _sentence_length, _num_sentences, _boundary = sentence_structure(_windows, SENTENCE_TABLES)
_rows = np.arange(len(_windows))
assert _valid.mean() > 0.9 and (_boundary[:, -1]).all()
assert _is_start[_rows[_valid], _start[_valid]].all() and (_num_sentences[_valid] >= 2).all(), "抜く文が文の先頭から始まっていない、または文が 2 つ未満"
assert ((_length[_valid] >= MIN_QUERY_TOKENS) & (_length[_valid] <= MAX_REMOVED_TOKENS)).all()
assert np.array_equal(_length[_valid], _sentence_length[_rows[_valid], _start[_valid]])
_again = choose_cloze_sentence(_windows, SENTENCE_TABLES, _uniform, MIN_QUERY_TOKENS, MAX_REMOVED_TOKENS)
assert all(np.array_equal(a, b) for a, b in zip((_valid, _start, _length), _again, strict=True)), "抜く文の選択が決定的でない"
_context, _context_length = remove_span(_windows, _start, _length)
_query, _query_length = extract_span(_windows, _start, _length, QUERY_LENGTH)
for _i in np.nonzero(_valid)[0][:300]:  # 文脈に抜いた文を戻すと元の窓に一致する(文が query の長さ以内のとき、query が文そのもの)
    _restored = np.concatenate([_context[_i, : _start[_i]], _windows[_i, _start[_i] : _start[_i] + _length[_i]], _context[_i, _start[_i] : _context_length[_i]]])
    assert np.array_equal(_restored, _windows[_i]) and _context_length[_i] == PASSAGE_LENGTH - _length[_i]
    assert (_context[_i, _context_length[_i] :] == 0).all() and (_query[_i, _query_length[_i] :] == 0).all()
    assert np.array_equal(_query[_i, : _query_length[_i]], _windows[_i, _start[_i] : _start[_i] + _query_length[_i]]) and _query_length[_i] == min(_length[_i], QUERY_LENGTH)
# 対象の文が 2 つある窓では、先頭の文を選ぶ確率が約 1/2(二項分布の標準誤差の 4 倍以内)
_eligible_count = (_is_start & (_sentence_length >= MIN_QUERY_TOKENS) & (_sentence_length <= MAX_REMOVED_TOKENS) & (_num_sentences >= 2)[:, None]).sum(axis=1)
_two = _eligible_count == 2
_first_eligible = np.array([np.nonzero(_is_start[i] & (_sentence_length[i] >= MIN_QUERY_TOKENS) & (_sentence_length[i] <= MAX_REMOVED_TOKENS))[0][0] for i in np.nonzero(_two)[0]])
_picked_first = float(np.mean(_start[_two] == _first_eligible))
assert abs(_picked_first - 0.5) < 4 * math.sqrt(0.25 / _two.sum()), (_picked_first, int(_two.sum()))
print(f"抜く文の選択: 範囲の長さの文の先頭から始まる、決定的、抜いた文を戻すと元の窓に一致する、パディングが 0、対象の文が 2 つの窓({int(_two.sum())} 個)で先頭の文を選ぶ割合 {_picked_first:.3f}: OK")

# --- passage 単位の評価の索引 ---
assert CLOZE.passage_ids.shape == (NUM_INDEX,) and (np.diff(CLOZE.passage_ids) > 0).all()
assert (CLOZE.context_lengths + CLOZE.removed_length == PASSAGE_LENGTH).all(), "索引の passage が「1 文を除いた形」でない"
assert (CLOZE.context_lengths >= PASSAGE_LENGTH - MAX_REMOVED_TOKENS).all() and (CLOZE.removed_length >= MIN_QUERY_TOKENS).all()
_original = CORPUS.passages[CLOZE.passage_ids]
for _i in np.linspace(0, NUM_INDEX - 1, 400).astype(int):
    _s, _n = int(CLOZE.removed_start[_i]), int(CLOZE.removed_length[_i])
    _ctx = CLOZE.context_tokens[_i, : CLOZE.context_lengths[_i]]
    assert np.array_equal(np.concatenate([_ctx[:_s], _original[_i, _s : _s + _n], _ctx[_s:]]), _original[_i]), "抜いた文を戻しても元の passage に一致しない"
    assert (CLOZE.context_tokens[_i, CLOZE.context_lengths[_i] :] == 0).all() and CLOZE.query_lengths[_i] == min(_n, QUERY_LENGTH)
    assert np.array_equal(CLOZE.query_tokens[_i, : CLOZE.query_lengths[_i]], _original[_i, _s : _s + CLOZE.query_lengths[_i]])
_permutation = np.random.default_rng(1).permutation(CORPUS.num_passages)
_permuted = build_cloze_index(
    CORPUS.passages[_permutation], CORPUS.passage_articles[_permutation], CORPUS.passage_windows[_permutation], SENTENCE_TABLES, QUERY_LENGTH, MIN_QUERY_TOKENS, MAX_REMOVED_TOKENS, CLOZE_SEED
)
_by_passage = dict(zip(_permutation[_permuted.passage_ids].tolist(), _permuted.removed_start.tolist(), strict=True))
assert set(_by_passage) == set(CLOZE.passage_ids.tolist()) and all(_by_passage[p] == s for p, s in zip(CLOZE.passage_ids.tolist(), CLOZE.removed_start.tolist(), strict=True)), "抜く文の選択が passage の並びに依る"
_other_seed = build_cloze_index(CORPUS.passages, CORPUS.passage_articles, CORPUS.passage_windows, SENTENCE_TABLES, QUERY_LENGTH, MIN_QUERY_TOKENS, MAX_REMOVED_TOKENS, CLOZE_SEED + 1)
assert not np.array_equal(_other_seed.removed_start, CLOZE.removed_start), "異なるシードで同じ選択(確認の検出力がない)"
_valid_u, _, _length_u = choose_cloze_sentence(
    CORPUS.passages, SENTENCE_TABLES, splitmix64_uniform(CLOZE_SEED, CORPUS.passage_articles, CORPUS.passage_windows), MIN_QUERY_TOKENS, MAX_REMOVED_TOKENS
)
_share_unpreferred = float((_length_u[_valid_u] > QUERY_LENGTH).mean())
assert float((CLOZE.removed_length > QUERY_LENGTH).mean()) < _share_unpreferred, "評価の索引で、収まる文を優先していない"
print(f"評価の索引で切り詰めた query の割合: 優先なし {_share_unpreferred:.1%}、収まる文を優先 {float((CLOZE.removed_length > QUERY_LENGTH).mean()):.1%}")
print(f"passage 単位の評価の索引: すべて「1 文を除いた形」(文脈の長さ + 抜いた文の長さ = {PASSAGE_LENGTH})、抜いた文を戻すと元の passage に一致する、抜く文の選択が passage の並びに依らず決定的、シードを変えると変わる: OK")

# --- 学習の組(Inverse Cloze Task。32 記事 x 2 組) ---
_pairs_a = small_pair_batches(200, ARTICLES_PER_BATCH, 5)
_pairs_b = small_pair_batches(200, ARTICLES_PER_BATCH, 5)
_fields = ("query_tokens", "query_lengths", "context_tokens", "context_lengths", "articles", "kept", "truncated", "window_starts")
assert all(np.array_equal(getattr(_pairs_a, k), getattr(_pairs_b, k)) for k in _fields), "学習の組がシードで決定的でない"
assert not np.array_equal(_pairs_a.context_tokens, small_pair_batches(200, ARTICLES_PER_BATCH, 6).context_tokens), "異なるシードで同じ組(確認の検出力がない)"
_articles = _pairs_a.articles
assert _articles.shape == (200, BATCH_SIZE) and (_articles[:, 0::2] == _articles[:, 1::2]).all(), "同じ記事の 2 組がバッチの中で隣り合っていない"
assert all(len(set(row[0::2].tolist())) == ARTICLES_PER_BATCH for row in _articles), "1 つのバッチに同じ記事が複数入っている"
assert (SAMPLING_WEIGHTS[_articles] > 0).all()
_window_starts = _pairs_a.window_starts
assert (np.abs(_window_starts[:, 0::2] - _window_starts[:, 1::2]) >= PASSAGE_LENGTH).all(), "同じ記事の 2 組の窓が重なっている"
assert (_window_starts >= 0).all() and (_window_starts + PASSAGE_LENGTH <= TRAIN_TOKEN_COUNTS[_articles]).all(), "窓が記事に収まっていない"
_kept = float(_pairs_a.kept.mean())
assert abs(_kept - KEEP_PROBABILITY) < 4 * math.sqrt(KEEP_PROBABILITY * (1 - KEEP_PROBABILITY) / _pairs_a.kept.size), _kept
assert (_pairs_a.context_lengths[_pairs_a.kept] == PASSAGE_LENGTH).all() and (_pairs_a.context_lengths[~_pairs_a.kept] < PASSAGE_LENGTH).all()
assert (_pairs_a.context_lengths >= PASSAGE_LENGTH - MAX_REMOVED_TOKENS).all() and (_pairs_a.query_lengths >= MIN_QUERY_TOKENS).all() and (_pairs_a.query_lengths <= QUERY_LENGTH).all()
assert (_pairs_a.context_tokens * (np.arange(PASSAGE_LENGTH)[None, None, :] >= _pairs_a.context_lengths[:, :, None])).sum() == 0
assert (_pairs_a.query_tokens * (np.arange(QUERY_LENGTH)[None, None, :] >= _pairs_a.query_lengths[:, :, None])).sum() == 0
# 窓の切り出しと、抜いた文・文脈の整合(ストリームから窓を取り直し、文脈に抜いた文を戻すと窓に一致する)
for _step, _slot in itertools.product((0, 57, 199), (0, 1, 30, 63)):
    _window = TRAIN_STREAM[TRAIN_STARTS[_articles[_step, _slot]] + _window_starts[_step, _slot] :][:PASSAGE_LENGTH].astype(np.int64)
    _context = _pairs_a.context_tokens[_step, _slot, : _pairs_a.context_lengths[_step, _slot]]
    _query = _pairs_a.query_tokens[_step, _slot, : _pairs_a.query_lengths[_step, _slot]]
    if _pairs_a.kept[_step, _slot]:
        assert np.array_equal(_context, _window)
    else:
        _removed = PASSAGE_LENGTH - len(_context)
        _starts = [s for s in range(len(_context) + 1) if np.array_equal(np.concatenate([_context[:s], _window[s : s + _removed], _context[s:]]), _window) and np.array_equal(_window[s : s + len(_query)], _query)]
        assert _starts and (_pairs_a.truncated[_step, _slot] == (_removed > QUERY_LENGTH)), (_step, _slot)
# 抜く文の選び方: query の長さの上限以内に収まる文を優先する。収まる文がある窓では、切り詰めない
_short_windows = CORPUS.passages[:3000]
_valid_s, _start_s, _length_s = choose_cloze_sentence(_short_windows, SENTENCE_TABLES, np.random.default_rng(4).random(len(_short_windows)), MIN_QUERY_TOKENS, MAX_REMOVED_TOKENS, QUERY_LENGTH)
_valid_n, _start_n, _length_n = choose_cloze_sentence(_short_windows, SENTENCE_TABLES, np.random.default_rng(4).random(len(_short_windows)), MIN_QUERY_TOKENS, MAX_REMOVED_TOKENS)
_is_start_s, _sentence_length_s, _num_sentences_s, _ = sentence_structure(_short_windows, SENTENCE_TABLES)
_has_short = (_is_start_s & (_sentence_length_s >= MIN_QUERY_TOKENS) & (_sentence_length_s <= QUERY_LENGTH) & (_num_sentences_s >= 2)[:, None]).any(axis=1)
assert np.array_equal(_valid_s, _valid_n), "優先しても、対象の窓の集合が変わっている"
assert (_length_s[_valid_s & _has_short] <= QUERY_LENGTH).all(), "収まる文がある窓で、長い文を選んでいる"
assert (_length_s[_valid_s & ~_has_short] > QUERY_LENGTH).all() and (_length_s[_valid_s] <= MAX_REMOVED_TOKENS).all()
_truncated_before, _truncated_after = float((_length_n[_valid_n] > QUERY_LENGTH).mean()), float((_length_s[_valid_s] > QUERY_LENGTH).mean())
assert _truncated_after < _truncated_before
print(
    f"学習の組(Inverse Cloze Task、{ARTICLES_PER_BATCH} 記事 x {PAIRS_PER_ARTICLE} 組): 200 ステップ x {BATCH_SIZE} 組で、シードで決定的、同じ記事の 2 組がバッチの中で隣り合い窓が重ならない、1 つのバッチに同じ記事が 1 回だけ、"
    f"文を文脈に残す割合 {_kept:.4f}(期待 {KEEP_PROBABILITY})、文脈・query の長さの範囲、パディングが 0、ストリームの窓と文脈・抜いた文の整合、窓を引き直した回数 {_pairs_a.num_attempts - _pairs_a.kept.size // 2}、"
    f"記事の全窓から選んだ組 {_pairs_a.num_exhaustive}: OK。切り詰めた query の割合(学習の組) {float(_pairs_a.truncated.mean()):.1%}"
    f"(窓の抜く文を一様に選ぶと {_truncated_before:.1%}、収まる文を優先すると {_truncated_after:.1%}): OK"
)

# --- 重ならない対象の窓の組が取れるかの判定(pairable_articles)と、取れない記事での停止 ---
_synthetic = np.array(
    tokenizer.encode("The cat sat on the mat. The dog ran far away from here today. " + "word " * 400), dtype=np.int32
)  # 文の区切りが先頭の数十トークンにしかない記事: 対象の窓の位置が先頭に偏る
assert len(_synthetic) >= PAIRS_PER_ARTICLE * PASSAGE_LENGTH
_real = [a for a in np.nonzero(SAMPLING_WEIGHTS > 0)[0][:6].tolist()]
_pieces = [_synthetic] + [TRAIN_STREAM[TRAIN_STARTS[a] : TRAIN_STARTS[a] + TRAIN_TOKEN_COUNTS[a]] for a in _real]
_test_stream = np.concatenate(_pieces).astype(np.int32)
_test_counts = np.array([len(x) for x in _pieces], dtype=np.int64)
_test_starts = np.concatenate([[0], np.cumsum(_test_counts)[:-1]])
_test_pairable = pairable_articles(_test_stream, _test_starts, _test_counts, np.ones(len(_pieces), dtype=bool), PASSAGE_LENGTH, SENTENCE_TABLES, MIN_QUERY_TOKENS, MAX_REMOVED_TOKENS)
_brute = []  # 全窓を調べる素朴な実装(組が取れるか = 対象の窓の位置の最大値 - 最小値 が窓の長さ以上)
for _k in range(len(_pieces)):
    _e = _eligible_window_positions(_test_stream, int(_test_starts[_k]), int(_test_counts[_k]), PASSAGE_LENGTH, SENTENCE_TABLES, MIN_QUERY_TOKENS, MAX_REMOVED_TOKENS)
    _brute.append(len(_e) > 0 and any((np.abs(_e - x) >= PASSAGE_LENGTH).any() for x in _e.tolist()))
assert _test_pairable.tolist() == _brute, "pairable_articles が全窓を調べる素朴な実装と一致しない"
assert not _test_pairable[0] and _test_pairable[1:].all(), "文の区切りが先頭にしかない記事は組が取れない記事と判定され、通常の記事は取れると判定される"
_weights_bad = np.ones(len(_pieces))  # 重みが正でも組が取れない記事を含む: 以前はここで例外になった
try:
    sample_cloze_pair_batches(_test_stream, _test_starts, _test_counts, _weights_bad, SENTENCE_TABLES, 200, 7, 2, PASSAGE_LENGTH, QUERY_LENGTH, MIN_QUERY_TOKENS, MAX_REMOVED_TOKENS, KEEP_PROBABILITY, seed=1)
    raise AssertionError("組が取れない記事を含む重みで、例外にならなかった")
except RuntimeError as _error:
    assert "取れない" in str(_error)
_weights_ok = np.where(_test_pairable, 1.0, 0.0)  # 判定で除くと、最後まで終わる(全ステップで、組が取れる記事だけが選ばれる)
_zero_sampled = sample_cloze_pair_batches(_test_stream, _test_starts, _test_counts, _weights_ok, SENTENCE_TABLES, 200, 5, 2, PASSAGE_LENGTH, QUERY_LENGTH, MIN_QUERY_TOKENS, MAX_REMOVED_TOKENS, KEEP_PROBABILITY, seed=1)
assert (_zero_sampled.articles != 0).all()
_cache: dict[int, np.ndarray] = {}
_with_cache = sample_cloze_pair_batches(_test_stream, _test_starts, _test_counts, _weights_ok, SENTENCE_TABLES, 200, 5, 2, PASSAGE_LENGTH, QUERY_LENGTH, MIN_QUERY_TOKENS, MAX_REMOVED_TOKENS, KEEP_PROBABILITY, seed=1, eligible_cache=_cache)
assert all(np.array_equal(getattr(_zero_sampled, f), getattr(_with_cache, f)) for f in ("query_tokens", "context_tokens", "articles", "window_starts", "kept")), "走査の結果の使い回しで出力が変わった"
print(f"重ならない対象の窓の組が取れるかの判定: 全窓を調べる素朴な実装と一致({len(_pieces)} 記事。区切りが先頭にしかない記事は取れない、通常の記事は取れる)、取れない記事を含む重みでは例外、除くと最後まで終わり、走査の結果を使い回しても出力は同じ: OK")

# --- 可変長のモデル(008 の重み、CPU、FP32) ---
_model = build_embedding_model().eval()
_torch_generator = torch.Generator().manual_seed(1)
_tokens = torch.randint(0, 8192, (6, 50), generator=_torch_generator)
_lengths = torch.tensor([50, 31, 10, 2, 1, 40])
with torch.no_grad():
    _padded = pooled_with_lengths(_model, _tokens, _lengths)
    assert _padded.dtype == torch.float32
    for _i in range(len(_lengths)):  # パディングなし(その行の長さだけ)の出力と一致する
        _reference = _model.pooled(_tokens[_i : _i + 1, : _lengths[_i]])[0]
        assert max_abs_difference(_padded[_i], _reference) <= 64 * EPSILON_FP32 * float(_reference.abs().max()), (_i, max_abs_difference(_padded[_i], _reference))
    _garbage = _tokens.clone()
    for _i in range(len(_lengths)):
        _garbage[_i, _lengths[_i] :] = torch.randint(0, 8192, (50 - int(_lengths[_i]),), generator=_torch_generator)
    assert torch.equal(pooled_with_lengths(_model, _garbage, _lengths), _padded), "パディングの内容が出力に影響している"
    assert torch.equal(pooled_with_lengths(_model, _tokens, torch.full((6,), 50), full_length=True), _model.pooled(_tokens)), "満長の経路が TextEmbeddingModel.pooled() と一致しない"
    _wide = encode_tokens_with_lengths(_model, _tokens, _lengths.numpy(), 6, (None, 16))
    _narrow = encode_tokens_with_lengths(_model, _tokens, _lengths.numpy(), 2, (None, 16))
    _repeat = max_abs_difference(_wide[256], encode_tokens_with_lengths(_model, _tokens, _lengths.numpy(), 6)[256])
    _unit = EPSILON_FP32 * float(_wide[256].abs().max())
    assert max_abs_difference(_wide[256], _narrow[256]) <= 16 * max(_repeat, _unit) and max_abs_difference(_wide[16], _narrow[16]) <= 16 * max(_repeat, _unit)
    _expected_unit = torch.nn.functional.normalize(_padded, dim=-1)
    assert max_abs_difference(_wide[256], _expected_unit) <= 16 * max(_repeat, _unit), "符号化の結果が元の並びに戻っていない、または正規化が違う"
print(f"可変長の平均プール: パディングなしの出力と一致(許容は丸めの単位 x 値の大きさの 64 倍)、パディングの内容が出力に影響しない(bit 単位)、満長の経路が pooled() と bit 単位で一致、バッチの大きさに依らず元の並びに戻る: OK")

_cross = build_cross_encoder(0).eval()
_reference_cross = CrossEncoder(_cross.backbone, SEPARATOR_TOKEN_ID, TERMINAL_TOKEN_ID, HEAD_INIT_STD).eval()
_reference_cross.head.load_state_dict(_cross.head.state_dict())
_query_tokens = torch.randint(0, 8192, (5, 12), generator=_torch_generator)
_passage_tokens = torch.randint(0, 8192, (5, 20), generator=_torch_generator)
_query_lengths = torch.tensor([12, 7, 1, 12, 4])
_passage_lengths = torch.tensor([20, 20, 3, 9, 15])
_inputs, _total = build_variable_cross_encoder_inputs(_query_tokens, _query_lengths, _passage_tokens, _passage_lengths, SEPARATOR_TOKEN_ID, TERMINAL_TOKEN_ID)
for _i in range(5):  # 列は左から詰まっている: [query][区切り][passage][終端]、終端の後は 0
    _lq, _lp = int(_query_lengths[_i]), int(_passage_lengths[_i])
    _row = _inputs[_i].tolist()
    assert _row[:_lq] == _query_tokens[_i, :_lq].tolist() and _row[_lq] == SEPARATOR_TOKEN_ID and _row[_lq + 1 : _lq + 1 + _lp] == _passage_tokens[_i, :_lp].tolist()
    assert _row[_lq + 1 + _lp] == TERMINAL_TOKEN_ID and sum(abs(v) for v in _row[_lq + 2 + _lp :]) == 0 and int(_total[_i]) == _lq + _lp + 2
_full_inputs = build_cross_encoder_inputs(_query_tokens, _passage_tokens, SEPARATOR_TOKEN_ID, TERMINAL_TOKEN_ID)
_inputs_full, _ = build_variable_cross_encoder_inputs(_query_tokens, torch.full((5,), 12), _passage_tokens, torch.full((5,), 20), SEPARATOR_TOKEN_ID, TERMINAL_TOKEN_ID)
assert torch.equal(_inputs_full, _full_inputs), "満長の列が 025 の build_cross_encoder_inputs() と一致しない"
with torch.no_grad():
    _scores = _cross(_query_tokens, _passage_tokens, _query_lengths, _passage_lengths)
    for _i in range(5):  # パディングなしの 025 の CrossEncoder と同じスコア
        _single = _reference_cross(_query_tokens[_i : _i + 1, : _query_lengths[_i]], _passage_tokens[_i : _i + 1, : _passage_lengths[_i]])[0]
        assert abs(float(_scores[_i]) - float(_single)) <= 64 * EPSILON_FP32 * max(abs(float(_single)), 1.0), (_i, float(_scores[_i]), float(_single))
    _garbage_q, _garbage_p = _query_tokens.clone(), _passage_tokens.clone()
    for _i in range(5):
        _garbage_q[_i, _query_lengths[_i] :] = 7
        _garbage_p[_i, _passage_lengths[_i] :] = 11
    assert torch.equal(_cross(_garbage_q, _garbage_p, _query_lengths, _passage_lengths), _scores), "パディングの内容がスコアに影響している"
    _changed_passage = _passage_tokens.clone()
    _changed_passage[0, 19] = (_changed_passage[0, 19] + 1) % 8192  # 満長の行 0 の passage の最後のトークン
    assert float(_cross(_query_tokens, _changed_passage, _query_lengths, _passage_lengths)[0]) != float(_scores[0]), "終端の位置の出力が passage の最後のトークンに依存しない"
    _changed_query = _query_tokens.clone()
    _changed_query[1, 0] = (_changed_query[1, 0] + 1) % 8192
    assert float(_cross(_changed_query, _passage_tokens, _query_lengths, _passage_lengths)[1]) != float(_scores[1]), "終端の位置の出力が query に依存しない"
print("可変長の cross-encoder: 列が左から詰まっている([query][区切り][passage][終端])、満長のとき 025 の入力と bit 単位で一致、パディングなしの 025 の CrossEncoder と同じスコア、パディングの内容がスコアに影響しない(bit 単位)、終端の位置が query と passage の両方に依存する: OK")
del _model, _cross, _reference_cross

# --- 指標: 単一の正例の順位と、同点の扱い ---
_generator = torch.Generator().manual_seed(2)
_passage_embeddings = torch.nn.functional.normalize(torch.randn(60, 8, generator=_generator, dtype=torch.float64), dim=-1)
_query_embeddings = torch.nn.functional.normalize(torch.randn(10, 8, generator=_generator, dtype=torch.float64), dim=-1)
_positive = np.random.default_rng(3).integers(0, 60, size=10)
_ranks = rank_single_positive(_query_embeddings, _passage_embeddings, _positive, chunk_size=4)
for _i in range(10):
    _scores_i = (_passage_embeddings @ _query_embeddings[_i]).numpy()
    _naive = 1 + sum(1 for p in range(60) if p != _positive[_i] and _scores_i[p] >= _scores_i[_positive[_i]])
    assert _ranks[_i] == _naive
_constant = torch.nn.functional.normalize(torch.ones(1, 8), dim=-1)
_constant_ranks = rank_single_positive(_constant.repeat(5, 1), _constant.repeat(30, 1), np.array([0, 5, 10, 20, 29]))
assert (_constant_ranks == 30).all() and not (_constant_ranks <= 10).any(), "同点を正例に不利に数えていない(全件が同じスコアなら、順位は最下位)"
_tie = rerank_single_positive(np.array([[1.0, 1.0, 0.0]]), np.array([[3, 7, 2]]), np.array([7]), 10)
assert _tie.rank.tolist() == [2], "同点の負例が正例より上位でない"
assert rerank_single_positive(np.array([[0.5, 0.9, 0.1]]), np.array([[4, 7, 2]]), np.array([7]), 10).rank.tolist() == [1]
_constant_rerank = rerank_single_positive(np.zeros((20, 50)), np.tile(np.arange(50), (20, 1)), np.arange(20), 100)
assert _constant_rerank.coverage() == 1.0 and _constant_rerank.recall_at(10) == 0.0 and (_constant_rerank.rank == 50).all(), "全候補に同じスコアを返すモデルの Recall@10 が 0 でない"
assert rerank_single_positive(np.zeros((1, 5)), np.array([[0, 1, 2, 3, 4]]), np.array([99]), 100).rank.tolist() == [6], "候補に正例がないときの順位が K + 1 でない"
print("指標: 単一の正例の順位が素朴な実装と一致、同点は正例に不利(全件同じスコアで順位が最下位、全候補に同じスコアを返すモデルの Recall@10 が 0)、候補に正例がなければ順位 K + 1: OK")

# --- 指標: 記事単位のラベル(同じ索引・同じ query。正解は同じ記事の他の passage、query の passage 自身は候補から除く)---
_articles_small = np.repeat(np.arange(15), 4)  # 15 記事 x 4 passage
_label_passages = torch.nn.functional.normalize(torch.randn(60, 8, generator=_generator, dtype=torch.float64), dim=-1)
_label_positions = np.array([0, 5, 6, 13, 22, 40, 41, 59])
_label_queries = torch.nn.functional.normalize(torch.randn(8, 8, generator=_generator, dtype=torch.float64), dim=-1)
_article_result = rank_article_label(_label_queries, _label_passages, _label_positions, _articles_small, chunk_size=3)
for _i, _p in enumerate(_label_positions):
    _scores_i = (_label_passages @ _label_queries[_i]).numpy()
    _mates = [m for m in range(60) if _articles_small[m] == _articles_small[_p] and m != _p]
    _best = max(_scores_i[m] for m in _mates)
    _naive = 1 + sum(1 for c in range(60) if c != _p and c not in _mates and _scores_i[c] >= _best)
    assert _article_result.rank[_i] == _naive, (_i, _article_result.rank[_i], _naive)
assert (_article_result.num_positives == 3).all()
_constant_article = rank_article_label(_constant.repeat(3, 1), _constant.repeat(30, 1), np.array([0, 9, 17]), np.repeat(np.arange(10), 3))
assert (_constant_article.rank == 28).all() and not (_constant_article.rank <= 10).any(), "同点を正解に不利に数えていない(記事単位のラベル)"
print("指標(記事単位のラベル): 素朴な実装と一致(query の passage 自身は候補から除く、最もスコアの高い正解の順位、同点は正解に不利)、全件同じスコアで Recall@10 が 0: OK")

# --- マスクつきの in-batch negatives(同じ記事の組の扱い)---
_u = torch.nn.functional.normalize(torch.randn(6, 16, generator=_generator, dtype=torch.float64), dim=-1)
_v = torch.nn.functional.normalize(torch.randn(6, 16, generator=_generator, dtype=torch.float64), dim=-1)
_ids = np.array([0, 0, 1, 1, 2, 2])
_ignore = torch.from_numpy((_ids[:, None] == _ids[None, :]) & ~np.eye(6, dtype=bool))
_masked = masked_info_nce_loss(_u, _v, TEMPERATURE, _ignore)
_logits = (_u @ _v.t() / TEMPERATURE).numpy()
_expected_terms = []
for _i in range(6):
    _candidates = [j for j in range(6) if j == _i or _ids[j] != _ids[_i]]  # 正例と、別の記事の passage だけ
    _expected_terms.append(-(_logits[_i, _i] - math.log(sum(math.exp(_logits[_i, j]) for j in _candidates))))
assert abs(float(_masked) - float(np.mean(_expected_terms))) <= 64 * EPSILON_FP64 * abs(float(np.mean(_expected_terms)))
_unmasked = masked_info_nce_loss(_u, _v, TEMPERATURE, torch.zeros(6, 6, dtype=torch.bool))
assert abs(float(_unmasked) - float(info_nce_loss(_u, _v, TEMPERATURE))) <= 64 * EPSILON_FP64 * float(_unmasked), "除かないとき、通常の InfoNCE と一致しない"
assert float(_masked) < float(_unmasked), "候補を除いたのに損失が下がっていない(負例が減ると下がる)"
print(f"マスクつきの in-batch negatives: 同じ記事の組の passage を負例から除いた損失が閉形式の値と一致(FP64)、除かないとき通常の InfoNCE と一致、除くと損失は下がる({float(_masked):.4f} < {float(_unmasked):.4f}): OK")

# --- 学習ループ(008 の重み、CPU、FP32、小さい組) ---
_schedule_small = sample_pair_schedule(TRAIN_TOKEN_COUNTS, SAMPLING_WEIGHTS, QUERY_LENGTH, PASSAGE_LENGTH, 4, BATCH_SIZE, seed=5)  # 024 の学習の組(満長の区間)。学習ループの等価性の確認だけに使う
_small = {k: v[:2, :8] for k, v in _schedule_small.items()}
_small_format024 = build_format024_batches(_small)
_stream_tensor = torch.from_numpy(TRAIN_STREAM.astype(np.int64))
for _case in ("r1_equivalence", "zero_learning_rate"):
    _lr = 1e-4 if _case == "r1_equivalence" else 0.0
    _reference_model = build_embedding_model(0)
    _new_model = build_embedding_model(0)
    _reference_history = train_text_embedding(
        _reference_model, _stream_tensor, TRAIN_STARTS, _small, QUERY_LENGTH, PASSAGE_LENGTH, _lr, 1, 0.0 if _lr == 0 else 1e-6, WEIGHT_DECAY,
        GRADIENT_CLIP_THRESHOLD, TEMPERATURE, matryoshka_dimensions=MATRYOSHKA_DIMENSIONS,
    )
    _new_batches = VariableBatches(*_small_format024, torch.device("cpu"))
    _initial_hash = state_hash(_new_model)
    _new_history = train_text_embedding_from_batches(
        _new_model, _new_batches, _lr, 1, 0.0 if _lr == 0 else 1e-6, WEIGHT_DECAY, GRADIENT_CLIP_THRESHOLD, TEMPERATURE, matryoshka_dimensions=MATRYOSHKA_DIMENSIONS
    )
    # 同じ演算・同じ順序なので、損失の系列は一致する(許容: 丸めの単位 x 損失の大きさ x 16)
    assert np.allclose(_new_history["loss"], _reference_history["loss"], rtol=16 * EPSILON_FP32, atol=0.0), (_new_history["loss"], _reference_history["loss"])
    if _case == "zero_learning_rate":
        assert state_hash(_new_model) == _initial_hash, "学習率 0 の学習でモデルが変化した"
        _measured = compute_pair_losses_from_batches(_new_model, _new_batches, range(2), TEMPERATURE, matryoshka_dimensions=MATRYOSHKA_DIMENSIONS)
        assert state_hash(_new_model) == _initial_hash and _new_model.training, "compute_pair_losses_from_batches() がモデルまたはモードを変更した"
        assert np.allclose(_new_history["loss"], _measured, rtol=16 * EPSILON_FP32, atol=0.0), "学習率 0 の訓練損失が、同じ組での学習前のモデルの損失と一致しない"
        _reference_pair = compute_pair_losses(_reference_model, _stream_tensor, TRAIN_STARTS, _small, range(2), QUERY_LENGTH, PASSAGE_LENGTH, TEMPERATURE, matryoshka_dimensions=MATRYOSHKA_DIMENSIONS)
        assert np.allclose(_measured, _reference_pair, rtol=16 * EPSILON_FP32, atol=0.0)
        # 可変長の組(Inverse Cloze Task の組)でも、R2(通常の InfoNCE の負例)と R1(同じ記事の組を負例から除く)の両方で、
        # 学習率 0 の訓練損失が同じ組での学習前のモデルの損失と一致し、モデルが変化しない
        _tiny_pairs = small_pair_batches(2, 4, 3)  # 4 記事 x 2 組 = 8 組
        _swap = np.arange(8) ^ 1
        for _label, _context_tokens, _context_lengths, _mask in (
            ("R2", _tiny_pairs.context_tokens, _tiny_pairs.context_lengths, False),
            ("R1", _tiny_pairs.context_tokens[:, _swap], _tiny_pairs.context_lengths[:, _swap], True),
        ):
            _variable = VariableBatches(
                _tiny_pairs.query_tokens, _tiny_pairs.query_lengths, _context_tokens, _context_lengths, torch.device("cpu"),
                article_ids=_tiny_pairs.articles, mask_same_article=_mask,
            )
            _variable_model = build_embedding_model(0)
            _variable_before = compute_pair_losses_from_batches(_variable_model, _variable, range(2), TEMPERATURE, matryoshka_dimensions=MATRYOSHKA_DIMENSIONS)
            _variable_history = train_text_embedding_from_batches(
                _variable_model, _variable, 0.0, 1, 0.0, WEIGHT_DECAY, GRADIENT_CLIP_THRESHOLD, TEMPERATURE, matryoshka_dimensions=MATRYOSHKA_DIMENSIONS
            )
            assert state_hash(_variable_model) == _initial_hash and np.allclose(_variable_history["loss"], _variable_before, rtol=16 * EPSILON_FP32, atol=0.0), _label
            _ratio = float(np.mean(_variable_history["loss"]) / np.mean(_variable_before))
            assert abs(_ratio - 1.0) <= 16 * EPSILON_FP32 and _ratio > P1_LOSS_RATIO, "更新なしの訓練損失 / 学習前のモデルの損失が 1 でない(P1(a) が学習率 0 で不成立にならない)"
            assert _variable.ignore(0) is None if not _mask else int(_variable.ignore(0).sum()) == 8  # R1: 各行で同じ記事のもう一方の組が 1 つ除かれる
            assert (_variable.negatives_per_row() == (7 if not _mask else 6)).all()
        assert _variable_history["data_stream_hash"] != _new_history["data_stream_hash"]
print("第 1 段の学習ループ: 満長の組で 024 の train_text_embedding() と同じ損失の系列、学習率 0 の学習(R1・R2 の組、R1 は同じ記事の組を負例から除く)で訓練損失が同じ組での学習前のモデルの損失と一致しモデルが変化しない(比が 1 で P1(a) は不成立): OK")

# --- cross-encoder の学習の組と学習ループ ---
_mined = np.stack([np.concatenate([[q], np.random.default_rng(q).choice(NUM_INDEX, size=RERANK_CANDIDATES - 1, replace=False)]) for q in TRAIN_ENTRIES[:60]])
_mined[::3, 0] = (_mined[::3, 0] + 1) % NUM_INDEX  # 3 つに 1 つは、先頭を別の passage に置き換える(正例が入らない、または別の位置)
_mined[::3] = np.where(_mined[::3] == TRAIN_ENTRIES[:60][::3, None], (_mined[::3] + 7) % NUM_INDEX, _mined[::3])  # 正例を上位から外す
_usable = usable_cloze_queries(_mined, TRAIN_ENTRIES[:60])
assert 0 < _usable.sum() < 60 and not _usable[::3].any()
_schedule_ce = sample_cloze_reranking_schedule(TRAIN_ENTRIES[:60], _mined, 6, BATCH_QUERIES, NUM_NEGATIVES, seed=7)
_again_ce = sample_cloze_reranking_schedule(TRAIN_ENTRIES[:60], _mined, 6, BATCH_QUERIES, NUM_NEGATIVES, seed=7)
assert _schedule_ce.digest() == _again_ce.digest() != sample_cloze_reranking_schedule(TRAIN_ENTRIES[:60], _mined, 6, BATCH_QUERIES, NUM_NEGATIVES, seed=8).digest()
for _step in range(6):
    for _b in range(BATCH_QUERIES):
        _row = _schedule_ce.candidate_ids[_step, _b]
        _q = int(_schedule_ce.query_ids[_step, _b])
        _members = _mined[np.nonzero(TRAIN_ENTRIES[:60] == _q)[0][0]]
        assert _row[0] == _q and _q in _members, "正例が先頭でない、または上位の候補に入っていない"
        assert len(set(_row.tolist())) == 1 + NUM_NEGATIVES and set(_row[1:].tolist()) <= set(_members.tolist()) and _q not in _row[1:].tolist()
assert _usable[np.searchsorted(TRAIN_ENTRIES[:60], _schedule_ce.query_ids.reshape(-1))].all(), "学習に使えない query が学習の組に入っている"
_tables = VariableTables(CLOZE.query_tokens, CLOZE.query_lengths, CLOZE.context_tokens, CLOZE.context_lengths, torch.device("cpu"))
_cross = build_cross_encoder(0)
_initial_hash = state_hash(_cross)
_before = compute_reranking_losses_variable(_cross, _tables, _schedule_ce, range(3))
assert state_hash(_cross) == _initial_hash and _before.shape == (3,) and _before.dtype == np.float64
_history = train_reranker_variable(_cross, _tables, _schedule_ce, 0.0, 1, 0.0, WEIGHT_DECAY, GRADIENT_CLIP_THRESHOLD)
assert state_hash(_cross) == _initial_hash, "学習率 0 の学習でモデルが変化した"
assert np.allclose(_history["loss"][:3], _before, rtol=16 * EPSILON_FP32, atol=0.0), "学習率 0 の訓練損失が、同じ組での学習前のモデルの損失と一致しない"
assert abs(float(np.mean(_history["loss"][:3])) - UNIFORM_LOSS_CROSS_ENCODER) < 0.2, "学習前の cross-encoder の損失が一様な予測の損失 ln(1 + n) の付近にない"
print(f"cross-encoder の学習: 学習の組の正例が先頭で上位の候補に入り、負例が正例を含まず重複しない、決定的。学習率 0 の訓練損失が同じ組での学習前のモデルの損失と一致しモデルが変化しない(学習前の損失 {float(np.mean(_before)):.3f}、一様な予測 {UNIFORM_LOSS_CROSS_ENCODER:.3f}): OK")
del _cross, _tables

# --- 既存モジュールの後方互換性: 変更前のコミットと比べて、src/・scripts/ の既存のファイルに変更も削除もない ---
_REFERENCE_COMMIT = "8ca43d1"  # 026 の変更を始める前の最後のコミット
_diff = subprocess.run(["git", "diff", "--name-only", "--diff-filter=MD", _REFERENCE_COMMIT, "--", "src", "scripts"], capture_output=True, text=True)
assert _diff.returncode == 0, _diff.stderr
assert _diff.stdout.strip() == "", f"既存のファイルが変更または削除されている: {_diff.stdout}"
_existing = subprocess.run(["git", "ls-tree", "-r", "--name-only", _REFERENCE_COMMIT, "src", "scripts"], capture_output=True, text=True).stdout.split()
print(f"既存モジュールの後方互換性: 参照コミット {_REFERENCE_COMMIT} の src/・scripts/ の既存のファイル {len(_existing)} 個に、変更も削除もない: OK")
CHECK_SECONDS = time.time() - _t0_checks
print(f"単体テストと不変条件の確認 {CHECK_SECONDS:.1f} 秒")
```


    config.json:   0%|          | 0.00/305 [00:00<?, ?B/s]



    model_state.pt: reconstructing file:   0%|          |  0.00B / 21.0MB            



    model_state.pt: downloading bytes:           |  0.00B            


    文の分割: 手で作った文章で規則どおり(改行・文末記号・閉じ引用符/括弧)、略語の後の空白では分割する(既知の誤り)、小数点では分割しない、区切りがなければ窓全体が 1 文: OK
    抜く文の選択: 範囲の長さの文の先頭から始まる、決定的、抜いた文を戻すと元の窓に一致する、パディングが 0、対象の文が 2 つの窓(66 個)で先頭の文を選ぶ割合 0.515: OK
    評価の索引で切り詰めた query の割合: 優先なし 31.9%、収まる文を優先 4.5%
    passage 単位の評価の索引: すべて「1 文を除いた形」(文脈の長さ + 抜いた文の長さ = 128)、抜いた文を戻すと元の passage に一致する、抜く文の選択が passage の並びに依らず決定的、シードを変えると変わる: OK
    学習の組(Inverse Cloze Task、32 記事 x 2 組): 200 ステップ x 64 組で、シードで決定的、同じ記事の 2 組がバッチの中で隣り合い窓が重ならない、1 つのバッチに同じ記事が 1 回だけ、文を文脈に残す割合 0.0969(期待 0.1)、文脈・query の長さの範囲、パディングが 0、ストリームの窓と文脈・抜いた文の整合、窓を引き直した回数 1556、記事の全窓から選んだ組 11: OK。切り詰めた query の割合(学習の組) 4.0%(窓の抜く文を一様に選ぶと 27.6%、収まる文を優先すると 3.5%): OK
    重ならない対象の窓の組が取れるかの判定: 全窓を調べる素朴な実装と一致(7 記事。区切りが先頭にしかない記事は取れない、通常の記事は取れる)、取れない記事を含む重みでは例外、除くと最後まで終わり、走査の結果を使い回しても出力は同じ: OK
    可変長の平均プール: パディングなしの出力と一致(許容は丸めの単位 x 値の大きさの 64 倍)、パディングの内容が出力に影響しない(bit 単位)、満長の経路が pooled() と bit 単位で一致、バッチの大きさに依らず元の並びに戻る: OK
    可変長の cross-encoder: 列が左から詰まっている([query][区切り][passage][終端])、満長のとき 025 の入力と bit 単位で一致、パディングなしの 025 の CrossEncoder と同じスコア、パディングの内容がスコアに影響しない(bit 単位)、終端の位置が query と passage の両方に依存する: OK
    指標: 単一の正例の順位が素朴な実装と一致、同点は正例に不利(全件同じスコアで順位が最下位、全候補に同じスコアを返すモデルの Recall@10 が 0)、候補に正例がなければ順位 K + 1: OK
    指標(記事単位のラベル): 素朴な実装と一致(query の passage 自身は候補から除く、最もスコアの高い正解の順位、同点は正解に不利)、全件同じスコアで Recall@10 が 0: OK
    マスクつきの in-batch negatives: 同じ記事の組の passage を負例から除いた損失が閉形式の値と一致(FP64)、除かないとき通常の InfoNCE と一致、除くと損失は下がる(8.1971 < 8.1988): OK
    第 1 段の学習ループ: 満長の組で 024 の train_text_embedding() と同じ損失の系列、学習率 0 の学習(R1・R2 の組、R1 は同じ記事の組を負例から除く)で訓練損失が同じ組での学習前のモデルの損失と一致しモデルが変化しない(比が 1 で P1(a) は不成立): OK
    cross-encoder の学習: 学習の組の正例が先頭で上位の候補に入り、負例が正例を含まず重複しない、決定的。学習率 0 の訓練損失が同じ組での学習前のモデルの損失と一致しモデルが変化しない(学習前の損失 2.040、一様な予測 2.079): OK
    既存モジュールの後方互換性: 参照コミット 8ca43d1 の src/・scripts/ の既存のファイル 82 個に、変更も削除もない: OK
    単体テストと不変条件の確認 44.6 秒




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/026_inverse_cloze_task_and_two_stage_retrieval.ipynb
