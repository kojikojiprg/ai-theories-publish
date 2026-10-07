---
title: "テキスト埋め込みと retriever / Text Embedding and Retriever(実装・実験編 3/7)"
---

この記事は後編(実装・実験編 3/7)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/024_text_embedding_and_retriever-practice-2)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/024_text_embedding_and_retriever-practice-4)。

### 5.4 モデルの構築・単体テスト・不変条件の確認

**モデルの構築**: 条件 $c$・シード $s$ のモデルは、次の手順で作る。

1. `torch.manual_seed(24200 + s)`の後に、008 と同じ構成の小型 GPT(4 層、$d_{\mathrm{model}} = 256$、ヘッド数 8、RoPE、RMSNorm、SwiGLU、正規化前置、埋め込みと出力層の重み共有、系列長 256、
   語彙サイズ 8192。構成は Hub の`config.json`から読む)をランダム初期化する。
2. 初期値が`"pretrained"`の条件(C1・C2・C3・C5)は、008 の学習済みの重み(`kojikojiprg/ai-theories-small-gpt-en`の`main`)を読み込む。`"random"`の条件(C4)はそのまま使う。
3. 条件のプーリングと注意マスクで`TextEmbeddingModel`に包む。

したがって、008 の重みから始める条件の初期値はシードによらず同じで(シードが動かすのは学習の組と順序だけ)、C4 の初期値だけがシードに依存する。

**単体テスト**(CPU、小さな次元。許容の誤差は、実際に計算される型の丸めの単位 × 値の大きさ から導き、固定の絶対値は使わない。比べる値がその型で計算されていることもアサーションで確かめる):

- BM25: 素朴な二重ループの参照実装と一致する。passage が固定長のとき、$b$ を変えても結果が変わらない。
- 評価指標: 手で計算できる小さな例で Recall@k・順位・nDCG@10 が一致する。ランダムな順位の期待値がモンテカルロ法と一致する。
- 隠れ状態: `attention="causal"`のとき、既存の`compute_final_hidden_states()`と bit 単位で一致し、`lm_head`を通すと`GPTLanguageModel.forward()`の logits と一致する。
  因果マスクのもとで位置 $t$ の出力が後ろのトークンに依存しない。双方向では依存する。
- プーリング・切り詰め: 平均・終端位置が定義どおり。切り詰めた埋め込みが先頭 $m$ 次元を再正規化したものと一致する。
- 損失: InfoNCE が閉形式の値と一致する。020 の`softmax_contrastive_loss()`(温度を固定した対称な損失)と一致する。Matryoshka Representation Learning の損失が、1 つの次元のとき InfoNCE と一致し、
  各次元の項の勾配が先頭 $m$ 次元にだけ流れる。
- 幾何: 異方性・alignment・uniformity が素朴な二重ループの値と一致する。
- 記事の分割と符号化: 候補の外の記事を使わない、3 つの部分が互いに素で和集合が候補、検索に使える記事だけが検証用・評価用、シードで決定的、使う記事だけを符号化する。
- 学習の組: 1 つのバッチに同じ記事が入らない、区間が記事に収まり重ならない、シードで決定的、query 側が先の割合が約 0.5。
- 学習ループ: 1 ステップ目の損失が、同じ組から手で計算した損失と一致する。学習率 0 の学習(更新がない)の訓練損失が、同じ組での学習前のモデルの損失(`compute_pair_losses()`)と一致し、
  `compute_pair_losses()`はモデルを変更しない。

**不変条件**: 検証用・評価用・学習用の記事が互いに素(5.3 節)、評価用の索引と query が決定的(5.3 節)、評価の値が符号化のバッチの大きさに依らない。

**既存モジュールの後方互換性**: 本トピックの`src/`は新しいファイルの追加だけである。`scripts/`では、コーパスのアーティファクトの運用スクリプト`scripts/promote_canonical_corpora.py`の英語のスペックに、記事の境界(`article_offsets`)の追加の対象にするフラグを足した。変更前のコミット(`c88c43d`)と比べて、`src/`・`scripts/`の既存のファイルに削除がなく、変更がそのスクリプトの 1 件(追加の行と説明の 1 行の書き換えだけ)であることを`git diff`で確かめる。


```python
_t0_checks = time.time()


def max_abs_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().cpu().double() - b.detach().cpu().double()).abs().max())  # MPS は FP64 を扱えないので CPU で計算する


EPSILON_FP16, EPSILON_FP32, EPSILON_FP64 = (float(torch.finfo(t).eps) for t in (torch.float16, torch.float32, torch.float64))

# --- 008 の構成と重み(Hub の main) ---
from huggingface_hub import hf_hub_download  # noqa: E402

REFERENCE_CONFIG = json.loads(Path(hf_hub_download(REFERENCE_MODEL_REPO_ID, "config.json")).read_text(encoding="utf-8"))
REFERENCE_STATE_PATH = hf_hub_download(REFERENCE_MODEL_REPO_ID, "model_state.pt")
assert REFERENCE_CONFIG["vocabulary_size"] == tokenizer.vocab_size == 8192 and REFERENCE_CONFIG["d_model"] == EMBEDDING_DIMENSION
assert REFERENCE_CONFIG["sequence_length"] >= PASSAGE_LENGTH and REFERENCE_CONFIG["tie_embeddings"]
REFERENCE_STATE = {k: v.clone() for k, v in torch.load(REFERENCE_STATE_PATH, map_location="cpu").items()}


def build_gpt(config: dict) -> GPTLanguageModel:
    # 006・008 と同じ構成(RoPE・RMSNorm・SwiGLU・正規化前置・重み共有)
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


def build_model(condition: str, seed_index: int) -> TextEmbeddingModel:
    spec = CONDITIONS[condition]
    torch.manual_seed(INIT_SEED_BASE + seed_index)
    backbone = build_gpt(REFERENCE_CONFIG)
    if spec["init"] == "pretrained":
        backbone.load_state_dict(REFERENCE_STATE)
    return TextEmbeddingModel(backbone, pooling=spec["pooling"], attention=spec["attention"])


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for _, value in sorted(model.state_dict().items()):
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


# --- BM25: 素朴な参照実装との一致 ---
_rng = np.random.default_rng(0)
_documents = [_rng.integers(0, 12, size=int(_rng.integers(5, 20))).tolist() for _ in range(30)]
_bm25 = BM25Index(_documents, 12, k1=1.2, b=0.75)
_average_length = float(np.mean([len(d) for d in _documents]))


def naive_bm25(query: list[int]) -> np.ndarray:
    scores = np.zeros(len(_documents))
    for term in set(query):
        n_t = sum(1 for d in _documents if term in d)
        idf = math.log(1 + (len(_documents) - n_t + 0.5) / (n_t + 0.5))
        for i, d in enumerate(_documents):
            f = d.count(term)
            if f:
                scores[i] += idf * f * (1.2 + 1) / (f + 1.2 * (1 - 0.75 + 0.75 * len(d) / _average_length))
    return scores


for _query in ([1, 2, 2, 3], [11], [0, 5, 7, 7, 7]):
    _a, _b = _bm25.score(_query), naive_bm25(_query)
    assert _a.dtype == np.float64
    # 許容の誤差 = 和の項数(query の語数 x 転置リストの長さの上限)x FP64 の丸めの単位 x 値の大きさ
    _tolerance = len(set(_query)) * EPSILON_FP64 * float(np.abs(_b).max()) * 16
    assert float(np.abs(_a - _b).max()) <= _tolerance, (_query, float(np.abs(_a - _b).max()), _tolerance)
_fixed = _rng.integers(0, 50, size=(40, 16))
_s0 = BM25Index(_fixed, 50, b=0.0).score_batch(_fixed[:5])
_s1 = BM25Index(_fixed, 50, b=1.0).score_batch(_fixed[:5])
assert _s0.dtype == np.float32 and float(np.abs(_s0 - _s1).max()) <= 16 * EPSILON_FP32 * float(np.abs(_s0).max())
print("BM25: 素朴な二重ループの参照実装と一致(FP64、許容は和の項数 x 丸めの単位 x 値の大きさの 16 倍)、固定長の passage では b を変えても結果が同じ: OK")

# --- 評価指標: 手で計算できる例 ---
_scores = torch.tensor([[0.9, 0.1, 0.5, 0.3, 0.8], [0.2, 0.7, 0.6, 0.9, 0.1]])
_article_of_passage = np.array([0, 0, 1, 1, 0])
_result = rank_candidates(_scores, np.array([0, 1]), _article_of_passage, np.array([0, 2]))
assert _result.rank.tolist() == [1, 1]  # query 0: 候補 1〜4、正解 {1, 4}、最上位の正解は 4(0.8)。query 1: 正解 {3}(0.9)
_result2 = rank_candidates(torch.tensor([[0.9, 0.1, 0.5, 0.3, 0.2]]), np.array([0]), _article_of_passage, np.array([0]))
assert _result2.rank.tolist() == [3] and _result2.hits(2).tolist() == [False] and _result2.hits(3).tolist() == [True]
_expected_ndcg = (1 / math.log2(4) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))  # 正解が順位 3 と 4、正解の数 2
assert abs(float(_result2.ndcg_at_10[0]) - _expected_ndcg) <= 16 * EPSILON_FP64 * _expected_ndcg
assert abs(_result2.mean_reciprocal_rank() - 1 / 3) <= 16 * EPSILON_FP64
_tie = rank_candidates(torch.tensor([[0.0, 0.5, 0.5, 0.5, 0.1]]), np.array([0]), np.array([0, 0, 1, 1, 0]), np.array([0]))
assert _tie.rank.tolist() == [3]  # 正解 {1, 4}: 最上位の正解は 1(0.5)。同点の正解でない候補 2・3 は、正解に不利に数える
# 積の形で計算するので、許容の誤差は項数(10)x FP64 の丸めの単位 x 値の大きさ(1 以下)
assert abs(expected_random_recall(np.array([1]), 20, 10) - 0.5) <= 10 * EPSILON_FP64 and expected_random_recall(np.array([2]), 5, 10) == 1.0
_g, _m, _k = 3, 40, 10
_monte_carlo = float(np.mean([(np.random.default_rng(i).permutation(_m)[:_k] < _g).any() for i in range(20000)]))
assert abs(_monte_carlo - expected_random_recall(np.array([_g]), _m, _k)) < 4 * math.sqrt(0.25 / 20000)  # 二項分布の標準誤差(最大 0.5 / 試行回数の平方根)の 4 倍以内
print("評価指標: 手計算の順位・nDCG@10・MRR・同点の扱い(正解に不利)が一致、ランダムな順位の期待値がモンテカルロ法と一致: OK")

# --- 隠れ状態・プーリング・切り詰め(008 の重み、CPU、FP32) ---
_backbone = build_gpt(REFERENCE_CONFIG)
_backbone.load_state_dict(REFERENCE_STATE)
_tokens = torch.randint(0, 8192, (3, 40), generator=torch.Generator().manual_seed(1))
with torch.no_grad():
    _causal = compute_hidden_states(_backbone, _tokens, "causal")
    assert _causal.dtype == torch.float32
    assert torch.equal(_causal, compute_final_hidden_states(_backbone, _tokens)), "既存の compute_final_hidden_states() と bit 単位で一致しない"
    assert torch.equal(_backbone.lm_head(_causal), _backbone(_tokens)), "lm_head を通した logits が forward() と一致しない"
    _changed = _tokens.clone()
    _changed[:, 30:] = torch.randint(0, 8192, (3, 10), generator=torch.Generator().manual_seed(2))
    _c1, _c2 = _causal, compute_hidden_states(_backbone, _changed, "causal")
    _d1, _d2 = compute_hidden_states(_backbone, _tokens, "bidirectional"), compute_hidden_states(_backbone, _changed, "bidirectional")
    assert torch.equal(_c1[:, :30], _c2[:, :30]) and not torch.equal(_c1[:, 30:], _c2[:, 30:]), "因果マスクで未来に依存している"
    assert not torch.equal(_d1[:, :30], _d2[:, :30]), "双方向なのに後ろのトークンに依存しない"
    _model = TextEmbeddingModel(_backbone, "mean", "causal")
    _model_last = TextEmbeddingModel(_backbone, "last", "causal")
    assert torch.equal(_model.pooled(_tokens), _causal.mean(dim=1)) and torch.equal(_model_last.pooled(_tokens), _causal[:, -1, :])
    _embedding = _model(_tokens)
    _norm_tolerance = 16 * EPSILON_FP32
    assert _embedding.dtype == torch.float32 and float((_embedding.norm(dim=-1) - 1).abs().max()) <= _norm_tolerance
    assert torch.equal(_model(_tokens, dimension=EMBEDDING_DIMENSION), _embedding)
    _pooled = _model.pooled(_tokens)
    _truncated = _model(_tokens, dimension=16)
    _expected = _pooled[:, :16] / _pooled[:, :16].norm(dim=-1, keepdim=True)
    assert float((_truncated - _expected).abs().max()) <= 16 * EPSILON_FP32 * float(_expected.abs().max())
    assert float((_truncated.norm(dim=-1) - 1).abs().max()) <= _norm_tolerance
print(
    "隠れ状態: causal は既存の compute_final_hidden_states() と bit 単位で一致、lm_head を通すと forward() と一致、因果マスクで位置 0〜29 の出力は後ろのトークンに依存しない、"
    f"双方向では依存する(最大差 {max_abs_difference(_d1, _c1):.2f})。平均・終端位置のプーリング、切り詰め後の再正規化(ノルムの誤差 <= {_norm_tolerance:.1e}): OK"
)

# --- 損失 ---
_u = torch.nn.functional.normalize(torch.randn(8, 16, generator=torch.Generator().manual_seed(3)), dim=-1)
_v = torch.nn.functional.normalize(torch.randn(8, 16, generator=torch.Generator().manual_seed(4)), dim=-1)
_symmetric = info_nce_loss(_u, _v, TEMPERATURE, symmetric=True)
_reference_loss = softmax_contrastive_loss(_u, _v, torch.tensor(math.log(1 / TEMPERATURE)))
assert _symmetric.dtype == _reference_loss.dtype == torch.float32
assert abs(float(_symmetric - _reference_loss)) <= 16 * EPSILON_FP32 * float(_reference_loss), (float(_symmetric), float(_reference_loss))
_u64, _v64 = _u[:2].double(), _v[:2].double()
_logits = (_u64 @ _v64.t()) / TEMPERATURE
_closed_form = -0.5 * (
    math.log(math.exp(float(_logits[0, 0])) / (math.exp(float(_logits[0, 0])) + math.exp(float(_logits[0, 1]))))
    + math.log(math.exp(float(_logits[1, 1])) / (math.exp(float(_logits[1, 0])) + math.exp(float(_logits[1, 1]))))
)
_loss64 = info_nce_loss(_u64, _v64, TEMPERATURE)
assert _loss64.dtype == torch.float64 and abs(float(_loss64) - _closed_form) <= 16 * EPSILON_FP64 * abs(_closed_form) * 20  # exp・log の合成(約 20 個の演算)
assert float(info_nce_loss(_u, _u, 0.01)) < 1e-6
_pooled_queries = torch.randn(8, 20, generator=torch.Generator().manual_seed(5), dtype=torch.float64, requires_grad=True)
_pooled_passages = torch.randn(8, 20, generator=torch.Generator().manual_seed(6), dtype=torch.float64)
_single, _ = matryoshka_representation_loss(_pooled_queries, _pooled_passages, (20,), TEMPERATURE)
assert torch.equal(_single, info_nce_loss(truncate_and_normalize(_pooled_queries), truncate_and_normalize(_pooled_passages), TEMPERATURE))
_total, _per_dimension = matryoshka_representation_loss(_pooled_queries, _pooled_passages, (4, 8, 16), TEMPERATURE)
assert abs(float(_total.detach()) - float(_per_dimension.mean())) <= 16 * EPSILON_FP64 * float(_total.detach())
for _index, _dimension in enumerate((4, 8, 16)):  # 各次元の項の勾配は、先頭 _dimension 次元にだけ流れる
    _term = info_nce_loss(truncate_and_normalize(_pooled_queries, _dimension), truncate_and_normalize(_pooled_passages, _dimension), TEMPERATURE)
    (_gradient,) = torch.autograd.grad(_term, _pooled_queries)
    assert float(_gradient[:, _dimension:].abs().max()) == 0.0 and float(_gradient[:, :_dimension].abs().max()) > 0.0
    assert abs(float(_term.detach()) - float(_per_dimension[_index])) <= 16 * EPSILON_FP64 * float(_term.detach())
print(
    "損失: InfoNCE が閉形式の値と一致(FP64)、020 の softmax_contrastive_loss()(温度を固定した対称な損失)と一致(FP32、許容は丸めの単位 x 値の 16 倍)、"
    "Matryoshka Representation Learning の損失は 1 つの次元のとき InfoNCE と bit 単位で一致、次元の項の平均に一致し、各項の勾配は先頭 m 次元にだけ流れる: OK"
)

# --- 幾何 ---
_e = torch.nn.functional.normalize(torch.randn(50, 8, generator=torch.Generator().manual_seed(7)), dim=-1).double()
_pairs = [(i, j) for i in range(50) for j in range(50) if i != j]
_manual_cosine = float(np.mean([float(_e[i] @ _e[j]) for i, j in _pairs]))
assert abs(_manual_cosine - mean_pairwise_cosine(_e)) <= len(_pairs) * EPSILON_FP64
assert abs(alignment(_e, _e)) == 0.0
_manual_uniformity = math.log(np.mean([math.exp(-2 * float((_e[i] - _e[j]).pow(2).sum())) for i in range(50) for j in range(i + 1, 50)]))
assert abs(_manual_uniformity - uniformity(_e)) <= len(_pairs) * EPSILON_FP64
print("幾何: 異方性・alignment・uniformity が素朴な二重ループの値と一致(FP64、許容は和の項数 x 丸めの単位): OK")

# --- 記事の分割と符号化 ---
_counts = np.array([300, 50, 400, 500, 90, 1000, 260, 700, 128, 256, 800, 33])
_candidates = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
_split = split_articles(_counts, 256, _candidates, 2, 3, seed=7)
assert set(_split.train) | set(_split.validation) | set(_split.evaluation) == set(_candidates), "候補の外の記事が入っている、または候補が漏れている"
assert not (set(_split.train) & set(_split.validation)) and not (set(_split.train) & set(_split.evaluation)) and not (set(_split.validation) & set(_split.evaluation))
assert len(_split.validation) == 2 and len(_split.evaluation) == 3
assert all(_counts[a] >= 256 for a in list(_split.validation) + list(_split.evaluation)), "検索に使えない短い記事が検証用・評価用に入っている"
assert split_articles(_counts, 256, _candidates, 2, 3, seed=7).digest() == _split.digest() != split_articles(_counts, 256, _candidates, 2, 3, seed=8).digest()
assert split_articles(_counts, 256, _candidates[::-1], 2, 3, seed=7).digest() == _split.digest(), "候補の並べ方に依存している"
assert not ({0, 1} & (set(_split.train) | set(_split.validation) | set(_split.evaluation))), "候補でない記事が入っている"


class _CharacterTokenizer:  # 文字の符号位置を 7 で割った余りを返す、確認用のトークナイザ
    def encode(self, text: str) -> list[int]:
        return [ord(ch) % 7 for ch in text]


_text, _spans = "abc\ndefgh\nij", [{"start": 0, "end": 3}, {"start": 4, "end": 9}, {"start": 10, "end": 12}]
_encoded = encode_articles(_CharacterTokenizer(), _text, _spans, [0, 2])
assert _encoded[1] is None and _encoded[0].dtype == _encoded[2].dtype == np.int32
assert _encoded[0].tolist() == _CharacterTokenizer().encode("abc") and _encoded[2].tolist() == _CharacterTokenizer().encode("ij")
_flat_check, _starts_check = concatenate_articles(_encoded, [0, 2])
assert _flat_check.dtype == np.int64 and _flat_check.tolist() == _encoded[0].tolist() + _encoded[2].tolist() and _starts_check.tolist() == [0, 3]
print("分割と符号化: 候補の外の記事を使わない、3 つの部分が互いに素で和集合が候補、検索に使える記事だけが検証用・評価用、シードで決定的で候補の並べ方に依らない、使う記事だけを符号化する: OK")

# --- 学習の組 ---
_schedule = sample_pair_schedule(TRAIN_TOKEN_COUNTS, SAMPLING_WEIGHTS, QUERY_LENGTH, PASSAGE_LENGTH, 200, BATCH_SIZE, seed=5)
_articles, _query_starts, _passage_starts = _schedule["article"], _schedule["query_start"], _schedule["passage_start"]
_lengths = TRAIN_TOKEN_COUNTS[_articles]
assert all(len(set(row.tolist())) == BATCH_SIZE for row in _articles), "1 つのバッチに同じ記事が入っている"
assert (SAMPLING_WEIGHTS[_articles] > 0).all()
assert (_query_starts >= 0).all() and (_passage_starts >= 0).all() and (_query_starts + QUERY_LENGTH <= _lengths).all() and (_passage_starts + PASSAGE_LENGTH <= _lengths).all(), "区間が記事に収まっていない"
assert ((_query_starts + QUERY_LENGTH <= _passage_starts) | (_passage_starts + PASSAGE_LENGTH <= _query_starts)).all(), "query 側と passage 側の区間が重なっている"
_again = sample_pair_schedule(TRAIN_TOKEN_COUNTS, SAMPLING_WEIGHTS, QUERY_LENGTH, PASSAGE_LENGTH, 200, BATCH_SIZE, seed=5)
assert all(np.array_equal(_schedule[k], _again[k]) for k in _schedule), "学習の組がシードで決定的でない"
_other = sample_pair_schedule(TRAIN_TOKEN_COUNTS, SAMPLING_WEIGHTS, QUERY_LENGTH, PASSAGE_LENGTH, 200, BATCH_SIZE, seed=6)
assert not np.array_equal(_schedule["article"], _other["article"]), "異なるシードで同じ組(確認の検出力がない)"
_query_first = float(np.mean(_query_starts < _passage_starts))
assert abs(_query_first - 0.5) < 4 * math.sqrt(0.25 / _query_starts.size), _query_first  # 二項分布の標準誤差の 4 倍以内
print(
    f"学習の組: 200 ステップ x {BATCH_SIZE} 組で、1 つのバッチに同じ記事なし、区間は記事に収まり重ならない、シードで決定的、query 側が先の割合 {_query_first:.3f}: OK"
)

# --- 学習ループ: 1 ステップ目の損失が、同じ組から手で計算した損失と一致する(CPU、FP32) ---
_small_schedule = {k: v[:2, :16] for k, v in _schedule.items()}
_stream_cpu = TRAIN_STREAM.cpu()
for _loss_kind in ("infonce", "matryoshka"):
    torch.manual_seed(11)
    _tiny = TextEmbeddingModel(build_gpt(REFERENCE_CONFIG), "mean", "causal")
    _tiny.backbone.load_state_dict(REFERENCE_STATE)
    _dimensions = MATRYOSHKA_DIMENSIONS if _loss_kind == "matryoshka" else None
    _starts = np.asarray(TRAIN_STARTS)[_small_schedule["article"][0]]
    _q = torch.stack([_stream_cpu[s + _small_schedule["query_start"][0][i] : s + _small_schedule["query_start"][0][i] + QUERY_LENGTH] for i, s in enumerate(_starts)])
    _p = torch.stack([_stream_cpu[s + _small_schedule["passage_start"][0][i] : s + _small_schedule["passage_start"][0][i] + PASSAGE_LENGTH] for i, s in enumerate(_starts)])
    with torch.no_grad():
        _pooled_queries, _pooled_passages = _tiny.pooled(_q), _tiny.pooled(_p)
        if _dimensions:
            _manual, _ = matryoshka_representation_loss(_pooled_queries, _pooled_passages, _dimensions, TEMPERATURE)
        else:
            _manual = info_nce_loss(truncate_and_normalize(_pooled_queries), truncate_and_normalize(_pooled_passages), TEMPERATURE)
    _history = train_text_embedding(
        _tiny, _stream_cpu, TRAIN_STARTS, _small_schedule, QUERY_LENGTH, PASSAGE_LENGTH, 1e-4, 1, 1e-6, WEIGHT_DECAY,
        GRADIENT_CLIP_THRESHOLD, TEMPERATURE, matryoshka_dimensions=_dimensions,
    )
    assert math.isfinite(_history["loss"][1]) and abs(_history["loss"][0] - float(_manual)) <= 16 * EPSILON_FP32 * float(_manual), (_loss_kind, _history["loss"][0], float(_manual))
    assert ("dimension_losses" in _history) == (_loss_kind == "matryoshka")
print("学習ループ: 1 ステップ目の損失が、同じ組から手で計算した損失と一致(通常の InfoNCE と Matryoshka Representation Learning の両方): OK")

# --- 同じ組での損失の測定: 学習率 0 の学習(更新がない)の訓練損失と一致し、モデルを変更しない(CPU、FP32) ---
for _loss_kind in ("infonce", "matryoshka"):
    torch.manual_seed(11)
    _tiny = TextEmbeddingModel(build_gpt(REFERENCE_CONFIG), "mean", "causal")
    _tiny.backbone.load_state_dict(REFERENCE_STATE)
    _dimensions = MATRYOSHKA_DIMENSIONS if _loss_kind == "matryoshka" else None
    _before_hash = state_hash(_tiny)
    _measured = compute_pair_losses(_tiny, _stream_cpu, TRAIN_STARTS, _small_schedule, range(2), QUERY_LENGTH, PASSAGE_LENGTH, TEMPERATURE, matryoshka_dimensions=_dimensions)
    assert state_hash(_tiny) == _before_hash and _tiny.training, "compute_pair_losses() がモデルまたはモードを変更した"
    _history = train_text_embedding(
        _tiny, _stream_cpu, TRAIN_STARTS, _small_schedule, QUERY_LENGTH, PASSAGE_LENGTH, 0.0, 1, 0.0, WEIGHT_DECAY,
        GRADIENT_CLIP_THRESHOLD, TEMPERATURE, matryoshka_dimensions=_dimensions,
    )
    assert state_hash(_tiny) == _before_hash, "学習率 0 の学習でモデルが変化した"
    assert _measured.shape == (2,) and _measured.dtype == np.float64
    assert np.allclose(_history["loss"], _measured, rtol=16 * EPSILON_FP32, atol=0.0), (_loss_kind, _history["loss"], _measured)
print("同じ組での損失の測定: 学習率 0 の学習の訓練損失と一致し(通常の InfoNCE と Matryoshka Representation Learning の両方)、モデルとモードを変更しない: OK")

# --- 符号化が、バッチの大きさに依らない(実行するデバイス、FP32) ---
_check_tokens = torch.from_numpy(EVALUATION_SET.passages[:300])
_model_device = build_model("C2", 0).to(device)
_wide = encode_tokens(_model_device, _check_tokens, 256)["by_dimension"][EMBEDDING_DIMENSION]
_repeat = max_abs_difference(_wide, encode_tokens(_model_device, _check_tokens, 256)["by_dimension"][EMBEDDING_DIMENSION])
_narrow = max_abs_difference(_wide, encode_tokens(_model_device, _check_tokens, 7)["by_dimension"][EMBEDDING_DIMENSION])
_unit = EPSILON_FP32 * float(_wide.abs().max())
assert _wide.dtype == torch.float32
BATCH_INDEPENDENCE_TOLERANCE = 16 * max(_repeat, _unit)
assert _narrow <= BATCH_INDEPENDENCE_TOLERANCE, (_narrow, BATCH_INDEPENDENCE_TOLERANCE)
print(
    f"符号化のバッチの大きさへの依存: 256 個ずつと 7 個ずつの最大差 {_narrow:.2e}(同じ経路の 2 回の差 {_repeat:.2e}、丸めの単位 x 値の大きさ {_unit:.2e}、"
    f"許容 {BATCH_INDEPENDENCE_TOLERANCE:.2e}): OK"
)
del _model_device, _wide

# --- 同じシードの条件どうしの対応: 学習の組は条件によらない / 008 の重みから始める条件の初期値は同じ ---
_models = {c: build_model(c, 0) for c in CONDITIONS}
_hashes = {c: state_hash(m) for c, m in _models.items()}
assert len({_hashes[c] for c in ("C1", "C2", "C3", "C5")}) == 1, "008 の重みから始める条件の初期値が一致しない"
assert _hashes["C4"] != _hashes["C2"] and state_hash(build_model("C4", 1)) != _hashes["C4"] and state_hash(build_model("C4", 0)) == _hashes["C4"]
assert state_hash(build_model("C2", 1)) == _hashes["C2"], "008 の重みから始める条件の初期値がシードに依存している"
_parameters = {c: sum(p.numel() for p in m.parameters()) for c, m in _models.items()}
assert len(set(_parameters.values())) == 1, "条件間でパラメータ数が異なる"
TOTAL_PARAMETERS = _parameters["C2"]
print(
    f"初期値: C1・C2・C3・C5 は 008 の重みで同一(ハッシュ {_hashes['C2']})、C4 はシードに依存するランダム初期化(シード 0 のハッシュ {_hashes['C4']})。"
    f"総パラメータ数は全条件で同じ({TOTAL_PARAMETERS:,}): OK"
)
del _models

# --- 既存モジュールの後方互換性: 変更前のコミットと比べて、src/・scripts/ の既存のファイルに削除がなく、変更は許可した 1 件の追加だけ ---
_REFERENCE_COMMIT = "c88c43d"  # 024 の変更を始める前の最後のコミット
_ALLOWED_CHANGED_FILE = "scripts/promote_canonical_corpora.py"  # en スペックに article_offsets を追加する対象のフラグを足した(コーパスのアーティファクトの運用)
_diff = subprocess.run(
    ["git", "diff", "--name-only", "--diff-filter=MD", _REFERENCE_COMMIT, "--", "src", "scripts"], capture_output=True, text=True
)
assert _diff.returncode == 0, _diff.stderr
assert set(_diff.stdout.split()) <= {_ALLOWED_CHANGED_FILE}, f"許可していない既存のファイルが変更されている: {_diff.stdout}"
_deleted = subprocess.run(
    ["git", "diff", "--name-only", "--diff-filter=D", _REFERENCE_COMMIT, "--", "src", "scripts"], capture_output=True, text=True
).stdout.strip()
assert _deleted == "", f"既存のファイルが削除されている: {_deleted}"
_script_diff = subprocess.run(
    ["git", "diff", "-U0", _REFERENCE_COMMIT, "--", _ALLOWED_CHANGED_FILE], capture_output=True, text=True
).stdout.splitlines()
_removed_lines = [line for line in _script_diff if line.startswith("-") and not line.startswith("---")]
assert len(_removed_lines) <= 1 and all("現在は``en_006``のみ" in line for line in _removed_lines), "許可した変更(フラグの追加と説明の 1 行)以外の削除がある"
_existing = subprocess.run(["git", "ls-tree", "-r", "--name-only", _REFERENCE_COMMIT, "src", "scripts"], capture_output=True, text=True).stdout.split()
print(
    f"既存モジュールの後方互換性: 参照コミット {_REFERENCE_COMMIT} の src/・scripts/ の既存のファイル {len(_existing)} 個に、削除がなく、"
    f"変更は {_ALLOWED_CHANGED_FILE} の 1 件のみ(追加の行と説明の 1 行の書き換えだけ): OK"
)
CHECK_SECONDS = time.time() - _t0_checks
print(f"単体テストと不変条件の確認 {CHECK_SECONDS:.1f} 秒")
```


    config.json:   0%|          | 0.00/305 [00:00<?, ?B/s]



    model_state.pt: reconstructing file:   0%|          |  0.00B / 21.0MB            



    model_state.pt: downloading bytes:           |  0.00B            


    BM25: 素朴な二重ループの参照実装と一致(FP64、許容は和の項数 x 丸めの単位 x 値の大きさの 16 倍)、固定長の passage では b を変えても結果が同じ: OK
    評価指標: 手計算の順位・nDCG@10・MRR・同点の扱い(正解に不利)が一致、ランダムな順位の期待値がモンテカルロ法と一致: OK
    隠れ状態: causal は既存の compute_final_hidden_states() と bit 単位で一致、lm_head を通すと forward() と一致、因果マスクで位置 0〜29 の出力は後ろのトークンに依存しない、双方向では依存する(最大差 5.79)。平均・終端位置のプーリング、切り詰め後の再正規化(ノルムの誤差 <= 1.9e-06): OK
    損失: InfoNCE が閉形式の値と一致(FP64)、020 の softmax_contrastive_loss()(温度を固定した対称な損失)と一致(FP32、許容は丸めの単位 x 値の 16 倍)、Matryoshka Representation Learning の損失は 1 つの次元のとき InfoNCE と bit 単位で一致、次元の項の平均に一致し、各項の勾配は先頭 m 次元にだけ流れる: OK
    幾何: 異方性・alignment・uniformity が素朴な二重ループの値と一致(FP64、許容は和の項数 x 丸めの単位): OK
    分割と符号化: 候補の外の記事を使わない、3 つの部分が互いに素で和集合が候補、検索に使える記事だけが検証用・評価用、シードで決定的で候補の並べ方に依らない、使う記事だけを符号化する: OK
    学習の組: 200 ステップ x 64 組で、1 つのバッチに同じ記事なし、区間は記事に収まり重ならない、シードで決定的、query 側が先の割合 0.505: OK
    学習ループ: 1 ステップ目の損失が、同じ組から手で計算した損失と一致(通常の InfoNCE と Matryoshka Representation Learning の両方): OK
    同じ組での損失の測定: 学習率 0 の学習の訓練損失と一致し(通常の InfoNCE と Matryoshka Representation Learning の両方)、モデルとモードを変更しない: OK
    符号化のバッチの大きさへの依存: 256 個ずつと 7 個ずつの最大差 6.71e-08(同じ経路の 2 回の差 0.00e+00、丸めの単位 x 値の大きさ 3.01e-08、許容 4.81e-07): OK
    初期値: C1・C2・C3・C5 は 008 の重みで同一(ハッシュ 9e2cba765a2ae1e2)、C4 はシードに依存するランダム初期化(シード 0 のハッシュ 3d130fa4012fdb09)。総パラメータ数は全条件で同じ(5,246,208): OK
    既存モジュールの後方互換性: 参照コミット c88c43d の src/・scripts/ の既存のファイル 65 個に、削除がなく、変更は scripts/promote_canonical_corpora.py の 1 件のみ(追加の行と説明の 1 行の書き換えだけ): OK
    単体テストと不変条件の確認 26.8 秒


### 5.5 学習と評価のヘルパー

- `evaluate_set()`: 評価用(または検証用・学習用の部分)の索引と query を FP32 で符号化し、全件の内積で順位をつける。切り詰める次元ごとの結果(`RankingResult`)を返す。
  指定すれば、プーリング方式の代わりに **指定した位置の出力** を埋め込みにした結果(実験 A の診断量)も返す。query と passage は長さが違うので、位置は系列の長さに対する割合
  (`POSITION_FRACTIONS`)で決める。
- `train_run()`: 条件・シード・学習率・ステップ数を受け取って学習し、検証用の集合の Recall@10(最終ステップ。追跡する条件では 5.2 節の途中の評価の位置ごと)を記録する。
  本番の学習(`main=True`)では、評価用の集合の query ごとの順位(切り詰める次元ごと)、学習用の部分の Recall@10(汎化の差の診断量)、C2 では位置ごとの結果、
  観察 F 用に C2 のシード 0 の passage の埋め込み、アップロード用に C5 のシード 0 の学習後の重みも記録する。モデルは評価の直後に破棄する。


```python
SET_TENSORS = {n: (torch.from_numpy(s.queries), torch.from_numpy(s.passages)) for n, s in RETRIEVAL_SETS.items()}
POSITIONS_QUERY = [position_for(f, QUERY_LENGTH) for f in POSITION_FRACTIONS]
POSITIONS_PASSAGE = [position_for(f, PASSAGE_LENGTH) for f in POSITION_FRACTIONS]
assert len(set(POSITIONS_QUERY)) == len(POSITIONS_QUERY) and len(set(POSITIONS_PASSAGE)) == len(POSITIONS_PASSAGE)
assert POSITIONS_QUERY[-1] == QUERY_LENGTH - 1 and POSITIONS_PASSAGE[-1] == PASSAGE_LENGTH - 1  # 割合 1 は最後の位置


def evaluate_set(
    model: TextEmbeddingModel,
    set_name: str,
    dimensions: tuple[int, ...] = (EMBEDDING_DIMENSION,),
    by_position: bool = False,
    keep_embeddings: bool = False,
) -> dict:
    retrieval_set = RETRIEVAL_SETS[set_name]
    queries, passages = SET_TENSORS[set_name]
    q = encode_tokens(model, queries, EVAL_ENCODE_BATCH, dimensions, positions=POSITIONS_QUERY if by_position else None)
    p = encode_tokens(model, passages, EVAL_ENCODE_BATCH, dimensions, positions=POSITIONS_PASSAGE if by_position else None)
    result = {"by_dimension": {m: evaluate_embeddings(q["by_dimension"][m], p["by_dimension"][m], retrieval_set) for m in dimensions}}
    if by_position:
        result["by_position"] = {
            f: evaluate_embeddings(q["by_position"][tq], p["by_position"][tp], retrieval_set)
            for f, tq, tp in zip(POSITION_FRACTIONS, POSITIONS_QUERY, POSITIONS_PASSAGE, strict=True)
        }
    if keep_embeddings:
        result["passage_embeddings"] = p["by_dimension"][EMBEDDING_DIMENSION].cpu()
        result["query_embeddings"] = q["by_dimension"][EMBEDDING_DIMENSION].cpu()
    return result


SCHEDULES: dict[tuple[int, int], dict] = {}


def get_schedule(seed_index: int, num_steps: int) -> dict:
    # 学習の組は条件によらずシードとステップ数だけで決まる(同じシードの条件どうしで揃う)。同じものを使い回す
    key = (seed_index, num_steps)
    if key not in SCHEDULES:
        SCHEDULES[key] = sample_pair_schedule(
            TRAIN_TOKEN_COUNTS, SAMPLING_WEIGHTS, QUERY_LENGTH, PASSAGE_LENGTH, num_steps, BATCH_SIZE, seed=PAIR_SEED_BASE + seed_index
        )
    return SCHEDULES[key]


SAVED_STATES: dict[tuple[str, int], dict] = {}  # アップロード用(C5 のシード 0)
OBSERVATION_EMBEDDINGS: dict = {}  # 観察 F 用(C2 のシード 0)


def train_run(condition: str, seed_index: int, learning_rate: float, num_steps: int, main: bool) -> dict:
    spec = CONDITIONS[condition]
    start = time.time()
    model = build_model(condition, seed_index).to(device)
    schedule = get_schedule(seed_index, num_steps)
    tracked = main and spec["track"]
    eval_steps = eval_steps_for(num_steps) if tracked else ()
    window = final_loss_window(num_steps)
    # 学習前のモデルの、最後の区間の組での損失(P1(a)の分母。同じ組での比較なので、バッチごとの難しさが打ち消される。6.1 節)
    initial_model_losses = compute_pair_losses(
        model, TRAIN_STREAM, TRAIN_STARTS, schedule, range(num_steps - window, num_steps), QUERY_LENGTH, PASSAGE_LENGTH, TEMPERATURE,
        matryoshka_dimensions=MATRYOSHKA_DIMENSIONS if spec["loss"] == "matryoshka" else None, use_fp16_autocast=USE_FP16_AUTOCAST,
    )

    def evaluation_fn(m) -> dict:
        return {"validation_recall": evaluate_set(m, "validation")["by_dimension"][EMBEDDING_DIMENSION].recall_at(10)}

    history = train_text_embedding(
        model, TRAIN_STREAM, TRAIN_STARTS, schedule, QUERY_LENGTH, PASSAGE_LENGTH,
        peak_learning_rate=learning_rate,
        warmup_steps=warmup_steps_for(num_steps),
        min_learning_rate=learning_rate * MIN_LEARNING_RATE_RATIO,
        weight_decay=WEIGHT_DECAY,
        gradient_clip_threshold=GRADIENT_CLIP_THRESHOLD,
        temperature=TEMPERATURE,
        matryoshka_dimensions=MATRYOSHKA_DIMENSIONS if spec["loss"] == "matryoshka" else None,
        use_fp16_autocast=USE_FP16_AUTOCAST,
        init_loss_scale=INIT_LOSS_SCALE,
        loss_scale_growth_interval=LOSS_SCALE_GROWTH_INTERVAL,
        evaluation_steps=eval_steps,
        evaluation_fn=evaluation_fn,
    )
    assert len(history["loss"]) == num_steps
    train_loss = np.array(history["loss"], dtype=np.float64)
    record = {
        "condition": condition, "seed": seed_index, "learning_rate": learning_rate, "num_steps": num_steps,
        "schedule_hash": history["data_stream_hash"],
        "train_loss": train_loss.astype(np.float32),
        "initial_train_loss": float(train_loss[:window].mean()),
        "final_train_loss": float(train_loss[-window:].mean()),
        "initial_model_loss_on_final_window": float(initial_model_losses.mean()),
        "finite": bool(np.isfinite(train_loss).all()),
        "skipped_steps": int(sum(history["step_skipped"])),
        "clip_rate": float(np.mean(history["gradient_clip_triggered"])),
        "final_loss_scale": float(history["loss_scale"][-1]),
        "eval_step": list(eval_steps),
        "eval_validation_recall": [history["evaluations"][s]["validation_recall"] for s in eval_steps],
    }
    if "dimension_losses" in history:
        record["final_dimension_losses"] = history["dimension_losses"][-window:].mean(axis=0)
    validation = evaluate_set(model, "validation")["by_dimension"][EMBEDDING_DIMENSION]
    record["validation_recall"] = validation.recall_at(10)
    if main:
        keep = (condition, seed_index) == OBSERVATION_RUN
        final = evaluate_set(model, "evaluation", MATRYOSHKA_DIMENSIONS, by_position=condition == "C2", keep_embeddings=keep)
        record["hits"] = {m: r.hits(10) for m, r in final["by_dimension"].items()}
        record["rank"] = {m: r.rank for m, r in final["by_dimension"].items()}
        record["ndcg_at_10"] = final["by_dimension"][EMBEDDING_DIMENSION].ndcg_at_10
        if "by_position" in final:
            record["position_hits"] = {f: r.hits(10) for f, r in final["by_position"].items()}
        record["train_probe_recall"] = evaluate_set(model, "train_probe")["by_dimension"][EMBEDDING_DIMENSION].recall_at(10)
        if keep:
            OBSERVATION_EMBEDDINGS.update(passages=final["passage_embeddings"], queries=final["query_embeddings"])
        if (condition, seed_index) == UPLOAD_RUN:
            SAVED_STATES[UPLOAD_RUN] = {k: v.detach().cpu().clone() for k, v in model.backbone.state_dict().items()}
    record["seconds"] = time.time() - start
    del model
    empty_device_cache()
    return record


def learning_precondition(record: dict, strict: bool) -> bool:
    # P1(a)(6.1 節): 訓練損失がすべてのステップで有限。strict(008 の重みから始める条件)なら、最後の区間の訓練損失が、同じ組での学習前のモデルの損失の P1_LOSS_RATIO 倍以下
    ok = record["finite"]
    if strict:
        ok = ok and record["final_train_loss"] <= P1_LOSS_RATIO * record["initial_model_loss_on_final_window"]
    return ok


def judge(delta: float, sigma: float) -> str:
    threshold = SIGMA_MULTIPLIER * sigma
    if delta > threshold:
        return "支持"
    if delta < -threshold:
        return "反証"
    return "判定不能"


def combined_sigma(per_seed: np.ndarray, bootstrap_contrast: np.ndarray) -> dict:
    seed_variance = float(np.var(per_seed, ddof=1)) / len(per_seed)
    bootstrap_variance = float(np.var(bootstrap_contrast, ddof=1))
    return {
        "sigma": math.sqrt(seed_variance + bootstrap_variance),
        "seed_term": math.sqrt(seed_variance),
        "bootstrap_term": math.sqrt(bootstrap_variance),
    }
```

## 6. 実験 / Experiments



## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/024_text_embedding_and_retriever.ipynb
