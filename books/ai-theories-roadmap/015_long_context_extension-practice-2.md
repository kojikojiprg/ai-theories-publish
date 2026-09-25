---
title: "長文脈拡張 / Long Context Extension(実装・実験編 2/4)"
---

この記事は後編(実装・実験編 2/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/015_long_context_extension-practice-1)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/015_long_context_extension-practice-3)。

### 5.6 評価データ: 検証の部分と記事の境界

008 と同じく、Hub のコーパスのアーティファクト(`kojikojiprg/ai-theories-corpus-en-pretraining`、356 記事)を文字列の段階で
分割し、末尾の 5% を検証の部分とする(訓練に使った部分は評価に使わない)。

**記事の境界**: アーティファクトの`corpus.txt`は記事を改行 1 つで連結したもので、記事の中にも改行があるため、
`corpus.txt`だけからは境界を復元できない(下のセルで、記事の中の改行の数を印字して確かめる)。境界は
`locate_wikipedia_article_spans()`で次のいずれかから得て、どちらを使ったかを印字する。

- **Hub の`metadata.json`の`article_offsets`**(あれば優先する): `scripts/promote_canonical_corpora.py`が、マニフェストの
  全記事を個別に取得して改行 1 つで連結した結果が`corpus.txt`と文字単位で完全に一致することを確かめてから書き込んだ、
  記事ごとの文字位置。記事数がマニフェストと一致し、オフセットが単調増加で重ならず、区切りが改行 1 つであることを
  読み込み時に確かめる。Colab での実行は Wikipedia API に依存しない。
- **Wikipedia API**(`article_offsets`がない場合): マニフェストの末尾の記事から 1 つずつ取得し、**その記事がコーパスの
  対応する位置の文字列と完全に一致すること** を確かめながら、検証の部分の先頭に達するまで遡る。

2 つの方法で評価窓の集合(ハッシュ)が完全に一致することを確かめる。ローカルでは、`article_offsets`がまだ Hub にない
間はスクリプトの dry-run が書き出した`metadata.json`と、Hub にある場合は Wikipedia API と照合する。Colab では、
API への依存をなくすため、この照合を省略する。

**評価窓**: 検証の部分を記事ごとの区間(先頭の区間は記事の途中から始まる)に分け、区間ごとに別々に符号化する。各区間の
先頭から、長さ $8L$(最大の評価長)の重ならない窓を切り出し、端数は捨てる。**したがってどの窓も 1 つの記事の中に収まる。**
本番ではすべての窓を使う。スモークテストでは、評価窓を 2 個以上持つ記事を先頭から 3 本選び、それぞれ先頭の 2 個の窓を使う
(記事を単位とするブートストラップの経路を、同じ記事に複数の窓があるクラスタを含めて実行するため。2 本以上の記事から選ばれ、
2 個以上の窓を持つ記事を含むことをアサーションで確かめる)。
評価長 $2L, 4L$ では、同じ窓の先頭の $2L, 4L$ トークンを使う(全水準で同じテキストの対応がとれる)。長さ $L$ の評価
(実験 D)では、各窓を 8 個の長さ $L$ の区間に分けて、それぞれ独立に評価する。

**bits-per-byte の分母**: 予測対象のトークンに対応する UTF-8 バイト数。トークナイザはバイトレベル BPE なので、各トークンの
バイト数は語彙の記号の長さで決まる。区間のトークンのバイト数の合計が区間の UTF-8 バイト数に等しく、復号が区間の文字列に
一致すること(可逆性)をアサーションで確かめる。位置 $j$ のトークン(窓の中の 0 始まりの番号)は、位置 $j-1$ の Query が
位置 $0, \dots, j-1$ の Key を見て予測する。位置 0 のトークンは予測しない。「位置の範囲 $[a, c)$」は、予測対象の
トークンの位置 $j$ の範囲を指す($[0, L)$ は $j = 1, \dots, L-1$)。


```python
_t0 = time.time()
corpus_text, corpus_metadata = load_wikipedia_corpus_with_fallback(
    "en", CORPUS_REPO_ID, WIKIPEDIA_CACHE_DIR, manifest_path=MANIFEST_PATH, return_metadata=True
)
assert len(corpus_text.encode("utf-8")) == corpus_metadata["raw_bytes"], (
    "コーパスの取得が破損している"
)
assert corpus_metadata["validation_ratio"] in (None, VALIDATION_RATIO)
train_text, validation_text = split_train_val_text(corpus_text, VALIDATION_RATIO)
VALIDATION_START = len(train_text)
print(
    f"コーパス: {len(corpus_text):,} 文字(取得元 {corpus_metadata['source']}、{time.time() - _t0:.1f}s)、"
    f"検証の部分: {len(validation_text):,} 文字(文字位置 {VALIDATION_START:,} から)"
)

# --- 記事の境界(Hub の metadata.json の article_offsets、なければ Wikipedia API) ---
_t0 = time.time()
article_spans, ARTICLE_SPANS_SOURCE = locate_wikipedia_article_spans(
    corpus_text,
    "en",
    WIKIPEDIA_CACHE_DIR,
    MANIFEST_PATH,
    repo_id=CORPUS_REPO_ID,
    start_position=VALIDATION_START,
)
ARTICLE_FETCH_SECONDS = time.time() - _t0
_source_label = {
    "metadata": "Hub の metadata.json の article_offsets",
    "wikipedia_api": "Wikipedia API から取得してコーパスと照合",
}[ARTICLE_SPANS_SOURCE]
print(f"記事の境界の取得元: {ARTICLE_SPANS_SOURCE}({_source_label}、{ARTICLE_FETCH_SECONDS:.1f}s)")
print(
    f"検証の部分にかかる記事: {len(article_spans)} 件(マニフェストの番号 "
    f"{article_spans[0]['manifest_index']}〜{article_spans[-1]['manifest_index']})"
)
_newlines_inside = sum(corpus_text.count("\n", sp["start"], sp["end"]) for sp in article_spans)
print(
    f"これらの記事の中の改行の数: {_newlines_inside:,}(記事の区切りと同じ改行 1 つが記事の中にもあるため、"
    "corpus.txt だけからは境界を復元できない)"
)

TOKEN_BYTE_LENGTHS = np.array(
    [len(tokenizer.id_to_symbol[i]) for i in range(VOCAB_SIZE)], dtype=np.int64
)  # バイトレベル BPE: 記号 1 文字 = 1 バイト
_window_length = MAX_FACTOR * L


def build_evaluation_windows(spans: list[dict]) -> dict:
    # 検証の部分を記事ごとの区間に分け(先頭の区間は記事の途中から)、区間ごとに符号化して長さ 8L の窓を切り出す。
    segments = [
        (sp["manifest_index"], corpus_text[max(sp["start"], VALIDATION_START) : sp["end"]])
        for sp in spans
    ]
    assert "\n".join(t for _, t in segments) == validation_text, (
        "区間を連結しても検証の部分にならない"
    )
    windows, info, rows, segment_ids = [], [], [], []
    for segment_number, (idx, text) in enumerate(segments):
        ids = tokenizer.encode(text)
        segment_ids.append(ids)
        assert tokenizer.decode(ids) == text, "符号化が可逆でない"
        assert int(TOKEN_BYTE_LENGTHS[ids].sum()) == len(text.encode("utf-8")), (
            "バイト数の合計が合わない"
        )
        count = len(ids) // _window_length
        for w in range(count):
            offset = w * _window_length
            windows.append(ids[offset : offset + _window_length])
            info.append(
                {
                    "segment": segment_number,
                    "manifest_index": idx,
                    "token_offset": offset,
                    "segment_tokens": len(ids),
                }
            )
        rows.append((idx, len(text), len(ids), count))
    tensor = torch.tensor(windows, dtype=torch.long)
    digest = hashlib.sha256(tensor.numpy().tobytes())
    digest.update(json.dumps(info).encode("utf-8"))
    return {
        "windows": tensor,
        "info": info,
        "rows": rows,
        "segment_ids": segment_ids,
        "hash": digest.hexdigest(),
    }


_evaluation_data = build_evaluation_windows(article_spans)
EVAL_WINDOWS_ALL = _evaluation_data["windows"]
WINDOW_INFO_ALL = _evaluation_data["info"]
_segment_ids = _evaluation_data["segment_ids"]
_segment_rows = _evaluation_data["rows"]
print(f"\n{'番号':>4} | {'文字数':>9} | {'トークン数':>10} | {'8L の窓':>7}")
for _row in _segment_rows:
    print(f"{_row[0]:>4} | {_row[1]:>9,} | {_row[2]:>10,} | {_row[3]:>7}")
_n_tokens = sum(r[2] for r in _segment_rows)
print(
    f"8L = {_window_length} トークン以上の記事の区間: {sum(r[2] >= _window_length for r in _segment_rows)} / "
    f"{len(_segment_rows)} 件、作れる評価窓: {len(EVAL_WINDOWS_ALL)} 個"
    f"(検証の部分の {_n_tokens:,} トークンのうち {len(EVAL_WINDOWS_ALL) * _window_length:,} トークンを使う)、"
    f"評価窓を持つ記事: {len({i['manifest_index'] for i in WINDOW_INFO_ALL})} 本、"
    f"全評価窓のハッシュ {_evaluation_data['hash'][:16]}"
)

# --- 2 つの方法(article_offsets と Wikipedia API)で評価窓の集合が一致すること ---
_alternative = None
if ARTICLE_SPANS_SOURCE == "wikipedia_api" and ARTICLE_OFFSETS_DRY_RUN_PATH.exists():
    _alternative = locate_wikipedia_article_spans(
        corpus_text,
        "en",
        WIKIPEDIA_CACHE_DIR,
        MANIFEST_PATH,
        metadata_path=ARTICLE_OFFSETS_DRY_RUN_PATH,
        start_position=VALIDATION_START,
    )
elif ARTICLE_SPANS_SOURCE == "metadata" and not IN_COLAB:
    _alternative = locate_wikipedia_article_spans(
        corpus_text, "en", WIKIPEDIA_CACHE_DIR, MANIFEST_PATH, start_position=VALIDATION_START
    )
if _alternative is not None:
    _alternative_spans, _alternative_source = _alternative
    assert _alternative_source != ARTICLE_SPANS_SOURCE
    assert _alternative_spans == article_spans, "2 つの方法で記事の範囲が一致しない"
    _alternative_hash = build_evaluation_windows(_alternative_spans)["hash"]
    assert _alternative_hash == _evaluation_data["hash"], "2 つの方法で評価窓の集合が一致しない"
    print(
        f"記事の範囲と評価窓の集合(ハッシュ {_alternative_hash[:16]})が、{ARTICLE_SPANS_SOURCE} と "
        f"{_alternative_source} で完全に一致: OK"
    )
else:
    print(
        "記事の境界の 2 つの方法による照合は省略した"
        "(Colab では Wikipedia API に依存しないため、またはローカルに dry-run の結果がないため)"
    )


def select_windows(info: list[dict], max_windows: int | None, per_article: int = 2) -> list[int]:
    # None ならすべて。そうでなければ、評価窓を per_article 個以上持つ記事を先頭から max_windows // per_article 本選び、
    # それぞれ先頭の per_article 個の窓を使う(同じ記事に複数の窓があるクラスタを含め、2 本以上の記事から選ぶ)。
    if max_windows is None:
        return list(range(len(info)))
    by_article: dict[int, list[int]] = {}
    for i, entry in enumerate(info):
        by_article.setdefault(entry["manifest_index"], []).append(i)
    eligible = [indices for indices in by_article.values() if len(indices) >= per_article]
    chosen = [
        i for indices in eligible[: max_windows // per_article] for i in indices[:per_article]
    ]
    return sorted(chosen)


# 本番(またはスモークテスト)で使う評価窓
_selected = select_windows(WINDOW_INFO_ALL, MAX_EVAL_WINDOWS)
EVAL_WINDOWS = EVAL_WINDOWS_ALL[_selected]
WINDOW_INFO = [WINDOW_INFO_ALL[i] for i in _selected]
WINDOW_ARTICLES = [
    i["manifest_index"] for i in WINDOW_INFO
]  # 記事を単位とするブートストラップのクラスタ
NUM_EVAL_ARTICLES = len(set(WINDOW_ARTICLES))
assert NUM_EVAL_ARTICLES >= 2, (
    "評価窓が 2 本以上の記事から選ばれていない(クラスタブートストラップの経路を通らない)"
)
# 同じ記事に複数の窓があるクラスタを含む(クラスタが窓と一致せず、記事を単位とする経路が窓を単位とする経路と区別される)
assert max(WINDOW_ARTICLES.count(a) for a in set(WINDOW_ARTICLES)) >= 2
NUM_EVAL_WINDOWS = len(EVAL_WINDOWS)
TARGET_BYTES = TOKEN_BYTE_LENGTHS[
    EVAL_WINDOWS[:, 1:].numpy()
]  # (窓, 8L - 1): 列 c は位置 j = c + 1
EVAL_WINDOWS_HASH = hashlib.sha256(EVAL_WINDOWS.numpy().tobytes()).hexdigest()[:16]

# 不変条件: どの窓も 1 つの記事の区間の中に収まり、その区間のトークン列の連続した部分列である(前提条件 P-C1 の根拠)
for _w, _info in zip(EVAL_WINDOWS, WINDOW_INFO, strict=True):
    _ids = _segment_ids[_info["segment"]]
    assert _info["token_offset"] + _window_length <= len(_ids) == _info["segment_tokens"]
    assert _w.tolist() == _ids[_info["token_offset"] : _info["token_offset"] + _window_length]
WINDOWS_WITHIN_ARTICLE = all(
    _info["token_offset"] + _window_length <= _info["segment_tokens"] for _info in WINDOW_INFO
)
# P0 の評価集合: モデルカードと同じ(検証の部分全体を長さ L の重ならない窓に区切る、記事の境界によらない)
P0_WINDOWS, P0_MASK = make_evaluation_windows(
    torch.tensor(tokenizer.encode(validation_text), dtype=torch.long), L
)
print(
    f"使う評価窓: {NUM_EVAL_WINDOWS} 個・{NUM_EVAL_ARTICLES} 本の記事(水準 {CURRENT_LEVEL_NAME!r})、"
    f"記事ごとの窓の数 {dict(sorted({a: WINDOW_ARTICLES.count(a) for a in set(WINDOW_ARTICLES)}.items()))}、"
    f"ハッシュ {EVAL_WINDOWS_HASH}、"
    f"すべて 1 つの記事の中: {WINDOWS_WITHIN_ARTICLE}"
)
```


    corpus.txt: reconstructing file:   0%|          |  0.00B / 24.3MB            



    corpus.txt: downloading bytes:           |  0.00B            



    metadata.json:   0%|          | 0.00/43.4k [00:00<?, ?B/s]


    コーパス取得元: kojikojiprg/ai-theories-corpus-en-pretraining(Hugging Face Hub)
    コーパス: 24,214,546 文字(取得元 hub、1.6s)、検証の部分: 1,210,727 文字(文字位置 23,003,819 から)
    記事の境界の取得元: metadata(Hub の metadata.json の article_offsets、0.1s)
    検証の部分にかかる記事: 18 件(マニフェストの番号 338〜355)
    これらの記事の中の改行の数: 9,333(記事の区切りと同じ改行 1 つが記事の中にもあるため、corpus.txt だけからは境界を復元できない)
    
      番号 |       文字数 |      トークン数 |   8L の窓
     338 |    53,678 |     13,305 |       6
     339 |    50,953 |     15,005 |       7
     340 |   165,445 |     38,455 |      18
     341 |    49,875 |     12,262 |       5
     342 |   122,195 |     32,780 |      16
     343 |    95,544 |     23,796 |      11
     344 |     2,416 |        701 |       0
     345 |       912 |        211 |       0
     346 |    14,666 |      4,847 |       2
     347 |    76,711 |     17,001 |       8
     348 |     1,070 |        331 |       0
     349 |    87,147 |     23,417 |      11
     350 |     1,828 |        503 |       0
     351 |   138,584 |     34,257 |      16
     352 |     7,974 |      1,942 |       0
     353 |    29,513 |      8,756 |       4
     354 |     2,552 |        632 |       0
     355 |   309,647 |     89,006 |      43
    8L = 2048 トークン以上の記事の区間: 12 / 18 件、作れる評価窓: 147 個(検証の部分の 317,207 トークンのうち 301,056 トークンを使う)、評価窓を持つ記事: 12 本、全評価窓のハッシュ cf5c8f203004310f
    記事の境界の 2 つの方法による照合は省略した(Colab では Wikipedia API に依存しないため、またはローカルに dry-run の結果がないため)
    使う評価窓: 147 個・12 本の記事(水準 'prod')、記事ごとの窓の数 {338: 6, 339: 7, 340: 18, 341: 5, 342: 16, 343: 11, 346: 2, 347: 8, 349: 11, 351: 16, 353: 4, 355: 43}、ハッシュ fda0bc5f5ca5b332、すべて 1 つの記事の中: True


### 5.7 微調整・較正のデータ

微調整(実験 B)は 008 の訓練の部分(コーパスの先頭 95%)で行う。その末尾の`CALIBRATION_CHARS`文字を較正用に取り分け、
微調整には使わない。**較正用のテキストは 008 の事前学習で使った部分であり、未見のテキストではない**(6.1 節)。
較正用の窓は長さ $4L$ の重ならない窓で、記事の境界はまたぎうる(学習率の選択と、前提条件の損失の測定にのみ使う)。

微調整の部分は`encode_text_to_memmap()`で`.cache/015_encoded/`に`uint16`の memmap として符号化する(同一セッション内の
再実行では再符号化しない)。キャッシュから読んだトークン列が、キャッシュを経由せず直接符号化したトークン列と一致すること
(単語の境界で切った先頭の部分で比べる)と、本番のステップ数が 1 エポックを超えないことを確かめる。


```python
finetune_text = train_text[:-CALIBRATION_CHARS]
calibration_text = train_text[-CALIBRATION_CHARS:]
_t0 = time.time()
finetune_ids = encode_text_to_memmap(tokenizer, finetune_text, ENCODED_CACHE_DIR / "finetune.u16")
FINETUNE_ENCODE_SECONDS = time.time() - _t0
# キャッシュを経由しない符号化との一致(単語の境界で切った先頭の部分)
_cut = re.compile(r"\S\s").search(finetune_text, 300_000).start() + 1
_direct = tokenizer.encode(finetune_text[:_cut])
assert np.array_equal(
    np.asarray(finetune_ids[: len(_direct)], dtype=np.int64), np.array(_direct)
), "キャッシュのトークン列が直接の符号化と一致しない"
print(
    f"微調整の部分: {len(finetune_text):,} 文字 -> {len(finetune_ids):,} トークン"
    f"({FINETUNE_ENCODE_SECONDS:.1f}s)、先頭 {len(_direct):,} トークンが直接の符号化と一致: OK"
)

_calibration_ids = torch.tensor(tokenizer.encode(calibration_text), dtype=torch.long)
_calibration_all, _ = make_evaluation_windows(_calibration_ids, FINETUNE_FACTOR * L)
_calibration_all = _calibration_all[
    : len(_calibration_ids) // (FINETUNE_FACTOR * L)
]  # 端数の窓は捨てる
assert len(_calibration_all) >= LEVELS["prod"]["CALIBRATION_WINDOWS"], "較正用の窓が足りない"
calibration_windows = _calibration_all[:CALIBRATION_WINDOWS]
calibration_mask = torch.ones_like(calibration_windows, dtype=torch.bool)
CALIBRATION_TARGET_BYTES = TOKEN_BYTE_LENGTHS[calibration_windows[:, 1:].numpy()]
print(
    f"較正用: {len(calibration_text):,} 文字 -> 長さ {FINETUNE_FACTOR * L} の窓 {len(_calibration_all)} 個のうち "
    f"{len(calibration_windows)} 個を使う"
)

TOKENS_PER_STEP = FINETUNE_BATCH_SIZE * FINETUNE_FACTOR * L
for _level in LEVELS.values():
    assert _level["FINETUNE_STEPS"] * TOKENS_PER_STEP <= len(finetune_ids), "1 エポックを超える"
print(
    f"微調整: 1 ステップ {TOKENS_PER_STEP:,} トークン、本番 {LEVELS['prod']['FINETUNE_STEPS']} ステップで "
    f"{LEVELS['prod']['FINETUNE_STEPS'] * TOKENS_PER_STEP / len(finetune_ids):.2f} エポック"
    f"(008 の事前学習 2,181 ステップ x 8,192 トークンの "
    f"{LEVELS['prod']['FINETUNE_STEPS'] * TOKENS_PER_STEP / (2181 * 8192):.1%})"
)
```

    .cache/015_encoded/finetune.u16: 5,914,639 トークンを uint16 memmap として書き出した
    微調整の部分: 22,843,819 文字 -> 5,914,639 トークン(15.5s)、先頭 73,390 トークンが直接の符号化と一致: OK
    較正用: 160,000 文字 -> 長さ 1024 の窓 40 個のうち 32 個を使う
    微調整: 1 ステップ 8,192 トークン、本番 400 ステップで 0.55 エポック(008 の事前学習 2,181 ステップ x 8,192 トークンの 18.3%)


### 5.8 評価のハーネス

- `set_rotary(model, method, s, base)`: モデルの全層の RoPE を、指定した手法・倍率の`RotaryPositionEmbedding`に差し替える
  (全層で 1 つのインスタンスを共有する。008 と同じ)。
- `token_losses(model, windows)`: 窓ごと・位置ごとの損失(nats)。列 $c$ は位置 $j = c + 1$ のトークンの損失。
- `range_bits(losses, a, c)`・`range_bytes(bytes, a, c)`: 位置の範囲 $[a, c)$ の、窓ごとの損失の合計(ビット)と分母(バイト)。
  bits-per-byte は窓をまたいで合計してから割る(比の平均ではなく、合計の比)。
- `truncated_token_losses(model, windows, a, c, context)`: 位置 $j \in [a, c)$ の各トークンを、直前の`context`トークン
  $x_{j-\mathrm{context}}, \dots, x_{j-1}$ **だけ** を入力として予測したときの損失(実験 C)。系列ごとに独立に順伝播するので、
  層を重ねても $j - \mathrm{context}$ より前のトークンの情報は入らない(スライディングウィンドウのマスクでは層ごとに受容野が
  広がるので使わない)。出力層は最後の位置にだけ適用する(`last_position_logits()`。モデルの順伝播と同じ計算で、
  最後の位置の logits が一致することを確かめる)。
- 同じ(モデル・手法・倍率・長さ)の損失は 1 回だけ計算し、`LOSS_CACHE`から再利用する(実験 A・C・D で共有する)。
  再利用した値が、計算し直した値と完全に一致することを 6.9 節で確かめる。


```python
def set_rotary(model: GPTLanguageModel, method: str, scale: float, base: float = ROPE_BASE):
    rope = make_rotary(method, scale, base)
    for block in model.blocks:
        block.self_attn.positional_transform = rope
    model.max_sequence_length = MAX_FACTOR * L
    return rope


def token_losses(
    model: GPTLanguageModel, windows: torch.Tensor, batch_size: int = EVAL_BATCH_SIZE
) -> np.ndarray:
    # 窓ごと・位置ごとの損失(nats)。形状 (窓, 長さ - 1)。
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size].to(device)
            logits = model(batch)[:, :-1]
            losses = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), batch[:, 1:].reshape(-1), reduction="none"
            ).view(batch.size(0), -1)
            rows.append(losses.float().cpu().numpy().astype(np.float64))
    return np.concatenate(rows)


def range_bits(losses: np.ndarray, start: int, end: int) -> np.ndarray:
    # 位置 j in [start, end) の損失の窓ごとの合計(ビット)。列 c は位置 j = c + 1。
    return losses[:, max(start, 1) - 1 : end - 1].sum(axis=1) / LOG2


def range_bytes(target_bytes: np.ndarray, start: int, end: int) -> np.ndarray:
    return target_bytes[:, max(start, 1) - 1 : end - 1].sum(axis=1).astype(np.float64)


def bits_per_byte(bits: np.ndarray, byte_counts: np.ndarray) -> float:
    return float(np.sum(bits) / np.sum(byte_counts))


def last_position_logits(model: GPTLanguageModel, tokens: torch.Tensor) -> torch.Tensor:
    # GPTLanguageModel.forward と同じ計算を行い、出力層だけを最後の位置に限る(実験 C の高速化)。
    hidden = model.token_embedding(tokens)
    mask = create_causal_mask(tokens.size(1), device=tokens.device)
    for block in model.blocks:
        hidden, _, _ = block(hidden, None, tgt_mask=mask)
    return model.lm_head(model.final_norm(hidden[:, -1]))


def truncated_token_losses(
    model: GPTLanguageModel, windows: torch.Tensor, start: int, end: int, context: int
) -> np.ndarray:
    # 位置 j in [start, end) のトークンを、直前の context トークンだけから予測したときの損失。形状 (窓, end - start)。
    assert start >= context
    sequences = windows[:, start - context : end].unfold(
        1, context + 1, 1
    )  # (窓, end - start, context + 1)
    flat = sequences.reshape(-1, context + 1)
    model.eval()
    rows = []
    with torch.no_grad():
        for s in range(0, len(flat), TRUNCATED_BATCH_SIZE):
            batch = flat[s : s + TRUNCATED_BATCH_SIZE].to(device)
            logits = last_position_logits(model, batch[:, :context])
            rows.append(
                F.cross_entropy(logits, batch[:, context], reduction="none").float().cpu().numpy()
            )
    return np.concatenate(rows).astype(np.float64).reshape(len(windows), end - start)


LOSS_CACHE: dict[tuple, np.ndarray] = {}


def cached_losses(key: tuple, compute) -> np.ndarray:
    if key not in LOSS_CACHE:
        LOSS_CACHE[key] = compute()
    return LOSS_CACHE[key]


def base_losses(method: str, factor: int) -> np.ndarray:
    # 微調整なしのモデルの、評価窓の先頭 factor x L トークンでの損失(s = factor)。
    def compute():
        set_rotary(base_model, method, factor)
        return token_losses(base_model, EVAL_WINDOWS[:, : factor * L])

    return cached_losses(("base", method, factor), compute)


# 最後の位置だけの logits が、モデルの順伝播の最後の位置の logits と一致すること
set_rotary(base_model, "yarn", FINETUNE_FACTOR)
with torch.no_grad():
    _probe_tokens = EVAL_WINDOWS[:2, :L].to(device)
    assert torch.allclose(
        last_position_logits(base_model, _probe_tokens), base_model(_probe_tokens)[:, -1], atol=1e-4
    )
# 切り詰めた評価の整合性: 位置 j = L のトークンは、全文脈でも直前の L トークンが文脈のすべてなので、両者が一致する
set_rotary(base_model, "none", 1)
_full = token_losses(base_model, EVAL_WINDOWS[:2, : L + 1])[:, L - 1]
_truncated = truncated_token_losses(base_model, EVAL_WINDOWS[:2], L, L + 1, L)[:, 0]
assert np.allclose(_full, _truncated, atol=1e-4), (_full, _truncated)
# 範囲の関数: [0, 8L) の分母は窓のバイト数から先頭トークンを除いたもの
assert np.array_equal(
    range_bytes(TARGET_BYTES, 0, MAX_FACTOR * L),
    TOKEN_BYTE_LENGTHS[EVAL_WINDOWS.numpy()].sum(axis=1)
    - TOKEN_BYTE_LENGTHS[EVAL_WINDOWS[:, 0].numpy()],
)
print(
    "評価のハーネス: 最後の位置だけの logits がモデルの順伝播と一致、"
    "切り詰めた評価(文脈 L)と全文脈の評価が位置 j = L で一致: OK"
)
```

    評価のハーネス: 最後の位置だけの logits がモデルの順伝播と一致、切り詰めた評価(文脈 L)と全文脈の評価が位置 j = L で一致: OK


### 5.9 微調整のハーネス

1 回の微調整(1 手法 × 1 シード × 1 学習率)を実行し、判定と診断に必要な量を記録して返す。モデルそのものは返さず、
保存もしない。

- ベースモデルを複製し、手法の RoPE に差し替える(位置補間・YaRN は $s = 4$ の静的なスケーリング、底の調整は底を
  $b' = b s^{d/(d-2)}$ にした変換なしの RoPE)。**全パラメータを学習する。**
- 学習は`train_language_model()`で、AdamW・warmup(ステップ数の 10%)+ cosine(最小学習率はピークの 1%)・
  gradient clipping・FP32、系列長 $4L$・バッチサイズ 8。シードは訓練のバッチの選び方を決め、**同じシードなら手法によらず
  同じバッチの列になる**(手法間で対応がとれる)。
- 学習の前後で、較正用の窓(長さ $4L$、全位置)の平均損失を同じ関数で測る(前提条件 P-B1)。
- `evaluate=True`のとき、評価窓の先頭 $4L$ トークンの位置ごとの損失を記録する。実験 C の切り詰めた評価
  (位置 $[3L, 4L)$ を直前の $L$ トークンから予測)も、YaRN では全シード、位置補間では診断用のシードで行う。


```python
def build_finetune_model(method: str) -> GPTLanguageModel:
    model = copy.deepcopy(base_model)
    if method == "adjusted_base":
        set_rotary(model, "none", 1, base=ADJUSTED_BASE)
    else:
        set_rotary(model, method, FINETUNE_FACTOR)
    return model


def mean_loss_nats(model: GPTLanguageModel, windows: torch.Tensor) -> float:
    return float(token_losses(model, windows).mean())


def run_finetune(
    method: str,
    seed: int,
    learning_rate: float,
    clip_threshold: float | None,
    num_steps: int,
    evaluate: bool = True,
) -> dict:
    model = build_finetune_model(method)
    initial_loss = mean_loss_nats(model, calibration_windows)
    optimizer = AdamW(list(model.parameters()), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    schedule = functools.partial(
        compute_warmup_cosine_learning_rate,
        warmup_steps=max(1, round(WARMUP_RATIO * num_steps)),
        total_steps=num_steps,
        peak_learning_rate=learning_rate,
        min_learning_rate=learning_rate * MIN_LEARNING_RATE_RATIO,
    )
    history = train_language_model(
        model,
        finetune_ids,
        calibration_windows,
        calibration_mask,
        int(CALIBRATION_TARGET_BYTES.sum()),
        num_steps=num_steps,
        batch_size=FINETUNE_BATCH_SIZE,
        sequence_length=FINETUNE_FACTOR * L,
        learning_rate=learning_rate,
        eval_interval=num_steps,
        device=device,
        seed=seed,
        optimizer=optimizer,
        learning_rate_schedule=schedule,
        gradient_clip_threshold=clip_threshold,
        evaluate_at_final_step=True,
    )
    calibration_losses = token_losses(model, calibration_windows)
    record = {
        "method": method,
        "seed": seed,
        "learning_rate": learning_rate,
        "clip_threshold": clip_threshold,
        "num_steps": num_steps,
        "history_length": len(history["train_loss"]),
        "initial_loss": initial_loss,
        "final_loss": float(calibration_losses.mean()),
        "calibration_bits_per_byte": bits_per_byte(
            range_bits(calibration_losses, L, FINETUNE_FACTOR * L),
            range_bytes(CALIBRATION_TARGET_BYTES, L, FINETUNE_FACTOR * L),
        ),
        "train_loss": history["train_loss"],
        "gradient_norm": history["gradient_norm"],
        "clip_trigger_ratio": float(np.mean(history["gradient_clip_triggered"])),
        "all_finite": bool(np.all(np.isfinite(history["train_loss"]))),
    }
    if evaluate:
        record["eval_losses"] = token_losses(model, EVAL_WINDOWS[:, : FINETUNE_FACTOR * L])
        record["eval_windows_hash"] = EVAL_WINDOWS_HASH
        if method == "yarn" or (method == "position_interpolation" and seed in DIAGNOSTIC_SEEDS):
            record["truncated_losses"] = truncated_token_losses(
                model, EVAL_WINDOWS, (FINETUNE_FACTOR - 1) * L, FINETUNE_FACTOR * L, L
            )
    del model
    return record
```

## 6. 実験 / Experiments

### 6.1 実験宣言セル: 共通の設定・検証すること・判定基準・前提条件

**この節の内容は本番実行の前に確定させ、結果を見た後に変更しない。**

#### 共通の設定

- **起点**: `kojikojiprg/ai-theories-small-gpt-en`の`main`(008 で事前学習、$L = 256$、$d = 32$、$b = 10000$、4 層・8 ヘッド)。
  トークナイザは`kojikojiprg/ai-theories-tokenizer-en`。
- **評価データ**: 008 の検証の部分(コーパスの末尾 5%)から、1 つの記事の中に収まる長さ $8L$ の重ならない窓を切り出したもの
  (5.6 節)。評価長 $2L, 4L$ では同じ窓の先頭の部分を使い、長さ $L$ の評価では各窓を 8 区間に分ける。
- **指標**: bits-per-byte。分母は、評価した(予測対象の)トークンに対応する UTF-8 バイト数。全条件でトークナイザが同じなので
  1 トークンあたりの指標でも条件間の比較は成り立つが、他のトピックとの一貫性のために bits-per-byte で報告する。位置の範囲
  $[a, c)$ の bits-per-byte は、全窓の損失の合計(ビット)を全窓のバイト数の合計で割る(合計の比)。
- **ばらつきの単位**: 微調整なしのモデルは決定的なので、実験 A・D のばらつきは **どの評価テキストを選んだか** に由来する
  (同じ分布の別のテキストで評価したらどう変わるか)。これを、**記事を単位とする対応付きのクラスタブートストラップ**
  (cluster bootstrap)で求める。反復ごとに、評価窓を持つ記事(本番では 12 本)を記事の数だけ復元抽出し、選ばれた記事の
  全評価窓を(重複して選ばれた記事は重複して)使って、各条件の合計の比を計算してから差をとる
  (`paired_cluster_bootstrap_ratio_of_sums()`、反復 $R = 10{,}000$ 回)。比べる全条件・全モデルに同じ再標本を使う(対応付き)。
  対比量の標準偏差は、反復ごとの差の標本標準偏差とする。記事ごとの難しさの違いは条件間で相殺されるので、対応のない
  比較より標準偏差が小さくなる。
- **判定基準の改訂の記録(実験 A〜D の標準偏差に共通、スモークテストの後・本番実行の前に改訂した)**:
  - **旧基準**: 評価窓を単位とする対応付きブートストラップ(反復ごとに評価窓を窓の数だけ復元抽出する、
    `paired_bootstrap_ratio_of_sums()`)で、$\sigma_A$・$\sigma_D$ と、実験 B・C の $\sigma_{\mathrm{window}}$ を求める。
  - **新基準**: 上のとおり、記事を単位とするクラスタブートストラップで同じ量を求める。$\sigma_{\mathrm{seed}}$ の導出、
    $\sigma = \sqrt{\sigma_{\mathrm{seed}}^2 + \sigma_{\mathrm{window}}^2}$ の形、対比量の定義、判定基準の形(支持: $\Delta - 2\sigma > 0$、
    反証: $\Delta + 2\sigma < 0$)、判定する水準は変えない。
  - **改訂の理由**: 本番の 147 個の評価窓は 12 本の記事から切り出したもので、1 本の記事だけで 43 個を占める。同じ記事の窓は
    話題・文体・固有名詞を共有し、互いに独立ではない。独立でない単位を独立なものとして復元抽出すると、標準偏差を過小に
    見積もる(擬似反復、pseudoreplication)。独立な単位は記事なので、記事を復元抽出する(実験 D で「同じ窓の 8 区間は
    相関するので窓を単位にする」としたのと同じ理由が、同じ記事の窓どうしにも当てはまる)。**この改訂は標本の独立性という
    一般論に基づくもので、スモークテストの結果の方向には依存しない。**
  - 旧基準による標準偏差は、判定に使わない診断量として引き続き計算・印字する。あわせて、1 本の記事が対比量を支配して
    いないかを見るため、記事ごとの対比量と記事ごとの評価窓の数を診断量として印字する。
  - **スモークテストの旧基準による結果(記録)**: コミット`a6b81cf`のセル出力(ローカルの MPS、評価窓 4 個はすべて
    記事 338 から、判定に使う条件のシード 2、ステップ 8)。実験 A: $\Delta_A = +0.1378$、$\sigma_A = 0.0114$、判定関数の結果は
    支持。実験 B: $\Delta_B = +0.0982$、$\sigma_{\mathrm{seed}} = 0.0037$、$\sigma_{\mathrm{window}} = 0.0093$、$\sigma_B = 0.0100$、支持。
    実験 C: $\Delta_C = +0.0054$、$\sigma_{\mathrm{seed}} = 0.0009$、$\sigma_{\mathrm{window}} = 0.0033$、$\sigma_C = 0.0034$、判定不能。
    実験 D: $\Delta_D = +0.0396$、$\sigma_D = 0.0022$、支持。いずれも動作確認のための値であり、結論ではない(評価窓が
    1 本の記事から選ばれていたため、この時点ではクラスタブートストラップの経路を実行できなかった。改訂後のスモークテストでは、
    評価窓を 3 本の記事から 2 個ずつ選ぶ)。
- **判定の形**: 対比量を $\Delta$、その標準偏差を $\sigma$ として、支持: $\Delta - 2\sigma > 0$、反証: $\Delta + 2\sigma < 0$、
  判定不能: それ以外(差が標準偏差の 2 倍以内に収まる)。前提条件が 1 つでも成立しなかった実験は、判定関数の結果に
  関わらず **前提不成立** とし、判定不能とは区別して報告する。
- **YaRN のハイパーパラメータは原論文の値に固定する** ($\alpha = 1$、$\beta = 32$、$\sqrt{1/t} = 0.1 \ln s + 1$)。
  評価集合で調整すると、判定に使うデータで YaRN だけを有利に選ぶことになる(位置補間には調整するハイパーパラメータが
  ないので、調整は YaRN にのみ働く)。原論文の値は実務で使われている値でもある。$L$ の小さい本モデルでは同じ $\alpha, \beta$
  でも外挿される部分空間が 1 個しかない(3.5 節)ことは、実験の前に config から決まる性質として承知したうえで固定する。
- **評価では KV キャッシュを使わない**(3.7 節)。
- **共通の前提条件 P0**: 読み込んだモデル(スケーリングなし)の、**モデルカードの値と同じ評価集合**(008 の検証の部分全体を
  長さ $L$ の重ならない窓に区切ったもの。記事の境界によらない)での bits-per-byte が、モデルカードの値 1.668067 と
  $|b_{\mathrm{P0}} - 1.668067| \le 0.002$ で一致すること。
  - **評価窓の違いの扱い**: 本ノートブックの評価窓(記事の中の窓)とモデルカードの評価窓は異なり、記事の境界をまたぐ窓の
    有無や窓の中の位置の分布が違うので、両者の bits-per-byte は一致する必要がない。そこで P0 は、モデルカードと **同じ**
    評価集合・同じ関数(`evaluate_bits_per_byte()`)で計算する。評価窓の違いがなくなるので、許容誤差は計算のデバイスや
    演算の順序による浮動小数点の差だけを見込めばよい(FP32 の損失の合計で $10^{-5}$ 程度)。0.002 はそれより十分大きく、
    重みの取り違え・トークナイザの取り違え・底 $b$ の誤り(いずれも bits-per-byte を 0.1 以上変えると見込まれる)より十分
    小さい。記事の中の窓でのスケーリングなしの bits-per-byte(長さ $L$ の区間)は、診断量として並べて印字する。
  - P0 はどの手法の効果とも独立な、読み込みの正しさの条件である。全実験の前提条件とする。

#### 実験 A: 微調整なしでの外挿(高周波の保存の効果)

- **検証すること**: 推論時にだけ適用した YaRN は、位置補間より、学習長を超える位置のトークンの bits-per-byte が低い。
- **条件**: スケーリングなし・位置補間・NTK-aware・NTK-by-parts(診断用、YaRN から温度の補正を除いたもの)・YaRN。
  評価長 $2L, 4L, 8L$(公比 2 の等比)、$s$ = 評価長 $/ L$。
- **判定する水準**: $4L$($s = 4$)のみ。$2L$・$8L$ は診断量とする。
- **対比量**: 評価長 $4L$ の同じ評価窓・同じ位置の範囲 $[L, 4L)$ での、位置補間と YaRN の bits-per-byte の差
  $\Delta_A = b_{\mathrm{interp}} - b_{\mathrm{yarn}}$。$b_{\mathrm{interp}}$・$b_{\mathrm{yarn}}$ はそれぞれの手法の、全窓の
  $[L, 4L)$ の損失の合計(ビット)をバイト数の合計で割った値である。正の値は YaRN の方が低い(良い)ことを表す。
- **標準偏差**: $\sigma_A$ = 記事を単位とする対応付きのクラスタブートストラップ(共通の設定)で、反復ごとに
  $b^{*}_{\mathrm{interp}} - b^{*}_{\mathrm{yarn}}$ を計算したときの標本標準偏差(改訂後。旧基準は評価窓を単位とする
  ブートストラップで、診断量として印字する。改訂の記録は共通の設定を参照)。
- **判定基準**: 支持: $\Delta_A - 2\sigma_A > 0$、反証: $\Delta_A + 2\sigma_A < 0$、判定不能: それ以外。
- **前提条件**: P0。
- **対比量と介入の直接の作用点の距離**: 介入(周波数の変換)が直接作用するのは各部分空間の回転角、すなわち Attention の
  logits である。bits-per-byte はそこから Attention の重み・4 層の残差の流れ・出力層を経た最終的な予測の質であり、
  作用点から遠い。それでも bits-per-byte を選ぶのは、「伸ばした長さで言語モデルとして使えるか」という検証したい主張そのもの
  だからである。作用点に近い量として、各部分空間の $r_i$ と $\theta'_i / \theta_i$(5.5 節)と、Attention のエントロピーの
  位置依存性を診断量として併記する。
- **診断量**: 位置の区間(幅 $L/4$)ごとの bits-per-byte の曲線(全条件・全評価長)。学習長以内の位置 $[0, L)$ の bits-per-byte。
  Attention のエントロピー(各 Query の注意の分布のエントロピーを層・ヘッド・窓で平均したもの)の Query の位置への依存性を、
  スケーリングなし・NTK-by-parts(温度の補正なし)・YaRN(温度の補正あり)について、評価長 $8L$ で比べる。
  $2L$・$8L$ での同じ対比量。記事ごとの $b_{\mathrm{interp}} - b_{\mathrm{yarn}}$ と記事ごとの評価窓の数。

#### 実験 B: 短い微調整の後の比較

- **検証すること**: 伸ばした長さ($4L$)で同じステップ数だけ微調整した後、YaRN は位置補間より、学習長を超える位置の
  bits-per-byte が低い。
- **条件**: 位置補間 + 微調整、YaRN + 微調整(判定に使う 2 条件、いずれも $s = 4$ の静的なスケーリング)。
  底の調整(ABF)+ 微調整(診断用。底を NTK-aware と同じ $b' = b s^{d/(d-2)}$、$s = 4$ にして、変換なしの RoPE で微調整する)。
  判定に使う 2 条件はシード $K = 5$ 個(0〜4)、診断用の底の調整はシード 2 個(0・1、実行時間を予算に収めるため)。
- **学習の構成**: 007 の部品(AdamW、warmup(ステップ数の 10%)+ cosine(最小学習率はピークの 1%)、gradient clipping)、
  FP32、全パラメータの微調整、重み減衰 0.1(008 の事前学習と同じ)、系列長 $4L$・バッチサイズ 8(1 ステップ 8,192 トークン)。
  データは 008 の訓練の部分(較正用に取り分けた末尾を除く)。シードはバッチの選び方を決め、同じシードなら手法によらず
  同じバッチの列になる。
- **ステップ数**: $T = 400$(本番)。**較正で決めず、実験の前に固定する。** 008 の事前学習の総トークン数の 18%、微調整の部分の
  約 0.55 エポックにあたる。YaRN の原論文 [4] が $s = 16$ の微調整に使ったステップ数(400)と同じ値を選んだ。ステップ数を
  結果を見て選ぶと、どちらかの手法に有利な時点を選ぶ余地が生じるので、事前に固定する。
- **学習率の較正の手順と採用条件**: 判定に使う 2 手法ごとに、学習率のグリッド $\{10^{-4}, 10^{-3.5}, 10^{-3}\}$(公比 $\sqrt{10}$)を、
  **本番と同じステップ数 $T$** ・シード 1000(本番のシードと重ならない)・gradient clipping なしで学習し、較正用の窓
  (訓練の部分の末尾から取り分けた、長さ $4L$ の窓 32 個)の位置 $[L, 4L)$ の bits-per-byte が最小の学習率を採用する。
  採用した学習率がグリッドの端なら、同じ公比でその側に 1 水準拡張して掃引を続ける(最大 3 回。下側は $10^{-5.5}$、上側は
  $10^{-1.5}$ まで。3 回拡張しても端のままなら前提条件 P-B2 の不成立とする。上限は再実行の記録を参照)。gradient clipping の閾値は、
  採用した学習率の較正の実行での勾配ノルムの 90% 分位点とする(008 と同じ決め方)。**較正には評価窓を使わない。**
  較正用の窓は 008 の事前学習で使ったテキストであり未見ではないが、全手法に同じ窓を使うので、学習率の選択の条件は手法間で
  揃っている。診断用の底の調整の条件は較正せず、YaRN で採用した学習率と gradient clipping の閾値を使う(実行時間を
  セッションの予算に収めるため。判定には使わない条件である)。
- **対比量**: シード $k$ ごとに、評価長 $4L$ の評価窓の位置 $[L, 4L)$ での差 $d_k = b_{\mathrm{interp}, k} - b_{\mathrm{yarn}, k}$
  (同じシードの 2 条件は同じバッチの列で学習しているので、シードで対応をとる)を求め、その平均 $\Delta_B = \frac{1}{K}\sum_k d_k$
  を対比量とする。
- **標準偏差の導出**: $\Delta_B$ のばらつきには、(1) 微調整の乱数(バッチの選び方)と (2) 評価窓の選び方の 2 つの独立な源がある。
  (1) は、固定した評価窓の上でのシード間のばらつきで、$\sigma_{\mathrm{seed}}^2 = \hat{s}_d^2 / K$($\hat{s}_d$ は $d_k$ の標本
  標準偏差、不偏分散による)。(2) は、学習済みの $2K$ 個のモデルを固定したまま記事を復元抽出する対応付きのクラスタ
  ブートストラップ(全モデルに同じ再標本。改訂後、旧基準は評価窓を単位とするブートストラップ)で、反復ごとに $\frac{1}{K}\sum_k (b^{*}_{\mathrm{interp}, k} - b^{*}_{\mathrm{yarn}, k})$ を計算したときの
  標本標準偏差 $\sigma_{\mathrm{window}}$。2 つの源は独立なので分散は和になり、$\sigma_B = \sqrt{\sigma_{\mathrm{seed}}^2 + \sigma_{\mathrm{window}}^2}$
  とする(誤差伝播)。
- **ノイズ床**: 1 条件(YaRN + 微調整)の、固定した評価窓の上での bits-per-byte のシード間の標本標準偏差 $\hat{s}_{\mathrm{yarn}}$ を
  ノイズ床として測り、診断量として印字する。判定の閾値には、ノイズ床そのものではなく対比量の標準偏差 $\sigma_B$ を使う。
- **判定基準**: 支持: $\Delta_B - 2\sigma_B > 0$、反証: $\Delta_B + 2\sigma_B < 0$、判定不能: それ以外。
- **前提条件**(いずれも手法の優劣とは独立な量で定義する):
  - P0。
  - P-B1: 3 条件(判定に使う 2 条件と底の調整)の全シードで、微調整後の較正用の窓(長さ $4L$、全位置、各手法の RoPE)の
    平均損失が、微調整前より下がっていること(微調整が実際に進んだこと)。
  - P-B2: 全ての実行で訓練損失が有限であり、判定に使う 2 手法の採用した学習率がグリッド(拡張の後)の内点であること
    (較正が機能したこと)。拡張の上限(再実行では 3 回)まで拡張しても端のままなら不成立とする。
- **対比量と介入の直接の作用点の距離**: 介入は周波数の変換の違いだが、微調整によって重みがその変換に適応するので、
  作用は周波数から重みの変化を経て予測に及ぶ。bits-per-byte は最終的な予測の質であり、作用点から遠い。検証したい主張
  (微調整した後も YaRN が有利か)がこの量についての主張なので、これを対比量とする。
- **診断量**: 学習長以内の位置 $[0, L)$ の bits-per-byte(微調整による短い文脈の性能の変化)、底の調整の条件の同じ量、
  訓練損失の曲線、gradient clipping の発動率、較正の全結果、記事ごとのシード平均の差と記事ごとの評価窓の数、
  旧基準の $\sigma_{\mathrm{window}}$。
- **アップロード**: 微調整したモデルは Hub にアップロードしない(条件比較のためのモデルであるため)。
- **再実行の記録(前提不成立による再実行)**:
  - **旧実行の出典**: コミット`982ff84`(Google Colab の Tesla T4、`SMOKE_TEST=False`、コミット`dcac404`のコードを未コミットの変更
    なしで実行したもの)のセル出力。
  - **旧実行の較正の結果**(較正用の窓の位置 $[L, 4L)$ の bits-per-byte): 位置補間は $10^{-4}$: 1.4407、$10^{-3.5}$: 1.4424、
    $10^{-3}$: 1.4727、下側に 1 回拡張した $10^{-4.5}$: 1.4739 で、$10^{-4}$ が内点として採用された。YaRN は $10^{-4}$: 1.4180、
    $10^{-3.5}$: 1.4327、$10^{-3}$: 1.4673 で $10^{-4}$ がグリッドの下端となり、1 回拡張した $10^{-4.5}$ でも 1.4161 と改善が続いて
    端のまま拡張の上限(1 回)に達した。
  - **旧実行の判定の一覧**: 実験 A 支持、実験 B 前提不成立(P-B2 不成立)、実験 C 前提不成立(P-B2 不成立)、実験 D 支持。
    実験 A は $\Delta_A = +0.1587$、$\sigma_A = 0.0049$、実験 D は $\Delta_D = +0.0407$、$\sigma_D = 0.0018$ だった。
  - **旧実行の実験 B・C の対比量と判定関数の結果**: 実験 B は $\Delta_B = +0.0072$、$\sigma_B = 0.0012$ で判定関数の結果は支持、
    実験 C は $\Delta_C = +0.0024$、$\sigma_C = 0.0007$ で判定関数の結果は支持だった。**これらは前提不成立の実行の値であり、
    結論として扱わない。**
  - **前提不成立と判断した根拠**: 事前に宣言した前提条件 P-B2(判定に使う 2 手法の採用した学習率がグリッドの内点であること)が、
    YaRN の較正で成立しなかった(拡張の上限に達しても採用した学習率がグリッドの端だった)。
  - **修正内容**: 較正で採用した学習率がグリッドの端だった場合の拡張の上限を、最大 1 回から最大 3 回に戻す(下側は $10^{-5.5}$、
    上側は $10^{-1.5}$ まで)。位置補間と YaRN の両方に同じ規則を適用する。拡張の上限は、第 1 段階の宣言では最大 2 回とし、
    本番実行の前の見直しで実行時間の見積もりを予算に収めるために最大 1 回に減らしていた。
  - **修正の理由**: P-B2 の成否は較正用の窓(訓練の部分)の値だけで決まり、実験 B・C の対比量(評価窓での手法間の差)を
    見ていない。拡張の上限を 1 回にしたのは、MPS での実行時間の見積もりを予算に収めるためだった。T4 での実測は予算の約 40%
    (見積もり 47.8 分)であり、元の上限(2 回)を超える 3 回に戻しても予算に収まる。修正は判定に使う 2 手法に同じ規則として
    適用され、観測結果の方向に依存しない。
  - **変えないもの**: 対比量の定義、標準偏差の導出(記事を単位とするクラスタブートストラップ、$\sigma_{\mathrm{seed}}$)、判定の形と
    閾値、判定する水準、$T = 400$、シード数、評価窓、P-B2 の定義(拡張の上限まで拡張しても端のままなら不成立)。診断用の
    底の調整の条件には、引き続き YaRN で採用した学習率と閾値を使う。
  - **再実行の結果をそのまま最終判定とする。** 旧実行で実験 B・C の判定関数の結果を見ていることを明記したうえで、再実行の結果が
    旧実行と異なっても、再実行の結果を採用する。実験 A・D も再実行で計算し直される。再実行の判定を最終判定とし、旧実行と
    判定が異なった場合は両方を 7 節で報告する。

#### 実験 C: 伸ばした文脈を実際に使っているか

- **検証すること**: 実験 B の YaRN + 微調整のモデルで、評価長 $4L$ の窓の最後の $L$ トークン(位置 $[3L, 4L)$)を、
  全文脈(位置 0 からの全トークン)で予測したときの bits-per-byte が、直前の $L$ トークンだけに切り詰めた文脈で予測した
  ときより低い。
- **切り詰めた文脈**: 位置 $j \in [3L, 4L)$ のトークンごとに、$x_{j-L}, \dots, x_{j-1}$ の $L$ トークンだけを入力として
  独立に順伝播し、最後の位置の予測を使う(5.8 節)。2 つの条件の違いは、**相対位置 $L$ 以上の Key があるかどうかだけ** である
  (RoPE は相対位置にのみ依存するので、入力の絶対位置の違いは影響しない)。モデル・RoPE(YaRN、$s = 4$)は同じ。
- **対比量**: シード $k$ ごとに、同じトークン(全評価窓の位置 $[3L, 4L)$)に対する差
  $c_k = b^{\mathrm{truncated}}_k - b^{\mathrm{full}}_k$ を求め、その平均 $\Delta_C = \frac{1}{K}\sum_k c_k$ を対比量とする。正の値は、
  $L$ より遠い文脈が予測を改善していること(伸ばした文脈を使っていること)を表す。
- **標準偏差の導出**: 実験 B と同じ。$\sigma_{\mathrm{seed}}^2 = \hat{s}_c^2 / K$、$\sigma_{\mathrm{window}}$ = 記事を単位とする
  対応付きのクラスタブートストラップ(改訂後。2 つの文脈長 × $K$ 個のモデルに同じ再標本)での $\frac{1}{K}\sum_k c^{*}_k$ の標本標準偏差、
  $\sigma_C = \sqrt{\sigma_{\mathrm{seed}}^2 + \sigma_{\mathrm{window}}^2}$。2 つの文脈長は同じトークン・同じバイト数を分母とする。
- **判定基準**: 支持: $\Delta_C - 2\sigma_C > 0$、反証: $\Delta_C + 2\sigma_C < 0$、判定不能: それ以外。
- **再実行**: 実験 C は実験 B のモデルを使うので、旧実行では実験 B と同じく P-B2 の不成立により前提不成立だった。再実行の
  記録(旧実行の値、根拠、修正内容と理由、再実行の結果を最終判定とすること)は実験 B の宣言のとおりである。
- **前提条件**: P0、P-B1・P-B2(実験 B のモデルを使うため)、P-C1: 全ての評価窓が 1 つの記事の中に収まっていること
  (記事の境界をまたぐと、遠い文脈が別の記事になり、使わないことが正しい振る舞いになるため)。
- **事前に述べておくこと**: この規模(4 層・$d_{\mathrm{model}} = 256$、$L = 256$)のモデルと Wikipedia の記事では、$L$ トークン
  (約 1,000 バイト)より遠い文脈が次のトークンの予測に与える情報は小さいと見込まれる。**差が標準偏差の 2 倍に届かず判定不能に
  なりうる。** その場合は「伸ばした文脈を使っていない」とは結論せず、この規模で検出できる効果がなかったと報告する。
- **対比量と介入の直接の作用点の距離**: 介入(文脈の切り詰め)が直接作用するのは、Attention が参照できる Key の集合である。
  bits-per-byte はそれを経た予測の質であり、「使っているか」という主張を直接表す量である。
- **診断量**: 同じ対比量を、微調整なしの各手法(スケーリングなし・位置補間・NTK-aware・YaRN、$s = 4$)と、微調整した位置補間
  (診断用のシード 2 個)で計算したもの。シードごとの $c_k$。記事ごとのシード平均の差と記事ごとの評価窓の数、旧基準の
  $\sigma_{\mathrm{window}}$。

#### 実験 D: 静的なスケーリングの短い系列でのコスト

- **検証すること**: 静的な YaRN($s = 4$)を常に適用すると、学習長 $L$ 以内の系列の bits-per-byte が、スケーリングなしより
  高くなる。
- **条件**: 各評価窓を 8 個の長さ $L$ の区間に分け、それぞれを独立な長さ $L$ の系列として評価する。静的な YaRN($s = 4$)と
  スケーリングなし。微調整なし。
- **対比量**: 全区間の bits-per-byte の差 $\Delta_D = b_{\mathrm{yarn}, s=4} - b_{\mathrm{none}}$。正の値は静的な YaRN のコスト
  (短い系列で悪化すること)を表す。
- **標準偏差**: 記事を単位とする対応付きのクラスタブートストラップでの差の標本標準偏差 $\sigma_D$(改訂後)。同じ窓の 8 区間は
  同じ記事の連続したテキストで互いに相関するので、区間ではなく、その窓が属する記事を単位として復元抽出する。旧基準は
  評価窓(長さ $8L$)を単位とするブートストラップで、診断量として印字する。
- **判定基準**: 支持: $\Delta_D - 2\sigma_D > 0$、反証: $\Delta_D + 2\sigma_D < 0$、判定不能: それ以外。
- **前提条件**: P0。
- **dynamic YaRN** が $l \le L$ でスケーリングなしと一致することは判定ではなく、アサーションで確かめる(5.5 節の RoPE の
  出力に加えて、この節で長さ $L$ の全区間の損失が完全に一致すること)。
- **対比量と介入の直接の作用点の距離**: 介入(周波数の変換と温度)が直接作用するのは回転角と logits であり、bits-per-byte は
  実験 A と同じく遠い下流の量である。「短い入力の性能が落ちるか」という主張そのものを表すので、これを対比量とする。
- **診断量**: 位置補間($s = 4$)での同じ量。記事ごとの $b_{\mathrm{yarn}, s=4} - b_{\mathrm{none}}$ と記事ごとの評価窓の数。dynamic YaRN と静的な YaRN の、評価長 $4L$ の位置 $[L, 4L)$ での比較。
  dynamic YaRN は系列長によって $s$ が変わるので、生成のように系列が伸びていく状況を、区間ごとの前方からの評価で近似する:
  位置 $j \in [(c-1)L, cL)$ のトークンを、先頭から $cL$ トークンの系列($s = c$)の順伝播で予測する($c = 1, 2, 3, 4$)。
  トークンごとに $s = \max(1, j/L)$ とする厳密な評価の、$L$ 刻みの近似である。

### 6.2 スケーリングの計測と外挿

本番のデータ量・回数がスモークテストと異なる重い処理について、3 点のデータ量で実行時間を実測し、
$\log t = \log a + b \log n$ をあてはめてべき指数 $b$ を推定し、本番のデータ量へ外挿する。外挿値には本番での実行回数を乗じる。

| 処理 | データ量 $n$ | 計測点 | 本番の実行回数 |
|---|---|---|---|
| 評価(長さ $2L, 4L, 8L$ の窓) | 窓の数 | 8・16・32 窓 | 実験 A: 5 手法(長さごと)、実験 B: 15 回の微調整の後に $4L$ で 1 回ずつ、実験 D の dynamic の診断量 |
| 評価(長さ $L$ の区間) | 区間の数 | 64・128・256 区間 | 実験 D: 4 条件 |
| 切り詰めた評価(実験 C) | 窓の数 | 2・4・8 窓 | YaRN × 5 シード + 位置補間 × 2 シード + 微調整なしの 4 手法 |
| 微調整(1 回) | ステップ数 | $T/8$・$T/4$・$T/2$(外挿の対象は本番の $T$) | 較正 2 手法 × 最大 6 水準 + 本番(判定用 2 手法 × 5 シード + 底の調整 × 2 シード) |
| P0 の評価 | 窓の数 | 64・128・256 窓 | 1 回 |
| 微調整の部分の符号化 | 文字数 | 25 万・50 万・100 万文字 | 1 回(セッション内でキャッシュ) |

微調整は、ステップ数によらない固定費(モデルの複製・学習前後の損失の測定)があるので小さいステップ数だけで $b$ を推定すると
$b < 1$ となり、本番の時間を過小に見積もりうる。そこで見積もりには、外挿値と、$T/2$ での実測値を 2 倍した値(固定費も
2 倍に数えるので安全側)の大きいほうを使う。見積もりの合計が予算の 80% を超える場合は警告を出す。
符号化は、計測ごとにトークナイザを読み込み直して単語のキャッシュを空にしてから測る。較正の微調整の回数は、拡張の上限
3 回を前提に 2 手法 × (3 + 3) = 最大 12 回とする。

**参考(旧実行の T4 での実測、コミット`982ff84`)**: 較正 827.1 秒(拡張の上限 1 回で、2 手法 × 4 水準 = 8 回)、本番の微調整
1,593.7 秒(12 回)。見積もりの基準は、引き続きこのノートブックを実行したデバイスとする。**CUDA 以外で計測した場合、外挿値は
その環境の値であり、T4 での時間とは異なる。** 記事の取得(ネットワーク)は 1 回限りなので、5.6 節の実測値をそのまま示す。


```python
def fit_and_extrapolate(label: str, sizes, times, target: float) -> float:
    fit = fit_power_law_exponent(sizes, times)
    extrapolated = fit.coefficient * target**fit.exponent
    detail = ", ".join(f"n={n}: {t:.3f}s" for n, t in zip(sizes, times, strict=True))
    print(
        f"[{label}] {detail} -> b={fit.exponent:.3f}(標準誤差 {fit.exponent_stderr:.3f}), "
        f"R^2={fit.r_squared:.4f}, n={target:,.0f} での外挿値 {extrapolated:.1f}s"
    )
    return extrapolated


_t0_scaling = time.time()
_prod = LEVELS["prod"]
_n_prod = len(EVAL_WINDOWS_ALL)  # 本番では作れる評価窓をすべて使う
set_rotary(base_model, "none", 1)
token_losses(
    base_model, EVAL_WINDOWS_ALL[:1, :L]
)  # 初回の呼び出しに伴うオーバーヘッドを計測から除く

_eval_ext = {}
_chunks = EVAL_WINDOWS_ALL[:32].reshape(-1, L)
_sizes = [64, 128, 256]
_eval_ext[1] = fit_and_extrapolate(
    "評価(長さ L の区間)",
    _sizes,
    [timed_call(lambda n=n: token_losses(base_model, _chunks[:n])) for n in _sizes],
    _n_prod * MAX_FACTOR,
)
for _factor in EVAL_LENGTH_FACTORS:
    _sizes = [8, 16, 32]
    _eval_ext[_factor] = fit_and_extrapolate(
        f"評価(長さ {_factor}L の窓)",
        _sizes,
        [
            timed_call(
                lambda n=n, f=_factor: token_losses(base_model, EVAL_WINDOWS_ALL[:n, : f * L])
            )
            for n in _sizes
        ],
        _n_prod,
    )
_sizes = [2, 4, 8]
_truncated_ext = fit_and_extrapolate(
    "切り詰めた評価(位置 [3L, 4L)、文脈 L)",
    _sizes,
    [
        timed_call(
            lambda n=n: truncated_token_losses(
                base_model, EVAL_WINDOWS_ALL[:n], (FINETUNE_FACTOR - 1) * L, FINETUNE_FACTOR * L, L
            )
        )
        for n in _sizes
    ],
    _n_prod,
)
_p0_total_bytes = len(validation_text.encode("utf-8"))
_sizes = [64, 128, 256]
_p0_ext = fit_and_extrapolate(
    "P0 の評価",
    _sizes,
    [
        timed_call(
            lambda n=n: evaluate_bits_per_byte(
                base_model, P0_WINDOWS[:n], P0_MASK[:n], _p0_total_bytes, device
            )
        )
        for n in _sizes
    ],
    len(P0_WINDOWS),
)
_steps_prod = _prod["FINETUNE_STEPS"]
_sizes = [_steps_prod // 8, _steps_prod // 4, _steps_prod // 2]
_finetune_times = [
    timed_call(lambda n=n: run_finetune("yarn", 0, 3e-4, None, n, evaluate=False)) for n in _sizes
]
_finetune_ext = max(
    fit_and_extrapolate("微調整(1 回)", _sizes, _finetune_times, _steps_prod),
    2 * _finetune_times[-1],
)
print(f"微調整 1 回の見積もり(外挿値と T/2 の実測値の 2 倍の大きいほう): {_finetune_ext:.1f}s")
_sizes = [250_000, 500_000, 1_000_000]
_encode_times = []
for _n_chars in _sizes:
    _fresh_tokenizer = load_bpe_id_tokenizer_json(TOKENIZER_JSON_PATH)  # 単語のキャッシュが空の状態
    _slice = finetune_text[2_000_000 : 2_000_000 + _n_chars]
    _encode_times.append(timed_call(lambda t=_fresh_tokenizer, x=_slice: t.encode(x)))
_encode_ext = fit_and_extrapolate("微調整の部分の符号化", _sizes, _encode_times, len(finetune_text))
_t_scaling = time.time() - _t0_scaling

_calibration_runs = len(JUDGED_FINETUNE_METHODS) * (len(LEARNING_RATE_GRID_K) + MAX_GRID_EXPANSIONS)
_main_runs = len(JUDGED_FINETUNE_METHODS) * _prod["NUM_SEEDS"] + _prod["NUM_DIAGNOSTIC_SEEDS"]
_truncated_runs = (
    _prod["NUM_SEEDS"] + _prod["NUM_DIAGNOSTIC_SEEDS"]
)  # YaRN 全シード + 位置補間の診断用のシード
ESTIMATE_SECONDS = {
    "微調整の部分の符号化(1 回)": _encode_ext,
    "P0 の評価(1 回)": _p0_ext,
    f"実験 A の評価({len(EVAL_METHODS)} 手法 x 3 長さ)": len(EVAL_METHODS)
    * sum(_eval_ext[f] for f in EVAL_LENGTH_FACTORS),
    f"実験 A のエントロピー(3 手法 x {_prod['ENTROPY_WINDOWS']} 窓、8L、フック込みで 2 倍と見込む)": 3
    * _eval_ext[MAX_FACTOR]
    * _prod["ENTROPY_WINDOWS"]
    / _n_prod
    * 2,
    f"較正の微調整(最大 {_calibration_runs} 回)": _calibration_runs * _finetune_ext,
    f"本番の微調整({_main_runs} 回)と 4L の評価": _main_runs
    * (_finetune_ext + _eval_ext[FINETUNE_FACTOR]),
    f"実験 C の切り詰めた評価({_truncated_runs} + 4 回)": (_truncated_runs + 4) * _truncated_ext,
    "実験 D の評価(4 条件 x 長さ L の区間)": 4 * _eval_ext[1],
    "実験 D の dynamic の診断量(前方からの評価)": _eval_ext[1] / MAX_FACTOR
    + _eval_ext[2]
    + 2 * _eval_ext[4],
    "スケーリング計測自体(このセルの実測)": _t_scaling,
}
_total_estimate = sum(ESTIMATE_SECONDS.values())
print(f"\n--- 本番実行の見積もり(実行回数を乗じた値、{device} 基準)---")
for _k, _v in ESTIMATE_SECONDS.items():
    print(f"  {_k}: {_v:,.1f}s")
print(f"  (参考)記事の取得(5.6 節の実測、ネットワーク): {ARTICLE_FETCH_SECONDS:.1f}s")
print(
    f"  合計: {_total_estimate:,.1f}s = {_total_estimate / 60:.1f} 分"
    f"(予算 {SESSION_BUDGET_SECONDS / 60:.0f} 分の {_total_estimate / SESSION_BUDGET_SECONDS:.1%})"
)
ESTIMATE_WARNING_FRACTION = 0.8
if _total_estimate > ESTIMATE_WARNING_FRACTION * SESSION_BUDGET_SECONDS:
    print(f"警告: 見積もりがセッションの予算の {ESTIMATE_WARNING_FRACTION:.0%} を超える")
if device.type != "cuda":
    print("注意: CUDA 以外での見積もりであり、T4 での時間とは異なる")
```

    [評価(長さ L の区間)] n=64: 0.151s, n=128: 0.235s, n=256: 0.416s -> b=0.733(標準誤差 0.053), R^2=0.9947, n=1,176 での外挿値 1.2s
    [評価(長さ 2L の窓)] n=8: 0.031s, n=16: 0.063s, n=32: 0.129s -> b=1.023(標準誤差 0.004), R^2=1.0000, n=147 での外挿値 0.6s
    [評価(長さ 4L の窓)] n=8: 0.088s, n=16: 0.178s, n=32: 0.354s -> b=1.001(標準誤差 0.004), R^2=1.0000, n=147 での外挿値 1.6s
    [評価(長さ 8L の窓)] n=8: 0.275s, n=16: 0.550s, n=32: 1.101s -> b=1.000(標準誤差 0.001), R^2=1.0000, n=147 での外挿値 5.1s
    [切り詰めた評価(位置 [3L, 4L)、文脈 L)] n=2: 0.597s, n=4: 1.189s, n=8: 2.389s -> b=1.001(標準誤差 0.003), R^2=1.0000, n=147 での外挿値 43.9s
    [P0 の評価] n=64: 0.104s, n=128: 0.209s, n=256: 0.419s -> b=1.003(標準誤差 0.003), R^2=1.0000, n=1,240 での外挿値 2.0s
    [微調整(1 回)] n=50: 13.340s, n=100: 26.575s, n=200: 59.846s -> b=1.083(標準誤差 0.051), R^2=0.9978, n=400 での外挿値 124.2s
    微調整 1 回の見積もり(外挿値と T/2 の実測値の 2 倍の大きいほう): 124.2s
    [微調整の部分の符号化] n=250000: 0.188s, n=500000: 0.360s, n=1000000: 0.604s -> b=0.843(標準誤差 0.055), R^2=0.9958, n=22,843,819 での外挿値 8.6s
    
    --- 本番実行の見積もり(実行回数を乗じた値、cuda 基準)---
      微調整の部分の符号化(1 回): 8.6s
      P0 の評価(1 回): 2.0s
      実験 A の評価(5 手法 x 3 長さ): 36.5s
      実験 A のエントロピー(3 手法 x 8 窓、8L、フック込みで 2 倍と見込む): 1.7s
      較正の微調整(最大 12 回): 1,490.3s
      本番の微調整(12 回)と 4L の評価: 1,509.9s
      実験 C の切り詰めた評価(7 + 4 回): 483.0s
      実験 D の評価(4 条件 x 長さ L の区間): 5.0s
      実験 D の dynamic の診断量(前方からの評価): 4.0s
      スケーリング計測自体(このセルの実測): 109.5s
      (参考)記事の取得(5.6 節の実測、ネットワーク): 0.1s
      合計: 3,650.5s = 60.8 分(予算 120 分の 50.7%)


### 6.2.1 ここまでの実行時間と、残りの見積もり

Colab で 6.2 節まで実行した時点で、続行するかを判断するためのセル。ここまでの実行時間(5.1 節のセットアップの開始から)と、
6.2 節の見積もりのうち未実行の処理(符号化・スケーリングの計測自体を除く)の合計を印字する。


```python
_done_items = ("微調整の部分の符号化(1 回)", "スケーリング計測自体(このセルの実測)")
_remaining = sum(v for k, v in ESTIMATE_SECONDS.items() if k not in _done_items)
_elapsed = time.time() - NOTEBOOK_START_TIME
print(f"ここまでの実行時間: {_elapsed / 60:.1f} 分")
print(f"残りの見積もり({device} 基準): {_remaining / 60:.1f} 分")
print(
    f"合計の見込み: {(_elapsed + _remaining) / 60:.1f} 分"
    f"(予算 {SESSION_BUDGET_SECONDS / 60:.0f} 分の {(_elapsed + _remaining) / SESSION_BUDGET_SECONDS:.1%})"
)
if _elapsed + _remaining > SESSION_BUDGET_SECONDS:
    print("警告: 合計の見込みがセッションの予算を超える。続行するかを判断すること")
```

    ここまでの実行時間: 3.7 分
    残りの見積もり(cuda 基準): 58.9 分
    合計の見込み: 62.6 分(予算 120 分の 52.1%)




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/03_efficient_training/015_long_context_extension.ipynb
