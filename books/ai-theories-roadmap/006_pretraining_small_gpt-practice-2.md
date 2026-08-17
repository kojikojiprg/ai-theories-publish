---
title: "小型 GPT の事前学習(Pretraining a Small GPT)(実装・実験編 2/4)"
---

この記事は後編(実装・実験編 2/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/006_pretraining_small_gpt-practice-1)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/006_pretraining_small_gpt-practice-3)。

### 5.6 ステップ数の決定(本番実行前の修正 10)

日英でバイトあたりのトークン数(fertility)が異なるため、同じ「1 エポック」に相当するステップ数も条件によって異なる。実験 F は日英の相対改善率を比較するため、**言語ごとにステップ数を変えると新たな交絡が入ってしまう**。そこで `NUM_STEPS`・`G_NUM_STEPS` はハードコードせず、**全言語・全トークナイザ条件の訓練トークン数を実測してから、最も条件の厳しい(訓練トークン数が最小の)条件を基準に計算式で決める**。

$$
\text{tokens\_per\_step} = \text{BATCH\_SIZE} \times \text{SEQUENCE\_LENGTH}
$$
$$
\text{min\_train\_tokens} = \min_{\text{lang, tokenizer}} \lvert \text{train\_ids}_{\text{lang, tokenizer}} \rvert
$$
$$
\text{NUM\_STEPS} = \left\lfloor \text{EPOCH\_CAP\_CF} \times \frac{\text{min\_train\_tokens}}{\text{tokens\_per\_step}} \right\rfloor, \qquad
\text{G\_NUM\_STEPS} = \left\lfloor \text{EPOCH\_CAP\_G} \times \frac{\text{min\_train\_tokens}}{\text{tokens\_per\_step}} \right\rfloor
$$

全条件で共通の `NUM_STEPS`(実験 C〜F)・`G_NUM_STEPS`(実験 G)を、最小トークン数の条件を基準に決めることで、**すべての条件が該当するエポック上限以内に収まることが構成上保証される**(訓練トークン数が最小条件以上の条件では、消化トークン数(分子は共通)に対するエポック比率が最小条件以下になるため)。`EVAL_INTERVAL`・`G_EVAL_INTERVAL` は、それぞれ 4 回・5 回程度の検証点が得られるように `NUM_STEPS`・`G_NUM_STEPS` から比例して決める。

**条件データのキャッシュ(本番実行前の修正 19)**: 訓練トークン数の実測(このセル)・実験 A(ランダム初期化の損失)・実験 C〜G(学習)は、いずれも同じ `(lang, tokenizer_name)` の組について `encode_corpus` による符号化を必要とする。符号化を毎回やり直すと、本番では 12 通りの条件に対して約 45 回の呼び出しが発生し(約 3.75 倍の重複)、コーパス全体(約 23 MB)の符号化を繰り返すのは無駄が大きい。そこで `(lang, tokenizer_name)` をキーにしたモジュールレベルの辞書でキャッシュする `get_condition_data()` を導入し、このセル以降のすべての箇所(`prepare_condition_data`、5.7 節)がこのキャッシュを経由するようにする。

キャッシュした `train_ids`・`windows`・`mask` は、学習側で書き換えられることがない(`get_random_batch` は `data[s:s+seq_len]` の読み取りスライスから新しいテンソルを `torch.stack` で作るのみで元のテンソルへの書き込みはなく、`evaluate_bits_per_byte` も `.to(device)` で読み取るのみ)ため、コピーせずそのまま返す。初回の符号化時のみ所要時間と `train_ids` のメモリ量を出力し、キャッシュヒット時は何も出力しない。


```python
_condition_data_cache: dict[tuple[str, str], dict] = {}


def get_condition_data(lang: str, tokenizer_name: str) -> dict:
    """言語・トークナイザ条件について、訓練データと検証窓を用意する(結果をキャッシュする、修正 19)。

    分母(total_bytes)は検証テキスト全体の UTF-8 バイト数であり、トークナイザに
    依存しない(本番実行前の修正 2)。同じ言語であれば全条件で完全に同一の値になる。
    """
    key = (lang, tokenizer_name)
    if key in _condition_data_cache:
        return _condition_data_cache[key]

    tokenizer = tokenizers[lang][tokenizer_name]
    t0 = time.time()
    train_ids = encode_corpus(tokenizer, train_text[lang])
    val_ids = encode_corpus(tokenizer, val_text[lang])
    elapsed = time.time() - t0
    windows, mask = make_evaluation_windows(val_ids, SEQUENCE_LENGTH)
    total_bytes = len(val_text[lang].encode("utf-8"))

    data = {
        "tokenizer": tokenizer,
        "vocab_size": get_vocab_size(tokenizer),
        "train_ids": train_ids,
        "windows": windows,
        "mask": mask,
        "total_bytes": total_bytes,
    }
    _condition_data_cache[key] = data

    train_mb = train_ids.element_size() * train_ids.nelement() / 1e6
    print(f"[符号化] {lang}/{tokenizer_name}: {elapsed:.2f}s  (train_ids: {train_mb:.1f} MB)")
    return data


tokens_per_step = BATCH_SIZE * SEQUENCE_LENGTH
train_token_counts = {}
for lang in ("ja", "en"):
    for tokenizer_name in tokenizer_conditions:
        data = get_condition_data(lang, tokenizer_name)
        train_token_counts[(lang, tokenizer_name)] = len(data["train_ids"])

min_train_tokens = min(train_token_counts.values())
min_condition = min(train_token_counts, key=train_token_counts.get)

NUM_STEPS = int(EPOCH_CAP_CF * min_train_tokens // tokens_per_step)
G_NUM_STEPS = int(EPOCH_CAP_G * min_train_tokens // tokens_per_step)
EVAL_INTERVAL = max(1, NUM_STEPS // 4)
G_EVAL_INTERVAL = max(1, G_NUM_STEPS // 5)

print(f"tokens_per_step = {tokens_per_step}")
print(f"min_train_tokens = {min_train_tokens:,}  (条件: {min_condition})")
print(f"NUM_STEPS = {NUM_STEPS}  (EPOCH_CAP_CF={EPOCH_CAP_CF})")
print(f"G_NUM_STEPS = {G_NUM_STEPS}  (EPOCH_CAP_G={EPOCH_CAP_G})")
print(f"EVAL_INTERVAL = {EVAL_INTERVAL}, G_EVAL_INTERVAL = {G_EVAL_INTERVAL}")

print(f"\n{'lang':4s} {'condition':12s} {'train_tokens':>12s} {'epoch(CF)':>10s} {'epoch(G)':>10s}")
for (lang, name), n_tokens in train_token_counts.items():
    epoch_cf = (NUM_STEPS * tokens_per_step) / n_tokens
    epoch_g = (G_NUM_STEPS * tokens_per_step) / n_tokens
    print(f"{lang:4s} {name:12s} {n_tokens:12,d} {epoch_cf:10.4f} {epoch_g:10.4f}")
    assert epoch_cf <= EPOCH_CAP_CF + 1e-9, f"{lang}/{name} が EPOCH_CAP_CF を超えている"
    assert epoch_g <= EPOCH_CAP_G + 1e-9, f"{lang}/{name} が EPOCH_CAP_G を超えている"
print("\n全条件がそれぞれのエポック上限以内であることを確認した")

# 実行時間の較正の節で測定した 1 ステップあたりの時間から、総実行時間を見積もる。
TOTAL_RUNS_C = 2 * len(NOISE_FLOOR_SEEDS)  # 実験 C: 日英 x 5 シード
TOTAL_RUNS_DEF = 2 * len(tokenizer_conditions)  # 実験 D〜F: 日英 x 5 条件(実験 C との共有分も含めた名目値)
TOTAL_RUNS_G = 2  # 実験 G: 日英各 1 ラン
total_steps = TOTAL_RUNS_C * NUM_STEPS + TOTAL_RUNS_DEF * NUM_STEPS + TOTAL_RUNS_G * G_NUM_STEPS
estimated_seconds = total_steps * median_step_time
print(
    f"\n見積もり総ラン数: 実験 C={TOTAL_RUNS_C}, 実験 D〜F={TOTAL_RUNS_DEF}, 実験 G={TOTAL_RUNS_G}"
    f"(合計 {TOTAL_RUNS_C + TOTAL_RUNS_DEF + TOTAL_RUNS_G} ラン、実際には一部を共有するため実行回数はこれより少ない)"
)
print(f"見積もり総ステップ数: {total_steps:,}")
print(f"見積もり総実行時間: {estimated_seconds / 60:.1f} 分({estimated_seconds / 3600:.2f} 時間)")
```

    [符号化] ja/character: 1.44s  (train_ids: 68.1 MB)
    [符号化] ja/bpe_v1024: 80.39s  (train_ids: 70.4 MB)
    [符号化] ja/bpe_v2048: 89.01s  (train_ids: 57.4 MB)
    [符号化] ja/bpe_v4096: 93.84s  (train_ids: 48.5 MB)
    [符号化] ja/bpe_v8192: 96.93s  (train_ids: 41.7 MB)
    [符号化] ja/unigram: 2.02s  (train_ids: 60.7 MB)
    [符号化] en/character: 4.09s  (train_ids: 184.0 MB)
    [符号化] en/bpe_v1024: 8.23s  (train_ids: 79.8 MB)
    [符号化] en/bpe_v2048: 9.36s  (train_ids: 65.6 MB)
    [符号化] en/bpe_v4096: 9.49s  (train_ids: 55.4 MB)
    [符号化] en/bpe_v8192: 8.40s  (train_ids: 47.7 MB)
    [符号化] en/unigram: 3.99s  (train_ids: 62.4 MB)
    tokens_per_step = 8192
    min_train_tokens = 5,217,974  (条件: ('ja', 'bpe_v8192'))
    NUM_STEPS = 318  (EPOCH_CAP_CF=0.5)
    G_NUM_STEPS = 636  (EPOCH_CAP_G=1.0)
    EVAL_INTERVAL = 79, G_EVAL_INTERVAL = 127
    
    lang condition    train_tokens  epoch(CF)   epoch(G)
    ja   character       8,507,563     0.3062     0.6124
    ja   bpe_v1024       8,799,191     0.2961     0.5921
    ja   bpe_v2048       7,172,792     0.3632     0.7264
    ja   bpe_v4096       6,061,518     0.4298     0.8595
    ja   bpe_v8192       5,217,974     0.4992     0.9985
    ja   unigram         7,586,435     0.3434     0.6868
    en   character      23,003,819     0.1132     0.2265
    en   bpe_v1024       9,971,837     0.2612     0.5225
    en   bpe_v2048       8,194,972     0.3179     0.6358
    en   bpe_v4096       6,926,577     0.3761     0.7522
    en   bpe_v8192       5,956,436     0.4374     0.8747
    en   unigram         7,801,569     0.3339     0.6678
    
    全条件がそれぞれのエポック上限以内であることを確認した
    
    見積もり総ラン数: 実験 C=10, 実験 D〜F=12, 実験 G=2(合計 24 ラン、実際には一部を共有するため実行回数はこれより少ない)
    見積もり総ステップ数: 8,268
    見積もり総実行時間: 18.7 分(0.31 時間)


### 5.7 データ準備・model 構築・学習実行のヘルパー関数

`encode_corpus`・`make_evaluation_windows`(トークナイザ条件ごとに訓練データ・検証窓を作る)、`GPTLanguageModel` の構築(3.4 節の通り RoPE・RMSNorm・SwiGLU を注入する)、`train_language_model` の呼び出しを、条件(言語・トークナイザ名)を受け取って一括で行うヘルパーにまとめる。実験 C・D・E・F・G はいずれもこのヘルパーを条件を変えて呼ぶだけで実行できる。`prepare_condition_data` は 5.6 節で定義した `get_condition_data`(キャッシュ付き、修正 19)の薄いラッパーであり、同じ条件を再度要求してもキャッシュヒットにより符号化をやり直さない。


```python
def prepare_condition_data(lang: str, tokenizer_name: str) -> dict:
    """言語・トークナイザ条件について、訓練データと検証窓を用意する(5.6 節のキャッシュ経由)。"""
    return get_condition_data(lang, tokenizer_name)


def build_model(vocab_size: int, seed: int, tie_embeddings: bool = True) -> GPTLanguageModel:
    """3.4 節の構成(RoPE・RMSNorm・SwiGLU)で GPTLanguageModel を構築する。

    修正 7: model サイズ(d_model・層数など)は実験 C・D・E・F・G のすべてで共通に
    固定する。model サイズと性能の関係は 009(スケーリング則)の主題であり、006 では
    変数として扱わない。

    ``tie_embeddings`` は既定で``True``(実験 A〜H すべてで使う本来の構成)。``False``は
    実験 A の判定基準未達の原因切り分け(重み共有の影響の有無)のためだけに使う
    (本番実行前の修正 26)。
    """
    torch.manual_seed(seed)
    d_k = D_MODEL // NUM_HEADS
    rope = RotaryPositionEmbedding(d_k, max_position=SEQUENCE_LENGTH)
    return GPTLanguageModel(
        vocabulary_size=vocab_size,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        max_sequence_length=SEQUENCE_LENGTH,
        positional_transform=rope,
        normalization_factory=RMSNorm,
        feed_forward_factory=functools.partial(SwiGLUFeedForwardNetwork, D_MODEL, SWIGLU_D_FF),
        tie_embeddings=tie_embeddings,
        dropout=DROPOUT,
    )


def run_condition(
    lang: str,
    tokenizer_name: str,
    seed: int,
    extended: bool = False,
) -> dict:
    """1 つの(言語, トークナイザ, seed)条件で学習を実行し、history と付随情報を返す。

    ``extended=True``(実験 G)は model サイズを変えず、ステップ数のみ
    `G_NUM_STEPS`・`G_EVAL_INTERVAL` に増やす(修正 7)。
    """
    data = prepare_condition_data(lang, tokenizer_name)
    model = build_model(data["vocab_size"], seed=seed)
    num_steps = G_NUM_STEPS if extended else NUM_STEPS
    eval_interval = G_EVAL_INTERVAL if extended else EVAL_INTERVAL
    history = train_language_model(
        model,
        data["train_ids"],
        data["windows"],
        data["mask"],
        data["total_bytes"],
        num_steps=num_steps,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        learning_rate=LEARNING_RATE,
        eval_interval=eval_interval,
        device=device,
        seed=seed,
    )
    return {
        "model": model,
        "history": history,
        "vocab_size": data["vocab_size"],
        "total_bytes": data["total_bytes"],
        "non_embedding_params": count_non_embedding_parameters(model),
        "total_params": sum(p.numel() for p in model.parameters()),
        "train_bytes": len(train_text[lang].encode("utf-8")),
        "train_tokens": len(data["train_ids"]),
    }


print("ヘルパー関数定義済み")
```

    ヘルパー関数定義済み


### 5.8 実験結果の永続化(本番実行前の修正 27)

判定関数の修正(修正 25)のような、学習ランのやり直しを必要としない軽微な変更のたびに、トークナイザ学習(本番で 34 分以上)を含む全工程の再実行が必要になっていた。そこで、学習ランの結果(条件ごとの最終 bits-per-byte・学習履歴・ノイズ床・パラメータ数・訓練トークン数・勾配ノルム統計)を JSON として `CACHE_DIR` に保存し、存在すれば読み込んで使う仕組みを導入する。

**保存する内容**: `run_condition()` の戻り値のうち、`model`(学習済み重み)を除いた JSON 化可能な部分(`history`・`vocab_size`・`total_bytes`・`non_embedding_params`・`total_params`・`train_bytes`・`train_tokens`)のみを保存する。**学習済み重みそのものは保存しない**(6 節末尾の生成例は、キャッシュから復元した実行では利用できない。次にモデルの学習ラン自体が必要になったとき、たとえば生成例を撮り直すときに再学習すればよい)。

**保存キー(cache key)**: 結果を左右する設定(`SMOKE_TEST`・`LM_CORPUS_BYTES`・`TOKENIZER_TRAIN_BYTES`・`MAX_CHUNK_BYTES`・`NUM_STEPS`・`G_NUM_STEPS`・model 設定・シード)をすべて含める。保存されたキーと現在の設定が完全に一致する場合のみキャッシュを使う。一致しない場合は通常通り再実行し(キャッシュは無効化され、新しい結果で上書きされる)、**アサーションによる不一致の検出でクラッシュさせることはしない**(設定変更のたびに再実行できなくなってしまうため)。


```python
RESULTS_CACHE_PATH = CACHE_DIR / "006_experiment_results_cache.json"


def build_results_cache_key() -> dict:
    """学習ランの結果を左右する設定のフィンガープリント(本番実行前の修正 27)。

    ここに含めた値のいずれか 1 つでも変われば、保存済みのキャッシュは無効になる
    (完全一致でのみキャッシュを使う)。
    """
    return {
        "SMOKE_TEST": SMOKE_TEST,
        "LM_CORPUS_BYTES": LM_CORPUS_BYTES,
        "TOKENIZER_TRAIN_BYTES": TOKENIZER_TRAIN_BYTES,
        "MAX_CHUNK_BYTES": MAX_CHUNK_BYTES,
        "NUM_STEPS": NUM_STEPS,
        "G_NUM_STEPS": G_NUM_STEPS,
        "D_MODEL": D_MODEL,
        "NUM_LAYERS": NUM_LAYERS,
        "NUM_HEADS": NUM_HEADS,
        "D_FF": D_FF,
        "SEQUENCE_LENGTH": SEQUENCE_LENGTH,
        "DROPOUT": DROPOUT,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "BPE_VOCAB_SIZES": list(BPE_VOCAB_SIZES),
        "UNIGRAM_VOCAB_SIZE": UNIGRAM_VOCAB_SIZE,
        "NOISE_FLOOR_SEEDS": list(NOISE_FLOOR_SEEDS),
    }


def _serialize_run(run: dict) -> dict:
    """``run_condition()`` の戻り値から、JSON 化できない ``model`` を除いた部分を取り出す。"""
    return {
        "history": run["history"],
        "vocab_size": run["vocab_size"],
        "total_bytes": run["total_bytes"],
        "non_embedding_params": run["non_embedding_params"],
        "total_params": run["total_params"],
        "train_bytes": run["train_bytes"],
        "train_tokens": run["train_tokens"],
    }


def save_results_cache(
    shared_runs: dict, noise_floor: dict, experiment_g_runs: dict, cache_key: dict
) -> None:
    """学習ランの結果を JSON として ``RESULTS_CACHE_PATH`` に保存する(修正 27)。"""
    payload = {
        "cache_key": cache_key,
        "shared_runs": {f"{lang}|{name}": _serialize_run(r) for (lang, name), r in shared_runs.items()},
        "noise_floor": noise_floor,
        "experiment_g": {lang: _serialize_run(r) for lang, r in experiment_g_runs.items()},
    }
    RESULTS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CACHE_PATH, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[キャッシュ] 実験結果を保存しました: {RESULTS_CACHE_PATH}")


def load_results_cache(current_key: dict) -> dict | None:
    """保存済みキャッシュを読み込む。設定が現在と完全一致する場合のみ返す(修正 27)。

    ファイルが存在しない、または保存された設定が現在の設定と異なる場合は ``None`` を返し、
    通常通りの再実行にフォールバックする(クラッシュさせない)。
    """
    if not RESULTS_CACHE_PATH.exists():
        return None
    with open(RESULTS_CACHE_PATH) as f:
        payload = json.load(f)
    if payload.get("cache_key") != current_key:
        print(f"[キャッシュ] 設定が変更されているため無効化します: {RESULTS_CACHE_PATH}")
        return None
    # 一致することを確認済みだが、上の比較と食い違えば実装の誤りであるため
    # 防御的にアサーションで再確認する(読み込み時の検証、修正 27)。
    assert payload["cache_key"] == current_key, "キャッシュの設定が現在の設定と一致しません"
    print(f"[キャッシュ] {RESULTS_CACHE_PATH} から読み込みました(設定が完全一致)")
    return payload


_results_cache_key = build_results_cache_key()
_cached_results = load_results_cache(_results_cache_key)
```



## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/02_pretraining/006_pretraining_small_gpt.ipynb
