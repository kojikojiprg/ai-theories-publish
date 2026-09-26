---
title: "SFT(指示チューニング) / Supervised Fine-Tuning (Instruction Tuning)(実装・実験編 2/4)"
---

この記事は後編(実装・実験編 2/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/016_supervised_fine_tuning-practice-1)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/016_supervised_fine_tuning-practice-3)。

### 5.7 1 条件の学習・評価を行うヘルパー

- `train_condition()`: 条件(1〜4)・シード・データ数 $N$ を受け取り、複製した
  ベースモデルに LoRA を掛けて $T$ ステップ学習する。シードは LoRA の $A$ の初期化(`torch.manual_seed`)と
  データの順序(`make_epoch_batches`)を決める。条件間でシードを対応させる(同じシード・同じ $N$ なら、
  条件 1〜4 でミニバッチに入る事例の添字は同一)。学習後に、凍結したパラメータがベースモデルと bit 単位で
  一致することを確かめる。
- `evaluate_model()`: 評価集合(条件の長さの水準)で、教師強制による事例ごとの負の対数尤度と、貪欲法の生成に
  よる事例ごとの形式の遵守・完全一致を記録する。条件 1 では未見テンプレートの評価集合でも生成する。


```python
def learning_rate_schedule(num_steps: int):
    return functools.partial(
        compute_warmup_cosine_learning_rate,
        warmup_steps=max(1, round(WARMUP_RATIO * num_steps)),
        total_steps=num_steps,
        peak_learning_rate=LEARNING_RATE,
        min_learning_rate=LEARNING_RATE * MIN_LEARNING_RATE_RATIO,
    )


def train_condition(
    condition: int,
    seed: int,
    num_examples: int,
    num_steps: int | None = None,
    examples: dict | None = None,
) -> tuple[nn.Module, dict]:
    num_steps = NUM_STEPS if num_steps is None else num_steps
    spec = CONDITIONS[condition]
    pool = (TRAIN_ENCODED if examples is None else examples)[spec["long"]]
    assert num_examples <= len(pool)
    batches = make_epoch_batches(num_examples, num_steps, BATCH_SIZE, seed)
    model = copy.deepcopy(base_model)
    torch.manual_seed(seed)  # LoRA の A の初期化
    replaced = apply_lora(model, LORA_TARGET_MODULES, rank=LORA_RANK, alpha=LORA_ALPHA)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history = train_instruction_tuning(
        model,
        pool[:num_examples],
        batches,
        optimizer,
        include_prompt_loss=not spec["mask"],
        device=device,
        learning_rate_schedule=learning_rate_schedule(num_steps),
        gradient_clip_threshold=None,  # gradient clipping は使わない(6.2 節の改訂 5)
    )
    # 凍結したパラメータがベースモデルと bit 単位で一致し、.grad が None であること
    frozen = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            continue
        assert p.grad is None and torch.equal(
            p, BASE_PARAMETERS[name.replace(".base_layer", "")]
        ), name
        frozen += 1
    record = {
        "condition": condition,
        "seed": seed,
        "num_examples": num_examples,
        "num_steps": num_steps,
        "history_length": len(history["step"]),
        "any_clip_triggered": any(history["gradient_clip_triggered"]),
        "gradient_norm": list(history["gradient_norm"]),
        "train_loss": list(history["loss"]),
        "num_replaced": len(replaced),
        "num_trainable_parameters": sum(p.numel() for p in trainable),
        "num_frozen_tensors_checked": frozen,
        "batches_hash": hash_json(batches),
        "max_epochs": num_steps * BATCH_SIZE / num_examples,
    }
    return model, record


def generate_and_score(model: nn.Module, encoded: list, examples: list) -> dict:
    generated = greedy_generate_until_stop(
        model,
        [list(x.token_ids[: x.prompt_length]) for x in encoded],
        tokenizer.decode,
        END_MARKER,
        MAX_NEW_TOKENS,
        device,
        GENERATION_BATCH_SIZE,
    )
    texts = [tokenizer.decode(g) for g in generated]
    scores = [score_generated_response(t, e.answer) for t, e in zip(texts, examples, strict=True)]
    return {
        "format_ok": np.array([s[0] for s in scores]),
        "exact": np.array([s[1] for s in scores]),
        "texts": texts,
    }


def evaluate_model(model: nn.Module, long: bool, include_unseen: bool = False) -> dict:
    losses = evaluate_instruction_negative_log_likelihood(
        model, EVAL_ENCODED[long], device, EVAL_BATCH_SIZE
    )
    result = {**losses, **generate_and_score(model, EVAL_ENCODED[long], EVAL_EXAMPLES)}
    result["response_loss"] = float(losses["response_sum"].sum() / losses["response_count"].sum())
    result["prompt_loss"] = float(losses["prompt_sum"].sum() / losses["prompt_count"].sum())
    if include_unseen:
        unseen = generate_and_score(model, EVAL_UNSEEN_ENCODED, EVAL_UNSEEN_EXAMPLES)
        result["unseen_exact"] = unseen["exact"]
        result["unseen_format_ok"] = unseen["format_ok"]
    return result


def run_condition(condition: int, seed: int, num_examples: int, keep_model: bool = False) -> dict:
    t0 = time.time()
    model, record = train_condition(condition, seed, num_examples)
    record["train_seconds"] = time.time() - t0
    t0 = time.time()
    record["evaluation"] = evaluate_model(
        model, CONDITIONS[condition]["long"], include_unseen=condition == 1
    )
    record["eval_seconds"] = time.time() - t0
    if keep_model:
        record["_model"] = model
    return record


def response_loss_by_seed(records: list) -> list[float]:
    return [r["evaluation"]["response_loss"] for r in records]


def three_way_verdict(value: float, sigma: float) -> str:
    # value > 2 sigma: 支持、value < -2 sigma: 反証、それ以外: 判定不能(value は期待する向きが正になるよう渡す)
    if not (math.isfinite(value) and math.isfinite(sigma)):
        return "判定不能"
    if value > 2 * sigma:
        return "支持"
    if value < -2 * sigma:
        return "反証"
    return "判定不能"


_t0 = time.time()
_check_model, _check_record = train_condition(1, 0, 64, num_steps=2)
_check_eval = evaluate_model(_check_model, False)
assert _check_record["history_length"] == 2 and _check_record["num_replaced"] == 2 * NUM_LAYERS
assert _check_record["num_trainable_parameters"] == LORA_PARAMETERS
print(
    f"ヘルパーの動作確認(条件 1、2 ステップ): 応答部分の負の対数尤度 {_check_eval['response_loss']:.4f}、"
    f"凍結テンソルの確認数 {_check_record['num_frozen_tensors_checked']}({time.time() - _t0:.1f}s)"
)
del _check_model
```

    ヘルパーの動作確認(条件 1、2 ステップ): 応答部分の負の対数尤度 7.8141、凍結テンソルの確認数 38(7.5s)



```python
# コピー能力の診断(6.8 節)で使う単語列: 学習データ・評価集合の入力と重複しない 8 語の列
COPY_SEED = 16_050
COPY_SEQUENCES = sample_word_sequences(
    np.random.default_rng(COPY_SEED),
    NUM_COPY_SEQUENCES,
    WORD_VOCABULARY,
    COPY_SEQUENCE_WORDS,
    COPY_SEQUENCE_WORDS,
    exclude={e.words for e in TRAIN_EXAMPLES} | set(EVAL_INPUTS),
)


@torch.no_grad()
def copy_losses(model: nn.Module) -> dict:
    # 単語列を 2 回繰り返した系列(テンプレートなし)で、1 回目・2 回目の単語の負の対数尤度の平均(nats)。
    # 1 回目は先頭の単語を除く(左の文脈がない)。各単語は単一トークンなので、系列長は単語数の 2 倍。
    model.eval()
    first, second = [], []
    for words in COPY_SEQUENCES:
        ids = tokenizer.encode("".join(" " + w for w in words) * 2)
        assert len(ids) == 2 * len(words)
        tokens = torch.tensor([ids], device=device)
        losses = F.cross_entropy(model(tokens)[0, :-1], tokens[0, 1:], reduction="none").cpu()
        first.append(losses[: len(words) - 1].mean().item())
        second.append(losses[len(words) - 1 :].mean().item())
    return {"first": float(np.mean(first)), "second": float(np.mean(second))}


def print_copy_losses(label: str, model: nn.Module) -> dict:
    result = copy_losses(model)
    print(
        f"{label}: 1 回目の単語 {result['first']:.3f} nats、2 回目の単語 {result['second']:.3f} nats"
        f"(差 {result['first'] - result['second']:+.3f}、{len(COPY_SEQUENCES)} 系列 x {COPY_SEQUENCE_WORDS} 語)"
    )
    return result


def response_total(record: dict) -> float:
    # 評価集合の応答部分の負の対数尤度の合計(nats)。L_{i,s} = response_total / 応答部分のトークン数の合計
    return float(record["evaluation"]["response_sum"].sum())


def prompt_total(record: dict) -> float:
    return float(record["evaluation"]["prompt_sum"].sum())


RESPONSE_TOKEN_COUNTS = np.array([len(x.response_ids) for x in EVAL_ENCODED[False]], dtype=np.int64)
RESPONSE_TOKEN_TOTAL = int(RESPONSE_TOKEN_COUNTS.sum())
PROMPT_TOKEN_COUNTS = {
    long: np.array([x.prompt_length - 1 for x in EVAL_ENCODED[long]], dtype=np.int64)
    for long in (False, True)
}
print(
    f"コピー能力の診断の単語列: {len(COPY_SEQUENCES)} 個(学習データ・評価集合の入力と重複しない)、"
    f"評価集合の応答部分のトークン数の合計 {RESPONSE_TOKEN_TOTAL}、指示部分(予測対象)の合計: "
    f"短い水準 {int(PROMPT_TOKEN_COUNTS[False].sum())}、長い水準 {int(PROMPT_TOKEN_COUNTS[True].sum())}"
)


def by_input(values: np.ndarray) -> np.ndarray:
    # 事例ごとの値(最後の軸が評価集合の事例 M 個、入力の順・同じ入力の中では課題の順)を、入力ごとの和にする。
    values = np.asarray(values)
    return values.reshape(*values.shape[:-1], NUM_EVAL_INPUTS, TASK_COUNT).sum(axis=-1)


RESPONSE_TOKEN_COUNTS_BY_INPUT = by_input(RESPONSE_TOKEN_COUNTS)  # n_x
PROMPT_TOKEN_COUNTS_BY_INPUT = {long: by_input(PROMPT_TOKEN_COUNTS[long]) for long in (False, True)}
assert np.array_equal(
    by_input(np.arange(EVAL_SIZE)),
    np.arange(EVAL_SIZE).reshape(NUM_EVAL_INPUTS, TASK_COUNT).sum(axis=1),
)
assert all(
    len({e.words for e in EVAL_EXAMPLES[i * TASK_COUNT : (i + 1) * TASK_COUNT]}) == 1
    for i in range(NUM_EVAL_INPUTS)
)
```

    コピー能力の診断の単語列: 64 個(学習データ・評価集合の入力と重複しない)、評価集合の応答部分のトークン数の合計 12356、指示部分(予測対象)の合計: 短い水準 33092、長い水準 132471


## 6. 実験 / Experiments

### 6.1 実験宣言セル: 共通の設定・検証すること・判定基準・前提条件

**この節の内容は本番実行の前に確定させ、結果を見た後に変更しない。** パイロット(6.2 節)を受けた改訂は、
本番実行の前に行ったものであり、6.2 節に旧設定・新設定・理由を記録した。

#### 共通の設定

- **起点**: `kojikojiprg/ai-theories-small-gpt-en`の`main`(008)、トークナイザは`kojikojiprg/ai-theories-tokenizer-en`。
- **微調整**: 全層の Query・Value 射影に LoRA($r = 8$、$\alpha = 8$、$\alpha / r = 1$)を掛け、ほかのパラメータは
  凍結する(012 と同じ)。全パラメータ微調整は行わない。AdamW(重み減衰 0)、warmup(ステップ数の 10%)+ cosine
  (最小学習率はピークの 1%)、fp32。**学習率は $10^{-2}$ に固定する**(012 の $10^{-1.5}$ からの改訂、6.2 節)。
- **gradient clipping は使わない**(全条件)。012 の学習の部品のうち、gradient clipping のみ踏襲しない。
  理由は 6.2 節の改訂 5 に記した(損失マスクの有無で勾配ノルムの尺度が異なり、どの閾値の決め方でも条件間に
  非対称な処理が混ざるため)。
- **学習ステップ数 $T = 2048$、バッチサイズ $b = 32$**(全実験で共通)。$T$ の決め方は 6.2 節。
- **データ数**: 条件 1〜4 は $N_{\max} = Tb = 65536$ 事例(1 エポックちょうど。各事例を 1 回ずつ使う)。
  実験 C の水準 $N$ では、先頭 $N$ 個の事例(入れ子の部分集合)を $T$ ステップ使う($Tb/N$ エポック。$N = 16$ では
  1 つのミニバッチに同じ事例が 2 回ずつ入る)。
- **シード**: 実験 A・B は $S_{\mathrm{AB}}$ 個、実験 C は $S_{\mathrm{C}}$ 個で、いずれも削る段階(下記)で決まる
  (段階 0 では両方 5)。シードは先頭から $0, 1, \dots$ を使う。シードは LoRA の $A$ の初期化とデータの順序を決める。
  条件間でシードを対応させる(同じシード・同じ $N$ なら、条件 1〜4 のミニバッチに入る事例の添字は同一)。
- **C-$N_{\max}$ と条件 1 の共有**: 実験 C の $N_{\max}$ の水準(C-$N_{\max}$)は条件 1 と同じ設定(損失マスクあり・
  短い水準・$N_{\max}$ 事例・同じシード)なので、同じ学習を重複して実行せず、条件 1 の学習の記録をそのまま使う
  (両者に共通するシード、すなわち実験 C の全シードについて。$S_{\mathrm{C}} \le S_{\mathrm{AB}}$ なので実験 C の
  シードは条件 1 のシードの先頭部分である)。共有していることをアサーションで確かめる。
- **決定的な実行**: 再現性のため、`torch.use_deterministic_algorithms(True)`と`CUBLAS_WORKSPACE_CONFIG`を設定する(5.1 節)。
- **採点**: 貪欲法で、終端記号`### End`の文字列が現れるまで、または上限 32 トークンまで生成する(KV キャッシュ(Key-Value Cache、010)を
  使う)。
  - **形式の遵守**: 上限トークン数以内に終端記号が現れ、それより前に他の区切り記号が現れないこと(区切り記号は
    すべて`###`で始まるので、「最初に現れる`###`が終端記号の先頭であること」と同値)。
  - **完全一致**: 形式を遵守し、かつ終端記号の前の文字列を空白で区切った単語の列が正解と一致すること。
    空白の正規化: 先頭・末尾の空白を除き、連続する空白・改行を 1 つの区切りとみなす。
  - **応答部分の負の対数尤度**: 教師強制で、評価集合の応答部分のトークン(終端記号を含む)の負の対数尤度の
    合計を、応答部分のトークン数の合計で割った値(nats / トークン)。**応答部分のトークン列は全条件で同一**
    (5.4 節で確認)なので、条件間でトークンあたりの値を比べてよい。
  - **指示部分の負の対数尤度**: 同様に、指示部分の予測対象のトークンについて求めた値。長さの水準ごとに
    指示部分が異なるので、同じ長さの水準の中でのみ比べる。
- **評価の長さの水準**: 各条件は、学習と同じ長さの水準の評価集合で評価する(条件 1・2 は短い水準、条件 3・4 は
  長い水準)。評価集合の入力・課題・指示文の選択は両水準で共通で、違いは前置きの有無のみである。

#### 条件の対応表(実験 A・B・C で共有)

| 番号 | 損失マスク | 指示の長さ | データ数 |
|---|---|---|---|
| 1 | あり | 短い | $N_{\max}$ |
| 2 | なし | 短い | $N_{\max}$ |
| 3 | あり | 長い | $N_{\max}$ |
| 4 | なし | 長い | $N_{\max}$ |
| C-$N$ | あり | 短い | $N \in \mathcal{N}$(公比 4 の等比、$\mathcal{N}$ は段階で決まる。段階 0 で $\{16, 64, \dots, 65536\}$) |

C-$N_{\max}$ は条件 1 と同一の学習である(共有する)。

#### 記号

- $x$: 入力(単語の列)、$X$: 評価用の入力の集合($|X| = 250$)、$K = 4$: 課題の数、$M = K|X| = 1000$: 評価集合の
  事例数。
- $s \in \{0, \dots, S-1\}$: シード。$S$ はシード数で、実験 A・B の式では $S = S_{\mathrm{AB}}$、実験 C の式では
  $S = S_{\mathrm{C}}$ である(いずれも選ばれた段階で決まる)。
- $\mathcal{N}$: 実験 C の水準の集合、$N_{\min}$: その最小値(選ばれた段階で決まる)。
- $L_{i,s}$: 条件 $i$・シード $s$ の、評価集合の応答部分の負の対数尤度(nats / トークン)。
- $L^{P}_{i,s}$: 条件 $i$・シード $s$ の、評価集合の指示部分の負の対数尤度(nats / トークン)。
- $p_{i,s}$: 条件 $i$・シード $s$ の、評価集合での完全一致率。$f_{i,s}$: 形式の遵守率。
- $L_{0}^{\mathrm{short}}$・$L_{0}^{\mathrm{long}}$: 微調整前のモデルの、短い水準・長い水準の評価集合での応答部分の
  負の対数尤度。

#### 共通の前提条件

- **P0**: 読み込んだ 008 の重みの検証 bits-per-byte が 1.6681(アップロードした重みの評価値 1.668067)と相対誤差
  1% 以内であること。008 と同じ分割(英語版 Wikipedia のコーパスの末尾 5%)・同じ評価(長さ 256 の重ならない窓)で
  測る。全実験に適用する。
- **P-L(学習の成立)**: 対象の全シードで、評価集合の応答部分の負の対数尤度が微調整前の値の 1/2 以下であること
  ($L_{i,s} \le 0.5 \, L_0^{\mathrm{short}}$(条件 1・2・C-$N_{\max}$)、$L_{i,s} \le 0.5 \, L_0^{\mathrm{long}}$(条件 3・4))。
  実験 A・B には条件 1〜4 の全シードに、実験 C には **C-$N_{\max}$ の全シードにのみ** 適用する。$N < N_{\max}$ の水準の
  同じ量(応答部分の負の対数尤度が微調整前の 1/2 以下か)は、診断量として水準・シードごとに印字する(判定には使わない。
  6.2 節の改訂 7)。検証する仮説(指示への依存・損失マスクと長さの交互作用・習得の速さの比較)のいずれとも独立に、
  学習が起きたことだけを確かめる量である。

#### 削る段階(実行時間の予算に応じた自動選択)

本番は Google Colab T4 の 1 セッション(予算 120 分)で完結させる。削る段階を次のとおり宣言する。

| 段階 | 実験 C の水準 $\mathcal{N}$ | 実験 C のシード数 $S_{\mathrm{C}}$ | 実験 A・B のシード数 $S_{\mathrm{AB}}$ |
|---|---|---|---|
| 0 | $\{16, 64, \dots, 65536\}$(7 水準) | 5 | 5 |
| 1 | $\{64, 256, \dots, 65536\}$(6 水準) | 5 | 5 |
| 2 | $\{64, 256, \dots, 65536\}$(6 水準) | 3 | 5 |
| 3 | $\{64, 256, \dots, 65536\}$(6 水準) | 3 | 3 |

- **選択の規則**: 本番の学習を始める前に(6.4 節)、スケーリングの計測(6.3 節)から 1 セッション全体の実行時間を
  段階ごとに見積もり、予算(T4 で 120 分)以内に収まる **最小の段階** を選ぶ。段階 3 でも予算を超える場合は、
  学習の前に例外で停止する。
- **選択は見積もりのみに基づき、どの実験の結果も参照しない。** 選択が行われる時点では、本番の学習を 1 回も行って
  いない(スケーリングの計測の学習は計測専用のデータで行い、評価集合の値を使わない)。
- シードを減らす場合は先頭のシード(0〜2)を使う。C-$N_{\max}$ と条件 1 の共有は両者に共通するシードについて行う。
- 選ばれた段階・各段階の見積もり・予算を 6.4 節に印字し、以降のすべてのセルは選ばれた段階の水準・シード数を使う。
  判定の式の $S$・$\mathcal{N}$・$N_{\min}$ は選ばれた段階の値であり、実際に使った値を判定の一覧(6.12 節)に印字する。
- **段階の順序の理由**: 失う情報の小さい順に並べた。まず実験 C の下端の分解能($N = 16$ の水準)、次に実験 C の
  精度(シード数)、最後に実験 B(と実験 A)の精度(シード数)を削る。実験 B は差の差を対比量とするため、シード数を
  減らしたときの影響が最も大きい。
- **段階 2・3 の代償**: シード数 3 では、シード間の標本標準偏差の推定が不安定になり(自由度 2)、$\sigma$ が偶然
  小さくも大きくもなる。閾値 $2\sigma$ の不確かさが大きくなり、平均的には判定不能に寄る。
- **段階 1 以降の代償**: $N_{90}$ の下端が 64 になる。形式の遵守率が $N = 64$ で既に飽和している場合は打ち切りとして
  記録され、実験 C の $\Delta$ はより保守的に、$\sigma_\Delta$ はより楽観的になりうる(実験 C の「打ち切りがある場合」)。

#### 実験 A: 指示への依存

**検証すること**: SFT 後のモデルは、指示を読んで出力を切り替えている。

**基準値 $c^*$**: 指示を無視して入力だけから出力を決める任意の決定的な戦略が、評価集合で達成できる完全一致率の
上界。入力 $x$ に対して指示を見ない戦略が出せる応答は 1 つだけなので、$x$ の $K$ 事例のうち正解にできるのは
$\max_y n_x(y)$ 個が最大である($n_x(y)$ は、入力 $x$ に対して応答 $y$ を正解とする課題の数)。評価集合が $X$ と
課題の直積であることを用いて

$$
c^* = \frac{1}{|X|} \sum_{x \in X} \max_y \frac{n_x(y)}{K}
$$

と厳密に計算する。本トピックの 4 課題は、互いに異なる 3 語以上の入力で応答がすべて異なるので $c^* = 1/K = 0.25$
である(5.4 節で計算して確かめる)。

**対比量**: $\bar{p} - c^*$。$\bar{p} = \frac{1}{S} \sum_s p_{1,s}$ は条件 1 の完全一致率のシード平均。

**標準偏差の導出**: $\bar{p}$ のばらつきには 2 つの独立な源がある。

1. 学習の乱数(シード): 評価集合を固定したとき、$p_{1,s}$ はシードごとにばらつく。シード間の標本標準偏差
   $s_p$(不偏)から、シード平均の分散は $s_p^2 / S$。
2. 評価集合の標本誤差: 評価集合は全シードで共有しているので、評価集合を引き直したときの変動は $S$ シードで
   平均しても減らない。**独立な標本の単位は入力 $x$ である**。評価集合は入力と課題の直積であり、同じ入力の
   $K$ 事例は同じ単語の並びを共有するので、正誤が独立でない(4 課題すべてで難しい入力・易しい入力がありうる)。
   課題は固定の 4 種類で、無作為に引いているのは入力である。そこで、入力 $x$ の完全一致率を 4 課題とシードで
   平均した

   $$
   a_x = \frac{1}{SK} \sum_s \sum_k c_{x,k,s}
   $$

   を入力ごとに求める($c_{x,k,s} \in \{0, 1\}$ は入力 $x$・課題 $k$・シード $s$ の条件 1 の正誤)。
   $\bar{p} = \frac{1}{|X|} \sum_x a_x$ であり、入力が独立に引かれているので、その分散は
   $\mathrm{Var}_x(a_x) / |X|$ で見積もれる($\mathrm{Var}_x$ は入力間の不偏分散)。$a_x$ にはシードによる変動も
   平均されて含まれるので、第 1 項と一部が重複し、$\sigma_A$ は保守的(大きめ)になる。

両者は独立なので

$$
\sigma_A^2 = \frac{s_p^2}{S} + \frac{\mathrm{Var}_x(a_x)}{|X|}
$$

とする。旧基準の第 2 項(事例を独立なベルヌーイ変数とみなす $\bar{p}(1 - \bar{p}) / M$)は診断量として併記する
(判定には使わない。改訂の経緯は 6.2 節の改訂 6)。

**判定**: $\bar{p} - c^* > 2\sigma_A$ なら支持、$\bar{p} - c^* < -2\sigma_A$ なら反証、それ以外は判定不能。

**前提条件 P-A**: 微調整前のモデルの完全一致率が $c^*$ 未満であること(微調整前から指示を読まずに解けている
わけではないことの確認)。ほかに P0・P-L(条件 1)を適用する。

**判定の偏りについての注記**: $T$ は、学習がほぼ完了する点(条件 1 の完全一致率が立ち上がった後)に選んだ
(6.2 節)。そのため、本実験の判定は支持に強く偏る。本実験の役割は、SFT の効果の提示と、$c^*$ に基づいて
「指示を読んでいる」ことを厳密に判定することにある。「SFT の後、指示を読まずに出せる完全一致率の上界を
超えている」ことは、完全一致率が高いというだけの主張より強く、指示を無視する戦略では説明できないことを意味する。

**作用点の記述**: SFT が直接作用するのは、指示に条件づけた応答トークンの予測分布である。完全一致率は生成を経た
下流の量だが、「指示に従う」という主張そのものを表す量なので対比量に選んだ。直接の作用点に近い診断量として、
条件 1 の応答部分の負の対数尤度を併記する。

**診断量**: 未見テンプレートでの完全一致率(学習に使わなかった指示文への汎化)、課題別の完全一致率、
旧基準の第 2 項 $\bar{p}(1 - \bar{p}) / M$。

#### 実験 B: 損失マスクの効果と指示の長さの交互作用

**検証すること**: 指示が長いほど、損失マスクの優位(マスクなしに対する、応答部分の負の対数尤度の差)が縮む
(Shi et al., NeurIPS 2024 [5])。

**対比量**: シードごとに

$$
D_s = (L_{4,s} - L_{3,s}) - (L_{2,s} - L_{1,s}), \qquad D = \frac{1}{S} \sum_s D_s
$$

$L_{2,s} - L_{1,s}$ は短い水準での「マスクなし − マスクあり」、$L_{4,s} - L_{3,s}$ は長い水準での同じ差である。
損失マスクが優位(マスクありのほうが応答部分の負の対数尤度が低い)なら差は正で、長い水準で優位が縮めば
$L_{4,s} - L_{3,s} < L_{2,s} - L_{1,s}$。**期待する方向は $D < 0$**。

**標準偏差の導出**: 実験 A と同じく、シードと評価集合の 2 つの独立な源を考える。

1. シード: 4 条件はシードで対応しているので、差の差 $D_s$ をシードごとに作ってからシード間の標本標準偏差 $s_D$ を
   とる。$D_s$ は 4 つの測定値の線形結合(係数 $+1, -1, -1, +1$)なので、独立な 4 条件から作るなら分散は各条件の
   分散の和(単一の測定の約 4 倍)になるが、対応があるとシードに共通の変動(同じ初期化・同じデータ順序による
   変動)が相殺されうる。$s_D$ を $D_s$ から直接求めることで、この相関を含めた誤差伝播が自動的に反映される。
   シード平均の分散は $s_D^2 / S$。
2. 評価集合: 実験 A と同じ理由で、独立な標本の単位は入力 $x$ である。評価集合の **入力** を 4 条件・全シードで
   **共通に** 復元抽出するブートストラップ(013 実験 D の対応付きの方式を、015 と同じく独立な単位のまとまりで
   行うもの)で $D$ を計算し直し、その分散 $\mathrm{Var}_{\mathrm{boot}}(D)$ を用いる。入力が選ばれると、その入力の
   $K$ 事例がまとめて選ばれる。応答部分のトークン列は全条件で同一なので、入力 $x$ の $K$ 事例の応答部分の
   トークン数の和 $n_x$ は条件によらない。入力 $x$ の $K$ 事例・条件 $i$・シード $s$ の応答部分の負の対数尤度の
   合計を $N_{x,i,s}$、入力 $x$ が選ばれた回数を $w_x$ とすると、$D$ は

   $$
   D^{*} = \frac{\sum_x w_x Q_x}{\sum_x w_x n_x}, \qquad
   Q_x = \frac{1}{S} \sum_s \left[ (N_{x,4,s} - N_{x,3,s}) - (N_{x,2,s} - N_{x,1,s}) \right]
   $$

   と、入力ごとの $Q_x$ と $n_x$ だけで書ける(分母が全条件で共通なので、比の差が分子の差の比になる)。
   事例を単位とするブートストラップ(旧基準)による分散は診断量として併記する(6.2 節の改訂 6)。

$$
\sigma_D^2 = \frac{s_D^2}{S} + \mathrm{Var}_{\mathrm{boot}}(D)
$$

**判定**: $D < -2\sigma_D$ なら支持、$D > 2\sigma_D$ なら反証、それ以外は判定不能。

**前提条件 P-B**: 損失マスクが実際に効いていること。長さの水準 $\lambda \in \{\mathrm{short}, \mathrm{long}\}$ ごとに、
マスクあり・なしの条件の組を $(a, u) = (1, 2)$(短い)・$(3, 4)$(長い)として、指示部分の負の対数尤度の差

$$
\Delta^{P}_{\lambda} = \frac{1}{S} \sum_s \left( L^{P}_{a,s} - L^{P}_{u,s} \right)
$$

が、両方の水準で $\Delta^{P}_{\lambda} > 2\sigma^{P}_{\lambda}$ であること(マスクなしのほうが指示部分の負の対数尤度が
低く、その差が標準偏差の 2 倍を超える)。$\sigma^{P}_{\lambda}$ は $\sigma_D$ と同じ方法(シード間の標本標準偏差 +
入力を共通に復元抽出するブートストラップ、分母は入力ごとの指示部分の予測対象のトークン数の和)で求める。この量は
介入(損失マスク)の直接の帰結であり、検証する仮説(応答部分での交互作用)とは独立である。ほかに P0・P-L(条件 1〜4)
を適用する。

**作用点の記述**: 介入(損失マスク)が直接作用するのは、指示部分のトークンが勾配に寄与するか否かである。
その直接の帰結は指示部分の負の対数尤度(前提条件 P-B に使う)。対比量の応答部分の負の対数尤度はそこから 1 段
下流で、介入が応答の予測に波及した結果を見る量である。完全一致率はさらに下流なので診断量とする。

**診断量**: 4 条件の完全一致率、指示部分の負の対数尤度、事例を単位とするブートストラップ(旧基準)による
$\mathrm{Var}_{\mathrm{boot}}(D)$。

#### 実験 C: 形式の習得と課題の習得の速さ

**検証すること**: 形式の遵守率は、課題の完全一致率より少ないデータ数で飽和する。008 のモデルは事前学習で
コピーの能力を獲得していない(3.7 節・6.2 節)。そのため本実験は、**形式の習得** と、**事前学習で獲得していない
能力(単語の並べ替えに必要なコピー)を SFT で新たに学ぶこと** の、習得の速さを比較する。LIMA の表層的アライメント
仮説は「能力は事前学習で獲得済み」を前提とするので、本実験はその追試ではない。支持された場合の結論は
「形式は、事前学習にない能力より少ないデータで習得される」に限る。

**学習予算**: 全水準で $T$ ステップ固定。小さい $N$ ではエポックを重ねる($N = 16$ で 4096 エポック、段階 1 以降の最小水準 $N = 64$ で 1024 エポック)。
データ数の部分集合は入れ子にする(先頭 $N$ 個)。

**量**: シードごと・指標(形式の遵守率 $f$、完全一致率 $p$)ごとに、飽和値を $N_{\max}$ での値とし、その 90% に
初めて達するデータ数 $N_{90}$ を、隣接する水準の間の $\log_2 N$ に対する線形補間で求める。水準 $N_j$ で初めて
閾値 $\tau = 0.9 \, v(N_{\max})$ 以上になったとき($v$ は指標の値)、

$$
\log_2 N_{90} = \log_2 N_{j-1} + \frac{\tau - v(N_{j-1})}{v(N_j) - v(N_{j-1})} \left( \log_2 N_j - \log_2 N_{j-1} \right)
$$

最小の水準で既に閾値以上なら $N_{90} = N_{\min}$ とし、打ち切り(censoring)として記録する。

**公比 4 による分解能の限界**: 水準は $\log_2 N$ で 2 ずつ離れている。$N_{90}$ は隣接する 2 水準の間を線形補間
するので、その間の曲線の形(例えば 2 水準の間の途中で急に立ち上がる)は分からず、$\log_2 N_{90}$ には最大で
約 2 の補間の誤差がありうる。この誤差はシードによらない系統的なもの(曲線の形が同じなら同じ向きにずれる)なので、
$s_\Delta$ に反映されない。したがって、$\Delta$ が 2 未満(水準 1 つぶん未満)の差であるときは、判定が支持でも
「水準の刻みより細かい差」であることに注意する。

**対比量**:

$$
\Delta = \frac{1}{S} \sum_s \left( \log_2 N_{90,s}^{\mathrm{task}} - \log_2 N_{90,s}^{\mathrm{format}} \right)
$$

task は完全一致率、format は形式の遵守率の $N_{90}$。**期待する方向は $\Delta > 0$**。

**標準偏差の導出**: 実験 A・B と同じく、シードと評価集合の 2 つの独立な源を考える。

1. シード: シードごとの差 $\log_2 N_{90,s}^{\mathrm{task}} - \log_2 N_{90,s}^{\mathrm{format}}$ の標本標準偏差 $s_\Delta$ から、
   シード平均の分散は $s_\Delta^2 / S$。
2. 評価集合: 評価集合は全シード・全水準で共有しているので、評価集合を引き直したときの変動は $S$ シードで平均しても
   減らない。独立な標本の単位は入力 $x$ である(実験 A と同じ理由)。そこで、評価集合の入力を復元抽出する
   ブートストラップの各反復で、**同じ再標本** から全水準・両指標・全シードの形式の遵守率と完全一致率を計算し直し、
   シードごとの $N_{90}$ と $\Delta$ を求める。こうすると、水準間・指標間・シード間で評価集合を共有していることによる
   相関が保たれる。その分散を $\mathrm{Var}_{\mathrm{boot}}(\Delta)$ とする。

$$
\sigma_\Delta^2 = \frac{s_\Delta^2}{S} + \mathrm{Var}_{\mathrm{boot}}(\Delta)
$$

(旧基準は第 1 項のみの $\sigma_\Delta = s_\Delta / \sqrt{S}$。6.2 節の改訂 6。旧基準の値は診断量として併記する。)
公比 4 による補間の誤差(上記)は、どちらの項にも反映されない。
**打ち切りがある場合**: 形式の遵守率が最小水準で既に飽和している(打ち切り)とき、真の $\log_2 N_{90}^{\mathrm{format}}$ は
記録値 $\log_2 N_{\min}$(段階 0 で $\log_2 16 = 4$、段階 1 以降で $\log_2 64 = 6$)以下なので、記録した $\Delta$ は
真の値以下(支持の向きに対して **保守的**)である。一方、打ち切られたシードは全て同じ値 $\log_2 N_{\min}$ を持つので、$s_\Delta$ はばらつきを過小に見積もり、$\sigma_\Delta$ は
**楽観的**(小さすぎる)になりうる。完全一致率の側が打ち切られた場合は、$\Delta$ は真の値以上(支持の向きに楽観的)に
なる。打ち切りの有無をシードごと・指標ごとに印字する。下端が 64 になる段階 1 以降では、形式の遵守率が $N = 64$ で
既に飽和して打ち切られる可能性が段階 0 より高く、この保守的・楽観的の両方の偏りが大きくなりうる。

**判定**: $\Delta > 2\sigma_\Delta$ なら支持、$\Delta < -2\sigma_\Delta$ なら反証、それ以外は判定不能。

**前提条件 P-C**: 両指標とも、飽和値が「90% 到達」を意味のあるものにする水準にあること。$N_{\max}$ での
完全一致率のシード平均 $\ge 2c^*$、かつ形式の遵守率のシード平均 $\ge 0.5$。ほかに P0・P-L(C-$N_{\max}$ の全シード)
を適用する。すなわち、実験 C の前提条件は P0・P-L(C-$N_{\max}$ の全シード)・P-C である。

**作用点の記述**: データ数は学習信号の量(見る事例の種類)に直接作用する。2 つの指標はいずれも生成を経た下流の量で
ある。両者を同じ尺度($\log_2 N_{90}$)で比べることで、生成による下流の減衰が両指標に共通にかかるという前提のもとで
相対比較する。各水準の応答部分の負の対数尤度も診断量として併記する。$N_{90}$ は閾値への到達という極値に近い統計
なので、全水準の両指標の曲線(シード平均と範囲)を診断量として図示する。

#### コピー能力の診断(判定基準を設けない観察)

学習データに含まれない 8 語の単語列(テンプレートなし)を 2 回繰り返した系列について、1 回目と 2 回目の単語の
負の対数尤度の平均を測る。コピーの能力があれば、2 回目の単語は 1 回目から予測でき、負の対数尤度が大きく下がる。
対象は、微調整前のモデルと、条件 1 のシード 0(C-$N_{\max}$ のシード 0 と同一の学習)。定性的な
観察であり、判定基準を設けない。

### 6.2 スモークテスト(パイロット)による設定の改訂

本番実行の前に、ノートブックの外(ローカルの MPS、スクリプトで実行)で、本番と同じスケール(同じモデル・同じ LoRA の
設定・$b = 32$、$T$ は 32〜2048)のパイロットを行った。以下はその記録である。パイロットの評価集合は、本番とは別の
乱数シードで生成した入力(100 または 250 個)と 4 課題の直積であり、本番の評価集合はパイロットの評価入力と重複しない
(5.4 節でアサーションにより確認)。**パイロットとスモークテストを受けて改訂したのは、学習率・$T$ の決め方・
実行の分割と実験 C の水準・実験 C の「検証すること」の本文の解釈の範囲・gradient clipping の不使用・判定の分散の
項の標本の単位・実験 C への前提条件 P-L の適用範囲・実行の構成(1 セッションと削る段階)であり、実験 A〜C の対比量・閾値の倍率(2 倍)・期待する方向は変更していない。** いずれも本番実行の
前の改訂である。

#### 改訂 1: 学習率($10^{-1.5} \to 10^{-2}$)

**旧設定**: 012 の LoRA で較正した学習率 $10^{-1.5}$(012 は Tiny Shakespeare への 87 ステップの適応で較正した)。

**観察**: 旧設定では、$T$ を増やしても完全一致率が 0 のままだった(形式の遵守率は 1 に近く、形式は学習している)。
学習率を変えた比較(条件 1 と同じ設定、損失マスクあり・短い水準、シード 0):

| 学習率 | $T$ | 応答部分の負の対数尤度(nats / トークン) | 形式の遵守率 | 完全一致率 | パイロット |
|---|---|---|---|---|---|
| $10^{-1.5}$(旧設定) | 256 | 3.438 | 0.97 | 0.000 | 1 |
| $10^{-2}$ | 256 | 3.132 | 0.95 | 0.000 | 1 |
| $3 \times 10^{-3}$ | 256 | 3.402 | 0.93 | 0.000 | 1 |
| $10^{-3}$ | 256 | 3.652 | 0.89 | 0.000 | 1 |
| $10^{-1.5}$(旧設定) | 512 | 3.504 | 1.00 | 0.000 | 2 |
| $10^{-1.5}$(旧設定) | 1024 | 3.686 | 1.00 | 0.000 | 2 |
| $10^{-2}$ | 1024 | 0.858 | 1.00 | 0.487 | 2 |
| $10^{-1.5}$(旧設定) | 2048 | 3.742 | 0.99 | 0.000 | 2 |
| $10^{-2}$ | 2048 | 0.479 | 1.00 | 0.948 | 2 |
| $3 \times 10^{-3}$ | 2048 | 0.862 | 1.00 | 0.510 | 2 |
| $10^{-3}$ | 2048 | 2.365 | 1.00 | 0.018 | 2 |

- パイロット 1: 評価入力 250 個(乱数シード 2016)、学習データ $Tb$ 事例(1 エポック)。
- パイロット 2: 評価入力 100 個(乱数シード 1)、学習データ $Tb$ 事例(1 エポック)。
- いずれも gradient clipping なし・シード 0・本番と同じ課題・テンプレートの形式・LoRA の設定・$b = 32$。
  $T = 256$ では 4 水準とも完全一致率が 0 で、応答部分の負の対数尤度が最も低かったのは $10^{-2}$。
  本番と同じ $T = 2048$ での比較でも、完全一致率・応答部分の負の対数尤度とも $10^{-2}$ が最良だった
  ($10^{-1.5}$ は $T = 512, 1024, 2048$ のいずれでも完全一致率 0)。
- 旧設定の学習率での $T$ の走査(パイロット 1、$T = 32 \sim 512$)でも完全一致率はすべて 0 だった(改訂 2)。

**改訂の理由**: 旧設定では **学習が立ち上がらない**(完全一致率が全ての $T$ で 0、応答部分の負の対数尤度が 3.4 以上で
下がらない)。改訂は「課題が学習されるか」だけに基づいており、検証する仮説(指示への依存・損失マスクと長さの
交互作用・習得の速さの比較)の向きには依存しない。学習率の選択にパイロットの評価集合の完全一致率を用いたので、
本番の評価集合はパイロットの評価入力と重複しないように生成した(5.4 節)。

#### 改訂 2: 学習ステップ数 $T$(相対 2% の較正規則の廃止、$T = 2048$ に固定)

**旧規則**: 条件 1・シード 0 で $T$ を等比の候補から走査し、「評価集合の応答部分の負の対数尤度の $T \to 2T$ での改善が、
$T$ 時点の値の 2% 未満」となる最小の $T$ を採用する。

**パイロットでの挙動**: 旧設定の学習率での $T$ の走査(パイロット 1、評価入力 250 個)で、応答部分の負の対数尤度は
$T = 32, 64, 128, 256, 512$ で 3.874・3.768・3.623・3.469・3.483 だった($T = 256 \to 512$ で悪化)。旧規則は
$T = 256$ を選ぶが、そこでは完全一致率は 0 で、何も学習していない。

**廃止の理由**:

1. **立ち上がり前の停滞で規則が満たされる**: 学習が立ち上がる前の平坦な区間では改善が小さく、規則が満たされてしまう
   (上の $T = 256$)。
2. **負の対数尤度が 0 に近づくと、相対改善が大きいまま残る**: 本トピックの課題は応答が決定的に決まるので、学習が
   進むと応答部分の負の対数尤度は 0 に近づく。改善の絶対値が小さくなっても、$T$ 時点の値に対する比は小さくならない
   (学習率 $10^{-2}$ で $T = 1024 \to 2048$ に 0.858 → 0.479、約 44% の改善)。現実的な $T$ で規則が満たされない。

**$T = 2048$ を選んだ理由**: 学習率 $10^{-2}$ では、完全一致率は $T = 256$ で 0 から、$T = 1024$ で約 0.5、$T = 2048$ で
約 0.95 に立ち上がる。$T = 1024$ は急な立ち上がりの途中にあり、シード間のばらつきが最大になる(下の表)。
$T = 2048$ はその後で、学習がほぼ完了している。これより大きい $T$ は、1 セッションの予算(6.3 節)に収まらない。
$T$ を学習がほぼ完了する点に選んだことによる実験 A への影響は 6.1 節に記した。

学習率 $10^{-2}$ でのシード間のばらつき(パイロット 2、評価入力 100 個):

| $T$ | シード 0 | シード 1 | シード 2 | 範囲 |
|---|---|---|---|---|
| 1024(完全一致率) | 0.487 | 0.713 | 0.540 | 0.226 |
| 2048(完全一致率) | 0.948 | 0.960 | 0.927 | 0.033 |
| 1024(応答部分の負の対数尤度) | 0.858 | 0.723 | 0.788 | 0.135 |
| 2048(応答部分の負の対数尤度) | 0.479 | 0.472 | 0.478 | 0.007 |

$T = 1024$ では、シード 0 と 2 で回転の課題の完全一致率が 0 のまま(シード 1 は 0.69)で、どの課題が立ち上がったかが
シードによって異なっていた。$T = 2048$ では 4 課題とも 0.9 以上に立ち上がった。

**前提条件 P-L の追加**: $T$ を較正で決めなくなったので、学習が成立したことを全条件・全シードで確かめる前提条件
P-L(6.1 節)を追加した。

#### 改訂 3: 実行の分割と実験 C の水準

2 つの Colab セッションに分ける(1.1 節)。実験 C の水準を公比 2 から公比 4($N \in \{16, \dots, 65536\}$)にし、条件 1 と
C-$N_{\max}$ の学習の共有をやめた(セッションが分かれるため)。いずれも実行時間の見積もりのみに基づく変更である。
公比 4 による $N_{90}$ の分解能の限界は 6.1 節に記した。**実行の分割と共有の廃止は、改訂 8 で取りやめた**(公比 4 は
維持する)。

#### 改訂 4: 実験 C の「検証すること」の本文(解釈の範囲の明確化)

パイロットで、008 のモデルがコピーの能力を持たないことを観察した(同じ単語列を 2 回繰り返しても、2 回目の単語の
負の対数尤度は約 10.4〜11.0 で、1 回目の約 11.2〜11.4 からほとんど下がらない。8〜32 語、各 50 系列)。これを受けて、
実験 C の「検証すること」の本文を、LIMA の仮説の追試ではなく「形式」と「事前学習にない能力」の習得の速さの比較と
明記した(6.1 節)。これは本番実行の前に得た観察に基づく **解釈の範囲の明確化** であり、判定基準(対比量・閾値の
導出式・期待する方向)は変更していない。

#### 改訂 5: gradient clipping を使わない

**旧案 1(当初の設定)**: 012 と同じ決め方で、条件ごとに clipping なしの較正の実行(シード 1000)の勾配ノルムの
90% 分位点を閾値とする。**問題点**: 損失マスクの有無で勾配ノルムの尺度が 2〜4 倍異なる(スモークテストの較正で、
閾値は条件 1 が 1.35、条件 2 が 3.15、条件 3 が 1.25、条件 4 が 4.76)。閾値の絶対値が条件間で異なるので、実験 B の
比較に、条件ごとに異なる値をとる非対称なハイパーパラメータが混ざる。

**旧案 2**: 条件 1 の閾値を全条件で共有する。**問題点**: 尺度の違いにより、損失マスクなしの条件でのみ clipping が
常時発動し、勾配が一律に縮小される。これは損失マスクなしの条件だけ実質的な学習率を下げることに等しく、実験 B の
対比量に交絡する。

**新設定**: 全条件で gradient clipping を使わない。012 の学習の部品(AdamW・warmup + cosine・fp32・LoRA の設定)の
うち、clipping のみ踏襲しない。**判断の根拠**: 本番前の確認(下記)で、clipping なしでも全条件が前提条件 P-L
(学習の成立)を満たした。P-L は検証する仮説とは独立な量であり、判断は仮説の向きに依存しない。あわせて条件ごとの
閾値の較正の実行(セッション 1 で 4 回、セッション 2 で 1 回)を削除した。

#### 改訂 6: 判定の分散の項の標本の単位(事例 → 入力)

**旧基準**: 評価集合の標本誤差を、事例を独立な単位として見積もる。実験 A は $\bar{p}(1 - \bar{p}) / M$(事例を独立な
ベルヌーイ変数とみなす)、実験 B は事例を復元抽出するブートストラップ、実験 C は評価集合の標本誤差を含めず
$\sigma_\Delta = s_\Delta / \sqrt{S}$。

**新基準**: 独立な標本の単位を入力 $x$ とする。実験 A は $\mathrm{Var}_x(a_x) / |X|$、実験 B・前提条件 P-B は入力を
共通に復元抽出するブートストラップ、実験 C は入力を復元抽出するブートストラップの分散 $\mathrm{Var}_{\mathrm{boot}}(\Delta)$
を加える(導出は 6.1 節)。対比量・閾値の倍率(2 倍)・期待する方向は変更していない。

**改訂の理由**: 評価集合は入力と課題の直積であり、同じ入力の $K$ 事例は同じ単語の並びを共有するので独立でない。
無作為に引いているのは入力であり、課題は固定の 4 種類である。したがって独立な標本の単位は入力である、という一般論に
基づく改訂であり、観測結果の向きには依存しない。**改訂の契機**: スモークテストで、実験 B の評価集合の分散の項を入力を
単位として計算すると、事例を単位とした値の約 2.5 倍($3.030 \times 10^{-3}$ 対 $1.234 \times 10^{-3}$)だった。

**スモークテストの旧基準による判定結果**(旧案 1 の gradient clipping あり、$T = 32$・2 シード・評価入力 16 個。
動作確認のみで結論ではなく、前提条件 P-L が不成立のため最終判定はいずれも「前提不成立」):

| 実験 | 対比量 | 旧基準の標準偏差 | 判定関数の結果 | 最終判定 |
|---|---|---|---|---|
| A | $\bar{p} - c^* = -0.2500$ | $\sigma_A = 0.0000$ | 反証 | 前提不成立 |
| B | $D = +0.34491$ | $\sigma_D = 0.05020$ | 反証 | 前提不成立 |
| C | $\Delta = -0.8375$ | $\sigma_\Delta = 0.8375$ | 判定不能 | 前提不成立 |

#### 改訂 7: 実験 C への前提条件 P-L の適用範囲(C-$N$ の全水準 → C-$N_{\max}$ のみ)

**旧定義**: P-L を実験 C の全水準・全シード(C-$N$、$N \in \{16, \dots, 65536\}$)に適用する。

**新定義**: 実験 C には C-$N_{\max}$ の全シードにのみ適用する。$N < N_{\max}$ の水準の同じ量は、診断量として水準・シードごとに
印字する(判定には使わない)。実験 A・B(条件 1〜4)への適用は変えない。

**改訂の理由**: 前提条件は、操作する変数から独立でなければならない。実験 C が操作する変数はデータ数 $N$ である。小さい $N$ で
評価集合の応答部分の負の対数尤度が下がらないこと(少数の事例の暗記により、評価集合に汎化しないこと)は、実験 C が測ろうと
している現象そのものである。全水準に適用すると、仮説の内容(少ないデータで何が習得されるか)によって前提の成否が決まって
しまう。$N_{\max}$ での学習の成立は、2 つの指標の飽和値を定義する前提として必要なので残す。この理由は観測結果の向きに
依存しない一般論である。

**スモークテストの旧定義による結果**(改訂 6 と同じ旧実行): P-L(C-$N$ の全水準・全シード)は False(8 実行のうち
満たしたのは 6、$N = 16$ の 2 シードが満たさなかった)で、旧定義のもとでの実験 C の最終判定は「前提不成立」だった
(同じ実行では P-C も不成立だった。動作確認のみで結論ではない)。

#### 改訂 8: 2 セッションの構成から、1 セッションと削る段階の自動選択へ

**旧構成**(改訂 3): 本番を 2 つの Colab セッション(セッション 1: 実験 A・B、セッション 2: 実験 C)に分け、切り替えの
フラグ・パートの識別子の印字・セルのメタデータで各セッションの出力を管理する。C-$N_{\max}$ は条件 1 と共有せず、
セッション 2 で独立に学習する。実験 C は 7 水準・5 シードで、セッション 2 が予算を超える場合のみ最小水準を 64 に上げる。

**新構成**: 本番を 1 セッション(T4 で予算 120 分)で完結させる。パートの切り替えの仕組みを撤去し、C-$N_{\max}$ を
条件 1 と共有する(同じ学習を重複して実行しない)。本番の学習の前に、見積もりのみから削る段階(6.1 節)を自動で選ぶ。

**理由**: 2 つのセッションに分けると、セッション間で共通のセルの出力が上書きされ、実行の手順と出力の管理の手間が
大きい。無料枠の 1 セッションで完結させる方針とし、予算に収まらない分は、あらかじめ宣言した順序で削る。この変更は
実行の手間と予算の方針のみに基づき、どの実験の結果にも依存しない(本番は未実施であり、スモークテストの判定の値も
参照していない)。**段階の順序の理由**: 失う情報の小さい順(実験 C の下端の分解能 → 実験 C の精度 → 実験 B の精度)
とした(6.1 節)。

#### 本番前の確認(第 1 段階で実施)

本番の設定($T = 2048$・学習率 $10^{-2}$・$b = 32$・シード 0)で条件 1〜4 を 1 回ずつ学習し(ノートブックの外、ローカルの MPS)、
次の真偽値のみを確かめた。応答部分の負の対数尤度の値や条件間の差は、実験 B の対比量を本番前に覗かないため印字していない。
条件 1 は、条件 2 の P-B の向き(マスクありの条件 1 との比較)に必要なので加えた。データは本番とも評価集合とも別の乱数シード
(16017)で生成した評価入力 250 個と学習データ 65,536 事例を使い、gradient clipping は使っていない(本番でも使わない、改訂 5)。

| 確認すること | 結果 |
|---|---|
| 条件 1 が P-L を満たす | True |
| 条件 2 が P-L を満たす | True |
| 条件 3 が P-L を満たす | True |
| 条件 4 が P-L を満たす | True |
| 条件 2 の指示部分の負の対数尤度が条件 1 より低い(P-B の向き、短い水準) | True |
| 条件 4 の指示部分の負の対数尤度が条件 3 より低い(P-B の向き、長い水準) | True |

いずれも真だったので、本番に進める。

### 6.3 スケーリングの計測と外挿(1 セッションの見積もり)

本番でデータ量がスモークテストの何倍にもなる重い処理について、3 点のデータ量で実行時間を実測し、
$\log t = \log a + b \log n$ をあてはめてべき指数 $b$ を推定し、本番のデータ量へ外挿する。本番では、この計測を
**本番の実行の冒頭に T4 上で** 行い、その値のみから削る段階を選ぶ(6.4 節)。

- **学習**: ステップ数 $T/16, T/8, T/4$(本番の $T$ で 128・256・512)を短い水準と長い水準で別々に計測する
  (長い水準は系列が約 3 倍長い)。外挿値と、$T/4$ の実測値の 4 倍の大きいほうを 1 回の学習の見積もりとする
  (ステップ数によらない固定費があると $b < 1$ となり、外挿値が過小になりうるため)。
- **教師強制の評価・生成**: 計測専用のデータ(`TIMING_DATA_SEED`、本番の評価集合とは別)で、事例数 64・128・256 を
  短い水準と長い水準で計測し、本番の $M = 1000$ に外挿する。生成は、終端記号で止まらない微調整前のモデルで
  計測する(上限 32 トークンまで生成するので、微調整後より長く、安全側)。
- **データの生成と符号化**: 学習データの事例数 $N_{\max}/16, N_{\max}/8, N_{\max}/4$ で計測し、$N_{\max}$ に外挿する。
- **P0 の評価・ハーネスの確認・スケーリングの計測自体**: 実測値をそのまま加える(スモークテストでも本番と同じ量を
  処理するため)。

外挿値に本番での実行回数を乗じ、**削る段階ごとに** 1 セッション全体の見積もりを合計する(6.4 節)。実験 A・B は
条件 1〜4 × $S_{\mathrm{AB}}$ 回(条件 1・2 は短い水準、条件 3・4 は長い水準。条件 1 は未見テンプレートの生成を加える)、
実験 C は C-$N_{\max}$ を条件 1 と共有するので $(|\mathcal{N}| - 1) \times S_{\mathrm{C}}$ 回(短い水準)の学習と評価である。
スモークテストでは、本番の見積もり(スモークテストを実行したデバイスでの外挿値)で段階の選択のコードを動かし、
あわせて予算を人為的に小さくした場合に「段階 0 以外が選ばれる」ことと「どの段階でも超えて停止する」ことを確かめる
(予算の定数は変えず、選択の関数に小さい予算を渡して確かめる)。


```python
def fit_and_extrapolate(label: str, sizes, times, target: float) -> float:
    fit = fit_power_law_exponent(sizes, times)
    extrapolated = fit.coefficient * target**fit.exponent
    detail = ", ".join(f"n={n}: {t:.2f}s" for n, t in zip(sizes, times, strict=True))
    print(
        f"[{label}] {detail} -> b={fit.exponent:.3f}(標準誤差 {fit.exponent_stderr:.3f}), "
        f"R^2={fit.r_squared:.4f}, n={target:,.0f} での外挿値 {extrapolated:.1f}s"
    )
    return extrapolated


_t0_scaling = time.time()
_prod = LEVELS["prod"]
_prod_steps = _prod["NUM_STEPS"]
_prod_eval_size = _prod["NUM_EVAL_INPUTS"] * TASK_COUNT
_timing = build_dataset(TIMING_DATA_SEED, 64, 4096)  # 計測専用(256 事例の評価集合)
_timing_encoded = {long: encode_examples(_timing["train"], long) for long in (False, True)}
_timing_eval = {long: encode_examples(_timing["eval"], long) for long in (False, True)}

# --- 学習(1 回あたり) ---
train_condition(1, 0, 256, num_steps=4, examples=_timing_encoded)  # 初回のオーバーヘッドを除く
TRAIN_ESTIMATE = {}
_step_sizes = [_prod_steps // 16, _prod_steps // 8, _prod_steps // 4]
for _long, _condition in ((False, 1), (True, 3)):
    _times = [
        timed_call(
            lambda n=n, c=_condition: train_condition(
                c, 0, len(_timing_encoded[False]), num_steps=n, examples=_timing_encoded
            )
        )
        for n in _step_sizes
    ]
    _fit = fit_and_extrapolate(
        f"学習({'長い' if _long else '短い'}水準)", _step_sizes, _times, _prod_steps
    )
    TRAIN_ESTIMATE[_long] = max(_fit, 4 * _times[-1])

# --- 教師強制の評価・生成(微調整前のモデル、計測専用のデータ) ---
NLL_ESTIMATE, GENERATION_ESTIMATE = {}, {}
_eval_sizes = [64, 128, 256]
for _long in (False, True):
    _label = "長い" if _long else "短い"
    NLL_ESTIMATE[_long] = fit_and_extrapolate(
        f"教師強制の評価({_label}水準)",
        _eval_sizes,
        [
            timed_call(
                lambda n=n, lg=_long: evaluate_instruction_negative_log_likelihood(
                    base_model, _timing_eval[lg][:n], device, EVAL_BATCH_SIZE
                )
            )
            for n in _eval_sizes
        ],
        _prod_eval_size,
    )
    GENERATION_ESTIMATE[_long] = fit_and_extrapolate(
        f"生成({_label}水準)",
        _eval_sizes,
        [
            timed_call(
                lambda n=n, lg=_long: greedy_generate_until_stop(
                    base_model,
                    [list(x.token_ids[: x.prompt_length]) for x in _timing_eval[lg][:n]],
                    tokenizer.decode,
                    END_MARKER,
                    MAX_NEW_TOKENS,
                    device,
                    GENERATION_BATCH_SIZE,
                )
            )
            for n in _eval_sizes
        ],
        _prod_eval_size,
    )

# --- データの生成と符号化 ---
_data_sizes = [PROD_N_MAX // 16, PROD_N_MAX // 8, PROD_N_MAX // 4]
DATA_ESTIMATE = fit_and_extrapolate(
    "データの生成と符号化(学習データ、両水準)",
    _data_sizes,
    [
        timed_call(
            lambda n=n: [
                encode_examples(build_dataset(TIMING_DATA_SEED, 16, n)["train"], lg)
                for lg in (False, True)
            ]
        )
        for n in _data_sizes
    ],
    PROD_N_MAX,
)
SCALING_SECONDS = time.time() - _t0_scaling
print(f"スケーリングの計測自体: {SCALING_SECONDS:.1f}s")
```

    [学習(短い水準)] n=128: 3.86s, n=256: 7.51s, n=512: 14.48s -> b=0.953(標準誤差 0.003), R^2=1.0000, n=2,048 での外挿値 54.4s
    [学習(長い水準)] n=128: 10.13s, n=256: 20.93s, n=512: 43.63s -> b=1.053(標準誤差 0.004), R^2=1.0000, n=2,048 での外挿値 187.6s
    [教師強制の評価(短い水準)] n=64: 0.02s, n=128: 0.05s, n=256: 0.10s -> b=1.003(標準誤差 0.009), R^2=0.9999, n=1,000 での外挿値 0.4s
    [生成(短い水準)] n=64: 2.68s, n=128: 3.58s, n=256: 3.65s -> b=0.223(標準誤差 0.112), R^2=0.7969, n=1,000 での外挿値 5.2s
    [教師強制の評価(長い水準)] n=64: 0.09s, n=128: 0.16s, n=256: 0.28s -> b=0.845(標準誤差 0.035), R^2=0.9983, n=1,000 での外挿値 0.9s
    [生成(長い水準)] n=64: 3.32s, n=128: 4.56s, n=256: 4.64s -> b=0.243(標準誤差 0.124), R^2=0.7920, n=1,000 での外挿値 6.8s
    [データの生成と符号化(学習データ、両水準)] n=4096: 1.02s, n=8192: 1.91s, n=16384: 4.96s -> b=1.140(標準誤差 0.137), R^2=0.9858, n=65,536 での外挿値 22.8s
    スケーリングの計測自体: 133.0s


### 6.4 共通の前提条件 P0 と、微調整前のモデルの評価

**P0**: 008 と同じ分割(英語版 Wikipedia のコーパスの末尾 5% が検証)・同じ評価(長さ 256 の重ならない窓の
bits-per-byte)で、読み込んだ重みを評価し、モデルカードの値 1.668067 と相対誤差 1% 以内であることを確かめる。

あわせて、微調整前のモデルを評価集合(短い水準・長い水準)で評価する。短い水準の完全一致率は前提条件 P-A に、
両水準の応答部分の負の対数尤度は前提条件 P-L の基準値($L_0^{\mathrm{short}}$・$L_0^{\mathrm{long}}$)に使う。

最後に、6.3 節の外挿値と本節の実測値から、削る段階ごとに 1 セッション全体の見積もりを出し、予算(T4 で 120 分)に
収まる最小の段階を選ぶ(6.1 節)。**選択は見積もりのみに基づき、どの実験の結果も参照しない。** 段階 3 でも予算を
超える場合は、ここで例外で停止する(本番の学習は始まっていない)。


```python
_t0_p0 = time.time()
_corpus_text, _corpus_metadata = load_wikipedia_corpus_with_fallback(
    "en", CORPUS_REPO_ID, WIKIPEDIA_CACHE_DIR, manifest_path=MANIFEST_PATH, return_metadata=True
)
assert len(_corpus_text.encode("utf-8")) == _corpus_metadata["raw_bytes"], (
    "コーパスの取得が破損している"
)
_, _validation_text = split_train_val_text(_corpus_text, P0_VALIDATION_RATIO)
_p0_ids = torch.tensor(tokenizer.encode(_validation_text), dtype=torch.long)
_p0_windows, _p0_mask = make_evaluation_windows(_p0_ids, CONTEXT_LENGTH)
P0_BITS_PER_BYTE = evaluate_bits_per_byte(
    base_model, _p0_windows, _p0_mask, len(_validation_text.encode("utf-8")), device
)
P0_SECONDS = time.time() - _t0_p0
P0_RELATIVE_ERROR = abs(P0_BITS_PER_BYTE - P0_REFERENCE_BITS_PER_BYTE) / P0_REFERENCE_BITS_PER_BYTE
precondition_status["P0"] = bool(P0_RELATIVE_ERROR <= P0_RELATIVE_TOLERANCE)
print(
    f"P0: 検証 bits-per-byte = {P0_BITS_PER_BYTE:.6f}(取得元 {_corpus_metadata['source']}、窓 {len(_p0_windows)} 個)、"
    f"モデルカードの値 {P0_REFERENCE_BITS_PER_BYTE}、相対誤差 {P0_RELATIVE_ERROR:.2e}"
    f"(許容 {P0_RELATIVE_TOLERANCE})-> {'成立' if precondition_status['P0'] else '不成立'}({P0_SECONDS:.1f}s)"
)
del _corpus_text, _validation_text, _p0_ids, _p0_windows, _p0_mask

# --- 微調整前のモデルの評価(P-A・P-L の基準値) ---
_t0 = time.time()
BASE_EVALUATION = {long: evaluate_model(base_model, long) for long in (False, True)}
BASE_EVAL_SECONDS = time.time() - _t0
BASE_RESPONSE_LOSS = {long: BASE_EVALUATION[long]["response_loss"] for long in (False, True)}
BASE_EXACT_MATCH = float(BASE_EVALUATION[False]["exact"].mean())
BASE_FORMAT_RATE = float(BASE_EVALUATION[False]["format_ok"].mean())
precondition_status["P-A"] = bool(BASE_EXACT_MATCH < C_STAR)
print(
    f"微調整前のモデル: 応答部分の負の対数尤度 L0(短い水準) = {BASE_RESPONSE_LOSS[False]:.4f}、"
    f"L0(長い水準) = {BASE_RESPONSE_LOSS[True]:.4f}、形式の遵守率(短い水準) = "
    f"{BASE_EVALUATION[False]['format_ok'].mean():.4f}、完全一致率(短い水準) = {BASE_EXACT_MATCH:.4f}"
)
print(
    f"P-A(微調整前の完全一致率 {BASE_EXACT_MATCH:.4f} < c* = {C_STAR:.4f}): "
    f"{'成立' if precondition_status['P-A'] else '不成立'}({BASE_EVAL_SECONDS:.1f}s)"
)

# --- 削る段階ごとの 1 セッション全体の見積もり(本番の実行回数を乗じた値、6.3 節) ---
_eval_short = NLL_ESTIMATE[False] + GENERATION_ESTIMATE[False]
_eval_long = NLL_ESTIMATE[True] + GENERATION_ESTIMATE[True]
COMMON_ESTIMATE = {
    "データの生成と符号化(外挿)": DATA_ESTIMATE,
    "P0(コーパスの取得・符号化・評価、実測)": P0_SECONDS,
    "微調整前のモデルの評価(2 水準、外挿)": _eval_short + _eval_long,
    "スケーリングの計測自体(実測)": SCALING_SECONDS,
    "生成・採点・損失のハーネスの確認(5.5 節、実測)": HARNESS_SECONDS,
}


def estimate_stage(stage: dict) -> dict:
    # 本番の水準(PROD_N_MAX)での、1 セッション全体の見積もり(秒)。
    seeds_ab, seeds_c = stage["NUM_SEEDS_AB"], stage["NUM_SEEDS_C"]
    levels = c_levels(stage["C_MIN_LEVEL"], PROD_N_MAX)
    return {
        **COMMON_ESTIMATE,
        f"実験 A・B の学習(条件 1〜4 x {seeds_ab} シード)": seeds_ab
        * (2 * TRAIN_ESTIMATE[False] + 2 * TRAIN_ESTIMATE[True]),
        f"実験 A・B の評価(条件 1〜4 x {seeds_ab} シード、条件 1 は未見テンプレートの生成を加える)": seeds_ab
        * (2 * _eval_short + 2 * _eval_long + GENERATION_ESTIMATE[False]),
        f"実験 C の学習と評価(({len(levels)} - 1) 水準 x {seeds_c} シード、C-N_max は条件 1 と共有)": (
            len(levels) - 1
        )
        * seeds_c
        * (TRAIN_ESTIMATE[False] + _eval_short),
    }


class StageBudgetExceededError(RuntimeError):
    pass


def select_stage(totals: dict[int, float], budget_seconds: float) -> int:
    # 予算以内に収まる最小の段階を返す。どの段階でも超える場合は例外(見積もりのみに基づく、6.1 節)。
    for stage in sorted(totals):
        if totals[stage] <= budget_seconds:
            return stage
    raise StageBudgetExceededError(
        f"段階 {max(totals)} でも見積もり {totals[max(totals)] / 60:.1f} 分が予算 {budget_seconds / 60:.1f} 分を超える"
    )


STAGE_ESTIMATES = {k: estimate_stage(v) for k, v in STAGES["prod"].items()}
STAGE_TOTALS = {k: sum(v.values()) for k, v in STAGE_ESTIMATES.items()}
print(f"\n--- 段階 0 の本番の見積もりの内訳({device} 基準)---")
for _k, _v in STAGE_ESTIMATES[0].items():
    print(f"  {_k}: {_v:,.1f}s")
print(
    f"\n--- 削る段階ごとの 1 セッション全体の見積もり({device} 基準、予算 {SESSION_BUDGET_SECONDS / 60:.0f} 分)---"
)
for _k, _total in STAGE_TOTALS.items():
    _stage = STAGES["prod"][_k]
    print(
        f"  段階 {_k}(実験 C の水準 {c_levels(_stage['C_MIN_LEVEL'], PROD_N_MAX)}、S_C = {_stage['NUM_SEEDS_C']}、"
        f"S_AB = {_stage['NUM_SEEDS_AB']}): {_total:,.1f}s = {_total / 60:.1f} 分"
        f"(予算の {_total / SESSION_BUDGET_SECONDS:.1%})"
    )
assert all(STAGE_TOTALS[k] >= STAGE_TOTALS[k + 1] for k in (0, 1, 2)), (
    "段階が上がると見積もりが減るはず"
)

# --- 段階の選択(見積もりのみに基づく。本番の学習はまだ 1 回も行っていない) ---
try:
    SELECTED_STAGE = select_stage(STAGE_TOTALS, SESSION_BUDGET_SECONDS)
    STAGE_SELECTION_MESSAGE = (
        f"予算 {SESSION_BUDGET_SECONDS / 60:.0f} 分に収まる最小の段階として、段階 {SELECTED_STAGE} を選んだ"
        f"(見積もり {STAGE_TOTALS[SELECTED_STAGE] / 60:.1f} 分、{device} 基準)"
    )
except StageBudgetExceededError as _error:
    if not SMOKE_TEST:
        print(f"\n警告: {_error}。本番の学習を始める前に停止する。計画を見直すこと。")
        raise
    SELECTED_STAGE = max(STAGE_TOTALS)
    STAGE_SELECTION_MESSAGE = f"{_error}(本番なら学習の前に停止する)。スモークテストのため停止せず、段階 {SELECTED_STAGE} で動作確認を続ける"
if FORCED_STAGE is not None:  # テスト専用の上書き(5.2 節。スモークテストでのみ有効)
    assert SMOKE_TEST
    print("\n" + "!" * 100)
    print(
        f"!!! テスト専用の上書き: AI_THEORIES_FORCE_STAGE={FORCED_STAGE} により、見積もりによる選択(段階 {SELECTED_STAGE})の"
        f"代わりに段階 {FORCED_STAGE} を使う(スモークテストのみ。この出力は通常の実行の記録ではない)"
    )
    print("!" * 100)
    STAGE_SELECTION_MESSAGE = (
        f"テスト専用の上書きで段階 {FORCED_STAGE} を強制(見積もりによる選択は段階 {SELECTED_STAGE})"
    )
    SELECTED_STAGE = FORCED_STAGE
print(f"\n段階の選択: {STAGE_SELECTION_MESSAGE}")

# --- スモークテストのみ: 予算を人為的に小さくして、選択の規則の分岐を確かめる(予算の定数は変えない) ---
if SMOKE_TEST:
    _budget_between = (STAGE_TOTALS[0] + STAGE_TOTALS[1]) / 2  # 段階 0 は超え、段階 1 は収まる予算
    assert select_stage(STAGE_TOTALS, _budget_between) == 1
    _budget_too_small = 0.5 * STAGE_TOTALS[3]  # どの段階でも超える予算
    try:
        select_stage(STAGE_TOTALS, _budget_too_small)
        raise AssertionError("どの段階でも超える予算で停止しなかった")
    except StageBudgetExceededError as _error:
        _stopped = str(_error)
    print(
        f"選択の規則の確認(人為的に小さい予算、確認のみ): 予算 {_budget_between / 60:.1f} 分 -> 段階 1 が選ばれる: OK / "
        f"予算 {_budget_too_small / 60:.1f} 分 -> 停止する({_stopped}): OK。予算の定数は {SESSION_BUDGET_SECONDS / 60:.0f} 分のまま"
    )
    assert SESSION_BUDGET_SECONDS == 120 * 60

# --- 選ばれた段階の水準・シード数(以降のすべてのセルがこれを使う) ---
STAGE = STAGES[CURRENT_LEVEL_NAME][SELECTED_STAGE]
NUM_SEEDS_AB, NUM_SEEDS_C = STAGE["NUM_SEEDS_AB"], STAGE["NUM_SEEDS_C"]
SEEDS_AB = tuple(range(NUM_SEEDS_AB))
SEEDS_C = tuple(range(NUM_SEEDS_C))
C_LEVELS = c_levels(STAGE["C_MIN_LEVEL"], N_MAX)
assert set(SEEDS_C) <= set(SEEDS_AB) and C_LEVELS[-1] == N_MAX
print(
    f"このノートブックで使う値(段階 {SELECTED_STAGE}、水準 {CURRENT_LEVEL_NAME!r}): 実験 A・B のシード {SEEDS_AB}、"
    f"実験 C のシード {SEEDS_C}、実験 C の水準 {C_LEVELS}"
)
if device.type != "cuda":
    print("注意: CUDA 以外での見積もりであり、T4 での時間とは異なる")
```


    corpus.txt: reconstructing file:   0%|          |  0.00B / 24.3MB            



    corpus.txt: downloading bytes:           |  0.00B            



    metadata.json:   0%|          | 0.00/43.4k [00:00<?, ?B/s]


    コーパス取得元: kojikojiprg/ai-theories-corpus-en-pretraining(Hugging Face Hub)
    P0: 検証 bits-per-byte = 1.668067(取得元 hub、窓 1240 個)、モデルカードの値 1.668067、相対誤差 1.57e-07(許容 0.01)-> 成立(6.6s)
    微調整前のモデル: 応答部分の負の対数尤度 L0(短い水準) = 8.2724、L0(長い水準) = 8.5841、形式の遵守率(短い水準) = 0.0000、完全一致率(短い水準) = 0.0000
    P-A(微調整前の完全一致率 0.0000 < c* = 0.2500): 成立(14.2s)
    
    --- 段階 0 の本番の見積もりの内訳(cuda 基準)---
      データの生成と符号化(外挿): 22.8s
      P0(コーパスの取得・符号化・評価、実測): 6.6s
      微調整前のモデルの評価(2 水準、外挿): 13.2s
      スケーリングの計測自体(実測): 133.0s
      生成・採点・損失のハーネスの確認(5.5 節、実測): 49.9s
      実験 A・B の学習(条件 1〜4 x 5 シード): 2,455.3s
      実験 A・B の評価(条件 1〜4 x 5 シード、条件 1 は未見テンプレートの生成を加える): 158.3s
      実験 C の学習と評価((7 - 1) 水準 x 5 シード、C-N_max は条件 1 と共有): 1,904.6s
    
    --- 削る段階ごとの 1 セッション全体の見積もり(cuda 基準、予算 120 分)---
      段階 0(実験 C の水準 (16, 64, 256, 1024, 4096, 16384, 65536)、S_C = 5、S_AB = 5): 4,743.9s = 79.1 分(予算の 65.9%)
      段階 1(実験 C の水準 (64, 256, 1024, 4096, 16384, 65536)、S_C = 5、S_AB = 5): 4,426.4s = 73.8 分(予算の 61.5%)
      段階 2(実験 C の水準 (64, 256, 1024, 4096, 16384, 65536)、S_C = 3、S_AB = 5): 3,791.6s = 63.2 分(予算の 52.7%)
      段階 3(実験 C の水準 (64, 256, 1024, 4096, 16384, 65536)、S_C = 3、S_AB = 3): 2,746.1s = 45.8 分(予算の 38.1%)
    
    段階の選択: 予算 120 分に収まる最小の段階として、段階 0 を選んだ(見積もり 79.1 分、cuda 基準)
    このノートブックで使う値(段階 0、水準 'prod'): 実験 A・B のシード (0, 1, 2, 3, 4)、実験 C のシード (0, 1, 2, 3, 4)、実験 C の水準 (16, 64, 256, 1024, 4096, 16384, 65536)




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/04_alignment/016_supervised_fine_tuning.ipynb
