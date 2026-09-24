---
title: "量子化の基礎(Quantization Basics)(実装・実験編 2/5)"
---

この記事は後編(実装・実験編 2/5)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/013_quantization_basics-practice-1)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/013_quantization_basics-practice-3)。

### 5.8 量子化したモデルの構築・評価のヘルパーと、キャッシュの一致

- `build_quantized_model(cid)`: 基盤モデルの`deepcopy`の`blocks`内の`nn.Linear`を、キャッシュした量子化済みの
  重みから作った`QuantizedLinear`に置き換える。
- `per_window_nll(model, windows, mask)`: 評価窓ごとの負の対数尤度(nats、窓内の予測対象トークンの和)。
  `evaluate_bits_per_byte()`と同じ規則(先頭トークンは予測しない、パディングは除外)で計算し、総和から
  求めた bits-per-byte が`evaluate_bits_per_byte()`と一致することを確認する。
- **キャッシュの一致**: キャッシュを使わず`quantize_linear_layers()`でその場で量子化したモデルと、
  キャッシュから作ったモデルで、全 buffer が`torch.equal`で一致し、Tiny Shakespeare の評価集合の
  bits-per-byte が完全に一致することを 6 条件すべてで確認する。
- **量子化対象外の重みが FP32 のまま不変であること**: 埋め込み行列・RMSNorm の重みが FP32 で、
  基盤モデルと bit 単位で一致することを 6 条件すべてで確認する。


```python
def build_quantized_model(cid: str) -> GPTLanguageModel:
    model = copy.deepcopy(base_model)
    if QUANT_CONDITIONS[cid] is not None:
        replaced = quantize_linear_layers(model.blocks, precomputed=QUANTIZED_CACHE[cid])
        assert replaced == TARGET_NAMES
    return model.eval()


@torch.no_grad()
def per_window_nll(
    model: nn.Module, windows: torch.Tensor, mask: torch.Tensor, batch_size: int = 16
) -> np.ndarray:
    # 評価窓ごとの負の対数尤度の和(nats)。evaluate_bits_per_byte と同じ規則で計算する。
    model.eval()
    sums = []
    for start in range(0, windows.size(0), batch_size):
        batch = windows[start : start + batch_size].to(device)
        batch_mask = mask[start : start + batch_size, 1:].to(device)
        logits = model(batch)[:, :-1, :]
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1), reduction="none"
        ).view(batch_mask.shape)
        sums.append((nll * batch_mask).sum(dim=1).cpu().double())
    return torch.cat(sums).numpy()


def bits_per_byte_from_nll(nll: np.ndarray, total_bytes: int) -> float:
    return float(nll.sum() / (LN2 * total_bytes))


def check_non_target_unchanged(model: nn.Module) -> int:
    # 量子化対象外のパラメータが FP32 のまま、基盤モデルと bit 単位で一致することを確認する。
    base_params = dict(base_model.named_parameters())
    params = dict(model.named_parameters())
    for name in NON_TARGET_PARAMETER_NAMES:
        assert params[name].dtype == torch.float32, name
        assert torch.equal(params[name], base_params[name]), name
    return len(NON_TARGET_PARAMETER_NAMES)


_probe = per_window_nll(base_model, evaluation_windows, evaluation_mask)
_direct = evaluate_bits_per_byte(
    base_model, evaluation_windows, evaluation_mask, evaluation_bytes, device
)
assert math.isclose(bits_per_byte_from_nll(_probe, evaluation_bytes), _direct, rel_tol=1e-6)
print(
    f"per_window_nll の総和による bits-per-byte {bits_per_byte_from_nll(_probe, evaluation_bytes):.8f} == "
    f"evaluate_bits_per_byte {_direct:.8f}(相対誤差 1e-6 以内): OK"
)

cache_check = {}
for _cid, _cfg in QUANT_CONDITIONS.items():
    if _cfg is None:
        continue
    _cached = build_quantized_model(_cid)
    _fresh = copy.deepcopy(base_model)
    quantize_linear_layers(_fresh.blocks, **_cfg)  # キャッシュを使わずその場で量子化する
    _s1, _s2 = _cached.state_dict(), _fresh.state_dict()
    assert _s1.keys() == _s2.keys() and all(torch.equal(_s1[k], _s2[k]) for k in _s1), _cid
    _b1 = evaluate_bits_per_byte(
        _cached, evaluation_windows, evaluation_mask, evaluation_bytes, device
    )
    _b2 = evaluate_bits_per_byte(
        _fresh, evaluation_windows, evaluation_mask, evaluation_bytes, device
    )
    assert _b1 == _b2, (_cid, _b1, _b2)
    _n_checked = check_non_target_unchanged(_cached)
    assert all(isinstance(_cached.blocks.get_submodule(n), QuantizedLinear) for n in TARGET_NAMES)
    assert not any(p.requires_grad for p in _cached.parameters())
    cache_check[_cid] = _b1
    del _cached, _fresh
print("キャッシュ前後の一致(全 buffer と Tiny Shakespeare の評価集合の bits-per-byte、6 条件): OK")
print(f"量子化対象外の {_n_checked} テンソルが FP32 のまま基盤モデルと bit 単位で一致(6 条件): OK")
assert hash_state(base_model.state_dict()) == BASE_STATE_HASH
```

    per_window_nll の総和による bits-per-byte 3.39275088 == evaluate_bits_per_byte 3.39275088(相対誤差 1e-6 以内): OK
    キャッシュ前後の一致(全 buffer と Tiny Shakespeare の評価集合の bits-per-byte、6 条件): OK
    量子化対象外の 10 テンソルが FP32 のまま基盤モデルと bit 単位で一致(6 条件): OK


### 5.9 QLoRA(実験 D)の学習・評価ヘルパー

1 回の微調整(1 条件 × 1 シード)を実行し、判定と診断に必要な量を記録して返す。

- 条件 1(`"fp32"`)は基盤モデルの`deepcopy`、条件 2(`"nf4"`)は NF4(ブロック 64 + 二重量子化)の
  キャッシュから作ったモデルに、`torch.manual_seed(seed)`の直後に`apply_lora()`で LoRA を掛ける
  (012 と同じ手順)。LoRA の $A$ は、両条件とも同じ device 上で同じ乱数から初期化される
  (5.9 節の末尾で、シード 0 の $A$ が両条件で完全に一致することを確認する)。
- 学習は`train_language_model()`、学習率・gradient clipping の閾値は 012 の較正結果(5.2 節)を使う。
- 訓練損失は 012 と同じ固定の 64 窓で、学習の前(その条件の基盤モデル)と後に同じ関数で測る。
- 学習後のモデルの評価集合の窓ごとの負の対数尤度を記録する(実験 D の新基準のブートストラップに使う)。
  その総和から求めた bits-per-byte が最終 bits-per-byte と一致することを確認する。
- 学習後に、LoRA 以外の全 state(パラメータと量子化した重みの buffer)が学習前と bit 単位で一致すること、
  凍結したパラメータの`.grad`が`None`であることを確認する。
- CUDA 環境では、学習中の最大 GPU メモリ(`torch.cuda.max_memory_allocated`)を記録する。


```python
def mean_loss_nats(model: nn.Module, windows: torch.Tensor, batch_size: int = 16) -> float:
    # 固定の窓(パディングなし)での平均の次トークン予測損失(nats / トークン、012 と同じ)。
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, windows.size(0), batch_size):
            batch = windows[start : start + batch_size].to(device)
            logits = model(batch)[:, :-1, :]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1), reduction="sum"
            )
            total += loss.item()
            count += batch[:, 1:].numel()
    return total / count


FINETUNE_BASE = {"fp32": "q0", "nf4": "q6"}  # 実験 D の条件 1・2 の基盤
INITIAL_TRAIN_LOSS = {
    cond: mean_loss_nats(build_quantized_model(cid), train_loss_windows)
    for cond, cid in FINETUNE_BASE.items()
}
print(f"学習開始時の訓練損失(固定の訓練窓、nats / トークン): {INITIAL_TRAIN_LOSS}")


def build_finetune_model(condition: str, seed: int) -> tuple[nn.Module, list[str]]:
    model = build_quantized_model(FINETUNE_BASE[condition])
    torch.manual_seed(seed)  # A の初期化(012 と同じ)
    replaced = apply_lora(model, LORA_TARGET_MODULES, rank=LORA_RANK, alpha=LORA_ALPHA)
    return model, replaced


def frozen_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if "lora_" not in k}


def run_finetune(
    condition: str, seed: int, num_steps: int | None = None, measure: bool = True
) -> dict:
    num_steps = NUM_STEPS if num_steps is None else num_steps
    model, replaced = build_finetune_model(condition, seed)
    frozen_before = hash_state(frozen_state(model))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=LORA_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    schedule = functools.partial(
        compute_warmup_cosine_learning_rate,
        warmup_steps=max(1, round(WARMUP_RATIO * num_steps)),
        total_steps=num_steps,
        peak_learning_rate=LORA_LEARNING_RATE,
        min_learning_rate=LORA_LEARNING_RATE * MIN_LEARNING_RATE_RATIO,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history = train_language_model(
        model,
        train_ids,
        evaluation_windows,
        evaluation_mask,
        evaluation_bytes,
        num_steps=num_steps,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        learning_rate=LORA_LEARNING_RATE,
        eval_interval=max(1, num_steps // EVAL_POINTS),
        device=device,
        seed=seed,
        optimizer=optimizer,
        learning_rate_schedule=schedule,
        gradient_clip_threshold=LORA_CLIP_THRESHOLD,
    )
    peak_memory = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
    if not history["eval_step"] or history["eval_step"][-1] != num_steps:
        history["eval_step"].append(num_steps)
        history["eval_bits_per_byte"].append(
            evaluate_bits_per_byte(
                model, evaluation_windows, evaluation_mask, evaluation_bytes, device
            )
        )
    # LoRA 以外の全 state が学習前と bit 単位で一致し、凍結パラメータの .grad が None であること
    assert hash_state(frozen_state(model)) == frozen_before, "凍結した state が学習中に変化した"
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    record = {
        "condition": condition,
        "seed": seed,
        "num_steps": num_steps,
        "history_length": len(history["step"]),
        "num_trainable_parameters": sum(p.numel() for p in trainable),
        "replaced": replaced,
        "evaluation_windows_hash": hash_tensor(evaluation_windows),
        "evaluation_denominator_bytes": evaluation_bytes,
        "eval_step": list(history["eval_step"]),
        "eval_bits_per_byte": list(history["eval_bits_per_byte"]),
        "final_bits_per_byte": history["eval_bits_per_byte"][-1],
        "clip_trigger_ratio": float(np.mean(history["gradient_clip_triggered"])),
        "peak_gpu_memory_bytes": peak_memory,
        "base_storage_bytes": storage_table[FINETUNE_BASE[condition]],
    }
    if measure:
        record["initial_train_loss"] = INITIAL_TRAIN_LOSS[condition]
        record["final_train_loss"] = mean_loss_nats(model, train_loss_windows)
        # 評価窓ごとの負の対数尤度(実験 D の新基準のブートストラップ・判定の一次情報)
        window_nll = per_window_nll(model, evaluation_windows, evaluation_mask)
        assert math.isclose(
            bits_per_byte_from_nll(window_nll, evaluation_bytes),
            record["final_bits_per_byte"],
            rel_tol=1e-6,
        ), "窓ごとの負の対数尤度の総和が最終 bits-per-byte と一致しない"
        record["evaluation_window_nll"] = window_nll.tolist()
    del model
    return record


# LoRA の A の初期化が両条件で一致すること(シード 0)、LoRA の置換先・パラメータ数が一致すること
_m1, _r1 = build_finetune_model("fp32", 0)
_m2, _r2 = build_finetune_model("nf4", 0)
assert _r1 == _r2 and len(_r1) == 2 * NUM_LAYERS
assert all(isinstance(_m2.get_submodule(n).base_layer, QuantizedLinear) for n in _r2)
assert all(isinstance(_m1.get_submodule(n).base_layer, nn.Linear) for n in _r1)
for _n in _r1:
    assert torch.equal(_m1.get_submodule(_n).lora_a, _m2.get_submodule(_n).lora_a), _n
    assert _m2.get_submodule(_n).lora_a.device.type == device.type
_lora_params = sum(p.numel() for p in _m1.parameters() if p.requires_grad)
assert _lora_params == sum(p.numel() for p in _m2.parameters() if p.requires_grad)
assert _lora_params == 2 * NUM_LAYERS * compute_lora_parameter_count(D_MODEL, D_MODEL, LORA_RANK)
with torch.no_grad():  # B = 0 なので、LoRA を掛けた直後の出力は各条件の基盤モデルと一致する
    _probe_tokens = evaluation_windows[:2].to(device)
    assert torch.equal(_m2(_probe_tokens), build_quantized_model("q6")(_probe_tokens))
del _m1, _m2
print(
    f"LoRA: 置換 {len(_r1)} 行列、学習可能パラメータ数 {_lora_params:,}(両条件で同一)、"
    "シード 0 の A の初期化が両条件で完全一致、NF4 の基盤に掛けた直後の出力 == NF4 の基盤: OK"
)

_t0 = time.time()
_smoke_record = run_finetune("nf4", seed=0, num_steps=2)
print(
    f"run_finetune の動作確認(NF4 + LoRA、2 ステップ): {time.time() - _t0:.2f}s, "
    f"final_bits_per_byte={_smoke_record['final_bits_per_byte']:.4f}"
)
```

    学習開始時の訓練損失(固定の訓練窓、nats / トークン): {'fp32': 6.246554385914522, 'nf4': 6.255347996132047}
    LoRA: 置換 8 行列、学習可能パラメータ数 32,768(両条件で同一)、シード 0 の A の初期化が両条件で完全一致、NF4 の基盤に掛けた直後の出力 == NF4 の基盤: OK
    run_finetune の動作確認(NF4 + LoRA、2 ステップ): 1.11s, final_bits_per_byte=3.2824


## 6. 実験 / Experiments

### 6.1 実験宣言セル: 共通の設定・検証すること・判定基準・前提条件

**この節の内容は本番実行の前に確定させ、結果を見た後に変更しない。**

#### 共通の設定

- **起点**: 基盤モデルは`kojikojiprg/ai-theories-small-gpt-en`の`main`ブランチ(008 で事前学習)、
  トークナイザは`kojikojiprg/ai-theories-tokenizer-en`。
- **量子化の対象**: 各 Decoder Block 内の`nn.Linear`の重み(5.4 節で数えた 28 行列: 形状
  $(256, 256)$ の注意機構の射影 16 行列、形状 $(683, 256)$ の SwiGLU の 2 つの入力側の行列 8 行列、
  形状 $(256, 683)$ の出力側の行列 4 行列)。埋め込み行列(出力層と共有)と RMSNorm の重みは FP32 のまま。
- **再構成誤差**: 行列 $m$ の重み $W_m$($N_m$ 要素)と逆量子化した $\hat{W}_m$ の平均二乗誤差
  $\mathrm{err}_m = \frac{1}{N_m} \sum_i (w_i - \hat{w}_i)^2$(FP64 で集計)。
- **量子化の条件の記号**:

| 記号 | 条件 |
|---|---|
| q0 | FP32(量子化なし) |
| q1 | INT8、absmax 方式、チャネル単位 |
| q2 | INT8、absmax 方式、テンソル単位 |
| q3 | INT4、absmax 方式、テンソル単位 |
| q4 | INT4、absmax 方式、ブロック 64 |
| q5 | NF4、ブロック 64(スケールは FP32) |
| q6 | NF4、ブロック 64 + 二重量子化 |

- **決定的な実験についての記録**: 実験 A〜C の対比量と、実験 D の $\Delta_{\mathrm{before}}$ は、固定された重みと
  評価集合だけから決まる決定的な値である。`SMOKE_TEST`で変わるのは、実験 B のブートストラップの反復回数
  (と、実験 D の $\sigma_{\mathrm{before}}$ などのブートストラップの反復回数)のみである。そのため、第 1 段階の
  スモークテストの出力は、実験 A〜C の対比量と $\Delta_{\mathrm{before}}$ については本番の値と実質的に同じである。
  実験 A〜C の判定基準は、第 1 段階の最初の実行より前に宣言したものであり、その後変更していない。本番実行では、
  実験 A・C の $\bar{r}$・$\bar{\ell}$ がスモークテストの出力から転記した値と一致するかを診断量として印字する
  (実行環境の違いで一致しない場合は、判定ではなく差として報告する)。

#### 前提条件 P0(全実験に共通): 起点モデルの読み込みが正しいこと

- **宣言**: q0(FP32)の英語 Wikipedia の検証集合(008 と同じ分割・非重複窓)での bits-per-byte
  $b_{\mathrm{P0}}$ が、008 のセル出力に記録された値 $b_{008} = 1.6707$ に対して
  $|b_{\mathrm{P0}} - b_{008}| / b_{008} \le 0.01$ を満たすこと。
- **期待値**: 008 の 5.10 節のセル出力(「最終検証 bits-per-byte = 1.6707」)から転記した定数であり、
  本ノートブック内の変数から計算しない。
- **許容誤差の根拠**: 読み込みの誤り(重みの取り違え、位置エンコーディング・正規化の構成の誤り、
  トークナイザの取り違え)は bits-per-byte を大きく(数十 % 以上)変えるので、相対 1% の許容誤差で検出できる。
  完全一致は要求しない。**第 1 段階のローカル実行(CPU・MPS)では、Hub の`main`の重みで 1.6681 を観測した**
  (相対差 0.16%)。010 の本番実行(T4)のセル出力も同じ重み・同じ検証集合で 1.6681 である。Hub の
  リポジトリは 008 のノートブックの実行後に作り直されており、008 のセル出力を生んだ実行と重みが bit 単位で
  同一であることは確認できない。この許容誤差はローカルの観測値を見た後に決めたが、量子化に関する
  仮説のいずれとも独立な量(FP32 の基盤モデルの読み込み)についての条件である。
  **本番実行(Google Colab T4)の値**: $b_{\mathrm{P0}} = 1.668067$、008 の記録値との相対誤差 0.1576%(第 1 段階の
  ローカル実行の観測値と同じ)。この値の追記は判定基準を変更するものではない。
- P0 が成立しない場合、実験 A〜D はすべて「前提不成立」とする。

#### 診断量(実験 A〜C 共通、判定には使わない): 下流の bits-per-byte の増加

条件 q1〜q6 ごとに、英語 Wikipedia の検証集合での bits-per-byte の増加 $b_{q} - b_{q0}$ を印字する。
不確かさは、評価窓(1240 窓)を復元抽出する対応付きブートストラップ(全条件に同じ再標本を使い、
反復ごとに差を取る)の標準偏差と 95% 区間で示す。bits-per-byte は窓ごとの負の対数尤度の和を窓ごとの
UTF-8 バイト数の和で割った比として再標本ごとに計算する。

#### 実験 A: ビット幅と再構成誤差の理論値

**検証すること**: absmax 方式・ブロック 64 の一様量子化で、$b \in \{4, 5, 6, 7, 8\}$ のとき、重みの再構成の
平均二乗誤差の実測値が理論値 $\overline{\Delta^2} / 12$(3.3 節)と一致する。

**対比量**: ビット幅 $b$ ごとに、行列 $m$ の $r_m = \log(\mathrm{measured}_m / \mathrm{predicted}_m)$ の
行列間平均 $\bar{r}$。$\mathrm{measured}_m = \mathrm{err}_m$、$\mathrm{predicted}_m = \overline{\Delta^2}_m / 12$
($\overline{\Delta^2}_m$ は行列 $m$ のブロックごとの $\Delta^2$ の平均)。

**標準誤差の導出**: $\mathrm{SE} = s_r / \sqrt{M}$($s_r$ は $r_m$ の行列間の標本標準偏差(不偏)、
$M = 28$ は行列数)。行列ごとの $r_m$ を独立な測定とみなす。

**判定(同等性の判定)**: 各ビット幅について

- 支持: $[\bar{r} - 2\,\mathrm{SE},\ \bar{r} + 2\,\mathrm{SE}] \subseteq [-\log 1.1,\ \log 1.1]$
- 反証: 区間全体が $[-\log 1.1, \log 1.1]$ の外にある($\bar{r} - 2\,\mathrm{SE} > \log 1.1$ または $\bar{r} + 2\,\mathrm{SE} < -\log 1.1$)
- 判定不能: 上記以外

5 つのビット幅すべてで支持の場合に実験 A を支持とする。1 つでも反証があれば反証、それ以外は判定不能とする。
$\log 1.1$(実測値と理論値の比が 10% 以内)は、結果を見る前に置いた実用上の基準である。

**直接の作用点**: 量子化が直接作用するのは重みそのものであり、対比量(重みの再構成誤差)は直接の作用点に
あたる。

**診断量**: $b \in \{2, 3\}$ の同じ量(低ビットで理論から外れることの観察)。ブロック内の最大値が誤差 0 で
表現されることによる系統的なずれの理論値 $\log(63/64) \approx -0.0157$(3.3 節)と $\bar{r}$ の比較。
隣り合うビット幅の誤差の比の実測値と理論値 $\left((2^b - 1)/(2^{b-1} - 1)\right)^2$ の比較。

**前提条件**: P0 のみ。

#### 実験 B: 粒度の効果と外れ値(尖度)の関係

**検証すること**: INT4(absmax 方式)でテンソル単位からブロック 64 に細かくしたときの誤差低減比
$\rho_m = \mathrm{err}_{\mathrm{tensor},m} / \mathrm{err}_{\mathrm{block64},m}$ が、行列 $m$ の重みの尖度
$\kappa_m$(3.5 節)と正の相関を持つ。

**交絡の分離**: 要素数の多い行列ほど、正規分布でも最大絶対値の期待値が大きくなる(3.5 節)ので、
テンソル単位の刻み幅が広がり $\rho_m$ が大きくなりうる。そのため行列の形状で層別する。形状が同じ行列の群
$g$(5.4 節の 3 群)の内部で $\log \rho_m$ と $\kappa_m$ の Spearman 相関 $\rho^{\mathrm{S}}_g$ を計算し、群の大きさ
$n_g$ で重み付けした平均

$$
\bar{\rho}^{\mathrm{S}} = \frac{\sum_g n_g\, \rho^{\mathrm{S}}_g}{\sum_g n_g}
$$

を対比量とする。

**判定**: 群の内部で行列を復元抽出する(群の大きさは保つ)ブートストラップ(本番 10,000 回、反復回数は
事前に固定)で $\bar{\rho}^{\mathrm{S}}$ の 95% 区間(2.5・97.5 パーセンタイル)を求める。

- 支持: 下限 $> 0$
- 反証: 上限 $< 0$
- 判定不能: 上記以外

再標本で群内の順位の分散が 0 になった群(同じ行列だけが選ばれた場合)の相関は 0 とする
(帰無仮説の側に寄せる保守的な扱い)。

**前提条件 P-B**: 形状の群が 2 つ以上あり、各群の行列数が 3 以上であること(層別した相関が計算できること)。

**直接の作用点**: 粒度が直接作用するのはスケール(刻み幅)であり、再構成誤差の比 $\rho_m$ はその直接の帰結で
ある。尖度は介入ではなく重みの性質であり、この実験は介入の効果の大きさと重みの性質の関係を問う。

**診断量**: ブロックサイズの等比の掃引 $\{16, 32, \dots, 65536\}$ とテンソル単位に対する誤差と、スケールの格納
オーバーヘッドを含めた実効ビット数の曲線。各行列の最大絶対値 / 標準偏差の比。形状の群の内部の行列数が
少ない(4 行列の群がある)ため、Spearman 相関のとりうる値が粗いことに注意する。

#### 実験 C: NF4 と INT4 の再構成誤差

**検証すること**: 同じブロックサイズ 64 で、NF4(q5)の再構成誤差が INT4(absmax 方式・対称、q4)より小さい。

**前提条件 P-C**: NF4 の最適性の前提である重みの正規性。行列ごとの尖度の絶対値 $|\kappa_m|$ の中央値が
1.0 以下であること。閾値の根拠: 正規分布の尖度は 0、ラプラス分布の尖度は 3 であり、1.0 はその間で
正規分布に近い側に置いた値である。これは実験 C の仮説(NF4 と INT4 の誤差の比)とは独立な、重み自体の量である。

**対比量**: 行列ごとの $\ell_m = \log(\mathrm{err}_{\mathrm{NF4},m} / \mathrm{err}_{\mathrm{INT4},m})$ の行列間平均 $\bar{\ell}$。
**標準誤差**: $\mathrm{SE} = s_\ell / \sqrt{M}$($s_\ell$ は行列間の標本標準偏差)。

**判定**: 支持は $\bar{\ell} + 2\,\mathrm{SE} < 0$、反証は $\bar{\ell} - 2\,\mathrm{SE} > 0$、それ以外は判定不能。

**直接の作用点**: データ型(符号語の配置)が直接作用するのは再構成誤差であり、対比量は直接の作用点にあたる。

**診断量**: 二重量子化の有無による誤差の差 $\log(\mathrm{err}_{q6} / \mathrm{err}_{q5})$、下流の bits-per-byte の差
($b_{q5} - b_{q4}$、$b_{q6} - b_{q5}$、上記の診断量の表から)。

#### 実験 D: QLoRA による量子化劣化の縮小

**条件**:

| 記号 | 条件 | シード数 |
|---|---|---|
| 条件 1 | FP32 の基盤(q0)+ LoRA | 5 |
| 条件 2 | NF4(ブロック 64 + 二重量子化、q6)の基盤 + LoRA | 5 |

微調整先(Tiny Shakespeare)・LoRA の設定(Query・Value、$r = 8$、$\alpha = 8$)・学習予算(012 と同じ
$E = 2$ から決まる $T$ ステップ、本番 87 ステップ)・学習率($10^{-1.5}$)・gradient clipping の閾値・シード数は
すべて 012 と同一とし、**条件 2 のために学習率を較正し直さない。** シードは、ミニバッチの順序と $A$ の初期化を
決める(両条件で同じシードは同じ順序・同じ初期値になる)。

**対比量**: 評価集合(Tiny Shakespeare の末尾 5%)の bits-per-byte について

$$
\Delta_{\mathrm{before}} = b_{\mathrm{NF4\ base}} - b_{\mathrm{FP32\ base}}, \qquad
\Delta_{\mathrm{after}} = \bar{b}_2 - \bar{b}_1, \qquad
c = \Delta_{\mathrm{before}} - \Delta_{\mathrm{after}}
$$

$b_{\mathrm{NF4\ base}}$・$b_{\mathrm{FP32\ base}}$ は微調整前の基盤モデルの値(学習を伴わない決定的な値)、
$\bar{b}_1$・$\bar{b}_2$ は微調整後の条件 1・2 のシード平均である。$c > 0$ なら、量子化による劣化が微調整で縮小した。

**標準偏差の導出**: 両条件は同じシード(ミニバッチの順序・$A$ の初期値)と同じ評価窓を共有する、対応付きの設計で
ある。シード $s$ ごとの差を

$$
d_s = b_{2,s} - b_{1,s}
$$

とする($b_{1,s}$・$b_{2,s}$ は同じシード $s$ で学習した条件 1・2 の微調整後の bits-per-byte)。
$\Delta_{\mathrm{after}} = \bar{d}$ であり、これは $\bar{b}_2 - \bar{b}_1$ と同じ値である。$c$ の標準偏差を

$$
\sigma_c = \sqrt{\frac{s_d^2}{n} + \sigma_{\mathrm{win},c}^2}
$$

とする。

- $s_d$: $d_s$ のシード間の標本標準偏差(不偏)、$n$: シード数。第 1 項は、シード(学習の乱数)による $\bar{d}$ の
  ばらつきである。
- $\sigma_{\mathrm{win},c}$: 評価窓(82 窓)を復元抽出する対応付きブートストラップで $c$ そのものを計算したときの
  標準偏差。各反復で同じ窓の再標本を使い、FP32 の基盤・NF4 の基盤・条件 1 の微調整後・条件 2 の微調整後の
  4 種類の bits-per-byte を、それぞれ「負の対数尤度の和 / UTF-8 バイト数の和」として計算する(微調整後の 2 種類は、
  窓ごとにシード平均した負の対数尤度を使う)。これらから $c^* = \Delta^*_{\mathrm{before}} - \Delta^*_{\mathrm{after}}$ を
  求める。第 2 項は、有限の評価集合による標本誤差($\Delta_{\mathrm{before}}$ と $\Delta_{\mathrm{after}}$ の両方の分)である。
  反復回数は`LEVELS`のブートストラップの反復回数に従う。

2 つの項は、学習の乱数によるばらつきと評価集合の標本誤差という別の要因によるものとして、独立とみなして足し合わせる。
微調整後のモデルの窓ごとの負の対数尤度は、シードごとに保持し、判定の一次情報として 6.9 節に全件印字する。

**判定基準の改訂(スモークテストの後、本番実行の前)**:

- **旧基準**: $\sigma_c = \sqrt{\sigma_1^2 / n + \sigma_2^2 / n + \sigma_{\mathrm{before}}^2}$($\sigma_1, \sigma_2$ は各条件の
  シード間の標本標準偏差、$\sigma_{\mathrm{before}}$ は $\Delta_{\mathrm{before}}$ の評価窓の対応付きブートストラップ
  標準偏差)。
- **新基準**: 上記の $\sigma_c = \sqrt{s_d^2 / n + \sigma_{\mathrm{win},c}^2}$。
- **改訂の理由**:
  - 両条件はシードと評価窓を共有する対応付きの設計であり、$\bar{b}_1$・$\bar{b}_2$ を独立とみなす旧基準は設計と
    整合しない。
  - 共有によって生じる正の共分散を誤差伝播に反映させると、差の分散は
    $\mathrm{Var}(\bar{b}_2 - \bar{b}_1) = \mathrm{Var}(\bar{b}_1) + \mathrm{Var}(\bar{b}_2) - 2\,\mathrm{Cov}(\bar{b}_1, \bar{b}_2)$
    となり、各条件の分散の和より小さくなる。$s_d^2 / n$ はこの共分散を含んだ差の分散の推定量である。
  - 旧基準は、$\Delta_{\mathrm{after}}$ の評価窓の標本誤差を含めていなかった($\sigma_{\mathrm{before}}$ は
    $\Delta_{\mathrm{before}}$ の分のみ)。
- **改訂の時点で分かっていたこと(透明性のための記録)**:
  - $\Delta_{\mathrm{before}}$ は決定的な値であり、スモークテストで得た値(約 0.0075)が本番の値と同じになる。
  - スモークテスト(12 ステップ・2 シード)の $c$ は負(約 $-0.0025$)だった。
  - この改訂は、閾値 $2\sigma_c$ を支持・反証の両方向に同じだけ変えるものであり、観測された $c$ の方向に依存しない。
- **変更しないもの**: 対比量 $c$ の定義、閾値の形($2\sigma_c$)、期待する差の方向、前提条件 P-D1・P-D2
  (P-D1 は従来どおり $\sigma_{\mathrm{before}}$ で判定する)、シード数・学習設定。
- **スモークテストの結果(両基準による判定、動作確認のみで結論ではない)**: 12 ステップ・2 シード・ブートストラップ 1,000 回で、$c = -0.002548$。旧基準では $\sigma_c = 0.012015$($2\sigma_c = 0.024029$)で判定不能、新基準では $s_d = 0.000556$、$\sigma_{\mathrm{win},c} = 0.001111$、$\sigma_c = 0.001179$($2\sigma_c = 0.002357$)で反証となった。この新基準による判定結果は、改訂を決めた後の再実行で初めて得たものである。
- 本番実行では、新基準による判定を行い、旧基準の $\sigma_c$ による判定を「旧基準(参考、判定には使わない)」と
  明示して併記する。

**判定**: 支持は $c > 2\sigma_c$、反証は $c < -2\sigma_c$、それ以外は判定不能。

**前提条件**:

- **P-D1**: $\Delta_{\mathrm{before}} > 2\sigma_{\mathrm{before}}$(量子化による劣化が評価集合で検出できること)。
  微調整の結果に依存しない量で定義している。
- **P-D2**: 両条件の全シードで学習が進んでいること。012 の前提条件 B1・C1 と同じ定義で、最終訓練損失が
  学習開始時の訓練損失から 5% 以上低下していること(最終訓練損失 $\le 0.95 \times$ 学習開始時の訓練損失)。
  訓練損失は 012 と同じ固定の 64 窓で測り、学習開始時の値はその条件の基盤モデル(条件 1 は FP32、
  条件 2 は NF4)で測る。

**直接の作用点**: LoRA が直接作用するのは各層(Query・Value の射影)の出力であり、bits-per-byte はその下流に
あたる。それでも bits-per-byte を対比量とするのは、QLoRA の主張そのものが「量子化した基盤の上での微調整が、
量子化しない基盤の上での微調整と同等の性能になる」という微調整後の性能についての主張だからである。

**診断量**:

- 回復率 $1 - \Delta_{\mathrm{after}} / \Delta_{\mathrm{before}}$。
- 学習中の最大 GPU メモリ(`torch.cuda.max_memory_allocated`、CUDA 環境のみ)。この規模では活性化のメモリが
  支配的であり(012 の 7.6 節・011 の実験 G)、また本実装は順伝播ごとに FP32 の重みを逆量子化して作るので、
  基盤の重みの削減がピークメモリの削減として現れるとは限らない。
- 基盤の重みの格納バイト数の実測値(5.6 節)。
- 条件 1 の最終 bits-per-byte と、012 のセル出力に記録された LoRA($r = 8$)の値の比較(同じ設定の再実行の
  再現性の確認。実行環境が異なれば一致しない)。

#### 観察(判定基準を設けない): LLM.int8() の外れ値の特徴次元

判定基準は設けない(3.5 節のとおり、外れ値の特徴次元は約 67 億パラメータ以上で顕在化すると報告されており、
本トピックのモデルで現れるかどうかについて検証すべき仮説を置かない)。FP32 の基盤モデルに英語 Wikipedia の
検証集合の先頭 64 窓を入力し、各層の量子化対象の行列への入力活性化(隠れ状態)について、特徴次元ごとの
最大絶対値を印字・可視化する。$d_{\mathrm{model}}$ 次元の隠れ状態(注意機構の射影への入力と SwiGLU の入力、
各層 2 つ、計 8 つ)について、原論文の基準(大きさ 6 以上、全隠れ状態の 25% 以上、系列位置の 6% 以上)を
満たす特徴次元の数を数える。

#### メモリ量の検証(実験ではなくアサーション)

パック格納の実測バイト数が、閉形式(5.6 節の表)と完全一致することをアサーションで確認する(5.6 節で実行済み)。
期待値は手で計算した独立な定数として持つ。

### 6.1 節の補足: 本番実行のやり直し

**この補足は、やり直した本番実行の前に記録したものである。**

- 初回の本番実行(Google Colab T4、`SMOKE_TEST=False`)の後、6.9 節(判定の一次情報)の出力の書式のみを修正して、
  本番実行をやり直した。初回の 6.9 節の出力は 72,727 文字・2,535 行あり、数値の配列が 1 値 1 行に展開されていた
  ため、GitHub のプレビューで読めなかった。修正は出力の書式のみであり、計算・判定基準・`primary_data`の中身は
  変更していない。
- **やり直した実行を正式な本番実行とする。** この宣言は、やり直しの結果を見る前に行ったものである。
- 初回の本番実行の 6.10 節の判定一覧(セル出力ではなく本文として転記する):

| 実験 | 対比量 | 前提条件 | 判定 |
|---|---|---|---|
| A | b=4: $\bar{r}=-0.0156$(SE 0.0006)、b=5: $-0.0159$(SE 0.0006)、b=6: $-0.0163$(SE 0.0005)、b=7: $-0.0156$(SE 0.0005)、b=8: $-0.0149$(SE 0.0006) | P0 成立 | 支持 |
| B | 層別 Spearman $= +0.3925$、95% 区間 $[+0.0735, +0.6878]$ | P0・P-B 成立 | 支持 |
| C | $\bar{\ell} = -0.18560$、SE $= 0.02105$ | P0・P-C 成立 | 支持 |
| D(新基準) | $c = +0.001910$、$\sigma_c = 0.003762$ | P0・P-D1・P-D2 成立 | 判定不能 |
| D(旧基準、参考) | $c = +0.001910$、$\sigma_c = 0.003710$ | 同上 | 判定不能(参考) |

- 実験 A〜C の対比量と実験 D の $\Delta_{\mathrm{before}}$ は決定的な値なので、やり直しの結果は初回と一致するはずで
  ある(実験 B のブートストラップ区間も、乱数のシードと反復回数が同じなので一致するはずである)。実験 D の微調整後の
  値は、GPU の非決定性により変わりうる。
- **やり直しの結果(やり直した実行の後に追記)**: やり直した実行の 6.10 節の判定一覧と判定の一次情報(6.9 節)の全数値は、
  実験 D の微調整後の値を含めて、初回の本番実行と一致した(7 節の冒頭)。

### 6.2 スケーリングの計測と外挿

本番のデータ量・回数がスモークテストと異なる、または繰り返しの多い重い処理について、3 点のデータ量で実行時間を
実測し、$\log t = \log a + b \log n$ をあてはめてべき指数 $b$ を推定し、本番のデータ量へ外挿する。外挿値には本番での
実行回数を乗じる。

| 処理 | データ量 $n$ | 本番の実行回数 |
|---|---|---|
| 量子化(NF4 の最近傍探索を含む) | 28 行列を連結したテンソルの要素数 | 行列全体の量子化の回数(下のセルで数える) |
| 逆量子化込みの評価 | 英語 Wikipedia の検証集合の評価窓の数 | 7 条件 + 確認用 1 回 + 観察 1 回 |
| QLoRA の学習(逆量子化のオーバーヘッドを含む) | ステップ数 | 条件 2 つ × 本番のシード数 |
| ブートストラップ(評価窓・層別 Spearman) | 反復回数 | 評価窓 2 回、層別 Spearman 1 回 |
| 符号化 | 英語 Wikipedia の検証テキストの文字数 | 1 回(キャッシュ後は再符号化しない) |

学習の計測点は本番のステップ数の 1/4・1/2・1 倍とする(012 と同じ理由で、固定費のため $b < 1$ になりうるので、
見積もりには外挿値と本番のステップ数での直接計測値の大きいほうを使う)。


```python
def timed(fn) -> float:
    sync_device()
    t0 = time.time()
    fn()
    sync_device()
    return time.time() - t0


def fit_and_extrapolate(label: str, sizes: list[int], times: list[float], target: int) -> float:
    fit = fit_power_law_exponent(sizes, times)
    extrapolated = fit.coefficient * target**fit.exponent
    detail = ", ".join(f"n={n}: {t:.3f}s" for n, t in zip(sizes, times, strict=True))
    print(
        f"[{label}] {detail} -> b={fit.exponent:.3f}(標準誤差 {fit.exponent_stderr:.3f}), "
        f"R^2={fit.r_squared:.4f}, n={target} での外挿値 {extrapolated:.2f}s"
    )
    return extrapolated


_t0_scaling = time.time()

# (1) 量子化: 28 行列を連結したテンソルの先頭 n 要素
_flat_all = torch.cat([w.flatten() for w in TARGET_WEIGHTS.values()])
_quant_sizes = [TARGET_NUM_ELEMENTS // 4, TARGET_NUM_ELEMENTS // 2, TARGET_NUM_ELEMENTS]
_quant_extrapolated = {}
for _label, _cfg in (
    ("absmax INT4 ブロック 64", QUANT_CONDITIONS["q4"]),
    ("NF4 ブロック 64", QUANT_CONDITIONS["q5"]),
    ("NF4 ブロック 64 + 二重量子化", QUANT_CONDITIONS["q6"]),
    ("absmax INT8 テンソル単位", QUANT_CONDITIONS["q2"]),
):
    quantize_weight(_flat_all[:1024], **_cfg)  # 初回呼び出しのオーバーヘッドを除く
    _times = [
        timed(lambda n=n, c=_cfg: quantize_weight(_flat_all[:n], **c).dequantize())
        for n in _quant_sizes
    ]
    _quant_extrapolated[_label] = fit_and_extrapolate(
        f"量子化 + 逆量子化 {_label}", _quant_sizes, _times, TARGET_NUM_ELEMENTS
    )
# 本番での行列全体の量子化の回数: 5.6 節のキャッシュ 6 + 5.8 節のキャッシュなしの確認 6 +
# 実験 A 7 ビット幅 + 実験 B(ブロックの掃引 13 + テンソル単位 1)+ 実験 C 0(キャッシュを再利用)
N_FULL_QUANTIZATIONS = 6 + 6 + len(BITS_JUDGED_A) + len(BITS_DIAGNOSTIC_A) + len(BLOCK_SWEEP_B) + 1
_quant_total = max(_quant_extrapolated.values()) * N_FULL_QUANTIZATIONS

# (2) 逆量子化込みの評価(最も逆量子化の重い q6)
_q6_model = build_quantized_model("q6")
_num_wiki_windows = wiki_windows.size(0)
_eval_sizes = [_num_wiki_windows // 4, _num_wiki_windows // 2, _num_wiki_windows]
per_window_nll(_q6_model, wiki_windows[:16], wiki_mask[:16])
_eval_extrapolated = fit_and_extrapolate(
    "評価 q6(英語 Wikipedia、逆量子化込み)",
    _eval_sizes,
    [
        timed(lambda n=n, m=_q6_model: per_window_nll(m, wiki_windows[:n], wiki_mask[:n]))
        for n in _eval_sizes
    ],
    _num_wiki_windows,
)
del _q6_model
N_WIKI_EVALUATIONS = (
    len(QUANT_CONDITIONS) + 1 + 1
)  # 7 条件 + evaluate_bits_per_byte の確認 1 + 観察 1

# (3) QLoRA の学習(学習曲線の評価・訓練損失の測定を含む)
_train_sizes = [PROD_NUM_STEPS // 4, PROD_NUM_STEPS // 2, PROD_NUM_STEPS]
_train_extrapolated = {}
for _cond in ("fp32", "nf4"):
    run_finetune(_cond, seed=0, num_steps=2, measure=False)  # 初回のオーバーヘッドを除く
    _times = [
        timed(lambda n=n, c=_cond: run_finetune(c, seed=0, num_steps=n)) for n in _train_sizes
    ]
    _fit_value = fit_and_extrapolate(f"学習 {_cond}", _train_sizes, _times, PROD_NUM_STEPS)
    _train_extrapolated[_cond] = max(_fit_value, _times[-1])
_prod_seeds = LEVELS["prod"]["NUM_SEEDS"]
_train_total = sum(_train_extrapolated.values()) * _prod_seeds

# (4) ブートストラップ
_prod_resamples = LEVELS["prod"]["BOOTSTRAP_RESAMPLES"]
_boot_sizes = [_prod_resamples // 4, _prod_resamples // 2, _prod_resamples]
_dummy_numerators = np.random.default_rng(0).random((len(QUANT_CONDITIONS), _num_wiki_windows))
_boot_window = fit_and_extrapolate(
    "ブートストラップ(評価窓、7 条件)",
    _boot_sizes,
    [
        timed(
            lambda n=n: paired_bootstrap_ratio_of_sums(_dummy_numerators, wiki_window_bytes, n, 0)
        )
        for n in _boot_sizes
    ],
    _prod_resamples,
)
_dummy_groups = [TARGET_SHAPES[n] for n in TARGET_NAMES]
_dummy_x = np.random.default_rng(1).random(NUM_TARGET_MATRICES)
_boot_spearman = fit_and_extrapolate(
    "ブートストラップ(層別 Spearman)",
    _boot_sizes,
    [
        timed(
            lambda n=n: bootstrap_stratified_spearman_correlation(
                _dummy_x, _dummy_x[::-1], _dummy_groups, n, 0
            )
        )
        for n in _boot_sizes
    ],
    _prod_resamples,
)

# (5) 符号化(トークナイザのチャンク単位のメモ化が効かないよう、計測ごとに取得し直す)
_encode_sizes = [len(wiki_val_text) // 4, len(wiki_val_text) // 2, len(wiki_val_text)]
_encode_times = []
for _n in _encode_sizes:
    _fresh_tokenizer, _ = load_bpe_id_tokenizer_from_hub(TOKENIZER_REPO_ID)
    _encode_times.append(
        timed(lambda n=_n, t=_fresh_tokenizer: encode_corpus(t, wiki_val_text[:n]))
    )
_encode_extrapolated = fit_and_extrapolate(
    "符号化(英語 Wikipedia の検証テキスト)", _encode_sizes, _encode_times, len(wiki_val_text)
)
_t_scaling = time.time() - _t0_scaling

_estimate = {
    f"量子化(行列全体 {N_FULL_QUANTIZATIONS} 回、最も遅い方式で見積もる)": _quant_total,
    f"評価(英語 Wikipedia、{N_WIKI_EVALUATIONS} 回)": _eval_extrapolated * N_WIKI_EVALUATIONS,
    f"QLoRA の学習(2 条件 x {_prod_seeds} シード)": _train_total,
    "ブートストラップ(評価窓 2 回 + 層別 Spearman 1 回)": 2 * _boot_window + _boot_spearman,
    "符号化(1 回)": _encode_extrapolated,
    "スケーリング計測自体(このセルの実測)": _t_scaling,
}
_total_estimate = sum(_estimate.values())
print(
    f"\n--- 本番実行の見積もり(PROD_NUM_STEPS={PROD_NUM_STEPS}、実行回数を乗じた値、{device} 基準)---"
)
for _k, _v in _estimate.items():
    print(f"  {_k}: {_v:.1f}s")
print(
    f"合計: {_total_estimate:.1f}s({_total_estimate / 60:.1f} 分)/ セッション予算 "
    f"{SESSION_BUDGET_SECONDS / 60:.0f} 分 / 余裕 {SESSION_BUDGET_SECONDS / _total_estimate:.2f} 倍"
)
if _total_estimate > SESSION_BUDGET_SECONDS:
    print("警告: 見積もりがセッション予算を超える。シード数・反復回数を見直す必要がある。")
print(f"注記: 上記は実行したデバイス({device})での計測に基づく値である。")
```

    [量子化 + 逆量子化 absmax INT4 ブロック 64] n=786688: 0.009s, n=1573376: 0.015s, n=3146752: 0.034s -> b=0.935(標準誤差 0.128), R^2=0.9817, n=3146752 での外挿値 0.03s
    [量子化 + 逆量子化 NF4 ブロック 64] n=786688: 0.037s, n=1573376: 0.063s, n=3146752: 0.135s -> b=0.942(標準誤差 0.086), R^2=0.9917, n=3146752 での外挿値 0.13s
    [量子化 + 逆量子化 NF4 ブロック 64 + 二重量子化] n=786688: 0.038s, n=1573376: 0.066s, n=3146752: 0.133s -> b=0.900(標準誤差 0.067), R^2=0.9945, n=3146752 での外挿値 0.13s
    [量子化 + 逆量子化 absmax INT8 テンソル単位] n=786688: 0.008s, n=1573376: 0.012s, n=3146752: 0.028s -> b=0.881(標準誤差 0.200), R^2=0.9511, n=3146752 での外挿値 0.03s
    [評価 q6(英語 Wikipedia、逆量子化込み)] n=310: 0.557s, n=620: 1.101s, n=1240: 2.217s -> b=0.997(標準誤差 0.008), R^2=0.9999, n=1240 での外挿値 2.21s
    [学習 fp32] n=21: 4.834s, n=43: 7.471s, n=87: 13.421s -> b=0.718(標準誤差 0.065), R^2=0.9920, n=87 での外挿値 13.07s
    [学習 nf4] n=21: 5.295s, n=43: 8.036s, n=87: 14.620s -> b=0.714(標準誤差 0.077), R^2=0.9885, n=87 での外挿値 14.16s
    [ブートストラップ(評価窓、7 条件)] n=2500: 0.394s, n=5000: 1.005s, n=10000: 1.586s -> b=1.005(標準誤差 0.200), R^2=0.9618, n=10000 での外挿値 1.72s
    [ブートストラップ(層別 Spearman)] n=2500: 0.655s, n=5000: 1.029s, n=10000: 2.816s -> b=1.053(標準誤差 0.231), R^2=0.9540, n=10000 での外挿値 2.57s
    [符号化(英語 Wikipedia の検証テキスト)] n=302681: 0.288s, n=605363: 0.780s, n=1210727: 0.841s -> b=0.773(標準誤差 0.384), R^2=0.8025, n=1210727 での外挿値 0.98s
    
    --- 本番実行の見積もり(PROD_NUM_STEPS=87、実行回数を乗じた値、cuda 基準)---
      量子化(行列全体 33 回、最も遅い方式で見積もる): 4.3s
      評価(英語 Wikipedia、9 回): 19.9s
      QLoRA の学習(2 条件 x 5 シード): 140.2s
      ブートストラップ(評価窓 2 回 + 層別 Spearman 1 回): 6.0s
      符号化(1 回): 1.0s
      スケーリング計測自体(このセルの実測): 69.6s
    合計: 241.0s(4.0 分)/ セッション予算 120 分 / 余裕 29.88 倍
    注記: 上記は実行したデバイス(cuda)での計測に基づく値である。


### 6.3 前提条件 P0 と、下流の bits-per-byte の増加(診断量)

全 7 条件について、英語 Wikipedia の検証集合の評価窓ごとの負の対数尤度を計算する(同じ評価窓・同じ分母)。
q0 の bits-per-byte を`evaluate_bits_per_byte()`でも計算し、窓ごとの値の総和と一致することを確認したうえで
P0 を判定する。


```python
def verdict_label(computed: str, preconditions: list[str]) -> str:
    return computed if all(precondition_status.get(p) for p in preconditions) else "前提不成立"


wiki_nll: dict[str, np.ndarray] = {}
wiki_bits_per_byte: dict[str, float] = {}
_t0 = time.time()
for _cid in QUANT_CONDITIONS:
    _model = build_quantized_model(_cid)
    wiki_nll[_cid] = per_window_nll(_model, wiki_windows, wiki_mask)
    wiki_bits_per_byte[_cid] = bits_per_byte_from_nll(wiki_nll[_cid], wiki_bytes)
    del _model
print(f"英語 Wikipedia の検証集合の評価(7 条件): {time.time() - _t0:.1f}s")

B_P0 = evaluate_bits_per_byte(base_model, wiki_windows, wiki_mask, wiki_bytes, device)
assert math.isclose(B_P0, wiki_bits_per_byte["q0"], rel_tol=1e-6)
P0_RELATIVE_ERROR = abs(B_P0 - P0_REFERENCE_BITS_PER_BYTE) / P0_REFERENCE_BITS_PER_BYTE
precondition_status["P0"] = bool(P0_RELATIVE_ERROR <= P0_RELATIVE_TOLERANCE)
print(
    f"P0: b_P0 = {B_P0:.6f}(008 の記録値 {P0_REFERENCE_BITS_PER_BYTE}、相対誤差 {P0_RELATIVE_ERROR:.4%}、"
    f"許容 {P0_RELATIVE_TOLERANCE:.0%})-> {'成立' if precondition_status['P0'] else '不成立'}"
)

# 対応付きブートストラップ(全条件に同じ再標本)
_condition_ids = list(QUANT_CONDITIONS)
_wiki_boot = paired_bootstrap_ratio_of_sums(
    np.stack([wiki_nll[c] for c in _condition_ids]) / LN2,
    wiki_window_bytes,
    BOOTSTRAP_RESAMPLES,
    seed=0,
)
assert _wiki_boot.shape == (BOOTSTRAP_RESAMPLES, len(_condition_ids))
diagnostic_bits_per_byte = {}
print(
    f"\n{_smoke_tag}{'条件':<22} | {'bits-per-byte':>13} | {'増加':>9} | {'増加の標準偏差':>13} | 95% 区間"
)
for _i, _cid in enumerate(_condition_ids):
    _increase = wiki_bits_per_byte[_cid] - wiki_bits_per_byte["q0"]
    _boot_increase = _wiki_boot[:, _i] - _wiki_boot[:, 0]
    _lo, _hi = np.percentile(_boot_increase, [2.5, 97.5])
    diagnostic_bits_per_byte[_cid] = {
        "label": QUANT_LABELS[_cid],
        "bits_per_byte": wiki_bits_per_byte[_cid],
        "increase": _increase,
        "increase_bootstrap_std": float(_boot_increase.std(ddof=1)),
        "increase_bootstrap_95": [float(_lo), float(_hi)],
        "storage_bytes": storage_table[_cid],
    }
    print(
        f"{QUANT_LABELS[_cid]:<22} | {wiki_bits_per_byte[_cid]:>13.6f} | {_increase:>+9.5f} | "
        f"{_boot_increase.std(ddof=1):>13.5f} | [{_lo:+.5f}, {_hi:+.5f}]"
    )
```

    英語 Wikipedia の検証集合の評価(7 条件): 17.4s
    P0: b_P0 = 1.668067(008 の記録値 1.6707、相対誤差 0.1576%、許容 1%)-> 成立
    
    条件                     | bits-per-byte |        増加 |       増加の標準偏差 | 95% 区間
    FP32                   |      1.668067 |  +0.00000 |       0.00000 | [+0.00000, +0.00000]
    INT8 (per-channel)     |      1.668079 |  +0.00001 |       0.00001 | [-0.00002, +0.00004]
    INT8 (per-tensor)      |      1.668198 |  +0.00013 |       0.00002 | [+0.00009, +0.00018]
    INT4 (per-tensor)      |      1.697354 |  +0.02929 |       0.00037 | [+0.02857, +0.03000]
    INT4 (block 64)        |      1.677536 |  +0.00947 |       0.00021 | [+0.00906, +0.00987]
    NF4 (block 64)         |      1.677038 |  +0.00897 |       0.00019 | [+0.00859, +0.00935]
    NF4 (block 64 + DQ)    |      1.677035 |  +0.00897 |       0.00019 | [+0.00859, +0.00935]


### 6.4 実験 A: ビット幅と再構成誤差の理論値


```python
def reconstruction_error(w: torch.Tensor, q) -> float:
    return float(((w.double() - q.dequantize().double()) ** 2).mean())


experiment_a = {}  # ビット幅 -> 行列ごとの実測値・予測値
for _bits in BITS_DIAGNOSTIC_A + BITS_JUDGED_A:
    _rows = {}
    for _name in TARGET_NAMES:
        _q = quantize_weight(
            TARGET_WEIGHTS[_name],
            method="absmax",
            bits=_bits,
            granularity="block",
            block_size=BLOCK_SIZE,
        )
        _rows[_name] = {
            "measured": reconstruction_error(TARGET_WEIGHTS[_name], _q),
            "predicted": float((_q.scale.double() ** 2).mean() / 12),
        }
    experiment_a[_bits] = _rows


def mean_and_standard_error(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(len(values)))


def equivalence_verdict(mean: float, se: float, margin: float) -> str:
    lo, hi = mean - 2 * se, mean + 2 * se
    if -margin <= lo and hi <= margin:
        return "支持"
    if lo > margin or hi < -margin:
        return "反証"
    return "判定不能"


summary_a = {}
print(
    f"{_smoke_tag}同等性の範囲: [-log 1.1, log 1.1] = [{-EQUIVALENCE_MARGIN_A:+.4f}, {EQUIVALENCE_MARGIN_A:+.4f}]"
)
print(f"{'b':>2} | {'r_bar':>9} | {'SE':>8} | {'r_bar - 2SE':>11} | {'r_bar + 2SE':>11} | 判定")
for _bits, _rows in experiment_a.items():
    _r = np.array([math.log(v["measured"] / v["predicted"]) for v in _rows.values()])
    assert len(_r) == NUM_TARGET_MATRICES
    _mean, _se = mean_and_standard_error(_r)
    _v = equivalence_verdict(_mean, _se, EQUIVALENCE_MARGIN_A)
    summary_a[_bits] = {"r_bar": _mean, "se": _se, "verdict": _v, "judged": _bits in BITS_JUDGED_A}
    _tag = "" if _bits in BITS_JUDGED_A else "(診断量、判定に使わない)"
    print(
        f"{_bits:>2} | {_mean:>+9.5f} | {_se:>8.5f} | {_mean - 2 * _se:>+11.5f} | {_mean + 2 * _se:>+11.5f} | {_v}{_tag}"
    )

# 診断量: 第 1 段階のスモークテストの出力(小数第 5 位まで印字した値)から転記した r_bar との差。
# 実験 A は決定的な計算なので、同じ実行環境なら丸めの範囲(5e-6)で一致する。
_R_BAR_SMOKE = {
    2: -0.00945,
    3: -0.01322,
    4: -0.01557,
    5: -0.01594,
    6: -0.01628,
    7: -0.01562,
    8: -0.01494,
}
for _bits in summary_a:
    print(
        f"診断量: b={_bits} の r_bar - スモークテストの転記値 = "
        f"{summary_a[_bits]['r_bar'] - _R_BAR_SMOKE[_bits]:+.2e}(転記値 {_R_BAR_SMOKE[_bits]:+.5f})"
    )

_judged = [summary_a[b]["verdict"] for b in BITS_JUDGED_A]
verdict_A = (
    "支持" if all(v == "支持" for v in _judged) else "反証" if "反証" in _judged else "判定不能"
)
print(f"{_smoke_tag}実験 A の判定関数の結果(b = 4〜8 の結合): {verdict_A}")

# 診断量: 系統的なずれの理論値と、隣り合うビット幅の誤差の比
print(
    f"\n{_smoke_tag}診断量: 最大値が誤差 0 で表現されることによるずれの理論値 log(63/64) = {math.log(63 / 64):+.5f}"
)
_all_bits = BITS_DIAGNOSTIC_A + BITS_JUDGED_A
for _b in _all_bits[:-1]:
    _ratio = np.mean(
        [
            experiment_a[_b][n]["measured"] / experiment_a[_b + 1][n]["measured"]
            for n in TARGET_NAMES
        ]
    )
    _theory = ((2**_b - 1) / (2 ** (_b - 1) - 1)) ** 2
    print(
        f"{_smoke_tag}診断量: err_{_b} / err_{_b + 1}(行列平均)= {_ratio:.3f}(理論値 {_theory:.3f})"
    )

fig, ax = plt.subplots(figsize=(7, 4))
_bits_axis = list(summary_a)
_means = [summary_a[b]["r_bar"] for b in _bits_axis]
_errs = [2 * summary_a[b]["se"] for b in _bits_axis]
ax.errorbar(
    _bits_axis,
    _means,
    yerr=_errs,
    marker="o",
    capsize=4,
    label="mean log(measured / predicted) +- 2 SE",
)
ax.axhspan(
    -EQUIVALENCE_MARGIN_A,
    EQUIVALENCE_MARGIN_A,
    color="tab:green",
    alpha=0.12,
    label="equivalence margin (log 1.1)",
)
ax.axhline(math.log(63 / 64), color="tab:red", linestyle="--", linewidth=1, label="log(63/64)")
ax.axvspan(1.5, 3.5, color="gray", alpha=0.1, label="diagnostic only (b = 2, 3)")
ax.set_xlabel("bit width b")
ax.set_ylabel("log(measured / predicted)")
ax.set_title("Experiment A: reconstruction error vs Delta^2 / 12 (absmax, block 64)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.show()
```

    同等性の範囲: [-log 1.1, log 1.1] = [-0.0953, +0.0953]
     b |     r_bar |       SE | r_bar - 2SE | r_bar + 2SE | 判定
     2 |  -0.00945 |  0.01465 |    -0.03875 |    +0.01985 | 支持(診断量、判定に使わない)
     3 |  -0.01322 |  0.00054 |    -0.01431 |    -0.01214 | 支持(診断量、判定に使わない)
     4 |  -0.01557 |  0.00062 |    -0.01682 |    -0.01432 | 支持
     5 |  -0.01594 |  0.00058 |    -0.01710 |    -0.01477 | 支持
     6 |  -0.01628 |  0.00050 |    -0.01727 |    -0.01528 | 支持
     7 |  -0.01562 |  0.00052 |    -0.01665 |    -0.01458 | 支持
     8 |  -0.01494 |  0.00060 |    -0.01613 |    -0.01374 | 支持
    診断量: b=2 の r_bar - スモークテストの転記値 = +2.79e-06(転記値 -0.00945)
    診断量: b=3 の r_bar - スモークテストの転記値 = -4.11e-06(転記値 -0.01322)
    診断量: b=4 の r_bar - スモークテストの転記値 = -2.03e-06(転記値 -0.01557)
    診断量: b=5 の r_bar - スモークテストの転記値 = +1.82e-06(転記値 -0.01594)
    診断量: b=6 の r_bar - スモークテストの転記値 = +1.05e-06(転記値 -0.01628)
    診断量: b=7 の r_bar - スモークテストの転記値 = +4.39e-06(転記値 -0.01562)
    診断量: b=8 の r_bar - スモークテストの転記値 = +3.61e-06(転記値 -0.01494)
    実験 A の判定関数の結果(b = 4〜8 の結合): 支持
    
    診断量: 最大値が誤差 0 で表現されることによるずれの理論値 log(63/64) = -0.01575
    診断量: err_2 / err_3(行列平均)= 9.060(理論値 9.000)
    診断量: err_3 / err_4(行列平均)= 5.457(理論値 5.444)
    診断量: err_4 / err_5(行列平均)= 4.594(理論値 4.592)
    診断量: err_5 / err_6(行列平均)= 4.273(理論値 4.271)
    診断量: err_6 / err_7(行列平均)= 4.127(理論値 4.130)
    診断量: err_7 / err_8(行列平均)= 4.061(理論値 4.064)



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/013_quantization_basics/output_32_1.png)
    




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/03_efficient_training/013_quantization_basics.ipynb
