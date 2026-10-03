---
title: "Mixture of Experts(MoE) / Mixture of Experts(実装・実験編 2/5)"
---

この記事は後編(実装・実験編 2/5)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/022_mixture_of_experts-practice-1)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/022_mixture_of_experts-practice-3)。

### 5.4 モデルの構築・単体テスト・不変条件の確認

**モデルの構築**: 条件 $c$・シード $s$ のモデルは、次の手順で作る。

1. `torch.manual_seed(22100 + s)`の後に、密なモデル(条件 D)を構築する。
2. 条件が D 以外なら、`torch.manual_seed(22300 + s)`の後に、順伝播ネットワークだけを差し替えたモデル(MoE、または中間次元を
   8 倍にした密なモデル)を構築し、**順伝播ネットワーク以外のすべてのパラメータ(埋め込み・注意機構・正規化層)を 1 のモデルから
   コピーする**。

したがって、同じシードの条件どうしは、順伝播ネットワーク以外の初期値が bit 単位で一致する。M8 と N8 は、順伝播ネットワークと
ルーターの初期値も一致する(違いは負荷分散損失の係数だけ)。

**単体テスト**(CPU、小さな次元):

- 固定形状の振り分け・集約が、トークンを 1 個ずつ処理する素朴なループの実装と一致する(top-1・top-2、ゲートの正規化の有無、
  破棄の起こる容量と起こらない容量)。
- 容量が十分に大きいとき、破棄が 0 になる。破棄された位置の出力は厳密に 0 になる。
- $E = 1$・top-1 の MoE 層が、同じ重みの密な SwiGLU の順伝播ネットワークと一致する。同じ乱数の状態から構築したとき、エキスパートの
  初期値が密な順伝播ネットワークの初期値と bit 単位で一致する。
- 破棄の因果性: 系列の後ろの位置の入力を変えても、前の位置の出力が変わらない。
- ルーターへの勾配: ゲートを通じてルーターに勾配が流れる。負荷分散損失の勾配はルーターにだけ流れる。
- 3.5 節の数値例($\sum_i f_i P_i = 0.461 < 1/2$)と、ロジットがすべて 0 のときの値($N \sum_i f_i P_i = 1$、$\mathcal{L}_z = (\log N)^2$)。
  **ルーターのロジット・確率・補助損失は、層や入力が FP64 でも FP32 で計算される**(3.6 節の設計。型は入力の型・デバイス・
  autocast の状態によらない)。そのため、補助損失の値は FP32 の精度でしか理論値に一致しない。型が FP32 であることを確かめたうえで、
  許容の誤差を FP32 の丸めの単位(machine epsilon)× 値の大きさの 16 倍とする。
- autocast の中でも、ルーターのロジットと補助損失が FP32 である。
- **破棄なしの評価が、バッチの組み方に依存しない**(実行するデバイス、FP32): 評価時に破棄をしない MoE のモデル(M8・M16、ルーターを
  鋭くして割り当てを偏らせたもの)で、窓を 16 個ずつのバッチで評価した窓ごとの負の対数尤度と、1 窓ずつ評価した値が一致する。
  許容の閾値は、同じ経路(16 個ずつ)を 2 回実行した差と、バッチの組み方に依存しないことが構造から分かっている密なモデルで
  同じ比較(16 個ずつと 1 窓ずつ)をした差の実測の、大きい方の 16 倍とする(丸めの単位 × 値の大きさを下限にする)。下限の意味:
  窓ごとの値は 255 トークンの負の対数尤度の和なので、各トークンの値が FP32 の丸め 1 単位ずつずれたときの和のずれが、この大きさ
  (machine epsilon × 窓の値)に相当する。あわせて、
  評価時にも capacity factor 2.0 で破棄する旧基準の設定では、同じ比較の差がこの閾値を超えること(確認の検出力)を確かめる。

**一致の許容の閾値**: 「同じ経路を 2 回実行した差」と、「同じ計算を別の実行の順序で行った差」(密な順伝播ネットワークを、
トークンをまとめて計算した結果と 1 個ずつ計算した結果の差)を同じ環境で実測し、その大きい方と、出力の大きさに対する丸めの単位
(machine epsilon × 出力の最大の絶対値)のうち最大のものの 16 倍を閾値とする。実測に基づかない固定の値は使わない。

**不変条件**: 全条件で順伝播ネットワーク以外のパラメータ数が同じ、エキスパート 1 個のパラメータ数が密なモデルの順伝播ネットワークと
同じ、トークンあたりの順伝播ネットワークの積和の回数(ルーターを除く)が同じ、同じシードの条件どうしでミニバッチの順序が同じ、
係数 0 の補助損失は補助損失なしと同じ学習になる。

**既存モジュールの後方互換性**: `src/models/gpt.py`と`src/training/trainer.py`を変更する前のコミット(`9f10e22`)の実装と、既定の
引数での出力・勾配・学習の履歴・学習後の重み・`evaluate_bits_per_byte()`の値が bit 単位で一致することを、別プロセスで両方の実装を
読み込んで比べる。


```python
_t0_checks = time.time()


def build_gpt(feed_forward_factory) -> GPTLanguageModel:
    # 006・008 と同じ構成(RoPE・RMSNorm・正規化前置・重み共有)。順伝播ネットワークだけを factory で差し替える
    return GPTLanguageModel(
        vocabulary_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        max_sequence_length=SEQUENCE_LENGTH,
        positional_transform=RotaryPositionEmbedding(D_MODEL // NUM_HEADS, max_position=SEQUENCE_LENGTH),
        normalization_factory=RMSNorm,
        feed_forward_factory=feed_forward_factory,
        tie_embeddings=True,
        dropout=0.0,
    )


def is_feed_forward_key(name: str) -> bool:
    return ".feed_forward." in name


def build_model(condition: str, seed_index: int) -> GPTLanguageModel:
    spec = CONDITIONS[condition]
    torch.manual_seed(INIT_SEED_BASE + seed_index)
    dense = build_gpt(functools.partial(SwiGLUFeedForwardNetwork, D_MODEL, SWIGLU_D_FF))
    if condition == "D":
        return dense
    torch.manual_seed(FEED_FORWARD_SEED_BASE + seed_index)
    if spec["kind"] == "dense":
        factory = functools.partial(SwiGLUFeedForwardNetwork, D_MODEL, spec["d_ff"])
    else:
        factory = functools.partial(
            MixtureOfExpertsFeedForward,
            D_MODEL,
            spec["d_ff"],
            spec["experts"],
            top_k=TOP_K,
            train_capacity_factor=TRAIN_CAPACITY_FACTOR,
            eval_capacity_factor=EVAL_CAPACITY_FACTOR,
        )
    model = build_gpt(factory)
    trunk = {k: v for k, v in dense.state_dict().items() if not is_feed_forward_key(k)}
    result = model.load_state_dict(trunk, strict=False)
    assert not result.unexpected_keys and all(is_feed_forward_key(k) for k in result.missing_keys)
    return model


def trunk_hash(model: GPTLanguageModel) -> str:
    # 順伝播ネットワーク以外のパラメータのハッシュ
    return sha256_of_tensors(v for k, v in sorted(model.state_dict().items()) if not is_feed_forward_key(k))


def naive_moe_forward(layer: MixtureOfExpertsFeedForward, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 素朴な参照実装: トークンを 1 個ずつ、エキスパートごとに処理する。破棄の優先順位は src/layers/moe.py と同じ。
    # ルーターのロジットは MoE 層と同じく FP32 で計算する(比べる対象は振り分け・集約・破棄である)
    flat = x.reshape(-1, x.size(-1))
    probabilities = torch.softmax(layer.compute_router_logits(flat), dim=-1)
    top_p, top_i = probabilities.topk(layer.top_k, dim=-1)
    if layer.renormalize_gates:
        top_p = top_p / top_p.sum(dim=-1, keepdim=True)
    capacity = compute_expert_capacity(flat.size(0), layer.num_experts, layer.top_k, layer.capacity_factor())
    used = [0] * layer.num_experts
    out = torch.zeros_like(flat)
    kept = torch.zeros(layer.top_k, flat.size(0), dtype=torch.bool)
    for r in range(layer.top_k):
        for t in range(flat.size(0)):
            e = int(top_i[t, r])
            if used[e] >= capacity:
                continue
            used[e] += 1
            kept[r, t] = True
            hidden = functional.silu(flat[t] @ layer.w[e].t()) * (flat[t] @ layer.v[e].t())
            out[t] += top_p[t, r] * (hidden @ layer.w2[e].t())
    return out.view(x.shape), kept


def max_abs_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).abs().max())


with torch.no_grad():
    # --- 許容の閾値の基準(FP64・CPU): 同じ経路の 2 回の差と、同じ計算を別の実行の順序で行った差 ---
    _d, _f = 32, 48
    torch.manual_seed(1)
    _dense = SwiGLUFeedForwardNetwork(_d, _f).double()
    _x = torch.randn(3, 40, _d, dtype=torch.float64)
    _batched = _dense(_x)
    _repeat = max_abs_difference(_batched, _dense(_x))
    _per_token = torch.stack([_dense(t) for t in _x.reshape(-1, _d)]).view(_x.shape)
    _reorder = max_abs_difference(_batched, _per_token)
    _unit = float(torch.finfo(torch.float64).eps) * float(_batched.abs().max())
    TOLERANCE_64 = 16 * max(_repeat, _reorder, _unit)
    print(
        f"許容の閾値の基準(FP64): 同じ経路の 2 回の差 {_repeat:.2e}、まとめて計算と 1 個ずつ計算の差 {_reorder:.2e}、"
        f"丸めの単位 x 出力の最大値 {_unit:.2e} -> 閾値 {TOLERANCE_64:.2e}"
    )

    # --- 振り分け・集約と素朴なループの一致 ---
    _cases = [
        (4, 1, 1.0, False), (4, 1, 1.25, False), (4, 1, 4.0, False), (8, 2, 1.0, False), (8, 2, 1.25, True), (8, 2, 8.0, True),
        (16, 1, 1.25, False), (1, 1, 1.0, False),
    ]  # (E, k, capacity factor, ゲートの正規化)
    _rows = []
    for _e, _k, _cf, _renormalize in _cases:
        torch.manual_seed(2)
        _layer = MixtureOfExpertsFeedForward(_d, _f, _e, top_k=_k, train_capacity_factor=_cf, renormalize_gates=_renormalize).double()
        _layer.router_weight.mul_(20.0)  # ルーターを鋭くして、割り当てを偏らせる(破棄を起こす)
        _y = _layer(_x)
        _reference, _kept = naive_moe_forward(_layer, _x)
        _difference = max_abs_difference(_y, _reference)
        _dropped = float(_layer.last_routing["dropped_fraction"])
        assert _difference <= TOLERANCE_64, (_e, _k, _cf, _difference)
        assert round(_dropped * _kept.numel()) == int((~_kept).sum())  # 破棄された割り当ての個数が一致
        if _cf >= _e:  # 容量 C = T: 破棄は起こらない
            assert _dropped == 0.0
        _all_dropped = ~_kept.any(dim=0)  # どの候補も破棄されたトークンの出力は厳密に 0
        assert bool((_y.reshape(-1, _d)[_all_dropped] == 0).all())
        _rows.append(f"(E={_e}, k={_k}, c={_cf}): 差 {_difference:.1e}・破棄 {_dropped:.3f}")
    print("振り分け・集約と素朴なループの一致(閾値以内): " + "、".join(_rows) + ": OK")
    assert any("破棄 0.000" not in r for r in _rows), "破棄の起こる場合が含まれていない"

    # --- 容量の計算と枠の割り当て ---
    assert compute_expert_capacity(8192, 8, 1, 1.25) == 1280 and compute_expert_capacity(8192, 16, 1, 2.0) == 1024
    assert compute_expert_capacity(100, 1, 1, 2.0) == 100  # 上限は T
    _slot, _keep = compute_dispatch_slots(torch.tensor([0, 1, 0, 0, 1, 2]), 3, 2)
    assert _slot.tolist() == [0, 2, 1, 6, 3, 4] and _keep.tolist() == [True, True, True, False, True, True]
    print("容量の計算と枠の割り当て(先着順、超過は捨て枠): OK")

    # --- E = 1 の MoE 層と密な順伝播ネットワークの一致(FP32) ---
    _x32 = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, _d)
    torch.manual_seed(5)
    _dense32 = SwiGLUFeedForwardNetwork(_d, _f)
    torch.manual_seed(5)
    _moe1 = MixtureOfExpertsFeedForward(_d, _f, 1)
    assert torch.equal(_dense32.w.weight, _moe1.w[0]) and torch.equal(_dense32.v.weight, _moe1.v[0])
    assert torch.equal(_dense32.w2.weight, _moe1.w2[0])
    _a, _b = _dense32(_x32), _moe1(_x32)
    _repeat32 = max(max_abs_difference(_a, _dense32(_x32)), max_abs_difference(_b, _moe1(_x32)))
    _permutation = torch.randperm(_x32[0].numel() // _d * BATCH_SIZE)
    _flat32 = _x32.reshape(-1, _d)
    _reorder32 = max_abs_difference(_dense32(_flat32[_permutation]), _dense32(_flat32)[_permutation])
    _unit32 = float(torch.finfo(torch.float32).eps) * float(_a.abs().max())
    TOLERANCE_32 = 16 * max(_repeat32, _reorder32, _unit32)
    _difference = max_abs_difference(_a, _b)
    assert _difference <= TOLERANCE_32, _difference
    assert float(_moe1.last_routing["dropped_fraction"]) == 0.0 and float(_moe1.last_routing["mean_max_probability"]) == 1.0
    print(
        f"E = 1 の MoE 層と密な SwiGLU: 同じ乱数の状態からの初期値が bit 単位で一致、出力の差 {_difference:.2e}"
        f"(bit 単位で一致: {torch.equal(_a, _b)})。基準: 同じ経路の 2 回の差 {_repeat32:.2e}、並べ替えの差 {_reorder32:.2e}、"
        f"丸めの単位 {_unit32:.2e} -> 閾値 {TOLERANCE_32:.2e}: OK"
    )

    # --- 破棄の因果性: 後ろの位置の入力を変えても、前の位置の出力は変わらない(1 系列、c = 1) ---
    torch.manual_seed(3)
    _layer = MixtureOfExpertsFeedForward(_d, _f, 4, train_capacity_factor=1.0).double()
    _layer.router_weight.mul_(20.0)
    _sequence = torch.randn(1, 64, _d, dtype=torch.float64)
    _changed = _sequence.clone()
    _changed[:, 40:] = torch.randn(1, 24, _d, dtype=torch.float64)
    _y1 = _layer(_sequence)
    _dropped_1 = float(_layer.last_routing["dropped_fraction"])
    _y2 = _layer(_changed)
    assert _dropped_1 > 0 and max_abs_difference(_y1[:, :40], _y2[:, :40]) <= TOLERANCE_64
    assert max_abs_difference(_y1[:, 40:], _y2[:, 40:]) > TOLERANCE_64
    print(f"破棄の因果性(破棄の割合 {_dropped_1:.3f}。位置 40 以降の入力を変えても、位置 0〜39 の出力は変わらない): OK")

    # --- 3.5 節の数値例と、ロジットがすべて 0 のときの値 ---
    _p = torch.tensor([[0.51, 0.49]] * 6 + [[0.0, 1.0]] * 4, dtype=torch.float64)
    _f_example = torch.bincount(_p.argmax(dim=-1), minlength=2).double() / len(_p)
    _value = float((_f_example * _p.mean(dim=0)).sum())
    # この数値例は FP64 のテンソルの演算だけで作る。許容の誤差は FP64 の丸めの単位の 16 倍(値は 1 以下)
    _eps64 = float(torch.finfo(torch.float64).eps)
    assert max(abs(a - b) for a, b in zip(_f_example.tolist(), [0.6, 0.4], strict=True)) <= 16 * _eps64
    assert abs(_value - 0.4612) <= 16 * _eps64 and _value < 0.5
    # ルーターのロジット・確率・補助損失は、層や入力が FP64 でも FP32 で計算される(設計どおり)。したがって補助損失の値は
    # FP32 の精度でしか理論値に一致しない。許容の誤差は、FP32 の丸めの単位 x 値の大きさ の 16 倍とする
    # (logsumexp と 2 乗と平均で、丸めの単位の数倍の相対誤差がありうる。実装(CPU の命令セットなど)によって数え方が変わる)
    _layer = MixtureOfExpertsFeedForward(_d, _f, 8).double()
    _layer.router_weight.zero_()
    _layer(_x)
    assert _layer.w.dtype == torch.float64 and _x.dtype == torch.float64  # 層と入力は FP64
    assert _layer.compute_router_logits(_x.reshape(-1, _d)).dtype == torch.float32  # それでもルーターは FP32
    assert all(v.dtype == torch.float32 for v in _layer.auxiliary_losses.values())
    assert all(_layer.last_routing[k].dtype == torch.float32 for k in ("assignment_fraction", "mean_router_probability"))
    _eps32 = float(torch.finfo(torch.float32).eps)
    _balance = float(_layer.auxiliary_losses[LOAD_BALANCING_LOSS_NAME])
    _z = float(_layer.auxiliary_losses[ROUTER_Z_LOSS_NAME])
    _z_expected = math.log(8) ** 2
    assert abs(_balance - 1.0) <= 16 * _eps32 * 1.0, _balance
    assert abs(_z - _z_expected) <= 16 * _eps32 * _z_expected, (_z, _z_expected)
    print(
        f"3.5 節の数値例: sum_i f_i P_i = {_value:.4f} < 0.5。ロジットがすべて 0(ルーターと補助損失は FP32): N sum_i f_i P_i - 1 = "
        f"{_balance - 1.0:+.2e}、L_z - (log N)^2 = {_z - _z_expected:+.2e}(許容 {16 * _eps32 * _z_expected:.2e}。FP32 の丸めの 1 単位は "
        f"{_eps32 * _z_expected:.2e}): OK"
    )

# --- ルーターへの勾配 ---
torch.manual_seed(4)
_layer = MixtureOfExpertsFeedForward(_d, _f, 4).double()
_layer(_x).pow(2).sum().backward()
assert float(_layer.router_weight.grad.abs().sum()) > 0, "ゲートを通じてルーターに勾配が流れていない"
_layer.zero_grad()
_layer(_x)
_layer.auxiliary_losses[LOAD_BALANCING_LOSS_NAME].backward()
assert float(_layer.router_weight.grad.abs().sum()) > 0
assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in (_layer.w, _layer.v, _layer.w2))
try:
    MixtureOfExpertsFeedForward(_d, _f, 4, top_k=1, renormalize_gates=True)
    raise AssertionError("top-1 でのゲートの正規化が受け付けられた")
except ValueError:
    pass
print("ルーターへの勾配(ゲートを通じて流れる。負荷分散損失の勾配はルーターにだけ流れる。top-1 のゲートの正規化は拒否): OK")

# --- autocast の中でも、ルーターと補助損失は FP32 ---
_autocast_device = "cuda" if device.type == "cuda" else "cpu"
_autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
_layer = MixtureOfExpertsFeedForward(_d, _f, 4).to(_autocast_device)
with torch.autocast(device_type=_autocast_device, dtype=_autocast_dtype):
    _input = torch.randn(4, 16, _d, device=_autocast_device)
    _output = _layer(_input)
    _logits = _layer.compute_router_logits(_input.reshape(-1, _d).to(_autocast_dtype))
    _expert_dtype = torch.bmm(_input, _layer.w[:4].transpose(1, 2)).dtype
assert _logits.dtype == torch.float32 and _expert_dtype == _autocast_dtype
assert all(v.dtype == torch.float32 for v in _layer.auxiliary_losses.values()) and bool(torch.isfinite(_output).all())
print(f"autocast({_autocast_device}、{_autocast_dtype})の中で、エキスパートの行列積は低精度、ルーターのロジットと補助損失は FP32: OK")

# --- モデル: パラメータ数・計算量・初期値の対応 ---
_models = {c: build_model(c, 0) for c in CONDITIONS}
_trunk_parameters = {
    c: sum(v.numel() for k, v in m.named_parameters() if not is_feed_forward_key(k)) for c, m in _models.items()
}
_feed_forward_parameters = {
    c: sum(v.numel() for k, v in m.named_parameters() if is_feed_forward_key(k)) for c, m in _models.items()
}
DENSE_FEED_FORWARD_PARAMETERS = 3 * D_MODEL * SWIGLU_D_FF  # 1 層あたり
assert len(set(_trunk_parameters.values())) == 1, _trunk_parameters  # 順伝播ネットワーク以外のパラメータ数は全条件で同じ
for _c, _spec in CONDITIONS.items():
    if _spec["kind"] == "moe":  # エキスパート 1 個 = 密なモデルの順伝播ネットワーク。加えてルーター E x d_model
        _expected = NUM_LAYERS * (_spec["experts"] * DENSE_FEED_FORWARD_PARAMETERS + _spec["experts"] * D_MODEL)
        assert all(tuple(l.w.shape[1:]) == (SWIGLU_D_FF, D_MODEL) and l.top_k == 1 for l in find_moe_layers(_models[_c]))
    else:
        _expected = NUM_LAYERS * 3 * D_MODEL * _spec["d_ff"]
    assert _feed_forward_parameters[_c] == _expected, (_c, _feed_forward_parameters[_c], _expected)
# トークンあたりの順伝播ネットワークの積和の回数(ルーターを除く): D と MoE(k = 1)は 3 d d_ff で同じ、W は 8 倍
FEED_FORWARD_MULTIPLY_ADDS = {c: TOP_K * 3 * D_MODEL * s["d_ff"] for c, s in CONDITIONS.items()}
assert all(FEED_FORWARD_MULTIPLY_ADDS[c] == FEED_FORWARD_MULTIPLY_ADDS["D"] for c in CONDITIONS if c != "W")
assert FEED_FORWARD_MULTIPLY_ADDS["W"] == WIDE_FACTOR * FEED_FORWARD_MULTIPLY_ADDS["D"]
assert _feed_forward_parameters["W"] == NUM_LAYERS * 8 * DENSE_FEED_FORWARD_PARAMETERS  # W は M8 のエキスパートの合計と同じ
TOTAL_PARAMETERS = {c: sum(p.numel() for p in m.parameters()) for c, m in _models.items()}
_hashes = {c: trunk_hash(m) for c, m in _models.items()}
assert len(set(_hashes.values())) == 1, "同じシードの条件どうしで、順伝播ネットワーク以外の初期値が一致しない"
assert trunk_hash(build_model("M8", 1)) != _hashes["M8"], "異なるシードで初期値が同じ(確認の検出力がない)"
_m8, _n8 = _models["M8"].state_dict(), _models["N8"].state_dict()
assert all(torch.equal(_m8[k], _n8[k]) for k in _m8), "M8 と N8 の初期値が一致しない"
assert count_non_embedding_parameters(_models["D"]) == 3_149_056  # 008 のモデルと同じ
print(
    f"順伝播ネットワーク以外のパラメータ数(全条件で同じ): {_trunk_parameters['D']:,}。順伝播ネットワークのパラメータ数 "
    f"{dumps_compact_json(_feed_forward_parameters)}、総パラメータ数 {dumps_compact_json(TOTAL_PARAMETERS)}"
)
print(
    f"トークンあたりの順伝播ネットワークの積和の回数(1 層、ルーターを除く): D・MoE {FEED_FORWARD_MULTIPLY_ADDS['D']:,}、"
    f"W {FEED_FORWARD_MULTIPLY_ADDS['W']:,}。ルーターの積和の回数は E x d_model(E = 16 で順伝播ネットワークの "
    f"{16 * D_MODEL / FEED_FORWARD_MULTIPLY_ADDS['D']:.2%})"
)
print("同じシードの条件どうしで順伝播ネットワーク以外の初期値が一致、M8 と N8 は全パラメータの初期値が一致: OK")

# --- モデル: E = 1 の MoE と密なモデルの出力の一致、補助損失の集約 ---
with torch.no_grad():
    _dense_model = _models["D"]
    _moe_model = build_gpt(functools.partial(MixtureOfExpertsFeedForward, D_MODEL, SWIGLU_D_FF, 1))
    _state = {k: v for k, v in _dense_model.state_dict().items() if not is_feed_forward_key(k)}
    for _i, _block in enumerate(_dense_model.blocks):
        for _name in ("w", "v", "w2"):
            _state[f"blocks.{_i}.feed_forward.{_name}"] = getattr(_block.feed_forward, _name).weight[None]
        _state[f"blocks.{_i}.feed_forward.router_weight"] = _moe_model.blocks[_i].feed_forward.router_weight
    _moe_model.load_state_dict(_state)
    _tokens = EVAL_WINDOWS[:4]
    _logits_dense, _logits_moe = _dense_model(_tokens), _moe_model(_tokens)
    _repeat_model = max_abs_difference(_logits_dense, _dense_model(_tokens))
    _unit_model = float(torch.finfo(torch.float32).eps) * float(_logits_dense.abs().max())
    _difference = max_abs_difference(_logits_dense, _logits_moe)
    assert _difference <= 16 * max(_repeat_model, _reorder32, _unit_model), _difference
    assert _dense_model.auxiliary_losses() == {}
    _m8_model = _models["M8"]
    _m8_model(_tokens)
    _auxiliary = _m8_model.auxiliary_losses()
    for _name in (LOAD_BALANCING_LOSS_NAME, ROUTER_Z_LOSS_NAME):
        _manual = sum(float(l.auxiliary_losses[_name]) for l in find_moe_layers(_m8_model))
        # FP32 の 4 個の値の和。許容の誤差は FP32 の丸めの単位 x 値の大きさ の 16 倍
        assert abs(float(_auxiliary[_name]) - _manual) <= 16 * float(torch.finfo(torch.float32).eps) * abs(_manual)
    assert len(find_moe_layers(_m8_model)) == NUM_LAYERS
print(
    f"E = 1 の MoE のモデルと密なモデルの logits の差 {_difference:.2e}(bit 単位で一致: {torch.equal(_logits_dense, _logits_moe)}、"
    f"閾値 {16 * max(_repeat_model, _reorder32, _unit_model):.2e})、補助損失は {NUM_LAYERS} 層の合計: OK"
)
del _dense_model, _moe_model, _m8_model


# --- 破棄なしの評価は、バッチの組み方に依存しない(実行するデバイス、FP32) ---
def _window_nll(model, windows: torch.Tensor, batch_size: int) -> np.ndarray:
    return evaluate_window_negative_log_likelihoods(model, windows, device, batch_size=batch_size).numpy()


_check_windows = EVAL_WINDOWS[: 2 * EVAL_BATCH_WINDOWS]
_dense_on_device = _models["D"].to(device)
_dense_batched = _window_nll(_dense_on_device, _check_windows, EVAL_BATCH_WINDOWS)
_dense_reference = float(np.abs(_dense_batched - _window_nll(_dense_on_device, _check_windows, 1)).max())
_dense_repeat = float(np.abs(_dense_batched - _window_nll(_dense_on_device, _check_windows, EVAL_BATCH_WINDOWS)).max())
_models["D"].cpu()
_rows = []
BATCH_INDEPENDENCE = {}
for _c in ("M8", "M16"):
    _model = _models[_c].to(device)
    with torch.no_grad():
        for _layer in find_moe_layers(_model):
            _layer.router_weight.mul_(20.0)  # ルーターを鋭くして、割り当てを偏らせる
    assert all(l.eval_capacity_factor is None for l in find_moe_layers(_model))
    _batched = _window_nll(_model, _check_windows, EVAL_BATCH_WINDOWS)
    _repeat = float(np.abs(_batched - _window_nll(_model, _check_windows, EVAL_BATCH_WINDOWS)).max())
    _single = _window_nll(_model, _check_windows, 1)
    _difference = float(np.abs(_batched - _single).max())
    _unit = float(np.finfo(np.float32).eps) * float(np.abs(_batched).max())
    _tolerance = 16 * max(_repeat, _dense_repeat, _dense_reference, _unit)
    assert _difference <= _tolerance, (_c, _difference, _tolerance)
    assert all(float(l.last_routing["dropped_fraction"]) == 0.0 for l in find_moe_layers(_model))
    # 旧基準(評価時にも capacity factor 2.0 で破棄する)では、同じ比較の差が閾値を超える(確認の検出力)
    for _layer in find_moe_layers(_model):
        _layer.eval_capacity_factor = REFERENCE_EVAL_CAPACITY_FACTOR
    _old_batched = _window_nll(_model, _check_windows, EVAL_BATCH_WINDOWS)
    _old_dropped = max(float(l.last_routing["dropped_fraction"]) for l in find_moe_layers(_model))
    _old_difference = float(np.abs(_old_batched - _window_nll(_model, _check_windows, 1)).max())
    assert _old_dropped > 0 and _old_difference > _tolerance, (_c, _old_dropped, _old_difference)
    BATCH_INDEPENDENCE[_c] = {
        "difference": _difference, "repeat": _repeat, "tolerance": _tolerance, "old_difference": _old_difference,
    }
    _rows.append(
        f"{_c}: 16 個ずつと 1 窓ずつの差 {_difference:.2e}(同じ経路の 2 回の差 {_repeat:.2e}、丸めの単位 x 値 {_unit:.2e}、閾値 {_tolerance:.2e})、"
        f"旧基準(破棄あり、最後のバッチの破棄 {_old_dropped:.3f})では差 {_old_difference:.2e}"
    )
    _models[_c].cpu()
print(
    f"破棄なしの評価の窓ごとの負の対数尤度(nats、窓 {len(_check_windows)} 個)。基準: 密なモデルの 16 個ずつと 1 窓ずつの差 "
    f"{_dense_reference:.2e}・同じ経路の 2 回の差 {_dense_repeat:.2e}。" + "。".join(_rows) + ": OK"
)
del _models, _model, _dense_on_device
empty_device_cache()


# --- ミニバッチの順序は条件によらない / 係数 0 の補助損失は補助損失なしと同じ学習になる(CPU、3 ステップ) ---
class _RecordingData:
    # get_random_batch() が切り出した区間の先頭位置を記録する(Tensor 以外の経路を通る)
    def __init__(self, data: np.ndarray) -> None:
        self.data, self.starts = data, []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: slice) -> np.ndarray:
        self.starts.append(item.start)
        return self.data[item]


def _short_training(condition: str, coefficients, data) -> dict:
    model = build_model(condition, 0)
    return train_language_model(
        model, data, None, None, 0, num_steps=3, batch_size=4, sequence_length=64, learning_rate=1e-3, eval_interval=10**9,
        device="cpu", seed=DATA_SEED_BASE, optimizer=AdamW(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY, foreach=True),
        gradient_clip_threshold=GRADIENT_CLIP_THRESHOLD, auxiliary_loss_coefficients=coefficients,
    )


_numpy_ids = TRAIN_IDS[:200_000].numpy()
_order = {}
for _c in ("D", "M8", "N8"):
    _recorder = _RecordingData(_numpy_ids)
    _short_training(_c, None, _recorder)
    _order[_c] = list(_recorder.starts)
assert _order["D"] == _order["M8"] == _order["N8"] and len(_order["D"]) == 3 * 4 * 2
_without = _short_training("M8", None, _numpy_ids)
_zero = _short_training("M8", {LOAD_BALANCING_LOSS_NAME: 0.0, ROUTER_Z_LOSS_NAME: 0.0}, _numpy_ids)
_with = _short_training("M8", {LOAD_BALANCING_LOSS_NAME: LOAD_BALANCING_ALPHA, ROUTER_Z_LOSS_NAME: ROUTER_Z_COEFFICIENT}, _numpy_ids)
assert _without["train_loss"] == _zero["train_loss"] and _without["gradient_norm"] == _zero["gradient_norm"]
assert _with["gradient_norm"] != _without["gradient_norm"], "補助損失が勾配に効いていない"
assert "auxiliary_losses" not in _without and set(_zero["auxiliary_losses"]) == {LOAD_BALANCING_LOSS_NAME, ROUTER_Z_LOSS_NAME}
print("同じシードの D・M8・N8 でミニバッチの切り出し位置が一致、係数 0 の補助損失は補助損失なしと同じ学習、係数が正なら勾配が変わる: OK")

# --- 既存モジュールの後方互換性(変更前のコミットとの bit 単位の一致) ---
_REFERENCE_COMMIT = "9f10e22"  # src/models/gpt.py・src/training/trainer.py を 022 で変更する前の最後のコミット
_WORKTREE_DIR = ROOT / ".cache" / "_022_backward_compat_worktree"
_compat_script = '''
import functools
import hashlib
import inspect
import json
import sys

import torch

from src.layers.feedforward import SwiGLUFeedForwardNetwork
from src.layers.normalization import RMSNorm
from src.layers.positional_encoding import RotaryPositionEmbedding
from src.models.gpt import GPTLanguageModel
from src.training.optimizer import AdamW
from src.training.schedule import compute_warmup_cosine_learning_rate
from src.training.trainer import evaluate_bits_per_byte, train_language_model

print("SRC_FILE:" + __import__("src").__file__, file=sys.stderr)
print("HAS_AUXILIARY:" + str(hasattr(GPTLanguageModel, "auxiliary_losses")), file=sys.stderr)
print("HAS_EVALUATION_FN:" + str("evaluation_fn" in inspect.signature(train_language_model).parameters), file=sys.stderr)


def h(t):
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()


def build(swiglu):
    torch.manual_seed(7)
    factory = functools.partial(SwiGLUFeedForwardNetwork, 64, 96) if swiglu else None
    return GPTLanguageModel(
        300, 64, 2, 4, 128, 32,
        positional_transform=RotaryPositionEmbedding(16, max_position=32) if swiglu else None,
        normalization_factory=RMSNorm if swiglu else None, feed_forward_factory=factory,
    )


out = {}
generator = torch.Generator().manual_seed(0)
data = torch.randint(0, 300, (4000,), generator=generator)
windows = torch.randint(0, 300, (6, 32), generator=generator)
mask = torch.ones(6, 32, dtype=torch.bool)
for swiglu in (True, False):
    key = "swiglu" if swiglu else "default"
    model = build(swiglu)
    logits = model(windows)
    out[key + "_logits"] = h(logits)
    torch.nn.functional.cross_entropy(logits.reshape(-1, 300), windows.reshape(-1)).backward()
    out[key + "_gradients"] = {n: h(p.grad) for n, p in model.named_parameters() if p.grad is not None}
    out[key + "_generate"] = model.generate(windows[:1, :4], 6, temperature=0.0, use_cache=True).tolist()
    out[key + "_bits_per_byte"] = evaluate_bits_per_byte(model, windows, mask, 500, "cpu")
    # 006 の既定の引数の学習
    model = build(swiglu)
    history = train_language_model(model, data, windows, mask, 500, 6, 4, 32, 1e-3, 4, "cpu", 3)
    out[key + "_history_default"] = history
    out[key + "_state_default"] = {n: h(t) for n, t in model.state_dict().items()}
    # 007 以降の引数(AdamW・スケジュール・gradient clipping・最終ステップの評価)
    model = build(swiglu)
    schedule = functools.partial(
        compute_warmup_cosine_learning_rate, warmup_steps=2, total_steps=6, peak_learning_rate=1e-3, min_learning_rate=1e-5
    )
    history = train_language_model(
        model, data, windows, mask, 500, 6, 4, 32, 1e-3, 4, "cpu", 3, optimizer=AdamW(model.parameters(), lr=1e-3, weight_decay=0.1),
        learning_rate_schedule=schedule, gradient_clip_threshold=0.5, evaluate_at_final_step=True,
    )
    out[key + "_history_full"] = history
    out[key + "_state_full"] = {n: h(t) for n, t in model.state_dict().items()}
print(json.dumps(out))
'''
subprocess.run(["git", "worktree", "remove", "--force", str(_WORKTREE_DIR)], capture_output=True)
_r = subprocess.run(["git", "worktree", "add", "--detach", str(_WORKTREE_DIR), _REFERENCE_COMMIT], capture_output=True, text=True)
assert _r.returncode == 0, _r.stderr
(_WORKTREE_DIR / "_compat_script.py").write_text(_compat_script)
(ROOT / "_compat_script.py").write_text(_compat_script)
try:
    _ref = subprocess.run([sys.executable, "_compat_script.py"], cwd=_WORKTREE_DIR, capture_output=True, text=True)
    assert _ref.returncode == 0, _ref.stderr
    assert "SRC_FILE:" + str((_WORKTREE_DIR / "src" / "__init__.py").resolve()) in _ref.stderr, _ref.stderr
    assert "HAS_AUXILIARY:False" in _ref.stderr and "HAS_EVALUATION_FN:False" in _ref.stderr, "参照側が変更後の実装を読み込んでいる"
    _new = subprocess.run([sys.executable, "_compat_script.py"], cwd=ROOT, capture_output=True, text=True)
    assert _new.returncode == 0, _new.stderr
    assert "SRC_FILE:" + str((ROOT / "src" / "__init__.py").resolve()) in _new.stderr, _new.stderr
    assert "HAS_AUXILIARY:True" in _new.stderr and "HAS_EVALUATION_FN:True" in _new.stderr, "現行側が変更後の実装を読み込んでいない"
    _ref_out, _new_out = json.loads(_ref.stdout), json.loads(_new.stdout)
    assert set(_ref_out) == set(_new_out)
    _mismatch = [k for k in _ref_out if _ref_out[k] != _new_out[k]]
    assert not _mismatch, _mismatch
    assert set(_new_out["swiglu_history_full"]) == set(_ref_out["swiglu_history_full"])  # 履歴のキーも増えていない
    print(
        f"参照コミット {_REFERENCE_COMMIT}(worktree 配下の src を読み込んだことを確認)と現行の src で、GPTLanguageModel の logits・勾配・"
        f"生成、evaluate_bits_per_byte() の値、train_language_model() の履歴(全キー・全ステップ)と学習後の重みが、既定の引数と "
        f"007 以降の引数の両方で bit 単位で一致(比べた項目 {len(_ref_out)} 個、不一致 0 個): OK"
    )
finally:
    (ROOT / "_compat_script.py").unlink(missing_ok=True)
    subprocess.run(["git", "worktree", "remove", "--force", str(_WORKTREE_DIR)], capture_output=True)
CHECK_SECONDS = time.time() - _t0_checks
print(f"単体テストと不変条件の確認 {CHECK_SECONDS:.1f} 秒")
```

    許容の閾値の基準(FP64): 同じ経路の 2 回の差 0.00e+00、まとめて計算と 1 個ずつ計算の差 5.55e-16、丸めの単位 x 出力の最大値 1.59e-16 -> 閾値 8.88e-15
    振り分け・集約と素朴なループの一致(閾値以内): (E=4, k=1, c=1.0): 差 2.5e-16・破棄 0.067、(E=4, k=1, c=1.25): 差 2.5e-16・破棄 0.000、(E=4, k=1, c=4.0): 差 2.5e-16・破棄 0.000、(E=8, k=2, c=1.0): 差 3.3e-16・破棄 0.062、(E=8, k=2, c=1.25): 差 3.3e-16・破棄 0.004、(E=8, k=2, c=8.0): 差 3.3e-16・破棄 0.000、(E=16, k=1, c=1.25): 差 3.9e-16・破棄 0.058、(E=1, k=1, c=1.0): 差 3.9e-16・破棄 0.000: OK
    容量の計算と枠の割り当て(先着順、超過は捨て枠): OK
    E = 1 の MoE 層と密な SwiGLU: 同じ乱数の状態からの初期値が bit 単位で一致、出力の差 0.00e+00(bit 単位で一致: True)。基準: 同じ経路の 2 回の差 0.00e+00、並べ替えの差 0.00e+00、丸めの単位 1.06e-07 -> 閾値 1.70e-06: OK
    破棄の因果性(破棄の割合 0.125。位置 40 以降の入力を変えても、位置 0〜39 の出力は変わらない): OK
    3.5 節の数値例: sum_i f_i P_i = 0.4612 < 0.5。ロジットがすべて 0(ルーターと補助損失は FP32): N sum_i f_i P_i - 1 = +0.00e+00、L_z - (log N)^2 = -4.73e-07(許容 8.25e-06。FP32 の丸めの 1 単位は 5.15e-07): OK
    ルーターへの勾配(ゲートを通じて流れる。負荷分散損失の勾配はルーターにだけ流れる。top-1 のゲートの正規化は拒否): OK
    autocast(cuda、torch.float16)の中で、エキスパートの行列積は低精度、ルーターのロジットと補助損失は FP32: OK
    順伝播ネットワーク以外のパラメータ数(全条件で同じ): 3,148,032。順伝播ネットワークのパラメータ数 {
      "D": 2098176,
      "M2": 4198400,
      "M4": 8396800,
      "M8": 16793600,
      "M16": 33587200,
      "N8": 16793600,
      "W": 16785408
    }、総パラメータ数 {
      "D": 5246208,
      "M2": 7346432,
      "M4": 11544832,
      "M8": 19941632,
      "M16": 36735232,
      "N8": 19941632,
      "W": 19933440
    }
    トークンあたりの順伝播ネットワークの積和の回数(1 層、ルーターを除く): D・MoE 524,544、W 4,196,352。ルーターの積和の回数は E x d_model(E = 16 で順伝播ネットワークの 0.78%)
    同じシードの条件どうしで順伝播ネットワーク以外の初期値が一致、M8 と N8 は全パラメータの初期値が一致: OK
    E = 1 の MoE のモデルと密なモデルの logits の差 0.00e+00(bit 単位で一致: True、閾値 3.01e-06)、補助損失は 4 層の合計: OK
    破棄なしの評価の窓ごとの負の対数尤度(nats、窓 32 個)。基準: 密なモデルの 16 個ずつと 1 窓ずつの差 2.19e-05・同じ経路の 2 回の差 0.00e+00。M8: 16 個ずつと 1 窓ずつの差 2.10e-05(同じ経路の 2 回の差 0.00e+00、丸めの単位 x 値 2.78e-04、閾値 4.44e-03)、旧基準(破棄あり、最後のバッチの破棄 0.130)では差 1.46e+00。M16: 16 個ずつと 1 窓ずつの差 1.34e-05(同じ経路の 2 回の差 0.00e+00、丸めの単位 x 値 2.77e-04、閾値 4.44e-03)、旧基準(破棄あり、最後のバッチの破棄 0.254)では差 2.81e+00: OK
    同じシードの D・M8・N8 でミニバッチの切り出し位置が一致、係数 0 の補助損失は補助損失なしと同じ学習、係数が正なら勾配が変わる: OK
    参照コミット 9f10e22(worktree 配下の src を読み込んだことを確認)と現行の src で、GPTLanguageModel の logits・勾配・生成、evaluate_bits_per_byte() の値、train_language_model() の履歴(全キー・全ステップ)と学習後の重みが、既定の引数と 007 以降の引数の両方で bit 単位で一致(比べた項目 16 個、不一致 0 個): OK
    単体テストと不変条件の確認 35.3 秒


### 5.5 学習と評価のヘルパー

- `evaluate_model()`: 評価窓ごとの負の対数尤度(FP32、破棄なし)と、MoE では窓ごと・層ごと・エキスパートごとの割り当ての個数を
  返す。窓は先頭から 16 個ずつ処理する。あわせて、**同じ 16 個ずつのバッチを capacity factor 2.0 で処理したとすれば破棄された
  はずの割り当ての割合** を、割り当ての個数から層ごとに計算する(バッチ $j$ の $T_j$ トークン(3 節の $T$ と同じ意味のバッチのトークン数で、学習ステップ数ではない)のうちエキスパート $i$ への割り当てを
  $n_{ij}$、容量を $C_j = \lceil 2 T_j / E \rceil$ として、$\sum_j \sum_i \max(0, n_{ij} - C_j) / \sum_j T_j$)。
- `summarize_routing()`: 評価集合全体の割り当てから、層ごとの $f_i$ の正規化エントロピー $H(f) / \log E$ とその層平均 $h$、
  最大のエキスパートの負荷の割合、ほとんど選ばれないエキスパートの数($f_i < 0.1 / E$)、破棄された割り当ての割合、
  トークンごとのルーターの確率の最大値の平均、ルーターのロジットのノルムの平均を作る。
- `train_run()`: 条件・シード・学習率・ステップ数を受け取って学習し、途中の評価の位置(5.2 節)ごとの bits-per-byte と負荷の診断量、
  学習の最終ステップの重みでの評価窓ごとの負の対数尤度と割り当ての個数を記録する。本番の学習では、最終ステップの重みで
  学習用の部分の窓の bits-per-byte も測る(汎化の差の診断量)。学習時の破棄の割合は、評価と評価の間の
  学習ステップについて積算した値を記録する。モデルは評価の直後に破棄する。


```python
def evaluate_model(model, windows: torch.Tensor, window_bytes: np.ndarray, collect_tokens: bool = False) -> dict:
    layers = find_moe_layers(model)
    counts = token_routing = None
    if layers:
        num_experts = layers[0].num_experts
        counts = torch.zeros(len(windows), len(layers), num_experts, dtype=torch.long)
        if collect_tokens:
            token_routing = torch.zeros(len(windows), len(layers), windows.size(1), dtype=torch.long)
        for layer in layers:
            layer.track_statistics = True
            layer.reset_statistics("eval")

    def callback(start: int, end: int) -> None:
        for index, layer in enumerate(layers):
            expert = layer.last_expert_index[:, 0].view(end - start, -1)  # (窓, 位置): top-1 の割り当て先(破棄の前)
            counts[start:end, index] = functional.one_hot(expert, num_experts).sum(dim=1).cpu()
            if collect_tokens:
                token_routing[start:end, index] = expert.cpu()

    nll = evaluate_window_negative_log_likelihoods(
        model, windows, device, batch_size=EVAL_BATCH_WINDOWS, batch_callback=callback if layers else None
    ).numpy()
    result = {"nll": nll, "bits_per_byte": float(nll.sum() / math.log(2) / window_bytes.sum())}
    if layers:
        result["counts"] = counts.numpy()
        result["layer_statistics"] = [layer.routing_statistics("eval") for layer in layers]
        assert int(counts.sum()) == len(layers) * windows.numel() == sum(s["tokens"] for s in result["layer_statistics"])
        assert all(s["dropped_fraction"] == 0.0 for s in result["layer_statistics"]), "評価で破棄が起きている"
        result["reference_dropped"] = reference_dropped_fraction(result["counts"], windows.size(1))
        if collect_tokens:
            result["token_routing"] = token_routing.numpy()
        for layer in layers:
            layer.reset_statistics("eval")
    return result


def reference_dropped_fraction(counts: np.ndarray, tokens_per_window: int) -> list[float]:
    # 窓 EVAL_BATCH_WINDOWS 個ずつのバッチを capacity factor 2.0 で処理したとすれば破棄されたはずの割り当ての割合(層ごと)。
    # counts: (窓, 層, E) の割り当ての個数。top-1 なので、エキスパート i の破棄は max(0, 割り当ての個数 - 容量)
    num_experts = counts.shape[2]
    overflow = np.zeros(counts.shape[1])
    for start in range(0, counts.shape[0], EVAL_BATCH_WINDOWS):
        batch = counts[start : start + EVAL_BATCH_WINDOWS]
        capacity = compute_expert_capacity(batch.shape[0] * tokens_per_window, num_experts, TOP_K, REFERENCE_EVAL_CAPACITY_FACTOR)
        overflow += np.maximum(batch.sum(axis=0) - capacity, 0).sum(axis=1)
    return (overflow / (counts.shape[0] * tokens_per_window)).tolist()


def summarize_routing(layer_statistics: list[dict], reference_dropped: list[float]) -> dict:
    fractions = np.array([s["assignment_fraction"] for s in layer_statistics])  # (層, E)
    num_experts = fractions.shape[1]
    entropies = [compute_normalized_entropy(f) for f in fractions]
    return {
        "h": float(np.mean(entropies)),  # f_i の正規化エントロピーの層平均
        "entropy": entropies,
        "max_load": fractions.max(axis=1).tolist(),
        "rare_experts": (fractions < RARE_EXPERT_FACTOR / num_experts).sum(axis=1).tolist(),
        "dropped": reference_dropped,  # capacity factor 2.0 なら破棄されたはずの割合(評価そのものは破棄なし)
        "max_probability": [s["mean_max_probability"] for s in layer_statistics],
        "logit_norm": [s["mean_router_logit_norm"] for s in layer_statistics],
    }


def train_run(
    condition: str,
    seed_index: int,
    learning_rate: float,
    num_steps: int,
    eval_windows: torch.Tensor,
    eval_bytes: np.ndarray,
    eval_steps: tuple[int, ...],
    collect_tokens: bool = False,
    measure_train_probe: bool = False,
) -> dict:
    spec = CONDITIONS[condition]
    is_moe = spec["kind"] == "moe"
    start = time.time()
    model = build_model(condition, seed_index)
    record = {
        "condition": condition, "seed": seed_index, "learning_rate": learning_rate, "num_steps": num_steps,
        "trunk_hash": trunk_hash(model),
    }
    model = model.to(device)
    set_statistics_tracking(model, True)
    layers = find_moe_layers(model)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY, foreach=True)
    schedule = functools.partial(
        compute_warmup_cosine_learning_rate,
        warmup_steps=warmup_steps_for(num_steps),
        total_steps=num_steps,
        peak_learning_rate=learning_rate,
        min_learning_rate=learning_rate * MIN_LEARNING_RATE_RATIO,
    )
    last_evaluation: dict = {}

    def evaluation_fn(m) -> tuple[float, dict]:
        train_statistics = [layer.routing_statistics("train") for layer in layers]
        for layer in layers:
            layer.reset_statistics("train")
        result = evaluate_model(m, eval_windows, eval_bytes)
        last_evaluation.clear()
        last_evaluation.update(result)
        diagnostics = {}
        if is_moe:
            diagnostics = summarize_routing(result["layer_statistics"], result["reference_dropped"])
            diagnostics["train_dropped"] = [s["dropped_fraction"] for s in train_statistics]  # 直前の評価からの学習ステップ
        return result["bits_per_byte"], diagnostics

    history = train_language_model(
        model, TRAIN_IDS, None, None, 0,
        num_steps=num_steps,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        learning_rate=learning_rate,
        eval_interval=num_steps,
        device=device,
        seed=DATA_SEED_BASE + seed_index,
        optimizer=optimizer,
        learning_rate_schedule=schedule,
        gradient_clip_threshold=GRADIENT_CLIP_THRESHOLD,
        autocast_dtype=torch.float16 if USE_FP16_AUTOCAST else None,
        loss_scaler=DynamicLossScaler(INIT_LOSS_SCALE, growth_interval=LOSS_SCALE_GROWTH_INTERVAL) if USE_FP16_AUTOCAST else None,
        evaluate_at_final_step=True,
        auxiliary_loss_coefficients=(
            {LOAD_BALANCING_LOSS_NAME: spec["alpha"], ROUTER_Z_LOSS_NAME: ROUTER_Z_COEFFICIENT} if is_moe else None
        ),
        eval_steps=eval_steps,
        evaluation_fn=evaluation_fn,
    )
    assert history["eval_step"][-1] == num_steps == len(history["train_loss"])
    window = final_loss_window(num_steps)
    train_loss = np.array(history["train_loss"], dtype=np.float64)
    record |= {
        "bits_per_byte": history["eval_bits_per_byte"][-1],  # 学習の最終ステップの重みで評価した値
        "nll": last_evaluation["nll"],
        "eval_step": history["eval_step"],
        "eval_bits_per_byte": history["eval_bits_per_byte"],
        "eval_diagnostics": history["eval_diagnostics"],
        "train_loss": train_loss.astype(np.float32),
        "final_train_loss": float(train_loss[-window:].mean()),
        "finite": bool(np.isfinite(train_loss).all()),
        "skipped_steps": int(sum(history["step_skipped"])),
        "clip_rate": float(np.mean(history["gradient_clip_triggered"])),
        "final_loss_scale": float(history["loss_scale"][-1]),
    }
    if is_moe:
        record["counts"] = last_evaluation["counts"]
        record["routing"] = history["eval_diagnostics"][-1]
        record["auxiliary"] = {k: float(np.mean(v[-window:])) for k, v in history["auxiliary_losses"].items()}
        if collect_tokens:
            record["token_routing"] = evaluate_model(model, eval_windows, eval_bytes, collect_tokens=True)["token_routing"]
    if measure_train_probe:  # 汎化の差の診断量: 学習用の部分の窓の bits-per-byte(最終ステップの重み)
        record["train_bits_per_byte"] = evaluate_model(model, TRAIN_PROBE_WINDOWS, TRAIN_PROBE_WINDOW_BYTES)["bits_per_byte"]
    record["seconds"] = time.time() - start
    del model, optimizer
    empty_device_cache()
    return record


def learning_precondition(record: dict) -> bool:
    # P1(6.1 節): 損失がすべてのステップで有限で、最後の区間の訓練損失が ln(V) の P1_LOSS_RATIO 倍以下
    return record["finite"] and record["final_train_loss"] <= P1_LOSS_RATIO * UNIFORM_LOSS


def load_precondition(record: dict) -> bool:
    # P2(6.1 節): 評価集合で、capacity factor 2.0 なら破棄されたはずの割り当ての割合と、最大のエキスパートの負荷の割合が、どの層でも閾値以下
    routing = record["routing"]
    experts = CONDITIONS[record["condition"]]["experts"]
    return max(routing["dropped"]) <= P2_DROPPED_MAX and max(routing["max_load"]) <= min(1.0, P2_LOAD_FACTOR / experts)


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

https://github.com/kojikojiprg/ai-theories/blob/main/theories/06_architectures/022_mixture_of_experts.ipynb
