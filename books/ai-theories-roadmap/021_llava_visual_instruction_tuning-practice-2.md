---
title: "LLaVA 型 Vision-Language 連結と視覚指示チューニング / LLaVA-style Vision-Language Connection and Visual Instruction Tuning(実装・実験編 2/3)"
---

この記事は後編(実装・実験編 2/3)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/021_llava_visual_instruction_tuning-practice-1)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/021_llava_visual_instruction_tuning-practice-3)。

### 5.5 モデルの構築と不変条件の確認

学習の部品(`build_llava()`・`add_lora()`・`train_stage1()`・`train_stage2()`)をここで定義し、次を確かめる。

- **埋め込みの列からの順伝播**: `language_model_hidden_states()`にトークン埋め込み`token_embedding(ids)`を渡して`lm_head`を掛けた
  logits が、既存の`GPTLanguageModel.forward(ids)`と bit 単位で一致すること(LoRA を掛ける前と後)。
- **視覚トークンの差し込み**: 枠の位置のトークン ID を別の値に変えても、隠れ状態が bit 単位で変わらないこと(差し込みで置き換わる)。
  枠の外のトークンを変えると変わること(確認の検出力)。右側のパディングを変えても、応答の位置の logits が変わらないこと(因果マスク)。
- **パッチ特徴の層**: `vision_patch_features(..., L - 1)`が、埋め込みと先頭 $L - 1$ ブロックを手で順に通したもののパッチの位置と一致し、
  その系列([CLS] を含む)に最終ブロックと最終正規化を掛けた [CLS] の表現が、020 の`forward_features()`と一致すること
  (取り出しているのが「最終層の一つ前」であることの確認)。
- **平均プール**: 平均プールの視覚トークンが、パッチトークンの平均に projection 層を掛けたものと一致すること。
- **損失**: `response_loss()`が、全位置の logits から応答部分の対数確率だけを足した FP64 の参照実装と一致し(相対誤差 $10^{-12}$ 以下)、
  016 の`compute_instruction_tuning_loss(..., include_prompt_loss=False)`(内部で FP32 に変換して計算する)とも FP32 の精度
  (相対誤差 $10^{-5}$ 以下)で一致すること(応答の位置だけを語彙に射影する最適化の確認)。
- **LoRA の初期値**: LoRA を掛けた直後の出力が、掛ける前と bit 単位で一致すること($B = 0$)。
- **凍結**: 第 1 段階を数ステップ学習した後、projection 層以外(言語モデル・画像 encoder)のパラメータが bit 単位で変わらないこと。
  第 2 段階では、projection 層と LoRA 以外が変わらないこと。
- **生成**: KV キャッシュを使う`greedy_generate_with_visual_tokens()`の出力が、キャッシュを使わずに毎ステップ全系列を順伝播して
  argmax をとる素朴な生成と一致すること(パッチトークン全体・平均プール、学習途中のモデル)。
- **対応のある初期値**: 同じシードで構築した P・R・L・M の projection 層の初期値、LoRA の初期値が bit 単位で一致すること。


```python
_t0_checks = time.time()


def build_llava(pooling: str, seed: int) -> LlavaStyleModel:
    # projection 層の初期化はシード s の PROJECTION_SEED_BASE + s で決める(条件によらず同じ形状・同じ値)
    torch.manual_seed(PROJECTION_SEED_BASE + seed)
    projection = VisualProjection(VISION_DIM, D_MODEL, PROJECTION_KIND)
    return LlavaStyleModel(build_language_model(), projection, VISUAL_START, pooling).to(device)


def add_lora(model: LlavaStyleModel, seed: int) -> list[str]:
    # 言語モデルの Query・Value 射影に LoRA を掛ける(A の初期化は LORA_SEED_BASE + s)。projection 層は学習可能のまま
    torch.manual_seed(LORA_SEED_BASE + seed)
    return apply_lora(model.language_model, LORA_TARGET_MODULES, rank=LORA_RANK, alpha=LORA_ALPHA)


def trainable_parameters(model: LlavaStyleModel) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def train_steps(model, data, steps, learning_rate, data_seed, evaluation_steps=(), evaluation_fn=None, batch_size=BATCH_SIZE):
    params = trainable_parameters(model)
    optimizer = AdamW(params, lr=learning_rate, weight_decay=WEIGHT_DECAY, foreach=True)
    warmup = warmup_steps_for(steps)

    def schedule(step: int) -> float:
        return compute_warmup_cosine_learning_rate(step, warmup, steps, learning_rate, learning_rate * MIN_LEARNING_RATE_RATIO)

    batches = make_step_batches(len(data), steps, batch_size, data_seed)
    history = train_visual_instruction(
        model, data, FEATURES["train"], batches, optimizer, params, schedule, GRADIENT_CLIP_THRESHOLD, evaluation_steps, evaluation_fn
    )
    history["batch_sha256"] = hashlib.sha256(batches.tobytes()).hexdigest()
    history["steps"] = steps
    history["trainable_names"] = sorted(n for n, p in model.named_parameters() if p.requires_grad)
    return history


def train_stage1(model, seed, learning_rate, steps=STAGE1_STEPS):
    # 第 1 段階: projection 層のみ(言語モデルは凍結)。キャプションの事例
    assert all(not p.requires_grad for p in model.language_model.parameters())
    return train_steps(model, ENCODED[(model.pooling, "caption", "train")], steps, learning_rate, STAGE1_DATA_SEED_BASE + seed)


def validation_nll(model, kind: str) -> float:
    # 検証集合の答え(またはキャプション)のトークンあたりの負の対数尤度(教師強制)
    result = evaluate_response_negative_log_likelihood(
        model, ENCODED[(model.pooling, kind, "validation")], FEATURES["validation"], EVAL_BATCH_SIZE
    )
    return float(result["sum"].sum() / result["count"].sum())


def train_stage2(model, seed, learning_rate, steps, intermediate=True):
    # 第 2 段階: projection 層 + LoRA。質問の事例。途中の評価は検証集合の負の対数尤度(診断量)
    evaluation_steps = intermediate_eval_steps_for(steps) if intermediate else ()
    return train_steps(
        model, ENCODED[(model.pooling, "question", "train")], steps, learning_rate, STAGE2_DATA_SEED_BASE + seed,
        evaluation_steps, lambda m: {"validation_nll": validation_nll(m, "question")},
    )


def visual_token_norm(model) -> float:
    # 検証用の画像の視覚トークンの平均ノルム(診断量)
    with torch.no_grad():
        return float(model.visual_tokens(FEATURES["validation"][:512]).norm(dim=-1).mean())


def parameter_snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in module.named_parameters()}


# --- 埋め込みの列からの順伝播が forward() と bit 単位で一致する(LoRA の前後) ---
_m = build_llava("patch", 0).eval()
_ids = ENCODED[("patch", "question", "validation")].token_ids[:16]
with torch.no_grad():
    _lm = _m.language_model
    assert torch.equal(_lm.lm_head(language_model_hidden_states(_lm, _lm.token_embedding(_ids))), _lm(_ids))
    _before_lora = _m(_ids, FEATURES["validation"][:16])
    add_lora(_m, 0)
    _m.to(device)
    for _name, _module in _m.language_model.named_modules():  # 検証のため LoRA の B を非零にする(確認の検出力)
        if isinstance(_module, LoRALinear):
            _module.lora_b.normal_(std=0.02, generator=torch.Generator(device=device).manual_seed(1))
    assert torch.equal(_lm.lm_head(language_model_hidden_states(_lm, _lm.token_embedding(_ids))), _lm(_ids))
del _m

# --- LoRA の初期値(B = 0)では出力が変わらない ---
_m = build_llava("patch", 0).eval()
with torch.no_grad():
    _f = FEATURES["validation"][:16]
    _out0 = _m(_ids, _f)
    assert torch.equal(_out0, _before_lora)  # 同じシードで構築すれば同じ出力
    _replaced = add_lora(_m, 0)
    _m.to(device)
    assert torch.equal(_m(_ids, _f), _out0)
    assert len(_replaced) == len(LORA_TARGET_MODULES) * NUM_LAYERS
    assert sum(p.numel() for n, p in _m.language_model.named_parameters() if p.requires_grad) == LORA_PARAMETERS

    # --- 視覚トークンの枠の位置のトークン ID は出力に影響しない / 枠の外は影響する / 右側のパディングは影響しない ---
    _slot = slice(VISUAL_START, VISUAL_START + NUM_PATCHES)
    _changed_slot = _ids.clone()
    _changed_slot[:, _slot] = torch.randint(1, 8000, _changed_slot[:, _slot].shape, generator=torch.Generator().manual_seed(2)).to(device)
    _h0 = _m.hidden_states(_ids, _f)
    assert torch.equal(_m.hidden_states(_changed_slot, _f), _h0)
    _changed_text = _ids.clone()
    _changed_text[:, VISUAL_START + NUM_PATCHES + 1] += 1
    assert (_m.hidden_states(_changed_text, _f) - _h0).abs().max() > 1e-4
    _d = ENCODED[("patch", "question", "validation")]
    _changed_pad = _ids.clone()
    _pad_positions = torch.arange(_ids.size(1), device=device)[None, :] >= _d.lengths[:16, None]
    _changed_pad[_pad_positions] = 777
    _mask = _d.response_target_mask[:16]
    assert torch.equal(_m.response_logits(_changed_pad, _f, _mask)[0], _m.response_logits(_ids, _f, _mask)[0])

# --- パッチ特徴は「最終層の一つ前」の層: 手で通したものと一致し、最終ブロックと最終正規化で forward_features() に戻る ---
with torch.no_grad():
    _x = normalize_scene_images(SCENES["eval"].images[:8].to(device))
    _hidden = VISION.embed(_x)
    for _block in VISION.blocks[:VISION_LAYER]:
        _hidden, _ = _block(_hidden)
    assert torch.equal(_hidden[:, 1:], vision_patch_features(VISION, _x, VISION_LAYER))
    _last, _ = VISION.blocks[VISION_LAYER](_hidden)
    assert VISION_LAYER == VISION_NUM_LAYERS - 1
    assert torch.equal(VISION.final_norm(_last)[:, 0], VISION.forward_features(_x))

# --- 平均プール ---
_mm = build_llava("mean", 0)
with torch.no_grad():
    _vt = _mm.visual_tokens(_f)
    assert _vt.shape == (16, 1, D_MODEL)
    assert torch.equal(_vt, _mm.projection(_f.mean(dim=1, keepdim=True)))
del _mm

# --- 損失: 応答の位置だけを射影した損失が、016 の損失マスクありの損失と FP64 で一致する ---
_cpu_model = copy.deepcopy(_m).to("cpu").double()
_ids_cpu, _f_cpu, _mask_cpu = _ids.cpu(), _f.cpu().double(), _mask.cpu()
# 016 の collate_instruction_batch() と同じ定義の指示部分のマスク(損失マスクありでは損失に使われない)
_prompt_mask = torch.arange(_ids.size(1) - 1)[None, :] < (_d.prompt_lengths[:16, None].cpu() - 1)
with torch.no_grad():
    _full = _cpu_model(_ids_cpu, _f_cpu)  # 全位置の logits(FP64)
    _ours = response_loss(_cpu_model, _ids_cpu, _f_cpu, _mask_cpu)
    # FP64 の参照実装: 全位置の対数確率から、応答部分の予測対象の位置だけを足す
    _log_probs = torch.log_softmax(_full[:, :-1], dim=-1).gather(-1, _ids_cpu[:, 1:, None])[..., 0]
    _manual = -(_log_probs * _mask_cpu).sum() / _mask_cpu.sum()
    # 016 の損失マスクありの損失(内部で FP32 に変換して計算する)
    _reference = compute_instruction_tuning_loss(_full, _ids_cpu, _prompt_mask, _mask_cpu, include_prompt_loss=False)
assert _ours["loss"].dtype == torch.float64 and math.isclose(float(_ours["loss"]), float(_manual), rel_tol=1e-12)
assert math.isclose(float(_ours["loss"]), float(_reference["loss"]), rel_tol=1e-5)
assert int(_ours["count"]) == int(_reference["response_count"])
LOSS_CHECK_DIFF = abs(float(_ours["loss"]) - float(_reference["loss"]))
del _cpu_model

# --- 凍結: 第 1 段階では projection 層だけ、第 2 段階では projection 層と LoRA だけが変わる ---
_frozen_check = {}
for _stage in (1, 2):
    _mc = build_llava("patch", 0)
    if _stage == 2:
        add_lora(_mc, 0)
        _mc.to(device)
    _before = parameter_snapshot(_mc)
    _vision_before = parameter_snapshot(VISION)
    _data = ENCODED[("patch", "caption" if _stage == 1 else "question", "train")]
    _h = train_steps(_mc, _data, 3, 1e-3, 0)
    _changed = {n for n, p in _mc.named_parameters() if not torch.equal(p.detach(), _before[n])}
    _expected = {n for n in _before if n.startswith("projection.") or (_stage == 2 and ".lora_" in n)}
    assert _changed == _expected, (_stage, sorted(_changed ^ _expected)[:5])
    assert all(torch.equal(p, _vision_before[n]) for n, p in VISION.named_parameters())
    assert all(math.isfinite(x) for x in _h["loss"])
    _frozen_check[_stage] = len(_changed)
    del _mc

# --- 生成: KV キャッシュの生成と、キャッシュなしの素朴な生成が一致する(学習途中のモデル) ---


def naive_greedy(model, prompt_ids: list[int], features_one: torch.Tensor, max_new: int) -> list[int]:
    out = []
    ids = torch.tensor([prompt_ids], device=device)
    for _ in range(max_new):
        with torch.no_grad():
            nxt = int(model(ids, features_one[None])[0, -1].argmax())
        out.append(nxt)
        if END_MARKER in tokenizer.decode(out):
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
    return out


_generation_check = {}
for _pooling in ("patch", "mean"):
    _mg = build_llava(_pooling, 0)
    add_lora(_mg, 0)
    _mg.to(device)
    train_steps(_mg, ENCODED[(_pooling, "question", "train")], 40, 3e-3, 0)  # 出力が自明でない状態にする
    _dv = ENCODED[(_pooling, "question", "validation")]
    _sel = torch.arange(0, 64, device=device) * 37
    _prompts = [row[:n] for row, n in zip(_dv.token_ids[_sel].tolist(), _dv.prompt_lengths[_sel].tolist(), strict=True)]
    _feat = FEATURES["validation"][_dv.image_indices[_sel]]
    _cached = greedy_generate_with_visual_tokens(_mg, _prompts, _feat, tokenizer.decode, END_MARKER, MAX_NEW_TOKENS, batch_size=16)
    _naive = [naive_greedy(_mg, p, _feat[i], MAX_NEW_TOKENS) for i, p in enumerate(_prompts)]
    assert _cached == _naive, _pooling
    _generation_check[_pooling] = len(_prompts)
    del _mg

# --- 対応のある初期値: 同じシードの P・R・L・M の projection 層と LoRA の初期値が一致する ---
_init = {}
for _cond, _spec in CONDITIONS.items():
    _mi = build_llava(_spec["pooling"], 3)
    add_lora(_mi, 3)
    _init[_cond] = {n: p.detach().cpu() for n, p in _mi.named_parameters() if p.requires_grad}
    del _mi
for _cond in CONDITIONS:
    assert _init[_cond].keys() == _init["P"].keys() and all(torch.equal(_init[_cond][k], _init["P"][k]) for k in _init["P"])
# L の第 2 段階のミニバッチの先頭 T_2 ステップは、R と同じ(同じシード・同じ規則)
_n_q = len(ENCODED[("patch", "question", "train")])
assert np.array_equal(
    make_step_batches(_n_q, STAGE1_STEPS + STAGE2_STEPS, BATCH_SIZE, 7)[:STAGE2_STEPS], make_step_batches(_n_q, STAGE2_STEPS, BATCH_SIZE, 7)
)
del _m, _init
empty_device_cache()
CHECK_SECONDS = time.time() - _t0_checks
print("埋め込みの列からの順伝播が GPTLanguageModel.forward() と bit 単位で一致(LoRA の前後): OK")
print(f"LoRA を掛けた直後の出力が掛ける前と bit 単位で一致、LoRA の学習可能パラメータ数 {LORA_PARAMETERS:,}: OK")
print("枠の位置のトークン ID は隠れ状態に影響しない・枠の外は影響する・右側のパディングは応答の logits に影響しない: OK")
print(f"パッチ特徴は {VISION_LAYER} ブロックの後(最終層の一つ前)で、最終ブロックと最終正規化で forward_features() に戻る: OK")
print("平均プールの視覚トークン = projection(パッチトークンの平均): OK")
print(f"応答の位置だけを射影した損失が FP64 の参照実装と一致(相対 1e-12)、016 の損失マスクありの損失(FP32 で計算)との差 {LOSS_CHECK_DIFF:.1e}: OK")
print(f"凍結: 第 1 段階で変わったのは projection 層のみ({_frozen_check[1]} テンソル)、第 2 段階は projection 層と LoRA のみ({_frozen_check[2]} テンソル)、画像 encoder は不変: OK")
print(f"KV キャッシュの生成とキャッシュなしの素朴な生成が一致: {_generation_check} 事例: OK")
print("同じシードの P・R・L・M の projection 層と LoRA の初期値が bit 単位で一致、L のミニバッチの先頭 T_2 ステップは R と同じ: OK")
print(f"確認の実行時間 {CHECK_SECONDS:.1f}s")
```

    埋め込みの列からの順伝播が GPTLanguageModel.forward() と bit 単位で一致(LoRA の前後): OK
    LoRA を掛けた直後の出力が掛ける前と bit 単位で一致、LoRA の学習可能パラメータ数 32,768: OK
    枠の位置のトークン ID は隠れ状態に影響しない・枠の外は影響する・右側のパディングは応答の logits に影響しない: OK
    パッチ特徴は 3 ブロックの後(最終層の一つ前)で、最終ブロックと最終正規化で forward_features() に戻る: OK
    平均プールの視覚トークン = projection(パッチトークンの平均): OK
    応答の位置だけを射影した損失が FP64 の参照実装と一致(相対 1e-12)、016 の損失マスクありの損失(FP32 で計算)との差 2.2e-07: OK
    凍結: 第 1 段階で変わったのは projection 層のみ(2 テンソル)、第 2 段階は projection 層と LoRA のみ(18 テンソル)、画像 encoder は不変: OK
    KV キャッシュの生成とキャッシュなしの素朴な生成が一致: {'patch': 64, 'mean': 64} 事例: OK
    同じシードの P・R・L・M の projection 層と LoRA の初期値が bit 単位で一致、L のミニバッチの先頭 T_2 ステップは R と同じ: OK
    確認の実行時間 14.0s


### 5.6 学習と評価のヘルパー

- **1 つの学習**: `(条件, シード)`の組で表す。`run_condition()`は、projection 層の構築 → (第 1 段階)→ LoRA を掛ける → 第 2 段階 →
  評価、を行い、記録を返す。第 1 段階の前後と第 2 段階の開始時・途中(25%・50%・75%)・終わりに検証集合の負の対数尤度を、
  第 1 段階の前後と第 2 段階の後に視覚トークンの平均ノルムを記録する(前提条件 P1 と診断量)。
- **評価**(`purpose="main"`のときのみ、学習の最終ステップの重み): 評価集合の全 5,776 事例を生成で採点し、事例ごとの正誤を記録する。
  続いて、画像を差し替えた評価(前提条件 P2)を行う。
- **較正の学習**(`purpose="calibration"`)は、評価集合を一切評価しない(記録に評価集合の鍵を持たないことを 6.10 節で確かめる)。


```python
def evaluate_generation(model, image_indices=None) -> np.ndarray:
    # 評価集合の各事例を貪欲法で生成して採点し、正誤(bool の配列、QUESTIONS["eval"] の順)を返す
    data = ENCODED[(model.pooling, "question", "eval")]
    texts = generate_answers(model, data, FEATURES["eval"], tokenizer.decode, MAX_NEW_TOKENS, image_indices, EVAL_BATCH_SIZE)
    return np.array([score_generated_response(t, a.split())[1] for t, a in zip(texts, EVAL_ANSWERS, strict=True)])


def accuracy_by_type(correct: np.ndarray) -> dict:
    return {"all": float(correct.mean())} | {t: float(correct[QUESTION_TYPE_OF_EVAL == t].mean()) for t in QUESTION_TYPES}


def run_condition(condition: str, seed: int, lr1: float | None, lr2: float, purpose: str, stage1_projection: dict | None = None) -> dict:
    # 1 つの学習(条件・シード)。stage1_projection を与えると、第 1 段階を行わずにその projection 層から第 2 段階を始める(較正用)
    assert purpose in ("calibration", "main")
    spec = CONDITIONS[condition]
    t0 = time.time()
    model = build_llava(spec["pooling"], seed)
    record = {"condition": condition, "seed": seed, "purpose": purpose, "lr1": lr1, "lr2": lr2, "pooling": spec["pooling"]}
    record["initial_trainable_sha256"] = hashlib.sha256(
        b"".join(p.detach().cpu().numpy().tobytes() for p in model.projection.parameters())
    ).hexdigest()
    record["norm_init"] = visual_token_norm(model)
    if spec["stage1"] and stage1_projection is None:
        record["stage1_nll_before"] = validation_nll(model, "caption")
        record["stage1_history"] = train_stage1(model, seed, lr1)
        record["stage1_nll_after"] = validation_nll(model, "caption")
        record["norm_after_stage1"] = visual_token_norm(model)
    elif stage1_projection is not None:
        model.projection.load_state_dict(stage1_projection)
    add_lora(model, seed)
    model.to(device)
    record["lora_init_sha256"] = hashlib.sha256(
        b"".join(p.detach().cpu().numpy().tobytes() for n, p in model.named_parameters() if ".lora_" in n)
    ).hexdigest()
    record["stage2_nll_start"] = validation_nll(model, "question")
    record["stage2_history"] = train_stage2(model, seed, lr2, spec["stage2_steps"])
    record["stage2_nll_end"] = validation_nll(model, "question")
    record["norm_after_stage2"] = visual_token_norm(model)
    if purpose == "main":  # 評価集合は本番の学習でのみ評価する(較正では評価しない)
        record["correct"] = evaluate_generation(model)
        record["correct_swapped"] = evaluate_generation(model, SWAPPED_EVAL_IMAGE_INDEX)
        record["accuracy"] = accuracy_by_type(record["correct"])
        record["accuracy_swapped"] = accuracy_by_type(record["correct_swapped"])
    record["seconds"] = time.time() - t0
    record["projection_state"] = {k: v.detach().cpu().clone() for k, v in model.projection.state_dict().items()}
    del model
    empty_device_cache()
    return record


def describe(record: dict) -> str:
    text = f"{record['condition']} s={record['seed']} lr1={record['lr1']} lr2={record['lr2']:.3g}:"
    if "stage1_nll_after" in record:
        text += f" 第 1 段階の検証 NLL {record['stage1_nll_before']:.3f} -> {record['stage1_nll_after']:.3f}、"
    text += f" 第 2 段階の検証 NLL {record['stage2_nll_start']:.3f} -> {record['stage2_nll_end']:.4f}"
    if "accuracy" in record:
        a, b = record["accuracy"], record["accuracy_swapped"]
        text += f"、正解率 {a['all']:.4f}(色 {a['color']:.4f}・位置関係 {a['spatial']:.4f})、差し替え {b['all']:.4f}"
    return text + f"、{record['seconds']:.0f}s"


def judge(value: float, sigma: float) -> str:
    # 支持 / 反証 / 判定不能(前提不成立は 6.11 節で前提条件から決める)
    if value > SIGMA_MULTIPLIER * sigma:
        return "支持"
    if value < -SIGMA_MULTIPLIER * sigma:
        return "反証"
    return "判定不能"


_tag = "[動作確認のみ、結論ではない] " if SMOKE_TEST else ""
_plot_tag = "[smoke test, not a result] " if SMOKE_TEST else ""  # 図のタイトル用(英字のみ)
```

### 5.7 線形プローブ(実験 B の診断量)

凍結した画像 encoder の最終層の一つ前の層の特徴から、次の 2 つの量を多クラスのロジスティック回帰(線形プローブ、Alain & Bengio [8])で
当てる。言語モデル・projection 層とは独立で、学習の条件にもシードにもよらない。

| 標的 | クラス | 必要な情報 |
|---|---|---|
| 配置(`layout`) | 位置関係の種類(左右 / 上下)× 左(または上)の図形の形 = $2 \times 5 = 10$ | どの形がどちらの側にあるか(位置と形の結びつき) |
| 形の組(`shape_pair`) | 2 つの図形の形の組(順序なし)= $\binom{5}{2} = 10$ | 位置によらない内容 |

入力は、パッチトークンの格子全体を平坦化したもの($64 \times 128 = 8{,}192$ 次元)と平均プール(128 次元)の 2 通り。学習用の画像の特徴を
学習用の平均と標準偏差で標準化し、重みとバイアスを 0 で初期化して、全バッチの AdamW(学習率 $10^{-2}$、重み減衰 $10^{-4}$)で
500 ステップ学習する。評価用の画像で正解率を測る。2 つの入力の次元が違うので、プローブの容量も違う(格子全体のほうが大きい)。
「平均プールでは配置が当たらないが形の組は当たる」なら、平均プールで失われるのが位置の情報であることを示唆する。


```python
PROBE_TARGETS = ("layout", "shape_pair")
PROBE_INPUTS = ("patch", "mean")
_shape_pairs = list(itertools.combinations(range(len(SHAPES)), 2))


def probe_labels(split: str, target: str) -> torch.Tensor:
    if target == "layout":  # 位置関係の種類 x 左(上)の図形の形
        labels = [m.relation * len(SHAPES) + m.first[1] for m in MEANINGS[split]]
    else:  # 形の組(順序なし)
        labels = [_shape_pairs.index(tuple(sorted((m.first[1], m.second[1])))) for m in MEANINGS[split]]
    return torch.tensor(labels, device=device)


def probe_inputs(split: str, inputs: str) -> torch.Tensor:
    return FEATURES[split].flatten(1) if inputs == "patch" else FEATURES[split].mean(dim=1)


def run_probe(target: str, inputs: str, steps: int = PROBE_STEPS) -> dict:
    x_train, x_eval = probe_inputs("train", inputs), probe_inputs("eval", inputs)
    mean, std = x_train.mean(dim=0), x_train.std(dim=0).clamp_min(1e-6)
    x_train, x_eval = (x_train - mean) / std, (x_eval - mean) / std
    y_train, y_eval = probe_labels("train", target), probe_labels("eval", target)
    num_classes = int(y_train.max()) + 1
    weight = torch.zeros(x_train.size(1), num_classes, device=device, requires_grad=True)
    bias = torch.zeros(num_classes, device=device, requires_grad=True)
    optimizer = AdamW([weight, bias], lr=PROBE_LEARNING_RATE, weight_decay=PROBE_WEIGHT_DECAY, foreach=True)
    for _ in range(steps):
        loss = functional.cross_entropy(x_train @ weight + bias, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_accuracy = float(((x_train @ weight + bias).argmax(dim=1) == y_train).float().mean())
        eval_accuracy = float(((x_eval @ weight + bias).argmax(dim=1) == y_eval).float().mean())
        majority = float(torch.bincount(y_eval).max()) / len(y_eval)
    return {"train": train_accuracy, "eval": eval_accuracy, "classes": num_classes, "majority": majority, "final_loss": float(loss.detach())}


# ラベルの確認: 配置と形の組がそれぞれ 10 クラスで、評価用の全クラスが学習用に現れる
for _target in PROBE_TARGETS:
    _ytr, _yev = probe_labels("train", _target), probe_labels("eval", _target)
    assert int(_ytr.max()) + 1 == 10 and set(_yev.tolist()) <= set(_ytr.tolist())
print(f"線形プローブ: 標的 {PROBE_TARGETS}(各 10 クラス)、入力 {PROBE_INPUTS}(次元 {NUM_PATCHES * VISION_DIM:,} / {VISION_DIM})")
```

    線形プローブ: 標的 ('layout', 'shape_pair')(各 10 クラス)、入力 ('patch', 'mean')(次元 8,192 / 128)


## 6. 実験 / Experiments

### 6.1 実験宣言セル: 共通の設定・検証すること・判定基準・前提条件

**この節の内容は本番実行の前に確定させ、結果を見た後に変更しない。** パイロット(6.2 節)を受けて決めた設定は、本番実行の前に
決めたものであり、6.2 節に経緯を記録した。

#### 共通の設定

- **画像 encoder**: 020 の CLIP の画像 encoder(NegCLIP・シード 0、`kojikojiprg/ai-theories-clip-synthetic-scenes`の`main`)。凍結する。
  最終層の一つ前の層(3 ブロックを通した後、最終正規化の前)のパッチトークン $N = 64$ 個(各 $d_v = 128$ 次元)を使う(3.4 節)。
- **言語モデル**: 008 の小型 GPT(`kojikojiprg/ai-theories-small-gpt-en`の`main`、4 層、$d = 256$、文脈長 256)とトークナイザ。
- **projection 層**: 線形写像 $128 \to 256$(`nn.Linear`の既定の初期化)。
- **第 1 段階**: projection 層のみを学習する。データは学習用の画像とキャプション(5.4 節)。ステップ数 $T_1 = 722$(キャプションのデータの
  2 エポック)。学習率は $6.4 \times 10^{-2}$ に **固定** し、較正しない(6.2 節のパイロットの経緯による。下の「学習率の較正」)。
- **第 2 段階**: projection 層と、言語モデルの Query・Value 射影への LoRA($r = 8$、$\alpha = 8$)を学習する。データは学習用の画像の
  質問と答え(色・位置関係)。ステップ数 $T_2 = 1{,}444$(質問のデータの 1 エポック)。学習率は条件ごとに較正する。
- **学習の部品**: AdamW(重み減衰 0、012・016 と同じ)、バッチ 32 事例、最初の 10% のステップで線形 warmup の後 cosine で最大値の
  0.01 倍まで減衰、gradient clipping の閾値 1.0(学習するパラメータのグローバルノルム)、FP32。損失は答えのトークンだけの負の
  対数尤度の平均(3.5 節)。
- **シード**: シード $s$ が決めるのは、projection 層の初期化(`torch.manual_seed(21100 + s)`)、LoRA の $A$ の初期化
  (`torch.manual_seed(21300 + s)`)、第 1 段階のミニバッチの順序(`make_step_batches(..., 21200 + s)`)、第 2 段階のミニバッチの順序
  (`make_step_batches(..., 21400 + s)`)である。同じシードの条件どうしはこれらを共有する(対応のある比較。6.10 節で確かめる)。
  シードは $0, 1, \dots$ を使い、数は削る段階で決まる。
- **評価集合**: 学習用・検証用と別の乱数シードで描いた評価用の画像 1,444 枚(学習用のキャプション 1,444 個 × 各 1 枚)の質問 5,776 個
  (色 2,888 個・位置関係 2,888 個)。テンプレートと答えの語彙は学習用と同じで、画像だけが学習で見ていないものである(5.4 節)。
- **正解率**: 指示部分(視覚トークンを含む)から貪欲法で生成し(上限 12 トークン)、終端記号`### End`の前の単語の列が正解と
  完全一致した割合(016 の`score_generated_response()`の完全一致)。2 択の評価は使わない。学習の最終ステップの重みで評価する。
  $\mathrm{acc}_{c,s}$ は条件 $c$・シード $s$ の評価集合全体の正解率、$\mathrm{acc}^{\mathrm{color}}_{c,s}$・$\mathrm{acc}^{\mathrm{spatial}}_{c,s}$ は
  色・位置関係の質問だけの正解率である。
- **検証集合**: 評価集合と別の乱数シードで描いた画像 1,444 枚の質問(第 2 段階)とキャプション(第 1 段階)。学習率の較正と前提条件 P1 に
  のみ使う。**評価集合は較正に使わない。**

#### 条件の対応表

| 記号 | 視覚トークン | 第 1 段階 | 第 2 段階のステップ数 | 使う実験 |
|---|---|---|---|---|
| P | パッチトークン全体($N_v = 64$) | あり($T_1$) | $T_2$ | 実験 A の条件 1・実験 B の条件 1 |
| R | パッチトークン全体($N_v = 64$) | なし(projection 層は乱数の初期値) | $T_2$ | 実験 A の条件 2 |
| L | パッチトークン全体($N_v = 64$) | なし(projection 層は乱数の初期値) | $T_1 + T_2$ | 実験 A の診断量(判定なし) |
| M | 平均プール($N_v = 1$) | あり($T_1$) | $T_2$ | 実験 B の条件 2 |

数式の添字の $c \in \{P, R, L, M\}$ はこの表の記号である。

#### 学習率の較正(本番の冒頭、6.5 節)

- **第 1 段階の学習率は較正しない**($6.4 \times 10^{-2}$ に固定、全条件・全シードで共通)。6.2 節のパイロットで、第 1 段階の検証集合の
  キャプションの負の対数尤度は、学習率を $8 \times 10^{-3}$ から $2.56 \times 10^{-1}$ まで 32 倍変えても条件内の相対差が 5% 以内
  (平均プールでは 0.3% 以内)で、格子の上端に向かって単調にわずかに下がり続けた。この指標では内点の最良値が定まらず、較正しても
  前提条件 P0 が学習率の選択と無関係な理由で不成立になる。原論文も第 1 段階の学習率は固定の値($2 \times 10^{-3}$)である。
  規則で決めた値なので、第 1 段階の学習率は較正に関する前提条件(P0)の対象外である。P と M に同じ値を使うので、この値が最適から
  ずれたときの対比量への偏りは、実験 A では P のみが第 1 段階を持つので **負の向き**(第 1 段階の効果を過小に見る側)、実験 B では
  P・M の両方に効き、向きは事前には定まらない。
- **第 2 段階の格子**: 条件 $c$ の中心 $\eta_{2,c}$(P・R・M とも $8 \times 10^{-3}$、6.2 節のパイロットで決めた)の $\{1/2, 1, 2\}$ 倍
  (公比 2 の 3 点)。
- **指標**: 本番と同じステップ数・全学習データで、**較正専用のシード**(シード番号 90。実験のシードと共有しない)の 1 シードで
  各格子点を学習し、**検証集合の答えのトークンあたりの負の対数尤度**(教師強制)が最小の学習率を選ぶ(同点なら小さい方)。
  生成の正解率は較正に使わない。
- **拡張の規則**: 最良の学習率が格子の端に来た場合は、その方向に公比 2 で 1 点だけ格子を拡張して学習し、改めて最良を選ぶ(1 回のみ)。
- **較正の起点**: P・M の第 2 段階の較正は、較正のシードで固定の学習率の第 1 段階を学習した projection 層から始める(視覚トークンの
  取り方ごとに 1 回)。R は乱数の初期値の projection 層から始める。
- **較正の方式**`CALIBRATION_MODE`(選ばれた実行計画から決まる、下の「実行計画」):
  - `"all"`: 第 2 段階の P・R・M の 3 つをそれぞれ較正する。L は R の値を使う(規則で決める。L は診断量で判定しないので、較正に
    関する前提条件の対象にしない)。
  - `"representative"`: 第 2 段階の P だけを較正し、R・L・M には P の値を使う(規則で決める)。

#### 共通の前提条件

- **P0(較正)**: 較正した各条件で、選んだ学習率が格子の **内点** であること(拡張後も最良が端なら不成立)。各実験が対象とするのは、
  その実験の条件の第 2 段階の学習率を決めた較正である(実験 A: P・R、実験 B: P・M)。第 1 段階の学習率は固定なので対象外。
  `"representative"`で規則で決めた条件の学習率は較正していないので、その条件の P0 は **対象外** になる(下の各実験の「規則で決めた
  学習率による偏り」を参照)。
- **P1(学習の成立)**: その実験の全条件・全シードの学習で、(a)損失がすべてのステップで有限であり、(b)第 2 段階の後の検証集合の
  答えの負の対数尤度(トークンあたり)が、第 2 段階の開始時(第 1 段階の後、または乱数の初期値の projection 層)の値の 0.5 倍以下で
  あること。第 1 段階を行う条件では、加えて(c)第 1 段階の損失がすべてのステップで有限であり、第 1 段階の後の検証集合の
  キャプションの負の対数尤度が、第 1 段階の開始時の値より **小さい** こと。検証したい仮説(条件間の正解率の差)とは独立な、
  「学習が進んだか」の量である。(b)の 0.5 倍は、学習が初期値から明らかに進んだことを表す下限として置いた(パイロットの値から
  決めたものではない)。答えの負の対数尤度は、画像の情報がなくても、形式のトークン(` ### End`など)と答えの語彙の学習だけで大きく
  下がるので、(b)は視覚トークンが運ぶ情報の量によらず満たしうる。(c)を下がり幅ではなく「下がったか」で定義する理由は 6.2 節の
  「スモークテストの後の改訂」に記した。
- **P2(視覚への依存)**: その実験の各条件で、評価集合の各事例の画像を **別の画像に差し替えた** ときの正解率のシード平均が
  **0.5 以下** であること(上限 1 から 0.5 以上離れていること)。差し替えは、評価用の画像の固定の置換(乱数シード 21004、どの画像も
  自分と異なる意味の画像に移る)で行い、全条件・全シードで同じ置換を使う。画像を見ずに質問の文字列だけから答える戦略の正解率の
  上界は、評価集合で約 0.2 である(5.4 節で厳密に計算する)。差し替えた画像での正解率がこの上界を大きく超えて 0.5 を上回るなら、
  モデルは画像以外の手がかり(質問の文字列と答えの偏り)で答えていることになり、正解率の差を視覚の情報の差として解釈できない。
- **P3(改善の余地)**: 基準となる条件の値が上限 1 から十分に離れていること。**差ではなく基準の条件の値で定義する。**
  - 実験 A: R の評価集合の正解率のシード平均 $\overline{\mathrm{acc}}_{R} \le 0.95$。
  - 実験 B: M の位置関係の質問の正解率のシード平均 $\overline{\mathrm{acc}}^{\mathrm{spatial}}_{M} \le 0.95$。
  - **閾値の根拠**: 評価集合の 1 つの質問の種類は 2,888 事例であり、正解率 0.95 の二項分布の標準偏差は
    $\sqrt{0.95 \cdot 0.05 / 2888} \approx 0.004$ である。2 条件の差の標準偏差はその約 $\sqrt{2}$ 倍、判定の閾値はその 2 倍の約 0.012 に
    シード間のばらつきが加わる。上限までの余地 0.05 は、この閾値の 4 倍程度であり、仮説どおりの改善があれば判定の閾値を超えうる
    大きさである。余地がこれより小さいと、仮説が正しくても差が判定の閾値に埋もれうる(020 の実験 B・C の天井効果)。

前提条件が 1 つでも成立しない実験は、判定関数の結果によらず **前提不成立** と記録する(判定不能とは区別する)。

#### 判定の形と標準偏差の単位

各実験の対比量 $\Delta$ とその標準偏差 $\sigma$ に対し、$\Delta > 2\sigma$ なら **支持**、$\Delta < -2\sigma$ なら **反証**、それ以外は
**判定不能** とする。$\sigma$ は、**シード間の分散** と、評価の単位を復元抽出する **対応付きのブートストラップの分散** の和の平方根とする
(013・018 と同じ形)。

$$
\sigma^2 = \frac{s_d^2}{S} + \mathrm{Var}_{\mathrm{boot}}(\Delta^{*})
$$

- $d_s$ はシード $s$ の対比量、$s_d^2$ はその標本分散(自由度 $S - 1$)、$S$ はシード数。第 1 項は、学習の乱数(初期化・ミニバッチの
  順序)によるばらつきの、シード平均の分散の推定である。
- 第 2 項は、評価集合の抽出によるばらつきである。評価の単位は **画像**(1 枚の画像の 4 つの質問は同じ画像を共有して相関するので、
  画像をクラスタとして復元抽出する。015 の`paired_cluster_bootstrap_ratio_of_sums()`)。全条件・全シードに **同じ再標本** を使い
  (対応付き)、再標本ごとにシード平均の対比量 $\Delta^{*}$ を計算して、その分散をとる。反復は本番 10,000 回(スモークテスト 1,000 回)。
- 2 つの項は、それぞれ別の乱数の源(学習 / 評価集合)によるばらつきであり、独立とみなして分散を足す。シード間の分散の推定にも
  評価集合の抽出のばらつきが一部含まれるので、和はやや保守的(大きめ)な推定になる。

#### 実行計画(削る段階と較正の方式の組)と、その自動選択

**削る段階**:

| 段階 | 内容 | 実験 A のシード数 $S_A$(R) | 実験 B のシード数 $S_B$(M) | P のシード数 | L(診断量) |
|---|---|---|---|---|---|
| 0 | 全実験 5 シード | 5 | 5 | 5 | 5 シード |
| 1 | L を除く | 5 | 5 | 5 | なし |
| 2 | 段階 1 に加え、実験 A のシード数を 3 にする | 3 | 5 | 5 | なし |
| 3 | 段階 2 に加え、実験 B のシード数を 3 にする | 3 | 3 | 3 | なし |

P は実験 A・B で共有するので、そのシード数は $\max(S_A, S_B)$ である。各実験の判定には、その実験のシード数の分の P を使う
(シードは先頭から)。

**実行計画**: 次の 8 通りに、優先順位の高い順に番号を付ける。計画 0〜3 は較正の方式`"all"`で段階 0〜3、計画 4〜7 は **下位の計画** で、
較正の方式`"representative"`で段階 0〜3 である。

| 計画 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 段階 | 0 | 1 | 2 | 3 | 0 | 1 | 2 | 3 |
| 較正の方式 | all | all | all | all | representative | representative | representative | representative |

- **選択の規則**: 本番の学習を始める前に(6.4 節)、スケーリングの計測(6.3 節)から、各計画の残りの実行時間(較正・本番の学習と評価・
  診断量・判定)を見積もり、それまでの経過時間を足して予算(T4 で 120 分)以内に収まる計画のうち **番号の最も小さい** ものを選ぶ。
  計画 7 でも超える場合は、学習の前に例外で停止する。**選択は見積もりのみに基づき、どの実験の結果も参照しない。**
- **見積もりの方法**: 学習は、定常状態の 1 ステップの時間(ウォームアップの後に 3 点のステップ数で計測し、ステップ数の差から求めた
  傾き)にステップ数を掛ける。べき乗則のあてはめ($\log t = \log a + b \log n$)も行い、べき指数 $b$ と、比例の値との差を印字する。
  評価(生成・教師強制)は、事例数を 3 点で変えて計測し、べき乗則で外挿する。
- **段階の順序の理由**: まず判定に使わない診断量(L)を除く。次に、対比量が差であり誤差の小さい実験 A のシード数を削る。実験 B は
  差の差を対比量とし、誤差伝播により標準偏差が大きくなりやすいので、シード数を最後に削る。
- **較正の方式を最後に変える理由**: 較正の範囲を削ると、規則で決めた学習率が最適からずれたときに対比量に偏りが生じうる
  (各実験の「規則で決めた学習率による偏り」)。シード数を削ることは精度を下げるだけで偏りを生じないので、先に行う。
- **学習量を下げる段階は設けない**: $T_2$ は、6.2 節のパイロットで、基準の条件の正解率が $T_2$ で上限に届かないこと(真偽のみ)を
  確かめた値である。ステップ数を下げると学習の飽和の程度が変わり、パイロットで確かめた状態から離れるので、削る対象にしない。

#### 実験 A: 第 1 段階(特徴の整列)の効果

**検証すること**: 第 2 段階の前に第 1 段階(projection 層のみの学習)を行うと、第 2 段階のステップ数を揃えたとき、評価集合の
正解率が上がる(Liu et al. [1] のアブレーションで、第 1 段階を省くと正解率が下がったことに対応する)。

**条件**: P(第 1 段階 → 第 2 段階)と R(第 2 段階のみ。projection 層は乱数の初期値)。第 2 段階のステップ数・データ・ミニバッチの順序・
LoRA と projection 層の初期値は同じシードで揃う。

**対比量**: $d_s = \mathrm{acc}_{P,s} - \mathrm{acc}_{R,s}$、$\Delta_A = \frac{1}{S_A} \sum_{s} d_s$($s = 0, \dots, S_A - 1$)。
**仮説の向き**: $\Delta_A > 0$。$\sigma_A$ は上の「判定の形と標準偏差の単位」のとおり(評価の単位は画像、評価集合の全 5,776 事例)。

**判定**: $\Delta_A > 2\sigma_A$ なら支持、$\Delta_A < -2\sigma_A$ なら反証、それ以外は判定不能。

**前提条件**: P0(第 2 段階の P・R)、P1(P・R の全シード)、P2(P・R)、P3($\overline{\mathrm{acc}}_{R} \le 0.95$)。

**揃えた量と揃えなかった量**:

- **揃えた量**: 第 2 段階のステップ数 $T_2$・第 2 段階で見る事例(同じシードでは同じ順序の同じ事例)・第 2 段階で学習するパラメータ
  (projection 層 + LoRA)・projection 層と LoRA の初期値(同じシード)。主の比較は「第 2 段階の学習量を揃えたときに、第 1 段階を
  前に置くことの効果」である(原論文のアブレーションと同じ形)。
- **揃えなかった量**: 総ステップ数(P は $T_1 + T_2$、R は $T_2$)と総計算量、P が第 1 段階で見るキャプションのデータ。したがって、
  P の優位は「第 1 段階の目的(凍結した言語モデルに合わせた整列)の効果」と「単に projection 層を長く学習した効果」の和でありうる
  (交絡)。
- **交絡を分離する水準(診断量、判定なし)**: L は、R と同じく第 1 段階なしで、第 2 段階を総ステップ数 $T_1 + T_2$ だけ学習する
  (第 1 段階のステップを第 2 段階に回す)。P と L の比較は総ステップ数を揃えた比較になる。ただし L は、P が第 1 段階に使う
  ステップでも LoRA を学習し、キャプションの代わりに質問のデータを見るので、「学習量」以外の違いが残る。L は削る段階 1 以降では
  学習しない。
- **学習の速さの違い**: R は乱数の初期値の projection 層から、言語モデルの埋め込みの尺度(5.4 節の印字)に合わない視覚トークンで
  第 2 段階を始めるので、P より収束が遅いと予想される。$T_2$ で学習が飽和していなければ、差は「到達点の差」ではなく「速さの差」を
  含む。途中の評価(第 2 段階の 25%・50%・75% の時点の検証集合の負の対数尤度)を診断量として記録する。

**対比量と介入の作用点の距離**: 介入(第 1 段階)が直接作用する量は、第 2 段階の開始時の projection 層、すなわち視覚トークンの配置で
あり、その直接の帰結は第 2 段階の開始時の検証集合の答えの負の対数尤度である。対比量の正解率は、その後の第 2 段階の学習を経た
下流の量であり、第 2 段階の学習が開始時の差を縮めると、作用点での差があっても対比量の差は小さくなる。原論文の主張は第 2 段階の
後の性能についてのものなので、これを対比量とし、作用点に近い量(第 2 段階の開始時と途中の検証集合の負の対数尤度、第 1 段階の後の
視覚トークンのノルム)を診断量として併記する。

**規則で決めた学習率による偏り**(`"representative"`の計画が選ばれた場合): R の第 2 段階の学習率は P の値になる。乱数の初期値の
projection 層から始める R の最適な学習率が P と異なり、その値が R にとって最適からずれていれば、R の正解率は較正した場合より低く
なり、$\Delta_A$ は **正の向き(支持の側)** に偏る。このとき R の P0 は対象外であり、支持の判定はこの偏りの可能性を含む。

#### 実験 B: 視覚トークンの取り方 × 質問の種類

**検証すること**: 視覚トークンを、パッチトークンの格子全体(P、$N_v = 64$)から、同じ層のパッチトークンの平均プール(M、$N_v = 1$)に
置き換えたときの正解率の低下は、色の質問より位置関係の質問で大きい(3.6 節の非対称)。

**条件**: P と M。どちらも第 1 段階 → 第 2 段階で、ステップ数・データ・ミニバッチの順序(同じ質問の並び)・projection 層と LoRA の
初期値(同じ形状なので同じ値)は同じシードで揃う。違いは視覚トークンの取り方だけである。

**対比量**: 色の質問の正解率の差を $c_s = \mathrm{acc}^{\mathrm{color}}_{P,s} - \mathrm{acc}^{\mathrm{color}}_{M,s}$、位置関係の質問の差を
$p_s = \mathrm{acc}^{\mathrm{spatial}}_{P,s} - \mathrm{acc}^{\mathrm{spatial}}_{M,s}$ として、交互作用 $d_s = p_s - c_s$、
$\Delta_B = \frac{1}{S_B} \sum_{s} d_s$($s = 0, \dots, S_B - 1$)。**仮説の向き**: $\Delta_B > 0$。

**標準偏差の導出**: $\Delta_B$ は 4 つの正解率(2 条件 × 2 種類)の差の差である。4 つの正解率がそれぞれ独立に分散 $v$ を持つなら、
$\mathrm{Var}(d_s) = 4v$ となり、1 つの差(分散 $2v$)の 2 倍になる。実際には同じ画像の色と位置関係の質問、同じシードの P と M は相関する
ので、相関を独立と仮定した計算はしない。シード間の分散 $s_d^2$ は $d_s$ そのものから、ブートストラップの分散は再標本ごとに 4 つの
正解率から $\Delta_B^{*}$ を作って求めるので、相関はどちらにも自動的に反映される。

**判定**: $\Delta_B > 2\sigma_B$ なら支持、$\Delta_B < -2\sigma_B$ なら反証、それ以外は判定不能。

**前提条件**: P0(第 2 段階の P・M)、P1(P・M の全シード)、P2(P・M)、
P3($\overline{\mathrm{acc}}^{\mathrm{spatial}}_{M} \le 0.95$)。

**揃えた量と揃えなかった量**:

- **揃えた量**: 第 1 段階・第 2 段階のステップ数、見る事例とその順序、学習するパラメータの数(projection 層は $N_v$ によらず
  同じ形状)、画像 encoder の層。
- **揃えなかった量**: 系列長(P は M より 63 トークン長い)と、それに比例する 1 ステップの計算量。学習率は条件ごとに較正する
  (`"all"`)。

**対比量と介入の作用点の距離**: 介入(平均プール)が直接作用する量は、言語モデルに渡る視覚トークンの持つ情報、すなわち視覚トークンから
図形の位置が読み出せるかどうかである。対比量の正解率は、その情報を言語モデルが学習を通じて使えたかどうかを経た下流の量である。
作用点に近い量として、**線形プローブ**(6.9 節)で、パッチトークンの格子と平均プールのそれぞれから図形の配置を当てる正解率を
診断量として測る。

**規則で決めた学習率による偏り**(`"representative"`の計画が選ばれた場合): M の第 2 段階の学習率は P の値になる(第 1 段階はもとから共通の固定値)。その値が
M にとって最適からずれていれば、M の正解率は色・位置関係のどちらの質問でも較正した場合より低くなり、$c_s$ と $p_s$ はどちらも正の
向きに偏る。交互作用 $\Delta_B = p - c$ が偏る向きは、学習率のずれの害が位置関係と色のどちらで大きいかによって決まり、**事前には
定まらない**(どちらの向きにも偏りうる)。このとき M の P0 は対象外である。

#### 診断量(判定なし)

- **L の正解率**(実験 A、段階 0 のみ): P・R との比較。
- **途中の検証集合の負の対数尤度**(第 2 段階の 25%・50%・75%・100% の時点、全条件)と、第 2 段階の開始時の値。
- **視覚トークンのノルム**: 第 1 段階の前後と第 2 段階の後の、視覚トークンの平均ノルム(言語モデルのトークン埋め込みのノルムとの比)。
- **線形プローブ**(実験 B、6.9 節): 画像 encoder の最終層の一つ前の層の特徴から、(1)図形の配置(位置関係の種類 × 左または上の
  図形の形、10 クラス)と、(2)対照として、位置によらない内容(2 つの図形の形の組、10 クラス)を当てる。入力は、パッチトークンの格子
  全体(64 × 128 を平坦化した 8,192 次元)と平均プール(128 次元)。学習用の画像の特徴で多クラスのロジスティック回帰を学習し、
  評価用の画像で正解率を測る。凍結した特徴だけを使い、言語モデルとは独立である。
- **質問の種類ごとの正解率**(全条件)と、テンプレートごとの正解率。

### 6.2 パイロットの記録(本番実行の前)

本番実行の前に、ローカル(Apple M シリーズの MPS)で次のパイロットを行い、5.2 節の設定を決めた。パイロットは **パイロット専用の
シード 95** で行い、第 2 段階の天井の確認には **パイロット専用の評価用の画像**(1,444 枚、描画のシード 21901)を使った。本番の
評価集合(シード 21003)には触れていない。**条件間の値を比べる量(対比量の向きの情報)は印字していない。** 印字したのは、各条件の中で
最良の学習率の位置と条件内の相対値、および基準の条件の正解率が 0.95 に届くかどうかの真偽だけである。

**1. 第 2 段階の学習率の格子の中心**($T_2 = 1{,}444$、格子 $5 \times 10^{-4}$〜$1.6 \times 10^{-2}$ の公比 2 の 6 点、検証集合の答えの負の
対数尤度): R・P・M のいずれも、最良の位置は $8 \times 10^{-3}$(6 点の 5 番目、内点)で、$1.6 \times 10^{-2}$ では悪化した。そこで、3 条件とも
格子の中心を $8 \times 10^{-3}$ とした(格子 $\{4, 8, 16\} \times 10^{-3}$)。

**2. 第 1 段階の学習率**(検証集合のキャプションの負の対数尤度):

- $T_1 = 361$(1 エポック)、格子 $5 \times 10^{-4}$〜$1.6 \times 10^{-2}$: パッチトークン全体・平均プールとも、最良の位置は格子の上端だった
  (条件内の相対値は、パッチトークン全体で下端の 1.33 倍から上端の 1.00 倍まで、平均プールで 1.04 倍から 1.00 倍まで単調に下がった)。
- 格子を $8 \times 10^{-3}$〜$1.28 \times 10^{-1}$ に広げても、最良の位置は上端だった。
- $T_1 \in \{722, 1{,}444\}$、格子 $8 \times 10^{-3}$〜$2.56 \times 10^{-1}$: いずれも最良の位置は上端で、条件内の相対値の幅はパッチトークン
  全体で 5% 以内(722 ステップ: 1.050〜1.000、1,444 ステップ: 1.040〜1.000)、平均プールで 0.3% 以内だった。

この指標は、第 1 段階の学習率に対してほとんど平坦で、上端に向かってわずかに下がり続けるので、内点の最良値で学習率を決める較正は
意味を持たない(較正しても前提条件 P0 が学習率の選択と無関係な理由で不成立になる)。そこで、**第 1 段階の学習率は較正せず、
$6.4 \times 10^{-2}$ に固定** した(パイロットの範囲の幾何的な中ほどの値で、パッチトークン全体の条件内の相対値は 722 ステップで約 1.014・
1,444 ステップで約 1.005、Adam の学習率として極端に大きな値を避けた)。$T_1$ は、原論文の第 1 段階の学習量(1 エポックで数千ステップ)に
比べて 1 エポック(361 ステップ)は少ないので、実行時間の予算の範囲で **2 エポック($T_1 = 722$)** とした。

パッチ特徴の平均ノルム(約 31)は言語モデルのトークン埋め込みの平均ノルム(約 0.85)の約 36 倍であり(5.4 節の印字)、`nn.Linear`の
既定の初期化の projection 層の出力のノルムも約 25 である。第 1 段階は、この尺度の違いを埋めることも含めて視覚トークンを言語モデルの
埋め込み空間に配置する。大きな学習率が好まれたのは、この尺度の違いによる可能性がある(事後的な推測であり、確かめていない)。

**3. 第 2 段階のステップ数と改善の余地**(基準の条件の正解率が 0.95 に届くかの真偽のみ):

| 設定 | R の全体の正解率 $\ge 0.95$ | M の位置関係の正解率 $\ge 0.95$ |
|---|---|---|
| $T_1 = 361$、$T_2 = 1{,}444$ | False | False |
| $T_1 = 361$、$T_2 = 722$ | False | False |
| $T_1 = 722$(第 1 段階の学習率 $6.4 \times 10^{-2}$)、$T_2 = 1{,}444$ | (R は $T_1$ によらない) | False |

候補 $T_2 \in \{1{,}444, 722\}$(第 2 段階のデータの 1 エポックと 1/2 エポック)のうち、どちらでも基準の条件が上限に届かなかったので、
学習量の大きい $T_2 = 1{,}444$(1 エポック)とした。この確認は真偽のみであり、正解率の値と P・R・M の間の差は見ていない。

**4. 1 ステップの時間**(ローカルの MPS、バッチ 32、系列長を全事例の最大長にそろえる): 第 2 段階・パッチトークン全体で約 80 ms で、
バッチサイズ 16・32・64 で時間はほぼ比例した(固定費は支配的ではない)。本番の T4 での時間は 6.3 節で本番の冒頭に計測し、ここでの値は
計画の選択に使わない。

**5. スモークテストの後の改訂(本番実行の前、前提条件 P1(c))**

- **旧基準**: 第 1 段階の後の検証集合のキャプションの負の対数尤度が、第 1 段階の開始時の値の **0.5 倍以下** であること。
- **新基準**: 第 1 段階の損失がすべてのステップで有限であり、第 1 段階の後の検証集合のキャプションの負の対数尤度が、開始時の値より
  **小さい** こと。
- **改訂の理由**(観測結果の向きによらない一般論): 前提条件は、検証したい仮説と独立な量で定義しなければならない。第 1 段階の
  キャプションの負の対数尤度がどこまで下がりうるかは、視覚トークンが運ぶ画像の情報の量で上限が決まる(言語モデルは凍結されており、
  キャプションの 2 つの色・2 つの形・位置関係は画像からしか分からない)。視覚トークンの情報の量は、実験 B の介入(平均プール)そのもので
  あるため、下がり幅の閾値を前提条件にすると、実験 B の仮説が正しいほど平均プールの条件が前提不成立になりやすい。第 2 段階の(b)は、
  答えの形式のトークンと答えの語彙の学習だけで満たしうるので、改訂しない。判定基準(対比量・閾値の導出式・仮説の向き)は変更していない。
- **旧基準によるスモークテストの結果**(ローカルの MPS、計画 0、3 シード、本番と同じステップ数): P の第 1 段階の比は約 0.41
  (6.34 → 2.57 など、3 シードとも 0.5 以下で成立)、M の比は約 0.75(6.15 → 4.59、3 シードとも不成立)で、実験 B は旧基準では前提不成立と
  なった。実験 A は旧基準でも前提条件がすべて成立した。改訂後の基準でのスモークテストの出力は、このノートブックのセル出力である。

**6. 本番実行との関係**(本番実行の後に追記)

本番実行(Google Colab T4)では計画 0(段階 0・較正の方式`"all"`)が選ばれ、ステップ数は上の設定($T_1 = 722$・$T_2 = 1{,}444$)のまま、
削る段階も規則で決める学習率(`"representative"`)も使われなかった。パイロットで決めた設定との関係は次のとおりである。

- **第 2 段階の学習率の格子の中心**: 本番の較正(較正専用のシード 90)でも、P・R・M の 3 条件とも格子の中心 $8 \times 10^{-3}$ が拡張なしで
  内点として選ばれた(6.5 節)。パイロット(シード 95、MPS)で決めた中心と一致した。
- **改善の余地**: 本番の基準の条件の値は、R の全体の正解率の平均 0.7382、M の位置関係の正解率の平均 0.5205 で、どちらも 0.95 を下回った
  (前提条件 P3 は成立)。パイロットの真偽の確認と同じ結果である。
- **1 ステップの時間**: T4 の第 2 段階・パッチトークン全体の 1 ステップは 36.8 ms で、ローカルの MPS(約 80〜95 ms)より短かった。計画の選択は
  T4 での計測のみに基づいており、パイロットの値は使っていない。
- **パイロットで見ていなかった量**: パイロットでは、基準の条件が上限に届くかの真偽だけを確かめ、対比量のシード間のばらつきは測って
  いなかった。本番では、このばらつきが判定の閾値の大部分を占めた(7.6 節)。

### 6.3 スケーリングの計測と外挿(1 セッションの見積もり)

本番と同じデバイスで、学習・評価の各処理の時間を 3 点で計測し、各計画の残りの実行時間を見積もる。データの準備(描画・符号化・
パッチ特徴)は、スモークテストでも本番と同じ規模で 5.4 節ですでに実行しているので、その実測の経過時間に含まれる(外挿しない)。

- **学習**(4 種類: 第 1 段階・第 2 段階 × パッチトークン全体・平均プール): 計測専用のシード(91)のモデルで、ウォームアップ
  (8 ステップ)の後に 16・32・64 ステップ(公比 2)を計測する。見積もりは **定常状態の 1 ステップの時間**(64 ステップと 16 ステップの
  時間の差 / 48)に本番のステップ数を掛けたものを主とし、べき乗則のあてはめのべき指数 $b$ と、べき乗則の外挿値と比例の値の差を
  印字する。参考として、第 2 段階・パッチトークン全体の 1 ステップの時間をバッチサイズ 16 でも計測し、1 ステップの時間がバッチサイズに
  比例する(計算量の効く)処理か、固定費が支配的な処理かを印字する。
- **評価**(生成による採点、教師強制の負の対数尤度): 検証集合の事例を 256・512・1,024 個(公比 2)使って計測し、べき乗則で評価集合・
  検証集合の事例数に外挿する(外挿値と比例の値の大きい方を使う)。計測には評価集合を使わない。
- **線形プローブ**: 本番と同じ特徴で 25・50・100 ステップを計測し、ステップ数に比例させて見積もる。


```python
_t0_scaling = time.time()


def extrapolate(label: str, sizes, times, targets) -> dict:
    # べき乗則の外挿値と、最大の計測点の実測値を比例で伸ばした値の大きい方を、外挿先ごとに返す(評価用)
    fit = fit_power_law_exponent(sizes, times)
    result, parts = {}, []
    for target in targets:
        extrapolated = fit.coefficient * target**fit.exponent
        proportional = times[-1] * target / sizes[-1]
        result[target] = max(extrapolated, proportional)
        parts.append(f"n={target:,}: 外挿 {extrapolated:.1f}s・比例 {proportional:.1f}s")
    detail = ", ".join(f"n={n:,}: {t:.2f}s" for n, t in zip(sizes, times, strict=True))
    print(f"[{label}] {detail} -> b={fit.exponent:.3f}(標準誤差 {fit.exponent_stderr:.3f}); " + "、".join(parts))
    return result


# (1) 学習: 定常状態の 1 ステップの時間
TRAINING_KINDS = [("stage1", "patch"), ("stage1", "mean"), ("stage2", "patch"), ("stage2", "mean")]
STEP_SECONDS: dict[tuple, float] = {}
STEP_FIT: dict[tuple, dict] = {}
_step_targets = {"stage1": (STAGE1_STEPS,), "stage2": (STAGE2_STEPS, STAGE1_STEPS + STAGE2_STEPS)}


def build_timing_model(stage: str, pooling: str) -> LlavaStyleModel:
    model = build_llava(pooling, TIMING_SEED_INDEX)
    if stage == "stage2":
        add_lora(model, TIMING_SEED_INDEX)
        model.to(device)
    return model


for _kind in TRAINING_KINDS:
    _stage, _pooling = _kind
    _model = build_timing_model(*_kind)
    _data = ENCODED[(_pooling, "caption" if _stage == "stage1" else "question", "train")]
    timed_call(lambda: train_steps(_model, _data, SCALING_WARMUP_STEPS, 1e-4, 0))  # ウォームアップ(計測しない)
    _times = [timed_call(lambda n=n: train_steps(_model, _data, n, 1e-4, 0)) for n in SCALING_STEP_COUNTS]
    _fit = fit_power_law_exponent(SCALING_STEP_COUNTS, _times)
    _per_step = (_times[-1] - _times[0]) / (SCALING_STEP_COUNTS[-1] - SCALING_STEP_COUNTS[0])
    STEP_SECONDS[_kind] = _per_step
    _parts = []
    for _target in _step_targets[_stage]:
        _proportional = _per_step * _target
        _power = _fit.coefficient * _target**_fit.exponent
        _parts.append(f"T={_target:,}: 比例 {_proportional:.1f}s・べき乗則 {_power:.1f}s(差 {_power - _proportional:+.1f}s)")
    STEP_FIT[_kind] = {"exponent": _fit.exponent, "times": _times}
    print(
        f"[学習 {_stage} {_pooling}] " + ", ".join(f"{n}: {t:.2f}s" for n, t in zip(SCALING_STEP_COUNTS, _times, strict=True))
        + f" -> 定常状態の 1 ステップ {_per_step * 1000:.1f} ms、b={_fit.exponent:.3f}; " + "、".join(_parts)
    )
    del _model
# 参考: バッチサイズ 16 の 1 ステップ(固定費が支配的かの判断材料)
_model = build_timing_model("stage2", "patch")
_data = ENCODED[("patch", "question", "train")]
timed_call(lambda: train_steps(_model, _data, SCALING_WARMUP_STEPS, 1e-4, 0, batch_size=16))
_t16 = [timed_call(lambda n=n: train_steps(_model, _data, n, 1e-4, 0, batch_size=16)) for n in (SCALING_STEP_COUNTS[0], SCALING_STEP_COUNTS[-1])]
STEP_SECONDS_BATCH16 = (_t16[1] - _t16[0]) / (SCALING_STEP_COUNTS[-1] - SCALING_STEP_COUNTS[0])
del _model
print(
    f"参考: 第 2 段階・パッチトークン全体の 1 ステップ、バッチ 16: {STEP_SECONDS_BATCH16 * 1000:.1f} ms、バッチ {BATCH_SIZE}: "
    f"{STEP_SECONDS[('stage2', 'patch')] * 1000:.1f} ms(比 {STEP_SECONDS[('stage2', 'patch')] / STEP_SECONDS_BATCH16:.2f}。"
    "2 に近いほど計算量が効き、1 に近いほど固定費が支配的)"
)

# (2) 評価: 生成による採点・教師強制の負の対数尤度(検証集合の事例で計測し、評価集合・検証集合の事例数に外挿する)
_timing_model = {p: build_timing_model("stage2", p) for p in NUM_VISUAL_TOKENS}
ESTIMATE_GENERATION, ESTIMATE_NLL = {}, {}
_n_eval = len(QUESTIONS["eval"])
_n_val = {"question": len(QUESTIONS["validation"]), "caption": len(CAPTION_EXAMPLES["validation"])}


def _subset(data, n):
    return type(data)(
        data.token_ids[:n], data.response_target_mask[:n], data.prompt_lengths[:n], data.lengths[:n],
        data.image_indices[:n], data.visual_start, data.num_visual_tokens,
    )


for _pooling, _model in _timing_model.items():
    _dv = ENCODED[(_pooling, "question", "validation")]
    generate_answers(_model, _subset(_dv, 64), FEATURES["validation"], tokenizer.decode, MAX_NEW_TOKENS, None, EVAL_BATCH_SIZE)
    _times = [
        timed_call(lambda n=n: generate_answers(_model, _subset(_dv, n), FEATURES["validation"], tokenizer.decode, MAX_NEW_TOKENS, None, EVAL_BATCH_SIZE))
        for n in SCALING_EVAL_SIZES
    ]
    ESTIMATE_GENERATION[_pooling] = extrapolate(f"生成の評価 {_pooling}(事例数)", SCALING_EVAL_SIZES, _times, (_n_eval,))[_n_eval]
    for _kind in ("question", "caption"):
        _dk = ENCODED[(_pooling, _kind, "validation")]
        _times = [
            timed_call(lambda n=n: evaluate_response_negative_log_likelihood(_model, _subset(_dk, n), FEATURES["validation"], EVAL_BATCH_SIZE))
            for n in SCALING_EVAL_SIZES
        ]
        ESTIMATE_NLL[(_pooling, _kind)] = extrapolate(
            f"検証集合の負の対数尤度 {_pooling} {_kind}(事例数)", SCALING_EVAL_SIZES, _times, (_n_val[_kind],)
        )[_n_val[_kind]]
del _timing_model
empty_device_cache()

# (3) 線形プローブ(5.7 節の関数を本番と同じ特徴で使う。最も重い入力(格子全体)の時間をステップ数に比例させる)
PROBE_TIMING_STEPS = (25, 50, 100)
_probe_times = [timed_call(lambda n=n: run_probe("layout", "patch", n)) for n in PROBE_TIMING_STEPS]
ESTIMATE_PROBE = len(PROBE_TARGETS) * len(PROBE_INPUTS) * max(
    extrapolate("線形プローブ(ステップ数)", PROBE_TIMING_STEPS, _probe_times, (PROBE_STEPS,))[PROBE_STEPS], 0.0
)
SCALING_SECONDS = time.time() - _t0_scaling
print(f"スケーリングの計測: {SCALING_SECONDS:.1f}s")
```

    [学習 stage1 patch] 16: 0.57s, 32: 1.13s, 64: 2.25s -> 定常状態の 1 ステップ 35.0 ms、b=0.993; T=722: 比例 25.3s・べき乗則 25.0s(差 -0.3s)
    [学習 stage1 mean] 16: 0.30s, 32: 0.58s, 64: 1.13s -> 定常状態の 1 ステップ 17.4 ms、b=0.962; T=722: 比例 12.5s・べき乗則 11.6s(差 -0.9s)
    [学習 stage2 patch] 16: 0.59s, 32: 1.18s, 64: 2.36s -> 定常状態の 1 ステップ 36.8 ms、b=0.999; T=1,444: 比例 53.1s・べき乗則 53.0s(差 -0.1s)、T=2,166: 比例 79.7s・べき乗則 79.5s(差 -0.2s)
    [学習 stage2 mean] 16: 0.34s, 32: 0.85s, 64: 1.70s -> 定常状態の 1 ステップ 28.3 ms、b=1.166; T=1,444: 比例 40.9s・べき乗則 66.7s(差 +25.8s)、T=2,166: 比例 61.3s・べき乗則 107.0s(差 +45.7s)
    参考: 第 2 段階・パッチトークン全体の 1 ステップ、バッチ 16: 20.1 ms、バッチ 32: 36.8 ms(比 1.83。2 に近いほど計算量が効き、1 に近いほど固定費が支配的)
    [生成の評価 patch(事例数)] n=256: 1.07s, n=512: 1.16s, n=1,024: 1.34s -> b=0.161(標準誤差 0.024); n=5,776: 外挿 1.8s・比例 7.5s
    [検証集合の負の対数尤度 patch question(事例数)] n=256: 0.15s, n=512: 0.32s, n=1,024: 0.65s -> b=1.061(標準誤差 0.033); n=5,776: 外挿 4.1s・比例 3.7s
    [検証集合の負の対数尤度 patch caption(事例数)] n=256: 0.17s, n=512: 0.33s, n=1,024: 0.68s -> b=1.021(標準誤差 0.017); n=1,444: 外挿 1.0s・比例 1.0s
    [生成の評価 mean(事例数)] n=256: 1.32s, n=512: 1.40s, n=1,024: 1.16s -> b=-0.091(標準誤差 0.103); n=5,776: 外挿 1.0s・比例 6.5s
    [検証集合の負の対数尤度 mean question(事例数)] n=256: 0.07s, n=512: 0.14s, n=1,024: 0.27s -> b=1.030(標準誤差 0.029); n=5,776: 外挿 1.6s・比例 1.5s
    [検証集合の負の対数尤度 mean caption(事例数)] n=256: 0.07s, n=512: 0.14s, n=1,024: 0.28s -> b=0.991(標準誤差 0.011); n=1,444: 外挿 0.4s・比例 0.4s
    [線形プローブ(ステップ数)] n=25: 0.15s, n=50: 0.27s, n=100: 0.54s -> b=0.920(標準誤差 0.031); n=500: 外挿 2.3s・比例 2.7s
    スケーリングの計測: 30.1s




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/05_vision_language/021_llava_visual_instruction_tuning.ipynb
