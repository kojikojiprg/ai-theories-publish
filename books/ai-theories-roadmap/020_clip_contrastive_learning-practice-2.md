---
title: "CLIP と対照学習 / CLIP and Contrastive Learning(実装・実験編 2/5)"
---

この記事は後編(実装・実験編 2/5)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/020_clip_contrastive_learning-practice-1)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/020_clip_contrastive_learning-practice-3)。

### 5.4 モデルの構築と不変条件の確認

- **パラメータ数**: 画像 encoder・テキスト encoder・射影・温度とバイアスのパラメータ数を印字する。
- **初期化の一致**: 同じシードで構築した softmax・sigmoid・NegCLIP のモデルの重みが、温度とバイアス(初期値が損失ごとに違う
  スカラー)を除いて bit 単位で一致すること。実験 A・C の対応のある差の前提である。
- **テキスト encoder**: 終端トークンより後ろのパディングを別のトークンに置き換えても、終端トークンの位置の特徴が bit 単位で変わらない
  こと(因果マスクの確認)と、終端トークンより前のトークンを変えると特徴が変わること(確認の検出力)。
- **損失**: 3 つの損失が、ループで書いた参照実装と FP64 で一致すること。NegCLIP の損失は、困難な負例が 0 個なら softmax の損失と
  一致し、正例と同じキャプションになった困難な負例をその行の分母から除くこと。バッチ内のマージンと検索の正解率も参照実装と比べる。
- **NegCLIP の固定長の困難な負例**: 使わない負例を埋め草(その行の正例のキャプション)に置き換えてマスクで除いた固定長の損失が、使う負例だけを
  並べた可変長の損失と FP64 で一致し、埋め草の埋め込みへの勾配が厳密に 0 であること(分母から除いた項が損失にも勾配にも寄与しない)。
- **学習の最適化の前後の一致**: 1 ステップの固定費を削る変更(6.3 節の前の「停止した本番の実行」を参照)の前のコミット(`81adad2`)を
  `git worktree`で取得し、同一環境のサブプロセスで、3 つの損失について小さなモデルを CPU・FP32 で 6 ステップ学習する。重み・損失・記録
  (勾配のノルム・学習率・温度・マージン・バッチの順序のハッシュ・テキスト encoder に入れたキャプションの集合など)を比べる。
  softmax・sigmoid は bit 単位で一致すること、NegCLIP は固定長化で分母の和の順序とテキスト encoder のバッチの大きさが変わるので、
  差の最大値を印字して $10^{-5}$ 以下であることを確かめる。
- **困難な負例の除外**: NegCLIP の学習ループを 3 ステップ回し、テキスト encoder に入れた全系列が学習用のキャプション(除外した組を
  含まない)であること、色の入れ替えの一部だけが分母から除かれ、順序の入れ替えは除かれないことを確かめる。除外をしない設定では
  除外した組を含むキャプションが入ることも確かめる(確認の検出力)。
- **バッチの切り出し**: `DistinctCaptionBatchSampler`の各バッチでキャプションが互いに異なり、各キャプションのシーンが巡回的に
  (使われた回数の差が 1 以内で)使われること。
- **既存モジュールの後方互換性**: `src/models/vit.py`・`src/training/optimizer.py`を変更する前のコミット(`50820bf`、019 の最終
  コミット)を`git worktree`で取得し、その worktree の`src`と現行の`src`で同じスクリプトをサブプロセスとして実行する
  (`sys.executable`で起動するので同一の環境)。スクリプトは CPU で、3 通りの ViT(パッチサイズ・位置埋め込みの方式)の logits・注意の
  重み・勾配・`state_dict`と、既定の`AdamW`で 3 ステップ更新した後の重みの SHA-256 を出力する。**期待値をハードコードせず、2 つの
  サブプロセスの出力どうしを比べる。** 参照側が worktree 配下の`src`を読み込み、変更前の実装(`forward_features`・`foreach`を持たない)
  であることも確かめる。
- **`AdamW(foreach=True)`**: 既定の経路と CPU で bit 単位で一致すること。実行したデバイスでの差の最大値も印字する(演算の起動の
  まとめ方が違うので、GPU では丸めの差がありうる。本トピックの全学習は同じ`foreach=True`の経路を使うので、条件間の比較には影響しない)。


```python
_t0_checks = time.time()


def build_clip(loss_type: str) -> CLIPDualEncoder:
    # 乱数の状態は呼び出し側が torch.manual_seed で決める
    vision = VisionTransformer(**VISION_CONFIG)
    text = CausalTextTransformer(len(VOCABULARY), end_token_id=VOCABULARY.end_id, **TEXT_CONFIG)
    return CLIPDualEncoder(vision, text, EMBEDDING_DIM, INIT_LOGIT_SCALE[loss_type], INIT_LOGIT_BIAS)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --- パラメータ数と、損失の間での初期化の一致 ---
_init_hashes = {}
for _loss in ("softmax", "sigmoid", "negclip"):
    torch.manual_seed(INIT_SEED_BASE)
    _m = build_clip(_loss)
    _init_hashes[_loss] = sha256_of_state(_m.state_dict(), exclude=("logit_scale", "logit_bias"))
PARAMETER_COUNTS = count_parameters_by_part(_m)
assert len(set(_init_hashes.values())) == 1, _init_hashes
print(
    "パラメータ数: "
    + "、".join(f"{k} {v:,}" for k, v in PARAMETER_COUNTS.items())
    + f"、合計 {count_parameters(_m):,}"
)
print("同じシードの softmax・sigmoid・NegCLIP のモデルの初期値が、温度とバイアスを除いて bit 単位で一致: OK")

# --- テキスト encoder: 因果マスクと終端トークンの位置 ---
torch.manual_seed(0)
_text = CausalTextTransformer(len(VOCABULARY), end_token_id=VOCABULARY.end_id, **TEXT_CONFIG).eval()
_tokens = UNIVERSE.token_ids[:: max(1, NUM_CAPTIONS // 64)][:64].clone()
_tokens = _tokens[(_tokens == VOCABULARY.pad_id).any(dim=1)]  # パディングを持つ(above の)キャプション
assert len(_tokens) > 0
_scrambled = _tokens.clone()
_pads = _scrambled == VOCABULARY.pad_id
_scrambled[_pads] = torch.randint(3, len(VOCABULARY), (int(_pads.sum()),), generator=torch.Generator().manual_seed(1))
_changed = _tokens.clone()
_changed[:, 2] = (_changed[:, 2] - 7 + 1) % len(COLOR_NAMES) + 7  # 1 つ目の図形の色を変える(終端より前)
assert VOCABULARY.tokens[7] == COLOR_NAMES[0]
with torch.no_grad():
    _f, _f_scrambled, _f_changed = _text(_tokens), _text(_scrambled), _text(_changed)
    _hidden = _text.token_embedding(_tokens) + _text.position_embedding
    _mask = _text.causal_mask
    for _block in _text.blocks:
        _hidden, _, _ = _block(_hidden, None, tgt_mask=_mask)
    _hidden = _text.final_norm(_hidden)
    _end = (_tokens == VOCABULARY.end_id).int().argmax(dim=1)
assert torch.equal(_f, _f_scrambled)  # 終端より後ろは特徴に影響しない
assert (_f - _f_changed).abs().max() > 1e-3  # 終端より前のトークンは影響する(確認の検出力)
assert torch.equal(_f, _hidden[torch.arange(len(_tokens)), _end])
print(
    f"テキスト encoder: 終端より後ろのパディングを置き換えても特徴が bit 単位で不変({len(_tokens)} 文)、終端より前を変えると変わる、"
    "終端トークンの位置の表現を取り出している: OK"
)

# --- 損失と評価の関数を参照実装と比べる(FP64、CPU) ---
_gen = torch.Generator().manual_seed(2)
_n, _e = 6, 5
_u = functional.normalize(torch.randn(_n, _e, generator=_gen, dtype=torch.float64), dim=-1)
_v = functional.normalize(torch.randn(_n, _e, generator=_gen, dtype=torch.float64), dim=-1)
_hard = functional.normalize(torch.randn(4, _e, generator=_gen, dtype=torch.float64), dim=-1)
_t, _b = torch.tensor(2.3, dtype=torch.float64), torch.tensor(-4.0, dtype=torch.float64)
_s = (_u @ _v.t()).tolist()
_scale = math.exp(2.3)
_ref_i2t = -sum(_scale * _s[i][i] - math.log(sum(math.exp(_scale * _s[i][j]) for j in range(_n))) for i in range(_n)) / _n
_ref_t2i = -sum(_scale * _s[i][i] - math.log(sum(math.exp(_scale * _s[j][i]) for j in range(_n))) for i in range(_n)) / _n
assert math.isclose(float(softmax_contrastive_loss(_u, _v, _t)), (_ref_i2t + _ref_t2i) / 2, rel_tol=1e-12)
_ref_sig = -sum(
    math.log(1 / (1 + math.exp(-(1 if i == j else -1) * (_scale * _s[i][j] - 4.0)))) for i in range(_n) for j in range(_n)
) / _n
assert math.isclose(float(sigmoid_contrastive_loss(_u, _v, _t, _b)), _ref_sig, rel_tol=1e-12)
_pos_ids = torch.tensor([10, 11, 12, 13, 14, 15])
assert torch.equal(
    negclip_loss(_u, _v, _hard[:0], _t, _pos_ids, torch.tensor([], dtype=torch.int64)), softmax_contrastive_loss(_u, _v, _t)
)
_hard_ids = torch.tensor([11, 20, 21, 22])  # 1 つ目の困難な負例は、行 1 の正例と同じキャプション
_sh = (_u @ _hard.t()).tolist()
_ref_neg_i2t = 0.0
for i in range(_n):
    _terms = [math.exp(_scale * _s[i][j]) for j in range(_n)]
    _terms += [math.exp(_scale * _sh[i][k]) for k in range(4) if int(_hard_ids[k]) != int(_pos_ids[i])]
    _ref_neg_i2t += -(_scale * _s[i][i] - math.log(sum(_terms))) / _n
assert math.isclose(float(negclip_loss(_u, _v, _hard, _t, _pos_ids, _hard_ids)), (_ref_neg_i2t + _ref_t2i) / 2, rel_tol=1e-12)
_ref_margin = sum(_s[i][i] - max(_s[i][j] for j in range(_n) if j != i) for i in range(_n)) / _n
assert math.isclose(float(in_batch_margin(_u, _v)), _ref_margin, rel_tol=1e-12)
_true = torch.tensor([0, 1, 3, 3, 4, 2])
_r = evaluate_retrieval(_u, _v, _true)
assert _r["num_candidates"] == _n and np.array_equal(_r["correct"], (torch.tensor(_s).argmax(dim=1) == _true).numpy())
print("損失(softmax・sigmoid・NegCLIP)・バッチ内のマージン・検索の正解率が参照実装と一致(FP64)、NegCLIP の偽の負例の除外: OK")

# --- バッチの切り出し ---
_sampler = DistinctCaptionBatchSampler(TRAIN_SCENE_CAPTION_IDS, TRAIN_COPIES, STANDARD_BATCH_SIZE, torch.Generator().manual_seed(3))
_uses = np.zeros(len(TRAIN_SCENE_CAPTION_IDS), dtype=np.int64)
for _ in range(400):
    _batch = _sampler.next_batch().numpy()
    assert len(np.unique(TRAIN_SCENE_CAPTION_IDS[_batch])) == STANDARD_BATCH_SIZE
    np.add.at(_uses, _batch, 1)
_per_caption = _uses.reshape(len(TRAIN_CAPTION_IDS), TRAIN_COPIES)
assert (_per_caption.max(axis=1) - _per_caption.min(axis=1) <= 1).all()
print(
    f"バッチの切り出し: 400 バッチ(N = {STANDARD_BATCH_SIZE})でキャプションが互いに異なり、各キャプションのシーンの使用回数の差が 1 以内: OK"
)

# --- NegCLIP の困難な負例から、除外した組を含むキャプションを除く(学習ループの確認) ---
_filter_check = {}
for _label, _allowed in (("除外あり", TRAIN_CAPTION_MASK), ("除外なし(確認の検出力)", None)):
    torch.manual_seed(INIT_SEED_BASE)
    _h = train_contrastive_model(
        build_clip("negclip").to(device), TRAIN_IMAGES, TOKENS, TRAIN_SCENE_CAPTION_IDS, TRAIN_COPIES, "negclip",
        num_steps=3, batch_size=64, peak_learning_rate=1e-4, warmup_steps=1, min_learning_rate=1e-6, weight_decay=WEIGHT_DECAY,
        gradient_clip_threshold=GRADIENT_CLIP_THRESHOLD, data_seed=5, hard_negative_ids=UNIVERSE.hard_negative_ids,
        hard_negative_allowed=_allowed, logit_scale_max=LOGIT_SCALE_MAX["negclip"],
    )
    _filter_check[_label] = _h
_h = _filter_check["除外あり"]
assert TRAIN_CAPTION_MASK[_h["text_caption_ids"]].all()  # テキスト encoder に入れた系列はすべて学習用のキャプション
assert _h["hard_negatives_used"]["order_swap"] == _h["hard_negatives_generated"]["order_swap"]
assert _h["hard_negatives_used"]["attribute_swap"] < _h["hard_negatives_generated"]["attribute_swap"]
assert not TRAIN_CAPTION_MASK[_filter_check["除外なし(確認の検出力)"]["text_caption_ids"]].all()
print(
    "NegCLIP の困難な負例の除外(3 ステップ、N = 64): 除外ありではテキスト encoder に入れた全系列が学習用のキャプション、"
    f"分母に加えた数 {_h['hard_negatives_used']}(規則が作った数 {_h['hard_negatives_generated']})。除外なしでは除外した組を含む"
    "キャプションが入る(確認の検出力): OK"
)
del _filter_check, _h

# --- NegCLIP の固定長の困難な負例: 埋め草をマスクで除いた損失と、可変長の損失の一致(FP64) ---
_gen = torch.Generator().manual_seed(6)
_n = 5
_u = functional.normalize(torch.randn(_n, 7, generator=_gen, dtype=torch.float64), dim=-1).requires_grad_()
_v = functional.normalize(torch.randn(_n, 7, generator=_gen, dtype=torch.float64), dim=-1).requires_grad_()
_hard_all = functional.normalize(torch.randn(2 * _n, 7, generator=_gen, dtype=torch.float64), dim=-1).requires_grad_()
_pos = torch.arange(100, 100 + _n)
_hard_ids_all = torch.arange(200, 200 + 2 * _n)
_valid = torch.tensor([True, False, True, True, False, True, False, True, True, True])
_t = torch.tensor(2.1, dtype=torch.float64)
_fixed = negclip_loss(_u, _v, _hard_all, _t, _pos, _hard_ids_all, hard_negative_mask=_valid)
_grads_fixed = torch.autograd.grad(_fixed, (_u, _v, _hard_all))
_variable = negclip_loss(_u, _v, _hard_all[_valid], _t, _pos, _hard_ids_all[_valid])
_grads_variable = torch.autograd.grad(_variable, (_u, _v, _hard_all))
assert math.isclose(float(_fixed), float(_variable), rel_tol=1e-12)
for _a, _b in zip(_grads_fixed, _grads_variable, strict=True):
    assert torch.allclose(_a, _b, rtol=1e-12, atol=1e-15)
assert torch.equal(_grads_fixed[2][~_valid], torch.zeros_like(_grads_fixed[2][~_valid]))  # 埋め草への勾配は厳密に 0
print("NegCLIP の固定長の困難な負例: マスクで除いた損失と勾配が可変長の損失と一致(FP64)、埋め草の埋め込みへの勾配が厳密に 0: OK")

# --- 学習の最適化の前後の一致(変更前のコミット 81adad2 の学習関数と比べる、CPU・FP32) ---
_TRAIN_REFERENCE_COMMIT = "81adad2"  # 1 ステップの固定費を削る変更の前のコミット(停止した本番の実行に使ったもの)
_TRAIN_WORKTREE_DIR = ROOT / ".cache" / "_020_training_equivalence_worktree"
_train_script = '''import math, sys, numpy as np, torch
from src.data.synthetic_scenes import CaptionVocabulary, build_caption_universe, render_scenes, repeat_caption_ids
from src.models.clip import CLIPDualEncoder, CausalTextTransformer
from src.models.vit import VisionTransformer
from src.training.contrastive import train_contrastive_model
print("SRC_FILE:" + __import__("src").__file__, file=sys.stderr)
torch.set_num_threads(1)
v = CaptionVocabulary(); u = build_caption_universe(v)
ids = repeat_caption_ids(u.train_caption_ids, 4); images = render_scenes(u, ids, 7).images
allowed = np.zeros(u.num_captions, dtype=bool); allowed[u.train_caption_ids] = True
out = {}
for loss in ("softmax", "sigmoid", "negclip"):
    torch.manual_seed(3)
    model = CLIPDualEncoder(VisionTransformer(32, 4, 3, 1, 32, 2, 2, 64, "learned"),
                            CausalTextTransformer(len(v), 10, 32, 2, 2, 64, v.end_id), 16,
                            math.log(10.0) if loss == "sigmoid" else math.log(1 / 0.07), -10.0)
    kw = dict(hard_negative_ids=u.hard_negative_ids, hard_negative_allowed=allowed) if loss == "negclip" else {}
    h = train_contrastive_model(model, images, u.token_ids, ids, 4, loss, 6, 16, 1e-3, 2, 1e-5, 0.1, 1.0, 11,
                                logit_scale_max=None if loss == "sigmoid" else math.log(100.0),
                                evaluation_steps=(3,), evaluation_fn=lambda m: {"scale": float(m.logit_scale)}, **kw)
    keep = {k: h[k] for k in ("loss", "gradient_norm", "learning_rate", "logit_scale", "logit_bias", "in_batch_margin",
                              "loss_scale", "step_skipped", "evaluations", "examples_seen", "data_stream_hash",
                              "distinct_captions_in_every_batch", "text_caption_ids", "hard_negatives_used",
                              "hard_negatives_generated") if k in h}
    keep["text_caption_ids"] = np.asarray(keep["text_caption_ids"]).tolist()
    out[loss] = {"state": {k: t.clone() for k, t in model.state_dict().items()}, "history": keep}
torch.save(out, sys.argv[1])
'''
subprocess.run(["git", "worktree", "remove", "--force", str(_TRAIN_WORKTREE_DIR)], capture_output=True)
_r = subprocess.run(
    ["git", "worktree", "add", "--detach", str(_TRAIN_WORKTREE_DIR), _TRAIN_REFERENCE_COMMIT], capture_output=True, text=True
)
assert _r.returncode == 0, _r.stderr
(_TRAIN_WORKTREE_DIR / "_train_equivalence.py").write_text(_train_script)
(ROOT / "_train_equivalence.py").write_text(_train_script)
_outputs = {"reference": Path(tempfile.mkdtemp()) / "reference.pt", "current": Path(tempfile.mkdtemp()) / "current.pt"}
try:
    for _label, _cwd in (("reference", _TRAIN_WORKTREE_DIR), ("current", ROOT)):
        _p = subprocess.run([sys.executable, "_train_equivalence.py", str(_outputs[_label])], cwd=_cwd, capture_output=True, text=True)
        assert _p.returncode == 0, _p.stderr[-3000:]
        assert "SRC_FILE:" + str((_cwd / "src" / "__init__.py").resolve()) in _p.stderr, _p.stderr[-2000:]
finally:
    (ROOT / "_train_equivalence.py").unlink(missing_ok=True)
    subprocess.run(["git", "worktree", "remove", "--force", str(_TRAIN_WORKTREE_DIR)], capture_output=True)
_reference, _current = (torch.load(_outputs[k], weights_only=False) for k in ("reference", "current"))
TRAINING_EQUIVALENCE = {}
for _loss in ("softmax", "sigmoid", "negclip"):
    _sa, _sb = _reference[_loss]["state"], _current[_loss]["state"]
    _ha, _hb = _reference[_loss]["history"], _current[_loss]["history"]
    assert _sa.keys() == _sb.keys() and _ha.keys() == _hb.keys()
    _bitwise = all(torch.equal(_sa[k], _sb[k]) for k in _sa) and _ha == _hb
    _weight_diff = max(float((_sa[k].double() - _sb[k].double()).abs().max()) for k in _sa)
    _loss_diff = max(abs(a - b) for a, b in zip(_ha["loss"], _hb["loss"], strict=True))
    _differing = sorted(k for k in _ha if _ha[k] != _hb[k])
    TRAINING_EQUIVALENCE[_loss] = {"bitwise": _bitwise, "weight_diff": _weight_diff, "loss_diff": _loss_diff, "differing": _differing}
    print(
        f"  {_loss}: bit 単位で一致 {_bitwise}、重みの差の最大値 {_weight_diff:.2e}、損失の差の最大値 {_loss_diff:.2e}、"
        f"値が異なる記録 {_differing or 'なし'}"
    )
assert TRAINING_EQUIVALENCE["softmax"]["bitwise"] and TRAINING_EQUIVALENCE["sigmoid"]["bitwise"]
_neg = TRAINING_EQUIVALENCE["negclip"]
assert _neg["weight_diff"] <= 1e-5 and _neg["loss_diff"] <= 1e-5
assert set(_neg["differing"]) <= {"loss", "gradient_norm", "in_batch_margin"}  # バッチの順序・入れた系列などは一致
print(
    f"学習の最適化の前後の一致(参照コミット {_TRAIN_REFERENCE_COMMIT}、CPU・FP32、6 ステップ): softmax・sigmoid は重みと全記録が bit 単位で一致、"
    "NegCLIP は固定長化による丸めの差のみ(10^-5 以下、バッチの順序・テキスト encoder に入れた系列の集合・負例の数は一致): OK"
)

# --- 既存モジュールの後方互換性(変更前のコミットとの bit 単位の一致) ---
_REFERENCE_COMMIT = "50820bf"  # src/models/vit.py・src/training/optimizer.py を 020 で変更する前の最後のコミット
_WORKTREE_DIR = ROOT / ".cache" / "_020_backward_compat_worktree"
_compat_script = '''
import hashlib
import inspect
import json
import sys

import torch

from src.models.vit import VisionTransformer
from src.training.optimizer import AdamW

print("SRC_FILE:" + __import__("src").__file__, file=sys.stderr)
print("HAS_FORWARD_FEATURES:" + str(hasattr(VisionTransformer, "forward_features")), file=sys.stderr)
print("HAS_FOREACH:" + str("foreach" in inspect.signature(AdamW.__init__).parameters), file=sys.stderr)


def h(t):
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()


out = {}
generator = torch.Generator().manual_seed(0)
images = torch.randn(4, 3, 32, 32, generator=generator)
labels = torch.randint(0, 10, (4,), generator=generator)
for patch, pe in ((4, "learned"), (8, "sinusoidal_2d"), (4, "none")):
    torch.manual_seed(5)
    model = VisionTransformer(32, patch, 3, 10, 64, 2, 4, 128, pe)
    torch.nn.init.normal_(model.head.weight, std=0.5)
    key = f"P{patch}_{pe}"
    out[key + "_state"] = {n: h(t) for n, t in model.state_dict().items()}
    logits, weights = model(images, return_attention_weights=True)
    out[key + "_logits"] = h(logits)
    out[key + "_logits_plain"] = h(model(images))
    out[key + "_attention"] = [h(w) for w in weights]
    torch.nn.functional.cross_entropy(logits, labels).backward()
    out[key + "_gradients"] = {n: h(p.grad) for n, p in model.named_parameters() if p.grad is not None}
    optimizer = AdamW(list(model.parameters()), lr=1e-2, weight_decay=0.1)
    for _ in range(3):
        optimizer.step()
    out[key + "_after_adamw"] = {n: h(p) for n, p in model.named_parameters()}
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
    assert "HAS_FORWARD_FEATURES:False" in _ref.stderr and "HAS_FOREACH:False" in _ref.stderr, "参照側が変更後の実装を読み込んでいる"
    _new = subprocess.run([sys.executable, "_compat_script.py"], cwd=ROOT, capture_output=True, text=True)
    assert _new.returncode == 0, _new.stderr
    assert "SRC_FILE:" + str((ROOT / "src" / "__init__.py").resolve()) in _new.stderr, _new.stderr
    assert "HAS_FORWARD_FEATURES:True" in _new.stderr and "HAS_FOREACH:True" in _new.stderr, "現行側が変更後の実装を読み込んでいない"
    _ref_out, _new_out = json.loads(_ref.stdout), json.loads(_new.stdout)
    assert set(_ref_out) == set(_new_out)
    _mismatch = [k for k in _ref_out if _ref_out[k] != _new_out[k]]
    print(f"  比べた項目 {len(_ref_out)} 個(logits・注意の重み・勾配・state_dict・AdamW の 3 ステップ後の重み)、不一致 {len(_mismatch)} 個")
    assert not _mismatch, _mismatch
    print(
        f"参照コミット {_REFERENCE_COMMIT}(worktree 配下の src を読み込んだことを確認)と現行の src で、VisionTransformer の出力・"
        "注意の重み・勾配・state_dict と、既定の AdamW の更新が bit 単位で一致: OK"
    )
finally:
    (ROOT / "_compat_script.py").unlink(missing_ok=True)
    subprocess.run(["git", "worktree", "remove", "--force", str(_WORKTREE_DIR)], capture_output=True)

# --- AdamW(foreach=True)と既定の経路の一致 ---
ADAMW_FOREACH_MAX_DIFF = {}
for _dev in sorted({torch.device("cpu"), device}, key=str):
    _models = []
    for _ in range(2):
        torch.manual_seed(INIT_SEED_BASE)
        _models.append(build_clip("softmax").to(_dev))
    _opts = [
        AdamW(list(_models[0].parameters()), lr=1e-3, weight_decay=WEIGHT_DECAY),
        AdamW(list(_models[1].parameters()), lr=1e-3, weight_decay=WEIGHT_DECAY, foreach=True),
    ]
    _g = torch.Generator().manual_seed(4)
    for _step in range(5):
        _grads = [torch.randn(p.shape, generator=_g) for p in _models[0].parameters()]
        for _model in _models:
            for _p, _gr in zip(_model.parameters(), _grads, strict=True):
                _p.grad = _gr.to(_dev)
        for _o in _opts:
            _o.step()
    ADAMW_FOREACH_MAX_DIFF[_dev.type] = max(
        float((a - b).detach().abs().max()) for a, b in zip(_models[0].parameters(), _models[1].parameters(), strict=True)
    )
    if _dev.type == "cpu":
        assert all(torch.equal(a, b) for a, b in zip(_models[0].parameters(), _models[1].parameters(), strict=True))
    assert ADAMW_FOREACH_MAX_DIFF[_dev.type] < 1e-6, ADAMW_FOREACH_MAX_DIFF
print(
    "AdamW(foreach=True)と既定の経路(CLIP の全パラメータ、5 ステップ): CPU で bit 単位で一致: OK。差の最大値 "
    + "、".join(f"{k}: {v:.1e}" for k, v in ADAMW_FOREACH_MAX_DIFF.items())
)
del _m, _text, _models, _opts
CHECK_SECONDS = time.time() - _t0_checks
print(f"確認の実行時間: {CHECK_SECONDS:.1f}s")
```

    パラメータ数: vision 806,016、text 795,136、projection 16,384、scale_and_bias 2、合計 1,617,538
    同じシードの softmax・sigmoid・NegCLIP のモデルの初期値が、温度とバイアスを除いて bit 単位で一致: OK
    テキスト encoder: 終端より後ろのパディングを置き換えても特徴が bit 単位で不変(32 文)、終端より前を変えると変わる、終端トークンの位置の表現を取り出している: OK
    損失(softmax・sigmoid・NegCLIP)・バッチ内のマージン・検索の正解率が参照実装と一致(FP64)、NegCLIP の偽の負例の除外: OK
    バッチの切り出し: 400 バッチ(N = 256)でキャプションが互いに異なり、各キャプションのシーンの使用回数の差が 1 以内: OK
    NegCLIP の困難な負例の除外(3 ステップ、N = 64): 除外ありではテキスト encoder に入れた全系列が学習用のキャプション、分母に加えた数 {'attribute_swap': 94, 'order_swap': 192}(規則が作った数 {'attribute_swap': 192, 'order_swap': 192})。除外なしでは除外した組を含むキャプションが入る(確認の検出力): OK
    NegCLIP の固定長の困難な負例: マスクで除いた損失と勾配が可変長の損失と一致(FP64)、埋め草の埋め込みへの勾配が厳密に 0: OK


    /tmp/ipykernel_2246/3667435015.py:143: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
    Consider using tensor.detach() first. (Triggered internally at /__w/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:822.)
      assert math.isclose(float(_fixed), float(_variable), rel_tol=1e-12)


      softmax: bit 単位で一致 True、重みの差の最大値 0.00e+00、損失の差の最大値 0.00e+00、値が異なる記録 なし
      sigmoid: bit 単位で一致 True、重みの差の最大値 0.00e+00、損失の差の最大値 0.00e+00、値が異なる記録 なし
      negclip: bit 単位で一致 False、重みの差の最大値 6.03e-07、損失の差の最大値 4.77e-07、値が異なる記録 ['gradient_norm', 'in_batch_margin', 'loss']
    学習の最適化の前後の一致(参照コミット 81adad2、CPU・FP32、6 ステップ): softmax・sigmoid は重みと全記録が bit 単位で一致、NegCLIP は固定長化による丸めの差のみ(10^-5 以下、バッチの順序・テキスト encoder に入れた系列の集合・負例の数は一致): OK
      比べた項目 18 個(logits・注意の重み・勾配・state_dict・AdamW の 3 ステップ後の重み)、不一致 0 個
    参照コミット 50820bf(worktree 配下の src を読み込んだことを確認)と現行の src で、VisionTransformer の出力・注意の重み・勾配・state_dict と、既定の AdamW の更新が bit 単位で一致: OK
    AdamW(foreach=True)と既定の経路(CLIP の全パラメータ、5 ステップ): CPU で bit 単位で一致: OK。差の最大値 cpu: 0.0e+00、cuda: 0.0e+00
    確認の実行時間: 20.6s


### 5.5 学習と評価のヘルパー

- **学習の鍵**: 1 つの学習を`(損失, N, s)`の組で表す。標準条件 S は`("softmax", 256, s)`であり、実験 A の $N = 256$ の softmax・実験 B の
  モデル・実験 C の標準のモデルはすべてこの鍵になる。同じ鍵の学習は 1 回だけ行い、記録を共有する。
- **シード $s$ が決めるもの**: 初期化(`torch.manual_seed(20100 + s)`でモデルを構築)と、バッチの順序(CPU の生成器`20200 + s`)。
  同じ $s$ の学習どうしは、損失によらず同じ初期値から始まり(温度とバイアスを除く)、同じ $N$ なら同じ順序で同じシーンを見る
  (6.11 節でアサーションにより確かめる)。
- **評価**(学習の最終ステップの重み、FP32): 既知の組み合わせの検証集合の検索(途中の 25%・50%・75% と最後)、未見の組み合わせの
  評価集合の検索の正解率 $M$ とマージン、2 択の正解率(色の入れ替え・順序の入れ替え・ランダムな負例)、既知の組み合わせの検証集合での
  2 択の正解率(実験 C の診断量)、modality gap(学習の前と後)、学習中のバッチ内のマージン(末尾 10% のステップの平均)。シード 0 では
  射影図のための埋め込みを残す。
- **較正の学習**(`purpose="calibration"`)では、未見の組み合わせの評価集合を一切評価しない(記録に評価集合の鍵を持たないことを
  6.11 節で確かめる)。
- 学習で見る事例数`EXAMPLES`は、6.4 節で実行計画を選んだ後に決まる。


```python
def key_of(loss_type: str, batch_size: int, seed: int) -> tuple:
    return (loss_type, batch_size, seed)


def build_model_for(key: tuple) -> CLIPDualEncoder:
    torch.manual_seed(INIT_SEED_BASE + key[2])
    return build_clip(key[0]).to(device)


def train_run(model, key, learning_rate, num_steps, evaluation_steps=(), evaluation_fn=None, execution_mode=None) -> dict:
    use_fp16, use_graph = EXECUTION_MODES[execution_mode or EXECUTION_MODE]
    return train_contrastive_model(
        model,
        TRAIN_IMAGES,
        TOKENS,
        TRAIN_SCENE_CAPTION_IDS,
        TRAIN_COPIES,
        key[0],
        num_steps=num_steps,
        batch_size=key[1],
        peak_learning_rate=learning_rate,
        warmup_steps=warmup_steps_for(num_steps),
        min_learning_rate=learning_rate * MIN_LEARNING_RATE_RATIO,
        weight_decay=WEIGHT_DECAY,
        gradient_clip_threshold=GRADIENT_CLIP_THRESHOLD,
        data_seed=DATA_SEED_BASE + key[2],
        hard_negative_ids=UNIVERSE.hard_negative_ids if key[0] == "negclip" else None,
        hard_negative_allowed=TRAIN_CAPTION_MASK,  # 除外した組を含むキャプションは困難な負例に使わない
        logit_scale_max=LOGIT_SCALE_MAX[key[0]],
        use_fp16_autocast=use_fp16,
        use_cuda_graph=use_graph,
        init_loss_scale=INIT_LOSS_SCALE,
        loss_scale_growth_interval=LOSS_SCALE_GROWTH_INTERVAL,
        evaluation_steps=evaluation_steps,
        evaluation_fn=evaluation_fn,
    )


def summary(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "correct"}


def evaluate_seen(model) -> dict:
    # 既知の組み合わせの検証集合の検索(候補は学習用の全キャプション)。学習率の較正と前提条件 P1 に使う
    text = encode_texts(model, TOKENS)
    return summary(evaluate_retrieval(encode_images(model, VALIDATION_IMAGES), text[TRAIN_CANDIDATES], VALIDATION_TRUE_INDEX))


def two_alternative_accuracies(image_embeddings, text_embeddings, positive_ids, random_sets=None) -> dict:
    # random_sets: 名前 -> (枚数, k) の負例のキャプションの番号。各列を 1 回の 2 択とし、列をまとめて正解率を出す
    result = {}
    for rule, ids in HARD_NEGATIVE_IDS_DEVICE.items():
        result[rule] = evaluate_two_alternative(image_embeddings, text_embeddings, positive_ids, ids[positive_ids])
    result["swap"] = np.concatenate([result["attribute_swap"], result["order_swap"]])
    for name, ids in (random_sets or {}).items():
        result[name] = np.concatenate(
            [evaluate_two_alternative(image_embeddings, text_embeddings, positive_ids, ids[:, k]) for k in range(ids.shape[1])]
        )
    return {k: float(v.mean()) for k, v in result.items()} | {f"{k}_count": int(v.size) for k, v in result.items()}


def evaluate_final(model, keep_projection: bool) -> dict:
    text = encode_texts(model, TOKENS)
    unseen_images = encode_images(model, UNSEEN_IMAGES)
    seen_images = encode_images(model, VALIDATION_IMAGES)
    unseen = evaluate_retrieval(unseen_images, text[UNSEEN_CANDIDATES], UNSEEN_TRUE_INDEX)
    result = {
        "unseen_retrieval": summary(unseen),
        "unseen_two_alternative": two_alternative_accuracies(unseen_images, text, UNSEEN_POSITIVE_IDS, RANDOM_NEGATIVE_SETS_DEVICE),
        "seen_two_alternative": two_alternative_accuracies(seen_images, text, VALIDATION_POSITIVE_IDS),
        "gap_unseen": modality_gap(unseen_images, text[UNSEEN_POSITIVE_IDS]),
        "gap_seen": modality_gap(seen_images, text[VALIDATION_POSITIVE_IDS]),
    }
    if keep_projection:
        result["projection"] = {
            "image": unseen_images[:PROJECTION_IMAGES].cpu().numpy(),
            "text": text[UNSEEN_POSITIVE_IDS[:PROJECTION_IMAGES]].cpu().numpy(),
        }
    return result


def learning_rate_for(key: tuple) -> float:
    # 6.5 節の較正の結果から、学習の鍵の学習率を決める(NegCLIP は標準条件 S の学習率を使う)
    loss_type, batch_size, _ = key
    base_loss = "softmax" if loss_type == "negclip" else loss_type
    if CALIBRATION_MODE == "all" or loss_type == "negclip":
        return CALIBRATION[(base_loss, batch_size)]["chosen"]
    return CALIBRATION[(base_loss, STANDARD_BATCH_SIZE)]["chosen"] * (batch_size / STANDARD_BATCH_SIZE) ** LR_BATCH_EXPONENT


def run(key: tuple, learning_rate: float, purpose: str, keep_state: bool = False) -> dict:
    assert purpose in ("calibration", "main")
    t0 = time.time()
    num_steps = steps_for(EXAMPLES, key[1])
    model = build_model_for(key)
    initial_hash = sha256_of_state(model.state_dict(), exclude=("logit_scale", "logit_bias"))
    record = {
        "key": key,
        "purpose": purpose,
        "learning_rate": learning_rate,
        "examples": EXAMPLES,
        "train_steps": num_steps,
        "initial_state_sha256": initial_hash,
        "execution_mode": EXECUTION_MODE,
    }
    if purpose == "main":  # 学習の前の modality gap(初期化の時点、3.8 節)
        _text0 = encode_texts(model, TOKENS)
        record["gap_unseen_at_init"] = modality_gap(encode_images(model, UNSEEN_IMAGES), _text0[UNSEEN_POSITIVE_IDS])
    record["history"] = train_run(
        model,
        key,
        learning_rate,
        num_steps,
        intermediate_eval_steps_for(num_steps),
        lambda m: {"seen_accuracy": evaluate_seen(m)["accuracy"]},
    )
    record["seen"] = evaluate_seen(model)  # 最終ステップの重み
    tail = max(1, round(MARGIN_TAIL_FRACTION * num_steps))
    record["train_margin_tail"] = float(np.mean(record["history"]["in_batch_margin"][-tail:]))
    if purpose == "main":  # 未見の組み合わせの評価集合は本番の学習でのみ評価する(較正では評価しない)
        record["final"] = evaluate_final(model, keep_projection=(key[2] == 0))
        if keep_state:
            record["state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    record["seconds"] = time.time() - t0
    del model
    empty_device_cache()
    return record


def describe(record: dict) -> str:
    loss_type, batch_size, seed = record["key"]
    text = f"{loss_type} N={batch_size} s={seed} lr={record['learning_rate']:.3g}: 既知の検証 {record['seen']['accuracy']:.4f}"
    if "final" in record:
        f = record["final"]
        text += (
            f"、M {f['unseen_retrieval']['accuracy']:.4f}、2 択 swap {f['unseen_two_alternative']['swap']:.4f}"
            f"・rand {f['unseen_two_alternative']['random']:.4f}"
        )
    h = record["history"]
    return (
        text + f"、最終の訓練損失 {np.mean(h['loss'][-10:]):.3f}、exp(t) {math.exp(h['logit_scale'][-1]):.1f}、"
        f"飛ばしたステップ {sum(h['step_skipped'])}、{record['seconds']:.0f}s"
    )


def judge(value: float, sigma: float) -> str:
    # 支持 / 反証 / 判定不能(前提不成立は 6.12 節で前提条件から決める)
    if value > SIGMA_MULTIPLIER * sigma:
        return "支持"
    if value < -SIGMA_MULTIPLIER * sigma:
        return "反証"
    return "判定不能"


def paired_contrast(values: np.ndarray) -> dict:
    # シードごとの対応のある量 d_s の平均と、その標準偏差 sd(d_s) / sqrt(n)
    values = np.asarray(values, dtype=np.float64)
    return {"value": float(values.mean()), "sigma": float(values.std(ddof=1) / math.sqrt(len(values))), "per_seed": values}


_tag = "[動作確認のみ、結論ではない] " if SMOKE_TEST else ""
_plot_tag = "[smoke test, not a result] " if SMOKE_TEST else ""  # 図のタイトル用(英字のみ)
```

## 6. 実験 / Experiments

### 6.1 実験宣言セル: 共通の設定・検証すること・判定基準・前提条件

**この節の内容は本番実行の前に確定させ、結果を見た後に変更しない。**

#### 共通の設定

- **モデル**: 画像 encoder は 019 の ViT($P = 4$ で系列長 64、出力次元 $d_i = 128$、4 層、4 ヘッド、$d_{\mathrm{ff}} = 512$、学習可能な 1 次元の
  位置埋め込み、[CLS] トークンによる読み出し)。テキスト encoder は因果マスクつきの Transformer(出力次元 $d_t = 128$、4 層、4 ヘッド、
  $d_{\mathrm{ff}} = 512$、系列長 10、語彙 20)。共通の埋め込みの次元 $d_e = 64$($E$ は学習で見る事例数に使う)。dropout なし、正規化前置、GELU。
- **学習**: AdamW(重み減衰 0.1、行列形の重みのみ)、最初の 10% のステップで線形 warmup の後 cosine で最大値の 0.01 倍まで減衰、
  gradient clipping の閾値 1.0。1 ステップの実行の方式(FP16 の autocast と動的損失スケーリング / FP32 / FP32 と CUDA graph)は、
  6.4 節で時間の見積もりのみから選び、全条件・全シードで同じものを使う(5.1 節)。温度の初期値と上限は損失ごとに
  原論文に従う(softmax・NegCLIP: $\tau = 0.07$、倍率 $e^t \le 100$。sigmoid: $t' = \log 10$、$b = -10$、切り詰めなし)。
- **データ**: 学習用のキャプション 1,444 個 × 各 32 枚 = 46,208 枚の固定のシーン。各バッチのキャプションは互いに異なる(5.3 節)。
- **学習量**: 学習で見る事例数 $E$(**6.4 節で実行計画として自動選択される値。候補は $2^{19}, 2^{18}$**)を全条件で揃え、
  バッチサイズ $N$ の条件のステップ数を $T_N = E / N$ とする(下の「学習量の揃え方」)。選ばれた $E$ は、較正を含む全条件・全シードで同一。
- **シード**: シード $s$ は初期化とバッチの順序を決める(5.5 節)。同じ $s$ の条件どうしは、損失によらず同じ初期値(温度とバイアスを
  除く)から始まり、同じ $N$ なら同じ順序で同じシーンを見る(対応のある比較)。シードは $0, 1, \dots$ を使い、数は削る段階で決まる。
- **評価の指標 $M$**: 未見の組み合わせの評価集合(796 個のキャプション × 各 4 枚 = 3,184 枚)の各画像について、**評価集合のすべての
  異なるキャプション 796 個** を候補として画像 → テキストの検索(zero-shot 分類と同じ形、3.6 節)を行い、コサイン類似度が最大の候補が
  正解である割合。候補の数は全条件で同一(796、6.11 節でアサーション)で、チャンス水準は $1/796 \approx 0.0013$。学習の最終ステップの
  重みで評価する。
- **2 択の正解率**: 未見の組み合わせの評価集合の各画像について、正例のキャプションと負例のキャプション 1 つのどちらの類似度が高いかを
  判定し、正例が高ければ正答とする。負例は、色の入れ替え・順序の入れ替え(各 1 個、あわせて $\mathrm{acc}_{\mathrm{swap}}$)と、ランダムな
  負例 2 個(あわせて $\mathrm{acc}_{\mathrm{rand}}$)。ランダムな負例は困難な負例と「既知 / 未見」の状態で対応させる: 1 個目は順序の
  入れ替えに対応させ、未見の組み合わせのキャプション(正例を除く)から一様に選ぶ。2 個目は色の入れ替えに対応させ、その画像の色の
  入れ替えのキャプションと同じ集合(学習用、または未見の組み合わせ)から、正例とその困難な負例を除いて一様に選ぶ(実験 B の
  「解釈の注意」、6.2 節の改訂 5)。どちらも各画像 2 回の 2 択で、試行の数(6,368)を揃える。
- 既知の組み合わせの検証集合(1,444 個 × 各 2 枚 = 2,888 枚、候補 1,444 個)は、学習率の較正と前提条件 P1 にのみ使う。**未見の組み合わせの
  評価集合は較正に使わない。**

#### 学習量の揃え方(見た事例数を揃える)

実験 A は、バッチサイズ $N \in \{16, 64, 256\}$ の間で **見た事例数 $E$ を揃え**、ステップ数を $T_N = E/N$ とする($N = 16$ は $N = 256$ の
16 倍のステップを学習する)。理由は次のとおり。

- ステップ数を揃えると、$N = 256$ は $N = 16$ の 16 倍の事例を見る。$N$ を大きくする効果が「負例が多いこと」と「データを多く見ること」の
  和になり、3.4・3.5 節の主張(負例の数への依存)と、単に学習量が多いことの効果を分離できない。
- 見た事例数を揃えると、encoder の順伝播・逆伝播の総計算量もほぼ揃う(類似度の行列の計算量は $1$ ステップあたり $O(N^2 E)$ で、全体で
  $O(N \cdot E_{\mathrm{total}})$ だが、encoder に比べて無視できる)。
- 代わりに、$N$ の小さい条件ほど更新の回数が多く、1 回の更新の勾配の雑音が大きい。この違いは学習率の較正(条件ごと)で一部を
  吸収するが、「更新の回数」と「負例の数」の効果は分離できない(交絡)。

#### 学習率の較正(本番の冒頭、6.5 節)

- **較正する条件**: 較正の方式`CALIBRATION_MODE`は、選ばれた実行計画から決まる(下の「実行計画」)。`"all"`(計画 0〜7)では、
  実験 A で学習する全条件(損失 × 段階が決める $N$ の水準)をそれぞれ較正する。
  `"representative"`(計画 8〜11)では、$N = 256$ の softmax と sigmoid の 2 条件のみを較正し、他の $N$ の学習率をべき乗の規則
  $\eta_N = \eta_{256} (N/256)^{3/4}$ で決める。
- **格子**: 条件 $(\ell, N)$ の格子は、中心 $\eta_c(\ell, N) = \eta_{c,\ell} (N/256)^{3/4}$ の $\{1/2, 1, 2\}$ 倍(公比 2 の 3 点)。
  $\eta_{c,\ell}$ は $N = 256$ の中心で、softmax は $2 \times 10^{-3}$、sigmoid は $10^{-3}$。softmax の格子は $N = 256$ で
  $\{10^{-3}, 2 \times 10^{-3}, 4 \times 10^{-3}\}$、$N = 64$ で約 $\{3.5 \times 10^{-4}, 7.1 \times 10^{-4}, 1.4 \times 10^{-3}\}$、$N = 16$ で
  $\{1.25 \times 10^{-4}, 2.5 \times 10^{-4}, 5 \times 10^{-4}\}$。sigmoid の格子はそれぞれその 1/2 倍。softmax の中心と指数 $3/4$ は
  6.2 節のパイロット(softmax のみ)で、sigmoid の中心は 6.2 節の改訂 3(最良の格子点の位置のみを見たパイロット)で決めた。指数 $3/4$ は、
  Adam でよく使われる平方根の規則(指数 $1/2$)と、SGD でよく使われる線形の規則(指数 1)の間にあたる。
- 本番と同じ $E$・全学習データで、**較正専用のシード**(シード番号 90。実験のシードと共有しない)の 1 シードで各格子点を学習し、
  **既知の組み合わせの検証集合の検索の正解率** が最大の学習率を選ぶ(同点なら小さい方)。
- **拡張の規則**: 最良の学習率が格子の端に来た場合は、その方向に公比 2 で 1 点だけ格子を拡張して学習し、改めて最良を選ぶ(1 回のみ)。
  019 の実験 C は、較正の最良値が拡張後も格子の端に来て前提不成立になった。その対策として、格子の中心と、条件ごとに中心をずらす規則を
  パイロットで決めた(当初の平方根の規則では、$N = 16$ のパイロットの最良値が格子の下端に来たため、指数を $3/4$ に改めた。6.2 節)。
- **NegCLIP** は、標準条件 S(softmax・$N = 256$)で選んだ学習率をそのまま使う(較正しない)。実験 C の対比を「損失の違い」だけに
  するためである。NegCLIP の最適な学習率が S と異なる可能性は、交絡として残る。
- 較正のシードを実験のシードと分けるのは、検証集合で選ばれた学習(勝者)がそのまま実験のシード 0 になり、その値だけが選択によって
  楽観的に偏るのを避けるためである。

#### 共通の前提条件

- **P0(較正)**: 較正した各条件で、選んだ学習率が格子の **内点** であること(拡張後も最良が端なら不成立)。
  `"representative"`では、平方根の規則で決めた条件の学習率は較正していないので、その条件の P0 は検証されない(較正した $N = 256$ の P0 で代える)。
- **P1(学習の成立)**: 各条件で、シード平均の **既知の組み合わせの検証集合の検索の正解率** が 0.1 以上であること(チャンス水準
  $1/1444 \approx 0.0007$ の約 144 倍)。検証したい仮説(損失・負例の種類による差)とは独立な、「学習が進んだか」の量である。

前提条件が 1 つでも成立しない実験は、判定関数の結果によらず **前提不成立** と記録する(判定不能とは区別する)。

#### 判定の形

各実験の対比量 $\Delta$ とその標準偏差 $\sigma$ に対し、$\Delta > 2\sigma$ なら **支持**、$\Delta < -2\sigma$ なら **反証**、それ以外は **判定不能** と
する。$\sigma$ はシード間のばらつきから求める(評価集合は全条件で共通なので、評価事例の抽出によるばらつきは条件間の比較では共通に効き、
対比量の標準偏差には含めない)。

#### 実行計画(学習で見る事例数 $E$ と削る段階の組)と、その自動選択

**削る段階**:

| 段階 | 内容 | 実験 A のシード数・$N$ の水準 | 実験 B・C のシード数 |
|---|---|---|---|
| 0 | 全実験 5 シード | 5、$N \in \{16, 64, 256\}$ | 5 |
| 1 | 実験 A の $N = 64$ を除く | 5、$N \in \{16, 256\}$ | 5 |
| 2 | 段階 1 に加え、実験 A のシード数を 3 にする | 3、$N \in \{16, 256\}$ | 5 |
| 3 | 段階 2 に加え、実験 B・C のシード数を 3 にする | 3、$N \in \{16, 256\}$ | 3 |

**学習で見る事例数の候補**: $E \in \{2^{19}, 2^{18}\}$(公比 2)。学習用の 46,208 枚に対するエポック数と、各条件のステップ数は次のとおり。

| $E$ | エポック数 | $T_{16}$ | $T_{64}$ | $T_{256}$ |
|---|---|---|---|---|
| $2^{19} = 524{,}288$ | 11.3 | 32,768 | 8,192 | 2,048 |
| $2^{18} = 262{,}144$ | 5.7 | 16,384 | 4,096 | 1,024 |

**実行計画**: 次の 12 通りに、優先順位の高い順に番号を付ける。計画 0〜7 は、較正の方式`"all"`で、$E$ の候補と段階 0〜3 を組み合わせ、
**$E$ の大きい順を第 1 キー、段階の番号の小さい順を第 2 キー** とする。計画 8〜11 は **下位の計画** で、$E = 2^{18}$・較正の方式
`"representative"`・段階 0〜3 である(停止した本番の実行を受けて追加した。6.2 節)。

| 計画 | $E$ | 段階 | 較正の方式 |
|---|---|---|---|
| 0〜3 | $2^{19}$ | 0〜3 | `"all"` |
| 4〜7 | $2^{18}$ | 0〜3 | `"all"` |
| 8〜11 | $2^{18}$ | 0〜3 | `"representative"` |

- **選択の規則**: 学習を始める前に(6.4 節、較正の前)、T4 上のスケーリングの計測(6.3 節)から、各計画の残りの実行時間(較正と本番の
  学習・評価)を見積もる。予算は「120 分 − データの準備・確認・計測にすでに使った実測時間(ノートブックの開始からの経過時間)」とし、
  予算内に収まる **番号の最も小さい計画** を選ぶ。**計画 0〜7(`"all"`)で収まる計画があれば、常にそれが優先される。** 計画 11 でも
  超える場合は、学習の前に例外で停止する。**選択は見積もりのみに基づき、どの実験の結果も参照しない。** 見積もりには、6.4 節で
  選んだ実行の方式の計測値を使う。
- **較正の方式を学習量 $E$ より優先する理由**(計画 8〜11 を最後に置く理由): 判定の中心は $N = 16$ の条件であり(対比量 $D$ は
  $\Delta_{16}$ と $\Delta_{256}$ の差)、パイロットでは $N = 16$ の最良の学習率が当初の規則から外れた(6.2 節の改訂 1)。そこで、
  $N = 16$ の学習率を本番と同じ規模の較正で確かめる価値を、学習量 $E$ の大きさより上に置き、`"all"`の計画を $E$ によらずすべて先にした。
  ただし、実行不能になって何も得られないよりは、規則で決めた学習率で実行するほうがよいので、`"representative"`の計画を下位に置いた。
  `"representative"`では、規則で決めた条件の P0 は検証されない(共通の前提条件の節の記述のとおり、較正した $N = 256$ の P0 で代える)。
- **$E$ を優先する理由**: 019 では、学習ステップ数が足りず、ViT が学習データにもほとんど当てはまっていなかった可能性が事後的に
  指摘された。学習が足りない状態では、損失や負例の違いが指標に現れる前に学習を打ち切ることになり、主張の対象(十分に学習した状態での
  差)とは別の量(学習の速さ)を測ることにつながる。シード数と $N = 64$ の水準は判定の **検出力** を決めるだけで、検出力の低下は
  判定不能を増やすが、誤った量を測ることにはならない。
- **段階の順序の理由**(同じ $E$ の中): $N = 64$ は判定に使わない診断の水準なので最初に削る。次に、最も重い実験 A($N = 16$ の学習は
  ステップ数が $N = 256$ の 16 倍で、1 回の学習が最も長い)のシード数を削る。実験 B・C は $N = 256$ のみで学習が軽く、削っても
  節約が小さいので最後に削る。
- **$E$ の下限 $2^{18}$ の根拠**: 6.2 節のパイロットで、$E = 2^{18}$ では softmax の $N = 16$・$N = 256$ の既知の組み合わせの検証集合の
  検索の正解率が P1 の閾値 0.1 を上回り、sigmoid の $N = 16$ も閾値を上回った(sigmoid は閾値を上回ったかどうかのみを確認した)。
  当初の候補にあった $2^{17}$ は、sigmoid の $N = 16$ が閾値を下回ったので候補から除いた(6.2 節)。

#### 実験 A: 負例数と損失関数の交互作用

**検証すること**: sigmoid 損失の softmax 損失(InfoNCE)に対する優位が、バッチサイズ $N$ が大きいほど縮む(SigLIP [4] の主張、3.5 節)。

**条件**: 損失 $\ell \in \{\mathrm{softmax}, \mathrm{sigmoid}\}$ × $N \in \{16, 64, 256\}$(公比 4 の等比の水準。段階 1 以降は $N \in \{16, 256\}$)。
見た事例数 $E$ を揃え、学習率は条件ごとに較正した値。$(\mathrm{softmax}, 256)$ は標準条件 S。

**対比量**: $M_{\ell, N, s}$ を損失 $\ell$・バッチサイズ $N$・シード $s$ の未見の組み合わせの検索の正解率とし、

$$
\Delta_{N,s} = M_{\mathrm{sigmoid}, N, s} - M_{\mathrm{softmax}, N, s}, \qquad
d_s = \Delta_{16,s} - \Delta_{256,s}, \qquad
D = \frac{1}{n_A} \sum_{s} d_s
$$

($n_A$ は実験 A のシード数)。$D > 0$ は「$N = 16$ での sigmoid の優位が $N = 256$ より大きい」(優位が $N$ とともに縮む)ことを表す。

**標準偏差の導出(誤差伝播)**: $d_s$ は、同じシード $s$ の 4 つの学習(2 損失 × 2 バッチサイズ、同じ初期値)の線形結合
$d_s = M_{\mathrm{sig},16,s} - M_{\mathrm{soft},16,s} - M_{\mathrm{sig},256,s} + M_{\mathrm{soft},256,s}$ である。4 つの測定の分散が同程度 $\sigma_M^2$ で
互いに独立なら $\mathrm{Var}(d_s) = 4 \sigma_M^2$ となり、単一の測定の 4 倍になる(差の差の誤差伝播)。同じシードの学習は初期値を共有するので
測定どうしに相関がありうるが、$d_s$ をシードごとに作ってその標本標準偏差をとれば、相関を含めた分散を直接推定できる。シード間は独立
なので

$$
\sigma_D = \mathrm{sd}(d_s) / \sqrt{n_A}
$$

($\mathrm{sd}$ は不偏標本標準偏差)。診断量として、相関を無視した $\sqrt{\sum_c \mathrm{sd}(M_c)^2 / n_A}$(4 条件 $c$ の和)も併記する。

**判定**: $D > 2\sigma_D$ なら支持、$D < -2\sigma_D$ なら反証、それ以外は判定不能。

**前提条件**: 対比量に使う 4 条件(softmax・sigmoid × $N \in \{16, 256\}$)の P0(較正した条件のみ)と P1。$N = 64$(段階 0 のみ)は判定に
使わない診断の水準なので、その P0・P1 は参考として印字するのみとする。

**作用点の記述**: 損失関数が直接作用するのは、バッチ内の類似度の logits(正例の組と負例の組の類似度の差)であり、$N$ が直接作用するのは
1 つの正例に対する負例の数である。その量に最も近いのは、**バッチ内の正例の類似度と最も類似度の高い負例の類似度の差(マージン)** で
ある。$M$ は、そこから「学習で見た組み合わせの外への汎化」と「候補 796 個の中での順位」の 2 段を経た下流の量だが、zero-shot の性能という
主張そのものなので対比量に選んだ。診断量として、学習中のバッチ内のマージン(末尾 10% のステップの平均。$N$ によって負例の数が違うので
$N$ の間では直接比べられない)と、未見の組み合わせの評価集合での固定の候補 796 個に対するマージン(正解の類似度 − 正解以外の候補の類似度の
最大値)の平均を $N$ ごとに記録する。$N = 64$ の結果(段階 0 のみ)は判定に使わず、図に示す。

**交絡と注意**: (1)更新の回数と負例の数(上の「学習量の揃え方」)。(2)3.7 節のとおり、合成のキャプションの空間が小さいので、$N$ が
大きいほどバッチに困難な負例が偶然含まれる確率が上がる($N = 256$ で、順序の入れ替えが約 17.7%、色の入れ替えが約 9.1%)。これは実画像でもバッチを大きくしたときに起きる現象の
強調された形である。(3)原論文のバッチサイズ(数千以上)とは規模が大きく違う。

#### 実験 B: 標準の対照学習における bag-of-words 化の存在

**モデル**: 標準条件 S(softmax・$N = 256$)の学習済みモデルを再利用する(追加の学習なし。シード数は実験 B・C のシード数 $n_{BC}$)。

**検証すること**: 語順・属性の結びつきが問われる 2 択の正解率 $\mathrm{acc}_{\mathrm{swap}}$ が、ランダムな負例との 2 択の正解率
$\mathrm{acc}_{\mathrm{rand}}$ より低い(3.7 節)。

**対比量**: 未見の組み合わせの評価集合で、同じ 2 択の形式での差 $b_s = \mathrm{acc}_{\mathrm{rand},s} - \mathrm{acc}_{\mathrm{swap},s}$ のシード平均
$B = \frac{1}{n_{BC}} \sum_s b_s$。困難な負例は 2 種類(色の入れ替え・順序の入れ替え)をまとめて $\mathrm{acc}_{\mathrm{swap}}$ とし、種類ごとの
正解率は診断量とする。

**標準偏差の導出**: $b_s$ は同じモデル(シード $s$)の 2 つの正解率の差で、シード間で独立なので $\sigma_B = \mathrm{sd}(b_s) / \sqrt{n_{BC}}$。
同じモデルの同じ画像で評価するので、モデルの良し悪しによる共通の変動は差で打ち消される。

**判定**: $B > 2\sigma_B$ なら支持、$B < -2\sigma_B$ なら反証、それ以外は判定不能。

**前提条件**: P0(softmax・256)、P1(softmax・256)。

**作用点の記述**: この実験の「介入」は評価時の負例の種類であり、2 択の判定そのものに直接作用する。対比量はその 2 択の正解率の差で、
作用点との間に他の段を挟まない。

**解釈の注意**: $\mathrm{acc}_{\mathrm{rand}}$ が 1 に近いと(天井)、$\mathrm{acc}_{\mathrm{swap}}$ との差の上限が抑えられる。差が小さい場合に、
bag-of-words 化がないのか天井のためかを区別するため、種類ごとの正解率と既知の組み合わせの検証集合での同じ量を診断量として併記する。

**既知性の交絡と対処**: 未見の組み合わせのキャプションの色の入れ替えは、約 88% が学習用のキャプションになる(5.3 節)。一方、ランダムな
負例をすべて未見の組み合わせから選ぶと、モデルが「学習で見たキャプションに高い類似度を出す」傾向を持つだけで、bag-of-words 化がなくても
$\mathrm{acc}_{\mathrm{swap}}$ が $\mathrm{acc}_{\mathrm{rand}}$ より下がり、$B$ が支持の方向に偏る。順序の入れ替えは常に未見のままなので、この問題は
ない。そこで、ランダムな負例を困難な負例と「既知 / 未見」の状態で対応させた(共通の設定の「2 択の正解率」)。診断量として、(1)旧定義
(すべて未見の組み合わせから選ぶ)の $\mathrm{acc}_{\mathrm{rand}}$、(2)既知性の偏りの大きさ(未見の正例と一様に選んだ学習用のキャプションの
2 択の正解率から、未見の正例と未見の組み合わせのキャプションの 2 択の正解率を引いたもの。負の値は学習用のキャプションに引き寄せられる
傾向を表す)を併記する(判定なし)。

#### 実験 C: 困難な負例による改善(NegCLIP)

**検証すること**: 学習時に困難な負例のキャプションを画像 → テキスト方向の分母に加えると(NegCLIP、3.7 節)、$\mathrm{acc}_{\mathrm{swap}}$ が上がる。

**条件**: NegCLIP・$N = 256$・見た事例数 $E$ は S と同一・学習率は S と同一(上の較正の節)。各正例に色の入れ替えと順序の入れ替えの
2 つの困難な負例のキャプションを作り、画像 → テキストの分母に加える(正例と同じキャプションになった列は、その行の分母から除く)。
**除外した (色, 形) の組を含む困難な負例は使わない**(テキスト encoder に入れず、分母にも加えない。3.7 節)。これを入れると、テキスト
encoder が学習中に除外した組を見ることになり、未見の組み合わせによる評価の前提が崩れるためである。色の入れ替えの約 48% が該当し、
分母に加わる困難な負例は、バッチ全体で平均約 $1.52N$ 個になる(順序の入れ替え $N$ 個 + 色の入れ替え約 $0.52N$ 個)。**困難な負例の画像は加えない**(原論文との差)。同じシードの NegCLIP と S は、同じ初期値から始まり同じ順序で同じシーンを
見る(6.11 節でアサーション)。

**対比量**: $c_s = \mathrm{acc}_{\mathrm{swap}}(\mathrm{NegCLIP}, s) - \mathrm{acc}_{\mathrm{swap}}(\mathrm{S}, s)$ のシード平均 $C = \frac{1}{n_{BC}} \sum_s c_s$
(未見の組み合わせの評価集合)。

**標準偏差の導出**: $c_s$ は同じシードの対応のある 2 つの学習の差で、シード間で独立なので $\sigma_C = \mathrm{sd}(c_s) / \sqrt{n_{BC}}$。

**判定**: $C > 2\sigma_C$ なら支持、$C < -2\sigma_C$ なら反証、それ以外は判定不能。

**前提条件**: P0(softmax・256)、P1(softmax・256)、P1(NegCLIP)。

**作用点の記述**: NegCLIP が直接作用するのは、学習中の画像と困難な負例のキャプションの類似度(分母に入った項)であり、それに最も近いのは
**学習で見た組み合わせでの** 困難な負例との 2 択の正解率(既知の組み合わせの検証集合の $\mathrm{acc}_{\mathrm{swap}}$)である。対比量は、それを
未見の組み合わせへ汎化させた 1 段下流の量である。既知の組み合わせの検証集合での同じ差を診断量として併記する。

**診断量(代償)**: 困難な負例を加えることで、ランダムな負例に対する zero-shot の検索が悪くならないかを見るため、$M$ の差
$M(\mathrm{NegCLIP}) - M(\mathrm{S})$ と $\mathrm{acc}_{\mathrm{rand}}$ の差を記録する(判定なし)。

**本文に明記する限界**: 学習と評価で困難な負例の **生成規則が同じ** である(色の入れ替えと順序の入れ替え)。未見の組み合わせで評価する
ことで、「学習で見たキャプションそのものを覚えた」可能性は除けるが、「この 2 つの規則で作られる負例を見分ける」ことに特化した学習の
可能性は除けない。別の規則の負例(例えば形の入れ替えや関係語の置き換え)への汎化は検証しない。

#### 観察: modality gap(判定基準を設けない)

**判定基準を設けない観察である。** 実験 A・C の学習済みモデルについて、未見の組み合わせの評価集合の画像の埋め込みと、その正例の
キャプションの埋め込みの重心間距離 $\Delta_{\mathrm{gap}}$(3.8 節)を、学習の前(初期化の時点)と後で記録する。シード 0 のモデルについて、
画像とテキストの埋め込み(各 400 個)を主成分分析で 2 次元に射影した図を示す。



## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/05_vision_language/020_clip_contrastive_learning.ipynb
