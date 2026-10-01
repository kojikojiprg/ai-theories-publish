---
title: "CLIP と対照学習 / CLIP and Contrastive Learning(実装・実験編 4/5)"
---

この記事は後編(実装・実験編 4/5)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/020_clip_contrastive_learning-practice-3)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/020_clip_contrastive_learning-practice-5)。

### 6.7 実験 A: 負例数と損失関数の交互作用


```python
A_M = {
    (loss, n): np.array([zero_shot_accuracy(key_of(loss, n, s)) for s in SEEDS_A]) for loss in A_LOSSES for n in A_BATCH_SIZES
}
A_DELTA = {n: A_M[("sigmoid", n)] - A_M[("softmax", n)] for n in A_BATCH_SIZES}
_n_small, _n_large = min(A_BATCH_SIZES), max(A_BATCH_SIZES)
assert (_n_small, _n_large) == (16, 256)
CONTRAST_A = paired_contrast(A_DELTA[_n_small] - A_DELTA[_n_large])
verdict_A = judge(CONTRAST_A["value"], CONTRAST_A["sigma"])
_naive_sigma = math.sqrt(sum(A_M[(loss, n)].var(ddof=1) for loss in A_LOSSES for n in (_n_small, _n_large)) / len(SEEDS_A))
for _loss in A_LOSSES:
    for _n in A_BATCH_SIZES:
        precondition_status[f"P1({_loss},{_n})"] = seen_accuracy_mean(_loss, _n, SEEDS_A) >= SEEN_RETRIEVAL_MIN
# 前提条件は対比量 D に使う 4 条件(N = 16・256)のみ。N = 64(段階 0 のみ、診断)の P0・P1 は参考として印字する
A_CONTRAST_CONDITIONS = [(loss, n) for n in (_n_small, _n_large) for loss in A_LOSSES]
A_PRECONDITIONS = [f"P0({c[0]},{c[1]})" for c in CALIBRATION_CONDITIONS if c in A_CONTRAST_CONDITIONS] + [
    f"P1({c[0]},{c[1]})" for c in A_CONTRAST_CONDITIONS
]
_a_reference = [p for p in precondition_status if p.startswith(("P0(", "P1(")) and ",64)" in p]

print(f"{_tag}未見の組み合わせの検索の正解率 M(候補 {len(UNSEEN_CAPTION_IDS)} 個、チャンス水準 {1 / len(UNSEEN_CAPTION_IDS):.4f})、シード {SEEDS_A}")
for _n in A_BATCH_SIZES:
    for _loss in A_LOSSES:
        _keys = [key_of(_loss, _n, s) for s in SEEDS_A]
        _eval_margin = np.mean([final_of(k)["unseen_retrieval"]["mean_margin"] for k in _keys])
        _train_margin = np.mean([RECORDS[k]["train_margin_tail"] for k in _keys])
        _scale = np.mean([math.exp(RECORDS[k]["history"]["logit_scale"][-1]) for k in _keys])
        print(
            f"{_tag}  N = {_n:3d} {_loss:7s}: M {np.round(A_M[(_loss, _n)], 4).tolist()} 平均 {A_M[(_loss, _n)].mean():.4f}、"
            f"既知の検証 {seen_accuracy_mean(_loss, _n, SEEDS_A):.4f}、評価集合のマージン {_eval_margin:+.4f}、"
            f"学習中のバッチ内のマージン(末尾 10%){_train_margin:+.4f}、最終の倍率 exp(t) {_scale:.1f}"
        )
    _c = paired_contrast(A_DELTA[_n])
    print(f"{_tag}  N = {_n:3d}: Delta_N = M_sigmoid - M_softmax = {_c['value']:+.4f}(sd/sqrt(n) {_c['sigma']:.4f}、シードごと {np.round(_c['per_seed'], 4).tolist()})")
print(f"{_tag}d_s = Delta_16 - Delta_256: {np.round(CONTRAST_A['per_seed'], 4).tolist()}")
print(
    f"{_tag}対比量 D = {CONTRAST_A['value']:+.4f}、sigma_D = sd(d_s)/sqrt({len(SEEDS_A)}) = {CONTRAST_A['sigma']:.4f}、"
    f"閾値 2 sigma_D = {2 * CONTRAST_A['sigma']:.4f} -> 判定関数の結果: {verdict_A}"
)
print(f"{_tag}診断量: 相関を無視した標準偏差 sqrt(sum sd(M_c)^2 / n) = {_naive_sigma:.4f}")
print(f"{_tag}前提条件: " + "、".join(f"{p} = {precondition_status.get(p)}" for p in A_PRECONDITIONS) + f"(P1 の閾値 {SEEN_RETRIEVAL_MIN})")
if _a_reference:
    print(f"{_tag}参考(N = 64、判定に使わない): " + "、".join(f"{p} = {precondition_status[p]}" for p in _a_reference))

_fig, _axes = plt.subplots(1, 3, figsize=(16, 4.2))
_x = np.log2(A_BATCH_SIZES)
for _i, _loss in enumerate(A_LOSSES):
    _values = np.array([A_M[(_loss, n)] for n in A_BATCH_SIZES])  # (水準, シード)
    for _s in range(len(SEEDS_A)):
        _axes[0].plot(_x, _values[:, _s], color=f"C{_i}", alpha=0.3)
    _axes[0].plot(_x, _values.mean(axis=1), marker="o", color=f"C{_i}", lw=2, label=_loss)
    _margins = [np.mean([RECORDS[key_of(_loss, n, s)]["train_margin_tail"] for s in SEEDS_A]) for n in A_BATCH_SIZES]
    _axes[2].plot(_x, _margins, marker="o", color=f"C{_i}", label=f"{_loss} (in-batch, train)")
    _eval_margins = [np.mean([final_of(key_of(_loss, n, s))["unseen_retrieval"]["mean_margin"] for s in SEEDS_A]) for n in A_BATCH_SIZES]
    _axes[2].plot(_x, _eval_margins, marker="s", ls="--", color=f"C{_i}", label=f"{_loss} (unseen, 796 candidates)")
_axes[0].set_ylabel("zero-shot retrieval accuracy M (unseen)")
_axes[0].set_title(f"{_plot_tag}Experiment A (thin = seeds)")
_axes[0].legend()
_deltas = np.array([A_DELTA[n] for n in A_BATCH_SIZES])
_axes[1].errorbar(_x, _deltas.mean(axis=1), yerr=_deltas.std(axis=1, ddof=1) / math.sqrt(len(SEEDS_A)), marker="o", capsize=4, color="black")
_axes[1].axhline(0, color="gray", ls="--")
_axes[1].set_ylabel("Delta_N = M_sigmoid - M_softmax (mean +- sd/sqrt(n))")
_axes[1].set_title("sigmoid advantage by batch size")
_axes[2].set_ylabel("margin (positive - hardest negative)")
_axes[2].set_title("margin diagnostics")
_axes[2].legend(fontsize=7)
for _ax in _axes:
    _ax.set_xticks(_x, [f"N={n}" for n in A_BATCH_SIZES])
plt.tight_layout()
plt.show()

_fig, _axes = plt.subplots(1, len(A_BATCH_SIZES), figsize=(5 * len(A_BATCH_SIZES), 3.6), squeeze=False)
for _j, _n in enumerate(A_BATCH_SIZES):
    _t = steps_for(EXAMPLES, _n)
    _steps = list(intermediate_eval_steps_for(_t)) + [_t]
    for _i, _loss in enumerate(A_LOSSES):
        _curves = np.array(
            [[RECORDS[key_of(_loss, _n, s)]["history"]["evaluations"][t]["seen_accuracy"] for t in _steps[:-1]] + [RECORDS[key_of(_loss, _n, s)]["seen"]["accuracy"]] for s in SEEDS_A]
        )
        _axes[0, _j].plot(np.array(_steps) * _n, _curves.mean(axis=0), marker="o", color=f"C{_i}", label=_loss)
    _axes[0, _j].set_xlabel("examples seen")
    _axes[0, _j].set_ylabel("seen-combination retrieval (seed mean)")
    _axes[0, _j].set_title(f"N = {_n}")
    _axes[0, _j].legend()
plt.tight_layout()
plt.show()
```

    未見の組み合わせの検索の正解率 M(候補 796 個、チャンス水準 0.0013)、シード (0, 1, 2)
      N =  16 softmax: M [0.4256, 0.4256, 0.4124] 平均 0.4212、既知の検証 0.3487、評価集合のマージン -0.0043、学習中のバッチ内のマージン(末尾 10%)+0.3835、最終の倍率 exp(t) 32.4
      N =  16 sigmoid: M [0.3486, 0.3379, 0.3043] 平均 0.3303、既知の検証 0.2447、評価集合のマージン -0.0089、学習中のバッチ内のマージン(末尾 10%)+0.4585、最終の倍率 exp(t) 13.6
      N =  16: Delta_N = M_sigmoid - M_softmax = -0.0909(sd/sqrt(n) 0.0091、シードごと [-0.0769, -0.0876, -0.108])
      N = 256 softmax: M [0.7563, 0.7054, 0.7365] 平均 0.7327、既知の検証 0.7889、評価集合のマージン +0.0466、学習中のバッチ内のマージン(末尾 10%)+0.1649、最終の倍率 exp(t) 33.6
      N = 256 sigmoid: M [0.5273, 0.5377, 0.5788] 平均 0.5479、既知の検証 0.5476、評価集合のマージン +0.0162、学習中のバッチ内のマージン(末尾 10%)+0.1499、最終の倍率 exp(t) 15.9
      N = 256: Delta_N = M_sigmoid - M_softmax = -0.1848(sd/sqrt(n) 0.0223、シードごと [-0.229, -0.1677, -0.1577])
    d_s = Delta_16 - Delta_256: [0.152, 0.0801, 0.0496]
    対比量 D = +0.0939、sigma_D = sd(d_s)/sqrt(3) = 0.0304、閾値 2 sigma_D = 0.0607 -> 判定関数の結果: 支持
    診断量: 相関を無視した標準偏差 sqrt(sum sd(M_c)^2 / n) = 0.0258
    前提条件: P0(softmax,256) = True、P0(sigmoid,256) = True、P1(softmax,16) = True、P1(sigmoid,16) = True、P1(softmax,256) = True、P1(sigmoid,256) = True(P1 の閾値 0.1)



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/020_clip_contrastive_learning/output_36_1.png)
    



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/020_clip_contrastive_learning/output_36_2.png)
    


### 6.8 実験 B: 標準の対照学習における bag-of-words 化の存在


```python
_b_keys = [key_of("softmax", STANDARD_BATCH_SIZE, s) for s in SEEDS_BC]
_two = {name: np.array([final_of(k)["unseen_two_alternative"][name] for k in _b_keys]) for name in ("random", "swap", "attribute_swap", "order_swap")}
_two_seen = {name: np.array([final_of(k)["seen_two_alternative"][name] for k in _b_keys]) for name in ("swap", "attribute_swap", "order_swap")}
CONTRAST_B = paired_contrast(_two["random"] - _two["swap"])
verdict_B = judge(CONTRAST_B["value"], CONTRAST_B["sigma"])
precondition_status[f"P1(softmax,{STANDARD_BATCH_SIZE},BC)"] = seen_accuracy_mean("softmax", STANDARD_BATCH_SIZE, SEEDS_BC) >= SEEN_RETRIEVAL_MIN
B_PRECONDITIONS = [f"P0(softmax,{STANDARD_BATCH_SIZE})", f"P1(softmax,{STANDARD_BATCH_SIZE},BC)"]

_counts = {name: final_of(_b_keys[0])["unseen_two_alternative"][f"{name}_count"] for name in ("random", "swap")}
print(f"{_tag}標準条件 S(softmax・N = {STANDARD_BATCH_SIZE})、シード {SEEDS_BC}、2 択の試行の数 {_counts}")
for _name, _values in _two.items():
    print(f"{_tag}  未見の組み合わせ acc_{_name}: {np.round(_values, 4).tolist()} 平均 {_values.mean():.4f}")
for _name, _values in _two_seen.items():
    print(f"{_tag}  既知の組み合わせ(診断量)acc_{_name}: 平均 {_values.mean():.4f}")
_old = np.array([final_of(k)["unseen_two_alternative"]["random_old"] for k in _b_keys])
_familiarity = np.array(
    [final_of(k)["unseen_two_alternative"]["random_train"] - final_of(k)["unseen_two_alternative"]["random_unseen"] for k in _b_keys]
)
print(f"{_tag}  診断量: 旧定義(すべて未見の組み合わせから)の acc_rand 平均 {_old.mean():.4f}(新定義との差 {(_old - _two['random']).mean():+.4f})")
_fam = paired_contrast(_familiarity)
print(
    f"{_tag}  診断量: 既知性の偏り = acc(未見の正例 vs 学習用のキャプション) - acc(未見の正例 vs 未見の組み合わせのキャプション) "
    f"= {_fam['value']:+.4f}(sd/sqrt(n) {_fam['sigma']:.4f}、シードごと {np.round(_familiarity, 4).tolist()})"
)
print(f"{_tag}b_s = acc_rand - acc_swap: {np.round(CONTRAST_B['per_seed'], 4).tolist()}")
print(
    f"{_tag}対比量 B = {CONTRAST_B['value']:+.4f}、sigma_B = sd(b_s)/sqrt({len(SEEDS_BC)}) = {CONTRAST_B['sigma']:.4f}、"
    f"閾値 {2 * CONTRAST_B['sigma']:.4f} -> 判定関数の結果: {verdict_B}"
)
print(f"{_tag}前提条件: " + "、".join(f"{p} = {precondition_status.get(p)}" for p in B_PRECONDITIONS))
assert _counts["random"] == _counts["swap"] == 2 * len(UNSEEN_IMAGES)

_fig, _ax = plt.subplots(figsize=(7, 4))
_names = ("random", "attribute_swap", "order_swap", "swap")
_ax.bar(range(4), [_two[n].mean() for n in _names], yerr=[_two[n].std(ddof=1) / math.sqrt(len(SEEDS_BC)) for n in _names], capsize=4, color=["gray", "C1", "C2", "C3"])
for _s in range(len(SEEDS_BC)):
    _ax.plot(range(4), [_two[n][_s] for n in _names], color="black", alpha=0.3, marker=".")
_ax.axhline(0.5, color="gray", ls="--", label="chance (2AFC)")
_ax.set_xticks(range(4), _names)
_ax.set_ylabel("two-alternative accuracy (unseen)")
_ax.set_title(f"{_plot_tag}Experiment B: softmax N={STANDARD_BATCH_SIZE} (dots = seeds)")
_ax.legend()
plt.tight_layout()
plt.show()
```

    標準条件 S(softmax・N = 256)、シード (0, 1, 2)、2 択の試行の数 {'random': 6368, 'swap': 6368}
      未見の組み合わせ acc_random: [0.9989, 0.9986, 0.9997] 平均 0.9991
      未見の組み合わせ acc_swap: [0.9991, 0.9983, 0.9989] 平均 0.9987
      未見の組み合わせ acc_attribute_swap: [0.9997, 0.9981, 0.9991] 平均 0.9990
      未見の組み合わせ acc_order_swap: [0.9984, 0.9984, 0.9987] 平均 0.9985
      既知の組み合わせ(診断量)acc_swap: 平均 1.0000
      既知の組み合わせ(診断量)acc_attribute_swap: 平均 1.0000
      既知の組み合わせ(診断量)acc_order_swap: 平均 1.0000
      診断量: 旧定義(すべて未見の組み合わせから)の acc_rand 平均 0.9992(新定義との差 +0.0001)
      診断量: 既知性の偏り = acc(未見の正例 vs 学習用のキャプション) - acc(未見の正例 vs 未見の組み合わせのキャプション) = +0.0006(sd/sqrt(n) 0.0004、シードごと [0.0006, 0.0013, 0.0])
    b_s = acc_rand - acc_swap: [-0.0002, 0.0003, 0.0008]
    対比量 B = +0.0003、sigma_B = sd(b_s)/sqrt(3) = 0.0003、閾値 0.0005 -> 判定関数の結果: 判定不能
    前提条件: P0(softmax,256) = True、P1(softmax,256,BC) = True



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/020_clip_contrastive_learning/output_38_1.png)
    


### 6.9 実験 C: 困難な負例による改善(NegCLIP)


```python
_std_keys = [key_of("softmax", STANDARD_BATCH_SIZE, s) for s in SEEDS_BC]
_neg_keys = [key_of("negclip", STANDARD_BATCH_SIZE, s) for s in SEEDS_BC]


def _collect(keys, group, name):
    return np.array([final_of(k)[group][name] for k in keys])


C_SWAP = {"S": _collect(_std_keys, "unseen_two_alternative", "swap"), "NegCLIP": _collect(_neg_keys, "unseen_two_alternative", "swap")}
CONTRAST_C = paired_contrast(C_SWAP["NegCLIP"] - C_SWAP["S"])
verdict_C = judge(CONTRAST_C["value"], CONTRAST_C["sigma"])
precondition_status[f"P1(negclip,{STANDARD_BATCH_SIZE})"] = seen_accuracy_mean("negclip", STANDARD_BATCH_SIZE, SEEDS_BC) >= SEEN_RETRIEVAL_MIN
C_PRECONDITIONS = B_PRECONDITIONS + [f"P1(negclip,{STANDARD_BATCH_SIZE})"]
_diag = {
    "既知の組み合わせの acc_swap(作用点に近い量)": (_collect(_neg_keys, "seen_two_alternative", "swap"), _collect(_std_keys, "seen_two_alternative", "swap")),
    "未見の acc_attribute_swap": (_collect(_neg_keys, "unseen_two_alternative", "attribute_swap"), _collect(_std_keys, "unseen_two_alternative", "attribute_swap")),
    "未見の acc_order_swap": (_collect(_neg_keys, "unseen_two_alternative", "order_swap"), _collect(_std_keys, "unseen_two_alternative", "order_swap")),
    "未見の acc_rand(代償)": (_collect(_neg_keys, "unseen_two_alternative", "random"), _collect(_std_keys, "unseen_two_alternative", "random")),
    "M(代償)": (np.array([zero_shot_accuracy(k) for k in _neg_keys]), np.array([zero_shot_accuracy(k) for k in _std_keys])),
}

print(f"{_tag}NegCLIP と標準条件 S(N = {STANDARD_BATCH_SIZE}、同じ E・同じ学習率 {learning_rate_for(_std_keys[0]):.3g})、シード {SEEDS_BC}")
for _name, _values in C_SWAP.items():
    print(f"{_tag}  {_name}: 未見の組み合わせ acc_swap {np.round(_values, 4).tolist()} 平均 {_values.mean():.4f}")
print(f"{_tag}c_s = acc_swap(NegCLIP) - acc_swap(S): {np.round(CONTRAST_C['per_seed'], 4).tolist()}")
print(
    f"{_tag}対比量 C = {CONTRAST_C['value']:+.4f}、sigma_C = sd(c_s)/sqrt({len(SEEDS_BC)}) = {CONTRAST_C['sigma']:.4f}、"
    f"閾値 {2 * CONTRAST_C['sigma']:.4f} -> 判定関数の結果: {verdict_C}"
)
print(f"{_tag}診断量(判定なし、NegCLIP - S のシード平均と sd/sqrt(n)):")
for _name, (_neg, _std) in _diag.items():
    _c = paired_contrast(_neg - _std)
    print(f"{_tag}  {_name}: NegCLIP {_neg.mean():.4f}・S {_std.mean():.4f}、差 {_c['value']:+.4f}({_c['sigma']:.4f})")
print(f"{_tag}前提条件: " + "、".join(f"{p} = {precondition_status.get(p)}" for p in C_PRECONDITIONS))

_fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
for _j, (_title, _pair) in enumerate((("acc_swap (unseen)", (C_SWAP["S"], C_SWAP["NegCLIP"])), ("M (unseen, cost)", (_diag["M(代償)"][1], _diag["M(代償)"][0])))):
    for _s in range(len(SEEDS_BC)):
        _axes[_j].plot([0, 1], [_pair[0][_s], _pair[1][_s]], color="gray", marker="o", alpha=0.6)
    _axes[_j].plot([0, 1], [_pair[0].mean(), _pair[1].mean()], color="black", marker="s", lw=2)
    _axes[_j].set_xticks([0, 1], ["S (softmax)", "NegCLIP"])
    _axes[_j].set_title(f"{_plot_tag}{_title} (lines = seeds)")
plt.tight_layout()
plt.show()
```

    NegCLIP と標準条件 S(N = 256、同じ E・同じ学習率 0.002)、シード (0, 1, 2)
      S: 未見の組み合わせ acc_swap [0.9991, 0.9983, 0.9989] 平均 0.9987
      NegCLIP: 未見の組み合わせ acc_swap [0.9987, 0.9994, 0.9991] 平均 0.9991
    c_s = acc_swap(NegCLIP) - acc_swap(S): [-0.0003, 0.0011, 0.0002]
    対比量 C = +0.0003、sigma_C = sd(c_s)/sqrt(3) = 0.0004、閾値 0.0008 -> 判定関数の結果: 判定不能
    診断量(判定なし、NegCLIP - S のシード平均と sd/sqrt(n)):
      既知の組み合わせの acc_swap(作用点に近い量): NegCLIP 0.9999・S 1.0000、差 -0.0001(0.0001)
      未見の acc_attribute_swap: NegCLIP 0.9992・S 0.9990、差 +0.0002(0.0006)
      未見の acc_order_swap: NegCLIP 0.9990・S 0.9985、差 +0.0004(0.0003)
      未見の acc_rand(代償): NegCLIP 0.9993・S 0.9991、差 +0.0003(0.0004)
      M(代償): NegCLIP 0.7622・S 0.7327、差 +0.0295(0.0078)
    前提条件: P0(softmax,256) = True、P1(softmax,256,BC) = True、P1(negclip,256) = True



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/020_clip_contrastive_learning/output_40_1.png)
    


### 6.10 観察: modality gap(判定基準を設けない)

**判定基準を設けない観察である。** 未見の組み合わせの評価集合で、画像の埋め込みの重心とキャプションの埋め込みの重心の距離
$\Delta_{\mathrm{gap}}$(3.8 節)を、学習の前(初期化の時点)と後で比べる。シード 0 のモデルについて、画像とテキストの埋め込み(各 400 個)を
合わせて主成分分析で 2 次元に射影した図を示す。単位球面上の埋め込みの重心間の距離は、最大で 2 である。


```python
_gap_keys = sorted({(k[0], k[1]) for k in RECORDS}, key=lambda c: (c[1], c[0]))
print(f"{_tag}modality gap(未見の組み合わせの評価集合、シード平均): 条件 | 学習前 | 学習後(未見)| 学習後(既知)")
for _c in _gap_keys:
    _keys = [k for k in RECORDS if (k[0], k[1]) == _c]
    print(
        f"{_tag}  {_c[0]:7s} N={_c[1]:3d}: {np.mean([RECORDS[k]['gap_unseen_at_init'] for k in _keys]):.3f} | "
        f"{np.mean([final_of(k)['gap_unseen'] for k in _keys]):.3f} | {np.mean([final_of(k)['gap_seen'] for k in _keys]):.3f}"
        f"({len(_keys)} シード)"
    )
_proj_conditions = [c for c in _gap_keys if key_of(*c, 0) in RECORDS]
_fig, _axes = plt.subplots(1, len(_proj_conditions), figsize=(3.6 * len(_proj_conditions), 3.6), squeeze=False)
for _j, _c in enumerate(_proj_conditions):
    _p = final_of(key_of(*_c, 0))["projection"]
    _all = np.concatenate([_p["image"], _p["text"]])
    _centered = _all - _all.mean(axis=0)
    _, _, _vt = np.linalg.svd(_centered, full_matrices=False)
    _xy = _centered @ _vt[:2].T
    _k = len(_p["image"])
    _axes[0, _j].scatter(_xy[:_k, 0], _xy[:_k, 1], s=4, alpha=0.5, label="image")
    _axes[0, _j].scatter(_xy[_k:, 0], _xy[_k:, 1], s=4, alpha=0.5, label="text")
    _axes[0, _j].set_title(f"{_plot_tag}{_c[0]} N={_c[1]} (seed 0)", fontsize=9)
    _axes[0, _j].legend(fontsize=7)
plt.tight_layout()
plt.show()
```

    modality gap(未見の組み合わせの評価集合、シード平均): 条件 | 学習前 | 学習後(未見)| 学習後(既知)
      sigmoid N= 16: 1.369 | 0.233 | 0.222(3 シード)
      softmax N= 16: 1.369 | 0.142 | 0.129(3 シード)
      negclip N=256: 1.369 | 0.158 | 0.100(3 シード)
      sigmoid N=256: 1.369 | 0.672 | 0.674(3 シード)
      softmax N=256: 1.369 | 0.148 | 0.096(3 シード)



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/020_clip_contrastive_learning/output_42_1.png)
    


### 6.11 不変条件のアサーションと`SMOKE_TEST`の配線

- すべての学習(較正を含む)で、見た事例数が $E$、ステップ数が $E/N$、学習率のスケジュールの形(最大値で割った列)が同じ $N$ の間で
  同一、すべてのバッチでキャプションが互いに異なる。
- 同じシードの学習は、損失・$N$ によらず同じ初期値(温度とバイアスを除く)から始まった。同じシード・同じ $N$ の学習は、損失によらず
  同じ順序で同じシーンを見た(バッチの添字の列のハッシュが一致)。
- 本番の学習はすべて同じ未見の組み合わせの評価集合(3,184 枚、候補 796 個)で評価した(画像のテンソルが読み込み直後と一致、2 択の試行の数が
  一致)。較正の学習は未見の組み合わせの評価集合を評価していない。
- 標準条件 S の記録を、実験 A の $N = 256$ の softmax・実験 B・実験 C の標準のモデルで共有した(同じ鍵)。
- **`SMOKE_TEST`の配線**: 5.2 節・6.4 節で決まった実効値(水準・計画・$E$・段階・シード数・$N$ の水準)が計画の表どおりであり、実際に
  使われた値(記録の鍵・学習の記録の長さ)と一致する。


```python
_all_records = list(RECORDS.values()) + [r for c in CALIBRATION.values() for r in c["records"].values()]
_shapes: dict[int, np.ndarray] = {}
for _r in _all_records:
    _h, _n = _r["history"], _r["key"][1]
    assert _r["examples"] == EXAMPLES and _h["examples_seen"] == EXAMPLES
    assert _r["train_steps"] == steps_for(EXAMPLES, _n) == len(_h["loss"]) == len(_h["learning_rate"])
    assert set(_h["evaluations"]) == set(intermediate_eval_steps_for(_r["train_steps"]))
    assert _h["distinct_captions_in_every_batch"]
    _normalized = np.array(_h["learning_rate"]) / _r["learning_rate"]
    _shapes.setdefault(_n, _normalized)
    assert np.allclose(_normalized, _shapes[_n], rtol=1e-12, atol=0), "学習率のスケジュールの形が異なる"
_init_by_seed: dict[int, set] = {}
_stream_by_seed_n: dict[tuple, set] = {}
for _r in _all_records:
    _init_by_seed.setdefault(_r["key"][2], set()).add(_r["initial_state_sha256"])
    _stream_by_seed_n.setdefault((_r["key"][2], _r["key"][1]), set()).add(_r["history"]["data_stream_hash"])
assert all(len(v) == 1 for v in _init_by_seed.values()), "同じシードで初期値が異なる"
assert all(len(v) == 1 for v in _stream_by_seed_n.values()), "同じシード・同じ N でバッチの順序が異なる"
for _r in _all_records:  # テキスト encoder に入れた系列(正例と困難な負例)は、除外した組を含まない
    assert TRAIN_CAPTION_MASK[_r["history"]["text_caption_ids"]].all(), _r["key"]
_negclip_records = [r for r in _all_records if r["key"][0] == "negclip"]
_attr_used = sum(r["history"]["hard_negatives_used"]["attribute_swap"] for r in _negclip_records)
_attr_generated = sum(r["history"]["hard_negatives_generated"]["attribute_swap"] for r in _negclip_records)
for _r in RECORDS.values():
    _f = _r["final"]
    assert _r["purpose"] == "main"
    assert _f["unseen_retrieval"]["num_candidates"] == len(UNSEEN_CAPTION_IDS) and _f["unseen_retrieval"]["num_images"] == len(UNSEEN_IMAGES)
    assert _f["unseen_two_alternative"]["swap_count"] == _f["unseen_two_alternative"]["random_count"] == 2 * len(UNSEEN_IMAGES)
for _c in CALIBRATION.values():
    assert all(r["purpose"] == "calibration" and "final" not in r for r in _c["records"].values())
assert sha256_of_tensor(UNSEEN_IMAGES) == UNSEEN_IMAGES_SHA256
for _s in set(SEEDS_A) | set(SEEDS_BC):  # S は 1 つの記録を共有する
    assert key_of("softmax", STANDARD_BATCH_SIZE, _s) in RECORDS
print(
    f"不変条件: 全 {len(_all_records)} 学習(本番 {len(RECORDS)}・較正 {len(_all_records) - len(RECORDS)})で見た事例数 E = {EXAMPLES:,}・"
    f"ステップ数 E/N・学習率のスケジュールの形(同じ N の間)が同一、全バッチでキャプションが互いに異なる、同じシードの初期値が同一"
    f"({len(_init_by_seed)} シード)、同じシード・同じ N のバッチの順序が損失によらず同一({len(_stream_by_seed_n)} 組)、"
    f"本番はすべて同じ評価集合(候補 {len(UNSEEN_CAPTION_IDS)} 個・{len(UNSEEN_IMAGES):,} 枚)で評価、較正は評価集合を評価していない、S を共有、"
    f"全学習でテキスト encoder に入れた系列が除外した組を含まない: OK。NegCLIP で分母に加えた色の入れ替えは規則が作った数の "
    f"{_attr_used / _attr_generated:.3f}(順序の入れ替えはすべて)"
)

_table = PLANS[CURRENT_LEVEL_NAME][SELECTED_PLAN]
assert _table["plan"] == SELECTED_PLAN and _table["E"] == EXAMPLES and _table["stage"] == SELECTED_STAGE
if SELECTED_PLAN < 8:  # 計画 0〜7: E の候補 x 段階 0〜3、較正の方式 "all"
    assert EXAMPLES == EXAMPLE_CANDIDATES[CURRENT_LEVEL_NAME][SELECTED_PLAN // 4] and _table["calibration_mode"] == "all"
else:  # 計画 8〜11: E の小さい方の候補、較正の方式 "representative"
    assert EXAMPLES == min(EXAMPLE_CANDIDATES[CURRENT_LEVEL_NAME]) and _table["calibration_mode"] == "representative"
assert SELECTED_STAGE == SELECTED_PLAN % 4 and CALIBRATION_MODE == _table["calibration_mode"]
_stage_table = STAGES[CURRENT_LEVEL_NAME][_table["stage"]]
_effective = {
    "level": CURRENT_LEVEL_NAME,
    "plan": SELECTED_PLAN,
    "stage": SELECTED_STAGE,
    "examples": EXAMPLES,
    "seeds_a": _stage_table["SEEDS_A"],
    "seeds_bc": _stage_table["SEEDS_BC"],
    "a_batch_sizes": list(_stage_table["A_BATCH_SIZES"]),
    "calibration_mode": _table["calibration_mode"],
    "calibrated_conditions": len(calibration_conditions(_stage_table, _table["calibration_mode"])),
    "execution_mode": EXECUTION_MODE,
    "runs": len(plan_runs(_stage_table)),
}
_examples_used = {r["history"]["examples_seen"] for r in _all_records}
assert len(_examples_used) == 1
_used = {
    "level": CURRENT_LEVEL_NAME,
    "plan": SELECTED_PLAN,
    "stage": SELECTED_STAGE,
    "examples": _examples_used.pop(),
    "seeds_a": len({k[2] for k in RECORDS if k[0] == "sigmoid"}),
    "seeds_bc": len({k[2] for k in RECORDS if k[0] == "negclip"}),
    "a_batch_sizes": sorted({k[1] for k in RECORDS if k[0] == "sigmoid"}),
    "calibration_mode": "all" if {(k[0], k[1]) for k in RECORDS if k[0] != "negclip"} <= set(CALIBRATION) else "representative",
    "calibrated_conditions": len(CALIBRATION),
    "execution_mode": ({r["execution_mode"] for r in _all_records} or {None}).pop(),
    "runs": len(RECORDS),
}
assert len({r["execution_mode"] for r in _all_records}) == 1  # 全学習で同じ実行の方式
assert _used == _effective, (_used, _effective)
assert (CURRENT_LEVEL_NAME == "smoke") == SMOKE_TEST
print(f"SMOKE_TEST={SMOKE_TEST} の配線: 実効値 {json.dumps(_effective)} と実際に使われた値が一致: OK")
```

    不変条件: 全 22 学習(本番 15・較正 7)で見た事例数 E = 262,144・ステップ数 E/N・学習率のスケジュールの形(同じ N の間)が同一、全バッチでキャプションが互いに異なる、同じシードの初期値が同一(4 シード)、同じシード・同じ N のバッチの順序が損失によらず同一(7 組)、本番はすべて同じ評価集合(候補 796 個・3,184 枚)で評価、較正は評価集合を評価していない、S を共有、全学習でテキスト encoder に入れた系列が除外した組を含まない: OK。NegCLIP で分母に加えた色の入れ替えは規則が作った数の 0.514(順序の入れ替えはすべて)
    SMOKE_TEST=False の配線: 実効値 {"level": "prod", "plan": 11, "stage": 3, "examples": 262144, "seeds_a": 3, "seeds_bc": 3, "a_batch_sizes": [16, 256], "calibration_mode": "representative", "calibrated_conditions": 2, "execution_mode": "fp32_eager", "runs": 15} と実際に使われた値が一致: OK


### 6.12 判定結果の一覧

前提条件が 1 つでも不成立の実験は、判定関数の結果に関わらず「前提不成立」とする(判定不能とは区別する)。判定関数の結果は参考として
別欄に残す。選ばれた計画($E$・段階)と、判定に実際に使ったシード数・水準を印字する。


```python
def verdict_label(computed: str, preconditions: list[str]) -> str:
    return computed if all(precondition_status.get(p) for p in preconditions) else "前提不成立"


print(f"{_tag}計画の選択: {PLAN_SELECTION_MESSAGE}")
print(
    f"{_tag}判定に使った値: 計画 {SELECTED_PLAN}(E = {EXAMPLES:,}、段階 {SELECTED_STAGE})、較正の方式 {CALIBRATION_MODE!r}、"
    f"実行の方式 {EXECUTION_MODE!r}、"
    f"n_A = {len(SEEDS_A)}(N {A_BATCH_SIZES})、n_BC = {len(SEEDS_BC)}"
)
VERDICT_ROWS = [
    ("A", f"D = {CONTRAST_A['value']:+.4f}(sigma {CONTRAST_A['sigma']:.4f})", A_PRECONDITIONS, verdict_A),
    ("B", f"B = {CONTRAST_B['value']:+.4f}(sigma {CONTRAST_B['sigma']:.4f})", B_PRECONDITIONS, verdict_B),
    ("C", f"C = {CONTRAST_C['value']:+.4f}(sigma {CONTRAST_C['sigma']:.4f})", C_PRECONDITIONS, verdict_C),
]
print(f"{_tag}実験 | 対比量 | 前提条件 | 判定関数の結果 | 最終判定")
for _name, _value, _pre, _computed in VERDICT_ROWS:
    print(f"{_tag}{_name} | {_value} | " + ", ".join(f"{p}={precondition_status.get(p)}" for p in _pre) + f" | {_computed} | {verdict_label(_computed, _pre)}")
print(f"\nノートブック全体の実行時間(アップロードの前まで): {(time.time() - NOTEBOOK_START_TIME) / 60:.1f} 分")
```

    計画の選択: 残りの予算 114.8 分に収まる番号の最も小さい計画として、計画 11 を選んだ(見積もり 114.4 分、cuda 基準、実行の方式 'fp32_eager')
    判定に使った値: 計画 11(E = 262,144、段階 3)、較正の方式 'representative'、実行の方式 'fp32_eager'、n_A = 3(N (16, 256))、n_BC = 3
    実験 | 対比量 | 前提条件 | 判定関数の結果 | 最終判定
    A | D = +0.0939(sigma 0.0304) | P0(softmax,256)=True, P0(sigmoid,256)=True, P1(softmax,16)=True, P1(sigmoid,16)=True, P1(softmax,256)=True, P1(sigmoid,256)=True | 支持 | 支持
    B | B = +0.0003(sigma 0.0003) | P0(softmax,256)=True, P1(softmax,256,BC)=True | 判定不能 | 判定不能
    C | C = +0.0003(sigma 0.0004) | P0(softmax,256)=True, P1(softmax,256,BC)=True, P1(negclip,256)=True | 判定不能 | 判定不能
    
    ノートブック全体の実行時間(アップロードの前まで): 97.5 分


### 6.13 アップロード(NegCLIP・シード 0 のモデル)

実験 C の NegCLIP・$N = 256$・シード 0 のモデル(両方の encoder・射影・温度とバイアス)を、`kojikojiprg/ai-theories-clip-synthetic-scenes`の
`main`ブランチにアップロードする。**対象は結果を見て選ばず、この指定で固定する**(6.1 節)。

- 重みを`model_state.pt`に保存して読み戻し、**読み戻した重みそのもので評価し直した値** をモデルカードの性能指標にする(学習中に記録した
  値を流用しない。読み戻した値と学習直後の評価の差も印字する)。
- `config.json`にモデルの構成・語彙(検証用)・データの生成規則の所在とコミット・学習の設定を書く。トークナイザとデータは
  `src/data/synthetic_scenes.py`の規則から決定的に再生成できるので同梱しない。
- **安全装置**: (1)`UPLOAD_ARTIFACTS = True`のときのみアップロードする(既定は`False`、`SMOKE_TEST`とは独立)。(2)`SMOKE_TEST = True`の
  重みはアップロードしない。(3)前提条件 P1(NegCLIP)が成立しない重み(学習が進んでいない重み)はアップロードしない。
  (4)Colab Secrets から`HF_TOKEN`を取得できない場合は、例外で停止せずスキップして印字する。(5)アップロードの後、`list_repo_files`で
  3 つのファイルが存在することを確かめ、`model_state.pt`をダウンロードし直して SHA-256 が一致することを確かめる。


```python
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

UPLOAD_FILES = ("model_state.pt", "config.json", "README.md")
_upload_record = RECORDS[UPLOAD_KEY]
_upload_dir = Path(tempfile.mkdtemp())
CHECKPOINT_PATH = _upload_dir / "model_state.pt"
torch.save(_upload_record["state_dict"], CHECKPOINT_PATH)
_reloaded_state = torch.load(CHECKPOINT_PATH, map_location="cpu")
assert _reloaded_state.keys() == _upload_record["state_dict"].keys()
assert all(torch.equal(_reloaded_state[k], _upload_record["state_dict"][k]) for k in _reloaded_state)
CHECKPOINT_SHA256 = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()

# 読み戻した重みそのもので評価し直す(モデルカードの性能指標)
torch.manual_seed(0)
_uploaded_model = build_clip("negclip")
_uploaded_model.load_state_dict(_reloaded_state)
_uploaded_model = _uploaded_model.to(device)
UPLOADED_METRICS = {"seen": evaluate_seen(_uploaded_model), "final": evaluate_final(_uploaded_model, keep_projection=False)}
del _uploaded_model
_recorded = {"M": zero_shot_accuracy(UPLOAD_KEY), "seen": _upload_record["seen"]["accuracy"]}
_reevaluated = {"M": UPLOADED_METRICS["final"]["unseen_retrieval"]["accuracy"], "seen": UPLOADED_METRICS["seen"]["accuracy"]}
print(f"保存・読み戻しで全 {len(_reloaded_state)} テンソルが一致: OK。バイト数 {CHECKPOINT_PATH.stat().st_size:,}、SHA-256 {CHECKPOINT_SHA256}")
print(f"読み戻した重みで評価し直した値 {_reevaluated}(学習直後の評価 {_recorded}、差 " + "、".join(f"{k} {_reevaluated[k] - _recorded[k]:+.2e}" for k in _recorded) + ")")
assert all(abs(_reevaluated[k] - _recorded[k]) <= 1 / len(UNSEEN_IMAGES) for k in _recorded)  # 高々 1 枚の違いまで(演算の丸め)

UPLOAD_CONFIG = {
    "architecture": "CLIPDualEncoder (src/models/clip.py): VisionTransformer (src/models/vit.py) + CausalTextTransformer",
    "vision_config": VISION_CONFIG,
    "text_config": TEXT_CONFIG | {"vocabulary_size": len(VOCABULARY), "end_token_id": VOCABULARY.end_id},
    "embedding_dim": EMBEDDING_DIM,
    "loss": "negclip",
    "vocabulary": list(VOCABULARY.tokens),
    "data_generator": "src/data/synthetic_scenes.py (build_caption_universe, render_scenes)",
    "holdout_pairs": [[COLOR_NAMES[c], SHAPES[s]] for c, s in sorted(HOLDOUT)],
    "render_seeds": {"train": TRAIN_RENDER_SEED, "validation": VALIDATION_RENDER_SEED, "unseen": UNSEEN_RENDER_SEED},
    "training": {
        "examples": EXAMPLES,
        "batch_size": STANDARD_BATCH_SIZE,
        "steps": steps_for(EXAMPLES, STANDARD_BATCH_SIZE),
        "peak_learning_rate": _upload_record["learning_rate"],
        "seed": UPLOAD_KEY_SEED,
        "hard_negatives": ["attribute_swap", "order_swap"],
    },
    "source_commit": execution_environment.get("git_commit"),
    "source_has_uncommitted_changes": execution_environment.get("git_has_uncommitted_changes"),
    "model_state_sha256": CHECKPOINT_SHA256,
}
CONFIG_PATH = _upload_dir / "config.json"
CONFIG_PATH.write_text(json.dumps(UPLOAD_CONFIG, ensure_ascii=False, indent=2))


def build_model_card() -> str:
    two = UPLOADED_METRICS["final"]["unseen_two_alternative"]
    commit = UPLOAD_CONFIG["source_commit"]
    return f'''---
language: en
license: mit
tags:
- ai-theories
- clip
- negclip
- contrastive-learning
- synthetic-data
- scratch-implementation
---

# ai-theories CLIP(合成のシーン、NegCLIP)

`ai-theories`(https://github.com/kojikojiprg/ai-theories)プロジェクトの成果物。
[020. CLIP と対照学習](https://github.com/kojikojiprg/ai-theories/blob/main/theories/05_vision_language/020_clip_contrastive_learning.ipynb)
の実験 C で学習した、NegCLIP(困難な負例のキャプションを画像 -> テキストの分母に加えた対照学習)のモデル(バッチサイズ {STANDARD_BATCH_SIZE}・
見た事例数 {EXAMPLES:,}・シード {UPLOAD_KEY_SEED})。画像 encoder(ViT)・テキスト encoder(因果マスクつきの Transformer)・共通の埋め込み空間への
射影・温度を含む。021 の入力として使うことを想定している。

スクラッチ実装であり、研究・教育目的のモデルである。品質保証は行っていない。商用・実運用での利用は想定しない。
学習データは 2 つの図形(8 色 x 5 種類)を左右または上下に並べた 32 x 32 の合成画像と、規則で生成した英語のキャプションのみであり、
実画像には使えない。

## 構成

`config.json` を参照。モデルのクラスは `src/models/clip.py` の `CLIPDualEncoder`(画像側は `src/models/vit.py` の `VisionTransformer`)。

**トークナイザとデータはこのリポジトリには同梱していない。** どちらも `src/data/synthetic_scenes.py` の規則から決定的に再生成できる
(`CaptionVocabulary`・`build_caption_universe()`・`render_scenes()`。描画のシードは `config.json` の `render_seeds`)。語彙は検証用に
`config.json` の `vocabulary` にも記載した。取得時点のリポジトリのコミット: `{commit}`。

## 性能指標(アップロードした重みそのものを評価した値)

- 未見の (色, 形) の組み合わせを含む評価集合({len(UNSEEN_IMAGES):,} 枚、候補のキャプション {len(UNSEEN_CAPTION_IDS)} 個)での zero-shot の
  画像 -> テキストの検索の top-1 正解率: {UPLOADED_METRICS["final"]["unseen_retrieval"]["accuracy"]:.4f}
- 同じ評価集合での 2 択の正解率: 困難な負例(色の入れ替え・順序の入れ替え){two["swap"]:.4f}
  (色の入れ替え {two["attribute_swap"]:.4f}・順序の入れ替え {two["order_swap"]:.4f})、ランダムな負例 {two["random"]:.4f}
- 既知の組み合わせの検証集合({len(VALIDATION_IMAGES):,} 枚、候補 {len(TRAIN_CAPTION_IDS):,} 個)での検索の正解率: {UPLOADED_METRICS["seen"]["accuracy"]:.4f}

## 検証情報

- SHA-256(`model_state.pt`): `{CHECKPOINT_SHA256}`

## 関連ノートブック

- [019. ViT と画像パッチ埋め込み](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/019_vision_transformer-theory)
- [020. CLIP と対照学習](https://github.com/kojikojiprg/ai-theories/blob/main/theories/05_vision_language/020_clip_contrastive_learning.ipynb)
'''


README_PATH = _upload_dir / "README.md"
README_PATH.write_text(build_model_card())


def upload_skip_reason(upload_artifacts: bool, smoke_test: bool, learned: bool) -> str | None:
    # アップロードをスキップする理由(スキップしないなら None)。判定の順序: UPLOAD_ARTIFACTS -> SMOKE_TEST -> P1(NegCLIP)
    if not upload_artifacts:
        return "UPLOAD_ARTIFACTS = False のため、アップロードをスキップした。"
    if smoke_test:
        return "SMOKE_TEST = True のため、スモークテストの重みはアップロードしない(スキップした)。"
    if not learned:
        return "前提条件 P1(NegCLIP)が成立していないため、アップロードをスキップした(学習が進んでいない重みは公開しない)。"
    return None


for _upload, _smoke, _learned in itertools.product((False, True), repeat=3):  # 分岐の順序の確認(全 8 通り)
    _reason = upload_skip_reason(_upload, _smoke, _learned)
    assert (_reason is None) == (_upload and not _smoke and _learned)
    assert _upload or _reason.startswith("UPLOAD_ARTIFACTS")
    assert not (_upload and _smoke) or _reason.startswith("SMOKE_TEST")
    assert not (_upload and not _smoke and not _learned) or _reason.startswith("前提条件 P1")
print("アップロードの分岐(UPLOAD_ARTIFACTS -> SMOKE_TEST -> P1(NegCLIP) -> HF_TOKEN)の全 8 通りの確認: OK")

_learned = bool(precondition_status.get(f"P1(negclip,{STANDARD_BATCH_SIZE})"))
_skip_reason = upload_skip_reason(UPLOAD_ARTIFACTS, SMOKE_TEST, _learned)
print(f"UPLOAD_ARTIFACTS = {UPLOAD_ARTIFACTS}、SMOKE_TEST = {SMOKE_TEST}、P1(NegCLIP) = {_learned}、アップロードする鍵 {UPLOAD_KEY}")
if _skip_reason is not None:
    print(_skip_reason)
else:
    _hf_token = None
    try:
        from google.colab import userdata

        _hf_token = userdata.get("HF_TOKEN")
    except Exception as _e:  # noqa: BLE001  # Colab Secrets 未設定・非 Colab 環境など理由を問わずスキップする
        print(f"Colab Secrets から HF_TOKEN を取得できなかったため、アップロードをスキップした: {type(_e).__name__}")
    if not _hf_token:
        print("HF_TOKEN がないため、アップロードをスキップした。")
    else:
        _api = HfApi(token=_hf_token)
        _api.create_repo(repo_id=HUB_REPO_ID, repo_type="model", exist_ok=True)
        for _path in (CHECKPOINT_PATH, CONFIG_PATH, README_PATH):
            _api.upload_file(path_or_fileobj=str(_path), path_in_repo=_path.name, repo_id=HUB_REPO_ID, revision="main")
        # upload_file が例外を送出しなかったことだけを成功の根拠にせず、存在と内容を確かめる
        _files = set(_api.list_repo_files(repo_id=HUB_REPO_ID, revision="main"))
        for _expected in UPLOAD_FILES:
            assert _expected in _files, f"{_expected} が {HUB_REPO_ID}@main に見つからない"
        _downloaded = hf_hub_download(HUB_REPO_ID, "model_state.pt", revision="main", force_download=True, token=_hf_token)
        assert hashlib.sha256(Path(_downloaded).read_bytes()).hexdigest() == CHECKPOINT_SHA256
        print(
            f"[OK] アップロード完了。list_repo_files で {UPLOAD_FILES} の存在を確認し、model_state.pt の SHA-256 が一致した: "
            f"https://huggingface.co/{HUB_REPO_ID}"
        )
print(f"\nノートブック全体の実行時間: {(time.time() - NOTEBOOK_START_TIME) / 60:.1f} 分")
```

    保存・読み戻しで全 110 テンソルが一致: OK。バイト数 6,504,819、SHA-256 efc2ab9dcd4730f4fe4b74b13c66f33d803fe7c65d9fa75929a8a13f425f1c17
    読み戻した重みで評価し直した値 {'M': 0.780464824120603, 'seen': 0.8691135734072022}(学習直後の評価 {'M': 0.780464824120603, 'seen': 0.8691135734072022}、差 M +0.00e+00、seen +0.00e+00)
    アップロードの分岐(UPLOAD_ARTIFACTS -> SMOKE_TEST -> P1(NegCLIP) -> HF_TOKEN)の全 8 通りの確認: OK
    UPLOAD_ARTIFACTS = True、SMOKE_TEST = False、P1(NegCLIP) = True、アップロードする鍵 ('negclip', 256, 0)



    Processing Files (0 / 0)      : |          |  0.00B /  0.00B            



    New Data Upload               : |          |  0.00B /  0.00B            



      ...mppfsw2noh/model_state.pt:   9%|8         |  562kB / 6.50MB            



    model_state.pt: reconstructing file:   0%|          |  0.00B / 6.50MB            



    model_state.pt: downloading bytes:           |  0.00B            


    [OK] アップロード完了。list_repo_files で ('model_state.pt', 'config.json', 'README.md') の存在を確認し、model_state.pt の SHA-256 が一致した: https://huggingface.co/kojikojiprg/ai-theories-clip-synthetic-scenes
    
    ノートブック全体の実行時間: 97.7 分




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/05_vision_language/020_clip_contrastive_learning.ipynb
