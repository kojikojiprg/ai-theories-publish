---
title: "KV キャッシュと推論の計算量(KV Cache and Inference Compute)(実装・実験編 2/4)"
---

この記事は後編(実装・実験編 2/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/010_kv_cache_and_inference_compute-practice-1)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/010_kv_cache_and_inference_compute-practice-3)。

### 7.2 実験 A: KV キャッシュによる生成時間の次数の低下

- **検証すること**: KV キャッシュを用いると、生成時間の生成長 $T$ に対する
  べき指数が小さくなる。
- **対比量**: $\Delta b = b_{\text{no cache}} - b_{\text{cache}}$
  ($b_{\text{no cache}}$ は KV キャッシュを使わない条件のべき指数、
  $b_{\text{cache}}$ は使う条件のべき指数)。$b$ は $\log t = \log a + b \log T$
  のあてはめ(`fit_power_law_exponent`)で推定する。
  - **介入の直接作用点**: この介入(KV キャッシュ)が直接作用するのは 1 ステップ
    あたりの計算量(3.1 節の $O(t \, d_{\mathrm{model}}^2 + t^2 d_{\mathrm{model}})$
    → $O(d_{\mathrm{model}}^2 + t \, d_{\mathrm{model}})$、キャッシュの有無で
    ステップ $t$ の計算コストそのものが変わる)である。対比量(生成時間全体の
    $T$ に対するべき指数)はこれを $T$ ステップ分積算した下流の量だが、積算は
    決定的な和であり(実測ノイズが積算過程で新たに混入しない)、直接の作用点との
    距離は小さい。
- **判定基準**: $\Delta b > 2\sigma_{\Delta b}$ なら支持、$\Delta b <
  -2\sigma_{\Delta b}$ なら反証、それ以外は判定不能。
  $\sigma_{\Delta b} = \sqrt{\sigma_{b,\text{no cache}}^2 + \sigma_{b,\text{cache}}^2}$
  とし、$\sigma_b$ は各条件のあてはめにおける回帰係数の標準誤差
  (`fit_power_law_exponent`の`exponent_stderr`)とする。
- **水準**: 生成長 $T$ を等比の刻みで 5 点、各点を反復計測する(反復回数は
  スモークテストと本番で異なり、セットアップセルの`SMOKE_A_REPEATS`・
  `PRODUCTION_A_REPEATS`で定義する)。
- **診断量**: 各 $T$ での実測時間の絶対値、および理論上のべき指数の上限(3 と 2、
  3.1 節)。**べき指数の絶対値は判定基準にしない**(本モデルの
  $d_{\mathrm{model}}=256$ 程度では $T d_{\mathrm{model}}^2$ の項が支配的になり、
  実測の指数は理論上限より小さく出る)。


```python
# --- 実験 A: 本番アーキテクチャ構成(重みはランダム初期化)で時間計測する ---
# max_sequence_length は、実験 A・C の本番水準(PRODUCTION_A_T_LEVELS の最大 2048、
# PRODUCTION_C_T_LARGE の 4096)にプロンプト長を足した値を上回る必要がある(重みを
# 持たない設定値であり、計測用モデルはランダム初期化なので本番の学習結果には
# 影響しない)。
TIMING_MODEL_MAX_SEQUENCE_LENGTH = 8192

torch.manual_seed(0)
_a_rope = RotaryPositionEmbedding(
    PROD_CONFIG["d_model"] // PROD_CONFIG["num_heads"], max_position=TIMING_MODEL_MAX_SEQUENCE_LENGTH
)
timing_model_a = GPTLanguageModel(
    PROD_CONFIG["vocabulary_size"], PROD_CONFIG["d_model"], PROD_CONFIG["num_layers"],
    PROD_CONFIG["num_heads"], PROD_CONFIG["d_ff"], max_sequence_length=TIMING_MODEL_MAX_SEQUENCE_LENGTH,
    positional_transform=_a_rope, normalization_factory=RMSNorm,
    feed_forward_factory=functools.partial(
        SwiGLUFeedForwardNetwork, PROD_CONFIG["d_model"], PROD_CONFIG["swiglu_d_ff"]
    ),
).to(TIMING_DEVICE)
timing_model_a.eval()

# A_T_LEVELS・A_REPEATS はセットアップセル(SMOKE_TEST の定義の直後)で定義済み。
A_PROMPT_LENGTH = 4


def warmup_generate(model, t_new_tokens, use_cache, prompt_length, device):
    '''計測ループの前に、指定条件でウォームアップ(P0、WARMUP_REPEATS 回)を行う。'''
    assert prompt_length + t_new_tokens <= model.max_sequence_length, (
        f"prompt_length({prompt_length}) + t_new_tokens({t_new_tokens}) が "
        f"model.max_sequence_length({model.max_sequence_length}) を超えている"
    )
    for _ in range(WARMUP_REPEATS):
        x = torch.randint(0, PROD_CONFIG["vocabulary_size"], (1, prompt_length), device=device)
        with torch.no_grad():
            model.generate(x, max_new_tokens=t_new_tokens, temperature=0.0, use_cache=use_cache)
        sync(device)


def measure_generate_once(model, t_new_tokens, use_cache, prompt_length, device):
    '''指定条件で 1 回だけ計測する(ウォームアップは含まない)。'''
    x = torch.randint(0, PROD_CONFIG["vocabulary_size"], (1, prompt_length), device=device)
    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(x, max_new_tokens=t_new_tokens, temperature=0.0, use_cache=use_cache)
    sync(device)
    return time.perf_counter() - t0


# ウォームアップは条件ごとにまとめて計測ループの前に行う(WARMUP_REPEATS 回、P0)。
for _t in A_T_LEVELS:
    for _use_cache in (False, True):
        warmup_generate(timing_model_a, _t, _use_cache, A_PROMPT_LENGTH, TIMING_DEVICE)

# 計測は for 反復: for 条件: の順で行う(for 条件: for 反復: ではない)。1 つの条件の
# 反復をまとめて連続実行すると、一過性の擾乱(OS のスケジューリング・サーマルスロット
# リングなど)が特定の条件に集中して入り、P0(変動係数)が不成立になりやすい。
# 反復をまたいで条件を切り替えることで、擾乱が全条件に均等に散るようにする。
experiment_a_raw: dict[str, dict[int, list[float]]] = {
    "nocache": {t: [] for t in A_T_LEVELS},
    "cache": {t: [] for t in A_T_LEVELS},
}
for _rep in range(A_REPEATS):
    for _t in A_T_LEVELS:
        experiment_a_raw["nocache"][_t].append(
            measure_generate_once(timing_model_a, _t, False, A_PROMPT_LENGTH, TIMING_DEVICE)
        )
        experiment_a_raw["cache"][_t].append(
            measure_generate_once(timing_model_a, _t, True, A_PROMPT_LENGTH, TIMING_DEVICE)
        )

for _t in A_T_LEVELS:
    for _cond in ("nocache", "cache"):
        _arr = np.array(experiment_a_raw[_cond][_t])
        _cv = _arr.std() / _arr.mean()
        print(f"T={_t:4d} {_cond:8s} mean={_arr.mean():.5f}s cv={_cv:.4f}")

# 判定の一次情報を全件印字する(セル出力が唯一の記録であるため、1 行の JSON では
# GitHub のノートブックプレビューで行が途中で切れて読めなくなる。indent=2 で
# 縦に展開する)。
print("\n[判定の一次情報 / experiment_a_raw]")
print(json.dumps(experiment_a_raw, ensure_ascii=False, indent=2))
```

    T= 128 nocache  mean=0.79258s cv=0.1030
    T= 128 cache    mean=0.68456s cv=0.1012
    T= 256 nocache  mean=1.60900s cv=0.1169
    T= 256 cache    mean=1.39933s cv=0.0970
    T= 512 nocache  mean=3.21300s cv=0.0805
    T= 512 cache    mean=2.73176s cv=0.0683
    T=1024 nocache  mean=7.87631s cv=0.0306
    T=1024 cache    mean=5.48363s cv=0.0515
    T=2048 nocache  mean=35.38919s cv=0.0071
    T=2048 cache    mean=11.00015s cv=0.0278
    
    [判定の一次情報 / experiment_a_raw]
    {
      "nocache": {
        "128": [
          0.7653453169999693,
          0.7832155620000094,
          1.0346706259999792,
          0.7436916340000153,
          0.7478950819999,
          0.7686832729999651,
          0.7700697830000536,
          0.7565151599999353,
          0.7733588579999378,
          0.7823926989999563
        ],
        "256": [
          1.6085147000000006,
          1.4842232270000295,
          1.739265932999956,
          1.4178371149999975,
          1.498448283000016,
          1.955587358999992,
          1.5105997079999725,
          1.4546403250001276,
          1.9322694949999004,
          1.488568756000177
        ],
        "512": [
          3.297010575999934,
          3.0285194690000026,
          3.111106079000024,
          3.379878809000047,
          3.0173127280000926,
          2.9763226670000904,
          3.644594698999981,
          2.9683244299999387,
          3.0298430400000598,
          3.6770929290000822
        ],
        "1024": [
          7.782063882999978,
          7.62329897799998,
          8.47113149400002,
          7.714849438999977,
          7.877687326,
          7.974087218000022,
          7.730403409000019,
          8.066104506000102,
          7.896156912000151,
          7.627355110000053
        ],
        "2048": [
          35.23076333300003,
          35.17464205299996,
          35.891652843999964,
          35.14740489199994,
          35.26860655100006,
          35.72361875199999,
          35.19792697800017,
          35.35647693300007,
          35.642459014999986,
          35.25839757299991
        ]
      },
      "cache": {
        "128": [
          0.6507022639999605,
          0.6474518880000915,
          0.8339384759999575,
          0.6251725559999386,
          0.6134086570000363,
          0.7949965230000089,
          0.6589638140000034,
          0.6684643090000009,
          0.7015271590000793,
          0.6509856430000127
        ],
        "256": [
          1.7465375709999762,
          1.295469066999999,
          1.4059925109999085,
          1.5424725769998986,
          1.320862520999981,
          1.3351481559999456,
          1.2973628070000132,
          1.3095069789999343,
          1.3998732960001234,
          1.3400958830000036
        ],
        "512": [
          2.547096355000008,
          3.1898153979999506,
          2.7629248099999586,
          2.6056170690000044,
          2.9381849760000023,
          2.6568409440000096,
          2.703817084999855,
          2.7242217860000437,
          2.6138267310000174,
          2.575273789999983
        ],
        "1024": [
          5.204348595000056,
          5.922362668999995,
          5.1918832630000225,
          5.210557936999976,
          5.786260109999944,
          5.256141574000026,
          5.500197403999891,
          5.7187076870000055,
          5.235772974000156,
          5.8100533669999095
        ],
        "2048": [
          10.934839506000003,
          10.56828681400009,
          10.84960030100001,
          10.949106391999976,
          10.587816277999991,
          11.426327829000002,
          11.065239596000083,
          11.04893227499997,
          11.598395301999972,
          10.972948305000045
        ]
      }
    }



```python
# --- 実験 A: 本番実行であることの保証 ---
# SMOKE_TEST=False のとき、実際に使った水準が PRODUCTION_* 定数と一致することを
# アサーションで確認する(A_T_LEVELS・A_REPEATS の代入だけでなく、実際に
# experiment_a_raw へ記録された水準・反復回数そのものを検証する)。
if not SMOKE_TEST:
    assert A_T_LEVELS == PRODUCTION_A_T_LEVELS, (
        f"本番実行のはずが A_T_LEVELS が PRODUCTION_A_T_LEVELS と一致しない: {A_T_LEVELS}"
    )
    assert A_REPEATS == PRODUCTION_A_REPEATS, (
        f"本番実行のはずが A_REPEATS が PRODUCTION_A_REPEATS と一致しない: {A_REPEATS}"
    )
    for _cond in ("nocache", "cache"):
        assert set(experiment_a_raw[_cond].keys()) == set(PRODUCTION_A_T_LEVELS), (
            f"experiment_a_raw[{_cond!r}] の水準が PRODUCTION_A_T_LEVELS と一致しない: "
            f"{sorted(experiment_a_raw[_cond].keys())}"
        )
        for _t, _vals in experiment_a_raw[_cond].items():
            assert len(_vals) == PRODUCTION_A_REPEATS, (
                f"experiment_a_raw[{_cond!r}][{_t}] の反復回数が PRODUCTION_A_REPEATS と"
                f"一致しない: {len(_vals)}"
            )
    print(f"[OK] 実験 A は本番水準(A_T_LEVELS={A_T_LEVELS}, A_REPEATS={A_REPEATS})で実行された")
```

    [OK] 実験 A は本番水準(A_T_LEVELS=[128, 256, 512, 1024, 2048], A_REPEATS=10)で実行された



```python
# --- 実験 A: 前提条件 P0-a・P0-b の確認(7.1 節) ---
_a_p0a_by_condition = {}
_a_p0b_by_condition = {}
for _cond, _by_t in experiment_a_raw.items():
    for _t, _vals in _by_t.items():
        _key = f"{_cond}_T{_t}"
        _a_p0a_by_condition[_key] = compute_p0a(_vals)
        _a_p0b_by_condition[_key] = compute_p0b(_vals)
        print(
            f"{_key:14s} P0-a: 先頭={_a_p0a_by_condition[_key]['head_mean']:.5f}s "
            f"末尾={_a_p0a_by_condition[_key]['tail_mean']:.5f}s "
            f"差={_a_p0a_by_condition[_key]['diff']:.5f}s "
            f"閾値={_a_p0a_by_condition[_key]['threshold']:.5f}s "
            f"成立={_a_p0a_by_condition[_key]['holds']}"
        )
        print(
            f"{'':14s} P0-b: 平均={_a_p0b_by_condition[_key]['mean']:.5f}s "
            f"標準誤差={_a_p0b_by_condition[_key]['stderr']:.5f}s "
            f"比率={_a_p0b_by_condition[_key]['ratio']:.4f} "
            f"成立={_a_p0b_by_condition[_key]['holds']}"
        )

A_P0A_HOLDS = all(v["holds"] for v in _a_p0a_by_condition.values())
A_P0B_HOLDS = all(v["holds"] for v in _a_p0b_by_condition.values())
A_P0_HOLDS = A_P0A_HOLDS and A_P0B_HOLDS
print(f"\nP0-a 全条件成立: {A_P0A_HOLDS} / P0-b 全条件成立: {A_P0B_HOLDS} / "
      f"P0(P0-a かつ P0-b)成立: {A_P0_HOLDS}")
```

    nocache_T128   P0-a: 先頭=0.86108s 末尾=0.77076s 差=0.09032s 閾値=0.17214s 成立=True
                   P0-b: 平均=0.79258s 標準誤差=0.02722s 比率=0.0343 成立=True
    nocache_T256   P0-a: 先頭=1.61067s 末尾=1.62516s 差=0.01449s 閾値=0.39662s 成立=True
                   P0-b: 平均=1.60900s 標準誤差=0.06271s 比率=0.0390 成立=True
    nocache_T512   P0-a: 先頭=3.14555s 末尾=3.22509s 差=0.07954s 閾値=0.54516s 成立=True
                   P0-b: 平均=3.21300s 標準誤差=0.08620s 比率=0.0268 成立=True
    nocache_T1024  P0-a: 先頭=7.95883s 末尾=7.86321s 差=0.09563s 閾値=0.50860s 成立=True
                   P0-b: 平均=7.87631s 標準誤差=0.08042s 比率=0.0102 成立=True
    nocache_T2048  P0-a: 先頭=35.43235s 末尾=35.41911s 差=0.01324s 閾値=0.52803s 成立=True
                   P0-b: 平均=35.38919s 標準誤差=0.08349s 比率=0.0024 成立=True
    cache_T128     P0-a: 先頭=0.71070s 末尾=0.67366s 差=0.03704s 閾値=0.14599s 成立=True
                   P0-b: 平均=0.68456s 標準誤差=0.02308s 比率=0.0337 成立=True
    cache_T256     P0-a: 先頭=1.48267s 末尾=1.34983s 差=0.13284s 閾値=0.28607s 成立=True
                   P0-b: 平均=1.39933s 標準誤差=0.04523s 比率=0.0323 成立=True
    cache_T512     P0-a: 先頭=2.83328s 末尾=2.63777s 差=0.19550s 閾値=0.39342s 成立=True
                   P0-b: 平均=2.73176s 標準誤差=0.06221s 比率=0.0228 成立=True
    cache_T1024    P0-a: 先頭=5.43953s 末尾=5.58818s 差=0.14865s 閾値=0.59518s 成立=True
                   P0-b: 平均=5.48363s 標準誤差=0.09411s 比率=0.0172 成立=True
    cache_T2048    P0-a: 先頭=10.78424s 末尾=11.20676s 差=0.42252s 閾値=0.64393s 成立=True
                   P0-b: 平均=11.00015s 標準誤差=0.10181s 比率=0.0093 成立=True
    
    P0-a 全条件成立: True / P0-b 全条件成立: True / P0(P0-a かつ P0-b)成立: True



```python
# --- 実験 A: べき乗則あてはめと判定 ---
_a_y_nocache = [float(np.mean(experiment_a_raw["nocache"][t])) for t in A_T_LEVELS]
_a_y_cache = [float(np.mean(experiment_a_raw["cache"][t])) for t in A_T_LEVELS]

experiment_a_fit_nocache = fit_power_law_exponent(A_T_LEVELS, _a_y_nocache)
experiment_a_fit_cache = fit_power_law_exponent(A_T_LEVELS, _a_y_cache)
print(f"b(キャッシュなし) = {experiment_a_fit_nocache.exponent:.4f} "
      f"+- {experiment_a_fit_nocache.exponent_stderr:.4f}")
print(f"b(キャッシュあり) = {experiment_a_fit_cache.exponent:.4f} "
      f"+- {experiment_a_fit_cache.exponent_stderr:.4f}")

_a_delta_b = experiment_a_fit_nocache.exponent - experiment_a_fit_cache.exponent
_a_sigma_delta_b = float(
    np.sqrt(experiment_a_fit_nocache.exponent_stderr**2 + experiment_a_fit_cache.exponent_stderr**2)
)
print(f"Delta_b = {_a_delta_b:.4f}, sigma_Delta_b = {_a_sigma_delta_b:.4f}, "
      f"Delta_b / sigma = {_a_delta_b / _a_sigma_delta_b:.2f}")

# 診断量: 理論上のべき指数の上限との対比(判定基準ではない)。
print(f"[診断量] 理論上限: キャッシュなし b<=3, キャッシュあり b<=2")

plot_log_log_fit(
    A_T_LEVELS, _a_y_nocache, experiment_a_fit_nocache.exponent, experiment_a_fit_nocache.coefficient,
    label="no cache", xlabel="T (new tokens)", ylabel="generation time (s)",
    title="Experiment A (smoke test, code verification only)",
)
plt.show()
# 本番実行後、このセルの出力(グラフ・数値)を踏まえて 8 節の考察を書く。
# このセル自体はスモークテストの数値であり、判定結果を結論として本文には記述しない。
```

    b(キャッシュなし) = 1.3253 +- 0.1366
    b(キャッシュあり) = 0.9983 +- 0.0050
    Delta_b = 0.3270, sigma_Delta_b = 0.1367, Delta_b / sigma = 2.39
    [診断量] 理論上限: キャッシュなし b<=3, キャッシュあり b<=2



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/010_kv_cache_and_inference_compute/output_33_1.png)
    



```python
# --- 実験 A: 本番水準・本番反復回数への外挿(CLAUDE.md 第 1 段階要件 1) ---
# スモークテストのべき乗則あてはめ(t = a * T^b)をそのまま使って本番水準の時間を見積もる。
# 各水準につきウォームアップ WARMUP_REPEATS 回 + PRODUCTION_A_REPEATS 回の呼び出しがある。
_a_calls_per_level = PRODUCTION_A_REPEATS + WARMUP_REPEATS
_a_extrapolated_nocache = sum(
    experiment_a_fit_nocache.coefficient * t**experiment_a_fit_nocache.exponent
    for t in PRODUCTION_A_T_LEVELS
) * _a_calls_per_level
_a_extrapolated_cache = sum(
    experiment_a_fit_cache.coefficient * t**experiment_a_fit_cache.exponent
    for t in PRODUCTION_A_T_LEVELS
) * _a_calls_per_level
EXPERIMENT_A_EXTRAPOLATED_SECONDS = _a_extrapolated_nocache + _a_extrapolated_cache
print(f"外挿: キャッシュなし {_a_extrapolated_nocache:.1f} 秒 + "
      f"キャッシュあり {_a_extrapolated_cache:.1f} 秒 = "
      f"{EXPERIMENT_A_EXTRAPOLATED_SECONDS:.1f} 秒"
      f"(水準 {PRODUCTION_A_T_LEVELS}、水準あたり {_a_calls_per_level} 回)")
```

    外挿: キャッシュなし 549.7 秒 + キャッシュあり 276.9 秒 = 826.6 秒(水準 [128, 256, 512, 1024, 2048]、水準あたり 13 回)


### 7.3 実験 B: prefill と decode のバッチサイズ依存性の差

- **検証すること**: 総実行時間のバッチサイズに対する依存は、decode のほうが
  prefill より弱い(メモリ帯域律速の署名)。
- **対比量**: $\Delta b_B = b_{\text{decode}}(B) - b_{\text{prefill}}(B)$。
  $b(B)$ は総実行時間のバッチサイズ $B$ に対するべき指数。
  - **介入の直接作用点**: バッチサイズ $B$ が直接作用するのは 1 回の行列演算の
    形状(3.3 節の $N = B \cdot S$)であり、対比量($b(B)$)はこの演算を 1 回
    実行した総時間から求める。prefill・decode いずれも「1 回の呼び出し」を
    そのまま計測するため、作用点との距離は小さい。
- **判定基準**: $\Delta b_B < -2\sigma_{\Delta b_B}$ なら支持、$>
  +2\sigma_{\Delta b_B}$ なら反証、それ以外は判定不能。標準偏差の導出は実験 A と同じ
  (`fit_power_law_exponent`の`exponent_stderr`を誤差伝播で合成)。
- **水準**: バッチサイズを等比の刻みで 5 点、各点を反復計測する(反復回数は
  スモークテストと本番で異なり、セットアップセルの`SMOKE_B_REPEATS`・
  `PRODUCTION_B_REPEATS`で定義する)。
- **前提条件 P1_B**: 全バッチサイズで out-of-memory が発生しないこと。
- **診断量**(判定基準は設けない): 演算強度の理論値(3.3 節)、および重み
  読み出しバイト数を T4 GPU の公称メモリ帯域で割った下限時間との比。**この絶対値の
  一致は判定基準にしない**(実測の達成帯域は公称値の 7 割前後であり、真偽を
  判定できる形になりません)。


```python
# --- 実験 B ---
# B_LEVELS・B_REPEATS・B_PREFILL_LEN はセットアップセル(SMOKE_TEST の定義の直後)で
# 定義済み(B_PREFILL_LEN が SMOKE_TEST に関わらず本番値に固定されている理由も
# そちらのコメントを参照)。

B_OOM_LEVELS: list[int] = []  # out-of-memory で前提不成立となりスキップした水準


def warmup_prefill(model, batch_size, seq_len, device):
    for _ in range(WARMUP_REPEATS):
        x = torch.randint(0, PROD_CONFIG["vocabulary_size"], (batch_size, seq_len), device=device)
        with torch.no_grad():
            model(x)
        sync(device)


def measure_prefill_once(model, batch_size, seq_len, device):
    x = torch.randint(0, PROD_CONFIG["vocabulary_size"], (batch_size, seq_len), device=device)
    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        model(x)
    sync(device)
    return time.perf_counter() - t0


def _one_decode_step(model, batch_size, prefill_len, device, timed: bool) -> float | None:
    '''prefill(計測しない)+ decode 1 ステップを行う。timed=True のときのみ
    decode ステップの時間を計測して返す(ウォームアップと本計測を同じ関数で
    共有し、実装の乖離を防ぐ)。
    '''
    kv_cache = KeyValueCache(PROD_CONFIG["num_layers"])
    context = torch.randint(0, PROD_CONFIG["vocabulary_size"], (batch_size, prefill_len), device=device)
    positions = torch.arange(prefill_len, device=device)
    with torch.no_grad():
        model(context, positions=positions, kv_cache=kv_cache)
    next_token = torch.randint(0, PROD_CONFIG["vocabulary_size"], (batch_size, 1), device=device)
    step_pos = torch.tensor([kv_cache.length], device=device)
    if not timed:
        with torch.no_grad():
            model(next_token, positions=step_pos, kv_cache=kv_cache)
        sync(device)
        return None
    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        model(next_token, positions=step_pos, kv_cache=kv_cache)
    sync(device)
    return time.perf_counter() - t0


def warmup_decode_step(model, batch_size, prefill_len, device):
    for _ in range(WARMUP_REPEATS):
        _one_decode_step(model, batch_size, prefill_len, device, timed=False)


def measure_decode_step_once(model, batch_size, prefill_len, device):
    return _one_decode_step(model, batch_size, prefill_len, device, timed=True)


# ウォームアップはバッチサイズごとにまとめて計測ループの前に行う(WARMUP_REPEATS 回、
# P0)。out-of-memory はウォームアップの時点で検出し、計測ループに進める前にその
# バッチサイズを除外する(計測開始後に一部の反復だけ欠けた不完全な水準を作らないため)。
_b_active_levels = list(B_LEVELS)
for _b in list(_b_active_levels):
    try:
        warmup_prefill(timing_model_a, _b, B_PREFILL_LEN, TIMING_DEVICE)
        warmup_decode_step(timing_model_a, _b, B_PREFILL_LEN, TIMING_DEVICE)
    except RuntimeError as e:
        if not is_oom_error(e):
            raise
        B_OOM_LEVELS.append(_b)
        _b_active_levels.remove(_b)
        print(f"[前提不成立] B={_b} でウォームアップ中に out-of-memory が発生したため"
              f"この水準をスキップする: {e}")

experiment_b_raw: dict[str, dict[int, list[float]]] = {
    "prefill": {b: [] for b in _b_active_levels},
    "decode": {b: [] for b in _b_active_levels},
}

# 計測は for 反復: for 条件: の順で行う(for 条件: for 反復: ではない)。1 つの条件の
# 反復をまとめて連続実行すると、一過性の擾乱が特定の条件に集中して入り、P0(変動係数)
# が不成立になりやすい。反復をまたいで条件(バッチサイズ)を切り替えることで、擾乱が
# 全条件に均等に散るようにする。ウォームアップ後に初めて out-of-memory が起きた場合
# (通常はウォームアップ時点で検出されるため稀)は、その時点でバッチサイズを打ち切り、
# それまでに記録した反復も破棄する(反復回数が水準間で不揃いにならないようにするため)。
for _rep in range(B_REPEATS):
    for _b in list(_b_active_levels):
        try:
            _prefill_time = measure_prefill_once(timing_model_a, _b, B_PREFILL_LEN, TIMING_DEVICE)
            _decode_time = measure_decode_step_once(timing_model_a, _b, B_PREFILL_LEN, TIMING_DEVICE)
        except RuntimeError as e:
            if not is_oom_error(e):
                raise
            B_OOM_LEVELS.append(_b)
            _b_active_levels.remove(_b)
            del experiment_b_raw["prefill"][_b]
            del experiment_b_raw["decode"][_b]
            print(f"[前提不成立] B={_b} で計測中(反復 {_rep + 1} 回目)に out-of-memory が"
                  f"発生したためこの水準を打ち切る(それまでの {_rep} 回分も破棄する): {e}")
            continue
        experiment_b_raw["prefill"][_b].append(_prefill_time)
        experiment_b_raw["decode"][_b].append(_decode_time)

for _b in _b_active_levels:
    for _cond in ("prefill", "decode"):
        _arr = np.array(experiment_b_raw[_cond][_b])
        print(f"B={_b:3d} {_cond:8s} mean={_arr.mean():.5f}s cv={_arr.std() / _arr.mean():.4f}")

B_P0_OOM_HOLDS = len(B_OOM_LEVELS) == 0
print(f"\n前提条件(全バッチサイズで OOM なし)成立: {B_P0_OOM_HOLDS}"
      f"(OOM が発生した水準: {B_OOM_LEVELS})")
print("\n[判定の一次情報 / experiment_b_raw]")
print(json.dumps(experiment_b_raw, ensure_ascii=False, indent=2))
```

    B=  1 prefill  mean=0.00740s cv=0.0977
    B=  1 decode   mean=0.00540s cv=0.1660
    B=  4 prefill  mean=0.01654s cv=0.0163
    B=  4 decode   mean=0.00582s cv=0.1811
    B= 16 prefill  mean=0.06148s cv=0.0171
    B= 16 decode   mean=0.00614s cv=0.2550
    B= 64 prefill  mean=0.26513s cv=0.0128
    B= 64 decode   mean=0.00764s cv=0.0678
    B=256 prefill  mean=1.07977s cv=0.0156
    B=256 decode   mean=0.02223s cv=0.0174
    
    前提条件(全バッチサイズで OOM なし)成立: True(OOM が発生した水準: [])
    
    [判定の一次情報 / experiment_b_raw]
    {
      "prefill": {
        "1": [
          0.007384354999885545,
          0.010491477999948984,
          0.007282327999973859,
          0.0070714910000333475,
          0.007136941999988267,
          0.008519527999851562,
          0.00752242299995487,
          0.007124402000044938,
          0.007161900000028254,
          0.00832689499998196,
          0.007135113999993337,
          0.007213741999976264,
          0.006982406999895829,
          0.00789564199999404,
          0.007108014000095864,
          0.006996602000072016,
          0.007211159000007683,
          0.007039803999987271,
          0.006998951000014131,
          0.006993880999971225,
          0.007042286000114473,
          0.006958314000030441,
          0.007941698999957225,
          0.007238296999958038,
          0.007101545000068654,
          0.007051766000131465,
          0.0083530360000168,
          0.006925280999894312,
          0.006963201999951707,
          0.006910978999940198
        ],
        "4": [
          0.016159455999968486,
          0.016748894000102155,
          0.016418032000046878,
          0.01648294899996472,
          0.01650347200006763,
          0.016847928999823125,
          0.016766434000146546,
          0.016523361000054138,
          0.016933765999965544,
          0.0165694159998111,
          0.016828052999926513,
          0.01681880299997829,
          0.016449291000071753,
          0.016943664999871544,
          0.016576701000076355,
          0.01654969500009429,
          0.016726025000025402,
          0.01672184299991386,
          0.016629165000040302,
          0.016648619000079634,
          0.01667798199991921,
          0.016339358000095672,
          0.01648971800000254,
          0.01667638000003535,
          0.016081234000012046,
          0.01652231900015977,
          0.015702471999929912,
          0.016465001999904416,
          0.016302233000033084,
          0.016090909000013198
        ],
        "16": [
          0.060462876999963555,
          0.05979709199982608,
          0.06105407299992294,
          0.06210563399986313,
          0.06106861900002514,
          0.06144992000008642,
          0.06264761800002816,
          0.0628504429998884,
          0.06336508300000787,
          0.06095107800001642,
          0.06384517399987999,
          0.06183459700014282,
          0.06184179899992159,
          0.06057436800006144,
          0.06288137899991852,
          0.061088254999958735,
          0.061125613000058365,
          0.06082124299996394,
          0.06315842499998325,
          0.059851418000107515,
          0.06199680800000351,
          0.06253978199993071,
          0.05969996899989383,
          0.06159888999991381,
          0.06121600000005856,
          0.060955672000091,
          0.06067567000013696,
          0.061527952000005826,
          0.060540199000115535,
          0.06093705399985083
        ],
        "64": [
          0.25956343099983314,
          0.26233901799992054,
          0.26255633099981424,
          0.26593946699995286,
          0.2688592489998882,
          0.2679966450000393,
          0.271237358999997,
          0.2699314419999155,
          0.2691033739999966,
          0.2681216940000013,
          0.2673713560000124,
          0.27273400200010656,
          0.26515526900016084,
          0.26585285300006944,
          0.26498731100014083,
          0.2662739520001196,
          0.2678494670001328,
          0.26427964000004067,
          0.2670150410001497,
          0.2663502109999172,
          0.26538918599999306,
          0.26130268699989756,
          0.2614058429999204,
          0.26138642999990225,
          0.26343052100014575,
          0.26088246800009074,
          0.26318729900003746,
          0.2624667589998353,
          0.26049694299990733,
          0.26053375299989057
        ],
        "256": [
          1.057241607999913,
          1.0561827449998873,
          1.0781793669998478,
          1.091431711000041,
          1.1014764430001378,
          1.1043132370000421,
          1.106168685000057,
          1.105253563999895,
          1.1033702260001519,
          1.095579472000054,
          1.0997432579999895,
          1.0993805059999886,
          1.093665276000138,
          1.0914280850001887,
          1.0850288140002249,
          1.0814351779999924,
          1.0810317140001189,
          1.0729520659999707,
          1.072423661999892,
          1.0756389400000899,
          1.0708902740000212,
          1.0684361529999933,
          1.064305547999993,
          1.055835069000068,
          1.0677474069998425,
          1.0576085239999884,
          1.0562015199998314,
          1.0599421169999914,
          1.0715471089999937,
          1.068564814000183
        ]
      },
      "decode": {
        "1": [
          0.004856118000134302,
          0.008147104000045147,
          0.005622599999924205,
          0.005376747999889631,
          0.004871906000062154,
          0.005993066000200997,
          0.005050143000062235,
          0.005188194000083968,
          0.00486280999984956,
          0.00625807800020084,
          0.004862519999960568,
          0.004931969999915964,
          0.004859692999843901,
          0.006450473999848327,
          0.00475369400010095,
          0.005184564000046521,
          0.0050125140001000545,
          0.004768849000129194,
          0.004726917999960278,
          0.006195144000002983,
          0.004861926000103267,
          0.0049627670000518265,
          0.006416584000135117,
          0.0049529820000771,
          0.0048980259998643305,
          0.005180634999987888,
          0.008133605000011812,
          0.004853024000112782,
          0.004928302000053009,
          0.004796475000148348
        ],
        "4": [
          0.005065649000016492,
          0.008038765000037529,
          0.005112178999979733,
          0.00522422800008826,
          0.00710529699995277,
          0.006155529000125171,
          0.005403909000051499,
          0.0049705119999998715,
          0.005052947999956814,
          0.006816581999828486,
          0.0051036629999998695,
          0.005686780999894836,
          0.005244589000085398,
          0.006945043000087026,
          0.004857850000007602,
          0.005361730000004172,
          0.004964328999903955,
          0.00850946900004601,
          0.0048550519998116215,
          0.007393262000050527,
          0.005036119999886068,
          0.004948340999817447,
          0.006429367000009734,
          0.005058938000047419,
          0.005827374000091368,
          0.005071472999816251,
          0.007587669000031383,
          0.006723857999986649,
          0.005094850999967093,
          0.004945339999949283
        ],
        "16": [
          0.005806486999972549,
          0.007939820000046893,
          0.005050422000067556,
          0.005438905000119121,
          0.005349085000034393,
          0.008518591999973069,
          0.006807407999986026,
          0.005108275000111462,
          0.00529071699997985,
          0.006776255999966452,
          0.005514252000011766,
          0.005642615999931877,
          0.005522013000017978,
          0.007208647999959794,
          0.0073696960000688705,
          0.005160476000128256,
          0.005447416999913912,
          0.005533217999982298,
          0.005117097000038484,
          0.005696496999917144,
          0.005117656999800602,
          0.005200568000191197,
          0.012922255000148652,
          0.005075956999917253,
          0.0052089159999013646,
          0.005226871999866489,
          0.0064942940000491944,
          0.005103970000163827,
          0.00693594800009123,
          0.006763661999912074
        ],
        "64": [
          0.007385129000113011,
          0.009087830999988,
          0.007369096000047648,
          0.007424230000196985,
          0.007518767999954434,
          0.008150363000140715,
          0.007424642999922071,
          0.007387354000002233,
          0.007511407000038162,
          0.009084648999987621,
          0.0074833070000295265,
          0.007396733999939897,
          0.0074049039999408706,
          0.00801609399991321,
          0.007368507999899521,
          0.0075779430001148285,
          0.007365380000010191,
          0.00804542599985325,
          0.007210324999959994,
          0.007245447000059357,
          0.007362315999898783,
          0.007237419000148293,
          0.007325635999904989,
          0.007166403999917748,
          0.007357731999945827,
          0.0076040660001126525,
          0.008062976000019262,
          0.007533678000072541,
          0.007373336999989988,
          0.008844828000064808
        ],
        "256": [
          0.023112612999966586,
          0.02197584000009556,
          0.022130516000061107,
          0.022144446000083917,
          0.022668931999987763,
          0.022244785999873784,
          0.022226807000151894,
          0.02218508400005703,
          0.02307189499993001,
          0.02324398599989763,
          0.02219220500001029,
          0.02218415700008336,
          0.02265326199994888,
          0.022161748999906195,
          0.022011116000157926,
          0.022150599000042348,
          0.02211950700007037,
          0.022081386000081693,
          0.021904439000081766,
          0.021946543000012753,
          0.02189288299996406,
          0.022432793000007223,
          0.021939128999974855,
          0.021791886000073646,
          0.021820737999860285,
          0.022492114999977275,
          0.021882962000063344,
          0.02184193499988396,
          0.021773432999907527,
          0.022515457999816135
        ]
      }
    }



```python
# --- 実験 B: 本番実行であることの保証 ---
# SMOKE_TEST=False のとき、実際に使った水準が PRODUCTION_* 定数と一致することを
# アサーションで確認する。OOM でスキップした水準がある場合、その水準は
# experiment_b_raw に記録されないため、「成功した水準 + OOM した水準」が
# PRODUCTION_B_LEVELS と一致することを確認する。
if not SMOKE_TEST:
    assert B_LEVELS == PRODUCTION_B_LEVELS, (
        f"本番実行のはずが B_LEVELS が PRODUCTION_B_LEVELS と一致しない: {B_LEVELS}"
    )
    assert B_REPEATS == PRODUCTION_B_REPEATS, (
        f"本番実行のはずが B_REPEATS が PRODUCTION_B_REPEATS と一致しない: {B_REPEATS}"
    )
    assert B_PREFILL_LEN == PRODUCTION_B_PREFILL_LEN, (
        f"本番実行のはずが B_PREFILL_LEN が PRODUCTION_B_PREFILL_LEN と一致しない: {B_PREFILL_LEN}"
    )
    for _cond in ("prefill", "decode"):
        assert set(experiment_b_raw[_cond].keys()) | set(B_OOM_LEVELS) == set(PRODUCTION_B_LEVELS), (
            f"experiment_b_raw[{_cond!r}] の水準(OOM 分を含む)が PRODUCTION_B_LEVELS と"
            f"一致しない: 成功={sorted(experiment_b_raw[_cond].keys())}, OOM={B_OOM_LEVELS}"
        )
        for _b, _vals in experiment_b_raw[_cond].items():
            assert len(_vals) == PRODUCTION_B_REPEATS, (
                f"experiment_b_raw[{_cond!r}][{_b}] の反復回数が PRODUCTION_B_REPEATS と"
                f"一致しない: {len(_vals)}"
            )
    print(f"[OK] 実験 B は本番水準(B_LEVELS={B_LEVELS}, B_REPEATS={B_REPEATS}, "
          f"B_PREFILL_LEN={B_PREFILL_LEN})で実行された")
```

    [OK] 実験 B は本番水準(B_LEVELS=[1, 4, 16, 64, 256], B_REPEATS=30, B_PREFILL_LEN=512)で実行された



```python
# --- 実験 B: 前提条件 P0-a・P0-b の確認(7.1 節) ---
_b_successful_levels = [b for b in B_LEVELS if b not in B_OOM_LEVELS]
_b_p0a_by_condition = {}
_b_p0b_by_condition = {}
for _cond, _by_b in experiment_b_raw.items():
    for _b, _vals in _by_b.items():
        _key = f"{_cond}_B{_b}"
        _b_p0a_by_condition[_key] = compute_p0a(_vals)
        _b_p0b_by_condition[_key] = compute_p0b(_vals)
        print(
            f"{_key:14s} P0-a: 先頭={_b_p0a_by_condition[_key]['head_mean']:.5f}s "
            f"末尾={_b_p0a_by_condition[_key]['tail_mean']:.5f}s "
            f"差={_b_p0a_by_condition[_key]['diff']:.5f}s "
            f"閾値={_b_p0a_by_condition[_key]['threshold']:.5f}s "
            f"成立={_b_p0a_by_condition[_key]['holds']}"
        )
        print(
            f"{'':14s} P0-b: 平均={_b_p0b_by_condition[_key]['mean']:.5f}s "
            f"標準誤差={_b_p0b_by_condition[_key]['stderr']:.5f}s "
            f"比率={_b_p0b_by_condition[_key]['ratio']:.4f} "
            f"成立={_b_p0b_by_condition[_key]['holds']}"
        )

B_P0A_HOLDS = all(v["holds"] for v in _b_p0a_by_condition.values())
B_P0B_HOLDS = all(v["holds"] for v in _b_p0b_by_condition.values())
B_P0_HOLDS = B_P0A_HOLDS and B_P0B_HOLDS
print(f"\nP0-a 全条件成立: {B_P0A_HOLDS} / P0-b 全条件成立: {B_P0B_HOLDS} / "
      f"P0(P0-a かつ P0-b)成立: {B_P0_HOLDS}")

_b_y_prefill = [float(np.mean(experiment_b_raw["prefill"][b])) for b in _b_successful_levels]
_b_y_decode = [float(np.mean(experiment_b_raw["decode"][b])) for b in _b_successful_levels]
experiment_b_fit_prefill = fit_power_law_exponent(_b_successful_levels, _b_y_prefill)
experiment_b_fit_decode = fit_power_law_exponent(_b_successful_levels, _b_y_decode)
print(f"b(prefill) = {experiment_b_fit_prefill.exponent:.4f} +- {experiment_b_fit_prefill.exponent_stderr:.4f}")
print(f"b(decode)  = {experiment_b_fit_decode.exponent:.4f} +- {experiment_b_fit_decode.exponent_stderr:.4f}")

_b_delta_b = experiment_b_fit_decode.exponent - experiment_b_fit_prefill.exponent
_b_sigma_delta_b = float(
    np.sqrt(experiment_b_fit_decode.exponent_stderr**2 + experiment_b_fit_prefill.exponent_stderr**2)
)
print(f"Delta_b_B = {_b_delta_b:.4f}, sigma = {_b_sigma_delta_b:.4f}, "
      f"Delta_b_B / sigma = {_b_delta_b / _b_sigma_delta_b:.2f}")

# 診断量: 演算強度の理論値、T4 公称帯域からの下限時間との比(判定基準ではない)。
_b_d_model = PROD_CONFIG["d_model"]
_b_weight_bytes = _b_d_model * _b_d_model * 4
_b_decode_ai = compute_arithmetic_intensity(2 * 1 * _b_d_model * _b_d_model, _b_weight_bytes + 1 * 2 * _b_d_model * 4)
print(f"[診断量] decode 1 層あたりの演算強度(概算): {_b_decode_ai:.3f} FLOPs/byte")

plot_log_log_fit(
    _b_successful_levels, _b_y_prefill, experiment_b_fit_prefill.exponent, experiment_b_fit_prefill.coefficient,
    label="prefill", xlabel="B (batch size)", ylabel="time (s)",
    title="Experiment B (smoke test, code verification only)",
)
plt.show()
```

    prefill_B1     P0-a: 先頭=0.00839s 末尾=0.00693s 差=0.00145s 閾値=0.00147s 成立=True
                   P0-b: 平均=0.00740s 標準誤差=0.00013s 比率=0.0181 成立=True
    prefill_B4     P0-a: 先頭=0.01644s 末尾=0.01629s 差=0.00016s 閾値=0.00055s 成立=True
                   P0-b: 平均=0.01654s 標準誤差=0.00005s 比率=0.0030 成立=True
    prefill_B16    P0-a: 先頭=0.06044s 末尾=0.06100s 差=0.00056s 閾値=0.00214s 成立=True
                   P0-b: 平均=0.06148s 標準誤差=0.00020s 比率=0.0032 成立=True
    prefill_B64    P0-a: 先頭=0.26149s 末尾=0.26117s 差=0.00032s 閾値=0.00693s 成立=True
                   P0-b: 平均=0.26513s 標準誤差=0.00063s 比率=0.0024 成立=True
    prefill_B256   P0-a: 先頭=1.06387s 末尾=1.06668s 差=0.00282s 閾値=0.03422s 成立=True
                   P0-b: 平均=1.07977s 標準誤差=0.00312s 比率=0.0029 成立=True
    decode_B1      P0-a: 先頭=0.00621s 末尾=0.00486s 差=0.00135s 閾値=0.00182s 成立=True
                   P0-b: 平均=0.00540s 標準誤差=0.00017s 比率=0.0308 成立=True
    decode_B4      P0-a: 先頭=0.00607s 末尾=0.00559s 差=0.00048s 閾値=0.00214s 成立=True
                   P0-b: 平均=0.00582s 標準誤差=0.00020s 比率=0.0336 成立=True
    decode_B16     P0-a: 先頭=0.00627s 末尾=0.00627s 差=0.00000s 閾値=0.00319s 成立=True
                   P0-b: 平均=0.00614s 標準誤差=0.00029s 比率=0.0474 成立=True
    decode_B64     P0-a: 先頭=0.00795s 末尾=0.00792s 差=0.00003s 閾値=0.00105s 成立=True
                   P0-b: 平均=0.00764s 標準誤差=0.00010s 比率=0.0126 成立=True
    decode_B256    P0-a: 先頭=0.02241s 末尾=0.02204s 差=0.00036s 閾値=0.00079s 成立=True
                   P0-b: 平均=0.02223s 標準誤差=0.00007s 比率=0.0032 成立=True
    
    P0-a 全条件成立: True / P0-b 全条件成立: True / P0(P0-a かつ P0-b)成立: True
    b(prefill) = 0.9190 +- 0.0531
    b(decode)  = 0.2238 +- 0.0842
    Delta_b_B = -0.6951, sigma = 0.0995, Delta_b_B / sigma = -6.98
    [診断量] decode 1 層あたりの演算強度(概算): 0.496 FLOPs/byte



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/010_kv_cache_and_inference_compute/output_38_1.png)
    



```python
# --- 実験 B: 本番水準への外挿(CLAUDE.md 第 1 段階要件 1) ---
_b_calls_per_level = PRODUCTION_B_REPEATS + WARMUP_REPEATS  # ウォームアップ WARMUP_REPEATS 回を含む
_b_extrapolated_prefill = sum(
    experiment_b_fit_prefill.coefficient * b**experiment_b_fit_prefill.exponent
    for b in PRODUCTION_B_LEVELS
) * _b_calls_per_level
_b_extrapolated_decode = sum(
    experiment_b_fit_decode.coefficient * b**experiment_b_fit_decode.exponent
    for b in PRODUCTION_B_LEVELS
) * _b_calls_per_level
EXPERIMENT_B_EXTRAPOLATED_SECONDS = _b_extrapolated_prefill + _b_extrapolated_decode
print(f"外挿: prefill {_b_extrapolated_prefill:.1f} 秒 + decode {_b_extrapolated_decode:.1f} 秒 = "
      f"{EXPERIMENT_B_EXTRAPOLATED_SECONDS:.1f} 秒"
      f"(水準 {PRODUCTION_B_LEVELS}、水準あたり {_b_calls_per_level} 回)")
```

    外挿: prefill 43.0 秒 + decode 1.5 秒 = 44.5 秒(水準 [1, 4, 16, 64, 256]、水準あたり 33 回)



```python
# --- 実験 B: 本番水準での必要メモリ量の閉形式による見積もり(較正は本番と同じ
# スケールで行うという規約に従い、閉形式で計算できる量は本番実行時まで未知にしない) ---
T4_MEMORY_BYTES = 16 * (1024**3)  # T4 GPU の公称メモリ容量(16 GB)

_b_prod_attention_score_bytes_per_layer = (
    PRODUCTION_B_LEVELS[-1] * PROD_CONFIG["num_heads"] * PRODUCTION_B_PREFILL_LEN**2 * 4
)
_b_prod_attention_score_bytes_total = _b_prod_attention_score_bytes_per_layer * PROD_CONFIG["num_layers"]
print(
    f"本番水準(batch_size={PRODUCTION_B_LEVELS[-1]}, prefill_len={PRODUCTION_B_PREFILL_LEN})での"
    f"Attention スコア行列のメモリ量:"
)
print(f"  1 層あたり: {_b_prod_attention_score_bytes_per_layer / 1e9:.2f} GB")
print(f"  {PROD_CONFIG['num_layers']} 層合計: {_b_prod_attention_score_bytes_total / 1e9:.2f} GB")
print(f"  T4 GPU 容量(16 GB)に対する比率: {_b_prod_attention_score_bytes_total / T4_MEMORY_BYTES:.1%}")
if _b_prod_attention_score_bytes_total > T4_MEMORY_BYTES:
    print("[警告] Attention スコア行列だけで T4 GPU の容量を超える。"
          "PRODUCTION_B_LEVELS の見直しが必要。")
else:
    print("[OK] Attention スコア行列のメモリ量は T4 GPU の容量に収まる見込み"
          "(重み・活性化・オプティマイザ状態は含まない概算であることに注意)。")
```

    本番水準(batch_size=256, prefill_len=512)でのAttention スコア行列のメモリ量:
      1 層あたり: 2.15 GB
      4 層合計: 8.59 GB
      T4 GPU 容量(16 GB)に対する比率: 50.0%
    [OK] Attention スコア行列のメモリ量は T4 GPU の容量に収まる見込み(重み・活性化・オプティマイザ状態は含まない概算であることに注意)。




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/03_efficient_training/010_kv_cache_and_inference_compute.ipynb
