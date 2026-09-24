---
title: "KV キャッシュと推論の計算量(KV Cache and Inference Compute)(実装・実験編 3/4)"
---

この記事は後編(実装・実験編 3/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/010_kv_cache_and_inference_compute-practice-2)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/010_kv_cache_and_inference_compute-practice-4)。

### 7.4 実験 C: Key / Value ヘッド数の削減による高速化の系列長依存性

- **検証すること**: Key / Value ヘッド数の削減による decode の高速化は、
  系列長が長いほど大きい。
- **対比量**: $r(T) = t_{g=h}(T) / t_{g=1}(T)$($t_{g=h}$ は通常の多頭注意機構
  ($g=h$)の生成時間、$t_{g=1}$ は MQA($g=1$)の生成時間)としたときの
  $\Delta r = r(T_{\max}) - r(T_{\min})$。$T_{\min}$・$T_{\max}$ は等比の刻みの
  両端とする。
  - **介入の直接作用点**: Key / Value ヘッド数の削減が直接作用するのは、
    1 decode ステップあたりの KV キャッシュの読み出しバイト数(3.2・3.4 節)
    である。対比量 $r(T)$ は「生成全体の時間比」という下流の量であり、direct な
    作用点(1 ステップの読み出しバイト数)からは 1 段階離れている(生成全体の
    時間には、Key/Value 削減の影響を受けない Query 射影・出力射影・
    Feed-Forward Network の計算も含まれるため)。
- **判定基準**: $\Delta r > 2\sigma_{\Delta r}$ なら支持、$< -2\sigma_{\Delta r}$
  なら反証、それ以外は判定不能。$\sigma_{\Delta r}$ は反復計測から誤差伝播で
  導く。$r = a/b$(比)の相対標準誤差は、$a$・$b$ それぞれの相対標準誤差の
  二乗和の平方根

  $$
  \frac{\sigma_r}{r} = \sqrt{\left(\frac{\sigma_a / \sqrt{n}}{a}\right)^2
      + \left(\frac{\sigma_b / \sqrt{n}}{b}\right)^2}
  $$

  で求め($\sigma_a$・$\sigma_b$ は反復計測の標本標準偏差、$n$ は反復回数)、
  $T_{\min}$・$T_{\max}$ それぞれで得た $\sigma_{r(T_{\min})}$・
  $\sigma_{r(T_{\max})}$ を $\sigma_{\Delta r} = \sqrt{\sigma_{r(T_{\min})}^2
  + \sigma_{r(T_{\max})}^2}$ として合成する。
- この実験は学習を伴わない。重みはランダム初期化でよく、Key / Value ヘッド数だけが
  異なる同一構成のモデルで計測する。


```python
# --- 実験 C ---
# C_T_SMALL・C_T_LARGE・C_REPEATS はセットアップセル(SMOKE_TEST の定義の直後)で定義済み。
C_PROMPT_LENGTH = 4

timing_model_mqa_c = convert_attention_to_grouped_query(
    timing_model_a, num_key_value_heads=1, init="random", seed=0
).to(TIMING_DEVICE)
timing_model_mqa_c.eval()


def warmup_decode_generate(model, t_new_tokens, prompt_length, device):
    assert prompt_length + t_new_tokens <= model.max_sequence_length, (
        f"prompt_length({prompt_length}) + t_new_tokens({t_new_tokens}) が "
        f"model.max_sequence_length({model.max_sequence_length}) を超えている"
    )
    for _ in range(WARMUP_REPEATS):
        x = torch.randint(0, PROD_CONFIG["vocabulary_size"], (1, prompt_length), device=device)
        with torch.no_grad():
            model.generate(x, max_new_tokens=t_new_tokens, temperature=0.0, use_cache=True)
        sync(device)


def measure_decode_generate_once(model, t_new_tokens, prompt_length, device):
    x = torch.randint(0, PROD_CONFIG["vocabulary_size"], (1, prompt_length), device=device)
    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(x, max_new_tokens=t_new_tokens, temperature=0.0, use_cache=True)
    sync(device)
    # t_new_tokens をそのまま返す(呼び出し側が別途 C_T_SMALL/C_T_LARGE を再度参照するの
    # ではなく、計測ループが実際に使った値を関数自身の返り値として記録するため。
    # C_T_SMALL/C_T_LARGE 以外の値を誤って渡す呼び出し側の配線ミスも、この返り値を
    # PRODUCTION_C_T_SMALL/PRODUCTION_C_T_LARGE と照合すれば検出できる)。
    return t_new_tokens, time.perf_counter() - t0


_c_conditions = {
    "full_small": (timing_model_a, C_T_SMALL),
    "mqa_small": (timing_model_mqa_c, C_T_SMALL),
    "full_large": (timing_model_a, C_T_LARGE),
    "mqa_large": (timing_model_mqa_c, C_T_LARGE),
}

# ウォームアップは条件ごとにまとめて計測ループの前に行う(WARMUP_REPEATS 回、P0)。
for _name, (_model, _t) in _c_conditions.items():
    warmup_decode_generate(_model, _t, C_PROMPT_LENGTH, TIMING_DEVICE)

# 計測は for 反復: for 条件: の順で行う(for 条件: for 反復: ではない)。1 つの条件の
# 反復をまとめて連続実行すると、一過性の擾乱が特定の条件に集中して入り、P0(変動係数)
# が不成立になりやすい。反復をまたいで条件を切り替えることで、擾乱が全条件に均等に
# 散るようにする。
experiment_c_raw: dict[str, list[float]] = {name: [] for name in _c_conditions}
experiment_c_t_used: dict[str, int] = {}
for _rep in range(C_REPEATS):
    for _name, (_model, _t) in _c_conditions.items():
        _t_actual, _elapsed = measure_decode_generate_once(_model, _t, C_PROMPT_LENGTH, TIMING_DEVICE)
        experiment_c_raw[_name].append(_elapsed)
        experiment_c_t_used[_name] = _t_actual  # 条件内で不変(反復ごとに再確認するだけ)

for _name, _vals in experiment_c_raw.items():
    _arr = np.array(_vals)
    print(f"{_name:12s} T={experiment_c_t_used[_name]:4d} mean={_arr.mean():.5f}s "
          f"cv={_arr.std() / _arr.mean():.4f}")

print("\n[判定の一次情報 / experiment_c_raw, experiment_c_t_used]")
print(json.dumps({"times": experiment_c_raw, "t_used": experiment_c_t_used}, ensure_ascii=False, indent=2))
```

    full_small   T=  64 mean=0.34262s cv=0.1197
    mqa_small    T=  64 mean=0.35594s cv=0.1177
    full_large   T=4096 mean=21.66006s cv=0.0139
    mqa_large    T=4096 mean=22.89708s cv=0.0170
    
    [判定の一次情報 / experiment_c_raw, experiment_c_t_used]
    {
      "times": {
        "full_small": [
          0.3914319030000115,
          0.3209350820000054,
          0.31778289199996834,
          0.335229844999958,
          0.354217725999888,
          0.31472350099988944,
          0.34140486299997974,
          0.31679277700004604,
          0.31100359699985347,
          0.445524862999946,
          0.32753599399984523,
          0.3153954540000541,
          0.41934152399971936,
          0.30963574399993377,
          0.31832674799989036
        ],
        "mqa_small": [
          0.45853652100004183,
          0.3373663370000486,
          0.33982854700002463,
          0.34494167599996217,
          0.3362102280000272,
          0.3370156240000597,
          0.3403603839999505,
          0.3330021819999729,
          0.3434773270000733,
          0.435223581000173,
          0.329177264000009,
          0.32585400800007847,
          0.41983300500032783,
          0.329054885000005,
          0.32926887400026317
        ],
        "full_large": [
          21.375361457000054,
          21.789071597999964,
          22.20800050300022,
          22.059733551999898,
          21.652500678000024,
          21.76936572799991,
          21.207845258999896,
          21.651807596000026,
          21.76480822099984,
          21.078534248000096,
          21.599899537999818,
          21.6364923000001,
          21.373879745999602,
          21.68306873199981,
          22.05053088500017
        ],
        "mqa_large": [
          23.432683795000003,
          23.169509930999993,
          23.403145257999995,
          23.361007264000136,
          22.92436603400006,
          23.11706805599988,
          22.82038496999985,
          22.739211352999973,
          22.58883995700012,
          22.656757438999875,
          22.717791871999907,
          22.255114854999647,
          22.656413948000136,
          23.39074067499996,
          22.223216445000162
        ]
      },
      "t_used": {
        "full_small": 64,
        "mqa_small": 64,
        "full_large": 4096,
        "mqa_large": 4096
      }
    }



```python
# --- 実験 C: 本番実行であることの保証 ---
# C_T_SMALL・C_T_LARGE(セットアップセルで定義した設定値)と PRODUCTION_* の比較だけでは、
# 「time_decode_generate の呼び出しに実際に渡された値」が PRODUCTION_* と一致するかを
# 検証できない(設定値どうしの比較は、設定値の定義自体が PRODUCTION_* を参照している
# ため常に成立する自己参照になる)。experiment_c_t_used は関数の返り値であり、実際に
# 計測ループが使った生成長そのものなので、これを PRODUCTION_* と照合する。
if not SMOKE_TEST:
    assert (C_T_SMALL, C_T_LARGE) == (PRODUCTION_C_T_SMALL, PRODUCTION_C_T_LARGE), (
        "本番実行のはずが (C_T_SMALL, C_T_LARGE) が "
        f"(PRODUCTION_C_T_SMALL, PRODUCTION_C_T_LARGE) と一致しない: ({C_T_SMALL}, {C_T_LARGE})"
    )
    assert C_REPEATS == PRODUCTION_C_REPEATS, (
        f"本番実行のはずが C_REPEATS が PRODUCTION_C_REPEATS と一致しない: {C_REPEATS}"
    )
    _c_expected_t = {
        "full_small": PRODUCTION_C_T_SMALL,
        "mqa_small": PRODUCTION_C_T_SMALL,
        "full_large": PRODUCTION_C_T_LARGE,
        "mqa_large": PRODUCTION_C_T_LARGE,
    }
    for _name, _t in experiment_c_t_used.items():
        assert _t == _c_expected_t[_name], (
            f"experiment_c_t_used[{_name!r}](time_decode_generate に実際に渡された生成長)が "
            f"期待値({_c_expected_t[_name]})と一致しない: {_t}。C_T_SMALL/C_T_LARGE 以外の値が"
            "渡された疑いがある。"
        )
    for _name, _vals in experiment_c_raw.items():
        assert len(_vals) == PRODUCTION_C_REPEATS, (
            f"experiment_c_raw[{_name!r}] の反復回数が PRODUCTION_C_REPEATS と一致しない: {len(_vals)}"
        )
    print(f"[OK] 実験 C は本番水準(実際に使われた生成長: {experiment_c_t_used}, "
          f"C_REPEATS={C_REPEATS})で実行された")
```

    [OK] 実験 C は本番水準(実際に使われた生成長: {'full_small': 64, 'mqa_small': 64, 'full_large': 4096, 'mqa_large': 4096}, C_REPEATS=15)で実行された



```python
# --- 実験 C: 前提条件 P0-a・P0-b の確認(7.1 節) ---
_c_p0a_by_condition = {}
_c_p0b_by_condition = {}
for _name, _vals in experiment_c_raw.items():
    _c_p0a_by_condition[_name] = compute_p0a(_vals)
    _c_p0b_by_condition[_name] = compute_p0b(_vals)
    print(
        f"{_name:12s} P0-a: 先頭={_c_p0a_by_condition[_name]['head_mean']:.5f}s "
        f"末尾={_c_p0a_by_condition[_name]['tail_mean']:.5f}s "
        f"差={_c_p0a_by_condition[_name]['diff']:.5f}s "
        f"閾値={_c_p0a_by_condition[_name]['threshold']:.5f}s "
        f"成立={_c_p0a_by_condition[_name]['holds']}"
    )
    print(
        f"{'':12s} P0-b: 平均={_c_p0b_by_condition[_name]['mean']:.5f}s "
        f"標準誤差={_c_p0b_by_condition[_name]['stderr']:.5f}s "
        f"比率={_c_p0b_by_condition[_name]['ratio']:.4f} "
        f"成立={_c_p0b_by_condition[_name]['holds']}"
    )

C_P0A_HOLDS = all(v["holds"] for v in _c_p0a_by_condition.values())
C_P0B_HOLDS = all(v["holds"] for v in _c_p0b_by_condition.values())
C_P0_HOLDS = C_P0A_HOLDS and C_P0B_HOLDS
print(f"\nP0-a 全条件成立: {C_P0A_HOLDS} / P0-b 全条件成立: {C_P0B_HOLDS} / "
      f"P0(P0-a かつ P0-b)成立: {C_P0_HOLDS}")

_c_full_small = np.array(experiment_c_raw["full_small"])
_c_mqa_small = np.array(experiment_c_raw["mqa_small"])
_c_full_large = np.array(experiment_c_raw["full_large"])
_c_mqa_large = np.array(experiment_c_raw["mqa_large"])


def _relative_stderr(arr: np.ndarray) -> float:
    return float((arr.std(ddof=1) / np.sqrt(len(arr))) / arr.mean())


r_small = float(_c_full_small.mean() / _c_mqa_small.mean())
r_large = float(_c_full_large.mean() / _c_mqa_large.mean())
delta_r = r_large - r_small

sigma_r_small = r_small * np.sqrt(_relative_stderr(_c_full_small) ** 2 + _relative_stderr(_c_mqa_small) ** 2)
sigma_r_large = r_large * np.sqrt(_relative_stderr(_c_full_large) ** 2 + _relative_stderr(_c_mqa_large) ** 2)
sigma_delta_r = float(np.sqrt(sigma_r_small**2 + sigma_r_large**2))

print(f"r(T={C_T_SMALL}) = {r_small:.4f}, r(T={C_T_LARGE}) = {r_large:.4f}")
print(f"Delta_r = {delta_r:.4f}, sigma_Delta_r = {sigma_delta_r:.4f}, "
      f"Delta_r / sigma = {delta_r / sigma_delta_r:.2f}")
```

    full_small   P0-a: 先頭=0.34338s 末尾=0.34910s 差=0.00572s 閾値=0.08490s 成立=True
                 P0-b: 平均=0.34262s 標準誤差=0.01096s 比率=0.0320 成立=True
    mqa_small    P0-a: 先頭=0.37858s 末尾=0.35939s 差=0.01919s 閾値=0.08676s 成立=True
                 P0-b: 平均=0.35594s 標準誤差=0.01120s 比率=0.0315 成立=True
    full_large   P0-a: 先頭=21.79081s 末尾=21.70249s 差=0.08832s 閾値=0.62392s 成立=True
                 P0-b: 平均=21.66006s 標準誤差=0.08055s 比率=0.0037 成立=True
    mqa_large    P0-a: 先頭=23.33511s 末尾=22.75679s 差=0.57832s 閾値=0.80447s 成立=True
                 P0-b: 平均=22.89708s 標準誤差=0.10386s 比率=0.0045 成立=True
    
    P0-a 全条件成立: True / P0-b 全条件成立: True / P0(P0-a かつ P0-b)成立: True
    r(T=64) = 0.9626, r(T=4096) = 0.9460
    Delta_r = -0.0166, sigma_Delta_r = 0.0435, Delta_r / sigma = -0.38



```python
# --- 実験 C: 本番水準への外挿(CLAUDE.md 第 1 段階要件 1) ---
# 実験 C 自体は T が 2 点(smoke)のみで独自にべき乗則をあてはめるには点数が足りない。
# 実験 C が測るのは実験 A の「キャッシュあり」と同じ計測(KV キャッシュを用いた
# generate())であるため、実験 A で得たべき乗則あてはめ(experiment_a_fit_cache)を
# 流用して T_min・T_max での時間を外挿する。
_c_calls_per_level = PRODUCTION_C_REPEATS + WARMUP_REPEATS  # ウォームアップ WARMUP_REPEATS 回を含む
_c_extrapolated_per_level = sum(
    experiment_a_fit_cache.coefficient * t**experiment_a_fit_cache.exponent
    for t in (PRODUCTION_C_T_SMALL, PRODUCTION_C_T_LARGE)
)
EXPERIMENT_C_EXTRAPOLATED_SECONDS = _c_extrapolated_per_level * _c_calls_per_level * 2  # 多頭注意機構・MQA の 2 条件
print(f"外挿: T={PRODUCTION_C_T_SMALL}・{PRODUCTION_C_T_LARGE} x 2 条件"
      f"(多頭注意機構・MQA) x {_c_calls_per_level} 回 = "
      f"{EXPERIMENT_C_EXTRAPOLATED_SECONDS:.1f} 秒"
      "(実験 A のキャッシュありのべき乗則あてはめを流用)")
```

    外挿: T=64・4096 x 2 条件(多頭注意機構・MQA) x 18 回 = 802.2 秒(実験 A のキャッシュありのべき乗則あてはめを流用)


### 7.5 実験 D: GQA への変換における平均プール初期化の効果

- **検証すること**: 多頭注意機構から GQA へ変換する際、平均プール初期化は
  ランダム初期化より、同一の追加学習ステップ数で低い bits-per-byte に到達する。
- **条件(主判定)**: GQA・平均プール初期化 + 追加学習 / GQA・ランダム初期化 +
  追加学習。この 2 条件は非埋め込みパラメータ数が完全に一致する(6 節で確認済み)。
- **条件(診断量)**: 元の多頭注意機構チェックポイント(追加学習なし)、
  MQA・平均プール初期化 + 追加学習。多頭注意機構との比較は Key / Value 射影の
  縮小により非埋め込みパラメータ数が揃わないため、**揃えられないことをここに
  明記したうえで参考値として報告する**。
- **Key / Value ヘッド数**: 元モデルのヘッド数 $h=8$ に対し、GQA は $g=4$、MQA は
  $g=1$。
- **対比量**: $\Delta\mathrm{bpb} = \mathrm{bpb}_{\text{random}} -
  \mathrm{bpb}_{\text{mean pool}}$($\mathrm{bpb}_{\text{random}}$ はランダム初期化、
  $\mathrm{bpb}_{\text{mean pool}}$ は平均プール初期化。いずれもシード平均、
  最終ステップの値)。
  - **介入の直接作用点**: この介入(初期化方式)が直接作用するのは初期化直後の
    重み(および、そこから計算される初期化直後の bits-per-byte)である。対比量は
    そこから追加学習を経た下流の量であり、直接の作用点からは 1 段階以上離れて
    いる。この距離を埋めるため、**初期化直後(追加学習ステップ数 0)の
    bits-per-byte を診断量として併記する**。
- **判定基準**: $\Delta\mathrm{bpb} > 2\sigma_{\Delta\mathrm{bpb}}$ なら支持、
  $< -2\sigma_{\Delta\mathrm{bpb}}$ なら反証、それ以外は判定不能。
  $\sigma_{\Delta\mathrm{bpb}} = \sqrt{\sigma_{\text{random}}^2/n +
  \sigma_{\text{mean pool}}^2/n}$($\sigma_{\text{random}}$・$\sigma_{\text{mean pool}}$
  は各条件のシード間標準偏差)、$n$ はシード数(3)。
- **前提条件**:
  - P1: 読み込んだ元チェックポイントの bits-per-byte が 008 の記録値(1.6622、
    008 8.1 節)と相対誤差 1% 以内で一致すること。
  - P2: 全条件で追加学習が実際に進んでいること。単一ステップの訓練損失は
    ミニバッチの引きによる雑音が大きいため(1 ステップだけの比較では、シードに
    よる違いが学習の進行ではなく最初のバッチの引きを反映してしまう)、**最初の
    数ステップの平均訓練損失と最後の数ステップの平均訓練損失を比較する**
    (`P2_AVERAGING_WINDOW`ステップ分の平均、最後の平均が最初の平均より低ければ
    進行とみなす)。
- **モデル**: `kojikojiprg/ai-theories-small-gpt-en`を起点とする(5.3 節で
  読み込み済み)。トークナイザは`kojikojiprg/ai-theories-tokenizer-en`
  (同梱されていないため)。
- **追加学習の最適化手法**: 007 で導入した AdamW・warmup + cosine スケジュールでは
  なく、006 と同じ Adam・固定学習率(3e-4)を使う。少数ステップの追加学習
  (uptraining)であり、両条件(平均プール初期化・ランダム初期化)に同一の
  最適化設定を等しく適用するため、この選択自体は対比量の成立に影響しない。
- **評価**: 006 の`evaluate_bits_per_byte()`と非重複の評価窓を用いる。
  `d_eval_text`(5.3 節)は`train_text`の内側から切り出した held-out スライスで
  あり、`val_text_008`(008 の検証集合、P1 の検証に使う)とも
  `d_train_text`(この実験の追加学習データ)とも重複しない。全条件で評価集合
  (`_full_eval_windows`)・追加学習ステップ数が完全に同一であることをアサーションで
  確認する。
  - **主判定への影響なし、診断量は楽観的に出ることの明記**: `d_eval_text`は
    `train_text`(008 が`base_model`の学習に使ったデータ)の内側から切り出した
    スライスであり、`base_model`は既に`d_eval_text`の内容を学習時に見ている。
    主判定(平均プール初期化 対 ランダム初期化)は両条件で評価集合が同一であり
    この影響は相殺されるため結論に影響しない。一方、診断量である
    初期化直後の bits-per-byte、および`D_ORIGINAL_BPB`(元の多頭注意機構
    チェックポイントの`d_eval_text`上の bits-per-byte)は、真に未知のテキストで
    評価した場合より **楽観的(低い)値** になっている可能性が高い。これらの
    診断量の絶対値に基づく主張はしない。


```python
# --- 実験 D: 前提条件 P1(元チェックポイントの品質確認) ---
_p1_val_ids = encode_corpus(tokenizer, val_text_008)
_p1_eval_windows, _p1_eval_mask = make_evaluation_windows(_p1_val_ids, PROD_CONFIG["sequence_length"])
_p1_eval_bytes = len(val_text_008.encode("utf-8"))
D_P1_BPB = evaluate_bits_per_byte(base_model, _p1_eval_windows, _p1_eval_mask, _p1_eval_bytes, DEVICE, batch_size=16)
D_P1_REFERENCE_BPB = 1.6622  # 008 8.1 節の本番実行結果
D_P1_RELATIVE_ERROR = abs(D_P1_BPB - D_P1_REFERENCE_BPB) / D_P1_REFERENCE_BPB
print(f"P1: 元チェックポイントの bits-per-byte = {D_P1_BPB:.4f}"
      f"(008 記録値 {D_P1_REFERENCE_BPB}、相対誤差 {D_P1_RELATIVE_ERROR:.2%})")
D_P1_HOLDS = D_P1_RELATIVE_ERROR <= 0.01
print(f"P1 成立: {D_P1_HOLDS}")
```

    P1: 元チェックポイントの bits-per-byte = 1.6681(008 記録値 1.6622、相対誤差 0.35%)
    P1 成立: True


#### 7.5 節の注記(後日の調査による): 前提条件 P1 の参照値の出典

**この注記は、本番実行の後に判明した事実の記録である。前提条件 P1 の宣言(参照値 1.6622、相対誤差 1% 以内)と
その判定は変更しない。**

- P1 の参照値 1.6622 は「008 8.1 節の本番実行結果」として転記したが、008 の本番モデルの値ではなかった。008 の
  8.1 節の較正の表にある、学習率 x4.0 の較正実行(学習率 1.2e-3、seed 0、gradient clipping なし)の最終
  bits-per-byte である。008 の本番モデルは、較正で決めた gradient clipping 閾値を使って seed 42 で別途学習した
  ものであり、較正実行とは別のモデルである。出典の記述が誤っていた。
- 008 が本番モデルについて印字した 1.6707 も、アップロードした重みの値ではない(学習途中の 2,000 ステップ時点の
  評価値である。008 の 5.10 節の注記を参照)。アップロードした重み(2,181 ステップ時点)の、同じ検証集合での
  値は 1.6681(1.668067、013 のセル出力)であり、これが P1 の参照値として本来使うべき値である。
- 上のセルの実測値 1.6681 は、正しい参照値 1.6681 と印字した桁で一致する。誤った参照値 1.6622 との比較(相対
  誤差 0.35%)でも、正しい参照値との比較でも、相対誤差 1% 以内であることは変わらない。したがって P1 の成否
  (成立)は変わらない。


```python
# --- 実験 D: 条件・学習ループの準備 ---
# D_NUM_STEPS はセットアップセル(SMOKE_TEST の定義の直後)で定義済み。
D_BATCH_SIZE = 8
D_SEEDS = [0, 1, 2]

# 全条件で共通の学習・評価データ(5.3 節で構築済み)。評価集合は _full_eval_windows /
# _full_eval_mask / _full_eval_bytes を全条件で共有する(下のアサーションで確認)。
# 本番は d_train_text 全体(SMOKE_D_TRAIN_TEXT_CHARS はセットアップセルで定義済み)。
D_TRAIN_TEXT_CHARS = SMOKE_D_TRAIN_TEXT_CHARS if SMOKE_TEST else len(d_train_text)
d_train_ids = encode_corpus(tokenizer, d_train_text[:D_TRAIN_TEXT_CHARS])
print(f"d_train_ids: {len(d_train_ids):,} トークン(smoke: 先頭 {D_TRAIN_TEXT_CHARS:,} 文字)")
print(f"評価窓: {_full_eval_windows.shape[0]} 個(全条件で共有)")

# 冗長な再計算の排除(7.6 節で検証): このセルで 1 回だけ符号化した d_train_ids・
# _full_eval_windows・_full_eval_mask を、以降の全条件・全シードの学習・評価ループで
# そのまま参照する(再符号化しない)。その証拠として、基となるメモリ領域のポインタを
# ここで記録しておく。
_d_train_ids_data_ptr_at_creation = d_train_ids.data_ptr()
_d_eval_windows_data_ptr_at_creation = _full_eval_windows.data_ptr()
_d_eval_mask_data_ptr_at_creation = _full_eval_mask.data_ptr()


def build_condition_model(num_key_value_heads: int, init: str, seed: int | None) -> GPTLanguageModel:
    return convert_attention_to_grouped_query(base_model, num_key_value_heads, init, seed=seed)


D_CONDITIONS = ["gqa_mean_pool", "gqa_random", "mqa_mean_pool_diagnostic"]
experiment_d_raw: dict[str, dict[str, list]] = {cond: {"seeds": []} for cond in D_CONDITIONS}
```

    d_train_ids: 5,836,675 トークン(smoke: 先頭 22,543,743 文字)
    評価窓: 469 個(全条件で共有)



```python
# --- 実験 D: 診断条件(元の多頭注意機構チェックポイント、追加学習なし) ---
D_ORIGINAL_BPB = evaluate_bits_per_byte(
    base_model, _full_eval_windows, _full_eval_mask, _full_eval_bytes, DEVICE, batch_size=16
)
print(f"[診断量] 元の多頭注意機構チェックポイント(追加学習なし)の d_eval_text 上の bpb: "
      f"{D_ORIGINAL_BPB:.4f}")
print("[注記] このモデルは GQA/MQA と非埋め込みパラメータ数が揃わないため、参考値として扱う。")
```

    [診断量] 元の多頭注意機構チェックポイント(追加学習なし)の d_eval_text 上の bpb: 1.3370
    [注記] このモデルは GQA/MQA と非埋め込みパラメータ数が揃わないため、参考値として扱う。



```python
# --- 実験 D: 主判定 2 条件 + 診断条件(MQA 平均プール)の学習ループ ---
_d_specs = {
    "gqa_mean_pool": {"num_key_value_heads": 4, "init": "mean_pool"},
    "gqa_random": {"num_key_value_heads": 4, "init": "random"},
    "mqa_mean_pool_diagnostic": {"num_key_value_heads": 1, "init": "mean_pool"},
}

_d_param_counts: dict[str, int] = {}
_d_trained_model_for_upload = None  # gqa_mean_pool, seed 0 を代表チェックポイントとして残す

# 前提条件 P2: 単一ステップの訓練損失は雑音が大きいため、最初と最後の数ステップの
# 平均で比較する(D_NUM_STEPS より大きくならないようにクリップする)。
P2_AVERAGING_WINDOW = min(3, D_NUM_STEPS)

for cond_name, spec in _d_specs.items():
    for seed in D_SEEDS:
        model = build_condition_model(spec["num_key_value_heads"], spec["init"], seed=seed)
        _d_param_counts.setdefault(cond_name, count_non_embedding_parameters(model))

        initial_bpb = evaluate_bits_per_byte(
            model, _full_eval_windows, _full_eval_mask, _full_eval_bytes, DEVICE, batch_size=16
        )
        history = train_language_model(
            model, d_train_ids, _full_eval_windows, _full_eval_mask, _full_eval_bytes,
            num_steps=D_NUM_STEPS, batch_size=D_BATCH_SIZE, sequence_length=PROD_CONFIG["sequence_length"],
            learning_rate=3e-4, eval_interval=D_NUM_STEPS, device=DEVICE, seed=seed,
        )
        final_bpb = history["eval_bits_per_byte"][-1]
        initial_loss_avg = float(np.mean(history["train_loss"][:P2_AVERAGING_WINDOW]))
        final_loss_avg = float(np.mean(history["train_loss"][-P2_AVERAGING_WINDOW:]))
        progressed = final_loss_avg < initial_loss_avg

        experiment_d_raw[cond_name]["seeds"].append({
            "seed": seed,
            "initial_bpb": initial_bpb,
            "final_bpb": final_bpb,
            "initial_train_loss": history["train_loss"][0],
            "final_train_loss": history["train_loss"][-1],
            "initial_train_loss_avg": initial_loss_avg,
            "final_train_loss_avg": final_loss_avg,
            "progressed": progressed,
            # train_language_model が実際に何ステップ学習したか(D_NUM_STEPS を再参照した
            # 値ではなく、返ってきた学習履歴 history["train_loss"] の実際の要素数)。
            "train_loss_len": len(history["train_loss"]),
        })
        print(f"{cond_name} seed={seed}: initial_bpb={initial_bpb:.4f} -> final_bpb={final_bpb:.4f} "
              f"(train_loss_avg(先頭{P2_AVERAGING_WINDOW})={initial_loss_avg:.4f} -> "
              f"train_loss_avg(末尾{P2_AVERAGING_WINDOW})={final_loss_avg:.4f}, progressed={progressed})")

        if cond_name == "gqa_mean_pool" and seed == 0:
            _d_trained_model_for_upload = model

print("\n[判定の一次情報 / experiment_d_raw]")
print(json.dumps(experiment_d_raw, ensure_ascii=False, indent=2))
```

    gqa_mean_pool seed=0: initial_bpb=2.3261 -> final_bpb=1.5078 (train_loss_avg(先頭3)=5.9303 -> train_loss_avg(末尾3)=4.2665, progressed=True)
    gqa_mean_pool seed=1: initial_bpb=2.3261 -> final_bpb=1.5099 (train_loss_avg(先頭3)=6.0673 -> train_loss_avg(末尾3)=3.8080, progressed=True)
    gqa_mean_pool seed=2: initial_bpb=2.3261 -> final_bpb=1.5127 (train_loss_avg(先頭3)=5.8584 -> train_loss_avg(末尾3)=3.9217, progressed=True)
    gqa_random seed=0: initial_bpb=3.3507 -> final_bpb=2.0584 (train_loss_avg(先頭3)=8.9267 -> train_loss_avg(末尾3)=5.5564, progressed=True)
    gqa_random seed=1: initial_bpb=3.2944 -> final_bpb=2.0503 (train_loss_avg(先頭3)=9.0736 -> train_loss_avg(末尾3)=5.0715, progressed=True)
    gqa_random seed=2: initial_bpb=3.3171 -> final_bpb=2.0640 (train_loss_avg(先頭3)=8.9541 -> train_loss_avg(末尾3)=5.3273, progressed=True)
    mqa_mean_pool_diagnostic seed=0: initial_bpb=3.3982 -> final_bpb=1.8894 (train_loss_avg(先頭3)=8.9770 -> train_loss_avg(末尾3)=5.1439, progressed=True)
    mqa_mean_pool_diagnostic seed=1: initial_bpb=3.3982 -> final_bpb=1.8745 (train_loss_avg(先頭3)=8.7945 -> train_loss_avg(末尾3)=4.6513, progressed=True)
    mqa_mean_pool_diagnostic seed=2: initial_bpb=3.3982 -> final_bpb=1.9037 (train_loss_avg(先頭3)=8.6923 -> train_loss_avg(末尾3)=4.8722, progressed=True)
    
    [判定の一次情報 / experiment_d_raw]
    {
      "gqa_mean_pool": {
        "seeds": [
          {
            "seed": 0,
            "initial_bpb": 2.3261236092861584,
            "final_bpb": 1.5078047714021159,
            "initial_train_loss": 6.181218147277832,
            "final_train_loss": 4.311408042907715,
            "initial_train_loss_avg": 5.930291811625163,
            "final_train_loss_avg": 4.266503254572551,
            "progressed": true,
            "train_loss_len": 300
          },
          {
            "seed": 1,
            "initial_bpb": 2.3261236092861584,
            "final_bpb": 1.5099310357496754,
            "initial_train_loss": 6.445452690124512,
            "final_train_loss": 3.635012626647949,
            "initial_train_loss_avg": 6.067255179087321,
            "final_train_loss_avg": 3.8080221811930337,
            "progressed": true,
            "train_loss_len": 300
          },
          {
            "seed": 2,
            "initial_bpb": 2.3261236092861584,
            "final_bpb": 1.5126804422834266,
            "initial_train_loss": 6.08494758605957,
            "final_train_loss": 4.314051628112793,
            "initial_train_loss_avg": 5.858405749003093,
            "final_train_loss_avg": 3.921694358189901,
            "progressed": true,
            "train_loss_len": 300
          }
        ]
      },
      "gqa_random": {
        "seeds": [
          {
            "seed": 0,
            "initial_bpb": 3.350713482371572,
            "final_bpb": 2.0583867075426885,
            "initial_train_loss": 9.065722465515137,
            "final_train_loss": 5.590782642364502,
            "initial_train_loss_avg": 8.926735242207846,
            "final_train_loss_avg": 5.556432882944743,
            "progressed": true,
            "train_loss_len": 300
          },
          {
            "seed": 1,
            "initial_bpb": 3.2944031878778888,
            "final_bpb": 2.0502686506113186,
            "initial_train_loss": 9.310342788696289,
            "final_train_loss": 4.802374839782715,
            "initial_train_loss_avg": 9.073649406433105,
            "final_train_loss_avg": 5.071537494659424,
            "progressed": true,
            "train_loss_len": 300
          },
          {
            "seed": 2,
            "initial_bpb": 3.317141639970767,
            "final_bpb": 2.0640386198103804,
            "initial_train_loss": 9.575708389282227,
            "final_train_loss": 5.574122428894043,
            "initial_train_loss_avg": 8.954072952270508,
            "final_train_loss_avg": 5.327313264211019,
            "progressed": true,
            "train_loss_len": 300
          }
        ]
      },
      "mqa_mean_pool_diagnostic": {
        "seeds": [
          {
            "seed": 0,
            "initial_bpb": 3.3982385264051067,
            "final_bpb": 1.8893831799988046,
            "initial_train_loss": 9.25997543334961,
            "final_train_loss": 5.203737258911133,
            "initial_train_loss_avg": 8.977038065592447,
            "final_train_loss_avg": 5.143870035807292,
            "progressed": true,
            "train_loss_len": 300
          },
          {
            "seed": 1,
            "initial_bpb": 3.3982385264051067,
            "final_bpb": 1.8745062649319946,
            "initial_train_loss": 9.108227729797363,
            "final_train_loss": 4.4525957107543945,
            "initial_train_loss_avg": 8.794454574584961,
            "final_train_loss_avg": 4.65130090713501,
            "progressed": true,
            "train_loss_len": 300
          },
          {
            "seed": 2,
            "initial_bpb": 3.3982385264051067,
            "final_bpb": 1.903744076623776,
            "initial_train_loss": 9.091979026794434,
            "final_train_loss": 5.16807746887207,
            "initial_train_loss_avg": 8.692324002583822,
            "final_train_loss_avg": 4.872226715087891,
            "progressed": true,
            "train_loss_len": 300
          }
        ]
      }
    }



```python
# --- 実験 D: 本番実行であることの保証 ---
# D_TRAIN_TEXT_CHARS == len(d_train_text) だけの比較は自己参照になる。
# D_TRAIN_TEXT_CHARS 自体が「SMOKE_TEST でなければ len(d_train_text)」という式で
# 定義されているため、この比較は d_train_text がループ中に再代入されない限り常に
# 成立し、d_train_text[:D_TRAIN_TEXT_CHARS] が実際に符号化に使われたかどうかを
# 何も検証しない(train_language_model に渡す直前で別の値にすり替わっていても
# 検出できない)。
#
# 代わりに、実際に学習に渡された d_train_ids を復号し、d_train_text 全体と
# 内容が完全一致することを確認する(005・006 で確立した「符号化・復号のラウンド
# トリップが完全一致すること」の検証パターンと同じ)。d_train_ids は
# tokenizer.encode(d_train_text[:D_TRAIN_TEXT_CHARS]) の結果そのものであり、
# これを復号した文字列が d_train_text 全体と一致して初めて、d_train_text の
# 一部だけが符号化に使われていないことを独立に確認できる。
if not SMOKE_TEST:
    _d_train_text_decoded = tokenizer.decode(d_train_ids)
    assert _d_train_text_decoded == d_train_text, (
        "本番実行のはずが、実際に学習に渡された d_train_ids を復号したテキストが "
        "d_train_text 全体と一致しない(d_train_text の一部だけが符号化に使われた"
        f"疑いがある): 復号後の文字数={len(_d_train_text_decoded):,}, "
        f"d_train_text 全体の文字数={len(d_train_text):,}"
    )
    print(f"[OK] d_train_ids を復号したテキストが d_train_text 全体"
          f"({len(d_train_text):,} 文字)と完全一致した(train_language_model に "
          "実際に渡された学習データが本番水準であることを確認)")

    # D_NUM_STEPS == PRODUCTION_D_NUM_STEPS だけの比較も同様に自己参照になりうる
    # (train_language_model の呼び出しで num_steps=D_NUM_STEPS 以外の値が渡されて
    # いても検出できない)。history["train_loss"] は train_language_model が
    # 実際に何ステップ回したかを直接反映する返り値であり、これを記録した
    # train_loss_len(実験 D: 主判定 2 条件 + 診断条件の学習ループのセルを参照)を
    # PRODUCTION_D_NUM_STEPS と照合する。
    for _cond in D_CONDITIONS:
        for _run in experiment_d_raw[_cond]["seeds"]:
            assert _run["train_loss_len"] == PRODUCTION_D_NUM_STEPS, (
                f"{_cond} seed={_run['seed']}: train_language_model が実際に学習した"
                f"ステップ数(history['train_loss'] の要素数 {_run['train_loss_len']})が "
                f"PRODUCTION_D_NUM_STEPS({PRODUCTION_D_NUM_STEPS})と一致しない。"
                "num_steps=D_NUM_STEPS 以外の値が渡された疑いがある。"
            )
    print(f"[OK] 実験 D の全 {sum(len(experiment_d_raw[c]['seeds']) for c in D_CONDITIONS)} "
          f"run で、実際の学習ステップ数(history['train_loss'] の長さ)が "
          f"PRODUCTION_D_NUM_STEPS({PRODUCTION_D_NUM_STEPS})と一致した")
```

    [OK] d_train_ids を復号したテキストが d_train_text 全体(22,543,743 文字)と完全一致した(train_language_model に 実際に渡された学習データが本番水準であることを確認)
    [OK] 実験 D の全 9 run で、実際の学習ステップ数(history['train_loss'] の長さ)が PRODUCTION_D_NUM_STEPS(300)と一致した



```python
# --- 実験 D: 不変条件 P2(全条件で追加学習が実際に進んでいること) ---
D_P2_HOLDS = all(
    run["progressed"] for cond in D_CONDITIONS for run in experiment_d_raw[cond]["seeds"]
)
print(f"P2(全条件で末尾{P2_AVERAGING_WINDOW}ステップ平均損失 < 先頭{P2_AVERAGING_WINDOW}ステップ平均損失)成立: {D_P2_HOLDS}")

# --- 不変条件: 実験 D の主判定 2 条件で非埋め込みパラメータ数が完全一致すること ---
assert _d_param_counts["gqa_mean_pool"] == _d_param_counts["gqa_random"], (
    "実験 D の主判定条件で非埋め込みパラメータ数が一致しない"
)
print(f"[OK] 主判定条件の非埋め込みパラメータ数が一致: {_d_param_counts['gqa_mean_pool']:,}")
```

    P2(全条件で末尾3ステップ平均損失 < 先頭3ステップ平均損失)成立: True
    [OK] 主判定条件の非埋め込みパラメータ数が一致: 2,886,912



```python
# --- 実験 D: 対比量の計算 ---
_d_mean_final = np.array([r["final_bpb"] for r in experiment_d_raw["gqa_mean_pool"]["seeds"]])
_d_random_final = np.array([r["final_bpb"] for r in experiment_d_raw["gqa_random"]["seeds"]])
_d_mean_initial = np.array([r["initial_bpb"] for r in experiment_d_raw["gqa_mean_pool"]["seeds"]])
_d_random_initial = np.array([r["initial_bpb"] for r in experiment_d_raw["gqa_random"]["seeds"]])

n_seeds = len(D_SEEDS)
delta_bpb = float(_d_random_final.mean() - _d_mean_final.mean())
sigma_delta_bpb = float(
    np.sqrt(_d_random_final.std(ddof=1) ** 2 / n_seeds + _d_mean_final.std(ddof=1) ** 2 / n_seeds)
)
print(f"Delta_bpb(最終) = {delta_bpb:.4f}, sigma = {sigma_delta_bpb:.4f}, "
      f"Delta_bpb / sigma = {delta_bpb / sigma_delta_bpb:.2f}" if sigma_delta_bpb > 0 else "sigma=0")

# 診断量: 初期化直後(追加学習ステップ数 0)の bpb(介入の直接作用点)。
print(f"\n[診断量] 初期化直後の bpb: mean_pool={_d_mean_initial.mean():.4f} "
      f"(seed毎={_d_mean_initial.tolist()}), random={_d_random_initial.mean():.4f} "
      f"(seed毎={_d_random_initial.tolist()})")

_d_mqa_final = np.array([r["final_bpb"] for r in experiment_d_raw["mqa_mean_pool_diagnostic"]["seeds"]])
print(f"[診断量] MQA・平均プール初期化 + 追加学習の最終 bpb(seed平均): {_d_mqa_final.mean():.4f}"
      "(多頭注意機構と非埋め込みパラメータ数が揃わないため参考値)")
```

    Delta_bpb(最終) = 0.5474, sigma = 0.0042, Delta_bpb / sigma = 129.17
    
    [診断量] 初期化直後の bpb: mean_pool=2.3261 (seed毎=[2.3261236092861584, 2.3261236092861584, 2.3261236092861584]), random=3.3208 (seed毎=[3.350713482371572, 3.2944031878778888, 3.317141639970767])
    [診断量] MQA・平均プール初期化 + 追加学習の最終 bpb(seed平均): 1.8892(多頭注意機構と非埋め込みパラメータ数が揃わないため参考値)


### 7.6 外挿値の合計とセッション予算との比較

5.3 節(符号化・追加学習・評価)の小計に、実験 A・B・C 自身の時間計測の外挿値を
合算し、セッション予算との比較を確定させる(CLAUDE.md 第 1 段階要件 1)。

**この外挿値の絶対値はデバイス依存であることに注意する。** 5.3 節は`DEVICE`
(ローカルでは MPS)、実験 A・B・C は`TIMING_DEVICE`(ローカルでは CPU)で測った
係数を、本番の Google Colab T4 GPU(CUDA)にそのまま適用している。べき乗則の
**指数** はハードウェアが変わってもある程度は保たれると期待できるが、**係数**
(1 回あたりの絶対時間)はデバイスによって大きく異なる。特に実験 C は本番で
decode ステップが延べ約 13 万回(`PRODUCTION_C_REPEATS` x 2 条件 x
(`PRODUCTION_C_T_SMALL` + `PRODUCTION_C_T_LARGE`))に達し、T4 GPU では
1 ステップあたりのカーネル起動オーバーヘッド(CPU 実行にはない要素)が支配項に
なりうる。したがって、以下の外挿値の合計は **セッション予算に収まるかどうかの
目安** として使い、本番実行後の実測値で必ず更新すること(この後の実験セルの
再実行や再計測は求めない。8 節の考察執筆時に、実際にかかった時間をあわせて
記録すればよい)。


```python
TOTAL_EXTRAPOLATED_SECONDS = (
    _subtotal_5_3_seconds
    + EXPERIMENT_A_EXTRAPOLATED_SECONDS
    + EXPERIMENT_B_EXTRAPOLATED_SECONDS
    + EXPERIMENT_C_EXTRAPOLATED_SECONDS
)
print(f"5.3 節(符号化・追加学習・評価)小計: {_subtotal_5_3_seconds:.1f} 秒")
print(f"実験 A: {EXPERIMENT_A_EXTRAPOLATED_SECONDS:.1f} 秒")
print(f"実験 B: {EXPERIMENT_B_EXTRAPOLATED_SECONDS:.1f} 秒")
print(f"実験 C: {EXPERIMENT_C_EXTRAPOLATED_SECONDS:.1f} 秒")
print(f"合計: {TOTAL_EXTRAPOLATED_SECONDS:.1f} 秒 / セッション予算: {SESSION_BUDGET_SECONDS} 秒")
if TOTAL_EXTRAPOLATED_SECONDS > SESSION_BUDGET_SECONDS:
    print("[警告] 外挿値の合計がセッション予算を超える。本番実行前にステップ数・水準数の見直しが必要。")
else:
    print("[OK] 外挿値の合計はセッション予算に対して十分な余裕がある。")
```

    5.3 節(符号化・追加学習・評価)小計: 34.6 秒
    実験 A: 826.6 秒
    実験 B: 44.5 秒
    実験 C: 802.2 秒
    合計: 1707.8 秒 / セッション予算: 7200 秒
    [OK] 外挿値の合計はセッション予算に対して十分な余裕がある。


### 7.7 冗長な再計算の排除の確認

`d_train_ids`・`_full_eval_windows`・`_full_eval_mask`は 5.3 節で 1 回だけ符号化し、
実験 D の条件 x シードの全 9 回の学習・評価ループでそのまま再利用した(再符号化して
いない)。基となるメモリ領域のポインタ(`Tensor.data_ptr()`)が、5.3 節での作成時と
実験 D のループを終えた現在とで一致することを確認する。


```python
assert d_train_ids.data_ptr() == _d_train_ids_data_ptr_at_creation, (
    "d_train_ids が実験 D のループ中に再代入・再符号化された可能性がある"
)
assert _full_eval_windows.data_ptr() == _d_eval_windows_data_ptr_at_creation
assert _full_eval_mask.data_ptr() == _d_eval_mask_data_ptr_at_creation
print("[OK] d_train_ids・_full_eval_windows・_full_eval_mask は、5.3 節で符号化した"
      "同一のメモリ領域を実験 D の全 9 回の学習・評価ループで再利用した"
      "(再符号化・再計算していない)")
```

    [OK] d_train_ids・_full_eval_windows・_full_eval_mask は、5.3 節で符号化した同一のメモリ領域を実験 D の全 9 回の学習・評価ループで再利用した(再符号化・再計算していない)


### 7.8 実験 D のチェックポイントのアップロード

判定の一次情報(実験 A・B・C の条件 x 水準 x 反復の生の実行時間、実験 D の条件 x
シードの学習曲線・bits-per-byte)は 7.2〜7.5 節の各計測セルの直後で既にセル出力へ
全件印字済みであり、それが唯一の記録である(ファイルへの書き出しは行わない)。

一方、実験 D の代表チェックポイント(GQA・平均プール初期化・シード 0)は、014・015
など後続トピックの入力になるモデルであるため、Hugging Face Hub の
`kojikojiprg/ai-theories-small-gpt-en`の`gqa`ブランチへアップロードする。ローカルへ
ダウンロードしてからスクリプト経由でアップロードする迂回はせず、**このノートブックの
アップロードセルから、Google Colab Secrets の`HF_TOKEN`を使って直接アップロードする。**

アップロードは`UPLOAD_ARTIFACTS`フラグ(既定`False`、`SMOKE_TEST`とは独立)で守り、
Colab Secrets から`HF_TOKEN`が取得できない場合は例外を送出せずスキップし、アップロード
後は`list_repo_files`で意図したファイルが実際に存在することを確認する。


```python
# 実験 D の代表チェックポイント(GQA・平均プール初期化・シード 0)を一時ファイルに
# 書き出す。state_dict は実行時のデバイス(本番の Google Colab では CUDA)に紐づく
# テンソルを持つため、保存前に全テンソルを CPU へ移す(Hugging Face Hub 上の
# アーティファクトとしてデバイスに依存せず後続トピックから扱えるようにするため)。
import tempfile

_gqa_checkpoint_path = Path(tempfile.mkdtemp()) / "010_gqa_uptrained.pt"
# tie_embeddings=True により token_embedding.weight と lm_head.weight は同一の
# ストレージを共有している。しかし state_dict() はパラメータごとに .detach() を
# 呼ぶため、共有元が同じでも Python オブジェクトとしては別物になる(id() では
# 共有を検出できない)。さらに、共有元が同じでも .cpu() を個別に呼ぶと(異なる
# device 間の転送は必ず新しいストレージを割り当てるため)共有関係が失われ、
# 埋め込み行列が 2 重に保存されてファイルサイズが不必要に増える。そこで、
# 元のストレージのポインタ(data_ptr())ごとに 1 回だけ .cpu() を呼び、
# 同じ data_ptr() には同じ変換後テンソルを再利用することで共有関係を保つ。
_cpu_tensor_cache: dict[int, torch.Tensor] = {}
_gqa_state_dict_cpu: dict[str, torch.Tensor] = {}
for _key, _value in _d_trained_model_for_upload.state_dict().items():
    _ptr = _value.data_ptr()
    if _ptr not in _cpu_tensor_cache:
        _cpu_tensor_cache[_ptr] = _value.detach().cpu()
    _gqa_state_dict_cpu[_key] = _cpu_tensor_cache[_ptr]
torch.save(_gqa_state_dict_cpu, _gqa_checkpoint_path)

# 保存の往復に欠損がないことの検証: 保存したファイルを map_location="cpu" で
# 読み戻し、元のモデルの state_dict と全テンソルが一致することを確認する。
_gqa_state_dict_reloaded = torch.load(_gqa_checkpoint_path, map_location="cpu")
assert _gqa_state_dict_reloaded.keys() == _gqa_state_dict_cpu.keys(), (
    "読み戻した state_dict のキー集合が元のモデルと一致しない"
)
for _key in _gqa_state_dict_cpu:
    assert torch.equal(_gqa_state_dict_reloaded[_key], _gqa_state_dict_cpu[_key]), (
        f"読み戻した state_dict のテンソルが元のモデルと一致しない: {_key}"
    )
print(f"[OK] 保存・読み戻し(map_location=\"cpu\")で全 {len(_gqa_state_dict_cpu)} テンソルが完全一致")

_checkpoint_bytes = _gqa_checkpoint_path.stat().st_size
_checkpoint_sha256 = hashlib.sha256(_gqa_checkpoint_path.read_bytes()).hexdigest()
print(f"書き出し先(一時ファイル): {_gqa_checkpoint_path}")
print(f"バイト数: {_checkpoint_bytes:,}")
print(f"SHA-256: {_checkpoint_sha256}")
```

    [OK] 保存・読み戻し(map_location="cpu")で全 39 テンソルが完全一致
    書き出し先(一時ファイル): /tmp/tmpxuqvskkr/010_gqa_uptrained.pt
    バイト数: 19,948,905
    SHA-256: f23e68cd4a2865f01612f20af10482fe7146deb84d16df18a78489e43037b9ca



```python
# --- Hugging Face Hub へのアップロード(既定ではスキップする) ---
GQA_TARGET_REPO_ID = "kojikojiprg/ai-theories-small-gpt-en"
GQA_TARGET_BRANCH = "gqa"
GQA_TOKENIZER_REPO_ID = "kojikojiprg/ai-theories-tokenizer-en"


def fetch_main_config_for_gqa(num_key_value_heads: int) -> dict:
    '''main ブランチの config.json を取得し、num_key_value_heads を加える。

    d_model・num_layers・num_heads などのフィールドを手で列挙して再構築すると、main 側に
    フィールドが増えたとき(swiglu_d_ff・dropout・tie_embeddings など)に本セルの更新漏れで
    gqa ブランチの config.json だけが値を欠いたまま取り残される。直接取得してそのまま
    コピーすれば、この種の更新漏れが構造的に起きない。
    '''
    from huggingface_hub import hf_hub_download

    _config_path = hf_hub_download(
        repo_id=GQA_TARGET_REPO_ID, filename="config.json", revision="main"
    )
    _config = json.loads(Path(_config_path).read_text(encoding="utf-8"))
    _config["num_key_value_heads"] = num_key_value_heads
    return _config


def build_gqa_model_card(config: dict, checkpoint_sha256: str) -> str:
    '''モデルカード(日本語メイン・英語併記)を作る。'''
    _num_heads = config["num_heads"]
    _num_key_value_heads = config["num_key_value_heads"]
    return f'''---
language: en
license: mit
tags:
- ai-theories
- gpt
- gqa
- scratch-implementation
---

# ai-theories 標準小型 GPT モデル(英語・Grouped-Query Attention、ブランチ: `{GQA_TARGET_BRANCH}`)

`ai-theories`(https://github.com/kojikojiprg/ai-theories)プロジェクトの成果物。
[kojikojiprg/ai-theories-small-gpt-en](https://huggingface.co/kojikojiprg/ai-theories-small-gpt-en)
の `main` ブランチ(通常の多頭注意機構、ヘッド数 {_num_heads})を起点に、
[010. KV キャッシュと推論の計算量](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/010_kv_cache_and_inference_compute-theory)
の実験 D で Key / Value ヘッド数を {_num_key_value_heads} に削減し(Grouped-Query
Attention)、平均プール初期化(Ainslie et al., "GQA: Training Generalized Multi-Query
Transformer Models from Multi-Head Checkpoints", EMNLP 2023 の uptraining)の後に
追加学習したチェックポイント(シード 0 の代表例)。

研究・教育目的のモデルであり、品質保証は行っていない。商用・実運用での利用は想定しない。

## 構成

`config.json` を参照。`num_key_value_heads` が `num_heads` より小さい点のみ、
`main` ブランチと異なる。

**トークナイザはこのリポジトリ・ブランチには同梱していない。**
[kojikojiprg/ai-theories-tokenizer-en](https://huggingface.co/{GQA_TOKENIZER_REPO_ID})
を使用すること(`main` ブランチと共通)。

## 検証情報

- SHA-256(`model_state.pt`): `{checkpoint_sha256}`

## 関連ノートブック

- [010. KV キャッシュと推論の計算量](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/010_kv_cache_and_inference_compute-theory)
'''


_hf_token = None
try:
    from google.colab import userdata

    _hf_token = userdata.get("HF_TOKEN")
except Exception as _e:  # noqa: BLE001  # Colab Secrets 未設定・非 Colab 環境など理由を問わずスキップする
    print(f"Colab Secrets から HF_TOKEN を取得できなかったため、アップロードをスキップした: {_e!r}")

if _hf_token:
    # gqa_mean_pool の Key / Value ヘッド数(実験 D の条件設定、_d_specs を参照)を
    # そのまま使う(ここで数値を独自に書き直さない)。
    _gqa_num_key_value_heads = _d_specs["gqa_mean_pool"]["num_key_value_heads"]
    _gqa_config = fetch_main_config_for_gqa(_gqa_num_key_value_heads)
    assert _gqa_config["num_key_value_heads"] < _gqa_config["num_heads"], (
        f"num_key_value_heads({_gqa_config['num_key_value_heads']}) が "
        f"num_heads({_gqa_config['num_heads']})未満ではない。"
        "GQA への変換になっていない疑いがある。"
    )
    _gqa_card_text = build_gqa_model_card(_gqa_config, _checkpoint_sha256)

    from huggingface_hub import HfApi

    _api = HfApi(token=_hf_token)
    _api.create_branch(repo_id=GQA_TARGET_REPO_ID, branch=GQA_TARGET_BRANCH, exist_ok=True)
    _api.upload_file(
        path_or_fileobj=str(_gqa_checkpoint_path),
        path_in_repo="model_state.pt",
        repo_id=GQA_TARGET_REPO_ID,
        revision=GQA_TARGET_BRANCH,
    )
    _api.upload_file(
        path_or_fileobj=json.dumps(_gqa_config, indent=2, ensure_ascii=False).encode("utf-8"),
        path_in_repo="config.json",
        repo_id=GQA_TARGET_REPO_ID,
        revision=GQA_TARGET_BRANCH,
    )
    _api.upload_file(
        path_or_fileobj=_gqa_card_text.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=GQA_TARGET_REPO_ID,
        revision=GQA_TARGET_BRANCH,
    )

    # upload_file が例外を送出しなかったことだけをアップロード成功の根拠にせず、
    # list_repo_files で意図したファイルが実際に存在することを確認する。
    _uploaded_files = set(
        _api.list_repo_files(repo_id=GQA_TARGET_REPO_ID, revision=GQA_TARGET_BRANCH)
    )
    for _expected in ("model_state.pt", "config.json", "README.md"):
        assert _expected in _uploaded_files, (
            f"アップロード後の確認: {_expected} が "
            f"{GQA_TARGET_REPO_ID}@{GQA_TARGET_BRANCH} に見つからない"
        )
    print(
        "[OK] アップロード完了かつ list_repo_files で存在を確認した: "
        f"https://huggingface.co/{GQA_TARGET_REPO_ID}/tree/{GQA_TARGET_BRANCH}"
    )
```


    Processing Files (0 / 0)      : |          |  0.00B /  0.00B            



    New Data Upload               : |          |  0.00B /  0.00B            



      ...skkr/010_gqa_uptrained.pt: 100%|##########| 19.9MB / 19.9MB            


    No files have been modified since last commit. Skipping to prevent empty commit.
    WARNING:huggingface_hub.hf_api:No files have been modified since last commit. Skipping to prevent empty commit.
    No files have been modified since last commit. Skipping to prevent empty commit.
    WARNING:huggingface_hub.hf_api:No files have been modified since last commit. Skipping to prevent empty commit.
    No files have been modified since last commit. Skipping to prevent empty commit.
    WARNING:huggingface_hub.hf_api:No files have been modified since last commit. Skipping to prevent empty commit.


    [OK] アップロード完了かつ list_repo_files で存在を確認した: https://huggingface.co/kojikojiprg/ai-theories-small-gpt-en/tree/gqa




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/03_efficient_training/010_kv_cache_and_inference_compute.ipynb
