---
title: "小型 GPT の事前学習(Pretraining a Small GPT)(実装・実験編 3/4)"
---

この記事は後編(実装・実験編 3/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/006_pretraining_small_gpt-practice-2)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/006_pretraining_small_gpt-practice-4)。

## 6. 実験 / Experiments

**実行環境についての注記**: 5.1 節で述べた通り、以下の実験セルは Claude Code によりローカル(Apple Silicon、MPS または CPU)でスモークテストとして実行したものである。判定基準・実験設計自体は本番(Google Colab T4 GPU、`SMOKE_TEST=False`)を前提に事前登録し、**宣言した判定基準は結果を見た後に変更しない**。スモークテストの結果は、多くの場合サンプルサイズ・学習量の不足により「判定不能」になることが予想されるが、これは想定通りであり、結論を曲げずにそのまま報告する。実験 D・E・F の結論は **支持 / 反証 / 判定不能** の 3 値で報告し、差がノイズ床(実験 C)の 2 倍以内に収まった場合は「判定不能」とする。全実験が 1 シード(実験 C・G を除く)であるため、単一の観測差から条件の効果を断定できない。

### 実験 A: ランダム初期化時の損失が $\ln V$ に一致するか

**検証すること**: 学習を一切行っていないランダム初期化直後の model は、各トークンについて語彙 $V$ 個からの一様分布に近い予測をするはずである。このとき交差エントロピー損失の期待値は $-\log(1/V) = \ln V$ になる。5 条件(文字レベル・BPE × 4 語彙サイズ・Unigram)× 2 言語のすべてでこれが成り立つかを検証する。

**判定基準(事前登録)**: 全条件で $\lvert \text{loss} - \ln V \rvert < 0.05$。

**本番実行前の修正 12**: 検証窓の一部(先頭 32 窓)ではなく **全体** を使って平均損失を計算する。窓の一部だけを使うと、統計的なばらつきだけで判定基準 0.05 を超える偽陽性のリスクがある(実測: スモークテストでの diff は 0.0025〜0.0267 と、判定基準に対する余裕が半分程度しかなかった)。メモリ制約に対応するため、`evaluate_bits_per_byte`(`src/training/trainer.py`)と同様にミニバッチに分けて負の対数尤度を積算し、パディング位置(`make_evaluation_windows` のマスク)を除いた実トークン数で平均する。


```python
def experiment_a_random_init_loss(lang: str, tokenizer_name: str, seed: int = 0, batch_size: int = 16) -> dict:
    data = prepare_condition_data(lang, tokenizer_name)
    model = build_model(data["vocab_size"], seed=seed).to(device)
    windows, mask = data["windows"], data["mask"]
    total_nll = 0.0
    total_count = 0
    with torch.no_grad():
        for start in range(0, windows.size(0), batch_size):
            batch = windows[start : start + batch_size].to(device)
            batch_mask = mask[start : start + batch_size, 1:].to(device)
            logits = model(batch)
            predicted = logits[:, :-1, :]
            targets = batch[:, 1:]
            per_token_nll = torch.nn.functional.cross_entropy(
                predicted.reshape(-1, data["vocab_size"]), targets.reshape(-1), reduction="none"
            ).view(batch_mask.shape)
            total_nll += (per_token_nll * batch_mask).sum().item()
            total_count += int(batch_mask.sum().item())
    loss = total_nll / total_count
    expected = np.log(data["vocab_size"])
    return {
        "vocab_size": data["vocab_size"],
        "loss": loss,
        "expected": expected,
        "diff": abs(loss - expected),
        "num_windows": windows.size(0),
        "num_tokens": total_count,
    }


experiment_a_results = []
for lang in ("ja", "en"):
    for tokenizer_name in tokenizer_conditions:
        r = experiment_a_random_init_loss(lang, tokenizer_name)
        r["lang"], r["tokenizer"] = lang, tokenizer_name
        experiment_a_results.append(r)
        status = "OK" if r["diff"] < 0.05 else "NG"
        print(
            f"[{status}] {lang}/{tokenizer_name}: V={r['vocab_size']:5d}  "
            f"windows={r['num_windows']}  tokens={r['num_tokens']:,}  "
            f"loss={r['loss']:.4f}  ln(V)={r['expected']:.4f}  diff={r['diff']:.4f}"
        )

all_pass_a = all(r["diff"] < 0.05 for r in experiment_a_results)
print(f"\n判定基準(全条件で diff < 0.05)を満たすか: {all_pass_a}")
```

    [OK] ja/character: V= 4654  windows=1750  tokens=446,016  loss=8.4946  ln(V)=8.4455  diff=0.0491
    [NG] ja/bpe_v1024: V= 1024  windows=1875  tokens=478,022  loss=6.9953  ln(V)=6.9315  diff=0.0639
    [NG] ja/bpe_v2048: V= 2048  windows=1577  tokens=402,087  loss=7.6782  ln(V)=7.6246  diff=0.0536
    [NG] ja/bpe_v4096: V= 4096  windows=1360  tokens=346,761  loss=8.3703  ln(V)=8.3178  diff=0.0525
    [OK] ja/bpe_v8192: V= 8192  windows=1198  tokens=305,449  loss=9.0600  ln(V)=9.0109  diff=0.0491
    [OK] ja/unigram: V= 4096  windows=1625  tokens=414,284  loss=8.3578  ln(V)=8.3178  diff=0.0400
    [NG] en/character: V= 2394  windows=4730  tokens=1,205,997  loss=7.9702  ln(V)=7.7807  diff=0.1895
    [NG] en/bpe_v1024: V= 1024  windows=2037  tokens=519,247  loss=6.9845  ln(V)=6.9315  diff=0.0531
    [OK] en/bpe_v2048: V= 2048  windows=1687  tokens=430,006  loss=7.6707  ln(V)=7.6246  diff=0.0461
    [NG] en/bpe_v4096: V= 4096  windows=1435  tokens=365,904  loss=8.3707  ln(V)=8.3178  diff=0.0529
    [NG] en/bpe_v8192: V= 8192  windows=1239  tokens=315,796  loss=9.0663  ln(V)=9.0109  diff=0.0554
    [NG] en/unigram: V= 4096  windows=1503  tokens=383,231  loss=8.3692  ln(V)=8.3178  diff=0.0514
    
    判定基準(全条件で diff < 0.05)を満たすか: False


#### 診断: 実験 A の判定基準未達の原因切り分け(本番実行前の修正 26)

本番実行の結果、実験 A は 12 条件中複数の条件で判定基準($\lvert \text{loss} - \ln V \rvert < 0.05$)を満たさなかった。**これは判定基準を満たさなかった原因を切り分けるための診断であり、判定基準の変更ではない。実験 A の判定は「満たさなかった」ままである**(`CLAUDE.md`: 本番実行の後に判定基準を改訂してはならない)。

**有力な仮説**: 重み共有(weight tying、3.3 節)により出力層`lm_head`が入力側のトークン埋め込み行列そのものであるため、正規化前置(Pre-Layer Normalization)の残差接続を通じて入力トークンの埋め込みが最終層の出力までほぼそのまま伝わり、初期状態でも「現在の入力トークンを繰り返す」方向に logits が系統的に偏りうる。ずれの大きさは検証テキストのトークン反復率(同じトークンが隣接して出現する頻度)と語彙サイズに依存するため、条件ごとに異なりうる。これは実装の誤りではなく、重み共有という設計そのものの性質である可能性が高い。

以下を測定する(**学習は行わない**、初期化直後の測定のみ)。

1. 全 12 条件について、`tie_embeddings=False`で構築した model の初期損失と $\ln V$ の差を測り、`tie_embeddings=True`(実験 A、上記)の diff と並べる
2. 検証テキストにおける隣接トークンの反復率(位置 $t$ と $t+1$ のトークンが一致する割合)を測り、`tie_embeddings=True`での diff との対応を見る
3. `tie_embeddings=True`の場合の初期 logits について、目標トークンの logit の平均値と、全語彙にわたる logit の標準偏差の平均値を測り、logits が一様分布からどれだけ離れているかを定量化する


```python
def measure_adjacent_repetition_rate(windows: torch.Tensor, mask: torch.Tensor) -> float:
    """検証テキストにおける隣接トークンの反復率(位置 t と t+1 のトークンが一致する割合、
    本番実行前の修正 26)。``mask`` は目標側(``windows[:, 1:]``)のパディングマスク。
    """
    current = windows[:, :-1]
    following = windows[:, 1:]
    valid = mask[:, 1:].bool()
    matches = (current == following) & valid
    return matches.sum().item() / valid.sum().item()


def diagnose_weight_tying_bias(lang: str, tokenizer_name: str, seed: int = 0, batch_size: int = 16) -> dict:
    """tie_embeddings=False での初期損失差と、tie_embeddings=True での初期 logits の
    一様分布からの偏りを測定する(実験 A の判定基準未達の原因切り分け、本番実行前の修正 26)。
    学習は行わない。
    """
    data = prepare_condition_data(lang, tokenizer_name)
    windows, mask = data["windows"], data["mask"]
    vocab_size = data["vocab_size"]
    expected = np.log(vocab_size)

    model_untied = build_model(vocab_size, seed=seed, tie_embeddings=False).to(device)
    model_tied = build_model(vocab_size, seed=seed, tie_embeddings=True).to(device)

    total_nll_untied = 0.0
    total_count_untied = 0
    target_logit_sum = 0.0
    vocab_std_sum = 0.0
    n_positions = 0

    with torch.no_grad():
        for start in range(0, windows.size(0), batch_size):
            batch = windows[start : start + batch_size].to(device)
            batch_mask = mask[start : start + batch_size, 1:].to(device)
            targets = batch[:, 1:]

            logits_untied = model_untied(batch)[:, :-1, :]
            nll_untied = torch.nn.functional.cross_entropy(
                logits_untied.reshape(-1, vocab_size), targets.reshape(-1), reduction="none"
            ).view(batch_mask.shape)
            total_nll_untied += (nll_untied * batch_mask).sum().item()
            total_count_untied += int(batch_mask.sum().item())

            logits_tied = model_tied(batch)[:, :-1, :]
            target_logits = logits_tied.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            vocab_std = logits_tied.std(dim=-1)
            target_logit_sum += (target_logits * batch_mask).sum().item()
            vocab_std_sum += (vocab_std * batch_mask).sum().item()
            n_positions += int(batch_mask.sum().item())

    loss_untied = total_nll_untied / total_count_untied
    return {
        "diff_untied": abs(loss_untied - expected),
        "repetition_rate": measure_adjacent_repetition_rate(windows, mask),
        "mean_target_logit": target_logit_sum / n_positions,
        "mean_vocab_logit_std": vocab_std_sum / n_positions,
    }


tied_diff_by_condition = {(r["lang"], r["tokenizer"]): r["diff"] for r in experiment_a_results}
diagnosis_results = []
print(
    f"{'lang':4s} {'condition':12s} {'diff(tied)':>11s} {'diff(untied)':>13s} "
    f"{'repeat_rate':>12s} {'target_logit':>13s} {'vocab_std':>10s}"
)
for lang in ("ja", "en"):
    for tokenizer_name in tokenizer_conditions:
        d = diagnose_weight_tying_bias(lang, tokenizer_name)
        d["lang"], d["tokenizer"] = lang, tokenizer_name
        d["diff_tied"] = tied_diff_by_condition[(lang, tokenizer_name)]
        diagnosis_results.append(d)
        print(
            f"{lang:4s} {tokenizer_name:12s} {d['diff_tied']:11.4f} {d['diff_untied']:13.4f} "
            f"{d['repetition_rate']:12.4f} {d['mean_target_logit']:13.4f} {d['mean_vocab_logit_std']:10.4f}"
        )
```

    lang condition     diff(tied)  diff(untied)  repeat_rate  target_logit  vocab_std
    ja   character         0.0491        0.1836       0.0049        0.0015     0.3202
    ja   bpe_v1024         0.0639        0.1538       0.0024       -0.0140     0.3202
    ja   bpe_v2048         0.0536        0.1673       0.0019       -0.0032     0.3201
    ja   bpe_v4096         0.0525        0.1715       0.0016       -0.0030     0.3204
    ja   bpe_v8192         0.0491        0.1670       0.0015        0.0009     0.3196
    ja   unigram           0.0400        0.1675       0.0040        0.0103     0.3208
    en   character         0.1895        0.1053       0.0197       -0.1369     0.3186
    en   bpe_v1024         0.0531        0.1808       0.0030       -0.0008     0.3192
    en   bpe_v2048         0.0461        0.1620       0.0018        0.0068     0.3214
    en   bpe_v4096         0.0529        0.1661       0.0011       -0.0028     0.3204
    en   bpe_v8192         0.0554        0.1651       0.0007       -0.0041     0.3195
    en   unigram           0.0514        0.1755       0.0092       -0.0017     0.3204


### 実験 B: 因果マスクが未来のトークンを遮断するか

**検証すること**: 位置 $t$ の logits は、位置 $t$ 以前(`input_ids[0..t]`)のみに依存し、位置 $t+1$ 以降を変更しても変化しないはずである。因果マスクが正しく機能しているかを、入力の一部を変更する前後で位置 $t$ の logits を比較して検証する。

**判定基準(事前登録)**: 位置 $t+1$ 以降の入力を変更したときの位置 $t$ の logits の変化(絶対値の最大)が $10^{-5}$ 未満。


```python
def experiment_b_causal_mask(lang: str = "ja", tokenizer_name: str | None = None, seed: int = 0) -> dict:
    if tokenizer_name is None:
        tokenizer_name = f"bpe_v{BPE_VOCAB_SIZES[0]}"
    data = prepare_condition_data(lang, tokenizer_name)
    model = build_model(data["vocab_size"], seed=seed).to(device)
    model.eval()

    torch.manual_seed(0)
    x = torch.randint(0, data["vocab_size"], (2, SEQUENCE_LENGTH), device=device)
    x_modified = x.clone()
    t = SEQUENCE_LENGTH // 2
    # 位置 t+1 以降を全く別のランダムなトークン列に差し替える。
    x_modified[:, t + 1 :] = torch.randint(
        0, data["vocab_size"], x_modified[:, t + 1 :].shape, device=device
    )

    with torch.no_grad():
        logits = model(x)
        logits_modified = model(x_modified)

    diff = (logits[:, : t + 1] - logits_modified[:, : t + 1]).abs().max().item()
    return {"max_diff_up_to_t": diff}


result_b = experiment_b_causal_mask()
print(f"位置 0..t の logits の最大差分: {result_b['max_diff_up_to_t']:.2e}")
print(f"判定基準(< 1e-5)を満たすか: {result_b['max_diff_up_to_t'] < 1e-5}")
```

    位置 0..t の logits の最大差分: 0.00e+00
    判定基準(< 1e-5)を満たすか: True


### 本番実行前の判定基準の改訂

**CLAUDE.md は「宣言した判定基準は、結果を見た後に変更しない」と定めている。** 以下の改訂は
**本番(Google Colab T4 GPU)実行の前** に行ったものであり、**本番の判定はまだ一度も行っていない**。
前回のスモークテスト実行(旧基準)で得られた「支持」という判定結果は、旧基準によるものとして
7.1'・7.2' 節にそのまま残す(削除しない)。

| 実験 | 旧基準 | 新基準 | 改訂理由 |
|---|---|---|---|
| D | 差 $> 2\sigma$($\sigma$: 単一条件・3 シードのノイズ床) | 差 $> 2\sqrt{2}\,\sigma_{\text{lang}}$($\sigma_{\text{lang}}$: 言語ごとのノイズ床) | 対比量(文字レベル $-$ 最良部分語)は独立な 2 測定の差であり、誤差伝播により標準偏差は $\sqrt{2}\,\sigma$ になる。$\sigma$ も言語ごとに測定するよう変更(修正 6) |
| E | 前半改善 $-$ 後半改善 $> 2\sigma$(語彙サイズ 3 点) | 隣接 3 改善幅が単調減少($> 2\sqrt{6}\,\sigma_{\text{lang}}$、語彙サイズ 4 点) | 語彙サイズを等比数列 4 点に変更(修正 3)。対比量(隣接改善幅の差)は 3 つの独立測定から作るため標準偏差は $\sqrt{6}\,\sigma$ になる |
| F | 相対改善率の差 $> 2(\sigma/\text{bpb})$ | 相対改善率の差 $> 4\,\sigma_{\text{rel}}$($\sigma_{\text{rel}}$ = 日英ノイズ床の大きい方) | 対比量は日英 $\times$ 2 点の計 4 つの独立測定から作るため標準偏差は $\approx 2\sigma_{\text{rel}}$ になる。事前にはどちらの言語のノイズ床が大きいか分からないため、保守的な側(大きい方)を採用する(修正 6) |
| G | model サイズとステップ数を同時に変更 | model サイズを固定し、ステップ数のみ増やす | 2 つの要因を同時に変えると性能向上の要因を分離できない。model サイズと性能の関係は 009 の主題であり、006 では変数として扱わない |

これらの改訂はいずれも、**観測結果の方向(支持が出たか反証が出たか)に依存しない一般論**
(対数軸上で等比刻みでなければ収穫逓減を検証できないこと、独立測定の差・和の誤差伝播、
2 つの交絡要因の分離)に基づく判断であり、特定の結果を導くために事後的に閾値を調整したもの
ではない。

### 実験 C: シード間ノイズ床の測定(日英それぞれ)

**検証すること**: バイトレベル BPE $V{=}2048$(スモークテストでの Unigram 一致条件、本番では $V{=}4096$)を、**日本語・英語それぞれ** で乱数シード(seed)を変えて学習し、最終検証 bits-per-byte のばらつきを言語ごとに測定する。日本語は文字あたりの情報量・系列長が英語と異なるため、シード間分散が言語間で同じである保証がない(修正 6)。このばらつきを、実験 D・E の判定基準で使う言語ごとの「ノイズ床(noise floor)」 $\sigma_{\text{ja}}$・$\sigma_{\text{en}}$ とする。

**判定基準を設けない**。本実験は後続実験(D・E・F)の閾値を得るための測定であり、真偽を判定する対象ではない。

**5 シードを使う理由(本番実行前の修正 11)**: 標本標準偏差 $s$ は真の標準偏差 $\sigma$ の不偏推定量ではなく、正規分布を仮定すると期待値は $E[s] = c_4(n)\,\sigma$($c_4$ は自由度に依存する不偏化係数、$n$ はシード数)になる。$n{=}3$ では $c_4(3) \approx 0.886$ であり、標本標準偏差は真の値を系統的に約 11.4% 過小評価する。$n{=}5$ では $c_4(5) \approx 0.940$ で過小評価が約 6.0% に縮む。判定閾値の分母として使う以上、この過小バイアスを縮めることが望ましいため、シード数を 3 から 5(seed = 0〜4)に増やす。


```python
t0 = time.time()
CENTER_BPE = f"bpe_v{BPE_VOCAB_SIZES[2]}"  # Unigram と厳密に一致させた条件(修正 4)

# D・E・F で共有するランを格納する辞書(条件 (lang, tokenizer_name) -> run_condition の戻り値)。
# 各言語の「中心語彙サイズの seed=0」は実験 C の 1 回目のランをそのまま再利用し、
# 二重に学習しない。
shared_runs = {}
noise_floor = {}

if _cached_results is not None:
    # キャッシュから復元する(修正 27)。shared_runs はこの時点で D・E・F 分も含めて
    # 全条件が復元されるため、直後の D・E・F 準備セルのループは自然にすべてスキップされる
    # (「if key in shared_runs: continue」が全条件に該当するため)。
    for key_str, run in _cached_results["shared_runs"].items():
        lang, name = key_str.split("|", 1)
        shared_runs[(lang, name)] = run
    noise_floor = {lang: float(v) for lang, v in _cached_results["noise_floor"].items()}
    print("[キャッシュ] 実験 C(ノイズ床)を含む学習ランをキャッシュから復元しました")
    for lang in ("ja", "en"):
        print(f"  {lang}: noise_floor = {noise_floor[lang]:.4f}")
else:
    for lang in ("ja", "en"):
        bpb_by_seed = []
        for seed in NOISE_FLOOR_SEEDS:
            result = run_condition(lang, CENTER_BPE, seed=seed)
            if seed == 0:
                shared_runs[(lang, CENTER_BPE)] = result
            final_bpb = result["history"]["eval_bits_per_byte"][-1]
            bpb_by_seed.append(final_bpb)
            print(f"{lang} seed={seed}: 最終検証 bits-per-byte = {final_bpb:.4f}")
        noise_floor[lang] = float(np.std(bpb_by_seed))
        print(f"  -> {lang} のノイズ床({len(NOISE_FLOOR_SEEDS)} シードの標準偏差): {noise_floor[lang]:.4f}\n")

    print(f"実行時間: {time.time() - t0:.1f}s")
    print(f"noise_floor = {noise_floor}")
```

    ja seed=0: 最終検証 bits-per-byte = 2.4469
    ja seed=1: 最終検証 bits-per-byte = 2.4570
    ja seed=2: 最終検証 bits-per-byte = 2.4510
    ja seed=3: 最終検証 bits-per-byte = 2.4631
    ja seed=4: 最終検証 bits-per-byte = 2.4431
      -> ja のノイズ床(5 シードの標準偏差): 0.0071
    
    en seed=0: 最終検証 bits-per-byte = 2.5838
    en seed=1: 最終検証 bits-per-byte = 2.5894
    en seed=2: 最終検証 bits-per-byte = 2.5730
    en seed=3: 最終検証 bits-per-byte = 2.5727
    en seed=4: 最終検証 bits-per-byte = 2.5794
      -> en のノイズ床(5 シードの標準偏差): 0.0064
    
    実行時間: 506.2s
    noise_floor = {'ja': 0.007117025702562842, 'en': 0.006404353387021051}


### 実験 D・E・F: 同一の学習ランを共有した 3 つの検証

実験 D(部分語分割は文字レベルより優れるか)・E(語彙サイズ増加は収穫逓減するか)・F(語彙サイズ増加の効きかたは日本語の方が大きいか)は、**同一の学習ラン(日英各 5 条件)を共有し、判定する指標だけを分ける**。各言語のバイトレベル BPE $V{=}2048$(Unigram と一致させた条件)は実験 C の seed=0 のランをそのまま再利用する(二重学習しない)。

- **実験 D の検証すること**: 部分語分割(BPE・Unigram)は文字レベルより bits-per-byte が低いか。
  **判定基準(事前登録、本番実行前に改訂)**: 日英とも、最良の部分語条件が文字レベルを $2\sqrt{2}\,\sigma_{\text{lang}}$ を超えて下回る。
  **選択バイアスについての注記**: 「最良の部分語条件」は 4 つの BPE 条件と Unigram、計 5 条件中の最小値(`min`)を取っている。複数条件から最小値を選ぶ操作自体が、単一条件どうしの比較よりも偶然小さい値を選びやすくする(選択バイアス)。ノイズ床 $\sigma_{\text{lang}}$ は単一条件の 3 シード標準偏差であり、この選択バイアスを補正していない点に注意する。
- **実験 E の検証すること**: 語彙サイズ増加の改善は収穫逓減するか。BPE の語彙サイズ 4 点 $V_1{<}V_2{<}V_3{<}V_4$(公比 2 の等比数列、修正 3)について、隣接改善幅 $g_1 = b_1 - b_2$、$g_2 = b_2 - b_3$、$g_3 = b_3 - b_4$($b_i$ は $V_i$ での bits-per-byte)が単調減少するか($g_1 > g_2 > g_3$)を検証する。
  **誤差伝播の式**: 各 $b_i$ は独立な 1 回の測定であり、標準偏差を $\sigma_{\text{lang}}$ とする。隣接する 2 つの改善幅の差 $d_{i} = g_i - g_{i+1} = b_i - 2b_{i+1} + b_{i+2}$ は 3 つの独立測定の線形結合であり、分散は $\text{Var}(d_i) = \sigma^2 + 4\sigma^2 + \sigma^2 = 6\sigma^2$、標準偏差は $\sqrt{6}\,\sigma_{\text{lang}}$ になる。
  **判定基準(事前登録、本番実行前に改訂)**: $d_1 = b_1 - 2b_2 + b_3$ と $d_2 = b_2 - 2b_3 + b_4$ の **両方** が $2\sqrt{6}\,\sigma_{\text{lang}}$ を超えて正であれば「支持」(単調減少)、両方が $2\sqrt{6}\,\sigma_{\text{lang}}$ を超えて負であれば「反証」(収穫逓増)、それ以外は「判定不能」とする。
- **実験 F の検証すること**: 語彙サイズ増加の効きかたは日本語の方が大きいか。$V_1{\to}V_4$(4 点の両端)の **相対** 改善率 $r_{\text{lang}} = (b_1 - b_4) / b_1$ を日英で比較する。
  **誤差伝播の式**: $r_{\text{lang}}$ は $b_1$・$b_4$ の 2 測定から作られ、日英で計 4 測定になる。ノイズ床由来の相対誤差を $\sigma_{\text{rel,lang}} = \sigma_{\text{lang}} / b_{\text{center,lang}}$(各言語の中心条件の bpb で正規化)と近似すると、対比量 $r_{\text{ja}} - r_{\text{en}}$ の標準偏差は $\approx 2\,\sigma_{\text{rel}}$($\sigma_{\text{rel}} = \max(\sigma_{\text{rel,ja}}, \sigma_{\text{rel,en}})$、事前にどちらの言語のノイズ床が大きいか分からないため保守的な側を採用する、修正 6)。
  **判定基準(事前登録、本番実行前に改訂)**: $r_{\text{ja}} - r_{\text{en}} > 4\,\sigma_{\text{rel}}$ であれば「支持」。

**同一計算予算についての注記(実験 D の交絡)**: 全条件を同じステップ数 `NUM_STEPS` で学習するため、語彙サイズが大きく fertility が低い(1 トークンあたりの情報量が多い)条件ほど、同じステップ数で **より多くの原文バイト数を消化する**。これは「同一計算予算(同一 FLOPs 相当)での比較」であり「同一データ量での比較」ではない。条件ごとの消化バイト数・(コーパスが小さい場合の)エポック数は 7 節で表にして報告する。


```python
t0 = time.time()

for lang in ("ja", "en"):
    for tokenizer_name in tokenizer_conditions:
        key = (lang, tokenizer_name)
        if key in shared_runs:
            continue  # 実験 C で学習済み(各言語・Unigram 一致条件・seed=0)
        shared_runs[key] = run_condition(lang, tokenizer_name, seed=0)
        bpb = shared_runs[key]["history"]["eval_bits_per_byte"][-1]
        print(f"{lang}/{tokenizer_name}: 最終検証 bits-per-byte = {bpb:.4f}  (経過 {time.time()-t0:.0f}s)")

print(f"\n合計実行時間: {time.time() - t0:.1f}s")
print(f"総ラン数: {len(shared_runs)}(想定: 2 言語 x 5 条件 = 10、うち各言語の Unigram 一致条件は実験 C と共有)")
```

    ja/character: 最終検証 bits-per-byte = 2.3375  (経過 54s)
    ja/bpe_v1024: 最終検証 bits-per-byte = 2.5356  (経過 103s)
    ja/bpe_v2048: 最終検証 bits-per-byte = 2.5183  (経過 151s)
    ja/bpe_v8192: 最終検証 bits-per-byte = 2.4132  (経過 207s)
    ja/unigram: 最終検証 bits-per-byte = 2.3695  (経過 259s)
    en/character: 最終検証 bits-per-byte = 2.9068  (経過 326s)
    en/bpe_v1024: 最終検証 bits-per-byte = 2.7596  (経過 377s)
    en/bpe_v2048: 最終検証 bits-per-byte = 2.7214  (経過 426s)
    en/bpe_v8192: 最終検証 bits-per-byte = 2.4236  (経過 482s)
    en/unigram: 最終検証 bits-per-byte = 2.4419  (経過 533s)
    
    合計実行時間: 533.1s
    総ラン数: 12(想定: 2 言語 x 5 条件 = 10、うち各言語の Unigram 一致条件は実験 C と共有)



```python
# 各条件の最終検証 bits-per-byte を一覧表にする。
final_bpb = {
    lang: {name: shared_runs[(lang, name)]["history"]["eval_bits_per_byte"][-1] for name in tokenizer_conditions}
    for lang in ("ja", "en")
}
# 分母(total_bytes)が全条件で完全に同一の定数になっていることを確認する(修正 2)。
for lang in ("ja", "en"):
    total_bytes_by_condition = {
        name: shared_runs[(lang, name)]["total_bytes"] for name in tokenizer_conditions
    }
    unique_values = set(total_bytes_by_condition.values())
    assert len(unique_values) == 1, f"{lang}: total_bytes が条件間で異なる: {total_bytes_by_condition}"
    print(f"{lang}: total_bytes は全条件で同一 = {unique_values.pop():,} バイト")

for lang in ("ja", "en"):
    print(f"--- {lang} ---")
    for name in tokenizer_conditions:
        r = shared_runs[(lang, name)]
        print(
            f"  {name:12s}: bpb={final_bpb[lang][name]:.4f}  "
            f"vocab={r['vocab_size']:5d}  non_emb_params={r['non_embedding_params']:,}  "
            f"train_tokens={r['train_tokens']:,}"
        )
```

    ja: total_bytes は全条件で同一 = 1,265,517 バイト
    en: total_bytes は全条件で同一 = 1,214,117 バイト
    --- ja ---
      character   : bpb=2.3375  vocab= 4654  non_emb_params=3,149,056  train_tokens=8,507,563
      bpe_v1024   : bpb=2.5356  vocab= 1024  non_emb_params=3,149,056  train_tokens=8,799,191
      bpe_v2048   : bpb=2.5183  vocab= 2048  non_emb_params=3,149,056  train_tokens=7,172,792
      bpe_v4096   : bpb=2.4469  vocab= 4096  non_emb_params=3,149,056  train_tokens=6,061,518
      bpe_v8192   : bpb=2.4132  vocab= 8192  non_emb_params=3,149,056  train_tokens=5,217,974
      unigram     : bpb=2.3695  vocab= 4096  non_emb_params=3,149,056  train_tokens=7,586,435
    --- en ---
      character   : bpb=2.9068  vocab= 2394  non_emb_params=3,149,056  train_tokens=23,003,819
      bpe_v1024   : bpb=2.7596  vocab= 1024  non_emb_params=3,149,056  train_tokens=9,971,837
      bpe_v2048   : bpb=2.7214  vocab= 2048  non_emb_params=3,149,056  train_tokens=8,194,972
      bpe_v4096   : bpb=2.5838  vocab= 4096  non_emb_params=3,149,056  train_tokens=6,926,577
      bpe_v8192   : bpb=2.4236  vocab= 8192  non_emb_params=3,149,056  train_tokens=5,956,436
      unigram     : bpb=2.4419  vocab= 4096  non_emb_params=3,149,056  train_tokens=7,801,569


#### 実験 D・E・F の判定(新基準)

**判定関数の欠落についての注記(本番実行前の修正 25)**: `judge_d`・`judge_f`には当初、「支持」と「判定不能」の 2 分岐しかなく、`judge_e`にはある「反証」の分岐が実装されていなかった。6 節冒頭で判定を支持 / 反証 / 判定不能の 3 値で事前登録しているため、これは判定基準の変更ではなく実装の欠落であり、修正した(`judge_d`は文字レベルの方が閾値を超えて低い場合、`judge_f`は英語の相対改善率の方が閾値を超えて大きい場合を反証とする)。この欠落により、Google Colab T4 での本番実行(`SMOKE_TEST=False`)の結果、実際には反証だった条件(実験 D の日本語、実験 F)が誤って「判定不能」と報告されていた。学習ランのやり直しは不要である(判定は`final_bpb`と`noise_floor`のみから計算されるため、5.8 節のキャッシュ・修正後の判定関数で再計算すればよい)。


```python
def judge_d(lang: str) -> str:
    best_subword_bpb = min(final_bpb[lang][name] for name in tokenizer_conditions if name != "character")
    char_bpb = final_bpb[lang]["character"]
    diff = char_bpb - best_subword_bpb  # 正なら部分語の方が低い(良い)
    threshold = 2 * (2 ** 0.5) * noise_floor[lang]
    if diff > threshold:
        return f"支持(文字レベル {char_bpb:.4f} - 最良部分語 {best_subword_bpb:.4f} = {diff:.4f} > 2√2σ {threshold:.4f})"
    if diff < -threshold:
        return f"反証(文字レベル {char_bpb:.4f} - 最良部分語 {best_subword_bpb:.4f} = {diff:.4f} < -2√2σ {threshold:.4f}、文字レベルの方が低い)"
    return f"判定不能(差 {diff:.4f} が 2√2σ {threshold:.4f} 以内)"


def judge_e(lang: str) -> str:
    b1, b2, b3, b4 = (final_bpb[lang][f"bpe_v{v}"] for v in BPE_VOCAB_SIZES)
    d1 = b1 - 2 * b2 + b3  # g1 - g2
    d2 = b2 - 2 * b3 + b4  # g2 - g3
    threshold = 2 * (6 ** 0.5) * noise_floor[lang]
    if d1 > threshold and d2 > threshold:
        return f"支持(d1={d1:.4f}, d2={d2:.4f} > 2√6σ {threshold:.4f}、単調減少)"
    if d1 < -threshold and d2 < -threshold:
        return f"反証(d1={d1:.4f}, d2={d2:.4f} < -2√6σ {threshold:.4f}、収穫逓増)"
    return f"判定不能(d1={d1:.4f}, d2={d2:.4f}、閾値 2√6σ {threshold:.4f})"


def judge_f() -> str:
    v1, v4 = BPE_VOCAB_SIZES[0], BPE_VOCAB_SIZES[-1]
    rel_ja = (final_bpb["ja"][f"bpe_v{v1}"] - final_bpb["ja"][f"bpe_v{v4}"]) / final_bpb["ja"][f"bpe_v{v1}"]
    rel_en = (final_bpb["en"][f"bpe_v{v1}"] - final_bpb["en"][f"bpe_v{v4}"]) / final_bpb["en"][f"bpe_v{v1}"]
    diff = rel_ja - rel_en
    rel_noise_ja = noise_floor["ja"] / final_bpb["ja"][CENTER_BPE]
    rel_noise_en = noise_floor["en"] / final_bpb["en"][CENTER_BPE]
    rel_noise = max(rel_noise_ja, rel_noise_en)  # 保守的な側(修正 6)
    threshold = 4 * rel_noise
    if diff > threshold:
        return f"支持(日本語相対改善率 {rel_ja:.4f} - 英語相対改善率 {rel_en:.4f} = {diff:.4f} > 4σ_rel {threshold:.4f})"
    if diff < -threshold:
        return f"反証(日本語相対改善率 {rel_ja:.4f} - 英語相対改善率 {rel_en:.4f} = {diff:.4f} < -4σ_rel {threshold:.4f}、英語の方が大きい)"
    return f"判定不能(差 {diff:.4f} が 4σ_rel {threshold:.4f} 以内)"


print("実験 D(部分語 < 文字レベル):")
print(f"  日本語: {judge_d('ja')}")
print(f"  英語:   {judge_d('en')}")
print("\n実験 E(語彙サイズ増加の収穫逓減、単調減少):")
print(f"  日本語: {judge_e('ja')}")
print(f"  英語:   {judge_e('en')}")
print("\n実験 F(語彙サイズ効果は日本語の方が大きい):")
print(f"  {judge_f()}")
```

    実験 D(部分語 < 文字レベル):
      日本語: 反証(文字レベル 2.3375 - 最良部分語 2.3695 = -0.0320 < -2√2σ 0.0201、文字レベルの方が低い)
      英語:   支持(文字レベル 2.9068 - 最良部分語 2.4236 = 0.4833 > 2√2σ 0.0181)
    
    実験 E(語彙サイズ増加の収穫逓減、単調減少):
      日本語: 判定不能(d1=-0.0541, d2=0.0377、閾値 2√6σ 0.0349)
      英語:   判定不能(d1=-0.0995, d2=-0.0226、閾値 2√6σ 0.0314)
    
    実験 F(語彙サイズ効果は日本語の方が大きい):
      反証(日本語相対改善率 0.0483 - 英語相対改善率 0.1218 = -0.0735 < -4σ_rel 0.0116、英語の方が大きい)


#### 可視化: 条件ごとの bits-per-byte(言語ごとのノイズ床の帯つき)


```python
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, lang in zip(axes, ("ja", "en")):
    plot_grouped_bar(
        {"bits-per-byte": {name: final_bpb[lang][name] for name in tokenizer_conditions}},
        title=f"{lang}: bits-per-byte by tokenizer condition",
        ylabel="bits-per-byte",
        xlabel="tokenizer condition",
        ax=ax,
        noise_band=noise_floor[lang],
    )
fig.tight_layout()
plt.show()
```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/006_pretraining_small_gpt/output_61_0.png)
    


#### 実験 D の交絡: 条件ごとの消化バイト数・エポック数


```python
print(f"{'lang':4s} {'condition':12s} {'train_bytes':>12s} {'train_tokens':>12s} {'digested_bytes':>15s} {'epochs':>8s}")
for lang in ("ja", "en"):
    for name in tokenizer_conditions:
        r = shared_runs[(lang, name)]
        tokens_per_step = BATCH_SIZE * SEQUENCE_LENGTH
        total_tokens_seen = tokens_per_step * NUM_STEPS
        bytes_per_token = r["train_bytes"] / r["train_tokens"]
        digested_bytes = total_tokens_seen * bytes_per_token
        epochs = digested_bytes / r["train_bytes"]
        print(f"{lang:4s} {name:12s} {r['train_bytes']:12,d} {r['train_tokens']:12,d} {digested_bytes:15,.0f} {epochs:8.2f}")
```

    lang condition     train_bytes train_tokens  digested_bytes   epochs
    ja   character      23,309,728    8,507,563       7,137,549     0.31
    ja   bpe_v1024      23,309,728    8,799,191       6,900,992     0.30
    ja   bpe_v2048      23,309,728    7,172,792       8,465,762     0.36
    ja   bpe_v4096      23,309,728    6,061,518      10,017,812     0.43
    ja   bpe_v8192      23,309,728    5,217,974      11,637,303     0.50
    ja   unigram        23,309,728    7,586,435       8,004,174     0.34
    en   character      23,117,476   23,003,819       2,617,927     0.11
    en   bpe_v1024      23,117,476    9,971,837       6,039,240     0.26
    en   bpe_v2048      23,117,476    8,194,972       7,348,691     0.32
    en   bpe_v4096      23,117,476    6,926,577       8,694,384     0.38
    en   bpe_v8192      23,117,476    5,956,436      10,110,462     0.44
    en   unigram        23,117,476    7,801,569       7,719,257     0.33


### 実験 G: 選定条件での本学習

**検証すること**: 実験 D〜F で選定したトークナイザ条件で、より多いステップ数(`G_NUM_STEPS`、5.6 節)で日英各 1 ラン学習し、学習が実際に進行するかを確認する。**model サイズは実験 C〜F と完全に同一のまま変えない**(本番実行前の修正 7)。model サイズを同時に変えると、性能向上が model サイズの増加によるものかステップ数の増加によるものか分離できなくなるためである。model サイズと性能の関係は 009(スケーリング則)の主題であり、006 では変数として扱わない。

**選定方法**: 実験 D〜F の判定が「判定不能」に終わった場合(スモークテストでは高い確率でそうなる、6 節冒頭の注記)、統計的な支持が得られた条件がないため、暫定的に **最終検証 bits-per-byte が最小だった条件** を選定する(参考程度の選定であることを明記する。本番で D〜F が支持を示せば、その結果に基づいて選定し直す)。

**判定基準(事前登録、本番実行前に改訂)**:
- 検証 bits-per-byte が学習を通じて **単調に改善する**(各評価点が直前の評価点以下)
- 最終値が同条件の短時間学習(実験 D〜F、`NUM_STEPS` ステップ)の値を下回る

**セル 66(実験 G の学習ラン)の実行状態についての注記(修正 37)**: 本セルは、経過時間の計測箇所の修正(修正 35、`t0`をループ内で言語ごとにリセットする変更)の後、**Google Colab の無料枠を使い切ったため再実行していない**。そのため出力・実行番号を含まない状態のままコミットされている。**修正内容は表示上の計測方法のみであり、実験 G の結果(検証 bits-per-byte の推移、判定)には一切影響しない。** それらは後続のセル(実験 G の学習曲線・判定)の出力として残っている。実行時間の実測値は 7.7 節に記録してある。


```python
selected_condition = {}
for lang in ("ja", "en"):
    best_name = min(tokenizer_conditions, key=lambda name: final_bpb[lang][name])
    selected_condition[lang] = best_name
    print(f"{lang}: 選定条件 = {best_name}(bits-per-byte = {final_bpb[lang][best_name]:.4f})")
```

    ja: 選定条件 = character(bits-per-byte = 2.3375)
    en: 選定条件 = bpe_v8192(bits-per-byte = 2.4236)



```python
if _cached_results is not None and "experiment_g" in _cached_results:
    # 修正 27: キャッシュには model(学習済み重み)は含まれないため、生成例のセルは
    # このキャッシュ復元では動作しない(生成例のセルで個別に対応する)。
    experiment_g_runs = {lang: dict(_cached_results["experiment_g"][lang]) for lang in ("ja", "en")}
    print("[キャッシュ] 実験 G をキャッシュから復元しました(model オブジェクトは含まれない)")
else:
    t_total_start = time.time()
    experiment_g_runs = {}
    for lang in ("ja", "en"):
        t_lang_start = time.time()
        experiment_g_runs[lang] = run_condition(lang, selected_condition[lang], seed=0, extended=True)
        print(f"{lang}: 完了(所要 {time.time()-t_lang_start:.0f}s)")
    print(f"\n合計実行時間: {time.time() - t_total_start:.1f}s")

# 修正 27: 今回の実行ですべて学習し直した場合(キャッシュ未使用)のみ、結果を保存する。
# 一部だけキャッシュを使った場合(通常は起きないが、万一 shared_runs の一部だけキャッシュに
# 存在しない場合など)は上書き保存しない。
if _cached_results is None:
    save_results_cache(shared_runs, noise_floor, experiment_g_runs, _results_cache_key)
else:
    print("[キャッシュ] 既存のキャッシュを使用したため、保存はスキップします")
```


```python
for lang in ("ja", "en"):
    history = experiment_g_runs[lang]["history"]
    short_run_bpb = final_bpb[lang][selected_condition[lang]]
    final_g_bpb = history["eval_bits_per_byte"][-1]
    bpb_series = history["eval_bits_per_byte"]
    is_monotonic = all(bpb_series[i] >= bpb_series[i + 1] for i in range(len(bpb_series) - 1))
    below_short_run = final_g_bpb < short_run_bpb
    print(
        f"{lang}({selected_condition[lang]}): "
        f"検証 bits-per-byte 推移 = {[round(v, 4) for v in bpb_series]}"
    )
    print(
        f"  単調改善: {is_monotonic}  最終値 {final_g_bpb:.4f} < 短時間学習の値 {short_run_bpb:.4f}: {below_short_run}  "
        f"(判定基準を満たすか: {is_monotonic and below_short_run})"
    )
```

    ja(character): 検証 bits-per-byte 推移 = [2.7621, 2.4231, 2.2846, 2.2021, 2.1458]
      単調改善: True  最終値 2.1458 < 短時間学習の値 2.3375: True  (判定基準を満たすか: True)
    en(bpe_v8192): 検証 bits-per-byte 推移 = [2.7555, 2.4989, 2.3596, 2.267, 2.1936]
      単調改善: True  最終値 2.1936 < 短時間学習の値 2.4236: True  (判定基準を満たすか: True)



```python
fig, ax = plt.subplots(figsize=(6.5, 4.2))
plot_learning_curves_multi_seed(
    {lang: [{"step": experiment_g_runs[lang]["history"]["eval_step"], "evaluation_loss": experiment_g_runs[lang]["history"]["eval_bits_per_byte"]}] for lang in ("ja", "en")},
    step_key="step",
    value_key="evaluation_loss",
    title="Experiment G: validation bits-per-byte over training",
    ylabel="bits-per-byte",
    ax=ax,
)
plt.show()
```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/006_pretraining_small_gpt/output_68_0.png)
    


### 実験 H: 素朴な学習設定の限界の観測

**検証すること**: 実験 G の学習中に記録した勾配ノルム(`train_language_model` が gradient clipping なしで測定した値)と訓練損失の推移を観察し、素朴な学習設定(Adam・固定学習率・gradient clipping なし)の限界の兆候(勾配ノルムのスパイク、損失の発散など)が見られるかを定性的に観察する。追加の学習は不要(実験 G の履歴をそのまま使う)。

**判定基準を設けない**。定性的な観察であることを明記する。


```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for lang, ax in zip(("ja", "en"), axes):
    history = experiment_g_runs[lang]["history"]
    ax.plot(history["step"], history["gradient_norm"])
    ax.set_title(f"{lang}: gradient norm per step (no clipping)")
    ax.set_xlabel("step")
    ax.set_ylabel("gradient norm")
    ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

for lang in ("ja", "en"):
    gn = experiment_g_runs[lang]["history"]["gradient_norm"]
    print(f"{lang}: gradient_norm min={min(gn):.3f} max={max(gn):.3f} mean={np.mean(gn):.3f} std={np.std(gn):.3f}")
```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/006_pretraining_small_gpt/output_70_0.png)
    


    ja: gradient_norm min=0.281 max=1.320 mean=0.691 std=0.165
    en: gradient_norm min=0.250 max=1.018 mean=0.550 std=0.130


### 生成例(greedy と temperature)

実験 G で学習した model(選定条件・`G_NUM_STEPS` ステップ)を使い、`generate()`(貪欲法と temperature sampling のみ、3.1 節)で生成した例を示す。デコーディング手法どうしの比較(top-k・top-p・beam search を含む)は 008 で扱うため、ここでは動作確認としての 2 例のみを示す。学習量がスモークテスト規模(`G_NUM_STEPS` ステップのみ)であるため、生成テキストは流暢さに欠けることが予想される(本番では Colab T4 での大規模な学習後に再生成する)。


```python
for lang in ("ja", "en"):
    if "model" not in experiment_g_runs[lang]:
        # 修正 27: キャッシュ復元では学習済み重みを保持していないため生成できない
        # (5.8 節)。生成例を撮り直すときは、実験 G を再学習してから実行する。
        print(f"=== {lang}({selected_condition[lang]}) ===")
        print(
            "[キャッシュ] model オブジェクトはキャッシュに含まれないため生成例をスキップします"
            "(修正 27: 永続化の対象は最終 bits-per-byte・学習履歴・ノイズ床・パラメータ数・"
            "訓練トークン数・勾配ノルム統計のみで、学習済み重みは含まない)"
        )
        print()
        continue
    model = experiment_g_runs[lang]["model"].to(device)
    tokenizer = tokenizers[lang][selected_condition[lang]]
    prompt_text = val_text[lang][:20]
    prompt_ids = encode_corpus(tokenizer, prompt_text).unsqueeze(0).to(device)

    greedy_ids = model.generate(prompt_ids, max_new_tokens=30, temperature=0.0)
    sampled_ids = model.generate(prompt_ids, max_new_tokens=30, temperature=0.8, seed=0)

    print(f"=== {lang}({selected_condition[lang]}) ===")
    print(f"prompt: {prompt_text!r}")
    print(f"greedy (temperature=0.0): {tokenizer.decode(greedy_ids[0].cpu().tolist())!r}")
    print(f"sampled (temperature=0.8): {tokenizer.decode(sampled_ids[0].cpu().tolist())!r}")
    print()
```

    === ja(character) ===
    prompt: 'に建設されたばかりの、「おっぱいドラゴン'
    greedy (temperature=0.0): 'に建設されたばかりの、「おっぱいドラゴンド」を「この日、これまでのことを発表した。\n\n1988年には'
    sampled (temperature=0.8): 'に建設されたばかりの、「おっぱいドラゴンシー」であるが、元島無殺を持うほか、敦坂の立ちということをと'
    
    === en(bpe_v8192) ===
    prompt: 'e sponsorship of men'
    greedy (temperature=0.0): 'e sponsorship of men in the first time in the first time in the first time in the first time in the first time in the first time in the first time in the'
    sampled (temperature=0.8): "e sponsorship of men at the  Dopt' finred the single victory was elected to the first raceers. Durier had in the was hold of the win to"
    




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/02_pretraining/006_pretraining_small_gpt.ipynb
