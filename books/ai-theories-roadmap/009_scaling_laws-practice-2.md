---
title: "スケーリング則(Scaling Laws)(実装・実験編 2/3)"
---

この記事は後編(実装・実験編 2/3)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/009_scaling_laws-practice-1)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/009_scaling_laws-practice-3)。

### 5.11.1 時間見積もりが予算超過と出た理由(ローカル実行固有の注記)

上のセルの出力は、**この第 1 段階(Claude Code、ローカル Mac、MPS バックエンド)での実測に基づく限り「予算内: False」である。この出力は実行結果をそのまま記録したものであり、加工していない。**

**診断**: 5.10 節の実測(`TIME_PER_STEP_BY_D_MODEL`)を見ると、1 ステップあたり時間は`d_model`=32・64・96・128 に対して 0.0762・0.1086・0.1407・0.1712 秒であり、$d_{model}$ に対するべき指数は 0.580 しかない(理論的には計算量が $d_{model}^2$ に比例するため 2 に近いことが期待される)。この傾向は過去の局所実行(べき指数 0.542)と一致しており、ローカル(MPS バックエンド)では 1 ステップあたりの実行時間が **計算量に比例する部分ではなく、Python のディスパッチ・MPS カーネル起動などのステップ数に比例する固定オーバーヘッドに支配されている** ことを引き続き示す。較正(5.9 節)は`train_language_model`の短い呼び出しを`d_model`=32・96・128 の複数の組み合わせで何度も行うため、固定オーバーヘッドの影響を特に強く受け、較正だけで 3734.7 秒(合計の 33.9%)を要している。

**Colab T4 GPU での実行時間のクロスチェック**: 5.3.1 節の候補設計比較セルで、過去の本番実行(Google Colab、T4 GPU)の実測値(旧設計 25 セル、学習グリッド実行時間 731.2 秒)から実効スループット($\approx1.874\times10^{11}$ FLOPs/秒)を逆算し、それを採用設計(4 水準・20 セル、`TARGET_TOKENS_PER_PARAM_MID=14`)に適用する計算を **既に実行済み** である。その結果は「推定時間: 80.1 分(66.7% of budget)」であり、セッション予算(7200 秒=120 分)に対して十分な余裕を持って収まる。

ローカル(Mac、MPS)で「予算内: False」(合計 11006.5 秒=183.4 分、予算の 152.9%)となったのは、この設計自体が計算量として過大であることを意味するのではなく、**ローカルのハードウェアが Colab の T4 GPU よりも小規模モデル・多数の短い学習呼び出しに対して相対的に遅い** ためである。実際、1 ステップあたりの実測スループット(局所実測 FLOPs/step 時間)を T4 換算スループットと比較すると、`d_model=32`では T4 換算の約 5.9 倍遅い一方、`d_model=128`では局所実測の方がむしろ速い(比 0.83 倍)。**小さい`d_model`ほど固定オーバーヘッドの相対的な影響が大きい** という診断と整合する結果である。

**この見積もりはあくまで実測 1 点からの外挿であり、保証ではない。** 第 2 段階(Colab、`SMOKE_TEST=False`)で 5.10 節のセルが実際に測定する`TIME_PER_STEP_BY_D_MODEL`の値を使えば、5.11 節のセルはこの外挿値ではなく実測値に基づいて再判定される。上記のクロスチェックにより、実測結果が「予算内: False」になる可能性は低いと考えられるが、こうじさんは Colab での本番実行時に 5.11 節のセルの出力(`予算内`)を必ず確認すること。


### 5.12 学習グリッドの実行

`SMOKE_TEST=True`(第 1 段階、Claude Code ローカル)では、この学習は主張を検証する本番実行ではなく、コードが最後まで動作することの確認が目的である。5.3 節で縮小したステップ数(本番ステップ数を一定比率で縮小する。5.8 節の等比構造(計算量予算の公比 2)自体は縮小後も保つ)で全 20 セルを実行し、実測の $N$・$D$・$L$ を記録する(実行結果は 6.2 節・後述のセルで JSON として永続化する)。



```python
grid_points: list[GridPoint] = []
_grid_histories = {}
t_grid_start = time.time()
for d in D_MODEL_LEVELS:
    n = _measured_N[d]
    for level_idx, c in enumerate(COMPUTE_BUDGETS):
        production_steps = _step_table[d][level_idx]
        num_steps = scaled_steps(production_steps)
        model = build_gpt_model(d).to(device)
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        warmup_steps = max(1, round(WARMUP_RATIO * num_steps))
        schedule = functools.partial(
            compute_warmup_cosine_learning_rate,
            warmup_steps=warmup_steps,
            total_steps=num_steps,
            peak_learning_rate=LEARNING_RATE,
            min_learning_rate=LEARNING_RATE * MIN_LEARNING_RATE_RATIO,
        )
        history = train_language_model(
            model,
            train_ids,
            eval_windows,
            eval_mask,
            total_eval_bytes,
            num_steps=num_steps,
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            learning_rate=LEARNING_RATE,
            eval_interval=max(1, num_steps),
            device=device,
            seed=SEED,
            optimizer=optimizer,
            learning_rate_schedule=schedule,
            gradient_clip_threshold=GRADIENT_CLIP_THRESHOLD,
        )
        final_bpb = history["eval_bits_per_byte"][-1]
        actual_D = num_steps * TOKENS_PER_STEP
        grid_points.append(
            GridPoint(d_model=d, non_embedding_params=n, tokens_trained=actual_D, loss=final_bpb)
        )
        _grid_histories[(d, level_idx)] = history

print(f"学習グリッド({_RUN_LABEL}、{len(grid_points)} セル)実行時間: {time.time() - t_grid_start:.1f} s")
# Colab のセッション終了でファイル出力(009_grid_results.json)が失われても、コミット済み
# のノートブックのセル出力から生データを復元できるよう、先頭 5 点ではなく全件を印字する
# (20 セル程度であり出力が過大にならない)。
for gp in grid_points:
    print(gp)

```

    学習グリッド(本番、20 セル)実行時間: 3252.3 s
    GridPoint(d_model=32, non_embedding_params=49312, tokens_trained=13942784, loss=1.6461692383080897)
    GridPoint(d_model=32, non_embedding_params=49312, tokens_trained=27885568, loss=1.5925333112742115)
    GridPoint(d_model=32, non_embedding_params=49312, tokens_trained=55771136, loss=1.5448785965545615)
    GridPoint(d_model=32, non_embedding_params=49312, tokens_trained=111550464, loss=1.4576733692623027)
    GridPoint(d_model=32, non_embedding_params=49312, tokens_trained=223100928, loss=1.4420022537276258)
    GridPoint(d_model=64, non_embedding_params=197440, tokens_trained=3481600, loss=2.026289334110765)
    GridPoint(d_model=64, non_embedding_params=197440, tokens_trained=6963200, loss=1.606780753125758)
    GridPoint(d_model=64, non_embedding_params=197440, tokens_trained=13926400, loss=1.3984719273375974)
    GridPoint(d_model=64, non_embedding_params=197440, tokens_trained=27860992, loss=1.3202211820458072)
    GridPoint(d_model=64, non_embedding_params=197440, tokens_trained=55721984, loss=1.28190075976377)
    GridPoint(d_model=96, non_embedding_params=443232, tokens_trained=1548288, loss=2.522791773920214)
    GridPoint(d_model=96, non_embedding_params=443232, tokens_trained=3104768, loss=1.9972502482798797)
    GridPoint(d_model=96, non_embedding_params=443232, tokens_trained=6201344, loss=1.6244555532733853)
    GridPoint(d_model=96, non_embedding_params=443232, tokens_trained=12410880, loss=1.378690390676437)
    GridPoint(d_model=96, non_embedding_params=443232, tokens_trained=24821760, loss=1.2512591962260573)
    GridPoint(d_model=128, non_embedding_params=787072, tokens_trained=876544, loss=2.820303838016335)
    GridPoint(d_model=128, non_embedding_params=787072, tokens_trained=1744896, loss=2.428276520312415)
    GridPoint(d_model=128, non_embedding_params=787072, tokens_trained=3497984, loss=1.9453209657383772)
    GridPoint(d_model=128, non_embedding_params=787072, tokens_trained=6987776, loss=1.5742909588640368)
    GridPoint(d_model=128, non_embedding_params=787072, tokens_trained=13975552, loss=1.2919314538446303)


### 5.13 前提条件 P0・P3 の判定量: 学習の進行・データ再利用の上限

最終検証 bits-per-byte が、一様分布相当の値の一定割合以下であることを、実験 A・B・C 共通の前提条件 P0 として後段(7 節)で判定する。ここでは判定に使う閾値を、検証したい仮説とは独立な量として確定させる。

**前提条件 P3(データ再利用の上限)**: いずれのグリッドセルも、訓練トークン数($D$)が訓練コーパスのトークン数の`EPOCH_REUSE_LIMIT`(=4.0)倍を超えないこと。同じコーパスを 4 エポックを超えて反復させると、言語モデルの事前学習では過学習・データ重複による汎化性能の低下が報告されている(Muennighoff et al., "Scaling Data-Constrained Language Models", NeurIPS 2023、006 の実装時にも参照した基準)。P0 と同様、検証したい仮説(スケーリング指数)とは独立な、学習グリッドの設計自体が意味を持つための前提条件である。



```python
PRECONDITION_BPB_UNIFORM = compute_bits_per_byte(
    len(val_ids) * np.log(VOCAB_SIZE), total_eval_bytes
)
PRECONDITION_BPB_RATIO = 0.9  # 一様分布相当の 90% 以下であれば学習が進んだとみなす
PRECONDITION_BPB_THRESHOLD = PRECONDITION_BPB_UNIFORM * PRECONDITION_BPB_RATIO
print(f"一様分布相当の bits-per-byte: {PRECONDITION_BPB_UNIFORM:.4f}")
print(f"前提条件 P0 の閾値: {PRECONDITION_BPB_THRESHOLD:.4f}")

EPOCH_REUSE_LIMIT = 4.0  # 前提条件 P3 の上限倍率(Muennighoff et al., 2023 を踏まえた基準)
print(f"前提条件 P3 の上限: 訓練コーパスの {EPOCH_REUSE_LIMIT}倍")

_max_final_bpb = max(gp.loss for gp in grid_points)
_max_epoch_ratio = max(gp.tokens_trained / len(train_ids) for gp in grid_points)
print(f"{_RUN_LABEL}グリッドでの最終 bits-per-byte の最大値: {_max_final_bpb:.4f}")
print(f"{_RUN_LABEL}グリッドでの最大データ再利用比率: {_max_epoch_ratio:.2f}倍")
print(
    f"(SMOKE_TEST={SMOKE_TEST} のため、この時点で P0 を満たさなくても第 1 段階の完了条件ではない。"
    "本番実行後に判定する。P3 は本番スケール(実測 D)で意味を持つ前提条件であり、"
    "スモーク実行では D が大きく縮小されるため判定に使わない。)"
)

```

    一様分布相当の bits-per-byte: 4.2947
    前提条件 P0 の閾値: 3.8652
    前提条件 P3 の上限: 訓練コーパスの 4.0倍
    本番グリッドでの最終 bits-per-byte の最大値: 2.8203
    本番グリッドでの最大データ再利用比率: 1.54倍
    (SMOKE_TEST=False のため、この時点で P0 を満たさなくても第 1 段階の完了条件ではない。本番実行後に判定する。P3 は本番スケール(実測 D)で意味を持つ前提条件であり、スモーク実行では D が大きく縮小されるため判定に使わない。)


### 5.14 ノイズ床の測定

中央の`d_model`・中央の計算量予算の 1 条件を`NOISE_FLOOR_SEEDS`シードで実行し、$L$ のシード間標本標準偏差を測定する(標本標準偏差の過小バイアスを抑えるため 5 シード。シード数自体は縮小しない)。



```python
_mid_d_model = D_MODEL_LEVELS[MID_D_MODEL_IDX]
_mid_level_idx = 2
_mid_production_steps = _step_table[_mid_d_model][_mid_level_idx]
_mid_num_steps = scaled_steps(_mid_production_steps)

_noise_floor_losses = []
for seed in range(NOISE_FLOOR_SEEDS):
    model = build_gpt_model(_mid_d_model).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    warmup_steps = max(1, round(WARMUP_RATIO * _mid_num_steps))
    schedule = functools.partial(
        compute_warmup_cosine_learning_rate,
        warmup_steps=warmup_steps,
        total_steps=_mid_num_steps,
        peak_learning_rate=LEARNING_RATE,
        min_learning_rate=LEARNING_RATE * MIN_LEARNING_RATE_RATIO,
    )
    history = train_language_model(
        model,
        train_ids,
        eval_windows,
        eval_mask,
        total_eval_bytes,
        num_steps=_mid_num_steps,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        learning_rate=LEARNING_RATE,
        eval_interval=max(1, _mid_num_steps),
        device=device,
        seed=1000 + seed,
        optimizer=optimizer,
        learning_rate_schedule=schedule,
        gradient_clip_threshold=GRADIENT_CLIP_THRESHOLD,
    )
    _noise_floor_losses.append(history["eval_bits_per_byte"][-1])

NOISE_STD = float(np.std(_noise_floor_losses, ddof=1))
print(f"ノイズ床の {NOISE_FLOOR_SEEDS} シード損失: {[round(v, 4) for v in _noise_floor_losses]}")
print(f"NOISE_STD(標本標準偏差、ddof=1): {NOISE_STD:.5f}")

```

    ノイズ床の 5 シード損失: [1.6134, 1.6362, 1.644, 1.6098, 1.6258]
    NOISE_STD(標本標準偏差、ddof=1): 0.01459


### 5.14.1 学習グリッドの生データの永続化

Colab のランタイムはセッション終了時に消え、ノートブックのセル出力(標準出力・図)以外は失われる。過去の本番実行では、JSON への保存自体は正しく動作していたにもかかわらず、保存後にファイルがリポジトリへコミット・push されないまま Colab のセッションが終了し、生データが失われたことがあった(グリッド生データの図からの読み取りに頼らざるを得なかった)。これを踏まえ、学習グリッド・ノイズ床・較正の結果を JSON としてディスクに書き出すことに加え、5.12 節の学習グリッド実行セルで`grid_points`を先頭 5 点ではなく全件(20 セル分)印字する。ノートブックのセル出力さえコミットされていれば、この JSON ファイル自体が失われても生データを復元できる。

保存先は`SMOKE_TEST`の値で分ける。本番(`SMOKE_TEST=False`)の結果のみ、リポジトリ管理下
(`.cache/`配下ではない)のパスに保存し、ノートブックと同じコミットでリポジトリに含める。
スモークテストの結果はキャッシュディレクトリに保存し、コミット対象にしない。

**本番実行後にこうじさんが行う作業(チェックリスト)**:
1. `SMOKE_TEST=False`に切り替えて Colab で本番実行する。
2. 実行後、`theories/02_pretraining/009_grid_results.json`が生成されていることを確認する。
3. ノートブック(実行結果を含む)と`009_grid_results.json`を **同じコミット** でリポジトリに追加してコミットする(`git add theories/02_pretraining/009_scaling_laws.ipynb theories/02_pretraining/009_grid_results.json`)。このコミットを忘れると、過去と同様にセッション終了時に生データ(の唯一の外部永続化先)が失われる。
4. 5.12 節のセル出力に`grid_points`が 20 件(4 つの`d_model` x 5 つの計算量予算)すべて印字されていることを確認する(印字されていれば、3. のコミットを万一忘れてもノートブック単体から生データを復元できる)。



```python
GRID_RESULTS_PATH = (
    ROOT / "theories" / "02_pretraining" / "009_grid_results.json"
    if not SMOKE_TEST
    else CACHE_DIR / "smoke_grid_results.json"
)

_grid_results_payload = {
    "smoke_test": SMOKE_TEST,
    "d_model_levels": D_MODEL_LEVELS,
    "compute_budgets": COMPUTE_BUDGETS,
    "target_tokens_per_param_mid": TARGET_TOKENS_PER_PARAM_MID,
    "learning_rate": LEARNING_RATE,
    "gradient_clip_threshold": GRADIENT_CLIP_THRESHOLD,
    "calibration": {
        "chosen_multiplier": _chosen_mult,
        "status": _calibration_status,
        "results": {str(k): v for k, v in _calibration_results.items()},
        "divergence_check": {str(k): v for k, v in _divergence_check.items()},
    },
    "grid_points": [
        {
            "d_model": gp.d_model,
            "non_embedding_params": gp.non_embedding_params,
            "tokens_trained": gp.tokens_trained,
            "loss": gp.loss,
        }
        for gp in grid_points
    ],
    "noise_floor_losses": _noise_floor_losses,
    "noise_std": NOISE_STD,
    "train_ids_length": len(train_ids),
}

GRID_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
GRID_RESULTS_PATH.write_text(
    json.dumps(_grid_results_payload, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(f"学習グリッドの生データを保存した: {GRID_RESULTS_PATH}({len(grid_points)} セル分)")

```

    学習グリッドの生データを保存した: theories/02_pretraining/009_grid_results.json(20 セル分)


### 5.15 解析パイプライン全体のスケーリング計測・外挿

ブートストラップ回数を 3 点以上振って解析パイプライン全体(`reconstruct_optimal_frontier` x 2 数え方 + Chinchilla パラメトリックあてはめ)の実行時間を実測し、本番のブートストラップ回数(1000 回)への外挿値を出す。ここでは学習グリッド(5.12 節)で実際に得た $(N, D, L)$ を使う(`SMOKE_TEST=True`の場合、値そのものは第 1 段階の判定には使わない、時間計測のみが目的)。



```python
_bootstrap_probe_counts = [10, 20, 40]
_bootstrap_probe_times = []
# システムジッター(他プロセスの CPU 占有・GC 停止など)による外れ値の影響を抑えるため、
# 各水準で 2 回計測し最小値を採用する(観測結果の方向に依存しない、標準的なタイミング
# 計測のロバスト化手法)。
for n_boot in _bootstrap_probe_counts:
    _repeat_times = []
    for _repeat in range(2):
        t0 = time.time()
        bootstrap_scaling_analysis(
            grid_points,
            COMPUTE_BUDGETS,
            FLOPS_PER_TOKEN_BODY,
            FLOPS_PER_TOKEN_BODY_OUTPUT,
            noise_std=max(NOISE_STD, 1e-4),
            n_bootstrap=n_boot,
            seed=_repeat,
        )
        _repeat_times.append(time.time() - t0)
    _bootstrap_probe_times.append(min(_repeat_times))

for n_boot, t in zip(_bootstrap_probe_counts, _bootstrap_probe_times, strict=True):
    print(f"n_bootstrap={n_boot}: {t:.2f} s(2 回計測の最小値)")

_bootstrap_fit = fit_power_law(_bootstrap_probe_counts, _bootstrap_probe_times)
_bootstrap_production_estimate = _bootstrap_fit.coefficient * 1000**_bootstrap_fit.exponent
print(f"\nべき指数 b={_bootstrap_fit.exponent:.3f}(反復あたり一定時間なら b~=1)")
print(f"本番(n_bootstrap=1000)への外挿: {_bootstrap_production_estimate:.1f} s")

```

    n_bootstrap=10: 13.90 s(2 回計測の最小値)
    n_bootstrap=20: 26.00 s(2 回計測の最小値)
    n_bootstrap=40: 59.37 s(2 回計測の最小値)
    
    べき指数 b=1.047(反復あたり一定時間なら b~=1)
    本番(n_bootstrap=1000)への外挿: 1671.8 s


### 5.16 解析パイプラインの動作確認

実験 A・B・C で使う関数呼び出しが実際に動作することを確認する。**この時点の数値は動作確認用の小さい`n_bootstrap`(=30)によるものであり、支持・反証・判定不能の結論を導かない**(7 節で改めて宣言する判定基準・前提条件を`N_BOOTSTRAP`で適用する本来の判定セルとは別物である)。



```python
_demo_result_body = reconstruct_optimal_frontier(grid_points, COMPUTE_BUDGETS, FLOPS_PER_TOKEN_BODY)
_demo_result_bo = reconstruct_optimal_frontier(grid_points, COMPUTE_BUDGETS, FLOPS_PER_TOKEN_BODY_OUTPUT)
print(f"[動作確認] num_interior_budgets(本体のみ)= {_demo_result_body.num_interior_budgets}")
print(f"[動作確認] num_interior_budgets(本体+出力層)= {_demo_result_bo.num_interior_budgets}")
if _demo_result_body.power_law_fit is not None:
    print(f"[動作確認] a_body(参考値、判定には使わない)= {_demo_result_body.power_law_fit.exponent:.3f}")

_demo_n = [gp.non_embedding_params for gp in grid_points]
_demo_d = [gp.tokens_trained for gp in grid_points]
_demo_l = [gp.loss for gp in grid_points]
_demo_chinchilla = fit_chinchilla_parametric(_demo_n, _demo_d, _demo_l)
print(
    f"[動作確認] Chinchilla パラメトリックあてはめ: alpha={_demo_chinchilla.alpha:.3f}, "
    f"beta={_demo_chinchilla.beta:.3f}, converged={_demo_chinchilla.converged}"
)

_demo_boot = bootstrap_scaling_analysis(
    grid_points,
    COMPUTE_BUDGETS,
    FLOPS_PER_TOKEN_BODY,
    FLOPS_PER_TOKEN_BODY_OUTPUT,
    noise_std=max(NOISE_STD, 1e-4),
    n_bootstrap=30,
    seed=0,
)
print(f"[動作確認] bootstrap num_successful={_demo_boot.num_successful}/{_demo_boot.num_attempted}")

```

    [動作確認] num_interior_budgets(本体のみ)= 3
    [動作確認] num_interior_budgets(本体+出力層)= 2
    [動作確認] a_body(参考値、判定には使わない)= 0.412
    [動作確認] Chinchilla パラメトリックあてはめ: alpha=0.681, beta=0.641, converged=False
    [動作確認] bootstrap num_successful=30/30


## 6. 不変条件のアサーション / Invariant Assertions

実験の妥当性が依存する量をすべて実行時のアサーションで確認する(表として出力するだけでは足りない)。



```python
# --- 系列長・バッチサイズ・語彙・評価集合が全条件で同一であること ---
assert tokenizer.vocab_size == VOCAB_SIZE
for gp in grid_points:
    pass  # tokens_trained は TOKENS_PER_STEP(= BATCH_SIZE * SEQUENCE_LENGTH)の整数倍
assert all(gp.tokens_trained % TOKENS_PER_STEP == 0 for gp in grid_points)
print("OK: 語彙サイズ・系列長・バッチサイズの整合")

# --- 実測 N が d_model について単調増加であり、隣接水準の比が目標公比 2 から ±15% 以内 ---
_N_sorted = [_measured_N[d] for d in D_MODEL_LEVELS]
assert all(_N_sorted[i] < _N_sorted[i + 1] for i in range(len(_N_sorted) - 1)), "N が単調増加でない"
_N_ratios_check = [_N_sorted[i + 1] / _N_sorted[i] for i in range(len(_N_sorted) - 1)]
# 32 -> 64 の 1 区間のみ、head_dim=32 固定の制約上 d_model がちょうど 2 倍になり
# (N はおよそ d_model の 2 乗で増えるため)N の比が公比 2 から大きく外れる構造的な
# 例外である(head_dim=32 固定に起因)。この 1 区間だけ許容範囲から除外し、
# それ以外の区間は従来どおり厳密に ±15% 以内を要求する。
_EXPECTED_RATIO_EXCEPTIONS = {(32, 64)}
_ratio_pairs = list(zip(D_MODEL_LEVELS[:-1], D_MODEL_LEVELS[1:], strict=True))
for (d_lo, d_hi), r in zip(_ratio_pairs, _N_ratios_check, strict=True):
    if (d_lo, d_hi) in _EXPECTED_RATIO_EXCEPTIONS:
        print(f"  (例外として許容: d_model {d_lo}->{d_hi} の N 比 {r:.3f}、head_dim=32 固定に起因)")
        continue
    assert 1.7 <= r <= 2.3, (
        f"N の比が公比 2 から ±15% を超えている(d_model {d_lo}->{d_hi}): {r:.3f}"
    )
print(f"OK: N の単調増加性・比率({[round(r, 3) for r in _N_ratios_check]}、例外区間を除く)")

# --- 各セルの最小ステップ数が意味のある学習と言える下限を満たすこと ---
# 過去の設計では前提条件 P3(データ再利用上限)により 100 ステップは達成できず、25
# (算出された最小値 28 に安全側の余裕を残した値)まで下げていた。コーパス拡張により
# 制約が P3 からセッション時間に入れ替わり(5.3.1 節)、100 ステップが達成可能になった
# ため、閾値を 100 に引き上げる(5.3.2 節、採用設計での予測値は 107)。
MIN_STEPS_FLOOR = 100
_min_production_steps = min(min(steps) for steps in _step_table.values())
assert _min_production_steps >= MIN_STEPS_FLOOR, (
    f"本番グリッドの最小ステップ数が下限を下回っている: {_min_production_steps} < {MIN_STEPS_FLOOR}"
)
print(f"OK: 本番グリッドの最小ステップ数 {_min_production_steps} >= {MIN_STEPS_FLOOR}")

# --- 符号化と復号のラウンドトリップが完全一致(5.2 節で検証済み、ここで再確認) ---
assert tokenizer.decode(tokenizer.encode(_roundtrip_sample)) == _roundtrip_sample
print("OK: ラウンドトリップ(再確認)")

# --- 全モデルサイズで 本体+出力層 >= 本体のみ の計算量(5.8 節で検証済み、再確認) ---
assert all(FLOPS_PER_TOKEN_BODY_OUTPUT[d] >= FLOPS_PER_TOKEN_BODY[d] for d in D_MODEL_LEVELS)
print("OK: 計算量の数え方の大小関係(再確認)")

# --- IsoFLOP プロファイルの構成に使う D が、実測 D の範囲内にあること(外挿していない) ---
_by_d_model_D_range = {}
for d in D_MODEL_LEVELS:
    ds = [gp.tokens_trained for gp in grid_points if gp.d_model == d]
    _by_d_model_D_range[d] = (min(ds), max(ds))
for c in COMPUTE_BUDGETS:
    for d in D_MODEL_LEVELS:
        d_required = c / FLOPS_PER_TOKEN_BODY[d]
        d_min, d_max = _by_d_model_D_range[d]
        # 範囲外の d_model はプロファイルから除外される設計(reconstruct_optimal_frontier)。
        # ここでは「範囲内と判定されたものが実際に範囲内にあるか」の整合性のみを確認する。
        if d_min <= d_required <= d_max:
            assert d_min - 1e-6 <= d_required <= d_max + 1e-6
print("OK: IsoFLOP プロファイル構成の D が実測範囲内(外挿していない)")

# --- トークナイザ語彙が 008 のものと一致すること(再学習した場合のみ意味を持つ) ---
if not _loaded_from_hub:
    assert tokenizer.vocab_size == VOCAB_SIZE, "再学習した語彙サイズが 008(8192)と一致しない"
    print("OK: 再学習した語彙サイズが 008 と一致")
else:
    print("OK: Hub から取得したトークナイザをそのまま使用(語彙は 008 と定義上同一)")

```

    OK: 語彙サイズ・系列長・バッチサイズの整合
      (例外として許容: d_model 32->64 の N 比 4.004、head_dim=32 固定に起因)
    OK: N の単調増加性・比率([4.004, 2.245, 1.776]、例外区間を除く)
    OK: 本番グリッドの最小ステップ数 107 >= 100
    OK: ラウンドトリップ(再確認)
    OK: 計算量の数え方の大小関係(再確認)
    OK: IsoFLOP プロファイル構成の D が実測範囲内(外挿していない)
    OK: Hub から取得したトークナイザをそのまま使用(語彙は 008 と定義上同一)


### 6.1 あてはめ関数の正しさ(合成データによる検証)

`src/scaling/laws.py`にスクラッチ実装したあてはめ関数が、既知のパラメータから生成した合成データからそのパラメータを正しく復元できることを検証する。



```python
_rng = np.random.default_rng(0)

# fit_power_law: 既知の指数・係数から生成した厳密なべき乗則を復元できること
_x = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
_true_a, _true_b = 3.0, -0.4
_y = _true_a * _x**_true_b
_pl_fit = fit_power_law(_x, _y)
assert abs(_pl_fit.exponent - _true_b) < 1e-6
assert abs(_pl_fit.coefficient - _true_a) < 1e-6
print(f"OK: fit_power_law(厳密) exponent={_pl_fit.exponent:.6f}(真値{_true_b})")

# fit_isoflop_parabola: 合成した放物線の頂点を解析解と一致して返すこと
_log_n = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
_a_true, _b_true, _c_true = 0.5, -4.0, 10.0
_loss_synth = _a_true * _log_n**2 + _b_true * _log_n + _c_true
_parabola_fit = fit_isoflop_parabola(_log_n, _loss_synth)
assert abs(_parabola_fit.a - _a_true) < 1e-8
assert abs(_parabola_fit.vertex_log_n - (-_b_true / (2 * _a_true))) < 1e-6
assert _parabola_fit.is_interior
print(f"OK: fit_isoflop_parabola(厳密) vertex_log_n={_parabola_fit.vertex_log_n:.6f}")

# fit_saturating_power_law: 既知の l_inf, x_c, alpha から生成したデータを復元できること
_x2 = np.array([1000.0, 2000.0, 4000.0, 8000.0, 16000.0])
_l_inf_true, _x_c_true, _alpha_true = 2.0, 5000.0, 0.4
_y2 = _l_inf_true + (_x_c_true / _x2) ** _alpha_true
_sat_fit = fit_saturating_power_law(_x2, _y2)
assert abs(_sat_fit.l_inf - _l_inf_true) < 1e-2
assert abs(_sat_fit.alpha - _alpha_true) < 1e-2
print(f"OK: fit_saturating_power_law(厳密) alpha={_sat_fit.alpha:.4f}(真値{_alpha_true})")

_y2_noisy = _y2 + _rng.normal(0, 0.01, size=_y2.shape)
_sat_fit_noisy = fit_saturating_power_law(_x2, _y2_noisy)
print(
    f"参考: fit_saturating_power_law(ノイズあり) l_inf={_sat_fit_noisy.l_inf:.3f}, "
    f"x_c={_sat_fit_noisy.x_c:.1f}, alpha={_sat_fit_noisy.alpha:.3f}"
)

# fit_chinchilla_parametric: Chinchilla の公表パラメータで生成した合成データを復元できること
_E, _A, _B, _alpha, _beta = 1.69, 406.4, 410.7, 0.34, 0.28
_Ns_synth = np.array([4e7, 1e8, 4e8, 1e9, 4e9, 1e10])
_Ds_synth = np.array([4e9, 1e10, 4e10, 1e11, 2e11, 4e11])
_NN, _DD = np.meshgrid(_Ns_synth, _Ds_synth)
_NN, _DD = _NN.ravel(), _DD.ravel()
_L_synth = _E + _A / _NN**_alpha + _B / _DD**_beta
_chin_fit = fit_chinchilla_parametric(_NN, _DD, _L_synth)
assert abs(_chin_fit.alpha - _alpha) < 0.02, _chin_fit.alpha
assert abs(_chin_fit.beta - _beta) < 0.02, _chin_fit.beta
print(f"OK: fit_chinchilla_parametric(厳密) alpha={_chin_fit.alpha:.4f}, beta={_chin_fit.beta:.4f}")

_a_exp, _b_exp = compute_optimal_allocation_exponents(_alpha, _beta)
assert abs(_a_exp - _beta / (_alpha + _beta)) < 1e-9
print(f"OK: compute_optimal_allocation_exponents a={_a_exp:.4f}, b={_b_exp:.4f}")

_L_synth_noisy = _L_synth * np.exp(_rng.normal(0, 0.02, size=_L_synth.shape))
_chin_fit_noisy = fit_chinchilla_parametric(_NN, _DD, _L_synth_noisy)
assert abs(_chin_fit_noisy.alpha - _alpha) < 0.05
assert abs(_chin_fit_noisy.beta - _beta) < 0.05
print(
    f"OK: fit_chinchilla_parametric(2%ノイズ) alpha={_chin_fit_noisy.alpha:.4f}, "
    f"beta={_chin_fit_noisy.beta:.4f}"
)

print("\n合成データによるあてはめ関数の検証: すべて成功")

```

    OK: fit_power_law(厳密) exponent=-0.400000(真値-0.4)
    OK: fit_isoflop_parabola(厳密) vertex_log_n=4.000000
    OK: fit_saturating_power_law(厳密) alpha=0.4000(真値0.4)
    参考: fit_saturating_power_law(ノイズあり) l_inf=1.959, x_c=5583.8, alpha=0.387
    OK: fit_chinchilla_parametric(厳密) alpha=0.3400, beta=0.2800
    OK: compute_optimal_allocation_exponents a=0.4516, b=0.5484
    OK: fit_chinchilla_parametric(2%ノイズ) alpha=0.3272, beta=0.3126
    
    合成データによるあてはめ関数の検証: すべて成功


## 7. 実験 / Experiments

実験 A(IsoFLOP プロファイルによるべき乗則の妥当性)・実験 B(計算量の数え方が指数に与える影響)・実験 C(2 つの推定法の内的整合性)の 3 実験を独立に宣言する。いずれも $N_{opt}(C)$ のべき指数 $a$ に関する実験であり、3 つの異なる角度から攻める構成である(0 節)。

### 7.1 学習グリッドの共通設定(再掲)

学習グリッド・固定条件・較正は 5 節で確定済みである。ここでは判定に関わる設計のみを再掲する。

- モデルサイズ: `n_layer=4`固定、`d_model`を 4 水準(32, 64, 96, 128)。head_dim を 32 に固定し`num_heads = d_model // 32`。実測 $N$ の目標公比は 2(32→64 の 1 区間を除き 6 節で ±15% 以内であることを確認済み。32→64 は head_dim=32 固定の制約上、この 1 区間だけ N が約 4 倍に飛ぶ構造的な例外であり、6 節で明示的に扱う)。
- 計算量予算: 公比 2 で 5 水準。グリッドの構成には本体のみの数え方($C=6ND$)を用いる。
- ノイズ床: 中央の`d_model`・中央の計算量予算の 1 条件を 5 シードで測定(5.14 節)。

### 7.2 固定条件(事前宣言)

以下を固定条件として事前に宣言する。009 では検証対象から除外し、Porian et al. (2024) の要因 2・3 に相当する既知の交絡として明示的に残す。

- **学習率は全`d_model`・全計算量予算で同一の値に固定する**(5.9 節で較正した`LEARNING_RATE`)。これは Porian らの要因 3(スケール依存の最適化ハイパーパラメータ調整)に相当する既知の交絡であり、009 では扱わない。
- **warmup は各ランの総ステップ数の 10%(比例方式)**。Porian らの要因 2(固定ステップ数 warmup が短いランを不利にする)を避ける側に固定する。
- **cosine decay の長さは各ランの総ステップ数に一致させる。**
- **バッチサイズ・系列長`n_ctx`は全条件で同一に固定する**(006・008 と揃える、`BATCH_SIZE=32`、`SEQUENCE_LENGTH=256`)。
- **アスペクト比(depth と width の比)**: `n_layer`を固定して`d_model`のみを振るため、大きいモデルほど幅広の形状になる。Kaplan et al. (2020) はモデルの形状(depth と width の比)がスケーリング則に与える影響は小さいと報告しており、この簡略化を固定条件として採用する。
- **gradient clipping の閾値**: 較正で決めた`GRADIENT_CLIP_THRESHOLD`(勾配ノルムの 90% 分位点方式、007・008 と同じ)。

### 7.3 判定関数(共通)

対比量とその標準偏差(パラメトリックブートストラップによる)から、支持 / 反証 / 判定不能(または支持 / 反証のみ)を返す判定関数を定義する。



```python
def judge_two_sided(point_estimate: float, boot_std: float, expected_negative: bool) -> str:
    '''支持 / 反証 / 判定不能の 3 値判定(実験 B 用)。

    閾値はブートストラップ標準偏差の 2 倍。期待する符号(``expected_negative``)と
    実際の符号が一致し、かつ閾値を超えていれば「支持」、逆符号かつ閾値超えなら
    「反証」、閾値以内なら「判定不能」。
    '''
    threshold = 2 * boot_std
    if abs(point_estimate) <= threshold:
        return "判定不能"
    is_negative = point_estimate < 0
    return "支持" if is_negative == expected_negative else "反証"


def judge_equivalence(point_estimate: float, boot_std: float) -> str:
    '''支持 / 反証の 2 値判定(実験 C 用、判定不能を設けない設計、7.6 節参照)。'''
    threshold = 2 * boot_std
    return "支持" if abs(point_estimate) <= threshold else "反証"


def judge_experiment_a(a_point: float, a_boot_std: float, log_c: list[float], n_opt: list[float]) -> str:
    '''実験 A の 3 値判定。

    (1) a が 0 と区別できるか(ブートストラップ標準偏差の 2 倍より |a| が大きいか)、
    (2) べき乗則(対数空間の直線)のあてはめの残差平方和が、対数を取らないナイーブな
    線形あてはめ(N_opt = m C + k)の残差平方和より小さいか、の 2 条件で判定する。
    残差平方和はそれぞれの当てはめが定義される空間(前者は対数空間、後者は生の
    数値空間)でそのまま比較する(009 6.5 節の判定基準)。
    '''
    threshold = 2 * a_boot_std
    if abs(a_point) <= threshold:
        return "判定不能"

    c_arr = np.exp(np.asarray(log_c))
    n_opt_arr = np.asarray(n_opt)
    power_fit = fit_power_law(c_arr, n_opt_arr)
    rss_power = float(np.sum(power_fit.residuals**2))

    design = np.stack([c_arr, np.ones_like(c_arr)], axis=1)
    coeffs, *_ = np.linalg.lstsq(design, n_opt_arr, rcond=None)
    rss_linear = float(np.sum((n_opt_arr - design @ coeffs) ** 2))

    return "支持" if rss_power < rss_linear else "反証"

```

### 7.4 実験 A: IsoFLOP プロファイルによる計算量最適配分の推定

- **検証すること**: 本体のみの数え方のもとで、$N_{opt}$ が $C$ のべき乗則として記述できるか。
- **判定基準(3 値: 支持 / 反証 / 判定不能)**
  - 対比量: $\log N_{opt}$ 対 $\log C$ のべき乗則あてはめの指数 $a_{\text{本体}}$。
  - 支持: ブートストラップによる $a_{\text{本体}}$ の標準偏差の 2 倍が、推定値 $a_{\text{本体}}$ 自身より小さい(すなわち $a$ が 0 と区別できる)。かつ、べき乗則(対数空間の直線)のあてはめの残差平方和が、対数を取らない線形あてはめの残差平方和より小さい。
  - 反証: 上記の後半が成立しない(べき乗則が対数を取らない線形より劣る)。
  - 判定不能: $a_{\text{本体}}$ が 0 と区別できない。
- **前提条件 P0**: 全ランで学習が進んでいること。最終検証 bits-per-byte が、一様分布相当の値の`PRECONDITION_BPB_RATIO`(=0.9)以下であること(5.13 節で閾値を確定済み)。
- **前提条件 P1**: 各計算量予算の IsoFLOP プロファイルの最小値が掃引範囲の内点にあること。内点となった予算が 3 つ未満の場合、実験 A 全体を前提不成立として記録する。
- **前提条件 P3**: いずれのグリッドセルも、訓練トークン数が訓練コーパスのトークン数の`EPOCH_REUSE_LIMIT`(=4.0)倍を超えないこと(5.13 節)。

$a$ の推定値が Chinchilla の 0.5 や Kaplan らの 0.73 に近いかどうかは判定基準にしない(計算量の範囲が 1.5 桁では絶対値の一致を主張できないため)。参考として値を報告するのはよいが、事前宣言した判定とは節を分け、事後的な参考であることを明示する。



```python
def run_experiment_a(grid, compute_budgets, noise_std, n_bootstrap, seed=0):
    result = reconstruct_optimal_frontier(grid, compute_budgets, FLOPS_PER_TOKEN_BODY)
    p0_ok = max(gp.loss for gp in grid) <= PRECONDITION_BPB_THRESHOLD
    p1_ok = result.num_interior_budgets >= 3
    p3_ok = max(gp.tokens_trained / len(train_ids) for gp in grid) <= EPOCH_REUSE_LIMIT

    if not (p0_ok and p1_ok and p3_ok):
        return {
            "verdict": "前提不成立",
            "p0_ok": p0_ok,
            "p1_ok": p1_ok,
            "p3_ok": p3_ok,
            "num_interior_budgets": result.num_interior_budgets,
            "result": result,
        }

    boot = bootstrap_scaling_analysis(
        grid,
        compute_budgets,
        FLOPS_PER_TOKEN_BODY,
        FLOPS_PER_TOKEN_BODY_OUTPUT,
        noise_std=noise_std,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    a_point = result.power_law_fit.exponent
    a_std = float(np.std(boot.a_body, ddof=1)) if boot.num_successful > 1 else float("nan")
    verdict = judge_experiment_a(a_point, a_std, result.frontier_log_c, list(np.exp(result.frontier_log_n_opt)))
    return {
        "verdict": verdict,
        "p0_ok": p0_ok,
        "p1_ok": p1_ok,
        "p3_ok": p3_ok,
        "num_interior_budgets": result.num_interior_budgets,
        "a_point": a_point,
        "a_boot_std": a_std,
        "result": result,
        "boot": boot,
    }


# [動作確認専用。判定には使わない] コードが動作することのみを、小さい n_bootstrap で確認する。
_DEMO_N_BOOTSTRAP = 30
_demo_exp_a = run_experiment_a(grid_points, COMPUTE_BUDGETS, max(NOISE_STD, 1e-4), n_bootstrap=_DEMO_N_BOOTSTRAP)
print(
    "[動作確認専用・判定には使わない] 実験 A:",
    {k: v for k, v in _demo_exp_a.items() if k not in ("result", "boot")},
)

# [実際の判定に使う結果] N_BOOTSTRAP(宣言済みの本番値、SMOKE_TEST=True の場合は縮小値)を使う。
exp_a_result = run_experiment_a(grid_points, COMPUTE_BUDGETS, max(NOISE_STD, 1e-4), n_bootstrap=N_BOOTSTRAP)
print(
    f"[{_RUN_LABEL}実行の判定結果{'(未確定、本番実行後に再確認)' if SMOKE_TEST else ''}] 実験 A:",
    {k: v for k, v in exp_a_result.items() if k not in ("result", "boot")},
)

```

    [動作確認専用・判定には使わない] 実験 A: {'verdict': '支持', 'p0_ok': np.True_, 'p1_ok': True, 'p3_ok': True, 'num_interior_budgets': 3, 'a_point': 0.41212091013645175, 'a_boot_std': 0.10046810438505305}
    [本番実行の判定結果] 実験 A: {'verdict': '支持', 'p0_ok': np.True_, 'p1_ok': True, 'p3_ok': True, 'num_interior_budgets': 3, 'a_point': 0.41212091013645175, 'a_boot_std': 0.09533302709206852}


### 7.5 実験 B: 最終層の計算量の扱いが指数に与える影響

- **検証すること**: 計算量の数え方に出力層を含めると、推定される指数 $a$ が小さくなるか(Porian et al., 2024 の要因 1 の再現)。
- **予測の方向と根拠**: 出力層を含めない数え方は小さいモデルの計算量を過小評価する。含めると小さいモデルが相対的に不利に評価され、低い計算量予算での $N_{opt}$ が上がる。その結果フロンティアの傾きは緩やかになるため、$a_{\text{本体+出力層}} < a_{\text{本体}}$ を予測する(3.2.1 節の数値表と整合する向き)。これは Kaplan らの 0.73 から Chinchilla の 0.5 への変化と同じ向きである。
- **判定基準(3 値: 支持 / 反証 / 判定不能)**
  - 対比量: $\Delta a = a_{\text{本体+出力層}} - a_{\text{本体}}$。**同一ブートストラップ反復内の差** として計算する(`bootstrap_scaling_analysis`の`delta_b`)。
  - 閾値: $\Delta a$ のブートストラップ標準偏差の 2 倍。
  - 支持: $\Delta a < 0$ かつ $|\Delta a|$ が閾値を超える。
  - 反証: $\Delta a > 0$ かつ $|\Delta a|$ が閾値を超える。
  - 判定不能: $|\Delta a|$ が閾値以内。
- **前提条件**: 実験 A の P0・P1・P3 に加え、本体+出力層の数え方でも内点となる予算が 3 つ以上あること。
- **追加の診断量(判定基準ではない)**: 注意機構の系列長依存項をさらに加えた 3 つめの数え方(`FLOPS_PER_TOKEN_FULL`)での $a$ も報告してよい。

この実験は **追加の学習を一切行わない**。実験 A と同一の学習結果を、計算量の数え方だけ変えて再解析する。



```python
def run_experiment_b(grid, compute_budgets, noise_std, n_bootstrap, seed=0):
    result_body = reconstruct_optimal_frontier(grid, compute_budgets, FLOPS_PER_TOKEN_BODY)
    result_bo = reconstruct_optimal_frontier(grid, compute_budgets, FLOPS_PER_TOKEN_BODY_OUTPUT)
    p0_ok = max(gp.loss for gp in grid) <= PRECONDITION_BPB_THRESHOLD
    p1_ok = result_body.num_interior_budgets >= 3 and result_bo.num_interior_budgets >= 3
    p3_ok = max(gp.tokens_trained / len(train_ids) for gp in grid) <= EPOCH_REUSE_LIMIT

    if not (p0_ok and p1_ok and p3_ok) or result_body.power_law_fit is None or result_bo.power_law_fit is None:
        return {
            "verdict": "前提不成立",
            "p0_ok": p0_ok,
            "p1_ok": p1_ok,
            "p3_ok": p3_ok,
            "num_interior_budgets_body": result_body.num_interior_budgets,
            "num_interior_budgets_bo": result_bo.num_interior_budgets,
        }

    boot = bootstrap_scaling_analysis(
        grid, compute_budgets, FLOPS_PER_TOKEN_BODY, FLOPS_PER_TOKEN_BODY_OUTPUT,
        noise_std=noise_std, n_bootstrap=n_bootstrap, seed=seed,
    )
    delta_point = result_bo.power_law_fit.exponent - result_body.power_law_fit.exponent
    delta_std = float(np.std(boot.delta_b, ddof=1)) if boot.num_successful > 1 else float("nan")
    verdict = judge_two_sided(delta_point, delta_std, expected_negative=True)

    # 追加の診断量(判定基準ではない): Attention の系列長依存項も加えた数え方
    result_full = reconstruct_optimal_frontier(grid, compute_budgets, FLOPS_PER_TOKEN_FULL)
    a_full = result_full.power_law_fit.exponent if result_full.power_law_fit is not None else None

    return {
        "verdict": verdict,
        "p0_ok": p0_ok,
        "p1_ok": p1_ok,
        "p3_ok": p3_ok,
        "a_body": result_body.power_law_fit.exponent,
        "a_body_output": result_bo.power_law_fit.exponent,
        "delta_point": delta_point,
        "delta_boot_std": delta_std,
        "a_full_diagnostic": a_full,
    }


# [動作確認専用。判定には使わない] コードが動作することのみを、小さい n_bootstrap で確認する。
_demo_exp_b = run_experiment_b(grid_points, COMPUTE_BUDGETS, max(NOISE_STD, 1e-4), n_bootstrap=_DEMO_N_BOOTSTRAP)
print("[動作確認専用・判定には使わない] 実験 B:", _demo_exp_b)

# [実際の判定に使う結果] N_BOOTSTRAP(宣言済みの本番値、SMOKE_TEST=True の場合は縮小値)を使う。
exp_b_result = run_experiment_b(grid_points, COMPUTE_BUDGETS, max(NOISE_STD, 1e-4), n_bootstrap=N_BOOTSTRAP)
print(
    f"[{_RUN_LABEL}実行の判定結果{'(未確定、本番実行後に再確認)' if SMOKE_TEST else ''}] 実験 B:",
    exp_b_result,
)

```

    [動作確認専用・判定には使わない] 実験 B: {'verdict': '前提不成立', 'p0_ok': np.True_, 'p1_ok': False, 'p3_ok': True, 'num_interior_budgets_body': 3, 'num_interior_budgets_bo': 2}
    [本番実行の判定結果] 実験 B: {'verdict': '前提不成立', 'p0_ok': np.True_, 'p1_ok': False, 'p3_ok': True, 'num_interior_budgets_body': 3, 'num_interior_budgets_bo': 2}


### 7.6 実験 C: 2 つの推定法の内的整合性

- **検証すること**: パラメトリックあてはめ(Approach 3)から導出した指数と、IsoFLOP プロファイル(Approach 2)から直接推定した指数が一致するか。
- **判定基準(3 値ではなく支持 / 反証の 2 値、理由は前提条件 P2 の項を参照)**
  - グリッド全体の $(N, D, L)$ に $L(N,D) = E + A/N^{\alpha} + B/D^{\beta}$ を対数空間の Huber 損失であてはめ、$a_{\text{Approach3}} = \beta / (\alpha + \beta)$ を求める。
  - 対比量: $\delta = a_{\text{Approach3}} - a_{\text{本体}}$。同一ブートストラップ反復内の差として計算する(`bootstrap_scaling_analysis`の`delta_c`)。
  - 閾値: $\delta$ のブートストラップ標準偏差の 2 倍。
  - 支持: $|\delta|$ が閾値以内(2 つの推定法が一致する)。
  - 反証: $|\delta|$ が閾値を超える。
  - **判定不能はこの実験では設けない**: 支持の条件が「差が小さいこと」であるため、ノイズが大きいと自動的に支持になってしまう。これを防ぐため前提条件 P2 を設ける。
- **前提条件 P2(識別可能性)**: ブートストラップによる $\alpha \cdot \beta$ の変動係数(標準偏差 / 推定値)が、`P2_CV_THRESHOLD`(=0.5、第 1 段階で宣言)以下であること。満たさない場合、パラメトリックあてはめは識別不能であり、実験 C を前提不成立として記録する。
- **前提条件 P3**: 実験 A の P3(5.13 節)と同一。
- 比較は本体のみの数え方で行う(3.3 節: $a = \beta/(\alpha+\beta)$ の導出が $C \propto ND$ を前提とするため)。



```python
P2_CV_THRESHOLD = 0.5  # alpha・beta の変動係数(標準偏差 / 推定値)の採用上限(第 1 段階で宣言)


def run_experiment_c(grid, compute_budgets, noise_std, n_bootstrap, seed=0):
    result_body = reconstruct_optimal_frontier(grid, compute_budgets, FLOPS_PER_TOKEN_BODY)
    p0_ok = max(gp.loss for gp in grid) <= PRECONDITION_BPB_THRESHOLD
    p1_ok = result_body.num_interior_budgets >= 3
    p3_ok = max(gp.tokens_trained / len(train_ids) for gp in grid) <= EPOCH_REUSE_LIMIT

    if not (p0_ok and p1_ok and p3_ok) or result_body.power_law_fit is None:
        return {"verdict": "前提不成立(P0/P1/P3)", "p0_ok": p0_ok, "p1_ok": p1_ok, "p3_ok": p3_ok}

    boot = bootstrap_scaling_analysis(
        grid, compute_budgets, FLOPS_PER_TOKEN_BODY, FLOPS_PER_TOKEN_BODY_OUTPUT,
        noise_std=noise_std, n_bootstrap=n_bootstrap, seed=seed,
    )
    if boot.num_successful < 2:
        return {"verdict": "前提不成立(ブートストラップ失敗)", "p0_ok": p0_ok, "p1_ok": p1_ok, "p3_ok": p3_ok}

    cv_alpha = float(np.std(boot.alpha, ddof=1) / np.mean(boot.alpha))
    cv_beta = float(np.std(boot.beta, ddof=1) / np.mean(boot.beta))
    p2_ok = cv_alpha <= P2_CV_THRESHOLD and cv_beta <= P2_CV_THRESHOLD

    n_all = [gp.non_embedding_params for gp in grid]
    d_all = [gp.tokens_trained for gp in grid]
    l_all = [gp.loss for gp in grid]
    chin_fit = fit_chinchilla_parametric(n_all, d_all, l_all)
    a_approach3, _ = compute_optimal_allocation_exponents(chin_fit.alpha, chin_fit.beta)

    if not p2_ok:
        return {
            "verdict": "前提不成立(P2: 識別不能)",
            "p0_ok": p0_ok,
            "p1_ok": p1_ok,
            "p2_ok": p2_ok,
            "p3_ok": p3_ok,
            "cv_alpha": cv_alpha,
            "cv_beta": cv_beta,
            "a_body": result_body.power_law_fit.exponent,
            "a_approach3": a_approach3,
        }

    delta_point = a_approach3 - result_body.power_law_fit.exponent
    delta_std = float(np.std(boot.delta_c, ddof=1))
    verdict = judge_equivalence(delta_point, delta_std)
    return {
        "verdict": verdict,
        "p0_ok": p0_ok,
        "p1_ok": p1_ok,
        "p2_ok": p2_ok,
        "p3_ok": p3_ok,
        "cv_alpha": cv_alpha,
        "cv_beta": cv_beta,
        "a_body": result_body.power_law_fit.exponent,
        "a_approach3": a_approach3,
        "delta_point": delta_point,
        "delta_boot_std": delta_std,
    }


# [動作確認専用。判定には使わない] コードが動作することのみを、小さい n_bootstrap で確認する。
_demo_exp_c = run_experiment_c(grid_points, COMPUTE_BUDGETS, max(NOISE_STD, 1e-4), n_bootstrap=_DEMO_N_BOOTSTRAP)
print("[動作確認専用・判定には使わない] 実験 C:", _demo_exp_c)

# [実際の判定に使う結果] N_BOOTSTRAP(宣言済みの本番値、SMOKE_TEST=True の場合は縮小値)を使う。
exp_c_result = run_experiment_c(grid_points, COMPUTE_BUDGETS, max(NOISE_STD, 1e-4), n_bootstrap=N_BOOTSTRAP)
print(
    f"[{_RUN_LABEL}実行の判定結果{'(未確定、本番実行後に再確認)' if SMOKE_TEST else ''}] 実験 C:",
    exp_c_result,
)

```

    [動作確認専用・判定には使わない] 実験 C: {'verdict': '支持', 'p0_ok': np.True_, 'p1_ok': True, 'p2_ok': True, 'p3_ok': True, 'cv_alpha': 0.15635186252962735, 'cv_beta': 0.022934206606268055, 'a_body': 0.41212091013645175, 'a_approach3': 0.48507313061987906, 'delta_point': 0.07295222048342731, 'delta_boot_std': 0.11303065013007245}
    [本番実行の判定結果] 実験 C: {'verdict': '支持', 'p0_ok': np.True_, 'p1_ok': True, 'p2_ok': True, 'p3_ok': True, 'cv_alpha': 0.12349161338632816, 'cv_beta': 0.023411756640098026, 'a_body': 0.41212091013645175, 'a_approach3': 0.48507313061987906, 'delta_point': 0.07295222048342731, 'delta_boot_std': 0.10726993476247197}


### 7.7 可視化

IsoFLOP プロファイル・フロンティアの描画関数(`plot_isoflop_profile`・`plot_optimal_frontier`)で、現在の学習グリッド(`grid_points`)の点推定(ブートストラップなし)を描画する。`SMOKE_TEST=True`の場合はスモークスケールのデータであり、動作確認以上の意味を持たない。`SMOKE_TEST=False`の場合は本番データの点推定を示すが、7 節で事前登録した判定基準(ブートストラップに基づく支持・反証・判定不能)そのものではない点に注意する。



```python
_PLOT_LABEL = "SMOKE" if SMOKE_TEST else "PRODUCTION"  # SMOKE_TEST に連動させる
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

_body_result_demo = reconstruct_optimal_frontier(grid_points, COMPUTE_BUDGETS, FLOPS_PER_TOKEN_BODY)
plot_isoflop_profile(
    _body_result_demo.profiles, title=f"[{_PLOT_LABEL}] IsoFLOP profiles (body-only)", ax=axes[0]
)

_bo_result_demo = reconstruct_optimal_frontier(grid_points, COMPUTE_BUDGETS, FLOPS_PER_TOKEN_BODY_OUTPUT)
_frontiers_demo = {
    "body-only": (
        _body_result_demo.frontier_log_c,
        _body_result_demo.frontier_log_n_opt,
        _body_result_demo.power_law_fit.exponent if _body_result_demo.power_law_fit else None,
    ),
    "body+output": (
        _bo_result_demo.frontier_log_c,
        _bo_result_demo.frontier_log_n_opt,
        _bo_result_demo.power_law_fit.exponent if _bo_result_demo.power_law_fit else None,
    ),
}
plot_optimal_frontier(_frontiers_demo, title=f"[{_PLOT_LABEL}] Optimal frontier", ax=axes[1])
plt.tight_layout()
plt.show()

```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/009_scaling_laws/output_62_0.png)
    




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/02_pretraining/009_scaling_laws.ipynb
