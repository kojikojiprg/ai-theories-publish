---
title: "スケーリング則(Scaling Laws)(実装・実験編 1/3)"
---

この記事は後編(実装・実験編 1/3)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/009_scaling_laws-theory)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/009_scaling_laws-practice-2)。

## 4. 実装方針 / Implementation Policy

- べき乗則あてはめ・飽和べき乗則あてはめ・IsoFLOP 放物線あてはめ・Chinchilla パラメトリックあてはめ・パラメトリックブートストラップは`src/scaling/laws.py`にスクラッチ実装する(Levenberg-Marquardt 法・IRLS もスクラッチ実装し、`scipy.optimize`等の外部最適化ライブラリには委譲しない。線形の最小二乗解`numpy.linalg.lstsq`のみ利用)。
- トークナイザは 008 で Hugging Face Hub(`kojikojiprg/ai-theories-small-gpt-en`)にアップロードした語彙サイズ 8192 の英語バイトレベル BPE を再利用する(5.2 節)。モデル重みはアップロードしない(009 は複数サイズのモデルを新規に学習するため)。
- モデル本体(`GPTLanguageModel`)・学習ループ(`train_language_model`)・optimizer(`AdamW`)・学習率スケジュール(`compute_warmup_cosine_learning_rate`)は 006・007・008 の実装をそのまま再利用し、本トピックでは変更しない。
- コーパスは`src/data/text.py`に追加した`load_english_wikipedia_corpus()`(006 の英語マニフェストを部分集合として含む形で拡張した`en_009_scaling.json`、5.4 節)を使う。
- IsoFLOP プロファイル・フロンティアの描画関数を`src/utils/visualization.py`に追加する(`plot_isoflop_profile`・`plot_optimal_frontier`)。
- 009 では学習済みチェックポイントを Hugging Face Hub にアップロードしない(多数の小型モデルを学習するため、008 のような「標準モデル」の位置づけを持たせない)。
- `SMOKE_TEST`フラグで第 1 段階(縮小スケール)と本番(Google Colab T4)を切り替える。model サイズの水準(`D_MODEL_LEVELS`)自体は独立変数であるため`SMOKE_TEST`で変更しない。縮小するのはコーパス量・学習ステップ数のみであり、等比刻み・条件間で一致させる値は本番設定との比率構造を保つ。


## 5. 実装 / Implementation

### 5.1 トークナイザの取得

008 で Hugging Face Hub にアップロードした`tokenizer.json`(`merges`・`vocab`・`byte_level`・`chunk_split_mode`・`max_chunk_bytes`・`symbol_to_id`を含む、008 5.12 節参照)を取得し、`BPETokenizer`を再構築する。ネットワーク越しの取得に失敗した場合は、008 と完全に同一の設定(`VOCAB_SIZE=8192`、`byte_level=True`、`max_chunk_bytes=64`)で BPE を再学習し、**語彙サイズが 008 のものと一致することをアサーションで確認する**(フォールバック時の学習コーパスは軽量化のため 008 の学習コーパス量よりは少ないが、語彙サイズ自体は一致するかを検証する。学習コーパス自体が完全一致しない場合、マージ規則そのものが 008 と一致する保証はないため、厳密なマージ規則の一致は要求しない)。



```python
HF_TOKENIZER_REPO_ID = "kojikojiprg/ai-theories-small-gpt-en"
MAX_CHUNK_BYTES = 64  # 006・008 と同一

# 語彙サイズの一致を確認するためのフォールバック学習用テキスト(トークナイザ取得より
# 前の段階なので、コーパス取得(5.4 節)を先取りせず tiny な埋め込みサンプルで代用する)
_fallback_seed_text = (
    "The quick brown fox jumps over the lazy dog. " * 500
    + "Scaling laws describe how loss decreases as compute, data, and model size increase. " * 500
)
tokenizer, _loaded_from_hub = load_bpe_id_tokenizer_from_hub(
    HF_TOKENIZER_REPO_ID,
    fallback_train_text=_fallback_seed_text,
    fallback_vocab_size=VOCAB_SIZE,
    fallback_max_chunk_bytes=MAX_CHUNK_BYTES,
)
print(f"tokenizer vocab_size={tokenizer.vocab_size}, loaded_from_hub={_loaded_from_hub}")
assert tokenizer.vocab_size == VOCAB_SIZE, "トークナイザの語彙サイズが 008(8192)と一致しない"

```


    tokenizer.json:   0%|          | 0.00/661k [00:00<?, ?B/s]


    tokenizer vocab_size=8192, loaded_from_hub=True


### 5.2 ラウンドトリップの検証

008・006 と同様、符号化(encode)と復号(decode)のラウンドトリップが一致することを確認する(不変条件、6 節)。



```python
_roundtrip_sample = (
    "Scaling laws for neural language models describe how test loss improves as a "
    "power-law function of model size, dataset size, and compute budget. "
    "スケーリング則という言葉自体は英語コーパスのみを扱う 009 では登場しないはずだが、"
    "マルチバイト文字を含む文字列でもラウンドトリップが崩れないことを確認しておく。"
)
_roundtrip_ok = tokenizer.decode(tokenizer.encode(_roundtrip_sample)) == _roundtrip_sample
assert _roundtrip_ok, "トークナイザのラウンドトリップが一致しない"
print("OK: ラウンドトリップ一致(マルチバイト文字を含む)")

```

    OK: ラウンドトリップ一致(マルチバイト文字を含む)


### 5.3 スケールの設定(SMOKE_TEST)

`SMOKE_TEST=True`はローカル(Claude Code、第 1 段階)での動作確認・本番実行可能性の検証用の縮小スケールであり、`SMOKE_TEST=False`(Google Colab T4、第 2 段階)が本番設定である。`D_MODEL_LEVELS`(3.2.1 節で確定済み)は独立変数であるため`SMOKE_TEST`で変更しない。縮小するのはコーパス量(取得する記事数)・学習ステップ数・ブートストラップ回数のみである。



```python
SMOKE_TEST = False  # Claude Code はこの True 側のみ実行する(Colab T4 では False に切り替える)
_RUN_LABEL = "スモーク" if SMOKE_TEST else "本番"  # SMOKE_TEST の値に連動させる

# キャッシュディレクトリは言語とデータ源(英語版 Wikipedia)で命名し、トピック番号を
# 含めない(どのトピックが最初に使ったかではなく、何のデータかで決まる)。006・008 など
# 英語版 Wikipedia を使う他トピックともこのディレクトリを共有する。
CACHE_DIR = ROOT / ".cache" / "wikipedia_en"

# --- 計算量予算グリッド設計(6.1 節) ---
# 中央の d_model・中央の計算量予算における目標トークン/パラメータ比(D/N 比)。
# Chinchilla の計算量最適比 D/N ≈ 20 に近づける意図だが、Colab 1 セッション予算(2 時間)・
# 前提条件 P3(データ再利用比率、5.13 節)の両方を満たす範囲に収める必要があり、20 には
# 遠く及ばない値に留める。
#
# 英語コーパスを拡張(マニフェスト記事数 999 -> 9826)し、Hugging Face Hub への
# 再アップロードを完了した。これにより前提条件 P3 の上限(TARGET_TOKENS_PER_PARAM_MID
# の上限)が約 3.88 から約 36.36 に引き上がった一方、Colab 1 セッション予算
# (SESSION_BUDGET_SECONDS)は変わらないため、制約は「P3(データ再利用)」から
# 「セッション時間」に入れ替わった。検討過程・採用根拠は次の Markdown セル
# (5.3.1・5.3.2 節)に実測値・数値表として記す。
#
# 採用: TARGET_TOKENS_PER_PARAM_MID=14(D_MODEL_LEVELS は [32, 64, 96, 128] を維持し、
# 5 水準への拡張は見送った。理由は 5.3.2 節)。
TARGET_TOKENS_PER_PARAM_MID = 14
COMPUTE_BUDGET_RATIO = 2.0  # 公比 2、5 水準
N_COMPUTE_LEVELS = 5

BATCH_SIZE = 32  # 006・008 と揃える
TOKENS_PER_STEP = BATCH_SIZE * SEQUENCE_LENGTH

WARMUP_RATIO = 0.1  # 007・008 と同じ(比例 warmup、Porian らの要因 2 を避ける側に固定)
MIN_LEARNING_RATE_RATIO = 0.01  # 007・008 と同じ
WEIGHT_DECAY = 0.1  # 007・008 と同じ
CLIP_QUANTILE = 0.90  # 007・008 と同じ分位点方式
SESSION_BUDGET_SECONDS = 2 * 60 * 60  # 006・007・008 と同じ基準

# --- ノイズ床(6.1 節) ---
NOISE_FLOOR_SEEDS = 5  # 中央 d_model・中央計算量予算を 5 シードで実行(縮小しない)

# --- ブートストラップ ---
if SMOKE_TEST:
    N_BOOTSTRAP = 60  # 本番: 1000(縮小した旨をここに明記する)
    LM_CORPUS_ARTICLES = 12  # 本番: マニフェスト全体(9826 記事)
    TOKENIZER_TRAIN_BYTES = 60_000  # 本番: 8,000,000(008 と同一)
else:
    N_BOOTSTRAP = 1000
    LM_CORPUS_ARTICLES = None  # None = マニフェスト全体
    TOKENIZER_TRAIN_BYTES = 8_000_000

VALIDATION_RATIO = 0.05
print(f"SMOKE_TEST={SMOKE_TEST}, N_BOOTSTRAP={N_BOOTSTRAP}, NOISE_FLOOR_SEEDS={NOISE_FLOOR_SEEDS}")
print(f"D_MODEL_LEVELS={D_MODEL_LEVELS}")

```

    SMOKE_TEST=False, N_BOOTSTRAP=1000, NOISE_FLOOR_SEEDS=5
    D_MODEL_LEVELS=[32, 64, 96, 128]


### 5.3.1 拡張コーパスに基づく計算量予算の再設計

**背景**: 過去の本番実行では、前提条件 P1(IsoFLOP プロファイルの最小値が掃引範囲の内点にあること)が繰り返し不成立(または内点予算が要求数に届かない)だった。診断の結果、根本原因は前提条件 P3(データ再利用が訓練コーパスの`EPOCH_REUSE_LIMIT`=4.0 倍以内、5.13 節)の天井にあることが判明した。計算量予算を上げないと最適点が掃引範囲の内側に入らないが、予算を上げると P3 に抵触するという構造である。

この制約を解消するため、英語コーパスを拡張し(マニフェスト`en_009_scaling.json`の記事数を 999 から 9826 に拡張、詳細は`src/data/wikipedia_manifests/README.md`)、Hugging Face Hub のデータセット(`kojikojiprg/ai-theories-corpus-en`)への再アップロードが完了した。以降のコードセルで、拡張後のコーパスを実際に Hub から取得し、実測バイト数・実測訓練トークン数を確認する(本番学習(SMOKE_TEST 分岐、5.12 節)とは独立な、設計を確定させるための事前実測)。

**制約の入れ替わり**: コーパスが約 9 倍規模になったため、P3 の天井(`TARGET_TOKENS_PER_PARAM_MID`の上限)も同程度引き上がる(次のコードセルで正確な倍率を実測する)。一方、Colab 1 セッションの予算(`SESSION_BUDGET_SECONDS`=7200 秒=120 分)は変わらない。学習時間は計算量予算にほぼ比例するため、**これまでとは逆に、P3(データ再利用)ではなくセッション時間が効く領域に入る**。

**前提不成立への対応としての位置づけ**: 判定基準(対比量の定義・閾値の導出式・期待する差の方向)・前提条件の閾値(P3 の 4.0 倍、P0 の閾値)は一切変更しない。変更するのは計算量予算(`TARGET_TOKENS_PER_PARAM_MID`)と掃引水準(`D_MODEL_LEVELS`の再検討)のみである。修正の根拠は「最適点が掃引範囲の端に位置し、放物線の頂点を推定できない」という実験デザイン上の問題であり、観測結果の方向(どの仮説が支持されたか)には依存しない。

**方法**: $N_{opt} \propto C^a$($a\approx0.5$、Chinchilla)を仮定した較正済みモデル $N_{opt} = 0.0276936\sqrt{C}$(過去の本番実行で得られた実測点 $C=1.41\times10^{13}, N_{opt}\approx104{,}000$ から較正)を用いて、候補設計ごとに 5 つの計算量予算での予測 $\log N_{opt}$ が掃引範囲のどこに来るかを評価する。時間見積もりには、過去の本番実行(Google Colab、T4 GPU)で実測した学習グリッド実行時間 731.2 秒(旧設計`D_MODEL_LEVELS=[32,64,96,128,192]`・`TARGET_TOKENS_PER_PARAM_MID=3.0`、25 セル、較正を含まない実行時間そのもの)から実効スループットを逆算し、任意の設計の時間見積もりに適用する、5.11.1 節と同じ方法論を用いる(ローカル(Mac、MPS)での実測はオーバーヘッド支配的で外挿の役に立たないため使わない、5.11.1 節)。

**この較正済みモデル・スループットはいずれも実測点 1 つからの粗い外挿であり、実際に前提条件 P1 を満たす保証はない。満たされなかった場合も、結論を曲げずにそのまま報告する。**



```python
# 拡張後コーパスの実測(本番学習(SMOKE_TEST 分岐)とは独立な事前実測)。前提条件
# P3(5.13 節)の上限を計算するために、この設計確定の時点で実測トークン数が分かって
# いる必要がある。取得は Hub 経由のため数秒、符号化(BPE のスクラッチ実装)はコーパス
# 全体で 1〜3 分程度を要する。
_full_raw_text, _full_corpus_meta = load_english_wikipedia_corpus_with_fallback(
    CACHE_DIR, return_metadata=True
)
_full_actual_bytes = len(_full_raw_text.encode("utf-8"))
assert _full_actual_bytes == _full_corpus_meta["raw_bytes"], (
    "拡張後コーパスの実測バイト数がアップロード時の記録値と一致しない"
)
assert _full_corpus_meta["fetched_article_count"] == _full_corpus_meta["manifest_article_count"], (
    "拡張後コーパスに取得失敗記事が残っている"
)
print(f"拡張後コーパス取得元: {_full_corpus_meta['source']}")
print(f"実測バイト数: {_full_actual_bytes:,}(アップロード時の記録値と一致)")
print(
    f"記事数: {_full_corpus_meta['fetched_article_count']:,} / "
    f"{_full_corpus_meta['manifest_article_count']:,}(スキップ 0 件)"
)

_full_train_text, _full_val_text = split_train_val_text(_full_raw_text, VALIDATION_RATIO)
# 009 コーパスのプレーンテキスト化・memmap 対応: encode_text_to_memmap() で uint16 の
# numpy.memmap として符号化する(トークン ID を Python の list[int] として保持すると
# Google Colab の RAM 制約(12 GB)を圧迫するため。src/data/text.py 参照)。
_full_train_memmap = encode_text_to_memmap(
    tokenizer, _full_train_text, CACHE_DIR / "memmap" / "measured_full_train_ids.memmap"
)
MEASURED_TRAIN_TOKENS_EXPANDED = len(_full_train_memmap)
print(f"実測訓練トークン数(train_ids 長): {MEASURED_TRAIN_TOKENS_EXPANDED:,}")

# 拡張前の実測訓練トークン数(比較用)
_OLD_TRAIN_TOKENS = 15_458_326
print(f"拡張前の実測訓練トークン数(履歴): {_OLD_TRAIN_TOKENS:,}")
print(f"拡張倍率(訓練トークン数): {MEASURED_TRAIN_TOKENS_EXPANDED / _OLD_TRAIN_TOKENS:.2f}倍")

```


    corpus.txt: reconstructing file:   0%|          |  0.00B /  546MB            



    corpus.txt: downloading bytes:           |  0.00B            



    metadata.json:   0%|          | 0.00/269 [00:00<?, ?B/s]


    コーパス取得元: kojikojiprg/ai-theories-corpus-en(Hugging Face Hub)
    拡張後コーパス取得元: hub
    実測バイト数: 545,716,632(アップロード時の記録値と一致)
    記事数: 9,826 / 9,826(スキップ 0 件)
    .cache/wikipedia_en/memmap/measured_full_train_ids.memmap: 144,844,795 トークンを uint16 memmap として書き出した
    実測訓練トークン数(train_ids 長): 144,844,795
    拡張前の実測訓練トークン数(履歴): 15,458,326
    拡張倍率(訓練トークン数): 9.37倍



```python
# 前提条件 P3 の上限(tau_max)と、時間予算からの許容範囲を実測トークン数から計算し、
# 候補設計(D_MODEL_LEVELS x TARGET_TOKENS_PER_PARAM_MID)を比較する。
EPOCH_REUSE_LIMIT_PREVIEW = 4.0  # 5.13 節で EPOCH_REUSE_LIMIT として正式に宣言する値と同じ


def _tau_max_for(mid_d, min_d, ratio, k_max):
    n_mid = _measured_N[mid_d]
    n_min = _measured_N[min_d]
    return EPOCH_REUSE_LIMIT_PREVIEW * n_min * MEASURED_TRAIN_TOKENS_EXPANDED / (n_mid**2 * ratio**k_max)


_tau_max_4lvl = _tau_max_for(96, 32, COMPUTE_BUDGET_RATIO, 2)
print(f"前提条件 P3 の上限(tau_max、D_MODEL_LEVELS=[32,64,96,128]): {_tau_max_4lvl:.2f}")


# 較正済み N_opt(C) モデル(a=0.5 仮定、過去の本番実行で得られた実測点から較正、5.3.1 節)
def _n_opt_calibrated(c):
    return 0.0276936 * c**0.5


# T4 換算の時間モデル: 過去の本番実行(Colab T4)で実測した学習グリッド実行時間
# 731.2 秒(旧設計 D_MODEL_LEVELS=[32,64,96,128,192]・TARGET_TOKENS_PER_PARAM_MID=3.0、
# 25 セル、較正を含まない)を実効スループットに逆算し、任意の設計の時間見積もりに使う
# (5.11.1 節と同じ方法論)。
_OLD_GRID_SECONDS = 731.2
_OLD_LEVELS = [32, 64, 96, 128, 192]
_OLD_TAU = 3.0
_old_n_mid = _measured_N[96]
_old_c_mid = 6 * _OLD_TAU * _old_n_mid**2
_old_total_flops = len(_OLD_LEVELS) * 7.75 * _old_c_mid  # 5 水準 x (budget比 0.25+0.5+1+2+4=7.75)
EFFECTIVE_THROUGHPUT = _old_total_flops / _OLD_GRID_SECONDS
print(f"実効スループット(T4 換算、過去の実測値から逆算): {EFFECTIVE_THROUGHPUT:.3e} FLOPs/s")

NUM_LR_TRIALS_ASSUMED = 2  # ローカル実行(較正)で適応的探索が実際に試行した回数

# 5 水準候補(d_model=192 を含む)の比較のため、D_MODEL_LEVELS(4 水準)のみを対象に
# 実測した _measured_N(5.2 節)を、192 について build_gpt_model で追加測定して拡張する。
_N_for_candidates = dict(_measured_N)
if 192 not in _N_for_candidates:
    _N_for_candidates[192] = count_non_embedding_parameters(build_gpt_model(192))
    print(f"d_model=192 の追加実測: N={_N_for_candidates[192]:,}(5 水準候補の比較用)")


def design_estimate(levels, tau):
    mid_d = levels[len(levels) // 2]
    n_mid = _N_for_candidates[mid_d]
    n_min = _N_for_candidates[levels[0]]
    n_max = _N_for_candidates[levels[-1]]
    c_mid = 6 * tau * n_mid**2
    budgets = [c_mid * COMPUTE_BUDGET_RATIO**k for k in range(-2, 3)]
    n_levels = len(levels)
    grid_noise_flops = (n_levels * 7.75 + 4) * c_mid
    calib_flops = 4 * c_mid * (NUM_LR_TRIALS_ASSUMED + 1 + n_min / n_mid + n_max / n_mid)
    total_time = (grid_noise_flops + calib_flops) / EFFECTIVE_THROUGHPUT
    min_steps = round(budgets[0] / (6 * n_max) / TOKENS_PER_STEP)
    p3_ratio = (budgets[-1] / (6 * n_min)) / MEASURED_TRAIN_TOKENS_EXPANDED
    log_n_min, log_n_max = np.log(n_min), np.log(n_max)
    margins = []
    n_interior = 0
    for c in budgets:
        log_n_opt = np.log(_n_opt_calibrated(c))
        interior = log_n_min < log_n_opt < log_n_max
        if interior:
            n_interior += 1
        margins.append((log_n_opt - log_n_min, log_n_max - log_n_opt))
    return dict(
        levels=levels,
        tau=tau,
        c_mid=c_mid,
        budgets=budgets,
        total_time=total_time,
        budget_fraction=total_time / SESSION_BUDGET_SECONDS,
        min_steps=min_steps,
        p3_ratio=p3_ratio,
        n_interior=n_interior,
        margins=margins,
    )


print(f"\n{'levels':<24}{'tau':>6}{'time(min)':>11}{'budget%':>9}{'min_steps':>11}{'P3':>7}{'interior':>10}")
_candidates = [([32, 64, 96, 128], t) for t in (11, 12, 13, 14, 15)] + [
    ([32, 64, 96, 128, 192], t) for t in (10, 11, 12, 13, 14)
]
for levels, tau in _candidates:
    r = design_estimate(levels, tau)
    print(
        f"{str(levels):<24}{tau:>6.1f}{r['total_time'] / 60:>11.1f}{r['budget_fraction']:>9.1%}"
        f"{r['min_steps']:>11d}{r['p3_ratio']:>7.2f}{r['n_interior']:>10d}/5"
    )

_chosen = design_estimate(D_MODEL_LEVELS, TARGET_TOKENS_PER_PARAM_MID)
print(f"\n採用設計: D_MODEL_LEVELS={D_MODEL_LEVELS}, TARGET_TOKENS_PER_PARAM_MID={TARGET_TOKENS_PER_PARAM_MID}")
print(f"  推定時間: {_chosen['total_time'] / 60:.1f} 分({_chosen['budget_fraction']:.1%} of budget)")
print(f"  最小ステップ数(予測): {_chosen['min_steps']}")
print(f"  P3 比率(予測): {_chosen['p3_ratio']:.3f}倍(上限 {EPOCH_REUSE_LIMIT_PREVIEW}倍)")
print(f"  内点予算数(予測): {_chosen['n_interior']}/5")
for c, (ml, mr) in zip(_chosen["budgets"], _chosen["margins"], strict=True):
    print(
        f"    C={c:.3e} N_opt(予測)={_n_opt_calibrated(c):,.0f} "
        f"margin_L={ml:+.3f} margin_R={mr:+.3f}(掃引 1 水準={np.log(COMPUTE_BUDGET_RATIO):.3f})"
    )

assert _chosen["budget_fraction"] < 1.0, "採用設計の時間見積もりがセッション予算を超えている"
assert _chosen["p3_ratio"] < EPOCH_REUSE_LIMIT_PREVIEW, "採用設計が前提条件 P3 の上限を超えている"
assert _chosen["n_interior"] == 5, "採用設計で IsoFLOP プロファイルの最小値が全予算で内点にならない"
print("\nOK: 採用設計が時間予算・P3・内点条件をすべて満たす(予測時点、本番実行前)")

```

    前提条件 P3 の上限(tau_max、D_MODEL_LEVELS=[32,64,96,128]): 36.36
    実効スループット(T4 換算、2 回目実測から逆算): 1.874e+11 FLOPs/s
    d_model=192 の追加実測: N=1,771,200(5 水準候補の比較用)
    
    levels                     tau  time(min)  budget%  min_steps     P3  interior
    [32, 64, 96, 128]         11.0       62.9    52.4%         84   1.21         5/5
    [32, 64, 96, 128]         12.0       68.6    57.2%         91   1.32         5/5
    [32, 64, 96, 128]         13.0       74.3    61.9%         99   1.43         5/5
    [32, 64, 96, 128]         14.0       80.1    66.7%        107   1.54         5/5
    [32, 64, 96, 128]         15.0       85.8    71.5%        114   1.65         5/5
    [32, 64, 96, 128, 192]    10.0       74.6    62.2%         34   1.10         4/5
    [32, 64, 96, 128, 192]    11.0       82.1    68.4%         37   1.21         5/5
    [32, 64, 96, 128, 192]    12.0       89.5    74.6%         41   1.32         5/5
    [32, 64, 96, 128, 192]    13.0       97.0    80.8%         44   1.43         5/5
    [32, 64, 96, 128, 192]    14.0      104.5    87.1%         47   1.54         5/5
    
    採用設計: D_MODEL_LEVELS=[32, 64, 96, 128], TARGET_TOKENS_PER_PARAM_MID=14
      推定時間: 80.1 分(66.7% of budget)
      最小ステップ数(予測): 107
      P3 比率(予測): 1.540倍(上限 4.0倍)
      内点予算数(予測): 5/5
        C=4.126e+12 N_opt(予測)=56,250 margin_L=+0.132 margin_R=+2.639(掃引 1 水準=0.693)
        C=8.251e+12 N_opt(予測)=79,549 margin_L=+0.478 margin_R=+2.292(掃引 1 水準=0.693)
        C=1.650e+13 N_opt(予測)=112,499 margin_L=+0.825 margin_R=+1.945(掃引 1 水準=0.693)
        C=3.300e+13 N_opt(予測)=159,098 margin_L=+1.171 margin_R=+1.599(掃引 1 水準=0.693)
        C=6.601e+13 N_opt(予測)=224,999 margin_L=+1.518 margin_R=+1.252(掃引 1 水準=0.693)
    
    OK: 採用設計が時間予算・P3・内点条件をすべて満たす(予測時点、本番実行前)


### 5.3.2 決定: 採用した設計とその根拠

**採用**: `D_MODEL_LEVELS=[32,64,96,128]`(変更なし)・`TARGET_TOKENS_PER_PARAM_MID=14`(3.7 から引き上げ)。

**なぜ 5 水準(`d_model=192`の復帰)を採用しなかったか**: 上のコードセルの候補比較表が示す通り、`d_model=192`を加えても前提条件 P1(内点予算数)は改善しない。IsoFLOP プロファイルの最小値の予測位置(較正済みモデル $N_{opt}(C)$)は掃引範囲の **左端(`d_model=32`)側で最も窮屈** であり(5 つの予算のうち margin_L が最小のもので +0.132 しかない一方、margin_R はどの予算でも +1.25 以上と大きく余裕がある)、この左端の余裕は`D_MODEL_LEVELS`の中央(`d_model=96`)・最小(`d_model=32`)のみで決まり、上限側に`d_model=192`を追加しても変化しない。一方`d_model=192`を追加すると、最小ステップ数(掃引の最大`d_model`・最小計算量予算の組で決まる)が`d_model=128`のときの 107(`tau=14`)から 37(`tau=11`、ほぼ同程度の時間予算)へと大きく悪化する。時間予算をさらに投入して`tau`を上げても(候補比較表の 5 水準`tau=14`行)、5 水準構成の最小ステップ数は 47 にとどまり、4 水準構成の`tau=11`(84 ステップ)にすら及ばない。**5 水準化は、既に十分な余裕がある右側マージンをさらに広げるだけで、実際に窮屈な左側マージンにも、最小ステップ数にも寄与しない。** よって 4 水準を維持する。

**幾何学的な限界(参考)**: 5 つの計算量予算(公比 2)は、較正済みモデルが $N_{opt}\propto\sqrt{C}$ であるため、$\ln N_{opt}$ を合計 $\ln(2)\times2=1.386$ だけ動かす(計算量予算の対数レンジ $\ln(2)\times4=2.773$ の半分)。4 水準の掃引範囲の対数幅は $\ln(787072/49312)=2.770$ であり、両端に掃引 1 水準分($\ln 2=0.693$)の余裕を同時に確保するには合計 $1.386+2\times0.693=2.772$ の幅が必要で、ほぼ限界に等しい(`tau`を上げても両端が同時に 1 水準分の余裕を持つのはほぼ不可能で、`tau\approx43`でようやく両端が閾値ぎりぎりで釣り合うが、この`tau`はセッション予算を大幅に超える)。**したがって「左端・右端の双方から掃引水準 1 つ分の距離を確保する」という望ましい条件は、この設計・この較正済みモデルの範囲では達成できない。** 採用した`tau=14`では、5 予算中 3 つ(中央から上側)がこの基準を満たし、残り 2 つ(下側、特に最下端)は内点ではあるが余裕が薄い。これは、右側に大きな余裕を残したまま左側だけを改善する自由度がこの設計にはないという構造的な理由による(較正済みモデルによれば`N_opt`は掃引範囲の中央`d_model=96`よりも一貫して小さい値を取り続けるため)。

**`TARGET_TOKENS_PER_PARAM_MID=14`を選んだ理由**: 候補比較表では`tau=13`で最小ステップ数がちょうど 99(閾値 100 未達)、`tau=14`で 107(閾値超過)となる。`tau=14`はセッション予算の 66.7% を使う見積もりであり、過去の本番実行がセッション終了によるデータ消失を経験している(5.14.1 節)ことを踏まえ、**残り約 33% を較正のばらつき・想定外の遅延に対する安全余裕として残す**(セッション予算をほぼ 100% 使い切る`tau\approx21`のような設計は、安全余裕がほぼゼロになるため採用しない)。

**前提条件 P3 は依然大きく余裕がある**: 採用設計での P3 比率(予測)は 1.54 倍(上限 4.0 倍)であり、拡張前の設計(3.7 倍、上限 3.88 倍とほぼ同水準)と比べて大幅に余裕ができた。今回の再設計を最終的に制約したのはセッション時間であり、P3 ではない(5.3.1 節で述べた通り)。

**最小ステップ数の閾値**: `MIN_STEPS_FLOOR`を **100** に引き上げる(6 節)。採用設計での予測値は 107 であり、閾値に対して約 7% の余裕がある。

**掃引水準の数**: 4 水準のまま変更しない。5 水準化による効果(IsoFLOP プロファイル 1 本あたりの点数増加、放物線あてはめの精度向上)よりも、最小ステップ数の悪化という実質的な悪影響のほうが大きいと判断した。


### 5.4 コーパスの取得

英語版 Wikipedia を 60〜100 MB 規模で取得する(`load_english_wikipedia_corpus_with_fallback`、`src/data/text.py`。Hugging Face Hub のデータセットから取得できない場合のみ `load_english_wikipedia_corpus` による直接取得にフォールバックする)。マニフェスト`en_009_scaling.json`は、006 の 356 記事(`en_006_pretraining.json`)に、Wikipedia の Special:LongPages(長大記事一覧、API `list=querypage&qppage=Longpages`)から追加で選定した記事を加えた計 9826 記事からなる(内訳・拡張経緯は`src/data/wikipedia_manifests/README.md`および 5.3.1 節を参照)。006 のマニフェストを部分集合として含む形で拡張しているため、006・008 で取得済みのキャッシュをそのまま再利用できる。

**実行時点のコーパスサイズについての注記**: 本番実行(Google Colab、`SMOKE_TEST=False`)で実際に取得したコーパスは 60,851,631 UTF-8 バイト(約 60.85 MB)であった。事前の概算(006 の 356 記事から実測した抽出比率 約 14.1% を追加記事の wikitext 合計に適用した見積もり、60 MB 弱)とよく一致しており、目標レンジ(60〜100 MB)の下限付近である。



```python
# SMOKE_TEST では記事数を絞ったマニフェストを一時的に使う。これは等比刻みなどの構造を
# 持たない単純な「取得する記事数」の縮小であり、縮小規則を保つべき等比数列や条件間で
# 一致させる値ではなく、記事の取得件数という単純なスカラー量である。
_full_manifest_path = Path("src/data/wikipedia_manifests/en_009_scaling.json")
_full_manifest = json.loads(_full_manifest_path.read_text(encoding="utf-8"))
print(f"マニフェスト全体: {len(_full_manifest)} 記事")

if SMOKE_TEST and LM_CORPUS_ARTICLES is not None:
    _smoke_manifest = dict(list(_full_manifest.items())[:LM_CORPUS_ARTICLES])
    _smoke_manifest_path = CACHE_DIR / "_smoke_manifest.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _smoke_manifest_path.write_text(
        json.dumps(_smoke_manifest, ensure_ascii=False), encoding="utf-8"
    )
    from src.data.text import load_wikipedia_corpus

    t0 = time.time()
    raw_text = load_wikipedia_corpus("en", CACHE_DIR, manifest_path=_smoke_manifest_path)
    print(f"スモークコーパス取得({LM_CORPUS_ARTICLES} 記事): {time.time() - t0:.2f} s")
else:
    # まず Hugging Face Hub のデータセット(kojikojiprg/ai-theories-corpus-en、
    # scripts/promote_canonical_corpora.ipynb でアップロードしたもの)から取得を
    # 試み、取得できない場合のみ Wikipedia API からの直接取得にフォールバックする
    # (取得元は load_english_wikipedia_corpus_with_fallback() 内で標準出力に明記される)。
    t0 = time.time()
    raw_text, _fetch_metadata = load_english_wikipedia_corpus_with_fallback(
        CACHE_DIR, return_metadata=True
    )
    print(f"本番コーパス取得(全 {len(_full_manifest)} 記事): {time.time() - t0:.2f} s")
    print(f"取得元: {_fetch_metadata['source']}")

    # 取得できたバイト数・記事数が期待値と一致することを確認する(2.1 節)。Hub 経由の
    # 取得は決定的であるため、実際に取得したバイト数がアップロード時に corpus.json へ
    # 記録された raw_bytes と一致しなければ、ダウンロードが途中で壊れている(切り詰め・
    # ネットワーク障害など)ことを意味する。また、アップロード元
    # (scripts/promote_canonical_corpora.ipynb)側で既にスキップ件数ゼロのアサーション
    # を課しているため、fetched_article_count は manifest_article_count と一致するはず
    # である(万一 Hub 側のデータが古い・不整合な場合に備えて、ここでも独立に確認する)。
    # フォールバック(直接取得)経路を通った場合は、その場で取得したスキップ件数が
    # 実際にゼロであることを確認する。
    if _fetch_metadata["source"] == "hub":
        _actual_bytes = len(raw_text.encode("utf-8"))
        assert _actual_bytes == _fetch_metadata["raw_bytes"], (
            "Hub から取得したコーパスの実際のバイト数が、アップロード時に corpus.json へ"
            f"記録された値と一致しない(実際: {_actual_bytes:,} バイト / "
            f"記録値: {_fetch_metadata['raw_bytes']:,} バイト)。"
            "ダウンロードが破損している可能性がある。"
        )
        assert (
            _fetch_metadata["fetched_article_count"] == _fetch_metadata["manifest_article_count"]
        ), (
            "Hub のコーパスが、一部記事の取得に失敗した状態のままアップロードされている"
            f"(取得できた記事数 {_fetch_metadata['fetched_article_count']} / "
            f"マニフェスト記事数 {_fetch_metadata['manifest_article_count']})。"
            "scripts/promote_canonical_corpora.ipynb 側のスキップ件数ゼロのアサーションを"
            "経ていないデータがアップロードされている可能性がある。"
        )
        print(
            f"[OK] Hub から取得したバイト数({_actual_bytes:,})がアップロード時の記録値と一致した。"
        )
    else:
        assert not _fetch_metadata["skipped_articles"], (
            f"直接取得で {len(_fetch_metadata['skipped_articles'])} 件の記事取得に失敗した"
            f"(スキップ: {[a['title'] for a in _fetch_metadata['skipped_articles']]})。"
        )
        print("[OK] 直接取得でスキップ件数がゼロであることを確認した。")

_raw_bytes = len(raw_text.encode("utf-8"))
print(f"取得したコーパス: {len(raw_text):,} 文字 / {_raw_bytes:,} UTF-8 バイト")

train_text, val_text = split_train_val_text(raw_text, VALIDATION_RATIO)
total_eval_bytes = len(val_text.encode("utf-8"))
print(
    f"train_text: {len(train_text):,} 文字, val_text: {len(val_text):,} 文字 "
    f"({total_eval_bytes:,} UTF-8 バイト)"
)

```

    マニフェスト全体: 9826 記事
    コーパス取得元: kojikojiprg/ai-theories-corpus-en(Hugging Face Hub)
    本番コーパス取得(全 9826 記事): 5.99 s
    取得元: hub
    [OK] Hub から取得したバイト数(545,716,632)がアップロード時の記録値と一致した。
    取得したコーパス: 542,892,375 文字 / 545,716,632 UTF-8 バイト
    train_text: 515,747,757 文字, val_text: 27,144,618 文字 (27,354,400 UTF-8 バイト)


### 5.5 コーパス取得のスケーリング計測・外挿

記事取得件数を 3 点以上振って所要時間を実測し、$\log t = \log a + b \log n$ のあてはめでべき指数 $b$ を推定して、マニフェスト全体(1000 記事)への外挿値を出す。取得はキャッシュされるため(`load_wikipedia_corpus`の記事単位キャッシュ)、計測に使った記事は以降の呼び出しで再取得されない。



```python
from src.data.text import _fetch_wikipedia_revision_plaintext  # スケーリング計測専用の内部関数


def measure_corpus_fetch_scaling(manifest: dict[str, int], article_counts: list[int]):
    '''記事取得件数を振って所要時間を実測する(006・008 の measure_*_scaling と同じ手法)。

    Wikimedia API のレート制限(429)による指数バックオフ待機が計測に混入すると、
    定常状態の 1 記事あたり時間から大きく外れた外れ値になり、べき乗則のあてはめが
    壊れる(1 回の待機だけで数百〜数千秒に達しうる)。計測専用に再試行回数・待機秒数を
    絞り(``max_retries=2, retry_wait_seconds=2.0``、既定値より小さい)、失敗した記事は
    スキップして計測対象から除外することで、この混入を避ける(スキップされた記事は
    本番の全件取得(``load_english_wikipedia_corpus``)では既定の再試行設定で再取得される
    ため、コーパスの完全性には影響しない)。プローブ水準の間に間隔を空け、レート制限の
    窓が回復する時間を与える。
    '''
    items = list(manifest.items())
    counts, times, per_article_times = [], [], []
    cumulative = 0.0
    prev_n = 0
    fetched = 0
    idx = prev_n
    for n in article_counts:
        while fetched < n and idx < len(items):
            title, revid = items[idx]
            idx += 1
            t0 = time.time()
            try:
                _fetch_wikipedia_revision_plaintext(
                    "en", title, revid, max_retries=2, retry_wait_seconds=2.0
                )
            except Exception as e:  # noqa: BLE001  # 計測専用: 失敗した記事はスキップする
                print(f"  (スキップ: {title!r}: {e!r})")
                continue
            elapsed = time.time() - t0
            cumulative += elapsed
            per_article_times.append(elapsed)
            fetched += 1
        counts.append(fetched)
        times.append(cumulative)
        time.sleep(3.0)  # レート制限の窓が回復する時間を与える
    return counts, times, per_article_times


# 未取得の記事で計測するため、_smoke_manifest / _full_manifest の先頭からは重複しない範囲を使う
_scaling_probe_manifest = dict(list(_full_manifest.items())[500:530])
_article_counts = [4, 8, 12]
_fetch_counts, _fetch_times, _fetch_per_article = measure_corpus_fetch_scaling(
    _scaling_probe_manifest, _article_counts
)
for n, t in zip(_fetch_counts, _fetch_times, strict=True):
    print(f"記事取得 {n} 件: 累積 {t:.2f} s")

_fetch_fit = fit_power_law(_fetch_counts, _fetch_times)
_fetch_extrapolated = _fetch_fit.coefficient * len(_full_manifest) ** _fetch_fit.exponent

# ロバスト性のガード: べき乗則あてはめは、観測範囲(12 記事)から目標(1000 記事、約 80 倍)
# まで大きく外挿するため、指数のわずかな乖離(レート制限による 1 回の待機など、ネット
# ワーク由来の外れ値)が外挿値を極端に増幅しうる(例: 指数が 1.0 から 1.6 に乖離しただけ
# で、80 倍の外挿では約 80^0.6 ≈ 18 倍も過大評価する)。指数の妥当範囲チェックだけでは
# 検出できないため、常に 1 記事あたり時間の中央値(レート制限のような単発の外れ値の
# 影響を受けにくい)による線形外挿も並行して計算し、両者が大きく乖離する場合は保守的な
# (小さい方の)中央値ベースの外挿を採用する(観測結果の方向に依存しない一般的な
# ロバスト統計の手法である)。
_median_per_article = float(np.median(_fetch_per_article))
_median_extrapolated = _median_per_article * len(_full_manifest)
_disagreement_ratio = max(_fetch_extrapolated, _median_extrapolated) / max(
    min(_fetch_extrapolated, _median_extrapolated), 1e-9
)
print(f"\nべき指数 b={_fetch_fit.exponent:.3f}(1 記事あたり一定時間なら b~=1)")
print(f"べき乗則による外挿: {_fetch_extrapolated:.1f} s ({_fetch_extrapolated / 60:.1f} 分)")
print(
    f"1 記事あたり時間の中央値({_median_per_article:.2f} s)による線形外挿: "
    f"{_median_extrapolated:.1f} s ({_median_extrapolated / 60:.1f} 分)"
)
if _disagreement_ratio > 3.0:
    _chosen_extrapolated = min(_fetch_extrapolated, _median_extrapolated)
    print(
        f"警告: 2 つの外挿が {_disagreement_ratio:.1f} 倍乖離している"
        "(観測範囲からの外挿倍率が大きく、外れ値の影響を受けやすいため)。"
        "保守的な(小さい方の)推定を採用する。"
    )
else:
    _chosen_extrapolated = _fetch_extrapolated
print(
    f"採用する外挿: マニフェスト全体({len(_full_manifest)} 記事)で {_chosen_extrapolated:.1f} s"
    f"(= {_chosen_extrapolated / 60:.1f} 分、条件数 x シード数 = 1 回のみ実行、キャッシュ後は再取得なし)"
)

```

    記事取得 4 件: 累積 6.05 s
    記事取得 8 件: 累積 22.44 s
    記事取得 12 件: 累積 36.69 s
    
    べき指数 b=1.667(1 記事あたり一定時間なら b~=1)
    べき乗則による外挿: 2833627.9 s (47227.1 分)
    1 記事あたり時間の中央値(1.55 s)による線形外挿: 15185.0 s (253.1 分)
    警告: 2 つの外挿が 186.6 倍乖離している(観測範囲からの外挿倍率が大きく、外れ値の影響を受けやすいため)。保守的な(小さい方の)推定を採用する。
    採用する外挿: マニフェスト全体(9826 記事)で 15185.0 s(= 253.1 分、条件数 x シード数 = 1 回のみ実行、キャッシュ後は再取得なし)


### 5.6 符号化のスケーリング計測・外挿

学習データ全体の符号化(encode)は 1 回だけ行われる処理だが、超線形にコストが増加すると本番実行時間の見積もりを大きく崩すため、006・008 と同じ手法でスケーリングを計測する。



```python
def measure_encode_scaling(tok, text: str, sizes_chars: list[int]):
    sizes, times = [], []
    for size in sizes_chars:
        prefix = text[:size]
        sizes.append(len(prefix))
        t0 = time.time()
        tok.encode(prefix)
        times.append(time.time() - t0)
    return sizes, times


_encode_scaling_sizes = [s for s in (5_000, 15_000, 40_000) if s <= len(train_text)]
if len(_encode_scaling_sizes) < 3:
    _encode_scaling_sizes = [len(train_text) // 4, len(train_text) // 2, len(train_text)]

_enc_sizes, _enc_times = measure_encode_scaling(tokenizer, train_text, _encode_scaling_sizes)
for s, t in zip(_enc_sizes, _enc_times, strict=True):
    print(f"符号化: {s:,} chars -> {t:.4f} s")

_enc_fit = fit_power_law(_enc_sizes, _enc_times)
_production_corpus_chars = 100_000_000  # 目標上限(100 MB)を安全側の外挿対象とする
_enc_extrapolated = _enc_fit.coefficient * _production_corpus_chars**_enc_fit.exponent
print(f"\nべき指数 b={_enc_fit.exponent:.3f}(線形なら b~=1)")
print(
    f"本番コーパス上限({_production_corpus_chars:,} 文字)への外挿(学習用+検証用 x2 回): "
    f"{2 * _enc_extrapolated:.1f} s"
)

```

    符号化: 5,000 chars -> 0.0015 s
    符号化: 15,000 chars -> 0.0042 s
    符号化: 40,000 chars -> 0.0098 s
    
    べき指数 b=0.905(線形なら b~=1)
    本番コーパス上限(100,000,000 文字)への外挿(学習用+検証用 x2 回): 23.6 s


### 5.7 コーパスの符号化・評価窓の作成

全`d_model`条件で同一の訓練データ・検証データ(evaluation windows)を共有する(不変条件、6 節)。

訓練データ(`train_ids`)は`encode_text_to_memmap`(`src/data/text.py`)で`uint16`の`numpy.memmap`として符号化する。トークン ID を Python の`list[int]`として保持すると、大規模コーパス(1 億トークン超)では整数オブジェクトのオーバーヘッドにより数 GB の RAM を消費し、Google Colab の RAM 制約(12 GB)を超えるため(背景は本節のセル実行結果、および`src/data/text.py`の`encode_text_to_memmap`の docstring を参照)。検証データ(`val_ids`)は規模が小さいため、従来通り`encode_corpus`で in-memory に符号化する。


```python
TRAIN_IDS_MEMMAP_PATH = CACHE_DIR / "memmap" / "train_ids.memmap"
train_ids = encode_text_to_memmap(tokenizer, train_text, TRAIN_IDS_MEMMAP_PATH)
val_ids = encode_corpus(tokenizer, val_text)
eval_windows, eval_mask = make_evaluation_windows(val_ids, SEQUENCE_LENGTH)
print(
    f"train_ids: {len(train_ids):,} トークン, val_ids: {len(val_ids):,} トークン, "
    f"eval_windows: {tuple(eval_windows.shape)}"
)

```

    .cache/wikipedia_en/memmap/train_ids.memmap: 144,844,795 トークンを uint16 memmap として書き出した
    train_ids: 144,844,795 トークン, val_ids: 9,036,810 トークン, eval_windows: (35301, 256)


### 5.8 学習グリッドの定義(計算量予算)

6.1 節の設計に従い、計算量予算を公比 2 で 5 水準構成する。グリッドの構成には **本体のみの数え方**($C = 6ND$、Kaplan らの流儀)を用いる。各セルのステップ数は $D = C / (6N)$ から決める。中央の`d_model`における目標トークン/パラメータ比`TARGET_TOKENS_PER_PARAM_MID`を起点に、中央の計算量予算 $C_{mid} = 6 \times \tau \times N_{mid}^2$ を定め、公比 2 で前後 2 水準ずつ計 5 水準を作る。



```python
FLOPS_PER_TOKEN_BODY = {
    d: estimate_flops_per_token(
        _measured_N[d], d, N_LAYER, SEQUENCE_LENGTH, VOCAB_SIZE, include_output_layer=False
    )
    for d in D_MODEL_LEVELS
}
FLOPS_PER_TOKEN_BODY_OUTPUT = {
    d: estimate_flops_per_token(
        _measured_N[d], d, N_LAYER, SEQUENCE_LENGTH, VOCAB_SIZE, include_output_layer=True
    )
    for d in D_MODEL_LEVELS
}
FLOPS_PER_TOKEN_FULL = {
    d: estimate_flops_per_token(
        _measured_N[d],
        d,
        N_LAYER,
        SEQUENCE_LENGTH,
        VOCAB_SIZE,
        include_output_layer=True,
        include_attention_seq_term=True,
    )
    for d in D_MODEL_LEVELS
}

for d in D_MODEL_LEVELS:
    assert FLOPS_PER_TOKEN_BODY_OUTPUT[d] >= FLOPS_PER_TOKEN_BODY[d], (
        f"d_model={d}: 本体+出力層の計算量が本体のみを下回っている(不変条件違反)"
    )
    assert FLOPS_PER_TOKEN_FULL[d] >= FLOPS_PER_TOKEN_BODY_OUTPUT[d]
print("OK: 全 d_model で 本体+出力層 >= 本体のみ の計算量(不変条件)")

_N_MID = _measured_N[D_MODEL_LEVELS[MID_D_MODEL_IDX]]
_C_MID = 6 * TARGET_TOKENS_PER_PARAM_MID * _N_MID**2
COMPUTE_BUDGETS = [_C_MID * COMPUTE_BUDGET_RATIO**k for k in range(-2, 3)]
print(f"C_mid={_C_MID:.3e}, COMPUTE_BUDGETS={[f'{c:.3e}' for c in COMPUTE_BUDGETS]}")

_step_table = {}
for d in D_MODEL_LEVELS:
    n = _measured_N[d]
    steps = [round(c / (6 * n) / TOKENS_PER_STEP) for c in COMPUTE_BUDGETS]
    _step_table[d] = steps
    print(f"d_model={d:>4} N={n:>9,} steps={steps}")

_total_production_steps = sum(sum(v) for v in _step_table.values())
print(
    f"\n本番グリッド({len(D_MODEL_LEVELS) * len(COMPUTE_BUDGETS)} セル)合計ステップ数: "
    f"{_total_production_steps:,}"
)

```

    OK: 全 d_model で 本体+出力層 >= 本体のみ の計算量(不変条件)
    C_mid=1.650e+13, COMPUTE_BUDGETS=['4.126e+12', '8.251e+12', '1.650e+13', '3.300e+13', '6.601e+13']
    d_model=  32 N=   49,312 steps=[1702, 3404, 6808, 13617, 27234]
    d_model=  64 N=  197,440 steps=[425, 850, 1700, 3401, 6802]
    d_model=  96 N=  443,232 steps=[189, 379, 757, 1515, 3030]
    d_model= 128 N=  787,072 steps=[107, 213, 427, 853, 1706]
    
    本番グリッド(20 セル)合計ステップ数: 75,119


### 5.9 較正: 学習率・gradient clipping 閾値

較正のスケールが本番より小さいと、本番でのみ現れる前提の不成立(学習の崩壊、閾値の未達など)を検出できない。そのため較正は中央の`d_model`・最大の計算量予算(そのセルの本番ステップ数)で行う。009 は Porian らの要因 3(スケール依存の最適化ハイパーパラメータ調整)を検証対象から明示的に除外し、**全`d_model`・全計算量予算で同一の学習率を使う**(6.2 節、固定条件として事前宣言済み)ため、較正するのは (1) 発散しない学習率、(2) その学習率での gradient clipping 閾値、の 2 つのみである。

**学習率の選定方針**: 固定 4 水準(x0.5, x1.0, x2.0, x4.0)による較正では、本番実行(`SMOKE_TEST=False`)で 4 水準すべてが単調に改善し続け、発散点にも改善の頭打ちにも到達しないまま探索が終わっていた(採用した学習率が真に「これ以上上げると悪化する」水準なのか確認できていなかった)。この問題に対処するため、倍率を 1 倍から倍々に増やしながら、**発散するか、直前の水準からの損失改善が`PLATEAU_RATIO`未満(頭打ち)になるまで** 探索を続ける適応的な方式に変更する。

- 発散した場合は、その 1 つ手前の(発散しなかった最後の)倍率を採用する(固定 4 水準時代と同じ設計方針を踏襲)。
- 発散せずに頭打ちになった場合は、改善が頭打ちになった時点の倍率(その時点でなお僅かに改善している最新の倍率)を採用する。
- `MAX_LR_MULTIPLIER`(32.0)に達しても発散も頭打ちも起きない場合は、その旨を出力し、`MAX_LR_MULTIPLIER`の水準を採用したうえで、**この探索範囲では真の最適値を捉えられていない可能性がある** ことを明記する。

**gradient clipping 閾値**: 採用した学習率での較正実行における勾配ノルムの`CLIP_QUANTILE`分位点(007・008 と同じ方式)。



```python
SMOKE_STEP_SCALE = 0.02 if SMOKE_TEST else 1.0  # 本番ステップ数に対する縮小比率(等比構造は保つ)


def scaled_steps(production_steps: int) -> int:
    return max(3, round(production_steps * SMOKE_STEP_SCALE))


CALIBRATION_STEPS = scaled_steps(_step_table[D_MODEL_LEVELS[MID_D_MODEL_IDX]][-1])  # 中央 d_model・最大計算量予算のステップ数(SMOKE_TEST では縮小)
BASE_LEARNING_RATE = 3e-4  # 006・007・008 と同じ出発点
START_LR_MULTIPLIER = 1.0  # 探索の起点(倍率)
MAX_LR_MULTIPLIER = 32.0  # この倍率に達しても発散・頭打ちが起きない場合は打ち切る
PLATEAU_RATIO = 0.02  # 直前の水準からの損失改善がこの割合未満なら「頭打ち」とみなす

_calib_warmup_steps = max(1, round(WARMUP_RATIO * CALIBRATION_STEPS))
print(f"CALIBRATION_STEPS={CALIBRATION_STEPS}, warmup_steps={_calib_warmup_steps}")


def run_calibration(
    d_model: int, num_steps: int, learning_rate: float, seed: int = 0, gradient_clip_threshold=None
):
    model = build_gpt_model(d_model).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    warmup_steps = max(1, round(WARMUP_RATIO * num_steps))
    schedule = functools.partial(
        compute_warmup_cosine_learning_rate,
        warmup_steps=warmup_steps,
        total_steps=num_steps,
        peak_learning_rate=learning_rate,
        min_learning_rate=learning_rate * MIN_LEARNING_RATE_RATIO,
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
        learning_rate=learning_rate,
        eval_interval=max(1, num_steps),
        device=device,
        seed=seed,
        optimizer=optimizer,
        learning_rate_schedule=schedule,
        gradient_clip_threshold=gradient_clip_threshold,
    )
    diverged = any(not np.isfinite(v) for v in history["train_loss"])
    return history, diverged


def calibrate_learning_rate(
    base_lr: float,
    d_model: int,
    num_steps: int,
    max_multiplier: float = MAX_LR_MULTIPLIER,
    plateau_ratio: float = PLATEAU_RATIO,
    start_multiplier: float = START_LR_MULTIPLIER,
):
    '''発散するか、直前の水準からの損失改善が plateau_ratio 未満になるまで倍率を倍増させながら
    学習率を探索する。

    発散した場合は、その 1 つ手前の(発散しなかった最後の)倍率を採用する。発散せずに
    改善が頭打ちになった場合は、頭打ちが検出された時点の倍率(その時点でなお僅かに
    改善している最新の倍率)を採用する。``max_multiplier`` に達しても発散も頭打ちも
    起きない場合、``max_multiplier`` の倍率を採用したうえで ``status="reached_max"``
    を返す(呼び出し側はこの場合、探索範囲では真の最適値を捉えられていない可能性を
    明記すること)。

    Returns:
        (採用した倍率, 倍率ごとの較正結果の辞書, 終了理由
        ``"diverged"``・``"plateaued"``・``"reached_max"`` のいずれか) のタプル。
    '''
    results: dict[float, dict] = {}
    mult = start_multiplier
    prev_mult, prev_loss = None, None
    while True:
        lr = base_lr * mult
        history, diverged = run_calibration(d_model, num_steps, lr)
        final_loss = history["train_loss"][-1] if not diverged else float("nan")
        results[mult] = {"lr": lr, "diverged": diverged, "final_loss": final_loss}
        loss_str = f"{final_loss:.4f}" if not diverged else "nan"
        print(f"lr x{mult}: diverged={diverged}, final_train_loss={loss_str}")

        if diverged:
            assert prev_mult is not None, (
                f"開始倍率(x{start_multiplier})から既に発散した。start_multiplier を"
                "下げて再較正が必要"
            )
            return prev_mult, results, "diverged"

        if prev_loss is not None:
            improvement = (prev_loss - final_loss) / prev_loss
            if improvement < plateau_ratio:
                return mult, results, "plateaued"

        if mult >= max_multiplier:
            return mult, results, "reached_max"

        prev_mult, prev_loss = mult, final_loss
        mult *= 2.0


_chosen_mult, _calibration_results, _calibration_status = calibrate_learning_rate(
    BASE_LEARNING_RATE, D_MODEL_LEVELS[MID_D_MODEL_IDX], CALIBRATION_STEPS
)
LEARNING_RATE = BASE_LEARNING_RATE * _chosen_mult
print(f"\n採用した学習率: x{_chosen_mult} = {LEARNING_RATE:.2e}(終了理由: {_calibration_status})")
if _calibration_status == "reached_max":
    print(
        f"警告: MAX_LR_MULTIPLIER(={MAX_LR_MULTIPLIER})に達しても発散も頭打ちも起きなかった。"
        "この探索範囲では真の最適な学習率を捉えられていない可能性がある。"
    )

# 最小・最大の d_model でも同じ学習率で発散しないことを確認する
_divergence_check = {}
for d in (D_MODEL_LEVELS[0], D_MODEL_LEVELS[-1]):
    _, diverged = run_calibration(d, CALIBRATION_STEPS, LEARNING_RATE)
    _divergence_check[d] = diverged
    print(f"d_model={d} での発散チェック: diverged={diverged}")
    assert not diverged, f"d_model={d} で採用した学習率が発散した(較正のやり直しが必要)"

# gradient clipping 閾値: 採用した学習率での較正実行における勾配ノルムの分位点
_clip_calib_history, _ = run_calibration(D_MODEL_LEVELS[MID_D_MODEL_IDX], CALIBRATION_STEPS, LEARNING_RATE)
GRADIENT_CLIP_THRESHOLD = float(
    np.quantile(_clip_calib_history["gradient_norm"], CLIP_QUANTILE)
)
print(f"gradient clipping 閾値(勾配ノルムの{CLIP_QUANTILE:.0%}分位点): {GRADIENT_CLIP_THRESHOLD:.4f}")

```

    CALIBRATION_STEPS=3030, warmup_steps=303
    lr x1.0: diverged=False, final_train_loss=5.6721
    lr x2.0: diverged=False, final_train_loss=5.0440
    lr x4.0: diverged=False, final_train_loss=4.6653
    lr x8.0: diverged=False, final_train_loss=4.4263
    lr x16.0: diverged=False, final_train_loss=4.2350
    lr x32.0: diverged=False, final_train_loss=4.1969
    
    採用した学習率: x32.0 = 9.60e-03(終了理由: plateaued)
    d_model=32 での発散チェック: diverged=False
    d_model=128 での発散チェック: diverged=False
    gradient clipping 閾値(勾配ノルムの90%分位点): 0.3054


### 5.10 1 ステップあたりの学習時間のスケーリング計測・外挿

まず代表として中央の`d_model`について、ステップ数を 3 点以上振って学習時間を実測し、$\log t = \log a + b \log(\text{steps})$ のあてはめでべき指数 $b$ を推定する(線形なら $b \approx 1$。JIT ウォームアップなどにより短いステップ数で相対的に遅くなる場合、$b$ が 1 から乖離する)。そのうえで、全`D_MODEL_LEVELS`(5.9 節の較正と同じ学習率・gradient clipping 閾値)について実際に一定ステップ数を実行し、`d_model`ごとの 1 ステップあたり学習時間を実測する(5 点しかないためこちらは外挿ではなく直接測定だが、`d_model`に対する依存の傾向をべき乗則としても報告する、診断用)。



```python
def measure_training_time(d_model: int, num_steps: int, seed: int = 0):
    model = build_gpt_model(d_model).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    warmup_steps = max(1, round(WARMUP_RATIO * num_steps))
    schedule = functools.partial(
        compute_warmup_cosine_learning_rate,
        warmup_steps=warmup_steps,
        total_steps=num_steps,
        peak_learning_rate=LEARNING_RATE,
        min_learning_rate=LEARNING_RATE * MIN_LEARNING_RATE_RATIO,
    )
    t0 = time.time()
    train_language_model(
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
        seed=seed,
        optimizer=optimizer,
        learning_rate_schedule=schedule,
        gradient_clip_threshold=GRADIENT_CLIP_THRESHOLD,
    )
    return time.time() - t0


_step_probe_counts = [5, 10, 20]
_step_probe_times = [measure_training_time(D_MODEL_LEVELS[MID_D_MODEL_IDX], k) for k in _step_probe_counts]
for k, t in zip(_step_probe_counts, _step_probe_times, strict=True):
    print(f"d_model={D_MODEL_LEVELS[MID_D_MODEL_IDX]}, steps={k}: {t:.3f} s")

_step_scaling_fit = fit_power_law(_step_probe_counts, _step_probe_times)
print(f"\nべき指数 b={_step_scaling_fit.exponent:.3f}(線形なら b~=1)")

# d_model ごとの 1 ステップあたり学習時間を直接測定する(5 点、production の値そのもの)
_PROBE_STEPS = 20
TIME_PER_STEP_BY_D_MODEL = {}
for d in D_MODEL_LEVELS:
    t_total = measure_training_time(d, _PROBE_STEPS)
    TIME_PER_STEP_BY_D_MODEL[d] = t_total / _PROBE_STEPS
    print(f"d_model={d}: {TIME_PER_STEP_BY_D_MODEL[d]:.4f} s/step")

_d_model_time_fit = fit_power_law(D_MODEL_LEVELS, list(TIME_PER_STEP_BY_D_MODEL.values()))
print(
    f"\n診断: 1 ステップあたり時間の d_model 依存のべき指数 = {_d_model_time_fit.exponent:.3f}"
    "(理論的には計算量が d_model の 2 乗に比例するため b~=2 に近いことが期待される)"
)

```

    d_model=96, steps=5: 25.316 s
    d_model=96, steps=10: 25.652 s
    d_model=96, steps=20: 26.173 s
    
    べき指数 b=0.024(線形なら b~=1)
    d_model=32: 1.0139 s/step
    d_model=64: 1.0923 s/step
    d_model=96: 1.3107 s/step
    d_model=128: 1.5851 s/step
    
    診断: 1 ステップあたり時間の d_model 依存のべき指数 = 0.309(理論的には計算量が d_model の 2 乗に比例するため b~=2 に近いことが期待される)


### 5.11 学習グリッド全体の合計時間の見積もり

4 つの`d_model` x 5 つの計算量予算(20 セル)+ノイズ床の追加 4 シード(中央`d_model`・中央計算量予算を計 5 シードにするための追加分)の合計時間を、5.10 節で実測した`d_model`ごとの 1 ステップあたり時間と、5.8 節で決めた本番ステップ数から計算し、Colab のセッション時間予算(`SESSION_BUDGET_SECONDS`)と比較する。



```python
_grid_total_seconds = 0.0
for d in D_MODEL_LEVELS:
    for steps in _step_table[d]:
        _grid_total_seconds += TIME_PER_STEP_BY_D_MODEL[d] * steps

_noise_floor_extra_seeds = NOISE_FLOOR_SEEDS - 1  # 中央セルは学習グリッドの中に既に 1 シード分含まれる
_noise_floor_mid_steps = _step_table[D_MODEL_LEVELS[MID_D_MODEL_IDX]][2]  # 中央 d_model・中央計算量予算
_grid_total_seconds += (
    TIME_PER_STEP_BY_D_MODEL[D_MODEL_LEVELS[MID_D_MODEL_IDX]] * _noise_floor_mid_steps * _noise_floor_extra_seeds
)

# 較正(5.9 節)で消費した時間も加算する。ここでの目的は「本番でどれだけ時間がかかるか」の
# 見積もりであるため、実際の較正実行に使う CALIBRATION_STEPS(SMOKE_TEST では縮小される)
# ではなく、常に本番スケールのステップ数(_step_table、5.8 節、SMOKE_TEST の影響を受けない)
# を使う。較正は中央 d_model で行うが、最小・最大 d_model での発散チェックはそれぞれの
# d_model の実測レートを使う。学習率探索の試行回数(_calibration_results の水準数)は
# 適応的探索(5.9 節、発散または頭打ちで終了)のため実行前には確定しないが、5.9 節が
# このセルより先に実行されるため、実際に試行した回数をそのまま使える(推測ではなく実測)。
_production_calibration_steps = _step_table[D_MODEL_LEVELS[MID_D_MODEL_IDX]][-1]  # 中央 d_model・最大計算量予算
_num_lr_trials = len(_calibration_results)  # 5.9 節の適応的探索が実際に試行した倍率の数
_calibration_seconds = (
    _num_lr_trials
    * TIME_PER_STEP_BY_D_MODEL[D_MODEL_LEVELS[MID_D_MODEL_IDX]]
    * _production_calibration_steps  # (1) 学習率の適応的探索(実際の試行回数)
    + TIME_PER_STEP_BY_D_MODEL[D_MODEL_LEVELS[0]] * _production_calibration_steps  # (2) 最小 d_model の発散確認
    + TIME_PER_STEP_BY_D_MODEL[D_MODEL_LEVELS[-1]] * _production_calibration_steps  # (2) 最大 d_model の発散確認
    + TIME_PER_STEP_BY_D_MODEL[D_MODEL_LEVELS[MID_D_MODEL_IDX]] * _production_calibration_steps  # (3) clip 閾値較正の再実行
)

_total_seconds = _grid_total_seconds + _calibration_seconds
print(
    f"学習グリッド({len(D_MODEL_LEVELS) * len(COMPUTE_BUDGETS)} セル): "
    f"{_grid_total_seconds:.1f} s ({_grid_total_seconds / 60:.1f} 分)"
)
print(f"較正: {_calibration_seconds:.1f} s ({_calibration_seconds / 60:.1f} 分)")
print(f"合計: {_total_seconds:.1f} s ({_total_seconds / 60:.1f} 分)")
print(f"セッション予算: {SESSION_BUDGET_SECONDS} s ({SESSION_BUDGET_SECONDS / 60:.0f} 分)")

_within_budget = _total_seconds <= SESSION_BUDGET_SECONDS
if not _within_budget:
    warnings.warn(
        f"学習グリッド+較正の合計見積もり時間({_total_seconds / 60:.1f} 分)が "
        f"セッション予算({SESSION_BUDGET_SECONDS / 60:.0f} 分)を超えている。"
        "TARGET_TOKENS_PER_PARAM_MID を下げるなどの調整が必要。",
        stacklevel=2,
    )
print(f"予算内: {_within_budget}")

```

    学習グリッド(20 セル): 84797.4 s (1413.3 分)
    較正: 35674.7 s (594.6 分)
    合計: 120472.1 s (2007.9 分)
    セッション予算: 7200 s (120 分)
    予算内: False


    /usr/local/lib/python3.13/dist-packages/IPython/core/interactiveshell.py:3553: UserWarning: 学習グリッド+較正の合計見積もり時間(2007.9 分)が セッション予算(120 分)を超えている。TARGET_TOKENS_PER_PARAM_MID を下げるなどの調整が必要。
      exec(code_obj, self.user_global_ns, self.user_ns)




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/02_pretraining/009_scaling_laws.ipynb
