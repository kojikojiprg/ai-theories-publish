---
title: "LoRA(Low-Rank Adaptation)(実装・実験編 2/3)"
---

この記事は後編(実装・実験編 2/3)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/012_low_rank_adaptation-practice-1)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/012_low_rank_adaptation-practice-3)。

### 6.1 実験宣言セル: 共通の設定・検証すること・判定基準・前提条件

**この節の内容は本番実行の前に確定させ、結果を見た後に変更しない。**

#### 共通の設定

- **起点**: ベースモデルは`kojikojiprg/ai-theories-small-gpt-en`の`main`ブランチ(008 で事前学習)、
  トークナイザは`kojikojiprg/ai-theories-tokenizer-en`。
- **データ**: Tiny Shakespeare を文字列の段階で連続区間に分割する(訓練 90%・較正用検証 5%・
  評価 5%)。学習率の較正は較正用検証集合で行い、判定はすべて評価集合で行う。
- **指標**: 評価集合の bits-per-byte(`evaluate_bits_per_byte()`、非重複窓)。全条件で
  トークナイザが同一なので、1 トークンあたりの指標で比べても条件間の比較は成り立つが、
  リポジトリ内の他トピックとの一貫性のために bits-per-byte を使う。
- **学習**: 007・008 と同じ部品を使う。AdamW(スクラッチ実装)、warmup(ステップ数の 10%)+
  cosine(最小学習率はピークの 1%)、gradient clipping、fp32。バッチサイズ 32、系列長は
  モデルの文脈長(256)。
- **重み減衰は全条件で 0 とする。** 同じ係数でも、全パラメータ微調整では重みそのものを
  原点へ引き寄せる(事前学習で獲得した重みを壊す方向に働く)のに対し、LoRA では $A$・$B$ を 0 へ、
  つまり更新量 $\Delta W$ を 0 へ(ベースモデルの方向へ)引き寄せる。疎微調整でも更新量 $\delta$ を
  0 へ引き寄せる。方式によって正則化の意味が異なるため、共通の係数を置くと方式間の比較に
  別の違いが混ざる。
- **gradient clipping の閾値**: 008 と同じ決め方をとる。すなわち、方式ごとに、較正(クリッピング
  なし)で採用した学習率の実行における勾配ノルムの 90% 分位点を閾値とする。勾配ノルムは
  学習可能なパラメータのみから計算されるため、方式ごとに尺度が大きく異なる(008 の事前学習での
  閾値の値そのものは、どの方式の勾配ノルムの尺度とも対応しない)。実験 C のすべての rank には、
  学習率と同じく LoRA($r=8$)の閾値を使う。
- **ステップ数**: 全条件で共通。総学習トークン数 = 訓練トークン数 × $E$ で決め、$E = 2$ とする
  ($T = \lfloor E N_{\mathrm{train}} / (B S) \rfloor$、$N_{\mathrm{train}}$ は訓練トークン数、
  $B$ はバッチサイズ、$S$ は系列長)。$T$ は、結果を見る前に宣言した $E = 2$ から計算式で決まる値
  である(本番では $T = 87$)。**したがって実験 A〜C の判定は、$T$ ステップという固定の学習予算の
  もとでの比較であり、各方式が収束した状態での性能の比較ではない。** 方式による学習の速さの違い
  (Biderman et al. 2024 [6] の指摘を含む)は判定に含まれる。これは判定基準の変更ではなく、判定が
  何についての主張かを明確にするための記述である。収束の度合いを見るため、学習曲線の最終区間の傾き
  (実験 A の「介入の直接作用点と対比量の距離」の段落を参照)を診断量として実験 A〜C で併記する。
- **判定の時点**: 最終ステップの評価値で判定する。評価集合での学習曲線を記録し、最良値と
  最終値の差(過学習の程度)を診断量として併記する。
- **学習率の較正**: 方式ごと(全パラメータ微調整・LoRA $r=8$・ランダムマスク疎微調整)に、
  公比 $\sqrt{10}$ の等比 5 水準のグリッドで、**本番と同じステップ数** で行う(シード 0、
  gradient clipping なし)。較正用検証集合の最終 bits-per-byte が最小の水準を採用する。
  **採用条件**: 選ばれた水準がグリッドの端でないこと(内点であること)。端だった場合は、その側に
  同じ公比で 1 水準ずつグリッドを拡張して較正をやり直す(最大 4 回)。実験 C のすべての rank には
  LoRA $r=8$ で較正した学習率を使う(原論文の設計どおり、3.5 節)。
- **シード数**: 判定に使う条件は各 5 シード、中間の rank($r \in \{2, 4, 16, 32\}$)は曲線の描画用で
  判定に使わないため 2 シード。シードは、全パラメータ微調整ではミニバッチの順序、LoRA では
  それに加えて $A$ の初期化、疎微調整ではそれに加えて更新位置の選択を決める。
- **LoRA のハイパーパラメータ**: $\alpha = 8$ で全 rank 共通。$r = 8$ のとき $\alpha / r = 1$ になり、
  原論文の「最初に試した $r$ に $\alpha$ を合わせる」設計に対応する。
- **訓練損失の定義(前提条件 B1・C1、実験 B の診断量)**: 訓練集合の非重複窓から等間隔に選んだ
  64 個の固定の窓での平均損失(nats / トークン)とする。学習開始時の値はベースモデルで、最終値は
  学習後のモデルで、同じ窓・同じ関数で測る。学習ループが記録するミニバッチごとの損失は、
  ステップごとに異なるデータに対する値でありミニバッチ間のばらつきを含むため、前後の比較には
  使わない(参考として記録はする)。

#### 記号と条件の対応

| 記号 | 条件 | シード数 |
|---|---|---|
| $b_0$ | ベースモデル(学習なし) | 1(決定的) |
| $b_1$ | 全パラメータ微調整 | 5 |
| $b_2$ | LoRA($r = 8$、Query・Value) | 5 |
| $b_3$ | ランダムマスク疎微調整(Query・Value、行列ごとに $k = 8(d_{\mathrm{in}} + d_{\mathrm{out}})$ 要素) | 5 |
| $b_r$ | rank $r$ の LoRA($r \in \{1, 2, 4, 8, 16, 32, 64\}$、$b_{r=8}$ は $b_2$ と同一の学習結果) | $r \in \{1, 8, 64\}$ は 5、それ以外は 2 |

$b_i$ は評価集合の最終 bits-per-byte、$\bar{b}_i$ はそのシード平均、$s_i$ はシード間の標本
標準偏差(不偏)、$n_i$ はシード数である。シード平均の標準誤差を
$\sigma_{\bar{b}_i} = s_i / \sqrt{n_i}$ とし、シード間は独立とする。$b_0$ は学習を伴わない決定的な
値なので分散 0 として扱う。

#### 実験 A: 全パラメータ微調整との比較

**検証すること**: Query・Value に LoRA($r=8$)を適用すると、全パラメータ微調整による
bits-per-byte の改善幅の $\delta$ 以上を回復する。

**対比量**: 改善幅の回復率

$$
\rho = \frac{b_0 - \bar{b}_2}{b_0 - \bar{b}_1}
$$

**標準偏差の導出**: $u = b_0 - \bar{b}_2$、$v = b_0 - \bar{b}_1$ とおく。$b_0$ は分散 0 なので
$\sigma_u = \sigma_{\bar{b}_2}$、$\sigma_v = \sigma_{\bar{b}_1}$ であり、$u$ と $v$ は独立な
シード集合から作られるので共分散は 0 である。$\rho = u / v$ にデルタ法(1 次のテイラー展開による
誤差伝播)を適用すると

$$
\sigma_\rho^2 \approx \left(\frac{\partial \rho}{\partial u}\right)^2 \sigma_u^2
+ \left(\frac{\partial \rho}{\partial v}\right)^2 \sigma_v^2
= \frac{\sigma_{\bar{b}_2}^2}{v^2} + \frac{u^2 \sigma_{\bar{b}_1}^2}{v^4}
$$

となる。診断量として、各条件のシードを独立に復元抽出するブートストラップ
(10,000 回)による $\rho$ の 95% 区間も併記する(判定には使わない)。

**判定**: $\delta = 0.9$ として、

- 支持: $\rho - 2\sigma_\rho \ge \delta$
- 反証: $\rho + 2\sigma_\rho < \delta$
- 判定不能: 上記以外

$\delta = 0.9$ は「全パラメータ微調整の改善幅の 9 割を回復すれば匹敵とみなす」という、結果を
見る前に置いた実用上の基準である。統計的に導いた値ではない。

**前提条件 A1**: 分母が確定していること。$b_0 - \bar{b}_1 \ge 10\,\sigma_{\bar{b}_1}$。
LoRA の結果に依存しない量で定義している。分母 $v$ が標準誤差に対して小さいと、$\rho$ が不安定に
なり、デルタ法の近似も成り立たなくなるため。

**介入の直接作用点と対比量の距離**: LoRA が直接制約するのは更新量 $\Delta W$ の rank であり、
bits-per-byte は、その更新量が順伝播と学習の全体を経由して現れる下流の量である。それでも
bits-per-byte を対比量とするのは、検証したい主張そのもの(全パラメータ微調整に匹敵する **適応性能**
が得られるか)が bits-per-byte で定義される量だからである。直接の作用点に近い診断量として、
Query・Value の各行列の $\lVert \Delta W \rVert_F / \lVert W_0 \rVert_F$ を両方式で併記する。
さらに、全パラメータ微調整の $\Delta W$ について、上位 8 個の特異値が占めるエネルギー
(特異値の二乗和)の割合を併記する(全パラメータ微調整の更新量が実際に低ランクに近いかの診断)。
また、判定が固定の学習予算($T$ ステップ)のもとでの比較であることを踏まえ、**学習曲線の最終区間の
傾き** を方式ごと(実験 C では rank ごと)に併記する。評価集合の学習曲線の最後の 2 つの記録点
$T' < T$ について

$$
g = \frac{b(T) - b(T')}{T - T'}
$$

をシードごとに求め、シード平均を印字する。$b(t)$ はステップ $t$ での評価集合の bits-per-byte、$T'$ は
最終ステップの 1 つ前の記録点である。$g$ が負で絶対値が大きいほど、最終ステップの時点でまだ学習が
進んでいた(収束していない)ことを示す。

**本文に明記すること**: 3.8 節のとおり、小規模かつドメイン差が大きい設定では LoRA が全パラメータ
微調整に届かない可能性がある(Biderman et al. 2024 [6])。反証・判定不能はいずれも起こりうる結果である。

#### 実験 B: 同じパラメータ数の対照との比較

**検証すること**: 学習可能パラメータ数を揃えたとき、低ランク構造の更新(LoRA $r=8$)のほうが、
Query・Value の重みからランダムに選んだ要素だけを更新する疎な微調整よりも、評価集合の
bits-per-byte が低い。

**対照**: 各対象行列について、LoRA と同数の $k = r(d_{\mathrm{in}} + d_{\mathrm{out}})$($r = 8$)
要素をシードごとに固定の乱数で選び、更新量を 0 で初期化する(5.6 節)。学習率は疎微調整用に
別途較正する。

**対比量**: $d = \bar{b}_3 - \bar{b}_2$。**標準偏差**: 独立な 2 つのシード平均の差なので
$\sigma_d = \sqrt{\sigma_{\bar{b}_3}^2 + \sigma_{\bar{b}_2}^2}$。

**判定**: 支持は $d > 2\sigma_d$、反証は $d < -2\sigma_d$、それ以外は判定不能。

**前提条件 B1**: 両方式(LoRA $r=8$・疎微調整)の全シードで、最終訓練損失が学習開始時の訓練損失
から 5% 以上低下していること(最終訓練損失 $\le 0.95 \times$ 学習開始時の訓練損失)。最適化が
実際に進んでいることの確認であり、どちらの方式が優れているかという仮説とは独立に定義している。

**介入の直接作用点と対比量の距離**: 介入は更新量のパラメータ化(帰納バイアス)であり、
bits-per-byte は下流の量である。差が生じた場合に、それが汎化の違いによるものか最適化の
しやすさの違いによるものかを切り分けるため、最終訓練損失の差(疎微調整 − LoRA のシード平均)を
診断量として併記する。

#### 実験 C: rank の飽和

**検証すること**: $r \in \{1, 2, 4, 8, 16, 32, 64\}$(公比 2 の等比)で、rank を 1 → 8 に上げた
ときの改善幅が、8 → 64 に上げたときの改善幅より大きい。両区間とも rank を 3 回 2 倍にする区間で
あり、刻み(対数尺度での幅)は揃っている。

**対比量**:

$$
c = \Delta_{\mathrm{low}} - \Delta_{\mathrm{high}}
= (\bar{b}_{r=1} - \bar{b}_{r=8}) - (\bar{b}_{r=8} - \bar{b}_{r=64})
= \bar{b}_{r=1} - 2\bar{b}_{r=8} + \bar{b}_{r=64}
$$

**標準偏差**: 3 つのシード平均は独立なシード集合から作られるので、係数 $(1, -2, 1)$ の
線形結合の分散として

$$
\sigma_c = \sqrt{\sigma_{\bar{b}_{r=1}}^2 + 4\sigma_{\bar{b}_{r=8}}^2 + \sigma_{\bar{b}_{r=64}}^2}
$$

**判定**: 支持は $c > 2\sigma_c$、反証は $c < -2\sigma_c$、それ以外は判定不能。

**前提条件 C1**: 全 rank(中間の rank を含む)の全シードで、最終訓練損失が学習開始時の訓練損失から
5% 以上低下していること。

**交絡の明記**: $\alpha / r$ スケーリングでは $r$ が大きいほど実効的な更新量が縮む
(rsLoRA の指摘、3.5 節)。そのため、飽和が観測されても、それが低ランク性によるもの
(小さい rank で必要な更新を表現しきれている)か、更新量の縮小によるもの(大きい rank が
十分に学習できていない)かを、この実験だけでは区別できない。**支持の判定は「低ランクで十分である」
ことの検証とはみなさない。** 診断量として、rank ごとの
$\lVert \frac{\alpha}{r} B A \rVert_F / \lVert W_0 \rVert_F$(Query・Value の全行列・全シードの平均)を
併記する。

**介入の直接作用点と対比量の距離**: rank が直接作用するのは更新量 $\Delta W$ の表現力(rank の上限)
であり、bits-per-byte は下流の量である。上記の更新量のノルムの比が、直接の作用点に近い診断量である。

#### 効率の測定(判定基準を設けない測定)

- 閉形式の表: 全パラメータ微調整と LoRA $r=8$ の学習可能パラメータ数・勾配のバイト数・
  AdamW 状態のバイト数(3.1 節)。閉形式の値と、実際のパラメータ・勾配・optimizer の状態から
  数えた値の一致をアサーションで確認する。
- 実測の表: CUDA 環境でのみ、`torch.cuda.max_memory_allocated()`による 1 ステップのピークメモリと、
  1 ステップあたりの時間を測る。CUDA でない場合はスキップした旨を印字する。
- 判定基準は設けない。この規模では活性化のメモリが支配的であり(3.1 節・011 の実験 G)、
  学習可能パラメータ数の削減がピークメモリの総量に対しては小さく見えうる。

#### 出力例(定性的な観察)

ベースモデル・全パラメータ微調整・LoRA $r=8$(いずれもシード 0 の学習結果)について、同じ
プロンプトから top-p サンプリング($p = 0.9$、シード固定)で生成した例を並べる。判定基準は設けない。

### 6.2 スケーリングの計測と外挿

本番のデータ量がスモークテストの何倍にもなる重い処理(学習(方式ごと)・評価・符号化)について、
3 点のデータ量で実行時間を実測し、$\log t = \log a + b \log n$ をあてはめてべき指数 $b$ を推定し、
本番のデータ量へ外挿する。学習はステップ数、評価は評価する窓の数、符号化は文字数をデータ量 $n$
とする。外挿値に本番での実行回数を乗じ、Colab 無料枠のセッション時間(2 時間を目安)に対する
余裕を確認する。

学習の計測点は本番のステップ数の 1/4・1/2・1 倍とする(3 点目は本番のステップ数そのものの直接計測に
なる)。ステップ数によらない固定費(モデルの複製・凍結パラメータの確認・訓練損失の測定など)があるため、
小さいステップ数だけで $b$ を推定すると $b < 1$ となり、本番の時間を過小に見積もるからである。
それでも $b < 1$ になる場合があるため、見積もりには外挿値と本番のステップ数での直接計測値の
大きいほうを使う。
学習の計測には、本番の学習と同じ学習曲線の評価(記録点ごとの評価集合の評価)が含まれる。

本番での学習の実行回数: 較正 3 方式 × 5 水準(拡張が起きた場合の予備として方式ごとに 2 水準を
上乗せする。較正は最終ステップのみ評価するので、学習曲線の評価を含む計測値で見積もるのは安全側になる)、
実験 A・B の 3 方式 × 5 シード = 15、実験 C の追加分 $r \in \{1, 64\}$ × 5 シード +
$r \in \{2, 4, 16, 32\}$ × 2 シード = 18。


```python
def time_training(method: str, num_steps: int) -> float:
    sync_device()
    t0 = time.time()
    run_condition(
        method,
        seed=0,
        learning_rate=1e-4,
        clip_threshold=None,
        num_steps=num_steps,
        measure_diagnostics=False,
    )
    sync_device()
    return time.time() - t0


def time_evaluation(num_windows: int) -> float:
    sync_device()
    t0 = time.time()
    evaluate_bits_per_byte(
        base_model,
        evaluation_windows[:num_windows],
        evaluation_mask[:num_windows],
        evaluation_bytes,
        device,
    )
    sync_device()
    return time.time() - t0


def time_encoding(num_chars: int) -> float:
    t0 = time.time()
    encode_corpus(tokenizer, raw_text[:num_chars])
    return time.time() - t0


def fit_and_extrapolate(label: str, sizes: list[int], times: list[float], target: int) -> float:
    fit = fit_power_law_exponent(sizes, times)
    extrapolated = fit.coefficient * target**fit.exponent
    detail = ", ".join(f"n={n}: {t:.2f}s" for n, t in zip(sizes, times, strict=True))
    print(
        f"[{label}] {detail} -> b={fit.exponent:.3f}(標準誤差 {fit.exponent_stderr:.3f}), "
        f"R^2={fit.r_squared:.4f}, n={target} での外挿値 {extrapolated:.2f}s"
    )
    return extrapolated


_t0_scaling = time.time()
time_training("lora", 2)  # 初回の呼び出しに伴うオーバーヘッド(カーネルの準備など)を計測から除く
_train_step_sizes = [PROD_NUM_STEPS // 4, PROD_NUM_STEPS // 2, PROD_NUM_STEPS]
_extrapolated_training = {}
for _method in ("full", "lora", "sparse"):
    _times = [time_training(_method, n) for n in _train_step_sizes]
    _fit_value = fit_and_extrapolate(
        f"学習 {_method}(学習曲線の評価を含む)", _train_step_sizes, _times, PROD_NUM_STEPS
    )
    # 固定費のため b < 1 になりうる。外挿値と本番ステップ数での直接計測値の大きいほうを使う(安全側)。
    _extrapolated_training[_method] = max(_fit_value, _times[-1])

_num_windows_all = evaluation_windows.size(0)
_eval_sizes = [max(1, _num_windows_all // 4), max(2, _num_windows_all // 2), _num_windows_all]
time_evaluation(_eval_sizes[0])  # 同上(初回のオーバーヘッドを除く)
_extrapolated_eval = fit_and_extrapolate(
    "評価(評価集合 1 回分、学習の計測に含まれる)",
    _eval_sizes,
    [time_evaluation(n) for n in _eval_sizes],
    _num_windows_all,
)

_encode_sizes = [len(raw_text) // 8, len(raw_text) // 4, len(raw_text) // 2]
_extrapolated_encoding = fit_and_extrapolate(
    "符号化", _encode_sizes, [time_encoding(n) for n in _encode_sizes], len(raw_text)
)
_t_scaling_measurement = time.time() - _t0_scaling

# --- 本番の実行回数を乗じた合計 ---
_prod_ranks = LEVELS["prod"]["RANKS"]
_prod_seeds_judged = LEVELS["prod"]["NUM_SEEDS_JUDGED"]
_prod_seeds_intermediate = LEVELS["prod"]["NUM_SEEDS_INTERMEDIATE"]
_prod_grid_levels = {
    m: len(g) + 2 for m, g in LEVELS["prod"]["LR_GRID_K"].items()
}  # +2 は拡張の予備
_n_rank_runs_judged = (
    sum(1 for r in _prod_ranks if r in JUDGED_RANKS and r != MAIN_RANK) * _prod_seeds_judged
)
_n_rank_runs_intermediate = (
    sum(1 for r in _prod_ranks if r not in JUDGED_RANKS) * _prod_seeds_intermediate
)

_estimate = {
    "較正(学習)": sum(_extrapolated_training[m] * n for m, n in _prod_grid_levels.items()),
    "実験 A・B(学習)": sum(_extrapolated_training[m] for m in ("full", "lora", "sparse"))
    * _prod_seeds_judged,
    "実験 C の追加分(学習)": _extrapolated_training["lora"]
    * (_n_rank_runs_judged + _n_rank_runs_intermediate),
    "符号化(1 回、キャッシュ後は再符号化しない)": _extrapolated_encoding,
    "スケーリング計測自体(このセルの実測)": _t_scaling_measurement,
}
_total_estimate = sum(_estimate.values())
print()
print(
    f"--- 本番実行の見積もり(PROD_NUM_STEPS={PROD_NUM_STEPS}、実行回数を乗じた値、{device} 基準)---"
)
print(
    f"実行回数: 較正 {_prod_grid_levels}(拡張の予備を含む)、実験 A・B {3 * _prod_seeds_judged}、"
    f"実験 C の追加分 {_n_rank_runs_judged + _n_rank_runs_intermediate}"
)
for _k, _v in _estimate.items():
    print(f"  {_k}: {_v:.1f}s")
print(
    f"合計: {_total_estimate:.1f}s({_total_estimate / 60:.1f} 分)/ セッション予算 "
    f"{SESSION_BUDGET_SECONDS / 60:.0f} 分 / 余裕 {SESSION_BUDGET_SECONDS / _total_estimate:.2f} 倍"
)
if _total_estimate > SESSION_BUDGET_SECONDS:
    print("警告: 見積もりがセッション予算を超える。シード数・rank の集合を見直す必要がある。")
print(f"注記: 上記は実行したデバイス({device})での計測に基づく値である。")
```

    [学習 full(学習曲線の評価を含む)] n=21: 4.42s, n=43: 7.15s, n=87: 13.41s -> b=0.781(標準誤差 0.064), R^2=0.9934, n=87 での外挿値 13.06s
    [学習 lora(学習曲線の評価を含む)] n=21: 4.13s, n=43: 6.43s, n=87: 11.84s -> b=0.741(標準誤差 0.071), R^2=0.9910, n=87 での外挿値 11.50s
    [学習 sparse(学習曲線の評価を含む)] n=21: 4.06s, n=43: 6.47s, n=87: 11.85s -> b=0.753(標準誤差 0.060), R^2=0.9936, n=87 での外挿値 11.56s
    [評価(評価集合 1 回分、学習の計測に含まれる)] n=20: 0.03s, n=41: 0.07s, n=82: 0.14s -> b=1.008(標準誤差 0.024), R^2=0.9995, n=82 での外挿値 0.14s
    [符号化] n=139424: 0.02s, n=278848: 0.06s, n=557697: 0.11s -> b=1.067(標準誤差 0.102), R^2=0.9910, n=1115394 での外挿値 0.23s
    
    --- 本番実行の見積もり(PROD_NUM_STEPS=87、実行回数を乗じた値、cuda 基準)---
    実行回数: 較正 {'full': 7, 'lora': 7, 'sparse': 7}(拡張の予備を含む)、実験 A・B 15、実験 C の追加分 18
      較正(学習): 259.7s
      実験 A・B(学習): 185.5s
      実験 C の追加分(学習): 213.1s
      符号化(1 回、キャッシュ後は再符号化しない): 0.2s
      スケーリング計測自体(このセルの実測): 70.8s
    合計: 729.3s(12.2 分)/ セッション予算 120 分 / 余裕 9.87 倍
    注記: 上記は実行したデバイス(cuda)での計測に基づく値である。


### 6.3 学習率の較正と gradient clipping の閾値

方式ごとに、`SMOKE_TEST`に従う水準のステップ数(本番では本番と同じステップ数)・シード 0・
gradient clipping なしで学習率のグリッドを掃引し、較正用検証集合の最終 bits-per-byte が最小の
水準を採用する。採用した水準がグリッドの端なら、同じ公比でその側に 1 水準拡張して掃引を続ける
(6.1 節)。gradient clipping の閾値は、採用した水準の実行の勾配ノルムの 90% 分位点とする。

**グリッドの中央の水準を決めた予備掃引の記録**:

- 本番と同じステップ数(87 ステップ)・シード 0 で、**較正用検証集合のみ** を使ってローカル(MPS)で
  行った。評価集合は使っていない。
- 目的は、各方式のグリッドの中央を端にならない位置に置くことのみである(予備掃引で較正用検証集合の
  bits-per-byte が最小だった水準: 全パラメータ微調整 $10^{-2.5}$、LoRA $10^{-1.5}$、
  疎微調整 $10^{-1}$)。本番の較正は、この節のコードで改めて行う。
- 判定基準(6.1 節)は予備掃引より前に確定しており、予備掃引の後に変更していない。
- 予備掃引の時点で、較正用検証集合での改善幅の回復率(実験 A の $\rho$ と同じ形の量を、較正用検証集合・
  シード 0 の値で計算したもの)がおよそ 0.6 であることを把握していた。これは透明性のための記録であり、
  判定には使わない。


```python
calibration_log: dict[str, dict] = {}
LEARNING_RATE: dict[str, float] = {}
CLIP_THRESHOLD: dict[str, float] = {}
_t0_calibration = time.time()

for _method in ("full", "lora", "sparse"):
    _results: dict[int, dict] = {}
    _grid = list(LR_GRID_K[_method])
    _expansions = 0
    while True:
        for _k in _grid:
            if _k in _results:
                continue
            _t0 = time.time()
            _rec = run_condition(
                _method,
                seed=0,
                learning_rate=learning_rate_from_k(_k),
                clip_threshold=None,
                split="calibration",
                measure_diagnostics=False,
                eval_interval=NUM_STEPS,
            )
            _bits_per_byte = _rec["final_bits_per_byte"]
            _results[_k] = {
                "learning_rate": learning_rate_from_k(_k),
                "final_calibration_bits_per_byte": _bits_per_byte
                if math.isfinite(_bits_per_byte)
                else None,
                "gradient_norm_quantile": float(np.quantile(_rec["gradient_norm"], CLIP_QUANTILE)),
                "elapsed_seconds": time.time() - _t0,
            }
            print(
                f"[{_method}] lr={learning_rate_from_k(_k):.2e}: 較正用検証 bits-per-byte = {_bits_per_byte:.4f}"
                f"({time.time() - _t0:.1f}s)"
            )
        _finite = {
            k: r for k, r in _results.items() if r["final_calibration_bits_per_byte"] is not None
        }
        assert _finite, f"{_method}: 全水準で発散した"
        _best = min(_finite, key=lambda k: _finite[k]["final_calibration_bits_per_byte"])
        _keys = sorted(_results)
        if _keys[0] < _best < _keys[-1]:
            break
        if _expansions >= MAX_GRID_EXPANSIONS:
            break
        _grid.append(_best - LR_GRID_K_STEP if _best == _keys[0] else _best + LR_GRID_K_STEP)
        _expansions += 1
        print(f"[{_method}] 端の水準が選ばれたため、グリッドを拡張する(拡張 {_expansions} 回目)")

    _interior = sorted(_results)[0] < _best < sorted(_results)[-1]
    precondition_status[f"calibration_interior_{_method}"] = _interior
    LEARNING_RATE[_method] = learning_rate_from_k(_best)
    CLIP_THRESHOLD[_method] = _results[_best]["gradient_norm_quantile"]
    calibration_log[_method] = {
        "grid": {f"{learning_rate_from_k(k):.3e}": _results[k] for k in sorted(_results)},
        "chosen_learning_rate": LEARNING_RATE[_method],
        "chosen_is_interior": _interior,
        "num_expansions": _expansions,
        "clip_threshold": CLIP_THRESHOLD[_method],
    }
    print(
        f"[{_method}] 採用: lr={LEARNING_RATE[_method]:.3e}(内点={_interior}、拡張 {_expansions} 回)、"
        f"gradient clipping 閾値={CLIP_THRESHOLD[_method]:.4f}"
    )

# 実験 C のすべての rank は LoRA(r=8)の較正結果を使う(6.1 節)
_t_calibration_total = time.time() - _t0_calibration
print(f"\n較正の合計時間: {_t_calibration_total:.1f}s")
print(
    "較正の採用条件(内点であること):",
    {m: precondition_status[f"calibration_interior_{m}"] for m in ("full", "lora", "sparse")},
)
if not SMOKE_TEST:
    assert all(
        precondition_status[f"calibration_interior_{m}"] for m in ("full", "lora", "sparse")
    ), (
        "拡張の上限まで拡張しても内点にならなかった方式がある。グリッドを見直して較正をやり直すこと。"
    )
```

    [full] lr=3.16e-04: 較正用検証 bits-per-byte = 2.3148(13.4s)
    [full] lr=1.00e-03: 較正用検証 bits-per-byte = 2.1481(14.4s)
    [full] lr=3.16e-03: 較正用検証 bits-per-byte = 2.0548(14.6s)
    [full] lr=1.00e-02: 較正用検証 bits-per-byte = 2.1382(15.3s)
    [full] lr=3.16e-02: 較正用検証 bits-per-byte = 2.9292(14.5s)
    [full] 採用: lr=3.162e-03(内点=True、拡張 0 回)、gradient clipping 閾値=0.9471
    [lora] lr=3.16e-03: 較正用検証 bits-per-byte = 2.7779(11.8s)
    [lora] lr=1.00e-02: 較正用検証 bits-per-byte = 2.6410(11.6s)
    [lora] lr=3.16e-02: 較正用検証 bits-per-byte = 2.5648(11.7s)
    [lora] lr=1.00e-01: 較正用検証 bits-per-byte = 3.3538(11.9s)
    [lora] lr=3.16e-01: 較正用検証 bits-per-byte = 3.6256(12.0s)
    [lora] 採用: lr=3.162e-02(内点=True、拡張 0 回)、gradient clipping 閾値=0.2861
    [sparse] lr=1.00e-02: 較正用検証 bits-per-byte = 2.6999(12.2s)
    [sparse] lr=3.16e-02: 較正用検証 bits-per-byte = 2.6183(12.1s)
    [sparse] lr=1.00e-01: 較正用検証 bits-per-byte = 2.6004(11.9s)
    [sparse] lr=3.16e-01: 較正用検証 bits-per-byte = 2.6967(11.7s)
    [sparse] lr=1.00e+00: 較正用検証 bits-per-byte = 3.2072(11.7s)
    [sparse] 採用: lr=1.000e-01(内点=True、拡張 0 回)、gradient clipping 閾値=0.1004
    
    較正の合計時間: 190.7s
    較正の採用条件(内点であること): {'full': True, 'lora': True, 'sparse': True}


### 6.4 本番の学習: ベースモデル・全パラメータ微調整・LoRA($r=8$)・疎微調整

$b_0$ はベースモデルの評価集合での bits-per-byte(学習を伴わない決定的な値)である。3 方式を
それぞれ判定用のシード数で学習する。出力例(6.11 節)とマージの確認(6.5 節)のため、
全パラメータ微調整と LoRA のシード 0 のモデルのみメモリ上に残す(保存はしない)。


```python
B0 = evaluate_bits_per_byte(
    base_model, evaluation_windows, evaluation_mask, evaluation_bytes, device
)
assert evaluate_bits_per_byte(
    base_model, evaluation_windows, evaluation_mask, evaluation_bytes, device
) == B0, "b0 が決定的でない"
print(f"b0(ベースモデル、評価集合)= {B0:.4f}")

main_records: dict[str, list[dict]] = {"full": [], "lora": [], "sparse": []}
kept_models: dict[str, dict] = {}
_t0_main = time.time()
for _method in ("full", "lora", "sparse"):
    for _seed in SEEDS_JUDGED:
        _t0 = time.time()
        _rec = run_condition(
            _method,
            _seed,
            LEARNING_RATE[_method],
            CLIP_THRESHOLD[_method],
            keep_model=(_seed == 0 and _method in ("full", "lora")),
        )
        if "_model" in _rec:
            kept_models[_method] = {"model": _rec.pop("_model"), "replaced": _rec.pop("_replaced")}
        main_records[_method].append(_rec)
        print(
            f"[{_method}] seed={_seed}: b={_rec['final_bits_per_byte']:.4f}, "
            f"訓練損失 {_rec['initial_train_loss']:.4f} -> {_rec['final_train_loss']:.4f}, "
            f"clipping 発動率={_rec['clip_trigger_ratio']:.2f}({time.time() - _t0:.1f}s)"
        )
print(f"本番の学習(3 方式)の合計時間: {time.time() - _t0_main:.1f}s")
```

    b0(ベースモデル、評価集合)= 3.3928
    [full] seed=0: b=2.2618, 訓練損失 6.2466 -> 3.3946, clipping 発動率=0.11(15.7s)
    [full] seed=1: b=2.2410, 訓練損失 6.2466 -> 3.3485, clipping 発動率=0.13(15.9s)
    [full] seed=2: b=2.2610, 訓練損失 6.2466 -> 3.3213, clipping 発動率=0.10(15.9s)
    [full] seed=3: b=2.2377, 訓練損失 6.2466 -> 3.3530, clipping 発動率=0.11(15.7s)
    [full] seed=4: b=2.2566, 訓練損失 6.2466 -> 3.3634, clipping 発動率=0.10(15.7s)
    [lora] seed=0: b=2.6077, 訓練損失 6.2466 -> 4.6590, clipping 発動率=0.10(13.4s)
    [lora] seed=1: b=2.6179, 訓練損失 6.2466 -> 4.6642, clipping 発動率=0.15(13.4s)
    [lora] seed=2: b=2.6007, 訓練損失 6.2466 -> 4.6437, clipping 発動率=0.10(13.5s)
    [lora] seed=3: b=2.6163, 訓練損失 6.2466 -> 4.6788, clipping 発動率=0.20(13.6s)
    [lora] seed=4: b=2.6138, 訓練損失 6.2466 -> 4.6821, clipping 発動率=0.20(13.7s)
    [sparse] seed=0: b=2.6396, 訓練損失 6.2466 -> 4.6676, clipping 発動率=0.11(13.3s)
    [sparse] seed=1: b=2.6362, 訓練損失 6.2466 -> 4.6555, clipping 発動率=0.13(13.1s)
    [sparse] seed=2: b=2.6353, 訓練損失 6.2466 -> 4.6505, clipping 発動率=0.11(13.1s)
    [sparse] seed=3: b=2.6271, 訓練損失 6.2466 -> 4.6521, clipping 発動率=0.09(13.2s)
    [sparse] seed=4: b=2.6574, 訓練損失 6.2466 -> 4.6559, clipping 発動率=0.11(13.3s)
    本番の学習(3 方式)の合計時間: 212.6s


### 6.5 実験 C の追加の学習(rank の掃引)

$r = 8$ は 6.4 節の LoRA の学習結果を共有する(同じ条件を再学習しない)。判定に使う $r = 1, 64$ は
判定用のシード数、中間の rank は描画用のシード数で学習する。学習率・gradient clipping の閾値は
すべて LoRA($r=8$)の較正結果を使う。


```python
rank_records: dict[int, list[dict]] = {MAIN_RANK: main_records["lora"]}
_t0_rank = time.time()
for _rank in RANKS:
    if _rank == MAIN_RANK:
        continue
    _seeds = SEEDS_JUDGED if _rank in JUDGED_RANKS else SEEDS_INTERMEDIATE
    rank_records[_rank] = []
    for _seed in _seeds:
        _rec = run_condition(
            "lora", _seed, LEARNING_RATE["lora"], CLIP_THRESHOLD["lora"], rank=_rank
        )
        rank_records[_rank].append(_rec)
        print(
            f"[lora r={_rank}] seed={_seed}: b={_rec['final_bits_per_byte']:.4f}, "
            f"訓練損失 {_rec['initial_train_loss']:.4f} -> {_rec['final_train_loss']:.4f}"
        )
rank_records = dict(sorted(rank_records.items()))
print(f"実験 C の追加の学習の合計時間: {time.time() - _t0_rank:.1f}s")
```

    [lora r=1] seed=0: b=2.7831, 訓練損失 6.2466 -> 5.1255
    [lora r=1] seed=1: b=2.7818, 訓練損失 6.2466 -> 5.1312
    [lora r=1] seed=2: b=2.7866, 訓練損失 6.2466 -> 5.1315
    [lora r=1] seed=3: b=2.7707, 訓練損失 6.2466 -> 5.1257
    [lora r=1] seed=4: b=2.7740, 訓練損失 6.2466 -> 5.1243
    [lora r=2] seed=0: b=2.7225, 訓練損失 6.2466 -> 4.9551
    [lora r=2] seed=1: b=2.7119, 訓練損失 6.2466 -> 4.9465
    [lora r=4] seed=0: b=2.6676, 訓練損失 6.2466 -> 4.7945
    [lora r=4] seed=1: b=2.6568, 訓練損失 6.2466 -> 4.7919
    [lora r=16] seed=0: b=2.5838, 訓練損失 6.2466 -> 4.5894
    [lora r=16] seed=1: b=2.5888, 訓練損失 6.2466 -> 4.5714
    [lora r=32] seed=0: b=2.5686, 訓練損失 6.2466 -> 4.5340
    [lora r=32] seed=1: b=2.5667, 訓練損失 6.2466 -> 4.5155
    [lora r=64] seed=0: b=2.5611, 訓練損失 6.2466 -> 4.5130
    [lora r=64] seed=1: b=2.5611, 訓練損失 6.2466 -> 4.4923
    [lora r=64] seed=2: b=2.5684, 訓練損失 6.2466 -> 4.4860
    [lora r=64] seed=3: b=2.5623, 訓練損失 6.2466 -> 4.5048
    [lora r=64] seed=4: b=2.5646, 訓練損失 6.2466 -> 4.5023
    実験 C の追加の学習の合計時間: 244.8s


### 6.6 不変条件のアサーション

実際に実行した全条件の記録を使って確認する(表として出力するだけでなく、すべて実行時に確認する)。

- 全条件で、評価集合の窓(ハッシュ)・評価の分母(UTF-8 バイト数)・ステップ数・学習履歴の長さが同一。
  分母の期待値は、評価ループとは独立に生テキストと分割位置から計算した値(5.4 節)と照合する。
- 全条件がエポック上限 $E$ を満たす。
- 学習可能パラメータ数が閉形式と一致する(全パラメータ微調整は $P$、LoRA は $4 L r d_{\mathrm{model}}$、
  疎微調整は LoRA $r=8$ と同数)。
- `SMOKE_TEST`の配線: 実際に使われたシード数・rank の集合・記録されたキーが、5.2 節で印字した
  実効水準と一致する。
- ベースモデル自体が学習の前後で変化していない。
- 学習後の LoRA(シード 0)で、マージ前後の出力が一致し、`unmerge()`後に元の重みへ戻る。


```python
_all_records = [r for rs in main_records.values() for r in rs] + [
    r for rank, rs in rank_records.items() if rank != MAIN_RANK for r in rs
]
_expected_keys = sorted(main_records["full"][0]["history_keys"])
for _r in _all_records:
    assert _r["split"] == "evaluation"
    assert _r["evaluation_windows_hash"] == EVALUATION_WINDOWS_HASH, "評価窓が条件間で異なる"
    assert _r["evaluation_denominator_bytes"] == EXPECTED_EVALUATION_BYTES, (
        "評価の分母が期待値と異なる"
    )
    assert _r["num_steps"] == NUM_STEPS and _r["history_length"] == NUM_STEPS, (
        "ステップ数が条件間で異なる"
    )
    assert _r["eval_step"][-1] == NUM_STEPS, "最終ステップの評価値が記録されていない"
    assert _r["history_keys"] == _expected_keys
    assert _r["num_steps"] * TOKENS_PER_STEP <= EPOCHS * len(train_ids), "エポック上限を超えた"
    assert _r["initial_train_loss"] == BASE_TRAIN_LOSS
    if _r["method"] == "full":
        assert _r["num_trainable_parameters"] == TOTAL_PARAMETERS
    elif _r["method"] == "lora":
        assert _r["num_trainable_parameters"] == 4 * NUM_LAYERS * _r["rank"] * D_MODEL
    else:
        assert _r["num_trainable_parameters"] == 4 * NUM_LAYERS * MAIN_RANK * D_MODEL
print(
    f"評価窓・評価の分母({EXPECTED_EVALUATION_BYTES:,} バイト)・ステップ数({NUM_STEPS})・"
    f"エポック上限・学習可能パラメータ数: 全 {len(_all_records)} 条件で OK"
)
assert (
    main_records["sparse"][0]["num_trainable_parameters"]
    == main_records["lora"][0]["num_trainable_parameters"]
)

# --- SMOKE_TEST の配線: 実際に使われた値と実効水準の照合 ---
assert all(len(rs) == NUM_SEEDS_JUDGED for rs in main_records.values())
assert all([r["seed"] for r in rs] == list(SEEDS_JUDGED) for rs in main_records.values())
assert tuple(rank_records) == tuple(RANKS), (tuple(rank_records), RANKS)
for _rank, _rs in rank_records.items():
    _n = NUM_SEEDS_JUDGED if _rank in JUDGED_RANKS else NUM_SEEDS_INTERMEDIATE
    assert len(_rs) == _n and all(r["rank"] == _rank for r in _rs)
    assert all(r["learning_rate"] == LEARNING_RATE["lora"] for r in _rs)
print(
    f"SMOKE_TEST の配線: シード数(判定 {NUM_SEEDS_JUDGED} / 中間 {NUM_SEEDS_INTERMEDIATE})・"
    f"rank の集合 {tuple(rank_records)}・学習率・記録キーが実効水準と一致: OK"
)

# --- ベースモデルが変化していないこと ---
assert (
    hash_tensor(torch.cat([p.detach().flatten().cpu() for p in base_model.parameters()]))
    == BASE_STATE_HASH
)
print("ベースモデルの重みは学習の前後で不変: OK")

# --- 学習後の LoRA でのマージ・アンマージ ---
_trained_merge_check = check_merge_roundtrip(
    kept_models["lora"]["model"], kept_models["lora"]["replaced"], _probe_tokens
)
assert _trained_merge_check["logits_changed_from_base"], (
    "学習後の LoRA の出力がベースと同一(学習されていない)"
)
print(
    f"学習後の LoRA(seed 0): マージ前後の出力の一致・unmerge 後の重みの復元: OK {_trained_merge_check}"
)
```

    評価窓・評価の分母(55,769 バイト)・ステップ数(87)・エポック上限・学習可能パラメータ数: 全 33 条件で OK
    SMOKE_TEST の配線: シード数(判定 5 / 中間 2)・rank の集合 (1, 2, 4, 8, 16, 32, 64)・学習率・記録キーが実効水準と一致: OK
    ベースモデルの重みは学習の前後で不変: OK
    学習後の LoRA(seed 0): マージ前後の出力の一致・unmerge 後の重みの復元: OK {'max_layer_error': 7.152557373046875e-06, 'max_logits_error': 1.811981201171875e-05, 'logits_changed_from_base': True}


### 6.7 判定の一次情報(全件)

判定の一次情報(各条件・各シードの評価集合の最終 bits-per-byte、学習開始時・最終の訓練損失、
較正の選択結果)を全件印字する。対比量はすべてこの出力から再計算できる。


```python
def primary_record(r: dict) -> dict:
    return {
        "seed": r["seed"],
        "final_bits_per_byte": r["final_bits_per_byte"],
        "initial_train_loss": r["initial_train_loss"],
        "final_train_loss": r["final_train_loss"],
    }


primary_data = {
    "smoke_test": SMOKE_TEST,
    "b0": B0,
    "calibration": {
        m: {
            "grid_final_calibration_bits_per_byte": {
                lr: v["final_calibration_bits_per_byte"] for lr, v in log["grid"].items()
            },
            "chosen_learning_rate": log["chosen_learning_rate"],
            "chosen_is_interior": log["chosen_is_interior"],
            "clip_threshold": log["clip_threshold"],
        }
        for m, log in calibration_log.items()
    },
    "b1_full": [primary_record(r) for r in main_records["full"]],
    "b2_lora_r8": [primary_record(r) for r in main_records["lora"]],
    "b3_sparse": [primary_record(r) for r in main_records["sparse"]],
    "b_rank": {
        str(k): [primary_record(r) for r in rs] for k, rs in rank_records.items() if k != MAIN_RANK
    },
}
print(json.dumps(primary_data, indent=2, ensure_ascii=False))
```

    {
      "smoke_test": false,
      "b0": 3.3927508811831832,
      "calibration": {
        "full": {
          "grid_final_calibration_bits_per_byte": {
            "3.162e-04": 2.314775427552616,
            "1.000e-03": 2.1481023635639778,
            "3.162e-03": 2.0547773001584866,
            "1.000e-02": 2.138175372367793,
            "3.162e-02": 2.929167797483712
          },
          "chosen_learning_rate": 0.0031622776601683794,
          "chosen_is_interior": true,
          "clip_threshold": 0.947061586380006
        },
        "lora": {
          "grid_final_calibration_bits_per_byte": {
            "3.162e-03": 2.7778558507255475,
            "1.000e-02": 2.641007578676406,
            "3.162e-02": 2.564793426579092,
            "1.000e-01": 3.353801542914629,
            "3.162e-01": 3.6255992486481117
          },
          "chosen_learning_rate": 0.03162277660168379,
          "chosen_is_interior": true,
          "clip_threshold": 0.28605208992958076
        },
        "sparse": {
          "grid_final_calibration_bits_per_byte": {
            "1.000e-02": 2.6999458185535308,
            "3.162e-02": 2.618324047429553,
            "1.000e-01": 2.6004232152943056,
            "3.162e-01": 2.6967213487586594,
            "1.000e+00": 3.207191381632571
          },
          "chosen_learning_rate": 0.1,
          "chosen_is_interior": true,
          "clip_threshold": 0.10036885887384424
        }
      },
      "b1_full": [
        {
          "seed": 0,
          "final_bits_per_byte": 2.261753018732079,
          "initial_train_loss": 6.246554385914522,
          "final_train_loss": 3.3946427289177388
        },
        {
          "seed": 1,
          "final_bits_per_byte": 2.2410188420045403,
          "initial_train_loss": 6.246554385914522,
          "final_train_loss": 3.3485009286917893
        },
        {
          "seed": 2,
          "final_bits_per_byte": 2.260983835517479,
          "initial_train_loss": 6.246554385914522,
          "final_train_loss": 3.3213474049287686
        },
        {
          "seed": 3,
          "final_bits_per_byte": 2.2376677234072124,
          "initial_train_loss": 6.246554385914522,
          "final_train_loss": 3.3529571533203124
        },
        {
          "seed": 4,
          "final_bits_per_byte": 2.2565791502079695,
          "initial_train_loss": 6.246554385914522,
          "final_train_loss": 3.363405055625766
        }
      ],
      "b2_lora_r8": [
        {
          "seed": 0,
          "final_bits_per_byte": 2.607666127234919,
          "initial_train_loss": 6.246554385914522,
          "final_train_loss": 4.658965705422794
        },
        {
          "seed": 1,
          "final_bits_per_byte": 2.617892988033689,
          "initial_train_loss": 6.246554385914522,
          "final_train_loss": 4.664204676011029
        },
        {
          "seed": 2,
          "final_bits_per_byte": 2.600680119165922,
          "initial_train_loss": 6.246554385914522,
          "final_train_loss": 4.643729894301471
        },
        {
          "seed": 3,
          "final_bits_per_byte": 2.616290870893274,
          "initial_train_loss": 6.246554385914522,
          "final_train_loss": 4.678813979204963
        },
        {
          "seed": 4,
          "final_bits_per_byte": 2.6137570138035047,
          "initial_train_loss": 6.246554385914522,
          "final_train_loss": 4.682102098651961
        }
      ],
      "b3_sparse": [
        {
          "seed": 0,
          "final_bits_per_byte": 2.639573765560938,
          "initial_train_loss": 6.246554385914522,
          "final_train_loss": 4.667633056640625
        },
        {
          "seed": 1,
          "final_bits_per_byte": 2.6362224353875425,
          "initial_train_loss": 6.246554385914522,
          "final_train_loss": 4.6555234422870715
        },
        {
          "seed": 2,
          "final_bits_per_byte": 2.6353373211193403,
          "initial_train_loss": 6.246554385914522,
          "final_train_loss": 4.650458122702206
        },
        {
          "seed": 3,
          "final_bits_per_byte": 2.6270704346953337,
          "initial_train_loss": 6.246554385914522,
          "final_train_loss": 4.652128092447916
        },
        {
          "seed": 4,
          "final_bits_per_byte": 2.657439059232543,
          "initial_train_loss": 6.246554385914522,
          "final_train_loss": 4.655873377182904
        }
      ],
      "b_rank": {
        "1": [
          {
            "seed": 0,
            "final_bits_per_byte": 2.783065702970397,
            "initial_train_loss": 6.246554385914522,
            "final_train_loss": 5.125500368604473
          },
          {
            "seed": 1,
            "final_bits_per_byte": 2.7818281756042706,
            "initial_train_loss": 6.246554385914522,
            "final_train_loss": 5.131210386986826
          },
          {
            "seed": 2,
            "final_bits_per_byte": 2.786602468514506,
            "initial_train_loss": 6.246554385914522,
            "final_train_loss": 5.131547755821078
          },
          {
            "seed": 3,
            "final_bits_per_byte": 2.770730057190796,
            "initial_train_loss": 6.246554385914522,
            "final_train_loss": 5.125692928538603
          },
          {
            "seed": 4,
            "final_bits_per_byte": 2.7739646510583915,
            "initial_train_loss": 6.246554385914522,
            "final_train_loss": 5.124293667662378
          }
        ],
        "2": [
          {
            "seed": 0,
            "final_bits_per_byte": 2.7225443088526413,
            "initial_train_loss": 6.246554385914522,
            "final_train_loss": 4.955084587545956
          },
          {
            "seed": 1,
            "final_bits_per_byte": 2.7119336585409175,
            "initial_train_loss": 6.246554385914522,
            "final_train_loss": 4.946509507123162
          }
        ],
        "4": [
          {
            "seed": 0,
            "final_bits_per_byte": 2.6676139133707717,
            "initial_train_loss": 6.246554385914522,
            "final_train_loss": 4.7944904402190565
          },
          {
            "seed": 1,
            "final_bits_per_byte": 2.656756615889929,
            "initial_train_loss": 6.246554385914522,
            "final_train_loss": 4.79187562231924
          }
        ],
        "16": [
          {
            "seed": 0,
            "final_bits_per_byte": 2.5838182644677516,
            "initial_train_loss": 6.246554385914522,
            "final_train_loss": 4.589415426815258
          },
          {
            "seed": 1,
            "final_bits_per_byte": 2.5887775159130864,
            "initial_train_loss": 6.246554385914522,
            "final_train_loss": 4.5714034734987745
          }
        ],
        "32": [
          {
            "seed": 0,
            "final_bits_per_byte": 2.5686047925960485,
            "initial_train_loss": 6.246554385914522,
            "final_train_loss": 4.533993709788603
          },
          {
            "seed": 1,
            "final_bits_per_byte": 2.566731529672605,
            "initial_train_loss": 6.246554385914522,
            "final_train_loss": 4.5155038871017155
          }
        ],
        "64": [
          {
            "seed": 0,
            "final_bits_per_byte": 2.561051523827174,
            "initial_train_loss": 6.246554385914522,
            "final_train_loss": 4.513031005859375
          },
          {
            "seed": 1,
            "final_bits_per_byte": 2.5610953737566335,
            "initial_train_loss": 6.246554385914522,
            "final_train_loss": 4.492334582758885
          },
          {
            "seed": 2,
            "final_bits_per_byte": 2.5683556728290062,
            "initial_train_loss": 6.246554385914522,
            "final_train_loss": 4.485976514629289
          },
          {
            "seed": 3,
            "final_bits_per_byte": 2.562323345463352,
            "initial_train_loss": 6.246554385914522,
            "final_train_loss": 4.504794012331495
          },
          {
            "seed": 4,
            "final_bits_per_byte": 2.5646156963666584,
            "initial_train_loss": 6.246554385914522,
            "final_train_loss": 4.502315266927083
          }
        ]
      }
    }


### 6.8 実験 A: 全パラメータ微調整との比較


```python
def seed_mean_and_standard_error(
    records: list[dict], key: str = "final_bits_per_byte"
) -> tuple[float, float]:
    values = np.array([r[key] for r in records], dtype=float)
    standard_error = (
        float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")
    )
    return float(values.mean()), standard_error


def three_way_verdict(value: float, sigma: float) -> str:
    if not (math.isfinite(value) and math.isfinite(sigma)):
        return "判定不能"
    if value > 2 * sigma:
        return "支持"
    if value < -2 * sigma:
        return "反証"
    return "判定不能"


b1_mean, sigma_b1 = seed_mean_and_standard_error(main_records["full"])
b2_mean, sigma_b2 = seed_mean_and_standard_error(main_records["lora"])
b3_mean, sigma_b3 = seed_mean_and_standard_error(main_records["sparse"])
_u, _v = B0 - b2_mean, B0 - b1_mean
rho = _u / _v
sigma_rho = math.sqrt(sigma_b2**2 / _v**2 + _u**2 * sigma_b1**2 / _v**4)

# 前提条件 A1(LoRA の結果に依存しない量で定義)
precondition_status["A1"] = bool(_v >= PRECONDITION_A1_FACTOR * sigma_b1)

if rho - 2 * sigma_rho >= DELTA_A:
    verdict_A = "支持"
elif rho + 2 * sigma_rho < DELTA_A:
    verdict_A = "反証"
else:
    verdict_A = "判定不能"

# 診断量: シードの復元抽出によるブートストラップ区間(判定には使わない)
_rng = np.random.default_rng(0)
_b1_values = np.array([r["final_bits_per_byte"] for r in main_records["full"]])
_b2_values = np.array([r["final_bits_per_byte"] for r in main_records["lora"]])
_boot = []
for _ in range(BOOTSTRAP_RESAMPLES):
    _b1s = _rng.choice(_b1_values, size=len(_b1_values), replace=True).mean()
    _b2s = _rng.choice(_b2_values, size=len(_b2_values), replace=True).mean()
    _boot.append((B0 - _b2s) / (B0 - _b1s))
rho_bootstrap_interval = tuple(float(x) for x in np.percentile(_boot, [2.5, 97.5]))

print(
    f"{_smoke_tag}b0={B0:.4f}, b1={b1_mean:.4f}(標準誤差 {sigma_b1:.4f}), b2={b2_mean:.4f}(標準誤差 {sigma_b2:.4f})"
)
print(
    f"{_smoke_tag}rho={rho:.4f}, sigma_rho={sigma_rho:.4f}, "
    f"[rho - 2 sigma, rho + 2 sigma]=[{rho - 2 * sigma_rho:.4f}, {rho + 2 * sigma_rho:.4f}], delta={DELTA_A}"
)
print(
    f"{_smoke_tag}前提条件 A1(b0 - b1 = {_v:.4f} >= {PRECONDITION_A1_FACTOR:.0f} x {sigma_b1:.4f}): "
    f"{precondition_status['A1']}"
)
print(f"{_smoke_tag}判定関数の結果: {verdict_A}")
print(
    f"{_smoke_tag}診断量: ブートストラップ 95% 区間 = [{rho_bootstrap_interval[0]:.4f}, "
    f"{rho_bootstrap_interval[1]:.4f}]"
)


def mean_over_records(records: list[dict], key: str) -> np.ndarray:
    # 行列ごとの値(層順、Query・Value の順)のシード平均
    return np.mean([r[key] for r in records], axis=0)


for _method in ("full", "lora"):
    _ratios = mean_over_records(main_records[_method], "relative_update_norms")
    print(
        f"{_smoke_tag}診断量 ||Delta W||_F / ||W0||_F [{_method}](層ごとに Query, Value): "
        + ", ".join(f"{x:.4f}" for x in _ratios)
    )
_energy = mean_over_records(main_records["full"], "top_rank_energy")
print(
    f"{_smoke_tag}診断量: 全パラメータ微調整の Delta W の上位 {MAIN_RANK} 特異値のエネルギー割合"
    f"(層ごとに Query, Value): " + ", ".join(f"{x:.4f}" for x in _energy)
)


def final_slope(record: dict) -> float:
    # 評価集合の学習曲線の最後の 2 つの記録点の傾き (b(T) - b(T')) / (T - T')
    steps, values = record["eval_step"], record["eval_bits_per_byte"]
    return (values[-1] - values[-2]) / (steps[-1] - steps[-2])


def mean_final_slope(records: list[dict]) -> float:
    return float(np.mean([final_slope(r) for r in records]))


for _method in ("full", "lora", "sparse"):
    print(
        f"{_smoke_tag}診断量: 学習曲線の最終区間の傾き(シード平均、bits-per-byte / ステップ)"
        f"[{_method}]: {mean_final_slope(main_records[_method]):+.6f}"
    )
for _method in ("full", "lora", "sparse"):
    _gaps = [min(r["eval_bits_per_byte"]) - r["final_bits_per_byte"] for r in main_records[_method]]
    print(
        f"{_smoke_tag}診断量: 最良値 - 最終値(過学習の程度)[{_method}]: "
        + ", ".join(f"{g:+.4f}" for g in _gaps)
    )
```

    b0=3.3928, b1=2.2516(標準誤差 0.0051), b2=2.6113(標準誤差 0.0032)
    rho=0.6848, sigma_rho=0.0041, [rho - 2 sigma, rho + 2 sigma]=[0.6766, 0.6931], delta=0.9
    前提条件 A1(b0 - b1 = 1.1412 >= 10 x 0.0051): True
    判定関数の結果: 反証
    診断量: ブートストラップ 95% 区間 = [0.6778, 0.6921]
    診断量 ||Delta W||_F / ||W0||_F [full](層ごとに Query, Value): 0.2622, 0.1630, 0.2783, 0.2015, 0.2794, 0.2165, 0.2849, 0.2279
    診断量 ||Delta W||_F / ||W0||_F [lora](層ごとに Query, Value): 0.8003, 0.5891, 1.0563, 0.7691, 1.1512, 0.9389, 1.2392, 1.1682
    診断量: 全パラメータ微調整の Delta W の上位 8 特異値のエネルギー割合(層ごとに Query, Value): 0.3326, 0.2861, 0.4058, 0.3416, 0.3637, 0.3397, 0.3824, 0.4063
    診断量: 学習曲線の最終区間の傾き(シード平均、bits-per-byte / ステップ)[full]: -0.000308
    診断量: 学習曲線の最終区間の傾き(シード平均、bits-per-byte / ステップ)[lora]: -0.000178
    診断量: 学習曲線の最終区間の傾き(シード平均、bits-per-byte / ステップ)[sparse]: -0.000145
    診断量: 最良値 - 最終値(過学習の程度)[full]: +0.0000, +0.0000, +0.0000, +0.0000, +0.0000
    診断量: 最良値 - 最終値(過学習の程度)[lora]: +0.0000, +0.0000, +0.0000, +0.0000, +0.0000
    診断量: 最良値 - 最終値(過学習の程度)[sparse]: +0.0000, +0.0000, +0.0000, +0.0000, +0.0000


評価集合の学習曲線(方式ごとにシードを重ね描き、ステップ 0 は $b_0$)と、最終 bits-per-byte の
シードごとの散布図を示す。


```python
def curve_histories(records: list[dict]) -> list[dict]:
    return [
        {"step": [0] + r["eval_step"], "bits_per_byte": [B0] + r["eval_bits_per_byte"]}
        for r in records
    ]


fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
plot_learning_curves_multi_seed(
    {
        "Full fine-tuning": curve_histories(main_records["full"]),
        f"LoRA (r={MAIN_RANK})": curve_histories(main_records["lora"]),
        "Random-mask sparse": curve_histories(main_records["sparse"]),
    },
    step_key="step",
    value_key="bits_per_byte",
    title="Evaluation bits-per-byte during fine-tuning",
    ylabel="bits-per-byte",
    ax=axes[0],
)
axes[0].axhline(B0, color="gray", linestyle=":", label="base model (b0)")
axes[0].legend(fontsize=8)
plot_seed_scatter(
    {
        "Full fine-tuning": [r["final_bits_per_byte"] for r in main_records["full"]],
        f"LoRA (r={MAIN_RANK})": [r["final_bits_per_byte"] for r in main_records["lora"]],
        "Random-mask sparse": [r["final_bits_per_byte"] for r in main_records["sparse"]],
    },
    title="Final evaluation bits-per-byte",
    ylabel="bits-per-byte",
    ax=axes[1],
)
fig.tight_layout()
plt.show()
```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/012_low_rank_adaptation/output_36_0.png)
    




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/03_efficient_training/012_low_rank_adaptation.ipynb
