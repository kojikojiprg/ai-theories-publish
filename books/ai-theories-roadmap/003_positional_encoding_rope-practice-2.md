---
title: "位置エンコーディング(Positional Encoding)/ RoPE(実装・実験編 2/3)"
---

この記事は後編(実装・実験編 2/3)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/003_positional_encoding_rope-practice-1)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/003_positional_encoding_rope-practice-3)。

### 6.2 実験 B: 位置エンコーディング方式の比較(学習を伴う)

**タスク設計についての注記**:`theories/README.md`の学習順序では 006「小型 GPT の事前学習」が本トピック(003)を前提としており、ここで言語モデリングを扱うと前提関係が逆転してしまう。そのため、位置情報が本質的に効く合成タスクとして **可変長 copy task** を用いる。

**タスク**: 系列 $[x_1, \dots, x_L, \mathrm{SEP}, x_1, \dots, x_L]$ を構成し、因果マスクを用いた Decoder のみの構成(EncoderBlock を因果マスクで使うことで、cross-attention を持たない GPT スタイルの decoder-only 構成になる、002・実験 1 と同じ手法)で後半部分(2 回目の $x_1, \dots, x_L$)を予測する。損失・精度は後半部分のみについて計算する。系列長 $L$ を学習の各ステップでランダムに変えることで、モデルが特定の絶対長ではなく「相対的な参照パターン」を学習するように促す。

**語彙サイズについての設計判断**: 語彙サイズを大きく取ると、先頭から連続する部分列の中で重複がほとんど発生しないため、「直前に出た同じトークンを探し、その次のトークンをコピーする」という **内容に基づく induction 機構**(位置情報がなくても学習できる)だけでタスクがある程度解けてしまい、位置エンコーディング方式間の差が出にくい。そこで語彙サイズを $6$ と小さく設定し、系列内で同じトークンが複数回出現するようにした。これにより「直前の同一トークンの次を見る」だけでは参照先が曖昧になり、位置情報(絶対位置または相対位置)に基づく参照が本質的に必要になる。


```python
CONTENT_VOCAB_SIZE = 6  # 小さい語彙サイズで内容ベース induction のみでの解決を妨げる(上記参照)
SEP_TOKEN_ID = CONTENT_VOCAB_SIZE
VOCAB_SIZE_B = CONTENT_VOCAB_SIZE + 1

D_MODEL_B, N_HEADS_B, D_FF_B, N_LAYERS_B = 64, 4, 256, 2
L_MIN_TRAIN, L_MAX_TRAIN = 4, 16  # 学習時の系列長 L の範囲
EXTRAPOLATION_RATIOS = [1.5, 2.0, 3.0, 4.0]
EXTRAPOLATION_LENGTHS = [int(L_MAX_TRAIN * r) for r in EXTRAPOLATION_RATIOS]
MAX_EVAL_LEN_B = 2 * max(EXTRAPOLATION_LENGTHS) + 1  # cos/sin キャッシュなどのバッファ用

TRAIN_STEPS_B = 2500
BATCH_SIZE_B = 64
LR_B = 3e-4
SEEDS_B = [0, 1, 2]  # 条件間の差を seed 間のばらつきと区別するため、複数 seed で学習する

T5_NUM_BUCKETS = 32
T5_MAX_DISTANCE = 64

print(f"学習時の L の範囲: [{L_MIN_TRAIN}, {L_MAX_TRAIN}]")
print(f"外挿評価の L: {EXTRAPOLATION_LENGTHS} (学習長 {L_MAX_TRAIN} の {EXTRAPOLATION_RATIOS} 倍)")
print(f"学習・評価に用いる seed: {SEEDS_B}")
```

    学習時の L の範囲: [4, 16]
    外挿評価の L: [24, 32, 48, 64] (学習長 16 の [1.5, 2.0, 3.0, 4.0] 倍)
    学習・評価に用いる seed: [0, 1, 2]



```python
def make_copy_batch(
    batch_size: int, seq_len_l: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """可変長 copy task のバッチを生成する。

    系列は [x_1, ..., x_L, SEP, x_1, ..., x_L] の形で、次のトークン予測用に
    (入力, 目標, 損失を計算する位置のマスク)を返す。損失マスクは後半部分
    (2 回目に現れる x_1, ..., x_L を予測する位置)のみ True になる。
    """
    first_half = torch.randint(0, CONTENT_VOCAB_SIZE, (batch_size, seq_len_l), device=device)
    sep = torch.full((batch_size, 1), SEP_TOKEN_ID, device=device)
    full_sequence = torch.cat([first_half, sep, first_half], dim=1)  # (B, 2L+1)

    input_ids = full_sequence[:, :-1]
    target_ids = full_sequence[:, 1:]
    total_len = input_ids.size(1)
    # 入力位置 t(0-indexed)の目標が後半部分の予測に対応するのは t >= L のとき
    loss_mask = torch.arange(total_len, device=device) >= seq_len_l
    return input_ids, target_ids, loss_mask


# 動作確認
_ids, _tgt, _mask = make_copy_batch(2, 4, "cpu")
print("input :", _ids[0].tolist())
print("target:", _tgt[0].tolist())
print("mask  :", _mask.tolist())
```

    input : [0, 5, 3, 0, 6, 0, 5, 3]
    target: [5, 3, 0, 6, 0, 5, 3, 0]
    mask  : [False, False, False, False, True, True, True, True]


**モデル**: 002 で実装した`EncoderBlock`(因果マスク付き、正規化前置)を 2 層積んだものを使う。埋め込みへの加算方式(正弦波・学習可能な絶対位置埋め込み)は埋め込み層に、RoPE・Shaw et al. 方式・T5・ALiBi は各層の`self_attn`を対応する`positional_transform`/`attention_score_bias`を指定した`MultiHeadAttention`に差し替えることで組み込む。`EncoderBlock`自体は変更しない。


```python
class PositionalEncodingCopyModel(nn.Module):
    """可変長 copy task 用のモデル(比較する 7 条件を 1 つのクラスで表現する)。

    embedding_mode: 埋め込みへの加算方式("none" / "sinusoidal" / "learned")。
    positional_transform_factory: 各層の self_attn に渡す QueryKeyPositionalTransform
        のファクトリ(例: RoPE)。None なら使わない。
    score_bias_factory: 各層の self_attn に渡す AttentionScoreBias のファクトリ
        (例: Shaw et al. 方式・T5・ALiBi)。None なら使わない。
    """

    def __init__(
        self,
        embedding_mode: str,
        positional_transform_factory=None,
        score_bias_factory=None,
        max_learned_len: int = 2 * L_MAX_TRAIN + 1,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE_B, D_MODEL_B)

        self.pos_embedding: nn.Module | None
        if embedding_mode == "sinusoidal":
            self.pos_embedding = SinusoidalPositionalEncoding(D_MODEL_B, max_len=MAX_EVAL_LEN_B)
        elif embedding_mode == "learned":
            # 学習時の最大系列長ちょうどに max_len を設定する(3.4 節: 外挿できないことを
            # 明示的に示すため、学習長を超える位置を要求すると ValueError が送出される)。
            self.pos_embedding = LearnedAbsolutePositionalEmbedding(
                D_MODEL_B, max_len=max_learned_len
            )
        else:
            self.pos_embedding = None

        self.blocks = nn.ModuleList()
        for _ in range(N_LAYERS_B):
            block = EncoderBlock(D_MODEL_B, N_HEADS_B, D_FF_B, norm_first=True)
            if positional_transform_factory is not None or score_bias_factory is not None:
                block.self_attn = MultiHeadAttention(
                    D_MODEL_B,
                    N_HEADS_B,
                    positional_transform=(
                        positional_transform_factory() if positional_transform_factory else None
                    ),
                    attention_score_bias=(score_bias_factory() if score_bias_factory else None),
                )
            self.blocks.append(block)

        self.output_proj = nn.Linear(D_MODEL_B, VOCAB_SIZE_B)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """順伝播。

        Returns:
            (logits, attn_weights_per_layer) のタプル。attn_weights_per_layer は
            層ごとの Attention 重み(形状 (B, h, S, S))のリストで、リストの長さは
            self.blocks の層数。最終層だけでなく全層を観測できるようにするため、
            上書きせずリストに積む(6.3 節: copy 機構がどの層・どのヘッドに
            載るかを層をまたいで調べるために必要)。
        """
        h = self.token_embedding(input_ids)
        if self.pos_embedding is not None:
            h = self.pos_embedding(h)
        seq_len = input_ids.size(1)
        mask = create_causal_mask(seq_len, device=input_ids.device)
        attn_weights_per_layer = []
        for block in self.blocks:
            h, attn_weights = block(h, mask)
            attn_weights_per_layer.append(attn_weights)
        logits = self.output_proj(h)
        return logits, attn_weights_per_layer


D_K_B = D_MODEL_B // N_HEADS_B

MODEL_FACTORIES = {
    "none": lambda: PositionalEncodingCopyModel("none"),
    "sinusoidal": lambda: PositionalEncodingCopyModel("sinusoidal"),
    "learned": lambda: PositionalEncodingCopyModel("learned"),
    "shaw": lambda: PositionalEncodingCopyModel(
        "none",
        score_bias_factory=lambda: ShawRelativePositionBias(
            D_K_B, max_relative_position=L_MAX_TRAIN
        ),
    ),
    "t5": lambda: PositionalEncodingCopyModel(
        "none",
        score_bias_factory=lambda: T5RelativePositionBias(
            N_HEADS_B, num_buckets=T5_NUM_BUCKETS, max_distance=T5_MAX_DISTANCE, bidirectional=False
        ),
    ),
    "alibi": lambda: PositionalEncodingCopyModel(
        "none", score_bias_factory=lambda: ALiBiPositionBias(N_HEADS_B)
    ),
    "rope": lambda: PositionalEncodingCopyModel(
        "none",
        positional_transform_factory=lambda: RotaryPositionEmbedding(
            D_K_B, max_position=MAX_EVAL_LEN_B
        ),
    ),
}
CONDITION_LABELS = {
    "none": "None (baseline)",
    "sinusoidal": "Sinusoidal",
    "learned": "Learned Absolute",
    "shaw": "Shaw et al.",
    "t5": "T5 relative bias",
    "alibi": "ALiBi",
    "rope": "RoPE",
}
print(f"比較する条件: {list(MODEL_FACTORIES.keys())}")
```

    比較する条件: ['none', 'sinusoidal', 'learned', 'shaw', 't5', 'alibi', 'rope']



```python
def evaluate_copy_task(
    model: nn.Module, seq_len_l: int, n_batches: int = 4, batch_size: int = 64, device: str = "cpu"
) -> tuple[float, float]:
    """指定した L で評価し、(平均損失, 精度) を返す。max_len を超える場合は (nan, nan)。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for _ in range(n_batches):
            input_ids, target_ids, loss_mask = make_copy_batch(batch_size, seq_len_l, device)
            try:
                logits, _ = model(input_ids)
            except ValueError:
                return float("nan"), float("nan")
            log_probs = torch.log_softmax(logits, dim=-1)
            masked_log_probs = log_probs[:, loss_mask].reshape(-1, VOCAB_SIZE_B)
            masked_targets = target_ids[:, loss_mask].reshape(-1)
            loss = nn.functional.nll_loss(masked_log_probs, masked_targets)
            total_loss += loss.item()
            pred = logits[:, loss_mask].argmax(dim=-1)
            correct += (pred == target_ids[:, loss_mask]).sum().item()
            total += pred.numel()
    return total_loss / n_batches, correct / total


def train_copy_model(name: str, seed: int = SEEDS_B[0]) -> tuple[nn.Module, float]:
    """copy task で 1 条件を学習し、(モデル, 学習時間[秒]) を返す。"""
    torch.manual_seed(seed)
    model = MODEL_FACTORIES[name]().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_B)

    start = time.time()
    for _ in range(TRAIN_STEPS_B):
        seq_len_l = torch.randint(L_MIN_TRAIN, L_MAX_TRAIN + 1, (1,)).item()
        input_ids, target_ids, loss_mask = make_copy_batch(BATCH_SIZE_B, seq_len_l, DEVICE)
        logits, _ = model(input_ids)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = nn.functional.nll_loss(
            log_probs[:, loss_mask].reshape(-1, VOCAB_SIZE_B), target_ids[:, loss_mask].reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    elapsed = time.time() - start
    return model, elapsed
```


```python
def _nan_safe_stat(arr: np.ndarray, func) -> list[float]:
    """列(系列長)ごとに集計する。全 seed が nan の列(学習可能な絶対位置埋め込みが
    max_len を超えて評価不能になった場合)は nan をそのまま返し、`np.nanmean`
    などを全 nan 列に対して呼んで出る RuntimeWarning を避ける。
    """
    result = []
    for col in arr.T:
        if np.all(np.isnan(col)):
            result.append(float("nan"))
        else:
            result.append(float(func(col)))
    return result


results_exp_b = {}
trained_models_b = {}
eval_lengths_b = [L_MAX_TRAIN] + EXTRAPOLATION_LENGTHS

for name in MODEL_FACTORIES:
    per_seed_losses, per_seed_accs, per_seed_times = [], [], []
    trained_models_b[name] = {}

    for seed in SEEDS_B:
        model_b, elapsed_b = train_copy_model(name, seed=seed)
        trained_models_b[name][seed] = model_b
        per_seed_times.append(elapsed_b)

        losses_b, accs_b = [], []
        for l_eval in eval_lengths_b:
            loss_b, acc_b = evaluate_copy_task(model_b, l_eval, device=DEVICE)
            losses_b.append(loss_b)
            accs_b.append(acc_b)
        per_seed_losses.append(losses_b)
        per_seed_accs.append(accs_b)

    losses_arr = np.array(per_seed_losses)  # (len(SEEDS_B), len(eval_lengths_b))
    accs_arr = np.array(per_seed_accs)

    results_exp_b[name] = {
        "lengths": eval_lengths_b,
        "losses_mean": _nan_safe_stat(losses_arr, np.nanmean),
        "losses_min": _nan_safe_stat(losses_arr, np.nanmin),
        "losses_max": _nan_safe_stat(losses_arr, np.nanmax),
        "accs_mean": _nan_safe_stat(accs_arr, np.nanmean),
        "accs_min": _nan_safe_stat(accs_arr, np.nanmin),
        "accs_max": _nan_safe_stat(accs_arr, np.nanmax),
        "time_mean": sum(per_seed_times) / len(per_seed_times),
        "per_seed_accs": per_seed_accs,
    }

    print(
        f"{CONDITION_LABELS[name]:20s} time={results_exp_b[name]['time_mean']:5.1f}s(seed 平均)  "
        + "  ".join(
            f"L={l_eval}:acc={mean:.3f}[{lo:.3f},{hi:.3f}]"
            for l_eval, mean, lo, hi in zip(
                eval_lengths_b,
                results_exp_b[name]["accs_mean"],
                results_exp_b[name]["accs_min"],
                results_exp_b[name]["accs_max"],
                strict=True,
            )
        )
    )
```

    None (baseline)      time= 21.7s(seed 平均)  L=16:acc=0.634[0.587,0.668]  L=24:acc=0.478[0.448,0.501]  L=32:acc=0.390[0.352,0.424]  L=48:acc=0.308[0.284,0.342]  L=64:acc=0.266[0.246,0.298]


    Sinusoidal           time= 21.8s(seed 平均)  L=16:acc=0.986[0.982,0.991]  L=24:acc=0.221[0.187,0.248]  L=32:acc=0.238[0.214,0.269]  L=48:acc=0.242[0.230,0.254]  L=64:acc=0.213[0.203,0.225]


    Learned Absolute     time= 21.8s(seed 平均)  L=16:acc=0.813[0.787,0.848]  L=24:acc=nan[nan,nan]  L=32:acc=nan[nan,nan]  L=48:acc=nan[nan,nan]  L=64:acc=nan[nan,nan]


    Shaw et al.          time= 23.7s(seed 平均)  L=16:acc=0.959[0.929,0.975]  L=24:acc=0.891[0.798,0.947]  L=32:acc=0.813[0.674,0.904]  L=48:acc=0.642[0.489,0.756]  L=64:acc=0.505[0.392,0.595]


    T5 relative bias     time= 22.2s(seed 平均)  L=16:acc=0.725[0.576,0.862]  L=24:acc=0.376[0.344,0.424]  L=32:acc=0.451[0.269,0.631]  L=48:acc=0.275[0.258,0.286]  L=64:acc=0.277[0.229,0.323]


    ALiBi                time= 22.1s(seed 平均)  L=16:acc=0.967[0.964,0.969]  L=24:acc=0.920[0.913,0.925]  L=32:acc=0.855[0.849,0.863]  L=48:acc=0.711[0.692,0.722]  L=64:acc=0.603[0.578,0.615]


    RoPE                 time= 23.8s(seed 平均)  L=16:acc=0.988[0.983,0.992]  L=24:acc=0.330[0.280,0.367]  L=32:acc=0.311[0.299,0.329]  L=48:acc=0.173[0.168,0.177]  L=64:acc=0.167[0.163,0.175]


#### 実験 B-1: 学習長内での性能

学習時と同じ範囲(ここでは $L = L_{\max} = 16$)での精度を比較する。ここで測っているのは **位置情報の表現力そのもの** である(学習長を超えないため、外挿性能は問わない)。


```python
fig, ax = plt.subplots(figsize=(7.5, 4.5))
names_b1 = list(MODEL_FACTORIES.keys())
means_b1 = np.array(
    [results_exp_b[name]["accs_mean"][0] for name in names_b1]
)  # 先頭が L_MAX_TRAIN
mins_b1 = np.array([results_exp_b[name]["accs_min"][0] for name in names_b1])
maxs_b1 = np.array([results_exp_b[name]["accs_max"][0] for name in names_b1])
yerr_b1 = np.vstack([means_b1 - mins_b1, maxs_b1 - means_b1])

colors_b1 = plt.cm.tab10(np.linspace(0, 1, len(names_b1)))
ax.bar(
    [CONDITION_LABELS[n] for n in names_b1],
    means_b1,
    yerr=yerr_b1,
    capsize=4,
    color=colors_b1,
)
ax.set_ylabel(f"Accuracy at L = {L_MAX_TRAIN} (within training range)")
ax.set_title(f"Experiment B-1: in-distribution accuracy (mean, min-max over {len(SEEDS_B)} seeds)")
ax.set_ylim(0, 1.05)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

for name in names_b1:
    r = results_exp_b[name]
    print(
        f"{CONDITION_LABELS[name]:20s} L={L_MAX_TRAIN}: "
        f"loss={r['losses_mean'][0]:.4f}  "
        f"acc(mean)={r['accs_mean'][0]:.4f}  "
        f"acc(min-max)=[{r['accs_min'][0]:.4f}, {r['accs_max'][0]:.4f}]"
    )
```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/003_positional_encoding_rope/output_47_0.png)
    


    None (baseline)      L=16: loss=0.9653  acc(mean)=0.6340  acc(min-max)=[0.5869, 0.6677]
    Sinusoidal           L=16: loss=0.0356  acc(mean)=0.9863  acc(min-max)=[0.9822, 0.9907]
    Learned Absolute     L=16: loss=0.4737  acc(mean)=0.8129  acc(min-max)=[0.7874, 0.8477]
    Shaw et al.          L=16: loss=0.1126  acc(mean)=0.9591  acc(min-max)=[0.9292, 0.9753]
    T5 relative bias     L=16: loss=0.7192  acc(mean)=0.7253  acc(min-max)=[0.5757, 0.8616]
    ALiBi                L=16: loss=0.0940  acc(mean)=0.9666  acc(min-max)=[0.9639, 0.9692]
    RoPE                 L=16: loss=0.0348  acc(mean)=0.9878  acc(min-max)=[0.9829, 0.9917]


#### 実験 B-2: 学習長を超える外挿性能

学習時の最大系列長($L_{\max}=16$)の 1.5, 2, 3, 4 倍の複数の長さで評価する。ここで測っているのは **外挿(length extrapolation)性能** である。


```python
fig, ax = plt.subplots(figsize=(8.5, 5.0))
for name in MODEL_FACTORIES:
    r = results_exp_b[name]
    lengths_b2 = r["lengths"]
    means_b2 = r["accs_mean"]
    mins_b2 = r["accs_min"]
    maxs_b2 = r["accs_max"]
    (line,) = ax.plot(lengths_b2, means_b2, marker="o", label=CONDITION_LABELS[name])
    ax.fill_between(lengths_b2, mins_b2, maxs_b2, color=line.get_color(), alpha=0.15)

ax.axvline(
    L_MAX_TRAIN,
    color="gray",
    linestyle="--",
    linewidth=1,
    label=f"Training max length (L={L_MAX_TRAIN})",
)
ax.text(
    EXTRAPOLATION_LENGTHS[1],
    0.05,
    "Learned Absolute: not evaluable beyond max_len (all-nan)",
    fontsize=8,
    color="gray",
    ha="center",
)
ax.set_xlabel("Evaluation sequence length L")
ax.set_ylabel("Accuracy")
ax.set_title(f"Experiment B-2: length extrapolation (mean, min-max band over {len(SEEDS_B)} seeds)")
ax.set_ylim(-0.05, 1.05)
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/003_positional_encoding_rope/output_49_0.png)
    


**T5 の相対距離バケットの補足分析**: T5(`num_buckets=32`・`max_distance=64`・`bidirectional=False`)が学習中に実際に受け取る相対距離の範囲と、評価時に受け取る範囲を比較する。学習時の入力系列長は最大でも $2 L_{\max} = 32$ であり、causal な自己注意で意味を持つ相対距離(Key が Query 以前にある場合)は高々 $31$ である。


```python
t5_probe = T5RelativePositionBias(
    N_HEADS_B, num_buckets=T5_NUM_BUCKETS, max_distance=T5_MAX_DISTANCE, bidirectional=False
)

max_rel_dist_train = 2 * L_MAX_TRAIN - 1  # 学習中に現れる最大の相対距離(n - m の絶対値)
train_rel_distances = torch.arange(0, max_rel_dist_train + 1)
train_buckets = t5_probe.relative_position_bucket(-train_rel_distances)
max_trained_bucket = train_buckets.max().item()

all_buckets = torch.arange(T5_NUM_BUCKETS)
untrained_buckets = (all_buckets > max_trained_bucket).sum().item()

eval_rel_dist_l64 = 2 * EXTRAPOLATION_LENGTHS[-1] - 1  # L=64 での最大相対距離
eval_bucket_l64 = t5_probe.relative_position_bucket(torch.tensor([-eval_rel_dist_l64])).item()

print(f"学習中に現れる最大相対距離: {max_rel_dist_train} -> バケット {max_trained_bucket}")
print(f"学習中に一度も勾配を受けないバケット数: {untrained_buckets} / {T5_NUM_BUCKETS}")
eval_l64 = EXTRAPOLATION_LENGTHS[-1]
print(f"評価 L={eval_l64} での最大相対距離: {eval_rel_dist_l64} -> バケット {eval_bucket_l64}")
```

    学習中に現れる最大相対距離: 31 -> バケット 23
    学習中に一度も勾配を受けないバケット数: 8 / 32
    評価 L=64 での最大相対距離: 127 -> バケット 31


#### 実験 B の結果・考察

実測値(6.2 節の実行結果より、$3$ seed(0, 1, 2)の平均値、括弧内は最小値・最大値)は以下の通り。

**実験 B-1(学習長内 $L=16$、3.2〜3.8 節との対応)**:

| 条件 | 損失(平均) | 精度(平均 [最小, 最大]) |
|---|---:|---|
| なし(対照群) | 0.9653 | 0.6340 [0.5869, 0.6677] |
| 正弦波方式 | 0.0356 | 0.9863 [0.9822, 0.9907] |
| 学習可能な絶対位置埋め込み | 0.4737 | 0.8129 [0.7874, 0.8477] |
| Shaw et al. 方式 | 0.1126 | 0.9591 [0.9292, 0.9753] |
| T5 の相対位置バイアス | 0.7192 | 0.7253 [0.5757, 0.8616] |
| ALiBi | 0.0940 | 0.9666 [0.9639, 0.9692] |
| RoPE | 0.0348 | 0.9878 [0.9829, 0.9917] |

3 seed の最小・最大の範囲を比較すると、条件間には次のようなまとまりが見える。

- **RoPE(0.988)・正弦波方式(0.986)** は、最小値(それぞれ 0.983・0.982)が ALiBi・Shaw et al. 方式の最大値(それぞれ 0.969・0.975)を上回っており、この 2 条件は他の条件よりはっきり高い精度に達したと言える。
- **ALiBi(0.967)・Shaw et al. 方式(0.959)** は範囲が重なっており(ALiBi の範囲 $[0.964, 0.969]$ は Shaw et al. 方式の範囲 $[0.929, 0.975]$ に完全に含まれる)、この 2 条件の間の差は $3$ seed では区別できない。
- **学習可能な絶対位置埋め込み(0.813)** は ALiBi・Shaw et al. 方式より明確に低い(最大値 0.848 が Shaw et al. 方式の最小値 0.929 を下回る)。
- **T5(0.725)・対照群(0.634)** は範囲が重なっており(対照群の範囲 $[0.587, 0.668]$ は T5 の範囲 $[0.576, 0.862]$ に含まれる)、両者の間の差は $3$ seed では区別できない。ただし T5 の分散は非常に大きく(範囲 $0.29$)、seed によって対照群並みの精度から学習可能な絶対位置埋め込みに匹敵する精度まで大きくばらついた。

**対照群(位置エンコーディングなし)がチャンスレベルを大きく上回る理由**: 語彙サイズ $6$ に対するチャンスレベルの精度は約 $1/6 \approx 0.167$ であるのに対し、対照群は $0.634$ に達している。これは 3.1 節で述べた通り、本タスクが因果マスク付き decoder-only 構成を用いているためである。因果マスクの下では位置 $m$ の Query が参照できる Key の集合が $\{0, \dots, m\}$ に制限され、この非対称性自体が暗黙の位置情報として働く。加えて、「直前に出た同一トークンの次を見る」という内容ベースの induction 機構(位置エンコーディングを必要としない)も一定の精度に寄与しうる。6.2 節冒頭で述べた通り、語彙サイズを $6$ と小さくしたことで induction 機構単独ではタスクを解けないようにしているが、部分的な寄与までは排除できていない。Haviv et al. [7] が報告する「位置エンコーディングを持たない因果的言語モデルも位置情報を獲得する」という知見と整合する結果である。

**実験 B-2(学習長超、3.4・3.7・3.8 節との対応)**:

| 条件 | L=24 | L=32 | L=48 | L=64($4\times$) |
|---|---|---|---|---|
| なし | 0.478 [0.448, 0.501] | 0.390 [0.352, 0.424] | 0.308 [0.284, 0.342] | 0.266 [0.246, 0.298] |
| 正弦波方式 | 0.221 [0.187, 0.248] | 0.238 [0.214, 0.269] | 0.242 [0.230, 0.254] | 0.213 [0.203, 0.225] |
| 学習可能な絶対位置埋め込み | nan | nan | nan | nan |
| Shaw et al. 方式 | 0.891 [0.798, 0.947] | 0.813 [0.674, 0.904] | 0.642 [0.489, 0.756] | 0.505 [0.392, 0.595] |
| T5 | 0.376 [0.344, 0.424] | 0.451 [0.269, 0.631] | 0.275 [0.258, 0.286] | 0.277 [0.229, 0.323] |
| ALiBi | 0.920 [0.913, 0.925] | 0.855 [0.849, 0.863] | 0.711 [0.692, 0.722] | 0.603 [0.578, 0.615] |
| RoPE | 0.330 [0.280, 0.367] | 0.311 [0.299, 0.329] | 0.173 [0.168, 0.177] | 0.167 [0.163, 0.175] |

学習可能な絶対位置埋め込みは、3.4 節で述べた通り学習時の`max_len`($=2L_{\max}+1=33$)を超える位置を要求すると`ValueError`が送出され、評価結果が全域で`nan`になった。これは「性能が劣化する」のではなく「原理的にそもそも動作しない」という、3.4 節の議論をそのまま裏付ける結果である。

実験前の想定では「RoPE・正弦波方式は $L=48$・$L=64$ あたりで対照群を下回る」という緩やかな劣化を見込んでいたが、$3$ seed の実測値はこれよりも厳しい結果を示した。**正弦波方式・RoPE はいずれも、$1.5$ 倍長($L=24$)の時点ですでに対照群を下回り、比較した $4$ つの外挿長すべてで対照群を下回り続けた**。いずれの外挿長でも $3$ seed の最小・最大の範囲に重なりはない($L=24$: 正弦波方式 $0.221$ / RoPE $0.330$ に対し対照群 $0.478$、$L=32$: $0.238$ / $0.311$ に対し $0.390$、$L=48$: $0.242$ / $0.173$ に対し $0.308$、$L=64$: $0.213$ / $0.167$ に対し $0.266$)。T5 も $L=24$ では対照群を下回り(T5 $0.376$ に対し対照群 $0.478$、重なりなし)、$L=32$〜$L=64$ では対照群との範囲が重なり区別できない水準まで落ち込んだ。

この結果は、3.1 節で追加した議論と接続すると解釈しやすい。因果マスクだけが与える暗黙の位置情報(対照群が使っている情報)は、系列長によらず一貫した性質(「位置 $m$ は $m+1$ 個の Key しか参照できない」)であり、学習長を超えても意味を失わない。一方、正弦波方式・RoPE・T5 が学習時に学習した位置パターンは学習長の範囲でのみ意味を持つよう最適化されており、学習長を超えた位置に対しては **学習時に見たことのない、意味をなさない信号** を Attention に持ち込むことになる。この意味をなさない信号が、因果マスク由来の暗黙の位置情報だけに頼るよりも有害に働きうる、というのが今回の結果である。

対照的に、**ALiBi(0.603 @ $L=64$)・Shaw et al. 方式(0.505 @ $L=64$)は、比較した $7$ 条件の中で最も緩やかな劣化を示し、全外挿長で対照群を明確に上回り続けた**。ただし ALiBi と Shaw et al. 方式の範囲は各長さで重なっており(例えば $L=64$: ALiBi $[0.578, 0.615]$ に対し Shaw et al. 方式 $[0.392, 0.595]$)、$3$ seed では両者の優劣を確定的には言えない。ALiBi は Shaw et al. 方式より分散が小さく安定して高い精度を維持した点が目立つが、Shaw et al. 方式も $k_{\text{clip}}=16$ による相対距離のクリップ(3.5 節)のおかげで、クリップ範囲を超えた遠距離では新しい情報が増えないため、対照群を上回る精度を保った。

これは、3.7 節で述べた通り **ALiBi が外挿を明示的な設計目標としている手法である**([5] のタイトル "Train Short, Test Long" の通り)ことの帰結であり、RoPE が素の状態では学習長を大きく超えると性能が急激に劣化しやすいという既知の傾向と整合する。**これは「ALiBi の方が優れた手法である」ことを意味しない** — 学習長内の表現力(実験 B-1)では RoPE・正弦波方式が ALiBi を上回っており、測っている軸が異なるためである(7 節でこの点を統合的に考察する)。

**T5 の挙動について**: 5.3 節の`T5RelativePositionBias`(`num_buckets=32`・`max_distance=64`・`bidirectional=False`)を用いた補足分析(6.2 節末尾)によると、学習中に現れる最大の相対距離は $31$ であり、これはバケット $23$ に対応する。すなわち、バケット $24$〜$31$(全 $32$ バケット中 $8$ 個、$25\%$)は **学習中に一度も勾配を受けていない**。評価時 $L=64$ では相対距離が最大 $127$ に達し、バケット $31$(学習中に一度も学習されていないオーバーフローバケット)に落ちる。$L=24$(相対距離最大 $47$)の時点ですでに相対距離 $32$ 以上でバケット $24$ 以降(未学習領域)に入る。これが、T5 が RoPE・正弦波方式と同様に外挿性能が低い(むしろ $L=24$ では対照群も下回る)ことの具体的な要因だと考えられる。

### 6.3 実験 C: Attention 重みの可視化

正弦波方式・ALiBi・RoPE(および対照群)について、copy task が実際に「参照すべき位置に注目できているか」を定量指標で確認したうえで、代表例をヒートマップで可視化する。学習長内($L=16$)と学習長を超える系列($L=32$、学習長の 2 倍)の両方で評価し、外挿時に何が壊れるのかを確認する。モデルは 2 層(`N_LAYERS_B`)・ヘッド数 $4$(`N_HEADS_B`)であり、指標は特定の層・ヘッドに決め打ちせず **全層・全ヘッド** を対象に計算する(理由は後述)。

**参照すべき位置のオフセットについて**:`make_copy_batch`(6.2 節)が構成する系列は`[x_1, ..., x_L, SEP, x_1, ..., x_L]`であり、`input_ids`はこの系列の末尾 1 トークンを除いたもの(長さ $2L$、0-indexed で位置 $0$〜$2L-1$)である。位置 $L$ が`SEP`、位置 $L+1+j$($j=0,\dots,L-1$)が 2 回目の $x_{j+1}$ に対応する。後半の位置 $m$($m \ge L$)の次のトークンを予測するための正解は`target_ids[m] = full_sequence[m+1]`であり、$m=L+j$ のとき`full_sequence[L+j+1] = x_{j+1} = \text{full\_sequence}[j]`(前半の位置 $j = m - L$ の値)と一致する。したがって、位置 $m$ が参照すべき前半の位置は $m - L$ である(以下の指標計算はこの値を用いる)。


```python
def compute_chance_level(seq_len_l: int, device: str = "cpu") -> float:
    """因果マスク下で一様な Attention 重みを仮定した場合のチャンスレベルを計算する。

    位置 m(m = L, ..., 2L-1)は m+1 個の Key(位置 0..m)に一様に注意を割り振ると
    仮定すると、参照位置 m-L への重みの期待値は 1/(m+1) である。指標は後半の
    全位置で平均するため、チャンスレベルは (1/L) * sum_{m=L}^{2L-1} 1/(m+1)。
    argmax が参照位置と一致する確率も、一様分布上ではどの位置も等確率で
    argmax になりうるとみなせるため、同じ 1/(m+1) の平均に一致する。
    """
    m_positions = torch.arange(seq_len_l, 2 * seq_len_l, device=device)
    return (1.0 / (m_positions + 1).float()).mean().item()


def compute_attention_concentration_all_heads(
    model: nn.Module, seq_len_l: int, device: str = "cpu", n_samples: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """全層・全ヘッドについて、参照位置への重みと argmax 一致率を計算する。

    Returns:
        (weight_at_ref, argmax_match) のタプル。いずれも形状 (n_layers, h) の
        Tensor(n_samples 個のバッチ・後半全位置で平均した値)。
    """
    model.eval()
    weight_sum: torch.Tensor | None = None
    match_sum: torch.Tensor | None = None
    with torch.no_grad():
        for _ in range(n_samples):
            input_ids, _, _ = make_copy_batch(1, seq_len_l, device)
            _, attn_weights_per_layer = model(input_ids)
            m_positions = torch.arange(seq_len_l, 2 * seq_len_l, device=device)
            ref_positions = m_positions - seq_len_l

            layer_weights, layer_matches = [], []
            for attn_weights in attn_weights_per_layer:
                attn = attn_weights[0]  # (h, S, S)
                w = attn[:, m_positions, ref_positions]  # (h, len(m_positions))
                layer_weights.append(w.mean(dim=-1))  # (h,)
                argmax_pred = attn[:, m_positions, :].argmax(dim=-1)  # (h, len(m_positions))
                match = (argmax_pred == ref_positions[None, :]).float().mean(dim=-1)  # (h,)
                layer_matches.append(match)

            layer_weights_t = torch.stack(layer_weights)  # (n_layers, h)
            layer_matches_t = torch.stack(layer_matches)
            if weight_sum is None:
                weight_sum, match_sum = layer_weights_t, layer_matches_t
            else:
                weight_sum = weight_sum + layer_weights_t
                match_sum = match_sum + layer_matches_t
    return weight_sum / n_samples, match_sum / n_samples


METRIC_CONDITIONS_C = ["none", "sinusoidal", "alibi", "rope"]
concentration_full = {}

for name in METRIC_CONDITIONS_C:
    for l_vis in (L_MAX_TRAIN, 2 * L_MAX_TRAIN):
        weight_per_seed, match_per_seed = [], []
        best_layer_per_seed, best_head_per_seed = [], []
        for seed in SEEDS_B:
            torch.manual_seed(seed)
            w, m = compute_attention_concentration_all_heads(
                trained_models_b[name][seed], l_vis, DEVICE
            )
            weight_per_seed.append(w)
            match_per_seed.append(m)
            best_layer, best_head = divmod(w.argmax().item(), w.size(1))
            best_layer_per_seed.append(best_layer)
            best_head_per_seed.append(best_head)
        concentration_full[(name, l_vis)] = {
            "weight_per_seed": weight_per_seed,  # 各 seed: (n_layers, h)
            "match_per_seed": match_per_seed,
            "best_layer_per_seed": best_layer_per_seed,  # 各 seed: 重み最大の層
            "best_head_per_seed": best_head_per_seed,  # 各 seed: 重み最大のヘッド
        }

# 表 1(主表): 条件 x 系列長ごとに、重み(weight_at_ref)が最大となる (層, ヘッド) を
# seed ごとに求め、その (層, ヘッド) における重みと argmax 一致率の両方を集計する。
# 2 つの指標を別々に最大化すると異なる (層, ヘッド) 由来の値が並びうるため、
# 同一の (層, ヘッド) の値を揃えて表にする。
print("=== 表 1: 重みが最大の (層, ヘッド) における重み・argmax 一致率(seed 平均[最小,最大]) ===")
table1_rows = {}
for name in METRIC_CONDITIONS_C:
    for l_vis in (L_MAX_TRAIN, 2 * L_MAX_TRAIN):
        r = concentration_full[(name, l_vis)]
        weight_at_best_per_seed = []
        match_at_best_per_seed = []
        for seed_idx in range(len(SEEDS_B)):
            bl, bh = r["best_layer_per_seed"][seed_idx], r["best_head_per_seed"][seed_idx]
            weight_at_best_per_seed.append(r["weight_per_seed"][seed_idx][bl, bh].item())
            match_at_best_per_seed.append(r["match_per_seed"][seed_idx][bl, bh].item())
        chance = compute_chance_level(l_vis)
        table1_rows[(name, l_vis)] = {
            "weight_mean": sum(weight_at_best_per_seed) / len(weight_at_best_per_seed),
            "weight_min": min(weight_at_best_per_seed),
            "weight_max": max(weight_at_best_per_seed),
            "match_mean": sum(match_at_best_per_seed) / len(match_at_best_per_seed),
            "match_min": min(match_at_best_per_seed),
            "match_max": max(match_at_best_per_seed),
            "chance": chance,
        }
        t = table1_rows[(name, l_vis)]
        w_str = f"{t['weight_mean']:.4f}[{t['weight_min']:.4f},{t['weight_max']:.4f}]"
        m_str = f"{t['match_mean']:.4f}[{t['match_min']:.4f},{t['match_max']:.4f}]"
        label = f"{CONDITION_LABELS[name]:16s} L={l_vis:3d}"
        print(f"{label}  weight={w_str}  argmax_match={m_str}  chance={chance:.4f}")

# 表 2(補助表): 条件 x 系列長ごとの層・ヘッドごとの重み(seed 平均)
print("\n=== 表 2: 層・ヘッドごとの参照位置への重み(seed 平均) ===")
table2_rows = {}
for name in METRIC_CONDITIONS_C:
    for l_vis in (L_MAX_TRAIN, 2 * L_MAX_TRAIN):
        r = concentration_full[(name, l_vis)]
        weight_layerhead_mean = torch.stack(r["weight_per_seed"]).mean(dim=0)  # (n_layers, h)
        table2_rows[(name, l_vis)] = weight_layerhead_mean
        label = f"{CONDITION_LABELS[name]:16s} L={l_vis:3d}"
        for layer_idx in range(weight_layerhead_mean.size(0)):
            head_values = ", ".join(
                f"h{h_idx}={v:.4f}"
                for h_idx, v in enumerate(weight_layerhead_mean[layer_idx].tolist())
            )
            print(f"{label}  layer={layer_idx}  {head_values}")

# 表 3(補足): seed ごとに重みが最大となる (層, ヘッド) と、比較用の最終層 head 0 の値。
# 「copy 機構が載る (層, ヘッド) が seed によって異なるか」を直接検証するための参考データ。
print("\n=== 表 3: seed ごとの最大値 (層, ヘッド) と最終層 head 0 の値 ===")
last_layer_idx = N_LAYERS_B - 1
for name in METRIC_CONDITIONS_C:
    for l_vis in (L_MAX_TRAIN, 2 * L_MAX_TRAIN):
        r = concentration_full[(name, l_vis)]
        label = f"{CONDITION_LABELS[name]:16s} L={l_vis:3d}"
        for seed_idx, seed in enumerate(SEEDS_B):
            bl, bh = r["best_layer_per_seed"][seed_idx], r["best_head_per_seed"][seed_idx]
            best_val = r["weight_per_seed"][seed_idx][bl, bh].item()
            head0_val = r["weight_per_seed"][seed_idx][last_layer_idx, 0].item()
            best_str = f"best=(layer={bl},head={bh})={best_val:.4f}"
            head0_str = f"head0(layer={last_layer_idx})={head0_val:.4f}"
            print(f"{label}  seed={seed}  {best_str}  {head0_str}")
```

    === 表 1: 重みが最大の (層, ヘッド) における重み・argmax 一致率(seed 平均[最小,最大]) ===
    None (baseline)  L= 16  weight=0.1553[0.1410,0.1767]  argmax_match=0.2500[0.2188,0.2969]  chance=0.0424
    None (baseline)  L= 32  weight=0.0559[0.0551,0.0566]  argmax_match=0.0977[0.0703,0.1250]  chance=0.0214
    Sinusoidal       L= 16  weight=0.5760[0.5529,0.6058]  argmax_match=0.9010[0.8594,0.9375]  chance=0.0424
    Sinusoidal       L= 32  weight=0.0583[0.0434,0.0674]  argmax_match=0.0859[0.0625,0.1055]  chance=0.0214
    ALiBi            L= 16  weight=0.5971[0.5725,0.6222]  argmax_match=0.8594[0.8438,0.8906]  chance=0.0424
    ALiBi            L= 32  weight=0.4815[0.4739,0.4945]  argmax_match=0.8307[0.8203,0.8477]  chance=0.0214
    RoPE             L= 16  weight=0.7018[0.6635,0.7396]  argmax_match=0.9193[0.8984,0.9453]  chance=0.0424
    RoPE             L= 32  weight=0.1553[0.1000,0.2132]  argmax_match=0.2161[0.1406,0.2891]  chance=0.0214
    
    === 表 2: 層・ヘッドごとの参照位置への重み(seed 平均) ===
    None (baseline)  L= 16  layer=0  h0=0.0389, h1=0.0378, h2=0.0364, h3=0.0415
    None (baseline)  L= 16  layer=1  h0=0.0836, h1=0.1553, h2=0.0392, h3=0.0633
    None (baseline)  L= 32  layer=0  h0=0.0204, h1=0.0199, h2=0.0192, h3=0.0214
    None (baseline)  L= 32  layer=1  h0=0.0364, h1=0.0559, h2=0.0224, h3=0.0280
    Sinusoidal       L= 16  layer=0  h0=0.0278, h1=0.0413, h2=0.0289, h3=0.0282
    Sinusoidal       L= 16  layer=1  h0=0.2819, h1=0.4162, h2=0.2218, h3=0.0888
    Sinusoidal       L= 32  layer=0  h0=0.0223, h1=0.0257, h2=0.0238, h3=0.0253
    Sinusoidal       L= 32  layer=1  h0=0.0338, h1=0.0510, h2=0.0545, h3=0.0361
    ALiBi            L= 16  layer=0  h0=0.0040, h1=0.0265, h2=0.0341, h3=0.0397
    ALiBi            L= 16  layer=1  h0=0.0138, h1=0.2730, h2=0.3427, h3=0.4546
    ALiBi            L= 32  layer=0  h0=0.0001, h1=0.0083, h2=0.0164, h3=0.0198
    ALiBi            L= 32  layer=1  h0=0.0003, h1=0.1482, h2=0.2510, h3=0.3588
    RoPE             L= 16  layer=0  h0=0.0170, h1=0.0213, h2=0.0571, h3=0.0252
    RoPE             L= 16  layer=1  h0=0.4930, h1=0.6784, h2=0.5339, h3=0.2617
    RoPE             L= 32  layer=0  h0=0.0037, h1=0.0089, h2=0.0090, h3=0.0037
    RoPE             L= 32  layer=1  h0=0.0851, h1=0.1164, h2=0.1004, h3=0.0473
    
    === 表 3: seed ごとの最大値 (層, ヘッド) と最終層 head 0 の値 ===
    None (baseline)  L= 16  seed=0  best=(layer=1,head=1)=0.1482  head0(layer=1)=0.0791
    None (baseline)  L= 16  seed=1  best=(layer=1,head=1)=0.1410  head0(layer=1)=0.1056
    None (baseline)  L= 16  seed=2  best=(layer=1,head=1)=0.1767  head0(layer=1)=0.0662
    None (baseline)  L= 32  seed=0  best=(layer=1,head=1)=0.0566  head0(layer=1)=0.0338
    None (baseline)  L= 32  seed=1  best=(layer=1,head=1)=0.0551  head0(layer=1)=0.0383
    None (baseline)  L= 32  seed=2  best=(layer=1,head=1)=0.0562  head0(layer=1)=0.0370
    Sinusoidal       L= 16  seed=0  best=(layer=1,head=0)=0.5691  head0(layer=1)=0.5691
    Sinusoidal       L= 16  seed=1  best=(layer=1,head=1)=0.6058  head0(layer=1)=0.1920
    Sinusoidal       L= 16  seed=2  best=(layer=1,head=1)=0.5529  head0(layer=1)=0.0845
    Sinusoidal       L= 32  seed=0  best=(layer=1,head=2)=0.0434  head0(layer=1)=0.0433
    Sinusoidal       L= 32  seed=1  best=(layer=1,head=1)=0.0674  head0(layer=1)=0.0285
    Sinusoidal       L= 32  seed=2  best=(layer=1,head=2)=0.0642  head0(layer=1)=0.0297
    ALiBi            L= 16  seed=0  best=(layer=1,head=3)=0.5725  head0(layer=1)=0.0107
    ALiBi            L= 16  seed=1  best=(layer=1,head=2)=0.6222  head0(layer=1)=0.0140
    ALiBi            L= 16  seed=2  best=(layer=1,head=3)=0.5967  head0(layer=1)=0.0167
    ALiBi            L= 32  seed=0  best=(layer=1,head=3)=0.4761  head0(layer=1)=0.0003
    ALiBi            L= 32  seed=1  best=(layer=1,head=2)=0.4945  head0(layer=1)=0.0003
    ALiBi            L= 32  seed=2  best=(layer=1,head=3)=0.4739  head0(layer=1)=0.0003
    RoPE             L= 16  seed=0  best=(layer=1,head=1)=0.7025  head0(layer=1)=0.0993
    RoPE             L= 16  seed=1  best=(layer=1,head=0)=0.7396  head0(layer=1)=0.7396
    RoPE             L= 16  seed=2  best=(layer=1,head=2)=0.6635  head0(layer=1)=0.6403
    RoPE             L= 32  seed=0  best=(layer=1,head=1)=0.1000  head0(layer=1)=0.0427
    RoPE             L= 32  seed=1  best=(layer=1,head=1)=0.2132  head0(layer=1)=0.0618
    RoPE             L= 32  seed=2  best=(layer=1,head=2)=0.1528  head0(layer=1)=0.1508



```python
def get_attention_weights(
    model: nn.Module, seq_len_l: int, layer_idx: int, head_idx: int, device: str = "cpu"
) -> torch.Tensor:
    """1 サンプルぶんの入力に対する、指定した層・ヘッドの Attention 重みを取得する。"""
    model.eval()
    input_ids, _, _ = make_copy_batch(1, seq_len_l, device)
    with torch.no_grad():
        _, attn_weights_per_layer = model(input_ids)
    return attn_weights_per_layer[layer_idx][0, head_idx]  # (S, S)


VIS_CONDITIONS_C = ["sinusoidal", "alibi", "rope"]
fig, axes = plt.subplots(2, len(VIS_CONDITIONS_C), figsize=(5.0 * len(VIS_CONDITIONS_C), 9.0))

# 可視化は代表として先頭の seed(SEEDS_B[0])で学習したモデルを使い、条件・系列長ごとに
# 表 2 の元データ(weight_per_seed[0] が SEEDS_B[0] に対応)から最も集中度が高い
# (層, ヘッド)を選んで描く。
for col, name in enumerate(VIS_CONDITIONS_C):
    model_c = trained_models_b[name][SEEDS_B[0]]
    for row, l_vis in enumerate((L_MAX_TRAIN, 2 * L_MAX_TRAIN)):
        r = concentration_full[(name, l_vis)]
        best_layer, best_head = r["best_layer_per_seed"][0], r["best_head_per_seed"][0]
        best_value = r["weight_per_seed"][0][best_layer, best_head].item()

        torch.manual_seed(SEED)
        attn_c = get_attention_weights(model_c, l_vis, best_layer, best_head, DEVICE)
        length_note = "  = train length" if l_vis == L_MAX_TRAIN else "  = 2x train length"
        title = (
            f"{CONDITION_LABELS[name]}  (L={l_vis}{length_note})\n"
            f"layer={best_layer}, head={best_head} (weight_at_ref={best_value:.3f})"
        )
        plot_attention_heatmap(
            attn_c, title=title, ax=axes[row, col], colorbar=(col == len(VIS_CONDITIONS_C) - 1)
        )

plt.tight_layout()
plt.show()
```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/003_positional_encoding_rope/output_55_0.png)
    


#### 実験 C の結果・考察

**表 1(重みが最大の (層, ヘッド) における重み・argmax 一致率、6.3 節の実行結果より、$3$ seed の平均 [最小, 最大])**: $2$ つの指標を別々に最大化すると異なる (層, ヘッド) 由来の値が並びうるため、参照位置への重みが最大となる (層, ヘッド) を seed ごとに特定し、その同一の (層, ヘッド) における重みと argmax 一致率を並べている。

| 条件 | $L$ | 参照位置への重み(最大の (層, ヘッド)) | argmax 一致率(同一の (層, ヘッド)) | チャンスレベル |
|---|---:|---|---|---:|
| なし(対照群) | 16 | 0.1553 [0.1410, 0.1767] | 0.2500 [0.2188, 0.2969] | 0.0424 |
| なし(対照群) | 32 | 0.0559 [0.0551, 0.0566] | 0.0977 [0.0703, 0.1250] | 0.0214 |
| 正弦波方式 | 16 | 0.5760 [0.5529, 0.6058] | 0.9010 [0.8594, 0.9375] | 0.0424 |
| 正弦波方式 | 32 | 0.0583 [0.0434, 0.0674] | 0.0859 [0.0625, 0.1055] | 0.0214 |
| ALiBi | 16 | 0.5971 [0.5725, 0.6222] | 0.8594 [0.8438, 0.8906] | 0.0424 |
| ALiBi | 32 | 0.4815 [0.4739, 0.4945] | 0.8307 [0.8203, 0.8477] | 0.0214 |
| RoPE | 16 | 0.7018 [0.6635, 0.7396] | 0.9193 [0.8984, 0.9453] | 0.0424 |
| RoPE | 32 | 0.1553 [0.1000, 0.2132] | 0.2161 [0.1406, 0.2891] | 0.0214 |

チャンスレベルは、位置 $m$ が因果マスクの下で参照可能な $m+1$ 個の Key に一様な重みを置くと仮定した場合の値であり、コード内で $\mathrm{chance}(L) = (1/L)\sum_{m=L}^{2L-1} 1/(m+1)$ として計算した($L=16$ で $0.0424$、$L=32$ で $0.0214$。一様分布からの argmax は各位置が等確率で選ばれるとみなせるため、同じ値をチャンスレベルとして用いる)。

**表 2(層・ヘッドごとの参照位置への重み、$3$ seed 平均)から分かること**: いずれの条件でも 1 層目(`layer=0`)の値はチャンスレベル前後にとどまり(最も高い RoPE の`layer=0`head 2 でも $0.0571$、チャンスレベル $0.0424$ の約 $1.35$ 倍)、2 層目(`layer=1`、最終層)に copy に関わる集中が現れる。したがって、この copy task を解く上で位置情報を直接使う処理は主に最終層で行われており、1 層目は参照位置への直接的な集中を担っていない(1 層目が実際に何を行っているかは本実験では測っていない)。

**head 0 のみでは指標にならないことについて**: 正弦波方式・RoPE はいずれも精度が seed 間でほぼ一定(6.2 節末尾の表より $L=16$ で正弦波方式 $[0.9822, 0.9907]$・RoPE $[0.9829, 0.9917]$)であるにもかかわらず、表 3(seed ごとの最大値 (層, ヘッド))が示す通り、$L=16$ で重みが最大となる (層, ヘッド) は seed によって実際に異なる。RoPE は seed 0 が`(layer=1, head=1)`、seed 1 が`(layer=1, head=0)`、seed 2 が`(layer=1, head=2)`と、$3$ seed すべてで異なるヘッドが最大値を取った。正弦波方式も seed 0 は`(layer=1, head=0)`である一方、seed 1・seed 2 は`(layer=1, head=1)`であり、必ずしも同一ではない。この結果として、最終層 head 0 単体の値も $L=16$ で正弦波方式が $[0.0845, 0.5691]$、RoPE が $[0.0993, 0.7396]$ と seed 間で大きく振れる(表 3)。一方、対照群(`none`)は $L=16$・$L=32$ のいずれでも $3$ seed すべてが`(layer=1, head=1)`で一致しており、seed への依存は条件によって程度が異なる(明示的な位置情報を持たない対照群では一貫し、正弦波方式・RoPE では変動が大きい)。いずれにせよ、copy 機構がどの (層, ヘッド) に載るかは条件の性質だけで決まるわけではなく **seed(初期化・学習の過程)にも依存しうる**。この理由から、以降は表 1(seed ごとに重み最大の (層, ヘッド) を特定した上での集計)を主指標として比較する。

**正弦波方式・RoPE**: 学習長内($L=16$)では、表 1 の最大値(正弦波方式 $0.576$・RoPE $0.702$)・argmax 一致率(正弦波方式 $0.901$・RoPE $0.919$)ともにチャンスレベル($0.0424$)の $13$〜$22$ 倍に達し、後半の各位置が対応する前半の位置に明確に集中して注目していることが確認できた(ヒートマップ上でも、SEP からのオフセット分だけずれた明るい対角線として観察できる)。学習長の 2 倍($L=32$)では、正弦波方式は重み $0.058$(チャンスレベル $0.0214$ の約 $2.7$ 倍)・一致率 $0.086$(約 $4.0$ 倍)、RoPE は重み $0.155$(約 $7.3$ 倍)・一致率 $0.216$(約 $10.1$ 倍)まで低下した。いずれもチャンスレベルを数倍〜十倍程度上回っており、参照位置への集中が完全に失われたわけではないが、$L=16$ 時点の水準(チャンスレベルの十数倍以上)からは大きく後退している。RoPE の方が正弦波方式よりチャンスレベルに対する倍率を高く保っており、これは実験 B-2 で観測した精度の関係(6.2 節末尾の表より、$L=32$ での $3$ seed 平均精度は正弦波方式 $0.238$・RoPE $0.311$、RoPE がわずかに高い)と整合する。

**ALiBi**: ALiBi は学習長内($L=16$)で重み $0.597$・一致率 $0.859$(チャンスレベルの約 $14$〜$20$ 倍)に達し、正弦波方式・RoPE と同水準の集中を示した。学習長の 2 倍($L=32$)でも重み $0.482$・一致率 $0.831$(チャンスレベルの約 $22$〜$39$ 倍)とほとんど低下しておらず、正弦波方式・RoPE とは対照的に集中がほぼ保たれた。これは実験 B-2 で観測した ALiBi の緩やかな精度低下($L=16$ で $3$ seed 平均 $0.967$、$L=32$ で $0.855$)と整合する。

**ALiBi の担当ヘッドの特定**: 表 2 より、ALiBi の`layer=1`における各ヘッドの重み($3$ seed 平均)は $L=16$ で head 0: $0.0138$、head 1: $0.2730$、head 2: $0.3427$、head 3: $0.4546$ であり、$L=32$ でも head 0: $0.0003$、head 1: $0.1482$、head 2: $0.2510$、head 3: $0.3588$ と、傾き $m_h$(3.7 節、$[0.25, 0.0625, 0.015625, 0.00390625]$、距離 $16$ でのバイアスはそれぞれ $-4.0$、$-1.0$、$-0.25$、$-0.0625$)が小さいヘッドほど値が単調に大きい。

ただし、表 2 は各ヘッドの値を $3$ seed で平均したものであり、seed ごとにどのヘッドが最大となるかは表 2 からは読み取れない。表 3 より、$L=16$・$L=32$ のいずれでも、重みが最大となるヘッドは head 3(seed 0・seed 2)と head 2(seed 1)であり、$3$ seed 中 $2$ seed が head 3、$1$ seed が head 2 であって、**head 3 単体に断定できるわけではない**。これは前段(「head 0 のみでは指標にならないことについて」)で述べた「copy 機構がどの (層, ヘッド) に載るかは seed に依存しうる」という傾向が ALiBi にも当てはまることを示している。ただし、傾きが大きい head 0・head 1 が最大となった seed は $3$ seed × $2$ 系列長のいずれの組み合わせにも $1$ つもなく、最大となるのは常に傾きが小さい側の $2$ ヘッド(head 2・head 3)に限られている。したがって、「傾きが小さいヘッドが長距離の参照を担当する」という 3.7 節の設計上の帰結は実測で裏付けられているが、そのうちどちらのヘッドが実際に最大となるかは seed に依存する、という粒度で述べるのが実測に即している。

表 2 の実測値を head ごとにチャンスレベル($0.0424$)との比で見ると、head 0 は $0.0138$ でチャンスレベルの約 $0.33$ 倍と **下回っている** のに対し、head 1 は $0.2730$(約 $6.4$ 倍)、head 2 は $0.3427$(約 $8.1$ 倍)、head 3 は $0.4546$(約 $10.7$ 倍)といずれもチャンスレベルを明確に上回り、バイアスが小さいヘッドほど重みが段階的に大きくなっている。すなわち、ヘッドは参照位置に「集中する / しない」の二分ではなく、傾き $m_h$ の大きさに応じて寄与の度合いが連続的に変化しており、距離 $16$ に対するバイアスが $-4.0$ と特に大きい head 0 だけが、重みをチャンスレベル未満まで押し下げられている。これは学習の結果ではなく、ALiBi の傾き $m_h$ が **学習可能パラメータを一切持たない固定値**(3.7 節)であることによる構造的な帰結であり、head 0 だけが設計上そもそも長距離の参照に強いペナルティを受ける。

**対照群との比較**: 表 1 には対照群(明示的な位置エンコーディングを持たない`none`)の値も含めている。$L=16$ での対照群は重み $0.1553$・argmax 一致率 $0.2500$ であり、チャンスレベル $0.0424$ のそれぞれ約 $3.7$ 倍・約 $5.9$ 倍である。すなわち **明示的な位置エンコーディングを一切持たない場合でも、参照位置への集中はチャンスレベルを明確に上回る**。これは 3.1 節で述べた「因果マスク自体が暗黙の位置情報を与える」という議論の、Attention の挙動レベルでの定量的な裏付けであり、実験 B-1 で対照群の精度がチャンスレベル($\approx 0.167$)を大きく上回って $0.634$ に達したこと(6.2 節)とも整合する。対照群の値は、以下で他条件の集中度を評価する際の「明示的な位置情報なしで到達しうる水準」の目安として使う。

学習長の 2 倍($L=32$)において、正弦波方式(重み $0.0583$・一致率 $0.0859$)は対照群(重み $0.0559$・一致率 $0.0977$)と **ほぼ同水準** であり、argmax 一致率では対照群を **下回っている**。これは実験 B-2 で正弦波方式が全外挿長で対照群を下回った($L=32$ で $0.238$ vs $0.390$、6.2 節)ことと完全に整合する。3.1 節・7 節で述べている「学習時に見たことのない位置に対する意味をなさない明示的な位置信号は、因果マスク由来の暗黙の位置情報だけに頼るより有害に働きうる」という主張が、精度だけでなく **Attention の集中度そのもの** でも裏付けられたことになる。

一方、RoPE($L=32$ で重み $0.1553$・一致率 $0.2161$)は対照群(重み $0.0559$・一致率 $0.0977$)を明確に上回っているにもかかわらず、精度では対照群を下回る($0.311$ vs $0.390$、6.2 節)。すなわち **参照位置への集中度の高さと精度は単調に対応しない**。なぜ集中度が対照群を上回るのに精度が対照群を下回るのかについて、本実験の結果だけから断定はできない。本指標が全層・全ヘッドのうち重みが最大となる $1$ つの (層, ヘッド) しか見ていないこと、また copy には参照位置への注目だけでなく、参照した値を正しく出力に反映する処理(値の読み出し・変換)も必要であることが、可能性として考えられる。

**実験 B-2 の精度との対応**: 正弦波方式は学習長を超えると集中度が対照群とほぼ同水準まで低下し、RoPE は対照群を上回るもののチャンスレベルの数倍程度まで低下する(上記「対照群との比較」を参照)。いずれも学習長内で最も高い集中度に達していたことと対照的であり、実験 B-2 で観測した精度の急落(6.2 節末尾の表)と対応する。ALiBi は学習長内・学習長超のいずれでも高い集中度を維持しており、実験 B-2 で観測した緩やかな精度低下と対応する。これは、ALiBi がヘッドごとに異なる距離スケールを担当させる設計(3.7 節)により、学習長を超えた長い距離でも(遠距離担当のヘッドが)相対的に一貫したバイアスを与え続けられることの帰結だと考えられる。ただし RoPE の例が示す通り、集中度の高さは精度の高さを保証しない。

可視化に用いたモデルは代表として seed $0$ で学習したものであり、ヒートマップのタイトルには実際に選ばれた層・ヘッド番号と、その重みの値を明記している。上記の定量指標は $3$ seed の平均値である点に注意(表 3 が示す通り、個別の seed の値はこれと大きく異なりうる)。



## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/01_foundations/003_positional_encoding_rope.ipynb
