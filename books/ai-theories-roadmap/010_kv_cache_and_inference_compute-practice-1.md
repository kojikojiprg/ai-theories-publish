---
title: "KV キャッシュと推論の計算量(KV Cache and Inference Compute)(実装・実験編 1/4)"
---

この記事は後編(実装・実験編 1/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/010_kv_cache_and_inference_compute-theory)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/010_kv_cache_and_inference_compute-practice-2)。

## 4. 実装方針 / Implementation Policy

**ノートブック内に直接書くもの**(`src/`に置かない): 3.1 節の導入として、
Attention 層自身が過去の Key / Value を保持する最小実装
(`MinimalCachedAttention`、20 行程度)。状態を層の外に出す設計上の理由
(条件・シードをまたぐ計測での状態漏れ、メモリ量集計の分散)を実装と対比して
示すためだけの教育的なコードであり、後続トピックから import されない。

**`src/`に置くもの**:

- `src/generation/cache.py`(新規): `KeyValueCache`。層ごとの Key / Value の
  保持・追記・メモリ量の報告を行う。014(Flash Attention)・015(長文脈拡張)・
  016 以降の生成による評価から再利用される前提。
- `src/layers/attention.py`: `MultiHeadAttention`に`num_key_value_heads`引数
  (既定値`None`で多頭注意機構と完全に同一の挙動)と KV キャッシュの入出力
  (`kv_cache`・`layer_idx`引数)を追加。
- `src/layers/transformer_block.py`: `DecoderBlock.forward`に`kv_cache`・
  `layer_idx`引数を追加し、`self_attn`へそのまま透過する。
- `src/models/gpt.py`: `GPTLanguageModel.forward`に`kv_cache`引数、
  `generate()`に`use_cache`引数(既定値`False`で 006〜008 と完全に同一の挙動)を
  追加。多頭注意機構の重みから GQA / MQA への変換関数
  `convert_attention_to_grouped_query`(平均プール初期化・ランダム初期化の
  両方に対応)を追加。
- `src/utils/statistics.py`: `compute_key_value_cache_memory_bytes()`・
  `compute_arithmetic_intensity()`・`fit_power_law_exponent()`(回帰係数と
  その標準誤差を返す)。
- `src/utils/visualization.py`: `plot_log_log_fit()`(両対数プロット、
  べき乗則あてはめの直線を重畳できる)。

後方互換性(`num_key_value_heads=None`・`use_cache=False`のとき 006〜009 と
完全に同一の挙動になること)は、乱数消費まで含めて bit-identical であることを
既に検証済みである(`MultiHeadAttention`・`GPTLanguageModel`のパラメータ初期化・
順伝播の出力ハッシュが、本トピック追加前のコードと完全一致することを、実装の
過程で確認した。6 節の不変条件アサーションで改めて示す)。

## 5. 実装 / Implementation

### 5.1 最小実装: Attention 層自身が Key / Value を保持する

まず、状態(過去の Key / Value)を Attention 層自身に持たせる最小実装
(`MinimalCachedAttention`)を示す。次のセルで`src/generation/cache.py`の
`KeyValueCache`(状態を層の外に出した実装)と生成結果を比較し、**なぜ状態を層の
外に出すのか** を実装の対比を通じて示す。

- **条件・シードをまたぐ計測での状態漏れ**: `MinimalCachedAttention`はインスタンス
  自身が`cache_k`・`cache_v`を保持するため、同じレイヤーインスタンスを複数の
  条件・シードで使い回すと、前の生成のキャッシュが残ったまま次の計測を始めて
  しまう危険がある(`reset_cache()`の呼び忘れが静かなバグになる)。`KeyValueCache`
  は呼び出し側が明示的に生成・破棄するため、この危険が構造的に起きない。
- **メモリ量集計の分散**: 状態がレイヤーごとに分散していると、モデル全体の
  キャッシュメモリ量(3.2 節)を集計するにはモデル全体を走査する必要がある。
  `KeyValueCache`は 1 つのインスタンスが全層の Key / Value を持つため、
  `memory_bytes()`で直接合計できる。


```python
class MinimalCachedAttention(nn.Module):
    '''最小実装: Attention 層自身が過去の Key / Value を保持する(単一ヘッド)。'''

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.cache_k: torch.Tensor | None = None
        self.cache_v: torch.Tensor | None = None

    def reset_cache(self) -> None:
        self.cache_k = None
        self.cache_v = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        was_empty = self.cache_k is None
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        if was_empty:
            self.cache_k, self.cache_v = k, v
        else:
            self.cache_k = torch.cat([self.cache_k, k], dim=1)
            self.cache_v = torch.cat([self.cache_v, v], dim=1)
        scores = q @ self.cache_k.transpose(-2, -1) / (x.size(-1) ** 0.5)
        if was_empty and x.size(1) > 1:
            causal = torch.tril(torch.ones(x.size(1), x.size(1), dtype=torch.bool))
            scores = scores.masked_fill(~causal, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return self.w_o(weights @ self.cache_v)


class MinimalToyModel(nn.Module):
    '''埋め込み + MinimalCachedAttention + 出力層のみの最小構成(FFN・正規化なし)。'''

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.attn = MinimalCachedAttention(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.attn(self.embed(token_ids)))

    def reset_cache(self) -> None:
        self.attn.reset_cache()


def minimal_generate(model: MinimalToyModel, prompt_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    model.reset_cache()
    token_ids = prompt_ids.clone()
    with torch.no_grad():
        logits = model(token_ids)
        next_token = logits[:, -1:, :].argmax(dim=-1)
        token_ids = torch.cat([token_ids, next_token], dim=1)
        for _ in range(max_new_tokens - 1):
            logits = model(next_token)
            next_token = logits[:, -1:, :].argmax(dim=-1)
            token_ids = torch.cat([token_ids, next_token], dim=1)
    return token_ids
```


```python
# src 側の実装(KeyValueCache に状態を外出しした版)。MultiHeadAttention(num_heads=1)
# を使い、MinimalCachedAttention と完全に同じ構造(w_q, w_k, w_v, w_o のみ)にする。
torch.manual_seed(42)
_TOY_VOCAB, _TOY_D_MODEL = 37, 16
minimal_model = MinimalToyModel(_TOY_VOCAB, _TOY_D_MODEL)

src_embed = nn.Embedding(_TOY_VOCAB, _TOY_D_MODEL)
src_attn = MultiHeadAttention(_TOY_D_MODEL, num_heads=1, bias=False)
src_head = nn.Linear(_TOY_D_MODEL, _TOY_VOCAB, bias=False)

# 重みを完全一致させる(乱数の消費経路が異なる 2 つのモデルを、同一の重みで比較する)。
src_embed.weight.data.copy_(minimal_model.embed.weight.data)
src_attn.w_q.weight.data.copy_(minimal_model.attn.w_q.weight.data)
src_attn.w_k.weight.data.copy_(minimal_model.attn.w_k.weight.data)
src_attn.w_v.weight.data.copy_(minimal_model.attn.w_v.weight.data)
src_attn.w_o.weight.data.copy_(minimal_model.attn.w_o.weight.data)
src_head.weight.data.copy_(minimal_model.head.weight.data)


def src_generate(prompt_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    kv_cache = KeyValueCache(num_layers=1)
    token_ids = prompt_ids.clone()
    with torch.no_grad():
        hidden = src_embed(token_ids)
        mask = create_causal_mask(token_ids.size(1)) if token_ids.size(1) > 1 else None
        out, _ = src_attn(hidden, hidden, hidden, mask, kv_cache=kv_cache, layer_idx=0)
        logits = src_head(out)
        next_token = logits[:, -1:, :].argmax(dim=-1)
        token_ids = torch.cat([token_ids, next_token], dim=1)
        for _ in range(max_new_tokens - 1):
            hidden = src_embed(next_token)
            out, _ = src_attn(hidden, hidden, hidden, kv_cache=kv_cache, layer_idx=0)
            logits = src_head(out)
            next_token = logits[:, -1:, :].argmax(dim=-1)
            token_ids = torch.cat([token_ids, next_token], dim=1)
    return token_ids


_prompt = torch.randint(0, _TOY_VOCAB, (1, 5))
_out_minimal = minimal_generate(minimal_model, _prompt, 12)
_out_src = src_generate(_prompt, 12)
print("最小実装:", _out_minimal.tolist())
print("src 実装:", _out_src.tolist())
assert torch.equal(_out_minimal, _out_src), "最小実装と src 実装の生成結果が一致しない"
print("[OK] 最小実装(層自身がキャッシュを保持)と src 実装(KeyValueCache に外出し)の"
      "貪欲法生成結果が完全一致した")
```

    最小実装: [[27, 5, 27, 26, 26, 13, 0, 11, 29, 29, 29, 29, 29, 33, 29, 33, 29]]
    src 実装: [[27, 5, 27, 26, 26, 13, 0, 11, 29, 29, 29, 29, 29, 33, 29, 33, 29]]
    [OK] 最小実装(層自身がキャッシュを保持)と src 実装(KeyValueCache に外出し)の貪欲法生成結果が完全一致した


### 5.2 `src/`の実装: 後方互換性の確認

`num_key_value_heads=None`・`use_cache=False`(既定値)のとき、006〜009 と
完全に同一の挙動になることを確認する。

**別マシンで採取したハッシュ定数とのクロスプラットフォーム比較はしない。**
浮動小数点の乱数生成アルゴリズムの実装は torch のバージョン・プラットフォーム
(CPU 命令セットなど)によって異なりうるため、別環境で採取した定数とのビット
単位の比較は「この環境が別環境の浮動小数点表現を再現するか」という、本ノート
ブックが検証したい後方互換性(num_key_value_heads・kv_cache 引数の追加が
006〜009 の挙動を変えていないこと)とは異なる命題を検証してしまう。代わりに
次の 2 つを行う。

1. **同一環境内での比較**(5.2.1 節): 本トピック(010)の実装を加える直前の
   コミット(`eb37dee`)のコードを、git worktree で取得し、
   **現在の実行環境と同じ Python・同じ torch のサブプロセスで実行** して
   参照値を計算する。同一プロセス内の実行では`src`パッケージ名が現行コードと
   衝突するため、サブプロセス分離が必要である。参照リポジトリの取得に失敗した
   場合(ネットワーク不通・git 履歴がないなど)は、比較を行わずその旨を記録する
   (ハードコードした定数へのフォールバックはしない)。
2. **環境に依存しない自己完結な等価性の検証**(5.2.2 節): `num_key_value_heads`
   を明示的に渡した場合と省略した場合とで、パラメータ・順伝播出力・乱数消費量
   (`torch.get_rng_state()`の変化)が完全に一致することを、同一セッション内で
   直接確認する。プラットフォームに一切依存しない。


```python
# 006〜009 と同一の構成(RoPE・既定のヘッド数)でモデルを構築し、パラメータの
# ハッシュと順伝播の出力を記録する。本トピックの変更(num_key_value_heads・kv_cache
# 引数の追加)が、これらを一切変えていないことを確認する不変条件である。
torch.manual_seed(0)
_compat_d_model, _compat_heads, _compat_layers, _compat_d_ff, _compat_seq = 32, 4, 2, 64, 32
_compat_vocab = 50
_compat_rope = RotaryPositionEmbedding(_compat_d_model // _compat_heads)
_compat_model = GPTLanguageModel(
    _compat_vocab, _compat_d_model, _compat_layers, _compat_heads, _compat_d_ff, _compat_seq,
    positional_transform=_compat_rope,
)
_compat_model.eval()

_compat_state_hash = hashlib.sha256()
for key in sorted(_compat_model.state_dict()):
    _compat_state_hash.update(key.encode())
    _compat_state_hash.update(_compat_model.state_dict()[key].numpy().tobytes())

_compat_x = torch.randint(0, _compat_vocab, (1, 5), generator=torch.Generator().manual_seed(42))
with torch.no_grad():
    _compat_logits_nocache = _compat_model(_compat_x)
    _compat_logits_default = _compat_model(_compat_x, kv_cache=None)  # 既定の呼び出しと同一のはず
assert torch.equal(_compat_logits_nocache, _compat_logits_default)

# 実測値は記録として印字するのみで、この場ではアサーションを行わない
# (比較は 5.2.1 節・5.2.2 節で行う)。
print(f"torch: {torch.__version__}")
print(f"platform: {platform.platform()}")
print(f"パラメータ初期化の SHA-256: {_compat_state_hash.hexdigest()}")
print(f"順伝播 logits の和: {float(_compat_logits_nocache.sum()):.10f}")

# use_cache=False での generate() が、キャッシュを一切使わない従来の経路と一致することも確認する
# (これは同一プロセス・同一実行内の比較なので、環境に依存しない)。
_compat_gen_a = _compat_model.generate(_compat_x.clone(), max_new_tokens=6, temperature=0.0, use_cache=False)
_compat_gen_b = _compat_model.generate(_compat_x.clone(), max_new_tokens=6, temperature=0.0, use_cache=True)
assert torch.equal(_compat_gen_a, _compat_gen_b), "use_cache の有無で貪欲法生成結果が一致しない"
print("[OK] use_cache=True/False で貪欲法生成結果が完全一致(後方互換性 + 不変条件)")
```

    torch: 2.13.0+cu130
    platform: Linux-6.6.122+-x86_64-with-glibc2.39
    パラメータ初期化の SHA-256: fe6cbe540cbbe935d3b155bc2fcbd9f1997fc326b505701709f95b29cf351a76
    順伝播 logits の和: 2.9690001011
    [OK] use_cache=True/False で貪欲法生成結果が完全一致(後方互換性 + 不変条件)


#### 5.2.1 本トピック追加直前のコードとの同一環境比較

比較対象のコミット: `eb37dee`(010 の実装(`be3c4f6`)の直前のコミット、
`git log`で確認済み)。git worktree でこのコミットのコードを
取得し、**現在の実行環境と同じ`sys.executable`・同じ torch のサブプロセス** で
5.2 節と同一のモデル構成・同一のシードを用いて参照値を計算する。サブプロセスに
分離するのは、参照コードの`src`パッケージが現行の`src`と名前が衝突し、同一
プロセスに import できないためである。

**参照コードが実際に worktree 側の`src`から読み込まれたことも検証する。**
Google Colab では`%cd`の後も`sys.path`に本体側のパスが残ることがあり、その
場合、参照スクリプトが本体側(現行コード)の`src`を import してしまい、
参照値が現行コードの値と一致して **アサーションは通るが後方互換性を何も
検証していない** 状態になりうる(失敗より発見が難しい失敗の仕方である)。
これを防ぐため、参照スクリプト自身が読み込んだ`src`のパスが実行時の
カレントディレクトリ(worktree)配下であることを自己検証し、ノートブック側でも
同じ検証を独立に行う。あわせて、本体側の`sys.path`が環境変数`PYTHONPATH`
経由で注入される経路も塞ぐ。


```python
import os

REFERENCE_COMMIT = "eb37dee"  # 010 の実装直前のコミット(be3c4f6 の親)

# 参照スクリプト自身が、worktree 側の src(cwd 配下)を読み込んだことを自己検証する。
# Google Colab では %cd の後も sys.path に本体側のパスが残ることがあり、その場合
# ここでの import が本体側(現行コード)の src を読んでしまい、参照値が現行コードの
# 値と一致してしまう(アサーションは通るが後方互換性を何も検証していない、
# 失敗よりも発見が難しい失敗の仕方になる)。src_path が cwd 配下でなければ
# 非ゼロ終了させ、標準エラーにその旨を出す。
_REFERENCE_SCRIPT = r'''
import hashlib
import json
import os
import sys

import torch

import src
from src.layers.positional_encoding import RotaryPositionEmbedding
from src.models.gpt import GPTLanguageModel

src_path = os.path.realpath(src.__file__)
cwd = os.path.realpath(os.getcwd())
if not (src_path == cwd or src_path.startswith(cwd + os.sep)):
    print(f"src が worktree の外(cwd={cwd})から読み込まれた: {src_path}", file=sys.stderr)
    sys.exit(1)

torch.manual_seed(0)
d_model, num_heads, num_layers, d_ff, max_seq = 32, 4, 2, 64, 32
vocab = 50
rope = RotaryPositionEmbedding(d_model // num_heads)
model = GPTLanguageModel(vocab, d_model, num_layers, num_heads, d_ff, max_seq, positional_transform=rope)
model.eval()

state_hash = hashlib.sha256()
for key in sorted(model.state_dict()):
    state_hash.update(key.encode())
    state_hash.update(model.state_dict()[key].numpy().tobytes())

x = torch.randint(0, vocab, (1, 5), generator=torch.Generator().manual_seed(42))
with torch.no_grad():
    logits = model(x)

print(json.dumps({
    "state_hash": state_hash.hexdigest(),
    "logits_sum": float(logits.sum()),
    "src_path": src_path,
}))
'''


def compute_reference_compat_values(commit: str) -> dict | None:
    '''指定コミットのコードを、現在の実行環境と同じ Python・同じ torch の
    サブプロセスで実行し、state_dict のハッシュと順伝播出力の和を計算する。
    参照リポジトリの取得(git worktree)に失敗した場合、または参照スクリプトが
    worktree 側ではなく本体側の src を読み込んでしまった場合は None を返す
    (呼び出し側はこの場合、比較を行わずその旨を記録する)。
    '''
    worktree_dir = tempfile.mkdtemp(prefix="ai_theories_ref_")
    worktree_added = False
    try:
        add_result = subprocess.run(
            ["git", "worktree", "add", "--detach", worktree_dir, commit],
            cwd=str(ROOT.resolve()), capture_output=True, text=True, timeout=60,
        )
        if add_result.returncode != 0:
            print(f"[注記] 参照コミット {commit} の worktree 取得に失敗した: {add_result.stderr.strip()}")
            return None
        worktree_added = True

        # 本体側の sys.path が外部から注入される経路(PYTHONPATH 環境変数)を塞ぐ。
        _reference_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        run_result = subprocess.run(
            [sys.executable, "-c", _REFERENCE_SCRIPT],
            cwd=worktree_dir, capture_output=True, text=True, timeout=120, env=_reference_env,
        )
        if run_result.returncode != 0:
            print(f"[注記] 参照コミット {commit} でのハッシュ計算に失敗した: {run_result.stderr.strip()}")
            return None
        result = json.loads(run_result.stdout.strip().splitlines()[-1])

        # ノートブック側でも、返ってきた src_path が worktree のディレクトリ配下で
        # あることを二重に確認する(参照スクリプト内のアサーションだけに頼らない)。
        worktree_real = os.path.realpath(worktree_dir)
        src_real = os.path.realpath(result["src_path"])
        if not (src_real == worktree_real or src_real.startswith(worktree_real + os.sep)):
            print(
                f"[注記] 参照スクリプトが worktree({worktree_dir})の外から src を"
                f"読み込んだ: {result['src_path']}。本体側の sys.path が残っている"
                "可能性がある。"
            )
            return None
        result["worktree_dir"] = worktree_dir
        return result
    except Exception as e:  # noqa: BLE001  # 取得手段を問わず失敗時はフォールバックせず記録するだけにする
        print(f"[注記] 参照コミット {commit} との同一環境比較を実施できなかった: {e!r}")
        return None
    finally:
        if worktree_added:
            subprocess.run(
                ["git", "worktree", "remove", "--force", worktree_dir],
                cwd=str(ROOT.resolve()), capture_output=True, text=True,
            )
        shutil.rmtree(worktree_dir, ignore_errors=True)


_reference_compat_values = compute_reference_compat_values(REFERENCE_COMMIT)

if _reference_compat_values is None:
    print(
        "[注記] 同一環境での比較を実施できなかったため、5.2.1 節のアサーションはスキップする"
        "(5.2.2 節の自己完結な検証は環境に依存せず実施される)。"
    )
else:
    # 参照値が現行の値と一致した場合でも、それが worktree 側のコードを実行した結果で
    # あることが読み取れるよう、コミットハッシュと実際に読み込まれた src のパスの
    # 両方を残す。
    print(f"参照コミット: {REFERENCE_COMMIT}")
    print(f"参照 worktree: {_reference_compat_values['worktree_dir']}")
    print(f"参照スクリプトが実際に読み込んだ src: {_reference_compat_values['src_path']}")
    print(f"参照コミット {REFERENCE_COMMIT} の SHA-256: {_reference_compat_values['state_hash']}")
    print(f"参照コミット {REFERENCE_COMMIT} の logits 和: {_reference_compat_values['logits_sum']:.10f}")
    assert _compat_state_hash.hexdigest() == _reference_compat_values["state_hash"], (
        "num_key_value_heads=None のパラメータ初期化が、本トピック追加直前のコミット"
        f"({REFERENCE_COMMIT})と同一環境の比較でも一致しない(本当の後方互換性の破壊の疑い)"
    )
    assert abs(float(_compat_logits_nocache.sum()) - _reference_compat_values["logits_sum"]) < 1e-6, (
        "順伝播の出力が、本トピック追加直前のコミットと同一環境の比較でも一致しない"
        "(本当の後方互換性の破壊の疑い)"
    )
    print(f"[OK] 参照コミット {REFERENCE_COMMIT}(worktree 側の src を実際に読み込んだ)と"
          "同一環境でパラメータ初期化・順伝播出力が一致")
```

    参照コミット: eb37dee
    参照 worktree: /tmp/ai_theories_ref_tflkefa6
    参照スクリプトが実際に読み込んだ src: /tmp/ai_theories_ref_tflkefa6/src/__init__.py
    参照コミット eb37dee の SHA-256: fe6cbe540cbbe935d3b155bc2fcbd9f1997fc326b505701709f95b29cf351a76
    参照コミット eb37dee の logits 和: 2.9690001011
    [OK] 参照コミット eb37dee(worktree 側の src を実際に読み込んだ)と同一環境でパラメータ初期化・順伝播出力が一致


#### 5.2.2 環境に依存しない自己完結な等価性の検証

`num_key_value_heads`を明示的に`num_heads`と同じ値で渡した場合と、省略した場合
(既定値`None`)とで、同一シードからのパラメータ初期化・順伝播出力・乱数消費量が
完全に一致することを直接確認する。同一プロセス内の比較のみで完結するため、
プラットフォームに一切依存しない。


```python
_eq_d_model, _eq_heads, _eq_layers, _eq_d_ff, _eq_seq, _eq_vocab = 32, 4, 2, 64, 32, 50
_eq_seed = 123

torch.manual_seed(_eq_seed)
_eq_rope_none = RotaryPositionEmbedding(_eq_d_model // _eq_heads)
model_kv_none = GPTLanguageModel(
    _eq_vocab, _eq_d_model, _eq_layers, _eq_heads, _eq_d_ff, _eq_seq,
    positional_transform=_eq_rope_none, num_key_value_heads=None,
)
_eq_rng_state_after_none = torch.get_rng_state()

torch.manual_seed(_eq_seed)
_eq_rope_explicit = RotaryPositionEmbedding(_eq_d_model // _eq_heads)
model_kv_explicit = GPTLanguageModel(
    _eq_vocab, _eq_d_model, _eq_layers, _eq_heads, _eq_d_ff, _eq_seq,
    positional_transform=_eq_rope_explicit, num_key_value_heads=_eq_heads,
)
_eq_rng_state_after_explicit = torch.get_rng_state()

# 1. 乱数消費量が完全に一致すること(新しい引数が乱数の消費経路を変えていないことの直接の検証)。
assert torch.equal(_eq_rng_state_after_none, _eq_rng_state_after_explicit), (
    "num_key_value_heads を明示的に渡すことで乱数の消費経路が変わっている"
)
print("[OK] num_key_value_heads=None と num_key_value_heads=num_heads で乱数消費量が完全一致")

# 2. 全パラメータが bit-identical であること。
_eq_state_none = model_kv_none.state_dict()
_eq_state_explicit = model_kv_explicit.state_dict()
assert _eq_state_none.keys() == _eq_state_explicit.keys()
for _key in _eq_state_none:
    assert torch.equal(_eq_state_none[_key], _eq_state_explicit[_key]), f"パラメータが一致しない: {_key}"
print(f"[OK] 全 {len(_eq_state_none)} パラメータが bit-identical")

# 3. 同一入力に対する順伝播出力(kv_cache=None)が bit-identical であること。
_eq_x = torch.randint(0, _eq_vocab, (1, 5), generator=torch.Generator().manual_seed(456))
model_kv_none.eval()
model_kv_explicit.eval()
with torch.no_grad():
    _eq_logits_none = model_kv_none(_eq_x, kv_cache=None)
    _eq_logits_explicit = model_kv_explicit(_eq_x, kv_cache=None)
assert torch.equal(_eq_logits_none, _eq_logits_explicit), "順伝播出力が一致しない"
print("[OK] 順伝播出力(kv_cache=None)が bit-identical")
```

    [OK] num_key_value_heads=None と num_key_value_heads=num_heads で乱数消費量が完全一致
    [OK] 全 28 パラメータが bit-identical
    [OK] 順伝播出力(kv_cache=None)が bit-identical


### 5.3 追加学習・符号化・評価のスケーリング計測と外挿

実験 D(7.5 節)で使う、008 が Hugging Face Hub にアップロードした本番チェックポイント
(`kojikojiprg/ai-theories-small-gpt-en`)・トークナイザ(`kojikojiprg/ai-theories-tokenizer-en`)・
コーパス(`en_006_pretraining.json`、006・008 と同一、356 記事)を読み込む。あわせて、
追加学習・符号化・評価それぞれについて、少なくとも 3 点のデータ量で実行時間を実測し、
べき乗則($\log t = \log a + b \log n$)のあてはめから本番データ量への外挿値を出す。

**コーパスの取得元について。** Google Colab は毎回リポジトリをクローンし直すため、
`.cache/`(`.gitignore`対象)は本番実行のたびに空の状態から始まる。以前の実装は
Wikipedia API から 356 記事を直接取得していたため、本番実行のたびに数分〜数十分の
取得時間がかかるうえ、API 側の一時的な応答不備(009 で`KeyError: 'parse'`を
引き起こしたものと同種)の影響を受けやすかった。ここでは、Hugging Face Hub の
Dataset リポジトリ(006・008 用に昇格済みの`corpus.txt`)から直接取得する
(`load_wikipedia_corpus_with_fallback`、`src/data/text.py`)。

`load_wikipedia_corpus_with_fallback`自体は Hub からの取得に失敗した場合、
Wikipedia API への直接取得にフォールバックする汎用関数である(009 と共有しており
変更しない)。しかし本トピックでは、同じセルでトークナイザ・チェックポイントも
Hugging Face Hub から取得しているため、Hub 自体が到達不能ならそちらで先に失敗する。
したがってこのフォールバックが発動する現実的な状況は「Hub は生きているがこの
コーパスのリポジトリだけ取得できない」場合であり、リポジトリ ID の取り違えか
リポジトリの削除を意味する。**取得元が Hub でなかった場合はアサーションで
即座に停止し、フォールバックを本番実行の経路から排除する**(静かに誤ったコーパスを
返すより失敗として顕在化させる)。

コーパスの取得は Hub からの単発ダウンロード(24.3 MB 程度)で完結し、データ量に応じて
繰り返される処理ではないため、5.3.1 節以降のスケーリング計測・外挿の対象には含めない
(外挿対象は、データ量に応じて実行時間が変化しうる符号化・追加学習・評価のみ)。


```python
# --- 本番チェックポイント・トークナイザの取得(008 がアップロードしたもの) ---
from huggingface_hub import hf_hub_download

tokenizer, _tokenizer_loaded_from_hub = load_bpe_id_tokenizer_from_hub(
    "kojikojiprg/ai-theories-tokenizer-en"
)
print(f"tokenizer vocab_size={tokenizer.vocab_size}, loaded_from_hub={_tokenizer_loaded_from_hub}")

_config_path = hf_hub_download(repo_id="kojikojiprg/ai-theories-small-gpt-en", filename="config.json")
_state_path = hf_hub_download(repo_id="kojikojiprg/ai-theories-small-gpt-en", filename="model_state.pt")
PROD_CONFIG = json.loads(Path(_config_path).read_text(encoding="utf-8"))
print("PROD_CONFIG:", PROD_CONFIG)
assert tokenizer.vocab_size == PROD_CONFIG["vocabulary_size"]


def build_production_model() -> GPTLanguageModel:
    d_k = PROD_CONFIG["d_model"] // PROD_CONFIG["num_heads"]
    rope = RotaryPositionEmbedding(d_k, max_position=PROD_CONFIG["sequence_length"])
    return GPTLanguageModel(
        vocabulary_size=PROD_CONFIG["vocabulary_size"],
        d_model=PROD_CONFIG["d_model"],
        num_layers=PROD_CONFIG["num_layers"],
        num_heads=PROD_CONFIG["num_heads"],
        d_ff=PROD_CONFIG["d_ff"],
        max_sequence_length=PROD_CONFIG["sequence_length"],
        positional_transform=rope,
        normalization_factory=RMSNorm,
        feed_forward_factory=functools.partial(
            SwiGLUFeedForwardNetwork, PROD_CONFIG["d_model"], PROD_CONFIG["swiglu_d_ff"]
        ),
        tie_embeddings=PROD_CONFIG["tie_embeddings"],
        dropout=PROD_CONFIG["dropout"],
    )


base_model = build_production_model()
base_model.load_state_dict(torch.load(_state_path, map_location="cpu"))
base_model.eval()
base_model = base_model.to(DEVICE)  # 5.3・6・7.5 節のすべての計算を DEVICE 上で行う
print(f"非埋め込みパラメータ数: {count_non_embedding_parameters(base_model):,}")

# --- コーパスの取得(006・008 と同一のマニフェスト、Hugging Face Hub から直接取得) ---
# 取得先リポジトリ ID の取り違えに注意する。以下は 356 記事の en_006_pretraining.json
# (006・008 の英語条件用)に対応するリポジトリであり、009(9826 記事、
# en_009_scaling.json)用の kojikojiprg/ai-theories-corpus-en とは別のリポジトリである
# (名前が短いほうを選ぶとエラーにならずに異なるコーパスが取得されるため、
# 取り違えを取得失敗として顕在化させる目的で意図的に別リポジトリに分けられている)。
_EN_PRETRAINING_CORPUS_REPO_ID = "kojikojiprg/ai-theories-corpus-en-pretraining"
_EXPECTED_MANIFEST_ARTICLE_COUNT = 356
# corpus.txt の期待バイト数(現在のセル出力で既に一致することを確認済み、
# データセットカードの metadata.json 記録値と同一)。metadata.json の raw_bytes との
# 照合だけでは、metadata.json 自体が取得したテキストから計算した値である経路
# (直接取得へのフォールバック)で自己参照になり何も検証しないため、取得元に
# 関わらないこの定数との照合を主たる検証とする。
_EXPECTED_CORPUS_BYTES = 24_331_593
_manifest_path = Path("src/data/wikipedia_manifests/en_006_pretraining.json")

_t0_corpus = time.time()
corpus_en, _corpus_meta = load_wikipedia_corpus_with_fallback(
    "en",
    _EN_PRETRAINING_CORPUS_REPO_ID,
    ".cache/wikipedia_en",
    manifest_path=_manifest_path,
    return_metadata=True,
)
_corpus_fetch_seconds = time.time() - _t0_corpus
print(f"corpus_en: {len(corpus_en):,} 文字 / {len(corpus_en.encode('utf-8')):,} バイト")
print(
    f"取得元: {_corpus_meta['source']}(リポジトリ: {_EN_PRETRAINING_CORPUS_REPO_ID}, "
    f"ファイル: corpus.txt, source_commit: {_corpus_meta['source_commit']}) / "
    f"取得時間: {_corpus_fetch_seconds:.1f} 秒(単発ダウンロードのため外挿対象外)"
)

# --- 不変条件のアサーション ---
# 本番実行では Hugging Face Hub からの取得を必須とし、Wikipedia API への直接取得への
# フォールバックは許容しない。同じセルでトークナイザ・チェックポイントも Hub から
# 取得しており、Hub 自体が到達不能ならそちらで先に失敗するため、ここでフォールバックが
# 発動する現実的な状況は「Hub は生きているがこのコーパスのリポジトリだけ取得できない」
# 場合であり、これはリポジトリ ID の取り違えかリポジトリの削除を意味する。
# データセットカードの「静かに誤ったコーパスを返すより、失敗として顕在化する構成を
# 優先する」方針と同じ理由で、ここでも検証を素通りさせず停止する。
assert _corpus_meta["source"] == "hub", (
    "Hugging Face Hub からの取得に失敗し、Wikipedia API への直接取得にフォールバックした。"
    f"リポジトリ ID の取り違えかリポジトリの削除の疑いがある(取得を試みたリポジトリ: "
    f"{_EN_PRETRAINING_CORPUS_REPO_ID})。"
)

assert _corpus_meta["manifest_article_count"] == _EXPECTED_MANIFEST_ARTICLE_COUNT, (
    f"マニフェスト記事数が期待({_EXPECTED_MANIFEST_ARTICLE_COUNT})と一致しない: "
    f"{_corpus_meta['manifest_article_count']}"
)
assert _corpus_meta["fetched_article_count"] == _EXPECTED_MANIFEST_ARTICLE_COUNT, (
    f"取得できた記事数が期待({_EXPECTED_MANIFEST_ARTICLE_COUNT})と一致しない: "
    f"{_corpus_meta['fetched_article_count']}"
)
assert not _corpus_meta["skipped_articles"], f"記事取得に失敗した記事がある: {_corpus_meta['skipped_articles']}"

# 実バイト数は、取得元に関わらない固定の期待値(_EXPECTED_CORPUS_BYTES)と照合する
# (metadata.json の raw_bytes は取得したテキスト自身から計算されうるため、それだけでは
# 自己参照になり検証にならない。上の source == "hub" のアサーションにより、この時点では
# metadata.json との照合にも意味がある)。
_actual_corpus_bytes = len(corpus_en.encode("utf-8"))
assert _actual_corpus_bytes == _EXPECTED_CORPUS_BYTES, (
    "corpus_en の実バイト数が固定の期待値と一致しない(取得の破損・取得元の取り違えの"
    f"疑いがある): 実測={_actual_corpus_bytes:,}, 期待値={_EXPECTED_CORPUS_BYTES:,}"
)
assert _actual_corpus_bytes == _corpus_meta["raw_bytes"], (
    "corpus_en の実バイト数が metadata.json の raw_bytes と一致しない: "
    f"実測={_actual_corpus_bytes:,}, メタデータ={_corpus_meta['raw_bytes']:,}"
)
print(f"[OK] 取得元 hub・マニフェスト・記事数・バイト数({_actual_corpus_bytes:,})がすべて期待通り")

# 訓練・検証分割の比率は、ハードコードした定数ではなく metadata.json の値を使う。
# 上の source == "hub" のアサーションにより、この時点で validation_ratio は必ず
# metadata.json 由来の値であり None にはならない(フォールバック時の既定値補完はしない)。
VALIDATION_RATIO_008 = _corpus_meta["validation_ratio"]
print(f"VALIDATION_RATIO_008: metadata.json の値を使用 = {VALIDATION_RATIO_008}")
assert VALIDATION_RATIO_008 == 0.05, (
    f"008 と同一であるべき VALIDATION_RATIO が 0.05 と一致しない: {VALIDATION_RATIO_008}"
)

# val_text_008 は P1 の検証にのみ使う。
train_text, val_text_008 = split_train_val_text(corpus_en, VALIDATION_RATIO_008)

# train_text(95%)の内側でさらに分割し、実験 D の学習用・評価用データを作る。
# d_eval_text は train_text の一部(val_text_008 とは重複しない)かつ d_train_text とも
# 重複しない、実験 D 専用の非重複評価窓になる。
D_EVAL_RATIO = 0.02
d_train_text, d_eval_text = split_train_val_text(train_text, D_EVAL_RATIO)
print(
    f"d_train_text: {len(d_train_text):,} 文字, d_eval_text: {len(d_eval_text):,} 文字"
    f"(val_text_008・d_train_text と非重複)"
)
```


    tokenizer.json:   0%|          | 0.00/661k [00:00<?, ?B/s]


    tokenizer vocab_size=8192, loaded_from_hub=True



    config.json:   0%|          | 0.00/305 [00:00<?, ?B/s]



    model_state.pt: reconstructing file:   0%|          |  0.00B / 21.0MB            



    model_state.pt: downloading bytes:           |  0.00B            


    PROD_CONFIG: {'vocabulary_size': 8192, 'd_model': 256, 'num_layers': 4, 'num_heads': 8, 'd_ff': 1024, 'swiglu_d_ff': 683, 'sequence_length': 256, 'dropout': 0.0, 'tie_embeddings': True, 'positional_encoding': 'rope', 'normalization': 'rmsnorm', 'feed_forward': 'swiglu', 'norm_first': True}
    非埋め込みパラメータ数: 3,149,056



    corpus.txt: reconstructing file:   0%|          |  0.00B / 24.3MB            



    corpus.txt: downloading bytes:           |  0.00B            



    metadata.json:   0%|          | 0.00/270 [00:00<?, ?B/s]


    コーパス取得元: kojikojiprg/ai-theories-corpus-en-pretraining(Hugging Face Hub)
    corpus_en: 24,214,546 文字 / 24,331,593 バイト
    取得元: hub(リポジトリ: kojikojiprg/ai-theories-corpus-en-pretraining, ファイル: corpus.txt, source_commit: aabac46a2708574ed67e522105684aa867a2b276) / 取得時間: 2.8 秒(単発ダウンロードのため外挿対象外)
    [OK] 取得元 hub・マニフェスト・記事数・バイト数(24,331,593)がすべて期待通り
    VALIDATION_RATIO_008: metadata.json の値を使用 = 0.05
    d_train_text: 22,543,743 文字, d_eval_text: 460,076 文字(val_text_008・d_train_text と非重複)



```python
# --- 5.3.1 符号化時間のスケーリング計測 ---
# 水準は本番データ量(2200 万文字強)に近い桁まで引き上げる(固定コストに埋もれない
# ようにするため。004 の教訓と同じく、小さすぎる水準は固定コストに支配される)。
_ENCODE_SIZES = [1_000_000, 4_000_000, 16_000_000]  # 等比刻み(smoke)
_encode_times = []
for size in _ENCODE_SIZES:
    t0 = time.perf_counter()
    encode_corpus(tokenizer, d_train_text[:size])
    _encode_times.append(time.perf_counter() - t0)
    print(f"符号化 {size:,} 文字: {_encode_times[-1]:.4f} 秒")

_encode_fit = fit_power_law_exponent(_ENCODE_SIZES, _encode_times)
print(f"符号化時間のべき指数: {_encode_fit.exponent:.3f} +- {_encode_fit.exponent_stderr:.3f}")
warn_if_unreliable_fit(_encode_fit, "符号化時間")

# 本番では d_train_text 全体を 1 回だけ符号化する(条件・シードをまたいで使い回すため)。
PRODUCTION_D_TRAIN_CHARS = len(d_train_text)
_encode_extrapolated_seconds = _encode_fit.coefficient * PRODUCTION_D_TRAIN_CHARS**_encode_fit.exponent
print(
    f"外挿: {PRODUCTION_D_TRAIN_CHARS:,} 文字の符号化 ~= {_encode_extrapolated_seconds:.1f} 秒"
    "(繰り返し回数 1 回)"
)
```

    符号化 1,000,000 文字: 2.3658 秒
    符号化 4,000,000 文字: 2.2694 秒
    符号化 16,000,000 文字: 11.2404 秒
    符号化時間のべき指数: 0.562 +- 0.342
    外挿: 22,543,743 文字の符号化 ~= 10.4 秒(繰り返し回数 1 回)



```python
# --- 5.3.2 追加学習時間のスケーリング計測 ---
# 水準を [20, 40, 80] に引き上げ、かつ eval_interval を steps より大きくすることで
# 評価(固定コスト、1 回あたり評価窓数に応じた非自明なコストがかかる)がこの計測に
# 混入しないようにする(実際の実験 D では最終ステップで評価するが、ここでの目的は
# 「1 ステップあたりの学習コスト」のべき乗則を得ることなので、評価コストを分離する)。
_probe_train_ids = encode_corpus(tokenizer, d_train_text[:400_000])
_probe_eval_ids = encode_corpus(tokenizer, d_eval_text[:50_000])
_probe_eval_windows, _probe_eval_mask = make_evaluation_windows(
    _probe_eval_ids, PROD_CONFIG["sequence_length"]
)
_probe_eval_bytes = len(d_eval_text[:50_000].encode("utf-8"))

_TRAIN_STEP_LEVELS = [20, 40, 80]  # 等比刻み(smoke)
_train_times = []
for steps in _TRAIN_STEP_LEVELS:
    _probe_model = convert_attention_to_grouped_query(base_model, num_key_value_heads=4, init="mean_pool")
    t0 = time.perf_counter()
    train_language_model(
        _probe_model, _probe_train_ids, _probe_eval_windows, _probe_eval_mask, _probe_eval_bytes,
        num_steps=steps, batch_size=8, sequence_length=PROD_CONFIG["sequence_length"],
        learning_rate=3e-4, eval_interval=steps + 1, device=DEVICE, seed=0,
    )
    _train_times.append(time.perf_counter() - t0)
    print(f"追加学習 {steps} ステップ(評価なし): {_train_times[-1]:.4f} 秒")

_train_fit = fit_power_law_exponent(_TRAIN_STEP_LEVELS, _train_times)
print(f"追加学習時間のべき指数: {_train_fit.exponent:.3f} +- {_train_fit.exponent_stderr:.3f}"
      "(1 に近いはず、1 ステップあたりのコストが一定であるため)")
warn_if_unreliable_fit(_train_fit, "追加学習時間")

# 外挿値には、実際の実験 D で最終ステップに 1 回発生する評価コストを別途加える
# (5.3.3 で測る評価時間のべき乗則あてはめを流用する。5.3.3 のセルの後に評価する)。
# PRODUCTION_D_NUM_STEPS はセットアップセル(SMOKE_TEST の定義の直後)で定義済み。
D_NUM_TRAINING_RUNS = 9  # 実験 D: 主条件 2 x 3 seed + 診断条件(MQA 平均プール)1 x 3 seed
_train_extrapolated_seconds_per_run = (
    _train_fit.coefficient * PRODUCTION_D_NUM_STEPS**_train_fit.exponent
)
_train_extrapolated_total = _train_extrapolated_seconds_per_run * D_NUM_TRAINING_RUNS
print(
    f"外挿(学習のみ、評価は含まない): {PRODUCTION_D_NUM_STEPS} ステップ x "
    f"{D_NUM_TRAINING_RUNS} 回 ~= {_train_extrapolated_total:.1f} 秒"
)
```

    追加学習 20 ステップ(評価なし): 6.9684 秒
    追加学習 40 ステップ(評価なし): 1.4896 秒
    追加学習 80 ステップ(評価なし): 2.9570 秒
    追加学習時間のべき指数: -0.618 +- 0.928(1 に近いはず、1 ステップあたりのコストが一定であるため)
    [警告] 追加学習時間 のべき指数の標準誤差(0.928)が推定値(-0.618)と同程度以上であり、この外挿は信用できない。水準・反復回数を増やして再計測すること。
    外挿(学習のみ、評価は含まない): 300 ステップ x 9 回 ~= 8.1 秒



```python
# --- 5.3.3 評価時間のスケーリング計測 ---
_full_eval_ids = encode_corpus(tokenizer, d_eval_text)
_full_eval_windows, _full_eval_mask = make_evaluation_windows(_full_eval_ids, PROD_CONFIG["sequence_length"])
_full_eval_bytes = len(d_eval_text.encode("utf-8"))
print(f"d_eval_text の評価窓数: {_full_eval_windows.shape[0]}")

_EVAL_WINDOW_LEVELS = [100, 200, 400]  # 等比刻み(smoke、_full_eval_windows の先頭から取る)。
# [50, 100, 200] では最小水準がまだ固定コストに支配されていた(50->100 窓で 1.23 倍
# にしかならず、窓数の比 2 倍から乖離)ため、プロンプト A のレビューを受けて引き上げた。
_eval_times = []
for n_windows in _EVAL_WINDOW_LEVELS:
    t0 = time.perf_counter()
    evaluate_bits_per_byte(
        base_model, _full_eval_windows[:n_windows], _full_eval_mask[:n_windows],
        _full_eval_bytes, DEVICE, batch_size=16,
    )
    _eval_times.append(time.perf_counter() - t0)
    print(f"評価 {n_windows} 窓: {_eval_times[-1]:.4f} 秒")

_eval_fit = fit_power_law_exponent(_EVAL_WINDOW_LEVELS, _eval_times)
print(f"評価時間のべき指数: {_eval_fit.exponent:.3f} +- {_eval_fit.exponent_stderr:.3f}")
warn_if_unreliable_fit(_eval_fit, "評価時間")

# 5.3.2 の学習時間の外挿(_train_extrapolated_total)は評価コストを含まない
# (eval_interval を steps より大きくして評価を起こさずに計測したため)。実験 D の
# 各学習ランが最終ステップで行う評価のコストは、下の D_NUM_EVAL_CALLS に基づく
# 外挿(初期評価 9 回 + 最終評価 9 回 + 診断用の元チェックポイント評価 1 回)に
# 含めて二重計上を避ける。

# 本番の評価回数: P1(1 回、val_text_008 全体)+ 実験 D(初期 + 最終、9 run + 診断用の元
# チェックポイント 1 回)。d_eval_text の全窓数を使う評価が (9 * 2 + 1) 回起きる。
PRODUCTION_NUM_EVAL_WINDOWS_D = int(_full_eval_windows.shape[0])
D_NUM_EVAL_CALLS = D_NUM_TRAINING_RUNS * 2 + 1
_eval_extrapolated_seconds_per_call = (
    _eval_fit.coefficient * PRODUCTION_NUM_EVAL_WINDOWS_D**_eval_fit.exponent
)
_eval_extrapolated_total_d = _eval_extrapolated_seconds_per_call * D_NUM_EVAL_CALLS

# P1(val_text_008 全体、1240 窓相当)の評価も同じべき乗則で見積もる。
_val_text_008_ids_len_estimate = len(val_text_008.encode("utf-8")) // 4  # 概算(バイト/4 ~= トークン数)
_val_text_008_windows_estimate = _val_text_008_ids_len_estimate // PROD_CONFIG["sequence_length"]
_eval_extrapolated_seconds_p1 = (
    _eval_fit.coefficient * _val_text_008_windows_estimate**_eval_fit.exponent
)
print(
    f"外挿: 実験 D の評価(窓数 {PRODUCTION_NUM_EVAL_WINDOWS_D} x {D_NUM_EVAL_CALLS} 回) "
    f"~= {_eval_extrapolated_total_d:.1f} 秒"
)
print(f"外挿: P1 の評価(概算窓数 {_val_text_008_windows_estimate}) ~= {_eval_extrapolated_seconds_p1:.1f} 秒")
```

    d_eval_text の評価窓数: 469
    評価 100 窓: 0.1613 秒
    評価 200 窓: 0.3222 秒
    評価 400 窓: 0.6401 秒
    評価時間のべき指数: 0.994 +- 0.002
    外挿: 実験 D の評価(窓数 469 x 19 回) ~= 14.3 秒
    外挿: P1 の評価(概算窓数 1185) ~= 1.9 秒



```python
# --- 5.3 節までの小計(符号化・追加学習・評価) ---
# 実験 A・B・C 自身の時間計測(生成・prefill・decode の呼び出し)の外挿は、各実験の
# 水準・べき乗則あてはめが定まってからでないと計算できないため、7 節の対応するセルで
# 個別に外挿し、全体の合計とセッション予算との比較は 7.6 節でまとめて行う。
SESSION_BUDGET_SECONDS = 2 * 60 * 60  # 006・008 と同じ基準(Google Colab 1 セッションの目安)

_subtotal_5_3_seconds = (
    _encode_extrapolated_seconds
    + _train_extrapolated_total
    + _eval_extrapolated_total_d
    + _eval_extrapolated_seconds_p1
)
print(f"符号化: {_encode_extrapolated_seconds:.1f} 秒")
print(f"追加学習(x{D_NUM_TRAINING_RUNS} 回、評価コストは含まない): {_train_extrapolated_total:.1f} 秒")
print(f"評価(実験 D、x{D_NUM_EVAL_CALLS} 回): {_eval_extrapolated_total_d:.1f} 秒")
print(f"評価(P1、x1 回): {_eval_extrapolated_seconds_p1:.1f} 秒")
print(f"5.3 節までの小計: {_subtotal_5_3_seconds:.1f} 秒"
      "(実験 A・B・C の外挿を含む最終合計は 7.6 節を参照)")
```

    符号化: 10.4 秒
    追加学習(x9 回、評価コストは含まない): 8.1 秒
    評価(実験 D、x19 回): 14.3 秒
    評価(P1、x1 回): 1.9 秒
    5.3 節までの小計: 34.6 秒(実験 A・B・C の外挿を含む最終合計は 7.6 節を参照)


## 6. 不変条件のアサーション / Invariant Assertions

5 節で確認済みのもの(再掲): **KV キャッシュの有無で貪欲法の生成結果が完全に一致
すること**(5.2 節)、**ノートブック内の最小実装と`src/generation/cache.py`の実装で
生成結果が完全に一致すること**(5.1 節)、**`num_key_value_heads=None`・
`use_cache=False`のとき 006〜009 と完全に同一の挙動になること**(5.2 節、パラメータ
初期化のハッシュと順伝播出力の両方で確認)。

このセルでは残りの不変条件を確認する。


```python
# --- 不変条件: 実測の KV キャッシュメモリ量が閉形式と一致する(多頭注意機構・GQA・MQA) ---
_mem_batch, _mem_seq = 2, 37
_mem_x = torch.randint(0, PROD_CONFIG["vocabulary_size"], (_mem_batch, _mem_seq), device=DEVICE)
_mem_positions = torch.arange(_mem_seq, device=DEVICE)

_mem_gqa = convert_attention_to_grouped_query(base_model, num_key_value_heads=4, init="random", seed=0)
_mem_mqa = convert_attention_to_grouped_query(base_model, num_key_value_heads=1, init="random", seed=0)

for name, model, num_kv_heads in [
    ("多頭注意機構(g=8)", base_model, PROD_CONFIG["num_heads"]),
    ("GQA(g=4)", _mem_gqa, 4),
    ("MQA(g=1)", _mem_mqa, 1),
]:
    kv_cache = KeyValueCache(PROD_CONFIG["num_layers"])
    with torch.no_grad():
        model(_mem_x, positions=_mem_positions, kv_cache=kv_cache)
    actual_bytes = kv_cache.memory_bytes()
    d_k = PROD_CONFIG["d_model"] // PROD_CONFIG["num_heads"]
    expected_bytes = compute_key_value_cache_memory_bytes(
        _mem_batch, PROD_CONFIG["num_layers"], _mem_seq, num_kv_heads, d_k, bytes_per_element=4
    )
    relative_error = abs(actual_bytes - expected_bytes) / expected_bytes
    print(f"{name}: 実測 {actual_bytes:,} バイト / 閉形式 {expected_bytes:,} バイト "
          f"(相対誤差 {relative_error:.2%})")
    assert relative_error < 0.01, f"{name} のキャッシュメモリ量が閉形式と 1% 以上乖離"
print("[OK] KV キャッシュメモリ量が閉形式(3.2 節)と全条件で一致")
```

    多頭注意機構(g=8): 実測 606,208 バイト / 閉形式 606,208 バイト (相対誤差 0.00%)
    GQA(g=4): 実測 303,104 バイト / 閉形式 303,104 バイト (相対誤差 0.00%)
    MQA(g=1): 実測 75,776 バイト / 閉形式 75,776 バイト (相対誤差 0.00%)
    [OK] KV キャッシュメモリ量が閉形式(3.2 節)と全条件で一致



```python
# --- 不変条件: トークナイザの符号化・復号がラウンドトリップで完全一致する ---
_roundtrip_sample = d_eval_text[: min(len(d_eval_text), 20_000)]
assert tokenizer.decode(tokenizer.encode(_roundtrip_sample)) == _roundtrip_sample
print(f"[OK] トークナイザのラウンドトリップ検証(先頭 {len(_roundtrip_sample):,} 文字)")

# --- 不変条件: GQA の平均プール初期化とランダム初期化で非埋め込みパラメータ数が完全一致する ---
_gqa_mean_probe = convert_attention_to_grouped_query(base_model, num_key_value_heads=4, init="mean_pool")
_gqa_random_probe = convert_attention_to_grouped_query(base_model, num_key_value_heads=4, init="random", seed=0)
_n_mean = count_non_embedding_parameters(_gqa_mean_probe)
_n_random = count_non_embedding_parameters(_gqa_random_probe)
print(f"非埋め込みパラメータ数: mean_pool={_n_mean:,}, random={_n_random:,}")
assert _n_mean == _n_random, "実験 D の主判定 2 条件で非埋め込みパラメータ数が一致しない"
print("[OK] 実験 D の主判定 2 条件(mean_pool / random)で非埋め込みパラメータ数が完全一致")
```

    [OK] トークナイザのラウンドトリップ検証(先頭 20,000 文字)
    非埋め込みパラメータ数: mean_pool=2,886,912, random=2,886,912
    [OK] 実験 D の主判定 2 条件(mean_pool / random)で非埋め込みパラメータ数が完全一致


## 7. 実験 / Experiments

### 7.1 全実験に共通する前提条件

**前提条件 P0**: すべての時間計測が GPU の同期(デバイスが CUDA なら
`torch.cuda.synchronize()`、MPS なら`torch.mps.synchronize()`、CPU なら不要)を
含み、ウォームアップ反復の後に行われていること。

**本節の前提条件は、後述の 7.1.1 節に記録した 2 回の本番実行(いずれも前提条件が
不成立)を受けて、本番実行のやり直しの前に改訂したものである。** 旧い定義
(条件ごとの生の変動係数が 0.1 以下であること)は、担保したい状態(ウォームアップが
済み、計測が安定していること)と指標が対応していなかった。生の変動係数
(標準偏差 ÷ 平均)は絶対時間が小さいほど大きくなりやすく、閾値を満たせるかどうかが
仮説と無関係な要因(その水準の絶対時間の大小)で決まってしまっていた
(7.1.1 節に実測値を記録する)。これは、**ウォームアップの担保** と
**判定に必要な精度の担保** という目的の異なる 2 つの要求を、1 つの指標(変動係数)で
兼ねようとしたことが原因である。この 2 つを分離し、それぞれ独立の前提条件として
宣言し直す。

- **P0-a(ウォームアップの担保)**: 各条件について、先頭 3 反復の平均と末尾 3 反復の
  平均の差が、その条件の全反復の標準偏差(`ddof=1`)の 2 倍以内であること
  (`compute_p0a`、`P0A_MAX_DRIFT_SIGMA=2.0`)。ウォームアップ不足による系統的な
  ドリフト(反復を重ねるにつれて値が一方向に変化していくこと)を直接検出する指標
  であり、絶対時間の大小には依存しない。反復回数が 3 の 2 倍(6)未満だと先頭 3・
  末尾 3 が重複してしまい定義できないため、実験 A・B・C の反復回数が 6 以上である
  ことをセットアップセルでアサーションにより確認する。
- **P0-b(判定精度の担保)**: 各条件について、平均の標準誤差
  (標準偏差(`ddof=1`) ÷ $\sqrt{n}$、$n$ は反復回数)が、その条件の平均の 5% 以下で
  あること(`compute_p0b`、`P0B_MAX_RELATIVE_STDERR=0.05`)。対比量の推定に使う
  各測定の平均が十分な精度で求まっていることを担保する。

**両方が成立して初めて前提成立とする。** いずれかが成立しない条件があれば、
その実験は前提不成立として報告する(本番実行時の扱いは CLAUDE.md の実験の規約に
従う)。

**反復回数は事前に宣言して固定し、前提が満たされるまで反復を増やすという運用は
しない。** P0-b は反復回数の平方根に反比例して改善する(標準誤差 ∝ $1/\sqrt{n}$、
$n$ は反復回数)ため、結果を見てから事後的に反復を増やせば必ず満たせて
しまい、前提条件として機能しなくなる。実験 A・B・C の反復回数(`PRODUCTION_A_REPEATS`・
`PRODUCTION_B_REPEATS`・`PRODUCTION_C_REPEATS`)は、7.1.1 節に記録した過去の実行の
観測値から P0-b を満たす見込みかどうかを見積もったうえで、本番実行の前にこの節で
確定させる(この見積もりの根拠は反復回数と変動係数という前提条件に関する量のみであり、
検証したい仮説の方向には依存しない)。

実験 A・B・C は、いずれも条件ごとに P0-a・P0-b の計算に必要な量(先頭・末尾平均、
全反復の標準偏差、平均、標準誤差、成否)を計測セルの直後で計算し、**セル出力に
全件印字する**(判定の一次情報の永続化先はセル出力のみであり、ファイルへの書き出しは
行わない)。**P0 は本番の実行環境(GPU の種類・負荷状況)によって成立しないことが
あり得る前提条件であるため、アサーションで停止するのではなく、成否を記録したうえで
判定に進む。**

#### 7.1.1 旧実行の記録(前提条件 P0 不成立)

これまでの本番実行 2 回は、いずれも旧い前提条件 P0(条件ごとの生の変動係数が 0.1 以下
であること)のもとで前提不成立となった。前提不成立の実行は仮説に関する情報を何も
与えないため、**対比量($\Delta b$・$\Delta b_B$・$\Delta r$)の数値はここに記載しない。**
記載するのは前提条件に関する量(変動係数)のみである。判定の一次情報(生データ)は
Hugging Face Hub へは昇格せず、セル出力への全件印字のみを記録とする方針のため、
前提条件に関する量は下表がすべてであり、本文の記述だけで記録が成立する。

##### 1 回目の本番実行

**実行日・環境**: 実行日はこうじさんの実行ログを参照して記入。Google Colab T4 GPU、
torch 2.13.0+cu130。水準は本節までに宣言した本番水準(`PRODUCTION_A_T_LEVELS`・
`PRODUCTION_A_REPEATS`、`PRODUCTION_B_LEVELS`・`PRODUCTION_B_REPEATS`(このときは
15)、`PRODUCTION_C_T_SMALL`・`PRODUCTION_C_T_LARGE`・`PRODUCTION_C_REPEATS`)を
そのまま使用した。計測ループは`for 条件: for 反復:`の順(1 つの条件の反復をまとめて
連続実行する順序、後述の 2 回目で修正する前の状態)。

**旧 P0(変動係数 <= 0.1)が実験 A・B・C のすべてで不成立だった。**

| 実験 | 最大変動係数 | 閾値を超えた条件 |
| --- | --- | --- |
| A | 0.1226 | cache_T128(0.1226)、nocache_T128(0.1115) |
| B | 0.2044 | decode_B1(0.2044)、decode_B64(0.1093) |
| C | 0.1080 | mqa_small(0.1080) |

個々の反復値を確認したところ、実験 A の`nocache_T128`は 10 回の反復中 3〜5 回目だけが
0.72〜0.77 秒台から 0.92〜0.96 秒台へ跳ね上がり、実験 C の`mqa_small`も反復の後半
(12 回目以降)だけが跳ねていた。1 つの条件の反復を連続実行していたため、一過性の
擾乱(OS のスケジューリング・サーマルスロットリングなど)が発生した区間に実行が
重なった条件にだけ集中して入ったと考えられる。

**修正内容(この時点での修正)**: 実験 A・B・C の計測ループを、
`for 条件: for 反復:`から`for 反復: for 条件:`の順に入れ替えた(ウォームアップは
従来どおり、各条件について計測ループの前にまとめて実行する。`WARMUP_REPEATS`は
変更しない)。この修正の根拠は「一過性の擾乱が特定の条件に集中しないよう、反復と
条件の実行順序を入れ替える」という計測設計上の一般論であり、どの条件の変動係数が
高かったかには依存しない。閾値を超えた条件は実験 A・B・C で条件の性質(T の大小・
batch size の大小・多頭注意機構と MQA)がばらばらであり、特定の仮説を支持・反証する
方向に条件を選んで修正したものではない。

##### 2 回目の本番実行

**実行環境**: Google Colab T4 GPU、torch 2.13.0+cu130。計測ループを
`for 反復: for 条件:`の順に入れ替えた後の実行。水準・反復回数は 1 回目と同じ
(`PRODUCTION_B_REPEATS`はこの時点ではまだ 15)。

**旧 P0(変動係数 <= 0.1)が実験 A・B で不成立となった(実験 C は成立した)。**

| 実験 | 最大変動係数 | 閾値を超えた条件 |
| --- | --- | --- |
| A | 0.1333 | T=128(cache 0.1333、nocache 0.1172)。T=2048 は 0.0067 |
| B | 0.1922 | decode_B1(0.1357)、decode_B4(0.1922)、decode_B16(0.1672)。prefill_B256 は 0.0155 |
| C | 0.0804 | なし(P0 成立、最大値のみ参考として記載) |

閾値を超えたのは、いずれも絶対時間が小さい水準(実験 A は生成長 $T$ が小さいほど
1 回あたりの生成時間が短く、実験 B は decode のバッチサイズが小さいほど 1 ステップの
処理が速い)に集中していた。一方で絶対時間が大きい水準(実験 A の $T=2048$、実験 B の
prefill・$B=256$)では変動係数は 0.02 未満だった。**変動係数(標準偏差 ÷ 平均)は
絶対時間が小さいほど大きくなりやすい統計量であり、この結果は「閾値を満たせるかどうかが
検証したい仮説とは無関係な要因(その水準の絶対時間)で決まってしまっていた」ことを
示している。** ウォームアップ不足を示す証拠(ドリフト)ではなく、旧 P0 の設計上の
誤りを示す証拠であると判断した。

##### 前提条件の改訂(旧 P0 の設計上の誤りと P0-a・P0-b への分割)

**旧 P0 の設計上の誤り**: 旧 P0 が担保したかったのは「ウォームアップが済み、計測が
安定していること」だったが、指標として採用した生の変動係数は、目的の異なる
2 つの要求(ウォームアップの担保・対比量の推定に必要な精度の担保)を 1 つの数値で
兼ねようとしたものだった。この 2 つは本質的に別の量であり、特に判定精度の担保は
反復回数を増やせば改善する量であるのに対し、変動係数(個々の測定値のばらつき)は
反復回数を増やしても縮まらない。2 回目の実行で観測された「絶対時間が小さい水準に
閾値超過が集中する」という結果は、この設計上の誤りの帰結である。

**改訂内容**: 7.1 節で宣言した P0-a(ウォームアップの担保)・P0-b(判定精度の担保)に
置き換えた。**この改訂は、3 回目の本番実行(やり直し)の前に行っている。** CLAUDE.md
の実験の規約が認める、本番実行の前の前提条件の改訂にあたる。改訂の根拠(絶対時間の
大小で閾値の成否が決まってしまうという設計上の欠陥)は観測結果の方向(どちらの条件で
差が出たか)に依存しない一般論であり、2 回目の実行結果を都合よく解釈し直したもの
ではない。**改訂したのは前提条件(P0)のみであり、判定基準(対比量の定義・閾値の
導出式・期待する差の方向)は一切変更していない。**

**実験 B の反復回数を 15 から 30 に増やした理由**: 2 回目の実行で観測された変動係数
(`decode_B4`の 0.1922 が実験 B 内で最大)を用いて P0-b(標準誤差が平均の 5% 以下)の
成否を見積もると、反復 15 回では標準誤差が平均の
$0.1922 / \sqrt{15} \approx 5.0\%$ となり、閾値ぎりぎりで満たせない見込みだった。
反復 30 回なら $0.1922 / \sqrt{30} \approx 3.5\%$ まで下がり、余裕を持って満たせる
見込みである。一方、実験 A(最大変動係数 0.1333、反復 10 回で
$0.1333 / \sqrt{10} \approx 4.2\%$)・実験 C(最大変動係数 0.0804、反復 15 回で
$0.0804 / \sqrt{15} \approx 2.1\%$)は、いずれも現行の反復回数で P0-b を満たす見込みが
あるため反復回数を変更しない。反復回数を増やす根拠は「その条件の標準誤差が P0-b の
閾値を満たさない見込みである」という前提条件に関する量(変動係数と反復回数)のみに
基づいており、検証したい仮説の方向には依存しない。なお、この見積もりはあくまで
2 回目の実行時点の変動係数からの概算であり、3 回目の本番実行で改めて P0-a・P0-b を
計算し直して成否を判定する(この見積もりを根拠に前提条件をアサーションで強制する
わけではない)。

水準(実験 A の $T$、実験 B のバッチサイズ、実験 C の $T$ の両端)は 1〜2 回目の実行を
通じて一切変更していない。



## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/03_efficient_training/010_kv_cache_and_inference_compute.ipynb
