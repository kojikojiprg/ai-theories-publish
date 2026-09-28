---
title: "報酬モデルと RLHF / Reward Models and RLHF(実装・実験編 4/4)"
---

この記事は後編(実装・実験編 4/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/017_reward_model_and_rlhf-practice-3)。

### 6.14 SFT モデルのアップロード(Hugging Face Hub)

参照方策 $\pi_{\mathrm{ref}}$(LoRA をマージした SFT モデル)を、`kojikojiprg/ai-theories-small-gpt-en`の新ブランチ
`sft-synthetic`にアップロードする。018 の DPO が参照方策として使う。

- **`UPLOAD_ARTIFACTS = True`のときだけ** アップロードする(既定は`False`。`SMOKE_TEST`とは独立)。スモークテストの重みは
  アップロードしない(`SMOKE_TEST = True`ではアップロードしない)。
- **前提条件 P0(6.5 節)が成立していない場合は、`UPLOAD_ARTIFACTS = True`・`SMOKE_TEST = False`でもアップロードしない。**
  P0 不成立の重みは、018 の参照方策として公開しない。スキップしたことと理由を印字する。判定の順序は`UPLOAD_ARTIFACTS`→
  `SMOKE_TEST`→ P0 →`HF_TOKEN`の取得である。
- Google Colab の Secrets から`HF_TOKEN`が取得できない場合は、例外で停止せず、アップロードをスキップしてその旨を印字する。
- 手順: ブランチ`sft-synthetic`を作成 → `model_state.pt`・`config.json`(`main`の`config.json`をそのまま使う。構造は同一)・
  `README.md`(モデルカード)をアップロード → `list_repo_files(..., revision="sft-synthetic")`で存在を確認し、
  `model_state.pt`をダウンロードし直して SHA-256 が一致することを確かめる。
- **トークナイザは同梱しない**(`kojikojiprg/ai-theories-tokenizer-en`を使う)。ブランチに`tokenizer.json`がないことも確かめる。
- モデルカードの性能指標は、アップロードする重みそのもの(6.5 節で評価したマージ後の重み)の値である。


```python
import tempfile  # noqa: E402

# 保存用の state_dict(CPU)。重み共有(token_embedding.weight と lm_head.weight)を保つため、元の
# ストレージごとに 1 回だけ CPU へ移す(010 と同じ扱い。共有を保たないと埋め込み行列が 2 重に保存される)
_cpu_cache: dict[int, torch.Tensor] = {}
_upload_state: dict[str, torch.Tensor] = {}
for _key, _value in reference_policy.state_dict().items():
    _ptr = _value.data_ptr()
    if _ptr not in _cpu_cache:
        _cpu_cache[_ptr] = _value.detach().cpu().clone()
    _upload_state[_key] = _cpu_cache[_ptr]
SFT_CHECKPOINT_PATH = Path(tempfile.mkdtemp()) / "model_state.pt"
torch.save(_upload_state, SFT_CHECKPOINT_PATH)
_reloaded = torch.load(SFT_CHECKPOINT_PATH, map_location="cpu")
assert _reloaded.keys() == BASE_STATE.keys()
assert all(torch.equal(_reloaded[k], REFERENCE_STATE[k]) for k in REFERENCE_STATE)
SFT_CHECKPOINT_SHA256 = hashlib.sha256(SFT_CHECKPOINT_PATH.read_bytes()).hexdigest()
print(f"保存・読み戻しで全 {len(_reloaded)} テンソルがマージ後の重みと一致: OK")
print(f"バイト数 {SFT_CHECKPOINT_PATH.stat().st_size:,}、SHA-256 {SFT_CHECKPOINT_SHA256}")


def build_sft_model_card() -> str:
    return f"""---
language: en
license: mit
tags:
- ai-theories
- gpt
- sft
- scratch-implementation
---

# ai-theories 小型 GPT の SFT モデル(合成の指示データ、ブランチ: `{SFT_BRANCH}`)

`ai-theories`(https://github.com/kojikojiprg/ai-theories)プロジェクトの成果物。
[kojikojiprg/ai-theories-small-gpt-en](https://huggingface.co/{MODEL_REPO_ID}) の `main` ブランチ(008 の事前学習済み
モデル)を、[016. SFT(指示チューニング)](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/016_supervised_fine_tuning-theory)
の条件 1 のレシピ(合成の指示データ 65,536 事例・損失マスクあり・Query / Value 射影への LoRA(r = 8、alpha = 8)・学習率 1e-2・
2,048 ステップ・シード 0)で、[017. 報酬モデルと RLHF](https://github.com/kojikojiprg/ai-theories/blob/main/theories/04_alignment/017_reward_model_and_rlhf.ipynb)
の中で学習し直し、LoRA を重みにマージしたもの。017 と 018(DPO)の参照方策として使う。

スクラッチ実装であり、研究・教育目的のモデルである。品質保証は行っていない。商用・実運用での利用は想定しない。

## 構成

`config.json` を参照(`main` ブランチと同一の構造。LoRA はマージ済みで、追加のパラメータはない)。

**トークナイザはこのリポジトリ・ブランチには同梱していない。**
[kojikojiprg/ai-theories-tokenizer-en](https://huggingface.co/{TOKENIZER_REPO_ID}) を使用すること(`main` ブランチと共通)。

## 性能指標(アップロードした重みそのものを評価した値)

- 016 の評価集合(250 入力 x 4 課題 = {len(P0_EXAMPLES)} 事例、貪欲法、終端記号 `### End` まで最大 32 トークン):
  完全一致率 {SFT_EXACT_MATCH:.4f}、形式の遵守率 {SFT_FORMAT_RATE:.4f}
- 同じ評価集合の応答部分の負の対数尤度(教師強制): {SFT_RESPONSE_NLL:.4f} nats / トークン
- 英語版 Wikipedia の検証 bits-per-byte(008 と同じ分割・評価): {SFT_BITS_PER_BYTE:.6f}(SFT 前 {BASE_BITS_PER_BYTE:.6f})

## 検証情報

- SHA-256(`model_state.pt`): `{SFT_CHECKPOINT_SHA256}`

## 関連ノートブック

- [016. SFT(指示チューニング)](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/016_supervised_fine_tuning-theory)
- [017. 報酬モデルと RLHF](https://github.com/kojikojiprg/ai-theories/blob/main/theories/04_alignment/017_reward_model_and_rlhf.ipynb)
"""


def upload_skip_reason(upload_artifacts: bool, smoke_test: bool, p0_holds: bool) -> str | None:
    # アップロードをスキップする理由(スキップしないなら None)。判定の順序: UPLOAD_ARTIFACTS -> SMOKE_TEST -> P0
    if not upload_artifacts:
        return "UPLOAD_ARTIFACTS = False のため、アップロードをスキップした。"
    if smoke_test:
        return "SMOKE_TEST = True のため、スモークテストの重みはアップロードしない(スキップした)。"
    if not p0_holds:
        return (
            "前提条件 P0 が成立していないため、アップロードをスキップした"
            "(P0 不成立の重みは 018 の参照方策として公開しない)。"
        )
    return None


# 分岐の順序の確認(全 8 通り)
for _upload, _smoke, _p0 in itertools.product((False, True), repeat=3):
    _reason = upload_skip_reason(_upload, _smoke, _p0)
    assert (_reason is None) == (_upload and not _smoke and _p0)
    assert _upload or _reason.startswith("UPLOAD_ARTIFACTS")
    assert not (_upload and _smoke) or _reason.startswith("SMOKE_TEST")
    assert not (_upload and not _smoke and not _p0) or _reason.startswith("前提条件 P0")
print("アップロードの分岐(UPLOAD_ARTIFACTS -> SMOKE_TEST -> P0 -> HF_TOKEN)の全 8 通りの確認: OK")

_skip_reason = upload_skip_reason(UPLOAD_ARTIFACTS, SMOKE_TEST, bool(precondition_status.get("P0")))
print(
    f"UPLOAD_ARTIFACTS = {UPLOAD_ARTIFACTS}、SMOKE_TEST = {SMOKE_TEST}、P0 = {precondition_status.get('P0')}"
)
if _skip_reason is not None:
    print(_skip_reason)
else:
    _hf_token = None
    try:
        from google.colab import userdata

        _hf_token = userdata.get("HF_TOKEN")
    except Exception as _e:  # noqa: BLE001  # Colab Secrets 未設定・非 Colab 環境など理由を問わずスキップする
        print(
            f"Colab Secrets から HF_TOKEN を取得できなかったため、アップロードをスキップした: {type(_e).__name__}"
        )
    if not _hf_token:
        print("HF_TOKEN がないため、アップロードをスキップした。")
    else:
        from huggingface_hub import HfApi

        _api = HfApi(token=_hf_token)
        _config_bytes = Path(_config_path).read_bytes()  # main の config.json(構造は同一)
        _api.create_branch(repo_id=MODEL_REPO_ID, branch=SFT_BRANCH, exist_ok=True)
        for _source, _target in (
            (str(SFT_CHECKPOINT_PATH), "model_state.pt"),
            (_config_bytes, "config.json"),
            (build_sft_model_card().encode("utf-8"), "README.md"),
        ):
            _api.upload_file(
                path_or_fileobj=_source,
                path_in_repo=_target,
                repo_id=MODEL_REPO_ID,
                revision=SFT_BRANCH,
            )
        # upload_file が例外を送出しなかったことだけを成功の根拠にせず、存在と内容を確かめる
        _files = set(_api.list_repo_files(repo_id=MODEL_REPO_ID, revision=SFT_BRANCH))
        for _expected in ("model_state.pt", "config.json", "README.md"):
            assert _expected in _files, (
                f"{_expected} が {MODEL_REPO_ID}@{SFT_BRANCH} に見つからない"
            )
        assert "tokenizer.json" not in _files, "トークナイザが同梱されている"
        _downloaded = hf_hub_download(
            MODEL_REPO_ID, "model_state.pt", revision=SFT_BRANCH, force_download=True
        )
        assert hashlib.sha256(Path(_downloaded).read_bytes()).hexdigest() == SFT_CHECKPOINT_SHA256
        print(
            f"[OK] アップロード完了。list_repo_files で存在を確認し、model_state.pt の SHA-256 が一致した: "
            f"https://huggingface.co/{MODEL_REPO_ID}/tree/{SFT_BRANCH}"
        )
```

    保存・読み戻しで全 39 テンソルがマージ後の重みと一致: OK
    バイト数 20,997,217、SHA-256 5814fbc024a27adc1624ee9a718fbbd5a7e28f5d20a0df40c8d4633126a91ab5
    アップロードの分岐(UPLOAD_ARTIFACTS -> SMOKE_TEST -> P0 -> HF_TOKEN)の全 8 通りの確認: OK
    UPLOAD_ARTIFACTS = True、SMOKE_TEST = False、P0 = True



    Processing Files (0 / 0)      : |          |  0.00B /  0.00B            



    New Data Upload               : |          |  0.00B /  0.00B            



      ...mpju7u3l_g/model_state.pt:  84%|########4 | 17.7MB / 21.0MB            


    No files have been modified since last commit. Skipping to prevent empty commit.
    WARNING:huggingface_hub.hf_api:No files have been modified since last commit. Skipping to prevent empty commit.



    model_state.pt: reconstructing file:   0%|          |  0.00B / 21.0MB            



    model_state.pt: downloading bytes:           |  0.00B            


    [OK] アップロード完了。list_repo_files で存在を確認し、model_state.pt の SHA-256 が一致した: https://huggingface.co/kojikojiprg/ai-theories-small-gpt-en/tree/sft-synthetic


## 7. 結果・考察 / Results and Discussion

本番実行(Google Colab T4)のセル出力に基づいて記す。判定は 6.1 節で事前に宣言した基準のみから導く(7.4〜7.6 節)。結果を見た後に立てた解釈は 7.8 節に分けて記し、検証済みの結論としては扱わない。

### 7.1 実行の概要

- **実行環境**(5.1 節の印字): Tesla T4(compute capability 7.5、総メモリ 14.56 GiB)、Python 3.13.15、torch 2.13.0+cu130(ビルド時の CUDA 13.0、cuDNN 92000)、コミット`108c33b`(未コミットの変更なし)、実行日時 2026-09-27T23:44:12(UTC)。決定的な実行の確認(5.7 節)は CUDA で通り、報酬モデルの学習(埋め込みを含む全パラメータ)も 2 回の実行で bit 単位で一致した。
- **スモークテストのコミットとの差**: スモークテストを行ったコミット`181c89d`から本番のコミット`108c33b`までの 4 つのコミットは、いずれも Google Colab からの本ノートブックの保存であり、`src/`・`requirements.txt`・`uv.lock`・`pyproject.toml`には変更がない。本番で使った実装と依存関係は、スモークテストで確かめたものと同一である。
- **セットアップセルの差し替え**: 最初の試行では、Colab のカーネルが起動時に読み込んだ numpy 2.1.3・matplotlib 3.10.0・pillow 11.3.0 が、セットアップセルのインストールで 2.5.2・3.11.1・12.3.0 に入れ替わり、5.5 節の`np.testing`で`AttributeError`が出た(メモリ上の古い版とディスク上の新しい版の混在)。手動の「セッションを再起動」では安定して解消できなかったため、5.1 節のセルの`if IN_COLAB:`のブロックを Colab 上で差し替えて実行した。差し替えた内容は、`/content/ai-theories`がなければ clone し、インストールの後に numpy・matplotlib・pillow のメモリ上とディスク上の版を比べ、食い違えばカーネルを終了して再実行を促すものである。本番の出力では食い違いなしで進んだ(インストールの出力が「Checked 59 packages」なのは、先の試行で同じ版がすでに入っていたためである)。この差し替えは実行環境の準備のみに関わり、実験のコード・データ・乱数・判定には影響しない。
- **削る段階**(6.4 節): T4 での見積もりは段階 0 で 83.3 分(予算 120 分の 69.5%)、段階 1〜4 で 69.8・59.0・52.9・47.5 分で、予算に収まる最小の段階として **段階 0** が選ばれた。実験 A・B・C はいずれも 5 シード(0〜4)、実験 C は 4 水準 $\{1024, 4096, 16384, 65536\}$ で実行した。PPO の反復回数は、宣言した実行時間の上限(2 水準の合計で 15 分、1 反復あたりの見積もり 3.52 秒)により、150 から **127** に打ち切られた。ノートブック全体の実行時間は 62.3 分だった(報酬モデル 20 本の学習とスコアリングで 2,647.1 秒、PPO で 494.7 秒)。
- **SFT モデルのアップロード**(6.14 節): 参照方策 $\pi_{\mathrm{ref}}$ の重みを`kojikojiprg/ai-theories-small-gpt-en`のブランチ`sft-synthetic`(https://huggingface.co/kojikojiprg/ai-theories-small-gpt-en/tree/sft-synthetic)にアップロードした。`list_repo_files`で`model_state.pt`・`config.json`・`README.md`の存在を確かめ、ダウンロードし直した`model_state.pt`の SHA-256 が`5814fbc024a27adc1624ee9a718fbbd5a7e28f5d20a0df40c8d4633126a91ab5`で一致した(`config.json`は`main`と同一の内容なので、変更なしとして書き込みが省略された)。
- **前提条件はすべて成立した。**

| 前提条件 | 値 | 基準 | 成否 |
|---|---|---|---|
| P0 | 完全一致率 0.8180、形式の遵守率 1.0000(016 の評価集合 1,000 事例) | 完全一致率が $0.8040 \pm 0.0867$ の範囲、形式の遵守率が 0.99 以上 | 成立 |
| P1 | 真の順序との一致率 0.6802・0.6601・0.6235・0.6789・0.6568(シード順、$\sigma_a$ は 0.0045〜0.0059) | 全シードで $a - 0.5 > 2\sigma_a$ | 成立 |
| P2 | 異なり応答数の比率の平均 0.9351(最小 0.3926) | $n_{\max} / M = 0.25$ 以上 | 成立 |

### 7.2 SFT(参照方策)の再現

- 016 の条件 1 のレシピでの SFT(2,048 ステップ、60.7 秒)の後、LoRA をマージした重みの、016 の評価集合での完全一致率は **0.8180**、形式の遵守率は 1.0000 だった(016 の 5 シード平均 0.804、許容 ±0.0867)。応答部分の負の対数尤度は 0.6005 nats / トークンだった。この 2 つの値は、016 の条件 1 のシード 0 の本番の値(0.8180・0.6005)と一致しており、同じデータ・同じシード・決定的な実行で 016 の学習を再現できている。
- LoRA のマージの前後で、logits の差の最大値は $3.67 \times 10^{-5}$、貪欲法の生成の一致率は 1.0000 だった。
- 英語版 Wikipedia の検証 bits-per-byte は、SFT の前の 1.668 から 3.152 に上がった(悪化した)。合成課題だけで SFT したモデルは、一般の文章の言語モデルとしての性能を大きく失っている。018 でこのモデルを参照方策として使うときは、合成課題に特化したモデルであることに注意する。

### 7.3 応答プールと選好データ

- 評価用の 256 プロンプトについて、$\pi_{\mathrm{ref}}$ から $M = 512$ 個ずつ応答を抽選した(144.3 秒)。異なり応答数の比率の平均は 0.9351 で、同じ応答の重複は少なかった。
- プールの $r^*$ の平均は 0.7257、5%・50%・95% 分位点は 0.5515・0.734・0.9167 で、完全一致($r^* = 1$)の割合は 0.0332 だった。貪欲法の完全一致率 0.818 に対して、温度 1.0 のサンプリングでは完全一致はまれで、ほとんどの応答は部分的に正しい。
- 評価用の組(65,536 組)の同点($\Delta r^* = 0$)の割合は 0.0259 で(同一の応答の組 0.0069、同一でない応答の同点 0.0190)、真の報酬は組の大半を区別できる段階的な値だった。
- 選好の組(65,536 組)から $\kappa = 12.47$($\Delta r^* \ne 0$ の組の $\lvert \Delta r^* \rvert$ の中央値 0.0881)が決まった。ラベルが真の順序と一致する割合はシード順に 0.7544・0.7543・0.7523・0.7552・0.7538 で、Bradley-Terry モデルからの期待値 0.7546 と整合する。

### 7.4 実験 A(報酬の尺度の較正)の判定結果

**判定: 支持。**

較正の傾き $\hat{a}_s$ はシード順に 1.0586・1.0634・1.0674・1.0363・1.0264 で、

$$
\rho = 1.0504, \qquad \sigma_\rho = 0.0254, \qquad [\rho - 2\sigma_\rho, \rho + 2\sigma_\rho] = [0.9996, 1.1013]
$$

だった(シード間の標本標準偏差 $s_\rho = 0.0180$、ブートストラップの分散 $5.813 \times 10^{-4}$)。区間が同等性の範囲 $[0.8, 1.2]$ に完全に含まれるので、事前に宣言した基準により **支持** となる。ただし区間の下端 0.9996 は 1 にごく近く、区間はほぼ全体が 1 以上の側にある(推定値が 1 をやや上回ることについては 7.8 節)。

- ブートストラップの再標本ごとの値を 1 ステップのニュートン法で近似したことの誤差は、シード 0 の先頭 20 再標本で最大 $1.81 \times 10^{-3}$ で、再標本の値のばらつき(標準偏差 $2.81 \times 10^{-2}$)の約 1/16 だった。
- 診断量:
  - 予測の忠実度 $b_s / \kappa$ はシード順に 0.3379・0.2889・0.2827・0.3241・0.3167(平均 0.3101)。回帰の切片は $-0.0004$〜$+0.0044$ で、ほぼ 0 だった。
  - プロンプト内の Spearman の順位相関の平均は 0.339〜0.501、真の順序との一致率は 0.624〜0.680 だった。
  - 評価用の組に抽選したラベルとの一致率は 0.590〜0.627 で、ベイズ最適な一致率の上限 0.7533 を 0.13〜0.16 下回った。
- **較正の傾きと予測の忠実度の対比**: 較正の傾きは 1 付近(1.05)なのに、予測の忠実度は約 0.31 だった。これは 3.3 節で事前に述べた回帰の希釈の帰結と合っている。報酬モデルは真の報酬の差を大きく取りこぼしている(順位相関 0.34〜0.50、ラベルとの一致率が上限より 0.13〜0.16 低い)。それでも、予測した差の大きさは、その予測が持つ情報に見合った対数オッズとして較正されている。報酬モデルが見分けられない分は差 0 の側に縮むので、$\hat{\Delta}$ を $\Delta r^*$ に回帰した傾きは $\kappa$ の約 0.31 倍になる。一方、$\hat{\Delta}$ を説明変数とするラベルの確率への合い方(較正の傾き)は 1 付近に保たれる。散布図(6.8 節)でも、$\Delta r^*$ の小さい組の多くは $\hat{\Delta}$ が 0 付近に縮んでおり、回帰直線($b = 4.21$、シード 0)は $\kappa \Delta r^*$ の直線よりずっと緩い。なお、スモークテストの後に改訂する前の対比量(6.2 節の改訂 1 の旧基準の $b / \kappa$)は約 0.31 で、同等性の範囲から大きく外れる。判定は改訂後の基準によるが、旧基準のままであれば、予測の不正確さを尺度の識別の失敗と取り違えていたことになる。

### 7.5 実験 B(best-of-n による過最適化)の判定結果

**判定: 支持。**

真の報酬の期待値 $\bar{R}(n)$(シード平均)は、$n = 1, 2, 4, \dots, 128$ で 0.7257・0.7621・0.7735・0.7736・0.7673・0.7591・0.7515・0.7444 だった。

$$
g_1 = +0.01552 \ (\sigma_{g_1} = 0.00239), \qquad g_2 = -0.00762 \ (\sigma_{g_2} = 0.00056)
$$

($\log_2 n$ あたり)。$g_1 > 2\sigma_{g_1} = 0.00477$ かつ $g_2 < -2\sigma_{g_2} = -0.00112$ なので、事前に宣言した基準により **支持** となる。

- シードごとに見ても、前半の傾きは 5 シードすべてで正(+0.00709〜+0.01939)、後半の傾きは 5 シードすべてで負(−0.00798〜−0.00710)だった。
- 真の報酬は $n = 4$〜8 で最大(0.7735・0.7736)となり、$n = 128$ では 0.7444 まで下がった。$n = 128$ でも、$n = 1$(参照方策そのもの、0.7257)よりは高い。
- 代理報酬の期待値は 1.7270 から 2.3378 まで単調に増え続けた(数学的に単調増加であり、判定には使わない)。代理報酬が上がり続ける一方で真の報酬が下がるという、過最適化の形である。
- 完全一致($r^* = 1$)の確率は、$n = 1$ の 0.0332 から $n = 4$ で最大の 0.0547 に上がり、$n = 128$ では 0.0026 まで下がった。強い最適化の圧力のもとでは、選ばれる応答が完全一致の応答であることがまれになった。
- KL ダイバージェンスの上界 $\log n - (n-1)/n$ との対応: 真の報酬が最大になる $n = 4$〜8 は上界で 0.636〜1.204 nats、$n = 128$ は 3.860 nats にあたる。いずれも上界であり(3.6 節)、プールの同じ応答の重複の分だけ、実際の KL ダイバージェンスはこれより小さい。

### 7.6 実験 C(選好データ量による緩和)の判定結果

**判定: 支持。**

| 学習の組の数 $N$ | エポック数 | $R(n_{\max})$ の平均 | 放物線の頂点($\log_2 n$) | 真の順序との一致率の平均 | 学習の最後の 100 ステップのラベルとの一致率 |
|---|---|---|---|---|---|
| 1,024 | 64 | 0.7111 | 2.14 | 0.5707 | 1.0000(全シード) |
| 4,096 | 16 | 0.7187 | 2.64 | 0.5926 | 0.9941〜0.9962 |
| 16,384 | 4 | 0.7309 | 3.23 | 0.6353 | 0.7241〜0.7897 |
| 65,536 | 1 | 0.7444 | 3.56 | 0.6599 | 0.5803〜0.6316 |

$R(n_{\max})$ の $\log_2 N$ に対する傾きは $c = +0.00562$、$\sigma_c = 0.00185$($2\sigma_c = 0.00369$)で、$c > 2\sigma_c$ なので、事前に宣言した基準により **支持** となる。

- 学習の組を増やすほど、$n_{\max} = 128$ での真の報酬は高く、放物線のあてはめの頂点は大きい $n$ の側に移り(過最適化が始まる最適化の圧力が強くなる)、真の順序との一致率は高くなった。
- シードごとの傾きは +0.00833・+0.00700・−0.00134・+0.00828・+0.00582 で、**シード 2 だけが負** だった。シード 2 の $N = 65536$ での $R(n_{\max})$ は 0.7106 で、同じシードの $N = 16384$(0.7307)より低い。
- **事前に宣言した交絡の現れ**: 学習の計算量を揃えたため、小さい水準では同じ組を多くのエポック反復している。$N = 1024$(64 エポック)では学習の最後の 100 ステップのラベルとの一致率が全シードで 1.0000(損失 0.0001)、$N = 4096$(16 エポック)でも 0.994〜0.996 で、確率的なラベル(期待される一致率 0.7546)をほぼそのまま記憶していた。$N = 65536$(1 エポック)では 0.58〜0.63 だった。したがって、この支持は「同じ計算量のもとで、異なる組が多いほど良い」ことを示すが、組の数そのものの効果と、反復による過適合の減少の効果は分離できない(6.1 節)。

### 7.7 PPO の動作確認(判定なし)

判定基準を設けない定性的な観察である。標準の水準・シード 0 の報酬モデルを報酬とし、各 127 反復(1 反復 64 プロンプト)を行った(β = 0.01 で 248.9 秒、β = 0.1 で 245.7 秒)。最初と最後の 10 反復の平均は次のとおりである。

| KL 係数 $\beta$ | KL ダイバージェンスの推定値(nats) | 代理報酬 | 真の報酬 | 応答の長さ(トークン) | クリップが効いた割合(平均) |
|---|---|---|---|---|---|
| 0.01 | 0.057 → 4.427 | 1.446 → 1.656 | 0.7165 → 0.6594 | 12.9 → 12.6 | 0.004 |
| 0.1 | 0.044 → 0.534 | 1.456 → 1.542 | 0.7162 → 0.7136 | 12.8 → 12.7 | 0.002 |

- $\beta = 0.01$ では、KL ダイバージェンスが 4 nats を超えるまで増え、代理報酬が上がる一方で、真の報酬は開始時の値を下回って下がった。図(6.11 節)では、真の報酬の下落が最も急なのは 60〜80 反復付近で、KL ダイバージェンスが急に増えた時期と重なる。実験 B の best-of-n と同じ向きの過最適化が、PPO でも見られた。
- $\beta = 0.1$ では、KL ダイバージェンスは 0.5 nats 前後に抑えられ、代理報酬はわずかに上がり、真の報酬はほぼ変わらなかった。
- どちらの条件でも、真の報酬が開始時の値から明確に上がる区間は見られなかった($\beta = 0.1$ の 10 反復の移動平均は 0.71〜0.73 の範囲で推移した)。best-of-n では参照方策(0.7257)より真の報酬が上がる $n$ があったのに対し、この設定の PPO(学習率 $10^{-3}$、127 反復)では、真の報酬の明確な改善は見られなかった。

### 7.8 事後的な解釈(検証済みの結論ではない)

以下は結果を見た後に立てた解釈であり、事前に宣言した判定の対象ではない。検証していないので、結論としては扱わない。

- **実験 A で較正の傾きが 5 シードすべてで 1 を上回ったこと**: $\hat{a}_s$ は 1.03〜1.07 で、すべて 1 より大きかった。これは、学習に使っていない組では、報酬モデルの報酬の差がラベルの対数オッズに比べてやや小さい(確信度が控えめである)ことを意味する。3.3 節の一階条件は学習の目的関数の最適点での性質なので、1 エポック(2,048 ステップ、学習率を cosine で下げて打ち切る)の学習では最適点に達しておらず、報酬の尺度が最適点より小さいところで止まった可能性がある。学習データの上での較正の傾きを測れば区別できるが、本番では測っていない。
- **実験 C でシード 2 の傾きが負になったこと**: シード 2 の標準の水準の報酬モデルは、P1 の真の順序との一致率が 0.6235 で、5 シードの中で最も低かった(他は 0.6568〜0.6802)。順位相関(0.339)とラベルとの一致率(0.5895)も最も低い。標準の水準の報酬モデルがたまたま弱かったために、$N = 65536$ での $R(n_{\max})$ が下がり、そのシードの傾きが負になった可能性がある。
- **実験 C の支持の中身**: 7.6 節のとおり、小さい水準ではラベルをほぼ記憶していた。$N = 1024$・4096 の報酬モデルの真の順序との一致率(0.57・0.59)が低いことは、組の数の少なさと、同じ組を 64・16 エポック反復した過適合の両方で説明できる。どちらが主因かは本トピックの設計では分離できない(宣言済みの交絡の再確認)。分離するには、エポック数を揃えて(計算量が組の数に比例する設計で)水準を振る実験が要る。

### 7.9 まとめと 018 への接続

- 016 の SFT をやり直した参照方策(完全一致率 0.818)の応答から、既知の真の報酬と Bradley-Terry モデルで選好データを作り、報酬モデルを学習した。報酬モデルの予測は真の報酬を大きく取りこぼしていた(予測の忠実度 0.31、順位相関 0.34〜0.50)が、報酬の差は、ラベルを生成した対数オッズの尺度に較正されていた(実験 A、較正の傾き 1.05)。
- best-of-n で最適化の圧力を強めると、真の報酬は $n = 4$〜8 付近で最大となり、その後は代理報酬が上がり続けるのに下がった(実験 B)。選好の組を増やすと、同じ強い圧力のもとでの真の報酬が上がり、過最適化が始まる $n$ が大きくなった(実験 C。組の数と反復回数の交絡を含む)。PPO でも、KL 係数が小さい条件では代理報酬が上がる一方で真の報酬が下がった。
- **018 への接続**: 018 の DPO(Direct Preference Optimization)は、`sft-synthetic`ブランチの参照方策、同じ真の報酬 $r^*$、同じ選好データの生成($\kappa$ と Bradley-Terry モデルによるラベルの抽選)をそのまま使える。3.4 節で導出した KL 正則化つきの目的関数の最適解を、報酬モデルを経由せずに学習する方法として、「真の報酬 対 KL ダイバージェンス」の平面で、本トピックの best-of-n(横軸は KL の上界)・PPO と比べることができる。


## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/04_alignment/017_reward_model_and_rlhf.ipynb
