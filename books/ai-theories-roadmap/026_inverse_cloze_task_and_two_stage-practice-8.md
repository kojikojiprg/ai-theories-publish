---
title: "Inverse Cloze Task と passage 単位の 2 段階検索 / Inverse Cloze Task and Two-Stage Retrieval at Passage Level(実装・実験編 8/8)"
---

この記事は後編(実装・実験編 8/8)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/026_inverse_cloze_task_and_two_stage-practice-7)。

## 注意

- 学習データは英語 Wikipedia の長大な記事に限られ、汎用の並べ替えモデルとして使えるものではない。
- 関連性は「query の文を抜いた元の passage ただ 1 つ」と定義した評価であり、同じ内容を含む別の passage は負例として扱われる。
'''


def write_embedding_files(directory: Path) -> dict:
    state = SAVED_STATES[("R2", UPLOAD_SEED)]
    torch.save(state, directory / "model_state.pt")
    config = dict(REFERENCE_CONFIG) | {
        "embedding_dimension": EMBEDDING_DIMENSION, "pooling": "mean", "attention": "causal", "matryoshka_dimensions": list(MATRYOSHKA_DIMENSIONS), "temperature": TEMPERATURE,
        "query_length": QUERY_LENGTH, "passage_length": PASSAGE_LENGTH, "variable_length_inputs": True, "training": "inverse_cloze_task", "keep_probability": KEEP_PROBABILITY,
        "tokenizer_repo": TOKENIZER_REPO_ID, "base_model_repo": REFERENCE_MODEL_REPO_ID, "source_notebook": "theories/07_retrieval/026_inverse_cloze_task_and_two_stage_retrieval.ipynb",
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    reloaded = build_embedding_model(0)
    reloaded.backbone.load_state_dict(torch.load(directory / "model_state.pt", map_location="cpu"))
    assert all(torch.equal(a, b) for a, b in zip(reloaded.backbone.state_dict().values(), state.values(), strict=True)), "書き出した重みが一致しない"
    reloaded = reloaded.to(device).eval()
    contexts, queries = embed_contexts(reloaded), embed_cloze_queries(reloaded, EVALUATION_ENTRIES)
    passage_ranks = rank_single_positive(queries, contexts, EVALUATION_ENTRIES)
    article = rank_article_label(queries, contexts, EVALUATION_ENTRIES, ENTRY_ARTICLES)
    format_queries, format_passages = embed_article_set(reloaded, "evaluation")
    format_024 = evaluate_embeddings(format_queries, format_passages, EVALUATION_SET)
    del reloaded, contexts, queries
    empty_device_cache()
    recorded = float(RUNS[("R2", UPLOAD_SEED)]["passage_hits"].mean())
    assert abs(float((passage_ranks <= 10).mean()) - recorded) <= 2 / len(EVALUATION_ENTRIES), (float((passage_ranks <= 10).mean()), recorded)  # 学習直後の値(記録)と、読み込み直した重みでの値が一致する
    card = EMBEDDING_CARD_TEMPLATE.format(
        branch=EMBEDDING_UPLOAD_BRANCH, repo_id=EMBEDDING_REPO_ID, matryoshka_dimensions=", ".join(str(m) for m in MATRYOSHKA_DIMENSIONS), num_steps=FIRST_STAGE_STEPS, batch_size=BATCH_SIZE,
        temperature=TEMPERATURE, learning_rate=LEARNING_RATE["R2"], weight_decay=WEIGHT_DECAY, clip=GRADIENT_CLIP_THRESHOLD, passage_length=PASSAGE_LENGTH, min_query=MIN_QUERY_TOKENS,
        max_removed=MAX_REMOVED_TOKENS, query_length=QUERY_LENGTH, keep_probability=KEEP_PROBABILITY, num_train_articles=f"{len(SPLIT.train):,}", num_queries=f"{len(EVALUATION_ENTRIES):,}",
        num_index=f"{NUM_INDEX:,}", recall_passage=float((passage_ranks <= 10).mean()), recall_passage_r1=float(RUNS[("R1", UPLOAD_SEED)]["passage_hits"].mean()),
        recall_article=float((article.rank <= 10).mean()), recall_article_r1=float(RUNS[("R1", UPLOAD_SEED)]["article_hits"].mean()), recall_format024=format_024.recall_at(10),
        recall_format024_r1=float(RUNS[("R1", UPLOAD_SEED)]["format024_hits"].mean()), articles_per_batch=ARTICLES_PER_BATCH, random_recall=RANDOM_RECALL_PASSAGE,
        smoke_note="動作確認の値であり、意味を持たない。" if SMOKE_TEST else "本番の実行の値である。",
    )
    (directory / "README.md").write_text(card, encoding="utf-8")
    return {"passage_recall": float((passage_ranks <= 10).mean()), "card": card, "files": sorted(p.name for p in directory.iterdir())}


def write_cross_encoder_files(directory: Path) -> dict:
    state = SAVED_STATES[("cross_encoder", UPLOAD_SEED)]
    torch.save(state, directory / "model_state.pt")
    config = dict(REFERENCE_CONFIG) | {
        "model_type": "cross_encoder", "separator_token_id": SEPARATOR_TOKEN_ID, "terminal_token_id": TERMINAL_TOKEN_ID,
        "head": "linear(d_model -> 1) applied to the final-normalized hidden state of the terminal position", "variable_length_inputs": True,
        "query_length": QUERY_LENGTH, "passage_length": PASSAGE_LENGTH, "rerank_candidates": RERANK_CANDIDATES, "tokenizer_repo": TOKENIZER_REPO_ID,
        "base_model_repo": REFERENCE_MODEL_REPO_ID, "first_stage_repo": EMBEDDING_REPO_ID, "first_stage_branch": EMBEDDING_UPLOAD_BRANCH,
        "source_notebook": "theories/07_retrieval/026_inverse_cloze_task_and_two_stage_retrieval.ipynb",
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    reloaded = build_cross_encoder(0)
    reloaded.load_state_dict(torch.load(directory / "model_state.pt", map_location="cpu"))
    assert all(torch.equal(a, b) for a, b in zip(reloaded.state_dict().values(), state.values(), strict=True)), "書き出した重みが一致しない"
    reloaded = reloaded.to(device)
    evaluated, scores = evaluate_rerank(reloaded, RERANK_EVALUATION_ENTRIES, MINED[UPLOAD_SEED]["evaluation"][RERANK_EVALUATION_POSITIONS])
    del reloaded
    empty_device_cache()
    assert abs(evaluated.recall_at(10) - float(CROSS_ENCODER_RUNS[UPLOAD_SEED]["hits"].mean())) <= 2 / len(RERANK_EVALUATION_ENTRIES), evaluated.recall_at(10)
    first = rerank_single_positive(
        np.broadcast_to(-np.arange(RERANK_CANDIDATES, dtype=np.float64), (len(RERANK_EVALUATION_ENTRIES), RERANK_CANDIDATES)), MINED[UPLOAD_SEED]["evaluation"][RERANK_EVALUATION_POSITIONS], RERANK_EVALUATION_ENTRIES, NUM_INDEX
    )
    card = CROSS_ENCODER_CARD_TEMPLATE.format(
        repo_id=CROSS_ENCODER_REPO_ID, rerank_candidates=RERANK_CANDIDATES, num_steps=CROSS_ENCODER_STEPS, batch_queries=BATCH_QUERIES, num_negatives=NUM_NEGATIVES, learning_rate=LEARNING_RATE["cross_encoder"],
        weight_decay=WEIGHT_DECAY, clip=GRADIENT_CLIP_THRESHOLD, query_length=QUERY_LENGTH, num_train_articles=f"{len(SPLIT.train):,}", separator_id=SEPARATOR_TOKEN_ID, terminal_id=TERMINAL_TOKEN_ID,
        num_queries=f"{len(RERANK_EVALUATION_ENTRIES):,}", num_index=f"{NUM_INDEX:,}", first_stage_recall=first.recall_at(10), recall=evaluated.recall_at(10), first_stage_mrr=first.mean_reciprocal_rank(),
        mrr=evaluated.mean_reciprocal_rank(), coverage=evaluated.coverage(), smoke_note="動作確認の値であり、意味を持たない。" if SMOKE_TEST else "本番の実行の値である。",
    )
    (directory / "README.md").write_text(card, encoding="utf-8")
    return {"recall": evaluated.recall_at(10), "first_stage_recall": first.recall_at(10), "card": card, "files": sorted(p.name for p in directory.iterdir())}


def get_hub_token():
    try:
        from google.colab import userdata  # noqa: E402

        return userdata.get("HF_TOKEN")
    except Exception:  # noqa: BLE001  # Colab 以外・Secrets 未設定・取得失敗のいずれでも、停止せずスキップする
        return None


with tempfile.TemporaryDirectory() as _embedding_dir, tempfile.TemporaryDirectory() as _cross_encoder_dir:
    _embedding = write_embedding_files(Path(_embedding_dir))
    _cross_encoder = write_cross_encoder_files(Path(_cross_encoder_dir))
    assert _embedding["files"] == ["README.md", "config.json", "model_state.pt"] and _cross_encoder["files"] == ["README.md", "config.json", "model_state.pt"], "トークナイザなど、同梱しないファイルが含まれている"
    print(f"{RUN_TAG}モデルカードの下書きと書き出したファイル {_embedding['files']} を作成し、重みを読み込み直して評価した")
    print("--- R2 のモデルカードの先頭 ---")
    print("\n".join(_embedding["card"].splitlines()[:14]))
    print("--- cross-encoder のモデルカードの先頭 ---")
    print("\n".join(_cross_encoder["card"].splitlines()[:14]))
    # 公開の条件(判定ではない)
    EMBEDDING_CONDITION = bool(_embedding["passage_recall"] > float(RUNS[("R1", UPLOAD_SEED)]["passage_hits"].mean()))
    CROSS_ENCODER_CONDITION = bool(_cross_encoder["recall"] > _cross_encoder["first_stage_recall"])
    print(f"{RUN_TAG}公開の条件(判定ではない): R2 のシード 0(読み込み直した重み)の passage 単位の Recall@10 {_embedding['passage_recall']:.4f} > R1 のシード 0 の {float(RUNS[('R1', UPLOAD_SEED)]['passage_hits'].mean()):.4f}: {EMBEDDING_CONDITION}")
    print(f"{RUN_TAG}公開の条件(判定ではない): cross-encoder のシード 0(読み込み直した重み)の並べ替えの Recall@10 {_cross_encoder['recall']:.4f} > 並べ替えなし {_cross_encoder['first_stage_recall']:.4f}: {CROSS_ENCODER_CONDITION}")
    if not UPLOAD_ARTIFACTS:
        print("UPLOAD_ARTIFACTS = False のため、アップロードは行わない(モデルカードの下書きの確認のみ)")
    elif not (EMBEDDING_CONDITION or CROSS_ENCODER_CONDITION):
        print("どちらのモデルも公開の条件を満たさないため、アップロードを行わない")
    else:
        _token = get_hub_token()
        if not _token:
            print("Colab Secrets から HF_TOKEN を取得できないため、アップロードをスキップする")
        else:
            from huggingface_hub import HfApi

            _api = HfApi(token=_token)
            if EMBEDDING_CONDITION:
                _api.create_branch(EMBEDDING_REPO_ID, branch=EMBEDDING_UPLOAD_BRANCH, repo_type="model", exist_ok=True)
                for _name in _embedding["files"]:
                    _api.upload_file(path_or_fileobj=str(Path(_embedding_dir) / _name), path_in_repo=_name, repo_id=EMBEDDING_REPO_ID, repo_type="model", revision=EMBEDDING_UPLOAD_BRANCH)
                _remote = sorted(_api.list_repo_files(EMBEDDING_REPO_ID, repo_type="model", revision=EMBEDDING_UPLOAD_BRANCH))
                assert not [n for n in _embedding["files"] if n not in _remote], f"アップロードしたはずのファイルがリポジトリにない: {_remote}"
                print(f"アップロード完了: {EMBEDDING_REPO_ID}(ブランチ {EMBEDDING_UPLOAD_BRANCH})のファイル {_remote}(意図したファイルがすべて存在することを list_repo_files() で確認)")
            else:
                print("R2 のシード 0 は公開の条件を満たさないため、アップロードしない")
            if CROSS_ENCODER_CONDITION:
                _api.create_repo(CROSS_ENCODER_REPO_ID, repo_type="model", exist_ok=True, private=False)
                for _name in _cross_encoder["files"]:
                    _api.upload_file(path_or_fileobj=str(Path(_cross_encoder_dir) / _name), path_in_repo=_name, repo_id=CROSS_ENCODER_REPO_ID, repo_type="model")
                _remote = sorted(_api.list_repo_files(CROSS_ENCODER_REPO_ID, repo_type="model"))
                assert not [n for n in _cross_encoder["files"] if n not in _remote], f"アップロードしたはずのファイルがリポジトリにない: {_remote}"
                print(f"アップロード完了: {CROSS_ENCODER_REPO_ID} のファイル {_remote}(意図したファイルがすべて存在することを list_repo_files() で確認)")
            else:
                print("cross-encoder のシード 0 は公開の条件を満たさないため、アップロードしない")
print(f"アップロードの準備 {time.time() - _t0_upload:.1f} 秒")
```

    モデルカードの下書きと書き出したファイル ['README.md', 'config.json', 'model_state.pt'] を作成し、重みを読み込み直して評価した
    --- R2 のモデルカードの先頭 ---
    ---
    language: en
    license: mit
    tags:
    - ai-theories
    - text-embedding
    - retrieval
    - inverse-cloze-task
    - matryoshka-representation-learning
    - scratch-implementation
    ---
    
    # ai-theories テキスト埋め込みモデル(英語、Inverse Cloze Task、`inverse-cloze-task` ブランチ)
    
    --- cross-encoder のモデルカードの先頭 ---
    ---
    language: en
    license: mit
    tags:
    - ai-theories
    - cross-encoder
    - reranking
    - retrieval
    - scratch-implementation
    ---
    
    # ai-theories cross-encoder(英語、passage 単位の並べ替えモデル)
    
    `ai-theories`(https://github.com/kojikojiprg/ai-theories)プロジェクトの成果物。008 の小型 GPT(`kojikojiprg/ai-theories-small-gpt-en`)を起点に、query と passage を連結して符号化し、
    公開の条件(判定ではない): R2 のシード 0(読み込み直した重み)の passage 単位の Recall@10 0.1348 > R1 のシード 0 の 0.0964: True
    公開の条件(判定ではない): cross-encoder のシード 0(読み込み直した重み)の並べ替えの Recall@10 0.2268 > 並べ替えなし 0.1348: True



    Processing Files (0 / 0)      : |          |  0.00B /  0.00B            



    New Data Upload               : |          |  0.00B /  0.00B            



      ...mpf7ttb3kd/model_state.pt:   3%|2         |  777kB / 29.4MB            


    アップロード完了: kojikojiprg/ai-theories-text-embedding-en(ブランチ inverse-cloze-task)のファイル ['.gitattributes', 'README.md', 'config.json', 'model_state.pt'](意図したファイルがすべて存在することを list_repo_files() で確認)



    Processing Files (0 / 0)      : |          |  0.00B /  0.00B            



    New Data Upload               : |          |  0.00B /  0.00B            



      ...mp30c0y7mp/model_state.pt:   3%|2         |  780kB / 29.4MB            


    アップロード完了: kojikojiprg/ai-theories-cross-encoder-en のファイル ['.gitattributes', 'README.md', 'config.json', 'model_state.pt'](意図したファイルがすべて存在することを list_repo_files() で確認)
    アップロードの準備 110.5 秒


## 7. 結果・考察 / Results and Discussion

この章の数値は、すべて本番のセル出力(括弧内に出力のある節を示す)による。7.1〜7.10 は、事前に宣言した判定基準による結論と、その周辺の記録である。結果を見た後に立てた解釈は **7.11 節にまとめ、検証済みの結論として扱わない。**

### 7.1 実行の概要

- **実行環境**(5.1 節の出力): Google Colab、GPU は Tesla T4(compute capability 7.5、総メモリ 14.56 GiB)、Python 3.13.15、torch 2.13.0+cu130(ビルド時の CUDA 13.0、cuDNN 92000)、Linux-6.6.122+。リポジトリのコミットは`47f25ba`、追跡しているファイルに未コミットの変更はなく、必要な祖先のコミット`90b3dde`の子孫であることを確認した。実行日時は 2026-10-14T04:11:03 UTC。学習の精度は FP16 の autocast と動的損失スケーリング(埋め込みの正規化・損失・評価は FP32)。
- **選ばれた計画**(6.4 節): 計画 0(削らない)。経過時間 15.4 分(データの準備 379.1 秒、スケーリングの計測 160.5 秒、パイロット 5.3 分を含む)の時点で、計画 0 の見積もりは残り 71.3 分、終了の見込み 86.7 分で、予算 120 分に収まった。第 1 段のシード数 5(実験 A・B)、cross-encoder のシード数 5(実験 C)、R1 と cross-encoder の学習率の較正あり、観察 D の索引の件数 16,384。**規則で学習率を決める段階(段階 4 以降)は使われなかったので、規則で決めた学習率による偏りはない。**
- **学習量**: 第 1 段は $T_1 = 1024$(1 ステップ 64 組。組を作れる学習用の記事 141,064,514 トークンに対して 0.059 エポック)。cross-encoder の学習ステップ数 $T_2$ は、パイロットで 3 つの条件がすべて成立した最小の 2048 が選ばれた(6.3 節)。$T_2 = 2048$ は 1 シードあたり 16,384 個の query を見て、学習に使える query 約 20,700 個に対して 0.78〜0.80 エポックにあたる。
- **パイロット**(6.3 節、シード 90): 第 1 段の R2(学習率 $4.8 \times 10^{-4}$)は、最後の区間の訓練損失 1.959(同じ組での学習前のモデルの損失 3.571、比 0.548)、検証用の Recall@10 は passage 単位のラベル 0.5108・記事単位のラベル 0.7028。第 1 段の上位 50 件に正例が入る割合(coverage)は、学習用 0.2305・検証用 0.2333・評価用 0.2312。cross-encoder($T_2 = 2048$、学習率 $2.4 \times 10^{-4}$)は、最後の区間の訓練損失 0.4307(学習前 2.1115)、検証用の query の並べ替えの Recall@10 0.2264(学習前 0.0394、増分 +0.1870、標準偏差 0.0168)で、学習の成立・増分が標準偏差の 2 倍以上・R2 だけ(0.1339)を上回る、の 3 条件がすべて成立した。
- **学習率**(6.5 節、較正専用のシード 90): R2 は $4.8 \times 10^{-4}$(内点。指標 0.5586・0.6068・0.5556)、R1 は $4.8 \times 10^{-4}$(内点。指標 0.5315・0.5605・0.4734)、cross-encoder は $9.6 \times 10^{-4}$(格子の上端を拡張した後の内点。検証用の並べ替えの Recall@10 0.2156・0.2264・0.2126・0.2018)。cross-encoder は、最良の $2.4 \times 10^{-4}$(0.2264)から標準偏差(0.0173〜0.0186)以内の候補のうち、学習の成立の余裕が最大の点を宣言した規則で選んだので、検証用の Recall@10 が最良の点より低い学習率(0.2126)が選ばれた。較正の P0 は実験 A・B・C のすべてで成立した。
- **判定の一覧**(6.13 節): 実験 A は支持、実験 B は反証、実験 C は前提不成立(P3 の不成立)。詳細は 7.2〜7.5 節。

### 7.2 前提条件

| 実験 | P0(較正) | P1(学習の成立) | P2(天井と床、余地) | P3(第 1 段の候補) |
|---|---|---|---|---|
| A | 成立(R1・R2 とも内点) | 成立 | 成立(R1 の passage 単位のラベル 0.0980 が 0.02〜0.95) | なし |
| B | 成立(R1・R2 とも内点) | 成立 | 成立(R1 の記事単位のラベル 0.1518 が 0.02〜0.95) | なし |
| C | 成立(R2・cross-encoder とも内点) | 成立 | 成立(R2 だけ 0.1304 が coverage 0.2260 − 0.05 以下) | **不成立** |

- **P1(実験 A・B)**: 最後の区間の訓練損失と、同じ組での学習前のモデルの損失の比の最大は 0.588(閾値 0.9 以下)。自分の定義の検証用の Recall@10 の学習前の 008 からの増分の最小は、R1(記事単位のラベル)・R2(passage 単位のラベル)とも +0.1398(閾値 0.04 以上)。
- **P1(実験 C)**: cross-encoder の全シードで、最後の区間の訓練損失が学習前のモデルの損失より標準誤差の 42.9〜69.8 倍低く(閾値 2 倍以上)、一様な予測の損失 2.0794 を下回った。検証用の query の並べ替えの Recall@10 の増分のシード平均は +0.1939(シードごとに +0.1949・+0.2037・+0.1772・+0.2057・+0.1880)、標準偏差 $\sigma_g = 0.0157$(シード間 0.0053・ブートストラップ 0.0148)で、増分は標準偏差の 12.3 倍(閾値 2 倍)だった。
- **P3(実験 C)**: 第 1 段の上位 50 件の coverage は 0.2260(閾値 0.35 以上)、coverage − $R(\mathrm{R2})$ は 0.0956(閾値 0.10 以上)で、**2 つの条件がどちらも成立しなかった。** 実験 C は仮説の判定に使わず、「前提不成立」として記録する(7.5 節)。
- **前提条件の閾値の下限と、想定される増分との対応**(6.1 節の「前提条件の閾値の下限と、想定される増分」): 6.1 節は、cross-encoder の P1(b)の閾値(標準偏差の 2 倍)の下限を約 0.014〜0.035 と見積もっていた。本番の閾値は $2 \times 0.0157 = 0.0314$ でこの範囲に入り、ブートストラップの項(0.0148)が標準偏差を支配した。増分 +0.1939 は、閾値の上端 0.035 の約 5.5 倍だった。
- **検出力の事前確認との対応**(6.1 節の「検出力の事前確認」): 対比量の標準偏差の見込みは $\sigma \approx 0.005$〜$0.010$(閾値 約 0.01〜0.02、最大で約 0.03)だった。本番の標準偏差は実験 A 0.0041(シード間 0.0027・ブートストラップ 0.0031)、実験 B 0.0036(0.0017・0.0032)、実験 C 0.0042(0.0012・0.0040)で、見込みの下限をやや下回り、閾値は 0.0082・0.0073・0.0084 だった。実験 B と C では、ブートストラップの項(評価の query が有限であることによる変動)がシード間の項より大きかった。観測された対比量の大きさは、実験 A が閾値の約 4.0 倍、実験 B が約 2.3 倍、実験 C が約 10.8 倍だった。

### 7.3 実験 A: passage 単位の評価での、関連性の定義の効果

- **判定**(6.8 節): passage 単位のラベルの Recall@10(評価用の query 3,227 個、索引 93,995 passage)は、R1 が 0.0964・0.0970・0.0964・0.1001・0.1004(平均 0.0980)、R2 が 0.1348・0.1236・0.1320・0.1360・0.1255(平均 0.1304)。シードごとの差 $d_s = R_s(\mathrm{R2}) - R_s(\mathrm{R1})$ は 5 シードすべてで正(+0.0384・+0.0267・+0.0356・+0.0359・+0.0251)。
  対比量は $\Delta_A = +0.0324$、標準偏差は $\sigma_A = 0.0041$(シード間 0.0027・ブートストラップ 0.0031、反復 10,000 回)、閾値 $2\sigma_A = 0.0082$。**$\Delta_A > 2\sigma_A$ なので、最終判定は支持。** 前提条件の P0・P1・P2 はすべて成立した。
- **参照**: 学習前の 008 の値は 0.0762、ランダムな順位の期待値は 0.000106。R1 は学習前より +0.0218、R2 は +0.0542 高い。R2 の平均は R1 の約 1.33 倍。
- **作用点に近い診断量**(判定には使わない):

| 診断量 | R1 | R2 |
|---|---|---|
| 上位 10 件のうち、正例と同じ記事の passage の割合 | 0.0403 | 0.0477 |
| 1 位が、同じ記事の別の passage である query の割合 | 0.0474 | 0.0554 |
| 平均逆順位(passage 単位のラベル) | 0.0531 | 0.0740 |
| Recall@10 のシード間の標準偏差 | 0.0020 | 0.0056 |
| 最後の区間の訓練損失(シード平均) | 2.766 | 2.000 |

  R2 は、上位 10 件に同じ記事の passage を入れる割合も、1 位を同じ記事の別の passage にする割合も R1 より高かった。同じ記事の他の passage をより多く上位に出しながら、正例を上位 10 件に入れる割合も高く、平均逆順位も高い。
- **切り詰めた query の割合**: 評価の索引で 4.5%(抜く文を一様に選ぶと 31.9%)、学習の組で 4.2〜4.4%(R1 と R2 は同じ組を見る)。
- **宣言していた偏りの向き(解釈の制約)**: R2 は、in-batch negatives に入れた同じ記事の組を負例として数え、同じ記事の困難な負例を見る。これは「関連性の定義を、正例と in-batch の負例の両方に一貫して当てはめる」という介入の一部で、$\Delta_A$ を正の向きに大きくしうる。一方、R1 は同じ記事の組を負例から除くので、1 行あたりの負例が 62 個で、R2 の 63 個より 1 つ少ない(6.6 節の出力: 62.0 と 63.0)。実験 A の対比量には、正例の対応の違いと、この同じ記事の組の負例としての扱いの違いの両方が含まれ、**本番の設計ではこの 2 つを分けられない。**
- 計画 0 だったので、規則で決めた R1 の学習率による偏り(仮説に有利な向き)はない。

### 7.4 実験 B: 記事単位の評価での、関連性の定義の効果

- **判定**(6.9 節): 同じ索引・同じ query での、記事単位のラベル(正解は同じ記事の他の passage)の Recall@10 は、R1 が 0.1546・0.1518・0.1481・0.1518・0.1528(平均 0.1518)、R2 が 0.1661・0.1726・0.1624・0.1695・0.1726(平均 0.1686)。シードごとの差 $d_s = R_s(\mathrm{R1}) - R_s(\mathrm{R2})$ は 5 シードすべてで負(−0.0115・−0.0208・−0.0143・−0.0177・−0.0198)。
  対比量は $\Delta_B = -0.0168$、$\sigma_B = 0.0036$(シード間 0.0017・ブートストラップ 0.0032)、閾値 $2\sigma_B = 0.0073$。**$\Delta_B < -2\sigma_B$ なので、最終判定は反証。** すなわち、R1 が R2 より高いという仮説に対して、逆の向きに閾値を超える差が現れた。前提条件の P0・P1・P2 はすべて成立した。
- **参照**: 学習前の 008 の記事単位のラベルの値は 0.1054。R1 は学習前より +0.0464、R2 は +0.0632 高い。ランダムな順位の期待値は 0.00112。
- **診断量**(判定には使わない): 記事単位のラベルの平均逆順位は R1 0.0888・R2 0.1026、正規化割引累積利得(上位 10 件)は R1 0.0376・R2 0.0436。Recall@10 のシード間の標準偏差は R1 0.0024・R2 0.0044。
- **024 と同じ形式の記事単位の評価**(024 の評価用の索引と query、固定長、評価用の 3,244 query): R1 0.6125、R2 0.6363、024 の Hub の重み(C5 のシード 0)0.6477(024 のモデルカードの値と一致)、学習前の 008 0.4834。R1 と R2 の差(R1 − R2)のシードごとの値は −0.0133・−0.0281・−0.0373・−0.0213・−0.0191 で、5 シードすべてで R1 が低い。この評価は判定には使わない。
- **解釈の制約**: 実験 A の解釈の制約として宣言した、R2 が同じ記事の困難な負例を見ること、R1 の負例が 1 行あたり 1 個少ないこと(62 個と 63 個)は、実験 B でも同じ条件の違いとして含まれる。本番の設計では、正例の対応の違いと同じ記事の組の負例としての扱いの違いを分けて測っていない(7.11 節)。計画 0 だったので、規則で決めた学習率による偏りはない。

### 7.5 実験 C: 2 段階の検索の効果

- **最終判定は前提不成立**(6.10 節)。P3(第 1 段の上位 50 件の coverage が 0.35 以上、かつ coverage − $R(\mathrm{R2})$ が 0.10 以上)が成立しなかった。coverage は 0.2260、coverage − $R(\mathrm{R2})$ は 0.0956 だった。実験 C は、支持・反証・判定不能のいずれとも判定しない。前提不成立は仮説に関する情報を与えないので、以下の値は結論ではなく、判定関数の参考値と診断量として記録する。
- **判定関数の参考値**(結論として読まない): 評価用の query 3,227 個での Recall@10 は、R2 だけが 0.1348・0.1236・0.1320・0.1360・0.1255(平均 0.1304)、R2 に cross-encoder の並べ替えを加えた構成(R3)が 0.2268・0.2135・0.2197・0.2240・0.2194(平均 0.2207)。シードごとの差は +0.0920・+0.0899・+0.0877・+0.0880・+0.0939。$\Delta_C = +0.0903$、$\sigma_C = 0.0042$(シード間 0.0012・ブートストラップ 0.0040)、閾値 $2\sigma_C = 0.0084$。判定関数の結果は「支持」だったが、前提不成立のため最終判定には使わない。
- **診断量**(判定には使わない):
  - 第 1 段 R2 だけ(索引全体の順位): 平均逆順位 0.0740、coverage 0.2260(シードごとに 0.2315・0.2216・0.2250・0.2281・0.2240)。
  - R3: 平均逆順位 0.1406、正規化割引累積利得(上位 10 件)0.1602、Recall@10 のシード間の標準偏差 0.0051。
  - cross-encoder の最後の区間の訓練損失は 0.348(一様な予測の損失は 2.079)。候補 50 件のスコアの標準偏差の中央値は 3.6727・3.4640・3.0315・3.7827・3.3261 で、出力が定数に崩れてはいない。
  - 並べ替えによる Recall@10 の変化は +0.0903。これは cross-encoder の追加の学習量(1 シードあたり 8 query × 8 候補 × 2048 ステップ)と交絡しており、実用の問いとして読む。1 query あたりの順伝播の回数は 50 回。
  - 学習に使える query は、シードごとに 20,601〜20,917 個(学習用の query 89,752 個の約 23%)。
- 実用の問いとしての範囲: この構成では、追加の順伝播 50 回(1 query あたり)で Recall@10 は R2 だけの 0.1304 から 0.2207 に上がった。ただし、これは前提不成立の実験の参考値であり、「アプリで並べ替えを足す価値がある」という結論としては扱わない。

### 7.6 実験 A と B の組み合わせ

- 事前に宣言した読み方の表(6.1 節)のうち、実験 A が **支持**、実験 B が **反証** の行・列にあたる。宣言した読み方は次のとおりである: **R2 が両方のラベルで優れる。定義の交差ではなく、passage 単位の定義の全面的な優位。**
- 入力の形式・索引・query は両条件で同じなので、この結果は入力の形式の一致の効果ではなく、関連性の定義(正例の対応と、in-batch の負例への一貫した当てはめ)の効果として読む。ただし、介入が正例の対応と同じ記事の組の負例としての扱いの 2 つを含むため、どちらが効いたかは分けられない(7.3 節)。
- 計画 0 だったので、規則で決めた学習率による割り引きはない。「各定義で学習した検索器が、その定義のラベルで優れる」という交差は、本番の結果では観察されなかった。

### 7.7 観察 D・E

**観察 D**(6.11 節。判定なし)。R2 のシード 0 の埋め込みの部分集合 16,384 passage、query 600 個、cross-encoder のシード 0 で並べ替え。HNSW(Hierarchical Navigable Small World)の構築は 53.8 秒($M = 5$、efConstruction = 100)、探索の幅は、全探索の上位 100 件との一致率が 0.9 に初めて届いた 160(一致率 0.901)。部分集合の値であり、実験 C の絶対値(索引全体 93,995 passage)とは比べない。

| 第 1 段 | 1 query あたりの距離計算 | $K$ | 第 2 段の順伝播 | coverage | 並べ替えなしの Recall@10 | 並べ替えた Recall@10 |
|---|---|---|---|---|---|---|
| 全探索 | 16,384 | 10 | 10 | 0.2200 | 0.2200 | 0.2200 |
| 全探索 | 16,384 | 20 | 20 | 0.2817 | 0.2200 | 0.2800 |
| 全探索 | 16,384 | 50 | 50 | 0.3733 | 0.2200 | 0.3667 |
| 全探索 | 16,384 | 100 | 100 | 0.4583 | 0.2200 | 0.4333 |
| HNSW | 736 | 10 | 10 | 0.2117 | 0.2117 | 0.2117 |
| HNSW | 736 | 20 | 20 | 0.2733 | 0.2117 | 0.2717 |
| HNSW | 736 | 50 | 50 | 0.3650 | 0.2117 | 0.3567 |
| HNSW | 736 | 100 | 100 | 0.4450 | 0.2117 | 0.4217 |

- $K$ を増やすと、coverage も並べ替えた後の Recall@10 も上がった(全探索で $K = 10 \to 100$ のとき 0.2200 → 0.4333)。並べ替えた Recall@10 は coverage の 99.4%($K = 20$ で全探索 0.2800 / 0.2817)から 94.5%($K = 100$ で 0.4333 / 0.4583)まで下がり、$K$ が大きいほど coverage との差が開いた。$K = 10$ は上位 10 件の中の並べ替えなので、Recall@10 は変わらない。
- HNSW は全探索より第 1 段の距離計算が約 4.5% で済み(736 / 16,384)、coverage と Recall@10 は全探索より 0.008〜0.013 低かった。索引の件数が小さい部分集合での値で、索引が大きいときの HNSW の距離計算の割合は別である。
- cross-encoder は $K = 50$ の候補で学習しているので、$K = 100$ の候補は学習時の分布の外にある。これは観察であり、$K$ の選択についての結論としては扱わない。

**観察 E**(判定なし。評価用の query のうち R2 だけが正例を上位 10 件に入れた 2 件と、R1 だけが入れた 1 件。シード 0)。表示は passage の先頭 60 文字程度で、選び方は評価用の query の並び順の先頭から機械的に決めた。

- 「Starring Jodie Whittaker as the Thirteenth Doctor.」: 正例は Doctor Who の別の主演の記述を含む passage。R2 は正例を 2 位、R1 は 154 位。R1 の上位は別のテレビ番組の出演者の一覧だった。
- 「It was released on DVD in Region 2 on 12 November 2007, ...」: 正例は Dreamland の放送の記述。R2 は 1 位、R1 は 33 位。R1 の上位は日付と月の表記(「5 June」など)を含む別の話題の passage だった。R2 の 2 位・3 位にも日付を含む別の話題の passage が入っていた。
- 「A limited edition Blu-ray box-set containing Series 1-7 of Doctor Who ...」: 正例は Doctor Who の Blu-ray の記述。R1 は 1 位、R2 は 163 位。R2 の上位に音楽のチャートの記述が入っていた。
- 3 件だけの例で、観察である。日付やリリースの表現といった表層の語を含む passage が上位に入った例が見られる。検証済みの結論ではない。

### 7.8 診断量と R1 の再現の確認

- **024 と同じ形式の記事単位の評価**(6.6 節): R1 のシード平均 0.6125(シード 0 は 0.6205)、R2 のシード平均 0.6363、024 の Hub の重み 0.6477、学習前の 008 0.4834。R1 のシード 0 は 024 の Hub の重みより 0.0271 低く、024 の C5 の記事単位の Recall@10 のシード間の標準偏差の目安(0.002〜0.009)と比べると大きい。本トピックの R1 は、024 の C5 と入力の形式(query は抜いた文、passage は文を除いた形)・正例の作り方・バッチ(同じ記事の組を含む)が違うので、ビット単位の一致も 024 の値との一致も期待しておらず、これは再現の失敗ではなく診断量の記録である。R2 のシード平均は Hub の重みより 0.0114 低かった。
- **学習時と推論時の分布の違いの検出**(6.10 節): 第 1 段 R2 の上位 50 件に正例が入る割合(coverage)の、学習用・検証用・評価用のシード平均は 0.2314・0.2406・0.2260。学習用の query は、第 1 段が学習で見た passage から作られるが、学習用の値は評価用と近く(+0.0054)、第 1 段が学習用の passage を覚えていることによる顕著な差は見られなかった($T_1 = 1024$ は 0.059 エポック)。検証用が評価用より高い(+0.0146)のは、記事の数(100 と 300)が異なるためと思われるが、この差の原因は確かめていない。

### 7.9 実行時間と見積もりの対応

- 全体の経過時間は、不変条件の確認(6.12 節)の時点で 73.8 分。選択時の見積もり(計画 0)は終了の見込み 86.7 分(経過時間 15.4 分 + 残り 71.3 分)で、実測は残りの時間が約 58.4 分、見積もりの約 82% だった。予算 120 分に対しては、アップロードの準備(110.5 秒)を加えても約 76 分で収まった。
- 処理ごとの対応: 第 1 段の較正 15.4 分(最悪の場合の見積もり 20.6 分)、第 1 段の本番 21.5 分(見積もり 22.1 分)、cross-encoder の本番 19.6 分(見積もり 21.0 分)、観察 D・E 1.6 分(見積もり 2.4 分)、パイロット 5.3 分、学習の組の事前生成 5.0 秒(見積もり 4.9 秒、1 ステップあたり 0.79〜0.83 ms)。学習の時間は、安全係数 1.1 を掛けた見積もりをやや下回った。
- 6.2 節の計測: 1 ステップの時間は R1 76.0 ms・R2 75.5 ms・cross-encoder 73.6 ms、べき指数は 1.001・0.994・1.005 で、ステップ数にほぼ比例した。バッチ 32 に対するバッチ 64 の 1 ステップの時間の比は 1.53(2 に近ければバッチサイズに比例、1 に近ければ固定費が支配的)。HNSW の構築のべき指数は 1.101。データの準備は 379.1 秒で、記事の符号化と検証が 332.0 秒を占めた(符号化のキャッシュは新規作成)。
- 学習の組の生成のべき指数は 0.907(最初の呼び出しは、記事のすべての窓から選ぶ処理の走査 1.186 秒を含む)で、見積もりに使ったステップ数への比例と矛盾しない。

### 7.10 Hugging Face Hub へのアップロード

- **公開の条件**(判定ではない。読み込み直した重みで評価した値): R2 のシード 0 の passage 単位のラベルの Recall@10 0.1348 は、R1 のシード 0 の 0.0964 を上回った。cross-encoder のシード 0 の並べ替えた Recall@10 0.2268 は、並べ替えなし 0.1348 を上回った。どちらも条件を満たしたので、アップロードを実行した。
- **アップロード先と確認**(`list_repo_files()`で意図したファイルの存在を確認した): R2 のシード 0 は`kojikojiprg/ai-theories-text-embedding-en`のブランチ`inverse-cloze-task`(`README.md`・`config.json`・`model_state.pt`、トークナイザは同梱しない)、cross-encoder のシード 0 は新しいリポジトリ`kojikojiprg/ai-theories-cross-encoder-en`(同じ 3 ファイル)。
- **制約**: どちらのモデルも、passage から抜いた 1 文を query にして学習している。利用者が書く質問の形とは異なり、後続のアプリでの性能は確かめていない。cross-encoder は、第 1 段の R2 の上位 50 件を並べ替える使い方で学習しており、別の第 1 段の候補への適用は確かめていない。実験 C は前提不成立なので、「並べ替えを足す価値がある」という結論は得ていない。

### 7.11 事後的な解釈(検証済みの結論ではない)

この節の内容は、結果を見た後に立てた解釈である。事前に宣言した判定基準による結論(7.3〜7.6 節)とは分けて読み、検証済みの結論として扱わない。

#### 7.11.1 実験 B が反証になった理由の考察

実験 B の仮説は「記事単位のラベルの評価では R1 が R2 より高い」だった。結果は逆に、R2 が閾値を超えて高かった(7.4 節)。

**原因の候補と根拠**:

1. **R2 だけが同じ記事の困難な負例を見ること**(宣言していた偏り)。R2 は、同じバッチの同じ記事の passage を負例として数える。R1 は同じ記事の passage を負例から除く。一般に同じ記事の passage は話題が近いので、それを区別する学習は、局所的な内容の手がかりを増やす可能性がある。
   - 根拠: 1 行あたりの負例は R1 が 62 個、R2 が 63 個で、同じ記事の passage の負例としての扱いの違いは 1 行あたり 1 個である(6.6 節)。ただし、この 1 個が学習に及ぼす影響の大きさを示す値は、本番の出力にない。**根拠となる数値のない候補。** 診断量(7.3 節)では、R2 の上位 10 件の同じ記事の passage の割合(0.0477)が R1(0.0403)より高く、R2 が同じ記事の passage を上位に出す割合は減っていない。「同じ記事の passage を遠ざける」だけでは、この向きは説明しにくい。
2. **R1 の学習の信号の弱さ**。R1 の正例は同じ記事の別の窓の文脈で、query の文と内容の関係が薄い場合がある。R2 の正例は query の文を抜いた窓の文脈なので、内容の関係が強い。
   - 根拠: 最後の区間の訓練損失は R1 が 2.766、R2 が 2.000(シード平均)。同じ組での学習前のモデルの損失との比は R1 が 0.578〜0.588、R2 が 0.560〜0.568 で、学習前からの低下の割合は R1 の方がやや小さい。R1 の損失の絶対値が高いこと自体は、正例の内容の関係の弱さと整合する。ただし、R1 の学習前の損失(4.69〜4.85)が R2(3.48〜3.58)より高いこととも整合し、因果は分けられない。
3. **学習量が足りず、より難しい目的の R1 が追いついていない可能性**。
   - 根拠: $T_1 = 1024$ は 0.059 エポックで、学習データのごく一部しか見ていない(5.3 節)。R1 の訓練損失は R2 より高いまま終わっている。学習の途中の曲線を数値として比べる値は、本番の出力にない。**根拠となる数値のない候補。**
4. **記事単位の評価でも、query の文を取り出した passage の近くの内容が手がかりになる可能性**。評価の索引では、query の文を抜いた passage も、同じ記事の他の passage も「1 文を除いた形」で、同じ記事の他の passage は、同じ固有名詞や語を含むことが多い。そのため、passage 単位の学習で身につく語の一致が、記事単位のラベルでも有利に働く可能性がある。
   - 根拠: R2 は記事単位のラベルでも学習前の 008 より +0.0632 高く、R1 の +0.0464 を上回った(7.4 節)。一方、024 と同じ形式の評価(query は固定長の窓、passage も固定長の窓で、1 文を除いた形ではない)でも、R2(0.6363)は R1(0.6125)より 5 シードすべてで高かった。評価の形式が「1 文を除いた形」に依存した手がかりだけでは、この差は説明しにくい。

**確かめる方法**(実行はしない。1 つの追加の実験で 1 つの検証事項を扱う):

- 正例の対応と、同じ記事の組の負例としての扱いを分ける 2 × 2 の条件。例えば、(a)R2 の正例で、同じ記事の組を負例から除く条件、(b)R1 の正例で、同じ記事の組を負例として数える条件。(a)が R2 に近ければ負例の扱いは寄与が小さく、(b)が R2 に近ければ負例の扱いが主因である。
- R2 で同じ記事の組をバッチに入れない条件(第 1 段階の当初の設計)。同じ記事の困難な負例が、R2 の優位に効いているかを見る。
- 学習ステップ数を増やした条件($T_1$ を数倍にして、R1 の訓練損失と記事単位のラベルの Recall@10 が R2 に追いつくかを見る)。学習の途中(初期に密な位置)の評価を併せて記録する。
- R1 の正例を、query の文に近い内容の窓に限る条件(正例の内容の関係の強さを揃える)。

**教訓**:

- 「関連性の定義を正例と in-batch の負例の両方に一貫して当てはめる」は、2 つの介入を一緒にしている。各介入の寄与を分けたい場合は、最初から 2 × 2 の条件にする。
- 「各定義で学習した検索器が、その定義のラベルで優れる」という交差を仮説にする場合、一方の定義の学習の信号が他方より強い(損失の絶対値が大きく違う)ときは、定義の違いと信号の強さの違いが交絡する。信号の強さを近づけるか、診断量として併記する。

#### 7.11.2 実験 C が前提不成立になった理由の考察

実験 C の事後的な解釈は、次の 2 つの問いを別々に扱う。

**問い 1: なぜ前提(P3)が崩れたか**

- P3 の閾値は、025 の記事単位のラベル(正例が 1 query あたり平均 1.18 個)の第 1 段の値(coverage 0.4094、並べ替えなしの Recall@10 0.2199)を目安に置いた。passage 単位のラベルは、正例が query の passage 1 つだけで、同じ種類の閾値が適切とは限らなかった。閾値は 025 の値を引き継いだままで、passage 単位で決め直していなかった。
- 根拠: 第 1 段 R2 の coverage は、本番の前のパイロットの出力(6.3 節)の時点で、学習用 0.2305・検証用 0.2333・評価用 0.2312 と、閾値 0.35 を下回っていた。この値は対比量の向きの情報を含まない量で、パイロットの時点で P3 を確かめられた。本番の coverage 0.2260 は、パイロットの値と近い。coverage − $R(\mathrm{R2})$ の 0.0956 は閾値 0.10 に 0.0044 届かなかった。
- 候補としては、(i)閾値の目安を決めた設計の問題(別の定義の値を引き継いだ)、(ii)第 1 段の学習量($T_1 = 1024$、0.059 エポック)が小さく coverage が低い、の 2 つがある。(ii)の根拠となる数値は、本番の出力にない。

**問い 2: 判定関数の参考値が、なぜ支持の向きになったか**(参考値を結論として書かない)

- 並べ替えで Recall@10 は 0.1304 から 0.2207 に上がり、coverage 0.2260 にほぼ達した。coverage に対する割合は、シードごとに 96.3%〜98.2%(平均で約 98%)で、上位 50 件に入った正例の大部分を上位 10 件に入れた計算になる。
- 候補: cross-encoder が上位 50 件の中で正例を見分ける力は高く、並べ替えの余地のほぼすべてを使い切っている。限界は、第 1 段の候補の側(coverage)にある。
- 根拠: 最後の区間の訓練損失 0.348(一様な予測 2.079)、候補 50 件のスコアの標準偏差の中央値 3.0〜3.8、検証用の増分 +0.1939(標準偏差の 12.3 倍)は、学習が成立していることを示す。ただし、評価の query にも正例が上位 50 件の中にないものがあり(coverage 0.2260)、それらは並べ替えで救えない。cross-encoder が上位 50 件の中で何を手がかりにして正例を選んでいるか(query の文を抜いた跡を検出しているのか、内容の一致を読んでいるのか)は、本番の出力から確かめられない。
- 確かめる方法(実行しない): 評価の索引の passage を、1 文を除いた形ではなく元の形にして、同じ cross-encoder の並べ替えを測る。coverage が高い第 1 段(例えば $K$ を増やした構成。観察 D)で、並べ替えた Recall@10 が coverage に追随するかを見る。

**025 との対比**(事後的な観察): 025 では、同じ構成の cross-encoder が学習量を 8 倍にしても並べ替えなしを上回らなかった(前提不成立)。本トピックでは、$T_2 = 2048$ で、検証用の query の並べ替えの Recall@10 が並べ替えなしを大きく上回った。025 の事後的な解釈(記事単位の関連性の定義が、query と passage の語どうしの相互作用を必要としていなかった可能性)と整合する見方はできる。**ただし、本トピックの実験はこの解釈を検証する設計ではなく、2 つのトピックの学習の組・ラベル・評価の形式が違うので、この対比は検証済みの結論ではない。**

**教訓**:

- 別のトピックから引き継いだ前提条件の閾値は、評価のラベルや正例の数など、閾値の意味を決める定義が変わるときに、新しい定義で決め直す。
- 本番の前のパイロットで前提条件の値(ここでは coverage)が分かる場合は、その時点で前提条件を確かめる仕組みを置く。本番の前に不成立が分かれば、設計を見直せた。

### 7.12 制約と一般化の範囲

- **コーパス**: 英語 Wikipedia の長大な記事 9,470 本(リスト系の記事が約 30.2%)の 1 つのコーパスで、評価用の記事は 300 本。1 つの起点のチェックポイント(008)、1 つのトークナイザ。
- **学習量**: 第 1 段は $T_1 = 1024$(0.059 エポック)、cross-encoder は $T_2 = 2048$(学習に使える query に対して約 0.79 エポック)。学習量を増やしたときに差がどうなるかは、本番の結果からは分からない。
- **文の分割**: トークンごとの表による規則で、略語の後の空白で分割する既知の誤りを含む。抜く文の長さは 4〜96 トークンで、query は先頭 32 トークンまで。32 トークンを超えて切られた query の割合は、評価の索引で 4.5%、学習の組で 4.2〜4.4%(抜く文を query の長さ以内に収まる文から優先して選んだ後の値)。
- **索引**: passage 単位の評価の索引は 93,995 passage で、文が 2 つ未満の passage など 491 個を除いた。評価用の query は、記事単位のラベルの正解が 1 つ以上ある query に限る(評価用 3,227 個)。
- **介入の束ね方**: 実験 A・B の介入は、正例の対応と、同じ記事の組の in-batch での扱いの 2 つを含み、本番の設計では分けられない(7.3 節、7.11.1 節)。R1 の負例は 1 行あたり 62 個、R2 は 63 個。
- **query の形式**: 学習も評価も、passage から抜いた 1 文を query にしている。利用者が書く質問の形とは異なる。
- **cross-encoder の学習量の選択**: $T_2$ の選択に、検証用の query の下流の指標を使っており、実験 C に有利な向きの選択の偏りがありうる(6.1 節)。今回は最小の 2048 が選ばれた。実験 C は前提不成立で、この偏りは判定には影響しない。
- **実験 C**: 前提不成立で、「アプリで並べ替えを足す価値がある」という結論は得ていない。観察 D の部分集合の値も、同じ制約のもとで読む。
- **偏り**: 計画 0 だったので、規則で決めた学習率による偏り(段階 4 以降)はない。


## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/07_retrieval/026_inverse_cloze_task_and_two_stage_retrieval.ipynb
