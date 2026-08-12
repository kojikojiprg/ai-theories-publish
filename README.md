# ai-theories-publish

## このリポジトリの位置づけ

本リポジトリは、[ai-theories](https://github.com/kojikojiprg/ai-theories) で学習した LLM(Large Language Model、大規模言語モデル)・VLM(Vision-Language Model、視覚言語モデル)の理論を Zenn で公開するための、Zenn 連携用リポジトリです。

`ai-theories` は本リポジトリに **参照専用の submodule として含まれ**、本リポジトリ側から直接編集することはありません。ノートブックの追加・修正は `ai-theories` 側で行い、本リポジトリでは Workflow を通じてその内容を Zenn 記事の下書きに変換・公開します。

## ディレクトリ構成

```
ai-theories-publish/
├── ai-theories/             # ai-theories リポジトリの submodule(参照専用)
├── articles/                # Zenn 記事(Markdown)
├── images/                  # 記事内で参照する画像
├── scripts/
│   ├── nb_to_zenn.py        # ノートブックを Zenn 記事に変換するスクリプト
│   └── manifest.json        # ノートブック番号 → 生成済み記事slugの対応表(生成物)
└── .github/workflows/
    └── publish_notebook.yml # 記事生成・pushを行う Workflow
```

## Workflowの実行手順

1. GitHubの Actions タブから `Publish Notebook` Workflow を開く。
2. `Run workflow` から、変換対象ノートブックの3桁連番(例: `003`)を `notebook_number` に入力して実行する。
3. Workflow が以下を自動で行う。
   - `ai-theories` submoduleの最新化
   - `scripts/nb_to_zenn.py` によるノートブック → Zenn記事(Markdown)への変換
   - `articles/` ・`images/` ・submodule参照の更新差分のコミット・push

Workflowは `workflow_dispatch` による手動実行のみをトリガーとします。push検知等による自動実行は行いません。

## 記事の自動分割(前編・後編)

Zennの記事本文には文字数上限(80,000文字)があります。生成した記事本文(frontmatterを除く)が閾値(70,000文字。上限に対して安全マージンを持たせた値)を超える場合、`scripts/nb_to_zenn.py` は自動的に以下の2記事に分割生成します。

- 前編(理論編): `<slug>-theory.md`(タイトル〜理論までのセクション)
- 後編(実装・実験編): `<slug>-practice.md`(実装方針〜結果・考察までのセクション)

分割時、画像フォルダ(`images/<slug>/`)は分割せず両記事で共有参照します。Zennの記事URLはslugから生成時点で確定するため、前編・後編それぞれの冒頭には相互リンク(`https://zenn.dev/<Zennユーザー名>/articles/<相手のslug>`)を確定URLとして埋め込み済みの状態で生成されます(公開後に手動でリンクを追加する必要はありません)。Zennユーザー名はスクリプト内の定数 `ZENN_USERNAME` で管理しており、環境変数 `ZENN_USERNAME` でも上書きできます。分割されない場合は、従来通り単一記事(`<slug>.md`)として生成されます。

## ノートブック間リンクの変換

ai-theoriesのノートブックは、Markdownセル内で他のノートブックを `[001](./001_attention_mechanism.ipynb)` のような相対パスのリンクで参照することがあります。`scripts/nb_to_zenn.py` はこれを検出し、`scripts/manifest.json`(ノートブック番号 → 生成済み記事slugの対応表。記事生成のたびに自動更新される生成物)を参照して、以下のようにリンク先を書き換えます。リンクテキスト(`[001]`の部分)は変更しません。

- リンク先のノートブック番号が `manifest.json` に登録済み(すでにZenn記事化済み)の場合: ZennのURL(`https://zenn.dev/<Zennユーザー名>/articles/<slug>`)に置き換える。前編・後編に分割された記事の場合は、前編(理論編)のリンクに固定する。
- リンク先のノートブックはai-theories内に存在するが、まだ `manifest.json` に登録されていない(未変換)場合: GitHubの絶対URL(`https://github.com/kojikojiprg/ai-theories/blob/main/theories/<カテゴリ>/<ファイル名>`)に置き換える(相対パスのままではZenn上でリンク切れになるため)。
- リンク先のノートブックがai-theories内に見つからない場合(未作成のトピックなど): リンクは変更せず、実行ログに警告を出力する。

## 実行後に手動で行う作業

Workflowによって生成される記事は `published: false` の下書きです。公開前に、以下を手動で行ってください。

- 実装セクションの要約(ノートブックの実装セルの内容がそのまま転記されるため、記事として読みやすい分量・粒度に要約する)
- frontmatterの `emoji` を、記事内容に合った絵文字に調整する
- frontmatterの `topics` を、記事内容に合わせて追記・調整する
- 内容を確認したうえで `published: true` に変更し、push する

## 用語・Markdown記法のルール

本リポジトリで日本語の文章(README・スクリプトのコメント・ログメッセージ等)を作成する際は、[ai-theories](https://github.com/kojikojiprg/ai-theories) の `CLAUDE.md` に定める「言語ルール」に準じます。主なルールは以下のとおりです。

- 日本語をメインの説明言語とし、専門用語や重要なキーワードには英語を併記する。
- 文中で独自に短縮した略語を使わない。正式名称を用い、初出時に日本語と英語を併記する。ただし、手法名・アーキテクチャ名として定着している略語(LoRA、RoPEなど)は使用してよい。
- 強調記法(`**` ・`*`)の前後が日本語の文字である場合は、防御的に半角スペースを入れる。ただし、閉じ記号の直後が句読点(。、)・閉じ括弧である場合や、開き記号の直前が開き括弧・句読点である場合はスペースを入れない。
- インラインコード(バッククォート)の前後にはスペースを入れない。
- リンク記法(`[...](...)`)は、リンクテキストの前後が日本語の文字である場合、強調記法と同じ扱いとする。

詳細は `ai-theories` の `CLAUDE.md` を参照してください。
