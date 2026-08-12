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
├── books/
│   └── ai-theories-roadmap/ # Zenn 本(学習ロードマップ、生成物)
├── scripts/
│   ├── nb_to_zenn.py        # ノートブックを Zenn 記事に変換するスクリプト
│   ├── generate_book.py     # theories/README.md・manifest.json から Zenn 本を生成するスクリプト
│   └── manifest.json        # ノートブック番号 → 生成済み記事slug・タイトルの対応表(生成物)
└── .github/workflows/
    └── publish_notebook.yml # 記事生成・pushを行う Workflow
```

## Workflowの実行手順

1. GitHubの Actions タブから `Publish Notebook` Workflow を開く。
2. `Run workflow` から、変換対象ノートブックの3桁連番(例: `003`)を `notebook_number` に入力して実行する。
3. Workflow が以下を自動で行う。
   - `ai-theories` submoduleの最新化
   - `scripts/nb_to_zenn.py` によるノートブック → Zenn記事(Markdown)への変換
   - `scripts/generate_book.py` による本(book)の再生成
   - `articles/` ・`images/` ・`books/` ・submodule参照の更新差分のコミット・push

Workflowは `workflow_dispatch` による手動実行のみをトリガーとします。push検知等による自動実行は行いません。

## 本(book)の自動生成

`books/ai-theories-roadmap/` は、[ai-theories](https://github.com/kojikojiprg/ai-theories) の `theories/README.md`(トピック一覧・推奨学習順序)と `scripts/manifest.json`(記事化状況)から `scripts/generate_book.py` によって**自動生成**される Zenn 本です。

- 各章はトピック 1 つに対応し、カテゴリ・前提知識・扱う内容の概要と、記事化済みであれば対応する Zenn 記事へのリンクを掲載します。まだ記事化されていないトピックの章は概要のみで、`🚧 このトピックはまだ記事化されていません(準備中)`と表示されます。
- `scripts/generate_book.py` は実行のたびに `books/ai-theories-roadmap/` 配下を**まるごと再生成**します(差分マージは行いません)。**このディレクトリ配下を手動編集しても、次回の Workflow 実行時に上書きされ、反映されません**。内容を変えたい場合は `ai-theories` 側の `theories/README.md` を修正してください。
- 生成される `config.yaml` の `published` は常に `false`(下書き)です。内容を確認のうえ、公開する場合は手動で `true` に切り替えてください。

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

## 記事末尾のナビゲーション(前後リンク)の自動生成

`scripts/manifest.json` に登録済みの記事を、ノートブック番号の昇順・各エントリの `slugs` 配列の順で連結した「フラットな並び」に基づき、各記事の直前・直後の記事へのリンクを記事本文の末尾に自動追記します。区間は `<!-- zenn-nav:start -->` 〜 `<!-- zenn-nav:end -->` のマーカーで囲まれており、後から機械的に検出・置換できるようになっています。

- 並びの先頭の記事(直前が存在しない)は「前」の行を省略する。
- 並びの末尾の記事(まだ次の番号が `manifest.json` に登録されていない)は「次」の行を `- 次: <次の番号>(未作成)` というリンクなしのプレーンテキストにする。
- 新しいノートブックを変換すると、直前の記事(`manifest.json` 内で対象番号未満の最大の番号のエントリの最後のslug)のnav区間だけを機械的に書き換え、「(未作成)」だった「次」を実リンクにバックフィルする。本文の他の部分(手動で編集済みの内容)には影響しない。

## "(未作成)" 表記の自動除去

記事本文(frontmatterを除く)に含まれる `(未作成)` ・ `(未作成 / TBD)` という文字列は、Zenn記事化された時点で不要になる注記であるため自動的に除去されます。この処理は本リポジトリ内で完結し、ai-theories側の `theories/README.md` には一切適用されません。

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
