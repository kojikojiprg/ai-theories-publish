# The AI Theories 投稿用リポジトリ

---

## このリポジトリの位置づけ

本リポジトリは、[ai-theories](https://github.com/kojikojiprg/ai-theories) で学習した LLM(Large Language Model、大規模言語モデル)・VLM(Vision-Language Model、視覚言語モデル)の理論を Zenn で公開するための、Zenn 連携用リポジトリです。

`ai-theories` は本リポジトリに **参照専用の submodule として含まれ**、本リポジトリ側から直接編集することはありません。ノートブックの追加・修正は `ai-theories` 側で行い、本リポジトリでは Workflow を通じてその内容を Zenn 本(book)の章の下書きに変換・公開します。

公開先は Zenn の本(book)**「LLM / VLM理論学習ロードマップ」(`books/ai-theories-roadmap/`)に一本化**しています。個別の記事(article)としての公開は行いません。

## ディレクトリ構成

```
ai-theories-publish/
├── ai-theories/              # ai-theories リポジトリの submodule(参照専用)
├── images/                   # 章内で参照する画像
├── books/
│   └── ai-theories-roadmap/  # Zenn 本(学習ロードマップ、生成物)
│       ├── config.yaml       # 本のタイトル・概要・章の並び順など
│       ├── 0_introduction.md # 「はじめに」章(本の目的・推奨学習順序の一覧表など)
│       └── <slug>.md         # 各トピックの章(ノートブックからの変換結果)
├── scripts/
│   ├── nb_to_zenn.py         # ノートブックを本の章に変換するスクリプト
│   ├── generate_book.py      # theories/README.md・manifest.json から「はじめに」章・config.yamlのchaptersを生成するスクリプト
│   └── manifest.json         # ノートブック番号 → 生成済み章slug・タイトルの対応表(生成物)
└── .github/workflows/
    └── publish_notebook.yml  # 章生成・pushを行う Workflow
```

---

## 章投稿手順

### Workflowの実行手順

1. GitHubの Actions タブから `Publish Notebook` Workflow を開く。
2. `Run workflow` から、変換対象ノートブックの3桁連番(例: `003`)を `notebook_number` に入力して実行する。
3. Workflow が以下を自動で行う。
   - `ai-theories` submoduleの最新化
   - `scripts/nb_to_zenn.py` によるノートブック → 本の章(Markdown)への変換(`books/ai-theories-roadmap/<slug>.md`)
   - `scripts/generate_book.py` による `0_introduction.md`・`config.yaml` の chapters の再構築
   - `images/` ・`books/` ・submodule参照の更新差分のコミット・push

Workflowは `workflow_dispatch` による手動実行のみをトリガーとします。push検知等による自動実行は行いません。

### Workflow実行後に手動で行う作業

Workflowによって生成される章は下書きです。公開前に、以下を手動で行ってください。

- 生成された章ファイル(`books/ai-theories-roadmap/<slug>.md`)の内容を確認する。実装セクションの要約(ノートブックの実装セルの内容がそのまま転記されるため、章として読みやすい分量・粒度に要約する)などが必要であれば調整する。
- `books/ai-theories-roadmap/config.yaml` の `published` を `true` に変更し、push する。

`0_introduction.md`・`config.yaml` の chapters は `scripts/generate_book.py` によって実行のたびに再構築されるため、直接の手動編集は次回 Workflow 実行時に上書きされます。内容を変えたい場合は、章の並び順は `ai-theories` 側の `theories/README.md`(推奨学習順序表)を、`config.yaml` の `title` ・`summary` ・`topics` ・`price` ・`published` は `books/ai-theories-roadmap/config.yaml` を直接編集してください(これらのフィールドは再生成時も既存の値を保持します)。

---

## Workflow実行内容

### 本(book)の「はじめに」章・章一覧の自動生成

`books/ai-theories-roadmap/0_introduction.md` と `config.yaml` の `chapters` 配列は、[ai-theories](https://github.com/kojikojiprg/ai-theories) の `theories/README.md`(トピック一覧・推奨学習順序)と `scripts/manifest.json`(章化状況)から `scripts/generate_book.py` によって**自動生成**されます。

- `0_introduction.md` には、本の目的・コードのライセンス・`ai-theories` 本体へのリンクに加え、`theories/README.md` の推奨学習順序表と同じ構成(トピック名・カテゴリ・前提知識・扱う内容)の一覧表を掲載します。`manifest.json` に章が存在するトピックはトピック名が該当章へのリンクになり、まだ章になっていないトピックは「🚧 準備中」と表示されます(行自体は残ります)。
- `config.yaml` の `chapters` 配列は、先頭が常に `0_introduction`、以降は `manifest.json` に存在する章slugをノートブック番号の昇順(前編・後編分割時は前編→後編の順)で並べたものに再構築されます。`title` ・`summary` ・`topics` ・`price` ・`published` は既存の値を保持します(`config.yaml` が存在しない場合のみ初期値を設定し、`published` の初期値は `false` です)。
- 各トピックの章本文自体は `scripts/generate_book.py` ではなく `scripts/nb_to_zenn.py` が生成します。`scripts/generate_book.py` は `0_introduction.md` と `config.yaml` の chapters 以外のファイルには一切触れません。

### 章の自動分割(前編・後編、および実装・実験編のさらなる分割)

Zennの**本(book)の章**の本文には、実際のデプロイエラー(「本文のmarkdownには最大50000文字まで使用できます」)により判明した**50,000文字**という上限があります(記事(article)の80,000文字上限とは異なる、より厳しい制限です)。生成した章本文(frontmatterを除く)が閾値(45,000文字。上限に対して安全マージンを持たせた値)を超える場合、`scripts/nb_to_zenn.py` は「実装方針」の見出しを境界に、以下の2つに分割生成します。

- 前編(理論編): `<slug>-theory.md`(タイトル〜理論までのセクション)
- 後編(実装・実験編): `<slug>-practice.md`(実装方針〜結果・考察までのセクション)

さらに、後編(実装・実験編)側の「実験」セクションなどが大きく、分割後もなお閾値を超える場合は、`##`見出し単位(収まらなければ`###`見出し単位)でさらに複数章に分割し、`<slug>-practice-1.md`, `<slug>-practice-2.md`, ... のように連番付きで生成します(理論編側が同様に閾値を超える場合は `<slug>-theory-1.md`, `<slug>-theory-2.md`, ... となります)。分割が1つの前編・1つの後編に収まる場合は、従来通り連番なしの `<slug>-theory.md` / `<slug>-practice.md` になります。

分割時、画像フォルダ(`images/<slug>/`)は分割せず全ての章で共有参照します。Zennの章URLはslugから生成時点で確定するため、分割された各章の冒頭には前後の章への相互リンク(`https://zenn.dev/<Zennユーザー名>/books/ai-theories-roadmap/viewer/<相手のslug>`)を確定URLとして埋め込み済みの状態で生成されます(公開後に手動でリンクを追加する必要はありません)。Zennユーザー名はスクリプト内の定数 `ZENN_USERNAME` で管理しており、環境変数 `ZENN_USERNAME` でも上書きできます。分割されない場合は、従来通り単一の章(`<slug>.md`)として生成されます。

### ノートブック間リンクの変換

ai-theoriesのノートブックは、Markdownセル内で他のノートブックを `[001](./001_attention_mechanism.ipynb)` のような相対パスのリンクで参照することがあります。`scripts/nb_to_zenn.py` はこれを検出し、`scripts/manifest.json`(ノートブック番号 → 生成済み章slugの対応表。章生成のたびに自動更新される生成物)を参照して、以下のようにリンク先を書き換えます。リンクテキスト(`[001]`の部分)は変更しません。

- リンク先のノートブック番号が `manifest.json` に登録済み(すでに章化済み)の場合: 本の章のURL(`https://zenn.dev/<Zennユーザー名>/books/ai-theories-roadmap/viewer/<slug>`)に置き換える。前編・後編に分割された章の場合は、前編(理論編)のリンクに固定する。
- リンク先のノートブックはai-theories内に存在するが、まだ `manifest.json` に登録されていない(未変換)場合: GitHubの絶対URL(`https://github.com/kojikojiprg/ai-theories/blob/main/theories/<カテゴリ>/<ファイル名>`)に置き換える(相対パスのままではZenn上でリンク切れになるため)。
- リンク先のノートブックがai-theories内に見つからない場合(未作成のトピックなど): リンクは変更せず、実行ログに警告を出力する。

---

## 用語・Markdown記法のルール

本リポジトリで日本語の文章(README・スクリプトのコメント・ログメッセージ等)を作成する際は、[ai-theories](https://github.com/kojikojiprg/ai-theories) の `CLAUDE.md` に定める「言語ルール」に準じます。主なルールは以下のとおりです。

- 日本語をメインの説明言語とし、専門用語や重要なキーワードには英語を併記する。
- 文中で独自に短縮した略語を使わない。正式名称を用い、初出時に日本語と英語を併記する。ただし、手法名・アーキテクチャ名として定着している略語(LoRA、RoPEなど)は使用してよい。
- 強調記法(`**` ・`*`)の前後が日本語の文字である場合は、防御的に半角スペースを入れる。ただし、閉じ記号の直後が句読点(。、)・閉じ括弧である場合や、開き記号の直前が開き括弧・句読点である場合はスペースを入れない。
- インラインコード(バッククォート)の前後にはスペースを入れない。
- リンク記法(`[...](...)`)は、リンクテキストの前後が日本語の文字である場合、強調記法と同じ扱いとする。

詳細は `ai-theories` の `CLAUDE.md` を参照してください。
