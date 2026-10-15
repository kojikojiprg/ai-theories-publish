#!/usr/bin/env python3
"""ai-theories の theories/README.md と scripts/manifest.json から Zenn本の
「はじめに」章(0_introduction.md)と config.yaml の chapters 配列を生成するスクリプト。

各トピックの本文(章ファイル)は scripts/nb_to_zenn.py が直接生成する。
このスクリプトは 0_introduction.md と config.yaml の再構築のみを行い、
他の章ファイルには一切触れない。
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import nb_to_zenn

REPO_ROOT = Path(__file__).resolve().parent.parent
THEORIES_README_PATH = REPO_ROOT / "ai-theories" / "theories" / "README.md"
MANIFEST_PATH = REPO_ROOT / "scripts" / "manifest.json"
BOOKS_DIR = REPO_ROOT / "books"
BOOK_SLUG = "ai-theories-roadmap"
BOOK_DIR = BOOKS_DIR / BOOK_SLUG
CONFIG_PATH = BOOK_DIR / "config.yaml"
INTRODUCTION_PATH = BOOK_DIR / "0_introduction.md"

# Zennのユーザー名。将来変わる可能性があるため、環境変数での上書きも可能にしておく(nb_to_zenn.pyと同様)。
ZENN_USERNAME = os.environ.get("ZENN_USERNAME", "kojikojiprg")

AI_THEORIES_GITHUB_URL = "https://github.com/kojikojiprg/ai-theories"

# config.yaml が存在しない場合の初期値。既存の config.yaml があればそちらの値を優先する。
DEFAULT_BOOK_TITLE = "LLM / VLM理論学習ロードマップ"
DEFAULT_BOOK_SUMMARY = (
    "LLM(大規模言語モデル)・VLM(視覚言語モデル)の理論を、"
    "PyTorchによるスクラッチ実装とあわせて基礎から学ぶロードマップです。"
)
DEFAULT_BOOK_TOPICS = ["llm", "vlm", "pytorch", "deeplearning"]
DEFAULT_PRICE = "0"
DEFAULT_PUBLISHED = "false"

# 「はじめに」章の本文の文字数の閾値。超える場合は検証エラーにする(章は分割しない)。
CHAR_THRESHOLD = nb_to_zenn.CHAR_THRESHOLD

# Zenn の Mermaid の制限(https://zenn.dev/zenn/articles/markdown-guide の「制限事項」)。
MERMAID_MAX_CHARS = 2000  # 1 ブロックあたりの文字数
MERMAID_MAX_CHAINS = 10  # 1 ブロックあたりの & の数(Chain 数)

OVERVIEW_HEADING = "## 全体像"
ORDER_HEADING = "## 推奨学習順序"
# 「推奨学習順序」の節の終わり。この後の「各トピックの詳細」は、文字数が多いため Zenn に載せない。
ORDER_END_HEADING = "### 各トピックの詳細"
# 表から除く列(見出しの先頭の文字列)。文字数が多いため。
DROPPED_COLUMN_PREFIXES = ("判定",)

TABLE_HEADER_PREFIX = "| #"
NUMBER_PATTERN = re.compile(r"^\d{3}$")

CONFIG_SCALAR_FIELDS = ("title", "summary", "topics", "price", "published")
CONFIG_FIELD_PATTERN = re.compile(
    r"^(" + "|".join(CONFIG_SCALAR_FIELDS) + r"):\s*(.*)$"
)


def build_zenn_book_chapter_url(chapter_slug: str) -> str:
    return f"https://zenn.dev/{ZENN_USERNAME}/books/{BOOK_SLUG}/viewer/{chapter_slug}"


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def is_separator_row(line: str) -> bool:
    stripped = line.replace("|", "").strip()
    return bool(stripped) and set(stripped) <= {"-", " "}


def parse_topics(readme_text: str) -> list[dict]:
    lines = readme_text.splitlines()
    topics: list[dict] = []
    in_table = False

    for line in lines:
        if line.startswith(TABLE_HEADER_PREFIX):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            break
        if is_separator_row(line):
            continue

        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue

        number, topic, category, prerequisites, contents = cells[:5]
        notebook = cells[-1]
        if not NUMBER_PATTERN.match(number):
            continue

        topics.append(
            {
                "number": number,
                "topic": topic,
                "category": category,
                "prerequisites": prerequisites,
                "contents": contents,
                "is_tbd": "未作成" in notebook,
            }
        )

    if not topics:
        print(
            f"エラー: {THEORIES_README_PATH.relative_to(REPO_ROOT)} から"
            "「推奨学習順序 / Recommended Order」の表を読み取れませんでした。",
            file=sys.stderr,
        )
        sys.exit(1)

    return topics


def build_chapter_slugs(manifest: dict) -> list[str]:
    slugs = []
    for number in sorted(manifest.keys(), key=int):
        slugs.extend(manifest[number]["slugs"])
    return slugs


def fail(message: str) -> None:
    print(f"エラー: {message}", file=sys.stderr)
    sys.exit(1)


def extract_section(readme_text: str, start_heading: str, end_heading: str) -> str:
    """start_heading で始まる行から、end_heading で始まる行の直前までを返す(前後の空行は除く)。"""
    lines = readme_text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(start_heading)), None)
    if start is None:
        fail(f"theories/README.md に「{start_heading}」の節がありません。")
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith(end_heading)), None
    )
    if end is None:
        fail(f"theories/README.md に「{end_heading}」の節がありません。")
    return "\n".join(lines[start:end]).strip("\n")


def validate_mermaid_blocks(text: str) -> int:
    """Mermaid のコードブロックが Zenn の制限内かを確かめる。超えていれば検証エラーにする。"""
    count = 0
    for match in nb_to_zenn.MERMAID_BLOCK_PATTERN.finditer(text):
        count += 1
        block = match.group(1)
        if len(block) > MERMAID_MAX_CHARS:
            fail(
                f"{count} 個目の Mermaid のブロックが {len(block)} 文字で、Zenn の上限"
                f"({MERMAID_MAX_CHARS} 文字)を超えています。"
            )
        if block.count("&") > MERMAID_MAX_CHAINS:
            fail(
                f"{count} 個目の Mermaid のブロックの & の数が {block.count('&')} で、Zenn の上限"
                f"({MERMAID_MAX_CHAINS})を超えています。"
            )
    return count


def convert_table(table_text: str, manifest: dict) -> str:
    """学習順序の表の、ノートブックの列のリンクを「#」の列に移し、ノートブックの列は除く。
    リンク先は Zenn の章 URL に解決する(章が未変換なら GitHub の URL)。
    章になっていない行は、「#」の列に「🚧 準備中」を添える。
    """
    drop: set[int] = set()
    out = []
    counts_total = {"zenn": 0, "github": 0, "unchanged": 0}
    for line in table_text.splitlines():
        if not line.startswith("|"):
            out.append(line)
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells[0].startswith("#"):
            drop = {i for i, cell in enumerate(cells) if cell.startswith(DROPPED_COLUMN_PREFIXES)}
        cells = [cell for i, cell in enumerate(cells) if i not in drop]
        if is_separator_row(line):
            out.append("|" + " --- |" * (len(cells) - 1))
            continue
        if cells[0].startswith("#"):
            out.append("| " + " | ".join(cells[:-1]) + " |")
            continue
        if not NUMBER_PATTERN.match(cells[0]):
            out.append(line)
            continue
        number, notebook = cells[0], cells[-1]
        if "未作成" in notebook:
            number_cell = f"{number} 🚧 準備中"
        else:
            resolved, counts = nb_to_zenn.rewrite_notebook_links(notebook, manifest)
            for key, value in counts.items():
                counts_total[key] += value
            match = re.search(r"\]\(([^)]+)\)", resolved)
            number_cell = f"[{number}]({match.group(1)})" if match else number
        out.append("| " + " | ".join([number_cell] + cells[1:-1]) + " |")
    print(
        f"学習順序表のリンク: Zennの章のURL {counts_total['zenn']}件、"
        f"GitHubのURL {counts_total['github']}件、変更なし {counts_total['unchanged']}件。"
    )
    return "\n".join(out)


INTRO_HEAD = f"""---
title: "はじめに"
---

## この本について

本書は [ai-theories]({AI_THEORIES_GITHUB_URL}) で学習している LLM(大規模言語モデル)・VLM(視覚言語モデル)の理論学習ロードマップです。古典的な機械学習理論(SVM、決定木など)は対象外とし、 **LLM / VLM に関する理論に限定** しています。

- 理論的に本質的な部分は、可能な限り **PyTorch によるスクラッチ実装** で確認します(Attention 計算、正規化、LoRA の低ランク分解など)。
- 学習・実験は **Google Colab 無料枠(T4 GPU など)で完結する規模** のモデル・データセットで構成しています。

## 本書の構成

以降の章は、[ai-theories]({AI_THEORIES_GITHUB_URL}) の各学習トピックのノートブックをそのまま章として収録したものです。この章には、トピックの全体像を示す図と推奨学習順序の表を、 `theories/README.md` から転載しています。表の「#」の列の番号をクリックすると、該当の章に移動します。 **まだ章になっていないトピックは「🚧 準備中」と表示** します。
"""

INTRO_TAIL = f"""## コードのライセンス

本書内で参照するコード・ノートブックのライセンスは、[ai-theories]({AI_THEORIES_GITHUB_URL}) 本体の LICENSE に準拠します。

## リンク

- [ai-theories(GitHub リポジトリ)]({AI_THEORIES_GITHUB_URL})
"""

def build_introduction_body(readme_text: str, manifest: dict) -> str:
    overview = extract_section(readme_text, OVERVIEW_HEADING, ORDER_HEADING)
    order = extract_section(readme_text, ORDER_HEADING, ORDER_END_HEADING)

    validate_mermaid_blocks(overview)
    order = convert_table(order, manifest)

    parts = [INTRO_HEAD, overview, order, INTRO_TAIL]
    return "\n\n".join(part.strip("\n") for part in parts) + "\n"


def generate_introduction(readme_text: str, manifest: dict) -> str:
    """「はじめに」章の全文を返す。本文が閾値を超えるときは検証エラーにする(章は分割しない)。"""
    body = build_introduction_body(readme_text, manifest)
    length = len(body.split("---\n", 2)[2])
    if length > CHAR_THRESHOLD:
        fail(f"「はじめに」の本文が {length} 文字で、閾値({CHAR_THRESHOLD} 文字)を超えます。")
    return body


def parse_existing_config(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("chapters:"):
            break
        match = CONFIG_FIELD_PATTERN.match(line)
        if match:
            values[match.group(1)] = match.group(2)
    return values


def build_config_yaml(existing: dict[str, str], chapter_slugs: list[str]) -> str:
    title = existing.get("title", f'"{DEFAULT_BOOK_TITLE}"')
    summary = existing.get("summary", f'"{DEFAULT_BOOK_SUMMARY}"')
    topics = existing.get("topics", json.dumps(DEFAULT_BOOK_TOPICS, ensure_ascii=False))
    price = existing.get("price", DEFAULT_PRICE)
    published = existing.get("published", DEFAULT_PUBLISHED)

    lines = [
        f"title: {title}",
        f"summary: {summary}",
        f"topics: {topics}",
        f"price: {price}",
        f"published: {published}",
        "chapters:",
        '  - "0_introduction"',
    ]
    for slug in chapter_slugs:
        # クォートしないとYAMLのフロースカラーの解釈やZennのchapters検証に
        # 影響しうるため、文字列であることを明示するために引用符で囲む。
        lines.append(f'  - "{slug}"')
    return "\n".join(lines) + "\n"


def generate_book() -> tuple[list[dict], list[str], int]:
    readme_text = THEORIES_README_PATH.read_text(encoding="utf-8")
    topics = parse_topics(readme_text)
    manifest = load_manifest()

    BOOK_DIR.mkdir(parents=True, exist_ok=True)

    chapter_slugs = build_chapter_slugs(manifest)
    existing_config = parse_existing_config(CONFIG_PATH)

    CONFIG_PATH.write_text(
        build_config_yaml(existing_config, chapter_slugs), encoding="utf-8"
    )
    introduction = generate_introduction(readme_text, manifest)
    INTRODUCTION_PATH.write_text(introduction, encoding="utf-8")
    print(
        f"「はじめに」の本文: {len(introduction.split('---' + chr(10), 2)[2])} 文字"
    )

    tbd_count = sum(1 for topic in topics if topic["number"] not in manifest)

    return topics, chapter_slugs, tbd_count


def main() -> None:
    topics, chapter_slugs, tbd_count = generate_book()

    print(f"{INTRODUCTION_PATH.relative_to(REPO_ROOT)} を再生成しました。")
    print(f"{CONFIG_PATH.relative_to(REPO_ROOT)} の chapters を再構築しました。")
    print("chapters:")
    print("  - 0_introduction")
    for slug in chapter_slugs:
        print(f"  - {slug}")
    print(f"準備中(未記事化)のトピック数: {tbd_count} / {len(topics)}")


if __name__ == "__main__":
    main()
