#!/usr/bin/env python3
"""ai-theories の theories/README.md と scripts/manifest.json から Zenn本を生成するスクリプト。

実行のたびに books/ai-theories-roadmap/ 配下をまるごと再生成する(差分マージは行わない)。
手動編集は前提としないため、既存ファイルはすべて削除してから生成し直す。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
THEORIES_README_PATH = REPO_ROOT / "ai-theories" / "theories" / "README.md"
MANIFEST_PATH = REPO_ROOT / "scripts" / "manifest.json"
BOOKS_DIR = REPO_ROOT / "books"
BOOK_SLUG = "ai-theories-roadmap"
BOOK_DIR = BOOKS_DIR / BOOK_SLUG

# Zennのユーザー名。将来変わる可能性があるため、環境変数での上書きも可能にしておく(nb_to_zenn.pyと同様)。
ZENN_USERNAME = os.environ.get("ZENN_USERNAME", "kojikojiprg")

AI_THEORIES_GITHUB_URL = "https://github.com/kojikojiprg/ai-theories"

BOOK_TITLE = "LLM / VLM理論学習ロードマップ"
BOOK_SUMMARY = (
    "LLM(大規模言語モデル)・VLM(視覚言語モデル)の理論を、"
    "PyTorchによるスクラッチ実装とあわせて基礎から学ぶロードマップです。"
)
BOOK_TOPICS = ["llm", "vlm", "pytorch", "deeplearning"]

TABLE_HEADER_PREFIX = "| #"
NUMBER_PATTERN = re.compile(r"^\d{3}$")
PREREQUISITE_NUMBER_PATTERN = re.compile(r"\d{3}")


def build_zenn_book_chapter_url(chapter_slug: str) -> str:
    return f"https://zenn.dev/{ZENN_USERNAME}/books/{BOOK_SLUG}/viewer/{chapter_slug}"


def build_zenn_article_url(slug: str) -> str:
    return f"https://zenn.dev/{ZENN_USERNAME}/articles/{slug}"


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

        number, topic, category, prerequisites, contents, notebook = cells[:6]
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


def parse_prerequisite_numbers(prerequisites: str) -> list[str]:
    if "なし" in prerequisites:
        return []
    return PREREQUISITE_NUMBER_PATTERN.findall(prerequisites)


def build_config_yaml(topics: list[dict]) -> str:
    lines = [
        f'title: "{BOOK_TITLE}"',
        f'summary: "{BOOK_SUMMARY}"',
        f"topics: {json.dumps(BOOK_TOPICS, ensure_ascii=False)}",
        "price: 0",
        "published: false",
        "chapters:",
        "  - 0_introduction",
    ]
    for topic in topics:
        lines.append(f"  - {topic['number']}")
    return "\n".join(lines) + "\n"


def build_introduction_body() -> str:
    return f"""---
title: "はじめに"
---

## この本について

本書は [ai-theories]({AI_THEORIES_GITHUB_URL}) で学習している LLM(大規模言語モデル)・VLM(視覚言語モデル)の理論学習ロードマップです。古典的な機械学習理論(SVM、決定木など)は対象外とし、**LLM / VLMに関する理論に限定**しています。

- 理論的に本質的な部分は、可能な限り**PyTorchによるスクラッチ実装**で確認します(Attention計算、正規化、LoRAの低ランク分解など)。
- 学習・実験は**Google Colab無料枠(T4 GPUなど)で完結する規模**のモデル・データセットで構成しています。

## 本書の構成

各章は、1つの学習トピックに対応しています。

- 章の本文には、トピックのカテゴリ・前提知識・扱う内容の概要を掲載します。
- 対応するノートブックがすでにZenn記事化されている場合は、章末にその記事へのリンクを掲載します。
- **記事がまだ準備中のトピックは、概要のみの掲載となります**(🚧 マークで示します)。準備が整い次第、記事へのリンクを追記します。

本書は [ai-theories]({AI_THEORIES_GITHUB_URL}) の `theories/README.md` と記事の生成状況をもとに自動生成しています。

## コードのライセンス

本書内で参照するコード・ノートブックのライセンスは、[ai-theories]({AI_THEORIES_GITHUB_URL}) 本体の LICENSE に準拠します。

## リンク

- [ai-theories(GitHubリポジトリ)]({AI_THEORIES_GITHUB_URL})
"""


def build_prerequisites_section(
    prerequisites: str, topics_by_number: dict[str, dict]
) -> str:
    numbers = parse_prerequisite_numbers(prerequisites)
    if not numbers:
        return prerequisites

    lines = []
    for number in numbers:
        prereq_topic = topics_by_number.get(number)
        label = f"{number}. {prereq_topic['topic']}" if prereq_topic else number
        url = build_zenn_book_chapter_url(number)
        lines.append(f"- [{label}]({url})")
    return "\n".join(lines)


def build_related_article_section(number: str, manifest: dict) -> str:
    entry = manifest.get(number)
    if entry is None:
        return "🚧 このトピックはまだ記事化されていません(準備中)"

    if entry["split"]:
        index = next(
            i for i, slug in enumerate(entry["slugs"]) if slug.endswith("-theory")
        )
    else:
        index = 0

    slug = entry["slugs"][index]
    title = entry["titles"][index]
    url = build_zenn_article_url(slug)
    return f"この章の詳細記事はこちら: [{title}]({url})"


def build_topic_body(
    topic: dict, topics_by_number: dict[str, dict], manifest: dict
) -> str:
    prerequisites_section = build_prerequisites_section(
        topic["prerequisites"], topics_by_number
    )
    related_article_section = build_related_article_section(topic["number"], manifest)

    return f"""---
title: "{topic['topic']}"
---

## カテゴリ / Category

{topic['category']}

## 前提知識 / Prerequisites

{prerequisites_section}

## 扱う内容 / Contents

{topic['contents']}

## 関連記事 / Related Article

{related_article_section}
"""


def generate_book() -> tuple[list[dict], int]:
    readme_text = THEORIES_README_PATH.read_text(encoding="utf-8")
    topics = parse_topics(readme_text)
    topics_by_number = {topic["number"]: topic for topic in topics}
    manifest = load_manifest()

    if BOOK_DIR.exists():
        shutil.rmtree(BOOK_DIR)
    BOOK_DIR.mkdir(parents=True)

    (BOOK_DIR / "config.yaml").write_text(
        build_config_yaml(topics), encoding="utf-8"
    )
    (BOOK_DIR / "0_introduction.md").write_text(
        build_introduction_body(), encoding="utf-8"
    )

    tbd_count = 0
    for topic in topics:
        body = build_topic_body(topic, topics_by_number, manifest)
        (BOOK_DIR / f"{topic['number']}.md").write_text(body, encoding="utf-8")
        if topic["number"] not in manifest:
            tbd_count += 1

    return topics, tbd_count


def main() -> None:
    topics, tbd_count = generate_book()

    print(f"{BOOK_DIR.relative_to(REPO_ROOT)} を再生成しました。")
    print(f"- config.yaml")
    print(f"- 0_introduction.md")
    for topic in topics:
        print(f"- {topic['number']}.md")
    print(f"準備中(未記事化)のトピック数: {tbd_count} / {len(topics)}")
    print("config.yaml の published は false(下書き)です。")


if __name__ == "__main__":
    main()
