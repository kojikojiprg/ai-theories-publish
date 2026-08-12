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


def resolve_chapter_slug(number: str, manifest: dict) -> str | None:
    entry = manifest.get(number)
    if entry is None:
        return None
    if entry["split"]:
        return next(s for s in entry["slugs"] if s.endswith("-theory"))
    return entry["slugs"][0]


def build_chapter_slugs(manifest: dict) -> list[str]:
    slugs = []
    for number in sorted(manifest.keys(), key=int):
        slugs.extend(manifest[number]["slugs"])
    return slugs


def build_topic_table_row(topic: dict, manifest: dict) -> str:
    slug = resolve_chapter_slug(topic["number"], manifest)
    if slug is not None:
        topic_cell = f"[{topic['topic']}]({build_zenn_book_chapter_url(slug)})"
    else:
        topic_cell = f"{topic['topic']}(🚧 準備中)"

    return (
        f"| {topic['number']} | {topic_cell} | {topic['category']} | "
        f"{topic['prerequisites']} | {topic['contents']} |"
    )


def build_topics_table(topics: list[dict], manifest: dict) -> str:
    lines = [
        "| # | トピック / Topic | カテゴリ / Category | 前提知識 / Prerequisites | 扱う内容 / Contents |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(build_topic_table_row(topic, manifest) for topic in topics)
    return "\n".join(lines)


def build_introduction_body(topics: list[dict], manifest: dict) -> str:
    topics_table = build_topics_table(topics, manifest)

    return f"""---
title: "はじめに"
---

## この本について

本書は [ai-theories]({AI_THEORIES_GITHUB_URL}) で学習している LLM(大規模言語モデル)・VLM(視覚言語モデル)の理論学習ロードマップです。古典的な機械学習理論(SVM、決定木など)は対象外とし、**LLM / VLMに関する理論に限定**しています。

- 理論的に本質的な部分は、可能な限り**PyTorchによるスクラッチ実装**で確認します(Attention計算、正規化、LoRAの低ランク分解など)。
- 学習・実験は**Google Colab無料枠(T4 GPUなど)で完結する規模**のモデル・データセットで構成しています。

## 本書の構成

以降の章は、[ai-theories]({AI_THEORIES_GITHUB_URL}) の各学習トピックのノートブックをそのまま章として収録したものです。下表は `theories/README.md` の推奨学習順序表と同じ構成(トピック名・カテゴリ・前提知識・扱う内容)の一覧です。トピック名がリンクになっているものは既に章として収録済みで、リンクをクリックすると該当の章に移動します。**まだ章になっていないトピックは「🚧 準備中」と表示**しています。

## 推奨学習順序 / Recommended Order

{topics_table}

## コードのライセンス

本書内で参照するコード・ノートブックのライセンスは、[ai-theories]({AI_THEORIES_GITHUB_URL}) 本体の LICENSE に準拠します。

## リンク

- [ai-theories(GitHubリポジトリ)]({AI_THEORIES_GITHUB_URL})
"""


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
    INTRODUCTION_PATH.write_text(
        build_introduction_body(topics, manifest), encoding="utf-8"
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
