#!/usr/bin/env python3
"""ai-theories のノートブックを Zenn 記事(Markdown)の下書きに変換するスクリプト。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import nbformat
from nbconvert import MarkdownExporter

REPO_ROOT = Path(__file__).resolve().parent.parent
AI_THEORIES_THEORIES_DIR = REPO_ROOT / "ai-theories" / "theories"
ARTICLES_DIR = REPO_ROOT / "articles"
IMAGES_DIR = REPO_ROOT / "images"

NOTEBOOK_NUMBER_PATTERN = re.compile(r"^\d{3}$")
SLUG_PATTERN = re.compile(r"^[a-z0-9_-]{12,50}$")
TITLE_HEADING_PATTERN = re.compile(r"^#\s*\d{3}\.\s*(.+?)\s*$")

DEFAULT_EMOJI = "📝"
DEFAULT_TOPICS = ["ai", "llm", "vlm", "pytorch", "machine learning"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebook_number",
        help="変換対象ノートブックの3桁連番(例: 003)",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help=(
            "公開先リポジトリの owner/repo(例: kojikojiprg/ai-theories-publish)。"
            "省略時は環境変数 GITHUB_REPOSITORY を使用する。"
        ),
    )
    return parser.parse_args()


def resolve_publish_repo(repo_arg: str | None) -> str:
    import os

    repo = repo_arg or os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        print(
            "エラー: 公開先リポジトリの owner/repo が指定されていません。"
            "--repo 引数、または環境変数 GITHUB_REPOSITORY を指定してください。",
            file=sys.stderr,
        )
        sys.exit(1)
    return repo


def find_notebook(notebook_number: str) -> Path:
    if not NOTEBOOK_NUMBER_PATTERN.match(notebook_number):
        print(
            f"エラー: notebook_number は3桁の数字で指定してください(指定値: {notebook_number})。",
            file=sys.stderr,
        )
        sys.exit(1)

    matches = sorted(AI_THEORIES_THEORIES_DIR.rglob(f"{notebook_number}_*.ipynb"))

    if not matches:
        print(
            f"エラー: theories/ 配下に {notebook_number}_*.ipynb に一致するノートブックが見つかりませんでした。",
            file=sys.stderr,
        )
        sys.exit(1)
    if len(matches) > 1:
        print(
            f"エラー: {notebook_number}_*.ipynb に一致するノートブックが複数見つかりました。"
            "番号の重複を確認してください。",
            file=sys.stderr,
        )
        for path in matches:
            print(f"  - {path.relative_to(REPO_ROOT)}", file=sys.stderr)
        sys.exit(1)

    return matches[0]


def extract_slug(notebook_path: Path, notebook_number: str) -> str:
    stem = notebook_path.stem
    slug = stem[len(notebook_number) + 1 :]

    if not SLUG_PATTERN.match(slug):
        print(
            f"エラー: ファイル名から抽出した slug '{slug}' が Zenn の要件"
            "(半角英小文字・数字・ハイフン・アンダースコアのみ、12〜50文字)を満たしていません。"
            "articles/ 配下のファイル名を手動で修正してください。",
            file=sys.stderr,
        )
        sys.exit(1)

    return slug


def extract_title(nb: nbformat.NotebookNode) -> str:
    first_cell = nb.cells[0]
    first_line = first_cell.source.splitlines()[0] if first_cell.source else ""
    match = TITLE_HEADING_PATTERN.match(first_line)
    if match:
        return match.group(1)
    return first_line.lstrip("#").strip()


def convert_to_markdown(nb: nbformat.NotebookNode, slug: str, publish_repo: str) -> str:
    exporter = MarkdownExporter()
    body, resources = exporter.from_notebook_node(nb)

    image_dir = IMAGES_DIR / slug
    image_dir.mkdir(parents=True, exist_ok=True)

    for output_name, output_data in resources.get("outputs", {}).items():
        (image_dir / Path(output_name).name).write_bytes(output_data)
        image_url = (
            f"https://raw.githubusercontent.com/{publish_repo}/main/images/{slug}/"
            f"{Path(output_name).name}"
        )
        body = body.replace(output_name, image_url)

    return body


def build_frontmatter(title: str) -> str:
    topics = json.dumps(DEFAULT_TOPICS, ensure_ascii=False)
    return (
        "---\n"
        f'title: "{title}"\n'
        f'emoji: "{DEFAULT_EMOJI}" # 仮の絵文字。公開前に手動で調整すること\n'
        'type: "tech"\n'
        f"topics: {topics} # 初期値。公開前に内容に応じて手動で追記・調整すること\n"
        "published: false\n"
        "---\n\n"
    )


def build_source_link(notebook_path: Path) -> str:
    category = notebook_path.parent.name
    filename = notebook_path.name
    url = (
        "https://github.com/kojikojiprg/ai-theories/blob/main/theories/"
        f"{category}/{filename}"
    )
    return f"\n\n## 元ノートブック(実装の全文はこちら)\n\n{url}\n"


def main() -> None:
    args = parse_args()
    publish_repo = resolve_publish_repo(args.repo)

    notebook_path = find_notebook(args.notebook_number)
    slug = extract_slug(notebook_path, args.notebook_number)

    nb = nbformat.read(notebook_path, as_version=4)
    title = extract_title(nb)
    body = convert_to_markdown(nb, slug, publish_repo)
    body += build_source_link(notebook_path)

    article_path = ARTICLES_DIR / f"{slug}.md"
    article_path.write_text(build_frontmatter(title) + body, encoding="utf-8")

    print(f"{article_path.relative_to(REPO_ROOT)} を生成しました。")
    print(
        "published: false の下書きとして生成しました。"
        "記事として公開する前に、実装セクションの要約・emoji・topicsの調整を"
        "手動で行ってください。"
    )


if __name__ == "__main__":
    main()
