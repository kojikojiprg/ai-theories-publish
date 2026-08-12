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
IMPLEMENTATION_PLAN_HEADING_PATTERN = re.compile(
    r"^##\s*\d+\.\s*実装方針(?:\s*/.*)?\s*$", re.MULTILINE
)

DEFAULT_EMOJI = "📝"
DEFAULT_TOPICS = ["ai", "llm", "vlm", "pytorch", "machine learning"]

# Zennの記事本文の文字数上限(80,000文字)に対して安全マージンを持たせた分割閾値。
CHAR_THRESHOLD = 70_000
CHAR_HARD_LIMIT = 80_000


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


def extract_slug(notebook_path: Path) -> str:
    slug = notebook_path.stem

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


def split_body_at_implementation_plan(body: str) -> tuple[str, str]:
    match = IMPLEMENTATION_PLAN_HEADING_PATTERN.search(body)
    if not match:
        print(
            "エラー: 記事本文が文字数上限に近いため前編・後編への分割を試みましたが、"
            "分割の境界となる「実装方針」の見出しが見つかりませんでした。"
            "ノートブック側の見出し表記(表記ゆれ)を確認してください。",
            file=sys.stderr,
        )
        sys.exit(1)
    return body[: match.start()], body[match.start() :]


def build_theory_part_intro() -> str:
    return (
        "この記事は前編(理論編)です。実装・実験編は近日公開予定です。\n\n"
        "<!-- TODO: 後編公開後にリンクを追加 -->\n\n"
    )


def build_practice_part_intro() -> str:
    return (
        "この記事は後編(実装・実験編)です。前編(理論編)はこちら: "
        "<!-- TODO: 前編のURLに置き換え -->\n\n"
    )


def write_article(article_path: Path, title: str, content_body: str) -> int:
    article_path.write_text(build_frontmatter(title) + content_body, encoding="utf-8")
    return len(content_body)


def check_char_limit(part_label: str, content_body: str) -> None:
    length = len(content_body)
    if length > CHAR_THRESHOLD:
        print(
            f"エラー: {part_label}の本文が {length} 文字となり、"
            f"分割後も閾値({CHAR_THRESHOLD}文字。Zennの上限は{CHAR_HARD_LIMIT}文字)を"
            "超えています。実装または実験セクションの内容をノートブック側か"
            "スクリプト側でさらに絞り込む必要があります。",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    args = parse_args()
    publish_repo = resolve_publish_repo(args.repo)

    notebook_path = find_notebook(args.notebook_number)
    slug = extract_slug(notebook_path)

    nb = nbformat.read(notebook_path, as_version=4)
    title = extract_title(nb)
    body = convert_to_markdown(nb, slug, publish_repo)
    source_link = build_source_link(notebook_path)

    if len(body + source_link) <= CHAR_THRESHOLD:
        article_path = ARTICLES_DIR / f"{slug}.md"
        length = write_article(article_path, title, body + source_link)
        print(f"{article_path.relative_to(REPO_ROOT)} を生成しました({length}文字)。")
        print(
            "published: false の下書きとして生成しました。"
            "記事として公開する前に、実装セクションの要約・emoji・topicsの調整を"
            "手動で行ってください。"
        )
        return

    theory_body, practice_body = split_body_at_implementation_plan(body)
    theory_content = build_theory_part_intro() + theory_body + source_link
    practice_content = build_practice_part_intro() + practice_body + source_link

    check_char_limit("前編(理論編)", theory_content)
    check_char_limit("後編(実装・実験編)", practice_content)

    theory_path = ARTICLES_DIR / f"{slug}-theory.md"
    practice_path = ARTICLES_DIR / f"{slug}-practice.md"

    theory_length = write_article(theory_path, f"{title}(理論編)", theory_content)
    practice_length = write_article(
        practice_path, f"{title}(実装・実験編)", practice_content
    )

    print(
        f"本文が閾値({CHAR_THRESHOLD}文字)を超えたため、"
        "前編(理論編)・後編(実装・実験編)の2記事に分割生成しました。"
    )
    print(f"{theory_path.relative_to(REPO_ROOT)} を生成しました({theory_length}文字)。")
    print(f"{practice_path.relative_to(REPO_ROOT)} を生成しました({practice_length}文字)。")
    print(
        "published: false の下書きとして生成しました。"
        "記事として公開する前に、実装セクションの要約・emoji・topicsの調整、"
        "および前編・後編間の相互リンク(本文中のTODOコメント箇所)の手動追加を"
        "行ってください。"
    )


if __name__ == "__main__":
    main()
