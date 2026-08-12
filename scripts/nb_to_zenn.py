#!/usr/bin/env python3
"""ai-theories のノートブックを Zenn 本(book)の章(Markdown)の下書きに変換するスクリプト。"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import nbformat
from nbconvert import MarkdownExporter

REPO_ROOT = Path(__file__).resolve().parent.parent
AI_THEORIES_THEORIES_DIR = REPO_ROOT / "ai-theories" / "theories"
IMAGES_DIR = REPO_ROOT / "images"
MANIFEST_PATH = REPO_ROOT / "scripts" / "manifest.json"

# 本(book)のスラッグ。scripts/generate_book.py と共通。
BOOK_SLUG = "ai-theories-roadmap"
BOOK_DIR = REPO_ROOT / "books" / BOOK_SLUG

NOTEBOOK_NUMBER_PATTERN = re.compile(r"^\d{3}$")
SLUG_PATTERN = re.compile(r"^[a-z0-9_-]{12,50}$")
TITLE_HEADING_PATTERN = re.compile(r"^#\s*\d{3}\.\s*(.+?)\s*$")
IMPLEMENTATION_PLAN_HEADING_PATTERN = re.compile(
    r"^##\s*\d+\.\s*実装方針(?:\s*/.*)?\s*$", re.MULTILINE
)
NOTEBOOK_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+\.ipynb)\)")
NOTEBOOK_LINK_NUMBER_PATTERN = re.compile(r"^(\d{3})_")
NOT_YET_CREATED_PATTERN = re.compile(r"( ?)\(未作成(?:\s*/\s*TBD)?\)( ?)")

# Zennの記事本文の文字数上限(80,000文字)に対して安全マージンを持たせた分割閾値。
CHAR_THRESHOLD = 70_000
CHAR_HARD_LIMIT = 80_000

# Zennのユーザー名。将来変わる可能性があるため、環境変数での上書きも可能にしておく。
ZENN_USERNAME = os.environ.get("ZENN_USERNAME", "kojikojiprg")


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


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def save_manifest(manifest: dict) -> None:
    # "slugs" は本(book)の章ファイル名(拡張子を除く)、すなわち
    # books/ai-theories-roadmap/<slug>.md を指す値であり、
    # Zennの章URL(.../books/ai-theories-roadmap/viewer/<slug>)の組み立てに使われる。
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def find_notebook_by_number(number: str) -> Path | None:
    matches = sorted(AI_THEORIES_THEORIES_DIR.rglob(f"{number}_*.ipynb"))
    if len(matches) != 1:
        return None
    return matches[0]


def resolve_notebook_link(relpath: str, manifest: dict) -> str | None:
    basename = Path(relpath).name
    number_match = NOTEBOOK_LINK_NUMBER_PATTERN.match(basename)
    if not number_match:
        return None
    number = number_match.group(1)

    target_notebook = find_notebook_by_number(number)
    if target_notebook is None:
        print(
            f"警告: リンク先ノートブック(番号: {number})が見つかりません。"
            "リンクは変更しません。",
            file=sys.stderr,
        )
        return None

    entry = manifest.get(number)
    if entry is not None:
        slugs = entry["slugs"]
        if entry["split"]:
            theory_slugs = [s for s in slugs if s.endswith("-theory")]
            chosen_slug = theory_slugs[0] if theory_slugs else slugs[0]
        else:
            chosen_slug = slugs[0]
        return build_zenn_book_chapter_url(chosen_slug)

    category = target_notebook.parent.name
    filename = target_notebook.name
    return (
        "https://github.com/kojikojiprg/ai-theories/blob/main/theories/"
        f"{category}/{filename}"
    )


def rewrite_notebook_links(body: str, manifest: dict) -> str:
    def replace(match: re.Match) -> str:
        link_text, relpath = match.group(1), match.group(2)
        new_url = resolve_notebook_link(relpath, manifest)
        if new_url is None:
            return match.group(0)
        return f"[{link_text}]({new_url})"

    return NOTEBOOK_LINK_PATTERN.sub(replace, body)


def remove_not_yet_created_markers(body: str) -> str:
    def replace(match: re.Match) -> str:
        leading_space, trailing_space = match.group(1), match.group(2)
        return " " if leading_space or trailing_space else ""

    return NOT_YET_CREATED_PATTERN.sub(replace, body)


def extract_slug(notebook_path: Path) -> str:
    slug = notebook_path.stem

    if not SLUG_PATTERN.match(slug):
        print(
            f"エラー: ファイル名から抽出した slug '{slug}' が Zenn の要件"
            "(半角英小文字・数字・ハイフン・アンダースコアのみ、12〜50文字)を満たしていません。"
            "theories/ 配下のノートブックのファイル名を確認してください。",
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
    return f'---\ntitle: "{title}"\n---\n\n'


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
            "エラー: 章の本文が文字数上限に近いため前編・後編への分割を試みましたが、"
            "分割の境界となる「実装方針」の見出しが見つかりませんでした。"
            "ノートブック側の見出し表記(表記ゆれ)を確認してください。",
            file=sys.stderr,
        )
        sys.exit(1)
    return body[: match.start()], body[match.start() :]


def build_zenn_book_chapter_url(slug: str) -> str:
    return f"https://zenn.dev/{ZENN_USERNAME}/books/{BOOK_SLUG}/viewer/{slug}"


def build_theory_part_intro(practice_slug: str) -> str:
    url = build_zenn_book_chapter_url(practice_slug)
    return f"この記事は前編(理論編)です。実装・実験編は [こちら]({url})。\n\n"


def build_practice_part_intro(theory_slug: str) -> str:
    url = build_zenn_book_chapter_url(theory_slug)
    return f"この記事は後編(実装・実験編)です。前編(理論編)は [こちら]({url})。\n\n"


def write_chapter(chapter_path: Path, title: str, content_body: str) -> int:
    chapter_path.write_text(build_frontmatter(title) + content_body, encoding="utf-8")
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
    manifest = load_manifest()

    BOOK_DIR.mkdir(parents=True, exist_ok=True)

    notebook_path = find_notebook(args.notebook_number)
    slug = extract_slug(notebook_path)

    nb = nbformat.read(notebook_path, as_version=4)
    title = extract_title(nb)
    body = convert_to_markdown(nb, slug, publish_repo)
    body = rewrite_notebook_links(body, manifest)
    body = remove_not_yet_created_markers(body)
    source_link = build_source_link(notebook_path)

    if len(body + source_link) <= CHAR_THRESHOLD:
        manifest[args.notebook_number] = {
            "slugs": [slug],
            "split": False,
            "titles": [title],
        }

        chapter_path = BOOK_DIR / f"{slug}.md"
        length = write_chapter(chapter_path, title, body + source_link)
        print(f"{chapter_path.relative_to(REPO_ROOT)} を生成しました({length}文字)。")
        print(
            "本(book)の下書きの章として生成しました。"
            "公開前に、実装セクションの要約などの調整を手動で行ってください。"
        )

        save_manifest(manifest)
        print(f"{MANIFEST_PATH.relative_to(REPO_ROOT)} を更新しました。")
        return

    theory_slug = f"{slug}-theory"
    practice_slug = f"{slug}-practice"
    theory_title = f"{title}(理論編)"
    practice_title = f"{title}(実装・実験編)"

    theory_body, practice_body = split_body_at_implementation_plan(body)
    theory_content = build_theory_part_intro(practice_slug) + theory_body + source_link
    practice_content = (
        build_practice_part_intro(theory_slug) + practice_body + source_link
    )

    check_char_limit("前編(理論編)", theory_content)
    check_char_limit("後編(実装・実験編)", practice_content)

    manifest[args.notebook_number] = {
        "slugs": [theory_slug, practice_slug],
        "split": True,
        "titles": [theory_title, practice_title],
    }

    theory_path = BOOK_DIR / f"{theory_slug}.md"
    practice_path = BOOK_DIR / f"{practice_slug}.md"

    theory_length = write_chapter(theory_path, theory_title, theory_content)
    practice_length = write_chapter(practice_path, practice_title, practice_content)

    print(
        f"本文が閾値({CHAR_THRESHOLD}文字)を超えたため、"
        "前編(理論編)・後編(実装・実験編)の2章に分割生成しました。"
    )
    print(f"{theory_path.relative_to(REPO_ROOT)} を生成しました({theory_length}文字)。")
    print(f"{practice_path.relative_to(REPO_ROOT)} を生成しました({practice_length}文字)。")
    print(
        "本(book)の下書きの章として生成しました。"
        "前編・後編間の相互リンクは確定URLで埋め込み済みです。"
        "公開前に、実装セクションの要約などの調整を手動で行ってください。"
    )

    save_manifest(manifest)
    print(f"{MANIFEST_PATH.relative_to(REPO_ROOT)} を更新しました。")


if __name__ == "__main__":
    main()
