#!/usr/bin/env python3
"""ai-theories のノートブックを Zenn 本(book)の章(Markdown)の下書きに変換するスクリプト。"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import zlib
from pathlib import Path

import nbformat
from nbconvert import MarkdownExporter
from traitlets.config import Config

import png_check

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
TOP_HEADING_PATTERN = re.compile(r"^##\s+\S.*$", re.MULTILINE)
SUB_HEADING_PATTERN = re.compile(r"^###\s+\S.*$", re.MULTILINE)
# リンク先のパスと、(あれば)見出しへのアンカー(`#...`)を分けて捉える。
NOTEBOOK_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)#\s]+\.ipynb)(#[^)\s]*)?\)")
NOTEBOOK_LINK_NUMBER_PATTERN = re.compile(r"^(\d{3})_")
NOT_YET_CREATED_PATTERN = re.compile(r"( ?)\(未作成(?:\s*/\s*TBD)?\)( ?)")

# Zennの本(book)の章(chapter)本文の文字数上限。実際のデプロイエラー
# (「本文のmarkdownには最大50000文字まで使用できます」)により判明した値。
# 記事(article)の80,000文字上限とは異なる、より厳しい制限であることに注意。
# Zennの章のslug(ファイル名から拡張子を除いたもの)の文字数の上限。
CHAPTER_SLUG_MAX_LENGTH = 50
# Zennの章のタイトルの文字数の上限(「Titleには最大70文字まで使用できます」)。
CHAPTER_TITLE_MAX_LENGTH = 70
CHAR_HARD_LIMIT = 50_000
# 分割要否の判定・パッキングに使う閾値。前後のナビゲーション文など
# 分割後に追加されるテキストの分だけ、CHAR_HARD_LIMIT に安全マージンを持たせている。
CHAR_THRESHOLD = 45_000
# `##`/`###`見出し単位でのパッキング(split_body_into_chunks)に使う閾値。
# パッキング後に build_part_intro() のナビゲーション文が追加されるため、
# CHAR_THRESHOLD よりさらに余裕を持たせておく必要がある
# (そうしないと、パッキング結果ぎりぎりの章にナビ文を足しただけで CHAR_THRESHOLD を
# 超えてしまうことがある)。
PART_OVERHEAD_MARGIN = 800
PACKING_THRESHOLD = CHAR_THRESHOLD - PART_OVERHEAD_MARGIN

# セルのタグによる Zenn での表示の制御(ai-theories の「ノートブックの圧縮と保守」で定めるタグ)。
TAG_HIDE = "zenn-hide"
TAG_HIDE_INPUT = "zenn-hide-input"
TAG_DETAILS = "zenn-details"
TAG_APPENDIX = "zenn-appendix"
TAG_IMAGES_ONLY = "zenn-images-only"
# タグごとに付けてよいセルの種類(ai-theories の CLAUDE.md「Zenn での表示を制御するセルのタグ」に従う)。
ZENN_TAG_TARGETS = {
    TAG_HIDE: ("code", "markdown", "raw"),
    TAG_HIDE_INPUT: ("code",),
    TAG_IMAGES_ONLY: ("code",),
    TAG_DETAILS: ("code", "markdown", "raw"),
    TAG_APPENDIX: ("markdown",),
}

# 折りたたみ(`:::details`)の見出しの既定値。
# 付録は記事に載せず、代わりにこの文とノートブックへのリンクを置く。
APPENDIX_REPLACEMENT_TEXT = (
    "付録(実行の手順・パイロット・設計の改訂の記録など)は、GitHub のノートブックを参照。"
)
CODE_CELL_DETAILS_SUMMARY = "コード"
MARKDOWN_CELL_DETAILS_SUMMARY = "詳細"

# 折りたたみの開始・終了を本文中に仮置きするための印。変換後に `:::details` へ置き換える。
FOLD_MARKER_PATTERN = re.compile(r"^@@ZENN_FOLD_(OPEN|CLOSE):(\d+)@@$")
FIRST_HEADING_PATTERN = re.compile(r"^#{1,6}[ \t]+(\S.*?)[ \t]*$", re.MULTILINE)
FOLD_OPEN_LINE_PATTERN = re.compile(r"^:::+details\b")
FOLD_CLOSE_LINE_PATTERN = re.compile(r"^:::+$")
CODE_FENCE_PATTERN = re.compile(r"^\s*(```+|~~~+)")

# Zennのユーザー名。将来変わる可能性があるため、環境変数での上書きも可能にしておく。
ZENN_USERNAME = os.environ.get("ZENN_USERNAME", "kojikojiprg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebook_number",
        help=(
            "変換対象ノートブックの3桁連番(例: 003)。`all` を指定すると、全ノートブックを"
            "2回に分けて変換する(1回目で章の構成と章名を確定し、2回目で、後の番号の"
            "ノートブックへのリンクもZennの章のURLに解決する)。"
        ),
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


def build_notebook_github_url(target_notebook: Path) -> str:
    category = target_notebook.parent.name
    filename = target_notebook.name
    return (
        "https://github.com/kojikojiprg/ai-theories/blob/main/theories/"
        f"{category}/{filename}"
    )


def resolve_notebook_link(relpath: str, manifest: dict, anchor: str = "") -> tuple[str, str] | None:
    """ノートブックへの相対リンクの書き換え先と、その種別("zenn" または "github")を返す。
    書き換えない場合は None を返す。

    - 書き換え先の章が `manifest` にあり、アンカーがない場合: Zennの章のURL("zenn")。
    - アンカー(ノートブック内の見出しへのリンク)がある場合: Zennの章のURLではアンカーを
      表せないため、GitHubのURLにアンカーを付けたもの("github")。
    - 章が `manifest` にない場合: GitHubのURL("github")。
    """
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
    if entry is not None and not anchor:
        slugs = entry["slugs"]
        if entry["split"]:
            theory_slugs = [s for s in slugs if s.endswith("-theory")]
            chosen_slug = theory_slugs[0] if theory_slugs else slugs[0]
        else:
            chosen_slug = slugs[0]
        return build_zenn_book_chapter_url(chosen_slug), "zenn"

    return build_notebook_github_url(target_notebook) + anchor, "github"


def rewrite_notebook_links(body: str, manifest: dict) -> tuple[str, dict[str, int]]:
    """本文中のノートブックへのリンクを書き換え、(書き換え後の本文, 種別ごとの件数)を返す。"""
    counts = {"zenn": 0, "github": 0, "unchanged": 0}

    def replace(match: re.Match) -> str:
        link_text, relpath, anchor = match.group(1), match.group(2), match.group(3) or ""
        resolved = resolve_notebook_link(relpath, manifest, anchor)
        if resolved is None:
            counts["unchanged"] += 1
            return match.group(0)
        new_url, kind = resolved
        counts[kind] += 1
        return f"[{link_text}]({new_url})"

    return NOTEBOOK_LINK_PATTERN.sub(replace, body), counts


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


def cell_tags(cell: nbformat.NotebookNode) -> set[str]:
    return set(cell.get("metadata", {}).get("tags", []))


def details_summary_and_cell(cell: nbformat.NotebookNode) -> tuple[str, nbformat.NotebookNode]:
    """`zenn-details` のセルについて、折りたたみの見出しの文言と、(必要なら見出し行を除いた)セルを返す。

    Markdownセルは最初の見出しを文言にし、その見出し行はセルから除く(見出しの重複と、
    折りたたみの内側に章の分割境界となる`##`見出しが残ることを避けるため)。
    見出しのないMarkdownセルは「詳細」、コードセルは「コード」とする。
    """
    if cell.cell_type == "code":
        return CODE_CELL_DETAILS_SUMMARY, cell
    if cell.cell_type != "markdown":
        return MARKDOWN_CELL_DETAILS_SUMMARY, cell
    match = FIRST_HEADING_PATTERN.search(cell.source)
    if not match:
        return MARKDOWN_CELL_DETAILS_SUMMARY, cell
    remaining = cell.source[: match.start()] + cell.source[match.end() :].lstrip("\n")
    return match.group(1), nbformat.v4.new_markdown_cell(remaining, metadata=cell.metadata)


def validate_zenn_tags(nb: nbformat.NotebookNode) -> None:
    """タグの使い方が CLAUDE.md の定めに合わなければ、エラーを印字して終了する。
    定め: 未知の `zenn-` のタグと、種類の合わないセルへのタグは認めない。
    `zenn-appendix` は 1 つのノートブックに 1 つまでで、見出し(`#` で始まる行)の Markdown セルに付ける。
    """
    problems = []
    appendix_indexes = []
    for index, cell in enumerate(nb.cells):
        for tag in sorted(cell_tags(cell)):
            if tag in ZENN_TAG_TARGETS:
                if cell.cell_type not in ZENN_TAG_TARGETS[tag]:
                    problems.append(
                        f"セル {index}: タグ {tag} は {cell.cell_type} セルには付けられません"
                        f"(付けてよい種類: {list(ZENN_TAG_TARGETS[tag])})。"
                    )
                if tag == TAG_APPENDIX:
                    appendix_indexes.append(index)
            elif tag.startswith("zenn-"):
                problems.append(f"セル {index}: 未知の Zenn のタグ {tag} です。")
    if len(appendix_indexes) > 1:
        problems.append(
            f"{TAG_APPENDIX} が {len(appendix_indexes)} 個のセル({appendix_indexes})にあります。"
            "1 つのノートブックに 1 つまでです。"
        )
    for index in appendix_indexes:
        lines = [line for line in nb.cells[index].source.splitlines() if line.strip()]
        if not lines or not lines[0].lstrip().startswith("#"):
            problems.append(
                f"セル {index}: {TAG_APPENDIX} は見出し(# で始まる行)の Markdown セルに付けます。"
            )
    if problems:
        for problem in problems:
            print(f"エラー: {problem}", file=sys.stderr)
        sys.exit(1)


def keep_images_only(cell: nbformat.NotebookNode) -> nbformat.NotebookNode | None:
    """`zenn-images-only` のコードセルについて、ソースと画像以外の出力を除いたセルを返す。
    画像の出力がなければ None(セルごと除く)を返す。
    """
    outputs = []
    for output in cell.get("outputs", []):
        if output.get("output_type") not in ("display_data", "execute_result"):
            continue
        images = {k: v for k, v in output.get("data", {}).items() if k.startswith("image/")}
        if not images:
            continue
        kept = nbformat.from_dict(output)
        kept["data"] = images
        kept["metadata"] = {}
        outputs.append(kept)
    if not outputs:
        return None
    kept_cell = nbformat.from_dict(cell)
    kept_cell["outputs"] = outputs
    tags = list(kept_cell.get("metadata", {}).get("tags", []))
    kept_cell["metadata"]["tags"] = tags + [TAG_HIDE_INPUT]
    return kept_cell


def apply_zenn_tags(
    nb: nbformat.NotebookNode, notebook_path: Path | None = None
) -> tuple[nbformat.NotebookNode, dict[int, str]]:
    """タグに従ってセルを加工する。
    `zenn-hide` のセルは除く。`zenn-images-only` のセルは画像の出力だけを残す。
    `zenn-details` のセルには折りたたみの開始・終了の印のセルを挿入する。
    `zenn-appendix` のセルから末尾までは除き、代わりにノートブックへのリンクの段落を置く。
    折りたたみの見出しの文言は、印の番号をキーとした辞書で返す。
    タグ付きのセルがなければ、ノートブックをそのまま返す。
    """
    if not any(tag.startswith("zenn-") for cell in nb.cells for tag in cell_tags(cell)):
        return nb, {}
    validate_zenn_tags(nb)

    summaries: dict[int, str] = {}
    cells: list[nbformat.NotebookNode] = []

    def marker(kind: str, number: int) -> nbformat.NotebookNode:
        return nbformat.v4.new_markdown_cell(f"@@ZENN_FOLD_{kind}:{number}@@")

    for cell in nb.cells:
        tags = cell_tags(cell)
        if TAG_APPENDIX in tags:
            text = APPENDIX_REPLACEMENT_TEXT
            if notebook_path is not None:
                text += "\n\n" + build_notebook_url(notebook_path)
            cells.append(nbformat.v4.new_markdown_cell(text))
            break
        if TAG_HIDE in tags:
            continue
        if TAG_IMAGES_ONLY in tags:
            cell = keep_images_only(cell)
            if cell is None:
                continue
        if TAG_DETAILS in tags:
            summary, cell = details_summary_and_cell(cell)
            number = len(summaries)
            summaries[number] = summary
            cells.extend([marker("OPEN", number), cell, marker("CLOSE", number)])
        else:
            cells.append(cell)

    nb = nbformat.from_dict(nb)
    nb.cells = cells
    return nb, summaries


def render_folds(body: str, summaries: dict[int, str]) -> str:
    """折りたたみの印を `:::details` に置き換える。
    入れ子の場合は、外側のコロンの数を内側より 1 つ多くする。コードブロック内の印は置き換えない。
    """
    lines = body.splitlines(keepends=True)
    markers: dict[int, tuple[str, int]] = {}
    in_code = False
    for index, line in enumerate(lines):
        stripped = line.rstrip("\n")
        if CODE_FENCE_PATTERN.match(stripped):
            in_code = not in_code
            continue
        match = None if in_code else FOLD_MARKER_PATTERN.match(stripped)
        if match:
            markers[index] = (match.group(1), int(match.group(2)))
    if not markers:
        return body

    # 各折りたたみの内側にある折りたたみの最大の段数。
    inner_levels: dict[int, int] = {}
    stack: list[int] = []
    for kind, number in markers.values():
        if kind == "OPEN":
            inner_levels[number] = 0
            stack.append(number)
        else:
            stack.pop()
            if stack:
                inner_levels[stack[-1]] = max(inner_levels[stack[-1]], inner_levels[number] + 1)

    for index, (kind, number) in markers.items():
        colons = ":" * (3 + inner_levels[number])
        lines[index] = (
            f"{colons}details {summaries[number]}\n" if kind == "OPEN" else f"{colons}\n"
        )
    return "".join(lines)


def find_fold_spans(text: str) -> list[tuple[int, int]]:
    """最も外側の折りたたみ(`:::details`〜`:::`)の範囲(開始位置, 終了位置)を返す。"""
    spans = []
    depth = 0
    in_code = False
    offset = 0
    span_start = 0
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if CODE_FENCE_PATTERN.match(stripped):
            in_code = not in_code
        elif not in_code:
            if FOLD_OPEN_LINE_PATTERN.match(stripped):
                if depth == 0:
                    span_start = offset
                depth += 1
            elif FOLD_CLOSE_LINE_PATTERN.match(stripped) and depth > 0:
                depth -= 1
                if depth == 0:
                    spans.append((span_start, offset + len(line)))
        offset += len(line)
    return spans


MERMAID_THEME_DIRECTIVE = "%%{init: {'theme': 'neutral'}}%%"
MERMAID_BLOCK_PATTERN = re.compile(r"^```mermaid[ \t]*\n(.*?)^```", re.MULTILINE | re.DOTALL)
MERMAID_OPEN_PATTERN = re.compile(r"^```mermaid[ \t]*\n", re.MULTILINE)


def add_mermaid_theme(body: str) -> str:
    """Mermaid のコードブロックの先頭に、neutral テーマの指示を足す。

    既定のテーマは、暗い背景でノードの外の矢印が背景に溶けて見えなくなる。neutral は明るい背景と
    暗い背景のどちらでも、線と文字が見える。すでにテーマの指示(`%%{init`)があるブロックは変えない。
    """

    def replace(match: re.Match) -> str:
        rest = body[match.end() :]
        first_line = rest.split("\n", 1)[0].lstrip()
        if first_line.startswith("%%{init"):
            return match.group(0)
        return match.group(0) + MERMAID_THEME_DIRECTIVE + "\n"

    return MERMAID_OPEN_PATTERN.sub(replace, body)


def check_png_is_opaque(name: str, data: bytes, slug: str) -> None:
    """透明な画素を含む PNG は検証エラーにする。暗い背景で図が読めなくなるため。
    変換の側で白に塗って直すことはしない(元のノートブックの図の背景を不透明にして直す)。
    """
    if not name.lower().endswith(".png"):
        return
    try:
        transparent = png_check.has_transparent_pixels(data)
    except (ValueError, zlib.error, struct.error, IndexError) as error:
        print(f"エラー: 画像 {slug}/{name} の PNG を検査できませんでした({error})。", file=sys.stderr)
        sys.exit(1)
    if transparent:
        print(
            f"エラー: 画像 {slug}/{name} に透明な画素があります。ノートブックの図の背景を"
            "不透明な白にして保存し直してください(変換では塗りません)。",
            file=sys.stderr,
        )
        sys.exit(1)


def convert_to_markdown(
    nb: nbformat.NotebookNode, slug: str, publish_repo: str, notebook_path: Path | None = None
) -> str:
    nb, summaries = apply_zenn_tags(nb, notebook_path)

    config = Config()
    config.TagRemovePreprocessor.enabled = True
    config.TagRemovePreprocessor.remove_input_tags = {TAG_HIDE_INPUT}
    exporter = MarkdownExporter(config=config)
    body, resources = exporter.from_notebook_node(nb)

    image_dir = IMAGES_DIR / slug
    image_dir.mkdir(parents=True, exist_ok=True)

    for output_name, output_data in resources.get("outputs", {}).items():
        check_png_is_opaque(Path(output_name).name, output_data, slug)
        (image_dir / Path(output_name).name).write_bytes(output_data)
        image_url = (
            f"https://raw.githubusercontent.com/{publish_repo}/main/images/{slug}/"
            f"{Path(output_name).name}"
        )
        body = body.replace(output_name, image_url)

    return add_mermaid_theme(render_folds(body, summaries))


def build_frontmatter(title: str) -> str:
    return f'---\ntitle: "{title}"\n---\n\n'


def build_notebook_url(notebook_path: Path) -> str:
    return build_notebook_github_url(notebook_path)


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


def split_at_headings(text: str, pattern: re.Pattern) -> list[str]:
    """`pattern` に一致する見出し単位で `text` を分割する。
    先頭の見出しより前のテキスト(存在する場合)も 1 つのチャンクとして扱う。
    """
    fold_spans = find_fold_spans(text)
    matches = [
        m
        for m in pattern.finditer(text)
        if not any(start <= m.start() < end for start, end in fold_spans)
    ]
    if not matches:
        return [text]

    chunks = []
    if matches[0].start() > 0:
        chunks.append(text[: matches[0].start()])
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunks.append(text[match.start() : end])
    return chunks


def pack_sections(sections: list[str], threshold: int) -> list[str]:
    """連続するセクションを、連結後の文字数が threshold を超えない範囲で
    先頭から貪欲に詰め込む。"""
    groups: list[str] = []
    current = ""
    for section in sections:
        candidate = current + section
        if current and len(candidate) > threshold:
            groups.append(current)
            current = section
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups


def split_body_into_chunks(body: str, threshold: int) -> list[str]:
    """`##`見出し単位で body を閾値以下のチャンクにパッキングする。
    1つの`##`セクションが単体で閾値を超える場合は、`###`見出し単位でさらに
    分割してからパッキングする(それでも単体で閾値を超えるセクションが
    残った場合は、そのまま1チャンクとして返し、呼び出し側の文字数検証に委ねる)。
    """
    top_sections = split_at_headings(body, TOP_HEADING_PATTERN)

    expanded: list[str] = []
    for section in top_sections:
        if len(section) > threshold:
            sub_sections = split_at_headings(section, SUB_HEADING_PATTERN)
            if len(sub_sections) > 1:
                expanded.extend(sub_sections)
                continue
        expanded.append(section)

    return pack_sections(expanded, threshold)


def build_zenn_book_chapter_url(slug: str) -> str:
    return f"https://zenn.dev/{ZENN_USERNAME}/books/{BOOK_SLUG}/viewer/{slug}"


def shorten_slug_for_chapters(slug: str, longest_suffix: int) -> str:
    """章のslugは Zenn の上限(50文字)以内でなければならない。章のslugは `<slug><接尾辞>` なので、
    ノートブックのファイル名が長く、接尾辞(`-practice-1` など)を付けると超える場合は、
    `_` 区切りの語の境界で切り詰めた短いslugを使う。収まる場合はそのまま返す。
    画像のフォルダ(`images/<slug>/`)は元のslugのままで、この短縮の影響を受けない。
    """
    limit = CHAPTER_SLUG_MAX_LENGTH - longest_suffix
    if len(slug) <= limit:
        return slug
    shortened = slug[:limit]
    if "_" in shortened and slug[limit] != "_":
        shortened = shortened.rsplit("_", 1)[0]
    return shortened


def build_part_slug(base_slug: str, kind: str, index: int, total: int) -> str:
    if total == 1:
        return f"{base_slug}-{kind}"
    return f"{base_slug}-{kind}-{index + 1}"


def fit_title(base_title: str, suffix: str = "") -> str:
    """章のタイトル(`base_title` + `suffix`)を、「日本語 / English」形式の日本語の部分だけにする。
    Zennの上限(70文字)を超える場合はエラーにして止める。
    """
    title = base_title.split(" / ", 1)[0] + suffix
    if len(title) > CHAPTER_TITLE_MAX_LENGTH:
        print(
            f"エラー: 章のタイトル「{title}」が {len(title)} 文字で、Zennの上限"
            f"({CHAPTER_TITLE_MAX_LENGTH}文字)を超えています。ノートブックの見出しを短くしてください。",
            file=sys.stderr,
        )
        sys.exit(1)
    return title


def build_part_title(base_title: str, kind: str, index: int, total: int) -> str:
    kind_label = "理論編" if kind == "theory" else "実装・実験編"
    if total == 1:
        return fit_title(base_title, f"({kind_label})")
    return fit_title(base_title, f"({kind_label} {index + 1}/{total})")


def build_part_intro(part: dict, prev_part: dict | None, next_part: dict | None) -> str:
    part_label = "前編" if part["kind"] == "theory" else "後編"
    kind_label = "理論編" if part["kind"] == "theory" else "実装・実験編"
    if part["total"] > 1:
        kind_label = f"{kind_label} {part['index'] + 1}/{part['total']}"

    lines = [f"この記事は{part_label}({kind_label})です。"]
    if prev_part is not None:
        url = build_zenn_book_chapter_url(prev_part["slug"])
        lines.append(f"前の内容は [こちら]({url})。")
    if next_part is not None:
        url = build_zenn_book_chapter_url(next_part["slug"])
        lines.append(f"続きは [こちら]({url})。")
    return "".join(lines) + "\n\n"


def write_chapter(chapter_path: Path, title: str, content_body: str) -> int:
    content_body = content_body.rstrip() + "\n"
    chapter_path.write_text(build_frontmatter(title) + content_body, encoding="utf-8")
    return len(content_body)


def check_char_limit(part_label: str, content_body: str) -> None:
    length = len(content_body)
    if length > CHAR_THRESHOLD:
        print(
            f"エラー: {part_label}の本文が {length} 文字となり、"
            f"分割後も閾値({CHAR_THRESHOLD}文字。Zennの本(book)の章の上限は"
            f"{CHAR_HARD_LIMIT}文字)を超えています。1つの`###`小節だけで"
            "閾値を超えているなど、見出し単位の自動分割では対応できない可能性があります。"
            "ノートブック側の内容を絞り込むか、見出しを追加して分割可能な単位を"
            "細かくしてください。",
            file=sys.stderr,
        )
        sys.exit(1)


def check_structure_unchanged(number: str, manifest: dict, new_slugs: list[str]) -> None:
    """2回目の変換で、1回目に確定した章の構成(章名)が変わっていないことを確かめる。
    リンクの置き換えで文字数が変わり、分割の結果が変わると、他のノートブックのリンク先も
    変わってしまうため、変わった場合はエラーにして止める。
    """
    expected = manifest.get(number, {}).get("slugs")
    if expected != new_slugs:
        print(
            f"エラー: ノートブック {number} の章の構成が、リンクの置き換えで変わりました"
            f"(1回目: {expected}、2回目: {new_slugs})。リンクの置き換えで文字数の閾値"
            f"({CHAR_THRESHOLD}文字)を超えた、または下回った可能性があります。",
            file=sys.stderr,
        )
        sys.exit(1)


def print_link_counts(number: str, counts: dict[str, int]) -> None:
    print(
        f"ノートブック間リンク({number}): Zennの章のURL {counts['zenn']}件、"
        f"GitHubのURL {counts['github']}件、変更なし {counts['unchanged']}件。"
    )


def convert_notebook(number: str, publish_repo: str, expect_same_structure: bool = False) -> None:
    manifest = load_manifest()

    BOOK_DIR.mkdir(parents=True, exist_ok=True)

    notebook_path = find_notebook(number)
    slug = extract_slug(notebook_path)

    nb = nbformat.read(notebook_path, as_version=4)
    title = fit_title(extract_title(nb))
    body = convert_to_markdown(nb, slug, publish_repo, notebook_path)
    body, link_counts = rewrite_notebook_links(body, manifest)
    print_link_counts(number, link_counts)
    body = remove_not_yet_created_markers(body)

    if len(body) <= CHAR_THRESHOLD:
        if expect_same_structure:
            check_structure_unchanged(number, manifest, [slug])
        manifest[number] = {
            "slugs": [slug],
            "split": False,
            "titles": [title],
        }

        chapter_path = BOOK_DIR / f"{slug}.md"
        length = write_chapter(chapter_path, title, body)
        print(f"{chapter_path.relative_to(REPO_ROOT)} を生成しました({length}文字)。")
        print(
            "本(book)の下書きの章として生成しました。"
            "公開前に、実装セクションの要約などの調整を手動で行ってください。"
        )

        save_manifest(manifest)
        print(f"{MANIFEST_PATH.relative_to(REPO_ROOT)} を更新しました。")
        return

    theory_body, practice_body = split_body_at_implementation_plan(body)
    theory_chunks = split_body_into_chunks(theory_body, PACKING_THRESHOLD)
    practice_chunks = split_body_into_chunks(practice_body, PACKING_THRESHOLD)

    parts: list[dict] = []
    for i, chunk in enumerate(theory_chunks):
        parts.append({"kind": "theory", "index": i, "total": len(theory_chunks), "body": chunk})
    for i, chunk in enumerate(practice_chunks):
        parts.append({"kind": "practice", "index": i, "total": len(practice_chunks), "body": chunk})

    longest_suffix = max(
        len(build_part_slug("", part["kind"], part["index"], part["total"])) for part in parts
    )
    chapter_base = shorten_slug_for_chapters(slug, longest_suffix)
    for part in parts:
        part["slug"] = build_part_slug(chapter_base, part["kind"], part["index"], part["total"])
        part["title"] = build_part_title(title, part["kind"], part["index"], part["total"])

    if expect_same_structure:
        check_structure_unchanged(number, manifest, [part["slug"] for part in parts])

    for i, part in enumerate(parts):
        prev_part = parts[i - 1] if i > 0 else None
        next_part = parts[i + 1] if i + 1 < len(parts) else None
        content = build_part_intro(part, prev_part, next_part) + part["body"]
        check_char_limit(part["title"], content)

        chapter_path = BOOK_DIR / f"{part['slug']}.md"
        length = write_chapter(chapter_path, part["title"], content)
        print(f"{chapter_path.relative_to(REPO_ROOT)} を生成しました({length}文字)。")

    print(
        f"本文が閾値({CHAR_THRESHOLD}文字)を超えたため、"
        f"{len(parts)}章(理論編{len(theory_chunks)}章・実装/実験編{len(practice_chunks)}章)に"
        "分割生成しました。"
    )
    print(
        "本(book)の下書きの章として生成しました。"
        "各章の冒頭には前後の章への相互リンクを確定URLで埋め込み済みです。"
        "公開前に、実装セクションの要約などの調整を手動で行ってください。"
    )

    manifest[number] = {
        "slugs": [part["slug"] for part in parts],
        "split": True,
        "titles": [part["title"] for part in parts],
    }
    save_manifest(manifest)
    print(f"{MANIFEST_PATH.relative_to(REPO_ROOT)} を更新しました。")


def main() -> None:
    args = parse_args()
    publish_repo = resolve_publish_repo(args.repo)

    if args.notebook_number != "all":
        convert_notebook(args.notebook_number, publish_repo)
        return

    numbers = sorted(
        {m.group(1) for p in AI_THEORIES_THEORIES_DIR.rglob("*.ipynb")
         if (m := NOTEBOOK_LINK_NUMBER_PATTERN.match(p.name))}
    )
    print(f"== 1回目: 章の構成と章名を確定する({len(numbers)}本) ==")
    for number in numbers:
        convert_notebook(number, publish_repo)
    print("== 2回目: 後の番号のノートブックへのリンクを解決する ==")
    for number in numbers:
        convert_notebook(number, publish_repo, expect_same_structure=True)


if __name__ == "__main__":
    main()
