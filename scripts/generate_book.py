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
import hashlib
import re
import subprocess
import sys
from pathlib import Path

import nb_to_zenn
import png_check

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
# 画像の URL に使う公開先リポジトリ。環境変数 GITHUB_REPOSITORY があればそちらを使う。
DEFAULT_PUBLISH_REPO = "kojikojiprg/ai-theories-publish"

# config.yaml が存在しない場合の初期値。既存の config.yaml があればそちらの値を優先する。
DEFAULT_BOOK_TITLE = "LLM / VLM理論学習ロードマップ"
DEFAULT_BOOK_SUMMARY = (
    "LLM(大規模言語モデル)・VLM(視覚言語モデル)の理論を、"
    "PyTorchによるスクラッチ実装とあわせて基礎から学ぶロードマップです。"
)
DEFAULT_BOOK_TOPICS = ["llm", "vlm", "pytorch", "deeplearning"]
DEFAULT_PRICE = "0"
DEFAULT_PUBLISHED = "false"

# 「はじめに」章の本文の文字数の閾値。超える場合は「各トピックの詳細」を載せない(章は分割しない)。
CHAR_THRESHOLD = nb_to_zenn.CHAR_THRESHOLD

# Zenn の Mermaid の制限(https://zenn.dev/zenn/articles/markdown-guide の「制限事項」)。
MERMAID_MAX_CHARS = 2000  # 1 ブロックあたりの文字数
MERMAID_MAX_CHAINS = 10  # 1 ブロックあたりの & の数(Chain 数)

# 全体像の図は、Zenn の埋め込みの Mermaid が描画に失敗することがあるため、画像にして埋め込む。
# 横長の図は Zenn の本文の幅で小さくなるため、描画のときだけ左右(LR)を上下(TB)にする。
MERMAID_CLI_PACKAGE = "@mermaid-js/mermaid-cli@11.12.0"
# Chrome の起動の設定ファイル(JSON)のパス。CI では --no-sandbox を指定するために使う。
PUPPETEER_CONFIG_ENV = "MERMAID_PUPPETEER_CONFIG"
OVERVIEW_IMAGE_DIR = nb_to_zenn.IMAGES_DIR / "introduction"
OVERVIEW_IMAGE_PREFIX = "overview"
RENDER_SCALE = 2
HEADING_LINE_PATTERN = re.compile(r"^#{1,6}[ \t]+(\S.*?)[ \t]*$")

OVERVIEW_HEADING = "## 全体像"
ORDER_HEADING = "## 推奨学習順序"
DETAILS_HEADING = "### 各トピックの詳細"
AFTER_DETAILS_HEADING = "### 実装済みの共通モジュール"
# GitHub 上の theories/README.md の「各トピックの詳細」の節へのアンカー。
DETAILS_GITHUB_URL = (
    f"{AI_THEORIES_GITHUB_URL}/blob/main/theories/README.md#各トピックの詳細--topic-details"
)

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


def reorient_for_page(source: str) -> str:
    """描画用に、先頭の `flowchart LR` を `flowchart TB` にする(図の中身は変えない)。"""
    return re.sub(r"^(\s*flowchart)[ \t]+(?:LR|RL)\b", r"\1 TB", source, count=1)


def render_mermaid_png(source: str, output_path: Path) -> None:
    """Mermaid のソースを、不透明な白背景の PNG にする(mermaid-cli を npx で呼ぶ)。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_input = output_path.with_suffix(".mmd")
    tmp_input.write_text(source, encoding="utf-8")
    command = [
        "npx", "-y", MERMAID_CLI_PACKAGE,
        "-i", str(tmp_input), "-o", str(output_path),
        "-t", "neutral", "-b", "white", "-s", str(RENDER_SCALE),
    ]
    config = os.environ.get(PUPPETEER_CONFIG_ENV)
    if config:
        command += ["-p", config]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
    finally:
        tmp_input.unlink(missing_ok=True)
    if result.returncode != 0 or not output_path.exists():
        fail(f"Mermaid の図を画像にできませんでした。\n{result.stderr.strip()}")


def replace_mermaid_with_images(text: str, repo: str) -> str:
    """Mermaid のコードブロックを、PNG の画像の記法に置き換える。
    画像は images/introduction/ に、ソースのハッシュを含む名前で書き出す(同じソースなら描画しない)。
    透明な画素を含む PNG は検証エラーにする。
    """
    OVERVIEW_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    lines = text.split("\n")
    count = 0

    def image_markdown(source: str, alt: str) -> str:
        nonlocal count
        count += 1
        rendered = reorient_for_page(source)
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:8]
        name = f"{OVERVIEW_IMAGE_PREFIX}-{count}-{digest}.png"
        path = OVERVIEW_IMAGE_DIR / name
        if not path.exists():
            render_mermaid_png(rendered, path)
        try:
            transparent = png_check.has_transparent_pixels(path.read_bytes())
        except (ValueError, OSError) as error:
            fail(f"画像 {name} を検査できませんでした({error})。")
        if transparent:
            fail(f"画像 {name} に透明な画素があります。")
        used.add(name)
        url = f"https://raw.githubusercontent.com/{repo}/main/images/introduction/{name}"
        return f"![{alt}]({url})"

    out: list[str] = []
    heading = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        match = HEADING_LINE_PATTERN.match(line)
        if match:
            heading = match.group(1)
        if re.match(r"^```mermaid[ \t]*$", line):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            if j >= len(lines):
                fail("閉じられていない Mermaid のコードブロックがあります。")
            source = "\n".join(lines[i + 1 : j]) + "\n"
            out.append(image_markdown(source, heading or "図"))
            i = j + 1
            continue
        out.append(line)
        i += 1

    for stale in OVERVIEW_IMAGE_DIR.glob(f"{OVERVIEW_IMAGE_PREFIX}-*.png"):
        if stale.name not in used:
            stale.unlink()
    return "\n".join(out)


DETAILS_OPEN = "<details>"
DETAILS_CLOSE = "</details>"
SUMMARY_PATTERN = re.compile(r"^<summary>(.+)</summary>$")


def convert_details(text: str) -> str:
    """<details><summary>X</summary> … </details> を Zenn の折りたたみ(:::details X … :::)に変換する。
    入れ子・対応の崩れ・本文や見出しの ::: との衝突は検証エラーにする。コードブロック内は対象外。
    """
    out: list[str] = []
    in_details = False
    in_code = False
    expect_summary = False
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if nb_to_zenn.CODE_FENCE_PATTERN.match(line):
            in_code = not in_code
        if not in_code and stripped == DETAILS_OPEN:
            if in_details:
                fail(f"<details> が入れ子になっています(「各トピックの詳細」の {number} 行目)。")
            in_details = True
            expect_summary = True
            continue
        if expect_summary:
            match = SUMMARY_PATTERN.match(stripped)
            if not match:
                fail(f"<details> の次の行が <summary>…</summary> ではありません({number} 行目)。")
            summary = match.group(1).strip()
            if ":::" in summary:
                fail(f"<summary> に ::: が含まれています({number} 行目)。")
            out.append(f":::details {summary}")
            expect_summary = False
            continue
        if not in_code and stripped == DETAILS_CLOSE:
            if not in_details:
                fail(f"対応する <details> のない </details> があります({number} 行目)。")
            in_details = False
            out.append(":::")
            continue
        if in_details and ":::" in line:
            fail(f"<details> の中に ::: があり、Zenn の折りたたみと衝突します({number} 行目)。")
        if not in_code and not in_details and (DETAILS_OPEN in line or DETAILS_CLOSE in line):
            fail(f"<details> の書き方が想定外です({number} 行目)。")
        out.append(line)
    if in_details or expect_summary:
        fail("閉じられていない <details> があります。")
    return "\n".join(out)


def convert_table(table_text: str, manifest: dict) -> str:
    """学習順序の節のノートブックへのリンクを Zenn の章 URL に解決する(章が未変換なら GitHub の URL)。
    章になっていない行は、ノートブックの列を「🚧 準備中」にする。
    """
    lines = []
    for line in table_text.splitlines():
        if line.startswith("|") and not is_separator_row(line) and "未作成" in line.split("|")[-2]:
            cells = line.rstrip().rstrip("|").split("|")
            cells[-1] = " 🚧 準備中 "
            line = "|".join(cells) + "|"
        lines.append(line)
    body, counts = nb_to_zenn.rewrite_notebook_links("\n".join(lines), manifest)
    print(
        f"学習順序表のリンク: Zennの章のURL {counts['zenn']}件、"
        f"GitHubのURL {counts['github']}件、変更なし {counts['unchanged']}件。"
    )
    return body


INTRO_HEAD = f"""---
title: "はじめに"
---

## この本について

本書は [ai-theories]({AI_THEORIES_GITHUB_URL}) で学習している LLM(大規模言語モデル)・VLM(視覚言語モデル)の理論学習ロードマップです。古典的な機械学習理論(SVM、決定木など)は対象外とし、 **LLM / VLM に関する理論に限定** しています。

- 理論的に本質的な部分は、可能な限り **PyTorch によるスクラッチ実装** で確認します(Attention 計算、正規化、LoRA の低ランク分解など)。
- 学習・実験は **Google Colab 無料枠(T4 GPU など)で完結する規模** のモデル・データセットで構成しています。

## 本書の構成

以降の章は、[ai-theories]({AI_THEORIES_GITHUB_URL}) の各学習トピックのノートブックをそのまま章として収録したものです。この章には、トピックの全体像を示す図、推奨学習順序の表、各トピックの詳細を、 `theories/README.md` から転載しています。表のノートブックの列のリンクをクリックすると、該当の章に移動します。 **まだ章になっていないトピックは「🚧 準備中」と表示** します。
"""

INTRO_TAIL = f"""## コードのライセンス

本書内で参照するコード・ノートブックのライセンスは、[ai-theories]({AI_THEORIES_GITHUB_URL}) 本体の LICENSE に準拠します。

## リンク

- [ai-theories(GitHub リポジトリ)]({AI_THEORIES_GITHUB_URL})
"""

DETAILS_LINK_TEXT = (
    "各トピックの詳細(理論の範囲・実験の設計・本番の結果)は、分量が多く Zenn の章の文字数の上限に"
    f"収まらないため、GitHub の [theories/README.md の「各トピックの詳細」の節]({DETAILS_GITHUB_URL}) に"
    "掲載しています。"
)


def build_introduction_body(readme_text: str, manifest: dict, include_details: bool) -> str:
    overview = extract_section(readme_text, OVERVIEW_HEADING, ORDER_HEADING)
    order = extract_section(readme_text, ORDER_HEADING, DETAILS_HEADING)
    details = extract_section(readme_text, DETAILS_HEADING, AFTER_DETAILS_HEADING)

    validate_mermaid_blocks(overview)
    # Zenn では画像として表示するため、GitHub 上の描画の説明は除く。
    overview = overview.replace(" Mermaid の図は GitHub 上で描画される。", "")
    overview = replace_mermaid_with_images(overview, os.environ.get("GITHUB_REPOSITORY", DEFAULT_PUBLISH_REPO))
    order = convert_table(order, manifest)
    if include_details:
        details_part = convert_details(details)
    else:
        details_part = f"{DETAILS_HEADING} / Topic Details\n\n{DETAILS_LINK_TEXT}"

    parts = [INTRO_HEAD, overview, order, details_part, INTRO_TAIL]
    return "\n\n".join(part.strip("\n") for part in parts) + "\n"


def generate_introduction(readme_text: str, manifest: dict) -> tuple[str, bool]:
    """「はじめに」章の全文と、「各トピックの詳細」を載せたかを返す。"""
    full = build_introduction_body(readme_text, manifest, include_details=True)
    content_length = len(full.split("---\n", 2)[2])
    if content_length <= CHAR_THRESHOLD:
        body, included = full, True
    else:
        print(
            f"「はじめに」の本文が {content_length} 文字で閾値({CHAR_THRESHOLD} 文字)を超えるため、"
            "「各トピックの詳細」は載せず、GitHub へのリンクにします。"
        )
        body, included = build_introduction_body(readme_text, manifest, include_details=False), False
    length = len(body.split("---\n", 2)[2])
    if length > CHAR_THRESHOLD:
        fail(f"「はじめに」の本文が詳細を除いても {length} 文字で、閾値({CHAR_THRESHOLD} 文字)を超えます。")
    return body, included


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
    introduction, included = generate_introduction(readme_text, manifest)
    INTRODUCTION_PATH.write_text(introduction, encoding="utf-8")
    print(
        f"「はじめに」の本文: {len(introduction.split('---' + chr(10), 2)[2])} 文字"
        f"(各トピックの詳細: {'載せた' if included else '載せない(GitHub へのリンク)'})"
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
