"""「はじめに」章の生成(generate_book.py)と PNG の透明画素の検査(png_check.py)の試験。

実行: uv run --with nbconvert python3 -m unittest discover -s scripts -p 'test_*.py'
"""

from __future__ import annotations

import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_book  # noqa: E402
import nb_to_zenn  # noqa: E402
import png_check  # noqa: E402


def make_png(color_type: int, pixels: bytes, width: int, height: int, extra: bytes = b"") -> bytes:
    """8 ビット・フィルタなしの PNG を作る。pixels は 1 行ぶんずつ連結した画素のバイト列。"""
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    row = width * channels
    raw = b"".join(b"\x00" + pixels[i * row : (i + 1) * row] for i in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (
        png_check.PNG_SIGNATURE
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
        + extra
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


class PngTest(unittest.TestCase):
    def test_opaque_rgba(self):
        self.assertFalse(png_check.has_transparent_pixels(make_png(6, bytes([1, 2, 3, 255] * 4), 2, 2)))

    def test_transparent_rgba(self):
        pixels = bytes([1, 2, 3, 255, 1, 2, 3, 0, 1, 2, 3, 255, 1, 2, 3, 255])
        self.assertTrue(png_check.has_transparent_pixels(make_png(6, pixels, 2, 2)))

    def test_rgb_has_no_alpha(self):
        self.assertFalse(png_check.has_transparent_pixels(make_png(2, bytes([9] * 12), 2, 2)))

    def test_gray_alpha(self):
        self.assertTrue(png_check.has_transparent_pixels(make_png(4, bytes([5, 255, 5, 128]), 2, 1)))

    def test_palette_with_transparent_entry_in_use(self):
        trns = struct.pack(">I", 2) + b"tRNS" + bytes([255, 0]) + struct.pack(">I", zlib.crc32(b"tRNS" + bytes([255, 0])))
        self.assertTrue(png_check.has_transparent_pixels(make_png(3, bytes([0, 1]), 2, 1, trns)))
        self.assertFalse(png_check.has_transparent_pixels(make_png(3, bytes([0, 0]), 2, 1, trns)))

    def test_conversion_rejects_transparent_png(self):
        data = make_png(6, bytes([1, 2, 3, 0]), 1, 1)
        with self.assertRaises(SystemExit):
            nb_to_zenn.check_png_is_opaque("a.png", data, "slug")
        nb_to_zenn.check_png_is_opaque("a.png", make_png(6, bytes([1, 2, 3, 255]), 1, 1), "slug")
        nb_to_zenn.check_png_is_opaque("a.svg", b"<svg/>", "slug")


class TableTest(unittest.TestCase):
    TABLE = (
        "| # | トピック | 何を学ぶか | ノートブック |\n| --- | --- | --- | --- |\n"
        "| 001 | A | a | [001_x.ipynb](./01/001_x.ipynb) |\n| 002 | B | b | 未作成 / TBD |\n\n> 注"
    )

    def test_verdict_column_is_dropped(self):
        table = (
            "| # | トピック | 判定 / Verdicts | ノートブック |\n| --- | --- | --- | --- |\n"
            "| 002 | B | A 支持 | 未作成 / TBD |"
        )
        lines = generate_book.convert_table(table, {}).splitlines()
        self.assertEqual(lines, ["| # | トピック |", "| --- | --- |", "| 002 🚧 準備中 | B |"])

    def test_link_moves_to_number_column_and_notebook_column_is_dropped(self):
        original = nb_to_zenn.find_notebook_by_number
        nb_to_zenn.find_notebook_by_number = lambda n: Path("/x/01_foundations/001_x.ipynb")
        try:
            result = generate_book.convert_table(self.TABLE, {"001": {"slugs": ["001_x"], "split": False}})
        finally:
            nb_to_zenn.find_notebook_by_number = original
        lines = result.splitlines()
        self.assertEqual(lines[0], "| # | トピック | 何を学ぶか |")
        self.assertEqual(lines[1], "| --- | --- | --- |")
        self.assertEqual(
            lines[2],
            "| [001](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/001_x) | A | a |",
        )
        self.assertEqual(lines[3], "| 002 🚧 準備中 | B | b |")
        self.assertEqual(lines[-1], "> 注")


class MermaidTest(unittest.TestCase):
    def block(self, body: str) -> str:
        return f"```mermaid\n{body}\n```\n"

    def test_within_limits(self):
        self.assertEqual(generate_book.validate_mermaid_blocks(self.block("flowchart LR\n  A --> B & C")), 1)

    def test_too_long_is_error(self):
        with self.assertRaises(SystemExit):
            generate_book.validate_mermaid_blocks(self.block("flowchart LR\n" + "A --> B\n" * 400))

    def test_exactly_limit_is_ok(self):
        body = "x" * (generate_book.MERMAID_MAX_CHARS - 1) + "\n"
        generate_book.validate_mermaid_blocks("```mermaid\n" + body + "```\n")

    def test_too_many_chains_is_error(self):
        with self.assertRaises(SystemExit):
            generate_book.validate_mermaid_blocks(self.block("flowchart LR\n  A --> " + " & ".join("B" * 1 for _ in range(12))))

    def test_blocks_are_made_vertical(self):
        source = "```mermaid\nflowchart LR\n  A --> B\n```"
        readme = f"## 全体像\n\n{source}\n\n## 推奨学習順序\n\n| # | a |\n| --- | --- |\n| 001 | x |\n\n### 各トピックの詳細\n"
        body = generate_book.build_introduction_body(readme, {})
        self.assertIn(source.replace("flowchart LR", "flowchart TB"), body)
        self.assertNotIn("flowchart LR", body)
        self.assertNotIn("%%{init", body)
        self.assertNotIn("![", body)

    def test_too_long_block_in_overview_is_error(self):
        readme = "## 全体像\n\n```mermaid\nflowchart LR\n" + "A --> B\n" * 400 + "```\n\n## 推奨学習順序\n\n| # | a |\n| --- | --- |\n\n### 各トピックの詳細\n"
        with self.assertRaises(SystemExit):
            generate_book.build_introduction_body(readme, {})


class IntroductionTest(unittest.TestCase):
    def readme(self, details_chars: int) -> str:
        return (
            "## 全体像\n\n図\n\n## 推奨学習順序\n\n| # | a | 判定 | c |\n| --- | --- | --- | --- |\n"
            "| 001 | x | y | v |\n\n### 各トピックの詳細\n\n<details>\n<summary>S</summary>\n\n"
            + "あ" * details_chars
            + "\n\n</details>\n\n### 実装済みの共通モジュール\n"
        )

    def test_details_section_is_not_included(self):
        body = generate_book.generate_introduction(self.readme(100_000), {})
        self.assertNotIn("各トピックの詳細", body)
        self.assertNotIn("あああ", body)
        self.assertLessEqual(len(body), generate_book.CHAR_THRESHOLD)

    def test_too_long_is_error(self):
        readme = self.readme(0).replace("図", "あ" * 46_000, 1)
        with self.assertRaises(SystemExit):
            generate_book.generate_introduction(readme, {})


if __name__ == "__main__":
    unittest.main()
