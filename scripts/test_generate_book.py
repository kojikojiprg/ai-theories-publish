"""「はじめに」章の生成(generate_book.py)と PNG の透明画素の検査(png_check.py)の試験。

実行: uv run --with nbconvert python3 -m unittest discover -s scripts -p 'test_*.py'
"""

from __future__ import annotations

import struct
import sys
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


class DetailsTest(unittest.TestCase):
    def test_converts(self):
        text = "前\n\n<details>\n<summary>001 注意機構</summary>\n\n本文 $x$\n\n</details>\n\n<details>\n<summary>B</summary>\n\n本文 2\n\n</details>"
        self.assertEqual(
            generate_book.convert_details(text),
            "前\n\n:::details 001 注意機構\n\n本文 $x$\n\n:::\n\n:::details B\n\n本文 2\n\n:::",
        )

    def test_colons_in_body_is_error(self):
        with self.assertRaises(SystemExit):
            generate_book.convert_details("<details>\n<summary>A</summary>\n\n:::message\nx\n:::\n\n</details>")

    def test_colons_in_summary_is_error(self):
        with self.assertRaises(SystemExit):
            generate_book.convert_details("<details>\n<summary>A ::: B</summary>\n\nx\n</details>")

    def test_nested_is_error(self):
        with self.assertRaises(SystemExit):
            generate_book.convert_details("<details>\n<summary>A</summary>\n<details>\n<summary>B</summary>\n</details>\n</details>")

    def test_unclosed_is_error(self):
        with self.assertRaises(SystemExit):
            generate_book.convert_details("<details>\n<summary>A</summary>\n\nx\n")

    def test_stray_close_is_error(self):
        with self.assertRaises(SystemExit):
            generate_book.convert_details("x\n</details>\n")

    def test_missing_summary_is_error(self):
        with self.assertRaises(SystemExit):
            generate_book.convert_details("<details>\n本文\n</details>")

    def test_code_block_is_left_alone(self):
        text = "```html\n<details>\n:::\n</details>\n```"
        self.assertEqual(generate_book.convert_details(text), text)


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

    def test_block_is_passed_through_with_theme_only(self):
        readme = (
            "## 全体像\n\n```mermaid\nflowchart LR\n  A --> B\n```\n\n## 推奨学習順序\n\n| # | a | b | c | d | e |\n"
            "| --- | --- | --- | --- | --- | --- |\n| 001 | x | y | z | w | [001_a.ipynb](./01/001_a.ipynb) |\n\n### 各トピックの詳細\n\n"
            "<details>\n<summary>S</summary>\n\nx\n\n</details>\n\n### 実装済みの共通モジュール\n"
        )
        original = nb_to_zenn.find_notebook_by_number
        nb_to_zenn.find_notebook_by_number = lambda n: None
        try:
            body = generate_book.build_introduction_body(readme, {}, include_details=True)
        finally:
            nb_to_zenn.find_notebook_by_number = original
        self.assertIn("```mermaid\n%%{init: {'theme': 'neutral'}}%%\nflowchart LR\n  A --> B\n```", body)
        self.assertIn(":::details S", body)


class IntroductionTest(unittest.TestCase):
    def readme(self, details_chars: int) -> str:
        return (
            "## 全体像\n\n図\n\n## 推奨学習順序\n\n| # | a | b | c | d | e |\n| --- | --- | --- | --- | --- | --- |\n"
            "| 001 | x | y | z | w | v |\n\n### 各トピックの詳細\n\n<details>\n<summary>S</summary>\n\n"
            + "あ" * details_chars
            + "\n\n</details>\n\n### 実装済みの共通モジュール\n"
        )

    def test_small_includes_details(self):
        body, included = generate_book.generate_introduction(self.readme(100), {})
        self.assertTrue(included)
        self.assertIn(":::details S", body)

    def test_large_replaces_details_with_link(self):
        body, included = generate_book.generate_introduction(self.readme(46_000), {})
        self.assertFalse(included)
        self.assertNotIn(":::details", body)
        self.assertIn(generate_book.DETAILS_GITHUB_URL, body)
        self.assertLessEqual(len(body), generate_book.CHAR_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
