"""セルのタグによる表示の制御(nb_to_zenn.py)の試験。

実行: uv run --with nbconvert python3 -m unittest scripts/test_nb_to_zenn_tags.py
"""

from __future__ import annotations

import base64
import sys
import tempfile
import unittest
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb_to_zenn  # noqa: E402

PNG_1PIXEL = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c6360000002000001e221bc330000000049454e44ae426082"
    )
).decode()


def code(source: str, *tags: str, output: str | None = None, image: bool = False):
    cell = new_code_cell(source)
    outputs = []
    if output:
        outputs.append(new_output("stream", name="stdout", text=output))
    if image:
        outputs.append(
            new_output("display_data", data={"image/png": PNG_1PIXEL, "text/plain": "<Figure>"})
        )
    cell.outputs = outputs
    if tags:
        cell.metadata["tags"] = list(tags)
    return cell


def markdown(source: str, *tags: str):
    cell = new_markdown_cell(source)
    if tags:
        cell.metadata["tags"] = list(tags)
    return cell


def convert(cells: list) -> tuple[str, list[str]]:
    nb = new_notebook(cells=[markdown("# 999. 試験")] + cells)
    with tempfile.TemporaryDirectory() as tmp:
        nb_to_zenn.IMAGES_DIR = Path(tmp)
        body = nb_to_zenn.convert_to_markdown(nb, "test_slug_for_tags", "o/r")
        images = sorted(p.name for p in (Path(tmp) / "test_slug_for_tags").iterdir())
    return body, images


class TagTest(unittest.TestCase):
    def test_no_tags_is_unchanged(self):
        cells = [markdown("## 見出し\n\n$$x^2$$"), code("print(1)", output="1\n", image=True)]
        nb = new_notebook(cells=[markdown("# 999. 試験")] + cells)
        expected, _ = nb_to_zenn.MarkdownExporter().from_notebook_node(nb)
        body, images = convert(cells)
        self.assertNotIn(":::", body)
        self.assertEqual(len(images), 1)
        self.assertEqual(
            body.replace("https://raw.githubusercontent.com/o/r/main/images/test_slug_for_tags/", ""),
            expected,
        )

    def test_hide_removes_cell_and_its_images(self):
        body, images = convert(
            [code("secret_code()", "zenn-hide", output="secret_output\n", image=True), code("print(2)")]
        )
        self.assertNotIn("secret_code", body)
        self.assertNotIn("secret_output", body)
        self.assertIn("print(2)", body)
        self.assertEqual(images, [])

    def test_hide_input_keeps_output_only(self):
        body, images = convert([code("hidden_source()", "zenn-hide-input", output="shown\n", image=True)])
        self.assertNotIn("hidden_source", body)
        self.assertIn("shown", body)
        self.assertEqual(len(images), 1)

    def test_details_code_cell(self):
        body, _ = convert([code("print(3)", "zenn-details", output="3\n", image=True)])
        self.assertIn(":::details コード\n", body)
        self.assertLess(body.index(":::details コード"), body.index("print(3)"))
        self.assertLess(body.index("![png]"), body.rindex(":::\n"))
        self.assertEqual(body.count(":::"), 2)

    def test_details_markdown_cell_uses_first_heading(self):
        body, _ = convert(
            [markdown("### 導出の詳細\n\n数式 $$\\frac{a}{b}$$ と表\n\n| a | b |\n|---|---|\n| 1 | 2 |", "zenn-details")]
        )
        self.assertIn(":::details 導出の詳細\n", body)
        self.assertNotIn("### 導出の詳細", body)
        self.assertIn("$$\\frac{a}{b}$$", body)
        self.assertIn("| a | b |", body)

    def test_details_markdown_cell_without_heading(self):
        body, _ = convert([markdown("本文のみ", "zenn-details")])
        self.assertIn(":::details 詳細\n", body)

    def test_appendix_runs_to_end(self):
        body, _ = convert(
            [markdown("## 本編"), markdown("## 付録 A", "zenn-appendix"), code("print(4)", image=True)]
        )
        self.assertIn(":::details 付録\n", body)
        self.assertIn("## 付録 A", body)
        self.assertTrue(body.rstrip().endswith(":::"))
        self.assertLess(body.index("## 本編"), body.index(":::details 付録"))

    def test_nested_details_in_appendix(self):
        body, _ = convert(
            [
                markdown("## 本編"),
                markdown("## 付録", "zenn-appendix"),
                markdown("### 内側の詳細\n\n中身", "zenn-details"),
                code("print(5)", "zenn-details", output="5\n"),
                markdown("## 付録の続き"),
            ]
        )
        lines = body.splitlines()
        self.assertIn("::::details 付録", lines)
        self.assertIn(":::details 内側の詳細", lines)
        self.assertIn(":::details コード", lines)
        self.assertEqual(lines.count(":::"), 2)
        self.assertEqual(lines.count("::::"), 1)

    def test_hide_inside_appendix_and_hidden_appendix_start(self):
        body, images = convert(
            [markdown("## 付録", "zenn-appendix"), code("x", "zenn-hide", image=True), markdown("残り")]
        )
        self.assertIn(":::details 付録", body)
        self.assertEqual(images, [])

    def test_split_does_not_cut_inside_fold(self):
        body, _ = convert(
            [
                markdown("## 一"),
                markdown("## 付録", "zenn-appendix"),
                markdown("## 付録の見出し"),
                markdown("### 付録の小見出し"),
            ]
        )
        chunks = nb_to_zenn.split_at_headings(body, nb_to_zenn.TOP_HEADING_PATTERN)
        for chunk in chunks:
            self.assertEqual(chunk.count(":::details"), chunk.count(":::\n") if ":::details" in chunk else 0)
        self.assertEqual(sum(":::details 付録" in c for c in chunks), 1)
        self.assertEqual(len([c for c in chunks if "## 付録の見出し" in c and ":::details" in c]), 1)

    def assert_rejected(self, cells):
        with self.assertRaises(SystemExit) as raised:
            convert(cells)
        self.assertEqual(raised.exception.code, 1)

    def test_rejects_second_appendix(self):
        self.assert_rejected(
            [markdown("## 付録", "zenn-appendix"), markdown("## 付録 2", "zenn-appendix")]
        )

    def test_rejects_appendix_without_heading_or_on_code_cell(self):
        self.assert_rejected([markdown("見出しなし", "zenn-appendix")])
        self.assert_rejected([code("# 見出しのように見える", "zenn-appendix")])

    def test_rejects_hide_input_on_markdown_and_unknown_zenn_tag(self):
        self.assert_rejected([markdown("本文", "zenn-hide-input")])
        self.assert_rejected([markdown("本文", "zenn-unknown")])


if __name__ == "__main__":
    unittest.main()
