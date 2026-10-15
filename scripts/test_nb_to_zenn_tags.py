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
        body = nb_to_zenn.convert_to_markdown(
            nb,
            "test_slug_for_tags",
            "o/r",
            Path("theories/01_test/999_test_notebook.ipynb"),
        )
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

    def test_appendix_is_replaced_by_link_paragraph(self):
        body, images = convert(
            [
                markdown("## 本編"),
                code("print(1)", output="1\n"),
                markdown("## 付録", "zenn-appendix"),
                markdown("付録の本文"),
                code("appendix_code()", output="appendix_output\n", image=True),
                markdown("### 内側の詳細", "zenn-details"),
            ]
        )
        link = "https://github.com/kojikojiprg/ai-theories/blob/main/theories/01_test/999_test_notebook.ipynb"
        self.assertIn(
            "付録(実行の手順・パイロット・設計の改訂の記録など)は、GitHub のノートブックを参照。\n\n" + link,
            body,
        )
        self.assertNotIn("付録の本文", body)
        self.assertNotIn("appendix_code", body)
        self.assertNotIn(":::", body)
        self.assertEqual(images, [])
        self.assertIn("print(1)", body)

    def test_hidden_cell_before_appendix_stays_hidden(self):
        body, images = convert(
            [code("x", "zenn-hide", image=True), markdown("## 付録", "zenn-appendix")]
        )
        self.assertEqual(images, [])
        self.assertIn("GitHub のノートブックを参照", body)

    def test_images_only_keeps_images_not_text(self):
        body, images = convert(
            [code("plot_it()", "zenn-images-only", output="printed_text\n", image=True)]
        )
        self.assertNotIn("plot_it", body)
        self.assertNotIn("printed_text", body)
        self.assertNotIn("Figure", body)
        self.assertIn("![png]", body)
        self.assertEqual(len(images), 1)

    def test_images_only_drops_cell_without_images(self):
        body, images = convert(
            [code("print_only()", "zenn-images-only", output="printed_text\n"), markdown("残る")]
        )
        self.assertNotIn("print_only", body)
        self.assertNotIn("printed_text", body)
        self.assertIn("残る", body)
        self.assertEqual(images, [])

    def test_images_only_in_details(self):
        body, images = convert([code("plot_it()", "zenn-images-only", "zenn-details", image=True)])
        self.assertIn(":::details コード", body)
        self.assertNotIn("plot_it", body)
        self.assertEqual(len(images), 1)

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

    def test_rejects_images_only_on_markdown_and_raw(self):
        self.assert_rejected([markdown("本文", "zenn-images-only")])
        raw = nbformat.v4.new_raw_cell("raw")
        raw.metadata["tags"] = ["zenn-images-only"]
        self.assert_rejected([raw])


class LinkTest(unittest.TestCase):
    """ノートブック間リンクの書き換え(前方への参照・後方への参照・アンカー付きの参照)。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        theories = Path(self.tmp.name)
        for category, name in [("01_a", "001_first_topic"), ("02_b", "002_second_topic")]:
            (theories / category).mkdir()
            (theories / category / f"{name}.ipynb").write_text("{}", encoding="utf-8")
        self.original = nb_to_zenn.AI_THEORIES_THEORIES_DIR
        nb_to_zenn.AI_THEORIES_THEORIES_DIR = theories
        self.manifest = {
            "001": {"slugs": ["001_first_topic"], "split": False, "titles": ["a"]},
            "002": {
                "slugs": ["002_second_topic-theory", "002_second_topic-practice"],
                "split": True,
                "titles": ["b", "c"],
            },
        }

    def tearDown(self):
        nb_to_zenn.AI_THEORIES_THEORIES_DIR = self.original
        self.tmp.cleanup()

    def rewrite(self, body, manifest=None):
        return nb_to_zenn.rewrite_notebook_links(body, self.manifest if manifest is None else manifest)

    def test_backward_reference_resolves_to_chapter_url(self):
        body, counts = self.rewrite("[001](./001_first_topic.ipynb)")
        self.assertEqual(body, f"[001]({nb_to_zenn.build_zenn_book_chapter_url('001_first_topic')})")
        self.assertEqual(counts, {"zenn": 1, "github": 0, "unchanged": 0})

    def test_forward_reference_resolves_to_chapter_url_when_manifest_is_complete(self):
        body, counts = self.rewrite("[002](../02_b/002_second_topic.ipynb)")
        self.assertEqual(
            body, f"[002]({nb_to_zenn.build_zenn_book_chapter_url('002_second_topic-theory')})"
        )
        self.assertEqual(counts["zenn"], 1)

    def test_forward_reference_falls_back_to_github_when_not_in_manifest(self):
        manifest = {k: v for k, v in self.manifest.items() if k != "002"}
        body, counts = self.rewrite("[002](../02_b/002_second_topic.ipynb)", manifest)
        self.assertEqual(
            body,
            "[002](https://github.com/kojikojiprg/ai-theories/blob/main/theories/02_b/002_second_topic.ipynb)",
        )
        self.assertEqual(counts, {"zenn": 0, "github": 1, "unchanged": 0})

    def test_anchor_reference_stays_on_github_with_anchor(self):
        body, counts = self.rewrite("[節](./001_first_topic.ipynb#3-理論)")
        self.assertEqual(
            body,
            "[節](https://github.com/kojikojiprg/ai-theories/blob/main/theories/01_a/001_first_topic.ipynb#3-理論)",
        )
        self.assertEqual(counts, {"zenn": 0, "github": 1, "unchanged": 0})

    def test_missing_notebook_is_unchanged(self):
        body, counts = self.rewrite("[9](./009_missing_topic.ipynb)")
        self.assertEqual(body, "[9](./009_missing_topic.ipynb)")
        self.assertEqual(counts["unchanged"], 1)

    def test_structure_change_is_an_error(self):
        nb_to_zenn.check_structure_unchanged("001", self.manifest, ["001_first_topic"])
        with self.assertRaises(SystemExit):
            nb_to_zenn.check_structure_unchanged("001", self.manifest, ["001_first_topic-theory"])


class TitleTest(unittest.TestCase):
    def test_title_is_japanese_part_only(self):
        self.assertEqual(nb_to_zenn.fit_title("短い / Short"), "短い")
        self.assertEqual(nb_to_zenn.build_part_title("短い / Short", "theory", 0, 1), "短い(理論編)")
        self.assertEqual(
            nb_to_zenn.build_part_title("短い / Short", "practice", 1, 3), "短い(実装・実験編 2/3)"
        )

    def test_title_without_separator_is_unchanged(self):
        self.assertEqual(nb_to_zenn.fit_title("区切りなし"), "区切りなし")

    def test_too_long_title_is_an_error(self):
        with self.assertRaises(SystemExit):
            nb_to_zenn.fit_title("あ" * 71)


class MermaidThemeTest(unittest.TestCase):
    def test_directive_is_added_to_each_block(self):
        body = "前\n\n```mermaid\nflowchart TB\n  A --> B\n```\n\n```mermaid\ngraph LR\n  C --> D\n```\n"
        result = nb_to_zenn.add_mermaid_theme(body)
        self.assertEqual(result.count(nb_to_zenn.MERMAID_THEME_DIRECTIVE), 2)
        self.assertIn("```mermaid\n%%{init: {'theme': 'neutral'}}%%\nflowchart TB\n", result)

    def test_existing_init_is_kept(self):
        body = "```mermaid\n%%{init: {'theme': 'dark'}}%%\nflowchart TB\n  A --> B\n```\n"
        self.assertEqual(nb_to_zenn.add_mermaid_theme(body), body)

    def test_other_code_blocks_are_unchanged(self):
        body = "```python\nprint(1)\n```\n"
        self.assertEqual(nb_to_zenn.add_mermaid_theme(body), body)

    def test_conversion_adds_directive(self):
        nb = new_notebook(cells=[new_markdown_cell("```mermaid\nflowchart TB\n  A --> B\n```")])
        with tempfile.TemporaryDirectory() as tmp:
            nb_to_zenn.IMAGES_DIR = Path(tmp)
            body = nb_to_zenn.convert_to_markdown(nb, "slug", "owner/repo")
        self.assertIn("%%{init: {'theme': 'neutral'}}%%\nflowchart TB", body)


if __name__ == "__main__":
    unittest.main()
