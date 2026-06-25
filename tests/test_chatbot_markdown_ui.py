from pathlib import Path
import json
import shutil
import subprocess
import unittest


class ChatbotMarkdownUiTests(unittest.TestCase):
    def test_chat_ui_uses_safe_markdown_renderer(self):
        source = Path("static/js/app.js").read_text(encoding="utf-8")

        self.assertIn("function renderMarkdownSafe", source)
        self.assertIn("function renderInlineMarkdown", source)
        self.assertIn("escapeHtml(text)", source)
        self.assertIn("renderMarkdownSafe(accumulated)", source)
        self.assertNotIn('accumulated.replace(/\\n/g, "<br>")', source)

    def test_chat_markdown_styles_exist(self):
        source = Path("static/css/style.css").read_text(encoding="utf-8")

        self.assertIn(".md-content", source)
        self.assertIn(".md-content ul", source)
        self.assertIn(".md-content code", source)

    def test_renderer_handles_compact_numbered_bold_items(self):
        if not shutil.which("node"):
            self.skipTest("node is not available")

        sample = (
            "Các thay đổi chính bao gồm:\n\n"
            "1.**Điều 6**: Tiến độ dự kiến tăng từ 180 ngày lên 210 ngày.\n"
            "2.**Điều 11**: Giá trị hợp đồng tăng.\n\n"
            "**Điểm cần chú ý**"
        )
        script = f"""
const fs = require('fs');
const vm = require('vm');
const src = fs.readFileSync('static/js/app.js', 'utf8').split('\\nclass App')[0];
const ctx = {{}};
vm.createContext(ctx);
vm.runInContext(src, ctx);
const html = ctx.renderMarkdownSafe({json.dumps(sample, ensure_ascii=False)});
if (html.includes('**')) {{
  console.error(html);
  process.exit(1);
}}
if (!html.includes('<ol>') || !html.includes('<strong>Điều 6</strong>')) {{
  console.error(html);
  process.exit(2);
}}
"""
        completed = subprocess.run(
            ["node", "-e", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
