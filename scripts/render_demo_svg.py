"""Render results/demo_output.txt as docs/demo.svg (a picture of the real `make demo` output)."""

from html import escape
from pathlib import Path

lines = Path("results/demo_output.txt").read_text().splitlines()
width, line_h, pad = 1080, 20, 20
height = pad * 2 + line_h * (len(lines) + 2)
rows = [f'<text x="{pad}" y="{pad + line_h}" fill="#7ee787">$ make demo</text>']
for i, line in enumerate(lines):
    y = pad + line_h * (i + 2)
    rows.append(f'<text x="{pad}" y="{y}" xml:space="preserve">{escape(line)}</text>')
svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
    f'viewBox="0 0 {width} {height}" font-family="Menlo, Consolas, monospace" font-size="13">'
    f'<rect width="100%" height="100%" rx="8" fill="#0d1117"/>'
    f'<g fill="#e6edf3">{"".join(rows)}</g></svg>'
)
Path("docs/demo.svg").write_text(svg)
print("wrote docs/demo.svg")
