import markdown
import sys

with open('classical_ml_interview_prep_guide.md', 'r', encoding='utf-8') as f:
    text = f.read()

html = markdown.markdown(text, extensions=['extra'])

css = """
<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 10px;
    line-height: 1.3;
    color: #24292e;
    margin: 0 auto;
}
.markdown-body {
    column-count: 2;
    column-gap: 20px;
    padding: 10px;
}
@page {
    margin: 10mm;
    size: A4;
}
h1 {
    font-size: 18px;
    text-align: center;
    column-span: all;
    border-bottom: 2px solid #eaecef;
    padding-bottom: 6px;
    margin-top: 0;
    margin-bottom: 12px;
}
h2 {
    font-size: 13px;
    color: #0366d6;
    margin-top: 12px;
    margin-bottom: 6px;
    border-bottom: 1px solid #eaecef;
    padding-bottom: 3px;
    break-after: avoid;
    page-break-after: avoid;
}
h3 {
    font-size: 11px;
    font-weight: bold;
    margin-top: 8px;
    margin-bottom: 3px;
    break-after: avoid;
    page-break-after: avoid;
    color: #111;
}
h3 + p, h3 + ul {
    break-before: avoid;
    page-break-before: avoid;
}
p, ul {
    margin-top: 0;
    margin-bottom: 6px;
}
li {
    margin-bottom: 2px;
}
a {
    color: #0366d6;
    text-decoration: none;
    font-weight: 500;
}
hr {
    display: none;
}
strong {
    font-weight: 600;
}
</style>
"""

full_html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>ML Interview Guide</title>{css}</head><body><div class='markdown-body'>{html}</div></body></html>"

with open('classical_ml_interview_prep_guide_printable.html', 'w', encoding='utf-8') as f:
    f.write(full_html)

print("HTML generated.")
