import re
import os
import subprocess
from pathlib import Path

md_path = Path("DOCUMENTO_DISENO_AJUSTADO.md")
html_path = Path("DOCUMENTO_DISENO_AJUSTADO.html")
pdf_path = Path("DOCUMENTO_DISENO_AJUSTADO.pdf")

content = md_path.read_text(encoding="utf-8")

# Custom converter for Markdown to high-quality HTML
lines = content.splitlines()
html_lines = []
in_table = False
table_header = True
in_code = False
code_lang = ""
code_buffer = []

def format_inline(text):
    # Bold italic
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'<strong><em>\1</em></strong>', text)
    # Bold
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    # Italic
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    # Math inline: $...$
    def replace_math(m):
        math_str = m.group(1)
        # convert basic math symbols to nice unicode/html
        math_str = math_str.replace(r'\sum', '&sum;')
        math_str = math_str.replace(r'\times', '&times;')
        math_str = math_str.replace(r'\mathbf', '<b>').replace(r'\text', '<span>')
        math_str = math_str.replace(r'\le', '&le;').replace(r'\ge', '&ge;')
        math_str = math_str.replace(r'\Delta', '&Delta;').replace(r'\mu', '&mu;').replace(r'\sigma', '&sigma;')
        math_str = math_str.replace(r'\hat{v}', 'v&#770;')
        math_str = math_str.replace(r'\in', '&isin;').replace(r'\mathbb{R}', '&#8477;')
        math_str = math_str.replace(r'\to', '&rarr;').replace(r'\longrightarrow', '&rarr;')
        return f'<span class="math-inline">{math_str}</span>'
    text = re.sub(r'\$(.*?)\$', replace_math, text)
    # Code inline
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    # Links
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)
    return text

i = 0
while i < len(lines):
    line = lines[i]
    stripped = line.strip()

    # Code block
    if stripped.startswith("```"):
        if in_code:
            in_code = False
            code_text = "\n".join(code_buffer)
            # escape html
            code_text = code_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if code_lang == "mermaid":
                html_lines.append(f'<div class="mermaid-block"><div class="code-title">Diagrama de Secuencia UML (Mermaid)</div><pre class="mermaid"><code>{code_text}</code></pre></div>')
            else:
                html_lines.append(f'<pre class="code-block"><code>{code_text}</code></pre>')
            code_buffer = []
            code_lang = ""
        else:
            in_code = True
            code_lang = stripped[3:].strip()
        i += 1
        continue

    if in_code:
        code_buffer.append(line)
        i += 1
        continue

    # Tables
    if stripped.startswith("|") and stripped.endswith("|"):
        cells = [c.strip() for c in stripped.split("|")[1:-1]]
        # check if separator line
        if all(re.match(r'^:?-+:?$', c) for c in cells):
            table_header = False
            i += 1
            continue
        
        if not in_table:
            in_table = True
            table_header = True
            html_lines.append('<div class="table-container"><table>')

        if table_header:
            html_lines.append('<thead><tr>')
            for c in cells:
                html_lines.append(f'<th>{format_inline(c)}</th>')
            html_lines.append('</tr></thead><tbody>')
            table_header = False
        else:
            html_lines.append('<tr>')
            for c in cells:
                html_lines.append(f'<td>{format_inline(c)}</td>')
            html_lines.append('</tr>')
        i += 1
        continue
    else:
        if in_table:
            in_table = False
            html_lines.append('</tbody></table></div>')

    # Math block $$...$$
    if stripped.startswith("$$"):
        math_lines = []
        if stripped == "$$":
            i += 1
            while i < len(lines) and lines[i].strip() != "$$":
                math_lines.append(lines[i].strip())
                i += 1
        else:
            m = re.match(r'^\$\$(.*?)\$\$$', stripped)
            if m:
                math_lines.append(m.group(1))
        raw_math = " ".join(math_lines)
        raw_math = raw_math.replace(r'\begin{aligned}', '').replace(r'\end{aligned}', '')
        raw_math = raw_math.replace(r'\quad', '&nbsp;&nbsp;').replace(r'\\', '<br>')
        raw_math = raw_math.replace(r'\sum', '&sum;').replace(r'\times', '&times;')
        raw_math = raw_math.replace(r'\mathbf', '<b>').replace(r'\text', '<span>')
        raw_math = raw_math.replace(r'\le', '&le;').replace(r'\ge', '&ge;')
        raw_math = raw_math.replace(r'\Delta', '&Delta;').replace(r'\mu', '&mu;').replace(r'\sigma', '&sigma;')
        raw_math = raw_math.replace(r'\hat{v}', 'v&#770;')
        raw_math = raw_math.replace(r'\in', '&isin;').replace(r'\mathbb{R}', '&#8477;')
        raw_math = raw_math.replace(r'\to', '&rarr;').replace(r'\longrightarrow', '&rarr;')
        html_lines.append(f'<div class="math-block">{raw_math}</div>')
        i += 1
        continue

    # Blockquote
    if stripped.startswith(">"):
        quote_text = stripped[1:].strip()
        html_lines.append(f'<blockquote class="alert-box">{format_inline(quote_text)}</blockquote>')
        i += 1
        continue

    # Headings
    if stripped.startswith("# "):
        title = stripped[2:].strip()
        html_lines.append(f'<h1 class="doc-title">{format_inline(title)}</h1>')
        i += 1
        continue
    elif stripped.startswith("## "):
        title = stripped[3:].strip()
        # Page break before major sections (Section 0, 1, 9, 10)
        page_break = ' page-break-before' if any(title.startswith(p) for p in ["TABLA DE CAMBIOS", "SECCIÓN 0", "1. INTRODUCCIÓN", "9. DESARROLLO", "10. CONCLUSIONES"]) else ''
        html_lines.append(f'<h2 class="section-title{page_break}">{format_inline(title)}</h2>')
        i += 1
        continue
    elif stripped.startswith("### "):
        title = stripped[4:].strip()
        html_lines.append(f'<h3 class="subsection-title">{format_inline(title)}</h3>')
        i += 1
        continue
    elif stripped.startswith("#### "):
        title = stripped[5:].strip()
        html_lines.append(f'<h4 class="sub-subsection-title">{format_inline(title)}</h4>')
        i += 1
        continue

    # Horizontal rule
    if stripped == "---":
        html_lines.append('<hr class="divider">')
        i += 1
        continue

    # List items
    if stripped.startswith("- ") or stripped.startswith("* "):
        item_text = stripped[2:].strip()
        html_lines.append(f'<li class="bullet-item">{format_inline(item_text)}</li>')
        i += 1
        continue

    # Numbered items
    m_num = re.match(r'^(\d+)\.\s+(.*)$', stripped)
    if m_num:
        num = m_num.group(1)
        item_text = m_num.group(2)
        html_lines.append(f'<div class="num-item"><span class="num-badge">{num}.</span> {format_inline(item_text)}</div>')
        i += 1
        continue

    # Empty line
    if not stripped:
        html_lines.append('<div class="spacer"></div>')
        i += 1
        continue

    # Regular paragraph
    html_lines.append(f'<p class="body-p">{format_inline(stripped)}</p>')
    i += 1

if in_table:
    html_lines.append('</tbody></table></div>')

body_content = "\n".join(html_lines)

full_html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>TRAFFICFLOW-AGENT — Diseño Ajustado de un Agente Inteligente</title>
<style>
  @page {{
    size: A4;
    margin: 20mm 15mm 20mm 15mm;
    @bottom-right {{
      content: counter(page);
    }}
  }}
  body {{
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, 'Helvetica Neue', Arial, sans-serif;
    color: #1a202c;
    background: #ffffff;
    line-height: 1.55;
    font-size: 10pt;
    margin: 0;
    padding: 0;
  }}
  .doc-title {{
    font-size: 16pt;
    font-weight: 800;
    color: #0f2942;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .section-title {{
    font-size: 13pt;
    font-weight: 700;
    color: #0d3b66;
    border-bottom: 2px solid #0d3b66;
    padding-bottom: 4px;
    margin-top: 22px;
    margin-bottom: 10px;
    break-after: avoid;
  }}
  .subsection-title {{
    font-size: 11pt;
    font-weight: 700;
    color: #1d4e89;
    margin-top: 14px;
    margin-bottom: 6px;
    break-after: avoid;
  }}
  .sub-subsection-title {{
    font-size: 10pt;
    font-weight: 600;
    color: #2b6cb0;
    margin-top: 10px;
    margin-bottom: 4px;
    break-after: avoid;
  }}
  .page-break-before {{
    page-break-before: always;
  }}
  p.body-p {{
    margin: 6px 0;
    text-align: justify;
    text-justify: inter-word;
  }}
  .divider {{
    border: none;
    border-top: 1px solid #cbd5e1;
    margin: 12px 0;
  }}
  .spacer {{
    height: 4px;
  }}
  /* Tables */
  .table-container {{
    margin: 10px 0;
    width: 100%;
    break-inside: auto;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 8.5pt;
    line-height: 1.35;
    margin-bottom: 8px;
    break-inside: auto;
  }}
  tr {{
    break-inside: avoid;
    break-after: auto;
  }}
  th {{
    background-color: #0d3b66;
    color: #ffffff;
    font-weight: 600;
    padding: 6px 8px;
    text-align: left;
    border: 1px solid #0d3b66;
  }}
  td {{
    padding: 5px 8px;
    border: 1px solid #cbd5e1;
    vertical-align: top;
  }}
  tr:nth-child(even) td {{
    background-color: #f8fafc;
  }}
  /* Alerts */
  .alert-box {{
    background: #f0f7ff;
    border-left: 4px solid #0284c7;
    padding: 8px 12px;
    margin: 8px 0;
    font-size: 9pt;
    color: #0369a1;
    border-radius: 0 4px 4px 0;
  }}
  /* Code and math */
  code {{
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 8.5pt;
    background: #f1f5f9;
    padding: 1px 4px;
    border-radius: 3px;
    color: #b91c1c;
  }}
  .code-block {{
    background: #0f172a;
    color: #f8fafc;
    padding: 10px 12px;
    border-radius: 6px;
    font-size: 8pt;
    line-height: 1.4;
    overflow-x: auto;
    font-family: 'Consolas', 'Courier New', monospace;
    margin: 8px 0;
    break-inside: avoid;
  }}
  .code-block code {{
    background: transparent;
    color: inherit;
    padding: 0;
  }}
  .mermaid-block {{
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 8px;
    margin: 10px 0;
    break-inside: avoid;
  }}
  .code-title {{
    font-weight: 700;
    font-size: 9pt;
    color: #0d3b66;
    margin-bottom: 4px;
    padding-bottom: 2px;
    border-bottom: 1px solid #e2e8f0;
  }}
  .mermaid code {{
    font-size: 7.5pt;
    color: #334155;
    background: transparent;
    display: block;
    white-space: pre-wrap;
  }}
  .math-block {{
    background: #f8fafc;
    border-left: 3px solid #3b82f6;
    padding: 8px 12px;
    margin: 8px 0;
    font-family: 'Cambria Math', 'Times New Roman', serif;
    font-size: 9.5pt;
    color: #1e3a8a;
    text-align: center;
  }}
  .math-inline {{
    font-family: 'Cambria Math', 'Times New Roman', serif;
    font-style: italic;
    color: #1e3a8a;
  }}
  .num-item {{
    margin: 4px 0 4px 12px;
    text-align: justify;
  }}
  .num-badge {{
    font-weight: bold;
    color: #0d3b66;
  }}
  li.bullet-item {{
    margin: 3px 0 3px 18px;
    text-align: justify;
  }}
  a {{
    color: #0284c7;
    text-decoration: none;
  }}
  a:hover {{
    text-decoration: underline;
  }}
</style>
</head>
<body>
{body_content}
</body>
</html>
"""

html_path.write_text(full_html, encoding="utf-8")
print(f"HTML generado: {html_path.stat().st_size} bytes")

import time

chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

if pdf_path.exists():
    try:
        pdf_path.unlink()
    except Exception:
        pass

ps_cmd = [
    "powershell", "-NoProfile", "-Command",
    f"Start-Process -FilePath '{chrome_path}' -ArgumentList @('--headless=new', '--no-sandbox', '--disable-gpu', '--no-pdf-header-footer', '--print-to-pdf={pdf_path.resolve()}', '{html_path.resolve()}') -Wait"
]

proc = subprocess.run(ps_cmd, capture_output=True, text=True)
time.sleep(2)
if pdf_path.exists():
    print(f"PDF generado exitosamente: {pdf_path.stat().st_size} bytes")
else:
    print("Error al generar PDF. Stdout:", proc.stdout, "Stderr:", proc.stderr)

