# -*- coding: utf-8 -*-
"""
Master Builder for SIH26066 Team Master Decision Guide
Compiles Markdown, HTML, and high-fidelity PDF via Playwright
Target: 26-28 pages with zero overflow stubs and clean scientific typography
"""

import os
import re
import sys
from pypdf import PdfReader
from playwright.sync_api import sync_playwright

from blueprint_parts.part_01_to_05 import get_parts as p_1_5
from blueprint_parts.part_06_to_10 import get_parts as p_6_10
from blueprint_parts.part_11_to_15 import get_parts as p_11_15
from blueprint_parts.part_16_to_20 import get_parts as p_16_20
from blueprint_parts.part_21_to_25 import get_parts as p_21_25
from blueprint_parts.part_26_to_30 import get_parts as p_26_30
from blueprint_parts.part_31_to_35 import get_parts as p_31_35

def build_all():
    parts = []
    parts.extend(p_1_5())
    parts.extend(p_6_10())
    parts.extend(p_11_15())
    parts.extend(p_16_20())
    parts.extend(p_21_25())
    parts.extend(p_26_30())
    parts.extend(p_31_35())

    print(f"Total parts loaded: {len(parts)}")
    
    # 1. Build Clean Markdown File
    md_content = "\n\n---\n\n".join(parts)
    md_path = os.path.abspath("docs/SIH26066_TEAM_MASTER_DECISION_GUIDE.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved Markdown to: {md_path} ({len(md_content)} characters)")

    # 2. Build Thematic Chapters for Optimal Pagination
    chapter_starts = {0, 1, 2, 4, 7, 9, 13, 14, 15, 17, 20, 22, 25, 26, 28, 32, 33, 34}

    html_body = convert_markdown_to_html(parts, chapter_starts)

    html_full = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SIH26066: OCEANEMBED — TEAM MASTER DECISION &amp; EXECUTION BLUEPRINT</title>
<style>
  @page {{
    size: A4;
    margin: 15mm 13mm 17mm 13mm;
  }}
  *, *:before, *:after {{
    box-sizing: border-box;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1e293b;
    background-color: #ffffff;
    font-size: 8.3pt;
    line-height: 1.38;
    margin: 0;
    padding: 0;
  }}
  h1 {{
    color: #0f172a;
    font-size: 13.5pt;
    font-weight: 800;
    margin-top: 8px;
    margin-bottom: 6px;
    padding-bottom: 3px;
    border-bottom: 2px solid #0284c7;
    letter-spacing: -0.02em;
    page-break-after: avoid;
  }}
  h2 {{
    color: #0369a1;
    font-size: 10.5pt;
    font-weight: 700;
    margin-top: 8px;
    margin-bottom: 4px;
    letter-spacing: -0.01em;
    page-break-after: avoid;
  }}
  h3 {{
    color: #0f172a;
    font-size: 9.0pt;
    font-weight: 700;
    margin-top: 6px;
    margin-bottom: 3px;
    page-break-after: avoid;
  }}
  p {{
    margin-top: 0;
    margin-bottom: 4px;
    text-align: justify;
  }}
  ul, ol {{
    margin-top: 0;
    margin-bottom: 4px;
    padding-left: 15px;
  }}
  li {{
    margin-bottom: 2px;
  }}
  strong {{
    color: #0f172a;
    font-weight: 600;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 5px 0 8px 0;
    font-size: 7.3pt;
    page-break-inside: avoid;
  }}
  th {{
    background-color: #0f172a;
    color: #f8fafc;
    font-weight: 600;
    text-align: left;
    padding: 4px 6px;
    border: 1px solid #334155;
  }}
  td {{
    padding: 3px 5px;
    border: 1px solid #cbd5e1;
    vertical-align: top;
  }}
  tr:nth-child(even) {{
    background-color: #f8fafc;
  }}
  pre, code {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  }}
  pre {{
    background-color: #0f172a;
    color: #f1f5f9;
    padding: 6px 8px;
    border-radius: 4px;
    font-size: 6.9pt;
    line-height: 1.22;
    overflow-x: hidden;
    margin: 4px 0 7px 0;
    page-break-inside: avoid;
    white-space: pre;
    border: 1px solid #334155;
  }}
  p code, li code, td code {{
    background-color: #f1f5f9;
    color: #0369a1;
    padding: 1px 3px;
    border-radius: 3px;
    font-size: 7.6pt;
    border: 1px solid #e2e8f0;
  }}
  blockquote {{
    margin: 5px 0;
    padding: 4px 8px;
    background-color: #f0f9ff;
    border-left: 4px solid #0284c7;
    color: #0369a1;
    font-size: 7.9pt;
    border-radius: 0 4px 4px 0;
    page-break-inside: avoid;
  }}
  blockquote p {{
    margin: 0;
  }}
  hr {{
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 8px 0;
  }}
  .badge {{
    display: inline-block;
    padding: 1px 4px;
    font-size: 6.5pt;
    font-weight: 700;
    border-radius: 3px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .badge-verified {{ background-color: #dcfce7; color: #15803d; border: 1px solid #86efac; }}
  .badge-pilot {{ background-color: #dbeafe; color: #1d4ed8; border: 1px solid #93c5fd; }}
  .badge-catalog {{ background-color: #f3e8ff; color: #7e22ce; border: 1px solid #d8b4fe; }}
  .badge-planned {{ background-color: #fef3c7; color: #b45309; border: 1px solid #fcd34d; }}
  .badge-risk {{ background-color: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }}
  .page-break {{
    page-break-before: always;
  }}
  .cover-container {{
    padding-top: 25px;
    text-align: left;
  }}
  .cover-badge {{
    display: inline-block;
    background-color: #0284c7;
    color: white;
    padding: 3px 8px;
    font-size: 7.8pt;
    font-weight: 700;
    letter-spacing: 0.08em;
    border-radius: 4px;
    margin-bottom: 10px;
  }}
  .cover-title {{
    font-size: 22pt;
    font-weight: 900;
    color: #0f172a;
    line-height: 1.15;
    margin-bottom: 6px;
    letter-spacing: -0.03em;
  }}
  .cover-subtitle {{
    font-size: 10.5pt;
    color: #475569;
    line-height: 1.35;
    margin-bottom: 14px;
    max-width: 90%;
  }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""
    html_path = os.path.abspath("docs/SIH26066_TEAM_MASTER_DECISION_GUIDE.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_full)
    print(f"Saved HTML to: {html_path}")

    # 3. Render High-Fidelity PDF via Playwright
    pdf_path = os.path.abspath("docs/SIH26066_TEAM_MASTER_DECISION_GUIDE.pdf")
    print("Rendering PDF via Playwright Chromium...")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"file:///{html_path.replace(os.sep, '/')}")
        
        header_html = """
        <div style="font-size: 6.8pt; font-family: -apple-system, sans-serif; width: 100%; text-align: right; padding-right: 13mm; color: #64748b; font-weight: 500;">
          SIH26066: OCEANEMBED — TEAM MASTER DECISION &amp; EXECUTION BLUEPRINT
        </div>
        """
        footer_html = """
        <div style="font-size: 6.8pt; font-family: -apple-system, sans-serif; width: 100%; text-align: center; color: #64748b; font-weight: 500;">
          Page <span class="pageNumber"></span> of <span class="totalPages"></span> &nbsp;|&nbsp;
          CONFIDENTIAL // INTERNAL TEAM USE ONLY &nbsp;|&nbsp; MINISTRY OF EARTH SCIENCES / INCOIS DOMAIN
        </div>
        """
        
        pdf_bytes = page.pdf(
            path=pdf_path,
            format="A4",
            display_header_footer=True,
            header_template=header_html,
            footer_template=footer_html,
            margin={"top": "15mm", "bottom": "15mm", "left": "13mm", "right": "13mm"},
            print_background=True
        )
        browser.close()
        
    print(f"Saved PDF to: {pdf_path}")
    
    # 4. Verify Page Count with PyPDF
    reader = PdfReader(pdf_path)
    page_count = len(reader.pages)
    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    print(f"==================================================")
    print(f"PDF VERIFICATION COMPLETE:")
    print(f"Path:       {pdf_path}")
    print(f"Page Count: {page_count} pages")
    print(f"File Size:  {file_size_mb:.2f} MB")
    print(f"==================================================")
    return pdf_path, md_path, page_count

def convert_markdown_to_html(parts, chapter_starts):
    html_sections = []
    
    for idx, part in enumerate(parts):
        lines = part.split("\n")
        in_code_block = False
        code_lines = []
        in_table = False
        table_lines = []
        section_html = []
        
        if idx in chapter_starts and idx > 0:
            section_html.append('<div class="page-break"></div>')
        elif idx > 0:
            section_html.append('<hr style="margin: 8px 0; border: none; border-top: 1px dashed #cbd5e1;"/>')
            
        for line in lines:
            if line.strip().startswith("```"):
                if in_code_block:
                    in_code_block = False
                    code_content = "\n".join(code_lines)
                    code_content = code_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    section_html.append(f"<pre><code>{code_content}</code></pre>")
                    code_lines = []
                else:
                    in_code_block = True
                    code_lines = []
                continue
                
            if in_code_block:
                code_lines.append(line)
                continue
                
            if line.strip().startswith("|") and line.strip().endswith("|"):
                if not in_table:
                    in_table = True
                    table_lines = [line]
                else:
                    table_lines.append(line)
                continue
            else:
                if in_table:
                    in_table = False
                    section_html.append(render_table(table_lines))
                    table_lines = []
            
            if line.startswith("# "):
                title_text = line[2:].strip()
                if idx == 0:
                    section_html.append(f'<div class="cover-container"><div class="cover-badge">SMART INDIA HACKATHON 2026 // PS: SIH26066</div><div class="cover-title">{title_text}</div>')
                else:
                    section_html.append(f"<h1>{title_text}</h1>")
                continue
            elif line.startswith("## "):
                sub_text = line[3:].strip()
                if idx == 0:
                    section_html.append(f'<div class="cover-subtitle">{sub_text}</div>')
                else:
                    section_html.append(f"<h2>{sub_text}</h2>")
                continue
            elif line.startswith("### "):
                section_html.append(f"<h3>{line[4:].strip()}</h3>")
                continue
            elif line.startswith("#### "):
                section_html.append(f"<h4>{line[5:].strip()}</h4>")
                continue
                
            if line.startswith("> "):
                section_html.append(f"<blockquote><p>{line[2:].strip()}</p></blockquote>")
                continue
                
            if line.strip() in ["---", "***", "___"]:
                section_html.append("<hr/>")
                continue
                
            if line.strip().startswith("- ") or line.strip().startswith("* "):
                clean_li = parse_inline_formatting(line.strip()[2:].strip())
                section_html.append(f"<ul><li>{clean_li}</li></ul>")
                continue
            if re.match(r"^\d+\.\s+", line.strip()):
                clean_li = parse_inline_formatting(re.sub(r"^\d+\.\s+", "", line.strip()))
                section_html.append(f"<ol><li>{clean_li}</li></ol>")
                continue
                
            if line.strip():
                clean_p = parse_inline_formatting(line.strip())
                section_html.append(f"<p>{clean_p}</p>")
                
        if in_table:
            section_html.append(render_table(table_lines))
        if in_code_block:
            code_content = "\n".join(code_lines).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            section_html.append(f"<pre><code>{code_content}</code></pre>")
            
        if idx == 0:
            section_html.append("</div>")
            
        html_sections.append("\n".join(section_html))
        
    full_body = "\n".join(html_sections)
    full_body = full_body.replace("</ul>\n<ul>", "\n").replace("</ol>\n<ol>", "\n")
    return full_body

def render_table(table_lines):
    if len(table_lines) < 2:
        return ""
    header_line = table_lines[0]
    header_cells = [c.strip() for c in header_line.strip("|").split("|")]
    
    html = ["<table><thead><tr>"]
    for c in header_cells:
        html.append(f"<th>{parse_inline_formatting(c)}</th>")
    html.append("</tr></thead><tbody>")
    
    for row in table_lines[2:]:
        cells = [c.strip() for c in row.strip("|").split("|")]
        html.append("<tr>")
        for cell in cells:
            html.append(f"<td>{parse_inline_formatting(cell)}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    return "".join(html)

def parse_inline_formatting(text):
    text = text.replace("`VERIFIED`", '<span class="badge badge-verified">VERIFIED</span>')
    text = text.replace("`PILOT-VERIFIED`", '<span class="badge badge-pilot">PILOT-VERIFIED</span>')
    text = text.replace("`OFFICIAL-CATALOG-VERIFIED`", '<span class="badge badge-catalog">CATALOG-VERIFIED</span>')
    text = text.replace("`PLANNED`", '<span class="badge badge-planned">PLANNED</span>')
    text = text.replace("`OPEN RISK / UNVERIFIED`", '<span class="badge badge-risk">OPEN RISK</span>')
    text = text.replace("`MEASURED`", '<span class="badge badge-verified">MEASURED</span>')
    text = text.replace("`PASSED`", '<span class="badge badge-verified">PASSED</span>')
    
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    
    text = text.replace("\\circ", "&deg;")
    text = text.replace("\\times", "&times;")
    text = text.replace("\\le", "&le;").replace("\\ge", "&ge;")
    text = text.replace("\\ll", "&Lt;")
    text = text.replace("\\sim", "~")
    text = text.replace("\\approx", "&asymp;")
    text = text.replace("\\in", "&isin;")
    text = text.replace("\\mathbb{R}", "R")
    text = text.replace("\\partial", "&part;")
    text = text.replace("\\Delta", "&Delta;")
    text = text.replace("\\dots", "...")
    text = text.replace("\\zeta", "&zeta;")
    text = text.replace("\\delta", "&delta;")
    text = text.replace("\\nabla", "&nabla;")
    text = text.replace("\\rho", "&rho;")
    text = text.replace("\\lambda", "&lambda;")
    text = text.replace("\\theta", "&theta;")
    text = text.replace("\\phi", "&phi;")
    text = text.replace("\\gamma", "&gamma;")
    text = text.replace("\\eta", "&eta;")
    text = text.replace("\\mathcal{L}", "L")
    text = text.replace("\\mathcal{D}", "D")
    
    text = re.sub(r"\\text\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1 / \2", text)
    text = text.replace("$", "")
    
    return text

if __name__ == "__main__":
    build_all()
