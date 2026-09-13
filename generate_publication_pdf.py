import os
import re
import base64
import subprocess
import markdown

WORKSPACE = "/Users/anuj9009/.gemini/antigravity-ide/scratch/butterfly_robustness"
PAPER_MD = os.path.join(WORKSPACE, "PAPER.md")
OUTPUT_HTML = os.path.join(WORKSPACE, "paper_rendered.html")
OUTPUT_PDF_REPO = os.path.join(WORKSPACE, "Overcoming_Subpopulation_Shift_Butterfly_Robustness_Anuj_Kumar.pdf")
OUTPUT_PDF_DOWNLOADS = "/Users/anuj9009/Downloads/Overcoming_Subpopulation_Shift_Butterfly_Robustness_Anuj_Kumar.pdf"

def img_to_base64(match):
    alt_text = match.group(1)
    img_path = match.group(2).strip()

    # Resolve local path
    if img_path.startswith("file://"):
        img_path = img_path.replace("file://", "")
    elif not os.path.isabs(img_path):
        img_path = os.path.join(WORKSPACE, img_path)

    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(img_path)[1].lower().replace(".", "")
        mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"
        return f'![{alt_text}](data:{mime};base64,{encoded})'
    return match.group(0)

def main():
    print(">>> Reading PAPER.md...")
    with open(PAPER_MD, "r", encoding="utf-8") as f:
        md_content = f.read()

    # 1. Convert markdown image links to base64 inline images
    print(">>> Encoding local figures to base64...")
    md_content = re.sub(r'!\[(.*?)\]\((.*?)\)', img_to_base64, md_content)

    # 2. Convert markdown to HTML
    print(">>> Converting Markdown to HTML...")
    html_body = markdown.markdown(
        md_content,
        extensions=[
            'tables',
            'fenced_code',
            'codehilite',
            'toc',
            'sane_lists'
        ]
    )

    # 3. Wrap in complete Academic Publication HTML template with MathJax & CSS
    html_full = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Overcoming Subpopulation Shift and Shortcut Learning in Deep Vision Classifiers</title>
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true
  }},
  options: {{
    ignoreHtmlClass: 'tex2jax_ignore',
    processHtmlClass: 'tex2jax_process'
  }}
}};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  @page {{
    size: letter;
    margin: 20mm 18mm 22mm 18mm;
    @bottom-center {{
      content: counter(page);
      font-family: 'Times New Roman', Times, serif;
      font-size: 9pt;
      color: #555;
    }}
  }}

  body {{
    font-family: 'Times New Roman', Times, Georgia, serif;
    font-size: 10.5pt;
    line-height: 1.55;
    color: #1a1a1a;
    background-color: #ffffff;
    max-width: 820px;
    margin: 0 auto;
    padding: 0;
  }}

  h1 {{
    font-size: 20pt;
    font-weight: bold;
    text-align: center;
    line-height: 1.25;
    margin-top: 0;
    margin-bottom: 12pt;
    color: #0f172a;
  }}

  h2 {{
    font-size: 13.5pt;
    font-weight: bold;
    border-bottom: 1.5px solid #334155;
    padding-bottom: 4px;
    margin-top: 24pt;
    margin-bottom: 8pt;
    color: #1e293b;
    page-break-after: avoid;
  }}

  h3 {{
    font-size: 11.5pt;
    font-weight: bold;
    margin-top: 14pt;
    margin-bottom: 6pt;
    color: #334155;
    page-break-after: avoid;
  }}

  p {{
    margin-top: 0;
    margin-bottom: 8pt;
    text-align: justify;
  }}

  /* Title block & metadata */
  p:has(strong:contains("Author")) {{
    text-align: center;
    font-size: 11pt;
  }}

  hr {{
    border: none;
    border-top: 1px solid #cbd5e1;
    margin: 18pt 0;
  }}

  /* Tables */
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 14pt 0;
    font-size: 9pt;
    page-break-inside: avoid;
  }}

  th {{
    border-top: 2px solid #0f172a;
    border-bottom: 1.5px solid #0f172a;
    padding: 6px 8px;
    text-align: left;
    font-weight: bold;
    background-color: #f8fafc;
  }}

  td {{
    border-bottom: 1px solid #e2e8f0;
    padding: 6px 8px;
    vertical-align: top;
  }}

  tr:last-child td {{
    border-bottom: 2px solid #0f172a;
  }}

  /* Code listings */
  pre {{
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 3.5px solid #2563eb;
    padding: 10px 12px;
    border-radius: 4px;
    font-family: 'Courier New', Courier, monospace;
    font-size: 8.5pt;
    line-height: 1.4;
    overflow-x: auto;
    page-break-inside: avoid;
    margin: 10pt 0;
  }}

  code {{
    font-family: 'Courier New', Courier, monospace;
    font-size: 9pt;
    background-color: #f1f5f9;
    padding: 1px 4px;
    border-radius: 3px;
  }}

  pre code {{
    background-color: transparent;
    padding: 0;
  }}

  /* Images / Figures */
  img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 12pt auto 6pt auto;
    border-radius: 4px;
  }}

  p:has(> img) {{
    text-align: center;
    page-break-inside: avoid;
  }}

  /* Blockquotes & Callouts */
  blockquote {{
    border-left: 3.5px solid #eab308;
    background-color: #fefce8;
    padding: 8px 14px;
    margin: 12pt 0;
    font-size: 9.5pt;
    border-radius: 0 4px 4px 0;
  }}

  /* Links */
  a {{
    color: #2563eb;
    text-decoration: none;
  }}

  a:hover {{
    text-decoration: underline;
  }}

  ul, ol {{
    margin-top: 0;
    margin-bottom: 8pt;
    padding-left: 20px;
  }}

  li {{
    margin-bottom: 3pt;
  }}
</style>
</head>
<body class="tex2jax_process">
{html_body}
</body>
</html>
"""

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_full)
    print(f">>> HTML successfully written to {OUTPUT_HTML}")

    # 4. Render to PDF via Headless Chrome with MathJax virtual time budget
    chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    print(">>> Compiling PDF via Headless Chrome (rendering MathJax & high-res figures)...")
    cmd = [
        chrome_bin,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=6000",
        f"--print-to-pdf={OUTPUT_PDF_REPO}",
        OUTPUT_HTML
    ]
    subprocess.run(cmd, check=True)

    # 5. Copy to Downloads folder
    subprocess.run(["cp", OUTPUT_PDF_REPO, OUTPUT_PDF_DOWNLOADS], check=True)

    repo_size = os.path.getsize(OUTPUT_PDF_REPO) / (1024 * 1024)
    print(f"\n=======================================================")
    print(f"✅ PUBLICATION PDF SUCCESSFULLY GENERATED!")
    print(f"=======================================================")
    print(f"Repository Copy: {OUTPUT_PDF_REPO} ({repo_size:.2f} MB)")
    print(f"Downloads Copy:  {OUTPUT_PDF_DOWNLOADS}")
    print(f"=======================================================\n")

if __name__ == "__main__":
    main()
