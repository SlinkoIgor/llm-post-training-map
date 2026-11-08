#!/usr/bin/env python3
"""Build a single-file, offline HTML version of the post-training map."""
import base64
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build():
    html = (ROOT / "index.html").read_text()
    css = (ROOT / "vendor/katex.min.css").read_text()
    # Keep one modern webfont per face; embed it for a portable offline file.
    def embed_font(match):
        relative = match.group(1)
        path = ROOT / "vendor" / relative
        data = base64.b64encode(path.read_bytes()).decode()
        return "url(data:font/woff2;base64," + data + ") format('woff2')"

    css = re.sub(
        r"url\((fonts/[^)]+\.woff2)\) format\([\"']woff2[\"']\)(?:,url\([^)]+\) format\([^)]+\))*",
        embed_font, css,
    )
    if re.search(r"url\(fonts/", css):
        raise RuntimeError("Unembedded font reference")
    katex_js = (ROOT / "vendor/katex.min.js").read_text()
    app_js = (ROOT / "app.js").read_text()
    # A literal closing script sequence must not appear inside an inline script.
    katex_js = re.sub(r"</script", r"<\\/script", katex_js, flags=re.I)
    app_js = re.sub(r"</script", r"<\\/script", app_js, flags=re.I)
    html = html.replace('<link rel="stylesheet" href="vendor/katex.min.css">', "<style>" + css + "</style>")
    html = re.sub(r'<link rel="stylesheet" href="styles\.css(?:\?[^"]*)?">', lambda _: "<style>" + (ROOT / "styles.css").read_text() + "</style>", html)
    html = html.replace('<script defer src="vendor/katex.min.js"></script>', "")
    html = re.sub(r'<script defer src="app\.js(?:\?[^"]*)?"></script>', "", html)
    html = html.replace("</body>", "<script>" + katex_js + "</script><script>" + app_js + "</script></body>")
    output = ROOT / "dist/post-training-map.html"
    output.parent.mkdir(exist_ok=True)
    output.write_text(html)
    print(str(output))
    print(f"{output.stat().st_size:,} bytes; scripts, styles, and fonts embedded.")
    return output


if __name__ == "__main__":
    build()
