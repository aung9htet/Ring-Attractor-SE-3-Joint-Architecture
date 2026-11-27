"""Turn the architecture SVGs into PDF and PNG (needs Chrome or Chromium, and Poppler's pdftoppm).

The SVGs are the figures; this only prints them to vector PDF at their journal size. Run on the host if the image has
no browser.
"""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..common import FIGURES

SIZES = ["fig1_model"]   # 178 mm wide; the height is read from the SVG


def main():
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    if not chrome:
        raise SystemExit("Chrome or Chromium is needed to print the architecture SVGs to PDF (the SVGs are already written).")
    for name in SIZES:
        svg = (FIGURES / (name + ".svg")).read_text()
        height = float(re.search(r'height="([0-9.]+)mm"', svg).group(1))
        html = ("<!doctype html><html><head><meta charset=\"utf-8\"><title>%s</title><style>@page{size:178mm %gmm;margin:0}"
                "html,body{margin:0;padding:0;width:178mm;height:%gmm}svg{display:block;width:178mm;height:%gmm}</style></head>"
                "<body>%s</body></html>") % (name, height, height, height, svg)
        with tempfile.TemporaryDirectory(prefix="architecture_") as folder:
            page = Path(folder) / (name + ".html")
            page.write_text(html)
            run = subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                                  "--print-to-pdf=" + str(FIGURES / (name + ".pdf")), page.as_uri()], capture_output=True, text=True)
        if run.returncode:
            raise SystemExit(run.stderr)
        subprocess.run(["pdftoppm", "-scale-to", "2200", "-png", "-singlefile", str(FIGURES / (name + ".pdf")), str(FIGURES / name)],
                       capture_output=True, check=True)
    print("architecture PDFs and PNGs written to analysis/figures/")


if __name__ == "__main__":
    main()
