"""Build the static site: inline web/detector.js into web/index.html -> web/dist/.

    python src/export_web.py && python src/build_site.py
Then serve web/dist/ from any static host (it is two files: index.html + model.json).
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
DIST = WEB / "dist"


def main():
    DIST.mkdir(parents=True, exist_ok=True)
    page = (WEB / "index.html").read_text(encoding="utf-8")
    js = (WEB / "detector.js").read_text(encoding="utf-8")
    assert "/*DETECTOR*/" in page
    (DIST / "index.html").write_text(page.replace("/*DETECTOR*/", js), encoding="utf-8")
    shutil.copy(WEB / "model.json", DIST / "model.json")
    print("built", DIST)


if __name__ == "__main__":
    main()
