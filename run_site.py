"""Start the MoMo Scam Check website on http://localhost:8765"""
import sys
import webbrowser
from pathlib import Path
from threading import Timer

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    uvicorn.run("api:app", host="127.0.0.1", port=port)
