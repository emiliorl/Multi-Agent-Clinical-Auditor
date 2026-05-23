#!/usr/bin/env python3
"""
serve_dashboard.py - Serve reports/ over HTTP so the dashboard can fetch
patients.json dynamically. Opens the browser automatically.

Usage:
    python serve_dashboard.py          # default port 8765
    python serve_dashboard.py 9000     # custom port
"""

if __name__ == "__main__":
    import os
    import sys
    import subprocess
    from pathlib import Path

    venv_dir = Path(__file__).parent.resolve() / "venv"
    if venv_dir.exists():
        try:
            is_in_venv = Path(sys.executable).resolve().is_relative_to(venv_dir)
        except AttributeError:
            try:
                Path(sys.executable).resolve().relative_to(venv_dir)
                is_in_venv = True
            except ValueError:
                is_in_venv = False

        if not is_in_venv:
            venv_python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            if venv_python.exists():
                sys.exit(subprocess.call([str(venv_python)] + sys.argv))

import http.server
import os
import sys
import threading
import webbrowser
from pathlib import Path

REPORTS_DIR = Path(__file__).parent / "reports"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765


class _Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(REPORTS_DIR), **kwargs)

    def log_message(self, fmt, *args):
        # Suppress per-request noise; only print startup banner
        pass


def main() -> None:
    if not REPORTS_DIR.exists():
        print(f"[ERROR] reports/ directory not found: {REPORTS_DIR}")
        sys.exit(1)

    url = f"http://localhost:{PORT}/dashboard.html"

    with http.server.HTTPServer(("", PORT), _Handler) as httpd:
        print("─" * 50)
        print(f"  Clinical Audit Dashboard")
        print(f"  http://localhost:{PORT}/dashboard.html")
        print(f"  Serving: {REPORTS_DIR.resolve()}")
        print(f"  Press Ctrl+C to stop")
        print("─" * 50)

        # Open browser after a short delay so the server is ready
        threading.Timer(0.5, webbrowser.open, args=[url]).start()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[INFO] Server stopped.")


if __name__ == "__main__":
    main()
