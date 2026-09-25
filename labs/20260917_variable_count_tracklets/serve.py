"""Loopback report preview with byte ranges for seeking the native source videos."""

from __future__ import annotations

import argparse
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self._remaining = None
        path = Path(self.translate_path(self.path))
        range_header = self.headers.get("Range")
        if not range_header or not path.is_file():
            return super().send_head()
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
        size = path.stat().st_size
        if match is None or not any(match.groups()):
            self.send_error(416, "Invalid byte range")
            return None
        first, last = match.groups()
        start = int(first) if first else max(0, size - int(last))
        stop = min(size - 1, int(last)) if first and last else size - 1
        if start > stop or start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        handle = path.open("rb")
        handle.seek(start)
        self._remaining = stop - start + 1
        self.send_response(206)
        self.send_header("Content-type", self.guess_type(str(path)))
        self.send_header("Content-Length", str(self._remaining))
        self.send_header("Content-Range", f"bytes {start}-{stop}/{size}")
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        return handle

    def copyfile(self, source, outputfile):
        try:
            if self._remaining is None:
                return super().copyfile(source, outputfile)
            remaining = self._remaining
            while remaining > 0:
                data = source.read(min(1024 * 1024, remaining))
                if not data:
                    break
                outputfile.write(data)
                remaining -= len(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # Browsers cancel a previous read when seeking.
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), partial(RangeHandler, directory=str(ROOT))
    )
    print(f"http://127.0.0.1:{args.port}/work/variable_count_tracklets_v1/report.html", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
