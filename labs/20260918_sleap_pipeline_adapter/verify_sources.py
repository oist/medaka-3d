"""Verify the installed upstream implementation against its public release tag."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from medaka3d.provenance import provenance

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work/sleap_pipeline_adapter_v1"
FILES = (
    "tracking/tracker.py",
    "tracking/candidates/local_queues.py",
    "tracking/utils.py",
    "evaluation.py",
)


def check(name):
    distribution = importlib.metadata.distribution("sleap-nn")
    path = Path(distribution.locate_file("sleap_nn")) / name
    url = f"https://raw.githubusercontent.com/talmolab/sleap-nn/v0.3.1/sleap_nn/{name}"
    with urllib.request.urlopen(url, timeout=30) as response:
        remote = response.read()
    local = path.read_bytes()
    return {
        "file": name,
        "installed_path": str(path),
        "github_url": url,
        "installed_sha256": hashlib.sha256(local).hexdigest(),
        "github_sha256": hashlib.sha256(remote).hexdigest(),
        "bytes_identical": local == remote,
    }


def main():
    version = importlib.metadata.version("sleap-nn")
    if version != "0.3.1":
        raise ValueError(f"Unexpected SLEAP version: {version}")
    if (OUT / "source_audit.json").exists():
        raise FileExistsError("Preserve the existing upstream verification")
    with provenance("sleap_upstream_verification", OUT) as record:
        with ThreadPoolExecutor(max_workers=4) as pool:
            files = list(pool.map(check, FILES))
        if not all(item["bytes_identical"] for item in files):
            raise AssertionError("Installed SLEAP differs from the public release")
        for item in files:
            record.input(item["file"], item["installed_path"], sha256=item["installed_sha256"])
        result = {"status": "passed", "version": version, "files": files}
        destination = OUT / "source_audit.json"
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        record.output(
            "audit", destination, sha256=hashlib.sha256(destination.read_bytes()).hexdigest()
        )
        print("Verified 4 upstream files against talmolab/sleap-nn v0.3.1")


if __name__ == "__main__":
    main()
