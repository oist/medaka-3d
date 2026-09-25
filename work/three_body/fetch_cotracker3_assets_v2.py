"""Acquire one official CoTracker3 checkpoint and source, without executing either.

Fresh destination only. Public GET requests carry no credentials or local data.
GitHub blobs and the HF LFS SHA/size are verified before success. No retry,
alternate mirror, installation, model import/deserialization, or GPU operation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import re
import shutil
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "work/three_body/assets"
COMMIT = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
TREE = "f0296ad047b50c1530063b67e575908257478cab"
PUBLISHED_WEIGHT_SHA = "205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218"
STAGE = "official_cotracker3_online_asset_acquisition_v2"
MAX_WEIGHT = 250_000_000
MAX_SOURCE = 5_000_000
METADATA_URLS = {
    "github_commit.json": "https://api.github.com/repos/facebookresearch/co-tracker/commits/main",
    "github_tree.json": f"https://api.github.com/repos/facebookresearch/co-tracker/git/trees/{TREE}?recursive=1",
    "huggingface_model.json": "https://huggingface.co/api/models/facebook/cotracker3?blobs=true",
}
CDN_HOSTS = {
    "cas-bridge.xethub.hf.co",
    "us.aws.cdn.hf.co",
    "cdn-lfs.huggingface.co",
    "cdn-lfs-us-1.huggingface.co",
    "cdn-lfs.hf.co",
}


def now():
    return datetime.now(UTC).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def public_url(url):
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.fragment
    ):
        raise ValueError("public HTTPS asset URL required")
    return parsed


class OfficialRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, allow_weight=False):
        super().__init__()
        self.allow_weight = allow_weight
        self.redirects = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        before, after = public_url(req.full_url), public_url(newurl)
        if not self.allow_weight or before.hostname not in ({"huggingface.co"} | CDN_HOSTS):
            raise ValueError("unexpected non-weight redirect")
        if after.hostname not in CDN_HOSTS:
            raise ValueError("weight redirect is not an official HF delivery host")
        # Signed CDN queries are not copied into logs or records.
        self.redirects.append(dict(status=code, host=after.hostname, path=after.path))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url, path, maximum_bytes, *, expected_sha=None, expected_size=None, weight=False):
    parsed = public_url(url)
    if parsed.hostname not in {"api.github.com", "raw.githubusercontent.com", "huggingface.co"}:
        raise ValueError("unexpected initial asset host")
    if weight and (expected_sha is None or expected_size is None):
        raise ValueError("weight needs upstream SHA and exact size before download")
    redirects = OfficialRedirect(weight)
    opener = urllib.request.build_opener(redirects)
    request = urllib.request.Request(url, headers={"User-Agent": "medaka3d-asset-audit/1"})
    path.parent.mkdir(parents=True, exist_ok=True)
    size, digest = 0, hashlib.sha256()
    with opener.open(request, timeout=45) as response:
        if response.status != 200:
            raise ValueError("official asset response is not 200")
        length = response.headers.get("Content-Length")
        length = int(length) if length is not None else None
        if length is not None and not 0 <= length <= maximum_bytes:
            raise ValueError(f"download size {length} exceeds cap {maximum_bytes}")
        with path.open("xb") as stream:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                size += len(block)
                if size > maximum_bytes:
                    raise ValueError("stream exceeded the registered size cap")
                stream.write(block)
                digest.update(block)
        headers = {
            key: response.headers.get(key)
            for key in ("Content-Length", "Content-Type", "ETag", "Last-Modified")
        }
    actual_sha = digest.hexdigest()
    if (length is not None and size != length) or (
        expected_size is not None and size != expected_size
    ):
        raise ValueError("download size differs from response or pinned upstream size")
    if expected_sha is not None and actual_sha != expected_sha:
        raise ValueError("download SHA differs from upstream LFS SHA")
    return dict(
        url=url,
        sha256=actual_sha,
        size_bytes=size,
        retrieved_at=now(),
        response_headers=headers,
        redirects=redirects.redirects,
        upstream_sha256=expected_sha,
    )


def resolve_metadata(commit, tree, model):
    if (
        commit["sha"] != COMMIT
        or tree["sha"] != TREE
        or commit["commit"]["tree"]["sha"] != TREE
        or tree.get("truncated") is not False
    ):
        raise ValueError("current official GitHub commit/tree differs or is truncated")
    revision = model["sha"]
    if model["id"] != "facebook/cotracker3" or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("official HF repository/revision differs")
    weights = [x for x in model["siblings"] if x["rfilename"] == "scaled_online.pth"]
    if len(weights) != 1:
        raise ValueError("exactly one scaled_online.pth metadata entry required")
    entry = weights[0]
    lfs = entry["lfs"]
    if lfs["sha256"] != PUBLISHED_WEIGHT_SHA or lfs["size"] != entry["size"]:
        raise ValueError("HF LFS checksum/size differs from published checkpoint")
    if type(entry["size"]) is not int or not 0 < entry["size"] <= MAX_WEIGHT:
        raise ValueError(f"weight size outside registered 250 MB cap: {entry['size']}")
    exact = {
        "README.md",
        "LICENSE.md",
        "setup.py",
        "hubconf.py",
        "online_demo.py",
        "cotracker/__init__.py",
        "cotracker/predictor.py",
    }
    selected = [
        x
        for x in tree["tree"]
        if x["path"] in exact
        or (x["path"].startswith("cotracker/models/") and x["path"].endswith(".py"))
        or "license" in Path(x["path"]).name.lower()
    ]
    if not exact.issubset({x["path"] for x in selected}):
        raise ValueError("required official source file missing")
    for item in selected:
        path = Path(item["path"])
        if (
            item["type"] != "blob"
            or item["mode"] != "100644"
            or path.is_absolute()
            or ".." in path.parts
        ):
            raise ValueError("source selection contains nonregular or escaping path")
        if not re.fullmatch(r"[0-9a-f]{40}", item["sha"]) or not 0 <= item["size"] <= 2_000_000:
            raise ValueError("source blob signature/size invalid")
    if sum(x["size"] for x in selected) > MAX_SOURCE:
        raise ValueError("source selection exceeds registered 5 MB cap")
    return revision, entry, sorted(selected, key=lambda x: x["path"])


def runtime():
    packages = {}
    for name in (
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "einops",
        "timm",
        "Pillow",
        "imageio",
        "matplotlib",
        "flow-vis",
        "tqdm",
        "scipy",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, executable=sys.executable, packages=packages)


def acquire(destination):
    destination = Path(destination).resolve()
    if (
        destination.parent != ASSETS.resolve()
        or not re.fullmatch(r"cotracker3_online_82e02e8_v[1-9][0-9]*", destination.name)
        or destination.exists()
    ):
        raise ValueError("fresh versioned CoTracker3 directory under workspace assets required")
    source_sha, context, started = sha(__file__), runtime(), now()
    destination.mkdir(parents=True)
    downloaded = []
    status = "failed"
    try:
        shutil.copyfile(__file__, destination / "fetch_cotracker3_assets.py")
        metadata = {}
        for name, url in METADATA_URLS.items():
            record = download(url, destination / "metadata" / name, 2_000_000)
            downloaded.append(dict(path="metadata/" + name, **record))
            metadata[name] = json.loads(
                (destination / "metadata" / name).read_text(encoding="utf-8")
            )
        revision, weight, sources = resolve_metadata(*(metadata[n] for n in METADATA_URLS))
        weight_url = (
            f"https://huggingface.co/facebook/cotracker3/resolve/{revision}/scaled_online.pth"
        )
        plan = dict(
            schema=STAGE,
            github_commit=COMMIT,
            github_tree=metadata["github_tree.json"]["sha"],
            huggingface_revision=revision,
            checkpoint=weight,
            source_blobs=sources,
            weight_url=weight_url,
            maximum_weight_bytes=MAX_WEIGHT,
            maximum_source_bytes=MAX_SOURCE,
            runtime=context,
            model_imported=False,
            weights_deserialized=False,
            GPU_used=False,
            packages_installed=False,
            network_retry=False,
            alternate_mirror=False,
        )
        write_json(destination / "acquisition_plan.json", plan)
        print(
            json.dumps(
                {
                    "phase": "metadata_pinned",
                    "commit": COMMIT,
                    "hf_revision": revision,
                    "weight_bytes": weight["size"],
                    "weight_sha256": weight["lfs"]["sha256"],
                    "source_files": len(sources),
                    "source_bytes": sum(x["size"] for x in sources),
                }
            ),
            flush=True,
        )
        for item in sources:
            path = destination / "source" / item["path"]
            record = download(
                f"https://raw.githubusercontent.com/facebookresearch/co-tracker/{COMMIT}/{item['path']}",
                path,
                2_000_000,
                expected_size=item["size"],
            )
            data = path.read_bytes()
            blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if blob != item["sha"]:
                raise ValueError("source bytes differ from pinned GitHub blob")
            if path.suffix == ".py":
                ast.parse(data, filename=item["path"])
            downloaded.append(dict(path="source/" + item["path"], github_blob_sha1=blob, **record))
            print(
                json.dumps(
                    {"phase": "source", "path": item["path"], "size_bytes": record["size_bytes"]}
                ),
                flush=True,
            )
        pointer_path = destination / "metadata/scaled_online.lfs.txt"
        pointer_record = download(
            f"https://huggingface.co/facebook/cotracker3/raw/{revision}/scaled_online.pth",
            pointer_path,
            4096,
        )
        pointer = pointer_path.read_text(encoding="utf-8").splitlines()
        if (
            f"oid sha256:{weight['lfs']['sha256']}" not in pointer
            or f"size {weight['size']}" not in pointer
        ):
            raise ValueError("pinned HF raw LFS pointer differs from repository metadata")
        downloaded.append(dict(path="metadata/scaled_online.lfs.txt", **pointer_record))
        path = destination / "weights/scaled_online.pth"
        record = download(
            weight_url,
            path,
            MAX_WEIGHT,
            expected_size=weight["size"],
            expected_sha=weight["lfs"]["sha256"],
            weight=True,
        )
        downloaded.append(dict(path="weights/scaled_online.pth", **record))
        for item in downloaded:
            if sha(destination / item["path"]) != item["sha256"]:
                raise ValueError("downloaded artifact changed before final manifest")
        if (
            runtime() != context
            or sha(__file__) != source_sha
            or sha(destination / "fetch_cotracker3_assets.py") != source_sha
        ):
            raise ValueError("acquisition source/runtime changed")
        write_json(
            destination / "assets.manifest.json",
            dict(
                schema="official_cotracker3_online_assets_v1",
                status="ok",
                github_commit=COMMIT,
                huggingface_revision=revision,
                checkpoint="scaled_online.pth",
                upstream_weight_sha256=weight["lfs"]["sha256"],
                upstream_weight_size_bytes=weight["size"],
                assets=downloaded,
                runtime=context,
                acquisition_source_sha256=source_sha,
                model_imported=False,
                weights_deserialized=False,
                GPU_used=False,
                packages_installed=False,
                license="CC-BY-NC-4.0; see official LICENSE.md and component notices",
                source_tree_scope=(
                    "predictor + models package; root demo/setup/hub/README and license records "
                    "are source-only; runtime import closure reviewed separately without imports"
                ),
            ),
        )
        status = "ok"
    except BaseException as error:
        write_json(
            destination / "failure.json",
            dict(
                schema=STAGE,
                status="failed",
                error_type=type(error).__name__,
                error=str(error)[:1500],
                completed_downloads=downloaded,
                failed_at=now(),
                model_imported=False,
                weights_deserialized=False,
                GPU_used=False,
                packages_installed=False,
            ),
        )
        status = "failed"
        raise
    finally:
        outputs = {
            str(p.relative_to(destination)): dict(
                path=str(p), sha256=sha(p), size_bytes=p.stat().st_size
            )
            for p in sorted(destination.rglob("*"))
            if p.is_file()
        }
        write_json(
            destination / "acquisition.provenance.json",
            dict(
                stage=STAGE,
                status=status,
                started_at=started,
                finished_at=now(),
                inputs={str(Path(__file__).resolve()): source_sha},
                outputs=outputs,
                no_model_load=True,
                no_GPU=True,
                no_package_install=True,
            ),
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(destination),
                "seal_sha256": sha(destination / "acquisition.provenance.json"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    acquire(parser.parse_args().output)
