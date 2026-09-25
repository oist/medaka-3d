"""Download only fixed official DINOv2 assets; never import model code or weights.

No package installation, Torch Hub execution, model load, GPU, retry or alternate
mirror. A network rejection is preserved as failure in the new output directory.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import shutil
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "work/three_body/assets"
COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
MODEL = "dinov2_vits14"
BASE = f"https://raw.githubusercontent.com/facebookresearch/dinov2/{COMMIT}/"
WEIGHT_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
SOURCE_PATHS = (
    "LICENSE",
    "MODEL_CARD.md",
    "README.md",
    "dinov2/__init__.py",
    "dinov2/hub/__init__.py",
    "dinov2/hub/backbones.py",
    "dinov2/hub/utils.py",
    "dinov2/models/__init__.py",
    "dinov2/models/vision_transformer.py",
    "dinov2/layers/__init__.py",
    "dinov2/layers/attention.py",
    "dinov2/layers/block.py",
    "dinov2/layers/dino_head.py",
    "dinov2/layers/drop_path.py",
    "dinov2/layers/layer_scale.py",
    "dinov2/layers/mlp.py",
    "dinov2/layers/patch_embed.py",
    "dinov2/layers/swiglu_ffn.py",
    "dinov2/data/transforms.py",  # Recorded preprocessing source; not imported.
)
STAGE = "official_dinov2_vits14_asset_acquisition_v1"


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


def requests():
    return [
        dict(path="source/" + path, url=BASE + path, maximum_bytes=2 * 1024**2)
        for path in SOURCE_PATHS
    ] + [
        dict(path="weights/dinov2_vits14_pretrain.pth", url=WEIGHT_URL, maximum_bytes=128 * 1024**2)
    ]


def validate_request(item):
    if item not in requests():
        raise ValueError("only exact registered official asset requests are allowed")
    url = urlparse(item["url"])
    if (
        url.scheme != "https"
        or url.hostname not in {"raw.githubusercontent.com", "dl.fbaipublicfiles.com"}
        or url.username is not None
        or url.password is not None
        or url.port is not None
    ):
        raise ValueError("fixed official HTTPS origin required")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("redirect rejected; no alternate URL or mirror will be tried")


def download(item, root, opener):
    validate_request(item)
    path = root / item["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(item["url"], headers={"User-Agent": "medaka3d-asset-audit/1"})
    size, digest = 0, hashlib.sha256()
    with opener.open(request, timeout=45) as response:
        if response.status != 200 or response.geturl() != item["url"]:
            raise ValueError("official response status/URL changed")
        length_header = response.headers.get("Content-Length")
        length = int(length_header) if length_header is not None else None
        if length is not None and not 0 <= length <= item["maximum_bytes"]:
            raise ValueError("registered download size cap exceeded")
        with path.open("xb") as stream:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                size += len(block)
                if size > item["maximum_bytes"]:
                    raise ValueError("registered streamed size cap exceeded")
                stream.write(block)
                digest.update(block)
        headers = {
            key: response.headers.get(key)
            for key in ("Content-Length", "Content-Type", "ETag", "Last-Modified")
        }
    if not size or (length is not None and size != length):
        raise ValueError("empty or truncated official response")
    if path.suffix == ".py":
        ast.parse(path.read_text(encoding="utf-8"), filename=item["path"])
    return dict(
        **item,
        size_bytes=size,
        sha256=digest.hexdigest(),
        retrieved_at=now(),
        response_headers=headers,
        published_sha256=None,
        sha256_origin="measured from downloaded bytes",
    )


def runtime():
    return dict(
        python=sys.version,
        executable=sys.executable,
        packages={
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "numpy", "Pillow")
        },
    )


def acquire(destination):
    destination = Path(destination).resolve()
    if not destination.is_relative_to(ASSETS) or destination.exists():
        raise ValueError("a new dedicated workspace assets directory is required")
    source_sha = sha(__file__)
    context = runtime()
    destination.mkdir(parents=True)
    outputs, downloaded = {}, []
    started = now()

    def record_file(path):
        outputs[str(path.relative_to(destination))] = dict(
            path=str(path), sha256=sha(path), size_bytes=path.stat().st_size
        )

    try:
        plan = dict(
            schema=STAGE,
            official_repository="https://github.com/facebookresearch/dinov2",
            commit=COMMIT,
            model=MODEL,
            with_registers=False,
            requests=requests(),
            weight_url=WEIGHT_URL,
            license="Apache-2.0; standard DINOv2 model card",
            official_source_unmodified=True,
            network_retry=False,
            alternate_mirror=False,
            model_imported=False,
            weights_deserialized=False,
            GPU_used=False,
            packages_installed=False,
            runtime=context,
            started_at=started,
        )
        write_json(destination / "acquisition_plan.json", plan)
        record_file(destination / "acquisition_plan.json")
        snapshot = destination / "fetch_dinov2_assets.py"
        shutil.copyfile(__file__, snapshot)
        if sha(snapshot) != source_sha:
            raise ValueError("acquisition source changed before snapshot")
        record_file(snapshot)
        opener = urllib.request.build_opener(NoRedirect())
        for item in requests():
            downloaded.append(download(item, destination, opener))
            record_file(destination / item["path"])
            print(
                json.dumps({key: downloaded[-1][key] for key in ("path", "size_bytes", "sha256")}),
                flush=True,
            )
        manifest = dict(
            schema="official_dinov2_vits14_assets_v1",
            status="ok",
            commit=COMMIT,
            model=MODEL,
            with_registers=False,
            license="Apache-2.0",
            assets=downloaded,
            runtime=context,
            acquisition_source_sha256=source_sha,
            model_imported=False,
            weights_deserialized=False,
            GPU_used=False,
            upstream_published_weight_sha256=None,
            hash_scope="Local SHA256 fixes the received official HTTPS bytes; "
            "no independently published weight checksum was obtained.",
        )
        for item in downloaded:
            if sha(destination / item["path"]) != item["sha256"]:
                raise ValueError("downloaded asset changed before manifest")
        if runtime() != context or sha(__file__) != source_sha:
            raise ValueError("runtime or acquisition source changed")
        write_json(destination / "assets.manifest.json", manifest)
        record_file(destination / "assets.manifest.json")
        if any(sha(v["path"]) != v["sha256"] for v in outputs.values()):
            raise ValueError("asset output changed before success seal")
        write_json(
            destination / "acquisition.provenance.json",
            dict(
                stage=STAGE,
                status="ok",
                started_at=started,
                finished_at=now(),
                inputs={str(Path(__file__).resolve()): source_sha},
                outputs=outputs,
                no_model_load=True,
                no_GPU=True,
                no_package_install=True,
            ),
        )
    except BaseException as error:
        write_json(
            destination / "failure.json",
            dict(
                stage=STAGE,
                status="failed",
                started_at=started,
                finished_at=now(),
                error=repr(error),
                completed_assets=downloaded,
                no_retry_or_alternate_network_path=True,
                success_seal=False,
            ),
        )
        raise
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    acquire(parser.parse_args().output)
