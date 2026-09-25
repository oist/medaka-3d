"""Synthetic HTTP/path/error fixtures; never contact a network or load a model."""

import hashlib
import io
import json
from types import SimpleNamespace

import fetch_dinov2_assets as m
import pytest


class Response(io.BytesIO):
    def __init__(self, body, url, *, size=None, status=200):
        super().__init__(body)
        self.status, self.url = status, url
        self.headers = {"Content-Length": str(len(body) if size is None else size)}

    def geturl(self):
        return self.url


def test_fixed_urls_are_commit_pinned_and_standard_nonregister_model():
    items = m.requests()
    assert len(items) == len(m.SOURCE_PATHS) + 1
    assert len({item["path"] for item in items}) == len(items)
    for item in items:
        m.validate_request(item)
    assert all(m.COMMIT in item["url"] for item in items[:-1])
    assert items[-1]["url"] == m.WEIGHT_URL
    assert "reg4" not in m.WEIGHT_URL


@pytest.mark.parametrize(
    "field,value",
    [
        ("url", "http://raw.githubusercontent.com/file"),
        ("url", "https://example.com/file"),
        ("url", "https://raw.githubusercontent.com/facebookresearch/dinov2/main/LICENSE"),
        ("path", "../../outside"),
        ("maximum_bytes", 2**40),
    ],
)
def test_unregistered_request_rejected(field, value):
    item = m.requests()[0] | {field: value}
    with pytest.raises(ValueError):
        m.validate_request(item)


def test_download_records_exact_bytes_no_execution(tmp_path):
    item = next(x for x in m.requests() if x["path"].endswith("backbones.py"))
    body = b"raise RuntimeError('never execute downloaded source')\n"
    calls = []

    def open_response(req, **kwargs):
        calls.append((req.full_url, kwargs))
        return Response(body, item["url"])

    result = m.download(item, tmp_path, SimpleNamespace(open=open_response))
    assert calls == [(item["url"], {"timeout": 45})]
    assert (tmp_path / item["path"]).read_bytes() == body
    assert result["size_bytes"] == len(body)
    assert result["sha256"] == hashlib.sha256(body).hexdigest()
    assert result["published_sha256"] is None


@pytest.mark.parametrize("fault", ["status", "url", "length", "cap", "empty"])
def test_bad_http_result_has_no_success(tmp_path, fault):
    item = m.requests()[0]
    body = b"some source text"
    response = Response(
        b"" if fault == "empty" else body,
        "https://example.com/redirect" if fault == "url" else item["url"],
        size=(
            item["maximum_bytes"] + 1
            if fault == "cap"
            else len(body) + 1
            if fault == "length"
            else None
        ),
        status=403 if fault == "status" else 200,
    )
    with pytest.raises(ValueError):
        m.download(item, tmp_path, SimpleNamespace(open=lambda *a, **k: response))


def test_redirect_is_rejected_without_followup():
    with pytest.raises(ValueError, match="redirect"):
        m.NoRedirect().redirect_request(None, None, 302, None, {}, "https://example.com")


def test_failure_preserved_without_retry_or_success_seal(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ASSETS", tmp_path)
    monkeypatch.setattr(m, "runtime", lambda: {"synthetic": True})
    calls = []

    def fail(item, root, opener):
        calls.append(item)
        raise PermissionError("synthetic network denied")

    monkeypatch.setattr(m, "download", fail)
    root = tmp_path / "new"
    with pytest.raises(PermissionError):
        m.acquire(root)
    assert len(calls) == 1
    failure = json.loads((root / "failure.json").read_text())
    assert failure["status"] == "failed" and failure["success_seal"] is False
    assert not (root / "acquisition.provenance.json").exists()
    assert not (root / "assets.manifest.json").exists()
    with pytest.raises(ValueError, match="new dedicated"):
        m.acquire(root)
    assert len(calls) == 1
