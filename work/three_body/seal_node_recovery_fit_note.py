"""Seal the local-fit visual QA note without repeating inference or modifying data."""

import hashlib
import json
from pathlib import Path

from medaka3d.provenance import provenance

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work/three_body/node_recovery_local_fit_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    seal_path = OUT / "fit.provenance.json"
    seal = json.loads(seal_path.read_text())
    assert seal["status"] == "ok"
    assert not (OUT / "report.provenance.json").exists()
    with provenance("local_fit_post_run_visual_qa", OUT, name="report") as record:
        record.input("data_seal", seal_path, sha256=sha(seal_path))
        record.input("sealer", Path(__file__), sha256=sha(__file__))
        for key, item in seal["outputs"].items():
            assert sha(item["path"]) == item["sha256"]
            record.input(key, item["path"], sha256=item["sha256"])
        report = OUT / "RESULT.md"
        record.param("inference_repeated", False)
        record.param("visual_qa_frames", [80261, 80431, 80344, 80370])
        record.output("result_note", report, sha256=sha(report))
    print(json.dumps({"note_sha256": sha(report), "report_seal_sha256":
                      sha(OUT / "report.provenance.json")}))


if __name__ == "__main__":
    main()
