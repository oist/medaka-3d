"""Seal documentation derived from the completed CPU parity audit; never edit it."""

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "labs/20260909_three_body"))
from prepare_sequence import sha256  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402


def main():
    root = REPO / "work/three_body/chunked_source_smoke_qa_v1"
    docs = REPO / "labs/20260909_three_body/CHUNKED_INPUTS.md"
    summary_path, seal_path = root / "summary.json", root / "audit.provenance.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal["status"] != "ok" or summary["status"] != "passed":
        raise ValueError("only a successful completed audit can be documented")
    existing = {str(summary_path): sha256(summary_path), str(seal_path): sha256(seal_path)}
    for info in seal["outputs"].values():
        digest = sha256(info["path"])
        if digest != info["sha256"]:
            raise ValueError("existing audit output checksum differs from its seal")
        existing[info["path"]] = digest
    if any(sha256(path) != digest for path, digest in summary["code_sha256"].items()):
        raise ValueError("frozen implementation/audit code changed before documentation")
    output = root / "RESULT.md"
    report_seal = root / "report.provenance.json"
    if output.exists() or report_seal.exists():
        raise FileExistsError("report already exists; no overwrite")
    text = f"""# 実動画 6 フレームのチャンク再開・再利用監査

**合格。** 333519:333525、166 fps、3 視点の入力を独立に新規検出し、
1 チャンク保存後の停止、残り 2 チャンクの作成、全 3 チャンクの再利用を確認した。
これは入力準備の同値性と再開処理の検証であり、追跡や ID の正解率は測定していない。

| 確認対象 | 結果 |
| --- | --- |
| 原画像 | 18 枚すべて、旧 FFV1 入力と BGR バイト完全一致 |
| 検出結果 | 55 検出・全 13 packed 配列の型、形、全バイトが一致 |
| 各視点の検出数 | top: 3,3,3,3,3,3 / side1: 4,4,4,4,4,4 / side2: 3,2,2,2,2,2 |
| 再開 | 初回保存したチャンクの manifest / provenance SHA を維持 |
| 3 回目 | 全チャンクを再利用し、完成 assembly の SHA も維持 |
| 新背景 | 登録した 5000,10000,…,45000 の 9 枚の中央値と全 3 視点で一致 |
| 旧背景との差 | 全 3 視点で異なる。登録した背景規則の変更として区別 |
| 原動画保存 | 元 MP4 の hardlink を確認。動画ペイロードのコピーは 0 バイト |
| 来歴 | 全入力・コードの監査前後 SHA が一致。監査 provenance は ok |

ゼロ検出フレームはこの実窓に含まれない。空観測の保持と途中失敗は CPU テストで
確認している。実装側 75 ケースと独立した追加回帰 3 ケースが通過した。
全 30 分の入力準備や追跡の完了を示す結果ではない。

初回は `incomplete.json` と `final_sequence_path: null` が保存された。
Python CLI が意図する終了コードは 2 だが、実測できた PowerShell wrapper の終了コードは 1。
この初回の native Python 終了コードは未計測であり、実測 2 とは扱わない。
監査では未完了の記録と、後に作られた assembly の時刻を照合した。

CPU 監査は {summary["audit_elapsed_s"]:.2f} 秒、画像 decode・比較は
{summary["cpu_decode_compare_s"]:.2f} 秒だった。GPU は使用していない。
繰り返しの元動画 full SHA と同時実行の影響を含むため、D2 の処理速度とは比較しない。

一次結果: [summary.json](summary.json)、[audit.provenance.json](audit.provenance.json)。
この文書の生成は別の [report.provenance.json](report.provenance.json) に記録し、
一次監査ファイルは変更していない。

```text
summary SHA256: {existing[str(summary_path)]}
audit seal SHA256: {existing[str(seal_path)]}
final manifest SHA256: {summary["resume_history"]["final_manifest_sha256"]}
final staging seal SHA256: {summary["resume_history"]["final_provenance_sha256"]}
```
"""
    with provenance("document_chunked_input_smoke", root, name="report") as record:
        record.param("gpu_used", False)
        record.param("existing_audit_outputs_modified", False)
        for path, digest in existing.items():
            record.input(path, path, sha256=digest)
        snapshots = root / "report_source_snapshot"
        snapshots.mkdir(exist_ok=False)
        for path in (Path(__file__), docs):
            digest = sha256(path)
            target = snapshots / path.name
            shutil.copyfile(path, target)
            if sha256(target) != digest:
                raise ValueError("report snapshot mismatch")
            record.input(str(path), path, sha256=digest)
            record.output("snapshot:" + path.name, target, sha256=digest)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(text)
        record.output("result", output, sha256=sha256(output))
        record.output("lab_documentation", docs, sha256=sha256(docs))
        if any(sha256(path) != digest for path, digest in existing.items()):
            raise ValueError("existing audit output changed during report generation")
    print(
        json.dumps(
            dict(
                result_sha256=sha256(output),
                report_seal_sha256=sha256(report_seal),
                documentation_sha256=sha256(docs),
            )
        )
    )


if __name__ == "__main__":
    main()
