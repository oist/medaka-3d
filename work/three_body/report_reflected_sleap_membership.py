"""Seal a bounded, post-measurement review without rendering or raw-video reads."""

# ruff: noqa: E501, RUF001
# Preserve the Japanese report's prose and Markdown paragraphs verbatim.

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from medaka3d.provenance import provenance

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "work/three_body"
MEASURE = BASE / "reflected_sleap_membership_13frames_v2"
OLD = BASE / "reflected_sleap_membership_13frames_v1"
OUT = MEASURE / "report_v1"
VIEWS = ("top_cam", "side1_cam", "side2_cam")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    if OUT.exists():
        raise ValueError("report output already exists")
    tracked = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        value = digest(path)
        if expected is not None and value != expected:
            raise ValueError(f"SHA mismatch: {path}")
        if str(path) in tracked and tracked[str(path)] != value:
            raise ValueError(f"input changed: {path}")
        tracked[str(path)] = value
        return path

    def read(path):
        return json.loads(bind(path).read_text(encoding="utf-8"))

    bind(__file__)
    measurement = read(MEASURE / "measurement.provenance.json")
    registration = read(MEASURE / "registration.provenance.json")
    plan = read(MEASURE / "plan.json")
    summary = read(MEASURE / "summary.json")
    old_measurement = read(OLD / "measurement.provenance.json")
    old_plan = read(OLD / "plan.json")
    assert measurement["status"] == registration["status"] == "ok"
    assert measurement["stage"] == "reflected_sleap_membership"
    assert registration["stage"] == "register_reflected_sleap_membership"
    assert registration["finished_at"] <= measurement["started_at"]
    assert summary["plan"] == plan
    assert old_plan["frames"] == plan["frames"]
    assert not plan["fit"] and not plan["GT"] and not plan["raw_image_decode"]
    assert old_measurement["status"] == "failed"
    assert "Rig.rays() takes 2 positional arguments but 3 were given" in old_measurement["error"]
    assert not (OLD / "summary.json").exists()
    expected_inputs = dict(plan["input_and_source_hashes"])
    expected_inputs[str(MEASURE / "plan.json")] = digest(MEASURE / "plan.json")
    expected_inputs[str(MEASURE / "registration.provenance.json")] = digest(
        MEASURE / "registration.provenance.json"
    )
    assert {k: v["sha256"] for k, v in measurement["inputs"].items()} == expected_inputs
    assert {k: v["sha256"] for k, v in registration["inputs"].items()} == plan[
        "input_and_source_hashes"
    ]
    for seal in (registration, measurement):
        for entry in seal["outputs"].values():
            path = bind(entry["path"], entry["sha256"])
            assert path.stat().st_size == entry["size"]
    for path, expected in plan["input_and_source_hashes"].items():
        if Path(path).suffix == ".py":
            bind(path, expected)
    relative = Path("work/three_body/measure_reflected_sleap_membership.py")
    old_source = bind(OLD / "code_snapshot" / relative).read_text(encoding="utf-8")
    new_source = bind(MEASURE / "code_snapshot" / relative).read_text(encoding="utf-8")
    assert old_source.replace(
        "reflected_sleap_membership_13frames_v1", "reflected_sleap_membership_13frames_v2"
    ).replace("rig.rays(view, flat[finite])", "camera.rays(flat[finite])") == new_source

    candidate_path = BASE / "persistent_mouth_continuous_one_minute_v1/candidate.npz"
    bind(candidate_path, expected_inputs[str(candidate_path)])
    with np.load(candidate_path, allow_pickle=False) as archive:
        frames = archive["frame_idx"]
        sources = archive["sources"]
        poses = archive["pose_xyz"]
    lookup = {int(frame): i for i, frame in enumerate(frames)}
    rows = summary["records"]
    assert summary["finite_SLEAP_nodes"] == len(rows) == 301
    assert len({(r["frame_idx"], r["view"], r["sleap_row"], r["node"]) for r in rows}) == 301
    assert sorted({r["frame_idx"] for r in rows}) == plan["frames"]
    selected_count = 0
    for row in rows:
        assert np.isfinite(row["uv"]).all() and row["node"] in (0, 1, 2)
        pixel = np.rint(row["uv"]).astype(np.int64)
        assert row["image_pixel_valid"] == bool(np.all((pixel >= 0) & (pixel < 1200)))
        t, v, n = lookup[row["frame_idx"]], VIEWS.index(row["view"]), row["node"]
        owners = np.flatnonzero(sources[t, :, v, n] == row["sleap_row"])
        assert owners.tolist() == [item["slot"] for item in row["selected"]]
        assert len(owners) <= 1
        for item in row["selected"]:
            np.testing.assert_array_equal(poses[t, item["slot"], n], item["pose_xyz"])
        selected_count += len(owners)
    assert selected_count == 218
    primary = [r for r in rows if r["frame_idx"] == 79381 and r["view"] == "top_cam"]
    assert len(primary) == 9
    assert all(3 not in r["D2_instances"] for r in primary)
    assert summary["primary_reflection_mask_SLEAP"] == []
    checks = dict(
        stage="independent_reflected_sleap_membership_report", status="verified",
        frames=plan["frames"], finite_observations=301, selected_observations=218,
        primary_frame=79381, primary_view="top_cam", primary_mask_index=3,
        primary_finite_observations=9, primary_mask_memberships=0,
        primary_records=primary, candidate_sha256=digest(candidate_path),
        measurement_outputs_rehashed=len(measurement["outputs"]),
        source_candidate_and_metadata_guards=True, selected_source_and_pose_exact=True,
        membership_scope="Saved membership counts and pixel rounding checked; mask membership "
        "implementation reviewed, not recomputed from masks in this report.",
        distance_scope="Source formula checked as clipped Euclidean distance on unit rays; "
        "no independent ray projection or rendering in this report.",
        raw_video_rehashed_or_decoded=False, GT_read=False, fit=False,
        v1_failed_before_observation_records=True,
        v1_v2_source_delta="Output directory and Rig.rays -> Camera.rays API repair only.",
    )
    text = """主対象の 79381・top・D2 マスク m3 には、保存済みの SLEAP 点が **0 点**でした。固定 13 フレーム・3 視点の有限観測は **301 点**、既存追跡で選ばれた観測は **218 点**です。これは選択された2Dノード観測数であり、魚数やID正解数ではありません。

79381 の top には有限点が9点あり、SLEAP row0 の口・胸は m2、尾はどの D2 マスクにも入りません。row1 の口は m0 と m1 の両方、胸・尾は m1、row2 の3点は m0 に入ります。重複所属も保存されています。m3 に SLEAP 点が無い結果は、まず反射候補マスクに対する画像モデルを検討する根拠になります。この主対象で SLEAP の反射点補正が必要だとは示していません。

所属判定はネイティブ画素の座標を `np.rint` で丸め、全 D2 マスクを調べたものです。マスク境界の丸め、マスクの漏れ・重複、未検出は残り、反射・実魚・個体IDの正解注釈ではありません。保存 source と同じ視点・同じ意味ノードの row を照合し、選択218観測の slot・3D座標が元 candidate と一致することを独立に確認しました。

光路距離は単位方向ベクトル上の射影を有限区間 `[0,L]` に切り詰めたユークリッド距離です。新しい三角測量や適合ではありません。壁 y=192/196 と床 z=0 は固定 nominal box の仮説で、床位置の独立実測はありません。工学上 z<0 の保存姿勢との不一致を、物理的に不可能な魚やID誤りの証拠にしていません。

初回 v1 は `Rig.rays(view, uv)` の API 誤用で、観測記録を追加する前に失敗しました。失敗 seal と source snapshot は保存済みです。v2 のソース差分は出力先と `camera.rays(uv)` への修正だけで、13フレームと測定条件は同じです。

今回の独立監査では、事前登録→計測の順序、入力ダイジェストの結合、全26測定出力と保存・現行ソースの SHA、元 candidate の SHA、301記録の一意性、218選択観測との対応を確認しました。原動画の全再ハッシュ・再読込、マスクからの所属再計算、再描画、fit、GT参照は行っていません。判定実装と距離式は読み取りで確認しました。

根拠: [測定結果](../summary.json)、[測定 seal](../measurement.provenance.json)、[固定計画](../plan.json)、[独立チェック](checks.json)、[この報告の seal](report.provenance.json)。
"""
    OUT.mkdir()
    with provenance("independent_reflected_sleap_membership_report", OUT, name="report") as rec:
        for path, value in tracked.items():
            rec.input(path, path, sha256=value)
        snapshot = OUT / "report_source.py"
        shutil.copyfile(__file__, snapshot)
        (OUT / "checks.json").write_text(json.dumps(checks, indent=2, allow_nan=False), encoding="utf-8")
        (OUT / "RESULT.md").write_text(text, encoding="utf-8")
        for name in ("report_source.py", "checks.json", "RESULT.md"):
            rec.output(name, OUT / name, sha256=digest(OUT / name))
        for path, value in tracked.items():
            assert digest(path) == value, f"input changed: {path}"
        for item in rec.outputs.values():
            assert digest(item["path"]) == item["sha256"]
    print(json.dumps({"output": str(OUT), "status": "ok", "finite": 301, "selected": 218}))


if __name__ == "__main__":
    main()
