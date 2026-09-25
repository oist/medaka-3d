"""Assemble the completed five-minute handoff from successful sealed stages."""
# ruff: noqa: E501, RUF001
from __future__ import annotations

from pathlib import Path

from sample_resumed_nodewise_appearance import WORK, read, sha, write

from medaka3d.provenance import provenance

OUTPUT = WORK / "resumed_nodewise_validation_20260912_v1"
STAGES = {
    "beam": ("resumed_nodewise_five_minute_identity_beam256_v1", "identity.provenance.json"),
    "evaluation": ("resumed_nodewise_beam_female57_comparison_v1", "evaluation.provenance.json"),
    "risks": ("resumed_nodewise_risk_review_v1", "risk.provenance.json"),
    "videos": ("resumed_nodewise_blind_review_v1", "review.provenance.json"),
}


def run():
    assert not OUTPUT.exists(), "fresh final report required"
    OUTPUT.mkdir()
    with provenance("resumed_nodewise_five_minute_validation_handoff_v1", OUTPUT, name="validation") as rec:
        bindings = {}
        for role, (directory, filename) in STAGES.items():
            path = WORK / directory / filename
            seal = read(path)
            assert seal["status"] == "ok", f"unfinished {role}"
            for name, output in seal["outputs"].items():
                member = Path(output.get("path", WORK / directory / name))
                assert sha(member) == output["sha256"], f"sealed output changed: {member}"
            bindings[role] = dict(path=str(path), sha256=sha(path), stage=seal["stage"])
            rec.input(role, path, sha256=sha(path))
        for name in (
            "resumed_nodewise_five_minute_identity_beam256_independent_qa_v1.json",
            "resumed_nodewise_beam_female57_comparison_independent_qa_v1.json",
            "resumed_nodewise_risk_review_independent_qa_v1.json",
        ):
            rec.input(name, WORK / name, sha256=sha(WORK / name))
        for name in ("audit.json", "audit.provenance.json"):
            path = WORK / "resumed_nodewise_blind_review_output_qa_v1" / name
            assert read(path)["status"] == "ok"
            rec.input("video_qa_" + name, path, sha256=sha(path))
        rec.input("source", Path(__file__), sha256=sha(__file__))
        beam = read(WORK / STAGES["beam"][0] / "summary.json")
        risk = read(WORK / STAGES["risks"][0] / "risk_summary.json")
        scenes = read(WORK / STAGES["risks"][0] / "scenes.json")["scenes"]
        videos_root = WORK / STAGES["videos"][0]
        video_summaries = [read(videos_root / s["scene_id"] / "summary.json") for s in scenes]
        for selected, rendered in zip(scenes, video_summaries, strict=True):
            assert rendered["status"] == "ok"
            assert rendered["native_interval"] == [selected["start_frame"], selected["stop_frame_exclusive"]]
            assert rendered["video_native_frames"] == selected["stop_frame_exclusive"] - selected["start_frame"]
            assert rendered["video_qa"]["all_frames_decoded"]
            assert rendered["model_outputs_shown"] is False
            assert rendered["human_truth_created"] is False
        evaluation = read(WORK / STAGES["evaluation"][0] / "summary.json")
        for method in ("nodewise_raw", "nodewise_beam"):
            for mode in ("raw_slots", "review_gated"):
                result = evaluation["methods"][method][mode]
                assert result["available_anchors"] == 57
                assert result["interval_counts"] == dict(preserved=56, swap=0, unscored=0)
        assert beam["solve_calls"] == 1 and beam["raw_beam_disagreement_frames"] == 0
        assert risk["all_high_risk_events_covered"]
        status = dict(
            schema="resumed_nodewise_five_minute_validation_handoff_v1",
            status="requested_computations_evaluation_and_videos_complete",
            interval=[66400, 116200], native_frames=49800, source_fps=166,
            raw_tracking_rerun=False, thirty_minute_tracking_started=False,
            inputs=bindings, raw_candidate_sha256=beam["raw_candidate_reference"]["candidate"]["sha256"],
            solve_calls=1, solve_seconds=beam["solve_seconds"], raw_beam_disagreement_frames=0,
            female_raw_and_beam=dict(available_anchors=57, preserved=56, swap=0, unscored=0),
            female_sparse=dict(available_anchors=57, preserved=55, swap=1, unscored=0),
            high_risk_scenes=risk["high_risk_scene_count"], low_risk_scenes=risk["random_low_risk_scene_count"],
            native_clip_frames=sum(s["video_native_frames"] for s in video_summaries),
            native_clip_seconds=risk["total_clip_seconds"],
            slow_playback_seconds=sum(s["playback_seconds"] for s in video_summaries),
            verified_raw_camera_frames=sum(s["native_bgr_hashes_verified"] for s in video_summaries),
            promotion_status="not_promoted", accepted_id_fraction=1/3,
            secondary_review_or_beam_ambiguity_accepted_cells=49798,
            secondary_review_or_beam_ambiguity_accepted_fraction=49798/149400,
            pending_human_review=dict(
                scene_count=len(scenes), anonymous_animal_probe_count=9*len(scenes),
                animal_continuity_edges=6*len(scenes), scored_edges=0,
                unscored_edges=6*len(scenes), unscored_fraction=1.0,
                confirmed_identity_failure_rate=None,
                sex_group_scores={key:dict(status="unscored_pending_human_correspondence", failure_rate=None)
                                  for key in ("female", "male_male", "female_male")},
                priority_scenes=["scene_005", "scene_009", "scene_013"],
            ),
        )
        write(OUTPUT / "summary.json", status)
        text = f"""# Nodewise 5分検証：探索・比較評価・確認動画の完了記録

2026-09-12。対象は録画 **06:40–11:40**、フレーム `[66400,116200)` の49,800フレーム。
今回依頼された全区間ID探索、同じメス57点の比較評価、オスを含む3カメラ確認動画を作成した。
既存追跡を再利用し、追跡・検出の再実行と30分追跡は行っていない。

**確認動画は[一覧](../resumed_nodewise_blind_review_v1/index.html)から開く。**
少数から確認する場合は [scene_005](../resumed_nodewise_blind_review_v1/scene_005/index.html)、
[scene_009](../resumed_nodewise_blind_review_v1/scene_009/index.html)、
[scene_013](../resumed_nodewise_blind_review_v1/scene_013/index.html)を先に確認できる。
これら3場面の結果だけで全21場面を合格とはしない。

| 同じメス57点・隣接56区間 | 維持 | スワップ区間 | 未評価 |
|---|---:|---:|---:|
| 凍結Sparse・raw | 55 | 1 | 0 |
| resumed Nodewise・raw | 56 | 0 | 0 |
| resumed Nodewise・beam | 56 | 0 | 0 |

Nodewiseはreview-gatedでも57点・維持56・スワップ0・未評価0。
Sparseはreview-gatedで20点、維持19・スワップ0・未評価37となる。
Sparseのraw変化区間は既報と同じ `82121–83265`。
これは注釈端点間の評価であり、区間内の入れ替わりと復帰の不在やオス同士の継続性は証明しない。

ID beamは既存幅256・胸/口の同一目的関数・資源上限・初期P012を保持した。
resumed producerの成功seal、restore_count=5、復元前後の設定と候補SHAを検証する新callerで、
全49,800フレームを**1回、CPU {beam['solve_seconds']:.3f}秒**で探索した。
rawとbeamの割当差は**0フレーム**。候補rawと並べ替え後streamは別ファイルに保存した。
650,081保持状態・3,900,450生成辺で固定上限内。GPUによる追跡の再実行は不要だった。

条件付き同点は6フレーム（`77652,77653,82719,82720,90142,90146`）。
枝刈りされたグラフ上の近似探索なので、gapは確率・大域最適性・実個体IDの証明ではない。
6フレームはすべて優先3場面の動画に含まれる。

| 支持・受理の補助指標 | 凍結Sparse | Nodewise raw / beam |
|---|---:|---:|
| 全ノードの観測支持率 | 98.3115% | 98.4248% |
| 元のreviewフラグによるID受理率 | 10.9277% | 33.3333% |
| 同じ112 frame/viewの口再投影誤差・平均 | 2.788 px | 2.760 px |
| 同じ112 frame/viewの口再投影誤差・中央値 | 2.095 px | 2.099 px |
| 同じ112 frame/viewの口再投影誤差・P95 | 7.763 px | 7.770 px |

観測率は総量で改善した。再投影誤差は平均が改善する一方、中央値・P95は僅かに増加したため、
すべての誤差指標で「悪化なし」とは認定しない。比較の支持集合と凍結屈折補正calibrationは同じ。

確認動画は**高リスク18場面＋seed固定の低リスク3場面**。
元映像 {risk['total_clip_seconds']:.3f}秒、全{status['native_clip_frames']:,} nativeフレームを20 fpsで保持し、
合計{status['slow_playback_seconds']:.2f}秒（約9分11秒）の低速動画にした。
各動画は上面・側面1・側面2を同期表示し、手法名、追跡ID、予測軌跡を重ねていない。
入口・中間・出口のPNGと、4フレーム間隔の原寸1200×1200静止画も付属する。

選定には、方式間の幾何対応変化、raw/beam不一致、低いmargin/gap、口・胸の支持欠落、
直接観測点の急な移動、2カメラでの外見変化、近接距離を使った。距離単独では高リスクにしなかった。
選定planはbeam結果を開く前にSHAで固定し、メス注釈を選定や調整には使っていない。
以前のraw診断とメスの既報結果は既知だったので、この場面選定をheld-out検証とは呼ばない。
外見は0.5秒ごとの画素色proxyであり、照明・姿勢・投影の影響がある。
18高リスク場面は全選定イベントを網羅し、raw/beam不一致区間は存在しなかった。

**3個体IDの自動採用は未達。** オス2匹のreviewフラグは全49,800フレームで残る。
元フラグの受理率は魚別 `[0%,100%,0%]`、全体33.33%で、90%全体・80%各魚の条件を満たさない。
同点・unknownをさらに除外する保守的な受理率は33.3320%（メスの2フレームを追加除外）で、別QAに記録した。
新しい人手確認は未実施で、21場面×3匹×2区間＝126区間をすべて未評価として保持した。
メス、オス–オス、メス–オスの接近失敗率は人手対応確定まで未評価であり、0件と認定しない。

確認画面では各場面の入口で3匹を任意のA/B/Cとして選び、中間・出口まで同じ魚を追う。
上面・側面の口位置をクリックし、「同じ魚」「曖昧」「見えない」を記録して判断JSONを保存する。
判断の保留は未評価のまま残る。生物学的な名前は不要。
人手回答は、凍結した各方式を採点する新しいレビュー評価として保存する。
回答を接続制約としてIDを補正する場合だけ別のassisted出力とし、今回の自動出力・スコアを上書きしない。

検証記録：beamの60テスト、評価の3テストと独立再計算、riskの3テストと103窓構成ケースの独立監査。
動画は元BGR {status['verified_raw_camera_frames']:,}枚のハッシュ一致、全動画のフレーム数・fps・再デコードと
3時点の原画像との画素平均絶対差8/255未満を確認した。HTML内JavaScriptの構文を検証した。
ブラウザのfile URLポリシーにより画面の実操作検証はできていない。
視覚点検は映像・配置のQAであり、人手ID正解には加えていない。

保存先：

- [ID探索のplan / provenance](../resumed_nodewise_five_minute_identity_beam256_v1/plan.json)
- [メス57点の全評価・CSV案内](../resumed_nodewise_beam_female57_comparison_v1/RESULT.md)
- [固定した場面選定plan](../resumed_nodewise_risk_review_v1/selection_plan.json)
- [危険場面ledger](../resumed_nodewise_risk_review_v1/risk_ledger.json)
- [3カメラ確認動画一覧](../resumed_nodewise_blind_review_v1/index.html)
- [機械可読の完了・未評価状態](summary.json)

Raw候補SHA-256: `{status['raw_candidate_sha256']}`。
"""
        (OUTPUT / "RESULT.md").write_text(text, encoding="utf-8")
        for name in ("summary.json", "RESULT.md"):
            rec.output(name, OUTPUT / name, sha256=sha(OUTPUT / name))
        for record in bindings.values():
            assert sha(record["path"]) == record["sha256"]
        print(OUTPUT / "RESULT.md")


if __name__ == "__main__":
    run()
