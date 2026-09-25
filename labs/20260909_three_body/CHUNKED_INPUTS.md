# 再開できる独立マスク入力の準備

`prepare_chunked_source_masks.py` は、原画像ごとの Detectron2 推論を独立した
チャンクに保存する。SLEAP、初期姿勢、個体 ID、人の正解ラベルは読まない。
追跡の実行・再初期化・追跡状態の復元も行わない。

## CLI と API

以下は新しい出力先で最初の 1 チャンクだけを作る例。実行するプロトコルを
明示し、既存の専用 Detectron2 環境を使う。

```powershell
& ..\medaka-mating-pipeline\.venv-detectron2\Scripts\python.exe -B labs/20260909_three_body/prepare_chunked_source_masks.py --protocol work/three_body/chunked_input_smoke_protocol.json --source-cache work/three_body/source_cache --out work/three_body/inputs/new_chunked_smoke --max-new-chunks 1
```

続きは同じコマンドから `--max-new-chunks 1` を外す。Python API は
`run(protocol_path, source_cache, out, max_new_chunks=None)`。
`max_new_chunks` は新規作成数だけを制限し、検証済みチャンクの再利用は数えない。
全体が未完了なら `PreparationPaused` を送出し、CLI は意図した終了コード 2 を返す。
`incomplete.json`、`final_sequence_path: null`、未完了の run provenance が残る。
保存済みチャンクの完了記録は維持する。PowerShell 等の外側の終了コードと
Python の終了コードは区別して記録する。

全 30 分の条件は [long_input_protocol.json](long_input_protocol.json) に登録済み。
49800:348600、166 fps、298800 フレームを 1660 フレームずつの 180 チャンクに分ける。
この文書作成時点では全 30 分も、登録長窓の先頭 10 秒の費用測定も未実行。
プロトコルの詳細は [LONG_INPUT_PROTOCOL.md](LONG_INPUT_PROTOCOL.md) を参照する。

## 保存と再利用

```text
out/
  chunks/f333519-333521/attempt_0001/
  chunks/f333521-333523/attempt_0001/
  runs/run_0001/incomplete.json
  runs/run_0002/summary.json
  assemblies/attempt_0001/manifest.json
```

各 attempt は上書きしない。再利用には、完了した 3 視点、正確な連続フレーム範囲、
プロトコル、入力・モデル・設定・コードの SHA、保存した実行環境、全出力の整合性が必要。
失敗した attempt や条件の異なる attempt は残し、次の番号で新規作成する。
設定やコードを変更した場合、既存チャンクは同じ実験条件として流用しない。

全チャンクが揃ってから、配列の値・順序・型を維持して結合する。
`frame_offsets` と `mask_byte_offsets` だけを累積位置へ直す。
各フレームの全検出を保存し、3 匹という仮定で検出数を制限しない。
ゼロ検出も `view_available=True` の空観測として保持する。

## 完了した入力を読む条件

成功した `runs/run_NNNN/summary.json` の `final_sequence_path` は、`out` に対する
完成 assembly の相対パス。run と assembly の provenance が `status: ok` で、
manifest・chunk index・参照チャンク・各入力出力の SHA が一致して初めて完了と扱う。
チャンクが数個完成していることだけでは、全体の完了にはならない。

最終 manifest は `raw_storage: original_source_video` を持つ
`SourceSequenceReader(sequence_path, sleap_root)` 互換形式。
元動画のフレーム位置、166 fps、1200×1200、各 BGR 画像の SHA を検証して指定範囲だけを読む。
reader 自体は内部検証用の未封印 descriptor も読めるため、利用側が先に完了記録を
検査する必要がある。既存の追跡 CLI へ自動的に組み込まれたことは意味しない。

## 実動画で確認した範囲

333519:333525 の 6 フレーム × 3 視点を、2 フレームずつ 3 チャンクで新規推論した。
最初に 1 チャンクで停止、次に残り 2 チャンクを作成、最後に全チャンクを再利用した。
最初のチャンクと完成 assembly のチェックサムは変わらなかった。

- 18 枚の native BGR 画像が、旧 FFV1 入力とバイト単位で一致。
- 55 検出、全 13 個の packed mask 配列が型・形・全バイトで一致。
- top は毎フレーム 3 検出、side1 は毎フレーム 4 検出、side2 は 3,2,2,2,2,2 検出。
- 背景は 5000,10000,…,45000 の登録済み 9 枚の中央値と全 3 視点で一致。
  旧背景とは全 3 視点で異なる。この変更は別条件として明示している。
- 全入力・コードの監査前後 SHA が一致。動画ペイロードのコピーは 0 バイト。

結果は [監査 summary](../../work/three_body/chunked_source_smoke_qa_v1/summary.json) と
[監査 provenance](../../work/three_body/chunked_source_smoke_qa_v1/audit.provenance.json)。
実装側の CPU テスト 75 件と、独立した追加回帰 3 件も通過している。
この小実窓にはゼロ検出フレームがなかったため、ゼロ検出の保持は CPU テストでの確認。
検出器の一般的な精度や、長時間の個体 ID 維持を示す結果ではない。

## 運用上の注意

原動画は hardlink なので、リンク先と同じファイル本体を共有する。必ず読み取り専用で
扱い、どちらの経路からも書き換えない。動画の再エンコードや複製は行わない。

開始時、各チャンクの封印時、結合時、呼び出し完了時に full SHA を計算するため、
約 15.8 GB の元入力を読む費用がある。短窓の所要時間を D2 推論速度とは呼ばない。
今回の CPU 監査は 54.60 秒、そのうち画像の decode・比較は 1.43 秒だった。
同時実行の影響もあるため、長区間の費用は別に測定する。
