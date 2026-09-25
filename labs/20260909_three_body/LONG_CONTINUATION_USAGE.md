# 長時間入力の継続実行

`run_long_continuation.py` は、登録した49800の初期姿勢から、完成した入力チャンクの末尾まで
同じ追跡状態で進む。対象全体は `[49800,348600)`、166 fps、298800フレームである。
独立D2の保存単位は1660フレーム（10秒）。ファイル境界で姿勢を初期化しない。

初回の実データ試行は `[49800,51460)` の1660フレームに限定した。全30分の完走や、
3匹の生物学的IDの正解確認を表す試行ではない。実行条件は
[long_tracking_protocol.json](long_tracking_protocol.json) に追跡前に固定した。

## 実行済み初回試行のコマンド

リポジトリ直下のPowerShellから、既存成果物を上書きしない新しい出力先を指定する。
以下の出力先は初回実行の記録であり、同名で再実行しない。

```powershell
& .venv/Scripts/python.exe -B labs/20260909_three_body/run_long_continuation.py `
  --seed-root work/three_body/long_seed_registration_49800_v1 `
  --input-stage-root work/three_body/inputs/long_source_masks_49800-348600_v1 `
  --through-frame 51460 `
  --output work/three_body/long_continuation_first1660_v1
```

前処理は専用のDetectron2環境、追跡は本リポジトリのCUDA対応環境を使う。
新しいパッケージのインストールは必要ない。初期値登録・SL・屈折校正・元動画・D2モデル・
背景規則・保存済みマスクと実装のSHAを検査するため、最初の進捗表示まで時間がかかる。

## 後続区間へ進む場合

次の10秒分の独立マスクだけを準備するコマンドは次のとおり。既に検査に通るチャンクは
再利用し、次の未完了チャンクを新しいattemptへ保存する。1回の制限で全180チャンクが
そろわなければ、親処理は `PreparationPaused` / `incomplete` と終了コード1を返す。
個々の成功チャンクのsealとは意味が異なる。

```powershell
& ../medaka-mating-pipeline/.venv-detectron2/Scripts/python.exe -B `
  labs/20260909_three_body/prepare_chunked_source_masks.py `
  --protocol labs/20260909_three_body/long_input_protocol.json `
  --source-cache work/three_body/source_cache `
  --out work/three_body/inputs/long_source_masks_49800-348600_v1 `
  --max-new-chunks 1
```

初回の成功sealと次のチャンク `[51460,53120)` がそろった場合に、次を実行できる。
同じ出力先が存在する場合は別名を使う。検証が通る前にこのコマンドを繰り返さない。

```powershell
& .venv/Scripts/python.exe -B labs/20260909_three_body/run_long_continuation.py `
  --seed-root work/three_body/long_seed_registration_49800_v1 `
  --input-stage-root work/three_body/inputs/long_source_masks_49800-348600_v1 `
  --through-frame 53120 `
  --resume work/three_body/long_continuation_first1660_v1 `
  --output work/three_body/long_continuation_through53120_v1
```

`--through-frame` は登録済みチャンクの終端だけを受理する。将来のチャンクが増えても
固定入力bindingは同じである。親が消費したチャンクのseal・内容・順序が変わると拒否する。
失敗した親や、保存された候補座標だけからの再開は受け付けない。

## 出力の読み方

- `continuation.provenance.json`: フェーズ全体の成功・失敗と入力、実装、出力のSHA。
- `checkpoint/`: 全候補の親履歴・累積コスト・速度・欠測と保留の状態・直前画像を保存。
- `candidate.npz`: 現在の終端で選んだ21配列。後続映像によって過去部分も変わり得る。
- `consumed_chunks.json`: 今回までの入力証拠。復元時は以前のprefixとの完全一致が必要。
- `summary.json`: 追跡時間、観測と受理の数、初期化・復元回数、暫定結果かどうか。
- `progress.jsonl`: 処理済みのnativeフレームと経過時間。

`full_sequence_complete` は348600まで完走した場合だけtrueとなる。
`prefix_backtrace_provisional` がtrueの結果では、勝った経路だけを次の初期値に流用しない。
復元は全候補を含むcheckpointから行う。履歴の保存・検証・復元費用は処理済み時間とともに増える。

`accepted_pectoral_count` は条件付きの出力可能数であり、正しいIDの正解数ではない。
IDが曖昧になった場面は、[原画確認の手順](../../work/three_body/minute_reviews/REVIEW_INSTRUCTIONS.md)
で得る実際の人手記録と照合する。固定された背景9枚は旧1分実験の背景とは異なるため、
この新しい長区間と旧1分の値だけからアルゴリズムの改善率を計算しない。
