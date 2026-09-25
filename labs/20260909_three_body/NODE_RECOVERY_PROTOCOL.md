新しいアルゴリズムarmとして、元v1の候補を全て保持し、先頭parentの現在の胸ヒレが直接支持され、口が直接支持されないslotに、口だけの再捕捉候補を追加する。入力取得の補助処理や正解ラベルの生成ではない。

`TrackerConfig(mode="detectron2", device="cuda", flow_proposals=True)` のその他の既定値を固定する。beam=6、opt_steps=14、元max_hypotheses=18を変更しない。元候補の後にNodeRecoveryConfig既定の最大2候補×3slotを追加し、実際の候補数は最大24。元候補18を24へ設定変更する操作は行わない。`NodeRecoveryThreeBodyTracker` の候補生成と既検証Graph版 `_fit` を組み合わせた `AcceleratedNodeRecoveryTracker` を使用する。

NodeRecoveryConfig既定: RMS≤1.5 mm、初期の口―胸ヒレ長の0.5–1.5倍、工程上の箱(0,0,-6)–(192,192,186) mm、各slot候補上限2、組合せ上限4096。追加する開始候補では口以外の座標とplanを変えず、他slotの口行は直接支持されていなくても保護する。その後の既存joint fitでは従来どおり全nodeが動き得る。最終的な観測・review判定は元trackerで再計算する。候補planは試行対応であり、観測確定sourceではない。直接胸ヒレ支持、ray一致や候補scoreはID正解を意味しない。

CLIは`--sequence`、`--input-root`、`--output`だけを受け付ける。許す全区間は333519:333719（200 frames）、76620:77286（666 frames）、76620:86580（9960 frames）。全て166fpsの連続native framesで、prefix指定や中間再初期化を許さない。以前に固定した最初の登録3D姿勢を1回だけ使い、外部resetは0。途中の人手GT、旧baseline軌跡、旧v1候補は推論に入力しない。

旧v1出力は推論が完全に戻り新candidateを保存した後に開く。同じ初期seed、全推論入力の解決済みpathとSHA、元TrackerConfigの一致を検証して、同じ共有candidate_metricsで比較する。GTの口注釈と全7 C34 armも推論後だけ評価し、端点未採点を残す。元19配列を全て保存し、累積口再捕捉diagnosticsはsummaryへ別に保存する。新armのsummaryは既存`evaluate_minute.load_completed`の完了・native frames・初期化回数・reset条件と互換にするが、algorithm_armで元v1と区別する。

コードとprotocol、校正、登録seed、原画・mask・背景・C01入力のsnapshot/SHAを記録し、保存終了前に再照合する。出力既存ディレクトリは拒否し、途中失敗の出力は成功sealにしない。元のrun_experiment、tracker、Graphコード、既存candidate、人手JSONを変更しない。runtimeは新armの読込・追跡だけを計測し、D2準備と評価を除く。

実装準備の段階では追跡を実行しない。実行時には各出力を別ディレクトリに保存する。候補の追加、幾何支持の回復や出力率増加だけでID維持成功とはしない。既に開封済みの3区間を使う開発比較であり、未見評価ではない。口以外の欠測、混合mask、参照側のID不一致、少数のメス端点だけでは検証できないオス同士のIDは引き続き限界として残す。
