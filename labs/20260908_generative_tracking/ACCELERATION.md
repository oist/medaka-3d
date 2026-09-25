# 加速の独立検証（2026-09-08）

RTX 5070 Ti を使う画像前処理と、同じ目的関数を保つ投影・Jacobian計算の整理を
別々に測定した。以下は短区間の部品検証であり、1分間の総処理速度やID精度の実証ではない。
長区間の速度は各runの `comparison.json: tracking_fps` を用いる。

## GPU画像前処理

既存 `.venv` の PyTorch `2.13.0+cu130` / CUDA 13.0 で実CUDAカーネルの動作を確認。
GPUは RTX 5070 Ti 16 GB。OpenCV 5.0.0 のCUDAデバイス数は0だったため、
`accelerated_observations.py` はRGB差分・整数ヒストグラムによる照明差補正・チャネル最大値を
GPUで処理し、連結成分・距離変換・Farnebackフローを3カメラ並列のCPU処理とした。
GPU不可時の暗黙CPUフォールバックはない。ドライバや学習済みモデルの追加は不要だった。

実画像 `76620:76628`、ウォームアップ後の前処理測定：

| 処理 | フローの最大辺 | 中央値 秒/フレーム | 前処理 fps |
|---|---:|---:|---:|
| CPU逐次 | 640 | 0.941 | 1.06 |
| CPU 3カメラ並列 | 640 | 0.354 | 2.82 |
| GPU差分＋CPU並列 | 640 | 0.313 | 3.21 |
| CPU 3カメラ並列 | 320 | 0.186 | 5.40 |
| GPU差分＋CPU並列 | 320 | 0.131 | 7.57 |

画像読込・背景準備を除外し、GPU転送・同期を含む。計測プロセスはcProfile下にある。
fpsは8フレームの合計時間から算出し、表の時間は中央値。主な改善はCPUのカメラ並列化で、
GPUのみの効果と混同しない。CUDA区間中央値は約0.026秒だった。

同じ640設定では、CPU逐次・CPU並列・GPUの全画像マップが実8フレームで完全一致し、
逐次CPUとGPUの追跡出力も完全一致した。320設定でもCPU/GPUのマップは完全一致。
一方、**640→320は観測設定の変更**であり、同等な実装最適化ではない。
実SLEAP点213箇所のフロー変位差は元画像座標で中央値0.494 px、95百分位1.478 px、
最大2.143 px。この8フレームの選択軌跡・source・reviewは変わらなかったが、
一般的な追跡結果の不変性は保証しない。元の166 fpsから時間方向の間引きは行わない。

実行・数値QA・来歴：
[`profile_image_acceleration.py`](profile_image_acceleration.py)、
`work/generative_tracking/acceleration_76620-76628/acceleration.json`。
`tests/test_accelerated_observations.py` のGPUを実使用する8テストも通過した。

## スレッド調査と投影計算

NumPy/SciPyのOpenBLAS各24スレッド、torch CPU 16、OpenCV 32を確認。
`threadpoolctl==3.6.0` のみ依存変更なしでローカルvenvに追加し、スコープを限定して比較した。
固定CUDA320画像を用いた8フレームのfit計測は既定4.079秒、BLAS 1/2スレッドとも約4.004秒、
BLAS1＋torch1＋OpenCV1は4.082秒。全出力差0だが改善は約2%で、主因ではなかった。
この数値にもcProfileの負荷がある。総16フレーム8.157秒中、屈折投影の累積時間は5.233秒、
`_fit` は5.600秒、画像スコアは2.339秒だった。
記録：`work/generative_tracking/fit_threads_76620-76628/`。

`fast_fit.py` は各残差行が1匹の9座標だけに依存する構造を使い、有限差分を
28→10バッチに圧縮する。全27座標を同時に解くソルバー、目的関数、停止条件は維持。
さらにノード・体軸サンプルを同じカメラへの1回の投影にまとめる。
画像スコアでも現在姿勢・体軸・前姿勢の3群を1回にまとめた。

保存済みの実画像 `76620:76623`、他の追跡プロセス終了後、cProfileなし、
同じ事前計算済みCUDA320画像を使った交互3反復の中央値：

| ReacquisitionTracker内の計算 | 3フレーム 秒 | fitのみ fps |
|---|---:|---:|
| 元のfitter＋元の画像スコア | 0.877 | 3.42 |
| 元のfitter＋投影をまとめた画像スコア | 0.827 | 3.63 |
| FastFit＋投影をまとめた画像スコア | 0.539 | 5.57 |

最後の条件は約38.5%短縮。全追跡の座標差最大 `1.67e-11 mm`、source・review・
appearance・quality差0。画像スコア単体60回は0.567→0.442秒、約22%短縮し、
投影・スコア・テンプレート・特徴・品質値は全て差0だった。
独立したGenerativeTracker比較でもFastFitは0.868→0.590秒、約32%短縮。
Jacobian差0、個別fitのコスト差最大 `3.20e-14`、source・observed・review差0だった。
これらは有限の検証例での数値的一致であり、全映像のビット単位一致は主張しない。
欠測・境界付近・画像・無効投影点を含む `tests/test_fast_fit.py` の7テストが通過した。

正しい比較の記録は以下。各ディレクトリにコードスナップショット・入力ハッシュ・来歴を保存した。

- `work/generative_tracking/batched_projection_76620-76623_standalone_explicit_fit/`
- `work/generative_tracking/fast_fit_76620-76623_standalone_batched/`
- 再実行用画像：`work/generative_tracking/fast_fit_76620-76623_contended/raw_frames.npz`

初期測定には二つの制約・訂正がある。`fast_fit_76620-76623_contended` は
他の2プロセスと競合中かつJacobian圧縮のみの測定で、約10%短縮は参考値とした。
`batched_projection_76620-76623_standalone` では実行時の親クラスが既にFastFitを使用しており、
`frozen` と名付けた追跡行を元のfitterとの比較には使えなかった。
`interpretation_correction.json` を保存し、fitterを明示した上記比較で置き換えた。
当該記録を削除・上書きしていない。GPU同期の仕様は
[PyTorch公式文書](https://docs.pytorch.org/docs/main/generated/torch.cuda.synchronize.html)、
スレッド制御は[threadpoolctl公式](https://github.com/joblib/threadpoolctl)を参照した。
