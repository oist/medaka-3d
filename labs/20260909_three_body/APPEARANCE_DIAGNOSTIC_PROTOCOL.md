# 独立instance内の色：保存slotを条件とした診断

実データ計算前の条件固定、2026-09-09。対象は既に開封済みの
v1_interval_detectron2（76620:77286、666 native frames）と同じD2 staging。
モデル・閾値・追跡・ID対応は変更しない。保存slotは仮ラベルで、実魚の正解ではない。

テンプレートは76620:76786の166枚、検証は76786,76796,...,77276の固定50時点。
500枚連続の追跡評価ではない。全3view・3slotで要求した全時点を分母に含め、
未観測、view unavailable、zero mask、非一意、他魚と共有、core不足を別集計する。
review中のslotも除外せず、reviewフラグを別保存する。

各viewでmouthとpectoralが直接支持され、両投影点が同一D2 instance内にある場合を
調べる。座標はnative pixelへ四捨五入（NumPy rint）し、距離の許容幅を探索しない。
両点を含むmaskが複数なら使わず、他のどのlatent slotのmouth/pectを含んでも使わない。
未観測の競合も残す。競合の投影が不明なら一意性を保証できず、当該viewの抽出を保留。
これは幾何に条件づけたmask対応であり、mask自体の独立推論とは区別する。

maskを5×5の楕円kernelでerode（半径2px、画像外は0）し、12画素以上のcoreを要求する。
色は元のBGR uint8を255で除したb,g,r、HSVのh（ラジアン）とs,v（0–1）から、
次の6成分を使う。全画像histogramや背景色をテンプレートに混ぜない。

```
c = max(b + g + r, 1e-8)
descriptor = [median(r/c), median(g/c),
              (1 + mean(s*cos(h)))/2, (1 + mean(s*sin(h)))/2,
              median(s), median(v)]
```

view・仮slotごとに有効な学習標本5枚以上で各成分の中央値をテンプレートとする。
スケール学習は行わず、全6成分のEuclidean距離で最も近いテンプレートを記録する。
3テンプレートが揃わない、特徴が欠測、最短距離が1e-12以内の同点なら未採点とする。
距離閾値・色空間・erode半径・特徴数の事後探索や採用判断は行わない。

保存物は特徴配列、テンプレート間距離、学習内距離、検証の自己slot距離と他slot距離、
3×4混同行列（最終列は未採点）、原画像・mask・candidate・校正・コードSHA、ROI例。
同じ仮slotへ分類された割合は条件付き整合性であり、ID正解率ではない。全666frameから
確定した保存trajectoryのslotを使うため、ラベル側も独立した将来予測検証ではない。
真の色変化と保存slotの入替は、このデータだけでは区別できない。

旧20260908 observations._appearanceは予測3ノードのtubeと背景差分foregroundが重なる
画素のRGB8bin×3のhistogramだった。今回の測定は独立D2 instanceのeroded coreと6成分
なので、旧histogramの数値や距離との直接比較・改善量の主張はしない。両方とも反射や
誤maskを完全に除外できるわけではない。実魚IDに使えるかは独立した人の対応で別検証が必要。
