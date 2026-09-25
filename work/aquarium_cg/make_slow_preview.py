"""Retiming only: 332 native source frames played at 30 fps, no interpolation."""
from pathlib import Path
import json
import subprocess

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance

root = Path(__file__).resolve().parent
source = root / "c34_76100_76432_native/aquarium.mp4"
output_dir = root / "c34_75270_78590"
target = output_dir / "teleport_slow_5.53x.mp4"
command = [
    "ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-i", str(source),
    "-vf", "setpts=166/30*PTS", "-r", "30", "-frames:v", "332", "-an",
    "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-threads", "2",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target),
]
with provenance("aquarium_slow_motion", output_dir) as record:
    record.input("native_movie", source, sha256=sha256(source))
    record.input("retiming_script", Path(__file__), sha256=sha256(Path(__file__)))
    record.param("ffmpeg_command", command)
    record.param("source_frames_inclusive", [76100, 76431])
    record.param("source_fps", 166)
    record.param("output_fps", 30)
    record.param("playback_speed", 30 / 166)
    record.param("source_duration_seconds", 2)
    record.param("playback_duration_seconds", 332 / 30)
    record.param("interpolation", "none; all 332 source frames retained")
    record.param("clock", "On-frame clock remains source-relative time (0 to 2 seconds)")
    subprocess.run(command, check=True)
    record.output("slow_movie", target, sha256=sha256(target))
    page = '''<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>テレポート候補のスロー再生</title>
<style>body{max-width:1500px;margin:32px auto;padding:0 20px;background:#0c1422;
color:#e9f1fa;font:16px system-ui}video{width:100%}p{line-height:1.7}a{color:#75cddb}</style>
<h1>テレポート候補 / 5.53倍スロー</h1>
<p>元の2秒を約11.07秒で再生。166 fpsの332フレームをすべて残しています。
画面内の時計は元動画の時間です。<br>
ID 0の14.6 mmの急移動候補は、再生開始から4.5秒付近 (元動画f76235) です。
赤い線は追跡上の急移動候補で、スワップ確定ではありません。</p>
<video controls preload="metadata" src="teleport_slow_5.53x.mp4"></video>
<p><a href="index.html">20秒の通常再生・全候補一覧へ</a></p></html>'''
    page_path = output_dir / "slow.html"
    page_path.write_text(page, encoding="utf-8")
    record.output("slow_player", page_path, sha256=sha256(page_path))
print(json.dumps({"output": str(target), "frames": 332, "seconds": 332 / 30}))
