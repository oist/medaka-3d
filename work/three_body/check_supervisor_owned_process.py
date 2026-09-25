"""Only a newly created CPU sleeper: inspect its own emitted PID and stop its tree."""

import argparse
import ctypes
import hashlib
import json
import subprocess
from ctypes import wintypes
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
output = args.output.resolve()
if output.exists() or not output.is_relative_to(REPO):
    raise ValueError("new workspace result required")

kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel.WaitForSingleObject.restype = wintypes.DWORD
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
kernel.CloseHandle.restype = wintypes.BOOL
script = "import os,json,time; print(json.dumps(dict(pid=os.getpid(),ppid=os.getppid())),flush=True); time.sleep(8)"
info = dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            CPU_sleeper_only=True, existing_processes_enumerated=False, GPU=False,
            natural_exit_upper_seconds=8)
process, handle = None, None
try:
    process = subprocess.Popen([str(REPO / ".venv/Scripts/python.exe"), "-B", "-c", script],
        stdout=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    info.update(json.loads(process.stdout.readline()))
    info["owned_launcher_pid"] = process.pid
    info["launcher_live_before_kill"] = process.poll() is None
    print(json.dumps(info), flush=True)
    handle = kernel.OpenProcess(0x00100000, False, info["pid"])
    info["real_child_handle_opened"] = bool(handle)
    info["real_child_handle_error"] = ctypes.get_last_error() if not handle else 0
    killed = subprocess.run(["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
        capture_output=True, text=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
    info.update(taskkill_returncode=killed.returncode, taskkill_stdout=killed.stdout,
                taskkill_stderr=killed.stderr)
    info["real_child_terminated_immediately"] = (
        kernel.WaitForSingleObject(handle, 1000) == 0 if handle else None)
    print(json.dumps(info), flush=True)
finally:
    if process is not None:
        # If tree control was denied, this fixture exits naturally after eight seconds.
        info["launcher_final_returncode"] = process.wait(timeout=12)
    if handle:
        info["real_child_final_terminated"] = kernel.WaitForSingleObject(handle, 1000) == 0
        kernel.CloseHandle(handle)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(info, stream, indent=2)
    print(json.dumps(info), flush=True)
if info.get("taskkill_returncode") != 0 or info.get("real_child_terminated_immediately") is not True:
    raise SystemExit(1)
