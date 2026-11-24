#!/usr/bin/env python
"""SFTP 上传诊断：对比 SSH exec 与 SFTP 的目录视角。"""
import pathlib
import sys

import paramiko

ROOT = pathlib.Path(__file__).resolve().parent.parent
env = {}
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and ":" in line:
        k, v = line.split(":", 1)
        env[k.strip()] = v.strip()

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(
    hostname=env["local_server_ip"],
    port=int(env.get("local_server_port", "22")),
    username=env["local_server_user"],
    password=env["local_server_pwd"],
    timeout=15, look_for_keys=False, allow_agent=False,
)

print("=== SSH exec: 检查目录 ===")
for cmd in [
    "pwd; whoami",
    "ls -la /home/r4900g3/yhh/human-count/crosscam/ 2>&1",
    "ls -la /home/r4900g3/yhh/human-count/crosscam/in/ 2>&1",
]:
    _, out, err = cli.exec_command(cmd, timeout=30)
    print(f"$ {cmd}")
    print(out.read().decode("utf-8", "replace"))
    e = err.read().decode("utf-8", "replace")
    if e.strip():
        print("[stderr]", e)

print("=== SFTP: 视角检查 ===")
sftp = cli.open_sftp()
try:
    print("sftp.getcwd() =", sftp.getcwd())
except Exception as ex:
    print("getcwd error:", ex)
try:
    print("listdir('/home/r4900g3/yhh/human-count/crosscam/') =",
          sftp.listdir("/home/r4900g3/yhh/human-count/crosscam/"))
except Exception as ex:
    print("listdir crosscam error:", type(ex).__name__, ex)
try:
    print("stat('/home/r4900g3/yhh/human-count') OK")
    sftp.stat("/home/r4900g3/yhh/human-count")
except Exception as ex:
    print("stat home error:", type(ex).__name__, ex)

print("=== SFTP: 尝试写入 /tmp 测试文件 ===")
try:
    with sftp.file("/tmp/_wb_test.txt", "wb") as f:
        f.write(b"hello")
    print("write /tmp/_wb_test.txt OK")
except Exception as ex:
    print("write /tmp FAILED:", type(ex).__name__, ex)

print("=== SFTP: 尝试写入 crosscam/in 测试文件 ===")
try:
    with sftp.file("/home/r4900g3/yhh/human-count/crosscam/in/_wb_test.txt", "wb") as f:
        f.write(b"hello")
    print("write crosscam/in/_wb_test.txt OK")
except Exception as ex:
    print("write crosscam/in FAILED:", type(ex).__name__, ex)

sftp.close()
cli.close()
