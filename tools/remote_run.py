#!/usr/bin/env python
"""极简 SSH 远程执行器。

用法:
    python tools/remote_run.py "shell command"
    python tools/remote_run.py --stdin-file local.sh
    python tools/remote_run.py --put local_path remote_path
    python tools/remote_run.py --get remote_path local_path

凭据从项目根目录 .env 读取（local_server_*）。
"""
import argparse
import os
import pathlib
import sys

import paramiko

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_env():
    env = {}
    p = ROOT / ".env"
    if not p.exists():
        sys.exit(f".env not found at {p}")
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        env[k.strip()] = v.strip()
    return env


def connect(env):
    host = env["local_server_ip"]
    port = int(env.get("local_server_port", "22"))
    user = env["local_server_user"]
    pwd = env["local_server_pwd"]
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(
        hostname=host,
        port=port,
        username=user,
        password=pwd,
        timeout=15,
        banner_timeout=20,
        auth_timeout=20,
        look_for_keys=False,
        allow_agent=False,
    )
    return cli


def run(cli, cmd, timeout=600):
    stdin, stdout, stderr = cli.exec_command(cmd, timeout=timeout, get_pty=False)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", help="remote shell command")
    ap.add_argument("--stdin-file", help="run local script file content on remote via bash")
    ap.add_argument("--put", nargs=2, metavar=("LOCAL", "REMOTE"))
    ap.add_argument("--get", nargs=2, metavar=("REMOTE", "LOCAL"))
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    env = load_env()
    cli = connect(env)
    try:
        if args.put:
            local, remote = args.put
            sftp = cli.open_sftp()
            sftp.put(local, remote)
            sftp.close()
            print(f"[put] {local} -> {remote}")
            return
        if args.get:
            remote, local = args.get
            sftp = cli.open_sftp()
            sftp.get(remote, local)
            sftp.close()
            print(f"[get] {remote} -> {local}")
            return

        cmd = args.cmd
        if args.stdin_file:
            cmd = pathlib.Path(args.stdin_file).read_text(encoding="utf-8")
        if not cmd:
            sys.exit("no command given")

        code, out, err = run(cli, cmd, timeout=args.timeout)
        if out:
            print(out, end="" if out.endswith("\n") else "\n")
        if err:
            print("--- STDERR ---", file=sys.stderr)
            print(err, end="" if err.endswith("\n") else "\n", file=sys.stderr)
        print(f"--- exit code: {code} ---")
        sys.exit(code)
    finally:
        cli.close()


if __name__ == "__main__":
    main()
