#!/usr/bin/env python
"""从远程拉取 human-count 推理产物到本地。"""
import pathlib
import sys

import paramiko

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from remote_run import load_env  # noqa: E402

REMOTE = "/home/r4900g3/yhh/human-count/output4/"
LOCAL = "D:/dev/human-count/output_dedup/"

FILES = [
    "clip01_h264.mp4",
    "clip02_h264.mp4",
    "clip01_first_frame.jpg",
    "clip02_first_frame.jpg",
    "clip01_detections.jsonl",
    "clip02_detections.jsonl",
    "clip01_summary.json",
    "clip02_summary.json",
]


def main():
    env = load_env()
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(
        hostname=env["local_server_ip"],
        port=int(env["local_server_port"]),
        username=env["local_server_user"],
        password=env["local_server_pwd"],
        timeout=15,
        look_for_keys=False,
        allow_agent=False,
    )
    sftp = cli.open_sftp()
    pathlib.Path(LOCAL).mkdir(parents=True, exist_ok=True)
    for f in FILES:
        sftp.get(REMOTE + f, LOCAL + f)
        print("got", f)
    sftp.close()
    cli.close()


if __name__ == "__main__":
    main()
