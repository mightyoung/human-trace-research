# -*- coding: utf-8 -*-
"""耐用性本地文件原语 —— 代际存储（CURRENT 指针模型）的基础设施层。

上游参考：G 包 `core/artifact_io.py`（POSIX 假设）。本版为 **Windows 实测修复版**
（2026-10-15 微实测结论）：
  - `sync_file` 必须用**可写句柄**（'rb+'）：Windows 下对只读 fd 调 os.fsync 抛
    `OSError(9, 'Bad file descriptor')`（FlushFileBuffers 需写访问）——G 包原版
    用 `open(path,'rb')` 在本机直接不可用；
  - `fsync_dir` 在 Windows 为 no-op：`os.open(目录)` 抛 PermissionError（实测），
    NTFS 由元数据日志保证 rename 的最终持久性、`os.replace` 同卷原子；
  - `writer_lock` Windows 分支 = `msvcrt.LK_NBLCK` 重试循环（实测可跨句柄互斥；
    阻塞版 LK_LOCK 约 10s 后直接抛错，无法表达"等待"语义）。

约定：所有写操作以"**完成标志放最后一步 + 目标原子替换**"模式使用本模块；
仅支持单一本地文件系统（同卷 os.replace），NFS / 对象存储 / 多主机分布式锁不适用。
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time

__all__ = [
    "canonical_json", "sha256_file", "fsync_dir", "sync_file",
    "atomic_write_bytes", "atomic_json", "atomic_output", "writer_lock",
]

# writer_lock 的同线程重入注册表：{绝对路径: (线程id, 重入深度)}
_held: dict = {}
_held_guard = threading.Lock()


def canonical_json(value) -> bytes:
    """确定性 JSON 序列化（键排序、无空格、禁 NaN/Inf），用于哈希与落盘。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fsync_dir(path) -> None:
    """尽力刷新目录项到磁盘（POSIX）。Windows：no-op（见模块 docstring 论证）。"""
    if os.name == "nt":
        return
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sync_file(path) -> None:
    """把文件内容刷到磁盘。Windows 必须用可写句柄 'rb+'（见模块 docstring）。"""
    with open(path, "rb+") as f:
        os.fsync(f.fileno())


def atomic_write_bytes(path, blob: bytes) -> None:
    """同目录 tmp → fsync → os.replace → fsync_dir：目标文件永不处于半写状态。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path, value) -> None:
    """原子写 JSON（canonical 序列化）。"""
    atomic_write_bytes(path, canonical_json(value))


@contextmanager
def atomic_output(path):
    """流式原子写：yield 二进制句柄；正常退出时 fsync + 原子替换。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextmanager
def writer_lock(path, timeout: float = 60.0, poll: float = 0.2):
    """跨进程写锁（本地单机；**同线程可重入**）。

    重入语义：管理工具常"持锁 → 调 save()（内部再锁）"；同线程嵌套申请直接放行，
    不同线程/进程照常通过 OS 锁互斥（Windows: msvcrt.LK_NBLCK 轮询，实测跨句柄互斥；
    POSIX: flock(LOCK_EX|LOCK_NB) 轮询）。锁文件保留（等待者打开同一文件）；
    超时 → TimeoutError（fail-loud，不无限挂起）。
    """
    path = Path(path)
    key = os.path.abspath(str(path))
    tid = threading.get_ident()
    with _held_guard:
        held = _held.get(key)
        if held is not None and held[0] == tid:      # 同线程重入
            _held[key] = (tid, held[1] + 1)
            nested = True
        else:
            nested = False
    if nested:
        try:
            yield
        finally:
            with _held_guard:
                t_, depth = _held.get(key, (tid, 1))
                if depth <= 1:
                    _held.pop(key, None)
                else:
                    _held[key] = (t_, depth - 1)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+b")
    locked = False
    try:
        if os.name == "nt":
            import msvcrt
            # 直接锁 [0,1) 区间：空文件也可锁、无需先写字节（写/读锁区会与他句柄冲突，
            # 实测 read(1) 在对方持锁时抛 PermissionError）。失败即时返回（实测 0.000s）。
            f.seek(0)
            t0 = time.monotonic()
            while not locked:
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    locked = True
                except OSError:
                    if time.monotonic() - t0 > timeout:
                        raise TimeoutError(f"writer_lock 等待超时（{timeout}s）：{path}")
                    time.sleep(poll)
        else:
            import fcntl
            t0 = time.monotonic()
            while not locked:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                except OSError:
                    if time.monotonic() - t0 > timeout:
                        raise TimeoutError(f"writer_lock 等待超时（{timeout}s）：{path}")
                    time.sleep(poll)
        with _held_guard:
            _held[key] = (tid, 1)
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    finally:
        with _held_guard:
            _held.pop(key, None)
        f.close()
