#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""带 HTTP Range 的极简静态服务器——给 switch_review.html 用。

为什么不能用 `python -m http.server`：它不支持 Range 请求，浏览器拿到 200 全量响应后
无法 seek 到指定秒数（复核页要在 1GB 视频里跳到第 1055 秒）。表现为"点了没反应/从头播"。

用法：
  PY tools/serve_range.py --dir tmp/m1_review --port 8903
然后浏览器打开 http://127.0.0.1:8903/switch_review.html
"""
import argparse
import os
import re
import http.server
import socketserver


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    def send_head(self):
        rng = self.headers.get('Range')
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().send_head()
        m = re.match(r'bytes=(\d*)-(\d*)', rng)
        if not m:
            return super().send_head()
        size = os.path.getsize(path)
        start = int(m.group(1)) if m.group(1) else 0
        end = int(m.group(2)) if m.group(2) else size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_error(416)
            return None
        f = open(path, 'rb')
        f.seek(start)
        self.send_response(206)
        ctype = self.guess_type(path)
        self.send_header('Content-Type', ctype)
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Range', 'bytes %d-%d/%d' % (start, end, size))
        self.send_header('Content-Length', str(end - start + 1))
        self.end_headers()
        return _Slice(f, end - start + 1)

    def log_message(self, fmt, *args):
        pass


class _Slice:
    """只读到指定长度就停的包装，避免把 1GB 一次性读进内存。"""

    def __init__(self, f, remaining):
        self.f = f
        self.remaining = remaining

    def read(self, n=-1):
        if self.remaining <= 0:
            return b''
        if n is None or n < 0 or n > self.remaining:
            n = self.remaining
        data = self.f.read(n)
        self.remaining -= len(data)
        return data

    def close(self):
        self.f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--port', type=int, default=8903)
    a = ap.parse_args()
    os.chdir(a.dir)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(('127.0.0.1', a.port), RangeHandler) as srv:
        print('serving %s at http://127.0.0.1:%d/  (Range enabled)' % (os.getcwd(), a.port), flush=True)
        srv.serve_forever()


if __name__ == '__main__':
    main()
