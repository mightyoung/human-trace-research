#!/usr/bin/env bash
# webapp 部署收尾监控 + 端到端验证（远程执行，经 remote_run --stdin-file）
# 逻辑：等 make_proxies 完成（proxy.log 出现 [done] 且 6 路 mp4 就绪）
#   → 逐项验证：health / 视频 Range / 首页与 JS / API 抽样
set -u
W=$HOME/human-count/webapp
M=$W/backend/data/media
LOG=$W/deploy/proxy.log

echo "== 等待 make_proxies 完成（每 10s 检查，上限 10 分钟）=="
for i in $(seq 1 60); do
  if grep -q '\[done\]' "$LOG" 2>/dev/null; then
    n=$(ls "$M"/*.mp4 2>/dev/null | wc -l)
    if [ "$n" -ge 6 ]; then echo "[ok] 代理完成（第 $i 轮检查，$n 路）"; break; fi
  fi
  sleep 10
done

echo "== proxy.log =="
cat "$LOG"
echo "== media 目录 =="
ls -la "$M"
echo "== health =="
curl -s -m 5 http://127.0.0.1:8099/api/health; echo
echo "== 视频 Range 逐路（期望 206）=="
for c in 305 316 318 320 323 343; do
  code=$(curl -s -m 15 -r 0-99 -o /dev/null -w '%{http_code}' "http://127.0.0.1:8099/media/$c.mp4")
  size=$(stat -c%s "$M/$c.mp4" 2>/dev/null || echo 0)
  echo "  $c: HTTP $code / $size bytes"
done
echo "== 首页与静态资源 =="
curl -s -m 5 -o /dev/null -w '  index: %{http_code}\n' http://127.0.0.1:8099/
js=$(curl -s -m 5 http://127.0.0.1:8099/ | grep -o 'assets/index-[^"]*\.js' | head -1)
if [ -n "$js" ]; then
  curl -s -m 5 -o /dev/null -w "  $js: %{http_code}\n" "http://127.0.0.1:8099/$js"
fi
echo "== API 抽样 =="
for ep in cams chains pairs metrics masks mirror media-list; do
  code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:8099/api/$ep")
  echo "  /api/$ep: $code"
done
echo "[WATCH DONE]"
