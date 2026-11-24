# 323_train_510 串人修复网格（在仓库根目录运行）
#   powershell -ExecutionPolicy Bypass -File tools\run_switch_grid.ps1
#   powershell -ExecutionPolicy Bypass -File tools\run_switch_grid.ps1 -ATTR 1   # 加"只换跟踪器"归因臂
#   可改：$SEG / $PKG / $OLD / $PY
#
# 注意 1：--det-cache 把"跟踪前去重"烘进缓存，缓存里存的是去重后的框。
#         换 --dedup-mask 必须重建缓存；同一缓存的各配置只变跟踪参数。
# 注意 2：python 往 stderr 写警告（torchreid 的 UserWarning 等）在 PS 5.1 下会被
#         $ErrorActionPreference="Stop" 当成 NativeCommandError 终止脚本。所以原生调用
#         统一走 Run()：临时改成 Continue、只按 $LASTEXITCODE 判失败。
param(
  [string]$SEG = "323_train_510",
  [string]$PKG = "deliverable/data-request-v2",                       # 含 gt\ 与 ledger\
  [string]$OLD = "output_crosscam_local/1012_yolo8_dd/{seg}/{seg}_detections.jsonl",  # 台账来源，勿改成 1012_final
  [string]$REID_DEVICE = "dml",
  [string]$PY = "C:/Users/muyizWindows/.workbuddy/binaries/python/envs/dml_torch/Scripts/python.exe",
  [int]$ATTR = 0                                                      # 1 = 追加归因臂（要再建一份缓存）
)
$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT
if (-not (Test-Path $PY)) { $PY = "python" }
$env:PYTHONPATH = "$ROOT/pylibs"
$env:TMPDIR = "$ROOT/tmp"; $env:TEMP = "$ROOT/tmp"

function Run([string]$exe, [string[]]$argv) {
  $prev = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  & $exe @argv 2>&1 | ForEach-Object { Write-Host "$_" }
  $code = $LASTEXITCODE
  $ErrorActionPreference = $prev
  if ($code -ne 0) { throw "失败（exit $code）：$exe $($argv -join ' ')" }
}

$CACHE = "output_crosscam_local/det_cache/${SEG}_s5_osnet.npz"
$OUT = "output_crosscam_local/1012_yolo_v2"

Write-Host "=== [0/4] 预检：--old 必须是台账那份 jsonl ==="
Run $PY @("tools/eval_switch_fix.py", "--gt", "$PKG/gt/gt_gallery_members_FILLED.csv",
          "--old", $OLD, "--new", "precheck=$OLD", "--segs", $SEG,
          "--out-dir", "output_crosscam_local/switch_eval/_precheck_$SEG")

Write-Host "=== [1/4] 建缓存（检测 CPU + 跟踪前去重 0.9 + OSNet-RGB 特征）==="
if (-not (Test-Path $CACHE)) {
  Run $PY @("tools/yolo_track_det_v2.py", "--seg", $SEG, "--stride", "5", "--reid", "osnet",
            "--reid-device", $REID_DEVICE, "--det-cache", $CACHE, "--track-stride", "10",
            "--tag", "v2_s10", "--out-dir", $OUT, "--force")
} else { Write-Host "缓存已存在：$CACHE" }

Write-Host "=== [2/4] 跑配置网格 ==="
$cfgs = @(
  # 名称            参数（s10 = gallery 口径 2.5FPS；s5 = 5FPS）
  @("oldthr_s10",  "--track-stride 10 --high 0.10 --low 0.05 --new 0.12 --match 0.85 --buffer-sec 24"),
  @("v2_s10",      "--track-stride 10"),
  @("v2_buf1_s10", "--track-stride 10 --buffer-sec 1"),
  @("v2_buf4_s10", "--track-stride 10 --buffer-sec 4"),
  @("v2_m70_s10",  "--track-stride 10 --match 0.70"),
  @("v2_m85_s10",  "--track-stride 10 --match 0.85"),
  @("v2_veto40_s10","--track-stride 10 --veto-sim 0.40"),
  @("v2_veto50_s10","--track-stride 10 --veto-sim 0.50"),
  @("v2_s5",       "--track-stride 5"),
  @("v2_veto40_s5","--track-stride 5 --veto-sim 0.40")
)
$newArgs = @()
foreach ($c in $cfgs) {
  $name = $c[0]
  $argv = @("tools/yolo_track_det_v2.py", "--seg", $SEG, "--det-cache", $CACHE,
            "--out-dir", $OUT, "--tag", $name, "--force") + $c[1].Split(" ")
  Write-Host "--- $name ---"
  Run $PY $argv
  $newArgs += @("--new", "$name=$OUT/$name/{seg}/{seg}_detections.jsonl")
}

if ($ATTR -eq 1) {
  Write-Host "=== [3/4] 归因臂：只换跟踪器（旧阈值 + 跟踪后 box-IoU0.5 + 平滑框）==="
  $CACHE0 = "output_crosscam_local/det_cache/${SEG}_s5_osnet_nodedup.npz"
  if (-not (Test-Path $CACHE0)) {
    Run $PY @("tools/yolo_track_det_v2.py", "--seg", $SEG, "--stride", "5", "--reid", "osnet",
              "--reid-device", $REID_DEVICE, "--dedup-mask", "0", "--det-cache", $CACHE0,
              "--track-stride", "10", "--out-dir", $OUT, "--tag", "attr_raw", "--force")
  }
  Run $PY @("tools/yolo_track_det_v2.py", "--seg", $SEG, "--det-cache", $CACHE0,
            "--box-source", "track", "--track-stride", "10", "--high", "0.10", "--low", "0.05",
            "--new", "0.12", "--match", "0.85", "--buffer-sec", "24",
            "--out-dir", $OUT, "--tag", "attr_raw", "--force")
  Run $PY @("tools/dedup_detections.py", "--in-dir", "$OUT/attr_raw",
            "--out-dir", "$OUT/attr_oldpipe", "--iou", "0.5", "--segs", $SEG)
  $newArgs += @("--new", "attr_trackonly_s10=$OUT/attr_oldpipe/{seg}/{seg}_detections.jsonl")
} else { Write-Host "=== [3/4] 跳过归因臂（-ATTR 1 可开）===" }

Write-Host "=== [4/4] 评估 ==="
Run $PY (@("tools/eval_switch_fix.py", "--gt", "$PKG/gt/gt_gallery_members_FILLED.csv",
           "--ledger", "$PKG/ledger/tracks_all.csv", "--windows", "$PKG/gt/switch_windows.json",
           "--old", $OLD, "--segs", $SEG,
           "--out-dir", "output_crosscam_local/switch_eval/$SEG") + $newArgs)
Write-Host "结果：output_crosscam_local/switch_eval/$SEG/summary.md"
