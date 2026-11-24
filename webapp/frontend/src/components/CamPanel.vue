<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { store, tracksAt, trackColor, statusColor, trajOf, delayOf, selectTrack, journeyOf, geomPredict, gidOf, ownerAt, suspectOf, CamMeta } from '../store'
import { api } from '../api'

const props = defineProps<{ cam: CamMeta; compact?: boolean }>()
const wrap = ref<HTMLElement>()
const video = ref<HTMLVideoElement>()
const canvas = ref<HTMLCanvasElement>()
const stageStyle = ref({ width: '0px', height: '0px' })
let ro: ResizeObserver | null = null
let raf = 0
let last = 0

const started = computed(() => store.t >= delayOf(props.cam.id))
const single = computed(() => store.camFilter === props.cam.id)
const activeCount = computed(() => tracksAt(props.cam.id, store.t).length)
/** 去重视图：本机位当前「承担责任」的人数（= owner 恰好落在本机位的 gperson 数） */
const ownedHere = computed(() => {
  if (!store.dedup) return 0
  return tracksAt(props.cam.id, store.t).filter(d => {
    const g = gidOf(props.cam.id, d.tid)
    if (g == null) return false
    const o = ownerAt(g, store.t)
    return !!o && o.cam === props.cam.id
  }).length
})

function fit() {
  const el = wrap.value
  if (!el) return
  const cw = el.clientWidth, ch = el.clientHeight
  const a = props.cam.w / props.cam.h
  let w = cw, h = cw / a
  if (h > ch) { h = ch; w = ch * a }
  stageStyle.value = { width: w + 'px', height: h + 'px' }
  const cv = canvas.value
  if (cv) {
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr)
  }
}

onMounted(() => {
  if (video.value) store.videoEls[props.cam.id] = video.value
  ro = new ResizeObserver(fit)
  if (wrap.value) ro.observe(wrap.value)
  fit()
  const loop = () => {
    const now = performance.now()
    if (now - last > 33) { last = now; draw() }
    raf = requestAnimationFrame(loop)
  }
  raf = requestAnimationFrame(loop)
})
onUnmounted(() => { ro?.disconnect(); cancelAnimationFrame(raf); delete store.videoEls[props.cam.id] })

function onMeta() {
  store.videoReady[props.cam.id] = true
  fit()
  const el = store.videoEls[props.cam.id]
  if (!el) return
  const d = store.t - delayOf(props.cam.id)
  if (d < 0) return
  // 首帧兜底：currentTime 已经是 0 时再赋 0 是 no-op，浏览器可能不解码、面板保持黑屏
  // （35min 素材体积大时尤其明显）。推一帧（0.04s）强制走一次 seek 出画面。
  const dur = el.duration || 0
  const target = Math.min(d, Math.max(0, dur - 0.05))
  el.currentTime = target < 0.02 ? 0.04 : target
}

function draw() {
  const cv = canvas.value
  if (!cv) return
  const ctx = cv.getContext('2d')
  if (!ctx) return
  const W = cv.width, H = cv.height
  ctx.clearRect(0, 0, W, H)
  if (!started.value) return
  const c = props.cam
  const sx = W / c.w, sy = H / c.h
  const dets = tracksAt(c.id, store.t)
  const sel = store.sel
  const chainSel = store.selChain

  // 0) 选中目标的「跨镜头接力轨迹」——在所有出现过的机位上画出它的整段轨迹
  const jr = sel ? journeyOf(sel.cam, sel.tid) : null
  if (jr) {
    const node = jr.nodes.find(n => n.cam === c.id)
    if (node) {
      for (const t of node.tids) {
        const pts = trajOf(c.id, t)
        if (pts.length < 2) continue
        ctx.beginPath()
        pts.forEach((p, i) => {
          const x = ((p[1] + p[3]) / 2) * sx, y = p[4] * sy
          i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)
        })
        ctx.strokeStyle = jr.color
        ctx.globalAlpha = 0.95
        ctx.lineWidth = 2.6
        ctx.setLineDash([])
        ctx.stroke()
        // 起止端点
        const first = pts[0], last = pts[pts.length - 1]
        for (const [p, fill] of [[first, 'transparent'], [last, jr.color]] as any) {
          ctx.beginPath()
          ctx.arc(((p[1] + p[3]) / 2) * sx, p[4] * sy, 3.4, 0, 6.284)
          ctx.fillStyle = fill; ctx.strokeStyle = jr.color; ctx.lineWidth = 1.4
          if (fill !== 'transparent') ctx.fill(); ctx.stroke()
        }
        ctx.globalAlpha = 1
      }
      // 该机位上的活动片段落点
      const act = node.tids.filter((t: number) => dets.some(d => d.tid === t))
      if (act.length) {
        ctx.setLineDash([])
      }
    }
  }

  // 1) 长程全轨迹（整段生命期，淡显）
  if (store.showFullLife) {
    for (const d of dets) {
      const pts = trajOf(c.id, d.tid)
      if (pts.length < 2) continue
      ctx.beginPath()
      pts.forEach((p, i) => {
        const x = ((p[1] + p[3]) / 2) * sx, y = p[4] * sy
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)
      })
      ctx.strokeStyle = trackColor(c.id, d.tid)
      ctx.globalAlpha = chainSel && !isHl(c.id, d.tid) ? 0.12 : 0.3
      ctx.lineWidth = 1.5
      ctx.setLineDash([])
      ctx.stroke()
      ctx.globalAlpha = 1
    }
  }

  // 2) 短程尾迹
  if (store.trail > 0) {
    const lookback = store.trail * store.stride / c.fps
    for (const d of dets) {
      const pts = trajOf(c.id, d.tid)
      if (pts.length < 2) continue
      const t0 = store.t - delayOf(c.id) - lookback
      const seg = pts.filter(p => p[0] / c.fps >= t0 && p[0] / c.fps <= store.t - delayOf(c.id) + 0.1)
      if (seg.length < 2) continue
      ctx.beginPath()
      seg.forEach((p, i) => {
        const x = ((p[1] + p[3]) / 2) * sx, y = p[4] * sy
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)
      })
      const hl = isHl(c.id, d.tid)
      ctx.strokeStyle = trackColor(c.id, d.tid)
      ctx.globalAlpha = hl ? 0.95 : (chainSel ? 0.25 : 0.6)
      ctx.lineWidth = hl ? 2.4 : 1.6
      ctx.setLineDash([])
      ctx.stroke()
      ctx.globalAlpha = 1
    }
  }

  // 3) 轨迹预测（几何外推，虚线 + 预测落脚点）
  if (store.showPrediction) {
    for (const d of dets) {
      const pr = geomPredict(c.id, d.tid, store.predHorizon)
      if (!pr || pr.speed < 6) continue
      const hl = isHl(c.id, d.tid)
      const col = trackColor(c.id, d.tid)
      ctx.setLineDash([5, 4])
      ctx.globalAlpha = hl ? 0.95 : (chainSel ? 0.3 : 0.7)
      ctx.strokeStyle = col
      ctx.lineWidth = hl ? 2.4 : 1.7
      ctx.beginPath()
      ctx.moveTo(pr.anchor[0] * W, pr.anchor[1] * H)
      for (const p of pr.pts) ctx.lineTo(p[0] * W, p[1] * H)
      ctx.stroke()
      ctx.setLineDash([])
      const end = pr.pts[pr.pts.length - 1]
      if (end) {
        ctx.beginPath()
        ctx.arc(end[0] * W, end[1] * H, 4.5, 0, 6.284)
        ctx.fillStyle = 'rgba(255,255,255,0.15)'
        ctx.fill(); ctx.strokeStyle = col; ctx.lineWidth = 1.8; ctx.stroke()
      }
      ctx.globalAlpha = 1
    }
  }

  // 4) boxes
  if (store.showBoxes) {
    for (const d of dets) {
      const hl = isHl(c.id, d.tid)
      const isSel = sel && sel.cam === c.id && sel.tid === d.tid
      const x = d.x1 * sx, y = d.y1 * sy, w = (d.x2 - d.x1) * sx, h = (d.y2 - d.y1) * sy
      const col = trackColor(c.id, d.tid)
      const sg = suspectOf(c.id, d.tid)      // ★ 疑似同人（换号）：虚线框
      ctx.globalAlpha = chainSel && !hl ? 0.3 : 1
      ctx.lineWidth = isSel ? 3 : hl ? 2.6 : 1.8
      ctx.strokeStyle = isSel ? '#ffffff' : col
      ctx.setLineDash(sg ? [7, 4] : [])
      ctx.strokeRect(x, y, w, h)
      ctx.setLineDash([])
      if (isSel) { ctx.globalAlpha = 0.16; ctx.fillStyle = col; ctx.fillRect(x, y, w, h) }
      ctx.globalAlpha = chainSel && !hl ? 0.3 : 1
      // foot point
      ctx.fillStyle = col
      ctx.beginPath(); ctx.arc((d.x1 + d.x2) / 2 * sx, d.y2 * sy, 3, 0, 6.284); ctx.fill()
      if (store.showIds) {
        const lab = labelOf(c.id, d.tid) + (sg ? (sg.grade === 'strong' ? ' ≈' : ' ?') : '')
        ctx.font = '600 13px ui-monospace, monospace'
        const tw = ctx.measureText(lab).width
        const ly = y - 17 < 0 ? y : y - 17   // 贴近上边缘时改为画在框内，避免遮挡 OSD
        ctx.globalAlpha = 0.85
        ctx.fillStyle = col
        ctx.fillRect(x, ly, tw + 10, 17)
        ctx.globalAlpha = 1
        ctx.fillStyle = '#07121f'
        ctx.fillText(lab, x + 5, ly + 12)
      }
      ctx.globalAlpha = 1
    }
  }

  // 5) masks
  if (store.showMasks) {
    const regs = store.masks?.cameras?.[c.id]
    if (regs?.enabled) {
      for (const poly of regs.regions || []) {
        ctx.beginPath()
        poly.forEach((p: number[], i: number) => {
          const x = p[0] * W, y = p[1] * H
          i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)
        })
        ctx.closePath()
        ctx.fillStyle = 'rgba(255,92,114,0.16)'
        ctx.strokeStyle = 'rgba(255,92,114,0.85)'
        ctx.lineWidth = 1.6
        ctx.fill(); ctx.stroke()
      }
    }
    // 用户新增的玻璃圈选
    const gm = store.labels?.glass_masks?.[c.id]
    if (gm?.length) {
      ctx.beginPath()
      gm.forEach((p: number[], i: number) => {
        const x = p[0] * W, y = p[1] * H
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)
      })
      ctx.closePath()
      ctx.fillStyle = 'rgba(78,161,255,0.15)'
      ctx.strokeStyle = 'rgba(78,161,255,0.9)'
      ctx.setLineDash([5, 4]); ctx.lineWidth = 1.6
      ctx.fill(); ctx.stroke(); ctx.setLineDash([])
    }
  }
}

/** 当前机位是否承载了选中目标的接力链节点 */
const journeyHere = computed(() => {
  const sel = store.sel
  if (!sel || !store.showJourney) return null
  const jr = journeyOf(sel.cam, sel.tid)
  if (!jr) return null
  const node = jr.nodes.find(n => n.cam === props.cam.id)
  return node ? { cid: jr.cid, color: jr.color, node } : null
})

function isHl(cam: string, tid: number): boolean {
  if (!store.selChain) return true
  return store.chainOf[`${cam}:${tid}`] === store.selChain
}

function labelOf(cam: string, tid: number): string {
  // ★ 去重视图 / 按人着色：显示全局人编号（◆ = 该时刻多机位同时可见，已按效用择一）
  if (store.dedup || store.colorBy === 'gperson') {
    const g = gidOf(cam, tid)
    if (g != null) {
      const o = ownerAt(g, store.t)
      return `g${String(g).padStart(4, '0')}${o && o.n_cams > 1 ? ' ◆' : ''}`
    }
    return `#${tid}`
  }
  const eid = store.trackEnt[`${cam}:${tid}`]
  if (store.colorBy === 'entity' && eid) return eid.replace(`${cam}-`, '')
  if (store.colorBy === 'chain') {
    const cid = store.chainOf[`${cam}:${tid}`]
    if (cid) return `${cid}·${tid}`
  }
  return `#${tid}`
}

function onClick(e: MouseEvent) {
  const cv = canvas.value
  if (!cv) return
  const r = cv.getBoundingClientRect()
  const px = (e.clientX - r.left) / r.width * props.cam.w
  const py = (e.clientY - r.top) / r.height * props.cam.h
  const dets = tracksAt(props.cam.id, store.t)
  let hit = null, best = Infinity
  for (const d of dets) {
    if (px >= d.x1 && px <= d.x2 && py >= d.y1 && py <= d.y2) {
      const area = (d.x2 - d.x1) * (d.y2 - d.y1)
      if (area < best) { best = area; hit = d }
    }
  }
  if (hit) selectTrack(props.cam.id, hit.tid)
  else selectTrack(null as any, null)
}
</script>

<template>
  <div class="cam" :class="{ sel: store.sel?.cam === cam.id, single }">
    <div class="cam-h">
      <span class="cid mono">{{ cam.id }}</span>
      <span class="cname">{{ cam.name }}</span>
      <span v-if="single" class="tiny dim show-single">{{ cam.role }}</span>
      <span v-else class="tiny dim hide-md">{{ cam.w }}×{{ cam.h }}</span>
      <div class="grow" />
      <span v-if="journeyHere" class="tag" :style="{ color: journeyHere.color, borderColor: journeyHere.color }"
            :title="`接力链 ${journeyHere.cid} 在本机位的时段`">
        <i class="dot" />{{ journeyHere.cid }} · {{ journeyHere.node.t0.toFixed(0) }}–{{ journeyHere.node.t1.toFixed(0) }}s
      </span>
      <span class="tag" :class="started ? (activeCount || !store.dedup ? 'ok' : '') : 'warn'">
        <i class="dot" />{{ started ? activeCount + ' 人' : `t≥${delayOf(cam.id).toFixed(0)}s 进入` }}
      </span>
      <span v-if="store.dedup && started" class="tag own" title="本机位当前负责的目标数（去重后唯一 owner）">
        <i class="dot" />负责 {{ ownedHere }}
      </span>
    </div>
    <div class="cam-body" ref="wrap">
      <div class="stage" :style="stageStyle">
        <video ref="video" :src="api.mediaUrl(cam.id)" muted playsinline preload="auto"
               @loadedmetadata="onMeta"></video>
        <canvas ref="canvas" @click="onClick" aria-label="检测框叠加层，点击选择目标"></canvas>
        <div v-if="!store.videoReady[cam.id]" class="ph">
          <span class="spinner" /><span>加载视频…</span>
        </div>
        <div v-else-if="!started" class="ph dim">
          该机位在会话 t={{ delayOf(cam.id).toFixed(1) }}s 之后才进入
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.cam {
  display: flex; flex-direction: column; min-height: 0; min-width: 0;
  background: var(--stage); border: 1px solid var(--stage-line);
  border-radius: var(--r-sm); overflow: hidden;
  transition: border-color var(--dur-fast) var(--ease-out-quart),
              box-shadow var(--dur-fast) var(--ease-out-quart);
}
.cam.sel { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent-line); }
.cam-h {
  display: flex; align-items: center; gap: var(--sp-2); padding: 4px var(--sp-2); flex: none;
  background: var(--surface-1); border-bottom: 1px solid var(--line);
  font-size: var(--text-xs); min-height: 28px;
}
.cam.single .cam-h { min-height: 34px; padding: 5px var(--sp-3); }
.cid { font-weight: 700; color: var(--accent); font-size: var(--text-sm); letter-spacing: -0.01em; }
.cname { color: var(--fg-2); }
.cam-body { flex: 1; min-height: 0; display: flex; align-items: center; justify-content: center; }
.stage { position: relative; }
.stage video { width: 100%; height: 100%; display: block; background: #000; }
.stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; cursor: crosshair; }
.ph {
  position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; gap: var(--sp-2);
  background: var(--stage); font-size: var(--text-xs); color: var(--fg-2); text-align: center; padding: var(--sp-3);
}
.hide-md, .show-single { display: none; }
@media (max-width: 767px) { .hide-md { display: none; } }
@media (min-width: 768px) and (max-width: 1439px) { .hide-md { display: none; } }
.show-single { display: inline; }
.tag.own { color: var(--accent); border-color: var(--accent-line); background: var(--accent-soft); }
</style>

