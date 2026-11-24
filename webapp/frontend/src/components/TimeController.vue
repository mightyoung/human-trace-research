<script setup lang="ts">
import { computed, ref } from 'vue'
import { store, seek, togglePlay, stepFrame, selectChain, durationSec, delayOf } from '../store'

const dur = computed(() => durationSec())
const drag = ref(false)
// 默认折叠：桌面 1600×1000 下展开的 6 行甘特把六路每个画面压到 112px 高，
// 折叠后每格约 170px（面积 +130%）。折叠态用一条 16px 的「在场人数」曲线保留概览。
const openTable = ref(false)

function pct(t: number) { return Math.max(0, Math.min(100, (t / dur.value) * 100)) }

function fmt(s: number) {
  s = Math.max(0, s)
  const m = Math.floor(s / 60), ss = Math.floor(s % 60)
  return `${String(m).padStart(2, '0')}:${String(ss).padStart(2, '0')}`
}
const wall = computed(() => {
  const base = 8 * 3600 + 55 * 60
  const sec = base + store.session.min_offset + store.t
  const h = Math.floor(sec / 3600) % 24, m = Math.floor(sec / 60) % 60, s = Math.floor(sec % 60)
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
})

function fromEvent(e: MouseEvent, el: HTMLElement) {
  // 以「轨道」而非「容器」为准：容器还含左侧标签列(118px)+内边距
  const track = (el.querySelector('.ruler-track') || el.querySelector('.row-track')) as HTMLElement | null
  const r = (track || el).getBoundingClientRect()
  const p = Math.max(0, Math.min(1, (e.clientX - r.left) / (r.width || 1)))
  seek(p * dur.value)
}
function onDown(e: MouseEvent) { drag.value = true; fromEvent(e, e.currentTarget as HTMLElement) }
function onMove(e: MouseEvent) { if (drag.value) fromEvent(e, e.currentTarget as HTMLElement) }
function onUp() { drag.value = false }

function barColor(camId: string, eid: string) {
  const cid = store.entChain[`${camId}:${eid}`]
  if (cid && store.chainColor[cid]) return store.chainColor[cid]
  return 'var(--line)'
}
function onBar(camId: string, eid: string) {
  const cid = store.entChain[`${camId}:${eid}`]
  if (cid) selectChain(cid)
}
const liveCount = computed(() =>
  store.cams.reduce((n, c) => n + (store.entSpan[c.id] || []).filter((e: any) => store.t >= e.t0 && store.t <= e.t1).length, 0))

// ---- 折叠态的「在场人数」曲线：把 6 路甘特压成一条，保留「哪段时间人多」的信息 ----
const BINS = 120
const density = computed(() => {
  const bins = new Array(BINS).fill(0)
  const d = dur.value || 1
  for (const c of store.cams) {
    for (const e of (store.entSpan[c.id] || [])) {
      const a = Math.max(0, Math.floor((e.t0 / d) * BINS))
      const b = Math.min(BINS - 1, Math.ceil((e.t1 / d) * BINS))
      for (let i = a; i <= b; i++) bins[i]++
    }
  }
  return bins
})
/** 人数柱：120 根，高度按当前最大值归一（面积图在 800px×20px 的比例下会被看成一条线） */
const densBars = computed(() => {
  const b = density.value
  const max = Math.max(1, ...b)
  return b.map((v, i) => ({ x: i, h: +((v / max) * 18).toFixed(2), v }))
})
const densMax = computed(() => Math.max(1, ...density.value))
/** 曲线上的峰值时刻（用于 hover 提示） */
const densPeak = computed(() => {
  const b = density.value
  let i = 0
  for (let k = 1; k < b.length; k++) if (b[k] > b[i]) i = k
  return { t: (i / BINS) * dur.value, v: b[i] }
})
</script>

<template>
  <div class="tc panel">
    <div class="ctrl row">
      <button class="icon-btn" @click="stepFrame(-1)" title="上一抽样帧" aria-label="上一抽样帧">
        <svg viewBox="0 0 16 16" width="13" height="13" fill="currentColor"><path d="M4 2h1.6v12H4zM13 2v12L6 8z" /></svg>
      </button>
      <button class="primary play" @click="togglePlay()" :aria-label="store.playing ? '暂停' : '播放'">
        <svg v-if="store.playing" viewBox="0 0 16 16" width="12" height="12" fill="currentColor"><path d="M3 2h3.5v12H3zM9.5 2H13v12H9.5z" /></svg>
        <svg v-else viewBox="0 0 16 16" width="12" height="12" fill="currentColor"><path d="M4 2l10 6-10 6z" /></svg>
        <span class="hide-xs">{{ store.playing ? '暂停' : '播放' }}</span>
      </button>
      <button class="icon-btn" @click="stepFrame(1)" title="下一抽样帧" aria-label="下一抽样帧">
        <svg viewBox="0 0 16 16" width="13" height="13" fill="currentColor"><path d="M10.4 2H12v12h-1.6zM3 2v12l7-6z" /></svg>
      </button>

      <div class="time mono">{{ fmt(store.t) }}<span class="dim"> / {{ fmt(dur) }}</span></div>

      <label class="sr-only" for="rate">播放速度</label>
      <select id="rate" v-model.number="store.rate" class="rate">
        <option :value="0.25">0.25×</option><option :value="0.5">0.5×</option>
        <option :value="1">1×</option><option :value="2">2×</option><option :value="4">4×</option>
      </select>

      <span class="tag mono hide-sm" title="按各机位 OSD 对齐后的墙钟时间">{{ wall }}</span>
      <span class="tag hide-sm" :class="liveCount ? 'ok' : ''"><i class="dot" />{{ liveCount }} 个实体在场</span>

      <div class="grow" />
      <span class="tiny dim hide-sm">会话时间轴｜t=0 ↔ 08:54:55</span>
      <button class="pill" :aria-pressed="openTable" @click="openTable = !openTable"
              :title="openTable ? '收起各机位实体甘特条（画面变大）' : '展开 6 行实体甘特条'">
        实体甘特 {{ openTable ? '▴' : '▾' }}
      </button>
    </div>

    <div class="ruler" @mousedown="onDown" @mousemove="onMove" @mouseup="onUp" @mouseleave="onUp"
         role="slider" :aria-valuemin="0" :aria-valuemax="Math.round(dur)"
         :aria-valuenow="Math.round(store.t)" aria-label="会话时间轴" tabindex="0"
         @keydown.left.prevent="stepFrame(-1)" @keydown.right.prevent="stepFrame(1)">
      <div class="row-label" />
      <div class="ruler-track">
        <div class="playhead" :style="{ left: pct(store.t) + '%' }" />
      </div>
    </div>

    <!-- 折叠态概览：6 路甘特压成一条「在场人数」曲线，保留「哪段时间人多」的信息 -->
    <div v-if="!openTable" class="density"
         @mousedown="onDown" @mousemove="onMove" @mouseup="onUp" @mouseleave="onUp">
      <div class="row-label"><span class="tiny dim ellip">在场人数</span></div>
      <div class="ruler-track dens-track" role="slider" tabindex="-1"
           :aria-label="`六路在场实体数曲线，峰值 ${densPeak.v} 人`"
           :title="`六路在场实体数（合并统计）｜峰值 ${densPeak.v} 人 @ ${fmt(densPeak.t)}｜点或拖可定位`">
        <svg viewBox="0 0 120 20" preserveAspectRatio="none" aria-hidden="true">
          <rect v-for="(b, i) in densBars" :key="i" :x="b.x" :y="20 - b.h" width="0.82" :height="b.h"
                class="dens-bar" />
        </svg>
        <div class="playhead" :style="{ left: pct(store.t) + '%' }" />
      </div>
    </div>

    <div v-show="openTable" class="table" @mousedown="onDown" @mousemove="onMove" @mouseup="onUp" @mouseleave="onUp">
      <div v-for="c in store.cams" :key="c.id" class="trow">
        <div class="row-label">
          <b class="mono">{{ c.id }}</b>
          <span class="tiny dim ellip">{{ c.name }}</span>
        </div>
        <div class="row-track">
          <div v-if="delayOf(c.id) > 0" class="pre" :style="{ width: pct(delayOf(c.id)) + '%' }"
               :title="`该机位 t=${delayOf(c.id).toFixed(0)}s 起进入窗口`" />
          <button v-for="(ent, i) in (store.entSpan[c.id] || [])" :key="i"
                  class="ebar" :class="{ hl: store.selChain && store.entChain[c.id + ':' + ent.eid] === store.selChain }"
                  :style="{ left: pct(ent.t0) + '%', width: Math.max(0.4, pct(ent.t1) - pct(ent.t0)) + '%', background: barColor(c.id, ent.eid) }"
                  :title="`${c.id}·${ent.eid} ｜ ${ent.n_frag} 段 ｜ ${ent.t0.toFixed(1)}s → ${ent.t1.toFixed(1)}s`"
                  @click.stop="onBar(c.id, ent.eid)" />
          <div class="playhead" :style="{ left: pct(store.t) + '%' }" />
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.tc { display: flex; flex-direction: column; overflow: hidden; }
.ctrl { padding: var(--sp-2) var(--sp-3); gap: 6px; border-bottom: 1px solid var(--line); }
.play { min-width: 76px; }
.time { font-size: var(--text-md); font-weight: 650; margin-left: var(--sp-1); font-variant-numeric: tabular-nums; white-space: nowrap; flex: none; }
.rate { width: 74px; min-height: 26px; padding: 2px 6px; font-size: var(--text-xs); flex: none; }

.ruler { display: flex; padding: 0 var(--sp-3); cursor: pointer; }
.ruler .row-label { width: 118px; flex: none; }
.ruler-track {
  flex: 1; height: 10px; background: var(--surface-inset); border-radius: var(--r-pill);
  position: relative; margin: 7px 0;
}
.table { cursor: crosshair; padding: 0 var(--sp-3) var(--sp-3); }
.trow { display: flex; align-items: center; height: 18px; }
.row-label { width: 118px; flex: none; display: flex; gap: 6px; align-items: baseline; min-width: 0; }
.row-label b { font-size: var(--text-xs); color: var(--accent); }
.row-track { flex: 1; position: relative; height: 12px; background: var(--surface-inset); border-radius: var(--r-xs); }
.ebar {
  position: absolute; top: 1px; height: 10px; padding: 0; border: 0; border-radius: var(--r-xs);
  opacity: .9; cursor: pointer; min-width: 2px; min-height: 0;
  transition: opacity var(--dur-instant) var(--ease-out-quart);
}
@media (hover: hover) { .ebar:hover { opacity: 1; outline: 1px solid var(--fg); } }
.ebar.hl { opacity: 1; box-shadow: 0 0 0 1px var(--fg) inset; }
.ebar:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.pre {
  position: absolute; top: 0; height: 12px; border-radius: var(--r-xs);
  background: repeating-linear-gradient(45deg, var(--surface-2), var(--surface-2) 4px, var(--surface-inset) 4px, var(--surface-inset) 8px);
}
.playhead {
  position: absolute; top: -4px; bottom: -4px; width: 2px; background: var(--signal);
  box-shadow: 0 0 6px color-mix(in oklch, var(--signal) 70%, transparent);
  pointer-events: none; z-index: 2; border-radius: 1px;
}
.ruler-track .playhead { top: -5px; bottom: -5px; }

/* 折叠态的人数柱：24px 高，替代 6 行甘特；面积图在 800×20 的比例下会被看成一条线，故用柱 */
.density { display: flex; padding: 0 var(--sp-3) var(--sp-2); cursor: pointer; align-items: center; }
.density .row-label { align-items: center; }
.dens-track { height: 24px; margin: 0; border-radius: var(--r-xs); overflow: hidden; }
/* svg 元素必须显式给尺寸：默认是 300×150，会被 24px 的容器裁掉，只剩最高的几根柱露头 */
.dens-track svg { display: block; width: 100%; height: 100%; }
.dens-bar { fill: color-mix(in oklch, var(--accent) 78%, transparent); }
.density .playhead { top: -2px; bottom: -2px; }
@media (max-width: 767px) {
  .ruler .row-label, .row-label { width: 74px; }
  .ruler-track { height: 14px; margin: 9px 0; }
  .trow { height: 22px; }
  .row-track { height: 14px; }
  .ebar { top: 2px; height: 10px; }
  /* 窄屏：控制条单行横向滚动 + 右缘渐隐 */
  .ctrl {
    overflow-x: auto; flex-wrap: nowrap; padding: var(--sp-2);
    padding-right: var(--sp-5);
    mask-image: linear-gradient(90deg, #000 0, #000 calc(100% - 30px), transparent 100%);
  }
  .ctrl::-webkit-scrollbar { height: 0; }
  .play { min-width: 0; }
  .time { font-size: var(--text-base); }
}
@media (max-width: 560px) { .hide-xs { display: none; } }
</style>
