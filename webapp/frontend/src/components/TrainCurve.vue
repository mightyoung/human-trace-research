<script setup lang="ts">
/** 训练曲线：val_f1（实线）+ val_acc（虚线），可选 train loss 细带。
 *  纯内联 SVG，无外部依赖；纵轴固定 0~1，跨变体可直接比形状。 */
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  series?: Record<string, { step: number; value: number }[]>
  bestEp?: number | null
  height?: number
  showLoss?: boolean
}>(), { series: () => ({}), bestEp: null, height: 230, showLoss: false })

const PAD = { l: 34, r: 12, t: 12, b: 22 }
const W = 660

const chart = computed(() => {
  const f1 = (props.series?.val_f1 || []) as { step: number; value: number }[]
  const acc = (props.series?.val_acc || []) as { step: number; value: number }[]
  if (!f1.length && !acc.length) return null

  const steps = [...f1, ...acc].map(p => p.step)
  const x0 = Math.min(...steps)
  const x1 = Math.max(...steps)
  const iw = W - PAD.l - PAD.r
  const ih = props.height - PAD.t - PAD.b
  const sx = (s: number) => PAD.l + (x1 === x0 ? iw / 2 : ((s - x0) / (x1 - x0)) * iw)
  const sy = (v: number) => PAD.t + ih * (1 - Math.max(0, Math.min(1, v)))

  const mk = (arr: { step: number; value: number }[]) =>
    arr.filter(p => p.value != null).map(p => ({ x: sx(p.step), y: sy(p.value), s: p.step, v: p.value }))
  const line = (pts: { x: number; y: number }[]) =>
    pts.map((p, i) => `${i ? 'L' : 'M'}${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ')

  const gap = Math.max(1, Math.ceil((x1 - x0) / 8))
  const tickX: number[] = []
  for (let s = x0; s <= x1; s += gap) tickX.push(s)

  const f1pts = mk(f1)
  return {
    f1line: line(f1pts), accline: line(mk(acc)), f1pts,
    ticksY: [0, 0.25, 0.5, 0.75, 1].map(t => ({ y: sy(t), label: t.toFixed(2).replace('0.', '.') })),
    tickX: tickX.map(s => ({ x: sx(s), label: String(s) })),
    best: props.bestEp ? { x: sx(props.bestEp), label: `best ep${props.bestEp}` } : null,
    lastPoint: f1pts.length ? f1pts[f1pts.length - 1] : null,
  }
})

const lossStrip = computed(() => {
  if (!props.showLoss) return null
  const arr = (props.series?.loss || []) as { step: number; value: number }[]
  if (arr.length < 2) return null
  const vals = arr.map(p => p.value)
  const lo = Math.min(...vals), hi = Math.max(...vals)
  const w = 660, h = 54, pad = 4
  const sx = (i: number) => pad + (i / (arr.length - 1)) * (w - pad * 2)
  const sy = (v: number) => pad + (h - pad * 2) * (1 - (v - lo) / Math.max(1e-6, hi - lo))
  const d = arr.map((p, i) => `${i ? 'L' : 'M'}${sx(i).toFixed(1)} ${sy(p.value).toFixed(1)}`).join(' ')
  return { d, lo: lo.toFixed(3), hi: hi.toFixed(3), w, h }
})

const f4 = (v: any) => (typeof v === 'number' ? v.toFixed(3) : '—')
const lastF1 = computed(() => {
  const a = (props.series?.val_f1 || []) as any[]
  return a.length ? a[a.length - 1].value : null
})
const lastAcc = computed(() => {
  const a = (props.series?.val_acc || []) as any[]
  return a.length ? a[a.length - 1].value : null
})
</script>

<template>
  <div class="curve">
    <div class="legend">
      <span><i class="sw f1" />val_f1 <b class="mono">{{ f4(lastF1) }}</b></span>
      <span><i class="sw acc" />val_acc <b class="mono dim">{{ f4(lastAcc) }}</b></span>
      <span v-if="chart?.best" class="tiny dim">{{ chart.best.label }}</span>
    </div>

    <div v-if="!chart" class="empty tiny">没有 epoch 级曲线（日志缺失或用例过短）</div>
    <template v-else>
      <svg class="chart" :viewBox="`0 0 ${W} ${height}`" :style="{ height: height + 'px' }"
           preserveAspectRatio="none" role="img" aria-label="训练曲线">
        <line v-for="t in chart.ticksY" :key="'gy' + t.label" :x1="PAD.l" :x2="W - PAD.r"
              :y1="t.y" :y2="t.y" class="grid" />
        <text v-for="t in chart.ticksY" :key="'ty' + t.label" :x="PAD.l - 6" :y="t.y + 3"
              class="axis" text-anchor="end">{{ t.label }}</text>
        <text v-for="t in chart.tickX" :key="'tx' + t.label" :x="t.x" :y="height - 6"
              class="axis" text-anchor="middle">{{ t.label }}</text>
        <g v-if="chart.best">
          <line :x1="chart.best.x" :x2="chart.best.x" :y1="PAD.t" :y2="height - PAD.b" class="bestline" />
          <text :x="chart.best.x + 4" :y="PAD.t + 9" class="bestlabel">{{ chart.best.label }}</text>
        </g>
        <path :d="chart.accline" class="accline" />
        <path :d="chart.f1line" class="f1line" />
        <circle v-if="chart.lastPoint" :cx="chart.lastPoint.x" :cy="chart.lastPoint.y" r="3.2" class="dot" />
      </svg>

      <div v-if="lossStrip" class="loss">
        <div class="loss-h tiny dim">
          train loss
          <span class="mono">{{ lossStrip.lo }} → {{ lossStrip.hi }}</span>
        </div>
        <svg :viewBox="`0 0 ${lossStrip.w} ${lossStrip.h}`" preserveAspectRatio="none" class="losssvg">
          <path :d="lossStrip.d" class="lossline" />
        </svg>
      </div>
    </template>
  </div>
</template>

<style scoped>
.curve { min-width: 0; }
.legend { display: flex; align-items: center; gap: var(--sp-3); font-size: var(--text-xs);
          color: var(--fg-2); margin-bottom: 6px; }
.legend b { font-weight: 600; }
.sw { width: 12px; height: 2px; display: inline-block; border-radius: 2px; margin-right: 4px;
      vertical-align: middle; }
.sw.f1 { background: var(--accent); }
.sw.acc { background: var(--fg-3); }
.chart { width: 100%; display: block; }
.chart .grid { stroke: var(--line); stroke-width: 1; }
.chart .axis { fill: var(--fg-3); font-size: 9px; font-family: var(--font-mono); }
.chart .bestline { stroke: var(--warn); stroke-width: 1; stroke-dasharray: 3 3; opacity: .6; }
.chart .bestlabel { fill: var(--warn); font-size: 9px; font-family: var(--font-mono); opacity: .9; }
.chart .accline { fill: none; stroke: var(--fg-3); stroke-width: 1.4; stroke-dasharray: 4 3; }
.chart .f1line { fill: none; stroke: var(--accent); stroke-width: 2; }
.chart .dot { fill: var(--accent); }
.loss { margin-top: var(--sp-2); border-top: 1px dashed var(--line); padding-top: 6px; }
.loss-h { display: flex; justify-content: space-between; margin-bottom: 2px; }
.losssvg { width: 100%; height: 54px; display: block; }
.lossline { fill: none; stroke: var(--fg-3); stroke-width: 1.2; }
</style>
