<script setup lang="ts">
import { computed } from 'vue'
import { store } from '../store'

interface Mini { cam: string; name: string; pts: number[][]; color: string; active: boolean }

const minis = computed<Mini[]>(() => {
  const out: Mini[] = []
  const pairs: { cam: string; tid: number }[] = []
  if (store.selChain) {
    for (const key of Object.keys(store.entChain)) {
      if (store.entChain[key] !== store.selChain) continue
      const [cam, eid] = key.split(':')
      for (const tid of (store.entities[cam]?.entities?.[eid]?.tracks || [])) pairs.push({ cam, tid })
    }
  } else if (store.sel) {
    pairs.push({ cam: store.sel.cam, tid: store.sel.tid })
  }
  for (const c of store.cams) {
    const mine = pairs.filter(p => p.cam === c.id)
    const all: number[][] = []
    for (const p of mine) {
      for (const q of (store.tracks[c.id]?.tracks?.[String(p.tid)] || [])) {
        all.push([(q[1] + q[3]) / 2 / c.w, q[4] / c.h])
      }
    }
    out.push({
      cam: c.id, name: c.name, pts: all,
      color: mine.length ? (store.chainColor[store.selChain || ''] || 'var(--accent)') : 'var(--line)',
      active: mine.length > 0
    })
  }
  return out
})

function path(pts: number[][]): string {
  if (!pts.length) return ''
  const step = Math.max(1, Math.floor(pts.length / 120))
  let d = ''
  for (let i = 0; i < pts.length; i += step) {
    const x = pts[i][0] * 100, y = pts[i][1] * 100
    d += (d ? 'L' : 'M') + x.toFixed(1) + ' ' + y.toFixed(1)
  }
  return d
}
const hasSel = computed(() => !!(store.selChain || store.sel))
const activeCount = computed(() => minis.value.filter(m => m.active).length)
</script>

<template>
  <div class="mm panel col">
    <div class="panel-h">
      落脚点轨迹
      <span class="sub">归一化 · 各机位视野</span>
      <div class="grow" />
      <span v-if="hasSel" class="tag mono">{{ activeCount }}/6 机位</span>
    </div>
    <div class="panel-b">
      <div v-if="!hasSel" class="empty" style="padding:var(--sp-4) var(--sp-2)">
        <b>未选中目标</b>
        <span class="tiny">选中接力链或轨迹后，这里显示它在各机位视野中的脚点轨迹</span>
      </div>
      <template v-else>
        <div class="grid">
          <div v-for="m in minis" :key="m.cam" class="cell" :class="{ off: !m.active }">
            <div class="ch">
              <b class="mono">{{ m.cam }}</b>
              <span class="tiny dim ellip">{{ m.name }}</span>
            </div>
            <svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" :aria-label="`${m.cam} 脚点轨迹`">
              <rect x="0" y="0" width="100" height="100" fill="none" stroke="var(--stage-line)" stroke-width="0.4" />
              <path v-if="m.pts.length" :d="path(m.pts)" fill="none" :stroke="m.color" stroke-width="1.8"
                    vector-effect="non-scaling-stroke" stroke-linejoin="round" stroke-linecap="round" />
              <circle v-if="m.pts.length" :cx="m.pts[m.pts.length - 1][0] * 100"
                      :cy="m.pts[m.pts.length - 1][1] * 100" r="2" :fill="m.color" />
            </svg>
          </div>
        </div>
        <p class="tiny dim" style="margin-top:var(--sp-2)">坐标为框底边中点（脚点）归一化位置；按 1/120 抽稀绘制。</p>
      </template>
    </div>
  </div>
</template>

<style scoped>
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.cell { background: var(--stage); border: 1px solid var(--stage-line); border-radius: var(--r-xs); overflow: hidden; }
.cell.off { opacity: .38; }
.ch { display: flex; gap: 5px; align-items: baseline; padding: 3px 6px; background: var(--surface-2); border-bottom: 1px solid var(--line); min-width: 0; }
.ch b { font-size: var(--text-xs); color: var(--accent); }
svg { display: block; width: 100%; aspect-ratio: 16/10; }
</style>
