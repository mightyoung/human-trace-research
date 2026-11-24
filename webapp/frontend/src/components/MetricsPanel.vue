<script setup lang="ts">
import { computed } from 'vue'
import { store } from '../store'

const c0 = computed(() => store.metrics?.c0 || {})
const inv = computed(() => c0.value?.inventory || {})
const perCam = computed(() => Object.keys(inv.value).map(k => ({ cam: k, ...inv.value[k] })))
const totals = computed(() => ({
  entities: store.metrics?.manifest?.totals?.n_entities ?? '—',
  edges: c0.value?.meta?.n_edges ?? store.chains.edges?.length ?? '—',
  chains: store.chains.n_chains,
  tracks: Object.values(store.entities).reduce((a: number, v: any) => a + (v.n_tracks || 0), 0)
}))
const metricsRows = computed(() => {
  const m = c0.value?.metrics || c0.value
  const out: { k: string; v: string }[] = []
  const get = (o: any, path: string[]) => path.reduce((a, p) => a?.[p], o)
  const push = (label: string, o: any, p: string[]) => {
    const v = get(o, p)
    if (v !== undefined && v !== null) {
      out.push({ k: label, v: typeof v === 'number' ? (Math.abs(v) < 1 ? v.toFixed(3) : v.toFixed(1)) : String(v) })
    }
  }
  push('出门连续性', m && m.exit, ['continuity'])
  push('进门归属率', m && m.entry, ['attribution'])
  if (m && m.exit && typeof m.exit === 'object') push('出门连续性', m.exit, ['continuity'])
  if (m && m.entry && typeof m.entry === 'object') push('进门归属率', m.entry, ['attribution'])
  return out
})
</script>

<template>
  <div class="mp panel col">
    <div class="panel-h">
      全局指标
      <div class="grow" />
      <span class="sub mono">clean2 · 2026-10-14</span>
    </div>
    <div class="panel-b col" style="gap:var(--sp-3)">
      <!-- 指标条：用分隔线而非卡片，避免卡片套卡片 -->
      <div class="stats">
        <div class="metric">
          <b>{{ totals.entities }}</b><span>实体</span>
        </div>
        <div class="metric">
          <b>{{ totals.chains }}</b><span>接力链</span>
        </div>
        <div class="metric">
          <b>{{ totals.edges }}</b><span>跨机边</span>
        </div>
        <div class="metric">
          <b>{{ totals.tracks }}</b><span>轨迹</span>
        </div>
      </div>

      <div v-if="metricsRows.length" class="kv">
        <template v-for="r in metricsRows" :key="r.k">
          <div>{{ r.k }}</div><div class="hl">{{ r.v }}</div>
        </template>
      </div>

      <div>
        <div class="label" style="margin-bottom:5px">各机位实体资产</div>
        <div class="table cam-table">
          <div class="th">机位</div><div class="th">实体</div><div class="th">可动</div>
          <div class="th">碎片</div><div class="th">p50 时长</div>
          <template v-for="r in perCam" :key="r.cam">
            <div class="mono">{{ r.cam }}</div>
            <div class="num">{{ r.n_entities }}</div>
            <div class="num">{{ r.n_moving }}</div>
            <div class="num">{{ r.n_frag_total }}</div>
            <div class="num">{{ r.dur_p50?.toFixed?.(1) }}s</div>
          </template>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.mp { min-height: 0; overflow: auto; }
.stats { display: grid; grid-template-columns: repeat(4, 1fr); }
.metric { padding: 0 var(--sp-2); border-left: 1px solid var(--line); }
.metric:first-child { padding-left: 0; border-left: 0; }
.metric > b { font-size: var(--text-xl); font-weight: 650; line-height: 1.1; }
.hl { color: var(--accent); }
.cam-table { grid-template-columns: auto repeat(4, minmax(0, 1fr)); }
</style>
