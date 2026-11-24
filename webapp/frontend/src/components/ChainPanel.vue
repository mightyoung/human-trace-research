<script setup lang="ts">
import { computed, ref } from 'vue'
import { store, selectChain } from '../store'

const q = ref('')
const chains = computed(() => (store.chains.chains || []).filter((c: any) => {
  if (!q.value) return true
  const s = c.id + ' ' + c.nodes.map((n: any) => n.cam + n.eid).join(' ')
  return s.toLowerCase().includes(q.value.toLowerCase())
}))
function colorOf(cid: string) { return store.chainColor[cid] || 'var(--fg-3)' }
function nodeColor(n: any) { return colorOf(store.entChain[`${n.cam}:${n.eid}`] || '') }

// ---- v2 数据源（--relay v2）时展示边的证据等级：audit 为弱证据、需人工复核 ----
const hasTier = computed(() => (store.chains.edges || []).some((e: any) => e.status))
const edgeIdx = computed(() => {
  const m = new Map<string, any>()
  for (const e of store.chains.edges || []) {
    if (e.status) m.set(`${e.a}|${e.b}`, e)
  }
  return m
})
function chainTier(c: any): { audit: number; active: number } | null {
  if (!hasTier.value) return null
  let audit = 0, active = 0
  for (let i = 1; i < (c.nodes || []).length; i++) {
    const a = `${c.nodes[i - 1].cam}:${c.nodes[i - 1].eid}`
    const b = `${c.nodes[i].cam}:${c.nodes[i].eid}`
    const e = edgeIdx.value.get(`${a}|${b}`) || edgeIdx.value.get(`${b}|${a}`)
    if (e?.status === 'audit') audit++
    else if (e) active++
  }
  return { audit, active }
}
</script>

<template>
  <div class="cp panel col grow">
    <div class="panel-h">
      跨机接力链
      <span class="tag info mono">{{ store.chains.n_chains }}</span>
      <div class="grow" />
      <span class="sub mono hide-sm">{{ store.chains.thr ? 'cos ≥ ' + store.chains.thr : 'relay v2' }}</span>
      <button v-if="store.selChain" class="icon-btn" title="清除高亮" aria-label="清除高亮"
              @click="selectChain(null)">✕</button>
    </div>

    <div class="search">
      <label class="sr-only" for="chain-q">搜索接力链</label>
      <svg class="si" viewBox="0 0 16 16" width="13" height="13" aria-hidden="true" fill="none"
           stroke="currentColor" stroke-width="1.6"><circle cx="7" cy="7" r="4.4" /><path d="M10.4 10.4 14 14" /></svg>
      <input id="chain-q" v-model="q" type="search" placeholder="链号 / 机位 / 实体" />
    </div>

    <div class="list scroll grow">
      <button v-for="c in chains" :key="c.id" class="chain"
              :aria-pressed="store.selChain === c.id"
              @click="selectChain(store.selChain === c.id ? null : c.id)">
        <span class="head row">
          <i class="swatch" :style="{ background: colorOf(c.id) }" />
          <b class="cid mono">{{ c.id }}</b>
          <span class="tag">{{ c.len }} 跳</span>
          <div class="grow" />
          <span v-if="chainTier(c)" class="tag" :class="chainTier(c)!.audit ? 'warn' : 'ok'"
              :title="chainTier(c)!.audit ? '含弱证据边（audit），需人工复核' : '全部为 active 边'">
          {{ chainTier(c)!.audit ? `audit ×${chainTier(c)!.audit}` : 'active' }}
        </span>
        <span class="tiny dim mono">{{ c.nodes[0]?.t0s?.toFixed(0) }}s 起</span>
        </span>
        <span class="nodes">
          <template v-for="(n, i) in c.nodes" :key="i">
            <span class="nd mono" :style="{ borderColor: nodeColor(n), color: nodeColor(n) }">{{ n.cam }}</span>
            <span v-if="i < c.nodes.length - 1" class="ar" aria-hidden="true">→</span>
          </template>
        </span>
      </button>

      <div v-if="!chains.length" class="empty">
        <svg class="art" viewBox="0 0 48 48" width="40" height="40" fill="none" stroke="currentColor" stroke-width="1.6">
          <circle cx="12" cy="24" r="5" /><circle cx="36" cy="14" r="5" /><circle cx="36" cy="34" r="5" />
          <path d="M16.6 21.6 31.6 15.6M16.6 26.4l15 6" stroke-dasharray="3 3" />
        </svg>
        <b>{{ q ? '没有匹配的接力链' : '暂无接力链' }}</b>
        <span class="tiny">{{ q ? '换个关键词试试，或清空搜索框' : '跨机候选对达到相似度阈值后会自动成链' }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.cp { min-height: 0; }
.search { position: relative; padding: var(--sp-2) var(--sp-3); border-bottom: 1px solid var(--line); }
.si { position: absolute; left: calc(var(--sp-3) + 8px); top: 50%; transform: translateY(-50%); color: var(--fg-3); pointer-events: none; }
.search input { padding-left: 27px; }

.list { padding: var(--sp-2); display: flex; flex-direction: column; gap: 3px; }
/* flex 列中列表项必须取自然高度：min-height:0 会让项被压缩，两行内容相互重叠 */
.chain {
  flex: none;
  display: flex; flex-direction: column; align-items: stretch; gap: 5px;
  padding: 7px var(--sp-2); border-radius: var(--r-sm);
  border: 1px solid transparent; background: transparent; text-align: left;
  width: 100%;
}
@media (hover: hover) { .chain:hover { background: var(--surface-2); } }
.chain[aria-pressed='true'] { background: var(--accent-soft); border-color: var(--accent-line); }
.head { gap: 6px; }
.swatch { width: 3px; height: 13px; border-radius: 2px; flex: none; }
.cid { font-size: var(--text-xs); font-weight: 700; }
.nodes { display: flex; align-items: center; gap: 3px; flex-wrap: wrap; }
.nd { font-size: var(--text-2xs); padding: 0 5px; border: 1px solid; border-radius: var(--r-xs); line-height: 15px; }
.ar { color: var(--fg-3); font-size: var(--text-2xs); }
.empty { padding: var(--sp-4) var(--sp-2); }
</style>
