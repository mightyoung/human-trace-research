<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { api } from '../api'

type Lab = { v?: number; c?: string; n?: string }
type EvLab = { pid?: string; fp?: boolean }

const route = useRoute()
const st = ref<any>(null)
const met = ref<any>(null)
const tab = ref<'pair' | 'event' | 's6b'>('pair')
const filter = ref('sug')
const sortDt = ref(false)
const saving = ref(false)
const dirtyN = ref(0)          // 未保存（未落盘）的改动项数
const savedAt = ref('')
const err = ref('')

const labels = ref<Record<string, Lab>>({})
const evLabels = ref<Record<string, EvLab>>({})

// ---- S6-b 待裁决（倒影降权 A/B 分歧清单）----
// ★ 读切换（B1①）：条目与判定统一来自台账（/api/ops/queue）；这里只保留**分组文案**
//   （标题/规则/hint），它不是数据，属于任务说明。数字一律以台账为准。
const s6b = ref<any>(null)
const s6bRows = ref<any[]>([])
const s6bTab = ref('relay_conflict')
const s6bOnlyUndone = ref(false)

const TASKS = ['t_truth_pair', 't_truth_event', 't_s6b_pending']

const pairs = computed<any[]>(() => st.value?.pairs || [])
const events = computed<any[]>(() => st.value?.events || [])
const confA = computed(() => new Set(Object.keys(st.value?.conflicts?.a || {})))
const confB = computed(() => new Set(Object.keys(st.value?.conflicts?.b || {})))
const sel = computed<any>(() => st.value?.meta?.select || {})

const filtered = computed<any[]>(() => {
  let l = pairs.value.filter((p: any) => {
    const lab = labels.value[p.pair_id] || {}
    if (filter.value === 'sug') return p.role === 'topk' || p.role === 'audit'
    if (filter.value === 'topk') return p.role === 'topk'
    if (filter.value === 'audit') return p.role === 'audit'
    if (filter.value === 'rest') return p.role === 'rest'
    if (filter.value === 'relay') return p.relay === 1
    if (filter.value === 'hi') return p.cos >= 0.95
    if (filter.value === 'gray') return p.cos >= 0.85 && p.cos < 0.95
    if (filter.value === 'lo') return p.cos < 0.85
    if (filter.value === 'nd') return lab.v === undefined
    return true
  })
  l = [...l].sort((a: any, b: any) =>
    sortDt.value ? a.dt - b.dt : (a.rank ?? 999) - (b.rank ?? 999))
  return l
})

const progress = computed(() => {
  const n = pairs.value.length
  const done = pairs.value.filter((p: any) => {
    const l = labels.value[p.pair_id]
    return !!l && l.v !== undefined
  }).length
  const eDone = events.value.filter((e: any) => {
    const r = evLabels.value[e.eid]
    return !!r && (!!r.pid || !!r.fp)
  }).length
  return { done, n, eDone, en: events.value.length }
})

function cls(p: any) {
  const lab = labels.value[p.pair_id]
  if (!lab || lab.v === undefined) return ''
  return lab.v === 1 ? 'same' : lab.v === 0 ? 'diff' : 'unk'
}

function pct(x: any) { return x == null ? '—' : (x * 100).toFixed(1) + '%' }
function f3(x: any) { return x == null ? '—' : Number(x).toFixed(3) }
function f2(x: any) { return x == null ? '—' : Number(x).toFixed(2) }

/**
 * 读切换（B1①）：条目 + 判定一律来自台账 `/api/ops/queue`。
 * - `payload` 是候选的全部展示字段（cos/dt/rank/role/缩略图定位…），逐字段等同于原 state.json；
 * - `label.data` 是原始判定对象（{v,c,n} / {pid,fp}），只有 `value` 会丢置信度与备注；
 * - `meta.select.k_topk/k_audit` 由分层现算（候选生成时的权威 role 已入库）。
 */
function pairLab(it: any): Lab | null {
  if (!it.label) return null
  const d = it.label.data
  if (d && typeof d === 'object' && !Array.isArray(d)) {
    const v = d.v === undefined || d.v === null ? undefined : Number(d.v)
    return { v: Number.isFinite(v as number) ? (v as number) : undefined, c: d.c, n: d.n }
  }
  const n = Number(it.label.value)
  return { v: Number.isFinite(n) ? n : undefined }
}
function evLab(it: any): EvLab | null {
  if (!it.label) return null
  const d = it.label.data
  if (d && typeof d === 'object' && !Array.isArray(d)) {
    return { pid: d.pid === undefined || d.pid === null || d.pid === '' ? undefined : String(d.pid),
             fp: !!d.fp }
  }
  const s = String(it.label.value || '')
  if (s === 'fp') return { fp: true }
  if (s.startsWith('pid:')) return { pid: s.slice(4) }
  return null
}

async function load() {
  const q: any = await api.opsQueue(TASKS)
  const tp: any[] = q.items['t_truth_pair'] || []
  const te: any[] = q.items['t_truth_event'] || []
  s6bRows.value = q.items['t_s6b_pending'] || []
  const mt = (q.tasks || []).find((t: any) => t.id === 't_truth_pair') || {}
  const kTop = tp.filter((i: any) => i.stratum === 'topk').length
  const kAud = tp.filter((i: any) => i.stratum === 'audit').length
  st.value = {
    meta: {
      title: mt.name || '', n_pairs: tp.length, n_events: te.length,
      note: mt.note || '',
      select: { k_topk: kTop, k_audit: kAud, budget: kTop + kAud }
    },
    pairs: tp.map((i: any) => ({ ...(i.payload || {}), pair_id: i.key })),
    events: te.map((i: any) => ({ ...(i.payload || {}), eid: i.key }))
  }
  const L: Record<string, Lab> = {}
  for (const it of tp) { const l = pairLab(it); if (l) L[it.key] = l }
  for (const it of s6bRows.value) { const l = pairLab(it); if (l) L[it.key] = l }
  labels.value = L
  const E: Record<string, EvLab> = {}
  for (const it of te) { const l = evLab(it); if (l) E[it.key] = l }
  evLabels.value = E
  // 互斥冲突：同侧被多对判为「同人」→ 前端提示（口径与老 state 一致）
  const aCnt: Record<string, number> = {}, bCnt: Record<string, number> = {}
  for (const p of st.value.pairs) {
    if (L[p.pair_id]?.v === 1) {
      aCnt[p.a] = (aCnt[p.a] || 0) + 1
      bCnt[p.b] = (bCnt[p.b] || 0) + 1
    }
  }
  const pick = (m: Record<string, number>) =>
    Object.fromEntries(Object.entries(m).filter(([, v]) => v > 1))
  st.value.conflicts = { a: pick(aCnt), b: pick(bCnt) }
}
async function loadMet() {
  try { met.value = await api.truthMetrics() } catch { /* 指标失败不阻断判定 */ }
}
onMounted(async () => {
  window.addEventListener('keydown', onKey)
  window.addEventListener('beforeunload', onBeforeUnload)
  try { await load(); await loadMet() } catch (e: any) { err.value = String(e) }
  // 支持 #/truth?tab=event|s6b 直达（复核时贴链接用）
  const t = String(route.query.tab || '')
  if (t === 'event') tab.value = 'event'
  else if (t === 's6b') await openS6b()
})
onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKey)
  window.removeEventListener('beforeunload', onBeforeUnload)
  if (dirtyN.value) flush()          // 切页时尽力落盘
})

// -------- 保存：显式「保存」按钮为主，8s 自动保存为安全网 --------
let timer: any = 0
let pending: any = { pair: {}, event: {} }

function countPending() {
  return Object.keys(pending.pair).length + Object.keys(pending.event).length
}
function queue(kind: 'pair' | 'event', id: string, val: any) {
  pending[kind][id] = val
  dirtyN.value = countPending()
  clearTimeout(timer)
  timer = setTimeout(() => { if (dirtyN.value) flush() }, 8000)   // 安全网
}
async function flush(manual = false) {
  if (!dirtyN.value) { if (manual) savedAt.value = '无改动可保存'; return }
  const body: any = {}
  if (Object.keys(pending.pair).length) body.pair = { ...pending.pair }
  if (Object.keys(pending.event).length) body.event = { ...pending.event }
  if (!Object.keys(body).length) return
  const n = countPending()
  const snapshot = { pair: { ...pending.pair }, event: { ...pending.event } }
  clearTimeout(timer)
  pending = { pair: {}, event: {} }
  dirtyN.value = 0
  saving.value = true
  try {
    await api.truthSave(body)
    const t = new Date()
    savedAt.value = `已保存 ${n} 项 · ${String(t.getHours()).padStart(2, '0')}:` +
      `${String(t.getMinutes()).padStart(2, '0')}:${String(t.getSeconds()).padStart(2, '0')}`
    await loadMet()          // 实时刷新指标
    err.value = ''
  } catch (e: any) {
    for (const k of ['pair', 'event'] as const) Object.assign(pending[k], snapshot[k])
    dirtyN.value = countPending()
    err.value = '保存失败（改动已保留，请重试）：' + e
  }
  saving.value = false
}

// Ctrl/Cmd+S 保存；离开页面前拦截未保存改动
function onKey(e: KeyboardEvent) {
  if ((e.ctrlKey || e.metaKey) && (e.key === 's' || e.key === 'S')) {
    e.preventDefault(); flush(true)
  }
}
function onBeforeUnload(e: BeforeUnloadEvent) {
  if (dirtyN.value) { e.preventDefault(); e.returnValue = '' }
}

function lab(p: any): Lab {
  if (!labels.value[p.pair_id]) labels.value[p.pair_id] = {}
  return labels.value[p.pair_id]
}
function setV(p: any, v: number) { lab(p).v = v; queue('pair', p.pair_id, { ...lab(p) }) }
function setC(p: any, c: string) { lab(p).c = c; queue('pair', p.pair_id, { ...lab(p) }) }
function setN(p: any, n: string) { lab(p).n = n; queue('pair', p.pair_id, { ...lab(p) }) }
function elab(e: any): EvLab {
  if (!evLabels.value[e.eid]) evLabels.value[e.eid] = {}
  return evLabels.value[e.eid]
}
function setPid(e: any, pid: string) { elab(e).pid = pid; queue('event', e.eid, { ...elab(e) }) }
function setFp(e: any, fp: boolean) { elab(e).fp = fp; queue('event', e.eid, { ...elab(e) }) }

async function refresh() {
  try { await load(); await loadMet(); err.value = '' } catch (e: any) { err.value = String(e) }
}

// ================= S6-b 待裁决 =================
async function loadS6b() {
  // 只取分组文案（标题/hint/规则）；条目与判定由 load() 从台账取
  s6b.value = await api.s6bPending()
  const g = s6b.value.groups || []
  if (g.length && !g.some((x: any) => x.key === s6bTab.value)) s6bTab.value = g[0].key
}
const s6bGroup = computed<any>(() =>
  (s6b.value?.groups || []).find((g: any) => g.key === s6bTab.value) || null)
const s6bItems = computed<any[]>(() => {
  const its = s6bRows.value
    .filter((i: any) => (i.payload?.group || '') === s6bTab.value)
    .map((i: any) => ({ ...(i.payload || {}), id: i.key }))
  return s6bOnlyUndone.value ? its.filter((i: any) => labels.value[i.id]?.v === undefined) : its
})
const s6bProg = computed(() => {
  const out: Record<string, { done: number; n: number }> = {}
  for (const g of (s6b.value?.groups || [])) out[g.key] = { done: 0, n: 0 }
  for (const i of s6bRows.value) {
    const k = i.payload?.group || 'g'
    if (!out[k]) out[k] = { done: 0, n: 0 }
    out[k].n++
    if (labels.value[i.key]?.v !== undefined) out[k].done++
  }
  return out
})
function s6bCls(i: any) {
  const l = labels.value[i.id]
  if (!l || l.v === undefined) return ''
  return l.v === 1 ? 'same' : l.v === 0 ? 'diff' : 'unk'
}
function setS6b(i: any, v: number) {
  if (!labels.value[i.id]) labels.value[i.id] = {}
  labels.value[i.id].v = v
  queue('pair', i.id, { ...labels.value[i.id] })
}
function setS6bC(i: any, c: string) {
  if (!labels.value[i.id]) labels.value[i.id] = {}
  labels.value[i.id].c = c
  queue('pair', i.id, { ...labels.value[i.id] })
}
function setS6bN(i: any, n: string) {
  if (!labels.value[i.id]) labels.value[i.id] = {}
  labels.value[i.id].n = n
  queue('pair', i.id, { ...labels.value[i.id] })
}
const S6B_KIND: Record<string, { q1: string; q0: string; q2: string }> = {
  relay_conflict: { q1: '同人', q0: '不是', q2: '看不清' },
  bridge_merge: { q1: '同人（该留）', q0: '不同人（该拆）', q2: '看不清' },
  reflection_entity: { q1: '是倒影', q0: '是真人', q2: '看不清' },
  face_conflict: { q1: '同人（人脸对）', q0: '不同人（人脸误判）', q2: '看不清' }
}
function s6bOptions(i: any) {
  const k = s6bGroup.value?.key || 'relay_conflict'
  return S6B_KIND[k] || S6B_KIND.relay_conflict
}
async function openS6b() {
  tab.value = 's6b'
  if (!s6b.value) {
    try { await loadS6b() } catch (e: any) { err.value = 'S6-b 载荷加载失败：' + e }
  }
}
</script>

<template>
  <div class="truth col">
    <div class="toolbar row wrap">
      <div class="seg">
        <button :aria-selected="tab === 'pair'" @click="tab = 'pair'">
          配对判定 {{ progress.done }}/{{ progress.n }}
        </button>
        <button :aria-selected="tab === 'event'" @click="tab = 'event'">
          事件人物编号 {{ progress.eDone }}/{{ progress.en }}
        </button>
        <button :aria-selected="tab === 's6b'" @click="openS6b()">
          S6-b 待裁决 {{ s6b ? (Object.values(s6bProg).reduce((a: any, b: any) => a + b.done, 0) +
            '/' + Object.values(s6bProg).reduce((a: any, b: any) => a + b.n, 0)) : '' }}
        </button>
      </div>

      <template v-if="tab === 's6b' && s6b">
        <div class="seg">
          <button v-for="g in s6b.groups" :key="g.key" :aria-selected="s6bTab === g.key"
                  @click="s6bTab = g.key">
            {{ g.title.split('（')[0] }} {{ s6bProg[g.key]?.done }}/{{ s6bProg[g.key]?.n }}
          </button>
        </div>
        <label class="tiny dim row" style="gap:4px">
          <input type="checkbox" v-model="s6bOnlyUndone" /> 只看未判
        </label>
      </template>

      <template v-if="tab === 'pair'">
        <select v-model="filter" aria-label="过滤">
          <option value="sug">★ 建议判定（Top-{{ sel.k_topk }} + 审计 {{ sel.k_audit }}）</option>
          <option value="topk">预测 Top-{{ sel.k_topk }}</option>
          <option value="audit">随机审计 {{ sel.k_audit }}</option>
          <option value="rest">其余 {{ st?.meta?.n_pairs - sel.k_topk - sel.k_audit }}</option>
          <option value="all">全部</option>
          <option value="relay">relay v5 支持</option>
          <option value="hi">cos ≥ 0.95</option>
          <option value="gray">灰区 0.85–0.95</option>
          <option value="lo">cos &lt; 0.85</option>
          <option value="nd">只看未标注</option>
        </select>
        <label class="tiny dim row" style="gap:4px">
          <input type="checkbox" v-model="sortDt" /> 按 dt 升序
        </label>
      </template>

      <span class="grow" />
      <span class="tag" :class="dirtyN || saving ? 'warn' : 'ok'">
        <i class="dot" />
        {{ saving ? '保存中…' : dirtyN ? ('未保存 ' + dirtyN + ' 项') : (savedAt || '已同步') }}
      </span>
      <button class="primary" :disabled="saving || !dirtyN" @click="flush(true)"
              title="保存全部改动（Ctrl+S / ⌘+S）">保存{{ dirtyN ? `（${dirtyN}）` : '' }}</button>
      <button class="ghost" @click="refresh" title="重新拉取（含互斥冲突与指标）">刷新</button>
      <a class="btn" :href="api.truthExportUrl('pair')">导出配对 CSV</a>
      <a class="btn" :href="api.truthExportUrl('event')">导出事件 CSV</a>
    </div>

    <!-- 实时指标 -->
    <div v-if="tab === 'pair' && met" class="stats">
      <div class="s">
        <b>{{ met.budget_done }}/{{ met.budget }}</b>
        <span class="tiny dim">预算进度（已标，其中已决 {{ met.decided }}）</span>
      </div>
      <div class="s">
        <b>{{ f3(met.rank_acc) }}</b>
        <span class="tiny dim">排序准确率（两两/AUC）</span>
      </div>
      <div class="s">
        <b>{{ f3(met.brier) }}</b>
        <span class="tiny dim">Brier（排名先验）</span>
      </div>
      <div class="s">
        <b>{{ met.est_total_same == null ? '—' : met.est_total_same.toFixed(1) }}</b>
        <span class="tiny dim">估计同人总数 / {{ met.n_pairs }}（{{ pct(met.prevalence) }}）</span>
      </div>
      <div class="s">
        <b>{{ met.arm_found }} vs {{ met.random_expected == null ? '—' : met.random_expected.toFixed(1) }}</b>
        <span class="tiny dim">同预算命中 vs 随机50期望
          <template v-if="met.efficiency != null"> · ×{{ f2(met.efficiency) }}</template>
        </span>
      </div>
      <div class="s">
        <b>{{ met.relay.tp }}/{{ met.relay.annotated }}</b>
        <span class="tiny dim">relay v5 精度（召回 {{ pct(met.relay.recall_on_judged) }}）</span>
      </div>
      <div class="s">
        <b>{{ met.pos }}/{{ met.neg }}/{{ met.unclear }}</b>
        <span class="tiny dim">同人 / 异人 / 看不清</span>
      </div>
    </div>

    <div v-if="err" class="banner">{{ err }}</div>
    <div v-if="tab === 's6b' && s6b" class="hint tiny dim">
      {{ s6b.title }} — {{ s6b.rules }}
    </div>
    <div v-else-if="st?.meta" class="hint tiny dim">{{ st.meta.note }}</div>

    <div class="scroll body">
      <template v-if="tab === 'pair'">
        <div class="cards">
          <div v-for="p in filtered" :key="p.pair_id" class="pcard" :class="cls(p)">
            <div class="cell">
              <img :src="api.truthThumbUrl(p.a)" loading="lazy" alt="" />
              <div class="cap tiny dim">A · 343 · {{ p.sa }} · t={{ p.ta.toFixed(0) }}s</div>
            </div>
            <div class="cell">
              <img :src="api.truthThumbUrl(p.b)" loading="lazy" alt="" />
              <div class="cap tiny dim">B · 316 · {{ p.sb }} · t={{ p.tb.toFixed(0) }}s</div>
            </div>
            <div class="meta col">
              <div class="small">
                <span class="tag" :class="p.role === 'topk' ? 'info' : p.role === 'audit' ? 'warn' : ''"
                      style="margin-right:6px">#{{ p.rank }} · {{ p.role }}</span>
                <b>{{ p.sa }} ↔ {{ p.sb }}</b>
              </div>
              <div class="tiny dim">
                dt {{ p.dt.toFixed(1) }}s · cos
                <b>{{ p.cos_missing ? '缺失' : p.cos.toFixed(3) }}</b>
                · pred <b>{{ p.pred.toFixed(3) }}</b>
                <span v-if="p.cos_missing" class="tag warn">cos 缺失·回退分</span>
                <span v-if="p.relay" class="tag info">relay v5</span>
                <span v-if="p.relay_v2" class="tag" :class="p.relay_v2 === 'active' ? 'ok' : 'warn'">
                  relay v2 · {{ p.relay_v2 }}
                </span>
                <span v-if="labels[p.pair_id]?.v === 1 && (confA.has(p.a) || confB.has(p.b))"
                      class="tag bad">⚠ 互斥冲突</span>
              </div>
              <div class="opt row wrap">
                <button :aria-selected="labels[p.pair_id]?.v === 1" @click="setV(p, 1)">同人</button>
                <button :aria-selected="labels[p.pair_id]?.v === 0" @click="setV(p, 0)">异人</button>
                <button :aria-selected="labels[p.pair_id]?.v === 2" @click="setV(p, 2)">看不清</button>
                <span class="tiny dim" style="margin-left:6px">置信</span>
                <button v-for="c in ['high', 'mid', 'low']" :key="c"
                        :aria-selected="labels[p.pair_id]?.c === c" @click="setC(p, c)">{{ c }}</button>
              </div>
              <input type="text" placeholder="备注（可选）" :value="labels[p.pair_id]?.n || ''"
                     @input="setN(p, ($event.target as HTMLInputElement).value)" />
            </div>
          </div>
        </div>
      </template>

      <template v-else-if="tab === 'event'">
        <div class="tiny dim" style="padding:0 12px 8px">
          同一人的事件填相同编号；反光伪影/误检勾 <b>FP</b>。编号可跨机位、跨时间。
        </div>
        <div class="egrid">
          <div v-for="e in events" :key="e.eid" class="ecard col">
            <img :src="api.truthThumbUrl(e.eid)" loading="lazy" alt="" />
            <div class="tiny dim ellip">{{ e.short }} · {{ e.cam }} · {{ e.kind }}/{{ e.edge }} · t={{ e.t.toFixed(0) }}s</div>
            <div class="row" style="gap:8px">
              <input type="text" style="width:78px" placeholder="编号"
                     :value="evLabels[e.eid]?.pid || ''"
                     @input="setPid(e, ($event.target as HTMLInputElement).value)" />
              <label class="tiny row" style="gap:4px">
                <input type="checkbox" :checked="!!evLabels[e.eid]?.fp"
                       @change="setFp(e, ($event.target as HTMLInputElement).checked)" /> FP
              </label>
            </div>
          </div>
        </div>
      </template>

      <!-- ============ S6-b 待裁决 ============ -->
      <template v-else>
        <div v-if="s6bGroup" class="tiny dim" style="padding:0 12px 8px">
          {{ s6bGroup.hint }} · 判定读台账；写入落 truth/labels.json 并同步台账（前缀 S6B_，不影响既有统计）。
        </div>
        <div class="cards">
          <div v-for="i in s6bItems" :key="i.id" class="pcard" :class="s6bCls(i)">
            <div class="cell">
              <div class="thumbrow">
                <img v-for="t in (i.side_a || [])" :key="'a' + t.f"
                     :src="api.s6bThumbUrl(t.f)" loading="lazy" alt="" />
                <div v-if="!(i.side_a || []).length" class="noimg tiny dim">无图</div>
              </div>
              <div class="cap tiny dim">
                {{ i.kind === 'mirror' ? 'A · 待判轨迹' : 'A' }} · {{ i.sa || i.entity }}
                <template v-if="i.side_a?.[0]?.gid"> · {{ i.side_a[0].gid }}</template>
              </div>
            </div>
            <div class="cell">
              <div class="thumbrow">
                <img v-for="t in (i.side_b || [])" :key="'b' + t.f"
                     :src="api.s6bThumbUrl(t.f)" loading="lazy" alt="" />
                <div v-if="!(i.side_b || []).length" class="noimg tiny dim">
                  {{ i.kind === 'mirror' ? '源不在 clean2' : '—' }}
                </div>
              </div>
              <div class="cap tiny dim">
                {{ i.kind === 'mirror' ? 'B · 其「源」参考' : 'B' }} · {{ i.sb }}
                <template v-if="i.side_b?.[0]?.gid"> · {{ i.side_b[0].gid }}</template>
              </div>
            </div>
            <div class="meta col">
              <div class="small">
                <span class="tag" :class="i.relay_v2 === 'active' ? 'ok' : (i.relay_v2 ? 'warn' : '')"
                      style="margin-right:6px">
                  {{ i.kind === 'mirror' ? (i.pure ? '纯倒影实体' : '混有真人') : (i.relay_v2 || 'bridge') }}
                </span>
                <b>{{ i.sa }}<span v-if="i.sb"> ↔ {{ i.sb }}</span></b>
              </div>
              <div class="tiny dim">
                <template v-if="i.cos != null">cos <b>{{ Number(i.cos).toFixed(3) }}</b> · </template>
                <template v-if="i.dt != null">dt {{ Number(i.dt).toFixed(1) }}s · </template>
                <template v-if="i.members">成员 {{ i.members.join(', ') }} · </template>
                <span v-if="i.machine === 'contradicted'" class="tag info">A/B 分歧</span>
              </div>
              <div v-if="i.note" class="tiny dim">{{ i.note }}</div>
              <div class="opt row wrap">
                <button :aria-selected="labels[i.id]?.v === 1" @click="setS6b(i, 1)">
                  {{ s6bOptions(i).q1 }}
                </button>
                <button :aria-selected="labels[i.id]?.v === 0" @click="setS6b(i, 0)">
                  {{ s6bOptions(i).q0 }}
                </button>
                <button :aria-selected="labels[i.id]?.v === 2" @click="setS6b(i, 2)">
                  {{ s6bOptions(i).q2 }}
                </button>
                <span class="tiny dim" style="margin-left:6px">置信</span>
                <button v-for="c in ['high', 'mid', 'low']" :key="c"
                        :aria-selected="labels[i.id]?.c === c" @click="setS6bC(i, c)">{{ c }}</button>
              </div>
              <input type="text" placeholder="备注（可选）" :value="labels[i.id]?.n || ''"
                     @input="setS6bN(i, ($event.target as HTMLInputElement).value)" />
            </div>
          </div>
          <div v-if="!s6bItems.length" class="tiny dim" style="padding:12px">
            本组已全部判定。
          </div>
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.truth { height: 100%; min-height: 0; }
.toolbar {
  flex: none; gap: 10px; padding: 8px 12px;
  border-bottom: 1px solid var(--line); background: var(--surface-1);
  align-items: center;
}
.stats {
  flex: none; display: flex; flex-wrap: wrap; gap: 8px;
  padding: 8px 12px; border-bottom: 1px solid var(--line); background: var(--surface-inset);
}
.s {
  display: flex; flex-direction: column; gap: 1px; min-width: 116px;
  padding: 5px 10px; border: 1px solid var(--line); border-radius: var(--r-sm);
  background: var(--surface-1);
}
.s b { font-size: var(--text-md, 13px); font-variant-numeric: tabular-nums; }
.banner {
  flex: none; margin: 8px 12px 0; padding: 8px 10px; border-radius: var(--r-sm);
  background: var(--danger-soft); border: 1px solid var(--danger-line); color: var(--danger);
  font-size: var(--text-sm);
}
.hint { flex: none; padding: 6px 12px; border-bottom: 1px solid var(--line); }
.body { flex: 1; min-height: 0; padding: 10px 12px 40px; }

.cards { display: flex; flex-direction: column; gap: 8px; }
.pcard {
  display: grid; grid-template-columns: 210px 210px minmax(0, 620px); gap: 12px;
  padding: 8px 10px; border: 1px solid var(--line); border-radius: var(--r-md, 10px);
  background: var(--surface-1); align-items: start;
}
.pcard.same { box-shadow: inset 4px 0 0 var(--ok); }
.pcard.diff { box-shadow: inset 4px 0 0 var(--danger); }
.pcard.unk { box-shadow: inset 4px 0 0 var(--warn); }
.cell { display: flex; flex-direction: column; gap: 3px; min-width: 0; }
.cell img {
  width: 100%; height: 218px; object-fit: contain;
  background: var(--surface-inset); border: 1px solid var(--line); border-radius: var(--r-sm);
}
/* S6-b：一侧可能有多张图（多成员/多 obs）→ 等分并排 */
.thumbrow { display: flex; gap: 4px; min-width: 0; }
.thumbrow img {
  flex: 1 1 0; min-width: 0; height: 218px; object-fit: contain;
  background: var(--surface-inset); border: 1px solid var(--line); border-radius: var(--r-sm);
}
.noimg {
  flex: 1 1 0; height: 218px; display: flex; align-items: center; justify-content: center;
  background: var(--surface-inset); border: 1px dashed var(--line); border-radius: var(--r-sm);
}
.cap { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.meta { gap: 5px; min-width: 0; }
.meta input[type='text'] { width: 100%; }
.opt { gap: 6px; }
.opt button[aria-selected='true'] {
  background: var(--accent-soft); color: var(--accent); border-color: var(--accent-line);
  font-weight: var(--weight-label, 600);
}
.egrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; }
.ecard {
  gap: 4px; padding: 8px; border: 1px solid var(--line); border-radius: var(--r-md, 10px);
  background: var(--surface-1);
}
.ecard img {
  width: 100%; max-height: 200px; object-fit: contain;
  background: var(--surface-inset); border-radius: var(--r-sm);
}
@media (max-width: 767px) {
  .pcard { grid-template-columns: 1fr 1fr; }
  .pcard .meta { grid-column: 1 / -1; }
}
</style>
