<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { api } from '../api'
import { store, trackSpan, frameOf, markLabels, trajOf } from '../store'
import FrameView from '../components/FrameView.vue'
import MaskEditor from '../components/MaskEditor.vue'
import AttrAnnotator from '../components/AttrAnnotator.vue'
import AttrIndex from '../components/AttrIndex.vue'
import AttrInfer from '../components/AttrInfer.vue'

const tab = ref<'pairs' | 'trackpair' | 'clusterver' | 'mergever' | 'mirror' | 'mask' | 'attr' | 'attrindex' | 'infer'>('pairs')
const pairs = ref<any[]>([])
const pairFilter = ref('UNDECIDED')
const selPair = ref(0)
const showList = ref(false)

const TAB_KEYS = ['pairs', 'trackpair', 'clusterver', 'mergever', 'mirror', 'mask', 'attr', 'attrindex', 'infer']

onMounted(async () => {
  // 支持 #/annotate?tab=attr 直达：手机端页签横向溢出，滑到「人员属性标注」很麻烦
  const m = window.location.hash.match(/[?&]tab=([a-z]+)/)
  if (m && TAB_KEYS.includes(m[1])) tab.value = m[1] as any
  pairs.value = await api.pairs()
  // 全判完时默认「待决定」= 空列表，首屏看着像坏了 → 自动落到「全部」
  if (!pairs.value.some((p: any) => pdecision(p) === 'UNLABELED')) pairFilter.value = 'ALL'
  loadTp()
  loadCv()
  loadMv()
  window.addEventListener('keydown', onKey)
})
onUnmounted(() => window.removeEventListener('keydown', onKey))

function onKey(e: KeyboardEvent) {
  const t = e.target as HTMLElement
  if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA')) return
  if (e.key === 'j' || e.key === 'J') { next(1) }
  else if (e.key === 'k' || e.key === 'K') { next(-1) }
  else if (tab.value === 'pairs' && /^[123]$/.test(e.key)) {
    decidePair(['CONF', 'MODERATE', 'REJ'][+e.key - 1])
  } else if (tab.value === 'trackpair' && /^[123]$/.test(e.key)) {
    // ★ 与「候选对」页签的语义不同：这里 1=同人 2=不同人 3=不确定（更接近直觉）
    tpSave((['same', 'diff', 'unsure'] as const)[+e.key - 1])
  } else if (tab.value === 'clusterver' && /^[1234]$/.test(e.key)) {
    // 簇级：1=整簇同一人 2=提交标记（有外人） 3=不确定 4=明显多人（未逐条标）
    if (e.key === '1') cvSave('pure')
    else if (e.key === '2') cvSave('mixed')
    else if (e.key === '3') cvSave('unsure')
    else cvSave('multi')
  } else if (tab.value === 'mergever' && /^[123]$/.test(e.key)) {
    mvSave((['same', 'diff', 'unsure'] as const)[+e.key - 1])
  } else if (tab.value === 'mirror' && /^[123]$/.test(e.key)) {
    decideMirror(['person', 'reflection', 'source'][+e.key - 1])
  }
}

// ---- 候选对 ----
function pkey(p: any) { return `${p.cam_a}:${p.tid_a}|${p.cam_b}:${p.tid_b}` }
function pdecision(p: any): string { return store.labels.pair_decisions[pkey(p)] || p.label || 'UNLABELED' }
const filteredPairs = computed(() => pairs.value.filter(p => {
  const d = pdecision(p)
  if (pairFilter.value === 'ALL') return true
  if (pairFilter.value === 'UNDECIDED') return d === 'UNLABELED'
  if (pairFilter.value === 'DECIDED') return d !== 'UNLABELED'
  return p.tier === pairFilter.value
}))
const decidedCount = computed(() => pairs.value.filter(p => pdecision(p) !== 'UNLABELED').length)
const cur = computed(() => filteredPairs.value[selPair.value] || null)
const progress = computed(() => pairs.value.length ? Math.round(decidedCount.value / pairs.value.length * 100) : 0)

const tMid = computed(() => {
  const p = cur.value
  if (!p) return null
  const a = trackSpan(p.cam_a, p.tid_a), b = trackSpan(p.cam_b, p.tid_b)
  if (!a) return null
  if (b) {
    const lo = Math.max(a.t0, b.t0), hi = Math.min(a.t1, b.t1)
    if (hi > lo) return (lo + hi) / 2
  }
  return a.t0 + 0.35 * (a.t1 - a.t0)
})
const frameA = computed(() => {
  const p = cur.value; if (!p || tMid.value === null) return 0
  return frameOf(p.cam_a, tMid.value)
})
const frameB = computed(() => {
  const p = cur.value; if (!p || tMid.value === null) return 0
  return frameOf(p.cam_b, tMid.value)
})

async function decidePair(val: string) {
  const p = cur.value; if (!p) return
  const key = pkey(p)
  markLabels({ pair_decisions: { [key]: val } })
  await api.saveLabels({ pair_decisions: { [key]: val } })
  const list = filteredPairs.value
  if (selPair.value < list.length - 1) selPair.value++
  else if (selPair.value >= list.length) selPair.value = Math.max(0, list.length - 1)
}

/**
 * 脚点轨迹迷你图 —— 画布即「相机画面」：16:9 viewBox(160×90)，画框边界 + 三分网格提供空间参照；
 * 轨迹按画面归一化坐标映射；起止点（绿=出生，红=消失）标注方向。
 * 面板本身画面同比例，因此轨迹形状不畸变、位置可与上方帧图直接对照。
 */
const MV_W = 160, MV_H = 90, MV_PAD = 5
type MiniView = { d: string; s: [number, number] | null; e: [number, number] | null; n: number; tight: boolean; pts: [number, number][]; ring: { x: number; y: number; r: number } | null }
function miniView(cam: string, tid: number): MiniView {
  const empty: MiniView = { d: '', s: null, e: null, n: 0, tight: false, pts: [], ring: null }
  const c = store.camMap[cam]; const pts = trajOf(cam, tid)
  if (!c || !pts.length) return empty
  const kx = MV_W - 2 * MV_PAD, ky = MV_H - 2 * MV_PAD
  const step = Math.max(1, Math.floor(pts.length / 120))
  const uv: [number, number][] = []
  for (let i = 0; i < pts.length; i += step) {
    uv.push([MV_PAD + (pts[i][1] + pts[i][3]) / 2 / c.w * kx, MV_PAD + pts[i][4] / c.h * ky])
  }
  const last = pts[pts.length - 1]
  uv.push([MV_PAD + (last[1] + last[3]) / 2 / c.w * kx, MV_PAD + last[4] / c.h * ky])
  let d = ''
  uv.forEach(([x, y], i) => { d += (i ? 'L' : 'M') + x.toFixed(1) + ' ' + y.toFixed(1) })
  const xs = uv.map(a => a[0]), ys = uv.map(a => a[1])
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys)
  // 近静止轨迹（包围盒极小）：叠加点阵，让"簇"本身可见，而不是一个孤点
  const tight = (x1 - x0) < 8 && (y1 - y0) < 8
  const s0 = uv[0], s1 = uv[uv.length - 1]
  const ring = tight
    ? { x: (x0 + x1) / 2, y: (y0 + y1) / 2, r: Math.max(4.5, (Math.max(x1 - x0, y1 - y0)) * 0.75 + 2.5) }
    : null
  return { d, s: s0, e: s1, n: pts.length, tight, pts: tight ? uv : [], ring }
}
const mvA = computed<MiniView>(() => (cur.value ? miniView(cur.value.cam_a, cur.value.tid_a) : { d: '', s: null, e: null, n: 0, tight: false, pts: [], ring: null }))
const mvB = computed<MiniView>(() => (cur.value ? miniView(cur.value.cam_b, cur.value.tid_b) : { d: '', s: null, e: null, n: 0, tight: false, pts: [], ring: null }))

// ═══ 轨迹同人复核（全量卷 v2559 / v2526）═══
// 与「跨机候选对」的区别：这里是**同一机位、不同时段**的两段轨迹，问的是「是不是同一个人」。
// 用途：这两卷没有任何身份真值 → 用人工对给全量聚类补一份可评估的真值。
// C(位置近) 与 D(位置远) 是同一外观区间的对照组，判定结果直接检验位置先验。
const tpPairs = ref<any[]>([])
const tpFilter = ref('UNDONE')
const tpSel = ref(0)
const tpBusy = ref(false)
const tpDiag = ref(false)
const tpStat = ref<any>(null)
const TP_ZH: Record<string, string> = { same: '同人', diff: '不同人', unsure: '不确定' }
const TP_KIND_ZH: Record<string, string> = {
  A: '簇内·测精度', B: '跨簇·测召回', C: '中等外观·位置近', D: '中等外观·位置远'
}

async function loadTp() {
  try {
    const d = await api.trackpairPairs()
    tpPairs.value = d.pairs || []
    tpStat.value = d
  } catch (e) {
    tpPairs.value = []
  }
}
async function loadTpStat() {
  try { tpStat.value = await api.trackpairStats() } catch (e) { /* 统计只是辅助，失败不打断标注 */ }
}

const tpShown = computed(() => tpPairs.value.filter(p => {
  const f = tpFilter.value
  if (f === 'ALL') return true
  if (f === 'UNDONE') return !p.label
  if (f === 'DONE') return !!p.label
  return p.kind === f
}))
const curTp = computed(() => tpShown.value[tpSel.value] || null)
const tpDone = computed(() => tpPairs.value.filter(p => !!p.label).length)
const tpRate = computed(() => tpPairs.value.length ? Math.round(tpDone.value / tpPairs.value.length * 100) : 0)
const tpBysame = computed(() => {
  // 按 A/B/C/D 汇总同人率 —— 只在「统计」折叠区展示，避免标注过程中被假设带偏
  const out: Record<string, { n: number; labeled: number; same: number; diff: number; unsure: number }> = {}
  for (const p of tpPairs.value) {
    const d = (out[p.kind] ||= { n: 0, labeled: 0, same: 0, diff: 0, unsure: 0 })
    d.n++
    if (p.label) { d.labeled++; (d as any)[p.label]++ }
  }
  return out
})

async function tpSave(val: string | null) {
  const p = curTp.value
  if (!p || tpBusy.value) return
  const prev = p.label
  p.label = val
  tpBusy.value = true
  try {
    await api.trackpairSave(p.id, val)
  } catch (e) {
    p.label = prev                 // 落盘失败就回滚，别让界面显示"已标"其实没存
    tpBusy.value = false
    return
  }
  tpBusy.value = false
  loadTpStat()
  // ★ 自动前进只能走一格。在「待复核」筛选下这条判定后已从列表消失，
  //   列表自身前移了一格，再 ++ 会漏掉一对（实测会从第 1 对直接跳到第 3 对）。
  const n = tpShown.value.length
  if (!n) return
  if (!tpShown.value.includes(p)) {
    if (tpSel.value >= n) tpSel.value = n - 1
  } else if (val && tpSel.value < n - 1) {
    tpSel.value++
  }
}
function tpNext(d: number) {
  const n = tpShown.value.length
  if (!n) return
  tpSel.value = (tpSel.value + d + n) % n
}
function tpDur(s: number) {
  if (s == null) return '—'
  const v = Math.round(s)
  return Math.floor(v / 60) + ':' + String(v % 60).padStart(2, '0')
}
function tpImg(key: string, idx: number | 'full') { return api.trackpairImgUrl(key, idx) }

// ---- 簇纯度核验（v2545 走廊机位）----
// 与「轨迹同人复核」是两种问题：那里问「这两条轨是不是同一个人」，这里问
// 「这一簇里混没混别的人、混了几条」。走廊机位没有身份真值，簇级核验是拿到
// 纯度（误合并率）最便宜的方式：15 个判断覆盖 129 条轨。
const cvItems = ref<any[]>([])
const cvFilter = ref('UNDONE')
const cvSel = ref(0)
const cvBusy = ref(false)
const cvStat = ref<any>(null)
const CV_VERDICT_ZH: Record<string, string> = {
  pure: '整簇同一人', mixed: '有外人', multi: '明显多人', unsure: '不确定'
}

async function loadCv() {
  try {
    const d = await api.clusterverItems()
    cvItems.value = (d.items || []).map((x: any) => ({ ...x, marks: { ...(x.marks || {}) } }))
    cvStat.value = d
  } catch (e) {
    cvItems.value = []
  }
}
async function loadCvStat() {
  try { cvStat.value = await api.clusterverStats() } catch (e) { /* 汇总只是辅助 */ }
}

const cvShown = computed(() => cvItems.value.filter(x => {
  const f = cvFilter.value
  if (f === 'ALL') return true
  if (f === 'UNDONE') return !x.verdict
  if (f === 'DONE') return !!x.verdict
  return x.verdict === f
}))
const curCv = computed(() => cvShown.value[cvSel.value] || null)
const cvDone = computed(() => cvItems.value.filter(x => !!x.verdict).length)
const cvRate = computed(() => cvItems.value.length ? Math.round(cvDone.value / cvItems.value.length * 100) : 0)
const cvOut = computed(() => {
  const c = curCv.value
  return c ? c.members.filter((m: any) => c.marks[m.k] === 'out').length : 0
})
const cvUnread = computed(() => {
  const c = curCv.value
  return c ? c.members.filter((m: any) => c.marks[m.k] === 'unread').length : 0
})

function cvImg(key: string, idx: number) { return api.clusterverCropUrl(key, idx) }

// 单击循环：主体 → 外人 → 看不清 → 主体。不做模式切换，标注时手不用离开图。
function cvCycle(m: any) {
  const c = curCv.value
  if (!c) return
  const cur = c.marks[m.k]
  if (!cur) c.marks[m.k] = 'out'
  else if (cur === 'out') c.marks[m.k] = 'unread'
  else delete c.marks[m.k]
}

async function cvSave(v: string | null) {
  const c = curCv.value
  if (!c || cvBusy.value) return
  // 「有外人」却没标任何外人 = 无从判断，直接不提交（按钮也是 disabled 的）
  if (v === 'mixed' && cvOut.value === 0) return
  const prevV = c.verdict
  const prevM = { ...c.marks }
  if (v !== 'mixed') c.marks = {}      // pure / unsure / 清除 时标记无意义
  c.verdict = v
  cvBusy.value = true
  try {
    const r: any = await api.clusterverSave(c.id, v, c.marks)
    // ★ POST 返回的是 {counts, estimate:{...}}，与 GET /stats 的扁平结构不同。
    //   直接塞进去会让「纯度」永远显示 —（曾踩过）。
    cvStat.value = { counts: r.counts, ...(r.estimate || {}) }
  } catch (e) {
    c.verdict = prevV
    c.marks = prevM
    cvBusy.value = false
    return
  }
  cvBusy.value = false
  // ★ 自动前进只能走一格：筛选=待核验时这条判完就从列表消失、列表自身前移一格，
  //   再 ++ 会跳过一簇（与 tpSave 同一个坑，实测会跳）。
  const n = cvShown.value.length
  if (!n) return
  if (!cvShown.value.includes(c)) {
    if (cvSel.value >= n) cvSel.value = n - 1
  } else if (v && cvSel.value < n - 1) {
    cvSel.value++
  }
}
function cvNext(d: number) {
  const n = cvShown.value.length
  if (!n) return
  cvSel.value = (cvSel.value + d + n) % n
}

// ---- 位置合并判读（步骤 2 的 26 组无真值合并 + 7 组时间重叠对）----
// 与簇核验的区别：那里问「这一簇纯不纯」，这里问「这次合并该不该发生」——看图直接判
// 两侧是不是同一个人。位置余弦 1.0 的组单独标注：两簇落进同一个 12×8 格子，
// 远景下一个格子约 213×180 px，相邻工位可能同格，所以那是弱证据，不能当强证据用。
const mvItems = ref<any[]>([])
const mvFilter = ref('UNDONE')
const mvSel = ref(0)
const mvBusy = ref(false)
const mvStat = ref<any>(null)
const MV_VERDICT_ZH: Record<string, string> = {
  same: '同一人（合并正确）', diff: '不同人（误合并）', unsure: '看不清'
}

async function loadMv() {
  try {
    const d = await api.mergeverItems()
    mvItems.value = d.items || []
    mvStat.value = d
  } catch (e) {
    mvItems.value = []
  }
}
const mvShown = computed(() => mvItems.value.filter(x => {
  const f = mvFilter.value
  if (f === 'ALL') return true
  if (f === 'UNDONE') return !x.verdict
  if (f === 'DONE') return !!x.verdict
  if (f === 'MERGE') return x.kind === 'merge'
  if (f === 'OVERLAP') return x.kind === 'overlap'
  if (f === 'POSONE') return !!x.pos_one
  return x.verdict === f
}))
const curMv = computed(() => mvShown.value[mvSel.value] || null)
const mvDone = computed(() => mvItems.value.filter(x => !!x.verdict).length)
const mvRate = computed(() => mvItems.value.length ? Math.round(mvDone.value / mvItems.value.length * 100) : 0)

async function mvSave(v: string | null) {
  const c = curMv.value
  if (!c || mvBusy.value) return
  const prev = c.verdict
  c.verdict = v
  mvBusy.value = true
  try {
    const r: any = await api.mergeverSave(c.id, v)
    mvStat.value = { counts: r.counts, ...(r.estimate || {}) }
  } catch (e) {
    c.verdict = prev
    mvBusy.value = false
    return
  }
  mvBusy.value = false
  // ★ 与 cvSave 同一个坑：待判筛选下判完该项即从列表消失，列表自己前移，不能再 ++
  const n = mvShown.value.length
  if (!n) return
  if (!mvShown.value.includes(c)) {
    if (mvSel.value >= n) mvSel.value = n - 1
  } else if (v && mvSel.value < n - 1) {
    mvSel.value++
  }
}
function mvNext(d: number) {
  const n = mvShown.value.length
  if (!n) return
  mvSel.value = (mvSel.value + d + n) % n
}

// ---- 倒影标注 ----
// ★ 只收「倒影 / 源」两类：掩膜区（玻璃遮蔽区）是另一件事，判断依据是区域而不是
//   镜像运动学，混进这个列表只会让人反复判同一条轨；它在跟踪页/图例里单独显示。
const mirrorCands = computed(() => {
  const set = new Map<string, any>()
  for (const [c, t] of (store.mirror.reflection || [])) set.set(`${c}:${t}`, { cam: String(c), tid: +t, kind: 'reflection' })
  for (const [c, t] of (store.mirror.source || [])) if (!set.has(`${c}:${t}`)) set.set(`${c}:${t}`, { cam: String(c), tid: +t, kind: 'source' })
  for (const cam of Object.keys(store.trackStatus)) {
    for (const tid of Object.keys(store.trackStatus[cam])) {
      const s = store.trackStatus[cam][tid]
      if (s === 'masked') continue          // 掩膜区不进倒影/源复核
      if (s !== 'keep' && !set.has(`${cam}:${tid}`)) set.set(`${cam}:${tid}`, { cam, tid: +tid, kind: s })
    }
  }
  return Array.from(set.values()).sort((a, b) => a.cam.localeCompare(b.cam) || a.tid - b.tid)
})
const selMirror = ref(0)
const mirrorFilter = ref('reflection')
const mirrorShown = computed(() => mirrorFilter.value === 'all'
  ? mirrorCands.value
  : mirrorCands.value.filter(c => c.kind === mirrorFilter.value))
const curMirror = computed(() => mirrorShown.value[selMirror.value] || null)
const mvM = computed<MiniView>(() => (curMirror.value ? miniView(curMirror.value.cam, curMirror.value.tid) : { d: '', s: null, e: null, n: 0, tight: false, pts: [], ring: null }))
const mirrorFrame = computed(() => {
  const c = curMirror.value; if (!c) return 0
  const s = trackSpan(c.cam, c.tid); return s ? Math.round(s.f0 + 0.4 * (s.f1 - s.f0)) : 0
})
function mdecision(cam: string, tid: number): string {
  return store.labels.mirror_decisions[`${cam}:${tid}`] || 'UNLABELED'
}
const mirrorDecided = computed(() => mirrorShown.value.filter(c => mdecision(c.cam, c.tid) !== 'UNLABELED').length)
async function decideMirror(val: string) {
  const c = curMirror.value; if (!c) return
  const key = `${c.cam}:${c.tid}`
  markLabels({ mirror_decisions: { [key]: val } })
  await api.saveLabels({ mirror_decisions: { [key]: val } })
  if (selMirror.value < mirrorShown.value.length - 1) selMirror.value++
}

function next(d: number) {
  if (tab.value === 'pairs') {
    const n = filteredPairs.value.length
    if (!n) return
    selPair.value = (selPair.value + d + n) % n
  } else if (tab.value === 'trackpair') {
    tpNext(d)
  } else if (tab.value === 'mirror') {
    const n = mirrorShown.value.length
    if (!n) return
    selMirror.value = (selMirror.value + d + n) % n
  }
}

// ---- 玻璃圈选 ----
const maskCam = ref('316')
async function saveMask(cam: string, pts: number[][]) {
  markLabels({ glass_masks: { [cam]: pts } })
  await api.saveLabels({ glass_masks: { [cam]: pts } })
}

const tierTag = (t: string) => t === 'AUTO' ? 'info' : t === 'REVIEW' ? 'warn' : ''
const DEC_ZH: Record<string, string> = { CONF: '同一人', MODERATE: '未决', REJ: '不同人', UNLABELED: '未标注' }
</script>

<template>
  <div class="ann">
    <!-- 任务切换 -->
    <div class="tabs">
      <div class="seg" role="tablist" aria-label="标注任务">
        <button role="tab" :aria-selected="tab === 'pairs'" @click="tab = 'pairs'">跨机候选对复核</button>
        <button role="tab" :aria-selected="tab === 'trackpair'" @click="tab = 'trackpair'">轨迹同人复核</button>
        <button role="tab" :aria-selected="tab === 'clusterver'" @click="tab = 'clusterver'">簇纯度核验</button>
        <button role="tab" :aria-selected="tab === 'mergever'" @click="tab = 'mergever'">位置合并判读</button>
        <button role="tab" :aria-selected="tab === 'mirror'" @click="tab = 'mirror'">倒影 / 源标注</button>
        <button role="tab" :aria-selected="tab === 'mask'" @click="tab = 'mask'">玻璃区域圈选</button>
        <button role="tab" :aria-selected="tab === 'attr'" @click="tab = 'attr'">人员属性标注</button>
        <button role="tab" :aria-selected="tab === 'attrindex'" @click="tab = 'attrindex'">属性索引</button>
        <button role="tab" :aria-selected="tab === 'infer'" @click="tab = 'infer'">图片试跑</button>
      </div>
      <div class="grow" />
      <template v-if="tab === 'trackpair'">
        <a class="pill only-sm" :href="api.trackpairExportUrl()" download>导出标注 JSON</a>
        <span class="tiny dim hide-sm">写入落 <span class="mono">data/trackpair_labels.json</span>（独立文件）</span>
        <span class="tiny dim hide-sm">｜<span class="kbd">J</span><span class="kbd">K</span> 切换 · <span class="kbd">1</span> 同人 <span class="kbd">2</span> 不同人 <span class="kbd">3</span> 不确定</span>
      </template>
      <template v-else-if="tab === 'mergever'">
        <a class="pill only-sm" :href="api.mergeverExportUrl()" download>导出判读 JSON</a>
        <span class="tiny dim hide-sm">写入落 <span class="mono">data/mergever_labels.json</span>（独立文件）</span>
        <span class="tiny dim hide-sm">｜<span class="kbd">J</span><span class="kbd">K</span> 切换 · <span class="kbd">1</span> 同一人 <span class="kbd">2</span> 不同人 <span class="kbd">3</span> 看不清</span>
      </template>
      <template v-else-if="tab === 'clusterver'">
        <a class="pill only-sm" :href="api.clusterverExportUrl()" download>导出标注 JSON</a>
        <span class="tiny dim hide-sm">写入落 <span class="mono">data/clusterver_labels.json</span>（独立文件）</span>
        <span class="tiny dim hide-sm">｜<span class="kbd">J</span><span class="kbd">K</span> 切换 · <span class="kbd">1</span> 整簇同一人 <span class="kbd">2</span> 提交标出的外人 <span class="kbd">3</span> 不确定</span>
      </template>
      <template v-else-if="tab === 'pairs' || tab === 'mirror'">
        <button class="pill only-sm" :aria-pressed="showList" @click="showList = !showList">
          {{ tab === 'pairs' ? '候选列表' : '轨迹列表' }}
        </button>
        <span class="tiny dim hide-sm">判定读台账 <span class="mono">ops/ops.db</span>；写入落 <span class="mono">data/labels.json</span> 并同步台账</span>
        <span class="tiny dim hide-sm">｜<span class="kbd">J</span><span class="kbd">K</span> 切换 · <span class="kbd">1</span>–<span class="kbd">3</span> 判定</span>
      </template>
      <span v-else-if="tab === 'attr'" class="tiny dim hide-sm">写入落 <span class="mono">data/attr_labels.json</span>（独立文件，不动 labels.json）</span>
    </div>

    <!-- 窄屏列表抽屉 -->
    <div v-if="showList" class="scrim" @click="showList = false" />
    <div v-if="showList && tab === 'pairs'" class="sheet bottom" role="dialog" aria-label="候选对列表">
      <div class="sheet-h">候选对 <span class="tag info mono">{{ decidedCount }}/{{ pairs.length }}</span>
        <div class="grow" /><button class="icon-btn" @click="showList = false" aria-label="关闭">✕</button></div>
      <div class="sheet-b" style="padding:0">
        <div class="filters">
          <select v-model="pairFilter">
            <option value="UNDECIDED">待决定</option><option value="ALL">全部</option>
            <option value="DECIDED">已决定</option><option value="AUTO">AUTO 层</option><option value="REVIEW">REVIEW 层</option>
          </select>
        </div>
        <button v-for="(p, i) in filteredPairs" :key="'m' + pkey(p)" class="pitem"
                :aria-selected="cur === p" @click="selPair = i; showList = false">
          <span class="row">
            <b class="mono">{{ p.tid_a }}</b><span class="dim">×</span><b class="mono">{{ p.tid_b }}</b>
            <span class="tag" :class="tierTag(p.tier)">{{ p.tier }}</span>
            <span class="grow" />
            <span class="tag" :class="pdecision(p) === 'UNLABELED' ? '' : pdecision(p) === 'REJ' ? 'bad' : 'ok'">
              {{ DEC_ZH[pdecision(p)] || pdecision(p) }}</span>
          </span>
        </button>
      </div>
    </div>
    <div v-if="showList && tab === 'mirror'" class="sheet bottom" role="dialog" aria-label="轨迹列表">
      <div class="sheet-h">待复核轨迹 <span class="tag info mono">{{ mirrorDecided }}/{{ mirrorShown.length }}</span>
        <div class="grow" /><button class="icon-btn" @click="showList = false" aria-label="关闭">✕</button></div>
      <div class="sheet-b" style="padding:0">
        <div class="filters">
          <select v-model="mirrorFilter" @change="selMirror = 0">
            <option value="reflection">倒影</option>
            <option value="source">源</option><option value="all">全部</option>
          </select>
        </div>
        <button v-for="(c, i) in mirrorShown" :key="'m' + c.cam + c.tid" class="pitem"
                :aria-selected="curMirror === c" @click="selMirror = i; showList = false">
          <span class="row">
            <b class="mono">{{ c.cam }}</b><span class="mono">#{{ c.tid }}</span>
            <span class="tag" :class="c.kind === 'reflection' ? 'bad' : 'ok'">{{ c.kind }}</span>
            <span class="grow" />
            <span class="tag">{{ mdecision(c.cam, c.tid) }}</span>
          </span>
        </button>
      </div>
    </div>

    <!-- ═══ 候选对 ═══ -->
    <div v-if="tab === 'pairs'" class="body grid">
      <aside class="list panel col">
        <div class="panel-h">
          候选对
          <span class="tag info mono">{{ decidedCount }}/{{ pairs.length }}</span>
          <div class="grow" />
          <span class="sub mono">{{ progress }}%</span>
        </div>
        <div class="filters">
          <label class="sr-only" for="pf">筛选</label>
          <select id="pf" v-model="pairFilter">
            <option value="UNDECIDED">待决定</option>
            <option value="ALL">全部</option>
            <option value="DECIDED">已决定</option>
            <option value="AUTO">AUTO 层</option>
            <option value="REVIEW">REVIEW 层</option>
          </select>
        </div>
        <div class="scroll grow">
          <button v-for="(p, i) in filteredPairs" :key="pkey(p)" class="pitem"
                  :aria-selected="cur === p" @click="selPair = i">
            <span class="row">
              <b class="mono">{{ p.tid_a }}</b><span class="dim">×</span><b class="mono">{{ p.tid_b }}</b>
              <span class="tag" :class="tierTag(p.tier)">{{ p.tier }}</span>
              <span class="grow" />
              <span class="tag" :class="pdecision(p) === 'UNLABELED' ? '' : pdecision(p) === 'REJ' ? 'bad' : 'ok'">
                {{ DEC_ZH[pdecision(p)] || pdecision(p) }}
              </span>
            </span>
            <span class="tiny dim mono">cos {{ p.reid_cos?.toFixed?.(3) }} · sp {{ p.sp_median?.toFixed?.(3) }} · 共视 {{ p.covis_sec }}s</span>
          </button>
          <div v-if="!filteredPairs.length" class="empty">
            <b>这个筛选下没有条目</b>
            <span class="tiny">把筛选切到「全部」可以看到已决定的记录</span>
          </div>
        </div>
      </aside>

      <section class="review col">
        <template v-if="cur">
          <div class="review-head">
            <button class="icon-btn" @click="next(-1)" aria-label="上一条" title="上一条 (K)">‹</button>
            <span class="bigpair mono">{{ cur.cam_a }}#{{ cur.tid_a }}</span>
            <span class="dim">↔</span>
            <span class="bigpair mono">{{ cur.cam_b }}#{{ cur.tid_b }}</span>
            <span class="tag" :class="tierTag(cur.tier)">{{ cur.tier }}</span>
            <div class="grow" />
            <span class="tiny dim mono">{{ selPair + 1 }} / {{ filteredPairs.length }}</span>
            <button class="icon-btn" @click="next(1)" aria-label="下一条" title="下一条 (J)">›</button>
          </div>

          <div class="frames row">
            <div class="col grow" style="gap:var(--sp-2)">
              <FrameView :cam="cur.cam_a" :f0="frameA" :title="store.camMap[cur.cam_a]?.name || cur.cam_a" />
              <div class="tmini panel">
                <div class="tmini-h">
                  <span class="label">{{ cur.cam_a }} 脚点轨迹（画面归一化）</span>
                  <div class="grow" />
                  <span v-if="mvA.n" class="tiny dim mono">{{ mvA.n }} 点</span>
                </div>
                <svg viewBox="0 0 160 90" aria-hidden="true">
                  <g class="mgrid"><path d="M53.3 0V90M106.7 0V90M0 30H160M0 60H160" /></g>
                  <rect class="mframe" x="0.5" y="0.5" width="159" height="89" rx="1.5" />
                  <path class="mtraj" :d="mvA.d" stroke="var(--accent)" />
                  <circle v-if="mvA.ring" :cx="mvA.ring.x" :cy="mvA.ring.y" :r="mvA.ring.r"
                          class="mring" stroke="var(--accent)" />
                  <template v-if="mvA.tight">
                    <circle v-for="(q, i) in mvA.pts" :key="'p' + i" :cx="q[0]" :cy="q[1]" r="1.1"
                            fill="var(--accent)" opacity=".8" />
                  </template>
                  <template v-if="mvA.s">
                    <circle class="mdot start" :cx="mvA.s[0]" :cy="mvA.s[1]" r="2.2" />
                    <circle class="mdot end" :cx="mvA.e![0]" :cy="mvA.e![1]" r="2.2" />
                  </template>
                </svg>
              </div>
            </div>
            <div class="col grow" style="gap:var(--sp-2)">
              <FrameView :cam="cur.cam_b" :f0="frameB" :title="store.camMap[cur.cam_b]?.name || cur.cam_b" />
              <div class="tmini panel">
                <div class="tmini-h">
                  <span class="label">{{ cur.cam_b }} 脚点轨迹（画面归一化）</span>
                  <div class="grow" />
                  <span v-if="mvB.n" class="tiny dim mono">{{ mvB.n }} 点</span>
                </div>
                <svg viewBox="0 0 160 90" aria-hidden="true">
                  <g class="mgrid"><path d="M53.3 0V90M106.7 0V90M0 30H160M0 60H160" /></g>
                  <rect class="mframe" x="0.5" y="0.5" width="159" height="89" rx="1.5" />
                  <path class="mtraj" :d="mvB.d" stroke="var(--signal)" />
                  <circle v-if="mvB.ring" :cx="mvB.ring.x" :cy="mvB.ring.y" :r="mvB.ring.r"
                          class="mring" stroke="var(--signal)" />
                  <template v-if="mvB.tight">
                    <circle v-for="(q, i) in mvB.pts" :key="'p' + i" :cx="q[0]" :cy="q[1]" r="1.1"
                            fill="var(--signal)" opacity=".8" />
                  </template>
                  <template v-if="mvB.s">
                    <circle class="mdot start" :cx="mvB.s[0]" :cy="mvB.s[1]" r="2.2" />
                    <circle class="mdot end" :cx="mvB.e![0]" :cy="mvB.e![1]" r="2.2" />
                  </template>
                </svg>
              </div>
            </div>
          </div>

          <div class="panel">
            <div class="panel-h">判据</div>
            <div class="panel-b">
              <div class="kv">
                <div>外观 cos（C-RADIO 桥接）</div><div>{{ cur.reid_cos?.toFixed?.(3) ?? '—' }}</div>
                <div>时空残差 中位 / p90</div><div>{{ cur.sp_median?.toFixed?.(3) ?? '—' }} / {{ cur.sp_p90?.toFixed?.(3) ?? '—' }}</div>
                <div>共视时长 · 样本</div><div>{{ cur.covis_sec }}s · {{ cur.sp_n }}</div>
                <div>运动同步 r</div><div>{{ cur.sync_r?.toFixed?.(3) ?? '—' }}</div>
                <div>备注</div><div>{{ cur.note || '—' }}</div>
              </div>
            </div>
          </div>

          <div class="actions">
            <button class="primary" @click="decidePair('CONF')"><span class="kbd">1</span> 同一人</button>
            <button @click="decidePair('MODERATE')"><span class="kbd">2</span> 未决</button>
            <button class="danger" @click="decidePair('REJ')"><span class="kbd">3</span> 不同人</button>
            <div class="grow" />
            <button class="ghost" @click="decidePair('UNLABELED')">清除判定</button>
          </div>
        </template>
        <div v-else class="empty" style="flex:1">
          <b>没有待复核的候选对</b>
          <span class="tiny">左侧把筛选切到「全部」可回看已决定的条目</span>
        </div>
      </section>
    </div>

    <!-- ═══ 轨迹同人复核（全量卷 v2559 / v2526）═══ -->
    <div v-else-if="tab === 'trackpair'" class="body grid">
      <aside class="list panel col">
        <div class="panel-h">
          待复核对
          <span class="tag info mono">{{ tpDone }}/{{ tpPairs.length }}</span>
          <div class="grow" />
          <span class="sub mono">{{ tpRate }}%</span>
        </div>
        <div class="filters">
          <label class="sr-only" for="tf">筛选</label>
          <select id="tf" v-model="tpFilter" @change="tpSel = 0">
            <option value="UNDONE">待复核</option>
            <option value="ALL">全部</option>
            <option value="DONE">已复核</option>
            <option value="A">A 簇内（测精度）</option>
            <option value="B">B 跨簇（测召回）</option>
            <option value="C">C 中等外观 · 位置近</option>
            <option value="D">D 中等外观 · 位置远</option>
          </select>
        </div>
        <div class="scroll grow">
          <button v-for="(p, i) in tpShown" :key="p.id" class="pitem"
                  :aria-selected="curTp === p" @click="tpSel = i">
            <span class="row">
              <b class="mono">{{ p.a }}</b><span class="dim">×</span><b class="mono">{{ p.b }}</b>
              <span class="grow" />
              <span class="tag" :class="p.label === 'same' ? 'ok' : p.label === 'diff' ? 'bad' : p.label === 'unsure' ? 'warn' : ''">
                {{ p.label ? TP_ZH[p.label] : '未复核' }}
              </span>
            </span>
            <span class="tiny dim mono">{{ p.vol }} · {{ TP_KIND_ZH[p.kind] }} · 时长 {{ tpDur(p.dur_a) }} / {{ tpDur(p.dur_b) }}</span>
          </button>
          <div v-if="!tpShown.length" class="empty">
            <b>这个筛选下没有条目</b>
            <span class="tiny">全部标完时「待复核」会空掉，切到「全部」可回看已复核的</span>
          </div>
        </div>
      </aside>

      <section class="review col">
        <template v-if="curTp">
          <div class="review-head">
            <button class="icon-btn" @click="tpNext(-1)" aria-label="上一条" title="上一条 (K)">‹</button>
            <span class="bigpair mono">{{ curTp.a }}</span>
            <span class="dim">↔</span>
            <span class="bigpair mono">{{ curTp.b }}</span>
            <span class="tag">{{ curTp.vol }}</span>
            <div class="grow" />
            <span class="tiny dim mono">{{ tpSel + 1 }} / {{ tpShown.length }}</span>
            <button class="pill" :aria-pressed="tpDiag" @click="tpDiag = !tpDiag">判据</button>
            <button class="icon-btn" @click="tpNext(1)" aria-label="下一条" title="下一条 (J)">›</button>
          </div>

          <div v-for="(k, si) in [curTp.a, curTp.b]" :key="'r' + k" class="trow">
            <div class="tfull">
              <img :src="tpImg(k, 'full')" :alt="k + ' 全景'" loading="lazy" />
              <span class="tiny dim">全景 · 红框为目标</span>
            </div>
            <div class="strips">
              <img v-for="j in [0, 1, 2]" :key="j" :src="tpImg(k, j)"
                   :alt="k + ' 第 ' + (j + 1) + ' 帧'" loading="lazy" />
            </div>
            <div class="tlabel tiny dim mono">{{ si === 0 ? 'A' : 'B' }}</div>
          </div>

          <div class="panel">
            <div class="panel-h">元信息</div>
            <div class="panel-b">
              <div class="kv">
                <div>卷 · 类别</div>
                <div><b>{{ curTp.vol }}</b> · {{ curTp.kind }} {{ TP_KIND_ZH[curTp.kind] }}</div>
                <div>时间间隔</div><div class="mono">{{ tpDur(curTp.gap) }}</div>
                <div>时长 A / B</div><div class="mono">{{ tpDur(curTp.dur_a) }} / {{ tpDur(curTp.dur_b) }}</div>
                <template v-if="tpDiag">
                  <div>外观余弦 cos</div><div class="mono">{{ curTp.cos?.toFixed?.(3) ?? '—' }}</div>
                  <div>位置距离</div><div class="mono">{{ curTp.dist?.toFixed?.(3) ?? '—' }}</div>
                  <div>当前是否同簇</div><div>{{ curTp.same_cluster ? '是' : '否' }}</div>
                </template>
              </div>
            </div>
          </div>

          <div class="actions">
            <button class="primary" @click="tpSave('same')"><span class="kbd">1</span> 同人</button>
            <button class="danger" @click="tpSave('diff')"><span class="kbd">2</span> 不同人</button>
            <button @click="tpSave('unsure')"><span class="kbd">3</span> 不确定</button>
            <div class="grow" />
            <button class="ghost" @click="tpSave(null)">清除判定</button>
          </div>

          <details class="panel" @toggle="loadTpStat">
            <summary class="panel-h">分类统计 <span class="tiny dim">（会泄露假设，标完再看<span v-if="tpStat?.updated"> · 服务端 {{ tpStat.updated }}</span>）</span></summary>
            <div class="panel-b">
              <table class="ttab">
                <thead>
                  <tr><th>类别</th><th>对</th><th>已标</th><th>同人</th><th>不同</th><th>同人率</th></tr>
                </thead>
                <tbody>
                  <tr v-for="(d, k) in tpBysame" :key="k">
                    <td><b>{{ k }}</b> <span class="dim">{{ TP_KIND_ZH[k] }}</span></td>
                    <td class="mono">{{ d.n }}</td>
                    <td class="mono">{{ d.labeled }}</td>
                    <td class="mono">{{ d.same }}</td>
                    <td class="mono">{{ d.diff }}</td>
                    <td class="mono"><b>{{ d.labeled ? (100 * d.same / d.labeled).toFixed(0) + '%' : '—' }}</b></td>
                  </tr>
                </tbody>
              </table>
              <div class="tiny dim" style="margin-top:8px">
                C 与 D 是同一外观区间的对照组：只有 <b>C 的同人率显著高于 D</b>，
                才说明「位置近」在同等外观下确实带信息。
              </div>
            </div>
          </details>
        </template>
        <div v-else class="empty" style="flex:1">
          <b>没有待复核的对</b>
          <span class="tiny">左侧把筛选切到「全部」可回看已复核的条目</span>
        </div>
      </section>
    </div>

    <!-- ═══ 簇纯度核验（v2545 走廊机位，无身份真值）═══ -->
    <div v-else-if="tab === 'clusterver'" class="body grid">
      <aside class="list panel col">
        <div class="panel-h">
          待核验簇
          <span class="tag info mono">{{ cvDone }}/{{ cvItems.length }}</span>
          <div class="grow" />
          <span class="sub mono">{{ cvRate }}%</span>
        </div>
        <div class="filters">
          <label class="sr-only" for="cvf">筛选</label>
          <select id="cvf" v-model="cvFilter" @change="cvSel = 0">
            <option value="UNDONE">待核验</option>
            <option value="ALL">全部</option>
            <option value="DONE">已核验</option>
            <option value="pure">整簇同一人</option>
            <option value="mixed">有外人</option>
            <option value="multi">明显多人</option>
            <option value="unsure">不确定</option>
          </select>
        </div>
        <div class="scroll grow">
          <button v-for="(c, i) in cvShown" :key="c.id" class="pitem"
                  :aria-selected="curCv === c" @click="cvSel = i">
            <span class="row">
              <b class="mono">簇 #{{ c.cid }}</b>
              <span class="dim">{{ c.n }} 轨</span>
              <span class="grow" />
              <span class="tag" :class="c.verdict === 'pure' ? 'ok'
                    : (c.verdict === 'mixed' || c.verdict === 'multi') ? 'bad'
                    : c.verdict === 'unsure' ? 'warn' : ''">
                {{ c.verdict ? CV_VERDICT_ZH[c.verdict] : '未核验' }}
              </span>
            </span>
            <span class="tiny dim mono">{{ c.vol }} · 跨度 {{ tpDur(c.dur) }} · 簇内时长和 {{ tpDur(c.span) }}</span>
          </button>
          <div v-if="!cvShown.length" class="empty">
            <b>这个筛选下没有条目</b>
            <span class="tiny">全部核验完时「待核验」会空掉，切到「全部」可回看</span>
          </div>
        </div>
      </aside>

      <section class="review col">
        <template v-if="curCv">
          <div class="review-head">
            <button class="icon-btn" @click="cvNext(-1)" aria-label="上一簇" title="上一簇 (K)">‹</button>
            <span class="bigpair mono">簇 #{{ curCv.cid }}</span>
            <span class="tag">{{ curCv.n }} 条轨</span>
            <span class="tag">{{ curCv.vol }}</span>
            <div class="grow" />
            <span class="tiny dim">外人 <b class="mono">{{ cvOut }}</b> · 看不清 <b class="mono">{{ cvUnread }}</b></span>
            <span class="tiny dim mono">{{ cvSel + 1 }} / {{ cvShown.length }}</span>
            <button class="icon-btn" @click="cvNext(1)" aria-label="下一簇" title="下一簇 (J)">›</button>
          </div>

          <div class="cvhint tiny">
            点缩略图循环切换：<b>主体</b> → <b class="cvo">外人</b> → <b class="cvu">看不清</b>。
            整簇是同一个人就直接按 <span class="kbd">1</span>；混了别的人先点出外人再按 <span class="kbd">2</span>。
            只剩剪影的轨标「看不清」，它会从纯度分母里剔除，不算成不同人。
          </div>

          <div class="cvgrid">
            <div v-for="m in curCv.members" :key="m.k" class="cvitem"
                 :class="curCv.marks[m.k] || ''" @click="cvCycle(m)">
              <div class="cvimgs">
                <img v-for="j in [0, 1, 2]" :key="j" :src="cvImg(m.k, j)"
                     :alt="m.k + ' 第 ' + (j + 1) + ' 帧'" loading="lazy" />
              </div>
              <div class="cvmeta tiny mono">
                #{{ m.k.split('#')[1] }} · {{ tpDur(m.t0) }} · {{ m.dur.toFixed(1) }}s · h{{ Math.round(m.h) }}
                <span v-if="curCv.marks[m.k] === 'out'" class="tag bad">外人</span>
                <span v-else-if="curCv.marks[m.k] === 'unread'" class="tag warn">看不清</span>
              </div>
            </div>
          </div>

          <div class="actions">
            <button class="primary" @click="cvSave('pure')"><span class="kbd">1</span> 整簇同一人</button>
            <button class="danger" :disabled="!cvOut" @click="cvSave('mixed')">
              <span class="kbd">2</span> 提交（外人 {{ cvOut }}）
            </button>
            <button @click="cvSave('unsure')"><span class="kbd">3</span> 不确定</button>
            <button title="一眼就是好几个人但没逐条标：只记数量，不计入纯度（宁可缺一条，也不虚高）"
                    @click="cvSave('multi')"><span class="kbd">4</span> 明显多人</button>
            <div class="grow" />
            <button class="ghost" @click="cvSave(null)">清除判定</button>
          </div>

          <details class="panel" @toggle="loadCvStat">
            <summary class="panel-h">汇总 <span class="tiny dim">（会泄露假设，核验完再看<span v-if="cvStat?.updated"> · 服务端 {{ cvStat.updated }}</span>）</span></summary>
            <div class="panel-b">
              <div class="kv">
                <div>已核验</div><div class="mono">{{ cvStat?.counts?.n - cvStat?.counts?.undone || 0 }} / {{ cvStat?.counts?.n ?? 0 }} 簇</div>
                <div>同一人 / 有外人 / 明显多人 / 不确定</div>
                <div class="mono">{{ cvStat?.counts?.pure ?? 0 }} / {{ cvStat?.counts?.mixed ?? 0 }} / {{ cvStat?.counts?.multi ?? 0 }} / {{ cvStat?.counts?.unsure ?? 0 }}</div>
                <div>簇纯度（按轨加权）</div>
                <div class="mono">{{ cvStat?.purity != null ? cvStat.purity.toFixed(4) : '—' }}</div>
                <div>簇内对精度（按对数加权）</div>
                <div class="mono">{{ cvStat?.pair_precision != null ? cvStat.pair_precision.toFixed(4) : '—' }}</div>
                <div>参与估计的轨</div><div class="mono">{{ cvStat?.n_tracks_judged ?? 0 }} / {{ cvStat?.counts?.n_tracks ?? 0 }}</div>
                <div>未量化的混簇</div><div class="mono">{{ cvStat?.n_multi_unquantified ?? 0 }} 簇（不计入纯度）</div>
              </div>
              <div class="tiny dim" style="margin-top:8px">
                「有外人」按「每个外人各是一个独立身份」保守计；「看不清」的轨从分母剔除；
                「明显多人」没逐条标，只记数量、不进纯度。
              </div>
            </div>
          </details>
        </template>
        <div v-else class="empty" style="flex:1">
          <b>没有待核验的簇</b>
          <span class="tiny">左侧把筛选切到「全部」可回看已核验的簇</span>
        </div>
      </section>
    </div>

    <!-- ═══ 位置合并判读（步骤 2：26 组无真值合并 + 7 组跨机位时间重叠）═══ -->
    <div v-else-if="tab === 'mergever'" class="body grid">
      <aside class="list panel col">
        <div class="panel-h">
          待判读
          <span class="tag info mono">{{ mvDone }}/{{ mvItems.length }}</span>
          <div class="grow" />
          <span class="sub mono">{{ mvRate }}%</span>
        </div>
        <div class="filters">
          <label class="sr-only" for="mvf">筛选</label>
          <select id="mvf" v-model="mvFilter" @change="mvSel = 0">
            <option value="UNDONE">待判读</option>
            <option value="ALL">全部</option>
            <option value="MERGE">位置合并</option>
            <option value="OVERLAP">时间重叠对</option>
            <option value="POSONE">位置余弦 = 1.0</option>
            <option value="DONE">已判读</option>
            <option value="same">同一人</option>
            <option value="diff">不同人</option>
            <option value="unsure">看不清</option>
          </select>
        </div>
        <div class="scroll grow">
          <button v-for="(c, i) in mvShown" :key="c.id" class="pitem"
                  :aria-selected="curMv === c" @click="mvSel = i">
            <span class="row">
              <b class="mono">{{ c.id }}</b>
              <span class="tiny dim">{{ c.kind === 'overlap' ? '重叠' : (c.pos_one ? 'pos=1.0' : 'pos=' + c.pos_cos.toFixed(2)) }}</span>
              <span class="grow" />
              <span class="tag" :class="c.verdict === 'same' ? 'ok' : c.verdict === 'diff' ? 'bad'
                    : c.verdict === 'unsure' ? 'warn' : ''">
                {{ c.verdict ? MV_VERDICT_ZH[c.verdict] : '未判读' }}
              </span>
            </span>
            <span class="tiny dim mono">
              {{ c.kind === 'overlap' ? c.pid + ' 重叠 ' + Math.round(c.overlap) + 's' : c.nA + ' 轨 × ' + c.nB + ' 轨 · cos ' + c.mean_cos.toFixed(2) }}
            </span>
          </button>
          <div v-if="!mvShown.length" class="empty">
            <b>这个筛选下没有条目</b>
            <span class="tiny">全部判完时「待判读」会空掉，切到「全部」可回看</span>
          </div>
        </div>
      </aside>

      <section class="review col">
        <template v-if="curMv">
          <div class="review-head">
            <button class="icon-btn" @click="mvNext(-1)" aria-label="上一条" title="上一条 (K)">‹</button>
            <span class="bigpair mono">{{ curMv.id }}</span>
            <span class="tag">{{ curMv.kind === 'overlap' ? '时间重叠对' : '位置合并' }}</span>
            <span v-if="curMv.pos_one" class="tag warn">pos = 1.000</span>
            <div class="grow" />
            <span class="tiny dim mono">{{ mvSel + 1 }} / {{ mvShown.length }}</span>
            <button class="icon-btn" @click="mvNext(1)" aria-label="下一条" title="下一条 (J)">›</button>
          </div>

          <div class="cvhint tiny">
            <template v-if="curMv.kind === 'overlap'">
              同一身份被标了两条<b>时间重叠</b>的轨。判断两侧是不是<b>同一个人</b>：
              明显不同人 → 真值同名噪声（交接统计要剔除）；确实是同一人 → 两机位同时可见，正常。
            </template>
            <template v-else>
              左 <b class="cvo">A</b>（{{ curMv.nA }} 轨）与右 <b class="cvu">B</b>（{{ curMv.nB }} 轨）因<b>位置画像</b>被合并，
              两侧都没有身份真值。判断是不是<b>同一个人</b>：是 → 合并正确；
              不同人 → 误合并（<b>换座风险</b>的直接证据）。每侧只抽了最长 4 条轨、每轨 3 帧。
            </template>
          </div>

          <div class="mvwrap">
            <img :src="api.mergeverStripUrl(curMv.img)" :alt="'合并判读 ' + curMv.id" class="mvimg" />
          </div>

          <div class="row" style="gap:8px;flex-wrap:wrap;margin-top:10px">
            <button class="ok" :disabled="mvBusy" @click="mvSave('same')">同一人 <span class="kbd">1</span></button>
            <button class="danger" :disabled="mvBusy" @click="mvSave('diff')">不同人 <span class="kbd">2</span></button>
            <button class="ghost" :disabled="mvBusy" @click="mvSave('unsure')">看不清 <span class="kbd">3</span></button>
            <div class="grow" />
            <button class="ghost" @click="mvSave(null)">清除判定</button>
          </div>

          <div class="kv" style="margin-top:10px;max-width:620px">
            <div>来源</div><div class="mono">{{ curMv.src }}</div>
            <template v-if="curMv.kind === 'overlap'">
              <div>身份 / 重叠</div><div class="mono">{{ curMv.pid }} · {{ Math.round(curMv.overlap) }} s</div>
              <div>A 轨时段</div><div class="mono">{{ curMv.tA?.[0] }} – {{ curMv.tA?.[1] }}</div>
              <div>B 轨时段</div><div class="mono">{{ curMv.tB?.[0] }} – {{ curMv.tB?.[1] }}</div>
            </template>
            <template v-else>
              <div>位置余弦 / 平均外观</div><div class="mono">{{ curMv.pos_cos.toFixed(3) }} / {{ curMv.mean_cos.toFixed(3) }}</div>
              <div>两侧簇规模</div><div class="mono">{{ curMv.nA }} 轨 × {{ curMv.nB }} 轨（新增 {{ curMv.nA * curMv.nB }} 对）</div>
            </template>
            <div>A 侧抽到的轨</div><div class="mono tiny">{{ (curMv.samples?.A || []).join('  ') }}</div>
            <div>B 侧抽到的轨</div><div class="mono tiny">{{ (curMv.samples?.B || []).join('  ') }}</div>
          </div>

          <details class="panel" style="margin-top:10px">
            <summary class="panel-h">汇总 <span class="tiny dim">（会泄露假设，判读过半再看）</span></summary>
            <div class="panel-b">
              <div class="kv">
                <div>合并·可判组数</div><div class="mono">{{ mvStat?.merges?.n_judged ?? 0 }}（同人 {{ mvStat?.merges?.n_same ?? 0 }} / 不同人 {{ mvStat?.merges?.n_diff ?? 0 }}）</div>
                <div>合并·同人率（组口径）</div>
                <div class="mono">{{ mvStat?.merges?.same_rate != null ? (mvStat.merges.same_rate * 100).toFixed(1) + '%' : '—' }}</div>
                <div>合并·同人率（对数加权）</div>
                <div class="mono">{{ mvStat?.merges?.same_rate_pairs != null ? (mvStat.merges.same_rate_pairs * 100).toFixed(1) + '%' : '—' }}</div>
                <div>pos=1.0 子集</div>
                <div class="mono">{{ mvStat?.pos_one?.n_judged ?? 0 }} 组 · 同人率 {{ mvStat?.pos_one?.same_rate != null ? (mvStat.pos_one.same_rate * 100).toFixed(1) + '%' : '—' }}</div>
                <div>重叠对·同人率</div>
                <div class="mono">{{ mvStat?.overlaps?.same_rate != null ? (mvStat.overlaps.same_rate * 100).toFixed(1) + '%' : '—' }}（{{ mvStat?.overlaps?.n_judged ?? 0 }} 组）</div>
              </div>
              <div class="tiny dim" style="margin-top:8px">
                停止条件：可判部分同人率 &lt; 95% → 位置合并只限定在已验证机位，不做默认配置。
                「看不清」一律剔出分母，不算判错。
              </div>
            </div>
          </details>
        </template>
        <div v-else class="empty" style="flex:1">
          <b>没有待判读的条目</b>
          <span class="tiny">左侧把筛选切到「全部」可回看</span>
        </div>
      </section>
    </div>

    <!-- ═══ 倒影 / 源 ═══ -->
    <div v-else-if="tab === 'mirror'" class="body grid">
      <aside class="list panel col">
        <div class="panel-h">
          待复核轨迹
          <span class="tag info mono">{{ mirrorDecided }}/{{ mirrorShown.length }}</span>
          <div class="grow" />
          <label class="sr-only" for="mf">类别筛选</label>
          <select id="mf" v-model="mirrorFilter" class="mini-sel" @change="selMirror = 0">
            <option value="reflection">倒影</option>
            <option value="source">源</option>
            <option value="all">全部</option>
          </select>
        </div>
        <div class="scroll grow">
          <button v-for="(c, i) in mirrorShown" :key="c.cam + ':' + c.tid" class="pitem"
                  :aria-selected="curMirror === c" @click="selMirror = i">
            <span class="row">
              <b class="mono">{{ c.cam }}</b><span class="mono">#{{ c.tid }}</span>
              <span class="tag" :class="c.kind === 'reflection' ? 'bad' : 'ok'">{{ c.kind }}</span>
              <span class="grow" />
              <span class="tag" :class="mdecision(c.cam, c.tid) === 'UNLABELED' ? '' : 'ok'">
                {{ mdecision(c.cam, c.tid) }}
              </span>
            </span>
          </button>
          <div v-if="!mirrorShown.length" class="empty"><b>该类别下没有轨迹</b><span class="tiny">换一个筛选类别试试</span></div>
        </div>
      </aside>

      <section class="review col">
        <template v-if="curMirror">
          <div class="review-head">
            <button class="icon-btn" @click="next(-1)" aria-label="上一条">‹</button>
            <span class="bigpair mono">{{ curMirror.cam }}#{{ curMirror.tid }}</span>
            <span class="tag" :class="curMirror.kind === 'reflection' ? 'bad' : 'ok'">自动判定 {{ curMirror.kind }}</span>
            <div class="grow" />
            <span class="tiny dim mono">{{ selMirror + 1 }} / {{ mirrorShown.length }}</span>
            <button class="icon-btn" @click="next(1)" aria-label="下一条">›</button>
          </div>

          <div class="frames row">
            <div class="col grow">
              <FrameView :cam="curMirror.cam" :f0="mirrorFrame" />
            </div>
            <div class="col side" style="gap:var(--sp-2)">
              <div class="tmini panel">
                <div class="tmini-h">
                  <span class="label">脚点轨迹（画面归一化）</span>
                  <div class="grow" />
                  <span v-if="mvM.n" class="tiny dim mono">{{ mvM.n }} 点</span>
                </div>
                <svg viewBox="0 0 160 90" aria-hidden="true">
                  <g class="mgrid"><path d="M53.3 0V90M106.7 0V90M0 30H160M0 60H160" /></g>
                  <rect class="mframe" x="0.5" y="0.5" width="159" height="89" rx="1.5" />
                  <path class="mtraj" :d="mvM.d" stroke="var(--danger)" />
                  <circle v-if="mvM.ring" :cx="mvM.ring.x" :cy="mvM.ring.y" :r="mvM.ring.r"
                          class="mring" stroke="var(--danger)" />
                  <template v-if="mvM.tight">
                    <circle v-for="(q, i) in mvM.pts" :key="'p' + i" :cx="q[0]" :cy="q[1]" r="1.1"
                            fill="var(--danger)" opacity=".8" />
                  </template>
                  <template v-if="mvM.s">
                    <circle class="mdot start" :cx="mvM.s[0]" :cy="mvM.s[1]" r="2.2" />
                    <circle class="mdot end" :cx="mvM.e![0]" :cy="mvM.e![1]" r="2.2" />
                  </template>
                </svg>
              </div>
              <div class="panel">
                <div class="panel-b tiny dim" style="line-height:1.7">
                  判据：镜像运动学 <span class="mono">v_R ≈ 2·v_脚 − v_源中心</span>。
                  请在画面中确认该轨迹是地上/玻璃中的反光，而不是真人。
                </div>
              </div>
            </div>
          </div>

          <div class="actions">
            <button class="primary" @click="decideMirror('person')"><span class="kbd">1</span> 真人</button>
            <button class="danger" @click="decideMirror('reflection')"><span class="kbd">2</span> 倒影</button>
            <button @click="decideMirror('source')"><span class="kbd">3</span> 源（被反射的真人）</button>
            <div class="grow" />
            <button class="ghost" @click="decideMirror('UNLABELED')">清除判定</button>
          </div>
        </template>
        <div v-else class="empty" style="flex:1"><b>左侧选择一条轨迹</b></div>
      </section>
    </div>

    <!-- ═══ 玻璃掩膜 ═══ -->
    <div v-else-if="tab === 'mask'" class="body maskbody">
      <section class="maskpanel panel col">
        <div class="panel-h">
          玻璃 / 反光区域圈选
          <div class="grow" />
          <label class="sr-only" for="mc">选择机位</label>
          <select id="mc" v-model="maskCam" class="mini-sel wide">
            <option v-for="c in store.cams" :key="c.id" :value="c.id">{{ c.id }} {{ c.name }}</option>
          </select>
        </div>
        <div class="panel-b">
          <MaskEditor :cam="maskCam" @save="saveMask" />
        </div>
      </section>
      <aside class="col side-col">
        <div class="panel">
          <div class="panel-h">已保存圈选</div>
          <div class="panel-b col" style="gap:5px">
            <div v-for="c in store.cams" :key="c.id" class="row tiny">
              <b class="mono">{{ c.id }}</b>
              <span class="dim ellip">{{ c.name }}</span>
              <div class="grow" />
              <span class="tag" :class="store.labels.glass_masks[c.id] ? 'ok' : ''">
                {{ store.labels.glass_masks[c.id] ? store.labels.glass_masks[c.id].length + ' 点' : '未标注' }}
              </span>
            </div>
          </div>
        </div>
        <div class="panel">
          <div class="panel-h">为什么需要这一步</div>
          <div class="panel-b tiny dim" style="line-height:1.7">
            玻璃幕墙 / 镜面会反射行人，产生「镜像伪轨迹」，污染同机计数与跨机匹配。<br />
            圈出反光区域后，脚点落入该区域的检测会被标红提示。<br />
            316（右侧玻璃墙）与 323（玻璃反射）已存草稿，可在此基础上修订。
          </div>
        </div>
      </aside>
    </div>

    <!-- ═══ 属性索引（全实体检索）═══ -->
    <div v-else-if="tab === 'attrindex'" class="attrwrap">
      <AttrIndex />
    </div>

    <!-- ═══ 图片试跑（上传一张图 → 调远程算法容器）═══ -->
    <div v-else-if="tab === 'infer'" class="attrwrap">
      <AttrInfer />
    </div>

    <!-- ═══ 人员属性标注 ═══ -->
    <div v-else class="attrwrap">
      <AttrAnnotator />
    </div>
  </div>
</template>

<style scoped>
.ann { display: flex; flex-direction: column; height: 100%; min-height: 0; }
.tabs {
  flex: none; display: flex; align-items: center; gap: var(--sp-3);
  padding: var(--sp-2) var(--sp-3); border-bottom: 1px solid var(--line); background: var(--surface-1);
  overflow-x: auto;
}
.tabs::-webkit-scrollbar { height: 0; }
.body { flex: 1; min-height: 0; padding: var(--sp-3); display: grid; gap: var(--sp-3); }
.grid { grid-template-columns: 288px minmax(0, 1fr); }
.maskbody { grid-template-columns: minmax(0, 1fr) 288px; }
.list { min-height: 0; }
.filters { padding: var(--sp-2) var(--sp-3); border-bottom: 1px solid var(--line); }
.mini-sel { width: 104px; min-height: 24px; padding: 2px 6px; font-size: var(--text-xs); flex: none; }
.mini-sel.wide { width: 168px; }
.pitem {
  flex: none;   /* 受限 flex 列中不可压缩，否则相邻项内容重叠 */
  display: flex; flex-direction: column; align-items: stretch; gap: 2px;
  width: 100%; padding: var(--sp-2) var(--sp-3); text-align: left;
  background: transparent; border: 0; border-bottom: 1px solid var(--line); border-radius: 0;
}
@media (hover: hover) { .pitem:hover { background: var(--surface-2); } }
.pitem[aria-selected='true'] { background: var(--surface-3); box-shadow: inset 2px 0 0 var(--accent); }

.review { min-height: 0; gap: var(--sp-3); overflow: auto; }
.review-head { display: flex; align-items: center; gap: var(--sp-2); }
.bigpair { font-size: var(--text-md); font-weight: 650; }
.frames { align-items: flex-start; gap: var(--sp-3); }
.side { width: 268px; flex: none; }
.side-col { width: 288px; min-height: 0; gap: var(--sp-3); }
.tmini-h { display: flex; align-items: center; gap: var(--sp-2); padding: 5px var(--sp-2); }
.tmini svg { display: block; width: 100%; aspect-ratio: 16/9; background: var(--stage); }
.mgrid path { stroke: var(--line-2); stroke-width: .5; opacity: .4; }
.mframe { fill: none; stroke: var(--line-2); stroke-width: .75; opacity: .9; }
.mtraj { fill: none; stroke-width: 1.7; vector-effect: non-scaling-stroke; stroke-linejoin: round; stroke-linecap: round; }
.mdot { stroke: rgba(0, 0, 0, .45); stroke-width: .6; }
.mring { fill: none; stroke-width: 1; stroke-dasharray: 3 2.5; opacity: .5; vector-effect: non-scaling-stroke; }
.mdot.start { fill: var(--ok); }
.mdot.end { fill: var(--danger); }
.maskpanel { min-height: 0; overflow: auto; }
.attrwrap { flex: 1; min-height: 0; display: flex; }
.attrwrap > * { flex: 1; min-width: 0; }

/* 轨迹同人复核：一行 = 一条轨（全景缩略图 + 3 帧裁图横向条带） */
.trow { display: flex; gap: var(--sp-2); align-items: flex-start; position: relative; }
.tfull { flex: none; width: 150px; display: flex; flex-direction: column; gap: 3px; }
.tfull img { width: 150px; border-radius: var(--r-sm); border: 1px solid var(--line); display: block; }
.strips {
  flex: 1; min-width: 0; display: flex; gap: 4px; overflow-x: auto;
  padding: 4px; background: var(--stage); border: 1px solid var(--line); border-radius: var(--r-sm);
}
.strips img { height: 176px; border-radius: 3px; display: block; flex: none; }
.tlabel {
  position: absolute; right: 6px; top: 4px; padding: 0 5px; border-radius: 3px;
  background: color-mix(in oklch, var(--bg) 72%, transparent); letter-spacing: .08em;
}
.ttab { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }

/* ── 位置合并判读 ── */
.mvwrap { display: flex; justify-content: center; align-items: flex-start; overflow: auto; max-height: 62vh; }
.mvimg { max-width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--line); background: #181818; }

/* ── 簇纯度核验 ───────────────────────────────────────────── */
.cvhint {
  color: var(--fg-3); background: var(--surface-2);
  border: 1px solid var(--line); border-radius: var(--r-sm); padding: 6px 10px;
}
.cvhint .cvo { color: var(--danger); }
.cvhint .cvu { color: var(--warn); }
.cvgrid {
  display: grid; gap: var(--sp-2);
  grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
  align-content: start;
}
.cvitem {
  border: 1px solid var(--line); border-radius: var(--r-sm);
  background: var(--surface-2); overflow: hidden; cursor: pointer;
  transition: border-color .12s, box-shadow .12s;
}
.cvitem:hover { border-color: var(--line-2); }
.cvimgs { display: flex; gap: 1px; background: var(--stage); }
/* ★ 三帧等分宽度而不是按原比例：裁图宽窄差很多（近景/远景），按比例排会把行撑歪 */
.cvimgs img {
  flex: 1 1 0; min-width: 0; width: 0; height: 182px;
  object-fit: cover; display: block;
}
.cvmeta {
  display: flex; align-items: center; gap: 4px;
  padding: 3px 6px; color: var(--fg-3); flex-wrap: wrap;
}
.cvitem.out { border-color: var(--danger); box-shadow: inset 0 0 0 1px var(--danger); }
.cvitem.out .cvimgs img { filter: saturate(.45) brightness(.8); }
.cvitem.out .cvmeta { color: var(--danger); }
.cvitem.unread { border-color: var(--warn-line); opacity: .55; }
.cvitem.unread .cvmeta { color: var(--warn); }
.ttab th {
  text-align: left; font-size: var(--text-2xs); letter-spacing: .08em; text-transform: uppercase;
  color: var(--fg-3); font-weight: 600; padding: 0 6px 4px; border-bottom: 1px solid var(--line);
}
.ttab td { padding: 5px 6px; border-bottom: 1px solid var(--line); font-size: var(--text-sm); text-align: left; }
.ttab td:not(:first-child), .ttab th:not(:first-child) { text-align: right; }

.actions {
  display: flex; align-items: center; gap: var(--sp-2); flex-wrap: wrap;
  padding: var(--sp-2) var(--sp-3);
  background: var(--surface-1); border: 1px solid var(--line); border-radius: var(--r-md);
  position: sticky; bottom: 0;
}
.actions .kbd { margin-right: 4px; }

@media (max-width: 1023px) {
  .grid, .maskbody { grid-template-columns: minmax(0, 1fr); }
  .list { display: none; }
  .side, .side-col { width: auto; }
  .frames { flex-direction: column; }
  .trow { flex-direction: column; }
  .tfull { width: 100%; }
  .tfull img { width: 100%; }
}
</style>
