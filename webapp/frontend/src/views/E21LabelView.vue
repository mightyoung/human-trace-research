<script setup lang="ts">
import { computed, onMounted, onBeforeUnmount, ref, watch, nextTick } from 'vue'
import { useRoute } from 'vue-router'

const days = ref<any[]>([])
const day = ref<string>('')
const curDate = ref<string>('')

/** 两级下拉：日期 → 机位。后端 /api/e21/days 已按 (date, legacy, cam) 排好序。 */
const dayGroups = computed(() => {
  const m = new Map<string, any[]>()
  for (const d of days.value) {
    const k = d.date || String(d.day).slice(0, 10)
    if (!m.has(k)) m.set(k, [])
    m.get(k)!.push(d)
  }
  return [...m.entries()].map(([date, items]) => ({ date, items }))
})
const curItems = computed(() => dayGroups.value.find(g => g.date === curDate.value)?.items || [])
const clusters = ref<any[]>([])
const busy = ref(false)
const msg = ref('')
const filter = ref<'todo' | 'done' | 'all'>('todo')
const q = ref('')
const persons = ref<Record<string, number>>({})
const nCropRej = ref(0)
const toast = ref('')          // 全屏模式下的即时回执（msg 在遮罩后面看不见）
let toastTimer: any = null
function flash(t: string) {
  toast.value = t
  clearTimeout(toastTimer)
  toastTimer = setTimeout(() => { toast.value = '' }, 1600)
}

const fs = ref(false)          // 全屏标注模式
const fsIdx = ref(0)
const zoom = ref('')
const nameBox = ref<HTMLInputElement | null>(null)
/** 排序：默认按簇号；可疑优先 = 有「时空互斥对」的排最前（唯一有物理证据的混人信号） */
const sortMode = ref<'cid' | 'risk' | 'size'>('cid')

async function jget(p: string) {
  const r = await fetch(p)
  if (!r.ok) throw new Error(`${p} -> ${r.status}`)
  return r.json()
}
async function jpost(p: string, body: any) {
  const r = await fetch(p, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  })
  if (!r.ok) throw new Error(`${p} -> ${r.status}`)
  return r.json()
}

async function loadDays() {
  try {
    const d = await jget('/api/e21/days')
    days.value = d.days || []
  } catch (e: any) { msg.value = '读取失败：' + String(e?.message || e) }
}

/** ★ 请求序号：切换日期/机位时会有并发请求，慢的旧响应后到会把新结果覆盖掉
 *  （实测：从 202 簇的段切到 26 簇的段，界面停在 202 簇的总数上）。 */
let loadSeq = 0
async function loadClusters() {
  if (!day.value) return
  const my = ++loadSeq
  busy.value = true
  try {
    const d = await jget(`/api/e21/clusters?day=${encodeURIComponent(day.value)}`)
    if (my !== loadSeq) return
    clusters.value = d.clusters || []
    const s = await jget(`/api/e21/summary?day=${encodeURIComponent(day.value)}`)
    if (my !== loadSeq) return
    persons.value = s.persons || {}
    nCropRej.value = s.n_crop_reject || 0
  } catch (e: any) {
    if (my === loadSeq) msg.value = '读取失败：' + String(e?.message || e)
  } finally { if (my === loadSeq) busy.value = false }
}
const route = useRoute()
onMounted(async () => {
  await loadDays()
  const dq = String(route.query.day || '')
  day.value = (dq && days.value.some(d => d.day === dq)) ? dq : (days.value[0]?.day || '')
  const g = dayGroups.value.find(x => x.items.some((i: any) => i.day === day.value))
  curDate.value = g?.date || dayGroups.value[0]?.date || ''
  await loadClusters()
  window.addEventListener('keydown', onKey)
  const f = String(route.query.filter || '')
  if (f === 'todo' || f === 'done' || f === 'all') filter.value = f as any
  // 支持 #/e21-label?fs=1&filter=all&cid=123 直接定位到某个簇（便于复核/投屏）
  if (route.query.fs === '1' && shown.value.length) {
    await enterFs()
    const cidq = Number(route.query.cid || 0)
    if (cidq) {
      const i = shown.value.findIndex(c => c.cid === cidq)
      if (i >= 0) fsIdx.value = i
    }
  }
})
onBeforeUnmount(() => window.removeEventListener('keydown', onKey))
watch(day, async () => {
  fs.value = false
  // 日期跟着机位走（深链或程序化切换时）；反过来由 curDate 的 watch 处理
  const g = dayGroups.value.find(x => x.items.some((i: any) => i.day === day.value))
  if (g && g.date !== curDate.value) curDate.value = g.date
  await loadClusters()
})
watch(curDate, (d) => {
  if (!d) return
  const items = dayGroups.value.find(g => g.date === d)?.items || []
  if (items.length && !items.some((i: any) => i.day === day.value)) day.value = items[0].day
})

const done = computed(() => clusters.value.filter(c => !!c.action).length)
const total = computed(() => clusters.value.length)
/** 有「同一时刻出现在两处」证据的簇数（唯一有物理依据的混人信号） */
const nSuspect = computed(() => clusters.value.filter((c: any) => (c.fm_pairs || 0) > 0).length)
const pct = computed(() => (total.value ? Math.round((done.value / total.value) * 100) : 0))
const nameList = computed(() => Object.keys(persons.value).sort((a, b) => persons.value[b] - persons.value[a]))

const shown = computed(() => {
  let a = clusters.value
  if (filter.value === 'todo') a = a.filter(c => !c.action)
  if (filter.value === 'done') a = a.filter(c => !!c.action)
  const k = q.value.trim()
  if (k) a = a.filter(c => (c.person || '').includes(k) || String(c.cid).includes(k))
  if (sortMode.value === 'risk') {
    a = [...a].sort((x: any, y: any) =>
      (y.fm_pairs || 0) - (x.fm_pairs || 0) ||
      (y.risk || 0) - (x.risk || 0) ||
      y.n_tracks - x.n_tracks || x.cid - y.cid)
  } else if (sortMode.value === 'size') {
    a = [...a].sort((x: any, y: any) => y.n_tracks - x.n_tracks || x.cid - y.cid)
  }
  return a
})
const cur = computed<any>(() => shown.value[Math.min(fsIdx.value, shown.value.length - 1)] || null)

function cropUrl(f: string) { return `/api/e21/crop/${day.value}/${f}` }
/** 少数卡片源帧是 HEVC 坏帧（拿不到有效画面，后端 404）→ 换成占位图，别显示破图图标 */
function imgErr(e: Event) {
  const t = e.target as HTMLImageElement
  if (!t || t.dataset.fallback) return
  t.dataset.fallback = '1'
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="160" height="320">
    <rect width="160" height="320" fill="#2b2b30"/>
    <text x="80" y="152" fill="#9a9aa2" font-size="13" text-anchor="middle">源帧损坏</text>
    <text x="80" y="174" fill="#71717a" font-size="11" text-anchor="middle">该时刻抽不出画面</text>
  </svg>`
  t.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg)
  t.title = '源帧损坏（HEVC 坏帧），该时刻无法抽出画面'
}
function isRej(c: any, k: number) { return (c.crop_reject || []).includes(k) }
function km(c: any, i: number) { return ((c.crops_meta || [])[i] || {}).k }
function perName(c: any, k: number) { return (c.crop_persons || {})[String(k)] || '' }

// 簇里混了两个人：单图标注与簇标签不一致（或单图之间不一致）
function mixed(c: any) {
  if (!c) return false
  const names = new Set<string>()
  for (let i = 0; i < (c.crops_meta || []).length; i++) {
    const n = perName(c, km(c, i))
    if (n) names.add(n)
  }
  if (names.size > 1) return true
  if (names.size === 1 && c.action === 'person' && !names.has(c.person)) return true
  return false
}

// ---- 多选：拆混合簇 ----
const sel = ref<number[]>([])
const selCid = ref<number | null>(null)
function isSel(c: any, k: number) { return selCid.value === c.cid && sel.value.includes(k) }
function toggleSel(c: any, k: number) {
  if (k === undefined || k === null) return
  if (selCid.value !== c.cid) { selCid.value = c.cid; sel.value = [] }
  const i = sel.value.indexOf(k)
  if (i >= 0) sel.value.splice(i, 1)
  else sel.value.push(k)
}
function selectAll(c: any) {
  if (selCid.value !== c.cid) { selCid.value = c.cid; sel.value = [] }
  const ks = (c.crops_meta || []).map((m: any) => m.k)
  sel.value = sel.value.length === ks.length ? [] : ks
}
const selC = computed<any>(() => clusters.value.find(c => c.cid === selCid.value) || null)
async function refreshCounts() {
  try {
    const s = await jget(`/api/e21/summary?day=${encodeURIComponent(day.value)}`)
    persons.value = s.persons || {}
    nCropRej.value = s.n_crop_reject || 0
  } catch { /* ignore */ }
}
async function batchCrop(c: any, action: 'person' | 'reject' | 'clear', person = '') {
  if (!c) return
  if (selCid.value !== c.cid || !sel.value.length) { flash('先在图上点选一张或多张'); return }
  const ks = [...sel.value]
  const name = (person || '').trim()
  if (action === 'person' && !name) { msg.value = '先填身份名再点「标为」'; nameBox.value?.focus(); return }
  try {
    const r = await jpost('/api/e21/crop_label', {
      day: day.value, cid: c.cid, ks, action, person: name
    })
    patch(c.cid, { crop_reject: r.crops || [], crop_persons: r.crop_persons || {} })
    dropPd(c, ks)
    sel.value = []
    flash(action === 'person' ? `${ks.length} 张标为 ${name}`
      : action === 'reject' ? `${ks.length} 张判非人` : `${ks.length} 张已撤销单图标注`)
    await refreshCounts()
  } catch (e: any) {
    const m = String(e?.message || e)
    msg.value = m.includes('400')
      ? '后端还是旧版（不认识逐图标人）：关掉控制台窗口、重新双击 start-console.cmd 再试'
      : '单图标注失败：' + m
  }
}

// 统一的判定状态：三种口径只在这里解释，模板不再各自写 if-else
function st(c: any) {
  if (!c || !c.action) return { text: '未标', cls: '' }
  if (c.action === 'person') return { text: c.person || '已命名', cls: 'ok' }
  if (c.action === 'reject') return { text: '整簇非人', cls: 'bad' }
  return { text: '已跳过', cls: 'warn' }
}

const draft = ref<Record<string, string>>({})
function draftOf(c: any) {
  if (draft.value[c.cid] === undefined) draft.value[c.cid] = c.person || ''
  return draft.value[c.cid]
}
function setDraft(c: any, v: string) { draft.value[c.cid] = v }

function patch(cid: number, patchObj: any) {
  const i = clusters.value.findIndex(x => x.cid === cid)
  if (i >= 0) clusters.value[i] = { ...clusters.value[i], ...patchObj }
}

// ---- 逐图命名：每张图一个名字，四张图可以是四个人 ----
const perDraft = ref<Record<string, string>>({})
function pdOf(c: any, i: number) {
  const key = `${c.cid}#${i}`
  if (perDraft.value[key] === undefined) perDraft.value[key] = perName(c, km(c, i))
  return perDraft.value[key]
}
function setPd(c: any, i: number, v: string) { perDraft.value[`${c.cid}#${i}`] = v }
function dropPd(c: any, ks: number[]) {
  for (const k of ks) delete perDraft.value[`${c.cid}#${k}`]
}
async function assignOne(c: any, i: number) {
  const k = km(c, i)
  const name = (pdOf(c, i) || '').trim()
  if (!name) { flash('先在框里打名字'); return }
  try {
    const r = await jpost('/api/e21/crop_label', {
      day: day.value, cid: c.cid, ks: [k], action: 'person', person: name
    })
    patch(c.cid, { crop_reject: r.crops || [], crop_persons: r.crop_persons || {} })
    flash(`第 ${i + 1} 张 → ${name}`)
    await refreshCounts()
  } catch (e: any) { msg.value = '单图命名失败：' + String(e?.message || e) }
}

async function toggleCrop(c: any, k: number) {
  if (k === undefined || k === null) return
  const on = isRej(c, k)
  try {
    const r = await jpost('/api/e21/crop_label', {
      day: day.value, cid: c.cid, k, action: on ? 'clear' : 'reject'
    })
    patch(c.cid, { crop_reject: r.crops || [], crop_persons: r.crop_persons || {} })
    dropPd(c, [k])
    await refreshCounts()
  } catch (e: any) { msg.value = '单图标记失败：' + String(e?.message || e) }
}

watch(() => cur.value?.cid, () => { sel.value = []; selCid.value = null })

async function save(c: any, action: 'person' | 'reject' | 'skip', person?: string) {
  const name = (person ?? draftOf(c) ?? '').trim()
  if (action === 'person' && !name) { msg.value = '先填身份名'; nameBox.value?.focus(); return }
  busy.value = true; msg.value = ''
  try {
    await jpost('/api/e21/label', { day: day.value, cid: c.cid, action, person: name })
    patch(c.cid, {
      action: action === 'skip' ? ((c.crop_reject || []).length ? null : undefined) : action,
      person: action === 'person' ? name : ''
    })
    if (action === 'person') persons.value = { ...persons.value, [name]: (persons.value[name] || 0) + 1 }
    // 判非人/跳过后清掉草稿，避免下次翻回来还挂着上一次输入的名字（看起来像「同名」）
    draft.value[c.cid] = action === 'person' ? name : ''
    sel.value = []
    flash(action === 'person' ? `已标为 ${name}` : action === 'reject' ? `#${c.cid} 已判整簇非人` : `#${c.cid} 已跳过`)
    if (fs.value) advance(action)
    await loadDays()
  } catch (e: any) { msg.value = '保存失败：' + String(e?.message || e) }
  finally { busy.value = false }
}
async function clear(c: any) {
  busy.value = true
  try {
    await jpost('/api/e21/label/clear', { day: day.value, cid: c.cid })
    draft.value[c.cid] = ''
    flash(`#${c.cid} 已撤销`)
    await loadClusters(); await loadDays()
  } finally { busy.value = false }
}
function useName(n: string) { if (cur.value) setDraft(cur.value, n) }

function advance(action: string) {
  // todo 过滤下保存后该项会移出列表 → 索引不动；其余情况前进一格
  if (filter.value !== 'todo' || action === 'skip') {
    fsIdx.value = Math.min(fsIdx.value + 1, Math.max(shown.value.length - 1, 0))
  }
  nextTick(() => nameBox.value?.focus())
}
function step(d: number) { fsIdx.value = Math.max(0, Math.min(fsIdx.value + d, shown.value.length - 1)) }

async function enterFs() {
  if (!shown.value.length) { msg.value = '当前筛选没有可标的簇'; return }
  fs.value = true
  fsIdx.value = 0
  try { await (document.documentElement as any).requestFullscreen?.() } catch { /* 允许被拒 */ }
  nextTick(() => nameBox.value?.focus())
}
async function exitFs() {
  fs.value = false
  try { if (document.fullscreenElement) await document.exitFullscreen() } catch { /* ignore */ }
}

function onKey(e: KeyboardEvent) {
  const tag = (e.target as HTMLElement)?.tagName
  const typing = tag === 'INPUT' || tag === 'TEXTAREA'
  if (!fs.value) {
    if (e.key === 'f' && !typing) enterFs()
    return
  }
  if (e.key === 'Escape') { exitFs(); return }
  if (e.key === 'ArrowRight') { step(1); e.preventDefault(); return }
  if (e.key === 'ArrowLeft') { step(-1); e.preventDefault(); return }
  if (typing && e.key !== 'Enter') return
  if (!cur.value) return
  if (e.key === 'Enter') { save(cur.value, 'person'); e.preventDefault() }
  else if (e.key === 'n' || e.key === 'N') save(cur.value, 'reject')
  else if (e.key === 's' || e.key === 'S') save(cur.value, 'skip')
  else if (e.key === 'a' || e.key === 'A') batchCrop(cur.value, 'person', draftOf(cur.value))
  else if (e.key === 'r' || e.key === 'R') batchCrop(cur.value, 'reject')
  else if (e.key === 'Backspace' || e.key === 'Delete' || e.key === '0') {
    if (selCid.value === cur.value?.cid && sel.value.length) batchCrop(cur.value, 'clear')
    else if (cur.value?.action) clear(cur.value)
    e.preventDefault()
  }
  else if (['1', '2', '3', '4'].includes(e.key)) {
    const c = cur.value
    const meta = c.crops_meta || []
    const kk = meta[Number(e.key) - 1]?.k
    if (kk !== undefined) toggleCrop(c, kk)
  }
}

function exportCsv() { window.open(`/api/e21/export?day=${encodeURIComponent(day.value)}`, '_blank') }
function fmtSpan(s: number) {
  if (s < 60) return `${Math.round(s)}s`
  if (s < 3600) return `${Math.round(s / 60)}m`
  return `${(s / 3600).toFixed(1)}h`
}
</script>

<template>
  <section class="wrap">
    <div class="panel">
      <div class="panel-h">
        <b>E21 全天 · 人工标簇</b>
        <span class="sub">整簇同一人就填名字回车；**每张图可以单独命名**（全屏模式每张图下方有小框，打名字回车）；也支持点图多选后批量「标为」。不是人就点「非人」。全屏按 F。</span>
        <div class="sp" />
        <button class="pill signal" :disabled="busy" @click="enterFs">全屏标注（F）</button>
        <button class="pill" :disabled="busy" @click="exportCsv">导出 CSV</button>
        <button class="pill" :disabled="busy" @click="loadClusters">刷新</button>
      </div>
      <div class="panel-b tight row" style="flex-wrap: wrap; gap: 8px">
        <select v-if="dayGroups.length" v-model="curDate" class="inp" style="width: 132px">
          <option v-for="g in dayGroups" :key="g.date" :value="g.date">{{ g.date }}</option>
        </select>
        <select v-if="curItems.length" v-model="day" class="inp" style="width: 268px">
          <option v-for="d in curItems" :key="d.day" :value="d.day">
            {{ d.cam }} · {{ d.n_labeled }}/{{ d.n_clusters }} 簇{{ d.n_suspect_todo ? ' · 可疑 ' + d.n_suspect_todo : '' }}{{ d.legacy ? '（旧版）' : '' }}
          </option>
        </select>
        <span class="tag mono">{{ done }}/{{ total }}（{{ pct }}%）</span>
        <span class="bar"><i :style="{ width: pct + '%' }" /></span>
        <div class="seg">
          <button class="pill" :class="{ on: filter === 'todo' }" @click="filter = 'todo'">未标 {{ total - done }}</button>
          <button class="pill" :class="{ on: filter === 'done' }" @click="filter = 'done'">已标 {{ done }}</button>
          <button class="pill" :class="{ on: filter === 'all' }" @click="filter = 'all'">全部 {{ total }}</button>
        </div>
        <input v-model="q" class="inp" placeholder="搜身份名 / 簇号" style="width: 170px" />
        <select v-model="sortMode" class="inp" style="width: 118px"
                title="可疑优先 = 把「同一时刻出现在两处」的簇排最前（唯一有物理证据的混人信号）">
          <option value="cid">默认顺序</option>
          <option value="risk">可疑优先</option>
          <option value="size">大簇优先</option>
        </select>
        <span v-if="nSuspect" class="tag mono" style="color:#e0707a;border-color:#e0707a">可疑 {{ nSuspect }}</span>
        <span class="tag mono">单图非人 {{ nCropRej }}</span>
        <span v-if="msg" class="tag warn">{{ msg }}</span>
      </div>
    </div>

    <div class="cols">
      <div class="col-main">
        <div v-if="!shown.length" class="empty">这一类没有簇。换筛选或点「刷新」。</div>
        <div class="cards">
          <article v-for="c in shown" :key="c.cid" class="card" :class="{ done: !!c.action, rej: c.action === 'reject' }">
            <header>
              <b>#{{ c.cid }}</b>
              <span class="tag mono">{{ c.n_tracks }} 轨</span>
              <span class="tag mono">{{ c.t0 }}–{{ c.t1 }}</span>
              <span class="tag mono">{{ fmtSpan(c.span_s) }}</span>
              <span class="tag mono">h≈{{ Math.round(c.h_med) }}</span>
              <span v-if="c.fm_pairs" class="tag mono" style="color:#e0707a;border-color:#e0707a"
                    title="簇内有多对轨在同一时刻出现在画面两处 —— 不可能是同一个人">互斥 {{ c.fm_pairs }} 对</span>
              <span v-if="mixed(c)" class="tag warn">混合</span>
              <div class="sp" />
              <span class="tag" :class="st(c).cls">{{ st(c).text }}</span>
            </header>
            <div class="thumbs">
              <div v-for="(f, i) in c.crops" :key="f" class="th"
                   :class="{ bad: isRej(c, km(c, i)), sel: isSel(c, km(c, i)) }"
                   @click="toggleSel(c, km(c, i))">
                <img :src="cropUrl(f)" loading="lazy" @dblclick="zoom = cropUrl(f)" @error="imgErr" />
                <span v-if="perName(c, km(c, i))" class="nm">{{ perName(c, km(c, i)) }}</span>
                <button class="zi" title="放大这一张" @click.stop="zoom = cropUrl(f)">⤢</button>
                <button class="x" :title="`第 ${i + 1} 张不是人`"
                        @click.stop="toggleCrop(c, km(c, i))">
                  {{ isRej(c, km(c, i)) ? '已判非人' : '非人' }}
                </button>
              </div>
              <div v-if="!c.crops.length" class="nothumb">无裁剪</div>
            </div>
            <div class="acts">
              <span class="state" :class="st(c).cls">{{ st(c).text }}</span>
              <input :value="draftOf(c)" class="inp" list="e21names"
                     :placeholder="c.action === 'reject' ? '改判为人请填名字…' : '身份名…'"
                     @input="setDraft(c, ($event.target as HTMLInputElement).value)"
                     @keyup.enter="save(c, 'person')" />
              <button class="pill ok" :disabled="busy" @click="save(c, 'person')">
                {{ c.action === 'person' ? '改名' : (c.action ? '改判为人' : '标为人') }}
              </button>
              <button class="pill" :class="{ on: c.action === 'reject' }" :disabled="busy" @click="save(c, 'reject')">
                {{ c.action === 'reject' ? '已判非人' : '非人' }}
              </button>
              <button class="pill" :class="{ on: c.action === 'skip' }" :disabled="busy" @click="save(c, 'skip')">跳过</button>
              <button v-if="c.action" class="pill" :disabled="busy" @click="clear(c)">撤销</button>
            </div>
          </article>
        </div>
      </div>

      <aside class="col-side">
        <div class="panel">
          <div class="panel-h"><b>已用身份</b><span class="sub">{{ nameList.length }} 个</span></div>
          <div class="panel-b tight">
            <div v-if="!nameList.length" class="muted">还没有。</div>
            <button v-for="n in nameList" :key="n" class="nameRow" @click="useName(n)">
              <span>{{ n }}</span><b>{{ persons[n] }}</b>
            </button>
          </div>
        </div>
      </aside>
    </div>

    <datalist id="e21names"><option v-for="n in nameList" :key="n" :value="n" /></datalist>

    <!-- 多选条：拆「一个簇里两个人」 -->
    <div v-if="sel.length && selC" class="selbar" :class="{ infs: fs }">
      <b>已选 {{ sel.length }} 张</b>
      <button class="pill ok" :disabled="busy" @click="batchCrop(selC, 'person', draftOf(selC))">
        标为「{{ draftOf(selC) || '先在上面填名字' }}」
      </button>
      <button class="pill" :disabled="busy" @click="batchCrop(selC, 'reject')">判非人</button>
      <button class="pill" :disabled="busy" @click="batchCrop(selC, 'clear')">撤销单图标注</button>
      <button class="pill" :disabled="busy" @click="selectAll(selC)">全选/反选</button>
      <button class="pill" :disabled="busy" @click="sel = []">取消</button>
    </div>

    <div v-if="zoom" class="lb" @click="zoom = ''"><img :src="zoom" /></div>

    <!-- 全屏标注 -->
    <div v-if="fs" class="fs">
      <header class="fs-h">
        <b>#{{ cur?.cid ?? '—' }}</b>
        <span class="tag st" :class="st(cur).cls">{{ st(cur).text }}</span>
        <span v-if="mixed(cur)" class="tag warn">混合</span>
        <span class="tag mono">{{ fsIdx + 1 }} / {{ shown.length }}</span>
        <span class="tag mono">{{ cur?.n_tracks }} 轨</span>
        <span class="tag mono">{{ cur?.t0 }}–{{ cur?.t1 }}</span>
        <span class="tag mono">{{ cur ? fmtSpan(cur.span_s) : '' }}</span>
        <span class="tag mono">h≈{{ Math.round(cur?.h_med || 0) }}</span>
        <span v-if="cur?.fm_pairs" class="tag mono" style="color:#e0707a;border-color:#e0707a">互斥 {{ cur.fm_pairs }} 对</span>
        <span v-if="cur && cur.crop_reject && cur.crop_reject.length" class="tag bad">
          {{ cur.crop_reject.length }} 张判非人
        </span>
        <div class="sp" />
        <span class="hint">图下小框 = 给这一张单独命名（Enter 存）· 点图多选 → A 标为 / R 非人 · Enter 整簇 · N 非人 · S 跳过 · ⌫ 撤销 · ←/→ 翻页 · Esc 退出</span>
        <button class="pill" @click="exitFs">退出</button>
      </header>

      <div v-if="cur" class="fs-body">
        <div v-for="(f, i) in cur.crops" :key="f" class="fs-th"
             :class="{ bad: isRej(cur, km(cur, i)), sel: isSel(cur, km(cur, i)),
                       named: !!perName(cur, km(cur, i)) }"
             @click="toggleSel(cur, km(cur, i))">
          <img :src="cropUrl(f)" @dblclick="zoom = cropUrl(f)" @error="imgErr" />
          <span class="fs-k">{{ i + 1 }}</span>
          <span v-if="perName(cur, km(cur, i))" class="fs-nm">{{ perName(cur, km(cur, i)) }}</span>
          <button class="fs-zoom" title="放大这一张" @click.stop="zoom = cropUrl(f)">⤢</button>
          <div class="fs-foot">
            <input class="fs-name" :value="pdOf(cur, i)" list="e21names"
                   :placeholder="`第 ${i + 1} 张：这图是谁？`"
                   @click.stop @keydown.stop
                   @input="setPd(cur, i, ($event.target as HTMLInputElement).value)"
                   @keyup.enter.stop="assignOne(cur, i)" />
            <button class="fs-x" @click.stop="toggleCrop(cur, km(cur, i))">
              {{ isRej(cur, km(cur, i)) ? '已判非人' : '不是人' }}
            </button>
          </div>
        </div>
        <div v-if="!cur.crops.length" class="empty">该簇没有裁剪图</div>
      </div>

      <footer class="fs-f">
        <span class="tag" :class="st(cur).cls">{{ st(cur).text }}</span>
        <input ref="nameBox" :value="cur ? draftOf(cur) : ''" class="inp big" list="e21names"
               :placeholder="cur && cur.action === 'reject' ? '本簇已判非人；改判为人请填名字' : '身份名，回车保存并下一条'"
               @input="cur && setDraft(cur, ($event.target as HTMLInputElement).value)"
               @keyup.enter="cur && save(cur, 'person')" />
        <button class="pill ok" :disabled="busy" @click="cur && save(cur, 'person')">
          {{ cur && cur.action === 'person' ? '改名 (Enter)' : (cur && cur.action ? '改判为人 (Enter)' : '标为人 (Enter)') }}
        </button>
        <button class="pill" :class="{ on: cur && cur.action === 'reject' }" :disabled="busy"
                @click="cur && save(cur, 'reject')">整簇非人 (N)</button>
        <button class="pill" :class="{ on: cur && cur.action === 'skip' }" :disabled="busy"
                @click="cur && save(cur, 'skip')">跳过 (S)</button>
        <button v-if="cur && cur.action" class="pill" :disabled="busy" @click="clear(cur)">撤销 (⌫)</button>
        <button class="pill" :disabled="busy" @click="step(-1)">←</button>
        <button class="pill" :disabled="busy" @click="step(1)">→</button>
        <div class="sp" />
        <div class="fs-names">
          <button v-for="n in nameList.slice(0, 8)" :key="n" class="pill" @click="useName(n)">{{ n }}</button>
        </div>
      </footer>
      <div v-if="toast" class="fs-toast">{{ toast }}</div>
    </div>
  </section>
</template>

<style scoped>
.wrap { padding: var(--sp-3); display: flex; flex-direction: column; flex-wrap: nowrap; gap: var(--sp-3); width: 100%; min-width: 0; }
.sp { flex: 1; }
.inp {
  background: var(--surface-inset); color: var(--fg); border: 1px solid var(--line);
  border-radius: 6px; padding: 4px 8px; font-size: var(--text-xs);
}
.inp.big { font-size: 15px; padding: 8px 12px; flex: 1; max-width: 520px; }
.bar { width: 160px; height: 6px; background: var(--surface-inset); border-radius: 99px; overflow: hidden; }
.bar i { display: block; height: 100%; background: var(--accent); }
.seg { display: flex; gap: 4px; }
.cols { display: grid; grid-template-columns: 1fr 260px; gap: var(--sp-3); align-items: start; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(310px, 1fr)); gap: var(--sp-2); }
.card {
  background: var(--surface-1); border: 1px solid var(--line); border-radius: 8px;
  padding: 8px; display: flex; flex-direction: column; gap: 6px;
}
.card.done { border-color: var(--ok-line); }
.card.rej { border-color: var(--danger-line); opacity: .75; }
.card header { display: flex; align-items: center; gap: 4px; flex-wrap: wrap; font-size: var(--text-xs); }
.thumbs { display: grid; grid-template-columns: repeat(4, 1fr); gap: 3px; }
.th { position: relative; cursor: pointer; }
.th img {
  width: 100%; aspect-ratio: 1 / 2; object-fit: cover; border-radius: 4px;
  background: var(--stage); display: block; user-select: none;
}
.th.sel img { outline: 3px solid var(--accent); }
.th .nm {
  position: absolute; top: 2px; left: 2px; max-width: calc(100% - 26px);
  font-size: 10px; line-height: 1.6; padding: 0 5px; border-radius: 4px;
  background: var(--ok); color: #fff; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; pointer-events: none;
}
.th .zi {
  position: absolute; top: 2px; right: 2px; width: 19px; height: 19px;
  border-radius: 4px; border: 1px solid transparent; background: oklch(0% 0 0 / .5);
  color: var(--fg); font-size: 11px; line-height: 1; cursor: pointer; opacity: .5;
}
.th:hover .zi { opacity: 1; }
.th .x {
  position: absolute; left: 2px; right: 2px; bottom: 2px;
  font-size: 10px; line-height: 1.5; padding: 1px 0; border-radius: 3px;
  background: oklch(0% 0 0 / .55); color: var(--fg); border: 1px solid transparent;
  opacity: .55; transition: opacity .12s; cursor: pointer;
}
.th:hover .x { opacity: 1; }
.th.bad img { outline: 2px solid var(--danger); filter: grayscale(.7) brightness(.7); }
.th.bad .x { opacity: 1; background: var(--danger); color: #fff; border-color: var(--danger-line); }
.nothumb { grid-column: span 4; color: var(--fg-3); font-size: var(--text-xs); padding: 12px 0; text-align: center; }
.acts { display: flex; gap: 4px; align-items: center; flex-wrap: wrap; }
.acts .inp { flex: 1; min-width: 70px; }
.acts .state {
  font-size: 11px; padding: 2px 7px; border-radius: 99px; white-space: nowrap;
  border: 1px solid var(--line); color: var(--fg-3);
}
.acts .state.ok { color: var(--ok); border-color: var(--ok-line); background: var(--ok-soft); font-weight: 600; }
.acts .state.bad { color: var(--danger); border-color: var(--danger-line); background: var(--danger-soft); font-weight: 600; }
.acts .state.warn { color: var(--warn); border-color: var(--warn-line); background: var(--warn-soft); }
.pill.ok { border-color: var(--ok-line); color: var(--ok); }
.empty { color: var(--fg-3); padding: 24px; text-align: center; }
.nameRow {
  display: flex; justify-content: space-between; width: 100%; gap: 8px;
  background: transparent; border: 0; border-bottom: 1px solid var(--line);
  color: var(--fg); padding: 4px 2px; font-size: var(--text-xs); cursor: pointer;
}
.nameRow:hover { color: var(--accent); }
.lb { position: fixed; inset: 0; background: var(--scrim); display: flex; align-items: center; justify-content: center; z-index: 90; cursor: zoom-out; }
.lb img { max-height: 88vh; max-width: 60vw; border-radius: 8px; box-shadow: var(--shadow-3); }

/* 全屏标注 */
.fs { position: fixed; inset: 0; z-index: 80; background: var(--bg); display: flex; flex-direction: column; }
.fs-h { display: flex; align-items: center; gap: 8px; padding: 8px 14px; border-bottom: 1px solid var(--line); font-size: var(--text-xs); flex-wrap: wrap; }
.fs-h .hint { color: var(--fg-3); font-size: 11px; }
.fs-h .tag.st { font-weight: 700; }
.fs-toast {
  position: fixed; left: 50%; bottom: 150px; transform: translateX(-50%);
  padding: 7px 16px; border-radius: 99px; font-size: 13px; font-weight: 600;
  background: var(--surface-2); border: 1px solid var(--ok-line); color: var(--ok);
  box-shadow: var(--shadow-2); pointer-events: none; z-index: 95;
}
.fs-body { flex: 1; display: flex; gap: 10px; padding: 12px; align-items: stretch; justify-content: center; min-height: 0; }
.fs-th { position: relative; flex: 1; max-width: 24vw; display: flex; flex-direction: column; min-height: 0; cursor: pointer; }
.fs-th img { width: 100%; flex: 1; min-height: 0; object-fit: contain; background: var(--stage); border-radius: 6px; cursor: pointer; user-select: none; }
.fs-th.sel img { outline: 4px solid var(--accent); }
.fs-th.named:not(.bad) img { box-shadow: 0 0 0 3px var(--ok); }
.fs-th.bad img { outline: 3px solid var(--danger); filter: grayscale(.75) brightness(.6); }
.fs-nm {
  position: absolute; top: 8px; right: 8px; padding: 2px 9px; border-radius: 6px;
  background: var(--ok); color: #fff; font-size: 13px; font-weight: 700;
  pointer-events: none; max-width: 55%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.fs-zoom {
  position: absolute; top: 6px; left: 42px; padding: 2px 8px; font-size: 12px;
  border-radius: 5px; border: 1px solid transparent; background: oklch(0% 0 0 / .5);
  color: var(--fg); cursor: pointer; opacity: .45;
}
.fs-th:hover .fs-zoom { opacity: 1; }
.selbar {
  position: fixed; left: 50%; bottom: 14px; transform: translateX(-50%);
  display: flex; align-items: center; gap: 8px; padding: 8px 14px;
  background: var(--surface-2); border: 1px solid var(--accent);
  border-radius: 999px; box-shadow: var(--shadow-2); z-index: 96;
  font-size: 12px; white-space: nowrap;
}
.selbar.infs { bottom: 84px; }
.fs-foot { display: flex; gap: 6px; margin-top: 6px; align-items: stretch; }
.fs-name {
  flex: 1; min-width: 0; font-size: 13px; padding: 6px 8px; border-radius: 6px;
  border: 1px solid var(--line-2); background: var(--surface-inset); color: var(--fg);
}
.fs-name::placeholder { color: var(--fg-3); font-size: 12px; }
.fs-name:focus { outline: 2px solid var(--accent); border-color: var(--accent); }
.fs-x {
  margin-top: 0; padding: 6px 10px; font-size: 12px; font-weight: 600; white-space: nowrap;
  border-radius: 6px; border: 1px solid var(--line-2); background: var(--surface-2);
  color: var(--fg); cursor: pointer;
}
.fs-x:hover { border-color: var(--danger-line); color: var(--danger); }
.fs-th.bad .fs-x { background: var(--danger); border-color: var(--danger-line); color: #fff; }
.fs-k {
  position: absolute; top: 6px; left: 6px; font-size: 11px; padding: 1px 6px;
  border-radius: 4px; background: oklch(0% 0 0 / .5); color: var(--fg);
}
.fs-f { display: flex; align-items: center; gap: 8px; padding: 10px 14px; border-top: 1px solid var(--line); }
.fs-names { display: flex; gap: 4px; flex-wrap: wrap; max-width: 46vw; justify-content: flex-end; }
@media (max-width: 1100px) { .cols { grid-template-columns: 1fr; } .fs-th { max-width: 48vw; } }
</style>
