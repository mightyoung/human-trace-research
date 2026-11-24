<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, reactive } from 'vue'
import { api } from '../api'

const schema = ref<any>({ fields: [], skip: [], band_zh: {}, canvas: [300, 500], count: 0 })
const items = ref<any[]>([])
const sel = ref(0)
const draft = reactive<Record<string, string>>({})
const saving = ref(false)
const dirty = ref(false)
const err = ref('')
const showFull = ref(false)
const showList = ref(false)

const filterCam = ref('all')
const filterBand = ref('all')
const filterStatus = ref('todo')
const filterRound = ref('all')
const roundOpts = computed(() => {
  const rs = new Set(items.value.map((i: any) => i.round || 'r1'))
  return ['r1', 'r2', 'r3', 'r4'].filter(r => rs.has(r))
})

// ---- 模型预填建议 ----
// 来源：EG2 图向量 + 逐字段逻辑回归（tmp/attr/make_prefill.py）。
// 只在该字段「有足够标注、且实测优于多数类」时才产出建议；
// pfields 里带每字段的 CV 准确率，用来告诉用户这条建议到底多可信。
const predict = ref<Record<string, any>>({})
const pfields = ref<Record<string, any>>({})
const pthr = ref(0.6)

function sug(fkey: string): { val: string; prob: number } | null {
  const it = cur.value
  if (!it) return null
  if (draft[fkey]) return null                 // 已有值（用户填过）就不再建议
  const p = predict.value[it.key]
  const v = p?.pred?.[fkey]
  const conf = p?.prob?.[fkey] ?? 0
  // ★ 必须过置信度阈值：S 档小框上模型会给出看起来很确定、实际不可判的结果
  if (!v || conf < pthr.value) return null
  return { val: v, prob: conf }
}
const sugN = computed(() => (schema.value.fields || []).filter((f: any) => !!sug(f.key)).length)

function acceptSug(fkey: string) {
  const s = sug(fkey)
  if (!s) return
  draft[fkey] = s.val
  schedule()
}

function acceptAllSug() {
  let n = 0
  for (const f of schema.value.fields || []) {
    const s = sug(f.key)
    if (s) { draft[f.key] = s.val; n++ }
  }
  if (n) schedule()
}

const cams = computed(() => Array.from(new Set(items.value.map(i => i.cam))).sort())
const bands = computed(() => Object.keys(schema.value.band_zh || {}))

function isDone(it: any) {
  return !!it.skip || (it.attrs && Object.keys(it.attrs).length > 0)
}

function match(it: any) {
  if (filterRound.value !== 'all' && (it.round || 'r1') !== filterRound.value) return false
  if (filterCam.value !== 'all' && it.cam !== filterCam.value) return false
  if (filterBand.value !== 'all' && it.band !== filterBand.value) return false
  if (filterStatus.value === 'todo' && isDone(it)) return false
  if (filterStatus.value === 'done' && !isDone(it)) return false
  return true
}

// ★ 列表用冻结快照：标注一条后它不该立刻从「待标注」里消失，否则 cur 会跳到别的条目。
//   重新筛列表靠切换筛选条件触发 refresh()。
const shownKeys = ref<string[]>([])
const shown = computed(() => {
  const m = new Map(items.value.map(i => [i.key, i]))
  return shownKeys.value.map(k => m.get(k)).filter((x: any) => !!x)
})
const cur = computed(() => shown.value[sel.value] || null)
function refresh() {
  shownKeys.value = items.value.filter(match).map(i => i.key)
  sel.value = 0
  loadDraft(cur.value)
}
watch([filterCam, filterBand, filterStatus, filterRound], refresh)

const labeledN = computed(() => items.value.filter(isDone).length)
const progress = computed(() => items.value.length
  ? Math.round(labeledN.value / items.value.length * 100) : 0)
const doneInView = computed(() => shown.value.filter(isDone).length)

function loadDraft(it: any) {
  for (const k of Object.keys(draft)) delete draft[k]
  if (it && it.attrs) for (const [k, v] of Object.entries(it.attrs)) draft[k] = v as string
}

onMounted(async () => {
  try {
    schema.value = await api.attrSchema()
    const d = await api.attrItems()
    items.value = d.items || []
    // 默认落在最新一轮且还有待标的批次上
    const rs = Array.from(new Set(items.value.map((i: any) => i.round || 'r1'))).sort()
    for (let i = rs.length - 1; i >= 0; i--) {
      const cand = items.value.filter((x: any) => (x.round || 'r1') === rs[i] && !isDone(x))
      if (cand.length) { filterRound.value = rs[i]; break }
    }
    if (!items.value.some(match)) filterStatus.value = 'all'
    refresh()
  } catch (e: any) {
    err.value = String(e?.message || e)
  }
  // 预填单独拉：拿不到不影响标注
  try {
    const p = await api.attrPredict()
    predict.value = p.items || {}
    pfields.value = p.fields || {}
    pthr.value = p.conf_threshold ?? 0.6
  } catch { /* 无建议时静默降级 */ }
  window.addEventListener('keydown', onKey)
})
// 手机端常见操作：点选后直接锁屏 / 切到别的 App。400ms 的 debounce 窗口内页面隐藏时
// 浏览器可能掐掉请求，所以在 visibilitychange 时补一次（请求带 keepalive）。
function flushOnHide() {
  if (document.visibilityState === 'hidden' && dirty.value) void flush()
}
onMounted(() => document.addEventListener('visibilitychange', flushOnHide))
onUnmounted(() => {
  window.removeEventListener('keydown', onKey)
  document.removeEventListener('visibilitychange', flushOnHide)
})

watch(sel, () => { loadDraft(cur.value); showFull.value = false })

// ---- 保存：点击即改本地 + 400ms 合并提交，切条目前强制 flush，避免丢标注 ----
let timer: any = null
function schedule() {
  dirty.value = true
  clearTimeout(timer)
  timer = setTimeout(() => { void flush() }, 400)
}

async function flush() {
  const it = cur.value
  if (!it || !dirty.value) return
  const key = it.key
  const attrs = { ...draft }
  const skip = it.skip || ''
  dirty.value = false
  saving.value = true
  try {
    const r = await api.attrSave(key, { attrs, skip })
    const idx = items.value.findIndex(x => x.key === key)
    if (idx >= 0) items.value[idx] = { ...items.value[idx], attrs, skip, at: r?.item?.at || null }
    err.value = ''
  } catch (e: any) {
    dirty.value = true
    err.value = `保存失败：${e?.message || e}`
  } finally {
    saving.value = false
  }
}

function setVal(field: string, val: string) {
  if (draft[field] === val) delete draft[field]
  else draft[field] = val
  schedule()
}

function clearAll() {
  for (const k of Object.keys(draft)) delete draft[k]
  if (cur.value) cur.value.skip = ''
  schedule()
}

async function markSkip(kind: string) {
  const it = cur.value
  if (!it) return
  for (const k of Object.keys(draft)) delete draft[k]
  it.skip = kind
  dirty.value = true
  await flush()
  await next(1)
}

async function next(d: number) {
  await flush()
  const n = shown.value.length
  if (!n) return
  sel.value = (sel.value + d + n) % n
}

function onKey(e: KeyboardEvent) {
  const t = e.target as HTMLElement
  if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA')) return
  const k = e.key.toLowerCase()
  if (k === 'j') { void next(1) }
  else if (k === 'k') { void next(-1) }
  else if (e.key === ' ') { e.preventDefault(); void next(1) }
  else if (k === 'x') { void markSkip('tiny') }
  else if (e.key === 'Escape') { showFull.value = false }
}

const fullFrameUrl = computed(() => cur.value
  ? api.frameUrl(cur.value.cam, cur.value.frame, 'all', 0, 0) : '')
// R4 批次用的是属性索引的代表帧（/attr_ent_crops/），其余批次用标注池的（/attr_crops/）
const cropUrl = computed(() => {
  if (!cur.value) return ''
  return cur.value.crop_base === 'ent'
    ? api.attrEntCropUrl(cur.value.img)
    : api.attrCropUrl(cur.value.img)
})
// R4 批次的目标单一：只补「裙」正例，界面上只留下装类型，避免一次看十几个字段拖慢标注
const visibleFields = computed(() => {
  const fs = schema.value.fields || []
  if (cur.value?.round === 'r4') return fs.filter((f: any) => f.key === 'bottom_type')
  return fs
})
const bandZh = (b: string) => (schema.value.band_zh || {})[b] || b
const skipZh = (s: string) => {
  const m: any = {}
  for (const x of schema.value.skip || []) m[x.key] = x.label
  return m[s] || s
}
</script>

<template>
  <div class="attr">
    <div class="tabs">
      <div class="seg" role="group" aria-label="筛选">
        <select v-if="roundOpts.length > 1" v-model="filterRound" class="mini-sel">
          <option value="all">全部批次</option>
          <option v-for="r in roundOpts" :key="r" :value="r">{{ r.toUpperCase() }}</option>
        </select>
        <select v-model="filterCam" class="mini-sel">
          <option value="all">全部机位</option>
          <option v-for="c in cams" :key="c" :value="c">{{ c }}</option>
        </select>
        <select v-model="filterBand" class="mini-sel">
          <option value="all">全部框高</option>
          <option v-for="b in bands" :key="b" :value="b">{{ bandZh(b) }}</option>
        </select>
        <select v-model="filterStatus" class="mini-sel">
          <option value="todo">待标注</option>
          <option value="done">已标注</option>
          <option value="all">全部</option>
        </select>
      </div>
      <div class="grow" />
      <button class="pill only-sm" :aria-pressed="showList" @click="showList = !showList">样本列表</button>
      <span class="tiny dim hide-sm">
        <span class="kbd">J</span><span class="kbd">K</span> / <span class="kbd">空格</span> 切换 ·
        <span class="kbd">X</span> 整条不可判 · 点选项即存
      </span>
      <a class="pill" :href="api.attrExportUrl('csv')">导出 CSV</a>
    </div>

    <div v-if="showList" class="scrim" @click="showList = false" />
    <div v-if="showList" class="sheet bottom" role="dialog" aria-label="样本列表">
      <div class="sheet-h">样本 <span class="tag info mono">{{ doneInView }}/{{ shown.length }}</span>
        <div class="grow" /><button class="icon-btn" @click="showList = false" aria-label="关闭">✕</button></div>
      <div class="sheet-b" style="padding:0">
        <button v-for="(it, i) in shown" :key="'m' + it.key" class="pitem"
                :aria-selected="cur === it" @click="sel = i; showList = false">
          <span class="row">
            <b class="mono">{{ it.key }}</b>
            <span class="tag">{{ it.band }}</span>
            <span class="grow" />
            <span class="tag" :class="isDone(it) ? (it.skip ? 'warn' : 'ok') : ''">
              {{ it.skip ? skipZh(it.skip) : (isDone(it) ? '已标' : '待标') }}
            </span>
          </span>
        </button>
      </div>
    </div>

    <div class="body grid">
      <aside class="list panel col">
        <div class="panel-h">
          样本
          <span class="tag info mono">{{ doneInView }}/{{ shown.length }}</span>
          <div class="grow" />
          <span class="sub mono">{{ progress }}%</span>
        </div>
        <div class="scroll grow">
          <button v-for="(it, i) in shown" :key="it.key" class="pitem"
                  :aria-selected="cur === it" @click="sel = i">
            <span class="row">
              <img class="thumb" :src="api.attrCropUrl(it.img)" :alt="it.key" loading="lazy" />
              <span class="col" style="gap:1px; min-width:0">
                <b class="mono">{{ it.key }}</b>
                <span class="tiny dim mono">h{{ it.h }} · {{ it.band }}</span>
              </span>
              <span class="grow" />
              <span v-if="isDone(it)" class="tag" :class="it.skip ? 'warn' : 'ok'">
                {{ it.skip ? '跳过' : '已标' }}
              </span>
            </span>
          </button>
          <div v-if="!shown.length" class="empty">
            <b>这个筛选下没有样本</b>
            <span class="tiny">把「状态」切到「全部」可回看已标注的条目</span>
          </div>
        </div>
      </aside>

      <section class="stage col">
        <template v-if="cur">
          <div class="stage-head">
            <button class="icon-btn" @click="next(-1)" aria-label="上一条" title="上一条 (K)">‹</button>
            <span class="bigkey mono">{{ cur.key }}</span>
            <span class="tag">{{ bandZh(cur.band) }}</span>
            <span class="tag" v-if="cur.entity">{{ cur.entity }}</span>
            <div class="grow" />
            <span class="tiny dim mono">{{ sel + 1 }} / {{ shown.length }}</span>
            <button class="icon-btn" @click="next(1)" aria-label="下一条" title="下一条 (J)">›</button>
          </div>

          <div class="shot panel">
            <img :src="cropUrl" :alt="cur.key + ' 裁剪图'" />
            <button class="ctx-btn" @click="showFull = !showFull">
              {{ showFull ? '收起全帧' : '看全帧上下文' }}
            </button>
          </div>

          <div v-if="showFull" class="fullframe panel">
            <img :src="fullFrameUrl" :alt="cur.cam + ' 全帧'" />
            <div class="tiny dim" style="padding:4px 8px">
              {{ cur.cam }} 第 {{ cur.frame }} 帧 · 用于判断手上物品 / 背包是否被裁剪框切掉
            </div>
          </div>

          <div class="meta tiny dim mono">
            原始框 {{ cur.w }}×{{ cur.h }} px · 档位 {{ cur.band }} ·
            轨迹 {{ cur.n }} 点 · 按像素放大显示，无新增细节
          </div>
        </template>
        <div v-else class="empty" style="flex:1"><b>左侧选择一条样本</b></div>
      </section>

      <aside class="fields panel col">
        <div class="panel-h">
          属性
          <span v-if="saving" class="tag info">保存中</span>
          <span v-else-if="dirty" class="tag warn">未保存</span>
          <div class="grow" />
          <span class="tiny dim">留空 = 判不了</span>
        </div>
        <div class="scroll grow fbody">
          <div v-for="f in visibleFields" :key="f.key" class="frow">
            <div class="flabel">
              <span :title="f.hint || ''">{{ f.label }}</span>
              <button v-if="sug(f.key)" class="sugbtn"
                      :title="`模型建议「${sug(f.key)!.val}」· 该字段实测准确率 ${Math.round((pfields[f.key]?.cv_acc || 0) * 100)}%`"
                      @click="acceptSug(f.key)">AI {{ sug(f.key)!.val }} ↓</button>
              <span v-else-if="f.hint" class="tiny dim hint" :title="f.hint">{{ f.hint }}</span>
            </div>
            <div class="chips" :style="{ gridTemplateColumns: `repeat(${f.cols || 4}, minmax(0,1fr))` }">
              <button v-for="o in f.opts" :key="o" class="chip"
                      :class="{ sugchip: sug(f.key)?.val === o }"
                      :aria-pressed="draft[f.key] === o"
                      @click="setVal(f.key, o)">{{ o }}</button>
            </div>
          </div>
          <div class="tiny dim" style="padding: var(--sp-2) 0">
            再点一次已选中的值可以取消。看不清就留空——留空本身是有用信息，会单独统计「可判率」。
          </div>
        </div>

        <div class="actions">
          <button class="danger" @click="markSkip('tiny')" title="框太小，判不了（快捷键 X）">太小</button>
          <button class="danger" @click="markSkip('notperson')" title="检测框不是人">非人</button>
          <button @click="markSkip('blur')" title="画面模糊">模糊</button>
          <button @click="markSkip('occluded')" title="严重遮挡">遮挡</button>
          <div class="grow" />
          <button v-if="sugN" class="sugall" @click="acceptAllSug"
                  title="把模型建议一次性填入所有空白字段">
            采纳全部 {{ sugN }}
          </button>
          <button class="ghost" @click="clearAll">清空</button>
          <button class="primary" @click="next(1)" title="空格">下一条 ›</button>
        </div>
        <div v-if="cur && cur.skip" class="tiny warnbar">
          已标记为「{{ skipZh(cur.skip) }}」——该条不计入属性统计
        </div>
        <div v-if="err" class="tiny errbar">{{ err }}</div>
      </aside>
    </div>
  </div>
</template>

<style scoped>
.attr { display: flex; flex-direction: column; height: 100%; min-height: 0; }
.tabs {
  flex: none; display: flex; align-items: center; gap: var(--sp-3);
  padding: var(--sp-2) var(--sp-3); border-bottom: 1px solid var(--line); background: var(--surface-1);
  overflow-x: auto;
}
.tabs::-webkit-scrollbar { height: 0; }
.seg { display: flex; align-items: center; gap: var(--sp-2); }
.mini-sel { width: 108px; min-height: 26px; padding: 2px 6px; font-size: var(--text-xs); flex: none; }
.body { flex: 1; min-height: 0; padding: var(--sp-3); display: grid; gap: var(--sp-3);
        grid-template-columns: 268px minmax(0, 1fr) 468px;
        grid-template-rows: minmax(0, 1fr); }

.pitem {
  flex: none; display: flex; flex-direction: column; align-items: stretch; gap: 2px;
  width: 100%; padding: 5px var(--sp-3); text-align: left;
  background: transparent; border: 0; border-bottom: 1px solid var(--line); border-radius: 0;
}
@media (hover: hover) { .pitem:hover { background: var(--surface-2); } }
.pitem[aria-selected='true'] { background: var(--surface-3); box-shadow: inset 2px 0 0 var(--accent); }
.thumb { width: 40px; height: 56px; object-fit: cover; border-radius: var(--r-xs);
         border: 1px solid var(--line); flex: none; background: var(--stage); }

.stage { min-height: 0; gap: var(--sp-2); overflow: auto; align-items: center; }
.stage-head { display: flex; align-items: center; gap: var(--sp-2); width: 100%; }
.bigkey { font-size: var(--text-md); font-weight: 650; }
.shot { position: relative; padding: var(--sp-2); display: flex; justify-content: center;
        width: 100%; box-sizing: border-box; }
.shot img { max-height: 58vh; max-width: 100%; min-height: 26vh; width: auto; height: auto;
            image-rendering: pixelated; border-radius: var(--r-sm); background: transparent; }
.ctx-btn {
  position: absolute; right: 10px; bottom: 10px; font-size: var(--text-xs);
  background: var(--surface-3); border: 1px solid var(--line-2); border-radius: var(--r-pill);
  padding: 3px 10px; color: var(--fg-2);
}
.fullframe { padding: var(--sp-2); width: 100%; }
.fullframe img { width: 100%; border-radius: var(--r-sm); }
.meta { padding-left: 2px; width: 100%; }

.fields { min-height: 0; }
.fbody { padding: var(--sp-2) var(--sp-3); }
.frow { padding: 3px 0; border-bottom: 1px dashed var(--line); }
.flabel { display: flex; align-items: baseline; gap: 6px; font-size: var(--text-xs); color: var(--fg-2); margin-bottom: 4px; min-width: 0; }
.flabel > span:first-child { flex: none; }
.flabel .hint { font-size: 10.5px; opacity: .7; white-space: nowrap; overflow: hidden;
                text-overflow: ellipsis; min-width: 0; }
.sugbtn {
  flex: none; margin-left: auto; padding: 0 5px; min-height: 17px; line-height: 15px;
  font-size: 10.5px; font-family: inherit; white-space: nowrap;
  border: 0.5px dashed var(--line-2); border-radius: var(--r-xs);
  background: transparent; color: var(--fg-2); cursor: pointer;
}
.sugbtn:hover { color: var(--fg); border-color: var(--fg-2); border-style: solid; }
.chips { display: grid; gap: 3px; }
.chip {
  min-height: 24px; padding: 1px 4px; font-size: var(--text-xs); line-height: 1.2;
  background: var(--surface-2); border: 1px solid var(--line); border-radius: var(--r-sm);
  color: var(--fg-2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
@media (hover: hover) { .chip:hover { border-color: var(--line-2); color: var(--fg); } }
.chip[aria-pressed='true'] {
  background: var(--signal); border-color: var(--signal); color: var(--ink-on-signal, #1a1a1a); font-weight: 600;
}
.chip.sugchip { border-style: dashed; border-color: var(--fg-2); color: var(--fg); }
.sugall {
  flex: none; padding: 4px 8px; font-size: var(--text-xs);
  border: 0.5px dashed var(--fg-2); background: transparent; color: var(--fg-2);
}
.sugall:hover { color: var(--fg); border-style: solid; }
.actions {
  display: flex; align-items: center; gap: 4px; flex-wrap: nowrap; overflow-x: auto;
  padding: 5px var(--sp-2); background: var(--surface-1);
  border-top: 1px solid var(--line); flex: none;
}
.actions > button { flex: none; padding: 4px 8px; font-size: var(--text-xs); }
.actions .kbd { margin-right: 4px; }
.warnbar { padding: 4px var(--sp-3); color: var(--warn); flex: none; }
.errbar { padding: 4px var(--sp-3); color: var(--danger); flex: none; }

@media (max-width: 1279px) {
  .body { grid-template-columns: 220px minmax(0, 1fr) 320px; }
}
@media (max-width: 1023px) {
  .body { grid-template-columns: minmax(0, 1fr); }
  .list { display: none; }
  .fields { max-height: 46vh; }
}
</style>
