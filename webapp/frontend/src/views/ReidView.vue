<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { store, seek, trackSpan, trackColor } from '../store'
import { api } from '../api'

const router = useRouter()

const COLORS = [
  ['white', '白'], ['black', '黑'], ['gray', '灰'], ['red', '红'], ['blue', '蓝'],
  ['navy', '藏青'], ['green', '绿'], ['yellow', '黄'], ['brown', '棕'], ['beige', '米白']
]
const GARMENTS = [
  ['top', '上衣'], ['jacket', '外套'], ['hoodie', '卫衣'], ['pants', '裤'], ['dress', '裙'], ['hat', '帽子']
]
const EXTRA = [
  ['-|backpack', '背包'], ['-|glasses', '眼镜'], ['-|mask', '口罩'], ['-|scarf', '围巾'],
  ['-|umbrella', '伞'], ['-|hair_long', '长发'], ['-|man', '男'], ['-|woman', '女'],
  ['-|clean', '保洁'], ['-|fur_coat', '毛皮领']
]
const EXAMPLES = ['白衣服灰帽子的人', '背书包的人', '藏青外套的男士', '戴眼镜穿黑T恤的男人', '保洁员在拖地']

const q = ref('')
const chips = ref<string[]>([])
const results = ref<any[]>([])
const matched = ref<boolean | null>(null)
const parts = ref<string[]>([])
const phrases = ref<string[]>([])
const busy = ref(false)
const errMsg = ref('')
const simOf = ref<any>(null)
const showGroups = ref(false)
const inputEl = ref<HTMLInputElement>()

onMounted(() => {
  window.addEventListener('keydown', onKey)
  loadEval()
})
onUnmounted(() => window.removeEventListener('keydown', onKey))
function onKey(e: KeyboardEvent) {
  if (e.key === '/' ) {
    const t = e.target as HTMLElement
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA')) return
    e.preventDefault(); inputEl.value?.focus()
  }
}
async function loadEval() {
  try { evalInfo.value = await api.reidTextEval() } catch { /* ignore */ }
}
const evalInfo = ref<any>(null)

function toggleChip(key: string) {
  const i = chips.value.indexOf(key)
  if (i >= 0) chips.value.splice(i, 1); else chips.value.push(key)
}
const chipQuery = computed(() => {
  const out: string[] = []
  for (const c of chips.value) {
    const [a, b] = c.split('|')
    if (a === '-' && b) out.push(EXTRA.find(e => e[0] === c)?.[1] || b)
    else if (b === 'color') out.push((COLORS.find(x => x[0] === a)?.[1] || a) + '衣服')
    else out.push((COLORS.find(x => x[0] === a)?.[1] || '') + (GARMENTS.find(x => x[0] === b)?.[1] || b))
  }
  return out.join('')
})

async function runSearch() {
  const query = q.value.trim() || chipQuery.value
  if (!query) return
  busy.value = true; errMsg.value = ''; simOf.value = null
  try {
    const r = await api.reidText(query, 30)
    results.value = r.results || []
    matched.value = r.matched
    parts.value = r.parts || []
    phrases.value = r.phrases || []
    q.value = query
  } catch (e: any) { errMsg.value = String(e?.message || e); results.value = [] }
  finally { busy.value = false }
}

async function findSimilar(cam: string, tid: number) {
  busy.value = true; errMsg.value = ''
  try {
    const r = await api.reidSimilar(cam, tid, 30)
    simOf.value = r.query
    results.value = r.results || []
    matched.value = true; parts.value = []; phrases.value = []
  } catch (e: any) { errMsg.value = String(e?.message || e) }
  finally { busy.value = false }
}

function focus(cam: string, tid: number) {
  const s = trackSpan(cam, tid)
  const mid = s ? (s.t0 + s.t1) / 2 : 0
  store.sel = { cam, tid }
  store.selChain = store.chainOf[`${cam}:${tid}`] || null
  store.showJourney = true
  seek(mid)
  router.push('/')
}

// ---- 跨镜头统一 ReID 分组 ----
const chains = computed(() => store.chains.chains || [])
const openChain = ref<string | null>(null)
function chainMembers(cid: string) {
  const out: { cam: string; tid: number }[] = []
  for (const key of Object.keys(store.entChain)) {
    if (store.entChain[key] !== cid) continue
    const [cam, eid] = key.split(':')
    for (const tid of (store.entities[cam]?.entities?.[eid]?.tracks || [])) out.push({ cam, tid })
  }
  return out
}
function focusChain(cid: string) {
  openChain.value = cid
  store.selChain = cid
  const m = chainMembers(cid)
  if (m.length) {
    const ch = chains.value.find((x: any) => x.id === cid)
    const start = Math.min(...(ch?.nodes || []).map((n: any) => n.t0s ?? n.t0 ?? 1e9))
    if (isFinite(start)) seek(Math.max(0, start))
    store.sel = { cam: m[0].cam, tid: m[0].tid }
  }
  router.push('/')
}
const maxScore = computed(() => Math.max(0.001, ...results.value.map(r => r.score || 0)))
</script>

<template>
  <div class="reid">
    <!-- 检索台 -->
    <div class="searchbar">
      <div class="input-wrap">
        <svg class="si" viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" fill="none"
             stroke="currentColor" stroke-width="1.6"><circle cx="7" cy="7" r="4.4" /><path d="M10.4 10.4 14 14" /></svg>
        <label class="sr-only" for="rq">语义查人</label>
        <input id="rq" ref="inputEl" v-model="q" type="search"
               placeholder="用自然语言描述目标：白衣服灰帽子的人 / 背书包的人 / 藏青外套的男士"
               @keyup.enter="runSearch" />
        <kbd class="kbd hintk">/</kbd>
      </div>
      <button class="primary" :disabled="busy" @click="runSearch">
        <span v-if="busy" class="spinner" />检索
      </button>
      <div class="grow" />
      <button class="pill only-sm" :aria-pressed="showGroups" @click="showGroups = !showGroups">身份分组</button>
      <span v-if="evalInfo" class="tag hide-sm" :class="evalInfo.hit5 >= 6 ? 'ok' : 'warn'"
            title="离线自评：已知人物正例集上的 top-5 命中率">
        离线自评 top-5 {{ evalInfo.hit5 }}/{{ evalInfo.n_query }}
      </span>
    </div>

    <!-- 示例 + 属性 -->
    <div class="strip scroll">
      <span class="label">示例</span>
      <button v-for="e in EXAMPLES" :key="e" class="pill" @click="q = e; runSearch()">{{ e }}</button>
    </div>
    <div class="strip scroll attrs">
      <span class="label">颜色</span>
      <button v-for="c in COLORS" :key="c[0]" class="pill"
              :aria-pressed="chips.includes(c[0] + '|color')" @click="toggleChip(c[0] + '|color')">{{ c[1] }}</button>
      <div class="sep" />
      <span class="label">服饰</span>
      <button v-for="g in GARMENTS" :key="g[0]" class="pill"
              :aria-pressed="chips.some(x => x.endsWith('|' + g[0]))" @click="toggleChip('any|' + g[0])">{{ g[1] }}</button>
      <div class="sep" />
      <span class="label">其他</span>
      <button v-for="e in EXTRA" :key="e[0]" class="pill"
              :aria-pressed="chips.includes(e[0])" @click="toggleChip(e[0])">{{ e[1] }}</button>
      <template v-if="chips.length">
        <div class="sep" />
        <button class="pill" style="border-color:var(--line-2)" @click="chips = []">清空</button>
        <button class="primary" @click="q = ''; runSearch()">按属性检索</button>
      </template>
    </div>

    <div class="body">
      <!-- 身份分组 -->
      <aside class="panel col groups" :class="{ open: showGroups }">
        <div class="panel-h">
          跨镜头统一 ReID
          <span class="tag info mono">{{ chains.length }}</span>
          <div class="grow" />
          <button class="icon-btn only-sm" @click="showGroups = false" aria-label="关闭">✕</button>
        </div>
        <div class="scroll grow">
          <button v-for="ch in chains" :key="ch.id" class="gitem" :aria-pressed="openChain === ch.id"
                  @click="focusChain(ch.id)">
            <span class="row">
              <b class="mono">{{ ch.id }}</b>
              <span class="tag">{{ ch.nodes.length }} 机位</span>
              <span class="grow" />
              <span class="tiny dim mono">{{ (ch.nodes || []).map((n: any) => n.cam).join('→') }}</span>
            </span>
            <span v-if="openChain === ch.id" class="row" style="gap:5px;margin-top:6px;flex-wrap:wrap">
              <img v-for="(m, i) in chainMembers(ch.id).slice(0, 10)" :key="i" class="thumb"
                   :src="api.thumbUrl(m.cam, m.tid, null, 96)" :alt="`${m.cam}#${m.tid}`"
                   :title="`${m.cam}#${m.tid} · 点击聚焦`" loading="lazy" @click.stop="focus(m.cam, m.tid)" />
            </span>
          </button>
        </div>
      </aside>

      <!-- 结果 -->
      <section class="col results">
        <div class="reshead">
          <span class="label">{{ results.length ? '检索结果' : '等待检索' }}</span>
          <span v-if="simOf" class="tag mono">以 {{ simOf.cam }}#{{ simOf.tid }} 找相似</span>
          <template v-else-if="parts.length">
            <span class="tag info mono">{{ parts.join(' + ') }}</span>
            <span class="tiny dim hide-sm">文本塔命中：{{ phrases.join(' ｜ ') }}</span>
          </template>
          <span v-else-if="matched === false && results.length" class="tag warn">未识别出属性，已退化为通用人形检索</span>
          <div class="grow" />
          <span class="tiny dim mono">{{ results.length }} 条</span>
        </div>

        <div v-if="errMsg" class="tiny" style="color:var(--danger);padding:0 var(--sp-3)">{{ errMsg }}</div>

        <div class="list scroll grow">
          <div v-if="busy" class="empty" style="padding:var(--sp-5)">
            <span class="spinner" /><b>检索中…</b>
            <span class="tiny">在 539 条全局轨迹上做 768 维余弦比对</span>
          </div>
          <div v-else-if="!results.length" class="empty" style="padding:var(--sp-6)">
            <svg class="art" viewBox="0 0 48 48" width="42" height="42" fill="none" stroke="currentColor"
                 stroke-width="1.6" stroke-linecap="round">
              <circle cx="21" cy="21" r="11" /><path d="M29.5 29.5 40 40" />
            </svg>
            <b>用文字描述你要找的人</b>
            <span class="tiny">例如「白衣服灰帽子的人」；也可以直接在左侧点开一个身份分组</span>
          </div>

          <article v-for="(r, i) in results" :key="i" class="rcard"
                   :style="{ '--c': trackColor(r.cam, r.tid) }">
            <img class="rthumb" :src="api.thumbUrl(r.cam, r.tid, r.f_rep, 160)" :alt="`${r.cam}#${r.tid}`"
                 loading="lazy" @click="focus(r.cam, r.tid)" />
            <div class="col grow" style="min-width:0;gap:3px">
              <div class="row">
                <span class="rank mono">#{{ i + 1 }}</span>
                <b class="mono">{{ r.cam }}#{{ r.tid }}</b>
                <span class="grow" />
                <span class="mono tiny">{{ r.score.toFixed(3) }}</span>
              </div>
              <div class="scorebar"><i :style="{ width: (r.score / maxScore * 100).toFixed(0) + '%' }" /></div>
              <div class="tiny dim mono">{{ r.t0.toFixed(0) }}–{{ r.t1.toFixed(0) }}s · 身高中位 {{ r.h_med }}</div>
              <div class="row" style="gap:5px;margin-top:2px">
                <button class="pill" @click="focus(r.cam, r.tid)">聚焦跳转</button>
                <button class="pill" @click="findSimilar(r.cam, r.tid)">找相似</button>
              </div>
            </div>
          </article>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.reid { display: flex; flex-direction: column; height: 100%; min-height: 0; }

.searchbar {
  flex: none; display: flex; align-items: center; gap: var(--sp-2);
  padding: var(--sp-3) var(--sp-3) var(--sp-2);
  border-bottom: 1px solid var(--line); background: var(--surface-1);
}
.input-wrap { position: relative; flex: 1; max-width: 620px; }
.si { position: absolute; left: 10px; top: 50%; transform: translateY(-50%); color: var(--fg-3); pointer-events: none; }
.input-wrap input { padding-left: 32px; padding-right: 34px; min-height: 34px; }
.hintk { position: absolute; right: 9px; top: 50%; transform: translateY(-50%); pointer-events: none; }

.strip {
  flex: none; display: flex; align-items: center; gap: 5px; flex-wrap: nowrap;
  padding: var(--sp-2) var(--sp-3); border-bottom: 1px solid var(--line); overflow-x: auto;
}
.strip::-webkit-scrollbar { height: 0; }
.strip.attrs { background: var(--surface-1); }
.strip .label { flex: none; }

.body { flex: 1; display: grid; grid-template-columns: 300px minmax(0, 1fr); gap: var(--sp-3); padding: var(--sp-3); min-height: 0; }
.groups { min-height: 0; }
.gitem {
  flex: none;   /* 受限 flex 列中不可压缩，否则相邻项内容重叠 */
  display: flex; flex-direction: column; align-items: stretch; gap: 0;
  width: 100%; padding: var(--sp-2) var(--sp-3); text-align: left;
  background: transparent; border: 0; border-bottom: 1px solid var(--line); border-radius: 0;
}
@media (hover: hover) { .gitem:hover { background: var(--surface-2); } }
.gitem[aria-pressed='true'] { background: var(--surface-3); box-shadow: inset 2px 0 0 var(--accent); }
.thumb { height: 50px; width: auto; border-radius: var(--r-xs); border: 1px solid var(--line); background: #000; cursor: pointer; }

.results { min-height: 0; gap: var(--sp-2); }
.reshead { display: flex; align-items: center; gap: var(--sp-2); flex-wrap: wrap; }
.list { display: flex; flex-direction: column; gap: 6px; padding-right: 2px; }
.rcard {
  display: flex; gap: var(--sp-3); padding: var(--sp-2);
  background: var(--surface-1); border: 1px solid var(--line);
  border-left: 3px solid var(--c, var(--accent)); border-radius: var(--r-md);
  transition: border-color var(--dur-instant) var(--ease-out-quart);
}
@media (hover: hover) { .rcard:hover { border-color: var(--line-2); } }
.rthumb { width: 92px; height: 68px; object-fit: cover; border-radius: var(--r-sm); background: #000; cursor: pointer; flex: none; }
.rank { color: var(--fg-3); font-size: var(--text-2xs); }

@media (max-width: 1023px) {
  .body { grid-template-columns: minmax(0, 1fr); }
  .groups {
    position: fixed; left: var(--sp-3); right: var(--sp-3); bottom: var(--sp-3);
    top: 30vh; z-index: calc(var(--z-drawer) + 1); display: none;
    box-shadow: var(--shadow-3);
  }
  .groups.open { display: flex; }
}
</style>
