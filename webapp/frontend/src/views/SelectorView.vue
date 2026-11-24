<script setup lang="ts">
/** 运行时身份选择器：形态条件化模板 + 人脸触发的实时池。
 *
 *  三件事在界面上必须一眼能分开：
 *   1. 库（历史模板）怎么给分 —— 按 exp(-|log(h_i/h_q)|/σ) 加权，形态差多少、权重多少要看得见
 *   2. 池（人脸卡写进来的近期样本）加了什么 —— 三栏并列，而不是只报一个融合分
 *   3. 口径 —— 查询一律留一；候选来自聚类实体，不是人工真值
 */
import { ref, computed, onMounted } from 'vue'
import { api } from '../api'

const st = ref<any>(null)
const tracks = ref<any[]>([])
const faces = ref<any[]>([])
const res = ref<any>(null)
const ev = ref<any>(null)
const errMsg = ref('')
const busy = ref(false)
const status = ref('')

const sel = ref<{ cam: string; tid: number } | null>(null)
const camFilter = ref('')
const entQuery = ref('')
const scope = ref<'cam' | 'all'>('cam')
const tau = ref(0.8)
// 判定模式：abs=绝对余弦 ≥ τ；margin=顶一减顶二 ≥ m（另加绝对下限 floor）
const dmode = ref<'abs' | 'margin'>('abs')
const dmargin = ref(0.1)
const dfloor = ref(0.6)

const pSigma = ref(0.6)
const pK = ref(3)
const pMax = ref(8)
const pMinH = ref(60)

async function loadState() {
  try {
    st.value = await api.selectorState()
    const p = st.value?.params || {}
    pSigma.value = p.sigma ?? 0.6
    pK.value = p.pool_k ?? 3
    pMax.value = p.pool_max ?? 8
    pMinH.value = p.min_h ?? 60
  } catch (e: any) { errMsg.value = '选择器未就绪：' + (e?.message || e) }
}
async function loadTracks() {
  try { const r = await api.selectorTracks('', 1); tracks.value = r.tracks || [] } catch { /* ignore */ }
}
async function loadFaces() {
  try { const r = await api.selectorFaces(); faces.value = r.persons || [] } catch { /* ignore */ }
}
onMounted(async () => {
  await loadState(); await loadTracks(); await loadFaces()
  const t = tracks.value.find(x => x.n_group >= 3) || tracks.value[0]
  if (t) pick(t.cam, t.tid)
  // 深链 `#/selector?bc=1` 直接跑一遍实时写池对账（1.3 s），便于截图与交接
  if (location.hash.includes('bc=1')) runBindCost()
})

// ---- 查询 ----
async function pick(cam: string, tid: number) {
  sel.value = { cam, tid }
  await runQuery()
}
async function runQuery() {
  if (!sel.value) return
  busy.value = true; errMsg.value = ''
  try {
    res.value = await api.selectorQuery(sel.value.cam, sel.value.tid, scope.value, tau.value,
                                        dmode.value, dmargin.value, dfloor.value)
  } catch (e: any) { errMsg.value = String(e?.message || e); res.value = null }
  finally { busy.value = false }
}

// ---- 池 ----
async function anchor(cam: string, tid: number, entity: string, src = 'face', q?: number) {
  bulk.value = true; errMsg.value = ''; status.value = ''
  try {
    const r = await api.selectorWitness({ entity, cam, tid, src, q, scope: scope.value })
    if (r?.ok === false) status.value = `跳过 ${cam}#${tid}：${r.reason}`
    else status.value = `已写入 ${entity} ← ${cam}#${tid}${r.dup ? '（原本就在池里）' : ''}` +
      (r.dropped ? `；FIFO 挤出 ${r.dropped.cam}#${r.dropped.tid}` : '') +
      (r.mismatch ? ` ⚠ 该轨自身实体是 ${r.track_entity}` : '')
    await afterPoolChange()
  } catch (e: any) { errMsg.value = String(e?.message || e) }
  finally { bulk.value = false }
}
async function revoke(row: any) {
  try {
    await api.selectorRevoke({ entity: row.entity, cam: row.cam, tid: row.tid, scope: row.scope || 'cam' })
    status.value = `已撤销 ${row.cam}#${row.tid}`
    await afterPoolChange()
  } catch (e: any) { errMsg.value = String(e?.message || e) }
}
async function resetPool() {
  if (!confirm('清空整个实时池？（只清池，不动 ReID 库与聚类）')) return
  try { const r = await api.selectorReset(); status.value = `已清空 ${r.cleared} 条`; await afterPoolChange() }
  catch (e: any) { errMsg.value = String(e?.message || e) }
}
async function afterPoolChange() {
  await loadState(); await loadFaces()
  if (sel.value) await runQuery()
}
const bulk = ref(false)

// ---- 参数 ----
async function saveParams() {
  try {
    await api.selectorParams({
      sigma: Number(pSigma.value), pool_k: Number(pK.value),
      pool_max: Number(pMax.value), min_h: Number(pMinH.value)
    })
    status.value = '参数已保存（与池同文件，便于复现）'
    await afterPoolChange()
  } catch (e: any) { errMsg.value = String(e?.message || e) }
}

// ---- 自评 ----
const evBusy = ref(false)
async function runEval(split: 'loo' | 'half', mode: 'real' | 'oracle') {
  evBusy.value = true
  try {
    ev.value = await api.selectorEval({ scope: scope.value, tau: tau.value, mode, split })
  } catch (e: any) { errMsg.value = String(e?.message || e) }
  finally { evBusy.value = false }
}

// ---- 实时写池对账（自动写池能不能开） ----
const bc = ref<any>(null)
const bcBusy = ref(false)
async function runBindCost() {
  bcBusy.value = true
  try {
    bc.value = await api.selectorBindCost({ tau: tau.value })
  } catch (e: any) { errMsg.value = String(e?.message || e) }
  finally { bcBusy.value = false }
}
const bcRows = computed(() => {
  const t = bc.value?.table
  if (!t) return []
  const label: Record<string, string> = {
    history: '不写池', online: '自学习（会写错）',
    online_or: '自学习上限', face: '真实人脸绑定',
  }
  return ['history', 'online', 'online_or', 'face']
    .filter(k => t[k])
    .map(k => ({ k, name: label[k], ...t[k] }))
})

// ---- 视图 ----
// 缩略图：走 /api/selector/thumb（后端按库里 (u,v,h_med,f_rep) 在原生 1440p 上裁）。
// ★ 不能用 /api/track/thumb —— 库的 tid 与渲染 tracks 不同源，会裁到别人。
const zoom = ref('')
function thumbUrl(cam?: string | null, tid?: number | null, w = 160) {
  if (!cam || tid == null) return ''
  return `/api/selector/thumb?cam=${cam}&tid=${tid}&w=${w}`
}
function repOf(x: any) { return x?.rep || null }

const entities = computed<any[]>(() => {
  const list = st.value?.entities || []
  const q = entQuery.value.trim().toLowerCase()
  return q ? list.filter((e: any) => e.entity.toLowerCase().includes(q)) : list
})
const cams = computed(() => Array.from(new Set(tracks.value.map(t => t.cam))).sort())
const tracksOf = computed(() => {
  const list = camFilter.value ? tracks.value.filter(t => t.cam === camFilter.value) : tracks.value
  return list.slice(0, 400)
})
const cands = computed<any[]>(() => res.value?.candidates || [])
const margin = computed(() => {
  const s = cands.value.map((c: any) => c.s_both).filter((v: any) => v != null)
  if (s.length < 2) return null
  return s[0] - s[1]
})
const second = computed(() => cands.value[1]?.entity || '—')
const faceInStore = computed(() => faces.value.filter(f => f.in_store))
function f3(v: any) { return v == null ? '—' : Number(v).toFixed(3) }
function pct(v: any) { return v == null ? '—' : Number(v).toFixed(1) + '%' }
function cls(s: any) { return s == null ? 'dim' : (s >= tau.value ? 'ok' : 'warn') }
// 判定结果着色：看后端给的 accepted（两种模式语义不同，不能只看分数）
function vcls(v: any) { return !v ? 'dim' : (v.accepted ? 'ok' : 'warn') }
const maxScore = computed(() => Math.max(0.001, ...cands.value.map(c => c.s_both ?? 0)))
</script>

<template>
  <div class="sel">
    <!-- 顶部：口径 + 参数 -->
    <div class="head">
      <div class="row wrap" style="gap:6px">
        <span class="tag info">运行时选择器</span>
        <span v-if="st" class="tag mono">{{ st.n_track }} 轨 / {{ st.n_entity }} 实体 / 可评 {{ st.n_eval_track }}</span>
        <span v-if="st" class="tag mono" :class="st.n_pool ? 'ok' : ''">池 {{ st.n_pool }}</span>
        <div class="grow" />
        <span class="tiny dim">库=历史模板（形态加权）· 池=人脸卡写入的近期样本 · 查询一律留一</span>
      </div>
      <div class="row wrap" style="gap:6px;margin-top:6px">
        <span class="label">作用域</span>
        <button class="pill" :aria-pressed="scope === 'cam'" @click="scope = 'cam'; runQuery()">按机位（推荐）</button>
        <button class="pill" :aria-pressed="scope === 'all'" @click="scope = 'all'; runQuery()"
                title="跨机位外观在实测里是随机水平（AUC 0.45–0.62，区间全跨 0.5），只作对照">全局（对照）</button>
        <div class="sep" />
        <span class="label">判定</span>
        <button class="pill" :aria-pressed="dmode === 'abs'" @click="dmode = 'abs'; runQuery()"
                title="绝对余弦：顶一分 ≥ τ 才接受（论文 Table V 的协议）">余弦 ≥ τ</button>
        <button class="pill" :aria-pressed="dmode === 'margin'" @click="dmode = 'margin'; runQuery()"
                title="相对间隔：顶一减顶二 ≥ m 才接受。等误受下严格精度更强，闭集适用，开集另需下限">间隔 ≥ m</button>
        <template v-if="dmode === 'abs'">
          <input v-model.number="tau" type="number" step="0.01" min="0.5" max="0.99" style="width:72px" @change="runQuery" />
        </template>
        <template v-else>
          <input v-model.number="dmargin" type="number" step="0.01" min="0.02" max="0.4" style="width:72px" @change="runQuery" />
          <span class="label">下限</span>
          <input v-model.number="dfloor" type="number" step="0.05" min="0.3" max="0.9" style="width:68px" @change="runQuery" />
        </template>
        <div class="sep" />
        <span class="label">σ</span>
        <input v-model.number="pSigma" type="number" step="0.05" min="0.15" max="1.5" style="width:68px" />
        <span class="label">池 k</span>
        <input v-model.number="pK" type="number" step="1" min="1" max="8" style="width:56px" />
        <span class="label">容量</span>
        <input v-model.number="pMax" type="number" step="1" min="1" max="32" style="width:56px" />
        <span class="label">质量门 px</span>
        <input v-model.number="pMinH" type="number" step="10" min="0" style="width:64px" />
        <button class="pill" @click="saveParams">保存</button>
        <div class="grow" />
        <button class="pill" @click="resetPool">清空池</button>
      </div>
      <div v-if="status" class="tiny" style="color:var(--accent);margin-top:4px">{{ status }}</div>
      <div v-if="errMsg" class="tiny" style="color:var(--danger);margin-top:4px">{{ errMsg }}</div>
    </div>

    <div class="body">
      <!-- 左：身份 -->
      <aside class="panel col left">
        <div class="panel-h">
          身份
          <span class="tag mono">{{ entities.length }}</span>
        </div>
        <div style="padding:6px 8px">
          <input v-model="entQuery" placeholder="搜实体名 305-E0…" style="width:100%" />
        </div>
        <div class="scroll grow">
          <button v-for="e in entities" :key="e.entity" class="eitem"
                  :title="`${e.entity} · ${e.cams.join('/')} · ${e.n_track} 轨 · 池 ${e.n_pool}`"
                  :aria-pressed="res?.query?.entity === e.entity"
                  @click="entQuery = e.entity">
            <img v-if="repOf(e)" class="eth" :src="thumbUrl(repOf(e).cam, repOf(e).tid, 96)"
                 :alt="e.entity" loading="lazy" @click.stop="zoom = thumbUrl(repOf(e).cam, repOf(e).tid, 512)" />
            <b class="mono ellip">{{ e.entity }}</b>
            <span class="grow" />
            <span class="tiny dim mono">{{ e.cams.join('/') }}</span>
            <span class="tag mono">{{ e.n_track }}</span>
            <span v-if="e.n_pool" class="tag ok mono">池</span>
          </button>
        </div>
      </aside>

      <!-- 中：查询 -->
      <section class="col mid">
        <div class="panel col" style="min-height:0;flex:none">
          <div class="panel-h">
            查询轨
            <span v-if="sel" class="tag mono">{{ sel.cam }}#{{ sel.tid }}</span>
            <span class="tag mono">{{ tracksOf.length }}</span>
            <div class="grow" />
            <span class="tiny dim hide-sm">缩略图按库内 (u,v,h,f_rep) 裁原生 1440p，点图放大</span>
            <select v-model="camFilter" style="width:96px">
              <option value="">全部机位</option>
              <option v-for="c in cams" :key="c" :value="c">{{ c }}</option>
            </select>
          </div>
          <div class="tstrip">
            <button v-for="t in tracksOf" :key="t.cam + ':' + t.tid" class="tchip"
                    :aria-pressed="sel?.cam === t.cam && sel?.tid === t.tid"
                    :title="`${t.cam}#${t.tid} · ${t.entity} · h ${Math.round(t.h_med)}px`"
                    @click="pick(t.cam, t.tid)">
              <span class="tthumb">
                <img class="tth" :src="thumbUrl(t.cam, t.tid, 96)" loading="lazy" alt="" />
                <span class="tzoom" @click.stop="zoom = thumbUrl(t.cam, t.tid, 512)">⤢</span>
              </span>
              <span class="col" style="gap:0;align-items:flex-start;min-width:0">
                <b class="mono tiny">{{ t.cam }}#{{ t.tid }}</b>
                <span class="tiny dim mono ellip" style="max-width:118px">{{ t.entity }}</span>
                <span class="tiny dim mono">h{{ Math.round(t.h_med) }} · ×{{ t.n_group }}</span>
              </span>
            </button>
          </div>
        </div>

        <!-- 三栏判定 -->
        <div v-if="res" class="verdicts">
          <div class="vcard" :class="res.top_lib?.accepted && res.top_lib?.is_own ? 'good' : ''">
            <span class="label">仅历史库 S0</span>
            <b class="mono ellip">{{ res.top_lib?.entity || '无模板' }}</b>
            <span class="mono" :class="vcls(res.top_lib)">{{ f3(res.top_lib?.score) }}</span>
            <span class="tiny" :class="vcls(res.top_lib)">{{ res.top_lib?.entity ? (res.top_lib.accepted ? '接受' : '拒绝') : '—' }}</span>
            <span class="tiny dim">候选 {{ res.top_lib?.n ?? 0 }} · {{ sel?.cam }} 机位内</span>
          </div>
          <div class="vcard" :class="res.top_both?.accepted && res.top_both?.is_own ? 'good' : ''">
            <span class="label">库 + 池 S1</span>
            <b class="mono ellip">{{ res.top_both?.entity || '—' }}</b>
            <span class="mono" :class="vcls(res.top_both)">{{ f3(res.top_both?.score) }}</span>
            <span class="tiny" :class="vcls(res.top_both)">{{ res.top_both?.entity ? (res.top_both.accepted ? '接受' : '拒绝') : '—' }}</span>
            <span class="tiny dim">池 {{ st?.n_pool ?? 0 }} 条 · 取两者较高</span>
          </div>
          <div class="vcard" :class="res.top_pool?.accepted && res.top_pool?.is_own ? 'good' : ''">
            <span class="label">仅池 S2</span>
            <b class="mono ellip">{{ res.top_pool?.entity || '池为空' }}</b>
            <span class="mono" :class="vcls(res.top_pool)">{{ f3(res.top_pool?.score) }}</span>
            <span class="tiny" :class="vcls(res.top_pool)">{{ res.top_pool?.entity ? (res.top_pool.accepted ? '接受' : '拒绝') : '—' }}</span>
            <span class="tiny dim">top-{{ pK }} 均值 · 不用历史模板</span>
          </div>
        </div>

        <div v-if="res" class="tiny dim marginline">
          判定：<b>{{ dmode === 'abs' ? `余弦 ≥ ${tau}` : `间隔 ≥ ${dmargin}（下限 ${dfloor}）` }}</b>
          · 与次名分差 <b class="mono">{{ f3(res.top_both?.margin ?? margin) }}</b>
          <span v-if="res.top_both?.entity"> · 次名 <span class="mono">{{ second }}</span></span>
          · 本包的难处正是这个分差小（同机位不同人常落在 0.87–0.95）
        </div>

        <!-- 候选表 -->
        <div class="panel col" style="min-height:0;flex:1">
          <div class="panel-h">
            候选
            <span v-if="res" class="tag mono">{{ cands.length }}</span>
            <div class="grow" />
            <span v-if="res?.query" class="tiny dim mono">
              查询身份 {{ res.query.entity }} · h_q {{ Math.round(res.query.h_med) }}px
              <template v-if="res.query_in_pool">· 在池中</template>
            </span>
          </div>
          <div class="scroll grow">
            <table class="cand">
              <thead>
                <tr><th>#</th><th>图</th><th>实体</th><th>S0 库</th><th>S1 库+池</th><th>S2 池</th><th>模板</th><th></th></tr>
              </thead>
              <tbody>
                <tr v-for="(c, i) in cands" :key="c.entity" :class="{ own: c.is_own }">
                  <td class="mono dim">{{ i + 1 }}</td>
                  <td class="cthcell">
                    <img v-if="repOf(c)" class="cth" :src="thumbUrl(repOf(c).cam, repOf(c).tid, 128)"
                         :alt="c.entity" loading="lazy"
                         :title="`${repOf(c).cam}#${repOf(c).tid} · 点开看大图`"
                         @click="zoom = thumbUrl(repOf(c).cam, repOf(c).tid, 512)" />
                    <span v-else class="tiny dim">—</span>
                  </td>
                  <td class="mono ellip">
                    <b>{{ c.entity }}</b><span v-if="c.is_own" class="tag ok" style="margin-left:5px">自身</span>
                  </td>
                  <td class="mono" :class="cls(c.s_lib)">{{ f3(c.s_lib) }}</td>
                  <td class="mono">
                    <div class="bar"><i :style="{ width: ((c.s_both ?? 0) / maxScore * 100).toFixed(0) + '%' }" /></div>
                    <span :class="cls(c.s_both)">{{ f3(c.s_both) }}</span>
                  </td>
                  <td class="mono" :class="cls(c.s_pool)">{{ f3(c.s_pool) }}</td>
                  <td class="tiny dim mono">{{ c.n_lib }} 库 / {{ c.n_pool }} 池</td>
                  <td>
                    <button class="pill" @click="anchor(res.query.cam, res.query.tid, c.entity, 'manual', null)"
                            title="把当前查询轨写进这个身份的池（人工锚定）">锚入</button>
                  </td>
                </tr>
              </tbody>
            </table>
            <div v-if="!cands.length" class="empty" style="padding:var(--sp-4)">
              <b>这条轨没有可比的身份</b>
              <span class="tiny">它所属的「机位×实体」组只有它自己；留一后模板为空，按口径不进候选</span>
            </div>
          </div>
        </div>

        <!-- 形态解释 -->
        <div v-if="res?.top_both" class="panel col">
          <div class="panel-h">模板构成 <span class="tag mono">{{ res.top_both.entity }}</span>
            <div class="grow" /><span class="tiny dim">权重 = exp(−|ln(h_i/h_q)| / σ)</span>
          </div>
          <div class="scroll" style="max-height:110px">
            <div v-for="(p, i) in (cands[0]?.lib_parts || [])" :key="i" class="row tiny" style="padding:3px 10px;gap:8px">
              <img class="pth2" :src="thumbUrl(p.cam, p.tid, 96)" loading="lazy" alt=""
                   @click="zoom = thumbUrl(p.cam, p.tid, 512)" />
              <span class="mono">{{ p.cam }}#{{ p.tid }}</span>
              <span class="dim mono">h {{ p.h }}px</span>
              <span class="mono">w {{ p.w }}</span>
              <span class="dim mono">cos {{ p.cos }}</span>
            </div>
          </div>
        </div>
      </section>

      <!-- 右：人脸卡 / 池 / 自评 -->
      <aside class="col right">
        <div class="panel col">
          <div class="panel-h">
            人脸卡锚定
            <span class="tag mono">{{ faceInStore.length }}/{{ faces.length }}</span>
            <div class="grow" />
            <span class="tiny dim">人工判定过的卡，不是自动识别</span>
          </div>
          <div class="scroll" style="max-height:196px">
            <div v-for="f in faceInStore" :key="f.label" class="frow">
              <img class="fth" :src="'/' + f.img" :alt="f.label" loading="lazy" />
              <span class="col" style="gap:1px;min-width:0;align-items:flex-start">
                <span class="row" style="gap:4px">
                  <b class="mono tiny">{{ f.label }}</b>
                  <span class="tag mono">{{ f.cam }}#{{ f.tid }}</span>
                </span>
                <span class="tiny dim mono">{{ f.entity }} · q {{ f3(f.q) }} · h {{ Math.round(f.h_med) }}px</span>
              </span>
              <button v-if="f.pool_of" class="pill" disabled title="已在池中">已锚</button>
              <button v-else class="pill" @click="anchor(f.cam, f.tid, f.entity, 'face', f.q)">锚定</button>
            </div>
            <div v-if="!faceInStore.length" class="empty tiny" style="padding:var(--sp-3)">没有人脸卡落在 ReID 库里</div>
          </div>
        </div>

        <div class="panel col" style="min-height:0;flex:1">
          <div class="panel-h">
            实时池
            <span class="tag mono">{{ st?.n_pool ?? 0 }}</span>
            <div class="grow" />
            <span class="tiny dim">FIFO {{ pMax }}</span>
          </div>
          <div class="scroll grow">
            <div v-for="(p, i) in (st?.pool || [])" :key="i" class="prow">
              <img class="pth" :src="thumbUrl(p.cam, p.tid, 96)" loading="lazy" alt=""
                   :title="`${p.cam}#${p.tid} · 点开看大图`"
                   @click="zoom = thumbUrl(p.cam, p.tid, 512)" />
              <span class="col" style="gap:1px;min-width:0;align-items:flex-start">
                <span class="row" style="gap:4px">
                  <b class="mono tiny ellip">{{ p.entity }}</b>
                  <span class="tag mono">{{ p.cam }}#{{ p.tid }}</span>
                </span>
                <span class="tiny dim mono">{{ p.src }} · h {{ Math.round(p.h_med) }}px · {{ p.at || '' }}</span>
                <span v-if="p.track_entity && p.track_entity !== p.entity" class="tiny" style="color:var(--warn)">
                  轨自身实体 {{ p.track_entity }}
                </span>
              </span>
              <button class="pill" @click="revoke(p)">撤销</button>
            </div>
            <div v-if="!st?.n_pool" class="empty tiny" style="padding:var(--sp-3)">
              池为空。点右侧人脸卡的「锚定」，或候选行的「锚入」
            </div>
          </div>
        </div>

        <div class="panel col">
          <div class="panel-h">包内自评 <div class="grow" /><span class="tiny dim">实体=聚类输出，非人工真值</span></div>
          <div style="padding:6px 8px" class="evbtns">
            <button class="pill" :disabled="evBusy" @click="runEval('loo', 'real')">留一 · 真实池</button>
            <button class="pill" :disabled="evBusy" @click="runEval('loo', 'oracle')">留一 · 池上限</button>
            <button class="pill" :disabled="evBusy" @click="runEval('half', 'real')">前后半 · 真实池</button>
            <button class="pill" :disabled="evBusy" @click="runEval('half', 'oracle')">前后半 · 池上限</button>
          </div>
          <div v-if="ev" class="evbox">
            <div class="row tiny" style="gap:6px">
              <span class="tag mono">{{ ev.split }}</span>
              <span class="tag mono">{{ ev.mode }}</span>
              <span class="tag mono">n={{ ev.n }}</span>
              <span v-if="ev.n_lib_pool" class="tag mono">库池 {{ ev.n_lib_pool }}</span>
            </div>
            <div class="evrow"><span>仅历史库</span><b>{{ pct(ev.s_lib.cov_pct) }}</b><b>{{ pct(ev.s_lib.same_pct) }}</b></div>
            <div class="evrow"><span>库 + 池</span><b>{{ pct(ev.s_both.cov_pct) }}</b><b>{{ pct(ev.s_both.same_pct) }}</b></div>
            <div class="evrow"><span>仅池</span><b>{{ pct(ev.s_pool.cov_pct) }}</b><b>{{ pct(ev.s_pool.same_pct) }}</b></div>
            <div class="tiny dim" style="margin-top:4px">覆盖率 / 命中率（τ={{ ev.tau }}，绝对余弦口径；上方「间隔 ≥ m」只作用于单条判定）</div>
            <div class="tiny dim" style="margin-top:4px">{{ ev.note }}</div>
          </div>
          <div class="tiny dim" style="padding:6px 8px;border-top:1px solid var(--line);line-height:1.45">
            证据边界：本包是 10 分钟切片，没有跨时段，池与历史库冗余，池的增益在这里应接近 0。
            池增益的实测证据在 8 段（上午库 → 下午查询）：库均值 <b>56.4%</b> → 形态加权
            <b>68.2%</b> → 加池 <b>79.3%</b>（n=179 可评，人脸识别全对的上限口径）。
          </div>
        </div>

        <div class="panel col">
          <div class="panel-h">实时写池对账 <div class="grow" />
            <span class="tiny dim">自动写池默认关</span></div>
          <div style="padding:6px 8px">
            <button class="pill" :disabled="bcBusy" @click="runBindCost">
              {{ bcBusy ? '量测中…' : '跑四方对账（≈1.3 s）' }}
            </button>
          </div>
          <div v-if="bc" class="evbox">
            <div class="evrow" style="font-weight:600"><span>方案</span><b>判对</b><b>Δ</b></div>
            <div v-for="r in bcRows" :key="r.k" class="evrow">
              <span>{{ r.name }}</span>
              <b>{{ r.pct }}%</b>
              <b :class="r.delta > 0 ? 'up' : r.delta < 0 ? 'down' : ''">
                {{ r.delta > 0 ? '+' : '' }}{{ r.delta }}pt
              </b>
            </div>
            <div class="tiny dim" style="margin-top:4px">
              误写：自学习 {{ bc.table.online?.n_wrong_write }}/{{ bc.table.online?.n_pool }}，
              人脸 {{ bc.face_bind?.n ? bc.face_bind.n - bc.face_bind.hit : 0 }}/{{ bc.face_bind?.n }}
              （人脸绑定精度
              {{ bc.face_bind?.prec != null ? (bc.face_bind.prec * 100).toFixed(1) + '%' : '—' }}）
            </div>
            <div class="tiny dim" style="margin-top:4px">
              <b>结论：自动写池不开。</b>写错的条目污染别人的池，三条带自动写入的方案
              相对基线全部 ≤ +0.2pt；连按真值写的上限也只有 +0.2pt ——
              本包的池内轨本就在历史库里，没有跨时段结构，测不出池的增益。
            </div>
          </div>
        </div>
      </aside>
    </div>

    <div v-if="zoom" class="lb" @click="zoom = ''">
      <img :src="zoom" alt="" />
      <span class="lb-tip">点任意处关闭</span>
    </div>
  </div>
</template>

<style scoped>
.sel { display: flex; flex-direction: column; height: 100%; min-height: 0; }
.head { flex: none; padding: var(--sp-2) var(--sp-3); border-bottom: 1px solid var(--line); background: var(--surface-1); }
.body {
  flex: 1; display: grid; grid-template-columns: 236px minmax(0, 1fr) 320px;
  gap: var(--sp-2); padding: var(--sp-2); min-height: 0;
}
.left, .right { min-height: 0; }
.mid { min-height: 0; gap: var(--sp-2); }

.eitem {
  flex: none; display: flex; align-items: center; gap: 5px; width: 100%;
  padding: 4px var(--sp-2); text-align: left; background: transparent;
  border: 0; border-bottom: 1px solid var(--line); border-radius: 0;
}
@media (hover: hover) { .eitem:hover { background: var(--surface-2); } }
.eitem[aria-pressed='true'] { background: var(--surface-3); box-shadow: inset 2px 0 0 var(--accent); }

.tstrip { display: flex; gap: 4px; padding: 6px 8px; overflow-x: auto; overflow-y: hidden; }
.tstrip::-webkit-scrollbar { height: 5px; }
.tchip {
  flex: none; display: flex; flex-direction: row; align-items: center; gap: 6px;
  padding: 3px 7px 3px 3px; border: 1px solid var(--line); border-radius: var(--r-sm);
  background: var(--surface-2); text-align: left;
}
@media (hover: hover) { .tchip:hover { border-color: var(--line-2); } }
.tchip[aria-pressed='true'] { border-color: var(--accent-line); background: var(--accent-soft); }
.tthumb { position: relative; flex: none; display: block; line-height: 0; }
.tth { width: 32px; height: 50px; object-fit: cover; object-position: 50% 22%; border-radius: 3px; background: #000; display: block; }
.tzoom {
  position: absolute; right: -3px; bottom: -3px; font-size: 9px; line-height: 1;
  padding: 2px 3px; border-radius: 3px; background: oklch(0% 0 0 / .62); color: #fff; cursor: zoom-in;
}

.eth { width: 24px; height: 38px; object-fit: cover; object-position: 50% 22%; border-radius: 3px; background: #000; flex: none; cursor: zoom-in; }
.cthcell { width: 44px; padding-right: 2px !important; }
.cth { width: 34px; height: 58px; object-fit: cover; object-position: 50% 20%; border-radius: 3px; background: #000; cursor: zoom-in; display: block; }
.pth { width: 32px; height: 50px; object-fit: cover; object-position: 50% 22%; border-radius: 3px; background: #000; flex: none; cursor: zoom-in; }
.pth2 { width: 22px; height: 34px; object-fit: cover; object-position: 50% 22%; border-radius: 2px; background: #000; flex: none; cursor: zoom-in; }

.lb {
  position: fixed; inset: 0; z-index: 90; background: oklch(0% 0 0 / .78);
  display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px;
}
.lb img { max-width: 92vw; max-height: 86vh; border-radius: 6px; box-shadow: 0 10px 44px oklch(0% 0 0 / .55); background: #000; }
.lb-tip { color: #fff; font-size: 12px; opacity: .72; }

.verdicts { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: var(--sp-2); }
.marginline { padding: 2px 2px 0; }
.vcard {
  display: flex; flex-direction: column; gap: 2px; padding: var(--sp-2);
  background: var(--surface-1); border: 1px solid var(--line); border-radius: var(--r-md);
}
.vcard.good { border-color: var(--ok-line); background: var(--ok-soft); }

.cand { width: 100%; border-collapse: collapse; }
.cand th {
  position: sticky; top: 0; background: var(--surface-2); text-align: left;
  font-weight: 600; font-size: var(--text-2xs); color: var(--fg-3);
  padding: 4px 8px; border-bottom: 1px solid var(--line);
}
.cand td { padding: 3px 8px; border-bottom: 1px solid var(--line); font-size: var(--text-2xs); }
.cand tr.own { background: var(--accent-soft); }
.bar { height: 3px; background: var(--surface-3); border-radius: 2px; overflow: hidden; }
.bar i { display: block; height: 100%; background: var(--accent); }
.ok { color: var(--ok); }
.warn { color: var(--warn); }
.dim { color: var(--fg-3); }

.frow, .prow {
  display: flex; align-items: center; gap: 6px; padding: 4px 8px;
  border-bottom: 1px solid var(--line);
}
.fth { width: 30px; height: 38px; object-fit: cover; border-radius: var(--r-xs); background: #000; flex: none; }
.evbtns { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }
.evbtns .pill { justify-content: center; }
.evbox { padding: 6px 8px; border-top: 1px solid var(--line); }
.evrow { display: grid; grid-template-columns: 1fr auto auto; gap: 8px; font-size: var(--text-2xs); padding: 1px 0; }
.evrow b { font-variant-numeric: tabular-nums; }
.evrow b.up { color: var(--ok-line); }
.evrow b.down { color: var(--warn-line); }

@media (max-width: 1279px) {
  .body { grid-template-columns: 200px minmax(0, 1fr) 280px; }
}
@media (max-width: 1023px) {
  .body { grid-template-columns: minmax(0, 1fr); overflow-y: auto; }
  .left, .right { max-height: none; }
}
</style>
