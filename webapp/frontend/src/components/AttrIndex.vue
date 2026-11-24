<script setup lang="ts">
/**
 * 属性索引：全实体 × 属性的检索界面。
 *
 * 与「人员属性标注」页签的区别：那边是标注池（733 条待标样本），这边是索引
 * （1202 个实体各一条结论，可直接当产品查）。
 *
 * ★ level 必须显示：字段能力随框高档位变化（下装在 S 档几乎不可判），
 *   如果把所有字段一视同仁标成「准确率」，用户会拿不可承诺的结论去做决策。
 * ★ 组合筛选是 AND：每行一个条件、各自带置信度门。后端按 level 做能力闸门，
 *   本档位不可判的字段不会参与筛选（否则 S 档也能筛出「背包=是」，与能力矩阵自相矛盾）。
 */
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { api } from '../api'

type Cond = { f: string; v: string; conf: number }

const items = ref<any[]>([])
const fields = ref<any[]>([])
const facets = ref<Record<string, Record<string, number>>>({})
const caps = ref<Record<string, Record<string, string | null>>>({})
const meta = ref<any>({})
const total = ref(0)
const nPerson = ref(0)
const matched = ref(0)
const capExcluded = ref(0)
const capExByField = ref<Record<string, number>>({})
const confExcluded = ref(0)
const dropOne = ref<{ f: string; v: string; n: number }[]>([])
const loading = ref(false)
const err = ref('')

const conds = ref<Cond[]>([])
const fCam = ref('')
const fBand = ref('')
const q = ref('')
const hideFake = ref(true)
const strongOnly = ref(false)
const limit = ref(60)

const CAM_LABEL: Record<string, string> = {
  '305': '305 · 走廊', '316': '316 · 反光', '318': '318', '320': '320',
  '323': '323 · 反光', '343': '343 · 反光'
}

// 取值重命名：模型只有「长裤 / 短裤」两类。补标 23 条疑似裙装后实测——
// 长外套把胯部盖住时，裙与短裤在画面上根本没有区别（20/23 是同一个人的 20 段素材，
// 只能看到露腿 + 及膝袜）。人工核对原片后确认那人穿的是裙——但单帧里没有这个证据，
// 所以这 23 条保留为真值、不进训练（no_train=hem_hidden）。
// 现在如实显示成「过膝 / 未过膝」，不假装能分裙和短裤。
const VAL_ZH: Record<string, Record<string, string>> = {}
// bottom_type 曾叫「下装长度 / 过膝·未过膝」——那时模型分不出裙与短裤，只能给长度。
// 新头有「裙」类且能判对（见能力表下方说明），就不再改叫法，直接用模型标签。
const LABEL_ZH: Record<string, string> = {}
function valZh(k: string, v: string) {
  return VAL_ZH[k]?.[v] || v
}
function labelOf(k: string) {
  return LABEL_ZH[k] || fields.value.find((x) => x.key === k)?.label || k
}

/**
 * 某字段在当前条件下的取值及命中数。
 * ★ 计数为 0 的取值也要列出来（置灰），否则用户会以为「戴帽子」这个字段不能用，
 *   而不是「这群人里没人戴帽子」——两者含义完全不同。
 */
function optsOf(k: string) {
  if (fieldBlocked(k)) return []
  const f = fields.value.find((x) => x.key === k)
  const cnt = facets.value[k] || {}
  const order = f ? f.opts : Object.keys(cnt)
  return order.map((v: string) => ({ v, n: cnt[v] || 0 }))
}
/** 该字段「任意取值」会命中多少：分面各取值之和 = 该字段有结论且置信达标的人数 */
function fieldN(k: string) {
  return Object.values(facets.value[k] || {}).reduce((a, b) => a + b, 0)
}
/**
 * 该字段在本档位下是否根本没有可靠结论。
 * ★ 不能用「optsOf 为空」判断：分面计数是「其他条件生效后」的数，
 *   别的条件把人筛空时 optsOf 也会空，但那是没命中，不是不可判 —— 两种情况要分开说。
 */
function fieldBlocked(k: string) {
  if (!k) return false
  const rows = caps.value[k]
  if (!rows) return false
  if (fBand.value) return !rows[fBand.value]
  return !rows.S && !rows.M && !rows.L
}

const capRows = computed(() =>
  Object.keys(caps.value).map((k) => {
    // 指标随索引一起存：holdout_acc = 域内按身份留出；ood_acc = PA-100K 域外 800 张
    const fm = (meta.value?.fields || {})[k] || {}
    return {
      key: k, label: labelOf(k), ...caps.value[k],
      holdout: fm.holdout_acc ?? null,
      ood: fm.ood_acc ?? null
    }
  })
)
const pct = (x: number | null) => (x == null ? '—' : (x * 100).toFixed(1) + '%')

async function load() {
  loading.value = true
  err.value = ''
  try {
    const d: any = await api.attrIndex(qparams())
    items.value = d.items || []
    fields.value = d.fields || []
    facets.value = d.facets || {}
    caps.value = d.capability || {}
    meta.value = d.meta || {}
    total.value = d.total || 0
    nPerson.value = d.n_person || 0
    matched.value = d.matched || 0
    capExcluded.value = d.cap_excluded || 0
    capExByField.value = d.cap_excluded_by_field || {}
    confExcluded.value = d.conf_excluded || 0
    dropOne.value = d.drop_one || []
  } catch (e: any) {
    err.value = String(e?.message || e)
  } finally {
    loading.value = false
  }
}

/**
 * 检索参数 —— 页面查询与导出共用一份。
 * ★ 导出如果自己拼参数，两个入口迟早给不同的人；这里保证「看到的就是导出的」。
 */
function qparams() {
  const cs = conds.value.filter((c) => c.f)
  return {
    conds: cs.length ? JSON.stringify(cs.map((c) => ({ f: c.f, v: c.v, conf: c.conf }))) : '',
    min_person: hideFake.value ? 0.5 : 0,
    cap: strongOnly.value ? 'ok' : 'judge',
    cam: fCam.value,
    band: fBand.value,
    q: q.value,
    limit: limit.value
  }
}
function exportUrl(fmt: 'csv' | 'zip') {
  return api.attrIndexExportUrl(qparams(), fmt)
}
/** 截图包体积预估：一张代表帧约 30 KB（实测 673 实体 / 20 MB） */
const zipMB = computed(() => Math.max(1, Math.round((matched.value * 30) / 1024)))
/** 页面只显示前 N 条，导出给的是全部 —— 差额必须说出来，否则用户以为导出=截图 */
const hiddenN = computed(() => Math.max(0, matched.value - items.value.length))

let timer: any = null
function loadSoon() {
  clearTimeout(timer)
  timer = setTimeout(load, 180)
}
onMounted(load)
watch([fCam, fBand, q, hideFake, strongOnly], load)
watch(conds, () => {
  // 同一字段只能有一条：AND 语义下「上衣颜色=黑 且 上衣颜色=白」永远为空，
  // 出现这种行只可能是改字段时撞上的，直接丢掉后者。
  const seen = new Set<string>()
  const dedup = conds.value.filter((c) => !!c.f && !seen.has(c.f) && (seen.add(c.f), true))
  if (dedup.length !== conds.value.length) {
    conds.value = dedup
    return
  }
  loadSoon()
}, { deep: true })

function addCond(f: string) {
  if (!f) return
  conds.value.push({ f, v: '', conf: 0.6 })
}
function delCond(i: number) {
  conds.value.splice(i, 1)
}
function delField(f: string) {
  conds.value = conds.value.filter((c) => c.f !== f)
}
function clearConds() {
  conds.value = []
}

function cropUrl(img: string) {
  return api.attrEntCropUrl(img)
}
function confClass(p: number) {
  return p >= 0.85 ? 'hi' : p >= 0.7 ? 'mid' : 'lo'
}
/** 卡片上要显示的、本档位可判的字段 */
function shownAttrs(it: any) {
  return (fields.value || [])
    .filter((f: any) => it.attrs?.[f.key] && it.level?.[f.key] !== 'n/a')
    .map((f: any) => ({
      key: f.key,
      label: labelOf(f.key),
      val: valZh(f.key, it.attrs[f.key]),
      p: it.prob?.[f.key] ?? 0,
      lv: it.level?.[f.key] || 'ok',
      // 这张卡为什么被筛出来：直接给命中的标签描边
      hit: conds.value.some((c) => c.f === f.key && (!c.v || c.v === it.attrs[f.key]))
    }))
}
/** 模型给了值、但本档位没有可靠证据的字段 —— 显式列出来，不让它混进结论 */
function naFields(it: any) {
  return (fields.value || [])
    .filter((f: any) => it.level?.[f.key] === 'n/a')
    .map((f: any) => labelOf(f.key))
}

// ---- 命名查询：只存「怎么问」，名单每次重算 ----
// 为什么不存结果：索引会重建（新素材/新模型），存下来的名单会过期且没人知道它过期了。
const saved = ref<any[]>([])
const qid = ref('')
const saveOn = ref(false)
const qname = ref('')
const qmsg = ref('')
const qbusy = ref(false)

async function loadQueries() {
  try {
    const d: any = await api.attrQueries()
    saved.value = d.items || []
  } catch (e: any) {
    qmsg.value = '读取命名查询失败：' + String(e?.message || e)
  }
}
function startSave() {
  const cur = saved.value.find((x) => x.id === qid.value)
  saveOn.value = true
  qname.value = cur ? cur.name : ''
  qmsg.value = ''
}
async function doSave() {
  const nm = qname.value.trim()
  if (!nm) { qmsg.value = '先起个名字'; return }
  qbusy.value = true
  try {
    const d: any = await api.attrQuerySave({
      id: qid.value,
      name: nm,
      conds: conds.value.filter((c) => c.f).map((c) => ({ f: c.f, v: c.v, conf: c.conf })),
      cam: fCam.value,
      band: fBand.value,
      cap: strongOnly.value ? 'ok' : 'judge'
    })
    qid.value = d.item.id
    saveOn.value = false
    qmsg.value = `已存「${d.item.name}」`
    await loadQueries()
  } catch (e: any) {
    qmsg.value = '保存失败：' + String(e?.message || e)
  } finally {
    qbusy.value = false
  }
}
function applyQuery() {
  const s = saved.value.find((x) => x.id === qid.value)
  if (!s) return
  conds.value = (s.conds || []).map((c: any) => ({ f: c.f, v: c.v || '', conf: Number(c.conf) || 0.6 }))
  fCam.value = s.cam || ''
  fBand.value = s.band || ''
  strongOnly.value = s.cap === 'ok'
  qmsg.value = `已套用「${s.name}」`
  api.attrQueryUsed(s.id).then(loadQueries).catch(() => {})
}
async function doDel() {
  const s = saved.value.find((x) => x.id === qid.value)
  if (!s) return
  qbusy.value = true
  try {
    await api.attrQueryDel(s.id)
    qid.value = ''
    qmsg.value = `已删除「${s.name}」`
    await loadQueries()
  } catch (e: any) {
    qmsg.value = '删除失败：' + String(e?.message || e)
  } finally {
    qbusy.value = false
  }
}
/** 下拉里写清「几条条件 + 机位/档位」，否则同名不同条件时分不清 */
function qdesc(x: any) {
  const bits = [`${(x.conds || []).length} 条`]
  if (x.cam) bits.push(x.cam)
  if (x.band) bits.push(x.band + ' 档')
  if (x.cap === 'ok') bits.push('可承诺')
  return bits.join(' · ')
}
onMounted(loadQueries)

// ---- 点开卡片：时间段 + 时间线帧 + 原片 ----
/**
 * 为什么需要这个：一张代表帧判不了人。同一个人坐姿/走动/远近，框高能差 3.76 倍，
 * 代表帧恰好落在最不像的那一帧时，看的人会直接判「不是他」。
 * 时间线帧按实体时间段均分，每帧带帧号与 mm:ss —— 想回原片核对时不用自己找。
 */
const sel = ref<number | null>(null)
const detail = ref<any>(null)
const detailErr = ref('')
const detailLoading = ref(false)
const videoOn = ref(false)

async function openDetail(i: number) {
  const it = items.value[i]
  if (!it) return
  sel.value = i
  videoOn.value = false
  detail.value = null
  detailErr.value = ''
  detailLoading.value = true
  try {
    detail.value = await api.attrEntity(it.entity)
  } catch (e: any) {
    detailErr.value = String(e?.message || e)
  } finally {
    detailLoading.value = false
  }
}
function closeDetail() {
  sel.value = null
  detail.value = null
  videoOn.value = false
}
function navDetail(d: number) {
  const next = (sel.value ?? 0) + d
  if (next < 0 || next >= items.value.length) return
  openDetail(next)
}
const videoUrl = computed(() =>
  detail.value?.video ? `${detail.value.video}#t=${detail.value.t0 ?? 0}` : ''
)
/** 弹层主体：详情到手前先用列表项撑住标题，避免打开瞬间空一行 */
const cur = computed(() => detail.value || (sel.value !== null ? items.value[sel.value] : null))
function onKey(e: KeyboardEvent) {
  if (sel.value === null) return
  if (e.key === 'Escape') closeDetail()
  else if (e.key === 'ArrowLeft') navDetail(-1)
  else if (e.key === 'ArrowRight') navDetail(1)
}
onMounted(() => window.addEventListener('keydown', onKey))
onUnmounted(() => window.removeEventListener('keydown', onKey))
</script>

<template>
  <div class="aidx">
    <div class="fbar">
      <span class="lab">条件</span>
      <div class="conds">
        <div class="crow" v-for="(c, i) in conds" :key="i">
          <select v-model="c.f" aria-label="属性字段">
            <option v-for="f in fields" :key="f.key" :value="f.key">{{ labelOf(f.key) }}</option>
          </select>
          <select v-model="c.v" :disabled="fieldBlocked(c.f)" aria-label="取值">
            <option value="">{{ fieldBlocked(c.f) ? '本档位不可判' : `任意取值（${fieldN(c.f)}）` }}</option>
            <option v-for="o in optsOf(c.f)" :key="o.v" :value="o.v" :disabled="!o.n">{{ valZh(c.f, o.v) }}（{{ o.n }}）</option>
          </select>
          <label class="rng" :title="'该条件只接受置信度不低于此值的实体'">
            ≥
            <b>{{ c.conf.toFixed(2) }}</b>
            <input type="range" min="0.5" max="0.99" step="0.01" v-model.number="c.conf" />
          </label>
          <button class="del" @click="delCond(i)" title="移除此条件">×</button>
        </div>
      </div>

      <select class="add" aria-label="添加条件" :value="''" @change="addCond(($event.target as HTMLSelectElement).value); ($event.target as HTMLSelectElement).value = ''">
        <option value="">+ 添加条件</option>
        <option v-for="f in fields" :key="f.key" :value="f.key" :disabled="conds.some((c) => c.f === f.key)">
          {{ labelOf(f.key) }}
        </option>
      </select>
      <button v-if="conds.length" class="ghost" @click="clearConds">清空</button>

      <span class="sep"></span>

      <select v-model="fCam" aria-label="机位">
        <option value="">全部机位</option>
        <option v-for="(v, k) in CAM_LABEL" :key="k" :value="k">{{ v }}</option>
      </select>

      <select v-model="fBand" aria-label="框高档位">
        <option value="">全部档位</option>
        <option value="S">S · 小 &lt;80px</option>
        <option value="M">M · 中 80–240</option>
        <option value="L">L · 大 ≥240</option>
      </select>

      <input v-model="q" class="q" placeholder="实体编号，如 305-E012" aria-label="搜索实体" />
      <label class="chk"><input type="checkbox" v-model="hideFake" /> 滤掉疑似非人</label>
      <label class="chk"><input type="checkbox" v-model="strongOnly" /> 只留可承诺字段</label>
    </div>

    <!-- 命名查询：把「怎么问」存下来，下次一键回到这个问法 -->
    <div class="qbar">
      <span class="lab">命名查询</span>
      <select v-model="qid" @change="applyQuery" aria-label="套用已存查询">
        <option value="">（未选择）</option>
        <option v-for="x in saved" :key="x.id" :value="x.id">{{ x.name }} · {{ qdesc(x) }}</option>
      </select>
      <button class="ghost" :disabled="qbusy" @click="startSave">
        {{ qid ? '覆盖保存' : '存为查询' }}
      </button>
      <template v-if="saveOn">
        <input v-model="qname" class="qname" maxlength="24" placeholder="名字，如：黑衣戴帽" aria-label="查询名" @keyup.enter="doSave" />
        <button class="ghost primary" :disabled="qbusy" @click="doSave">保存</button>
        <button class="ghost" @click="saveOn = false">取消</button>
      </template>
      <button v-if="qid" class="ghost danger" :disabled="qbusy" @click="doDel">删除</button>
      <span class="dim qmsg">{{ qmsg }}</span>
    </div>

    <div class="stat">
      <span v-if="loading">检索中…</span>
      <template v-else>
        匹配 <b>{{ matched }}</b> / 有效实体 <b>{{ nPerson }}</b>（全库 {{ total }}）
        <span v-if="capExcluded" class="capex" :title="'这些实体在该字段的框高档位下没有可靠证据（不可判，或开了「只留可承诺字段」时仅到参考档），已被排除在结果之外'">
          · 另有 <b>{{ capExcluded }}</b> 个因「档位能力不足」被排除<template v-if="Object.keys(capExByField).length">（<template v-for="(n, f) in capExByField" :key="f">{{ labelOf(f) }} {{ n }}；</template>）</template>
        </span>
        <span v-if="confExcluded" class="confex" title="取值相符、档位也能判，只是模型置信度没到滑块设的门 —— 把某个条件往左拖就能看到">
          · 另有 <b>{{ confExcluded }}</b> 个取值相符但置信度没过门（拖低滑块可看到）
        </span>
        <span v-if="meta.built" class="dim">　索引构建 {{ meta.built }}　训练 {{ meta.n_train }} 域内<template v-if="meta.n_pa"> + {{ meta.n_pa }} 公开（域内×{{ meta.dom_reps }}）</template> 条　
          is_person AUC {{ meta.is_person_auc }}</span>
      </template>
      <span v-if="err" class="err">{{ err }}</span>
    </div>

    <div class="exbar" v-if="matched && !loading">
      <span class="dim">
        导出当前条件的全部 <b>{{ matched }}</b> 条<template v-if="hiddenN">（页面只列了 {{ items.length }} 条，导出含其余 {{ hiddenN }} 条）</template>
      </span>
      <a class="btn" :href="exportUrl('csv')" download>导出 CSV</a>
      <a class="btn" :href="exportUrl('zip')" download>导出截图包 ≈{{ zipMB }} MB</a>
      <span class="dim longhint">CSV = 9 个字段的取值与置信度 + 帧号（可回原片定位）；截图包 = CSV + 每条一张代表帧 + 查询条件 + 说明</span>
    </div>

    <details class="cap">
      <summary>能承诺到什么程度（按框高档位）</summary>
      <table>
        <thead>
          <tr><th>字段</th><th>S</th><th>M</th><th>L</th><th title="域内 long35 按身份留出（索引与服务同一套头）">域内</th><th title="PA-100K 800 张，室外监控，从没参与训练">域外</th></tr>
        </thead>
        <tbody>
          <tr v-for="r in capRows" :key="r.key">
            <td>{{ r.label }}</td>
            <td :class="r.S || 'na'">{{ r.S === 'ok' ? '可承诺' : r.S === 'ref' ? '参考' : '不可判' }}</td>
            <td :class="r.M || 'na'">{{ r.M === 'ok' ? '可承诺' : r.M === 'ref' ? '参考' : '不可判' }}</td>
            <td :class="r.L || 'na'">{{ r.L === 'ok' ? '可承诺' : r.L === 'ref' ? '参考' : '不可判' }}</td>
            <td class="num">{{ pct(r.holdout) }}</td>
            <td class="num" :class="{ lowood: r.ood != null && r.ood < 0.8 }">{{ pct(r.ood) }}</td>
          </tr>
        </tbody>
      </table>
      <p class="dim note">
        「不可判」= 该框高档位下模型给不出可靠结论，<b>不参与筛选也不显示在卡片上</b>；
        「参考」= 能看但别当结论用。S 档下装两字段几乎不可判（框里看不到腿），这不是模型问题。
        年龄段已从索引中移除（三分类弱于多数类基线）。<br />
        <b>卡片上的数字是模型置信度，不是准确率</b>——某个字段整体准多少，看右边「域内 / 域外」两列：
        域内是 long35 按身份留出的准确率，域外是 PA-100K 800 张（室外监控，从没参与训练）。
        背包 / 拎东西域外召回只有 .25 / .19，即使置信度很高也只能当提示。<br />
        <b>觉得人太少，先把某个条件的置信度往下调。</b>softmax 没有校准，每道门都在吃召回：
        「黑衣 + 女」在 ≥0.70 时只剩 5 人，≥0.60 有 9 人，≥0.50 有 14 人，而未过门的一共 21 人。
        几个条件一起叠的时候，默认 0.60 仍然会漏，别把「查不到」当成「没有」。<br />
        <b>裙 / 短裤只在裙摆（裤腿）可见时分得开</b>：长外套或长上衣盖住胯部时，两者在画面上没有区别
        （补标 23 条里 19 条是这种情况，且大多是同一个人），模型会判成短裤，而且<b>置信度虚高</b>
        （实测某条 .997）。能看到裙摆的样本判得对（.748），索引里已经有「裙」这一类。
        要确认仍建议看原片里走动时的下摆。
      </p>
    </details>

    <div v-if="!items.length && !loading && !err" class="empty">
      <template v-if="conds.length && dropOne.length">
        <p>没有同时满足全部条件的实体。逐条去掉试试：</p>
        <div class="drops">
          <button v-for="d in dropOne" :key="d.f" @click="delField(d.f)">
            去掉「{{ labelOf(d.f) }}{{ d.v ? ' = ' + valZh(d.f, d.v) : ' 任意' }}」→ {{ d.n }} 个
          </button>
        </div>
        <p class="dim">或者放宽某个条件的置信下限、取消机位/档位限制。</p>
      </template>
      <template v-else>没有匹配的实体。放宽置信度、去掉取值或取消条件试试。</template>
    </div>

    <div class="grid">
      <article class="card" v-for="(it, i) in items" :key="it.entity"
               tabindex="0" role="button" :aria-label="`查看 ${it.entity} 的时间线帧与原片`"
               @click="openDetail(i)" @keydown.enter="openDetail(i)">
        <div class="ph" :class="'b' + it.band">
          <img v-if="it.crops?.length" :src="cropUrl(it.crops[0])" :alt="`${it.entity} 的代表帧`" loading="lazy" />
          <div v-else class="noimg">无图</div>
        </div>
        <div class="head">
          <span class="eid mono">{{ it.entity }}</span>
          <span class="tag" :class="it.band">{{ it.band }}</span>
          <span class="dim h">{{ Math.round(it.h) }}px</span>
        </div>
        <div v-if="it.summary" class="sumline" :title="`模型对该人的整体描述：${it.summary}`">
          {{ it.summary }}
        </div>
        <div class="tags">
          <span v-for="a in shownAttrs(it)" :key="a.key" class="attr" :class="[confClass(a.p), a.lv, { hit: a.hit }]">
            <i>{{ a.label }}</i>{{ a.val }}
            <em>{{ a.p.toFixed(2) }}</em>
          </span>
        </div>
        <div v-if="naFields(it).length" class="na" title="该框高档位下没有可靠证据，这些字段不参与筛选">
          不可判（{{ it.band }} 档）：<span v-for="(l, i) in naFields(it)" :key="l">{{ l }}<template v-if="i < naFields(it).length - 1"> · </template></span>
        </div>
        <div class="foot dim">
          帧 {{ it.n_crop }} · 像人 {{ it.is_person }}
          <span v-if="it.crops?.length > 1" class="dim">· 另有 {{ it.crops.length - 1 }} 帧</span>
        </div>
      </article>
    </div>

    <div v-if="matched > items.length" class="more">
      只显示前 {{ items.length }} 条 —— 用条件缩小范围，或
      <button @click="limit += 120; load()">再看 120 条</button>
    </div>

    <!-- 点开卡片：时间段 + 时间线帧 + 原片 -->
    <div v-if="sel !== null && cur" class="mask" @click.self="closeDetail">
      <div class="msheet" role="dialog" aria-modal="true" aria-label="实体时间线与原片">
        <header class="mhead">
          <span class="eid mono">{{ cur.entity }}</span>
          <span class="tag" :class="cur.band">{{ cur.band }}</span>
          <span class="dim" v-if="detail">{{ detail.cam }} · {{ detail.cam_name }} · 框高 {{ Math.round(detail.h) }}px</span>
          <span class="dim" v-if="detail">
            出现 {{ detail.mmss.t0 }}–{{ detail.mmss.t1 }}（{{ detail.dur }}s）· 轨迹 {{ detail.tracks.join(' / ') }}
          </span>
          <button class="mclose" @click="closeDetail" aria-label="关闭">×</button>
        </header>

        <div class="mbar">
          <button class="ghost" :disabled="sel <= 0" @click="navDetail(-1)">← 上一个</button>
          <span class="dim">{{ sel + 1 }} / {{ items.length }}</span>
          <button class="ghost" :disabled="sel >= items.length - 1" @click="navDetail(1)">下一个 →</button>
          <span class="sep"></span>
          <button class="ghost" v-if="detail" @click="videoOn = !videoOn">
            {{ videoOn ? '收起原片' : '在原片看这段' }}
          </button>
          <span class="dim kbd">←/→ 切换 · Esc 关闭</span>
        </div>

        <p v-if="detailErr" class="err">{{ detailErr }}</p>
        <p v-else-if="detailLoading && !detail" class="dim">正在从原片取帧…</p>

        <video v-if="videoOn && detail" class="vid" :src="videoUrl" controls autoplay playsinline></video>

        <div class="frames">
          <figure v-if="detail?.context" class="wide">
            <a :href="detail.context.url" target="_blank" rel="noopener" title="在新标签打开原尺寸">
              <img :src="detail.context.url" :alt="`${cur.entity} 带环境的画面`" loading="lazy" />
            </a>
            <figcaption>环境 · {{ detail.context.mmss }} · 帧 {{ detail.context.f }} · 黄框 = 这条实体（画面里可能还有别人）</figcaption>
          </figure>
          <figure v-for="f in (detail?.frames || [])" :key="f.tid + '-' + f.f">
            <a :href="f.url" target="_blank" rel="noopener" :title="`在新标签打开原尺寸（帧 ${f.f}）`">
              <img :src="f.url" :alt="`${cur.entity} 帧 ${f.f}`" loading="lazy" />
            </a>
            <figcaption>{{ f.mmss }} · 帧 {{ f.f }} · 轨 {{ f.tid }}</figcaption>
          </figure>
        </div>
        <p v-if="detail && !detail.frames.length" class="dim">
          该实体跨 [{{ detail.f0 }}, {{ detail.f1 }}] 帧，但轨迹表里没有可裁的观测点 —— 只能看代表帧。
        </p>

        <div class="mtags" v-if="detail">
          <span v-for="a in shownAttrs(detail)" :key="a.key" class="attr" :class="[confClass(a.p), a.lv]">
            <i>{{ a.label }}</i>{{ a.val }}<em>{{ a.p.toFixed(2) }}</em>
          </span>
          <span v-if="detail.na.length" class="dim">不可判（{{ detail.band }} 档）：{{ detail.na.join(' · ') }}</span>
        </div>
        <p class="dim note" v-if="detail">
          {{ detail.video_note }}。帧号与 mm:ss 是<b>段内相对时间</b>，跨机位对时请用画面 OSD。
        </p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.aidx { display: flex; flex-direction: column; gap: 0.75rem; }
.fbar {
  display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
  padding: 10px 12px; background: var(--surface-1);
  border: 1px solid var(--line); border-radius: 10px;
  position: sticky; top: 0; z-index: 5;
}
.fbar .lab { font-size: 12px; color: var(--fg-3); letter-spacing: .06em; }
.conds { display: flex; flex-wrap: wrap; gap: 6px; }
.crow {
  display: flex; align-items: center; gap: 6px;
  padding: 4px 6px 4px 8px; border: 1px solid var(--line-2); border-radius: 8px;
  background: var(--surface-2);
}
.fbar select, .fbar input.q {
  background: var(--surface-2); color: var(--fg);
  border: 1px solid var(--line); border-radius: 7px;
  padding: 5px 8px; font-size: 13px;
  width: auto; min-width: 116px; flex: 0 0 auto;
}
.crow select { min-width: 104px; padding: 3px 6px; font-size: 12px; }
.fbar input.q { width: 164px; }
.fbar select.add { min-width: 104px; border-style: dashed; color: var(--fg-2); }
.crow .del {
  flex: none; width: 20px; height: 20px; line-height: 1; padding: 0;
  background: transparent; border: 1px solid var(--line); border-radius: 50%;
  color: var(--fg-3); cursor: pointer; font-size: 13px;
}
.crow .del:hover { color: var(--danger); border-color: var(--danger); }
.ghost {
  background: transparent; color: var(--fg-2); border: 1px solid var(--line);
  border-radius: 7px; padding: 4px 9px; font-size: 12px; cursor: pointer;
}
.sep { width: 1px; height: 18px; background: var(--line); margin: 0 2px; }
.rng { display: flex; align-items: center; gap: 5px; font-size: 11px; color: var(--fg-2); }
.rng input { width: 76px; }
.rng b { color: var(--fg); font-variant-numeric: tabular-nums; }
.chk { display: flex; align-items: center; gap: 4px; font-size: 12px; color: var(--fg-2); }
.qbar {
  display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
  padding: 8px 12px; background: var(--surface-1);
  border: 1px solid var(--line); border-radius: 10px;
}
.qbar .lab { font-size: 12px; color: var(--fg-3); letter-spacing: .06em; }
.qbar select, .qbar input.qname {
  background: var(--surface-2); color: var(--fg);
  border: 1px solid var(--line); border-radius: 7px;
  padding: 4px 8px; font-size: 12px; width: auto; min-width: 92px; flex: 0 0 auto;
}
.qbar select { min-width: 208px; }
.qbar input.qname { width: 150px; }
.ghost.primary { border-color: var(--line-2); color: var(--accent); }
.ghost.danger { color: var(--danger); border-color: color-mix(in oklch, var(--danger) 40%, transparent); }
.qmsg { font-size: 11.5px; }
.stat { font-size: 13px; color: var(--fg-2); display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }
.stat b { color: var(--fg); font-variant-numeric: tabular-nums; }
.capex { color: var(--warn); }
.confex { color: var(--fg-2); }
.err { color: var(--danger); }
.dim { color: var(--fg-3); font-size: 12px; }
.exbar {
  display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
  padding: 8px 12px; background: var(--surface-1);
  border: 1px solid var(--line); border-radius: 10px;
}
.exbar .dim b { color: var(--fg); font-variant-numeric: tabular-nums; }
.btn {
  font-size: 12.5px; padding: 4px 11px; border-radius: 7px; text-decoration: none;
  background: var(--surface-2); color: var(--fg); border: 1px solid var(--line);
}
.btn:hover { border-color: color-mix(in oklch, var(--accent) 55%, transparent); color: var(--accent); }
/* 手机上这行长说明会把导出条撑到 4 行，按钮被挤出首屏；留按钮与计数就够 */
@media (max-width: 720px) {
  .exbar .longhint { display: none; }
  /* 414 px 下条件行会超 1 px（两个 104 的选择框 + 76 的滑块），收窄一点正好放下 */
  .crow select { min-width: 92px; }
  .rng input { width: 56px; }
}
.cap { background: var(--surface-1); border: 1px solid var(--line); border-radius: 10px; padding: 8px 12px; }
.cap summary { cursor: pointer; font-size: 13px; color: var(--fg-2); }
.cap table { border-collapse: collapse; margin-top: 8px; font-size: 12px; }
.cap th, .cap td { padding: 3px 10px; text-align: left; border-bottom: 1px solid var(--line); }
.cap td.ok { color: var(--ok); }
.cap td.ref { color: var(--warn); }
.cap td.na { color: var(--fg-3); }
.cap td.num { text-align: right; font-variant-numeric: tabular-nums; }
.cap td.lowood { color: var(--danger); }
.cap .note { margin: 8px 0 2px; line-height: 1.5; }
.empty { padding: 24px; text-align: center; color: var(--fg-3); font-size: 13px; }
.grid {
  display: grid; gap: 10px;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
}
.card {
  background: var(--surface-1); border: 1px solid var(--line);
  border-radius: 10px; overflow: hidden; display: flex; flex-direction: column;
}
.ph {
  height: 132px; background: var(--surface-2);
  display: flex; align-items: center; justify-content: center; overflow: hidden;
}
.ph img { max-width: 100%; max-height: 100%; object-fit: contain; image-rendering: pixelated; }
.noimg { color: var(--fg-3); font-size: 12px; }
.head { display: flex; align-items: center; gap: 6px; padding: 6px 8px 2px; }
.eid { font-size: 12px; color: var(--fg); }
.tag {
  font-size: 10px; padding: 1px 5px; border-radius: 4px;
  background: var(--surface-3); color: var(--fg-2);
}
.tag.L { background: color-mix(in oklch, var(--accent) 22%, transparent); color: var(--accent); }
.h { margin-left: auto; }
.sumline {
  padding: 3px 10px 1px; font-size: 12.5px; line-height: 1.45;
  color: color-mix(in oklch, var(--accent) 78%, var(--fg)); font-weight: 550;
  border-left: 3px solid color-mix(in oklch, var(--accent) 45%, transparent);
  margin: 0 8px; border-radius: 2px;
}
.tags { display: flex; flex-wrap: wrap; gap: 4px; padding: 4px 8px 4px; }
.attr {
  font-size: 11px; padding: 2px 6px; border-radius: 5px;
  background: var(--surface-2); color: var(--fg);
  border: 1px solid transparent; display: inline-flex; align-items: center; gap: 3px;
}
.attr i { font-style: normal; color: var(--fg-3); }
.attr em { font-style: normal; color: var(--fg-3); font-variant-numeric: tabular-nums; }
.attr.hi { border-color: color-mix(in oklch, var(--ok) 45%, transparent); }
.attr.mid { border-color: color-mix(in oklch, var(--warn) 40%, transparent); }
.attr.lo { opacity: 0.62; }
.attr.ref { border-style: dashed; }
/* 命中当前条件的标签：描边加重，一眼看出这张卡为什么被筛出来 */
.attr.hit {
  border-color: var(--accent); border-width: 1.5px;
  background: color-mix(in oklch, var(--accent) 13%, var(--surface-2));
}
.empty { padding: 24px; text-align: center; color: var(--fg-3); font-size: 13px; }
.empty p { margin: 4px 0; }
.drops { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin: 10px 0; max-width: 640px; margin-inline: auto; }
.drops button {
  background: var(--surface-2); color: var(--fg); border: 1px solid var(--line);
  border-radius: 8px; padding: 5px 11px; font-size: 12px; cursor: pointer;
}
.drops button:hover { border-color: color-mix(in oklch, var(--accent) 50%, transparent); }
.na {
  padding: 2px 8px 4px; font-size: 10.5px; line-height: 1.5;
  color: var(--fg-3); opacity: .8;
}
.foot { padding: 0 8px 7px; }
.card { cursor: pointer; }
.card:hover { border-color: color-mix(in oklch, var(--accent) 40%, var(--line)); }
.card:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

/* ---- 点开卡片的弹层 ---- */
.mask {
  position: fixed; inset: 0; z-index: 60;
  background: color-mix(in oklch, #000 58%, transparent);
  display: flex; align-items: center; justify-content: center; padding: 16px;
}
.msheet {
  width: min(1040px, 100%); max-height: 92vh; overflow: auto;
  background: var(--surface-1); border: 1px solid var(--line); border-radius: 12px;
  padding: 12px 14px 14px; display: flex; flex-direction: column; gap: 10px;
}
.mhead { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.mhead .eid { font-size: 14px; }
.mclose {
  margin-left: auto; width: 26px; height: 26px; line-height: 1; padding: 0;
  background: transparent; border: 1px solid var(--line); border-radius: 50%;
  color: var(--fg-2); cursor: pointer; font-size: 15px;
}
.mclose:hover { color: var(--danger); border-color: var(--danger); }
.mbar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; font-size: 12px; }
.mbar .ghost[disabled] { opacity: .4; cursor: default; }
.mbar .kbd { margin-left: auto; }
.vid { width: 100%; max-height: 46vh; background: #000; border-radius: 8px; }
.frames {
  display: grid; gap: 8px; align-items: start;
  grid-template-columns: repeat(auto-fill, minmax(168px, 1fr));
}
.frames figure { margin: 0; background: var(--surface-2); border-radius: 8px; overflow: hidden; }
.frames figure.wide { grid-column: 1 / -1; }
.frames figure.wide img { max-height: 50vh; }
.frames a { display: block; cursor: zoom-in; }
.frames img { width: 100%; max-height: 44vh; object-fit: contain; display: block; }
.frames figcaption {
  font-size: 10.5px; color: var(--fg-3); padding: 3px 6px;
  font-variant-numeric: tabular-nums;
}
.mtags { display: flex; flex-wrap: wrap; gap: 5px; align-items: center; }
.mtags .note { line-height: 1.5; }
@media (max-width: 720px) {
  .mask { padding: 0; align-items: flex-end; }
  .msheet { max-height: 96vh; border-radius: 12px 12px 0 0; }
  .frames { grid-template-columns: repeat(2, 1fr); }
  .mbar .kbd { display: none; }
}
.more { font-size: 13px; color: var(--fg-2); padding: 6px 2px 16px; }
.more button {
  background: var(--surface-2); color: var(--fg); border: 1px solid var(--line);
  border-radius: 6px; padding: 3px 10px; font-size: 12px; cursor: pointer;
}
</style>
