<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { api } from '../api'

const data = ref<any>(null)
const err = ref('')
const camFilter = ref('all')
const onlyFused = ref(false)
const onlyTodo = ref(false)
const onlyLf = ref(false)
const search = ref('')
const sel = ref<any>(null)
const selVariant = ref<'single' | 'fused' | 'enh'>('single')
const lightbox = ref('')

// ---- 人工标注（独立文件 faces_labels.json，不改 GIL）----
const VERD: Record<string, { t: string; cls: string }> = {
  person: { t: '真人', cls: 'ok' },
  nonperson: { t: '非人', cls: 'bad' },
  unclear: { t: '看不清', cls: 'warn' }
}
const decisions = [
  { k: 'CONF', t: '确认', cls: 'on-conf' },
  { k: 'REJ', t: '否决', cls: 'on-rej' },
  { k: 'MAYBE', t: '存疑', cls: 'on-maybe' }
]
const lab = ref<{ persons: Record<string, any>; pairs: Record<string, any> }>({ persons: {}, pairs: {} })
const labErr = ref('')
const saving = ref(false)
const nameDraft = ref('')

// ★ 读切换（B1①）：判定唯一来源 = 台账 `/api/ops/queue`；台账不可用（远程预览）回落老接口。
//   台账 `label.data` 存的是原始对象（{verdict,name,at} / {decision,a,b,at}），与老 json 同构。
async function loadLab() {
  try {
    const q: any = await api.opsQueue(['t_face_person', 't_face_pair'])
    const persons: Record<string, any> = {}, pairs: Record<string, any> = {}
    for (const it of q.items['t_face_person'] || []) {
      const d = it.label?.data
      if (d && typeof d === 'object') persons[it.key] = d
      else if (it.label?.value) persons[it.key] = { verdict: it.label.value }
    }
    for (const it of q.items['t_face_pair'] || []) {
      const d = it.label?.data
      if (d && typeof d === 'object') pairs[it.key] = d
      else if (it.label?.value) pairs[it.key] = { decision: it.label.value }
    }
    lab.value = { persons, pairs }
    labErr.value = ''
  } catch {
    try {
      const r = await api.facesLabels()
      lab.value = { persons: r.persons || {}, pairs: r.pairs || {} }
      labErr.value = ''
    } catch { labErr.value = '标注接口不可用（后端未更新？）' }
  }
}
function markOf(p: any) { return lab.value.persons[p.label] || null }
function meta(p: any) {
  const m = markOf(p) || {}
  return {
    name: m.name || '', vc: m.verdict ? (VERD[m.verdict]?.cls || '') : '',
    vt: m.verdict ? (VERD[m.verdict]?.t || m.verdict) : ''
  }
}
const nMarked = computed(() => (data.value?.persons || []).filter((p: any) => markOf(p)?.verdict).length)
const nLf = computed(() => (data.value?.persons || []).filter((p: any) => p.lf).length)
// ---- L0 质检（只读 flag，来源 face_live_flags_v0.json）----
const LF_TIER: Record<string, { t: string; cls: string }> = {
  warn: { t: '警示 · 人审', cls: 'bad' },
  hint: { t: '提示', cls: 'warn' },
  isolated: { t: '倒影 · 已隔离', cls: 'info' }
}
const lfRows = computed(() => {
  const L = data.value?.liveness
  if (!L) return []
  const rows: any[] = []
  for (const t of ['warn', 'hint', 'isolated']) {
    for (const k of (L.review?.[t] || [])) {
      const r = L.tids?.[k]
      if (r) rows.push({ key: k, tier: t, ...r })
    }
  }
  return rows
})
const nNamed = computed(() => (data.value?.persons || []).filter((p: any) => markOf(p)?.name).length)
const nPairMarked = computed(() => (data.value?.embed?.top || []).filter((c: any) => pairDec(c)).length)
function pairDec(c: any) { return lab.value.pairs[`${c.a}@${c.b}`]?.decision || null }

async function save(fn: () => Promise<any>) {
  saving.value = true
  try { await fn(); await loadLab() } finally { saving.value = false }
}
function setVerdict(p: any, v: string) {
  return save(() => api.faceLabel(p.label, { verdict: markOf(p)?.verdict === v ? null : v }))
}
function saveName(p: any) { return save(() => api.faceLabel(p.label, { name: nameDraft.value })) }
function clearMark(p: any) {
  nameDraft.value = ''
  return save(() => api.faceLabel(p.label, { verdict: null, name: null }))
}
function setPair(c: any, d: string) {
  return save(() => api.facePair(c.a, c.b, pairDec(c) === d ? null : d))
}

onMounted(async () => {
  try {
    const r = await fetch('/faces/face_data.json')
    if (!r.ok) throw new Error('HTTP ' + r.status)
    data.value = await r.json()
  } catch (e: any) {
    err.value = String(e)
  }
  loadLab()
})

const cams = ['all', '305', '316', '318', '320', '323', '343']
const camCount = computed<Record<string, number>>(() => {
  const m: Record<string, number> = { all: 0 }
  for (const p of data.value?.persons || []) {
    m.all = (m.all || 0) + 1
    m[p.cam] = (m[p.cam] || 0) + 1
  }
  return m
})
const persons = computed(() => {
  let ps = data.value?.persons || []
  if (camFilter.value !== 'all') ps = ps.filter((p: any) => p.cam === camFilter.value)
  if (onlyFused.value) ps = ps.filter((p: any) => p.img_fused)
  if (onlyTodo.value) ps = ps.filter((p: any) => !markOf(p))
  if (onlyLf.value) ps = ps.filter((p: any) => p.lf)
  const q = search.value.trim().toLowerCase()
  if (q) ps = ps.filter((p: any) => p.label.toLowerCase().includes(q)
    || String(p.tid).includes(q) || (markOf(p)?.name || '').toLowerCase().includes(q))
  return ps
})

function open(p: any) {
  sel.value = p
  selVariant.value = p.img_fused ? 'fused' : p.img_enh ? 'enh' : 'single'
  nameDraft.value = markOf(p)?.name || ''
}
const selImg = computed(() => {
  if (!sel.value) return ''
  const p = sel.value
  return selVariant.value === 'fused' ? (p.img_fused || p.img_single)
    : selVariant.value === 'enh' ? (p.img_enh || p.img_single) : p.img_single
})
const selFusion = computed(() => {
  if (!sel.value) return null
  return (data.value?.fusion || []).find((f: any) => f.cam === sel.value.cam && f.tid === sel.value.tid) || null
})
const selEnh = computed(() => {
  if (!sel.value) return null
  return (data.value?.enhance?.accepted || []).find((r: any) => r.cam === sel.value.cam && r.tid === sel.value.tid) || null
})
const s = computed(() => data.value?.stats || {})
const emb = computed(() => data.value?.embed || {})
function f3(v: any, n = 3) { return typeof v === 'number' ? v.toFixed(n) : v }
</script>

<template>
  <div class="faces">
    <div v-if="err" class="panel" style="margin:12px">
      <div class="panel-h">人脸库数据加载失败</div>
      <div class="panel-b small">请确认 <span class="mono">/faces/face_data.json</span> 已随 dist 部署。{{ err }}</div>
    </div>

    <template v-else-if="data">
      <!-- KPI -->
      <div class="kpis">
        <div class="kpi panel">
          <div class="kv"><b>{{ s.gp_with_face }}</b><i>/ {{ s.gperson_total }}</i></div>
          <div class="tiny dim">覆盖全局人（{{ (s.gp_cover * 100).toFixed(1) }}%）</div>
        </div>
        <div class="kpi panel">
          <div class="kv"><b>{{ s.usable }}</b><i>/ {{ s.shots }}</i></div>
          <div class="tiny dim">可用脸 / 检出脸（{{ (s.usable_rate * 100).toFixed(1) }}%）</div>
        </div>
        <div class="kpi panel">
          <div class="kv"><b>{{ s.fused_accepted }}</b><i>tid</i></div>
          <div class="tiny dim">多帧融合接受（{{ s.gp_fused }} 人升级）</div>
        </div>
        <div class="kpi panel">
          <div class="kv"><b>{{ s.enh_accepted }}</b><i>张</i></div>
          <div class="tiny dim">离线增强接受（复过质量门）</div>
        </div>
        <div class="kpi panel">
          <div class="kv"><b>{{ s.n_cand }}</b><i>对</i></div>
          <div class="tiny dim">人脸候选合并（{{ data.review ? `已复核 ${data.review.conf}C/${data.review.rej}R` : '待人工审计' }}）</div>
        </div>
        <div class="kpi panel">
          <div class="kv"><b>{{ nLf }}</b><i>/ {{ (data.persons || []).length }}</i></div>
          <div class="tiny dim">L0 质检注记人（最佳脸来源 · 待人审）</div>
        </div>
      </div>

      <!-- 工具条 -->
      <div class="tool panel">
        <div class="chips">
          <button v-for="c in cams" :key="c" class="chip" :class="{ on: camFilter === c }"
                  @click="camFilter = c">{{ c === 'all' ? '全部' : c }} <i>{{ camCount[c] || 0 }}</i></button>
        </div>
        <label class="tog"><input type="checkbox" v-model="onlyFused" /> 只看有融合</label>
        <label class="tog"><input type="checkbox" v-model="onlyTodo" /> 只看未标注</label>
        <label class="tog"><input type="checkbox" v-model="onlyLf" /> 只看质检</label>
        <input class="find" v-model="search" placeholder="搜索 gperson / tid / 命名…" />
        <span class="tag mono" :class="labErr ? 'bad' : (nMarked + nPairMarked ? 'ok' : '')"
              :title="labErr || '人工判定进度（存 faces_labels.json）'">
          已判 {{ nMarked }}/{{ (data.persons || []).length }} 人 · 对 {{ nPairMarked }}/{{ (data.embed?.top || []).length }}
        </span>
        <a class="chip dl" :href="api.facesExportUrl()" download="faces_labels.csv"
           :class="{ dis: labErr }" title="导出人工标注 CSV">导出标注</a>
        <span class="tiny dim grow">
          Tag {{ data.meta.tag }} · {{ data.meta.created }} · 人脸产物只读；判定读台账，写入落 faces_labels.json 并同步台账，未改 GIL
        </span>
      </div>

      <!-- 人卡网格 -->
      <div class="grid">
        <button class="pcard" v-for="p in persons" :key="p.label" @click="open(p)"
                :title="`${p.label} · ${p.cam}·t${p.tid} · q=${f3(p.q)}`">
          <img :src="p.img_best" loading="lazy" alt="" />
          <div class="prow">
            <b class="mono">{{ p.label }}</b>
            <span class="tiny dim">{{ p.cam }}·t{{ p.tid }}</span>
            <span class="tiny mono">q{{ f3(p.q, 2) }}</span>
          </div>
          <div class="brow">
            <span v-if="p.img_fused" class="tag ok">融合</span>
            <span v-if="p.img_enh" class="tag info">增强</span>
            <span v-if="!p.img_fused && !p.img_enh" class="tag">单帧</span>
            <span v-if="p.lf" class="tag" :class="p.lf.tier === 'warn' ? 'bad' : 'warn'"
                  :title="'L0 质检（' + (p.lf.tier === 'warn' ? '警示' : '提示') + '）：' + p.lf.note">质检</span>
            <div class="grow" />
            <span v-if="meta(p).name" class="tag sig nn" :title="'命名：' + meta(p).name">{{ meta(p).name }}</span>
            <span v-if="meta(p).vc" class="tag" :class="meta(p).vc">{{ meta(p).vt }}</span>
          </div>
        </button>
      </div>
      <div v-if="!persons.length" class="tiny dim" style="padding:0 14px 10px">无匹配结果</div>

      <!-- 多帧合成 A/B -->
      <section class="panel sect">
        <div class="panel-h">多帧合成清晰化 · A/B 对照
          <span class="tag info">同序列 A/B</span>
          <span class="tag">{{ s.fused_accepted }} 接受</span>
          <div class="grow" />
          <span class="tiny dim">左=最优单帧 ｜ 右=多帧融合（标签含 K 与清晰度比值）</span>
        </div>
        <div class="strip">
          <img v-for="f in (data.fusion || []).filter((x: any) => x.ab)" :key="f.cam + f.tid"
               :src="f.ab" loading="lazy" @click="lightbox = f.ab"
               :title="`${f.cam}·t${f.tid} K=${f.k} ratio=${f.ratio}`" />
        </div>
      </section>

      <!-- 候选合并 -->
      <section class="panel sect">
        <div class="panel-h">人脸嵌入 · 候选合并证据
          <span class="tag" :class="data.review?.completed ? 'ok' : 'warn'">{{ data.review?.completed ? `复核完成 ${data.review.conf}C/${data.review.rej}R` : '待人工审计' }}</span>
          <div class="grow" />
          <span class="tiny dim">w600k_mbf ONNX · 余弦相似度仅为可行性证据；裁决落 faces_labels.json，未做任何 GIL 变更</span>
        </div>
        <div class="rsumm" v-if="data.review?.completed">
          <div class="tiny">
            复核 28/28 完成：<b>{{ data.review.conf }}</b> 确认 / <b>{{ data.review.rej }}</b> 否决 →
            <b class="hl">{{ data.review.n_clusters }} 个合并簇</b>（{{ data.review.gp_before }} gperson → {{ data.review.gp_after }}，净 -9）·
            must-link 约束已生成（<span class="mono">face_mustlink_v1.json</span>），待 GIL 线窗口吸收
          </div>
          <div class="clist">
            <span v-for="c in (data.review.clusters || [])" :key="c.cid" class="tag mono">
              簇{{ c.cid }} · {{ c.labels.join('+') }}{{ c.cross ? ' · 跨机位' : '' }}
            </span>
          </div>
        </div>
        <div class="panel-b cands">
          <table class="tbl">
            <thead><tr><th>对比（点击放大）</th><th>cos</th><th>gperson</th><th>裁决</th></tr></thead>
            <tbody>
              <tr v-for="(c, i) in (emb.top || [])" :key="i">
                <td>
                  <div class="cpair">
                    <figure>
                      <img v-if="c.a_img" :src="c.a_img" alt="" @click="lightbox = c.a_img" />
                      <figcaption class="mono">{{ c.a }}</figcaption>
                    </figure>
                    <figure>
                      <img v-if="c.b_img" :src="c.b_img" alt="" @click="lightbox = c.b_img" />
                      <figcaption class="mono">{{ c.b }}</figcaption>
                    </figure>
                  </div>
                </td>
                <td>
                  <div class="mono" :class="c.cos >= 0.75 ? 'hl' : ''">{{ f3(c.cos) }}</div>
                  <span v-if="c.ext_cluster" class="tag ok ctag" :title="c.ext_note || ''">簇{{ c.ext_cluster }}·续</span>
                  <span v-else-if="c.caution" class="tag warn ctag" :title="c.caution_note || ''">320 谨慎</span>
                </td>
                <td class="tiny dim">{{ c.la }} ↔ {{ c.lb }}</td>
                <td>
                  <div class="row" style="gap:3px">
                    <button v-for="d in decisions" :key="d.k" class="mini"
                            :class="[pairDec(c) === d.k ? d.cls : '', { on: pairDec(c) === d.k }]"
                            :disabled="saving || !!labErr" @click="setPair(c, d.k)"
                            :title="`${d.t}（再点一次取消）`">{{ d.t }}</button>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
          <div class="cand-side">
            <img v-if="data.evidence?.merge_clusters" :src="data.evidence.merge_clusters"
                 @click="lightbox = data.evidence.merge_clusters" />
            <div class="tiny dim">↑ 5 簇最终拼图（复核 28/28 结论，点击放大）</div>
            <img v-if="data.evidence?.candidates" :src="data.evidence.candidates"
                 @click="lightbox = data.evidence.candidates" />
            <div class="tiny dim">上：候选对拼图（红色=高相似）；识别模型对低清/口罩脸会退化，低分不直接判异人</div>
          </div>
        </div>
      </section>

      <!-- L0 证据质检 -->
      <section class="panel sect" v-if="data.liveness?.tids">
        <div class="panel-h">L0 证据质检（零模型规则 · 只读 flag）
          <span class="tag bad">警示 {{ (data.liveness.review?.warn || []).length }}</span>
          <span class="tag warn">提示 {{ (data.liveness.review?.hint || []).length }}</span>
          <span class="tag info">倒影已隔离 {{ (data.liveness.review?.isolated || []).length }}</span>
          <div class="grow" />
          <span class="tiny dim">{{ data.liveness.summary?.n_shots }} 帧全量扫描 · flag {{ data.liveness.summary?.n_flagged }} 条 · 只读标注，不进 GIL 主链</span>
        </div>
        <div class="panel-b" style="overflow:auto">
          <table class="tbl">
            <thead><tr><th>轨迹</th><th>分级</th><th>命中规则</th><th>最佳脸</th><th>说明（目检定性与证据）</th></tr></thead>
            <tbody>
              <tr v-for="r in lfRows" :key="r.key">
                <td class="mono">{{ r.key }}</td>
                <td><span class="tag" :class="LF_TIER[r.tier]?.cls">{{ LF_TIER[r.tier]?.t }}</span></td>
                <td class="tiny mono">{{ (r.flags || []).join(' · ') }}</td>
                <td class="tiny">{{ r.top1?.usable ? (r.top1?.flags?.length ? '被 flag' : '未命中') : '不在池' }} · q{{ f3(r.top1?.q, 2) }}</td>
                <td class="tiny dim">{{ r.note }}</td>
              </tr>
            </tbody>
          </table>
          <div class="tiny dim" style="padding-top:6px">
            低质误检轨迹 {{ (data.liveness.review?.ignore || []).length }} 条（非可用，无需行动）；完整清单与 per-shot flag 见
            <span class="mono">face_live_flags_v0.json</span>；倒影条目在 p0c 已移出人层。
          </div>
        </div>
      </section>

      <!-- 证据墙 -->
      <section class="panel sect">
        <div class="panel-h">证据与溯源
          <div class="grow" />
          <span class="tiny dim">对齐诊断（316-88 逐帧栈）· 全楼人脸墙</span>
        </div>
        <div class="strip big">
          <img v-if="data.evidence?.montage_align" :src="data.evidence.montage_align"
               @click="lightbox = data.evidence.montage_align" />
          <img v-if="data.evidence?.wall" :src="data.evidence.wall" @click="lightbox = data.evidence.wall" />
        </div>
      </section>

      <div class="tiny dim foot">
        参数均为经验初值（待 val 校准）｜ L0 质检为规则判定（只读 flag，待人审；警示 1 例=后脑勺检错进池）｜
        增强/融合图与原始图成对存储，不进 GIL 身份判定主链
      </div>

      <!-- 详情弹窗 -->
      <div v-if="sel" class="modal" @click.self="sel = null">
        <div class="mbox panel">
          <div class="panel-h">
            <span class="mono">{{ sel.label }}</span>
            <span class="tag">{{ sel.cam }}·t{{ sel.tid }}</span>
            <span v-if="sel.src === 'fused'" class="tag ok">融合升级</span>
            <div class="grow" />
            <button class="ghost" @click="sel = null">关闭</button>
          </div>
          <div class="mbody">
            <div class="mleft">
              <img class="mimg" :src="selImg" @click="lightbox = selImg" alt="" />
              <div class="mtabs">
                <button class="chip" :class="{ on: selVariant === 'single' }" @click="selVariant = 'single'">单帧</button>
                <button class="chip" :class="{ on: selVariant === 'fused' }" :disabled="!sel.img_fused"
                        @click="selVariant = 'fused'">融合</button>
                <button class="chip" :class="{ on: selVariant === 'enh' }" :disabled="!sel.img_enh"
                        @click="selVariant = 'enh'">增强</button>
              </div>
              <img v-if="sel.ab && selVariant === 'fused'" class="mab" :src="sel.ab" @click="lightbox = sel.ab" />
            </div>
            <div class="mright col">
              <table class="kv-t">
                <tr><td>质量 q</td><td class="mono">{{ f3(sel.q) }}</td></tr>
                <tr><td>脸宽 / 瞳距</td><td class="mono">{{ sel.w }} / {{ sel.pd }} px</td></tr>
                <tr><td>清晰度 lap</td><td class="mono">{{ sel.lap }}</td></tr>
                <tr><td>该人可用脸</td><td class="mono">{{ sel.n_usable }} 张</td></tr>
                <tr v-if="selFusion"><td>融合 K / 比值</td><td class="mono">{{ selFusion.k }} / {{ f3(selFusion.ratio as any, 2) }}</td></tr>
                <tr v-if="selFusion"><td>ECC / NCC</td><td class="mono">{{ f3(selFusion.ecc_min as any, 2) }} / {{ f3(selFusion.ncc_min as any, 2) }}</td></tr>
                <tr v-if="selEnh"><td>增强模式</td><td class="mono">{{ selEnh.mode }} (dq {{ f3(selEnh.dq as any) }})</td></tr>
              </table>
              <div v-if="sel.lf" class="lfnote" :class="sel.lf.tier === 'warn' ? 'lfbad' : ''">
                <b>L0 质检（{{ sel.lf.tier === 'warn' ? '警示 · 建议人审' : '提示' }}）</b>{{ sel.lf.note }}
              </div>
              <div class="labbox col">
                <div class="row" style="gap:6px">
                  <span class="label">人工标注</span>
                  <span v-if="markOf(sel)" class="tiny dim mono">更新于 {{ markOf(sel).at }}</span>
                  <div class="grow" />
                  <span v-if="labErr" class="tiny" style="color:var(--danger)">{{ labErr }}</span>
                </div>
                <div class="row wrap" style="gap:5px">
                  <button v-for="(v, k) in VERD" :key="k" class="chip" :class="{ on: markOf(sel)?.verdict === k }"
                          :disabled="saving || !!labErr" @click="setVerdict(sel, k)"
                          :title="`判定为「${v.t}」（再点一次取消）`">{{ v.t }}</button>
                  <div class="sep" />
                  <label class="sr-only" for="fnm">命名</label>
                  <input id="fnm" class="find" v-model="nameDraft" placeholder="命名 / 标签，如：白衣男"
                         @keyup.enter="saveName(sel)" />
                  <button class="chip" :disabled="saving || !!labErr" @click="saveName(sel)">保存命名</button>
                  <button v-if="markOf(sel)" class="chip del" :disabled="saving" @click="clearMark(sel)">清除标注</button>
                  <span v-if="saving" class="spinner" />
                </div>
              </div>
              <div class="tiny dim">
                单帧=质量池最优；融合=同轨迹多帧合成（结构一致度掩膜，防鬼影）；增强=离线低光/逆光校正（复过质量门）。
              </div>
              <div class="tiny dim">标注写入独立文件 <span class="mono">faces_labels.json</span>（人工审计线）｜ 人脸产物与 GIL 均只读。</div>
            </div>
          </div>
        </div>
      </div>

      <div v-if="lightbox" class="lightbox" @click="lightbox = ''">
        <img :src="lightbox" />
      </div>
    </template>
  </div>
</template>

<style scoped>
.faces { padding: var(--sp-2) var(--sp-3) var(--sp-4); display: flex; flex-direction: column; gap: var(--sp-2); overflow: auto; }
.faces > * { flex: none; }   /* 受限 flex 列：子项不可压缩，否则 section 被压没（与 Visualize/Cluster 页一致） */
.kpis { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: var(--sp-2); }
.kpi { padding: 10px 12px; }
.kv b { font-size: 22px; font-family: var(--font-mono); }
.kv i { font-style: normal; font-size: 12px; color: var(--fg-3); margin-left: 4px; }
.tool { display: flex; align-items: center; gap: var(--sp-3); padding: 8px 12px; flex-wrap: wrap; }
.chips { display: flex; gap: 6px; flex-wrap: wrap; }
.chip { border: 1px solid var(--line); background: transparent; color: var(--fg-2);
  border-radius: var(--r-pill); padding: 3px 10px; font-size: 12px; cursor: pointer; }
.chip i { font-style: normal; color: var(--fg-3); font-size: 11px; margin-left: 3px; }
.chip.on { border-color: var(--accent-line); color: var(--accent); background: var(--accent-soft); }
.chip:disabled { opacity: .4; cursor: not-allowed; }
.tog { font-size: 12px; color: var(--fg-2); display: flex; gap: 4px; align-items: center; }
.find { background: var(--panel-2); border: 1px solid var(--line); border-radius: var(--r-sm);
  color: var(--fg); font-size: 12px; padding: 4px 8px; width: 170px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: var(--sp-2); }
.pcard { border: 1px solid var(--line); background: var(--panel); border-radius: var(--r-md);
  padding: 0; overflow: hidden; cursor: pointer; text-align: left; color: inherit; }
.pcard:hover { border-color: var(--accent-line); }
.pcard img { width: 100%; height: 132px; object-fit: contain; background: var(--panel-2); display: block; }
.prow { display: flex; align-items: center; gap: 6px; padding: 6px 8px 2px; }
.prow .grow { flex: 1; }
.brow { display: flex; gap: 4px; padding: 0 8px 8px; }
.sect { overflow: hidden; }
.strip { display: flex; gap: 8px; overflow-x: auto; padding: 10px 12px; }
.strip img { height: 150px; border-radius: var(--r-sm); border: 1px solid var(--line); cursor: zoom-in; flex: none; background: var(--panel-2); }
.strip.big img { height: 240px; }
.cands { display: flex; gap: var(--sp-3); }
.rsumm { padding: var(--sp-2) var(--sp-3); display: flex; flex-direction: column; gap: 6px; border-bottom: 1px solid var(--line); }
.clist { display: flex; flex-wrap: wrap; gap: 4px; }
.ctag { margin-top: 3px; }
.cpair { display: flex; gap: 10px; }
.cpair figure { margin: 0; text-align: center; }
.cpair img { height: 58px; max-width: 104px; object-fit: contain; background: var(--panel-2);
  border: 1px solid var(--line); border-radius: var(--r-xs); cursor: zoom-in; display: block; }
.cpair figcaption { font-size: 10px; color: var(--fg-3); margin-top: 2px; }
.tbl { border-collapse: collapse; font-size: 12px; min-width: 320px; }
.tbl th, .tbl td { border-bottom: 1px solid var(--line); padding: 3px 10px 3px 0; text-align: left; }
.tbl th { color: var(--fg-3); font-weight: 500; }
.tbl td.hl { color: var(--ok); }
.cand-side { flex: 1; min-width: 0; }
.cand-side img { max-width: min(560px, 100%); border: 1px solid var(--line); border-radius: var(--r-sm); cursor: zoom-in; display: block; margin-bottom: 4px; }
.foot { padding: 2px 4px 6px; }
.modal { position: fixed; inset: 0; background: rgba(0,0,0,.55); display: flex; align-items: center; justify-content: center; z-index: 60; }
.mbox { width: min(880px, 94vw); max-height: 90vh; overflow: auto; }
.mbody { display: flex; gap: var(--sp-3); padding: var(--sp-3); }
.mleft { flex: 0 0 46%; display: flex; flex-direction: column; gap: 8px; }
.mimg { width: 100%; background: var(--panel-2); border: 1px solid var(--line); border-radius: var(--r-sm); cursor: zoom-in; }
.mtabs { display: flex; gap: 6px; }
.mab { width: 100%; border: 1px solid var(--line); border-radius: var(--r-sm); cursor: zoom-in; }
.mright { flex: 1; min-width: 0; gap: 8px; }
.kv-t { border-collapse: collapse; font-size: 12px; }
.kv-t td { border-bottom: 1px solid var(--line); padding: 3px 12px 3px 0; }
.kv-t td:first-child { color: var(--fg-3); }
.lightbox { position: fixed; inset: 0; background: rgba(0,0,0,.82); display: flex; align-items: center;
  justify-content: center; z-index: 80; cursor: zoom-out; }
.lightbox img { max-width: 94vw; max-height: 92vh; }

/* ---- 标注 ---- */
.nn { max-width: 76px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.dl { text-decoration: none; }
.dl.dis { opacity: .45; pointer-events: none; }
.mini { border: 1px solid var(--line); background: transparent; color: var(--fg-3);
  border-radius: var(--r-xs); padding: 1px 7px; font-size: 11px; cursor: pointer; white-space: nowrap; }
.mini:hover { border-color: var(--line-2); color: var(--fg-2); }
.mini.on { font-weight: 600; }
.mini.on-conf { border-color: var(--ok); color: var(--ok); background: var(--ok-soft, rgba(56,211,159,.12)); }
.mini.on-rej { border-color: var(--danger); color: var(--danger); background: var(--danger-soft, rgba(255,92,114,.12)); }
.mini.on-maybe { border-color: var(--warn); color: var(--warn); background: rgba(240,180,60,.12); }
.mini:disabled { opacity: .5; cursor: default; }
.labbox { border-top: 1px solid var(--line); padding-top: var(--sp-2); gap: 6px; margin-top: var(--sp-1); }
.lfnote { border-left: 3px solid var(--warn); background: rgba(240,180,60,.10); border-radius: 4px;
  padding: 6px 8px; font-size: 12px; line-height: 1.5; }
.lfnote.lfbad { border-left-color: var(--danger); background: rgba(255,92,114,.10); }
.chip.del { color: var(--danger); border-color: var(--danger-line, var(--line)); }
@media (max-width: 1100px) {
  .kpis { grid-template-columns: repeat(2, 1fr); }
  .mbody { flex-direction: column; }
  .cands { flex-direction: column; }
}
</style>
