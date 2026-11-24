<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { api } from '../api'

const d = ref<any>(null)
const err = ref('')
const busy = ref(false)
const status = ref('')
const msgKind = ref<'ok' | 'warn' | 'err' | 'info'>('info')
const picked = ref<Record<string, boolean>>({})
const lvl = ref<'queue' | 'low' | 'all'>('queue')
const showCards = ref(false)
const cards = ref<any>(null)

async function load() {
  err.value = ''
  try {
    d.value = await api.autoFace()
    picked.value = {}
  } catch (e: any) {
    d.value = null
    err.value = String(e?.message || e)
  }
}

async function refresh() {
  busy.value = true
  status.value = '重算中…'
  msgKind.value = 'info'
  try {
    const r = await api.autoFaceRefresh()
    status.value = `已重算：${r.n_suggestions} 条建议（用时 ${r.created}）`
    msgKind.value = 'ok'
    await load()
  } catch (e: any) {
    status.value = `重算失败：${e?.message || e}`
    msgKind.value = 'err'
  } finally { busy.value = false }
}

const rows = computed<any[]>(() => {
  const all = d.value?.suggestions || []
  if (lvl.value === 'all') return all
  return all.filter((r: any) => r.level === lvl.value)
})
const nPicked = computed(() => Object.values(picked.value).filter(Boolean).length)
const allPicked = computed(() => rows.value.length > 0 && rows.value.every((r: any) => picked.value[r.key]))

function toggleAll() {
  const on = !allPicked.value
  const p: Record<string, boolean> = { ...picked.value }
  for (const r of rows.value) p[r.key] = on
  picked.value = p
}

async function apply(keys: string[] | null, minScore?: number) {
  if (!keys || !keys.length) { status.value = '没有选中任何条目'; msgKind.value = 'warn'; return }
  busy.value = true
  status.value = `写入中（${keys.length} 条）…`
  msgKind.value = 'info'
  try {
    const r = await api.autoFaceApply({ keys, min_score: minScore ?? null })
    const ok = new Set<string>((r.applied || []).map((x: any) => x.key))
    // ★ 原地标记，不整表重载：重载会闪一下、把勾选全清掉，用户会以为「没反应」。
    for (const s of d.value?.suggestions || []) if (ok.has(s.key)) s.in_pool = true
    const p = { ...picked.value }
    for (const k of ok) delete p[k]
    picked.value = p
    const parts = [`写入池 ${r.n_applied} 条`]
    if (r.n_dup) parts.push(`${r.n_dup} 条已在池中`)
    const other = (r.skipped || []).length - (r.n_dup || 0)
    if (other > 0) parts.push(`${other} 条未写入（框高未过质量门等）`)
    if (r.n_pool != null) parts.push(`池内共 ${r.n_pool} 条`)
    status.value = parts.join('，')
    msgKind.value = r.n_applied > 0 ? 'ok' : 'warn'
  } catch (e: any) {
    status.value = `写入失败：${e?.message || e}`
    msgKind.value = 'err'
  } finally { busy.value = false }
}
function applyPicked() { apply(rows.value.filter((r: any) => picked.value[r.key]).map((r: any) => r.key)) }
function applyAllQueue() {
  const keys = (d.value?.suggestions || [])
    .filter((r: any) => r.level === 'queue' && !r.in_pool).map((r: any) => r.key)
  if (!keys.length) { status.value = '队列里没有待采纳的条目'; msgKind.value = 'warn'; return }
  apply(keys)
}
function toggleRow(r: any) {
  picked.value = { ...picked.value, [r.key]: !picked.value[r.key] }
}

async function loadCards() {
  showCards.value = !showCards.value
  if (!showCards.value || cards.value) return
  try { cards.value = await api.autoFaceCards(0) } catch (e: any) { status.value = String(e?.message || e) }
}

const pct = (v: number | null | undefined) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)
const f3 = (v: number | null | undefined) => (v == null ? '—' : Number(v).toFixed(3))
const ci = (a?: number[] | null) => (a && a.length === 2 ? `[${a[0].toFixed(3)}, ${a[1].toFixed(3)}]` : '')
onMounted(load)
</script>

<template>
  <div class="af">
    <!-- 顶栏 -->
    <div class="panel head">
      <div class="row wrap">
        <b>自动人脸</b>
        <span v-if="d" class="tag mono">
          {{ d.coverage.n_face }} / {{ d.coverage.n_store }} 轨有脸 · {{ pct(d.coverage.pct_of_store) }}
        </span>
        <span v-if="d" class="tag warn">自动写池已关闭</span>
        <div class="grow" />
        <button class="pill" :disabled="busy" @click="refresh">重算建议</button>
        <button class="pill" :disabled="busy || !rows.length" @click="toggleAll">
          {{ allPicked ? '取消全选' : '全选本页' }}
        </button>
        <button class="pill signal" :disabled="busy || !nPicked" @click="applyPicked">
          采纳选中（{{ nPicked }}）
        </button>
        <button class="pill" :disabled="busy || !d || !d.n_queue" @click="applyAllQueue">
          采纳队列全部
        </button>
      </div>
      <div v-if="status" class="notice" :class="msgKind">{{ status }}</div>
      <div v-if="err" class="tiny" style="margin-top:4px;color:var(--danger)">
        {{ err }}<span v-if="err.includes('尚未')"> — 先跑 <code>python tools/face_query_scan.py</code> 再点「重算建议」</span>
      </div>
    </div>

    <div v-if="d" class="body">
      <!-- 左：复核队列 -->
      <div class="panel col" style="min-height:0">
        <div class="panel-h">
          复核队列
          <span class="sub">{{ d.n_sugg }} 条建议 · τ={{ d.params.tau }}</span>
          <div class="grow" />
          <button class="pill" :aria-pressed="lvl === 'queue'" @click="lvl = 'queue'">队列 {{ d.n_queue }}</button>
          <button class="pill" :aria-pressed="lvl === 'low'" @click="lvl = 'low'">低分 {{ d.n_sugg - d.n_queue }}</button>
          <button class="pill" :aria-pressed="lvl === 'all'" @click="lvl = 'all'">全部</button>
        </div>
        <div class="scroll grow">
          <div v-for="r in rows" :key="r.key" class="item" :class="{ done: r.in_pool }"
               :aria-pressed="!!picked[r.key]" @click="toggleRow(r)">
            <label class="chk" @click.stop>
              <input type="checkbox" v-model="picked[r.key]" />
            </label>
            <img class="face" :src="api.autoFaceCropUrl(r.crop)" :alt="r.key" loading="lazy" />
            <img v-if="r.match_crop" class="face dimmed" :src="api.autoFaceCropUrl(r.match_crop)"
                 :alt="r.match_key" loading="lazy" />
            <div class="col" style="gap:2px;min-width:0">
              <div class="row" style="gap:6px">
                <b class="mono tiny">{{ r.key }}</b>
                <span class="tiny dim">→</span>
                <b class="mono tiny">{{ r.ident }}</b>
                <span v-if="r.margin != null" class="tag mono">次名差 {{ f3(r.margin) }}</span>
                <span v-if="!r.same_entity" class="tag ok">跨实体</span>
                <span v-else class="tag">同实体</span>
                <span v-if="r.in_pool" class="tag ok">已在池</span>
              </div>
              <div class="tiny dim mono">
                cos {{ f3(r.score) }} · pd {{ r.pd ?? '—' }} · lap {{ r.lap ?? '—' }} · 框高 {{ Math.round(r.h_med) }}
                · 模板 {{ r.match_key }} · 该身份 {{ r.ent_tracks }} 轨
              </div>
            </div>
            <div class="grow" />
            <button class="pill" :disabled="busy || r.in_pool" @click.stop="apply([r.key])">
              {{ r.in_pool ? '已在池' : '采纳' }}
            </button>
          </div>
          <div v-if="!rows.length" class="tiny dim" style="padding:12px">这一类没有条目</div>
        </div>
      </div>

      <!-- 右：口径与数字 -->
      <div class="col right" style="min-height:0">
        <div class="panel">
          <div class="panel-h">这条通道能做什么<span class="sub">口径全写在 faceauto.py 顶部</span></div>
          <div class="panel-b tight">
            <div class="kv"><span>库内轨</span><b class="mono">{{ d.coverage.n_store }}</b></div>
            <div class="kv"><span>参与扫描（框高 ≥ {{ d.coverage.scan_min_h }} px）</span><b class="mono">{{ d.coverage.n_scan }}</b></div>
            <div class="kv"><span>采到可用脸</span><b class="mono">{{ d.coverage.n_face }}（{{ pct(d.coverage.pct_of_store) }}）</b></div>
            <div class="kv"><span>检出人脸总数</span><b class="mono">{{ d.coverage.faces_detected }}</b></div>
            <div class="hint">
              面部像素预算约 0.054×框高；本库框高中位 142 px，所以覆盖率上不去是采集几何决定的。
            </div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-h">识别力（留一）<span class="sub">n 小，带区间</span></div>
          <div class="panel-b tight">
            <table class="tb">
              <thead><tr><th>通道</th><th>AUC</th><th>rank-1</th></tr></thead>
              <tbody>
                <tr>
                  <td>人脸（mbf）</td>
                  <td class="mono">{{ f3(d.face.auc) }} <span class="tiny dim">{{ ci(d.face.auc_ci) }}</span></td>
                  <td class="mono">{{ d.face.rank1_hit }}/{{ d.face.rank1_n }} = {{ pct(d.face.rank1) }}</td>
                </tr>
                <tr>
                  <td>外观（同批轨）</td>
                  <td class="mono">{{ f3(d.appearance.auc) }} <span class="tiny dim">{{ ci(d.appearance.auc_ci) }}</span></td>
                  <td class="mono">{{ d.appearance.rank1_hit }}/{{ d.appearance.rank1_n }} = {{ pct(d.appearance.rank1) }}</td>
                </tr>
              </tbody>
            </table>
            <div class="hint">
              <b>人脸不比外观强</b>，这两个数必须一起看。人脸的价值在别处：{{ d.n_solo_ent }} / {{ d.coverage.n_face }}
              条有脸轨所属实体是单轨实体，外观留一后<b>没有任何模板可比</b> —— 对这些轨，
              外观连问题都问不出来，人脸是唯一能给身份的通道。
            </div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-h">为什么自动写池是关的</div>
          <div class="panel-b tight">
            <table class="tb">
              <thead><tr><th>τ</th><th>接受</th><th>对</th><th>精度</th><th>召回</th></tr></thead>
              <tbody>
                <tr v-for="r in d.sweep" :key="r.tau">
                  <td class="mono">{{ r.tau.toFixed(2) }}</td>
                  <td class="mono">{{ r.accepted }}</td>
                  <td class="mono">{{ r.hit }}</td>
                  <td class="mono">{{ f3(r.prec) }}</td>
                  <td class="mono">{{ f3(r.recall) }}</td>
                </tr>
              </tbody>
            </table>
            <div class="hint">
              精度在 τ 0.45→0.50 升到 0.667，到 0.55 又掉回 0.40 —— <b>非单调</b>，且每档只有个位数样本。
              没有任何阈值足以直接写进池，所以只进复核队列；要开自动必须显式改 <code>TAU_AUTO</code>。
            </div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-h">
            卡片图库自检
            <div class="grow" />
            <button class="pill" @click="loadCards">{{ showCards ? '收起' : '展开' }}</button>
          </div>
          <div v-if="showCards" class="panel-b tight">
            <div v-if="!cards" class="tiny dim">加载中…</div>
            <div v-else>
              <div class="kv"><span>卡片数 / 可编码</span><b class="mono">{{ cards.meta.n }} / {{ cards.meta.n_ok }}</b></div>
              <div class="kv"><span>SCRFD 检出 / 兜底对齐</span><b class="mono">{{ cards.meta.align_det }} / {{ cards.meta.align_fallback }}</b></div>
              <div class="kv"><span>img_best 是融合模板</span><b class="mono">{{ cards.meta.fused_best }} / {{ cards.meta.n }}</b></div>
              <div class="kv"><span>卡标签与 GIL 一致</span><b class="mono">{{ cards.meta.label_vs_gil }} / {{ cards.meta.n }}</b></div>
              <div class="kv"><span>跨身份余弦 max / p99</span>
                <b class="mono">{{ f3(cards.meta.impostor.max) }} / {{ f3(cards.meta.impostor.p99) }}</b></div>
              <div class="hint">{{ cards.meta.note }}</div>
            </div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-h">采集口径</div>
          <div class="panel-b tight tiny dim">
            {{ d.gil_note }}。匹配用视频侧模板 + 留一；帧按框高挑（面部像素 ≈0.054×框高），
            ROI 取框上 55%，人脸须落在本轨框内。
            <br />采集脚本 <code>tools/face_query_scan.py</code>，离线跑一次约 220 s。
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.af { display: flex; flex-direction: column; gap: var(--sp-2); height: 100%; min-height: 0; padding: var(--sp-2); }
.head { flex: none; padding: var(--sp-2) var(--sp-3); }
.body { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(320px, 0.9fr); gap: var(--sp-2); flex: 1; min-height: 0; }
.right { gap: var(--sp-2); overflow: auto; min-height: 0; }

.item {
  display: flex; align-items: center; gap: 8px;
  padding: 5px var(--sp-2); border-bottom: 1px solid var(--line);
  background: transparent; cursor: pointer;
}
.item[aria-pressed='true'] { background: var(--surface-2); box-shadow: inset 2px 0 0 var(--accent); }
.item.done { opacity: 0.66; }
.item.done .face { filter: saturate(0.85); }

.notice {
  margin-top: 6px; padding: 5px 8px; border-radius: var(--r-xs);
  font-size: var(--text-xs); border: 1px solid var(--line-2);
  color: var(--fg-2); background: var(--surface-2);
}
.notice.ok { color: var(--ok); border-color: var(--ok-line); background: var(--ok-soft); }
.notice.warn { color: var(--warn); border-color: var(--warn-line); background: var(--warn-soft); }
.notice.err { color: var(--danger); border-color: var(--danger-line); background: var(--danger-soft); }
.chk { display: flex; align-items: center; padding: 0 2px; }
.face {
  width: 40px; height: 40px; object-fit: cover; border-radius: var(--r-xs);
  background: #000; flex: none; border: 1px solid var(--line-2);
}
.face.dimmed { opacity: 0.75; width: 34px; height: 34px; border-style: dashed; }

.kv { display: flex; justify-content: space-between; gap: 8px; padding: 2px 0; font-size: var(--text-xs); }
.kv > span { color: var(--fg-3); }
.hint { margin-top: 6px; font-size: var(--text-xs); color: var(--fg-3); line-height: 1.5; }
.hint b { color: var(--fg-2); }
.tb { width: 100%; border-collapse: collapse; font-size: var(--text-xs); }
.tb th { text-align: left; color: var(--fg-3); font-weight: 500; padding: 2px 4px; }
.tb td { padding: 2px 4px; border-top: 1px solid var(--line); }
code { font-family: var(--font-mono); font-size: 0.92em; color: var(--fg-2); }
@media (max-width: 1100px) { .body { grid-template-columns: minmax(0, 1fr); } }
</style>
