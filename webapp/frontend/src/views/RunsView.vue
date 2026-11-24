<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import { api } from '../api'
import TrainCurve from '../components/TrainCurve.vue'

const route = useRoute()
const list = ref<any[]>([])
const cur = ref<any>(null)
const logs = ref<string[]>([])
const err = ref('')
const loading = ref(true)
const filter = ref('')
const kindFilter = ref('')

onMounted(async () => {
  await loadList()
  const qid = String(route.query.id || '')
  if (qid) await pick(qid)
  else if (list.value.length) await pick(list.value[0].id)
})

watch([filter, kindFilter], loadList)

// 从总览页点「最近运行」跳过来时 hash 不变、只有 query 变，需监听
watch(() => route.query.id, (v) => {
  const id = String(v || '')
  if (id && id !== cur.value?.id) pick(id)
})

async function loadList() {
  loading.value = true
  err.value = ''
  try {
    const r = await api.opsRuns(200, kindFilter.value, filter.value)
    list.value = r.runs || []
    if (cur.value && !list.value.some(x => x.id === cur.value.id)) cur.value = null
  } catch (e: any) {
    err.value = String(e?.message || e)
  } finally {
    loading.value = false
  }
}

async function pick(id: string) {
  try {
    cur.value = await api.opsRun(id)
    const lg = await api.opsRunLog(id, 160)
    logs.value = lg.lines || []
  } catch (e: any) {
    err.value = String(e?.message || e)
  }
}

// 指标结构化：标量逐行、嵌套折叠摘要，避免整块 JSON 粘在一起难读
const metricRows = computed(() => {
  const m = cur.value?.metrics
  if (!m || typeof m !== 'object') return []
  const out: { k: string; v: string; note?: string }[] = []
  if (Array.isArray(m.items)) {
    for (const it of m.items) {
      if (!it || typeof it !== 'object') continue
      const n = (it.value !== undefined && it.value !== null) ? it.value : it.n
      out.push({ k: String(it.name ?? '?'), v: String(n ?? '—'), note: it.note })
    }
    return out
  }
  for (const [k, v] of Object.entries(m)) {
    if (k === '_raw') continue
    if (v === null || v === undefined) { out.push({ k, v: '—' }); continue }
    if (typeof v !== 'object') { out.push({ k, v: String(v) }); continue }
    if (Array.isArray(v)) { out.push({ k, v: `${v.length} 项` }); continue }
    const ks = Object.keys(v)
    out.push({ k, v: ks.length <= 3 ? JSON.stringify(v) : `${ks.length} 个字段` })
  }
  return out
})

const params = computed(() => {
  const p = cur.value?.params || {}
  return Object.entries(p).filter(([k]) => k !== 'cmd')
})
const cmd = computed(() => (cur.value?.params?.cmd || []).join(' ') || cur.value?.entrypoint || '')

function dur(r: any) {
  if (!r?.started_at || !r?.ended_at) return '—'
  const a = Date.parse(r.started_at.replace(' ', 'T'))
  const b = Date.parse(r.ended_at.replace(' ', 'T'))
  if (!isFinite(a) || !isFinite(b)) return '—'
  const s = Math.max(0, Math.round((b - a) / 1000))
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m${String(s % 60).padStart(2, '0')}s`
}
function size(n: number | null) {
  if (n == null) return '—'
  if (n < 1024) return `${n} B`
  if (n < 1048576) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1048576).toFixed(1)} MB`
}
const KINDS = [
  { v: '', label: '全部类型', t: '不按类型过滤' },
  { v: 'train', label: '训练', t: 'kind=train' },
  { v: 'eval', label: '评测', t: 'kind=eval' },
]
const statusTag = (s: string) =>
  ({ succeeded: 'ok', running: 'info', failed: 'bad', killed: 'warn', stale: 'warn' } as any)[s] || ''
</script>

<template>
  <div class="runs">
    <!-- 左：列表 -->
    <aside class="col side" style="gap:var(--sp-2);min-height:0">
      <section class="panel" style="min-height:0;display:flex;flex-direction:column">
        <div class="panel-h">
          运行记录
          <span class="tag">{{ list.length }}</span>
          <div class="grow" />
          <button class="ghost tiny" @click="loadList" title="刷新">刷新</button>
        </div>
        <div class="panel-b tight">
          <div class="seg" role="tablist">
            <button v-for="s in ['', 'running', 'succeeded', 'failed', 'stale']" :key="s"
                    :aria-selected="filter === s" @click="filter = s">
              {{ s === '' ? '全部' : s }}
            </button>
          </div>
          <div class="seg" role="tablist" style="margin-top:6px">
            <button v-for="k in KINDS" :key="k.v" :aria-selected="kindFilter === k.v"
                    :title="k.t" @click="kindFilter = k.v">{{ k.label }}</button>
          </div>
        </div>
        <div class="scroll grow" style="padding:0 var(--sp-2) var(--sp-2)">
          <div v-if="err" class="empty tiny">{{ err }}</div>
          <div v-else-if="!list.length && !loading" class="empty tiny">没有匹配的 run</div>
          <button v-for="r in list" :key="r.id" class="item" :class="{ on: cur?.id === r.id }" @click="pick(r.id)">
            <div class="row" style="gap:6px">
              <span class="tag" :class="statusTag(r.status)">{{ r.status }}</span>
              <span v-if="r.kind === 'train'" class="tag">train</span>
              <span class="ellip small" style="font-weight:600">{{ r.name }}</span>
            </div>
            <div class="row tiny dim" style="gap:6px">
              <span class="mono">{{ r.id }}</span>
              <div class="grow" />
              <span>{{ dur(r) }}</span>
            </div>
          </button>
        </div>
      </section>
    </aside>

    <!-- 右：详情 -->
    <div class="col main" style="gap:var(--sp-3);min-height:0">
      <div v-if="!cur" class="panel">
        <div class="panel-h">运行详情</div>
        <div class="panel-b empty small">
          左侧选择一条运行记录。run 由 <span class="mono">tools/ops_run.py</span> 产生：
          参数、环境、产物 sha1、退出码、日志都会自动登记。
        </div>
      </div>

      <template v-else>
        <section class="panel">
          <div class="panel-h">
            <span class="mono">{{ cur.name }}</span>
            <span class="tag" :class="statusTag(cur.status)">{{ cur.status }}</span>
            <span v-if="cur.exit_code !== null" class="tag" :class="cur.exit_code === 0 ? 'ok' : 'bad'">
              exit {{ cur.exit_code }}
            </span>
            <div class="grow" />
            <span class="tiny dim">{{ dur(cur) }}</span>
          </div>
          <div class="panel-b">
            <div class="kv">
              <div>run_id</div><div class="mono">{{ cur.id }}</div>
              <div>类型</div><div>{{ cur.kind }}</div>
              <div>开始 / 结束</div><div class="mono">{{ cur.started_at }} → {{ cur.ended_at || '—' }}</div>
              <div>主机 / GPU</div><div>{{ cur.host || '—' }} · {{ cur.gpu || '—' }}</div>
              <div>git</div><div class="mono">{{ cur.git_rev || '—（未纳入版本控制）' }}</div>
              <div>代码指纹</div>
              <div class="mono ellip" :title="cur.env?.code?.sha1 || ''">
                <template v-if="cur.env?.code">{{ cur.env.code.path }} · {{ (cur.env.code.sha1 || '').slice(0, 12) }}</template>
                <template v-else>—</template>
              </div>
              <div>环境</div>
              <div class="mono ellip">
                py{{ cur.env?.python || '?' }} · torch{{ cur.env?.torch || '?' }}
              </div>
              <div>数据集</div>
              <div class="mono ellip" :title="cur.dataset_id || ''">
                <template v-if="cur.dataset">
                  {{ cur.dataset.name }} · {{ (cur.dataset.snapshot_hash || '').slice(0, 12) || '无 hash' }}
                  <span class="dim">({{ cur.dataset.file_count }} 文件 · {{ cur.dataset.hash_alg || '—' }})</span>
                </template>
                <template v-else>{{ cur.dataset_id || '—' }}</template>
              </div>
              <div>模型</div>
              <div class="mono ellip" :title="cur.model_id || ''">
                <template v-if="cur.model">
                  {{ cur.model.name }} · {{ (cur.model.sha1 || '').slice(0, 12) }}
                  <span class="dim">({{ cur.model.backbone || '—' }})</span>
                </template>
                <template v-else>{{ cur.model_id || '—' }}</template>
              </div>
              <div>输入指纹</div>
              <div class="mono ellip" :title="cur.input_hash || ''">
                {{ (cur.input_hash || '').slice(0, 16) || '—（历史 run 未记录）' }}
              </div>
              <div>产物目录</div><div class="mono">{{ cur.out_dir || '—' }}</div>
            </div>
            <div v-if="cmd" class="cmd mono">{{ cmd }}</div>
          </div>
        </section>

        <section v-if="cur.kind === 'train'" class="panel">
          <div class="panel-h">
            训练曲线
            <span v-if="cur.experiment" class="tag">{{ cur.experiment.name }}</span>
            <div class="grow" />
            <RouterLink class="tiny link" :to="`/train?run=${encodeURIComponent(cur.id)}`">
              在训练页打开 →
            </RouterLink>
          </div>
          <div class="panel-b">
            <TrainCurve :series="cur.series || {}" :best-ep="cur.params?.best_ep ?? null" :height="170" />
          </div>
        </section>

        <section class="panel">
          <div class="panel-h">产物 <span class="tag">{{ cur.artifacts?.length || 0 }}</span>
            <div class="grow" />
            <span class="tiny dim">uri 相对仓库根，sha1 只对 ≤512MB 的文件计算</span>
          </div>
          <div class="panel-b">
            <div v-if="!cur.artifacts?.length" class="empty tiny">本次运行未登记产物（未指定 --out 或目录为空）</div>
            <div v-else class="table arts">
              <div class="th">文件</div><div class="th">类型</div><div class="th num">大小</div><div class="th">sha1</div>
              <template v-for="a in cur.artifacts" :key="a.id">
                <div class="mono ellip" :title="a.uri">{{ a.uri }}</div>
                <div class="dim">{{ a.kind }}</div>
                <div class="num">{{ size(a.bytes) }}</div>
                <div class="mono dim ellip">{{ a.sha1 ? a.sha1.slice(0, 12) : '—' }}</div>
              </template>
            </div>
          </div>
        </section>

        <div class="grid2">
          <section class="panel">
            <div class="panel-h">参数快照</div>
            <div class="panel-b">
              <div v-if="!params.length" class="empty tiny">无显式参数（用 --tag k=v 记录）</div>
              <div v-else class="kv">
                <template v-for="[k, v] in params" :key="k">
                  <div class="dim">{{ k }}</div><div class="mono ellip">{{ v }}</div>
                </template>
              </div>
            </div>
          </section>

          <section class="panel">
            <div class="panel-h">指标</div>
            <div class="panel-b">
              <div v-if="!cur.metrics" class="empty tiny">
                未找到 <span class="mono">out_dir/metrics.json</span> 或 <span class="mono">summary.json</span>
              </div>
              <template v-else>
                <div v-if="!metricRows.length" class="empty tiny">指标为空</div>
                <div v-else class="mtable">
                  <div v-for="(r, i) in metricRows" :key="i" class="mrow">
                    <span class="mk mono">{{ r.k }}</span>
                    <span class="mv mono">{{ r.v }}</span>
                    <span v-if="r.note" class="tiny dim mn">{{ r.note }}</span>
                  </div>
                </div>
                <details class="raw">
                  <summary class="tiny dim">原始 JSON</summary>
                  <pre class="mono pre">{{ JSON.stringify(cur.metrics, null, 1) }}</pre>
                </details>
              </template>
            </div>
          </section>
        </div>

        <section class="panel" style="min-height:120px">
          <div class="panel-h">日志尾部 <span class="tiny dim">{{ cur.log_uri || '无' }}</span></div>
          <div class="panel-b">
            <pre v-if="logs.length" class="mono pre logs">{{ logs.join('\n') }}</pre>
            <div v-else class="empty tiny">无日志</div>
          </div>
        </section>
      </template>
    </div>
  </div>
</template>

<style scoped>
.runs { display: grid; grid-template-columns: 300px minmax(0, 1fr); gap: var(--sp-3); padding: var(--sp-3); min-height: 0; height: 100%; }
@media (max-width: 1023px) { .runs { grid-template-columns: 1fr; height: auto; } }
.main { overflow: auto; padding-right: 2px; }
.item { display: flex; flex-direction: column; gap: 2px; width: 100%; text-align: left;
        background: transparent; border: 1px solid transparent; border-radius: var(--r-sm);
        padding: 6px 7px; color: inherit; cursor: pointer; }
.item:hover { background: var(--surface-2); }
.item.on { background: var(--accent-soft); border-color: var(--accent-line); }
.cmd { margin-top: var(--sp-2); padding: 7px 9px; background: var(--surface-inset);
       border-radius: var(--r-sm); font-size: var(--text-xs); color: var(--fg-2);
       overflow-x: auto; white-space: nowrap; }
.arts { grid-template-columns: minmax(0, 1fr) 52px 68px 106px; gap: 0 var(--sp-3); }
.arts > * { padding: 8px 0; border-bottom: 1px solid var(--line); min-width: 0; }
/* 详情里的值多为长字符串：左对齐 + 允许断行，比右对齐可读 */
:deep(.kv) > :nth-child(even) { text-align: left; word-break: break-all; }
.grid2 { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: var(--sp-3); }
@media (max-width: 1180px) { .grid2 { grid-template-columns: minmax(0, 1fr); } }
.pre { margin: 0; font-size: var(--text-xs); line-height: 1.6; color: var(--fg-2);
       white-space: pre-wrap; word-break: break-all; }
.mtable { display: flex; flex-direction: column; }
.mrow { display: grid; grid-template-columns: minmax(90px, 1fr) auto minmax(0, 1.2fr);
        gap: var(--sp-2); align-items: baseline; padding: 3px 0;
        border-bottom: 1px dashed var(--line); }
.mrow:last-child { border-bottom: 0; }
.mk { color: var(--fg-2); font-size: var(--text-xs); }
.mv { color: var(--fg-1); font-size: var(--text-sm); font-variant-numeric: tabular-nums;
      text-align: right; }
.mn { text-align: right; }
.raw { margin-top: var(--sp-3); }
.raw summary { cursor: pointer; }
.logs { max-height: 260px; overflow: auto; }
.link { color: var(--accent); text-decoration: none; }
.link:hover { text-decoration: underline; }
.link { color: var(--accent); text-decoration: none; }
.link:hover { text-decoration: underline; }
</style>
