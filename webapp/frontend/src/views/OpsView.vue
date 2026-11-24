<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { api } from '../api'

const ov = ref<any>(null)
const err = ref('')
const loading = ref(true)

onMounted(load)

async function load() {
  loading.value = true
  err.value = ''
  try {
    ov.value = await api.opsOverview()
  } catch (e: any) {
    err.value = String(e?.message || e)
  } finally {
    loading.value = false
  }
}

const ann = computed(() => ov.value?.annotation || {})
const runs = computed(() => ov.value?.runs || {})
const train = computed(() => ov.value?.train || {})
const recent = computed<any[]>(() => runs.value.recent || [])
const jobs = computed(() => ov.value?.jobs || {})
const counts = computed(() => ov.value?.counts || {})

// 待判最多的前 5 个任务（总览只放最需要动的）
const topTasks = computed<any[]>(() =>
  [...(ann.value.by_task || [])].sort((a, b) => (b.pending || 0) - (a.pending || 0)).slice(0, 6))

const risks = computed(() => {
  const out: { level: string; text: string; hint: string }[] = []
  if (!ov.value) return out
  if ((runs.value.stale || 0) > 0) {
    out.push({ level: 'warn', text: `${runs.value.stale} 个 run 已过期（stale）`, hint: '输入或模型变了但未重跑，产物不可与新结果混用' })
  }
  if ((runs.value.failed || 0) > 0) {
    out.push({ level: 'bad', text: `${runs.value.failed} 个 run 失败`, hint: '查看退出码与日志尾部' })
  }
  if ((runs.value.running || 0) > 0) {
    out.push({ level: 'info', text: `${runs.value.running} 个 run 正在运行`, hint: '完成后才会登记产物与指标' })
  }
  if ((ann.value.pending || 0) > 0) {
    out.push({ level: 'info', text: `${ann.value.pending} 条标注待判`, hint: '按待判数排序，优先清队首任务' })
  }
  // 派发已决定不纳入本期，不再作为"风险"常驻提示（卡片仍在 #/jobs 可查）
  if ((train.value.runs || 0) > 0) {
    out.push({
      level: 'info',
      text: `${train.value.runs} 个训练变体来自历史回填`,
      hint: '日志里没有 commit 与配置快照，只能对齐指标，不能证明可复现；新训练请走 ops run 包装器',
    })
  }
  if (!out.length) out.push({ level: 'ok', text: '无阻塞项', hint: '台账与运行状态均正常' })
  return out
})

function pctOf(t: any) { return Math.max(0, Math.min(100, t.pct || 0)) }
function dur(r: any) {
  if (!r.started_at || !r.ended_at) return '—'
  const a = Date.parse(r.started_at.replace(' ', 'T'))
  const b = Date.parse(r.ended_at.replace(' ', 'T'))
  if (!isFinite(a) || !isFinite(b)) return '—'
  const s = Math.max(0, Math.round((b - a) / 1000))
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m${String(s % 60).padStart(2, '0')}s`
}
const statusTag = (s: string) =>
  ({ succeeded: 'ok', running: 'info', failed: 'bad', killed: 'warn', stale: 'warn' } as any)[s] || ''
</script>

<template>
  <div class="ops col" style="gap:var(--sp-3);min-height:0">
    <div v-if="err" class="panel">
      <div class="panel-h">台账不可用 <span class="tag bad">503</span></div>
      <div class="panel-b col" style="gap:8px">
        <p class="small">{{ err }}</p>
        <p class="tiny dim">
          台账为本机主库（<span class="mono">ops/ops.db</span>）。远程只做算力与预览，
          未部署台账属预期行为。在本机执行 <span class="mono">python tools/ops_db.py init</span>
          与 <span class="mono">python tools/ops_ingest.py</span> 后刷新。
        </p>
        <div><button class="primary" @click="load">重试</button></div>
      </div>
    </div>

    <template v-else-if="loading">
      <div class="panel"><div class="panel-b row" style="gap:8px"><span class="spinner" /><span class="small dim">读取台账…</span></div></div>
    </template>

    <template v-else>
      <!-- KPI -->
      <div class="kpis">
        <section class="panel kpi">
          <div class="panel-b metric">
            <b>{{ ann.pending ?? 0 }}</b>
            <span>待判条目</span>
            <div class="scorebar" style="margin-top:6px"><i :style="{ width: (100 - (ann.pct || 0)) + '%' }" /></div>
          </div>
        </section>
        <section class="panel kpi">
          <div class="panel-b metric">
            <b>{{ ann.done ?? 0 }}<em class="tiny dim" style="font-style:normal"> / {{ ann.items ?? 0 }}</em></b>
            <span>已判 · 完成 {{ ann.pct ?? 0 }}%</span>
            <div class="scorebar" style="margin-top:6px"><i :style="{ width: (ann.pct || 0) + '%' }" /></div>
          </div>
        </section>
        <section class="panel kpi">
          <div class="panel-b metric">
            <b>{{ train.runs ?? 0 }}</b>
            <span>训练变体 · {{ train.experiments ?? 0 }} 个批次</span>
            <div class="scorebar" style="margin-top:6px"><i style="width:100%" /></div>
          </div>
        </section>
        <section class="panel kpi">
          <div class="panel-b metric">
            <b>{{ runs.total ?? 0 }}</b>
            <span>运行记录 · 进行中 {{ runs.running ?? 0 }}</span>
            <div class="row tiny" style="margin-top:6px;gap:6px">
              <span class="tag" :class="runs.failed ? 'bad' : 'ok'">失败 {{ runs.failed ?? 0 }}</span>
              <span class="tag" :class="runs.stale ? 'warn' : 'ok'">stale {{ runs.stale ?? 0 }}</span>
            </div>
          </div>
        </section>
        <section class="panel kpi">
          <div class="panel-b metric">
            <b>{{ counts.dataset ?? 0 }}<em class="tiny dim" style="font-style:normal"> 数据集 · {{ counts.model ?? 0 }} 权重</em></b>
            <span>血缘已登记（run → 数据 → 权重）</span>
            <div class="tiny dim" style="margin-top:6px">判读任务 <b>{{ jobs.total ?? 0 }}</b> 张 · 派发不纳入本期</div>
          </div>
        </section>
      </div>

      <div class="grid">
        <!-- 左：待判队列 -->
        <section class="panel">
          <div class="panel-h">
            标注任务 · 按待判数排序
            <div class="grow" />
            <RouterLink to="/annotate" class="tiny" style="color:var(--accent)">去标注 →</RouterLink>
          </div>
          <div class="panel-b">
            <div class="table tasks">
              <div class="th">任务</div><div class="th num">已判 / 总</div>
              <div class="th" style="min-width:88px">进度</div><div class="th num">待判</div>
              <template v-for="t in topTasks" :key="t.task_id">
                <div class="ellip" :title="t.name">{{ t.name }}</div>
                <div class="num">{{ t.done }} / {{ t.items }}</div>
                <div class="pcell">
                  <span class="tiny dim tabular">{{ pctOf(t).toFixed(0) }}%</span>
                  <div class="pbar"><i :style="{ width: pctOf(t) + '%' }" /></div>
                </div>
                <div class="num" :class="t.pending > 0 ? '' : 'dim'">{{ t.pending }}</div>
              </template>
            </div>
            <p class="tiny dim" style="margin-top:var(--sp-2)">
              台账口径：候选全集入库，未判即为缺口；同一键在不同历史文件中的判定已合并。
            </p>
          </div>
        </section>

        <!-- 右：最近运行 -->
        <section class="panel">
          <div class="panel-h">
            最近运行
            <div class="grow" />
            <RouterLink to="/runs" class="tiny" style="color:var(--accent)">全部 →</RouterLink>
          </div>
          <div class="panel-b col" style="gap:2px">
            <div v-if="!recent.length" class="empty tiny">还没有 run。用 <span class="mono">ops_run.py</span> 包一次命令即可。</div>
            <RouterLink v-for="r in recent" :key="r.id" :to="'/runs?id=' + r.id" class="runrow">
              <span class="tag" :class="statusTag(r.status)">{{ r.status }}</span>
              <span class="ellip mono small">{{ r.name }}</span>
              <div class="grow" />
              <span class="tiny dim">{{ r.kind }}</span>
              <span class="tiny dim tabular">{{ dur(r) }}</span>
            </RouterLink>
          </div>
        </section>
      </div>

      <!-- 风险 -->
      <section class="panel">
        <div class="panel-h">风险与待办</div>
        <div class="panel-b col" style="gap:2px">
          <div v-for="(r, i) in risks" :key="i" class="riskrow">
            <span class="tag" :class="r.level">{{ r.level === 'ok' ? '正常' : r.level === 'bad' ? '阻塞' : r.level === 'warn' ? '注意' : '提示' }}</span>
            <div class="col" style="gap:0">
              <span class="small">{{ r.text }}</span>
              <span class="tiny dim">{{ r.hint }}</span>
            </div>
          </div>
        </div>
      </section>

      <p class="tiny dim">
        权威库 <span class="mono">{{ ov.db }}</span> · 生成于 {{ ov.generated_at }} · 口径
        <span class="mono">local-first</span>（远程只做算力与预览）
      </p>
    </template>
  </div>
</template>

<style scoped>
.ops { padding: var(--sp-3) var(--sp-3) var(--sp-4); overflow: auto; }
.kpis { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: var(--sp-2); align-items: stretch; }
.kpi { display: flex; flex-direction: column; }
.kpi .panel-b { padding: var(--sp-3); flex: 1; }
@media (max-width: 900px) { .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.kpi .metric > b em { font-weight: 400; }
.grid { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr); gap: var(--sp-3); }
@media (max-width: 1023px) { .grid { grid-template-columns: minmax(0, 1fr); } }
.tasks { grid-template-columns: minmax(0, 1fr) 72px 116px 40px; align-items: center; }
.tasks > * { padding: 8px 0; }
.pcell { display: flex; align-items: center; gap: 6px; min-width: 0; }
.pbar { flex: 1; height: 6px; background: var(--surface-3); border-radius: 3px; overflow: hidden; min-width: 36px; }
.pbar > i { display: block; height: 100%; background: var(--accent); border-radius: 3px; }
:deep(.scorebar) { height: 6px; background: var(--surface-3); }
:deep(.scorebar > i) { background: var(--accent); min-width: 2px; }
.runrow { display: flex; align-items: center; gap: var(--sp-2); padding: 5px 6px; border-radius: var(--r-sm); color: inherit; }
.runrow:hover { background: var(--surface-2); text-decoration: none; }
.riskrow { display: flex; gap: var(--sp-2); align-items: flex-start; padding: 5px 0; border-bottom: 1px solid var(--line); }
.riskrow:last-child { border-bottom: 0; }
</style>
