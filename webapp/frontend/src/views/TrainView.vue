<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { api } from '../api'
import TrainCurve from '../components/TrainCurve.vue'

const route = useRoute()
const exps = ref<any[]>([])
const err = ref('')
const loading = ref(true)
const expId = ref('')
const runId = ref('')

onMounted(load)

async function load() {
  loading.value = true
  err.value = ''
  try {
    const r = await api.opsTrain()
    exps.value = r.experiments || []
    // 从运行页跳转过来时带 ?run=，定位到该变体所在批次
    const wanted = String(route.query.run || '')
    const hit = wanted
      ? exps.value.find(e => (e.variants || []).some((v: any) => v.id === wanted))
      : null
    if (hit) {
      expId.value = hit.id
      runId.value = wanted
      return
    }
    if (exps.value.length) {
      expId.value = exps.value[0].id
      const best = curExp.value?.variants?.[0]
      if (best) runId.value = best.id
    }
  } catch (e: any) {
    err.value = String(e?.message || e)
  } finally {
    loading.value = false
  }
}

const curExp = computed(() => exps.value.find(e => e.id === expId.value) || null)
const variants = computed(() => curExp.value?.variants || [])
const cur = computed(() => variants.value.find(v => v.id === runId.value) || variants.value[0] || null)

function pickExp(id: string) {
  expId.value = id
  const v = exps.value.find(e => e.id === id)?.variants || []
  runId.value = v[0]?.id || ''
}

// 表格里的迷你曲线
// ★ 按自身 min-max 缩放：固定 0~1 纵轴时，val_f1 只从 .45 变到 .49，
//   在 18px 高度里不足 1px → 12 个变体的缩略图全是同一条平线，看不出差异。
//   span 过小（<0.005）时按平线画，避免把噪声放大成假趋势。
function spark(series: any[]) {
  const arr = (series || []) as { step: number; value: number }[]
  if (arr.length < 2) return ''
  const w = 100, h = 18, pad = 1.5
  const vs = arr.map(p => p.value).filter(v => typeof v === 'number')
  if (!vs.length) return ''
  const lo = Math.min(...vs), hi = Math.max(...vs)
  const span = hi - lo
  const norm = (v: number) => (span < 0.005 ? 0.5 : (v - lo) / span)
  return arr.map((p, i) => {
    const x = pad + (i / (arr.length - 1)) * (w - pad * 2)
    const y = pad + (h - pad * 2) * (1 - Math.max(0, Math.min(1, norm(p.value))))
    return `${i ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`
  }).join(' ')
}
function sparkTitle(v: any) {
  const arr = (v?.series?.val_f1 || []) as { value: number }[]
  if (arr.length < 2) return '无曲线'
  const lo = Math.min(...arr.map(p => p.value)), hi = Math.max(...arr.map(p => p.value))
  return hi - lo < 0.005 ? `val_f1 基本不变（${lo.toFixed(3)}）`
    : `val_f1 ${lo.toFixed(3)} → ${hi.toFixed(3)}（按自身范围缩放）`
}

const f4 = (v: any) => (typeof v === 'number' ? v.toFixed(3) : '—')
// 新训练脚本的 params 里没有 epochs/best_ep，退回曲线与 metrics
const nEp = (v: any) => {
  const s = v?.series || {}
  const steps = [...(s.val_f1 || []), ...(s.val_acc || []), ...(s.loss || [])].map((p: any) => p.step)
  return steps.length ? Math.max(...steps) : '—'
}
// 最佳 epoch：优先脚本显式记录；没有就从 val_f1 曲线取峰值点（metrics_full 的 rep 结构不带 best_ep）
const peak = (v: any) => {
  const s = v?.series?.val_f1 || []
  if (!s.length) return null
  return s.reduce((a: any, b: any) => (b.value > a.value ? b : a), s[0])
}
const bestEp = (v: any) => v?.params?.best_ep ?? v?.metrics?.best_ep ?? peak(v)?.step ?? '—'
const bestValF1 = (v: any) => {
  const hit = v?.params?.best_val_f1 ?? v?.metrics?.best_val_f1 ?? peak(v)?.value
  return hit == null ? null : f4(hit)
}
const cfgRows = computed(() => {
  const p = cur.value?.params || {}
  return [
    ['变体', p.variant], ['任务', p.task], ['身份模式', p.id_mode],
    ['epochs', p.epochs ?? (cur.value ? nEp(cur.value) : null)],
    ['最佳 epoch', cur.value ? bestEp(cur.value) : p.best_ep],
    ['最佳 val_f1', cur.value ? bestValF1(cur.value) : null],
    ['train / val', p.n_train != null ? `${p.n_train} / ${p.n_val}` : null],
    ['test', p.n_test], ['模型参数', p.params], ['设备', p.device],
  ].filter(([, v]) => v !== null && v !== undefined && v !== '') as [string, any][]
})
const camRows = computed(() => {
  const pc = cur.value?.metrics?.per_cam || {}
  return Object.entries(pc).filter(([, v]) => typeof v === 'number') as [string, number][]
})
</script>

<template>
  <div class="train">
    <!-- 左：实验批次 -->
    <aside class="col side" style="gap:var(--sp-2);min-height:0">
      <section class="panel" style="min-height:0;display:flex;flex-direction:column">
        <div class="panel-h">
          训练批次
          <span class="tag">{{ exps.length }}</span>
          <div class="grow" />
          <button class="ghost tiny" @click="load" title="刷新">刷新</button>
        </div>
        <div class="scroll grow" style="padding:var(--sp-2)">
          <div v-if="err" class="empty tiny">{{ err }}</div>
          <div v-else-if="!exps.length && !loading" class="empty tiny">
            还没有训练记录。用 <span class="mono">tools/ops_ingest_train.py</span> 回填历史，或在训练脚本里写 <span class="mono">run_metric</span>。
          </div>
          <button v-for="e in exps" :key="e.id" class="item" :class="{ on: expId === e.id }" @click="pickExp(e.id)">
            <div class="row" style="gap:6px">
              <span class="ellip small" style="font-weight:600;color:var(--fg)">{{ e.name }}</span>
              <div class="grow" />
              <span class="tag">{{ e.n_runs }}</span>
            </div>
            <div class="tiny dim ellip" :title="e.conclusion">{{ e.conclusion || '未汇总结论' }}</div>
            <div class="tiny dim mono">{{ (e.last_at || '').slice(0, 16) }}</div>
          </button>
        </div>
      </section>
    </aside>

    <!-- 右：变体 + 曲线 + 配置 -->
    <div class="col main" style="gap:var(--sp-3);min-height:0">
      <section class="panel">
        <div class="panel-h">
          变体
          <span class="tag">{{ variants.length }}</span>
          <div class="grow" />
          <span class="tiny dim">按 macro_f1 降序 · 点行看曲线</span>
        </div>
        <div class="panel-b">
          <div v-if="!variants.length" class="empty tiny">该批次无变体</div>
          <div v-else class="table vars">
            <div class="th">变体</div><div class="th num">epoch</div><div class="th num">macro_f1</div>
            <div class="th num">acc</div><div class="th num">best ep</div><div class="th">val_f1</div>
            <template v-for="v in variants" :key="v.id">
              <button class="rowbtn mono ellip" :class="{ on: cur?.id === v.id }" :title="v.name"
                      @click="runId = v.id">{{ v.name }}</button>
              <div class="num mono">{{ v.params?.epochs ?? nEp(v) }}</div>
              <div class="num mono">{{ f4(v.metrics?.macro_f1) }}</div>
              <div class="num mono dim">{{ f4(v.metrics?.acc) }}</div>
              <div class="num mono dim">{{ bestEp(v) }}</div>
              <svg class="spark" viewBox="0 0 100 18" preserveAspectRatio="none">
                <title>{{ sparkTitle(v) }}</title>
                <path :d="spark(v.series?.val_f1)" fill="none" stroke="var(--accent)" stroke-width="1.2"
                      vector-effect="non-scaling-stroke" />
              </svg>
            </template>
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-h">
          训练曲线
          <span v-if="cur" class="mono tiny dim">{{ cur.name }}</span>
          <div class="grow" />
        </div>
        <div class="panel-b">
          <TrainCurve :series="cur?.series || {}" :best-ep="cur?.params?.best_ep ?? null"
                      :height="230" show-loss />
        </div>
      </section>

      <div class="grid2">
        <section class="panel">
          <div class="panel-h">训练配置</div>
          <div class="panel-b">
            <div v-if="!cfgRows.length" class="empty tiny">无配置记录</div>
            <div v-else class="kv">
              <template v-for="[k, v] in cfgRows" :key="k">
                <div class="dim">{{ k }}</div><div class="mono ellip">{{ v }}</div>
              </template>
            </div>
          </div>
        </section>

        <section class="panel">
          <div class="panel-h">终值指标</div>
          <div class="panel-b">
            <div v-if="!cur?.metrics" class="empty tiny">无</div>
            <template v-else>
              <div class="kv">
                <div class="dim">测试 acc</div><div class="mono">{{ f4(cur.metrics.acc) }}</div>
                <div class="dim">macro_f1</div><div class="mono">{{ f4(cur.metrics.macro_f1) }}</div>
                <div class="dim">样本 n</div><div class="mono">{{ cur.metrics.n ?? '—' }}</div>
              </div>
              <div v-if="camRows.length" class="cams">
                <div v-for="[c, v] in camRows" :key="c" class="camrow">
                  <span class="mono tiny dim">cam{{ c }}</span>
                  <div class="minibar"><i :style="{ width: Math.max(0, v) * 100 + '%' }" /></div>
                  <span class="mono tiny">{{ f4(v) }}</span>
                </div>
              </div>
            </template>
          </div>
        </section>
      </div>
    </div>
  </div>
</template>

<style scoped>
.train { display: grid; grid-template-columns: 292px minmax(0, 1fr); gap: var(--sp-3);
         padding: var(--sp-3); min-height: 0; height: 100%; }
@media (max-width: 1023px) { .train { grid-template-columns: 1fr; height: auto; } }
.main { overflow: auto; padding-right: 2px; }
.item { display: flex; flex-direction: column; gap: 3px; width: 100%; text-align: left;
        background: transparent; border: 1px solid transparent; border-radius: var(--r-sm);
        padding: 7px; color: inherit; cursor: pointer; margin-bottom: 2px; }
.item:hover { background: var(--surface-2); }
.item.on { background: var(--accent-soft); border-color: var(--accent-line); }

.vars { grid-template-columns: 232px 56px 82px 60px 62px minmax(110px, 1fr); gap: 0 var(--sp-2); }
.vars > * { padding: 5px 0; border-bottom: 1px solid var(--line); min-width: 0; }
.rowbtn { width: 100%; justify-content: flex-start; text-align: left; background: transparent;
          border: 0; color: var(--fg); cursor: pointer; font-size: var(--text-xs);
          padding: 5px 6px; border-radius: var(--r-sm); min-height: 24px; }
.rowbtn:hover { background: var(--surface-2); }
.rowbtn.on { background: var(--accent-soft); color: var(--fg); font-weight: 600; }
.spark { width: 100%; max-width: 240px; height: 18px; align-self: center; }

.grid2 { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: var(--sp-3); }
@media (max-width: 1180px) { .grid2 { grid-template-columns: minmax(0, 1fr); } }
.cams { margin-top: var(--sp-2); display: grid; gap: 4px; }
.camrow { display: grid; grid-template-columns: 52px minmax(0, 1fr) 42px; gap: var(--sp-2); align-items: center; }
.minibar { height: 5px; background: var(--surface-3); border-radius: 3px; overflow: hidden; }
.minibar i { display: block; height: 100%; background: var(--accent); }
</style>
