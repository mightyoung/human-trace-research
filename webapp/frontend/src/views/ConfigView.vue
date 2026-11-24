<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed, nextTick, watch } from 'vue'
import * as echarts from 'echarts'
import { api } from '../api'
import { theme } from '../theme'

const DEFAULTS = { sp_threshold: 0.125, auto_reid_min: 0.5, gray_mult: 2.0, min_covis_sec: 0, min_sp_n: 0 }
const params = ref<any>({ ...DEFAULTS })
const result = ref<any>(null)
const running = ref(false)
const saved = ref(false)
const sweepKey = ref('sp_threshold')
const sweepPoints = ref<any[]>([])
const gil = ref<any>(null)
const chartEl = ref<HTMLElement>()
let chart: echarts.ECharts | null = null

const paramDefs = [
  { k: 'sp_threshold', label: '时空残差门', min: 0.02, max: 0.4, step: 0.005, hint: '位置映射残差中位数阈值，越小越严' },
  { k: 'auto_reid_min', label: '外观门', min: 0.3, max: 0.9, step: 0.01, hint: 'AUTO 层要求的外观余弦下限' },
  { k: 'gray_mult', label: '灰区倍数', min: 1, max: 4, step: 0.1, hint: 'REVIEW 灰区 = sp_threshold × 该倍数' },
  { k: 'min_covis_sec', label: '最小共视时长', min: 0, max: 10, step: 0.5, hint: '低于该共视时长直接 DROP', unit: 's' },
  { k: 'min_sp_n', label: '最小样本数', min: 0, max: 20, step: 1, hint: '时空配对样本数下限', unit: '个' }
]

onMounted(async () => {
  try { params.value = { ...params.value, ...(await api.config()) } } catch { /* 首次运行无 config.json */ }
  try { gil.value = await api.gil() } catch { /* GIL 概况可选 */ }
  await run()
  startSweep()
})
onUnmounted(() => { chart?.dispose(); window.removeEventListener('resize', resizable) })
window.addEventListener('resize', resizable)
function resizable() { chart?.resize() }

async function run() {
  running.value = true
  try { result.value = await api.evaluate(params.value) } finally { running.value = false }
}
async function save() {
  await api.saveConfig(params.value)
  saved.value = true
  setTimeout(() => (saved.value = false), 1600)
}
function resetDefaults() { params.value = { ...DEFAULTS }; run() }

const inference = computed(() => result.value?.inference_params || {})
const validationRows = computed(() => {
  const v = result.value?.validation || {}
  return Object.keys(v).map(k => ({ tier: k, ...v[k] }))
})

async function startSweep() {
  const key = sweepKey.value
  const values = key === 'sp_threshold' ? [0.03, 0.05, 0.08, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4]
    : key === 'auto_reid_min' ? [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
      : [1, 1.5, 2, 2.5, 3, 4]
  const r = await api.sweep(params.value, key, values)
  sweepPoints.value = r.points
  await nextTick()
  renderChart()
}

/** 从主题变量取色，保证浅色/深色主题下图表都可读 */
function cssVar(name: string, fallback: string) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return v || fallback
}
function renderChart() {
  if (!chartEl.value) return
  if (!chart) chart = echarts.init(chartEl.value, undefined, { renderer: 'canvas' })
  const fg2 = cssVar('--fg-2', '#a7b3c2')
  const fg3 = cssVar('--fg-3', '#6b7987')
  const line = cssVar('--line', '#2a3340')
  const ok = cssVar('--ok', '#38d39f')
  const warn = cssVar('--warn', '#f2b138')
  const accent = cssVar('--accent', '#4ea1ff')
  const pts = sweepPoints.value
  chart.setOption({
    backgroundColor: 'transparent',
    grid: { left: 46, right: 52, top: 36, bottom: 30 },
    tooltip: { trigger: 'axis' },
    legend: { top: 2, textStyle: { color: fg2 }, data: ['AUTO', 'REVIEW', 'DROP', 'AUTO 精度'] },
    xAxis: {
      type: 'category', data: pts.map(p => p.value), name: sweepKey.value,
      nameTextStyle: { color: fg3 }, axisLabel: { color: fg2 },
      axisLine: { lineStyle: { color: line } }
    },
    yAxis: [
      { type: 'value', axisLabel: { color: fg2 }, splitLine: { lineStyle: { color: line } } },
      { type: 'value', min: 0, max: 1, axisLabel: { color: fg2 }, splitLine: { show: false } }
    ],
    series: [
      { name: 'AUTO', type: 'line', smooth: true, symbolSize: 5, data: pts.map(p => p.AUTO), itemStyle: { color: ok } },
      { name: 'REVIEW', type: 'line', smooth: true, symbolSize: 5, data: pts.map(p => p.REVIEW), itemStyle: { color: warn } },
      { name: 'DROP', type: 'line', smooth: true, symbolSize: 5, data: pts.map(p => p.DROP), itemStyle: { color: fg3 } },
      {
        name: 'AUTO 精度', type: 'line', yAxisIndex: 1, smooth: true, symbolSize: 5,
        data: pts.map(p => p.auto_precision), itemStyle: { color: accent }, lineStyle: { type: 'dashed' }
      }
    ]
  }, true)
  chart.resize()
}
function onSweepKey() { startSweep() }
watch(theme, () => nextTick().then(renderChart))
</script>

<template>
  <div class="cfg">
    <!-- 左：参数 -->
    <div class="col side" style="gap:var(--sp-3);min-height:0">
      <section class="panel">
        <div class="panel-h">
          跨机判定参数
          <span class="tag info">可在线重算</span>
          <div class="grow" />
          <button class="ghost tiny" @click="resetDefaults" title="恢复默认值并重算">恢复默认</button>
        </div>
        <div class="panel-b col" style="gap:var(--sp-3)">
          <div v-for="d in paramDefs" :key="d.k" class="pdef">
            <div class="row">
              <label class="small" :for="'p-' + d.k">{{ d.label }}</label>
              <div class="grow" />
              <span class="mono tiny dim">{{ params[d.k] }}{{ d.unit ? ' ' + d.unit : '' }}</span>
              <input :id="'p-' + d.k" type="number" v-model.number="params[d.k]"
                     :min="d.min" :max="d.max" :step="d.step" class="num-in" />
            </div>
            <input type="range" v-model.number="params[d.k]" :min="d.min" :max="d.max" :step="d.step"
                   :aria-label="d.label + ' 滑块'" />
            <div class="tiny dim">{{ d.hint }}</div>
          </div>
          <div class="row" style="gap:var(--sp-2)">
            <button class="primary grow" @click="run" :disabled="running">
              <span v-if="running" class="spinner" />{{ running ? '重算中…' : '运行重算' }}
            </button>
            <button @click="save">{{ saved ? '已保存 ✓' : '保存配置' }}</button>
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-h">检测 / 跟踪级参数 <span class="tag warn">需 GPU 重跑</span></div>
        <div class="panel-b">
          <div class="kv">
            <template v-for="(v, k) in inference" :key="k"><div>{{ k }}</div><div>{{ v }}</div></template>
          </div>
          <p class="tiny dim" style="margin-top:var(--sp-2)">
            这些参数影响 GroundingDINO 推理与去重/起轨，改动必须在远程 GPU 重跑推理。
            本页不伪造结果，只展示当前生效值。
          </p>
        </div>
      </section>

      <section class="panel">
        <div class="panel-h">数据口径</div>
        <div class="panel-b col" style="gap:5px">
          <div class="row tiny"><span class="dim">检测视图</span><div class="grow" /><code>DET_VIEW = clean2</code></div>
          <div class="row tiny"><span class="dim">清洗集</span><div class="grow" /><code>cleanset_v2</code></div>
          <div class="row tiny"><span class="dim">实体图</span><div class="grow" /><code>stitch_clean2</code></div>
          <div class="row tiny"><span class="dim">接力链</span><div class="grow" /><code>cradio_v5</code></div>
          <p class="tiny dim" style="margin-top:4px">切换数据集需重新构建数据包（<span class="mono">dataprep/build_web_data.py</span>）。</p>
        </div>
      </section>

      <section class="panel">
        <div class="panel-h">
          GIL 统一存储
          <span class="tag info mono">v1</span>
          <div class="grow" />
          <span v-if="gil?.available && gil.created" class="sub mono">{{ gil.created }}</span>
        </div>
        <div v-if="gil?.available" class="panel-b col" style="gap:5px">
          <div class="row tiny"><span class="dim">观测 / 片段 / 实体</span><div class="grow" />
            <code>{{ gil.tables?.obs }} / {{ gil.tables?.frag }} / {{ gil.tables?.entity }}</code></div>
          <div class="row tiny"><span class="dim">边 relay active / audit</span><div class="grow" />
            <code>{{ gil.edges_by?.['relay:active'] || 0 }} / {{ gil.edges_by?.['relay:audit'] || 0 }}</code></div>
          <div class="row tiny"><span class="dim">边 v5 / 同机拼接</span><div class="grow" />
            <code>{{ gil.edges_by?.['relay_v5:active'] || 0 }} / {{ gil.edges_by?.['stitch:active'] || 0 }}</code></div>
          <div class="row tiny"><span class="dim">统一标注 / 缺嵌入边</span><div class="grow" />
            <code>{{ gil.tables?.annotation }} / {{ gil.skipped_edges }}</code></div>
          <div class="row tiny"><span class="dim">全局人 / 访问 / 责任</span><div class="grow" />
            <code class="dim">{{ gil.tables?.gperson }} / {{ gil.tables?.visit }} / {{ gil.tables?.owner }} · 待建</code></div>
          <p class="tiny dim" style="margin-top:4px">
            模型 {{ gil.meta?.model_version }} · {{ gil.meta?.dim }}d · 库哈希
            <span class="mono">{{ gil.files?.['gil.sqlite'] }}</span>
          </p>
        </div>
        <div v-else class="panel-b">
          <p class="tiny dim">未检测到 GIL 存储（<span class="mono">analysis/gil/store_v1</span>）。
            运行 <span class="mono">tools/gil_migrate.py</span> 并重建数据包后并入。</p>
        </div>
      </section>
    </div>

    <!-- 右：结果 -->
    <div class="col grow" style="gap:var(--sp-3);min-height:0">
      <section class="panel">
        <div class="panel-h">
          当前参数下的分层结果
          <div class="grow" />
          <span class="sub mono">共 {{ (result?.summary?.AUTO || 0) + (result?.summary?.REVIEW || 0) + (result?.summary?.DROP || 0) }} 对候选</span>
        </div>
        <div class="panel-b">
          <div class="strip" v-if="result">
            <div class="metric"><b class="c-ok">{{ result.summary.AUTO }}</b><span>AUTO 自动通过</span></div>
            <div class="metric"><b class="c-warn">{{ result.summary.REVIEW }}</b><span>REVIEW 待复核</span></div>
            <div class="metric"><b class="dim">{{ result.summary.DROP }}</b><span>DROP 丢弃</span></div>
            <div class="metric">
              <b class="c-accent">{{ result.auto_precision === null ? '—' : (result.auto_precision * 100).toFixed(0) + '%' }}</b>
              <span>AUTO 精度</span>
            </div>
            <div class="metric"><b>{{ result.auc_reid_cos?.toFixed(3) ?? '—' }}</b><span>外观 AUC</span></div>
          </div>
        </div>
      </section>

      <section class="panel chart-panel">
        <div class="panel-h">
          阈值扫描
          <label class="sr-only" for="sk">扫描参数</label>
          <select id="sk" v-model="sweepKey" class="mini-sel" @change="onSweepKey">
            <option value="sp_threshold">时空残差门</option>
            <option value="auto_reid_min">外观门</option>
            <option value="gray_mult">灰区倍数</option>
          </select>
          <div class="grow" />
          <span class="tiny dim hide-sm">柱线为各层数量随阈值变化，虚线为 AUTO 精度</span>
        </div>
        <div ref="chartEl" class="chart grow"></div>
      </section>

      <section class="panel">
        <div class="panel-h">分层 × 人工标签对照 <span class="sub">用于检验自动分层是否可靠</span></div>
        <div class="panel-b">
          <div class="table val-table">
            <div class="th">层</div><div class="th">确认</div><div class="th">未决</div><div class="th">否决</div><div class="th">未标</div>
            <template v-for="r in validationRows" :key="r.tier">
              <div class="bold">{{ r.tier }}</div>
              <div class="num c-ok">{{ r.CONF }}</div>
              <div class="num c-warn">{{ r.MODERATE }}</div>
              <div class="num c-bad">{{ r.REJ }}</div>
              <div class="num dim">{{ r.UNLABELED }}</div>
            </template>
          </div>
          <p v-if="result?.note" class="tiny dim" style="margin-top:var(--sp-2)">{{ result.note }}</p>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.cfg {
  display: grid; grid-template-columns: 352px minmax(0, 1fr);
  gap: var(--sp-3); padding: var(--sp-3); height: 100%; min-height: 0;
}
.side { overflow: auto; }
.pdef { display: flex; flex-direction: column; gap: 3px; }
.pdef input[type='range'] { width: 100%; accent-color: var(--accent); }
.num-in { width: 76px; min-height: 24px; padding: 2px 6px; font-size: var(--text-xs); flex: none; }
.mini-sel { width: 168px; min-height: 24px; padding: 2px 6px; font-size: var(--text-xs); flex: none; }

.chart-panel { flex: 1; min-height: 0; display: flex; flex-direction: column; }
.chart { min-height: 200px; width: 100%; }

.strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(108px, 1fr)); }
.strip .metric { padding: 0 var(--sp-3); border-left: 1px solid var(--line); }
.strip .metric:first-child { padding-left: 0; border-left: 0; }
.c-ok { color: var(--ok); }
.c-warn { color: var(--warn); }
.c-bad { color: var(--danger); }
.c-accent { color: var(--accent); }

.val-table { grid-template-columns: 1.2fr repeat(4, minmax(0, 0.8fr)); max-width: 520px; }

@media (max-width: 1023px) {
  .cfg { grid-template-columns: minmax(0, 1fr); overflow: auto; }
  .side { overflow: visible; }
}
</style>
