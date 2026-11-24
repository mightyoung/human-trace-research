<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { api } from '../api'

/**
 * 科研工作台：实验卡 + 四关审批 + 人机分工。
 * 口径（来自 2026-11-23 研究报告 §08/§09/§10）：
 *  - 卡只写「问题/基线/候选/主指标/失败条件/预算」，不写成绩；
 *  - 关口只记人的决定，append-only；同意才推进，退回必须写理由；
 *  - 七件套只是证据索引，勾上不等于验收通过。
 */

const cards = ref<any[]>([])
const meta = ref<any>({ gates: {}, decisions: {}, states: {} })
const sel = ref('')
const err = ref('')
const msg = ref('')
const loading = ref(true)
const busy = ref('')

const form = ref<{ gate: string; decision: string; note: string } | null>(null)
const runId = ref('')
const runRole = ref('baseline')

const GATE_ORDER = ['A', 'B', 'C', 'D']
const TONE: Record<string, string> = {
  adopted: 'ok', archived: '', draft: '', proposed: 'warn', approved: 'info',
  building: 'info', testing: 'info', running: 'info', scored: 'info',
  reviewed: 'warn', accept_pending: 'warn', revise: 'bad', paused: 'warn',
  failed: 'bad'
}

onMounted(load)

async function load() {
  loading.value = true
  err.value = ''
  try {
    const r = await api.wbCards()
    cards.value = r.cards || []
    meta.value = r
    if (!cards.value.some(c => c.id === sel.value)) sel.value = cards.value[0]?.id || ''
  } catch (e: any) {
    err.value = String(e?.message || e)
  } finally {
    loading.value = false
  }
}

const cur = computed<any>(() => cards.value.find(c => c.id === sel.value) || null)
const pending = computed(() => cards.value.filter(c => needsHuman(c)))
const gateList = computed(() => GATE_ORDER.map(k => ({ k, ...(meta.value.gates[k] || {}) })))
const evidenceDone = computed(() =>
  cards.value.reduce((n, c) => n + (c.evidence || []).filter((i: any) => i.done).length, 0))
const evidenceTotal = computed(() =>
  cards.value.reduce((n, c) => n + (c.evidence || []).length, 0))

function needsHuman(c: any) {
  if (['proposed', 'accept_pending', 'reviewed'].includes(c.state)) return true
  return !c.gate_state?.A
}

function gateOf(c: any, k: string) {
  return c?.gate_state?.[k] || null
}

function decTone(dec?: string) {
  return dec === 'approve' ? 'ok' : dec === 'return' ? 'bad' : dec === 'hold' ? 'warn' : ''
}

function decLabel(dec?: string) {
  return dec ? (meta.value.decisions[dec] || dec) : '未决'
}

function stateLabel(s: string) {
  return meta.value.states?.[s] || s
}

function nextStates(c: any) {
  return c.next_states || []
}

function openGate(k: string) {
  form.value = form.value?.gate === k
    ? null
    : { gate: k, decision: 'approve', note: '' }
}

async function submitGate() {
  if (!cur.value || !form.value) return
  const f = form.value
  busy.value = 'gate'
  err.value = ''
  msg.value = ''
  try {
    const r = await api.wbGate(cur.value.id, { gate: f.gate, decision: f.decision, note: f.note })
    replace(r.card)
    msg.value = `${cur.value.id} 关口 ${f.gate} → ${decLabel(f.decision)}` +
      (r.moved_to ? `，状态联动为「${stateLabel(r.moved_to)}」` : '')
    form.value = null
  } catch (e: any) {
    err.value = `记录失败：${String(e?.message || e)}`
  } finally {
    busy.value = ''
  }
}

async function advance(to: string) {
  if (!cur.value || !to) return
  busy.value = 'state'
  err.value = ''
  msg.value = ''
  try {
    const r = await api.wbState(cur.value.id, { state: to })
    replace(r.card)
    msg.value = `${cur.value.id} → ${stateLabel(to)}`
  } catch (e: any) {
    err.value = `推进失败：${String(e?.message || e)}`
  } finally {
    busy.value = ''
  }
}

async function toggleEvidence(it: any) {
  if (!cur.value) return
  busy.value = 'ev'
  err.value = ''
  try {
    const r = await api.wbEvidence(cur.value.id, { key: it.k, done: !it.done })
    replace(r.card)
  } catch (e: any) {
    err.value = `勾选失败：${String(e?.message || e)}`
  } finally {
    busy.value = ''
  }
}

async function addRun() {
  if (!cur.value || !runId.value.trim()) return
  busy.value = 'run'
  err.value = ''
  msg.value = ''
  try {
    const r = await api.wbRun(cur.value.id, { run_id: runId.value.trim(), role: runRole.value })
    replace(r.card)
    msg.value = `已挂载 ${runId.value.trim()}`
    runId.value = ''
  } catch (e: any) {
    err.value = `挂载失败：${String(e?.message || e)}`
  } finally {
    busy.value = ''
  }
}

function replace(card: any) {
  const i = cards.value.findIndex(c => c.id === card.id)
  if (i >= 0) cards.value[i] = card
  else cards.value.push(card)
}
</script>

<template>
  <div class="wb col" style="gap:var(--sp-3)">
    <div v-if="err" class="panel">
      <div class="panel-h">工作台不可用 <span class="tag bad">503</span></div>
      <div class="panel-b col" style="gap:8px">
        <p class="small">{{ err }}</p>
        <p class="tiny dim">
          实验卡与审批记录都在本机主库 <span class="mono">ops/ops.db</span>。远程只做算力与预览，
          未部署台账属预期行为。先跑 <span class="mono">python tools/ops_db.py init</span> 再刷新。
        </p>
        <div><button class="primary" @click="load">重试</button></div>
      </div>
    </div>

    <template v-else-if="loading">
      <div class="panel"><div class="panel-b row" style="gap:8px">
        <span class="spinner" /><span class="small dim">读取实验卡…</span>
      </div></div>
    </template>

    <template v-else>
      <!-- 摘要 -->
      <div class="kpis">
        <section class="panel kpi"><div class="panel-b metric">
          <b>{{ cards.length }}</b><span>实验卡 · 五张来自 11-23 报告</span>
        </div></section>
        <section class="panel kpi"><div class="panel-b metric">
          <b>{{ pending.length }}</b><span>待你确认</span>
          <div class="tiny dim" style="margin-top:4px">关口未决或成果待验收</div>
        </div></section>
        <section class="panel kpi"><div class="panel-b metric">
          <b>{{ evidenceDone }}<em class="tiny dim" style="font-style:normal"> / {{ evidenceTotal }}</em></b>
          <span>七件套已交</span>
          <div class="scorebar" style="margin-top:6px">
            <i :style="{ width: (evidenceTotal ? evidenceDone / evidenceTotal * 100 : 0) + '%' }" />
          </div>
        </div></section>
        <section class="panel kpi"><div class="panel-b metric">
          <b>{{ cards.filter(c => c.state === 'adopted').length }}</b><span>已采纳</span>
          <div class="tiny dim" style="margin-top:4px">采纳 ≠ 实验成功，结论另记</div>
        </div></section>
      </div>

      <div class="layout">
        <!-- 左：卡列表 -->
        <section class="panel list">
          <div class="panel-h">
            实验卡
            <div class="grow" />
            <span class="tiny dim">待确认优先</span>
          </div>
          <div class="panel-b col" style="gap:2px">
            <button v-for="c in [...cards].sort((a, b) => Number(needsHuman(b)) - Number(needsHuman(a)))"
                    :key="c.id" class="cardrow" :class="{ on: c.id === sel }"
                    :aria-pressed="c.id === sel" @click="sel = c.id">
              <div class="row" style="gap:6px">
                <span class="mono tiny dim">{{ c.id }}</span>
                <div class="grow" />
                <span class="tag" :class="TONE[c.state]">{{ stateLabel(c.state) }}</span>
              </div>
              <div class="ellip small" :title="c.title">{{ c.title }}</div>
              <div class="row" style="gap:4px">
                <span v-for="g in GATE_ORDER" :key="g" class="dotgate"
                      :class="gateOf(c, g)?.decision || 'none'"
                      :title="g + ' ' + (meta.gates[g]?.name || '') + '：' + decLabel(gateOf(c, g)?.decision)">
                  {{ g }}
                </span>
                <div class="grow" />
                <span v-if="needsHuman(c)" class="tag sig">待确认</span>
              </div>
            </button>
          </div>
          <div class="panel-b tiny dim" style="border-top:1px solid var(--line)">
            来源：<span class="mono">multicamera_reid_research_report_2026-11-23.html</span> §10。
            只种计划、不种成绩 —— 截至 11-23 五张卡均无项目实测结果。
          </div>
        </section>

        <!-- 右：卡详情 -->
        <section v-if="cur" class="detail col" style="gap:var(--sp-3)">
          <div class="panel">
            <div class="panel-h">
              <span class="mono">{{ cur.id }}</span>
              <span class="sub ellip" :title="cur.title">{{ cur.title }}</span>
              <div class="grow" />
              <span class="tag" :class="TONE[cur.state]">{{ stateLabel(cur.state) }}</span>
              <select class="state-sel" :value="''" :disabled="busy === 'state' || !nextStates(cur).length"
                      aria-label="推进状态" @change="advance(($event.target as HTMLSelectElement).value)">
                <option value="">推进状态…</option>
                <option v-for="s in nextStates(cur)" :key="s" :value="s">{{ stateLabel(s) }}</option>
              </select>
            </div>
            <div class="panel-b col" style="gap:var(--sp-3)">
              <div class="qa">
                <div>
                  <div class="tiny dim">一个主要问题</div>
                  <p class="small">{{ cur.question }}</p>
                </div>
                <div>
                  <div class="tiny dim">可证伪假设</div>
                  <p class="small">{{ cur.hypothesis }}</p>
                </div>
              </div>

              <div class="pair">
                <div class="box">
                  <div class="tiny dim">基线（不改）</div>
                  <p class="small">{{ cur.baseline.text }}</p>
                </div>
                <div class="box accent">
                  <div class="tiny dim">候选改动（单变量）</div>
                  <p class="small">{{ cur.candidate.text }}</p>
                </div>
              </div>

              <div class="kv3">
                <div>
                  <div class="tiny dim">主指标</div>
                  <p class="small">{{ cur.metric.primary }}</p>
                </div>
                <div>
                  <div class="tiny dim">同时报</div>
                  <p class="small">{{ (cur.metric.secondary || []).join(' · ') || '—' }}</p>
                </div>
                <div>
                  <div class="tiny dim">失败条件</div>
                  <p class="small">{{ cur.metric.failure || '—' }}</p>
                </div>
              </div>

              <div class="row-top wrap" style="gap:var(--sp-3)">
                <div class="tiny dim">预算
                  <span class="mono" style="color:var(--fg-2)">{{ cur.budget.data || '—' }}</span>
                </div>
                <div v-if="cur.risk" class="tiny" style="color:var(--warn)">风险：{{ cur.risk }}</div>
              </div>
            </div>
          </div>

          <!-- 四关 -->
          <div class="panel">
            <div class="panel-h">
              四个确认关口 <span class="sub">人做决定 → 执行器自动推进</span>
            </div>
            <div class="panel-b col" style="gap:6px">
              <div v-for="g in gateList" :key="g.k" class="gate" :class="{ open: form?.gate === g.k }">
                <div class="row" style="gap:var(--sp-2)">
                  <span class="gkey mono">{{ g.k }}</span>
                  <b class="small">{{ g.name }}</b>
                  <span class="tag" :class="decTone(gateOf(cur, g.k)?.decision)">
                    {{ decLabel(gateOf(cur, g.k)?.decision) }}
                  </span>
                  <div class="grow" />
                  <span class="tiny dim ellip" style="max-width:38%" :title="g.auto">通过后：{{ g.auto }}</span>
                  <button class="ghost tiny" @click="openGate(g.k)">
                    {{ form?.gate === g.k ? '收起' : '记录决定' }}
                  </button>
                </div>
                <div class="tiny dim" style="margin-left:26px">可复核输出：{{ g.output }}</div>
                <div v-if="gateOf(cur, g.k)?.note" class="tiny" style="margin-left:26px;color:var(--fg-2)">
                  最近：{{ gateOf(cur, g.k).note }}
                  <span class="dim">· {{ gateOf(cur, g.k).at }}</span>
                </div>

                <div v-if="form?.gate === g.k" class="gateform col" style="gap:6px">
                  <div class="seg" role="tablist" aria-label="决定">
                    <button v-for="(lab, k) in meta.decisions" :key="k" role="tab"
                            :aria-selected="form.decision === k" @click="form.decision = k">{{ lab }}</button>
                  </div>
                  <textarea v-model="form.note" rows="2"
                            :placeholder="form.decision === 'approve' ? '同意也要写清依据与范围（会进审批记录）' : '退回/暂缓必须写理由'"
                            aria-label="审批意见" />
                  <div class="row">
                    <span class="tiny dim">
                      {{ g.k === 'A' ? 'A 同意 → 状态转「已批准」；退回 → 「需修订」'
                        : g.k === 'D' ? 'D 同意 → 「已采纳」；退回 → 「需修订」'
                        : '记录即生效，不改状态' }}
                    </span>
                    <div class="grow" />
                    <button class="primary" :disabled="busy === 'gate'" @click="submitGate">提交</button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div class="two">
            <!-- 人机分工 -->
            <div class="panel">
              <div class="panel-h">这张卡：谁做什么</div>
              <div class="panel-b col" style="gap:var(--sp-2)">
                <div>
                  <div class="tiny dim">你必须掌控</div>
                  <ul class="roles"><li v-for="(t, i) in cur.roles.human" :key="i" class="small">{{ t }}</li></ul>
                </div>
                <div>
                  <div class="tiny dim">Agent 可自主</div>
                  <ul class="roles"><li v-for="(t, i) in cur.roles.agent" :key="i" class="small">{{ t }}</li></ul>
                </div>
                <p class="tiny dim" style="margin:0">
                  候选生成者只能基于开发数据迭代；封存评测器拒绝回写真值、删难例或事后改评分。
                </p>
              </div>
            </div>

            <!-- 七件套 -->
            <div class="panel">
              <div class="panel-h">
                七件套交付物 <span class="sub">{{ cur.evidence.filter((i: any) => i.done).length }}/{{ cur.evidence.length }}</span>
              </div>
              <div class="panel-b col" style="gap:1px">
                <label v-for="it in cur.evidence" :key="it.k" class="ev row">
                  <input type="checkbox" :checked="it.done" :disabled="busy === 'ev'"
                         @change="toggleEvidence(it)" />
                  <span class="small grow" :class="it.done ? '' : 'dim'">{{ it.label }}</span>
                  <span v-if="it.note" class="tiny dim mono ellip" :title="it.note"
                        style="max-width:42%">{{ it.note }}</span>
                  <span v-else class="tiny dim">未交</span>
                </label>
              </div>
            </div>
          </div>

          <!-- 运行挂载 + 审批记录 -->
          <div class="panel">
            <div class="panel-h">
              运行与审批记录
              <div class="grow" />
              <span class="tiny dim">{{ cur.gate_log.length }} 条关口事件</span>
            </div>
            <div class="panel-b col" style="gap:var(--sp-2)">
              <div class="row wrap" style="gap:6px">
                <input v-model="runId" type="text" placeholder="挂载 run_id（如 run_20261102_133458_9831b1）"
                       aria-label="run id" style="max-width:340px" />
                <select v-model="runRole" aria-label="角色" style="max-width:110px">
                  <option value="baseline">baseline</option>
                  <option value="variant">variant</option>
                  <option value="ablation">ablation</option>
                </select>
                <button :disabled="busy === 'run' || !runId.trim()" @click="addRun">挂载</button>
                <div class="grow" />
                <RouterLink to="/runs" class="tiny" style="color:var(--accent)">去运行台账 →</RouterLink>
              </div>
              <div v-if="!cur.runs.length" class="tiny dim">
                还没有挂载运行。卡上的结论必须能指到具体 run，而不是口头「跑过了」。
              </div>
              <div v-for="r in cur.runs" :key="r.run_id" class="row" style="gap:6px">
                <span class="tag mono">{{ r.role }}</span>
                <RouterLink :to="'/runs?id=' + r.run_id" class="mono small ellip"
                            style="color:var(--accent)">{{ r.run_id }}</RouterLink>
                <div class="grow" />
                <span class="tiny dim">{{ r.at }}</span>
              </div>

              <div class="log">
                <div v-if="!cur.gate_log.length" class="tiny dim">尚无审批记录。</div>
                <div v-for="(e, i) in cur.gate_log" :key="i" class="logrow">
                  <span class="mono tiny dim">{{ e.at }}</span>
                  <span class="tag" :class="decTone(e.decision)">{{ e.gate }} · {{ decLabel(e.decision) }}</span>
                  <span class="tiny grow">{{ e.note || '—' }}</span>
                  <span class="tiny dim">{{ e.actor }}</span>
                </div>
              </div>
            </div>
          </div>

          <p v-if="msg" class="tiny" style="color:var(--ok)">{{ msg }}</p>
        </section>
      </div>
    </template>
  </div>
</template>

<style scoped>
.wb { padding: var(--sp-3) var(--sp-3) var(--sp-4); overflow: auto; min-height: 0; }
.kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: var(--sp-2); }
.kpi .panel-b { padding: var(--sp-3); }
.kpi .metric > b em { font-weight: 400; }
@media (max-width: 900px) { .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } }

.layout { display: grid; grid-template-columns: minmax(240px, 300px) minmax(0, 1fr); gap: var(--sp-3); align-items: start; }
@media (max-width: 1023px) { .layout { grid-template-columns: minmax(0, 1fr); } }
.list { position: sticky; top: 0; }
@media (max-width: 1023px) { .list { position: static; } }

.cardrow {
  display: flex; flex-direction: column; gap: 3px; width: 100%; text-align: left;
  padding: 7px 8px; border-radius: var(--r-sm); border: 1px solid transparent;
  background: transparent; color: inherit; min-height: 0;
}
.cardrow:hover { background: var(--surface-2); }
.cardrow.on { background: var(--accent-soft); border-color: var(--accent-line); }
.dotgate {
  display: inline-flex; align-items: center; justify-content: center;
  min-width: 15px; height: 15px; padding: 0 3px; border-radius: var(--r-xs);
  font-family: var(--font-mono); font-size: 9.5px; line-height: 1;
  border: 1px solid var(--line); color: var(--fg-3);
}
.dotgate.approve { color: var(--ok); border-color: var(--ok-line); background: var(--ok-soft); }
.dotgate.return { color: var(--danger); border-color: var(--danger-line); background: var(--danger-soft); }
.dotgate.hold { color: var(--warn); border-color: var(--warn-line); background: var(--warn-soft); }

.detail { min-width: 0; }
.state-sel { width: auto; min-width: 104px; max-width: 150px; }
.qa { display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr); gap: var(--sp-3); }
@media (max-width: 767px) { .qa { grid-template-columns: minmax(0, 1fr); } }
.qa p, .box p, .kv3 p { margin: 2px 0 0; }
.pair { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--sp-2); }
@media (max-width: 767px) { .pair { grid-template-columns: minmax(0, 1fr); } }
.box { border: 1px solid var(--line); border-radius: var(--r-sm); padding: var(--sp-2) var(--sp-3); }
.box.accent { border-color: var(--accent-line); background: var(--accent-soft); }
.kv3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: var(--sp-3); }
@media (max-width: 900px) { .kv3 { grid-template-columns: minmax(0, 1fr); gap: var(--sp-2); } }

.gate { border: 1px solid var(--line); border-radius: var(--r-sm); padding: 6px var(--sp-2); }
.gate.open { border-color: var(--accent-line); background: var(--surface-2); }
.gkey {
  display: inline-flex; align-items: center; justify-content: center;
  width: 16px; height: 16px; border-radius: var(--r-xs);
  background: var(--surface-3); color: var(--fg-2); font-size: 10px;
}
.gateform { margin: 6px 0 2px 26px; }
.two { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--sp-3); }
@media (max-width: 900px) { .two { grid-template-columns: minmax(0, 1fr); } }
.roles { margin: 2px 0 0; padding-left: 16px; }
.roles li { margin: 1px 0; }
.ev { padding: 3px 0; gap: 6px; cursor: pointer; border-bottom: 1px solid var(--line); }
.ev:last-child { border-bottom: 0; }
.ev input { width: auto; min-height: 0; accent-color: var(--accent); }
.log { border-top: 1px solid var(--line); margin-top: 4px; padding-top: 6px; display: flex; flex-direction: column; gap: 3px; }
.logrow { display: flex; align-items: baseline; gap: var(--sp-2); }
</style>
