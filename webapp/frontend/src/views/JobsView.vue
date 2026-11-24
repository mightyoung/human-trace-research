<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { api } from '../api'

const jobs = ref<any[]>([])
const err = ref('')
const msg = ref('')
const loading = ref(true)
const busy = ref('')
const evDraft = ref<Record<string, string>>({})
const creating = ref(false)
const newTitle = ref('')

const COLS = [
  { key: 'backlog', title: '待办', tone: '' },
  { key: 'active', title: '进行中', tone: 'info' },
  { key: 'review', title: '待验收', tone: 'warn' },
  { key: 'done', title: '已完成', tone: 'ok' }
]

onMounted(load)

async function load() {
  loading.value = true
  err.value = ''
  try {
    const r = await api.opsJobs()
    jobs.value = r.jobs || []
  } catch (e: any) {
    err.value = String(e?.message || e)
  } finally {
    loading.value = false
  }
}

const grouped = computed(() => {
  const g: Record<string, any[]> = { backlog: [], active: [], review: [], done: [] }
  for (const j of jobs.value) {
    if (j.status === 'backlog' || j.status === 'blocked') g.backlog.push(j)
    else if (j.status === 'assigned' || j.status === 'doing') g.active.push(j)
    else if (j.status === 'review') g.review.push(j)
    else if (j.status === 'done') g.done.push(j)
  }
  return g
})

async function move(j: any, status: string) {
  busy.value = j.id
  err.value = ''
  msg.value = ''
  try {
    const body: any = { status }
    if (status === 'done') {
      const ev = (evDraft.value[j.id] || '').trim()
      // label 类且队列已清零 → 证据由服务端按事实生成，不必手填
      const auto = j.kind === 'label' && j.live && j.live.pending === 0
      if (!ev && !auto) { err.value = '关闭任务必须填写证据（文件路径 / run_id / 指标）'; return }
      if (ev) body.evidence = { files: [ev], by: 'console' }
    }
    await api.opsJobStatus(j.id, body)
    msg.value = `${j.title} → ${status}`
    await load()
  } catch (e: any) {
    err.value = String(e?.message || e)
  } finally {
    busy.value = ''
  }
}

async function sync() {
  busy.value = 'sync'
  err.value = ''
  msg.value = ''
  try {
    const r = await api.opsJobsSync()
    msg.value = r.n
      ? `已把 ${r.n} 张清零的卡推到待验收：${r.moved.map((m: any) => m.title).join('、')}`
      : '没有需要同步的卡（进行中的卡都还有待判）'
    await load()
  } catch (e: any) {
    err.value = String(e?.message || e)
  } finally {
    busy.value = ''
  }
}

async function derive() {
  busy.value = 'derive'
  err.value = ''
  try {
    const r = await api.opsJobsDerive()
    msg.value = r.n ? `已派生 ${r.n} 张任务卡` : '没有需要派生的任务（待判为 0 或已存在）'
    await load()
  } catch (e: any) {
    err.value = String(e?.message || e)
  } finally {
    busy.value = ''
  }
}

async function createJob() {
  const t = newTitle.value.trim()
  if (!t) return
  busy.value = 'create'
  err.value = ''
  try {
    await api.opsJobCreate({ title: t, kind: 'review', status: 'backlog' })
    msg.value = `已创建：${t}`
    newTitle.value = ''
    creating.value = false
    await load()
  } catch (e: any) {
    err.value = String(e?.message || e)
  } finally {
    busy.value = ''
  }
}

const KIND: Record<string, string> = {
  label: '标注', run: '运行', review: '复核', write: '写作', fix: '修复'
}
const STATUS_CN: Record<string, string> = {
  backlog: '待办', assigned: '已认领', doing: '进行中',
  review: '待验收', done: '完成', blocked: '阻塞'
}
function nextOf(j: any) {
  const s = j.status
  if (s === 'backlog' || s === 'blocked') return { to: 'assigned', text: '认领' }
  if (s === 'assigned') return { to: 'doing', text: '开始' }
  if (s === 'doing') return { to: 'review', text: '提交验收' }
  if (s === 'review') return { to: 'done', text: '通过' }
  return null
}
</script>

<template>
  <div class="jobs col" style="gap:var(--sp-3);min-height:0">
    <div class="row" style="gap:var(--sp-2);flex-wrap:wrap">
      <div class="col" style="gap:1px">
        <b class="small">任务派发</b>
        <span class="tiny dim">关闭任务必须挂证据；「派生」按标注待判量自动建卡</span>
      </div>
      <div class="grow" />
      <input v-if="creating" v-model="newTitle" class="new-in mono" autofocus
             placeholder="任务标题，回车创建" @keyup.enter="createJob" @keyup.esc="creating = false" />
      <button v-if="creating" class="tiny primary" @click="createJob">创建</button>
      <button v-else class="tiny" @click="creating = true">新建任务</button>
      <button class="tiny" :disabled="busy === 'sync'" @click="sync">
        <span v-if="busy === 'sync'" class="spinner" />同步状态
      </button>
      <button class="tiny" :disabled="busy === 'derive'" @click="derive">
        <span v-if="busy === 'derive'" class="spinner" />派生标注任务
      </button>
      <button class="tiny" @click="load">刷新</button>
    </div>

    <div v-if="err" class="tag bad" style="align-self:flex-start">{{ err }}</div>
    <div v-else-if="msg" class="tag ok" style="align-self:flex-start">{{ msg }}</div>

    <div v-if="loading" class="panel"><div class="panel-b row" style="gap:8px">
      <span class="spinner" /><span class="small dim">读取任务卡…</span></div></div>

    <div v-else class="board">
      <section v-for="c in COLS" :key="c.key" class="col colwrap">
        <div class="colhead">
          <span class="tag" :class="c.tone">{{ c.title }}</span>
          <span class="tiny dim">{{ grouped[c.key].length }}</span>
        </div>
        <div class="colbody">
          <div v-if="!grouped[c.key].length" class="empty tiny">空</div>
          <article v-for="j in grouped[c.key]" :key="j.id" class="card2">
            <div class="row" style="gap:5px;align-items:flex-start">
              <span class="tag">{{ KIND[j.kind] || j.kind }}</span>
              <span v-if="j.priority" class="tag" :class="j.priority >= 50 ? 'warn' : ''">P{{ j.priority }}</span>
              <div class="grow" />
              <span v-if="j.status === 'blocked'" class="tag bad">阻塞</span>
            </div>
            <div class="title2">{{ j.title }}</div>
            <div v-if="j.goal" class="tiny dim">{{ j.goal }}</div>
            <div v-if="j.live" class="prog">
              <div class="row" style="gap:6px">
                <span class="tiny mono">{{ j.live.done }} / {{ j.live.items }}</span>
                <div class="grow" />
                <span class="tiny" :class="j.drained ? 'ok' : 'dim'">
                  {{ j.drained ? '队列已清零' : `剩 ${j.live.pending} 条` }}
                </span>
              </div>
              <div class="scorebar"><i :style="{ width: j.live.pct + '%' }" /></div>
            </div>
            <ul v-if="j.acceptance?.length" class="acc">
              <li v-for="(a, i) in j.acceptance.slice(0, 3)" :key="i">{{ a }}</li>
            </ul>
            <div v-if="j.authorization" class="tiny auth">授权：{{ j.authorization }}</div>
            <div v-if="j.evidence" class="tiny ev">
              证据：<span class="mono">{{ (j.evidence.files || []).join(', ') || j.evidence.run_id || '已挂' }}</span>
            </div>

            <div v-if="j.status === 'review'" class="row" style="gap:4px;margin-top:4px">
              <input v-model="evDraft[j.id]" class="ev-in mono"
                     :placeholder="j.kind === 'label' && j.drained
                       ? '可留空：关闭时自动记录队列清零' : '证据路径或 run_id'" />
            </div>

            <div class="row" style="gap:4px;margin-top:6px">
              <button v-if="nextOf(j)" class="tiny primary"
                      :disabled="busy === j.id" @click="move(j, nextOf(j)!.to)">
                {{ nextOf(j)!.text }}
              </button>
              <button v-if="j.status === 'review'" class="tiny"
                      :disabled="busy === j.id" @click="move(j, 'doing')">退回</button>
              <div class="grow" />
              <span class="tiny dim mono">{{ j.id.slice(-6) }}</span>
            </div>
          </article>
        </div>
      </section>
    </div>

    <p class="tiny dim">
      状态机：待办 → 认领 → 进行中 → 待验收 → 完成；任一步可转「阻塞」。
      关闭时服务端校验证据，缺失则拒绝（返回 400）。
    </p>
  </div>
</template>

<style scoped>
.jobs { padding: var(--sp-3); overflow: auto; }
.board { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: var(--sp-2); align-items: start; }
@media (max-width: 1200px) { .board { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 720px) { .board { grid-template-columns: minmax(0, 1fr); } }
.colwrap { gap: 6px; min-height: 120px; }
.colhead { display: flex; align-items: center; gap: 6px; padding: 0 2px 4px; }
.colbody { display: flex; flex-direction: column; gap: 6px; }
.card2 { background: var(--surface-1); border: 1px solid var(--line); border-radius: var(--r-md);
         padding: 8px 9px; display: flex; flex-direction: column; gap: 3px; min-width: 0; }
.card2:hover { border-color: var(--line-2); }
.title2 { font-size: var(--text-sm); font-weight: 600; line-height: 1.35; word-break: break-word; }
.acc { margin: 2px 0 0; padding-left: 15px; font-size: var(--text-xs); color: var(--fg-2); }
.acc li { margin: 1px 0; }
.auth { color: var(--fg-3); border-left: 2px solid var(--line-2); padding-left: 6px; margin-top: 3px; }
.prog { display: flex; flex-direction: column; gap: 3px; margin: 2px 0 1px; }
.prog .ok { color: var(--ok); }
.ev { color: var(--ok); }
.ev-in { flex: 1; min-width: 0; font-size: var(--text-xs); padding: 3px 6px;
         background: var(--surface-2); border: 1px solid var(--line); border-radius: var(--r-sm);
         color: var(--fg); }
.ev-in:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
.new-in { width: 240px; font-size: var(--text-xs); padding: 4px 8px;
          background: var(--surface-1); border: 1px solid var(--line);
          border-radius: var(--r-sm); color: var(--fg); }
.new-in:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
</style>
