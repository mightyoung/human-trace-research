<script setup lang="ts">
import { computed } from 'vue'
import { store, selectTrack, selectChain, seek, delayOf, suspectOf, suspectMembers } from '../store'

const sel = computed(() => store.sel)
const info = computed(() => {
  const s = store.sel
  if (!s) return null
  const c = store.camMap[s.cam]
  const eid = store.trackEnt[`${s.cam}:${s.tid}`]
  const cid = store.chainOf[`${s.cam}:${s.tid}`]
  const pts = store.tracks[s.cam]?.tracks?.[String(s.tid)] || []
  const f0 = pts.length ? pts[0][0] : 0
  const f1 = pts.length ? pts[pts.length - 1][0] : 0
  const d = delayOf(s.cam)
  return {
    cam: s.cam, tid: s.tid, name: c?.name, eid, cid,
    status: store.trackStatus[s.cam]?.[String(s.tid)] || 'keep',
    t0: d + f0 / (c?.fps || 25), t1: d + f1 / (c?.fps || 25),
    n: pts.length, span: (f1 - f0) / (c?.fps || 25),
    avgScore: pts.length ? pts.reduce((a, p) => a + (p[5] || 0), 0) / pts.length : 0
  }
})
const chainNodes = computed(() => {
  if (!store.selChain) return []
  const ch = (store.chains.chains || []).find((x: any) => x.id === store.selChain)
  return ch ? ch.nodes : []
})
const ents = computed(() => {
  const s = store.sel
  if (!s) return []
  const eid = store.trackEnt[`${s.cam}:${s.tid}`]
  if (!eid) return []
  return store.entities[s.cam]?.entities?.[eid]?.tracks || []
})
const statusMeta: Record<string, { label: string; cls: string }> = {
  keep: { label: '保留', cls: 'ok' },
  reflection: { label: '倒影', cls: 'bad' },
  masked: { label: '掩膜区', cls: 'warn' }
}

// ---- 疑似同人（换号）：同一人被拆成多段的证据链 ----
const sgGroup = computed(() => {
  const s = store.sel
  return s ? suspectOf(s.cam, s.tid) : null
})
const sgMembers = computed(() => {
  const g = sgGroup.value, s = store.sel
  if (!g || !s) return [] as any[]
  const c = store.camMap[s.cam]
  const fps = c?.fps || 25
  const d = delayOf(s.cam)
  return suspectMembers(g).map(tid => {
    const pts = store.tracks[s.cam]?.tracks?.[String(tid)] || []
    const has = pts.length > 0
    return {
      tid,
      eid: store.trackEnt[`${s.cam}:${tid}`] || '—',
      t0: has ? d + pts[0][0] / fps : 0,
      t1: has ? d + pts[pts.length - 1][0] / fps : 0,
      cur: tid === s.tid
    }
  })
})
/** 当前段参与的那条连接（只取一条，用于展示证据数值） */
const sgLink = computed(() => {
  const g = sgGroup.value, s = store.sel
  if (!g || !s) return null
  const hits = (g.links || []).filter((l: any) => l.a_tid === s.tid || l.b_tid === s.tid)
  return hits.sort((a: any, b: any) => b.cos - a.cos)[0] || null
})
function gotoSeg(t: number, tid: number) {
  const s = store.sel
  if (!s) return
  seek(Math.max(0, t + 0.2))
  selectTrack(s.cam, tid)
}
</script>

<template>
  <div class="ins panel col">
    <div class="panel-h">
      轨迹 / 身份检查器
      <div class="grow" />
      <span v-if="info" class="sub mono">{{ info.cam }}#{{ info.tid }}</span>
    </div>

    <div v-if="!info && !chainNodes.length" class="empty">
      <svg class="art" viewBox="0 0 48 48" width="40" height="40" fill="none" stroke="currentColor"
           stroke-width="1.6" stroke-linecap="round">
        <path d="M8 34c4-12 10-18 16-18s10 6 16 18" stroke-dasharray="3 3" />
        <circle cx="24" cy="14" r="5" />
      </svg>
      <b>未选中目标</b>
      <span class="tiny">在任一画面点击检测框 → 查看轨迹、同机实体与跨机接力链</span>
    </div>

    <div v-else class="panel-b col" style="gap:var(--sp-3)">
      <template v-if="info">
      <div class="row wrap" style="gap:6px">
        <span class="tag info mono">{{ info.cam }} #{{ info.tid }}</span>
        <span class="tag" :class="statusMeta[info.status]?.cls">
          <i class="dot" />{{ statusMeta[info.status]?.label || info.status }}
        </span>
        <div class="grow" />
        <span class="tiny dim">{{ info.name }}</span>
      </div>

      <div class="kv">
        <div>同机实体</div><div>{{ info.eid || '—' }}</div>
        <div>接力链</div><div>{{ info.cid || '—' }}</div>
        <div>存活区间</div><div>{{ info.t0.toFixed(1) }} → {{ info.t1.toFixed(1) }}s</div>
        <div>持续</div><div>{{ info.span.toFixed(1) }}s · {{ info.n }} 点</div>
        <div>平均检分</div><div>{{ info.avgScore.toFixed(3) }}</div>
      </div>

      <div v-if="ents.length > 1">
        <div class="label" style="margin-bottom:5px">同实体碎片 · {{ ents.length }} 段</div>
        <div class="row wrap" style="gap:4px">
          <button v-for="t in ents" :key="t" class="frag mono" :aria-pressed="t === info.tid"
                  @click="selectTrack(info.cam, t)">#{{ t }}</button>
        </div>
      </div>
      </template>

      <div v-if="chainNodes.length">
        <div class="between" style="margin-bottom:5px">
          <span class="label">接力链 {{ store.selChain }}</span>
          <button class="ghost tiny" @click="selectChain(null)">清除高亮</button>
        </div>
        <div class="col" style="gap:3px">
          <button v-for="(n, i) in chainNodes" :key="i" class="node" :aria-pressed="n.cam === (info?.cam)"
                  @click="seek(Math.max(0, n.t0s))">
            <i class="idx" :style="{ background: store.chainColor[store.selChain] }">{{ i + 1 }}</i>
            <b class="mono">{{ n.cam }}</b>
            <span class="mono tiny dim">{{ n.eid.replace(n.cam + '-', '') }}</span>
            <div class="grow" />
            <span class="tiny dim mono">{{ n.t0s?.toFixed(1) }}s</span>
          </button>
        </div>
      </div>

      <!-- ★ 疑似同人（换号）：外观很像但拼接没连上的相邻段，纯标注不改口径 -->
      <div v-if="sgGroup" class="sus">
        <div class="between" style="margin-bottom:5px">
          <span class="label">疑似同一人 · {{ sgMembers.length }} 段</span>
          <span class="tag" :class="sgGroup.grade === 'strong' ? 'warn' : ''">
            <i class="dot" />{{ sgGroup.grade === 'strong' ? '高度相似' : '可能同人' }}
          </span>
        </div>
        <div class="tiny dim" style="margin-bottom:6px; line-height:1.55">
          轮廓虚线（<span class="mono">#id ≈</span> 高度相似 / <span class="mono">#id ?</span> 可能同人）
          表示「这段可能是同一个人换了 id」：两人靠近时检测框被合并 → 跟踪断开 → 重新起号。
          点任一段可跳转核对。
        </div>
        <div class="col" style="gap:3px">
          <button v-for="(m, i) in sgMembers" :key="m.tid" class="node" :aria-pressed="m.cur"
                  @click="gotoSeg(m.t0, m.tid)">
            <i class="idx" :style="{ background: m.cur ? 'var(--accent)' : 'var(--surface-3)',
                                     color: m.cur ? 'var(--accent-ink)' : 'var(--fg-2)' }">{{ i + 1 }}</i>
            <b class="mono">#{{ m.tid }}</b>
            <span class="mono tiny dim">{{ m.t0.toFixed(1) }}–{{ m.t1.toFixed(1) }}s</span>
            <div class="grow" />
            <span class="tiny dim">{{ m.cur ? '当前段' : '跳转 →' }}</span>
          </button>
        </div>
        <div v-if="sgLink" class="tiny dim mono" style="margin-top:5px">
          证据 cos {{ sgLink.cos }} · 间隙 {{ sgLink.gap_s }}s · 脚点差 {{ sgLink.dist_h }}×框高
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.ins { min-height: 0; }
.frag {
  font-size: var(--text-2xs); padding: 2px 8px; min-height: 22px;
  border-radius: var(--r-xs); background: var(--surface-2); border: 1px solid var(--line);
}
.frag[aria-pressed='true'] { background: var(--accent-soft); color: var(--accent); border-color: var(--accent-line); font-weight: var(--weight-label); }
.node {
  display: flex; align-items: center; gap: var(--sp-2); padding: 5px var(--sp-2);
  border-radius: var(--r-sm); background: var(--surface-2); border: 1px solid transparent;
  text-align: left; width: 100%; min-height: 0; font-size: var(--text-xs);
}
@media (hover: hover) { .node:hover { background: var(--surface-3); } }
.node[aria-pressed='true'] { border-color: var(--accent-line); }
.idx {
  width: 17px; height: 17px; border-radius: 50%; color: var(--accent-ink);
  font-size: var(--text-2xs); font-weight: 700; font-style: normal;
  display: flex; align-items: center; justify-content: center; flex: none;
}
/* 疑似同人块：左侧竖条提示「这是推断，不是判定」 */
.sus {
  padding: var(--sp-2) 0 var(--sp-2) var(--sp-2);
  border-left: 2px dashed var(--warn-line, var(--line));
  margin-left: 1px;
}
</style>
