<script setup lang="ts">
import { computed } from 'vue'
import { store, personsAt, tracksAt, ownerArms, setArm } from '../store'

/** S4 去重视图面板：把「同一人跨机位只算一个目标」这件事变得可看、可验。 */
const meta = computed(() => store.owner?.meta || null)
const counts = computed(() => meta.value?.counts || {})
const cov = computed(() => meta.value?.coverage || {})
const c0 = computed(() => meta.value?.c0_v7 || {})
const conc = computed(() => meta.value?.concurrency || {})
const byCam = computed(() => meta.value?.coverage_by_cam || {})
const cams = computed<string[]>(() => store.owner?.cams || [])

/** ★ S6-b 三档身份口径对照（A 现状 / B 倒影降权 / H 人工裁决版；同 cap12 口径） */
const arms = computed(() => {
  const A = ownerArms.A?.meta, B = ownerArms.B?.meta, H = ownerArms.H?.meta
  if (!A) return null
  const n = (m: any) => m?.counts?.gperson
  return {
    a: n(A), b: n(B), h: n(H),
    aOwn: A.counts?.owner_intervals, bOwn: B?.counts?.owner_intervals, hOwn: H?.counts?.owner_intervals,
    aVisit: A.counts?.visit, bVisit: B?.counts?.visit, hVisit: H?.counts?.visit,
    bReady: !!B, hReady: !!H, cur: store.arm
  }
})

/** 当前时刻全局去重结果：每个 gperson 一个（owner 机位唯一） */
const persons = computed(() => (store.owner ? personsAt(store.t) : []))
const multiCam = computed(() => persons.value.filter(p => p.n_cams > 1))

/** 未去重 / 去重后 的**画框数**对照（同一时刻、同一抽样帧，量纲一致） */
const boxBefore = computed(() => cams.value.reduce((a, c) => a + tracksAt(c, store.t, { raw: true }).length, 0))
const boxAfter = computed(() => cams.value.reduce((a, c) => a + tracksAt(c, store.t).length, 0))
/** 各机位当前负责的目标数 */
const perCam = computed(() => {
  const m: Record<string, number> = {}
  for (const c of cams.value) m[c] = 0
  for (const p of persons.value) m[p.cam] = (m[p.cam] || 0) + 1
  return m
})

const reasonLabel: Record<string, string> = {
  sole: '单机位', visibility: '重叠择一', handoff: '交接'
}
const pct = (v: number | null | undefined) => (v == null ? '—' : (v * 100).toFixed(1) + '%')
const covBars = computed(() => cams.value.map(c => {
  const x = byCam.value[c] || {}
  return { cam: c, portal: x.exit_portal || 0, covered: x.exit_covered || 0, rate: x.exit_cont }
}))
const covMax = computed(() => Math.max(0.62, ...covBars.value.map(b => b.rate || 0)))
</script>

<template>
  <div class="op panel col">
    <div class="panel-h">
      去重视图 · 责任划分
      <div class="grow" />
      <button class="pill mini-btn" :aria-pressed="store.dedup" @click="store.dedup = !store.dedup">
        {{ store.dedup ? '已去重' : '去重' }}
      </button>
    </div>

    <div v-if="!meta" class="panel-b">
      <p class="tiny dim">未找到 <span class="mono">gil_owner.json</span>。
        运行 <span class="mono">python tools/gil_visit_owner.py</span> 生成后重新构建前端。</p>
    </div>

    <div v-else class="panel-b col" style="gap:var(--sp-3)">
      <!-- ★ S6-b 三档身份口径（同 cap12）：A 现状 / B 倒影降权 / H 人工裁决版 -->
      <div class="arm">
        <div class="row" style="align-items:center;gap:6px">
          <b class="tiny">身份口径</b>
          <span class="grow" />
          <div class="seg" role="group" aria-label="身份口径">
            <button v-for="w in (['A', 'B', 'H'] as const)" :key="w" class="pill mini-btn"
                    :aria-pressed="arms?.cur === w"
                    :disabled="!arms || (w === 'B' && !arms.bReady) || (w === 'H' && !arms.hReady)"
                    @click="setArm(w)">{{ w === 'A' ? '现状' : w === 'B' ? '降权' : '裁决版' }}</button>
          </div>
        </div>
        <div class="tiny dim" v-if="arms">
          人 <b class="mono">{{ arms.a }}</b> / <b class="mono">{{ arms.b }}</b> /
          <b class="mono hl">{{ arms.h ?? '—' }}</b> ·
          owner <span class="mono">{{ arms.aOwn }}/{{ arms.bOwn }}/{{ arms.hOwn }}</span> ·
          通行 <span class="mono">{{ arms.aVisit }}/{{ arms.bVisit }}/{{ arms.hVisit }}</span>
          <div style="margin-top:2px">顺序 = 现状 / 倒影降权 / <b>人工裁决版</b>（S6-b 30 项裁决回灌）</div>
          <div v-if="!arms.hReady" style="margin-top:2px">
            裁决版载荷缺失：运行 <span class="mono">gil_visit_owner.py --gperson-map …/gperson_human_v1.json --web-out public/gil_owner_human.json</span>
          </div>
        </div>
      </div>

      <!-- 一把抓的数字 -->
      <div class="stats">
        <div class="metric"><b>{{ counts.gperson ?? '—' }}</b><span>全局人</span></div>
        <div class="metric"><b>{{ pct(counts.dedup_vs_entity) }}</b><span>去重率</span></div>
        <div class="metric"><b>{{ counts.visit ?? '—' }}</b><span>通行段</span></div>
        <div class="metric"><b>{{ meta.owner_violations }}</b><span>owner 冲突</span></div>
      </div>

      <div class="kv">
        <div>轨迹 / 机位内实体</div><div class="hl mono">{{ counts.obs }} → {{ counts.entity }}</div>
        <div>owner 区间</div><div class="mono">{{ counts.owner_intervals }}</div>
        <div>同一时刻被 2 机位看到</div><div class="mono">{{ pct(conc['2']) }}</div>
        <div>owner 不变量（任一时刻唯一）</div>
        <div class="mono" :class="meta.owner_violations ? 'bad' : 'good'">
          {{ meta.owner_violations === 0 ? '0 违反 ✓' : meta.owner_violations + ' 违反' }}
        </div>
      </div>

      <!-- 门户覆盖：身份级 vs 边级 -->
      <div>
        <div class="label" style="margin-bottom:5px">
          出门覆盖 · 身份级 <span class="mono">{{ pct(cov.exit_cont) }}</span>
          <span class="tiny dim">（+30s 时间门 {{ pct(cov.exit_cont_strict) }}）</span>
        </div>
        <div class="bars">
          <div v-for="b in covBars" :key="b.cam" class="brow">
            <span class="mono bcam">{{ b.cam }}</span>
            <span class="track"><i :style="{ width: ((b.rate || 0) / covMax * 100) + '%' }" /></span>
            <span class="mono bnum">{{ pct(b.rate) }}</span>
            <span class="tiny dim bfrac">{{ b.covered }}/{{ b.portal }}</span>
          </div>
        </div>
        <p class="tiny dim" style="margin-top:5px">
          C0 v7 实体级（边级上界）{{ pct(c0.exit_continuity_v7) }}
          <span class="mono">({{ c0.portal_exit_covered }}/{{ c0.portal_exit_n }})</span>
          —— 差值即「图上有边、但身份层不承认同人」
        </p>
      </div>

      <!-- 当前时刻 -->
      <div>
        <div class="label" style="margin-bottom:5px">
          当前时刻 t = <span class="mono">{{ store.t.toFixed(1) }}s</span>
          <span class="tiny dim" :class="{ dedup_drop: boxAfter < boxBefore }">
            （未去重 {{ boxBefore }} 框 → 去重后 {{ boxAfter }} 框，少画 {{ Math.max(0, boxBefore - boxAfter) }} 个重复框）
          </span>
        </div>
        <div v-if="!persons.length" class="tiny dim">此刻没有已归属的全局人。</div>
        <div v-else class="table t">
          <div class="th">人</div><div class="th">负责机位</div><div class="th">同时可见</div><div class="th">依据</div>
          <template v-for="p in persons" :key="p.gid">
            <div class="mono">g{{ String(p.gid).padStart(4, '0') }}</div>
            <div class="mono hl">{{ p.cam }}</div>
            <div class="num">{{ p.n_cams }}</div>
            <div class="tiny" :class="{ 'ho': p.reason === 'handoff' }">{{ reasonLabel[p.reason] || p.reason }}</div>
          </template>
        </div>
        <p v-if="multiCam.length" class="tiny dim" style="margin-top:5px">
          ★ 有 {{ multiCam.length }} 个目标此刻被多个机位同时看到，已按可见性效用择一（标签带 ◆ 标记）
        </p>
        <p class="tiny dim" style="margin-top:3px">
          责任目标 {{ persons.length }} 个（每人在此仅一条 owner 记录；与画框数的差异来自抽样帧对齐）
        </p>
      </div>

      <!-- 各机位责任 -->
      <div>
        <div class="label" style="margin-bottom:5px">各机位当前负责</div>
        <div class="table c">
          <template v-for="c in cams" :key="c">
            <div class="mono">{{ c }}</div><div class="num">{{ perCam[c] || 0 }}</div>
          </template>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.op { min-height: 0; overflow: auto; }
.op .panel-b > * { flex: none; }          /* 全站铁律：受限 flex 列中列表项不得被压缩 */
.stats { display: grid; grid-template-columns: repeat(4, 1fr); }
.metric { padding: 0 var(--sp-2); border-left: 1px solid var(--line); }
.metric:first-child { padding-left: 0; border-left: 0; }
.metric > b { font-size: var(--text-lg); font-weight: 650; line-height: 1.1; }
.hl { color: var(--accent); }
.good { color: var(--ok); }
.bad { color: var(--danger); }
.ho { color: var(--warn); }
.dedup_drop { color: var(--accent); }
.mini-btn { padding: 2px var(--sp-2); font-size: var(--text-xs); min-height: 24px; }
/* ★ S6-b 倒影降权开关 */
.arm {
  flex: none; padding: var(--sp-2) var(--sp-3); border: 1px solid var(--line);
  border-radius: var(--r-sm); background: var(--surface-inset);
  display: flex; flex-direction: column; gap: 3px;
}
.arm .row { align-items: center; }

.bars { display: flex; flex-direction: column; gap: 3px; }
.brow { display: grid; grid-template-columns: 34px minmax(0, 1fr) 46px 44px; align-items: center; gap: 6px; }
.bcam { color: var(--fg-2); font-size: var(--text-xs); }
.brow .track { height: 6px; background: var(--surface-2); border-radius: var(--r-pill); overflow: hidden; }
.brow .track > i { display: block; height: 100%; background: var(--accent); border-radius: var(--r-pill); }
.bnum, .bfrac { font-size: var(--text-xs); }
.bfrac { text-align: right; }

.table.t { grid-template-columns: 58px 66px minmax(0, 1fr) 60px; }
.table.c { grid-template-columns: repeat(6, minmax(0, 1fr)); text-align: center; }
</style>
