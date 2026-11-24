<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { store, journeyOf, localEntity, seek, selectChain, trackSpan, delayOf } from '../store'
import { api } from '../api'

const sel = computed(() => store.sel)
const jr = computed(() => (sel.value ? journeyOf(sel.value.cam, sel.value.tid) : null))
const ent = computed(() => (sel.value ? localEntity(sel.value.cam, sel.value.tid) : null))

// 选中目标时，向后端取一次模型侧预测/意图（帧号夹到该轨迹自身时间窗内）
const pred = ref<any>(null)
const busy = ref(false)
watch([() => store.sel, () => store.t], async () => {
  const s = store.sel
  if (!s) { pred.value = null; return }
  busy.value = true
  try {
    const c = store.camMap[s.cam]
    let f = Math.round((store.t - delayOf(s.cam)) * c.fps)
    const sp = trackSpan(s.cam, s.tid)
    if (sp) f = Math.min(Math.max(f, sp.f0), sp.f1)
    pred.value = await api.predictAll(s.cam, s.tid, Math.max(0, f), store.predHorizon)
  } catch { pred.value = null } finally { busy.value = false }
})

const CLASS_ZH: Record<string, string> = {
  stop: '停下', forward: '直行', backward: '折返', turn_left: '左转', turn_right: '右转'
}

function pickNode(node: any) {
  selectChain(jr.value?.cid || null)
  if (node.tids.length) store.sel = { cam: node.cam, tid: node.tids[0] }
  seek(Math.max(0, node.t0))
  store.showJourney = true
}
</script>

<template>
  <div class="panel col">
    <div class="panel-h">
      跨镜头轨迹
      <span v-if="sel" class="tag info mono">{{ sel.cam }}#{{ sel.tid }}</span>
      <div class="grow" />
      <span v-if="jr" class="tag mono" :style="{ color: jr.color, borderColor: jr.color }">{{ jr.cid }}</span>
      <span v-else-if="sel" class="tag warn">无接力链</span>
    </div>

    <div v-if="!sel" class="empty" style="padding:var(--sp-4)">
      <svg class="art" viewBox="0 0 48 48" width="40" height="40" fill="none" stroke="currentColor"
           stroke-width="1.6" stroke-linecap="round">
        <circle cx="9" cy="30" r="4.4" /><circle cx="24" cy="18" r="4.4" /><circle cx="39" cy="30" r="4.4" />
        <path d="M12.7 27.2 20.3 21M27.7 21l7.6 6.2" stroke-dasharray="3 3" />
      </svg>
      <b>未选中目标</b>
      <span class="tiny">点击任一画面中的检测框，这里会出现它跨机位的完整旅程</span>
    </div>

    <div v-else class="panel-b col" style="gap:var(--sp-3)">
      <div v-if="ent" class="tiny dim">
        同机实体 <b class="mono">{{ ent.eid }}</b> · 片段 <span class="mono">{{ ent.tracks.join(', ') }}</span>
      </div>

      <!-- 旅程时间线 -->
      <div v-if="jr" class="tl">
        <button v-for="(n, i) in jr.nodes" :key="i" class="tl-node" @click="pickNode(n)">
          <span class="rail">
            <i class="tl-dot" :style="{ background: jr.color }" />
            <i v-if="i < jr.nodes.length - 1" class="tl-line" />
          </span>
          <span class="tl-body col">
            <span class="row">
              <b class="mono">{{ n.cam }}</b>
              <span class="tiny dim mono">{{ n.eid.replace(n.cam + '-', '') }}</span>
              <span class="grow" />
              <span class="tiny mono dim">{{ n.t0.toFixed(0) }}–{{ n.t1.toFixed(0) }}s</span>
            </span>
            <span class="row" style="gap:5px;margin-top:5px;flex-wrap:wrap">
              <img v-for="t in n.tids.slice(0, 4)" :key="t" class="thumb"
                   :src="api.thumbUrl(n.cam, t, null, 96)" :alt="`${n.cam}#${t}`"
                   :title="`${n.cam}#${t}`" loading="lazy" />
              <span v-if="!n.tids.length" class="tiny dim">无轨迹</span>
            </span>
          </span>
        </button>
      </div>
      <div v-else class="tiny dim">该目标未纳入跨机接力链（可能只在单一机位出现）。</div>

      <!-- 预测 -->
      <div class="pred">
        <div class="between">
          <span class="label">轨迹预测</span>
          <span v-if="busy" class="tiny dim">计算中…</span>
        </div>
        <div v-if="pred?.future" class="row wrap" style="gap:var(--sp-3)">
          <span class="tiny">速度 <b class="mono">{{ pred.future.speed_px_s }}</b> px/s</span>
          <span class="tiny">方向 <b class="mono">{{ pred.future.heading_deg }}°</b></span>
          <span class="tiny">外推 <b class="mono">{{ pred.future.horizon_s }}s</b></span>
        </div>
        <div v-else-if="!busy" class="tiny dim">几何外推不可用（该时刻历史点不足）</div>
        <div v-if="pred?.intent?.available" class="row" style="gap:6px;margin-top:2px">
          <span class="tag sig">意图 {{ CLASS_ZH[pred.intent.top] || pred.intent.top }}</span>
          <span class="mono tiny">{{ (pred.intent.confidence * 100).toFixed(0) }}%</span>
          <span class="tiny dim">未来 {{ pred.intent.horizon_s }}s</span>
        </div>
        <div v-else-if="!busy" class="tiny dim">意图模型：{{ pred?.intent?.reason || '不可用' }}</div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.tl { display: flex; flex-direction: column; }
.tl-node {
  display: flex; gap: var(--sp-2); padding: 5px 6px; border-radius: var(--r-sm);
  background: transparent; border: 1px solid transparent; text-align: left;
  width: 100%; min-height: 0; align-items: stretch;
}
@media (hover: hover) { .tl-node:hover { background: var(--surface-2); } }
.rail { position: relative; width: 9px; flex: none; display: flex; justify-content: center; }
.tl-dot { width: 9px; height: 9px; border-radius: 50%; margin-top: 4px; box-shadow: 0 0 0 2px var(--surface-1); }
.tl-line { position: absolute; top: 15px; bottom: -8px; width: 1px; background: var(--line); }
.tl-body { flex: 1; min-width: 0; }
.thumb {
  height: 46px; width: auto; border-radius: var(--r-xs);
  border: 1px solid var(--line); object-fit: cover; background: #000;
}
.pred {
  border-top: 1px solid var(--line); padding-top: var(--sp-2);
  display: flex; flex-direction: column; gap: 5px;
}
</style>
