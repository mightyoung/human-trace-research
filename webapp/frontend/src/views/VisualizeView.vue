<script setup lang="ts">
import { watch, onMounted, onUnmounted, ref, computed } from 'vue'
import { store, setPlaying, stepFrame, togglePlay, seek, delayOf, ownerArms, setArm, selectTrack } from '../store'
import CamPanel from '../components/CamPanel.vue'
import TimeController from '../components/TimeController.vue'
import ChainPanel from '../components/ChainPanel.vue'
import TrackInspector from '../components/TrackInspector.vue'
import MetricsPanel from '../components/MetricsPanel.vue'
import Minimap from '../components/Minimap.vue'
import JourneyPanel from '../components/JourneyPanel.vue'
import OwnerPanel from '../components/OwnerPanel.vue'

const views = [
  { k: 'all', label: '全部检测', hint: '含倒影与掩膜区内的原始检测' },
  { k: 'clean', label: '去镜保留', hint: '仅保留清洗后的人形轨迹' },
  { k: 'reflection', label: '倒影轨', hint: '被判定为玻璃/地板倒影的轨迹' },
  { k: 'masked', label: '掩膜区', hint: '脚点落在玻璃掩膜内的轨迹' }
]
const colorBys = [
  { k: 'chain', label: '接力链' }, { k: 'status', label: '清洗状态' },
  { k: 'entity', label: '同机实体' }, { k: 'track', label: '轨迹 ID' },
  { k: 'gperson', label: '全局人' }
]
const trails = [
  { v: 0, label: '关' }, { v: 10, label: '2s' }, { v: 30, label: '6s' }, { v: 60, label: '12s' },
  { v: 100000, label: '全程' }
]

const showLeft = ref(false)
const showRight = ref(false)
const narrow = ref(false)

function sync() {
  narrow.value = window.innerWidth < 1024
  if (!narrow.value) { showLeft.value = false; showRight.value = false }
}
onMounted(() => {
  applyRate()
  sync()
  window.addEventListener('resize', sync)
  window.addEventListener('keydown', onKey)
  if (window.innerWidth < 768) store.camFilter = store.masterCam
  // 深链接：`#/?t=1234` 直达某一秒；`#/?t=1234&sel=305:140` 同时选中某条轨迹（分享/复核用）
  const qs = new URLSearchParams(location.hash.split('?')[1] || '')
  const qsel = qs.get('sel') || ''
  if (qsel.includes(':')) {
    const [qcam, qtid] = qsel.split(':')
    if (qcam && qtid && Number.isFinite(+qtid)) selectTrack(qcam, +qtid)
  }
  const qt = Number(qs.get('t'))
  if (Number.isFinite(qt) && qt > 0) { seek(qt); return }
  // 首屏：会话窗口起点早于各机位首帧，直接停在「所有机位都进入」的时刻，避免六路全黑
  if (store.t < 1) {
    const allIn = Math.max(0, ...store.cams.map(c => delayOf(c.id)))
    if (allIn > 0) seek(allIn + 1)
  }
})
onUnmounted(() => {
  window.removeEventListener('resize', sync)
  window.removeEventListener('keydown', onKey)
})

function applyRate() {
  for (const k of Object.keys(store.videoEls)) {
    const el = store.videoEls[k]
    if (el) el.playbackRate = store.rate
  }
}
watch(() => store.rate, applyRate)

function onKey(e: KeyboardEvent) {
  const t = e.target as HTMLElement
  if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA')) return
  if (e.code === 'Space') { e.preventDefault(); togglePlay() }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); stepFrame(-1) }
  else if (e.key === 'ArrowRight') { e.preventDefault(); stepFrame(1) }
  else if (/^[1-6]$/.test(e.key)) {
    const c = store.cams[+e.key - 1]
    if (c) { store.masterCam = c.id; store.camFilter = c.id }
  } else if (e.key === 'Escape') { showLeft.value = false; showRight.value = false }
}

const shownCams = computed(() =>
  store.camFilter === 'all' ? store.cams : store.cams.filter(c => c.id === store.camFilter))
const single = computed(() => store.camFilter !== 'all')
const hasSel = computed(() => !!(store.sel || store.selChain))

/** 去重视图：打开时同时把着色切到「全局人」，否则框色仍是轨迹色，看不出效果 */
function toggleDedup() {
  store.dedup = !store.dedup
  store.colorBy = store.dedup ? 'gperson' : 'chain'
}

/** ★ 三档身份口径臂（S6-b 裁决回灌）：A 现状 / B 倒影降权 / H 人工裁决版（同 cap12 口径） */
const armReady = computed(() => ({ A: !!ownerArms.A, B: !!ownerArms.B, H: !!ownerArms.H }))
const armN = computed(() => ({
  A: ownerArms.A?.meta?.counts?.gperson, B: ownerArms.B?.meta?.counts?.gperson,
  H: ownerArms.H?.meta?.counts?.gperson
}))
const armTitle = computed(() =>
  `身份口径三档（同 cap12）：A 现状 ${armN.value.A ?? '—'} 人 ｜ B 倒影降权 ${armN.value.B ?? '—'} 人`
  + ` ｜ H 人工裁决版 ${armN.value.H ?? '—'} 人。打开去重视图才看得出效果。`)
function pickArm(w: 'A' | 'B' | 'H') {
  if (!armReady.value[w]) return
  setArm(w)
  if (!store.dedup) toggleDedup()
}

function activityOf(camId: string): boolean {
  const c = store.camMap[camId]
  if (!c) return false
  const local = store.t - (c.offset - store.session.min_offset)
  if (local < 0) return false
  const f = Math.round(local * c.fps)
  const fk = Math.round(f / store.stride) * store.stride
  return ((store.frameIdx[camId]?.[fk]) || []).length > 0
}
void setPlaying
</script>

<template>
  <div class="viz">
    <!-- ── 工具条 ─────────────────────────────────────────── -->
    <div class="toolbar scroll">
      <div class="grp">
        <span class="label">视图</span>
        <div class="seg" role="tablist" aria-label="数据视图">
          <button v-for="v in views" :key="v.k" role="tab" :title="v.hint"
                  :aria-selected="store.view === v.k" @click="store.view = v.k as any">{{ v.label }}</button>
        </div>
      </div>
      <div class="sep hide-sm" />
      <div class="grp">
        <span class="label hide-sm">着色</span>
        <div class="seg" role="tablist" aria-label="着色方式">
          <button v-for="c in colorBys" :key="c.k" role="tab"
                  :aria-selected="store.colorBy === c.k" @click="store.colorBy = c.k as any">{{ c.label }}</button>
        </div>
      </div>
      <div class="sep hide-sm" />
      <div class="grp">
        <span class="label">尾迹</span>
        <div class="seg" role="tablist" aria-label="尾迹长度">
          <button v-for="t in trails" :key="t.v" role="tab"
                  :aria-selected="store.trail === t.v" @click="store.trail = t.v">{{ t.label }}</button>
        </div>
      </div>
      <div class="sep hide-sm" />
      <div class="grp rail-btns">
        <button class="pill" :aria-pressed="showLeft" @click="showLeft = !showLeft">接力链</button>
        <button class="pill" :aria-pressed="showRight" @click="showRight = !showRight">检查器</button>
        <div class="sep" />
      </div>
      <div class="grp">
        <button class="pill" :aria-pressed="store.showFullLife"
                title="绘制每条轨迹的整段生命期折线" @click="store.showFullLife = !store.showFullLife">长程全轨迹</button>
        <button class="pill" :aria-pressed="store.showPrediction"
                title="按最近速度做几何外推，虚线为预测落脚点" @click="store.showPrediction = !store.showPrediction">轨迹预测</button>
        <label v-if="store.showPrediction" class="sr-only" for="horizon">预测时长</label>
        <select v-if="store.showPrediction" id="horizon" v-model.number="store.predHorizon" class="mini">
          <option :value="1">1s</option><option :value="2">2s</option><option :value="3">3s</option>
        </select>
        <button class="pill" :aria-pressed="store.showIds" @click="store.showIds = !store.showIds">ID</button>
        <button class="pill" :aria-pressed="store.showBoxes" @click="store.showBoxes = !store.showBoxes">检测框</button>
        <button class="pill" :aria-pressed="store.showMasks" @click="store.showMasks = !store.showMasks">玻璃掩膜</button>
        <button class="pill" :aria-pressed="store.showSuspects"
                :title="`疑似同人（换号）候选：${store.suspects.n_groups || 0} 组 / ${store.suspects.n_pairs || 0} 段。` +
                        `外观相似但拼接未连上的相邻段会画虚线框并标 #id ≈`"
                @click="store.showSuspects = !store.showSuspects">
          同人候选<span v-if="store.suspects.n_groups" class="tiny dim"> {{ store.suspects.n_groups }}</span>
        </button>
        <div class="sep" />
        <button class="pill dedup-pill" :aria-pressed="store.dedup"
                title="同一人跨机位只保留责任机位那一个框（Q3 责任划分）"
                @click="toggleDedup()">去重视图</button>
        <div class="seg" role="group" aria-label="身份口径">
          <button v-for="w in (['A', 'B', 'H'] as const)" :key="w" class="pill dedup-pill"
                  :aria-pressed="store.arm === w"
                  :disabled="!store.owner?.cams?.length || !armReady[w]"
                  :title="armTitle" @click="pickArm(w)">
            {{ w === 'A' ? '现状' : w === 'B' ? '倒影降权' : '人工裁决版' }}
          </button>
        </div>
      </div>

      <div class="grow" />

      <span class="hint hide-sm">
        点击框 → 跨机联动 ｜ <span class="kbd">空格</span> 播放 ｜
        <span class="kbd">←</span><span class="kbd">→</span> 逐帧 ｜ <span class="kbd">1</span>–<span class="kbd">6</span> 机位
      </span>
    </div>

    <!-- ── 机位切换（窄屏 / 单机位模式） ───────────────────── -->
    <div v-if="single || narrow" class="camstrip scroll">
      <button class="pill" :aria-pressed="!single" @click="store.camFilter = 'all'">全部</button>
      <button v-for="c in store.cams" :key="c.id" class="pill cam-chip"
              :aria-pressed="store.camFilter === c.id" :class="{ live: activityOf(c.id) }"
              @click="store.camFilter = store.camFilter === c.id ? 'all' : c.id">
        <i class="dot" />
        <b class="mono">{{ c.id }}</b>
        <span class="hide-xs">{{ c.name }}</span>
      </button>
    </div>

    <!-- ── 主体 ──────────────────────────────────────────── -->
    <div class="body" :class="{ narrow }">
      <aside v-if="!narrow" class="rail rail-l col">
        <ChainPanel class="grow" />
      </aside>

      <section class="center col">
        <div class="grid6" :class="{ one: single }">
          <CamPanel v-for="c in shownCams" :key="c.id" :cam="c" />
        </div>
        <TimeController />
      </section>

      <aside v-if="!narrow" class="rail rail-r col scroll">
        <template v-if="hasSel">
          <JourneyPanel />
          <TrackInspector />
          <Minimap />
        </template>
        <!-- 未选中任何目标时用一张指引替代三个空面板，减少无意义占位 -->
        <section v-else class="panel">
          <div class="panel-h">从这里开始</div>
          <div class="panel-b col" style="gap:var(--sp-3)">
            <div class="step">
              <i>1</i>
              <div class="col">
                <b>点击画面中的检测框</b>
                <span class="tiny dim">选中一个目标，它在其他机位的关联段会同时高亮，右侧出现跨镜头旅程</span>
              </div>
            </div>
            <div class="step">
              <i>2</i>
              <div class="col">
                <b>用底部时间轴定位</b>
                <span class="tiny dim">拖动横轴或点甘特条跳转；<span class="kbd">空格</span> 播放、<span class="kbd">←</span><span class="kbd">→</span> 逐帧</span>
              </div>
            </div>
            <div class="step">
              <i>3</i>
              <div class="col">
                <b>按需叠加图层</b>
                <span class="tiny dim">工具条里的「长程全轨迹」「轨迹预测」「玻璃掩膜」适合逐项打开对比</span>
              </div>
            </div>
            <div class="step">
              <i>4</i>
              <div class="col">
                <b>找不到人？用语义检索</b>
                <span class="tiny dim">顶部导航切到「ReID · 语义」，直接输入「白衣服灰帽子的人」</span>
              </div>
            </div>
          </div>
        </section>
        <MetricsPanel />
        <OwnerPanel />
      </aside>
    </div>

    <!-- ── 抽屉（窄屏） ───────────────────────────────────── -->
    <template v-if="narrow">
      <div v-if="showLeft" class="scrim" @click="showLeft = false" />
      <div v-if="showLeft" class="sheet left" role="dialog" aria-label="接力链">
        <div class="sheet-h">
          接力链
          <div class="grow" />
          <button class="icon-btn" @click="showLeft = false" aria-label="关闭">✕</button>
        </div>
        <div class="sheet-b col" style="padding:0"><ChainPanel class="grow" /></div>
      </div>

      <div v-if="showRight" class="scrim" @click="showRight = false" />
      <div v-if="showRight" class="sheet bottom" role="dialog" aria-label="检查器">
        <div class="sheet-h">
          检查器
          <div class="grow" />
          <button class="icon-btn" @click="showRight = false" aria-label="关闭">✕</button>
        </div>
        <div class="sheet-b col scroll" style="gap:var(--sp-3)">
          <template v-if="hasSel">
            <JourneyPanel />
            <TrackInspector />
            <Minimap />
          </template>
          <div v-else class="empty">
            <b>还没有选中目标</b>
            <span class="tiny">点击画面里的检测框，这里会显示它的跨镜头旅程、轨迹属性与外推预测</span>
          </div>
          <MetricsPanel />
          <OwnerPanel />
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.viz { display: flex; flex-direction: column; height: 100%; min-height: 0; }

.toolbar {
  flex: none; display: flex; align-items: center; gap: var(--sp-2);
  padding: var(--sp-2) var(--sp-3);
  background: var(--surface-1); border-bottom: 1px solid var(--line);
  overflow-x: auto; overflow-y: hidden;
}
.toolbar::-webkit-scrollbar { height: 0; }
.grp { display: flex; align-items: center; gap: 6px; flex: none; }
.hint { font-size: var(--text-2xs); color: var(--fg-3); display: flex; align-items: center; gap: 4px; }
select.mini { width: 62px; min-height: 24px; padding: 2px 6px; font-size: var(--text-xs); }
.rail-btns { display: none; }
@media (max-width: 1023px) {
  .rail-btns { display: flex; order: -1; }
  /* 窄屏：单行横向滚动 + 右缘渐隐提示可滑动 */
  .toolbar {
    padding-right: var(--sp-5);
    mask-image: linear-gradient(90deg, #000 0, #000 calc(100% - 34px), transparent 100%);
  }
  .toolbar .label { display: none; }
}

.camstrip {
  flex: none; display: flex; gap: 6px; padding: var(--sp-2) var(--sp-3);
  border-bottom: 1px solid var(--line); background: var(--surface-1);
  overflow-x: auto;
}
.camstrip::-webkit-scrollbar { height: 0; }
.cam-chip { flex: none; }
.cam-chip .dot { color: var(--fg-3); opacity: .4; }
.cam-chip.live .dot { color: var(--ok); opacity: 1; }
@media (max-width: 560px) { .hide-xs { display: none; } }
@media (max-width: 1023px) {
  .camstrip {
    padding-right: var(--sp-5);
    mask-image: linear-gradient(90deg, #000 0, #000 calc(100% - 28px), transparent 100%);
  }
}
.step { display: flex; gap: var(--sp-2); align-items: flex-start; }
.step > i {
  flex: none; width: 20px; height: 20px; border-radius: var(--r-pill);
  background: var(--accent-soft); color: var(--accent); border: 1px solid var(--accent-line);
  font-size: var(--text-2xs); font-weight: 700; font-style: normal;
  display: flex; align-items: center; justify-content: center; margin-top: 1px;
}
.step b { font-size: var(--text-sm); font-weight: var(--weight-label); }

.body {
  flex: 1; display: grid; gap: var(--sp-3); padding: var(--sp-3); min-height: 0;
  grid-template-columns: 232px minmax(0, 1fr) 316px;
}
.body.narrow { grid-template-columns: minmax(0, 1fr); padding: var(--sp-2); }
@media (max-width: 1439px) and (min-width: 1024px) {
  .body:not(.narrow) { grid-template-columns: 210px minmax(0, 1fr) 288px; }
}
.rail { min-height: 0; gap: var(--sp-3); }
.rail-r { overflow: auto; }
/* 右栏子面板取自然高度：flex 默认 shrink 会把面板压扁，内容溢出后还会被下方面板盖住 */
.rail-r > * { flex: none; }
/* 中列：瓦片墙吃满剩余高度 + 时间轴按内容高度贴在底部
   注意：类名 .center 与 theme.css 的全局居中工具类同名，必须显式复位对齐方式 */
.center {
  min-height: 0; display: grid;
  grid-template-columns: minmax(0, 1fr);
  grid-template-rows: minmax(0, 1fr) auto;
  align-items: stretch; justify-items: stretch; justify-content: stretch;
  gap: var(--sp-3); container-type: inline-size;
}
.grid6 {
  width: 100%; min-height: 0; display: grid; gap: var(--sp-2);
  grid-template-columns: repeat(2, minmax(0, 1fr));
  grid-template-rows: repeat(3, minmax(0, 1fr));
}
.grid6.one { grid-template-columns: minmax(0, 1fr); grid-template-rows: minmax(0, 1fr); }
/* 中列够宽时回到 3×2 —— 容器查询，与视口宽度解耦。
   阈值取 900 而非 1000：1600 视口下中列实际约 990px，差 10px 就会掉回 2×3，
   而 2×3 的格子是 488×147（3.3:1）→ 16:9 画面被压到 261×147、两侧大量黑边；
   3×2 的格子 325×183（≈16:9）几乎贴合画面。 */
@container (min-width: 900px) {
  .grid6 { grid-template-columns: repeat(3, minmax(0, 1fr)); grid-template-rows: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 767px) {
  .grid6 { grid-template-columns: minmax(0, 1fr); grid-template-rows: repeat(auto-fit, minmax(0, 1fr)); overflow: auto; }
  .grid6 .cam-body { aspect-ratio: 16 / 9; }
}
</style>
