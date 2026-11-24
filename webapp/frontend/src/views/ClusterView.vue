<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { api } from '../api'

const report = ref<any>(null)
const state = ref<any>(null)
const textEval = ref<any>(null)
const busy = ref(false)
const msg = ref('')
const sel = ref<Set<number>>(new Set())
const openCid = ref<string | null>(null)
const moveTarget = ref<string>('')
const labelDraft = ref('')
const tab = ref<'train' | 'desk'>('train')
const trainTab = ref<'cluster' | 'intent' | 'text'>('cluster')
const showClusterList = ref(false)

async function loadState() {
  busy.value = true
  try { state.value = await api.clusterState() } catch (e: any) { msg.value = String(e?.message || e) }
  finally { busy.value = false }
}
onMounted(async () => {
  try { report.value = await api.reidReport() } catch { /* ignore */ }
  try { textEval.value = await api.reidTextEval() } catch { /* ignore */ }
  await loadState()
  if (state.value?.clusters?.length) openCid.value = state.value.clusters[0].cid
  window.addEventListener('keydown', onKey)
})

const clusters = computed(() => state.value?.clusters || [])
const openCluster = computed(() => clusters.value.find((c: any) => c.cid === openCid.value) || null)
const multiClusters = computed(() => clusters.value.filter((c: any) => c.size > 1))

function pick(cid: string) {
  openCid.value = cid
  labelDraft.value = clusters.value.find((c: any) => c.cid === cid)?.label || ''
  sel.value = new Set()
  zoomIdx.value = null
}
function toggle(i: number, cid: string) {
  if (sel.value.has(i)) sel.value.delete(i); else sel.value.add(i)
  sel.value = new Set(sel.value)
  openCid.value = cid
}
function toggleAll(c: any, v: boolean) {
  for (const m of c.members) { if (v) sel.value.add(m.i); else sel.value.delete(m.i) }
  sel.value = new Set(sel.value)
}
function clearSel() { sel.value = new Set() }

async function act(fn: () => Promise<any>, note: string) {
  busy.value = true; msg.value = ''
  try { await fn(); msg.value = note; sel.value = new Set(); await loadState() }
  catch (e: any) { msg.value = '失败：' + String(e?.message || e) }
  finally { busy.value = false }
}
function doRename(c: any) { act(() => api.clusterRename(c.cid, labelDraft.value), `已重命名为「${labelDraft.value || '未命名'}」`) }
function doDelete(c: any) { act(() => api.clusterDelete(c.cid), `已删除簇 ${c.cid}`); if (openCid.value === c.cid) openCid.value = null }
function doMove(cid: string | null, createNew = false) {
  act(() => api.clusterMove([...sel.value], cid, createNew, labelDraft.value || '新簇'),
      createNew ? '已新建簇并移入选中项' : `已移入 ${cid}`)
}
function doRemove() { if (openCluster.value) act(() => api.clusterRemove(openCluster.value.cid, [...sel.value]), '已移出到未分配') }
function doReset() { act(() => api.clusterReset(), '已重置为自动聚类结果') }

// ---- 图片放大（lightbox）：hover ⤢ / 双击卡片打开，←/→ 翻簇内成员，Esc 关闭 ----
const zoomIdx = ref<number | null>(null)
const zoomMembers = computed<any[]>(() => openCluster.value?.members || [])
const zoomMember = computed<any>(() => (zoomIdx.value == null ? null : zoomMembers.value[zoomIdx.value] || null))
const zoomTotal = computed(() => zoomMembers.value.length)
function openZoom(m: any) {
  const i = zoomMembers.value.indexOf(m)
  if (i >= 0) zoomIdx.value = i
}
function zoomStep(d: number) {
  if (zoomIdx.value == null || !zoomTotal.value) return
  zoomIdx.value = (zoomIdx.value + d + zoomTotal.value) % zoomTotal.value
}
function onKey(e: KeyboardEvent) {
  if (zoomIdx.value == null) return
  if (e.key === 'Escape') zoomIdx.value = null
  else if (e.key === 'ArrowLeft') zoomStep(-1)
  else if (e.key === 'ArrowRight') zoomStep(1)
  else return
  e.preventDefault()
}
onBeforeUnmount(() => window.removeEventListener('keydown', onKey))

const reportCluster = computed(() => report.value?.cluster_report || null)
/**
 * 方法对比表折叠：同一方法/簇数/各项指标完全相同的多行（只差 thr）合并成一行，
 * 参数显示为 thr 区间。否则表格里会出现 4 行数字一模一样的记录，看着像 bug
 * ——「阈值在这个区间不影响结果」本身是有信息量的结论，要显式说出来。
 */
const clusterRows = computed(() => {
  const rs = (reportCluster.value?.results || []) as any[]
  const out: any[] = []
  for (const r of rs) {
    const key = [r.method, r.k ?? r.K ?? '', r.n_clusters, r.f1, r.prec, r.rec, r.score].join('|')
    const hit = out.find(x => x.key === key)
    if (hit) {
      if (r.thr != null) hit.thrs.push(r.thr)
      if (r === reportCluster.value.best) hit.isBest = true
      continue
    }
    out.push({ key, ...r, isBest: r === reportCluster.value.best,
               thrs: r.thr != null ? [r.thr] : [] })
  }
  for (const r of out) r.param = r.k != null
    ? `k=${r.k} thr=${r.thrs.length > 1
        ? `${Math.min(...r.thrs)}–${Math.max(...r.thrs)}`
        : r.thrs[0] ?? '—'}`
    : `K=${r.K}`
  return out
})
const intentM = computed(() => report.value?.intent_metrics || null)
const intentClasses = ['stop', 'forward', 'backward', 'turn_left', 'turn_right']
const ZH: Record<string, string> = { stop: '停下', forward: '直行', backward: '折返', turn_left: '左转', turn_right: '右转' }

const curve = computed(() => {
  const h = intentM.value?.hist || []
  if (!h.length) return null
  const w = 320, hh = 92
  const n = h.length
  const y = (v: number) => hh - v * (hh - 10) - 5
  const pts = (key: string) => h.map((e: any, i: number) => `${(i / Math.max(1, n - 1)) * w},${y(e[key])}`).join(' ')
  return { w, h: hh, acc: pts('val_acc'), f1: pts('val_macro_f1'), n }
})
</script>

<template>
  <div class="cv">
    <div class="topbar">
      <div class="seg" role="tablist" aria-label="页面分区">
        <button role="tab" :aria-selected="tab === 'train'" @click="tab = 'train'">训练情况</button>
        <button role="tab" :aria-selected="tab === 'desk'" @click="tab = 'desk'">聚类工作台</button>
      </div>
      <div class="grow" />
      <span v-if="state" class="tag mono hide-sm">
        {{ state.method }} · k={{ state.k }} · thr={{ state.thr }} · {{ state.n }} 轨 · {{ state.n_total }} 簇
      </span>
      <span v-if="busy" class="tiny dim row"><span class="spinner" />处理中</span>
      <button v-if="tab === 'desk'" class="pill only-sm" :aria-pressed="showClusterList"
              @click="showClusterList = !showClusterList">簇列表</button>
    </div>

    <div v-if="msg" class="msgbar tiny">{{ msg }}</div>

    <!-- ═══════════ 训练情况 ═══════════ -->
    <div v-if="tab === 'train'" class="body scroll">
      <div class="seg sub" role="tablist" aria-label="训练内容">
        <button role="tab" :aria-selected="trainTab === 'cluster'" @click="trainTab = 'cluster'">无监督聚类</button>
        <button role="tab" :aria-selected="trainTab === 'intent'" @click="trainTab = 'intent'">意图模型</button>
        <button role="tab" :aria-selected="trainTab === 'text'" @click="trainTab = 'text'">语义检索</button>
      </div>

      <!-- 聚类 -->
      <template v-if="trainTab === 'cluster'">
        <section class="panel">
          <div class="panel-h">
            方法对比
            <span class="sub">1395 训练轨 · 无标签选择指标 = 分片一致率 − 随机对碰撞率</span>
          </div>
          <div class="panel-b table-scroll">
            <div class="table method-table">
              <div class="th">方法</div><div class="th">参数</div><div class="th">簇数</div>
              <div class="th">F1</div><div class="th">精确</div><div class="th">召回</div><div class="th">综合分</div>
              <template v-for="(r, i) in clusterRows" :key="i">
                <div :class="{ best: r.isBest }">{{ r.method }}</div>
                <div class="mono">{{ r.param }}</div>
                <div class="num">{{ r.n_clusters }}</div>
                <div class="num">{{ r.f1.toFixed(3) }}</div>
                <div class="num">{{ r.prec.toFixed(3) }}</div>
                <div class="num">{{ r.rec.toFixed(3) }}</div>
                <div class="num">{{ r.score.toFixed(3) }}</div>
              </template>
            </div>
            <p v-if="reportCluster" class="tiny dim" style="margin-top:var(--sp-3)">
              分片判别基线 AUC <b class="mono">{{ reportCluster.split_half.auc.toFixed(4) }}</b> ·
              正对均相似 <b class="mono">{{ reportCluster.split_half.pos_mean.toFixed(3) }}</b> /
              负对 <b class="mono">{{ reportCluster.split_half.neg_mean.toFixed(3) }}</b> ·
              未知阈值 <b class="mono">{{ reportCluster.thr_calib.thr_unknown.toFixed(3) }}</b>
            </p>
          </div>
        </section>

        <section class="panel">
          <div class="panel-h">相似度分布</div>
          <div class="panel-b">
            <img class="fig" :src="'/figs/cluster_sim_dist.png'" alt="正对与随机对的余弦相似度分布" />
            <p class="tiny dim" style="margin-top:var(--sp-2)">
              工作台使用 <b class="mono">mutual-kNN k=3 / thr=0.95</b>：在 539 条全局轨迹上得到
              {{ state?.n_total ?? '—' }} 簇（最大 16 项）。阈值来自身份阈值 ~0.95 的标定。
            </p>
          </div>
        </section>
      </template>

      <!-- 意图 -->
      <template v-else-if="trainTab === 'intent'">
        <section class="panel">
          <div class="panel-h">轨迹运动意图模型 <span class="sub">BERT 范式 Transformer 编码器</span></div>
          <div class="panel-b col" style="gap:var(--sp-4)">
            <div v-if="intentM" class="strip">
              <div class="metric"><b>{{ (intentM.test.acc * 100).toFixed(1) }}%</b><span>测试准确率</span></div>
              <div class="metric"><b>{{ intentM.test.macro_f1.toFixed(3) }}</b><span>macro-F1</span></div>
              <div class="metric"><b>1s</b><span>预测未来时长</span></div>
              <div class="metric"><b>{{ (intentM.hist || []).length }}</b><span>训练轮次</span></div>
            </div>
            <div v-if="curve" class="chart">
              <svg :viewBox="`0 0 ${curve.w} ${curve.h}`" width="100%" :height="curve.h" role="img" aria-label="训练曲线">
                <polyline :points="curve.acc" fill="none" stroke="var(--accent)" stroke-width="2" />
                <polyline :points="curve.f1" fill="none" stroke="var(--ok)" stroke-width="2" />
              </svg>
              <div class="row tiny" style="gap:var(--sp-4)">
                <span class="row" style="gap:5px"><i class="lg" style="background:var(--accent)" />验证准确率</span>
                <span class="row" style="gap:5px"><i class="lg" style="background:var(--ok)" />验证 macro-F1</span>
                <span class="dim">共 {{ curve.n }} 轮</span>
              </div>
            </div>
            <div v-if="intentM" class="table cls-table">
              <div class="th">类别</div><div class="th">精确</div><div class="th">召回</div><div class="th">F1</div><div class="th">样本</div>
              <template v-for="c in intentClasses" :key="c">
                <div>{{ ZH[c] }} <span class="dim mono">{{ c }}</span></div>
                <div class="num">{{ intentM.test[c].precision.toFixed(3) }}</div>
                <div class="num">{{ intentM.test[c].recall.toFixed(3) }}</div>
                <div class="num">{{ intentM.test[c].f1.toFixed(3) }}</div>
                <div class="num">{{ intentM.test[c].support }}</div>
              </template>
            </div>
          </div>
        </section>
        <section class="panel">
          <div class="panel-h">混淆矩阵</div>
          <div class="panel-b"><img class="fig" :src="'/figs/intent_confusion.png'" alt="意图模型测试集混淆矩阵" /></div>
        </section>
      </template>

      <!-- 语义 -->
      <template v-else>
        <section class="panel">
          <div class="panel-h">语义查人 · 离线自评</div>
          <div class="panel-b col" style="gap:var(--sp-3)">
            <p class="small">
              文本塔 EmbeddingGemma2 在本地离线编码 <b>109 条属性短语</b>（13 色 × 6 服饰 + 独立概念），
              与 C-RADIO×Gemma2 桥接的 <b>768 维</b>图像向量位于同一空间；运行时把中文 query
              解析成概念后取均值检索。
            </p>
            <div v-if="textEval" class="strip">
              <div class="metric"><b>{{ textEval.hit5 }}/{{ textEval.n_query }}</b><span>top-5 命中</span></div>
              <div class="metric"><b>539</b><span>全局轨迹</span></div>
              <div class="metric"><b>109</b><span>属性短语</span></div>
            </div>
            <div v-if="textEval" class="table txt-table">
              <div class="th">查询</div><div class="th">最佳正例排名</div><div class="th">top-5</div>
              <template v-for="(c, i) in textEval.cases" :key="i">
                <div class="ellip">{{ c.query }}</div>
                <div class="num">{{ c.best_rank ?? '—' }}</div>
                <div><span class="tag" :class="c.hit5 ? 'ok' : 'bad'">{{ c.hit5 ? '命中' : '未命中' }}</span></div>
              </template>
            </div>
            <p class="tiny dim">
              「黑卫衣」「毛皮领」为已知失败硬例，如实标注 —— 属性词表组合不是任意句子的 open-vocab。
            </p>
          </div>
        </section>
      </template>
    </div>

    <!-- ═══════════ 聚类工作台 ═══════════ -->
    <div v-else class="body desk">
      <aside class="panel col clist" :class="{ open: showClusterList }">
        <div class="panel-h">
          簇列表 <span class="tag info mono">{{ clusters.length }}</span>
          <div class="grow" />
          <button class="icon-btn only-sm" @click="showClusterList = false" aria-label="关闭">✕</button>
        </div>
        <div class="scroll grow">
          <button v-for="c in clusters" :key="c.cid" class="citem" :aria-selected="openCid === c.cid"
                  @click="pick(c.cid); showClusterList = false">
            <span class="tag mono">{{ c.cid }}</span>
            <b class="ellip">{{ c.label || '未命名' }}</b>
            <span v-if="c.origin === 'manual'" class="tag ok" title="人工新建/整理过">人工</span>
            <span class="grow" />
            <span class="tag mono">{{ c.size }}</span>
          </button>
          <div v-if="state?.unassigned?.length" class="citem" style="cursor:default">
            <span class="tag warn">未分配</span>
            <span class="grow" /><span class="tag mono">{{ state.unassigned.length }}</span>
          </div>
        </div>
      </aside>

      <section class="col grow" style="min-height:0;gap:var(--sp-3)">
        <div v-if="openCluster" class="panel">
          <div class="panel-h">
            簇 <span class="mono">{{ openCluster.cid }}</span>
            <span class="tag mono">{{ openCluster.size }} 项</span>
            <div class="grow" />
            <span v-if="sel.size" class="tag sig mono">已选 {{ sel.size }}</span>
          </div>
          <div class="panel-b col" style="gap:var(--sp-2)">
            <div class="row wrap">
              <label class="sr-only" for="clabel">簇名称</label>
              <input id="clabel" v-model="labelDraft" type="text" style="max-width:240px"
                     placeholder="给这个身份起个名字，如：白衣男 / 保洁员"
                     @keyup.enter="doRename(openCluster)" />
              <button class="primary" @click="doRename(openCluster)">保存名称</button>
              <div class="sep" />
              <button class="pill" @click="toggleAll(openCluster, true)">全选</button>
              <button class="pill" :disabled="!sel.size" @click="clearSel">清空选择</button>
            </div>

            <!-- 渐进式披露：仅在有选中项时展开批量操作 -->
            <div v-if="sel.size" class="batch">
              <span class="label">对选中 {{ sel.size }} 项</span>
              <label class="sr-only" for="mv">目标簇</label>
              <select id="mv" v-model="moveTarget" class="mini-sel">
                <option value="">选择目标簇…</option>
                <option v-for="c in clusters.filter(x => x.cid !== openCluster.cid)" :key="c.cid" :value="c.cid">
                  {{ c.cid }} {{ c.label || '未命名' }}（{{ c.size }}）
                </option>
              </select>
              <button :disabled="!moveTarget" @click="doMove(moveTarget)">移入该簇</button>
              <button @click="doMove(null, true)">新建簇并移入</button>
              <button class="warn" @click="doRemove">移出到未分配</button>
              <div class="grow" />
              <button class="danger ghost" @click="doDelete(openCluster)">删除整簇 {{ openCluster.cid }}</button>
            </div>
            <div v-else class="row tiny dim">
              点选下方任一缩略图即可加入选区，批量操作会自动出现。
              <span class="grow" />
              <button class="pill" @click="doReset" title="丢弃所有人工修改，回到自动聚类结果">重置为自动聚类</button>
            </div>
          </div>
        </div>

        <div class="panel scroll grow" style="min-height:0">
          <div class="panel-h">
            簇成员
            <span class="sub">缩略图取自各机位视频对应帧 · 点击加入选区 · 双击或 ⤢ 放大</span>
            <div class="grow" />
            <span class="tiny dim mono hide-sm">{{ openCluster?.members?.length || 0 }} 项</span>
          </div>
          <div class="grid">
            <button v-for="m in (openCluster?.members || [])" :key="m.i" class="mcard"
                    :aria-pressed="sel.has(m.i)" @click="toggle(m.i, openCluster!.cid)"
                    @dblclick="openZoom(m)" :title="`${m.cam}#${m.tid} · 双击放大`">
              <span class="mthumb">
                <img :src="api.thumbUrl(m.cam, m.tid, m.f_rep, 160)" :alt="`${m.cam}#${m.tid}`" loading="lazy" />
                <span class="zbtn" role="button" aria-label="放大查看"
                      @click.stop.prevent="openZoom(m)" @dblclick.stop.prevent>⤢</span>
              </span>
              <span class="tiny mono">{{ m.cam }}#{{ m.tid }}</span>
              <span class="tiny dim mono">{{ m.t0.toFixed(0) }}–{{ m.t1.toFixed(0) }}s</span>
            </button>
            <div v-if="!openCluster" class="empty" style="grid-column:1/-1">
              <b>未选择簇</b><span class="tiny">从左侧簇列表挑一个开始编辑</span>
            </div>
          </div>
        </div>
      </section>

      <!-- 放大预览 -->
      <div v-if="zoomMember" class="zbox" @click.self="zoomIdx = null">
        <div class="zcard panel">
          <div class="panel-h">
            <span class="tag info mono">{{ zoomMember.cam }}#{{ zoomMember.tid }}</span>
            <span class="tiny dim mono">{{ zoomMember.t0.toFixed(0) }}–{{ zoomMember.t1.toFixed(0) }}s · 代表帧 {{ zoomMember.f_rep }}</span>
            <div class="grow" />
            <span class="tiny dim mono">{{ (zoomIdx ?? 0) + 1 }} / {{ zoomTotal }}</span>
            <button class="icon-btn" aria-label="上一项" :disabled="zoomTotal < 2" @click="zoomStep(-1)">‹</button>
            <button class="icon-btn" aria-label="下一项" :disabled="zoomTotal < 2" @click="zoomStep(1)">›</button>
            <button class="icon-btn" aria-label="关闭（Esc）" @click="zoomIdx = null">✕</button>
          </div>
          <div class="zimg">
            <img :src="api.thumbUrl(zoomMember.cam, zoomMember.tid, zoomMember.f_rep, 384)"
                 :alt="`${zoomMember.cam}#${zoomMember.tid}`" />
          </div>
          <div class="panel-b row wrap" style="gap:var(--sp-2)">
            <button class="pill" :aria-pressed="sel.has(zoomMember.i)"
                    @click="toggle(zoomMember.i, openCluster!.cid)">
              {{ sel.has(zoomMember.i) ? '✓ 已在选区 · 点击移除' : '加入选区' }}
            </button>
            <div class="grow" />
            <span class="tiny dim"><span class="kbd">←</span><span class="kbd">→</span> 切换 ｜ <span class="kbd">Esc</span> 关闭</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.cv { display: flex; flex-direction: column; height: 100%; min-height: 0; }
.topbar {
  flex: none; display: flex; align-items: center; gap: var(--sp-2);
  padding: var(--sp-2) var(--sp-3); border-bottom: 1px solid var(--line); background: var(--surface-1);
}
.msgbar { flex: none; padding: 5px var(--sp-3); color: var(--ok); border-bottom: 1px solid var(--line); }
.body { flex: 1; min-height: 0; padding: var(--sp-3); display: flex; flex-direction: column; gap: var(--sp-3); }
.body.scroll { overflow: auto; }
.body.desk { display: grid; grid-template-columns: 276px minmax(0, 1fr); }

.seg.sub { align-self: flex-start; }

.strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); }
.strip .metric { padding: 0 var(--sp-3); border-left: 1px solid var(--line); }
.strip .metric:first-child { padding-left: 0; border-left: 0; }

.table { display: grid; gap: 2px var(--sp-3); font-size: var(--text-sm); }
.table > * { padding: 3px 0; border-bottom: 1px solid var(--line); min-width: 0; }
.table .th {
  color: var(--fg-3); font-size: var(--text-2xs); font-weight: var(--weight-label);
  text-transform: uppercase; letter-spacing: 0.06em; border-bottom-color: var(--line-2);
}
.table .best { color: var(--accent); font-weight: var(--weight-label); }
.table .num { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
.method-table { grid-template-columns: 1.1fr 1.3fr repeat(5, minmax(0, 0.7fr)); }
.cls-table { grid-template-columns: 1fr repeat(4, minmax(0, 0.6fr)); max-width: 620px; }
.txt-table { grid-template-columns: 1.6fr 1fr 0.8fr; max-width: 680px; }
.table-scroll { overflow-x: auto; }

.chart { border: 1px solid var(--line); border-radius: var(--r-sm); padding: var(--sp-2) var(--sp-3); }
.lg { width: 12px; height: 3px; border-radius: 2px; display: inline-block; font-style: normal; }
.fig { width: 100%; max-width: 620px; border: 1px solid var(--line); border-radius: var(--r-sm); background: #fff; }

.clist { min-height: 0; }
.citem {
  flex: none;   /* 受限 flex 列中不可压缩，否则相邻项内容重叠 */
  display: flex; align-items: center; gap: 6px; width: 100%;
  padding: var(--sp-2) var(--sp-3); text-align: left;
  background: transparent; border: 0; border-bottom: 1px solid var(--line); border-radius: 0;
  font-size: var(--text-sm);
}
@media (hover: hover) { .citem:hover { background: var(--surface-2); } }
.citem[aria-selected='true'] { background: var(--surface-3); box-shadow: inset 2px 0 0 var(--accent); }

.mini-sel { width: 200px; min-height: 26px; padding: 2px 6px; font-size: var(--text-xs); }
.batch {
  display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
  padding: var(--sp-2); border-radius: var(--r-sm);
  background: var(--surface-2); border: 1px solid var(--line-2);
}

.grid {
  display: grid; gap: var(--sp-2); padding: var(--sp-3);
  grid-template-columns: repeat(auto-fill, minmax(104px, 1fr));
}
.mcard {
  display: flex; flex-direction: column; gap: 2px; align-items: center;
  padding: 5px; min-height: 0; background: var(--surface-2);
  border: 1px solid var(--line); border-radius: var(--r-sm);
}
@media (hover: hover) { .mcard:hover { border-color: var(--line-2); } }
.mcard[aria-pressed='true'] {
  border-color: var(--accent); background: var(--accent-soft);
  box-shadow: 0 0 0 2px var(--accent-line) inset;
}
.mcard img { width: 100%; height: 86px; object-fit: cover; border-radius: var(--r-xs); background: #000; }

/* 图片放大 */
.mthumb { position: relative; display: block; width: 100%; line-height: 0; }
.zbtn {
  position: absolute; top: 3px; right: 3px; width: 21px; height: 21px; line-height: 1;
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; color: #fff; background: rgba(0, 0, 0, .55);
  border: 1px solid rgba(255, 255, 255, .28); border-radius: var(--r-xs);
  opacity: 0; transition: opacity var(--dur-instant) var(--ease-out-quart); cursor: zoom-in;
}
@media (hover: hover) { .mcard:hover .zbtn, .mcard:focus-visible .zbtn { opacity: 1; } }
@media (hover: none) { .zbtn { opacity: .92; } }
.zbox {
  position: fixed; inset: 0; z-index: 80; display: flex; align-items: center; justify-content: center;
  background: rgba(0, 0, 0, .68); padding: var(--sp-3);
}
.zcard { width: min(620px, 94vw); max-height: 92vh; overflow: auto; }
.zimg { display: flex; justify-content: center; background: var(--stage, #05070a); }
.zimg img { max-width: 100%; max-height: 58vh; display: block; }

@media (max-width: 1023px) {
  .body.desk { grid-template-columns: minmax(0, 1fr); }
  .clist {
    position: fixed; left: var(--sp-3); right: var(--sp-3); bottom: var(--sp-3);
    top: 25vh; z-index: calc(var(--z-drawer) + 1); display: none; box-shadow: var(--shadow-3);
  }
  .clist.open { display: flex; }
  .method-table { grid-template-columns: 96px 120px repeat(5, 72px); min-width: 600px; }
  .cls-table { grid-template-columns: 1.2fr repeat(4, 64px); min-width: 380px; }
  .txt-table { grid-template-columns: 1.4fr 88px 72px; min-width: 320px; }
}
</style>
