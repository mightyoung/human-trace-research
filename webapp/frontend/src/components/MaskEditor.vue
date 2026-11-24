<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed, watch } from 'vue'
import { api } from '../api'
import { store } from '../store'

const props = defineProps<{ cam: string }>()
const emit = defineEmits<{ (e: 'save', cam: string, pts: number[][]): void }>()

const frame = ref(0)
const pts = ref<number[][]>([])
const imgWrap = ref<HTMLElement>()
const canvas = ref<HTMLCanvasElement>()
const imgEl = ref<HTMLImageElement>()
const imgUrl = computed(() => api.frameUrl(props.cam, frame.value, 'all', 0, 0) + `&_=${frame.value}`)
let ro: ResizeObserver | null = null


/** 从主题变量取色，保证浅色/深色主题下画布都可读 */
function cv(name: string, fallback: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback
}

function redraw() {
  const maskFill = 'rgba(255,92,114,0.14)'
  const maskStroke = cv('--danger', '#ff5c72')
  const savedFill = 'rgba(56,211,159,0.12)'
  const savedStroke = cv('--ok', '#38d39f')
  const accent = cv('--accent', '#4ea1ff')
  const cvs = canvas.value, wrap = imgWrap.value
  if (!cvs || !wrap) return
  const w = wrap.clientWidth, h = wrap.clientHeight
  const dpr = Math.min(2, window.devicePixelRatio || 1)
  cvs.width = Math.round(w * dpr); cvs.height = Math.round(h * dpr)
  const ctx = cvs.getContext('2d')!
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, w, h)
  // 项目既有掩膜（只读参考）
  const cur = store.masks?.cameras?.[props.cam]
  if (cur?.regions?.length) {
    for (const poly of cur.regions) {
      ctx.beginPath()
      poly.forEach((p: number[], i: number) => { const x = p[0] * w, y = p[1] * h; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y) })
      ctx.closePath()
      ctx.fillStyle = maskFill; ctx.strokeStyle = maskStroke
      ctx.lineWidth = 1.4; ctx.fill(); ctx.stroke()
    }
  }
  // 已保存的圈选
  const saved = store.labels?.glass_masks?.[props.cam]
  if (saved?.length) {
    ctx.beginPath()
    saved.forEach((p: number[], i: number) => { const x = p[0] * w, y = p[1] * h; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y) })
    ctx.closePath()
    ctx.fillStyle = savedFill; ctx.strokeStyle = savedStroke
    ctx.setLineDash([6, 4]); ctx.lineWidth = 1.4; ctx.fill(); ctx.stroke(); ctx.setLineDash([])
  }
  // 正在编辑
  if (pts.value.length) {
    ctx.beginPath()
    pts.value.forEach((p, i) => { const x = p[0] * w, y = p[1] * h; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y) })
    ctx.strokeStyle = accent; ctx.lineWidth = 1.8; ctx.stroke()
    for (const p of pts.value) {
      ctx.beginPath(); ctx.arc(p[0] * w, p[1] * h, 3.2, 0, 6.284)
      ctx.fillStyle = accent; ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.stroke()
    }
  }
}

function onClick(e: MouseEvent) {
  const wrap = imgWrap.value
  if (!wrap) return
  const r = wrap.getBoundingClientRect()
  const u = (e.clientX - r.left) / r.width
  const v = (e.clientY - r.top) / r.height
  if (u < 0 || u > 1 || v < 0 || v > 1) return
  pts.value.push([+u.toFixed(4), +v.toFixed(4)])
  redraw()
}

onMounted(() => {
  ro = new ResizeObserver(redraw)
  if (imgWrap.value) ro.observe(imgWrap.value)
  redraw()
})
onUnmounted(() => ro?.disconnect())
watch(() => [props.cam, frame.value, pts.value.length], redraw)
watch(() => store.labels.glass_masks, redraw, { deep: true })

function undo() { pts.value.pop(); redraw() }
function clear() { pts.value = []; redraw() }
function save() {
  if (pts.value.length < 3) return
  emit('save', props.cam, pts.value.slice())
  pts.value = []
  redraw()
}
</script>

<template>
  <div class="me">
    <div class="row" style="margin-bottom:var(--sp-2); gap:var(--sp-2)">
      <label class="tiny dim" for="mf">帧</label>
      <input id="mf" type="number" v-model.number="frame" class="frame-in" min="0" />
      <div class="grow" />
      <span class="tiny dim">点击画面依次落点，≥3 点后可保存</span>
    </div>
    <div class="stage" ref="imgWrap" @click="onClick">
      <img ref="imgEl" :src="imgUrl" :alt="`${cam} 第 ${frame} 帧（圈选）`" />
      <canvas ref="canvas"></canvas>
    </div>
    <div class="row wrap" style="margin-top:var(--sp-2); gap:var(--sp-2)">
      <span class="tag mono">{{ pts.length }} 点</span>
      <button @click="undo" :disabled="!pts.length">撤销</button>
      <button @click="clear" :disabled="!pts.length">清空</button>
      <button class="primary" @click="save" :disabled="pts.length < 3">保存区域</button>
      <div class="grow" />
      <span class="tiny dim">绿虚线＝已保存 ｜ 红＝项目既有草稿</span>
    </div>
  </div>
</template>

<style scoped>
.stage { position: relative; background: var(--stage); line-height: 0; cursor: crosshair; }
.stage img { width: 100%; display: block; }
.stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; }
.frame-in { width: 96px; min-height: 26px; padding: 2px 6px; font-size: var(--text-xs); }
</style>
