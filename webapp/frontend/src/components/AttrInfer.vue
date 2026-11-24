<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { api } from '../api'

/** 与后端 ATTR_FIELDS 同序；服务返回的是英文键，这里只做显示映射 */
const FIELDS: [string, string][] = [
  ['gender', '性别'], ['top_color', '上衣颜色'], ['top_type', '上衣类型'],
  ['bottom_color', '下装颜色'], ['bottom_type', '下装类型'],
  ['glasses', '戴眼镜'], ['hat', '戴帽子'], ['backpack', '背包'], ['carrying', '拎东西']
]

const st = ref<any>({ ok: false })
const file = ref<File | null>(null)
const preview = ref('')
const busy = ref(false)
const err = ref('')
const res = ref<any>(null)
const sel = ref(-1)
const drag = ref(false)
const whole = ref(false)   // 已裁好的单人图：boxes=full，跳过检测

onMounted(async () => {
  try {
    st.value = await api.attrInferStatus()
  } catch (e: any) {
    st.value = { ok: false, error: String(e?.message || e), hint: '后端是否在跑？' }
  }
})
onBeforeUnmount(() => {
  if (preview.value) URL.revokeObjectURL(preview.value)
})

function pick() {
  ;(document.getElementById('ai-file') as HTMLInputElement | null)?.click()
}
function onFile(f: File | null) {
  if (!f) {
    if (preview.value) URL.revokeObjectURL(preview.value)
    file.value = null
    preview.value = ''
    res.value = null
    sel.value = -1
    return
  }
  if (!/^image\//.test(f.type)) {
    err.value = `不是图片：${f.type || f.name}`
    return
  }
  if (f.size > 12 * 2 ** 20) {
    err.value = `图片 ${(f.size / 2 ** 20).toFixed(1)} MB，超过 12 MB 上限`
    return
  }
  err.value = ''
  res.value = null
  sel.value = -1
  if (preview.value) URL.revokeObjectURL(preview.value)
  file.value = f
  preview.value = URL.createObjectURL(f)
}
function onDrop(e: DragEvent) {
  drag.value = false
  onFile(e.dataTransfer?.files?.[0] || null)
}

async function run() {
  if (!file.value || busy.value) return
  busy.value = true
  err.value = ''
  const t0 = performance.now()
  try {
    const d = await api.attrInfer(file.value, { crops: 1, boxes: whole.value ? 'full' : '' })
    // 端到端含上传与代理；服务自己的 timing_ms 不含上传，两者要分开报
    d._e2e = Math.round(performance.now() - t0)
    res.value = d
    sel.value = d.persons?.length ? 0 : -1
  } catch (e: any) {
    err.value = String(e?.message || e)
    // 失败后重新探活：服务中途挂了的话徽标要立刻变红，不能一直显示「在线」
    api.attrInferStatus().then((x: any) => (st.value = x)).catch(() => {})
  } finally {
    busy.value = false
  }
}

const W = computed(() => res.value?.image?.w || 1)
const H = computed(() => res.value?.image?.h || 1)
function boxStyle(p: any) {
  const [x1, y1, x2, y2] = p.box
  return {
    left: `${(x1 / W.value) * 100}%`,
    top: `${(y1 / H.value) * 100}%`,
    width: `${((x2 - x1) / W.value) * 100}%`,
    height: `${((y2 - y1) / H.value) * 100}%`
  }
}
function capClass(a: any) {
  return a?.v == null ? 'na' : a.cap === 'ref' ? 'ref' : 'ok'
}
function confPct(c: any) {
  return c == null ? '' : `${Math.round(c * 100)}%`
}
</script>

<template>
  <div class="ai">
    <input id="ai-file" type="file" accept="image/*" hidden
           @change="onFile(($event.target as HTMLInputElement).files?.[0] || null)" />
    <div class="hd">
      <b>单图试跑</b>
      <span class="dim">上传一张图 → 每个人 + 9 个属性。推理跑在远程 hc-attr 容器里，本机只做转发。</span>
      <span class="grow" />
      <span class="pill" :class="st.ok ? 'on' : 'off'" :title="st.url || ''">
        <template v-if="st.ok">服务在线 · {{ st.health?.device || '?' }} · 加载 {{ st.health?.loaded_in_s ?? '?' }}s</template>
        <template v-else>服务离线</template>
      </span>
    </div>

    <div
      v-if="!res"
      class="dz" :class="{ drag, has: !!preview }"
      @click="pick"
      @dragover.prevent="drag = true"
      @dragleave="drag = false"
      @drop.prevent="onDrop"
    >
      <img v-if="preview" :src="preview" alt="预览" />
      <div v-else class="ph">
        <b>点击选择图片</b>，或拖进来
        <span class="dim">JPG / PNG，≤ 12 MB。合影、监控帧、手机照都能试；域外精度会下降，见页底。</span>
      </div>
    </div>

    <div class="ops">
      <button class="btn primary" :disabled="!file || busy" @click.stop="run">
        {{ busy ? '识别中…' : '开始识别' }}
      </button>
      <button v-if="res" class="btn" :disabled="busy" @click.stop="pick">换一张</button>
      <button v-if="file" class="btn" :disabled="busy" @click.stop="onFile(null)">清空</button>
      <label class="ck" title="已经裁成单人的图（例如别人给你的人像裁图）：跳过检测，整张图当作一个人">
        <input v-model="whole" type="checkbox" :disabled="busy" /> 整图当一个人
      </label>
      <span v-if="file" class="dim">{{ file.name }} · {{ (file.size / 1024).toFixed(0) }} KB</span>
      <span class="grow" />
      <span v-if="res" class="dim">
        端到端 {{ res._e2e }} ms（代理 {{ res._proxy?.ms }} ms）｜服务内
        检测 {{ res.timing_ms?.det }} · 嵌入 {{ res.timing_ms?.emb }} · 头 {{ res.timing_ms?.head }} ms
      </span>
    </div>

    <p v-if="err" class="err">{{ err }}<template v-if="!st.ok && st.hint">　{{ st.hint }}</template></p>

    <div v-if="res" class="body">
      <div class="pic">
        <img :src="preview" alt="原图" />
        <div v-for="(p, i) in res.persons" :key="i" class="bx"
             :class="{ low: p.is_person < 0.5, sel: i === sel }" :style="boxStyle(p)"
             @click.stop="sel = i">
          <span>{{ i }}</span>
        </div>
      </div>

      <div class="list">
        <div v-for="(p, i) in res.persons" :key="i" class="pcard"
             :class="{ sel: i === sel, low: p.is_person < 0.5 }" @click="sel = i">
          <div class="prow">
            <img v-if="p.crop_b64" :src="'data:image/jpeg;base64,' + p.crop_b64" alt="" />
            <div class="pmeta">
              <b>#{{ i }}</b>
              <span class="tag">{{ p.band }} 档</span>
              <span class="dim">框高 {{ Math.round(p.h) }}px</span>
              <span v-if="p.is_person < 0.5" class="warn">疑似非人</span>
            </div>
            <span class="grow" />
            <span class="ip" :class="{ bad: p.is_person < 0.5 }"
                  title="像人的置信度；低于 0.5 多为倒影/海报/货架">人 {{ p.is_person.toFixed(2) }}</span>
          </div>
          <div class="chips">
            <span v-for="f in FIELDS" :key="f[0]" class="chip" :class="capClass(p.attrs[f[0]])">
              <i>{{ f[1] }}</i>
              <template v-if="p.attrs[f[0]]?.v != null">
                <b>{{ p.attrs[f[0]].v }}</b><em>{{ confPct(p.attrs[f[0]].conf) }}</em>
              </template>
              <template v-else><b class="na">不可判</b></template>
            </span>
          </div>
        </div>
        <p v-if="!res.persons.length" class="dim">
          这张图没有检出人。若画面里确实有人，多半是框太小或被遮挡——检测阈值不在这页的调节范围。
        </p>
      </div>
    </div>

    <p class="foot">
      口径：<b>标签在前的是「可承诺/参考」，灰色「不可判」= 该框高档位没有可靠证据</b>，不是模型漏了。
      置信度不是准确率。「人 0.xx」是域内素材训练的判据，<b>域外裁图会偏低</b>
      （实测已裁好的单人图只有 0.24），只作提示、不参与过滤。域外（PA-100K 800 张）实测：性别 .894、上衣型 .899、眼镜 .838、
      帽 TPR .52、背包 TPR .25、拎东西 .19 —— <b>换个场地拍的照片，精度会明显低于本素材</b>。
      正式用法见「属性索引」页的能力表。
    </p>
  </div>
</template>

<style scoped>
.ai { display: flex; flex-direction: column; gap: 10px; padding: 12px 14px 24px; height: 100%; overflow: auto; }
.hd { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.grow { flex: 1; }
.dim { color: var(--fg-3); font-size: 12px; }
.pill {
  font-size: 12px; padding: 2px 8px; border-radius: 999px;
  border: 1px solid var(--line); white-space: nowrap;
}
.pill.on { color: var(--ok); border-color: color-mix(in oklch, var(--ok) 45%, transparent); }
.pill.off { color: var(--danger); border-color: color-mix(in oklch, var(--danger) 45%, transparent); }

.dz {
  border: 1px dashed var(--line); border-radius: 10px; background: var(--surface-1);
  min-height: 118px; display: flex; align-items: center; justify-content: center;
  cursor: pointer; overflow: hidden; transition: border-color .15s, background .15s;
}
.dz:hover, .dz.drag { border-color: var(--accent); background: var(--surface-2); }
.dz.has { min-height: 0; }
.dz > img { max-height: 260px; max-width: 100%; display: block; }
.ph { display: flex; flex-direction: column; align-items: center; gap: 4px; padding: 18px; text-align: center; }

.ops { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; font-size: 12px; }
.btn {
  font: inherit; font-size: 12px; padding: 4px 10px; border-radius: 7px; cursor: pointer;
  background: var(--surface-2); color: var(--fg-2); border: 1px solid var(--line);
}
.btn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
.btn.primary { background: var(--accent); color: #fff; border-color: transparent; }
.btn.primary:hover:not(:disabled) { filter: brightness(1.08); color: #fff; }
.btn:disabled { opacity: .5; cursor: default; }
.ck { display: inline-flex; align-items: center; gap: 4px; color: var(--fg-2); cursor: pointer; }
.err { color: var(--danger); font-size: 12px; margin: 0; white-space: pre-wrap; }

.body { display: flex; gap: 12px; align-items: flex-start; }
.pic { position: relative; flex: 0 1 60%; max-width: 60%; background: #000; border-radius: 8px; overflow: hidden; }
.pic > img { display: block; width: 100%; height: auto; }
.bx { position: absolute; border: 2px solid var(--accent); border-radius: 2px; cursor: pointer; }
.bx.low { border-color: var(--danger); border-style: dashed; }
.bx.sel { border-width: 3px; box-shadow: 0 0 0 1px rgba(255, 255, 255, .55) inset; }
.bx > span {
  position: absolute; top: 0; left: 0; transform: translateY(-100%);
  background: var(--accent); color: #fff; font-size: 11px; line-height: 14px; padding: 0 4px;
}
.bx.low > span { background: var(--danger); }

.list { flex: 1 1 40%; min-width: 0; display: flex; flex-direction: column; gap: 8px; max-height: 66vh; overflow: auto; padding-right: 2px; }
.pcard { border: 1px solid var(--line); border-radius: 8px; background: var(--surface-1); padding: 7px 9px; cursor: pointer; }
.pcard.sel { border-color: var(--accent); }
.pcard.low { border-style: dashed; }
.prow { display: flex; align-items: center; gap: 8px; }
.prow > img { width: 44px; max-height: 76px; object-fit: contain; border-radius: 4px; background: var(--surface-2); flex: none; }
.pmeta { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; font-size: 12px; }
.tag { font-size: 11px; padding: 0 5px; border-radius: 4px; background: var(--surface-2); color: var(--fg-2); }
.warn { font-size: 11px; color: var(--danger); }
.ip { font-size: 11px; color: var(--fg-3); white-space: nowrap; }
.ip.bad { color: var(--danger); }

.chips { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 6px; }
.chip {
  display: inline-flex; align-items: baseline; gap: 4px; font-size: 11px;
  border: 1px solid var(--line); border-radius: 6px; padding: 1px 6px; background: var(--surface-2);
}
.chip i { font-style: normal; color: var(--fg-3); }
.chip b { font-weight: 600; }
.chip em { font-style: normal; color: var(--fg-3); font-size: 10px; }
.chip.ref { border-color: color-mix(in oklch, var(--warn) 55%, var(--line)); }
.chip.ref em { color: var(--warn); }
.chip.na { opacity: .55; }
.chip b.na { font-weight: 400; color: var(--fg-3); }
.foot { font-size: 12px; color: var(--fg-3); line-height: 1.6; margin: 4px 0 0; max-width: 1100px; }

@media (max-width: 860px) {
  .body { flex-direction: column; }
  .pic { flex: 1 1 auto; max-width: 100%; width: 100%; }
  .list { flex: 1 1 auto; width: 100%; max-height: none; }
}
</style>
